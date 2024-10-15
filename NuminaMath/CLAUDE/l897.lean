import Mathlib

namespace NUMINAMATH_CALUDE_walking_speed_difference_l897_89770

theorem walking_speed_difference (child_distance child_time elderly_distance elderly_time : ℝ) :
  child_distance = 15 ∧ 
  child_time = 3.5 ∧ 
  elderly_distance = 10 ∧ 
  elderly_time = 4 → 
  (elderly_time * 60 / elderly_distance) - (child_time * 60 / child_distance) = 10 :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_difference_l897_89770


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l897_89764

/-- Given five collinear points A, B, C, D, E in that order, with specific distances between them,
    prove that the minimum sum of squared distances from these points to any point P on AD is 237. -/
theorem min_sum_squared_distances (A B C D E P : ℝ) : 
  (A < B) → (B < C) → (C < D) → (D < E) →  -- Points are collinear and in order
  (B - A = 1) → (C - B = 1) → (D - C = 3) → (E - D = 12) →  -- Given distances
  (A ≤ P) → (P ≤ D) →  -- P is on segment AD
  ∃ (m : ℝ), ∀ (Q : ℝ), (A ≤ Q) → (Q ≤ D) → 
    (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2 ≥ m ∧ 
    m = 237 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l897_89764


namespace NUMINAMATH_CALUDE_perfect_square_sum_of_powers_l897_89731

theorem perfect_square_sum_of_powers (x y z : ℕ+) :
  ∃ (k : ℕ), (4:ℕ)^(x:ℕ) + (4:ℕ)^(y:ℕ) + (4:ℕ)^(z:ℕ) = k^2 ↔
  ∃ (b z' : ℕ+), x = 2*b - 1 + z' ∧ y = b + z' ∧ z = z' :=
sorry

end NUMINAMATH_CALUDE_perfect_square_sum_of_powers_l897_89731


namespace NUMINAMATH_CALUDE_cereal_box_ratio_l897_89763

/-- Theorem: Cereal Box Ratio
Given 3 boxes of cereal where:
- The first box contains 14 ounces
- The total amount in all boxes is 33 ounces
- The second box contains 5 ounces less than the third box
Then the ratio of cereal in the second box to the first box is 1:2
-/
theorem cereal_box_ratio (box1 box2 box3 : ℝ) : 
  box1 = 14 →
  box1 + box2 + box3 = 33 →
  box2 = box3 - 5 →
  box2 / box1 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cereal_box_ratio_l897_89763


namespace NUMINAMATH_CALUDE_problem_statement_l897_89711

theorem problem_statement (a b : ℝ) : 
  |a + 2| + (b - 1)^2 = 0 → (a + b)^2005 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l897_89711


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l897_89751

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x, ax^2 + b*x - 1 < 0 ↔ -1/2 < x ∧ x < 1) →
  (∀ x, (a*x + 2) / (b*x + 1) < 0 ↔ x < -1 ∨ x > 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l897_89751


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l897_89712

theorem parabola_focus_distance (p : ℝ) (h1 : p > 0) : 
  let focus : ℝ × ℝ := (p / 2, 0)
  let distance_to_line (point : ℝ × ℝ) : ℝ := 
    |-(point.1) + point.2 - 1| / Real.sqrt 2
  distance_to_line focus = Real.sqrt 2 → p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l897_89712


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_average_l897_89755

theorem consecutive_odd_integers_average (n : ℕ) (first : ℤ) :
  n = 10 →
  first = 145 →
  first % 2 = 1 →
  let sequence := List.range n |>.map (λ i => first + 2 * i)
  (sequence.sum / n : ℚ) = 154 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_average_l897_89755


namespace NUMINAMATH_CALUDE_library_books_pages_l897_89718

theorem library_books_pages (num_books : ℕ) (total_pages : ℕ) (h1 : num_books = 8) (h2 : total_pages = 3824) :
  total_pages / num_books = 478 := by
sorry

end NUMINAMATH_CALUDE_library_books_pages_l897_89718


namespace NUMINAMATH_CALUDE_prism_coloring_iff_divisible_by_three_l897_89739

/-- Represents a prism with an n-gon base -/
structure Prism :=
  (n : ℕ)

/-- Represents a coloring of the prism vertices -/
def Coloring (p : Prism) := Fin (2 * p.n) → Fin 3

/-- Predicate to check if a coloring is valid -/
def is_valid_coloring (p : Prism) (c : Coloring p) : Prop :=
  ∀ v : Fin (2 * p.n), 
    ∃ (c1 c2 : Fin 3), c1 ≠ c v ∧ c2 ≠ c v ∧ c1 ≠ c2 ∧
    ∃ (v1 v2 : Fin (2 * p.n)), v1 ≠ v ∧ v2 ≠ v ∧ v1 ≠ v2 ∧
    c v1 = c1 ∧ c v2 = c2

theorem prism_coloring_iff_divisible_by_three (p : Prism) :
  (∃ c : Coloring p, is_valid_coloring p c) ↔ p.n % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_prism_coloring_iff_divisible_by_three_l897_89739


namespace NUMINAMATH_CALUDE_dogs_not_liking_either_l897_89795

theorem dogs_not_liking_either (total : ℕ) (watermelon : ℕ) (salmon : ℕ) (both : ℕ)
  (h1 : total = 75)
  (h2 : watermelon = 12)
  (h3 : salmon = 55)
  (h4 : both = 7) :
  total - (watermelon + salmon - both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_liking_either_l897_89795


namespace NUMINAMATH_CALUDE_matrix_rank_two_l897_89768

/-- Given an n×n matrix A where A_ij = i + j, prove that the rank of A is 2 -/
theorem matrix_rank_two (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)
  (h : ∀ (i j : Fin n), A i j = (i.val + 1 : ℝ) + (j.val + 1 : ℝ)) :
  Matrix.rank A = 2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_rank_two_l897_89768


namespace NUMINAMATH_CALUDE_smallest_other_integer_l897_89761

theorem smallest_other_integer (a b x : ℕ+) : 
  (a = 70 ∨ b = 70) →
  (Nat.gcd a b = x + 7) →
  (Nat.lcm a b = x * (x + 7)) →
  (min a b ≠ 70 → min a b ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l897_89761


namespace NUMINAMATH_CALUDE_least_five_digit_square_and_cube_l897_89798

theorem least_five_digit_square_and_cube : 
  (∀ n : ℕ, n < 15625 → ¬(∃ a b : ℕ, n = a^2 ∧ n = b^3 ∧ n ≥ 10000)) ∧ 
  (∃ a b : ℕ, 15625 = a^2 ∧ 15625 = b^3) := by
  sorry

end NUMINAMATH_CALUDE_least_five_digit_square_and_cube_l897_89798


namespace NUMINAMATH_CALUDE_equation_system_implies_third_equation_l897_89721

theorem equation_system_implies_third_equation (a b : ℝ) :
  a^2 - 3*a*b + 2*b^2 + a - b = 0 →
  a^2 - 2*a*b + b^2 - 5*a + 7*b = 0 →
  a*b - 12*a + 15*b = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_system_implies_third_equation_l897_89721


namespace NUMINAMATH_CALUDE_tim_laundry_cycle_l897_89724

/-- Ronald's laundry cycle in days -/
def ronald_cycle : ℕ := 6

/-- Number of days until they both do laundry on the same day again -/
def next_common_day : ℕ := 18

/-- Tim's laundry cycle in days -/
def tim_cycle : ℕ := 3

theorem tim_laundry_cycle :
  (ronald_cycle ∣ next_common_day) ∧
  (tim_cycle ∣ next_common_day) ∧
  (tim_cycle < ronald_cycle) ∧
  (∀ x : ℕ, x < tim_cycle → ¬(x ∣ next_common_day ∧ x ∣ ronald_cycle)) :=
sorry

end NUMINAMATH_CALUDE_tim_laundry_cycle_l897_89724


namespace NUMINAMATH_CALUDE_pyramid_base_edge_length_l897_89753

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- Height of the pyramid -/
  pyramid_height : ℝ
  /-- Radius of the hemisphere -/
  hemisphere_radius : ℝ
  /-- The hemisphere is tangent to the four faces of the pyramid -/
  tangent_to_faces : Bool

/-- Theorem: Edge length of the base of the pyramid -/
theorem pyramid_base_edge_length (p : PyramidWithHemisphere) 
  (h1 : p.pyramid_height = 4)
  (h2 : p.hemisphere_radius = 3)
  (h3 : p.tangent_to_faces = true) :
  ∃ (edge_length : ℝ), edge_length = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_edge_length_l897_89753


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l897_89793

theorem sum_of_squares_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 2 ∧ b = -6 ∧ c = 4 → x₁^2 + x₂^2 = 5 :=
by sorry


end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l897_89793


namespace NUMINAMATH_CALUDE_smallest_value_l897_89775

theorem smallest_value (a b : ℝ) (h : b < 0) : (a + b < a) ∧ (a + b < a - b) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_l897_89775


namespace NUMINAMATH_CALUDE_inequality_solution_l897_89705

theorem inequality_solution (x : ℝ) : (x + 4) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l897_89705


namespace NUMINAMATH_CALUDE_quadratic_b_value_l897_89762

/-- The value of b in a quadratic function y = x² - bx + c passing through (1,n) and (3,n) -/
theorem quadratic_b_value (n : ℝ) : 
  ∃ (b c : ℝ), (∀ x : ℝ, x^2 - b*x + c = n ↔ x = 1 ∨ x = 3) → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_b_value_l897_89762


namespace NUMINAMATH_CALUDE_candy_distribution_l897_89743

theorem candy_distribution (total red blue : ℕ) (h1 : total = 25689) (h2 : red = 1342) (h3 : blue = 8965) :
  let remaining := total - (red + blue)
  ∃ (green : ℕ), green * 3 = remaining ∧ green = 5127 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l897_89743


namespace NUMINAMATH_CALUDE_pumpkin_ravioli_weight_l897_89733

theorem pumpkin_ravioli_weight (brother_ravioli_count : ℕ) (total_weight : ℝ) : 
  brother_ravioli_count = 12 → total_weight = 15 → 
  (total_weight / brother_ravioli_count : ℝ) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_ravioli_weight_l897_89733


namespace NUMINAMATH_CALUDE_segment_count_is_21_l897_89756

/-- A configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  triple_intersection : Bool
  triple_intersection_count : ℕ

/-- Calculate the number of non-overlapping line segments in a given configuration -/
def count_segments (config : LineConfiguration) : ℕ :=
  config.num_lines * 4 - config.triple_intersection_count

/-- The specific configuration given in the problem -/
def problem_config : LineConfiguration :=
  { num_lines := 6
  , triple_intersection := true
  , triple_intersection_count := 3 }

/-- Theorem stating that the number of non-overlapping line segments in the given configuration is 21 -/
theorem segment_count_is_21 : count_segments problem_config = 21 := by
  sorry

end NUMINAMATH_CALUDE_segment_count_is_21_l897_89756


namespace NUMINAMATH_CALUDE_systematic_sampling_l897_89740

theorem systematic_sampling 
  (population_size : ℕ) 
  (num_groups : ℕ) 
  (sample_size : ℕ) 
  (first_draw : ℕ) :
  population_size = 60 →
  num_groups = 6 →
  sample_size = 6 →
  first_draw = 3 →
  let interval := population_size / num_groups
  let fifth_group_draw := first_draw + interval * 4
  fifth_group_draw = 43 := by
sorry


end NUMINAMATH_CALUDE_systematic_sampling_l897_89740


namespace NUMINAMATH_CALUDE_star_example_l897_89785

-- Define the ★ operation
def star (m n p q : ℚ) : ℚ := (m + 1) * (p + 1) * ((q + 1) / (n + 1))

-- Theorem statement
theorem star_example : star (5/11) (11/1) (7/2) (2/1) = 12 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l897_89785


namespace NUMINAMATH_CALUDE_triple_sum_squares_and_fourth_powers_l897_89701

theorem triple_sum_squares_and_fourth_powers (t : ℤ) : 
  (4*t)^2 + (3 - 2*t - t^2)^2 + (3 + 2*t - t^2)^2 = 2*(3 + t^2)^2 ∧
  (4*t)^4 + (3 - 2*t - t^2)^4 + (3 + 2*t - t^2)^4 = 2*(3 + t^2)^4 := by
  sorry

end NUMINAMATH_CALUDE_triple_sum_squares_and_fourth_powers_l897_89701


namespace NUMINAMATH_CALUDE_solution_to_equation_sum_of_fourth_powers_l897_89747

-- Define the equation for part 1
def equation (x : ℝ) : Prop := x^4 - x^2 - 6 = 0

-- Theorem for part 1
theorem solution_to_equation :
  ∀ x : ℝ, equation x ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 :=
sorry

-- Define the conditions for part 2
def condition (a b : ℝ) : Prop :=
  a^4 - 3*a^2 + 1 = 0 ∧ b^4 - 3*b^2 + 1 = 0 ∧ a ≠ b

-- Theorem for part 2
theorem sum_of_fourth_powers (a b : ℝ) :
  condition a b → a^4 + b^4 = 7 :=
sorry

end NUMINAMATH_CALUDE_solution_to_equation_sum_of_fourth_powers_l897_89747


namespace NUMINAMATH_CALUDE_carts_needed_is_15_l897_89772

-- Define the total volume of goods
def total_volume : ℚ := 1

-- Define the daily capacity of each vehicle type
def large_truck_capacity : ℚ := total_volume / (3 * 4)
def small_truck_capacity : ℚ := total_volume / (4 * 5)
def cart_capacity : ℚ := total_volume / (20 * 6)

-- Define the work done in the first 2 days
def work_done_2_days : ℚ := 2 * (2 * large_truck_capacity + 3 * small_truck_capacity + 7 * cart_capacity)

-- Define the remaining work
def remaining_work : ℚ := total_volume - work_done_2_days

-- Define the number of carts needed for the last 2 days
def carts_needed : ℕ := (remaining_work / (2 * cart_capacity)).ceil.toNat

-- Theorem statement
theorem carts_needed_is_15 : carts_needed = 15 := by
  sorry

end NUMINAMATH_CALUDE_carts_needed_is_15_l897_89772


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l897_89730

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y →
    (∃ k : ℝ, (3 * x^2 + 1) * k = 1 ∧ 4 * k = 1) ↔ 
    ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

#check tangent_parallel_points

end NUMINAMATH_CALUDE_tangent_parallel_points_l897_89730


namespace NUMINAMATH_CALUDE_square_difference_40_39_l897_89774

theorem square_difference_40_39 : (40 : ℕ)^2 - (39 : ℕ)^2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_40_39_l897_89774


namespace NUMINAMATH_CALUDE_area_after_shortening_other_side_l897_89714

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle -/
def original : Rectangle := { length := 5, width := 7 }

/-- The rectangle after shortening one side by 2 -/
def shortened : Rectangle := { length := 3, width := 7 }

/-- The rectangle after shortening the other side by 2 -/
def other_shortened : Rectangle := { length := 5, width := 5 }

theorem area_after_shortening_other_side :
  shortened.area = 21 → other_shortened.area = 25 := by sorry

end NUMINAMATH_CALUDE_area_after_shortening_other_side_l897_89714


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l897_89720

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) :
  area = (1 / 2) * leg1 * leg2 →
  leg1 = 30 →
  area = 150 →
  leg2 * leg2 + leg1 * leg1 = hypotenuse * hypotenuse →
  leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l897_89720


namespace NUMINAMATH_CALUDE_horners_rule_for_specific_polynomial_v3_value_at_3_l897_89732

def horner_step (a : ℕ) (x v : ℕ) : ℕ := v * x + a

def horners_rule (coeffs : List ℕ) (x : ℕ) : ℕ :=
  coeffs.foldl (horner_step x) 0

theorem horners_rule_for_specific_polynomial (x : ℕ) :
  horners_rule [1, 1, 3, 2, 0, 1] x = x^5 + 2*x^3 + 3*x^2 + x + 1 := by sorry

theorem v3_value_at_3 :
  let coeffs := [1, 1, 3, 2, 0, 1]
  let x := 3
  let v0 := 1
  let v1 := horner_step 0 x v0
  let v2 := horner_step 2 x v1
  let v3 := horner_step 3 x v2
  v3 = 36 := by sorry

end NUMINAMATH_CALUDE_horners_rule_for_specific_polynomial_v3_value_at_3_l897_89732


namespace NUMINAMATH_CALUDE_derivative_value_l897_89766

theorem derivative_value (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 2 + x^3) :
  f' 2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_derivative_value_l897_89766


namespace NUMINAMATH_CALUDE_reciprocal_inequality_for_negative_numbers_l897_89715

theorem reciprocal_inequality_for_negative_numbers (a b : ℝ) 
  (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_for_negative_numbers_l897_89715


namespace NUMINAMATH_CALUDE_one_greater_one_less_than_one_l897_89765

theorem one_greater_one_less_than_one (a b : ℝ) (h : ((1 + a * b) / (a + b))^2 < 1) :
  (a > 1 ∧ -1 < b ∧ b < 1) ∨ (-1 < a ∧ a < 1 ∧ b > 1) :=
sorry

end NUMINAMATH_CALUDE_one_greater_one_less_than_one_l897_89765


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l897_89771

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_root_existence (a b c : ℝ) (ha : a ≠ 0) :
  quadratic_function a b c (-3) = -11 →
  quadratic_function a b c (-2) = -5 →
  quadratic_function a b c (-1) = -1 →
  quadratic_function a b c 0 = 1 →
  quadratic_function a b c 1 = 1 →
  ∃ x₁ : ℝ, quadratic_function a b c x₁ = 0 ∧ -1 < x₁ ∧ x₁ < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l897_89771


namespace NUMINAMATH_CALUDE_factorization_equality_l897_89778

theorem factorization_equality (a : ℝ) : (a + 3) * (a - 7) + 25 = (a - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l897_89778


namespace NUMINAMATH_CALUDE_hyperbola_equation_l897_89757

/-- Given a hyperbola with the standard form equation, prove that under certain conditions, 
    it has a specific equation. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (c : ℝ), c - a = 1 ∧ b = Real.sqrt 3) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l897_89757


namespace NUMINAMATH_CALUDE_second_car_speed_l897_89700

/-- Given two cars traveling in opposite directions for 2.5 hours,
    with one car traveling at 60 mph and the total distance between them
    being 310 miles after 2.5 hours, prove that the speed of the second car is 64 mph. -/
theorem second_car_speed (car1_speed : ℝ) (car2_speed : ℝ) (time : ℝ) (total_distance : ℝ) :
  car1_speed = 60 →
  time = 2.5 →
  total_distance = 310 →
  car1_speed * time + car2_speed * time = total_distance →
  car2_speed = 64 := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l897_89700


namespace NUMINAMATH_CALUDE_parallel_transitive_perpendicular_from_line_l897_89744

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Axioms for parallel and perpendicular relations
axiom parallel_symm {a b : Plane} : parallel a b → parallel b a
axiom perpendicular_symm {a b : Plane} : perpendicular a b → perpendicular b a

-- Theorem 1
theorem parallel_transitive {α β γ : Plane} :
  parallel α β → parallel α γ → parallel β γ := by sorry

-- Theorem 2
theorem perpendicular_from_line {m : Line} {α β : Plane} :
  line_perpendicular m α → line_parallel m β → perpendicular α β := by sorry

end NUMINAMATH_CALUDE_parallel_transitive_perpendicular_from_line_l897_89744


namespace NUMINAMATH_CALUDE_second_workshop_production_l897_89735

/-- Given three workshops producing boots with samples forming an arithmetic sequence,
    prove that the second workshop's production is 1200 pairs. -/
theorem second_workshop_production
  (total_production : ℕ)
  (a b c : ℕ)
  (h1 : total_production = 3600)
  (h2 : a + c = 2 * b)  -- arithmetic sequence property
  (h3 : a + b + c > 0)  -- ensure division is valid
  : (b : ℚ) / (a + b + c : ℚ) * total_production = 1200 :=
by sorry

end NUMINAMATH_CALUDE_second_workshop_production_l897_89735


namespace NUMINAMATH_CALUDE_woodworker_legs_count_l897_89786

/-- The number of furniture legs made by a woodworker -/
def total_furniture_legs (chairs tables : ℕ) : ℕ :=
  4 * chairs + 4 * tables

/-- Theorem: A woodworker who has built 6 chairs and 4 tables has made 40 furniture legs in total -/
theorem woodworker_legs_count : total_furniture_legs 6 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_legs_count_l897_89786


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l897_89789

theorem smallest_positive_integer_congruence :
  ∃ y : ℕ+, 
    (∀ z : ℕ+, (42 * z.val + 8) % 24 = 4 → y ≤ z) ∧
    (42 * y.val + 8) % 24 = 4 ∧
    y.val = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l897_89789


namespace NUMINAMATH_CALUDE_marble_distribution_l897_89780

theorem marble_distribution (sets : Nat) (marbles_per_set : Nat) (marbles_per_student : Nat) :
  sets = 3 →
  marbles_per_set = 32 →
  marbles_per_student = 4 →
  (sets * marbles_per_set) % marbles_per_student = 0 →
  (sets * marbles_per_set) / marbles_per_student = 24 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l897_89780


namespace NUMINAMATH_CALUDE_triangle_properties_l897_89728

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a ≠ t.b ∧
  2 * Real.sin (t.A - t.B) = t.a * Real.sin t.A - t.b * Real.sin t.B ∧
  (1/2) * t.a * t.b * Real.sin t.C = 1 ∧
  Real.tan t.C = 2

-- State the theorem
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.c = 2 ∧ t.a + t.b = 1 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l897_89728


namespace NUMINAMATH_CALUDE_c_profit_is_400_l897_89758

/-- Represents the investment and profit distribution for three individuals --/
structure BusinessInvestment where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ

/-- Calculates C's share of the profit based on the given investments and total profit --/
def c_profit_share (investment : BusinessInvestment) : ℕ :=
  (investment.c_investment * investment.total_profit) / (investment.a_investment + investment.b_investment + investment.c_investment)

/-- Theorem stating that C's share of the profit is 400 given the specific investments and total profit --/
theorem c_profit_is_400 (investment : BusinessInvestment)
  (h1 : investment.a_investment = 800)
  (h2 : investment.b_investment = 1000)
  (h3 : investment.c_investment = 1200)
  (h4 : investment.total_profit = 1000) :
  c_profit_share investment = 400 := by
  sorry

#eval c_profit_share ⟨800, 1000, 1200, 1000⟩

end NUMINAMATH_CALUDE_c_profit_is_400_l897_89758


namespace NUMINAMATH_CALUDE_total_score_three_probability_l897_89788

def yellow_balls : ℕ := 2
def white_balls : ℕ := 3
def total_balls : ℕ := yellow_balls + white_balls

def yellow_score : ℕ := 1
def white_score : ℕ := 2

def prob_yellow (balls_left : ℕ) : ℚ := yellow_balls / balls_left
def prob_white (balls_left : ℕ) : ℚ := white_balls / balls_left

theorem total_score_three_probability :
  (prob_yellow total_balls * prob_white (total_balls - 1) +
   prob_white total_balls * prob_yellow (total_balls - 1)) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_total_score_three_probability_l897_89788


namespace NUMINAMATH_CALUDE_shirt_production_l897_89707

theorem shirt_production (machines1 machines2 : ℕ) 
  (production1 production2 : ℕ) (time1 time2 : ℕ) : 
  machines1 = 12 → 
  machines2 = 20 → 
  production1 = 24 → 
  production2 = 45 → 
  time1 = 18 → 
  time2 = 22 → 
  production1 * time1 + production2 * time2 = 1422 := by
sorry

end NUMINAMATH_CALUDE_shirt_production_l897_89707


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l897_89716

theorem quadratic_root_difference (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ > 2 ∧ r₂ ≤ 2 ∧ 
   x^2 - 5*x + 6 = (x - r₁) * (x - r₂)) → 
  ∃ r₁ r₂ : ℝ, r₁ - r₂ = 1 ∧ 
   x^2 - 5*x + 6 = (x - r₁) * (x - r₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l897_89716


namespace NUMINAMATH_CALUDE_cube_edge_length_l897_89796

theorem cube_edge_length (surface_area : ℝ) (edge_length : ℝ) :
  surface_area = 96 ∧ surface_area = 6 * edge_length^2 → edge_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l897_89796


namespace NUMINAMATH_CALUDE_log_sum_property_l897_89767

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_sum_property (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  ∃ (f : ℝ → ℝ) (f_inv : ℝ → ℝ),
    (∀ x > 0, f x = Real.log x / Real.log a) ∧
    (∀ x, f (f_inv x) = x) ∧
    (f_inv 2 = 9) →
    f 9 + f 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_property_l897_89767


namespace NUMINAMATH_CALUDE_phi_value_l897_89704

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is decreasing on an interval [a, b] if for all x, y in [a, b],
    x < y implies f(x) > f(y) -/
def IsDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x > f y

theorem phi_value (f : ℝ → ℝ) (φ : ℝ) 
    (h1 : f = λ x => 2 * Real.sin (2 * x + φ + π / 3))
    (h2 : IsOdd f)
    (h3 : IsDecreasingOn f 0 (π / 4)) :
    φ = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_l897_89704


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l897_89738

def f (x : ℝ) := x^3 - x + 3

theorem tangent_line_at_one : 
  ∃ (a b : ℝ), ∀ (x y : ℝ), 
    (y = f x ∧ x = 1) → (y = a * x + b) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l897_89738


namespace NUMINAMATH_CALUDE_negation_equivalence_l897_89737

theorem negation_equivalence :
  (¬ (∀ x : ℝ, ∃ n : ℕ, n ≥ x)) ↔ (∃ x : ℝ, ∀ n : ℕ, (n : ℝ) < x) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l897_89737


namespace NUMINAMATH_CALUDE_inequality_relationship_l897_89736

theorem inequality_relationship (a b : ℝ) : 
  (∀ a b, a > b → a + 1 > b - 2) ∧ 
  (∃ a b, a + 1 > b - 2 ∧ ¬(a > b)) := by
sorry

end NUMINAMATH_CALUDE_inequality_relationship_l897_89736


namespace NUMINAMATH_CALUDE_cafeteria_bill_l897_89782

/-- The total amount spent by Mell and her friends at the cafeteria -/
theorem cafeteria_bill (coffee_price ice_cream_price cake_price : ℕ) 
  (h1 : coffee_price = 4)
  (h2 : ice_cream_price = 3)
  (h3 : cake_price = 7)
  (mell_coffee mell_cake : ℕ)
  (h4 : mell_coffee = 2)
  (h5 : mell_cake = 1)
  (friend_count : ℕ)
  (h6 : friend_count = 2) :
  (mell_coffee * (friend_count + 1) * coffee_price) + 
  (mell_cake * (friend_count + 1) * cake_price) + 
  (friend_count * ice_cream_price) = 51 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_bill_l897_89782


namespace NUMINAMATH_CALUDE_opposite_of_neg_six_l897_89787

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- The opposite of -6 is 6. -/
theorem opposite_of_neg_six : opposite (-6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_six_l897_89787


namespace NUMINAMATH_CALUDE_square_of_119_l897_89791

theorem square_of_119 : 119^2 = 14161 := by
  sorry

end NUMINAMATH_CALUDE_square_of_119_l897_89791


namespace NUMINAMATH_CALUDE_compute_expression_l897_89741

theorem compute_expression : 18 * (250 / 3 + 36 / 9 + 16 / 32 + 2) = 1617 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l897_89741


namespace NUMINAMATH_CALUDE_intersection_M_N_l897_89799

def M : Set ℝ := { x | -3 < x ∧ x ≤ 5 }
def N : Set ℝ := { x | -5 < x ∧ x < 5 }

theorem intersection_M_N : M ∩ N = { x | -3 < x ∧ x < 5 } := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l897_89799


namespace NUMINAMATH_CALUDE_only_yellow_river_certain_l897_89781

-- Define the type for events
inductive Event
  | MoonlightInFrontOfBed
  | LonelySmokeInDesert
  | ReachForStarsWithHand
  | YellowRiverFlowsIntoSea

-- Define a function to check if an event is certain
def isCertain (e : Event) : Prop :=
  match e with
  | Event.YellowRiverFlowsIntoSea => True
  | _ => False

-- Theorem stating that only the Yellow River flowing into the sea is certain
theorem only_yellow_river_certain :
  ∀ (e : Event), isCertain e ↔ e = Event.YellowRiverFlowsIntoSea :=
by
  sorry

#check only_yellow_river_certain

end NUMINAMATH_CALUDE_only_yellow_river_certain_l897_89781


namespace NUMINAMATH_CALUDE_school_comparison_l897_89717

theorem school_comparison (students_A : ℝ) (qualified_A : ℝ) (students_B : ℝ) (qualified_B : ℝ)
  (h1 : qualified_A = 0.7 * students_A)
  (h2 : qualified_B = 1.5 * qualified_A)
  (h3 : qualified_B = 0.875 * students_B) :
  (students_B - students_A) / students_A = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_school_comparison_l897_89717


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_product_l897_89719

theorem unique_solution_quadratic_product (k : ℝ) : 
  (∃! x : ℝ, k * x^2 + (k + 5) * x + 5 = 0) → k = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_product_l897_89719


namespace NUMINAMATH_CALUDE_average_speed_is_25_l897_89748

-- Define the given conditions
def workdays : ℕ := 5
def work_distance : ℝ := 20
def weekend_ride : ℝ := 200
def total_time : ℝ := 16

-- Define the total distance
def total_distance : ℝ := 2 * workdays * work_distance + weekend_ride

-- Theorem to prove
theorem average_speed_is_25 : 
  total_distance / total_time = 25 := by sorry

end NUMINAMATH_CALUDE_average_speed_is_25_l897_89748


namespace NUMINAMATH_CALUDE_solution_l897_89723

-- Define the set of points satisfying the equation
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + p.2)^2 = p.1^2 + p.2^2}

-- Define the x-axis and y-axis
def X_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}
def Y_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

-- Theorem stating that S is equivalent to the union of X_axis and Y_axis
theorem solution : S = X_axis ∪ Y_axis := by
  sorry

end NUMINAMATH_CALUDE_solution_l897_89723


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l897_89702

theorem billion_to_scientific_notation :
  ∀ (x : ℝ), x = 508 → (x * (10^9 : ℝ)) = 5.08 * (10^11 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l897_89702


namespace NUMINAMATH_CALUDE_parabola_y_order_l897_89783

/-- Given that (-3, y₁), (1, y₂), and (-1/2, y₃) are points on the graph of y = x² - 2x + 3,
    prove that y₂ < y₃ < y₁ -/
theorem parabola_y_order (y₁ y₂ y₃ : ℝ) 
    (h₁ : y₁ = (-3)^2 - 2*(-3) + 3)
    (h₂ : y₂ = 1^2 - 2*1 + 3)
    (h₃ : y₃ = (-1/2)^2 - 2*(-1/2) + 3) :
  y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_order_l897_89783


namespace NUMINAMATH_CALUDE_negative_four_less_than_negative_sqrt_fourteen_l897_89794

theorem negative_four_less_than_negative_sqrt_fourteen : -4 < -Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_negative_four_less_than_negative_sqrt_fourteen_l897_89794


namespace NUMINAMATH_CALUDE_discretionary_income_ratio_l897_89769

/-- Represents Jill's financial situation --/
structure JillFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  vacationFundPercentage : ℝ
  savingsPercentage : ℝ
  socializingPercentage : ℝ
  remainingAmount : ℝ

/-- Theorem stating the ratio of discretionary income to net salary --/
theorem discretionary_income_ratio (j : JillFinances) 
  (h1 : j.netSalary = 3700)
  (h2 : j.vacationFundPercentage = 0.3)
  (h3 : j.savingsPercentage = 0.2)
  (h4 : j.socializingPercentage = 0.35)
  (h5 : j.remainingAmount = 111)
  (h6 : j.discretionaryIncome * (1 - (j.vacationFundPercentage + j.savingsPercentage + j.socializingPercentage)) = j.remainingAmount) :
  j.discretionaryIncome / j.netSalary = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_discretionary_income_ratio_l897_89769


namespace NUMINAMATH_CALUDE_problem_solution_l897_89734

structure Problem where
  -- Define the parabola
  p : ℝ
  parabola : ℝ → ℝ → Prop
  parabola_def : ∀ x y, parabola x y ↔ y^2 = 2*p*x

  -- Define points O, P, and Q
  O : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ

  -- Define line l
  l : ℝ → ℝ → Prop

  -- Conditions
  O_is_origin : O = (0, 0)
  P_coordinates : P = (2, 1)
  p_positive : p > 0
  l_perpendicular_to_OP : (l 2 1) ∧ (∀ x y, l x y → (x - 2) = (y - 1) * (2 / 1))
  Q_on_l : l Q.1 Q.2
  Q_on_parabola : parabola Q.1 Q.2
  OPQ_right_isosceles : (Q.1 - O.1)^2 + (Q.2 - O.2)^2 = (P.1 - O.1)^2 + (P.2 - O.2)^2

theorem problem_solution (prob : Problem) : prob.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l897_89734


namespace NUMINAMATH_CALUDE_min_number_after_operations_l897_89752

def board_operation (S : Finset ℕ) : Finset ℕ :=
  sorry

def min_after_operations (n : ℕ) : ℕ :=
  sorry

theorem min_number_after_operations :
  (min_after_operations 111 = 0) ∧ (min_after_operations 110 = 1) :=
sorry

end NUMINAMATH_CALUDE_min_number_after_operations_l897_89752


namespace NUMINAMATH_CALUDE_expression_proof_l897_89710

theorem expression_proof (x : ℝ) (E : ℝ) : 
  ((x + 3)^2 / E = 3) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ - x₂ = 12 ∧ 
    ((x₁ + 3)^2 / E = 3) ∧ 
    ((x₂ + 3)^2 / E = 3)) → 
  (E = (x + 3)^2 / 3 ∧ E = 12) := by
sorry

end NUMINAMATH_CALUDE_expression_proof_l897_89710


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l897_89750

theorem same_remainder_divisor : ∃! (n : ℕ), n > 0 ∧ 
  ∃ (r : ℕ), r > 0 ∧ r < n ∧ 
  (2287 % n = r) ∧ (2028 % n = r) ∧ (1806 % n = r) :=
by sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l897_89750


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l897_89725

theorem inscribed_circle_radius : ∃ (r : ℝ), 
  (1 / r = 1 / 6 + 1 / 10 + 1 / 15 + 3 * Real.sqrt (1 / (6 * 10) + 1 / (6 * 15) + 1 / (10 * 15))) ∧
  r = 30 / (10 * Real.sqrt 26 + 3) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l897_89725


namespace NUMINAMATH_CALUDE_power_equation_l897_89777

theorem power_equation : 32^4 * 4^5 = 2^30 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l897_89777


namespace NUMINAMATH_CALUDE_isosceles_triangle_is_convex_l897_89742

-- Define an isosceles triangle
structure IsoscelesTriangle where
  sides : Fin 3 → ℝ
  is_isosceles : ∃ (i j : Fin 3), i ≠ j ∧ sides i = sides j

-- Define a convex polygon
def is_convex (polygon : Fin n → ℝ × ℝ) : Prop :=
  ∀ i j : Fin n, ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 →
    ∃ k : Fin n, polygon k = (1 - t) • (polygon i) + t • (polygon j)

-- Theorem statement
theorem isosceles_triangle_is_convex (T : IsoscelesTriangle) :
  is_convex (λ i : Fin 3 => sorry) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_is_convex_l897_89742


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_2pi_3_l897_89773

theorem cos_2alpha_minus_2pi_3 (α : Real) 
  (h : Real.sin (α + π / 6) = 1 / 3) : 
  Real.cos (2 * α - 2 * π / 3) = - 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_2pi_3_l897_89773


namespace NUMINAMATH_CALUDE_margo_round_trip_distance_l897_89729

/-- Calculates the total distance covered in a round trip given the time for each leg and the average speed -/
def total_distance (outward_time return_time avg_speed : ℚ) : ℚ :=
  avg_speed * (outward_time + return_time) / 60

/-- Proves that the total distance covered in the given scenario is 4 miles -/
theorem margo_round_trip_distance :
  total_distance (15 : ℚ) (25 : ℚ) (6 : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_margo_round_trip_distance_l897_89729


namespace NUMINAMATH_CALUDE_problem_solution_l897_89709

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) : (x^4 * y^2) / 8 = 162 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l897_89709


namespace NUMINAMATH_CALUDE_f_sum_equals_two_l897_89749

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

noncomputable def f_deriv : ℝ → ℝ := deriv f

theorem f_sum_equals_two :
  f 2017 + f_deriv 2017 + f (-2017) - f_deriv (-2017) = 2 := by sorry

end NUMINAMATH_CALUDE_f_sum_equals_two_l897_89749


namespace NUMINAMATH_CALUDE_shirts_produced_l897_89726

theorem shirts_produced (shirts_per_minute : ℕ) (minutes_worked : ℕ) : 
  shirts_per_minute = 2 → minutes_worked = 4 → shirts_per_minute * minutes_worked = 8 := by
  sorry

end NUMINAMATH_CALUDE_shirts_produced_l897_89726


namespace NUMINAMATH_CALUDE_ellipse_and_line_equations_l897_89703

/-- Given an ellipse with the specified properties, prove its standard equation and the equations of the intersecting line. -/
theorem ellipse_and_line_equations 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (e : ℝ) 
  (h_e : e = Real.sqrt 2 / 2) 
  (h_point : a^2 * (1/2)^2 + b^2 * (Real.sqrt 2 / 2)^2 = 1) 
  (k : ℝ) 
  (h_intersection : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / (2 * a^2) + y₁^2 / (2 * b^2) = 1 ∧
    x₂^2 / (2 * a^2) + y₂^2 / (2 * b^2) = 1 ∧
    y₁ = k * (x₁ + 1) ∧
    y₂ = k * (x₂ + 1) ∧
    ((x₁ - 1)^2 + y₁^2 + (x₂ - 1)^2 + y₂^2 + 2 * ((x₁ - 1) * (x₂ - 1) + y₁ * y₂))^(1/2) = 2 * Real.sqrt 26 / 3) :
  (a^2 = 2 ∧ b^2 = 1) ∧ (k = 1 ∨ k = -1) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_and_line_equations_l897_89703


namespace NUMINAMATH_CALUDE_cycling_speed_problem_l897_89727

/-- Proves that given the conditions of the problem, B's cycling speed is 20 kmph -/
theorem cycling_speed_problem (a_speed b_speed : ℝ) (delay meeting_distance : ℝ) : 
  a_speed = 10 →
  delay = 7 →
  meeting_distance = 140 →
  b_speed * delay = meeting_distance →
  b_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_cycling_speed_problem_l897_89727


namespace NUMINAMATH_CALUDE_original_number_before_increase_l897_89706

theorem original_number_before_increase (x : ℝ) : x * 1.3 = 650 → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_before_increase_l897_89706


namespace NUMINAMATH_CALUDE_even_periodic_function_value_l897_89745

/-- A function that is even and has a period of 2 -/
def EvenPeriodicFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ (∀ x, f (x + 2) = f x)

theorem even_periodic_function_value 
  (f : ℝ → ℝ) 
  (h_even_periodic : EvenPeriodicFunction f)
  (h_def : ∀ x ∈ Set.Ioo 0 1, f x = x + 1) :
  ∀ x ∈ Set.Ioo 1 2, f x = 3 - x := by
sorry

end NUMINAMATH_CALUDE_even_periodic_function_value_l897_89745


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l897_89776

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Define the open interval (2, 3]
def open_interval : Set ℝ := {x | 2 < x ∧ x ≤ 3}

-- State the theorem
theorem complement_A_intersect_B : (Aᶜ ∩ B) = open_interval := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l897_89776


namespace NUMINAMATH_CALUDE_eggs_per_box_l897_89784

/-- Given a chicken coop with hens that lay eggs daily, prove the number of eggs per box -/
theorem eggs_per_box 
  (num_hens : ℕ) 
  (eggs_per_hen_per_day : ℕ) 
  (days_per_week : ℕ) 
  (boxes_per_week : ℕ) 
  (h1 : num_hens = 270)
  (h2 : eggs_per_hen_per_day = 1)
  (h3 : days_per_week = 7)
  (h4 : boxes_per_week = 315) :
  (num_hens * eggs_per_hen_per_day * days_per_week) / boxes_per_week = 6 :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_box_l897_89784


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l897_89779

theorem correct_quotient_proof (dividend : ℕ) (wrong_divisor correct_divisor wrong_quotient : ℕ) 
  (h1 : dividend = wrong_divisor * wrong_quotient)
  (h2 : wrong_divisor = 121)
  (h3 : correct_divisor = 215)
  (h4 : wrong_quotient = 432) :
  dividend / correct_divisor = 243 := by
sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l897_89779


namespace NUMINAMATH_CALUDE_first_series_seasons_l897_89790

/-- Represents the number of seasons in the first movie series -/
def S : ℕ := sorry

/-- Represents the number of seasons in the second movie series -/
def second_series_seasons : ℕ := 14

/-- Represents the original number of episodes per season -/
def original_episodes_per_season : ℕ := 16

/-- Represents the number of episodes lost per season -/
def lost_episodes_per_season : ℕ := 2

/-- Represents the total number of episodes remaining after the loss -/
def total_remaining_episodes : ℕ := 364

/-- Theorem stating that the number of seasons in the first movie series is 12 -/
theorem first_series_seasons :
  S = 12 :=
by sorry

end NUMINAMATH_CALUDE_first_series_seasons_l897_89790


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_length_l897_89797

/-- Represents a trapezoid ABCD with specific side lengths and angle -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  cos_BCD : ℝ
  h_AB : AB = 27
  h_CD : CD = 28
  h_BC : BC = 5
  h_cos_BCD : cos_BCD = -2/7

/-- The length of the diagonal AC in the trapezoid -/
def diagonal_AC (t : Trapezoid) : Set ℝ :=
  {28, 2 * Real.sqrt 181}

/-- Theorem stating that the diagonal AC of the trapezoid is either 28 or 2√181 -/
theorem trapezoid_diagonal_length (t : Trapezoid) :
  ∃ x ∈ diagonal_AC t, x = (Real.sqrt ((t.AB - t.BC)^2 + (t.CD * Real.sqrt (1 - t.cos_BCD^2))^2)) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_length_l897_89797


namespace NUMINAMATH_CALUDE_mari_buttons_l897_89722

theorem mari_buttons (kendra_buttons : ℕ) (mari_buttons : ℕ) : 
  kendra_buttons = 15 →
  mari_buttons = 5 * kendra_buttons + 4 →
  mari_buttons = 79 := by
  sorry

end NUMINAMATH_CALUDE_mari_buttons_l897_89722


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l897_89759

/-- Given a parabola with equation y² = 4ax where a < 0, 
    prove that the coordinates of its focus are (a, 0) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a < 0) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4*a*x}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (a, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l897_89759


namespace NUMINAMATH_CALUDE_pyramid_paint_theorem_l897_89746

/-- Represents a pyramid-like structure with a given number of floors -/
structure PyramidStructure where
  floors : Nat

/-- Calculates the number of painted faces on one side of the structure -/
def sideFaces (p : PyramidStructure) : Nat :=
  (p.floors * (p.floors + 1)) / 2

/-- Calculates the total number of red-painted faces -/
def redFaces (p : PyramidStructure) : Nat :=
  4 * sideFaces p

/-- Calculates the total number of blue-painted faces -/
def blueFaces (p : PyramidStructure) : Nat :=
  sideFaces p

/-- Calculates the total number of painted faces -/
def totalPaintedFaces (p : PyramidStructure) : Nat :=
  redFaces p + blueFaces p

/-- Theorem stating the ratio of red to blue painted faces and the total number of painted faces -/
theorem pyramid_paint_theorem (p : PyramidStructure) (h : p.floors = 25) :
  redFaces p / blueFaces p = 4 ∧ totalPaintedFaces p = 1625 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_paint_theorem_l897_89746


namespace NUMINAMATH_CALUDE_rectangle_max_area_max_area_achievable_l897_89713

/-- Given a rectangle with perimeter 40 inches, its maximum area is 100 square inches. -/
theorem rectangle_max_area :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * x + 2 * y = 40 →
  ∀ a : ℝ,
  (0 < a ∧ ∃ w h : ℝ, w > 0 ∧ h > 0 ∧ 2 * w + 2 * h = 40 ∧ a = w * h) →
  x * y ≥ a :=
by sorry

/-- The maximum area of 100 square inches is achievable. -/
theorem max_area_achievable :
  ∃ x y : ℝ,
  x > 0 ∧ y > 0 ∧
  2 * x + 2 * y = 40 ∧
  x * y = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_max_area_achievable_l897_89713


namespace NUMINAMATH_CALUDE_rectangles_in_4x5_grid_l897_89708

/-- The number of rectangles in a grid with sides along the grid lines -/
def count_rectangles (m n : ℕ) : ℕ :=
  (m * (m + 1) * n * (n + 1)) / 4

/-- Theorem: In a 4 × 5 grid, the total number of rectangles with sides along the grid lines is 24 -/
theorem rectangles_in_4x5_grid :
  count_rectangles 4 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_4x5_grid_l897_89708


namespace NUMINAMATH_CALUDE_same_gender_probability_l897_89760

def num_male : ℕ := 3
def num_female : ℕ := 2
def total_volunteers : ℕ := num_male + num_female
def num_to_select : ℕ := 2

-- Function to calculate combinations
def combination (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def total_ways : ℕ := combination total_volunteers num_to_select
def same_gender_ways : ℕ := combination num_male num_to_select + combination num_female num_to_select

theorem same_gender_probability : 
  (same_gender_ways : ℚ) / total_ways = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_same_gender_probability_l897_89760


namespace NUMINAMATH_CALUDE_non_acute_triangle_sides_count_l897_89792

/-- Given a triangle with two sides of lengths 20 and 19, this function returns the number of possible integer lengths for the third side that make the triangle not acute. -/
def count_non_acute_triangle_sides : ℕ :=
  let a : ℕ := 20
  let b : ℕ := 19
  let possible_sides := (Finset.range 37).filter (fun s => 
    (s > 1 ∧ s < 39) ∧  -- Triangle inequality
    ((s * s ≥ a * a + b * b) ∨  -- s is the longest side (obtuse or right triangle)
     (a * a ≥ s * s + b * b)))  -- a is the longest side (obtuse or right triangle)
  possible_sides.card

/-- Theorem stating that there are exactly 16 possible integer lengths for the third side of a triangle with sides 20 and 19 that make it not acute. -/
theorem non_acute_triangle_sides_count : count_non_acute_triangle_sides = 16 := by
  sorry

end NUMINAMATH_CALUDE_non_acute_triangle_sides_count_l897_89792


namespace NUMINAMATH_CALUDE_wendy_recycling_points_l897_89754

def points_per_bag : ℕ := 5
def total_bags : ℕ := 11
def unrecycled_bags : ℕ := 2

theorem wendy_recycling_points :
  (total_bags - unrecycled_bags) * points_per_bag = 45 :=
by sorry

end NUMINAMATH_CALUDE_wendy_recycling_points_l897_89754
