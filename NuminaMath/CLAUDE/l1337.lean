import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1337_133749

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 < x + 6} = {x : ℝ | -2 < x ∧ x < 3} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1337_133749


namespace NUMINAMATH_CALUDE_bacteria_growth_l1337_133770

/-- The factor by which the bacteria population increases each minute -/
def growth_factor : ℕ := 2

/-- The number of minutes that pass -/
def time : ℕ := 4

/-- The function that calculates the population after n minutes -/
def population (n : ℕ) : ℕ := growth_factor ^ n

/-- Theorem stating that after 4 minutes, the population is 16 times the original -/
theorem bacteria_growth :
  population time = 16 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1337_133770


namespace NUMINAMATH_CALUDE_bread_cost_calculation_l1337_133747

/-- Calculates the total cost of bread for a committee luncheon --/
def calculate_bread_cost (committee_size : ℕ) (sandwiches_per_person : ℕ) 
  (bread_types : ℕ) (croissant_pack_size : ℕ) (croissant_pack_price : ℚ)
  (ciabatta_pack_size : ℕ) (ciabatta_pack_price : ℚ)
  (multigrain_pack_size : ℕ) (multigrain_pack_price : ℚ)
  (discount_threshold : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  sorry

/-- The total cost of bread for the committee luncheon is $51.36 --/
theorem bread_cost_calculation :
  calculate_bread_cost 24 2 3 12 8 10 9 20 7 50 0.1 0.07 = 51.36 := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_calculation_l1337_133747


namespace NUMINAMATH_CALUDE_tangyuan_purchase_solution_l1337_133733

/-- Represents the number and price of tangyuan bags for two brands -/
structure TangyuanPurchase where
  brandA_quantity : ℕ
  brandB_quantity : ℕ
  brandA_price : ℕ
  brandB_price : ℕ

/-- Checks if a TangyuanPurchase satisfies all conditions -/
def is_valid_purchase (p : TangyuanPurchase) : Prop :=
  p.brandA_quantity + p.brandB_quantity = 1000 ∧
  p.brandA_quantity = 2 * p.brandB_quantity + 20 ∧
  p.brandB_price = p.brandA_price + 6 ∧
  5 * p.brandA_price = 3 * p.brandB_price

/-- The theorem to be proved -/
theorem tangyuan_purchase_solution :
  ∃ (p : TangyuanPurchase),
    is_valid_purchase p ∧
    p.brandA_quantity = 670 ∧
    p.brandB_quantity = 330 ∧
    p.brandA_price = 9 ∧
    p.brandB_price = 15 :=
  sorry

end NUMINAMATH_CALUDE_tangyuan_purchase_solution_l1337_133733


namespace NUMINAMATH_CALUDE_carpenter_rate_proof_l1337_133776

def carpenter_hourly_rate (total_estimate : ℚ) (material_cost : ℚ) (job_duration : ℚ) : ℚ :=
  (total_estimate - material_cost) / job_duration

theorem carpenter_rate_proof (total_estimate : ℚ) (material_cost : ℚ) (job_duration : ℚ)
  (h1 : total_estimate = 980)
  (h2 : material_cost = 560)
  (h3 : job_duration = 15) :
  carpenter_hourly_rate total_estimate material_cost job_duration = 28 := by
  sorry

end NUMINAMATH_CALUDE_carpenter_rate_proof_l1337_133776


namespace NUMINAMATH_CALUDE_min_value_expression_l1337_133727

theorem min_value_expression (a b m n : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_sum : a + b = 1) (h_prod : m * n = 2) :
  2 ≤ (a * m + b * n) * (b * m + a * n) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1337_133727


namespace NUMINAMATH_CALUDE_cafeteria_pies_problem_l1337_133718

def cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

theorem cafeteria_pies_problem :
  cafeteria_pies 75 19 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_problem_l1337_133718


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1337_133756

theorem no_solution_for_equation :
  ¬ ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (2 / a + 2 / b = 1 / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1337_133756


namespace NUMINAMATH_CALUDE_inequality_proof_l1337_133758

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d)) + (b^3 / (c + d + a)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1337_133758


namespace NUMINAMATH_CALUDE_shortest_path_is_3_sqrt_2_l1337_133721

/-- A polyhedron with right dihedral angles that unfolds into three adjacent unit squares -/
structure RightAnglePolyhedron where
  -- We don't need to define the full structure, just the properties we need
  unfoldsToThreeUnitSquares : Bool

/-- Two vertices on the polyhedron -/
structure Vertex where
  -- We don't need to define the full structure, just declare it exists

/-- The shortest path between two vertices on the surface of the polyhedron -/
def shortestPath (p : RightAnglePolyhedron) (v1 v2 : Vertex) : ℝ :=
  sorry

/-- Theorem: The shortest path between opposite corners of the unfolded net is 3√2 -/
theorem shortest_path_is_3_sqrt_2 (p : RightAnglePolyhedron) (x y : Vertex) :
  p.unfoldsToThreeUnitSquares → shortestPath p x y = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_is_3_sqrt_2_l1337_133721


namespace NUMINAMATH_CALUDE_permutation_ratio_l1337_133779

/-- The number of permutations of m elements chosen from n elements -/
def A (n m : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial (n - m))

/-- Theorem stating that the ratio of A(n,m) to A(n-1,m-1) equals n -/
theorem permutation_ratio (n m : ℕ) (h : n ≥ m) :
  A n m / A (n - 1) (m - 1) = n := by sorry

end NUMINAMATH_CALUDE_permutation_ratio_l1337_133779


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l1337_133785

theorem rectangular_garden_area (perimeter width length : ℝ) : 
  perimeter = 72 →
  length = 3 * width →
  2 * length + 2 * width = perimeter →
  length * width = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l1337_133785


namespace NUMINAMATH_CALUDE_equation_solution_system_solution_l1337_133778

-- Equation 1
theorem equation_solution (x : ℚ) : 
  (3 * x + 1) / 5 = 1 - (4 * x + 3) / 2 ↔ x = -7 / 26 := by sorry

-- System of equations
theorem system_solution (x y : ℚ) : 
  (3 * x - 4 * y = 14 ∧ 5 * x + 4 * y = 2) ↔ (x = 2 ∧ y = -2) := by sorry

end NUMINAMATH_CALUDE_equation_solution_system_solution_l1337_133778


namespace NUMINAMATH_CALUDE_function_upper_bound_l1337_133728

/-- A function satisfying the given conditions -/
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, x ≥ 0 → ∃ (fx : ℝ), f x = fx) ∧  -- f is defined for x ≥ 0
  (∀ x y, x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)) ∧
  (∃ M : ℝ, M > 0 ∧ ∀ x, 0 ≤ x → x ≤ 1 → |f x| ≤ M)

/-- The main theorem -/
theorem function_upper_bound (f : ℝ → ℝ) (h : satisfies_conditions f) :
  ∀ x, x ≥ 0 → f x ≤ x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_upper_bound_l1337_133728


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_line_l1337_133736

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_line 
  (α β γ : Plane) (l : Line) 
  (distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  perpendicular_line_plane l α → 
  parallel_line_plane l β → 
  perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_line_l1337_133736


namespace NUMINAMATH_CALUDE_expression_value_l1337_133709

theorem expression_value (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : c - d = -3) : 
  (b - c) - (-d - a) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1337_133709


namespace NUMINAMATH_CALUDE_school_students_count_l1337_133763

theorem school_students_count (football cricket both neither : ℕ) 
  (h1 : football = 325)
  (h2 : cricket = 175)
  (h3 : neither = 50)
  (h4 : both = 90) :
  football + cricket - both + neither = 460 :=
by sorry

end NUMINAMATH_CALUDE_school_students_count_l1337_133763


namespace NUMINAMATH_CALUDE_wedding_catering_budget_l1337_133780

/-- Calculates the total catering budget for a wedding given the specified conditions. -/
theorem wedding_catering_budget 
  (total_guests : ℕ) 
  (steak_to_chicken_ratio : ℕ) 
  (steak_cost chicken_cost : ℕ) : 
  total_guests = 80 → 
  steak_to_chicken_ratio = 3 → 
  steak_cost = 25 → 
  chicken_cost = 18 → 
  (total_guests * steak_cost * steak_to_chicken_ratio + total_guests * chicken_cost) / (steak_to_chicken_ratio + 1) = 1860 := by
  sorry

#eval (80 * 25 * 3 + 80 * 18) / (3 + 1)

end NUMINAMATH_CALUDE_wedding_catering_budget_l1337_133780


namespace NUMINAMATH_CALUDE_larger_number_problem_l1337_133745

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 45) (h2 : x - y = 5) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1337_133745


namespace NUMINAMATH_CALUDE_mean_not_imply_concentration_stability_range_imply_concentration_stability_std_dev_imply_concentration_stability_variance_imply_concentration_stability_l1337_133737

/- Define a dataset as a list of real numbers -/
def Dataset := List ℝ

/- Define statistical measures -/
def range (data : Dataset) : ℝ := sorry
def mean (data : Dataset) : ℝ := sorry
def standardDeviation (data : Dataset) : ℝ := sorry
def variance (data : Dataset) : ℝ := sorry

/- Define a measure of concentration and stability -/
def isConcentratedAndStable (data : Dataset) : Prop := sorry

/- Theorem stating that mean does not imply concentration and stability -/
theorem mean_not_imply_concentration_stability :
  ∃ (data1 data2 : Dataset), mean data1 < mean data2 ∧
    (isConcentratedAndStable data1 ↔ ¬isConcentratedAndStable data2) := by sorry

/- Theorems stating that other measures imply concentration and stability -/
theorem range_imply_concentration_stability :
  ∀ (data1 data2 : Dataset), range data1 < range data2 →
    (isConcentratedAndStable data1 → isConcentratedAndStable data2) := by sorry

theorem std_dev_imply_concentration_stability :
  ∀ (data1 data2 : Dataset), standardDeviation data1 < standardDeviation data2 →
    (isConcentratedAndStable data1 → isConcentratedAndStable data2) := by sorry

theorem variance_imply_concentration_stability :
  ∀ (data1 data2 : Dataset), variance data1 < variance data2 →
    (isConcentratedAndStable data1 → isConcentratedAndStable data2) := by sorry

end NUMINAMATH_CALUDE_mean_not_imply_concentration_stability_range_imply_concentration_stability_std_dev_imply_concentration_stability_variance_imply_concentration_stability_l1337_133737


namespace NUMINAMATH_CALUDE_painting_area_is_5400_l1337_133730

/-- The area of a painting inside a uniform frame -/
def painting_area (outer_width : ℝ) (outer_height : ℝ) (frame_width : ℝ) : ℝ :=
  (outer_width - 2 * frame_width) * (outer_height - 2 * frame_width)

/-- Theorem: The area of the painting inside the frame is 5400 cm² -/
theorem painting_area_is_5400 :
  painting_area 90 120 15 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_painting_area_is_5400_l1337_133730


namespace NUMINAMATH_CALUDE_bryans_milk_volume_l1337_133722

/-- The volume of milk in the first bottle, given the conditions of Bryan's milk purchase --/
theorem bryans_milk_volume (total_volume : ℚ) (second_bottle : ℚ) (third_bottle : ℚ) 
  (h1 : total_volume = 3)
  (h2 : second_bottle = 750 / 1000)
  (h3 : third_bottle = 250 / 1000) :
  total_volume - second_bottle - third_bottle = 2 := by
  sorry

end NUMINAMATH_CALUDE_bryans_milk_volume_l1337_133722


namespace NUMINAMATH_CALUDE_figure_214_is_triangle_l1337_133724

/-- Represents the figures in the sequence -/
inductive Figure
| triangle : Figure
| square : Figure
| circle : Figure

/-- The pattern of the sequence -/
def pattern : List Figure := 
  [Figure.triangle, Figure.square, Figure.triangle, Figure.circle]

/-- The length of the pattern -/
def pattern_length : Nat := pattern.length

/-- The figure at a given position in the sequence -/
def figure_at_position (n : Nat) : Figure :=
  pattern[n % pattern_length]'
  (by 
    have h : n % pattern_length < pattern_length := 
      Nat.mod_lt n (Nat.zero_lt_succ _)
    exact h
  )

/-- Theorem: The 214th figure in the sequence is a triangle -/
theorem figure_214_is_triangle : 
  figure_at_position 213 = Figure.triangle :=
sorry

end NUMINAMATH_CALUDE_figure_214_is_triangle_l1337_133724


namespace NUMINAMATH_CALUDE_tom_age_ratio_l1337_133732

/-- Tom's current age -/
def T : ℕ := sorry

/-- Number of years ago mentioned in the second condition -/
def N : ℕ := 5

/-- Sum of the current ages of Tom's three children -/
def children_sum : ℕ := T / 2

/-- Tom's age N years ago -/
def tom_age_N_years_ago : ℕ := T - N

/-- Sum of the ages of Tom's children N years ago -/
def children_sum_N_years_ago : ℕ := children_sum - 3 * N

/-- The theorem stating the ratio of T to N -/
theorem tom_age_ratio : T / N = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_age_ratio_l1337_133732


namespace NUMINAMATH_CALUDE_one_third_1206_percent_of_400_l1337_133735

theorem one_third_1206_percent_of_400 : 
  (1206 / 3) / 400 * 100 = 100.5 := by sorry

end NUMINAMATH_CALUDE_one_third_1206_percent_of_400_l1337_133735


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1337_133775

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ p q : ℝ, p + q = -27 ∧ 81 - 27*x - x^2 = 0 → x = p ∨ x = q) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1337_133775


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_undefined_l1337_133774

theorem sqrt_x_minus_3_undefined (x : ℕ+) : 
  ¬ (∃ (y : ℝ), y^2 = (x : ℝ) - 3) ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_undefined_l1337_133774


namespace NUMINAMATH_CALUDE_lowest_common_multiple_even_14_to_21_l1337_133757

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem lowest_common_multiple_even_14_to_21 :
  ∀ n : ℕ, n > 0 →
  (∀ k : ℕ, 14 ≤ k → k ≤ 21 → is_even k → divides k n) →
  n ≥ 5040 :=
sorry

end NUMINAMATH_CALUDE_lowest_common_multiple_even_14_to_21_l1337_133757


namespace NUMINAMATH_CALUDE_fraction_value_l1337_133714

theorem fraction_value (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (1 / x + 1 / y) / (1 / x - 1 / y) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1337_133714


namespace NUMINAMATH_CALUDE_floor_sum_equals_129_l1337_133739

theorem floor_sum_equals_129 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + 2*b^2 = 2016)
  (h2 : c^2 + 2*d^2 = 2016)
  (h3 : a*c = 1024)
  (h4 : b*d = 1024) :
  ⌊a + b + c + d⌋ = 129 := by
sorry

end NUMINAMATH_CALUDE_floor_sum_equals_129_l1337_133739


namespace NUMINAMATH_CALUDE_path_bounds_l1337_133734

/-- Represents a tile with two segments -/
structure Tile :=
  (segments : Fin 2 → Unit)

/-- Represents a 2N × 2N board assembled with tiles -/
structure Board (N : ℕ) :=
  (tiles : Fin (4 * N^2) → Tile)

/-- The number of paths on a board -/
def num_paths (N : ℕ) (board : Board N) : ℕ := sorry

theorem path_bounds (N : ℕ) (board : Board N) :
  4 * N ≤ num_paths N board ∧ num_paths N board ≤ 2 * N^2 + 2 * N :=
sorry

end NUMINAMATH_CALUDE_path_bounds_l1337_133734


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l1337_133792

/-- The x-coordinate of the vertex of a quadratic function f(x) = x^2 + 2px + 3q -/
def vertex_x_coord (p q : ℝ) : ℝ := -p

/-- The quadratic function f(x) = x^2 + 2px + 3q -/
def f (p q x : ℝ) : ℝ := x^2 + 2*p*x + 3*q

theorem vertex_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∀ x : ℝ, f p q x ≥ f p q (vertex_x_coord p q) :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l1337_133792


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1337_l1337_133760

theorem largest_prime_factor_of_1337 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 1337 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 1337 → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1337_l1337_133760


namespace NUMINAMATH_CALUDE_cos_18_degrees_l1337_133767

open Real

theorem cos_18_degrees : cos (18 * π / 180) = (Real.sqrt (10 + 2 * Real.sqrt 5)) / 4 := by sorry

end NUMINAMATH_CALUDE_cos_18_degrees_l1337_133767


namespace NUMINAMATH_CALUDE_students_liking_computing_l1337_133708

theorem students_liking_computing (total : ℕ) (both : ℕ) 
  (h1 : total = 33)
  (h2 : both = 3)
  (h3 : ∀ (pe_only computing_only : ℕ), 
    pe_only + computing_only + both = total → 
    pe_only = computing_only / 2) :
  ∃ (pe_only computing_only : ℕ),
    pe_only + computing_only + both = total ∧
    pe_only = computing_only / 2 ∧
    computing_only + both = 23 := by
sorry

end NUMINAMATH_CALUDE_students_liking_computing_l1337_133708


namespace NUMINAMATH_CALUDE_egg_count_l1337_133791

theorem egg_count (initial_eggs : ℕ) (added_eggs : ℕ) : 
  initial_eggs = 7 → added_eggs = 4 → initial_eggs + added_eggs = 11 := by
  sorry

#check egg_count

end NUMINAMATH_CALUDE_egg_count_l1337_133791


namespace NUMINAMATH_CALUDE_f_min_at_three_l1337_133782

/-- The quadratic function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- Theorem: The function f(x) = 3x^2 - 18x + 7 has a minimum value when x = 3 -/
theorem f_min_at_three : 
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_min_at_three_l1337_133782


namespace NUMINAMATH_CALUDE_books_not_shared_l1337_133783

/-- The number of books that are in either Emily's or Olivia's collection, but not both -/
def books_in_either_not_both (shared_books : ℕ) (emily_total : ℕ) (olivia_unique : ℕ) : ℕ :=
  (emily_total - shared_books) + olivia_unique

/-- Theorem stating the number of books in either Emily's or Olivia's collection, but not both -/
theorem books_not_shared (shared_books : ℕ) (emily_total : ℕ) (olivia_unique : ℕ) 
  (h1 : shared_books = 15)
  (h2 : emily_total = 23)
  (h3 : olivia_unique = 8) :
  books_in_either_not_both shared_books emily_total olivia_unique = 16 := by
  sorry

end NUMINAMATH_CALUDE_books_not_shared_l1337_133783


namespace NUMINAMATH_CALUDE_mary_picked_14_oranges_l1337_133746

/-- The number of oranges picked by Jason -/
def jason_oranges : ℕ := 41

/-- The total number of oranges picked -/
def total_oranges : ℕ := 55

/-- The number of oranges picked by Mary -/
def mary_oranges : ℕ := total_oranges - jason_oranges

theorem mary_picked_14_oranges : mary_oranges = 14 := by
  sorry

end NUMINAMATH_CALUDE_mary_picked_14_oranges_l1337_133746


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l1337_133771

theorem rectangular_field_dimensions (m : ℝ) : 
  (3 * m + 8 > 0) →
  (m - 3 > 0) →
  (3 * m + 8) * (m - 3) = 85 →
  m = (1 + Real.sqrt 1309) / 6 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l1337_133771


namespace NUMINAMATH_CALUDE_A_intersect_B_empty_l1337_133702

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem A_intersect_B_empty : A ∩ B = ∅ := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_empty_l1337_133702


namespace NUMINAMATH_CALUDE_div_2880_by_smallest_is_square_smaller_divisors_not_square_l1337_133742

/-- The smallest positive integer that divides 2880 and results in a perfect square -/
def smallest_divisor_to_square : ℕ := 10

/-- 2880 divided by the smallest divisor is a perfect square -/
theorem div_2880_by_smallest_is_square :
  ∃ m : ℕ, 2880 / smallest_divisor_to_square = m ^ 2 :=
sorry

/-- For any positive integer smaller than the smallest divisor, 
    dividing 2880 by it does not result in a perfect square -/
theorem smaller_divisors_not_square :
  ∀ k : ℕ, 0 < k → k < smallest_divisor_to_square →
  ¬∃ m : ℕ, 2880 / k = m ^ 2 :=
sorry

end NUMINAMATH_CALUDE_div_2880_by_smallest_is_square_smaller_divisors_not_square_l1337_133742


namespace NUMINAMATH_CALUDE_kays_total_exercise_time_l1337_133765

/-- Kay's weekly exercise routine -/
structure ExerciseRoutine where
  aerobics : ℕ
  weightTraining : ℕ

/-- The total exercise time is the sum of aerobics and weight training times -/
def totalExerciseTime (routine : ExerciseRoutine) : ℕ :=
  routine.aerobics + routine.weightTraining

/-- Kay's actual exercise routine -/
def kaysRoutine : ExerciseRoutine :=
  { aerobics := 150, weightTraining := 100 }

/-- Theorem: Kay's total exercise time is 250 minutes per week -/
theorem kays_total_exercise_time :
  totalExerciseTime kaysRoutine = 250 := by
  sorry

end NUMINAMATH_CALUDE_kays_total_exercise_time_l1337_133765


namespace NUMINAMATH_CALUDE_bucket_fill_time_l1337_133704

/-- Given that two-thirds of a bucket is filled in 100 seconds,
    prove that the time taken to fill the bucket completely is 150 seconds. -/
theorem bucket_fill_time (fill_rate : ℝ) (h : fill_rate * (2/3) = 1/100) :
  (1 / fill_rate) = 150 := by
  sorry

end NUMINAMATH_CALUDE_bucket_fill_time_l1337_133704


namespace NUMINAMATH_CALUDE_tiger_speed_l1337_133762

/-- Proves that the tiger's speed is 30 kmph given the problem conditions -/
theorem tiger_speed (tiger_head_start : ℝ) (zebra_chase_time : ℝ) (zebra_speed : ℝ)
  (h1 : tiger_head_start = 5)
  (h2 : zebra_chase_time = 6)
  (h3 : zebra_speed = 55) :
  tiger_head_start * (zebra_speed * zebra_chase_time / (tiger_head_start + zebra_chase_time)) = 30 * tiger_head_start :=
by sorry

end NUMINAMATH_CALUDE_tiger_speed_l1337_133762


namespace NUMINAMATH_CALUDE_eight_digit_divisible_by_11_l1337_133797

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def digit (d : ℕ) : Prop :=
  d ≥ 0 ∧ d < 10

theorem eight_digit_divisible_by_11 (m : ℕ) :
  digit m →
  is_divisible_by_11 (73400000 + m * 100000 + 8527) →
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_eight_digit_divisible_by_11_l1337_133797


namespace NUMINAMATH_CALUDE_tan_sum_eq_neg_one_l1337_133755

theorem tan_sum_eq_neg_one (α β : ℝ) 
  (h : 2 * Real.sin β * Real.sin (α - π/4) = Real.sin (α - β + π/4)) : 
  Real.tan (α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_eq_neg_one_l1337_133755


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l1337_133793

theorem cylinder_volume_equality (r h x : ℝ) : 
  r = 5 ∧ h = 7 ∧ x > 0 ∧ 
  π * (2 * r + x)^2 * h = π * r^2 * (3 * h + x) → 
  x = (5 + Real.sqrt 9125) / 14 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l1337_133793


namespace NUMINAMATH_CALUDE_range_of_a_l1337_133777

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x ≥ a
def q (x : ℝ) : Prop := |x - 1| < 1

-- Define the property that p is necessary but not sufficient for q
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  necessary_not_sufficient a → a ≤ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l1337_133777


namespace NUMINAMATH_CALUDE_eight_vases_needed_l1337_133706

/-- Represents the number of flowers of each type -/
structure FlowerCounts where
  roses : ℕ
  tulips : ℕ
  lilies : ℕ

/-- Represents the capacity of a vase for each flower type -/
structure VaseCapacity where
  roses : ℕ
  tulips : ℕ
  lilies : ℕ

/-- Calculates the minimum number of vases needed -/
def minVasesNeeded (flowers : FlowerCounts) (capacity : VaseCapacity) : ℕ :=
  sorry

/-- Theorem stating that 8 vases are needed for the given flower counts -/
theorem eight_vases_needed :
  let flowers := FlowerCounts.mk 20 15 5
  let capacity := VaseCapacity.mk 6 8 4
  minVasesNeeded flowers capacity = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_vases_needed_l1337_133706


namespace NUMINAMATH_CALUDE_max_value_product_max_value_achieved_l1337_133794

theorem max_value_product (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  x^2 * y^3 * z ≤ 9/16 := by
  sorry

theorem max_value_achieved (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  ∃ x y z, x^2 * y^3 * z = 9/16 ∧ x + y + z = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_product_max_value_achieved_l1337_133794


namespace NUMINAMATH_CALUDE_max_value_sum_l1337_133787

theorem max_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt (3 * b^2) = Real.sqrt ((1 - a) * (1 + a))) : 
  ∃ (x : ℝ), x = a + Real.sqrt (3 * b^2) ∧ x ≤ Real.sqrt 2 ∧ 
  ∀ (y : ℝ), y = a + Real.sqrt (3 * b^2) → y ≤ x := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_l1337_133787


namespace NUMINAMATH_CALUDE_polygon_sides_count_l1337_133743

theorem polygon_sides_count : ∃ n : ℕ, 
  n > 2 ∧ 
  (n * (n - 3)) / 2 = 2 * n ∧ 
  ∀ m : ℕ, m > 2 → (m * (m - 3)) / 2 = 2 * m → m = n :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l1337_133743


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l1337_133772

/-- Represents a distribution of balls into boxes -/
def Distribution := List Nat

/-- Calculates the number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distributeWays (n : Nat) (k : Nat) : Nat :=
  sorry

/-- The number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distributeWays 6 3 = 122 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l1337_133772


namespace NUMINAMATH_CALUDE_smallest_AAB_value_l1337_133768

/-- Represents a digit (1 to 9) -/
def Digit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- Represents a two-digit number AB -/
def TwoDigitNumber (A B : Digit) : ℕ := 10 * A.val + B.val

/-- Represents a three-digit number AAB -/
def ThreeDigitNumber (A B : Digit) : ℕ := 100 * A.val + 10 * A.val + B.val

/-- The main theorem -/
theorem smallest_AAB_value :
  ∀ (A B : Digit),
    A ≠ B →
    TwoDigitNumber A B = (ThreeDigitNumber A B) / 7 →
    ∀ (A' B' : Digit),
      A' ≠ B' →
      TwoDigitNumber A' B' = (ThreeDigitNumber A' B') / 7 →
      ThreeDigitNumber A B ≤ ThreeDigitNumber A' B' →
      ThreeDigitNumber A B = 664 :=
sorry

end NUMINAMATH_CALUDE_smallest_AAB_value_l1337_133768


namespace NUMINAMATH_CALUDE_range_of_function_l1337_133740

theorem range_of_function : 
  ∀ (x : ℝ), 12 ≤ |x + 5| - |x - 3| + 4 ∧ 
  (∃ (x₁ x₂ : ℝ), |x₁ + 5| - |x₁ - 3| + 4 = 12 ∧ |x₂ + 5| - |x₂ - 3| + 4 = 18) ∧
  (∀ (y : ℝ), (∃ (x : ℝ), y = |x + 5| - |x - 3| + 4) → 12 ≤ y ∧ y ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_range_of_function_l1337_133740


namespace NUMINAMATH_CALUDE_base_9_101_to_decimal_l1337_133789

def base_9_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ (digits.length - 1 - i))) 0

theorem base_9_101_to_decimal :
  base_9_to_decimal [1, 0, 1] = 82 := by
  sorry

end NUMINAMATH_CALUDE_base_9_101_to_decimal_l1337_133789


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1337_133716

theorem perfect_square_trinomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - a*x + 9 = (x + b)^2) → a = 6 ∨ a = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1337_133716


namespace NUMINAMATH_CALUDE_inscribed_cube_properties_l1337_133748

/-- Given a cube with a sphere inscribed in it, and another cube inscribed in that sphere,
    this theorem proves the surface area and volume of the inner cube. -/
theorem inscribed_cube_properties (outer_cube_surface_area : ℝ) 
  (h : outer_cube_surface_area = 96) :
  ∃ (inner_cube_surface_area inner_cube_volume : ℝ),
    inner_cube_surface_area = 32 ∧
    inner_cube_volume = 64 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_properties_l1337_133748


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l1337_133786

theorem product_remainder_mod_five :
  (14452 * 15652 * 16781) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l1337_133786


namespace NUMINAMATH_CALUDE_log_ratio_squared_l1337_133790

theorem log_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1)
  (h1 : Real.log a / Real.log 3 = Real.log 81 / Real.log b) (h2 : a * b = 243) :
  (Real.log (a / b) / Real.log 3)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l1337_133790


namespace NUMINAMATH_CALUDE_tan_theta_eq_two_implies_expression_eq_neg_two_l1337_133784

theorem tan_theta_eq_two_implies_expression_eq_neg_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / 
  (Real.sin (π / 2 - θ) - Real.sin (π - θ)) = -2 := by sorry

end NUMINAMATH_CALUDE_tan_theta_eq_two_implies_expression_eq_neg_two_l1337_133784


namespace NUMINAMATH_CALUDE_largest_fraction_l1337_133754

theorem largest_fraction :
  let f1 := 397 / 101
  let f2 := 487 / 121
  let f3 := 596 / 153
  let f4 := 678 / 173
  let f5 := 796 / 203
  f2 > f1 ∧ f2 > f3 ∧ f2 > f4 ∧ f2 > f5 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l1337_133754


namespace NUMINAMATH_CALUDE_exists_x_y_sequences_l1337_133798

/-- The sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 4
  | 1 => 22
  | (n + 2) => 6 * a (n + 1) - a n

/-- Theorem stating the existence of sequences x_n and y_n satisfying the given property -/
theorem exists_x_y_sequences :
  ∃ (x y : ℕ → ℕ), ∀ n, a n = (y n ^ 2 + 7 : ℚ) / ((x n : ℚ) - y n) :=
sorry

end NUMINAMATH_CALUDE_exists_x_y_sequences_l1337_133798


namespace NUMINAMATH_CALUDE_increasing_linear_function_k_range_l1337_133723

theorem increasing_linear_function_k_range (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((k + 2) * x₁ + 1) < ((k + 2) * x₂ + 1)) →
  k > -2 :=
by sorry

end NUMINAMATH_CALUDE_increasing_linear_function_k_range_l1337_133723


namespace NUMINAMATH_CALUDE_intersection_angle_proof_l1337_133766

/-- Given two curves in polar coordinates and a ray that intersects both curves, 
    prove that the angle of the ray is π/4 when the product of the distances 
    from the origin to the intersection points is 12. -/
theorem intersection_angle_proof (θ₀ : Real) 
  (h1 : 0 < θ₀) (h2 : θ₀ < Real.pi / 2) : 
  let curve_m := fun (θ : Real) => 4 * Real.cos θ
  let curve_n := fun (ρ θ : Real) => ρ^2 * Real.sin (2 * θ) = 18
  let ray := fun (ρ : Real) => (ρ * Real.cos θ₀, ρ * Real.sin θ₀)
  let point_a := (curve_m θ₀ * Real.cos θ₀, curve_m θ₀ * Real.sin θ₀)
  let point_b := 
    (Real.sqrt (18 / Real.sin (2 * θ₀)) * Real.cos θ₀, 
     Real.sqrt (18 / Real.sin (2 * θ₀)) * Real.sin θ₀)
  (curve_m θ₀ * Real.sqrt (18 / Real.sin (2 * θ₀)) = 12) → 
  θ₀ = Real.pi / 4 := by
sorry


end NUMINAMATH_CALUDE_intersection_angle_proof_l1337_133766


namespace NUMINAMATH_CALUDE_parabola_range_l1337_133719

/-- The parabola y = x^2 + 2x + 4 -/
def parabola (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem parabola_range :
  ∀ a b : ℝ, -2 ≤ a → a < 3 → b = parabola a → 3 ≤ b ∧ b < 19 :=
by sorry

end NUMINAMATH_CALUDE_parabola_range_l1337_133719


namespace NUMINAMATH_CALUDE_initial_marble_difference_l1337_133761

/-- The number of marbles Ed and Doug initially had, and the number Ed currently has -/
structure MarbleCount where
  ed_initial : ℕ
  doug_initial : ℕ
  ed_current : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : MarbleCount) : Prop :=
  m.ed_current = 45 ∧
  m.ed_initial > m.doug_initial ∧
  m.ed_current = m.doug_initial - 11 + 21

/-- The theorem stating the initial difference in marbles -/
theorem initial_marble_difference (m : MarbleCount) 
  (h : marble_problem m) : m.ed_initial - m.doug_initial = 10 := by
  sorry


end NUMINAMATH_CALUDE_initial_marble_difference_l1337_133761


namespace NUMINAMATH_CALUDE_fraction_simplification_l1337_133720

theorem fraction_simplification (x y : ℝ) (h : x^2 ≠ 4*y^2) :
  (-x + 2*y) / (x^2 - 4*y^2) = -1 / (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1337_133720


namespace NUMINAMATH_CALUDE_veterinary_clinic_payment_l1337_133744

/-- Veterinary clinic problem -/
theorem veterinary_clinic_payment
  (dog_charge : ℕ)
  (cat_charge : ℕ)
  (parrot_charge : ℕ)
  (rabbit_charge : ℕ)
  (dogs : ℕ)
  (cats : ℕ)
  (parrots : ℕ)
  (rabbits : ℕ)
  (h1 : dog_charge = 60)
  (h2 : cat_charge = 40)
  (h3 : parrot_charge = 70)
  (h4 : rabbit_charge = 50)
  (h5 : dogs = 25)
  (h6 : cats = 45)
  (h7 : parrots = 15)
  (h8 : rabbits = 10) :
  dog_charge * dogs + cat_charge * cats + parrot_charge * parrots + rabbit_charge * rabbits = 4850 := by
  sorry

end NUMINAMATH_CALUDE_veterinary_clinic_payment_l1337_133744


namespace NUMINAMATH_CALUDE_movie_ticket_price_l1337_133788

/-- The cost of a movie date, given ticket price, combo meal price, and candy price -/
def movie_date_cost (ticket_price : ℚ) (combo_price : ℚ) (candy_price : ℚ) : ℚ :=
  2 * ticket_price + combo_price + 2 * candy_price

/-- Theorem stating that the movie ticket price is $10.00 given the conditions of Connor's date -/
theorem movie_ticket_price :
  ∃ (ticket_price : ℚ),
    movie_date_cost ticket_price 11 2.5 = 36 ∧
    ticket_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_price_l1337_133788


namespace NUMINAMATH_CALUDE_sum_of_f_92_and_neg_92_l1337_133700

/-- Given a polynomial function f(x) = ax^7 + bx^5 - cx^3 + dx + 3 where f(92) = 2,
    prove that f(92) + f(-92) = 6 -/
theorem sum_of_f_92_and_neg_92 (a b c d : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^7 + b * x^5 - c * x^3 + d * x + 3) 
  (h2 : f 92 = 2) : 
  f 92 + f (-92) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_92_and_neg_92_l1337_133700


namespace NUMINAMATH_CALUDE_lavinias_son_older_than_daughter_l1337_133764

def katies_daughter_age : ℕ := 12

def lavinias_daughter_age (k : ℕ) : ℕ :=
  k / 3

def lavinias_son_age (k : ℕ) : ℕ :=
  2 * k

theorem lavinias_son_older_than_daughter :
  lavinias_son_age katies_daughter_age - lavinias_daughter_age katies_daughter_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_lavinias_son_older_than_daughter_l1337_133764


namespace NUMINAMATH_CALUDE_parabola_shift_l1337_133753

/-- Given a parabola with equation y = 3x², prove that after shifting 2 units right and 5 units up, the new equation is y = 3(x-2)² + 5 -/
theorem parabola_shift (x y : ℝ) : 
  (y = 3 * x^2) → 
  (∃ (new_y : ℝ), new_y = 3 * (x - 2)^2 + 5 ∧ 
    new_y = y + 5 ∧ 
    ∀ (new_x : ℝ), new_x = x - 2 → 3 * new_x^2 = 3 * (x - 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l1337_133753


namespace NUMINAMATH_CALUDE_half_equals_fifty_percent_l1337_133796

theorem half_equals_fifty_percent (muffin : ℝ) (h : muffin > 0) :
  (1 / 2 : ℝ) * muffin = (50 / 100 : ℝ) * muffin := by sorry

end NUMINAMATH_CALUDE_half_equals_fifty_percent_l1337_133796


namespace NUMINAMATH_CALUDE_salt_solution_volume_l1337_133710

/-- Proves that the initial volume of a solution is 80 gallons, given the conditions of the problem -/
theorem salt_solution_volume : 
  ∀ (V : ℝ), 
  (0.1 * V = 0.08 * (V + 20)) → 
  V = 80 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_volume_l1337_133710


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1337_133712

theorem quadratic_equation_solution : 
  ∀ y : ℝ, y^2 - 2*y + 1 = -(y - 1)*(y - 3) → y = 1 ∨ y = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1337_133712


namespace NUMINAMATH_CALUDE_wild_ducks_geese_meeting_l1337_133799

/-- The number of days it takes wild ducks to fly from South Sea to North Sea -/
def wild_ducks_days : ℕ := 7

/-- The number of days it takes geese to fly from North Sea to South Sea -/
def geese_days : ℕ := 9

/-- The equation representing the meeting of wild ducks and geese -/
def meeting_equation (x : ℝ) : Prop :=
  (1 / wild_ducks_days : ℝ) * x + (1 / geese_days : ℝ) * x = 1

/-- Theorem stating that the solution to the meeting equation represents
    the number of days it takes for wild ducks and geese to meet -/
theorem wild_ducks_geese_meeting :
  ∃ x : ℝ, x > 0 ∧ meeting_equation x ∧
    ∀ y : ℝ, y > 0 ∧ meeting_equation y → x = y :=
sorry

end NUMINAMATH_CALUDE_wild_ducks_geese_meeting_l1337_133799


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_three_even_three_odd_count_six_digit_numbers_with_three_even_three_odd_l1337_133781

theorem six_digit_numbers_with_three_even_three_odd : ℕ :=
  let first_digit_choices := 9
  let position_choices := Nat.choose 5 2
  let same_parity_fill := 5^2
  let opposite_parity_fill := 2^3
  first_digit_choices * position_choices * same_parity_fill * opposite_parity_fill

theorem count_six_digit_numbers_with_three_even_three_odd :
  six_digit_numbers_with_three_even_three_odd = 90000 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_three_even_three_odd_count_six_digit_numbers_with_three_even_three_odd_l1337_133781


namespace NUMINAMATH_CALUDE_find_divisor_l1337_133741

theorem find_divisor (d : ℕ) (h1 : d > 0) (h2 : 1050 % d = 0) (h3 : 1049 % d ≠ 0) : d = 1050 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1337_133741


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1337_133713

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, mx^2 + 8*m*x + 60 < 0 ↔ -5 < x ∧ x < -3) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1337_133713


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l1337_133715

-- Define the points and lines
def P : ℝ × ℝ := (2, 3)
def A : ℝ × ℝ := (1, 1)
def incident_line (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the reflected ray
def reflected_ray (x y : ℝ) : Prop := 4*x - 5*y + 1 = 0

-- Theorem statement
theorem reflected_ray_equation :
  ∃ (x₀ y₀ : ℝ), 
    incident_line x₀ y₀ ∧  -- The incident ray strikes the line x + y + 1 = 0
    (∃ (t : ℝ), (1 - t) • P.1 + t • x₀ = P.1 ∧ (1 - t) • P.2 + t • y₀ = P.2) ∧  -- The incident ray passes through P
    reflected_ray A.1 A.2  -- The reflected ray passes through A
  → ∀ (x y : ℝ), reflected_ray x y ↔ 4*x - 5*y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l1337_133715


namespace NUMINAMATH_CALUDE_area_perimeter_product_l1337_133729

/-- A square on a grid with vertices at (1,5), (5,5), (5,1), and (1,1) -/
structure GridSquare where
  v1 : (ℕ × ℕ) := (1, 5)
  v2 : (ℕ × ℕ) := (5, 5)
  v3 : (ℕ × ℕ) := (5, 1)
  v4 : (ℕ × ℕ) := (1, 1)

/-- Calculate the side length of the GridSquare -/
def sideLength (s : GridSquare) : ℕ :=
  (s.v2.1 - s.v1.1)

/-- Calculate the area of the GridSquare -/
def area (s : GridSquare) : ℕ :=
  (sideLength s) ^ 2

/-- Calculate the perimeter of the GridSquare -/
def perimeter (s : GridSquare) : ℕ :=
  4 * (sideLength s)

/-- Theorem: The product of the area and perimeter of the GridSquare is 256 -/
theorem area_perimeter_product (s : GridSquare) : 
  area s * perimeter s = 256 := by
  sorry


end NUMINAMATH_CALUDE_area_perimeter_product_l1337_133729


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1337_133703

def A : Set ℝ := {x | x^2 - 4*x - 5 > 0}
def B : Set ℝ := {x | 4 - x^2 > 0}

theorem intersection_of_A_and_B :
  A ∩ B = {x | -2 < x ∧ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1337_133703


namespace NUMINAMATH_CALUDE_trig_calculation_l1337_133759

theorem trig_calculation :
  Real.sin (π / 3) + Real.tan (π / 4) - Real.cos (π / 6) * Real.tan (π / 3) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_calculation_l1337_133759


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l1337_133769

-- Define proposition p
def p : Prop := ∀ x : ℝ, (Real.exp x > 1) → (x > 0)

-- Define proposition q
def q : Prop := ∀ x : ℝ, (|x - 3| > 1) → (x > 4)

-- Theorem to prove
theorem p_or_q_is_true : p ∨ q := by sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l1337_133769


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1337_133731

/-- Simple interest calculation -/
theorem simple_interest_problem (interest_rate : ℚ) (time_period : ℚ) (earned_interest : ℕ) :
  interest_rate = 50 / 3 →
  time_period = 3 / 4 →
  earned_interest = 8625 →
  ∃ (principal : ℕ), 
    principal = 6900000 ∧
    earned_interest = (principal * interest_rate * time_period : ℚ).num / 100 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1337_133731


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_correct_negation_incorrect_disjunction_false_implication_l1337_133752

-- Definition for the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

-- Proposition 1
theorem sufficient_not_necessary : 
  (∃ x : ℝ, x ≠ 1 ∧ quadratic_eq x) ∧ 
  (∀ x : ℝ, x = 1 → quadratic_eq x) := by sorry

-- Proposition 2
theorem contrapositive_correct :
  (∀ x : ℝ, quadratic_eq x → x = 1) ↔ 
  (∀ x : ℝ, x ≠ 1 → ¬(quadratic_eq x)) := by sorry

-- Proposition 3
theorem negation_incorrect :
  ¬(∃ x : ℝ, x > 0 ∧ x^2 + x + 1 < 0) ≠ 
  (∀ x : ℝ, x ≤ 0 → x^2 + x + 1 ≥ 0) := by sorry

-- Proposition 4
theorem disjunction_false_implication :
  ¬(∀ p q : Prop, ¬(p ∨ q) → (¬p ∧ ¬q)) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_correct_negation_incorrect_disjunction_false_implication_l1337_133752


namespace NUMINAMATH_CALUDE_f_1994_4_l1337_133773

def f (x : ℚ) : ℚ := (2 + x) / (2 - 2*x)

def f_n : ℕ → (ℚ → ℚ)
| 0 => id
| (n+1) => f ∘ (f_n n)

theorem f_1994_4 : f_n 1994 4 = 1/4 := by sorry

end NUMINAMATH_CALUDE_f_1994_4_l1337_133773


namespace NUMINAMATH_CALUDE_apartment_expenditure_difference_l1337_133725

def akeno_expenditure : ℕ := 2985
def lev_expenditure : ℕ := akeno_expenditure / 3
def extra_akeno : ℕ := 1172

theorem apartment_expenditure_difference :
  ∃ (ambrocio_expenditure : ℕ),
    ambrocio_expenditure < lev_expenditure ∧
    akeno_expenditure = lev_expenditure + ambrocio_expenditure + extra_akeno ∧
    lev_expenditure - ambrocio_expenditure = 177 :=
by sorry

end NUMINAMATH_CALUDE_apartment_expenditure_difference_l1337_133725


namespace NUMINAMATH_CALUDE_largest_n_value_l1337_133795

/-- A function that checks if for any group of at least 145 candies,
    there is a type of candy which appears exactly 10 times -/
def has_type_with_10_occurrences (candies : List Nat) : Prop :=
  ∀ (group : List Nat), group.length ≥ 145 → group ⊆ candies →
    ∃ (type : Nat), (group.filter (· = type)).length = 10

/-- The theorem stating the largest possible value of n -/
theorem largest_n_value :
  ∀ (n : Nat),
    n > 145 →
    (∀ (candies : List Nat), candies.length = n →
      has_type_with_10_occurrences candies) →
    n ≤ 160 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_value_l1337_133795


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l1337_133726

/-- Given a right triangle with sides 3, 4, and 5, x is the side length of a square
    inscribed with one vertex at the right angle, and y is the side length of a square
    inscribed with one side on the hypotenuse. -/
def triangle_with_squares (x y : ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a^2 + b^2 = c^2 ∧
    a = 3 ∧ b = 4 ∧ c = 5 ∧
    x / 4 = (3 - x) / 3 ∧
    4/3 * y + y + 3/4 * y = 5

theorem inscribed_squares_ratio :
  ∀ x y : ℝ, triangle_with_squares x y → x / y = 37 / 35 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l1337_133726


namespace NUMINAMATH_CALUDE_math_competition_problem_solving_l1337_133711

theorem math_competition_problem_solving (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.85) 
  (h2 : p2 = 0.80) 
  (h3 : p3 = 0.75) : 
  (p1 + p2 + p3 - 2) ≥ 0.40 := by
sorry

end NUMINAMATH_CALUDE_math_competition_problem_solving_l1337_133711


namespace NUMINAMATH_CALUDE_money_division_l1337_133717

/-- Given an amount of money divided between A and B in the ratio 1:2, where A receives $200,
    prove that the total amount to be divided is $600. -/
theorem money_division (a b total : ℕ) : 
  (a : ℚ) / b = 1 / 2 →  -- The ratio of A's share to B's share is 1:2
  a = 200 →              -- A gets $200
  total = a + b →        -- Total is the sum of A's and B's shares
  total = 600 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l1337_133717


namespace NUMINAMATH_CALUDE_misha_notebooks_l1337_133738

theorem misha_notebooks (a b c : ℕ) 
  (h1 : a + 6 = b + c)  -- Vera bought 6 notebooks less than Misha and Vasya together
  (h2 : b + 10 = a + c) -- Vasya bought 10 notebooks less than Vera and Misha together
  : c = 8 := by  -- Misha bought 8 notebooks
  sorry

end NUMINAMATH_CALUDE_misha_notebooks_l1337_133738


namespace NUMINAMATH_CALUDE_difference_is_198_l1337_133701

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  hundreds_lt_10 : hundreds < 10
  tens_lt_10 : tens < 10
  units_lt_10 : units < 10
  hundreds_gt_0 : hundreds > 0

/-- Condition: hundreds digit is 2 more than units digit -/
def hundreds_2_more_than_units (n : ThreeDigitNumber) : Prop :=
  n.hundreds = n.units + 2

/-- The value of the three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed number -/
def reversed (n : ThreeDigitNumber) : Nat :=
  100 * n.units + 10 * n.tens + n.hundreds

/-- The main theorem -/
theorem difference_is_198 (n : ThreeDigitNumber) 
  (h : hundreds_2_more_than_units n) : 
  value n - reversed n = 198 := by
  sorry


end NUMINAMATH_CALUDE_difference_is_198_l1337_133701


namespace NUMINAMATH_CALUDE_exchange_to_hundred_bills_l1337_133750

def twenty_bills : ℕ := 10
def ten_bills : ℕ := 8
def five_bills : ℕ := 4

def total_amount : ℕ := twenty_bills * 20 + ten_bills * 10 + five_bills * 5

theorem exchange_to_hundred_bills :
  (total_amount / 100 : ℕ) = 3 := by sorry

end NUMINAMATH_CALUDE_exchange_to_hundred_bills_l1337_133750


namespace NUMINAMATH_CALUDE_factorization_x4_plus_324_l1337_133705

theorem factorization_x4_plus_324 (x : ℝ) : 
  x^4 + 324 = (x^2 - 18*x + 162) * (x^2 + 18*x + 162) := by sorry

end NUMINAMATH_CALUDE_factorization_x4_plus_324_l1337_133705


namespace NUMINAMATH_CALUDE_solve_for_y_l1337_133751

theorem solve_for_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 25) (h4 : x / y = 36) : y = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1337_133751


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_eight_thirds_l1337_133707

theorem greatest_integer_less_than_negative_eight_thirds :
  Int.floor (-8/3 : ℚ) = -3 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_eight_thirds_l1337_133707
