import Mathlib

namespace NUMINAMATH_CALUDE_probability_log_equals_one_l1477_147784

def set_A : Finset ℕ := {1, 2, 3, 4, 5, 6}
def set_B : Finset ℕ := {1, 2, 3}

def favorable_outcomes : Finset (ℕ × ℕ) := 
  {(2, 1), (4, 2), (6, 3)}

def total_outcomes : ℕ := Finset.card set_A * Finset.card set_B

theorem probability_log_equals_one :
  (Finset.card favorable_outcomes : ℚ) / total_outcomes = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_probability_log_equals_one_l1477_147784


namespace NUMINAMATH_CALUDE_bob_initial_bushels_bob_extra_ears_l1477_147776

/-- Represents the number of ears of corn in a bushel -/
def ears_per_bushel : ℕ := 14

/-- Represents the number of ears of corn Bob has left after giving some away -/
def ears_left : ℕ := 357

/-- Represents the minimum number of full bushels Bob has left -/
def min_bushels_left : ℕ := ears_left / ears_per_bushel

/-- Theorem stating that Bob initially had at least 25 bushels of corn -/
theorem bob_initial_bushels :
  min_bushels_left ≥ 25 := by
  sorry

/-- Theorem stating that Bob has some extra ears that don't make up a full bushel -/
theorem bob_extra_ears :
  ears_left % ears_per_bushel > 0 := by
  sorry

end NUMINAMATH_CALUDE_bob_initial_bushels_bob_extra_ears_l1477_147776


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1477_147765

/-- The eccentricity of a hyperbola with equation x²/4 - y²/12 = 1 is 2 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = 2 ∧ 
  ∀ x y : ℝ, x^2/4 - y^2/12 = 1 → 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 = 4 ∧ b^2 = 12 ∧ c^2 = a^2 + b^2 ∧ e = c/a :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1477_147765


namespace NUMINAMATH_CALUDE_jim_scuba_diving_bags_l1477_147754

/-- The number of smaller bags Jim found while scuba diving -/
def number_of_smaller_bags : ℕ := by sorry

theorem jim_scuba_diving_bags :
  let hours_diving : ℕ := 8
  let coins_per_hour : ℕ := 25
  let treasure_chest_coins : ℕ := 100
  let total_coins := hours_diving * coins_per_hour
  let remaining_coins := total_coins - treasure_chest_coins
  let coins_per_smaller_bag := treasure_chest_coins / 2
  number_of_smaller_bags = remaining_coins / coins_per_smaller_bag :=
by sorry

end NUMINAMATH_CALUDE_jim_scuba_diving_bags_l1477_147754


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1477_147755

theorem fraction_subtraction : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1477_147755


namespace NUMINAMATH_CALUDE_triangle_properties_l1477_147760

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = 1)
  (h2 : 2 * Real.cos t.C - 2 * t.a - t.c = 0) :
  t.B = 2 * Real.pi / 3 ∧ 
  Real.sqrt 3 / 6 = Real.sqrt (((t.a * t.c) / (4 * Real.sin t.A)) ^ 2 - (t.b / 2) ^ 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1477_147760


namespace NUMINAMATH_CALUDE_installation_rate_one_each_possible_solutions_l1477_147740

/-- Represents the number of air conditioners installed by different worker combinations -/
structure InstallationRate where
  skilled : ℕ → ℕ
  new : ℕ → ℕ

/-- Represents the total number of air conditioners to be installed -/
def total_ac : ℕ := 500

/-- Represents the number of days to complete the installation -/
def days : ℕ := 20

/-- Given conditions on installation rates -/
axiom condition1 {r : InstallationRate} : r.skilled 1 + r.new 3 = 11
axiom condition2 {r : InstallationRate} : r.skilled 2 = r.new 5

/-- Theorem stating the installation rate of 1 skilled worker and 1 new worker -/
theorem installation_rate_one_each (r : InstallationRate) : 
  r.skilled 1 + r.new 1 = 7 := by sorry

/-- Theorem stating the possible solutions for m skilled workers and n new workers -/
theorem possible_solutions (m n : ℕ) : 
  (m ≠ 0 ∧ n ≠ 0 ∧ 5 * m + 2 * n = 25) ↔ (m = 1 ∧ n = 10) ∨ (m = 3 ∧ n = 5) := by sorry

end NUMINAMATH_CALUDE_installation_rate_one_each_possible_solutions_l1477_147740


namespace NUMINAMATH_CALUDE_inequalities_solution_l1477_147773

theorem inequalities_solution (x : ℝ) : 
  (2 * (-x + 2) > -3 * x + 5 → x > 1) ∧
  ((7 - x) / 3 ≤ (x + 2) / 2 + 1 → x ≥ 2 / 5) := by
sorry

end NUMINAMATH_CALUDE_inequalities_solution_l1477_147773


namespace NUMINAMATH_CALUDE_complex_modulus_l1477_147722

theorem complex_modulus (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l1477_147722


namespace NUMINAMATH_CALUDE_sum_a4_a5_a6_l1477_147748

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_a4_a5_a6 (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = 2 → a 3 = -10 →
  a 4 + a 5 + a 6 = -66 := by
  sorry

end NUMINAMATH_CALUDE_sum_a4_a5_a6_l1477_147748


namespace NUMINAMATH_CALUDE_sum_of_ages_l1477_147756

theorem sum_of_ages (petra_age mother_age : ℕ) : 
  petra_age = 11 → 
  mother_age = 36 → 
  mother_age = 2 * petra_age + 14 → 
  petra_age + mother_age = 47 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1477_147756


namespace NUMINAMATH_CALUDE_calculation_proof_l1477_147759

theorem calculation_proof : ((4 + 6 + 5) * 2) / 4 - (3 * 2 / 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1477_147759


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1477_147747

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 14) : 
  x^3 + y^3 = 176 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1477_147747


namespace NUMINAMATH_CALUDE_optimal_bus_rental_solution_l1477_147775

/-- Represents a bus rental problem with two types of buses -/
structure BusRental where
  tourists : ℕ
  capacity_A : ℕ
  cost_A : ℕ
  capacity_B : ℕ
  cost_B : ℕ
  max_total_buses : ℕ
  max_B_minus_A : ℕ

/-- Represents a solution to the bus rental problem -/
structure BusRentalSolution where
  buses_A : ℕ
  buses_B : ℕ
  total_cost : ℕ

/-- Check if a solution is valid for a given bus rental problem -/
def is_valid_solution (problem : BusRental) (solution : BusRentalSolution) : Prop :=
  solution.buses_A * problem.capacity_A + solution.buses_B * problem.capacity_B ≥ problem.tourists ∧
  solution.buses_A + solution.buses_B ≤ problem.max_total_buses ∧
  solution.buses_B - solution.buses_A ≤ problem.max_B_minus_A ∧
  solution.total_cost = solution.buses_A * problem.cost_A + solution.buses_B * problem.cost_B

/-- The main theorem stating that the given solution is optimal -/
theorem optimal_bus_rental_solution (problem : BusRental)
  (h_problem : problem = {
    tourists := 900,
    capacity_A := 36,
    cost_A := 1600,
    capacity_B := 60,
    cost_B := 2400,
    max_total_buses := 21,
    max_B_minus_A := 7
  })
  (solution : BusRentalSolution)
  (h_solution : solution = {
    buses_A := 5,
    buses_B := 12,
    total_cost := 36800
  }) :
  is_valid_solution problem solution ∧
  ∀ (other : BusRentalSolution), is_valid_solution problem other → other.total_cost ≥ solution.total_cost :=
by sorry


end NUMINAMATH_CALUDE_optimal_bus_rental_solution_l1477_147775


namespace NUMINAMATH_CALUDE_youtube_likes_problem_l1477_147749

theorem youtube_likes_problem (likes dislikes : ℕ) : 
  dislikes = likes / 2 + 100 →
  dislikes + 1000 = 2600 →
  likes = 3000 := by
sorry

end NUMINAMATH_CALUDE_youtube_likes_problem_l1477_147749


namespace NUMINAMATH_CALUDE_thabos_book_collection_difference_l1477_147701

/-- Theorem: Thabo's Book Collection Difference --/
theorem thabos_book_collection_difference :
  ∀ (paperback_fiction paperback_nonfiction hardcover_nonfiction : ℕ),
  -- Total number of books is 180
  paperback_fiction + paperback_nonfiction + hardcover_nonfiction = 180 →
  -- More paperback nonfiction than hardcover nonfiction
  paperback_nonfiction > hardcover_nonfiction →
  -- Twice as many paperback fiction as paperback nonfiction
  paperback_fiction = 2 * paperback_nonfiction →
  -- 30 hardcover nonfiction books
  hardcover_nonfiction = 30 →
  -- Prove: Difference between paperback nonfiction and hardcover nonfiction is 20
  paperback_nonfiction - hardcover_nonfiction = 20 := by
  sorry

end NUMINAMATH_CALUDE_thabos_book_collection_difference_l1477_147701


namespace NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l1477_147750

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 24 ∣ n^2 ∧ 720 ∣ n^3 → n ≥ 60 :=
by sorry

theorem sixty_satisfies : 24 ∣ 60^2 ∧ 720 ∣ 60^3 :=
by sorry

theorem smallest_n_is_sixty : ∃! n : ℕ, n > 0 ∧ 24 ∣ n^2 ∧ 720 ∣ n^3 ∧ ∀ m : ℕ, (m > 0 ∧ 24 ∣ m^2 ∧ 720 ∣ m^3) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l1477_147750


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1477_147781

/-- An isosceles triangle with side lengths a and b satisfying a certain equation -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  isosceles : True  -- We don't need to specify which sides are equal for this problem
  equation : Real.sqrt (2 * a - 3 * b + 5) + (2 * a + 3 * b - 13)^2 = 0

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.a + t.a + t.b

/-- Theorem stating that the perimeter is either 7 or 8 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) :
  perimeter t = 7 ∨ perimeter t = 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1477_147781


namespace NUMINAMATH_CALUDE_problem_1_l1477_147728

theorem problem_1 (m n : ℕ) (h1 : 3^m = 8) (h2 : 3^n = 2) : 3^(2*m - 3*n + 1) = 24 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1477_147728


namespace NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l1477_147769

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l1477_147769


namespace NUMINAMATH_CALUDE_arctg_sum_quarter_pi_l1477_147791

theorem arctg_sum_quarter_pi (a b : ℝ) : 
  a = (1 : ℝ) / 2 → 
  (a + 1) * (b + 1) = 2 → 
  Real.arctan a + Real.arctan b = π / 4 := by
sorry

end NUMINAMATH_CALUDE_arctg_sum_quarter_pi_l1477_147791


namespace NUMINAMATH_CALUDE_total_cost_theorem_l1477_147734

def sandwich_cost : ℕ := 3
def soda_cost : ℕ := 2
def num_sandwiches : ℕ := 5
def num_sodas : ℕ := 8

theorem total_cost_theorem : 
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l1477_147734


namespace NUMINAMATH_CALUDE_rose_rice_problem_l1477_147763

theorem rose_rice_problem (x : ℚ) : 
  (10000 * (1 - x) * (3/4) = 750) → x = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_rose_rice_problem_l1477_147763


namespace NUMINAMATH_CALUDE_negative_inequality_l1477_147779

theorem negative_inequality (m n : ℝ) (h : m > n) : -m < -n := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l1477_147779


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1477_147735

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_4th : a 4 = 6)
  (h_sum : a 3 + a 5 = a 10) :
  a 12 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1477_147735


namespace NUMINAMATH_CALUDE_hyperbola_condition_l1477_147798

/-- The equation represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop := (3 - k) * (k - 1) < 0

/-- The condition k > 3 -/
def condition (k : ℝ) : Prop := k > 3

theorem hyperbola_condition (k : ℝ) :
  (condition k → is_hyperbola k) ∧ ¬(is_hyperbola k → condition k) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l1477_147798


namespace NUMINAMATH_CALUDE_height_of_d_l1477_147725

theorem height_of_d (h_abc : ℝ) (h_abcd : ℝ) 
  (avg_abc : (h_abc + h_abc + h_abc) / 3 = 130)
  (avg_abcd : (h_abc + h_abc + h_abc + h_abcd) / 4 = 126) :
  h_abcd = 114 := by
  sorry

end NUMINAMATH_CALUDE_height_of_d_l1477_147725


namespace NUMINAMATH_CALUDE_product_inequality_l1477_147700

theorem product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1477_147700


namespace NUMINAMATH_CALUDE_min_value_cube_square_sum_l1477_147715

theorem min_value_cube_square_sum (x y z : ℝ) 
  (h_non_neg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_constraint : 5*x + 16*y + 33*z ≥ 136) : 
  x^3 + y^3 + z^3 + x^2 + y^2 + z^2 ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cube_square_sum_l1477_147715


namespace NUMINAMATH_CALUDE_log_xy_value_l1477_147712

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^2) = 1) (h2 : Real.log (x^2 * y) = 1) :
  Real.log (x * y) = 2/3 := by sorry

end NUMINAMATH_CALUDE_log_xy_value_l1477_147712


namespace NUMINAMATH_CALUDE_years_since_stopped_babysitting_l1477_147792

/-- Represents the age when Jane started babysitting -/
def start_age : ℕ := 18

/-- Represents Jane's current age -/
def current_age : ℕ := 32

/-- Represents the current age of the oldest person Jane could have babysat -/
def oldest_babysat_current_age : ℕ := 23

/-- Represents the maximum age ratio between Jane and the children she babysat -/
def max_age_ratio : ℚ := 1/2

/-- Theorem stating that Jane stopped babysitting 14 years ago -/
theorem years_since_stopped_babysitting :
  current_age - (oldest_babysat_current_age - (start_age * max_age_ratio).floor) = 14 := by
  sorry

end NUMINAMATH_CALUDE_years_since_stopped_babysitting_l1477_147792


namespace NUMINAMATH_CALUDE_initial_average_calculation_l1477_147771

theorem initial_average_calculation (n : ℕ) (wrong_mark correct_mark : ℝ) (corrected_avg : ℝ) :
  n = 30 ∧ wrong_mark = 90 ∧ correct_mark = 15 ∧ corrected_avg = 57.5 →
  (n * corrected_avg + (wrong_mark - correct_mark)) / n = 60 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l1477_147771


namespace NUMINAMATH_CALUDE_bird_sale_ratio_is_half_l1477_147718

/-- Represents the initial counts of animals in the pet store -/
structure InitialCounts where
  birds : ℕ
  puppies : ℕ
  cats : ℕ
  spiders : ℕ

/-- Represents the changes in animal counts -/
structure Changes where
  puppies_adopted : ℕ
  spiders_loose : ℕ

/-- Calculates the ratio of birds sold to initial birds -/
def bird_sale_ratio (initial : InitialCounts) (changes : Changes) (final_count : ℕ) : ℚ :=
  let total_initial := initial.birds + initial.puppies + initial.cats + initial.spiders
  let birds_sold := total_initial - changes.puppies_adopted - changes.spiders_loose - final_count
  birds_sold / initial.birds

/-- Theorem stating the ratio of birds sold to initial birds is 1:2 -/
theorem bird_sale_ratio_is_half 
  (initial : InitialCounts)
  (changes : Changes)
  (final_count : ℕ)
  (h_initial : initial = ⟨12, 9, 5, 15⟩)
  (h_changes : changes = ⟨3, 7⟩)
  (h_final : final_count = 25) :
  bird_sale_ratio initial changes final_count = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bird_sale_ratio_is_half_l1477_147718


namespace NUMINAMATH_CALUDE_michaels_number_l1477_147724

theorem michaels_number (m : ℕ) :
  m % 75 = 0 ∧ m % 40 = 0 ∧ 1000 ≤ m ∧ m ≤ 3000 →
  m = 1800 ∨ m = 2400 ∨ m = 3000 := by
sorry

end NUMINAMATH_CALUDE_michaels_number_l1477_147724


namespace NUMINAMATH_CALUDE_ringbinder_price_decrease_l1477_147703

def original_backpack_price : ℝ := 50
def original_ringbinder_price : ℝ := 20
def backpack_price_increase : ℝ := 5
def num_ringbinders : ℕ := 3
def total_spent : ℝ := 109

theorem ringbinder_price_decrease :
  ∃ (x : ℝ),
    x = 2 ∧
    (original_backpack_price + backpack_price_increase) +
    num_ringbinders * (original_ringbinder_price - x) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_ringbinder_price_decrease_l1477_147703


namespace NUMINAMATH_CALUDE_system_solution_l1477_147706

theorem system_solution (x y : ℚ) :
  (3 * x - 7 * y = 31) ∧ (5 * x + 2 * y = -10) → x = -336/205 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1477_147706


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l1477_147770

/-- For a regular polygon where each exterior angle measures 20 degrees, 
    the sum of the measures of its interior angles is 2880 degrees. -/
theorem sum_interior_angles_regular_polygon : 
  ∀ (n : ℕ), 
    n > 2 → 
    (360 : ℝ) / n = 20 → 
    (n - 2 : ℝ) * 180 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l1477_147770


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1477_147762

-- Problem 1
theorem problem_1 (a : ℝ) (h : a = Real.sqrt 3 - 1) : 
  (a^2 + a) * ((a + 1) / a) = 3 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a = 1/2) : 
  (a + 1) / (a^2 - 1) - (a + 1) / (1 - a) = -5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1477_147762


namespace NUMINAMATH_CALUDE_sophies_shopping_l1477_147738

theorem sophies_shopping (total_budget : ℚ) (trouser_cost : ℚ) (additional_items : ℕ) (additional_item_cost : ℚ) (num_shirts : ℕ) :
  total_budget = 260 →
  trouser_cost = 63 →
  additional_items = 4 →
  additional_item_cost = 40 →
  num_shirts = 2 →
  ∃ (shirt_cost : ℚ), 
    shirt_cost * num_shirts + trouser_cost + (additional_items : ℚ) * additional_item_cost = total_budget ∧
    shirt_cost = 37 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sophies_shopping_l1477_147738


namespace NUMINAMATH_CALUDE_det_special_matrix_l1477_147729

/-- The determinant of the matrix [[1, a, b], [1, a+b, b+c], [1, a, a+c]] is ab + b^2 + bc -/
theorem det_special_matrix (a b c : ℝ) : 
  Matrix.det ![![1, a, b], ![1, a+b, b+c], ![1, a, a+c]] = a*b + b^2 + b*c := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l1477_147729


namespace NUMINAMATH_CALUDE_sector_properties_l1477_147726

-- Define the sector
def Sector (R : ℝ) (α : ℝ) : Prop :=
  R > 0 ∧ α > 0 ∧ (1 / 2) * R^2 * α = 1 ∧ 2 * R + R * α = 4

-- Theorem statement
theorem sector_properties :
  ∃ (R α : ℝ), Sector R α ∧ α = 2 ∧ 2 * Real.sin 1 = 2 * R * Real.sin (α / 2) :=
sorry

end NUMINAMATH_CALUDE_sector_properties_l1477_147726


namespace NUMINAMATH_CALUDE_neg_three_point_fourteen_gt_neg_pi_l1477_147707

theorem neg_three_point_fourteen_gt_neg_pi : -3.14 > -Real.pi := by sorry

end NUMINAMATH_CALUDE_neg_three_point_fourteen_gt_neg_pi_l1477_147707


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_plus_2_l1477_147787

theorem floor_sqrt_50_squared_plus_2 : ⌊Real.sqrt 50⌋^2 + 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_plus_2_l1477_147787


namespace NUMINAMATH_CALUDE_lamplighter_monkey_distance_l1477_147702

/-- Represents the speed and duration of a monkey's movement. -/
structure MonkeyMovement where
  speed : ℝ
  duration : ℝ

/-- Calculates the total distance traveled by a Lamplighter monkey. -/
def totalDistance (running : MonkeyMovement) (swinging : MonkeyMovement) : ℝ :=
  running.speed * running.duration + swinging.speed * swinging.duration

/-- Theorem: A Lamplighter monkey travels 175 feet given the specified conditions. -/
theorem lamplighter_monkey_distance :
  let running : MonkeyMovement := ⟨15, 5⟩
  let swinging : MonkeyMovement := ⟨10, 10⟩
  totalDistance running swinging = 175 := by
  sorry

#eval totalDistance ⟨15, 5⟩ ⟨10, 10⟩

end NUMINAMATH_CALUDE_lamplighter_monkey_distance_l1477_147702


namespace NUMINAMATH_CALUDE_equation_solution_l1477_147783

theorem equation_solution (x : ℚ) : (40 : ℚ) / 60 = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1477_147783


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1477_147716

theorem necessary_but_not_sufficient (a b c d : ℝ) :
  ((a > b ∧ c > d) → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1477_147716


namespace NUMINAMATH_CALUDE_no_extreme_points_implies_a_leq_two_l1477_147789

/-- Given a function f(x) = x - 1/x - a*ln(x), if f has no extreme value points for x > 0,
    then a ≤ 2 --/
theorem no_extreme_points_implies_a_leq_two (a : ℝ) :
  (∀ x > 0, ∃ y > 0, (x - 1/x - a * Real.log x) < (y - 1/y - a * Real.log y) ∨
                     (x - 1/x - a * Real.log x) > (y - 1/y - a * Real.log y)) →
  a ≤ 2 := by
sorry


end NUMINAMATH_CALUDE_no_extreme_points_implies_a_leq_two_l1477_147789


namespace NUMINAMATH_CALUDE_cookies_theorem_l1477_147772

def cookies_problem (initial : ℕ) (first_friend : ℕ) (second_friend : ℕ) (eaten : ℕ) (bought : ℕ) (third_friend : ℕ) : Prop :=
  let remaining_after_first := initial - first_friend
  let remaining_after_second := remaining_after_first - second_friend
  let remaining_after_eating := remaining_after_second - eaten
  let remaining_after_buying := remaining_after_eating + bought
  let final_remaining := remaining_after_buying - third_friend
  final_remaining = 67

theorem cookies_theorem : cookies_problem 120 34 29 20 45 15 := by
  sorry

end NUMINAMATH_CALUDE_cookies_theorem_l1477_147772


namespace NUMINAMATH_CALUDE_sum_of_y_coordinates_is_negative_six_l1477_147766

/-- A circle passes through points (2,0) and (4,0) and is tangent to the line y = x. -/
def CircleThroughPointsAndTangentToLine : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center.1 - 2)^2 + center.2^2 = radius^2 ∧
    (center.1 - 4)^2 + center.2^2 = radius^2 ∧
    (|center.1 - center.2| / Real.sqrt 2) = radius

/-- The sum of all possible y-coordinates of the center of the circle is -6. -/
theorem sum_of_y_coordinates_is_negative_six
  (h : CircleThroughPointsAndTangentToLine) :
  ∃ (y₁ y₂ : ℝ), y₁ + y₂ = -6 ∧
    ∀ (center : ℝ × ℝ) (radius : ℝ),
      (center.1 - 2)^2 + center.2^2 = radius^2 →
      (center.1 - 4)^2 + center.2^2 = radius^2 →
      (|center.1 - center.2| / Real.sqrt 2) = radius →
      center.2 = y₁ ∨ center.2 = y₂ :=
sorry

end NUMINAMATH_CALUDE_sum_of_y_coordinates_is_negative_six_l1477_147766


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1477_147782

/-- Given a right triangle with sides a and b, hypotenuse c, and a perpendicular from
    the right angle vertex dividing c into segments r and s, prove that if a : b = 2 : 3,
    then r : s = 4 : 9. -/
theorem right_triangle_segment_ratio (a b c r s : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : r > 0) (h5 : s > 0) (h6 : a^2 + b^2 = c^2) (h7 : r + s = c) (h8 : r * c = a^2)
    (h9 : s * c = b^2) (h10 : a / b = 2 / 3) : r / s = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1477_147782


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1477_147717

/-- Given a line L1 defined by 2x + 3y = 9, and another line L2 that is perpendicular to L1
    with a y-intercept of -4, the x-intercept of L2 is 8/3. -/
theorem perpendicular_line_x_intercept :
  ∀ (L1 L2 : Set (ℝ × ℝ)),
  (∀ x y, (x, y) ∈ L1 ↔ 2 * x + 3 * y = 9) →
  (∃ m : ℝ, ∀ x y, (x, y) ∈ L2 ↔ y = m * x - 4) →
  (∀ x y₁ y₂, (x, y₁) ∈ L1 ∧ (x, y₂) ∈ L2 → (y₁ - y₂) * (x - 0) = -(1 : ℝ)) →
  (∃ x : ℝ, (x, 0) ∈ L2 ∧ x = 8 / 3) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1477_147717


namespace NUMINAMATH_CALUDE_complex_square_root_expression_l1477_147720

theorem complex_square_root_expression : 
  (2 * Real.sqrt 12 - 4 * Real.sqrt 27 + 3 * Real.sqrt 75 + 7 * Real.sqrt 8 - 3 * Real.sqrt 18) * 
  (4 * Real.sqrt 48 - 3 * Real.sqrt 27 - 5 * Real.sqrt 18 + 2 * Real.sqrt 50) = 97 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_expression_l1477_147720


namespace NUMINAMATH_CALUDE_chicken_surprise_weight_theorem_l1477_147744

/-- The weight of one serving of Chicken Surprise -/
def chicken_surprise_serving_weight (total_servings : ℕ) (chicken_weight_pounds : ℚ) (stuffing_weight_ounces : ℕ) : ℚ :=
  (chicken_weight_pounds * 16 + stuffing_weight_ounces) / total_servings

/-- Theorem: Given 12 servings of Chicken Surprise, 4.5 pounds of chicken, and 24 ounces of stuffing, one serving of Chicken Surprise is 8 ounces. -/
theorem chicken_surprise_weight_theorem :
  chicken_surprise_serving_weight 12 (9/2) 24 = 8 := by
  sorry

end NUMINAMATH_CALUDE_chicken_surprise_weight_theorem_l1477_147744


namespace NUMINAMATH_CALUDE_math_competition_schools_l1477_147721

/- Define the problem setup -/
structure MathCompetition where
  num_students_per_school : ℕ
  andrea_rank : ℕ
  beth_rank : ℕ
  carla_rank : ℕ

/- Define the conditions -/
def valid_competition (comp : MathCompetition) : Prop :=
  comp.num_students_per_school = 4 ∧
  comp.andrea_rank < comp.beth_rank ∧
  comp.andrea_rank < comp.carla_rank ∧
  comp.beth_rank = 48 ∧
  comp.carla_rank = 75

/- Define Andrea's rank as the median -/
def andrea_is_median (comp : MathCompetition) (total_students : ℕ) : Prop :=
  comp.andrea_rank = (total_students + 1) / 2 ∨
  comp.andrea_rank = (total_students + 2) / 2

/- Theorem statement -/
theorem math_competition_schools (comp : MathCompetition) :
  valid_competition comp →
  ∃ (total_students : ℕ),
    andrea_is_median comp total_students ∧
    total_students % comp.num_students_per_school = 0 ∧
    total_students / comp.num_students_per_school = 23 :=
sorry

end NUMINAMATH_CALUDE_math_competition_schools_l1477_147721


namespace NUMINAMATH_CALUDE_linear_system_solution_l1477_147723

theorem linear_system_solution (m : ℕ) (x y : ℝ) : 
  (2 * x - y = 4 * m - 5) →
  (x + 4 * y = -7 * m + 2) →
  (x + y > -3) →
  (m = 0 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1477_147723


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1477_147788

theorem inequality_equivalence (x : Real) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (2 * Real.cos x ≤ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ∧
   Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ≤ Real.sqrt 2) ↔
  x ∈ Set.Icc (Real.pi / 4) (7 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1477_147788


namespace NUMINAMATH_CALUDE_swap_values_l1477_147753

/-- Swaps the values of two variables using an intermediate variable -/
theorem swap_values (a b : ℕ) : 
  let a_init := a
  let b_init := b
  let c := a_init
  let a_new := b_init
  let b_new := c
  (a_new = b_init ∧ b_new = a_init) := by sorry

end NUMINAMATH_CALUDE_swap_values_l1477_147753


namespace NUMINAMATH_CALUDE_a_nine_equals_a_three_times_a_seven_l1477_147768

def exponential_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = q ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem a_nine_equals_a_three_times_a_seven
  (a : ℕ → ℝ) (q : ℝ) (h : exponential_sequence a q) :
  a 9 = a 3 * a 7 := by
  sorry

end NUMINAMATH_CALUDE_a_nine_equals_a_three_times_a_seven_l1477_147768


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l1477_147733

/-- Given two parallel vectors a and b in R², prove that their linear combination results in (14, 7) -/
theorem parallel_vectors_sum (m : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![m, 2]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  (3 • a + 2 • b : Fin 2 → ℝ) = ![14, 7] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l1477_147733


namespace NUMINAMATH_CALUDE_color_film_fraction_l1477_147796

theorem color_film_fraction (x y : ℝ) (x_pos : x > 0) (y_pos : y > 0) : 
  let total_bw := 20 * x
  let total_color := 6 * y
  let selected_bw := y / (5 * x) * total_bw
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  selected_color / total_selected = 30 / 31 := by
sorry


end NUMINAMATH_CALUDE_color_film_fraction_l1477_147796


namespace NUMINAMATH_CALUDE_discount_rate_pony_jeans_discount_rate_pony_jeans_proof_l1477_147799

theorem discount_rate_pony_jeans : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (fox_price pony_price total_savings discount_sum : ℝ) =>
    let fox_pairs : ℝ := 3
    let pony_pairs : ℝ := 2
    let total_pairs : ℝ := fox_pairs + pony_pairs
    fox_price = 15 ∧ 
    pony_price = 18 ∧ 
    total_savings = 9 ∧ 
    discount_sum = 22 ∧
    ∃ (fox_discount pony_discount : ℝ),
      fox_discount + pony_discount = discount_sum ∧
      fox_pairs * (fox_discount / 100 * fox_price) + 
        pony_pairs * (pony_discount / 100 * pony_price) = total_savings ∧
      pony_discount = 10

theorem discount_rate_pony_jeans_proof 
  (fox_price pony_price total_savings discount_sum : ℝ) :
  discount_rate_pony_jeans fox_price pony_price total_savings discount_sum :=
by sorry

end NUMINAMATH_CALUDE_discount_rate_pony_jeans_discount_rate_pony_jeans_proof_l1477_147799


namespace NUMINAMATH_CALUDE_reciprocal_problems_l1477_147737

theorem reciprocal_problems :
  (1 / 1.5 = 2/3) ∧ (1 / 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problems_l1477_147737


namespace NUMINAMATH_CALUDE_six_cube_forming_configurations_l1477_147704

/-- Represents a position where an additional square can be attached -/
inductive AttachmentPosition
| TopLeft | TopCenter | TopRight
| MiddleLeft | MiddleRight
| BottomLeft | BottomCenter | BottomRight
| LeftCenter | RightCenter

/-- Represents the cross-shaped arrangement of squares -/
structure CrossArrangement :=
  (center : Square)
  (top : Square)
  (right : Square)
  (bottom : Square)
  (left : Square)

/-- Represents a configuration with an additional square attached -/
structure Configuration :=
  (base : CrossArrangement)
  (attachment : AttachmentPosition)

/-- Predicate to check if a configuration can form a cube with one face missing -/
def can_form_cube (config : Configuration) : Prop :=
  sorry

/-- The main theorem stating that exactly 6 configurations can form a cube -/
theorem six_cube_forming_configurations :
  ∃ (valid_configs : Finset Configuration),
    (∀ c ∈ valid_configs, can_form_cube c) ∧
    (∀ c : Configuration, can_form_cube c → c ∈ valid_configs) ∧
    valid_configs.card = 6 :=
  sorry

end NUMINAMATH_CALUDE_six_cube_forming_configurations_l1477_147704


namespace NUMINAMATH_CALUDE_stating_initial_amount_is_200_l1477_147764

/-- Represents the exchange rate from U.S. dollars to Canadian dollars -/
def exchange_rate : ℚ := 6 / 5

/-- Represents the amount spent in Canadian dollars -/
def amount_spent : ℚ := 80

/-- 
Given an initial amount of U.S. dollars, calculates the remaining amount 
of Canadian dollars after exchanging and spending
-/
def remaining_amount (d : ℚ) : ℚ := (4 / 5) * d

/-- 
Theorem stating that given the exchange rate and spending conditions, 
the initial amount of U.S. dollars is 200
-/
theorem initial_amount_is_200 : 
  ∃ d : ℚ, d = 200 ∧ 
  exchange_rate * d - amount_spent = remaining_amount d :=
sorry

end NUMINAMATH_CALUDE_stating_initial_amount_is_200_l1477_147764


namespace NUMINAMATH_CALUDE_eighth_root_unity_l1477_147757

theorem eighth_root_unity : ∃ n : ℕ, n ∈ Finset.range 8 ∧
  (Complex.I + Complex.tan (π / 8)) / (Complex.tan (π / 8) - Complex.I) =
  Complex.exp (2 * n * π * Complex.I / 8) := by
  sorry

end NUMINAMATH_CALUDE_eighth_root_unity_l1477_147757


namespace NUMINAMATH_CALUDE_completing_square_transformation_l1477_147793

theorem completing_square_transformation (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l1477_147793


namespace NUMINAMATH_CALUDE_expression_reduction_l1477_147795

theorem expression_reduction (a b c : ℝ) 
  (h1 : a^2 + c^2 - b^2 - 2*a*c ≠ 0)
  (h2 : a - b + c ≠ 0)
  (h3 : a - c + b ≠ 0) :
  (a^2 + b^2 - c^2 - 2*a*b) / (a^2 + c^2 - b^2 - 2*a*c) = 
  ((a - b + c) * (a - b - c)) / ((a - c + b) * (a - c - b)) := by
  sorry

end NUMINAMATH_CALUDE_expression_reduction_l1477_147795


namespace NUMINAMATH_CALUDE_parametric_eq_represents_line_l1477_147778

/-- Prove that the given parametric equations represent the line x + y - 2 = 0 --/
theorem parametric_eq_represents_line :
  ∀ (t : ℝ), (3 + t) + (1 - t) - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parametric_eq_represents_line_l1477_147778


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l1477_147727

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

theorem collinear_points_k_value :
  ∀ k : ℚ,
  let p1 : Point := ⟨2, -1⟩
  let p2 : Point := ⟨10, k⟩
  let p3 : Point := ⟨23, 4⟩
  collinear p1 p2 p3 → k = 19 / 21 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l1477_147727


namespace NUMINAMATH_CALUDE_special_sequence_2023_l1477_147780

/-- A sequence of positive terms with a special property -/
structure SpecialSequence where
  a : ℕ → ℕ+
  S : ℕ → ℕ
  property : ∀ n, 2 * S n = (a n).val * ((a n).val + 1)

/-- The 2023rd term of a special sequence is 2023 -/
theorem special_sequence_2023 (seq : SpecialSequence) : seq.a 2023 = ⟨2023, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_2023_l1477_147780


namespace NUMINAMATH_CALUDE_pencil_cost_is_13_l1477_147743

/-- Represents the data for the pencil purchase problem -/
structure PencilPurchaseData where
  total_students : ℕ
  buyers : ℕ
  total_cost : ℕ
  pencil_cost : ℕ
  pencils_per_student : ℕ

/-- The conditions of the pencil purchase problem -/
def pencil_purchase_conditions (data : PencilPurchaseData) : Prop :=
  data.total_students = 50 ∧
  data.buyers > data.total_students / 2 ∧
  data.pencil_cost > data.pencils_per_student ∧
  data.buyers * data.pencil_cost * data.pencils_per_student = data.total_cost ∧
  data.total_cost = 2275

/-- The theorem stating that under the given conditions, the pencil cost is 13 cents -/
theorem pencil_cost_is_13 (data : PencilPurchaseData) :
  pencil_purchase_conditions data → data.pencil_cost = 13 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_is_13_l1477_147743


namespace NUMINAMATH_CALUDE_suv_highway_efficiency_l1477_147711

/-- Represents the fuel efficiency of an SUV -/
structure SUVFuelEfficiency where
  city_mpg : ℝ
  highway_mpg : ℝ
  max_distance : ℝ
  tank_capacity : ℝ

/-- Theorem stating the highway fuel efficiency of the SUV -/
theorem suv_highway_efficiency (suv : SUVFuelEfficiency)
  (h1 : suv.city_mpg = 7.6)
  (h2 : suv.max_distance = 268.4)
  (h3 : suv.tank_capacity = 22) :
  suv.highway_mpg = 12.2 := by
  sorry

#check suv_highway_efficiency

end NUMINAMATH_CALUDE_suv_highway_efficiency_l1477_147711


namespace NUMINAMATH_CALUDE_monty_hall_probabilities_l1477_147774

/-- Represents the three doors in the Monty Hall problem -/
inductive Door : Type
  | door1 : Door
  | door2 : Door
  | door3 : Door

/-- Represents the possible contents behind a door -/
inductive Content : Type
  | car : Content
  | goat : Content

/-- The Monty Hall game setup -/
structure MontyHallGame where
  prize_door : Door
  initial_choice : Door
  opened_door : Door
  h_prize_not_opened : opened_door ≠ prize_door
  h_opened_is_goat : opened_door ≠ initial_choice

/-- The probability of winning by sticking with the initial choice -/
def prob_stick_wins (game : MontyHallGame) : ℚ :=
  1 / 3

/-- The probability of winning by switching doors -/
def prob_switch_wins (game : MontyHallGame) : ℚ :=
  2 / 3

theorem monty_hall_probabilities (game : MontyHallGame) :
  prob_stick_wins game = 1 / 3 ∧ prob_switch_wins game = 2 / 3 := by
  sorry

#check monty_hall_probabilities

end NUMINAMATH_CALUDE_monty_hall_probabilities_l1477_147774


namespace NUMINAMATH_CALUDE_inequalities_proof_l1477_147713

theorem inequalities_proof (a b : ℝ) (h1 : a > b) (h2 : b ≥ 2) : 
  (b^2 > 3*b - a) ∧ (a*b > a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1477_147713


namespace NUMINAMATH_CALUDE_distribute_4_balls_3_boxes_l1477_147752

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 15 ways to distribute 4 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_4_balls_3_boxes : distribute_balls 4 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_distribute_4_balls_3_boxes_l1477_147752


namespace NUMINAMATH_CALUDE_f_upper_bound_f_max_value_condition_l1477_147719

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3 - abs (2 * x - 1)

-- Part 1
theorem f_upper_bound :
  ∀ x : ℝ, (x ≤ 0 ∨ x ≥ 2) → f 1 x ≤ 2 :=
sorry

-- Part 2
theorem f_max_value_condition :
  ∀ a : ℝ, (∃ x : ℝ, ∀ y : ℝ, f a y ≤ f a x) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_f_upper_bound_f_max_value_condition_l1477_147719


namespace NUMINAMATH_CALUDE_total_spider_legs_l1477_147797

/-- The number of spiders in the room -/
def num_spiders : ℕ := 5

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs in the room is 40 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_l1477_147797


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1477_147732

theorem abs_neg_three_eq_three : abs (-3 : ℤ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1477_147732


namespace NUMINAMATH_CALUDE_laptop_final_price_l1477_147714

/-- Calculate the final price of a laptop given the original price, discount rate, tax rate, and commission rate. -/
def calculate_final_price (original_price discount_rate tax_rate commission_rate : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let price_after_tax := discounted_price * (1 + tax_rate)
  price_after_tax * (1 + commission_rate)

/-- Theorem stating that the final price of the laptop is 1199.52 dollars given the specified conditions. -/
theorem laptop_final_price :
  calculate_final_price 1200 0.15 0.12 0.05 = 1199.52 := by
  sorry

end NUMINAMATH_CALUDE_laptop_final_price_l1477_147714


namespace NUMINAMATH_CALUDE_verify_conditions_max_boxes_A_l1477_147736

/-- Represents the price of a box of paint model A in yuan -/
def price_A : ℕ := 24

/-- Represents the price of a box of paint model B in yuan -/
def price_B : ℕ := 16

/-- Represents the total number of boxes to be purchased -/
def total_boxes : ℕ := 200

/-- Represents the maximum total cost in yuan -/
def max_cost : ℕ := 3920

/-- Verification of the given conditions -/
theorem verify_conditions : 
  price_A + 2 * price_B = 56 ∧ 
  2 * price_A + price_B = 64 := by sorry

/-- Theorem stating the maximum number of boxes of paint A that can be purchased -/
theorem max_boxes_A : 
  (∀ m : ℕ, m ≤ total_boxes → 
    m * price_A + (total_boxes - m) * price_B ≤ max_cost → 
    m ≤ 90) ∧ 
  90 * price_A + (total_boxes - 90) * price_B ≤ max_cost := by sorry

end NUMINAMATH_CALUDE_verify_conditions_max_boxes_A_l1477_147736


namespace NUMINAMATH_CALUDE_gcd_4557_1953_5115_l1477_147731

theorem gcd_4557_1953_5115 : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4557_1953_5115_l1477_147731


namespace NUMINAMATH_CALUDE_popsicle_stick_difference_l1477_147745

theorem popsicle_stick_difference :
  let num_boys : ℕ := 10
  let num_girls : ℕ := 12
  let sticks_per_boy : ℕ := 15
  let sticks_per_girl : ℕ := 12
  let total_boys_sticks : ℕ := num_boys * sticks_per_boy
  let total_girls_sticks : ℕ := num_girls * sticks_per_girl
  total_boys_sticks - total_girls_sticks = 6 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_stick_difference_l1477_147745


namespace NUMINAMATH_CALUDE_rebecca_eggs_l1477_147777

theorem rebecca_eggs (num_groups : ℕ) (eggs_per_group : ℕ) 
  (h1 : num_groups = 11) (h2 : eggs_per_group = 2) : 
  num_groups * eggs_per_group = 22 := by
sorry

end NUMINAMATH_CALUDE_rebecca_eggs_l1477_147777


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1477_147751

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ≠ 0 ∧ a.1 * t = b.1 ∧ a.2 * t = b.2

/-- The problem statement -/
theorem parallel_vectors_k_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (6, k)
  are_parallel a b → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1477_147751


namespace NUMINAMATH_CALUDE_speed_difference_l1477_147730

/-- Given distances and times for cycling and walking, prove the speed difference --/
theorem speed_difference (school_distance : ℝ) (cycle_time : ℝ) 
  (park_distance : ℝ) (walk_time : ℝ) 
  (h1 : school_distance = 9.3) 
  (h2 : cycle_time = 0.6)
  (h3 : park_distance = 0.9)
  (h4 : walk_time = 0.2) :
  (school_distance / cycle_time) - (park_distance / walk_time) = 11 := by
  sorry


end NUMINAMATH_CALUDE_speed_difference_l1477_147730


namespace NUMINAMATH_CALUDE_fiona_probability_l1477_147758

/-- Represents a lily pad with its number and whether it contains a predator or food -/
structure LilyPad where
  number : Nat
  hasPredator : Bool
  hasFood : Bool

/-- Represents the possible moves Fiona can make -/
inductive Move
  | Forward
  | ForwardTwo
  | Backward

/-- Represents Fiona's current position and the probability of reaching that position -/
structure FionaState where
  position : Nat
  probability : Rat

/-- The probability of each move -/
def moveProbability : Rat := 1 / 3

/-- The total number of lily pads -/
def totalPads : Nat := 15

/-- Creates the initial state of the lily pads -/
def initLilyPads : List LilyPad := sorry

/-- Checks if a move is valid given Fiona's current position -/
def isValidMove (currentPos : Nat) (move : Move) : Bool := sorry

/-- Calculates Fiona's new position after a move -/
def newPosition (currentPos : Nat) (move : Move) : Nat := sorry

/-- Calculates the probability of Fiona reaching pad 13 without landing on pads 4 or 8 -/
def probReachPad13 (initialState : FionaState) (lilyPads : List LilyPad) : Rat := sorry

theorem fiona_probability :
  probReachPad13 ⟨0, 1⟩ initLilyPads = 16 / 177147 := by sorry

end NUMINAMATH_CALUDE_fiona_probability_l1477_147758


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1477_147742

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋ ↔ 4/3 ≤ x ∧ x < 7/3 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1477_147742


namespace NUMINAMATH_CALUDE_x_varies_linearly_with_z_l1477_147709

/-- Given that x varies as the cube of y and y varies as the cube root of z,
    prove that x varies linearly with z. -/
theorem x_varies_linearly_with_z 
  (k : ℝ) (j : ℝ) (x y z : ℝ → ℝ) 
  (h1 : ∀ t, x t = k * (y t)^3) 
  (h2 : ∀ t, y t = j * (z t)^(1/3)) :
  ∃ m : ℝ, ∀ t, x t = m * z t :=
sorry

end NUMINAMATH_CALUDE_x_varies_linearly_with_z_l1477_147709


namespace NUMINAMATH_CALUDE_gcd_difference_perfect_square_l1477_147785

theorem gcd_difference_perfect_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = k ^ 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_difference_perfect_square_l1477_147785


namespace NUMINAMATH_CALUDE_cos_rational_angle_irrational_l1477_147705

open Real

theorem cos_rational_angle_irrational (p q : ℤ) (h : q ≠ 0) :
  let x := cos (p / q * π)
  x ≠ 0 ∧ x ≠ 1/2 ∧ x ≠ -1/2 ∧ x ≠ 1 ∧ x ≠ -1 → Irrational x :=
by sorry

end NUMINAMATH_CALUDE_cos_rational_angle_irrational_l1477_147705


namespace NUMINAMATH_CALUDE_unique_solution_l1477_147739

-- Define the color type
inductive Color
| Red
| Blue

-- Define the clothing type
structure Clothing where
  tshirt : Color
  shorts : Color

-- Define the children
structure Children where
  alyna : Clothing
  bohdan : Clothing
  vika : Clothing
  grysha : Clothing

-- Define the conditions
def satisfiesConditions (c : Children) : Prop :=
  c.alyna.tshirt = Color.Red ∧
  c.bohdan.tshirt = Color.Red ∧
  c.alyna.shorts ≠ c.bohdan.shorts ∧
  c.vika.tshirt ≠ c.grysha.tshirt ∧
  c.vika.shorts = Color.Blue ∧
  c.grysha.shorts = Color.Blue ∧
  c.alyna.tshirt ≠ c.vika.tshirt ∧
  c.alyna.shorts ≠ c.vika.shorts

-- Define the correct answer
def correctAnswer : Children :=
  { alyna := { tshirt := Color.Red, shorts := Color.Red }
  , bohdan := { tshirt := Color.Red, shorts := Color.Blue }
  , vika := { tshirt := Color.Blue, shorts := Color.Blue }
  , grysha := { tshirt := Color.Red, shorts := Color.Blue }
  }

-- Theorem statement
theorem unique_solution :
  ∀ c : Children, satisfiesConditions c → c = correctAnswer := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1477_147739


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_theorem_l1477_147767

theorem right_triangle_acute_angle_theorem :
  ∀ (x y : ℝ),
  x > 0 ∧ y > 0 →
  x + y = 90 →
  y = 5 * x →
  y = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_theorem_l1477_147767


namespace NUMINAMATH_CALUDE_cleo_utility_equality_l1477_147741

/-- Utility function for Cleo's activities -/
def utility (reading : ℝ) (painting : ℝ) : ℝ := reading * painting

/-- Time spent painting on Saturday -/
def saturday_painting (t : ℝ) : ℝ := t

/-- Time spent reading on Saturday -/
def saturday_reading (t : ℝ) : ℝ := 10 - 2 * t

/-- Time spent painting on Sunday -/
def sunday_painting (t : ℝ) : ℝ := 5 - t

/-- Time spent reading on Sunday -/
def sunday_reading (t : ℝ) : ℝ := 2 * t + 4

theorem cleo_utility_equality :
  ∃ t : ℝ, utility (saturday_reading t) (saturday_painting t) = utility (sunday_reading t) (sunday_painting t) ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_cleo_utility_equality_l1477_147741


namespace NUMINAMATH_CALUDE_sphere_radius_is_60_37_l1477_147746

/-- A triangular pyramid with perpendicular lateral edges and a sphere touching all lateral faces -/
structure PerpendicularPyramid where
  /-- The side lengths of the triangular base -/
  base_side_1 : ℝ
  base_side_2 : ℝ
  base_side_3 : ℝ
  /-- The radius of the sphere touching all lateral faces -/
  sphere_radius : ℝ
  /-- The lateral edges are pairwise perpendicular -/
  lateral_edges_perpendicular : True
  /-- The center of the sphere lies on the base -/
  sphere_center_on_base : True
  /-- The base side lengths satisfy the given conditions -/
  base_side_1_sq : base_side_1^2 = 61
  base_side_2_sq : base_side_2^2 = 52
  base_side_3_sq : base_side_3^2 = 41

/-- The theorem stating that the radius of the sphere is 60/37 -/
theorem sphere_radius_is_60_37 (p : PerpendicularPyramid) : p.sphere_radius = 60 / 37 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_is_60_37_l1477_147746


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l1477_147794

/-- Represents the number of distinct balls -/
def num_balls : ℕ := 4

/-- Represents the number of distinct boxes -/
def num_boxes : ℕ := 4

/-- Calculates the number of ways to place all balls into boxes leaving exactly one box empty -/
def ways_one_empty : ℕ := sorry

/-- Calculates the number of ways to place all balls into boxes with exactly one box containing two balls -/
def ways_one_two_balls : ℕ := sorry

/-- Calculates the number of ways to place all balls into boxes leaving exactly two boxes empty -/
def ways_two_empty : ℕ := sorry

theorem ball_placement_theorem :
  ways_one_empty = 144 ∧
  ways_one_two_balls = 144 ∧
  ways_two_empty = 84 := by sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l1477_147794


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_neg_five_l1477_147710

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of -5 is 5 -/
theorem opposite_of_neg_five : opposite (-5 : ℝ) = 5 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_neg_five_l1477_147710


namespace NUMINAMATH_CALUDE_convex_polygon_contains_half_homothety_l1477_147786

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  convex : Convex ℝ (convexHull ℝ vertices)

/-- A homothety transformation -/
def homothety (center : ℝ × ℝ) (k : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (center.1 + k * (p.1 - center.1), center.2 + k * (p.2 - center.2))

/-- The theorem stating that a convex polygon contains its image under a 1/2 homothety -/
theorem convex_polygon_contains_half_homothety (P : ConvexPolygon) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ P.vertices →
    homothety center (1/2) p ∈ convexHull ℝ P.vertices :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_contains_half_homothety_l1477_147786


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l1477_147790

/-- The probability of selecting two non-defective pens from a box of 9 pens with 3 defective pens -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 9)
  (h2 : defective_pens = 3)
  (h3 : defective_pens < total_pens) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l1477_147790


namespace NUMINAMATH_CALUDE_parallelogram_area_l1477_147761

theorem parallelogram_area (base height : ℝ) (h1 : base = 14) (h2 : height = 24) :
  base * height = 336 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1477_147761


namespace NUMINAMATH_CALUDE_pages_read_day5_l1477_147708

def pages_day1 : ℕ := 63
def pages_day2 : ℕ := 95 -- Rounded up from 94.5
def pages_day3 : ℕ := pages_day2 + 20
def pages_day4 : ℕ := 86 -- Rounded down from 86.25
def total_pages : ℕ := 480

theorem pages_read_day5 : 
  total_pages - (pages_day1 + pages_day2 + pages_day3 + pages_day4) = 121 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_day5_l1477_147708
