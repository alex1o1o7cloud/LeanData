import Mathlib

namespace NUMINAMATH_CALUDE_clubsuit_symmetry_forms_intersecting_lines_l3934_393457

-- Define the operation ♣
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Theorem statement
theorem clubsuit_symmetry_forms_intersecting_lines :
  ∃ (l₁ l₂ : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), clubsuit x y = clubsuit y x ↔ (x, y) ∈ l₁ ∪ l₂) ∧
    (l₁ ≠ l₂) ∧
    (∃ (p : ℝ × ℝ), p ∈ l₁ ∧ p ∈ l₂) :=
sorry


end NUMINAMATH_CALUDE_clubsuit_symmetry_forms_intersecting_lines_l3934_393457


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3934_393465

/-- Given two quadratic equations, if the roots of the first are each three less than
    the roots of the second, then the constant term of the first equation is 24/5. -/
theorem quadratic_root_relation (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ ∃ y, 5*y^2 - 4*y - 9 = 0 ∧ x = y - 3) →
  c = 24/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3934_393465


namespace NUMINAMATH_CALUDE_inequality_proof_l3934_393404

theorem inequality_proof (x : ℝ) (h1 : (3/2 : ℝ) ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3934_393404


namespace NUMINAMATH_CALUDE_janes_change_l3934_393405

/-- The change Jane receives when buying an apple -/
theorem janes_change (apple_price : ℚ) (paid_amount : ℚ) (change : ℚ) : 
  apple_price = 0.75 → paid_amount = 5 → change = paid_amount - apple_price → change = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_janes_change_l3934_393405


namespace NUMINAMATH_CALUDE_george_blocks_count_l3934_393434

/-- Calculates the total number of blocks given the number of large boxes, small boxes per large box,
    blocks per small box, and individual blocks outside the boxes. -/
def totalBlocks (largBoxes smallBoxesPerLarge blocksPerSmall individualBlocks : ℕ) : ℕ :=
  largBoxes * smallBoxesPerLarge * blocksPerSmall + individualBlocks

/-- Proves that George has 366 blocks in total -/
theorem george_blocks_count :
  totalBlocks 5 8 9 6 = 366 := by
  sorry

end NUMINAMATH_CALUDE_george_blocks_count_l3934_393434


namespace NUMINAMATH_CALUDE_max_value_implies_t_equals_one_l3934_393453

theorem max_value_implies_t_equals_one (t : ℝ) :
  (∀ x ∈ Set.Icc 0 3, |x^2 - 2*x - t| ≤ 2) →
  (∃ x ∈ Set.Icc 0 3, |x^2 - 2*x - t| = 2) →
  t = 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_t_equals_one_l3934_393453


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l3934_393447

theorem simplify_trigonometric_expression :
  let x := Real.sin (15 * π / 180)
  let y := Real.cos (15 * π / 180)
  Real.sqrt (x^4 + 4 * y^2) - Real.sqrt (y^4 + 4 * x^2) = (1 / 2) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l3934_393447


namespace NUMINAMATH_CALUDE_f_of_5_equals_22_l3934_393475

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 2

-- State the theorem
theorem f_of_5_equals_22 : f 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_equals_22_l3934_393475


namespace NUMINAMATH_CALUDE_height_of_column_G_l3934_393424

-- Define the regular octagon vertices
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (0, -8)
def D : ℝ × ℝ := (-8, 0)
def G : ℝ × ℝ := (0, 8)

-- Define the heights of columns A, B, C, D
def height_A : ℝ := 15
def height_B : ℝ := 12
def height_C : ℝ := 14
def height_D : ℝ := 13

-- Theorem statement
theorem height_of_column_G : 
  ∃ (height_G : ℝ), height_G = 15.5 :=
by
  sorry

end NUMINAMATH_CALUDE_height_of_column_G_l3934_393424


namespace NUMINAMATH_CALUDE_multiples_difference_squared_l3934_393497

def a : ℕ := (Finset.filter (λ x => x % 7 = 0) (Finset.range 60)).card

def b : ℕ := (Finset.filter (λ x => x % 3 = 0 ∨ x % 7 = 0) (Finset.range 60)).card

theorem multiples_difference_squared : (a - b)^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_multiples_difference_squared_l3934_393497


namespace NUMINAMATH_CALUDE_calculate_expression_l3934_393459

theorem calculate_expression : 
  20062006 * 2007 + 20072007 * 2008 - 2006 * 20072007 - 2007 * 20082008 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3934_393459


namespace NUMINAMATH_CALUDE_china_population_scientific_notation_l3934_393463

/-- Represents the population of China in millions at the end of 2021 -/
def china_population : ℝ := 1412.60

/-- Proves that the population of China expressed in scientific notation is 1.4126 × 10^9 -/
theorem china_population_scientific_notation :
  (china_population * 1000000 : ℝ) = 1.4126 * (10 ^ 9) := by
  sorry

end NUMINAMATH_CALUDE_china_population_scientific_notation_l3934_393463


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_inequality_l3934_393429

theorem least_integer_absolute_value_inequality :
  ∃ (x : ℤ), (3 * |x| + 4 < 19) ∧ (∀ (y : ℤ), y < x → 3 * |y| + 4 ≥ 19) :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_inequality_l3934_393429


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3934_393403

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem f_derivative_at_zero :
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3934_393403


namespace NUMINAMATH_CALUDE_seating_arrangement_l3934_393487

theorem seating_arrangement (total_people : ℕ) (rows_of_nine : ℕ) (rows_of_ten : ℕ) : 
  total_people = 54 →
  total_people = 9 * rows_of_nine + 10 * rows_of_ten →
  rows_of_nine > 0 →
  rows_of_ten = 0 := by
sorry

end NUMINAMATH_CALUDE_seating_arrangement_l3934_393487


namespace NUMINAMATH_CALUDE_rectangular_box_existence_l3934_393446

theorem rectangular_box_existence : ∃ (a b c : ℕ), 
  a * b * c ≥ 1995 ∧ 2 * (a * b + b * c + a * c) = 958 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_existence_l3934_393446


namespace NUMINAMATH_CALUDE_triangle_altitude_excircle_radii_inequality_l3934_393455

/-- Given a triangle ABC with sides a, b, and c, altitude mc from vertex C to side AB,
    and radii ra and rb of the excircles opposite to vertices A and B respectively,
    prove that the altitude mc is at most the geometric mean of ra and rb. -/
theorem triangle_altitude_excircle_radii_inequality 
  (a b c mc ra rb : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ mc > 0 ∧ ra > 0 ∧ rb > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_altitude : mc = (2 * (a * b * c).sqrt) / (a + b + c)) 
  (h_excircle_a : ra = (a * b * c).sqrt / (b + c - a)) 
  (h_excircle_b : rb = (a * b * c).sqrt / (a + c - b)) : 
  mc ≤ Real.sqrt (ra * rb) :=
sorry

end NUMINAMATH_CALUDE_triangle_altitude_excircle_radii_inequality_l3934_393455


namespace NUMINAMATH_CALUDE_line_slope_l3934_393454

/-- The slope of the line 4x - 7y = 28 is 4/7 -/
theorem line_slope (x y : ℝ) : 4 * x - 7 * y = 28 → (y - (-4)) / (x - 0) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l3934_393454


namespace NUMINAMATH_CALUDE_initial_persons_count_l3934_393436

/-- Represents the number of days to complete the work initially -/
def initial_days : ℕ := 18

/-- Represents the number of days worked before adding more persons -/
def days_before_addition : ℕ := 6

/-- Represents the number of persons added -/
def persons_added : ℕ := 4

/-- Represents the number of days to complete the remaining work after adding persons -/
def remaining_days : ℕ := 9

/-- Represents the total amount of work -/
def total_work : ℚ := 1

/-- Theorem stating the initial number of persons working on the project -/
theorem initial_persons_count : 
  ∃ (P : ℕ), 
    (P * initial_days : ℚ) * total_work = 
    (P * days_before_addition + (P + persons_added) * remaining_days : ℚ) * total_work ∧ 
    P = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_persons_count_l3934_393436


namespace NUMINAMATH_CALUDE_circle_sum_of_center_and_radius_l3934_393438

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 6*y - 4 = -y^2 + 12*x - 12

-- Define the center and radius of the circle
def circle_center_radius (a' b' r' : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - a')^2 + (y - b')^2 = r'^2

-- Theorem statement
theorem circle_sum_of_center_and_radius :
  ∃ a' b' r', circle_center_radius a' b' r' ∧ a' + b' + r' = 3 + Real.sqrt 37 :=
sorry

end NUMINAMATH_CALUDE_circle_sum_of_center_and_radius_l3934_393438


namespace NUMINAMATH_CALUDE_platform_length_calculation_l3934_393428

/-- Calculates the length of a platform given train parameters --/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 200 →
  train_speed_kmph = 80 →
  crossing_time = 22 →
  ∃ (platform_length : ℝ), abs (platform_length - 288.84) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_platform_length_calculation_l3934_393428


namespace NUMINAMATH_CALUDE_triangle_height_l3934_393473

theorem triangle_height (C b area h : Real) : 
  C = π / 3 → 
  b = 4 → 
  area = 2 * Real.sqrt 3 → 
  area = (1 / 2) * b * h * Real.sin C → 
  h = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l3934_393473


namespace NUMINAMATH_CALUDE_max_plots_for_given_garden_l3934_393485

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  width : ℝ
  length : ℝ

/-- Represents the constraints for partitioning the garden -/
structure PartitionConstraints where
  fencing_available : ℝ
  min_plots_per_row : ℕ

/-- Calculates the maximum number of square plots given garden dimensions and constraints -/
def max_square_plots (garden : GardenDimensions) (constraints : PartitionConstraints) : ℕ :=
  sorry

/-- Theorem stating the maximum number of square plots for the given problem -/
theorem max_plots_for_given_garden :
  let garden := GardenDimensions.mk 30 60
  let constraints := PartitionConstraints.mk 3000 4
  max_square_plots garden constraints = 1250 := by
  sorry

end NUMINAMATH_CALUDE_max_plots_for_given_garden_l3934_393485


namespace NUMINAMATH_CALUDE_square_difference_ratio_l3934_393462

theorem square_difference_ratio : (1722^2 - 1715^2) / (1730^2 - 1705^2) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_ratio_l3934_393462


namespace NUMINAMATH_CALUDE_triangle_tangent_sum_product_l3934_393417

/-- Given a triangle ABC with angles α, β, and γ, 
    the sum of the tangents of these angles equals the product of their tangents. -/
theorem triangle_tangent_sum_product (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) : 
  Real.tan α + Real.tan β + Real.tan γ = Real.tan α * Real.tan β * Real.tan γ := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_sum_product_l3934_393417


namespace NUMINAMATH_CALUDE_square_root_equation_l3934_393430

theorem square_root_equation (a b : ℝ) 
  (h1 : Real.sqrt a = 2*b - 3)
  (h2 : Real.sqrt a = 3*b + 8) : 
  a = 25 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_square_root_equation_l3934_393430


namespace NUMINAMATH_CALUDE_find_b_value_l3934_393451

theorem find_b_value (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l3934_393451


namespace NUMINAMATH_CALUDE_power_division_equals_729_l3934_393411

theorem power_division_equals_729 : (3 : ℕ) ^ 15 / (27 : ℕ) ^ 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_729_l3934_393411


namespace NUMINAMATH_CALUDE_expression_equality_l3934_393422

theorem expression_equality : (2 * Real.sqrt 2 - 1)^2 + (1 + Real.sqrt 3) * (1 - Real.sqrt 3) = 7 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3934_393422


namespace NUMINAMATH_CALUDE_solution_value_l3934_393441

theorem solution_value (a b : ℝ) : 
  (2 * a + b = 3) → (6 * a + 3 * b - 1 = 8) := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3934_393441


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_singleton_one_l3934_393486

def A : Set ℝ := {-1, 1, 1/2, 3}

def B : Set ℝ := {y | ∃ x ∈ A, y = x^2}

theorem A_intersect_B_eq_singleton_one : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_singleton_one_l3934_393486


namespace NUMINAMATH_CALUDE_correct_operation_l3934_393402

theorem correct_operation : 
  (-2^2 ≠ 4) ∧ 
  ((-2)^3 ≠ -6) ∧ 
  ((-1/2)^3 = -1/8) ∧ 
  ((-7/3)^3 ≠ -8/27) :=
by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3934_393402


namespace NUMINAMATH_CALUDE_cos_2x_over_cos_pi_4_plus_x_l3934_393407

theorem cos_2x_over_cos_pi_4_plus_x (x : Real) 
  (h1 : x ∈ Set.Ioo 0 (π/4)) 
  (h2 : Real.sin (π/4 - x) = 5/13) : 
  Real.cos (2*x) / Real.cos (π/4 + x) = 24/13 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_over_cos_pi_4_plus_x_l3934_393407


namespace NUMINAMATH_CALUDE_fruit_cost_theorem_l3934_393492

/-- Calculates the cost of remaining fruit after discount and loss -/
def remaining_fruit_cost (pear_price apple_price pineapple_price plum_price : ℚ)
                         (pear_qty apple_qty pineapple_qty plum_qty : ℕ)
                         (apple_discount : ℚ) (fruit_loss_ratio : ℚ) : ℚ :=
  let total_cost := pear_price * pear_qty + 
                    apple_price * apple_qty * (1 - apple_discount) + 
                    pineapple_price * pineapple_qty + 
                    plum_price * plum_qty
  total_cost * (1 - fruit_loss_ratio)

/-- The theorem to be proved -/
theorem fruit_cost_theorem : 
  remaining_fruit_cost 1.5 0.75 2 0.5 6 4 2 1 0.25 0.5 = 7.88 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_theorem_l3934_393492


namespace NUMINAMATH_CALUDE_largest_m_base_10_l3934_393421

theorem largest_m_base_10 (m : ℕ) (A B C : ℕ) : 
  m > 0 ∧ 
  m = 25 * A + 5 * B + C ∧ 
  m = 81 * C + 9 * B + A ∧ 
  A < 5 ∧ B < 5 ∧ C < 5 ∧
  A < 9 ∧ B < 9 ∧ C < 9 →
  m ≤ 61 := by
sorry

end NUMINAMATH_CALUDE_largest_m_base_10_l3934_393421


namespace NUMINAMATH_CALUDE_min_value_implies_a_l3934_393469

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a - 1

theorem min_value_implies_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f a x ≥ -2) ∧ 
  (∃ x ∈ Set.Icc 0 1, f a x = -2) → 
  a = -1 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l3934_393469


namespace NUMINAMATH_CALUDE_expected_voters_for_candidate_A_l3934_393460

theorem expected_voters_for_candidate_A (total_voters : ℝ) (dem_percent : ℝ) 
  (dem_for_A : ℝ) (rep_for_A : ℝ) (h1 : dem_percent = 0.6) 
  (h2 : dem_for_A = 0.75) (h3 : rep_for_A = 0.3) : 
  (dem_percent * dem_for_A + (1 - dem_percent) * rep_for_A) * 100 = 57 := by
  sorry

end NUMINAMATH_CALUDE_expected_voters_for_candidate_A_l3934_393460


namespace NUMINAMATH_CALUDE_sheep_problem_l3934_393484

theorem sheep_problem (mary_initial : ℕ) (bob_multiplier : ℕ) (bob_additional : ℕ) (difference : ℕ) : 
  mary_initial = 300 →
  bob_multiplier = 2 →
  bob_additional = 35 →
  difference = 69 →
  (mary_initial + (bob_multiplier * mary_initial + bob_additional - difference - mary_initial)) = 566 :=
by
  sorry

end NUMINAMATH_CALUDE_sheep_problem_l3934_393484


namespace NUMINAMATH_CALUDE_inequality_proof_l3934_393498

theorem inequality_proof (n : ℕ+) : (2 * n.val ^ 2 + 3 * n.val + 1) ^ n.val ≥ 6 ^ n.val * (n.val.factorial) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3934_393498


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3934_393448

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), d = 12 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3934_393448


namespace NUMINAMATH_CALUDE_divisibility_property_l3934_393467

theorem divisibility_property (n : ℕ) : ∃ k : ℤ, 1 + ⌊(3 + Real.sqrt 5)^n⌋ = k * 2^n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3934_393467


namespace NUMINAMATH_CALUDE_article_cost_price_l3934_393494

/-- Given an article with marked price M and cost price C,
    prove that if 0.95M = 1.25C = 75, then C = 60. -/
theorem article_cost_price (M C : ℝ) (h : 0.95 * M = 1.25 * C ∧ 0.95 * M = 75) : C = 60 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l3934_393494


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3934_393450

theorem inequality_solution_set : 
  ¬(∀ x : ℝ, -3 * x > 9 ↔ x < -3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3934_393450


namespace NUMINAMATH_CALUDE_companion_pair_example_companion_pair_value_companion_pair_expression_l3934_393479

/-- Definition of companion rational number pairs -/
def is_companion_pair (a b : ℚ) : Prop := a - b = a * b + 1

/-- Theorem 1: (-1/2, -3) is a companion rational number pair -/
theorem companion_pair_example : is_companion_pair (-1/2) (-3) := by sorry

/-- Theorem 2: When (x+1, 5) is a companion rational number pair, x = -5/2 -/
theorem companion_pair_value (x : ℚ) : 
  is_companion_pair (x + 1) 5 → x = -5/2 := by sorry

/-- Theorem 3: For any companion rational number pair (a,b), 
    3ab-a+1/2(a+b-5ab)+1 = 1/2 -/
theorem companion_pair_expression (a b : ℚ) :
  is_companion_pair a b → 3*a*b - a + 1/2*(a+b-5*a*b) + 1 = 1/2 := by sorry

end NUMINAMATH_CALUDE_companion_pair_example_companion_pair_value_companion_pair_expression_l3934_393479


namespace NUMINAMATH_CALUDE_nicole_bought_23_candies_l3934_393440

def nicole_candies (x : ℕ) : Prop :=
  ∃ (y : ℕ), 
    (2 * x) / 3 = y + 5 + 10 ∧ 
    y ≥ 0 ∧
    x > 0

theorem nicole_bought_23_candies : 
  ∃ (x : ℕ), nicole_candies x ∧ x = 23 := by sorry

end NUMINAMATH_CALUDE_nicole_bought_23_candies_l3934_393440


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l3934_393477

/-- The quadratic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*x + 1

/-- Predicate that checks if f has only one zero for a given a -/
def has_only_one_zero (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Theorem stating that a=1 is sufficient but not necessary for f to have only one zero -/
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → has_only_one_zero a) ∧
  ¬(∀ a : ℝ, has_only_one_zero a → a = 1) :=
sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l3934_393477


namespace NUMINAMATH_CALUDE_identical_numbers_l3934_393406

theorem identical_numbers (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y^2 = y + 1 / x^2) (h2 : y^2 + 1 / x = x^2 + 1 / y) :
  x = y :=
by sorry

end NUMINAMATH_CALUDE_identical_numbers_l3934_393406


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3934_393423

theorem arithmetic_calculation : 5 * 7 + 6 * 9 + 8 * 4 + 7 * 6 = 163 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3934_393423


namespace NUMINAMATH_CALUDE_area_between_circles_l3934_393401

-- Define the circles
def outer_circle_radius : ℝ := 12
def chord_length : ℝ := 20

-- Define the theorem
theorem area_between_circles :
  ∃ (inner_circle_radius : ℝ),
    inner_circle_radius > 0 ∧
    inner_circle_radius < outer_circle_radius ∧
    chord_length^2 = 4 * (outer_circle_radius^2 - inner_circle_radius^2) ∧
    π * (outer_circle_radius^2 - inner_circle_radius^2) = 100 * π :=
by
  sorry


end NUMINAMATH_CALUDE_area_between_circles_l3934_393401


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l3934_393439

theorem cyclist_speed_problem (v : ℝ) :
  v > 0 →
  (20 : ℝ) / (9 / v + 11 / 9) = 9.8019801980198 →
  ∃ ε > 0, |v - 11.03| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l3934_393439


namespace NUMINAMATH_CALUDE_square_area_proof_l3934_393464

theorem square_area_proof (x : ℝ) : 
  (5 * x - 20 : ℝ) = (25 - 2 * x : ℝ) → 
  (5 * x - 20 : ℝ) > 0 → 
  ((5 * x - 20 : ℝ) * (5 * x - 20 : ℝ)) = 7225 / 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l3934_393464


namespace NUMINAMATH_CALUDE_exponent_and_square_of_negative_two_l3934_393478

theorem exponent_and_square_of_negative_two :
  (-2^2 = -4) ∧ ((-2)^3 = -8) ∧ ((-2)^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_exponent_and_square_of_negative_two_l3934_393478


namespace NUMINAMATH_CALUDE_twenty_two_oclock_is_ten_pm_l3934_393458

/-- Converts 24-hour time format to 12-hour time format -/
def convert_24_to_12 (hour : ℕ) : ℕ × String :=
  if hour < 12 then (if hour = 0 then 12 else hour, "AM")
  else ((if hour = 12 then 12 else hour - 12), "PM")

/-- Theorem stating that 22:00 in 24-hour format is equivalent to 10:00 PM in 12-hour format -/
theorem twenty_two_oclock_is_ten_pm :
  convert_24_to_12 22 = (10, "PM") :=
sorry

end NUMINAMATH_CALUDE_twenty_two_oclock_is_ten_pm_l3934_393458


namespace NUMINAMATH_CALUDE_percent_increase_in_sales_l3934_393435

theorem percent_increase_in_sales (sales_this_year sales_last_year : ℝ) 
  (h1 : sales_this_year = 460)
  (h2 : sales_last_year = 320) :
  (sales_this_year - sales_last_year) / sales_last_year * 100 = 43.75 := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_in_sales_l3934_393435


namespace NUMINAMATH_CALUDE_tangent_half_angle_identity_l3934_393426

theorem tangent_half_angle_identity (α : Real) (m : Real) 
  (h : Real.tan (α / 2) = m) : 
  (1 - 2 * Real.sin (α / 2) ^ 2) / (1 + Real.sin α) = (1 - m) / (1 + m) := by
  sorry

end NUMINAMATH_CALUDE_tangent_half_angle_identity_l3934_393426


namespace NUMINAMATH_CALUDE_function_through_points_l3934_393470

theorem function_through_points (a p q : ℝ) : 
  a > 0 →
  2^p / (2^p + a*p) = 6/5 →
  2^q / (2^q + a*q) = -1/5 →
  2^(p+q) = 16*p*q →
  a = 4 :=
by sorry

end NUMINAMATH_CALUDE_function_through_points_l3934_393470


namespace NUMINAMATH_CALUDE_mechanic_work_hours_l3934_393461

/-- Calculates the number of hours a mechanic worked given the total cost, 
    cost of parts, and labor rate per minute. -/
theorem mechanic_work_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (num_parts : ℕ) 
  (labor_rate_per_minute : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_cost = 20) 
  (h3 : num_parts = 2) 
  (h4 : labor_rate_per_minute = 0.5) : 
  (total_cost - part_cost * num_parts) / (labor_rate_per_minute * 60) = 6 := by
sorry

end NUMINAMATH_CALUDE_mechanic_work_hours_l3934_393461


namespace NUMINAMATH_CALUDE_geometric_progression_values_l3934_393466

theorem geometric_progression_values (p : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*p - 1) = |p - 8| * r ∧ (4*p + 5) = (2*p - 1) * r) ↔ 
  (p = -1 ∨ p = 39/8) := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_values_l3934_393466


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3934_393444

theorem complex_magnitude_problem (z : ℂ) (h : (1 - Complex.I * Real.sqrt 3) * z = Complex.I) :
  Complex.abs z = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3934_393444


namespace NUMINAMATH_CALUDE_max_profit_at_max_price_max_profit_value_l3934_393416

/-- Represents the souvenir selling scenario with given conditions -/
structure SouvenirSales where
  cost_price : ℝ := 6
  base_price : ℝ := 8
  base_sales : ℝ := 200
  price_sales_ratio : ℝ := 10
  max_price : ℝ := 12

/-- Calculates daily sales based on selling price -/
def daily_sales (s : SouvenirSales) (x : ℝ) : ℝ :=
  s.base_sales - s.price_sales_ratio * (x - s.base_price)

/-- Calculates daily profit based on selling price -/
def daily_profit (s : SouvenirSales) (x : ℝ) : ℝ :=
  (x - s.cost_price) * (daily_sales s x)

/-- Theorem stating the maximum profit occurs at the maximum allowed price -/
theorem max_profit_at_max_price (s : SouvenirSales) :
  ∀ x, s.cost_price ≤ x ∧ x ≤ s.max_price →
    daily_profit s x ≤ daily_profit s s.max_price :=
sorry

/-- Theorem stating the value of the maximum profit -/
theorem max_profit_value (s : SouvenirSales) :
  daily_profit s s.max_price = 960 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_max_price_max_profit_value_l3934_393416


namespace NUMINAMATH_CALUDE_percentage_increase_girls_to_total_l3934_393413

def boys : ℕ := 2000
def girls : ℕ := 5000

theorem percentage_increase_girls_to_total : 
  (((boys + girls) - girls : ℚ) / girls) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_girls_to_total_l3934_393413


namespace NUMINAMATH_CALUDE_unique_tangent_length_l3934_393427

theorem unique_tangent_length (m n t₁ : ℝ) : 
  (30 : ℝ) = m + n →
  t₁^2 = m * n →
  m ∈ Set.Ioo 0 30 →
  ∃ k : ℕ, m = 2 * k →
  ∃! t₁ : ℝ, t₁ > 0 ∧ t₁^2 = m * (30 - m) :=
sorry

end NUMINAMATH_CALUDE_unique_tangent_length_l3934_393427


namespace NUMINAMATH_CALUDE_ninth_grade_maximizes_profit_l3934_393499

/-- Represents the profit function for a product with different quality grades. -/
def profit_function (k : ℕ) : ℝ :=
  let profit_per_piece := 8 + 2 * (k - 1)
  let pieces_produced := 60 - 3 * (k - 1)
  (profit_per_piece * pieces_produced : ℝ)

/-- Theorem stating that the 9th quality grade maximizes the profit. -/
theorem ninth_grade_maximizes_profit :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → profit_function k ≤ profit_function 9 := by
  sorry

#check ninth_grade_maximizes_profit

end NUMINAMATH_CALUDE_ninth_grade_maximizes_profit_l3934_393499


namespace NUMINAMATH_CALUDE_h_negative_a_equals_negative_two_l3934_393481

-- Define the functions
variable (f g h : ℝ → ℝ)
variable (a : ℝ)

-- Define the properties of the functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem h_negative_a_equals_negative_two 
  (hf_even : is_even f)
  (hg_odd : is_odd g)
  (hf_a : f a = 2)
  (hg_a : g a = 3)
  (hh : ∀ x, h x = f x + g x - 1) :
  h (-a) = -2 := by sorry

end NUMINAMATH_CALUDE_h_negative_a_equals_negative_two_l3934_393481


namespace NUMINAMATH_CALUDE_least_four_digit_divisible_by_digits_l3934_393483

/-- A function that checks if a number is a four-digit positive integer with all different digits -/
def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10)

/-- A function that checks if a number is divisible by all of its digits -/
def divisible_by_digits (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 4 → (n % ((n / 10^i) % 10) = 0 ∨ (n / 10^i) % 10 = 0)

/-- The main theorem stating that 1236 is the least number satisfying the conditions -/
theorem least_four_digit_divisible_by_digits :
  is_valid_number 1236 ∧ 
  divisible_by_digits 1236 ∧
  (∀ m : ℕ, m < 1236 → ¬(is_valid_number m ∧ divisible_by_digits m)) :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_divisible_by_digits_l3934_393483


namespace NUMINAMATH_CALUDE_band_practice_schedule_l3934_393415

theorem band_practice_schedule (anthony ben carlos dean : ℕ) 
  (h1 : anthony = 5)
  (h2 : ben = 6)
  (h3 : carlos = 8)
  (h4 : dean = 9) :
  Nat.lcm anthony (Nat.lcm ben (Nat.lcm carlos dean)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_band_practice_schedule_l3934_393415


namespace NUMINAMATH_CALUDE_rectangle_side_lengths_l3934_393495

theorem rectangle_side_lengths (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) 
  (h4 : a * b = 2 * a + 2 * b) : a < 4 ∧ b > 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_lengths_l3934_393495


namespace NUMINAMATH_CALUDE_sequence_sum_99_100_l3934_393474

def sequence_term (n : ℕ) : ℚ :=
  let group := (n.sqrt : ℕ)
  let position := n - (group - 1) * group
  ↑(group + 1 - position) / position

theorem sequence_sum_99_100 : 
  sequence_term 99 + sequence_term 100 = 37 / 24 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_99_100_l3934_393474


namespace NUMINAMATH_CALUDE_inequality_proof_l3934_393442

theorem inequality_proof (n : ℕ) : 
  2 * Real.sqrt (n + 1 : ℝ) - 2 * Real.sqrt (n : ℝ) < 1 / Real.sqrt (n : ℝ) ∧ 
  1 / Real.sqrt (n : ℝ) < 2 * Real.sqrt (n : ℝ) - 2 * Real.sqrt ((n - 1) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3934_393442


namespace NUMINAMATH_CALUDE_problem_solution_l3934_393476

theorem problem_solution :
  ∀ (a b c : ℝ),
    (∃ (x : ℝ), x > 0 ∧ (a - 2)^2 = x ∧ (7 - 2*a)^2 = x) →
    ((3*b + 1)^(1/3) = -2) →
    (c = ⌊Real.sqrt 39⌋) →
    (a = 5 ∧ b = -3 ∧ c = 6 ∧ 
     (∃ (y : ℝ), y^2 = 5*a + 2*b - c ∧ (y = Real.sqrt 13 ∨ y = -Real.sqrt 13))) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3934_393476


namespace NUMINAMATH_CALUDE_circular_platform_area_l3934_393408

/-- The area of a circular platform with a diameter of 2 yards is π square yards. -/
theorem circular_platform_area (diameter : ℝ) (h : diameter = 2) : 
  (π * (diameter / 2)^2 : ℝ) = π := by sorry

end NUMINAMATH_CALUDE_circular_platform_area_l3934_393408


namespace NUMINAMATH_CALUDE_pushups_total_l3934_393414

def zachary_pushups : ℕ := 47

def david_pushups (zachary : ℕ) : ℕ := zachary + 15

def emily_pushups (david : ℕ) : ℕ := 2 * david

def total_pushups (zachary david emily : ℕ) : ℕ := zachary + david + emily

theorem pushups_total :
  total_pushups zachary_pushups (david_pushups zachary_pushups) (emily_pushups (david_pushups zachary_pushups)) = 233 := by
  sorry

end NUMINAMATH_CALUDE_pushups_total_l3934_393414


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3934_393468

def A : Set ℝ := {x | x < 3}
def B : Set ℝ := {x | 2 - x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3934_393468


namespace NUMINAMATH_CALUDE_complete_square_result_l3934_393471

/-- Given a quadratic equation 16x^2 + 32x - 512 = 0, prove that when solved by completing the square
    to the form (x + r)^2 = s, the value of s is 33. -/
theorem complete_square_result (x r s : ℝ) : 
  (16 * x^2 + 32 * x - 512 = 0) →
  ((x + r)^2 = s) →
  (s = 33) := by
sorry

end NUMINAMATH_CALUDE_complete_square_result_l3934_393471


namespace NUMINAMATH_CALUDE_additional_planes_needed_l3934_393418

def current_planes : ℕ := 29
def row_size : ℕ := 8

theorem additional_planes_needed :
  (row_size - (current_planes % row_size)) % row_size = 3 := by sorry

end NUMINAMATH_CALUDE_additional_planes_needed_l3934_393418


namespace NUMINAMATH_CALUDE_fifth_day_distance_l3934_393472

def running_distance (day : ℕ) : ℕ :=
  2 + (day - 1)

theorem fifth_day_distance : running_distance 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fifth_day_distance_l3934_393472


namespace NUMINAMATH_CALUDE_multiples_count_theorem_l3934_393410

def count_multiples (n : ℕ) (d : ℕ) : ℕ :=
  (n / d : ℕ)

def count_multiples_of_2_or_3_not_4_or_5 (upper_bound : ℕ) : ℕ :=
  count_multiples upper_bound 2 + count_multiples upper_bound 3 -
  count_multiples upper_bound 6 - count_multiples upper_bound 4 -
  count_multiples upper_bound 5 + count_multiples upper_bound 20

theorem multiples_count_theorem (upper_bound : ℕ) :
  upper_bound = 200 →
  count_multiples_of_2_or_3_not_4_or_5 upper_bound = 53 := by
  sorry

end NUMINAMATH_CALUDE_multiples_count_theorem_l3934_393410


namespace NUMINAMATH_CALUDE_quadrilaterals_on_circle_l3934_393480

/-- The number of distinct convex quadrilaterals formed from points on a circle -/
theorem quadrilaterals_on_circle (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 4) :
  (Nat.choose n k) = 495 := by
  sorry

end NUMINAMATH_CALUDE_quadrilaterals_on_circle_l3934_393480


namespace NUMINAMATH_CALUDE_no_hall_with_101_people_l3934_393419

/-- Represents a person in the hall -/
inductive Person
| knight : Person
| liar : Person

/-- Represents the hall with people and their pointing relationships -/
structure Hall :=
  (people : Finset Nat)
  (type : Nat → Person)
  (points_to : Nat → Nat)
  (in_hall : ∀ n, n ∈ people → points_to n ∈ people)
  (all_pointed_at : ∀ n ∈ people, ∃ m ∈ people, points_to m = n)
  (knight_points_to_liar : ∀ n ∈ people, type n = Person.knight → type (points_to n) = Person.liar)
  (liar_points_to_knight : ∀ n ∈ people, type n = Person.liar → type (points_to n) = Person.knight)

/-- Theorem stating that it's impossible to have exactly 101 people in the hall -/
theorem no_hall_with_101_people : ¬ ∃ (h : Hall), Finset.card h.people = 101 := by
  sorry

end NUMINAMATH_CALUDE_no_hall_with_101_people_l3934_393419


namespace NUMINAMATH_CALUDE_f_min_value_l3934_393496

/-- The quadratic function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The minimum value of f(x) is 2 -/
theorem f_min_value : ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l3934_393496


namespace NUMINAMATH_CALUDE_gcf_lcm_problem_l3934_393443

-- Define GCF (Greatest Common Factor)
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM (Least Common Multiple)
def LCM (c d : ℕ) : ℕ := Nat.lcm c d

-- Theorem statement
theorem gcf_lcm_problem : GCF (LCM 9 21) (LCM 10 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_problem_l3934_393443


namespace NUMINAMATH_CALUDE_expression_values_l3934_393456

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := (a / abs a) + (b / abs b) + (c / abs c) + (d / abs d) + ((a * b * c * d) / abs (a * b * c * d))
  expr = 5 ∨ expr = 1 ∨ expr = -3 ∨ expr = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l3934_393456


namespace NUMINAMATH_CALUDE_coin_toss_sequences_l3934_393420

/-- The number of coin tosses in the sequence -/
def n : ℕ := 15

/-- The number of "HH" (heads followed by heads) in the sequence -/
def hh_count : ℕ := 2

/-- The number of "HT" (heads followed by tails) in the sequence -/
def ht_count : ℕ := 3

/-- The number of "TH" (tails followed by heads) in the sequence -/
def th_count : ℕ := 4

/-- The number of "TT" (tails followed by tails) in the sequence -/
def tt_count : ℕ := 5

/-- The total number of distinct sequences -/
def total_sequences : ℕ := 2522520

/-- Theorem stating that the number of distinct sequences of n coin tosses
    with exactly hh_count "HH", ht_count "HT", th_count "TH", and tt_count "TT"
    is equal to total_sequences -/
theorem coin_toss_sequences :
  (Nat.factorial (n - 1)) / (Nat.factorial hh_count * Nat.factorial ht_count *
  Nat.factorial th_count * Nat.factorial tt_count) = total_sequences := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_sequences_l3934_393420


namespace NUMINAMATH_CALUDE_rectangle_to_square_l3934_393432

/-- Given a rectangle with area 54 m², if one side is tripled and the other is halved to form a square, 
    the side length of the resulting square is 9 m. -/
theorem rectangle_to_square (a b : ℝ) (h1 : a * b = 54) (h2 : 3 * a = b / 2) : 
  3 * a = 9 ∧ b / 2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l3934_393432


namespace NUMINAMATH_CALUDE_no_solution_for_divisibility_l3934_393489

theorem no_solution_for_divisibility (n : ℕ) (hn : n ≥ 1) : ¬(9 ∣ (7^n + n^3)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_divisibility_l3934_393489


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3934_393449

-- Define set A
def A : Set ℝ := {y | ∃ x, y = |x|}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 1 - 2*x - x^2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {y | 0 ≤ y ∧ y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3934_393449


namespace NUMINAMATH_CALUDE_rowing_speed_l3934_393490

theorem rowing_speed (downstream_distance upstream_distance : ℝ)
                     (total_time : ℝ)
                     (current_speed : ℝ)
                     (h1 : downstream_distance = 3.5)
                     (h2 : upstream_distance = 3.5)
                     (h3 : total_time = 5/3)
                     (h4 : current_speed = 2) :
  ∃ still_water_speed : ℝ,
    still_water_speed = 5 ∧
    downstream_distance / (still_water_speed + current_speed) +
    upstream_distance / (still_water_speed - current_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_rowing_speed_l3934_393490


namespace NUMINAMATH_CALUDE_inequality_of_means_l3934_393488

theorem inequality_of_means (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : (x - y)^2 > 396*x*y) (h2 : 2.0804*x*y > x^2 + y^2) :
  1.01 * Real.sqrt (x*y) > (x + y)/2 ∧ (x + y)/2 > 100 * (2*x*y/(x + y)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_means_l3934_393488


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3934_393493

theorem geometric_sequence_third_term
  (a₁ : ℝ)
  (a₅ : ℝ)
  (h₁ : a₁ = 4)
  (h₂ : a₅ = 1296)
  (h₃ : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → ∃ r : ℝ, a₁ * r^(n-1) = a₁ * (a₅ / a₁)^((n-1)/4)) :
  ∃ a₃ : ℝ, a₃ = 36 ∧ a₃ = a₁ * (a₅ / a₁)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3934_393493


namespace NUMINAMATH_CALUDE_yanna_payment_l3934_393437

def shirt_price : ℕ := 5
def sandal_price : ℕ := 3
def num_shirts : ℕ := 10
def num_sandals : ℕ := 3
def change : ℕ := 41

theorem yanna_payment :
  shirt_price * num_shirts + sandal_price * num_sandals + change = 100 := by
  sorry

end NUMINAMATH_CALUDE_yanna_payment_l3934_393437


namespace NUMINAMATH_CALUDE_omega_function_iff_strictly_increasing_l3934_393400

def OmegaFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

theorem omega_function_iff_strictly_increasing (f : ℝ → ℝ) :
  OmegaFunction f ↔ StrictMono f := by sorry

end NUMINAMATH_CALUDE_omega_function_iff_strictly_increasing_l3934_393400


namespace NUMINAMATH_CALUDE_parallel_vectors_l3934_393445

/-- Two vectors in ℝ² -/
def a (m : ℝ) : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (2, -3)

/-- Definition of parallel vectors in ℝ² -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

/-- Theorem: If a(m) is parallel to b, then m = -3/2 -/
theorem parallel_vectors (m : ℝ) :
  parallel (a m) b → m = -3/2 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l3934_393445


namespace NUMINAMATH_CALUDE_sum_of_squares_l3934_393452

theorem sum_of_squares (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 4) 
  (h2 : a/x + b/y + c/z = 3) : 
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3934_393452


namespace NUMINAMATH_CALUDE_multiplier_is_three_l3934_393482

theorem multiplier_is_three (n : ℝ) (h1 : 3 * n = (26 - n) + 14) (h2 : n = 10) : 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_is_three_l3934_393482


namespace NUMINAMATH_CALUDE_equation_solution_l3934_393491

theorem equation_solution :
  let x : ℝ := 32
  let equation (number : ℝ) := 35 - (23 - (15 - x)) = 12 * 2 / (1 / number)
  ∃ (number : ℝ), equation number ∧ number = -4.8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3934_393491


namespace NUMINAMATH_CALUDE_square_area_difference_l3934_393433

/-- Given two squares ABCD and EGFO with the specified conditions, 
    prove that the difference between their areas is 11.5 -/
theorem square_area_difference (a b : ℕ+) 
  (h1 : (a.val : ℝ)^2 / 2 - (b.val : ℝ)^2 / 2 = 3.25) 
  (h2 : (b.val : ℝ) > (a.val : ℝ)) : 
  (a.val : ℝ)^2 - (b.val : ℝ)^2 = -11.5 := by
  sorry

end NUMINAMATH_CALUDE_square_area_difference_l3934_393433


namespace NUMINAMATH_CALUDE_business_value_calculation_l3934_393425

theorem business_value_calculation (owned_share : ℚ) (sold_portion : ℚ) (sale_price : ℕ) :
  owned_share = 2/3 →
  sold_portion = 3/4 →
  sale_price = 75000 →
  (sale_price : ℚ) / (owned_share * sold_portion) = 150000 := by
  sorry

end NUMINAMATH_CALUDE_business_value_calculation_l3934_393425


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_plus_i_times_i_l3934_393409

theorem imaginary_part_of_2_plus_i_times_i (i : ℂ) : 
  i ^ 2 = -1 → Complex.im ((2 + i) * i) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_plus_i_times_i_l3934_393409


namespace NUMINAMATH_CALUDE_smaller_number_of_product_and_difference_l3934_393431

theorem smaller_number_of_product_and_difference (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ x * y = 323 ∧ x - y = 2 → y = 17 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_of_product_and_difference_l3934_393431


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3934_393412

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) : 
  z.im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3934_393412
