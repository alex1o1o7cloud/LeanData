import Mathlib

namespace NUMINAMATH_CALUDE_root_difference_for_arithmetic_progression_cubic_l2599_259978

theorem root_difference_for_arithmetic_progression_cubic (a b c d : ℝ) :
  (∃ x y z : ℝ, 
    (49 * x^3 - 105 * x^2 + 63 * x - 10 = 0) ∧
    (49 * y^3 - 105 * y^2 + 63 * y - 10 = 0) ∧
    (49 * z^3 - 105 * z^2 + 63 * z - 10 = 0) ∧
    (y - x = z - y) ∧
    (x < y) ∧ (y < z)) →
  (z - x = 2 * Real.sqrt 11 / 7) :=
by sorry

end NUMINAMATH_CALUDE_root_difference_for_arithmetic_progression_cubic_l2599_259978


namespace NUMINAMATH_CALUDE_percentage_increase_l2599_259926

theorem percentage_increase (N : ℝ) (P : ℝ) : 
  N = 40 →
  N + (P / 100) * N - (N - (30 / 100) * N) = 22 →
  P = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l2599_259926


namespace NUMINAMATH_CALUDE_inequality_proof_l2599_259975

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x*y + y*z + z*x ≤ 1) : 
  (x + 1/x) * (y + 1/y) * (z + 1/z) ≥ 8 * (x + y) * (y + z) * (z + x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2599_259975


namespace NUMINAMATH_CALUDE_unique_dataset_l2599_259950

def is_valid_dataset (x : Fin 4 → ℕ) : Prop :=
  (∀ i, x i > 0) ∧
  (x 0 ≤ x 1) ∧ (x 1 ≤ x 2) ∧ (x 2 ≤ x 3) ∧
  (x 0 + x 1 + x 2 + x 3 = 8) ∧
  ((x 1 + x 2) / 2 = 2) ∧
  ((x 0 - 2)^2 + (x 1 - 2)^2 + (x 2 - 2)^2 + (x 3 - 2)^2 = 4)

theorem unique_dataset :
  ∀ x : Fin 4 → ℕ, is_valid_dataset x → (x 0 = 1 ∧ x 1 = 1 ∧ x 2 = 3 ∧ x 3 = 3) :=
sorry

end NUMINAMATH_CALUDE_unique_dataset_l2599_259950


namespace NUMINAMATH_CALUDE_shoe_tying_time_difference_l2599_259938

theorem shoe_tying_time_difference (jack_shoe_time toddler_count total_time : ℕ) :
  jack_shoe_time = 4 →
  toddler_count = 2 →
  total_time = 18 →
  (total_time - jack_shoe_time) / toddler_count - jack_shoe_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_shoe_tying_time_difference_l2599_259938


namespace NUMINAMATH_CALUDE_product_xy_is_264_l2599_259933

theorem product_xy_is_264 (x y : ℝ) 
  (eq1 : -3 * x + 4 * y = 28) 
  (eq2 : 3 * x - 2 * y = 8) : 
  x * y = 264 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_is_264_l2599_259933


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l2599_259935

theorem quadratic_perfect_square (m : ℝ) :
  (∃ (a b : ℝ), ∀ x, (6*x^2 + 16*x + 3*m) / 6 = (a*x + b)^2) →
  m = 32/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l2599_259935


namespace NUMINAMATH_CALUDE_at_least_four_2x2_squares_sum_greater_than_100_l2599_259922

/-- Represents a square on the 8x8 board -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the board configuration -/
def Board := Square → Fin 64

/-- Checks if a given 2x2 square has a sum greater than 100 -/
def is_sum_greater_than_100 (board : Board) (top_left : Square) : Prop :=
  let sum := (board top_left).val + 
              (board ⟨top_left.row, top_left.col.succ⟩).val + 
              (board ⟨top_left.row.succ, top_left.col⟩).val + 
              (board ⟨top_left.row.succ, top_left.col.succ⟩).val
  sum > 100

/-- The main theorem to be proved -/
theorem at_least_four_2x2_squares_sum_greater_than_100 (board : Board) 
  (h_unique : ∀ (s1 s2 : Square), board s1 = board s2 → s1 = s2) :
  ∃ (s1 s2 s3 s4 : Square), 
    s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4 ∧
    is_sum_greater_than_100 board s1 ∧
    is_sum_greater_than_100 board s2 ∧
    is_sum_greater_than_100 board s3 ∧
    is_sum_greater_than_100 board s4 :=
  sorry

end NUMINAMATH_CALUDE_at_least_four_2x2_squares_sum_greater_than_100_l2599_259922


namespace NUMINAMATH_CALUDE_increasing_order_x_xx_xxx_l2599_259969

theorem increasing_order_x_xx_xxx (x : ℝ) (h1 : 1 < x) (h2 : x < 1.1) :
  x < x^x ∧ x^x < x^(x^x) := by sorry

end NUMINAMATH_CALUDE_increasing_order_x_xx_xxx_l2599_259969


namespace NUMINAMATH_CALUDE_iron_cars_count_l2599_259927

/-- Represents the initial state and rules for a train delivery problem -/
structure TrainProblem where
  coal_cars : ℕ
  wood_cars : ℕ
  station_distance : ℕ
  travel_time : ℕ
  max_coal_deposit : ℕ
  max_iron_deposit : ℕ
  max_wood_deposit : ℕ
  total_delivery_time : ℕ

/-- Calculates the number of iron cars given a TrainProblem -/
def calculate_iron_cars (problem : TrainProblem) : ℕ :=
  let num_stations := problem.total_delivery_time / problem.travel_time
  num_stations * problem.max_iron_deposit

/-- Theorem stating that for the given problem, the number of iron cars is 12 -/
theorem iron_cars_count (problem : TrainProblem) 
  (h1 : problem.coal_cars = 6)
  (h2 : problem.wood_cars = 2)
  (h3 : problem.station_distance = 6)
  (h4 : problem.travel_time = 25)
  (h5 : problem.max_coal_deposit = 2)
  (h6 : problem.max_iron_deposit = 3)
  (h7 : problem.max_wood_deposit = 1)
  (h8 : problem.total_delivery_time = 100) :
  calculate_iron_cars problem = 12 := by
  sorry

end NUMINAMATH_CALUDE_iron_cars_count_l2599_259927


namespace NUMINAMATH_CALUDE_three_digit_numbers_after_exclusion_l2599_259954

/-- The count of three-digit numbers (100 to 999) -/
def total_three_digit_numbers : ℕ := 900

/-- The count of numbers in the form ABA where A and B are digits and A ≠ 0 -/
def count_ABA : ℕ := 81

/-- The count of numbers in the form AAB or BAA where A and B are digits and A ≠ 0 -/
def count_AAB_BAA : ℕ := 81

/-- The total count of excluded numbers -/
def total_excluded : ℕ := count_ABA + count_AAB_BAA

theorem three_digit_numbers_after_exclusion :
  total_three_digit_numbers - total_excluded = 738 := by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_after_exclusion_l2599_259954


namespace NUMINAMATH_CALUDE_pete_total_books_matt_year2_increase_l2599_259904

/-- The number of books Matt read in the first year -/
def matt_year1 : ℕ := 50

/-- The number of books Matt read in the second year -/
def matt_year2 : ℕ := 75

/-- The number of books Pete read in the first year -/
def pete_year1 : ℕ := 2 * matt_year1

/-- The number of books Pete read in the second year -/
def pete_year2 : ℕ := 2 * pete_year1

/-- Theorem stating that Pete read 300 books across both years -/
theorem pete_total_books : pete_year1 + pete_year2 = 300 := by
  sorry

/-- Verification that Matt's second year reading increased by 50% -/
theorem matt_year2_increase : matt_year2 = (3 * matt_year1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_pete_total_books_matt_year2_increase_l2599_259904


namespace NUMINAMATH_CALUDE_right_plus_acute_is_obtuse_quarter_circle_is_right_angle_l2599_259987

-- Define angles in degrees
def RightAngle : ℝ := 90
def FullCircle : ℝ := 360

-- Define angle types
def IsAcuteAngle (θ : ℝ) : Prop := 0 < θ ∧ θ < RightAngle
def IsObtuseAngle (θ : ℝ) : Prop := RightAngle < θ ∧ θ < 180

theorem right_plus_acute_is_obtuse (θ : ℝ) (h : IsAcuteAngle θ) :
  IsObtuseAngle (RightAngle + θ) := by sorry

theorem quarter_circle_is_right_angle :
  FullCircle / 4 = RightAngle := by sorry

end NUMINAMATH_CALUDE_right_plus_acute_is_obtuse_quarter_circle_is_right_angle_l2599_259987


namespace NUMINAMATH_CALUDE_elevator_descent_time_l2599_259957

/-- Represents the elevator descent problem -/
def elevator_descent (total_floors : ℕ) 
  (first_half_time : ℕ) 
  (mid_floor_time : ℕ) 
  (final_floor_time : ℕ) : Prop :=
  let first_half := total_floors / 2
  let mid_section := 5
  let final_section := 5
  let total_time := first_half_time + mid_floor_time * mid_section + final_floor_time * final_section
  total_floors = 20 ∧ 
  first_half_time = 15 ∧ 
  mid_floor_time = 5 ∧ 
  final_floor_time = 16 ∧ 
  total_time / 60 = 2

/-- Theorem stating that the elevator descent takes 2 hours -/
theorem elevator_descent_time : 
  elevator_descent 20 15 5 16 := by sorry

end NUMINAMATH_CALUDE_elevator_descent_time_l2599_259957


namespace NUMINAMATH_CALUDE_custom_op_5_3_l2599_259946

-- Define the custom operation
def custom_op (m n : ℕ) : ℕ := n ^ 2 - m

-- Theorem statement
theorem custom_op_5_3 : custom_op 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_5_3_l2599_259946


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2599_259994

/-- The range of k for which the quadratic equation (k+1)x^2 - 2x + 1 = 0 has two real roots -/
theorem quadratic_equation_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (k + 1) * x₁^2 - 2 * x₁ + 1 = 0 ∧ 
    (k + 1) * x₂^2 - 2 * x₂ + 1 = 0) ↔ 
  (k ≤ 0 ∧ k ≠ -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2599_259994


namespace NUMINAMATH_CALUDE_max_fly_path_2x1x1_box_l2599_259982

/-- Represents a rectangular box with dimensions a, b, and c -/
structure Box where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the maximum path length for a fly in the given box -/
def maxFlyPathLength (box : Box) : ℝ :=
  sorry

/-- Theorem stating the maximum fly path length for a 2x1x1 box -/
theorem max_fly_path_2x1x1_box :
  let box : Box := { a := 2, b := 1, c := 1 }
  maxFlyPathLength box = 4 + 4 * Real.sqrt 5 + Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_max_fly_path_2x1x1_box_l2599_259982


namespace NUMINAMATH_CALUDE_quadrant_I_solution_l2599_259953

theorem quadrant_I_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 3 ∧ c * x + y = 4 ∧ x > 0 ∧ y > 0) ↔ -1 < c ∧ c < 4/3 :=
by sorry

end NUMINAMATH_CALUDE_quadrant_I_solution_l2599_259953


namespace NUMINAMATH_CALUDE_exists_n_plus_sum_of_digits_eq_125_l2599_259985

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Theorem stating the existence of a natural number n such that n + S(n) = 125 -/
theorem exists_n_plus_sum_of_digits_eq_125 :
  ∃ n : ℕ, n + sumOfDigits n = 125 ∧ n = 121 :=
sorry

end NUMINAMATH_CALUDE_exists_n_plus_sum_of_digits_eq_125_l2599_259985


namespace NUMINAMATH_CALUDE_EF_equals_5_sqrt_35_div_3_l2599_259919

/-- A rectangle ABCD with a point E inside -/
structure Rectangle :=
  (A B C D E : ℝ × ℝ)
  (is_rectangle : A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ C.2 = D.2)
  (AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 30)
  (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 40)
  (E_inside : E.1 > A.1 ∧ E.1 < C.1 ∧ E.2 > A.2 ∧ E.2 < C.2)
  (EA_length : Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 10)
  (EB_length : Real.sqrt ((E.1 - B.1)^2 + (E.2 - B.2)^2) = 30)

/-- The length of EF, where F is the foot of the perpendicular from E to AD -/
def EF_length (r : Rectangle) : ℝ := r.E.2 - r.A.2

theorem EF_equals_5_sqrt_35_div_3 (r : Rectangle) : 
  EF_length r = 5 * Real.sqrt 35 / 3 :=
sorry

end NUMINAMATH_CALUDE_EF_equals_5_sqrt_35_div_3_l2599_259919


namespace NUMINAMATH_CALUDE_parabola_shift_l2599_259914

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_shift :
  let original := Parabola.mk 2 0 0
  let shifted := shift_parabola original 2 1
  shifted = Parabola.mk 2 (-8) 7 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_l2599_259914


namespace NUMINAMATH_CALUDE_ant_count_approximation_l2599_259902

/-- Represents the dimensions of a rectangular field in feet -/
structure FieldDimensions where
  width : ℝ
  length : ℝ

/-- Calculates the number of ants in a rectangular field given the specified conditions -/
def calculateAnts (field : FieldDimensions) (antDensity : ℝ) (rockCoverage : ℝ) : ℝ :=
  let inchesPerFoot : ℝ := 12
  let fieldAreaInches : ℝ := field.width * field.length * inchesPerFoot * inchesPerFoot
  let antHabitatArea : ℝ := fieldAreaInches * (1 - rockCoverage)
  antHabitatArea * antDensity

/-- Theorem stating that the number of ants in the field is approximately 26 million -/
theorem ant_count_approximation :
  let field : FieldDimensions := { width := 200, length := 500 }
  let antDensity : ℝ := 2  -- ants per square inch
  let rockCoverage : ℝ := 0.1  -- 10% of the field covered by rocks
  abs (calculateAnts field antDensity rockCoverage - 26000000) ≤ 500000 := by
  sorry


end NUMINAMATH_CALUDE_ant_count_approximation_l2599_259902


namespace NUMINAMATH_CALUDE_shampoo_bottles_l2599_259909

theorem shampoo_bottles (small_capacity large_capacity current_amount : ℕ) 
  (h1 : small_capacity = 40)
  (h2 : large_capacity = 800)
  (h3 : current_amount = 120) : 
  (large_capacity - current_amount) / small_capacity = 17 := by
  sorry

end NUMINAMATH_CALUDE_shampoo_bottles_l2599_259909


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2599_259952

theorem geometric_series_sum : 
  let a : ℝ := 2/3
  let r : ℝ := 2/3
  let series_sum : ℝ := ∑' i, a * r^(i - 1)
  series_sum = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2599_259952


namespace NUMINAMATH_CALUDE_camera_price_difference_l2599_259989

/-- The list price of Camera Y in dollars -/
def list_price : ℝ := 59.99

/-- The discount percentage at Budget Buys -/
def budget_buys_discount : ℝ := 0.15

/-- The discount amount at Frugal Finds in dollars -/
def frugal_finds_discount : ℝ := 20

/-- The sale price at Budget Buys in dollars -/
def budget_buys_price : ℝ := list_price * (1 - budget_buys_discount)

/-- The sale price at Frugal Finds in dollars -/
def frugal_finds_price : ℝ := list_price - frugal_finds_discount

/-- The price difference in cents -/
def price_difference_cents : ℝ := (budget_buys_price - frugal_finds_price) * 100

theorem camera_price_difference :
  price_difference_cents = 1099.15 := by
  sorry

end NUMINAMATH_CALUDE_camera_price_difference_l2599_259989


namespace NUMINAMATH_CALUDE_beverage_probabilities_l2599_259979

/-- The probability of a single bottle of X beverage being qualified -/
def p_qualified : ℝ := 0.8

/-- The number of people drinking the beverage -/
def num_people : ℕ := 3

/-- The number of bottles each person drinks -/
def bottles_per_person : ℕ := 2

/-- The probability that a person drinks two qualified bottles -/
def p_two_qualified : ℝ := p_qualified ^ bottles_per_person

/-- The probability that exactly two out of three people drink two qualified bottles -/
def p_two_out_of_three : ℝ := 
  (num_people.choose 2 : ℝ) * p_two_qualified ^ 2 * (1 - p_two_qualified) ^ (num_people - 2)

theorem beverage_probabilities :
  p_two_qualified = 0.64 ∧ p_two_out_of_three = 0.44 := by sorry

end NUMINAMATH_CALUDE_beverage_probabilities_l2599_259979


namespace NUMINAMATH_CALUDE_ann_bill_money_problem_l2599_259999

/-- Ann and Bill's money problem -/
theorem ann_bill_money_problem (bill_initial : ℕ) (transfer : ℕ) (ann_initial : ℕ) :
  bill_initial = 1111 →
  transfer = 167 →
  ann_initial + transfer = bill_initial - transfer →
  ann_initial = 777 := by
  sorry

end NUMINAMATH_CALUDE_ann_bill_money_problem_l2599_259999


namespace NUMINAMATH_CALUDE_quadratic_solution_product_l2599_259992

theorem quadratic_solution_product (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) → 
  (3 * q^2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = 122 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_product_l2599_259992


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2599_259916

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- Check if two lines are coincident -/
def are_coincident (l1 l2 : Line) : Prop :=
  are_parallel l1 l2 ∧ l1.a * l2.c = l2.a * l1.c

/-- The problem statement -/
theorem parallel_lines_a_value :
  ∃ (a : ℝ), 
    let l1 : Line := ⟨a, 3, 1⟩
    let l2 : Line := ⟨2, a+1, 1⟩
    are_parallel l1 l2 ∧ ¬are_coincident l1 l2 ∧ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2599_259916


namespace NUMINAMATH_CALUDE_quadratic_standard_form_l2599_259972

theorem quadratic_standard_form : 
  ∃ (a b c : ℝ), ∀ x, 5 * x^2 = 6 * x - 8 ↔ a * x^2 + b * x + c = 0 ∧ a = 5 ∧ b = -6 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_standard_form_l2599_259972


namespace NUMINAMATH_CALUDE_oplus_three_one_l2599_259942

-- Define the operation ⊕ for real numbers
def oplus (a b : ℝ) : ℝ := 3 * a + 4 * b

-- State the theorem
theorem oplus_three_one : oplus 3 1 = 13 := by
  sorry

end NUMINAMATH_CALUDE_oplus_three_one_l2599_259942


namespace NUMINAMATH_CALUDE_shopkeeper_bananas_l2599_259945

theorem shopkeeper_bananas (oranges : ℕ) (bananas : ℕ) : 
  oranges = 600 →
  (510 : ℝ) + 0.95 * bananas = 0.89 * (oranges + bananas) →
  bananas = 400 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_bananas_l2599_259945


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l2599_259932

/-- The solution set of the inequality |2x-a|+a≤4 -/
def SolutionSet (a : ℝ) : Set ℝ := {x : ℝ | |2*x - a| + a ≤ 4}

/-- The theorem stating that if the solution set of |2x-a|+a≤4 is {x|-1≤x≤2}, then a = 1 -/
theorem solution_set_implies_a_equals_one :
  SolutionSet 1 = {x : ℝ | -1 ≤ x ∧ x ≤ 2} → 1 = 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l2599_259932


namespace NUMINAMATH_CALUDE_existence_and_not_forall_l2599_259990

theorem existence_and_not_forall : 
  (∃ x : ℝ, x > 2) ∧ ¬(∀ x : ℝ, x^3 > x^2) :=
by
  sorry

end NUMINAMATH_CALUDE_existence_and_not_forall_l2599_259990


namespace NUMINAMATH_CALUDE_combination_permutation_equality_l2599_259936

theorem combination_permutation_equality (n : ℕ) (hn : n > 0) : 
  3 * (Nat.choose (2 * n) 3) = 5 * (Nat.factorial n / Nat.factorial (n - 3)) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_equality_l2599_259936


namespace NUMINAMATH_CALUDE_f_max_min_difference_l2599_259997

noncomputable def f (x : ℝ) := Real.exp (Real.sin x + Real.cos x) - (1/2) * Real.sin (2 * x)

theorem f_max_min_difference :
  (⨆ (x : ℝ), f x) - (⨅ (x : ℝ), f x) = Real.exp (Real.sqrt 2) - Real.exp (-Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_difference_l2599_259997


namespace NUMINAMATH_CALUDE_cannot_transform_to_target_l2599_259943

/-- Represents a parabola equation in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines a simple transformation of a parabola --/
inductive SimpleTransformation
  | right : SimpleTransformation  -- Move 2 units right
  | up : SimpleTransformation     -- Move 1 unit up

/-- Applies a simple transformation to a parabola --/
def applyTransformation (p : Parabola) (t : SimpleTransformation) : Parabola :=
  match t with
  | SimpleTransformation.right => { a := p.a, b := p.b - 2 * p.a, c := p.c + p.a }
  | SimpleTransformation.up => { a := p.a, b := p.b, c := p.c + 1 }

/-- Applies a sequence of simple transformations to a parabola --/
def applyTransformations (p : Parabola) (ts : List SimpleTransformation) : Parabola :=
  ts.foldl applyTransformation p

theorem cannot_transform_to_target : 
  ∀ (ts : List SimpleTransformation),
    ts.length = 2 → 
    applyTransformations { a := 1, b := 6, c := 5 } ts ≠ { a := 1, b := 0, c := 1 } :=
sorry

end NUMINAMATH_CALUDE_cannot_transform_to_target_l2599_259943


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_l2599_259920

theorem largest_divisor_of_m (m : ℕ+) (h : 54 ∣ m ^ 2) :
  ∃ (d : ℕ), d ∣ m ∧ d = 18 ∧ ∀ (k : ℕ), k ∣ m → k ≤ d :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_l2599_259920


namespace NUMINAMATH_CALUDE_tenth_term_is_one_over_120_l2599_259961

def a (n : ℕ+) : ℚ := 1 / (n * (n + 2))

theorem tenth_term_is_one_over_120 : a 10 = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_one_over_120_l2599_259961


namespace NUMINAMATH_CALUDE_four_times_angle_triangle_l2599_259947

theorem four_times_angle_triangle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  (α = 40 ∧ β = 4 * γ) ∨ (α = 40 ∧ γ = 4 * β) ∨ (β = 40 ∧ α = 4 * γ) →  -- One angle is 40° and another is 4 times the third
  ((β = 130 ∧ γ = 10) ∨ (β = 112 ∧ γ = 28)) ∨ 
  ((α = 130 ∧ γ = 10) ∨ (α = 112 ∧ γ = 28)) ∨ 
  ((α = 130 ∧ β = 10) ∨ (α = 112 ∧ β = 28)) :=
by sorry

end NUMINAMATH_CALUDE_four_times_angle_triangle_l2599_259947


namespace NUMINAMATH_CALUDE_symmetric_point_line_equation_l2599_259915

/-- Given points A and M, if B is symmetric to A with respect to M, 
    and line l passes through the origin and point B, 
    then the equation of line l is 7x + 5y = 0 -/
theorem symmetric_point_line_equation 
  (A : ℝ × ℝ) (M : ℝ × ℝ) (B : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  A = (3, 1) →
  M = (4, -3) →
  B.1 = 2 * M.1 - A.1 →
  B.2 = 2 * M.2 - A.2 →
  (0, 0) ∈ l →
  B ∈ l →
  ∀ (x y : ℝ), (x, y) ∈ l ↔ 7 * x + 5 * y = 0 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_line_equation_l2599_259915


namespace NUMINAMATH_CALUDE_magnitude_of_3_minus_4i_l2599_259984

theorem magnitude_of_3_minus_4i :
  Complex.abs (3 - 4*Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_3_minus_4i_l2599_259984


namespace NUMINAMATH_CALUDE_collector_problem_l2599_259948

/-- The number of items in the collection --/
def n : ℕ := 10

/-- The probability of finding each item --/
def p : ℝ := 0.1

/-- The probability of having exactly k items missing in the second collection
    when the first collection is complete --/
def prob_missing (k : ℕ) : ℝ := sorry

theorem collector_problem :
  (prob_missing 1 = prob_missing 2) ∧
  (∀ k ∈ Finset.range 9, prob_missing (k + 2) > prob_missing (k + 3)) :=
sorry

end NUMINAMATH_CALUDE_collector_problem_l2599_259948


namespace NUMINAMATH_CALUDE_glass_count_l2599_259900

/-- Given glasses with a capacity of 6 ounces that are 4/5 full, 
    prove that if 12 ounces of water are needed to fill all glasses, 
    there are 10 glasses. -/
theorem glass_count (glass_capacity : ℚ) (initial_fill : ℚ) (total_water_needed : ℚ) :
  glass_capacity = 6 →
  initial_fill = 4 / 5 →
  total_water_needed = 12 →
  (total_water_needed / (glass_capacity * (1 - initial_fill))) = 10 := by
  sorry


end NUMINAMATH_CALUDE_glass_count_l2599_259900


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_fifteen_zeros_l2599_259959

/-- The number of trailing zeros in n! when expressed in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

theorem fifteen_factorial_base_fifteen_zeros :
  trailingZeros 15 15 = 3 :=
sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_fifteen_zeros_l2599_259959


namespace NUMINAMATH_CALUDE_expression_equality_l2599_259991

theorem expression_equality : 40 + 5 * 12 / (180 / 3) = 41 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2599_259991


namespace NUMINAMATH_CALUDE_johns_initial_money_l2599_259929

theorem johns_initial_money (initial_amount : ℚ) : 
  (initial_amount * (1 - (3/8 + 3/10)) = 65) → initial_amount = 200 := by
  sorry

end NUMINAMATH_CALUDE_johns_initial_money_l2599_259929


namespace NUMINAMATH_CALUDE_square_side_length_l2599_259995

theorem square_side_length (perimeter area : ℝ) (h_perimeter : perimeter = 48) (h_area : area = 144) :
  ∃ (side : ℝ), side * 4 = perimeter ∧ side * side = area ∧ side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2599_259995


namespace NUMINAMATH_CALUDE_function_value_determines_parameter_l2599_259949

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3^x + 1 else x^2 + a*x

theorem function_value_determines_parameter (a : ℝ) : f a (f a 0) = 6 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_determines_parameter_l2599_259949


namespace NUMINAMATH_CALUDE_car_purchase_cost_difference_l2599_259923

/-- Calculates the difference in individual cost for buying a car when the group size changes --/
theorem car_purchase_cost_difference 
  (base_cost : ℕ) 
  (discount_per_person : ℕ) 
  (car_wash_earnings : ℕ) 
  (original_group_size : ℕ) 
  (new_group_size : ℕ) : 
  base_cost = 1700 →
  discount_per_person = 50 →
  car_wash_earnings = 500 →
  original_group_size = 6 →
  new_group_size = 5 →
  (base_cost - new_group_size * discount_per_person - car_wash_earnings) / new_group_size -
  (base_cost - original_group_size * discount_per_person - car_wash_earnings) / original_group_size = 40 := by
  sorry


end NUMINAMATH_CALUDE_car_purchase_cost_difference_l2599_259923


namespace NUMINAMATH_CALUDE_tire_circumference_constant_l2599_259930

/-- The circumference of a tire remains constant given car speed and tire rotation rate -/
theorem tire_circumference_constant
  (v : ℝ) -- Car speed in km/h
  (n : ℝ) -- Tire rotation rate in rpm
  (h1 : v = 120) -- Car speed is 120 km/h
  (h2 : n = 400) -- Tire rotation rate is 400 rpm
  : ∃ (C : ℝ), C = 5 ∧ ∀ (grade : ℝ), C = 5 := by
  sorry

end NUMINAMATH_CALUDE_tire_circumference_constant_l2599_259930


namespace NUMINAMATH_CALUDE_linear_regression_estimate_l2599_259924

/-- Given a linear regression equation y = 0.50x - 0.81, prove that when x = 25, y = 11.69 -/
theorem linear_regression_estimate (x y : ℝ) : 
  y = 0.50 * x - 0.81 → x = 25 → y = 11.69 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_estimate_l2599_259924


namespace NUMINAMATH_CALUDE_tax_rate_on_other_items_l2599_259965

-- Define the total amount spent (excluding taxes)
def total_amount : ℝ := 100

-- Define the percentages spent on each category
def clothing_percent : ℝ := 0.5
def food_percent : ℝ := 0.25
def other_percent : ℝ := 0.25

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.1
def food_tax_rate : ℝ := 0
def total_tax_rate : ℝ := 0.1

-- Define the amounts spent on each category
def clothing_amount : ℝ := total_amount * clothing_percent
def food_amount : ℝ := total_amount * food_percent
def other_amount : ℝ := total_amount * other_percent

-- Define the tax paid on clothing
def clothing_tax : ℝ := clothing_amount * clothing_tax_rate

-- Define the total tax paid
def total_tax : ℝ := total_amount * total_tax_rate

-- Define the tax paid on other items
def other_tax : ℝ := total_tax - clothing_tax

-- Theorem to prove
theorem tax_rate_on_other_items :
  other_tax / other_amount = 0.2 := by sorry

end NUMINAMATH_CALUDE_tax_rate_on_other_items_l2599_259965


namespace NUMINAMATH_CALUDE_sanchez_sum_problem_l2599_259967

theorem sanchez_sum_problem (x y : ℕ+) : x - y = 5 → x * y = 84 → x + y = 19 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_sum_problem_l2599_259967


namespace NUMINAMATH_CALUDE_gretchen_objects_l2599_259958

/-- The number of objects Gretchen can carry per trip -/
def objects_per_trip : ℕ := 3

/-- The number of trips Gretchen took -/
def number_of_trips : ℕ := 6

/-- The total number of objects Gretchen found underwater -/
def total_objects : ℕ := objects_per_trip * number_of_trips

theorem gretchen_objects : total_objects = 18 := by
  sorry

end NUMINAMATH_CALUDE_gretchen_objects_l2599_259958


namespace NUMINAMATH_CALUDE_alan_cd_purchase_cost_l2599_259964

/-- The price of a CD by "AVN" in dollars -/
def avnPrice : ℝ := 12

/-- The price of a CD by "The Dark" in dollars -/
def darkPrice : ℝ := 2 * avnPrice

/-- The cost of CDs by "The Dark" and "AVN" in dollars -/
def mainCost : ℝ := 2 * darkPrice + avnPrice

/-- The cost of 90s music CDs in dollars -/
def mixCost : ℝ := 0.4 * mainCost

/-- The total cost of Alan's purchase in dollars -/
def totalCost : ℝ := mainCost + mixCost

theorem alan_cd_purchase_cost :
  totalCost = 84 := by sorry

end NUMINAMATH_CALUDE_alan_cd_purchase_cost_l2599_259964


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2599_259921

theorem contrapositive_equivalence (a b m : ℝ) :
  (¬(a > b → a * (m^2 + 1) > b * (m^2 + 1))) ↔ (a * (m^2 + 1) ≤ b * (m^2 + 1) → a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2599_259921


namespace NUMINAMATH_CALUDE_hyperbola_slope_theorem_l2599_259974

/-- A hyperbola passing through specific points with given asymptote slopes -/
structure Hyperbola where
  -- Points the hyperbola passes through
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ
  point4 : ℝ × ℝ
  -- Slope of one asymptote
  slope1 : ℚ
  -- Slope of the other asymptote
  slope2 : ℚ
  -- Condition that the hyperbola passes through the given points
  passes_through : point1 = (2, 5) ∧ point2 = (7, 3) ∧ point3 = (1, 1) ∧ point4 = (10, 10)
  -- Condition that slope1 is 20/17
  slope1_value : slope1 = 20/17
  -- Condition that the product of slopes is -1
  slopes_product : slope1 * slope2 = -1

theorem hyperbola_slope_theorem (h : Hyperbola) :
  h.slope2 = -17/20 ∧ (100 * 17 + 20 = 1720) := by
  sorry

#check hyperbola_slope_theorem

end NUMINAMATH_CALUDE_hyperbola_slope_theorem_l2599_259974


namespace NUMINAMATH_CALUDE_twenty_solutions_implies_twenty_or_twentythree_l2599_259963

/-- Given a positive integer n, count_solutions n returns the number of solutions
    to the equation 3x + 3y + 2z = n in positive integers x, y, and z -/
def count_solutions (n : ℕ+) : ℕ :=
  sorry

theorem twenty_solutions_implies_twenty_or_twentythree (n : ℕ+) :
  count_solutions n = 20 → n = 20 ∨ n = 23 := by
  sorry

end NUMINAMATH_CALUDE_twenty_solutions_implies_twenty_or_twentythree_l2599_259963


namespace NUMINAMATH_CALUDE_nina_running_distance_l2599_259911

theorem nina_running_distance (x : ℝ) : 
  2 * x + 0.6666666666666666 = 0.8333333333333334 → 
  x = 0.08333333333333337 := by
  sorry

end NUMINAMATH_CALUDE_nina_running_distance_l2599_259911


namespace NUMINAMATH_CALUDE_smallest_candy_count_l2599_259998

theorem smallest_candy_count : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (7 ∣ (n + 6)) ∧ 
  (4 ∣ (n - 9)) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (7 ∣ (m + 6)) ∧ (4 ∣ (m - 9))) → False) ∧
  n = 113 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l2599_259998


namespace NUMINAMATH_CALUDE_rectangle_area_divisible_by_12_l2599_259905

theorem rectangle_area_divisible_by_12 (a b c : ℕ) 
  (h1 : a * a + b * b = c * c) : 
  12 ∣ (a * b) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_divisible_by_12_l2599_259905


namespace NUMINAMATH_CALUDE_range_of_t_range_of_a_l2599_259918

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3|

-- Part 1
theorem range_of_t (t : ℝ) : f t + f (2 * t) < 9 ↔ -1 < t ∧ t < 5 := by sorry

-- Part 2
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ f (2 * x) + |x + a| ≤ 3) ↔ a ∈ Set.Icc (-4) 0 := by sorry

end NUMINAMATH_CALUDE_range_of_t_range_of_a_l2599_259918


namespace NUMINAMATH_CALUDE_expression_evaluation_l2599_259910

theorem expression_evaluation : (50 - (2050 - 150)) + (2050 - (150 - 50)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2599_259910


namespace NUMINAMATH_CALUDE_mrs_hilt_friends_l2599_259913

/-- The number of friends carrying pears -/
def friends_with_pears : ℕ := 9

/-- The number of friends carrying oranges -/
def friends_with_oranges : ℕ := 6

/-- The total number of friends Mrs. Hilt met -/
def total_friends : ℕ := friends_with_pears + friends_with_oranges

theorem mrs_hilt_friends :
  total_friends = 15 := by sorry

end NUMINAMATH_CALUDE_mrs_hilt_friends_l2599_259913


namespace NUMINAMATH_CALUDE_complex_power_eight_l2599_259944

theorem complex_power_eight (a b : ℝ) (h : (a : ℂ) + Complex.I = 1 - b * Complex.I) :
  (a + b * Complex.I) ^ 8 = 16 := by sorry

end NUMINAMATH_CALUDE_complex_power_eight_l2599_259944


namespace NUMINAMATH_CALUDE_f_properties_l2599_259917

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 2 / (3^x + 1)

theorem f_properties :
  ∀ (a : ℝ),
  -- 1. Range of f when a = 1
  (∀ y : ℝ, y ∈ Set.range (f 1) ↔ 1 < y ∧ y < 3) ∧
  -- 2. f is strictly decreasing
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) ∧
  -- 3. If f is odd and f(f(x)) + f(m) < 0 has solutions, then m > -1
  (((∀ x : ℝ, f a (-x) = -f a x) ∧
    (∃ x : ℝ, f a (f a x) + f a m < 0)) → m > -1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2599_259917


namespace NUMINAMATH_CALUDE_intersection_of_sets_l2599_259962

theorem intersection_of_sets : 
  let A : Set ℤ := {0, 1, 2, 4}
  let B : Set ℤ := {-1, 0, 1, 3}
  A ∩ B = {0, 1} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l2599_259962


namespace NUMINAMATH_CALUDE_pentagonal_grid_toothpicks_l2599_259941

/-- The number of toothpicks in the base of the pentagonal grid -/
def base_toothpicks : ℕ := 10

/-- The number of toothpicks in each of the four non-base sides -/
def side_toothpicks : ℕ := 8

/-- The number of sides excluding the base -/
def num_sides : ℕ := 4

/-- The number of vertices in a pentagon -/
def num_vertices : ℕ := 5

/-- The total number of toothpicks needed for the framed pentagonal grid -/
def total_toothpicks : ℕ := base_toothpicks + num_sides * side_toothpicks + num_vertices

theorem pentagonal_grid_toothpicks : total_toothpicks = 47 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_grid_toothpicks_l2599_259941


namespace NUMINAMATH_CALUDE_spirits_bottle_cost_l2599_259956

/-- Calculates the cost of a bottle of spirits given the number of servings,
    price per serving, and profit per bottle. -/
def bottle_cost (servings : ℕ) (price_per_serving : ℚ) (profit : ℚ) : ℚ :=
  servings * price_per_serving - profit

/-- Proves that the cost of a bottle of spirits is $30.00 under given conditions. -/
theorem spirits_bottle_cost :
  bottle_cost 16 8 98 = 30 := by
  sorry

end NUMINAMATH_CALUDE_spirits_bottle_cost_l2599_259956


namespace NUMINAMATH_CALUDE_intersection_of_two_lines_l2599_259955

/-- The intersection point of two lines in a 2D plane -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Checks if a point satisfies the equation of a line -/
def satisfiesLine (p : IntersectionPoint) (a b c : ℝ) : Prop :=
  a * p.x + b * p.y = c

/-- The unique intersection point of two lines -/
def uniqueIntersection (p : IntersectionPoint) (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  satisfiesLine p a1 b1 c1 ∧
  satisfiesLine p a2 b2 c2 ∧
  ∀ q : IntersectionPoint, satisfiesLine q a1 b1 c1 ∧ satisfiesLine q a2 b2 c2 → q = p

theorem intersection_of_two_lines :
  uniqueIntersection ⟨3, -1⟩ 2 (-1) 7 3 2 7 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_two_lines_l2599_259955


namespace NUMINAMATH_CALUDE_papaya_problem_l2599_259940

def remaining_green_papayas (initial : Nat) (friday_yellow : Nat) : Nat :=
  initial - friday_yellow - (2 * friday_yellow)

theorem papaya_problem :
  remaining_green_papayas 14 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_papaya_problem_l2599_259940


namespace NUMINAMATH_CALUDE_unique_prime_in_form_l2599_259988

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def form_number (B A : ℕ) : ℕ := 210000 + B * 100 + A

theorem unique_prime_in_form :
  ∃! B : ℕ, B < 10 ∧ ∃ A : ℕ, A < 10 ∧ is_prime (form_number B A) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_in_form_l2599_259988


namespace NUMINAMATH_CALUDE_rectangle_area_l2599_259968

theorem rectangle_area (L W : ℝ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) :
  L * W = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2599_259968


namespace NUMINAMATH_CALUDE_probability_one_pair_one_triplet_proof_l2599_259977

/-- The probability of rolling six standard six-sided dice and getting exactly
    one pair, one triplet, and the remaining dice showing different values. -/
def probability_one_pair_one_triplet : ℚ := 25 / 162

/-- The number of possible outcomes when rolling six standard six-sided dice. -/
def total_outcomes : ℕ := 6^6

/-- The number of successful outcomes (one pair, one triplet, remaining different). -/
def successful_outcomes : ℕ := 7200

theorem probability_one_pair_one_triplet_proof :
  probability_one_pair_one_triplet = successful_outcomes / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_probability_one_pair_one_triplet_proof_l2599_259977


namespace NUMINAMATH_CALUDE_prob_odd_sum_is_two_thirds_l2599_259934

/-- A type representing the cards labeled 0, 1, and 2 -/
inductive Card : Type
  | zero : Card
  | one : Card
  | two : Card

/-- A function to convert a Card to its numerical value -/
def cardValue : Card → ℕ
  | Card.zero => 0
  | Card.one => 1
  | Card.two => 2

/-- A predicate to check if the sum of two cards is odd -/
def isSumOdd (c1 c2 : Card) : Prop :=
  Odd (cardValue c1 + cardValue c2)

/-- The set of all possible card combinations -/
def allCombinations : Finset (Card × Card) :=
  sorry

/-- The set of card combinations with odd sum -/
def oddSumCombinations : Finset (Card × Card) :=
  sorry

/-- Theorem stating the probability of drawing two cards with odd sum is 2/3 -/
theorem prob_odd_sum_is_two_thirds :
    (Finset.card oddSumCombinations : ℚ) / (Finset.card allCombinations : ℚ) = 2 / 3 :=
  sorry

end NUMINAMATH_CALUDE_prob_odd_sum_is_two_thirds_l2599_259934


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2599_259980

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1 / a₀ + 1 / b₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2599_259980


namespace NUMINAMATH_CALUDE_quadratic_coefficient_value_l2599_259976

theorem quadratic_coefficient_value (b : ℝ) (n : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 88 = (x + n)^2 + 16) → 
  b = 12 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_value_l2599_259976


namespace NUMINAMATH_CALUDE_infinitely_many_odd_terms_l2599_259971

theorem infinitely_many_odd_terms (n : ℕ) (hn : n > 1) :
  ∀ m : ℕ, ∃ k > m, Odd (⌊(n^k : ℝ) / k⌋) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_odd_terms_l2599_259971


namespace NUMINAMATH_CALUDE_range_of_a_l2599_259912

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x + 2 else a / x

/-- The range of f(x) is (0, +∞) -/
def has_range_open_zero_to_inf (a : ℝ) : Prop :=
  ∀ y > 0, ∃ x, f a x = y

/-- The range of a is [1, +∞) -/
def a_range_closed_one_to_inf (a : ℝ) : Prop :=
  a ≥ 1

/-- Theorem statement -/
theorem range_of_a (a : ℝ) :
  has_range_open_zero_to_inf a → a_range_closed_one_to_inf a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2599_259912


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2599_259966

theorem negation_of_universal_proposition :
  (¬ (∀ a : ℝ, a > 0 → Real.exp a ≥ 1)) ↔ (∃ a : ℝ, a > 0 ∧ Real.exp a < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2599_259966


namespace NUMINAMATH_CALUDE_remainder_problem_l2599_259986

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0)
  (h2 : k < 168)
  (h3 : k % 5 = 2)
  (h4 : k % 6 = 5)
  (h5 : k % 8 = 7)
  (h6 : k % 11 = 3) :
  k % 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2599_259986


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2599_259996

theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (Real.arctan ((b/a) / (1 - (b/a)^2)) * 2 = π / 4) →
  a / b = Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2599_259996


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l2599_259906

theorem abs_inequality_equivalence (x : ℝ) : 
  |2*x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l2599_259906


namespace NUMINAMATH_CALUDE_frosting_calculation_l2599_259901

/-- Calculates the total number of frosting cans needed for a bakery order -/
theorem frosting_calculation (layer_cake_frosting : ℝ) (single_item_frosting : ℝ) 
  (tiered_cake_frosting : ℝ) (mini_cupcake_pair_frosting : ℝ) 
  (layer_cakes : ℕ) (tiered_cakes : ℕ) (cupcake_dozens : ℕ) 
  (mini_cupcakes : ℕ) (single_cakes : ℕ) (brownie_pans : ℕ) :
  layer_cake_frosting = 1 →
  single_item_frosting = 0.5 →
  tiered_cake_frosting = 1.5 →
  mini_cupcake_pair_frosting = 0.25 →
  layer_cakes = 4 →
  tiered_cakes = 8 →
  cupcake_dozens = 10 →
  mini_cupcakes = 30 →
  single_cakes = 15 →
  brownie_pans = 24 →
  layer_cakes * layer_cake_frosting +
  tiered_cakes * tiered_cake_frosting +
  cupcake_dozens * single_item_frosting +
  (mini_cupcakes / 2) * mini_cupcake_pair_frosting +
  single_cakes * single_item_frosting +
  brownie_pans * single_item_frosting = 44.25 := by
sorry

end NUMINAMATH_CALUDE_frosting_calculation_l2599_259901


namespace NUMINAMATH_CALUDE_percentage_born_in_july_l2599_259937

theorem percentage_born_in_july (total : ℕ) (born_in_july : ℕ) 
  (h1 : total = 120) (h2 : born_in_july = 18) : 
  (born_in_july : ℚ) / (total : ℚ) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_born_in_july_l2599_259937


namespace NUMINAMATH_CALUDE_passion_fruit_crates_l2599_259939

theorem passion_fruit_crates (total grapes mangoes : ℕ) 
  (h1 : total = 50)
  (h2 : grapes = 13)
  (h3 : mangoes = 20) :
  total - (grapes + mangoes) = 17 := by
  sorry

end NUMINAMATH_CALUDE_passion_fruit_crates_l2599_259939


namespace NUMINAMATH_CALUDE_third_side_is_fifteen_l2599_259993

/-- A triangle with two known sides and perimeter -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  perimeter : ℝ

/-- Calculate the third side of a triangle given two sides and the perimeter -/
def thirdSide (t : Triangle) : ℝ :=
  t.perimeter - t.side1 - t.side2

/-- Theorem: The third side of the specific triangle is 15 -/
theorem third_side_is_fifteen : 
  let t : Triangle := { side1 := 7, side2 := 10, perimeter := 32 }
  thirdSide t = 15 := by
  sorry

end NUMINAMATH_CALUDE_third_side_is_fifteen_l2599_259993


namespace NUMINAMATH_CALUDE_special_sequence_2023_l2599_259983

/-- A sequence satisfying the given conditions -/
def special_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ m n : ℕ, m > 0 → n > 0 → a (m + n) = a m + a n

/-- The 2023rd term of the special sequence equals 6069 -/
theorem special_sequence_2023 (a : ℕ → ℕ) (h : special_sequence a) : a 2023 = 6069 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_2023_l2599_259983


namespace NUMINAMATH_CALUDE_supermarket_spending_difference_l2599_259931

/-- 
Given:
- initial_amount: The initial amount in Olivia's wallet
- atm_amount: The amount collected from the ATM
- final_amount: The amount left after visiting the supermarket

Prove that the difference between the amount spent at the supermarket
and the amount collected from the ATM is 39 dollars.
-/
theorem supermarket_spending_difference 
  (initial_amount atm_amount final_amount : ℕ) 
  (h1 : initial_amount = 53)
  (h2 : atm_amount = 91)
  (h3 : final_amount = 14) :
  (initial_amount + atm_amount - final_amount) - atm_amount = 39 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_difference_l2599_259931


namespace NUMINAMATH_CALUDE_expected_value_S_squared_l2599_259903

/-- ω is a primitive 2018th root of unity -/
def ω : ℂ :=
  sorry

/-- The set of complex numbers from which subsets are chosen -/
def complexSet : Finset ℂ :=
  sorry

/-- S is the sum of elements in a randomly chosen subset of complexSet -/
def S : Finset ℂ → ℂ :=
  sorry

/-- The expected value of |S|² -/
def expectedValueS : ℝ :=
  sorry

theorem expected_value_S_squared :
  expectedValueS = 1009 / 2 :=
sorry

end NUMINAMATH_CALUDE_expected_value_S_squared_l2599_259903


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2599_259973

theorem quadratic_roots_property (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) →
  (3 * e^2 + 4 * e - 7 = 0) →
  (d - 2) * (e - 2) = 13/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2599_259973


namespace NUMINAMATH_CALUDE_symmetry_condition_l2599_259925

/-- Given a curve y = (2px + q) / (rx - 2s) where p, q, r, s are nonzero real numbers,
    if the line y = x is an axis of symmetry for this curve, then r - 2s = 0. -/
theorem symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (2*p*x + q) / (r*x - 2*s) ↔ x = (2*p*y + q) / (r*y - 2*s)) →
  r - 2*s = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_condition_l2599_259925


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2599_259970

theorem isosceles_triangle_perimeter : ∀ (a b : ℝ),
  a^2 - 9*a + 18 = 0 →
  b^2 - 9*b + 18 = 0 →
  a ≠ b →
  (∃ (leg base : ℝ), (leg = max a b ∧ base = min a b) ∧
    2*leg + base = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2599_259970


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a2_l2599_259981

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a2 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 3 + a 11 = 50) 
  (h_a4 : a 4 = 13) : 
  a 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a2_l2599_259981


namespace NUMINAMATH_CALUDE_inequality_theorem_l2599_259951

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (h₁ : x₁ * y₁ - z₁^2 > 0) (h₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ - z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ∧
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ - z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔
   x₁ * y₁ * x₂ * y₂ - z₁^2 * x₂^2 - z₂^2 * x₁^2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_inequality_theorem_l2599_259951


namespace NUMINAMATH_CALUDE_diamonds_in_tenth_figure_l2599_259907

/-- The number of diamonds in the outer circle of the nth figure -/
def outer_diamonds (n : ℕ) : ℕ := 4 + 6 * (n - 1)

/-- The total number of diamonds in the nth figure -/
def total_diamonds (n : ℕ) : ℕ := 3 * n^2 + n

theorem diamonds_in_tenth_figure : total_diamonds 10 = 310 := by sorry

end NUMINAMATH_CALUDE_diamonds_in_tenth_figure_l2599_259907


namespace NUMINAMATH_CALUDE_wheat_cost_is_30_l2599_259908

/-- Represents the farm's cultivation scenario -/
structure FarmScenario where
  totalLand : ℕ
  cornCost : ℕ
  totalBudget : ℕ
  wheatAcres : ℕ

/-- Calculates the cost of wheat cultivation per acre -/
def wheatCostPerAcre (scenario : FarmScenario) : ℕ :=
  (scenario.totalBudget - (scenario.cornCost * (scenario.totalLand - scenario.wheatAcres))) / scenario.wheatAcres

/-- Theorem stating the cost of wheat cultivation per acre is 30 -/
theorem wheat_cost_is_30 (scenario : FarmScenario) 
    (h1 : scenario.totalLand = 500)
    (h2 : scenario.cornCost = 42)
    (h3 : scenario.totalBudget = 18600)
    (h4 : scenario.wheatAcres = 200) :
  wheatCostPerAcre scenario = 30 := by
  sorry

#eval wheatCostPerAcre { totalLand := 500, cornCost := 42, totalBudget := 18600, wheatAcres := 200 }

end NUMINAMATH_CALUDE_wheat_cost_is_30_l2599_259908


namespace NUMINAMATH_CALUDE_max_value_of_C_l2599_259960

theorem max_value_of_C (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a := 1 / y
  let b := y + 1 / x
  let C := min x (min a b)
  ∀ ε > 0, C ≤ Real.sqrt 2 + ε ∧ ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧
    let a' := 1 / y'
    let b' := y' + 1 / x'
    let C' := min x' (min a' b')
    C' > Real.sqrt 2 - ε :=
sorry

end NUMINAMATH_CALUDE_max_value_of_C_l2599_259960


namespace NUMINAMATH_CALUDE_plane_line_relations_l2599_259928

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (lineparallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Axioms
axiom parallel_trans {α β γ : Plane} : parallel α β → parallel β γ → parallel α γ
axiom perpendicular_trans {l m : Line} {α β : Plane} : 
  perpendicular l α → perpendicular l β → perpendicular m α → perpendicular m β

-- Theorem
theorem plane_line_relations 
  (α β : Plane) (m n : Line) 
  (h_diff_planes : α ≠ β) 
  (h_diff_lines : m ≠ n) :
  (parallel α β ∧ contains α m → lineparallel m β) ∧
  (perpendicular n α ∧ perpendicular n β ∧ perpendicular m α → perpendicular m β) :=
by sorry

end NUMINAMATH_CALUDE_plane_line_relations_l2599_259928
