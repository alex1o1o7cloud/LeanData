import Mathlib

namespace NUMINAMATH_CALUDE_max_value_cube_sum_ratio_l2578_257850

theorem max_value_cube_sum_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^3 / (x^3 + y^3 + z^3) ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cube_sum_ratio_l2578_257850


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_eight_gcd_of_136_and_16_is_136_less_than_150_main_result_l2578_257832

theorem greatest_integer_with_gcd_eight (n : ℕ) : n < 150 ∧ n.gcd 16 = 8 → n ≤ 136 :=
by sorry

theorem gcd_of_136_and_16 : Nat.gcd 136 16 = 8 :=
by sorry

theorem is_136_less_than_150 : 136 < 150 :=
by sorry

theorem main_result : ∃ (n : ℕ), n < 150 ∧ n.gcd 16 = 8 ∧ 
  ∀ (m : ℕ), m < 150 ∧ m.gcd 16 = 8 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_eight_gcd_of_136_and_16_is_136_less_than_150_main_result_l2578_257832


namespace NUMINAMATH_CALUDE_tshirt_company_profit_l2578_257875

/-- Calculates the daily profit of a t-shirt company given specific conditions -/
theorem tshirt_company_profit (
  num_employees : ℕ
  ) (shirts_per_employee : ℕ
  ) (shift_hours : ℕ
  ) (hourly_wage : ℚ
  ) (per_shirt_bonus : ℚ
  ) (shirt_price : ℚ
  ) (nonemployee_expenses : ℚ
  ) (h1 : num_employees = 20
  ) (h2 : shirts_per_employee = 20
  ) (h3 : shift_hours = 8
  ) (h4 : hourly_wage = 12
  ) (h5 : per_shirt_bonus = 5
  ) (h6 : shirt_price = 35
  ) (h7 : nonemployee_expenses = 1000
  ) : (num_employees * shirts_per_employee * shirt_price) -
      (num_employees * shift_hours * hourly_wage +
       num_employees * shirts_per_employee * per_shirt_bonus +
       nonemployee_expenses) = 9080 := by
  sorry


end NUMINAMATH_CALUDE_tshirt_company_profit_l2578_257875


namespace NUMINAMATH_CALUDE_log_difference_divided_l2578_257844

theorem log_difference_divided : (Real.log 1 - Real.log 25) / 100 = -20 := by sorry

end NUMINAMATH_CALUDE_log_difference_divided_l2578_257844


namespace NUMINAMATH_CALUDE_Q_subset_P_l2578_257868

-- Define the sets P and Q
def P : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def Q : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem Q_subset_P : Q ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_Q_subset_P_l2578_257868


namespace NUMINAMATH_CALUDE_joseph_total_distance_l2578_257840

/-- The total distance Joseph ran over 3 days, given he ran 900 meters each day. -/
def total_distance (distance_per_day : ℕ) (days : ℕ) : ℕ :=
  distance_per_day * days

/-- Theorem stating that Joseph ran 2700 meters in total. -/
theorem joseph_total_distance :
  total_distance 900 3 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_joseph_total_distance_l2578_257840


namespace NUMINAMATH_CALUDE_no_integer_cube_equals_3n2_plus_3n_plus_7_l2578_257847

theorem no_integer_cube_equals_3n2_plus_3n_plus_7 :
  ¬ ∃ (m n : ℤ), m^3 = 3*n^2 + 3*n + 7 := by
sorry

end NUMINAMATH_CALUDE_no_integer_cube_equals_3n2_plus_3n_plus_7_l2578_257847


namespace NUMINAMATH_CALUDE_fourth_month_sale_l2578_257889

def average_sale : ℕ := 6600
def num_months : ℕ := 6
def sale_month1 : ℕ := 6435
def sale_month2 : ℕ := 6927
def sale_month3 : ℕ := 6855
def sale_month5 : ℕ := 6562
def sale_month6 : ℕ := 5591

theorem fourth_month_sale (x : ℕ) : 
  (sale_month1 + sale_month2 + sale_month3 + x + sale_month5 + sale_month6) / num_months = average_sale →
  x = 7230 := by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l2578_257889


namespace NUMINAMATH_CALUDE_service_fee_is_24_percent_l2578_257841

/-- Calculates the service fee percentage given the cost of food, tip, and total amount spent. -/
def service_fee_percentage (food_cost tip total_spent : ℚ) : ℚ :=
  ((total_spent - food_cost - tip) / food_cost) * 100

/-- Theorem stating that the service fee percentage is 24% given the problem conditions. -/
theorem service_fee_is_24_percent :
  let food_cost : ℚ := 50
  let tip : ℚ := 5
  let total_spent : ℚ := 61
  service_fee_percentage food_cost tip total_spent = 24 := by
  sorry

end NUMINAMATH_CALUDE_service_fee_is_24_percent_l2578_257841


namespace NUMINAMATH_CALUDE_twelve_foldable_configurations_l2578_257891

/-- Represents a position on the periphery of the cross-shaped arrangement -/
inductive PeripheryPosition
| Top
| Right
| Bottom
| Left

/-- Represents the cross-shaped arrangement of 5 squares with an additional square -/
structure CrossArrangement :=
  (additional_square_position : PeripheryPosition)
  (additional_square_offset : Fin 3)

/-- Predicate to determine if a given arrangement can be folded into a cube with one face open -/
def can_fold_to_cube (arrangement : CrossArrangement) : Prop :=
  sorry

/-- The main theorem stating that exactly 12 configurations can be folded into a cube -/
theorem twelve_foldable_configurations :
  (∃ (configurations : Finset CrossArrangement),
    configurations.card = 12 ∧
    (∀ c ∈ configurations, can_fold_to_cube c) ∧
    (∀ c : CrossArrangement, can_fold_to_cube c → c ∈ configurations)) :=
sorry

end NUMINAMATH_CALUDE_twelve_foldable_configurations_l2578_257891


namespace NUMINAMATH_CALUDE_impossible_tiling_l2578_257882

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (is_square : size * size = size^2)

/-- Represents a tile -/
structure Tile :=
  (length : Nat)
  (width : Nat)

/-- Represents a tiling configuration -/
structure TilingConfiguration :=
  (board : Chessboard)
  (tile : Tile)
  (num_tiles : Nat)
  (central_square_uncovered : Bool)

/-- Main theorem: Impossibility of specific tiling -/
theorem impossible_tiling (config : TilingConfiguration) : 
  config.board.size = 13 ∧ 
  config.tile.length = 4 ∧ 
  config.tile.width = 1 ∧
  config.num_tiles = 42 ∧
  config.central_square_uncovered = true
  → False :=
sorry

end NUMINAMATH_CALUDE_impossible_tiling_l2578_257882


namespace NUMINAMATH_CALUDE_original_fraction_l2578_257842

theorem original_fraction (x y : ℚ) : 
  (x * (1 + 12/100)) / (y * (1 - 2/100)) = 6/7 → x/y = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_l2578_257842


namespace NUMINAMATH_CALUDE_gingers_garden_water_usage_l2578_257805

/-- Represents the problem of calculating water usage in Ginger's garden --/
theorem gingers_garden_water_usage 
  (hours_worked : ℕ) 
  (bottle_capacity : ℕ) 
  (total_water_used : ℕ) 
  (h1 : hours_worked = 8)
  (h2 : bottle_capacity = 2)
  (h3 : total_water_used = 26) :
  (total_water_used - hours_worked * bottle_capacity) / bottle_capacity = 5 := by
  sorry

#check gingers_garden_water_usage

end NUMINAMATH_CALUDE_gingers_garden_water_usage_l2578_257805


namespace NUMINAMATH_CALUDE_mean_temperature_is_88_point_2_l2578_257879

def temperatures : List ℝ := [78, 80, 82, 85, 88, 90, 92, 95, 97, 95]

theorem mean_temperature_is_88_point_2 :
  (temperatures.sum / temperatures.length : ℝ) = 88.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_88_point_2_l2578_257879


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2578_257811

theorem rectangle_perimeter (square_side : ℝ) (rect_length rect_breadth : ℝ) :
  square_side = 8 →
  rect_length = 8 →
  rect_breadth = 4 →
  let new_length := square_side + rect_length
  let new_breadth := square_side
  2 * (new_length + new_breadth) = 48 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2578_257811


namespace NUMINAMATH_CALUDE_negation_of_exists_lt_is_forall_ge_l2578_257806

theorem negation_of_exists_lt_is_forall_ge :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_lt_is_forall_ge_l2578_257806


namespace NUMINAMATH_CALUDE_inequality_proof_l2578_257855

theorem inequality_proof (p : ℝ) (x y z v : ℝ) 
  (hp : p ≥ 2) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hv : v ≥ 0) :
  (x + y)^p + (z + v)^p + (x + z)^p + (y + v)^p ≤ 
  x^p + y^p + z^p + v^p + (x + y + z + v)^p :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2578_257855


namespace NUMINAMATH_CALUDE_cannot_obtain_1998_pow_7_initial_condition_final_not_divisible_main_result_l2578_257830

def board_operation (n : ℕ) : ℕ :=
  let last_digit := n % 10
  (n / 10) + 5 * last_digit

theorem cannot_obtain_1998_pow_7 (n : ℕ) (h : 7 ∣ n) :
  ∀ k : ℕ, 7 ∣ (board_operation^[k] n) ∧ (board_operation^[k] n) ≠ 1998^7 :=
by sorry

theorem initial_condition : 7 ∣ 7^1998 :=
by sorry

theorem final_not_divisible : ¬(7 ∣ 1998^7) :=
by sorry

theorem main_result : ∀ k : ℕ, (board_operation^[k] 7^1998) ≠ 1998^7 :=
by sorry

end NUMINAMATH_CALUDE_cannot_obtain_1998_pow_7_initial_condition_final_not_divisible_main_result_l2578_257830


namespace NUMINAMATH_CALUDE_hot_dog_packs_l2578_257803

theorem hot_dog_packs (n : ℕ) : 
  (∃ m : ℕ, m < n ∧ 12 * m ≡ 6 [MOD 8]) → 
  12 * n ≡ 6 [MOD 8] → 
  (∀ k : ℕ, k < n → k ≠ n → 12 * k ≡ 6 [MOD 8] → 
    (∃ l : ℕ, l < k ∧ 12 * l ≡ 6 [MOD 8])) → 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_hot_dog_packs_l2578_257803


namespace NUMINAMATH_CALUDE_mikes_work_hours_l2578_257819

/-- Given that Mike worked for a total of 15 hours over 5 days, 
    prove that he worked 3 hours each day. -/
theorem mikes_work_hours (total_hours : ℕ) (total_days : ℕ) 
  (h1 : total_hours = 15) (h2 : total_days = 5) :
  total_hours / total_days = 3 := by
  sorry

end NUMINAMATH_CALUDE_mikes_work_hours_l2578_257819


namespace NUMINAMATH_CALUDE_youngest_child_age_problem_l2578_257845

/-- The age of the youngest child given the conditions of the problem -/
def youngest_child_age (n : ℕ) (interval : ℕ) (total_age : ℕ) : ℕ :=
  (total_age - (n - 1) * n * interval / 2) / n

/-- Theorem stating the age of the youngest child under the given conditions -/
theorem youngest_child_age_problem :
  youngest_child_age 5 2 50 = 6 := by sorry

end NUMINAMATH_CALUDE_youngest_child_age_problem_l2578_257845


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2578_257804

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 1) % 12 = 0 ∧ 
  (n + 1) % 18 = 0 ∧ 
  (n + 1) % 24 = 0 ∧ 
  (n + 1) % 32 = 0 ∧ 
  (n + 1) % 40 = 0

theorem smallest_number_divisible_by_all : 
  is_divisible_by_all 2879 ∧ ∀ m < 2879, ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2578_257804


namespace NUMINAMATH_CALUDE_two_face_painted_count_l2578_257863

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  is_painted : Bool

/-- Represents a cube that has been cut into unit cubes -/
structure CutCube (n : ℕ) extends PaintedCube n where
  unit_cubes : Fin n → Fin n → Fin n → PaintedCube 1

/-- Returns the number of unit cubes with exactly two painted faces -/
def count_two_face_painted (c : CutCube 4) : ℕ := sorry

/-- Theorem stating that a 4-inch painted cube cut into 1-inch cubes has 24 cubes with exactly two painted faces -/
theorem two_face_painted_count (c : CutCube 4) : count_two_face_painted c = 24 := by sorry

end NUMINAMATH_CALUDE_two_face_painted_count_l2578_257863


namespace NUMINAMATH_CALUDE_new_basis_from_old_l2578_257848

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem new_basis_from_old (a b c : V) 
  (h : LinearIndependent ℝ ![a, b, c]) 
  (h_span : Submodule.span ℝ {a, b, c} = ⊤) :
  LinearIndependent ℝ ![a + b, b + c, c + a] ∧ 
  Submodule.span ℝ {a + b, b + c, c + a} = ⊤ := by
sorry

end NUMINAMATH_CALUDE_new_basis_from_old_l2578_257848


namespace NUMINAMATH_CALUDE_floor_times_self_162_l2578_257877

theorem floor_times_self_162 (x : ℝ) : ⌊x⌋ * x = 162 → x = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_162_l2578_257877


namespace NUMINAMATH_CALUDE_floor_rate_per_square_meter_l2578_257860

/-- Given a rectangular room with length 8 m and width 4.75 m, and a total flooring cost of Rs. 34,200, the rate per square meter is Rs. 900. -/
theorem floor_rate_per_square_meter :
  let length : ℝ := 8
  let width : ℝ := 4.75
  let total_cost : ℝ := 34200
  let area : ℝ := length * width
  let rate_per_sq_meter : ℝ := total_cost / area
  rate_per_sq_meter = 900 := by sorry

end NUMINAMATH_CALUDE_floor_rate_per_square_meter_l2578_257860


namespace NUMINAMATH_CALUDE_locus_of_P_l2578_257865

/-- The locus of point P given the conditions in the problem -/
theorem locus_of_P (F Q T P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  F = (2, 0) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ (k : ℝ), y = k * (x - 2)) →
  Q.1 = 0 →
  Q ∈ l →
  (T.2 = 0 ∧ (Q.1 - T.1) * (F.1 - Q.1) = (F.2 - Q.2) * (Q.2 - T.2)) →
  (T.1 - Q.1)^2 + (T.2 - Q.2)^2 = (P.1 - Q.1)^2 + (P.2 - Q.2)^2 →
  P.2^2 = 8 * P.1 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_P_l2578_257865


namespace NUMINAMATH_CALUDE_torn_sheets_count_l2578_257828

/-- Represents a book with consecutively numbered pages. -/
structure Book where
  first_torn_page : Nat
  last_torn_page : Nat

/-- Checks if two numbers have the same digits. -/
def same_digits (a b : Nat) : Prop :=
  sorry

/-- Calculates the number of torn sheets given a Book. -/
def torn_sheets (book : Book) : Nat :=
  (book.last_torn_page - book.first_torn_page + 1) / 2

/-- The main theorem stating the number of torn sheets. -/
theorem torn_sheets_count (book : Book) :
    book.first_torn_page = 185
  → same_digits book.first_torn_page book.last_torn_page
  → Even book.last_torn_page
  → book.last_torn_page > book.first_torn_page
  → torn_sheets book = 167 := by
  sorry

end NUMINAMATH_CALUDE_torn_sheets_count_l2578_257828


namespace NUMINAMATH_CALUDE_equation_solution_l2578_257873

theorem equation_solution (x y : ℝ) (h1 : x ≠ 0) (h2 : 2*x + y ≠ 0) 
  (h3 : (x + y) / x = y / (2*x + y)) : x = -y/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2578_257873


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2578_257820

theorem complex_equation_solution (z : ℂ) (h : (1 + 3*Complex.I)*z = Complex.I - 3) : z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2578_257820


namespace NUMINAMATH_CALUDE_count_special_numbers_l2578_257846

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def satisfies_condition (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n = a + b^2 + c^3

theorem count_special_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ satisfies_condition n) ∧
                    (∀ n, is_three_digit n → satisfies_condition n → n ∈ S) ∧
                    Finset.card S = 4 :=
sorry

end NUMINAMATH_CALUDE_count_special_numbers_l2578_257846


namespace NUMINAMATH_CALUDE_difference_of_squares_l2578_257839

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2578_257839


namespace NUMINAMATH_CALUDE_unique_base_conversion_l2578_257851

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : Nat) : Nat :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base b to base 10 -/
def baseBToBase10 (n : Nat) (b : Nat) : Nat :=
  (n / 100) * b^2 + ((n / 10) % 10) * b + (n % 10)

theorem unique_base_conversion :
  ∃! (b : Nat), b > 0 ∧ base6ToBase10 125 = baseBToBase10 221 b :=
by sorry

end NUMINAMATH_CALUDE_unique_base_conversion_l2578_257851


namespace NUMINAMATH_CALUDE_quadratic_equation_in_y_l2578_257817

theorem quadratic_equation_in_y (x y : ℝ) 
  (eq1 : 3 * x^2 + 5 * x + 4 * y + 2 = 0)
  (eq2 : 3 * x + y + 4 = 0) : 
  y^2 + 15 * y + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_in_y_l2578_257817


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2578_257862

theorem inequality_solution_set (x : ℝ) :
  (3 - 2*x - x^2 ≤ 0) ↔ (x ≤ -3 ∨ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2578_257862


namespace NUMINAMATH_CALUDE_truck_speed_problem_l2578_257824

theorem truck_speed_problem (v : ℝ) : 
  v > 0 →  -- Truck speed is positive
  (60 * 4 = v * 5) →  -- Car catches up after 4 hours
  v = 48 := by
sorry

end NUMINAMATH_CALUDE_truck_speed_problem_l2578_257824


namespace NUMINAMATH_CALUDE_solve_for_x_l2578_257849

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2578_257849


namespace NUMINAMATH_CALUDE_bonnets_per_orphanage_l2578_257821

/-- The number of bonnets made on Monday -/
def monday_bonnets : ℕ := 10

/-- The number of bonnets made on Tuesday and Wednesday combined -/
def tuesday_wednesday_bonnets : ℕ := 2 * monday_bonnets

/-- The number of bonnets made on Thursday -/
def thursday_bonnets : ℕ := monday_bonnets + 5

/-- The number of bonnets made on Friday -/
def friday_bonnets : ℕ := thursday_bonnets - 5

/-- The total number of bonnets made -/
def total_bonnets : ℕ := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets

/-- The number of orphanages -/
def num_orphanages : ℕ := 5

/-- Theorem stating the number of bonnets sent to each orphanage -/
theorem bonnets_per_orphanage : total_bonnets / num_orphanages = 11 := by
  sorry

end NUMINAMATH_CALUDE_bonnets_per_orphanage_l2578_257821


namespace NUMINAMATH_CALUDE_direct_variation_problem_l2578_257866

-- Define the direct variation relationship
def direct_variation (y x : ℝ) := ∃ k : ℝ, y = k * x

-- State the theorem
theorem direct_variation_problem :
  ∀ y : ℝ → ℝ,
  (∀ x : ℝ, direct_variation (y x) x) →
  y 4 = 8 →
  y (-8) = -16 :=
by
  sorry

end NUMINAMATH_CALUDE_direct_variation_problem_l2578_257866


namespace NUMINAMATH_CALUDE_shen_win_probability_correct_l2578_257812

/-- Represents a player in the game -/
inductive Player
| Shen
| Ling
| Ru

/-- The number of slips each player puts in the bucket initially -/
def initial_slips : Nat := 4

/-- The total number of slips in the bucket -/
def total_slips : Nat := 13

/-- The number of slips Shen needs to win -/
def shen_win_condition : Nat := 4

/-- Calculates the probability of Shen winning the game -/
def shen_win_probability : Rat :=
  67 / 117

/-- Theorem stating that the calculated probability is correct -/
theorem shen_win_probability_correct :
  shen_win_probability = 67 / 117 := by sorry

end NUMINAMATH_CALUDE_shen_win_probability_correct_l2578_257812


namespace NUMINAMATH_CALUDE_square_of_sum_l2578_257829

theorem square_of_sum (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l2578_257829


namespace NUMINAMATH_CALUDE_shortest_side_is_15_l2578_257801

/-- Represents a triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  perimeter_eq : a + b + c = 72
  side_eq : a = 30

/-- Calculates the semiperimeter of a triangle -/
def semiperimeter (t : IntTriangle) : ℚ :=
  (t.a + t.b + t.c) / 2

/-- Calculates the area of a triangle using Heron's formula -/
def area (t : IntTriangle) : ℚ :=
  let s := semiperimeter t
  (s * (s - t.a) * (s - t.b) * (s - t.c)).sqrt

/-- Main theorem: The shortest side of the triangle is 15 -/
theorem shortest_side_is_15 (t : IntTriangle) (area_int : ∃ n : ℕ, area t = n) :
  min t.a (min t.b t.c) = 15 := by
  sorry

#check shortest_side_is_15

end NUMINAMATH_CALUDE_shortest_side_is_15_l2578_257801


namespace NUMINAMATH_CALUDE_brainiacs_liking_neither_count_l2578_257881

/-- The number of brainiacs who like neither rebus teasers nor math teasers -/
def brainiacs_liking_neither (total : ℕ) (rebus : ℕ) (math : ℕ) (both : ℕ) : ℕ :=
  total - (rebus + math - both)

/-- Theorem stating the number of brainiacs liking neither type of teaser -/
theorem brainiacs_liking_neither_count :
  let total := 100
  let rebus := 2 * math
  let both := 18
  let math_not_rebus := 20
  let math := both + math_not_rebus
  brainiacs_liking_neither total rebus math both = 4 := by
  sorry

#eval brainiacs_liking_neither 100 76 38 18

end NUMINAMATH_CALUDE_brainiacs_liking_neither_count_l2578_257881


namespace NUMINAMATH_CALUDE_max_cube_sum_under_constraints_l2578_257823

theorem max_cube_sum_under_constraints {a b c d : ℝ} 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 20)
  (sum_linear : a + b + c + d = 10) :
  a^3 + b^3 + c^3 + d^3 ≤ 500 ∧ 
  ∃ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 = 20 ∧ 
                   x + y + z + w = 10 ∧ 
                   x^3 + y^3 + z^3 + w^3 = 500 :=
by sorry

end NUMINAMATH_CALUDE_max_cube_sum_under_constraints_l2578_257823


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2578_257813

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2578_257813


namespace NUMINAMATH_CALUDE_ranch_minimum_animals_l2578_257869

theorem ranch_minimum_animals (ponies horses : ℕ) : 
  ponies > 0 →
  horses = ponies + 3 →
  ∃ (ponies_with_horseshoes icelandic_ponies : ℕ),
    ponies_with_horseshoes = (3 * ponies) / 10 ∧
    icelandic_ponies = (5 * ponies_with_horseshoes) / 8 →
  ponies + horses ≥ 35 :=
by
  sorry

end NUMINAMATH_CALUDE_ranch_minimum_animals_l2578_257869


namespace NUMINAMATH_CALUDE_integer_sum_problem_l2578_257852

theorem integer_sum_problem (x y : ℕ+) 
  (h1 : x.val - y.val = 8) 
  (h2 : x.val * y.val = 120) : 
  x.val + y.val = 2 * Real.sqrt 136 := by
sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l2578_257852


namespace NUMINAMATH_CALUDE_integer_triple_divisibility_l2578_257872

theorem integer_triple_divisibility :
  ∀ a b c : ℤ,
    1 < a ∧ a < b ∧ b < c →
    (a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1 →
    ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end NUMINAMATH_CALUDE_integer_triple_divisibility_l2578_257872


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2578_257833

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  2 * X^4 + 10 * X^3 - 45 * X^2 - 55 * X + 52 = 
  (X^2 + 8 * X - 6) * q + (-211 * X + 142) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2578_257833


namespace NUMINAMATH_CALUDE_distance_to_directrix_l2578_257809

/-- The distance from a point on a parabola to its directrix -/
theorem distance_to_directrix (p : ℝ) (h : p > 0) : 
  let A : ℝ × ℝ := (1, Real.sqrt 5)
  let C := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  A ∈ C → |1 - (-p/2)| = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_directrix_l2578_257809


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l2578_257857

/-- Proves that the ratio of Rahul's age to Deepak's age is 4:3 -/
theorem rahul_deepak_age_ratio : 
  ∀ (rahul_age deepak_age : ℕ),
  deepak_age = 12 →
  rahul_age + 10 = 26 →
  (rahul_age : ℚ) / (deepak_age : ℚ) = 4 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l2578_257857


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2578_257893

/-- Given a geometric series with first term a and common ratio r -/
def geometric_series (a r : ℝ) : ℕ → ℝ := fun n => a * r^n

/-- Sum of the geometric series up to infinity -/
def series_sum (a r : ℝ) : ℝ := 24

/-- Sum of terms with odd powers of r -/
def odd_powers_sum (a r : ℝ) : ℝ := 9

/-- Theorem: If the sum of a geometric series is 24 and the sum of terms with odd powers of r is 9, then r = 3/5 -/
theorem geometric_series_ratio (a r : ℝ) (h1 : series_sum a r = 24) (h2 : odd_powers_sum a r = 9) :
  r = 3/5 := by sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2578_257893


namespace NUMINAMATH_CALUDE_black_region_area_l2578_257892

theorem black_region_area (large_square_side : ℝ) (small_square_side : ℝ) :
  large_square_side = 10 →
  small_square_side = 4 →
  (large_square_side ^ 2) - 2 * (small_square_side ^ 2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_black_region_area_l2578_257892


namespace NUMINAMATH_CALUDE_function_domain_range_equality_l2578_257859

theorem function_domain_range_equality (a : ℝ) (h1 : a > 1) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*a*x + 5
  (∀ x, f x ∈ Set.Icc 1 a ↔ x ∈ Set.Icc 1 a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_domain_range_equality_l2578_257859


namespace NUMINAMATH_CALUDE_chord_intersection_angle_l2578_257896

theorem chord_intersection_angle (θ : Real) : 
  θ ∈ Set.Icc 0 (Real.pi / 2) →
  (∃ (x y : Real), 
    x * Real.sin θ + y * Real.cos θ - 1 = 0 ∧
    (x - 1)^2 + (y - Real.cos θ)^2 = 1/4 ∧
    ∃ (x' y' : Real), 
      x' * Real.sin θ + y' * Real.cos θ - 1 = 0 ∧
      (x' - 1)^2 + (y' - Real.cos θ)^2 = 1/4 ∧
      (x - x')^2 + (y - y')^2 = 3/4) →
  θ = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_chord_intersection_angle_l2578_257896


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l2578_257878

theorem min_value_expression (x y : ℝ) : 
  (x * y - 1/2)^2 + (x - y)^2 ≥ 1/4 :=
sorry

theorem min_value_attainable : 
  ∃ x y : ℝ, (x * y - 1/2)^2 + (x - y)^2 = 1/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l2578_257878


namespace NUMINAMATH_CALUDE_sqrt_300_simplification_l2578_257837

theorem sqrt_300_simplification : Real.sqrt 300 = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_300_simplification_l2578_257837


namespace NUMINAMATH_CALUDE_max_value_theorem_l2578_257858

theorem max_value_theorem (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_sum : a + b + c + d ≤ 4) : 
  (a^2 * (a + b))^(1/4) + (b^2 * (b + c))^(1/4) + 
  (c^2 * (c + d))^(1/4) + (d^2 * (d + a))^(1/4) ≤ 4 * 2^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2578_257858


namespace NUMINAMATH_CALUDE_no_real_solutions_l2578_257818

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  (3 * x^2) / (x - 2) - (3 * x + 10) / 4 + (9 - 9 * x) / (x - 2) - 3 = 0

-- Theorem stating that the equation has no real solutions
theorem no_real_solutions : ¬∃ x : ℝ, original_equation x :=
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2578_257818


namespace NUMINAMATH_CALUDE_pentagonal_country_routes_fifty_cities_routes_no_forty_six_routes_l2578_257838

/-- Definition of a pentagonal country -/
def PentagonalCountry (n : ℕ) := n > 0

/-- Number of air routes in a pentagonal country -/
def airRoutes (n : ℕ) : ℕ := (n * 5) / 2

theorem pentagonal_country_routes (n : ℕ) (h : PentagonalCountry n) : 
  airRoutes n = (n * 5) / 2 :=
sorry

theorem fifty_cities_routes : 
  airRoutes 50 = 125 :=
sorry

theorem no_forty_six_routes : 
  ¬ ∃ (n : ℕ), PentagonalCountry n ∧ airRoutes n = 46 :=
sorry

end NUMINAMATH_CALUDE_pentagonal_country_routes_fifty_cities_routes_no_forty_six_routes_l2578_257838


namespace NUMINAMATH_CALUDE_min_cuts_for_ten_pieces_l2578_257894

/-- The number of pieces resulting from n vertical cuts on a cylindrical cake -/
def num_pieces (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The minimum number of vertical cuts needed to divide a cylindrical cake into at least 10 pieces -/
theorem min_cuts_for_ten_pieces : ∃ (n : ℕ), n ≥ 4 ∧ num_pieces n ≥ 10 ∧ ∀ (m : ℕ), m < n → num_pieces m < 10 :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_ten_pieces_l2578_257894


namespace NUMINAMATH_CALUDE_line_slope_problem_l2578_257883

theorem line_slope_problem (n : ℝ) : 
  n > 0 → 
  (n - 5) / (2 - n) = 2 * n → 
  n = 2.5 := by
sorry

end NUMINAMATH_CALUDE_line_slope_problem_l2578_257883


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2578_257867

theorem sphere_surface_area (r : ℝ) (h : r = 3) : 4 * π * r^2 = 36 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2578_257867


namespace NUMINAMATH_CALUDE_set_operations_l2578_257834

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3 ∨ 4 < x ∧ x < 6}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 5}

-- Define the complement operation
def complement (S : Set ℝ) : Set ℝ := {x | x ∉ S}

-- State the theorem
theorem set_operations :
  (complement A = {x | x < 1 ∨ (3 < x ∧ x ≤ 4) ∨ 6 ≤ x}) ∧
  (complement B = {x | x < 2 ∨ 5 ≤ x}) ∧
  (A ∩ (complement B) = {x | 1 ≤ x ∧ x < 2 ∨ 5 ≤ x ∧ x < 6}) ∧
  ((complement A) ∪ B = {x | x < 1 ∨ (2 ≤ x ∧ x < 5) ∨ 6 ≤ x}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l2578_257834


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2578_257899

theorem consecutive_integers_sum (n : ℕ) (hn : n > 0) :
  (∃ m : ℤ, (Finset.range n).sum (λ i => m - i) = n) ↔ n % 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2578_257899


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l2578_257810

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 280

/-- A debt that can be resolved using pigs and goats -/
def resolvable_debt (d : ℕ) : Prop :=
  ∃ (p g : ℤ), d = pig_value * p + goat_value * g

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 40

theorem smallest_resolvable_debt_is_correct :
  (resolvable_debt smallest_resolvable_debt) ∧
  (∀ d : ℕ, d > 0 ∧ d < smallest_resolvable_debt → ¬(resolvable_debt d)) :=
sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l2578_257810


namespace NUMINAMATH_CALUDE_larger_segment_is_50_l2578_257802

/-- Represents a triangle with sides a, b, c and an altitude h dropped on side c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  x : ℝ  -- shorter segment of side c
  valid_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b
  altitude_property : a^2 = x^2 + h^2 ∧ b^2 = (c - x)^2 + h^2

/-- The larger segment of the side c in a triangle with sides 40, 50, 90 is 50 --/
theorem larger_segment_is_50 :
  ∀ t : Triangle, t.a = 40 ∧ t.b = 50 ∧ t.c = 90 → (t.c - t.x = 50) :=
by sorry

end NUMINAMATH_CALUDE_larger_segment_is_50_l2578_257802


namespace NUMINAMATH_CALUDE_ellipse_circle_inequality_l2578_257827

theorem ellipse_circle_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : x₁^2 / a^2 + y₁^2 / b^2 = 1)
  (h₂ : x₂^2 / a^2 + y₂^2 / b^2 = 1)
  (x y : ℝ)
  (h_circle : (x - x₁) * (x - x₂) + (y - y₁) * (y - y₂) = 0) :
  x^2 + y^2 ≤ (3/2) * a^2 + (1/2) * b^2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_circle_inequality_l2578_257827


namespace NUMINAMATH_CALUDE_sin_cos_range_l2578_257880

theorem sin_cos_range (x : ℝ) : 
  -1 ≤ Real.sin x + Real.cos x + Real.sin x * Real.cos x ∧ 
  Real.sin x + Real.cos x + Real.sin x * Real.cos x ≤ 1/2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_range_l2578_257880


namespace NUMINAMATH_CALUDE_other_candidate_votes_l2578_257888

theorem other_candidate_votes
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (winning_candidate_percentage : ℚ)
  (h_total : total_votes = 9000)
  (h_invalid : invalid_percentage = 30 / 100)
  (h_winning : winning_candidate_percentage = 60 / 100) :
  ↑total_votes * (1 - invalid_percentage) * (1 - winning_candidate_percentage) = 2520 :=
by sorry

end NUMINAMATH_CALUDE_other_candidate_votes_l2578_257888


namespace NUMINAMATH_CALUDE_train_passing_tree_l2578_257864

/-- Proves that a train of given length and speed takes a specific time to pass a tree -/
theorem train_passing_tree (train_length : ℝ) (train_speed_kmh : ℝ) (time : ℝ) :
  train_length = 280 →
  train_speed_kmh = 72 →
  time = train_length / (train_speed_kmh * (5/18)) →
  time = 14 := by
  sorry

#check train_passing_tree

end NUMINAMATH_CALUDE_train_passing_tree_l2578_257864


namespace NUMINAMATH_CALUDE_fraction_divisibility_l2578_257897

theorem fraction_divisibility (a b n : ℕ) (hodd : Odd n) 
  (hnum : n ∣ (a^n + b^n)) (hden : n ∣ (a + b)) : 
  n ∣ ((a^n + b^n) / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_divisibility_l2578_257897


namespace NUMINAMATH_CALUDE_point_inside_circle_l2578_257898

theorem point_inside_circle (a : ℝ) : 
  let P : ℝ × ℝ := (5*a + 1, 12*a)
  ((P.1 - 1)^2 + P.2^2 < 1) ↔ (abs a < 1/13) :=
sorry

end NUMINAMATH_CALUDE_point_inside_circle_l2578_257898


namespace NUMINAMATH_CALUDE_parabola_properties_l2578_257835

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties (a b c m : ℝ) :
  -- Conditions
  (∃ C : ℝ, C > 0 ∧ parabola a b c C = 0) →  -- Intersects positive y-axis
  (parabola a b c 1 = 2) →                   -- Vertex at (1, 2)
  (parabola a b c (-1) = m) →                -- Passes through (-1, m)
  (m < 0) →                                  -- m is negative
  -- Conclusions
  (2 * a + b = 0) ∧                          -- Conclusion ②
  (-2 < a ∧ a < -1/2) ∧                      -- Conclusion ③
  (∀ n : ℝ, (∀ x : ℝ, parabola a b c x ≠ n) → n > 2) ∧  -- Conclusion ④
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ parabola a b c x₁ = 1 ∧ parabola a b c x₂ = 1 ∧ x₁ + x₂ = 2)  -- Conclusion ⑥
  := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2578_257835


namespace NUMINAMATH_CALUDE_fgh_supermarkets_l2578_257836

theorem fgh_supermarkets (total : ℕ) (difference : ℕ) (us_count : ℕ) : 
  total = 84 → difference = 10 → us_count = total / 2 + difference / 2 → us_count = 47 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_l2578_257836


namespace NUMINAMATH_CALUDE_total_miles_theorem_l2578_257895

/-- The total miles run by Bill and Julia on Saturday and Sunday -/
def total_miles (bill_sunday : ℕ) : ℕ :=
  let bill_saturday := bill_sunday - 4
  let julia_sunday := 2 * bill_sunday
  bill_saturday + bill_sunday + julia_sunday

/-- Theorem: Given the conditions, Bill and Julia ran 36 miles in total -/
theorem total_miles_theorem (bill_sunday : ℕ) 
  (h1 : bill_sunday = 10) : total_miles bill_sunday = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_theorem_l2578_257895


namespace NUMINAMATH_CALUDE_max_value_of_f_l2578_257807

-- Define the function
def f (x : ℝ) := abs (x^2 - 4) - 6*x

-- State the theorem
theorem max_value_of_f :
  ∃ (b : ℝ), b = 12 ∧ 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 5 → f x ≤ b) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 5 ∧ f x = b) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2578_257807


namespace NUMINAMATH_CALUDE_peters_exam_score_l2578_257871

theorem peters_exam_score :
  ∀ (e m h : ℕ),
  e + m + h = 25 →
  2 * e + 3 * m + 5 * h = 84 →
  m % 2 = 0 →
  h % 3 = 0 →
  2 * e + 3 * (m / 2) + 5 * (h / 3) = 40 :=
by sorry

end NUMINAMATH_CALUDE_peters_exam_score_l2578_257871


namespace NUMINAMATH_CALUDE_base_7_divisibility_l2578_257831

def is_base_7_digit (x : ℕ) : Prop := x ≤ 6

def base_7_to_decimal (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

theorem base_7_divisibility (x : ℕ) : 
  is_base_7_digit x → (base_7_to_decimal x) % 29 = 0 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_7_divisibility_l2578_257831


namespace NUMINAMATH_CALUDE_lcm_of_23_46_827_l2578_257814

theorem lcm_of_23_46_827 : Nat.lcm 23 (Nat.lcm 46 827) = 38042 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_23_46_827_l2578_257814


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2578_257884

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r = 3) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2578_257884


namespace NUMINAMATH_CALUDE_student_average_age_l2578_257861

theorem student_average_age (num_students : ℕ) (teacher_age : ℕ) (avg_increase : ℕ) :
  num_students = 15 →
  teacher_age = 26 →
  avg_increase = 1 →
  (num_students * 10 + teacher_age) / (num_students + 1) = 10 + avg_increase :=
by sorry

end NUMINAMATH_CALUDE_student_average_age_l2578_257861


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_rational_inequality_solution_l2578_257885

-- Part 1
theorem quadratic_inequality_solution (b c : ℝ) :
  (∀ x, 5 * x^2 - b * x + c < 0 ↔ -1 < x ∧ x < 3) →
  b + c = -5 :=
sorry

-- Part 2
theorem rational_inequality_solution :
  {x : ℝ | (2 * x - 5) / (x + 4) ≥ 0} = {x : ℝ | x ≥ 5/2 ∨ x < -4} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_rational_inequality_solution_l2578_257885


namespace NUMINAMATH_CALUDE_john_soap_cost_l2578_257854

/-- The amount of money John spent on soap -/
def soap_cost (num_bars : ℕ) (weight_per_bar : ℚ) (price_per_pound : ℚ) : ℚ :=
  num_bars * weight_per_bar * price_per_pound

/-- Proof that John spent $15 on soap -/
theorem john_soap_cost :
  soap_cost 20 (3/2) (1/2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_soap_cost_l2578_257854


namespace NUMINAMATH_CALUDE_smallest_even_five_digit_number_has_eight_in_tens_place_l2578_257800

-- Define a type for digits
inductive Digit : Type
  | one : Digit
  | three : Digit
  | five : Digit
  | six : Digit
  | eight : Digit

-- Define a function to convert Digit to Nat
def digitToNat : Digit → Nat
  | Digit.one => 1
  | Digit.three => 3
  | Digit.five => 5
  | Digit.six => 6
  | Digit.eight => 8

-- Define a function to check if a number is even
def isEven (n : Nat) : Bool :=
  n % 2 == 0

-- Define a function to construct a five-digit number from Digits
def makeNumber (a b c d e : Digit) : Nat :=
  10000 * (digitToNat a) + 1000 * (digitToNat b) + 100 * (digitToNat c) + 10 * (digitToNat d) + (digitToNat e)

-- Define the theorem
theorem smallest_even_five_digit_number_has_eight_in_tens_place :
  ∀ (a b c d e : Digit),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e →
    isEven (makeNumber a b c d e) →
    (∀ (x y z w v : Digit),
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ v ∧
      z ≠ w ∧ z ≠ v ∧
      w ≠ v →
      isEven (makeNumber x y z w v) →
      makeNumber a b c d e ≤ makeNumber x y z w v) →
    d = Digit.eight :=
  sorry

end NUMINAMATH_CALUDE_smallest_even_five_digit_number_has_eight_in_tens_place_l2578_257800


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2578_257853

theorem quadratic_function_property (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  (f 0 = f 4 ∧ f 0 > f 1) → (a > 0 ∧ 4 * a + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2578_257853


namespace NUMINAMATH_CALUDE_dividend_calculation_l2578_257843

theorem dividend_calculation (x : ℕ) (h : x > 1) :
  let divisor := 3 * x^2
  let quotient := 5 * x
  let remainder := 7 * x + 9
  let dividend := divisor * quotient + remainder
  dividend = 15 * x^3 + 7 * x + 9 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2578_257843


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l2578_257870

/-- Represents the speed of a swimmer in still water and the speed of the stream -/
structure SwimmerSpeeds where
  man : ℝ  -- Speed of the man in still water
  stream : ℝ  -- Speed of the stream

/-- Calculates the effective speed when swimming downstream -/
def downstream_speed (s : SwimmerSpeeds) : ℝ := s.man + s.stream

/-- Calculates the effective speed when swimming upstream -/
def upstream_speed (s : SwimmerSpeeds) : ℝ := s.man - s.stream

/-- Theorem: Given the conditions of the swimming problem, the man's speed in still water is 8 km/h -/
theorem swimmer_speed_in_still_water :
  ∃ (s : SwimmerSpeeds),
    (downstream_speed s * 4 = 48) ∧
    (upstream_speed s * 6 = 24) ∧
    (s.man = 8) := by
  sorry

#check swimmer_speed_in_still_water

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l2578_257870


namespace NUMINAMATH_CALUDE_same_height_siblings_l2578_257825

-- Define the number of siblings
def num_siblings : ℕ := 5

-- Define the total height of all siblings
def total_height : ℕ := 330

-- Define the height of one sibling
def one_sibling_height : ℕ := 60

-- Define Eliza's height
def eliza_height : ℕ := 68

-- Define the height difference between Eliza and one sibling
def height_difference : ℕ := 2

-- Theorem to prove
theorem same_height_siblings (h : ℕ) : 
  h * 2 + one_sibling_height + eliza_height + (eliza_height + height_difference) = total_height →
  h = 66 := by
  sorry


end NUMINAMATH_CALUDE_same_height_siblings_l2578_257825


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2578_257826

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x := by sorry

theorem negation_of_proposition : 
  (¬ ∃ x : ℝ, 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, 2*x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2578_257826


namespace NUMINAMATH_CALUDE_fourth_power_equality_l2578_257890

theorem fourth_power_equality (x : ℝ) : x^4 = (-3)^4 → x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_equality_l2578_257890


namespace NUMINAMATH_CALUDE_range_of_negative_values_l2578_257886

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y, x < y ∧ y ≤ 0 → f x > f y

-- State the theorem
theorem range_of_negative_values (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_decreasing : decreasing_on_neg f) 
  (h_zero : f 3 = 0) : 
  {x : ℝ | f x < 0} = Set.Ioo (-3) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_negative_values_l2578_257886


namespace NUMINAMATH_CALUDE_triangle_properties_l2578_257808

/-- Represents a triangle with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem stating properties of triangles -/
theorem triangle_properties (t : Triangle) :
  (t.A > t.B → Real.sin t.A > Real.sin t.B) ∧
  (t.A = π / 6 ∧ t.b = 4 ∧ t.a = 3 → ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ 
    t1.A = t.A ∧ t1.b = t.b ∧ t1.a = t.a ∧
    t2.A = t.A ∧ t2.b = t.b ∧ t2.a = t.a) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2578_257808


namespace NUMINAMATH_CALUDE_train_passing_platform_l2578_257815

/-- Given a train of length 1200 meters that crosses a tree in 120 seconds,
    prove that it takes 180 seconds to pass a platform of length 600 meters. -/
theorem train_passing_platform
  (train_length : ℝ)
  (tree_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 1200)
  (h2 : tree_crossing_time = 120)
  (h3 : platform_length = 600) :
  (train_length + platform_length) / (train_length / tree_crossing_time) = 180 :=
sorry

end NUMINAMATH_CALUDE_train_passing_platform_l2578_257815


namespace NUMINAMATH_CALUDE_gcd_sum_problem_l2578_257887

def is_valid (a b c : ℕ+) : Prop :=
  Nat.gcd a.val (Nat.gcd b.val c.val) = 1 ∧
  Nat.gcd a.val (b.val + c.val) > 1 ∧
  Nat.gcd b.val (c.val + a.val) > 1 ∧
  Nat.gcd c.val (a.val + b.val) > 1

theorem gcd_sum_problem :
  (∃ a b c : ℕ+, is_valid a b c ∧ a.val + b.val + c.val = 2015) ∧
  (∀ a b c : ℕ+, is_valid a b c → a.val + b.val + c.val ≥ 30) ∧
  (∃ a b c : ℕ+, is_valid a b c ∧ a.val + b.val + c.val = 30) := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_problem_l2578_257887


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l2578_257856

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 4*y + 12 = 0

-- Define the point A
def point_A : ℝ × ℝ := (-2, 3)

-- Define a line in slope-intercept form
def line (k b : ℝ) (x y : ℝ) : Prop :=
  y = k * x + b

-- Define the property of being tangent to the circle
def is_tangent_to_circle (k b : ℝ) : Prop :=
  ∃ (x y : ℝ), line k b x y ∧ circle_C x y

-- Define the property of passing through the reflection of A
def passes_through_reflection (k b : ℝ) : Prop :=
  line k b (-2) (-3)

-- State the theorem
theorem reflected_ray_equation :
  ∃ (k b : ℝ), 
    is_tangent_to_circle k b ∧
    passes_through_reflection k b ∧
    ((k = 4/3 ∧ b = -1/3) ∨ (k = 3/4 ∧ b = -3/2)) :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l2578_257856


namespace NUMINAMATH_CALUDE_alice_spending_percentage_l2578_257874

theorem alice_spending_percentage (alice_initial bob_initial alice_final : ℝ) :
  bob_initial = 0.9 * alice_initial →
  alice_final = 0.9 * bob_initial →
  (alice_initial - alice_final) / alice_initial = 0.19 := by
  sorry

end NUMINAMATH_CALUDE_alice_spending_percentage_l2578_257874


namespace NUMINAMATH_CALUDE_jack_sugar_final_amount_l2578_257876

/-- Given Jack's sugar transactions, prove the final amount of sugar. -/
theorem jack_sugar_final_amount
  (initial : ℕ)  -- Initial amount of sugar
  (used : ℕ)     -- Amount of sugar used
  (bought : ℕ)   -- Amount of sugar bought
  (h1 : initial = 65)
  (h2 : used = 18)
  (h3 : bought = 50) :
  initial - used + bought = 97 := by
  sorry

end NUMINAMATH_CALUDE_jack_sugar_final_amount_l2578_257876


namespace NUMINAMATH_CALUDE_sum_a_b_values_l2578_257816

theorem sum_a_b_values (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 2) (h3 : |a-b| = b-a) :
  a + b = -1 ∨ a + b = -5 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_values_l2578_257816


namespace NUMINAMATH_CALUDE_parabola_symmetric_points_l2578_257822

/-- A parabola with parameter p > 0 has two distinct points symmetrical with respect to the line x + y = 1 if and only if 0 < p < 2/3 -/
theorem parabola_symmetric_points (p : ℝ) :
  (p > 0) →
  (∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    (A.2)^2 = 2*p*A.1 ∧
    (B.2)^2 = 2*p*B.1 ∧
    (∃ (C : ℝ × ℝ),
      C.1 + C.2 = 1 ∧
      C.1 = (A.1 + B.1) / 2 ∧
      C.2 = (A.2 + B.2) / 2)) ↔
  (0 < p ∧ p < 2/3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_points_l2578_257822
