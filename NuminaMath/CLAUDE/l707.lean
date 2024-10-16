import Mathlib

namespace NUMINAMATH_CALUDE_floor_of_e_equals_two_l707_70776

noncomputable def e : ℝ := Real.exp 1

theorem floor_of_e_equals_two : ⌊e⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_e_equals_two_l707_70776


namespace NUMINAMATH_CALUDE_marathon_positions_l707_70749

/-- Represents a marathon with participants -/
structure Marathon where
  total_participants : ℕ
  john_from_right : ℕ
  john_from_left : ℕ
  mike_ahead : ℕ

/-- Theorem about the marathon positions -/
theorem marathon_positions (m : Marathon) 
  (h1 : m.john_from_right = 28)
  (h2 : m.john_from_left = 42)
  (h3 : m.mike_ahead = 10) :
  m.total_participants = 69 ∧ 
  m.john_from_left - m.mike_ahead = 32 ∧ 
  m.john_from_right - m.mike_ahead = 18 := by
  sorry


end NUMINAMATH_CALUDE_marathon_positions_l707_70749


namespace NUMINAMATH_CALUDE_expression_evaluation_l707_70719

theorem expression_evaluation (a b c : ℝ) : 
  let d := a + b + c
  2 * (a^2 * b^2 + a^2 * c^2 + a^2 * d^2 + b^2 * c^2 + b^2 * d^2 + c^2 * d^2) - 
  (a^4 + b^4 + c^4 + d^4) + 8 * a * b * c * d = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l707_70719


namespace NUMINAMATH_CALUDE_raja_monthly_savings_l707_70727

/-- Raja's monthly savings calculation --/
theorem raja_monthly_savings :
  let monthly_income : ℝ := 24999.999999999993
  let household_percentage : ℝ := 0.60
  let clothes_percentage : ℝ := 0.10
  let medicines_percentage : ℝ := 0.10
  let total_spent_percentage : ℝ := household_percentage + clothes_percentage + medicines_percentage
  let savings_percentage : ℝ := 1 - total_spent_percentage
  let savings : ℝ := savings_percentage * monthly_income
  ⌊savings⌋ = 5000 := by sorry

end NUMINAMATH_CALUDE_raja_monthly_savings_l707_70727


namespace NUMINAMATH_CALUDE_bird_families_to_asia_count_l707_70710

/-- The number of bird families that flew away to Asia -/
def bird_families_to_asia (total_migrated : ℕ) (to_africa : ℕ) : ℕ :=
  total_migrated - to_africa

/-- Theorem stating that 80 bird families flew away to Asia -/
theorem bird_families_to_asia_count : 
  bird_families_to_asia 118 38 = 80 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_to_asia_count_l707_70710


namespace NUMINAMATH_CALUDE_h_negative_two_equals_eleven_l707_70765

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 1
def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_negative_two_equals_eleven : h (-2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_h_negative_two_equals_eleven_l707_70765


namespace NUMINAMATH_CALUDE_incorrect_accuracy_statement_l707_70726

def accurate_to_nearest_hundred (x : ℝ) : Prop :=
  ∃ n : ℤ, x = (n : ℝ) * 100 ∧ |x - (n : ℝ) * 100| ≤ 50

theorem incorrect_accuracy_statement :
  ¬(accurate_to_nearest_hundred 2130) :=
sorry

end NUMINAMATH_CALUDE_incorrect_accuracy_statement_l707_70726


namespace NUMINAMATH_CALUDE_ellipse_m_range_l707_70761

/-- The equation of an ellipse with parameter m -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (16 - m) + y^2 / (m + 4) = 1

/-- The condition for the equation to represent an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  (16 - m > 0) ∧ (m + 4 > 0) ∧ (16 - m ≠ m + 4)

/-- Theorem stating the range of m for which the equation represents an ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m ↔ (m > -4 ∧ m < 16 ∧ m ≠ 6) :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l707_70761


namespace NUMINAMATH_CALUDE_van_rental_equation_l707_70718

theorem van_rental_equation (x : ℕ) (h : x > 0) :
  (180 : ℝ) / x - 180 / (x + 2) = 3 ↔
  (∃ (y : ℝ), y > 0 ∧ 180 / x = y ∧ 180 / (x + 2) = y - 3) :=
by sorry

end NUMINAMATH_CALUDE_van_rental_equation_l707_70718


namespace NUMINAMATH_CALUDE_sum_marked_sides_ge_one_l707_70713

/-- A rectangle within a unit square --/
structure Rectangle where
  width : ℝ
  height : ℝ
  markedSide : ℝ
  width_pos : 0 < width
  height_pos : 0 < height
  in_unit_square : width ≤ 1 ∧ height ≤ 1
  marked_side_valid : markedSide = width ∨ markedSide = height

/-- A partition of the unit square into rectangles --/
def UnitSquarePartition := List Rectangle

/-- The sum of the marked sides in a partition --/
def sumMarkedSides (partition : UnitSquarePartition) : ℝ :=
  partition.map (·.markedSide) |>.sum

/-- The total area of rectangles in a partition --/
def totalArea (partition : UnitSquarePartition) : ℝ :=
  partition.map (λ r => r.width * r.height) |>.sum

/-- Theorem: The sum of marked sides in any valid partition is at least 1 --/
theorem sum_marked_sides_ge_one (partition : UnitSquarePartition) 
  (h_valid : totalArea partition = 1) : 
  sumMarkedSides partition ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_sum_marked_sides_ge_one_l707_70713


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_S_l707_70714

def U : Set ℕ := Set.univ

def S : Set ℕ := {x ∈ U | x^2 - x = 0}

def T : Set ℕ := {x ∈ U | ∃ k : ℤ, 6 = k * (x - 2)}

theorem S_intersect_T_eq_S : S ∩ T = S := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_S_l707_70714


namespace NUMINAMATH_CALUDE_min_distance_curve_to_line_l707_70788

/-- Given a > 0 and b = -1/2 * a^2 + 3 * ln(a), and a point Q(m, n) on the line y = 2x + 1/2,
    the minimum value of (a-m)^2 + (b-n)^2 is 9/5 -/
theorem min_distance_curve_to_line (a b m n : ℝ) (ha : a > 0) 
  (hb : b = -1/2 * a^2 + 3 * Real.log a) (hq : n = 2 * m + 1/2) :
  ∃ (min_val : ℝ), min_val = 9/5 ∧ 
  ∀ (x y : ℝ), (y = -1/2 * x^2 + 3 * Real.log x) → 
  (a - m)^2 + (b - n)^2 ≤ (x - m)^2 + (y - n)^2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_curve_to_line_l707_70788


namespace NUMINAMATH_CALUDE_expression_evaluation_l707_70753

theorem expression_evaluation (x y : ℤ) (hx : x = -6) (hy : y = -3) :
  (x - y)^2 - x*y = -9 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l707_70753


namespace NUMINAMATH_CALUDE_seven_equidistant_planes_l707_70773

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A type representing a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Function to check if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Function to check if a plane is equidistant from four points -/
def isEquidistant (plane : Plane3D) (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Function to count the number of planes equidistant from four points -/
def countEquidistantPlanes (p1 p2 p3 p4 : Point3D) : ℕ := sorry

/-- Theorem stating that there are exactly 7 equidistant planes for four non-coplanar points -/
theorem seven_equidistant_planes
  (p1 p2 p3 p4 : Point3D)
  (h : ¬ areCoplanar p1 p2 p3 p4) :
  countEquidistantPlanes p1 p2 p3 p4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_equidistant_planes_l707_70773


namespace NUMINAMATH_CALUDE_cos_13_cos_17_minus_sin_17_sin_13_l707_70739

theorem cos_13_cos_17_minus_sin_17_sin_13 :
  Real.cos (13 * π / 180) * Real.cos (17 * π / 180) - 
  Real.sin (17 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_13_cos_17_minus_sin_17_sin_13_l707_70739


namespace NUMINAMATH_CALUDE_sams_adventure_books_l707_70782

/-- The number of adventure books Sam bought at the school's book fair -/
def adventure_books : ℕ := by sorry

/-- The number of mystery books Sam bought -/
def mystery_books : ℕ := 17

/-- The number of new books Sam bought -/
def new_books : ℕ := 15

/-- The number of used books Sam bought -/
def used_books : ℕ := 15

/-- The total number of books Sam bought -/
def total_books : ℕ := new_books + used_books

theorem sams_adventure_books : adventure_books = 13 := by sorry

end NUMINAMATH_CALUDE_sams_adventure_books_l707_70782


namespace NUMINAMATH_CALUDE_edward_tickets_l707_70752

theorem edward_tickets (booth_tickets : ℕ) (ride_cost : ℕ) (num_rides : ℕ) : 
  booth_tickets = 23 → ride_cost = 7 → num_rides = 8 →
  ∃ total_tickets : ℕ, total_tickets = booth_tickets + ride_cost * num_rides :=
by
  sorry

end NUMINAMATH_CALUDE_edward_tickets_l707_70752


namespace NUMINAMATH_CALUDE_unique_positive_solution_l707_70769

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x ∧ x = Real.sqrt (1 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l707_70769


namespace NUMINAMATH_CALUDE_expression_values_l707_70785

theorem expression_values (a b : ℝ) (h : (2 * a) / (a + b) + b / (a - b) = 2) :
  (3 * a - b) / (a + 5 * b) = 1 ∨ (3 * a - b) / (a + 5 * b) = 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l707_70785


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_positive_n_for_unique_solution_l707_70733

theorem unique_solution_quadratic (n : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + n * x + 16 = 0) ↔ n = 24 ∨ n = -24 :=
by sorry

theorem positive_n_for_unique_solution :
  ∃ n : ℝ, n > 0 ∧ (∃! x : ℝ, 9 * x^2 + n * x + 16 = 0) ∧ n = 24 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_positive_n_for_unique_solution_l707_70733


namespace NUMINAMATH_CALUDE_triangle_inequality_fraction_l707_70798

theorem triangle_inequality_fraction (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b) : 
  (a + b) / (1 + a + b) > c / (1 + c) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_fraction_l707_70798


namespace NUMINAMATH_CALUDE_girls_together_arrangement_person_not_in_middle_l707_70700

-- Define the number of boys and girls
def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_people : ℕ := num_boys + num_girls

-- Define permutation and combination functions
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement A
theorem girls_together_arrangement :
  (A num_girls num_girls) * (A (num_boys + 1) (num_boys + 1)) =
  A num_girls num_girls * A 5 5 := by sorry

-- Statement C
theorem person_not_in_middle :
  (C (total_people - 1) 1) * (A (total_people - 1) (total_people - 1)) =
  C 6 1 * A 6 6 := by sorry

end NUMINAMATH_CALUDE_girls_together_arrangement_person_not_in_middle_l707_70700


namespace NUMINAMATH_CALUDE_six_star_nine_l707_70758

-- Define the star operation
def star (a b : ℕ) : ℚ :=
  (a * b : ℚ) / (a + b - 3 : ℚ)

-- Theorem statement
theorem six_star_nine :
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ a + b > 3) →
  star 6 9 = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_six_star_nine_l707_70758


namespace NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_4_l707_70737

def numbers : List Nat := [4624, 4634, 4644, 4652, 4672]

def is_divisible_by_4 (n : Nat) : Bool :=
  n % 4 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_not_divisible_by_4 :
  ∃ n ∈ numbers, ¬is_divisible_by_4 n ∧ units_digit n * tens_digit n = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_not_divisible_by_4_l707_70737


namespace NUMINAMATH_CALUDE_triangle_side_length_l707_70760

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  B = π / 6 →  -- 30° in radians
  (1 / 2) * a * c * Real.sin B = 3 / 2 →  -- Area formula
  Real.sin A + Real.sin C = 2 * Real.sin B →  -- Given condition
  b = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l707_70760


namespace NUMINAMATH_CALUDE_max_candies_eaten_l707_70784

/-- Represents the state of the board and the total candies eaten -/
structure BoardState :=
  (numbers : List ℕ)
  (candies : ℕ)

/-- Represents one step of Karlson's process -/
def step (state : BoardState) : BoardState :=
  sorry

/-- The initial state of the board -/
def initial_state : BoardState :=
  { numbers := List.replicate 40 1, candies := 0 }

/-- Applies the step function n times -/
def apply_n_steps (n : ℕ) (state : BoardState) : BoardState :=
  sorry

theorem max_candies_eaten :
  ∃ (final_state : BoardState),
    final_state = apply_n_steps 40 initial_state ∧
    final_state.candies ≤ 780 ∧
    ∀ (other_final_state : BoardState),
      other_final_state = apply_n_steps 40 initial_state →
      other_final_state.candies ≤ final_state.candies :=
sorry

end NUMINAMATH_CALUDE_max_candies_eaten_l707_70784


namespace NUMINAMATH_CALUDE_eyes_seeing_airplane_l707_70751

/-- Given 200 students on a field and 3/4 of them looking up at an airplane,
    prove that the number of eyes that saw the airplane is 300. -/
theorem eyes_seeing_airplane (total_students : ℕ) (fraction_looking_up : ℚ) : 
  total_students = 200 →
  fraction_looking_up = 3/4 →
  (total_students : ℚ) * fraction_looking_up * 2 = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_eyes_seeing_airplane_l707_70751


namespace NUMINAMATH_CALUDE_cube_equation_result_l707_70789

theorem cube_equation_result (x : ℝ) (h : x^3 + 3*x = 9) : x^6 + 27*x^3 = 324 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_result_l707_70789


namespace NUMINAMATH_CALUDE_max_value_ab_l707_70774

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt 2 = Real.sqrt (2^x * 2^y) → a * b ≥ x * y) ∧ a * b = 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_value_ab_l707_70774


namespace NUMINAMATH_CALUDE_states_joined_fraction_l707_70766

theorem states_joined_fraction :
  let total_states : ℕ := 30
  let states_1780_to_1789 : ℕ := 12
  let states_1790_to_1799 : ℕ := 5
  let states_1780_to_1799 : ℕ := states_1780_to_1789 + states_1790_to_1799
  (states_1780_to_1799 : ℚ) / total_states = 17 / 30 := by
  sorry

end NUMINAMATH_CALUDE_states_joined_fraction_l707_70766


namespace NUMINAMATH_CALUDE_total_cars_is_180_l707_70725

/-- The total number of cars produced over two days, given the production on the first day and that the second day's production is twice the first day's. -/
def total_cars (day1_production : ℕ) : ℕ :=
  day1_production + 2 * day1_production

/-- Theorem stating that the total number of cars produced is 180 when 60 cars were produced on the first day. -/
theorem total_cars_is_180 : total_cars 60 = 180 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_is_180_l707_70725


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l707_70775

theorem system_of_equations_solution :
  ∀ s t : ℝ,
  (11 * s + 7 * t = 240) →
  (s = (1/2) * t + 3) →
  (t = 414/25 ∧ s = 11.28) :=
by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l707_70775


namespace NUMINAMATH_CALUDE_silver_coins_removed_l707_70735

theorem silver_coins_removed (total_coins : ℕ) (initial_gold_percent : ℚ) (final_gold_percent : ℚ) :
  total_coins = 200 →
  initial_gold_percent = 2 / 100 →
  final_gold_percent = 20 / 100 →
  (total_coins : ℚ) * initial_gold_percent = (total_coins - (total_coins : ℚ) * initial_gold_percent * (1 / final_gold_percent - 1)) * final_gold_percent →
  ⌊total_coins - (total_coins : ℚ) * initial_gold_percent * (1 / final_gold_percent)⌋ = 180 :=
by sorry

end NUMINAMATH_CALUDE_silver_coins_removed_l707_70735


namespace NUMINAMATH_CALUDE_number_base_conversion_l707_70731

theorem number_base_conversion :
  ∃! (x y z b : ℕ),
    (x * b^2 + y * b + z = 1989) ∧
    (b^2 ≤ 1989) ∧
    (1989 < b^3) ∧
    (x + y + z = 27) ∧
    (0 ≤ x) ∧ (x < b) ∧
    (0 ≤ y) ∧ (y < b) ∧
    (0 ≤ z) ∧ (z < b) ∧
    (x = 5 ∧ y = 9 ∧ z = 13 ∧ b = 19) := by
  sorry

end NUMINAMATH_CALUDE_number_base_conversion_l707_70731


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l707_70747

/-- The area of a circle with diameter endpoints C(-2, 3) and D(4, -1) is 13π. -/
theorem circle_area_from_diameter_endpoints :
  let C : ℝ × ℝ := (-2, 3)
  let D : ℝ × ℝ := (4, -1)
  let diameter_squared := (D.1 - C.1)^2 + (D.2 - C.2)^2
  let radius_squared := diameter_squared / 4
  let circle_area := π * radius_squared
  circle_area = 13 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l707_70747


namespace NUMINAMATH_CALUDE_even_Z_tetrominoes_l707_70772

/-- Represents a lattice polygon -/
structure LatticePolygon where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents an S-tetromino -/
inductive STetromino

/-- Represents a Z-tetromino -/
inductive ZTetromino

/-- Represents either an S-tetromino or a Z-tetromino -/
inductive Tetromino
  | S : STetromino → Tetromino
  | Z : ZTetromino → Tetromino

/-- Predicate indicating if a lattice polygon can be tiled with S-tetrominoes -/
def canBeTiledWithS (P : LatticePolygon) : Prop := sorry

/-- Represents a tiling of a lattice polygon using S- and Z-tetrominoes -/
def Tiling (P : LatticePolygon) := List Tetromino

/-- Counts the number of Z-tetrominoes in a tiling -/
def countZTetrominoes (tiling : Tiling P) : Nat := sorry

/-- Main theorem: For any lattice polygon that can be tiled with S-tetrominoes,
    any tiling using S- and Z-tetrominoes will contain an even number of Z-tetrominoes -/
theorem even_Z_tetrominoes (P : LatticePolygon) (h : canBeTiledWithS P) :
  ∀ (tiling : Tiling P), Even (countZTetrominoes tiling) := by
  sorry

end NUMINAMATH_CALUDE_even_Z_tetrominoes_l707_70772


namespace NUMINAMATH_CALUDE_exists_bound_factorial_digit_sum_l707_70729

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number b such that for all natural numbers n > b,
    the sum of the digits of n! is greater than or equal to 10^100 -/
theorem exists_bound_factorial_digit_sum :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (n.factorial) ≥ 10^100 := by sorry

end NUMINAMATH_CALUDE_exists_bound_factorial_digit_sum_l707_70729


namespace NUMINAMATH_CALUDE_blackboard_numbers_l707_70794

theorem blackboard_numbers (n : ℕ) (S : ℕ) (x : ℕ) : 
  S / n = 30 →
  (S + 100) / (n + 1) = 40 →
  (S + 100 + x) / (n + 2) = 50 →
  x = 120 := by
sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l707_70794


namespace NUMINAMATH_CALUDE_profit_starts_third_year_option1_more_cost_effective_l707_70709

/-- Represents the financial state of a fishing company -/
structure FishingCompany where
  initialCost : ℕ
  firstYearExpenses : ℕ
  annualExpenseIncrease : ℕ
  annualIncome : ℕ

/-- Calculates the year when the company starts to make a profit -/
def yearOfFirstProfit (company : FishingCompany) : ℕ :=
  sorry

/-- Calculates the more cost-effective option between two selling strategies -/
def moreCostEffectiveOption (company : FishingCompany) (option1Value : ℕ) (option2Value : ℕ) : Bool :=
  sorry

/-- Theorem stating that the company starts to make a profit in the third year -/
theorem profit_starts_third_year (company : FishingCompany) 
  (h1 : company.initialCost = 980000)
  (h2 : company.firstYearExpenses = 120000)
  (h3 : company.annualExpenseIncrease = 40000)
  (h4 : company.annualIncome = 500000) :
  yearOfFirstProfit company = 3 :=
sorry

/-- Theorem stating that the first option (selling for 260,000) is more cost-effective -/
theorem option1_more_cost_effective (company : FishingCompany)
  (h1 : company.initialCost = 980000)
  (h2 : company.firstYearExpenses = 120000)
  (h3 : company.annualExpenseIncrease = 40000)
  (h4 : company.annualIncome = 500000) :
  moreCostEffectiveOption company 260000 80000 = true :=
sorry

end NUMINAMATH_CALUDE_profit_starts_third_year_option1_more_cost_effective_l707_70709


namespace NUMINAMATH_CALUDE_complex_numbers_satisfying_conditions_l707_70787

theorem complex_numbers_satisfying_conditions :
  ∀ z : ℂ,
    (∃ t : ℝ, z + 10 / z = t ∧ 1 < t ∧ t ≤ 6) ∧
    (∃ a b : ℤ, z = ↑a + ↑b * I) →
    z = 1 + 3 * I ∨ z = 1 - 3 * I ∨ z = 3 + I ∨ z = 3 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_numbers_satisfying_conditions_l707_70787


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l707_70781

/-- Given three lines in the 2D plane:
    l₁: 3x + 4y - 2 = 0
    l₂: 2x + y + 2 = 0
    l₃: 3x - 2y + 4 = 0
    Prove that the line l: 2x - 3y - 22 = 0 passes through the intersection of l₁ and l₂,
    and is perpendicular to l₃. -/
theorem intersection_and_perpendicular_line 
  (l₁ : Real → Real → Prop) 
  (l₂ : Real → Real → Prop)
  (l₃ : Real → Real → Prop)
  (l : Real → Real → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ 3*x + 4*y - 2 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 2*x + y + 2 = 0)
  (h₃ : ∀ x y, l₃ x y ↔ 3*x - 2*y + 4 = 0)
  (h : ∀ x y, l x y ↔ 2*x - 3*y - 22 = 0) :
  (∃ x y, l₁ x y ∧ l₂ x y ∧ l x y) ∧ 
  (∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l x₂ y₂ → l₃ x₁ y₁ → l₃ x₂ y₂ → 
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) = 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l707_70781


namespace NUMINAMATH_CALUDE_expression_equals_seventy_percent_l707_70797

theorem expression_equals_seventy_percent (y : ℝ) (c : ℝ) (h1 : y > 0) 
  (h2 : (8 * y) / 20 + (c * y) / 10 = 0.7 * y) : c = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_seventy_percent_l707_70797


namespace NUMINAMATH_CALUDE_part_one_part_two_l707_70728

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part 1
theorem part_one (m : ℝ) (h1 : m > 0) 
  (h2 : Set.Icc (-2 : ℝ) 2 = {x | f (x + 1/2) ≤ 2*m + 1}) : 
  m = 3/2 := by sorry

-- Part 2
theorem part_two : 
  (∃ a : ℝ, ∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) ∧ 
  (∀ a : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) → a ≥ 4) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l707_70728


namespace NUMINAMATH_CALUDE_inequality_proof_l707_70768

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a * b + b * c + c * a = 1) : 
  (((1 / a + 6 * b) ^ (1/3 : ℝ)) + 
   ((1 / b + 6 * c) ^ (1/3 : ℝ)) + 
   ((1 / c + 6 * a) ^ (1/3 : ℝ))) ≤ 1 / (a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l707_70768


namespace NUMINAMATH_CALUDE_time_difference_for_trips_l707_70744

/-- Given a truck traveling at a constant speed, this theorem proves the time difference
    between two trips of different distances. -/
theorem time_difference_for_trips
  (speed : ℝ) (distance1 : ℝ) (distance2 : ℝ)
  (h1 : speed = 60)  -- Speed in miles per hour
  (h2 : distance1 = 570)  -- Distance of first trip in miles
  (h3 : distance2 = 540)  -- Distance of second trip in miles
  : (distance1 - distance2) / speed * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_for_trips_l707_70744


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l707_70704

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- State the theorem
theorem derivative_f_at_zero :
  (deriv f) 0 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l707_70704


namespace NUMINAMATH_CALUDE_smallest_bdf_value_l707_70745

theorem smallest_bdf_value (a b c d e f : ℕ+) : 
  let expr := (a : ℚ) / b * (c : ℚ) / d * (e : ℚ) / f
  (expr + 3 = ((a + 1 : ℕ+) : ℚ) / b * (c : ℚ) / d * (e : ℚ) / f) →
  (expr + 4 = (a : ℚ) / b * ((c + 1 : ℕ+) : ℚ) / d * (e : ℚ) / f) →
  (expr + 5 = (a : ℚ) / b * (c : ℚ) / d * ((e + 1 : ℕ+) : ℚ) / f) →
  (∀ k : ℕ+, (b * d * f : ℕ) = k → k ≥ 60) ∧ 
  (∃ b' d' f' : ℕ+, (b' * d' * f' : ℕ) = 60) :=
by sorry

end NUMINAMATH_CALUDE_smallest_bdf_value_l707_70745


namespace NUMINAMATH_CALUDE_former_apartment_size_l707_70711

/-- Calculates the size of John's former apartment given his new living situation and savings --/
theorem former_apartment_size
  (former_rent_per_sqft : ℝ)
  (new_apartment_cost : ℝ)
  (yearly_savings : ℝ)
  (h1 : former_rent_per_sqft = 2)
  (h2 : new_apartment_cost = 2800)
  (h3 : yearly_savings = 1200) :
  (new_apartment_cost / 2 + yearly_savings / 12) / former_rent_per_sqft = 750 :=
by sorry

end NUMINAMATH_CALUDE_former_apartment_size_l707_70711


namespace NUMINAMATH_CALUDE_train_length_calculation_l707_70796

/-- Calculates the length of a train given the speeds of a jogger and the train,
    the initial distance between them, and the time it takes for the train to pass the jogger. -/
theorem train_length_calculation (jogger_speed train_speed : ℝ) (initial_distance passing_time : ℝ) :
  jogger_speed = 9 * (5 / 18) →
  train_speed = 45 * (5 / 18) →
  initial_distance = 180 →
  passing_time = 30 →
  (train_speed - jogger_speed) * passing_time - initial_distance = 120 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l707_70796


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l707_70790

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (2 * (-1)^2 - 3 * m * (-1) + 1 = 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l707_70790


namespace NUMINAMATH_CALUDE_sin_2theta_value_l707_70716

theorem sin_2theta_value (θ : Real) (h : (Real.sqrt 2 * Real.cos (2 * θ)) / Real.cos (π / 4 + θ) = Real.sqrt 3 * Real.sin (2 * θ)) : 
  Real.sin (2 * θ) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l707_70716


namespace NUMINAMATH_CALUDE_factorization_m_squared_minus_3m_l707_70720

theorem factorization_m_squared_minus_3m (m : ℝ) : m^2 - 3*m = m*(m - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_m_squared_minus_3m_l707_70720


namespace NUMINAMATH_CALUDE_composition_result_l707_70736

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 4 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x + 2

-- State the theorem
theorem composition_result (c d : ℝ) :
  (∀ x, f c (g c x) = 12 * x + d) → d = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_composition_result_l707_70736


namespace NUMINAMATH_CALUDE_intersection_complement_problem_l707_70762

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

theorem intersection_complement_problem :
  N ∩ (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_problem_l707_70762


namespace NUMINAMATH_CALUDE_legs_in_pool_l707_70722

/-- The number of people in Karen and Donald's family -/
def karen_donald_family : ℕ := 8

/-- The number of people in Tom and Eva's family -/
def tom_eva_family : ℕ := 6

/-- The total number of people in both families -/
def total_people : ℕ := karen_donald_family + tom_eva_family

/-- The number of people not in the pool -/
def people_not_in_pool : ℕ := 6

/-- The number of legs per person -/
def legs_per_person : ℕ := 2

theorem legs_in_pool : 
  (total_people - people_not_in_pool) * legs_per_person = 16 := by
  sorry

end NUMINAMATH_CALUDE_legs_in_pool_l707_70722


namespace NUMINAMATH_CALUDE_line_through_point_l707_70793

theorem line_through_point (b : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + b) → (2 = 2 * 1 + b) → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l707_70793


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l707_70754

theorem solve_quadratic_equation (x : ℝ) (h1 : 3 * x^2 - 9 * x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l707_70754


namespace NUMINAMATH_CALUDE_park_perimeter_calculation_l707_70734

/-- The perimeter of a rectangular park with given length and breadth. -/
def park_perimeter (length breadth : ℝ) : ℝ :=
  2 * (length + breadth)

/-- Theorem stating that the perimeter of a rectangular park with length 300 m and breadth 200 m is 1000 m. -/
theorem park_perimeter_calculation :
  park_perimeter 300 200 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_park_perimeter_calculation_l707_70734


namespace NUMINAMATH_CALUDE_fruit_display_total_l707_70705

/-- The number of bananas on the display -/
def num_bananas : ℕ := 5

/-- The number of oranges on the display -/
def num_oranges : ℕ := 2 * num_bananas

/-- The number of apples on the display -/
def num_apples : ℕ := 2 * num_oranges

/-- The total number of fruits on the display -/
def total_fruits : ℕ := num_bananas + num_oranges + num_apples

theorem fruit_display_total :
  total_fruits = 35 :=
by sorry

end NUMINAMATH_CALUDE_fruit_display_total_l707_70705


namespace NUMINAMATH_CALUDE_max_y_over_x_l707_70786

theorem max_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 1) :
  ∃ (M : ℝ), M = Real.sqrt 3 / 3 ∧ ∀ (x' y' : ℝ), (x' - 2)^2 + y'^2 = 1 → |y' / x'| ≤ M := by
  sorry

end NUMINAMATH_CALUDE_max_y_over_x_l707_70786


namespace NUMINAMATH_CALUDE_smallest_number_of_oranges_l707_70777

/-- Represents the number of oranges in a container --/
def container_capacity : ℕ := 15

/-- Represents the number of containers that are not full --/
def short_containers : ℕ := 3

/-- Represents the number of oranges missing from each short container --/
def missing_oranges : ℕ := 2

/-- Represents the minimum number of oranges --/
def min_oranges : ℕ := 201

theorem smallest_number_of_oranges (n : ℕ) : 
  n * container_capacity - short_containers * missing_oranges > min_oranges →
  ∃ (m : ℕ), m * container_capacity - short_containers * missing_oranges > min_oranges ∧
             m * container_capacity - short_containers * missing_oranges ≤ 
             n * container_capacity - short_containers * missing_oranges →
  n * container_capacity - short_containers * missing_oranges ≥ 204 :=
by sorry

#check smallest_number_of_oranges

end NUMINAMATH_CALUDE_smallest_number_of_oranges_l707_70777


namespace NUMINAMATH_CALUDE_inequality_equivalence_l707_70756

theorem inequality_equivalence (x : ℝ) : (x - 1) / 2 ≤ -1 ↔ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l707_70756


namespace NUMINAMATH_CALUDE_set_equality_l707_70750

theorem set_equality : 
  {x : ℕ | x - 1 ≤ 2} = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_set_equality_l707_70750


namespace NUMINAMATH_CALUDE_cost_price_calculation_l707_70757

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 207)
  (h2 : profit_percentage = 0.15) : 
  ∃ (cost_price : ℝ), cost_price = 180 ∧ selling_price = cost_price * (1 + profit_percentage) :=
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l707_70757


namespace NUMINAMATH_CALUDE_circle_intersection_tangent_slope_l707_70770

noncomputable def C₁ (x y : ℝ) := x^2 + y^2 - 6*x + 4*y + 9 = 0

noncomputable def C₂ (m x y : ℝ) := (x + m)^2 + (y + m + 5)^2 = 2*m^2 + 8*m + 10

def on_coordinate_axes (x y : ℝ) := x = 0 ∨ y = 0

theorem circle_intersection_tangent_slope 
  (m : ℝ) (h_m : m ≠ -3) (x₀ y₀ : ℝ) (h_axes : on_coordinate_axes x₀ y₀)
  (h_tangent : ∃ (T₁_x T₁_y T₂_x T₂_y : ℝ), 
    C₁ T₁_x T₁_y ∧ C₂ m T₂_x T₂_y ∧ 
    (x₀ - T₁_x)^2 + (y₀ - T₁_y)^2 = (x₀ - T₂_x)^2 + (y₀ - T₂_y)^2) :
  (m = 5 → ∃! (n : ℕ), n = 2 ∧ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ C₂ m x₁ y₁ ∧ C₂ m x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) ∧
  (x₀ + y₀ + 1 = 0 ∧ ((x₀ = 0 ∧ y₀ = -1) ∨ (x₀ = -1 ∧ y₀ = 0))) ∧
  (∀ (k : ℝ), (∀ (x y : ℝ), C₁ x y → (y + 2 = k * (x - 3)) → 
    (∀ (m : ℝ), m ≠ -3 → ∃ (x' y' : ℝ), C₂ m x' y' ∧ y' + 2 = k * (x' - 3))) → k > 0) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_tangent_slope_l707_70770


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l707_70712

/-- Given a quadratic inequality and its solution set, prove the solution set of a related inequality -/
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Iic (-3 : ℝ) ∪ Set.Ici 4 = {x : ℝ | a * x^2 + b * x + c ≤ 0}) :
  {x : ℝ | -3 ≤ x ∧ x ≤ 5} = {x : ℝ | b * x^2 + 2 * a * x - c - 3 * b ≤ 0} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l707_70712


namespace NUMINAMATH_CALUDE_expression_simplification_l707_70763

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 2) :
  (a / (a^2 - 4*a + 4) + (a + 2) / (2*a - a^2)) / (2 / (a^2 - 2*a)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l707_70763


namespace NUMINAMATH_CALUDE_tim_payment_l707_70740

/-- The total amount Tim paid for his and his cat's medical visits -/
def total_payment (doctor_visit_cost : ℝ) (doctor_insurance_coverage : ℝ) 
  (cat_visit_cost : ℝ) (cat_insurance_coverage : ℝ) : ℝ :=
  (doctor_visit_cost - doctor_visit_cost * doctor_insurance_coverage) +
  (cat_visit_cost - cat_insurance_coverage)

/-- Theorem stating that Tim paid $135 in total -/
theorem tim_payment : 
  total_payment 300 0.75 120 60 = 135 := by
  sorry

end NUMINAMATH_CALUDE_tim_payment_l707_70740


namespace NUMINAMATH_CALUDE_binomial_coefficient_10_3_l707_70778

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_10_3_l707_70778


namespace NUMINAMATH_CALUDE_existence_of_equal_function_values_l707_70708

theorem existence_of_equal_function_values (n : ℕ) (h_n : n ≤ 44) 
  (f : ℕ+ × ℕ+ → Fin n) : 
  ∃ (i j l k m p : ℕ+), 
    f (i, j) = f (i, k) ∧ f (i, j) = f (l, j) ∧ f (i, j) = f (l, k) ∧
    1989 * m ≤ i ∧ i < l ∧ l < 1989 + 1989 * m ∧
    1989 * p ≤ j ∧ j < k ∧ k < 1989 + 1989 * p :=
by sorry

end NUMINAMATH_CALUDE_existence_of_equal_function_values_l707_70708


namespace NUMINAMATH_CALUDE_min_black_edges_on_border_l707_70702

/-- Represents a small square in the grid -/
structure SmallSquare where
  blackTriangles : Fin 4
  blackEdges : Fin 4

/-- Represents the 5x5 grid -/
def Grid := Matrix (Fin 5) (Fin 5) SmallSquare

/-- Checks if two adjacent small squares have consistent edge colors -/
def consistentEdges (s1 s2 : SmallSquare) : Prop :=
  s1.blackEdges = s2.blackEdges

/-- Counts the number of black edges on the border of the grid -/
def countBorderBlackEdges (g : Grid) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of black edges on the border -/
theorem min_black_edges_on_border (g : Grid) 
  (h1 : ∀ (i j : Fin 5), (g i j).blackTriangles = 3)
  (h2 : ∀ (i j k l : Fin 5), (j = k + 1 ∨ i = l + 1) → consistentEdges (g i j) (g k l)) :
  countBorderBlackEdges g ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_min_black_edges_on_border_l707_70702


namespace NUMINAMATH_CALUDE_exactly_two_valid_positions_l707_70746

/-- Represents a position where an additional square can be placed -/
inductive Position
| Left : Position
| Right : Position
| Top : Position
| Bottom : Position
| FrontLeft : Position
| FrontRight : Position

/-- Represents the 'F' shape configuration -/
structure FShape :=
  (squares : Fin 6 → Unit)

/-- Represents the modified shape with an additional square -/
structure ModifiedShape :=
  (base : FShape)
  (additional_square : Position)

/-- Predicate to check if a modified shape can be folded into a valid 3D structure -/
def can_fold_to_valid_structure (shape : ModifiedShape) : Prop :=
  sorry

/-- The main theorem stating there are exactly two valid positions -/
theorem exactly_two_valid_positions :
  ∃ (p₁ p₂ : Position), p₁ ≠ p₂ ∧
    (∀ (shape : ModifiedShape),
      can_fold_to_valid_structure shape ↔ shape.additional_square = p₁ ∨ shape.additional_square = p₂) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_positions_l707_70746


namespace NUMINAMATH_CALUDE_row_col_product_equality_l707_70723

theorem row_col_product_equality 
  (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h_row_col_sum : 
    a₁ + a₂ + a₃ = b₁ + b₂ + b₃ ∧ 
    b₁ + b₂ + b₃ = c₁ + c₂ + c₃ ∧ 
    c₁ + c₂ + c₃ = a₁ + b₁ + c₁ ∧ 
    a₁ + b₁ + c₁ = a₂ + b₂ + c₂ ∧ 
    a₂ + b₂ + c₂ = a₃ + b₃ + c₃) : 
  a₁*b₁*c₁ + a₂*b₂*c₂ + a₃*b₃*c₃ = a₁*a₂*a₃ + b₁*b₂*b₃ + c₁*c₂*c₃ :=
by
  sorry

end NUMINAMATH_CALUDE_row_col_product_equality_l707_70723


namespace NUMINAMATH_CALUDE_interior_triangle_area_l707_70701

theorem interior_triangle_area (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) :
  (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_interior_triangle_area_l707_70701


namespace NUMINAMATH_CALUDE_journey_duration_l707_70783

/-- Represents Tom's journey to Virgo island -/
def TomJourney : Prop :=
  let first_flight : ℝ := 5
  let first_layover : ℝ := 1
  let second_flight : ℝ := 2 * first_flight
  let second_layover : ℝ := 2
  let third_flight : ℝ := first_flight / 2
  let third_layover : ℝ := 3
  let first_boat : ℝ := 1.5
  let final_layover : ℝ := 0.75
  let final_boat : ℝ := 2 * (first_flight - third_flight)
  let total_time : ℝ := first_flight + first_layover + second_flight + second_layover +
                        third_flight + third_layover + first_boat + final_layover + final_boat
  total_time = 30.75

theorem journey_duration : TomJourney := by
  sorry

end NUMINAMATH_CALUDE_journey_duration_l707_70783


namespace NUMINAMATH_CALUDE_unique_integer_solution_to_equation_l707_70795

theorem unique_integer_solution_to_equation :
  ∀ x y : ℤ, x^4 + y^4 = 3*x^3*y → x = 0 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_solution_to_equation_l707_70795


namespace NUMINAMATH_CALUDE_modulus_of_z_l707_70743

theorem modulus_of_z : Complex.abs ((1 - Complex.I) * Complex.I) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l707_70743


namespace NUMINAMATH_CALUDE_museum_clock_position_l707_70767

/-- A special clock with the given properties -/
structure SpecialClock where
  positions : ℕ
  jump_interval : ℕ
  jump_distance : ℕ

/-- Calculate the position of the clock hand after a given number of minutes -/
def clock_position (clock : SpecialClock) (initial_position : ℕ) (minutes : ℕ) : ℕ :=
  (initial_position + (minutes / clock.jump_interval) * clock.jump_distance) % clock.positions

theorem museum_clock_position : 
  let clock := SpecialClock.mk 20 7 9
  let minutes_between_8pm_and_8am := 12 * 60
  clock_position clock 9 minutes_between_8pm_and_8am = 2 := by
  sorry

end NUMINAMATH_CALUDE_museum_clock_position_l707_70767


namespace NUMINAMATH_CALUDE_problem_solution_l707_70759

def M : Set ℝ := {y | ∃ x, y = 3^x}
def N : Set ℝ := {-1, 0, 1}

theorem problem_solution : (Set.univ \ M) ∩ N = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_problem_solution_l707_70759


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l707_70721

theorem polygon_angle_sum (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 + (180 - 180 / n)) = 2007 → 
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l707_70721


namespace NUMINAMATH_CALUDE_youngest_not_first_or_last_l707_70732

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

theorem youngest_not_first_or_last (total_people : ℕ) (youngest_person : ℕ) : 
  total_people = 5 → 
  youngest_person = 1 → 
  number_of_arrangements total_people - 
  (2 * number_of_arrangements (total_people - 1)) = 72 :=
sorry

end NUMINAMATH_CALUDE_youngest_not_first_or_last_l707_70732


namespace NUMINAMATH_CALUDE_angle_terminal_side_x_value_l707_70764

theorem angle_terminal_side_x_value (x : ℝ) (θ : ℝ) :
  x < 0 →
  (∃ y : ℝ, y = 3 ∧ (x^2 + y^2).sqrt * Real.cos θ = x) →
  Real.cos θ = (Real.sqrt 10 / 10) * x →
  x = -1 :=
by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_x_value_l707_70764


namespace NUMINAMATH_CALUDE_only_set_D_forms_triangle_l707_70791

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- The given sets of line segments -/
def set_A : Vector ℝ 3 := ⟨[5, 11, 6], by simp⟩
def set_B : Vector ℝ 3 := ⟨[8, 8, 16], by simp⟩
def set_C : Vector ℝ 3 := ⟨[10, 5, 4], by simp⟩
def set_D : Vector ℝ 3 := ⟨[6, 9, 14], by simp⟩

/-- Theorem: Among the given sets, only set D can form a triangle -/
theorem only_set_D_forms_triangle :
  ¬(can_form_triangle set_A[0] set_A[1] set_A[2]) ∧
  ¬(can_form_triangle set_B[0] set_B[1] set_B[2]) ∧
  ¬(can_form_triangle set_C[0] set_C[1] set_C[2]) ∧
  can_form_triangle set_D[0] set_D[1] set_D[2] :=
by sorry

end NUMINAMATH_CALUDE_only_set_D_forms_triangle_l707_70791


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l707_70730

theorem shaded_area_calculation (r : Real) (h : r = 1) : 
  6 * (π * r^2) + 4 * (1/2 * π * r^2) = 8 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l707_70730


namespace NUMINAMATH_CALUDE_ring_stack_height_is_117_l707_70771

/-- Calculates the distance from the top of the top ring to the bottom of the bottom ring in a stack of linked rings. -/
def ring_stack_height (top_diameter : ℝ) (top_thickness : ℝ) (bottom_diameter : ℝ) 
  (diameter_decrease : ℝ) (thickness_decrease : ℝ) : ℝ :=
  sorry

/-- The distance from the top of the top ring to the bottom of the bottom ring is 117 cm. -/
theorem ring_stack_height_is_117 : 
  ring_stack_height 30 2 10 2 0.1 = 117 := by sorry

end NUMINAMATH_CALUDE_ring_stack_height_is_117_l707_70771


namespace NUMINAMATH_CALUDE_stratified_sample_green_and_carp_l707_70707

/-- Represents the total number of fish -/
def total_fish : ℕ := 200

/-- Represents the sample size -/
def sample_size : ℕ := 20

/-- Represents the number of green fish -/
def green_fish : ℕ := 20

/-- Represents the number of carp -/
def carp : ℕ := 40

/-- Represents the sum of green fish and carp -/
def green_and_carp : ℕ := green_fish + carp

/-- Theorem stating the number of green fish and carp in the stratified sample -/
theorem stratified_sample_green_and_carp :
  (green_and_carp : ℚ) * sample_size / total_fish = 6 := by sorry

end NUMINAMATH_CALUDE_stratified_sample_green_and_carp_l707_70707


namespace NUMINAMATH_CALUDE_no_natural_solutions_l707_70780

theorem no_natural_solutions :
  ∀ (x y : ℕ), y^2 ≠ x^2 + x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l707_70780


namespace NUMINAMATH_CALUDE_child_admission_is_five_l707_70741

/-- Calculates the admission price for children given the following conditions:
  * Adult admission is $8
  * Total amount paid is $201
  * Total number of tickets is 33
  * Number of children's tickets is 21
-/
def childAdmissionPrice (adultPrice totalPaid totalTickets childTickets : ℕ) : ℕ :=
  (totalPaid - adultPrice * (totalTickets - childTickets)) / childTickets

/-- Proves that the admission price for children is $5 under the given conditions -/
theorem child_admission_is_five :
  childAdmissionPrice 8 201 33 21 = 5 := by
  sorry

end NUMINAMATH_CALUDE_child_admission_is_five_l707_70741


namespace NUMINAMATH_CALUDE_remaining_amount_after_purchase_l707_70779

def lollipop_price : ℚ := 1.5
def gummy_pack_price : ℚ := 2
def chips_price : ℚ := 1.25
def chocolate_price : ℚ := 1.75
def discount_rate : ℚ := 0.1
def tax_rate : ℚ := 0.05
def initial_amount : ℚ := 25

def total_cost : ℚ := 4 * lollipop_price + 2 * gummy_pack_price + 3 * chips_price + chocolate_price

def discounted_cost : ℚ := total_cost * (1 - discount_rate)

def final_cost : ℚ := discounted_cost * (1 + tax_rate)

theorem remaining_amount_after_purchase : 
  initial_amount - final_cost = 10.35 := by sorry

end NUMINAMATH_CALUDE_remaining_amount_after_purchase_l707_70779


namespace NUMINAMATH_CALUDE_license_plate_difference_l707_70742

/-- The number of letters in the English alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible license plates in Alphazia -/
def alphazia_plates : ℕ := num_letters^4 * num_digits^3

/-- The number of possible license plates in Betaland -/
def betaland_plates : ℕ := num_letters^5 * num_digits^2

/-- The difference in the number of possible license plates between Alphazia and Betaland -/
def plate_difference : ℤ := alphazia_plates - betaland_plates

theorem license_plate_difference :
  plate_difference = -731161600 := by sorry

end NUMINAMATH_CALUDE_license_plate_difference_l707_70742


namespace NUMINAMATH_CALUDE_highDiveVelocity_l707_70738

/-- The height function for a high-dive swimmer -/
def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

/-- The instantaneous velocity of the high-dive swimmer at t=1s -/
theorem highDiveVelocity : 
  (deriv h) 1 = -3.3 := by sorry

end NUMINAMATH_CALUDE_highDiveVelocity_l707_70738


namespace NUMINAMATH_CALUDE_expression_value_l707_70748

theorem expression_value : 3 * 4^2 - (8 / 2) = 44 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l707_70748


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l707_70724

theorem smallest_sum_of_squares (x y z : ℝ) : 
  (x + 4) * (y - 4) = 0 → 
  3 * z - 2 * y = 5 → 
  x^2 + y^2 + z^2 ≥ 457/9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l707_70724


namespace NUMINAMATH_CALUDE_consecutive_20_divisibility_l707_70755

theorem consecutive_20_divisibility (n : ℤ) : 
  (∃ k ∈ Finset.range 20, (n + k) % 9 = 0) ∧ 
  (∃ k ∈ Finset.range 20, (n + k) % 9 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_20_divisibility_l707_70755


namespace NUMINAMATH_CALUDE_simultaneous_strike_l707_70792

def cymbal_interval : ℕ := 7
def triangle_interval : ℕ := 2

theorem simultaneous_strike :
  ∃ (n : ℕ), n > 0 ∧ n % cymbal_interval = 0 ∧ n % triangle_interval = 0 ∧
  ∀ (m : ℕ), 0 < m ∧ m < n → (m % cymbal_interval ≠ 0 ∨ m % triangle_interval ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_strike_l707_70792


namespace NUMINAMATH_CALUDE_stating_max_valid_pairs_l707_70715

/-- Represents the maximum value that can be used in the pairs -/
def maxValue : ℕ := 2018

/-- Represents a pair of natural numbers (a, b) where a < b ≤ maxValue -/
structure ValidPair where
  a : ℕ
  b : ℕ
  h1 : a < b
  h2 : b ≤ maxValue

/-- Represents a set of valid pairs satisfying the given conditions -/
def ValidPairSet := Set ValidPair

/-- 
  Given a set of valid pairs, returns the number of pairs in the set
  satisfying the condition that if (a, b) is in the set, 
  then (c, a) and (b, d) are not in the set for any c and d
-/
def countValidPairs (s : ValidPairSet) : ℕ := sorry

/-- The maximum number of valid pairs that can be written on the board -/
def maxPairs : ℕ := 1018081

/-- 
  Theorem stating that the maximum number of valid pairs 
  that can be written on the board is 1018081
-/
theorem max_valid_pairs : 
  ∀ s : ValidPairSet, countValidPairs s ≤ maxPairs ∧ 
  ∃ s : ValidPairSet, countValidPairs s = maxPairs := by sorry

end NUMINAMATH_CALUDE_stating_max_valid_pairs_l707_70715


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l707_70717

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x ≠ 1 ∧ x ≠ 2) ↔ x^2 - 3*x + 2 = 0) ↔
  (x^2 - 3*x + 2 = 0 → x = 1 ∨ x = 2) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l707_70717


namespace NUMINAMATH_CALUDE_unequal_weight_l707_70703

-- Define the shapes as variables
variable (square circle big_circle triangle big_triangle : ℕ)

-- Define the balance conditions
def balance1 : Prop := 4 * square = big_circle + circle
def balance2 : Prop := 2 * circle + big_circle = 2 * triangle

-- Define the weight of the original combination
def original_weight : ℕ := triangle + big_circle + square

-- Define the weight of the option to be proven unequal
def option_d_weight : ℕ := 2 * big_triangle + square

-- Theorem statement
theorem unequal_weight 
  (h1 : balance1 square circle big_circle)
  (h2 : balance2 circle big_circle triangle)
  (h3 : big_triangle = triangle) :
  option_d_weight square big_triangle ≠ original_weight triangle big_circle square :=
sorry

end NUMINAMATH_CALUDE_unequal_weight_l707_70703


namespace NUMINAMATH_CALUDE_absent_student_percentage_l707_70799

theorem absent_student_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (boys_absent_fraction : ℚ)
  (girls_absent_fraction : ℚ)
  (h1 : total_students = 150)
  (h2 : boys = 90)
  (h3 : girls = 60)
  (h4 : boys_absent_fraction = 1 / 6)
  (h5 : girls_absent_fraction = 1 / 4)
  (h6 : total_students = boys + girls) :
  (↑boys * boys_absent_fraction + ↑girls * girls_absent_fraction) / ↑total_students = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_absent_student_percentage_l707_70799


namespace NUMINAMATH_CALUDE_disjoint_subsets_count_l707_70706

theorem disjoint_subsets_count (S : Finset ℕ) : 
  S = Finset.range 12 →
  (Finset.powerset S).card = 2^12 →
  let n := (3^12 - 2 * 2^12 + 1) / 2
  (n : ℕ) = 261625 ∧ n % 1000 = 625 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_count_l707_70706
