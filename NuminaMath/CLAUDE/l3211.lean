import Mathlib

namespace NUMINAMATH_CALUDE_complex_cube_root_sum_l3211_321177

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem complex_cube_root_sum (a b : ℝ) (h : i^3 = a - b*i) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_sum_l3211_321177


namespace NUMINAMATH_CALUDE_eggs_per_year_is_3320_l3211_321131

/-- Represents the number of eggs used for each family member on a given day --/
structure EggUsage where
  children : Nat
  husband : Nat
  lisa : Nat

/-- Represents the egg usage for each day of the week and holidays --/
structure WeeklyEggUsage where
  monday : EggUsage
  tuesday : EggUsage
  wednesday : EggUsage
  thursday : EggUsage
  friday : EggUsage
  holiday : EggUsage

/-- Calculates the total number of eggs used in a year based on the weekly egg usage and number of holidays --/
def totalEggsPerYear (usage : WeeklyEggUsage) (numHolidays : Nat) : Nat :=
  let weekdayTotal := 
    (usage.monday.children * 3 + usage.monday.husband + usage.monday.lisa) * 52 +
    (usage.tuesday.children * 2 + usage.tuesday.husband + usage.tuesday.lisa + 2) * 52 +
    (usage.wednesday.children * 4 + usage.wednesday.husband + usage.wednesday.lisa) * 52 +
    (usage.thursday.children * 3 + usage.thursday.husband + usage.thursday.lisa) * 52 +
    (usage.friday.children * 4 + usage.friday.husband + usage.friday.lisa) * 52
  let holidayTotal := (usage.holiday.children * 4 + usage.holiday.husband + usage.holiday.lisa) * numHolidays
  weekdayTotal + holidayTotal

/-- The main theorem to prove --/
theorem eggs_per_year_is_3320 : 
  ∃ (usage : WeeklyEggUsage) (numHolidays : Nat),
    usage.monday = EggUsage.mk 2 3 2 ∧
    usage.tuesday = EggUsage.mk 2 3 2 ∧
    usage.wednesday = EggUsage.mk 3 4 3 ∧
    usage.thursday = EggUsage.mk 1 2 1 ∧
    usage.friday = EggUsage.mk 2 3 2 ∧
    usage.holiday = EggUsage.mk 2 2 2 ∧
    numHolidays = 8 ∧
    totalEggsPerYear usage numHolidays = 3320 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_year_is_3320_l3211_321131


namespace NUMINAMATH_CALUDE_ground_school_cost_proof_l3211_321146

/-- Represents the cost of a private pilot course -/
def total_cost : ℕ := 1275

/-- Represents the additional cost of the flight portion compared to the ground school portion -/
def flight_additional_cost : ℕ := 625

/-- Represents the cost of the flight portion -/
def flight_cost : ℕ := 950

/-- Represents the cost of the ground school portion -/
def ground_school_cost : ℕ := total_cost - flight_cost

theorem ground_school_cost_proof : ground_school_cost = 325 := by
  sorry

end NUMINAMATH_CALUDE_ground_school_cost_proof_l3211_321146


namespace NUMINAMATH_CALUDE_square_root_of_factorial_fraction_l3211_321112

theorem square_root_of_factorial_fraction : 
  Real.sqrt (Nat.factorial 9 / 126) = 24 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_factorial_fraction_l3211_321112


namespace NUMINAMATH_CALUDE_mn_length_l3211_321199

/-- Triangle XYZ with given side lengths -/
structure Triangle (X Y Z : ℝ × ℝ) where
  xy_length : dist X Y = 130
  xz_length : dist X Z = 112
  yz_length : dist Y Z = 125

/-- L is the intersection of angle bisector of X with YZ -/
def L (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry

/-- K is the intersection of angle bisector of Y with XZ -/
def K (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry

/-- M is the foot of the perpendicular from Z to YK -/
def M (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry

/-- N is the foot of the perpendicular from Z to XL -/
def N (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The main theorem -/
theorem mn_length (X Y Z : ℝ × ℝ) (t : Triangle X Y Z) : 
  dist (M X Y Z) (N X Y Z) = 53.5 := by sorry

end NUMINAMATH_CALUDE_mn_length_l3211_321199


namespace NUMINAMATH_CALUDE_negation_equivalence_l3211_321121

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3211_321121


namespace NUMINAMATH_CALUDE_square_sum_of_coefficients_l3211_321113

theorem square_sum_of_coefficients (a b c : ℝ) : 
  36 - 4 * Real.sqrt 2 - 6 * Real.sqrt 3 + 12 * Real.sqrt 6 = (a * Real.sqrt 2 + b * Real.sqrt 3 + c)^2 →
  a^2 + b^2 + c^2 = 14 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_coefficients_l3211_321113


namespace NUMINAMATH_CALUDE_divide_decimals_l3211_321184

theorem divide_decimals : (0.08 : ℚ) / (0.002 : ℚ) = 40 := by sorry

end NUMINAMATH_CALUDE_divide_decimals_l3211_321184


namespace NUMINAMATH_CALUDE_max_xy_value_l3211_321149

theorem max_xy_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 2 * x + 3 * y = 4) :
  ∃ (M : ℝ), M = 2/3 ∧ xy ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀ * y₀ = M ∧ 2 * x₀ + 3 * y₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l3211_321149


namespace NUMINAMATH_CALUDE_distribute_negation_l3211_321155

theorem distribute_negation (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b := by
  sorry

end NUMINAMATH_CALUDE_distribute_negation_l3211_321155


namespace NUMINAMATH_CALUDE_trajectory_equation_1_trajectory_equation_1_converse_l3211_321154

/-- Given points A(3,0) and B(-3,0), and a point P(x,y) such that the product of slopes of AP and BP is -2,
    prove that the trajectory of P satisfies the equation x²/9 + y²/18 = 1 for x ≠ ±3 -/
theorem trajectory_equation_1 (x y : ℝ) (h : x ≠ 3 ∧ x ≠ -3) :
  (y / (x - 3)) * (y / (x + 3)) = -2 → x^2 / 9 + y^2 / 18 = 1 := by
sorry

/-- The converse: if a point P(x,y) satisfies x²/9 + y²/18 = 1 for x ≠ ±3,
    then the product of slopes of AP and BP is -2 -/
theorem trajectory_equation_1_converse (x y : ℝ) (h : x ≠ 3 ∧ x ≠ -3) :
  x^2 / 9 + y^2 / 18 = 1 → (y / (x - 3)) * (y / (x + 3)) = -2 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_1_trajectory_equation_1_converse_l3211_321154


namespace NUMINAMATH_CALUDE_red_shirt_pairs_l3211_321141

theorem red_shirt_pairs 
  (total_students : ℕ) 
  (green_students : ℕ) 
  (red_students : ℕ) 
  (total_pairs : ℕ) 
  (green_green_pairs : ℕ) : 
  total_students = 132 →
  green_students = 64 →
  red_students = 68 →
  total_pairs = 66 →
  green_green_pairs = 28 →
  ∃ (red_red_pairs : ℕ), red_red_pairs = 30 :=
by sorry

end NUMINAMATH_CALUDE_red_shirt_pairs_l3211_321141


namespace NUMINAMATH_CALUDE_brock_cookies_proof_l3211_321142

/-- Represents the number of cookies Brock bought -/
def brock_cookies : ℕ := 7

theorem brock_cookies_proof (total_cookies : ℕ) (stone_cookies : ℕ) (remaining_cookies : ℕ) 
  (h1 : total_cookies = 5 * 12)
  (h2 : stone_cookies = 2 * 12)
  (h3 : remaining_cookies = 15)
  (h4 : total_cookies = stone_cookies + 3 * brock_cookies + remaining_cookies) :
  brock_cookies = 7 := by
  sorry

end NUMINAMATH_CALUDE_brock_cookies_proof_l3211_321142


namespace NUMINAMATH_CALUDE_nested_sum_equals_2002_l3211_321174

def nested_sum (n : ℕ) : ℚ :=
  if n = 0 then 2
  else n + 1 + (1 / 2) * nested_sum (n - 1)

theorem nested_sum_equals_2002 : nested_sum 1001 = 2002 := by
  sorry

end NUMINAMATH_CALUDE_nested_sum_equals_2002_l3211_321174


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l3211_321187

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 0, f a x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 0, f a x = 1) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l3211_321187


namespace NUMINAMATH_CALUDE_find_divisor_l3211_321158

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 161 →
  quotient = 10 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 16 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3211_321158


namespace NUMINAMATH_CALUDE_conic_section_focus_l3211_321173

/-- The conic section defined by parametric equations x = t^2 and y = 2t -/
def conic_section (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

/-- The focus of the conic section -/
def focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the conic section defined by parametric equations x = t^2 and y = 2t is (1, 0) -/
theorem conic_section_focus :
  ∀ t : ℝ, ∃ a : ℝ, a > 0 ∧ (conic_section t).2^2 = 4 * a * (conic_section t).1 ∧ focus = (a, 0) :=
sorry

end NUMINAMATH_CALUDE_conic_section_focus_l3211_321173


namespace NUMINAMATH_CALUDE_isosceles_triangle_proof_l3211_321120

theorem isosceles_triangle_proof (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_equation : 2 * Real.sin A * Real.cos B = Real.sin C) : A = B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_proof_l3211_321120


namespace NUMINAMATH_CALUDE_time_to_put_30_toys_is_14_minutes_l3211_321118

/-- The time required to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (toys_in_per_cycle : ℕ) (toys_out_per_cycle : ℕ) (cycle_duration : ℕ) : ℚ :=
  let net_increase := toys_in_per_cycle - toys_out_per_cycle
  let cycles_needed := (total_toys - toys_in_per_cycle) / net_increase
  let total_seconds := cycles_needed * cycle_duration + cycle_duration
  total_seconds / 60

/-- Theorem: The time to put 30 toys in the box is 14 minutes -/
theorem time_to_put_30_toys_is_14_minutes :
  time_to_put_toys_in_box 30 3 2 30 = 14 := by
  sorry

end NUMINAMATH_CALUDE_time_to_put_30_toys_is_14_minutes_l3211_321118


namespace NUMINAMATH_CALUDE_sqrt_product_equals_120_sqrt_3_l3211_321114

theorem sqrt_product_equals_120_sqrt_3 : 
  Real.sqrt 75 * Real.sqrt 48 * Real.sqrt 12 = 120 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_120_sqrt_3_l3211_321114


namespace NUMINAMATH_CALUDE_rational_segment_existence_l3211_321171

theorem rational_segment_existence (f : ℚ → ℤ) :
  ∃ a b : ℚ, f a + f b ≤ 2 * f ((a + b) / 2) := by
  sorry

end NUMINAMATH_CALUDE_rational_segment_existence_l3211_321171


namespace NUMINAMATH_CALUDE_sum_squares_lengths_eq_k_squared_l3211_321126

/-- A regular k-gon inscribed in a unit circle -/
structure RegularKGon (k : ℕ) where
  (k_pos : k > 0)

/-- The sum of squares of lengths of all sides and diagonals of a regular k-gon -/
def sum_squares_lengths (k : ℕ) (P : RegularKGon k) : ℝ :=
  sorry

/-- Theorem: The sum of squares of lengths of all sides and diagonals of a regular k-gon
    inscribed in a unit circle is equal to k^2 -/
theorem sum_squares_lengths_eq_k_squared (k : ℕ) (P : RegularKGon k) :
  sum_squares_lengths k P = k^2 :=
sorry

end NUMINAMATH_CALUDE_sum_squares_lengths_eq_k_squared_l3211_321126


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3211_321127

theorem quadratic_minimum (x : ℝ) : x^2 + 6*x ≥ -9 ∧ ∃ y : ℝ, y^2 + 6*y = -9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3211_321127


namespace NUMINAMATH_CALUDE_compute_expression_l3211_321166

theorem compute_expression : 3 * 3^3 - 9^50 / 9^48 = 0 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3211_321166


namespace NUMINAMATH_CALUDE_number_problem_l3211_321170

theorem number_problem (x : ℚ) (h : (7/8) * x = 28) : (x + 16) * (5/16) = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3211_321170


namespace NUMINAMATH_CALUDE_integer_fraction_values_l3211_321153

theorem integer_fraction_values (k : ℤ) : 
  (∃ n : ℤ, (2 * k^2 + k - 8) / (k - 1) = n) ↔ k ∈ ({6, 2, 0, -4} : Set ℤ) := by
sorry

end NUMINAMATH_CALUDE_integer_fraction_values_l3211_321153


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3211_321135

theorem system_of_equations_solution (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) :
  (∃ (x y : ℝ), a₁ * x - b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂ ∧ x = 2 ∧ y = -1) →
  (∃ (x y : ℝ), a₁ * (x + 3) - b₁ * (y - 2) = c₁ ∧ a₂ * (x + 3) + b₂ * (y - 2) = c₂ ∧ x = -1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3211_321135


namespace NUMINAMATH_CALUDE_compound_interest_problem_l3211_321132

def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r) ^ t

theorem compound_interest_problem : 
  ∃ (P : ℝ) (r : ℝ), 
    compound_interest P r 2 = 8880 ∧ 
    compound_interest P r 3 = 9261 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l3211_321132


namespace NUMINAMATH_CALUDE_subset_condition_l3211_321190

def A : Set ℝ := {x | 3*x + 6 > 0 ∧ 2*x - 10 < 0}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m < 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_l3211_321190


namespace NUMINAMATH_CALUDE_income_left_is_2_15_percent_l3211_321185

/-- Calculates the percentage of income left after one year given initial expenses and yearly changes. -/
def income_left_after_one_year (
  food_expense : ℝ)
  (education_expense : ℝ)
  (transportation_expense : ℝ)
  (medical_expense : ℝ)
  (rent_percentage_of_remaining : ℝ)
  (expense_increase_rate : ℝ)
  (income_increase_rate : ℝ) : ℝ :=
  let initial_expenses := food_expense + education_expense + transportation_expense + medical_expense
  let remaining_after_initial := 1 - initial_expenses
  let initial_rent := remaining_after_initial * rent_percentage_of_remaining
  let increased_expenses := initial_expenses * (1 + expense_increase_rate)
  let new_remaining := 1 - increased_expenses
  let new_rent := new_remaining * rent_percentage_of_remaining
  1 - (increased_expenses + new_rent)

/-- Theorem stating that given the specified conditions, the percentage of income left after one year is 2.15%. -/
theorem income_left_is_2_15_percent :
  income_left_after_one_year 0.35 0.25 0.15 0.10 0.80 0.05 0.10 = 0.0215 := by
  sorry

end NUMINAMATH_CALUDE_income_left_is_2_15_percent_l3211_321185


namespace NUMINAMATH_CALUDE_hens_and_cows_l3211_321156

theorem hens_and_cows (total_animals : ℕ) (total_feet : ℕ) (hens : ℕ) (cows : ℕ) : 
  total_animals = 48 →
  total_feet = 140 →
  total_animals = hens + cows →
  total_feet = 2 * hens + 4 * cows →
  hens = 26 := by
sorry

end NUMINAMATH_CALUDE_hens_and_cows_l3211_321156


namespace NUMINAMATH_CALUDE_chores_ratio_l3211_321192

/-- Proves that the ratio of time spent on other chores to vacuuming is 3:1 -/
theorem chores_ratio (vacuum_time other_chores_time total_time : ℕ) : 
  vacuum_time = 3 → 
  total_time = 12 → 
  other_chores_time = total_time - vacuum_time →
  (other_chores_time : ℚ) / vacuum_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_chores_ratio_l3211_321192


namespace NUMINAMATH_CALUDE_tan_of_angle_on_x_plus_y_equals_zero_l3211_321150

/-- An angle whose terminal side lies on the line x + y = 0 -/
structure AngleOnXPlusYEqualsZero where
  α : Real
  terminal_side : ∀ (x y : Real), x + y = 0 → (∃ (t : Real), x = t * Real.cos α ∧ y = t * Real.sin α)

/-- The tangent of an angle whose terminal side lies on the line x + y = 0 is -1 -/
theorem tan_of_angle_on_x_plus_y_equals_zero (θ : AngleOnXPlusYEqualsZero) : Real.tan θ.α = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_of_angle_on_x_plus_y_equals_zero_l3211_321150


namespace NUMINAMATH_CALUDE_no_geometric_sequence_sin_angles_l3211_321161

theorem no_geometric_sequence_sin_angles :
  ¬∃ a : Real, 0 < a ∧ a < 2 * Real.pi ∧
  ∃ r : Real, (Real.sin (2 * a) = r * Real.sin a) ∧
             (Real.sin (3 * a) = r * Real.sin (2 * a)) := by
  sorry

end NUMINAMATH_CALUDE_no_geometric_sequence_sin_angles_l3211_321161


namespace NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l3211_321151

theorem unique_solution_ceiling_equation :
  ∃! b : ℝ, b + ⌈b⌉ = 25.3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l3211_321151


namespace NUMINAMATH_CALUDE_uniform_rod_weight_l3211_321139

/-- Represents the weight of a uniform rod -/
def rod_weight (length : ℝ) (weight_per_meter : ℝ) : ℝ :=
  length * weight_per_meter

/-- Theorem: For a uniform rod where 9 m weighs 34.2 kg, 11.25 m of the same rod weighs 42.75 kg -/
theorem uniform_rod_weight :
  ∀ (weight_per_meter : ℝ),
    rod_weight 9 weight_per_meter = 34.2 →
    rod_weight 11.25 weight_per_meter = 42.75 := by
  sorry

end NUMINAMATH_CALUDE_uniform_rod_weight_l3211_321139


namespace NUMINAMATH_CALUDE_hyperbola_proof_l3211_321178

/-- Given hyperbola -/
def given_hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

/-- Given ellipse -/
def given_ellipse (x y : ℝ) : Prop := y^2 / 8 + x^2 / 2 = 1

/-- Desired hyperbola -/
def desired_hyperbola (x y : ℝ) : Prop := y^2 / 2 - x^2 / 4 = 1

/-- The theorem to be proved -/
theorem hyperbola_proof :
  ∀ x y : ℝ,
  (∃ k : ℝ, k ≠ 0 ∧ given_hyperbola (k*x) (k*y)) ∧  -- Same asymptotes condition
  (∃ fx fy : ℝ, given_ellipse fx fy ∧ desired_hyperbola fx fy) →  -- Shared focus condition
  desired_hyperbola x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_proof_l3211_321178


namespace NUMINAMATH_CALUDE_first_pickup_fraction_proof_l3211_321148

/-- Represents the carrying capacity of the bus -/
def bus_capacity : ℕ := 80

/-- Represents the number of people waiting at the second pickup point -/
def second_pickup_waiting : ℕ := 50

/-- Represents the number of people who couldn't board at the second pickup point -/
def unable_to_board : ℕ := 18

/-- Represents the fraction of bus capacity that entered at the first pickup point -/
def first_pickup_fraction : ℚ := 3 / 5

theorem first_pickup_fraction_proof :
  first_pickup_fraction = (bus_capacity - (second_pickup_waiting - unable_to_board)) / bus_capacity :=
by sorry

end NUMINAMATH_CALUDE_first_pickup_fraction_proof_l3211_321148


namespace NUMINAMATH_CALUDE_apple_pile_count_l3211_321125

theorem apple_pile_count : ∃! n : ℕ,
  50 < n ∧ n < 70 ∧
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧
  n % 1 = 0 ∧ n % 2 = 0 ∧ n % 4 = 0 ∧ n % 6 = 0 ∧ n % 10 = 0 ∧
  n % 12 = 0 ∧ n % 15 = 0 ∧ n % 20 = 0 ∧ n % 30 = 0 ∧ n % 60 = 0 ∧
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_apple_pile_count_l3211_321125


namespace NUMINAMATH_CALUDE_white_square_area_main_white_square_area_l3211_321115

-- Define the cube's side length
def cubeSide : ℝ := 12

-- Define the total amount of blue paint
def totalBluePaint : ℝ := 432

-- Define the number of faces on a cube
def numFaces : ℕ := 6

-- Theorem statement
theorem white_square_area (cubeSide : ℝ) (totalBluePaint : ℝ) (numFaces : ℕ) :
  cubeSide > 0 →
  totalBluePaint > 0 →
  numFaces = 6 →
  let totalSurfaceArea := numFaces * cubeSide * cubeSide
  let bluePaintPerFace := totalBluePaint / numFaces
  let whiteSquareArea := cubeSide * cubeSide - bluePaintPerFace
  whiteSquareArea = 72 := by
  sorry

-- Main theorem using the defined constants
theorem main_white_square_area : 
  let totalSurfaceArea := numFaces * cubeSide * cubeSide
  let bluePaintPerFace := totalBluePaint / numFaces
  let whiteSquareArea := cubeSide * cubeSide - bluePaintPerFace
  whiteSquareArea = 72 := by
  sorry

end NUMINAMATH_CALUDE_white_square_area_main_white_square_area_l3211_321115


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l3211_321198

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x = -6 ∧ 
  (∀ y : ℝ, (y^2 - y - 30) / (y - 6) = 2 / (y + 4) → y ≥ x) ∧
  (x^2 - x - 30) / (x - 6) = 2 / (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l3211_321198


namespace NUMINAMATH_CALUDE_children_off_bus_l3211_321109

theorem children_off_bus (initial : ℕ) (remaining : ℕ) (h1 : initial = 43) (h2 : remaining = 21) :
  initial - remaining = 22 := by
  sorry

end NUMINAMATH_CALUDE_children_off_bus_l3211_321109


namespace NUMINAMATH_CALUDE_opposite_lateral_angle_is_90_l3211_321160

/-- A regular quadrangular pyramid -/
structure RegularQuadrangularPyramid where
  /-- The angle between a lateral face and the base plane -/
  lateral_base_angle : ℝ
  /-- The angle between a lateral face and the base plane is 45° -/
  angle_is_45 : lateral_base_angle = 45

/-- The angle between opposite lateral faces of the pyramid -/
def opposite_lateral_angle (p : RegularQuadrangularPyramid) : ℝ := sorry

/-- Theorem: In a regular quadrangular pyramid where the lateral face forms a 45° angle 
    with the base plane, the angle between opposite lateral faces is 90° -/
theorem opposite_lateral_angle_is_90 (p : RegularQuadrangularPyramid) :
  opposite_lateral_angle p = 90 := by sorry

end NUMINAMATH_CALUDE_opposite_lateral_angle_is_90_l3211_321160


namespace NUMINAMATH_CALUDE_prove_income_expenditure_ratio_l3211_321186

def income_expenditure_ratio (income savings : ℕ) : Prop :=
  ∃ (expenditure : ℕ),
    savings = income - expenditure ∧
    income * 8 = expenditure * 15

theorem prove_income_expenditure_ratio :
  income_expenditure_ratio 15000 7000 := by
  sorry

end NUMINAMATH_CALUDE_prove_income_expenditure_ratio_l3211_321186


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l3211_321130

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length

/-- Proof that the bridge length is approximately 131.98 meters -/
theorem bridge_length_proof :
  ∃ ε > 0, |bridge_length 110 36 24.198064154867613 - 131.98| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l3211_321130


namespace NUMINAMATH_CALUDE_coin_black_region_probability_l3211_321107

/-- The probability of a coin partially covering a black region on a specially painted square. -/
theorem coin_black_region_probability : 
  let square_side : ℝ := 10
  let triangle_leg : ℝ := 3
  let diamond_side : ℝ := 3 * Real.sqrt 2
  let coin_diameter : ℝ := 2
  let valid_region_side : ℝ := square_side - coin_diameter
  let valid_region_area : ℝ := valid_region_side ^ 2
  let triangle_area : ℝ := 1/2 * triangle_leg ^ 2
  let diamond_area : ℝ := diamond_side ^ 2
  let overlap_area : ℝ := 48 + 4 * Real.sqrt 2 + 2 * Real.pi
  overlap_area / valid_region_area = (48 + 4 * Real.sqrt 2 + 2 * Real.pi) / 64 := by
  sorry

end NUMINAMATH_CALUDE_coin_black_region_probability_l3211_321107


namespace NUMINAMATH_CALUDE_schedule_five_courses_nine_periods_l3211_321195

/-- The number of ways to schedule courses -/
def schedule_ways (n_courses n_periods : ℕ) : ℕ :=
  Nat.choose n_periods n_courses * Nat.factorial n_courses

/-- Theorem stating the number of ways to schedule 5 courses in 9 periods -/
theorem schedule_five_courses_nine_periods :
  schedule_ways 5 9 = 15120 := by
  sorry

end NUMINAMATH_CALUDE_schedule_five_courses_nine_periods_l3211_321195


namespace NUMINAMATH_CALUDE_exp_two_log_five_equals_twentyfive_l3211_321124

theorem exp_two_log_five_equals_twentyfive : 
  Real.exp (2 * Real.log 5) = 25 := by sorry

end NUMINAMATH_CALUDE_exp_two_log_five_equals_twentyfive_l3211_321124


namespace NUMINAMATH_CALUDE_james_amy_balloon_difference_l3211_321117

/-- 
Given that James has 232 balloons and Amy has 101 balloons, 
prove that James has 131 more balloons than Amy.
-/
theorem james_amy_balloon_difference : 
  let james_balloons : ℕ := 232
  let amy_balloons : ℕ := 101
  james_balloons - amy_balloons = 131 := by
sorry

end NUMINAMATH_CALUDE_james_amy_balloon_difference_l3211_321117


namespace NUMINAMATH_CALUDE_largest_root_range_l3211_321191

def polynomial (x b₃ b₂ b₁ b₀ : ℝ) : ℝ := x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀

def is_valid_coefficient (b : ℝ) : Prop := abs b < 3

theorem largest_root_range :
  ∃ s : ℝ, 3 < s ∧ s < 4 ∧
  (∀ b₃ b₂ b₁ b₀ : ℝ, is_valid_coefficient b₃ → is_valid_coefficient b₂ →
    is_valid_coefficient b₁ → is_valid_coefficient b₀ →
    (∀ x : ℝ, x > s → polynomial x b₃ b₂ b₁ b₀ ≠ 0)) ∧
  (∃ b₃ b₂ b₁ b₀ : ℝ, is_valid_coefficient b₃ ∧ is_valid_coefficient b₂ ∧
    is_valid_coefficient b₁ ∧ is_valid_coefficient b₀ ∧
    polynomial s b₃ b₂ b₁ b₀ = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_root_range_l3211_321191


namespace NUMINAMATH_CALUDE_max_difference_is_five_point_five_l3211_321180

/-- A structure representing a set of segments on the ray (0, +∞) -/
structure SegmentSet where
  /-- The left end of the leftmost segment -/
  a : ℝ
  /-- The right end of the rightmost segment -/
  b : ℝ
  /-- The number of segments (more than two) -/
  n : ℕ
  /-- n > 2 -/
  h_n : n > 2
  /-- a > 0 -/
  h_a : a > 0
  /-- b > a -/
  h_b : b > a
  /-- For any two different segments, there exist numbers that differ by a factor of 2 -/
  factor_of_two : ∀ i j, i ≠ j → i < n → j < n → ∃ x y, x ∈ Set.Icc (a + i) (a + i + 1) ∧ y ∈ Set.Icc (a + j) (a + j + 1) ∧ (x = 2 * y ∨ y = 2 * x)

/-- The theorem stating that the maximum value of b - a is 5.5 -/
theorem max_difference_is_five_point_five (s : SegmentSet) : 
  (∃ (s' : SegmentSet), s'.b - s'.a ≥ s.b - s.a) → s.b - s.a ≤ 5.5 := by
  sorry

end NUMINAMATH_CALUDE_max_difference_is_five_point_five_l3211_321180


namespace NUMINAMATH_CALUDE_inverse_square_problem_l3211_321136

-- Define the relationship between x and y
def inverse_square_relation (k : ℝ) (x y : ℝ) : Prop :=
  x = k / (y ^ 2)

theorem inverse_square_problem (k : ℝ) :
  inverse_square_relation k 1 3 →
  inverse_square_relation k (1/9) 9 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_square_problem_l3211_321136


namespace NUMINAMATH_CALUDE_intersection_complement_M_and_N_l3211_321102

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}

theorem intersection_complement_M_and_N : 
  (Set.univ \ M) ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_M_and_N_l3211_321102


namespace NUMINAMATH_CALUDE_daisies_per_bouquet_l3211_321167

/-- Represents a flower shop selling bouquets of roses and daisies. -/
structure FlowerShop where
  roses_per_bouquet : ℕ
  total_bouquets : ℕ
  rose_bouquets : ℕ
  daisy_bouquets : ℕ
  total_flowers : ℕ

/-- Theorem stating the number of daisies in each bouquet. -/
theorem daisies_per_bouquet (shop : FlowerShop)
  (h1 : shop.roses_per_bouquet = 12)
  (h2 : shop.total_bouquets = 20)
  (h3 : shop.rose_bouquets = 10)
  (h4 : shop.daisy_bouquets = 10)
  (h5 : shop.total_flowers = 190)
  (h6 : shop.total_bouquets = shop.rose_bouquets + shop.daisy_bouquets) :
  (shop.total_flowers - shop.roses_per_bouquet * shop.rose_bouquets) / shop.daisy_bouquets = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_daisies_per_bouquet_l3211_321167


namespace NUMINAMATH_CALUDE_range_of_a_l3211_321110

-- Define the function f(x) = |x+3| - |x-1|
def f (x : ℝ) : ℝ := |x + 3| - |x - 1|

-- Define the property that the solution set is non-empty
def has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, f x ≤ a^2 - 5*a

-- Theorem statement
theorem range_of_a (a : ℝ) :
  has_solution a → (a ≥ 4 ∨ a ≤ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3211_321110


namespace NUMINAMATH_CALUDE_quadruple_batch_cans_l3211_321101

/-- Represents the number of cans for each ingredient in a normal batch of chili --/
structure NormalBatch where
  chilis : ℕ
  beans : ℕ
  tomatoes : ℕ

/-- Defines a normal batch of chili according to Carla's recipe --/
def carla_normal_batch : NormalBatch where
  chilis := 1
  beans := 2
  tomatoes := 3  -- 50% more than beans, so 2 * 1.5 = 3

/-- Calculates the total number of cans for a given batch size --/
def total_cans (batch : NormalBatch) (multiplier : ℕ) : ℕ :=
  multiplier * (batch.chilis + batch.beans + batch.tomatoes)

/-- Theorem: A quadruple batch of Carla's chili requires 24 cans of food --/
theorem quadruple_batch_cans : 
  total_cans carla_normal_batch 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_quadruple_batch_cans_l3211_321101


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3211_321194

theorem complex_magnitude_product : Complex.abs ((12 - 9*Complex.I) * (8 + 15*Complex.I)) = 255 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3211_321194


namespace NUMINAMATH_CALUDE_sqrt_294_simplification_l3211_321128

theorem sqrt_294_simplification : Real.sqrt 294 = 7 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_294_simplification_l3211_321128


namespace NUMINAMATH_CALUDE_farmer_apples_l3211_321123

theorem farmer_apples (apples_given : ℕ) (apples_left : ℕ) : apples_given = 88 → apples_left = 39 → apples_given + apples_left = 127 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l3211_321123


namespace NUMINAMATH_CALUDE_lcm_of_10_and_21_l3211_321165

theorem lcm_of_10_and_21 : Nat.lcm 10 21 = 210 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_10_and_21_l3211_321165


namespace NUMINAMATH_CALUDE_last_term_is_zero_l3211_321175

def first_term : ℤ := 0
def differences : List ℤ := [2, 4, -1, 0, -5, -3, 3]

theorem last_term_is_zero :
  first_term + differences.sum = 0 := by sorry

end NUMINAMATH_CALUDE_last_term_is_zero_l3211_321175


namespace NUMINAMATH_CALUDE_expression_simplification_l3211_321163

theorem expression_simplification :
  ((3 + 5 + 7 + 9) / 3) - ((4 * 6 + 13) / 5) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3211_321163


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_all_ones_l3211_321137

def is_all_ones (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 1

theorem smallest_multiplier_for_all_ones :
  ∃! N : ℕ, (N > 0) ∧ 
    is_all_ones (999999 * N) ∧
    (∀ m : ℕ, m > 0 → is_all_ones (999999 * m) → N ≤ m) ∧
    N = 111112 := by sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_all_ones_l3211_321137


namespace NUMINAMATH_CALUDE_function_value_at_cos_15_degrees_l3211_321164

theorem function_value_at_cos_15_degrees 
  (f : ℝ → ℝ) 
  (h : ∀ x, f (Real.sin x) = Real.cos (2 * x) - 1) :
  f (Real.cos (15 * π / 180)) = -Real.sqrt 3 / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_cos_15_degrees_l3211_321164


namespace NUMINAMATH_CALUDE_train_length_calculation_train_length_is_120m_l3211_321122

/-- Given a jogger and a train moving in the same direction, calculate the length of the train. -/
theorem train_length_calculation (jogger_speed : ℝ) (train_speed : ℝ) 
  (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let distance_covered := relative_speed * passing_time
  distance_covered - initial_distance

/-- The length of the train is 120 meters given the specified conditions. -/
theorem train_length_is_120m : 
  train_length_calculation 9 45 270 39 = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_train_length_is_120m_l3211_321122


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_unique_minimum_l3211_321104

theorem min_value_theorem (x : ℝ) (h : x > 0) : x^2 + 10*x + 100/x^3 ≥ 40 := by
  sorry

theorem equality_condition : ∃ x > 0, x^2 + 10*x + 100/x^3 = 40 := by
  sorry

theorem unique_minimum (x : ℝ) (h1 : x > 0) (h2 : x^2 + 10*x + 100/x^3 = 40) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_unique_minimum_l3211_321104


namespace NUMINAMATH_CALUDE_fred_marbles_l3211_321134

theorem fred_marbles (total : ℕ) (dark_blue : ℕ) (green : ℕ) (yellow : ℕ) (red : ℕ) : 
  total = 120 →
  dark_blue ≥ total / 3 →
  green = 10 →
  yellow = 5 →
  red = total - (dark_blue + green + yellow) →
  red = 65 := by
  sorry

end NUMINAMATH_CALUDE_fred_marbles_l3211_321134


namespace NUMINAMATH_CALUDE_matthew_crackers_l3211_321181

theorem matthew_crackers (friends : ℕ) (cakes : ℕ) (eaten_crackers : ℕ) :
  friends = 4 →
  cakes = 98 →
  eaten_crackers = 8 →
  ∃ (initial_crackers : ℕ),
    initial_crackers = 128 ∧
    ∃ (given_per_friend : ℕ),
      given_per_friend * friends ≤ cakes ∧
      given_per_friend * friends ≤ initial_crackers ∧
      initial_crackers = given_per_friend * friends + eaten_crackers * friends :=
by
  sorry

end NUMINAMATH_CALUDE_matthew_crackers_l3211_321181


namespace NUMINAMATH_CALUDE_cafeteria_green_apples_l3211_321169

theorem cafeteria_green_apples :
  let red_apples : ℕ := 43
  let students_wanting_fruit : ℕ := 2
  let extra_apples : ℕ := 73
  let green_apples : ℕ := red_apples + extra_apples + students_wanting_fruit - red_apples
  green_apples = 32 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_green_apples_l3211_321169


namespace NUMINAMATH_CALUDE_a_divisible_by_133_l3211_321129

/-- Sequence definition -/
def a (n : ℕ) : ℕ := 11^(n+2) + 12^(2*n+1)

/-- Main theorem: a_n is divisible by 133 for all n ≥ 0 -/
theorem a_divisible_by_133 (n : ℕ) : 133 ∣ a n := by sorry

end NUMINAMATH_CALUDE_a_divisible_by_133_l3211_321129


namespace NUMINAMATH_CALUDE_stratified_sampling_survey_l3211_321179

/-- Given a stratified sampling survey with a total population of 2400 (including 1000 female students),
    if 80 female students are included in a sample of size n, and the sampling fraction is consistent
    across all groups, then n = 192. -/
theorem stratified_sampling_survey (total_population : ℕ) (female_students : ℕ) (sample_size : ℕ) 
    (sampled_females : ℕ) (h1 : total_population = 2400) (h2 : female_students = 1000) 
    (h3 : sampled_females = 80) (h4 : sampled_females * total_population = sample_size * female_students) : 
    sample_size = 192 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_survey_l3211_321179


namespace NUMINAMATH_CALUDE_calculate_expression_l3211_321116

theorem calculate_expression : 
  |-Real.sqrt 3| + (1/2)⁻¹ + (Real.pi + 1)^0 - Real.tan (60 * π / 180) = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3211_321116


namespace NUMINAMATH_CALUDE_fabric_sale_meters_l3211_321193

-- Define the price per meter in kopecks
def price_per_meter : ℕ := 436

-- Define the maximum revenue in kopecks
def max_revenue : ℕ := 50000

-- Define a predicate for valid revenue
def valid_revenue (x : ℕ) : Prop :=
  (price_per_meter * x) % 1000 = 728 ∧
  price_per_meter * x ≤ max_revenue

-- Theorem statement
theorem fabric_sale_meters :
  ∃ (x : ℕ), valid_revenue x ∧ x = 98 := by sorry

end NUMINAMATH_CALUDE_fabric_sale_meters_l3211_321193


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3211_321145

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3211_321145


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3211_321159

/-- The volume of a tetrahedron given its edge lengths -/
def tetrahedron_volume (ab ac ad cd bd bc : ℝ) : ℝ :=
  -- Definition to be filled
  sorry

/-- Theorem: The volume of the specific tetrahedron is 48 cubic units -/
theorem specific_tetrahedron_volume :
  tetrahedron_volume 6 7 8 9 10 11 = 48 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3211_321159


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3211_321152

theorem sum_of_two_numbers (larger smaller : ℕ) : 
  larger = 22 → larger - smaller = 10 → larger + smaller = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3211_321152


namespace NUMINAMATH_CALUDE_coneSurface_is_cone_l3211_321108

/-- A surface in spherical coordinates (ρ, θ, φ) defined by ρ = c sin φ, where c is a positive constant -/
def coneSurface (c : ℝ) (h : c > 0) (ρ θ φ : ℝ) : Prop :=
  ρ = c * Real.sin φ

/-- The shape described by the coneSurface is a cone -/
theorem coneSurface_is_cone (c : ℝ) (h : c > 0) :
  ∃ (cone : Set (ℝ × ℝ × ℝ)), ∀ (ρ θ φ : ℝ),
    coneSurface c h ρ θ φ ↔ (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) ∈ cone :=
sorry

end NUMINAMATH_CALUDE_coneSurface_is_cone_l3211_321108


namespace NUMINAMATH_CALUDE_sixteen_power_divided_by_four_l3211_321183

theorem sixteen_power_divided_by_four (n : ℕ) : n = 16^2023 → n/4 = 4^4045 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_power_divided_by_four_l3211_321183


namespace NUMINAMATH_CALUDE_line_segment_proportion_l3211_321103

theorem line_segment_proportion (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a / c = c / b → a = 4 → b = 9 → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_proportion_l3211_321103


namespace NUMINAMATH_CALUDE_system_solution_equation_solution_l3211_321197

-- System of equations
theorem system_solution :
  ∃! (x y : ℝ), x + 2*y = 9 ∧ 3*x - 2*y = 3 ∧ x = 3 ∧ y = 3 := by sorry

-- Single equation
theorem equation_solution :
  ∃! (x : ℝ), x ≠ 3 ∧ (2-x)/(x-3) + 3 = 2/(3-x) ∧ x = 5/2 := by sorry

end NUMINAMATH_CALUDE_system_solution_equation_solution_l3211_321197


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3211_321143

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_edge_length := 1.5 * L
  let new_area := 6 * new_edge_length^2
  (new_area - original_area) / original_area * 100 = 125 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3211_321143


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l3211_321182

/-- A quadrilateral with specific properties -/
structure Quadrilateral :=
  (EF HG EH FG : ℕ)
  (right_angle_F : EF ^ 2 + FG ^ 2 = 25)
  (right_angle_H : EH ^ 2 + HG ^ 2 = 25)
  (different_sides : ∃ (a b : ℕ), (a ≠ b) ∧ ((a = EF ∧ b = FG) ∨ (a = EH ∧ b = HG) ∨ (a = EF ∧ b = HG) ∨ (a = EH ∧ b = FG)))

/-- The area of the quadrilateral EFGH is 12 -/
theorem area_of_quadrilateral (q : Quadrilateral) : (q.EF * q.FG + q.EH * q.HG) / 2 = 12 :=
sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l3211_321182


namespace NUMINAMATH_CALUDE_expression_evaluation_l3211_321168

theorem expression_evaluation :
  (3^102 + 7^103)^2 - (3^102 - 7^103)^2 = 240 * 10^206 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3211_321168


namespace NUMINAMATH_CALUDE_fibonacci_6_l3211_321119

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_6 : fibonacci 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_6_l3211_321119


namespace NUMINAMATH_CALUDE_square_not_always_positive_l3211_321133

theorem square_not_always_positive : ¬(∀ a : ℝ, a^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l3211_321133


namespace NUMINAMATH_CALUDE_energetic_time_proof_l3211_321106

def initial_speed : ℝ := 25
def tired_speed : ℝ := 15
def rest_time : ℝ := 0.5
def total_distance : ℝ := 132
def total_time : ℝ := 8

theorem energetic_time_proof :
  ∃ x : ℝ, 
    x ≥ 0 ∧
    x ≤ total_time - rest_time ∧
    initial_speed * x + tired_speed * (total_time - rest_time - x) = total_distance ∧
    x = 39 / 20 := by
  sorry

end NUMINAMATH_CALUDE_energetic_time_proof_l3211_321106


namespace NUMINAMATH_CALUDE_probability_P_equals_1_plus_i_l3211_321144

/-- The set of vertices of a regular hexagon in the complex plane -/
def V : Set ℂ := {1, -1, Complex.I, -Complex.I, (1/2) + (Real.sqrt 3 / 2) * Complex.I, -(1/2) - (Real.sqrt 3 / 2) * Complex.I}

/-- The number of elements chosen from V -/
def n : ℕ := 10

/-- The product of n randomly chosen elements from V -/
noncomputable def P : ℂ := sorry

/-- The probability that P equals 1 + i -/
noncomputable def prob_P_equals_1_plus_i : ℝ := sorry

/-- Theorem stating the probability of P equaling 1 + i -/
theorem probability_P_equals_1_plus_i : prob_P_equals_1_plus_i = 120 / 24649 := by sorry

end NUMINAMATH_CALUDE_probability_P_equals_1_plus_i_l3211_321144


namespace NUMINAMATH_CALUDE_x_power_n_plus_reciprocal_l3211_321196

theorem x_power_n_plus_reciprocal (θ : ℝ) (x : ℂ) (n : ℕ) (h1 : 0 < θ) (h2 : θ < π) (h3 : x + 1/x = 2 * Real.cos θ) :
  x^n + 1/x^n = 2 * Real.cos (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_x_power_n_plus_reciprocal_l3211_321196


namespace NUMINAMATH_CALUDE_divisibility_equation_solutions_l3211_321189

theorem divisibility_equation_solutions (n x y z t : ℕ+) :
  (n ^ x.val ∣ n ^ y.val + n ^ z.val) ∧ (n ^ y.val + n ^ z.val = n ^ t.val) →
  ((n = 2 ∧ y = x ∧ z = x + 1 ∧ t = x + 2) ∨
   (n = 3 ∧ y = x ∧ z = x ∧ t = x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_equation_solutions_l3211_321189


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_l3211_321162

-- Define the original proposition
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 3*x + m = 0

-- Define the contrapositive
def contrapositive (P Q : Prop) : Prop := ¬Q → ¬P

-- Theorem statement
theorem contrapositive_real_roots :
  contrapositive (m < 0) (has_real_roots m) ↔ (¬(has_real_roots m) → m ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_l3211_321162


namespace NUMINAMATH_CALUDE_arcsin_of_one_equals_pi_div_two_l3211_321111

theorem arcsin_of_one_equals_pi_div_two : Real.arcsin 1 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_of_one_equals_pi_div_two_l3211_321111


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l3211_321147

/-- Represents the number of ways to arrange frogs with color restrictions -/
def frog_arrangements (n_green n_red n_blue : ℕ) : ℕ :=
  2 * (n_red.factorial * n_green.factorial)

/-- Theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  frog_arrangements 3 4 1 = 288 :=
by sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l3211_321147


namespace NUMINAMATH_CALUDE_base_notes_on_hour_l3211_321172

/-- Represents the number of notes rung at each quarter-hour mark --/
def quarter_hour_notes : Fin 3 → ℕ
| 0 => 2  -- quarter past
| 1 => 4  -- half past
| 2 => 6  -- three-quarters past

/-- The total number of notes rung from 1:00 p.m. to 5:00 p.m. --/
def total_notes : ℕ := 103

/-- The number of hours from 1:00 p.m. to 5:00 p.m. --/
def hours : ℕ := 5

/-- Calculates the total notes rung at quarter-hour marks between two consecutive hours --/
def notes_between_hours : ℕ := (Finset.sum Finset.univ quarter_hour_notes)

/-- Theorem stating that the number of base notes rung on the hour is 8 --/
theorem base_notes_on_hour : 
  ∃ (B : ℕ), 
    hours * B + (Finset.sum (Finset.range (hours + 1)) id) + 
    (hours - 1) * notes_between_hours = total_notes ∧ B = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_notes_on_hour_l3211_321172


namespace NUMINAMATH_CALUDE_work_completion_time_l3211_321176

theorem work_completion_time (a b : ℝ) 
  (h1 : a + b = 1 / 16)  -- A and B together finish in 16 days
  (h2 : a = 1 / 32)      -- A alone finishes in 32 days
  : 1 / b = 32 :=        -- B alone finishes in 32 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3211_321176


namespace NUMINAMATH_CALUDE_iphone_discount_l3211_321100

theorem iphone_discount (iphone_price iwatch_price iwatch_discount cashback_rate final_price : ℝ) :
  iphone_price = 800 →
  iwatch_price = 300 →
  iwatch_discount = 0.1 →
  cashback_rate = 0.02 →
  final_price = 931 →
  ∃ (iphone_discount : ℝ),
    iphone_discount = 0.15 ∧
    final_price = (1 - cashback_rate) * (iphone_price * (1 - iphone_discount) + iwatch_price * (1 - iwatch_discount)) :=
by sorry

end NUMINAMATH_CALUDE_iphone_discount_l3211_321100


namespace NUMINAMATH_CALUDE_x_squared_positive_necessary_not_sufficient_l3211_321140

theorem x_squared_positive_necessary_not_sufficient :
  (∀ x : ℝ, x > 0 → x^2 > 0) ∧
  (∃ x : ℝ, x^2 > 0 ∧ x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_positive_necessary_not_sufficient_l3211_321140


namespace NUMINAMATH_CALUDE_common_roots_of_polynomials_l3211_321188

theorem common_roots_of_polynomials :
  let f (x : ℝ) := x^4 + 2*x^3 - x^2 - 2*x - 3
  let g (x : ℝ) := x^4 + 3*x^3 + x^2 - 4*x - 6
  let r₁ := (-1 + Real.sqrt 13) / 2
  let r₂ := (-1 - Real.sqrt 13) / 2
  (f r₁ = 0 ∧ f r₂ = 0) ∧ (g r₁ = 0 ∧ g r₂ = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_common_roots_of_polynomials_l3211_321188


namespace NUMINAMATH_CALUDE_min_eating_time_is_23_5_l3211_321157

/-- Represents the eating rates and constraints for Amy and Ben -/
structure EatingProblem where
  total_carrots : ℕ
  total_muffins : ℕ
  wait_time : ℕ
  amy_carrot_rate : ℕ
  amy_muffin_rate : ℕ
  ben_carrot_rate : ℕ
  ben_muffin_rate : ℕ

/-- Calculates the minimum time to eat all food given the problem constraints -/
def min_eating_time (problem : EatingProblem) : ℚ :=
  sorry

/-- Theorem stating that the minimum eating time for the given problem is 23.5 minutes -/
theorem min_eating_time_is_23_5 : 
  let problem : EatingProblem := {
    total_carrots := 1000
    total_muffins := 1000
    wait_time := 5
    amy_carrot_rate := 40
    amy_muffin_rate := 70
    ben_carrot_rate := 60
    ben_muffin_rate := 30
  }
  min_eating_time problem = 47/2 := by
  sorry

end NUMINAMATH_CALUDE_min_eating_time_is_23_5_l3211_321157


namespace NUMINAMATH_CALUDE_ticket_sales_total_l3211_321105

/-- Calculates the total money collected from ticket sales -/
def total_money_collected (adult_price child_price : ℕ) (total_tickets children_tickets : ℕ) : ℕ :=
  let adult_tickets := total_tickets - children_tickets
  adult_price * adult_tickets + child_price * children_tickets

/-- Theorem stating that the total money collected is $104 -/
theorem ticket_sales_total : 
  total_money_collected 6 4 21 11 = 104 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_total_l3211_321105


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3211_321138

theorem quadratic_inequality_solution_set (a b c : ℝ) (h1 : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c < 0) → (a < 0 ∧ b^2 - 4*a*c < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3211_321138
