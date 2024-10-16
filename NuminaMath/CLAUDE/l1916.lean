import Mathlib

namespace NUMINAMATH_CALUDE_paperback_count_l1916_191622

theorem paperback_count (total_books hardbacks selections : ℕ) : 
  total_books = 6 → 
  hardbacks = 4 → 
  selections = 14 →
  (∃ paperbacks : ℕ, 
    paperbacks + hardbacks = total_books ∧
    paperbacks = 2 ↔ 
    (Nat.choose paperbacks 1 * Nat.choose hardbacks 3 +
     Nat.choose paperbacks 2 * Nat.choose hardbacks 2 = selections)) :=
by sorry

end NUMINAMATH_CALUDE_paperback_count_l1916_191622


namespace NUMINAMATH_CALUDE_distance_to_destination_l1916_191621

/-- Proves that the distance to a destination is 144 km given specific rowing conditions --/
theorem distance_to_destination (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) : 
  rowing_speed = 10 →
  current_speed = 2 →
  total_time = 30 →
  let downstream_speed := rowing_speed + current_speed
  let upstream_speed := rowing_speed - current_speed
  ∃ (distance : ℝ), 
    distance / downstream_speed + distance / upstream_speed = total_time ∧
    distance = 144 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_destination_l1916_191621


namespace NUMINAMATH_CALUDE_parabola_properties_l1916_191641

-- Define the function f(x) = -x^2
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem parabola_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f x < f y) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1916_191641


namespace NUMINAMATH_CALUDE_extremum_derivative_zero_relation_l1916_191661

/-- A function f has a derivative at x₀ -/
def has_derivative_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ (f' : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - f x₀ - f' * (x - x₀)| ≤ ε * |x - x₀|

/-- x₀ is an extremum point of f -/
def is_extremum_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x₀ ≤ f x ∨ f x₀ ≥ f x

/-- The derivative of f at x₀ is 0 -/
def derivative_zero (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ (f' : ℝ), has_derivative_at f x₀ ∧ f' = 0

theorem extremum_derivative_zero_relation (f : ℝ → ℝ) (x₀ : ℝ) :
  (has_derivative_at f x₀ →
    (is_extremum_point f x₀ → derivative_zero f x₀) ∧
    ¬(derivative_zero f x₀ → is_extremum_point f x₀)) :=
sorry

end NUMINAMATH_CALUDE_extremum_derivative_zero_relation_l1916_191661


namespace NUMINAMATH_CALUDE_anthony_pencil_count_l1916_191633

/-- Given Anthony's initial pencil count and the number of pencils Kathryn gives him,
    prove that the total number of pencils Anthony has is equal to the sum of these two quantities. -/
theorem anthony_pencil_count (initial : ℕ) (given : ℕ) : initial + given = initial + given :=
by sorry

end NUMINAMATH_CALUDE_anthony_pencil_count_l1916_191633


namespace NUMINAMATH_CALUDE_approximation_place_l1916_191636

def number : ℕ := 345000000

theorem approximation_place (n : ℕ) (h : n = number) : 
  ∃ (k : ℕ), n ≥ 10^6 ∧ n < 10^7 ∧ k * 10^6 = n ∧ k < 1000 :=
by sorry

end NUMINAMATH_CALUDE_approximation_place_l1916_191636


namespace NUMINAMATH_CALUDE_cube_equation_solution_l1916_191640

theorem cube_equation_solution :
  ∃! x : ℝ, (12 - x)^3 = x^3 ∧ x = 12 := by sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l1916_191640


namespace NUMINAMATH_CALUDE_toy_production_on_time_l1916_191659

/-- Proves that the toy production can be completed on time --/
theorem toy_production_on_time (total_toys : ℕ) (first_three_days_avg : ℕ) (remaining_days_avg : ℕ) 
  (available_days : ℕ) (h1 : total_toys = 3000) (h2 : first_three_days_avg = 250) 
  (h3 : remaining_days_avg = 375) (h4 : available_days = 11) : 
  (3 + ((total_toys - 3 * first_three_days_avg) / remaining_days_avg : ℕ)) ≤ available_days := by
  sorry

#check toy_production_on_time

end NUMINAMATH_CALUDE_toy_production_on_time_l1916_191659


namespace NUMINAMATH_CALUDE_unique_obtainable_pair_l1916_191630

-- Define the calculator operations
def calc_op1 (p : ℕ × ℕ) : ℕ × ℕ := (p.1 + p.2, p.1)
def calc_op2 (p : ℕ × ℕ) : ℕ × ℕ := (2 * p.1 + p.2 + 1, p.1 + p.2 + 1)

-- Define a predicate for pairs obtainable by the calculator
inductive Obtainable : ℕ × ℕ → Prop where
  | initial : Obtainable (1, 1)
  | op1 {p : ℕ × ℕ} : Obtainable p → Obtainable (calc_op1 p)
  | op2 {p : ℕ × ℕ} : Obtainable p → Obtainable (calc_op2 p)

-- State the theorem
theorem unique_obtainable_pair :
  ∀ n : ℕ, ∃! k : ℕ, Obtainable (n, k) :=
sorry

end NUMINAMATH_CALUDE_unique_obtainable_pair_l1916_191630


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l1916_191609

/-- The number of diagonals in a convex n-gon --/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon is a polygon with 7 sides --/
def is_heptagon (n : ℕ) : Prop := n = 7

theorem heptagon_diagonals :
  ∀ n : ℕ, is_heptagon n → num_diagonals n = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l1916_191609


namespace NUMINAMATH_CALUDE_cubic_expression_value_l1916_191631

theorem cubic_expression_value : 7^3 - 4 * 7^2 + 4 * 7 - 2 = 173 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l1916_191631


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1916_191687

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1916_191687


namespace NUMINAMATH_CALUDE_stamp_cost_theorem_l1916_191678

theorem stamp_cost_theorem (total_stamps : ℕ) (high_value_stamps : ℕ) (high_value : ℚ) (low_value : ℚ) :
  total_stamps = 20 →
  high_value_stamps = 18 →
  high_value = 37 / 100 →
  low_value = 20 / 100 →
  (high_value_stamps * high_value + (total_stamps - high_value_stamps) * low_value) = 706 / 100 := by
  sorry

end NUMINAMATH_CALUDE_stamp_cost_theorem_l1916_191678


namespace NUMINAMATH_CALUDE_max_candies_karlson_candy_theorem_l1916_191635

/-- Represents the process of combining numbers and counting products -/
def combine_numbers (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

/-- The maximum number of candies Karlson can eat -/
theorem max_candies : combine_numbers 26 = 325 := by
  sorry

/-- Proves that the maximum number of candies is achieved -/
theorem karlson_candy_theorem (initial_count : ℕ) (operation_count : ℕ) 
  (h1 : initial_count = 26) (h2 : operation_count = 25) : 
  combine_numbers initial_count = 325 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_karlson_candy_theorem_l1916_191635


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_A_subset_B_iff_l1916_191626

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem 1
theorem intersection_A_complement_B (a : ℝ) (h : a = -2) :
  A a ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2
theorem A_subset_B_iff (a : ℝ) :
  A a ∩ B = A a ↔ a < -4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_A_subset_B_iff_l1916_191626


namespace NUMINAMATH_CALUDE_dark_tile_fraction_for_given_floor_l1916_191667

/-- Represents a tiled floor with a repeating pattern -/
structure TiledFloor :=
  (pattern_size : Nat)
  (dark_tiles_in_quarter : Nat)

/-- Calculates the fraction of dark tiles in the entire floor -/
def dark_tile_fraction (floor : TiledFloor) : Rat :=
  sorry

/-- Theorem stating the fraction of dark tiles in the given floor configuration -/
theorem dark_tile_fraction_for_given_floor :
  let floor := TiledFloor.mk 8 10
  dark_tile_fraction floor = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_for_given_floor_l1916_191667


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l1916_191608

-- Define complex numbers a and b
variable (a b : ℂ)

-- Define real number t
variable (t : ℝ)

-- State the theorem
theorem complex_product_magnitude 
  (h1 : Complex.abs a = 2)
  (h2 : Complex.abs b = 5)
  (h3 : a * b = t - 3 * Complex.I)
  (h4 : t > 0) :
  t = Real.sqrt 91 := by
sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l1916_191608


namespace NUMINAMATH_CALUDE_weekly_calories_burned_l1916_191657

/-- Represents the duration of activities in minutes for a spinning class -/
structure ClassDuration :=
  (cycling : Nat)
  (strength : Nat)
  (stretching : Nat)

/-- Represents the calorie burn rates per minute for each activity -/
structure CalorieBurnRates :=
  (cycling : Nat)
  (strength : Nat)
  (stretching : Nat)

def monday_class : ClassDuration := ⟨40, 20, 10⟩
def wednesday_class : ClassDuration := ⟨50, 25, 5⟩
def friday_class : ClassDuration := ⟨30, 30, 15⟩

def burn_rates : CalorieBurnRates := ⟨12, 8, 3⟩

def total_calories_burned (classes : List ClassDuration) (rates : CalorieBurnRates) : Nat :=
  let total_cycling := classes.foldl (fun acc c => acc + c.cycling) 0
  let total_strength := classes.foldl (fun acc c => acc + c.strength) 0
  let total_stretching := classes.foldl (fun acc c => acc + c.stretching) 0
  total_cycling * rates.cycling + total_strength * rates.strength + total_stretching * rates.stretching

theorem weekly_calories_burned :
  total_calories_burned [monday_class, wednesday_class, friday_class] burn_rates = 2130 := by
  sorry

end NUMINAMATH_CALUDE_weekly_calories_burned_l1916_191657


namespace NUMINAMATH_CALUDE_quadratic_roots_l1916_191674

theorem quadratic_roots (d : ℚ) : 
  (∀ x : ℚ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt (2*d))/2 ∨ x = (-7 - Real.sqrt (2*d))/2) → 
  d = 49/6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1916_191674


namespace NUMINAMATH_CALUDE_gcd_12m_18n_min_l1916_191693

/-- For positive integers m and n with gcd(m, n) = 10, the smallest possible value of gcd(12m, 18n) is 60 -/
theorem gcd_12m_18n_min (m n : ℕ+) (h : Nat.gcd m.val n.val = 10) :
  ∃ (k : ℕ+), (∀ (a b : ℕ+), Nat.gcd (12 * a.val) (18 * b.val) ≥ k.val) ∧
              (Nat.gcd (12 * m.val) (18 * n.val) = k.val) ∧
              k = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12m_18n_min_l1916_191693


namespace NUMINAMATH_CALUDE_proposition_truth_values_l1916_191619

open Real

theorem proposition_truth_values :
  ∃ (p q : Prop),
  (∀ x, 0 < x → x < π / 2 → (p ↔ sin x > x)) ∧
  (∀ x, 0 < x → x < π / 2 → (q ↔ tan x > x)) ∧
  (¬(p ∧ q)) ∧
  (p ∨ q) ∧
  (¬(p ∨ ¬q)) ∧
  ((¬p) ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l1916_191619


namespace NUMINAMATH_CALUDE_emilys_speed_l1916_191615

/-- Given a distance of 10 miles traveled in 2 hours, prove the speed is 5 miles per hour -/
theorem emilys_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 10)
  (h2 : time = 2)
  (h3 : speed = distance / time) :
  speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_emilys_speed_l1916_191615


namespace NUMINAMATH_CALUDE_classroom_students_l1916_191634

theorem classroom_students (n : ℕ) : 
  n < 50 → n % 8 = 5 → n % 6 = 3 → (n = 21 ∨ n = 45) := by
  sorry

end NUMINAMATH_CALUDE_classroom_students_l1916_191634


namespace NUMINAMATH_CALUDE_pet_food_discount_l1916_191600

theorem pet_food_discount (msrp : ℝ) (regular_discount : ℝ) (final_price : ℝ) (additional_discount : ℝ) : 
  msrp = 40 →
  regular_discount = 0.3 →
  final_price = 22.4 →
  additional_discount = (msrp * (1 - regular_discount) - final_price) / (msrp * (1 - regular_discount)) →
  additional_discount = 0.2 := by
sorry

end NUMINAMATH_CALUDE_pet_food_discount_l1916_191600


namespace NUMINAMATH_CALUDE_hillarys_remaining_money_l1916_191628

/-- Calculates the amount Hillary is left with after selling crafts and accounting for all costs and transactions. -/
theorem hillarys_remaining_money
  (base_price : ℝ)
  (cost_per_craft : ℝ)
  (crafts_sold : ℕ)
  (extra_money : ℝ)
  (tax_rate : ℝ)
  (deposit_amount : ℝ)
  (h1 : base_price = 12)
  (h2 : cost_per_craft = 4)
  (h3 : crafts_sold = 3)
  (h4 : extra_money = 7)
  (h5 : tax_rate = 0.1)
  (h6 : deposit_amount = 26)
  : ∃ (remaining : ℝ), remaining = 1.9 ∧ remaining ≥ 0 := by
  sorry

#check hillarys_remaining_money

end NUMINAMATH_CALUDE_hillarys_remaining_money_l1916_191628


namespace NUMINAMATH_CALUDE_combined_land_area_l1916_191652

/-- The combined area of two rectangular tracts of land -/
theorem combined_land_area (length1 width1 length2 width2 : ℕ) 
  (h1 : length1 = 300) 
  (h2 : width1 = 500) 
  (h3 : length2 = 250) 
  (h4 : width2 = 630) : 
  length1 * width1 + length2 * width2 = 307500 := by
  sorry

#check combined_land_area

end NUMINAMATH_CALUDE_combined_land_area_l1916_191652


namespace NUMINAMATH_CALUDE_function_composition_condition_l1916_191617

theorem function_composition_condition (a b : ℤ) :
  (∃ (f g : ℤ → ℤ), ∀ x, f (g x) = x + a ∧ g (f x) = x + b) ↔ |a| = |b| :=
by sorry

end NUMINAMATH_CALUDE_function_composition_condition_l1916_191617


namespace NUMINAMATH_CALUDE_decreasing_implies_a_le_10_l1916_191681

/-- A quadratic function f(x) = x^2 + 2(a-5)x - 6 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-5)*x - 6

/-- The function f is decreasing on the interval (-∞, -5] -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y → y ≤ -5 → f a x ≥ f a y

theorem decreasing_implies_a_le_10 (a : ℝ) :
  is_decreasing_on_interval a → a ≤ 10 := by sorry

end NUMINAMATH_CALUDE_decreasing_implies_a_le_10_l1916_191681


namespace NUMINAMATH_CALUDE_set_b_forms_triangle_l1916_191672

/-- Triangle inequality theorem: A set of three line segments can form a triangle if and only if
    the sum of the lengths of any two sides is greater than the length of the third side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that line segments of lengths 8, 6, and 4 can form a triangle. -/
theorem set_b_forms_triangle : can_form_triangle 8 6 4 := by
  sorry

end NUMINAMATH_CALUDE_set_b_forms_triangle_l1916_191672


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_2006_l1916_191696

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_2006 :
  units_digit (factorial_sum 2006) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_2006_l1916_191696


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1916_191663

theorem complex_equation_solution (z : ℂ) :
  z / (z - Complex.I) = Complex.I → z = (1 : ℂ) / 2 + Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1916_191663


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_88_875_l1916_191670

/-- Represents the grid and shapes configuration --/
structure GridConfig where
  gridSize : ℕ
  squareSide : ℝ
  smallCircleDiameter : ℝ
  largeCircleDiameter : ℝ
  hexagonSide : ℝ
  smallCircleCount : ℕ

/-- Calculates the coefficients A, B, and C for the shaded area expression --/
def calculateCoefficients (config : GridConfig) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem stating that the sum of coefficients equals 88.875 for the given configuration --/
theorem sum_of_coefficients_equals_88_875 : 
  let config : GridConfig := {
    gridSize := 6,
    squareSide := 1.5,
    smallCircleDiameter := 1.5,
    largeCircleDiameter := 3,
    hexagonSide := 1.5,
    smallCircleCount := 4
  }
  let (A, B, C) := calculateCoefficients config
  A + B + C = 88.875 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_88_875_l1916_191670


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1916_191624

theorem polynomial_divisibility (x₀ a₁ a₂ a₃ a₄ : ℝ) 
  (h1 : x₀^4 + a₁*x₀^3 + a₂*x₀^2 + a₃*x₀ + a₄ = 0)
  (h2 : 4*x₀^3 + 3*a₁*x₀^2 + 2*a₂*x₀ + a₃ = 0) :
  ∃ g : ℝ → ℝ, ∀ x : ℝ, 
    x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄ = (x - x₀)^2 * g x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1916_191624


namespace NUMINAMATH_CALUDE_min_value_of_fraction_lower_bound_achievable_l1916_191603

theorem min_value_of_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (a + b) / (a * b * c) ≥ 16 / 9 :=
sorry

theorem lower_bound_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 ∧ (a + b) / (a * b * c) = 16 / 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_lower_bound_achievable_l1916_191603


namespace NUMINAMATH_CALUDE_congruent_triangles_corresponding_angles_l1916_191653

-- Define a triangle
def Triangle := ℝ × ℝ × ℝ

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define the property of corresponding angles being congruent
def corresponding_angles_congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem statement
theorem congruent_triangles_corresponding_angles 
  (t1 t2 : Triangle) : congruent t1 t2 → corresponding_angles_congruent t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_congruent_triangles_corresponding_angles_l1916_191653


namespace NUMINAMATH_CALUDE_maria_reading_capacity_l1916_191658

/-- The number of books Maria can read given her reading speed, book length, and available time -/
def books_read (reading_speed : ℕ) (pages_per_book : ℕ) (available_hours : ℕ) : ℕ :=
  (reading_speed * available_hours) / pages_per_book

/-- Theorem: Maria can read 3 books of 360 pages each in 9 hours at a speed of 120 pages per hour -/
theorem maria_reading_capacity :
  books_read 120 360 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_maria_reading_capacity_l1916_191658


namespace NUMINAMATH_CALUDE_locus_and_fixed_points_l1916_191637

-- Define the points and lines
def F : ℝ × ℝ := (1, 0)
def H : ℝ × ℝ := (1, 2)
def l : Set (ℝ × ℝ) := {p | p.1 = -1}

-- Define the locus C
def C : Set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

-- Define a function to represent a line passing through F and not perpendicular to x-axis
def line_through_F (m : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m * (p.1 - 1)}

-- Define the circle with diameter MN
def circle_MN (m : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.1 - 3 + p.2^2 + (4/m)*p.2 = 0}

-- State the theorem
theorem locus_and_fixed_points :
  ∀ (m : ℝ), m ≠ 0 →
  (∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ A ∈ line_through_F m ∧ B ∈ line_through_F m) →
  ((-3, 0) ∈ circle_MN m ∧ (1, 0) ∈ circle_MN m) :=
sorry

end NUMINAMATH_CALUDE_locus_and_fixed_points_l1916_191637


namespace NUMINAMATH_CALUDE_local_minimum_condition_l1916_191610

/-- The function f(x) = x(x-a)² has a local minimum at x=2 if and only if a = 2 -/
theorem local_minimum_condition (a : ℝ) :
  (∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → x * (x - a)^2 ≥ 2 * (2 - a)^2) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_condition_l1916_191610


namespace NUMINAMATH_CALUDE_triangle_side_length_l1916_191601

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 2 →
  b = Real.sqrt 3 - 1 →
  C = π / 6 →
  c^2 = 5 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1916_191601


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1916_191650

/-- A regular polygon with an exterior angle of 18° has 20 sides -/
theorem regular_polygon_sides (n : ℕ) (ext_angle : ℝ) : 
  ext_angle = 18 → n * ext_angle = 360 → n = 20 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1916_191650


namespace NUMINAMATH_CALUDE_quadrilateral_angle_measure_l1916_191682

theorem quadrilateral_angle_measure (A B C D : ℝ) : 
  A = 105 → B = C → A + B + C + D = 360 → D = 180 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_measure_l1916_191682


namespace NUMINAMATH_CALUDE_quadruplet_equation_equivalence_l1916_191668

theorem quadruplet_equation_equivalence (x y z w : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  (1 + 1/x + 2*(x+1)/(x*y) + 3*(x+1)*(y+2)/(x*y*z) + 4*(x+1)*(y+2)*(z+3)/(x*y*z*w) = 0) ↔
  ((x+1)*(y+2)*(z+3)*(w+4) = 0) := by
sorry

end NUMINAMATH_CALUDE_quadruplet_equation_equivalence_l1916_191668


namespace NUMINAMATH_CALUDE_qiqi_problem_solving_l1916_191660

/-- Represents the number of problems completed in a given time -/
structure ProblemRate where
  problems : ℕ
  minutes : ℕ

/-- Calculates the number of problems that can be completed in a given time,
    given a known problem rate -/
def calculateProblems (rate : ProblemRate) (time : ℕ) : ℕ :=
  (rate.problems * time) / rate.minutes

theorem qiqi_problem_solving :
  let initialRate : ProblemRate := ⟨15, 5⟩
  calculateProblems initialRate 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_qiqi_problem_solving_l1916_191660


namespace NUMINAMATH_CALUDE_total_balloons_l1916_191618

def fred_balloons : ℕ := 5
def sam_balloons : ℕ := 6
def mary_balloons : ℕ := 7

theorem total_balloons : fred_balloons + sam_balloons + mary_balloons = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l1916_191618


namespace NUMINAMATH_CALUDE_harkamal_payment_l1916_191612

/-- The total amount Harkamal paid to the shopkeeper -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 1145 to the shopkeeper -/
theorem harkamal_payment :
  total_amount 8 70 9 65 = 1145 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_payment_l1916_191612


namespace NUMINAMATH_CALUDE_min_value_theorem_l1916_191695

theorem min_value_theorem (x : ℝ) (h : x > 9) :
  (x^2 + 81) / (x - 9) ≥ 27 ∧ ∃ y > 9, (y^2 + 81) / (y - 9) = 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1916_191695


namespace NUMINAMATH_CALUDE_midsize_rental_cost_l1916_191665

/-- Represents the types of rental cars --/
inductive CarType
| Economy
| MidSize
| Luxury

/-- Represents the rental rates for a car type --/
structure RentalRates where
  dailyRate : Nat
  weeklyRate : Nat
  discountedRateUpTo10Days : Nat
  discountedRateAfter10Days : Nat

/-- Calculate the rental cost for a given number of days --/
def calculateRentalCost (rates : RentalRates) (days : Nat) : Nat :=
  if days ≤ 7 then
    min (days * rates.dailyRate) rates.weeklyRate
  else
    rates.weeklyRate + 
    (min (days - 7) 3 * rates.discountedRateUpTo10Days) +
    (max (days - 10) 0 * rates.discountedRateAfter10Days)

/-- Apply a percentage discount to a given amount --/
def applyDiscount (amount : Nat) (discountPercent : Nat) : Nat :=
  amount - (amount * discountPercent / 100)

/-- Theorem: The cost of renting a mid-size car for 13 days with a 10% discount is $306 --/
theorem midsize_rental_cost : 
  let midSizeRates : RentalRates := {
    dailyRate := 30,
    weeklyRate := 190,
    discountedRateUpTo10Days := 25,
    discountedRateAfter10Days := 20
  }
  let rentalDays := 13
  let discountPercent := 10
  applyDiscount (calculateRentalCost midSizeRates rentalDays) discountPercent = 306 := by
  sorry

end NUMINAMATH_CALUDE_midsize_rental_cost_l1916_191665


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1916_191688

theorem quadratic_inequality_solution (x : ℝ) :
  -3 * x^2 + 5 * x + 4 < 0 ↔ -4/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1916_191688


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1916_191642

/-- Proves that a rectangle with perimeter 34 and length 5 more than width has width 6 and length 11 -/
theorem rectangle_dimensions :
  ∀ (w l : ℕ), 
    (2 * w + 2 * l = 34) →  -- Perimeter is 34
    (l = w + 5) →           -- Length is 5 more than width
    (w = 6 ∧ l = 11) :=     -- Width is 6 and length is 11
by
  sorry

#check rectangle_dimensions

end NUMINAMATH_CALUDE_rectangle_dimensions_l1916_191642


namespace NUMINAMATH_CALUDE_largest_area_error_l1916_191666

theorem largest_area_error (actual_side : ℝ) (max_error_percent : ℝ) :
  actual_side = 30 →
  max_error_percent = 20 →
  let max_measured_side := actual_side * (1 + max_error_percent / 100)
  let actual_area := actual_side ^ 2
  let max_measured_area := max_measured_side ^ 2
  let max_percent_error := (max_measured_area - actual_area) / actual_area * 100
  max_percent_error = 44 := by
sorry

end NUMINAMATH_CALUDE_largest_area_error_l1916_191666


namespace NUMINAMATH_CALUDE_infinitely_many_special_even_numbers_l1916_191648

theorem infinitely_many_special_even_numbers :
  ∃ (n : ℕ → ℕ), 
    (∀ k, Even (n k)) ∧ 
    (∀ k, n k < n (k + 1)) ∧
    (∀ k, (n k) ∣ (2^(n k) + 2)) ∧
    (∀ k, (n k - 1) ∣ (2^(n k) + 1)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_special_even_numbers_l1916_191648


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1916_191677

theorem possible_values_of_a (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 5) + 3 = (x + b) * (x + c)) →
  (a = 1 ∨ a = 9) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1916_191677


namespace NUMINAMATH_CALUDE_sugar_calculation_l1916_191680

theorem sugar_calculation (num_packs : ℕ) (pack_weight : ℕ) (leftover : ℕ) :
  num_packs = 30 →
  pack_weight = 350 →
  leftover = 50 →
  num_packs * pack_weight + leftover = 10550 := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l1916_191680


namespace NUMINAMATH_CALUDE_nora_paid_90_dimes_l1916_191669

/-- The number of dimes in one dollar -/
def dimes_per_dollar : ℕ := 10

/-- The cost of the watch in dollars -/
def watch_cost : ℕ := 9

/-- The number of dimes Nora paid for the watch -/
def dimes_paid : ℕ := watch_cost * dimes_per_dollar

theorem nora_paid_90_dimes : dimes_paid = 90 := by
  sorry

end NUMINAMATH_CALUDE_nora_paid_90_dimes_l1916_191669


namespace NUMINAMATH_CALUDE_remainder_theorem_l1916_191646

theorem remainder_theorem (n : ℤ) (h : n % 7 = 3) : (5 * n - 11) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1916_191646


namespace NUMINAMATH_CALUDE_new_average_score_is_correct_l1916_191638

/-- Represents the grace mark criteria for different score ranges -/
inductive GraceMarkCriteria where
  | below30 : GraceMarkCriteria
  | between30and40 : GraceMarkCriteria
  | above40 : GraceMarkCriteria

/-- Returns the grace marks for a given criteria -/
def graceMarks (c : GraceMarkCriteria) : ℕ :=
  match c with
  | GraceMarkCriteria.below30 => 5
  | GraceMarkCriteria.between30and40 => 3
  | GraceMarkCriteria.above40 => 1

/-- Calculates the new average score after applying grace marks -/
def newAverageScore (
  classSize : ℕ
  ) (initialAverage : ℚ
  ) (studentsPerRange : ℕ
  ) : ℚ :=
  let initialTotal := classSize * initialAverage
  let totalGraceMarks := 
    studentsPerRange * (graceMarks GraceMarkCriteria.below30 + 
                        graceMarks GraceMarkCriteria.between30and40 + 
                        graceMarks GraceMarkCriteria.above40)
  (initialTotal + totalGraceMarks) / classSize

/-- Theorem stating that the new average score is approximately 39.57 -/
theorem new_average_score_is_correct :
  let classSize := 35
  let initialAverage := 37
  let studentsPerRange := 10
  abs (newAverageScore classSize initialAverage studentsPerRange - 39.57) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_new_average_score_is_correct_l1916_191638


namespace NUMINAMATH_CALUDE_quadratic_function_ratio_l1916_191647

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  max_at_neg_two : ∀ x : ℝ, a * x^2 + b * x + c ≤ a^2
  max_value : a * (-2)^2 + b * (-2) + c = a^2
  passes_through_point : a * (-1)^2 + b * (-1) + c = 6

/-- Theorem stating that (a + c) / b = 1/2 for the given quadratic function -/
theorem quadratic_function_ratio (f : QuadraticFunction) : (f.a + f.c) / f.b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_ratio_l1916_191647


namespace NUMINAMATH_CALUDE_no_integer_points_between_l1916_191643

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- The line passing through points A(2, 3) and B(50, 500) -/
def line (p : IntPoint) : Prop :=
  (p.y - 3) * 48 = 497 * (p.x - 2)

/-- A point is strictly between A and B if its x-coordinate is between 2 and 50 exclusively -/
def strictly_between (p : IntPoint) : Prop :=
  2 < p.x ∧ p.x < 50

theorem no_integer_points_between : 
  ¬ ∃ p : IntPoint, line p ∧ strictly_between p :=
sorry

end NUMINAMATH_CALUDE_no_integer_points_between_l1916_191643


namespace NUMINAMATH_CALUDE_y₁_gt_y₂_l1916_191675

/-- A linear function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (1, f 1)

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (3, f 3)

/-- y₁ coordinate of point A -/
def y₁ : ℝ := (A.2)

/-- y₂ coordinate of point B -/
def y₂ : ℝ := (B.2)

theorem y₁_gt_y₂ : y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_gt_y₂_l1916_191675


namespace NUMINAMATH_CALUDE_minimize_expression_l1916_191620

theorem minimize_expression (a b : ℝ) (h1 : a + b = -2) (h2 : b < 0) :
  ∃ (min_a : ℝ), min_a = 2 ∧
  ∀ (x : ℝ), x + b = -2 → (1 / (2 * |x|) - |x| / b) ≥ (1 / (2 * |min_a|) - |min_a| / b) :=
sorry

end NUMINAMATH_CALUDE_minimize_expression_l1916_191620


namespace NUMINAMATH_CALUDE_square_ratio_sum_l1916_191697

theorem square_ratio_sum (area_ratio : ℚ) (a b c : ℕ) : 
  area_ratio = 75 / 128 →
  (∃ (side_ratio : ℝ), side_ratio = Real.sqrt (area_ratio) ∧ 
    side_ratio = a * Real.sqrt b / c) →
  a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_square_ratio_sum_l1916_191697


namespace NUMINAMATH_CALUDE_circle_center_transformation_l1916_191698

/-- Reflect a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Translate a point vertically -/
def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

/-- The transformation described in the problem -/
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_up (reflect_y p) 12

theorem circle_center_transformation :
  transform (3, -4) = (-3, 8) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l1916_191698


namespace NUMINAMATH_CALUDE_no_solution_exists_l1916_191602

theorem no_solution_exists : ¬∃ (a b c d : ℤ),
  (a * b * c * d - a = 1961) ∧
  (a * b * c * d - b = 961) ∧
  (a * b * c * d - c = 61) ∧
  (a * b * c * d - d = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1916_191602


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1916_191623

theorem sin_cos_identity : Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (200 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1916_191623


namespace NUMINAMATH_CALUDE_slope_of_line_l_l1916_191605

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define the line l passing through M with slope m
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y - M.2 = m * (x - M.1)

-- Define the intersection points A and B
def intersection_points (m : ℝ) :=
  ∃ (xa ya xb yb : ℝ),
    ellipse xa ya ∧ ellipse xb yb ∧
    line_l m xa ya ∧ line_l m xb yb ∧
    (xa, ya) ≠ (xb, yb)

-- Define M as the trisection point of AB
def M_is_trisection (m : ℝ) :=
  ∃ (xa ya xb yb : ℝ),
    ellipse xa ya ∧ ellipse xb yb ∧
    line_l m xa ya ∧ line_l m xb yb ∧
    2 * M.1 = xa + xb ∧ 2 * M.2 = ya + yb

-- The main theorem
theorem slope_of_line_l :
  ∃ (m : ℝ), intersection_points m ∧ M_is_trisection m ∧
  (m = (-4 + Real.sqrt 7) / 6 ∨ m = (-4 - Real.sqrt 7) / 6) :=
sorry

end NUMINAMATH_CALUDE_slope_of_line_l_l1916_191605


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1916_191685

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n, S n = (n : ℝ) * (a 0 + a (n-1)) / 2

/-- Theorem: If S_n / S_2n = (n+1) / (4n+2) for an arithmetic sequence,
    then a_3 / a_5 = 3/5 -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : ∀ n, seq.S n / seq.S (2*n) = (n + 1 : ℝ) / (4*n + 2)) : 
  seq.a 3 / seq.a 5 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1916_191685


namespace NUMINAMATH_CALUDE_equation_solution_l1916_191655

theorem equation_solution : ∃! x : ℝ, (x^2 - x - 2) / (x + 2) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1916_191655


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1916_191606

theorem regular_polygon_sides (n : ℕ) (h : n > 0) : 
  (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1916_191606


namespace NUMINAMATH_CALUDE_kristine_has_more_cd_difference_l1916_191692

/-- The number of CDs Dawn has -/
def dawn_cds : ℕ := 10

/-- The total number of CDs Kristine and Dawn have together -/
def total_cds : ℕ := 27

/-- Kristine's CDs -/
def kristine_cds : ℕ := total_cds - dawn_cds

/-- The statement that Kristine has more CDs than Dawn -/
theorem kristine_has_more : kristine_cds > dawn_cds := by sorry

/-- The main theorem: Kristine has 7 more CDs than Dawn -/
theorem cd_difference : kristine_cds - dawn_cds = 7 := by sorry

end NUMINAMATH_CALUDE_kristine_has_more_cd_difference_l1916_191692


namespace NUMINAMATH_CALUDE_impossibility_of_filling_l1916_191639

/-- A brick is made of four unit cubes: one unit cube with three unit cubes
    attached to three of its faces, all sharing a common vertex. -/
structure Brick :=
  (cubes : Fin 4 → Unit)

/-- A rectangular parallelepiped with dimensions 11 × 12 × 13 -/
def Parallelepiped := Fin 11 × Fin 12 × Fin 13

/-- A function that represents filling the parallelepiped with bricks -/
def FillParallelepiped := Parallelepiped → Brick

/-- Theorem stating that it's impossible to fill the 11 × 12 × 13 parallelepiped with the given bricks -/
theorem impossibility_of_filling :
  ¬ ∃ (f : FillParallelepiped), True :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_filling_l1916_191639


namespace NUMINAMATH_CALUDE_coefficient_of_inverse_x_l1916_191673

theorem coefficient_of_inverse_x (x : ℝ) : 
  (∃ c : ℝ, (x / 2 - 2 / x)^5 = c / x + (terms_without_inverse_x : ℝ)) → 
  (∃ c : ℝ, (x / 2 - 2 / x)^5 = -20 / x + (terms_without_inverse_x : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_inverse_x_l1916_191673


namespace NUMINAMATH_CALUDE_negation_equivalence_l1916_191649

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m < 0) ↔ (∀ x : ℤ, x^2 + 2*x + m ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1916_191649


namespace NUMINAMATH_CALUDE_trapezoid_area_trapezoid_area_proof_l1916_191645

/-- The area of a trapezoid bounded by y = x + 1, y = 12, y = 7, and the y-axis -/
theorem trapezoid_area : ℝ :=
  let line1 : ℝ → ℝ := λ x ↦ x + 1
  let line2 : ℝ → ℝ := λ _ ↦ 12
  let line3 : ℝ → ℝ := λ _ ↦ 7
  let y_axis : ℝ → ℝ := λ _ ↦ 0
  42.5
  
#check trapezoid_area

/-- Proof that the area of the trapezoid is 42.5 square units -/
theorem trapezoid_area_proof : trapezoid_area = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_trapezoid_area_proof_l1916_191645


namespace NUMINAMATH_CALUDE_students_absent_eq_three_l1916_191689

/-- The number of cupcakes in a dozen -/
def dozen : ℕ := 12

/-- The number of cupcakes Dani brings -/
def cupcakes_brought : ℕ := (2 * dozen) + (dozen / 2)

/-- The total number of people in the class (including Dani) -/
def total_people : ℕ := 29

/-- The number of cupcakes left after distribution -/
def cupcakes_left : ℕ := 4

/-- The number of students who called in sick -/
def students_absent : ℕ := total_people - (cupcakes_brought - cupcakes_left)

theorem students_absent_eq_three : students_absent = 3 := by
  sorry

end NUMINAMATH_CALUDE_students_absent_eq_three_l1916_191689


namespace NUMINAMATH_CALUDE_train_length_problem_l1916_191644

/-- Given a platform length, time to pass, and train speed, calculates the length of the train -/
def train_length (platform_length time_to_pass train_speed : ℝ) : ℝ :=
  train_speed * time_to_pass - platform_length

/-- Theorem stating that under the given conditions, the train length is 50 meters -/
theorem train_length_problem :
  let platform_length : ℝ := 100
  let time_to_pass : ℝ := 10
  let train_speed : ℝ := 15
  train_length platform_length time_to_pass train_speed = 50 := by
sorry

#eval train_length 100 10 15

end NUMINAMATH_CALUDE_train_length_problem_l1916_191644


namespace NUMINAMATH_CALUDE_log_16_2_l1916_191616

theorem log_16_2 : Real.log 2 / Real.log 16 = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_log_16_2_l1916_191616


namespace NUMINAMATH_CALUDE_intersection_M_N_l1916_191604

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 2 < 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1916_191604


namespace NUMINAMATH_CALUDE_soup_feeding_theorem_l1916_191656

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Represents the soup feeding scenario -/
structure SoupScenario where
  can_capacity : SoupCan
  total_cans : ℕ
  children_fed : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remaining_adults_fed (scenario : SoupScenario) : ℕ :=
  let cans_for_children := scenario.children_fed / scenario.can_capacity.children
  let remaining_cans := scenario.total_cans - cans_for_children
  remaining_cans * scenario.can_capacity.adults

/-- Theorem: Given 8 cans of soup, where each can feeds 4 adults or 6 children,
    after feeding 24 children, the remaining soup can feed 16 adults -/
theorem soup_feeding_theorem (scenario : SoupScenario)
  (h1 : scenario.can_capacity = ⟨4, 6⟩)
  (h2 : scenario.total_cans = 8)
  (h3 : scenario.children_fed = 24) :
  remaining_adults_fed scenario = 16 := by
  sorry

end NUMINAMATH_CALUDE_soup_feeding_theorem_l1916_191656


namespace NUMINAMATH_CALUDE_identity_equals_one_l1916_191611

theorem identity_equals_one (a b c x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (x - b) * (x - c) / ((a - b) * (a - c)) +
  (x - c) * (x - a) / ((b - c) * (b - a)) +
  (x - a) * (x - b) / ((c - a) * (c - b)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_identity_equals_one_l1916_191611


namespace NUMINAMATH_CALUDE_defective_product_probability_l1916_191679

theorem defective_product_probability 
  (total_products : ℕ) 
  (genuine_products : ℕ) 
  (defective_products : ℕ) 
  (h1 : total_products = genuine_products + defective_products)
  (h2 : genuine_products = 16)
  (h3 : defective_products = 4) :
  let prob_second_defective : ℚ := defective_products - 1 / (total_products - 1)
  prob_second_defective = 3 / 19 := by
sorry

end NUMINAMATH_CALUDE_defective_product_probability_l1916_191679


namespace NUMINAMATH_CALUDE_always_integer_l1916_191664

theorem always_integer (m : ℕ) : ∃ k : ℤ, (m : ℚ) / 3 + (m : ℚ)^2 / 2 + (m : ℚ)^3 / 6 = k := by
  sorry

end NUMINAMATH_CALUDE_always_integer_l1916_191664


namespace NUMINAMATH_CALUDE_regular_nonagon_side_length_l1916_191671

/-- A regular nonagon with perimeter 171 centimeters has sides of length 19 centimeters -/
theorem regular_nonagon_side_length : 
  ∀ (perimeter side_length : ℝ),
    perimeter = 171 →
    side_length * 9 = perimeter →
    side_length = 19 :=
by sorry

end NUMINAMATH_CALUDE_regular_nonagon_side_length_l1916_191671


namespace NUMINAMATH_CALUDE_chosen_numbers_divisibility_l1916_191627

theorem chosen_numbers_divisibility 
  (S : Finset ℕ) 
  (h_card : S.card = 250) 
  (h_bound : ∀ n ∈ S, n ≤ 501) :
  ∀ t : ℤ, ∃ a₁ a₂ a₃ a₄ : ℕ, 
    a₁ ∈ S ∧ a₂ ∈ S ∧ a₃ ∈ S ∧ a₄ ∈ S ∧ 
    23 ∣ (a₁ + a₂ + a₃ + a₄ - t) :=
by sorry

end NUMINAMATH_CALUDE_chosen_numbers_divisibility_l1916_191627


namespace NUMINAMATH_CALUDE_age_difference_proof_l1916_191625

-- Define the ages of Betty, Mary, and Albert
def betty_age : ℕ := 11
def albert_age (betty_age : ℕ) : ℕ := 4 * betty_age
def mary_age (albert_age : ℕ) : ℕ := albert_age / 2

-- Theorem statement
theorem age_difference_proof :
  albert_age betty_age - mary_age (albert_age betty_age) = 22 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1916_191625


namespace NUMINAMATH_CALUDE_welders_count_l1916_191662

/-- The number of welders initially working on the order -/
def initial_welders : ℕ := 72

/-- The number of days needed to complete the order with all welders -/
def initial_days : ℕ := 5

/-- The number of welders that leave after the first day -/
def leaving_welders : ℕ := 12

/-- The number of additional days needed to complete the order after some welders leave -/
def additional_days : ℕ := 6

/-- The theorem stating that the initial number of welders is 72 -/
theorem welders_count :
  initial_welders = 72 ∧
  (1 : ℚ) / (initial_days * initial_welders) = 
  (1 : ℚ) / (additional_days * (initial_welders - leaving_welders)) :=
by sorry

end NUMINAMATH_CALUDE_welders_count_l1916_191662


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1916_191699

theorem polynomial_division_remainder :
  ∃ (q r : Polynomial ℚ),
    3 * X^4 + 14 * X^3 - 50 * X^2 - 72 * X + 55 = (X^2 + 8 * X - 4) * q + r ∧
    r = 224 * X - 113 ∧
    r.degree < (X^2 + 8 * X - 4).degree :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1916_191699


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l1916_191614

theorem solve_cubic_equation (y : ℝ) :
  5 * y^(1/3) + 3 * (y / y^(2/3)) = 10 - y^(1/3) ↔ y = (10/9)^3 := by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l1916_191614


namespace NUMINAMATH_CALUDE_fold_lines_cover_outside_l1916_191691

/-- A circle with center O and radius R -/
structure Circle where
  O : ℝ × ℝ
  R : ℝ

/-- A point A inside the circle -/
structure InnerPoint (c : Circle) where
  A : ℝ × ℝ
  dist_OA : Real.sqrt ((A.1 - c.O.1)^2 + (A.2 - c.O.2)^2) < c.R

/-- A point on the circumference of the circle -/
def CircumferencePoint (c : Circle) : Type :=
  { p : ℝ × ℝ // Real.sqrt ((p.1 - c.O.1)^2 + (p.2 - c.O.2)^2) = c.R }

/-- The set of all points on a fold line -/
def FoldLine (c : Circle) (A : InnerPoint c) (A' : CircumferencePoint c) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • A.A + t • A'.val }

/-- The set of all points on all possible fold lines -/
def AllFoldLines (c : Circle) (A : InnerPoint c) : Set (ℝ × ℝ) :=
  ⋃ (A' : CircumferencePoint c), FoldLine c A A'

/-- The set of points outside and on the circle -/
def OutsideAndOnCircle (c : Circle) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | Real.sqrt ((p.1 - c.O.1)^2 + (p.2 - c.O.2)^2) ≥ c.R }

/-- The main theorem -/
theorem fold_lines_cover_outside (c : Circle) (A : InnerPoint c) :
  AllFoldLines c A = OutsideAndOnCircle c := by sorry

end NUMINAMATH_CALUDE_fold_lines_cover_outside_l1916_191691


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1916_191632

theorem inequality_system_solution (x : ℝ) :
  x + 3 ≥ 2 ∧ 2 * (x + 4) > 4 * x + 2 → -1 ≤ x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1916_191632


namespace NUMINAMATH_CALUDE_positive_X_value_l1916_191676

-- Define the # relation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- State the theorem
theorem positive_X_value (X : ℝ) (h1 : hash X 7 = 250) (h2 : X > 0) : X = Real.sqrt 201 := by
  sorry

end NUMINAMATH_CALUDE_positive_X_value_l1916_191676


namespace NUMINAMATH_CALUDE_abs_diff_segments_of_cyclic_quad_with_incircle_l1916_191683

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicQuadWithIncircle where
  -- Side lengths of the quadrilateral
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Conditions for a valid quadrilateral
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d
  -- Condition for cyclic quadrilateral (sum of opposite sides are equal)
  cyclic : a + c = b + d
  -- Additional condition for having an inscribed circle
  has_incircle : True

/-- Theorem stating the absolute difference between segments -/
theorem abs_diff_segments_of_cyclic_quad_with_incircle 
  (q : CyclicQuadWithIncircle) 
  (h1 : q.a = 80) 
  (h2 : q.b = 100) 
  (h3 : q.c = 140) 
  (h4 : q.d = 120) 
  (x y : ℝ) 
  (h5 : x + y = q.c) : 
  |x - y| = 166.36 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_segments_of_cyclic_quad_with_incircle_l1916_191683


namespace NUMINAMATH_CALUDE_proportional_segments_l1916_191684

/-- A set of four line segments (a, b, c, d) is proportional if a * d = b * c -/
def isProportional (a b c d : ℝ) : Prop := a * d = b * c

/-- The set of line segments (2, 4, 8, 16) is proportional -/
theorem proportional_segments : isProportional 2 4 8 16 := by
  sorry

end NUMINAMATH_CALUDE_proportional_segments_l1916_191684


namespace NUMINAMATH_CALUDE_initial_toys_count_l1916_191607

/-- 
Given that Emily sold some toys and has some left, this theorem proves
the initial number of toys she had.
-/
theorem initial_toys_count 
  (sold : ℕ) -- Number of toys sold
  (remaining : ℕ) -- Number of toys remaining
  (h1 : sold = 3) -- Condition: Emily sold 3 toys
  (h2 : remaining = 4) -- Condition: Emily now has 4 toys left
  : sold + remaining = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_toys_count_l1916_191607


namespace NUMINAMATH_CALUDE_complex_fraction_theorem_l1916_191694

theorem complex_fraction_theorem (x y : ℂ) 
  (h : (x + y) / (x - y) - (x - y) / (x + y) = 3) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = -2.871 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_theorem_l1916_191694


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1916_191629

theorem expand_and_simplify (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * ((7 / x^2) - 5 * x^3) = 3 / x^2 - 15 * x^3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1916_191629


namespace NUMINAMATH_CALUDE_books_read_by_tony_dean_breanna_l1916_191686

/-- The number of different books read by Tony, Dean, and Breanna -/
def totalDifferentBooks (tonyBooks deanBooks breannaBooks sharedTonyDean sharedAll : ℕ) : ℕ :=
  tonyBooks + deanBooks + breannaBooks - sharedTonyDean - sharedAll

/-- Theorem stating the total number of different books read -/
theorem books_read_by_tony_dean_breanna : 
  totalDifferentBooks 23 12 17 3 1 = 48 := by
  sorry

#eval totalDifferentBooks 23 12 17 3 1

end NUMINAMATH_CALUDE_books_read_by_tony_dean_breanna_l1916_191686


namespace NUMINAMATH_CALUDE_carrots_and_cauliflower_cost_l1916_191690

/-- The cost of a bunch of carrots and a cauliflower given specific pricing conditions -/
theorem carrots_and_cauliflower_cost :
  ∀ (p c f o : ℝ),
    p + c + f + o = 30 →  -- Total cost
    o = 3 * p →           -- Oranges cost thrice potatoes
    f = p + c →           -- Cauliflower costs sum of potatoes and carrots
    c + f = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_carrots_and_cauliflower_cost_l1916_191690


namespace NUMINAMATH_CALUDE_emilys_skirt_cost_l1916_191613

theorem emilys_skirt_cost (art_supplies_cost shoes_original_price total_spent : ℝ)
  (skirt_count : ℕ) (shoe_discount_rate : ℝ) :
  art_supplies_cost = 20 →
  skirt_count = 2 →
  shoes_original_price = 30 →
  shoe_discount_rate = 0.15 →
  total_spent = 50 →
  let shoes_discounted_price := shoes_original_price * (1 - shoe_discount_rate)
  let skirts_total_cost := total_spent - art_supplies_cost - shoes_discounted_price
  let skirt_cost := skirts_total_cost / skirt_count
  skirt_cost = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_emilys_skirt_cost_l1916_191613


namespace NUMINAMATH_CALUDE_sam_not_buying_book_probability_l1916_191651

theorem sam_not_buying_book_probability (p : ℚ) 
  (h : p = 5 / 8) : 1 - p = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sam_not_buying_book_probability_l1916_191651


namespace NUMINAMATH_CALUDE_triangular_structures_stability_l1916_191654

/-- A structure used in construction -/
inductive ConstructionStructure
| Bridge
| CableCarSupport
| Truss

/-- A geometric shape -/
inductive Shape
| Triangle
| Other

/-- Stability property of a shape -/
def isStable : Shape → Prop :=
  fun s => match s with
  | Shape.Triangle => true
  | Shape.Other => false

/-- The shape used in a construction structure -/
def shapeUsed : ConstructionStructure → Shape :=
  fun _ => Shape.Triangle

/-- Theorem stating that triangular structures are used in certain constructions due to their stability -/
theorem triangular_structures_stability :
  ∀ (c : ConstructionStructure), isStable (shapeUsed c) :=
by sorry

end NUMINAMATH_CALUDE_triangular_structures_stability_l1916_191654
