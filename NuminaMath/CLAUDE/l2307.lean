import Mathlib

namespace NUMINAMATH_CALUDE_repeated_root_condition_l2307_230796

/-- The equation has a repeated root if and only if m = -1 -/
theorem repeated_root_condition (m : ℝ) : 
  (∃ x : ℝ, (x - 6) / (x - 5) + 1 = m / (x - 5) ∧ 
   ∀ y : ℝ, y ≠ x → (y - 6) / (y - 5) + 1 ≠ m / (y - 5)) ↔ 
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_repeated_root_condition_l2307_230796


namespace NUMINAMATH_CALUDE_sum_and_product_zero_l2307_230722

theorem sum_and_product_zero (a b : ℝ) 
  (h1 : 2*a + 2*b + a*b = 1) 
  (h2 : a + b + 3*a*b = -2) : 
  a + b + a*b = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_and_product_zero_l2307_230722


namespace NUMINAMATH_CALUDE_unique_solution_linear_system_l2307_230763

theorem unique_solution_linear_system :
  ∃! (x y : ℝ), (2 * x - y = 1) ∧ (x + y = 2) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_linear_system_l2307_230763


namespace NUMINAMATH_CALUDE_max_overtakes_relay_race_l2307_230703

/-- Represents a relay race between two teams -/
structure RelayRace where
  num_runners : ℕ
  num_segments : ℕ
  runners_per_team : ℕ

/-- Represents the maximum number of overtakes in a relay race -/
def max_overtakes (race : RelayRace) : ℕ :=
  2 * (race.num_runners - 1)

/-- Theorem stating the maximum number of overtakes in the specific relay race scenario -/
theorem max_overtakes_relay_race :
  ∀ (race : RelayRace),
    race.num_runners = 20 →
    race.num_segments = 20 →
    race.runners_per_team = 20 →
    max_overtakes race = 38 := by
  sorry

end NUMINAMATH_CALUDE_max_overtakes_relay_race_l2307_230703


namespace NUMINAMATH_CALUDE_smallest_integer_l2307_230747

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 24) :
  b ≥ 360 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l2307_230747


namespace NUMINAMATH_CALUDE_divides_two_pow_minus_one_l2307_230762

theorem divides_two_pow_minus_one (n : ℕ) : n ≥ 1 → (n ∣ 2^n - 1 ↔ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_divides_two_pow_minus_one_l2307_230762


namespace NUMINAMATH_CALUDE_conference_left_handed_fraction_l2307_230720

/-- Represents the fraction of left-handed participants in a conference -/
def left_handed_fraction (total : ℕ) (red : ℕ) (blue : ℕ) (red_left : ℚ) (blue_left : ℚ) : ℚ :=
  (red_left * red + blue_left * blue) / total

/-- Theorem stating the fraction of left-handed participants in the conference -/
theorem conference_left_handed_fraction :
  ∀ (total : ℕ) (red : ℕ) (blue : ℕ),
  total > 0 →
  red + blue = total →
  red = blue →
  left_handed_fraction total red blue (1/3) (2/3) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_conference_left_handed_fraction_l2307_230720


namespace NUMINAMATH_CALUDE_sum_of_other_digits_l2307_230736

def is_form_76h4 (n : ℕ) : Prop :=
  ∃ h : ℕ, n = 7000 + 600 + 10 * h + 4

theorem sum_of_other_digits (n : ℕ) (h : ℕ) :
  is_form_76h4 n →
  h = 1 →
  n % 9 = 0 →
  (7 + 6 + 4 : ℕ) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_other_digits_l2307_230736


namespace NUMINAMATH_CALUDE_min_abs_z_complex_l2307_230767

theorem min_abs_z_complex (z : ℂ) (h : Complex.abs (z - 2*I) + Complex.abs (z - 5) = 7) :
  ∃ (w : ℂ), Complex.abs w = 10/7 ∧ ∀ z', Complex.abs (z' - 2*I) + Complex.abs (z' - 5) = 7 → Complex.abs w ≤ Complex.abs z' :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_complex_l2307_230767


namespace NUMINAMATH_CALUDE_crayon_count_l2307_230739

theorem crayon_count (initial_crayons added_crayons : ℕ) 
  (h1 : initial_crayons = 9)
  (h2 : added_crayons = 3) :
  initial_crayons + added_crayons = 12 := by
sorry

end NUMINAMATH_CALUDE_crayon_count_l2307_230739


namespace NUMINAMATH_CALUDE_equation_solution_l2307_230793

theorem equation_solution (x y : ℝ) : 3 * x - 4 * y = 5 → x = (1/3) * (5 + 4 * y) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2307_230793


namespace NUMINAMATH_CALUDE_shaded_area_in_circle_l2307_230709

theorem shaded_area_in_circle (r : ℝ) (h1 : r > 0) : 
  let circle_area := π * r^2
  let sector_area := 2 * π
  let sector_fraction := 1 / 8
  let triangle_area := r^2 / 2
  sector_area = sector_fraction * circle_area → 
  sector_area - triangle_area = 2 * π - 4 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_in_circle_l2307_230709


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l2307_230727

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l2307_230727


namespace NUMINAMATH_CALUDE_coefficient_a3b3_is_1400_l2307_230718

/-- The coefficient of a^3b^3 in (a+b)^6(c+1/c)^8 -/
def coefficient_a3b3 : ℕ :=
  (Nat.choose 6 3) * (Nat.choose 8 4)

/-- Theorem: The coefficient of a^3b^3 in (a+b)^6(c+1/c)^8 is 1400 -/
theorem coefficient_a3b3_is_1400 : coefficient_a3b3 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a3b3_is_1400_l2307_230718


namespace NUMINAMATH_CALUDE_difference_of_squares_numbers_l2307_230792

def is_difference_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > b ∧ n = a * a - b * b

theorem difference_of_squares_numbers :
  is_difference_of_squares 2020 ∧
  is_difference_of_squares 2022 ∧
  is_difference_of_squares 2023 ∧
  is_difference_of_squares 2024 ∧
  ¬is_difference_of_squares 2021 :=
sorry

end NUMINAMATH_CALUDE_difference_of_squares_numbers_l2307_230792


namespace NUMINAMATH_CALUDE_min_tiles_for_region_l2307_230702

/-- The number of tiles needed to cover a rectangular region -/
def tiles_needed (tile_length : ℕ) (tile_width : ℕ) (region_length : ℕ) (region_width : ℕ) : ℕ :=
  let region_area := region_length * region_width
  let tile_area := tile_length * tile_width
  (region_area + tile_area - 1) / tile_area

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

/-- Theorem stating the minimum number of tiles needed to cover the given region -/
theorem min_tiles_for_region : 
  tiles_needed 5 6 (3 * feet_to_inches) (4 * feet_to_inches) = 58 := by
  sorry

#eval tiles_needed 5 6 (3 * feet_to_inches) (4 * feet_to_inches)

end NUMINAMATH_CALUDE_min_tiles_for_region_l2307_230702


namespace NUMINAMATH_CALUDE_acid_dilution_l2307_230731

/-- Given m ounces of an m% acid solution, when x ounces of water are added,
    a new solution of (m-20)% concentration is formed. Assuming m > 25,
    prove that x = 20m / (m-20). -/
theorem acid_dilution (m : ℝ) (x : ℝ) (h : m > 25) :
  (m * m / 100 = (m - 20) / 100 * (m + x)) → x = 20 * m / (m - 20) := by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l2307_230731


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l2307_230700

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Stirling number of the second kind: number of ways to partition n objects into k non-empty subsets -/
def stirling2 (n k : ℕ) : ℕ := sorry

theorem distribute_five_balls_three_boxes : 
  distribute_balls 5 3 = 41 := by sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l2307_230700


namespace NUMINAMATH_CALUDE_milk_cost_percentage_l2307_230755

-- Define the costs and total amount
def sandwich_cost : ℝ := 4
def juice_cost : ℝ := 2 * sandwich_cost
def total_paid : ℝ := 21

-- Define the total cost of sandwich and juice
def sandwich_juice_total : ℝ := sandwich_cost + juice_cost

-- Define the cost of milk
def milk_cost : ℝ := total_paid - sandwich_juice_total

-- The theorem to prove
theorem milk_cost_percentage : 
  (milk_cost / sandwich_juice_total) * 100 = 75 := by sorry

end NUMINAMATH_CALUDE_milk_cost_percentage_l2307_230755


namespace NUMINAMATH_CALUDE_new_ratio_is_25_to_1_l2307_230799

/-- Represents the ratio of students to teachers -/
structure Ratio where
  students : ℕ
  teachers : ℕ

def initial_ratio : Ratio := { students := 50, teachers := 1 }
def initial_teachers : ℕ := 3
def student_increase : ℕ := 50
def teacher_increase : ℕ := 5

def new_ratio : Ratio :=
  { students := initial_ratio.students * initial_teachers + student_increase,
    teachers := initial_teachers + teacher_increase }

theorem new_ratio_is_25_to_1 : new_ratio = { students := 25, teachers := 1 } := by
  sorry

end NUMINAMATH_CALUDE_new_ratio_is_25_to_1_l2307_230799


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2307_230754

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) := ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_third_term : a 3 = 27)
  (h_ninth_term : a 9 = 3) :
  a 6 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2307_230754


namespace NUMINAMATH_CALUDE_min_sum_box_dimensions_l2307_230781

theorem min_sum_box_dimensions : 
  ∀ (l w h : ℕ+), 
  l * w * h = 3003 → 
  ∀ (a b c : ℕ+), 
  a * b * c = 3003 → 
  l + w + h ≤ a + b + c ∧
  ∃ (x y z : ℕ+), x * y * z = 3003 ∧ x + y + z = 45 := by
sorry

end NUMINAMATH_CALUDE_min_sum_box_dimensions_l2307_230781


namespace NUMINAMATH_CALUDE_abs_equals_diff_exists_l2307_230743

theorem abs_equals_diff_exists : ∃ x : ℝ, |x - 1| = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_equals_diff_exists_l2307_230743


namespace NUMINAMATH_CALUDE_circle_positions_l2307_230756

theorem circle_positions (a b d : ℝ) (h1 : a = 4) (h2 : b = 10) (h3 : b > a) :
  (∃ d, d = b - a) ∧
  (∃ d, d = b + a) ∧
  (∃ d, d > b + a) ∧
  (∃ d, d > b - a) :=
by sorry

end NUMINAMATH_CALUDE_circle_positions_l2307_230756


namespace NUMINAMATH_CALUDE_overtaking_points_l2307_230760

theorem overtaking_points (track_length : ℕ) (pedestrian_speed : ℝ) (cyclist_speed : ℝ) : 
  track_length = 55 →
  cyclist_speed = 1.55 * pedestrian_speed →
  pedestrian_speed > 0 →
  (∃ n : ℕ, n * (cyclist_speed - pedestrian_speed) = track_length ∧ n = 11) :=
by sorry

end NUMINAMATH_CALUDE_overtaking_points_l2307_230760


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2307_230798

theorem min_value_quadratic :
  let f (x : ℝ) := x^2 + 14*x + 10
  ∃ (y_min : ℝ), (∀ (x : ℝ), f x ≥ y_min) ∧ (∃ (x : ℝ), f x = y_min) ∧ y_min = -39 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2307_230798


namespace NUMINAMATH_CALUDE_valid_configurations_l2307_230757

/-- A configuration of lines and points on a plane -/
structure PlaneConfiguration where
  n : ℕ  -- number of points
  lines : Fin 3 → Set (ℝ × ℝ)  -- three lines represented as sets of points
  points : Fin n → ℝ × ℝ  -- n points

/-- Predicate to check if a point is on a line -/
def isOnLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  p ∈ l

/-- Predicate to check if a point is on either side of a line -/
def isOnEitherSide (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  ¬(isOnLine p l)

/-- The main theorem stating the possible values of n -/
theorem valid_configurations (c : PlaneConfiguration) :
  (∀ l : Fin 3, ∃! (s₁ s₂ : Finset (Fin c.n)),
    s₁.card = 2 ∧ s₂.card = 2 ∧ 
    (∀ i ∈ s₁, isOnEitherSide (c.points i) (c.lines l)) ∧
    (∀ i ∈ s₂, isOnEitherSide (c.points i) (c.lines l)) ∧
    (∀ i : Fin c.n, i ∉ s₁ ∧ i ∉ s₂ → isOnLine (c.points i) (c.lines l))) →
  c.n = 0 ∨ c.n = 1 ∨ c.n = 3 ∨ c.n = 4 ∨ c.n = 6 ∨ c.n = 7 :=
by sorry

end NUMINAMATH_CALUDE_valid_configurations_l2307_230757


namespace NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l2307_230772

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l2307_230772


namespace NUMINAMATH_CALUDE_volume_ratio_is_twenty_l2307_230717

-- Define the dimensions of the shapes
def cube_edge : ℝ := 1  -- 1 meter
def cuboid_width : ℝ := 0.5  -- 50 cm in meters
def cuboid_length : ℝ := 0.5  -- 50 cm in meters
def cuboid_height : ℝ := 0.2  -- 20 cm in meters

-- Define the volume functions
def cube_volume (edge : ℝ) : ℝ := edge ^ 3
def cuboid_volume (width length height : ℝ) : ℝ := width * length * height

-- Theorem statement
theorem volume_ratio_is_twenty :
  (cube_volume cube_edge) / (cuboid_volume cuboid_width cuboid_length cuboid_height) = 20 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_is_twenty_l2307_230717


namespace NUMINAMATH_CALUDE_star_operations_l2307_230728

/-- The ☆ operation for rational numbers -/
def star (a b : ℚ) : ℚ := a * b - a + b

/-- Theorem stating the results of the given operations -/
theorem star_operations :
  (star 2 (-3) = -11) ∧ (star (-2) (star 1 3) = -3) := by
  sorry

end NUMINAMATH_CALUDE_star_operations_l2307_230728


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2307_230723

theorem polynomial_evaluation : 
  let x : ℤ := -2
  2 * x^4 + 3 * x^3 - x^2 + 2 * x + 5 = 5 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2307_230723


namespace NUMINAMATH_CALUDE_garden_length_l2307_230785

/-- Proves that a rectangular garden with length twice its width and 300 yards of fencing has a length of 100 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- length is twice the width
  2 * length + 2 * width = 300 →  -- 300 yards of fencing encloses the garden
  length = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l2307_230785


namespace NUMINAMATH_CALUDE_negative_three_x_squared_times_two_x_l2307_230745

theorem negative_three_x_squared_times_two_x (x : ℝ) : (-3 * x)^2 * (2 * x) = 18 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_x_squared_times_two_x_l2307_230745


namespace NUMINAMATH_CALUDE_remaining_money_l2307_230776

def base_8_to_10 (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

def savings : ℕ := 5555

def ticket_cost : ℕ := 1200

theorem remaining_money :
  base_8_to_10 savings - ticket_cost = 1725 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l2307_230776


namespace NUMINAMATH_CALUDE_angle_terminal_side_l2307_230777

open Real

theorem angle_terminal_side (α : Real) :
  (tan α < 0 ∧ cos α < 0) → 
  (π / 2 < α ∧ α < π) :=
by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l2307_230777


namespace NUMINAMATH_CALUDE_train_length_is_200_emily_steps_l2307_230759

/-- Represents the movement of Emily relative to a train -/
structure EmilyAndTrain where
  emily_step : ℝ
  train_step : ℝ
  train_length : ℝ

/-- The conditions of Emily's run relative to the train -/
def emily_run_conditions (et : EmilyAndTrain) : Prop :=
  ∃ (e t : ℝ),
    et.emily_step = e ∧
    et.train_step = t ∧
    et.train_length = 300 * e + 300 * t ∧
    et.train_length = 90 * e - 90 * t

/-- The theorem stating that under the given conditions, 
    the train length is 200 times Emily's step length -/
theorem train_length_is_200_emily_steps 
  (et : EmilyAndTrain) 
  (h : emily_run_conditions et) : 
  et.train_length = 200 * et.emily_step := by
  sorry

end NUMINAMATH_CALUDE_train_length_is_200_emily_steps_l2307_230759


namespace NUMINAMATH_CALUDE_function_value_theorem_l2307_230794

theorem function_value_theorem (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f ((1/2) * x - 1) = 2 * x + 3) →
  f m = 6 →
  m = -(1/4) := by
sorry

end NUMINAMATH_CALUDE_function_value_theorem_l2307_230794


namespace NUMINAMATH_CALUDE_floor_abs_sum_l2307_230771

theorem floor_abs_sum : ⌊|(-3.1 : ℝ)|⌋ + |⌊(-3.1 : ℝ)⌋| = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l2307_230771


namespace NUMINAMATH_CALUDE_little_john_height_l2307_230725

/-- Conversion factor from centimeters to meters -/
def cm_to_m : ℝ := 0.01

/-- Conversion factor from millimeters to meters -/
def mm_to_m : ℝ := 0.001

/-- Little John's height in meters, centimeters, and millimeters -/
def height_m : ℝ := 2
def height_cm : ℝ := 8
def height_mm : ℝ := 3

/-- Theorem stating that Little John's height in meters is 2.083 -/
theorem little_john_height : 
  height_m + height_cm * cm_to_m + height_mm * mm_to_m = 2.083 := by
  sorry

end NUMINAMATH_CALUDE_little_john_height_l2307_230725


namespace NUMINAMATH_CALUDE_representatives_selection_theorem_l2307_230753

/-- The number of ways to select representatives from a group of students -/
def select_representatives (total_students : ℕ) (num_representatives : ℕ) (restricted_student : ℕ) : ℕ :=
  (total_students - 1) * (total_students - 1) * (total_students - 2)

/-- Theorem stating the number of ways to select 3 representatives from 5 students,
    with one student restricted from being the Mathematics representative -/
theorem representatives_selection_theorem :
  select_representatives 5 3 1 = 48 := by
  sorry

end NUMINAMATH_CALUDE_representatives_selection_theorem_l2307_230753


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2307_230761

/-- Proves that the repeating decimal 0.35̄ is equal to 5/14 -/
theorem repeating_decimal_to_fraction : 
  ∀ x : ℚ, (∃ n : ℕ, x = (35 : ℚ) / (100^n - 1)) → x = 5/14 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2307_230761


namespace NUMINAMATH_CALUDE_parallel_perpendicular_transitivity_l2307_230733

-- Define the space
variable (S : Type*) [MetricSpace S]

-- Define lines and planes
variable (Line Plane : Type*)

-- Define the lines m and n, and the plane α
variable (m n : Line) (α : Plane)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the distinct relation for lines
variable (distinct : Line → Line → Prop)

-- Theorem statement
theorem parallel_perpendicular_transitivity 
  (h_distinct : distinct m n)
  (h_parallel : parallel m n)
  (h_perpendicular : perpendicular n α) :
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_transitivity_l2307_230733


namespace NUMINAMATH_CALUDE_inequality_solution_condition_l2307_230710

theorem inequality_solution_condition (a : ℝ) : 
  (∃! x y : ℤ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ 
    (∀ z : ℤ, z < 0 → ((z + a) / 2 ≥ 1) ↔ (z = x ∨ z = y))) 
  → 4 ≤ a ∧ a < 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_condition_l2307_230710


namespace NUMINAMATH_CALUDE_edward_money_problem_l2307_230780

def initial_amount : ℕ := 14
def spent_amount : ℕ := 17
def received_amount : ℕ := 10
def final_amount : ℕ := 7

theorem edward_money_problem :
  initial_amount - spent_amount + received_amount = final_amount :=
by sorry

end NUMINAMATH_CALUDE_edward_money_problem_l2307_230780


namespace NUMINAMATH_CALUDE_product_units_digit_of_first_five_composite_l2307_230779

def first_five_composite_numbers : List Nat := [4, 6, 8, 9, 10]

def units_digit (n : Nat) : Nat := n % 10

def product_units_digit (numbers : List Nat) : Nat :=
  units_digit (numbers.foldl (·*·) 1)

theorem product_units_digit_of_first_five_composite : 
  product_units_digit first_five_composite_numbers = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_units_digit_of_first_five_composite_l2307_230779


namespace NUMINAMATH_CALUDE_largest_common_term_l2307_230732

theorem largest_common_term (n m : ℕ) : 
  163 = 3 + 8 * n ∧ 
  163 = 5 + 9 * m ∧ 
  163 ≤ 200 ∧ 
  ∀ k, k > 163 → k ≤ 200 → (k - 3) % 8 ≠ 0 ∨ (k - 5) % 9 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l2307_230732


namespace NUMINAMATH_CALUDE_computer_price_after_15_years_l2307_230758

/-- Proves that a computer's price after 15 years of depreciation is 2400 yuan,
    given an initial price of 8100 yuan and a 1/3 price decrease every 5 years. -/
theorem computer_price_after_15_years
  (initial_price : ℝ)
  (price_decrease_ratio : ℝ)
  (price_decrease_period : ℕ)
  (total_time : ℕ)
  (h1 : initial_price = 8100)
  (h2 : price_decrease_ratio = 1 / 3)
  (h3 : price_decrease_period = 5)
  (h4 : total_time = 15)
  : initial_price * (1 - price_decrease_ratio) ^ (total_time / price_decrease_period) = 2400 :=
sorry

end NUMINAMATH_CALUDE_computer_price_after_15_years_l2307_230758


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2307_230782

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₆ = 6 and a₉ = 9, prove that a₃ = 3 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_a6 : a 6 = 6) 
  (h_a9 : a 9 = 9) : 
  a 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2307_230782


namespace NUMINAMATH_CALUDE_reciprocal_sum_greater_than_four_l2307_230784

theorem reciprocal_sum_greater_than_four 
  (a b c : ℝ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (pos_c : 0 < c) 
  (sum_of_squares : a^2 + b^2 + c^2 = 1) : 
  1/a + 1/b + 1/c > 4 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_greater_than_four_l2307_230784


namespace NUMINAMATH_CALUDE_total_books_calculation_l2307_230738

def initial_books : ℕ := 9
def added_books : ℕ := 10

theorem total_books_calculation :
  initial_books + added_books = 19 :=
by sorry

end NUMINAMATH_CALUDE_total_books_calculation_l2307_230738


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2307_230791

/-- A square inscribed in a right triangle -/
structure InscribedSquare where
  /-- The side length of the inscribed square -/
  side : ℝ
  /-- The distance from one vertex of the right triangle to where the square touches the hypotenuse -/
  dist1 : ℝ
  /-- The distance from the other vertex of the right triangle to where the square touches the hypotenuse -/
  dist2 : ℝ
  /-- The constraint that the square is properly inscribed in the right triangle -/
  inscribed : side * side = dist1 * dist2

/-- The theorem stating that a square inscribed in a right triangle with specific measurements has an area of 975 -/
theorem inscribed_square_area (s : InscribedSquare) 
    (h1 : s.dist1 = 15) 
    (h2 : s.dist2 = 65) : 
  s.side * s.side = 975 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2307_230791


namespace NUMINAMATH_CALUDE_work_ratio_women_to_men_l2307_230786

/-- The ratio of work done by women to men given specific work conditions -/
theorem work_ratio_women_to_men :
  let men_count : ℕ := 15
  let men_days : ℕ := 21
  let men_hours_per_day : ℕ := 8
  let women_count : ℕ := 21
  let women_days : ℕ := 36
  let women_hours_per_day : ℕ := 5
  let men_total_hours : ℕ := men_count * men_days * men_hours_per_day
  let women_total_hours : ℕ := women_count * women_days * women_hours_per_day
  (men_total_hours : ℚ) / women_total_hours = 2 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_work_ratio_women_to_men_l2307_230786


namespace NUMINAMATH_CALUDE_eliminate_first_power_term_l2307_230750

theorem eliminate_first_power_term (a m : ℝ) : 
  (∀ k, (a + m) * (a + 1/2) = k * a^2 + c) ↔ m = -1/2 := by sorry

end NUMINAMATH_CALUDE_eliminate_first_power_term_l2307_230750


namespace NUMINAMATH_CALUDE_oranges_thrown_away_l2307_230713

theorem oranges_thrown_away (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 40 → new = 21 → final = 36 → initial - (initial - final + new) = 25 := by
  sorry

end NUMINAMATH_CALUDE_oranges_thrown_away_l2307_230713


namespace NUMINAMATH_CALUDE_triomino_corner_reachability_l2307_230734

/-- Represents an L-triomino on a board -/
structure Triomino where
  center : Nat × Nat
  leg1 : Nat × Nat
  leg2 : Nat × Nat

/-- Represents a board of size m × n -/
structure Board (m n : Nat) where
  triomino : Triomino

/-- Defines a valid initial position of the triomino -/
def initial_position (m n : Nat) : Board m n :=
  { triomino := { center := (0, 0), leg1 := (0, 1), leg2 := (1, 0) } }

/-- Defines a valid rotation of the triomino -/
def can_rotate (b : Board m n) : Prop :=
  ∃ new_position : Triomino, true  -- We assume any rotation is possible

/-- Defines if a triomino can reach the bottom right corner -/
def can_reach_corner (m n : Nat) : Prop :=
  ∃ final_position : Board m n, 
    final_position.triomino.center = (m - 1, n - 1)

/-- The main theorem to be proved -/
theorem triomino_corner_reachability (m n : Nat) :
  can_reach_corner m n ↔ m % 2 = 1 ∧ n % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_triomino_corner_reachability_l2307_230734


namespace NUMINAMATH_CALUDE_certain_number_proof_l2307_230790

theorem certain_number_proof (x : ℝ) : 
  (0.02: ℝ)^2 + x^2 + (0.035 : ℝ)^2 = 100 * ((0.002 : ℝ)^2 + (0.052 : ℝ)^2 + (0.0035 : ℝ)^2) → 
  x = 0.52 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2307_230790


namespace NUMINAMATH_CALUDE_business_value_proof_l2307_230708

theorem business_value_proof (total_shares : ℚ) (man_shares : ℚ) (sold_fraction : ℚ) (sale_price : ℚ) :
  total_shares = 1 →
  man_shares = 1 / 3 →
  sold_fraction = 3 / 5 →
  sale_price = 2000 →
  (man_shares * sold_fraction * total_shares⁻¹) * (total_shares / (man_shares * sold_fraction)) * sale_price = 10000 :=
by sorry

end NUMINAMATH_CALUDE_business_value_proof_l2307_230708


namespace NUMINAMATH_CALUDE_product_of_numbers_l2307_230752

theorem product_of_numbers (x y : ℝ) : 
  x + y = 25 → x - y = 7 → x * y = 144 := by sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2307_230752


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2307_230730

-- Define the types of cows
inductive CowType
| Holstein
| Jersey

-- Define the cost of each cow type
def cow_cost : CowType → Nat
| CowType.Holstein => 260
| CowType.Jersey => 170

-- Define the number of hearts in a standard deck
def hearts_in_deck : Nat := 52

-- Define the number of cows
def total_cows : Nat := 2 * hearts_in_deck

-- Define the ratio of Holstein to Jersey cows
def holstein_ratio : Nat := 3
def jersey_ratio : Nat := 2

-- Define the sales tax rate
def sales_tax_rate : Rat := 5 / 100

-- Define the transportation cost per cow
def transport_cost_per_cow : Nat := 20

-- Define the function to calculate the total cost
def total_cost : ℚ :=
  let holstein_count := (holstein_ratio * total_cows) / (holstein_ratio + jersey_ratio)
  let jersey_count := (jersey_ratio * total_cows) / (holstein_ratio + jersey_ratio)
  let base_cost := holstein_count * cow_cost CowType.Holstein + jersey_count * cow_cost CowType.Jersey
  let sales_tax := base_cost * sales_tax_rate
  let transport_cost := total_cows * transport_cost_per_cow
  (base_cost + sales_tax + transport_cost : ℚ)

-- Theorem statement
theorem total_cost_is_correct : total_cost = 26324.50 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2307_230730


namespace NUMINAMATH_CALUDE_sixth_number_in_sequence_l2307_230714

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n

theorem sixth_number_in_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_sum : a 2 + a 3 = 24) :
  a 6 = 128 := by
sorry

end NUMINAMATH_CALUDE_sixth_number_in_sequence_l2307_230714


namespace NUMINAMATH_CALUDE_perfect_square_sum_in_partition_l2307_230765

theorem perfect_square_sum_in_partition (n : ℕ) (A B : Set ℕ) 
  (h1 : n ≥ 15)
  (h2 : A ⊆ Finset.range (n + 1))
  (h3 : B ⊆ Finset.range (n + 1))
  (h4 : A ∩ B = ∅)
  (h5 : A ∪ B = Finset.range (n + 1))
  (h6 : A ≠ Finset.range (n + 1))
  (h7 : B ≠ Finset.range (n + 1)) :
  ∃ (x y : ℕ), (x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ ∃ (k : ℕ), x + y = k^2) ∨
               (x ∈ B ∧ y ∈ B ∧ x ≠ y ∧ ∃ (k : ℕ), x + y = k^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_sum_in_partition_l2307_230765


namespace NUMINAMATH_CALUDE_median_exists_l2307_230742

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
  2 * (s.filter (· ≤ m)).card ≥ s.card ∧
  2 * (s.filter (· ≥ m)).card ≥ s.card

theorem median_exists : ∃ a : ℝ, is_median {a, 2, 4, 0, 5} 4 := by
  sorry

end NUMINAMATH_CALUDE_median_exists_l2307_230742


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l2307_230701

/-- Represents a quadrilateral with extended sides --/
structure ExtendedQuadrilateral where
  -- Original quadrilateral sides
  WZ : ℝ
  WX : ℝ
  XY : ℝ
  YZ : ℝ
  -- Extended sides
  ZW' : ℝ
  XX' : ℝ
  YY' : ℝ
  Z'W : ℝ
  -- Area of original quadrilateral
  area : ℝ

/-- Theorem stating the area of the extended quadrilateral --/
theorem extended_quadrilateral_area 
  (q : ExtendedQuadrilateral) 
  (h1 : q.WZ = 10 ∧ q.ZW' = 10)
  (h2 : q.WX = 6 ∧ q.XX' = 6)
  (h3 : q.XY = 7 ∧ q.YY' = 7)
  (h4 : q.YZ = 12 ∧ q.Z'W = 12)
  (h5 : q.area = 15) :
  ∃ (area_extended : ℝ), area_extended = 45 := by
  sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l2307_230701


namespace NUMINAMATH_CALUDE_count_divisible_integers_l2307_230715

theorem count_divisible_integers : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (1764 : ℤ) ∣ (m^2 - 3)) ∧ 
    (∀ m : ℕ, m > 0 ∧ (1764 : ℤ) ∣ (m^2 - 3) → m ∈ S) ∧
    S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_integers_l2307_230715


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l2307_230748

theorem cheryl_material_usage
  (material1 : ℚ)
  (material2 : ℚ)
  (leftover : ℚ)
  (h1 : material1 = 4 / 19)
  (h2 : material2 = 2 / 13)
  (h3 : leftover = 4 / 26)
  : material1 + material2 - leftover = 52 / 247 :=
by sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l2307_230748


namespace NUMINAMATH_CALUDE_base_10_to_base_12_250_l2307_230719

def base_12_digit (n : ℕ) : Char :=
  if n < 10 then Char.ofNat (n + 48)
  else if n = 10 then 'A'
  else 'B'

def to_base_12 (n : ℕ) : List Char :=
  if n < 12 then [base_12_digit n]
  else (to_base_12 (n / 12)) ++ [base_12_digit (n % 12)]

theorem base_10_to_base_12_250 :
  to_base_12 250 = ['1', 'A'] :=
sorry

end NUMINAMATH_CALUDE_base_10_to_base_12_250_l2307_230719


namespace NUMINAMATH_CALUDE_balloon_difference_l2307_230764

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 5

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 3

/-- Theorem stating the difference in number of balloons between Allan and Jake -/
theorem balloon_difference : allan_balloons - jake_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l2307_230764


namespace NUMINAMATH_CALUDE_max_value_parabola_l2307_230787

/-- The maximum value of y = -3x^2 + 7, where x is a real number, is 7. -/
theorem max_value_parabola :
  ∀ x : ℝ, -3 * x^2 + 7 ≤ 7 ∧ ∃ x₀ : ℝ, -3 * x₀^2 + 7 = 7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_parabola_l2307_230787


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2307_230741

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def CommonDifference (a : ℕ → ℝ) : ℝ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a5 : a 5 = 10)
  (h_a10 : a 10 = -5) :
  CommonDifference a = -3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2307_230741


namespace NUMINAMATH_CALUDE_count_nondegenerate_triangles_l2307_230735

/-- A point in the integer grid -/
structure GridPoint where
  s : Nat
  t : Nat
  s_bound : s ≤ 4
  t_bound : t ≤ 4

/-- A triangle represented by three grid points -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Predicate to check if three points are collinear -/
def collinear (p1 p2 p3 : GridPoint) : Prop :=
  (p2.s - p1.s) * (p3.t - p1.t) = (p3.s - p1.s) * (p2.t - p1.t)

/-- Predicate to check if a triangle is nondegenerate -/
def nondegenerate (t : GridTriangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all valid grid points -/
def gridPoints : Finset GridPoint :=
  sorry

/-- The set of all possible triangles formed by grid points -/
def allTriangles : Finset GridTriangle :=
  sorry

/-- The set of all nondegenerate triangles -/
def nondegenerateTriangles : Finset GridTriangle :=
  sorry

theorem count_nondegenerate_triangles :
  Finset.card nondegenerateTriangles = 2170 :=
sorry

end NUMINAMATH_CALUDE_count_nondegenerate_triangles_l2307_230735


namespace NUMINAMATH_CALUDE_function_properties_l2307_230740

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) (f'' : ℝ → ℝ) 
  (h_even : is_even f)
  (h_deriv : ∀ x, HasDerivAt f (f'' x) x)
  (h_eq : ∀ x, f (x - 1/2) + f (x + 1) = 0)
  (h_val : Real.exp 3 * f 2018 = 1)
  (h_ineq : ∀ x, f x > f'' (-x)) :
  {x : ℝ | f (x - 1) > 1 / Real.exp x} = {x : ℝ | x > 3} := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2307_230740


namespace NUMINAMATH_CALUDE_permutation_identities_l2307_230712

def A (n m : ℕ) : ℕ := (n :: List.range m).prod

theorem permutation_identities :
  (∀ n m : ℕ, A (n + 1) (m + 1) - A n m = n^2 * A (n - 1) (m - 1)) ∧
  (∀ n m : ℕ, A n m = n * A (n - 1) (m - 1)) := by
  sorry

end NUMINAMATH_CALUDE_permutation_identities_l2307_230712


namespace NUMINAMATH_CALUDE_divisibility_of_power_plus_one_l2307_230751

theorem divisibility_of_power_plus_one (n : ℕ) : 
  ∃ k : ℤ, (2 : ℤ) ^ (3 ^ n) + 1 = k * (3 : ℤ) ^ (n + 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_power_plus_one_l2307_230751


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2307_230707

theorem coefficient_x_cubed_in_expansion : 
  let expansion := (fun x => (2 * x + 1) * (x - 1)^5)
  ∃ a b c d e f, ∀ x, 
    expansion x = a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f ∧ 
    c = -10 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2307_230707


namespace NUMINAMATH_CALUDE_range_of_power_function_l2307_230705

/-- The range of f(x) = x^k + c on [0, ∞) is [c, ∞) when k > 0 -/
theorem range_of_power_function (k : ℝ) (c : ℝ) (h : k > 0) :
  Set.range (fun x : ℝ => x^k + c) = Set.Ici c :=
by sorry


end NUMINAMATH_CALUDE_range_of_power_function_l2307_230705


namespace NUMINAMATH_CALUDE_trigonometric_problem_l2307_230766

open Real

theorem trigonometric_problem (α : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : sin α = 2 * Real.sqrt 5 / 5) :
  (tan α = 2) ∧ 
  ((4 * sin (π - α) + 2 * cos (2 * π - α)) / (sin (π/2 - α) - sin α) = -10) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l2307_230766


namespace NUMINAMATH_CALUDE_bike_rental_fixed_fee_bike_rental_fixed_fee_proof_l2307_230724

/-- The fixed fee for renting a bike, given the total cost formula and a specific rental case. -/
theorem bike_rental_fixed_fee : ℝ → Prop :=
  fun fixed_fee =>
    let total_cost := fun (hours : ℝ) => fixed_fee + 7 * hours
    total_cost 9 = 80 → fixed_fee = 17

/-- Proof of the bike rental fixed fee theorem -/
theorem bike_rental_fixed_fee_proof : bike_rental_fixed_fee 17 := by
  sorry

end NUMINAMATH_CALUDE_bike_rental_fixed_fee_bike_rental_fixed_fee_proof_l2307_230724


namespace NUMINAMATH_CALUDE_raft_travel_time_l2307_230716

/-- The number of days it takes for a ship to travel from Chongqing to Shanghai -/
def ship_cq_to_sh : ℝ := 5

/-- The number of days it takes for a ship to travel from Shanghai to Chongqing -/
def ship_sh_to_cq : ℝ := 7

/-- The number of days it takes for a raft to drift from Chongqing to Shanghai -/
def raft_cq_to_sh : ℝ := 35

/-- Theorem stating that the raft travel time satisfies the given conditions -/
theorem raft_travel_time :
  1 / ship_cq_to_sh - 1 / raft_cq_to_sh = 1 / ship_sh_to_cq + 1 / raft_cq_to_sh :=
by sorry

end NUMINAMATH_CALUDE_raft_travel_time_l2307_230716


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l2307_230746

theorem stratified_sampling_proportion (second_year_total : ℕ) (third_year_total : ℕ) (third_year_sample : ℕ) :
  second_year_total = 1600 →
  third_year_total = 1400 →
  third_year_sample = 70 →
  (third_year_sample : ℚ) / third_year_total = 80 / second_year_total :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l2307_230746


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l2307_230797

/-- A quadratic equation with coefficients a, b, and c has two distinct real roots if and only if its discriminant is positive. -/
axiom quadratic_two_distinct_roots (a b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ b^2 - 4*a*c > 0

/-- For the quadratic equation x^2 + 2x + k = 0 to have two distinct real roots, k must be less than 1. -/
theorem quadratic_distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + k = 0 ∧ y^2 + 2*y + k = 0) ↔ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l2307_230797


namespace NUMINAMATH_CALUDE_emily_game_lives_l2307_230711

def game_lives (initial : ℕ) (lost : ℕ) (final : ℕ) : ℕ :=
  final - (initial - lost)

theorem emily_game_lives :
  game_lives 42 25 41 = 24 := by
  sorry

end NUMINAMATH_CALUDE_emily_game_lives_l2307_230711


namespace NUMINAMATH_CALUDE_parabola_c_value_l2307_230744

/-- A parabola with vertex at (-2, 3) passing through (2, 7) has c = 4 in its equation y = ax^2 + bx + c -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Condition 1
  (3 = a * (-2)^2 + b * (-2) + c) →       -- Condition 2 (vertex)
  (3 = a * 4 + b * (-2) + c) →            -- Condition 2 (vertex)
  (7 = a * 2^2 + b * 2 + c) →             -- Condition 3
  c = 4 := by
sorry


end NUMINAMATH_CALUDE_parabola_c_value_l2307_230744


namespace NUMINAMATH_CALUDE_four_digit_cubes_divisible_by_16_l2307_230749

theorem four_digit_cubes_divisible_by_16 :
  (∃! (s : Finset ℕ), s = {n : ℕ | 1000 ≤ (2*n)^3 ∧ (2*n)^3 ≤ 9999} ∧ Finset.card s = 3) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_cubes_divisible_by_16_l2307_230749


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2307_230706

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^3 + 1/x^3 = 332 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2307_230706


namespace NUMINAMATH_CALUDE_greg_and_sarah_apples_l2307_230775

/-- Represents the number of apples each person has -/
structure AppleDistribution where
  greg : ℕ
  sarah : ℕ
  susan : ℕ
  mark : ℕ
  mom : ℕ

/-- Checks if the apple distribution satisfies the given conditions -/
def is_valid_distribution (d : AppleDistribution) : Prop :=
  d.greg = d.sarah ∧
  d.susan = 2 * d.greg ∧
  d.mark = d.susan - 5 ∧
  d.mom = 49

/-- Theorem stating that Greg and Sarah have 18 apples in total -/
theorem greg_and_sarah_apples (d : AppleDistribution) 
  (h : is_valid_distribution d) : d.greg + d.sarah = 18 := by
  sorry

end NUMINAMATH_CALUDE_greg_and_sarah_apples_l2307_230775


namespace NUMINAMATH_CALUDE_four_digit_square_completion_l2307_230773

theorem four_digit_square_completion : 
  ∃ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧ 
    ∃ k : ℕ, (400 * 10000 + n) = k^2 :=
sorry

end NUMINAMATH_CALUDE_four_digit_square_completion_l2307_230773


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2307_230737

theorem cube_volume_ratio : 
  let cube1_edge_length : ℚ := 10  -- in inches
  let cube2_edge_length : ℚ := 5 * 12  -- 5 feet converted to inches
  let volume_ratio := (cube1_edge_length / cube2_edge_length) ^ 3
  volume_ratio = 1 / 216 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2307_230737


namespace NUMINAMATH_CALUDE_increasing_order_abc_l2307_230770

theorem increasing_order_abc (a b c : ℝ) : 
  a = 2^(4/3) → b = 3^(2/3) → c = 25^(1/3) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_increasing_order_abc_l2307_230770


namespace NUMINAMATH_CALUDE_red_hair_ratio_l2307_230778

theorem red_hair_ratio (red_hair_count : ℕ) (total_count : ℕ) 
  (h1 : red_hair_count = 9) 
  (h2 : total_count = 48) : 
  (red_hair_count : ℚ) / total_count = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_red_hair_ratio_l2307_230778


namespace NUMINAMATH_CALUDE_medium_pizza_slices_l2307_230768

theorem medium_pizza_slices :
  -- Define the number of slices for small and large pizzas
  let small_slices : ℕ := 6
  let large_slices : ℕ := 12
  -- Define the total number of pizzas and the number of each size
  let total_pizzas : ℕ := 15
  let small_pizzas : ℕ := 4
  let medium_pizzas : ℕ := 5
  -- Define the total number of slices
  let total_slices : ℕ := 136
  -- Calculate the number of large pizzas
  let large_pizzas : ℕ := total_pizzas - small_pizzas - medium_pizzas
  -- Define the number of slices in a medium pizza as a variable
  ∀ medium_slices : ℕ,
  -- If the total slices equation holds
  (small_pizzas * small_slices + medium_pizzas * medium_slices + large_pizzas * large_slices = total_slices) →
  -- Then the number of slices in a medium pizza must be 8
  medium_slices = 8 := by
sorry

end NUMINAMATH_CALUDE_medium_pizza_slices_l2307_230768


namespace NUMINAMATH_CALUDE_inequality_proof_l2307_230788

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : (x^2 / (1 + x^2)) + (y^2 / (1 + y^2)) + (z^2 / (1 + z^2)) = 2) : 
  (x / (1 + x^2)) + (y / (1 + y^2)) + (z / (1 + z^2)) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2307_230788


namespace NUMINAMATH_CALUDE_a_range_if_increasing_l2307_230795

/-- The sequence defined by a_n = an^2 + n -/
def a_seq (a : ℝ) (n : ℕ) : ℝ := a * n^2 + n

/-- The theorem stating that if the sequence is increasing, then a is non-negative -/
theorem a_range_if_increasing (a : ℝ) :
  (∀ n : ℕ, a_seq a n < a_seq a (n + 1)) → a ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_a_range_if_increasing_l2307_230795


namespace NUMINAMATH_CALUDE_tangent_circles_exist_l2307_230774

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is on a circle -/
def IsOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Predicate to check if a point is on a line -/
def IsOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Predicate to check if two circles are externally tangent -/
def AreExternallyTangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- Predicate to check if a circle touches another circle at a point -/
def CircleTouchesCircleAt (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  IsOnCircle p c1 ∧ IsOnCircle p c2 ∧ AreExternallyTangent c1 c2

/-- Predicate to check if a circle touches a line at a point -/
def CircleTouchesLineAt (c : Circle) (l : Line) (p : ℝ × ℝ) : Prop :=
  IsOnCircle p c ∧ IsOnLine p l

/-- The main theorem -/
theorem tangent_circles_exist (k : Circle) (e : Line) (P Q : ℝ × ℝ)
    (h_P : IsOnCircle P k) (h_Q : IsOnLine Q e) :
    ∃ (c1 c2 : Circle),
      c1.radius = c2.radius ∧
      AreExternallyTangent c1 c2 ∧
      CircleTouchesCircleAt c1 k P ∧
      CircleTouchesLineAt c2 e Q := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_exist_l2307_230774


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2307_230726

theorem polynomial_expansion (t : ℚ) : 
  (3*t^3 + 2*t^2 - 4*t + 3) * (-4*t^3 + 3*t - 5) = 
  -12*t^6 - 8*t^5 + 25*t^4 - 21*t^3 - 22*t^2 + 29*t - 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2307_230726


namespace NUMINAMATH_CALUDE_ellipse_area_irrational_l2307_230721

/-- The area of an ellipse with rational semi-major and semi-minor axes is irrational -/
theorem ellipse_area_irrational (a b : ℚ) (h_a : a > 0) (h_b : b > 0) : 
  Irrational (Real.pi * (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_area_irrational_l2307_230721


namespace NUMINAMATH_CALUDE_banana_arrangements_eq_60_l2307_230704

def banana_arrangements : ℕ :=
  let total_letters : ℕ := 6
  let b_count : ℕ := 1
  let n_count : ℕ := 2
  let a_count : ℕ := 3
  Nat.factorial total_letters / (Nat.factorial b_count * Nat.factorial n_count * Nat.factorial a_count)

theorem banana_arrangements_eq_60 : banana_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_eq_60_l2307_230704


namespace NUMINAMATH_CALUDE_function_properties_l2307_230789

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := Real.log (a^x - b^x) / Real.log 10

-- State the theorem
theorem function_properties (a b : ℝ) (ha : a > 1) (hb : 0 < b) (hab : b < 1) :
  -- 1. Domain of f is (0, +∞)
  (∀ x : ℝ, x > 0 → (a^x - b^x > 0)) ∧
  -- 2. No two distinct points with same y-value
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f a b x₁ ≠ f a b x₂) ∧
  -- 3. Condition for f to be positive on (1, +∞)
  (a ≥ b + 1 → ∀ x : ℝ, x > 1 → f a b x > 0) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2307_230789


namespace NUMINAMATH_CALUDE_star_sum_equals_396_l2307_230783

def star (a b : ℕ) : ℕ := a * a - b * b

theorem star_sum_equals_396 : 
  (List.range 18).foldl (λ acc i => acc + star (i + 3) (i + 2)) 0 = 396 := by
  sorry

end NUMINAMATH_CALUDE_star_sum_equals_396_l2307_230783


namespace NUMINAMATH_CALUDE_prob_odd_sum_is_two_thirds_l2307_230769

/-- A card is represented by a natural number between 1 and 4 -/
def Card := {n : ℕ // 1 ≤ n ∧ n ≤ 4}

/-- The set of all possible cards -/
def allCards : Finset Card := sorry

/-- A function to check if the sum of two cards is odd -/
def isOddSum (c1 c2 : Card) : Bool := sorry

/-- The set of all pairs of cards -/
def allPairs : Finset (Card × Card) := sorry

/-- The set of all pairs of cards with odd sum -/
def oddSumPairs : Finset (Card × Card) := sorry

/-- The probability of drawing two cards with odd sum -/
def probOddSum : ℚ := (Finset.card oddSumPairs : ℚ) / (Finset.card allPairs : ℚ)

theorem prob_odd_sum_is_two_thirds : probOddSum = 2/3 := by sorry

end NUMINAMATH_CALUDE_prob_odd_sum_is_two_thirds_l2307_230769


namespace NUMINAMATH_CALUDE_batsman_average_after_25th_innings_l2307_230729

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: A batsman's average after 25 innings -/
theorem batsman_average_after_25th_innings 
  (stats : BatsmanStats)
  (h1 : stats.innings = 24)
  (h2 : newAverage stats 80 = stats.average + 3)
  : newAverage stats 80 = 8 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_after_25th_innings_l2307_230729
