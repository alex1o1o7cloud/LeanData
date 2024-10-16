import Mathlib

namespace NUMINAMATH_CALUDE_division_problem_l3809_380992

theorem division_problem :
  ∃! x : ℕ, x < 50 ∧ ∃ m : ℕ, 100 = m * x + 6 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_division_problem_l3809_380992


namespace NUMINAMATH_CALUDE_pump_count_proof_l3809_380919

/-- The number of pumps in the first scenario -/
def num_pumps : ℕ := 3

/-- The number of hours worked per day in the first scenario -/
def hours_per_day_1 : ℕ := 8

/-- The number of days to empty the tank in the first scenario -/
def days_to_empty_1 : ℕ := 2

/-- The number of pumps in the second scenario -/
def num_pumps_2 : ℕ := 8

/-- The number of hours worked per day in the second scenario -/
def hours_per_day_2 : ℕ := 6

/-- The number of days to empty the tank in the second scenario -/
def days_to_empty_2 : ℕ := 1

/-- The capacity of the tank in pump-hours -/
def tank_capacity : ℕ := num_pumps_2 * hours_per_day_2 * days_to_empty_2

theorem pump_count_proof :
  num_pumps * hours_per_day_1 * days_to_empty_1 = tank_capacity :=
by sorry

end NUMINAMATH_CALUDE_pump_count_proof_l3809_380919


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l3809_380998

/-- The area of the circle described by the polar equation r = 3 cos θ - 4 sin θ is 25π/4 -/
theorem circle_area_from_polar_equation :
  let r : ℝ → ℝ := λ θ => 3 * Real.cos θ - 4 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ, (r θ * Real.cos θ - center.1)^2 + (r θ * Real.sin θ - center.2)^2 = radius^2) ∧
    π * radius^2 = 25 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l3809_380998


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3809_380930

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where √3b = 2c sin B, c = √7, and a + b = 5, prove that:
    1. The angle C is equal to π/3
    2. The area of triangle ABC is (3√3)/2 -/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π/2 →
  0 < B → B < π/2 →
  0 < C → C < π/2 →
  Real.sqrt 3 * b = 2 * c * Real.sin B →
  c = Real.sqrt 7 →
  a + b = 5 →
  C = π/3 ∧ (1/2 * a * b * Real.sin C = (3 * Real.sqrt 3) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3809_380930


namespace NUMINAMATH_CALUDE_fraction_of_total_l3809_380996

theorem fraction_of_total (total : ℝ) (r_amount : ℝ) (h1 : total = 5000) (h2 : r_amount = 2000.0000000000002) :
  r_amount / total = 0.40000000000000004 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_total_l3809_380996


namespace NUMINAMATH_CALUDE_unique_x_satisfying_three_inequalities_l3809_380995

theorem unique_x_satisfying_three_inequalities :
  ∃! (x : ℕ), (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37) ∧
              ¬(2 * x ≥ 21) ∧ ¬(x > 7) :=
by sorry

end NUMINAMATH_CALUDE_unique_x_satisfying_three_inequalities_l3809_380995


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3809_380931

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I)^2 = 1 + Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3809_380931


namespace NUMINAMATH_CALUDE_bud_uncle_age_ratio_l3809_380986

/-- The ratio of Bud's age to his uncle's age -/
def age_ratio (bud_age uncle_age : ℕ) : ℚ :=
  bud_age / uncle_age

/-- Bud's age -/
def bud_age : ℕ := 8

/-- Bud's uncle's age -/
def uncle_age : ℕ := 24

theorem bud_uncle_age_ratio :
  age_ratio bud_age uncle_age = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_bud_uncle_age_ratio_l3809_380986


namespace NUMINAMATH_CALUDE_incorrect_expressions_l3809_380972

theorem incorrect_expressions (x y : ℝ) (h : x / y = 2 / 5) :
  ((x + 3 * y) / (2 * y) ≠ 13 / 10) ∧ ((2 * y - x) / (3 * y) ≠ 7 / 15) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expressions_l3809_380972


namespace NUMINAMATH_CALUDE_physics_to_music_ratio_l3809_380965

/-- Proves that the ratio of physics marks to music marks is 1:2 given the marks in other subjects and total marks -/
theorem physics_to_music_ratio (science music social_studies total : ℕ) (physics : ℚ) :
  science = 70 →
  music = 80 →
  social_studies = 85 →
  total = 275 →
  physics = music * (1 / 2) →
  science + music + social_studies + physics = total →
  physics / music = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_physics_to_music_ratio_l3809_380965


namespace NUMINAMATH_CALUDE_only_sperm_has_one_set_zygote_has_two_sets_somatic_cell_has_two_sets_spermatogonium_is_somatic_cell_sperm_formed_by_meiosis_l3809_380918

-- Define the types of cells
inductive CellType
  | Zygote
  | SomaticCell
  | Spermatogonium
  | Sperm

-- Define a function that returns the number of chromosome sets for each cell type
def chromosome_sets (cell : CellType) : ℕ :=
  match cell with
  | CellType.Zygote => 2
  | CellType.SomaticCell => 2
  | CellType.Spermatogonium => 2
  | CellType.Sperm => 1

-- Theorem: Only sperm has one set of chromosomes
theorem only_sperm_has_one_set :
  ∀ (cell : CellType), chromosome_sets cell = 1 ↔ cell = CellType.Sperm :=
by
  sorry

-- Additional facts to support the theorem
theorem zygote_has_two_sets : chromosome_sets CellType.Zygote = 2 :=
by
  sorry

theorem somatic_cell_has_two_sets : chromosome_sets CellType.SomaticCell = 2 :=
by
  sorry

theorem spermatogonium_is_somatic_cell :
  chromosome_sets CellType.Spermatogonium = chromosome_sets CellType.SomaticCell :=
by
  sorry

theorem sperm_formed_by_meiosis : chromosome_sets CellType.Sperm = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_only_sperm_has_one_set_zygote_has_two_sets_somatic_cell_has_two_sets_spermatogonium_is_somatic_cell_sperm_formed_by_meiosis_l3809_380918


namespace NUMINAMATH_CALUDE_rectangle_properties_l3809_380944

/-- Properties of a rectangle with specific dimensions --/
theorem rectangle_properties (w : ℝ) (h : w > 0) :
  let l := 4 * w
  let perimeter := 2 * l + 2 * w
  perimeter = 200 →
  (l * w = 1600 ∧ perimeter - (perimeter - 5) = 5) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_properties_l3809_380944


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l3809_380905

def in_quadrant_I_or_II (x y : ℝ) : Prop := (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)

theorem points_in_quadrants_I_and_II (x y : ℝ) :
  y > 3 * x → y > 6 - x → in_quadrant_I_or_II x y := by
  sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l3809_380905


namespace NUMINAMATH_CALUDE_fibonacci_inequality_l3809_380964

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (min (fibonacci n / fibonacci (n - 1)) (fibonacci (n + 1) / fibonacci n) < a / b ∧
   a / b < max (fibonacci n / fibonacci (n - 1)) (fibonacci (n + 1) / fibonacci n)) →
  b ≥ fibonacci (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_inequality_l3809_380964


namespace NUMINAMATH_CALUDE_function_difference_implies_m_value_l3809_380937

theorem function_difference_implies_m_value :
  ∀ (f g : ℝ → ℝ) (m : ℝ),
    (∀ x, f x = 4 * x^2 - 3 * x + 5) →
    (∀ x, g x = x^2 - m * x - 8) →
    f 5 - g 5 = 20 →
    m = -13.6 := by
  sorry

end NUMINAMATH_CALUDE_function_difference_implies_m_value_l3809_380937


namespace NUMINAMATH_CALUDE_tissue_with_mitotic_and_meiotic_cells_is_gonad_l3809_380988

structure Cell where
  chromosomeCount : ℕ

structure Tissue where
  cells : Set Cell

def isSomaticCell (c : Cell) : Prop := sorry

def isGermCell (c : Cell) (sc : Cell) : Prop :=
  isSomaticCell sc ∧ c.chromosomeCount = sc.chromosomeCount / 2

def containsMitoticCells (t : Tissue) : Prop :=
  ∃ c ∈ t.cells, isSomaticCell c

def containsMeioticCells (t : Tissue) : Prop :=
  ∃ c sc, c ∈ t.cells ∧ isGermCell c sc

def isGonad (t : Tissue) : Prop :=
  containsMitoticCells t ∧ containsMeioticCells t

theorem tissue_with_mitotic_and_meiotic_cells_is_gonad (t : Tissue) :
  containsMitoticCells t → containsMeioticCells t → isGonad t :=
by sorry

end NUMINAMATH_CALUDE_tissue_with_mitotic_and_meiotic_cells_is_gonad_l3809_380988


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l3809_380934

-- Define the function
def f (x : ℝ) : ℝ := x^(2/3)

-- State the theorem
theorem f_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l3809_380934


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l3809_380993

/-- The area of a square with one side on y = 8 and endpoints on y = x^2 + 4x + 3 is 36 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (8 = x₁^2 + 4*x₁ + 3) ∧
  (8 = x₂^2 + 4*x₂ + 3) ∧
  ((x₂ - x₁)^2 = 36) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l3809_380993


namespace NUMINAMATH_CALUDE_line_slope_relation_l3809_380938

/-- Theorem: For a straight line y = kx + b passing through points A(-3, y₁) and B(4, y₂),
    if k < 0, then y₁ > y₂. -/
theorem line_slope_relation (k b y₁ y₂ : ℝ) : 
  k < 0 → 
  y₁ = k * (-3) + b →
  y₂ = k * 4 + b →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_line_slope_relation_l3809_380938


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l3809_380927

theorem least_number_with_remainder (n : ℕ) : 
  (n % 6 = 4 ∧ n % 7 = 4 ∧ n % 9 = 4 ∧ n % 18 = 4) →
  (∀ m : ℕ, m < n → ¬(m % 6 = 4 ∧ m % 7 = 4 ∧ m % 9 = 4 ∧ m % 18 = 4)) →
  n = 130 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l3809_380927


namespace NUMINAMATH_CALUDE_estimate_sqrt_expression_l3809_380999

theorem estimate_sqrt_expression :
  7 < Real.sqrt 36 * Real.sqrt (1/2) + Real.sqrt 8 ∧
  Real.sqrt 36 * Real.sqrt (1/2) + Real.sqrt 8 < 8 :=
by sorry

end NUMINAMATH_CALUDE_estimate_sqrt_expression_l3809_380999


namespace NUMINAMATH_CALUDE_special_function_value_l3809_380947

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

theorem special_function_value :
  ∀ f : ℝ → ℝ, special_function f → f 1 = 2 → f (-3) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l3809_380947


namespace NUMINAMATH_CALUDE_cupboard_sale_percentage_below_cost_l3809_380916

def cost_price : ℕ := 3750
def additional_amount : ℕ := 1200
def profit_percentage : ℚ := 16 / 100

def selling_price_with_profit : ℚ := cost_price + profit_percentage * cost_price
def actual_selling_price : ℚ := selling_price_with_profit - additional_amount

theorem cupboard_sale_percentage_below_cost (cost_price : ℕ) (additional_amount : ℕ) 
  (profit_percentage : ℚ) (selling_price_with_profit : ℚ) (actual_selling_price : ℚ) :
  (cost_price - actual_selling_price) / cost_price = 16 / 100 :=
by sorry

end NUMINAMATH_CALUDE_cupboard_sale_percentage_below_cost_l3809_380916


namespace NUMINAMATH_CALUDE_functional_equation_l3809_380963

theorem functional_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x - f y) = 1 - x - y) →
  (∀ x : ℝ, f x = 1/2 - x) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_l3809_380963


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l3809_380907

open Real

theorem function_inequality_implies_m_bound (m : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 ℯ ∧ m * (x₀ - 1 / x₀) - 2 * log x₀ < -m / x₀) →
  m < 2 / ℯ := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l3809_380907


namespace NUMINAMATH_CALUDE_fourth_animal_is_sheep_l3809_380952

/-- Represents the different types of animals -/
inductive Animal
  | Horse
  | Cow
  | Pig
  | Sheep
  | Rabbit
  | Squirrel

/-- The sequence of animals entering the fence -/
def animalSequence : List Animal :=
  [Animal.Horse, Animal.Cow, Animal.Pig, Animal.Sheep, Animal.Rabbit, Animal.Squirrel]

/-- Theorem stating that the 4th animal in the sequence is a sheep -/
theorem fourth_animal_is_sheep :
  animalSequence[3] = Animal.Sheep := by sorry

end NUMINAMATH_CALUDE_fourth_animal_is_sheep_l3809_380952


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3809_380939

theorem quadratic_inequality_equivalence : 
  ∀ x : ℝ, x * (2 * x + 3) < -2 ↔ x ∈ Set.Ioo (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3809_380939


namespace NUMINAMATH_CALUDE_volume_of_rotated_composite_shape_l3809_380975

/-- The volume of a solid formed by rotating a composite shape about the x-axis -/
theorem volume_of_rotated_composite_shape (π : ℝ) :
  let lower_rectangle_height : ℝ := 4
  let lower_rectangle_width : ℝ := 1
  let upper_rectangle_height : ℝ := 1
  let upper_rectangle_width : ℝ := 5
  let volume_lower := π * lower_rectangle_height^2 * lower_rectangle_width
  let volume_upper := π * upper_rectangle_height^2 * upper_rectangle_width
  volume_lower + volume_upper = 21 * π := by
  sorry

end NUMINAMATH_CALUDE_volume_of_rotated_composite_shape_l3809_380975


namespace NUMINAMATH_CALUDE_comic_collection_equality_l3809_380922

/-- Kymbrea's initial comic book collection --/
def kymbrea_initial : ℕ := 50

/-- Kymbrea's monthly comic book addition rate --/
def kymbrea_rate : ℕ := 3

/-- LaShawn's initial comic book collection --/
def lashawn_initial : ℕ := 20

/-- LaShawn's monthly comic book addition rate --/
def lashawn_rate : ℕ := 7

/-- The number of months after which LaShawn's collection will be greater than or equal to Kymbrea's --/
def months_until_equal : ℕ := 8

theorem comic_collection_equality :
  ∀ m : ℕ, m < months_until_equal →
    (lashawn_initial + lashawn_rate * m < kymbrea_initial + kymbrea_rate * m) ∧
    (lashawn_initial + lashawn_rate * months_until_equal ≥ kymbrea_initial + kymbrea_rate * months_until_equal) :=
by sorry

end NUMINAMATH_CALUDE_comic_collection_equality_l3809_380922


namespace NUMINAMATH_CALUDE_pet_store_cages_l3809_380956

/-- Given a pet store with an initial number of puppies, some sold, and a fixed number per cage,
    calculate the number of cages needed for the remaining puppies. -/
theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
    (h1 : initial_puppies = 102)
    (h2 : sold_puppies = 21)
    (h3 : puppies_per_cage = 9)
    (h4 : sold_puppies < initial_puppies) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 9 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3809_380956


namespace NUMINAMATH_CALUDE_oakwood_academy_walking_students_l3809_380915

theorem oakwood_academy_walking_students (total : ℚ) :
  let bus : ℚ := 1 / 3
  let car : ℚ := 1 / 5
  let cycle : ℚ := 1 / 8
  let walk : ℚ := total - (bus + car + cycle)
  walk = 41 / 120 := by
  sorry

end NUMINAMATH_CALUDE_oakwood_academy_walking_students_l3809_380915


namespace NUMINAMATH_CALUDE_smallest_base_for_repeating_decimal_l3809_380983

/-- Represents a repeating decimal in base k -/
def RepeatingDecimal (k : ℕ) (n : ℕ) := (k : ℚ) ^ 2 / ((k : ℚ) ^ 2 - 1) * (4 * k + 1)

/-- The smallest integer k > 10 such that 17/85 has a repeating decimal representation of 0.414141... in base k -/
theorem smallest_base_for_repeating_decimal :
  ∃ (k : ℕ), k > 10 ∧ RepeatingDecimal k 2 = 17 / 85 ∧
  ∀ (m : ℕ), m > 10 ∧ m < k → RepeatingDecimal m 2 ≠ 17 / 85 := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_repeating_decimal_l3809_380983


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3809_380971

theorem triangle_perimeter : 
  ∀ (a b c : ℝ), 
    a = 10 ∧ b = 6 ∧ c = 7 → 
    a + b > c ∧ a + c > b ∧ b + c > a → 
    a + b + c = 23 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3809_380971


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3809_380962

theorem absolute_value_inequality (x a : ℝ) (h1 : |x - 4| + |x - 3| < a) (h2 : a > 0) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3809_380962


namespace NUMINAMATH_CALUDE_hexagonal_grid_path_theorem_l3809_380914

/-- Represents a point in the hexagonal grid -/
structure HexPoint where
  x : ℤ
  y : ℤ

/-- Represents a direction in the hexagonal grid -/
inductive HexDirection
  | Right
  | UpRight
  | UpLeft
  | Left
  | DownLeft
  | DownRight

/-- Represents a path in the hexagonal grid -/
def HexPath := List (HexPoint × HexDirection)

/-- Function to calculate the length of a path -/
def pathLength (path : HexPath) : ℕ := path.length

/-- Function to check if a path is valid in the hexagonal grid -/
def isValidPath (path : HexPath) : Prop := sorry

/-- Function to find the longest continuous segment in the same direction -/
def longestContinuousSegment (path : HexPath) : ℕ := sorry

/-- Theorem: In a hexagonal grid, if the shortest path between two points is 20 units,
    then there exists a continuous segment of at least 10 units in the same direction -/
theorem hexagonal_grid_path_theorem (A B : HexPoint) (path : HexPath) :
  isValidPath path →
  pathLength path = 20 →
  (∀ p : HexPath, isValidPath p → pathLength p ≥ 20) →
  longestContinuousSegment path ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_grid_path_theorem_l3809_380914


namespace NUMINAMATH_CALUDE_secretary_donuts_donut_problem_l3809_380984

theorem secretary_donuts (initial : ℕ) (bill_eaten : ℕ) (final : ℕ) : ℕ :=
  let remaining_after_bill := initial - bill_eaten
  let remaining_after_coworkers := final * 2
  let secretary_taken := remaining_after_bill - remaining_after_coworkers
  secretary_taken

theorem donut_problem :
  secretary_donuts 50 2 22 = 4 := by sorry

end NUMINAMATH_CALUDE_secretary_donuts_donut_problem_l3809_380984


namespace NUMINAMATH_CALUDE_simultaneous_divisibility_l3809_380906

theorem simultaneous_divisibility (x y : ℤ) :
  (17 ∣ (2 * x + 3 * y)) ↔ (17 ∣ (9 * x + 5 * y)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_divisibility_l3809_380906


namespace NUMINAMATH_CALUDE_decimal_to_fraction_035_l3809_380978

def decimal_to_fraction (d : ℚ) : ℕ × ℕ := sorry

theorem decimal_to_fraction_035 :
  (decimal_to_fraction 0.35).1 = 7 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_035_l3809_380978


namespace NUMINAMATH_CALUDE_opposite_of_neg_three_l3809_380928

/-- The opposite of a real number x is the number that, when added to x, yields zero. -/
def opposite (x : ℝ) : ℝ := -x

/-- The opposite of -3 is 3. -/
theorem opposite_of_neg_three : opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_three_l3809_380928


namespace NUMINAMATH_CALUDE_sum_of_exponents_l3809_380959

theorem sum_of_exponents (x y z : ℕ) 
  (h : 800670 = 8 * 10^x + 6 * 10^y + 7 * 10^z) : 
  x + y + z = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_exponents_l3809_380959


namespace NUMINAMATH_CALUDE_floor_sqrt_95_l3809_380997

theorem floor_sqrt_95 : ⌊Real.sqrt 95⌋ = 9 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_95_l3809_380997


namespace NUMINAMATH_CALUDE_fraction_multiplication_result_l3809_380969

theorem fraction_multiplication_result : (3 / 4 : ℚ) * (1 / 2 : ℚ) * (2 / 5 : ℚ) * 5000 = 750 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_result_l3809_380969


namespace NUMINAMATH_CALUDE_bus_meeting_time_l3809_380946

structure BusJourney where
  totalDistance : ℝ
  distanceToCountyTown : ℝ
  bus1DepartureTime : ℝ
  bus1ArrivalCountyTown : ℝ
  bus1StopTime : ℝ
  bus1ArrivalProvincialCapital : ℝ
  bus2DepartureTime : ℝ
  bus2Speed : ℝ

def meetingTime (j : BusJourney) : ℝ := sorry

theorem bus_meeting_time (j : BusJourney) 
  (h1 : j.totalDistance = 189)
  (h2 : j.distanceToCountyTown = 54)
  (h3 : j.bus1DepartureTime = 8.5)
  (h4 : j.bus1ArrivalCountyTown = 9.25)
  (h5 : j.bus1StopTime = 0.25)
  (h6 : j.bus1ArrivalProvincialCapital = 11)
  (h7 : j.bus2DepartureTime = 9)
  (h8 : j.bus2Speed = 60) :
  meetingTime j = 72 / 60 := by sorry

end NUMINAMATH_CALUDE_bus_meeting_time_l3809_380946


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l3809_380900

theorem no_solutions_in_interval (a : ℤ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 7, (x - 2 * (a : ℝ) + 1)^2 - 2*x + 4*(a : ℝ) - 10 ≠ 0) ↔ 
  (a ≤ -3 ∨ a ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l3809_380900


namespace NUMINAMATH_CALUDE_eight_team_tournament_l3809_380990

/-- The number of matches in a single-elimination tournament -/
def num_matches (n : ℕ) : ℕ := n - 1

/-- Theorem: A single-elimination tournament with 8 teams requires 7 matches -/
theorem eight_team_tournament : num_matches 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_eight_team_tournament_l3809_380990


namespace NUMINAMATH_CALUDE_sum_of_roots_l3809_380951

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3809_380951


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3809_380968

theorem simplify_and_evaluate (m n : ℝ) :
  (m + n)^2 - 2*m*(m + n) = n^2 - m^2 ∧
  (let m := 2; let n := -3; (m + n)^2 - 2*m*(m + n) = 5) :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3809_380968


namespace NUMINAMATH_CALUDE_min_framing_for_specific_picture_l3809_380935

/-- Calculate the minimum number of linear feet of framing needed for an enlarged picture with border -/
def min_framing_feet (original_width original_height enlargement_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  ⌈(perimeter_inches : ℚ) / 12⌉₊

/-- Theorem stating the minimum number of linear feet of framing needed for the specific picture -/
theorem min_framing_for_specific_picture :
  min_framing_feet 5 7 4 3 = 10 := by sorry

end NUMINAMATH_CALUDE_min_framing_for_specific_picture_l3809_380935


namespace NUMINAMATH_CALUDE_line_plane_relationship_l3809_380958

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Define the containment relation between a line and a plane
variable (contains : Plane → Line → Prop)

-- Define the parallelism relation between two lines
variable (parallel_lines : Line → Line → Prop)

-- Define the "on different planes" relation between two lines
variable (different_planes : Line → Line → Prop)

-- State the theorem
theorem line_plane_relationship (m n : Line) (α : Plane) 
  (h1 : parallel m α) (h2 : contains α n) :
  parallel_lines m n ∨ different_planes m n :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l3809_380958


namespace NUMINAMATH_CALUDE_count_possible_sums_l3809_380991

def bag_A : Finset ℕ := {0, 1, 3, 5}
def bag_B : Finset ℕ := {0, 2, 4, 6}

def possible_sums : Finset ℕ := (bag_A.product bag_B).image (fun p => p.1 + p.2)

theorem count_possible_sums : possible_sums.card = 10 := by
  sorry

end NUMINAMATH_CALUDE_count_possible_sums_l3809_380991


namespace NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l3809_380917

/-- Given a triangle with sides 13, 84, and 85, the shortest altitude has length 1092/85 -/
theorem shortest_altitude_right_triangle :
  ∀ (a b c h : ℝ),
  a = 13 ∧ b = 84 ∧ c = 85 →
  a^2 + b^2 = c^2 →
  h = (2 * (a * b / 2)) / c →
  h = 1092 / 85 := by
sorry

end NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l3809_380917


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3809_380976

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  first_positive : a 1 > 0
  sum_condition : a 1 + a 3 + a 5 = 6
  product_condition : a 1 * a 3 * a 5 = 0
  is_arithmetic : ∀ n m : ℕ+, a (n + m) - a n = m * (a 2 - a 1)

/-- The general term of the sequence -/
def general_term (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  5 - n

/-- The b_n term -/
def b (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  1 / (n * (seq.a n - 6))

/-- The sum of the first n terms of b_n -/
def S (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  -n / (n + 1)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ+, seq.a n = general_term seq n) ∧
  (∀ n : ℕ+, S seq n = -n / (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3809_380976


namespace NUMINAMATH_CALUDE_olympic_high_school_contest_l3809_380921

theorem olympic_high_school_contest (f s : ℕ) : 
  f > 0 → s > 0 → (2 * f) / 5 = (4 * s) / 5 → f = 2 * s := by
  sorry

#check olympic_high_school_contest

end NUMINAMATH_CALUDE_olympic_high_school_contest_l3809_380921


namespace NUMINAMATH_CALUDE_spinner_probability_l3809_380904

theorem spinner_probability : 
  ∀ (total_sections favorable_sections : ℕ),
    total_sections = 6 →
    favorable_sections = 2 →
    (favorable_sections : ℚ) / total_sections = 2 / 6 :=
by sorry

end NUMINAMATH_CALUDE_spinner_probability_l3809_380904


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3809_380932

theorem fraction_equivalence : 
  let n : ℚ := 13/2
  (4 + n) / (7 + n) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3809_380932


namespace NUMINAMATH_CALUDE_integer_triple_divisibility_l3809_380966

theorem integer_triple_divisibility :
  ∀ p q r : ℕ,
    1 < p → p < q → q < r →
    (p * q * r - 1) % ((p - 1) * (q - 1) * (r - 1)) = 0 →
    ((p = 2 ∧ q = 4 ∧ r = 8) ∨ (p = 3 ∧ q = 5 ∧ r = 15)) :=
by sorry

end NUMINAMATH_CALUDE_integer_triple_divisibility_l3809_380966


namespace NUMINAMATH_CALUDE_equal_diagonal_distances_l3809_380912

/-- Represents a cuboid with edge lengths a, b, and c. -/
structure Cuboid where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a pair of diagonals on adjacent faces of a cuboid. -/
inductive DiagonalPair
  | AB_AC
  | AB_BC
  | AC_BC

/-- Calculates the distance between a pair of diagonals on adjacent faces of a cuboid. -/
def diagonalDistance (cuboid : Cuboid) (pair : DiagonalPair) : ℝ :=
  sorry

/-- Theorem stating that the distances between diagonals of each pair of adjacent faces are equal
    for a cuboid with edge lengths 7, 14, and 21. -/
theorem equal_diagonal_distances (cuboid : Cuboid)
    (h1 : cuboid.a = 7)
    (h2 : cuboid.b = 14)
    (h3 : cuboid.c = 21) :
    ∀ p q : DiagonalPair, diagonalDistance cuboid p = diagonalDistance cuboid q :=
  sorry

end NUMINAMATH_CALUDE_equal_diagonal_distances_l3809_380912


namespace NUMINAMATH_CALUDE_sufficient_fabric_l3809_380929

/-- Represents the dimensions of a rectangular piece of fabric -/
structure FabricDimensions where
  length : ℕ
  width : ℕ

/-- Checks if a piece of fabric can be cut into at least n smaller pieces -/
def canCutInto (fabric : FabricDimensions) (piece : FabricDimensions) (n : ℕ) : Prop :=
  ∃ (l w : ℕ), 
    l * piece.length ≤ fabric.length ∧ 
    w * piece.width ≤ fabric.width ∧ 
    l * w ≥ n

theorem sufficient_fabric : 
  let fabric := FabricDimensions.mk 140 75
  let dress := FabricDimensions.mk 45 26
  canCutInto fabric dress 8 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_fabric_l3809_380929


namespace NUMINAMATH_CALUDE_parallel_lines_d_value_l3809_380987

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

/-- The slope of the first line -/
def slope₁ : ℝ := -3

/-- The slope of the second line -/
def slope₂ (d : ℝ) : ℝ := -6 * d

theorem parallel_lines_d_value :
  ∀ d : ℝ, parallel slope₁ (slope₂ d) → d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_d_value_l3809_380987


namespace NUMINAMATH_CALUDE_power_values_l3809_380994

theorem power_values (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) :
  a^(4*m + 3*n) = 432 ∧ a^(5*m - 2*n) = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_power_values_l3809_380994


namespace NUMINAMATH_CALUDE_spontaneous_reaction_l3809_380970

theorem spontaneous_reaction (ΔH ΔS : ℝ) (h1 : ΔH = -98.2) (h2 : ΔS = 70.5 / 1000) :
  ∀ T : ℝ, T ≥ 0 → ΔH - T * ΔS < 0 := by
sorry

end NUMINAMATH_CALUDE_spontaneous_reaction_l3809_380970


namespace NUMINAMATH_CALUDE_bicycle_sale_percentage_prove_bicycle_sale_percentage_l3809_380926

/-- The percentage of the suggested retail price that John paid for a bicycle -/
theorem bicycle_sale_percentage : ℝ → ℝ → ℝ → Prop :=
  fun wholesale_price suggested_retail_price johns_price =>
    suggested_retail_price = wholesale_price * (1 + 0.4) →
    johns_price = suggested_retail_price / 3 →
    johns_price / suggested_retail_price = 1 / 3

/-- Proof of the bicycle sale percentage theorem -/
theorem prove_bicycle_sale_percentage :
  ∀ (wholesale_price suggested_retail_price johns_price : ℝ),
    bicycle_sale_percentage wholesale_price suggested_retail_price johns_price := by
  sorry

#check prove_bicycle_sale_percentage

end NUMINAMATH_CALUDE_bicycle_sale_percentage_prove_bicycle_sale_percentage_l3809_380926


namespace NUMINAMATH_CALUDE_cost_of_660_candies_l3809_380902

/-- The cost of buying a given number of chocolate candies -/
def cost_of_candies (num_candies : ℕ) : ℚ :=
  let num_boxes : ℕ := (num_candies + 29) / 30
  let base_cost : ℚ := 7 * num_boxes
  if num_boxes > 20 then
    base_cost * (1 - 1/10)
  else
    base_cost

/-- Theorem: The cost of buying 660 chocolate candies is $138.60 -/
theorem cost_of_660_candies : cost_of_candies 660 = 1386/10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_660_candies_l3809_380902


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3809_380950

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h_fifth : a 5 = Nat.factorial 7)
  (h_eighth : a 8 = Nat.factorial 8) :
  a 1 = 315 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3809_380950


namespace NUMINAMATH_CALUDE_ferris_wheel_problem_l3809_380973

theorem ferris_wheel_problem (capacity : ℕ) (waiting : ℕ) (h1 : capacity = 56) (h2 : waiting = 92) :
  waiting - capacity = 36 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_problem_l3809_380973


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l3809_380941

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane
  (m : Line) (α β : Plane)
  (h1 : perp_planes α β)
  (h2 : perpendicular m β)
  (h3 : ¬ contains α m) :
  parallel m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l3809_380941


namespace NUMINAMATH_CALUDE_inverse_sum_mod_23_l3809_380924

theorem inverse_sum_mod_23 : 
  (((13⁻¹ : ZMod 23) + (17⁻¹ : ZMod 23) + (19⁻¹ : ZMod 23))⁻¹ : ZMod 23) = 8 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_23_l3809_380924


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_sum_l3809_380974

-- Define the arithmetic sequence a_n and its sum S_n
def a (n : ℕ) : ℝ := sorry
def S (n : ℕ) : ℝ := sorry

-- Define T_n as the sum of first n terms of 1/S_n
def T (n : ℕ) : ℝ := sorry

-- State the given conditions
axiom S_3_eq_15 : S 3 = 15
axiom a_3_plus_a_8 : a 3 + a 8 = 2 * a 5 + 2

-- State the theorem to be proved
theorem arithmetic_sequence_and_sum (n : ℕ) : 
  a n = 2 * n + 1 ∧ T n < 3/4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_sum_l3809_380974


namespace NUMINAMATH_CALUDE_parabola_segment_length_squared_l3809_380954

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = 3x^2 - 4x + 5 -/
def onParabola (p : Point) : Prop :=
  p.y = 3 * p.x^2 - 4 * p.x + 5

/-- The origin (0, 0) is the midpoint of two points -/
def originIsMidpoint (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = -q.y

/-- The square of the distance between two points -/
def squareDistance (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- The main theorem -/
theorem parabola_segment_length_squared :
  ∀ p q : Point,
  onParabola p → onParabola q → originIsMidpoint p q →
  squareDistance p q = 8900 / 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_segment_length_squared_l3809_380954


namespace NUMINAMATH_CALUDE_sqrt_300_simplification_l3809_380923

theorem sqrt_300_simplification : Real.sqrt 300 = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_300_simplification_l3809_380923


namespace NUMINAMATH_CALUDE_remainder_sum_powers_mod_seven_l3809_380933

theorem remainder_sum_powers_mod_seven :
  (9^7 + 8^8 + 7^9) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_powers_mod_seven_l3809_380933


namespace NUMINAMATH_CALUDE_total_eggs_calculation_l3809_380980

theorem total_eggs_calculation (eggs_per_omelet : ℕ) (num_people : ℕ) (omelets_per_person : ℕ)
  (h1 : eggs_per_omelet = 4)
  (h2 : num_people = 3)
  (h3 : omelets_per_person = 3) :
  eggs_per_omelet * num_people * omelets_per_person = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_calculation_l3809_380980


namespace NUMINAMATH_CALUDE_division_problem_l3809_380953

theorem division_problem : (144 / 6) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3809_380953


namespace NUMINAMATH_CALUDE_words_per_page_l3809_380977

theorem words_per_page (total_pages : Nat) (total_words_mod : Nat) (modulus : Nat) 
  (h1 : total_pages = 154)
  (h2 : total_words_mod = 145)
  (h3 : modulus = 221)
  (h4 : ∃ (words_per_page : Nat), words_per_page ≤ 120 ∧ 
        (total_pages * words_per_page) % modulus = total_words_mod) :
  ∃ (words_per_page : Nat), words_per_page = 96 ∧
    (total_pages * words_per_page) % modulus = total_words_mod := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_l3809_380977


namespace NUMINAMATH_CALUDE_abs_sum_eq_sum_abs_iff_product_nonneg_l3809_380913

theorem abs_sum_eq_sum_abs_iff_product_nonneg (x y : ℝ) :
  abs (x + y) = abs x + abs y ↔ x * y ≥ 0 := by sorry

end NUMINAMATH_CALUDE_abs_sum_eq_sum_abs_iff_product_nonneg_l3809_380913


namespace NUMINAMATH_CALUDE_tangent_problem_l3809_380936

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (π + α) = -1/3)
  (h2 : Real.tan (α + β) = (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α)) :
  (Real.tan (α + β) = 5/16) ∧ (Real.tan β = 31/43) := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l3809_380936


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l3809_380920

/-- Given a train of length 1400 m that crosses a tree in 100 sec,
    prove that it takes 150 sec to pass a platform of length 700 m. -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1400)
  (h2 : tree_crossing_time = 100)
  (h3 : platform_length = 700) :
  (train_length + platform_length) / (train_length / tree_crossing_time) = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l3809_380920


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3809_380961

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 0 → x ≠ 1 → x ≠ -1 →
  (-x^2 + 5*x - 6) / (x^3 - x) = 6 / x + (-7*x + 5) / (x^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3809_380961


namespace NUMINAMATH_CALUDE_line_equation_through_ellipse_points_l3809_380948

/-- The equation of a line passing through two points on an ellipse -/
theorem line_equation_through_ellipse_points 
  (A B : ℝ × ℝ) -- Two points on the ellipse
  (h_ellipse_A : (A.1^2 / 16) + (A.2^2 / 12) = 1) -- A is on the ellipse
  (h_ellipse_B : (B.1^2 / 16) + (B.2^2 / 12) = 1) -- B is on the ellipse
  (h_midpoint : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (2, 1)) -- Midpoint of AB is (2, 1)
  : ∃ (a b c : ℝ), a * A.1 + b * A.2 + c = 0 ∧ 
                    a * B.1 + b * B.2 + c = 0 ∧ 
                    (a, b, c) = (3, 2, -8) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_ellipse_points_l3809_380948


namespace NUMINAMATH_CALUDE_reshuffling_theorem_l3809_380979

def total_employees : ℕ := 10000

def current_proportions : List (String × ℚ) := [
  ("Senior Managers", 2/5),
  ("Junior Managers", 3/10),
  ("Engineers", 1/5),
  ("Marketing Team", 1/10)
]

def desired_proportions : List (String × ℚ) := [
  ("Senior Managers", 7/20),
  ("Junior Managers", 1/5),
  ("Engineers", 1/4),
  ("Marketing Team", 1/5)
]

def calculate_changes (current : List (String × ℚ)) (desired : List (String × ℚ)) (total : ℕ) : 
  List (String × ℤ) :=
  sorry

theorem reshuffling_theorem : 
  calculate_changes current_proportions desired_proportions total_employees = 
    [("Senior Managers", -500), 
     ("Junior Managers", -1000), 
     ("Engineers", 500), 
     ("Marketing Team", 1000)] :=
by sorry

end NUMINAMATH_CALUDE_reshuffling_theorem_l3809_380979


namespace NUMINAMATH_CALUDE_grandma_gift_amount_l3809_380908

/-- Calculates the amount grandma gave each person given the initial amount, expenses, and remaining amount. -/
theorem grandma_gift_amount
  (initial_amount : ℝ)
  (gasoline_cost : ℝ)
  (lunch_cost : ℝ)
  (gift_cost_per_person : ℝ)
  (num_people : ℕ)
  (remaining_amount : ℝ)
  (h1 : initial_amount = 50)
  (h2 : gasoline_cost = 8)
  (h3 : lunch_cost = 15.65)
  (h4 : gift_cost_per_person = 5)
  (h5 : num_people = 2)
  (h6 : remaining_amount = 36.35) :
  (remaining_amount - (initial_amount - (gasoline_cost + lunch_cost + gift_cost_per_person * num_people))) / num_people = 10 :=
by sorry

end NUMINAMATH_CALUDE_grandma_gift_amount_l3809_380908


namespace NUMINAMATH_CALUDE_sequences_theorem_l3809_380981

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n + 1

-- Define the sum S_n of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (n + 2)

-- Define the geometric sequence b_n
def b (n : ℕ) : ℚ := 3^n

-- Define T_n as the sum of the first n terms of 1/S_n
def T (n : ℕ) : ℚ := 3/4 - (2*n + 3) / (2 * (n+1) * (n+2))

-- State the theorem
theorem sequences_theorem (n : ℕ) : 
  (a n = 2 * n + 1) ∧ 
  (b n = 3^n) ∧ 
  (T n = 3/4 - (2*n + 3) / (2 * (n+1) * (n+2))) ∧
  (a 1 = b 1) ∧ 
  (a 4 = b 2) ∧ 
  (a 13 = b 3) :=
by sorry

end NUMINAMATH_CALUDE_sequences_theorem_l3809_380981


namespace NUMINAMATH_CALUDE_divisibility_problem_l3809_380903

theorem divisibility_problem (n : ℕ) (h : n = 856) :
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 8) = 24 * k₁ ∧ (n + 8) = 32 * k₂ ∧ (n + 8) = 36 * k₃ ∧ (n + 8) = 3 * k₄) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3809_380903


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l3809_380982

theorem cubic_sum_theorem (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l3809_380982


namespace NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l3809_380949

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 2 ≤ 5) ∨
               (5 = y - 2 ∧ x + 3 ≤ 5) ∨
               (x + 3 = y - 2 ∧ 5 ≤ x + 3)}

-- Define what it means for a set to be three rays with a common point
def is_three_rays_with_common_point (S : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, ∃ r₁ r₂ r₃ : Set (ℝ × ℝ),
    S = r₁ ∪ r₂ ∪ r₃ ∧
    r₁ ∩ r₂ = {p} ∧ r₁ ∩ r₃ = {p} ∧ r₂ ∩ r₃ = {p} ∧
    (∀ q ∈ r₁, ∃ t : ℝ, t ≥ 0 ∧ q = p + t • (0, -1)) ∧
    (∀ q ∈ r₂, ∃ t : ℝ, t ≥ 0 ∧ q = p + t • (-1, 0)) ∧
    (∀ q ∈ r₃, ∃ t : ℝ, t ≥ 0 ∧ q = p + t • (1, 1))

-- State the theorem
theorem T_is_three_rays_with_common_point : is_three_rays_with_common_point T := by
  sorry

end NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l3809_380949


namespace NUMINAMATH_CALUDE_railway_ticket_types_l3809_380957

/-- The number of stations on the railway --/
def num_stations : ℕ := 25

/-- The number of different types of tickets needed for a railway with n stations --/
def num_ticket_types (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of different types of tickets needed for a railway with 25 stations is 300 --/
theorem railway_ticket_types : num_ticket_types num_stations = 300 := by
  sorry

end NUMINAMATH_CALUDE_railway_ticket_types_l3809_380957


namespace NUMINAMATH_CALUDE_like_terms_sum_zero_l3809_380960

theorem like_terms_sum_zero (a b : ℝ) (m n : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (a^(m+1) * b^3 + (n-1) * a^2 * b^3 = 0) → (m = 1 ∧ n = 0) := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_zero_l3809_380960


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3809_380945

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3809_380945


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3809_380985

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ r s : ℝ, (25 - 10*r - r^2 = 0) ∧ (25 - 10*s - s^2 = 0) ∧ (r + s = -10)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3809_380985


namespace NUMINAMATH_CALUDE_coin_not_touching_lines_l3809_380955

/-- The probability that a randomly tossed coin doesn't touch parallel lines -/
theorem coin_not_touching_lines (a r : ℝ) (h : r < a) :
  let p := (a - r) / a
  0 ≤ p ∧ p ≤ 1 ∧ p = (a - r) / a :=
by sorry

end NUMINAMATH_CALUDE_coin_not_touching_lines_l3809_380955


namespace NUMINAMATH_CALUDE_similar_polygons_perimeter_ratio_l3809_380925

/-- If the ratio of the areas of two similar polygons is 4:9, then the ratio of their perimeters is 2:3 -/
theorem similar_polygons_perimeter_ratio (A B : ℝ) (P Q : ℝ) 
  (h_area : A / B = 4 / 9) (h_positive : A > 0 ∧ B > 0 ∧ P > 0 ∧ Q > 0)
  (h_area_perimeter : A / B = (P / Q)^2) : P / Q = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_similar_polygons_perimeter_ratio_l3809_380925


namespace NUMINAMATH_CALUDE_fort_soldiers_count_l3809_380967

/-- The initial number of soldiers in the fort -/
def initial_soldiers : ℕ := 480

/-- The number of additional soldiers joining the fort -/
def additional_soldiers : ℕ := 528

/-- The number of days provisions last with initial soldiers -/
def initial_days : ℕ := 30

/-- The number of days provisions last with additional soldiers -/
def new_days : ℕ := 25

/-- The daily consumption per soldier initially (in kg) -/
def initial_consumption : ℚ := 3

/-- The daily consumption per soldier after additional soldiers join (in kg) -/
def new_consumption : ℚ := 5/2

theorem fort_soldiers_count :
  initial_soldiers * initial_consumption * initial_days =
  (initial_soldiers + additional_soldiers) * new_consumption * new_days :=
sorry

end NUMINAMATH_CALUDE_fort_soldiers_count_l3809_380967


namespace NUMINAMATH_CALUDE_remaining_time_is_three_l3809_380943

/-- Represents the time needed to finish plowing a field with two tractors -/
def time_to_finish (time_a time_b worked_time : ℚ) : ℚ :=
  let rate_a : ℚ := 1 / time_a
  let rate_b : ℚ := 1 / time_b
  let remaining_work : ℚ := 1 - (rate_a * worked_time)
  let combined_rate : ℚ := rate_a + rate_b
  remaining_work / combined_rate

/-- Theorem stating that the remaining time to finish plowing is 3 hours -/
theorem remaining_time_is_three :
  time_to_finish 20 15 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_time_is_three_l3809_380943


namespace NUMINAMATH_CALUDE_part_one_part_two_l3809_380901

-- Define the sets A and B
def A : Set ℝ := {x | x < -3 ∨ x > 7}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Part (1)
theorem part_one (m : ℝ) : 
  (Set.univ \ A) ∪ B m = Set.univ \ A ↔ m ≤ 4 := by sorry

-- Part (2)
theorem part_two (m : ℝ) : 
  (∃ (a b : ℝ), (Set.univ \ A) ∩ B m = {x | a ≤ x ∧ x ≤ b} ∧ b - a ≥ 1) ↔ 
  (3 ≤ m ∧ m ≤ 5) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3809_380901


namespace NUMINAMATH_CALUDE_greatest_lower_bound_reciprocal_sum_l3809_380909

theorem greatest_lower_bound_reciprocal_sum (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  (1 / a + 1 / b ≥ 4) ∧ ∀ m > 4, ∃ a b, 0 < a ∧ 0 < b ∧ a + b = 1 ∧ 1 / a + 1 / b < m :=
sorry

end NUMINAMATH_CALUDE_greatest_lower_bound_reciprocal_sum_l3809_380909


namespace NUMINAMATH_CALUDE_smallest_multiple_thirty_two_satisfies_smallest_multiple_is_32_l3809_380911

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 900 * x % 1152 = 0 → x ≥ 32 := by
  sorry

theorem thirty_two_satisfies : 900 * 32 % 1152 = 0 := by
  sorry

theorem smallest_multiple_is_32 : 
  ∃ (x : ℕ), x > 0 ∧ 900 * x % 1152 = 0 ∧ ∀ (y : ℕ), y > 0 ∧ 900 * y % 1152 = 0 → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_thirty_two_satisfies_smallest_multiple_is_32_l3809_380911


namespace NUMINAMATH_CALUDE_tan_product_from_cos_sum_diff_l3809_380910

theorem tan_product_from_cos_sum_diff (α β : ℝ) 
  (h1 : Real.cos (α + β) = 2/3) 
  (h2 : Real.cos (α - β) = 1/3) : 
  Real.tan α * Real.tan β = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_from_cos_sum_diff_l3809_380910


namespace NUMINAMATH_CALUDE_charity_fundraising_l3809_380989

theorem charity_fundraising (donation_percentage : ℚ) (num_organizations : ℕ) (amount_per_org : ℚ) :
  donation_percentage = 80 / 100 →
  num_organizations = 8 →
  amount_per_org = 250 →
  (num_organizations : ℚ) * amount_per_org / donation_percentage = 2500 :=
by sorry

end NUMINAMATH_CALUDE_charity_fundraising_l3809_380989


namespace NUMINAMATH_CALUDE_solutions_absolute_value_equation_l3809_380940

theorem solutions_absolute_value_equation :
  (∀ x : ℝ, |x| = 1 ↔ x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_solutions_absolute_value_equation_l3809_380940


namespace NUMINAMATH_CALUDE_smallest_difference_fraction_l3809_380942

theorem smallest_difference_fraction :
  ∀ p q : ℕ, 
    0 < q → q < 1001 → 
    |123 / 1001 - (p : ℚ) / q| ≥ |123 / 1001 - 94 / 765| := by
  sorry

end NUMINAMATH_CALUDE_smallest_difference_fraction_l3809_380942
