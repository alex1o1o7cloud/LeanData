import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_solution_l219_21931

/-- Given nonzero real numbers c and d such that 2x^2 + cx + d = 0 has solutions 2c and 2d,
    prove that c = 1/2 and d = -5/8 -/
theorem quadratic_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0)
  (h : ∀ x, 2 * x^2 + c * x + d = 0 ↔ x = 2 * c ∨ x = 2 * d) :
  c = 1/2 ∧ d = -5/8 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l219_21931


namespace NUMINAMATH_CALUDE_smallest_nonzero_place_12000_l219_21942

/-- The smallest place value with a non-zero digit in 12000 is the hundreds place -/
theorem smallest_nonzero_place_12000 : 
  ∀ n : ℕ, n > 0 ∧ n < 1000 → (12000 / 10^n) % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonzero_place_12000_l219_21942


namespace NUMINAMATH_CALUDE_geometric_sequence_property_not_necessary_condition_l219_21988

/-- A sequence is geometric if the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem geometric_sequence_property :
  ∀ a b c d : ℝ,
  (∃ s : ℕ → ℝ, IsGeometricSequence s ∧ s 0 = a ∧ s 1 = b ∧ s 2 = c ∧ s 3 = d) →
  a * d = b * c :=
sorry

theorem not_necessary_condition :
  ∃ a b c d : ℝ, a * d = b * c ∧
  ¬(∃ s : ℕ → ℝ, IsGeometricSequence s ∧ s 0 = a ∧ s 1 = b ∧ s 2 = c ∧ s 3 = d) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_not_necessary_condition_l219_21988


namespace NUMINAMATH_CALUDE_dave_walking_probability_l219_21969

/-- Represents the number of gates in the airport terminal -/
def total_gates : ℕ := 15

/-- Represents the number of gates Dave can be assigned to -/
def dave_gates : ℕ := 10

/-- Represents the distance between adjacent gates in feet -/
def gate_distance : ℕ := 100

/-- Represents the maximum walking distance in feet -/
def max_walk_distance : ℕ := 300

/-- Calculates the number of valid gate combinations for Dave's initial and new gates -/
def total_combinations : ℕ := dave_gates * (dave_gates - 1)

/-- Calculates the number of valid gate combinations where Dave walks 300 feet or less -/
def valid_combinations : ℕ := 58

/-- The probability of Dave walking 300 feet or fewer to his new gate -/
def probability : ℚ := valid_combinations / total_combinations

theorem dave_walking_probability :
  probability = 29 / 45 := by sorry

end NUMINAMATH_CALUDE_dave_walking_probability_l219_21969


namespace NUMINAMATH_CALUDE_toms_journey_ratio_l219_21908

/-- Proves that the ratio of running time to swimming time is 1:2 given the conditions of Tom's journey --/
theorem toms_journey_ratio (swim_speed swim_time run_speed total_distance : ℝ) : 
  swim_speed = 2 →
  swim_time = 2 →
  run_speed = 4 * swim_speed →
  total_distance = 12 →
  total_distance = swim_speed * swim_time + run_speed * (total_distance - swim_speed * swim_time) / run_speed →
  (total_distance - swim_speed * swim_time) / run_speed / swim_time = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_toms_journey_ratio_l219_21908


namespace NUMINAMATH_CALUDE_cost_per_watt_hour_is_020_l219_21912

/-- Calculates the cost per watt-hour given the number of bulbs, wattage per bulb,
    number of days, and total monthly expense. -/
def cost_per_watt_hour (num_bulbs : ℕ) (watts_per_bulb : ℕ) (days : ℕ) (total_expense : ℚ) : ℚ :=
  total_expense / (num_bulbs * watts_per_bulb * days : ℚ)

/-- Theorem stating that the cost per watt-hour is $0.20 under the given conditions. -/
theorem cost_per_watt_hour_is_020 :
  cost_per_watt_hour 40 60 30 14400 = 1/5 := by sorry

end NUMINAMATH_CALUDE_cost_per_watt_hour_is_020_l219_21912


namespace NUMINAMATH_CALUDE_museum_pictures_l219_21906

theorem museum_pictures (zoo_pics : ℕ) (deleted_pics : ℕ) (remaining_pics : ℕ) (museum_pics : ℕ) : 
  zoo_pics = 15 → 
  deleted_pics = 31 → 
  remaining_pics = 2 → 
  zoo_pics + museum_pics - deleted_pics = remaining_pics → 
  museum_pics = 18 := by
sorry

end NUMINAMATH_CALUDE_museum_pictures_l219_21906


namespace NUMINAMATH_CALUDE_chord_length_line_circle_intersection_l219_21971

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length_line_circle_intersection : 
  ∃ (A B : ℝ × ℝ),
    (A.1 + A.2 = 2) ∧ 
    (B.1 + B.2 = 2) ∧ 
    (A.1^2 + A.2^2 = 4) ∧ 
    (B.1^2 + B.2^2 = 4) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_line_circle_intersection_l219_21971


namespace NUMINAMATH_CALUDE_alice_bracelet_profit_l219_21922

/-- Alice's bracelet sale profit calculation -/
theorem alice_bracelet_profit :
  ∀ (total_bracelets : ℕ) 
    (material_cost given_away price : ℚ),
  total_bracelets = 52 →
  material_cost = 3 →
  given_away = 8 →
  price = 1/4 →
  (total_bracelets - given_away : ℚ) * price - material_cost = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_alice_bracelet_profit_l219_21922


namespace NUMINAMATH_CALUDE_average_speed_calculation_l219_21987

/-- Calculate the average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_calculation (speed1 speed2 : ℝ) (h1 : speed1 = 90) (h2 : speed2 = 55) :
  (speed1 + speed2) / 2 = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l219_21987


namespace NUMINAMATH_CALUDE_pyramid_volume_scaling_l219_21977

/-- Given a pyramid with a rectangular base and initial volume,
    calculate the new volume after scaling its dimensions. -/
theorem pyramid_volume_scaling (l w h : ℝ) (V : ℝ) :
  V = (1 / 3) * l * w * h →
  V = 60 →
  (1 / 3) * (3 * l) * (2 * w) * (2 * h) = 720 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_scaling_l219_21977


namespace NUMINAMATH_CALUDE_longest_side_of_obtuse_consecutive_integer_triangle_l219_21990

-- Define a triangle with consecutive integer side lengths
def ConsecutiveIntegerSidedTriangle (a b c : ℕ) : Prop :=
  (b = a + 1) ∧ (c = b + 1) ∧ (a ≥ 1)

-- Define an obtuse triangle
def ObtuseTriangle (a b c : ℕ) : Prop :=
  (a^2 + b^2 < c^2) ∨ (a^2 + c^2 < b^2) ∨ (b^2 + c^2 < a^2)

theorem longest_side_of_obtuse_consecutive_integer_triangle :
  ∀ a b c : ℕ,
  ConsecutiveIntegerSidedTriangle a b c →
  ObtuseTriangle a b c →
  c = 4 :=
sorry

end NUMINAMATH_CALUDE_longest_side_of_obtuse_consecutive_integer_triangle_l219_21990


namespace NUMINAMATH_CALUDE_water_composition_ratio_l219_21918

theorem water_composition_ratio :
  ∀ (total_mass : ℝ) (hydrogen_mass : ℝ),
    total_mass = 117 →
    hydrogen_mass = 13 →
    (hydrogen_mass / (total_mass - hydrogen_mass) = 1 / 8) :=
by
  sorry

end NUMINAMATH_CALUDE_water_composition_ratio_l219_21918


namespace NUMINAMATH_CALUDE_disk_difference_l219_21953

/-- Given a bag of disks with blue, yellow, and green colors, prove the difference between green and blue disks -/
theorem disk_difference (total : ℕ) (blue_ratio yellow_ratio green_ratio : ℕ) : 
  total = 126 →
  blue_ratio = 3 →
  yellow_ratio = 7 →
  green_ratio = 8 →
  (green_ratio * (total / (blue_ratio + yellow_ratio + green_ratio))) - 
  (blue_ratio * (total / (blue_ratio + yellow_ratio + green_ratio))) = 35 := by
  sorry

end NUMINAMATH_CALUDE_disk_difference_l219_21953


namespace NUMINAMATH_CALUDE_profit_share_difference_theorem_l219_21954

/-- Calculates the difference between profit shares of two partners given their investments and a known profit share of the third partner. -/
def profit_share_difference (invest_a invest_b invest_c b_profit : ℕ) : ℕ :=
  let total_investment := invest_a + invest_b + invest_c
  let total_profit := b_profit * total_investment / invest_b
  let a_profit := total_profit * invest_a / total_investment
  let c_profit := total_profit * invest_c / total_investment
  c_profit - a_profit

/-- The difference between profit shares of a and c is 600 given their investments and b's profit share. -/
theorem profit_share_difference_theorem :
  profit_share_difference 8000 10000 12000 1500 = 600 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_theorem_l219_21954


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_7_l219_21930

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_7 : 
  units_digit (factorial_sum 7) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_7_l219_21930


namespace NUMINAMATH_CALUDE_wilsons_theorem_l219_21934

theorem wilsons_theorem (p : ℕ) (hp : Prime p) : (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l219_21934


namespace NUMINAMATH_CALUDE_min_value_expression_l219_21927

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) :
  x^2 + 8*x*y + 4*y^2 + 4*z^2 ≥ 384 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l219_21927


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_of_sum_l219_21994

/-- A function that returns true if a number has unique digits -/
def hasUniqueDigits (n : Nat) : Bool :=
  sorry

/-- A function that returns the sum of digits of a number -/
def sumOfDigits (n : Nat) : Nat :=
  sorry

/-- A function that checks if two numbers use all digits from 1 to 9 exactly once between them -/
def useAllDigitsOnce (x y : Nat) : Bool :=
  sorry

theorem smallest_sum_of_digits_of_sum (x y : Nat) : 
  x ≥ 100 ∧ x < 1000 ∧ 
  y ≥ 100 ∧ y < 1000 ∧ 
  hasUniqueDigits x ∧ 
  hasUniqueDigits y ∧ 
  useAllDigitsOnce x y ∧
  x + y < 1000 →
  ∃ (T : Nat), T = x + y ∧ sumOfDigits T ≥ 21 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_of_sum_l219_21994


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l219_21936

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if a seating arrangement is valid -/
def is_valid_seating (s : CircularSeating) : Prop :=
  s.seated_people > 0 ∧ 
  s.seated_people ≤ s.total_chairs ∧
  ∀ (new_seat : ℕ), new_seat < s.total_chairs → 
    ∃ (adjacent : ℕ), adjacent < s.total_chairs ∧ 
      (new_seat + 1) % s.total_chairs = adjacent ∨ 
      (new_seat + s.total_chairs - 1) % s.total_chairs = adjacent

/-- The main theorem to prove -/
theorem smallest_valid_seating :
  ∀ (s : CircularSeating), 
    s.total_chairs = 72 → 
    (is_valid_seating s ↔ s.seated_people ≥ 18) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l219_21936


namespace NUMINAMATH_CALUDE_guess_two_digit_number_l219_21938

/-- A two-digit number is between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem guess_two_digit_number (x : ℕ) (h : TwoDigitNumber x) :
  (2 * x + 5) * 5 = 715 → x = 69 := by
  sorry

end NUMINAMATH_CALUDE_guess_two_digit_number_l219_21938


namespace NUMINAMATH_CALUDE_tourist_cyclist_speed_problem_l219_21901

/-- Represents the problem of finding the maximum speed of a tourist and the corresponding speed of a cyclist --/
theorem tourist_cyclist_speed_problem 
  (distance : ℝ) 
  (min_cyclist_time : ℝ) 
  (cyclist_speed_increase : ℝ) 
  (meet_time : ℝ) :
  distance = 8 ∧ 
  min_cyclist_time = 0.5 ∧ 
  cyclist_speed_increase = 0.25 ∧
  meet_time = 1/6 →
  ∃ (tourist_speed cyclist_speed : ℝ),
    tourist_speed = 7 ∧
    cyclist_speed = 16 ∧
    (∀ x : ℕ, x > tourist_speed → 
      ¬(∃ y : ℝ, 
        distance / y ≥ min_cyclist_time ∧
        x * (distance / y + meet_time) + y * meet_time * (1 + cyclist_speed_increase) = distance)) :=
by sorry

end NUMINAMATH_CALUDE_tourist_cyclist_speed_problem_l219_21901


namespace NUMINAMATH_CALUDE_opposite_teal_is_violet_l219_21985

-- Define the colors
inductive Color
  | Blue
  | Orange
  | Yellow
  | Violet
  | Teal
  | Pink

-- Define a cube as a function from face positions to colors
def Cube := Fin 6 → Color

-- Define face positions
def top : Fin 6 := 0
def bottom : Fin 6 := 1
def left : Fin 6 := 2
def right : Fin 6 := 3
def front : Fin 6 := 4
def back : Fin 6 := 5

-- Define the theorem
theorem opposite_teal_is_violet (cube : Cube) :
  (∀ (view : Fin 3), cube top = Color.Violet) →
  (∀ (view : Fin 3), cube left = Color.Orange) →
  (cube front = Color.Blue ∨ cube front = Color.Yellow ∨ cube front = Color.Pink) →
  (∃ (face : Fin 6), cube face = Color.Teal) →
  (∀ (face1 face2 : Fin 6), face1 ≠ face2 → cube face1 ≠ cube face2) →
  (cube bottom = Color.Teal → cube top = Color.Violet) :=
by sorry

end NUMINAMATH_CALUDE_opposite_teal_is_violet_l219_21985


namespace NUMINAMATH_CALUDE_intersection_when_a_half_range_of_a_when_disjoint_l219_21937

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_when_a_half :
  A (1/2) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

theorem range_of_a_when_disjoint (a : ℝ) :
  (A a).Nonempty → (A a ∩ B = ∅) →
  (-2 < a ∧ a ≤ -1/2) ∨ (a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_half_range_of_a_when_disjoint_l219_21937


namespace NUMINAMATH_CALUDE_jordans_rectangle_width_l219_21910

-- Define the rectangle type
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the area function for rectangles
def area (r : Rectangle) : ℝ := r.length * r.width

-- State the theorem
theorem jordans_rectangle_width 
  (carol_rect : Rectangle)
  (jordan_rect : Rectangle)
  (h1 : carol_rect.length = 5)
  (h2 : carol_rect.width = 24)
  (h3 : jordan_rect.length = 2)
  (h4 : area carol_rect = area jordan_rect) :
  jordan_rect.width = 60 := by
  sorry

end NUMINAMATH_CALUDE_jordans_rectangle_width_l219_21910


namespace NUMINAMATH_CALUDE_functional_equation_solution_l219_21995

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + f y) = f x * f y) : 
  (∀ x : ℚ, f x = 0) ∨ (∀ x : ℚ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l219_21995


namespace NUMINAMATH_CALUDE_concert_ticket_price_l219_21983

theorem concert_ticket_price (num_tickets : ℕ) (total_spent : ℚ) (h1 : num_tickets = 8) (h2 : total_spent = 32) : 
  total_spent / num_tickets = 4 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l219_21983


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l219_21943

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_perpendicular_parallel 
  (m n : Line) (α β : Plane) : 
  (perpendicularLP m α ∧ perpendicularLP n β ∧ perpendicular m n → perpendicularPP α β) ∧
  (perpendicularLP m α ∧ parallelLP n β ∧ parallel m n → perpendicularPP α β) :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l219_21943


namespace NUMINAMATH_CALUDE_cans_collection_proof_l219_21980

/-- The number of cans collected on a given day -/
def cans_on_day (a b : ℚ) (d : ℕ) : ℚ := a * d^2 + b

theorem cans_collection_proof (a b : ℚ) :
  cans_on_day a b 1 = 4 ∧
  cans_on_day a b 2 = 9 ∧
  cans_on_day a b 3 = 14 →
  a = 5/3 ∧ b = 7/3 ∧ cans_on_day a b 7 = 84 := by
  sorry

end NUMINAMATH_CALUDE_cans_collection_proof_l219_21980


namespace NUMINAMATH_CALUDE_max_adjacent_squares_l219_21914

/-- A square with side length 1 -/
def UnitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

/-- Two squares are adjacent if they share at least one point on their boundaries -/
def Adjacent (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ (frontier s1) ∩ (frontier s2)

/-- Two squares are non-overlapping if their interiors are disjoint -/
def NonOverlapping (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  interior s1 ∩ interior s2 = ∅

/-- A configuration of squares adjacent to a given square -/
def AdjacentSquares (n : ℕ) : Prop :=
  ∃ (squares : Fin n → Set (ℝ × ℝ)),
    (∀ i, squares i = UnitSquare) ∧
    (∀ i, Adjacent (squares i) UnitSquare) ∧
    (∀ i j, i ≠ j → NonOverlapping (squares i) (squares j))

/-- The maximum number of non-overlapping unit squares that can be placed adjacent to a given unit square is 8 -/
theorem max_adjacent_squares :
  (∀ n, AdjacentSquares n → n ≤ 8) ∧ AdjacentSquares 8 := by sorry

end NUMINAMATH_CALUDE_max_adjacent_squares_l219_21914


namespace NUMINAMATH_CALUDE_number_of_paths_in_grid_l219_21951

-- Define the grid dimensions
def grid_width : ℕ := 7
def grid_height : ℕ := 6

-- Define the total number of steps
def total_steps : ℕ := grid_width + grid_height

-- Theorem statement
theorem number_of_paths_in_grid : 
  (Nat.choose total_steps grid_height : ℕ) = 1716 := by
  sorry

end NUMINAMATH_CALUDE_number_of_paths_in_grid_l219_21951


namespace NUMINAMATH_CALUDE_min_cost_halloween_bags_l219_21946

/-- Represents the cost calculation for Halloween goodie bags --/
def halloween_bags_cost (total_students : ℕ) (vampire_count : ℕ) (pumpkin_count : ℕ) 
  (pack_size : ℕ) (pack_cost : ℕ) (individual_cost : ℕ) : ℕ := 
  let vampire_packs := vampire_count / pack_size
  let vampire_individuals := vampire_count % pack_size
  let pumpkin_packs := pumpkin_count / pack_size
  let pumpkin_individuals := pumpkin_count % pack_size
  vampire_packs * pack_cost + vampire_individuals * individual_cost +
  pumpkin_packs * pack_cost + pumpkin_individuals * individual_cost

/-- Theorem stating the minimum cost for Halloween goodie bags --/
theorem min_cost_halloween_bags : 
  halloween_bags_cost 25 11 14 5 3 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_halloween_bags_l219_21946


namespace NUMINAMATH_CALUDE_stratified_sample_probability_l219_21949

/-- Represents the number of classes selected from each grade -/
structure GradeSelection where
  grade1 : Nat
  grade2 : Nat
  grade3 : Nat

/-- The probability of selecting two classes from the same grade in a stratified sample -/
def probability_same_grade (selection : GradeSelection) : Rat :=
  let total_combinations := (selection.grade1 + selection.grade2 + selection.grade3).choose 2
  let same_grade_combinations := selection.grade1.choose 2
  same_grade_combinations / total_combinations

theorem stratified_sample_probability 
  (selection : GradeSelection)
  (h_ratio : selection.grade1 = 3 ∧ selection.grade2 = 2 ∧ selection.grade3 = 1) :
  probability_same_grade selection = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_probability_l219_21949


namespace NUMINAMATH_CALUDE_prime_representation_l219_21923

theorem prime_representation (p : ℕ) (hp : Prime p) (hp2 : p > 2) :
  (p % 8 = 1 ↔ ∃ x y : ℤ, p = x^2 + 16*y^2) ∧
  (p % 8 = 5 ↔ ∃ x y : ℤ, p = 4*x^2 + 4*x*y + 5*y^2) :=
by sorry

end NUMINAMATH_CALUDE_prime_representation_l219_21923


namespace NUMINAMATH_CALUDE_tamika_always_wins_l219_21903

def tamika_set : Finset ℕ := {11, 12, 13}
def carlos_set : Finset ℕ := {4, 6, 7}

theorem tamika_always_wins :
  ∀ (a b : ℕ) (c d : ℕ),
    a ∈ tamika_set → b ∈ tamika_set → a ≠ b →
    c ∈ carlos_set → d ∈ carlos_set → c ≠ d →
    a * b > c * d := by
  sorry

#check tamika_always_wins

end NUMINAMATH_CALUDE_tamika_always_wins_l219_21903


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l219_21970

theorem complex_product_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*Complex.I
  let z₂ : ℂ := 2 + Complex.I
  (z₁ * z₂).re = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l219_21970


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l219_21917

theorem algebraic_expression_value (a b : ℝ) (h : a - 2*b + 3 = 0) :
  5 + 2*b - a = 8 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l219_21917


namespace NUMINAMATH_CALUDE_doubly_underlined_count_l219_21950

def count_doubly_underlined (n : ℕ) : ℕ :=
  let multiples_of_6_not_4 := (n / 6 + 1) / 2
  let multiples_of_4_not_3 := 2 * (n / 4 + 1) / 3
  multiples_of_6_not_4 + multiples_of_4_not_3

theorem doubly_underlined_count :
  count_doubly_underlined 2016 = 504 := by
  sorry

end NUMINAMATH_CALUDE_doubly_underlined_count_l219_21950


namespace NUMINAMATH_CALUDE_vector_inequality_l219_21957

variable (V : Type*) [NormedAddCommGroup V]

theorem vector_inequality (v w : V) : 
  ‖v‖ + ‖w‖ ≤ ‖v + w‖ + ‖v - w‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_inequality_l219_21957


namespace NUMINAMATH_CALUDE_hyperbola_foci_l219_21982

/-- The hyperbola equation -/
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (1 + k^2) - y^2 / (8 - k^2) = 1

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(-3, 0), (3, 0)}

theorem hyperbola_foci (k : ℝ) (h : 1 + k^2 > 0) :
  ∃ (x y : ℝ), hyperbola_equation x y k →
  (x, y) ∈ foci_coordinates :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l219_21982


namespace NUMINAMATH_CALUDE_pet_shop_theorem_l219_21976

def pet_shop_problem (parakeet_cost : ℕ) : Prop :=
  let puppy_cost := 3 * parakeet_cost
  let kitten_cost := 2 * parakeet_cost
  let total_cost := 2 * puppy_cost + 2 * kitten_cost + 3 * parakeet_cost
  parakeet_cost = 10 → total_cost = 130

theorem pet_shop_theorem : pet_shop_problem 10 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_theorem_l219_21976


namespace NUMINAMATH_CALUDE_eva_process_terminates_l219_21929

/-- Represents a deck of cards -/
def Deck := List Nat

/-- Flips the first n cards in the deck -/
def flipCards (n : Nat) (deck : Deck) : Deck :=
  (deck.take n).reverse ++ deck.drop n

/-- Performs one step of Eva's operation -/
def evaStep (deck : Deck) : Deck :=
  match deck with
  | [] => []
  | k :: rest => flipCards k deck

/-- Predicate to check if the process has terminated -/
def isTerminated (deck : Deck) : Prop :=
  match deck with
  | 1 :: _ => True
  | _ => False

/-- Theorem stating that Eva's process always terminates -/
theorem eva_process_terminates (initial_deck : Deck) 
  (h_valid : initial_deck.length = 100 ∧ initial_deck.toFinset = Finset.range 100) :
  ∃ (n : Nat), isTerminated (n.iterate evaStep initial_deck) := by
  sorry

end NUMINAMATH_CALUDE_eva_process_terminates_l219_21929


namespace NUMINAMATH_CALUDE_ladder_problem_l219_21933

theorem ladder_problem (ladder_length height_on_wall : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height_on_wall = 12) :
  ∃ (distance_from_wall : ℝ), 
    distance_from_wall ^ 2 + height_on_wall ^ 2 = ladder_length ^ 2 ∧ 
    distance_from_wall = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l219_21933


namespace NUMINAMATH_CALUDE_bottles_ratio_l219_21935

/-- The number of bottles Paul drinks per day -/
def paul_bottles : ℕ := 3

/-- The number of bottles Donald drinks per day -/
def donald_bottles : ℕ := 9

/-- Donald drinks more than twice the number of bottles Paul drinks -/
axiom donald_drinks_more : donald_bottles > 2 * paul_bottles

/-- The ratio of bottles Donald drinks to bottles Paul drinks is 3:1 -/
theorem bottles_ratio : (donald_bottles : ℚ) / paul_bottles = 3 := by
  sorry

end NUMINAMATH_CALUDE_bottles_ratio_l219_21935


namespace NUMINAMATH_CALUDE_f_at_7_equals_3_l219_21932

-- Define the function f
def f (p q : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x + 5

-- State the theorem
theorem f_at_7_equals_3 (p q b : ℝ) :
  (f p q (-7) = Real.sqrt 2 * b + 1) →
  f p q 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_at_7_equals_3_l219_21932


namespace NUMINAMATH_CALUDE_sports_club_intersection_l219_21925

/-- Given a sports club with the following properties:
  - There are 30 total members
  - 16 members play badminton
  - 19 members play tennis
  - 2 members play neither badminton nor tennis
  Prove that 7 members play both badminton and tennis -/
theorem sports_club_intersection (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 30 →
  badminton = 16 →
  tennis = 19 →
  neither = 2 →
  badminton + tennis - (total - neither) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_intersection_l219_21925


namespace NUMINAMATH_CALUDE_solve_equation_l219_21958

theorem solve_equation (x : ℚ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l219_21958


namespace NUMINAMATH_CALUDE_transform_f_to_g_l219_21915

def f (x : ℝ) : ℝ := 4 * (x - 3)^2 + 4
def g (x : ℝ) : ℝ := 4 * (x + 3)^2 - 4

theorem transform_f_to_g : 
  ∀ x : ℝ, g x = f (x + 6) - 8 := by sorry

end NUMINAMATH_CALUDE_transform_f_to_g_l219_21915


namespace NUMINAMATH_CALUDE_stating_wrapping_paper_area_theorem_l219_21945

/-- Represents a rectangular box. -/
structure Box where
  a : ℝ
  b : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  h_pos : 0 < h

/-- Calculates the area of the square wrapping paper needed for a given box. -/
def wrappingPaperArea (box : Box) : ℝ :=
  (box.a + 2 * box.h) ^ 2

/-- 
Theorem stating that the area of the square wrapping paper for a rectangular box
with base dimensions a × b and height h, wrapped as described in the problem,
is (a + 2h)².
-/
theorem wrapping_paper_area_theorem (box : Box) :
  wrappingPaperArea box = (box.a + 2 * box.h) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_stating_wrapping_paper_area_theorem_l219_21945


namespace NUMINAMATH_CALUDE_probability_two_red_correct_l219_21968

def bag_red_balls : ℕ := 9
def bag_white_balls : ℕ := 3
def total_balls : ℕ := bag_red_balls + bag_white_balls
def drawn_balls : ℕ := 4

def probability_two_red : ℚ :=
  (Nat.choose bag_red_balls 2 * Nat.choose bag_white_balls 2) / Nat.choose total_balls drawn_balls

theorem probability_two_red_correct :
  probability_two_red = (Nat.choose bag_red_balls 2 * Nat.choose bag_white_balls 2) / Nat.choose total_balls drawn_balls :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_correct_l219_21968


namespace NUMINAMATH_CALUDE_ellipse_chord_slopes_product_l219_21902

/-- Theorem: Product of slopes for chord through center of ellipse -/
theorem ellipse_chord_slopes_product (a b x₀ y₀ x₁ y₁ : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hP : x₀^2 / a^2 + y₀^2 / b^2 = 1)  -- P is on the ellipse
  (hP₁ : x₁^2 / a^2 + y₁^2 / b^2 = 1)  -- P₁ is on the ellipse
  (hP₂ : (-x₁)^2 / a^2 + (-y₁)^2 / b^2 = 1)  -- P₂ is on the ellipse
  (k₁ : ℝ) (hk₁ : k₁ = (y₀ - y₁) / (x₀ - x₁))  -- Slope of PP₁
  (k₂ : ℝ) (hk₂ : k₂ = (y₀ - (-y₁)) / (x₀ - (-x₁)))  -- Slope of PP₂
  : k₁ * k₂ = -b^2 / a^2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_chord_slopes_product_l219_21902


namespace NUMINAMATH_CALUDE_expression_evaluation_l219_21947

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  (x + 2*y)^2 - (x + y)*(x - y) = -11/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l219_21947


namespace NUMINAMATH_CALUDE_player_one_wins_with_2023_coins_l219_21952

/-- Represents the possible moves for each player -/
inductive Move
| three : Move
| five : Move
| two : Move
| four : Move

/-- Represents a player in the game -/
inductive Player
| one : Player
| two : Player

/-- The game state -/
structure GameState where
  coins : ℕ
  currentPlayer : Player

/-- Determines if a move is valid for a given player -/
def validMove (player : Player) (move : Move) : Bool :=
  match player, move with
  | Player.one, Move.three => true
  | Player.one, Move.five => true
  | Player.two, Move.two => true
  | Player.two, Move.four => true
  | _, _ => false

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : Option GameState :=
  if validMove state.currentPlayer move then
    let newCoins := match move with
      | Move.three => state.coins - 3
      | Move.five => state.coins - 5
      | Move.two => state.coins - 2
      | Move.four => state.coins - 4
    let newPlayer := match state.currentPlayer with
      | Player.one => Player.two
      | Player.two => Player.one
    some { coins := newCoins, currentPlayer := newPlayer }
  else
    none

/-- Determines if a player has a winning strategy from a given game state -/
def hasWinningStrategy (state : GameState) : Prop :=
  sorry

/-- The main theorem: Player 1 has a winning strategy when starting with 2023 coins -/
theorem player_one_wins_with_2023_coins :
  hasWinningStrategy { coins := 2023, currentPlayer := Player.one } :=
  sorry

end NUMINAMATH_CALUDE_player_one_wins_with_2023_coins_l219_21952


namespace NUMINAMATH_CALUDE_slope_range_l219_21992

theorem slope_range (α : Real) (h : π/3 < α ∧ α < 5*π/6) :
  ∃ k : Real, (k < -Real.sqrt 3 / 3 ∨ k > Real.sqrt 3) ∧ k = Real.tan α :=
by
  sorry

end NUMINAMATH_CALUDE_slope_range_l219_21992


namespace NUMINAMATH_CALUDE_vector_AB_after_translation_l219_21960

def point_A : ℝ × ℝ := (3, 7)
def point_B : ℝ × ℝ := (5, 2)
def vector_a : ℝ × ℝ := (1, 2)

def vector_AB : ℝ × ℝ := (point_B.1 - point_A.1, point_B.2 - point_A.2)

theorem vector_AB_after_translation :
  vector_AB = (2, -5) := by sorry

end NUMINAMATH_CALUDE_vector_AB_after_translation_l219_21960


namespace NUMINAMATH_CALUDE_tan_sum_product_fifteen_thirty_l219_21966

theorem tan_sum_product_fifteen_thirty : 
  Real.tan (15 * π / 180) + Real.tan (30 * π / 180) + Real.tan (15 * π / 180) * Real.tan (30 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_fifteen_thirty_l219_21966


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l219_21904

theorem quadratic_inequality_solution_set (a b : ℝ) : 
  (∀ x, x^2 - (a+1)*x + b ≤ 0 ↔ -4 ≤ x ∧ x ≤ 3) → a + b = -14 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l219_21904


namespace NUMINAMATH_CALUDE_dice_roll_probability_l219_21964

-- Define a dice roll
def DiceRoll : Type := Fin 6

-- Define a point as a pair of dice rolls
def Point : Type := DiceRoll × DiceRoll

-- Define the condition for a point to be inside the circle
def InsideCircle (p : Point) : Prop :=
  (p.1.val + 1)^2 + (p.2.val + 1)^2 < 17

-- Define the total number of possible outcomes
def TotalOutcomes : Nat := 36

-- Define the number of favorable outcomes
def FavorableOutcomes : Nat := 8

-- Theorem statement
theorem dice_roll_probability :
  (FavorableOutcomes : ℚ) / TotalOutcomes = 2 / 9 :=
sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l219_21964


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l219_21979

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l219_21979


namespace NUMINAMATH_CALUDE_neither_a_nor_b_probability_l219_21984

def prob_a : ℝ := 0.20
def prob_b : ℝ := 0.40
def prob_a_and_b : ℝ := 0.15

theorem neither_a_nor_b_probability :
  1 - (prob_a + prob_b - prob_a_and_b) = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_neither_a_nor_b_probability_l219_21984


namespace NUMINAMATH_CALUDE_original_paint_intensity_l219_21948

/-- 
Given a paint mixture where 20% of the original paint is replaced with a 25% solution,
resulting in a mixture with 45% intensity, prove that the original paint intensity was 50%.
-/
theorem original_paint_intensity 
  (original_intensity : ℝ) 
  (replaced_fraction : ℝ) 
  (replacement_solution_intensity : ℝ) 
  (final_intensity : ℝ) : 
  replaced_fraction = 0.2 →
  replacement_solution_intensity = 25 →
  final_intensity = 45 →
  (1 - replaced_fraction) * original_intensity + 
    replaced_fraction * replacement_solution_intensity = final_intensity →
  original_intensity = 50 := by
sorry

end NUMINAMATH_CALUDE_original_paint_intensity_l219_21948


namespace NUMINAMATH_CALUDE_square_diagonals_sum_l219_21916

theorem square_diagonals_sum (x y : ℝ) (h1 : x^2 + y^2 = 145) (h2 : x^2 - y^2 = 85) :
  x * Real.sqrt 2 + y * Real.sqrt 2 = Real.sqrt 230 + Real.sqrt 60 := by
  sorry

#check square_diagonals_sum

end NUMINAMATH_CALUDE_square_diagonals_sum_l219_21916


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a4_l219_21921

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_a4 (seq : ArithmeticSequence) 
    (h1 : seq.S 6 = 24) (h2 : seq.S 9 = 63) : seq.a 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a4_l219_21921


namespace NUMINAMATH_CALUDE_triangle_altitude_segment_l219_21920

theorem triangle_altitude_segment (a b c h x : ℝ) : 
  a = 40 → b = 90 → c = 100 → 
  a^2 = x^2 + h^2 → 
  b^2 = (c - x)^2 + h^2 → 
  c - x = 82.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_segment_l219_21920


namespace NUMINAMATH_CALUDE_exponent_calculation_l219_21991

theorem exponent_calculation (a : ℝ) : a^3 * (a^3)^2 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l219_21991


namespace NUMINAMATH_CALUDE_correct_sample_size_l219_21996

/-- Given a population with total students and girls, and a sample size,
    calculate the number of girls in the sample using stratified sampling. -/
def girlsInSample (totalStudents girls sampleSize : ℕ) : ℕ :=
  (girls * sampleSize) / totalStudents

/-- Theorem stating that for the given population and sample size,
    the number of girls in the sample should be 20. -/
theorem correct_sample_size :
  girlsInSample 30000 4000 150 = 20 := by
  sorry

end NUMINAMATH_CALUDE_correct_sample_size_l219_21996


namespace NUMINAMATH_CALUDE_find_x_value_l219_21965

theorem find_x_value (x y z : ℝ) 
  (h1 : x ≠ 0)
  (h2 : x / 3 = z + 2 * y^2)
  (h3 : x / 6 = 3 * z - y) :
  x = 168 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l219_21965


namespace NUMINAMATH_CALUDE_second_strategy_more_economical_l219_21940

/-- Proves that the second purchasing strategy (constant money spent) is more economical than
    the first strategy (constant quantity purchased) for two purchases of the same item. -/
theorem second_strategy_more_economical (p₁ p₂ x y : ℝ) 
    (hp₁ : p₁ > 0) (hp₂ : p₂ > 0) (hx : x > 0) (hy : y > 0) :
  (2 * p₁ * p₂) / (p₁ + p₂) ≤ (p₁ + p₂) / 2 := by
  sorry

#check second_strategy_more_economical

end NUMINAMATH_CALUDE_second_strategy_more_economical_l219_21940


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l219_21967

/-- A line passing through (2, 1) with equal intercepts on x and y axes -/
structure EqualInterceptLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (2, 1)
  point_condition : 1 = 2 * m + b
  -- The line has equal intercepts on x and y axes
  equal_intercepts : (m ≠ -1 → -b / (1 + m) = -b / m) ∧ (m = -1 → b = 0)

/-- The equation of the line is either x+y-3=0 or y = 1/2x -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = -1 ∧ l.b = 3) ∨ (l.m = 1/2 ∧ l.b = 0) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l219_21967


namespace NUMINAMATH_CALUDE_polynomial_simplification_l219_21986

theorem polynomial_simplification (x : ℝ) :
  3 + 5*x - 7*x^2 - 9 + 11*x - 13*x^2 + 15 - 17*x + 19*x^2 = 9 - x - x^2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l219_21986


namespace NUMINAMATH_CALUDE_sunday_school_average_class_size_l219_21924

/-- Represents the number of students in each age group -/
structure AgeGroups where
  three_year_olds : Nat
  four_year_olds : Nat
  five_year_olds : Nat
  six_year_olds : Nat
  seven_year_olds : Nat
  eight_year_olds : Nat

/-- Calculates the average class size given the age groups -/
def averageClassSize (groups : AgeGroups) : Rat :=
  let class1 := groups.three_year_olds + groups.four_year_olds
  let class2 := groups.five_year_olds + groups.six_year_olds
  let class3 := groups.seven_year_olds + groups.eight_year_olds
  let totalStudents := class1 + class2 + class3
  (totalStudents : Rat) / 3

/-- The specific age groups given in the problem -/
def sundaySchoolGroups : AgeGroups := {
  three_year_olds := 13,
  four_year_olds := 20,
  five_year_olds := 15,
  six_year_olds := 22,
  seven_year_olds := 18,
  eight_year_olds := 25
}

theorem sunday_school_average_class_size :
  averageClassSize sundaySchoolGroups = 113 / 3 := by
  sorry

#eval averageClassSize sundaySchoolGroups

end NUMINAMATH_CALUDE_sunday_school_average_class_size_l219_21924


namespace NUMINAMATH_CALUDE_moray_eel_eats_twenty_l219_21959

/-- The number of guppies eaten by a moray eel per day, given the total number of guppies needed,
    the number of betta fish, and the number of guppies eaten by each betta fish per day. -/
def moray_eel_guppies (total_guppies : ℕ) (num_betta : ℕ) (betta_guppies : ℕ) : ℕ :=
  total_guppies - (num_betta * betta_guppies)

/-- Theorem stating that the number of guppies eaten by the moray eel is 20,
    given the conditions in the problem. -/
theorem moray_eel_eats_twenty :
  moray_eel_guppies 55 5 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_moray_eel_eats_twenty_l219_21959


namespace NUMINAMATH_CALUDE_unique_prime_squared_plus_minus_six_prime_l219_21997

theorem unique_prime_squared_plus_minus_six_prime :
  ∃! p : ℕ, Nat.Prime p ∧ Nat.Prime (p^2 - 6) ∧ Nat.Prime (p^2 + 6) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_squared_plus_minus_six_prime_l219_21997


namespace NUMINAMATH_CALUDE_solve_for_y_l219_21962

theorem solve_for_y (x y : ℤ) (h1 : x + y = 290) (h2 : x - y = 200) : y = 45 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l219_21962


namespace NUMINAMATH_CALUDE_carries_box_capacity_l219_21955

/-- Represents a rectangular box with height, width, and length -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box -/
def Box.volume (b : Box) : ℝ := b.height * b.width * b.length

/-- Represents the number of jellybeans a box can hold -/
def jellybeanCapacity (b : Box) (density : ℝ) : ℝ := b.volume * density

/-- Theorem: Carrie's box capacity given Bert's box capacity -/
theorem carries_box_capacity
  (bert_box : Box)
  (bert_capacity : ℝ)
  (density : ℝ)
  (h1 : jellybeanCapacity bert_box density = bert_capacity)
  (h2 : bert_capacity = 150)
  (carrie_box : Box)
  (h3 : carrie_box.height = 3 * bert_box.height)
  (h4 : carrie_box.width = 2 * bert_box.width)
  (h5 : carrie_box.length = 4 * bert_box.length) :
  jellybeanCapacity carrie_box density = 3600 := by
  sorry

end NUMINAMATH_CALUDE_carries_box_capacity_l219_21955


namespace NUMINAMATH_CALUDE_perpendicular_line_correct_parallel_lines_correct_l219_21928

-- Define the given line l
def line_l (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define point A
def point_A : ℝ × ℝ := (3, 2)

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := x + 2 * y - 7 = 0

-- Define the parallel lines
def parallel_line_1 (x y : ℝ) : Prop := 2 * x - y + 6 = 0
def parallel_line_2 (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Theorem for the perpendicular line
theorem perpendicular_line_correct :
  (perp_line point_A.1 point_A.2) ∧
  (∀ x y : ℝ, line_l x y → (x - point_A.1) * 1 + (y - point_A.2) * 2 = 0) :=
sorry

-- Theorem for the parallel lines
theorem parallel_lines_correct :
  (∀ x y : ℝ, (parallel_line_1 x y ∨ parallel_line_2 x y) →
    (abs (6 - 1) / Real.sqrt (2^2 + 1) = Real.sqrt 5 ∨
     abs (-4 - 1) / Real.sqrt (2^2 + 1) = Real.sqrt 5)) ∧
  (∀ x y : ℝ, line_l x y → (2 * 1 + 1 * 1 = 2 * 1 + 1 * 1)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_correct_parallel_lines_correct_l219_21928


namespace NUMINAMATH_CALUDE_fourth_term_equals_eleven_l219_21981

/-- Given a sequence {aₙ} where Sₙ = 2n² - 3n, prove that a₄ = 11 -/
theorem fourth_term_equals_eleven (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, S n = 2 * n^2 - 3 * n) →
  (∀ n, a n = S n - S (n-1)) →
  a 4 = 11 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_equals_eleven_l219_21981


namespace NUMINAMATH_CALUDE_exists_A_all_A_digit_numbers_A_minus_1_expressible_l219_21956

/-- Represents the concatenation operation -/
def concatenate (a b : ℕ) : ℕ := sorry

/-- Checks if a number is m-expressible -/
def is_m_expressible (n m : ℕ) : Prop := sorry

/-- The main theorem to be proved -/
theorem exists_A_all_A_digit_numbers_A_minus_1_expressible :
  ∃ A : ℕ, ∀ n : ℕ, (10^(A-1) ≤ n ∧ n < 10^A) → is_m_expressible n (A-1) := by
  sorry

end NUMINAMATH_CALUDE_exists_A_all_A_digit_numbers_A_minus_1_expressible_l219_21956


namespace NUMINAMATH_CALUDE_soda_count_l219_21913

/-- Proves that given 2 sandwiches at $2.49 each and some sodas at $1.87 each,
    if the total cost is $12.46, then the number of sodas purchased is 4. -/
theorem soda_count (sandwich_cost soda_cost total_cost : ℚ) (sandwich_count : ℕ) :
  sandwich_cost = 249/100 →
  soda_cost = 187/100 →
  total_cost = 1246/100 →
  sandwich_count = 2 →
  ∃ (soda_count : ℕ), soda_count = 4 ∧
    sandwich_count * sandwich_cost + soda_count * soda_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_soda_count_l219_21913


namespace NUMINAMATH_CALUDE_real_part_of_2_minus_i_l219_21978

theorem real_part_of_2_minus_i : Complex.re (2 - Complex.I) = 2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_2_minus_i_l219_21978


namespace NUMINAMATH_CALUDE_sum_equals_rounded_sum_jo_equals_alex_sum_l219_21999

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def sum_to_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sum_rounded_to_five (n : ℕ) : ℕ :=
  (List.range n).map round_to_nearest_five |>.sum

theorem sum_equals_rounded_sum (n : ℕ) : sum_to_n n = sum_rounded_to_five n := by
  sorry

-- The main theorem
theorem jo_equals_alex_sum : sum_to_n 200 = sum_rounded_to_five 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_rounded_sum_jo_equals_alex_sum_l219_21999


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l219_21911

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) : 
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l219_21911


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l219_21919

theorem tangent_point_coordinates (x y : ℝ) :
  y = x^2 →  -- curve equation
  (2 * x = -3) →  -- slope condition
  (x = -3/2 ∧ y = 9/4)  -- coordinates of point P
  := by sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l219_21919


namespace NUMINAMATH_CALUDE_total_cash_reward_l219_21998

/-- Represents a subject with its grade, credit hours, and cash reward per grade point -/
structure Subject where
  name : String
  grade : Nat
  creditHours : Nat
  cashRewardPerPoint : Nat

/-- Calculates the total cash reward for a given subject -/
def subjectReward (s : Subject) : Nat :=
  s.grade * s.cashRewardPerPoint

/-- Represents the artwork reward -/
def artworkReward : Nat := 20

/-- List of subjects with their respective information -/
def subjects : List Subject := [
  ⟨"Mathematics", 2, 5, 5⟩,
  ⟨"English", 3, 4, 4⟩,
  ⟨"Spanish", 3, 4, 4⟩,
  ⟨"Physics", 3, 4, 3⟩,
  ⟨"Chemistry", 3, 3, 3⟩,
  ⟨"History", 4, 3, 5⟩
]

/-- Calculates the total cash reward for all subjects -/
def totalSubjectsReward : Nat :=
  (subjects.map subjectReward).sum

/-- Theorem: The total cash reward Milo gets is $92 -/
theorem total_cash_reward : totalSubjectsReward + artworkReward = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_cash_reward_l219_21998


namespace NUMINAMATH_CALUDE_ryan_bus_meet_once_l219_21909

/-- Represents the movement of Ryan and the bus on a linear trail --/
structure TrailMovement where
  ryan_speed : ℝ
  bus_speed : ℝ
  bench_distance : ℝ
  regular_stop_time : ℝ
  extra_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of times Ryan and the bus meet --/
def number_of_meetings (movement : TrailMovement) : ℕ :=
  sorry

/-- The specific trail movement scenario described in the problem --/
def problem_scenario : TrailMovement :=
  { ryan_speed := 6
  , bus_speed := 15
  , bench_distance := 300
  , regular_stop_time := 45
  , extra_stop_time := 90
  , initial_distance := 300 }

/-- Theorem stating that Ryan and the bus meet exactly once --/
theorem ryan_bus_meet_once :
  number_of_meetings problem_scenario = 1 := by
  sorry

end NUMINAMATH_CALUDE_ryan_bus_meet_once_l219_21909


namespace NUMINAMATH_CALUDE_paint_area_is_127_l219_21993

/-- Calculates the area to be painted on a wall with two windows. -/
def areaToPaint (wallHeight wallLength window1Height window1Width window2Height window2Width : ℝ) : ℝ :=
  wallHeight * wallLength - (window1Height * window1Width + window2Height * window2Width)

/-- Proves that the area to be painted is 127 square feet given the specified dimensions. -/
theorem paint_area_is_127 :
  areaToPaint 10 15 3 5 2 4 = 127 := by
  sorry

#eval areaToPaint 10 15 3 5 2 4

end NUMINAMATH_CALUDE_paint_area_is_127_l219_21993


namespace NUMINAMATH_CALUDE_tan_alpha_minus_beta_equals_one_l219_21941

theorem tan_alpha_minus_beta_equals_one (α β : Real) 
  (h : Real.tan β = (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α)) : 
  Real.tan (α - β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_beta_equals_one_l219_21941


namespace NUMINAMATH_CALUDE_remainder_zero_mod_eight_l219_21944

theorem remainder_zero_mod_eight :
  (71^7 - 73^10) * (73^5 + 71^3) ≡ 0 [ZMOD 8] := by
sorry

end NUMINAMATH_CALUDE_remainder_zero_mod_eight_l219_21944


namespace NUMINAMATH_CALUDE_max_m_value_l219_21972

def f (x : ℝ) : ℝ := |2*x + 1| + |3*x - 2|

theorem max_m_value (h : Set.Icc (-4/5 : ℝ) (6/5) = {x : ℝ | f x ≤ 5}) :
  ∃ m : ℝ, m = 2 ∧ 
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ m^2 - 3*m + 5) ∧
  (∀ m' : ℝ, m' > m → ∃ x : ℝ, |x - 1| + |x + 2| < m'^2 - 3*m' + 5) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l219_21972


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_equals_one_l219_21963

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_square_equals_one :
  (¬ ∃ x : ℝ, x^2 = 1) ↔ (∀ x : ℝ, x^2 ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_equals_one_l219_21963


namespace NUMINAMATH_CALUDE_alligator_walking_time_l219_21900

/-- The combined walking time of alligators given Paul's initial journey time and additional return time -/
theorem alligator_walking_time (initial_time return_additional_time : ℕ) :
  initial_time = 4 ∧ return_additional_time = 2 →
  initial_time + (initial_time + return_additional_time) = 10 := by
  sorry

#check alligator_walking_time

end NUMINAMATH_CALUDE_alligator_walking_time_l219_21900


namespace NUMINAMATH_CALUDE_sack_of_rice_weight_l219_21974

theorem sack_of_rice_weight (cost : ℝ) (price_per_kg : ℝ) (profit : ℝ) (weight : ℝ) : 
  cost = 50 → 
  price_per_kg = 1.20 → 
  profit = 10 → 
  price_per_kg * weight = cost + profit → 
  weight = 50 := by
sorry

end NUMINAMATH_CALUDE_sack_of_rice_weight_l219_21974


namespace NUMINAMATH_CALUDE_correct_final_bill_amount_l219_21926

/-- Calculates the final bill amount after applying two late fees -/
def final_bill_amount (original_bill : ℝ) (first_fee_rate : ℝ) (second_fee_rate : ℝ) : ℝ :=
  let after_first_fee := original_bill * (1 + first_fee_rate)
  after_first_fee * (1 + second_fee_rate)

/-- Theorem stating that the final bill amount is correct -/
theorem correct_final_bill_amount :
  final_bill_amount 250 0.02 0.03 = 262.65 := by
  sorry

#eval final_bill_amount 250 0.02 0.03

end NUMINAMATH_CALUDE_correct_final_bill_amount_l219_21926


namespace NUMINAMATH_CALUDE_birds_on_fence_l219_21907

theorem birds_on_fence (initial_birds : ℕ) (storks_joined : ℕ) (stork_bird_difference : ℕ) :
  initial_birds = 3 →
  storks_joined = 6 →
  stork_bird_difference = 1 →
  ∃ (birds_joined : ℕ), birds_joined = 2 ∧
    storks_joined = initial_birds + birds_joined + stork_bird_difference :=
by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l219_21907


namespace NUMINAMATH_CALUDE_unique_number_satisfying_condition_l219_21905

theorem unique_number_satisfying_condition : ∃! x : ℚ, ((x / 3) * 24) - 7 = 41 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_condition_l219_21905


namespace NUMINAMATH_CALUDE_centerville_snail_count_l219_21975

/-- The number of snails removed from Centerville -/
def snails_removed : ℕ := 3482

/-- The number of snails remaining in Centerville -/
def snails_remaining : ℕ := 8278

/-- The original number of snails in Centerville -/
def original_snails : ℕ := snails_removed + snails_remaining

theorem centerville_snail_count : original_snails = 11760 := by
  sorry

end NUMINAMATH_CALUDE_centerville_snail_count_l219_21975


namespace NUMINAMATH_CALUDE_initial_distance_problem_l219_21989

theorem initial_distance_problem (enrique_speed jamal_speed meeting_time : ℝ) 
  (h1 : enrique_speed = 16)
  (h2 : jamal_speed = 23)
  (h3 : meeting_time = 8) :
  enrique_speed * meeting_time + jamal_speed * meeting_time = 312 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_problem_l219_21989


namespace NUMINAMATH_CALUDE_flower_beds_fraction_l219_21939

/-- Represents a rectangular yard with two congruent isosceles right triangular flower beds -/
structure YardWithFlowerBeds where
  /-- Length of the shorter parallel side of the trapezoid -/
  short_side : ℝ
  /-- Length of the longer parallel side of the trapezoid -/
  long_side : ℝ
  /-- Assumption that the short side is 20 meters -/
  short_side_eq : short_side = 20
  /-- Assumption that the long side is 30 meters -/
  long_side_eq : long_side = 30

/-- The fraction of the yard occupied by the flower beds is 1/6 -/
theorem flower_beds_fraction (yard : YardWithFlowerBeds) : 
  (yard.long_side - yard.short_side)^2 / (4 * yard.long_side * (yard.long_side - yard.short_side)) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_flower_beds_fraction_l219_21939


namespace NUMINAMATH_CALUDE_min_packs_for_120_cans_l219_21973

/-- Represents the available pack sizes for soda cans -/
def PackSizes : List Nat := [6, 12, 24, 30]

/-- The total number of cans needed -/
def TotalCans : Nat := 120

/-- A function that checks if a combination of packs can exactly make the total number of cans -/
def canMakeTotalCans (packs : List Nat) : Bool :=
  (packs.map (fun size => size * (packs.count size))).sum = TotalCans

/-- Theorem stating that the minimum number of packs needed to buy exactly 120 cans is 4 -/
theorem min_packs_for_120_cans :
  ∃ (packs : List Nat),
    packs.all (PackSizes.contains ·) ∧
    canMakeTotalCans packs ∧
    packs.length = 4 ∧
    (∀ (other_packs : List Nat),
      other_packs.all (PackSizes.contains ·) →
      canMakeTotalCans other_packs →
      other_packs.length ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_min_packs_for_120_cans_l219_21973


namespace NUMINAMATH_CALUDE_equal_roots_iff_discriminant_zero_equal_roots_h_l219_21961

/-- For a quadratic equation ax² + bx + c = 0, the discriminant is b² - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation has equal roots if and only if its discriminant is zero -/
theorem equal_roots_iff_discriminant_zero (a b c : ℝ) (ha : a ≠ 0) :
  ∃ x, a*x^2 + b*x + c = 0 ∧ (∀ y, a*y^2 + b*y + c = 0 → y = x) ↔ discriminant a b c = 0 :=
sorry

/-- The value of h for which the equation 3x² - 4x + h/3 = 0 has equal roots -/
theorem equal_roots_h : ∃! h : ℝ, discriminant 3 (-4) (h/3) = 0 ∧ h = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_iff_discriminant_zero_equal_roots_h_l219_21961
