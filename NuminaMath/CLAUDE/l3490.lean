import Mathlib

namespace NUMINAMATH_CALUDE_scientific_notation_properties_l3490_349025

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : Float
  exponent : Int

/-- Counts the number of significant figures in a scientific notation number -/
def count_significant_figures (n : ScientificNotation) : Nat :=
  sorry

/-- Determines the place value of the last significant digit -/
def last_significant_place (n : ScientificNotation) : String :=
  sorry

/-- The main theorem -/
theorem scientific_notation_properties :
  let n : ScientificNotation := { coefficient := 6.30, exponent := 5 }
  count_significant_figures n = 3 ∧
  last_significant_place n = "ten thousand's place" :=
  sorry

end NUMINAMATH_CALUDE_scientific_notation_properties_l3490_349025


namespace NUMINAMATH_CALUDE_fraction_equality_implies_power_equality_l3490_349049

theorem fraction_equality_implies_power_equality
  (a b c : ℝ) (k : ℕ) 
  (h_odd : Odd k)
  (h_eq : 1/a + 1/b + 1/c = 1/(a+b+c)) :
  1/a^k + 1/b^k + 1/c^k = 1/(a^k + b^k + c^k) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_power_equality_l3490_349049


namespace NUMINAMATH_CALUDE_max_value_and_inequality_l3490_349018

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| - 2 * |x + 1|

-- State the theorem
theorem max_value_and_inequality :
  -- Part 1: The maximum value of f(x) is 2
  (∃ (k : ℝ), ∀ (x : ℝ), f x ≤ k ∧ ∃ (x₀ : ℝ), f x₀ = k) ∧ 
  (∀ (k : ℝ), (∀ (x : ℝ), f x ≤ k ∧ ∃ (x₀ : ℝ), f x₀ = k) → k = 2) ∧
  -- Part 2: For m > 0 and n > 0, if 1/m + 1/(2n) = 2, then m + 2n ≥ 2
  ∀ (m n : ℝ), m > 0 → n > 0 → 1/m + 1/(2*n) = 2 → m + 2*n ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_inequality_l3490_349018


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l3490_349014

theorem fourth_term_of_geometric_progression :
  let a₁ : ℝ := Real.sqrt 4
  let a₂ : ℝ := (4 : ℝ) ^ (1/4)
  let a₃ : ℝ := (4 : ℝ) ^ (1/8)
  let r : ℝ := a₂ / a₁
  let a₄ : ℝ := a₃ * r
  a₄ = (1/4 : ℝ) ^ (1/8) :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l3490_349014


namespace NUMINAMATH_CALUDE_special_function_at_eight_l3490_349074

/-- A monotonic function on (0, +∞) satisfying certain conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧ 
  (∀ x, x > 0 → f x > -4/x) ∧
  (∀ x, x > 0 → f (f x + 4/x) = 3)

/-- The main theorem stating that f(8) = 7/2 for a SpecialFunction -/
theorem special_function_at_eight (f : ℝ → ℝ) (h : SpecialFunction f) : f 8 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_eight_l3490_349074


namespace NUMINAMATH_CALUDE_freds_weekend_earnings_l3490_349089

/-- Fred's earnings from delivering newspapers -/
def newspaper_earnings : ℕ := 16

/-- Fred's earnings from washing cars -/
def car_washing_earnings : ℕ := 74

/-- Fred's total weekend earnings -/
def weekend_earnings : ℕ := 90

/-- Theorem stating that Fred's weekend earnings equal the sum of his newspaper delivery and car washing earnings -/
theorem freds_weekend_earnings : 
  newspaper_earnings + car_washing_earnings = weekend_earnings := by
  sorry

end NUMINAMATH_CALUDE_freds_weekend_earnings_l3490_349089


namespace NUMINAMATH_CALUDE_monochromatic_unit_area_triangle_exists_l3490_349037

-- Define a type for colors
inductive Color
  | Red
  | Green
  | Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring of the plane
def Coloring := Point → Color

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ :=
  sorry

-- Check if all vertices of a triangle have the same color
def monochromatic (t : Triangle) (coloring : Coloring) : Prop :=
  coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c

-- Main theorem
theorem monochromatic_unit_area_triangle_exists (coloring : Coloring) :
  ∃ t : Triangle, triangleArea t = 1 ∧ monochromatic t coloring := by
  sorry


end NUMINAMATH_CALUDE_monochromatic_unit_area_triangle_exists_l3490_349037


namespace NUMINAMATH_CALUDE_perimeter_after_cuts_l3490_349004

/-- The perimeter of a square after cutting out shapes --/
theorem perimeter_after_cuts (initial_side : ℝ) (green_side : ℝ) : 
  initial_side = 10 → green_side = 2 → 
  (4 * initial_side) + (4 * green_side) = 44 := by
  sorry

#check perimeter_after_cuts

end NUMINAMATH_CALUDE_perimeter_after_cuts_l3490_349004


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l3490_349059

/-- The gain percent when a cycle is bought for 450 Rs and sold for 520 Rs -/
def gain_percent (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that the gain percent is 15.56% -/
theorem cycle_gain_percent : 
  gain_percent 450 520 = 15.56 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l3490_349059


namespace NUMINAMATH_CALUDE_base5_500_l3490_349024

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Converts a list of digits in base 5 to a natural number --/
def fromBase5 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 5 * acc + d) 0

theorem base5_500 : toBase5 500 = [4, 0, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_base5_500_l3490_349024


namespace NUMINAMATH_CALUDE_total_gratuity_is_23_02_l3490_349001

-- Define the structure for menu items
structure MenuItem where
  name : String
  basePrice : Float
  taxRate : Float

-- Define the menu items
def nyStriploin : MenuItem := ⟨"NY Striploin", 80, 0.10⟩
def wineGlass : MenuItem := ⟨"Glass of wine", 10, 0.15⟩
def dessert : MenuItem := ⟨"Dessert", 12, 0.05⟩
def waterBottle : MenuItem := ⟨"Bottle of water", 3, 0⟩

-- Define the gratuity rate
def gratuityRate : Float := 0.20

-- Function to calculate the total price with tax for an item
def totalPriceWithTax (item : MenuItem) : Float :=
  item.basePrice * (1 + item.taxRate)

-- Function to calculate the gratuity for an item
def calculateGratuity (item : MenuItem) : Float :=
  totalPriceWithTax item * gratuityRate

-- Theorem stating that the total gratuity is $23.02
theorem total_gratuity_is_23_02 :
  calculateGratuity nyStriploin +
  calculateGratuity wineGlass +
  calculateGratuity dessert +
  calculateGratuity waterBottle = 23.02 := by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_total_gratuity_is_23_02_l3490_349001


namespace NUMINAMATH_CALUDE_midpoint_of_intersections_l3490_349096

/-- The line equation y = x - 3 -/
def line_eq (x y : ℝ) : Prop := y = x - 3

/-- The parabola equation y^2 = 2x -/
def parabola_eq (x y : ℝ) : Prop := y^2 = 2*x

/-- A point (x, y) is on both the line and the parabola -/
def intersection_point (x y : ℝ) : Prop := line_eq x y ∧ parabola_eq x y

/-- There exist two distinct intersection points -/
axiom two_intersections : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  x₁ ≠ x₂ ∧ intersection_point x₁ y₁ ∧ intersection_point x₂ y₂

theorem midpoint_of_intersections : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    intersection_point x₁ y₁ ∧ 
    intersection_point x₂ y₂ ∧ 
    ((x₁ + x₂) / 2 = 4 ∧ (y₁ + y₂) / 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_midpoint_of_intersections_l3490_349096


namespace NUMINAMATH_CALUDE_triangle_is_obtuse_l3490_349015

theorem triangle_is_obtuse (A : ℝ) (h1 : 0 < A ∧ A < π) 
  (h2 : Real.sin A + Real.cos A = 7/12) : 
  π/2 < A ∧ A < π :=
by sorry

end NUMINAMATH_CALUDE_triangle_is_obtuse_l3490_349015


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3490_349045

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem fifteenth_term_of_sequence : 
  let a₁ : ℚ := 5
  let r : ℚ := 1/2
  let n : ℕ := 15
  geometric_sequence a₁ r n = 5/16384 := by
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3490_349045


namespace NUMINAMATH_CALUDE_largest_even_three_digit_number_with_conditions_l3490_349070

theorem largest_even_three_digit_number_with_conditions :
  ∃ (x : ℕ), 
    x = 972 ∧
    x % 2 = 0 ∧
    100 ≤ x ∧ x < 1000 ∧
    x % 5 = 2 ∧
    Nat.gcd 30 (Nat.gcd x 15) = 3 ∧
    ∀ (y : ℕ), 
      y % 2 = 0 → 
      100 ≤ y → y < 1000 → 
      y % 5 = 2 → 
      Nat.gcd 30 (Nat.gcd y 15) = 3 → 
      y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_even_three_digit_number_with_conditions_l3490_349070


namespace NUMINAMATH_CALUDE_set_operations_l3490_349034

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : Set ℝ := {x | x^2 - 4*x ≤ 0}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 3}) ∧
  (A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 4}) ∧
  ((Aᶜ ∩ Bᶜ) = {x : ℝ | x < -1 ∨ x > 4}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3490_349034


namespace NUMINAMATH_CALUDE_round_trip_distance_l3490_349067

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The distance to the star in light-years -/
def star_distance : ℝ := 25

/-- The duration of the round trip in years -/
def trip_duration : ℝ := 50

/-- The total distance traveled by light in a round trip to the star over the given duration -/
def total_distance : ℝ := 2 * star_distance * light_year_distance

theorem round_trip_distance : total_distance = 5.87e14 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_distance_l3490_349067


namespace NUMINAMATH_CALUDE_cone_volume_l3490_349044

/-- Given a cone whose lateral surface, when unrolled, forms a sector with radius 3 and 
    central angle 2π/3, prove that its volume is (2√2/3)π -/
theorem cone_volume (r l : ℝ) (h : ℝ) : 
  r = 1 → l = 3 → h = 2 * Real.sqrt 2 → 
  (1/3) * π * r^2 * h = (2 * Real.sqrt 2 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l3490_349044


namespace NUMINAMATH_CALUDE_range_of_f_l3490_349081

theorem range_of_f (x : ℝ) : 
  let f := fun (x : ℝ) => Real.sin x^4 - Real.sin x * Real.cos x + Real.cos x^4
  0 ≤ f x ∧ f x ≤ 9/8 ∧ 
  (∃ y : ℝ, f y = 0) ∧ 
  (∃ z : ℝ, f z = 9/8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l3490_349081


namespace NUMINAMATH_CALUDE_odd_integers_square_l3490_349013

theorem odd_integers_square (a b : ℕ) (ha : Odd a) (hb : Odd b) (hab : ∃ k : ℕ, a^b * b^a = k^2) :
  ∃ m : ℕ, a * b = m^2 := by
sorry

end NUMINAMATH_CALUDE_odd_integers_square_l3490_349013


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3490_349046

theorem cubic_roots_sum_cubes (p q r : ℂ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  p^3 - p^2 + p - 2 = 0 →
  q^3 - q^2 + q - 2 = 0 →
  r^3 - r^2 + r - 2 = 0 →
  p^3 + q^3 + r^3 = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3490_349046


namespace NUMINAMATH_CALUDE_absolute_value_difference_l3490_349028

theorem absolute_value_difference : |-3 * (7 - 15)| - |(5 - 7)^2 + (-4)^2| = 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_difference_l3490_349028


namespace NUMINAMATH_CALUDE_randy_blocks_theorem_l3490_349006

/-- The number of blocks Randy used to build a tower -/
def blocks_used : ℕ := 25

/-- The number of blocks Randy has left -/
def blocks_left : ℕ := 72

/-- The initial number of blocks Randy had -/
def initial_blocks : ℕ := blocks_used + blocks_left

theorem randy_blocks_theorem : initial_blocks = 97 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_theorem_l3490_349006


namespace NUMINAMATH_CALUDE_general_solution_is_correct_l3490_349055

-- Define the system of equations
def system (x : Fin 4 → ℝ) : Prop :=
  x 0 + 7 * x 1 - 8 * x 2 + 9 * x 3 = 4

-- Define the general solution
def general_solution (α : Fin 3 → ℝ) : Fin 4 → ℝ
  | 0 => -7 * α 0 + 8 * α 1 - 9 * α 2 + 4
  | 1 => α 0
  | 2 => α 1
  | 3 => α 2

-- Theorem statement
theorem general_solution_is_correct :
  ∀ α : Fin 3 → ℝ, system (general_solution α) :=
by
  sorry

end NUMINAMATH_CALUDE_general_solution_is_correct_l3490_349055


namespace NUMINAMATH_CALUDE_min_intersection_size_l3490_349033

theorem min_intersection_size (total blue_eyes backpack : ℕ) 
  (h_total : total = 35)
  (h_blue : blue_eyes = 18)
  (h_backpack : backpack = 24) :
  blue_eyes + backpack - total ≤ (blue_eyes ⊓ backpack) :=
by sorry

end NUMINAMATH_CALUDE_min_intersection_size_l3490_349033


namespace NUMINAMATH_CALUDE_cost_of_leftover_drinks_l3490_349005

theorem cost_of_leftover_drinks : 
  let soda_bought := 30
  let soda_price := 2
  let energy_bought := 20
  let energy_price := 3
  let smoothie_bought := 15
  let smoothie_price := 4
  let soda_consumed := 10
  let energy_consumed := 14
  let smoothie_consumed := 5
  
  let soda_leftover := soda_bought - soda_consumed
  let energy_leftover := energy_bought - energy_consumed
  let smoothie_leftover := smoothie_bought - smoothie_consumed
  
  let leftover_cost := soda_leftover * soda_price + 
                       energy_leftover * energy_price + 
                       smoothie_leftover * smoothie_price
  
  leftover_cost = 98 := by sorry

end NUMINAMATH_CALUDE_cost_of_leftover_drinks_l3490_349005


namespace NUMINAMATH_CALUDE_sine_function_parameters_l3490_349053

theorem sine_function_parameters (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c < 0) :
  (∀ x, a * Real.sin (b * x) + c ≤ 3) ∧
  (∃ x, a * Real.sin (b * x) + c = 3) ∧
  (∀ x, a * Real.sin (b * x) + c ≥ -5) ∧
  (∃ x, a * Real.sin (b * x) + c = -5) →
  a = 4 ∧ c = -1 := by
sorry

end NUMINAMATH_CALUDE_sine_function_parameters_l3490_349053


namespace NUMINAMATH_CALUDE_parking_savings_l3490_349076

/-- Calculates the yearly savings when renting a parking space monthly instead of weekly. -/
theorem parking_savings (weekly_rate : ℕ) (monthly_rate : ℕ) : 
  weekly_rate = 10 → monthly_rate = 24 → (52 * weekly_rate) - (12 * monthly_rate) = 232 := by
  sorry

end NUMINAMATH_CALUDE_parking_savings_l3490_349076


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3490_349093

theorem inequality_solution_set :
  {x : ℝ | 2 * x^2 + 2 * x - 3 > 7 - x} = {x : ℝ | x < -2 ∨ x > 5/2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3490_349093


namespace NUMINAMATH_CALUDE_complement_of_hit_at_least_once_l3490_349079

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ω.1 ∨ ω.2

-- Define the event of missing the target both times
def miss_both_times (ω : Ω) : Prop :=
  ¬ω.1 ∧ ¬ω.2

-- Theorem stating that missing both times is the complement of hitting at least once
theorem complement_of_hit_at_least_once :
  ∀ ω : Ω, miss_both_times ω ↔ ¬(hit_at_least_once ω) :=
sorry

end NUMINAMATH_CALUDE_complement_of_hit_at_least_once_l3490_349079


namespace NUMINAMATH_CALUDE_beetle_probability_theorem_l3490_349091

/-- Represents the probability of a beetle touching a horizontal edge first -/
def beetle_horizontal_edge_probability (start_x start_y : ℕ) (grid_size : ℕ) : ℝ :=
  sorry

/-- The grid is 10x10 -/
def grid_size : ℕ := 10

/-- The beetle starts at (3, 4) -/
def start_x : ℕ := 3
def start_y : ℕ := 4

/-- Theorem stating the probability of the beetle touching a horizontal edge first -/
theorem beetle_probability_theorem :
  beetle_horizontal_edge_probability start_x start_y grid_size = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_beetle_probability_theorem_l3490_349091


namespace NUMINAMATH_CALUDE_circle_area_in_square_l3490_349073

theorem circle_area_in_square (square_area : ℝ) (h : square_area = 400) :
  let square_side := Real.sqrt square_area
  let circle_radius := square_side / 2
  let circle_area := Real.pi * circle_radius ^ 2
  circle_area = 100 * Real.pi := by sorry

end NUMINAMATH_CALUDE_circle_area_in_square_l3490_349073


namespace NUMINAMATH_CALUDE_find_N_l3490_349031

theorem find_N (X Y Z N : ℝ) 
  (h1 : 0.15 * X = 0.25 * N + Y) 
  (h2 : X + Y = Z) : 
  N = 4.6 * X - 4 * Z := by
sorry

end NUMINAMATH_CALUDE_find_N_l3490_349031


namespace NUMINAMATH_CALUDE_factor_tree_value_l3490_349052

-- Define the structure of the factor tree
structure FactorTree :=
  (A B C D E : ℝ)

-- Define the conditions of the factor tree
def valid_factor_tree (t : FactorTree) : Prop :=
  t.A^2 = t.B * t.C ∧
  t.B = 2 * t.D ∧
  t.D = 2 * 4 ∧
  t.C = 7 * t.E ∧
  t.E = 7 * 2

-- Theorem statement
theorem factor_tree_value (t : FactorTree) (h : valid_factor_tree t) : 
  t.A = 28 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_value_l3490_349052


namespace NUMINAMATH_CALUDE_different_signs_implies_range_l3490_349016

theorem different_signs_implies_range (m : ℝ) : 
  ((2 - m) * (|m| - 3) < 0) → ((-3 < m ∧ m < 2) ∨ m > 3) := by
sorry

end NUMINAMATH_CALUDE_different_signs_implies_range_l3490_349016


namespace NUMINAMATH_CALUDE_distance_is_sqrt_206_l3490_349062

def point : ℝ × ℝ × ℝ := (2, 3, 1)

def line_point : ℝ × ℝ × ℝ := (8, 10, 12)

def line_direction : ℝ × ℝ × ℝ := (2, 3, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_sqrt_206 : 
  distance_to_line point line_point line_direction = Real.sqrt 206 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_sqrt_206_l3490_349062


namespace NUMINAMATH_CALUDE_find_a_l3490_349087

theorem find_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 7) (h3 : c = 4) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3490_349087


namespace NUMINAMATH_CALUDE_tv_cash_savings_l3490_349027

/-- Calculates the savings when buying a television by cash instead of installments -/
theorem tv_cash_savings 
  (cash_price : ℕ) 
  (down_payment : ℕ) 
  (monthly_payment : ℕ) 
  (num_months : ℕ) : 
  cash_price = 400 →
  down_payment = 120 →
  monthly_payment = 30 →
  num_months = 12 →
  down_payment + monthly_payment * num_months - cash_price = 80 := by
sorry

end NUMINAMATH_CALUDE_tv_cash_savings_l3490_349027


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3490_349002

theorem angle_measure_proof (x : ℝ) : 
  (180 - x) = 3 * (90 - x) + 10 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3490_349002


namespace NUMINAMATH_CALUDE_symmetric_point_example_l3490_349023

/-- Given a point (x, y) in the plane, the point symmetric to it with respect to the x-axis is (x, -y) -/
def symmetric_point_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The coordinates of the point symmetric to (3, 8) with respect to the x-axis are (3, -8) -/
theorem symmetric_point_example : symmetric_point_x_axis (3, 8) = (3, -8) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_example_l3490_349023


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3490_349043

theorem complex_equation_solution (z : ℂ) : z + z * Complex.I = 1 + 5 * Complex.I → z = 3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3490_349043


namespace NUMINAMATH_CALUDE_rose_price_calculation_l3490_349060

/-- Calculates the price per rose given the initial number of roses, 
    remaining roses, and total earnings -/
def price_per_rose (initial_roses : ℕ) (remaining_roses : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (initial_roses - remaining_roses)

theorem rose_price_calculation (initial_roses remaining_roses total_earnings : ℕ) 
  (h1 : initial_roses = 9)
  (h2 : remaining_roses = 4)
  (h3 : total_earnings = 35) :
  price_per_rose initial_roses remaining_roses total_earnings = 7 := by
  sorry

end NUMINAMATH_CALUDE_rose_price_calculation_l3490_349060


namespace NUMINAMATH_CALUDE_minimum_occupied_seats_l3490_349063

theorem minimum_occupied_seats (total_seats : ℕ) (h : total_seats = 120) :
  let min_occupied := (total_seats + 2) / 3
  min_occupied = 40 ∧
  ∀ n : ℕ, n < min_occupied → ∃ i : ℕ, i < total_seats ∧ 
    (∀ j : ℕ, j < total_seats → (j = i ∨ j = i + 1) → n ≤ j) :=
by sorry

end NUMINAMATH_CALUDE_minimum_occupied_seats_l3490_349063


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l3490_349066

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | |x - 1| < 1}

def B : Set ℝ := {x | x < 1 ∨ x ≥ 4}

theorem union_of_A_and_complement_of_B :
  A ∪ (U \ B) = {x : ℝ | 0 < x ∧ x < 4} :=
by sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l3490_349066


namespace NUMINAMATH_CALUDE_max_xy_on_line_segment_l3490_349021

/-- Given points A(2,0) and B(0,1), prove that the maximum value of xy for any point P(x,y) on the line segment AB is 1/2 -/
theorem max_xy_on_line_segment : 
  ∀ x y : ℝ, 
  0 ≤ x ∧ x ≤ 2 → -- Condition for x being on the line segment
  x / 2 + y = 1 → -- Equation of the line AB
  x * y ≤ (1 : ℝ) / 2 ∧ 
  ∃ x₀ y₀ : ℝ, 0 ≤ x₀ ∧ x₀ ≤ 2 ∧ x₀ / 2 + y₀ = 1 ∧ x₀ * y₀ = (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_on_line_segment_l3490_349021


namespace NUMINAMATH_CALUDE_parabola_directrix_l3490_349007

/-- Given a parabola with equation y = -3x^2 + 6x - 5, its directrix is y = -23/12 -/
theorem parabola_directrix (x y : ℝ) : 
  y = -3 * x^2 + 6 * x - 5 →
  ∃ (k : ℝ), k = -23/12 ∧ k = y - (1/(4 * -3)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3490_349007


namespace NUMINAMATH_CALUDE_smallest_apocalyptic_number_l3490_349042

/-- A number is apocalyptic if it has 6 different positive divisors that sum to 3528 -/
def IsApocalyptic (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ : ℕ),
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧
    d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧
    d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧
    d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧
    d₅ ≠ d₆ ∧
    d₁ > 0 ∧ d₂ > 0 ∧ d₃ > 0 ∧ d₄ > 0 ∧ d₅ > 0 ∧ d₆ > 0 ∧
    d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧ d₅ ∣ n ∧ d₆ ∣ n ∧
    d₁ + d₂ + d₃ + d₄ + d₅ + d₆ = 3528

theorem smallest_apocalyptic_number :
  IsApocalyptic 1440 ∧ ∀ m : ℕ, m < 1440 → ¬IsApocalyptic m := by
  sorry

end NUMINAMATH_CALUDE_smallest_apocalyptic_number_l3490_349042


namespace NUMINAMATH_CALUDE_base_4_to_16_digits_l3490_349035

theorem base_4_to_16_digits : ∀ n : ℕ,
  (4^4 ≤ n) ∧ (n < 4^5) →
  (16^2 ≤ n) ∧ (n < 16^3) :=
by sorry

end NUMINAMATH_CALUDE_base_4_to_16_digits_l3490_349035


namespace NUMINAMATH_CALUDE_match_total_weight_l3490_349088

/-- The number of times Terrell lifts the original weights -/
def original_lifts : ℕ := 10

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 25

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 20

/-- The number of weights used in each setup -/
def num_weights : ℕ := 2

/-- The total weight lifted with the original setup in pounds -/
def total_weight : ℕ := num_weights * original_weight * original_lifts

/-- The number of lifts required with the new weights to match the total weight -/
def required_lifts : ℚ := total_weight / (num_weights * new_weight)

theorem match_total_weight : required_lifts = 12.5 := by sorry

end NUMINAMATH_CALUDE_match_total_weight_l3490_349088


namespace NUMINAMATH_CALUDE_stream_speed_l3490_349022

/-- Given a boat traveling a round trip with known parameters, prove the speed of the stream -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) : 
  boat_speed = 16 → 
  distance = 7560 → 
  total_time = 960 → 
  ∃ (stream_speed : ℝ), 
    stream_speed = 2 ∧ 
    distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l3490_349022


namespace NUMINAMATH_CALUDE_saras_quarters_l3490_349094

theorem saras_quarters (initial_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 21)
  (h2 : final_quarters = 70) : 
  final_quarters - initial_quarters = 49 := by
  sorry

end NUMINAMATH_CALUDE_saras_quarters_l3490_349094


namespace NUMINAMATH_CALUDE_pencil_length_l3490_349078

/-- The length of the purple section of the pencil in centimeters -/
def purple_length : ℝ := 3.5

/-- The length of the black section of the pencil in centimeters -/
def black_length : ℝ := 2.8

/-- The length of the blue section of the pencil in centimeters -/
def blue_length : ℝ := 1.6

/-- The length of the green section of the pencil in centimeters -/
def green_length : ℝ := 0.9

/-- The length of the yellow section of the pencil in centimeters -/
def yellow_length : ℝ := 1.2

/-- The total length of the pencil is the sum of all colored sections -/
theorem pencil_length : 
  purple_length + black_length + blue_length + green_length + yellow_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l3490_349078


namespace NUMINAMATH_CALUDE_tangent_lines_perpendicular_range_l3490_349056

/-- The problem statement --/
theorem tangent_lines_perpendicular_range (a : ℝ) : 
  ∃ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀ ≤ 3/2 ∧
  let f (x : ℝ) := (a*x - 1) * Real.exp x
  let g (x : ℝ) := (1 - x) * Real.exp (-x)
  let f' (x : ℝ) := (a*x + a - 1) * Real.exp x
  let g' (x : ℝ) := (x - 2) * Real.exp (-x)
  f' x₀ * g' x₀ = -1 →
  1 ≤ a ∧ a ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_perpendicular_range_l3490_349056


namespace NUMINAMATH_CALUDE_a_zero_sufficient_a_zero_not_necessary_l3490_349092

def f (a b x : ℝ) : ℝ := x^2 + a * abs x + b

-- Sufficient condition
theorem a_zero_sufficient (a b : ℝ) :
  a = 0 → ∀ x, f a b x = f a b (-x) :=
sorry

-- Not necessary condition
theorem a_zero_not_necessary :
  ∃ a b : ℝ, a ≠ 0 ∧ (∀ x, f a b x = f a b (-x)) :=
sorry

end NUMINAMATH_CALUDE_a_zero_sufficient_a_zero_not_necessary_l3490_349092


namespace NUMINAMATH_CALUDE_rental_cost_difference_l3490_349036

/-- Calculates the difference in rental costs between a ski boat and a sailboat for a given duration. -/
theorem rental_cost_difference 
  (sailboat_cost_per_day : ℕ)
  (ski_boat_cost_per_hour : ℕ)
  (hours_per_day : ℕ)
  (num_days : ℕ)
  (h1 : sailboat_cost_per_day = 60)
  (h2 : ski_boat_cost_per_hour = 80)
  (h3 : hours_per_day = 3)
  (h4 : num_days = 2) :
  ski_boat_cost_per_hour * hours_per_day * num_days - sailboat_cost_per_day * num_days = 360 :=
by
  sorry

#check rental_cost_difference

end NUMINAMATH_CALUDE_rental_cost_difference_l3490_349036


namespace NUMINAMATH_CALUDE_x_equals_one_necessary_and_sufficient_l3490_349041

theorem x_equals_one_necessary_and_sufficient :
  ∀ x : ℝ, (x^2 - 2*x + 1 = 0) ↔ (x = 1) := by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_necessary_and_sufficient_l3490_349041


namespace NUMINAMATH_CALUDE_prob_laurent_ge_2chloe_l3490_349000

/-- Represents a uniform distribution over a real interval -/
structure UniformDist (a b : ℝ) where
  (a_le_b : a ≤ b)

/-- The probability that a random variable from distribution Y is at least twice 
    a random variable from distribution X -/
noncomputable def prob_y_ge_2x (X : UniformDist 0 1000) (Y : UniformDist 0 2000) : ℝ :=
  (1000 * 1000 / 2) / (1000 * 2000)

/-- Theorem stating that the probability of Laurent's number being at least 
    twice Chloe's number is 1/4 -/
theorem prob_laurent_ge_2chloe :
  ∀ (X : UniformDist 0 1000) (Y : UniformDist 0 2000),
  prob_y_ge_2x X Y = 1/4 := by sorry

end NUMINAMATH_CALUDE_prob_laurent_ge_2chloe_l3490_349000


namespace NUMINAMATH_CALUDE_equation_solution_l3490_349032

theorem equation_solution (x : ℝ) (h : x > 1) :
  (x^2 / (x - 1)) + Real.sqrt (x - 1) + (Real.sqrt (x - 1) / x^2) =
  ((x - 1) / x^2) + (1 / Real.sqrt (x - 1)) + (x^2 / Real.sqrt (x - 1)) ↔
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3490_349032


namespace NUMINAMATH_CALUDE_students_not_taking_test_l3490_349083

theorem students_not_taking_test
  (total_students : ℕ)
  (correct_q1 : ℕ)
  (correct_q2 : ℕ)
  (h1 : total_students = 25)
  (h2 : correct_q1 = 22)
  (h3 : correct_q2 = 20)
  : total_students - max correct_q1 correct_q2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_students_not_taking_test_l3490_349083


namespace NUMINAMATH_CALUDE_expression_value_l3490_349097

theorem expression_value : 
  let a : ℚ := 1/2
  (2 * a⁻¹ + a⁻¹ / 2) / a = 10 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3490_349097


namespace NUMINAMATH_CALUDE_vector_to_line_parallel_l3490_349030

/-- A vector pointing from the origin to a line parallel to another vector -/
theorem vector_to_line_parallel (t : ℝ) : ∃ (k : ℝ), ∃ (a b : ℝ),
  (a = 3 * t + 1 ∧ b = t + 1) ∧  -- Point on the line
  (∃ (c : ℝ), a = 3 * c ∧ b = c) ∧  -- Parallel to (3, 1)
  a = 3 * k - 2 ∧ b = k :=  -- The form of the vector
by sorry

end NUMINAMATH_CALUDE_vector_to_line_parallel_l3490_349030


namespace NUMINAMATH_CALUDE_two_digit_divisors_of_723_with_remainder_30_l3490_349075

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divides_with_remainder (d q r : ℕ) : Prop := ∃ k, d * k + r = q

theorem two_digit_divisors_of_723_with_remainder_30 :
  ∃! (S : Finset ℕ),
    (∀ n ∈ S, is_two_digit n ∧ divides_with_remainder n 723 30) ∧
    S.card = 4 ∧
    S = {33, 63, 77, 99} :=
by sorry

end NUMINAMATH_CALUDE_two_digit_divisors_of_723_with_remainder_30_l3490_349075


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_24_with_cube_root_between_7_9_and_8_l3490_349086

theorem unique_integer_divisible_by_24_with_cube_root_between_7_9_and_8 :
  ∃! n : ℕ+, 
    (∃ k : ℕ, n.val = 24 * k) ∧ 
    (7.9 : ℝ) < (n.val : ℝ)^(1/3) ∧ 
    (n.val : ℝ)^(1/3) < 8 ∧
    n.val = 504 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_24_with_cube_root_between_7_9_and_8_l3490_349086


namespace NUMINAMATH_CALUDE_medical_team_combinations_l3490_349069

theorem medical_team_combinations (n_male : Nat) (n_female : Nat) 
  (h1 : n_male = 6) (h2 : n_female = 5) : 
  (n_male.choose 2) * (n_female.choose 1) = 75 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_combinations_l3490_349069


namespace NUMINAMATH_CALUDE_initial_ducks_l3490_349012

theorem initial_ducks (initial : ℕ) (joined : ℕ) (total : ℕ) : 
  joined = 20 → total = 33 → initial + joined = total → initial = 13 := by
sorry

end NUMINAMATH_CALUDE_initial_ducks_l3490_349012


namespace NUMINAMATH_CALUDE_money_ratio_proof_l3490_349003

theorem money_ratio_proof (alison brittany brooke kent : ℕ) : 
  alison = brittany / 2 →
  brittany = 4 * brooke →
  kent = 1000 →
  alison = 4000 →
  brooke / kent = 2 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_proof_l3490_349003


namespace NUMINAMATH_CALUDE_complement_of_union_l3490_349020

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

-- State the theorem
theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3490_349020


namespace NUMINAMATH_CALUDE_flower_beds_count_l3490_349090

/-- Calculates the total number of flower beds in a garden with three sections. -/
def totalFlowerBeds (seeds1 seeds2 seeds3 : ℕ) (seedsPerBed1 seedsPerBed2 seedsPerBed3 : ℕ) : ℕ :=
  (seeds1 / seedsPerBed1) + (seeds2 / seedsPerBed2) + (seeds3 / seedsPerBed3)

/-- Proves that the total number of flower beds is 105 given the specific conditions. -/
theorem flower_beds_count :
  totalFlowerBeds 470 320 210 10 10 8 = 105 := by
  sorry

#eval totalFlowerBeds 470 320 210 10 10 8

end NUMINAMATH_CALUDE_flower_beds_count_l3490_349090


namespace NUMINAMATH_CALUDE_logarithm_domain_l3490_349017

theorem logarithm_domain (a : ℝ) : 
  (∀ x : ℝ, x < 2 → ∃ y : ℝ, y = Real.log (a - 3 * x)) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_domain_l3490_349017


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3490_349064

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  (x₁^2 + 3*x₁ - 1 = 0) → 
  (x₂^2 + 3*x₂ - 1 = 0) → 
  (x₁^2 - 3*x₂ + 1 = 11) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3490_349064


namespace NUMINAMATH_CALUDE_square_diagonal_l3490_349054

theorem square_diagonal (p : ℝ) (h : p = 200 * Real.sqrt 2) :
  let s := p / 4
  s * Real.sqrt 2 = 100 := by sorry

end NUMINAMATH_CALUDE_square_diagonal_l3490_349054


namespace NUMINAMATH_CALUDE_z_cube_coefficient_coefficient_is_17_l3490_349009

/-- The coefficient of z^3 in the expansion of (3z^3 + 2z^2 - 4z - 1)(4z^4 + z^3 - 2z^2 + 3) is 17 -/
theorem z_cube_coefficient (z : ℝ) : 
  (3 * z^3 + 2 * z^2 - 4 * z - 1) * (4 * z^4 + z^3 - 2 * z^2 + 3) = 
  12 * z^7 + 11 * z^6 - 20 * z^5 - 8 * z^4 + 17 * z^3 + 8 * z^2 - 12 * z - 3 := by
  sorry

/-- The coefficient of z^3 in the expansion is 17 -/
theorem coefficient_is_17 : 
  ∃ (a b c d e f g h : ℝ), 
    (3 * z^3 + 2 * z^2 - 4 * z - 1) * (4 * z^4 + z^3 - 2 * z^2 + 3) = 
    a * z^7 + b * z^6 + c * z^5 + d * z^4 + 17 * z^3 + e * z^2 + f * z + g := by
  sorry

end NUMINAMATH_CALUDE_z_cube_coefficient_coefficient_is_17_l3490_349009


namespace NUMINAMATH_CALUDE_max_rectangles_in_oblique_prism_l3490_349008

/-- Represents an oblique prism -/
structure ObliquePrism where
  base : Set (Point)
  lateral_edges : Set (Line)

/-- Counts the number of rectangular faces in an oblique prism -/
def count_rectangular_faces (prism : ObliquePrism) : ℕ := sorry

/-- The maximum number of rectangular faces in any oblique prism -/
def max_rectangular_faces : ℕ := 4

/-- Theorem stating that the maximum number of rectangular faces in an oblique prism is 4 -/
theorem max_rectangles_in_oblique_prism (prism : ObliquePrism) :
  count_rectangular_faces prism ≤ max_rectangular_faces :=
sorry

end NUMINAMATH_CALUDE_max_rectangles_in_oblique_prism_l3490_349008


namespace NUMINAMATH_CALUDE_students_in_jungkooks_class_l3490_349050

theorem students_in_jungkooks_class :
  let glasses_wearers : Nat := 9
  let non_glasses_wearers : Nat := 16
  glasses_wearers + non_glasses_wearers = 25 :=
by sorry

end NUMINAMATH_CALUDE_students_in_jungkooks_class_l3490_349050


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3490_349068

theorem fraction_equation_solution :
  ∃! x : ℚ, (x + 5) / (x - 3) = (x - 2) / (x + 2) ∧ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3490_349068


namespace NUMINAMATH_CALUDE_flash_catches_ace_l3490_349095

/-- The time it takes for Flash to catch up to Ace in a race -/
theorem flash_catches_ace (v a y : ℝ) (hv : v > 0) (ha : a > 0) (hy : y > 0) :
  let t := (v + Real.sqrt (v^2 + 2*a*y)) / a
  2 * (v * t + y) = a * t^2 := by sorry

end NUMINAMATH_CALUDE_flash_catches_ace_l3490_349095


namespace NUMINAMATH_CALUDE_dress_price_proof_l3490_349029

theorem dress_price_proof (P : ℝ) (Pd : ℝ) (Pf : ℝ) 
  (h1 : Pd = 0.85 * P) 
  (h2 : Pf = 1.25 * Pd) 
  (h3 : P - Pf = 5.25) : 
  Pd = 71.40 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_proof_l3490_349029


namespace NUMINAMATH_CALUDE_power_function_conditions_l3490_349039

def α_set : Set ℚ := {-1, 1, 2, 3/5, 7/2}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_domain_R (f : ℝ → ℝ) : Prop :=
  ∀ x, ∃ y, f x = y

def satisfies_conditions (α : ℚ) : Prop :=
  let f := fun x => x ^ (α : ℝ)
  has_domain_R f ∧ is_odd_function f

theorem power_function_conditions :
  ∀ α ∈ α_set, satisfies_conditions α ↔ α ∈ ({1, 3/5} : Set ℚ) :=
sorry

end NUMINAMATH_CALUDE_power_function_conditions_l3490_349039


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequalities_l3490_349040

theorem smallest_integer_satisfying_inequalities :
  ∀ x : ℤ, (x + 8 > 10 ∧ -3*x < -9) → x ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequalities_l3490_349040


namespace NUMINAMATH_CALUDE_distance_from_origin_l3490_349071

theorem distance_from_origin (x y n : ℝ) : 
  x > 1 →
  y = 8 →
  (x - 1)^2 + (y - 6)^2 = 12^2 →
  n^2 = x^2 + y^2 →
  n = Real.sqrt (205 + 2 * Real.sqrt 140) :=
by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l3490_349071


namespace NUMINAMATH_CALUDE_parabola_satisfies_equation_l3490_349082

/-- A parabola with vertex at the origin, symmetric about coordinate axes, passing through (2, -3) -/
structure Parabola where
  /-- The parabola passes through the point (2, -3) -/
  passes_through : (2 : ℝ)^2 + (-3 : ℝ)^2 ≠ 0

/-- The equation of the parabola -/
def parabola_equation (p : Parabola) : Prop :=
  (∀ x y : ℝ, y^2 = 9/2 * x) ∨ (∀ x y : ℝ, x^2 = -4/3 * y)

/-- Theorem stating that the parabola satisfies the given equation -/
theorem parabola_satisfies_equation (p : Parabola) : parabola_equation p := by
  sorry

end NUMINAMATH_CALUDE_parabola_satisfies_equation_l3490_349082


namespace NUMINAMATH_CALUDE_game_prime_exists_l3490_349098

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem game_prime_exists : 
  ∃ p : ℕ, 
    is_prime p ∧ 
    ∃ (a b c d : ℕ), 
      p = a * 1000 + b * 100 + c * 10 + d ∧
      a ∈ ({4, 7, 8} : Set ℕ) ∧
      b ∈ ({4, 5, 9} : Set ℕ) ∧
      c ∈ ({1, 2, 3} : Set ℕ) ∧
      d < 10 ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      p = 8923 :=
by
  sorry

end NUMINAMATH_CALUDE_game_prime_exists_l3490_349098


namespace NUMINAMATH_CALUDE_unknown_number_proof_l3490_349038

theorem unknown_number_proof (x : ℝ) : x^2 + 94^2 = 19872 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l3490_349038


namespace NUMINAMATH_CALUDE_weekend_run_ratio_l3490_349047

/-- Represents the miles run by Bill and Julia over a weekend --/
structure WeekendRun where
  billSaturday : ℝ
  billSunday : ℝ
  juliaSunday : ℝ
  m : ℝ

/-- Conditions for a valid WeekendRun --/
def ValidWeekendRun (run : WeekendRun) : Prop :=
  run.billSunday = run.billSaturday + 4 ∧
  run.juliaSunday = run.m * run.billSunday ∧
  run.billSaturday + run.billSunday + run.juliaSunday = 32

theorem weekend_run_ratio (run : WeekendRun) 
  (h : ValidWeekendRun run) :
  run.juliaSunday / run.billSunday = run.m :=
by
  sorry

#check weekend_run_ratio

end NUMINAMATH_CALUDE_weekend_run_ratio_l3490_349047


namespace NUMINAMATH_CALUDE_leo_has_more_leo_excess_marbles_l3490_349026

/-- The number of marbles Ben has -/
def ben_marbles : ℕ := 56

/-- The total number of marbles in the jar -/
def total_marbles : ℕ := 132

/-- Leo's marbles are the difference between the total and Ben's marbles -/
def leo_marbles : ℕ := total_marbles - ben_marbles

/-- The statement that Leo has more marbles than Ben -/
theorem leo_has_more : leo_marbles > ben_marbles := by sorry

/-- The main theorem: Leo has 20 more marbles than Ben -/
theorem leo_excess_marbles : leo_marbles - ben_marbles = 20 := by sorry

end NUMINAMATH_CALUDE_leo_has_more_leo_excess_marbles_l3490_349026


namespace NUMINAMATH_CALUDE_single_intersection_l3490_349072

def f (a x : ℝ) : ℝ := (a - 1) * x^2 - 4 * x + 2 * a

theorem single_intersection (a : ℝ) : 
  (∃! x, f a x = 0) ↔ (a = -1 ∨ a = 2 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_single_intersection_l3490_349072


namespace NUMINAMATH_CALUDE_product_of_sums_l3490_349084

theorem product_of_sums : (-1-2-3-4-5-6-7-8-9-10) * (1-2+3-4+5-6+7-8+9-10) = 275 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_l3490_349084


namespace NUMINAMATH_CALUDE_clock_angle_at_2_30_l3490_349099

/-- The number of degrees in a circle -/
def circle_degrees : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The angle moved by the hour hand in one hour -/
def hour_hand_degrees_per_hour : ℚ := circle_degrees / clock_hours

/-- The angle moved by the minute hand in one minute -/
def minute_hand_degrees_per_minute : ℚ := circle_degrees / minutes_per_hour

/-- The position of the hour hand at 2:30 -/
def hour_hand_position : ℚ := 2.5 * hour_hand_degrees_per_hour

/-- The position of the minute hand at 2:30 -/
def minute_hand_position : ℚ := 30 * minute_hand_degrees_per_minute

/-- The angle between the hour hand and minute hand at 2:30 -/
def angle_between_hands : ℚ := |minute_hand_position - hour_hand_position|

theorem clock_angle_at_2_30 :
  min angle_between_hands (circle_degrees - angle_between_hands) = 105 :=
sorry

end NUMINAMATH_CALUDE_clock_angle_at_2_30_l3490_349099


namespace NUMINAMATH_CALUDE_regular_decagon_interior_angle_regular_decagon_interior_angle_proof_l3490_349077

/-- The measure of an interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle : ℝ :=
  let n : ℕ := 10  -- number of sides in a decagon
  let total_interior_angle_sum : ℝ := (n - 2) * 180
  let interior_angle : ℝ := total_interior_angle_sum / n
  144

/-- Proof of the theorem -/
theorem regular_decagon_interior_angle_proof :
  regular_decagon_interior_angle = 144 := by
  sorry

end NUMINAMATH_CALUDE_regular_decagon_interior_angle_regular_decagon_interior_angle_proof_l3490_349077


namespace NUMINAMATH_CALUDE_sum_smallest_largest_consecutive_integers_l3490_349048

/-- Given an even number of consecutive integers with arithmetic mean z,
    the sum of the smallest and largest integers is equal to 2z. -/
theorem sum_smallest_largest_consecutive_integers (m : ℕ) (z : ℚ) (h_even : Even m) (h_pos : 0 < m) :
  let b : ℚ := (2 * z * m - m^2 + m) / (2 * m)
  (b + (b + m - 1)) = 2 * z :=
by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_consecutive_integers_l3490_349048


namespace NUMINAMATH_CALUDE_mean_inequality_l3490_349019

theorem mean_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (a + b + c) / 3 > (a * b * c) ^ (1/3) ∧ 
  (a * b * c) ^ (1/3) > 2 * a * b * c / (a * b + b * c + c * a) := by
sorry

end NUMINAMATH_CALUDE_mean_inequality_l3490_349019


namespace NUMINAMATH_CALUDE_parallelogram_area_bound_l3490_349065

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- A parallelogram -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  is_parallelogram : sorry

/-- The center of a polygon -/
def center (vertices : Fin n → ℝ × ℝ) : ℝ × ℝ := sorry

/-- The area of a polygon -/
def area (vertices : Fin n → ℝ × ℝ) : ℝ := sorry

/-- A parallelogram is inscribed in a hexagon if all its vertices are on or inside the hexagon -/
def inscribed (p : Parallelogram) (h : RegularHexagon) : Prop := sorry

theorem parallelogram_area_bound (h : RegularHexagon) (p : Parallelogram) 
  (h_inscribed : inscribed p h) 
  (h_center : center p.vertices = center h.vertices) : 
  area p.vertices ≤ (2/3) * area h.vertices := 
sorry

end NUMINAMATH_CALUDE_parallelogram_area_bound_l3490_349065


namespace NUMINAMATH_CALUDE_tourist_distribution_eq_105_l3490_349010

/-- The number of ways to distribute 8 tourists among 4 guides with exactly 2 tourists per guide -/
def tourist_distribution : ℕ :=
  (Nat.choose 8 2 * Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2) / 24

theorem tourist_distribution_eq_105 : tourist_distribution = 105 := by
  sorry

end NUMINAMATH_CALUDE_tourist_distribution_eq_105_l3490_349010


namespace NUMINAMATH_CALUDE_unique_root_condition_l3490_349057

/-- The equation ln(x+a) - 4(x+a)^2 + a = 0 has a unique root at x = 3 if and only if a = (3 ln 2 + 1) / 2 -/
theorem unique_root_condition (a : ℝ) : 
  (∃! x : ℝ, Real.log (x + a) - 4 * (x + a)^2 + a = 0 ∧ x = 3) ↔ 
  a = (3 * Real.log 2 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_unique_root_condition_l3490_349057


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3490_349061

/-- Given a line y = kx (k > 0) intersecting a circle (x-2)^2 + y^2 = 1 at two points A and B,
    where the distance AB = (2/5)√5, prove that k = 1/2 -/
theorem line_circle_intersection (k : ℝ) (h_k_pos : k > 0) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 - 2)^2 + (k * A.1)^2 = 1 ∧ 
    (B.1 - 2)^2 + (k * B.1)^2 = 1 ∧ 
    (A.1 - B.1)^2 + (k * A.1 - k * B.1)^2 = (2/5)^2 * 5) → 
  k = 1/2 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3490_349061


namespace NUMINAMATH_CALUDE_unique_solution_l3490_349051

/-- Represents a three-digit number formed by digits U, H, and A -/
def three_digit_number (U H A : Nat) : Nat := 100 * U + 10 * H + A

/-- Represents a two-digit number formed by digits U and H -/
def two_digit_number (U H : Nat) : Nat := 10 * U + H

/-- Checks if a number is a valid digit (0-9) -/
def is_digit (n : Nat) : Prop := n ≤ 9

/-- Checks if three numbers are distinct -/
def are_distinct (a b c : Nat) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The main theorem stating the unique solution to the puzzle -/
theorem unique_solution :
  ∃! (U H A : Nat),
    is_digit U ∧ is_digit H ∧ is_digit A ∧
    are_distinct U H A ∧
    U ≠ 0 ∧
    three_digit_number U H A = Nat.lcm (two_digit_number U H) (Nat.lcm (two_digit_number U A) (two_digit_number H A)) ∧
    U = 1 ∧ H = 5 ∧ A = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3490_349051


namespace NUMINAMATH_CALUDE_range_of_f_l3490_349058

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 4*x - 1

-- Define the domain
def Domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem range_of_f :
  { y | ∃ x ∈ Domain, f x = y } = { y | -6 ≤ y ∧ y ≤ 3 } :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3490_349058


namespace NUMINAMATH_CALUDE_isabels_bouquets_l3490_349011

theorem isabels_bouquets (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) :
  initial_flowers = 66 →
  flowers_per_bouquet = 8 →
  wilted_flowers = 10 →
  (initial_flowers - wilted_flowers) / flowers_per_bouquet = 7 :=
by sorry

end NUMINAMATH_CALUDE_isabels_bouquets_l3490_349011


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l3490_349080

/-- Given vectors a and b in ℝ³, prove that a - 5b equals the expected result. -/
theorem vector_subtraction_and_scalar_multiplication (a b : ℝ × ℝ × ℝ) :
  a = (-5, 3, 2) → b = (2, -1, 4) → a - 5 • b = (-15, 8, -18) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l3490_349080


namespace NUMINAMATH_CALUDE_polynomial_roots_b_value_l3490_349085

theorem polynomial_roots_b_value (A B C D : ℤ) : 
  (∀ z : ℤ, z > 0 → (z^6 - 10*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 16 = 0) → 
   (∃ x₁ x₂ x₃ x₄ x₅ x₆ : ℤ, 
      x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0 ∧ x₆ > 0 ∧
      z^6 - 10*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 16 = 
      (z - x₁) * (z - x₂) * (z - x₃) * (z - x₄) * (z - x₅) * (z - x₆))) →
  B = -88 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_b_value_l3490_349085
