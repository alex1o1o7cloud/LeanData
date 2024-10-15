import Mathlib

namespace NUMINAMATH_CALUDE_sum_squares_units_digit_3003_l492_49251

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def square (n : ℕ) : ℕ := n * n

def units_digit (n : ℕ) : ℕ := n % 10

theorem sum_squares_units_digit_3003 :
  units_digit (List.sum (List.map square (first_n_odd_integers 3003))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_units_digit_3003_l492_49251


namespace NUMINAMATH_CALUDE_perpendicular_condition_l492_49298

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the "in plane" relation for a line
variable (in_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_condition 
  (l m n : Line) (α : Plane) 
  (h1 : in_plane m α) 
  (h2 : in_plane n α) :
  (perp_line_plane l α → perp_line_line l m ∧ perp_line_line l n) ∧ 
  ¬(perp_line_line l m ∧ perp_line_line l n → perp_line_plane l α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l492_49298


namespace NUMINAMATH_CALUDE_bottom_right_height_l492_49203

/-- Represents a rectangle with area and height -/
structure Rectangle where
  area : ℝ
  height : Option ℝ

/-- Represents the layout of six rectangles -/
structure RectangleLayout where
  topLeft : Rectangle
  topMiddle : Rectangle
  topRight : Rectangle
  bottomLeft : Rectangle
  bottomMiddle : Rectangle
  bottomRight : Rectangle

/-- Given the layout of rectangles, prove the height of the bottom right rectangle is 5 -/
theorem bottom_right_height (layout : RectangleLayout) :
  layout.topLeft.area = 18 ∧
  layout.bottomLeft.area = 12 ∧
  layout.bottomMiddle.area = 16 ∧
  layout.topMiddle.area = 32 ∧
  layout.topRight.area = 48 ∧
  layout.bottomRight.area = 30 ∧
  layout.topLeft.height = some 6 →
  layout.bottomRight.height = some 5 := by
  sorry

#check bottom_right_height

end NUMINAMATH_CALUDE_bottom_right_height_l492_49203


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l492_49259

theorem smallest_divisible_by_1_to_10 : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l492_49259


namespace NUMINAMATH_CALUDE_nineteen_power_calculation_l492_49252

theorem nineteen_power_calculation : (19^11 / 19^8) * 19^3 = 47015881 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_power_calculation_l492_49252


namespace NUMINAMATH_CALUDE_jacobs_flock_total_l492_49299

/-- Represents the composition of Jacob's flock -/
structure Flock where
  goats : ℕ
  sheep : ℕ

/-- Theorem stating the total number of animals in Jacob's flock -/
theorem jacobs_flock_total (f : Flock) 
  (h1 : f.goats = f.sheep / 2)  -- One third of animals are goats, so goats = (sheep + goats) / 3 = sheep / 2
  (h2 : f.sheep = f.goats + 12) -- There are 12 more sheep than goats
  : f.goats + f.sheep = 36 := by
  sorry


end NUMINAMATH_CALUDE_jacobs_flock_total_l492_49299


namespace NUMINAMATH_CALUDE_opposites_sum_l492_49204

theorem opposites_sum (a b : ℝ) (h : a + b = 0) : 2006*a + 2 + 2006*b = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposites_sum_l492_49204


namespace NUMINAMATH_CALUDE_rectangle_to_square_trapezoid_l492_49267

theorem rectangle_to_square_trapezoid (width height area_square : ℝ) (y : ℝ) : 
  width = 16 →
  height = 9 →
  area_square = width * height →
  y = (Real.sqrt area_square) / 2 →
  y = 6 := by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_trapezoid_l492_49267


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l492_49213

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = focus + t • (Q - focus) ∨ Q = focus + t • (P - focus)

-- Define the theorem
theorem parabola_intersection_length 
  (P Q : ℝ × ℝ) 
  (h_P : parabola P.1 P.2) 
  (h_Q : parabola Q.1 Q.2) 
  (h_line : line_through_focus P Q) 
  (h_sum : P.1 + Q.1 = 9) : 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 11 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l492_49213


namespace NUMINAMATH_CALUDE_quadratic_inequality_integer_solutions_l492_49223

theorem quadratic_inequality_integer_solutions :
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ 7 * x^2 + 25 * x + 24 ≤ 30) ∧ Finset.card S = 7 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_integer_solutions_l492_49223


namespace NUMINAMATH_CALUDE_weight_of_b_l492_49236

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 43)
  (h2 : (a + b) / 2 = 48)
  (h3 : (b + c) / 2 = 42) :
  b = 51 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l492_49236


namespace NUMINAMATH_CALUDE_pet_store_gerbils_l492_49287

/-- The initial number of gerbils in a pet store -/
def initial_gerbils : ℕ := 68

/-- The number of gerbils sold -/
def sold_gerbils : ℕ := 14

/-- The difference between the initial number and the number sold -/
def difference : ℕ := 54

theorem pet_store_gerbils : 
  initial_gerbils = sold_gerbils + difference := by sorry

end NUMINAMATH_CALUDE_pet_store_gerbils_l492_49287


namespace NUMINAMATH_CALUDE_stating_medication_duration_l492_49207

/-- Represents the number of pills in one supply of medication -/
def pills_per_supply : ℕ := 60

/-- Represents the fraction of a pill taken each time -/
def pill_fraction : ℚ := 1/3

/-- Represents the number of days between each dose -/
def days_between_doses : ℕ := 3

/-- Represents the number of types of medication -/
def medication_types : ℕ := 2

/-- Represents the approximate number of days in a month -/
def days_per_month : ℕ := 30

/-- 
Theorem stating that the combined supply of medication will last 540 days,
which is approximately 18 months.
-/
theorem medication_duration :
  (pills_per_supply : ℚ) * days_between_doses / pill_fraction * medication_types = 540 ∧
  540 / days_per_month = 18 := by
  sorry


end NUMINAMATH_CALUDE_stating_medication_duration_l492_49207


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l492_49281

theorem smallest_number_with_conditions : ∃ n : ℕ, 
  (n = 1801) ∧ 
  (∀ m : ℕ, m < n → 
    (11 ∣ n) ∧ 
    (n % 2 = 1) ∧ 
    (n % 3 = 1) ∧ 
    (n % 4 = 1) ∧ 
    (n % 5 = 1) ∧ 
    (n % 6 = 1) ∧ 
    (n % 8 = 1) → 
    ¬((11 ∣ m) ∧ 
      (m % 2 = 1) ∧ 
      (m % 3 = 1) ∧ 
      (m % 4 = 1) ∧ 
      (m % 5 = 1) ∧ 
      (m % 6 = 1) ∧ 
      (m % 8 = 1))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l492_49281


namespace NUMINAMATH_CALUDE_square_area_ratio_l492_49200

theorem square_area_ratio (a b : ℝ) (h : a > 0) (k : b > 0) (perimeter_ratio : 4 * a = 4 * (4 * b)) :
  a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l492_49200


namespace NUMINAMATH_CALUDE_min_visible_sum_l492_49296

/-- Represents a die in the cube -/
structure Die where
  faces : Fin 6 → Nat
  sum_opposite : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents the large 4x4x4 cube -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Calculates the sum of visible faces on the large cube -/
def visibleSum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the minimum sum of visible faces -/
theorem min_visible_sum (cube : LargeCube) : 
  visibleSum cube ≥ 144 :=
sorry

end NUMINAMATH_CALUDE_min_visible_sum_l492_49296


namespace NUMINAMATH_CALUDE_derivative_of_f_l492_49224

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log 2

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2^x * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l492_49224


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l492_49248

theorem product_of_sums_and_differences (P Q R S : ℝ) : P * Q * R * S = 1 :=
  by
  have h1 : P = Real.sqrt 2011 + Real.sqrt 2010 := by sorry
  have h2 : Q = -Real.sqrt 2011 - Real.sqrt 2010 := by sorry
  have h3 : R = Real.sqrt 2011 - Real.sqrt 2010 := by sorry
  have h4 : S = Real.sqrt 2010 - Real.sqrt 2011 := by sorry
  sorry

#check product_of_sums_and_differences

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l492_49248


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l492_49263

-- Define the circles O₁ and O₂ in polar coordinates
def circle_O₁ (ρ θ : ℝ) : Prop := ρ = Real.sin θ
def circle_O₂ (ρ θ : ℝ) : Prop := ρ = Real.cos θ

-- Define the line in Cartesian coordinates
def intersection_line (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), circle_O₁ ρ θ ∧ circle_O₂ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  intersection_line x y :=
sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l492_49263


namespace NUMINAMATH_CALUDE_dog_arrangement_count_l492_49283

theorem dog_arrangement_count : ∀ (n m k : ℕ),
  n = 15 →
  m = 3 →
  k = 12 →
  (Nat.choose k 3) * (Nat.choose (k - 3) 6) = 18480 :=
by
  sorry

end NUMINAMATH_CALUDE_dog_arrangement_count_l492_49283


namespace NUMINAMATH_CALUDE_prime_power_sum_l492_49219

theorem prime_power_sum (a b c d e : ℕ) : 
  2^a * 3^b * 5^c * 7^d * 11^e = 6930 → 2*a + 3*b + 5*c + 7*d + 11*e = 31 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l492_49219


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l492_49231

/-- Given two vectors a and b in ℝ², prove that |2a + b| = 3 -/
theorem magnitude_of_sum (a b : ℝ × ℝ) :
  (‖a‖ = 1) →
  (b = (1, 2)) →
  (a • b = 0) →
  ‖2 • a + b‖ = 3 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_l492_49231


namespace NUMINAMATH_CALUDE_sample_grade12_is_40_l492_49262

/-- Represents the stratified sampling problem for a school with three grades. -/
structure School where
  total_students : ℕ
  grade10_students : ℕ
  grade11_students : ℕ
  sample_size : ℕ

/-- Calculates the number of students to be sampled from grade 12 in a stratified sampling. -/
def sampleGrade12 (s : School) : ℕ :=
  s.sample_size - (s.grade10_students * s.sample_size / s.total_students + s.grade11_students * s.sample_size / s.total_students)

/-- Theorem stating that for the given school parameters, the number of students
    to be sampled from grade 12 is 40. -/
theorem sample_grade12_is_40 (s : School)
    (h1 : s.total_students = 2400)
    (h2 : s.grade10_students = 820)
    (h3 : s.grade11_students = 780)
    (h4 : s.sample_size = 120) :
    sampleGrade12 s = 40 := by
  sorry

end NUMINAMATH_CALUDE_sample_grade12_is_40_l492_49262


namespace NUMINAMATH_CALUDE_pathway_width_l492_49275

theorem pathway_width (r₁ r₂ : ℝ) (h₁ : r₁ > r₂) (h₂ : 2 * π * r₁ - 2 * π * r₂ = 20 * π) : 
  r₁ - r₂ + 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_pathway_width_l492_49275


namespace NUMINAMATH_CALUDE_completing_square_transformation_l492_49254

theorem completing_square_transformation (x : ℝ) :
  x^2 - 8*x - 11 = 0 ↔ (x - 4)^2 = 27 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l492_49254


namespace NUMINAMATH_CALUDE_cricket_average_increase_l492_49284

theorem cricket_average_increase (innings : ℕ) (current_average : ℚ) (next_innings_runs : ℕ) 
  (h1 : innings = 13)
  (h2 : current_average = 22)
  (h3 : next_innings_runs = 92) : 
  let total_runs : ℚ := innings * current_average
  let new_total_runs : ℚ := total_runs + next_innings_runs
  let new_average : ℚ := new_total_runs / (innings + 1)
  new_average - current_average = 5 := by sorry

end NUMINAMATH_CALUDE_cricket_average_increase_l492_49284


namespace NUMINAMATH_CALUDE_table_tennis_tournament_l492_49249

theorem table_tennis_tournament (n : ℕ) (x : ℕ) : 
  n > 3 → 
  Nat.choose (n - 3) 2 + 6 - x = 50 → 
  Nat.choose n 2 = 50 → 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_table_tennis_tournament_l492_49249


namespace NUMINAMATH_CALUDE_prob_three_is_half_l492_49270

/-- The decimal representation of 7/11 -/
def decimal_rep : ℚ := 7 / 11

/-- The repeating sequence in the decimal representation -/
def repeating_sequence : List ℕ := [6, 3]

/-- The probability of selecting a specific digit from the repeating sequence -/
def prob_digit (d : ℕ) : ℚ :=
  (repeating_sequence.count d : ℚ) / repeating_sequence.length

theorem prob_three_is_half :
  prob_digit 3 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_three_is_half_l492_49270


namespace NUMINAMATH_CALUDE_village_households_l492_49258

/-- The number of households in a village where:
    1. Each household requires 20 litres of water per month
    2. 2000 litres of water lasts for 10 months for all households -/
def number_of_households : ℕ := 10

/-- The amount of water required per household per month (in litres) -/
def water_per_household_per_month : ℕ := 20

/-- The total amount of water available (in litres) -/
def total_water : ℕ := 2000

/-- The number of months the water supply lasts -/
def months_supply : ℕ := 10

theorem village_households :
  number_of_households * water_per_household_per_month * months_supply = total_water :=
by sorry

end NUMINAMATH_CALUDE_village_households_l492_49258


namespace NUMINAMATH_CALUDE_blocks_needed_per_color_l492_49280

/-- Represents the dimensions of a clay block -/
structure BlockDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a cylindrical pot -/
structure PotDimensions where
  height : ℝ
  diameter : ℝ

/-- Calculates the number of blocks needed for each color -/
def blocksPerColor (block : BlockDimensions) (pot : PotDimensions) (layerHeight : ℝ) : ℕ :=
  sorry

/-- Theorem stating that 7 blocks of each color are needed -/
theorem blocks_needed_per_color 
  (block : BlockDimensions)
  (pot : PotDimensions)
  (layerHeight : ℝ)
  (h1 : block.length = 4)
  (h2 : block.width = 3)
  (h3 : block.height = 2)
  (h4 : pot.height = 10)
  (h5 : pot.diameter = 5)
  (h6 : layerHeight = 2.5) :
  blocksPerColor block pot layerHeight = 7 := by
  sorry

end NUMINAMATH_CALUDE_blocks_needed_per_color_l492_49280


namespace NUMINAMATH_CALUDE_theresa_work_hours_l492_49276

/-- The average number of hours Theresa needs to work per week -/
def required_average : ℝ := 9

/-- The number of weeks Theresa needs to maintain the average -/
def total_weeks : ℕ := 7

/-- The hours Theresa worked in the first 6 weeks -/
def first_six_weeks : List ℝ := [10, 8, 9, 11, 6, 8]

/-- The sum of hours Theresa worked in the first 6 weeks -/
def sum_first_six : ℝ := first_six_weeks.sum

/-- The number of hours Theresa needs to work in the seventh week -/
def hours_seventh_week : ℝ := 11

theorem theresa_work_hours :
  (sum_first_six + hours_seventh_week) / total_weeks = required_average := by
  sorry

end NUMINAMATH_CALUDE_theresa_work_hours_l492_49276


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l492_49235

/-- Represents a geometric sequence -/
structure GeometricSequence where
  firstTerm : ℝ
  ratio : ℝ

/-- Returns the nth term of a geometric sequence -/
def GeometricSequence.nthTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.firstTerm * seq.ratio ^ (n - 1)

theorem geometric_sequence_first_term
  (seq : GeometricSequence)
  (h3 : seq.nthTerm 3 = 720)
  (h7 : seq.nthTerm 7 = 362880) :
  seq.firstTerm = 20 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l492_49235


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l492_49261

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sequence_term (n : ℕ) : ℕ := factorial n + n

def sum_sequence (n : ℕ) : ℕ := 
  List.sum (List.map sequence_term (List.range n))

theorem units_digit_of_sum : 
  (sum_sequence 10) % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l492_49261


namespace NUMINAMATH_CALUDE_min_wires_for_unit_cube_l492_49214

/-- Represents a piece of wire with a given length -/
structure Wire where
  length : ℕ

/-- Represents a cube -/
structure Cube where
  edgeLength : ℕ
  numEdges : ℕ := 12
  numVertices : ℕ := 8

def availableWires : List Wire := [
  { length := 1 },
  { length := 2 },
  { length := 3 },
  { length := 4 },
  { length := 5 },
  { length := 6 },
  { length := 7 }
]

def targetCube : Cube := { edgeLength := 1 }

/-- Returns the minimum number of wire pieces needed to form the cube -/
def minWiresForCube (wires : List Wire) (cube : Cube) : ℕ := sorry

theorem min_wires_for_unit_cube :
  minWiresForCube availableWires targetCube = 4 := by sorry

end NUMINAMATH_CALUDE_min_wires_for_unit_cube_l492_49214


namespace NUMINAMATH_CALUDE_larger_number_proof_l492_49202

theorem larger_number_proof (x y : ℕ) (h1 : x > y) (h2 : x + y = 830) (h3 : x = 22 * y + 2) : x = 794 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l492_49202


namespace NUMINAMATH_CALUDE_q_polynomial_form_l492_49215

/-- Given a function q(x) satisfying the equation 
    q(x) + (2x^6 + 4x^4 + 12x^2) = (10x^4 + 36x^3 + 37x^2 + 5),
    prove that q(x) = -2x^6 + 6x^4 + 36x^3 + 25x^2 + 5 -/
theorem q_polynomial_form (x : ℝ) (q : ℝ → ℝ) 
    (h : ∀ x, q x + (2*x^6 + 4*x^4 + 12*x^2) = 10*x^4 + 36*x^3 + 37*x^2 + 5) :
  q x = -2*x^6 + 6*x^4 + 36*x^3 + 25*x^2 + 5 := by
  sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l492_49215


namespace NUMINAMATH_CALUDE_prob_one_of_three_wins_l492_49295

/-- The probability that one of three mutually exclusive events occurs is the sum of their individual probabilities -/
theorem prob_one_of_three_wins (pX pY pZ : ℚ) 
  (hX : pX = 1/6) (hY : pY = 1/10) (hZ : pZ = 1/8) : 
  pX + pY + pZ = 47/120 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_of_three_wins_l492_49295


namespace NUMINAMATH_CALUDE_square_diagonal_quadrilateral_l492_49260

/-- Given a square with side length a, this theorem proves the properties of a quadrilateral
    formed by the endpoints of a diagonal and the centers of inscribed circles of the two
    isosceles right triangles created by that diagonal. -/
theorem square_diagonal_quadrilateral (a : ℝ) (h : a > 0) :
  ∃ (perimeter area : ℝ),
    perimeter = 4 * a * Real.sqrt (2 - Real.sqrt 2) ∧
    area = a^2 * (Real.sqrt 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_quadrilateral_l492_49260


namespace NUMINAMATH_CALUDE_function_range_l492_49253

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem function_range :
  {y | ∃ x ∈ domain, f x = y} = {y | 2 ≤ y ∧ y ≤ 6} := by sorry

end NUMINAMATH_CALUDE_function_range_l492_49253


namespace NUMINAMATH_CALUDE_bill_bouquet_profit_l492_49234

/-- Represents the number of roses in a bouquet Bill buys -/
def roses_per_bought_bouquet : ℕ := 7

/-- Represents the number of roses in a bouquet Bill sells -/
def roses_per_sold_bouquet : ℕ := 5

/-- Represents the price of a bouquet (both buying and selling) in dollars -/
def price_per_bouquet : ℕ := 20

/-- Represents the target profit in dollars -/
def target_profit : ℕ := 1000

/-- Calculates the number of bouquets Bill needs to buy to earn the target profit -/
def bouquets_to_buy : ℕ :=
  let bought_bouquets_per_operation := roses_per_sold_bouquet
  let sold_bouquets_per_operation := roses_per_bought_bouquet
  let profit_per_operation := sold_bouquets_per_operation * price_per_bouquet - bought_bouquets_per_operation * price_per_bouquet
  let operations_needed := target_profit / profit_per_operation
  operations_needed * bought_bouquets_per_operation

theorem bill_bouquet_profit :
  bouquets_to_buy = 125 := by sorry

end NUMINAMATH_CALUDE_bill_bouquet_profit_l492_49234


namespace NUMINAMATH_CALUDE_hyperbola_specific_equation_l492_49220

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  /-- The distance from the center to a focus -/
  c : ℝ
  /-- The slope of the asymptotes -/
  m : ℝ

/-- The equation of the hyperbola given its parameters -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / (h.c^2 / (1 + h.m^2)) - y^2 / (h.c^2 * h.m^2 / (1 + h.m^2)) = 1

theorem hyperbola_specific_equation :
  let h : Hyperbola := ⟨5, 3/4⟩
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2/16 - y^2/9 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_specific_equation_l492_49220


namespace NUMINAMATH_CALUDE_distance_on_quadratic_curve_l492_49211

/-- The distance between two points on a quadratic curve -/
theorem distance_on_quadratic_curve (m k a c : ℝ) :
  let y (x : ℝ) := m * x^2 + k
  let point1 := (a, y a)
  let point2 := (c, y c)
  let distance := Real.sqrt ((c - a)^2 + (y c - y a)^2)
  distance = |a - c| * Real.sqrt (1 + m^2 * (c + a)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_quadratic_curve_l492_49211


namespace NUMINAMATH_CALUDE_trajectory_and_fixed_points_l492_49243

-- Define the points and lines
def F : ℝ × ℝ := (1, 0)
def H : ℝ × ℝ := (1, 2)
def l : Set (ℝ × ℝ) := {p | p.1 = -1}

-- Define the trajectory C
def C : Set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

-- Define the function for the circle with MN as diameter
def circle_MN (m : ℝ) : Set (ℝ × ℝ) := 
  {p | p.1^2 + 2*p.1 - 3 + p.2^2 + (4/m)*p.2 = 0}

-- Theorem statement
theorem trajectory_and_fixed_points :
  -- Part 1: Trajectory C
  (∀ Q : ℝ × ℝ, (∃ P : ℝ × ℝ, P ∈ l ∧ 
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (Q.1 - F.1)^2 + (Q.2 - F.2)^2) 
    → Q ∈ C) ∧
  -- Part 2: Fixed points on the circle
  (∀ m : ℝ, m ≠ 0 → (-3, 0) ∈ circle_MN m ∧ (1, 0) ∈ circle_MN m) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_fixed_points_l492_49243


namespace NUMINAMATH_CALUDE_dust_retention_proof_l492_49273

/-- The average annual dust retention of a locust leaf in milligrams. -/
def locust_dust_retention : ℝ := 22

/-- The average annual dust retention of a ginkgo leaf in milligrams. -/
def ginkgo_dust_retention : ℝ := 2 * locust_dust_retention - 4

theorem dust_retention_proof :
  11 * ginkgo_dust_retention = 20 * locust_dust_retention :=
by sorry

end NUMINAMATH_CALUDE_dust_retention_proof_l492_49273


namespace NUMINAMATH_CALUDE_circle_equation_l492_49237

/-- Theorem: The equation of a circle with center (1, -1) and radius 2 is (x-1)^2 + (y+1)^2 = 4 -/
theorem circle_equation (x y : ℝ) : 
  (∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (1, -1) ∧ 
    radius = 2 ∧ 
    ((x - center.1)^2 + (y - center.2)^2 = radius^2)) ↔ 
  ((x - 1)^2 + (y + 1)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l492_49237


namespace NUMINAMATH_CALUDE_min_value_shifted_l492_49294

/-- A quadratic function f(x) with a minimum value of 2 -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + 5 - c

/-- The function g(x) which is f(x-2015) -/
def g (c : ℝ) (x : ℝ) : ℝ := f c (x - 2015)

theorem min_value_shifted (c : ℝ) (h : ∃ (m : ℝ), ∀ (x : ℝ), f c x ≥ m ∧ ∃ (x₀ : ℝ), f c x₀ = m) 
  (hmin : ∃ (x₀ : ℝ), f c x₀ = 2) :
  ∃ (m : ℝ), ∀ (x : ℝ), g c x ≥ m ∧ ∃ (x₀ : ℝ), g c x₀ = m ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_shifted_l492_49294


namespace NUMINAMATH_CALUDE_hyperbola_parameter_sum_l492_49268

/-- Theorem about the sum of parameters for a specific hyperbola -/
theorem hyperbola_parameter_sum :
  let center : ℝ × ℝ := (1, 3)
  let focus : ℝ × ℝ := (1, 9)
  let vertex : ℝ × ℝ := (1, 0)
  let h : ℝ := center.1
  let k : ℝ := center.2
  let a : ℝ := |k - vertex.2|
  let c : ℝ := |k - focus.2|
  let b : ℝ := Real.sqrt (c^2 - a^2)
  h + k + a + b = 7 + 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_parameter_sum_l492_49268


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l492_49210

theorem sin_2alpha_value (α : Real) :
  2 * Real.cos (2 * α) = Real.sin (π / 4 - α) →
  Real.sin (2 * α) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l492_49210


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l492_49225

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_monotone : is_monotone_increasing_on_nonneg f) :
  {a : ℝ | f 1 < f a} = {a : ℝ | a < -1 ∨ 1 < a} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l492_49225


namespace NUMINAMATH_CALUDE_jar_water_problem_l492_49297

theorem jar_water_problem (capacity_x : ℝ) (capacity_y : ℝ) 
  (h1 : capacity_y = (1 / 2) * capacity_x) 
  (h2 : capacity_x > 0) :
  let initial_water_x := (1 / 2) * capacity_x
  let initial_water_y := (1 / 2) * capacity_y
  let final_water_x := initial_water_x + initial_water_y
  final_water_x = (3 / 4) * capacity_x := by
sorry

end NUMINAMATH_CALUDE_jar_water_problem_l492_49297


namespace NUMINAMATH_CALUDE_unique_six_digit_reverse_multiple_l492_49228

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.reverse.foldl (λ acc d => acc * 10 + d) 0

theorem unique_six_digit_reverse_multiple : 
  ∃! n : ℕ, is_six_digit n ∧ n * 9 = reverse_digits n :=
by sorry

end NUMINAMATH_CALUDE_unique_six_digit_reverse_multiple_l492_49228


namespace NUMINAMATH_CALUDE_vector_simplification_l492_49205

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification 
  (A B C D : V) 
  (h1 : A - C = A - B - (C - B)) 
  (h2 : B - D = B - C - (D - C)) : 
  A - C - (B - D) + (C - D) - (A - B) = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_simplification_l492_49205


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l492_49227

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (1 / a + 1 / b + 1 / c) ≥ 3 := by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l492_49227


namespace NUMINAMATH_CALUDE_min_production_avoids_loss_min_production_is_minimal_l492_49290

/-- The daily production cost function for a shoe factory -/
def cost (n : ℕ) : ℝ := 4000 + 50 * n

/-- The daily revenue function for a shoe factory -/
def revenue (n : ℕ) : ℝ := 90 * n

/-- The daily profit function for a shoe factory -/
def profit (n : ℕ) : ℝ := revenue n - cost n

/-- The minimum number of pairs of shoes that must be produced daily to avoid loss -/
def min_production : ℕ := 100

theorem min_production_avoids_loss :
  ∀ n : ℕ, n ≥ min_production → profit n ≥ 0 :=
sorry

theorem min_production_is_minimal :
  ∀ m : ℕ, (∀ n : ℕ, n ≥ m → profit n ≥ 0) → m ≥ min_production :=
sorry

end NUMINAMATH_CALUDE_min_production_avoids_loss_min_production_is_minimal_l492_49290


namespace NUMINAMATH_CALUDE_max_distinct_values_exists_four_valued_function_l492_49240

/-- A function that assigns a number to each vector in space -/
def VectorFunction (n : ℕ) := (Fin n → ℝ) → ℝ

/-- The property of the vector function as described in the problem -/
def HasMaxProperty (n : ℕ) (f : VectorFunction n) : Prop :=
  ∀ (u v : Fin n → ℝ) (α β : ℝ), 
    f (fun i => α * u i + β * v i) ≤ max (f u) (f v)

/-- The theorem stating that a function with the given property can take at most 4 distinct values -/
theorem max_distinct_values (n : ℕ) (f : VectorFunction n) 
    (h : HasMaxProperty n f) : 
    ∃ (S : Finset ℝ), (∀ v, f v ∈ S) ∧ Finset.card S ≤ 4 := by
  sorry

/-- The theorem stating that there exists a function taking exactly 4 distinct values -/
theorem exists_four_valued_function : 
    ∃ (f : VectorFunction 3), HasMaxProperty 3 f ∧ 
      ∃ (S : Finset ℝ), (∀ v, f v ∈ S) ∧ Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_distinct_values_exists_four_valued_function_l492_49240


namespace NUMINAMATH_CALUDE_intersection_curve_length_theorem_l492_49246

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  vertex : Point3D
  edge_length : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The length of the curve formed by the intersection of a unit cube's surface
    and a sphere centered at one of the cube's vertices with radius 2√3/3 -/
def intersection_curve_length (c : Cube) (s : Sphere) : ℝ := sorry

/-- Main theorem statement -/
theorem intersection_curve_length_theorem (c : Cube) (s : Sphere) :
  c.edge_length = 1 ∧
  s.center = c.vertex ∧
  s.radius = 2 * Real.sqrt 3 / 3 →
  intersection_curve_length c s = 5 * Real.sqrt 3 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_curve_length_theorem_l492_49246


namespace NUMINAMATH_CALUDE_largest_common_number_l492_49286

/-- First sequence with initial term 5 and common difference 9 -/
def sequence1 (n : ℕ) : ℕ := 5 + 9 * n

/-- Second sequence with initial term 3 and common difference 8 -/
def sequence2 (m : ℕ) : ℕ := 3 + 8 * m

/-- Theorem stating that 167 is the largest common number in both sequences within the range 1 to 200 -/
theorem largest_common_number :
  ∃ (n m : ℕ),
    sequence1 n = sequence2 m ∧
    sequence1 n = 167 ∧
    sequence1 n ≤ 200 ∧
    ∀ (k l : ℕ), sequence1 k = sequence2 l → sequence1 k ≤ 200 → sequence1 k ≤ 167 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_number_l492_49286


namespace NUMINAMATH_CALUDE_remaining_time_for_finger_exerciser_l492_49292

theorem remaining_time_for_finger_exerciser 
  (total_time : Nat) 
  (piano_time : Nat) 
  (writing_time : Nat) 
  (reading_time : Nat) 
  (h1 : total_time = 120)
  (h2 : piano_time = 30)
  (h3 : writing_time = 25)
  (h4 : reading_time = 38) :
  total_time - (piano_time + writing_time + reading_time) = 27 := by
  sorry

end NUMINAMATH_CALUDE_remaining_time_for_finger_exerciser_l492_49292


namespace NUMINAMATH_CALUDE_sequence_can_be_arithmetic_and_geometric_l492_49278

theorem sequence_can_be_arithmetic_and_geometric :
  ∃ (a d : ℝ) (n : ℕ), a + d = 9 ∧ a + n * d = 729 ∧ a = 3 ∧
  ∃ (b r : ℝ) (m : ℕ), b * r = 9 ∧ b * r^m = 729 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_can_be_arithmetic_and_geometric_l492_49278


namespace NUMINAMATH_CALUDE_integer_product_sum_l492_49271

theorem integer_product_sum (x y : ℤ) : y = x + 2 ∧ x * y = 644 → x + y = 50 := by
  sorry

end NUMINAMATH_CALUDE_integer_product_sum_l492_49271


namespace NUMINAMATH_CALUDE_total_games_is_506_l492_49216

/-- Represents the structure of a soccer league --/
structure SoccerLeague where
  total_teams : Nat
  divisions : Nat
  teams_per_division : Nat
  regular_season_intra_division_matches : Nat
  regular_season_cross_division_matches : Nat
  mid_season_tournament_matches_per_team : Nat
  mid_season_tournament_intra_division_matches : Nat
  playoff_teams_per_division : Nat
  playoff_stages : Nat

/-- Calculates the total number of games in the soccer league season --/
def total_games (league : SoccerLeague) : Nat :=
  -- Regular season games
  (league.teams_per_division * (league.teams_per_division - 1) * league.regular_season_intra_division_matches * league.divisions) / 2 +
  (league.total_teams * league.regular_season_cross_division_matches) / 2 +
  -- Mid-season tournament games
  (league.total_teams * league.mid_season_tournament_matches_per_team) / 2 +
  -- Playoff games
  (league.playoff_teams_per_division * 2 - 1) * 2 * league.divisions

/-- Theorem stating that the total number of games in the given league structure is 506 --/
theorem total_games_is_506 (league : SoccerLeague) 
  (h1 : league.total_teams = 24)
  (h2 : league.divisions = 2)
  (h3 : league.teams_per_division = 12)
  (h4 : league.regular_season_intra_division_matches = 2)
  (h5 : league.regular_season_cross_division_matches = 4)
  (h6 : league.mid_season_tournament_matches_per_team = 5)
  (h7 : league.mid_season_tournament_intra_division_matches = 3)
  (h8 : league.playoff_teams_per_division = 4)
  (h9 : league.playoff_stages = 3) :
  total_games league = 506 := by
  sorry

end NUMINAMATH_CALUDE_total_games_is_506_l492_49216


namespace NUMINAMATH_CALUDE_allie_billie_meeting_l492_49279

/-- The distance Allie skates before meeting Billie -/
def allie_distance (ab_distance : ℝ) (allie_speed billie_speed : ℝ) (allie_angle : ℝ) : ℝ :=
  let x := 160
  x

theorem allie_billie_meeting 
  (ab_distance : ℝ) 
  (allie_speed billie_speed : ℝ) 
  (allie_angle : ℝ) 
  (h1 : ab_distance = 100)
  (h2 : allie_speed = 8)
  (h3 : billie_speed = 7)
  (h4 : allie_angle = 60 * π / 180)
  (h5 : ∀ x, x > 0 → x ≠ 160 → 
    (x / allie_speed ≠ 
    Real.sqrt (x^2 + ab_distance^2 - 2 * x * ab_distance * Real.cos allie_angle) / billie_speed)) :
  allie_distance ab_distance allie_speed billie_speed allie_angle = 160 := by
  sorry

end NUMINAMATH_CALUDE_allie_billie_meeting_l492_49279


namespace NUMINAMATH_CALUDE_cakes_served_today_l492_49242

theorem cakes_served_today (lunch_cakes dinner_cakes : ℕ) 
  (h1 : lunch_cakes = 6) (h2 : dinner_cakes = 9) : 
  lunch_cakes + dinner_cakes = 15 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_today_l492_49242


namespace NUMINAMATH_CALUDE_sqrt_x_minus_two_real_l492_49264

theorem sqrt_x_minus_two_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_two_real_l492_49264


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l492_49241

theorem polynomial_expansion_equality (x : ℝ) :
  (3*x^2 + 4*x + 8)*(x - 2) - (x - 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x - 2)*(x + 6) =
  6*x^3 - 4*x^2 - 26*x + 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l492_49241


namespace NUMINAMATH_CALUDE_sharona_bought_four_more_pencils_l492_49269

/-- The price of a single pencil in cents -/
def pencil_price : ℕ := 11

/-- The number of pencils Jamar bought -/
def jamar_pencils : ℕ := 13

/-- The number of pencils Sharona bought -/
def sharona_pencils : ℕ := 17

/-- The amount Jamar paid in cents -/
def jamar_paid : ℕ := 143

/-- The amount Sharona paid in cents -/
def sharona_paid : ℕ := 187

theorem sharona_bought_four_more_pencils :
  pencil_price > 1 ∧
  pencil_price * jamar_pencils = jamar_paid ∧
  pencil_price * sharona_pencils = sharona_paid →
  sharona_pencils - jamar_pencils = 4 :=
by sorry

end NUMINAMATH_CALUDE_sharona_bought_four_more_pencils_l492_49269


namespace NUMINAMATH_CALUDE_max_fall_time_bound_l492_49229

/-- Represents the movement rules and conditions for ants on an m × m checkerboard. -/
structure AntCheckerboard (m : ℕ) :=
  (m_pos : m > 0)
  (board_size : Fin m → Fin m → Bool)
  (ant_positions : Set (Fin m × Fin m))
  (ant_directions : (Fin m × Fin m) → (Int × Int))
  (collision_rules : (Fin m × Fin m) → (Int × Int) → (Int × Int))

/-- The maximum time for the last ant to fall off the board. -/
def max_fall_time (m : ℕ) (board : AntCheckerboard m) : ℚ :=
  3 * m / 2 - 1

/-- Theorem stating that the maximum time for the last ant to fall off is 3m/2 - 1. -/
theorem max_fall_time_bound (m : ℕ) (board : AntCheckerboard m) :
  ∀ (t : ℚ), (∃ (ant : Fin m × Fin m), ant ∈ board.ant_positions) →
  t ≤ max_fall_time m board :=
sorry

end NUMINAMATH_CALUDE_max_fall_time_bound_l492_49229


namespace NUMINAMATH_CALUDE_solution_set_nonempty_iff_m_in_range_l492_49291

open Set

theorem solution_set_nonempty_iff_m_in_range (m : ℝ) :
  (∃ x : ℝ, |x - m| + |x + 2| < 4) ↔ m ∈ Ioo (-6) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_nonempty_iff_m_in_range_l492_49291


namespace NUMINAMATH_CALUDE_base_subtraction_equality_l492_49222

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their original bases
def num1 : List Nat := [5, 2, 3]  -- 325 in base 6 (reversed for easier conversion)
def num2 : List Nat := [1, 3, 2]  -- 231 in base 5 (reversed for easier conversion)

-- State the theorem
theorem base_subtraction_equality :
  to_base_10 num1 6 - to_base_10 num2 5 = 59 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_equality_l492_49222


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l492_49255

/-- The complex number z is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

/-- Given that (2-ai)/(1+i) is pure imaginary and a is real, prove that a = 2 -/
theorem complex_pure_imaginary_condition (a : ℝ) 
  (h : isPureImaginary ((2 - a * Complex.I) / (1 + Complex.I))) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l492_49255


namespace NUMINAMATH_CALUDE_actual_weight_calculation_l492_49218

/-- The dealer's percent -/
def dealer_percent : ℝ := 53.84615384615387

/-- The actual weight used per kg -/
def actual_weight : ℝ := 0.4615384615384613

/-- Theorem stating that the actual weight used per kg is correct given the dealer's percent -/
theorem actual_weight_calculation (ε : ℝ) (h : ε > 0) : 
  |actual_weight - (1 - dealer_percent / 100)| < ε :=
by sorry

end NUMINAMATH_CALUDE_actual_weight_calculation_l492_49218


namespace NUMINAMATH_CALUDE_nick_pennsylvania_quarters_l492_49230

/-- Given a total number of quarters, calculate the number of Pennsylvania state quarters -/
def pennsylvania_quarters (total : ℕ) : ℕ :=
  let state_quarters := (2 * total) / 5
  (state_quarters / 2 : ℕ)

theorem nick_pennsylvania_quarters :
  pennsylvania_quarters 35 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nick_pennsylvania_quarters_l492_49230


namespace NUMINAMATH_CALUDE_cos_two_alpha_l492_49282

theorem cos_two_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin (α - π / 4) = 1 / 3) : 
  Real.cos (2 * α) = -4 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_l492_49282


namespace NUMINAMATH_CALUDE_halfway_point_l492_49217

theorem halfway_point (a b : ℚ) (ha : a = 1/8) (hb : b = 3/10) :
  (a + b) / 2 = 17/80 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_l492_49217


namespace NUMINAMATH_CALUDE_inequality_solution_set_l492_49201

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 + 5*x - 6 > 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -6 ∨ x > 1}

-- Theorem statement
theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l492_49201


namespace NUMINAMATH_CALUDE_reggie_loses_by_21_points_l492_49293

/-- Represents the types of basketball shots -/
inductive ShotType
  | Layup
  | FreeThrow
  | ThreePointer
  | HalfCourt

/-- Returns the point value for a given shot type -/
def pointValue (shot : ShotType) : ℕ :=
  match shot with
  | ShotType.Layup => 1
  | ShotType.FreeThrow => 2
  | ShotType.ThreePointer => 3
  | ShotType.HalfCourt => 5

/-- Calculates the total points for a set of shots -/
def totalPoints (layups freeThrows threePointers halfCourt : ℕ) : ℕ :=
  layups * pointValue ShotType.Layup +
  freeThrows * pointValue ShotType.FreeThrow +
  threePointers * pointValue ShotType.ThreePointer +
  halfCourt * pointValue ShotType.HalfCourt

/-- Theorem stating the difference in points between Reggie's brother and Reggie -/
theorem reggie_loses_by_21_points :
  totalPoints 3 2 5 4 - totalPoints 4 3 2 1 = 21 := by
  sorry

#eval totalPoints 3 2 5 4 - totalPoints 4 3 2 1

end NUMINAMATH_CALUDE_reggie_loses_by_21_points_l492_49293


namespace NUMINAMATH_CALUDE_pencils_per_box_l492_49233

theorem pencils_per_box (total_boxes : ℕ) (kept_pencils : ℕ) (num_friends : ℕ) (pencils_per_friend : ℕ) : 
  total_boxes = 10 →
  kept_pencils = 10 →
  num_friends = 5 →
  pencils_per_friend = 8 →
  (total_boxes * (kept_pencils + num_friends * pencils_per_friend)) / total_boxes = 5 :=
by sorry

end NUMINAMATH_CALUDE_pencils_per_box_l492_49233


namespace NUMINAMATH_CALUDE_factorial_8_divisors_l492_49257

-- Define 8!
def factorial_8 : ℕ := 8*7*6*5*4*3*2*1

-- Define a function to count positive divisors
def count_positive_divisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem factorial_8_divisors : count_positive_divisors factorial_8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_factorial_8_divisors_l492_49257


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l492_49245

theorem largest_integer_with_remainder : ∃ n : ℕ, 
  (n < 120) ∧ 
  (n % 8 = 7) ∧ 
  (∀ m : ℕ, m < 120 ∧ m % 8 = 7 → m ≤ n) ∧
  (n = 119) := by
sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l492_49245


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l492_49274

theorem sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((1/2 : ℝ)^a < (1/2 : ℝ)^b → Real.log (a + 1) > Real.log b) ∧
  ¬(Real.log (a + 1) > Real.log b → (1/2 : ℝ)^a < (1/2 : ℝ)^b) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l492_49274


namespace NUMINAMATH_CALUDE_parabola_directrix_l492_49289

/-- Given a parabola y = ax^2 with directrix y = -2, prove that a = 1/8 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 ∧ y = -2 → a = 1/8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l492_49289


namespace NUMINAMATH_CALUDE_other_number_is_25_l492_49209

theorem other_number_is_25 (x y : ℤ) : 
  (3 * x + 4 * y + 2 * x = 160) → 
  ((x = 12 ∧ y ≠ 12) ∨ (y = 12 ∧ x ≠ 12)) → 
  (x = 25 ∨ y = 25) := by
sorry

end NUMINAMATH_CALUDE_other_number_is_25_l492_49209


namespace NUMINAMATH_CALUDE_remainder_problem_l492_49277

theorem remainder_problem (R : ℕ) : 
  (29 = Nat.gcd (1255 - 8) (1490 - R)) →
  (1255 % 29 = 8) →
  (1490 % 29 = R) →
  R = 11 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l492_49277


namespace NUMINAMATH_CALUDE_minimize_y_l492_49244

/-- The function y in terms of x, a, b, and c -/
def y (x a b c : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + c * x

/-- The theorem stating that (a + b - c/2) / 2 minimizes y -/
theorem minimize_y (a b c : ℝ) :
  ∃ (x : ℝ), ∀ (z : ℝ), y z a b c ≥ y ((a + b - c/2) / 2) a b c :=
sorry

end NUMINAMATH_CALUDE_minimize_y_l492_49244


namespace NUMINAMATH_CALUDE_hyperbola_equation_l492_49221

/-- Given an ellipse and a hyperbola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (x y : ℝ) :
  -- Given ellipse equation
  (x^2 / 144 + y^2 / 169 = 1) →
  -- Hyperbola passes through (0, 2)
  (∃ (a b : ℝ), y^2 / a^2 - x^2 / b^2 = 1 ∧ 2^2 / a^2 - 0^2 / b^2 = 1) →
  -- Hyperbola shares a common focus with the ellipse
  (∃ (c : ℝ), c^2 = 169 - 144 ∧ c^2 = a^2 + b^2) →
  -- Prove the equation of the hyperbola
  (y^2 / 4 - x^2 / 21 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l492_49221


namespace NUMINAMATH_CALUDE_g_composition_half_l492_49247

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

-- State the theorem
theorem g_composition_half : g (g (1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_half_l492_49247


namespace NUMINAMATH_CALUDE_star_problem_l492_49206

-- Define the ⭐ operation
def star (x y : ℚ) : ℚ := (x + y) / 4

-- Theorem statement
theorem star_problem : star (star 3 9) 4 = 7 / 4 := by sorry

end NUMINAMATH_CALUDE_star_problem_l492_49206


namespace NUMINAMATH_CALUDE_layla_score_comparison_l492_49265

/-- Represents a player in the game -/
inductive Player : Type
| Layla : Player
| Nahima : Player
| Ramon : Player
| Aria : Player

/-- Represents a round in the game -/
inductive Round : Type
| First : Round
| Second : Round
| Third : Round

/-- The scoring function for the game -/
def score (p : Player) (r : Round) : ℕ → ℕ :=
  match r with
  | Round.First => (· * 2)
  | Round.Second => (· * 3)
  | Round.Third => id

/-- The total score of a player across all rounds -/
def totalScore (p : Player) (s1 s2 s3 : ℕ) : ℕ :=
  score p Round.First s1 + score p Round.Second s2 + score p Round.Third s3

theorem layla_score_comparison :
  ∀ (nahima_total ramon_total aria_total : ℕ),
  totalScore Player.Layla 120 90 (760 - score Player.Layla Round.First 120 - score Player.Layla Round.Second 90) = 760 →
  nahima_total + ramon_total + aria_total = 1330 - 760 →
  760 - score Player.Layla Round.First 120 - score Player.Layla Round.Second 90 =
    nahima_total + ramon_total + aria_total - 320 :=
by sorry

end NUMINAMATH_CALUDE_layla_score_comparison_l492_49265


namespace NUMINAMATH_CALUDE_cat_weight_difference_l492_49208

/-- Given the weights of two cats belonging to Meg and Anne, prove the weight difference --/
theorem cat_weight_difference 
  (weight_meg : ℝ) 
  (weight_anne : ℝ) 
  (h1 : weight_meg / weight_anne = 13 / 21)
  (h2 : weight_meg = 20 + 0.5 * weight_anne) :
  weight_anne - weight_meg = 64 := by
  sorry

end NUMINAMATH_CALUDE_cat_weight_difference_l492_49208


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l492_49250

/-- Given a grocery store inventory, calculate the number of diet soda bottles. -/
theorem diet_soda_bottles (total_bottles regular_soda_bottles : ℕ) 
  (h1 : total_bottles = 17)
  (h2 : regular_soda_bottles = 9) :
  total_bottles - regular_soda_bottles = 8 := by
  sorry

#check diet_soda_bottles

end NUMINAMATH_CALUDE_diet_soda_bottles_l492_49250


namespace NUMINAMATH_CALUDE_disjoint_subsets_prime_products_l492_49256

/-- A function that constructs 100 disjoint subsets of positive integers -/
def construct_subsets : Fin 100 → Set ℕ := sorry

/-- Predicate to check if a number is a product of m distinct primes from a set -/
def is_product_of_m_primes (n : ℕ) (m : ℕ) (S : Set ℕ) : Prop := sorry

/-- Main theorem statement -/
theorem disjoint_subsets_prime_products :
  ∃ (A : Fin 100 → Set ℕ), 
    (∀ i j, i ≠ j → Disjoint (A i) (A j)) ∧
    (∀ (S : Set ℕ) (hS : Set.Infinite S) (h_prime : ∀ p ∈ S, Nat.Prime p),
      ∃ (m : ℕ) (a : Fin 100 → ℕ), 
        ∀ i, a i ∈ A i ∧ is_product_of_m_primes (a i) m S) :=
sorry

end NUMINAMATH_CALUDE_disjoint_subsets_prime_products_l492_49256


namespace NUMINAMATH_CALUDE_ternary_2101211_equals_octal_444_l492_49238

/-- Converts a ternary number represented as a list of digits to its decimal value. -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Converts a decimal number to its octal representation as a list of digits. -/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Theorem stating that the ternary number 2101211 is equal to the octal number 444. -/
theorem ternary_2101211_equals_octal_444 :
  decimal_to_octal (ternary_to_decimal [1, 1, 2, 1, 0, 1, 2]) = [4, 4, 4] := by
  sorry

#eval ternary_to_decimal [1, 1, 2, 1, 0, 1, 2]
#eval decimal_to_octal (ternary_to_decimal [1, 1, 2, 1, 0, 1, 2])

end NUMINAMATH_CALUDE_ternary_2101211_equals_octal_444_l492_49238


namespace NUMINAMATH_CALUDE_parabola_axis_distance_l492_49272

/-- Given a parabola x^2 = ay, if the distance from the point (0,1) to its axis of symmetry is 2, then a = -12 or a = 4. -/
theorem parabola_axis_distance (a : ℝ) : 
  (∀ x y : ℝ, x^2 = a*y → 
    (|y - 1 - (-a/4)| = 2 ↔ (a = -12 ∨ a = 4))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_distance_l492_49272


namespace NUMINAMATH_CALUDE_opposite_of_five_l492_49239

theorem opposite_of_five : -(5 : ℤ) = -5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_five_l492_49239


namespace NUMINAMATH_CALUDE_obstacle_course_time_l492_49212

/-- Represents the times for each segment of the obstacle course -/
structure ObstacleCourse :=
  (first_run : List Int)
  (door_opening : Int)
  (second_run : List Int)

/-- Calculates the total time to complete the obstacle course -/
def total_time (course : ObstacleCourse) : Int :=
  (course.first_run.sum + course.door_opening + course.second_run.sum)

/-- The theorem to prove -/
theorem obstacle_course_time :
  let course := ObstacleCourse.mk [225, 130, 88, 45, 120] 73 [175, 108, 75, 138]
  total_time course = 1177 := by
  sorry

end NUMINAMATH_CALUDE_obstacle_course_time_l492_49212


namespace NUMINAMATH_CALUDE_prize_problem_solution_l492_49266

/-- Represents the prices and quantities of notebooks and pens -/
structure PrizeInfo where
  notebook_price : ℕ
  pen_price : ℕ
  notebook_quantity : ℕ
  pen_quantity : ℕ

/-- Theorem stating the solution to the prize problem -/
theorem prize_problem_solution :
  ∃ (info : PrizeInfo),
    -- Each notebook costs 3 yuan more than each pen
    info.notebook_price = info.pen_price + 3 ∧
    -- The number of notebooks purchased for 390 yuan is the same as the number of pens purchased for 300 yuan
    390 / info.notebook_price = 300 / info.pen_price ∧
    -- The total cost of purchasing prizes for 50 students should not exceed 560 yuan
    info.notebook_quantity + info.pen_quantity = 50 ∧
    info.notebook_price * info.notebook_quantity + info.pen_price * info.pen_quantity ≤ 560 ∧
    -- The notebook price is 13 yuan
    info.notebook_price = 13 ∧
    -- The pen price is 10 yuan
    info.pen_price = 10 ∧
    -- The maximum number of notebooks that can be purchased is 20
    info.notebook_quantity = 20 ∧
    -- This is the maximum possible number of notebooks
    ∀ (other_info : PrizeInfo),
      other_info.notebook_price = other_info.pen_price + 3 →
      other_info.notebook_quantity + other_info.pen_quantity = 50 →
      other_info.notebook_price * other_info.notebook_quantity + other_info.pen_price * other_info.pen_quantity ≤ 560 →
      other_info.notebook_quantity ≤ info.notebook_quantity :=
by
  sorry


end NUMINAMATH_CALUDE_prize_problem_solution_l492_49266


namespace NUMINAMATH_CALUDE_triangle_on_parabola_bc_length_l492_49285

/-- Parabola function -/
def parabola (x : ℝ) : ℝ := x^2

/-- Triangle ABC -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Check if a point lies on the parabola -/
def onParabola (p : ℝ × ℝ) : Prop :=
  p.2 = parabola p.1

/-- Check if two points have the same y-coordinate (i.e., line is parallel to x-axis) -/
def parallelToXAxis (p q : ℝ × ℝ) : Prop :=
  p.2 = q.2

/-- Calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Calculate the length of a line segment -/
noncomputable def segmentLength (p q : ℝ × ℝ) : ℝ :=
  sorry

/-- Main theorem -/
theorem triangle_on_parabola_bc_length (t : Triangle) :
  onParabola t.A ∧ onParabola t.B ∧ onParabola t.C ∧
  t.A = (1, 1) ∧
  parallelToXAxis t.B t.C ∧
  triangleArea t = 50 →
  ∃ ε > 0, |segmentLength t.B t.C - 5.8| < ε :=
sorry

end NUMINAMATH_CALUDE_triangle_on_parabola_bc_length_l492_49285


namespace NUMINAMATH_CALUDE_soda_cost_l492_49232

/-- The cost of items in cents -/
structure Cost where
  burger : ℕ
  soda : ℕ

/-- The given conditions of the problem -/
def problem_conditions (c : Cost) : Prop :=
  2 * c.burger + c.soda = 210 ∧ c.burger + 2 * c.soda = 240

/-- The theorem stating that under the given conditions, a soda costs 90 cents -/
theorem soda_cost (c : Cost) : problem_conditions c → c.soda = 90 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l492_49232


namespace NUMINAMATH_CALUDE_second_frog_hops_l492_49226

/-- Represents the number of hops taken by each frog -/
structure FrogHops :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)

/-- The conditions of the frog hopping problem -/
def frog_problem (hops : FrogHops) : Prop :=
  hops.first = 4 * hops.second ∧
  hops.second = 2 * hops.third ∧
  hops.fourth = 3 * hops.second ∧
  hops.first + hops.second + hops.third + hops.fourth = 156 ∧
  60 ≤ 120  -- represents the time constraint (60 meters in 2 minutes or less)

/-- The theorem stating that the second frog takes 18 hops -/
theorem second_frog_hops :
  ∃ (hops : FrogHops), frog_problem hops ∧ hops.second = 18 :=
by sorry

end NUMINAMATH_CALUDE_second_frog_hops_l492_49226


namespace NUMINAMATH_CALUDE_complex_equation_solution_l492_49288

theorem complex_equation_solution (z : ℂ) :
  (Complex.I * Real.sqrt 3 + 3 * Complex.I) * z = Complex.I * Real.sqrt 3 →
  z = (Real.sqrt 3 / 4 : ℂ) + (Complex.I / 4) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l492_49288
