import Mathlib

namespace NUMINAMATH_CALUDE_matrix_product_is_zero_l941_94130

def A (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, d, -e;
    -d, 0, f;
    e, -f, 0]

def B (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![d^2, d*e, d*f;
    d*e, e^2, e*f;
    d*f, e*f, f^2]

theorem matrix_product_is_zero (d e f : ℝ) :
  A d e f * B d e f = 0 := by sorry

end NUMINAMATH_CALUDE_matrix_product_is_zero_l941_94130


namespace NUMINAMATH_CALUDE_removed_triangles_area_l941_94112

/-- Given a square with side length x and isosceles right triangles removed from each corner to form a rectangle with perimeter 32, the combined area of the four removed triangles is x²/2. -/
theorem removed_triangles_area (x : ℝ) (r s : ℝ) : 
  x > 0 → 
  2 * (r + s) + 2 * |r - s| = 32 → 
  (r + s)^2 + (r - s)^2 = x^2 → 
  2 * r * s = x^2 / 2 :=
by sorry

#check removed_triangles_area

end NUMINAMATH_CALUDE_removed_triangles_area_l941_94112


namespace NUMINAMATH_CALUDE_wednesday_fraction_is_one_fourth_l941_94111

/-- Represents the daily fabric delivery and earnings for a textile company. -/
structure TextileDelivery where
  monday_yards : ℕ
  tuesday_multiplier : ℕ
  fabric_cost : ℕ
  total_earnings : ℕ

/-- Calculates the fraction of fabric delivered on Wednesday compared to Tuesday. -/
def wednesday_fraction (d : TextileDelivery) : ℚ :=
  let monday_earnings := d.monday_yards * d.fabric_cost
  let tuesday_yards := d.monday_yards * d.tuesday_multiplier
  let tuesday_earnings := tuesday_yards * d.fabric_cost
  let wednesday_earnings := d.total_earnings - monday_earnings - tuesday_earnings
  let wednesday_yards := wednesday_earnings / d.fabric_cost
  wednesday_yards / tuesday_yards

/-- Theorem stating that the fraction of fabric delivered on Wednesday compared to Tuesday is 1/4. -/
theorem wednesday_fraction_is_one_fourth (d : TextileDelivery) 
    (h1 : d.monday_yards = 20)
    (h2 : d.tuesday_multiplier = 2)
    (h3 : d.fabric_cost = 2)
    (h4 : d.total_earnings = 140) : 
  wednesday_fraction d = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_fraction_is_one_fourth_l941_94111


namespace NUMINAMATH_CALUDE_wire_length_proof_l941_94127

-- Define the area of the square field
def field_area : ℝ := 69696

-- Define the number of times the wire goes around the field
def rounds : ℕ := 15

-- Theorem statement
theorem wire_length_proof :
  let side_length := Real.sqrt field_area
  let perimeter := 4 * side_length
  rounds * perimeter = 15840 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_proof_l941_94127


namespace NUMINAMATH_CALUDE_plans_equal_at_325_miles_unique_intersection_at_325_miles_l941_94162

/-- Represents a car rental plan with an initial fee and a per-mile rate -/
structure RentalPlan where
  initialFee : ℝ
  perMileRate : ℝ

/-- The two rental plans available -/
def plan1 : RentalPlan := { initialFee := 65, perMileRate := 0.4 }
def plan2 : RentalPlan := { initialFee := 0, perMileRate := 0.6 }

/-- The cost of a rental plan for a given number of miles -/
def rentalCost (plan : RentalPlan) (miles : ℝ) : ℝ :=
  plan.initialFee + plan.perMileRate * miles

/-- The theorem stating that the two plans cost the same at 325 miles -/
theorem plans_equal_at_325_miles :
  rentalCost plan1 325 = rentalCost plan2 325 := by
  sorry

/-- The theorem stating that 325 is the unique point where the plans cost the same -/
theorem unique_intersection_at_325_miles :
  ∀ m : ℝ, rentalCost plan1 m = rentalCost plan2 m → m = 325 := by
  sorry

end NUMINAMATH_CALUDE_plans_equal_at_325_miles_unique_intersection_at_325_miles_l941_94162


namespace NUMINAMATH_CALUDE_simplify_expression_l941_94133

theorem simplify_expression (x y : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 9*y = 45*x + 9*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l941_94133


namespace NUMINAMATH_CALUDE_final_amount_calculation_l941_94176

/-- Calculate the final amount after two years of compound interest --/
def final_amount (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount_after_first_year := initial_amount * (1 + rate1)
  amount_after_first_year * (1 + rate2)

/-- Theorem stating the final amount after two years of compound interest --/
theorem final_amount_calculation :
  final_amount 7644 0.04 0.05 = 8347.248 := by
  sorry

#eval final_amount 7644 0.04 0.05

end NUMINAMATH_CALUDE_final_amount_calculation_l941_94176


namespace NUMINAMATH_CALUDE_cost_of_pencils_l941_94156

/-- The cost of a single pencil in cents -/
def pencil_cost : ℕ := 3

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of pencils to calculate the cost for -/
def num_pencils : ℕ := 500

/-- Theorem: The cost of 500 pencils in dollars is 15.00 -/
theorem cost_of_pencils : 
  (num_pencils * pencil_cost) / cents_per_dollar = 15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_pencils_l941_94156


namespace NUMINAMATH_CALUDE_smallest_positive_angle_same_terminal_side_l941_94179

/-- Given an angle α = 2012°, this theorem states that the smallest positive angle 
    with the same terminal side as α is 212°. -/
theorem smallest_positive_angle_same_terminal_side (α : Real) : 
  α = 2012 → ∃ (θ : Real), θ = 212 ∧ 
  θ > 0 ∧ 
  θ < 360 ∧
  ∃ (k : ℤ), α = θ + 360 * k := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_same_terminal_side_l941_94179


namespace NUMINAMATH_CALUDE_prob_greater_than_three_is_half_l941_94189

/-- The probability of rolling a number greater than 3 on a standard six-sided die is 1/2. -/
theorem prob_greater_than_three_is_half : 
  let outcomes := Finset.range 6
  let favorable := {4, 5, 6}
  Finset.card favorable / Finset.card outcomes = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_greater_than_three_is_half_l941_94189


namespace NUMINAMATH_CALUDE_second_draw_pink_probability_l941_94193

/-- Represents a bag of marbles -/
structure Bag where
  red : ℕ
  green : ℕ
  pink : ℕ
  purple : ℕ

/-- The probability of drawing a pink marble in the second draw -/
def second_draw_pink_prob (bagA bagB bagC : Bag) : ℚ :=
  let total_A := bagA.red + bagA.green
  let total_B := bagB.pink + bagB.purple
  let total_C := bagC.pink + bagC.purple
  let prob_red := bagA.red / total_A
  let prob_green := bagA.green / total_A
  let prob_pink_B := bagB.pink / total_B
  let prob_pink_C := bagC.pink / total_C
  prob_red * prob_pink_B + prob_green * prob_pink_C

theorem second_draw_pink_probability :
  let bagA : Bag := { red := 5, green := 5, pink := 0, purple := 0 }
  let bagB : Bag := { red := 0, green := 0, pink := 8, purple := 2 }
  let bagC : Bag := { red := 0, green := 0, pink := 3, purple := 7 }
  second_draw_pink_prob bagA bagB bagC = 11 / 20 := by
  sorry

#eval second_draw_pink_prob
  { red := 5, green := 5, pink := 0, purple := 0 }
  { red := 0, green := 0, pink := 8, purple := 2 }
  { red := 0, green := 0, pink := 3, purple := 7 }

end NUMINAMATH_CALUDE_second_draw_pink_probability_l941_94193


namespace NUMINAMATH_CALUDE_meeting_seating_arrangement_l941_94186

theorem meeting_seating_arrangement (n : ℕ) (h : n = 7) : 
  Nat.choose n 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_meeting_seating_arrangement_l941_94186


namespace NUMINAMATH_CALUDE_intersection_A_B_l941_94174

def A : Set ℝ := {x | 1 / x < 1}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {-1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l941_94174


namespace NUMINAMATH_CALUDE_books_from_first_shop_l941_94125

theorem books_from_first_shop 
  (total_cost_first : ℝ) 
  (books_second : ℕ) 
  (cost_second : ℝ) 
  (avg_price : ℝ) 
  (h1 : total_cost_first = 1160)
  (h2 : books_second = 50)
  (h3 : cost_second = 920)
  (h4 : avg_price = 18.08695652173913)
  : ∃ (books_first : ℕ), books_first = 65 ∧ 
    (total_cost_first + cost_second) / (books_first + books_second : ℝ) = avg_price :=
by sorry

end NUMINAMATH_CALUDE_books_from_first_shop_l941_94125


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l941_94148

theorem exponential_function_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l941_94148


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_is_two_l941_94195

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a parabola with equation y² = -8x -/
def Parabola := Unit

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given a hyperbola and a parabola satisfying certain conditions, 
    the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity_is_two 
  (h : Hyperbola) 
  (p : Parabola) 
  (A B O : Point)
  (h_asymptotes : A.x = 2 ∧ B.x = 2)  -- Asymptotes intersect directrix x = 2
  (h_origin : O.x = 0 ∧ O.y = 0)      -- O is the origin
  (h_area : abs ((A.x - O.x) * (B.y - O.y) - (B.x - O.x) * (A.y - O.y)) / 2 = 4 * Real.sqrt 3)
  : h.a / Real.sqrt (h.a^2 - h.b^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_is_two_l941_94195


namespace NUMINAMATH_CALUDE_second_largest_divisor_sum_l941_94152

theorem second_largest_divisor_sum (n : ℕ) : 
  n > 1 → 
  (∃ p : ℕ, Prime p ∧ p ∣ n ∧ n + n / p = 2013) → 
  n = 1342 := by
sorry

end NUMINAMATH_CALUDE_second_largest_divisor_sum_l941_94152


namespace NUMINAMATH_CALUDE_subset_pairs_count_for_six_elements_l941_94188

-- Define a function that counts the number of valid subset pairs
def countValidSubsetPairs (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3 * countValidSubsetPairs (n - 1) - 1

-- Theorem statement
theorem subset_pairs_count_for_six_elements :
  countValidSubsetPairs 6 = 365 := by
  sorry

end NUMINAMATH_CALUDE_subset_pairs_count_for_six_elements_l941_94188


namespace NUMINAMATH_CALUDE_national_day_2020_l941_94147

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem national_day_2020 (national_day_2019 : DayOfWeek) 
  (h1 : national_day_2019 = DayOfWeek.Tuesday) 
  (h2 : advanceDay national_day_2019 2 = DayOfWeek.Thursday) : 
  advanceDay national_day_2019 2 = DayOfWeek.Thursday := by
  sorry

#check national_day_2020

end NUMINAMATH_CALUDE_national_day_2020_l941_94147


namespace NUMINAMATH_CALUDE_figure_perimeter_l941_94170

theorem figure_perimeter (total_area : ℝ) (square_area : ℝ) (rect_width rect_length : ℝ) :
  total_area = 130 →
  3 * square_area + rect_width * rect_length = total_area →
  rect_length = 2 * rect_width →
  square_area = rect_width ^ 2 →
  (3 * square_area.sqrt + rect_width + rect_length) * 2 = 11 * Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_figure_perimeter_l941_94170


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l941_94106

theorem degree_to_radian_conversion (π : ℝ) (h : π > 0) :
  -(630 : ℝ) * (π / 180) = -(7 * π / 2) := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l941_94106


namespace NUMINAMATH_CALUDE_polynomial_equality_l941_94124

theorem polynomial_equality (a k n : ℤ) : 
  (∀ x : ℝ, (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n) → 
  a - n + k = 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l941_94124


namespace NUMINAMATH_CALUDE_symmetry_about_y_equals_x_l941_94137

/-- The set of points (x, y) satisfying the given conditions is symmetric about y = x -/
theorem symmetry_about_y_equals_x (r : ℝ) :
  ∀ (x y : ℝ), x^2 + y^2 ≤ r^2 ∧ x + y > 0 →
  ∃ (x' y' : ℝ), x'^2 + y'^2 ≤ r^2 ∧ x' + y' > 0 ∧ x' = y ∧ y' = x :=
by sorry

end NUMINAMATH_CALUDE_symmetry_about_y_equals_x_l941_94137


namespace NUMINAMATH_CALUDE_rectangle_area_l941_94113

theorem rectangle_area (diagonal : ℝ) (side_ratio : ℝ) (area : ℝ) : 
  diagonal = 15 * Real.sqrt 2 →
  side_ratio = 3 →
  area = (diagonal^2 / (1 + side_ratio^2)) * side_ratio →
  area = 135 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l941_94113


namespace NUMINAMATH_CALUDE_unique_real_number_from_complex_cube_l941_94120

theorem unique_real_number_from_complex_cube : 
  ∃! x : ℝ, ∃ a b : ℕ+, x = (a : ℝ)^3 - 3*a*b^2 ∧ 3*a^2*b - b^3 = 107 :=
sorry

end NUMINAMATH_CALUDE_unique_real_number_from_complex_cube_l941_94120


namespace NUMINAMATH_CALUDE_problem_1_l941_94144

theorem problem_1 : Real.sqrt 3 ^ 2 + |-(Real.sqrt 3 / 3)| - (π - Real.sqrt 2) ^ 0 - Real.tan (π / 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l941_94144


namespace NUMINAMATH_CALUDE_sticker_problem_l941_94187

theorem sticker_problem (x : ℝ) : 
  (x * (1 - 0.25) * (1 - 0.20) = 45) → x = 75 := by
sorry

end NUMINAMATH_CALUDE_sticker_problem_l941_94187


namespace NUMINAMATH_CALUDE_molecular_weight_3_moles_CaOH2_l941_94128

/-- Atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- Number of Calcium atoms in Ca(OH)2 -/
def num_Ca : ℕ := 1

/-- Number of Oxygen atoms in Ca(OH)2 -/
def num_O : ℕ := 2

/-- Number of Hydrogen atoms in Ca(OH)2 -/
def num_H : ℕ := 2

/-- Number of moles of Ca(OH)2 -/
def num_moles : ℝ := 3

/-- Molecular weight of Ca(OH)2 in g/mol -/
def molecular_weight_CaOH2 : ℝ :=
  num_Ca * atomic_weight_Ca + num_O * atomic_weight_O + num_H * atomic_weight_H

theorem molecular_weight_3_moles_CaOH2 :
  num_moles * molecular_weight_CaOH2 = 222.30 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_3_moles_CaOH2_l941_94128


namespace NUMINAMATH_CALUDE_evaluate_expression_l941_94167

theorem evaluate_expression (x y z : ℝ) (hx : x = 5) (hy : y = 10) (hz : z = 3) :
  z * (y - 2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l941_94167


namespace NUMINAMATH_CALUDE_karen_beats_tom_l941_94109

theorem karen_beats_tom (karen_speed : ℝ) (tom_speed : ℝ) (karen_delay : ℝ) (winning_margin : ℝ) :
  karen_speed = 60 →
  tom_speed = 45 →
  karen_delay = 4 / 60 →
  winning_margin = 4 →
  (tom_speed * (karen_delay + (winning_margin + tom_speed * karen_delay) / (karen_speed - tom_speed))) = 21 :=
by sorry

end NUMINAMATH_CALUDE_karen_beats_tom_l941_94109


namespace NUMINAMATH_CALUDE_sum_odd_positions_arithmetic_sequence_l941_94116

/-- Represents an arithmetic sequence with the given properties -/
def ArithmeticSequence (n : ℕ) (d : ℕ) (total_sum : ℕ) :=
  {seq : ℕ → ℕ | 
    (∀ i, i > 0 → i < n → seq (i + 1) = seq i + d) ∧
    (Finset.sum (Finset.range n) seq = total_sum)}

/-- Sum of terms at odd positions in the sequence -/
def SumOddPositions (seq : ℕ → ℕ) (n : ℕ) : ℕ :=
  Finset.sum (Finset.filter (λ i => i % 2 = 1) (Finset.range n)) seq

theorem sum_odd_positions_arithmetic_sequence :
  ∀ (seq : ℕ → ℕ),
    seq ∈ ArithmeticSequence 1500 2 7500 →
    SumOddPositions seq 1500 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_positions_arithmetic_sequence_l941_94116


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l941_94101

theorem geometric_sequence_sixth_term 
  (a : ℕ → ℝ)  -- The geometric sequence
  (h1 : a 1 = 4)  -- First term is 4
  (h2 : a 9 = 39304)  -- Last term is 39304
  (h3 : ∀ n : ℕ, 1 < n → n < 9 → a n = a 1 * (a 2 / a 1) ^ (n - 1))  -- Geometric sequence property
  : a 6 = 31104 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l941_94101


namespace NUMINAMATH_CALUDE_binomial_expansion_equality_l941_94118

theorem binomial_expansion_equality (a b : ℝ) (n : ℕ) :
  (∃ k : ℕ, k > 0 ∧ k < n ∧
    (Nat.choose n 0) * a^n = (Nat.choose n 2) * a^(n-2) * b^2) →
  a^2 = n * (n - 1) * b :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_equality_l941_94118


namespace NUMINAMATH_CALUDE_palace_windows_and_doors_l941_94105

structure Palace where
  rooms : ℕ
  grid_size : ℕ
  outer_walls : ℕ
  internal_partitions : ℕ

def window_count (p : Palace) : ℕ :=
  4 * p.grid_size

def door_count (p : Palace) : ℕ :=
  p.internal_partitions * p.grid_size

theorem palace_windows_and_doors (p : Palace)
  (h1 : p.rooms = 100)
  (h2 : p.grid_size = 10)
  (h3 : p.outer_walls = 4)
  (h4 : p.internal_partitions = 18) :
  window_count p = 40 ∧ door_count p = 180 := by
  sorry

end NUMINAMATH_CALUDE_palace_windows_and_doors_l941_94105


namespace NUMINAMATH_CALUDE_labeling_existence_condition_l941_94138

/-- A labeling of lattice points in Z^2 with positive integers -/
def Labeling := ℤ × ℤ → ℕ+

/-- The property that only finitely many distinct labels occur -/
def FiniteLabels (l : Labeling) : Prop :=
  ∃ (n : ℕ), ∀ (p : ℤ × ℤ), l p ≤ n

/-- The distance condition for a given c > 0 -/
def DistanceCondition (c : ℝ) (l : Labeling) : Prop :=
  ∀ (i : ℕ+) (p q : ℤ × ℤ), l p = i ∧ l q = i → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 : ℝ) ≥ c^(i : ℝ)

/-- The main theorem -/
theorem labeling_existence_condition (c : ℝ) :
  (c > 0 ∧
   ∃ (l : Labeling), FiniteLabels l ∧ DistanceCondition c l) ↔
  (c > 0 ∧ c < Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_labeling_existence_condition_l941_94138


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l941_94107

/-- A proportional function where y increases as x increases -/
structure ProportionalFunction where
  k : ℝ
  increasing : k > 0

/-- The point P with coordinates (3, k) -/
def P (f : ProportionalFunction) : ℝ × ℝ := (3, f.k)

/-- Definition of the first quadrant -/
def isInFirstQuadrant (point : ℝ × ℝ) : Prop :=
  point.1 > 0 ∧ point.2 > 0

/-- Theorem: P(3,k) is in the first quadrant for a proportional function where y increases as x increases -/
theorem point_in_first_quadrant (f : ProportionalFunction) :
  isInFirstQuadrant (P f) := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l941_94107


namespace NUMINAMATH_CALUDE_grid_product_theorem_l941_94154

theorem grid_product_theorem : ∃ (a b c d e f g h i : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
   g ≠ h ∧ g ≠ i ∧
   h ≠ i) ∧
  (a * b * c = 120 ∧
   d * e * f = 120 ∧
   g * h * i = 120 ∧
   a * d * g = 120 ∧
   b * e * h = 120 ∧
   c * f * i = 120) ∧
  (∀ (p : ℕ), (∃ (x y z u v w : ℕ),
    x ≠ y ∧ x ≠ z ∧ x ≠ u ∧ x ≠ v ∧ x ≠ w ∧
    y ≠ z ∧ y ≠ u ∧ y ≠ v ∧ y ≠ w ∧
    z ≠ u ∧ z ≠ v ∧ z ≠ w ∧
    u ≠ v ∧ u ≠ w ∧
    v ≠ w ∧
    x * y * z = p ∧ u * v * w = p) → p ≥ 120) :=
by sorry

end NUMINAMATH_CALUDE_grid_product_theorem_l941_94154


namespace NUMINAMATH_CALUDE_ascending_order_l941_94136

theorem ascending_order (a b c : ℝ) 
  (ha : a = Real.rpow 0.8 0.7)
  (hb : b = Real.rpow 0.8 0.9)
  (hc : c = Real.rpow 1.2 0.8) :
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_l941_94136


namespace NUMINAMATH_CALUDE_log_equality_implies_x_value_log_inequality_implies_x_range_l941_94141

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the conditions
variable (a : ℝ)
variable (x : ℝ)
variable (h1 : a > 0)
variable (h2 : a ≠ 1)

-- Theorem 1
theorem log_equality_implies_x_value :
  log a (3*x + 1) = log a (-3*x) → x = -1/6 :=
by sorry

-- Theorem 2
theorem log_inequality_implies_x_range :
  log a (3*x + 1) > log a (-3*x) →
  ((0 < a ∧ a < 1 → -1/3 < x ∧ x < -1/6) ∧
   (a > 1 → -1/6 < x ∧ x < 0)) :=
by sorry

end NUMINAMATH_CALUDE_log_equality_implies_x_value_log_inequality_implies_x_range_l941_94141


namespace NUMINAMATH_CALUDE_not_first_class_probability_l941_94190

theorem not_first_class_probability 
  (P_A P_B P_C : ℝ) 
  (h_A : P_A = 0.65) 
  (h_B : P_B = 0.2) 
  (h_C : P_C = 0.1) :
  1 - P_A = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_not_first_class_probability_l941_94190


namespace NUMINAMATH_CALUDE_air_conditioner_price_l941_94135

/-- The selling price per unit of the air conditioner fan before the regulation. -/
def price_before : ℝ := 880

/-- The subsidy amount per unit after the regulation. -/
def subsidy : ℝ := 80

/-- The total amount spent on purchases after the regulation. -/
def total_spent : ℝ := 60000

/-- The ratio of units purchased after the regulation to before. -/
def purchase_ratio : ℝ := 1.1

theorem air_conditioner_price :
  (total_spent / (price_before - subsidy) = (total_spent / price_before) * purchase_ratio) ∧
  (price_before > 0) ∧ 
  (price_before > subsidy) := by sorry

end NUMINAMATH_CALUDE_air_conditioner_price_l941_94135


namespace NUMINAMATH_CALUDE_polynomial_subtraction_l941_94173

theorem polynomial_subtraction (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 - x^2 + 15) - (x^6 + x^5 - 2 * x^4 + x^3 + 5) =
  x^6 + 2 * x^5 + 3 * x^4 - x^3 + x^2 + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_subtraction_l941_94173


namespace NUMINAMATH_CALUDE_club_leadership_selection_l941_94102

theorem club_leadership_selection (num_girls num_boys : ℕ) 
  (h1 : num_girls = 15) 
  (h2 : num_boys = 15) : 
  num_girls * num_boys = 225 := by
  sorry

end NUMINAMATH_CALUDE_club_leadership_selection_l941_94102


namespace NUMINAMATH_CALUDE_vector_dot_product_and_perpendicular_l941_94114

/-- Given vectors a and b, function f, and function g as defined in the problem -/
theorem vector_dot_product_and_perpendicular (x : ℝ) :
  let a : ℝ × ℝ := (-Real.sin x, 2)
  let b : ℝ × ℝ := (1, Real.cos x)
  let f : ℝ → ℝ := λ x => a.1 * b.1 + a.2 * b.2
  let g : ℝ → ℝ := λ x => (Real.sin (π + x) + 4 * Real.cos (2*π - x)) / 
                          (Real.sin (π/2 - x) - 4 * Real.sin (-x))
  -- Part 1
  f (π/6) = Real.sqrt 3 - 1/2 ∧
  -- Part 2
  (a.1 * b.1 + a.2 * b.2 = 0 → g x = 2/9) := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_and_perpendicular_l941_94114


namespace NUMINAMATH_CALUDE_jimin_weight_l941_94149

theorem jimin_weight (T J : ℝ) (h1 : T - J = 4) (h2 : T + J = 88) : J = 42 := by
  sorry

end NUMINAMATH_CALUDE_jimin_weight_l941_94149


namespace NUMINAMATH_CALUDE_ball_probabilities_l941_94115

/-- Represents the contents of a box with black and white balls -/
structure Box where
  total : ℕ
  black : ℕ
  white : ℕ
  black_ratio : ℚ
  white_ratio : ℚ
  ratio_sum_one : black_ratio + white_ratio = 1
  contents_match : black + white = total

/-- The setup of the three boxes as per the problem -/
def box_setup : (Box × Box × Box) := sorry

/-- The probability of selecting all black balls when choosing one from each box -/
def prob_all_black (boxes : Box × Box × Box) : ℚ := sorry

/-- The probability of selecting a white ball from all boxes combined -/
def prob_white_combined (boxes : Box × Box × Box) : ℚ := sorry

/-- Main theorem stating the probabilities as per the problem -/
theorem ball_probabilities :
  let boxes := box_setup
  prob_all_black boxes = 1/20 ∧ prob_white_combined boxes = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l941_94115


namespace NUMINAMATH_CALUDE_sum_condition_iff_divisible_l941_94191

/-- An arithmetic progression with first term a and common difference d. -/
structure ArithmeticProgression (α : Type*) [Ring α] where
  a : α
  d : α

/-- The nth term of an arithmetic progression. -/
def ArithmeticProgression.nthTerm {α : Type*} [Ring α] (ap : ArithmeticProgression α) (n : ℕ) : α :=
  ap.a + n • ap.d

/-- Condition for the sum of two terms to be another term in the progression. -/
def SumCondition {α : Type*} [Ring α] (ap : ArithmeticProgression α) : Prop :=
  ∀ n k : ℕ, ∃ p : ℕ, ap.nthTerm n + ap.nthTerm k = ap.nthTerm p

/-- Theorem: The sum condition holds if and only if the first term is divisible by the common difference. -/
theorem sum_condition_iff_divisible {α : Type*} [CommRing α] (ap : ArithmeticProgression α) :
    SumCondition ap ↔ ∃ m : α, ap.a = m * ap.d :=
  sorry

end NUMINAMATH_CALUDE_sum_condition_iff_divisible_l941_94191


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l941_94103

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x - a = 0 ∧ x = 2) → 
  (∃ y : ℝ, y^2 + 2*y - a = 0 ∧ y = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l941_94103


namespace NUMINAMATH_CALUDE_arithmetic_sum_10_l941_94140

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) : ℕ := n * (2 * n + 1)

/-- Theorem: The sum of the first 10 terms of the arithmetic sequence with general term a_n = 2n + 1 is 120 -/
theorem arithmetic_sum_10 : arithmetic_sum 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_10_l941_94140


namespace NUMINAMATH_CALUDE_cindy_calculation_l941_94166

theorem cindy_calculation (x : ℝ) : 
  (x - 12) / 4 = 28 → (x - 5) / 8 = 14.875 := by sorry

end NUMINAMATH_CALUDE_cindy_calculation_l941_94166


namespace NUMINAMATH_CALUDE_sum_expression_equals_1215_l941_94159

theorem sum_expression_equals_1215 
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 2018) 
  (h2 : 3*a + 8*b + 24*c + 37*d = 2018) : 
  3*b + 8*c + 24*d + 37*a = 1215 := by
  sorry

end NUMINAMATH_CALUDE_sum_expression_equals_1215_l941_94159


namespace NUMINAMATH_CALUDE_spend_fifty_is_negative_fifty_l941_94199

-- Define a type for monetary transactions
inductive MonetaryTransaction
| Receive (amount : ℤ)
| Spend (amount : ℤ)

-- Define a function to represent the sign of a transaction
def transactionSign (t : MonetaryTransaction) : ℤ :=
  match t with
  | MonetaryTransaction.Receive _ => 1
  | MonetaryTransaction.Spend _ => -1

-- State the theorem
theorem spend_fifty_is_negative_fifty 
  (h1 : transactionSign (MonetaryTransaction.Receive 80) = 1)
  (h2 : transactionSign (MonetaryTransaction.Spend 50) = -transactionSign (MonetaryTransaction.Receive 50)) :
  transactionSign (MonetaryTransaction.Spend 50) * 50 = -50 := by
  sorry

end NUMINAMATH_CALUDE_spend_fifty_is_negative_fifty_l941_94199


namespace NUMINAMATH_CALUDE_candy_bar_cost_candy_bar_cost_is_7_l941_94164

def chocolate_cost : ℕ := 3
def extra_cost : ℕ := 4

theorem candy_bar_cost : ℕ :=
  chocolate_cost + extra_cost

#check candy_bar_cost

theorem candy_bar_cost_is_7 : candy_bar_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_candy_bar_cost_is_7_l941_94164


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l941_94184

theorem sum_of_fractions_equals_seven : 
  let S := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
           1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 
           1 / (Real.sqrt 12 - 3)
  S = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l941_94184


namespace NUMINAMATH_CALUDE_solve_equation_l941_94197

theorem solve_equation (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l941_94197


namespace NUMINAMATH_CALUDE_james_chocolate_sales_l941_94172

/-- Calculates the number of chocolate bars James sold this week -/
def chocolate_bars_sold_this_week (total : ℕ) (sold_last_week : ℕ) (to_sell : ℕ) : ℕ :=
  total - (sold_last_week + to_sell)

/-- Proves that James sold 2 chocolate bars this week -/
theorem james_chocolate_sales : chocolate_bars_sold_this_week 18 5 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_chocolate_sales_l941_94172


namespace NUMINAMATH_CALUDE_right_triangle_30_60_90_properties_l941_94185

/-- A right triangle with one leg of 15 inches and the opposite angle of 30 degrees -/
structure RightTriangle30_60_90 where
  /-- The length of one leg of the triangle -/
  short_leg : ℝ
  /-- The angle opposite the short leg -/
  opposite_angle : ℝ
  /-- The triangle is a right triangle -/
  is_right_triangle : True
  /-- The short leg is 15 inches -/
  short_leg_length : short_leg = 15
  /-- The opposite angle is 30 degrees -/
  opposite_angle_measure : opposite_angle = 30

/-- The hypotenuse of the triangle -/
def hypotenuse (t : RightTriangle30_60_90) : ℝ := 2 * t.short_leg

/-- The altitude from the hypotenuse to the right angle -/
def altitude (t : RightTriangle30_60_90) : ℝ := 1.5 * t.short_leg

theorem right_triangle_30_60_90_properties (t : RightTriangle30_60_90) :
  hypotenuse t = 30 ∧ altitude t = 22.5 := by
  sorry

#check right_triangle_30_60_90_properties

end NUMINAMATH_CALUDE_right_triangle_30_60_90_properties_l941_94185


namespace NUMINAMATH_CALUDE_angle4_value_l941_94157

-- Define the angles as real numbers
variable (angle1 angle2 angle3 angle4 : ℝ)

-- State the theorem
theorem angle4_value
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4) :
  angle4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_angle4_value_l941_94157


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_same_foci_l941_94196

/-- Given a hyperbola and an ellipse with the same foci, prove that m = 1/11 -/
theorem hyperbola_ellipse_same_foci (m : ℝ) : 
  (∃ (c : ℝ), c^2 = 2*m ∧ c^2 = (m+1)/6) → m = 1/11 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_same_foci_l941_94196


namespace NUMINAMATH_CALUDE_complement_of_A_l941_94198

def U : Set Nat := {1,2,3,4,5,6,7}
def A : Set Nat := {1,2,4,5}

theorem complement_of_A :
  (U \ A) = {3,6,7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l941_94198


namespace NUMINAMATH_CALUDE_gcd_1337_382_l941_94158

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1337_382_l941_94158


namespace NUMINAMATH_CALUDE_smallest_distance_complex_circles_l941_94171

theorem smallest_distance_complex_circles (z w : ℂ) 
  (hz : Complex.abs (z + 1 + 3*I) = 1)
  (hw : Complex.abs (w - 7 - 8*I) = 3) :
  ∃ (min_dist : ℝ), 
    (∀ (z' w' : ℂ), 
      Complex.abs (z' + 1 + 3*I) = 1 → 
      Complex.abs (w' - 7 - 8*I) = 3 → 
      Complex.abs (z' - w') ≥ min_dist) ∧
    (∃ (z₀ w₀ : ℂ), 
      Complex.abs (z₀ + 1 + 3*I) = 1 ∧ 
      Complex.abs (w₀ - 7 - 8*I) = 3 ∧ 
      Complex.abs (z₀ - w₀) = min_dist) ∧
    min_dist = Real.sqrt 185 - 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_complex_circles_l941_94171


namespace NUMINAMATH_CALUDE_farm_feet_count_l941_94123

/-- Given a farm with hens and cows, prove the total number of feet -/
theorem farm_feet_count (total_heads : ℕ) (hen_count : ℕ) : 
  total_heads = 50 → hen_count = 28 → 
  (hen_count * 2 + (total_heads - hen_count) * 4 = 144) :=
by
  sorry

#check farm_feet_count

end NUMINAMATH_CALUDE_farm_feet_count_l941_94123


namespace NUMINAMATH_CALUDE_equation_solutions_l941_94161

-- Define the equation
def equation (x : ℝ) : Prop :=
  (59 - 3*x)^(1/4) + (17 + 3*x)^(1/4) = 4

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 20 ∨ x = -10) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l941_94161


namespace NUMINAMATH_CALUDE_marble_distribution_correct_l941_94132

/-- Represents the distribution of marbles among four boys -/
structure MarbleDistribution where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The rule for distributing marbles based on a parameter x -/
def distributionRule (x : ℕ) : MarbleDistribution :=
  { first := 3 * x + 2
  , second := x + 1
  , third := 2 * x - 1
  , fourth := x }

/-- Theorem stating that the given distribution satisfies the problem conditions -/
theorem marble_distribution_correct : ∃ x : ℕ, 
  let d := distributionRule x
  d.first = 22 ∧
  d.second = 8 ∧
  d.third = 12 ∧
  d.fourth = 7 ∧
  d.first + d.second + d.third + d.fourth = 49 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_correct_l941_94132


namespace NUMINAMATH_CALUDE_quadratic_solution_l941_94160

theorem quadratic_solution (a : ℝ) : (1 : ℝ)^2 + 1 + 2*a = 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l941_94160


namespace NUMINAMATH_CALUDE_fish_sold_correct_l941_94165

/-- The number of fish initially in stock -/
def initial_stock : ℕ := 200

/-- The number of fish in the new stock -/
def new_stock : ℕ := 200

/-- The final number of fish in stock -/
def final_stock : ℕ := 300

/-- The fraction of remaining fish that become spoiled -/
def spoilage_rate : ℚ := 1/3

/-- The number of fish sold -/
def fish_sold : ℕ := 50

theorem fish_sold_correct :
  (initial_stock - fish_sold - (initial_stock - fish_sold) * spoilage_rate + new_stock : ℚ) = final_stock :=
sorry

end NUMINAMATH_CALUDE_fish_sold_correct_l941_94165


namespace NUMINAMATH_CALUDE_evaluate_expression_l941_94146

theorem evaluate_expression : (20 ^ 40) / (80 ^ 10) = 5 ^ 10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l941_94146


namespace NUMINAMATH_CALUDE_min_sum_of_product_l941_94139

theorem min_sum_of_product (a b c : ℕ+) (h : a * b * c = 2450) :
  ∃ (x y z : ℕ+), x * y * z = 2450 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 76 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l941_94139


namespace NUMINAMATH_CALUDE_perpendicular_sum_equals_perimeter_l941_94108

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define perpendicular points on angle bisectors
structure PerpendicularPoints (T : Triangle) :=
  (A1 A2 B1 B2 C1 C2 : ℝ × ℝ)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem perpendicular_sum_equals_perimeter (T : Triangle) (P : PerpendicularPoints T) :
  2 * (distance P.A1 P.A2 + distance P.B1 P.B2 + distance P.C1 P.C2) =
  distance T.A T.B + distance T.B T.C + distance T.C T.A := by sorry

end NUMINAMATH_CALUDE_perpendicular_sum_equals_perimeter_l941_94108


namespace NUMINAMATH_CALUDE_laptops_in_shop_l941_94168

theorem laptops_in_shop (rows : ℕ) (laptops_per_row : ℕ) 
  (h1 : rows = 5) (h2 : laptops_per_row = 8) : 
  rows * laptops_per_row = 40 := by
  sorry

end NUMINAMATH_CALUDE_laptops_in_shop_l941_94168


namespace NUMINAMATH_CALUDE_max_matches_C_proof_l941_94183

/-- Represents a player in the tournament -/
inductive Player : Type
| A : Player
| B : Player
| C : Player
| D : Player

/-- The number of matches won by a player -/
def matches_won : Player → Nat
| Player.A => 2
| Player.B => 1
| _ => 0  -- We don't know for C and D, so we set it to 0

/-- The total number of matches in a round-robin tournament with 4 players -/
def total_matches : Nat := 6

/-- The maximum number of matches C can win -/
def max_matches_C : Nat := 3

/-- Theorem stating the maximum number of matches C can win -/
theorem max_matches_C_proof :
  ∀ (c_wins : Nat),
  c_wins ≤ max_matches_C ∧
  c_wins + matches_won Player.A + matches_won Player.B ≤ total_matches :=
sorry

end NUMINAMATH_CALUDE_max_matches_C_proof_l941_94183


namespace NUMINAMATH_CALUDE_remainder_equality_l941_94145

theorem remainder_equality (Q Q' E S S' s s' : ℕ) 
  (hQ : Q > Q') 
  (hS : S = Q % E) 
  (hS' : S' = Q' % E) 
  (hs : s = (Q^2 * Q') % E) 
  (hs' : s' = (S^2 * S') % E) : 
  s = s' := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l941_94145


namespace NUMINAMATH_CALUDE_area_regular_octagon_in_circle_l941_94117

/-- The area of a regular octagon inscribed in a circle -/
theorem area_regular_octagon_in_circle (r : ℝ) (h : r^2 * Real.pi = 256 * Real.pi) :
  8 * ((2 * r * Real.sin (Real.pi / 8))^2 * Real.sqrt 2 / 4) = 
    8 * (2 * 16 * Real.sin (Real.pi / 8))^2 * Real.sqrt 2 / 4 := by
  sorry

#check area_regular_octagon_in_circle

end NUMINAMATH_CALUDE_area_regular_octagon_in_circle_l941_94117


namespace NUMINAMATH_CALUDE_arithmetic_proof_l941_94110

theorem arithmetic_proof : (1) - 2^3 / (-1/5) - 1/2 * (-4)^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l941_94110


namespace NUMINAMATH_CALUDE_final_ball_is_green_l941_94192

/-- Represents the colors of balls in the bag -/
inductive Color
| Red
| Green

/-- Represents the state of the bag -/
structure BagState where
  red : Nat
  green : Nat

/-- The process of drawing and modifying balls -/
def drawProcess (state : BagState) : BagState :=
  sorry

/-- The theorem to be proved -/
theorem final_ball_is_green (initial : BagState) 
  (h1 : initial.red = 2020) 
  (h2 : initial.green = 2021) :
  ∃ (final : BagState), 
    (final.red + final.green = 1) ∧ 
    (final.green = 1) ∧
    (∃ (n : Nat), (drawProcess^[n] initial) = final) :=
  sorry

end NUMINAMATH_CALUDE_final_ball_is_green_l941_94192


namespace NUMINAMATH_CALUDE_total_pay_is_1980_l941_94126

/-- Calculates the total monthly pay for Josh and Carl given their work hours and rates -/
def total_monthly_pay (josh_hours_per_day : ℕ) (work_days_per_week : ℕ) (weeks_per_month : ℕ)
  (carl_hours_less : ℕ) (josh_hourly_rate : ℚ) : ℚ :=
  let josh_monthly_hours := josh_hours_per_day * work_days_per_week * weeks_per_month
  let carl_monthly_hours := (josh_hours_per_day - carl_hours_less) * work_days_per_week * weeks_per_month
  let carl_hourly_rate := josh_hourly_rate / 2
  josh_monthly_hours * josh_hourly_rate + carl_monthly_hours * carl_hourly_rate

theorem total_pay_is_1980 :
  total_monthly_pay 8 5 4 2 9 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_is_1980_l941_94126


namespace NUMINAMATH_CALUDE_square_minus_product_l941_94155

theorem square_minus_product : (422 + 404)^2 - (4 * 422 * 404) = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_l941_94155


namespace NUMINAMATH_CALUDE_reeya_average_score_l941_94153

theorem reeya_average_score : 
  let scores : List ℝ := [50, 60, 70, 80, 80]
  (scores.sum / scores.length : ℝ) = 68 := by
sorry

end NUMINAMATH_CALUDE_reeya_average_score_l941_94153


namespace NUMINAMATH_CALUDE_solution_triples_l941_94178

theorem solution_triples (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a + b + c = 1/a + 1/b + 1/c) ∧ (a^2 + b^2 + c^2 = 1/a^2 + 1/b^2 + 1/c^2) →
  ((a = 1 ∧ c = 1/b) ∨ (b = 1/a ∧ c = 1) ∨ (b = 1 ∧ c = 1/a) ∨
   (a = -1 ∧ c = 1/b) ∨ (b = -1 ∧ c = 1/a) ∨ (b = 1/a ∧ c = -1)) :=
by sorry

end NUMINAMATH_CALUDE_solution_triples_l941_94178


namespace NUMINAMATH_CALUDE_calculation_result_l941_94181

/-- The smallest two-digit prime number -/
def smallest_two_digit_prime : ℕ := 11

/-- The largest one-digit prime number -/
def largest_one_digit_prime : ℕ := 7

/-- The smallest one-digit prime number -/
def smallest_one_digit_prime : ℕ := 2

/-- Theorem stating the result of the calculation -/
theorem calculation_result :
  smallest_two_digit_prime * (largest_one_digit_prime ^ 2) - smallest_one_digit_prime = 537 := by
  sorry


end NUMINAMATH_CALUDE_calculation_result_l941_94181


namespace NUMINAMATH_CALUDE_quadratic_roots_for_negative_k_l941_94182

theorem quadratic_roots_for_negative_k (k : ℝ) (h : k < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + x₁ + k - 1 = 0 ∧ x₂^2 + x₂ + k - 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_for_negative_k_l941_94182


namespace NUMINAMATH_CALUDE_average_difference_l941_94163

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 170) : 
  a - c = -120 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l941_94163


namespace NUMINAMATH_CALUDE_audience_with_envelopes_l941_94169

theorem audience_with_envelopes (total_audience : ℕ) (winners : ℕ) (winning_percentage : ℚ) :
  total_audience = 100 →
  winners = 8 →
  winning_percentage = 1/5 →
  (winners : ℚ) / (winning_percentage * total_audience) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_audience_with_envelopes_l941_94169


namespace NUMINAMATH_CALUDE_minimum_score_for_average_increase_l941_94122

def larry_scores : List ℕ := [75, 65, 85, 95, 60]
def target_increase : ℕ := 5

theorem minimum_score_for_average_increase 
  (scores : List ℕ) 
  (target_increase : ℕ) 
  (h1 : scores = larry_scores) 
  (h2 : target_increase = 5) : 
  ∃ (next_score : ℕ),
    (next_score = 106) ∧ 
    ((scores.sum + next_score) / (scores.length + 1) : ℚ) = 
    (scores.sum / scores.length : ℚ) + target_increase ∧
    ∀ (x : ℕ), x < next_score → 
      ((scores.sum + x) / (scores.length + 1) : ℚ) < 
      (scores.sum / scores.length : ℚ) + target_increase := by
  sorry

end NUMINAMATH_CALUDE_minimum_score_for_average_increase_l941_94122


namespace NUMINAMATH_CALUDE_sum_altitudes_less_perimeter_l941_94194

/-- A triangle with sides a, b, c and corresponding altitudes h₁, h₂, h₃ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_h₁ : 0 < h₁
  pos_h₂ : 0 < h₂
  pos_h₃ : 0 < h₃
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b
  altitude_relation : h₁ * a = 2 * area ∧ h₂ * b = 2 * area ∧ h₃ * c = 2 * area
  area_pos : 0 < area

/-- The sum of the altitudes of a triangle is less than its perimeter -/
theorem sum_altitudes_less_perimeter (t : Triangle) : t.h₁ + t.h₂ + t.h₃ < t.a + t.b + t.c := by
  sorry

end NUMINAMATH_CALUDE_sum_altitudes_less_perimeter_l941_94194


namespace NUMINAMATH_CALUDE_circle_M_properties_l941_94119

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 3 = 0

-- Define the center of a circle
def is_center (cx cy : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y ↔ (x - cx)^2 + (y - cy)^2 = (x - cx)^2 + (y - cy)^2

-- Define a tangent line to a circle
def is_tangent_line (m b : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃! x y, circle x y ∧ y = m*x + b

-- Main theorem
theorem circle_M_properties :
  (is_center (-2) 1 circle_M) ∧
  (∀ m b : ℝ, is_tangent_line m b circle_M ∧ 0 = m*(-3) + b → b = -3) :=
sorry

end NUMINAMATH_CALUDE_circle_M_properties_l941_94119


namespace NUMINAMATH_CALUDE_largest_x_absolute_value_inequality_l941_94100

theorem largest_x_absolute_value_inequality : 
  ∃ (x_max : ℝ), x_max = 199 ∧ 
  (∀ (x : ℝ), abs (x^2 - 4*x - 39601) ≥ abs (x^2 + 4*x - 39601) → x ≤ x_max) ∧
  abs (x_max^2 - 4*x_max - 39601) ≥ abs (x_max^2 + 4*x_max - 39601) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_absolute_value_inequality_l941_94100


namespace NUMINAMATH_CALUDE_fermat_quotient_perfect_square_no_fermat_quotient_perfect_square_l941_94142

theorem fermat_quotient_perfect_square (p : ℕ) (h : Prime p) :
  (∃ (x : ℕ), (7^(p-1) - 1) / p = x^2) ↔ p = 3 :=
sorry

theorem no_fermat_quotient_perfect_square (p : ℕ) (h : Prime p) :
  ¬∃ (x : ℕ), (11^(p-1) - 1) / p = x^2 :=
sorry

end NUMINAMATH_CALUDE_fermat_quotient_perfect_square_no_fermat_quotient_perfect_square_l941_94142


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_three_l941_94121

theorem sqrt_expression_equals_three :
  (Real.sqrt 48 - 3 * Real.sqrt (1/3)) / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_three_l941_94121


namespace NUMINAMATH_CALUDE_max_value_on_circle_l941_94134

theorem max_value_on_circle (x y z : ℝ) : 
  x^2 + y^2 = 4 → z = 2*x + y → z ≤ 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l941_94134


namespace NUMINAMATH_CALUDE_prime_square_problem_l941_94131

theorem prime_square_problem (c : ℕ) (h1 : Nat.Prime c) 
  (h2 : ∃ m : ℕ, m > 0 ∧ 11 * c + 1 = m ^ 2) : c = 13 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_problem_l941_94131


namespace NUMINAMATH_CALUDE_lego_pieces_sold_l941_94129

/-- The number of single Lego pieces sold -/
def single_pieces : ℕ := sorry

/-- The total earnings in cents -/
def total_earnings : ℕ := 1000

/-- The number of double pieces sold -/
def double_pieces : ℕ := 45

/-- The number of triple pieces sold -/
def triple_pieces : ℕ := 50

/-- The number of quadruple pieces sold -/
def quadruple_pieces : ℕ := 165

/-- The cost of each circle in cents -/
def circle_cost : ℕ := 1

theorem lego_pieces_sold :
  single_pieces = 100 :=
by sorry

end NUMINAMATH_CALUDE_lego_pieces_sold_l941_94129


namespace NUMINAMATH_CALUDE_at_least_one_triangle_inside_l941_94104

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a pentagon -/
structure Pentagon :=
  (vertices : Fin 5 → Point)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (vertices : Fin 3 → Point)

/-- Checks if a pentagon is convex and equilateral -/
def isConvexEquilateralPentagon (p : Pentagon) : Prop :=
  sorry

/-- Constructs equilateral triangles on the sides of a pentagon -/
def constructTriangles (p : Pentagon) : Fin 5 → EquilateralTriangle :=
  sorry

/-- Checks if a triangle is entirely contained within a pentagon -/
def isTriangleContained (t : EquilateralTriangle) (p : Pentagon) : Prop :=
  sorry

/-- The main theorem -/
theorem at_least_one_triangle_inside (p : Pentagon) 
  (h : isConvexEquilateralPentagon p) :
  ∃ (i : Fin 5), isTriangleContained (constructTriangles p i) p :=
sorry

end NUMINAMATH_CALUDE_at_least_one_triangle_inside_l941_94104


namespace NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l941_94177

theorem polygon_sides_and_diagonals :
  ∀ n : ℕ,
  (n > 2) →
  (180 * (n - 2) = 3 * 360 - 180) →
  (n = 7 ∧ (n * (n - 3)) / 2 = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_and_diagonals_l941_94177


namespace NUMINAMATH_CALUDE_team_formation_ways_l941_94180

/-- The number of ways to choose 2 players from a group of 5 players -/
def choose_teams (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- There are 5 friends in total -/
def total_players : ℕ := 5

/-- The size of the smaller team -/
def team_size : ℕ := 2

theorem team_formation_ways :
  choose_teams total_players team_size = 10 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_ways_l941_94180


namespace NUMINAMATH_CALUDE_two_digit_number_property_l941_94151

theorem two_digit_number_property (N : ℕ) : 
  (10 ≤ N) ∧ (N < 100) →
  (4 * (N / 10) + 2 * (N % 10) = N / 2) →
  (N = 32 ∨ N = 64 ∨ N = 96) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l941_94151


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l941_94143

theorem smallest_solution_quartic_equation :
  ∃ x : ℝ, x^4 - 14*x^2 + 49 = 0 ∧ 
  (∀ y : ℝ, y^4 - 14*y^2 + 49 = 0 → x ≤ y) ∧
  x = -Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l941_94143


namespace NUMINAMATH_CALUDE_exam_attendance_l941_94175

theorem exam_attendance (passed_percentage : ℝ) (failed_count : ℕ) : 
  passed_percentage = 35 → 
  failed_count = 351 → 
  (failed_count : ℝ) / (100 - passed_percentage) * 100 = 540 := by
sorry

end NUMINAMATH_CALUDE_exam_attendance_l941_94175


namespace NUMINAMATH_CALUDE_race_orders_theorem_l941_94150

-- Define the number of racers
def num_racers : ℕ := 6

-- Define the function to calculate the number of possible orders
def possible_orders (n : ℕ) : ℕ := Nat.factorial n

-- Theorem statement
theorem race_orders_theorem : possible_orders num_racers = 720 := by
  sorry

end NUMINAMATH_CALUDE_race_orders_theorem_l941_94150
