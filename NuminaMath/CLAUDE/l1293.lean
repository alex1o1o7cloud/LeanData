import Mathlib

namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_two_l1293_129376

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ b₁ m₂ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x + b₁ = y ↔ m₂ * x + b₂ = y) ↔ m₁ = m₂

/-- Definition of line l₁ -/
def l₁ (x y : ℝ) : Prop := 2 * x - y - 1 = 0

/-- Definition of line l₂ -/
def l₂ (a x y : ℝ) : Prop := 2 * x + (a + 1) * y + 2 = 0

/-- Theorem: If l₁ is parallel to l₂, then a = -2 -/
theorem parallel_lines_imply_a_eq_neg_two :
  (∀ x y : ℝ, l₁ x y ↔ l₂ a x y) → a = -2 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_two_l1293_129376


namespace NUMINAMATH_CALUDE_ribbon_cutting_theorem_l1293_129371

/-- Represents the cutting time for a pair of centimeters -/
structure CutTimePair :=
  (first : Nat)
  (second : Nat)

/-- Calculates the total cutting time for the ribbon -/
def totalCutTime (ribbonLength : Nat) (cutTimePair : CutTimePair) : Nat :=
  (ribbonLength / 2) * (cutTimePair.first + cutTimePair.second)

/-- Calculates the length of ribbon cut in half the total time -/
def ribbonCutInHalfTime (ribbonLength : Nat) (cutTimePair : CutTimePair) : Nat :=
  ((totalCutTime ribbonLength cutTimePair) / 2) / (cutTimePair.first + cutTimePair.second) * 2

theorem ribbon_cutting_theorem (ribbonLength : Nat) (cutTimePair : CutTimePair) :
  ribbonLength = 200 →
  cutTimePair = { first := 35, second := 40 } →
  totalCutTime ribbonLength cutTimePair = 3750 ∧
  ribbonLength - ribbonCutInHalfTime ribbonLength cutTimePair = 150 :=
by sorry

#eval totalCutTime 200 { first := 35, second := 40 }
#eval 200 - ribbonCutInHalfTime 200 { first := 35, second := 40 }

end NUMINAMATH_CALUDE_ribbon_cutting_theorem_l1293_129371


namespace NUMINAMATH_CALUDE_vector_b_magnitude_l1293_129367

def a : ℝ × ℝ := (2, 1)

theorem vector_b_magnitude (b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = 10) 
  (h2 : Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5 * Real.sqrt 2) : 
  Real.sqrt (b.1^2 + b.2^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_b_magnitude_l1293_129367


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1293_129354

theorem complex_fraction_simplification : (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1293_129354


namespace NUMINAMATH_CALUDE_maria_trip_distance_l1293_129302

/-- Given a total trip distance of 400 miles, with stops at 1/2 of the total distance
    and 1/4 of the remaining distance after the first stop, the distance traveled
    after the second stop is 150 miles. -/
theorem maria_trip_distance : 
  let total_distance : ℝ := 400
  let first_stop_fraction : ℝ := 1/2
  let second_stop_fraction : ℝ := 1/4
  let distance_to_first_stop := total_distance * first_stop_fraction
  let remaining_after_first_stop := total_distance - distance_to_first_stop
  let distance_to_second_stop := remaining_after_first_stop * second_stop_fraction
  let distance_after_second_stop := remaining_after_first_stop - distance_to_second_stop
  distance_after_second_stop = 150 := by
sorry

end NUMINAMATH_CALUDE_maria_trip_distance_l1293_129302


namespace NUMINAMATH_CALUDE_symmetric_complex_division_l1293_129300

/-- Two complex numbers are symmetric about y = x if their real and imaginary parts are swapped -/
def symmetric_about_y_eq_x (z₁ z₂ : ℂ) : Prop :=
  z₁.re = z₂.im ∧ z₁.im = z₂.re

theorem symmetric_complex_division (z₁ z₂ : ℂ) : 
  symmetric_about_y_eq_x z₁ z₂ → z₁ = 1 + 2*I → z₁ / z₂ = 4/5 + 3/5*I := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_division_l1293_129300


namespace NUMINAMATH_CALUDE_offspring_trisomy_is_heritable_variation_l1293_129379

-- Define the genotype structure
structure Genotype where
  allele1 : Char
  allele2 : Char
  allele3 : Char
  allele4 : Char

-- Define the chromosome structure
structure Chromosome where
  gene1 : Char
  gene2 : Char

-- Define the diploid tomato
def diploidTomato : Genotype := { allele1 := 'A', allele2 := 'a', allele3 := 'B', allele4 := 'b' }

-- Define the offspring with trisomy
def offspringTrisomy : Genotype := { allele1 := 'A', allele2 := 'a', allele3 := 'B', allele4 := 'b' }

-- Define the property of genes being on different homologous chromosomes
def genesOnDifferentChromosomes (g : Genotype) : Prop :=
  ∃ (c1 c2 : Chromosome), (c1.gene1 = g.allele1 ∧ c1.gene2 = g.allele3) ∧
                          (c2.gene1 = g.allele2 ∧ c2.gene2 = g.allele4)

-- Define heritable variation
def heritableVariation (parent offspring : Genotype) : Prop :=
  parent ≠ offspring ∧ ∃ (gene : Char), (gene ∈ [parent.allele1, parent.allele2, parent.allele3, parent.allele4]) ∧
                                        (gene ∈ [offspring.allele1, offspring.allele2, offspring.allele3, offspring.allele4])

-- Theorem statement
theorem offspring_trisomy_is_heritable_variation :
  genesOnDifferentChromosomes diploidTomato →
  heritableVariation diploidTomato offspringTrisomy :=
by sorry

end NUMINAMATH_CALUDE_offspring_trisomy_is_heritable_variation_l1293_129379


namespace NUMINAMATH_CALUDE_correct_factorization_l1293_129315

theorem correct_factorization (x : ℝ) : 4 * x^2 - 4 * x + 1 = (2 * x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1293_129315


namespace NUMINAMATH_CALUDE_expand_product_l1293_129372

theorem expand_product (x : ℝ) : (x + 5) * (x - 4^2) = x^2 - 11*x - 80 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1293_129372


namespace NUMINAMATH_CALUDE_height_on_side_BC_l1293_129369

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where a = √3, b = √2, and 1 + 2cos(B+C) = 0, 
    prove that the height h on side BC is equal to (√3 + 1) / 2. -/
theorem height_on_side_BC (A B C : ℝ) (a b c : ℝ) (h : ℝ) : 
  a = Real.sqrt 3 → 
  b = Real.sqrt 2 → 
  1 + 2 * Real.cos (B + C) = 0 → 
  h = (Real.sqrt 3 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_height_on_side_BC_l1293_129369


namespace NUMINAMATH_CALUDE_balloon_final_height_l1293_129370

/-- Represents the sequence of balloon movements -/
def BalloonMovements : List Int := [6, -2, 3, -2]

/-- Calculates the final height of the balloon after a sequence of movements -/
def finalHeight (movements : List Int) : Int :=
  movements.foldl (· + ·) 0

/-- Theorem stating that the final height of the balloon is 5 meters -/
theorem balloon_final_height :
  finalHeight BalloonMovements = 5 := by
  sorry

end NUMINAMATH_CALUDE_balloon_final_height_l1293_129370


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l1293_129346

theorem cubic_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) : 
  a^3 + b^3 + c^3 = -27 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l1293_129346


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1293_129303

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- Theorem statement
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (c : Line) (α β : Plane) :
  parallel c α → perpendicular c β → perpendicularPlanes α β := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1293_129303


namespace NUMINAMATH_CALUDE_smallest_sum_consecutive_integers_l1293_129345

theorem smallest_sum_consecutive_integers : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (l : ℤ), n = (9 : ℤ) * (l + 4)) ∧ 
  (∃ (m : ℤ), n = (5 : ℤ) * (2 * m + 9)) ∧ 
  (∃ (k : ℤ), n = (11 : ℤ) * (k + 5)) ∧ 
  (∀ (n' : ℕ), n' > 0 → 
    (∃ (l : ℤ), n' = (9 : ℤ) * (l + 4)) → 
    (∃ (m : ℤ), n' = (5 : ℤ) * (2 * m + 9)) → 
    (∃ (k : ℤ), n' = (11 : ℤ) * (k + 5)) → 
    n ≤ n') ∧ 
  n = 495 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_consecutive_integers_l1293_129345


namespace NUMINAMATH_CALUDE_total_spent_is_40_l1293_129325

def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def num_ingredients : ℕ := 5
def apron_cost : ℕ := recipe_book_cost + 1

def total_cost : ℕ := recipe_book_cost + baking_dish_cost + (ingredient_cost * num_ingredients) + apron_cost

theorem total_spent_is_40 : total_cost = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_40_l1293_129325


namespace NUMINAMATH_CALUDE_min_value_of_power_difference_l1293_129384

theorem min_value_of_power_difference (m n : ℕ) : 12^m - 5^n ≥ 7 ∧ ∃ m n : ℕ, 12^m - 5^n = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_power_difference_l1293_129384


namespace NUMINAMATH_CALUDE_multiply_72516_by_9999_l1293_129334

theorem multiply_72516_by_9999 : 72516 * 9999 = 724787484 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72516_by_9999_l1293_129334


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l1293_129368

/-- 
Given three lines that intersect at the same point:
1. y = 2x + 7
2. y = -3x - 6
3. y = 4x + m
Prove that m = 61/5
-/
theorem intersection_of_three_lines (x y m : ℝ) : 
  (y = 2*x + 7) ∧ 
  (y = -3*x - 6) ∧ 
  (y = 4*x + m) → 
  m = 61/5 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l1293_129368


namespace NUMINAMATH_CALUDE_journey_satisfies_equations_l1293_129339

/-- Represents Li Hai's journey from point A to point B -/
structure Journey where
  totalDistance : ℝ
  uphillSpeed : ℝ
  downhillSpeed : ℝ
  totalTime : ℝ
  uphillTime : ℝ
  downhillTime : ℝ

/-- Checks if the given journey satisfies the system of equations -/
def satisfiesEquations (j : Journey) : Prop :=
  j.uphillTime + j.downhillTime = j.totalTime ∧
  (j.uphillSpeed * j.uphillTime / 60 + j.downhillSpeed * j.downhillTime / 60) * 1000 = j.totalDistance

/-- Theorem stating that Li Hai's journey satisfies the system of equations -/
theorem journey_satisfies_equations :
  ∀ (j : Journey),
    j.totalDistance = 1200 ∧
    j.uphillSpeed = 3 ∧
    j.downhillSpeed = 5 ∧
    j.totalTime = 16 →
    satisfiesEquations j :=
  sorry

#check journey_satisfies_equations

end NUMINAMATH_CALUDE_journey_satisfies_equations_l1293_129339


namespace NUMINAMATH_CALUDE_absolute_value_minus_sqrt_l1293_129398

theorem absolute_value_minus_sqrt (a : ℝ) (h : a < -1) : |1 + a| - Real.sqrt (a^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_minus_sqrt_l1293_129398


namespace NUMINAMATH_CALUDE_reflection_matrix_condition_l1293_129388

def reflection_matrix (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, b; -3/2, 1/2]

theorem reflection_matrix_condition (a b : ℚ) :
  (reflection_matrix a b) ^ 2 = 1 ↔ a = -1/2 ∧ b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_reflection_matrix_condition_l1293_129388


namespace NUMINAMATH_CALUDE_dropped_student_score_l1293_129347

theorem dropped_student_score
  (initial_students : ℕ)
  (initial_average : ℚ)
  (remaining_students : ℕ)
  (remaining_average : ℚ)
  (h1 : initial_students = 16)
  (h2 : initial_average = 62.5)
  (h3 : remaining_students = 15)
  (h4 : remaining_average = 62)
  (h5 : initial_students = remaining_students + 1) :
  (initial_students : ℚ) * initial_average - (remaining_students : ℚ) * remaining_average = 70 := by
  sorry

#check dropped_student_score

end NUMINAMATH_CALUDE_dropped_student_score_l1293_129347


namespace NUMINAMATH_CALUDE_solve_candy_bar_problem_l1293_129360

def candy_bar_problem (initial_amount : ℚ) (num_candy_bars : ℕ) (remaining_amount : ℚ) : Prop :=
  ∃ (price_per_bar : ℚ),
    initial_amount - num_candy_bars * price_per_bar = remaining_amount ∧
    price_per_bar > 0

theorem solve_candy_bar_problem :
  candy_bar_problem 4 10 1 → (4 : ℚ) - 1 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_candy_bar_problem_l1293_129360


namespace NUMINAMATH_CALUDE_student_sums_l1293_129312

theorem student_sums (total : ℕ) (right : ℕ) (wrong : ℕ) : 
  total = 48 → 
  wrong = 3 * right → 
  total = right + wrong → 
  wrong = 36 := by sorry

end NUMINAMATH_CALUDE_student_sums_l1293_129312


namespace NUMINAMATH_CALUDE_beidou_timing_accuracy_l1293_129387

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem beidou_timing_accuracy : 
  toScientificNotation 0.0000000099 = ScientificNotation.mk 9.9 (-9) sorry := by
  sorry

end NUMINAMATH_CALUDE_beidou_timing_accuracy_l1293_129387


namespace NUMINAMATH_CALUDE_smallest_triangle_leg_l1293_129327

/-- Represents a 45-45-90 triangle -/
structure Triangle45 where
  hypotenuse : ℝ
  leg : ℝ
  hyp_leg_relation : leg = hypotenuse / Real.sqrt 2

/-- A sequence of four 45-45-90 triangles where the hypotenuse of one is the leg of the next -/
def TriangleSequence (t1 t2 t3 t4 : Triangle45) : Prop :=
  t1.leg = t2.hypotenuse ∧ t2.leg = t3.hypotenuse ∧ t3.leg = t4.hypotenuse

theorem smallest_triangle_leg 
  (t1 t2 t3 t4 : Triangle45) 
  (seq : TriangleSequence t1 t2 t3 t4) 
  (largest_hyp : t1.hypotenuse = 16) : 
  t4.leg = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_leg_l1293_129327


namespace NUMINAMATH_CALUDE_simplify_expression_l1293_129364

theorem simplify_expression : 18 * (8 / 16) * (3 / 27) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1293_129364


namespace NUMINAMATH_CALUDE_product_inequality_l1293_129351

theorem product_inequality (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  (x^2 - 2*x + 2) * (y^2 - 2*y + 2) * (z^2 - 2*z + 2) ≤ (x*y*z)^2 - 2*(x*y*z) + 2 :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_l1293_129351


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1293_129357

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y, x > 0 ∧ y > 0 → x * y > 0) ∧
  (∃ x y, x * y > 0 ∧ ¬(x > 0 ∧ y > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1293_129357


namespace NUMINAMATH_CALUDE_last_digit_77_base_4_l1293_129338

def last_digit_base_4 (n : ℕ) : ℕ :=
  n % 4

theorem last_digit_77_base_4 :
  last_digit_base_4 77 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_77_base_4_l1293_129338


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1293_129383

theorem abs_neg_three_eq_three : |(-3 : ℝ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1293_129383


namespace NUMINAMATH_CALUDE_sum_of_digits_l1293_129352

/-- Given two single-digit numbers a and b, if ab + ba = 202, then a + b = 12 -/
theorem sum_of_digits (a b : ℕ) : 
  a < 10 → b < 10 → (10 * a + b) + (10 * b + a) = 202 → a + b = 12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_l1293_129352


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l1293_129337

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l1293_129337


namespace NUMINAMATH_CALUDE_matrix_addition_a_l1293_129323

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; -1, 3]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![-1, 3; 1, -4]

theorem matrix_addition_a : A + B = !![1, 7; 0, -1] := by sorry

end NUMINAMATH_CALUDE_matrix_addition_a_l1293_129323


namespace NUMINAMATH_CALUDE_rounded_product_less_than_original_l1293_129317

theorem rounded_product_less_than_original
  (x y z : ℝ)
  (hx_pos : x > 0)
  (hy_pos : y > 0)
  (hz_pos : z > 0)
  (hxy : x > 2*y) :
  (x + z) * (y - z) < x * y :=
by sorry

end NUMINAMATH_CALUDE_rounded_product_less_than_original_l1293_129317


namespace NUMINAMATH_CALUDE_slab_rate_calculation_l1293_129389

/-- Given a room with specified dimensions and total flooring cost, 
    calculate the rate per square meter for the slabs. -/
theorem slab_rate_calculation (length width total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : width = 3.75)
    (h3 : total_cost = 16500) :
  total_cost / (length * width) = 800 := by
  sorry

#check slab_rate_calculation

end NUMINAMATH_CALUDE_slab_rate_calculation_l1293_129389


namespace NUMINAMATH_CALUDE_sum_of_hidden_numbers_l1293_129349

/-- Represents a standard six-sided die with faces numbered 1 through 6 -/
def Die := Fin 6

/-- The sum of all numbers on a standard die -/
def dieTotalSum : ℕ := 21

/-- The visible numbers on the seven sides of the stacked dice -/
def visibleNumbers : List ℕ := [2, 3, 4, 4, 5, 5, 6]

/-- The number of dice stacked -/
def numberOfDice : ℕ := 3

/-- Theorem stating that the sum of numbers not visible on the stacked dice is 34 -/
theorem sum_of_hidden_numbers :
  (numberOfDice * dieTotalSum) - (visibleNumbers.sum) = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_hidden_numbers_l1293_129349


namespace NUMINAMATH_CALUDE_cubic_inequality_l1293_129382

theorem cubic_inequality (x y : ℝ) (h : x > y) : ¬(x^3 < y^3 ∨ x^3 = y^3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1293_129382


namespace NUMINAMATH_CALUDE_equation_negative_roots_a_range_l1293_129362

theorem equation_negative_roots_a_range :
  ∀ a : ℝ,
  (∀ x : ℝ, x < 0 → 4^x - 2^(x-1) + a = 0) →
  (-1/2 < a ∧ a ≤ 1/16) :=
by sorry

end NUMINAMATH_CALUDE_equation_negative_roots_a_range_l1293_129362


namespace NUMINAMATH_CALUDE_cats_owned_by_olly_l1293_129378

def shoes_per_animal : ℕ := 4

def num_dogs : ℕ := 3

def num_ferrets : ℕ := 1

def total_shoes : ℕ := 24

def num_cats : ℕ := (total_shoes - (num_dogs + num_ferrets) * shoes_per_animal) / shoes_per_animal

theorem cats_owned_by_olly :
  num_cats = 2 := by sorry

end NUMINAMATH_CALUDE_cats_owned_by_olly_l1293_129378


namespace NUMINAMATH_CALUDE_order_of_four_numbers_l1293_129319

theorem order_of_four_numbers (m n p q : ℝ) 
  (h1 : m < n) 
  (h2 : p < q) 
  (h3 : (p - m) * (p - n) < 0) 
  (h4 : (q - m) * (q - n) < 0) : 
  m < p ∧ p < q ∧ q < n := by sorry

end NUMINAMATH_CALUDE_order_of_four_numbers_l1293_129319


namespace NUMINAMATH_CALUDE_initial_girls_count_l1293_129309

theorem initial_girls_count (p : ℕ) : 
  (60 : ℚ) / 100 * p = 18 ∧ 
  ((60 : ℚ) / 100 * p - 3) / p = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l1293_129309


namespace NUMINAMATH_CALUDE_hexagon_area_l1293_129343

-- Define the hexagon points
def hexagon_points : List (ℤ × ℤ) := [(0, 0), (1, 2), (2, 3), (4, 2), (3, 0), (0, 0)]

-- Function to calculate the area of a polygon given its vertices
def polygon_area (points : List (ℤ × ℤ)) : ℚ :=
  sorry

-- Theorem statement
theorem hexagon_area : polygon_area hexagon_points = 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_l1293_129343


namespace NUMINAMATH_CALUDE_solve_lawyer_problem_l1293_129350

def lawyer_problem (upfront_fee : ℝ) (hourly_rate : ℝ) (court_hours : ℝ) (total_payment : ℝ) : Prop :=
  let court_cost := hourly_rate * court_hours
  let total_cost := upfront_fee + court_cost
  let prep_cost := total_payment - total_cost
  let prep_hours := prep_cost / hourly_rate
  let johns_payment := total_payment / 2
  johns_payment = 4000 ∧ prep_hours / court_hours = 2 / 5

theorem solve_lawyer_problem : 
  lawyer_problem 1000 100 50 8000 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_lawyer_problem_l1293_129350


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_l1293_129333

-- Define the total number of balls and the number of each color
def total_balls : ℕ := 7
def red_balls : ℕ := 5
def black_balls : ℕ := 2

-- Define the number of balls drawn
def drawn_balls : ℕ := 3

-- Define the events
def exactly_one_black (outcome : Finset ℕ) : Prop :=
  outcome.card = drawn_balls ∧ (outcome.filter (λ x => x > red_balls)).card = 1

def exactly_two_black (outcome : Finset ℕ) : Prop :=
  outcome.card = drawn_balls ∧ (outcome.filter (λ x => x > red_balls)).card = 2

-- Theorem statement
theorem mutually_exclusive_not_opposite :
  (∃ outcome, exactly_one_black outcome ∧ exactly_two_black outcome = False) ∧
  (∃ outcome, ¬(exactly_one_black outcome ∨ exactly_two_black outcome)) := by
  sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_l1293_129333


namespace NUMINAMATH_CALUDE_prepaid_card_cost_l1293_129386

/-- The cost of a prepaid phone card given call cost, call duration, and remaining balance -/
theorem prepaid_card_cost 
  (cost_per_minute : ℚ) 
  (call_duration : ℕ) 
  (remaining_balance : ℚ) : 
  cost_per_minute = 16/100 →
  call_duration = 22 →
  remaining_balance = 2648/100 →
  remaining_balance + cost_per_minute * call_duration = 30 := by
sorry

end NUMINAMATH_CALUDE_prepaid_card_cost_l1293_129386


namespace NUMINAMATH_CALUDE_p_range_l1293_129355

-- Define the function p(x)
def p (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

-- State the theorem
theorem p_range :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ p x = y) ↔ y ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_p_range_l1293_129355


namespace NUMINAMATH_CALUDE_sum_equals_thirty_l1293_129336

theorem sum_equals_thirty : 1 + 2 + 3 - 4 + 5 + 6 + 7 - 8 + 9 + 10 + 11 - 12 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_thirty_l1293_129336


namespace NUMINAMATH_CALUDE_inequality_solutions_range_l1293_129359

theorem inequality_solutions_range (a : ℝ) : 
  (∀ x : ℕ+, x < a ↔ x ≤ 5) → (5 < a ∧ a < 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_range_l1293_129359


namespace NUMINAMATH_CALUDE_second_number_in_sum_l1293_129335

theorem second_number_in_sum (a b c : ℝ) : 
  a = 3.15 → c = 0.458 → a + b + c = 3.622 → b = 0.014 := by
  sorry

end NUMINAMATH_CALUDE_second_number_in_sum_l1293_129335


namespace NUMINAMATH_CALUDE_cost_price_is_640_l1293_129322

/-- The cost price of an article given its selling price and profit percentage -/
def costPrice (sellingPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  sellingPrice / (1 + profitPercentage / 100)

/-- Theorem stating that the cost price is 640 given the conditions -/
theorem cost_price_is_640 (sellingPrice : ℚ) (profitPercentage : ℚ) 
  (h1 : sellingPrice = 800)
  (h2 : profitPercentage = 25) : 
  costPrice sellingPrice profitPercentage = 640 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_640_l1293_129322


namespace NUMINAMATH_CALUDE_sprinkles_problem_l1293_129311

theorem sprinkles_problem (initial_cans : ℕ) : 
  (initial_cans / 2 - 3 = 3) → initial_cans = 12 := by
  sorry

end NUMINAMATH_CALUDE_sprinkles_problem_l1293_129311


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1293_129366

theorem consecutive_integers_average (a : ℤ) (c : ℚ) : 
  (a > 0) →
  (c = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7) →
  ((c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 = a + 6) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1293_129366


namespace NUMINAMATH_CALUDE_line_equation_through_points_l1293_129377

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℚ
  y₁ : ℚ
  x₂ : ℚ
  y₂ : ℚ

/-- The slope of a line -/
def Line.slope (l : Line) : ℚ := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)

/-- The y-intercept of a line -/
def Line.yIntercept (l : Line) : ℚ := l.y₁ - l.slope * l.x₁

/-- The equation of a line in the form y = mx + b -/
def Line.equation (l : Line) (x : ℚ) : ℚ := l.slope * x + l.yIntercept

theorem line_equation_through_points :
  let l : Line := { x₁ := 2, y₁ := 3, x₂ := -1, y₂ := -1 }
  ∀ x, l.equation x = (4/3) * x + (1/3) := by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l1293_129377


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1293_129329

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of the ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ :=
  sorry

theorem ellipse_foci_distance :
  let e : ParallelAxisEllipse := ⟨(6, 0), (0, 2)⟩
  foci_distance e = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1293_129329


namespace NUMINAMATH_CALUDE_may_largest_drop_l1293_129392

/-- Represents the months in the first half of the year -/
inductive Month
| January
| February
| March
| April
| May
| June

/-- The price change for each month -/
def price_change (m : Month) : ℝ :=
  match m with
  | .January  => -1.25
  | .February => 2.75
  | .March    => -0.75
  | .April    => 1.50
  | .May      => -3.00
  | .June     => -1.00

/-- Definition of a price drop -/
def is_price_drop (x : ℝ) : Prop := x < 0

/-- The month with the largest price drop -/
def largest_drop (m : Month) : Prop :=
  ∀ n : Month, is_price_drop (price_change n) →
    price_change n ≥ price_change m

theorem may_largest_drop :
  largest_drop Month.May :=
sorry

end NUMINAMATH_CALUDE_may_largest_drop_l1293_129392


namespace NUMINAMATH_CALUDE_math_team_selection_count_l1293_129391

-- Define the number of boys and girls in the math club
def num_boys : ℕ := 10
def num_girls : ℕ := 10

-- Define the required number of boys and girls in the team
def required_boys : ℕ := 4
def required_girls : ℕ := 3

-- Define the total team size
def team_size : ℕ := required_boys + required_girls

-- Theorem statement
theorem math_team_selection_count :
  (Nat.choose num_boys required_boys) * (Nat.choose num_girls required_girls) = 25200 := by
  sorry

end NUMINAMATH_CALUDE_math_team_selection_count_l1293_129391


namespace NUMINAMATH_CALUDE_rectangle_length_equals_two_l1293_129314

theorem rectangle_length_equals_two
  (square_side : ℝ)
  (rectangle_width : ℝ)
  (h1 : square_side = 4)
  (h2 : rectangle_width = 8)
  (h3 : square_side ^ 2 = rectangle_width * rectangle_length) :
  rectangle_length = 2 :=
by
  sorry

#check rectangle_length_equals_two

end NUMINAMATH_CALUDE_rectangle_length_equals_two_l1293_129314


namespace NUMINAMATH_CALUDE_counterexample_exists_negative_four_is_counterexample_l1293_129342

theorem counterexample_exists : ∃ a : ℝ, a < 3 ∧ a^2 ≥ 9 :=
  by
  use -4
  constructor
  · -- Prove -4 < 3
    sorry
  · -- Prove (-4)^2 ≥ 9
    sorry

theorem negative_four_is_counterexample : -4 < 3 ∧ (-4)^2 ≥ 9 :=
  by
  constructor
  · -- Prove -4 < 3
    sorry
  · -- Prove (-4)^2 ≥ 9
    sorry

end NUMINAMATH_CALUDE_counterexample_exists_negative_four_is_counterexample_l1293_129342


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1293_129375

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (2 * x - 6) / (x + 1)) ↔ x ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1293_129375


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l1293_129310

def n : Nat := 483045

theorem largest_prime_factors_difference (p q : Nat) :
  Nat.Prime p ∧ Nat.Prime q ∧
  p ∣ n ∧ q ∣ n ∧
  (∀ r, Nat.Prime r → r ∣ n → r ≤ p ∧ r ≤ q) →
  p ≠ q →
  (max p q) - (min p q) = 8 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l1293_129310


namespace NUMINAMATH_CALUDE_circle_inequality_m_range_l1293_129305

theorem circle_inequality_m_range :
  ∀ m : ℝ,
  (∀ x y : ℝ, x^2 + (y - 1)^2 = 1 → x + y + m ≥ 0) ↔
  m > -1 :=
by sorry

end NUMINAMATH_CALUDE_circle_inequality_m_range_l1293_129305


namespace NUMINAMATH_CALUDE_valid_paintings_count_l1293_129326

/-- Represents a 3x3 grid of squares that can be painted green or red -/
def Grid := Fin 3 → Fin 3 → Bool

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Bool :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Checks if a grid painting is valid (no green square adjacent to a red square) -/
def valid_painting (g : Grid) : Bool :=
  ∀ p1 p2 : Fin 3 × Fin 3, adjacent p1 p2 → (g p1.1 p1.2 = g p2.1 p2.2)

/-- Counts the number of valid grid paintings -/
def count_valid_paintings : Nat :=
  (List.filter valid_painting (List.map (λf : Fin 9 → Bool => λi j => f (3 * i + j)) 
    (List.map (λn : Fin 512 => λi => n.val.testBit i) (List.range 512)))).length

/-- The main theorem stating that the number of valid paintings is 10 -/
theorem valid_paintings_count : count_valid_paintings = 10 := by
  sorry

end NUMINAMATH_CALUDE_valid_paintings_count_l1293_129326


namespace NUMINAMATH_CALUDE_complex_equation_real_part_l1293_129399

theorem complex_equation_real_part (z : ℂ) (a b : ℝ) (h1 : z = a + b * Complex.I) 
  (h2 : b > 0) (h3 : z * (z + 2 * Complex.I) * (z - 2 * Complex.I) * (z + 5 * Complex.I) = 8000) :
  a^3 - 4*a = 8000 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_part_l1293_129399


namespace NUMINAMATH_CALUDE_area_equals_scientific_notation_l1293_129381

-- Define the area of the radio telescope
def telescope_area : ℝ := 250000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.5 * (10 ^ 5)

-- Theorem stating that the area is equal to its scientific notation representation
theorem area_equals_scientific_notation : telescope_area = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_area_equals_scientific_notation_l1293_129381


namespace NUMINAMATH_CALUDE_monthly_parking_fee_l1293_129332

/-- Proves that the monthly parking fee is $40 given the specified conditions -/
theorem monthly_parking_fee (weekly_fee : ℕ) (yearly_savings : ℕ) (weeks_per_year : ℕ) (months_per_year : ℕ) :
  weekly_fee = 10 →
  yearly_savings = 40 →
  weeks_per_year = 52 →
  months_per_year = 12 →
  ∃ (monthly_fee : ℕ), monthly_fee = 40 ∧ weeks_per_year * weekly_fee - months_per_year * monthly_fee = yearly_savings :=
by sorry

end NUMINAMATH_CALUDE_monthly_parking_fee_l1293_129332


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1293_129365

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 2 → a ≥ 1) ∧ (∃ a, a ≥ 1 ∧ a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1293_129365


namespace NUMINAMATH_CALUDE_box_2_neg2_3_l1293_129353

def box (a b c : ℤ) : ℚ := (a ^ b) - (b ^ c) + (c ^ a)

theorem box_2_neg2_3 : box 2 (-2) 3 = 5 / 4 := by sorry

end NUMINAMATH_CALUDE_box_2_neg2_3_l1293_129353


namespace NUMINAMATH_CALUDE_quadratic_equation_root_zero_l1293_129397

theorem quadratic_equation_root_zero (m : ℝ) : 
  (m - 1 ≠ 0) →
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) →
  ((m - 1) * 0^2 + 2 * 0 + m^2 - 1 = 0) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_zero_l1293_129397


namespace NUMINAMATH_CALUDE_sum_of_odd_divisors_90_l1293_129308

/-- The sum of the positive odd divisors of 90 -/
def sumOfOddDivisors90 : ℕ := sorry

/-- Theorem stating that the sum of the positive odd divisors of 90 is 78 -/
theorem sum_of_odd_divisors_90 : sumOfOddDivisors90 = 78 := by sorry

end NUMINAMATH_CALUDE_sum_of_odd_divisors_90_l1293_129308


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1293_129344

theorem complex_equation_solution (z : ℂ) :
  (3 + 4*I) * z = 1 - 2*I → z = -1/5 - 2/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1293_129344


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1293_129318

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := 4 * i / (1 + i)
  Complex.im z = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1293_129318


namespace NUMINAMATH_CALUDE_tangent_line_sum_l1293_129330

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : ∀ x, f 1 + 3 * (x - 1) = 3 * x - 2) : 
  f 1 + (deriv f) 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l1293_129330


namespace NUMINAMATH_CALUDE_number_problem_l1293_129374

theorem number_problem (x : ℝ) : (36 / 100 * x = 129.6) → x = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1293_129374


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_two_tan_l1293_129373

theorem sqrt_difference_equals_negative_two_tan (α : Real) 
  (h : α ∈ Set.Ioo (-Real.pi) (-Real.pi/2)) : 
  Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) - 
  Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) = 
  -2 * Real.tan α :=
by sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_two_tan_l1293_129373


namespace NUMINAMATH_CALUDE_divide_decimals_l1293_129396

theorem divide_decimals : (0.25 : ℚ) / (0.005 : ℚ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_divide_decimals_l1293_129396


namespace NUMINAMATH_CALUDE_line_passes_through_point_l1293_129321

theorem line_passes_through_point (A B C : ℝ) :
  A - B + C = 0 →
  ∀ (x y : ℝ), A * x + B * y + C = 0 ↔ (x = 1 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l1293_129321


namespace NUMINAMATH_CALUDE_book_cost_price_l1293_129385

theorem book_cost_price (selling_price : ℝ) (profit_percentage : ℝ) (cost_price : ℝ) : 
  selling_price = 270 → 
  profit_percentage = 20 → 
  selling_price = cost_price * (1 + profit_percentage / 100) → 
  cost_price = 225 := by
sorry

end NUMINAMATH_CALUDE_book_cost_price_l1293_129385


namespace NUMINAMATH_CALUDE_range_of_a_l1293_129356

/-- A monotonically decreasing function on [-2, 2] -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc (-2) 2 → y ∈ Set.Icc (-2) 2 → x < y → f x > f y

theorem range_of_a (f : ℝ → ℝ) (h1 : MonoDecreasing f) 
    (h2 : ∀ a, f (a + 1) < f (2 * a)) :
    Set.Icc (-1) 1 \ {1} = {a | a + 1 ∈ Set.Icc (-2) 2 ∧ 2 * a ∈ Set.Icc (-2) 2 ∧ f (a + 1) < f (2 * a)} :=
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1293_129356


namespace NUMINAMATH_CALUDE_fourDigitNumbersTheorem_l1293_129358

/-- Represents the multiset of numbers on the cards -/
def cardNumbers : Multiset ℕ := {1, 1, 1, 2, 2, 3, 4}

/-- Number of cards drawn -/
def cardsDrawn : ℕ := 4

/-- Function to calculate the number of different four-digit numbers -/
def fourDigitNumbersCount (cards : Multiset ℕ) (drawn : ℕ) : ℕ := sorry

/-- Theorem stating that the number of different four-digit numbers is 114 -/
theorem fourDigitNumbersTheorem : fourDigitNumbersCount cardNumbers cardsDrawn = 114 := by
  sorry

end NUMINAMATH_CALUDE_fourDigitNumbersTheorem_l1293_129358


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1293_129304

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 + 2 * t^2 - 4 * t + 1) * (-2 * t^2 + 3 * t - 5) =
  -6 * t^5 + 5 * t^4 - t^3 - 24 * t^2 + 23 * t - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1293_129304


namespace NUMINAMATH_CALUDE_triangle_inequality_variant_l1293_129361

theorem triangle_inequality_variant (x y z : ℝ) :
  (|x| < |y - z| ∧ |y| < |z - x|) → |z| ≥ |x - y| := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_variant_l1293_129361


namespace NUMINAMATH_CALUDE_inequality_proof_l1293_129363

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h : (a + b + c + d) * (1/a + 1/b + 1/c + 1/d) = 20) :
  (a^2 + b^2 + c^2 + d^2) * (1/a^2 + 1/b^2 + 1/c^2 + 1/d^2) ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1293_129363


namespace NUMINAMATH_CALUDE_volunteer_selection_schemes_l1293_129307

def num_candidates : ℕ := 5
def num_volunteers : ℕ := 4
def num_jobs : ℕ := 4

def driver_only_volunteer : ℕ := 1
def versatile_volunteers : ℕ := num_candidates - driver_only_volunteer

theorem volunteer_selection_schemes :
  (versatile_volunteers.factorial / (versatile_volunteers - (num_jobs - 1)).factorial) +
  (versatile_volunteers.factorial / (versatile_volunteers - num_jobs).factorial) = 48 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_selection_schemes_l1293_129307


namespace NUMINAMATH_CALUDE_exists_natural_not_in_five_gp_l1293_129320

/-- A geometric progression with integer terms -/
structure GeometricProgression where
  first_term : ℤ
  common_ratio : ℤ
  common_ratio_nonzero : common_ratio ≠ 0

/-- The nth term of a geometric progression -/
def GeometricProgression.nth_term (gp : GeometricProgression) (n : ℕ) : ℤ :=
  gp.first_term * gp.common_ratio ^ n

/-- Theorem: There exists a natural number not in any of five given geometric progressions -/
theorem exists_natural_not_in_five_gp (gp1 gp2 gp3 gp4 gp5 : GeometricProgression) :
  ∃ (k : ℕ), (∀ n : ℕ, gp1.nth_term n ≠ k) ∧
             (∀ n : ℕ, gp2.nth_term n ≠ k) ∧
             (∀ n : ℕ, gp3.nth_term n ≠ k) ∧
             (∀ n : ℕ, gp4.nth_term n ≠ k) ∧
             (∀ n : ℕ, gp5.nth_term n ≠ k) :=
  sorry

end NUMINAMATH_CALUDE_exists_natural_not_in_five_gp_l1293_129320


namespace NUMINAMATH_CALUDE_harrys_seed_purchase_l1293_129328

/-- Represents the number of packets of each seed type and the total spent -/
structure SeedPurchase where
  pumpkin : ℕ
  tomato : ℕ
  chili : ℕ
  total_spent : ℚ

/-- Calculates the total cost of a seed purchase -/
def calculate_total_cost (purchase : SeedPurchase) : ℚ :=
  2.5 * purchase.pumpkin + 1.5 * purchase.tomato + 0.9 * purchase.chili

/-- Theorem stating that Harry's purchase of 3 pumpkin, 4 tomato, and 5 chili pepper seed packets
    totaling $18 is correct -/
theorem harrys_seed_purchase :
  ∃ (purchase : SeedPurchase),
    purchase.pumpkin = 3 ∧
    purchase.tomato = 4 ∧
    purchase.chili = 5 ∧
    purchase.total_spent = 18 ∧
    calculate_total_cost purchase = purchase.total_spent :=
  sorry

end NUMINAMATH_CALUDE_harrys_seed_purchase_l1293_129328


namespace NUMINAMATH_CALUDE_sum_of_digits_of_expression_l1293_129324

-- Define the expression
def expression : ℕ := (2 + 4)^15

-- Function to get the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Function to get the ones digit of a number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem sum_of_digits_of_expression :
  tens_digit expression + ones_digit expression = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_expression_l1293_129324


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l1293_129348

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- The property that g satisfies the given conditions -/
def SatisfiesConditions (g : ThirdDegreePolynomial) : Prop :=
  ∀ x : ℝ, x ∈ ({-1, 0, 2, 4, 5, 8} : Set ℝ) → |g x| = 10

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : SatisfiesConditions g) : |g 3| = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_property_l1293_129348


namespace NUMINAMATH_CALUDE_equation_solution_l1293_129340

theorem equation_solution : ∃! x : ℚ, 2 * x - 5/6 = 7/18 + 1/2 ∧ x = 31/36 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1293_129340


namespace NUMINAMATH_CALUDE_square_root_problem_l1293_129306

theorem square_root_problem (x : ℝ) :
  (Real.sqrt 1.21 / Real.sqrt 0.64) + (Real.sqrt x / Real.sqrt 0.49) = 3.0892857142857144 →
  x = 1.44 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1293_129306


namespace NUMINAMATH_CALUDE_broken_line_coverage_coin_covers_broken_line_l1293_129395

/-- A closed broken line in a 2D plane -/
structure ClosedBrokenLine where
  points : Set (ℝ × ℝ)
  is_closed : True  -- Placeholder for the closed property
  length : ℝ

/-- Theorem: Any closed broken line of length 5 can be covered by a circle of radius 1.25 -/
theorem broken_line_coverage (L : ClosedBrokenLine) (h : L.length = 5) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ L.points → dist center p ≤ 1.25 := by
  sorry

/-- Corollary: A coin with diameter > 2.5 can cover a 5 cm closed broken line -/
theorem coin_covers_broken_line (L : ClosedBrokenLine) (h : L.length = 5) 
  (coin_diameter : ℝ) (hd : coin_diameter > 2.5) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ L.points → dist center p ≤ coin_diameter / 2 := by
  sorry

end NUMINAMATH_CALUDE_broken_line_coverage_coin_covers_broken_line_l1293_129395


namespace NUMINAMATH_CALUDE_committee_probability_l1293_129393

/-- The probability of selecting exactly 2 boys in a 5-person committee
    chosen randomly from a group of 30 members (12 boys and 18 girls) -/
theorem committee_probability (total : Nat) (boys : Nat) (girls : Nat) (committee_size : Nat) :
  total = 30 →
  boys = 12 →
  girls = 18 →
  committee_size = 5 →
  (Nat.choose boys 2 * Nat.choose girls 3 : ℚ) / Nat.choose total committee_size = 26928 / 71253 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l1293_129393


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l1293_129331

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x - 3 = 0) ∧ (1^2 + k*1 - 3 = 0) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l1293_129331


namespace NUMINAMATH_CALUDE_least_period_scaled_least_period_sum_sine_cosine_least_period_sin_cos_least_period_cos_sin_l1293_129341

-- Definition of periodic function
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Definition of least period
def least_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  is_periodic f T ∧ ∀ T', 0 < T' ∧ T' < T → ¬ is_periodic f T'

-- Theorem 1
theorem least_period_scaled (g : ℝ → ℝ) :
  least_period g π → least_period (fun x ↦ g (x / 3)) (3 * π) := by sorry

-- Theorem 2
theorem least_period_sum_sine_cosine :
  least_period (fun x ↦ Real.sin (8 * x) + Real.cos (4 * x)) (π / 2) := by sorry

-- Theorem 3
theorem least_period_sin_cos :
  least_period (fun x ↦ Real.sin (Real.cos x)) (2 * π) := by sorry

-- Theorem 4
theorem least_period_cos_sin :
  least_period (fun x ↦ Real.cos (Real.sin x)) π := by sorry

end NUMINAMATH_CALUDE_least_period_scaled_least_period_sum_sine_cosine_least_period_sin_cos_least_period_cos_sin_l1293_129341


namespace NUMINAMATH_CALUDE_greatest_power_of_three_dividing_30_factorial_l1293_129301

-- Define v as 30!
def v : ℕ := Nat.factorial 30

-- Define the property that 3^k is a factor of v
def is_factor_of_v (k : ℕ) : Prop := ∃ m : ℕ, v = m * (3^k)

-- Theorem statement
theorem greatest_power_of_three_dividing_30_factorial :
  (∃ k : ℕ, is_factor_of_v k ∧ ∀ j : ℕ, j > k → ¬is_factor_of_v j) ∧
  (∀ k : ℕ, (∃ j : ℕ, j > k ∧ is_factor_of_v j) → k ≤ 14) ∧
  is_factor_of_v 14 := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_dividing_30_factorial_l1293_129301


namespace NUMINAMATH_CALUDE_max_tickets_proof_l1293_129316

/-- Represents the maximum number of tickets Jane can buy given the following conditions:
  * Each ticket costs $15
  * Jane has a budget of $180
  * If more than 10 tickets are bought, there's a discount of $2 per ticket
-/
def max_tickets : ℕ := 13

/-- The cost of a ticket without discount -/
def ticket_cost : ℕ := 15

/-- Jane's budget -/
def budget : ℕ := 180

/-- The discount per ticket when buying more than 10 tickets -/
def discount : ℕ := 2

/-- The threshold for applying the discount -/
def discount_threshold : ℕ := 10

theorem max_tickets_proof :
  (∀ n : ℕ, n ≤ discount_threshold → n * ticket_cost ≤ budget) ∧
  (∀ n : ℕ, n > discount_threshold → n * (ticket_cost - discount) ≤ budget) ∧
  (∀ n : ℕ, n > max_tickets → 
    (if n ≤ discount_threshold then n * ticket_cost > budget
     else n * (ticket_cost - discount) > budget)) :=
sorry

end NUMINAMATH_CALUDE_max_tickets_proof_l1293_129316


namespace NUMINAMATH_CALUDE_expression_simplification_l1293_129380

theorem expression_simplification (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (3 * x^2 - 2 * x) / ((x - 3) * (x + 2)) - (5 * x - 6) / ((x - 3) * (x + 2)) = (3 * x - 2) / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1293_129380


namespace NUMINAMATH_CALUDE_new_person_weight_l1293_129390

/-- Given a group of 8 people, if replacing a person weighing 75 kg with a new person
    increases the average weight by 2.5 kg, then the weight of the new person is 95 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 75 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 95 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1293_129390


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l1293_129394

theorem quadratic_perfect_square (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 150*x + c = (x + a)^2) → c = 5625 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l1293_129394


namespace NUMINAMATH_CALUDE_sin_cos_sum_13_17_l1293_129313

theorem sin_cos_sum_13_17 :
  Real.sin (13 * π / 180) * Real.cos (17 * π / 180) +
  Real.cos (13 * π / 180) * Real.sin (17 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_13_17_l1293_129313
