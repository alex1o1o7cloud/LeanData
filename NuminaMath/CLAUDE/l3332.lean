import Mathlib

namespace NUMINAMATH_CALUDE_expression_value_l3332_333295

theorem expression_value (a b c d e f : ℝ) 
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : |e| = Real.sqrt 2)
  (h4 : Real.sqrt f = 8) :
  (1/2) * a * b + (c + d) / 5 + e^2 + f^(1/3) = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3332_333295


namespace NUMINAMATH_CALUDE_unique_number_exists_l3332_333257

theorem unique_number_exists : ∃! N : ℕ, 
  (∃ Q : ℕ, N = 11 * Q) ∧ 
  (N / 11 + N + 11 = 71) := by
sorry

end NUMINAMATH_CALUDE_unique_number_exists_l3332_333257


namespace NUMINAMATH_CALUDE_inscribed_rectangles_area_sum_l3332_333284

/-- A structure representing a rectangle --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- A structure representing two inscribed rectangles sharing a common vertex --/
structure InscribedRectangles where
  outer : Rectangle
  common_vertex : ℝ  -- Position of K on AB, 0 ≤ common_vertex ≤ outer.width

/-- Calculate the area of a rectangle --/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculate the sum of areas of the two inscribed rectangles --/
def InscribedRectangles.sumOfAreas (ir : InscribedRectangles) : ℝ :=
  ir.common_vertex * ir.outer.height

/-- Theorem stating that the sum of areas of inscribed rectangles equals the area of the outer rectangle --/
theorem inscribed_rectangles_area_sum (ir : InscribedRectangles) :
  ir.sumOfAreas = ir.outer.area := by sorry

end NUMINAMATH_CALUDE_inscribed_rectangles_area_sum_l3332_333284


namespace NUMINAMATH_CALUDE_factorization_equality_l3332_333254

theorem factorization_equality (m : ℝ) : m^2 * (m - 1) + 4 * (1 - m) = (m - 1) * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3332_333254


namespace NUMINAMATH_CALUDE_basketball_team_selection_l3332_333276

def total_players : ℕ := 12
def team_size : ℕ := 5
def captain_count : ℕ := 1
def regular_player_count : ℕ := 4

theorem basketball_team_selection :
  (total_players.choose captain_count) * ((total_players - captain_count).choose regular_player_count) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l3332_333276


namespace NUMINAMATH_CALUDE_fraction_sum_rounded_l3332_333296

theorem fraction_sum_rounded : 
  let sum := (3 : ℚ) / 20 + 7 / 200 + 8 / 2000 + 3 / 20000
  round (sum * 10000) / 10000 = (1892 : ℚ) / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_rounded_l3332_333296


namespace NUMINAMATH_CALUDE_kim_total_water_consumption_l3332_333210

/-- The amount of water Kim drinks from various sources -/
def kim_water_consumption (quart_to_ounce : Real) (bottle_quarts : Real) (can_ounces : Real) 
  (shared_bottle_ounces : Real) (jake_fraction : Real) : Real :=
  let bottle_ounces := bottle_quarts * quart_to_ounce
  let kim_shared_fraction := 1 - jake_fraction
  bottle_ounces + can_ounces + (kim_shared_fraction * shared_bottle_ounces)

/-- Theorem stating that Kim's total water consumption is 79.2 ounces -/
theorem kim_total_water_consumption :
  kim_water_consumption 32 1.5 12 32 (2/5) = 79.2 := by
  sorry

end NUMINAMATH_CALUDE_kim_total_water_consumption_l3332_333210


namespace NUMINAMATH_CALUDE_sequence_properties_l3332_333272

/-- Sequence a_n with given properties -/
def sequence_a (n : ℕ) : ℝ :=
  sorry

/-- Sum of first n terms of sequence a_n -/
def S (n : ℕ) : ℝ :=
  sorry

/-- Sum of first n terms of sequence a_n / 2^n -/
def T (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating the properties of the sequence and its sums -/
theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_a n / S n = 2 / (n + 1)) ∧
  sequence_a 1 = 1 →
  (∀ n : ℕ, n ≥ 1 → sequence_a n = n) ∧
  (∀ n : ℕ, n ≥ 1 → T n = 2 - (n + 2) * (1/2)^n) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3332_333272


namespace NUMINAMATH_CALUDE_jonessa_take_home_pay_l3332_333270

/-- Given Jonessa's pay and tax rate, calculate her take-home pay -/
theorem jonessa_take_home_pay (total_pay : ℝ) (tax_rate : ℝ) 
  (h1 : total_pay = 500)
  (h2 : tax_rate = 0.1) : 
  total_pay * (1 - tax_rate) = 450 := by
sorry

end NUMINAMATH_CALUDE_jonessa_take_home_pay_l3332_333270


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l3332_333207

/-- Given a journey with the following properties:
  * Total distance is 112 km
  * Total time is 5 hours
  * The first half is traveled at 21 km/hr
  Prove that the speed for the second half is 24 km/hr -/
theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ)
  (h1 : total_distance = 112)
  (h2 : total_time = 5)
  (h3 : first_half_speed = 21)
  : (2 * total_distance) / (2 * total_time - total_distance / first_half_speed) = 24 :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l3332_333207


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3332_333279

/-- A polynomial with integer coefficients of the form x^3 + b₂x^2 + b₁x + 18 = 0 -/
def IntPolynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ :=
  x^3 + b₂ * x^2 + b₁ * x + 18

/-- The set of all possible integer roots of the polynomial -/
def PossibleRoots : Set ℤ :=
  {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  ∀ x : ℤ, IntPolynomial b₂ b₁ x = 0 → x ∈ PossibleRoots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3332_333279


namespace NUMINAMATH_CALUDE_closest_point_l3332_333221

/-- The vector v as a function of t -/
def v (t : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 1 + 5*t
  | 1 => -2 + 4*t
  | 2 => -4 - 2*t

/-- The vector a -/
def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3
  | 1 => 2
  | 2 => 6

/-- The direction vector of v -/
def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 4
  | 2 => -2

/-- Theorem: The value of t that minimizes the distance between v and a is 2/15 -/
theorem closest_point : 
  (∀ t : ℝ, (v t - a) • direction = 0 → t = 2/15) ∧ 
  (v (2/15) - a) • direction = 0 := by
  sorry

end NUMINAMATH_CALUDE_closest_point_l3332_333221


namespace NUMINAMATH_CALUDE_tree_planting_cost_l3332_333241

/-- The cost to plant one tree given temperature drop and total cost -/
theorem tree_planting_cost 
  (temp_drop_per_tree : ℝ) 
  (total_temp_drop : ℝ) 
  (total_cost : ℝ) : 
  temp_drop_per_tree = 0.1 → 
  total_temp_drop = 1.8 → 
  total_cost = 108 → 
  (total_cost / (total_temp_drop / temp_drop_per_tree) = 6) :=
by
  sorry

#check tree_planting_cost

end NUMINAMATH_CALUDE_tree_planting_cost_l3332_333241


namespace NUMINAMATH_CALUDE_sum_of_squares_l3332_333267

/-- Given a sequence {aₙ} where the sum of its first n terms S = 2n - 1,
    T is the sum of the first n terms of the sequence {aₙ²} -/
def T (n : ℕ) : ℚ :=
  (16^n - 1) / 15

/-- The sum of the first n terms of the original sequence -/
def S (n : ℕ) : ℕ :=
  2 * n - 1

/-- Theorem stating that T is the correct sum for the sequence {aₙ²} -/
theorem sum_of_squares (n : ℕ) : T n = (16^n - 1) / 15 :=
  by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3332_333267


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3332_333287

/-- Given an arithmetic progression where the sum of n terms is 5n + 4n^2,
    prove that the r-th term is 8r + 1 -/
theorem arithmetic_progression_rth_term (n : ℕ) (r : ℕ) :
  (∀ n, ∃ S : ℕ → ℕ, S n = 5*n + 4*n^2) →
  ∃ a : ℕ → ℕ, a r = 8*r + 1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3332_333287


namespace NUMINAMATH_CALUDE_matrix_not_invertible_l3332_333288

def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2 + x, 9],
    ![4 - x, 10]]

theorem matrix_not_invertible (x : ℝ) :
  ¬(IsUnit (A x).det) ↔ x = 16 / 19 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_l3332_333288


namespace NUMINAMATH_CALUDE_circle_intersections_l3332_333253

/-- A circle C with equation x^2 + y^2 - 2x - 4y - 4 = 0 -/
def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y - 4 = 0

/-- x₁ and x₂ are x-coordinates of intersection points with x-axis -/
def x_intersections (x₁ x₂ : ℝ) : Prop := C x₁ 0 ∧ C x₂ 0 ∧ x₁ ≠ x₂

/-- y₁ and y₂ are y-coordinates of intersection points with y-axis -/
def y_intersections (y₁ y₂ : ℝ) : Prop := C 0 y₁ ∧ C 0 y₂ ∧ y₁ ≠ y₂

theorem circle_intersections 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (hx : x_intersections x₁ x₂) 
  (hy : y_intersections y₁ y₂) : 
  abs (x₁ - x₂) = 2 * Real.sqrt 5 ∧ 
  y₁ + y₂ = 4 ∧ 
  x₁ * x₂ = y₁ * y₂ := by
  sorry

end NUMINAMATH_CALUDE_circle_intersections_l3332_333253


namespace NUMINAMATH_CALUDE_derivative_of_y_l3332_333275

noncomputable def y (x : ℝ) : ℝ := (Real.sin (x^2))^3

theorem derivative_of_y (x : ℝ) :
  deriv y x = 3 * x * Real.sin (x^2) * Real.sin (2 * x^2) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_y_l3332_333275


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3332_333234

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a)
  (h_eq : a 2 * a 4 * a 5 = a 3 * a 6)
  (h_prod : a 9 * a 10 = -8) :
  a 7 = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3332_333234


namespace NUMINAMATH_CALUDE_function_properties_l3332_333206

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - 4| + |x - a|

-- State the theorem
theorem function_properties (a : ℝ) 
  (h1 : ∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m ∧ ∃ (y : ℝ), f a y = m) 
  (h2 : ∀ (x : ℝ), f a x ≥ a) :
  (a = 2) ∧ 
  (∀ (x : ℝ), f 2 x ≤ 5 ↔ 1/2 ≤ x ∧ x ≤ 11/2) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l3332_333206


namespace NUMINAMATH_CALUDE_percentage_loss_calculation_l3332_333255

theorem percentage_loss_calculation (cost_price selling_price : ℝ) :
  cost_price = 1600 →
  selling_price = 1440 →
  (cost_price - selling_price) / cost_price * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_loss_calculation_l3332_333255


namespace NUMINAMATH_CALUDE_kindergarten_sample_size_l3332_333285

/-- Represents a kindergarten with students and a height measurement sample -/
structure Kindergarten where
  total_students : ℕ
  sample_size : ℕ

/-- Defines the sample size of a kindergarten height measurement -/
def sample_size (k : Kindergarten) : ℕ := k.sample_size

/-- Theorem: The sample size of the kindergarten height measurement is 31 -/
theorem kindergarten_sample_size :
  ∀ (k : Kindergarten),
  k.total_students = 310 →
  k.sample_size = 31 →
  sample_size k = 31 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_sample_size_l3332_333285


namespace NUMINAMATH_CALUDE_probability_of_123456_l3332_333235

def num_cards : ℕ := 12
def num_distinct : ℕ := 6

def total_arrangements : ℕ := (Finset.prod (Finset.range num_distinct) (fun i => Nat.choose (num_cards - 2*i) 2))

def favorable_arrangements : ℕ := (Finset.prod (Finset.range num_distinct) (fun i => 2*i + 1))

theorem probability_of_123456 :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 720 :=
sorry

end NUMINAMATH_CALUDE_probability_of_123456_l3332_333235


namespace NUMINAMATH_CALUDE_tape_length_sum_l3332_333236

/-- Given three tapes A, B, and C with the following properties:
  * The length of tape A is 35 cm
  * The length of tape A is half the length of tape B
  * The length of tape C is 21 cm less than twice the length of tape A
  Prove that the sum of the lengths of tape B and tape C is 119 cm -/
theorem tape_length_sum (length_A length_B length_C : ℝ) : 
  length_A = 35 →
  length_A = length_B / 2 →
  length_C = 2 * length_A - 21 →
  length_B + length_C = 119 := by
  sorry

end NUMINAMATH_CALUDE_tape_length_sum_l3332_333236


namespace NUMINAMATH_CALUDE_snail_noodles_problem_l3332_333231

/-- Snail noodles problem -/
theorem snail_noodles_problem 
  (price_A : ℝ) 
  (price_B : ℝ) 
  (quantity_A : ℝ) 
  (quantity_B : ℝ) 
  (h1 : price_A * quantity_A = 800)
  (h2 : price_B * quantity_B = 900)
  (h3 : price_B = 1.5 * price_A)
  (h4 : quantity_B = quantity_A - 2)
  (h5 : ∀ a : ℝ, 0 ≤ a ∧ a ≤ 15 → 
    90 * a + 135 * (30 - a) ≥ 90 * 15 + 135 * 15) :
  price_A = 100 ∧ price_B = 150 ∧ 
  (∃ (a : ℝ), 0 ≤ a ∧ a ≤ 15 ∧ 
    90 * a + 135 * (30 - a) = 3375 ∧
    ∀ (b : ℝ), 0 ≤ b ∧ b ≤ 15 → 
      90 * b + 135 * (30 - b) ≥ 3375) :=
sorry

end NUMINAMATH_CALUDE_snail_noodles_problem_l3332_333231


namespace NUMINAMATH_CALUDE_min_translation_for_symmetry_l3332_333216

/-- The minimum positive translation that makes the graph of a sine function symmetric about the origin -/
theorem min_translation_for_symmetry (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = 2 * Real.sin (x + π / 3 - φ)) →
  φ > 0 →
  (∀ x, f x = -f (-x)) →
  φ ≥ π / 3 ∧ 
  ∃ (φ_min : ℝ), φ_min = π / 3 ∧ 
    ∀ (ψ : ℝ), ψ > 0 → 
      (∀ x, 2 * Real.sin (x + π / 3 - ψ) = -(2 * Real.sin (-x + π / 3 - ψ))) → 
      ψ ≥ φ_min :=
by sorry


end NUMINAMATH_CALUDE_min_translation_for_symmetry_l3332_333216


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3332_333200

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 3 = 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 8 = 1) ∧ 
  (n % 7 = 2) ∧ 
  (∀ m : ℕ, m > 1 → m % 3 = 1 → m % 5 = 1 → m % 8 = 1 → m % 7 = 2 → m ≥ n) ∧
  n = 481 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3332_333200


namespace NUMINAMATH_CALUDE_problem_statement_l3332_333280

theorem problem_statement (x y z : ℝ) 
  (h1 : (1/x) + (2/y) + (3/z) = 0)
  (h2 : (1/x) - (6/y) - (5/z) = 0) :
  (x/y) + (y/z) + (z/x) = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3332_333280


namespace NUMINAMATH_CALUDE_all_propositions_false_l3332_333246

-- Define a plane α
variable (α : Set (ℝ × ℝ × ℝ))

-- Define lines in 3D space
def Line3D : Type := Set (ℝ × ℝ × ℝ)

-- Define the projection of a line onto a plane
def project (l : Line3D) (p : Set (ℝ × ℝ × ℝ)) : Line3D := sorry

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define parallel lines
def parallel (l1 l2 : Line3D) : Prop := sorry

-- Define intersecting lines
def intersect (l1 l2 : Line3D) : Prop := sorry

-- Define coincident lines
def coincide (l1 l2 : Line3D) : Prop := sorry

-- Define a line not on a plane
def not_on_plane (l : Line3D) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

theorem all_propositions_false (α : Set (ℝ × ℝ × ℝ)) :
  ∀ (m n : Line3D),
    not_on_plane m α → not_on_plane n α →
    (¬ (perpendicular (project m α) (project n α) → perpendicular m n)) ∧
    (¬ (perpendicular m n → perpendicular (project m α) (project n α))) ∧
    (¬ (intersect (project m α) (project n α) → intersect m n ∨ coincide m n)) ∧
    (¬ (parallel (project m α) (project n α) → parallel m n ∨ coincide m n)) :=
by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l3332_333246


namespace NUMINAMATH_CALUDE_sunscreen_price_proof_l3332_333264

/-- Calculates the discounted price of sunscreen for a year --/
def discounted_sunscreen_price (bottles_per_month : ℕ) (months_per_year : ℕ) 
  (price_per_bottle : ℚ) (discount_percentage : ℚ) : ℚ :=
  let total_bottles := bottles_per_month * months_per_year
  let total_price := total_bottles * price_per_bottle
  let discount_amount := total_price * (discount_percentage / 100)
  total_price - discount_amount

/-- Proves that the discounted price of sunscreen for a year is $252.00 --/
theorem sunscreen_price_proof :
  discounted_sunscreen_price 1 12 30 30 = 252 := by
  sorry

#eval discounted_sunscreen_price 1 12 30 30

end NUMINAMATH_CALUDE_sunscreen_price_proof_l3332_333264


namespace NUMINAMATH_CALUDE_set_membership_solution_l3332_333225

theorem set_membership_solution (x : ℝ) :
  let A : Set ℝ := {2, x, x^2 + x}
  6 ∈ A → x = 6 ∨ x = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_set_membership_solution_l3332_333225


namespace NUMINAMATH_CALUDE_incorrect_proportion_l3332_333251

theorem incorrect_proportion (a b m n : ℝ) (h : a * b = m * n) :
  ¬(m / a = n / b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_proportion_l3332_333251


namespace NUMINAMATH_CALUDE_window_area_ratio_l3332_333281

/-- Represents the window design with a rectangle and semicircles at each end -/
structure WindowDesign where
  /-- Total length of the window, including semicircles -/
  ad : ℝ
  /-- Diameter of the semicircles (width of the window) -/
  ab : ℝ
  /-- Ratio of total length to semicircle diameter is 4:3 -/
  ratio_condition : ad / ab = 4 / 3
  /-- The width of the window is 40 inches -/
  width_condition : ab = 40

/-- The ratio of the rectangle area to the semicircles area is 8/(3π) -/
theorem window_area_ratio (w : WindowDesign) :
  let r := w.ab / 2  -- radius of semicircles
  let rect_length := w.ad - w.ab  -- length of rectangle
  let rect_area := rect_length * w.ab  -- area of rectangle
  let semicircles_area := π * r^2  -- area of semicircles (full circle)
  rect_area / semicircles_area = 8 / (3 * π) := by
  sorry

end NUMINAMATH_CALUDE_window_area_ratio_l3332_333281


namespace NUMINAMATH_CALUDE_eleven_pictures_left_to_color_l3332_333211

/-- The number of pictures left to color given two coloring books and some already colored pictures. -/
def pictures_left_to_color (book1_pictures book2_pictures colored_pictures : ℕ) : ℕ :=
  book1_pictures + book2_pictures - colored_pictures

/-- Theorem stating that given the specific numbers in the problem, 11 pictures are left to color. -/
theorem eleven_pictures_left_to_color :
  pictures_left_to_color 23 32 44 = 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_pictures_left_to_color_l3332_333211


namespace NUMINAMATH_CALUDE_number_plus_seven_equals_six_l3332_333205

theorem number_plus_seven_equals_six : 
  ∃ x : ℤ, x + 7 = 6 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_seven_equals_six_l3332_333205


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3332_333273

theorem expression_simplification_and_evaluation :
  let m : ℝ := Real.sqrt 3
  (m - (m + 9) / (m + 1)) / ((m^2 + 3*m) / (m + 1)) = 1 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3332_333273


namespace NUMINAMATH_CALUDE_power_calculation_l3332_333259

theorem power_calculation : ((16^10 / 16^8)^3 * 8^3) / 2^9 = 16777216 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l3332_333259


namespace NUMINAMATH_CALUDE_power_product_equals_128_l3332_333265

theorem power_product_equals_128 (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_128_l3332_333265


namespace NUMINAMATH_CALUDE_sin_2alpha_equals_one_minus_p_squared_l3332_333271

theorem sin_2alpha_equals_one_minus_p_squared (α : ℝ) (p : ℝ) 
  (h : Real.sin α - Real.cos α = p) : 
  Real.sin (2 * α) = 1 - p^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_equals_one_minus_p_squared_l3332_333271


namespace NUMINAMATH_CALUDE_circle_trajectory_l3332_333250

/-- Given two circles and a line of symmetry, prove the trajectory of a third circle's center -/
theorem circle_trajectory (a l : ℝ) :
  let circle1 := {(x, y) : ℝ × ℝ | x^2 + y^2 - a*x + 2*y + 1 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let symmetry_line := {(x, y) : ℝ × ℝ | y = x - l}
  let point_C := (-a, a)
  ∃ (center : ℝ × ℝ → Prop),
    (∀ (x y : ℝ), center (x, y) ↔ 
      ((x + a)^2 + (y - a)^2 = x^2) ∧  -- P passes through C(-a, a) and is tangent to y-axis
      (∃ (r : ℝ), ∀ (p : ℝ × ℝ), p ∈ circle1 ↔ 
        ∃ (q : ℝ × ℝ), q ∈ circle2 ∧ 
          (p.1 + q.1) / 2 = (p.2 + q.2) / 2 - l)) → -- symmetry condition
    (∀ (x y : ℝ), center (x, y) ↔ y^2 + 4*x - 4*y + 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_trajectory_l3332_333250


namespace NUMINAMATH_CALUDE_co_molecular_weight_l3332_333244

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (carbon_count oxygen_count : ℕ) : ℝ :=
  carbon_count * carbon_weight + oxygen_count * oxygen_weight

/-- Theorem: The molecular weight of CO is 28.01 g/mol -/
theorem co_molecular_weight :
  molecular_weight 1 1 = 28.01 := by sorry

end NUMINAMATH_CALUDE_co_molecular_weight_l3332_333244


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_of_2014000000_l3332_333282

theorem fifth_largest_divisor_of_2014000000 :
  ∃ (d : ℕ), d ∣ 2014000000 ∧
  (∀ (x : ℕ), x ∣ 2014000000 → x ≠ 2014000000 → x ≠ 1007000000 → x ≠ 503500000 → x ≠ 251750000 → x ≤ d) ∧
  d = 125875000 :=
by sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_of_2014000000_l3332_333282


namespace NUMINAMATH_CALUDE_annual_output_scientific_notation_l3332_333286

/-- The annual output of the photovoltaic power station in kWh -/
def annual_output : ℝ := 448000

/-- The scientific notation representation of the annual output -/
def scientific_notation : ℝ := 4.48 * (10 ^ 5)

theorem annual_output_scientific_notation : annual_output = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_annual_output_scientific_notation_l3332_333286


namespace NUMINAMATH_CALUDE_total_notebooks_bought_l3332_333294

/-- Represents the number of notebooks in a large pack -/
def large_pack_size : ℕ := 7

/-- Represents the number of large packs Wilson bought -/
def large_packs_bought : ℕ := 7

/-- Theorem stating that the total number of notebooks Wilson bought is 49 -/
theorem total_notebooks_bought : large_pack_size * large_packs_bought = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_notebooks_bought_l3332_333294


namespace NUMINAMATH_CALUDE_exponent_division_l3332_333213

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3332_333213


namespace NUMINAMATH_CALUDE_ice_volume_problem_l3332_333260

theorem ice_volume_problem (V : ℝ) : 
  (V * (1/4) * (1/4) = 0.4) → V = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_ice_volume_problem_l3332_333260


namespace NUMINAMATH_CALUDE_percentage_increase_l3332_333215

theorem percentage_increase (x : ℝ) (h : x = 77.7) : 
  (x - 70) / 70 * 100 = 11 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3332_333215


namespace NUMINAMATH_CALUDE_correct_ways_select_four_correct_ways_select_five_l3332_333258

/-- Number of distinct red balls -/
def num_red_balls : ℕ := 4

/-- Number of distinct white balls -/
def num_white_balls : ℕ := 7

/-- Score for selecting a red ball -/
def red_score : ℕ := 2

/-- Score for selecting a white ball -/
def white_score : ℕ := 1

/-- The number of ways to select 4 balls such that the number of red balls
    is not less than the number of white balls -/
def ways_select_four : ℕ := 115

/-- The number of ways to select 5 balls such that the total score
    is at least 7 points -/
def ways_select_five : ℕ := 301

/-- Theorem stating the correct number of ways to select 4 balls -/
theorem correct_ways_select_four :
  ways_select_four = Nat.choose num_red_balls 4 +
    Nat.choose num_red_balls 3 * Nat.choose num_white_balls 1 +
    Nat.choose num_red_balls 2 * Nat.choose num_white_balls 2 := by sorry

/-- Theorem stating the correct number of ways to select 5 balls -/
theorem correct_ways_select_five :
  ways_select_five = Nat.choose num_red_balls 2 * Nat.choose num_white_balls 3 +
    Nat.choose num_red_balls 3 * Nat.choose num_white_balls 2 +
    Nat.choose num_red_balls 4 * Nat.choose num_white_balls 1 := by sorry

end NUMINAMATH_CALUDE_correct_ways_select_four_correct_ways_select_five_l3332_333258


namespace NUMINAMATH_CALUDE_power_of_three_divides_a_l3332_333299

def a : ℕ → ℤ
  | 0 => 3
  | n + 1 => (3 * a n ^ 2 + 1) / 2 - a n

theorem power_of_three_divides_a (k : ℕ) : 
  (3 ^ (k + 1) : ℤ) ∣ a (3 ^ k) := by sorry

end NUMINAMATH_CALUDE_power_of_three_divides_a_l3332_333299


namespace NUMINAMATH_CALUDE_rhombus_equations_l3332_333269

/-- A rhombus with given properties -/
structure Rhombus where
  /-- Point A of the rhombus -/
  A : ℝ × ℝ
  /-- Point C of the rhombus -/
  C : ℝ × ℝ
  /-- Point P on the line BC -/
  P : ℝ × ℝ
  /-- Assertion that ABCD is a rhombus -/
  is_rhombus : A = (-4, 7) ∧ C = (2, -3) ∧ P = (3, -1)

/-- The equation of line AD in a rhombus -/
def line_AD (r : Rhombus) : ℝ → ℝ → Prop :=
  fun x y => 2 * x - y + 15 = 0

/-- The equation of diagonal BD in a rhombus -/
def diagonal_BD (r : Rhombus) : ℝ → ℝ → Prop :=
  fun x y => 3 * x - 5 * y + 13 = 0

/-- Main theorem about the equations of line AD and diagonal BD in the given rhombus -/
theorem rhombus_equations (r : Rhombus) :
  (∀ x y, line_AD r x y ↔ y = 2 * x + 15) ∧
  (∀ x y, diagonal_BD r x y ↔ y = (3 * x + 13) / 5) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_equations_l3332_333269


namespace NUMINAMATH_CALUDE_celine_erasers_l3332_333242

/-- The number of erasers collected by each person -/
structure EraserCollection where
  gabriel : ℕ
  celine : ℕ
  julian : ℕ
  erica : ℕ

/-- The conditions of the eraser collection problem -/
def EraserProblem (ec : EraserCollection) : Prop :=
  ec.celine = 2 * ec.gabriel ∧
  ec.julian = 2 * ec.celine ∧
  ec.erica = 3 * ec.julian ∧
  ec.gabriel + ec.celine + ec.julian + ec.erica = 151

theorem celine_erasers (ec : EraserCollection) (h : EraserProblem ec) : ec.celine = 16 := by
  sorry

end NUMINAMATH_CALUDE_celine_erasers_l3332_333242


namespace NUMINAMATH_CALUDE_chewing_gum_cost_l3332_333277

/-- Proves that the cost of each pack of chewing gum is $1, given the initial amount,
    purchases, and remaining amount. -/
theorem chewing_gum_cost
  (initial_amount : ℝ)
  (num_gum_packs : ℕ)
  (num_chocolate_bars : ℕ)
  (chocolate_bar_price : ℝ)
  (num_candy_canes : ℕ)
  (candy_cane_price : ℝ)
  (remaining_amount : ℝ)
  (h1 : initial_amount = 10)
  (h2 : num_gum_packs = 3)
  (h3 : num_chocolate_bars = 5)
  (h4 : chocolate_bar_price = 1)
  (h5 : num_candy_canes = 2)
  (h6 : candy_cane_price = 0.5)
  (h7 : remaining_amount = 1) :
  (initial_amount - remaining_amount
    - (num_chocolate_bars * chocolate_bar_price + num_candy_canes * candy_cane_price))
  / num_gum_packs = 1 := by
sorry


end NUMINAMATH_CALUDE_chewing_gum_cost_l3332_333277


namespace NUMINAMATH_CALUDE_train_length_calculation_l3332_333289

theorem train_length_calculation (crossing_time : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) :
  crossing_time = 25.997920166386688 →
  bridge_length = 160 →
  train_speed_kmph = 36 →
  let train_speed_mps := train_speed_kmph * (5/18)
  let total_distance := train_speed_mps * crossing_time
  let train_length := total_distance - bridge_length
  train_length = 99.97920166386688 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3332_333289


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3332_333220

/-- Given an ellipse with the following properties:
    1. The chord passing through the focus and perpendicular to the major axis has a length of √2
    2. The distance from the focus to the corresponding directrix is 1
    This theorem states that the eccentricity of the ellipse is √2/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * b^2 / a = Real.sqrt 2) (h4 : a^2 / c - c = 1) : 
  c / a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3332_333220


namespace NUMINAMATH_CALUDE_teacher_instructions_l3332_333245

theorem teacher_instructions (x : ℤ) : 4 * (3 * (x + 3) - 2) = 4 * (3 * x + 7) := by
  sorry

end NUMINAMATH_CALUDE_teacher_instructions_l3332_333245


namespace NUMINAMATH_CALUDE_min_inequality_solution_l3332_333239

theorem min_inequality_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z ≤ min (4 * (x - 1 / y)) (min (4 * (y - 1 / z)) (4 * (z - 1 / x)))) :
  x = Real.sqrt 2 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_inequality_solution_l3332_333239


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l3332_333268

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l3332_333268


namespace NUMINAMATH_CALUDE_arithmetic_sequences_equal_sum_l3332_333226

/-- Sum of first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ d n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequences_equal_sum :
  ∃! (n : ℕ), n > 0 ∧ arithmetic_sum 5 5 n = arithmetic_sum 22 3 n :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_equal_sum_l3332_333226


namespace NUMINAMATH_CALUDE_bank_coins_l3332_333266

theorem bank_coins (total_coins dimes quarters : ℕ) (h1 : total_coins = 11) (h2 : dimes = 2) (h3 : quarters = 7) :
  ∃ nickels : ℕ, nickels = total_coins - dimes - quarters :=
by
  sorry

end NUMINAMATH_CALUDE_bank_coins_l3332_333266


namespace NUMINAMATH_CALUDE_angle_C_is_60_degrees_area_is_10_sqrt_3_l3332_333240

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  (t.a - t.c) * (Real.sin t.A + Real.sin t.C) = (t.a - t.b) * Real.sin t.B

-- Theorem 1: Measure of angle C
theorem angle_C_is_60_degrees (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t) : 
  t.C = Real.pi / 3 :=
sorry

-- Theorem 2: Area of triangle when a = 5 and c = 7
theorem area_is_10_sqrt_3 (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t)
  (h3 : t.a = 5)
  (h4 : t.c = 7) : 
  (1/2) * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_angle_C_is_60_degrees_area_is_10_sqrt_3_l3332_333240


namespace NUMINAMATH_CALUDE_article_cost_calculation_l3332_333293

/-- Proves that if an article is sold for $25 with a 25% gain, it was bought for $20. -/
theorem article_cost_calculation (selling_price : ℝ) (gain_percent : ℝ) : 
  selling_price = 25 → gain_percent = 25 → 
  ∃ (cost_price : ℝ), cost_price = 20 ∧ selling_price = cost_price * (1 + gain_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_article_cost_calculation_l3332_333293


namespace NUMINAMATH_CALUDE_circle_center_l3332_333247

/-- The center of the circle defined by x^2 + y^2 - 4x - 2y - 5 = 0 is (2, 1) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 - 4*x - 2*y - 5 = 0) → (∃ r : ℝ, (x - 2)^2 + (y - 1)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l3332_333247


namespace NUMINAMATH_CALUDE_gummy_bear_distribution_l3332_333223

theorem gummy_bear_distribution (initial_candies : ℕ) (num_siblings : ℕ) (josh_eat : ℕ) (leftover : ℕ) :
  initial_candies = 100 →
  num_siblings = 3 →
  josh_eat = 16 →
  leftover = 19 →
  ∃ (sibling_candies : ℕ),
    sibling_candies * num_siblings + 2 * (josh_eat + leftover) = initial_candies ∧
    sibling_candies = 10 :=
by sorry

end NUMINAMATH_CALUDE_gummy_bear_distribution_l3332_333223


namespace NUMINAMATH_CALUDE_tom_dance_frequency_l3332_333217

/-- Represents the number of times Tom dances per week -/
def dance_frequency (hours_per_session : ℕ) (years : ℕ) (total_hours : ℕ) (weeks_per_year : ℕ) : ℕ :=
  (total_hours / (years * weeks_per_year)) / hours_per_session

/-- Proves that Tom dances 4 times a week given the conditions -/
theorem tom_dance_frequency :
  dance_frequency 2 10 4160 52 = 4 := by
sorry

end NUMINAMATH_CALUDE_tom_dance_frequency_l3332_333217


namespace NUMINAMATH_CALUDE_sum_of_four_powers_of_eight_l3332_333249

theorem sum_of_four_powers_of_eight :
  (8 : ℝ)^5 + (8 : ℝ)^5 + (8 : ℝ)^5 + (8 : ℝ)^5 = (8 : ℝ)^(17/3) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_powers_of_eight_l3332_333249


namespace NUMINAMATH_CALUDE_solution_2015_squared_l3332_333292

theorem solution_2015_squared : 
  ∃ x : ℚ, (2015 + x)^2 = x^2 ∧ x = -2015/2 := by
sorry

end NUMINAMATH_CALUDE_solution_2015_squared_l3332_333292


namespace NUMINAMATH_CALUDE_total_molecular_weight_theorem_l3332_333218

/-- Calculates the total molecular weight of given compounds -/
def totalMolecularWeight (Al_weight S_weight H_weight O_weight C_weight : ℝ) : ℝ :=
  let Al2S3_weight := 2 * Al_weight + 3 * S_weight
  let H2O_weight := 2 * H_weight + O_weight
  let CO2_weight := C_weight + 2 * O_weight
  7 * Al2S3_weight + 5 * H2O_weight + 4 * CO2_weight

/-- The total molecular weight of 7 moles of Al2S3, 5 moles of H2O, and 4 moles of CO2 is 1317.12 grams -/
theorem total_molecular_weight_theorem :
  totalMolecularWeight 26.98 32.06 1.01 16.00 12.01 = 1317.12 := by
  sorry

end NUMINAMATH_CALUDE_total_molecular_weight_theorem_l3332_333218


namespace NUMINAMATH_CALUDE_categorize_numbers_l3332_333248

def given_numbers : Set ℝ := {7/3, 1, 0, -1.4, Real.pi/2, 0.1010010001, -9}

def is_positive (x : ℝ) : Prop := x > 0

def is_fraction (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

theorem categorize_numbers :
  let positive_numbers : Set ℝ := {7/3, 1, Real.pi/2, 0.1010010001}
  let fraction_numbers : Set ℝ := {7/3, -1.4, 0.1010010001}
  (∀ x ∈ given_numbers, is_positive x ↔ x ∈ positive_numbers) ∧
  (∀ x ∈ given_numbers, is_fraction x ↔ x ∈ fraction_numbers) := by
  sorry

end NUMINAMATH_CALUDE_categorize_numbers_l3332_333248


namespace NUMINAMATH_CALUDE_weight_of_a_l3332_333229

theorem weight_of_a (a b c d : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  ∃ e : ℝ, e = d + 5 ∧ (b + c + d + e) / 4 = 79 →
  a = 77 := by
sorry

end NUMINAMATH_CALUDE_weight_of_a_l3332_333229


namespace NUMINAMATH_CALUDE_original_number_of_people_l3332_333228

theorem original_number_of_people (x : ℕ) : 
  (3 * x / 4 : ℚ) - (3 * x / 20 : ℚ) = 16 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_original_number_of_people_l3332_333228


namespace NUMINAMATH_CALUDE_distance_after_5_hours_l3332_333232

/-- The distance between two people after walking in opposite directions for a given time -/
def distance_between (speed1 speed2 time : ℝ) : ℝ :=
  (speed1 * time) + (speed2 * time)

/-- Theorem: The distance between two people walking in opposite directions for 5 hours,
    with speeds of 5 km/hr and 10 km/hr respectively, is 75 km -/
theorem distance_after_5_hours :
  distance_between 5 10 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_5_hours_l3332_333232


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_neg_150_l3332_333297

theorem largest_multiple_of_15_less_than_neg_150 :
  ∀ n : ℤ, n * 15 < -150 → n * 15 ≤ -165 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_neg_150_l3332_333297


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_31_over_6_l3332_333214

theorem greatest_integer_less_than_negative_31_over_6 :
  ⌊-31/6⌋ = -6 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_31_over_6_l3332_333214


namespace NUMINAMATH_CALUDE_perimeter_of_modified_square_l3332_333230

/-- The perimeter of a figure ABFCDE formed by cutting a right triangle from a square and translating it -/
theorem perimeter_of_modified_square (side_length : ℝ) (triangle_leg : ℝ) 
  (h1 : side_length = 20)
  (h2 : triangle_leg = 12) : 
  let hypotenuse := Real.sqrt (2 * triangle_leg ^ 2)
  let perimeter := 2 * side_length + (side_length - triangle_leg) + hypotenuse + 2 * triangle_leg
  perimeter = 72 + 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_modified_square_l3332_333230


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3332_333222

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3332_333222


namespace NUMINAMATH_CALUDE_double_price_increase_rate_l3332_333208

/-- The rate of price increase that, when applied twice, doubles the original price -/
theorem double_price_increase_rate : 
  ∃ x : ℝ, (1 + x) * (1 + x) = 2 ∧ x > 0 :=
by sorry

end NUMINAMATH_CALUDE_double_price_increase_rate_l3332_333208


namespace NUMINAMATH_CALUDE_brett_red_marbles_l3332_333212

/-- The number of red marbles Brett has -/
def red_marbles : ℕ := sorry

/-- The number of blue marbles Brett has -/
def blue_marbles : ℕ := sorry

/-- Brett has 24 more blue marbles than red marbles -/
axiom more_blue : blue_marbles = red_marbles + 24

/-- Brett has 5 times as many blue marbles as red marbles -/
axiom five_times : blue_marbles = 5 * red_marbles

theorem brett_red_marbles : red_marbles = 6 := by sorry

end NUMINAMATH_CALUDE_brett_red_marbles_l3332_333212


namespace NUMINAMATH_CALUDE_square_area_ratio_l3332_333274

/-- If the perimeter of one square is 4 times the perimeter of another square,
    then the area of the larger square is 16 times the area of the smaller square. -/
theorem square_area_ratio (s L : ℝ) (hs : s > 0) (hL : L > 0) 
    (h_perimeter : 4 * L = 4 * (4 * s)) : L^2 = 16 * s^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3332_333274


namespace NUMINAMATH_CALUDE_log_base_conversion_l3332_333201

theorem log_base_conversion (a : ℝ) (h : Real.log 16 / Real.log 14 = a) :
  Real.log 14 / Real.log 8 = 4 / (3 * a) := by
  sorry

end NUMINAMATH_CALUDE_log_base_conversion_l3332_333201


namespace NUMINAMATH_CALUDE_f_properties_l3332_333224

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6))) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), 1 ≤ f x ∧ f x ≤ 2) ∧
  (f 0 = 1) ∧
  (f (Real.pi / 6) = 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3332_333224


namespace NUMINAMATH_CALUDE_x_y_inequality_l3332_333291

theorem x_y_inequality (x y : ℝ) 
  (h1 : x < 1) 
  (h2 : 1 < y) 
  (h3 : 2 * Real.log x + Real.log (1 - x) ≥ 3 * Real.log y + Real.log (y - 1)) :
  x^3 + y^3 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_x_y_inequality_l3332_333291


namespace NUMINAMATH_CALUDE_manufacturing_quality_probability_l3332_333209

theorem manufacturing_quality_probability 
  (defect_rate1 : ℝ) 
  (defect_rate2 : ℝ) 
  (h1 : defect_rate1 = 0.03) 
  (h2 : defect_rate2 = 0.05) 
  (independent : True) -- Representing the independence of processes
  : (1 - defect_rate1) * (1 - defect_rate2) = 0.9215 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_quality_probability_l3332_333209


namespace NUMINAMATH_CALUDE_cos_graph_transformation_l3332_333283

theorem cos_graph_transformation (x : ℝ) : 
  let f (x : ℝ) := Real.cos ((1/2 : ℝ) * x - π/6)
  let g (x : ℝ) := f (x + π/3)
  let h (x : ℝ) := g (2 * x)
  h x = Real.cos x := by sorry

end NUMINAMATH_CALUDE_cos_graph_transformation_l3332_333283


namespace NUMINAMATH_CALUDE_least_multiple_remainder_l3332_333298

theorem least_multiple_remainder (m : ℕ) : 
  (m % 23 = 0) → 
  (m % 1821 = 710) → 
  (m = 3024) → 
  (m % 24 = 0) := by
sorry

end NUMINAMATH_CALUDE_least_multiple_remainder_l3332_333298


namespace NUMINAMATH_CALUDE_equal_selection_probability_l3332_333256

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents the probability of an individual being selected in a sampling method -/
def selectionProbability (method : SamplingMethod) (individual : ℕ) : ℝ := sorry

/-- The theorem stating that all three sampling methods have equal selection probability for all individuals -/
theorem equal_selection_probability (population : Finset ℕ) :
  ∀ (method : SamplingMethod) (i j : ℕ), i ∈ population → j ∈ population →
    selectionProbability method i = selectionProbability method j :=
  sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l3332_333256


namespace NUMINAMATH_CALUDE_certain_number_proof_l3332_333204

theorem certain_number_proof (p q : ℝ) (h1 : 3 / q = 18) (h2 : p - q = 7/12) : 3 / p = 4 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3332_333204


namespace NUMINAMATH_CALUDE_leahs_coin_value_l3332_333290

/-- Represents the number and value of coins --/
structure CoinCollection where
  pennies : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in cents --/
def totalValue (coins : CoinCollection) : ℕ :=
  coins.pennies + 25 * coins.quarters

/-- Theorem stating the value of Leah's coin collection --/
theorem leahs_coin_value :
  ∀ (coins : CoinCollection),
    coins.pennies + coins.quarters = 15 →
    coins.pennies = 2 * (coins.quarters + 1) →
    totalValue coins = 110 := by
  sorry


end NUMINAMATH_CALUDE_leahs_coin_value_l3332_333290


namespace NUMINAMATH_CALUDE_value_of_expression_l3332_333227

theorem value_of_expression (x : ℝ) (h : x = 5) : 2 * x^2 + 3 = 53 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3332_333227


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3332_333203

/-- Given a train crossing a bridge and a lamp post, calculate the bridge length -/
theorem bridge_length_calculation (train_length : ℝ) (bridge_crossing_time : ℝ) (lamp_post_crossing_time : ℝ) :
  train_length = 833.33 →
  bridge_crossing_time = 120 →
  lamp_post_crossing_time = 30 →
  ∃ bridge_length : ℝ, bridge_length = 2500 := by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l3332_333203


namespace NUMINAMATH_CALUDE_gcd_problem_l3332_333263

theorem gcd_problem (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : 
  Nat.gcd X Y = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3332_333263


namespace NUMINAMATH_CALUDE_existence_of_point_S_l3332_333252

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- Checks if a triangle is parallel to a plane -/
def is_parallel_to_plane (t : Triangle) (p : Plane) : Prop := sorry

/-- Finds the intersection point of a line and a plane -/
def line_plane_intersection (p1 p2 : Point3D) (plane : Plane) : Point3D := sorry

/-- The main theorem -/
theorem existence_of_point_S (α : Plane) (ABC MNP : Triangle) 
  (h : ¬ is_parallel_to_plane ABC α) : 
  ∃ (S : Point3D), 
    let A' := line_plane_intersection S ABC.A α
    let B' := line_plane_intersection S ABC.B α
    let C' := line_plane_intersection S ABC.C α
    let A'B'C' : Triangle := ⟨A', B', C'⟩
    are_congruent A'B'C' MNP := by
  sorry

end NUMINAMATH_CALUDE_existence_of_point_S_l3332_333252


namespace NUMINAMATH_CALUDE_circular_track_circumference_l3332_333261

/-- The circumference of a circular track given two cyclists' speeds and meeting time -/
theorem circular_track_circumference 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (meeting_time : ℝ) 
  (h1 : speed1 = 7) 
  (h2 : speed2 = 8) 
  (h3 : meeting_time = 45) : 
  speed1 * meeting_time + speed2 * meeting_time = 675 := by
  sorry

end NUMINAMATH_CALUDE_circular_track_circumference_l3332_333261


namespace NUMINAMATH_CALUDE_factorization_problems_l3332_333262

theorem factorization_problems :
  (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = (2*x + 3*y) * (2*x - 3*y)) ∧
  (∀ a b : ℝ, -16 * a^2 + 25 * b^2 = (5*b + 4*a) * (5*b - 4*a)) ∧
  (∀ x y : ℝ, x^3 * y - x * y^3 = x * y * (x + y) * (x - y)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l3332_333262


namespace NUMINAMATH_CALUDE_blue_cards_count_l3332_333202

theorem blue_cards_count (red_cards : ℕ) (blue_prob : ℚ) (blue_cards : ℕ) : 
  red_cards = 8 →
  blue_prob = 6/10 →
  (blue_cards : ℚ) / (blue_cards + red_cards) = blue_prob →
  blue_cards = 12 := by
sorry

end NUMINAMATH_CALUDE_blue_cards_count_l3332_333202


namespace NUMINAMATH_CALUDE_factorization_equality_l3332_333238

theorem factorization_equality (x : ℝ) : 4 * x - x^2 - 4 = -(x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3332_333238


namespace NUMINAMATH_CALUDE_inequality_solution_existence_condition_l3332_333278

-- Define the functions f and g
def f (a x : ℝ) := |2 * x + a| - |2 * x + 3|
def g (x : ℝ) := |x - 1| - 3

-- Theorem for the first part of the problem
theorem inequality_solution (x : ℝ) :
  |g x| < 2 ↔ -4 < x ∧ x < 6 := by sorry

-- Theorem for the second part of the problem
theorem existence_condition (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) ↔ 0 ≤ a ∧ a ≤ 6 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_existence_condition_l3332_333278


namespace NUMINAMATH_CALUDE_train_speed_problem_l3332_333233

theorem train_speed_problem (length1 length2 speed1 time : ℝ) 
  (h1 : length1 = 500)
  (h2 : length2 = 750)
  (h3 : speed1 = 60)
  (h4 : time = 44.99640028797697) : 
  ∃ speed2 : ℝ, 
    speed2 = 40 ∧ 
    (length1 + length2) / 1000 = (speed1 + speed2) * (time / 3600) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3332_333233


namespace NUMINAMATH_CALUDE_max_distinct_sums_diffs_is_64_l3332_333219

/-- Given a set of five natural numbers including 100, 200, and 400,
    this function returns the maximum number of distinct non-zero natural numbers
    that can be obtained by performing addition and subtraction operations,
    where each number is used at most once in each expression
    and at least two numbers are used. -/
def max_distinct_sums_diffs (a b : ℕ) : ℕ :=
  64

/-- Theorem stating that the maximum number of distinct non-zero natural numbers
    obtainable from the given set of numbers under the specified conditions is 64. -/
theorem max_distinct_sums_diffs_is_64 (a b : ℕ) :
  max_distinct_sums_diffs a b = 64 := by
  sorry

end NUMINAMATH_CALUDE_max_distinct_sums_diffs_is_64_l3332_333219


namespace NUMINAMATH_CALUDE_rectangle_area_l3332_333237

/-- Given a rectangle with perimeter 120 cm and length twice the width, prove its area is 800 cm² -/
theorem rectangle_area (width : ℝ) (length : ℝ) : 
  (2 * (length + width) = 120) →  -- Perimeter condition
  (length = 2 * width) →          -- Length-width relationship
  (length * width = 800) :=       -- Area to prove
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3332_333237


namespace NUMINAMATH_CALUDE_carmen_dogs_l3332_333243

def problem (initial_cats : ℕ) (adopted_cats : ℕ) (cat_dog_difference : ℕ) : Prop :=
  let remaining_cats := initial_cats - adopted_cats
  ∃ (dogs : ℕ), remaining_cats = dogs + cat_dog_difference

theorem carmen_dogs : 
  problem 28 3 7 → ∃ (dogs : ℕ), dogs = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_carmen_dogs_l3332_333243
