import Mathlib

namespace NUMINAMATH_CALUDE_stick_pieces_l335_33525

def stick_length : ℕ := 60

def marks_10 : List ℕ := [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
def marks_12 : List ℕ := [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
def marks_15 : List ℕ := [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]

def all_marks : List ℕ := marks_10 ++ marks_12 ++ marks_15

theorem stick_pieces : 
  (all_marks.toFinset.card) + 1 = 28 := by sorry

end NUMINAMATH_CALUDE_stick_pieces_l335_33525


namespace NUMINAMATH_CALUDE_fraction_equality_l335_33559

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : (2 * a) / (2 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l335_33559


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l335_33579

theorem binomial_coefficient_equation_solution (x : ℕ) : 
  (Nat.choose 12 (x + 1) = Nat.choose 12 (2 * x - 1)) ↔ (x = 2 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l335_33579


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l335_33514

-- Define the function representing the curve
def f (x : ℝ) : ℝ := -2 * x^2 + 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -4 * x

-- Theorem statement
theorem tangent_slope_at_zero : f' 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l335_33514


namespace NUMINAMATH_CALUDE_remainder_of_7_pow_2023_mod_17_l335_33549

theorem remainder_of_7_pow_2023_mod_17 : 7^2023 % 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_7_pow_2023_mod_17_l335_33549


namespace NUMINAMATH_CALUDE_line_equation_midpoint_line_equation_max_distance_l335_33523

/-- A line passing through point M(1, 2) and intersecting the x-axis and y-axis -/
structure Line where
  m : ℝ × ℝ := (1, 2)
  intersects_x_axis : ℝ → Prop
  intersects_y_axis : ℝ → Prop

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : LineEquation) : ℝ := sorry

theorem line_equation_midpoint (l : Line) : 
  (∃ p q : ℝ × ℝ, l.intersects_x_axis p.1 ∧ l.intersects_y_axis q.2 ∧ 
   l.m = ((p.1 + q.1) / 2, (p.2 + q.2) / 2)) → 
  ∃ eq : LineEquation, eq.a = 2 ∧ eq.b = 1 ∧ eq.c = -4 :=
sorry

theorem line_equation_max_distance (l : Line) :
  (∀ eq : LineEquation, distance_point_to_line (0, 0) eq ≤ 
   distance_point_to_line (0, 0) ⟨1, 2, -5⟩) →
  ∃ eq : LineEquation, eq.a = 1 ∧ eq.b = 2 ∧ eq.c = -5 :=
sorry

end NUMINAMATH_CALUDE_line_equation_midpoint_line_equation_max_distance_l335_33523


namespace NUMINAMATH_CALUDE_average_weight_solution_l335_33560

def average_weight_problem (a b c : ℝ) : Prop :=
  let avg_abc := (a + b + c) / 3
  let avg_ab := (a + b) / 2
  (avg_abc = 45) ∧ (avg_ab = 40) ∧ (b = 31) → ((b + c) / 2 = 43)

theorem average_weight_solution :
  ∀ a b c : ℝ, average_weight_problem a b c :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_solution_l335_33560


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l335_33583

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = π) : -- Condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l335_33583


namespace NUMINAMATH_CALUDE_decimal_259_to_base5_l335_33587

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: decimal_to_base5 (n / 5)

theorem decimal_259_to_base5 :
  decimal_to_base5 259 = [4, 1, 0, 2] := by sorry

end NUMINAMATH_CALUDE_decimal_259_to_base5_l335_33587


namespace NUMINAMATH_CALUDE_student_sums_proof_l335_33566

def total_sums (right_sums wrong_sums : ℕ) : ℕ :=
  right_sums + wrong_sums

theorem student_sums_proof (right_sums : ℕ) 
  (h1 : right_sums = 18) 
  (h2 : ∃ wrong_sums : ℕ, wrong_sums = 2 * right_sums) : 
  total_sums right_sums (2 * right_sums) = 54 := by
  sorry

end NUMINAMATH_CALUDE_student_sums_proof_l335_33566


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l335_33545

-- Define the sum of interior angles of a polygon
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the given sum of angles
def given_sum : ℝ := 2340

-- Theorem statement
theorem convex_polygon_sides : 
  ∃ (n : ℕ), n > 2 ∧ 
  sum_interior_angles n - given_sum > 0 ∧ 
  sum_interior_angles n - given_sum ≤ 360 ∧
  n = 16 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l335_33545


namespace NUMINAMATH_CALUDE_star_perimeter_sum_l335_33524

theorem star_perimeter_sum (X Y Z : ℕ) : 
  Prime X → Prime Y → Prime Z →
  X < Z → Z < Y → X + Y < 2 * Z →
  X + Y + Z ≥ 20 := by
sorry

end NUMINAMATH_CALUDE_star_perimeter_sum_l335_33524


namespace NUMINAMATH_CALUDE_quadratic_square_completion_l335_33564

theorem quadratic_square_completion (d e : ℤ) : 
  (∀ x, x^2 - 10*x + 13 = 0 ↔ (x + d)^2 = e) → d + e = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_square_completion_l335_33564


namespace NUMINAMATH_CALUDE_gcd_180_294_l335_33502

theorem gcd_180_294 : Nat.gcd 180 294 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_294_l335_33502


namespace NUMINAMATH_CALUDE_gcd_lcm_2970_1722_l335_33529

theorem gcd_lcm_2970_1722 : 
  (Nat.gcd 2970 1722 = 6) ∧ (Nat.lcm 2970 1722 = 856170) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_2970_1722_l335_33529


namespace NUMINAMATH_CALUDE_quadratic_roots_l335_33582

/-- Given a quadratic function f(x) = ax^2 + bx with specific values, 
    prove that the roots of f(x) = 6 are -2 and 3. -/
theorem quadratic_roots (a b : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x^2 + b * x)
    (h_m2 : f (-2) = 6)
    (h_m1 : f (-1) = 2)
    (h_0  : f 0 = 0)
    (h_1  : f 1 = 0)
    (h_2  : f 2 = 2)
    (h_3  : f 3 = 6) :
  (∃ x, f x = 6) ∧ (f (-2) = 6 ∧ f 3 = 6) ∧ 
  (∀ x, f x = 6 → x = -2 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l335_33582


namespace NUMINAMATH_CALUDE_negation_of_proposition_l335_33527

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l335_33527


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l335_33586

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter : ℝ := outer_cube_edge
  let inner_cube_diagonal : ℝ := sphere_diameter
  let inner_cube_edge : ℝ := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume : ℝ := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l335_33586


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l335_33513

def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem set_intersection_theorem :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l335_33513


namespace NUMINAMATH_CALUDE_x_plus_q_in_terms_of_q_l335_33595

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x + 3| = q) (h2 : x > -3) : x + q = 2*q - 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_q_in_terms_of_q_l335_33595


namespace NUMINAMATH_CALUDE_right_triangle_30_60_90_properties_l335_33511

/-- A right triangle with one leg of 15 inches and the angle opposite that leg being 30° --/
structure RightTriangle30_60_90 where
  /-- The length of one leg of the triangle --/
  leg : ℝ
  /-- The angle opposite the given leg in radians --/
  angle : ℝ
  /-- The triangle is a right triangle --/
  is_right_triangle : leg > 0
  /-- The length of the given leg is 15 inches --/
  leg_length : leg = 15
  /-- The angle opposite the given leg is 30° (π/6 radians) --/
  angle_measure : angle = π / 6

/-- The length of the hypotenuse in the given right triangle --/
def hypotenuse_length (t : RightTriangle30_60_90) : ℝ := 30

/-- The length of the altitude from the hypotenuse to the right angle in the given triangle --/
def altitude_length (t : RightTriangle30_60_90) : ℝ := 22.5

/-- Theorem stating the length of the hypotenuse and altitude in the given right triangle --/
theorem right_triangle_30_60_90_properties (t : RightTriangle30_60_90) :
  hypotenuse_length t = 30 ∧ altitude_length t = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_30_60_90_properties_l335_33511


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_one_l335_33536

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b ∧ l1.a * l2.c ≠ l2.a * l1.c

/-- The first line: x + (1+m)y = 2-m -/
def line1 (m : ℝ) : Line :=
  { a := 1, b := 1 + m, c := m - 2 }

/-- The second line: 2mx + 4y = -16 -/
def line2 (m : ℝ) : Line :=
  { a := 2 * m, b := 4, c := 16 }

/-- The theorem stating that the lines are parallel iff m = 1 -/
theorem lines_parallel_iff_m_eq_one :
  ∀ m : ℝ, parallel (line1 m) (line2 m) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_one_l335_33536


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l335_33530

theorem sqrt_equation_solution : ∃ x : ℝ, x = 2209 / 64 ∧ Real.sqrt x + Real.sqrt (x + 3) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l335_33530


namespace NUMINAMATH_CALUDE_ellipse_range_l335_33590

-- Define the set of real numbers m for which the equation represents an ellipse
def ellipse_set (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) - y^2 / (m + 1) = 1 ∧ 
  (m + 2 > 0) ∧ (-m - 1 > 0) ∧ (m + 2 ≠ -m - 1)

-- Define the target range for m
def target_range (m : ℝ) : Prop :=
  (m > -2 ∧ m < -3/2) ∨ (m > -3/2 ∧ m < -1)

-- Theorem statement
theorem ellipse_range :
  ∀ m : ℝ, ellipse_set m ↔ target_range m :=
sorry

end NUMINAMATH_CALUDE_ellipse_range_l335_33590


namespace NUMINAMATH_CALUDE_absolute_value_of_complex_product_l335_33517

open Complex

theorem absolute_value_of_complex_product : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + i) * (1 + 3*i)
  Complex.abs z = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_complex_product_l335_33517


namespace NUMINAMATH_CALUDE_jack_book_pages_l335_33520

/-- Calculates the total number of pages in a book given the daily reading rate and the number of days to finish. -/
def total_pages (pages_per_day : ℕ) (days_to_finish : ℕ) : ℕ :=
  pages_per_day * days_to_finish

/-- Proves that the book Jack is reading has 299 pages. -/
theorem jack_book_pages :
  let pages_per_day : ℕ := 23
  let days_to_finish : ℕ := 13
  total_pages pages_per_day days_to_finish = 299 := by
  sorry

end NUMINAMATH_CALUDE_jack_book_pages_l335_33520


namespace NUMINAMATH_CALUDE_angle_abc_measure_l335_33593

theorem angle_abc_measure (θ : ℝ) : 
  θ > 0 ∧ θ < 180 → -- Angle measure is positive and less than 180°
  (θ / 2) = (1 / 3) * (180 - θ) → -- Condition about angle bisector
  θ = 72 := by
sorry

end NUMINAMATH_CALUDE_angle_abc_measure_l335_33593


namespace NUMINAMATH_CALUDE_bathing_suit_sets_proof_l335_33568

-- Define the constants from the problem
def total_time : ℕ := 60
def runway_time : ℕ := 2
def num_models : ℕ := 6
def evening_wear_sets : ℕ := 3

-- Define the function to calculate bathing suit sets per model
def bathing_suit_sets_per_model : ℕ :=
  let total_evening_wear_time := num_models * evening_wear_sets * runway_time
  let remaining_time := total_time - total_evening_wear_time
  let total_bathing_suit_trips := remaining_time / runway_time
  total_bathing_suit_trips / num_models

-- Theorem statement
theorem bathing_suit_sets_proof :
  bathing_suit_sets_per_model = 2 :=
by sorry

end NUMINAMATH_CALUDE_bathing_suit_sets_proof_l335_33568


namespace NUMINAMATH_CALUDE_prob_A_not_lose_l335_33561

-- Define the probabilities
def prob_A_win : ℝ := 0.3
def prob_draw : ℝ := 0.5

-- Define the property of mutually exclusive events
def mutually_exclusive (p q : ℝ) : Prop := p + q ≤ 1

-- State the theorem
theorem prob_A_not_lose : 
  mutually_exclusive prob_A_win prob_draw →
  prob_A_win + prob_draw = 0.8 :=
by
  sorry

end NUMINAMATH_CALUDE_prob_A_not_lose_l335_33561


namespace NUMINAMATH_CALUDE_max_phi_difference_l335_33594

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem max_phi_difference (n : ℕ) (h : 1 ≤ n ∧ n ≤ 100) :
  (phi (n^2 + 2*n) - phi (n^2) ≤ 72) ∧
  (∃ m : ℕ, 1 ≤ m ∧ m ≤ 100 ∧ phi (m^2 + 2*m) - phi (m^2) = 72) :=
sorry

end NUMINAMATH_CALUDE_max_phi_difference_l335_33594


namespace NUMINAMATH_CALUDE_exists_counterexample_l335_33580

-- Define the types for cards
inductive Letter : Type
| S | T | U

inductive Number : Type
| Two | Five | Seven | Eleven

-- Define a card as a pair of a Letter and a Number
def Card : Type := Letter × Number

-- Define the set of cards
def cards : List Card := [
  (Letter.S, Number.Two),
  (Letter.S, Number.Five),
  (Letter.S, Number.Seven),
  (Letter.S, Number.Eleven),
  (Letter.T, Number.Two),
  (Letter.T, Number.Five),
  (Letter.T, Number.Seven),
  (Letter.T, Number.Eleven),
  (Letter.U, Number.Two),
  (Letter.U, Number.Five),
  (Letter.U, Number.Seven),
  (Letter.U, Number.Eleven)
]

-- Define what a consonant is
def isConsonant (l : Letter) : Bool :=
  match l with
  | Letter.S => true
  | Letter.T => true
  | Letter.U => false

-- Define what a prime number is
def isPrime (n : Number) : Bool :=
  match n with
  | Number.Two => true
  | Number.Five => true
  | Number.Seven => true
  | Number.Eleven => true

-- Sam's statement
def samsStatement (c : Card) : Bool :=
  ¬(isConsonant c.1) ∨ isPrime c.2

-- Theorem to prove
theorem exists_counterexample :
  ∃ c ∈ cards, ¬(samsStatement c) :=
sorry

end NUMINAMATH_CALUDE_exists_counterexample_l335_33580


namespace NUMINAMATH_CALUDE_technician_permanent_percentage_l335_33544

/-- Represents the composition of workers in a factory -/
structure Factory where
  total_workers : ℕ
  technicians : ℕ
  non_technicians : ℕ
  permanent_non_technicians : ℕ
  temporary_workers : ℕ

/-- The conditions of the factory -/
def factory_conditions (f : Factory) : Prop :=
  f.technicians = f.total_workers / 2 ∧
  f.non_technicians = f.total_workers / 2 ∧
  f.permanent_non_technicians = f.non_technicians / 2 ∧
  f.temporary_workers = f.total_workers / 2

/-- The theorem to be proved -/
theorem technician_permanent_percentage (f : Factory) 
  (h : factory_conditions f) : 
  (f.technicians - (f.temporary_workers - f.permanent_non_technicians)) * 2 = f.technicians := by
  sorry

#check technician_permanent_percentage

end NUMINAMATH_CALUDE_technician_permanent_percentage_l335_33544


namespace NUMINAMATH_CALUDE_kevin_bought_three_muffins_l335_33532

/-- The number of muffins Kevin bought -/
def num_muffins : ℕ := 3

/-- The cost of juice in dollars -/
def juice_cost : ℚ := 145/100

/-- The total amount paid in dollars -/
def total_paid : ℚ := 370/100

/-- The cost of each muffin in dollars -/
def muffin_cost : ℚ := 75/100

/-- Theorem stating that the number of muffins Kevin bought is 3 -/
theorem kevin_bought_three_muffins :
  num_muffins = 3 ∧
  juice_cost + (num_muffins : ℚ) * muffin_cost = total_paid :=
sorry

end NUMINAMATH_CALUDE_kevin_bought_three_muffins_l335_33532


namespace NUMINAMATH_CALUDE_boats_in_lake_l335_33569

theorem boats_in_lake (people_per_boat : ℕ) (total_people : ℕ) (number_of_boats : ℕ) : 
  people_per_boat = 3 → total_people = 15 → number_of_boats * people_per_boat = total_people → 
  number_of_boats = 5 := by
  sorry

end NUMINAMATH_CALUDE_boats_in_lake_l335_33569


namespace NUMINAMATH_CALUDE_halfway_fraction_l335_33578

theorem halfway_fraction (a b : ℚ) (ha : a = 1/7) (hb : b = 1/4) :
  (a + b) / 2 = 11/56 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l335_33578


namespace NUMINAMATH_CALUDE_problem_solution_l335_33526

theorem problem_solution (a b : ℤ) : 
  (5 + a = 6 - b) → (6 + b = 9 + a) → (5 - a = 6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l335_33526


namespace NUMINAMATH_CALUDE_existence_of_non_divisible_pair_l335_33531

theorem existence_of_non_divisible_pair (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧
    ¬(p^2 ∣ a^(p-1) - 1) ∧ ¬(p^2 ∣ (a+1)^(p-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_divisible_pair_l335_33531


namespace NUMINAMATH_CALUDE_special_determinant_l335_33533

open Matrix

/-- The determinant of an n×n matrix with diagonal elements b and all other elements a
    is equal to [b+(n-1)a](b-a)^(n-1) -/
theorem special_determinant (n : ℕ) (a b : ℝ) :
  let M : Matrix (Fin n) (Fin n) ℝ := λ i j => if i = j then b else a
  det M = (b + (n - 1) * a) * (b - a) ^ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_special_determinant_l335_33533


namespace NUMINAMATH_CALUDE_quadratic_discriminant_problem_l335_33592

theorem quadratic_discriminant_problem (m : ℝ) : 
  ((-3)^2 - 4*1*(-m) = 13) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_problem_l335_33592


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_eight_equals_four_l335_33573

theorem sqrt_two_times_sqrt_eight_equals_four : Real.sqrt 2 * Real.sqrt 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_eight_equals_four_l335_33573


namespace NUMINAMATH_CALUDE_truck_rental_miles_l335_33558

theorem truck_rental_miles (rental_fee charge_per_mile total_paid : ℚ) : 
  rental_fee = 20.99 →
  charge_per_mile = 0.25 →
  total_paid = 95.74 →
  (total_paid - rental_fee) / charge_per_mile = 299 := by
  sorry

end NUMINAMATH_CALUDE_truck_rental_miles_l335_33558


namespace NUMINAMATH_CALUDE_circle_passes_through_origin_circle_passes_through_four_zero_circle_passes_through_neg_one_one_is_circle_equation_l335_33591

/-- A circle passing through the points (0,0), (4,0), and (-1,1) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

/-- The circle passes through the point (0,0) -/
theorem circle_passes_through_origin :
  circle_equation 0 0 := by sorry

/-- The circle passes through the point (4,0) -/
theorem circle_passes_through_four_zero :
  circle_equation 4 0 := by sorry

/-- The circle passes through the point (-1,1) -/
theorem circle_passes_through_neg_one_one :
  circle_equation (-1) 1 := by sorry

/-- The equation represents a circle -/
theorem is_circle_equation :
  ∃ (h k r : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2 := by sorry

end NUMINAMATH_CALUDE_circle_passes_through_origin_circle_passes_through_four_zero_circle_passes_through_neg_one_one_is_circle_equation_l335_33591


namespace NUMINAMATH_CALUDE_function_identity_l335_33540

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : 
  ∀ n : ℕ+, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l335_33540


namespace NUMINAMATH_CALUDE_complex_division_result_l335_33599

theorem complex_division_result : Complex.I * 2 / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l335_33599


namespace NUMINAMATH_CALUDE_circle_existence_l335_33562

-- Define the lines and the given circle
def line1 (x y : ℝ) : Prop := x + y = 7
def line2 (x y : ℝ) : Prop := x - 7*y = -33
def given_circle (x y : ℝ) : Prop := x^2 + y^2 - 28*x + 6*y + 165 = 0

-- Define the distance ratio condition
def distance_ratio (x y u v : ℝ) : Prop :=
  |x + y - 7| / Real.sqrt 2 = 5 * |x - 7*y + 33| / Real.sqrt 50

-- Define the intersection point of the two lines
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define the orthogonality condition
def orthogonal_intersection (x y u v r : ℝ) : Prop :=
  (u - 14)^2 + (v + 3)^2 = r^2 + 40

-- Define the two resulting circles
def circle1 (x y : ℝ) : Prop := (x - 11)^2 + (y - 8)^2 = 87
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 168

theorem circle_existence :
  ∃ (x y u₁ v₁ u₂ v₂ : ℝ),
    (∀ (a b : ℝ), intersection_point a b → (circle1 a b ∨ circle2 a b)) ∧
    distance_ratio u₁ v₁ u₁ v₁ ∧
    distance_ratio u₂ v₂ u₂ v₂ ∧
    orthogonal_intersection u₁ v₁ u₁ v₁ (Real.sqrt 87) ∧
    orthogonal_intersection u₂ v₂ u₂ v₂ (Real.sqrt 168) :=
  sorry


end NUMINAMATH_CALUDE_circle_existence_l335_33562


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l335_33553

/-- The amount Mary spent on clothing -/
def total_spent : ℚ := 25.31

/-- The amount Mary spent on the shirt -/
def shirt_cost : ℚ := 13.04

/-- The number of shops Mary visited -/
def shops_visited : ℕ := 2

/-- The amount Mary spent on the jacket -/
def jacket_cost : ℚ := total_spent - shirt_cost

theorem jacket_cost_calculation : jacket_cost = 12.27 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l335_33553


namespace NUMINAMATH_CALUDE_zoo_visitors_l335_33508

/-- Proves that the number of adults who went to the zoo is 51, given the total number of people,
    ticket prices, and total sales. -/
theorem zoo_visitors (total_people : ℕ) (adult_price kid_price : ℕ) (total_sales : ℕ)
    (h_total : total_people = 254)
    (h_adult_price : adult_price = 28)
    (h_kid_price : kid_price = 12)
    (h_sales : total_sales = 3864) :
    ∃ (adults : ℕ), adults = 51 ∧
    ∃ (kids : ℕ), adults + kids = total_people ∧
    adult_price * adults + kid_price * kids = total_sales :=
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l335_33508


namespace NUMINAMATH_CALUDE_bug_position_after_2015_jumps_l335_33572

/-- Represents the five points on the circle -/
inductive Point
  | one
  | two
  | three
  | four
  | five

/-- Determines if a point is odd-numbered -/
def isOdd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true

/-- Performs one jump according to the rules -/
def jump (p : Point) : Point :=
  match p with
  | Point.one => Point.three
  | Point.two => Point.three
  | Point.three => Point.five
  | Point.four => Point.five
  | Point.five => Point.two

/-- Performs n jumps starting from a given point -/
def jumpNTimes (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => jump (jumpNTimes start n)

theorem bug_position_after_2015_jumps :
  jumpNTimes Point.five 2015 = Point.five :=
sorry

end NUMINAMATH_CALUDE_bug_position_after_2015_jumps_l335_33572


namespace NUMINAMATH_CALUDE_number_of_cows_bought_l335_33512

/-- Prove that the number of cows bought is 2 -/
theorem number_of_cows_bought (total_cost : ℕ) (num_goats : ℕ) (goat_price : ℕ) (cow_price : ℕ) :
  total_cost = 1400 →
  num_goats = 8 →
  goat_price = 60 →
  cow_price = 460 →
  (total_cost - num_goats * goat_price) / cow_price = 2 := by
  sorry

#check number_of_cows_bought

end NUMINAMATH_CALUDE_number_of_cows_bought_l335_33512


namespace NUMINAMATH_CALUDE_distance_traveled_l335_33588

theorem distance_traveled (initial_reading lunch_reading : ℝ) 
  (h1 : initial_reading = 212.3)
  (h2 : lunch_reading = 372.0) :
  lunch_reading - initial_reading = 159.7 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l335_33588


namespace NUMINAMATH_CALUDE_terminating_decimal_expansion_13_200_l335_33505

theorem terminating_decimal_expansion_13_200 : 
  ∃ (n : ℕ) (a : ℤ), (13 : ℚ) / 200 = (a : ℚ) / (10 ^ n) ∧ (a : ℚ) / (10 ^ n) = 0.052 :=
by
  sorry

end NUMINAMATH_CALUDE_terminating_decimal_expansion_13_200_l335_33505


namespace NUMINAMATH_CALUDE_circle_Q_equation_no_perpendicular_bisector_l335_33539

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 0)

-- Define line l₁ passing through P and intersecting circle C at M and N
def line_l₁ (x y : ℝ) : Prop := ∃ (t : ℝ), x = 2 + t ∧ y = t ∧ circle_C x y

-- Define the length of MN
def MN_length : ℝ := 4

-- Define line ax - y + 1 = 0
def line_AB (a x y : ℝ) : Prop := a*x - y + 1 = 0

-- Theorem 1: Equation of circle Q
theorem circle_Q_equation : 
  ∀ x y : ℝ, (∃ M N : ℝ × ℝ, line_l₁ M.1 M.2 ∧ line_l₁ N.1 N.2 ∧ 
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = MN_length^2) →
  ((x - 2)^2 + y^2 = 4) := 
sorry

-- Theorem 2: Non-existence of a
theorem no_perpendicular_bisector :
  ¬ ∃ a : ℝ, ∀ A B : ℝ × ℝ, 
    (line_AB a A.1 A.2 ∧ circle_C A.1 A.2 ∧ 
     line_AB a B.1 B.2 ∧ circle_C B.1 B.2 ∧ A ≠ B) →
    (∃ l₂ : ℝ → ℝ → Prop, 
      l₂ point_P.1 point_P.2 ∧
      l₂ ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧
      (B.2 - A.2) * (point_P.1 - A.1) = (point_P.2 - A.2) * (B.1 - A.1)) :=
sorry

end NUMINAMATH_CALUDE_circle_Q_equation_no_perpendicular_bisector_l335_33539


namespace NUMINAMATH_CALUDE_no_three_naturals_with_prime_sums_l335_33542

theorem no_three_naturals_with_prime_sums :
  ¬ ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.Prime (a + b) ∧ 
    Nat.Prime (a + c) ∧ 
    Nat.Prime (b + c) :=
sorry

end NUMINAMATH_CALUDE_no_three_naturals_with_prime_sums_l335_33542


namespace NUMINAMATH_CALUDE_factory_sampling_is_systematic_l335_33563

/-- Represents a sampling method -/
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

/-- Represents the characteristics of a sampling process -/
structure SamplingProcess where
  orderedArrangement : Bool
  fixedInterval : Bool

/-- Determines the sampling method based on the sampling process characteristics -/
def determineSamplingMethod (process : SamplingProcess) : SamplingMethod :=
  if process.orderedArrangement && process.fixedInterval then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom -- Default case, not actually used in this problem

/-- Theorem stating that the given sampling process is systematic sampling -/
theorem factory_sampling_is_systematic 
  (process : SamplingProcess)
  (h1 : process.orderedArrangement = true)
  (h2 : process.fixedInterval = true) :
  determineSamplingMethod process = SamplingMethod.Systematic := by
  sorry


end NUMINAMATH_CALUDE_factory_sampling_is_systematic_l335_33563


namespace NUMINAMATH_CALUDE_inequality_statements_l335_33584

theorem inequality_statements (a b c : ℝ) :
  (a > b ∧ b > 0 ∧ c < 0 → c / (a^2) > c / (b^2)) ∧
  (c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_statements_l335_33584


namespace NUMINAMATH_CALUDE_initial_temperature_l335_33551

theorem initial_temperature (T : ℝ) : (2 * T - 30) * 0.70 + 24 = 59 ↔ T = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_temperature_l335_33551


namespace NUMINAMATH_CALUDE_journey_to_the_west_readers_l335_33570

theorem journey_to_the_west_readers (total : ℕ) (either : ℕ) (dream : ℕ) (both : ℕ) 
  (h1 : total = 100)
  (h2 : either = 90)
  (h3 : dream = 80)
  (h4 : both = 60)
  (h5 : either ≤ total)
  (h6 : dream ≤ total)
  (h7 : both ≤ dream)
  (h8 : both ≤ either) : 
  ∃ (journey : ℕ), journey = 70 ∧ journey = either + both - dream := by
  sorry

end NUMINAMATH_CALUDE_journey_to_the_west_readers_l335_33570


namespace NUMINAMATH_CALUDE_boxes_sold_proof_l335_33555

/-- The number of boxes sold on Friday -/
def friday_boxes : ℕ := 40

/-- The number of boxes sold on Saturday -/
def saturday_boxes : ℕ := 2 * friday_boxes - 10

/-- The number of boxes sold on Sunday -/
def sunday_boxes : ℕ := (saturday_boxes) / 2

theorem boxes_sold_proof :
  friday_boxes + saturday_boxes + sunday_boxes = 145 :=
by sorry

end NUMINAMATH_CALUDE_boxes_sold_proof_l335_33555


namespace NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_or_q_true_l335_33503

theorem not_p_or_q_false_implies_p_or_q_true (p q : Prop) :
  ¬(¬(p ∨ q)) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_or_q_true_l335_33503


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l335_33546

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_plane_to_plane : Plane → Plane → Prop)
variable (perpendicular_plane_to_line : Plane → Line → Prop)

-- State the theorems
theorem planes_parallel_to_same_plane_are_parallel
  (p1 p2 p3 : Plane)
  (h1 : parallel_plane_to_plane p1 p3)
  (h2 : parallel_plane_to_plane p2 p3) :
  parallel_planes p1 p2 :=
sorry

theorem planes_perpendicular_to_same_line_are_parallel
  (p1 p2 : Plane) (l : Line)
  (h1 : perpendicular_plane_to_line p1 l)
  (h2 : perpendicular_plane_to_line p2 l) :
  parallel_planes p1 p2 :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l335_33546


namespace NUMINAMATH_CALUDE_find_v5_l335_33574

def sequence_relation (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 2) = 2 * v (n + 1) + v n

theorem find_v5 (v : ℕ → ℝ) (h1 : sequence_relation v) (h2 : v 4 = 15) (h3 : v 7 = 255) :
  v 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_find_v5_l335_33574


namespace NUMINAMATH_CALUDE_pizza_area_increase_l335_33552

theorem pizza_area_increase (r : ℝ) (hr : r > 0) :
  let medium_area := π * r^2
  let large_radius := 1.1 * r
  let large_area := π * large_radius^2
  (large_area - medium_area) / medium_area = 0.21 := by sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l335_33552


namespace NUMINAMATH_CALUDE_total_balls_l335_33537

def ball_count (red blue yellow : ℕ) : Prop :=
  red + blue + yellow > 0 ∧ 2 * blue = 3 * red ∧ 4 * red = 2 * yellow

theorem total_balls (red blue yellow : ℕ) :
  ball_count red blue yellow → yellow = 40 → red + blue + yellow = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_l335_33537


namespace NUMINAMATH_CALUDE_cubic_expression_equality_l335_33575

theorem cubic_expression_equality : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_equality_l335_33575


namespace NUMINAMATH_CALUDE_parallelepiped_volume_exists_parallelepiped_with_volume_144_l335_33556

/-- Represents the dimensions of a rectangular parallelepiped with a right triangle base -/
structure Parallelepiped where
  a : ℕ
  height : ℕ
  base_is_right_triangle : a^2 + (a+1)^2 = (a+2)^2

/-- The volume of the parallelepiped is 144 -/
theorem parallelepiped_volume (p : Parallelepiped) (h : p.height = 12) : a * (a + 1) * p.height = 144 := by
  sorry

/-- There exists a parallelepiped satisfying the conditions with volume 144 -/
theorem exists_parallelepiped_with_volume_144 : ∃ (p : Parallelepiped), p.height = 12 ∧ a * (a + 1) * p.height = 144 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_exists_parallelepiped_with_volume_144_l335_33556


namespace NUMINAMATH_CALUDE_adjacent_units_conversion_rate_l335_33554

-- Define the units of length
inductive LengthUnit
  | Kilometer
  | Meter
  | Decimeter
  | Centimeter
  | Millimeter

-- Define the concept of adjacent units
def adjacent (u v : LengthUnit) : Prop :=
  (u = LengthUnit.Kilometer ∧ v = LengthUnit.Meter) ∨
  (u = LengthUnit.Meter ∧ v = LengthUnit.Decimeter) ∨
  (u = LengthUnit.Decimeter ∧ v = LengthUnit.Centimeter) ∨
  (u = LengthUnit.Centimeter ∧ v = LengthUnit.Millimeter)

-- Define the conversion rate function
def conversionRate (u v : LengthUnit) : ℕ := 10

-- State the theorem
theorem adjacent_units_conversion_rate (u v : LengthUnit) :
  adjacent u v → conversionRate u v = 10 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_units_conversion_rate_l335_33554


namespace NUMINAMATH_CALUDE_students_not_participating_l335_33519

theorem students_not_participating (total : ℕ) (football : ℕ) (tennis : ℕ) (basketball : ℕ)
  (football_tennis : ℕ) (football_basketball : ℕ) (tennis_basketball : ℕ) (all_three : ℕ) :
  total = 50 →
  football = 30 →
  tennis = 25 →
  basketball = 18 →
  football_tennis = 12 →
  football_basketball = 10 →
  tennis_basketball = 8 →
  all_three = 5 →
  total - (football + tennis + basketball - football_tennis - football_basketball - tennis_basketball + all_three) = 2 :=
by sorry

end NUMINAMATH_CALUDE_students_not_participating_l335_33519


namespace NUMINAMATH_CALUDE_zach_ben_score_difference_l335_33565

theorem zach_ben_score_difference :
  ∀ (zach_score ben_score : ℕ),
    zach_score = 42 →
    ben_score = 21 →
    zach_score - ben_score = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_zach_ben_score_difference_l335_33565


namespace NUMINAMATH_CALUDE_transformation_result_l335_33598

def initial_point : ℝ × ℝ × ℝ := (1, 1, 1)

def rotate_y_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def transformation_sequence (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotate_y_180 |> reflect_yz |> reflect_xz |> rotate_y_180 |> reflect_xz

theorem transformation_result :
  transformation_sequence initial_point = (-1, 1, 1) := by
  sorry

#eval transformation_sequence initial_point

end NUMINAMATH_CALUDE_transformation_result_l335_33598


namespace NUMINAMATH_CALUDE_savings_calculation_l335_33500

theorem savings_calculation (income expenditure : ℕ) 
  (h1 : income = 36000)
  (h2 : income * 8 = expenditure * 9) : 
  income - expenditure = 4000 :=
sorry

end NUMINAMATH_CALUDE_savings_calculation_l335_33500


namespace NUMINAMATH_CALUDE_smallest_multiple_with_remainder_l335_33585

theorem smallest_multiple_with_remainder : ∃ (n : ℕ), 
  (∀ (m : ℕ), m < n → 
    (m % 5 = 0 ∧ m % 7 = 0 ∧ m % 3 = 1) → False) ∧ 
  n % 5 = 0 ∧ n % 7 = 0 ∧ n % 3 = 1 :=
by
  use 70
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_remainder_l335_33585


namespace NUMINAMATH_CALUDE_money_left_calculation_l335_33597

/-- The amount of money John has left after purchasing pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_cost := 4 * drink_cost + small_pizza_cost + 2 * large_pizza_cost
  50 - total_cost

/-- Theorem stating that the money left is equal to 50 - 13q -/
theorem money_left_calculation (q : ℝ) : money_left q = 50 - 13 * q := by
  sorry

end NUMINAMATH_CALUDE_money_left_calculation_l335_33597


namespace NUMINAMATH_CALUDE_april_earnings_l335_33550

/-- Calculates the total money earned from selling flowers -/
def total_money_earned (rose_price tulip_price daisy_price : ℕ) 
                       (roses_sold tulips_sold daisies_sold : ℕ) : ℕ :=
  rose_price * roses_sold + tulip_price * tulips_sold + daisy_price * daisies_sold

/-- Proves that April earned $78 from selling flowers -/
theorem april_earnings : 
  total_money_earned 4 3 2 9 6 12 = 78 := by sorry

end NUMINAMATH_CALUDE_april_earnings_l335_33550


namespace NUMINAMATH_CALUDE_part_I_part_II_l335_33543

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → m^2 - 3*m + x - 1 ≤ 0

def q (m a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧ m - a*x ≤ 0

-- Part I
theorem part_I : 
  ∃ S : Set ℝ, S = {m : ℝ | (m < 1 ∨ (1 < m ∧ m ≤ 2)) ∧ 
  ((p m ∧ ¬q m 1) ∨ (¬p m ∧ q m 1))} := by sorry

-- Part II
theorem part_II : 
  ∃ S : Set ℝ, S = {a : ℝ | a ≥ 2 ∨ a ≤ -2} ∧
  ∀ m : ℝ, (p m → q m a) ∧ ¬(q m a → p m) := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l335_33543


namespace NUMINAMATH_CALUDE_perfect_cube_units_digits_l335_33528

theorem perfect_cube_units_digits :
  ∀ d : Fin 10, ∃ n : ℤ, (n^3) % 10 = d.val :=
by sorry

end NUMINAMATH_CALUDE_perfect_cube_units_digits_l335_33528


namespace NUMINAMATH_CALUDE_incircle_tangent_inequality_l335_33538

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the incircle points
variable (A₁ B₁ : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h_triangle : Triangle A B C)
variable (h_incircle : IsIncircle A₁ B₁ A B C)
variable (h_AC_gt_BC : dist A C > dist B C)

-- State the theorem
theorem incircle_tangent_inequality :
  dist A A₁ > dist B B₁ := by sorry

end NUMINAMATH_CALUDE_incircle_tangent_inequality_l335_33538


namespace NUMINAMATH_CALUDE_downstream_speed_l335_33571

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- Theorem stating the downstream speed of a man given his upstream and still water speeds -/
theorem downstream_speed (s : RowingSpeed) (h1 : s.upstream = 25) (h2 : s.stillWater = 40) :
  s.downstream = 55 := by
  sorry

#check downstream_speed

end NUMINAMATH_CALUDE_downstream_speed_l335_33571


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l335_33518

theorem quadratic_rewrite (b : ℝ) (h1 : b < 0) :
  (∃ n : ℝ, ∀ x : ℝ, x^2 + b*x + 2/3 = (x + n)^2 + 1/4) →
  b = -Real.sqrt 15 / 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l335_33518


namespace NUMINAMATH_CALUDE_sin_negative_31pi_over_6_l335_33548

theorem sin_negative_31pi_over_6 : Real.sin (-31 * Real.pi / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_31pi_over_6_l335_33548


namespace NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l335_33589

def geometric_sequence (a₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₀ * r^n

theorem sum_of_fourth_and_fifth_terms (a₀ : ℝ) (r : ℝ) :
  (geometric_sequence a₀ r 5 = 4) →
  (geometric_sequence a₀ r 6 = 1) →
  (geometric_sequence a₀ r 2 = 256) →
  (geometric_sequence a₀ r 3 + geometric_sequence a₀ r 4 = 80) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l335_33589


namespace NUMINAMATH_CALUDE_seating_arrangement_l335_33522

/-- The number of students per row and total number of students in a seating arrangement problem. -/
theorem seating_arrangement (S R : ℕ) 
  (h1 : S = 5 * R + 6)  -- When 5 students sit in a row, 6 are left without seats
  (h2 : S = 12 * (R - 3))  -- When 12 students sit in a row, 3 rows are empty
  : R = 6 ∧ S = 36 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l335_33522


namespace NUMINAMATH_CALUDE_georges_income_proof_l335_33557

/-- George's monthly income in dollars -/
def monthly_income : ℝ := 240

/-- The amount George spent on groceries in dollars -/
def grocery_expense : ℝ := 20

/-- The amount George has left in dollars -/
def amount_left : ℝ := 100

/-- Theorem stating that George's monthly income is correct given the conditions -/
theorem georges_income_proof :
  monthly_income / 2 - grocery_expense = amount_left := by sorry

end NUMINAMATH_CALUDE_georges_income_proof_l335_33557


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l335_33515

theorem triangle_cosine_theorem (X Y Z : Real) :
  -- Triangle XYZ
  X + Y + Z = Real.pi →
  -- sin X = 4/5
  Real.sin X = 4/5 →
  -- cos Y = 12/13
  Real.cos Y = 12/13 →
  -- Then cos Z = -16/65
  Real.cos Z = -16/65 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l335_33515


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l335_33541

def daily_incomes : List ℝ := [45, 50, 60, 65, 70]
def num_days : ℕ := 5

theorem cab_driver_average_income : 
  (daily_incomes.sum / num_days : ℝ) = 58 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l335_33541


namespace NUMINAMATH_CALUDE_average_age_increase_l335_33507

theorem average_age_increase (n : ℕ) (m : ℕ) (avg_29 : ℝ) (age_30 : ℕ) :
  n = 30 →
  m = 29 →
  avg_29 = 12 →
  age_30 = 80 →
  let total_29 := m * avg_29
  let new_total := total_29 + age_30
  let new_avg := new_total / n
  abs (new_avg - avg_29 - 2.27) < 0.01 := by
sorry


end NUMINAMATH_CALUDE_average_age_increase_l335_33507


namespace NUMINAMATH_CALUDE_negation_equivalence_l335_33516

theorem negation_equivalence :
  (¬ (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l335_33516


namespace NUMINAMATH_CALUDE_exactly_one_positive_l335_33547

theorem exactly_one_positive (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (product_one : a * b * c = 1) :
  (a > 0 ∧ b ≤ 0 ∧ c ≤ 0) ∨
  (a ≤ 0 ∧ b > 0 ∧ c ≤ 0) ∨
  (a ≤ 0 ∧ b ≤ 0 ∧ c > 0) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_positive_l335_33547


namespace NUMINAMATH_CALUDE_gcd_2024_1728_l335_33509

theorem gcd_2024_1728 : Nat.gcd 2024 1728 = 8 := by sorry

end NUMINAMATH_CALUDE_gcd_2024_1728_l335_33509


namespace NUMINAMATH_CALUDE_x_in_P_sufficient_not_necessary_for_x_in_Q_l335_33577

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def Q : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1) + Real.sqrt (3 - x)}

-- State the theorem
theorem x_in_P_sufficient_not_necessary_for_x_in_Q :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by
  sorry

end NUMINAMATH_CALUDE_x_in_P_sufficient_not_necessary_for_x_in_Q_l335_33577


namespace NUMINAMATH_CALUDE_faye_pencils_l335_33534

/-- The number of rows of pencils and crayons -/
def num_rows : ℕ := 30

/-- The number of pencils in each row -/
def pencils_per_row : ℕ := 24

/-- The total number of pencils -/
def total_pencils : ℕ := num_rows * pencils_per_row

theorem faye_pencils : total_pencils = 720 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencils_l335_33534


namespace NUMINAMATH_CALUDE_lcm_of_prime_and_nonmultiple_lcm_1227_40_l335_33504

theorem lcm_of_prime_and_nonmultiple (p n : ℕ) (h_prime : Nat.Prime p) (h_not_dvd : ¬p ∣ n) :
  Nat.lcm p n = p * n :=
by sorry

theorem lcm_1227_40 :
  Nat.lcm 1227 40 = 49080 :=
by sorry

end NUMINAMATH_CALUDE_lcm_of_prime_and_nonmultiple_lcm_1227_40_l335_33504


namespace NUMINAMATH_CALUDE_set_union_problem_l335_33567

theorem set_union_problem (a b : ℝ) :
  let A : Set ℝ := {-1, a}
  let B : Set ℝ := {2^a, b}
  A ∩ B = {1} → A ∪ B = {-1, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l335_33567


namespace NUMINAMATH_CALUDE_power_multiplication_l335_33596

theorem power_multiplication (x : ℝ) : x^6 * x^2 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l335_33596


namespace NUMINAMATH_CALUDE_solve_linear_equation_l335_33521

theorem solve_linear_equation (x : ℝ) : 3*x - 5*x + 8*x = 240 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l335_33521


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l335_33535

/-- Represents a 4x4x4 cube with shaded faces -/
structure ShadedCube where
  /-- The number of smaller cubes along each edge of the large cube -/
  size : Nat
  /-- The number of shaded cubes in the central area of each face -/
  centralShaded : Nat
  /-- The number of shaded corner cubes per face -/
  cornerShaded : Nat

/-- Calculates the total number of uniquely shaded cubes -/
def totalShadedCubes (cube : ShadedCube) : Nat :=
  sorry

/-- Theorem stating that the total number of shaded cubes is 16 -/
theorem shaded_cubes_count (cube : ShadedCube) 
  (h1 : cube.size = 4)
  (h2 : cube.centralShaded = 4)
  (h3 : cube.cornerShaded = 1) : 
  totalShadedCubes cube = 16 := by
  sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l335_33535


namespace NUMINAMATH_CALUDE_grant_total_earnings_l335_33510

/-- Grant's earnings as a freelance math worker over three months -/
def grant_earnings : ℕ → ℕ
| 0 => 350  -- First month
| 1 => 2 * 350 + 50  -- Second month
| 2 => 4 * (grant_earnings 0 + grant_earnings 1)  -- Third month
| _ => 0  -- Other months (not relevant for this problem)

/-- The total earnings for the first three months -/
def total_earnings : ℕ := grant_earnings 0 + grant_earnings 1 + grant_earnings 2

theorem grant_total_earnings : total_earnings = 5500 := by
  sorry

end NUMINAMATH_CALUDE_grant_total_earnings_l335_33510


namespace NUMINAMATH_CALUDE_rectangular_plot_dimensions_l335_33506

theorem rectangular_plot_dimensions (length breadth : ℝ) : 
  length = 55 →
  breadth + (length - breadth) = length →
  4 * breadth + 2 * (length - breadth) = 5300 / 26.5 →
  length - breadth = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_dimensions_l335_33506


namespace NUMINAMATH_CALUDE_solve_problem_l335_33501

-- Define the sets A and B as functions of m
def A (m : ℤ) : Set ℤ := {-4, 2*m-1, m^2}
def B (m : ℤ) : Set ℤ := {9, m-5, 1-m}

-- Define the universal set U
def U : Set ℤ := Set.univ

-- State the theorem
theorem solve_problem (m : ℤ) 
  (h_intersection : A m ∩ B m = {9}) : 
  m = -3 ∧ A m ∩ (U \ B m) = {-4, -7} := by
  sorry


end NUMINAMATH_CALUDE_solve_problem_l335_33501


namespace NUMINAMATH_CALUDE_orange_apple_pear_weight_equivalence_l335_33576

/-- Represents the weight of a fruit -/
structure FruitWeight where
  weight : ℝ

/-- Represents the count of fruits -/
structure FruitCount where
  count : ℕ

/-- Given 9 oranges weigh the same as 6 apples and 1 pear, 
    prove that 36 oranges weigh the same as 24 apples and 4 pears -/
theorem orange_apple_pear_weight_equivalence 
  (orange : FruitWeight) 
  (apple : FruitWeight) 
  (pear : FruitWeight) 
  (h : 9 * orange.weight = 6 * apple.weight + pear.weight) : 
  36 * orange.weight = 24 * apple.weight + 4 * pear.weight := by
  sorry

end NUMINAMATH_CALUDE_orange_apple_pear_weight_equivalence_l335_33576


namespace NUMINAMATH_CALUDE_linear_equation_solution_l335_33581

/-- The linear equation 5x - y = 2 is satisfied by the point (1, 3) -/
theorem linear_equation_solution : 5 * 1 - 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l335_33581
