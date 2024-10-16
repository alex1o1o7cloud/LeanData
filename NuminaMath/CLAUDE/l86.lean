import Mathlib

namespace NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_l86_8626

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_to_same_line_are_parallel
  (a b c : Line) :
  parallel a c → parallel b c → parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_l86_8626


namespace NUMINAMATH_CALUDE_oliver_workout_total_l86_8692

/-- Oliver's workout schedule over four days -/
def workout_schedule (monday tuesday wednesday thursday : ℕ) : Prop :=
  monday = 4 ∧ 
  tuesday = monday - 2 ∧ 
  wednesday = 2 * monday ∧ 
  thursday = 2 * tuesday

/-- The total workout hours over four days -/
def total_hours (monday tuesday wednesday thursday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday

/-- Theorem: Given Oliver's workout schedule, the total hours worked out is 18 -/
theorem oliver_workout_total :
  ∀ (monday tuesday wednesday thursday : ℕ),
  workout_schedule monday tuesday wednesday thursday →
  total_hours monday tuesday wednesday thursday = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_oliver_workout_total_l86_8692


namespace NUMINAMATH_CALUDE_three_digit_number_rearrangement_l86_8608

def digit_sum (n : ℕ) : ℕ := n / 100 + (n / 10) % 10 + n % 10

def rearrangement_sum (abc : ℕ) : ℕ :=
  let a := abc / 100
  let b := (abc / 10) % 10
  let c := abc % 10
  (a * 100 + c * 10 + b) +
  (b * 100 + c * 10 + a) +
  (b * 100 + a * 10 + c) +
  (c * 100 + a * 10 + b) +
  (c * 100 + b * 10 + a)

theorem three_digit_number_rearrangement (abc : ℕ) :
  abc ≥ 100 ∧ abc < 1000 ∧ rearrangement_sum abc = 2670 → abc = 528 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_rearrangement_l86_8608


namespace NUMINAMATH_CALUDE_fort_blocks_count_l86_8696

/-- Represents the dimensions of a fort --/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed to construct a fort --/
def blocksNeeded (d : FortDimensions) (wallThickness : ℕ) (floorThickness : ℕ) : ℕ :=
  let outerVolume := d.length * d.width * d.height
  let innerLength := d.length - 2 * wallThickness
  let innerWidth := d.width - 2 * wallThickness
  let innerHeight := d.height - floorThickness
  let innerVolume := innerLength * innerWidth * innerHeight
  let topLayerVolume := d.length * d.width
  outerVolume - innerVolume + topLayerVolume

/-- Theorem stating that the number of blocks needed for the given fort is 912 --/
theorem fort_blocks_count :
  let fortDims : FortDimensions := ⟨15, 12, 7⟩
  blocksNeeded fortDims 2 1 = 912 := by
  sorry

end NUMINAMATH_CALUDE_fort_blocks_count_l86_8696


namespace NUMINAMATH_CALUDE_parabola_intersection_l86_8668

/-- 
Given a parabola y = 2x² translated right by p units and down by q units,
prove that it intersects y = x - 4 at exactly one point when p = q = 31/8.
-/
theorem parabola_intersection (p q : ℝ) : 
  (∃! x, 2*(x - p)^2 - q = x - 4) ↔ (p = 31/8 ∧ q = 31/8) := by
  sorry

#check parabola_intersection

end NUMINAMATH_CALUDE_parabola_intersection_l86_8668


namespace NUMINAMATH_CALUDE_mollys_age_l86_8650

/-- Given three friends with a total average age of 40, where Jared is ten years older than Hakimi,
    and Hakimi is 40 years old, prove that Molly is 30 years old. -/
theorem mollys_age (total_average : ℕ) (hakimi_age : ℕ) (jared_age : ℕ) (molly_age : ℕ) : 
  total_average = 40 →
  hakimi_age = 40 →
  jared_age = hakimi_age + 10 →
  (hakimi_age + jared_age + molly_age) / 3 = total_average →
  molly_age = 30 := by
sorry

end NUMINAMATH_CALUDE_mollys_age_l86_8650


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l86_8663

theorem quadratic_roots_sum_minus_product (x₁ x₂ : ℝ) : 
  (x₁^2 - x₁ - 2022 = 0) → 
  (x₂^2 - x₂ - 2022 = 0) → 
  x₁ + x₂ - x₁ * x₂ = 2023 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l86_8663


namespace NUMINAMATH_CALUDE_jellybean_average_increase_l86_8602

theorem jellybean_average_increase (initial_bags : ℕ) (initial_average : ℚ) (additional_jellybeans : ℕ) : 
  initial_bags = 34 →
  initial_average = 117 →
  additional_jellybeans = 362 →
  (((initial_bags : ℚ) * initial_average + additional_jellybeans) / (initial_bags + 1 : ℚ)) - initial_average = 7 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_average_increase_l86_8602


namespace NUMINAMATH_CALUDE_largest_circle_equation_l86_8633

/-- The standard equation of the circle with the largest radius, which is tangent to a line and has its center at (1, 0) -/
theorem largest_circle_equation (m : ℝ) : 
  ∃ (x y : ℝ), 
    (∀ (x' y' : ℝ), (2 * m * x' - y' - 4 * m + 1 = 0) → 
      ((x' - 1)^2 + y'^2 ≤ (x - 1)^2 + y^2)) ∧ 
    ((x - 1)^2 + y^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_largest_circle_equation_l86_8633


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l86_8627

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < 1038239 → ¬(618 ∣ (m + 1) ∧ 3648 ∣ (m + 1) ∧ 60 ∣ (m + 1))) ∧ 
  (618 ∣ (1038239 + 1) ∧ 3648 ∣ (1038239 + 1) ∧ 60 ∣ (1038239 + 1)) := by
  sorry


end NUMINAMATH_CALUDE_smallest_number_divisible_l86_8627


namespace NUMINAMATH_CALUDE_weight_of_new_person_l86_8652

/-- Given a group of 4 persons with a total weight W, if replacing a person
    weighing 65 kg with a new person increases the average weight by 1.5 kg,
    then the weight of the new person is 71 kg. -/
theorem weight_of_new_person (W : ℝ) : 
  (W - 65 + 71) / 4 = W / 4 + 1.5 := by sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l86_8652


namespace NUMINAMATH_CALUDE_terminal_side_quadrant_l86_8638

/-- The quadrant in which an angle falls -/
inductive Quadrant
| I
| II
| III
| IV

/-- Determines the quadrant of an angle in degrees -/
def angle_quadrant (angle : Int) : Quadrant :=
  let normalized_angle := angle % 360
  if 0 ≤ normalized_angle && normalized_angle < 90 then Quadrant.I
  else if 90 ≤ normalized_angle && normalized_angle < 180 then Quadrant.II
  else if 180 ≤ normalized_angle && normalized_angle < 270 then Quadrant.III
  else Quadrant.IV

theorem terminal_side_quadrant :
  angle_quadrant (-1060) = Quadrant.I :=
sorry

end NUMINAMATH_CALUDE_terminal_side_quadrant_l86_8638


namespace NUMINAMATH_CALUDE_hospital_nurses_count_l86_8670

theorem hospital_nurses_count 
  (total_staff : ℕ) 
  (doctor_ratio : ℕ) 
  (nurse_ratio : ℕ) 
  (h1 : total_staff = 200)
  (h2 : doctor_ratio = 4)
  (h3 : nurse_ratio = 6) :
  (nurse_ratio * total_staff) / (doctor_ratio + nurse_ratio) = 120 := by
  sorry

end NUMINAMATH_CALUDE_hospital_nurses_count_l86_8670


namespace NUMINAMATH_CALUDE_gcd_problem_l86_8614

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 1632 * k) : 
  Int.gcd (a^2 + 13*a + 36) (a + 6) = 6 := by
sorry

end NUMINAMATH_CALUDE_gcd_problem_l86_8614


namespace NUMINAMATH_CALUDE_regular_polygon_area_l86_8636

theorem regular_polygon_area (n : ℕ) (R : ℝ) (h : R > 0) :
  (1 / 2 : ℝ) * n * R^2 * Real.sin (2 * Real.pi / n) = 3 * R^2 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_area_l86_8636


namespace NUMINAMATH_CALUDE_sum_a_b_equals_one_l86_8683

theorem sum_a_b_equals_one (a b : ℝ) (h : a = Real.sqrt (2 * b - 4) + Real.sqrt (4 - 2 * b) - 1) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_one_l86_8683


namespace NUMINAMATH_CALUDE_perfect_linear_correlation_l86_8644

/-- A scatter plot where all points lie on a straight line with non-zero slope -/
structure PerfectLinearScatterPlot where
  points : Set (ℝ × ℝ)
  non_zero_slope : ℝ
  line_equation : ℝ → ℝ
  all_points_on_line : ∀ (x y : ℝ), (x, y) ∈ points → y = line_equation x
  slope_non_zero : non_zero_slope ≠ 0

/-- The correlation coefficient of a scatter plot -/
def correlation_coefficient (plot : PerfectLinearScatterPlot) : ℝ :=
  sorry

/-- Theorem: The correlation coefficient of a perfect linear scatter plot is 1 -/
theorem perfect_linear_correlation (plot : PerfectLinearScatterPlot) :
  correlation_coefficient plot = 1 :=
sorry

end NUMINAMATH_CALUDE_perfect_linear_correlation_l86_8644


namespace NUMINAMATH_CALUDE_remainder_theorem_l86_8646

theorem remainder_theorem : (9^6 + 5^7 + 3^8) % 7 = 4 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l86_8646


namespace NUMINAMATH_CALUDE_chocolates_bought_at_cost_price_l86_8630

/-- The number of chocolates bought at the cost price -/
def n : ℕ := 65

/-- The cost price of one chocolate -/
def C : ℝ := 1

/-- The selling price of one chocolate -/
def S : ℝ := 1.3 * C

/-- The gain percent -/
def gain_percent : ℝ := 30

theorem chocolates_bought_at_cost_price :
  (n * C = 50 * S) ∧ 
  (gain_percent = (S - C) / C * 100) →
  n = 65 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_bought_at_cost_price_l86_8630


namespace NUMINAMATH_CALUDE_smallest_integer_sequence_sum_l86_8632

theorem smallest_integer_sequence_sum (B : ℤ) : B = -2022 ↔ 
  (∃ n : ℕ, (Finset.range n).sum (λ i => B + i) = 2023) ∧ 
  (∀ k < B, ¬∃ m : ℕ, (Finset.range m).sum (λ i => k + i) = 2023) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_sequence_sum_l86_8632


namespace NUMINAMATH_CALUDE_relationship_A_and_p_l86_8662

theorem relationship_A_and_p (x y p : ℝ) (A : ℝ) 
  (h1 : A = (x^2 - 3*y^2) / (3*x^2 + y^2))
  (h2 : p*x*y / (x^2 - (2+p)*x*y + 2*p*y^2) - y / (x - 2*y) = 1/2)
  (h3 : x ≠ 0)
  (h4 : y ≠ 0)
  (h5 : x ≠ 2*y)
  (h6 : x ≠ p*y) :
  A = (9*p^2 - 3) / (27*p^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_relationship_A_and_p_l86_8662


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l86_8666

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  a 8 = 24 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l86_8666


namespace NUMINAMATH_CALUDE_remainder_zero_mod_eight_l86_8694

theorem remainder_zero_mod_eight :
  (71^7 - 73^10) * (73^5 + 71^3) ≡ 0 [ZMOD 8] := by
sorry

end NUMINAMATH_CALUDE_remainder_zero_mod_eight_l86_8694


namespace NUMINAMATH_CALUDE_mrs_randall_third_grade_years_l86_8674

/-- Represents the number of years Mrs. Randall has been teaching -/
def total_teaching_years : ℕ := 26

/-- Represents the number of years Mrs. Randall taught second grade -/
def second_grade_years : ℕ := 8

/-- Represents the number of years Mrs. Randall has taught third grade -/
def third_grade_years : ℕ := total_teaching_years - second_grade_years

theorem mrs_randall_third_grade_years :
  third_grade_years = 18 :=
by sorry

end NUMINAMATH_CALUDE_mrs_randall_third_grade_years_l86_8674


namespace NUMINAMATH_CALUDE_area_NPQ_approx_l86_8658

/-- Triangle XYZ with given side lengths -/
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  xy_length : dist X Y = 15
  xz_length : dist X Z = 20
  yz_length : dist Y Z = 13

/-- P is the circumcenter of triangle XYZ -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Q is the incenter of triangle XYZ -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- N is the center of a circle tangent to sides XZ, YZ, and the circumcircle of XYZ -/
def excircle_center (t : Triangle) : ℝ × ℝ := sorry

/-- The area of triangle NPQ -/
def area_NPQ (t : Triangle) : ℝ := sorry

/-- Theorem stating the area of triangle NPQ is approximately 49.21 -/
theorem area_NPQ_approx (t : Triangle) : 
  abs (area_NPQ t - 49.21) < 0.01 := by sorry

end NUMINAMATH_CALUDE_area_NPQ_approx_l86_8658


namespace NUMINAMATH_CALUDE_game_board_probability_l86_8654

/-- Represents a triangle on the game board --/
structure GameTriangle :=
  (is_isosceles_right : Bool)
  (num_subdivisions : Nat)
  (num_shaded : Nat)

/-- Calculates the probability of landing in a shaded region --/
def probability_shaded (t : GameTriangle) : ℚ :=
  t.num_shaded / t.num_subdivisions

/-- The main theorem stating the probability for the specific game board configuration --/
theorem game_board_probability (t : GameTriangle) :
  t.is_isosceles_right = true →
  t.num_subdivisions = 6 →
  t.num_shaded = 2 →
  probability_shaded t = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_game_board_probability_l86_8654


namespace NUMINAMATH_CALUDE_square_diagonals_sum_l86_8618

theorem square_diagonals_sum (x y : ℝ) (h1 : x^2 + y^2 = 145) (h2 : x^2 - y^2 = 85) :
  x * Real.sqrt 2 + y * Real.sqrt 2 = Real.sqrt 230 + Real.sqrt 60 := by
  sorry

#check square_diagonals_sum

end NUMINAMATH_CALUDE_square_diagonals_sum_l86_8618


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l86_8665

/-- The minimum distance between points on two specific curves -/
theorem min_distance_between_curves : ∃ (d : ℝ),
  (∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 →
    let p := (x₁, (1/2) * Real.exp x₁)
    let q := (x₂, Real.log (2 * x₂))
    d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
  d = Real.sqrt 2 * (1 - Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l86_8665


namespace NUMINAMATH_CALUDE_pyramid_inscribed_cube_volume_l86_8675

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid where
  base_side : ℝ
  height : ℝ

/-- A cube inscribed in the pyramid -/
structure InscribedCube where
  edge_length : ℝ

/-- The volume of a cube -/
def cube_volume (c : InscribedCube) : ℝ := c.edge_length ^ 3

theorem pyramid_inscribed_cube_volume 
  (p : Pyramid) 
  (c : InscribedCube) 
  (h_base : p.base_side = 2) 
  (h_height : p.height = Real.sqrt 6) 
  (h_cube_edge : c.edge_length = Real.sqrt 6 / 3) : 
  cube_volume c = 2 * Real.sqrt 6 / 9 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_inscribed_cube_volume_l86_8675


namespace NUMINAMATH_CALUDE_solve_nested_function_l86_8641

def f (p : ℝ) : ℝ := 2 * p + 20

theorem solve_nested_function : ∃ p : ℝ, f (f (f p)) = -4 ∧ p = -18 := by
  sorry

end NUMINAMATH_CALUDE_solve_nested_function_l86_8641


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l86_8619

/-- Given three consecutive integers whose product is 210 and whose sum of squares is minimal, 
    the sum of the two smallest of these integers is 11. -/
theorem consecutive_integers_problem :
  ∀ n : ℤ, 
  (n - 1) * n * (n + 1) = 210 → 
  (∀ m : ℤ, m * (m + 1) * (m + 2) = 210 → 
    (n - 1)^2 + n^2 + (n + 1)^2 ≤ (m - 1)^2 + m^2 + (m + 1)^2) →
  (n - 1) + n = 11 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l86_8619


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l86_8686

theorem triangle_determinant_zero (A B C : ℝ) (h : A + B + C = π) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.cos A ^ 2, Real.tan A, 1],
    ![Real.cos B ^ 2, Real.tan B, 1],
    ![Real.cos C ^ 2, Real.tan C, 1]
  ]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l86_8686


namespace NUMINAMATH_CALUDE_range_of_a_l86_8691

theorem range_of_a (p q : Prop) (h_p : ∀ x : ℝ, x ∈ Set.Icc 0 1 → ∃ a : ℝ, a ≥ Real.exp x) 
  (h_q : ∃ (a : ℝ) (x : ℝ), x^2 + 4*x + a = 0) (h_pq : p ∧ q) :
  ∃ a : ℝ, a ∈ Set.Icc (Real.exp 1) 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l86_8691


namespace NUMINAMATH_CALUDE_total_available_seats_l86_8628

/-- Represents a bus with its seating configuration and broken seats -/
structure Bus where
  columns : ℕ
  rows_left : ℕ
  rows_right : ℕ
  broken_seats : ℕ

/-- Calculates the number of available seats in a bus -/
def available_seats (bus : Bus) : ℕ :=
  bus.columns * (bus.rows_left + bus.rows_right) - bus.broken_seats

/-- The list of buses with their configurations -/
def buses : List Bus := [
  ⟨4, 10, 0, 2⟩,   -- Bus 1
  ⟨5, 8, 0, 4⟩,    -- Bus 2
  ⟨3, 12, 0, 3⟩,   -- Bus 3
  ⟨4, 6, 8, 1⟩,    -- Bus 4
  ⟨6, 8, 10, 5⟩,   -- Bus 5
  ⟨5, 8, 2, 4⟩     -- Bus 6 (2 rows with 2 seats each unavailable)
]

/-- Theorem stating that the total number of available seats is 311 -/
theorem total_available_seats :
  (buses.map available_seats).sum = 311 := by
  sorry


end NUMINAMATH_CALUDE_total_available_seats_l86_8628


namespace NUMINAMATH_CALUDE_exists_subset_with_constant_gcd_l86_8698

/-- A function that checks if a natural number is the product of at most 2000 distinct primes -/
def is_product_of_limited_primes (n : ℕ) : Prop :=
  ∃ (primes : Finset ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ primes.card ≤ 2000 ∧ n = primes.prod id

/-- The main theorem -/
theorem exists_subset_with_constant_gcd 
  (A : Set ℕ) 
  (h_infinite : Set.Infinite A) 
  (h_limited_primes : ∀ a ∈ A, is_product_of_limited_primes a) :
  ∃ (B : Set ℕ) (k : ℕ), Set.Infinite B ∧ B ⊆ A ∧ 
    ∀ (b1 b2 : ℕ), b1 ∈ B → b2 ∈ B → b1 ≠ b2 → Nat.gcd b1 b2 = k :=
sorry

end NUMINAMATH_CALUDE_exists_subset_with_constant_gcd_l86_8698


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l86_8631

theorem quadratic_no_real_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k - 1 ≠ 0) → k > 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l86_8631


namespace NUMINAMATH_CALUDE_set_operations_l86_8640

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 0 ≤ x ∧ x < 5}

def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}

theorem set_operations :
  (A ∩ B = {x | 0 ≤ x ∧ x < 4}) ∧
  (A ∪ B = {x | -2 ≤ x ∧ x < 5}) ∧
  (A ∩ (U \ B) = {x | 4 ≤ x ∧ x < 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l86_8640


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l86_8672

theorem money_left_after_purchase (initial_amount spent_amount : ℕ) : 
  initial_amount = 90 → spent_amount = 78 → initial_amount - spent_amount = 12 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l86_8672


namespace NUMINAMATH_CALUDE_inscribed_triangle_existence_l86_8669

/-- A circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A triangle defined by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Check if a triangle is inscribed in a circle -/
def isInscribed (t : Triangle) (c : Circle) : Prop :=
  sorry

/-- Calculate an angle of a triangle -/
def angle (t : Triangle) (vertex : ℝ × ℝ) : ℝ :=
  sorry

/-- Calculate the length of a median in a triangle -/
def medianLength (t : Triangle) (vertex : ℝ × ℝ) : ℝ :=
  sorry

/-- The main theorem -/
theorem inscribed_triangle_existence (k : Circle) (α : ℝ) (s_b : ℝ) :
  ∃ n : Fin 3, ∃ triangles : Fin n → Triangle,
    (∀ i, isInscribed (triangles i) k) ∧
    (∀ i, ∃ v, angle (triangles i) v = α) ∧
    (∀ i, ∃ v, medianLength (triangles i) v = s_b) :=
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_existence_l86_8669


namespace NUMINAMATH_CALUDE_max_remaining_pairwise_sums_l86_8685

theorem max_remaining_pairwise_sums (a b c d : ℝ) : 
  let sums : List ℝ := [a + b, a + c, a + d, b + c, b + d, c + d]
  (172 ∈ sums) ∧ (305 ∈ sums) ∧ (250 ∈ sums) ∧ (215 ∈ sums) →
  (∃ (x y : ℝ), x ∈ sums ∧ y ∈ sums ∧ x ≠ 172 ∧ x ≠ 305 ∧ x ≠ 250 ∧ x ≠ 215 ∧
                 y ≠ 172 ∧ y ≠ 305 ∧ y ≠ 250 ∧ y ≠ 215 ∧ x ≠ y ∧
                 x + y ≤ 723) ∧
  (∃ (a' b' c' d' : ℝ), 
    let sums' : List ℝ := [a' + b', a' + c', a' + d', b' + c', b' + d', c' + d']
    (172 ∈ sums') ∧ (305 ∈ sums') ∧ (250 ∈ sums') ∧ (215 ∈ sums') ∧
    (∃ (x' y' : ℝ), x' ∈ sums' ∧ y' ∈ sums' ∧ x' ≠ 172 ∧ x' ≠ 305 ∧ x' ≠ 250 ∧ x' ≠ 215 ∧
                     y' ≠ 172 ∧ y' ≠ 305 ∧ y' ≠ 250 ∧ y' ≠ 215 ∧ x' ≠ y' ∧
                     x' + y' = 723)) :=
by sorry

end NUMINAMATH_CALUDE_max_remaining_pairwise_sums_l86_8685


namespace NUMINAMATH_CALUDE_cups_in_class_l86_8611

/-- Represents the number of cups brought to class by students --/
def cups_brought (total_students : ℕ) (num_boys : ℕ) (cups_per_boy : ℕ) : ℕ :=
  num_boys * cups_per_boy

/-- Theorem stating that given the conditions, 50 cups were brought to class --/
theorem cups_in_class (total_students : ℕ) (num_boys : ℕ) (cups_per_boy : ℕ) :
  total_students = 30 →
  num_boys = 10 →
  cups_per_boy = 5 →
  2 * num_boys = total_students - num_boys →
  cups_brought total_students num_boys cups_per_boy = 50 := by
  sorry


end NUMINAMATH_CALUDE_cups_in_class_l86_8611


namespace NUMINAMATH_CALUDE_total_change_is_390_cents_l86_8607

/-- Represents the amount of money in different currencies -/
structure Money where
  cad_quarters : ℕ
  gbp : ℕ
  usd_quarters : ℕ

/-- Represents exchange rates -/
structure ExchangeRates where
  cad_to_usd : ℚ
  gbp_to_usd : ℚ

/-- Represents a transaction -/
structure Transaction where
  spend_amount : ℚ
  change_amount : ℚ
  currency : String

/-- Calculate the total change in US cents after a series of transactions -/
def calculate_total_change (
  initial_money : Money
  ) (
  exchange_rates : ExchangeRates
  ) (
  transactions : List Transaction
  ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem total_change_is_390_cents (
  initial_money : Money
  ) (
  exchange_rates : ExchangeRates
  ) (
  transactions : List Transaction
  ) : 
  initial_money.cad_quarters = 14 ∧
  initial_money.gbp = 5 ∧
  initial_money.usd_quarters = 20 ∧
  exchange_rates.cad_to_usd = 4/5 ∧
  exchange_rates.gbp_to_usd = 7/5 ∧
  transactions = [
    { spend_amount := 1, change_amount := 0, currency := "CAD" },
    { spend_amount := 2, change_amount := 1, currency := "GBP" },
    { spend_amount := 5, change_amount := 5/2, currency := "USD" }
  ] →
  calculate_total_change initial_money exchange_rates transactions = 390 :=
  sorry

end NUMINAMATH_CALUDE_total_change_is_390_cents_l86_8607


namespace NUMINAMATH_CALUDE_bobby_total_candy_and_chocolate_l86_8616

def candy_initial : ℕ := 33
def candy_additional : ℕ := 4
def chocolate : ℕ := 14

theorem bobby_total_candy_and_chocolate :
  candy_initial + candy_additional + chocolate = 51 := by
  sorry

end NUMINAMATH_CALUDE_bobby_total_candy_and_chocolate_l86_8616


namespace NUMINAMATH_CALUDE_seating_arrangements_eq_48_l86_8676

/-- The number of ways to seat 7 people around a round table with constraints -/
def seating_arrangements : ℕ :=
  let total_people : ℕ := 7
  let fixed_people : ℕ := 3  -- Alice, Bob, and Carol
  let remaining_people : ℕ := total_people - fixed_people
  let ways_to_arrange_bob_and_carol : ℕ := 2
  ways_to_arrange_bob_and_carol * (Nat.factorial remaining_people)

/-- Theorem stating that the number of seating arrangements is 48 -/
theorem seating_arrangements_eq_48 : seating_arrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_eq_48_l86_8676


namespace NUMINAMATH_CALUDE_binomial_1500_1_l86_8651

theorem binomial_1500_1 : Nat.choose 1500 1 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1500_1_l86_8651


namespace NUMINAMATH_CALUDE_inequality_solution_set_l86_8661

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x + 2| - |x - 1| < a) → a > -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l86_8661


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l86_8601

theorem polygon_interior_angle_sum (n : ℕ) (h : n > 2) :
  (n * 40 = 360) →
  (n - 2) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l86_8601


namespace NUMINAMATH_CALUDE_not_eventually_periodic_l86_8690

/-- The rightmost non-zero digit in the decimal representation of n! -/
def rightmost_nonzero_digit (n : ℕ) : ℕ :=
  sorry

/-- The sequence of rightmost non-zero digits of factorials -/
def a : ℕ → ℕ := rightmost_nonzero_digit

/-- The sequence (a_n)_{n ≥ 0} is not periodic from any certain point onwards -/
theorem not_eventually_periodic :
  ∀ p q : ℕ, ∃ n : ℕ, n ≥ q ∧ a n ≠ a (n + p) :=
sorry

end NUMINAMATH_CALUDE_not_eventually_periodic_l86_8690


namespace NUMINAMATH_CALUDE_average_percentage_increase_l86_8648

/-- Given an item with original price of 100 yuan, increased first by 40% and then by 10%,
    prove that the average percentage increase x per time satisfies (1 + 40%)(1 + 10%) = (1 + x)² -/
theorem average_percentage_increase (original_price : ℝ) (first_increase second_increase : ℝ) 
  (x : ℝ) (h1 : original_price = 100) (h2 : first_increase = 0.4) (h3 : second_increase = 0.1) :
  (1 + first_increase) * (1 + second_increase) = (1 + x)^2 := by
  sorry

end NUMINAMATH_CALUDE_average_percentage_increase_l86_8648


namespace NUMINAMATH_CALUDE_solution_set_inequality_l86_8622

theorem solution_set_inequality (x : ℝ) (h : x ≠ 0) :
  (2*x - 1) / x < 1 ↔ 0 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l86_8622


namespace NUMINAMATH_CALUDE_function_difference_l86_8610

theorem function_difference (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 - 3 * x + 5
  let g : ℝ → ℝ := λ x ↦ 2 * x^2 - k * x + 1
  (f 10 - g 10 = 20) → k = -21.4 := by
sorry

end NUMINAMATH_CALUDE_function_difference_l86_8610


namespace NUMINAMATH_CALUDE_two_lines_at_constant_distance_l86_8695

/-- A line in a plane -/
structure Line where
  -- Add necessary fields to define a line

/-- Distance between two lines in a plane -/
def distance (l1 l2 : Line) : ℝ :=
  sorry

/-- Theorem: There are exactly two lines at a constant distance of 2 from a given line -/
theorem two_lines_at_constant_distance (l : Line) :
  ∃! (pair : (Line × Line)), (distance l pair.1 = 2 ∧ distance l pair.2 = 2) :=
sorry

end NUMINAMATH_CALUDE_two_lines_at_constant_distance_l86_8695


namespace NUMINAMATH_CALUDE_smallest_shift_l86_8637

-- Define a periodic function g with period 25
def g (x : ℝ) : ℝ := sorry

-- Define the period of g
def period : ℝ := 25

-- State the periodicity of g
axiom g_periodic (x : ℝ) : g (x - period) = g x

-- Define the property we want to prove
def property (a : ℝ) : Prop :=
  ∀ x, g ((x - a) / 4) = g (x / 4)

-- State the theorem
theorem smallest_shift :
  (∃ a > 0, property a) ∧ (∀ a > 0, property a → a ≥ 100) :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_l86_8637


namespace NUMINAMATH_CALUDE_divisible_by_four_or_seven_count_divisible_by_four_or_seven_l86_8615

theorem divisible_by_four_or_seven (n : Nat) : 
  (∃ k : Nat, n = 4 * k ∨ n = 7 * k) ↔ n ∈ Finset.filter (λ x : Nat => x % 4 = 0 ∨ x % 7 = 0) (Finset.range 61) :=
sorry

theorem count_divisible_by_four_or_seven : 
  (Finset.filter (λ x : Nat => x % 4 = 0 ∨ x % 7 = 0) (Finset.range 61)).card = 21 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_four_or_seven_count_divisible_by_four_or_seven_l86_8615


namespace NUMINAMATH_CALUDE_f_neg_two_eq_three_l86_8660

-- Define the function f
def f (x : ℝ) : ℝ := -2 * (x + 1) + 1

-- Theorem statement
theorem f_neg_two_eq_three : f (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_three_l86_8660


namespace NUMINAMATH_CALUDE_xyz_product_zero_l86_8684

theorem xyz_product_zero (x y z : ℝ) 
  (eq1 : x + 1/y = 1) 
  (eq2 : y + 1/z = 1) 
  (eq3 : z + 1/x = 1) : 
  x * y * z = 0 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_zero_l86_8684


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l86_8635

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1/2 →
  a 2 * a 4 = 4 * (a 3 - 1) →
  a 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l86_8635


namespace NUMINAMATH_CALUDE_only_one_correct_proposition_l86_8671

-- Define a predicate for each proposition
def proposition1 (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (x + 1)) → (∃ a b, ∀ x, f x = a * Real.sin (b * x) + a * Real.cos (b * x))

def proposition2 : Prop :=
  ∃ x : ℝ, x^2 - x > 0

def proposition3 (A B : ℝ) : Prop :=
  Real.sin A > Real.sin B ↔ A > B

def proposition4 (f : ℝ → ℝ) : Prop :=
  (∃ x ∈ Set.Ioo 2015 2017, f x = 0) → f 2015 * f 2017 < 0

def proposition5 (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ (1 / x) * (1 / y) = -1

-- Theorem stating that only one proposition is correct
theorem only_one_correct_proposition :
  (¬ ∀ f, proposition1 f) ∧
  (¬ proposition2) ∧
  (∀ A B, proposition3 A B) ∧
  (¬ ∀ f, proposition4 f) ∧
  (¬ proposition5 Real.log) :=
sorry

end NUMINAMATH_CALUDE_only_one_correct_proposition_l86_8671


namespace NUMINAMATH_CALUDE_circles_tangent_radii_product_l86_8620

/-- Given two circles in a plane with radii r₁ and r₂, and distance d between their centers,
    if their common external tangent has length 2017 and their common internal tangent has length 2009,
    then the product of their radii is 8052. -/
theorem circles_tangent_radii_product (r₁ r₂ d : ℝ) 
  (h_external : d^2 - (r₁ + r₂)^2 = 2017^2)
  (h_internal : d^2 - (r₁ - r₂)^2 = 2009^2) :
  r₁ * r₂ = 8052 := by
  sorry

end NUMINAMATH_CALUDE_circles_tangent_radii_product_l86_8620


namespace NUMINAMATH_CALUDE_line_equation_slope_5_through_0_2_l86_8645

/-- The equation of a line with slope 5 passing through (0, 2) -/
theorem line_equation_slope_5_through_0_2 :
  ∀ (x y : ℝ), (5 * x - y + 2 = 0) ↔ 
  (∃ (t : ℝ), x = t ∧ y = 5 * t + 2) := by sorry

end NUMINAMATH_CALUDE_line_equation_slope_5_through_0_2_l86_8645


namespace NUMINAMATH_CALUDE_train_speed_l86_8603

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 360 →
  bridge_length = 140 →
  time = 50 →
  (train_length + bridge_length) / time * 3.6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l86_8603


namespace NUMINAMATH_CALUDE_outfit_combinations_l86_8643

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (shoes : ℕ) :
  shirts = 4 → pants = 5 → shoes = 3 →
  shirts * pants * shoes = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l86_8643


namespace NUMINAMATH_CALUDE_curve_symmetry_l86_8677

/-- A curve f is symmetric with respect to the line x - y - 3 = 0 if and only if
    it can be expressed as f(y+3, x-3) = 0 for all x and y. -/
theorem curve_symmetry (f : ℝ → ℝ → ℝ) :
  (∀ x y, f x y = 0 ↔ f (y + 3) (x - 3) = 0) ↔
  (∀ x y, (x - y = 3) → (f x y = 0 ↔ f y x = 0)) :=
sorry

end NUMINAMATH_CALUDE_curve_symmetry_l86_8677


namespace NUMINAMATH_CALUDE_painted_equals_unpainted_l86_8656

/-- Represents a cube with edge length n, painted on two adjacent faces and sliced into unit cubes -/
structure PaintedCube where
  n : ℕ
  n_gt_two : n > 2

/-- The number of smaller cubes with exactly two faces painted -/
def two_faces_painted (c : PaintedCube) : ℕ := c.n - 2

/-- The number of smaller cubes completely without paint -/
def unpainted (c : PaintedCube) : ℕ := (c.n - 2)^3

/-- Theorem stating that the number of cubes with two faces painted equals the number of unpainted cubes if and only if n = 3 -/
theorem painted_equals_unpainted (c : PaintedCube) : 
  two_faces_painted c = unpainted c ↔ c.n = 3 := by
  sorry

end NUMINAMATH_CALUDE_painted_equals_unpainted_l86_8656


namespace NUMINAMATH_CALUDE_max_min_difference_l86_8629

theorem max_min_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (heq : x + 2 * y = 4) :
  ∃ (max min : ℝ), 
    (∀ z w : ℝ, z ≠ 0 → w ≠ 0 → z + 2 * w = 4 → |2 * z - w| / (|z| + |w|) ≤ max) ∧
    (∀ z w : ℝ, z ≠ 0 → w ≠ 0 → z + 2 * w = 4 → min ≤ |2 * z - w| / (|z| + |w|)) ∧
    max - min = 5 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_l86_8629


namespace NUMINAMATH_CALUDE_book_purchase_ratio_l86_8624

/-- The number of people who purchased only book A -/
def only_A : ℕ := 1000

/-- The number of people who purchased both books A and B -/
def both_A_and_B : ℕ := 500

/-- The number of people who purchased only book B -/
def only_B : ℕ := both_A_and_B / 2

/-- The total number of people who purchased book A -/
def total_A : ℕ := only_A + both_A_and_B

/-- The total number of people who purchased book B -/
def total_B : ℕ := only_B + both_A_and_B

/-- The ratio of people who purchased book A to those who purchased book B -/
def ratio : ℚ := total_A / total_B

theorem book_purchase_ratio : ratio = 2 := by sorry

end NUMINAMATH_CALUDE_book_purchase_ratio_l86_8624


namespace NUMINAMATH_CALUDE_period_multiple_l86_8623

/-- A function f is periodic with period l if f(x + l) = f(x) for all x in the domain of f -/
def IsPeriodic (f : ℝ → ℝ) (l : ℝ) : Prop :=
  ∀ x, f (x + l) = f x

/-- If l is a period of f, then nl is also a period of f for any natural number n -/
theorem period_multiple {f : ℝ → ℝ} {l : ℝ} (h : IsPeriodic f l) (n : ℕ) :
  IsPeriodic f (n * l) := by
  sorry


end NUMINAMATH_CALUDE_period_multiple_l86_8623


namespace NUMINAMATH_CALUDE_flu_virus_diameter_scientific_notation_l86_8678

theorem flu_virus_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000054 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 5.4 ∧ n = -6 :=
sorry

end NUMINAMATH_CALUDE_flu_virus_diameter_scientific_notation_l86_8678


namespace NUMINAMATH_CALUDE_no_valid_n_l86_8605

/-- Represents the number of matches won by women -/
def women_wins (n : ℕ) : ℚ := 3 * (n * (4 * n - 1) / 8)

/-- Represents the number of matches won by men -/
def men_wins (n : ℕ) : ℚ := 5 * (n * (4 * n - 1) / 8)

/-- Represents the total number of matches played -/
def total_matches (n : ℕ) : ℕ := n * (4 * n - 1) / 2

theorem no_valid_n : ∀ n : ℕ, n > 0 →
  (women_wins n + men_wins n = total_matches n) →
  (3 * men_wins n = 5 * women_wins n) →
  False :=
sorry

end NUMINAMATH_CALUDE_no_valid_n_l86_8605


namespace NUMINAMATH_CALUDE_grand_hall_expenditure_l86_8649

/-- Calculates the total expenditure for covering a rectangular floor with a mat -/
def total_expenditure (length width cost_per_sqm : ℝ) : ℝ :=
  length * width * cost_per_sqm

/-- Proves that the total expenditure for covering a 50m × 30m floor with a mat 
    costing Rs. 100 per square meter is Rs. 150,000 -/
theorem grand_hall_expenditure :
  total_expenditure 50 30 100 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_grand_hall_expenditure_l86_8649


namespace NUMINAMATH_CALUDE_card_probabilities_l86_8606

def deck : Finset ℕ := {1, 2, 3, 4}

def prob_first_draw (n : ℕ) : ℚ :=
  if n ∈ deck then 1 / (deck.card : ℚ) else 0

def prob_second_draw (n : ℕ) : ℚ :=
  (deck.card - 1 : ℚ) / (deck.card : ℚ) * (1 / (deck.card - 1 : ℚ))

def prob_any_draw (n : ℕ) : ℚ :=
  prob_first_draw n + prob_second_draw n

theorem card_probabilities :
  prob_first_draw 4 = 1/4 ∧
  prob_second_draw 4 = 1/4 ∧
  prob_any_draw 4 = 1/2 := by sorry

end NUMINAMATH_CALUDE_card_probabilities_l86_8606


namespace NUMINAMATH_CALUDE_apples_after_sharing_l86_8647

/-- The total number of apples Craig and Judy have after sharing -/
def total_apples_after_sharing (craig_initial : ℕ) (judy_initial : ℕ) (craig_shared : ℕ) (judy_shared : ℕ) : ℕ :=
  (craig_initial - craig_shared) + (judy_initial - judy_shared)

/-- Theorem: Given the initial and shared apple counts, Craig and Judy have 19 apples together after sharing -/
theorem apples_after_sharing :
  total_apples_after_sharing 20 11 7 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_apples_after_sharing_l86_8647


namespace NUMINAMATH_CALUDE_vector_dot_product_problem_l86_8612

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem vector_dot_product_problem (x : ℝ) :
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (5, -3)
  dot_product a b = 7 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_problem_l86_8612


namespace NUMINAMATH_CALUDE_square_sum_of_system_l86_8657

theorem square_sum_of_system (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 10344 / 169 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_system_l86_8657


namespace NUMINAMATH_CALUDE_real_part_of_complex_number_l86_8664

theorem real_part_of_complex_number (i : ℂ) (h : i^2 = -1) : 
  Complex.re ((-1 + 2*i)*i) = -2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_number_l86_8664


namespace NUMINAMATH_CALUDE_farmer_cow_division_l86_8679

theorem farmer_cow_division (herd : ℕ) : 
  (herd / 3 : ℕ) + (herd / 6 : ℕ) + (herd / 8 : ℕ) + 9 = herd → herd = 24 := by
  sorry

end NUMINAMATH_CALUDE_farmer_cow_division_l86_8679


namespace NUMINAMATH_CALUDE_minimal_hexahedron_volume_l86_8609

/-- A trihedral angle -/
structure TrihedralAngle where
  planarAngle : ℝ

/-- The configuration of two trihedral angles -/
structure TrihedralAngleConfiguration where
  angle1 : TrihedralAngle
  angle2 : TrihedralAngle
  vertexDistance : ℝ
  isEquidistant : Bool

/-- The volume of the hexahedron bounded by the faces of two trihedral angles -/
def hexahedronVolume (config : TrihedralAngleConfiguration) : ℝ := sorry

/-- The theorem stating the minimal volume of the hexahedron -/
theorem minimal_hexahedron_volume 
  (config : TrihedralAngleConfiguration) 
  (h1 : config.angle1.planarAngle = π/3) 
  (h2 : config.angle2.planarAngle = π/2)
  (h3 : config.isEquidistant = true) :
  hexahedronVolume config = (config.vertexDistance^3 * Real.sqrt 3) / 20 := by
  sorry


end NUMINAMATH_CALUDE_minimal_hexahedron_volume_l86_8609


namespace NUMINAMATH_CALUDE_jessy_reading_plan_l86_8699

/-- The number of pages Jessy initially plans to read each time -/
def pages_per_reading : ℕ := sorry

/-- The total number of pages in the book -/
def total_pages : ℕ := 140

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of times Jessy reads per day -/
def readings_per_day : ℕ := 3

/-- The additional pages Jessy needs to read per day to achieve her goal -/
def additional_pages_per_day : ℕ := 2

theorem jessy_reading_plan :
  pages_per_reading = 6 ∧
  days_in_week * readings_per_day * pages_per_reading + 
  days_in_week * additional_pages_per_day = total_pages := by
  sorry

end NUMINAMATH_CALUDE_jessy_reading_plan_l86_8699


namespace NUMINAMATH_CALUDE_beta_value_l86_8673

theorem beta_value (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = Real.sqrt 5 / 5)
  (h_sin_α_β : Real.sin (α - β) = -(Real.sqrt 10) / 10) : β = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_beta_value_l86_8673


namespace NUMINAMATH_CALUDE_derivative_f_zero_dne_l86_8682

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 6 * x + x * Real.sin (1 / x) else 0

theorem derivative_f_zero_dne :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < |h| → |h| < δ →
    |((f (0 + h) - f 0) / h) - L| < ε :=
sorry

end NUMINAMATH_CALUDE_derivative_f_zero_dne_l86_8682


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l86_8688

/-- Given a complex number z satisfying z(1+i) = 2, prove that z has a positive real part and a negative imaginary part. -/
theorem z_in_fourth_quadrant (z : ℂ) (h : z * (1 + Complex.I) = 2) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l86_8688


namespace NUMINAMATH_CALUDE_time_per_bone_l86_8681

/-- Proves that analyzing 206 bones in 206 hours with equal time per bone results in 1 hour per bone -/
theorem time_per_bone (total_time : ℕ) (num_bones : ℕ) (time_per_bone : ℚ) :
  total_time = 206 →
  num_bones = 206 →
  time_per_bone = total_time / num_bones →
  time_per_bone = 1 := by
  sorry

#check time_per_bone

end NUMINAMATH_CALUDE_time_per_bone_l86_8681


namespace NUMINAMATH_CALUDE_two_lines_exist_l86_8625

/-- A parabola defined by the equation y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- The point P(2,4) -/
def P : ℝ × ℝ := (2, 4)

/-- A line that has exactly one common point with the parabola -/
def SingleIntersectionLine (l : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ l ∩ Parabola

/-- A line that passes through point P -/
def LineThroughP (l : Set (ℝ × ℝ)) : Prop :=
  P ∈ l

/-- The theorem stating that there are exactly two lines satisfying the conditions -/
theorem two_lines_exist : 
  ∃! (l1 l2 : Set (ℝ × ℝ)), 
    l1 ≠ l2 ∧ 
    LineThroughP l1 ∧ 
    LineThroughP l2 ∧ 
    SingleIntersectionLine l1 ∧ 
    SingleIntersectionLine l2 ∧
    ∀ l, LineThroughP l ∧ SingleIntersectionLine l → l = l1 ∨ l = l2 :=
sorry

end NUMINAMATH_CALUDE_two_lines_exist_l86_8625


namespace NUMINAMATH_CALUDE_function_bound_l86_8687

def function_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) - f x = 2 * x + 1

def bounded_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → |f x| ≤ 1

theorem function_bound (f : ℝ → ℝ) 
  (h1 : function_property f) 
  (h2 : bounded_on_unit_interval f) : 
  ∀ x : ℝ, |f x| ≤ 2 + x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l86_8687


namespace NUMINAMATH_CALUDE_quilt_shaded_half_l86_8634

/-- Represents a square quilt composed of unit squares -/
structure Quilt :=
  (size : ℕ)
  (shaded_rows : ℕ)

/-- The fraction of the quilt that is shaded -/
def shaded_fraction (q : Quilt) : ℚ :=
  q.shaded_rows / q.size

/-- Theorem: For a 4x4 quilt with 2 shaded rows, the shaded fraction is 1/2 -/
theorem quilt_shaded_half (q : Quilt) 
  (h1 : q.size = 4) 
  (h2 : q.shaded_rows = 2) : 
  shaded_fraction q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quilt_shaded_half_l86_8634


namespace NUMINAMATH_CALUDE_segment_sum_is_132_div_7_l86_8667

/-- Represents an acute triangle with two altitudes dividing its sides. -/
structure AcuteTriangleWithAltitudes where
  /-- Length of the first known segment -/
  a : ℝ
  /-- Length of the second known segment -/
  b : ℝ
  /-- Length of the third known segment -/
  c : ℝ
  /-- Length of the unknown segment -/
  y : ℝ
  /-- Condition that all segment lengths are positive -/
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hy : y > 0
  /-- Condition that the triangle is acute -/
  acute : True  -- We don't have enough information to express this condition precisely

/-- The sum of all segments on the sides of the triangle cut by the altitudes -/
def segmentSum (t : AcuteTriangleWithAltitudes) : ℝ :=
  t.a + t.b + t.c + t.y

/-- Theorem stating that for a triangle with segments 7, 4, 5, and y, the sum is 132/7 -/
theorem segment_sum_is_132_div_7 (t : AcuteTriangleWithAltitudes)
  (h1 : t.a = 7) (h2 : t.b = 4) (h3 : t.c = 5) :
  segmentSum t = 132 / 7 := by
  sorry

end NUMINAMATH_CALUDE_segment_sum_is_132_div_7_l86_8667


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l86_8680

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 3 + 56 / 99) ∧ (x = 353 / 99) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l86_8680


namespace NUMINAMATH_CALUDE_smallest_b_value_b_equals_one_l86_8613

def gcd_notation (a b : ℕ) : ℕ := Nat.gcd a b

theorem smallest_b_value (b : ℕ) : b > 0 → (gcd_notation (gcd_notation 16 20) (gcd_notation 18 b) = 2) → b ≥ 1 :=
by
  sorry

theorem b_equals_one : ∃ (b : ℕ), b > 0 ∧ (gcd_notation (gcd_notation 16 20) (gcd_notation 18 b) = 2) ∧ 
  ∀ (c : ℕ), c > 0 → (gcd_notation (gcd_notation 16 20) (gcd_notation 18 c) = 2) → b ≤ c :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_b_value_b_equals_one_l86_8613


namespace NUMINAMATH_CALUDE_function_satisfying_condition_is_zero_function_l86_8689

theorem function_satisfying_condition_is_zero_function 
  (f : ℝ → ℝ) (h : ∀ x y : ℝ, f x + f y = f (f x * f y)) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_condition_is_zero_function_l86_8689


namespace NUMINAMATH_CALUDE_prime_sqrt_sum_integer_l86_8659

theorem prime_sqrt_sum_integer (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ n : ℕ, ∃ m : ℕ, (Nat.sqrt (p + n) + Nat.sqrt n : ℕ) = m :=
sorry

end NUMINAMATH_CALUDE_prime_sqrt_sum_integer_l86_8659


namespace NUMINAMATH_CALUDE_booknote_unique_letters_l86_8642

def word : String := "booknote"

def letter_set : Finset Char := word.toList.toFinset

theorem booknote_unique_letters : Finset.card letter_set = 6 := by
  sorry

end NUMINAMATH_CALUDE_booknote_unique_letters_l86_8642


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l86_8617

theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ p r : ℝ) 
  (h1 : k ≠ 0)
  (h2 : p ≠ 1)
  (h3 : r ≠ 1)
  (h4 : p ≠ r)
  (h5 : a₂ = k * p)
  (h6 : a₃ = k * p^2)
  (h7 : b₂ = k * r)
  (h8 : b₃ = k * r^2)
  (h9 : a₃ - b₃ = 4 * (a₂ - b₂)) :
  p + r = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l86_8617


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l86_8600

/-- The number of friends in the group -/
def total_friends : ℕ := 10

/-- The number of friends who paid -/
def paying_friends : ℕ := 9

/-- The extra amount each paying friend contributed -/
def extra_payment : ℚ := 3

/-- The total bill at the restaurant -/
def total_bill : ℚ := 270

/-- Theorem stating that the given scenario results in the correct total bill -/
theorem restaurant_bill_proof :
  (paying_friends : ℚ) * (total_bill / total_friends + extra_payment) = total_bill :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l86_8600


namespace NUMINAMATH_CALUDE_modified_deck_choose_two_l86_8621

/-- Represents a modified deck of cards -/
structure ModifiedDeck :=
  (normal_suits : Nat)  -- Number of suits with 13 cards
  (reduced_suit : Nat)  -- Number of suits with 12 cards

/-- Calculates the number of ways to choose 2 cards from different suits in a modified deck -/
def choose_two_cards (deck : ModifiedDeck) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem modified_deck_choose_two (d : ModifiedDeck) :
  d.normal_suits = 3 ∧ d.reduced_suit = 1 → choose_two_cards d = 1443 :=
sorry

end NUMINAMATH_CALUDE_modified_deck_choose_two_l86_8621


namespace NUMINAMATH_CALUDE_quadratic_reciprocity_legendre_symbol_two_l86_8604

-- Define the Legendre symbol
noncomputable def legendre_symbol (a p : ℕ) : ℤ := sorry

-- Quadratic reciprocity law
theorem quadratic_reciprocity (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hodd_p : Odd p) (hodd_q : Odd q) :
  legendre_symbol p q * legendre_symbol q p = (-1) ^ ((p - 1) * (q - 1) / 4) := by sorry

-- Legendre symbol of 2
theorem legendre_symbol_two (m : ℕ) (hm : Nat.Prime m) (hodd_m : Odd m) :
  legendre_symbol 2 m = (-1) ^ ((m^2 - 1) / 8) := by sorry

end NUMINAMATH_CALUDE_quadratic_reciprocity_legendre_symbol_two_l86_8604


namespace NUMINAMATH_CALUDE_coprime_iterations_exists_coprime_polynomial_l86_8639

/-- The polynomial f(x) = x^2007 - x^2006 + 1 -/
def f (x : ℤ) : ℤ := x^2007 - x^2006 + 1

/-- The m-th iteration of f -/
def f_iter (m : ℕ) (x : ℤ) : ℤ :=
  match m with
  | 0 => x
  | m+1 => f (f_iter m x)

theorem coprime_iterations (n : ℤ) (m : ℕ) : Int.gcd n (f_iter m n) = 1 := by
  sorry

/-- The main theorem stating that the polynomial f satisfies the required property -/
theorem exists_coprime_polynomial :
  ∃ (f : ℤ → ℤ), (∀ (x : ℤ), ∃ (a b c : ℤ), f x = a * x^2007 + b * x^2006 + c) ∧
                 (∀ (n : ℤ) (m : ℕ), Int.gcd n (f_iter m n) = 1) := by
  sorry

end NUMINAMATH_CALUDE_coprime_iterations_exists_coprime_polynomial_l86_8639


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l86_8693

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

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l86_8693


namespace NUMINAMATH_CALUDE_expression_evaluation_l86_8697

theorem expression_evaluation (a b c : ℝ) 
  (ha : a = 15) (hb : b = 19) (hc : c = 13) : 
  (a^2 * (1/c - 1/b) + b^2 * (1/a - 1/c) + c^2 * (1/b - 1/a)) / 
  (a * (1/c - 1/b) + b * (1/a - 1/c) + c * (1/b - 1/a)) = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l86_8697


namespace NUMINAMATH_CALUDE_f_properties_l86_8653

open Real

noncomputable def f (x : ℝ) := Real.log (Real.exp (2 * x) + 1) - x

theorem f_properties :
  (∀ x, ∃ y, f x = y) ∧
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 ≤ x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l86_8653


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l86_8655

/-- Given two 2D vectors a and b, where a = (2, -1) and b = (-4, x),
    if a and b are parallel, then x = 2. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![-4, x]
  (∃ (k : ℝ), k ≠ 0 ∧ b = k • a) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l86_8655
