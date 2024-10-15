import Mathlib

namespace NUMINAMATH_CALUDE_algebraic_simplification_l374_37448

variable (a b m n : ℝ)

theorem algebraic_simplification :
  (5 * a * b^2 - 2 * a^2 * b + 3 * a * b^2 - a^2 * b - 4 * a * b^2 = 4 * a * b^2 - 3 * a^2 * b) ∧
  (-5 * m * n^2 - (2 * m^2 * n - 2 * (m^2 * n - 2 * m * n^2)) = -9 * m * n^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l374_37448


namespace NUMINAMATH_CALUDE_prob_at_least_one_head_l374_37463

/-- The probability of getting at least one head when tossing five coins,
    each with a 3/4 chance of heads, is 1023/1024. -/
theorem prob_at_least_one_head :
  let n : ℕ := 5  -- number of coins
  let p : ℚ := 3/4  -- probability of heads for each coin
  1 - (1 - p)^n = 1023/1024 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_head_l374_37463


namespace NUMINAMATH_CALUDE_sequence_sum_l374_37491

theorem sequence_sum : 
  let seq1 := [2, 13, 24, 35]
  let seq2 := [8, 18, 28, 38, 48]
  let seq3 := [4, 7]
  (seq1.sum + seq2.sum + seq3.sum) = 225 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l374_37491


namespace NUMINAMATH_CALUDE_remainder_8_pow_305_mod_9_l374_37479

theorem remainder_8_pow_305_mod_9 : 8^305 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_8_pow_305_mod_9_l374_37479


namespace NUMINAMATH_CALUDE_reciprocal_equality_implies_equality_l374_37402

theorem reciprocal_equality_implies_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / a = 1 / b → a = b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equality_implies_equality_l374_37402


namespace NUMINAMATH_CALUDE_digits_making_864n_divisible_by_4_l374_37414

theorem digits_making_864n_divisible_by_4 : 
  ∃! (s : Finset Nat), 
    (∀ n ∈ s, n < 10) ∧ 
    (∀ n, n ∈ s ↔ (864 * 10 + n) % 4 = 0) ∧
    s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_digits_making_864n_divisible_by_4_l374_37414


namespace NUMINAMATH_CALUDE_problem_solution_l374_37405

theorem problem_solution :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 ∧ x ≠ 3 →
    (3*x - 8) / (x - 1) - (x + 1) / x / ((x^2 - 1) / (x^2 - 3*x)) = (2*x - 5) / (x - 1)) ∧
  ((Real.sqrt 12 - (-1/2)⁻¹ - |Real.sqrt 3 + 3| + (2023 - Real.pi)^0) = Real.sqrt 3) ∧
  ((2*2 - 5) / (2 - 1) = -1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l374_37405


namespace NUMINAMATH_CALUDE_center_cell_value_l374_37465

theorem center_cell_value (a b c d e f g h i : ℝ) : 
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ (f > 0) ∧ (g > 0) ∧ (h > 0) ∧ (i > 0) →
  (a * b * c = 1) ∧ (d * e * f = 1) ∧ (g * h * i = 1) ∧
  (a * d * g = 1) ∧ (b * e * h = 1) ∧ (c * f * i = 1) →
  (a * b * d * e = 2) ∧ (b * c * e * f = 2) ∧ (d * e * g * h = 2) ∧ (e * f * h * i = 2) →
  e = 1 :=
by sorry

end NUMINAMATH_CALUDE_center_cell_value_l374_37465


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_l374_37425

theorem shaded_area_rectangle (length width : ℝ) (h1 : length = 8) (h2 : width = 4) : 
  let rectangle_area := length * width
  let triangle_area := (1 / 2) * length * width
  let shaded_area := rectangle_area - triangle_area
  shaded_area = 16 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_l374_37425


namespace NUMINAMATH_CALUDE_mean_of_first_set_l374_37404

def first_set (x : ℝ) : List ℝ := [28, x, 70, 88, 104]
def second_set (x : ℝ) : List ℝ := [50, 62, 97, 124, x]

theorem mean_of_first_set :
  ∀ x : ℝ,
  (List.sum (second_set x)) / 5 = 75.6 →
  (List.sum (first_set x)) / 5 = 67 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_of_first_set_l374_37404


namespace NUMINAMATH_CALUDE_inequality_proof_l374_37498

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ (1/2) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l374_37498


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l374_37415

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  ‖a‖ = 1 →
  b = (Real.sqrt 3, 1) →
  a • b = 0 →
  ‖2 • a + b‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l374_37415


namespace NUMINAMATH_CALUDE_divides_or_divides_l374_37466

theorem divides_or_divides (m n : ℤ) (h : m.lcm n + m.gcd n = m + n) :
  n ∣ m ∨ m ∣ n := by sorry

end NUMINAMATH_CALUDE_divides_or_divides_l374_37466


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l374_37459

/-- Given a circle with area X and circumference Y, if X/Y = 10, then the radius is 20 -/
theorem circle_radius_from_area_circumference_ratio (X Y : ℝ) (h1 : X > 0) (h2 : Y > 0) (h3 : X / Y = 10) :
  ∃ r : ℝ, r > 0 ∧ X = π * r^2 ∧ Y = 2 * π * r ∧ r = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l374_37459


namespace NUMINAMATH_CALUDE_point_coordinates_l374_37410

/-- A point in the coordinate plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the coordinate plane. -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance from a point to the x-axis. -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis. -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: If a point P is in the second quadrant, its distance to the x-axis is 4,
    and its distance to the y-axis is 5, then its coordinates are (-5, 4). -/
theorem point_coordinates (P : Point) 
    (h1 : SecondQuadrant P) 
    (h2 : DistanceToXAxis P = 4) 
    (h3 : DistanceToYAxis P = 5) : 
    P.x = -5 ∧ P.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l374_37410


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l374_37494

def is_factor (a b : ℕ) : Prop := b % a = 0

def are_non_consecutive (a b : ℕ) : Prop := b > a + 1

theorem smallest_non_factor_product (x y : ℕ) : 
  x ≠ y →
  x > 0 →
  y > 0 →
  is_factor x 48 →
  is_factor y 48 →
  are_non_consecutive x y →
  ¬(is_factor (x * y) 48) →
  ∀ (a b : ℕ), a ≠ b → a > 0 → b > 0 → 
    is_factor a 48 → is_factor b 48 → 
    are_non_consecutive a b → ¬(is_factor (a * b) 48) →
    x * y ≤ a * b →
  x * y = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l374_37494


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l374_37490

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2 else 3^(x + 1)

theorem f_composition_negative_two : f (f (-2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l374_37490


namespace NUMINAMATH_CALUDE_unique_integer_triangle_l374_37434

/-- A triangle with integer sides and an altitude --/
structure IntegerTriangle where
  a : ℕ  -- side BC
  b : ℕ  -- side CA
  c : ℕ  -- side AB
  h : ℕ  -- altitude AD
  bd : ℕ -- length of BD
  dc : ℕ -- length of DC

/-- The triangle satisfies the given conditions --/
def satisfies_conditions (t : IntegerTriangle) : Prop :=
  ∃ (n : ℕ), t.h = n ∧ t.a = n + 1 ∧ t.b = n + 2 ∧ t.c = n + 3 ∧
  t.a^2 = t.bd^2 + t.h^2 ∧
  t.c^2 = (t.bd + t.dc)^2 + t.h^2

/-- The theorem stating the existence and uniqueness of the triangle --/
theorem unique_integer_triangle :
  ∃! (t : IntegerTriangle), satisfies_conditions t ∧ 
    t.a = 14 ∧ t.b = 13 ∧ t.c = 15 ∧ t.h = 12 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_triangle_l374_37434


namespace NUMINAMATH_CALUDE_remainder_of_9876543210_mod_101_l374_37478

theorem remainder_of_9876543210_mod_101 : 9876543210 % 101 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_9876543210_mod_101_l374_37478


namespace NUMINAMATH_CALUDE_max_visible_cuboids_6x6x6_l374_37447

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube composed of smaller cuboids -/
structure CompositeCube where
  side_length : ℕ
  small_cuboid : Cuboid
  num_small_cuboids : ℕ

/-- Function to calculate the maximum number of visible small cuboids -/
def max_visible_cuboids (cube : CompositeCube) : ℕ :=
  sorry

/-- Theorem stating the maximum number of visible small cuboids for the given problem -/
theorem max_visible_cuboids_6x6x6 :
  let small_cuboid : Cuboid := ⟨3, 2, 1⟩
  let large_cube : CompositeCube := ⟨6, small_cuboid, 36⟩
  max_visible_cuboids large_cube = 31 :=
by sorry

end NUMINAMATH_CALUDE_max_visible_cuboids_6x6x6_l374_37447


namespace NUMINAMATH_CALUDE_deposit_calculation_l374_37418

theorem deposit_calculation (total_cost remaining_amount : ℝ) 
  (h1 : total_cost = 550)
  (h2 : remaining_amount = 495)
  (h3 : remaining_amount = total_cost - 0.1 * total_cost) :
  0.1 * total_cost = 55 := by
  sorry

end NUMINAMATH_CALUDE_deposit_calculation_l374_37418


namespace NUMINAMATH_CALUDE_perfect_square_condition_l374_37438

theorem perfect_square_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ z : ℝ, x^2 + 2*x*y + y^2 - a*(x + y) + 25 = z^2) → 
  (a = 10 ∨ a = -10) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l374_37438


namespace NUMINAMATH_CALUDE_incorrect_inference_l374_37477

-- Define the types for our geometric objects
variable (Point : Type)
variable (Line : Type)
variable (Plane : Type)

-- Define the relationships between geometric objects
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)
variable (line_on_plane : Line → Plane → Prop)

-- State the theorem
theorem incorrect_inference
  (l : Line) (α : Plane) (A : Point) :
  ¬(∀ (l : Line) (α : Plane) (A : Point),
    (¬ line_on_plane l α ∧ on_line A l) → ¬ on_plane A α) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_inference_l374_37477


namespace NUMINAMATH_CALUDE_binary_1101100_eq_108_l374_37496

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1101100₂ -/
def binary_1101100 : List Bool := [false, false, true, true, false, true, true]

/-- Theorem stating that 1101100₂ is equal to 108 in decimal -/
theorem binary_1101100_eq_108 : binary_to_decimal binary_1101100 = 108 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101100_eq_108_l374_37496


namespace NUMINAMATH_CALUDE_king_crown_cost_l374_37470

/-- Calculates the total cost of a purchase with a tip -/
def totalCostWithTip (originalCost tipPercentage : ℚ) : ℚ :=
  originalCost * (1 + tipPercentage / 100)

/-- Proves that the king pays $22,000 for a $20,000 crown with a 10% tip -/
theorem king_crown_cost :
  totalCostWithTip 20000 10 = 22000 := by
  sorry

end NUMINAMATH_CALUDE_king_crown_cost_l374_37470


namespace NUMINAMATH_CALUDE_charlie_age_when_jenny_thrice_bobby_l374_37475

/-- 
Given:
- Jenny is older than Charlie by 12 years
- Charlie is older than Bobby by 7 years

Prove that Charlie is 18 years old when Jenny's age is three times Bobby's age.
-/
theorem charlie_age_when_jenny_thrice_bobby (jenny charlie bobby : ℕ) 
  (h1 : jenny = charlie + 12)
  (h2 : charlie = bobby + 7) :
  (jenny = 3 * bobby) → charlie = 18 := by
  sorry

end NUMINAMATH_CALUDE_charlie_age_when_jenny_thrice_bobby_l374_37475


namespace NUMINAMATH_CALUDE_tan_sum_difference_pi_fourth_l374_37468

theorem tan_sum_difference_pi_fourth (a : ℝ) : 
  Real.tan (a + π/4) - Real.tan (a - π/4) = 2 * Real.tan (2*a) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_difference_pi_fourth_l374_37468


namespace NUMINAMATH_CALUDE_lines_not_parallel_l374_37419

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem lines_not_parallel : 
  let m : ℝ := -1
  let l1 : Line := { a := 1, b := m, c := 6 }
  let l2 : Line := { a := m - 2, b := 3, c := 2 * m }
  ¬ parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_lines_not_parallel_l374_37419


namespace NUMINAMATH_CALUDE_correct_weight_proof_l374_37403

/-- Proves that the correct weight is 65 kg given the problem conditions -/
theorem correct_weight_proof (n : ℕ) (initial_avg : ℝ) (misread_weight : ℝ) (correct_avg : ℝ) :
  n = 20 ∧ initial_avg = 58.4 ∧ misread_weight = 56 ∧ correct_avg = 58.85 →
  ∃ (correct_weight : ℝ),
    correct_weight = 65 ∧
    n * correct_avg = (n * initial_avg - misread_weight + correct_weight) :=
by sorry


end NUMINAMATH_CALUDE_correct_weight_proof_l374_37403


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l374_37436

/-- Given three circles with radii r₁, r₂, r₃, where r₁ is the largest,
    the radius of the circle inscribed in the quadrilateral formed by
    the tangents as described in the problem is
    (r₁ * r₂ * r₃) / (r₁ * r₃ + r₁ * r₂ - r₂ * r₃). -/
theorem inscribed_circle_radius
  (r₁ r₂ r₃ : ℝ)
  (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₃ > 0)
  (h₄ : r₁ > r₂) (h₅ : r₁ > r₃) :
  ∃ (r : ℝ), r = (r₁ * r₂ * r₃) / (r₁ * r₃ + r₁ * r₂ - r₂ * r₃) ∧
  r > 0 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l374_37436


namespace NUMINAMATH_CALUDE_square_of_difference_l374_37484

theorem square_of_difference (x : ℝ) : (x - 1)^2 = x^2 + 1 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l374_37484


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_sum_less_than_nineteen_nineteen_is_smallest_l374_37409

theorem smallest_whole_number_above_sum : ℕ → Prop :=
  fun n => (∃ (m : ℕ), m > n ∧ 
    (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/6 : ℚ) + (6 + 1/12 : ℚ) < m) ∧
    (∀ (k : ℕ), k < n → 
    (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/6 : ℚ) + (6 + 1/12 : ℚ) ≥ k)

theorem sum_less_than_nineteen :
  (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/6 : ℚ) + (6 + 1/12 : ℚ) < 19 :=
by sorry

theorem nineteen_is_smallest : smallest_whole_number_above_sum 19 :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_sum_less_than_nineteen_nineteen_is_smallest_l374_37409


namespace NUMINAMATH_CALUDE_average_salary_feb_to_may_l374_37445

def average_salary_jan_to_apr : ℝ := 8000
def salary_may : ℝ := 6500
def salary_jan : ℝ := 5700

theorem average_salary_feb_to_may :
  let total_jan_to_apr := average_salary_jan_to_apr * 4
  let total_feb_to_apr := total_jan_to_apr - salary_jan
  let total_feb_to_may := total_feb_to_apr + salary_may
  (total_feb_to_may / 4 : ℝ) = 8200 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_feb_to_may_l374_37445


namespace NUMINAMATH_CALUDE_cafe_staff_remaining_l374_37408

/-- Calculates the total number of remaining staff given the initial numbers and dropouts. -/
def remaining_staff (initial_chefs initial_waiters chefs_dropout waiters_dropout : ℕ) : ℕ :=
  (initial_chefs - chefs_dropout) + (initial_waiters - waiters_dropout)

/-- Theorem stating that given the specific numbers in the problem, the total remaining staff is 23. -/
theorem cafe_staff_remaining :
  remaining_staff 16 16 6 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_cafe_staff_remaining_l374_37408


namespace NUMINAMATH_CALUDE_front_wheel_revolutions_l374_37416

/-- Given a front wheel with perimeter 30 and a back wheel with perimeter 20,
    if the back wheel revolves 360 times, then the front wheel revolves 240 times. -/
theorem front_wheel_revolutions
  (front_perimeter : ℕ) (back_perimeter : ℕ) (back_revolutions : ℕ)
  (h1 : front_perimeter = 30)
  (h2 : back_perimeter = 20)
  (h3 : back_revolutions = 360) :
  (back_perimeter * back_revolutions) / front_perimeter = 240 := by
  sorry

end NUMINAMATH_CALUDE_front_wheel_revolutions_l374_37416


namespace NUMINAMATH_CALUDE_triangle_measure_l374_37427

/-- Given an equilateral triangle with side length 7.5 meters, 
    prove that three times the square of the side length is 168.75 meters. -/
theorem triangle_measure (side_length : ℝ) : 
  side_length = 7.5 → 3 * (side_length ^ 2) = 168.75 := by
  sorry

#check triangle_measure

end NUMINAMATH_CALUDE_triangle_measure_l374_37427


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l374_37435

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifteenth_term_of_sequence : 
  arithmetic_sequence 3 3 15 = 45 := by
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l374_37435


namespace NUMINAMATH_CALUDE_bluegrass_percentage_in_mixtureX_l374_37413

-- Define the seed mixtures and their compositions
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

-- Define the given conditions
def mixtureX : SeedMixture := { ryegrass := 40, bluegrass := 0, fescue := 0 }
def mixtureY : SeedMixture := { ryegrass := 25, bluegrass := 0, fescue := 75 }

-- Define the combined mixture
def combinedMixture : SeedMixture := { ryegrass := 38, bluegrass := 0, fescue := 0 }

-- Weight percentage of mixture X in the combined mixture
def weightPercentageX : ℝ := 86.67

-- Theorem to prove
theorem bluegrass_percentage_in_mixtureX : mixtureX.bluegrass = 60 := by
  sorry

end NUMINAMATH_CALUDE_bluegrass_percentage_in_mixtureX_l374_37413


namespace NUMINAMATH_CALUDE_fraction_addition_theorem_l374_37482

theorem fraction_addition_theorem : (5 * 8) / 10 + 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_theorem_l374_37482


namespace NUMINAMATH_CALUDE_orange_ring_weight_l374_37455

/-- Given the weights of three rings (purple, white, and orange) that sum to a total weight,
    prove that the weight of the orange ring is equal to the total weight minus the sum of
    the purple and white ring weights. -/
theorem orange_ring_weight
  (purple_weight white_weight total_weight : ℚ)
  (h1 : purple_weight = 0.3333333333333333)
  (h2 : white_weight = 0.4166666666666667)
  (h3 : total_weight = 0.8333333333333334)
  (h4 : ∃ orange_weight : ℚ, purple_weight + white_weight + orange_weight = total_weight) :
  ∃ orange_weight : ℚ, orange_weight = total_weight - (purple_weight + white_weight) :=
by
  sorry


end NUMINAMATH_CALUDE_orange_ring_weight_l374_37455


namespace NUMINAMATH_CALUDE_bookman_purchase_theorem_l374_37406

theorem bookman_purchase_theorem (hardback_price : ℕ) (paperback_price : ℕ) 
  (hardback_count : ℕ) (total_sold : ℕ) (remaining_value : ℕ) :
  hardback_price = 20 →
  paperback_price = 10 →
  hardback_count = 10 →
  total_sold = 14 →
  remaining_value = 360 →
  ∃ (total_copies : ℕ),
    total_copies = hardback_count + (remaining_value / paperback_price) + (total_sold - hardback_count) ∧
    total_copies = 50 :=
by sorry

end NUMINAMATH_CALUDE_bookman_purchase_theorem_l374_37406


namespace NUMINAMATH_CALUDE_max_value_of_expression_l374_37412

theorem max_value_of_expression (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (heq : 2*a + 3*b = 5) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 2*x + 3*y = 5 → (2*x + 2)*(3*y + 1) ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l374_37412


namespace NUMINAMATH_CALUDE_m_greater_than_n_l374_37480

theorem m_greater_than_n (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) : 
  a * b > a + b - 1 := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l374_37480


namespace NUMINAMATH_CALUDE_middle_number_is_nine_l374_37497

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem middle_number_is_nine (a b c : ℤ) : 
  is_odd a ∧ is_odd b ∧ is_odd c ∧  -- a, b, c are odd numbers
  b = a + 2 ∧ c = b + 2 ∧            -- a, b, c are consecutive
  a + b + c = a + 20                 -- sum is 20 more than first number
  → b = 9 := by sorry

end NUMINAMATH_CALUDE_middle_number_is_nine_l374_37497


namespace NUMINAMATH_CALUDE_greater_number_problem_l374_37431

theorem greater_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * (x - y) = 12) (h3 : x > y) : x = 22 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l374_37431


namespace NUMINAMATH_CALUDE_jean_jail_time_l374_37474

/-- Calculates the total jail time for Jean based on his charges --/
def total_jail_time (arson_counts : ℕ) (burglary_charges : ℕ) (arson_sentence : ℕ) (burglary_sentence : ℕ) : ℕ :=
  let petty_larceny_charges := 6 * burglary_charges
  let petty_larceny_sentence := burglary_sentence / 3
  arson_counts * arson_sentence + 
  burglary_charges * burglary_sentence + 
  petty_larceny_charges * petty_larceny_sentence

/-- Theorem stating that Jean's total jail time is 216 months --/
theorem jean_jail_time :
  total_jail_time 3 2 36 18 = 216 := by
  sorry

#eval total_jail_time 3 2 36 18

end NUMINAMATH_CALUDE_jean_jail_time_l374_37474


namespace NUMINAMATH_CALUDE_multiply_44_22_l374_37449

theorem multiply_44_22 : 44 * 22 = 88 * 11 := by
  sorry

end NUMINAMATH_CALUDE_multiply_44_22_l374_37449


namespace NUMINAMATH_CALUDE_roes_speed_l374_37486

/-- Proves that Roe's speed is 40 miles per hour given the conditions of the problem -/
theorem roes_speed (teena_speed : ℝ) (initial_distance : ℝ) (time : ℝ) (final_distance : ℝ)
  (h1 : teena_speed = 55)
  (h2 : initial_distance = 7.5)
  (h3 : time = 1.5)
  (h4 : final_distance = 15)
  (h5 : teena_speed * time - initial_distance - final_distance = time * roe_speed) :
  roe_speed = 40 :=
by sorry

end NUMINAMATH_CALUDE_roes_speed_l374_37486


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l374_37458

/-- Represents a quadratic function y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Evaluates the quadratic function at a given x -/
def evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The theorem to be proved -/
theorem quadratic_expression_value (f : QuadraticFunction)
  (h1 : evaluate f (-2) = -2.5)
  (h2 : evaluate f (-1) = -5)
  (h3 : evaluate f 0 = -2.5)
  (h4 : evaluate f 1 = 5)
  (h5 : evaluate f 2 = 17.5) :
  16 * f.a - 4 * f.b + f.c = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l374_37458


namespace NUMINAMATH_CALUDE_remaining_money_l374_37429

def initial_amount : ℚ := 2 * 20 + 2 * 10 + 3 * 5 + 2 * 1 + 4.5
def cake_cost : ℚ := 17.5
def gift_cost : ℚ := 12.7
def donation : ℚ := 5.3

theorem remaining_money :
  initial_amount - (cake_cost + gift_cost + donation) = 46 :=
by sorry

end NUMINAMATH_CALUDE_remaining_money_l374_37429


namespace NUMINAMATH_CALUDE_box_tape_theorem_l374_37460

theorem box_tape_theorem (L S : ℝ) (h1 : L > 0) (h2 : S > 0) :
  5 * (L + 2 * S) + 240 = 540 → S = (60 - L) / 2 := by
  sorry

end NUMINAMATH_CALUDE_box_tape_theorem_l374_37460


namespace NUMINAMATH_CALUDE_expression_evaluation_l374_37423

theorem expression_evaluation : 200 * (200 - 2^3) - (200^2 - 2^4) = -1584 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l374_37423


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l374_37453

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l374_37453


namespace NUMINAMATH_CALUDE_investment_income_is_500_l374_37430

/-- Calculates the yearly income from investments given the total amount,
    amounts invested at different rates, and their corresponding interest rates. -/
def yearly_income (total : ℝ) (amount1 : ℝ) (rate1 : ℝ) (amount2 : ℝ) (rate2 : ℝ) (rate3 : ℝ) : ℝ :=
  amount1 * rate1 + amount2 * rate2 + (total - amount1 - amount2) * rate3

/-- Theorem stating that the yearly income from the given investment scenario is $500 -/
theorem investment_income_is_500 :
  yearly_income 10000 4000 0.05 3500 0.04 0.064 = 500 := by
  sorry

#eval yearly_income 10000 4000 0.05 3500 0.04 0.064

end NUMINAMATH_CALUDE_investment_income_is_500_l374_37430


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l374_37452

/-- A polynomial is a perfect square trinomial if it can be expressed as (ax + b)^2 -/
def is_perfect_square_trinomial (a b m : ℝ) : Prop :=
  ∀ x, x^2 + m*x + 4 = (a*x + b)^2

/-- If x^2 + mx + 4 is a perfect square trinomial, then m = 4 or m = -4 -/
theorem perfect_square_trinomial_m_value (m : ℝ) :
  (∃ a b : ℝ, is_perfect_square_trinomial a b m) → m = 4 ∨ m = -4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l374_37452


namespace NUMINAMATH_CALUDE_binomial_coefficient_arithmetic_sequence_l374_37442

theorem binomial_coefficient_arithmetic_sequence (n : ℕ) : 
  (2 * Nat.choose n 9 = Nat.choose n 8 + Nat.choose n 10) ↔ (n = 14 ∨ n = 23) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_arithmetic_sequence_l374_37442


namespace NUMINAMATH_CALUDE_total_attendance_percentage_l374_37420

/-- Represents the departments in the company -/
inductive Department
  | IT
  | HR
  | Marketing

/-- Represents the genders of employees -/
inductive Gender
  | Male
  | Female

/-- Attendance rate for each department and gender -/
def attendance_rate (d : Department) (g : Gender) : ℝ :=
  match d, g with
  | Department.IT, Gender.Male => 0.25
  | Department.IT, Gender.Female => 0.60
  | Department.HR, Gender.Male => 0.30
  | Department.HR, Gender.Female => 0.50
  | Department.Marketing, Gender.Male => 0.10
  | Department.Marketing, Gender.Female => 0.45

/-- Employee composition for each department and gender -/
def employee_composition (d : Department) (g : Gender) : ℝ :=
  match d, g with
  | Department.IT, Gender.Male => 0.40
  | Department.IT, Gender.Female => 0.25
  | Department.HR, Gender.Male => 0.30
  | Department.HR, Gender.Female => 0.20
  | Department.Marketing, Gender.Male => 0.30
  | Department.Marketing, Gender.Female => 0.55

/-- Calculate the total attendance percentage -/
def total_attendance : ℝ :=
  (attendance_rate Department.IT Gender.Male * employee_composition Department.IT Gender.Male) +
  (attendance_rate Department.IT Gender.Female * employee_composition Department.IT Gender.Female) +
  (attendance_rate Department.HR Gender.Male * employee_composition Department.HR Gender.Male) +
  (attendance_rate Department.HR Gender.Female * employee_composition Department.HR Gender.Female) +
  (attendance_rate Department.Marketing Gender.Male * employee_composition Department.Marketing Gender.Male) +
  (attendance_rate Department.Marketing Gender.Female * employee_composition Department.Marketing Gender.Female)

/-- Theorem: The total attendance percentage is 71.75% -/
theorem total_attendance_percentage : total_attendance = 0.7175 := by
  sorry

end NUMINAMATH_CALUDE_total_attendance_percentage_l374_37420


namespace NUMINAMATH_CALUDE_shane_photos_january_l374_37426

/-- Calculates the number of photos taken per day in January given the total number of photos
    in the first two months and the number of photos taken each week in February. -/
def photos_per_day_january (total_photos : ℕ) (photos_per_week_feb : ℕ) : ℕ :=
  let photos_feb := photos_per_week_feb * 4
  let photos_jan := total_photos - photos_feb
  photos_jan / 31

/-- Theorem stating that given 146 total photos in the first two months and 21 photos
    per week in February, Shane took 2 photos per day in January. -/
theorem shane_photos_january : photos_per_day_january 146 21 = 2 := by
  sorry

#eval photos_per_day_january 146 21

end NUMINAMATH_CALUDE_shane_photos_january_l374_37426


namespace NUMINAMATH_CALUDE_perfect_square_factorization_l374_37451

theorem perfect_square_factorization (a : ℝ) : a^2 - 2*a + 1 = (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factorization_l374_37451


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l374_37417

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l374_37417


namespace NUMINAMATH_CALUDE_infinite_series_sum_l374_37487

/-- The sum of the infinite series ∑_{n=1}^∞ (3^n / (1 + 3^n + 3^{n+1} + 3^{2n+1})) is equal to 1/4 -/
theorem infinite_series_sum : 
  ∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1)) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_infinite_series_sum_l374_37487


namespace NUMINAMATH_CALUDE_circle_ellipse_intersection_l374_37432

-- Define the circle C
def C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 8

-- Define the point D
def D : ℝ × ℝ := (1, 0)

-- Define the ellipse E (trajectory of P)
def E (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the point C
def C_point : ℝ × ℝ := (-1, 0)

-- Define the perpendicular foot W
def W (x0 y0 : ℝ) : Prop := 
  ∃ (k1 k2 : ℝ), 
    (y0 = k1 * (x0 + 1)) ∧ 
    (y0 = k2 * (x0 - 1)) ∧ 
    (k1 * k2 = -1) ∧
    E x0 y0

-- Define the theorem
theorem circle_ellipse_intersection :
  ∀ (x0 y0 : ℝ), W x0 y0 →
    (x0^2 / 2 + y0^2 < 1) ∧
    (∃ (area : ℝ), area = 16/9 ∧ 
      ∀ (q r s t : ℝ × ℝ), 
        E q.1 q.2 → E r.1 r.2 → E s.1 s.2 → E t.1 t.2 →
        q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t →
        area ≤ (abs ((q.1 - s.1) * (r.2 - t.2) - (q.2 - s.2) * (r.1 - t.1))) / 2) :=
sorry

end NUMINAMATH_CALUDE_circle_ellipse_intersection_l374_37432


namespace NUMINAMATH_CALUDE_expression_evaluation_l374_37400

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = b - 4)
  (h2 : b = a + 4)
  (h3 : a = 3)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 10) / (c + 7) = 117 / 40 := by
  sorry

#check expression_evaluation

end NUMINAMATH_CALUDE_expression_evaluation_l374_37400


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l374_37488

/-- The time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 255.03) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30.0024 :=
by sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l374_37488


namespace NUMINAMATH_CALUDE_tank_fill_time_proof_l374_37454

/-- The time (in hours) it takes to fill the tank with the leak -/
def fill_time_with_leak : ℝ := 11

/-- The time (in hours) it takes for the tank to become empty due to the leak -/
def empty_time_due_to_leak : ℝ := 110

/-- The time (in hours) it takes to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 10

theorem tank_fill_time_proof :
  (1 / fill_time_without_leak) - (1 / empty_time_due_to_leak) = (1 / fill_time_with_leak) :=
sorry

end NUMINAMATH_CALUDE_tank_fill_time_proof_l374_37454


namespace NUMINAMATH_CALUDE_first_three_digits_after_decimal_l374_37401

theorem first_three_digits_after_decimal (x : ℝ) : 
  x = (10^2003 + 1)^(12/11) → 
  ∃ (n : ℕ), (x - n) * 1000 ≥ 909 ∧ (x - n) * 1000 < 910 := by
  sorry

end NUMINAMATH_CALUDE_first_three_digits_after_decimal_l374_37401


namespace NUMINAMATH_CALUDE_dvd_packs_after_discount_l374_37472

theorem dvd_packs_after_discount (original_price discount available : ℕ) : 
  original_price = 107 → 
  discount = 106 → 
  available = 93 → 
  (available / (original_price - discount) : ℕ) = 93 := by
sorry

end NUMINAMATH_CALUDE_dvd_packs_after_discount_l374_37472


namespace NUMINAMATH_CALUDE_train_overtake_l374_37461

/-- Proves that Train B overtakes Train A at 285 miles from the station -/
theorem train_overtake (speed_A speed_B : ℝ) (time_diff : ℝ) : 
  speed_A = 30 →
  speed_B = 38 →
  time_diff = 2 →
  speed_B > speed_A →
  (speed_A * time_diff + speed_A * ((speed_B * time_diff) / (speed_B - speed_A))) = 285 := by
  sorry

#check train_overtake

end NUMINAMATH_CALUDE_train_overtake_l374_37461


namespace NUMINAMATH_CALUDE_fraction_simplification_l374_37492

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l374_37492


namespace NUMINAMATH_CALUDE_five_level_pieces_l374_37467

/-- Calculates the number of pieces in a square-based pyramid -/
def pyramid_pieces (levels : ℕ) : ℕ :=
  let rods := levels * (levels + 1) * 2
  let connectors := levels * 4
  rods + connectors

/-- Properties of a two-level square-based pyramid -/
axiom two_level_total : pyramid_pieces 2 = 20
axiom two_level_rods : 2 * (2 + 1) * 2 = 12
axiom two_level_connectors : 2 * 4 = 8

/-- Theorem: A five-level square-based pyramid requires 80 pieces -/
theorem five_level_pieces : pyramid_pieces 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_five_level_pieces_l374_37467


namespace NUMINAMATH_CALUDE_blue_pens_count_l374_37441

theorem blue_pens_count (total_pens : ℕ) (red_pens : ℕ) (black_pens : ℕ) (blue_pens : ℕ) 
  (h1 : total_pens = 31)
  (h2 : total_pens = red_pens + black_pens + blue_pens)
  (h3 : black_pens = red_pens + 5)
  (h4 : blue_pens = 2 * black_pens) :
  blue_pens = 18 := by
  sorry

end NUMINAMATH_CALUDE_blue_pens_count_l374_37441


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l374_37440

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

-- Define the line
def line (x y : ℝ) : Prop := 2*x + y = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l374_37440


namespace NUMINAMATH_CALUDE_cubic_sum_of_roots_similar_triangles_sqrt_difference_l374_37437

-- Problem 1
theorem cubic_sum_of_roots (p q : ℝ) : 
  p^2 - 3*p - 2 = 0 → q^2 - 3*q - 2 = 0 → p^3 + q^3 = 45 := by sorry

-- Problem 2
theorem similar_triangles (A H B C K : ℝ) :
  A - H = 45 → C - K = 36 → B - K = 12 → 
  (A - H) / (C - K) = (B - H) / (B - K) →
  B - H = 15 := by sorry

-- Problem 3
theorem sqrt_difference (x : ℝ) :
  Real.sqrt (2*x + 23) + Real.sqrt (2*x - 1) = 12 →
  Real.sqrt (2*x + 23) - Real.sqrt (2*x - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_of_roots_similar_triangles_sqrt_difference_l374_37437


namespace NUMINAMATH_CALUDE_coin_arrangement_count_l374_37456

def coin_diameter_10_filler : ℕ := 19
def coin_diameter_50_filler : ℕ := 22
def total_length : ℕ := 1000
def min_coins : ℕ := 50

theorem coin_arrangement_count : 
  ∃ (x y : ℕ), 
    x * coin_diameter_10_filler + y * coin_diameter_50_filler = total_length ∧ 
    x + y ≥ min_coins ∧
    Nat.choose (x + y) y = 270725 := by
  sorry

end NUMINAMATH_CALUDE_coin_arrangement_count_l374_37456


namespace NUMINAMATH_CALUDE_hex_B1E_equals_2846_l374_37421

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | '1' => 1
  | 'E' => 14
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.foldr (fun c acc => 16 * acc + hex_to_dec c) 0

/-- The hexadecimal number B1E is equal to 2846 in decimal -/
theorem hex_B1E_equals_2846 : hex_string_to_dec "B1E" = 2846 := by
  sorry

end NUMINAMATH_CALUDE_hex_B1E_equals_2846_l374_37421


namespace NUMINAMATH_CALUDE_polynomial_factorization_l374_37433

theorem polynomial_factorization (p q : ℕ) (n : ℕ) (a : ℤ) :
  Prime p ∧ Prime q ∧ p ≠ q ∧ n ≥ 3 →
  (∃ (g h : Polynomial ℤ),
    (Polynomial.degree g > 0) ∧
    (Polynomial.degree h > 0) ∧
    (X^n + a * X^(n-1) + (p * q : ℤ) = g * h)) ↔
  (a = (-1)^n * (p * q : ℤ) + 1 ∨ a = -(p * q : ℤ) - 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l374_37433


namespace NUMINAMATH_CALUDE_average_weight_of_children_l374_37457

theorem average_weight_of_children (num_boys num_girls : ℕ) (avg_weight_boys avg_weight_girls : ℚ) :
  num_boys = 8 →
  num_girls = 6 →
  avg_weight_boys = 160 →
  avg_weight_girls = 130 →
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 147 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_of_children_l374_37457


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l374_37499

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 3) ↔ x ≥ 3/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l374_37499


namespace NUMINAMATH_CALUDE_age_ratio_change_l374_37446

theorem age_ratio_change (man_age son_age : ℕ) (h1 : man_age = 36) (h2 : son_age = 12) 
  (h3 : man_age = 3 * son_age) : 
  ∃ y : ℕ, man_age + y = 2 * (son_age + y) ∧ y = 12 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_change_l374_37446


namespace NUMINAMATH_CALUDE_power_of_three_mod_ten_l374_37462

theorem power_of_three_mod_ten : 3^19 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_ten_l374_37462


namespace NUMINAMATH_CALUDE_election_vote_count_l374_37439

/-- Given that the ratio of votes for candidate A to candidate B is 2:1,
    and candidate A received 14 votes, prove that the total number of
    votes for both candidates is 21. -/
theorem election_vote_count (votes_A : ℕ) (votes_B : ℕ) : 
  votes_A = 14 → 
  votes_A = 2 * votes_B → 
  votes_A + votes_B = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_election_vote_count_l374_37439


namespace NUMINAMATH_CALUDE_flour_yield_l374_37407

theorem flour_yield (total : ℚ) : 
  (total - (1 / 10) * total = 1) → total = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_flour_yield_l374_37407


namespace NUMINAMATH_CALUDE_equation_represents_line_l374_37469

/-- The equation (2x + 3y - 1)(-1) = 0 represents a single straight line in the Cartesian plane. -/
theorem equation_represents_line : ∃ (a b c : ℝ) (h : (a, b) ≠ (0, 0)),
  ∀ (x y : ℝ), (2*x + 3*y - 1)*(-1) = 0 ↔ a*x + b*y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_equation_represents_line_l374_37469


namespace NUMINAMATH_CALUDE_sticker_collection_probability_l374_37489

theorem sticker_collection_probability : 
  let total_stickers : ℕ := 18
  let selected_stickers : ℕ := 10
  let uncollected_stickers : ℕ := 6
  let collected_stickers : ℕ := 12
  (Nat.choose uncollected_stickers uncollected_stickers * Nat.choose collected_stickers (selected_stickers - uncollected_stickers)) / 
  Nat.choose total_stickers selected_stickers = 5 / 442 := by
sorry

end NUMINAMATH_CALUDE_sticker_collection_probability_l374_37489


namespace NUMINAMATH_CALUDE_expression_equality_l374_37424

theorem expression_equality (x y : ℝ) (h : x - 2*y + 2 = 5) : 2*x - 4*y - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l374_37424


namespace NUMINAMATH_CALUDE_symmetrical_point_y_axis_l374_37485

/-- Represents a point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

theorem symmetrical_point_y_axis :
  let A : Point := ⟨-1, 2⟩
  reflect_y_axis A = ⟨1, 2⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetrical_point_y_axis_l374_37485


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l374_37450

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l374_37450


namespace NUMINAMATH_CALUDE_pencil_distribution_ways_l374_37483

/-- The number of ways to distribute n identical objects among k distinct groups,
    where each group receives at least one object. -/
def distribute_with_minimum (n k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- The number of ways to distribute 8 pencils among 4 friends,
    where each friend receives at least one pencil. -/
def pencil_distribution : ℕ :=
  distribute_with_minimum 8 4

theorem pencil_distribution_ways : pencil_distribution = 35 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_ways_l374_37483


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l374_37464

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = π → A = π * r^2 → d = 2 * r → d = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l374_37464


namespace NUMINAMATH_CALUDE_min_sqrt_equality_characterization_l374_37443

theorem min_sqrt_equality_characterization (a b c : ℝ) 
  (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (min (Real.sqrt ((a * b + 1) / (a * b * c)))
       (min (Real.sqrt ((b * c + 1) / (a * b * c)))
            (Real.sqrt ((a * c + 1) / (a * b * c))))
   = Real.sqrt ((1 - a) / a) + Real.sqrt ((1 - b) / b) + Real.sqrt ((1 - c) / c))
  ↔ ∃ r : ℝ, r > 0 ∧ 
    ((a = 1 / (1 + r^2) ∧ b = 1 / (1 + 1/r^2) ∧ c = (r + 1/r)^2 / (1 + (r + 1/r)^2)) ∨
     (b = 1 / (1 + r^2) ∧ c = 1 / (1 + 1/r^2) ∧ a = (r + 1/r)^2 / (1 + (r + 1/r)^2)) ∨
     (c = 1 / (1 + r^2) ∧ a = 1 / (1 + 1/r^2) ∧ b = (r + 1/r)^2 / (1 + (r + 1/r)^2))) :=
by sorry

end NUMINAMATH_CALUDE_min_sqrt_equality_characterization_l374_37443


namespace NUMINAMATH_CALUDE_min_value_function_l374_37411

theorem min_value_function (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 - x*y + y^2) + Real.sqrt (x^2 - 9*x + 27) + Real.sqrt (y^2 - 15*y + 75) ≥ 7 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l374_37411


namespace NUMINAMATH_CALUDE_valid_seating_count_l374_37471

-- Define the set of people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Eric : Person

-- Define a seating arrangement as a function from position to person
def SeatingArrangement := Fin 5 → Person

-- Define the condition that two people cannot sit next to each other
def NotAdjacent (arr : SeatingArrangement) (p1 p2 : Person) : Prop :=
  ∀ i : Fin 4, arr i ≠ p1 ∨ arr (i + 1) ≠ p2

-- Define a valid seating arrangement
def ValidSeating (arr : SeatingArrangement) : Prop :=
  NotAdjacent arr Person.Alice Person.Bob ∧
  NotAdjacent arr Person.Alice Person.Carla ∧
  NotAdjacent arr Person.Derek Person.Eric ∧
  NotAdjacent arr Person.Derek Person.Carla

-- The theorem to prove
theorem valid_seating_count :
  ∃ (arrangements : Finset SeatingArrangement),
    (∀ arr ∈ arrangements, ValidSeating arr) ∧
    (∀ arr, ValidSeating arr → arr ∈ arrangements) ∧
    arrangements.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_valid_seating_count_l374_37471


namespace NUMINAMATH_CALUDE_sum_of_roots_is_fifteen_l374_37493

/-- A function g: ℝ → ℝ that satisfies g(3+x) = g(3-x) for all real x -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

/-- The theorem stating that if g is symmetric about 3 and has exactly 5 distinct real roots,
    then the sum of these roots is 15 -/
theorem sum_of_roots_is_fifteen
  (g : ℝ → ℝ)
  (h_symmetric : SymmetricAboutThree g)
  (h_five_roots : ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x ∈ s, g x = 0) :
  ∃ (s : Finset ℝ), s.card = 5 ∧ (∀ x ∈ s, g x = 0) ∧ (s.sum id = 15) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_fifteen_l374_37493


namespace NUMINAMATH_CALUDE_franks_fruits_l374_37476

/-- The total number of fruits left after Frank's dog eats some -/
def fruits_left (apples_on_tree apples_on_ground oranges_on_tree oranges_on_ground apples_eaten oranges_eaten : ℕ) : ℕ :=
  (apples_on_tree + apples_on_ground - apples_eaten) + (oranges_on_tree + oranges_on_ground - oranges_eaten)

/-- Theorem stating the total number of fruits left in Frank's scenario -/
theorem franks_fruits :
  fruits_left 5 8 7 10 3 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_franks_fruits_l374_37476


namespace NUMINAMATH_CALUDE_f_max_value_l374_37428

/-- The function f(x) defined as |tx-2| - |tx+1| where t is a real number -/
def f (t : ℝ) (x : ℝ) : ℝ := |t*x - 2| - |t*x + 1|

/-- The maximum value of f(x) is 3 -/
theorem f_max_value (t : ℝ) : 
  ∃ (M : ℝ), M = 3 ∧ ∀ x, f t x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l374_37428


namespace NUMINAMATH_CALUDE_factorial_8_divisors_l374_37422

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- The number of positive divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factorial_8_divisors :
  numDivisors (factorial 8) = 96 := by
  sorry

end NUMINAMATH_CALUDE_factorial_8_divisors_l374_37422


namespace NUMINAMATH_CALUDE_initial_deposit_proof_l374_37495

/-- Calculates the final amount after simple interest --/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the initial deposit was 6200, given the problem conditions --/
theorem initial_deposit_proof (rate : ℝ) : 
  (∃ (principal : ℝ), 
    simpleInterest principal rate 5 = 7200 ∧ 
    simpleInterest principal (rate + 0.03) 5 = 8130) → 
  (∃ (principal : ℝ), 
    simpleInterest principal rate 5 = 7200 ∧ 
    simpleInterest principal (rate + 0.03) 5 = 8130 ∧ 
    principal = 6200) := by
  sorry

#check initial_deposit_proof

end NUMINAMATH_CALUDE_initial_deposit_proof_l374_37495


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l374_37481

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ (F₁ F₂ P : ℝ × ℝ),
    -- F₁ and F₂ are the foci of the hyperbola
    (F₁.1 < 0 ∧ F₁.2 = 0) ∧ 
    (F₂.1 > 0 ∧ F₂.2 = 0) ∧ 
    -- P is on the hyperbola in the first quadrant
    (P.1 > 0 ∧ P.2 > 0) ∧
    (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
    -- P is on the circle with center O and radius OF₁
    (P.1^2 + P.2^2 = F₁.1^2) ∧
    -- The area of triangle PF₁F₂ is a²
    (abs (P.1 * F₂.1 - P.2 * F₂.2) / 2 = a^2) →
    -- The eccentricity is √2
    ((F₂.1 - F₁.1) / 2) / a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l374_37481


namespace NUMINAMATH_CALUDE_shelter_ratio_l374_37444

theorem shelter_ratio (num_cats : ℕ) (num_dogs : ℕ) : 
  num_cats = 45 →
  (num_cats : ℚ) / (num_dogs + 12 : ℚ) = 15 / 11 →
  (num_cats : ℚ) / (num_dogs : ℚ) = 15 / 7 :=
by sorry

end NUMINAMATH_CALUDE_shelter_ratio_l374_37444


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l374_37473

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∃ k : ℕ+, Nat.gcd (13 * k + 4) (8 * k + 3) = 3) ∧
  (∀ n : ℕ+, Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l374_37473
