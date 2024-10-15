import Mathlib

namespace NUMINAMATH_CALUDE_mark_kate_difference_l2217_221704

/-- Represents the project with three workers -/
structure Project where
  kate_hours : ℕ
  pat_hours : ℕ
  mark_hours : ℕ

/-- Conditions of the project -/
def valid_project (p : Project) : Prop :=
  p.pat_hours = 2 * p.kate_hours ∧
  p.mark_hours = p.kate_hours + 6 ∧
  p.kate_hours + p.pat_hours + p.mark_hours = 198

theorem mark_kate_difference (p : Project) (h : valid_project p) :
  p.mark_hours - p.kate_hours = 6 := by
  sorry

end NUMINAMATH_CALUDE_mark_kate_difference_l2217_221704


namespace NUMINAMATH_CALUDE_original_phone_number_proof_l2217_221741

def is_valid_phone_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def first_upgrade (n : ℕ) : ℕ :=
  let d1 := n / 100000
  let rest := n % 100000
  d1 * 1000000 + 800000 + rest

def second_upgrade (n : ℕ) : ℕ :=
  2000000 + n

theorem original_phone_number_proof :
  ∃! n : ℕ, is_valid_phone_number n ∧
    second_upgrade (first_upgrade n) = 81 * n ∧
    n = 282500 :=
sorry

end NUMINAMATH_CALUDE_original_phone_number_proof_l2217_221741


namespace NUMINAMATH_CALUDE_cheerful_team_tasks_l2217_221711

theorem cheerful_team_tasks (correct_points : ℕ) (incorrect_points : ℕ) (total_points : ℤ) (max_tasks : ℕ) :
  correct_points = 9 →
  incorrect_points = 5 →
  total_points = 57 →
  max_tasks = 15 →
  ∃ (x y : ℕ),
    x + y ≤ max_tasks ∧
    (x : ℤ) * correct_points - (y : ℤ) * incorrect_points = total_points ∧
    x = 8 :=
by sorry

end NUMINAMATH_CALUDE_cheerful_team_tasks_l2217_221711


namespace NUMINAMATH_CALUDE_quadratic_root_property_l2217_221735

theorem quadratic_root_property (n : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ + n = 0) → 
  (x₂^2 - 3*x₂ + n = 0) → 
  (x₁ + x₂ - 2 = x₁ * x₂) → 
  n = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l2217_221735


namespace NUMINAMATH_CALUDE_son_work_time_l2217_221763

-- Define the work rates
def man_rate : ℚ := 1 / 10
def combined_rate : ℚ := 1 / 5

-- Define the son's work rate
def son_rate : ℚ := combined_rate - man_rate

-- Theorem to prove
theorem son_work_time :
  son_rate = 1 / 10 ∧ (1 / son_rate : ℚ) = 10 :=
by sorry

end NUMINAMATH_CALUDE_son_work_time_l2217_221763


namespace NUMINAMATH_CALUDE_triangle_problem_l2217_221770

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (2*b - c) * Real.cos A - a * Real.cos C = 0 →
  a = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  A = π/3 ∧ b = 2 ∧ c = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2217_221770


namespace NUMINAMATH_CALUDE_smallest_triple_sum_of_squares_l2217_221703

/-- A function that checks if a number can be expressed as the sum of three squares -/
def isSumOfThreeSquares (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = a^2 + b^2 + c^2

/-- A function that counts the number of ways a number can be expressed as the sum of three squares -/
def countSumOfThreeSquares (n : ℕ) : ℕ :=
  (Finset.filter (fun (triple : ℕ × ℕ × ℕ) => 
    let (a, b, c) := triple
    n = a^2 + b^2 + c^2
  ) (Finset.product (Finset.range (n + 1)) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1))))).card

/-- Theorem stating that 110 is the smallest positive integer that can be expressed as the sum of three squares in at least three different ways -/
theorem smallest_triple_sum_of_squares : 
  (∀ m : ℕ, m < 110 → countSumOfThreeSquares m < 3) ∧ 
  countSumOfThreeSquares 110 ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_triple_sum_of_squares_l2217_221703


namespace NUMINAMATH_CALUDE_always_uninfected_cell_l2217_221723

/-- Represents a square grid of cells -/
structure Grid (n : ℕ) where
  side_length : ℕ
  cells : Fin n → Fin n → Bool

/-- Represents the state of infection in the grid -/
structure InfectionState (n : ℕ) where
  grid : Grid n
  infected_cells : Set (Fin n × Fin n)

/-- A function to determine if a cell can be infected based on its neighbors -/
def can_be_infected (state : InfectionState n) (cell : Fin n × Fin n) : Bool :=
  sorry

/-- The perimeter of the infected region -/
def infected_perimeter (state : InfectionState n) : ℕ :=
  sorry

/-- Theorem stating that there will always be at least one uninfected cell -/
theorem always_uninfected_cell (n : ℕ) (initial_state : InfectionState n) :
  ∃ (cell : Fin n × Fin n), cell ∉ initial_state.infected_cells ∧
    ∀ (final_state : InfectionState n),
      (∀ (c : Fin n × Fin n), c ∉ initial_state.infected_cells →
        (c ∈ final_state.infected_cells → can_be_infected initial_state c)) →
      cell ∉ final_state.infected_cells :=
  sorry

end NUMINAMATH_CALUDE_always_uninfected_cell_l2217_221723


namespace NUMINAMATH_CALUDE_determinant_evaluation_l2217_221769

theorem determinant_evaluation (x : ℝ) : 
  Matrix.det !![x + 2, x - 1, x; x - 1, x + 2, x; x, x, x + 3] = 14 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_determinant_evaluation_l2217_221769


namespace NUMINAMATH_CALUDE_radical_product_simplification_l2217_221702

theorem radical_product_simplification (q : ℝ) (hq : q > 0) :
  2 * Real.sqrt (20 * q) * Real.sqrt (10 * q) * Real.sqrt (15 * q) = 60 * q * Real.sqrt (30 * q) := by
  sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l2217_221702


namespace NUMINAMATH_CALUDE_least_number_of_tiles_l2217_221768

def room_length : ℕ := 672
def room_width : ℕ := 432

theorem least_number_of_tiles (length : ℕ) (width : ℕ) 
  (h1 : length = room_length) (h2 : width = room_width) : 
  ∃ (tile_size : ℕ), tile_size > 0 ∧ 
  length % tile_size = 0 ∧ 
  width % tile_size = 0 ∧
  (length / tile_size) * (width / tile_size) = 126 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_tiles_l2217_221768


namespace NUMINAMATH_CALUDE_straight_line_angle_sum_l2217_221716

-- Define the theorem
theorem straight_line_angle_sum 
  (x y : ℝ) 
  (h1 : x + y = 76)  -- Given condition
  (h2 : 3 * x + 2 * y = 180)  -- Straight line segment condition
  : x = 28 := by
  sorry


end NUMINAMATH_CALUDE_straight_line_angle_sum_l2217_221716


namespace NUMINAMATH_CALUDE_number_problem_l2217_221730

theorem number_problem (x : ℝ) : (x / 6) * 12 = 15 → x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2217_221730


namespace NUMINAMATH_CALUDE_complex_multiplication_opposites_l2217_221736

theorem complex_multiplication_opposites (a : ℝ) (i : ℂ) (h1 : a > 0) (h2 : i * i = -1) :
  (Complex.re (a * i * (a + i)) = -Complex.im (a * i * (a + i))) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_opposites_l2217_221736


namespace NUMINAMATH_CALUDE_sin_240_degrees_l2217_221766

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l2217_221766


namespace NUMINAMATH_CALUDE_simplify_expression_l2217_221729

theorem simplify_expression : (5 + 4 + 6) / 3 - 2 / 3 = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2217_221729


namespace NUMINAMATH_CALUDE_sum_greater_than_product_l2217_221742

theorem sum_greater_than_product (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : Real.arctan x + Real.arctan y + Real.arctan z < π) : 
  x + y + z > x * y * z := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_product_l2217_221742


namespace NUMINAMATH_CALUDE_roots_equation_result_l2217_221762

theorem roots_equation_result (γ δ : ℝ) : 
  γ^2 - 3*γ - 2 = 0 → δ^2 - 3*δ - 2 = 0 → 7*γ^4 + 10*δ^3 = 1363 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_result_l2217_221762


namespace NUMINAMATH_CALUDE_pythagorean_equivalent_l2217_221773

theorem pythagorean_equivalent (t : ℝ) : 
  (∃ (a b : ℚ), (2 * t) / (1 + t^2) = a ∧ (1 - t^2) / (1 + t^2) = b) → 
  ∃ (q : ℚ), (t : ℝ) = q :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_equivalent_l2217_221773


namespace NUMINAMATH_CALUDE_tom_seashells_l2217_221767

/-- The number of broken seashells Tom found -/
def broken_seashells : ℕ := 4

/-- The number of unbroken seashells Tom found -/
def unbroken_seashells : ℕ := 3

/-- The total number of seashells Tom found -/
def total_seashells : ℕ := broken_seashells + unbroken_seashells

theorem tom_seashells : total_seashells = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_l2217_221767


namespace NUMINAMATH_CALUDE_revenue_maximized_at_13_l2217_221795

/-- Revenue function for book sales -/
def R (p : ℝ) : ℝ := p * (130 - 5 * p)

/-- Theorem stating that the revenue is maximized at p = 13 -/
theorem revenue_maximized_at_13 :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 26 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 26 → R p ≥ R q ∧
  p = 13 :=
sorry

end NUMINAMATH_CALUDE_revenue_maximized_at_13_l2217_221795


namespace NUMINAMATH_CALUDE_increasing_sequences_with_divisibility_property_l2217_221748

theorem increasing_sequences_with_divisibility_property :
  ∃ (a b : ℕ → ℕ), 
    (∀ n : ℕ, a n < a (n + 1)) ∧ 
    (∀ n : ℕ, b n < b (n + 1)) ∧
    (∀ n : ℕ, (a n * (a n + 1)) ∣ (b n ^ 2 + 1)) :=
by
  let a : ℕ → ℕ := λ n => (2^(2*n) + 1)^2
  let b : ℕ → ℕ := λ n => 2^(n*(2^(2*n) + 1)) + (2^(2*n) + 1)^2 * (2^(n*(2^(2*n)+1)) - (2^(2*n) + 1))
  sorry

end NUMINAMATH_CALUDE_increasing_sequences_with_divisibility_property_l2217_221748


namespace NUMINAMATH_CALUDE_inverse_of_10_mod_1729_l2217_221732

theorem inverse_of_10_mod_1729 : ∃ x : ℕ, x ≤ 1728 ∧ (10 * x) % 1729 = 1 :=
by
  use 1537
  sorry

end NUMINAMATH_CALUDE_inverse_of_10_mod_1729_l2217_221732


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2217_221765

theorem polynomial_evaluation (x : ℝ) (h : x = 4) : x^3 - x^2 + x - 1 = 51 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2217_221765


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2217_221758

theorem right_triangle_side_length 
  (Q R S : ℝ × ℝ) 
  (right_angle_Q : (R.1 - Q.1) * (S.1 - Q.1) + (R.2 - Q.2) * (S.2 - Q.2) = 0) 
  (cos_R : (Q.1 - R.1) / Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 5/13) 
  (RS_length : Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 13) : 
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2217_221758


namespace NUMINAMATH_CALUDE_morio_age_at_michiko_birth_l2217_221798

/-- Proves that Morio's age when Michiko was born is 38 years old -/
theorem morio_age_at_michiko_birth 
  (teresa_current_age : ℕ) 
  (morio_current_age : ℕ) 
  (teresa_age_at_birth : ℕ) 
  (h1 : teresa_current_age = 59) 
  (h2 : morio_current_age = 71) 
  (h3 : teresa_age_at_birth = 26) :
  morio_current_age - (teresa_current_age - teresa_age_at_birth) = 38 :=
by
  sorry


end NUMINAMATH_CALUDE_morio_age_at_michiko_birth_l2217_221798


namespace NUMINAMATH_CALUDE_gcd_problem_l2217_221792

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 17 * (2 * k + 1)) :
  Nat.gcd (Int.natAbs (4 * b^2 + 63 * b + 144)) (Int.natAbs (2 * b + 7)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2217_221792


namespace NUMINAMATH_CALUDE_rectangle_length_is_three_times_width_l2217_221718

/-- Represents the construction of a large square from six identical smaller squares and a rectangle -/
structure SquareConstruction where
  /-- Side length of each small square -/
  s : ℝ
  /-- Assertion that s is positive -/
  s_pos : 0 < s

/-- The length of the rectangle in the construction -/
def rectangleLength (c : SquareConstruction) : ℝ := 3 * c.s

/-- The width of the rectangle in the construction -/
def rectangleWidth (c : SquareConstruction) : ℝ := c.s

/-- The theorem stating that the length of the rectangle is 3 times its width -/
theorem rectangle_length_is_three_times_width (c : SquareConstruction) :
  rectangleLength c = 3 * rectangleWidth c := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_is_three_times_width_l2217_221718


namespace NUMINAMATH_CALUDE_three_layer_rug_area_l2217_221727

/-- Given three rugs with a total area, floor area covered when overlapped, and area covered by two layers,
    calculate the area covered by three layers. -/
theorem three_layer_rug_area (total_area floor_area two_layer_area : ℝ) 
    (h1 : total_area = 204)
    (h2 : floor_area = 140)
    (h3 : two_layer_area = 24) :
  total_area - floor_area = 64 := by
  sorry

#check three_layer_rug_area

end NUMINAMATH_CALUDE_three_layer_rug_area_l2217_221727


namespace NUMINAMATH_CALUDE_dog_fur_objects_l2217_221796

theorem dog_fur_objects (burrs : ℕ) (ticks : ℕ) : 
  burrs = 12 → ticks = 6 * burrs → burrs + ticks = 84 := by
  sorry

end NUMINAMATH_CALUDE_dog_fur_objects_l2217_221796


namespace NUMINAMATH_CALUDE_symmetry_line_is_correct_l2217_221726

/-- The line of symmetry between two circles -/
def line_of_symmetry (c1 c2 : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  fun p => ∃ (q : ℝ × ℝ), c1 q ∧ c2 (2 * p.1 - q.1, 2 * p.2 - q.2)

/-- First circle: x^2 + y^2 = 9 -/
def circle1 : ℝ × ℝ → Prop :=
  fun p => p.1^2 + p.2^2 = 9

/-- Second circle: x^2 + y^2 - 4x + 4y - 1 = 0 -/
def circle2 : ℝ × ℝ → Prop :=
  fun p => p.1^2 + p.2^2 - 4*p.1 + 4*p.2 - 1 = 0

/-- The equation of the line of symmetry: x - y - 2 = 0 -/
def symmetry_line : ℝ × ℝ → Prop :=
  fun p => p.1 - p.2 - 2 = 0

theorem symmetry_line_is_correct : 
  line_of_symmetry circle1 circle2 = symmetry_line :=
sorry

end NUMINAMATH_CALUDE_symmetry_line_is_correct_l2217_221726


namespace NUMINAMATH_CALUDE_sun_division_l2217_221752

theorem sun_division (x y z : ℝ) (total : ℝ) : 
  (y = 0.45 * x) →
  (z = 0.50 * x) →
  (y = 63) →
  (total = x + y + z) →
  total = 273 := by
sorry

end NUMINAMATH_CALUDE_sun_division_l2217_221752


namespace NUMINAMATH_CALUDE_workshop_production_balance_l2217_221794

/-- Represents the production balance in a workshop --/
theorem workshop_production_balance 
  (total_workers : ℕ) 
  (bolts_per_worker : ℕ) 
  (nuts_per_worker : ℕ) 
  (nuts_per_bolt : ℕ) 
  (x : ℕ) : 
  total_workers = 16 → 
  bolts_per_worker = 1200 → 
  nuts_per_worker = 2000 → 
  nuts_per_bolt = 2 → 
  x ≤ total_workers →
  2 * bolts_per_worker * x = nuts_per_worker * (total_workers - x) := by
  sorry

#check workshop_production_balance

end NUMINAMATH_CALUDE_workshop_production_balance_l2217_221794


namespace NUMINAMATH_CALUDE_quadratic_shift_l2217_221783

/-- The original quadratic function -/
def f (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 - b * x + 3

/-- The mistakenly drawn function -/
def g (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 + b * x + 3

/-- The shifted original function -/
def f_shifted (b : ℝ) (x : ℝ) : ℝ := f b (x + 6)

theorem quadratic_shift (b : ℝ) : 
  (∀ x, g b x = f_shifted b x) → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_l2217_221783


namespace NUMINAMATH_CALUDE_sin_plus_cos_range_l2217_221788

theorem sin_plus_cos_range (A B C : Real) (a b c : Real) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  b^2 = a * c →            -- Given condition
  ∃ (x : Real), 1 < x ∧ x ≤ Real.sqrt 2 ∧ x = Real.sin B + Real.cos B :=
by sorry

end NUMINAMATH_CALUDE_sin_plus_cos_range_l2217_221788


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l2217_221743

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {-3+a, 2*a-1, a^2+1}

theorem set_intersection_and_union (a : ℝ) :
  (A a) ∩ (B a) = {-3} →
  a = -1 ∧ (A a) ∪ (B a) = {-4, -3, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l2217_221743


namespace NUMINAMATH_CALUDE_joans_kittens_l2217_221731

/-- Given that Joan initially had 8 kittens and received 2 more from her friends,
    prove that she now has 10 kittens in total. -/
theorem joans_kittens (initial : Nat) (received : Nat) (total : Nat) : 
  initial = 8 → received = 2 → total = initial + received → total = 10 := by
sorry

end NUMINAMATH_CALUDE_joans_kittens_l2217_221731


namespace NUMINAMATH_CALUDE_division_problem_l2217_221700

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) : 
  dividend = divisor + 2016 →
  quotient = 15 →
  dividend = divisor * quotient →
  dividend = 2160 :=
by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2217_221700


namespace NUMINAMATH_CALUDE_max_value_x_plus_inverse_l2217_221709

theorem max_value_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_plus_inverse_l2217_221709


namespace NUMINAMATH_CALUDE_quadratic_roots_relationship_l2217_221707

theorem quadratic_roots_relationship (m₁ m₂ x₁ x₂ x₃ x₄ : ℝ) :
  (∀ x, m₁ * x^2 + (1/3) * x + 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  (∀ x, m₂ * x^2 + (1/3) * x + 1 = 0 ↔ x = x₃ ∨ x = x₄) →
  x₁ < x₃ →
  x₃ < x₄ →
  x₄ < x₂ →
  x₂ < 0 →
  m₂ > m₁ ∧ m₁ > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relationship_l2217_221707


namespace NUMINAMATH_CALUDE_triangle_perimeter_triangle_perimeter_proof_l2217_221733

/-- Given a triangle with sides of lengths 15 cm, 6 cm, and 12 cm, its perimeter is 33 cm. -/
theorem triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b c : ℝ),
      a = 15 ∧ b = 6 ∧ c = 12 ∧
      perimeter = a + b + c ∧
      perimeter = 33

-- The proof is omitted
theorem triangle_perimeter_proof : triangle_perimeter 33 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_triangle_perimeter_proof_l2217_221733


namespace NUMINAMATH_CALUDE_distance_center_to_line_l2217_221728

/-- The distance from the center of the unit circle to a line ax + by + c = 0, 
    where a^2 + b^2 ≠ 4c^2 and c ≠ 0, is 1/2. -/
theorem distance_center_to_line (a b c : ℝ) 
  (h1 : a^2 + b^2 ≠ 4 * c^2) (h2 : c ≠ 0) : 
  let d := |c| / Real.sqrt (a^2 + b^2)
  d = 1/2 := by sorry

end NUMINAMATH_CALUDE_distance_center_to_line_l2217_221728


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l2217_221710

theorem complex_modulus_equality (t : ℝ) :
  t > 0 → (Complex.abs (8 + 2 * t * Complex.I) = 14 ↔ t = Real.sqrt 33) := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l2217_221710


namespace NUMINAMATH_CALUDE_congruence_remainders_l2217_221724

theorem congruence_remainders (x : ℤ) 
  (h1 : x ≡ 25 [ZMOD 35])
  (h2 : x ≡ 31 [ZMOD 42]) :
  (x ≡ 10 [ZMOD 15]) ∧ (x ≡ 13 [ZMOD 18]) := by
  sorry

end NUMINAMATH_CALUDE_congruence_remainders_l2217_221724


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2217_221782

/-- Proves that for the line 3x + 5y + d = 0, if the sum of its x-intercept and y-intercept is 16, then d = -30 -/
theorem line_intercepts_sum (d : ℝ) : 
  (∃ (x y : ℝ), 3 * x + 5 * y + d = 0 ∧ x + y = 16) → d = -30 := by
sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2217_221782


namespace NUMINAMATH_CALUDE_manuscript_revision_cost_l2217_221781

/-- The cost per page for manuscript revision --/
def revision_cost (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) 
  (initial_cost_per_page : ℚ) (total_cost : ℚ) : ℚ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let initial_typing_cost := total_pages * initial_cost_per_page
  let revision_pages := pages_revised_once + 2 * pages_revised_twice
  (total_cost - initial_typing_cost) / revision_pages

theorem manuscript_revision_cost :
  revision_cost 100 20 30 10 1400 = 5 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_revision_cost_l2217_221781


namespace NUMINAMATH_CALUDE_monotone_increasing_interval_l2217_221737

/-- The function f(x) = ax / (x^2 + 1) is monotonically increasing on (-1, 1) for a > 0 -/
theorem monotone_increasing_interval (a : ℝ) (h : a > 0) :
  StrictMonoOn (fun x => a * x / (x^2 + 1)) (Set.Ioo (-1) 1) := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_interval_l2217_221737


namespace NUMINAMATH_CALUDE_max_value_theorem_l2217_221786

/-- A function that checks if three numbers can form a triangle with non-zero area -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if a set of eleven consecutive integers contains a triangle-forming trio -/
def has_triangle_trio (start : ℕ) : Prop :=
  ∃ i j k, i < j ∧ j < k ∧ k < start + 11 ∧ can_form_triangle (start + i) (start + j) (start + k)

/-- The theorem stating that 499 is the maximum value satisfying the condition -/
theorem max_value_theorem : 
  (∀ start : ℕ, 5 ≤ start ∧ start ≤ 489 → has_triangle_trio start) ∧
  ¬(∀ start : ℕ, 5 ≤ start ∧ start ≤ 490 → has_triangle_trio start) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2217_221786


namespace NUMINAMATH_CALUDE_inequality_proof_l2217_221787

theorem inequality_proof (x y : ℝ) 
  (h : x^2 + x*y + y^2 = (x + y)^2 - x*y ∧ 
       x^2 + x*y + y^2 = (x + y - Real.sqrt (x*y)) * (x + y + Real.sqrt (x*y))) : 
  x + y + Real.sqrt (x*y) ≤ 3*(x + y - Real.sqrt (x*y)) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2217_221787


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2217_221771

/-- Given a hyperbola with asymptotes y = ±(2/3)x and real axis length 12,
    its standard equation is either (x²/36) - (y²/16) = 1 or (y²/36) - (x²/16) = 1 -/
theorem hyperbola_standard_equation
  (asymptote_slope : ℝ)
  (real_axis_length : ℝ)
  (h1 : asymptote_slope = 2/3)
  (h2 : real_axis_length = 12) :
  (∃ (x y : ℝ), x^2/36 - y^2/16 = 1) ∨
  (∃ (x y : ℝ), y^2/36 - x^2/16 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2217_221771


namespace NUMINAMATH_CALUDE_irreducible_polynomial_l2217_221744

/-- A polynomial of the form x^n + 5x^(n-1) + 3 is irreducible over ℤ[X] for any integer n > 1 -/
theorem irreducible_polynomial (n : ℕ) (hn : n > 1) :
  Irreducible (Polynomial.monomial n 1 + Polynomial.monomial (n-1) 5 + Polynomial.monomial 0 3 : Polynomial ℤ) := by
  sorry

end NUMINAMATH_CALUDE_irreducible_polynomial_l2217_221744


namespace NUMINAMATH_CALUDE_quadrilateral_fourth_angle_l2217_221714

theorem quadrilateral_fourth_angle
  (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 = 100)
  (h2 : angle2 = 60)
  (h3 : angle3 = 90)
  (h_sum : angle1 + angle2 + angle3 + angle4 = 360) :
  angle4 = 110 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_fourth_angle_l2217_221714


namespace NUMINAMATH_CALUDE_cherry_price_level_6_l2217_221712

noncomputable def cherryPrice (a b x : ℝ) : ℝ := Real.exp (a * x + b)

theorem cherry_price_level_6 (a b : ℝ) :
  (cherryPrice a b 1 / cherryPrice a b 5 = 3) →
  (cherryPrice a b 3 = 60) →
  ∃ ε > 0, |cherryPrice a b 6 - 170| < ε := by
sorry

end NUMINAMATH_CALUDE_cherry_price_level_6_l2217_221712


namespace NUMINAMATH_CALUDE_sqrt_meaningful_l2217_221778

theorem sqrt_meaningful (x : ℝ) : ∃ y : ℝ, y ^ 2 = x - 3 ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_l2217_221778


namespace NUMINAMATH_CALUDE_larger_number_of_product_18_sum_15_l2217_221755

theorem larger_number_of_product_18_sum_15 (x y : ℝ) : 
  x * y = 18 → x + y = 15 → max x y = 12 := by
sorry

end NUMINAMATH_CALUDE_larger_number_of_product_18_sum_15_l2217_221755


namespace NUMINAMATH_CALUDE_existence_of_m_l2217_221739

/-- n(m) denotes the number of factors of 2 in m! -/
def n (m : ℕ) : ℕ := sorry

theorem existence_of_m : ∃ m : ℕ, m > 1990^1990 ∧ m = 3^1990 + n m := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_l2217_221739


namespace NUMINAMATH_CALUDE_max_odd_numbers_in_even_product_l2217_221706

theorem max_odd_numbers_in_even_product (numbers : Finset ℕ) :
  numbers.card = 7 →
  (numbers.prod (fun x ↦ x)) % 2 = 0 →
  (numbers.filter (fun x ↦ x % 2 = 1)).card ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_odd_numbers_in_even_product_l2217_221706


namespace NUMINAMATH_CALUDE_solution_sets_union_l2217_221745

-- Define the solution sets A and B
def A (p q : ℝ) : Set ℝ := {x | x^2 - (p-1)*x + q = 0}
def B (p q : ℝ) : Set ℝ := {x | x^2 + (q-1)*x + p = 0}

-- State the theorem
theorem solution_sets_union (p q : ℝ) :
  (∃ (p q : ℝ), A p q ∩ B p q = {-2}) →
  (∃ (p q : ℝ), A p q ∪ B p q = {-2, -1, 1}) :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_union_l2217_221745


namespace NUMINAMATH_CALUDE_power_inequality_l2217_221784

theorem power_inequality (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  a^m + a^(-m) > a^n + a^(-n) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2217_221784


namespace NUMINAMATH_CALUDE_unique_function_theorem_l2217_221749

-- Define the property that the function must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y + 1) = x + y + 1

-- State the theorem
theorem unique_function_theorem :
  ∃! f : ℝ → ℝ, SatisfiesProperty f ∧ ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l2217_221749


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l2217_221722

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℝ × ℝ := (-1, -2)

/-- First line equation: 2x + 3y + 8 = 0 -/
def line1 (p : ℝ × ℝ) : Prop :=
  2 * p.1 + 3 * p.2 + 8 = 0

/-- Second line equation: x - y - 1 = 0 -/
def line2 (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 - 1 = 0

/-- Theorem stating that the intersection_point satisfies both line equations
    and is the unique point that does so -/
theorem intersection_point_is_unique :
  line1 intersection_point ∧ 
  line2 intersection_point ∧ 
  ∀ p : ℝ × ℝ, line1 p ∧ line2 p → p = intersection_point :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l2217_221722


namespace NUMINAMATH_CALUDE_sum_of_integers_l2217_221756

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + 2*r = 10)
  (eq2 : q - r + s = 9)
  (eq3 : r - 2*s + p = 6)
  (eq4 : s - p + q = 7) :
  p + q + r + s = 32 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2217_221756


namespace NUMINAMATH_CALUDE_equation_solution_l2217_221753

theorem equation_solution : 
  ∃ x : ℚ, (1 : ℚ) / 3 + 1 / x = 7 / 12 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2217_221753


namespace NUMINAMATH_CALUDE_sqrt_fourth_power_eq_256_l2217_221799

theorem sqrt_fourth_power_eq_256 (y : ℝ) : (Real.sqrt y) ^ 4 = 256 → y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fourth_power_eq_256_l2217_221799


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l2217_221772

/-- The volume of a tetrahedron OABC where:
  - Triangle ABC has sides of length 7, 8, and 9
  - A is on the positive x-axis, B on the positive y-axis, and C on the positive z-axis
  - O is the origin (0, 0, 0)
-/
theorem tetrahedron_volume : ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a^2 + b^2 : ℝ) = 49 ∧
  (b^2 + c^2 : ℝ) = 64 ∧
  (c^2 + a^2 : ℝ) = 81 ∧
  (1/6 : ℝ) * a * b * c = 8 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l2217_221772


namespace NUMINAMATH_CALUDE_count_integer_pairs_verify_count_l2217_221746

/-- The number of positive integer pairs (b,s) satisfying log₄(b²⁰s¹⁹⁰) = 4012 -/
theorem count_integer_pairs : Nat := by sorry

/-- Verifies that the count is correct -/
theorem verify_count : count_integer_pairs = 210 := by sorry

end NUMINAMATH_CALUDE_count_integer_pairs_verify_count_l2217_221746


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l2217_221790

/-- The ratio of upstream to downstream swimming time -/
theorem upstream_downstream_time_ratio 
  (swim_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : swim_speed = 9) 
  (h2 : stream_speed = 3) : 
  (swim_speed - stream_speed)⁻¹ / (swim_speed + stream_speed)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l2217_221790


namespace NUMINAMATH_CALUDE_census_survey_is_D_census_suitability_criterion_l2217_221779

-- Define the survey options
inductive SurveyOption
| A : SurveyOption  -- West Lake Longjing tea quality
| B : SurveyOption  -- Xiaoshan TV station viewership
| C : SurveyOption  -- Xiaoshan people's happiness index
| D : SurveyOption  -- Classmates' health status

-- Define the property of being suitable for a census
def SuitableForCensus (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.D => True
  | _ => False

-- Define the property of having a small quantity of subjects
def HasSmallQuantity (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.D => True
  | _ => False

-- Theorem stating that the survey suitable for a census is option D
theorem census_survey_is_D :
  ∀ (option : SurveyOption),
    SuitableForCensus option ↔ option = SurveyOption.D :=
  sorry

-- Theorem stating that a survey is suitable for a census if and only if it has a small quantity of subjects
theorem census_suitability_criterion :
  ∀ (option : SurveyOption),
    SuitableForCensus option ↔ HasSmallQuantity option :=
  sorry

end NUMINAMATH_CALUDE_census_survey_is_D_census_suitability_criterion_l2217_221779


namespace NUMINAMATH_CALUDE_sector_area_l2217_221705

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (area : ℝ) : 
  arc_length = 3 → central_angle = 1 → area = (arc_length * arc_length) / (2 * central_angle) → area = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2217_221705


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2217_221760

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  ab + bc + 2*c*a ≤ 9/2 ∧ ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧ ab + bc + 2*c*a = 9/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2217_221760


namespace NUMINAMATH_CALUDE_equal_elevation_angles_l2217_221759

/-- Given two flagpoles of heights h and k, separated by 2a units on a horizontal plane,
    this theorem characterizes the set of points where the angles of elevation to the tops
    of the poles are equal. -/
theorem equal_elevation_angles
  (h k a : ℝ) (h_pos : h > 0) (k_pos : k > 0) (a_pos : a > 0) :
  let P : ℝ × ℝ → Prop := λ (x, y) ↦ 
    h / Real.sqrt ((x + a)^2 + y^2) = k / Real.sqrt ((x - a)^2 + y^2)
  (h = k → ∀ y, P (0, y)) ∧ 
  (h ≠ k → ∃ c r, ∀ x y, P (x, y) ↔ (x - c)^2 + y^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_equal_elevation_angles_l2217_221759


namespace NUMINAMATH_CALUDE_twenty_nine_impossible_l2217_221725

/-- Represents the score for a test with 10 questions. -/
structure TestScore where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  sum_is_ten : correct + unanswered + incorrect = 10

/-- Calculates the total score for a given TestScore. -/
def totalScore (ts : TestScore) : Nat :=
  3 * ts.correct + ts.unanswered

/-- Theorem stating that 29 is not a possible total score. -/
theorem twenty_nine_impossible : ¬∃ (ts : TestScore), totalScore ts = 29 := by
  sorry

end NUMINAMATH_CALUDE_twenty_nine_impossible_l2217_221725


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2217_221774

-- Define the universal set U
def U : Set ℝ := {x | x^2 ≤ 4}

-- Define set A
def A : Set ℝ := {x | |x + 1| ≤ 1}

-- State the theorem
theorem complement_of_A_in_U : 
  {x ∈ U | x ∉ A} = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2217_221774


namespace NUMINAMATH_CALUDE_litter_patrol_aluminum_cans_l2217_221764

/-- The number of glass bottles picked up by the Litter Patrol -/
def glass_bottles : ℕ := 10

/-- The total number of pieces of litter picked up by the Litter Patrol -/
def total_litter : ℕ := 18

/-- The number of aluminum cans picked up by the Litter Patrol -/
def aluminum_cans : ℕ := total_litter - glass_bottles

theorem litter_patrol_aluminum_cans : aluminum_cans = 8 := by
  sorry

end NUMINAMATH_CALUDE_litter_patrol_aluminum_cans_l2217_221764


namespace NUMINAMATH_CALUDE_square_field_dimensions_l2217_221701

/-- Proves that a square field with the given fence properties has a side length of 16000 meters -/
theorem square_field_dimensions (x : ℝ) : 
  x > 0 ∧ 
  (1.6 * x = x^2 / 10000) → 
  x = 16000 := by
sorry

end NUMINAMATH_CALUDE_square_field_dimensions_l2217_221701


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2217_221793

theorem absolute_value_inequality_solution_set :
  {x : ℝ | 3 ≤ |5 - 2*x| ∧ |5 - 2*x| < 9} = 
  {x : ℝ | -2 < x ∧ x ≤ 1} ∪ {x : ℝ | 4 ≤ x ∧ x < 7} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2217_221793


namespace NUMINAMATH_CALUDE_solution_set_of_f_less_than_two_range_of_m_when_f_geq_m_squared_l2217_221775

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| + |x|

-- Theorem for part (1)
theorem solution_set_of_f_less_than_two (m : ℝ) (h : f 1 m = 1) :
  {x : ℝ | f x m < 2} = Set.Ioo (-1/2) (3/2) := by sorry

-- Theorem for part (2)
theorem range_of_m_when_f_geq_m_squared :
  {m : ℝ | ∀ x, f x m ≥ m^2} = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_less_than_two_range_of_m_when_f_geq_m_squared_l2217_221775


namespace NUMINAMATH_CALUDE_cod_fish_sold_l2217_221715

theorem cod_fish_sold (total : ℕ) (haddock_percent : ℚ) (halibut_percent : ℚ) 
  (h1 : total = 220)
  (h2 : haddock_percent = 40 / 100)
  (h3 : halibut_percent = 40 / 100) :
  (total : ℚ) * (1 - haddock_percent - halibut_percent) = 44 := by sorry

end NUMINAMATH_CALUDE_cod_fish_sold_l2217_221715


namespace NUMINAMATH_CALUDE_max_common_chord_length_l2217_221717

-- Define the circles
def circle1 (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x + 2*a*y + 2*a^2 - 1 = 0

def circle2 (b : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*b*x + 2*b*y + 2*b^2 - 2 = 0

-- Define the common chord
def commonChord (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | circle1 a p.1 p.2 ∧ circle2 b p.1 p.2}

-- Theorem statement
theorem max_common_chord_length (a b : ℝ) :
  ∃ (l : ℝ), l = 2 ∧ ∀ (p q : ℝ × ℝ), p ∈ commonChord a b → q ∈ commonChord a b →
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ l :=
by sorry

end NUMINAMATH_CALUDE_max_common_chord_length_l2217_221717


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2217_221776

def repeating_decimal (a b : ℕ) : ℚ := (a : ℚ) / (99 : ℚ) + (b : ℚ) / (100 : ℚ)

theorem decimal_to_fraction :
  ∃ (n d : ℕ), d ≠ 0 ∧ 
  repeating_decimal 36 0 = (n : ℚ) / (d : ℚ) ∧
  ∀ (n' d' : ℕ), d' ≠ 0 → repeating_decimal 36 0 = (n' : ℚ) / (d' : ℚ) → d ≤ d' ∧
  d = 11 :=
sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2217_221776


namespace NUMINAMATH_CALUDE_average_difference_approx_l2217_221708

def total_students : ℕ := 180
def total_teachers : ℕ := 6
def class_enrollments : List ℕ := [80, 40, 40, 10, 5, 5]

def teacher_average (students : ℕ) (teachers : ℕ) (enrollments : List ℕ) : ℚ :=
  (enrollments.sum : ℚ) / teachers

def student_average (students : ℕ) (enrollments : List ℕ) : ℚ :=
  (enrollments.map (λ n => n * n)).sum / students

theorem average_difference_approx (ε : ℚ) (hε : ε > 0) :
  ∃ δ : ℚ, δ > 0 ∧ 
    |teacher_average total_students total_teachers class_enrollments - 
     student_average total_students class_enrollments + 24.17| < δ ∧ δ < ε :=
by sorry

end NUMINAMATH_CALUDE_average_difference_approx_l2217_221708


namespace NUMINAMATH_CALUDE_sum_of_solutions_g_l2217_221740

def f (x : ℝ) : ℝ := -x^2 + 10*x - 20

def g : ℝ → ℝ := (f^[2010])

theorem sum_of_solutions_g (h : ∃ (S : Finset ℝ), S.card = 2^2010 ∧ ∀ x ∈ S, g x = 2) :
  ∃ (S : Finset ℝ), S.card = 2^2010 ∧ (∀ x ∈ S, g x = 2) ∧ (S.sum id = 5 * 2^2010) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_g_l2217_221740


namespace NUMINAMATH_CALUDE_train_length_calculation_l2217_221761

theorem train_length_calculation (pole_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  pole_time = 50 →
  platform_length = 500 →
  platform_time = 100 →
  ∃ (train_length : ℝ) (train_speed : ℝ),
    train_length = train_speed * pole_time ∧
    train_length + platform_length = train_speed * platform_time ∧
    train_length = 500 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2217_221761


namespace NUMINAMATH_CALUDE_semicircle_area_comparison_l2217_221719

theorem semicircle_area_comparison : ∀ (short_side long_side : ℝ),
  short_side = 8 →
  long_side = 12 →
  let large_semicircle_area := π * (long_side / 2)^2
  let small_semicircle_area := π * (short_side / 2)^2
  large_semicircle_area = small_semicircle_area * 2.25 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_area_comparison_l2217_221719


namespace NUMINAMATH_CALUDE_max_value_squared_sum_l2217_221751

/-- Given a point P(x,y) satisfying certain conditions, 
    the maximum value of x^2 + y^2 is 18. -/
theorem max_value_squared_sum (x y : ℝ) 
  (h1 : x ≥ 1) 
  (h2 : y ≥ x) 
  (h3 : x - 2*y + 3 ≥ 0) : 
  ∃ (max : ℝ), max = 18 ∧ x^2 + y^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_squared_sum_l2217_221751


namespace NUMINAMATH_CALUDE_newspaper_expense_difference_is_142_l2217_221734

/-- Calculates the difference in annual newspaper expenses between Juanita and Grant -/
def newspaper_expense_difference : ℝ :=
  let grant_base_cost : ℝ := 200
  let grant_discount_rate : ℝ := 0.1
  let juanita_mon_wed_price : ℝ := 0.5
  let juanita_thu_sat_price : ℝ := 0.75
  let juanita_sun_price : ℝ := 2.5
  let juanita_sun_coupon : ℝ := 0.25
  let weeks_per_year : ℕ := 52
  let months_per_year : ℕ := 12

  let grant_annual_cost : ℝ := grant_base_cost * (1 - grant_discount_rate)
  let juanita_mon_wed_annual : ℝ := 3 * juanita_mon_wed_price * weeks_per_year
  let juanita_thu_sat_annual : ℝ := 3 * juanita_thu_sat_price * weeks_per_year
  let juanita_sun_annual : ℝ := juanita_sun_price * weeks_per_year - juanita_sun_coupon * months_per_year
  let juanita_annual_cost : ℝ := juanita_mon_wed_annual + juanita_thu_sat_annual + juanita_sun_annual

  juanita_annual_cost - grant_annual_cost

theorem newspaper_expense_difference_is_142 :
  newspaper_expense_difference = 142 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_expense_difference_is_142_l2217_221734


namespace NUMINAMATH_CALUDE_right_triangle_vector_k_l2217_221747

-- Define a right-angled triangle ABC
structure RightTriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Define the theorem
theorem right_triangle_vector_k (k : ℝ) (triangle : RightTriangleABC) 
  (hBA : triangle.B.1 - triangle.A.1 = k ∧ triangle.B.2 - triangle.A.2 = 1)
  (hBC : triangle.B.1 - triangle.C.1 = 2 ∧ triangle.B.2 - triangle.C.2 = 3) :
  k = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_vector_k_l2217_221747


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2217_221789

theorem polynomial_factorization :
  ∃ (a b c d : ℤ), ∀ (x : ℝ),
    x^4 + x^3 + x^2 + x + 12 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2217_221789


namespace NUMINAMATH_CALUDE_boat_breadth_l2217_221750

/-- Given a boat with the following properties:
  - length of 7 meters
  - sinks by 1 cm when a man gets on it
  - the man's mass is 210 kg
  - the density of water is 1000 kg/m³
  - the acceleration due to gravity is 9.81 m/s²
  Prove that the breadth of the boat is 3 meters. -/
theorem boat_breadth (length : ℝ) (sink_depth : ℝ) (man_mass : ℝ) (water_density : ℝ) (gravity : ℝ) :
  length = 7 →
  sink_depth = 0.01 →
  man_mass = 210 →
  water_density = 1000 →
  gravity = 9.81 →
  ∃ (breadth : ℝ), breadth = 3 ∧ man_mass = (length * breadth * sink_depth) * water_density :=
by sorry

end NUMINAMATH_CALUDE_boat_breadth_l2217_221750


namespace NUMINAMATH_CALUDE_function_composition_fraction_l2217_221797

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem function_composition_fraction :
  f (g (f 3)) / g (f (g 3)) = 59 / 19 := by sorry

end NUMINAMATH_CALUDE_function_composition_fraction_l2217_221797


namespace NUMINAMATH_CALUDE_simplify_2A_3B_value_2A_3B_specific_value_2A_3B_independent_l2217_221721

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3*x^2 - x + 2*y - 4*x*y
def B (x y : ℝ) : ℝ := 2*x^2 - 3*x - y + x*y

-- Theorem 1: Simplification of 2A - 3B
theorem simplify_2A_3B (x y : ℝ) :
  2 * A x y - 3 * B x y = 7*x + 7*y - 11*x*y :=
sorry

-- Theorem 2: Value of 2A - 3B under specific conditions
theorem value_2A_3B_specific (x y : ℝ) (h1 : x + y = 6/7) (h2 : x*y = -1) :
  2 * A x y - 3 * B x y = 17 :=
sorry

-- Theorem 3: Value of 2A - 3B when independent of y
theorem value_2A_3B_independent :
  ∃ (x : ℝ), ∀ (y : ℝ), 2 * A x y - 3 * B x y = 49/11 :=
sorry

end NUMINAMATH_CALUDE_simplify_2A_3B_value_2A_3B_specific_value_2A_3B_independent_l2217_221721


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2217_221713

-- Define the quadratic function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem quadratic_function_properties :
  (f 0 = 0) ∧
  (∀ x, f (x + 2) = f (x + 1) + 2*x + 1) ∧
  (∀ m, (∀ x ∈ Set.Icc (-1) m, f x ∈ Set.Icc (-1) 3) → m ∈ Set.Icc 1 3) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l2217_221713


namespace NUMINAMATH_CALUDE_sum_greater_than_six_random_event_l2217_221754

def numbers : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 10}

def sumGreaterThanSix (a b c : ℕ) : Prop := a + b + c > 6

theorem sum_greater_than_six_random_event :
  ∃ (a b c : ℕ), a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ sumGreaterThanSix a b c ∧
  ∃ (x y z : ℕ), x ∈ numbers ∧ y ∈ numbers ∧ z ∈ numbers ∧ ¬sumGreaterThanSix x y z :=
sorry

end NUMINAMATH_CALUDE_sum_greater_than_six_random_event_l2217_221754


namespace NUMINAMATH_CALUDE_midline_characterization_l2217_221791

/-- Triangle type -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Function to calculate the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Function to check if a point is inside a triangle -/
def is_inside (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Function to check if a point is on the midline of a triangle -/
def on_midline (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem midline_characterization (t : Triangle) (M : ℝ × ℝ) :
  is_inside M t →
  (on_midline M t ↔ area ⟨M, t.A, t.B⟩ = area ⟨M, t.B, t.C⟩ + area ⟨M, t.C, t.A⟩) :=
by sorry

end NUMINAMATH_CALUDE_midline_characterization_l2217_221791


namespace NUMINAMATH_CALUDE_line_through_circle_centers_l2217_221777

/-- Given two circles that pass through (1, 1), prove the equation of the line through their centers -/
theorem line_through_circle_centers (D₁ E₁ D₂ E₂ : ℝ) : 
  (1^2 + 1^2 + D₁*1 + E₁*1 + 3 = 0) →
  (1^2 + 1^2 + D₂*1 + E₂*1 + 3 = 0) →
  ∃ (k : ℝ), ∀ (x y : ℝ), (x = D₁ ∧ y = E₁) ∨ (x = D₂ ∧ y = E₂) → x + y + 5 = k := by
  sorry

#check line_through_circle_centers

end NUMINAMATH_CALUDE_line_through_circle_centers_l2217_221777


namespace NUMINAMATH_CALUDE_problem_solution_l2217_221757

theorem problem_solution (x : ℝ) (f : ℝ → ℝ) 
  (h1 : x > 0) 
  (h2 : x + 17 = 60 * f x) 
  (h3 : x = 3) : 
  f 3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2217_221757


namespace NUMINAMATH_CALUDE_power_of_power_l2217_221720

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2217_221720


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_squared_l2217_221780

/-- A square inscribed in an ellipse with specific properties -/
structure InscribedSquare where
  /-- The ellipse equation: x^2 + 3y^2 = 3 -/
  ellipse : ∀ (x y : ℝ), x^2 + 3 * y^2 = 3 → True
  /-- One vertex of the square is at (0, 1) -/
  vertex : ∃ (v : ℝ × ℝ), v = (0, 1)
  /-- One diagonal of the square lies along the y-axis -/
  diagonal_on_y_axis : ∃ (v1 v2 : ℝ × ℝ), v1.1 = 0 ∧ v2.1 = 0 ∧ v1 ≠ v2

/-- The theorem stating the square of the side length of the inscribed square -/
theorem inscribed_square_side_length_squared (s : InscribedSquare) :
  ∃ (side_length : ℝ), side_length^2 = 5/3 - 2 * Real.sqrt (2/3) :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_squared_l2217_221780


namespace NUMINAMATH_CALUDE_stage_25_l2217_221785

/-- Represents the number of toothpicks in a stage of the triangle pattern. -/
def toothpicks (n : ℕ) : ℕ := 3 * n

/-- The triangle pattern starts with 1 toothpick per side in Stage 1. -/
axiom stage_one : toothpicks 1 = 3

/-- Each stage adds one toothpick to each side of the triangle. -/
axiom stage_increase (n : ℕ) : toothpicks (n + 1) = toothpicks n + 3

/-- The number of toothpicks in the 25th stage is 75. -/
theorem stage_25 : toothpicks 25 = 75 := by sorry

end NUMINAMATH_CALUDE_stage_25_l2217_221785


namespace NUMINAMATH_CALUDE_not_always_equal_to_self_l2217_221738

-- Define the ❤ operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Theorem stating that the statement "x ❤ 0 = x for all x" is false
theorem not_always_equal_to_self : ¬ ∀ x : ℝ, heartsuit x 0 = x := by
  sorry

end NUMINAMATH_CALUDE_not_always_equal_to_self_l2217_221738
