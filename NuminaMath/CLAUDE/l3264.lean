import Mathlib

namespace union_when_m_is_4_intersection_condition_l3264_326454

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem 1: When m = 4, A ∪ B = {x | -2 ≤ x ≤ 7}
theorem union_when_m_is_4 :
  A ∪ B 4 = {x : ℝ | -2 ≤ x ∧ x ≤ 7} := by sorry

-- Theorem 2: B ∩ A = B if and only if m ∈ (-∞, 3]
theorem intersection_condition :
  ∀ m : ℝ, B m ∩ A = B m ↔ m ≤ 3 := by sorry

end union_when_m_is_4_intersection_condition_l3264_326454


namespace triangle_angle_inequality_l3264_326410

theorem triangle_angle_inequality (A B C α : Real) : 
  A + B + C = π →
  A > 0 → B > 0 → C > 0 →
  α = min (2 * A - B) (min (3 * B - 2 * C) (π / 2 - A)) →
  α ≤ 2 * π / 9 := by
  sorry

end triangle_angle_inequality_l3264_326410


namespace cuboid_dimensions_l3264_326468

theorem cuboid_dimensions (x y v : ℕ) 
  (h1 : x * y * v - v = 602)
  (h2 : x * y * v - x = 605)
  (h3 : v = x + 3)
  (hx : x > 0)
  (hy : y > 0)
  (hv : v > 0) :
  x = 11 ∧ y = 4 ∧ v = 14 := by
sorry

end cuboid_dimensions_l3264_326468


namespace equation_proof_l3264_326485

/-- Given a > 0 and -∛(√a) ≤ b < ∛(a³ - √a), prove that A = 1 when
    2.334 A = √(a³-b³+√a) · (√(a³/² + √(b³+√a)) · √(a³/² - √(b³+√a))) / √((a³+b³)² - a(4a²b³+1)) -/
theorem equation_proof (a b A : ℝ) 
  (ha : a > 0) 
  (hb : -Real.rpow a (1/6) ≤ b ∧ b < Real.rpow (a^3 - Real.sqrt a) (1/3)) 
  (heq : 2.334 * A = Real.sqrt (a^3 - b^3 + Real.sqrt a) * 
    (Real.sqrt (Real.sqrt (a^3) + Real.sqrt (b^3 + Real.sqrt a)) * 
     Real.sqrt (Real.sqrt (a^3) - Real.sqrt (b^3 + Real.sqrt a))) / 
    Real.sqrt ((a^3 + b^3)^2 - a * (4 * a^2 * b^3 + 1))) : 
  A = 1 := by
  sorry

end equation_proof_l3264_326485


namespace ellipse_properties_hyperbola_properties_l3264_326445

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Theorem for the ellipse
theorem ellipse_properties :
  ∀ x y : ℝ, ellipse x y →
  (∃ c : ℝ, c = Real.sqrt 2 ∧ 
   ((x - c)^2 + y^2 = 4 ∨ (x + c)^2 + y^2 = 4)) ∧
  (x = -2 * Real.sqrt 2 ∨ x = 2 * Real.sqrt 2) :=
sorry

-- Theorem for the hyperbola
theorem hyperbola_properties :
  ∀ x y : ℝ, hyperbola x y →
  (hyperbola (Real.sqrt 2) 2) ∧
  (∃ k : ℝ, k = 2 ∧ (y = k*x ∨ y = -k*x)) :=
sorry

end ellipse_properties_hyperbola_properties_l3264_326445


namespace xyz_acronym_length_l3264_326458

theorem xyz_acronym_length :
  let straight_segments : ℕ := 6
  let slanted_segments : ℕ := 6
  let straight_length : ℝ := 1
  let slanted_length : ℝ := Real.sqrt 2
  (straight_segments : ℝ) * straight_length + (slanted_segments : ℝ) * slanted_length = 6 + 6 * Real.sqrt 2 := by
  sorry

end xyz_acronym_length_l3264_326458


namespace trapezoid_area_is_147_l3264_326422

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD with intersection point E of diagonals -/
structure Trapezoid :=
  (A B C D E : Point)

/-- The area of a triangle -/
def triangle_area (p1 p2 p3 : Point) : ℝ := sorry

/-- The area of a trapezoid -/
def trapezoid_area (t : Trapezoid) : ℝ := sorry

/-- Theorem: Area of trapezoid ABCD is 147 square units -/
theorem trapezoid_area_is_147 (ABCD : Trapezoid) :
  (ABCD.A.x - ABCD.B.x) * (ABCD.C.y - ABCD.D.y) = (ABCD.C.x - ABCD.D.x) * (ABCD.A.y - ABCD.B.y) →
  triangle_area ABCD.A ABCD.B ABCD.E = 75 →
  triangle_area ABCD.A ABCD.D ABCD.E = 30 →
  trapezoid_area ABCD = 147 := by
  sorry

end trapezoid_area_is_147_l3264_326422


namespace triangle_existence_and_perimeter_l3264_326461

/-- A triangle with sides a, b, and c is valid if it satisfies the triangle inequality theorem -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- The perimeter of a triangle with sides a, b, and c -/
def triangle_perimeter (a b c : ℝ) : ℝ :=
  a + b + c

/-- Theorem: The given lengths form a valid triangle with perimeter 44 -/
theorem triangle_existence_and_perimeter :
  let a := 15
  let b := 11
  let c := 18
  is_valid_triangle a b c ∧ triangle_perimeter a b c = 44 := by sorry

end triangle_existence_and_perimeter_l3264_326461


namespace arithmetic_calculation_l3264_326466

theorem arithmetic_calculation : (-3 + 2) * 3 - (-4) = 1 := by
  sorry

end arithmetic_calculation_l3264_326466


namespace power_division_result_l3264_326470

theorem power_division_result : (3 : ℕ)^12 / 27^2 = 729 := by
  sorry

end power_division_result_l3264_326470


namespace expression_evaluation_l3264_326487

theorem expression_evaluation :
  let x : ℝ := 2 - Real.sqrt 3
  (7 + 4 * Real.sqrt 3) * x^2 - (2 + Real.sqrt 3) * x + Real.sqrt 3 = 2 + Real.sqrt 3 := by
sorry

end expression_evaluation_l3264_326487


namespace sara_payment_l3264_326408

/-- The amount Sara gave to the seller -/
def amount_given (book1_price book2_price change : ℝ) : ℝ :=
  book1_price + book2_price + change

/-- Theorem stating the amount Sara gave to the seller -/
theorem sara_payment (book1_price book2_price change : ℝ) 
  (h1 : book1_price = 5.5)
  (h2 : book2_price = 6.5)
  (h3 : change = 8) :
  amount_given book1_price book2_price change = 20 := by
sorry

end sara_payment_l3264_326408


namespace prob_both_selected_l3264_326495

/-- The probability of both brothers being selected in an exam -/
theorem prob_both_selected (p_x p_y : ℚ) (h_x : p_x = 1/5) (h_y : p_y = 2/3) :
  p_x * p_y = 2/15 := by
  sorry

end prob_both_selected_l3264_326495


namespace parabola_intersection_l3264_326483

theorem parabola_intersection (k α β : ℝ) : 
  (∀ x, x^2 - (k-1)*x - 3*k - 2 = 0 ↔ x = α ∨ x = β) →
  α^2 + β^2 = 17 →
  k = 2 :=
by sorry

end parabola_intersection_l3264_326483


namespace triangle_area_doubles_l3264_326479

theorem triangle_area_doubles (a b θ : ℝ) (ha : a > 0) (hb : b > 0) (hθ : 0 < θ ∧ θ < π) :
  let area := (1 / 2) * a * b * Real.sin θ
  let new_area := (1 / 2) * (2 * a) * b * Real.sin θ
  new_area = 2 * area := by sorry

end triangle_area_doubles_l3264_326479


namespace solve_equation_l3264_326443

theorem solve_equation (y : ℚ) (h : 1/3 - 1/4 = 1/y) : y = 12 := by
  sorry

end solve_equation_l3264_326443


namespace vector_expression_inequality_l3264_326456

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given arbitrary points P, A, B, Q in a real vector space V, 
    the expression PA + AB - BQ is not always equal to PQ. -/
theorem vector_expression_inequality (P A B Q : V) :
  ¬ (∀ (P A B Q : V), (A - P) + (B - A) - (Q - B) = Q - P) :=
sorry

end vector_expression_inequality_l3264_326456


namespace n_value_for_specific_x_and_y_l3264_326423

theorem n_value_for_specific_x_and_y :
  let x : ℕ := 3
  let y : ℕ := 1
  let n : ℤ := x - 3 * y^(x - y) + 1
  n = 1 := by sorry

end n_value_for_specific_x_and_y_l3264_326423


namespace parallel_lines_distance_l3264_326437

/-- Given a circle intersected by three equally spaced parallel lines creating
    chords of lengths 40, 40, and 36, the distance between adjacent lines is √38. -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ), 
    chord1 = 40 ∧ 
    chord2 = 40 ∧ 
    chord3 = 36 ∧ 
    chord1^2 = 4 * (r^2 - (d/2)^2) ∧ 
    chord2^2 = 4 * (r^2 - (3*d/2)^2) ∧ 
    chord3^2 = 4 * (r^2 - d^2)) → 
  d = Real.sqrt 38 := by
sorry

end parallel_lines_distance_l3264_326437


namespace complement_of_A_in_U_l3264_326446

-- Define the sets U and A
def U : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def A : Set ℝ := {x | 3 ≤ 2*x - 1 ∧ 2*x - 1 < 5}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = Set.Icc 0 2 ∪ Set.Ici 3 := by sorry

end complement_of_A_in_U_l3264_326446


namespace equation_solution_l3264_326418

theorem equation_solution : ∃ x : ℚ, (5 * x + 9 * x = 420 - 10 * (x - 4)) ∧ x = 115 / 6 := by
  sorry

end equation_solution_l3264_326418


namespace exponent_problem_l3264_326435

theorem exponent_problem (x m n : ℝ) (hm : x^m = 5) (hn : x^n = -2) : x^(m+2*n) = 20 := by
  sorry

end exponent_problem_l3264_326435


namespace sqrt_sum_quotient_l3264_326429

theorem sqrt_sum_quotient : (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 75 = 12/5 := by
  sorry

end sqrt_sum_quotient_l3264_326429


namespace deer_distribution_l3264_326486

theorem deer_distribution (a₁ : ℚ) (d : ℚ) :
  a₁ = 5/3 ∧ 
  5 * a₁ + (5 * 4)/2 * d = 5 →
  a₁ + 2*d = 1 :=
by sorry

end deer_distribution_l3264_326486


namespace highest_power_of_three_dividing_M_l3264_326480

def M : ℕ := sorry

theorem highest_power_of_three_dividing_M :
  ∃ (j : ℕ), (3^j ∣ M) ∧ ¬(3^(j+1) ∣ M) ∧ j = 1 := by sorry

end highest_power_of_three_dividing_M_l3264_326480


namespace rectangle_area_change_l3264_326476

theorem rectangle_area_change (L W : ℝ) (h1 : L > 0) (h2 : W > 0) : 
  let new_length := 1.4 * L
  let new_width := W / 2
  let original_area := L * W
  let new_area := new_length * new_width
  (new_area / original_area) = 0.7 := by sorry

end rectangle_area_change_l3264_326476


namespace rice_price_reduction_l3264_326465

theorem rice_price_reduction (x : ℝ) (h : x > 0) :
  let original_amount := 30
  let price_reduction_factor := 0.75
  let new_amount := original_amount / price_reduction_factor
  new_amount = 40 := by
sorry

end rice_price_reduction_l3264_326465


namespace square_area_proof_l3264_326439

theorem square_area_proof (x : ℝ) : 
  (5 * x - 20 : ℝ) = (25 - 4 * x : ℝ) → 
  (5 * x - 20 : ℝ) ^ 2 = 25 := by
  sorry

end square_area_proof_l3264_326439


namespace donut_selection_count_l3264_326434

/-- The number of types of donuts available -/
def num_donut_types : ℕ := 3

/-- The number of donuts Pat wants to buy -/
def num_donuts_to_buy : ℕ := 4

/-- The number of ways to select donuts -/
def num_selections : ℕ := (num_donuts_to_buy + num_donut_types - 1).choose (num_donut_types - 1)

theorem donut_selection_count : num_selections = 15 := by
  sorry

end donut_selection_count_l3264_326434


namespace simple_interest_rate_l3264_326493

/-- Given a principal amount P and a time period of 10 years,
    prove that the rate of simple interest is 6% per annum
    when the simple interest is 3/5 of the principal amount. -/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) :
  let SI := (3/5) * P  -- Simple interest is 3/5 of principal
  let T := 10  -- Time period in years
  let r := 6  -- Rate percent per annum
  SI = (P * r * T) / 100  -- Simple interest formula
  := by sorry

end simple_interest_rate_l3264_326493


namespace parallelogram_roots_theorem_l3264_326453

/-- The polynomial in question -/
def polynomial (b : ℝ) (z : ℂ) : ℂ :=
  z^4 - 8*z^3 + 13*b*z^2 - 5*(2*b^2 + 4*b - 4)*z + 4

/-- Predicate to check if four complex numbers form a parallelogram -/
def form_parallelogram (z₁ z₂ z₃ z₄ : ℂ) : Prop :=
  (z₁ + z₃ = z₂ + z₄) ∧ (z₁ - z₂ = z₄ - z₃)

/-- The main theorem -/
theorem parallelogram_roots_theorem :
  ∃! (b : ℝ), b = (3/2) ∧
  ∃ (z₁ z₂ z₃ z₄ : ℂ),
    (polynomial b z₁ = 0) ∧
    (polynomial b z₂ = 0) ∧
    (polynomial b z₃ = 0) ∧
    (polynomial b z₄ = 0) ∧
    form_parallelogram z₁ z₂ z₃ z₄ :=
by
  sorry

end parallelogram_roots_theorem_l3264_326453


namespace partition_six_into_three_l3264_326450

/-- The number of ways to partition a set of n elements into k disjoint subsets -/
def partitionWays (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to partition a set of 6 elements into 3 disjoint subsets is 15 -/
theorem partition_six_into_three : partitionWays 6 3 = 15 := by sorry

end partition_six_into_three_l3264_326450


namespace nicky_dmv_wait_l3264_326488

/-- The time Nicky spent waiting to take a number, in minutes. -/
def initial_wait : ℕ := 20

/-- The time Nicky spent waiting for his number to be called, in minutes. -/
def number_wait : ℕ := 4 * initial_wait + 14

/-- The total time Nicky spent waiting at the DMV, in minutes. -/
def total_wait : ℕ := initial_wait + number_wait

theorem nicky_dmv_wait : total_wait = 114 := by
  sorry

end nicky_dmv_wait_l3264_326488


namespace unique_egyptian_fraction_representation_l3264_326484

theorem unique_egyptian_fraction_representation (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2) :
  ∃! (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (2 : ℚ) / p = 1 / x + 1 / y := by
  sorry

end unique_egyptian_fraction_representation_l3264_326484


namespace f_nonnegative_implies_a_range_l3264_326464

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

-- State the theorem
theorem f_nonnegative_implies_a_range (a b : ℝ) :
  (∀ x ≥ 2, f a b x ≥ 0) → a ∈ Set.Ioo (-9 : ℝ) (-3 : ℝ) :=
by sorry

end f_nonnegative_implies_a_range_l3264_326464


namespace fraction_equation_solution_l3264_326482

theorem fraction_equation_solution (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (2 : ℚ) / 7 = 1 / (a : ℚ) + 1 / (b : ℚ) → a = 28 ∧ b = 4 := by
  sorry

end fraction_equation_solution_l3264_326482


namespace monday_temp_is_43_l3264_326455

/-- Represents the temperatures for each day of the week --/
structure WeekTemperatures where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The theorem stating that Monday's temperature is 43 degrees --/
theorem monday_temp_is_43 (w : WeekTemperatures) 
  (avg_mon_to_thu : (w.monday + w.tuesday + w.wednesday + w.thursday) / 4 = 48)
  (avg_tue_to_fri : (w.tuesday + w.wednesday + w.thursday + w.friday) / 4 = 46)
  (one_day_43 : w.monday = 43 ∨ w.tuesday = 43 ∨ w.wednesday = 43 ∨ w.thursday = 43 ∨ w.friday = 43)
  (friday_35 : w.friday = 35) : 
  w.monday = 43 := by
  sorry


end monday_temp_is_43_l3264_326455


namespace factorization_identities_l3264_326478

theorem factorization_identities :
  (∀ m : ℝ, m^3 - 16*m = m*(m+4)*(m-4)) ∧
  (∀ a x : ℝ, -4*a^2*x + 12*a*x - 9*x = -x*(2*a-3)^2) := by
  sorry

end factorization_identities_l3264_326478


namespace train_length_proof_l3264_326481

/-- The length of a train in meters -/
def train_length : ℝ := 1200

/-- The time in seconds it takes for the train to cross a tree -/
def tree_crossing_time : ℝ := 120

/-- The time in seconds it takes for the train to pass a platform -/
def platform_passing_time : ℝ := 150

/-- The length of the platform in meters -/
def platform_length : ℝ := 300

theorem train_length_proof :
  (train_length / tree_crossing_time = (train_length + platform_length) / platform_passing_time) →
  train_length = 1200 := by
sorry

end train_length_proof_l3264_326481


namespace smallest_prime_with_42_divisors_l3264_326401

-- Define a function to count the number of divisors
def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

-- Define the function F(p) = p^3 + 2p^2 + p
def F (p : ℕ) : ℕ := p^3 + 2*p^2 + p

-- Main theorem
theorem smallest_prime_with_42_divisors :
  ∃ (p : ℕ), Nat.Prime p ∧ 
             count_divisors (F p) = 42 ∧ 
             (∀ q < p, Nat.Prime q → count_divisors (F q) ≠ 42) ∧
             p = 23 := by
  sorry

end smallest_prime_with_42_divisors_l3264_326401


namespace equation_solution_l3264_326420

theorem equation_solution : ∃ x : ℚ, (1 / 3 - 1 / 4 : ℚ) = 1 / (2 * x) ∧ x = 6 := by
  sorry

end equation_solution_l3264_326420


namespace tetromino_tiling_divisibility_l3264_326440

/-- Represents a T-tetromino tile -/
structure TTetromino :=
  (size : Nat)
  (shape : Unit)
  (h_size : size = 4)

/-- Represents a rectangle that can be tiled with T-tetrominoes -/
structure TileableRectangle :=
  (m n : Nat)
  (tiles : List TTetromino)
  (h_tiling : tiles.length * 4 = m * n)  -- Complete tiling without gaps or overlaps

/-- 
If a rectangle can be tiled with T-tetrominoes, then its dimensions are divisible by 4 
-/
theorem tetromino_tiling_divisibility (rect : TileableRectangle) : 
  4 ∣ rect.m ∧ 4 ∣ rect.n :=
sorry

end tetromino_tiling_divisibility_l3264_326440


namespace stone150_is_8_l3264_326426

/-- Represents the circular arrangement of stones with the given counting pattern. -/
def StoneCircle := Fin 15

/-- The number of counts before the pattern repeats. -/
def patternLength : ℕ := 28

/-- Maps a count to its corresponding stone in the circle. -/
def countToStone (count : ℕ) : StoneCircle :=
  sorry

/-- The stone that is counted as 150. -/
def stone150 : StoneCircle :=
  countToStone 150

/-- The original stone number that corresponds to the 150th count. -/
theorem stone150_is_8 : stone150 = ⟨8, sorry⟩ :=
  sorry

end stone150_is_8_l3264_326426


namespace internet_bill_is_100_l3264_326427

/-- Represents the financial transactions and balances in Liza's checking account --/
structure AccountState where
  initialBalance : ℕ
  rentPayment : ℕ
  paycheckDeposit : ℕ
  electricityBill : ℕ
  phoneBill : ℕ
  finalBalance : ℕ

/-- Calculates the internet bill given the account state --/
def calculateInternetBill (state : AccountState) : ℕ :=
  state.initialBalance + state.paycheckDeposit - state.rentPayment - state.electricityBill - state.phoneBill - state.finalBalance

/-- Theorem stating that the internet bill is $100 given the specified account state --/
theorem internet_bill_is_100 (state : AccountState) 
  (h1 : state.initialBalance = 800)
  (h2 : state.rentPayment = 450)
  (h3 : state.paycheckDeposit = 1500)
  (h4 : state.electricityBill = 117)
  (h5 : state.phoneBill = 70)
  (h6 : state.finalBalance = 1563) :
  calculateInternetBill state = 100 := by
  sorry

end internet_bill_is_100_l3264_326427


namespace sum_squares_formula_l3264_326459

theorem sum_squares_formula (m n : ℝ) (h : m + n = 3) : 
  2*m^2 + 4*m*n + 2*n^2 - 6 = 12 := by
  sorry

end sum_squares_formula_l3264_326459


namespace oreo_count_l3264_326414

/-- The number of Oreos James has -/
def james_oreos : ℕ := 43

/-- The number of Oreos Jordan has -/
def jordan_oreos : ℕ := (james_oreos - 7) / 4

/-- The total number of Oreos between James and Jordan -/
def total_oreos : ℕ := james_oreos + jordan_oreos

theorem oreo_count : total_oreos = 52 := by
  sorry

end oreo_count_l3264_326414


namespace xyz_sum_product_bounds_l3264_326460

theorem xyz_sum_product_bounds (x y z : ℝ) : 
  5 * (x + y + z) = x^2 + y^2 + z^2 → 
  ∃ (M m : ℝ), 
    (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 → 
      a * b + a * c + b * c ≤ M) ∧
    (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 → 
      m ≤ a * b + a * c + b * c) ∧
    M + 10 * m = 31 := by
  sorry

end xyz_sum_product_bounds_l3264_326460


namespace sequence_2003_l3264_326494

theorem sequence_2003 (a : ℕ → ℕ) (h1 : a 1 = 0) (h2 : ∀ n : ℕ, a (n + 1) = a n + 2 * n) : 
  a 2003 = 2003 * 2002 := by
sorry

end sequence_2003_l3264_326494


namespace intersection_empty_set_l3264_326431

theorem intersection_empty_set (A : Set α) : ¬(¬(A ∩ ∅ = ∅)) := by
  sorry

end intersection_empty_set_l3264_326431


namespace shirts_arrangement_l3264_326472

/-- The number of ways to arrange shirts -/
def arrange_shirts (red : Nat) (green : Nat) : Nat :=
  Nat.factorial (red + green) / (Nat.factorial red * Nat.factorial green)

/-- The number of ways to arrange shirts with green shirts together -/
def arrange_shirts_green_together (red : Nat) (green : Nat) : Nat :=
  arrange_shirts red 1

theorem shirts_arrangement :
  arrange_shirts 3 2 - arrange_shirts_green_together 3 2 = 6 := by
  sorry

end shirts_arrangement_l3264_326472


namespace h_piecewise_l3264_326438

/-- Piecewise function g(x) -/
noncomputable def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then 3 - x
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (9 - (x - 1.5)^2) - 3
  else if 2 ≤ x ∧ x ≤ 4 then 3 * (x - 2)
  else 0

/-- Function h(x) = g(x) + g(-x) -/
noncomputable def h (x : ℝ) : ℝ := g x + g (-x)

theorem h_piecewise :
  ∀ x : ℝ,
    ((-4 ≤ x ∧ x < -3) → h x = -3 * (x + 2)) ∧
    ((-3 ≤ x ∧ x < 0) → h x = 6) ∧
    ((0 ≤ x ∧ x < 2) → h x = 2 * Real.sqrt (9 - (x - 1.5)^2) - 6) ∧
    ((2 ≤ x ∧ x ≤ 4) → h x = 3 * (x - 2)) := by
  sorry

end h_piecewise_l3264_326438


namespace arithmetic_progression_x_value_l3264_326417

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem arithmetic_progression_x_value :
  ∀ x : ℝ, is_arithmetic_progression (x - 3) (x + 2) (2*x - 1) → x = 8 := by
  sorry

end arithmetic_progression_x_value_l3264_326417


namespace stairs_fibonacci_equivalence_nine_steps_ways_l3264_326457

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

def climbStairs : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => climbStairs n + climbStairs (n + 1)

theorem stairs_fibonacci_equivalence (n : ℕ) : climbStairs n = fibonacci (n + 1) := by
  sorry

theorem nine_steps_ways : climbStairs 9 = 55 := by
  sorry

end stairs_fibonacci_equivalence_nine_steps_ways_l3264_326457


namespace arithmetic_and_geometric_means_l3264_326416

theorem arithmetic_and_geometric_means : 
  (let a := (5 + 17) / 2
   a = 11) ∧
  (let b := Real.sqrt (4 * 9)
   b = 6 ∨ b = -6) := by sorry

end arithmetic_and_geometric_means_l3264_326416


namespace dice_trick_existence_l3264_326498

def DicePair : Type := { p : ℕ × ℕ // p.1 ≤ p.2 ∧ p.1 ≥ 1 ∧ p.2 ≤ 6 }

theorem dice_trick_existence :
  ∃ f : DicePair → ℕ,
    Function.Bijective f ∧
    (∀ p : DicePair, 3 ≤ f p ∧ f p ≤ 21) :=
sorry

end dice_trick_existence_l3264_326498


namespace custom_op_four_six_l3264_326448

def custom_op (a b : ℤ) : ℤ := 4*a - 2*b + a*b

theorem custom_op_four_six : custom_op 4 6 = 28 := by
  sorry

end custom_op_four_six_l3264_326448


namespace saturday_hourly_rate_l3264_326405

/-- Calculates the hourly rate for Saturday work given the following conditions:
  * After-school hourly rate is $4.00
  * Total weekly hours worked is 18
  * Total weekly earnings is $88.00
  * Saturday hours worked is 8.0
-/
theorem saturday_hourly_rate
  (after_school_rate : ℝ)
  (total_hours : ℝ)
  (total_earnings : ℝ)
  (saturday_hours : ℝ)
  (h1 : after_school_rate = 4)
  (h2 : total_hours = 18)
  (h3 : total_earnings = 88)
  (h4 : saturday_hours = 8) :
  (total_earnings - after_school_rate * (total_hours - saturday_hours)) / saturday_hours = 6 :=
by sorry

end saturday_hourly_rate_l3264_326405


namespace quarters_count_l3264_326415

/-- Calculates the number of quarters in a jar given the following conditions:
  * The jar contains 123 pennies, 85 nickels, 35 dimes, and an unknown number of quarters.
  * The total cost of ice cream for 5 family members is $15.
  * After spending on ice cream, 48 cents remain. -/
def quarters_in_jar (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (ice_cream_cost : ℚ) (remaining_cents : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of quarters in the jar is 26. -/
theorem quarters_count : quarters_in_jar 123 85 35 15 48 = 26 := by
  sorry

end quarters_count_l3264_326415


namespace parallel_vectors_m_l3264_326413

def vector_a : Fin 3 → ℝ := ![2, 4, 3]
def vector_b (m : ℝ) : Fin 3 → ℝ := ![4, 8, m]

def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ i, u i = k * v i

theorem parallel_vectors_m (m : ℝ) :
  parallel vector_a (vector_b m) → m = 6 := by
  sorry

end parallel_vectors_m_l3264_326413


namespace log_difference_equals_one_l3264_326467

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_difference_equals_one (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : log a 3 > log a 2) : 
  (log a (2 * a) - log a a = 1) → a = 2 := by
  sorry

end log_difference_equals_one_l3264_326467


namespace inequality_solution_range_l3264_326452

theorem inequality_solution_range (k : ℝ) :
  (∃ x : ℝ, |x + 1| + k < x) ↔ k < -1 := by sorry

end inequality_solution_range_l3264_326452


namespace max_value_of_expression_l3264_326474

theorem max_value_of_expression (m n t : ℝ) (hm : m > 0) (hn : n > 0) (ht : t > 0)
  (heq : m^2 - 3*m*n + 4*n^2 - t = 0) :
  ∃ (m₀ n₀ t₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ t₀ > 0 ∧
    m₀^2 - 3*m₀*n₀ + 4*n₀^2 - t₀ = 0 ∧
    (∀ m' n' t' : ℝ, m' > 0 → n' > 0 → t' > 0 → m'^2 - 3*m'*n' + 4*n'^2 - t' = 0 →
      t₀/(m₀*n₀) ≤ t'/(m'*n')) ∧
    (∀ m' n' t' : ℝ, m' > 0 → n' > 0 → t' > 0 → m'^2 - 3*m'*n' + 4*n'^2 - t' = 0 →
      m' + 2*n' - t' ≤ 2) ∧
    m₀ + 2*n₀ - t₀ = 2 :=
sorry

end max_value_of_expression_l3264_326474


namespace curve_not_parabola_l3264_326424

/-- The equation of the curve -/
def curve_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + (m + 1) * y^2 = m * (m + 1)

/-- Definition of a parabola in general form -/
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧
    ∀ x y, f x y ↔ a * x^2 + b * x * y + c * y^2 + d * x + e * y = 0

/-- Theorem: The curve cannot be a parabola -/
theorem curve_not_parabola :
  ∀ m : ℝ, ¬(is_parabola (curve_equation m)) :=
sorry

end curve_not_parabola_l3264_326424


namespace max_value_product_l3264_326496

theorem max_value_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hsum : 5 * a + 3 * b < 90) :
  a * b * (90 - 5 * a - 3 * b) ≤ 1800 := by
sorry

end max_value_product_l3264_326496


namespace rectangular_field_perimeter_l3264_326471

def rectangle_perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

theorem rectangular_field_perimeter :
  let length : ℝ := 15
  let width : ℝ := 20
  rectangle_perimeter length width = 70 := by
  sorry

end rectangular_field_perimeter_l3264_326471


namespace average_marks_of_passed_boys_l3264_326447

theorem average_marks_of_passed_boys
  (total_boys : ℕ)
  (overall_average : ℚ)
  (passed_boys : ℕ)
  (failed_average : ℚ)
  (h1 : total_boys = 120)
  (h2 : overall_average = 38)
  (h3 : passed_boys = 115)
  (h4 : failed_average = 15)
  : ∃ (passed_average : ℚ), passed_average = 39 ∧
    overall_average * total_boys = passed_average * passed_boys + failed_average * (total_boys - passed_boys) := by
  sorry

end average_marks_of_passed_boys_l3264_326447


namespace count_five_digit_numbers_with_one_odd_l3264_326491

/-- The count of five-digit numbers with exactly one odd digit -/
def five_digit_numbers_with_one_odd : ℕ :=
  let odd_digits := 5  -- Count of odd digits (1, 3, 5, 7, 9)
  let even_digits := 5  -- Count of even digits (0, 2, 4, 6, 8)
  let first_digit_odd := odd_digits * even_digits^4
  let other_digit_odd := 4 * odd_digits * (even_digits - 1) * even_digits^3
  first_digit_odd + other_digit_odd

theorem count_five_digit_numbers_with_one_odd :
  five_digit_numbers_with_one_odd = 10625 := by
  sorry

end count_five_digit_numbers_with_one_odd_l3264_326491


namespace dibromoalkane_formula_l3264_326402

/-- The mass fraction of bromine in a dibromoalkane -/
def bromine_mass_fraction : ℝ := 0.851

/-- The atomic mass of carbon in g/mol -/
def carbon_mass : ℝ := 12

/-- The atomic mass of hydrogen in g/mol -/
def hydrogen_mass : ℝ := 1

/-- The atomic mass of bromine in g/mol -/
def bromine_mass : ℝ := 80

/-- The general formula of a dibromoalkane is CₙH₂ₙBr₂ -/
def dibromoalkane_mass (n : ℕ) : ℝ :=
  n * carbon_mass + 2 * n * hydrogen_mass + 2 * bromine_mass

/-- Theorem: If the mass fraction of bromine in a dibromoalkane is 85.1%, then n = 2 -/
theorem dibromoalkane_formula :
  ∃ (n : ℕ), (2 * bromine_mass) / (dibromoalkane_mass n) = bromine_mass_fraction ∧ n = 2 := by
  sorry

end dibromoalkane_formula_l3264_326402


namespace carlys_dogs_l3264_326425

theorem carlys_dogs (total_nails : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) :
  total_nails = 164 →
  three_legged_dogs = 3 →
  nails_per_paw = 4 →
  ∃ (four_legged_dogs : ℕ),
    four_legged_dogs * 4 * nails_per_paw + three_legged_dogs * 3 * nails_per_paw = total_nails ∧
    four_legged_dogs + three_legged_dogs = 11 :=
by sorry

end carlys_dogs_l3264_326425


namespace tangent_slope_at_point_one_l3264_326406

-- Define the curve function
def f (x : ℝ) : ℝ := x^2 + 3*x - 1

-- Define the derivative of the curve function
def f' (x : ℝ) : ℝ := 2*x + 3

theorem tangent_slope_at_point_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  f' x₀ = 5 ∧ f x₀ = y₀ ∧ y₀ = 3 :=
by sorry

end tangent_slope_at_point_one_l3264_326406


namespace correct_statements_l3264_326497

theorem correct_statements :
  (∀ a : ℝ, ¬(- a < 0) → a ≤ 0) ∧
  (∀ a : ℝ, |-(a^2)| = (-a)^2) ∧
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → a / |a| + b / |b| = 0 → a * b / |a * b| = -1) ∧
  (∀ a b : ℝ, |a| = -b → |b| = b → a = b) :=
by sorry

end correct_statements_l3264_326497


namespace problem_solution_l3264_326428

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 6) 
  (h3 : x = 0) : 
  y = 21/2 := by
sorry

end problem_solution_l3264_326428


namespace smallest_b_value_l3264_326477

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 4) 
  (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 4) : 
  b ≥ 2 ∧ ∃ (a' b' : ℕ+), b' = 2 ∧ a' - b' = 4 ∧ 
    Nat.gcd ((a'^3 + b'^3) / (a' + b')) (a' * b') = 4 :=
sorry

end smallest_b_value_l3264_326477


namespace system_integer_solutions_determinant_l3264_326407

theorem system_integer_solutions_determinant (a b c d : ℤ) :
  (∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) →
  (a * d - b * c = 1 ∨ a * d - b * c = -1) :=
by sorry

end system_integer_solutions_determinant_l3264_326407


namespace smallest_prime_dividing_sum_l3264_326473

theorem smallest_prime_dividing_sum : 
  ∀ p : Nat, Prime p → p ∣ (2^14 + 7^9) → p ≥ 7 :=
by sorry

end smallest_prime_dividing_sum_l3264_326473


namespace cosine_equality_problem_l3264_326432

theorem cosine_equality_problem :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (1018 * π / 180) ∧ n = 62 := by
  sorry

end cosine_equality_problem_l3264_326432


namespace sum_of_digits_greatest_prime_divisor_18447_l3264_326475

def greatest_prime_divisor (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_greatest_prime_divisor_18447 :
  sum_of_digits (greatest_prime_divisor 18447) = 20 := by sorry

end sum_of_digits_greatest_prime_divisor_18447_l3264_326475


namespace statue_cost_l3264_326409

theorem statue_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) :
  selling_price = 670 ∧ 
  profit_percentage = 35 ∧ 
  selling_price = original_cost * (1 + profit_percentage / 100) →
  original_cost = 496.30 := by
sorry

end statue_cost_l3264_326409


namespace brother_travel_distance_l3264_326411

theorem brother_travel_distance (total_time : ℝ) (speed_diff : ℝ) (distance_diff : ℝ) :
  total_time = 120 ∧ speed_diff = 4 ∧ distance_diff = 40 →
  ∃ (x y : ℝ),
    x = 20 ∧ y = 60 ∧
    total_time / x - total_time / y = speed_diff ∧
    y - x = distance_diff :=
by sorry

end brother_travel_distance_l3264_326411


namespace power_sum_l3264_326421

theorem power_sum (a m n : ℝ) (h1 : a^m = 4) (h2 : a^n = 8) : a^(m+n) = 32 := by
  sorry

end power_sum_l3264_326421


namespace intersection_A_B_union_complement_B_A_l3264_326419

open Set

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 0 < x}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem for the union of complement of B and A
theorem union_complement_B_A : (𝒰 \ B) ∪ A = {x | x < 2} := by sorry

end intersection_A_B_union_complement_B_A_l3264_326419


namespace largest_factorial_as_consecutive_product_l3264_326441

theorem largest_factorial_as_consecutive_product : 
  ∀ n : ℕ, n > 0 → 
  (∃ k : ℕ, k > 0 ∧ n.factorial = (List.range (n - 5)).prod.succ) → 
  n ≤ 0 :=
sorry

end largest_factorial_as_consecutive_product_l3264_326441


namespace absolute_value_inequality_l3264_326430

theorem absolute_value_inequality (x : ℝ) :
  (3 ≤ |x + 3| ∧ |x + 3| ≤ 7) ↔ ((-10 ≤ x ∧ x ≤ -6) ∨ (0 ≤ x ∧ x ≤ 4)) := by
  sorry

end absolute_value_inequality_l3264_326430


namespace cosine_of_angle_between_vectors_l3264_326436

/-- Given planar vectors a and b satisfying the conditions,
    prove that the cosine of the angle between them is 1/2 -/
theorem cosine_of_angle_between_vectors
  (a b : ℝ × ℝ)  -- Planar vectors represented as pairs of real numbers
  (h1 : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 5)  -- a · (a + b) = 5
  (h2 : a.1^2 + a.2^2 = 4)  -- |a| = 2
  (h3 : b.1^2 + b.2^2 = 1)  -- |b| = 1
  : (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = 1/2 := by
  sorry


end cosine_of_angle_between_vectors_l3264_326436


namespace polynomial_divisibility_l3264_326489

theorem polynomial_divisibility (m n : ℤ) :
  (∀ (x y : ℤ), (107 ∣ (x^3 + m*x + n) - (y^3 + m*y + n)) → (107 ∣ (x - y))) →
  (107 ∣ m) := by
  sorry

end polynomial_divisibility_l3264_326489


namespace remainder_theorem_l3264_326444

theorem remainder_theorem (n : ℤ) : 
  (2 * n) % 11 = 2 → n % 22 = 1 :=
by sorry

end remainder_theorem_l3264_326444


namespace walking_distance_approx_2_9_l3264_326463

/-- Represents a journey with cycling and walking portions -/
structure Journey where
  total_time : ℝ
  bike_speed : ℝ
  walk_speed : ℝ
  bike_fraction : ℝ
  walk_fraction : ℝ

/-- Calculates the walking distance for a given journey -/
def walking_distance (j : Journey) : ℝ :=
  let total_distance := (j.bike_speed * j.bike_fraction + j.walk_speed * j.walk_fraction) * j.total_time
  total_distance * j.walk_fraction

/-- Theorem stating that for the given journey parameters, the walking distance is approximately 2.9 km -/
theorem walking_distance_approx_2_9 :
  let j : Journey := {
    total_time := 1,
    bike_speed := 20,
    walk_speed := 4,
    bike_fraction := 2/3,
    walk_fraction := 1/3
  }
  ∃ ε > 0, |walking_distance j - 2.9| < ε :=
sorry

end walking_distance_approx_2_9_l3264_326463


namespace birds_ate_one_third_of_tomatoes_l3264_326490

theorem birds_ate_one_third_of_tomatoes
  (initial_tomatoes : ℕ)
  (remaining_tomatoes : ℕ)
  (h1 : initial_tomatoes = 21)
  (h2 : remaining_tomatoes = 14) :
  (initial_tomatoes - remaining_tomatoes : ℚ) / initial_tomatoes = 1 / 3 :=
by sorry

end birds_ate_one_third_of_tomatoes_l3264_326490


namespace two_red_two_blue_probability_l3264_326433

def total_marbles : ℕ := 15 + 9

def red_marbles : ℕ := 15

def blue_marbles : ℕ := 9

def marbles_selected : ℕ := 4

theorem two_red_two_blue_probability :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2) / Nat.choose total_marbles marbles_selected = 108 / 361 :=
by sorry

end two_red_two_blue_probability_l3264_326433


namespace ratio_sum_difference_l3264_326400

theorem ratio_sum_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / y = (x + y) / (x - y) → x / y = 1 + Real.sqrt 2 := by
  sorry

end ratio_sum_difference_l3264_326400


namespace max_b_minus_a_l3264_326499

theorem max_b_minus_a (a b : ℝ) (ha : a < 0)
  (h : ∀ x : ℝ, (3 * x^2 + a) * (2 * x + b) ≥ 0) :
  ∃ (max : ℝ), max = 1/3 ∧ b - a ≤ max ∧
  ∀ (a' b' : ℝ), a' < 0 → (∀ x : ℝ, (3 * x^2 + a') * (2 * x + b') ≥ 0) →
  b' - a' ≤ max :=
sorry

end max_b_minus_a_l3264_326499


namespace sin_plus_power_cos_pi_third_l3264_326449

theorem sin_plus_power_cos_pi_third :
  Real.sin 3 + 2^(8-3) * Real.cos (π/3) = Real.sin 3 + 16 := by
  sorry

end sin_plus_power_cos_pi_third_l3264_326449


namespace total_animal_eyes_l3264_326403

theorem total_animal_eyes (num_snakes num_alligators : ℕ) 
  (snake_eyes alligator_eyes : ℕ) : ℕ :=
  by
    -- Define the number of snakes and alligators
    have h1 : num_snakes = 18 := by sorry
    have h2 : num_alligators = 10 := by sorry
    
    -- Define the number of eyes for each snake and alligator
    have h3 : snake_eyes = 2 := by sorry
    have h4 : alligator_eyes = 2 := by sorry
    
    -- Calculate total number of eyes
    have h5 : num_snakes * snake_eyes + num_alligators * alligator_eyes = 56 := by sorry
    
    exact 56

#check total_animal_eyes

end total_animal_eyes_l3264_326403


namespace johnsRemainingMoneyTheorem_l3264_326492

/-- The amount of money John has left after purchasing pizzas and drinks -/
def johnsRemainingMoney (d : ℝ) : ℝ :=
  let drinkCost := d
  let mediumPizzaCost := 3 * d
  let largePizzaCost := 4 * d
  let totalCost := 5 * drinkCost + mediumPizzaCost + 2 * largePizzaCost
  50 - totalCost

/-- Theorem stating that John's remaining money is 50 - 16d -/
theorem johnsRemainingMoneyTheorem (d : ℝ) :
  johnsRemainingMoney d = 50 - 16 * d :=
by sorry

end johnsRemainingMoneyTheorem_l3264_326492


namespace exists_grid_with_partitions_l3264_326442

/-- A cell in the grid --/
structure Cell :=
  (x : Nat) (y : Nat)

/-- A shape in the grid --/
structure Shape :=
  (cells : List Cell)

/-- The grid --/
def Grid := List Cell

/-- Predicate to check if a shape is valid (contains 5 cells) --/
def isValidShape5 (s : Shape) : Prop :=
  s.cells.length = 5

/-- Predicate to check if a shape is valid (contains 4 cells) --/
def isValidShape4 (s : Shape) : Prop :=
  s.cells.length = 4

/-- Predicate to check if shapes are equal (up to rotation and flipping) --/
def areShapesEqual (s1 s2 : Shape) : Prop :=
  sorry  -- Implementation of shape equality check

/-- Theorem stating the existence of a grid with the required properties --/
theorem exists_grid_with_partitions :
  ∃ (g : Grid) (partition1 partition2 : List Shape),
    g.length = 20 ∧
    partition1.length = 4 ∧
    (∀ s ∈ partition1, isValidShape5 s) ∧
    (∀ i j, i < partition1.length → j < partition1.length → i ≠ j →
      areShapesEqual (partition1.get ⟨i, sorry⟩) (partition1.get ⟨j, sorry⟩)) ∧
    partition2.length = 5 ∧
    (∀ s ∈ partition2, isValidShape4 s) ∧
    (∀ i j, i < partition2.length → j < partition2.length → i ≠ j →
      areShapesEqual (partition2.get ⟨i, sorry⟩) (partition2.get ⟨j, sorry⟩)) :=
by
  sorry


end exists_grid_with_partitions_l3264_326442


namespace gumball_range_l3264_326404

theorem gumball_range (x : ℤ) : 
  let carolyn := 17
  let lew := 12
  let total := carolyn + lew + x
  let avg := total / 3
  (19 ≤ avg ∧ avg ≤ 25) →
  (max x - min x = 18) :=
by sorry

end gumball_range_l3264_326404


namespace equation_solution_l3264_326462

theorem equation_solution (x : ℝ) : 
  (Real.sqrt ((3 + Real.sqrt 5) ^ x)) ^ 2 + (Real.sqrt ((3 - Real.sqrt 5) ^ x)) ^ 2 = 18 ↔ 
  x = 2 ∨ x = -2 :=
sorry

end equation_solution_l3264_326462


namespace moving_points_theorem_l3264_326412

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- The main theorem -/
theorem moving_points_theorem (ABC : Triangle) (P Q : Point) (t : ℝ) :
  (ABC.B.x - ABC.A.x)^2 + (ABC.B.y - ABC.A.y)^2 = 36 →  -- AB = 6 cm
  (ABC.C.x - ABC.B.x)^2 + (ABC.C.y - ABC.B.y)^2 = 64 →  -- BC = 8 cm
  (ABC.C.x - ABC.B.x) * (ABC.B.y - ABC.A.y) = (ABC.C.y - ABC.B.y) * (ABC.B.x - ABC.A.x) →  -- ABC is right-angled at B
  P.x = ABC.A.x + t →  -- P moves from A towards B
  P.y = ABC.A.y →
  Q.x = ABC.B.x + 2 * t →  -- Q moves from B towards C
  Q.y = ABC.B.y →
  triangleArea P ABC.B Q = 5 →  -- Area of PBQ is 5 cm²
  t = 1  -- Time P moves is 1 second
  := by sorry

end moving_points_theorem_l3264_326412


namespace f_two_zero_l3264_326469

/-- A mapping f that takes a point (x,y) to (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that f(2,0) = (2,2) -/
theorem f_two_zero : f (2, 0) = (2, 2) := by
  sorry

end f_two_zero_l3264_326469


namespace school_walk_time_difference_l3264_326451

/-- Proves that a child walking to school is 6 minutes late when walking at 5 m/min,
    given the conditions of the problem. -/
theorem school_walk_time_difference (distance : ℝ) (slow_rate fast_rate : ℝ) (early_time : ℝ) :
  distance = 630 →
  slow_rate = 5 →
  fast_rate = 7 →
  early_time = 30 →
  distance / fast_rate + early_time = distance / slow_rate →
  distance / slow_rate - distance / fast_rate = 6 :=
by sorry

end school_walk_time_difference_l3264_326451
