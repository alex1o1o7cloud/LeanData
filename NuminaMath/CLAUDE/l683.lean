import Mathlib

namespace NUMINAMATH_CALUDE_certain_number_equation_l683_68334

theorem certain_number_equation (x : ℤ) : 34 + x - 53 = 28 ↔ x = 47 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l683_68334


namespace NUMINAMATH_CALUDE_peter_erasers_count_l683_68357

def initial_erasers : ℕ := 8
def multiplier : ℕ := 3

theorem peter_erasers_count : 
  initial_erasers + multiplier * initial_erasers = 32 :=
by sorry

end NUMINAMATH_CALUDE_peter_erasers_count_l683_68357


namespace NUMINAMATH_CALUDE_sum_of_odds_is_even_product_zero_implies_factor_zero_exists_even_prime_l683_68362

-- Definition of odd number
def isOdd (n : Int) : Prop := ∃ k : Int, n = 2 * k + 1

-- Statement 1
theorem sum_of_odds_is_even (x y : Int) (hx : isOdd x) (hy : isOdd y) : 
  ∃ k : Int, x + y = 2 * k := by sorry

-- Statement 2
theorem product_zero_implies_factor_zero (x y : ℝ) (h : x * y = 0) : 
  x = 0 ∨ y = 0 := by sorry

-- Definition of prime number
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

-- Statement 3
theorem exists_even_prime : ∃ p : Nat, isPrime p ∧ ¬isOdd p := by sorry

end NUMINAMATH_CALUDE_sum_of_odds_is_even_product_zero_implies_factor_zero_exists_even_prime_l683_68362


namespace NUMINAMATH_CALUDE_range_of_m_satisfying_conditions_l683_68393

def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m + 6)*x + 1

theorem range_of_m_satisfying_conditions : 
  {m : ℝ | (∀ a ∈ Set.Icc 1 2, |m - 5| ≤ Real.sqrt (a^2 + 8)) ∧ 
           ¬(∃ (max min : ℝ), ∀ x, f m x ≤ max ∧ f m x ≥ min)} = 
  Set.Icc 2 6 := by sorry

end NUMINAMATH_CALUDE_range_of_m_satisfying_conditions_l683_68393


namespace NUMINAMATH_CALUDE_negation_of_implication_is_true_l683_68322

theorem negation_of_implication_is_true : 
  ¬(∀ a : ℝ, a ≤ 2 → a^2 < 4) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_is_true_l683_68322


namespace NUMINAMATH_CALUDE_sequence_increasing_iff_k_greater_than_neg_three_l683_68344

theorem sequence_increasing_iff_k_greater_than_neg_three (k : ℝ) :
  (∀ n : ℕ, (n^2 + k*n + 2) < ((n+1)^2 + k*(n+1) + 2)) ↔ k > -3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_iff_k_greater_than_neg_three_l683_68344


namespace NUMINAMATH_CALUDE_isosceles_triangle_locus_l683_68392

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a circle -/
def CircleEquation (center : Point) (radius : ℝ) (p : Point) : Prop :=
  (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2

/-- The locus equation for point C -/
def LocusEquation (p : Point) : Prop :=
  p.x^2 + p.y^2 - 3*p.x + p.y = 2

theorem isosceles_triangle_locus :
  ∀ (C : Point),
    let A : Point := ⟨3, -2⟩
    let B : Point := ⟨0, 1⟩
    let M : Point := ⟨3/2, -1/2⟩  -- Midpoint of AB
    let r : ℝ := (3 * Real.sqrt 2) / 2  -- Radius of the circle
    C ≠ A ∧ C ≠ B →  -- Exclude points A and B
    (CircleEquation M r C ↔ LocusEquation C) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_locus_l683_68392


namespace NUMINAMATH_CALUDE_trigonometric_expression_simplification_l683_68317

theorem trigonometric_expression_simplification :
  let original_expression := (Real.sin (20 * π / 180) + Real.sin (40 * π / 180) + 
                              Real.sin (60 * π / 180) + Real.sin (80 * π / 180)) / 
                             (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * 
                              Real.cos (30 * π / 180) * Real.cos (40 * π / 180))
  let simplified_expression := (4 * Real.sin (50 * π / 180)) / 
                               (Real.cos (30 * π / 180) * Real.cos (40 * π / 180))
  original_expression = simplified_expression := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_simplification_l683_68317


namespace NUMINAMATH_CALUDE_complete_factorization_l683_68330

theorem complete_factorization (x : ℝ) : 
  x^8 - 256 = (x^4 + 16) * (x^2 + 4) * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_complete_factorization_l683_68330


namespace NUMINAMATH_CALUDE_existence_implies_upper_bound_l683_68329

theorem existence_implies_upper_bound (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 3 ∧ x^2 - 3*x - a > 0) → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_existence_implies_upper_bound_l683_68329


namespace NUMINAMATH_CALUDE_last_three_digits_of_6_to_150_l683_68380

theorem last_three_digits_of_6_to_150 :
  6^150 % 1000 = 126 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_6_to_150_l683_68380


namespace NUMINAMATH_CALUDE_min_sum_inverse_ratio_l683_68347

theorem min_sum_inverse_ratio (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a)) ≥ 1 / (3 * Real.rpow 2 (1/3)) :=
sorry

end NUMINAMATH_CALUDE_min_sum_inverse_ratio_l683_68347


namespace NUMINAMATH_CALUDE_sum_product_bounds_l683_68337

theorem sum_product_bounds (a b c k : ℝ) (h : a + b + c = k) (h_nonzero : k ≠ 0) :
  -2/3 * k^2 ≤ a*b + a*c + b*c ∧ a*b + a*c + b*c ≤ k^2/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_bounds_l683_68337


namespace NUMINAMATH_CALUDE_percentage_problem_l683_68375

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.15 * 1500 - 15 → x = 840 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l683_68375


namespace NUMINAMATH_CALUDE_some_number_value_l683_68363

theorem some_number_value (some_number : ℝ) : 
  (3.242 * 10) / some_number = 0.032420000000000004 → some_number = 1000 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l683_68363


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l683_68310

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1)) / (a * b * c) ≥ 216 :=
by
  sorry

theorem min_value_achieved (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    ((x^2 + 4*x + 1) * (y^2 + 4*y + 1) * (z^2 + 4*z + 1)) / (x * y * z) = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l683_68310


namespace NUMINAMATH_CALUDE_hyperbola_proof_1_hyperbola_proof_2_l683_68311

-- Part 1
def hyperbola_equation_1 (x y : ℝ) : Prop :=
  x^2 / 5 - y^2 = 1

theorem hyperbola_proof_1 (c : ℝ) (h1 : c = Real.sqrt 6) :
  hyperbola_equation_1 (-5) 2 ∧
  ∃ a b : ℝ, c^2 = a^2 + b^2 ∧ hyperbola_equation_1 x y ↔ x^2 / a^2 - y^2 / b^2 = 1 :=
sorry

-- Part 2
def hyperbola_equation_2 (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

theorem hyperbola_proof_2 :
  hyperbola_equation_2 3 (-4 * Real.sqrt 2) ∧
  hyperbola_equation_2 (9/4) 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_proof_1_hyperbola_proof_2_l683_68311


namespace NUMINAMATH_CALUDE_sauce_correction_l683_68314

theorem sauce_correction (x : ℝ) : 
  (0.4 * x - 1 + 2.5 = 0.6 * x - 1.5) → x = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_sauce_correction_l683_68314


namespace NUMINAMATH_CALUDE_associativity_of_mul_l683_68332

-- Define the set S and its binary operation
variable {S : Type}
variable (add : S → S → S)

-- Define the properties of the set S
variable (h1 : ∀ (a b c : S), add (add a c) (add b c) = add a b)
variable (h2 : ∃ (e : S), (∀ (a : S), add a e = a ∧ add a a = e))

-- Define the * operation
def mul (add : S → S → S) (e : S) (a b : S) : S := add a (add e b)

-- State the theorem
theorem associativity_of_mul 
  (add : S → S → S) 
  (h1 : ∀ (a b c : S), add (add a c) (add b c) = add a b)
  (h2 : ∃ (e : S), (∀ (a : S), add a e = a ∧ add a a = e)) :
  ∀ (a b c : S), mul add (Classical.choose h2) (mul add (Classical.choose h2) a b) c = 
                 mul add (Classical.choose h2) a (mul add (Classical.choose h2) b c) :=
by
  sorry


end NUMINAMATH_CALUDE_associativity_of_mul_l683_68332


namespace NUMINAMATH_CALUDE_min_packages_lcm_l683_68379

/-- The load capacity of Sarah's trucks -/
def sarah_capacity : ℕ := 18

/-- The load capacity of Ryan's trucks -/
def ryan_capacity : ℕ := 11

/-- The load capacity of Emily's trucks -/
def emily_capacity : ℕ := 15

/-- The minimum number of packages each business must have shipped -/
def min_packages : ℕ := 990

theorem min_packages_lcm :
  Nat.lcm (Nat.lcm sarah_capacity ryan_capacity) emily_capacity = min_packages :=
sorry

end NUMINAMATH_CALUDE_min_packages_lcm_l683_68379


namespace NUMINAMATH_CALUDE_green_peaches_count_l683_68390

theorem green_peaches_count (red_peaches : ℕ) (green_peaches : ℕ) : 
  red_peaches = 5 → green_peaches = red_peaches + 6 → green_peaches = 11 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l683_68390


namespace NUMINAMATH_CALUDE_fixed_point_theorem_tangent_dot_product_range_l683_68382

-- Define the curves C and M
def C (x y : ℝ) : Prop := y^2 = 4*x
def M (x y : ℝ) : Prop := (x-1)^2 + y^2 = 4 ∧ x ≥ 1

-- Define the line l
def L (m n : ℝ) (x y : ℝ) : Prop := x = m*y + n

-- Define points A and B on curve C and line l
def A_B_on_C_and_L (m n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  C x₁ y₁ ∧ C x₂ y₂ ∧ L m n x₁ y₁ ∧ L m n x₂ y₂

-- Theorem 1
theorem fixed_point_theorem (m n x₁ y₁ x₂ y₂ : ℝ) :
  A_B_on_C_and_L m n x₁ y₁ x₂ y₂ →
  x₁*x₂ + y₁*y₂ = -4 →
  ∃ (m : ℝ), L m 2 2 0 :=
sorry

-- Theorem 2
theorem tangent_dot_product_range (m n x₁ y₁ x₂ y₂ : ℝ) :
  A_B_on_C_and_L m n x₁ y₁ x₂ y₂ →
  (∃ (x y : ℝ), M x y ∧ L m n x y) →
  (x₁-1)*(x₂-1) + y₁*y₂ ≤ -8 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_tangent_dot_product_range_l683_68382


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l683_68355

theorem shaded_area_percentage (total_squares : ℕ) (shaded_squares : ℕ) : 
  total_squares = 16 → shaded_squares = 3 → 
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l683_68355


namespace NUMINAMATH_CALUDE_complex_square_l683_68388

theorem complex_square (z : ℂ) (i : ℂ) : z = 2 - 3 * i → i^2 = -1 → z^2 = -5 - 12 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l683_68388


namespace NUMINAMATH_CALUDE_rotations_composition_l683_68366

/-- A rotation in the plane. -/
structure Rotation where
  center : ℝ × ℝ
  angle : ℝ

/-- Represents the composition of two rotations. -/
def compose_rotations (r1 r2 : Rotation) : Rotation :=
  sorry

/-- The angles of a triangle formed by the centers of two rotations and their composition. -/
def triangle_angles (r1 r2 : Rotation) : ℝ × ℝ × ℝ :=
  sorry

theorem rotations_composition 
  (O₁ O₂ : ℝ × ℝ) (α β : ℝ) 
  (h1 : 0 ≤ α ∧ α < 2 * π) 
  (h2 : 0 ≤ β ∧ β < 2 * π) 
  (h3 : α + β ≠ 2 * π) :
  let r1 : Rotation := ⟨O₁, α⟩
  let r2 : Rotation := ⟨O₂, β⟩
  let r_composed := compose_rotations r1 r2
  let angles := triangle_angles r1 r2
  (r_composed.angle = α + β) ∧
  ((α + β < 2 * π → angles = (α/2, β/2, π - (α + β)/2)) ∧
   (α + β > 2 * π → angles = (π - α/2, π - β/2, (α + β)/2))) :=
by sorry

end NUMINAMATH_CALUDE_rotations_composition_l683_68366


namespace NUMINAMATH_CALUDE_unit_digit_4137_754_l683_68385

theorem unit_digit_4137_754 : (4137^754) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_4137_754_l683_68385


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_2501_l683_68389

theorem largest_prime_factor_of_2501 : ∃ p : ℕ, p.Prime ∧ p ∣ 2501 ∧ ∀ q : ℕ, q.Prime → q ∣ 2501 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_2501_l683_68389


namespace NUMINAMATH_CALUDE_machine_working_time_yesterday_l683_68397

/-- The total working time of an industrial machine, including downtime -/
def total_working_time (shirts_produced : ℕ) (production_rate : ℕ) (downtime : ℕ) : ℕ :=
  shirts_produced * production_rate + downtime

/-- Proof that the machine worked for 38 minutes yesterday -/
theorem machine_working_time_yesterday :
  total_working_time 9 2 20 = 38 := by
  sorry

end NUMINAMATH_CALUDE_machine_working_time_yesterday_l683_68397


namespace NUMINAMATH_CALUDE_base_seven_division_1452_14_l683_68321

/-- Represents a number in base 7 --/
def BaseSevenNum := List Nat

/-- Converts a base 7 number to base 10 --/
def to_base_ten (n : BaseSevenNum) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (7 ^ i)) 0

/-- Converts a base 10 number to base 7 --/
def to_base_seven (n : Nat) : BaseSevenNum :=
  sorry

/-- Performs division in base 7 --/
def base_seven_div (a b : BaseSevenNum) : BaseSevenNum :=
  to_base_seven ((to_base_ten a) / (to_base_ten b))

theorem base_seven_division_1452_14 :
  base_seven_div [2, 5, 4, 1] [4, 1] = [3, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base_seven_division_1452_14_l683_68321


namespace NUMINAMATH_CALUDE_manufacturer_not_fraudulent_l683_68327

/-- Represents the mass of a bread bag -/
structure BreadBag where
  labeledMass : ℝ
  tolerance : ℝ
  measuredMass : ℝ

/-- Determines if the manufacturer has engaged in fraudulent behavior -/
def isFraudulent (bag : BreadBag) : Prop :=
  bag.measuredMass < bag.labeledMass - bag.tolerance ∨ 
  bag.measuredMass > bag.labeledMass + bag.tolerance

theorem manufacturer_not_fraudulent (bag : BreadBag) 
  (h1 : bag.labeledMass = 200)
  (h2 : bag.tolerance = 3)
  (h3 : bag.measuredMass = 198) : 
  ¬(isFraudulent bag) := by
  sorry

#check manufacturer_not_fraudulent

end NUMINAMATH_CALUDE_manufacturer_not_fraudulent_l683_68327


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l683_68396

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 1024 ways to distribute 5 distinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l683_68396


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l683_68345

/-- The minimum distance from a point on the circle x^2 + y^2 - 2x - 2y = 0 to the line x + y - 8 = 0 is 2√2. -/
theorem min_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | (p.1^2 + p.2^2 - 2*p.1 - 2*p.2) = 0}
  let line := {p : ℝ × ℝ | p.1 + p.2 - 8 = 0}
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧
    ∀ (p : ℝ × ℝ), p ∈ circle →
      ∀ (q : ℝ × ℝ), q ∈ line →
        d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l683_68345


namespace NUMINAMATH_CALUDE_graphing_to_scientific_ratio_l683_68350

/-- Represents the cost of calculators and the transaction details -/
structure CalculatorPurchase where
  basic_cost : ℝ
  scientific_cost : ℝ
  graphing_cost : ℝ
  total_spent : ℝ

/-- The conditions of the calculator purchase problem -/
def calculator_problem : CalculatorPurchase :=
  { basic_cost := 8
  , scientific_cost := 16
  , graphing_cost := 72 - 8 - 16
  , total_spent := 100 - 28 }

/-- Theorem stating that the ratio of graphing to scientific calculator cost is 3:1 -/
theorem graphing_to_scientific_ratio :
  calculator_problem.graphing_cost / calculator_problem.scientific_cost = 3 := by
  sorry


end NUMINAMATH_CALUDE_graphing_to_scientific_ratio_l683_68350


namespace NUMINAMATH_CALUDE_stream_bottom_width_l683_68349

/-- Represents the trapezoidal cross-section of a stream -/
structure StreamCrossSection where
  topWidth : ℝ
  bottomWidth : ℝ
  depth : ℝ
  area : ℝ

/-- The area of a trapezoid is equal to the average of its parallel sides multiplied by its height -/
def trapezoidAreaFormula (s : StreamCrossSection) : Prop :=
  s.area = (s.topWidth + s.bottomWidth) / 2 * s.depth

theorem stream_bottom_width
  (s : StreamCrossSection)
  (h1 : s.topWidth = 10)
  (h2 : s.depth = 80)
  (h3 : s.area = 640)
  (h4 : trapezoidAreaFormula s) :
  s.bottomWidth = 6 := by
  sorry

end NUMINAMATH_CALUDE_stream_bottom_width_l683_68349


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l683_68331

/-- Given an arithmetic sequence {a_n} with a₁ ≠ 0, if S₁, S₂, S₄ form a geometric sequence, 
    then a₂/a₁ = 1 or a₂/a₁ = 3 -/
theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (a 1 ≠ 0) →                          -- first term not zero
  (∀ n, S n = (n : ℝ) / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) →  -- sum formula
  (∃ r, S 2 = r * S 1 ∧ S 4 = r * S 2) →  -- geometric sequence condition
  a 2 / a 1 = 1 ∨ a 2 / a 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l683_68331


namespace NUMINAMATH_CALUDE_final_sum_after_transformations_l683_68359

theorem final_sum_after_transformations (S a b : ℝ) (h : a + b = S) :
  3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by sorry

end NUMINAMATH_CALUDE_final_sum_after_transformations_l683_68359


namespace NUMINAMATH_CALUDE_final_segment_speed_final_segment_speed_is_90_l683_68325

/-- Calculates the average speed for the final segment of a journey given specific conditions. -/
theorem final_segment_speed (total_distance : ℝ) (total_time : ℝ) (first_hour_speed : ℝ) 
  (stop_time : ℝ) (second_segment_speed : ℝ) (second_segment_time : ℝ) : ℝ :=
  let net_driving_time := total_time - stop_time / 60
  let first_segment_distance := first_hour_speed * 1
  let second_segment_distance := second_segment_speed * second_segment_time
  let remaining_distance := total_distance - (first_segment_distance + second_segment_distance)
  let remaining_time := net_driving_time - (1 + second_segment_time)
  remaining_distance / remaining_time

/-- Proves that the average speed for the final segment is 90 mph under given conditions. -/
theorem final_segment_speed_is_90 : 
  final_segment_speed 150 3 45 30 50 0.75 = 90 := by
  sorry

end NUMINAMATH_CALUDE_final_segment_speed_final_segment_speed_is_90_l683_68325


namespace NUMINAMATH_CALUDE_no_real_roots_l683_68320

theorem no_real_roots : ∀ x : ℝ, x^2 - 4*x + 8 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l683_68320


namespace NUMINAMATH_CALUDE_cats_remaining_l683_68384

def siamese_cats : ℕ := 38
def house_cats : ℕ := 25
def cats_sold : ℕ := 45

theorem cats_remaining : siamese_cats + house_cats - cats_sold = 18 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_l683_68384


namespace NUMINAMATH_CALUDE_ned_chocolate_pieces_l683_68339

theorem ned_chocolate_pieces : 
  ∀ (boxes_bought boxes_given pieces_per_box : ℝ),
    boxes_bought = 14.0 →
    boxes_given = 7.0 →
    pieces_per_box = 6.0 →
    (boxes_bought - boxes_given) * pieces_per_box = 42.0 := by
  sorry

end NUMINAMATH_CALUDE_ned_chocolate_pieces_l683_68339


namespace NUMINAMATH_CALUDE_train_bridge_problem_l683_68313

/-- Given a train crossing a bridge, this theorem proves the length and speed of the train. -/
theorem train_bridge_problem (bridge_length : ℝ) (total_time : ℝ) (on_bridge_time : ℝ) 
  (h1 : bridge_length = 1000)
  (h2 : total_time = 60)
  (h3 : on_bridge_time = 40) :
  ∃ (train_length : ℝ) (train_speed : ℝ),
    train_length = 200 ∧ 
    train_speed = 20 ∧
    (bridge_length + train_length) / total_time = (bridge_length - train_length) / on_bridge_time :=
by
  sorry

#check train_bridge_problem

end NUMINAMATH_CALUDE_train_bridge_problem_l683_68313


namespace NUMINAMATH_CALUDE_probability_after_removal_l683_68326

theorem probability_after_removal (total : ℕ) (blue : ℕ) (removed : ℕ) 
  (h1 : total = 25)
  (h2 : blue = 9)
  (h3 : removed = 5)
  (h4 : removed < blue)
  (h5 : removed < total) :
  (blue - removed : ℚ) / (total - removed) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_after_removal_l683_68326


namespace NUMINAMATH_CALUDE_children_creativity_center_contradiction_l683_68300

theorem children_creativity_center_contradiction (N : ℕ) (d : Fin N → ℕ) : 
  N = 32 ∧ 
  (∀ i, d i = 6) ∧ 
  (∀ i j, i ≠ j → d i + d j = 13) → 
  False :=
sorry

end NUMINAMATH_CALUDE_children_creativity_center_contradiction_l683_68300


namespace NUMINAMATH_CALUDE_student_A_most_stable_l683_68358

/-- Represents a student with their score variance -/
structure Student where
  name : String
  variance : Real

/-- Theorem: Given the variances of four students' scores, prove that student A has the most stable performance -/
theorem student_A_most_stable
  (students : Finset Student)
  (hA : Student.mk "A" 3.8 ∈ students)
  (hB : Student.mk "B" 5.5 ∈ students)
  (hC : Student.mk "C" 10 ∈ students)
  (hD : Student.mk "D" 6 ∈ students)
  (h_count : students.card = 4)
  : ∀ s ∈ students, (Student.mk "A" 3.8).variance ≤ s.variance :=
by sorry


end NUMINAMATH_CALUDE_student_A_most_stable_l683_68358


namespace NUMINAMATH_CALUDE_loafer_cost_l683_68377

/-- Calculate the cost of each pair of loafers given the sales and commission information -/
theorem loafer_cost (commission_rate : ℚ) (suit_price shirt_price : ℚ) 
  (suit_count shirt_count loafer_count : ℕ) (total_commission : ℚ) : 
  commission_rate = 15 / 100 →
  suit_count = 2 →
  shirt_count = 6 →
  loafer_count = 2 →
  suit_price = 700 →
  shirt_price = 50 →
  total_commission = 300 →
  (suit_count : ℚ) * suit_price * commission_rate + 
  (shirt_count : ℚ) * shirt_price * commission_rate + 
  (loafer_count : ℚ) * (total_commission / (loafer_count : ℚ)) = total_commission →
  total_commission / (loafer_count : ℚ) / commission_rate = 150 := by
sorry

end NUMINAMATH_CALUDE_loafer_cost_l683_68377


namespace NUMINAMATH_CALUDE_green_caterpillar_length_l683_68376

theorem green_caterpillar_length 
  (orange_length : ℝ) 
  (length_difference : ℝ) 
  (h1 : orange_length = 1.17)
  (h2 : length_difference = 1.83) : 
  orange_length + length_difference = 3.00 := by
  sorry

end NUMINAMATH_CALUDE_green_caterpillar_length_l683_68376


namespace NUMINAMATH_CALUDE_train_length_l683_68365

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length. -/
theorem train_length (train_speed : ℝ) (time_to_pass : ℝ) (bridge_length : ℝ) :
  train_speed = 45 →
  time_to_pass = 36 →
  bridge_length = 140 →
  (train_speed * 1000 / 3600) * time_to_pass - bridge_length = 310 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l683_68365


namespace NUMINAMATH_CALUDE_eagle_eye_camera_is_analogical_reasoning_l683_68352

-- Define the different types of reasoning
inductive ReasoningType
  | Inductive
  | Analogical
  | Deductive

-- Define a structure for a reasoning process
structure ReasoningProcess where
  description : String
  type : ReasoningType

-- Define the four options
def optionA : ReasoningProcess :=
  { description := "People derive that the probability of getting heads when flipping a coin is 1/2 through numerous experiments",
    type := ReasoningType.Inductive }

def optionB : ReasoningProcess :=
  { description := "Scientists invent the eagle eye camera by studying the eyes of eagles",
    type := ReasoningType.Analogical }

def optionC : ReasoningProcess :=
  { description := "Determine the acidity or alkalinity of a solution by testing its pH value",
    type := ReasoningType.Deductive }

def optionD : ReasoningProcess :=
  { description := "Determine whether a function is periodic based on the definition of a periodic function in mathematics",
    type := ReasoningType.Deductive }

-- Theorem to prove
theorem eagle_eye_camera_is_analogical_reasoning :
  optionB.type = ReasoningType.Analogical :=
by sorry

end NUMINAMATH_CALUDE_eagle_eye_camera_is_analogical_reasoning_l683_68352


namespace NUMINAMATH_CALUDE_interview_probability_l683_68336

/-- The total number of students in at least one club -/
def total_students : ℕ := 30

/-- The number of students in the Robotics club -/
def robotics_students : ℕ := 22

/-- The number of students in the Drama club -/
def drama_students : ℕ := 19

/-- The probability of selecting two students who are not both from the same single club -/
theorem interview_probability : 
  (Nat.choose total_students 2 - (Nat.choose (robotics_students + drama_students - total_students) 2 + 
   Nat.choose (drama_students - (robotics_students + drama_students - total_students)) 2)) / 
  Nat.choose total_students 2 = 352 / 435 := by sorry

end NUMINAMATH_CALUDE_interview_probability_l683_68336


namespace NUMINAMATH_CALUDE_no_set_M_exists_l683_68398

theorem no_set_M_exists : ¬ ∃ (M : Set ℕ),
  (∀ m : ℕ, m > 1 → ∃ a b : ℕ, a ∈ M ∧ b ∈ M ∧ a + b = m) ∧
  (∀ a b c d : ℕ, a ∈ M → b ∈ M → c ∈ M → d ∈ M → 
    a > 10 → b > 10 → c > 10 → d > 10 → 
    a + b = c + d → (a = c ∨ a = d)) :=
by sorry

end NUMINAMATH_CALUDE_no_set_M_exists_l683_68398


namespace NUMINAMATH_CALUDE_angle_A_range_l683_68368

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the convexity property
def is_convex (q : Quadrilateral) : Prop := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle measure function
def angle_measure (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_A_range (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_AB : distance q.A q.B = 8)
  (h_BC : distance q.B q.C = 4)
  (h_CD : distance q.C q.D = 6)
  (h_DA : distance q.D q.A = 6) :
  0 < angle_measure q.B q.A q.D ∧ angle_measure q.B q.A q.D < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_range_l683_68368


namespace NUMINAMATH_CALUDE_flower_beds_count_l683_68305

/-- Given that there are 25 seeds in each flower bed and 750 seeds planted altogether,
    prove that the number of flower beds is 30. -/
theorem flower_beds_count (seeds_per_bed : ℕ) (total_seeds : ℕ) (num_beds : ℕ) 
    (h1 : seeds_per_bed = 25)
    (h2 : total_seeds = 750)
    (h3 : num_beds * seeds_per_bed = total_seeds) :
  num_beds = 30 := by
  sorry

end NUMINAMATH_CALUDE_flower_beds_count_l683_68305


namespace NUMINAMATH_CALUDE_five_fridays_september_implies_five_mondays_october_l683_68373

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month -/
structure Month where
  days : Nat
  first_day : DayOfWeek

/-- Given a day, returns the next day of the week -/
def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the occurrences of a specific day in a month -/
def count_day_occurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: If September has five Fridays, October has five Mondays -/
theorem five_fridays_september_implies_five_mondays_october 
  (september : Month) 
  (october : Month) :
  september.days = 30 →
  october.days = 31 →
  count_day_occurrences september DayOfWeek.Friday = 5 →
  count_day_occurrences october DayOfWeek.Monday = 5 :=
  sorry

end NUMINAMATH_CALUDE_five_fridays_september_implies_five_mondays_october_l683_68373


namespace NUMINAMATH_CALUDE_vector_operation_l683_68381

/-- Given vectors a and b in R², prove that 2a - b equals the expected result. -/
theorem vector_operation (a b : Fin 2 → ℝ) (h1 : a = ![2, 1]) (h2 : b = ![-3, 4]) :
  (2 • a) - b = ![7, -2] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l683_68381


namespace NUMINAMATH_CALUDE_coffee_ounces_per_cup_l683_68323

/-- Proves that the number of ounces of coffee per cup is 0.5 --/
theorem coffee_ounces_per_cup : 
  ∀ (people : ℕ) (cups_per_person_per_day : ℕ) (cost_per_ounce : ℚ) (weekly_expenditure : ℚ),
    people = 4 →
    cups_per_person_per_day = 2 →
    cost_per_ounce = 1.25 →
    weekly_expenditure = 35 →
    (weekly_expenditure / cost_per_ounce) / (people * cups_per_person_per_day * 7 : ℚ) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_coffee_ounces_per_cup_l683_68323


namespace NUMINAMATH_CALUDE_jean_friday_calls_l683_68308

/-- The number of calls Jean answered on each day of the week --/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The average number of calls per day --/
def average_calls : ℕ := 40

/-- The number of working days in a week --/
def working_days : ℕ := 5

/-- Jean's call data for the week --/
def jean_calls : WeekCalls := {
  monday := 35,
  tuesday := 46,
  wednesday := 27,
  thursday := 61,
  friday := 31  -- This is what we want to prove
}

/-- Theorem stating that Jean answered 31 calls on Friday --/
theorem jean_friday_calls : 
  jean_calls.friday = 31 :=
by sorry

end NUMINAMATH_CALUDE_jean_friday_calls_l683_68308


namespace NUMINAMATH_CALUDE_budget_calculation_l683_68324

theorem budget_calculation (initial_budget : ℕ) 
  (shirt_cost pants_cost coat_cost socks_cost belt_cost shoes_cost : ℕ) :
  initial_budget = 200 →
  shirt_cost = 30 →
  pants_cost = 46 →
  coat_cost = 38 →
  socks_cost = 11 →
  belt_cost = 18 →
  shoes_cost = 41 →
  initial_budget - (shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost + shoes_cost) = 16 := by
  sorry

end NUMINAMATH_CALUDE_budget_calculation_l683_68324


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_37_l683_68387

theorem smallest_four_digit_divisible_by_37 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 1036 → ¬(37 ∣ n)) ∧ 
  1000 ≤ 1036 ∧ 
  1036 < 10000 ∧ 
  37 ∣ 1036 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_37_l683_68387


namespace NUMINAMATH_CALUDE_equation_has_four_solutions_l683_68318

-- Define the equation
def equation (x : ℝ) : Prop := (2*x^2 - 10*x + 3)^2 = 4

-- State the theorem
theorem equation_has_four_solutions :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  equation a ∧ equation b ∧ equation c ∧ equation d ∧
  (∀ x : ℝ, equation x → (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
sorry

end NUMINAMATH_CALUDE_equation_has_four_solutions_l683_68318


namespace NUMINAMATH_CALUDE_min_value_fraction_l683_68399

theorem min_value_fraction (a : ℝ) (h1 : 0 < a) (h2 : a < 3) :
  1 / a + 9 / (3 - a) ≥ 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l683_68399


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l683_68340

/-- The speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 250) 
  (h2 : bridge_length = 300) 
  (h3 : crossing_time = 45) : 
  ∃ (speed : ℝ), abs (speed - (train_length + bridge_length) / crossing_time) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l683_68340


namespace NUMINAMATH_CALUDE_perfect_square_sums_l683_68319

theorem perfect_square_sums : ∃ (x y : ℕ+), 
  (∃ (a : ℕ+), (x + y : ℕ) = a^2) ∧ 
  (∃ (b : ℕ+), (x^2 + y^2 : ℕ) = b^2) ∧ 
  (∃ (c : ℕ+), (x^3 + y^3 : ℕ) = c^2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sums_l683_68319


namespace NUMINAMATH_CALUDE_sequence_sum_l683_68309

theorem sequence_sum (n : ℕ) (x : ℕ → ℚ) (h1 : x 1 = 1) 
  (h2 : ∀ k ∈ Finset.range (n - 1), x (k + 1) = x k + 1/2) : 
  Finset.sum (Finset.range n) (λ i => x (i + 1)) = (n^2 + 3*n) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l683_68309


namespace NUMINAMATH_CALUDE_line_equation_through_points_l683_68367

/-- A line passing through two points. -/
structure Line2D where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0. -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The equation of the line passing through (-1, 2) and (2, 5) is x - y + 3 = 0. -/
theorem line_equation_through_points :
  let l : Line2D := { point1 := (-1, 2), point2 := (2, 5) }
  let eq : LineEquation := { a := 1, b := -1, c := 3 }
  (∀ x y : ℝ, (x = l.point1.1 ∧ y = l.point1.2) ∨ (x = l.point2.1 ∧ y = l.point2.2) →
    eq.a * x + eq.b * y + eq.c = 0) ∧
  (∀ x y : ℝ, eq.a * x + eq.b * y + eq.c = 0 →
    ∃ t : ℝ, x = l.point1.1 + t * (l.point2.1 - l.point1.1) ∧
              y = l.point1.2 + t * (l.point2.2 - l.point1.2)) :=
by
  sorry


end NUMINAMATH_CALUDE_line_equation_through_points_l683_68367


namespace NUMINAMATH_CALUDE_remainder_of_n_l683_68302

theorem remainder_of_n (n : ℕ) 
  (h1 : n^2 % 7 = 3) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_l683_68302


namespace NUMINAMATH_CALUDE_tank_circumference_l683_68338

theorem tank_circumference (h_A h_B c_A : ℝ) (h_A_pos : h_A > 0) (h_B_pos : h_B > 0) (c_A_pos : c_A > 0) :
  h_A = 10 →
  h_B = 6 →
  c_A = 6 →
  (π * (c_A / (2 * π))^2 * h_A) = 0.6 * (π * (c_B / (2 * π))^2 * h_B) →
  c_B = 10 :=
by
  sorry

#check tank_circumference

end NUMINAMATH_CALUDE_tank_circumference_l683_68338


namespace NUMINAMATH_CALUDE_square_partition_exists_equilateral_triangle_partition_exists_l683_68356

-- Define a structure for a triangle
structure Triangle :=
  (a b c : ℝ)

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define what it means for two triangles to be congruent
def areCongruent (t1 t2 : Triangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

-- Define a structure for a square
structure Square :=
  (side : ℝ)

-- Define a structure for an equilateral triangle
structure EquilateralTriangle :=
  (side : ℝ)

-- Theorem for the square partition
theorem square_partition_exists (s : Square) : 
  ∃ (t1 t2 t3 t4 : Triangle), 
    (isIsosceles t1 ∧ isIsosceles t2 ∧ isIsosceles t3 ∧ isIsosceles t4) ∧
    (¬ areCongruent t1 t2 ∧ ¬ areCongruent t1 t3 ∧ ¬ areCongruent t1 t4 ∧
     ¬ areCongruent t2 t3 ∧ ¬ areCongruent t2 t4 ∧ ¬ areCongruent t3 t4) :=
sorry

-- Theorem for the equilateral triangle partition
theorem equilateral_triangle_partition_exists (et : EquilateralTriangle) : 
  ∃ (t1 t2 t3 t4 : Triangle), 
    (isIsosceles t1 ∧ isIsosceles t2 ∧ isIsosceles t3 ∧ isIsosceles t4) ∧
    (¬ areCongruent t1 t2 ∧ ¬ areCongruent t1 t3 ∧ ¬ areCongruent t1 t4 ∧
     ¬ areCongruent t2 t3 ∧ ¬ areCongruent t2 t4 ∧ ¬ areCongruent t3 t4) :=
sorry

end NUMINAMATH_CALUDE_square_partition_exists_equilateral_triangle_partition_exists_l683_68356


namespace NUMINAMATH_CALUDE_possible_values_of_a_l683_68328

-- Define the sets P and M
def P : Set ℝ := {x | x^2 = 1}
def M (a : ℝ) : Set ℝ := {x | a * x = 1}

-- Define the set of possible values for a
def A : Set ℝ := {1, -1, 0}

-- Statement to prove
theorem possible_values_of_a (a : ℝ) : M a ⊆ P → a ∈ A := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l683_68328


namespace NUMINAMATH_CALUDE_larger_box_jellybean_count_l683_68386

/-- The number of jellybeans in a box with dimensions thrice as large -/
def jellybeans_in_larger_box (original_capacity : ℕ) : ℕ :=
  original_capacity * 27

/-- Theorem: A box with dimensions thrice as large holds 4050 jellybeans -/
theorem larger_box_jellybean_count :
  jellybeans_in_larger_box 150 = 4050 := by
  sorry

end NUMINAMATH_CALUDE_larger_box_jellybean_count_l683_68386


namespace NUMINAMATH_CALUDE_midpoint_property_l683_68370

/-- Given two points A and B in a 2D plane, prove that if C is the midpoint of AB,
    then 2x - 4y = -15, where (x, y) are the coordinates of C. -/
theorem midpoint_property (A B C : ℝ × ℝ) : 
  A = (17, 10) → B = (-2, 5) → C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  2 * C.1 - 4 * C.2 = -15 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_property_l683_68370


namespace NUMINAMATH_CALUDE_lcm_problem_l683_68354

theorem lcm_problem (a b c : ℕ) (h1 : Nat.lcm a b = 945) (h2 : Nat.lcm b c = 525) :
  Nat.lcm a c = 675 ∨ Nat.lcm a c = 4725 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l683_68354


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l683_68374

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the foci of the hyperbola
def foci (F₁ F₂ : ℝ × ℝ) : Prop := ∃ c : ℝ, c > 0 ∧ F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop := is_on_hyperbola P.1 P.2

-- Define the right angle condition
def right_angle (F₁ P F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 =
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2

-- State the theorem
theorem hyperbola_triangle_area
  (F₁ F₂ P : ℝ × ℝ)
  (h_foci : foci F₁ F₂)
  (h_on_hyperbola : point_on_hyperbola P)
  (h_right_angle : right_angle F₁ P F₂) :
  (abs ((F₁.1 - P.1) * (F₂.2 - P.2) - (F₁.2 - P.2) * (F₂.1 - P.1))) / 2 = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l683_68374


namespace NUMINAMATH_CALUDE_mod_inverse_five_mod_thirtythree_l683_68372

theorem mod_inverse_five_mod_thirtythree :
  ∃ x : ℕ, x < 33 ∧ (5 * x) % 33 = 1 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_mod_inverse_five_mod_thirtythree_l683_68372


namespace NUMINAMATH_CALUDE_unique_intersection_points_l683_68315

/-- The first curve -/
def curve1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := -x^3 + 9 * x^2 - 4 * x + 2

/-- The intersection points of the two curves -/
def intersection_points : Set (ℝ × ℝ) := {(0, 2), (6, 86)}

/-- Theorem stating that the intersection_points are the only intersection points of curve1 and curve2 -/
theorem unique_intersection_points :
  ∀ x y : ℝ, curve1 x = curve2 x ∧ y = curve1 x ↔ (x, y) ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_unique_intersection_points_l683_68315


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l683_68307

theorem polynomial_functional_equation (a b c d : ℝ) :
  let f (x : ℝ) := a * x^3 + b * x^2 + c * x + d
  (∀ x, f x * f (-x) = f (x^3)) ↔ 
  ((a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨ (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l683_68307


namespace NUMINAMATH_CALUDE_point_distance_on_x_axis_l683_68301

theorem point_distance_on_x_axis (a : ℝ) : 
  let A : ℝ × ℝ := (a, 0)
  let B : ℝ × ℝ := (-3, 0)
  (‖A - B‖ = 5) → (a = -8 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_point_distance_on_x_axis_l683_68301


namespace NUMINAMATH_CALUDE_line_through_points_l683_68353

/-- A line passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  point1 : (ℝ × ℝ)
  point2 : (ℝ × ℝ)
  eq1 : a * point1.1 + b = point1.2
  eq2 : a * point2.1 + b = point2.2

/-- Theorem: For a line y = ax + b passing through (3, 4) and (7, 16), a - b = 8 -/
theorem line_through_points (l : Line) 
  (h1 : l.point1 = (3, 4))
  (h2 : l.point2 = (7, 16)) : 
  l.a - l.b = 8 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l683_68353


namespace NUMINAMATH_CALUDE_stamp_problem_l683_68342

/-- Represents the number of ways to make a certain amount with given coin denominations -/
def numWays (amount : ℕ) (coins : List ℕ) : ℕ :=
  sorry

/-- The minimum number of coins needed to make the amount -/
def minCoins (amount : ℕ) (coins : List ℕ) : ℕ :=
  sorry

theorem stamp_problem :
  let stamps := [5, 7]
  minCoins 50 stamps = 8 :=
by sorry

end NUMINAMATH_CALUDE_stamp_problem_l683_68342


namespace NUMINAMATH_CALUDE_spencer_walk_distance_l683_68335

theorem spencer_walk_distance (total : ℝ) (house_to_library : ℝ) (post_office_to_home : ℝ)
  (h1 : total = 0.8)
  (h2 : house_to_library = 0.3)
  (h3 : post_office_to_home = 0.4) :
  total - house_to_library - post_office_to_home = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_spencer_walk_distance_l683_68335


namespace NUMINAMATH_CALUDE_table_height_is_36_l683_68304

/-- Represents a cuboidal block of wood -/
structure Block where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Represents the arrangement in Figure 1 -/
def figure1 (b : Block) (table_height : ℝ) : ℝ :=
  b.length + table_height - b.depth

/-- Represents the arrangement in Figure 2 -/
def figure2 (b : Block) (table_height : ℝ) : ℝ :=
  2 * b.length + table_height

/-- Theorem stating the height of the table given the conditions -/
theorem table_height_is_36 (b : Block) (h : ℝ) :
  figure1 b h = 36 → figure2 b h = 46 → h = 36 := by
  sorry

#check table_height_is_36

end NUMINAMATH_CALUDE_table_height_is_36_l683_68304


namespace NUMINAMATH_CALUDE_inequality_solution_l683_68383

theorem inequality_solution (x : ℝ) : (2 - x < 1) ↔ (x > 1) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l683_68383


namespace NUMINAMATH_CALUDE_smallest_of_three_powers_l683_68394

theorem smallest_of_three_powers : 127^8 < 63^10 ∧ 63^10 < 33^12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_powers_l683_68394


namespace NUMINAMATH_CALUDE_matthew_water_bottle_fills_l683_68369

/-- Represents the number of times Matthew needs to fill his water bottle per week -/
def fill_times_per_week (glasses_per_day : ℕ) (ounces_per_glass : ℕ) (bottle_size : ℕ) : ℕ :=
  (7 * glasses_per_day * ounces_per_glass) / bottle_size

/-- Proves that Matthew will fill his water bottle 4 times per week -/
theorem matthew_water_bottle_fills :
  fill_times_per_week 4 5 35 = 4 := by
  sorry

end NUMINAMATH_CALUDE_matthew_water_bottle_fills_l683_68369


namespace NUMINAMATH_CALUDE_complex_modulus_inequality_l683_68306

open Complex

theorem complex_modulus_inequality (z : ℂ) (h : abs z = 1) :
  abs ((z + 1) + Complex.I * (7 - z)) ≠ 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_inequality_l683_68306


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l683_68316

theorem greatest_three_digit_number : ∃ n : ℕ, 
  n = 978 ∧ 
  n < 1000 ∧ 
  n > 99 ∧
  ∃ k : ℕ, n = 8 * k + 2 ∧
  ∃ m : ℕ, n = 7 * m + 4 ∧
  ∀ x : ℕ, x < 1000 ∧ x > 99 ∧ (∃ a : ℕ, x = 8 * a + 2) ∧ (∃ b : ℕ, x = 7 * b + 4) → x ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l683_68316


namespace NUMINAMATH_CALUDE_inequality_equivalence_l683_68303

theorem inequality_equivalence (y : ℝ) : 
  (7 / 36 + |y - 13 / 72| < 11 / 24) ↔ (-1 / 12 < y ∧ y < 4 / 9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l683_68303


namespace NUMINAMATH_CALUDE_discount_calculation_l683_68312

def list_price : ℝ := 70
def final_price : ℝ := 61.11
def first_discount : ℝ := 10

theorem discount_calculation (x : ℝ) :
  list_price * (1 - first_discount / 100) * (1 - x / 100) = final_price →
  x = 3 := by sorry

end NUMINAMATH_CALUDE_discount_calculation_l683_68312


namespace NUMINAMATH_CALUDE_pirates_walking_distance_l683_68395

/-- The number of miles walked per day on the first two islands -/
def miles_per_day_first_two_islands (
  num_islands : ℕ)
  (days_per_island : ℚ)
  (miles_per_day_last_two : ℕ)
  (total_miles : ℕ) : ℚ :=
  (total_miles - 2 * (miles_per_day_last_two * days_per_island)) /
  (2 * days_per_island)

/-- Theorem stating that the miles walked per day on the first two islands is 20 -/
theorem pirates_walking_distance :
  miles_per_day_first_two_islands 4 (3/2) 25 135 = 20 := by
  sorry

end NUMINAMATH_CALUDE_pirates_walking_distance_l683_68395


namespace NUMINAMATH_CALUDE_chord_bisected_at_P_l683_68378

/-- The equation of an ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 / 4 = 1

/-- A point is inside the ellipse if the left side of the equation is less than 1 -/
def inside_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 / 4 < 1

/-- The fixed point P -/
def P : ℝ × ℝ := (1, 1)

/-- A chord is bisected at a point if that point is the midpoint of the chord -/
def is_bisected_at (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

/-- The equation of a line -/
def line_equation (x y : ℝ) : Prop := 2 * x + y - 3 = 0

theorem chord_bisected_at_P :
  inside_ellipse P.1 P.2 →
  ∀ A B : ℝ × ℝ,
    ellipse A.1 A.2 →
    ellipse B.1 B.2 →
    is_bisected_at A B P →
    ∀ x y : ℝ, (x, y) ∈ Set.Icc A B → line_equation x y :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_at_P_l683_68378


namespace NUMINAMATH_CALUDE_digit_interchange_effect_l683_68333

theorem digit_interchange_effect (n : ℕ) (p q : ℕ) 
  (h1 : n = 9)
  (h2 : p < 10 ∧ q < 10)
  (h3 : p ≠ q)
  (original_sum : ℕ) 
  (new_sum : ℕ)
  (h4 : new_sum = original_sum - n)
  (h5 : new_sum = original_sum - (10*p + q - (10*q + p))) :
  p - q = 1 ∨ q - p = 1 :=
sorry

end NUMINAMATH_CALUDE_digit_interchange_effect_l683_68333


namespace NUMINAMATH_CALUDE_gcd_12740_220_minus_10_l683_68348

theorem gcd_12740_220_minus_10 : Nat.gcd 12740 220 - 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12740_220_minus_10_l683_68348


namespace NUMINAMATH_CALUDE_units_digit_square_equal_l683_68343

theorem units_digit_square_equal (a b : ℕ) (h : (a % 10 + b % 10) = 10) : 
  (a^2 % 10) = (b^2 % 10) := by
sorry

end NUMINAMATH_CALUDE_units_digit_square_equal_l683_68343


namespace NUMINAMATH_CALUDE_parabola_intersection_l683_68361

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 6 * x + 2
  let g (x : ℝ) := 9 * x^2 - 4 * x - 5
  (f (-7/3) = g (-7/3) ∧ f (-7/3) = 9) ∧
  (f (1/2) = g (1/2) ∧ f (1/2) = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l683_68361


namespace NUMINAMATH_CALUDE_trig_identity_l683_68391

theorem trig_identity : 
  (2 * Real.sin (46 * π / 180) - Real.sqrt 3 * Real.cos (74 * π / 180)) / Real.cos (16 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l683_68391


namespace NUMINAMATH_CALUDE_problem_statement_l683_68341

theorem problem_statement (x y m n : ℤ) 
  (hxy : x > y) 
  (hmn : m > n) 
  (hsum_xy : x + y = 7) 
  (hprod_xy : x * y = 12) 
  (hsum_mn : m + n = 13) 
  (hsum_squares : m^2 + n^2 = 97) : 
  (x - y) - (m - n) = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l683_68341


namespace NUMINAMATH_CALUDE_sally_weekday_pages_l683_68364

/-- The number of pages Sally reads on weekdays -/
def weekday_pages : ℕ := sorry

/-- The number of pages Sally reads on weekends -/
def weekend_pages : ℕ := 20

/-- The number of weeks it takes Sally to finish the book -/
def weeks_to_finish : ℕ := 2

/-- The total number of pages in the book -/
def total_pages : ℕ := 180

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

theorem sally_weekday_pages :
  weekday_pages = 10 :=
by sorry

end NUMINAMATH_CALUDE_sally_weekday_pages_l683_68364


namespace NUMINAMATH_CALUDE_function_domain_range_implies_a_value_l683_68351

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 2*a - 3)*x^2 + (a - 3)*x + 1

-- State the theorem
theorem function_domain_range_implies_a_value :
  (∀ x, ∃ y, f a x = y) → -- Domain is ℝ
  (∀ y, ∃ x, f a x = y) → -- Range is ℝ
  a = -1 := by sorry

end NUMINAMATH_CALUDE_function_domain_range_implies_a_value_l683_68351


namespace NUMINAMATH_CALUDE_distance_calculation_l683_68346

/-- The distance between A and B's homes from the city -/
def distance_difference : ℝ := 3

/-- The ratio of A's walking speed to B's walking speed -/
def walking_speed_ratio : ℝ := 1.5

/-- The ratio of B's truck speed to A's car speed -/
def vehicle_speed_ratio : ℝ := 1.5

/-- The ratio of A's car speed to A's walking speed -/
def car_to_walk_ratio : ℝ := 2

/-- B's distance from the city -/
def b_distance : ℝ := 13.5

/-- A's distance from the city -/
def a_distance : ℝ := 16.5

/-- The theorem stating that given the conditions, A lives 16.5 km from the city and B lives 13.5 km from the city -/
theorem distance_calculation :
  (a_distance - b_distance = distance_difference) ∧
  (a_distance = 16.5) ∧
  (b_distance = 13.5) := by
  sorry

#check distance_calculation

end NUMINAMATH_CALUDE_distance_calculation_l683_68346


namespace NUMINAMATH_CALUDE_smallest_prime_congruence_l683_68371

theorem smallest_prime_congruence : 
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p = 71 ∧ 
    ∀ (q : ℕ), Nat.Prime q → q < p → 
      ¬(∃ (q_inv : ℕ), (q * q_inv) % 143 = 1 ∧ (q + q_inv) % 143 = 25) ∧
    ∃ (p_inv : ℕ), (p * p_inv) % 143 = 1 ∧ (p + p_inv) % 143 = 25 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_congruence_l683_68371


namespace NUMINAMATH_CALUDE_card_combinations_l683_68360

def deck_size : ℕ := 60
def hand_size : ℕ := 15

theorem card_combinations :
  Nat.choose deck_size hand_size = 660665664066 := by
  sorry

end NUMINAMATH_CALUDE_card_combinations_l683_68360
