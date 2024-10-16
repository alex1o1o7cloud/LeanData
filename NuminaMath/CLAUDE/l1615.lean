import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_vector_of_parallel_lines_l1615_161504

/-- Given two parallel lines l and m in 2D space, this theorem proves that
    the vector perpendicular to both lines, normalized such that its
    components sum to 7, is (2, 5). -/
theorem perpendicular_vector_of_parallel_lines :
  ∀ (l m : ℝ → ℝ × ℝ),
  (∃ (k : ℝ), k ≠ 0 ∧ (l 0).1 - (l 1).1 = k * ((m 0).1 - (m 1).1) ∧
                    (l 0).2 - (l 1).2 = k * ((m 0).2 - (m 1).2)) →
  ∃ (v : ℝ × ℝ),
    v.1 + v.2 = 7 ∧
    v.1 * ((l 0).1 - (l 1).1) + v.2 * ((l 0).2 - (l 1).2) = 0 ∧
    v = (2, 5) :=
by sorry


end NUMINAMATH_CALUDE_perpendicular_vector_of_parallel_lines_l1615_161504


namespace NUMINAMATH_CALUDE_negative_quarter_to_11_times_negative_four_to_12_l1615_161556

theorem negative_quarter_to_11_times_negative_four_to_12 :
  (-0.25)^11 * (-4)^12 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_quarter_to_11_times_negative_four_to_12_l1615_161556


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l1615_161512

theorem scientific_notation_equality : 2912000 = 2.912 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l1615_161512


namespace NUMINAMATH_CALUDE_complement_intersection_equals_four_l1615_161563

-- Define the universe
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define sets M and N
def M : Set Nat := {1, 3, 5}
def N : Set Nat := {3, 4, 5}

-- State the theorem
theorem complement_intersection_equals_four :
  (U \ M) ∩ N = {4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_four_l1615_161563


namespace NUMINAMATH_CALUDE_available_storage_space_l1615_161532

/-- Represents a two-story warehouse with boxes stored on the second floor -/
structure Warehouse :=
  (second_floor_space : ℝ)
  (first_floor_space : ℝ)
  (box_space : ℝ)

/-- The conditions of the warehouse problem -/
def warehouse_conditions (w : Warehouse) : Prop :=
  w.first_floor_space = 2 * w.second_floor_space ∧
  w.box_space = w.second_floor_space / 4 ∧
  w.box_space = 5000

/-- The theorem stating the available storage space in the warehouse -/
theorem available_storage_space (w : Warehouse) 
  (h : warehouse_conditions w) : 
  w.first_floor_space + w.second_floor_space - w.box_space = 55000 := by
  sorry

end NUMINAMATH_CALUDE_available_storage_space_l1615_161532


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1615_161552

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  (1/3)^2 + (1/4)^2 + (1/6)^2 = (37 * x / 85) * ((1/5)^2 + (1/7)^2 + (1/8)^2) * y →
  Real.sqrt x / Real.sqrt y = 1737 / 857 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1615_161552


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l1615_161542

/-- Given a function f(x) = sin x + cos x with f'(x) = 3f(x), prove that tan 2x = -4/3 -/
theorem tan_double_angle_special_case (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.sin x + Real.cos x) 
  (h2 : ∀ x, deriv f x = 3 * f x) : 
  ∀ x, Real.tan (2 * x) = -4/3 := by sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l1615_161542


namespace NUMINAMATH_CALUDE_correct_matching_probability_l1615_161589

-- Define the number of students and pictures
def num_students : ℕ := 4

-- Define the function to calculate the factorial
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define the total number of possible arrangements
def total_arrangements : ℕ := factorial num_students

-- Define the number of correct arrangements
def correct_arrangements : ℕ := 1

-- State the theorem
theorem correct_matching_probability :
  (correct_arrangements : ℚ) / total_arrangements = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l1615_161589


namespace NUMINAMATH_CALUDE_coeff_x3_sum_l1615_161513

/-- The coefficient of x^3 in the expansion of (1-x)^n -/
def coeff_x3 (n : ℕ) : ℤ := (-1)^3 * Nat.choose n 3

/-- The sum of coefficients of x^3 in the expansion of (1-x)^5 + (1-x)^6 + (1-x)^7 + (1-x)^8 -/
def total_coeff : ℤ := coeff_x3 5 + coeff_x3 6 + coeff_x3 7 + coeff_x3 8

theorem coeff_x3_sum : total_coeff = -121 := by sorry

end NUMINAMATH_CALUDE_coeff_x3_sum_l1615_161513


namespace NUMINAMATH_CALUDE_five_by_five_not_coverable_l1615_161546

/-- Represents a checkerboard with width and height -/
structure Checkerboard :=
  (width : ℕ)
  (height : ℕ)

/-- Checks if a checkerboard can be covered by dominos -/
def can_be_covered_by_dominos (board : Checkerboard) : Prop :=
  (board.width * board.height) % 2 = 0 ∧
  (board.width * board.height) / 2 = (board.width * board.height + 1) / 2

theorem five_by_five_not_coverable :
  ¬(can_be_covered_by_dominos ⟨5, 5⟩) :=
by sorry

end NUMINAMATH_CALUDE_five_by_five_not_coverable_l1615_161546


namespace NUMINAMATH_CALUDE_x_is_integer_l1615_161557

theorem x_is_integer (x : ℝ) 
  (h1 : ∃ n : ℤ, x^1960 - x^1919 = n)
  (h2 : ∃ m : ℤ, x^2001 - x^1960 = m)
  (h3 : ∃ k : ℤ, x^2001 - x^1919 = k) : 
  ∃ z : ℤ, x = z := by
sorry

end NUMINAMATH_CALUDE_x_is_integer_l1615_161557


namespace NUMINAMATH_CALUDE_derek_added_water_l1615_161572

theorem derek_added_water (initial_amount final_amount : ℝ) (h1 : initial_amount = 3) (h2 : final_amount = 9.8) :
  final_amount - initial_amount = 6.8 := by
  sorry

end NUMINAMATH_CALUDE_derek_added_water_l1615_161572


namespace NUMINAMATH_CALUDE_corn_acres_l1615_161524

theorem corn_acres (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : beans_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) : 
  (total_land * corn_ratio) / (beans_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acres_l1615_161524


namespace NUMINAMATH_CALUDE_function_upper_bound_l1615_161553

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)

/-- A function that is bounded on [0,1] -/
def IsBoundedOnUnitInterval (f : ℝ → ℝ) : Prop :=
  ∃ M > 0, ∀ x, 0 ≤ x → x ≤ 1 → |f x| ≤ M

theorem function_upper_bound
    (f : ℝ → ℝ)
    (h1 : SatisfiesInequality f)
    (h2 : IsBoundedOnUnitInterval f) :
    ∀ x, x ≥ 0 → f x ≤ (1/2) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_upper_bound_l1615_161553


namespace NUMINAMATH_CALUDE_inequality_proof_l1615_161580

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_sum : a + b + c + d = 1) : 
  (a^2 / (1 + a)) + (b^2 / (1 + b)) + (c^2 / (1 + c)) + (d^2 / (1 + d)) ≥ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1615_161580


namespace NUMINAMATH_CALUDE_parabola_vertex_l1615_161587

/-- A parabola is defined by the equation y = -3(x-1)^2 - 2 -/
def parabola (x y : ℝ) : Prop := y = -3 * (x - 1)^2 - 2

/-- The vertex of a parabola is the point where it reaches its maximum or minimum -/
def is_vertex (x y : ℝ) : Prop := parabola x y ∧ ∀ x' y', parabola x' y' → y ≤ y'

/-- The vertex of the parabola y = -3(x-1)^2 - 2 is at (1, -2) -/
theorem parabola_vertex : is_vertex 1 (-2) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1615_161587


namespace NUMINAMATH_CALUDE_age_difference_l1615_161595

/-- Given three people a, b, and c, prove that a is 2 years older than b -/
theorem age_difference (a b c : ℕ) : 
  b = 2 * c →
  a + b + c = 22 →
  b = 8 →
  a - b = 2 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l1615_161595


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_in_range_l1615_161574

theorem quadratic_always_nonnegative_implies_a_in_range (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + a ≥ 0) → a ∈ Set.Icc 0 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_in_range_l1615_161574


namespace NUMINAMATH_CALUDE_train_passing_jogger_l1615_161547

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) 
  (h1 : jogger_speed = 9 * (1000 / 3600))
  (h2 : train_speed = 45 * (1000 / 3600))
  (h3 : train_length = 120)
  (h4 : initial_distance = 200) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 32 := by
  sorry

#check train_passing_jogger

end NUMINAMATH_CALUDE_train_passing_jogger_l1615_161547


namespace NUMINAMATH_CALUDE_two_different_expressions_equal_seven_l1615_161568

/-- An arithmetic expression using digits of 4 and basic operations -/
inductive Expr
  | four : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression -/
def eval : Expr → ℚ
  | Expr.four => 4
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Count the number of 4's used in an expression -/
def count_fours : Expr → ℕ
  | Expr.four => 1
  | Expr.add e1 e2 => count_fours e1 + count_fours e2
  | Expr.sub e1 e2 => count_fours e1 + count_fours e2
  | Expr.mul e1 e2 => count_fours e1 + count_fours e2
  | Expr.div e1 e2 => count_fours e1 + count_fours e2

/-- Check if two expressions are equivalent under commutative and associative properties -/
def are_equivalent : Expr → Expr → Prop := sorry

theorem two_different_expressions_equal_seven :
  ∃ (e1 e2 : Expr),
    eval e1 = 7 ∧
    eval e2 = 7 ∧
    count_fours e1 = 4 ∧
    count_fours e2 = 4 ∧
    ¬(are_equivalent e1 e2) :=
  sorry

end NUMINAMATH_CALUDE_two_different_expressions_equal_seven_l1615_161568


namespace NUMINAMATH_CALUDE_sam_initial_yellow_marbles_l1615_161543

/-- The number of yellow marbles Sam had initially -/
def initial_yellow_marbles : ℝ := 86.0

/-- The number of yellow marbles Joan gave to Sam -/
def joan_yellow_marbles : ℝ := 25.0

/-- The total number of yellow marbles Sam has now -/
def total_yellow_marbles : ℝ := 111

theorem sam_initial_yellow_marbles :
  initial_yellow_marbles + joan_yellow_marbles = total_yellow_marbles :=
by sorry

end NUMINAMATH_CALUDE_sam_initial_yellow_marbles_l1615_161543


namespace NUMINAMATH_CALUDE_function_value_implies_a_value_l1615_161516

noncomputable def f (a x : ℝ) : ℝ := x - Real.log (x + 2) + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem function_value_implies_a_value :
  ∃ (a x₀ : ℝ), f a x₀ = 3 → a = -Real.log 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_implies_a_value_l1615_161516


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l1615_161571

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem stating that the opposite of -2023 is 2023
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l1615_161571


namespace NUMINAMATH_CALUDE_mean_temperature_l1615_161520

def temperatures : List ℝ := [-7, -4, -4, -5, 1, 3, 2, 4]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : ℝ) = -1.25 := by
sorry

end NUMINAMATH_CALUDE_mean_temperature_l1615_161520


namespace NUMINAMATH_CALUDE_mersenne_factor_square_plus_nine_l1615_161539

theorem mersenne_factor_square_plus_nine (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ k : ℕ, n.val = 2^k :=
sorry

end NUMINAMATH_CALUDE_mersenne_factor_square_plus_nine_l1615_161539


namespace NUMINAMATH_CALUDE_projection_equals_three_l1615_161593

/-- Given vectors a and b in ℝ², with a specific angle between them, 
    prove that the projection of b onto a is 3. -/
theorem projection_equals_three (a b : ℝ × ℝ) (angle : ℝ) : 
  a = (1, Real.sqrt 3) → 
  b = (3, Real.sqrt 3) → 
  angle = π / 6 → 
  (b.1 * a.1 + b.2 * a.2) / Real.sqrt (a.1^2 + a.2^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_projection_equals_three_l1615_161593


namespace NUMINAMATH_CALUDE_function_monotonicity_l1615_161555

def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ f y ≤ f x

theorem function_monotonicity (f : ℝ → ℝ) 
  (h : ∀ a b x, a < x ∧ x < b → min (f a) (f b) < f x ∧ f x < max (f a) (f b)) :
  is_monotonic f := by
  sorry

end NUMINAMATH_CALUDE_function_monotonicity_l1615_161555


namespace NUMINAMATH_CALUDE_macaroon_weight_l1615_161523

theorem macaroon_weight (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) :
  total_macaroons = 12 →
  weight_per_macaroon = 5 →
  num_bags = 4 →
  total_macaroons % num_bags = 0 →
  (total_macaroons - total_macaroons / num_bags) * weight_per_macaroon = 45 := by
  sorry

end NUMINAMATH_CALUDE_macaroon_weight_l1615_161523


namespace NUMINAMATH_CALUDE_intersection_x_sum_l1615_161530

/-- The sum of x-coordinates of intersection points of two congruences -/
theorem intersection_x_sum : ∃ (S : Finset ℤ),
  (∀ x ∈ S, ∃ y : ℤ, 
    (y ≡ 7*x + 3 [ZMOD 20] ∧ y ≡ 13*x + 17 [ZMOD 20]) ∧
    (x ≥ 0 ∧ x < 20)) ∧
  (∀ x : ℤ, x ≥ 0 → x < 20 →
    (∃ y : ℤ, y ≡ 7*x + 3 [ZMOD 20] ∧ y ≡ 13*x + 17 [ZMOD 20]) →
    x ∈ S) ∧
  S.sum id = 12 :=
sorry

end NUMINAMATH_CALUDE_intersection_x_sum_l1615_161530


namespace NUMINAMATH_CALUDE_double_mean_value_function_range_l1615_161510

/-- Definition of a double mean value function -/
def is_double_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    (deriv^[2] f x₁ = (f b - f a) / (b - a)) ∧
    (deriv^[2] f x₂ = (f b - f a) / (b - a))

/-- The main theorem -/
theorem double_mean_value_function_range (a : ℝ) (m : ℝ) :
  is_double_mean_value_function (fun x => 2 * x^3 - x^2 + m) 0 (2 * a) →
  1/8 < a ∧ a < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_double_mean_value_function_range_l1615_161510


namespace NUMINAMATH_CALUDE_total_pencils_l1615_161511

theorem total_pencils (jessica_pencils sandy_pencils jason_pencils : ℕ) 
  (h1 : jessica_pencils = 8)
  (h2 : sandy_pencils = 8)
  (h3 : jason_pencils = 8) :
  jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l1615_161511


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1615_161518

theorem inequality_system_solution (x : ℝ) : 
  (7 - 2*(x + 1) ≥ 1 - 6*x ∧ (1 + 2*x) / 3 > x - 1) ↔ -1 ≤ x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1615_161518


namespace NUMINAMATH_CALUDE_percent_of_x_is_v_l1615_161528

theorem percent_of_x_is_v (x y z v : ℝ) 
  (h1 : 0.45 * z = 0.39 * y)
  (h2 : y = 0.75 * x)
  (h3 : v = 0.8 * z) :
  v = 0.52 * x :=
by sorry

end NUMINAMATH_CALUDE_percent_of_x_is_v_l1615_161528


namespace NUMINAMATH_CALUDE_decimal_to_octal_conversion_l1615_161564

/-- Converts a natural number to its octal representation -/
def toOctal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: toOctal (n / 8)

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 521

/-- The expected octal representation -/
def expectedOctal : List ℕ := [1, 1, 0, 1]

theorem decimal_to_octal_conversion :
  toOctal decimalNumber = expectedOctal := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_octal_conversion_l1615_161564


namespace NUMINAMATH_CALUDE_equation_has_real_root_when_K_zero_l1615_161517

/-- The equation x = K³(x³ - 3x² + 2x + 1) has at least one real root when K = 0 -/
theorem equation_has_real_root_when_K_zero :
  ∃ x : ℝ, x = 0^3 * (x^3 - 3*x^2 + 2*x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_when_K_zero_l1615_161517


namespace NUMINAMATH_CALUDE_cubic_expression_value_l1615_161594

theorem cubic_expression_value (r s : ℝ) : 
  3 * r^2 - 4 * r - 7 = 0 →
  3 * s^2 - 4 * s - 7 = 0 →
  r ≠ s →
  (3 * r^3 - 3 * s^3) / (r - s) = 37 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l1615_161594


namespace NUMINAMATH_CALUDE_union_of_sets_l1615_161562

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 3}
  A ∪ B = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1615_161562


namespace NUMINAMATH_CALUDE_randys_trip_length_l1615_161598

theorem randys_trip_length :
  ∀ (total_length : ℝ),
  (total_length / 2 : ℝ) + 30 + (total_length / 4 : ℝ) = total_length →
  total_length = 120 := by
sorry

end NUMINAMATH_CALUDE_randys_trip_length_l1615_161598


namespace NUMINAMATH_CALUDE_difference_of_squares_l1615_161533

theorem difference_of_squares (m : ℝ) : m^2 - 144 = (m - 12) * (m + 12) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1615_161533


namespace NUMINAMATH_CALUDE_function_property_l1615_161502

/-- Given a function f and a real number a, if f(a) + f(1) = 0, then a = -3 -/
theorem function_property (f : ℝ → ℝ) (a : ℝ) (h : f a + f 1 = 0) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1615_161502


namespace NUMINAMATH_CALUDE_circle_symmetry_implies_m_equals_one_l1615_161527

/-- A circle with equation x^2 + y^2 + 2x - 4y = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - 4*p.2 = 0}

/-- A line with equation 3x + y + m = 0 -/
def Line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3*p.1 + p.2 + m = 0}

/-- The center of a circle -/
def center (c : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Symmetry of a circle about a line -/
def isSymmetric (c : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Prop := sorry

theorem circle_symmetry_implies_m_equals_one :
  isSymmetric Circle (Line m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_implies_m_equals_one_l1615_161527


namespace NUMINAMATH_CALUDE_triangle_two_solutions_l1615_161534

theorem triangle_two_solutions (a b : ℝ) (A B : ℝ) :
  b = 2 →
  B = π / 4 →
  (∃ (C : ℝ), 0 < C ∧ C < π ∧ A + B + C = π ∧ a / Real.sin A = b / Real.sin B) →
  (∃ (C' : ℝ), 0 < C' ∧ C' < π ∧ C' ≠ C ∧ A + B + C' = π ∧ a / Real.sin A = b / Real.sin B) →
  2 < a ∧ a < 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_two_solutions_l1615_161534


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1615_161535

theorem trigonometric_identities (α : Real) 
  (h : (1 + Real.tan α) / (1 - Real.tan α) = 2) : 
  (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 5 ∧ 
  Real.sin α * Real.cos α + 2 = 23 / 10 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1615_161535


namespace NUMINAMATH_CALUDE_truck_rental_problem_l1615_161501

/-- The total number of trucks on Monday morning -/
def total_trucks : ℕ := 30

/-- The number of trucks rented out during the week -/
def rented_trucks : ℕ := 20

/-- The number of trucks returned by Saturday morning -/
def returned_trucks : ℕ := rented_trucks / 2

/-- The number of trucks on the lot Saturday morning -/
def saturday_trucks : ℕ := returned_trucks

theorem truck_rental_problem :
  (returned_trucks = rented_trucks / 2) →
  (saturday_trucks ≥ 10) →
  (rented_trucks = 20) →
  (total_trucks = rented_trucks + (rented_trucks - returned_trucks)) :=
by sorry

end NUMINAMATH_CALUDE_truck_rental_problem_l1615_161501


namespace NUMINAMATH_CALUDE_original_paint_intensity_l1615_161569

/-- Proves that the original paint intensity was 50% given the mixing conditions --/
theorem original_paint_intensity 
  (replaced_fraction : ℚ) 
  (replacement_intensity : ℚ) 
  (final_intensity : ℚ) : 
  replaced_fraction = 2/3 → 
  replacement_intensity = 1/5 → 
  final_intensity = 3/10 → 
  (1 - replaced_fraction) * (1/2) + replaced_fraction * replacement_intensity = final_intensity := by
  sorry

#eval (1 - 2/3) * (1/2) + 2/3 * (1/5) == 3/10

end NUMINAMATH_CALUDE_original_paint_intensity_l1615_161569


namespace NUMINAMATH_CALUDE_square_diagonals_equal_l1615_161519

/-- A structure representing a parallelogram -/
structure Parallelogram :=
  (diagonals_equal : Bool)

/-- A structure representing a square, which is a special case of a parallelogram -/
structure Square extends Parallelogram

/-- Theorem stating that the diagonals of a parallelogram are equal -/
axiom parallelogram_diagonals_equal :
  ∀ (p : Parallelogram), p.diagonals_equal = true

/-- Theorem stating that a square is a parallelogram -/
axiom square_is_parallelogram :
  ∀ (s : Square), ∃ (p : Parallelogram), s = ⟨p⟩

/-- Theorem to prove: The diagonals of a square are equal -/
theorem square_diagonals_equal (s : Square) :
  s.diagonals_equal = true := by sorry

end NUMINAMATH_CALUDE_square_diagonals_equal_l1615_161519


namespace NUMINAMATH_CALUDE_center_value_of_arithmetic_array_l1615_161514

/-- Represents a 4x4 array where each row and column is an arithmetic sequence -/
def ArithmeticArray := Fin 4 → Fin 4 → ℚ

/-- The common difference of an arithmetic sequence given its first and last terms -/
def commonDifference (a₁ a₄ : ℚ) : ℚ := (a₄ - a₁) / 3

/-- Checks if a sequence is arithmetic -/
def isArithmeticSequence (seq : Fin 4 → ℚ) : Prop :=
  ∀ i j : Fin 4, i.val < j.val → seq j - seq i = commonDifference (seq 0) (seq 3) * (j - i)

/-- Properties of our specific arithmetic array -/
def isValidArray (arr : ArithmeticArray) : Prop :=
  (∀ i : Fin 4, isArithmeticSequence (λ j => arr i j)) ∧  -- Each row is arithmetic
  (∀ j : Fin 4, isArithmeticSequence (λ i => arr i j)) ∧  -- Each column is arithmetic
  arr 0 0 = 3 ∧ arr 0 3 = 21 ∧                            -- First row conditions
  arr 3 0 = 15 ∧ arr 3 3 = 45                             -- Fourth row conditions

theorem center_value_of_arithmetic_array (arr : ArithmeticArray) 
  (h : isValidArray arr) : arr 1 1 = 14 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_center_value_of_arithmetic_array_l1615_161514


namespace NUMINAMATH_CALUDE_triangle_ratio_l1615_161551

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2b * sin(2A) = 3a * sin(B) and c = 2b, then a/b = √2 -/
theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  2 * b * Real.sin (2 * A) = 3 * a * Real.sin B →
  c = 2 * b →
  a / b = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l1615_161551


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1615_161583

theorem smaller_number_problem (x y : ℤ) : 
  x + y = 64 → y = x + 12 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1615_161583


namespace NUMINAMATH_CALUDE_six_people_arrangement_l1615_161592

/-- The number of ways to arrange n people in a line with two specific people always next to each other -/
def arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- Theorem: For 6 people with two specific people always next to each other, there are 240 possible arrangements -/
theorem six_people_arrangement : arrangements 6 = 240 := by
  sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l1615_161592


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1615_161549

theorem sum_with_radical_conjugate : 
  let x : ℝ := 16 - Real.sqrt 2023
  let y : ℝ := 16 + Real.sqrt 2023
  x + y = 32 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1615_161549


namespace NUMINAMATH_CALUDE_cylinder_volume_l1615_161579

/-- The volume of a cylinder with height 4 and circular faces with circumference 10π is 100π. -/
theorem cylinder_volume (h : ℝ) (c : ℝ) (v : ℝ) : 
  h = 4 → c = 10 * Real.pi → v = Real.pi * (c / (2 * Real.pi))^2 * h → v = 100 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l1615_161579


namespace NUMINAMATH_CALUDE_triangle_height_calculation_l1615_161558

theorem triangle_height_calculation (base area height : Real) : 
  base = 8.4 → area = 24.36 → area = (base * height) / 2 → height = 5.8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_calculation_l1615_161558


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l1615_161508

theorem smallest_number_divisible (n : ℕ) : n = 92160 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 37 * 47 * 53 * k)) ∧ 
  (∃ k : ℕ, (n + 7) = 37 * 47 * 53 * k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l1615_161508


namespace NUMINAMATH_CALUDE_ellipse_properties_l1615_161570

-- Define the ellipse C
def ellipse_C (x y a b c : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0 ∧ a = 2*c

-- Define the circle P1
def circle_P1 (x y r : ℝ) : Prop :=
  (x + 4*Real.sqrt 3 / 7)^2 + (y - 3*Real.sqrt 3 / 7)^2 = r^2 ∧ r > 0

-- Define the theorem
theorem ellipse_properties :
  ∀ (a b c : ℝ),
  ellipse_C (Real.sqrt 3) ((Real.sqrt 3) / 2) a b c →
  ellipse_C (-a + 2*c) 0 a b c →
  (∃ (x y r : ℝ), circle_P1 x y r ∧ ellipse_C x y a b c) →
  (∃ (k : ℝ), k > 1 ∧
    (∀ (x y : ℝ), y = k*(x + 1) → 
      (∃ (p q : ℝ), ellipse_C p (k*(p + 1)) a b c ∧ 
                    ellipse_C q (k*(q + 1)) a b c ∧
                    9/4 < (1 + k^2) * (9 / (3 + 4*k^2)) ∧
                    (1 + k^2) * (9 / (3 + 4*k^2)) ≤ 12/5))) →
  c / a = 1/2 ∧ a = 2 ∧ b = Real.sqrt 3 :=
sorry


end NUMINAMATH_CALUDE_ellipse_properties_l1615_161570


namespace NUMINAMATH_CALUDE_customers_left_l1615_161596

theorem customers_left (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 14 → new = 39 → final = 50 → initial - (initial - final + new) = 3 := by
  sorry

end NUMINAMATH_CALUDE_customers_left_l1615_161596


namespace NUMINAMATH_CALUDE_airplane_seats_l1615_161540

theorem airplane_seats (total_seats : ℕ) (first_class : ℕ) : 
  total_seats = 567 → 
  first_class + 3 * first_class + (7 * first_class + 5) = total_seats →
  7 * first_class + 5 = 362 := by
sorry

end NUMINAMATH_CALUDE_airplane_seats_l1615_161540


namespace NUMINAMATH_CALUDE_pet_shop_kittens_l1615_161537

theorem pet_shop_kittens (num_puppies : ℕ) (puppy_cost kitten_cost total_stock : ℚ) : 
  num_puppies = 2 →
  puppy_cost = 20 →
  kitten_cost = 15 →
  total_stock = 100 →
  (total_stock - num_puppies * puppy_cost) / kitten_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_pet_shop_kittens_l1615_161537


namespace NUMINAMATH_CALUDE_banana_arrangements_l1615_161507

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (letterFrequencies : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (letterFrequencies.map Nat.factorial).prod

/-- The number of distinct arrangements of the letters in "banana" -/
def bananaArrangements : ℕ :=
  distinctArrangements 6 [1, 2, 3]

theorem banana_arrangements :
  bananaArrangements = 60 := by
  sorry

#eval bananaArrangements

end NUMINAMATH_CALUDE_banana_arrangements_l1615_161507


namespace NUMINAMATH_CALUDE_infinite_divisibility_by_prime_l1615_161536

theorem infinite_divisibility_by_prime (p : ℕ) (hp : Prime p) :
  Set.Infinite {n : ℕ | n > 0 ∧ p ∣ (2^n - n)} :=
sorry

end NUMINAMATH_CALUDE_infinite_divisibility_by_prime_l1615_161536


namespace NUMINAMATH_CALUDE_rotation_180_transforms_rectangle_l1615_161521

-- Define the points of rectangle ABCD
def A : ℝ × ℝ := (-3, 2)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (-1, 5)
def D : ℝ × ℝ := (-3, 5)

-- Define the points of rectangle A'B'C'D'
def A' : ℝ × ℝ := (3, -2)
def B' : ℝ × ℝ := (1, -2)
def C' : ℝ × ℝ := (1, -5)
def D' : ℝ × ℝ := (3, -5)

-- Define the 180° rotation transformation
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement
theorem rotation_180_transforms_rectangle :
  rotate180 A = A' ∧
  rotate180 B = B' ∧
  rotate180 C = C' ∧
  rotate180 D = D' := by
  sorry


end NUMINAMATH_CALUDE_rotation_180_transforms_rectangle_l1615_161521


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_fraction_equals_3_5_l1615_161575

theorem tan_alpha_2_implies_fraction_equals_3_5 (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_fraction_equals_3_5_l1615_161575


namespace NUMINAMATH_CALUDE_meaningful_reciprocal_range_l1615_161503

theorem meaningful_reciprocal_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x + 2)) ↔ x ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_reciprocal_range_l1615_161503


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l1615_161578

theorem basketball_lineup_combinations (n : Nat) (k : Nat) (m : Nat) : 
  n = 20 → k = 13 → m = 1 →
  n * Nat.choose (n - 1) (k - m) = 1007760 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l1615_161578


namespace NUMINAMATH_CALUDE_absent_children_absent_children_solution_l1615_161599

theorem absent_children (total_children : ℕ) (initial_bananas_per_child : ℕ) (extra_bananas : ℕ) : ℕ :=
  let total_bananas := total_children * initial_bananas_per_child
  let final_bananas_per_child := initial_bananas_per_child + extra_bananas
  let absent_children := total_children - (total_bananas / final_bananas_per_child)
  absent_children

theorem absent_children_solution :
  absent_children 320 2 2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_absent_children_absent_children_solution_l1615_161599


namespace NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l1615_161586

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_sixth : 12 / (1 / 6) = 72 := by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l1615_161586


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_sqrt_three_l1615_161554

theorem trigonometric_expression_equals_sqrt_three (α : Real) (h : α = -35 * Real.pi / 6) :
  (2 * Real.sin (Real.pi + α) * Real.cos (Real.pi - α) - Real.cos (Real.pi + α)) /
  (1 + Real.sin α ^ 2 + Real.sin (Real.pi - α) - Real.cos (Real.pi + α) ^ 2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_sqrt_three_l1615_161554


namespace NUMINAMATH_CALUDE_joy_reading_rate_l1615_161559

/-- Represents Joy's reading rate in pages per hour -/
def reading_rate (pages_per_20min : ℕ) (pages_per_5hours : ℕ) : ℚ :=
  (pages_per_20min * 3)

/-- Theorem stating Joy's reading rate is 24 pages per hour -/
theorem joy_reading_rate :
  reading_rate 8 120 = 24 := by sorry

end NUMINAMATH_CALUDE_joy_reading_rate_l1615_161559


namespace NUMINAMATH_CALUDE_parallelogram_uniqueness_l1615_161582

/-- Represents a parallelogram in 2D space -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents a point in 2D space -/
structure Point :=
  (x y : ℝ)

/-- The measure of an angle in radians -/
def Angle := ℝ

/-- The length of a line segment -/
def Length := ℝ

/-- Checks if two parallelograms are congruent -/
def are_congruent (p1 p2 : Parallelogram) : Prop :=
  sorry

/-- Constructs a parallelogram given the required parameters -/
def construct_parallelogram (α ε : Angle) (bd : Length) : Parallelogram :=
  sorry

/-- Theorem stating the uniqueness of the constructed parallelogram -/
theorem parallelogram_uniqueness (α ε : Angle) (bd : Length) :
  ∀ p1 p2 : Parallelogram,
    (p1 = construct_parallelogram α ε bd) →
    (p2 = construct_parallelogram α ε bd) →
    are_congruent p1 p2 :=
  sorry

end NUMINAMATH_CALUDE_parallelogram_uniqueness_l1615_161582


namespace NUMINAMATH_CALUDE_circle_center_tangent_parabola_l1615_161561

/-- A circle that passes through (1,0) and is tangent to y = x^2 at (1,1) has its center at (1,1) -/
theorem circle_center_tangent_parabola : 
  ∀ (center : ℝ × ℝ),
  (∀ (p : ℝ × ℝ), p.1^2 = p.2 → (center.1 - p.1)^2 + (center.2 - p.2)^2 = (center.1 - 1)^2 + center.2^2) →
  (center.1 - 1)^2 + (center.2 - 1)^2 = (center.1 - 1)^2 + center.2^2 →
  center = (1, 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_tangent_parabola_l1615_161561


namespace NUMINAMATH_CALUDE_max_b_when_a_is_e_min_a_minus_b_l1615_161590

open Real

-- Define the condition that e^x ≥ ax + b for all x
def condition (a b : ℝ) : Prop := ∀ x, exp x ≥ a * x + b

theorem max_b_when_a_is_e :
  (condition e b) → b ≤ 0 :=
sorry

theorem min_a_minus_b :
  ∃ a b, condition a b ∧ ∀ a' b', condition a' b' → a - b ≤ a' - b' ∧ a - b = -1/e :=
sorry

end NUMINAMATH_CALUDE_max_b_when_a_is_e_min_a_minus_b_l1615_161590


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1615_161545

theorem consecutive_integers_sum (n : ℕ) : 
  (n + 2 = 9) → (n + (n + 1) + (n + 2) = 24) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1615_161545


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l1615_161567

theorem complementary_angles_difference (a b : Real) : 
  a + b = 90 →  -- angles are complementary
  a / b = 5 / 4 →  -- ratio of angles is 5:4
  (max a b - min a b) = 10 :=  -- positive difference is 10
by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l1615_161567


namespace NUMINAMATH_CALUDE_increasing_quadratic_implies_a_bound_l1615_161560

/-- Given a quadratic function f(x) = 2x^2 - 4(1-a)x + 1, 
    if f is increasing on [3,+∞), then a ≥ -2 -/
theorem increasing_quadratic_implies_a_bound (a : ℝ) : 
  (∀ x ≥ 3, ∀ y ≥ x, (2*y^2 - 4*(1-a)*y + 1) ≥ (2*x^2 - 4*(1-a)*x + 1)) →
  a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_increasing_quadratic_implies_a_bound_l1615_161560


namespace NUMINAMATH_CALUDE_odd_function_property_l1615_161529

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_even : is_even (fun x ↦ f (x + 2))) 
  (h_f_neg_one : f (-1) = 1) : 
  f 2017 + f 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1615_161529


namespace NUMINAMATH_CALUDE_adam_new_books_l1615_161526

theorem adam_new_books (initial_books sold_books final_books : ℕ) 
  (h1 : initial_books = 33) 
  (h2 : sold_books = 11)
  (h3 : final_books = 45) :
  final_books - (initial_books - sold_books) = 23 := by
  sorry

end NUMINAMATH_CALUDE_adam_new_books_l1615_161526


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1615_161585

theorem arithmetic_calculation : 10 * (1/8) - 6.4 / 8 + 1.2 * 0.125 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1615_161585


namespace NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l1615_161573

/-- A triangle in a 2D plane --/
structure Triangle :=
  (A B C : EuclideanSpace ℝ (Fin 2))

/-- The circumcenter of a triangle --/
def circumcenter (t : Triangle) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- Reflect a point across a line defined by two points --/
def reflect_point (p q r : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- Given three points that are reflections of a fourth point across the sides of a triangle,
    reconstruct the original triangle --/
def reconstruct_triangle (O1 O2 O3 : EuclideanSpace ℝ (Fin 2)) : Triangle :=
  sorry

theorem triangle_reconstruction_uniqueness 
  (t : Triangle) 
  (O : EuclideanSpace ℝ (Fin 2)) 
  (hO : O = circumcenter t) 
  (O1 : EuclideanSpace ℝ (Fin 2)) 
  (hO1 : O1 = reflect_point O t.B t.C) 
  (O2 : EuclideanSpace ℝ (Fin 2)) 
  (hO2 : O2 = reflect_point O t.C t.A) 
  (O3 : EuclideanSpace ℝ (Fin 2)) 
  (hO3 : O3 = reflect_point O t.A t.B) :
  reconstruct_triangle O1 O2 O3 = t :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l1615_161573


namespace NUMINAMATH_CALUDE_ticket_sales_difference_l1615_161577

/-- Proves the difference in ticket sales given ticket prices and total sales -/
theorem ticket_sales_difference (student_price non_student_price : ℕ) 
  (total_sales total_tickets : ℕ) : 
  student_price = 6 →
  non_student_price = 9 →
  total_sales = 10500 →
  total_tickets = 1700 →
  ∃ (student_tickets non_student_tickets : ℕ),
    student_tickets + non_student_tickets = total_tickets ∧
    student_price * student_tickets + non_student_price * non_student_tickets = total_sales ∧
    student_tickets - non_student_tickets = 1500 :=
by sorry

end NUMINAMATH_CALUDE_ticket_sales_difference_l1615_161577


namespace NUMINAMATH_CALUDE_sequence_properties_l1615_161548

def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 2 * a n + n

def b (n : ℕ) (a : ℕ → ℝ) : ℝ := n * (1 - a n)

def geometric_sequence (u : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, u (n + 1) = r * u n

def sum_of_sequence (u : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum u

theorem sequence_properties (a : ℕ → ℝ) :
  (∀ n : ℕ, S n a = 2 * a n + n) →
  (geometric_sequence (λ n => a n - 1)) ∧
  (∀ n : ℕ, sum_of_sequence (b · a) n = (n - 1) * 2^(n + 1) + 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1615_161548


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l1615_161565

theorem arithmetic_sequence_count : 
  let a₁ : ℝ := 2.5
  let d : ℝ := 5
  let aₙ : ℝ := 62.5
  (aₙ - a₁) / d + 1 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l1615_161565


namespace NUMINAMATH_CALUDE_min_components_for_reliability_l1615_161515

/-- The probability of a single component working properly -/
def p : ℝ := 0.5

/-- The minimum required probability for the entire circuit to work properly -/
def min_prob : ℝ := 0.95

/-- The function that calculates the probability of the entire circuit working properly -/
def circuit_prob (n : ℕ) : ℝ := 1 - p^n

/-- Theorem stating the minimum number of components required -/
theorem min_components_for_reliability :
  ∃ n : ℕ, (∀ m : ℕ, m < n → circuit_prob m < min_prob) ∧ circuit_prob n ≥ min_prob :=
sorry

end NUMINAMATH_CALUDE_min_components_for_reliability_l1615_161515


namespace NUMINAMATH_CALUDE_max_cos_sum_in_triangle_l1615_161550

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_angles : 0 < angleA ∧ 0 < angleB ∧ 0 < angleC
  h_sum_angles : angleA + angleB + angleC = π
  h_cosine_law : b^2 + c^2 - a^2 = b * c

-- Theorem statement
theorem max_cos_sum_in_triangle (t : Triangle) : 
  ∃ (max : ℝ), max = 1 ∧ ∀ (x : ℝ), x = Real.cos t.angleB + Real.cos t.angleC → x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_cos_sum_in_triangle_l1615_161550


namespace NUMINAMATH_CALUDE_binomial_30_3_squared_l1615_161538

theorem binomial_30_3_squared : (Nat.choose 30 3)^2 = 16483600 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_squared_l1615_161538


namespace NUMINAMATH_CALUDE_sin_in_M_l1615_161597

/-- The set of functions f that satisfy f(x + T) = T * f(x) for some non-zero constant T -/
def M : Set (ℝ → ℝ) :=
  {f | ∃ (T : ℝ), T ≠ 0 ∧ ∀ x, f (x + T) = T * f x}

/-- Theorem stating the condition for sin(kx) to be in set M -/
theorem sin_in_M (k : ℝ) : 
  (fun x ↦ Real.sin (k * x)) ∈ M ↔ ∃ m : ℤ, k = m * Real.pi :=
sorry

end NUMINAMATH_CALUDE_sin_in_M_l1615_161597


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l1615_161506

/-- The equation of a hyperbola with one focus at (-3,0) -/
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 - y^2 / m = 1

/-- The focus of the hyperbola is at (-3,0) -/
def focus_at_minus_three : ℝ × ℝ := (-3, 0)

/-- Theorem stating that m = 8 for the given hyperbola -/
theorem hyperbola_m_value :
  ∃ (m : ℝ), (∀ (x y : ℝ), hyperbola_equation x y m) ∧ 
  (focus_at_minus_three.1 = -3) ∧ (focus_at_minus_three.2 = 0) →
  m = 8 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l1615_161506


namespace NUMINAMATH_CALUDE_triangle_properties_l1615_161584

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (3 * Real.cos B * Real.cos C + 1 = 3 * Real.sin B * Real.sin C + Real.cos (2 * A)) →
  (A = π / 3) ∧
  (a = 2 * Real.sqrt 3 → ∃ (max_value : ℝ), max_value = 4 * Real.sqrt 7 ∧
    ∀ (b' c' : ℝ), b' + 2 * c' ≤ max_value) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1615_161584


namespace NUMINAMATH_CALUDE_point_on_line_expression_l1615_161505

/-- For any point (a,b) on the line y = 4x + 3, the expression 4a - b - 2 equals -5 -/
theorem point_on_line_expression (a b : ℝ) : b = 4 * a + 3 → 4 * a - b - 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_expression_l1615_161505


namespace NUMINAMATH_CALUDE_reciprocal_of_point_six_repeating_l1615_161541

def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  d / (1 - (1/10))

theorem reciprocal_of_point_six_repeating :
  (repeating_decimal_to_fraction (6/10))⁻¹ = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_point_six_repeating_l1615_161541


namespace NUMINAMATH_CALUDE_parabola_fixed_point_l1615_161591

theorem parabola_fixed_point (u : ℝ) : 
  let f : ℝ → ℝ := λ x => 5 * x^2 + u * x + 3 * u
  f (-3) = 45 := by
sorry

end NUMINAMATH_CALUDE_parabola_fixed_point_l1615_161591


namespace NUMINAMATH_CALUDE_jackies_lotion_order_l1615_161588

/-- The number of lotion bottles Jackie ordered -/
def lotion_bottles : ℕ := 3

/-- The free shipping threshold in cents -/
def free_shipping_threshold : ℕ := 5000

/-- The total cost of shampoo and conditioner in cents -/
def shampoo_conditioner_cost : ℕ := 2000

/-- The cost of one bottle of lotion in cents -/
def lotion_cost : ℕ := 600

/-- The additional amount Jackie needs to spend to reach the free shipping threshold in cents -/
def additional_spend : ℕ := 1200

theorem jackies_lotion_order :
  lotion_bottles * lotion_cost = free_shipping_threshold - shampoo_conditioner_cost - additional_spend :=
by sorry

end NUMINAMATH_CALUDE_jackies_lotion_order_l1615_161588


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1615_161566

-- Define the quadratic function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Define the linear function g
def g (x : ℝ) : ℝ := x - 1

-- Theorem statement
theorem quadratic_inequality (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 1 < x ∧ x < 2) →
  (a = 3 ∧ b = 2) ∧
  (∀ c : ℝ,
    (c > -1 → ∀ x, f a b x > c * g x ↔ x > c + 2 ∨ x < 1) ∧
    (c < -1 → ∀ x, f a b x > c * g x ↔ x > 1 ∨ x < c + 2) ∧
    (c = -1 → ∀ x, f a b x > c * g x ↔ x ≠ 1)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_inequality_l1615_161566


namespace NUMINAMATH_CALUDE_car_truck_distance_difference_l1615_161500

theorem car_truck_distance_difference 
  (truck_distance : ℝ) 
  (truck_time : ℝ) 
  (car_time : ℝ) 
  (speed_difference : ℝ) 
  (h1 : truck_distance = 296)
  (h2 : truck_time = 8)
  (h3 : car_time = 5.5)
  (h4 : speed_difference = 18) : 
  let truck_speed := truck_distance / truck_time
  let car_speed := truck_speed + speed_difference
  let car_distance := car_speed * car_time
  car_distance - truck_distance = 6.5 := by
sorry

end NUMINAMATH_CALUDE_car_truck_distance_difference_l1615_161500


namespace NUMINAMATH_CALUDE_different_color_probability_l1615_161544

theorem different_color_probability (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ)
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 3)
  (h3 : black_balls = 1) :
  (white_balls * black_balls) / ((total_balls * (total_balls - 1)) / 2) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_different_color_probability_l1615_161544


namespace NUMINAMATH_CALUDE_number_puzzle_l1615_161522

theorem number_puzzle : ∃ x : ℤ, x - 29 + 64 = 76 ∧ x = 41 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l1615_161522


namespace NUMINAMATH_CALUDE_penny_drawing_probability_l1615_161525

/-- The number of shiny pennies in the box -/
def shiny_pennies : ℕ := 3

/-- The number of dull pennies in the box -/
def dull_pennies : ℕ := 4

/-- The total number of pennies in the box -/
def total_pennies : ℕ := shiny_pennies + dull_pennies

/-- The probability of drawing more than four pennies until the third shiny penny appears -/
def prob_more_than_four_draws : ℚ := 31 / 35

theorem penny_drawing_probability :
  prob_more_than_four_draws = 31 / 35 :=
sorry

end NUMINAMATH_CALUDE_penny_drawing_probability_l1615_161525


namespace NUMINAMATH_CALUDE_smallest_integer_l1615_161509

theorem smallest_integer (x : ℕ) (m n : ℕ) (h1 : n = 36) (h2 : 0 < x) 
  (h3 : Nat.gcd m n = x + 3) (h4 : Nat.lcm m n = x * (x + 3)) :
  m ≥ 3 ∧ (∃ (x : ℕ), m = 3 ∧ 0 < x ∧ 
    Nat.gcd 3 36 = x + 3 ∧ Nat.lcm 3 36 = x * (x + 3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_l1615_161509


namespace NUMINAMATH_CALUDE_birthday_celebration_attendees_l1615_161581

theorem birthday_celebration_attendees :
  ∀ (n : ℕ), 
  (12 * (n + 2) = 16 * n) → 
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_birthday_celebration_attendees_l1615_161581


namespace NUMINAMATH_CALUDE_bijection_existence_l1615_161531

theorem bijection_existence :
  (∃ f : ℕ+ × ℕ+ → ℕ+, Function.Bijective f ∧
    (f (1, 1) = 1) ∧
    (∀ i > 1, ∃ d > 1, ∀ j, d ∣ f (i, j)) ∧
    (∀ j > 1, ∃ d > 1, ∀ i, d ∣ f (i, j))) ∧
  (∃ g : ℕ+ × ℕ+ → {n : ℕ+ // n ≠ 1}, Function.Bijective g ∧
    (∀ i, ∃ d > 1, ∀ j, d ∣ (g (i, j)).val) ∧
    (∀ j, ∃ d > 1, ∀ i, d ∣ (g (i, j)).val)) :=
by sorry

end NUMINAMATH_CALUDE_bijection_existence_l1615_161531


namespace NUMINAMATH_CALUDE_part_one_part_two_l1615_161576

/-- Given real numbers a and b, define the functions f and g -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

/-- Define the derivatives of f and g -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a
def g' (b : ℝ) (x : ℝ) : ℝ := 2*x + b

/-- Define consistent monotonicity on an interval -/
def consistent_monotonicity (a b : ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, f' a x * g' b x ≥ 0

/-- Part 1: Prove that if a > 0 and f, g have consistent monotonicity on [-1, +∞), then b ≥ 2 -/
theorem part_one (a b : ℝ) (ha : a > 0)
  (h_cons : consistent_monotonicity a b { x | x ≥ -1 }) : b ≥ 2 := by
  sorry

/-- Part 2: Prove that if a < 0, a ≠ b, and f, g have consistent monotonicity on (min a b, max a b),
    then |a - b| ≤ 1/3 -/
theorem part_two (a b : ℝ) (ha : a < 0) (hab : a ≠ b)
  (h_cons : consistent_monotonicity a b (Set.Ioo (min a b) (max a b))) : |a - b| ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1615_161576
