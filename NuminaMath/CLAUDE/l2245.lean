import Mathlib

namespace NUMINAMATH_CALUDE_lisa_children_count_l2245_224562

/-- The number of Lisa's children -/
def num_children : ℕ := 4

/-- The number of spoons in the new cutlery set -/
def new_cutlery_spoons : ℕ := 25

/-- The number of decorative spoons -/
def decorative_spoons : ℕ := 2

/-- The number of baby spoons per child -/
def baby_spoons_per_child : ℕ := 3

/-- The total number of spoons Lisa has -/
def total_spoons : ℕ := 39

/-- Theorem stating that the number of Lisa's children is 4 -/
theorem lisa_children_count : 
  num_children * baby_spoons_per_child + new_cutlery_spoons + decorative_spoons = total_spoons :=
by sorry

end NUMINAMATH_CALUDE_lisa_children_count_l2245_224562


namespace NUMINAMATH_CALUDE_badge_ratio_l2245_224592

/-- Proves that the ratio of delegates who made their own badges to delegates without pre-printed badges is 1:2 -/
theorem badge_ratio (total : ℕ) (pre_printed : ℕ) (no_badge : ℕ) 
  (h1 : total = 36)
  (h2 : pre_printed = 16)
  (h3 : no_badge = 10) : 
  (total - pre_printed - no_badge) / (total - pre_printed) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_badge_ratio_l2245_224592


namespace NUMINAMATH_CALUDE_gcd_18_30_l2245_224599

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l2245_224599


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2245_224541

theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = x + m ∧ x^2 / 2 + y^2 = 1 → 
    ∃! p : ℝ × ℝ, p.1^2 / 2 + p.2^2 = 1 ∧ p.2 = p.1 + m) ↔ 
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2245_224541


namespace NUMINAMATH_CALUDE_brownies_before_division_l2245_224595

def initial_brownies : ℕ := 24  -- 2 dozen

def father_ate (n : ℕ) : ℕ := n / 3

def mooney_ate (n : ℕ) : ℕ := n / 4  -- 25% = 1/4

def benny_ate (n : ℕ) : ℕ := n * 2 / 5

def snoopy_ate : ℕ := 3

def mother_baked_wednesday : ℕ := 18  -- 1.5 dozen

def mother_baked_thursday : ℕ := 36  -- 3 dozen

def final_brownies : ℕ :=
  let after_father := initial_brownies - father_ate initial_brownies
  let after_mooney := after_father - mooney_ate after_father
  let after_benny := after_mooney - benny_ate after_mooney
  let after_snoopy := after_benny - snoopy_ate
  after_snoopy + mother_baked_wednesday + mother_baked_thursday

theorem brownies_before_division :
  final_brownies = 59 := by sorry

end NUMINAMATH_CALUDE_brownies_before_division_l2245_224595


namespace NUMINAMATH_CALUDE_square_sum_ge_twice_product_l2245_224542

theorem square_sum_ge_twice_product (x y : ℝ) : x^2 + y^2 ≥ 2*x*y := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_twice_product_l2245_224542


namespace NUMINAMATH_CALUDE_half_area_of_rectangle_l2245_224534

/-- Half the area of a rectangle with width 25 cm and height 16 cm is 200 cm². -/
theorem half_area_of_rectangle (width height : ℝ) (h1 : width = 25) (h2 : height = 16) :
  (width * height) / 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_half_area_of_rectangle_l2245_224534


namespace NUMINAMATH_CALUDE_three_number_sum_l2245_224576

theorem three_number_sum (A B C : ℝ) (h1 : A/B = 2/3) (h2 : B/C = 5/8) (h3 : B = 30) :
  A + B + C = 98 := by
sorry

end NUMINAMATH_CALUDE_three_number_sum_l2245_224576


namespace NUMINAMATH_CALUDE_problem_statement_l2245_224527

theorem problem_statement (x y : ℝ) (hx : x = 1/3) (hy : y = 3) :
  (1/4) * x^3 * y^8 = 60.75 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2245_224527


namespace NUMINAMATH_CALUDE_upper_bound_y_l2245_224535

theorem upper_bound_y (x y : ℤ) (U : ℤ) : 
  (3 < x ∧ x < 6) → 
  (6 < y ∧ y < U) → 
  (∀ (x' y' : ℤ), (3 < x' ∧ x' < 6) → (6 < y' ∧ y' < U) → y' - x' ≤ 4) →
  (∃ (x' y' : ℤ), (3 < x' ∧ x' < 6) ∧ (6 < y' ∧ y' < U) ∧ y' - x' = 4) →
  U = 10 :=
by sorry

end NUMINAMATH_CALUDE_upper_bound_y_l2245_224535


namespace NUMINAMATH_CALUDE_intersection_equality_implies_x_values_l2245_224578

theorem intersection_equality_implies_x_values (x : ℝ) : 
  let A : Set ℝ := {1, 4, x}
  let B : Set ℝ := {1, x^2}
  (A ∩ B = B) → (x = -2 ∨ x = 2 ∨ x = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_x_values_l2245_224578


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l2245_224549

theorem modulus_of_complex_number (θ : Real) (h : 2 * Real.pi < θ ∧ θ < 3 * Real.pi) :
  Complex.abs (1 - Real.cos θ + Complex.I * Real.sin θ) = -2 * Real.sin (θ / 2) := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l2245_224549


namespace NUMINAMATH_CALUDE_nabla_equation_solution_l2245_224547

-- Define the ∇ operation
def nabla (a b : ℤ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem nabla_equation_solution :
  ∀ b : ℤ, b ≠ 3 → nabla 3 b = -4 → b = 5 := by
  sorry

end NUMINAMATH_CALUDE_nabla_equation_solution_l2245_224547


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_10_l2245_224508

theorem binomial_coefficient_20_10 
  (h1 : Nat.choose 18 8 = 43758)
  (h2 : Nat.choose 18 9 = 48620)
  (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_10_l2245_224508


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2245_224532

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 1 > x + 1 ∧ (4 * x - 5) / 3 ≤ x) ↔ (1 < x ∧ x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2245_224532


namespace NUMINAMATH_CALUDE_complex_product_real_implies_ratio_l2245_224551

theorem complex_product_real_implies_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∃ (r : ℝ), (3 - 4 * Complex.I) * (p + q * Complex.I) = r) : p / q = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_implies_ratio_l2245_224551


namespace NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l2245_224597

theorem complex_power_one_minus_i_six :
  (1 - Complex.I) ^ 6 = 8 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l2245_224597


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2245_224557

theorem polynomial_division_theorem (x : ℚ) :
  (4 * x^2 - 4/3 * x + 2) * (3 * x + 4) + 10/3 = 12 * x^3 + 24 * x^2 - 10 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2245_224557


namespace NUMINAMATH_CALUDE_larger_number_problem_l2245_224564

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 10) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2245_224564


namespace NUMINAMATH_CALUDE_point_on_curve_l2245_224520

/-- The curve C defined by y = x^3 - 10x + 3 -/
def C : ℝ → ℝ := λ x ↦ x^3 - 10*x + 3

/-- The derivative of curve C -/
def C' : ℝ → ℝ := λ x ↦ 3*x^2 - 10

theorem point_on_curve (x y : ℝ) :
  x < 0 →  -- P is in the second quadrant (x < 0)
  y > 0 →  -- P is in the second quadrant (y > 0)
  y = C x →  -- P lies on the curve C
  C' x = 2 →  -- The slope of the tangent line at P is 2
  x = -2 ∧ y = 15 := by  -- P has coordinates (-2, 15)
  sorry

end NUMINAMATH_CALUDE_point_on_curve_l2245_224520


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2245_224573

theorem simplify_polynomial (x : ℝ) : 
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2245_224573


namespace NUMINAMATH_CALUDE_equation_solutions_l2245_224583

theorem equation_solutions (x : ℝ) : 
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12) ↔ 
  (x = 10 ∨ x = -1) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2245_224583


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2245_224511

theorem simplify_and_evaluate (a : ℚ) (h : a = -3) : 
  (a - 2) / ((1 + 2*a + a^2) * (a - 3*a/(a+1))) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2245_224511


namespace NUMINAMATH_CALUDE_solution_characterization_l2245_224570

/-- The set of solutions to the system of equations:
    a² + b = c²
    b² + c = a²
    c² + a = b²
-/
def SolutionSet : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (0, 1, -1), (-1, 0, 1), (1, -1, 0)}

/-- A triplet (a, b, c) satisfies the system of equations -/
def SatisfiesSystem (t : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c) := t
  a^2 + b = c^2 ∧ b^2 + c = a^2 ∧ c^2 + a = b^2

theorem solution_characterization :
  ∀ t : ℝ × ℝ × ℝ, SatisfiesSystem t ↔ t ∈ SolutionSet := by
  sorry


end NUMINAMATH_CALUDE_solution_characterization_l2245_224570


namespace NUMINAMATH_CALUDE_storage_b_has_five_pieces_l2245_224509

/-- Represents a storage device with a number of data pieces -/
structure StorageDevice :=
  (pieces : ℕ)

/-- Represents the state of three storage devices A, B, and C -/
structure StorageState :=
  (A : StorageDevice)
  (B : StorageDevice)
  (C : StorageDevice)

/-- Performs the described operations on the storage devices -/
def performOperations (n : ℕ) (initial : StorageState) : StorageState :=
  { A := ⟨2 * (n - 2)⟩,
    B := ⟨n + 3 - (n - 2)⟩,
    C := ⟨n - 1⟩ }

/-- The theorem stating that after the operations, storage device B has 5 data pieces -/
theorem storage_b_has_five_pieces (n : ℕ) (h : n ≥ 2) :
  (performOperations n { A := ⟨0⟩, B := ⟨0⟩, C := ⟨0⟩ }).B.pieces = 5 := by
  sorry

#check storage_b_has_five_pieces

end NUMINAMATH_CALUDE_storage_b_has_five_pieces_l2245_224509


namespace NUMINAMATH_CALUDE_composite_sum_of_squares_l2245_224552

theorem composite_sum_of_squares (a b : ℤ) : 
  (∃ x y : ℕ, x^2 + a*x + b + 1 = 0 ∧ y^2 + a*y + b + 1 = 0 ∧ x ≠ y) → 
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n :=
by sorry

end NUMINAMATH_CALUDE_composite_sum_of_squares_l2245_224552


namespace NUMINAMATH_CALUDE_no_integer_solution_l2245_224586

theorem no_integer_solution (n : ℝ) (hn : n ≠ 0) :
  ¬ ∃ z : ℤ, n / (z : ℝ) = n / ((z : ℝ) + 1) + n / ((z : ℝ) + 25) :=
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2245_224586


namespace NUMINAMATH_CALUDE_equation_solution_l2245_224579

theorem equation_solution : 
  ∀ (x y : ℝ), (16 * x^2 + 1) * (y^2 + 1) = 16 * x * y ↔ 
  ((x = 1/4 ∧ y = 1) ∨ (x = -1/4 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2245_224579


namespace NUMINAMATH_CALUDE_stamp_purchase_problem_l2245_224575

theorem stamp_purchase_problem :
  ∀ (x y z : ℕ),
  (x : ℤ) + 2 * y + 5 * z = 100 →  -- Total cost in cents
  y = 10 * x →                    -- Relation between 1-cent and 2-cent stamps
  x > 0 ∧ y > 0 ∧ z > 0 →         -- All stamp quantities are positive
  x = 5 ∧ y = 50 ∧ z = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_stamp_purchase_problem_l2245_224575


namespace NUMINAMATH_CALUDE_amber_guppies_l2245_224519

/-- The number of guppies in Amber's pond -/
theorem amber_guppies (initial_adults : ℕ) (first_batch_dozens : ℕ) (second_batch : ℕ) :
  initial_adults + (first_batch_dozens * 12) + second_batch =
  initial_adults + first_batch_dozens * 12 + second_batch :=
by sorry

end NUMINAMATH_CALUDE_amber_guppies_l2245_224519


namespace NUMINAMATH_CALUDE_sequence_constant_iff_perfect_square_l2245_224537

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sequence a_k defined recursively -/
def sequenceA (A : ℕ) : ℕ → ℕ
  | 0 => A
  | k + 1 => sequenceA A k + sumOfDigits (sequenceA A k)

/-- A number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- The sequence eventually becomes constant -/
def eventuallyConstant (A : ℕ) : Prop := ∃ N : ℕ, ∀ k ≥ N, sequenceA A k = sequenceA A N

/-- Main theorem -/
theorem sequence_constant_iff_perfect_square (A : ℕ) :
  eventuallyConstant A ↔ isPerfectSquare A := by sorry

end NUMINAMATH_CALUDE_sequence_constant_iff_perfect_square_l2245_224537


namespace NUMINAMATH_CALUDE_largest_expression_l2245_224545

theorem largest_expression : 
  let a := 2 + 0 + 1 + 3
  let b := 2 * 0 + 1 + 3
  let c := 2 + 0 * 1 + 3
  let d := 2 + 0 + 1 * 3
  let e := 2 * 0 * 1 * 3
  (a ≥ b) ∧ (a ≥ c) ∧ (a ≥ d) ∧ (a ≥ e) :=
by sorry

end NUMINAMATH_CALUDE_largest_expression_l2245_224545


namespace NUMINAMATH_CALUDE_group_size_l2245_224596

/-- The number of members in the group -/
def n : ℕ := sorry

/-- The total collection in paise -/
def total_collection : ℕ := 1369

/-- Each member contributes as many paise as there are members -/
axiom member_contribution : n = n

/-- The total collection is the product of the number of members and their contribution -/
axiom total_collection_eq : n * n = total_collection

theorem group_size : n = 37 := by sorry

end NUMINAMATH_CALUDE_group_size_l2245_224596


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2245_224548

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  prop1 : a 2 * a 6 = 16
  prop2 : a 4 + a 8 = 8

/-- The main theorem -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : seq.a 20 / seq.a 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2245_224548


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l2245_224546

theorem sum_remainder_mod_seven : (2^2003 + 2003^2) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l2245_224546


namespace NUMINAMATH_CALUDE_four_students_three_activities_l2245_224507

/-- The number of different sign-up methods for students choosing activities -/
def signUpMethods (numStudents : ℕ) (numActivities : ℕ) : ℕ :=
  numActivities ^ numStudents

/-- Theorem: Four students signing up for three activities, with each student
    choosing exactly one activity, results in 81 different sign-up methods -/
theorem four_students_three_activities :
  signUpMethods 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_four_students_three_activities_l2245_224507


namespace NUMINAMATH_CALUDE_coffee_mix_solution_l2245_224565

/-- Represents the coffee mix problem -/
structure CoffeeMix where
  total_mix : ℝ
  columbian_price : ℝ
  brazilian_price : ℝ
  ethiopian_price : ℝ
  mix_price : ℝ
  ratio_columbian : ℝ
  ratio_brazilian : ℝ
  ratio_ethiopian : ℝ

/-- Theorem stating the correct amounts of each coffee type -/
theorem coffee_mix_solution (mix : CoffeeMix)
  (h_total : mix.total_mix = 150)
  (h_columbian_price : mix.columbian_price = 9.5)
  (h_brazilian_price : mix.brazilian_price = 4.25)
  (h_ethiopian_price : mix.ethiopian_price = 7.25)
  (h_mix_price : mix.mix_price = 6.7)
  (h_ratio : mix.ratio_columbian = 2 ∧ mix.ratio_brazilian = 3 ∧ mix.ratio_ethiopian = 5) :
  ∃ (columbian brazilian ethiopian : ℝ),
    columbian = 30 ∧
    brazilian = 45 ∧
    ethiopian = 75 ∧
    columbian + brazilian + ethiopian = mix.total_mix ∧
    columbian / mix.ratio_columbian = brazilian / mix.ratio_brazilian ∧
    columbian / mix.ratio_columbian = ethiopian / mix.ratio_ethiopian :=
by
  sorry


end NUMINAMATH_CALUDE_coffee_mix_solution_l2245_224565


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2245_224593

theorem two_digit_number_property : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  n = 3 * ((n / 10) + (n % 10)) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2245_224593


namespace NUMINAMATH_CALUDE_point_on_line_l2245_224594

/-- A line passing through point (1,3) with slope 2 -/
def line_l (b : ℝ) : ℝ → ℝ := λ x ↦ 2 * x + b

/-- The y-coordinate of point P -/
def point_p_y : ℝ := 3

/-- The x-coordinate of point P -/
def point_p_x : ℝ := 1

/-- The y-coordinate of point Q -/
def point_q_y : ℝ := 5

/-- The x-coordinate of point Q -/
def point_q_x : ℝ := 2

theorem point_on_line :
  ∃ b : ℝ, line_l b point_p_x = point_p_y ∧ line_l b point_q_x = point_q_y := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2245_224594


namespace NUMINAMATH_CALUDE_largest_number_theorem_l2245_224540

theorem largest_number_theorem (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_products_eq : p * q + p * r + q * r = 1)
  (product_eq : p * q * r = 2) :
  max p (max q r) = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_theorem_l2245_224540


namespace NUMINAMATH_CALUDE_daniel_added_four_eggs_l2245_224568

/-- The number of eggs Daniel put in the box -/
def eggs_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Daniel put 4 eggs in the box -/
theorem daniel_added_four_eggs (initial final : ℕ) 
  (h1 : initial = 7) 
  (h2 : final = 11) : 
  eggs_added initial final = 4 := by
  sorry

end NUMINAMATH_CALUDE_daniel_added_four_eggs_l2245_224568


namespace NUMINAMATH_CALUDE_toothpicks_at_250_l2245_224500

/-- Calculates the number of toothpicks at a given stage -/
def toothpicks (stage : ℕ) : ℕ :=
  if stage = 0 then 0
  else if stage % 50 = 0 then 2 * toothpicks (stage - 1)
  else if stage = 1 then 5
  else toothpicks (stage - 1) + 5

/-- The number of toothpicks at the 250th stage is 15350 -/
theorem toothpicks_at_250 : toothpicks 250 = 15350 := by
  sorry

#eval toothpicks 250  -- This line is optional, for verification purposes

end NUMINAMATH_CALUDE_toothpicks_at_250_l2245_224500


namespace NUMINAMATH_CALUDE_square_binomial_constant_l2245_224524

theorem square_binomial_constant (b : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + b = (4 * x + c)^2) → b = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_constant_l2245_224524


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2245_224518

theorem inscribed_circle_radius (a b c : ℝ) (r : ℝ) : 
  a = 5 → b = 12 → c = 13 →
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = 1.5 * (a + b + c) - 12 →
  r = 33 / 15 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2245_224518


namespace NUMINAMATH_CALUDE_magic_square_sum_divisible_by_three_l2245_224515

/-- Represents a 3x3 magic square -/
def MagicSquare : Type := Fin 3 → Fin 3 → ℕ

/-- The sum of a row, column, or diagonal in a magic square -/
def magic_sum (square : MagicSquare) : ℕ := square 0 0 + square 0 1 + square 0 2

/-- Predicate to check if a square is a valid magic square -/
def is_magic_square (square : MagicSquare) : Prop :=
  (∀ i : Fin 3, square i 0 + square i 1 + square i 2 = magic_sum square) ∧
  (∀ j : Fin 3, square 0 j + square 1 j + square 2 j = magic_sum square) ∧
  (square 0 0 + square 1 1 + square 2 2 = magic_sum square) ∧
  (square 0 2 + square 1 1 + square 2 0 = magic_sum square)

theorem magic_square_sum_divisible_by_three (square : MagicSquare) 
  (h : is_magic_square square) : 
  ∃ k : ℕ, magic_sum square = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_divisible_by_three_l2245_224515


namespace NUMINAMATH_CALUDE_statues_painted_l2245_224512

theorem statues_painted (total_paint : ℚ) (paint_per_statue : ℚ) :
  total_paint = 3/6 ∧ paint_per_statue = 1/6 → total_paint / paint_per_statue = 3 :=
by sorry

end NUMINAMATH_CALUDE_statues_painted_l2245_224512


namespace NUMINAMATH_CALUDE_find_two_fake_coins_l2245_224553

/-- Represents the state of our coin testing process -/
structure CoinState where
  total : Nat
  fake : Nat
  deriving Repr

/-- Represents the result of a test -/
inductive TestResult
  | Signal
  | NoSignal
  deriving Repr

/-- A function that simulates a test -/
def test (coins : Nat) (state : CoinState) : TestResult := sorry

/-- A function that updates the state based on a test result -/
def updateState (coins : Nat) (state : CoinState) (result : TestResult) : CoinState := sorry

/-- A function that represents a single step in our testing strategy -/
def testStep (state : CoinState) : CoinState := sorry

/-- The main theorem stating that we can find two fake coins in five steps -/
theorem find_two_fake_coins 
  (initial_state : CoinState) 
  (h1 : initial_state.total = 49) 
  (h2 : initial_state.fake = 24) : 
  ∃ (final_state : CoinState), 
    (final_state.total = 2 ∧ final_state.fake = 2) ∧ 
    (∃ (s1 s2 s3 s4 : CoinState), 
      s1 = testStep initial_state ∧ 
      s2 = testStep s1 ∧ 
      s3 = testStep s2 ∧ 
      s4 = testStep s3 ∧ 
      final_state = testStep s4) :=
sorry

end NUMINAMATH_CALUDE_find_two_fake_coins_l2245_224553


namespace NUMINAMATH_CALUDE_max_value_of_reciprocal_sum_l2245_224531

theorem max_value_of_reciprocal_sum (t q r₁ r₂ : ℝ) : 
  (∀ x, x^2 - t*x + q = 0 ↔ x = r₁ ∨ x = r₂) →
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2010 → r₁ + r₂ = r₁^n + r₂^n) →
  (∃ M : ℝ, M = (1 : ℝ) / r₁^2010 + (1 : ℝ) / r₂^2010 ∧ 
   ∀ t' q' r₁' r₂' : ℝ, 
     (∀ x, x^2 - t'*x + q' = 0 ↔ x = r₁' ∨ x = r₂') →
     (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2010 → r₁' + r₂' = r₁'^n + r₂'^n) →
     (1 : ℝ) / r₁'^2010 + (1 : ℝ) / r₂'^2010 ≤ M) →
  M = 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_reciprocal_sum_l2245_224531


namespace NUMINAMATH_CALUDE_marker_carton_cost_l2245_224554

/-- Proves that the cost of each carton of markers is $20 given the specified conditions --/
theorem marker_carton_cost (
  pencil_cartons : ℕ)
  (pencil_boxes_per_carton : ℕ)
  (pencil_box_cost : ℕ)
  (marker_cartons : ℕ)
  (marker_boxes_per_carton : ℕ)
  (total_spent : ℕ)
  (h1 : pencil_cartons = 20)
  (h2 : pencil_boxes_per_carton = 10)
  (h3 : pencil_box_cost = 2)
  (h4 : marker_cartons = 10)
  (h5 : marker_boxes_per_carton = 5)
  (h6 : total_spent = 600)
  : (total_spent - pencil_cartons * pencil_boxes_per_carton * pencil_box_cost) / marker_cartons = 20 := by
  sorry

end NUMINAMATH_CALUDE_marker_carton_cost_l2245_224554


namespace NUMINAMATH_CALUDE_circles_intersect_l2245_224598

/-- Two circles in a plane -/
structure TwoCircles where
  /-- The first circle: x² + y² - 2x = 0 -/
  c1 : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 1}
  /-- The second circle: x² + y² + 4y = 0 -/
  c2 : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 + 2)^2 = 4}

/-- The circles intersect if there exists a point that belongs to both circles -/
def intersect (tc : TwoCircles) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ tc.c1 ∧ p ∈ tc.c2

/-- Theorem stating that the two given circles intersect -/
theorem circles_intersect : ∀ tc : TwoCircles, intersect tc := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l2245_224598


namespace NUMINAMATH_CALUDE_article_cost_l2245_224529

/-- Proves that the cost of an article is 50 Rs given the profit conditions -/
theorem article_cost (original_profit : Real) (reduced_cost_percentage : Real) 
  (price_reduction : Real) (new_profit : Real) :
  original_profit = 0.25 →
  reduced_cost_percentage = 0.20 →
  price_reduction = 10.50 →
  new_profit = 0.30 →
  ∃ (cost : Real), cost = 50 ∧
    (cost + original_profit * cost) - price_reduction = 
    (cost - reduced_cost_percentage * cost) + new_profit * (cost - reduced_cost_percentage * cost) :=
by sorry

end NUMINAMATH_CALUDE_article_cost_l2245_224529


namespace NUMINAMATH_CALUDE_power_fraction_equality_l2245_224514

theorem power_fraction_equality : (7^14 : ℕ) / (49^6 : ℕ) = 49 := by sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l2245_224514


namespace NUMINAMATH_CALUDE_square_sum_equals_four_l2245_224550

theorem square_sum_equals_four (x y : ℝ) (h1 : x + y = -4) (h2 : x = 6 / y) : x^2 + y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_four_l2245_224550


namespace NUMINAMATH_CALUDE_quadratic_has_real_roots_root_condition_implies_value_minimum_value_of_y_l2245_224585

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (2*k + 2)*x + 4*k

-- Part 1: Prove that the equation always has real roots
theorem quadratic_has_real_roots (k : ℝ) :
  ∃ x : ℝ, quadratic k x = 0 := by sorry

-- Part 2: Given the condition on roots, find the value of the expression
theorem root_condition_implies_value (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : quadratic k x₁ = 0) (h2 : quadratic k x₂ = 0)
  (h3 : x₂/x₁ + x₁/x₂ - 2 = 0) :
  (1 + 4/(k^2 - 4)) * ((k + 2)/k) = -1 := by sorry

-- Part 3: Find the minimum value of y
theorem minimum_value_of_y (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : quadratic k x₁ = 0) (h2 : quadratic k x₂ = 0)
  (h3 : x₁ > x₂) (h4 : k < 1/2) :
  ∃ y_min : ℝ, y_min = 3/4 ∧ ∀ y : ℝ, y ≥ y_min → ∃ x₂ : ℝ, y = x₂^2 - k*x₁ + 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_has_real_roots_root_condition_implies_value_minimum_value_of_y_l2245_224585


namespace NUMINAMATH_CALUDE_bicycle_discount_price_l2245_224536

theorem bicycle_discount_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 200 →
  discount1 = 0.4 →
  discount2 = 0.2 →
  original_price * (1 - discount1) * (1 - discount2) = 96 := by
sorry

end NUMINAMATH_CALUDE_bicycle_discount_price_l2245_224536


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l2245_224539

theorem ferris_wheel_capacity (num_seats : ℕ) (people_per_seat : ℕ) 
  (h1 : num_seats = 4) (h2 : people_per_seat = 5) :
  num_seats * people_per_seat = 20 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l2245_224539


namespace NUMINAMATH_CALUDE_travel_agency_comparison_l2245_224543

/-- Represents the total cost for Travel Agency A -/
def cost_a (x : ℝ) : ℝ := 2 * 500 + 500 * x * 0.7

/-- Represents the total cost for Travel Agency B -/
def cost_b (x : ℝ) : ℝ := (x + 2) * 500 * 0.8

theorem travel_agency_comparison (x : ℝ) :
  (x < 4 → cost_a x > cost_b x) ∧
  (x = 4 → cost_a x = cost_b x) ∧
  (x > 4 → cost_a x < cost_b x) :=
sorry

end NUMINAMATH_CALUDE_travel_agency_comparison_l2245_224543


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l2245_224521

/-- Given a shirt with an initial price and a required price increase to achieve
    a 100% markup, calculate the initial markup percentage. -/
theorem initial_markup_percentage
  (initial_price : ℝ)
  (price_increase : ℝ)
  (h1 : initial_price = 27)
  (h2 : price_increase = 3)
  (h3 : initial_price + price_increase = 2 * (initial_price - (initial_price - (initial_price / (1 + 1))))): 
  (initial_price - (initial_price / (1 + 1))) / (initial_price / (1 + 1)) * 100 = 80 :=
by sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l2245_224521


namespace NUMINAMATH_CALUDE_digit_equation_sum_l2245_224538

theorem digit_equation_sum (A B C D U : ℕ) : 
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ U) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ U) ∧
  (C ≠ D) ∧ (C ≠ U) ∧
  (D ≠ U) ∧
  (A < 10) ∧ (B < 10) ∧ (C < 10) ∧ (D < 10) ∧ (U < 10) ∧ (U > 0) ∧
  ((10 * A + B) * (10 * C + D) = 111 * U) →
  A + B + C + D + U = 17 := by
sorry

end NUMINAMATH_CALUDE_digit_equation_sum_l2245_224538


namespace NUMINAMATH_CALUDE_parabolas_cyclic_quadrilateral_l2245_224587

/-- A parabola in the xy-plane --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop

/-- Two parabolas have perpendicular axes --/
def perpendicular_axes (p1 p2 : Parabola) : Prop := sorry

/-- Two parabolas intersect at four distinct points --/
def four_distinct_intersections (p1 p2 : Parabola) : Prop := sorry

/-- Four points in the plane form a cyclic quadrilateral --/
def cyclic_quadrilateral (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

/-- The main theorem --/
theorem parabolas_cyclic_quadrilateral (p1 p2 : Parabola) :
  perpendicular_axes p1 p2 →
  four_distinct_intersections p1 p2 →
  ∃ q1 q2 q3 q4 : ℝ × ℝ,
    (p1.eq q1.1 q1.2 ∧ p2.eq q1.1 q1.2) ∧
    (p1.eq q2.1 q2.2 ∧ p2.eq q2.1 q2.2) ∧
    (p1.eq q3.1 q3.2 ∧ p2.eq q3.1 q3.2) ∧
    (p1.eq q4.1 q4.2 ∧ p2.eq q4.1 q4.2) ∧
    cyclic_quadrilateral q1 q2 q3 q4 :=
by sorry

end NUMINAMATH_CALUDE_parabolas_cyclic_quadrilateral_l2245_224587


namespace NUMINAMATH_CALUDE_nina_travel_period_l2245_224503

/-- Nina's travel pattern over two months -/
def two_month_distance : ℕ := 400 + 800

/-- Total distance Nina wants to travel -/
def total_distance : ℕ := 14400

/-- Number of two-month periods needed to reach the total distance -/
def num_two_month_periods : ℕ := total_distance / two_month_distance

/-- Duration of Nina's travel period in months -/
def travel_period_months : ℕ := num_two_month_periods * 2

/-- Theorem stating that Nina's travel period is 24 months -/
theorem nina_travel_period :
  travel_period_months = 24 :=
sorry

end NUMINAMATH_CALUDE_nina_travel_period_l2245_224503


namespace NUMINAMATH_CALUDE_probability_one_has_no_growth_pie_l2245_224523

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := total_pies - growth_pies
def pies_given : ℕ := 3

def probability_no_growth_pie : ℚ := 7/10

theorem probability_one_has_no_growth_pie :
  (1 : ℚ) - (Nat.choose shrink_pies (pies_given - 1) : ℚ) / (Nat.choose total_pies pies_given : ℚ) = probability_no_growth_pie :=
sorry

end NUMINAMATH_CALUDE_probability_one_has_no_growth_pie_l2245_224523


namespace NUMINAMATH_CALUDE_terminal_side_negative_pi_in_fourth_quadrant_l2245_224584

/-- The terminal side of -π radians lies in the fourth quadrant -/
theorem terminal_side_negative_pi_in_fourth_quadrant :
  let angle : ℝ := -π
  (angle > -2*π ∧ angle ≤ -3*π/2) ∨ (angle > 3*π/2 ∧ angle ≤ 2*π) :=
by sorry

end NUMINAMATH_CALUDE_terminal_side_negative_pi_in_fourth_quadrant_l2245_224584


namespace NUMINAMATH_CALUDE_jellybean_probability_l2245_224533

/-- The number of jellybean colors -/
def num_colors : ℕ := 5

/-- The number of jellybeans in the sample -/
def sample_size : ℕ := 5

/-- The probability of selecting exactly 2 distinct colors when randomly choosing
    5 jellybeans from a set of 5 equally proportioned colors -/
theorem jellybean_probability : 
  (num_colors.choose 2 * (2^sample_size - 2)) / (num_colors^sample_size) = 12/125 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l2245_224533


namespace NUMINAMATH_CALUDE_pond_soil_volume_l2245_224504

/-- The volume of soil extracted from a rectangular pond -/
def soil_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

/-- Theorem: The volume of soil extracted from a rectangular pond
    with dimensions 20 m × 15 m × 5 m is 1500 cubic meters -/
theorem pond_soil_volume :
  soil_volume 20 15 5 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_pond_soil_volume_l2245_224504


namespace NUMINAMATH_CALUDE_base_subtraction_l2245_224506

/-- Converts a number from base b to base 10 --/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The problem statement --/
theorem base_subtraction :
  let base7_num := [5, 4, 3, 2, 1]
  let base8_num := [1, 2, 3, 4, 5]
  toBase10 base7_num 7 - toBase10 base8_num 8 = 8190 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l2245_224506


namespace NUMINAMATH_CALUDE_sector_area_l2245_224556

/-- The area of a circular sector with central angle 3/4π and radius 4 is 6π. -/
theorem sector_area : 
  let central_angle : Real := 3/4 * Real.pi
  let radius : Real := 4
  let sector_area : Real := 1/2 * central_angle * radius^2
  sector_area = 6 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sector_area_l2245_224556


namespace NUMINAMATH_CALUDE_fibConversionAccuracy_l2245_224528

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define the Fibonacci representation of a number
def fibRep (n : ℕ) : List ℕ := sorry

-- Define the conversion function using Fibonacci representation
def kmToMilesFib (km : ℕ) : ℕ := sorry

-- Define the exact conversion from km to miles
def kmToMilesExact (km : ℕ) : ℚ :=
  (km : ℚ) / 1.609

-- Main theorem
theorem fibConversionAccuracy :
  ∀ n : ℕ, n ≤ 100 →
    |((kmToMilesFib n : ℚ) - kmToMilesExact n)| < 2/3 := by sorry

end NUMINAMATH_CALUDE_fibConversionAccuracy_l2245_224528


namespace NUMINAMATH_CALUDE_x_value_l2245_224572

theorem x_value : ∃ (x : ℝ), x > 0 ∧ Real.sqrt ((4 * x) / 3) = x ∧ x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2245_224572


namespace NUMINAMATH_CALUDE_fraction_equality_l2245_224525

theorem fraction_equality (a b : ℝ) : (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2245_224525


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2245_224589

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 44) (h2 : 5 * x = 6 * y) : min x y = 20 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2245_224589


namespace NUMINAMATH_CALUDE_woman_completes_in_40_days_l2245_224561

-- Define the efficiency ratio between man and woman
def efficiency_ratio : ℝ := 1.25

-- Define the number of days it takes the man to complete the task
def man_days : ℝ := 32

-- Define the function to calculate the woman's days
def woman_days : ℝ := efficiency_ratio * man_days

-- Theorem to prove
theorem woman_completes_in_40_days : 
  woman_days = 40 := by sorry

end NUMINAMATH_CALUDE_woman_completes_in_40_days_l2245_224561


namespace NUMINAMATH_CALUDE_equation_solution_l2245_224591

theorem equation_solution : ∃! x : ℚ, 3 * x - 5 = |-20 + 6| := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2245_224591


namespace NUMINAMATH_CALUDE_cake_supplies_cost_l2245_224569

/-- Proves that the cost of supplies for a cake is $54 given the specified conditions -/
theorem cake_supplies_cost (hours_per_day : ℕ) (days_worked : ℕ) (hourly_rate : ℕ) (profit : ℕ) : 
  hours_per_day = 2 →
  days_worked = 4 →
  hourly_rate = 22 →
  profit = 122 →
  (hours_per_day * days_worked * hourly_rate) - profit = 54 :=
by sorry

end NUMINAMATH_CALUDE_cake_supplies_cost_l2245_224569


namespace NUMINAMATH_CALUDE_triangle_count_on_circle_l2245_224577

theorem triangle_count_on_circle (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) : 
  Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_on_circle_l2245_224577


namespace NUMINAMATH_CALUDE_binomial_expression_is_integer_l2245_224522

theorem binomial_expression_is_integer (m n : ℕ) : 
  ∃ k : ℤ, k = (m.factorial * (2*n + 2*m).factorial) / 
              ((2*m).factorial * n.factorial * (n+m).factorial) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expression_is_integer_l2245_224522


namespace NUMINAMATH_CALUDE_area_difference_zero_l2245_224571

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ

/-- A point inside the polygon -/
def InteriorPoint (p : RegularPolygon n) := ℝ × ℝ

/-- The area difference function between black and white triangles -/
def areaDifference (p : RegularPolygon n) (point : InteriorPoint p) : ℝ := sorry

/-- Theorem stating that the area difference is always zero -/
theorem area_difference_zero (n : ℕ) (p : RegularPolygon n) (point : InteriorPoint p) :
  areaDifference p point = 0 := by sorry

end NUMINAMATH_CALUDE_area_difference_zero_l2245_224571


namespace NUMINAMATH_CALUDE_total_transport_is_405_l2245_224555

/-- Calculates the total number of people transported by two boats over two days -/
def total_people_transported (boat_a_capacity : ℕ) (boat_b_capacity : ℕ)
  (day1_a_trips : ℕ) (day1_b_trips : ℕ)
  (day2_a_trips : ℕ) (day2_b_trips : ℕ) : ℕ :=
  (boat_a_capacity * day1_a_trips + boat_b_capacity * day1_b_trips) +
  (boat_a_capacity * day2_a_trips + boat_b_capacity * day2_b_trips)

/-- Theorem stating that the total number of people transported is 405 -/
theorem total_transport_is_405 :
  total_people_transported 20 15 7 5 5 6 = 405 := by
  sorry

#eval total_people_transported 20 15 7 5 5 6

end NUMINAMATH_CALUDE_total_transport_is_405_l2245_224555


namespace NUMINAMATH_CALUDE_school_paintable_area_l2245_224566

/-- Represents the dimensions of a classroom -/
structure ClassroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total wall area to be painted in all classrooms -/
def totalPaintableArea (dimensions : ClassroomDimensions) (numClassrooms : ℕ) (unpaintableArea : ℝ) : ℝ :=
  let wallArea := 2 * (dimensions.length * dimensions.height + dimensions.width * dimensions.height)
  let paintableArea := wallArea - unpaintableArea
  numClassrooms * paintableArea

/-- Theorem stating the total paintable area for the given school -/
theorem school_paintable_area :
  let dimensions : ClassroomDimensions := ⟨15, 12, 10⟩
  let numClassrooms : ℕ := 4
  let unpaintableArea : ℝ := 80
  totalPaintableArea dimensions numClassrooms unpaintableArea = 1840 := by
  sorry

#check school_paintable_area

end NUMINAMATH_CALUDE_school_paintable_area_l2245_224566


namespace NUMINAMATH_CALUDE_last_digit_sum_l2245_224510

def is_valid_pair (a b : Nat) : Prop :=
  (a * 10 + b) % 17 = 0 ∨ (a * 10 + b) % 23 = 0

def valid_sequence (s : List Nat) : Prop :=
  s.length = 2000 ∧
  s.head? = some 3 ∧
  ∀ i, i < 1999 → is_valid_pair (s.get! i) (s.get! (i + 1))

theorem last_digit_sum (s : List Nat) (a b : Nat) :
  valid_sequence s →
  (s.getLast? = some a ∨ s.getLast? = some b) →
  a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_last_digit_sum_l2245_224510


namespace NUMINAMATH_CALUDE_cube_and_fifth_power_sum_l2245_224558

theorem cube_and_fifth_power_sum (a : ℝ) (h : (a + 1/a)^2 = 11) :
  (a^3 + 1/a^3, a^5 + 1/a^5) = (8 * Real.sqrt 11, 71 * Real.sqrt 11) ∨
  (a^3 + 1/a^3, a^5 + 1/a^5) = (-8 * Real.sqrt 11, -71 * Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_cube_and_fifth_power_sum_l2245_224558


namespace NUMINAMATH_CALUDE_coefficient_x_fourth_power_l2245_224526

theorem coefficient_x_fourth_power (x : ℝ) : 
  (Finset.range 11).sum (fun k => (-1)^k * Nat.choose 10 k * x^(10 - 2*k)) = -120 * x^4 + 
    (Finset.range 11).sum (fun k => if k ≠ 3 then (-1)^k * Nat.choose 10 k * x^(10 - 2*k) else 0) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_fourth_power_l2245_224526


namespace NUMINAMATH_CALUDE_largest_valid_number_l2245_224530

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

theorem largest_valid_number :
  (96433469 : ℕ).digits 10 = [9, 6, 4, 3, 3, 4, 6, 9] ∧
  is_valid_number 96433469 ∧
  ∀ m : ℕ, m > 96433469 → ¬ is_valid_number m :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l2245_224530


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2245_224513

/-- An isosceles triangle with side lengths 2 and 4 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  is_isosceles : base = 2 ∧ leg = 4

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.base + 2 * t.leg

/-- Theorem: The perimeter of an isosceles triangle with side lengths 2 and 4 is 10 -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, perimeter t = 10 := by
  sorry

#check isosceles_triangle_perimeter

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2245_224513


namespace NUMINAMATH_CALUDE_disneyland_attractions_ordering_l2245_224574

def number_of_attractions : ℕ := 6

theorem disneyland_attractions_ordering :
  let total_permutations := Nat.factorial number_of_attractions
  let valid_permutations := total_permutations / 2
  valid_permutations = 360 :=
by sorry

end NUMINAMATH_CALUDE_disneyland_attractions_ordering_l2245_224574


namespace NUMINAMATH_CALUDE_max_value_expression_l2245_224501

theorem max_value_expression (a b : ℝ) (h : a^2 + b^2 = 3 + a*b) :
  (∃ x y : ℝ, x^2 + y^2 = 3 + x*y ∧ (2*x - 3*y)^2 + (x + 2*y)*(x - 2*y) ≤ 22) ∧
  (∃ x y : ℝ, x^2 + y^2 = 3 + x*y ∧ (2*x - 3*y)^2 + (x + 2*y)*(x - 2*y) = 22) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2245_224501


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2245_224502

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h1 : geometric_sequence a) 
  (h2 : a 1 + a 4 = 18) (h3 : a 2 * a 3 = 32) : 
  ∃ q : ℝ, (q = 1/2 ∨ q = 2) ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2245_224502


namespace NUMINAMATH_CALUDE_variance_unchanged_by_constant_shift_l2245_224544

def ages : List ℝ := [15, 13, 15, 14, 13]
def variance (xs : List ℝ) : ℝ := sorry

theorem variance_unchanged_by_constant_shift (c : ℝ) :
  variance ages = variance (ages.map (· + c)) :=
by sorry

end NUMINAMATH_CALUDE_variance_unchanged_by_constant_shift_l2245_224544


namespace NUMINAMATH_CALUDE_power_difference_square_equals_42_times_10_to_1007_l2245_224588

theorem power_difference_square_equals_42_times_10_to_1007 :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 42 * 10^1007 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_square_equals_42_times_10_to_1007_l2245_224588


namespace NUMINAMATH_CALUDE_starting_lineup_count_l2245_224581

/-- Represents a football team -/
structure FootballTeam where
  total_members : ℕ
  offensive_linemen : ℕ
  hm : offensive_linemen ≤ total_members

/-- Calculates the number of ways to choose a starting lineup -/
def starting_lineup_combinations (team : FootballTeam) : ℕ :=
  team.offensive_linemen * (team.total_members - 1) * (team.total_members - 2) * (team.total_members - 3)

/-- Theorem stating the number of ways to choose a starting lineup for the given team -/
theorem starting_lineup_count (team : FootballTeam) 
  (h1 : team.total_members = 12) 
  (h2 : team.offensive_linemen = 4) : 
  starting_lineup_combinations team = 3960 := by
  sorry

#eval starting_lineup_combinations ⟨12, 4, by norm_num⟩

end NUMINAMATH_CALUDE_starting_lineup_count_l2245_224581


namespace NUMINAMATH_CALUDE_max_value_sum_sqrt_l2245_224517

theorem max_value_sum_sqrt (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 8) : 
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_sqrt_l2245_224517


namespace NUMINAMATH_CALUDE_boys_average_age_l2245_224563

/-- Prove that the average age of boys is 12 years in a school with given conditions -/
theorem boys_average_age
  (total_students : ℕ)
  (girls_count : ℕ)
  (girls_avg_age : ℝ)
  (school_avg_age : ℝ)
  (h1 : total_students = 600)
  (h2 : girls_count = 150)
  (h3 : girls_avg_age = 11)
  (h4 : school_avg_age = 11.75) :
  let boys_count : ℕ := total_students - girls_count
  let boys_total_age : ℝ := school_avg_age * total_students - girls_avg_age * girls_count
  boys_total_age / boys_count = 12 := by
  sorry

end NUMINAMATH_CALUDE_boys_average_age_l2245_224563


namespace NUMINAMATH_CALUDE_smallest_among_three_l2245_224580

theorem smallest_among_three : 
  min ((-2)^3) (min (-3^2) (-(-1))) = -3^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_among_three_l2245_224580


namespace NUMINAMATH_CALUDE_unique_bases_sum_l2245_224560

def recurring_decimal (a b : ℕ) (base : ℕ) : ℚ :=
  (a : ℚ) / (base ^ 2 - 1 : ℚ) * base + (b : ℚ) / (base ^ 2 - 1 : ℚ)

theorem unique_bases_sum :
  ∃! (R₁ R₂ : ℕ), 
    R₁ > 1 ∧ R₂ > 1 ∧
    recurring_decimal 3 7 R₁ = recurring_decimal 2 5 R₂ ∧
    recurring_decimal 7 3 R₁ = recurring_decimal 5 2 R₂ ∧
    R₁ + R₂ = 19 :=
by sorry

end NUMINAMATH_CALUDE_unique_bases_sum_l2245_224560


namespace NUMINAMATH_CALUDE_interest_calculation_years_l2245_224505

/-- Calculates the number of years for a given interest scenario -/
def calculate_years (principal : ℝ) (rate : ℝ) (interest_difference : ℝ) : ℝ :=
  let f : ℝ → ℝ := λ n => (1 + rate)^n - 1 - rate * n - interest_difference / principal
  -- We assume the existence of a root-finding function
  sorry

theorem interest_calculation_years :
  let principal : ℝ := 1300
  let rate : ℝ := 0.10
  let interest_difference : ℝ := 13
  calculate_years principal rate interest_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_years_l2245_224505


namespace NUMINAMATH_CALUDE_crayon_distribution_l2245_224559

theorem crayon_distribution (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) : 
  total_crayons = 24 → num_people = 3 → crayons_per_person = total_crayons / num_people → crayons_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_crayon_distribution_l2245_224559


namespace NUMINAMATH_CALUDE_unique_function_solution_l2245_224567

theorem unique_function_solution (f : ℝ → ℝ) : 
  (∀ x ≥ 1, f x ≥ 1) → 
  (∀ x ≥ 1, f x ≤ 2 * (x + 1)) → 
  (∀ x ≥ 1, f (x + 1) = (1 / x) * ((f x)^2 - 1)) → 
  (∀ x ≥ 1, f x = x + 1) := by
sorry

end NUMINAMATH_CALUDE_unique_function_solution_l2245_224567


namespace NUMINAMATH_CALUDE_product_mod_seven_l2245_224516

theorem product_mod_seven : (2015 * 2016 * 2017 * 2018) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l2245_224516


namespace NUMINAMATH_CALUDE_museum_trip_l2245_224582

def bus_trip (first_bus : ℕ) : Prop :=
  let second_bus := 2 * first_bus
  let third_bus := second_bus - 6
  let fourth_bus := first_bus + 9
  let total_people := first_bus + second_bus + third_bus + fourth_bus
  (first_bus ≤ 45) ∧ 
  (second_bus ≤ 45) ∧ 
  (third_bus ≤ 45) ∧ 
  (fourth_bus ≤ 45) ∧ 
  (total_people = 75)

theorem museum_trip : bus_trip 12 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_l2245_224582


namespace NUMINAMATH_CALUDE_marbles_exceed_500_on_day_5_l2245_224590

def marble_sequence (n : ℕ) : ℕ := 4^n

theorem marbles_exceed_500_on_day_5 :
  ∀ k : ℕ, k < 5 → marble_sequence k ≤ 500 ∧ marble_sequence 5 > 500 :=
by sorry

end NUMINAMATH_CALUDE_marbles_exceed_500_on_day_5_l2245_224590
