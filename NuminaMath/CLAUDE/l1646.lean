import Mathlib

namespace NUMINAMATH_CALUDE_black_ball_probability_l1646_164681

theorem black_ball_probability 
  (n₁ n₂ k₁ k₂ : ℕ) 
  (h_total : n₁ + n₂ = 25)
  (h_white_prob : (k₁ : ℚ) / n₁ * k₂ / n₂ = 54 / 100) :
  (n₁ - k₁ : ℚ) / n₁ * (n₂ - k₂) / n₂ = 4 / 100 := by
sorry

end NUMINAMATH_CALUDE_black_ball_probability_l1646_164681


namespace NUMINAMATH_CALUDE_expression_equivalence_l1646_164615

theorem expression_equivalence :
  let original := -1/2 + Real.sqrt 3 / 2
  let a := -(1 + Real.sqrt 3) / 2
  let b := (Real.sqrt 3 - 1) / 2
  let c := -(1 - Real.sqrt 3) / 2
  let d := (-1 + Real.sqrt 3) / 2
  (a ≠ original) ∧ (b = original) ∧ (c = original) ∧ (d = original) := by
sorry

end NUMINAMATH_CALUDE_expression_equivalence_l1646_164615


namespace NUMINAMATH_CALUDE_inverse_sum_property_l1646_164683

-- Define a function f with domain ℝ and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the property that f is invertible
def is_inverse (f f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- State the theorem
theorem inverse_sum_property
  (h1 : is_inverse f f_inv)
  (h2 : ∀ x : ℝ, f x + f (-x) = 1) :
  ∀ x : ℝ, f_inv (2010 - x) + f_inv (x - 2009) = 0 :=
sorry

end NUMINAMATH_CALUDE_inverse_sum_property_l1646_164683


namespace NUMINAMATH_CALUDE_horseshoe_profit_is_5000_l1646_164646

/-- Calculates the profit for a horseshoe manufacturing company --/
def horseshoe_profit (initial_outlay : ℕ) (cost_per_set : ℕ) (price_per_set : ℕ) (num_sets : ℕ) : ℤ :=
  (price_per_set * num_sets : ℤ) - (initial_outlay + cost_per_set * num_sets : ℤ)

/-- Proves that the profit for the given conditions is $5,000 --/
theorem horseshoe_profit_is_5000 :
  horseshoe_profit 10000 20 50 500 = 5000 := by
  sorry

#eval horseshoe_profit 10000 20 50 500

end NUMINAMATH_CALUDE_horseshoe_profit_is_5000_l1646_164646


namespace NUMINAMATH_CALUDE_cyclists_time_apart_l1646_164684

/-- Calculates the time taken for two cyclists to be 200 miles apart -/
theorem cyclists_time_apart (v_east : ℝ) (v_west : ℝ) (distance : ℝ) : 
  v_east = 22 →
  v_west = v_east + 4 →
  distance = 200 →
  (distance / (v_east + v_west) : ℝ) = 25 / 6 := by
  sorry

#check cyclists_time_apart

end NUMINAMATH_CALUDE_cyclists_time_apart_l1646_164684


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l1646_164664

/-- Given a rectangle ABCD with vertices A(0,0), B(0,2), C(3,2), and D(3,0),
    point E as the midpoint of diagonal BD, and point F on DA such that DF = 1/4 DA,
    prove that the ratio of the area of triangle DFE to the area of quadrilateral ABEF is 3/17. -/
theorem rectangle_area_ratio :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 2)
  let C : ℝ × ℝ := (3, 2)
  let D : ℝ × ℝ := (3, 0)
  let E : ℝ × ℝ := ((D.1 + B.1) / 2, (D.2 + B.2) / 2)
  let F : ℝ × ℝ := (D.1 - (D.1 - A.1) / 4, A.2)
  let area_DFE := abs ((D.1 - F.1) * E.2) / 2
  let area_ABE := abs (B.1 * E.2 - E.1 * B.2) / 2
  let area_AEF := abs ((F.1 - A.1) * E.2) / 2
  let area_ABEF := area_ABE + area_AEF
  area_DFE / area_ABEF = 3 / 17 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l1646_164664


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1646_164608

theorem complex_fraction_evaluation : 
  let f1 : ℚ := 7 / 18
  let f2 : ℚ := 9 / 2  -- 4 1/2 as improper fraction
  let f3 : ℚ := 1 / 6
  let f4 : ℚ := 40 / 3  -- 13 1/3 as improper fraction
  let f5 : ℚ := 15 / 4  -- 3 3/4 as improper fraction
  let f6 : ℚ := 5 / 16
  let f7 : ℚ := 23 / 8  -- 2 7/8 as improper fraction
  (((f1 * f2 + f3) / (f4 - f5 / f6)) * f7) = 529 / 128 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1646_164608


namespace NUMINAMATH_CALUDE_min_disks_for_lilas_problem_l1646_164632

/-- Represents the storage problem with given file sizes and quantities --/
structure StorageProblem where
  total_files : ℕ
  disk_capacity : ℚ
  large_files : ℕ
  large_file_size : ℚ
  medium_files : ℕ
  medium_file_size : ℚ
  small_file_size : ℚ

/-- Calculates the minimum number of disks required for the given storage problem --/
def min_disks_required (problem : StorageProblem) : ℕ :=
  sorry

/-- The specific storage problem instance --/
def lilas_problem : StorageProblem :=
  { total_files := 40
  , disk_capacity := 2
  , large_files := 4
  , large_file_size := 1.2
  , medium_files := 10
  , medium_file_size := 1
  , small_file_size := 0.6 }

/-- Theorem stating that the minimum number of disks required for Lila's problem is 16 --/
theorem min_disks_for_lilas_problem :
  min_disks_required lilas_problem = 16 :=
sorry

end NUMINAMATH_CALUDE_min_disks_for_lilas_problem_l1646_164632


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l1646_164631

theorem multiply_mixed_number : 8 * (9 + 2/5) = 75 + 1/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l1646_164631


namespace NUMINAMATH_CALUDE_trapezoid_area_sum_l1646_164622

/-- Given a trapezoid with side lengths 5, 6, 8, and 9, the sum of all possible areas is 28√3 + 42√2. -/
theorem trapezoid_area_sum :
  ∀ (s₁ s₂ s₃ s₄ : ℝ),
  s₁ = 5 ∧ s₂ = 6 ∧ s₃ = 8 ∧ s₄ = 9 →
  ∃ (A₁ A₂ : ℝ),
  (A₁ = (s₁ + s₄) * Real.sqrt 3 ∧
   A₂ = (s₂ + s₃) * Real.sqrt 2) →
  A₁ + A₂ = 28 * Real.sqrt 3 + 42 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_sum_l1646_164622


namespace NUMINAMATH_CALUDE_square_gt_abs_l1646_164638

theorem square_gt_abs (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_gt_abs_l1646_164638


namespace NUMINAMATH_CALUDE_remaining_black_cards_after_removal_l1646_164652

/-- Represents a deck of cards -/
structure Deck :=
  (black_cards : ℕ)

/-- Calculates the number of remaining black cards after removing some -/
def remaining_black_cards (d : Deck) (removed : ℕ) : ℕ :=
  d.black_cards - removed

/-- Theorem stating that removing 5 black cards from a deck with 26 black cards leaves 21 black cards -/
theorem remaining_black_cards_after_removal :
  ∀ (d : Deck), d.black_cards = 26 → remaining_black_cards d 5 = 21 := by
  sorry


end NUMINAMATH_CALUDE_remaining_black_cards_after_removal_l1646_164652


namespace NUMINAMATH_CALUDE_bowling_pins_difference_l1646_164673

theorem bowling_pins_difference (patrick_first : ℕ) (richard_first_diff : ℕ) (richard_second_diff : ℕ) : 
  patrick_first = 70 →
  richard_first_diff = 15 →
  richard_second_diff = 3 →
  (patrick_first + richard_first_diff + (2 * (patrick_first + richard_first_diff) - richard_second_diff)) -
  (patrick_first + 2 * (patrick_first + richard_first_diff)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_bowling_pins_difference_l1646_164673


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_probability_l1646_164626

/-- Represents an isosceles right-angled triangle -/
structure IsoscelesRightTriangle where
  leg_length : ℝ

/-- Represents a point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The probability of choosing a point within distance 1 from the right angle -/
def probability_within_distance (t : IsoscelesRightTriangle) : ℝ :=
  sorry

theorem isosceles_right_triangle_probability 
  (t : IsoscelesRightTriangle) 
  (h : t.leg_length = 2) : 
  probability_within_distance t = π / 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_probability_l1646_164626


namespace NUMINAMATH_CALUDE_circuit_current_l1646_164613

/-- Given a voltage V and impedance Z as complex numbers,
    prove that the current I = V / Z equals the expected value. -/
theorem circuit_current (V Z : ℂ) (hV : V = 2 + 3*I) (hZ : Z = 4 - 2*I) :
  V / Z = (1 / 10 : ℂ) + (4 / 5 : ℂ) * I :=
by sorry

end NUMINAMATH_CALUDE_circuit_current_l1646_164613


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l1646_164644

theorem ratio_sum_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 3 / 4) (h4 : b = 12) : a + b = 21 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l1646_164644


namespace NUMINAMATH_CALUDE_series_sum_equals_n_l1646_164658

/-- The floor function, denoted as ⌊x⌋, returns the greatest integer less than or equal to x. -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The sum of the series for a given positive integer n -/
noncomputable def series_sum (n : ℕ+) : ℝ :=
  ∑' k : ℕ, (floor ((n : ℝ) + 2^k) / 2^(k+1))

/-- Theorem stating that the sum of the series equals n for every positive integer n -/
theorem series_sum_equals_n (n : ℕ+) : series_sum n = n :=
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_n_l1646_164658


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1646_164627

def U : Set ℕ := {x | 1 < x ∧ x < 5}

def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : 
  (U \ A) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1646_164627


namespace NUMINAMATH_CALUDE_correct_both_problems_l1646_164648

theorem correct_both_problems (total : ℕ) (correct_sets : ℕ) (correct_functions : ℕ) (wrong_both : ℕ)
  (h1 : total = 50)
  (h2 : correct_sets = 40)
  (h3 : correct_functions = 31)
  (h4 : wrong_both = 4) :
  correct_sets + correct_functions - (total - wrong_both) = 25 := by
sorry

end NUMINAMATH_CALUDE_correct_both_problems_l1646_164648


namespace NUMINAMATH_CALUDE_ellipse_specific_constants_l1646_164691

/-- Definition of an ellipse passing through a point -/
def ellipse_passes_through (f1 f2 p : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2)
  let d2 := Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2)
  let c := Real.sqrt ((f2.1 - f1.1)^2 + (f2.2 - f1.2)^2) / 2
  d1 + d2 = 2 * Real.sqrt (c^2 + (d1 + d2)^2 / 4)

/-- The standard form equation of an ellipse -/
def ellipse_equation (x y h k a b : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- Theorem: Ellipse with given foci and point has specific equation constants -/
theorem ellipse_specific_constants :
  let f1 : ℝ × ℝ := (8, 1)
  let f2 : ℝ × ℝ := (8, 9)
  let p : ℝ × ℝ := (17, 5)
  ellipse_passes_through f1 f2 p →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), ellipse_equation x y 8 5 9 (Real.sqrt 97) ↔
      ellipse_equation x y 8 5 a b) :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_specific_constants_l1646_164691


namespace NUMINAMATH_CALUDE_verify_statement_with_flipped_cards_l1646_164662

/-- Represents a card with a letter on one side and a natural number on the other. -/
structure Card where
  letter : Char
  number : Nat

/-- Checks if a character is a vowel. -/
def isVowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']

/-- Checks if a natural number is even. -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/-- Represents the set of cards on the table. -/
def cardsOnTable : List Card := [
  { letter := 'A', number := 0 },
  { letter := 'B', number := 0 },
  { letter := 'C', number := 4 },
  { letter := 'D', number := 5 }
]

/-- The statement to verify for each card. -/
def statementToVerify (c : Card) : Prop :=
  isVowel c.letter → isEven c.number

/-- The set of cards that need to be flipped to verify the statement. -/
def cardsToFlip : List Card :=
  cardsOnTable.filter (fun c => c.letter = 'A' ∨ c.number = 4 ∨ c.number = 5)

/-- Theorem stating that flipping the cards A, 4, and 5 is necessary and sufficient
    to verify the given statement for all cards on the table. -/
theorem verify_statement_with_flipped_cards :
  (∀ c ∈ cardsOnTable, statementToVerify c) ↔
  (∀ c ∈ cardsToFlip, statementToVerify c) :=
sorry

end NUMINAMATH_CALUDE_verify_statement_with_flipped_cards_l1646_164662


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l1646_164600

theorem pentagon_area_sum (u v : ℤ) : 
  0 < v → v < u → (u^2 + 3*u*v = 150) → u + v = 15 := by
  sorry

#check pentagon_area_sum

end NUMINAMATH_CALUDE_pentagon_area_sum_l1646_164600


namespace NUMINAMATH_CALUDE_cubes_occupy_two_thirds_l1646_164649

/-- The dimensions of the rectangular box in inches -/
def box_dimensions : Fin 3 → ℕ
| 0 => 8
| 1 => 6
| 2 => 12
| _ => 0

/-- The side length of a cube in inches -/
def cube_side_length : ℕ := 4

/-- The volume of the rectangular box -/
def box_volume : ℕ := (box_dimensions 0) * (box_dimensions 1) * (box_dimensions 2)

/-- The volume occupied by cubes -/
def cubes_volume : ℕ := 
  ((box_dimensions 0) / cube_side_length) * 
  ((box_dimensions 1) / cube_side_length) * 
  ((box_dimensions 2) / cube_side_length) * 
  (cube_side_length ^ 3)

/-- The percentage of the box volume occupied by cubes -/
def volume_percentage : ℚ := (cubes_volume : ℚ) / (box_volume : ℚ) * 100

theorem cubes_occupy_two_thirds : volume_percentage = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cubes_occupy_two_thirds_l1646_164649


namespace NUMINAMATH_CALUDE_inequality_proof_l1646_164689

theorem inequality_proof (x : ℝ) (h : x > 0) : x + (2016^2016) / (x^2016) ≥ 2017 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1646_164689


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_abs_value_function_l1646_164643

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x - 3) - 4 + 4

-- Theorem statement
theorem minimum_point_of_translated_abs_value_function :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = 0 ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_abs_value_function_l1646_164643


namespace NUMINAMATH_CALUDE_a_greater_than_b_l1646_164605

theorem a_greater_than_b : 
  let a := (-12) * (-23) * (-34) * (-45)
  let b := (-123) * (-234) * (-345)
  a > b := by
sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l1646_164605


namespace NUMINAMATH_CALUDE_function_identity_l1646_164675

theorem function_identity (f : ℤ → ℤ) 
  (h : ∀ m n : ℤ, f (m^2 + f n) = f m^2 + n) : 
  ∀ n : ℤ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l1646_164675


namespace NUMINAMATH_CALUDE_range_of_k_l1646_164629

-- Define the inequality condition
def inequality_condition (k : ℝ) : Prop :=
  ∀ x > 0, Real.exp (x + 1) - (Real.log x + 2 * k) / x - k ≥ 0

-- Theorem statement
theorem range_of_k (k : ℝ) :
  inequality_condition k → k ∈ Set.Iic 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_l1646_164629


namespace NUMINAMATH_CALUDE_complex_sum_equality_l1646_164650

theorem complex_sum_equality : ∃ (r θ : ℝ), 
  5 * Complex.exp (2 * π * Complex.I / 13) + 5 * Complex.exp (17 * π * Complex.I / 26) = 
  r * Complex.exp (θ * Complex.I) ∧ 
  r = 5 * Real.sqrt 2 ∧ 
  θ = 21 * π / 52 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l1646_164650


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1646_164630

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- The unique solution is z = -11
  use (-11 : ℚ)
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1646_164630


namespace NUMINAMATH_CALUDE_x_values_l1646_164661

def U : Set ℕ := Set.univ

def A (x : ℕ) : Set ℕ := {1, 4, x}

def B (x : ℕ) : Set ℕ := {1, x^2}

theorem x_values (x : ℕ) : (Set.compl (A x) ⊂ Set.compl (B x)) → (x = 0 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_x_values_l1646_164661


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l1646_164603

/-- Given three lines in the plane that intersect at two points, 
    prove that the parameter a must be either 1 or -2. -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (y₁ + 2*x₁ - 4 = 0 ∧ x₁ - y₁ + 1 = 0 ∧ a*x₁ - y₁ + 2 = 0) ∧
    (y₂ + 2*x₂ - 4 = 0 ∧ x₂ - y₂ + 1 = 0 ∧ a*x₂ - y₂ + 2 = 0) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) →
  a = 1 ∨ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l1646_164603


namespace NUMINAMATH_CALUDE_easel_cost_l1646_164678

def paintbrush_cost : ℚ := 2.4
def paints_cost : ℚ := 9.2
def rose_has : ℚ := 7.1
def rose_needs : ℚ := 11

theorem easel_cost : 
  let total_cost := rose_has + rose_needs
  let other_items_cost := paintbrush_cost + paints_cost
  total_cost - other_items_cost = 6.5 := by sorry

end NUMINAMATH_CALUDE_easel_cost_l1646_164678


namespace NUMINAMATH_CALUDE_hypotenuse_length_l1646_164696

/-- A right triangle with given perimeter and difference between median and altitude. -/
structure RightTriangle where
  /-- Side length BC -/
  a : ℝ
  /-- Side length AC -/
  b : ℝ
  /-- Hypotenuse length AB -/
  c : ℝ
  /-- Perimeter of the triangle -/
  perimeter_eq : a + b + c = 72
  /-- Pythagorean theorem -/
  pythagoras : a^2 + b^2 = c^2
  /-- Difference between median and altitude -/
  median_altitude_diff : c / 2 - (a * b) / c = 7

/-- The hypotenuse of a right triangle with the given properties is 32 cm. -/
theorem hypotenuse_length (t : RightTriangle) : t.c = 32 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l1646_164696


namespace NUMINAMATH_CALUDE_hcf_of_36_and_84_l1646_164671

theorem hcf_of_36_and_84 : Nat.gcd 36 84 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_36_and_84_l1646_164671


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_range_of_a_l1646_164680

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x ≥ a^2 - a - 2} = {a : ℝ | -1 ≤ a ∧ a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_range_of_a_l1646_164680


namespace NUMINAMATH_CALUDE_trapezoid_DG_length_l1646_164625

-- Define the trapezoids and their properties
structure Trapezoid where
  BC : ℝ
  AD : ℝ
  CT : ℝ
  TD : ℝ
  DG : ℝ

-- Define the theorem
theorem trapezoid_DG_length (ABCD AEFG : Trapezoid) : 
  ABCD.BC = 4 →
  ABCD.AD = 7 →
  ABCD.CT = 1 →
  ABCD.TD = 2 →
  -- ABCD and AEFG are right trapezoids with BC ∥ EF and CD ∥ FG (assumed)
  -- ABCD and AEFG have the same area (assumed)
  AEFG.DG = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_DG_length_l1646_164625


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l1646_164688

theorem infinitely_many_solutions (a b : ℤ) (h_coprime : Nat.Coprime a.natAbs b.natAbs) :
  ∃ (S : Set (ℤ × ℤ × ℤ)), Set.Infinite S ∧
    ∀ (x y z : ℤ), (x, y, z) ∈ S →
      a * x^2 + b * y^2 = z^3 ∧ Nat.Coprime x.natAbs y.natAbs :=
by sorry


end NUMINAMATH_CALUDE_infinitely_many_solutions_l1646_164688


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l1646_164668

/-- Given a cylinder with volume 72π cm³, prove that a cone with double the height 
    and the same radius as the cylinder has a volume of 48π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  (π * r^2 * h = 72 * π) → 
  (1/3 : ℝ) * π * r^2 * (2 * h) = 48 * π := by
sorry


end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l1646_164668


namespace NUMINAMATH_CALUDE_bouquet_composition_l1646_164621

/-- Represents a bouquet of branches -/
structure Bouquet :=
  (white : ℕ)
  (blue : ℕ)

/-- The conditions for our specific bouquet -/
def ValidBouquet (b : Bouquet) : Prop :=
  b.white + b.blue = 7 ∧
  b.white ≥ 1 ∧
  ∀ (x y : ℕ), x < y → x < 7 → y < 7 → (x = b.white → y = b.blue)

/-- The theorem to be proved -/
theorem bouquet_composition (b : Bouquet) (h : ValidBouquet b) : b.white = 1 ∧ b.blue = 6 := by
  sorry


end NUMINAMATH_CALUDE_bouquet_composition_l1646_164621


namespace NUMINAMATH_CALUDE_employed_females_percentage_proof_l1646_164639

/-- The percentage of employed people in town X -/
def employed_percentage : ℝ := 64

/-- The percentage of employed males in town X -/
def employed_males_percentage : ℝ := 48

/-- The percentage of employed females out of the total employed population in town X -/
def employed_females_percentage : ℝ := 25

theorem employed_females_percentage_proof :
  (employed_percentage - employed_males_percentage) / employed_percentage * 100 = employed_females_percentage :=
by sorry

end NUMINAMATH_CALUDE_employed_females_percentage_proof_l1646_164639


namespace NUMINAMATH_CALUDE_cost_of_450_candies_l1646_164620

/-- The cost of buying a given number of chocolate candies -/
def cost_of_candies (candies_per_box : ℕ) (cost_per_box : ℚ) (total_candies : ℕ) : ℚ :=
  (total_candies / candies_per_box) * cost_per_box

/-- Theorem: The cost of 450 chocolate candies is $112.50 -/
theorem cost_of_450_candies :
  cost_of_candies 30 (7.5 : ℚ) 450 = (112.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_cost_of_450_candies_l1646_164620


namespace NUMINAMATH_CALUDE_determinant_equals_four_l1646_164601

/-- The determinant of a 2x2 matrix [[a, b], [c, d]] is ad - bc. -/
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- The matrix in question, parameterized by x. -/
def matrix (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3*x, 2],
    ![x, 2*x]]

theorem determinant_equals_four (x : ℝ) : 
  det2x2 (3*x) 2 x (2*x) = 4 ↔ x = -2/3 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_determinant_equals_four_l1646_164601


namespace NUMINAMATH_CALUDE_batsman_matches_proof_l1646_164602

theorem batsman_matches_proof (first_matches : Nat) (second_matches : Nat) 
  (first_average : Nat) (second_average : Nat) (overall_average : Nat) :
  first_matches = 30 ∧ 
  second_matches = 15 ∧ 
  first_average = 50 ∧ 
  second_average = 26 ∧ 
  overall_average = 42 →
  first_matches + second_matches = 45 := by
  sorry

end NUMINAMATH_CALUDE_batsman_matches_proof_l1646_164602


namespace NUMINAMATH_CALUDE_periodic_function_value_l1646_164667

theorem periodic_function_value (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 4) = f x) 
  (h2 : f 0.5 = 9) : 
  f 8.5 = 9 := by sorry

end NUMINAMATH_CALUDE_periodic_function_value_l1646_164667


namespace NUMINAMATH_CALUDE_sequence_sum_l1646_164604

theorem sequence_sum (n : ℕ) (S_n : ℝ) (a : ℕ → ℝ) : 
  (∀ k ≥ 1, a k = 1 / (Real.sqrt (k + 1) + Real.sqrt k)) →
  S_n = Real.sqrt 101 - 1 →
  n = 100 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1646_164604


namespace NUMINAMATH_CALUDE_faster_train_speed_faster_train_speed_result_l1646_164624

/-- Calculates the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed (length_train1 length_train2 : ℝ)
                            (speed_slower : ℝ)
                            (crossing_time : ℝ) : ℝ :=
  let total_length := length_train1 + length_train2
  let total_length_km := total_length / 1000
  let crossing_time_hours := crossing_time / 3600
  let relative_speed := total_length_km / crossing_time_hours
  speed_slower + relative_speed

/-- The speed of the faster train is approximately 45.95 kmph -/
theorem faster_train_speed_result :
  ∃ ε > 0, |faster_train_speed 200 150 40 210 - 45.95| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_faster_train_speed_result_l1646_164624


namespace NUMINAMATH_CALUDE_min_height_box_l1646_164659

theorem min_height_box (x : ℝ) (h : x > 0) : 
  (2*x^2 + 4*x*(x + 4) ≥ 120) → (x + 4 ≥ 8) :=
by
  sorry

#check min_height_box

end NUMINAMATH_CALUDE_min_height_box_l1646_164659


namespace NUMINAMATH_CALUDE_solution_values_l1646_164623

-- Define the solution sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x : ℝ | x^2 + x - 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the solution set of x^2 + ax + b < 0
def solution_set (a b : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + b < 0}

-- Theorem statement
theorem solution_values :
  ∃ (a b : ℝ), solution_set a b = A_intersect_B ∧ a = -1 ∧ b = -2 :=
sorry

end NUMINAMATH_CALUDE_solution_values_l1646_164623


namespace NUMINAMATH_CALUDE_work_completion_time_l1646_164617

/-- The number of days it takes for A and B together to complete the work -/
def total_days : ℕ := 24

/-- The speed ratio of A to B -/
def speed_ratio : ℕ := 3

/-- The number of days it takes for A alone to complete the work -/
def days_for_A : ℕ := 32

theorem work_completion_time :
  speed_ratio * total_days = (speed_ratio + 1) * days_for_A :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1646_164617


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l1646_164645

def rahul_future_age : ℕ := 32
def years_to_future : ℕ := 4
def deepak_age : ℕ := 21

theorem rahul_deepak_age_ratio :
  let rahul_age := rahul_future_age - years_to_future
  (rahul_age : ℚ) / deepak_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l1646_164645


namespace NUMINAMATH_CALUDE_triangle_properties_l1646_164607

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : a = 5) 
  (h3 : c = 6) 
  (h4 : Real.sin B = 3/5) :
  b = Real.sqrt 13 ∧ 
  Real.sin A = (3 * Real.sqrt 13) / 13 ∧ 
  Real.sin (2 * A + π/4) = (7 * Real.sqrt 2) / 26 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1646_164607


namespace NUMINAMATH_CALUDE_banana_sharing_l1646_164612

/-- Proves that sharing 21 bananas equally among 3 friends results in 7 bananas per friend -/
theorem banana_sharing (total_bananas : ℕ) (num_friends : ℕ) (bananas_per_friend : ℕ) :
  total_bananas = 21 →
  num_friends = 3 →
  bananas_per_friend = total_bananas / num_friends →
  bananas_per_friend = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_sharing_l1646_164612


namespace NUMINAMATH_CALUDE_triangle_area_l1646_164665

/-- The area of a triangle with base 15 cm and height 20 cm is 150 cm². -/
theorem triangle_area : 
  let base : ℝ := 15
  let height : ℝ := 20
  let area : ℝ := (base * height) / 2
  area = 150 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1646_164665


namespace NUMINAMATH_CALUDE_boys_height_ratio_l1646_164642

theorem boys_height_ratio (total_students : ℕ) (boys_under_6ft : ℕ) 
  (h1 : total_students = 100)
  (h2 : boys_under_6ft = 10) :
  (boys_under_6ft : ℚ) / (total_students / 2 : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_height_ratio_l1646_164642


namespace NUMINAMATH_CALUDE_unique_solution_l1646_164660

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  x > 0 ∧ (x - 2) > 0 ∧ (x + 2) > 0 ∧
  log10 x + log10 (x - 2) = log10 3 + log10 (x + 2)

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 6 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1646_164660


namespace NUMINAMATH_CALUDE_summer_pizza_sales_l1646_164655

/-- Given information about pizza sales in different seasons, prove that summer sales are 2 million pizzas. -/
theorem summer_pizza_sales :
  let spring_percent : ℝ := 0.3
  let spring_sales : ℝ := 4.8
  let autumn_sales : ℝ := 7
  let winter_sales : ℝ := 2.2
  let total_sales : ℝ := spring_sales / spring_percent
  let summer_sales : ℝ := total_sales - spring_sales - autumn_sales - winter_sales
  summer_sales = 2 := by
  sorry


end NUMINAMATH_CALUDE_summer_pizza_sales_l1646_164655


namespace NUMINAMATH_CALUDE_day_after_2005_squared_days_l1646_164606

theorem day_after_2005_squared_days (start_day : ℕ) : 
  start_day % 7 = 0 → (start_day + 2005^2) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_day_after_2005_squared_days_l1646_164606


namespace NUMINAMATH_CALUDE_ab_inequality_l1646_164651

theorem ab_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b = 2) : a * b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_inequality_l1646_164651


namespace NUMINAMATH_CALUDE_ellipse_and_range_of_m_l1646_164682

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the square formed by foci and vertices of minor axis -/
def square_perimeter (a b : ℝ) : Prop := 4 * a = 4 * Real.sqrt 2 ∧ b = Real.sqrt (a^2 - b^2)

/-- Definition of point B -/
def point_B (m : ℝ) : ℝ × ℝ := (0, m)

/-- Definition of point D symmetric to B with respect to origin -/
def point_D (m : ℝ) : ℝ × ℝ := (0, -m)

/-- Definition of line l passing through B -/
def line_l (x y m k : ℝ) : Prop := y = k * x + m

/-- Definition of intersection points E and F -/
def intersection_points (x y m k : ℝ) : Prop :=
  ellipse_C x y (Real.sqrt 2) 1 ∧ line_l x y m k

/-- Definition of D being inside circle with diameter EF -/
def D_inside_circle (m : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ,
    intersection_points x₁ y₁ m k ∧
    intersection_points x₂ y₂ m k ∧
    (0 - (x₁ + x₂)/2)^2 + (-m - (y₁ + y₂)/2)^2 < ((x₁ - x₂)^2 + (y₁ - y₂)^2) / 4

/-- Main theorem -/
theorem ellipse_and_range_of_m (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) 
  (h₃ : square_perimeter a b) :
  (ellipse_C x y (Real.sqrt 2) 1 ↔ ellipse_C x y a b) ∧
  (∀ m : ℝ, m > 0 → D_inside_circle m → 0 < m ∧ m < Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_range_of_m_l1646_164682


namespace NUMINAMATH_CALUDE_factorization_proofs_l1646_164687

theorem factorization_proofs (x y : ℝ) :
  (x^2*y - 2*x*y + x*y^2 = x*y*(x - 2 + y)) ∧
  (x^2 - 3*x + 2 = (x - 1)*(x - 2)) ∧
  (4*x^4 - 64 = 4*(x^2 + 4)*(x + 2)*(x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proofs_l1646_164687


namespace NUMINAMATH_CALUDE_principal_calculation_l1646_164641

/-- Prove that given the conditions, the principal amount is 1500 --/
theorem principal_calculation (P : ℝ) : 
  P * (1 + 0.1)^2 - P - (P * 0.1 * 2) = 15 → P = 1500 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l1646_164641


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_equals_sum_minus_l1646_164647

theorem sqrt_sum_squares_equals_sum_minus (a b c : ℝ) :
  (Real.sqrt (a^2 + b^2 + c^2) = a + b - c) ↔ (a * b = c * (a + b) ∧ a + b - c ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_equals_sum_minus_l1646_164647


namespace NUMINAMATH_CALUDE_min_value_cubic_expression_l1646_164663

theorem min_value_cubic_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 12 * b^3 + 27 * c^3 + 1 / (9 * a * b * c) ≥ 4 ∧
  (8 * a^3 + 12 * b^3 + 27 * c^3 + 1 / (9 * a * b * c) = 4 ↔ 
    a = 1 / Real.rpow 8 (1/3) ∧ b = 1 / Real.rpow 12 (1/3) ∧ c = 1 / Real.rpow 27 (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cubic_expression_l1646_164663


namespace NUMINAMATH_CALUDE_smallest_matching_set_size_l1646_164666

theorem smallest_matching_set_size : ∃ (N₁ N₂ : Nat), 
  (10000 ≤ N₁ ∧ N₁ < 100000) ∧ 
  (10000 ≤ N₂ ∧ N₂ < 100000) ∧ 
  ∀ (A : Nat), 
    (10000 ≤ A ∧ A < 100000) → 
    (∀ (i j : Fin 5), i ≤ j → (A / 10^(4 - i.val) % 10) ≤ (A / 10^(4 - j.val) % 10)) →
    ∃ (k : Fin 5), 
      ((N₁ / 10^(4 - k.val)) % 10 = (A / 10^(4 - k.val)) % 10) ∨ 
      ((N₂ / 10^(4 - k.val)) % 10 = (A / 10^(4 - k.val)) % 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_matching_set_size_l1646_164666


namespace NUMINAMATH_CALUDE_dice_throw_outcomes_l1646_164628

/-- The number of possible outcomes for a single dice throw -/
def single_throw_outcomes : ℕ := 6

/-- The number of times the dice is thrown -/
def number_of_throws : ℕ := 2

/-- The total number of different outcomes when throwing a dice twice in succession -/
def total_outcomes : ℕ := single_throw_outcomes ^ number_of_throws

theorem dice_throw_outcomes : total_outcomes = 36 := by
  sorry

end NUMINAMATH_CALUDE_dice_throw_outcomes_l1646_164628


namespace NUMINAMATH_CALUDE_special_triangle_tan_b_l1646_164694

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi
  positive : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the properties of the specific triangle
def SpecialTriangle (t : Triangle) : Prop :=
  -- tan A, tan B, tan C are integers
  ∃ (a b c : ℤ), (Real.tan t.A = a) ∧ (Real.tan t.B = b) ∧ (Real.tan t.C = c) ∧
  -- A > B > C
  (t.A > t.B) ∧ (t.B > t.C) ∧
  -- tan A, tan B, tan C are positive
  (0 < a) ∧ (0 < b) ∧ (0 < c)

-- Theorem statement
theorem special_triangle_tan_b (t : Triangle) (h : SpecialTriangle t) : 
  Real.tan t.B = 2 := by sorry

end NUMINAMATH_CALUDE_special_triangle_tan_b_l1646_164694


namespace NUMINAMATH_CALUDE_sheep_flock_size_l1646_164676

theorem sheep_flock_size :
  ∀ (x y : ℕ),
  (x - 1 : ℚ) / y = 7 / 5 →
  x / (y - 1 : ℚ) = 5 / 3 →
  x + y = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_sheep_flock_size_l1646_164676


namespace NUMINAMATH_CALUDE_quadratic_rewrite_product_l1646_164679

theorem quadratic_rewrite_product (a b c : ℤ) : 
  (∀ x : ℝ, 16 * x^2 - 40 * x - 72 = (a * x + b)^2 + c) → a * b = -20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_product_l1646_164679


namespace NUMINAMATH_CALUDE_total_egg_rolls_l1646_164690

-- Define the number of egg rolls rolled by each person
def omar_rolls : ℕ := 219
def karen_rolls : ℕ := 229
def lily_rolls : ℕ := 275

-- Theorem to prove
theorem total_egg_rolls : omar_rolls + karen_rolls + lily_rolls = 723 := by
  sorry

end NUMINAMATH_CALUDE_total_egg_rolls_l1646_164690


namespace NUMINAMATH_CALUDE_power_of_power_l1646_164616

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1646_164616


namespace NUMINAMATH_CALUDE_rain_probabilities_l1646_164654

/-- The probability of rain in place A -/
def prob_A : ℝ := 0.2

/-- The probability of rain in place B -/
def prob_B : ℝ := 0.3

/-- The probability of no rain in both places A and B -/
def prob_neither : ℝ := (1 - prob_A) * (1 - prob_B)

/-- The probability of rain in exactly one of places A or B -/
def prob_exactly_one : ℝ := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B

/-- The probability of rain in at least one of places A or B -/
def prob_at_least_one : ℝ := 1 - prob_neither

/-- The probability of rain in at most one of places A or B -/
def prob_at_most_one : ℝ := prob_neither + prob_exactly_one

theorem rain_probabilities :
  prob_neither = 0.56 ∧
  prob_exactly_one = 0.38 ∧
  prob_at_least_one = 0.44 ∧
  prob_at_most_one = 0.94 := by
  sorry

end NUMINAMATH_CALUDE_rain_probabilities_l1646_164654


namespace NUMINAMATH_CALUDE_courtyard_tile_cost_l1646_164669

/-- Calculate the total cost of tiles for a courtyard -/
theorem courtyard_tile_cost : 
  let courtyard_length : ℝ := 10
  let courtyard_width : ℝ := 25
  let tiles_per_sqft : ℝ := 4
  let green_tile_percentage : ℝ := 0.4
  let green_tile_cost : ℝ := 3
  let red_tile_cost : ℝ := 1.5

  let total_area : ℝ := courtyard_length * courtyard_width
  let total_tiles : ℝ := total_area * tiles_per_sqft
  let green_tiles : ℝ := green_tile_percentage * total_tiles
  let red_tiles : ℝ := total_tiles - green_tiles

  let total_cost : ℝ := green_tiles * green_tile_cost + red_tiles * red_tile_cost

  total_cost = 2100 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_tile_cost_l1646_164669


namespace NUMINAMATH_CALUDE_waiter_customers_l1646_164692

/-- The number of customers who left -/
def customers_left : ℕ := 12

/-- The number of tables after some customers left -/
def tables_after : ℕ := 4

/-- The number of people at each table after some customers left -/
def people_per_table : ℕ := 8

/-- The initial number of customers in the waiter's section -/
def initial_customers : ℕ := 44

theorem waiter_customers :
  initial_customers = (tables_after * people_per_table) + customers_left :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l1646_164692


namespace NUMINAMATH_CALUDE_male_red_ants_percentage_l1646_164619

/-- Represents the percentage of red ants in the total population -/
def red_percentage : ℝ := 0.85

/-- Represents the percentage of female ants among red ants -/
def female_red_percentage : ℝ := 0.45

/-- Calculates the percentage of male red ants in the total population -/
def male_red_percentage : ℝ := red_percentage * (1 - female_red_percentage)

/-- Theorem stating that the percentage of male red ants in the total population is 46.75% -/
theorem male_red_ants_percentage : 
  male_red_percentage = 0.4675 := by sorry

end NUMINAMATH_CALUDE_male_red_ants_percentage_l1646_164619


namespace NUMINAMATH_CALUDE_monthly_growth_rate_price_reduction_for_profit_l1646_164656

-- Define the given constants
def initial_cost : ℝ := 40
def initial_price : ℝ := 60
def march_sales : ℝ := 192
def may_sales : ℝ := 300
def sales_increase_per_reduction : ℝ := 20  -- 40 pieces per 2 yuan reduction

-- Define the target profit
def target_profit : ℝ := 6080

-- Part 1: Monthly average growth rate
theorem monthly_growth_rate : ∃ (x : ℝ), 
  march_sales * (1 + x)^2 = may_sales ∧ x = 0.25 := by sorry

-- Part 2: Price reduction for target profit
theorem price_reduction_for_profit : ∃ (m : ℝ),
  (initial_price - m - initial_cost) * (may_sales + sales_increase_per_reduction * m) = target_profit ∧
  m = 4 := by sorry

end NUMINAMATH_CALUDE_monthly_growth_rate_price_reduction_for_profit_l1646_164656


namespace NUMINAMATH_CALUDE_unique_solution_cubic_l1646_164640

theorem unique_solution_cubic (c : ℝ) : c = 3/4 ↔ 
  ∃! (b : ℝ), b > 0 ∧ 
    ∃! (x : ℝ), x^3 + x^2 + (b^2 + 1/b^2) * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_l1646_164640


namespace NUMINAMATH_CALUDE_range_of_S_l1646_164618

theorem range_of_S (a b : ℝ) 
  (h : ∀ x ∈ Set.Icc 0 1, |a * x + b| ≤ 1) : 
  ∃ S : ℝ, S = (a + 1) * (b + 1) ∧ -2 ≤ S ∧ S ≤ 9/4 :=
sorry

end NUMINAMATH_CALUDE_range_of_S_l1646_164618


namespace NUMINAMATH_CALUDE_triangle_side_length_l1646_164670

/-- Given a triangle ABC with the specified properties, prove that AC = 5√3 -/
theorem triangle_side_length (A B C : ℝ) (BC : ℝ) :
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  A + B + C = π →
  2 * Real.sin (A - B) + Real.cos (B + C) = 2 →
  BC = 5 →
  ∃ (AC : ℝ), AC = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1646_164670


namespace NUMINAMATH_CALUDE_cookie_sale_total_l1646_164653

theorem cookie_sale_total (raisin_cookies : ℕ) (ratio : ℚ) : 
  raisin_cookies = 42 → ratio = 6/1 → raisin_cookies + (raisin_cookies / ratio.num) = 49 :=
by sorry

end NUMINAMATH_CALUDE_cookie_sale_total_l1646_164653


namespace NUMINAMATH_CALUDE_correct_smaller_type_pages_l1646_164634

/-- Represents the number of pages in smaller type -/
def smaller_type_pages : ℕ := 17

/-- Represents the number of pages in larger type -/
def larger_type_pages : ℕ := 21 - smaller_type_pages

/-- The total number of words in the article -/
def total_words : ℕ := 48000

/-- The number of words per page in larger type -/
def words_per_page_large : ℕ := 1800

/-- The number of words per page in smaller type -/
def words_per_page_small : ℕ := 2400

/-- The total number of pages -/
def total_pages : ℕ := 21

theorem correct_smaller_type_pages : 
  smaller_type_pages = 17 ∧ 
  larger_type_pages + smaller_type_pages = total_pages ∧
  words_per_page_large * larger_type_pages + words_per_page_small * smaller_type_pages = total_words :=
by sorry

end NUMINAMATH_CALUDE_correct_smaller_type_pages_l1646_164634


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l1646_164672

theorem complex_modulus_equality (n : ℝ) :
  n > 0 → Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 13 → n = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l1646_164672


namespace NUMINAMATH_CALUDE_min_value_expression_l1646_164611

theorem min_value_expression (x y : ℝ) :
  Real.sqrt (x^2 + y^2 - 2*x - 2*y + 2) + 
  Real.sqrt (x^2 + y^2 - 2*x + 4*y + 2*Real.sqrt 3*y + 8 + 4*Real.sqrt 3) + 
  Real.sqrt (x^2 + y^2 + 8*x + 4*Real.sqrt 3*x - 4*y + 32 + 16*Real.sqrt 3) ≥ 
  3 * Real.sqrt 6 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1646_164611


namespace NUMINAMATH_CALUDE_tangent_product_simplification_l1646_164610

theorem tangent_product_simplification :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_simplification_l1646_164610


namespace NUMINAMATH_CALUDE_hot_dog_bun_packages_l1646_164677

theorem hot_dog_bun_packages : ∃ n : ℕ, n > 0 ∧ 12 * n % 9 = 0 ∧ ∀ m : ℕ, m > 0 → 12 * m % 9 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_bun_packages_l1646_164677


namespace NUMINAMATH_CALUDE_max_books_borrowed_l1646_164609

/-- Given a college with the following book borrowing statistics:
    - 200 total students
    - 10 students borrowed 0 books
    - 30 students borrowed 1 book each
    - 40 students borrowed 2 books each
    - 50 students borrowed 3 books each
    - 25 students borrowed 5 books each
    - The average number of books per student is 3

    Prove that the maximum number of books any single student could have borrowed is 215. -/
theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (three_books : ℕ) (five_books : ℕ) (avg_books : ℚ) :
  total_students = 200 →
  zero_books = 10 →
  one_book = 30 →
  two_books = 40 →
  three_books = 50 →
  five_books = 25 →
  avg_books = 3 →
  (zero_books + one_book + two_books + three_books + five_books : ℚ) / total_students = avg_books →
  ∃ (max_books : ℕ), max_books = 215 ∧ 
    ∀ (student_books : ℕ), student_books ≤ max_books :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l1646_164609


namespace NUMINAMATH_CALUDE_distance_case1_distance_case2_distance_formula_l1646_164674

-- Define a function to calculate the distance between two points on a number line
def distance (x1 x2 : ℝ) : ℝ := |x2 - x1|

-- Theorem for Case 1
theorem distance_case1 : distance 2 3 = 1 := by sorry

-- Theorem for Case 2
theorem distance_case2 : distance (-4) (-8) = 4 := by sorry

-- General theorem
theorem distance_formula (x1 x2 : ℝ) : 
  distance x1 x2 = |x2 - x1| := by sorry

end NUMINAMATH_CALUDE_distance_case1_distance_case2_distance_formula_l1646_164674


namespace NUMINAMATH_CALUDE_fifteenth_in_base_8_l1646_164633

/-- Converts a decimal number to its representation in base 8 -/
def to_base_8 (n : ℕ) : ℕ := sorry

/-- The fifteenth number in base 10 -/
def fifteenth : ℕ := 15

/-- The representation of the fifteenth number in base 8 -/
def fifteenth_base_8 : ℕ := 17

theorem fifteenth_in_base_8 :
  to_base_8 fifteenth = fifteenth_base_8 := by sorry

end NUMINAMATH_CALUDE_fifteenth_in_base_8_l1646_164633


namespace NUMINAMATH_CALUDE_max_third_side_length_l1646_164657

theorem max_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 15) :
  ∃ (c : ℝ), c ≤ 21 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧
  ∀ (d : ℝ), (d > 21 ∨ d ≤ 0 ∨ a + b ≤ d ∨ a + d ≤ b ∨ b + d ≤ a) →
  ¬(∃ (e : ℕ), e > 21 ∧ (e : ℝ) = d) :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_length_l1646_164657


namespace NUMINAMATH_CALUDE_john_outfit_cost_l1646_164635

/-- Calculates the final cost of John's outfit in Euros -/
def outfit_cost_in_euros (pants_cost shirt_percent_increase shirt_discount outfit_tax
                          hat_cost hat_discount hat_tax
                          shoes_cost shoes_discount shoes_tax
                          usd_to_eur_rate : ℝ) : ℝ :=
  let shirt_cost := pants_cost * (1 + shirt_percent_increase)
  let shirt_discounted := shirt_cost * (1 - shirt_discount)
  let outfit_cost := (pants_cost + shirt_discounted) * (1 + outfit_tax)
  let hat_discounted := hat_cost * (1 - hat_discount)
  let hat_with_tax := hat_discounted * (1 + hat_tax)
  let shoes_discounted := shoes_cost * (1 - shoes_discount)
  let shoes_with_tax := shoes_discounted * (1 + shoes_tax)
  let total_usd := outfit_cost + hat_with_tax + shoes_with_tax
  total_usd * usd_to_eur_rate

/-- The final cost of John's outfit in Euros is approximately 175.93 -/
theorem john_outfit_cost :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
  |outfit_cost_in_euros 50 0.6 0.15 0.07 25 0.1 0.06 70 0.2 0.08 0.85 - 175.93| < ε :=
sorry

end NUMINAMATH_CALUDE_john_outfit_cost_l1646_164635


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1646_164636

theorem unique_solution_for_equation : 
  ∀ m n : ℕ+, 1 + 5 * 2^(m : ℕ) = (n : ℕ)^2 ↔ m = 4 ∧ n = 9 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1646_164636


namespace NUMINAMATH_CALUDE_f_neg_two_f_is_even_l1646_164686

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2 * abs x + 2

-- Define the domain
def domain : Set ℝ := {x : ℝ | -5 ≤ x ∧ x ≤ 5}

-- Theorem 1: f(-2) = 10
theorem f_neg_two : f (-2) = 10 := by sorry

-- Theorem 2: f is an even function on the domain
theorem f_is_even : ∀ x ∈ domain, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_f_neg_two_f_is_even_l1646_164686


namespace NUMINAMATH_CALUDE_probability_of_odd_product_l1646_164637

def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {0, 1, 3}

def is_product_odd (a b : ℕ) : Bool := (a * b) % 2 = 1

def favorable_outcomes : Finset (ℕ × ℕ) :=
  A.product B |>.filter (fun (a, b) => is_product_odd a b)

theorem probability_of_odd_product :
  (favorable_outcomes.card : ℚ) / ((A.card * B.card) : ℚ) = 4 / 9 := by
  sorry

#eval favorable_outcomes -- To check the favorable outcomes
#eval favorable_outcomes.card -- To check the number of favorable outcomes
#eval A.card * B.card -- To check the total number of outcomes

end NUMINAMATH_CALUDE_probability_of_odd_product_l1646_164637


namespace NUMINAMATH_CALUDE_subway_distance_difference_l1646_164698

def distance (s : ℝ) : ℝ := 0.5 * s^3 + s^2

theorem subway_distance_difference : 
  distance 7 - distance 4 = 172.5 := by sorry

end NUMINAMATH_CALUDE_subway_distance_difference_l1646_164698


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l1646_164697

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 10) 
  (h2 : sum_first_two = 7) : 
  ∃ a r : ℝ, 
    S = a / (1 - r) ∧ 
    sum_first_two = a + a * r ∧ 
    a = 10 * (1 + Real.sqrt (3 / 10)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l1646_164697


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l1646_164699

theorem alpha_beta_sum (α β : ℝ) : 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 72*x + 1233) / (x^2 + 81*x - 3969)) →
  α + β = 143 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l1646_164699


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l1646_164614

/-- Given a sequence {a_n} where the sum of its first n terms is S_n = 2n^2 - 3n,
    prove that {a_n} is an arithmetic sequence. -/
theorem sequence_is_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = 2 * n^2 - 3 * n) :
    ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d :=
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l1646_164614


namespace NUMINAMATH_CALUDE_convex_quadrilateral_exists_l1646_164693

-- Define a point in a 2D plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a set of 5 points
def FivePoints := Fin 5 → Point

-- Define the property that no three points are collinear
def NoThreeCollinear (points : FivePoints) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (points i).x * ((points j).y - (points k).y) +
    (points j).x * ((points k).y - (points i).y) +
    (points k).x * ((points i).y - (points j).y) ≠ 0

-- Define a convex quadrilateral
def ConvexQuadrilateral (a b c d : Point) : Prop :=
  -- This is a simplified definition. In practice, we would need to define
  -- convexity more rigorously.
  true

-- The main theorem
theorem convex_quadrilateral_exists (points : FivePoints) 
  (h : NoThreeCollinear points) : 
  ∃ (i j k l : Fin 5), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
    ConvexQuadrilateral (points i) (points j) (points k) (points l) :=
by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_exists_l1646_164693


namespace NUMINAMATH_CALUDE_knowledge_competition_probabilities_l1646_164685

/-- Represents the types of questions available in the competition -/
inductive QuestionType
  | Easy1
  | Easy2
  | Medium
  | Hard

/-- The point value associated with each question type -/
def pointValue : QuestionType → ℕ
  | QuestionType.Easy1 => 10
  | QuestionType.Easy2 => 10
  | QuestionType.Medium => 20
  | QuestionType.Hard => 40

/-- The probability of selecting each question type -/
def selectionProbability : QuestionType → ℚ
  | _ => 1/4

theorem knowledge_competition_probabilities :
  let differentValueProb := 1 - (2/4 * 2/4 + 1/4 * 1/4 + 1/4 * 1/4)
  let greaterValueProb := 1/4 * 2/4 + 1/4 * 3/4
  differentValueProb = 5/8 ∧ greaterValueProb = 5/16 := by
  sorry

#check knowledge_competition_probabilities

end NUMINAMATH_CALUDE_knowledge_competition_probabilities_l1646_164685


namespace NUMINAMATH_CALUDE_correct_calculation_l1646_164695

/-- Represents the cost and quantity relationships of items A and B -/
structure ItemRelationship where
  cost_difference : ℝ  -- Cost difference between A and B
  quantity_ratio : ℝ   -- Ratio of quantities purchasable for 480 yuan
  total_items : ℕ      -- Total number of items to be purchased
  max_cost : ℝ         -- Maximum total cost allowed

/-- Calculates the costs of items A and B and the minimum number of B items to purchase -/
def calculate_costs_and_min_b (r : ItemRelationship) : 
  (ℝ × ℝ × ℕ) :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating the correctness of the calculation -/
theorem correct_calculation (r : ItemRelationship) 
  (h1 : r.cost_difference = 4)
  (h2 : r.quantity_ratio = 3/4)
  (h3 : r.total_items = 200)
  (h4 : r.max_cost = 3000) :
  calculate_costs_and_min_b r = (16, 12, 50) :=
sorry

end NUMINAMATH_CALUDE_correct_calculation_l1646_164695
