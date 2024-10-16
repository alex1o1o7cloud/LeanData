import Mathlib

namespace NUMINAMATH_CALUDE_max_obtuse_dihedral_angles_l226_22698

/-- A tetrahedron is a polyhedron with four faces. -/
structure Tetrahedron where
  -- We don't need to define the internal structure for this problem

/-- A dihedral angle is the angle between two intersecting planes. -/
structure DihedralAngle where
  -- We don't need to define the internal structure for this problem

/-- An obtuse angle is an angle greater than 90 degrees but less than 180 degrees. -/
def isObtuse (angle : DihedralAngle) : Prop :=
  sorry  -- Definition of obtuse angle

/-- A tetrahedron has exactly 6 dihedral angles. -/
axiom tetrahedron_has_six_dihedral_angles (t : Tetrahedron) :
  ∃ (angles : Finset DihedralAngle), angles.card = 6

/-- The maximum number of obtuse dihedral angles in a tetrahedron is 3. -/
theorem max_obtuse_dihedral_angles (t : Tetrahedron) :
  ∃ (angles : Finset DihedralAngle),
    (∀ a ∈ angles, isObtuse a) ∧
    angles.card = 3 ∧
    ∀ (other_angles : Finset DihedralAngle),
      (∀ a ∈ other_angles, isObtuse a) →
      other_angles.card ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_obtuse_dihedral_angles_l226_22698


namespace NUMINAMATH_CALUDE_potato_bag_weight_l226_22681

/-- If a bag of potatoes weighs 12 lbs divided by half of its weight, then the weight of the bag is 24 lbs. -/
theorem potato_bag_weight (w : ℝ) (h : w = 12 / (w / 2)) : w = 24 :=
sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l226_22681


namespace NUMINAMATH_CALUDE_expression_problem_l226_22679

theorem expression_problem (a : ℝ) (h : 5 * a = 3125) :
  ∃ b : ℝ, 5 * b = 25 ∧ b = 5 := by
sorry

end NUMINAMATH_CALUDE_expression_problem_l226_22679


namespace NUMINAMATH_CALUDE_expense_recording_l226_22630

/-- Represents the recording of a financial transaction -/
inductive FinancialRecord
  | income (amount : ℤ)
  | expense (amount : ℤ)

/-- Records an income of 5 yuan as +5 -/
def record_income : FinancialRecord := FinancialRecord.income 5

/-- Theorem: If income of 5 yuan is recorded as +5, then expenses of 5 yuan should be recorded as -5 -/
theorem expense_recording (h : record_income = FinancialRecord.income 5) :
  FinancialRecord.expense 5 = FinancialRecord.expense (-5) :=
sorry

end NUMINAMATH_CALUDE_expense_recording_l226_22630


namespace NUMINAMATH_CALUDE_unique_solution_for_diophantine_equation_l226_22664

theorem unique_solution_for_diophantine_equation :
  ∀ m a b : ℤ,
    m > 1 ∧ a > 1 ∧ b > 1 →
    (m + 1) * a = m * b + 1 →
    m = 2 ∧ a = 3 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_diophantine_equation_l226_22664


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l226_22653

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the given tetrahedron is approximately 13.416 -/
theorem volume_of_specific_tetrahedron :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := 3,
    RS := 7
  }
  abs (tetrahedronVolume t - 13.416) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l226_22653


namespace NUMINAMATH_CALUDE_min_value_and_range_l226_22615

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x
def g (a x : ℝ) : ℝ := -x^2 + a*x - 3

def e : ℝ := Real.exp 1

theorem min_value_and_range (t : ℝ) (h : t > 0) :
  (∃ (x : ℝ), x ∈ Set.Icc t (t + 2) ∧
    (∀ (y : ℝ), y ∈ Set.Icc t (t + 2) → f x ≤ f y) ∧
    ((0 < t ∧ t < 1/e → f x = -1/e) ∧
     (t ≥ 1/e → f x = t * Real.log t))) ∧
  (∀ (a : ℝ), (∃ (x₀ : ℝ), x₀ ∈ Set.Icc (1/e) e ∧ 2 * f x₀ ≥ g a x₀) →
    a ≤ -2 + 1/e + 3*e) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_range_l226_22615


namespace NUMINAMATH_CALUDE_repeating_decimal_product_l226_22608

/-- Represents a 5-digit repeating decimal. -/
def RepeatingDecimal (d₁ d₂ d₃ d₄ d₅ : ℕ) : ℚ :=
  (d₁ * 10000 + d₂ * 1000 + d₃ * 100 + d₄ * 10 + d₅) / 99999

theorem repeating_decimal_product (a₁ a₂ a₃ a₄ : ℕ) (b₁ b₂ b₃ b₄ b₅ : ℕ) :
  let a := RepeatingDecimal a₁ a₂ a₃ a₄ 1
  let b := 1 + RepeatingDecimal b₁ b₂ b₃ b₄ b₅
  a * b = 1 → b₅ = 2 := by
  sorry

#check repeating_decimal_product

end NUMINAMATH_CALUDE_repeating_decimal_product_l226_22608


namespace NUMINAMATH_CALUDE_line_plane_intersection_equivalence_l226_22663

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (intersect : Line → Line → Prop)
variable (within : Line → Plane → Prop)
variable (intersects_plane : Line → Plane → Prop)
variable (planes_intersect : Plane → Plane → Prop)

-- Define the specific lines and planes
variable (l m : Line)
variable (α β : Plane)

-- State the theorem
theorem line_plane_intersection_equivalence 
  (h1 : intersect l m)
  (h2 : within l α)
  (h3 : within m α)
  (h4 : ¬ within l β)
  (h5 : ¬ within m β) :
  (intersects_plane l β ∨ intersects_plane m β) ↔ planes_intersect α β := by
  sorry

end NUMINAMATH_CALUDE_line_plane_intersection_equivalence_l226_22663


namespace NUMINAMATH_CALUDE_cosine_sum_identity_l226_22695

theorem cosine_sum_identity (α : ℝ) : 
  Real.cos (π/4 - α) * Real.cos (α + π/12) - Real.sin (π/4 - α) * Real.sin (α + π/12) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_identity_l226_22695


namespace NUMINAMATH_CALUDE_positive_root_irrational_l226_22651

-- Define the equation
def f (x : ℝ) : ℝ := x^5 + x

-- Define the property of being a solution to the equation
def is_solution (x : ℝ) : Prop := f x = 10

-- State the theorem
theorem positive_root_irrational :
  ∃ x > 0, is_solution x ∧ ¬ (∃ (p q : ℤ), q ≠ 0 ∧ x = p / q) :=
by sorry

end NUMINAMATH_CALUDE_positive_root_irrational_l226_22651


namespace NUMINAMATH_CALUDE_max_third_altitude_exists_max_altitude_l226_22677

/-- An isosceles triangle with specific altitude properties -/
structure IsoscelesTriangle where
  -- The lengths of the sides
  AB : ℝ
  BC : ℝ
  -- The altitudes
  h_AB : ℝ
  h_AC : ℝ
  h_BC : ℕ
  -- Isosceles property
  isIsosceles : AB = BC
  -- Given altitude lengths
  alt_AB : h_AB = 6
  alt_AC : h_AC = 18
  -- Triangle inequality
  triangle_inequality : AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB

/-- The theorem stating the maximum possible integer length of the third altitude -/
theorem max_third_altitude (t : IsoscelesTriangle) : t.h_BC ≤ 6 := by
  sorry

/-- The existence of such a triangle with the maximum third altitude -/
theorem exists_max_altitude : ∃ t : IsoscelesTriangle, t.h_BC = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_third_altitude_exists_max_altitude_l226_22677


namespace NUMINAMATH_CALUDE_max_shapes_from_7x7_grid_l226_22633

/-- Represents a grid with dimensions n x n -/
structure Grid (n : ℕ) where
  size : ℕ := n * n

/-- Represents a shape that can be cut from the grid -/
inductive Shape
  | Square : Shape  -- 2x2 square
  | Rectangle : Shape  -- 1x4 rectangle

/-- The size of a shape in terms of grid cells -/
def shapeSize : Shape → ℕ
  | Shape.Square => 4
  | Shape.Rectangle => 4

/-- The maximum number of shapes that can be cut from a grid -/
def maxShapes (g : Grid 7) : ℕ :=
  g.size / shapeSize Shape.Square

/-- Checks if a number of shapes can be equally divided between squares and rectangles -/
def isEquallyDivisible (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k

theorem max_shapes_from_7x7_grid :
  ∃ (n : ℕ), maxShapes (Grid.mk 7) = n ∧ isEquallyDivisible n ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_shapes_from_7x7_grid_l226_22633


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l226_22661

theorem sum_with_radical_conjugate : 
  let x : ℝ := 12 - Real.sqrt 5000
  let y : ℝ := 12 + Real.sqrt 5000  -- radical conjugate
  x + y = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l226_22661


namespace NUMINAMATH_CALUDE_complex_equality_l226_22614

theorem complex_equality (ω : ℂ) :
  Complex.abs (ω - 2) = Complex.abs (ω - 2 * Complex.I) →
  ω.re = ω.im :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_l226_22614


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_congruences_l226_22638

theorem smallest_integer_satisfying_congruences : ∃ b : ℕ, b > 0 ∧
  b % 3 = 2 ∧
  b % 4 = 3 ∧
  b % 5 = 4 ∧
  b % 6 = 5 ∧
  ∀ k : ℕ, k > 0 ∧ k % 3 = 2 ∧ k % 4 = 3 ∧ k % 5 = 4 ∧ k % 6 = 5 → k ≥ b :=
by
  -- Proof goes here
  sorry

#eval 59 % 3  -- Should output 2
#eval 59 % 4  -- Should output 3
#eval 59 % 5  -- Should output 4
#eval 59 % 6  -- Should output 5

end NUMINAMATH_CALUDE_smallest_integer_satisfying_congruences_l226_22638


namespace NUMINAMATH_CALUDE_add_fractions_three_fourths_five_ninths_l226_22631

theorem add_fractions_three_fourths_five_ninths :
  (3 : ℚ) / 4 + (5 : ℚ) / 9 = (47 : ℚ) / 36 := by
  sorry

end NUMINAMATH_CALUDE_add_fractions_three_fourths_five_ninths_l226_22631


namespace NUMINAMATH_CALUDE_sand_art_proof_l226_22671

/-- The amount of sand needed to fill a rectangular patch and a square patch -/
def total_sand_needed (rect_length rect_width square_side sand_per_inch : ℕ) : ℕ :=
  ((rect_length * rect_width) + (square_side * square_side)) * sand_per_inch

/-- Proof that the total amount of sand needed is 201 grams -/
theorem sand_art_proof :
  total_sand_needed 6 7 5 3 = 201 := by
  sorry

end NUMINAMATH_CALUDE_sand_art_proof_l226_22671


namespace NUMINAMATH_CALUDE_arrangements_count_l226_22689

/-- The number of applicants --/
def num_applicants : ℕ := 5

/-- The number of students to be selected --/
def num_selected : ℕ := 3

/-- The number of events --/
def num_events : ℕ := 3

/-- Function to calculate the number of arrangements --/
def num_arrangements (n_applicants : ℕ) (n_selected : ℕ) (n_events : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of arrangements --/
theorem arrangements_count :
  num_arrangements num_applicants num_selected num_events = 48 :=
sorry

end NUMINAMATH_CALUDE_arrangements_count_l226_22689


namespace NUMINAMATH_CALUDE_loss_percent_example_l226_22619

/-- Calculate the loss percent given the cost price and selling price -/
def loss_percent (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that the loss percent is 100/3% when an article is bought for 1200 and sold for 800 -/
theorem loss_percent_example : loss_percent 1200 800 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_loss_percent_example_l226_22619


namespace NUMINAMATH_CALUDE_incorrect_reasoning_l226_22624

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the belonging relation
variable (belongs_to : Point → Line → Prop)
variable (belongs_to_plane : Point → Plane → Prop)

-- Define the subset relation for a line and a plane
variable (line_subset_plane : Line → Plane → Prop)

-- State the theorem
theorem incorrect_reasoning 
  (l : Line) (α : Plane) (A : Point) :
  ¬(∀ (l : Line) (α : Plane) (A : Point), 
    (¬(line_subset_plane l α) ∧ belongs_to A l) → ¬(belongs_to_plane A α)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_reasoning_l226_22624


namespace NUMINAMATH_CALUDE_initial_kids_count_l226_22685

/-- The number of kids still awake after the first round of napping -/
def kids_after_first_round (initial : ℕ) : ℕ := initial / 2

/-- The number of kids still awake after the second round of napping -/
def kids_after_second_round (initial : ℕ) : ℕ := kids_after_first_round initial / 2

/-- Theorem stating that the initial number of kids ready for a nap is 20 -/
theorem initial_kids_count : ∃ (initial : ℕ), 
  kids_after_second_round initial = 5 ∧ initial = 20 := by
  sorry

#check initial_kids_count

end NUMINAMATH_CALUDE_initial_kids_count_l226_22685


namespace NUMINAMATH_CALUDE_neon_signs_blink_together_l226_22675

theorem neon_signs_blink_together (a b c d : ℕ) 
  (ha : a = 7) (hb : b = 11) (hc : c = 13) (hd : d = 17) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 17017 := by
  sorry

end NUMINAMATH_CALUDE_neon_signs_blink_together_l226_22675


namespace NUMINAMATH_CALUDE_lottery_increment_proof_l226_22672

/-- Represents the increment in the price of each successive ticket -/
def increment : ℝ := 1

/-- The number of lottery tickets -/
def num_tickets : ℕ := 5

/-- The price of the first ticket -/
def first_ticket_price : ℝ := 1

/-- The profit Lily plans to keep -/
def profit : ℝ := 4

/-- The prize money for the lottery winner -/
def prize : ℝ := 11

/-- The total amount collected from selling all tickets -/
def total_collected (x : ℝ) : ℝ :=
  first_ticket_price + (first_ticket_price + x) + (first_ticket_price + 2*x) + 
  (first_ticket_price + 3*x) + (first_ticket_price + 4*x)

theorem lottery_increment_proof :
  total_collected increment = profit + prize :=
sorry

end NUMINAMATH_CALUDE_lottery_increment_proof_l226_22672


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l226_22621

/-- An arithmetic sequence {a_n} with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  sum_first_three : a 1 + a 2 + a 3 = -3
  product_first_three : a 1 * a 2 * a 3 = 8
  geometric_subsequence : ∃ r : ℚ, a 2 = r * a 3 ∧ a 3 = r * a 1

/-- The general formula for the arithmetic sequence -/
def general_formula (seq : ArithmeticSequence) (n : ℕ) : ℤ := 3 * n - 7

/-- The sum of the first n terms of the absolute values of the sequence -/
def sum_abs_terms (seq : ArithmeticSequence) (n : ℕ) : ℕ := n^2 - n + 10

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = general_formula seq n) ∧
  (∀ n, sum_abs_terms seq n = n^2 - n + 10) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l226_22621


namespace NUMINAMATH_CALUDE_max_value_implies_a_l226_22654

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - 3

theorem max_value_implies_a (a : ℝ) (h_a : a ≠ 0) :
  (∀ x ∈ Set.Icc (-3/2 : ℝ) 2, f a x ≤ 1) ∧
  (∃ x ∈ Set.Icc (-3/2 : ℝ) 2, f a x = 1) →
  a = 3/4 ∨ a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l226_22654


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l226_22662

theorem rahul_deepak_age_ratio :
  ∀ (rahul_age deepak_age : ℕ),
    deepak_age = 12 →
    rahul_age + 6 = 22 →
    rahul_age / deepak_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l226_22662


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_989_l226_22620

theorem largest_prime_factor_of_989 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 989 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 989 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_989_l226_22620


namespace NUMINAMATH_CALUDE_product_inequality_l226_22609

theorem product_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) :
  a * c > b * d := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l226_22609


namespace NUMINAMATH_CALUDE_smallest_multiple_with_conditions_l226_22683

theorem smallest_multiple_with_conditions : ∃! n : ℕ, 
  n > 0 ∧ 
  47 ∣ n ∧ 
  n % 97 = 7 ∧ 
  n % 31 = 28 ∧ 
  ∀ m : ℕ, m > 0 → 47 ∣ m → m % 97 = 7 → m % 31 = 28 → n ≤ m :=
by
  use 79618
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_conditions_l226_22683


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l226_22660

theorem quadratic_roots_sum_of_squares : 
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 3*x₁ - 5 = 0) → (x₂^2 - 3*x₂ - 5 = 0) → (x₁ ≠ x₂) → 
  x₁^2 + x₂^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l226_22660


namespace NUMINAMATH_CALUDE_average_weight_increase_l226_22696

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 62 →
  new_weight = 90 →
  (new_weight - old_weight) / initial_count = 3.5 :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l226_22696


namespace NUMINAMATH_CALUDE_four_digit_numbers_count_l226_22647

/-- The number of different four-digit numbers that can be formed using two 1s, one 2, and one 0 -/
def four_digit_numbers : ℕ :=
  let zero_placements := 3  -- 0 can be placed in hundreds, tens, or ones place
  let two_placements := 3   -- 2 can be placed in any of the remaining 3 positions
  zero_placements * two_placements

/-- Proof that the number of different four-digit numbers formed is 9 -/
theorem four_digit_numbers_count : four_digit_numbers = 9 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_count_l226_22647


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_separate_from_circle_l226_22674

/-- The hyperbola with equation x^2 - my^2 and eccentricity 3 has asymptotes that are separate from the circle (  )x^2 + y^2 = 7 -/
theorem hyperbola_asymptotes_separate_from_circle 
  (m : ℝ) 
  (hyperbola : ℝ → ℝ → Prop) 
  (circle : ℝ → ℝ → Prop) 
  (eccentricity : ℝ) :
  (∀ x y, hyperbola x y ↔ x^2 - m*y^2 = 1) →
  (∀ x y, circle x y ↔ x^2 + y^2 = 7) →
  eccentricity = 3 →
  ∃ d : ℝ, d > Real.sqrt 7 ∧ 
    (∀ x y, y = 2*Real.sqrt 2*x ∨ y = -2*Real.sqrt 2*x → 
      d ≤ Real.sqrt ((x - 3)^2 + y^2)) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_asymptotes_separate_from_circle_l226_22674


namespace NUMINAMATH_CALUDE_cecil_money_problem_l226_22618

theorem cecil_money_problem (cecil : ℝ) (catherine : ℝ) (carmela : ℝ) 
  (h1 : catherine = 2 * cecil - 250)
  (h2 : carmela = 2 * cecil + 50)
  (h3 : cecil + catherine + carmela = 2800) :
  cecil = 600 := by
sorry

end NUMINAMATH_CALUDE_cecil_money_problem_l226_22618


namespace NUMINAMATH_CALUDE_circle_center_l226_22688

/-- The center of a circle defined by the equation (x+2)^2 + (y-1)^2 = 1 is at the point (-2, 1) -/
theorem circle_center (x y : ℝ) : 
  ((x + 2)^2 + (y - 1)^2 = 1) → ((-2, 1) : ℝ × ℝ) = (x, y) := by
sorry

end NUMINAMATH_CALUDE_circle_center_l226_22688


namespace NUMINAMATH_CALUDE_hedgehog_strawberry_baskets_l226_22646

theorem hedgehog_strawberry_baskets :
  ∀ (baskets : ℕ) (strawberries_per_basket : ℕ) (hedgehogs : ℕ) (strawberries_eaten_per_hedgehog : ℕ),
    strawberries_per_basket = 900 →
    hedgehogs = 2 →
    strawberries_eaten_per_hedgehog = 1050 →
    (baskets * strawberries_per_basket : ℚ) * (2 : ℚ) / 9 = 
      baskets * strawberries_per_basket - hedgehogs * strawberries_eaten_per_hedgehog →
    baskets = 3 := by
  sorry

end NUMINAMATH_CALUDE_hedgehog_strawberry_baskets_l226_22646


namespace NUMINAMATH_CALUDE_num_possible_lists_l226_22692

def num_balls : ℕ := 15
def selections_per_list : ℕ := 2
def num_selections : ℕ := 2

def num_ways_to_select (n k : ℕ) : ℕ := Nat.choose n k

theorem num_possible_lists : 
  (num_ways_to_select num_balls selections_per_list) ^ num_selections = 11025 := by
  sorry

end NUMINAMATH_CALUDE_num_possible_lists_l226_22692


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l226_22616

open Real

theorem sin_shift_equivalence (x : ℝ) :
  3 * sin (2 * x + π / 4) = 3 * sin (2 * (x + π / 8)) :=
by sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l226_22616


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l226_22693

theorem cyclic_sum_inequality (x1 x2 x3 x4 x5 : ℝ) 
  (h_pos : x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ x5 > 0) 
  (h_prod : x1 * x2 * x3 * x4 * x5 = 1) : 
  (x1 + x1*x2*x3)/(1 + x1*x2 + x1*x2*x3*x4) +
  (x2 + x2*x3*x4)/(1 + x2*x3 + x2*x3*x4*x5) +
  (x3 + x3*x4*x5)/(1 + x3*x4 + x3*x4*x5*x1) +
  (x4 + x4*x5*x1)/(1 + x4*x5 + x4*x5*x1*x2) +
  (x5 + x5*x1*x2)/(1 + x5*x1 + x5*x1*x2*x3) ≥ 10/3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l226_22693


namespace NUMINAMATH_CALUDE_problem_solution_l226_22613

theorem problem_solution (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) :
  x^2 - y^2 = 80 ∧ x * y = 96 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l226_22613


namespace NUMINAMATH_CALUDE_amy_tickets_l226_22617

/-- The number of tickets Amy started with -/
def initial_tickets : ℕ := 33

/-- The number of tickets Amy bought -/
def bought_tickets : ℕ := 21

/-- The total number of tickets Amy had -/
def total_tickets : ℕ := 54

theorem amy_tickets : initial_tickets + bought_tickets = total_tickets := by
  sorry

end NUMINAMATH_CALUDE_amy_tickets_l226_22617


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l226_22626

theorem gcd_of_polynomial_and_multiple (x : ℤ) : 
  (∃ k : ℤ, x = 11739 * k) → 
  Nat.gcd ((3*x + 4)*(5*x + 3)*(11*x + 5)*(x + 11)).natAbs x.natAbs = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l226_22626


namespace NUMINAMATH_CALUDE_elizabeth_granola_profit_l226_22643

/-- Calculates the net profit for Elizabeth's granola bag sales --/
theorem elizabeth_granola_profit : 
  let full_price : ℝ := 6.00
  let low_cost : ℝ := 2.50
  let high_cost : ℝ := 3.50
  let low_cost_bags : ℕ := 10
  let high_cost_bags : ℕ := 10
  let full_price_low_cost_sold : ℕ := 7
  let full_price_high_cost_sold : ℕ := 8
  let discounted_low_cost_bags : ℕ := 3
  let discounted_high_cost_bags : ℕ := 2
  let low_cost_discount : ℝ := 0.20
  let high_cost_discount : ℝ := 0.30

  let total_cost : ℝ := low_cost * low_cost_bags + high_cost * high_cost_bags
  let full_price_revenue : ℝ := full_price * (full_price_low_cost_sold + full_price_high_cost_sold)
  let discounted_low_price : ℝ := full_price * (1 - low_cost_discount)
  let discounted_high_price : ℝ := full_price * (1 - high_cost_discount)
  let discounted_revenue : ℝ := discounted_low_price * discounted_low_cost_bags + 
                                 discounted_high_price * discounted_high_cost_bags
  let total_revenue : ℝ := full_price_revenue + discounted_revenue
  let net_profit : ℝ := total_revenue - total_cost

  net_profit = 52.80 := by sorry

end NUMINAMATH_CALUDE_elizabeth_granola_profit_l226_22643


namespace NUMINAMATH_CALUDE_horner_rule_operations_l226_22606

/-- Horner's Rule evaluation for a polynomial -/
def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ × ℕ × ℕ :=
  let rec go : List ℤ → ℤ → ℕ → ℕ → ℤ × ℕ × ℕ
    | [], acc, mults, adds => (acc, mults, adds)
    | c :: cs, acc, mults, adds => go cs (c + x * acc) (mults + 1) (adds + 1)
  go (coeffs.reverse.tail) (coeffs.reverse.head!) 0 0

/-- The polynomial f(x) = 3x^6 + 4x^5 + 5x^4 + 6x^3 + 7x^2 + 8x + 1 -/
def f_coeffs : List ℤ := [1, 8, 7, 6, 5, 4, 3]

theorem horner_rule_operations :
  let (_, mults, adds) := horner_eval f_coeffs 4
  mults = 6 ∧ adds = 6 := by sorry

end NUMINAMATH_CALUDE_horner_rule_operations_l226_22606


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l226_22657

theorem quadratic_equation_solution : 
  ∀ a b : ℝ, 
  (∀ x : ℝ, x^2 - 6*x + 13 = 25 ↔ x = a ∨ x = b) → 
  a ≥ b → 
  3*a + 2*b = 15 + Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l226_22657


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_T_l226_22673

/-- Sum of reciprocals of non-zero digits from 1 to 5^n -/
def T (n : ℕ) : ℚ :=
  -- Definition of T_n (implementation details omitted)
  sorry

/-- The smallest positive integer n for which T_n is an integer -/
theorem smallest_n_for_integer_T : 
  (∀ k < 504, ¬ (T k).isInt) ∧ (T 504).isInt := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_T_l226_22673


namespace NUMINAMATH_CALUDE_goals_scored_over_two_days_l226_22680

/-- The total number of goals scored by Gina and Tom over two days -/
def total_goals (gina_day1 gina_day2 tom_day1 tom_day2 : ℕ) : ℕ :=
  gina_day1 + gina_day2 + tom_day1 + tom_day2

/-- Theorem stating the total number of goals scored by Gina and Tom over two days -/
theorem goals_scored_over_two_days :
  ∃ (gina_day1 gina_day2 tom_day1 tom_day2 : ℕ),
    gina_day1 = 2 ∧
    tom_day1 = gina_day1 + 3 ∧
    tom_day2 = 6 ∧
    gina_day2 = tom_day2 - 2 ∧
    total_goals gina_day1 gina_day2 tom_day1 tom_day2 = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_goals_scored_over_two_days_l226_22680


namespace NUMINAMATH_CALUDE_bowling_ball_weight_is_18_l226_22686

-- Define the weight of one bowling ball
def bowling_ball_weight : ℝ := sorry

-- Define the weight of one kayak
def kayak_weight : ℝ := sorry

-- Theorem to prove the weight of one bowling ball
theorem bowling_ball_weight_is_18 :
  (10 * bowling_ball_weight = 6 * kayak_weight) →
  (3 * kayak_weight = 90) →
  bowling_ball_weight = 18 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_is_18_l226_22686


namespace NUMINAMATH_CALUDE_pine_seedlings_in_sample_l226_22632

/-- Represents a forest with seedlings -/
structure Forest where
  total_seedlings : ℕ
  pine_seedlings : ℕ
  sample_size : ℕ

/-- Calculates the expected number of pine seedlings in a sample -/
def expected_pine_seedlings (f : Forest) : ℚ :=
  (f.pine_seedlings : ℚ) * (f.sample_size : ℚ) / (f.total_seedlings : ℚ)

/-- Theorem stating the expected number of pine seedlings in the sample -/
theorem pine_seedlings_in_sample (f : Forest) 
  (h1 : f.total_seedlings = 30000)
  (h2 : f.pine_seedlings = 4000)
  (h3 : f.sample_size = 150) :
  expected_pine_seedlings f = 20 := by
  sorry

end NUMINAMATH_CALUDE_pine_seedlings_in_sample_l226_22632


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l226_22656

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular n α) : 
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l226_22656


namespace NUMINAMATH_CALUDE_tangent_point_for_equal_volume_l226_22645

theorem tangent_point_for_equal_volume (ξ η : ℝ) : 
  ξ^2 + η^2 = 1 →  -- Point (ξ, η) is on the unit circle
  0 < ξ →          -- ξ is positive (first quadrant)
  ξ < 1 →          -- ξ is less than 1 (valid tangent)
  (((1 - ξ^2)^2 / (3 * ξ)) - ((1 - ξ)^2 * (2 + ξ) / 3)) * π = 4 * π / 3 →  -- Volume equation
  ξ = 3 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_for_equal_volume_l226_22645


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l226_22623

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 7th term of the sequence -/
def a_7 (a : ℕ → ℝ) (m : ℝ) : Prop := a 7 = m

/-- The 14th term of the sequence -/
def a_14 (a : ℕ → ℝ) (n : ℝ) : Prop := a 14 = n

/-- Theorem: In an arithmetic sequence, if a₇ = m and a₁₄ = n, then a₂₁ = 2n - m -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (m n : ℝ) 
  (h1 : arithmetic_sequence a) (h2 : a_7 a m) (h3 : a_14 a n) : 
  a 21 = 2 * n - m := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l226_22623


namespace NUMINAMATH_CALUDE_dmv_waiting_time_l226_22684

theorem dmv_waiting_time (x : ℝ) : 
  x + (4 * x + 14) = 114 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_dmv_waiting_time_l226_22684


namespace NUMINAMATH_CALUDE_inequality_holds_l226_22641

theorem inequality_holds (φ : Real) (h : φ > 0 ∧ φ < Real.pi / 2) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l226_22641


namespace NUMINAMATH_CALUDE_translation_right_4_units_l226_22611

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point to the right -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_right_4_units :
  let P : Point := { x := -5, y := 4 }
  let P' : Point := translateRight P 4
  P'.x = -1 ∧ P'.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_translation_right_4_units_l226_22611


namespace NUMINAMATH_CALUDE_oil_depth_conversion_l226_22691

/-- Represents a right cylindrical tank with oil -/
structure OilTank where
  height : ℝ
  baseDiameter : ℝ
  sideOilDepth : ℝ

/-- Calculates the upright oil depth given a tank configuration -/
noncomputable def uprightOilDepth (tank : OilTank) : ℝ :=
  sorry

/-- Theorem stating the relationship between side oil depth and upright oil depth -/
theorem oil_depth_conversion (tank : OilTank) 
  (h1 : tank.height = 12)
  (h2 : tank.baseDiameter = 6)
  (h3 : tank.sideOilDepth = 2) :
  ∃ (ε : ℝ), abs (uprightOilDepth tank - 2.4) < ε ∧ ε < 0.1 :=
sorry

end NUMINAMATH_CALUDE_oil_depth_conversion_l226_22691


namespace NUMINAMATH_CALUDE_goldfish_problem_l226_22600

/-- The number of goldfish that died -/
def goldfish_died (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem goldfish_problem :
  let initial : ℕ := 89
  let remaining : ℕ := 57
  goldfish_died initial remaining = 32 := by
sorry

end NUMINAMATH_CALUDE_goldfish_problem_l226_22600


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l226_22658

/-- A geometric sequence with common ratio q < 0 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q < 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 2 = 1 - a 1 →
  a 4 = 4 - a 3 →
  a 4 + a 5 = -8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l226_22658


namespace NUMINAMATH_CALUDE_permutations_not_adjacent_l226_22648

/-- The number of permutations of three 'a's, four 'b's, and two 'c's -/
def total_permutations : ℕ := Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 4 * Nat.factorial 2)

/-- Permutations where all 'a's are adjacent -/
def perm_a_adjacent : ℕ := Nat.factorial 7 / (Nat.factorial 4 * Nat.factorial 2)

/-- Permutations where all 'b's are adjacent -/
def perm_b_adjacent : ℕ := Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Permutations where all 'c's are adjacent -/
def perm_c_adjacent : ℕ := Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 4)

/-- Permutations where both 'a's and 'b's are adjacent -/
def perm_ab_adjacent : ℕ := Nat.factorial 4 / Nat.factorial 2

/-- Permutations where both 'a's and 'c's are adjacent -/
def perm_ac_adjacent : ℕ := Nat.factorial 6 / Nat.factorial 4

/-- Permutations where both 'b's and 'c's are adjacent -/
def perm_bc_adjacent : ℕ := Nat.factorial 5 / Nat.factorial 3

/-- Permutations where 'a's, 'b's, and 'c's are all adjacent -/
def perm_abc_adjacent : ℕ := Nat.factorial 3

theorem permutations_not_adjacent : 
  total_permutations - (perm_a_adjacent + perm_b_adjacent + perm_c_adjacent - 
  perm_ab_adjacent - perm_ac_adjacent - perm_bc_adjacent + perm_abc_adjacent) = 871 := by
  sorry

end NUMINAMATH_CALUDE_permutations_not_adjacent_l226_22648


namespace NUMINAMATH_CALUDE_car_average_speed_l226_22601

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 90) (h2 : speed2 = 50) :
  (speed1 + speed2) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l226_22601


namespace NUMINAMATH_CALUDE_machine_production_theorem_l226_22687

/-- Given that 4 machines produce x units in 6 days at a constant rate,
    prove that 16 machines will produce 2x units in 3 days. -/
theorem machine_production_theorem 
  (x : ℝ) -- x is the number of units produced by 4 machines in 6 days
  (h1 : x > 0) -- x is positive
  : 
  let rate := x / (4 * 6) -- rate of production per machine per day
  16 * rate * 3 = 2 * x := by
sorry

end NUMINAMATH_CALUDE_machine_production_theorem_l226_22687


namespace NUMINAMATH_CALUDE_multiple_between_factorials_l226_22699

theorem multiple_between_factorials (n : ℕ) (h : n ≥ 4) :
  ∃ k : ℕ, n.factorial < k * n^3 ∧ k * n^3 < (n + 1).factorial := by
  sorry

end NUMINAMATH_CALUDE_multiple_between_factorials_l226_22699


namespace NUMINAMATH_CALUDE_profit_per_meter_cloth_l226_22622

theorem profit_per_meter_cloth (cloth_length : ℝ) (selling_price : ℝ) (cost_price_per_meter : ℝ)
  (h1 : cloth_length = 80)
  (h2 : selling_price = 6900)
  (h3 : cost_price_per_meter = 66.25) :
  (selling_price - cloth_length * cost_price_per_meter) / cloth_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_per_meter_cloth_l226_22622


namespace NUMINAMATH_CALUDE_train_length_calculation_l226_22602

-- Define the given parameters
def bridge_length : ℝ := 120
def crossing_time : ℝ := 20
def train_speed : ℝ := 66.6

-- State the theorem
theorem train_length_calculation :
  let total_distance := train_speed * crossing_time
  let train_length := total_distance - bridge_length
  train_length = 1212 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l226_22602


namespace NUMINAMATH_CALUDE_intersection_k_value_l226_22670

/-- Given two lines that intersect at a point, find the value of k -/
theorem intersection_k_value (k : ℝ) : 
  (∀ x y, y = 2 * x + 3 → (x = 1 ∧ y = 5)) →  -- Line m passes through (1, 5)
  (∀ x y, y = k * x + 2 → (x = 1 ∧ y = 5)) →  -- Line n passes through (1, 5)
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_k_value_l226_22670


namespace NUMINAMATH_CALUDE_library_books_count_l226_22644

theorem library_books_count :
  ∀ (n : ℕ),
    500 < n ∧ n < 650 ∧
    ∃ (r : ℕ), n = 12 * r + 7 ∧
    ∃ (l : ℕ), n = 25 * l - 5 →
    n = 595 :=
by sorry

end NUMINAMATH_CALUDE_library_books_count_l226_22644


namespace NUMINAMATH_CALUDE_probability_adjacent_vertices_decagon_l226_22669

/-- A decagon is a polygon with 10 vertices -/
def Decagon := Fin 10

/-- Two vertices in a decagon are adjacent if their indices differ by 1 (mod 10) -/
def adjacent (a b : Decagon) : Prop :=
  (a.val + 1) % 10 = b.val ∨ (b.val + 1) % 10 = a.val

/-- The total number of ways to choose 2 distinct vertices from a decagon -/
def total_choices : ℕ := 10 * 9 / 2

/-- The number of ways to choose 2 adjacent vertices from a decagon -/
def adjacent_choices : ℕ := 10

theorem probability_adjacent_vertices_decagon :
  (adjacent_choices : ℚ) / total_choices = 2 / 9 := by
  sorry

#eval (adjacent_choices : ℚ) / total_choices

end NUMINAMATH_CALUDE_probability_adjacent_vertices_decagon_l226_22669


namespace NUMINAMATH_CALUDE_cylinder_dimensions_from_sphere_l226_22697

/-- Given a sphere and a right circular cylinder with equal surface areas,
    prove that the height and diameter of the cylinder are both 14 cm
    when the radius of the sphere is 7 cm. -/
theorem cylinder_dimensions_from_sphere (r : ℝ) (h d : ℝ) : 
  r = 7 →  -- radius of the sphere is 7 cm
  h = d →  -- height and diameter of cylinder are equal
  4 * Real.pi * r^2 = 2 * Real.pi * (d/2) * h →  -- surface areas are equal
  h = 14 ∧ d = 14 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_dimensions_from_sphere_l226_22697


namespace NUMINAMATH_CALUDE_isolation_process_complete_l226_22634

/-- Represents a step in the process of isolating and counting bacteria --/
inductive ProcessStep
  | SoilSampling
  | SampleDilution
  | SpreadingDilution
  | SelectingColonies
  | Identification

/-- Represents the process of isolating and counting bacteria that decompose urea in soil --/
def IsolationProcess : List ProcessStep := 
  [ProcessStep.SoilSampling, 
   ProcessStep.SampleDilution, 
   ProcessStep.SpreadingDilution, 
   ProcessStep.SelectingColonies, 
   ProcessStep.Identification]

/-- The theorem states that the IsolationProcess contains all necessary steps in the correct order --/
theorem isolation_process_complete : 
  IsolationProcess = 
    [ProcessStep.SoilSampling, 
     ProcessStep.SampleDilution, 
     ProcessStep.SpreadingDilution, 
     ProcessStep.SelectingColonies, 
     ProcessStep.Identification] := by
  sorry


end NUMINAMATH_CALUDE_isolation_process_complete_l226_22634


namespace NUMINAMATH_CALUDE_kris_age_l226_22650

/-- Herbert's age next year -/
def herbert_next_year : ℕ := 15

/-- Age difference between Kris and Herbert -/
def age_difference : ℕ := 10

/-- Herbert's current age -/
def herbert_current : ℕ := herbert_next_year - 1

/-- Kris's current age -/
def kris_current : ℕ := herbert_current + age_difference

theorem kris_age : kris_current = 24 := by
  sorry

end NUMINAMATH_CALUDE_kris_age_l226_22650


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_l226_22637

/-- Represents the weight used by the dealer in grams -/
def dealer_weight : ℝ := 500

/-- Represents the standard weight of 1 kg in grams -/
def standard_weight : ℝ := 1000

/-- The dealer's profit percentage -/
def profit_percentage : ℝ := 50

theorem dishonest_dealer_profit :
  dealer_weight / standard_weight = 1 - (100 / (100 + profit_percentage)) :=
sorry

end NUMINAMATH_CALUDE_dishonest_dealer_profit_l226_22637


namespace NUMINAMATH_CALUDE_first_student_completion_time_l226_22607

/-- Given a race with 4 students, prove that if the average completion time of the last 3 students
    is 35 seconds, and the average completion time of all 4 students is 30 seconds,
    then the completion time of the first student is 15 seconds. -/
theorem first_student_completion_time
  (n : ℕ)
  (avg_last_three : ℝ)
  (avg_all : ℝ)
  (h1 : n = 4)
  (h2 : avg_last_three = 35)
  (h3 : avg_all = 30)
  : (n : ℝ) * avg_all - (n - 1 : ℝ) * avg_last_three = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_first_student_completion_time_l226_22607


namespace NUMINAMATH_CALUDE_travis_apple_sales_proof_l226_22666

/-- Calculates the total money Travis will take home from selling apples -/
def travis_apple_sales (total_apples : ℕ) (apples_per_box : ℕ) (price_per_box : ℕ) : ℕ :=
  (total_apples / apples_per_box) * price_per_box

/-- Proves that Travis will take home $7000 from selling his apples -/
theorem travis_apple_sales_proof :
  travis_apple_sales 10000 50 35 = 7000 := by
  sorry

end NUMINAMATH_CALUDE_travis_apple_sales_proof_l226_22666


namespace NUMINAMATH_CALUDE_equation_satisfied_at_five_l226_22678

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 3

-- Define the constant c
def c : ℝ := 11

-- Theorem statement
theorem equation_satisfied_at_five :
  2 * (f 5) - c = f (5 - 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_satisfied_at_five_l226_22678


namespace NUMINAMATH_CALUDE_rational_function_value_at_one_l226_22655

/-- A structure representing a rational function with specific properties. -/
structure RationalFunction where
  r : ℝ → ℝ  -- Numerator polynomial
  s : ℝ → ℝ  -- Denominator polynomial
  is_quadratic_r : ∃ a b c : ℝ, ∀ x, r x = a * x^2 + b * x + c
  is_quadratic_s : ∃ a b c : ℝ, ∀ x, s x = a * x^2 + b * x + c
  hole_at_4 : r 4 = 0 ∧ s 4 = 0
  zero_at_0 : r 0 = 0
  horizontal_asymptote : ∀ ε > 0, ∃ M, ∀ x > M, |r x / s x + 2| < ε
  vertical_asymptote : s 3 = 0 ∧ r 3 ≠ 0

/-- Theorem stating that for a rational function with the given properties, r(1)/s(1) = 1 -/
theorem rational_function_value_at_one (f : RationalFunction) : f.r 1 / f.s 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_at_one_l226_22655


namespace NUMINAMATH_CALUDE_unique_stamp_value_l226_22694

/-- Given stamps of denominations 6, n, and n+1 cents, 
    this function checks if 115 cents is the greatest 
    postage that cannot be formed -/
def is_valid_stamp_set (n : ℕ) : Prop :=
  n > 0 ∧ 
  (∀ m : ℕ, m > 115 → ∃ a b c : ℕ, m = 6*a + n*b + (n+1)*c) ∧
  ¬(∃ a b c : ℕ, 115 = 6*a + n*b + (n+1)*c)

/-- The theorem stating that 24 is the only value of n 
    that satisfies the stamp condition -/
theorem unique_stamp_value : 
  (∃! n : ℕ, is_valid_stamp_set n) ∧ 
  (∀ n : ℕ, is_valid_stamp_set n → n = 24) :=
sorry

end NUMINAMATH_CALUDE_unique_stamp_value_l226_22694


namespace NUMINAMATH_CALUDE_grid_filling_exists_l226_22635

/-- A function representing the grid filling -/
def GridFilling (n : ℕ) := Fin n → Fin n → Fin (2*n - 1)

/-- Predicate to check if a number is a power of 2 -/
def IsPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

/-- Predicate to check if the grid filling is valid -/
def IsValidFilling (n : ℕ) (f : GridFilling n) : Prop :=
  (∀ k : Fin n, ∀ i j : Fin n, i ≠ j → f k i ≠ f k j) ∧
  (∀ k : Fin n, ∀ i j : Fin n, i ≠ j → f i k ≠ f j k)

theorem grid_filling_exists (n : ℕ) (h : IsPowerOfTwo n) :
  ∃ f : GridFilling n, IsValidFilling n f :=
sorry

end NUMINAMATH_CALUDE_grid_filling_exists_l226_22635


namespace NUMINAMATH_CALUDE_factory_production_correct_factory_produces_90_refrigerators_per_hour_l226_22639

/-- Represents the production of a factory making refrigerators and coolers -/
structure FactoryProduction where
  refrigerators_per_hour : ℕ
  coolers_per_hour : ℕ
  total_products : ℕ
  days : ℕ
  hours_per_day : ℕ

/-- The conditions of the factory production problem -/
def factory_conditions : FactoryProduction where
  refrigerators_per_hour := 90  -- This is what we want to prove
  coolers_per_hour := 90 + 70
  total_products := 11250
  days := 5
  hours_per_day := 9

/-- Theorem stating that the given conditions satisfy the problem requirements -/
theorem factory_production_correct (fp : FactoryProduction) : 
  fp.coolers_per_hour = fp.refrigerators_per_hour + 70 →
  fp.total_products = (fp.refrigerators_per_hour + fp.coolers_per_hour) * fp.days * fp.hours_per_day →
  fp.refrigerators_per_hour = 90 :=
by
  sorry

/-- The main theorem proving that the factory produces 90 refrigerators per hour -/
theorem factory_produces_90_refrigerators_per_hour : 
  factory_conditions.refrigerators_per_hour = 90 :=
by
  apply factory_production_correct factory_conditions
  · -- Prove that coolers_per_hour = refrigerators_per_hour + 70
    sorry
  · -- Prove that total_products = (refrigerators_per_hour + coolers_per_hour) * days * hours_per_day
    sorry

end NUMINAMATH_CALUDE_factory_production_correct_factory_produces_90_refrigerators_per_hour_l226_22639


namespace NUMINAMATH_CALUDE_homework_students_l226_22625

theorem homework_students (total : ℕ) (reading : ℕ) (games : ℕ) (homework : ℕ) : 
  total = 24 ∧ 
  reading = total / 2 ∧ 
  games = total / 3 ∧ 
  homework = total - (reading + games) →
  homework = 4 := by
sorry

end NUMINAMATH_CALUDE_homework_students_l226_22625


namespace NUMINAMATH_CALUDE_phi_equals_theta_is_plane_l226_22642

/-- Spherical coordinates in 3D space -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- A generalized plane in 3D space -/
structure GeneralizedPlane where
  equation : SphericalCoord → Prop

/-- The specific equation φ = θ -/
def phiEqualsThetaPlane : GeneralizedPlane where
  equation := fun coord => coord.φ = coord.θ

/-- Theorem: The equation φ = θ in spherical coordinates describes a generalized plane -/
theorem phi_equals_theta_is_plane : 
  ∃ (p : GeneralizedPlane), p = phiEqualsThetaPlane :=
sorry

end NUMINAMATH_CALUDE_phi_equals_theta_is_plane_l226_22642


namespace NUMINAMATH_CALUDE_sin_cos_sum_14_46_l226_22649

theorem sin_cos_sum_14_46 :
  Real.sin (14 * π / 180) * Real.cos (46 * π / 180) +
  Real.sin (46 * π / 180) * Real.cos (14 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_14_46_l226_22649


namespace NUMINAMATH_CALUDE_female_democrats_count_l226_22627

def meeting_participants (total_participants : ℕ) 
  (female_participants : ℕ) (male_participants : ℕ) : Prop :=
  female_participants + male_participants = total_participants

def democrat_ratio (female_democrats : ℕ) (male_democrats : ℕ) 
  (female_participants : ℕ) (male_participants : ℕ) : Prop :=
  female_democrats = female_participants / 2 ∧ 
  male_democrats = male_participants / 4

def total_democrats (female_democrats : ℕ) (male_democrats : ℕ) 
  (total_participants : ℕ) : Prop :=
  female_democrats + male_democrats = total_participants / 3

theorem female_democrats_count : 
  ∀ (total_participants female_participants male_participants 
     female_democrats male_democrats : ℕ),
  total_participants = 990 →
  meeting_participants total_participants female_participants male_participants →
  democrat_ratio female_democrats male_democrats female_participants male_participants →
  total_democrats female_democrats male_democrats total_participants →
  female_democrats = 165 := by
  sorry

end NUMINAMATH_CALUDE_female_democrats_count_l226_22627


namespace NUMINAMATH_CALUDE_abc_sum_l226_22690

theorem abc_sum (A B C : Nat) : 
  A < 10 → B < 10 → C < 10 →  -- A, B, C are single digits
  A ≠ B → B ≠ C → A ≠ C →     -- A, B, C are different
  (100 * A + 10 * B + C) * 4 = 1436 →  -- ABC + ABC + ABC + ABC = 1436
  A + B + C = 17 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_l226_22690


namespace NUMINAMATH_CALUDE_linear_equation_solution_l226_22636

theorem linear_equation_solution :
  ∃! x : ℝ, 8 * x = 2 * x - 6 :=
by
  use -1
  constructor
  · -- Prove that -1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check linear_equation_solution

end NUMINAMATH_CALUDE_linear_equation_solution_l226_22636


namespace NUMINAMATH_CALUDE_joshua_bottle_caps_l226_22652

/-- The number of bottle caps Joshua bought -/
def bottle_caps_bought (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that the number of bottle caps Joshua bought
    is the difference between his final and initial counts -/
theorem joshua_bottle_caps 
  (initial : ℕ) 
  (final : ℕ) 
  (h1 : initial = 40) 
  (h2 : final = 47) :
  bottle_caps_bought initial final = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_joshua_bottle_caps_l226_22652


namespace NUMINAMATH_CALUDE_like_terms_exponents_l226_22628

-- Define a structure for algebraic terms
structure AlgebraicTerm where
  coefficient : ℚ
  a_exponent : ℕ
  b_exponent : ℕ

-- Define what it means for two terms to be like terms
def are_like_terms (t1 t2 : AlgebraicTerm) : Prop :=
  t1.a_exponent = t2.a_exponent ∧ t1.b_exponent = t2.b_exponent

theorem like_terms_exponents 
  (m n : ℕ) 
  (h : are_like_terms 
    (AlgebraicTerm.mk (3 : ℚ) m 2) 
    (AlgebraicTerm.mk (2/3 : ℚ) 1 n)) : 
  m = 1 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l226_22628


namespace NUMINAMATH_CALUDE_total_pictures_uploaded_l226_22604

/-- Proves that the total number of pictures uploaded is 25 -/
theorem total_pictures_uploaded (first_album : ℕ) (num_other_albums : ℕ) (pics_per_other_album : ℕ) 
  (h1 : first_album = 10)
  (h2 : num_other_albums = 5)
  (h3 : pics_per_other_album = 3) :
  first_album + num_other_albums * pics_per_other_album = 25 := by
  sorry

#check total_pictures_uploaded

end NUMINAMATH_CALUDE_total_pictures_uploaded_l226_22604


namespace NUMINAMATH_CALUDE_solve_equation_l226_22667

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.1) : x = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l226_22667


namespace NUMINAMATH_CALUDE_cubic_identity_l226_22676

theorem cubic_identity (x : ℝ) (h : x^3 + 1/x^3 = 116) : x + 1/x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l226_22676


namespace NUMINAMATH_CALUDE_license_advantages_18_vs_30_l226_22665

/-- Represents the age at which a person gets a driver's license -/
inductive LicenseAge
| Age18 : LicenseAge
| Age30 : LicenseAge

/-- Represents the advantages of getting a driver's license -/
structure LicenseAdvantages where
  insuranceCostSavings : Bool
  rentalCarFlexibility : Bool
  employmentOpportunities : Bool

/-- Theorem stating that getting a license at 18 has more advantages than at 30 -/
theorem license_advantages_18_vs_30 :
  ∃ (adv18 adv30 : LicenseAdvantages),
    (adv18.insuranceCostSavings = true ∧
     adv18.rentalCarFlexibility = true ∧
     adv18.employmentOpportunities = true) ∧
    (adv30.insuranceCostSavings = false ∨
     adv30.rentalCarFlexibility = false ∨
     adv30.employmentOpportunities = false) :=
by sorry

end NUMINAMATH_CALUDE_license_advantages_18_vs_30_l226_22665


namespace NUMINAMATH_CALUDE_tomato_cucumber_ratio_l226_22668

/-- Given the initial quantities of tomatoes and cucumbers, and the amounts picked,
    prove that the ratio of remaining tomatoes to remaining cucumbers is 7:68. -/
theorem tomato_cucumber_ratio
  (initial_tomatoes : ℕ)
  (initial_cucumbers : ℕ)
  (tomatoes_picked_yesterday : ℕ)
  (tomatoes_picked_today : ℕ)
  (cucumbers_picked_total : ℕ)
  (h1 : initial_tomatoes = 171)
  (h2 : initial_cucumbers = 225)
  (h3 : tomatoes_picked_yesterday = 134)
  (h4 : tomatoes_picked_today = 30)
  (h5 : cucumbers_picked_total = 157)
  : (initial_tomatoes - (tomatoes_picked_yesterday + tomatoes_picked_today)) /
    (initial_cucumbers - cucumbers_picked_total) = 7 / 68 :=
by sorry

end NUMINAMATH_CALUDE_tomato_cucumber_ratio_l226_22668


namespace NUMINAMATH_CALUDE_unique_row_with_47_l226_22629

def pascal_triangle (n k : ℕ) : ℕ := Nat.choose n k

def contains_47 (row : ℕ) : Prop :=
  ∃ k, pascal_triangle row k = 47

theorem unique_row_with_47 :
  (∃! row, contains_47 row) ∧ (∀ row, contains_47 row → row = 47) :=
sorry

end NUMINAMATH_CALUDE_unique_row_with_47_l226_22629


namespace NUMINAMATH_CALUDE_power_multiplication_l226_22659

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l226_22659


namespace NUMINAMATH_CALUDE_cutting_tool_distance_l226_22682

-- Define the circle and points
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def is_right_angle (A B C : ℝ × ℝ) : Prop :=
  (C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 0

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2

-- State the theorem
theorem cutting_tool_distance (O A B C : ℝ × ℝ) :
  O = (0, 0) →
  A ∈ Circle O (Real.sqrt 72) →
  C ∈ Circle O (Real.sqrt 72) →
  distance_squared A B = 64 →
  distance_squared B C = 9 →
  is_right_angle A B C →
  distance_squared O B = 50 := by
  sorry

end NUMINAMATH_CALUDE_cutting_tool_distance_l226_22682


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l226_22603

/-- For a quadratic equation x^2 - 2x + m = 0 to have real roots, m must be less than or equal to 1 -/
theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l226_22603


namespace NUMINAMATH_CALUDE_work_rates_solution_l226_22610

/-- Work rates of workers -/
structure WorkRates where
  casey : ℚ
  bill : ℚ
  alec : ℚ

/-- Given conditions about job completion times -/
def job_conditions (w : WorkRates) : Prop :=
  10 * (w.casey + w.bill) = 1 ∧
  9 * (w.casey + w.alec) = 1 ∧
  8 * (w.alec + w.bill) = 1

/-- Theorem stating the work rates of Casey, Bill, and Alec -/
theorem work_rates_solution :
  ∃ w : WorkRates,
    job_conditions w ∧
    w.casey = (12.8 - 41) / 720 ∧
    w.bill = 41 / 720 ∧
    w.alec = 49 / 720 := by
  sorry

end NUMINAMATH_CALUDE_work_rates_solution_l226_22610


namespace NUMINAMATH_CALUDE_system_solution_l226_22605

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Calculates the distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Calculates the distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Represents the system of equations -/
def satisfiesSystem (p : Point) (m : ℝ) : Prop :=
  2 * p.x - p.y = m ∧ 3 * p.x + 2 * p.y = m + 7

theorem system_solution :
  (∃ p : Point, satisfiesSystem p 0 ∧ p.x = 1 ∧ p.y = 2) ∧
  (∃ p : Point, ∃ m : ℝ,
    satisfiesSystem p m ∧
    isInSecondQuadrant p ∧
    distanceToXAxis p = 3 ∧
    distanceToYAxis p = 2 ∧
    m = -7) :=
sorry

end NUMINAMATH_CALUDE_system_solution_l226_22605


namespace NUMINAMATH_CALUDE_airplane_seats_proof_l226_22612

-- Define the total number of seats
def total_seats : ℕ := 540

-- Define the number of First Class seats
def first_class_seats : ℕ := 54

-- Define the proportion of Business Class seats
def business_class_proportion : ℚ := 3 / 10

-- Define the proportion of Economy Class seats
def economy_class_proportion : ℚ := 6 / 10

-- Theorem statement
theorem airplane_seats_proof :
  (first_class_seats : ℚ) + 
  (business_class_proportion * total_seats) + 
  (economy_class_proportion * total_seats) = total_seats ∧
  economy_class_proportion = 2 * business_class_proportion :=
by sorry


end NUMINAMATH_CALUDE_airplane_seats_proof_l226_22612


namespace NUMINAMATH_CALUDE_middle_circle_radius_l226_22640

/-- Represents the radii of five circles in an arithmetic sequence -/
def CircleRadii := Fin 5 → ℝ

/-- The property that the radii form an arithmetic sequence -/
def is_arithmetic_sequence (r : CircleRadii) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 4, r (i + 1) = r i + d

/-- The theorem statement -/
theorem middle_circle_radius 
  (r : CircleRadii) 
  (h_arithmetic : is_arithmetic_sequence r)
  (h_smallest : r 0 = 6)
  (h_largest : r 4 = 30) :
  r 2 = 18 := by
sorry

end NUMINAMATH_CALUDE_middle_circle_radius_l226_22640
