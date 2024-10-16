import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_expansion_sum_l267_26704

theorem polynomial_expansion_sum (m : ℝ) (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + m * x)^6 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6) →
  (a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 64) →
  (m = 1 ∨ m = -3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_sum_l267_26704


namespace NUMINAMATH_CALUDE_missing_bulbs_l267_26716

theorem missing_bulbs (total_fixtures : ℕ) (capacity_per_fixture : ℕ) 
  (fixtures_with_4 : ℕ) (fixtures_with_3 : ℕ) (fixtures_with_1 : ℕ) (fixtures_with_0 : ℕ) :
  total_fixtures = 24 →
  capacity_per_fixture = 4 →
  fixtures_with_1 = 2 * fixtures_with_4 →
  fixtures_with_0 = fixtures_with_3 / 2 →
  fixtures_with_4 + fixtures_with_3 + (total_fixtures - fixtures_with_4 - fixtures_with_3 - fixtures_with_1) + fixtures_with_1 + fixtures_with_0 = total_fixtures →
  4 * fixtures_with_4 + 3 * fixtures_with_3 + 2 * (total_fixtures - fixtures_with_4 - fixtures_with_3 - fixtures_with_1) + fixtures_with_1 = total_fixtures * capacity_per_fixture / 2 →
  total_fixtures * capacity_per_fixture - (4 * fixtures_with_4 + 3 * fixtures_with_3 + 2 * (total_fixtures - fixtures_with_4 - fixtures_with_3 - fixtures_with_1) + fixtures_with_1) = 48 :=
by sorry

end NUMINAMATH_CALUDE_missing_bulbs_l267_26716


namespace NUMINAMATH_CALUDE_expense_difference_l267_26747

theorem expense_difference (alice_paid bob_paid carol_paid : ℕ) 
  (h_alice : alice_paid = 120)
  (h_bob : bob_paid = 150)
  (h_carol : carol_paid = 210) : 
  let total := alice_paid + bob_paid + carol_paid
  let each_share := total / 3
  let alice_owes := each_share - alice_paid
  let bob_owes := each_share - bob_paid
  alice_owes - bob_owes = 30 := by
  sorry

end NUMINAMATH_CALUDE_expense_difference_l267_26747


namespace NUMINAMATH_CALUDE_expression_simplification_l267_26775

theorem expression_simplification (x : ℝ) (h : x = 2 + Real.sqrt 2) :
  (((x^2 - 4) / (x^2 - 4*x + 4)) / ((x + 2) / (x + 1))) - (x / (x - 2)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l267_26775


namespace NUMINAMATH_CALUDE_moles_of_CH3Cl_formed_l267_26721

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String

-- Define the available moles
def available_CH4 : ℝ := 1
def available_Cl2 : ℝ := 1

-- Define the reaction
def methane_chlorine_reaction : Reaction :=
  { reactant1 := "CH4"
  , reactant2 := "Cl2"
  , product1 := "CH3Cl"
  , product2 := "HCl" }

-- Theorem statement
theorem moles_of_CH3Cl_formed (reaction : Reaction) 
  (h1 : reaction = methane_chlorine_reaction)
  (h2 : available_CH4 = 1)
  (h3 : available_Cl2 = 1) :
  ∃ (moles_CH3Cl : ℝ), moles_CH3Cl = 1 :=
sorry

end NUMINAMATH_CALUDE_moles_of_CH3Cl_formed_l267_26721


namespace NUMINAMATH_CALUDE_terminal_side_first_quadrant_l267_26789

-- Define the angle in degrees
def angle : ℤ := -685

-- Define a function to normalize an angle to the range [0, 360)
def normalizeAngle (a : ℤ) : ℤ :=
  (a % 360 + 360) % 360

-- Define a function to determine the quadrant of an angle
def quadrant (a : ℤ) : ℕ :=
  let normalizedAngle := normalizeAngle a
  if 0 ≤ normalizedAngle ∧ normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle ∧ normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle ∧ normalizedAngle < 270 then 3
  else 4

-- Theorem statement
theorem terminal_side_first_quadrant :
  quadrant angle = 1 := by sorry

end NUMINAMATH_CALUDE_terminal_side_first_quadrant_l267_26789


namespace NUMINAMATH_CALUDE_factorization_equality_l267_26702

/-- Proves that the factorization of 3x(x - 5) + 4(x - 5) - 2x^2 is (x - 15)(x + 4) for all real x -/
theorem factorization_equality (x : ℝ) : 3*x*(x - 5) + 4*(x - 5) - 2*x^2 = (x - 15)*(x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l267_26702


namespace NUMINAMATH_CALUDE_arthur_total_distance_l267_26772

/-- Represents the distance walked in a single direction --/
structure DirectionalDistance :=
  (blocks : ℕ)

/-- Calculates the total number of blocks walked --/
def total_blocks (east west north south : DirectionalDistance) : ℕ :=
  east.blocks + west.blocks + north.blocks + south.blocks

/-- Converts blocks to miles --/
def blocks_to_miles (blocks : ℕ) : ℚ :=
  (blocks : ℚ) * (1 / 4 : ℚ)

/-- Theorem: Arthur's total walking distance is 5.75 miles --/
theorem arthur_total_distance :
  let east := DirectionalDistance.mk 8
  let north := DirectionalDistance.mk 10
  let south := DirectionalDistance.mk 5
  let west := DirectionalDistance.mk 0
  blocks_to_miles (total_blocks east west north south) = 5.75 := by
  sorry

end NUMINAMATH_CALUDE_arthur_total_distance_l267_26772


namespace NUMINAMATH_CALUDE_two_numbers_difference_l267_26739

theorem two_numbers_difference (x y : ℝ) : 
  x < y ∧ x + y = 34 ∧ y = 22 → y - x = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l267_26739


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l267_26710

/-- The line equation: x - y + 1 = 0 -/
def line_equation (x y : ℝ) : Prop := x - y + 1 = 0

/-- The circle equation: (x + 1)^2 + y^2 = 1 -/
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 0)

theorem line_passes_through_circle_center :
  line_equation (circle_center.1) (circle_center.2) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_circle_center_l267_26710


namespace NUMINAMATH_CALUDE_twenty_squares_in_four_by_five_grid_l267_26776

/-- Represents a grid of points -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Counts the number of squares of a given size in a grid -/
def countSquares (g : Grid) (size : Nat) : Nat :=
  (g.rows - size + 1) * (g.cols - size + 1)

/-- The total number of squares in a grid -/
def totalSquares (g : Grid) : Nat :=
  countSquares g 1 + countSquares g 2 + countSquares g 3

/-- Theorem: In a 4x5 grid, the total number of squares is 20 -/
theorem twenty_squares_in_four_by_five_grid :
  totalSquares ⟨4, 5⟩ = 20 := by
  sorry

#eval totalSquares ⟨4, 5⟩

end NUMINAMATH_CALUDE_twenty_squares_in_four_by_five_grid_l267_26776


namespace NUMINAMATH_CALUDE_section_b_average_weight_l267_26738

/-- Proves that the average weight of section B is 30 kg given the class composition and weight information -/
theorem section_b_average_weight 
  (num_students_a : ℕ) 
  (num_students_b : ℕ) 
  (avg_weight_a : ℝ) 
  (avg_weight_total : ℝ) :
  num_students_a = 26 →
  num_students_b = 34 →
  avg_weight_a = 50 →
  avg_weight_total = 38.67 →
  (num_students_a * avg_weight_a + num_students_b * 30) / (num_students_a + num_students_b) = avg_weight_total :=
by
  sorry

#eval (26 * 50 + 34 * 30) / (26 + 34) -- Should output approximately 38.67

end NUMINAMATH_CALUDE_section_b_average_weight_l267_26738


namespace NUMINAMATH_CALUDE_jungkook_apples_l267_26756

theorem jungkook_apples (initial_apples given_apples : ℕ) : 
  initial_apples = 8 → given_apples = 7 → initial_apples + given_apples = 15 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_apples_l267_26756


namespace NUMINAMATH_CALUDE_furniture_production_max_profit_l267_26735

/-- Represents the problem of maximizing profit in furniture production --/
theorem furniture_production_max_profit :
  let x : ℝ := 16  -- Number of sets of type A furniture
  let y : ℝ := -0.3 * x + 80  -- Total profit function
  let time_constraint : Prop := (5/4) * x + (5/3) * (100 - x) ≤ 160  -- Time constraint
  let total_sets : Prop := x + (100 - x) = 100  -- Total number of sets
  let profit_decreasing : Prop := ∀ x₁ x₂, x₁ < x₂ → (-0.3 * x₁ + 80) > (-0.3 * x₂ + 80)  -- Profit decreases as x increases
  
  -- The following conditions hold:
  time_constraint ∧
  total_sets ∧
  profit_decreasing ∧
  (∀ x' : ℝ, x' ≥ 0 → x' ≤ 100 → (5/4) * x' + (5/3) * (100 - x') ≤ 160 → y ≥ -0.3 * x' + 80) →
  
  -- Then the maximum profit is achieved:
  y = 75.2 := by sorry

end NUMINAMATH_CALUDE_furniture_production_max_profit_l267_26735


namespace NUMINAMATH_CALUDE_special_geometric_sequence_q_values_l267_26762

/-- A geometric sequence with special properties -/
structure SpecialGeometricSequence where
  a : ℕ+ → ℕ+
  q : ℕ+
  first_term : a 1 = 2^81
  geometric : ∀ n : ℕ+, a (n + 1) = a n * q
  product_closure : ∀ m n : ℕ+, ∃ p : ℕ+, a m * a n = a p

/-- The set of all possible values for the common ratio q -/
def possible_q_values : Set ℕ+ :=
  {2^81, 2^27, 2^9, 2^3, 2}

/-- Main theorem: The set of all possible values of q for a SpecialGeometricSequence -/
theorem special_geometric_sequence_q_values (seq : SpecialGeometricSequence) :
  seq.q ∈ possible_q_values := by
  sorry

end NUMINAMATH_CALUDE_special_geometric_sequence_q_values_l267_26762


namespace NUMINAMATH_CALUDE_total_players_l267_26759

theorem total_players (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ)
  (h1 : kabadi = 10)
  (h2 : kho_kho_only = 30)
  (h3 : both = 5) :
  kabadi - both + kho_kho_only = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_players_l267_26759


namespace NUMINAMATH_CALUDE_labeled_cube_probabilities_l267_26781

/-- A cube with 6 faces, where 1 face is labeled with 1, 2 faces are labeled with 2, and 3 faces are labeled with 3 -/
structure LabeledCube where
  total_faces : ℕ
  faces_with_1 : ℕ
  faces_with_2 : ℕ
  faces_with_3 : ℕ
  face_sum : total_faces = faces_with_1 + faces_with_2 + faces_with_3
  face_distribution : faces_with_1 = 1 ∧ faces_with_2 = 2 ∧ faces_with_3 = 3

/-- The probability of an event occurring when rolling the cube -/
def probability (cube : LabeledCube) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / cube.total_faces

theorem labeled_cube_probabilities (cube : LabeledCube) :
  (probability cube cube.faces_with_2 = 1/3) ∧
  (∀ n, probability cube (cube.faces_with_1) ≤ probability cube n ∧
        probability cube (cube.faces_with_2) ≤ probability cube n →
        n = cube.faces_with_3) ∧
  (probability cube (cube.faces_with_1 + cube.faces_with_2) =
   probability cube cube.faces_with_3) :=
by sorry

end NUMINAMATH_CALUDE_labeled_cube_probabilities_l267_26781


namespace NUMINAMATH_CALUDE_complement_of_A_l267_26752

-- Define the set A
def A : Set ℝ := {x : ℝ | x^2 + 2*x ≥ 0}

-- State the theorem
theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l267_26752


namespace NUMINAMATH_CALUDE_parabola_intersection_l267_26763

theorem parabola_intersection (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2 * b * x₁ + c = 0 ∧ a * x₂^2 + 2 * b * x₂ + c = 0) ∨
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ b * x₁^2 + 2 * c * x₁ + a = 0 ∧ b * x₂^2 + 2 * c * x₂ + a = 0) ∨
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ c * x₁^2 + 2 * a * x₁ + b = 0 ∧ c * x₂^2 + 2 * a * x₂ + b = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_l267_26763


namespace NUMINAMATH_CALUDE_train_speed_l267_26719

/-- The speed of a train given its length and time to cross an electric pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 700) (h2 : time = 20) :
  length / time = 35 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l267_26719


namespace NUMINAMATH_CALUDE_two_roots_theorem_l267_26736

theorem two_roots_theorem (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (x₁ - a) * (x₁ - b) + (x₁ - a) * (x₁ - c) + (x₁ - b) * (x₁ - c) = 0 ∧
    (x₂ - a) * (x₂ - b) + (x₂ - a) * (x₂ - c) + (x₂ - b) * (x₂ - c) = 0 ∧
    a < x₁ ∧ x₁ < b ∧ b < x₂ ∧ x₂ < c :=
by sorry

end NUMINAMATH_CALUDE_two_roots_theorem_l267_26736


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l267_26714

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  (∃ a : ℝ, a = 1 → abs a = 1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ abs a = 1) := by
  sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l267_26714


namespace NUMINAMATH_CALUDE_consecutive_integers_operation_l267_26707

theorem consecutive_integers_operation (n : ℕ) (h1 : n = 9) : 
  let f : ℕ → ℕ → ℕ := λ x y => x + y + 162
  f n (n + 1) = n * (n + 1) + 91 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_operation_l267_26707


namespace NUMINAMATH_CALUDE_profit_sales_ratio_change_l267_26793

/-- Calculates the percent change between two ratios -/
def percent_change (old_ratio new_ratio : ℚ) : ℚ :=
  ((new_ratio - old_ratio) / old_ratio) * 100

theorem profit_sales_ratio_change :
  let first_quarter_profit : ℚ := 5
  let first_quarter_sales : ℚ := 15
  let third_quarter_profit : ℚ := 14
  let third_quarter_sales : ℚ := 35
  let first_quarter_ratio := first_quarter_profit / first_quarter_sales
  let third_quarter_ratio := third_quarter_profit / third_quarter_sales
  percent_change first_quarter_ratio third_quarter_ratio = 20 := by
sorry

#eval percent_change (5/15) (14/35)

end NUMINAMATH_CALUDE_profit_sales_ratio_change_l267_26793


namespace NUMINAMATH_CALUDE_inverse_square_relation_l267_26751

/-- Given that x varies inversely as the square of y, prove that y = 6 when x = 0.25,
    given that y = 3 when x = 1. -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) (h1 : x = k / (y ^ 2)) 
    (h2 : 1 = k / (3 ^ 2)) (h3 : 0.25 = k / (y ^ 2)) : y = 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l267_26751


namespace NUMINAMATH_CALUDE_students_with_cat_and_dog_l267_26754

theorem students_with_cat_and_dog (total : ℕ) (cat : ℕ) (dog : ℕ) (neither : ℕ) 
  (h1 : total = 28)
  (h2 : cat = 17)
  (h3 : dog = 10)
  (h4 : neither = 5)
  : ∃ both : ℕ, both = cat + dog - (total - neither) :=
by
  sorry

end NUMINAMATH_CALUDE_students_with_cat_and_dog_l267_26754


namespace NUMINAMATH_CALUDE_commute_time_difference_l267_26725

/-- Given a set of 5 commuting times (a, b, 8, 9, 10) with an average of 9 and a variance of 2, prove that |a-b| = 4 -/
theorem commute_time_difference (a b : ℝ) 
  (h_mean : (a + b + 8 + 9 + 10) / 5 = 9)
  (h_variance : ((a - 9)^2 + (b - 9)^2 + (8 - 9)^2 + (9 - 9)^2 + (10 - 9)^2) / 5 = 2) :
  |a - b| = 4 := by
sorry

end NUMINAMATH_CALUDE_commute_time_difference_l267_26725


namespace NUMINAMATH_CALUDE_alley_width_l267_26796

theorem alley_width (l k h w : Real) : 
  l > 0 → 
  k > 0 → 
  h > 0 → 
  w > 0 → 
  k = l * Real.sin (π / 3) → 
  h = l * Real.sin (π / 6) → 
  w = k / Real.sqrt 3 → 
  w = h * Real.sqrt 3 → 
  w = l / 2 := by sorry

end NUMINAMATH_CALUDE_alley_width_l267_26796


namespace NUMINAMATH_CALUDE_hyperbola_equation_l267_26744

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the hyperbola -/
def Hyperbola.contains (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Checks if a line is an asymptote of the hyperbola -/
def Hyperbola.is_asymptote (h : Hyperbola) (m : ℝ) : Prop :=
  m = h.b / h.a ∨ m = -h.b / h.a

/-- The main theorem -/
theorem hyperbola_equation (h : Hyperbola) :
  h.contains 3 (Real.sqrt 2) ∧
  h.is_asymptote (1/3) ∧
  h.is_asymptote (-1/3) →
  h.a^2 = 153 ∧ h.b^2 = 17 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l267_26744


namespace NUMINAMATH_CALUDE_f_1384_bounds_l267_26701

/-- An n-mino is a shape made up of n equal squares connected edge-to-edge. -/
def Mino (n : ℕ) : Type := Unit  -- We don't need to define the full structure for this proof

/-- f(n) is the least number such that there exists an f(n)-mino containing every n-mino -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating the bounds for f(1384) -/
theorem f_1384_bounds : 10000 ≤ f 1384 ∧ f 1384 ≤ 960000 := by sorry

end NUMINAMATH_CALUDE_f_1384_bounds_l267_26701


namespace NUMINAMATH_CALUDE_jessica_exam_time_l267_26700

/-- Calculates the remaining time for Jessica to finish her exam -/
def remaining_time (total_time minutes_used questions_total questions_answered : ℕ) : ℕ :=
  total_time - minutes_used

/-- Proves that Jessica will have 48 minutes left when she finishes the exam -/
theorem jessica_exam_time : remaining_time 60 12 80 16 = 48 := by
  sorry

end NUMINAMATH_CALUDE_jessica_exam_time_l267_26700


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l267_26717

theorem gcd_lcm_product (a b : ℕ) (ha : a = 24) (hb : b = 54) :
  Nat.gcd a b * Nat.lcm a b = a * b := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l267_26717


namespace NUMINAMATH_CALUDE_find_number_l267_26753

theorem find_number : ∃! x : ℝ, 0.6 * ((x / 1.2) - 22.5) + 10.5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l267_26753


namespace NUMINAMATH_CALUDE_smallest_deletion_for_order_l267_26741

theorem smallest_deletion_for_order (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℕ, k = n - Int.ceil (Real.sqrt n) ∧
    (∀ perm : List ℕ, perm.length = n → perm.toFinset = Finset.range n →
      ∃ subseq : List ℕ, subseq.length = n - k ∧ 
        (subseq.Sorted (·<·) ∨ subseq.Sorted (·>·)) ∧
        subseq.toFinset ⊆ perm.toFinset) ∧
    (∀ k' : ℕ, k' < k →
      ∃ perm : List ℕ, perm.length = n ∧ perm.toFinset = Finset.range n ∧
        ∀ subseq : List ℕ, subseq.length > n - k' →
          subseq.toFinset ⊆ perm.toFinset →
            ¬(subseq.Sorted (·<·) ∨ subseq.Sorted (·>·))) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_deletion_for_order_l267_26741


namespace NUMINAMATH_CALUDE_cost_change_l267_26758

theorem cost_change (t : ℝ) (b₁ b₂ : ℝ) (h : t * b₂^4 = 16 * t * b₁^4) :
  b₂ = 2 * b₁ := by
  sorry

end NUMINAMATH_CALUDE_cost_change_l267_26758


namespace NUMINAMATH_CALUDE_monkey_to_snake_ratio_l267_26705

/-- Represents the number of animals in John's zoo --/
structure ZooAnimals where
  snakes : ℕ
  monkeys : ℕ
  lions : ℕ
  pandas : ℕ
  dogs : ℕ

/-- Conditions for John's zoo --/
def zoo_conditions (z : ZooAnimals) : Prop :=
  z.snakes = 15 ∧
  z.lions = z.monkeys - 5 ∧
  z.pandas = z.lions + 8 ∧
  z.dogs * 3 = z.pandas ∧
  z.snakes + z.monkeys + z.lions + z.pandas + z.dogs = 114

/-- Theorem stating the ratio of monkeys to snakes is 2:1 --/
theorem monkey_to_snake_ratio (z : ZooAnimals) (h : zoo_conditions z) :
  z.monkeys = 2 * z.snakes := by
  sorry

end NUMINAMATH_CALUDE_monkey_to_snake_ratio_l267_26705


namespace NUMINAMATH_CALUDE_transaction_gain_per_year_l267_26731

/-- Calculate simple interest -/
def simpleInterest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem transaction_gain_per_year
  (principal : ℚ)
  (borrowRate lendRate : ℚ)
  (time : ℚ)
  (h1 : principal = 8000)
  (h2 : borrowRate = 4)
  (h3 : lendRate = 6)
  (h4 : time = 2) :
  (simpleInterest principal lendRate time - simpleInterest principal borrowRate time) / time = 160 := by
  sorry

end NUMINAMATH_CALUDE_transaction_gain_per_year_l267_26731


namespace NUMINAMATH_CALUDE_correct_outfits_l267_26742

-- Define the colors
inductive Color
| Red
| Blue

-- Define the clothing types
inductive ClothingType
| Tshirt
| Shorts

-- Define a structure for a child's outfit
structure Outfit :=
  (tshirt : Color)
  (shorts : Color)

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

def outfit : Child → Outfit
| Child.Alyna => ⟨Color.Red, Color.Red⟩
| Child.Bohdan => ⟨Color.Red, Color.Blue⟩
| Child.Vika => ⟨Color.Blue, Color.Blue⟩
| Child.Grysha => ⟨Color.Red, Color.Blue⟩

theorem correct_outfits :
  (outfit Child.Alyna).tshirt = Color.Red ∧
  (outfit Child.Bohdan).tshirt = Color.Red ∧
  (outfit Child.Alyna).shorts ≠ (outfit Child.Bohdan).shorts ∧
  (outfit Child.Vika).tshirt ≠ (outfit Child.Grysha).tshirt ∧
  (outfit Child.Vika).shorts = Color.Blue ∧
  (outfit Child.Grysha).shorts = Color.Blue ∧
  (outfit Child.Alyna).tshirt ≠ (outfit Child.Vika).tshirt ∧
  (outfit Child.Alyna).shorts ≠ (outfit Child.Vika).shorts ∧
  (∀ c : Child, (outfit c).tshirt = Color.Red ∨ (outfit c).tshirt = Color.Blue) ∧
  (∀ c : Child, (outfit c).shorts = Color.Red ∨ (outfit c).shorts = Color.Blue) :=
by sorry

#check correct_outfits

end NUMINAMATH_CALUDE_correct_outfits_l267_26742


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l267_26760

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 36)
  (area2 : w * h = 18)
  (area3 : l * h = 12) :
  l * w * h = 36 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l267_26760


namespace NUMINAMATH_CALUDE_product_of_sum_of_roots_l267_26788

theorem product_of_sum_of_roots (x : ℝ) :
  (Real.sqrt (8 + x) + Real.sqrt (15 - x) = 6) →
  (8 + x) * (15 - x) = 169 / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_roots_l267_26788


namespace NUMINAMATH_CALUDE_brendan_grass_cutting_l267_26703

/-- Brendan's grass cutting capacity over a week -/
theorem brendan_grass_cutting (initial_capacity : ℝ) (increase_percentage : ℝ) (days_in_week : ℕ) :
  initial_capacity = 8 →
  increase_percentage = 0.5 →
  days_in_week = 7 →
  (initial_capacity + initial_capacity * increase_percentage) * days_in_week = 84 := by
  sorry

end NUMINAMATH_CALUDE_brendan_grass_cutting_l267_26703


namespace NUMINAMATH_CALUDE_peach_difference_l267_26712

theorem peach_difference (jill_peaches steven_peaches jake_peaches : ℕ) : 
  jill_peaches = 12 →
  jake_peaches + 1 = jill_peaches →
  steven_peaches = jake_peaches + 16 →
  steven_peaches - jill_peaches = 15 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l267_26712


namespace NUMINAMATH_CALUDE_inequality_proof_l267_26748

theorem inequality_proof (a b c e f : ℝ) 
  (h1 : a > b) (h2 : e > f) (h3 : c > 0) : f - a*c < e - b*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l267_26748


namespace NUMINAMATH_CALUDE_min_value_quadratic_l267_26798

theorem min_value_quadratic : 
  (∀ x : ℝ, x^2 + 4*x + 5 ≥ 1) ∧ (∃ x : ℝ, x^2 + 4*x + 5 = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l267_26798


namespace NUMINAMATH_CALUDE_tank_fill_time_l267_26749

/-- The time to fill a tank with a pump and a leak -/
theorem tank_fill_time (pump_rate leak_rate : ℝ) (pump_rate_pos : pump_rate > 0) 
  (leak_rate_pos : leak_rate > 0) (pump_faster : pump_rate > leak_rate) :
  let fill_time := 1 / (pump_rate - leak_rate)
  fill_time = 1 / (1 / 2 - 1 / 26) :=
by
  sorry

#eval 1 / (1 / 2 - 1 / 26)

end NUMINAMATH_CALUDE_tank_fill_time_l267_26749


namespace NUMINAMATH_CALUDE_larger_interior_angle_measure_l267_26791

/-- A circular arch bridge constructed with congruent isosceles trapezoids -/
structure CircularArchBridge where
  /-- The number of trapezoids in the bridge construction -/
  num_trapezoids : ℕ
  /-- The measure of the larger interior angle of each trapezoid in degrees -/
  larger_interior_angle : ℝ
  /-- The two end trapezoids rest horizontally on the ground -/
  end_trapezoids_horizontal : Prop

/-- Theorem stating the measure of the larger interior angle in a circular arch bridge with 12 trapezoids -/
theorem larger_interior_angle_measure (bridge : CircularArchBridge) 
  (h1 : bridge.num_trapezoids = 12)
  (h2 : bridge.end_trapezoids_horizontal) :
  bridge.larger_interior_angle = 97.5 := by
  sorry

#check larger_interior_angle_measure

end NUMINAMATH_CALUDE_larger_interior_angle_measure_l267_26791


namespace NUMINAMATH_CALUDE_linear_coefficient_of_given_quadratic_l267_26766

/-- The coefficient of the linear term in a quadratic equation ax^2 + bx + c = 0 is b. -/
def linearCoefficient (a b c : ℝ) : ℝ := b

/-- The quadratic equation x^2 - 2x - 1 = 0 -/
def quadraticEquation (x : ℝ) : Prop := x^2 - 2*x - 1 = 0

theorem linear_coefficient_of_given_quadratic :
  linearCoefficient 1 (-2) (-1) = -2 := by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_given_quadratic_l267_26766


namespace NUMINAMATH_CALUDE_tangent_line_equation_no_collinear_intersection_l267_26765

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = k * x + 2}

-- Define the circle Q
def circle_Q : Set (ℝ × ℝ) :=
  {(x, y) | x^2 + y^2 - 12*x + 32 = 0}

-- Define the point P
def point_P : ℝ × ℝ := (0, 2)

-- Define the center of the circle Q
def center_Q : ℝ × ℝ := (6, 0)

-- Define the tangency condition
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) ∈ line_l k ∩ circle_Q ∧
  ∀ (x' y' : ℝ), (x', y') ∈ line_l k ∩ circle_Q → (x', y') = (x, y)

-- Define the intersection condition
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ line_l k ∩ circle_Q ∧
  (x₂, y₂) ∈ line_l k ∩ circle_Q ∧ (x₁, y₁) ≠ (x₂, y₂)

-- Define collinearity condition
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), B - A = t • (C - A)

theorem tangent_line_equation :
  ∀ k : ℝ, is_tangent k →
  (∀ x y : ℝ, (x, y) ∈ line_l k ↔ y = 2 ∨ 3*x + 4*y = 8) :=
sorry

theorem no_collinear_intersection :
  ¬∃ k : ℝ, intersects_at_two_points k ∧
  (∀ A B : ℝ × ℝ, A ∈ circle_Q → B ∈ circle_Q → A ≠ B →
   are_collinear (0, 0) (A.1 + B.1, A.2 + B.2) (6, -2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_no_collinear_intersection_l267_26765


namespace NUMINAMATH_CALUDE_extreme_values_and_maximum_b_l267_26774

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / x * Real.exp x

noncomputable def g (a b x : ℝ) : ℝ := a * (x - 1) * Real.exp x - f a b x

theorem extreme_values_and_maximum_b :
  (∀ x : ℝ, x ≠ 0 → f 2 1 x ≤ 1 / Real.exp 1) ∧
  (∀ x : ℝ, x ≠ 0 → f 2 1 x ≥ 4 * Real.sqrt (Real.exp 1)) ∧
  (∃ x : ℝ, x ≠ 0 ∧ f 2 1 x = 1 / Real.exp 1) ∧
  (∃ x : ℝ, x ≠ 0 ∧ f 2 1 x = 4 * Real.sqrt (Real.exp 1)) ∧
  (∀ b : ℝ, (∀ x : ℝ, x > 0 → g 1 b x ≥ 1) → b ≤ -1 - 1 / Real.exp 1) ∧
  (∀ x : ℝ, x > 0 → g 1 (-1 - 1 / Real.exp 1) x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_maximum_b_l267_26774


namespace NUMINAMATH_CALUDE_savings_multiple_l267_26718

/-- Represents a worker's monthly finances -/
structure WorkerFinances where
  takehome : ℝ  -- Monthly take-home pay
  savingsRate : ℝ  -- Fraction of take-home pay saved each month
  months : ℕ  -- Number of months

/-- Calculates the total amount saved over a given number of months -/
def totalSaved (w : WorkerFinances) : ℝ :=
  w.takehome * w.savingsRate * w.months

/-- Calculates the amount not saved in one month -/
def monthlyUnsaved (w : WorkerFinances) : ℝ :=
  w.takehome * (1 - w.savingsRate)

/-- Theorem stating that for a worker saving 1/4 of their take-home pay,
    the total saved over 12 months is 4 times the monthly unsaved amount -/
theorem savings_multiple (w : WorkerFinances)
    (h1 : w.savingsRate = 1/4)
    (h2 : w.months = 12) :
    totalSaved w = 4 * monthlyUnsaved w := by
  sorry


end NUMINAMATH_CALUDE_savings_multiple_l267_26718


namespace NUMINAMATH_CALUDE_total_students_l267_26797

theorem total_students (general biology chemistry math arts : ℕ) : 
  general = 30 ∧ 
  biology = 2 * general ∧ 
  chemistry = general + 10 ∧ 
  math = (3 * (general + biology + chemistry)) / 5 ∧ 
  arts * 20 / 100 = general → 
  general + biology + chemistry + math + arts = 358 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l267_26797


namespace NUMINAMATH_CALUDE_expression_value_l267_26770

theorem expression_value : 
  let x : ℕ := 2
  2 + 2 * (2 * 2) = 10 := by sorry

end NUMINAMATH_CALUDE_expression_value_l267_26770


namespace NUMINAMATH_CALUDE_sum_of_four_cubes_equals_1812_l267_26785

theorem sum_of_four_cubes_equals_1812 :
  (303 : ℤ)^3 + (301 : ℤ)^3 + (-302 : ℤ)^3 + (-302 : ℤ)^3 = 1812 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_cubes_equals_1812_l267_26785


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l267_26722

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_third_term (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 = 2 → a 5 + a 7 = 2 * a 4 + 4 → a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l267_26722


namespace NUMINAMATH_CALUDE_angle_b_in_axisymmetric_triangle_l267_26726

-- Define an axisymmetric triangle
structure AxisymmetricTriangle :=
  (A B C : ℝ)
  (axisymmetric : True)  -- This is a placeholder for the axisymmetric property
  (sum_of_angles : A + B + C = 180)

-- Theorem statement
theorem angle_b_in_axisymmetric_triangle 
  (triangle : AxisymmetricTriangle) 
  (angle_a_value : triangle.A = 70) :
  triangle.B = 70 ∨ triangle.B = 55 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_in_axisymmetric_triangle_l267_26726


namespace NUMINAMATH_CALUDE_quadratic_inequality_implications_l267_26730

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 - 6 * x + 3

-- State the theorem
theorem quadratic_inequality_implications (a : ℝ) :
  (∀ x : ℝ, f a x > 0) →
  (a > 3 ∧ ∀ b : ℝ, b > 3 → a + 9 / (a - 1) ≤ b + 9 / (b - 1)) ∧
  (a + 9 / (a - 1) ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implications_l267_26730


namespace NUMINAMATH_CALUDE_tank_problem_solution_l267_26706

def tank_problem (capacity : ℝ) (initial_fill : ℝ) (empty_percent : ℝ) (refill_percent : ℝ) : ℝ :=
  let initial_volume := capacity * initial_fill
  let emptied_volume := initial_volume * empty_percent
  let remaining_volume := initial_volume - emptied_volume
  let refilled_volume := remaining_volume * refill_percent
  remaining_volume + refilled_volume

theorem tank_problem_solution :
  tank_problem 8000 (3/4) 0.4 0.3 = 4680 := by
  sorry

end NUMINAMATH_CALUDE_tank_problem_solution_l267_26706


namespace NUMINAMATH_CALUDE_hyperbola_C_different_asymptote_l267_26713

-- Define the hyperbolas
def hyperbola_A (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1
def hyperbola_B (x y : ℝ) : Prop := y^2 / 4 - x^2 / 9 = 1
def hyperbola_C (x y : ℝ) : Prop := x^2 / 4 - y^2 / 9 = 1
def hyperbola_D (x y : ℝ) : Prop := y^2 / 12 - x^2 / 27 = 1

-- Define the asymptote
def is_asymptote (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y → (2 * x = 3 * y ∨ 2 * x = -3 * y)

-- Theorem statement
theorem hyperbola_C_different_asymptote :
  ¬(is_asymptote hyperbola_C) ∧
  (is_asymptote hyperbola_A) ∧
  (is_asymptote hyperbola_B) ∧
  (is_asymptote hyperbola_D) := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_C_different_asymptote_l267_26713


namespace NUMINAMATH_CALUDE_least_frood_drop_beats_eat_l267_26783

theorem least_frood_drop_beats_eat : 
  ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, k > 0 → k < n → (k * (k + 1)) / 2 ≤ 15 * k) ∧ (n * (n + 1)) / 2 > 15 * n :=
by sorry

end NUMINAMATH_CALUDE_least_frood_drop_beats_eat_l267_26783


namespace NUMINAMATH_CALUDE_count_valid_insertions_l267_26743

/-- The number of different three-digit numbers that can be inserted into 689???20312 to make it approximately 69 billion when rounded -/
def valid_insertions : ℕ :=
  let ten_million_digits := {5, 6, 7, 8, 9}
  let other_digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  (Finset.card ten_million_digits) * (Finset.card other_digits) * (Finset.card other_digits)

theorem count_valid_insertions : valid_insertions = 500 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_insertions_l267_26743


namespace NUMINAMATH_CALUDE_tracy_candies_problem_l267_26734

theorem tracy_candies_problem (x : ℕ) : 
  x > 0 ∧ 
  x % 4 = 0 ∧ 
  (x * 3 / 4 * 2 / 3 - 20 - 3 = 7) → 
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_tracy_candies_problem_l267_26734


namespace NUMINAMATH_CALUDE_function_properties_l267_26755

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem function_properties (a : ℝ) :
  (∀ x : ℝ, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) →
  (a = 2 ∧ 
   ∀ x : ℝ, f 2 (3*x) + f 2 (x+3) ≥ 5/3 ∧
   ∃ x : ℝ, f 2 (3*x) + f 2 (x+3) = 5/3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l267_26755


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l267_26790

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_cond : x + y + z = 2)
  (x_bound : x ≥ -1/2)
  (y_bound : y ≥ -2)
  (z_bound : z ≥ -3)
  (xy_cond : 2*x + y = 1) :
  ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ + y₀ + z₀ = 2 ∧ 
    2*x₀ + y₀ = 1 ∧
    x₀ ≥ -1/2 ∧ 
    y₀ ≥ -2 ∧ 
    z₀ ≥ -3 ∧
    ∀ x y z, 
      x + y + z = 2 → 
      2*x + y = 1 → 
      x ≥ -1/2 → 
      y ≥ -2 → 
      z ≥ -3 →
      Real.sqrt (4*x + 2) + Real.sqrt (3*y + 6) + Real.sqrt (4*z + 12) ≤ 
      Real.sqrt (4*x₀ + 2) + Real.sqrt (3*y₀ + 6) + Real.sqrt (4*z₀ + 12) ∧
      Real.sqrt (4*x₀ + 2) + Real.sqrt (3*y₀ + 6) + Real.sqrt (4*z₀ + 12) = Real.sqrt 68 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l267_26790


namespace NUMINAMATH_CALUDE_spy_is_A_l267_26769

/-- Represents the three defendants -/
inductive Defendant : Type
  | A
  | B
  | C

/-- Represents the role of each defendant -/
inductive Role : Type
  | Spy
  | Knight
  | Liar

/-- The statement made by each defendant -/
def statement (d : Defendant) : Prop :=
  match d with
  | Defendant.A => ∃ r, r = Role.Spy
  | Defendant.B => ∃ r, r = Role.Knight
  | Defendant.C => ∃ r, r = Role.Spy

/-- The role assigned to each defendant -/
def assigned_role : Defendant → Role := sorry

/-- A defendant tells the truth if they are the Knight or if they are the Spy and claim to be the Spy -/
def tells_truth (d : Defendant) : Prop :=
  (assigned_role d = Role.Knight) ∨
  (assigned_role d = Role.Spy ∧ statement d)

theorem spy_is_A :
  (∃! d : Defendant, assigned_role d = Role.Spy) ∧
  (∃! d : Defendant, assigned_role d = Role.Knight) ∧
  (∃! d : Defendant, assigned_role d = Role.Liar) ∧
  (tells_truth Defendant.B) →
  assigned_role Defendant.A = Role.Spy := by
  sorry


end NUMINAMATH_CALUDE_spy_is_A_l267_26769


namespace NUMINAMATH_CALUDE_expression_simplification_l267_26771

theorem expression_simplification (x y : ℝ) :
  (-2 * x^2 * y) * (-3 * x * y)^2 / (3 * x * y^2) = -6 * x^3 * y := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l267_26771


namespace NUMINAMATH_CALUDE_qi_winning_probability_l267_26723

-- Define the horse strengths
structure HorseStrengths where
  tian_top_better_than_qi_middle : Prop
  tian_top_worse_than_qi_top : Prop
  tian_middle_better_than_qi_bottom : Prop
  tian_middle_worse_than_qi_middle : Prop
  tian_bottom_worse_than_qi_bottom : Prop

-- Define the probability of Qi's horse winning
def probability_qi_wins (strengths : HorseStrengths) : ℚ := 2/3

-- Theorem statement
theorem qi_winning_probability (strengths : HorseStrengths) :
  probability_qi_wins strengths = 2/3 := by sorry

end NUMINAMATH_CALUDE_qi_winning_probability_l267_26723


namespace NUMINAMATH_CALUDE_approx48000_accurate_to_thousand_l267_26727

/-- Represents an approximate value with its numerical value and accuracy -/
structure ApproximateValue where
  value : ℕ
  accuracy : ℕ

/-- Checks if the given approximate value is accurate to thousand -/
def isAccurateToThousand (av : ApproximateValue) : Prop :=
  av.accuracy = 1000

/-- The approximate value 48,000 -/
def approx48000 : ApproximateValue :=
  { value := 48000, accuracy := 1000 }

/-- Theorem stating that 48,000 is accurate to thousand -/
theorem approx48000_accurate_to_thousand :
  isAccurateToThousand approx48000 := by
  sorry

end NUMINAMATH_CALUDE_approx48000_accurate_to_thousand_l267_26727


namespace NUMINAMATH_CALUDE_rectangular_prism_base_area_l267_26787

theorem rectangular_prism_base_area :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a % 5 = 0 ∧ b % 5 = 0 ∧ a * b = 450 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_base_area_l267_26787


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l267_26720

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a < x^2 + 1
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, a < 3 - x₀^2

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬(p a)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l267_26720


namespace NUMINAMATH_CALUDE_c_minus_a_positive_l267_26784

/-- A quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The graph of the quadratic function is a downward-opening parabola -/
def is_downward_opening (f : QuadraticFunction) : Prop :=
  f.a < 0

/-- The y-intercept of the quadratic function is positive -/
def has_positive_y_intercept (f : QuadraticFunction) : Prop :=
  f.c > 0

/-- Theorem stating that if a quadratic function's graph is a downward-opening parabola
    with a positive y-intercept, then c - a > 0 -/
theorem c_minus_a_positive (f : QuadraticFunction)
  (h1 : is_downward_opening f)
  (h2 : has_positive_y_intercept f) :
  f.c - f.a > 0 := by
  sorry

end NUMINAMATH_CALUDE_c_minus_a_positive_l267_26784


namespace NUMINAMATH_CALUDE_difference_set_Q_P_l267_26745

-- Define the sets P and Q
def P : Set ℝ := {x | 1 - 2/x < 0}
def Q : Set ℝ := {x | |x - 2| < 1}

-- Define the difference set
def difference_set (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem difference_set_Q_P : 
  difference_set Q P = {x | 2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_difference_set_Q_P_l267_26745


namespace NUMINAMATH_CALUDE_square_sum_inequality_l267_26777

theorem square_sum_inequality (a b c : ℝ) : 
  a^2 + b^2 + a*b + b*c + c*a < 0 → a^2 + b^2 < c^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l267_26777


namespace NUMINAMATH_CALUDE_power_sum_equation_l267_26737

theorem power_sum_equation : 
  let x : ℚ := 1/2
  2^(0 : ℤ) + x^(-2 : ℤ) = 5 := by sorry

end NUMINAMATH_CALUDE_power_sum_equation_l267_26737


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l267_26732

theorem no_real_solution_for_log_equation :
  ¬∃ (x : ℝ), (Real.log (x + 5) + Real.log (x - 2) = Real.log (x^2 - 3*x - 10)) ∧ 
              (x + 5 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 3*x - 10 > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l267_26732


namespace NUMINAMATH_CALUDE_area_of_region_l267_26795

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 5 = 4*y - 6*x + 9

-- Theorem statement
theorem area_of_region :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    (π * radius^2 = 17 * π) :=
sorry

end NUMINAMATH_CALUDE_area_of_region_l267_26795


namespace NUMINAMATH_CALUDE_amp_four_neg_three_l267_26773

-- Define the & operation
def amp (x y : ℤ) : ℤ := x * (y + 2) + x * y

-- Theorem statement
theorem amp_four_neg_three : amp 4 (-3) = -16 := by
  sorry

end NUMINAMATH_CALUDE_amp_four_neg_three_l267_26773


namespace NUMINAMATH_CALUDE_odd_number_multiple_square_differences_l267_26746

theorem odd_number_multiple_square_differences : ∃ (n : ℕ), 
  Odd n ∧ (∃ (a b c d : ℕ), a ≠ c ∧ b ≠ d ∧ n = a^2 - b^2 ∧ n = c^2 - d^2) := by
  sorry

end NUMINAMATH_CALUDE_odd_number_multiple_square_differences_l267_26746


namespace NUMINAMATH_CALUDE_mean_median_difference_l267_26724

/-- Represents the score distribution in the math competition -/
structure ScoreDistribution where
  score72 : Float
  score84 : Float
  score86 : Float
  score92 : Float
  score98 : Float
  sum_to_one : score72 + score84 + score86 + score92 + score98 = 1

/-- Calculates the median score given the score distribution -/
def median (d : ScoreDistribution) : Float :=
  86

/-- Calculates the mean score given the score distribution -/
def mean (d : ScoreDistribution) : Float :=
  72 * d.score72 + 84 * d.score84 + 86 * d.score86 + 92 * d.score92 + 98 * d.score98

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score72 = 0.15)
  (h2 : d.score84 = 0.30)
  (h3 : d.score86 = 0.25)
  (h4 : d.score92 = 0.10) :
  mean d - median d = 0.3 := by
  sorry

#check mean_median_difference

end NUMINAMATH_CALUDE_mean_median_difference_l267_26724


namespace NUMINAMATH_CALUDE_doctors_visit_cost_l267_26733

theorem doctors_visit_cost (cast_cost insurance_coverage out_of_pocket : ℝ) :
  cast_cost = 200 →
  insurance_coverage = 0.6 →
  out_of_pocket = 200 →
  ∃ (visit_cost : ℝ),
    visit_cost = 300 ∧
    out_of_pocket = (1 - insurance_coverage) * (visit_cost + cast_cost) :=
by sorry

end NUMINAMATH_CALUDE_doctors_visit_cost_l267_26733


namespace NUMINAMATH_CALUDE_parabola_directrix_tangent_circle_l267_26780

/-- The value of p for a parabola y^2 = 2px (p > 0) with directrix tangent to the circle x^2 + y^2 + 2x = 0 -/
theorem parabola_directrix_tangent_circle (p : ℝ) : 
  p > 0 ∧ 
  (∃ x y : ℝ, y^2 = 2*p*x) ∧
  (∃ x y : ℝ, x^2 + y^2 + 2*x = 0) ∧
  (∃ x : ℝ, x = -p/2) ∧  -- directrix equation
  (∃ x y : ℝ, x^2 + y^2 + 2*x = 0 ∧ x = -p/2)  -- tangency condition
  → p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_tangent_circle_l267_26780


namespace NUMINAMATH_CALUDE_pentagon_fencing_cost_l267_26782

/-- Calculates the total cost of fencing a pentagon park -/
def fencing_cost (sides : Fin 5 → ℝ) (costs : Fin 5 → ℝ) : ℝ :=
  (sides 0 * costs 0) + (sides 1 * costs 1) + (sides 2 * costs 2) + 
  (sides 3 * costs 3) + (sides 4 * costs 4)

theorem pentagon_fencing_cost :
  let sides : Fin 5 → ℝ := ![50, 75, 60, 80, 65]
  let costs : Fin 5 → ℝ := ![2, 3, 4, 3.5, 5]
  fencing_cost sides costs = 1170 := by sorry

end NUMINAMATH_CALUDE_pentagon_fencing_cost_l267_26782


namespace NUMINAMATH_CALUDE_school_sections_l267_26711

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 288) :
  let section_size := Nat.gcd boys girls
  let boy_sections := boys / section_size
  let girl_sections := girls / section_size
  boy_sections + girl_sections = 29 := by
sorry

end NUMINAMATH_CALUDE_school_sections_l267_26711


namespace NUMINAMATH_CALUDE_expression_evaluation_l267_26778

theorem expression_evaluation :
  let f (x : ℝ) := 2 * x^2 + 3 * x - 4
  f 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l267_26778


namespace NUMINAMATH_CALUDE_circle_center_is_two_neg_three_l267_26767

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*x + y^2 + 6*y - 11 = 0

/-- The center of a circle -/
def circle_center (h k : ℝ) : ℝ × ℝ := (h, k)

/-- Theorem: The center of the circle defined by x^2 - 4x + y^2 + 6y - 11 = 0 is (2, -3) -/
theorem circle_center_is_two_neg_three :
  ∃ (r : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - 2)^2 + (y - (-3))^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_is_two_neg_three_l267_26767


namespace NUMINAMATH_CALUDE_inequality_proof_l267_26768

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (x^4)/(y*(1-y^2)) + (y^4)/(z*(1-z^2)) + (z^4)/(x*(1-x^2)) ≥ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l267_26768


namespace NUMINAMATH_CALUDE_smallest_satisfying_polygon_l267_26728

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

def satisfies_conditions (n : ℕ) : Prop :=
  (number_of_diagonals n) * 4 = n * 7 ∧
  (number_of_diagonals n + n) % 2 = 0 ∧
  number_of_diagonals n + n > 50

theorem smallest_satisfying_polygon : 
  satisfies_conditions 12 ∧ 
  ∀ m : ℕ, m < 12 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_smallest_satisfying_polygon_l267_26728


namespace NUMINAMATH_CALUDE_projection_implies_y_value_l267_26715

/-- Given two vectors v and w in R², where v = (2, y) and w = (7, 2),
    if the projection of v onto w is (8, 16/7), then y = 163/7. -/
theorem projection_implies_y_value (y : ℝ) :
  let v : ℝ × ℝ := (2, y)
  let w : ℝ × ℝ := (7, 2)
  let proj_w_v : ℝ × ℝ := ((v.1 * w.1 + v.2 * w.2) / (w.1^2 + w.2^2)) • w
  proj_w_v = (8, 16/7) →
  y = 163/7 := by
sorry

end NUMINAMATH_CALUDE_projection_implies_y_value_l267_26715


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l267_26779

theorem chinese_remainder_theorem_example :
  ∀ x : ℤ, x ≡ 9 [ZMOD 17] → x ≡ 5 [ZMOD 11] → x ≡ 60 [ZMOD 187] := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l267_26779


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_under_1000_l267_26792

theorem greatest_multiple_of_5_and_7_under_1000 : ∃ n : ℕ, n = 980 ∧ 
  (∀ m : ℕ, m < 1000 ∧ 5 ∣ m ∧ 7 ∣ m → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_under_1000_l267_26792


namespace NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l267_26708

theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let water_height := (2 : ℝ) / 3 * h
  let water_radius := (2 : ℝ) / 3 * r
  let cone_volume := (1 : ℝ) / 3 * π * r^2 * h
  let water_volume := (1 : ℝ) / 3 * π * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l267_26708


namespace NUMINAMATH_CALUDE_power_of_two_equality_l267_26757

theorem power_of_two_equality (x : ℕ) : (1 / 8 : ℚ) * (2 : ℚ)^36 = (2 : ℚ)^x → x = 33 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l267_26757


namespace NUMINAMATH_CALUDE_students_per_class_l267_26799

theorem students_per_class 
  (cards_per_student : ℕ) 
  (periods_per_day : ℕ) 
  (cards_per_pack : ℕ) 
  (cost_per_pack : ℚ) 
  (total_spent : ℚ) 
  (h1 : cards_per_student = 10) 
  (h2 : periods_per_day = 6) 
  (h3 : cards_per_pack = 50) 
  (h4 : cost_per_pack = 3) 
  (h5 : total_spent = 108) : 
  (total_spent / cost_per_pack * cards_per_pack / cards_per_student) / periods_per_day = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_per_class_l267_26799


namespace NUMINAMATH_CALUDE_quadratic_factorization_l267_26761

theorem quadratic_factorization :
  ∀ x : ℝ, 2 * x^2 + 4 * x - 6 = 2 * (x - 1) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l267_26761


namespace NUMINAMATH_CALUDE_fish_count_l267_26740

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 14

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 24 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l267_26740


namespace NUMINAMATH_CALUDE_child_b_share_after_investment_l267_26750

def total_amount : ℝ := 4500
def ratio_sum : ℕ := 2 + 3 + 4
def child_b_ratio : ℕ := 3
def interest_rate : ℝ := 0.04
def time_period : ℝ := 1

theorem child_b_share_after_investment :
  let principal := (child_b_ratio : ℝ) / ratio_sum * total_amount
  let interest := principal * interest_rate * time_period
  principal + interest = 1560 := by sorry

end NUMINAMATH_CALUDE_child_b_share_after_investment_l267_26750


namespace NUMINAMATH_CALUDE_star_equation_solution_l267_26729

-- Define the star operation
def star (a b : ℝ) : ℝ := 2 * a * b - 3 * b - a

-- State the theorem
theorem star_equation_solution :
  ∀ y : ℝ, star 4 y = 80 → y = 16.8 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l267_26729


namespace NUMINAMATH_CALUDE_max_a_value_l267_26794

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) → 
  a ≤ 1 ∧ ∀ b : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - b ≥ 0) → b ≤ a :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l267_26794


namespace NUMINAMATH_CALUDE_matthew_age_difference_l267_26764

/-- Given three children whose ages sum to 35, with Matthew 2 years older than Rebecca
    and Freddy being 15, prove that Matthew is 4 years younger than Freddy. -/
theorem matthew_age_difference (matthew rebecca freddy : ℕ) : 
  matthew + rebecca + freddy = 35 →
  matthew = rebecca + 2 →
  freddy = 15 →
  freddy - matthew = 4 := by
sorry

end NUMINAMATH_CALUDE_matthew_age_difference_l267_26764


namespace NUMINAMATH_CALUDE_min_value_expression_l267_26786

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 + a^2 * b^2) / (a * b) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l267_26786


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l267_26709

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + x - 2 ≥ 0)) ↔ (∃ x : ℝ, x^2 + x - 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l267_26709
