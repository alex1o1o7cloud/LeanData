import Mathlib

namespace NUMINAMATH_CALUDE_annie_extracurricular_hours_l2514_251419

/-- The number of hours Annie spends on extracurriculars before midterms -/
def extracurricular_hours : ℕ :=
  let chess_hours : ℕ := 2
  let drama_hours : ℕ := 8
  let glee_hours : ℕ := 3
  let weekly_hours : ℕ := chess_hours + drama_hours + glee_hours
  let semester_weeks : ℕ := 12
  let midterm_weeks : ℕ := semester_weeks / 2
  let sick_weeks : ℕ := 2
  let active_weeks : ℕ := midterm_weeks - sick_weeks
  weekly_hours * active_weeks

/-- Theorem stating that Annie spends 52 hours on extracurriculars before midterms -/
theorem annie_extracurricular_hours : extracurricular_hours = 52 := by
  sorry

end NUMINAMATH_CALUDE_annie_extracurricular_hours_l2514_251419


namespace NUMINAMATH_CALUDE_total_payment_for_bikes_l2514_251431

/-- The payment for painting a bike -/
def paint_fee : ℕ := 5

/-- The additional payment for selling a bike compared to painting it -/
def sell_bonus : ℕ := 8

/-- The number of bikes Henry sells and paints -/
def num_bikes : ℕ := 8

/-- The total payment for selling and painting one bike -/
def payment_per_bike : ℕ := paint_fee + (paint_fee + sell_bonus)

/-- Theorem stating the total payment for selling and painting 8 bikes -/
theorem total_payment_for_bikes : num_bikes * payment_per_bike = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_for_bikes_l2514_251431


namespace NUMINAMATH_CALUDE_fifteen_apples_solution_l2514_251421

/-- The number of friends sharing the apples -/
def num_friends : ℕ := 5

/-- The function representing the number of apples remaining after each friend takes their share -/
def apples_remaining (initial_apples : ℚ) (friend : ℕ) : ℚ :=
  match friend with
  | 0 => initial_apples
  | n + 1 => (apples_remaining initial_apples n / 2) - (1 / 2)

/-- The theorem stating that 15 is the correct initial number of apples -/
theorem fifteen_apples_solution :
  ∃ (initial_apples : ℚ),
    initial_apples = 15 ∧
    apples_remaining initial_apples num_friends = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_apples_solution_l2514_251421


namespace NUMINAMATH_CALUDE_min_additional_matches_for_square_grid_l2514_251405

/-- Calculates the number of matches needed for a rectangular grid -/
def matches_for_grid (rows : ℕ) (cols : ℕ) : ℕ :=
  (rows + 1) * cols + (cols + 1) * rows

/-- Represents the problem of finding the minimum additional matches needed -/
theorem min_additional_matches_for_square_grid :
  let initial_matches := matches_for_grid 3 7
  let min_square_size := (initial_matches / 4 : ℕ).sqrt.succ
  let square_matches := matches_for_grid min_square_size min_square_size
  square_matches - initial_matches = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_additional_matches_for_square_grid_l2514_251405


namespace NUMINAMATH_CALUDE_bob_cleaning_time_l2514_251481

/-- Given that Alice takes 30 minutes to clean her room and Bob takes 1/3 of Alice's time,
    prove that Bob takes 10 minutes to clean his room. -/
theorem bob_cleaning_time (alice_time bob_time : ℚ) : 
  alice_time = 30 → bob_time = (1/3) * alice_time → bob_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_bob_cleaning_time_l2514_251481


namespace NUMINAMATH_CALUDE_goldfish_cost_graph_is_finite_set_of_points_l2514_251461

def goldfish_cost (n : ℕ) : ℚ := 20 * n + 10

def valid_purchase (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10

theorem goldfish_cost_graph_is_finite_set_of_points :
  ∃ (S : Set (ℕ × ℚ)),
    Finite S ∧
    (∀ p ∈ S, valid_purchase p.1 ∧ p.2 = goldfish_cost p.1) ∧
    (∀ n, valid_purchase n → (n, goldfish_cost n) ∈ S) :=
  sorry

end NUMINAMATH_CALUDE_goldfish_cost_graph_is_finite_set_of_points_l2514_251461


namespace NUMINAMATH_CALUDE_binomial_10_3_l2514_251430

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l2514_251430


namespace NUMINAMATH_CALUDE_same_prime_factors_implies_power_of_two_l2514_251449

theorem same_prime_factors_implies_power_of_two (b m n : ℕ) 
  (hb : b ≠ 1) (hmn : m ≠ n) 
  (h_same_factors : ∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) :
  ∃ k : ℕ, b + 1 = 2^k := by
sorry

end NUMINAMATH_CALUDE_same_prime_factors_implies_power_of_two_l2514_251449


namespace NUMINAMATH_CALUDE_triangle_area_10_24_26_l2514_251406

/-- The area of a triangle with side lengths 10, 24, and 26 is 120 -/
theorem triangle_area_10_24_26 : 
  ∀ (a b c area : ℝ), 
    a = 10 → b = 24 → c = 26 →
    (a * a + b * b = c * c) →  -- Pythagorean theorem condition
    area = (1/2) * a * b →
    area = 120 := by sorry

end NUMINAMATH_CALUDE_triangle_area_10_24_26_l2514_251406


namespace NUMINAMATH_CALUDE_complex_power_eight_l2514_251448

theorem complex_power_eight : (2 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^8 = -128 - 128 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_eight_l2514_251448


namespace NUMINAMATH_CALUDE_complex_quadrant_l2514_251440

theorem complex_quadrant (z : ℂ) (h : (1 - I) * z = 3 + 5*I) : 
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l2514_251440


namespace NUMINAMATH_CALUDE_dogwood_trees_after_planting_l2514_251439

/-- The number of dogwood trees in the park after a week of planting -/
def total_trees (initial : ℕ) (monday tuesday wednesday thursday friday saturday sunday : ℕ) : ℕ :=
  initial + monday + tuesday + wednesday + thursday + friday + saturday + sunday

/-- Theorem stating the total number of dogwood trees after the week's planting -/
theorem dogwood_trees_after_planting :
  total_trees 7 3 2 5 1 6 4 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_after_planting_l2514_251439


namespace NUMINAMATH_CALUDE_range_of_m_l2514_251462

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x - floor x}

-- Define set B
def B (m : ℝ) : Set ℝ := {y | 0 ≤ y ∧ y ≤ m}

-- State the theorem
theorem range_of_m (m : ℝ) :
  (A ⊂ B m) ↔ m ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2514_251462


namespace NUMINAMATH_CALUDE_angle_sum_l2514_251471

theorem angle_sum (a b : Real) (h1 : 0 < a ∧ a < π/2) (h2 : 0 < b ∧ b < π/2)
  (h3 : 4 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 2)
  (h4 : 4 * Real.cos (2 * a) + 3 * Real.cos (2 * b) = 1) :
  a + b = π/2 := by sorry

end NUMINAMATH_CALUDE_angle_sum_l2514_251471


namespace NUMINAMATH_CALUDE_sequence_a_general_term_l2514_251492

/-- Sequence a_n with sum S_n satisfying the given conditions -/
def sequence_a (n : ℕ) : ℚ := sorry

/-- Sum of first n terms of sequence a_n -/
def S (n : ℕ) : ℚ := sorry

/-- The main theorem to prove -/
theorem sequence_a_general_term :
  ∀ n : ℕ, n > 0 →
  (2 * S n - n * sequence_a n = n) ∧
  (sequence_a 2 = 3) →
  sequence_a n = 2 * n - 1 := by sorry

end NUMINAMATH_CALUDE_sequence_a_general_term_l2514_251492


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2514_251402

/-- Prove that in a group of 8 persons, if the average weight increases by 2.5 kg
    when a new person weighing 90 kg replaces one of them,
    then the weight of the replaced person is 70 kg. -/
theorem weight_of_replaced_person
  (original_group_size : ℕ)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : original_group_size = 8)
  (h2 : weight_increase = 2.5)
  (h3 : new_person_weight = 90)
  : ℝ :=
by
  sorry

#check weight_of_replaced_person

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2514_251402


namespace NUMINAMATH_CALUDE_line_equation_correct_l2514_251459

/-- A line in 2D space --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def point_on_line (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a vector is parallel to a line --/
def vector_parallel_to_line (v : Vector2D) (l : Line2D) : Prop :=
  l.a * v.x + l.b * v.y = 0

/-- The line we're considering --/
def line_l : Line2D :=
  { a := 1, b := 2, c := -1 }

/-- The point A --/
def point_A : Point2D :=
  { x := 1, y := 0 }

/-- The direction vector of line l --/
def direction_vector : Vector2D :=
  { x := 2, y := -1 }

theorem line_equation_correct :
  point_on_line line_l point_A ∧
  vector_parallel_to_line direction_vector line_l :=
sorry

end NUMINAMATH_CALUDE_line_equation_correct_l2514_251459


namespace NUMINAMATH_CALUDE_babysitter_hours_l2514_251455

/-- The number of hours Milly hires the babysitter -/
def hours : ℕ := sorry

/-- The hourly rate of the current babysitter -/
def current_rate : ℕ := 16

/-- The hourly rate of the new babysitter -/
def new_rate : ℕ := 12

/-- The extra charge per scream for the new babysitter -/
def scream_charge : ℕ := 3

/-- The number of times the kids scream per babysitting gig -/
def scream_count : ℕ := 2

/-- The amount saved by switching to the new babysitter -/
def savings : ℕ := 18

theorem babysitter_hours : 
  current_rate * hours = new_rate * hours + scream_charge * scream_count + savings :=
by sorry

end NUMINAMATH_CALUDE_babysitter_hours_l2514_251455


namespace NUMINAMATH_CALUDE_square_prism_properties_l2514_251411

/-- A right prism with a square base -/
structure SquarePrism where
  base_side : ℝ
  height : ℝ

/-- The lateral surface area of a square prism -/
def lateral_surface_area (p : SquarePrism) : ℝ := 4 * p.base_side * p.height

/-- The total surface area of a square prism -/
def surface_area (p : SquarePrism) : ℝ := 2 * p.base_side^2 + lateral_surface_area p

/-- The volume of a square prism -/
def volume (p : SquarePrism) : ℝ := p.base_side^2 * p.height

/-- Theorem about the surface area and volume of a specific square prism -/
theorem square_prism_properties :
  ∃ (p : SquarePrism), 
    lateral_surface_area p = 6^2 ∧ 
    surface_area p = 40.5 ∧ 
    volume p = 3.375 := by
  sorry


end NUMINAMATH_CALUDE_square_prism_properties_l2514_251411


namespace NUMINAMATH_CALUDE_quadratic_function_and_triangle_area_l2514_251468

def QuadraticFunction (a b c : ℝ) := fun (x : ℝ) ↦ a * x^2 + b * x + c

theorem quadratic_function_and_triangle_area 
  (a b c : ℝ) 
  (h_opens_upward : a > 0)
  (h_not_origin : QuadraticFunction a b c 0 ≠ 0)
  (h_vertex : QuadraticFunction a b c 1 = -2)
  (x₁ x₂ : ℝ) 
  (h_roots : QuadraticFunction a b c x₁ = 0 ∧ QuadraticFunction a b c x₂ = 0)
  (h_y_intercept : (QuadraticFunction a b c 0)^2 = |x₁ * x₂|) :
  ((a = 1 ∧ b = -2 ∧ c = -1 ∧ (x₁ - x₂)^2 / 4 = 2) ∨
   (a = 1 + Real.sqrt 2 ∧ b = -(2 + 2 * Real.sqrt 2) ∧ c = Real.sqrt 2 - 1 ∧
    (x₁ - x₂)^2 / 4 = 2 * (Real.sqrt 2 - 1))) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_and_triangle_area_l2514_251468


namespace NUMINAMATH_CALUDE_c_value_theorem_l2514_251473

theorem c_value_theorem : ∃ c : ℝ, 
  (∀ x : ℝ, x * (3 * x + 1) < c ↔ -7/3 < x ∧ x < 2) ∧ c = 14 := by
  sorry

end NUMINAMATH_CALUDE_c_value_theorem_l2514_251473


namespace NUMINAMATH_CALUDE_no_arithmetic_mean_l2514_251427

theorem no_arithmetic_mean (f1 f2 f3 : ℚ) : 
  f1 = 5/8 ∧ f2 = 3/4 ∧ f3 = 9/12 →
  (f1 ≠ (f2 + f3) / 2) ∧ (f2 ≠ (f1 + f3) / 2) ∧ (f3 ≠ (f1 + f2) / 2) :=
by sorry

#check no_arithmetic_mean

end NUMINAMATH_CALUDE_no_arithmetic_mean_l2514_251427


namespace NUMINAMATH_CALUDE_unique_right_triangle_area_twice_perimeter_l2514_251454

/-- A right triangle with integer leg lengths -/
structure RightTriangle where
  a : ℕ  -- First leg
  b : ℕ  -- Second leg
  c : ℕ  -- Hypotenuse
  hyp : c^2 = a^2 + b^2  -- Pythagorean theorem

/-- The area of a right triangle is equal to twice its perimeter -/
def areaEqualsTwicePerimeter (t : RightTriangle) : Prop :=
  (t.a * t.b : ℕ) = 4 * (t.a + t.b + t.c)

/-- There exists exactly one right triangle with integer leg lengths
    where the area is equal to twice the perimeter -/
theorem unique_right_triangle_area_twice_perimeter :
  ∃! t : RightTriangle, areaEqualsTwicePerimeter t := by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_area_twice_perimeter_l2514_251454


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l2514_251482

theorem shaded_area_fraction (a r : ℝ) (h1 : a = 1/4) (h2 : r = 1/16) :
  let S := a / (1 - r)
  S = 4/15 := by sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l2514_251482


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_B_subset_A_range_l2514_251410

-- Define set A
def A : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - a^2 - 1) < 0}

-- Part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B (Real.sqrt 2)) = {x | 1 ≤ x ∧ x ≤ Real.sqrt 2 ∨ 3 ≤ x ∧ x ≤ 4} := by sorry

-- Part 2
theorem B_subset_A_range :
  ∀ a : ℝ, B a ⊆ A → 1 ≤ a ∧ a ≤ Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_B_subset_A_range_l2514_251410


namespace NUMINAMATH_CALUDE_supermarket_difference_l2514_251469

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 41

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 60

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

/-- There are more supermarkets in the US than in Canada -/
axiom more_in_us : us_supermarkets > canada_supermarkets

theorem supermarket_difference : us_supermarkets - canada_supermarkets = 22 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_difference_l2514_251469


namespace NUMINAMATH_CALUDE_wall_width_proof_l2514_251425

theorem wall_width_proof (wall_height : ℝ) (painting_width : ℝ) (painting_height : ℝ) (painting_percentage : ℝ) :
  wall_height = 5 →
  painting_width = 2 →
  painting_height = 4 →
  painting_percentage = 0.16 →
  ∃ (wall_width : ℝ), 
    wall_width = 10 ∧
    painting_width * painting_height = painting_percentage * (wall_height * wall_width) :=
by
  sorry

end NUMINAMATH_CALUDE_wall_width_proof_l2514_251425


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l2514_251491

theorem student_multiplication_problem (x : ℝ) (y : ℝ) 
  (h1 : x = 129)
  (h2 : x * y - 148 = 110) : 
  y = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l2514_251491


namespace NUMINAMATH_CALUDE_compound_interest_rate_calculation_l2514_251400

/-- Compound interest rate calculation -/
theorem compound_interest_rate_calculation
  (initial_investment : ℝ)
  (investment_duration : ℕ)
  (final_amount : ℝ)
  (h1 : initial_investment = 6500)
  (h2 : investment_duration = 2)
  (h3 : final_amount = 7372.46)
  (h4 : final_amount = initial_investment * (1 + interest_rate) ^ investment_duration) :
  ∃ (interest_rate : ℝ), 0.0664 - 0.0001 < interest_rate ∧ interest_rate < 0.0664 + 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_rate_calculation_l2514_251400


namespace NUMINAMATH_CALUDE_set_relations_imply_a_and_m_ranges_l2514_251429

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 1 = 0}

-- State the theorem
theorem set_relations_imply_a_and_m_ranges :
  ∀ a m : ℝ,
  (A ∪ B a = A) →
  (A ∩ C m = C m) →
  ((a = 2 ∨ a = 3) ∧ (-2 < m ∧ m ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_set_relations_imply_a_and_m_ranges_l2514_251429


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l2514_251422

/-- The value of k for which a line through (3, -12) and (k, 24) is parallel to 4x - 6y = 18 -/
theorem parallel_line_k_value : 
  ∃ k : ℝ, 
    (∀ x y : ℝ, (y - (-12)) / (x - 3) = (24 - (-12)) / (k - 3) → 
      (y - (-12)) / (x - 3) = 2 / 3) → 
    k = 57 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l2514_251422


namespace NUMINAMATH_CALUDE_digit_count_8_pow_12_times_5_pow_18_l2514_251460

theorem digit_count_8_pow_12_times_5_pow_18 : 
  (Nat.log 10 (8^12 * 5^18) + 1 : ℕ) = 24 := by sorry

end NUMINAMATH_CALUDE_digit_count_8_pow_12_times_5_pow_18_l2514_251460


namespace NUMINAMATH_CALUDE_operational_not_basic_l2514_251483

-- Define an enumeration of algorithm structures
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Operational
  | Loop

-- Define a function to determine if a structure is basic
def isBasicStructure : AlgorithmStructure → Bool
  | AlgorithmStructure.Sequential => true
  | AlgorithmStructure.Conditional => true
  | AlgorithmStructure.Loop => true
  | AlgorithmStructure.Operational => false

-- Theorem: Operational is the only non-basic structure among the given options
theorem operational_not_basic :
  ∀ s : AlgorithmStructure, ¬(isBasicStructure s) ↔ s = AlgorithmStructure.Operational :=
by sorry

#check operational_not_basic

end NUMINAMATH_CALUDE_operational_not_basic_l2514_251483


namespace NUMINAMATH_CALUDE_natural_number_equation_solutions_l2514_251414

theorem natural_number_equation_solutions (a b : ℕ) :
  a * (a + 5) = b * (b + 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_natural_number_equation_solutions_l2514_251414


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l2514_251456

theorem nested_subtraction_simplification : 2 - (-2 - 2) - (-2 - (-2 - 2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l2514_251456


namespace NUMINAMATH_CALUDE_sum_of_fractions_integer_l2514_251458

theorem sum_of_fractions_integer (a b : ℤ) :
  (a ≠ 0 ∧ b ≠ 0) →
  (∃ k : ℤ, (a : ℚ) / b + (b : ℚ) / a = k) ↔ (a = b ∨ a = -b) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_integer_l2514_251458


namespace NUMINAMATH_CALUDE_roots_sum_product_l2514_251486

theorem roots_sum_product (p q r : ℂ) : 
  (2 * p ^ 3 - 5 * p ^ 2 + 7 * p - 3 = 0) →
  (2 * q ^ 3 - 5 * q ^ 2 + 7 * q - 3 = 0) →
  (2 * r ^ 3 - 5 * r ^ 2 + 7 * r - 3 = 0) →
  p * q + q * r + r * p = 7 / 2 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_product_l2514_251486


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l2514_251472

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_power_dividing_factorial :
  let n := 2006
  ∃ k : ℕ, k = 34 ∧
    (∀ m : ℕ, n^m ∣ factorial n → m ≤ k) ∧
    n^k ∣ factorial n ∧
    n = 2 * 17 * 59 :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l2514_251472


namespace NUMINAMATH_CALUDE_sprained_vs_normal_time_difference_l2514_251403

/-- The time it takes Ann to frost a cake normally, in minutes -/
def normal_time : ℕ := 5

/-- The time it takes Ann to frost a cake with a sprained wrist, in minutes -/
def sprained_time : ℕ := 8

/-- The number of cakes Ann needs to frost -/
def num_cakes : ℕ := 10

/-- Theorem stating the difference in time to frost 10 cakes between sprained and normal conditions -/
theorem sprained_vs_normal_time_difference : 
  sprained_time * num_cakes - normal_time * num_cakes = 30 := by
  sorry

end NUMINAMATH_CALUDE_sprained_vs_normal_time_difference_l2514_251403


namespace NUMINAMATH_CALUDE_i_power_sum_i_power_sum_proof_l2514_251475

theorem i_power_sum : Complex → Prop :=
  fun i => i * i = -1 → i^20 + i^35 = 1 - i

-- The proof would go here, but we're skipping it as requested
theorem i_power_sum_proof : i_power_sum Complex.I :=
  sorry

end NUMINAMATH_CALUDE_i_power_sum_i_power_sum_proof_l2514_251475


namespace NUMINAMATH_CALUDE_yellow_score_mixture_l2514_251457

theorem yellow_score_mixture (white_ratio black_ratio total_yellow : ℕ) 
  (h1 : white_ratio = 7)
  (h2 : black_ratio = 6)
  (h3 : total_yellow = 78) :
  (2 : ℚ) / 3 * (white_ratio - black_ratio) / (white_ratio + black_ratio) * total_yellow = 4 := by
  sorry

end NUMINAMATH_CALUDE_yellow_score_mixture_l2514_251457


namespace NUMINAMATH_CALUDE_equation_solution_l2514_251445

theorem equation_solution : 
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0 :=
by
  -- The unique value of x that satisfies the equation for all y is 3/2
  use 3/2
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2514_251445


namespace NUMINAMATH_CALUDE_cheolsu_weight_l2514_251412

/-- Proves that Cheolsu's weight is 36 kg given the problem conditions -/
theorem cheolsu_weight (c m f : ℝ) 
  (h1 : (c + m + f) / 3 = m)  -- average equals mother's weight
  (h2 : c = (2/3) * m)        -- Cheolsu's weight is 2/3 of mother's
  (h3 : f = 72)               -- Father's weight is 72 kg
  : c = 36 := by
  sorry

#check cheolsu_weight

end NUMINAMATH_CALUDE_cheolsu_weight_l2514_251412


namespace NUMINAMATH_CALUDE_min_decimal_digits_fraction_l2514_251416

theorem min_decimal_digits_fraction (n : ℕ) (d : ℕ) (h : n = 987654321 ∧ d = 2^30 * 5^5) :
  (∃ k : ℕ, k = 30 ∧ 
    ∀ m : ℕ, m < k → ∃ r : ℚ, r ≠ 0 ∧ (n : ℚ) / d * 10^m - ((n : ℚ) / d * 10^m).floor ≠ 0) ∧
    (∃ q : ℚ, (n : ℚ) / d = q ∧ (q * 10^30).floor / 10^30 = q) :=
sorry

end NUMINAMATH_CALUDE_min_decimal_digits_fraction_l2514_251416


namespace NUMINAMATH_CALUDE_perpendicular_lines_and_planes_l2514_251446

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line_line : Line → Line → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Define the relation of a line lying within a plane
variable (line_in_plane : Line → Plane → Prop)

-- Given conditions
variable (l m : Line) (α β : Plane)
variable (h1 : perp_line_plane l α)
variable (h2 : line_in_plane m α)

-- Theorem to prove
theorem perpendicular_lines_and_planes :
  (perp_line_line l m → line_in_plane m β) ∧
  (perp_plane_plane α β → perp_line_plane m β) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_and_planes_l2514_251446


namespace NUMINAMATH_CALUDE_opposite_numbers_theorem_l2514_251433

theorem opposite_numbers_theorem (a : ℚ) : (4 * a + 9) + (3 * a + 5) = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_theorem_l2514_251433


namespace NUMINAMATH_CALUDE_range_of_a_l2514_251478

theorem range_of_a (a : ℝ) : Real.sqrt ((1 - 2*a)^2) = 2*a - 1 → a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2514_251478


namespace NUMINAMATH_CALUDE_investment_ratio_l2514_251485

/-- Prove that the investment ratio between A and C is 3:1 --/
theorem investment_ratio (a b c : ℕ) (total_profit c_profit : ℕ) : 
  a = 3 * b → -- A and B invested in a ratio of 3:1
  total_profit = 60000 → -- The total profit was 60000
  c_profit = 20000 → -- C received 20000 from the profit
  3 * c = a := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l2514_251485


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2514_251489

/-- An arithmetic sequence with the given properties has a common difference of 1/3. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)  -- The arithmetic sequence
  (h1 : a 3 + a 5 = 2)  -- First condition
  (h2 : a 7 + a 10 + a 13 = 9)  -- Second condition
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  : ∃ d : ℚ, d = 1/3 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2514_251489


namespace NUMINAMATH_CALUDE_equation_solution_l2514_251490

theorem equation_solution (x : ℝ) :
  (x / 5) / 3 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2514_251490


namespace NUMINAMATH_CALUDE_babysitting_earnings_l2514_251479

/-- Represents the babysitting rates based on child's age --/
def BabysittingRate : ℕ → ℕ
  | age => if age < 2 then 5 else if age ≤ 5 then 7 else 8

/-- Calculates the total earnings from babysitting --/
def TotalEarnings (childrenAges : List ℕ) (hours : List ℕ) : ℕ :=
  List.sum (List.zipWith (λ age hour => BabysittingRate age * hour) childrenAges hours)

theorem babysitting_earnings :
  let janeStartAge : ℕ := 18
  let childA : ℕ := janeStartAge / 2
  let childB : ℕ := childA - 2
  let childC : ℕ := childB + 3
  let childD : ℕ := childC
  let childrenAges : List ℕ := [childA, childB, childC, childD]
  let hours : List ℕ := [50, 90, 130, 70]
  TotalEarnings childrenAges hours = 2720 := by
  sorry


end NUMINAMATH_CALUDE_babysitting_earnings_l2514_251479


namespace NUMINAMATH_CALUDE_distinct_values_count_l2514_251407

def original_expression : ℕ → ℕ := λ n => 3^(3^(3^3))

def parenthesization1 : ℕ → ℕ := λ n => 3^((3^3)^3)
def parenthesization2 : ℕ → ℕ := λ n => 3^(3^(3^3 + 1))
def parenthesization3 : ℕ → ℕ := λ n => (3^(3^3))^3

theorem distinct_values_count :
  ∃ (S : Finset ℕ),
    S.card = 4 ∧
    (∀ x ∈ S, x ≠ original_expression 0) ∧
    (∀ x ∈ S, (x = parenthesization1 0) ∨ (x = parenthesization2 0) ∨ (x = parenthesization3 0) ∨
              (∃ y, x = 3^y ∧ y ≠ 3^(3^3))) :=
by sorry

end NUMINAMATH_CALUDE_distinct_values_count_l2514_251407


namespace NUMINAMATH_CALUDE_real_solutions_quadratic_l2514_251442

theorem real_solutions_quadratic (x y : ℝ) : 
  (3 * y^2 + 6 * x * y + 2 * x + 4 = 0) → 
  (∃ y : ℝ, 3 * y^2 + 6 * x * y + 2 * x + 4 = 0) ↔ 
  (x ≤ -2/3 ∨ x ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_real_solutions_quadratic_l2514_251442


namespace NUMINAMATH_CALUDE_min_value_of_max_function_l2514_251467

theorem min_value_of_max_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  ∃ (t : ℝ), t = 4 ∧ ∀ (s : ℝ), s ≥ max (x^2) (4 / (y * (x - y))) → s ≥ t :=
sorry

end NUMINAMATH_CALUDE_min_value_of_max_function_l2514_251467


namespace NUMINAMATH_CALUDE_profit_without_discount_is_fifty_percent_l2514_251450

/-- Represents the profit percentage and discount percentage as ratios -/
structure ProfitDiscount where
  profit : ℚ
  discount : ℚ

/-- Calculates the profit percentage without discount given the profit percentage with discount -/
def profit_without_discount (pd : ProfitDiscount) : ℚ :=
  (1 + pd.profit) / (1 - pd.discount) - 1

/-- Theorem stating that a 42.5% profit with a 5% discount results in a 50% profit without discount -/
theorem profit_without_discount_is_fifty_percent :
  let pd : ProfitDiscount := { profit := 425/1000, discount := 5/100 }
  profit_without_discount pd = 1/2 := by
sorry

end NUMINAMATH_CALUDE_profit_without_discount_is_fifty_percent_l2514_251450


namespace NUMINAMATH_CALUDE_apple_picking_problem_l2514_251441

theorem apple_picking_problem (x : ℝ) : 
  x + (3/4) * x + 600 = 2600 → x = 1142 := by
  sorry

end NUMINAMATH_CALUDE_apple_picking_problem_l2514_251441


namespace NUMINAMATH_CALUDE_function_f_negative_two_l2514_251477

/-- A function satisfying the given properties -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, f (a + b) = f a * f b) ∧
  (∀ x : ℝ, f x > 0) ∧
  (f 1 = 1/3)

/-- The main theorem -/
theorem function_f_negative_two (f : ℝ → ℝ) (h : FunctionF f) : f (-2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_function_f_negative_two_l2514_251477


namespace NUMINAMATH_CALUDE_hardware_contract_probability_l2514_251487

theorem hardware_contract_probability 
  (p_not_software : ℝ) 
  (p_at_least_one : ℝ) 
  (p_both : ℝ) 
  (h1 : p_not_software = 3/5) 
  (h2 : p_at_least_one = 9/10) 
  (h3 : p_both = 0.3) : 
  ∃ p_hardware : ℝ, p_hardware = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_hardware_contract_probability_l2514_251487


namespace NUMINAMATH_CALUDE_P_on_x_axis_P_distance_to_y_axis_l2514_251432

-- Define the point P as a function of a
def P (a : ℝ) : ℝ × ℝ := (2 * a - 1, a + 3)

-- Condition 1: P lies on the x-axis
theorem P_on_x_axis (a : ℝ) :
  P a = (-7, 0) ↔ (P a).2 = 0 :=
sorry

-- Condition 2: Distance from P to y-axis is 5
theorem P_distance_to_y_axis (a : ℝ) :
  (abs (P a).1 = 5) ↔ (P a = (-5, 1) ∨ P a = (5, 6)) :=
sorry

end NUMINAMATH_CALUDE_P_on_x_axis_P_distance_to_y_axis_l2514_251432


namespace NUMINAMATH_CALUDE_poes_speed_l2514_251480

theorem poes_speed (teena_speed : ℝ) (initial_distance : ℝ) (final_distance : ℝ) (time : ℝ) :
  teena_speed = 55 →
  initial_distance = 7.5 →
  final_distance = 15 →
  time = 1.5 →
  ∃ (poe_speed : ℝ), 
    poe_speed = 40 ∧
    teena_speed * time - poe_speed * time = initial_distance + final_distance :=
by
  sorry

end NUMINAMATH_CALUDE_poes_speed_l2514_251480


namespace NUMINAMATH_CALUDE_ratio_lcm_problem_l2514_251466

theorem ratio_lcm_problem (a b x : ℕ+) (h_ratio : a.val * x.val = 8 * b.val) 
  (h_lcm : Nat.lcm a.val b.val = 432) (h_a : a = 48) : b = 72 := by
  sorry

end NUMINAMATH_CALUDE_ratio_lcm_problem_l2514_251466


namespace NUMINAMATH_CALUDE_green_or_purple_probability_l2514_251409

/-- The probability of drawing a green or purple marble from a bag -/
theorem green_or_purple_probability 
  (green : ℕ) (purple : ℕ) (white : ℕ) 
  (h_green : green = 4) 
  (h_purple : purple = 3) 
  (h_white : white = 6) : 
  (green + purple : ℚ) / (green + purple + white) = 7 / 13 := by
  sorry

end NUMINAMATH_CALUDE_green_or_purple_probability_l2514_251409


namespace NUMINAMATH_CALUDE_problem_statement_l2514_251423

noncomputable section

def f (x : ℝ) := Real.exp x * Real.sin x - Real.cos x
def g (x : ℝ) := x * Real.cos x - Real.sqrt 2 * Real.exp x

theorem problem_statement :
  (∀ m : ℝ, (∀ x₁ ∈ Set.Icc 0 (Real.pi / 2), ∃ x₂ ∈ Set.Icc 0 (Real.pi / 2), f x₁ + g x₂ ≥ m) → m ≤ -1 - Real.sqrt 2) ∧
  (∀ x > -1, f x - g x > 0) :=
by sorry

end

end NUMINAMATH_CALUDE_problem_statement_l2514_251423


namespace NUMINAMATH_CALUDE_xz_length_l2514_251434

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- ∠X = 90°
  t.X = 90

def has_hypotenuse_10 (t : Triangle) : Prop :=
  -- YZ = 10
  t.Y = 10

def satisfies_trig_relation (t : Triangle) : Prop :=
  -- tan Z = 3 sin Z
  Real.tan t.Z = 3 * Real.sin t.Z

-- Theorem statement
theorem xz_length (t : Triangle) 
  (h1 : is_right_triangle t) 
  (h2 : has_hypotenuse_10 t) 
  (h3 : satisfies_trig_relation t) : 
  -- XZ = 10/3
  t.Z = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_xz_length_l2514_251434


namespace NUMINAMATH_CALUDE_pythagorean_sum_number_with_conditions_l2514_251413

def is_pythagorean_sum_number (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  c^2 + d^2 = 10 * a + b

def G (n : ℕ) : ℚ :=
  let c := (n / 10) % 10
  let d := n % 10
  (c + d : ℚ) / 9

def P (n : ℕ) : ℚ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (10 * a - 2 * c * d + b : ℚ) / 3

theorem pythagorean_sum_number_with_conditions :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
    is_pythagorean_sum_number n ∧
    (∃ k : ℤ, G n = k) ∧
    P n = 3 →
  n = 3772 ∨ n = 3727 :=
sorry

end NUMINAMATH_CALUDE_pythagorean_sum_number_with_conditions_l2514_251413


namespace NUMINAMATH_CALUDE_eggs_to_buy_l2514_251426

def total_eggs_needed : ℕ := 222
def eggs_received : ℕ := 155

theorem eggs_to_buy : total_eggs_needed - eggs_received = 67 := by
  sorry

end NUMINAMATH_CALUDE_eggs_to_buy_l2514_251426


namespace NUMINAMATH_CALUDE_ice_pop_price_is_correct_l2514_251474

/-- The selling price of an ice-pop that allows a school to buy pencils, given:
  * The cost to make each ice-pop
  * The cost of each pencil
  * The number of ice-pops that need to be sold
  * The number of pencils to be bought
-/
def ice_pop_selling_price (make_cost : ℚ) (pencil_cost : ℚ) (pops_sold : ℕ) (pencils_bought : ℕ) : ℚ :=
  make_cost + (pencil_cost * pencils_bought - make_cost * pops_sold) / pops_sold

/-- Theorem stating that the selling price of each ice-pop is $1.20 under the given conditions -/
theorem ice_pop_price_is_correct :
  ice_pop_selling_price 0.90 1.80 300 100 = 1.20 := by
  sorry

end NUMINAMATH_CALUDE_ice_pop_price_is_correct_l2514_251474


namespace NUMINAMATH_CALUDE_average_score_theorem_l2514_251436

/-- The average score of a class given the proportions of students scoring different points -/
theorem average_score_theorem (p3 p2 p1 p0 : ℝ) 
  (h_p3 : p3 = 0.3) 
  (h_p2 : p2 = 0.5) 
  (h_p1 : p1 = 0.1) 
  (h_p0 : p0 = 0.1)
  (h_sum : p3 + p2 + p1 + p0 = 1) : 
  3 * p3 + 2 * p2 + 1 * p1 + 0 * p0 = 2 := by
  sorry

#check average_score_theorem

end NUMINAMATH_CALUDE_average_score_theorem_l2514_251436


namespace NUMINAMATH_CALUDE_impossible_arrangement_l2514_251476

/-- A grid of integers -/
def Grid := Matrix (Fin 25) (Fin 41) ℤ

/-- Predicate to check if a grid satisfies the adjacency condition -/
def SatisfiesAdjacencyCondition (g : Grid) : Prop :=
  ∀ i j i' j', (i = i' ∧ |j - j'| = 1) ∨ (j = j' ∧ |i - i'| = 1) →
    |g i j - g i' j'| ≤ 16

/-- Predicate to check if a grid contains distinct integers -/
def ContainsDistinctIntegers (g : Grid) : Prop :=
  ∀ i j i' j', g i j = g i' j' → i = i' ∧ j = j'

/-- Theorem stating the impossibility of the arrangement -/
theorem impossible_arrangement : 
  ¬∃ (g : Grid), SatisfiesAdjacencyCondition g ∧ ContainsDistinctIntegers g :=
sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l2514_251476


namespace NUMINAMATH_CALUDE_grid_property_l2514_251417

/-- Represents a 3x3 grid -/
structure Grid :=
  (cells : Matrix (Fin 3) (Fin 3) ℤ)

/-- Represents an operation on the grid -/
inductive Operation
  | add_adjacent : Fin 3 → Fin 3 → Fin 3 → Fin 3 → Operation
  | subtract_adjacent : Fin 3 → Fin 3 → Fin 3 → Fin 3 → Operation

/-- Applies an operation to a grid -/
def apply_operation (g : Grid) (op : Operation) : Grid :=
  sorry

/-- The sum of all cells in a grid -/
def grid_sum (g : Grid) : ℤ :=
  sorry

/-- The difference between shaded and non-shaded cells -/
def shaded_difference (g : Grid) (shaded : Set (Fin 3 × Fin 3)) : ℤ :=
  sorry

/-- Theorem stating the property of the grid after operations -/
theorem grid_property (initial : Grid) (final : Grid) (ops : List Operation) 
    (shaded : Set (Fin 3 × Fin 3)) :
  (∀ op ∈ ops, grid_sum (apply_operation initial op) = grid_sum initial) →
  (∀ op ∈ ops, shaded_difference (apply_operation initial op) shaded = shaded_difference initial shaded) →
  (∃ A : ℤ, final.cells 0 0 = A ∧ 4 * 2010 + A - 4 * 2010 = 5) →
  final.cells 0 0 = 5 :=
by sorry

end NUMINAMATH_CALUDE_grid_property_l2514_251417


namespace NUMINAMATH_CALUDE_original_number_proof_l2514_251498

theorem original_number_proof (x : ℝ) : 
  (x * 1.375 - x * 0.575 = 85) → x = 106.25 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2514_251498


namespace NUMINAMATH_CALUDE_fred_savings_period_l2514_251428

/-- The number of weeks Fred needs to save to buy the mountain bike -/
def weeks_to_save (bike_cost : ℕ) (birthday_money : ℕ) (weekly_earnings : ℕ) : ℕ :=
  (bike_cost - birthday_money) / weekly_earnings

theorem fred_savings_period :
  weeks_to_save 600 150 18 = 25 := by
  sorry

end NUMINAMATH_CALUDE_fred_savings_period_l2514_251428


namespace NUMINAMATH_CALUDE_patients_ages_problem_l2514_251424

theorem patients_ages_problem : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x - y = 44 ∧ x * y = 1280 ∧ x = 64 ∧ y = 20 := by
  sorry

end NUMINAMATH_CALUDE_patients_ages_problem_l2514_251424


namespace NUMINAMATH_CALUDE_pie_remainder_l2514_251401

theorem pie_remainder (carlos_share : ℝ) (maria_fraction : ℝ) : 
  carlos_share = 0.6 → 
  maria_fraction = 0.5 → 
  (1 - carlos_share) * (1 - maria_fraction) = 0.2 := by
sorry

end NUMINAMATH_CALUDE_pie_remainder_l2514_251401


namespace NUMINAMATH_CALUDE_trapezoid_in_square_l2514_251488

theorem trapezoid_in_square (s : ℝ) (x : ℝ) : 
  s = 2 → -- Side length of the square
  (1/3) * s^2 = (1/2) * (s + x) * (s/2) → -- Area of trapezoid is 1/3 of square's area
  x = 2/3 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_in_square_l2514_251488


namespace NUMINAMATH_CALUDE_amy_chore_money_l2514_251494

/-- Calculates the money earned from chores given initial amount, birthday money, and final amount --/
def money_from_chores (initial_amount birthday_money final_amount : ℕ) : ℕ :=
  final_amount - initial_amount - birthday_money

/-- Theorem stating that Amy's money from chores is 13 dollars --/
theorem amy_chore_money :
  money_from_chores 2 3 18 = 13 := by
  sorry

end NUMINAMATH_CALUDE_amy_chore_money_l2514_251494


namespace NUMINAMATH_CALUDE_product_one_inequality_l2514_251464

theorem product_one_inequality (a b c d e : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) 
  (h_prod : a * b * c * d * e = 1) : 
  a^2 / b^2 + b^2 / c^2 + c^2 / d^2 + d^2 / e^2 + e^2 / a^2 ≥ a + b + c + d + e := by
sorry

end NUMINAMATH_CALUDE_product_one_inequality_l2514_251464


namespace NUMINAMATH_CALUDE_surviving_positions_32_l2514_251435

/-- Represents the selection process for an international exchange event. -/
def SelectionProcess (n : ℕ) : Prop :=
  n > 0 ∧ ∃ k, 2^k = n

/-- Represents a valid initial position in the selection process. -/
def ValidPosition (n : ℕ) (p : ℕ) : Prop :=
  1 ≤ p ∧ p ≤ n

/-- Represents a position that survives all elimination rounds. -/
def SurvivingPosition (n : ℕ) (p : ℕ) : Prop :=
  ValidPosition n p ∧ ∃ k, 2^k = p

/-- The main theorem stating that positions 16 and 32 are the only surviving positions in a 32-student selection process. -/
theorem surviving_positions_32 :
  SelectionProcess 32 →
  ∀ p, SurvivingPosition 32 p ↔ (p = 16 ∨ p = 32) :=
by sorry

end NUMINAMATH_CALUDE_surviving_positions_32_l2514_251435


namespace NUMINAMATH_CALUDE_newspaper_photos_l2514_251420

/-- The total number of photos in a newspaper -/
def total_photos (pages_with_4 : ℕ) (pages_with_6 : ℕ) : ℕ :=
  pages_with_4 * 4 + pages_with_6 * 6

/-- Theorem stating that the total number of photos is 208 -/
theorem newspaper_photos : total_photos 25 18 = 208 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_photos_l2514_251420


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l2514_251452

theorem inequality_holds_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔ 
  (a ≤ -2 ∨ a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l2514_251452


namespace NUMINAMATH_CALUDE_divided_proportion_problem_l2514_251470

theorem divided_proportion_problem (total : ℚ) (a b c : ℚ) (h1 : total = 782) 
  (h2 : a = 1/2) (h3 : b = 1/3) (h4 : c = 3/4) : 
  (total * a) / (a + b + c) = 247 := by
  sorry

end NUMINAMATH_CALUDE_divided_proportion_problem_l2514_251470


namespace NUMINAMATH_CALUDE_g_5_equals_104_l2514_251495

def g (x : ℝ) : ℝ := 3 * x^4 - 22 * x^3 + 47 * x^2 - 44 * x + 24

theorem g_5_equals_104 : g 5 = 104 := by sorry

end NUMINAMATH_CALUDE_g_5_equals_104_l2514_251495


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_equal_diagonal_regular_polygon_l2514_251438

/-- A regular polygon with all diagonals equal -/
structure EqualDiagonalRegularPolygon where
  /-- The number of sides of the polygon -/
  sides : ℕ
  /-- The polygon is regular -/
  regular : True
  /-- All diagonals of the polygon are equal -/
  equal_diagonals : True
  /-- The polygon has at least 3 sides -/
  sides_ge_three : sides ≥ 3

/-- The sum of interior angles of a polygon -/
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The sum of interior angles of a regular polygon with all diagonals equal
    is either 360° or 540° -/
theorem sum_of_interior_angles_equal_diagonal_regular_polygon
  (p : EqualDiagonalRegularPolygon) :
  sum_of_interior_angles p.sides = 360 ∨ sum_of_interior_angles p.sides = 540 :=
sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_equal_diagonal_regular_polygon_l2514_251438


namespace NUMINAMATH_CALUDE_log_equation_solution_l2514_251493

-- Define the equation
def log_equation (x : ℝ) : Prop :=
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 4 = 15

-- State the theorem
theorem log_equation_solution :
  ∀ x : ℝ, x > 0 → log_equation x → x = 2^(90/11) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2514_251493


namespace NUMINAMATH_CALUDE_baker_remaining_pastries_l2514_251404

/-- The number of pastries Baker made -/
def pastries_made : ℕ := 148

/-- The number of pastries Baker sold -/
def pastries_sold : ℕ := 103

/-- The number of pastries Baker still has -/
def pastries_remaining : ℕ := pastries_made - pastries_sold

theorem baker_remaining_pastries : pastries_remaining = 45 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_pastries_l2514_251404


namespace NUMINAMATH_CALUDE_solve_sock_problem_l2514_251451

def sock_problem (lisa_initial : ℕ) (sandra : ℕ) (total : ℕ) : Prop :=
  let cousin := sandra / 5
  let before_mom := lisa_initial + sandra + cousin
  ∃ (mom : ℕ), before_mom + mom = total

theorem solve_sock_problem :
  sock_problem 12 20 80 → ∃ (mom : ℕ), mom = 44 := by
  sorry

end NUMINAMATH_CALUDE_solve_sock_problem_l2514_251451


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2514_251465

theorem divisibility_theorem (n : ℕ) (h : ∃ m : ℕ, 2^n - 2 = n * m) :
  ∃ k : ℕ, 2^(2^n - 1) - 2 = (2^n - 1) * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2514_251465


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2514_251408

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  a 3 = 3 →
  a 6 = 1 / 9 →
  a 4 * a 5 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2514_251408


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l2514_251444

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- State the theorem
theorem solution_set_implies_sum (a b : ℝ) :
  (∀ x, 1 < x ∧ x < b ↔ f a x < 0) →
  a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l2514_251444


namespace NUMINAMATH_CALUDE_greatest_value_of_a_l2514_251437

noncomputable def f (a : ℝ) : ℝ :=
  (5 * Real.sqrt ((2 * a) ^ 2 + 1) - 4 * a ^ 2 - 2 * a) / (Real.sqrt (1 + 4 * a ^ 2) + 5)

theorem greatest_value_of_a :
  ∃ (a_max : ℝ), a_max = Real.sqrt 6 ∧
  f a_max = 1 ∧
  ∀ (a : ℝ), f a = 1 → a ≤ a_max :=
sorry

end NUMINAMATH_CALUDE_greatest_value_of_a_l2514_251437


namespace NUMINAMATH_CALUDE_jacks_paycheck_l2514_251447

theorem jacks_paycheck (paycheck : ℝ) : 
  (0.2 * (0.8 * paycheck) = 20) → paycheck = 125 := by
  sorry

end NUMINAMATH_CALUDE_jacks_paycheck_l2514_251447


namespace NUMINAMATH_CALUDE_highest_power_of_three_in_M_l2514_251453

def M : ℕ := sorry  -- Definition of M as concatenation of 2-digit integers from 10 to 81

theorem highest_power_of_three_in_M : 
  ∃ (k : ℕ), (3^2 ∣ M) ∧ ¬(3^(2+1) ∣ M) :=
sorry

end NUMINAMATH_CALUDE_highest_power_of_three_in_M_l2514_251453


namespace NUMINAMATH_CALUDE_sanda_exercise_days_l2514_251484

theorem sanda_exercise_days 
  (javier_daily_minutes : ℕ) 
  (javier_days : ℕ) 
  (sanda_daily_minutes : ℕ) 
  (total_minutes : ℕ) :
  javier_daily_minutes = 50 →
  javier_days = 7 →
  sanda_daily_minutes = 90 →
  total_minutes = 620 →
  (javier_daily_minutes * javier_days + sanda_daily_minutes * (total_minutes - javier_daily_minutes * javier_days) / sanda_daily_minutes = total_minutes) →
  (total_minutes - javier_daily_minutes * javier_days) / sanda_daily_minutes = 3 :=
by sorry

end NUMINAMATH_CALUDE_sanda_exercise_days_l2514_251484


namespace NUMINAMATH_CALUDE_triangle_shape_l2514_251415

theorem triangle_shape (A B C : Real) (a b c : Real) :
  (A + B + C = Real.pi) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a * Real.cos A = b * Real.cos B) →
  (A = B ∨ A + B = Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l2514_251415


namespace NUMINAMATH_CALUDE_horse_and_saddle_cost_l2514_251463

/-- The total cost of a horse and saddle, given their relative costs -/
def total_cost (saddle_cost : ℕ) (horse_cost_multiplier : ℕ) : ℕ :=
  saddle_cost + horse_cost_multiplier * saddle_cost

/-- Theorem: The total cost of a horse and saddle is $5000 -/
theorem horse_and_saddle_cost :
  total_cost 1000 4 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_horse_and_saddle_cost_l2514_251463


namespace NUMINAMATH_CALUDE_job_completion_theorem_l2514_251499

/-- The number of days it takes the initial group of machines to finish the job -/
def initial_days : ℕ := 40

/-- The number of additional machines added -/
def additional_machines : ℕ := 4

/-- The number of days it takes after adding more machines -/
def reduced_days : ℕ := 30

/-- The number of machines initially working on the job -/
def initial_machines : ℕ := 16

theorem job_completion_theorem :
  (initial_machines : ℚ) / initial_days = (initial_machines + additional_machines : ℚ) / reduced_days :=
by sorry

#check job_completion_theorem

end NUMINAMATH_CALUDE_job_completion_theorem_l2514_251499


namespace NUMINAMATH_CALUDE_f_increasing_and_range_l2514_251443

noncomputable def f (x : ℝ) : ℝ := 1 - 2 / (2^x + 1)

theorem f_increasing_and_range :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (Set.range (fun x => f x) ∩ Set.Icc 0 1 = Set.Icc 0 (1/3)) := by sorry

end NUMINAMATH_CALUDE_f_increasing_and_range_l2514_251443


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2514_251418

theorem complex_fraction_simplification (z : ℂ) :
  z = 1 - I → 2 / z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2514_251418


namespace NUMINAMATH_CALUDE_equation_solution_l2514_251497

theorem equation_solution : 
  ∃ (x : ℝ), (4 * x - 5) / (5 * x - 10) = 3 / 4 ∧ x = -10 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2514_251497


namespace NUMINAMATH_CALUDE_apples_eaten_by_dog_l2514_251496

theorem apples_eaten_by_dog (apples_on_tree : ℕ) (apples_on_ground : ℕ) (apples_remaining : ℕ) : 
  apples_on_tree = 5 → apples_on_ground = 8 → apples_remaining = 10 →
  apples_on_tree + apples_on_ground - apples_remaining = 3 := by
sorry

end NUMINAMATH_CALUDE_apples_eaten_by_dog_l2514_251496
