import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_inequality_l74_7422

theorem absolute_value_inequality (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) → |a| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l74_7422


namespace NUMINAMATH_CALUDE_percent_relation_l74_7423

theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.14 * a) 
  (h2 : b = 0.35 * a) : 
  c = 0.4 * b := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l74_7423


namespace NUMINAMATH_CALUDE_smallest_possible_value_l74_7496

theorem smallest_possible_value (m n x : ℕ+) : 
  m = 60 →
  Nat.gcd m.val n.val = x.val + 5 →
  Nat.lcm m.val n.val = 2 * x.val * (x.val + 5) →
  (∀ n' : ℕ+, n'.val < n.val → 
    (Nat.gcd m.val n'.val ≠ x.val + 5 ∨ 
     Nat.lcm m.val n'.val ≠ 2 * x.val * (x.val + 5))) →
  n.val = 75 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_value_l74_7496


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l74_7414

theorem complex_magnitude_product : Complex.abs ((12 - 9*Complex.I) * (8 + 15*Complex.I)) = 255 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l74_7414


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l74_7498

theorem modulus_of_complex_number : 
  let z : ℂ := Complex.I * (1 + 2 * Complex.I)
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l74_7498


namespace NUMINAMATH_CALUDE_may_has_greatest_difference_l74_7429

/-- Represents the sales data for a single month --/
structure MonthSales where
  drummers : ℕ
  bugle : ℕ
  flute : ℕ

/-- Calculates the percentage difference between the highest and lowest sales --/
def percentageDifference (sales : MonthSales) : ℚ :=
  let max := max sales.drummers (max sales.bugle sales.flute)
  let min := min sales.drummers (min sales.bugle sales.flute)
  (max - min : ℚ) / min * 100

/-- The sales data for each month --/
def salesData : List MonthSales := [
  ⟨5, 4, 6⟩,  -- January
  ⟨6, 4, 5⟩,  -- February
  ⟨5, 5, 5⟩,  -- March
  ⟨4, 6, 4⟩,  -- April
  ⟨3, 4, 7⟩   -- May
]

/-- Theorem stating that May has the greatest percentage difference --/
theorem may_has_greatest_difference : 
  ∀ (i : Fin 5), i.val ≠ 4 → 
    percentageDifference (salesData[4]) > percentageDifference (salesData[i.val]) := by
  sorry


end NUMINAMATH_CALUDE_may_has_greatest_difference_l74_7429


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l74_7439

theorem quadratic_root_condition (d : ℚ) : 
  (∀ x : ℚ, 2 * x^2 + 11 * x + d = 0 ↔ x = (-11 + Real.sqrt 15) / 4 ∨ x = (-11 - Real.sqrt 15) / 4) → 
  d = 53 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l74_7439


namespace NUMINAMATH_CALUDE_S_value_l74_7465

/-- The sum Sₙ for n points on a line and a point off the line -/
def S (n : ℕ) (l : Set (ℝ × ℝ)) (Q : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem stating the value of Sₙ based on n -/
theorem S_value (n : ℕ) (l : Set (ℝ × ℝ)) (Q : ℝ × ℝ) 
  (h1 : n ≥ 3)
  (h2 : ∃ (p : Fin n → ℝ × ℝ), (∀ i, p i ∈ l) ∧ (∀ i j, i ≠ j → p i ≠ p j))
  (h3 : Q ∉ l) :
  S n l Q = if n = 3 then 1 else 0 :=
sorry

end NUMINAMATH_CALUDE_S_value_l74_7465


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l74_7485

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x

-- State the theorem
theorem f_satisfies_conditions :
  -- Condition 1: The domain is ℝ (implicitly satisfied by the definition)
  -- Condition 2: For any a, b ∈ ℝ where a + b = 0, f(a) + f(b) = 0
  (∀ a b : ℝ, a + b = 0 → f a + f b = 0) ∧
  -- Condition 3: For any x ∈ ℝ, if m < 0, then f(x) > f(x + m)
  (∀ x m : ℝ, m < 0 → f x > f (x + m)) :=
by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l74_7485


namespace NUMINAMATH_CALUDE_equivalent_statements_l74_7421

theorem equivalent_statements (A B : Prop) :
  ((A ∧ B) → ¬(A ∨ B)) ↔ ((A ∨ B) → ¬(A ∧ B)) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_statements_l74_7421


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l74_7452

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 28 x = Nat.choose 28 (3 * x - 8)) → (x = 4 ∨ x = 9) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l74_7452


namespace NUMINAMATH_CALUDE_expression_simplification_l74_7438

theorem expression_simplification (x y : ℝ) :
  3*x + 4*y + 5*x^2 + 2 - (8 - 5*x - 3*y - 2*x^2) = 7*x^2 + 8*x + 7*y - 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l74_7438


namespace NUMINAMATH_CALUDE_two_numbers_difference_l74_7487

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (prod_eq : x * y = 162) : 
  |x - y| = 6 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l74_7487


namespace NUMINAMATH_CALUDE_circle_alignment_exists_l74_7426

-- Define the circle type
structure Circle where
  circumference : ℝ
  marked_points : ℕ
  arc_length : ℝ

-- Define the theorem
theorem circle_alignment_exists (c1 c2 : Circle)
  (h1 : c1.circumference = 100)
  (h2 : c2.circumference = 100)
  (h3 : c1.marked_points = 100)
  (h4 : c2.arc_length < 1) :
  ∃ (alignment : ℝ), ∀ (point : ℕ) (arc : ℝ),
    point < c1.marked_points →
    arc < c2.arc_length →
    (point : ℝ) * c1.circumference / c1.marked_points + alignment ≠ arc :=
sorry

end NUMINAMATH_CALUDE_circle_alignment_exists_l74_7426


namespace NUMINAMATH_CALUDE_circuit_disconnection_possibilities_l74_7473

theorem circuit_disconnection_possibilities :
  let n : ℕ := 7  -- number of resistors
  let total_possibilities : ℕ := 2^n - 1  -- total number of ways at least one resistor can be disconnected
  total_possibilities = 63 := by
  sorry

end NUMINAMATH_CALUDE_circuit_disconnection_possibilities_l74_7473


namespace NUMINAMATH_CALUDE_smallest_divisible_by_2022_l74_7493

theorem smallest_divisible_by_2022 : 
  ∀ n : ℕ, n > 1 ∧ n < 79 → ¬(2022 ∣ (n^7 - 1)) ∧ (2022 ∣ (79^7 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_2022_l74_7493


namespace NUMINAMATH_CALUDE_bacteria_count_l74_7409

theorem bacteria_count (original : ℕ) (increase : ℕ) (current : ℕ) : 
  original = 600 → increase = 8317 → current = original + increase → current = 8917 := by
sorry

end NUMINAMATH_CALUDE_bacteria_count_l74_7409


namespace NUMINAMATH_CALUDE_polynomial_modulus_bound_l74_7468

theorem polynomial_modulus_bound (a b c d : ℂ) 
  (ha : Complex.abs a = 1) (hb : Complex.abs b = 1) 
  (hc : Complex.abs c = 1) (hd : Complex.abs d = 1) : 
  ∃ z : ℂ, Complex.abs z = 1 ∧ 
    Complex.abs (a * z^3 + b * z^2 + c * z + d) ≥ Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_modulus_bound_l74_7468


namespace NUMINAMATH_CALUDE_identity_function_proof_l74_7450

theorem identity_function_proof (f : ℝ → ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 0 1) →
  (∀ x ∈ Set.Icc 0 1, f (2 * x - f x) = x) →
  (∀ x ∈ Set.Icc 0 1, f x = x) := by
sorry

end NUMINAMATH_CALUDE_identity_function_proof_l74_7450


namespace NUMINAMATH_CALUDE_roots_sum_and_product_inequality_l74_7481

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

theorem roots_sum_and_product_inequality 
  (x₁ x₂ : ℝ) 
  (h_pos₁ : x₁ > 0) 
  (h_pos₂ : x₂ > 0) 
  (h_distinct : x₁ ≠ x₂) 
  (h_root₁ : f x₁ = 3 * Real.exp 1 * x₁ + 3 * Real.exp 1 * Real.log x₁)
  (h_root₂ : f x₂ = 3 * Real.exp 1 * x₂ + 3 * Real.exp 1 * Real.log x₂) :
  x₁ + x₂ + Real.log (x₁ * x₂) > 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_inequality_l74_7481


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l74_7469

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := λ x => -3 * x^2 + 2 * x + 8
  {x : ℝ | f x > 0} = Set.Ioo (-4/3 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l74_7469


namespace NUMINAMATH_CALUDE_expression_evaluation_l74_7460

theorem expression_evaluation :
  ∃ m : ℤ, (3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2 = m * 10^1003 ∧ m = 1372 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l74_7460


namespace NUMINAMATH_CALUDE_gcd_problem_l74_7405

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = (2 * k + 1) * 8723) :
  Int.gcd (8 * b^2 + 55 * b + 144) (4 * b + 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l74_7405


namespace NUMINAMATH_CALUDE_inscribed_cube_properties_l74_7445

/-- Given a cube with surface area 54 square meters, containing an inscribed sphere 
    which in turn contains an inscribed smaller cube, prove the surface area and volume 
    of the inner cube. -/
theorem inscribed_cube_properties (outer_cube : Real) (sphere : Real) (inner_cube : Real) :
  (6 * outer_cube ^ 2 = 54) →
  (sphere = outer_cube / 2) →
  (inner_cube * Real.sqrt 3 = outer_cube) →
  (6 * inner_cube ^ 2 = 18 ∧ inner_cube ^ 3 = 3 * Real.sqrt 3) := by
  sorry

#check inscribed_cube_properties

end NUMINAMATH_CALUDE_inscribed_cube_properties_l74_7445


namespace NUMINAMATH_CALUDE_intersection_point_theorem_l74_7408

/-- A point is an intersection point of a line and a hyperbola if it satisfies both equations -/
def is_intersection_point (n : ℕ+) : Prop :=
  let x : ℝ := n
  let y : ℝ := n^2
  y = n * x ∧ y = n^3 / x

/-- For any positive integer n, the point (n, n²) is an intersection point of y=nx and y=n³/x -/
theorem intersection_point_theorem (n : ℕ+) : is_intersection_point n := by
  sorry

#check intersection_point_theorem

end NUMINAMATH_CALUDE_intersection_point_theorem_l74_7408


namespace NUMINAMATH_CALUDE_right_prism_cut_count_l74_7425

theorem right_prism_cut_count : 
  let b : ℕ := 2023
  let count := (Finset.filter 
    (fun p : ℕ × ℕ => 
      let (a, c) := p
      a ≤ b ∧ b ≤ c ∧ a * c = b * b ∧ a < c)
    (Finset.product (Finset.range (b + 1)) (Finset.range (b * b + 1)))).card
  count = 13 := by
sorry

end NUMINAMATH_CALUDE_right_prism_cut_count_l74_7425


namespace NUMINAMATH_CALUDE_amount_per_painting_l74_7484

/-- Hallie's art earnings -/
def total_earnings : ℕ := 300

/-- Number of paintings sold -/
def paintings_sold : ℕ := 3

/-- Theorem: The amount earned per painting is $100 -/
theorem amount_per_painting :
  total_earnings / paintings_sold = 100 := by sorry

end NUMINAMATH_CALUDE_amount_per_painting_l74_7484


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_exists_l74_7488

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- Side lengths of the quadrilateral -/
  sides : Fin 4 → ℕ+
  /-- Diagonal lengths of the quadrilateral -/
  diagonals : Fin 2 → ℕ+
  /-- Area of the quadrilateral -/
  area : ℕ+
  /-- Radius of the circumcircle -/
  radius : ℕ+
  /-- The quadrilateral is inscribed in a circle -/
  inscribed : True
  /-- The side lengths are pairwise distinct -/
  distinct_sides : ∀ i j, i ≠ j → sides i ≠ sides j

/-- There exists an inscribed quadrilateral with integer parameters -/
theorem inscribed_quadrilateral_exists : 
  ∃ q : InscribedQuadrilateral, True :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_exists_l74_7488


namespace NUMINAMATH_CALUDE_sports_club_members_l74_7411

/-- A sports club with members who play badminton, tennis, both, or neither -/
structure SportsClub where
  badminton : ℕ
  tennis : ℕ
  both : ℕ
  neither : ℕ

/-- The total number of members in the sports club -/
def total_members (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - club.both + club.neither

/-- Theorem stating that the total number of members in the given sports club is 30 -/
theorem sports_club_members :
  ∃ (club : SportsClub),
    club.badminton = 17 ∧
    club.tennis = 21 ∧
    club.both = 10 ∧
    club.neither = 2 ∧
    total_members club = 30 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l74_7411


namespace NUMINAMATH_CALUDE_water_left_proof_l74_7449

-- Define the initial amount of water
def initial_water : ℚ := 7/2

-- Define the amount of water used
def water_used : ℚ := 9/4

-- Theorem statement
theorem water_left_proof : initial_water - water_used = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_water_left_proof_l74_7449


namespace NUMINAMATH_CALUDE_sandwich_combinations_l74_7479

def num_toppings : ℕ := 10
def num_patty_types : ℕ := 3

theorem sandwich_combinations :
  (2^num_toppings) * num_patty_types = 3072 := by sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l74_7479


namespace NUMINAMATH_CALUDE_river_current_speed_l74_7419

/-- The speed of a river's current given a swimmer's performance -/
theorem river_current_speed 
  (swimmer_still_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : swimmer_still_speed = 3)
  (h2 : distance = 8)
  (h3 : time = 5) :
  swimmer_still_speed - (distance / time) = (1.4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_river_current_speed_l74_7419


namespace NUMINAMATH_CALUDE_proposition_truth_l74_7463

theorem proposition_truth : 
  (∀ x > 0, Real.log (x + 1) > 0) ∧ 
  ¬(∀ a b : ℝ, a > b → a^2 > b^2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_truth_l74_7463


namespace NUMINAMATH_CALUDE_product_units_digit_base_7_l74_7446

theorem product_units_digit_base_7 : 
  (359 * 72) % 7 = 4 := by sorry

end NUMINAMATH_CALUDE_product_units_digit_base_7_l74_7446


namespace NUMINAMATH_CALUDE_circle_intersection_problem_l74_7453

/-- Given two circles C₁ and C₂ with equations as defined below, prove that the value of a is 4 -/
theorem circle_intersection_problem (a b x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 + y₁^2 - 2*x₁ + 4*y₁ - b^2 + 5 = 0) →  -- C₁ equation for point A
  (x₂^2 + y₂^2 - 2*x₂ + 4*y₂ - b^2 + 5 = 0) →  -- C₁ equation for point B
  (x₁^2 + y₁^2 - 2*(a-6)*x₁ - 2*a*y₁ + 2*a^2 - 12*a + 27 = 0) →  -- C₂ equation for point A
  (x₂^2 + y₂^2 - 2*(a-6)*x₂ - 2*a*y₂ + 2*a^2 - 12*a + 27 = 0) →  -- C₂ equation for point B
  ((y₁ + y₂)/(x₁ + x₂) + (x₁ - x₂)/(y₁ - y₂) = 0) →  -- Given condition
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →  -- Distinct points condition
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_problem_l74_7453


namespace NUMINAMATH_CALUDE_line_y_intercept_l74_7499

/-- A line with slope -3 and x-intercept (4, 0) has y-intercept (0, 12) -/
theorem line_y_intercept (line : ℝ → ℝ) (slope : ℝ) (x_intercept : ℝ × ℝ) :
  slope = -3 →
  x_intercept = (4, 0) →
  (∀ x, line x = slope * x + line 0) →
  line 4 = 0 →
  line 0 = 12 := by
sorry

end NUMINAMATH_CALUDE_line_y_intercept_l74_7499


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l74_7428

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 5 = 82 →
  a 2 * a 4 = 81 →
  a 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l74_7428


namespace NUMINAMATH_CALUDE_molecular_weight_CuCO3_l74_7455

/-- The atomic weight of Copper in g/mol -/
def Cu_weight : ℝ := 63.55

/-- The atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The molecular weight of CuCO3 in g/mol -/
def CuCO3_weight : ℝ := Cu_weight + C_weight + 3 * O_weight

/-- The number of moles of CuCO3 -/
def moles : ℝ := 8

/-- Theorem: The molecular weight of 8 moles of CuCO3 is 988.48 grams -/
theorem molecular_weight_CuCO3 : moles * CuCO3_weight = 988.48 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_CuCO3_l74_7455


namespace NUMINAMATH_CALUDE_lcm_of_10_and_21_l74_7454

theorem lcm_of_10_and_21 : Nat.lcm 10 21 = 210 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_10_and_21_l74_7454


namespace NUMINAMATH_CALUDE_cost_price_calculation_l74_7458

/-- Proves that the cost price of an article is 1200, given that it was sold at a 40% profit for 1680. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
    (h1 : selling_price = 1680)
    (h2 : profit_percentage = 40) : 
  selling_price / (1 + profit_percentage / 100) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l74_7458


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l74_7430

/-- Given a rectangular metallic sheet with length 48 m and width w,
    if squares of 8 m are cut from each corner to form an open box with volume 5120 m³,
    then the width w of the original sheet is 36 m. -/
theorem metallic_sheet_width (w : ℝ) : 
  w > 0 →  -- Ensuring positive width
  (48 - 2 * 8) * (w - 2 * 8) * 8 = 5120 →  -- Volume equation
  w = 36 := by
sorry


end NUMINAMATH_CALUDE_metallic_sheet_width_l74_7430


namespace NUMINAMATH_CALUDE_y_completion_time_l74_7464

/-- Represents the time it takes for worker y to complete the job alone -/
def y_time : ℝ := 12

/-- Represents the time it takes for workers x and y to complete the job together -/
def xy_time : ℝ := 20

/-- Represents the number of days x worked alone before y joined -/
def x_solo_days : ℝ := 4

/-- Represents the total number of days the job took to complete -/
def total_days : ℝ := 10

/-- Represents the portion of work completed in one day -/
def work_unit : ℝ := 1

theorem y_completion_time :
  (x_solo_days * (work_unit / xy_time)) +
  ((total_days - x_solo_days) * (work_unit / xy_time + work_unit / y_time)) = work_unit :=
sorry

end NUMINAMATH_CALUDE_y_completion_time_l74_7464


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l74_7444

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_with_current = 21) 
  (h2 : current_speed = 4.3) : 
  speed_with_current - 2 * current_speed = 12.4 := by
  sorry

#check mans_speed_against_current

end NUMINAMATH_CALUDE_mans_speed_against_current_l74_7444


namespace NUMINAMATH_CALUDE_circle_tangent_range_l74_7447

/-- Given a circle with equation x^2 + y^2 + ax + 2y + a^2 = 0 and a fixed point A(1, 2),
    this theorem states the range of values for a that allows two tangents from point A to the circle. -/
theorem circle_tangent_range (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + a*x + 2*y + a^2 = 0) →
  (∃ (t : ℝ), (1 + t*(-a/2 - 1))^2 + (2 + t*(-1 - (-1)))^2 = ((4 - 3*a^2)/4)) →
  a ∈ Set.Ioo (-2*Real.sqrt 3/3) (2*Real.sqrt 3/3) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_range_l74_7447


namespace NUMINAMATH_CALUDE_parallel_resistors_l74_7404

theorem parallel_resistors (x y r : ℝ) (hx : x = 3) (hy : y = 5) 
  (hr : 1 / r = 1 / x + 1 / y) : r = 1.875 := by
  sorry

end NUMINAMATH_CALUDE_parallel_resistors_l74_7404


namespace NUMINAMATH_CALUDE_sum_of_powers_of_three_and_negative_three_l74_7486

theorem sum_of_powers_of_three_and_negative_three : 
  (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_three_and_negative_three_l74_7486


namespace NUMINAMATH_CALUDE_inequality_proof_l74_7471

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) * (a + c) ≥ 2 * Real.sqrt (a * b * c * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l74_7471


namespace NUMINAMATH_CALUDE_no_such_function_l74_7475

theorem no_such_function : ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, x > 0 → y > 0 → f (x + y) > f x * (1 + y * f x) := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_l74_7475


namespace NUMINAMATH_CALUDE_nonzero_real_solution_l74_7431

theorem nonzero_real_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x + 1 / y = 12) (eq2 : y + 1 / x = 7 / 15) :
  x = 6 + 3 * Real.sqrt (8 / 7) ∨ x = 6 - 3 * Real.sqrt (8 / 7) :=
by sorry

end NUMINAMATH_CALUDE_nonzero_real_solution_l74_7431


namespace NUMINAMATH_CALUDE_gcd_problem_l74_7441

theorem gcd_problem (b : ℤ) (h : 570 ∣ b) : Int.gcd (4*b^3 + b^2 + 5*b + 95) b = 95 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l74_7441


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l74_7482

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) : 
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 := by
  sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) : 
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l74_7482


namespace NUMINAMATH_CALUDE_knights_arrangement_exists_l74_7406

/-- Represents a knight in King Arthur's court -/
structure Knight where
  id : ℕ

/-- Represents the relationship between knights -/
inductive Relationship
  | Friend
  | Enemy

/-- Represents the seating arrangement of knights around a round table -/
def Arrangement := List Knight

/-- Function to determine if two knights are enemies -/
def areEnemies (k1 k2 : Knight) : Prop := sorry

/-- Function to count the number of enemies a knight has -/
def enemyCount (k : Knight) (knights : List Knight) : ℕ := sorry

/-- Function to check if an arrangement is valid (no adjacent enemies) -/
def isValidArrangement (arr : Arrangement) : Prop := sorry

/-- Main theorem: There exists a valid arrangement of knights -/
theorem knights_arrangement_exists (n : ℕ) (knights : List Knight) :
  knights.length = 2 * n →
  (∀ k ∈ knights, enemyCount k knights ≤ n - 1) →
  ∃ arr : Arrangement, arr.length = 2 * n ∧ isValidArrangement arr :=
sorry

end NUMINAMATH_CALUDE_knights_arrangement_exists_l74_7406


namespace NUMINAMATH_CALUDE_quadratic_roots_l74_7440

theorem quadratic_roots (b c : ℝ) (h1 : 3 ∈ {x : ℝ | x^2 + b*x + c = 0}) 
  (h2 : 5 ∈ {x : ℝ | x^2 + b*x + c = 0}) :
  {y : ℝ | (y^2 + 4)^2 + b*(y^2 + 4) + c = 0} = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l74_7440


namespace NUMINAMATH_CALUDE_passes_through_first_and_third_quadrants_l74_7436

def proportional_function (x : ℝ) : ℝ := x

theorem passes_through_first_and_third_quadrants :
  (∀ x : ℝ, x > 0 → proportional_function x > 0) ∧
  (∀ x : ℝ, x < 0 → proportional_function x < 0) :=
by sorry

end NUMINAMATH_CALUDE_passes_through_first_and_third_quadrants_l74_7436


namespace NUMINAMATH_CALUDE_second_train_speed_l74_7443

/-- Proves that the speed of the second train is 80 kmph given the conditions of the problem -/
theorem second_train_speed (first_train_speed : ℝ) (time_difference : ℝ) (meeting_distance : ℝ) :
  first_train_speed = 40 →
  time_difference = 1 →
  meeting_distance = 80 →
  (meeting_distance - first_train_speed * time_difference) / time_difference = 80 :=
by
  sorry

#check second_train_speed

end NUMINAMATH_CALUDE_second_train_speed_l74_7443


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l74_7420

/-- A "T" shaped figure composed of unit squares -/
structure TShape :=
  (top_row : Fin 3 → Unit)
  (bottom_column : Fin 2 → Unit)

/-- The perimeter of a TShape -/
def perimeter (t : TShape) : ℕ :=
  14

theorem t_shape_perimeter :
  ∀ (t : TShape), perimeter t = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l74_7420


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l74_7416

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (non_overlapping : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line
  (m : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : perpendicular m β)
  (h3 : non_overlapping α β) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l74_7416


namespace NUMINAMATH_CALUDE_cubic_system_solution_method_l74_7457

/-- A cubic polynomial -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + c * x + d

/-- The statement of the theorem -/
theorem cubic_system_solution_method
  (a b c d : ℝ) (p : ℝ → ℝ) (hp : p = CubicPolynomial a b c d) :
  ∃ (cubic : ℝ → ℝ) (quadratic : ℝ → ℝ),
    (∀ x y : ℝ, x = p y ∧ y = p x ↔ 
      (cubic x = 0 ∧ quadratic y = 0) ∨ 
      (cubic y = 0 ∧ quadratic x = 0)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_system_solution_method_l74_7457


namespace NUMINAMATH_CALUDE_vector_sum_and_scale_l74_7400

theorem vector_sum_and_scale :
  let v1 : Fin 2 → ℝ := ![5, -3]
  let v2 : Fin 2 → ℝ := ![-4, 9]
  v1 + 2 • v2 = ![-3, 15] := by
sorry

end NUMINAMATH_CALUDE_vector_sum_and_scale_l74_7400


namespace NUMINAMATH_CALUDE_square_digit_sum_100_bound_l74_7407

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem square_digit_sum_100_bound (n : ℕ) :
  sum_of_digits (n^2) = 100 → n ≤ 100 := by sorry

end NUMINAMATH_CALUDE_square_digit_sum_100_bound_l74_7407


namespace NUMINAMATH_CALUDE_first_day_over_100_l74_7477

def paperclips (day : ℕ) : ℕ :=
  if day = 0 then 5
  else if day = 1 then 7
  else 7 + 3 * (day - 1)

def dayOfWeek (day : ℕ) : String :=
  match day % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem first_day_over_100 :
  (∀ d : ℕ, d < 33 → paperclips d ≤ 100) ∧
  paperclips 33 > 100 ∧
  dayOfWeek 33 = "Friday" := by
  sorry

end NUMINAMATH_CALUDE_first_day_over_100_l74_7477


namespace NUMINAMATH_CALUDE_first_term_is_two_l74_7467

/-- A sequence of 5 terms where the differences between consecutive terms form an arithmetic sequence -/
def ArithmeticSequenceOfDifferences (a : Fin 5 → ℕ) : Prop :=
  ∃ d : ℕ, ∀ i : Fin 3, a (i + 1) - a i = d + i

theorem first_term_is_two (a : Fin 5 → ℕ) 
  (h1 : a 1 = 4) 
  (h2 : a 2 = 7)
  (h3 : a 3 = 11)
  (h4 : a 4 = 16)
  (h5 : ArithmeticSequenceOfDifferences a) : 
  a 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_two_l74_7467


namespace NUMINAMATH_CALUDE_complement_intersection_M_N_l74_7403

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 + 2) / (p.1 - 2) = 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ≠ p.1 - 4}

-- Theorem statement
theorem complement_intersection_M_N : 
  (U \ M) ∩ (U \ N) = {(2, -2)} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_M_N_l74_7403


namespace NUMINAMATH_CALUDE_spiders_went_loose_l74_7427

theorem spiders_went_loose (initial_birds initial_puppies initial_cats initial_spiders : ℕ)
  (birds_sold puppies_adopted animals_left : ℕ) :
  initial_birds = 12 →
  initial_puppies = 9 →
  initial_cats = 5 →
  initial_spiders = 15 →
  birds_sold = initial_birds / 2 →
  puppies_adopted = 3 →
  animals_left = 25 →
  initial_spiders - (animals_left - ((initial_birds - birds_sold) + (initial_puppies - puppies_adopted) + initial_cats)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_spiders_went_loose_l74_7427


namespace NUMINAMATH_CALUDE_expected_votes_for_candidate_a_l74_7483

theorem expected_votes_for_candidate_a (total_voters : ℕ) 
  (dem_percent : ℝ) (rep_percent : ℝ) (dem_vote_a : ℝ) (rep_vote_a : ℝ) :
  dem_percent = 0.6 →
  rep_percent = 0.4 →
  dem_vote_a = 0.75 →
  rep_vote_a = 0.3 →
  dem_percent + rep_percent = 1 →
  (dem_percent * dem_vote_a + rep_percent * rep_vote_a) * 100 = 57 := by
  sorry

end NUMINAMATH_CALUDE_expected_votes_for_candidate_a_l74_7483


namespace NUMINAMATH_CALUDE_additional_girls_needed_prove_additional_girls_l74_7402

theorem additional_girls_needed (initial_girls initial_boys : ℕ) 
  (target_ratio : ℚ) (additional_girls : ℕ) : Prop :=
  initial_girls = 2 →
  initial_boys = 6 →
  target_ratio = 5/8 →
  (initial_girls + additional_girls : ℚ) / 
    (initial_girls + initial_boys + additional_girls) = target_ratio →
  additional_girls = 8

theorem prove_additional_girls : 
  ∃ (additional_girls : ℕ), 
    additional_girls_needed 2 6 (5/8) additional_girls :=
sorry

end NUMINAMATH_CALUDE_additional_girls_needed_prove_additional_girls_l74_7402


namespace NUMINAMATH_CALUDE_parallel_lines_a_values_l74_7434

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 : ℝ} : 
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The problem statement -/
theorem parallel_lines_a_values (a : ℝ) :
  (∃ x y : ℝ, y = a * x - 2 ∧ 3 * x - (a + 2) * y + 1 = 0) →
  (∀ x y : ℝ, y = a * x - 2 ↔ 3 * x - (a + 2) * y + 1 = 0) →
  a = 1 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_values_l74_7434


namespace NUMINAMATH_CALUDE_margo_travel_distance_l74_7432

/-- The total distance traveled by Margo -/
def total_distance (bicycle_time walk_time average_rate : ℚ) : ℚ :=
  average_rate * (bicycle_time + walk_time) / 60

/-- Theorem: Given the conditions, Margo traveled 4 miles -/
theorem margo_travel_distance :
  let bicycle_time : ℚ := 15
  let walk_time : ℚ := 25
  let average_rate : ℚ := 6
  total_distance bicycle_time walk_time average_rate = 4 := by
  sorry

end NUMINAMATH_CALUDE_margo_travel_distance_l74_7432


namespace NUMINAMATH_CALUDE_six_digit_palindrome_divisibility_l74_7492

theorem six_digit_palindrome_divisibility (a b : Nat) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≤ 9) :
  let ab := 10 * a + b
  let ababab := 100000 * ab + 1000 * ab + ab
  ∃ (k1 k2 k3 k4 : Nat), ababab = 101 * k1 ∧ ababab = 7 * k2 ∧ ababab = 11 * k3 ∧ ababab = 13 * k4 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_palindrome_divisibility_l74_7492


namespace NUMINAMATH_CALUDE_nine_n_sum_of_squares_nine_n_sum_of_squares_not_div_by_three_l74_7474

theorem nine_n_sum_of_squares (n a b c : ℕ) (h : n = a^2 + b^2 + c^2) :
  ∃ (p₁ q₁ r₁ p₂ q₂ r₂ p₃ q₃ r₃ : ℤ), 
    (p₁ ≠ 0 ∧ q₁ ≠ 0 ∧ r₁ ≠ 0 ∧ p₂ ≠ 0 ∧ q₂ ≠ 0 ∧ r₂ ≠ 0 ∧ p₃ ≠ 0 ∧ q₃ ≠ 0 ∧ r₃ ≠ 0) ∧
    (9 * n = (p₁ * a + q₁ * b + r₁ * c)^2 + (p₂ * a + q₂ * b + r₂ * c)^2 + (p₃ * a + q₃ * b + r₃ * c)^2) :=
sorry

theorem nine_n_sum_of_squares_not_div_by_three (n a b c : ℕ) (h₁ : n = a^2 + b^2 + c^2) 
  (h₂ : ¬(3 ∣ a) ∨ ¬(3 ∣ b) ∨ ¬(3 ∣ c)) :
  ∃ (x y z : ℕ), 
    (¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z)) ∧
    (9 * n = x^2 + y^2 + z^2) :=
sorry

end NUMINAMATH_CALUDE_nine_n_sum_of_squares_nine_n_sum_of_squares_not_div_by_three_l74_7474


namespace NUMINAMATH_CALUDE_vector_perpendicular_to_difference_l74_7415

/-- Given vectors a = (-1, 2) and b = (1, 3), prove that a is perpendicular to (a - b) -/
theorem vector_perpendicular_to_difference (a b : ℝ × ℝ) :
  a = (-1, 2) →
  b = (1, 3) →
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_vector_perpendicular_to_difference_l74_7415


namespace NUMINAMATH_CALUDE_treasure_hunt_probability_l74_7472

def num_islands : ℕ := 7
def num_treasure_islands : ℕ := 4

def prob_treasure : ℚ := 1/3
def prob_traps : ℚ := 1/6
def prob_neither : ℚ := 1/2

theorem treasure_hunt_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  35/648 := by
  sorry

end NUMINAMATH_CALUDE_treasure_hunt_probability_l74_7472


namespace NUMINAMATH_CALUDE_largest_angle_in_special_right_triangle_l74_7461

theorem largest_angle_in_special_right_triangle :
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  ∃ (x : ℝ), a = 3*x ∧ b = 2*x →
  max a (max b c) = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_right_triangle_l74_7461


namespace NUMINAMATH_CALUDE_fabric_sale_meters_l74_7413

-- Define the price per meter in kopecks
def price_per_meter : ℕ := 436

-- Define the maximum revenue in kopecks
def max_revenue : ℕ := 50000

-- Define a predicate for valid revenue
def valid_revenue (x : ℕ) : Prop :=
  (price_per_meter * x) % 1000 = 728 ∧
  price_per_meter * x ≤ max_revenue

-- Theorem statement
theorem fabric_sale_meters :
  ∃ (x : ℕ), valid_revenue x ∧ x = 98 := by sorry

end NUMINAMATH_CALUDE_fabric_sale_meters_l74_7413


namespace NUMINAMATH_CALUDE_problem_solution_l74_7495

theorem problem_solution :
  let a : ℚ := -1/2
  let x : ℤ := 8
  let y : ℤ := 5
  (a * (a^4 - a + 1) * (a - 2) = 125/64) ∧
  ((x + 2*y) * (x - y) - (2*x - y) * (-x - y) = 87) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l74_7495


namespace NUMINAMATH_CALUDE_sugar_weight_loss_fraction_l74_7437

theorem sugar_weight_loss_fraction (green_beans_weight sugar_weight rice_weight remaining_weight : ℝ) :
  green_beans_weight = 60 →
  rice_weight = green_beans_weight - 30 →
  sugar_weight = green_beans_weight - 10 →
  remaining_weight = 120 →
  (green_beans_weight + (2/3 * rice_weight) + sugar_weight - remaining_weight) / sugar_weight = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sugar_weight_loss_fraction_l74_7437


namespace NUMINAMATH_CALUDE_coat_discount_proof_l74_7476

theorem coat_discount_proof :
  ∃ (p q : ℕ), 
    p < 10 ∧ q < 10 ∧
    21250 * (1 - p / 100) * (1 - q / 100) = 19176 ∧
    ((p = 4 ∧ q = 6) ∨ (p = 6 ∧ q = 4)) := by
  sorry

end NUMINAMATH_CALUDE_coat_discount_proof_l74_7476


namespace NUMINAMATH_CALUDE_multiple_of_reciprocal_l74_7456

theorem multiple_of_reciprocal (x : ℝ) (m : ℝ) (h1 : x > 0) (h2 : x + 17 = m * (1 / x)) (h3 : x = 3) : m = 60 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_reciprocal_l74_7456


namespace NUMINAMATH_CALUDE_factory_output_restoration_l74_7491

theorem factory_output_restoration (O : ℝ) (O_pos : O > 0) :
  let increase1 := 1.2
  let increase2 := 1.5
  let increase3 := 1.25
  let final_output := O * increase1 * increase2 * increase3
  let decrease_percent := (1 - 1 / (increase1 * increase2 * increase3)) * 100
  ∃ ε > 0, |decrease_percent - 55.56| < ε :=
by sorry

end NUMINAMATH_CALUDE_factory_output_restoration_l74_7491


namespace NUMINAMATH_CALUDE_chores_ratio_l74_7412

/-- Proves that the ratio of time spent on other chores to vacuuming is 3:1 -/
theorem chores_ratio (vacuum_time other_chores_time total_time : ℕ) : 
  vacuum_time = 3 → 
  total_time = 12 → 
  other_chores_time = total_time - vacuum_time →
  (other_chores_time : ℚ) / vacuum_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_chores_ratio_l74_7412


namespace NUMINAMATH_CALUDE_blue_candy_probability_l74_7401

def green_candies : ℕ := 5
def blue_candies : ℕ := 3
def red_candies : ℕ := 4

def total_candies : ℕ := green_candies + blue_candies + red_candies

theorem blue_candy_probability :
  (blue_candies : ℚ) / total_candies = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_blue_candy_probability_l74_7401


namespace NUMINAMATH_CALUDE_cos_2000_in_terms_of_tan_20_l74_7494

theorem cos_2000_in_terms_of_tan_20 (a : ℝ) (h : Real.tan (20 * π / 180) = a) :
  Real.cos (2000 * π / 180) = -1 / Real.sqrt (1 + a^2) := by
  sorry

end NUMINAMATH_CALUDE_cos_2000_in_terms_of_tan_20_l74_7494


namespace NUMINAMATH_CALUDE_fred_marbles_l74_7433

theorem fred_marbles (total : ℕ) (dark_blue : ℕ) (green : ℕ) (yellow : ℕ) (red : ℕ) : 
  total = 120 →
  dark_blue ≥ total / 3 →
  green = 10 →
  yellow = 5 →
  red = total - (dark_blue + green + yellow) →
  red = 65 := by
  sorry

end NUMINAMATH_CALUDE_fred_marbles_l74_7433


namespace NUMINAMATH_CALUDE_pentagon_area_relationship_l74_7448

/-- Represents the areas of different parts of a pentagon -/
structure PentagonAreas where
  x : ℝ  -- Area of the smaller similar pentagon
  y : ℝ  -- Area of one type of surrounding region
  z : ℝ  -- Area of another type of surrounding region
  total : ℝ  -- Total area of the larger pentagon

/-- Theorem about the relationship between areas in a specially divided pentagon -/
theorem pentagon_area_relationship (p : PentagonAreas) 
  (h_positive : p.x > 0 ∧ p.y > 0 ∧ p.z > 0 ∧ p.total > 0)
  (h_similar : ∃ (k : ℝ), k > 0 ∧ p.x = k^2 * p.total)
  (h_total : p.total = p.x + 5*p.y + 5*p.z) :
  p.y = p.z ∧ 
  p.y = (p.total - p.x) / 10 ∧
  p.total = p.x + 10*p.y := by
  sorry


end NUMINAMATH_CALUDE_pentagon_area_relationship_l74_7448


namespace NUMINAMATH_CALUDE_finite_decimal_is_rational_l74_7459

theorem finite_decimal_is_rational (x : ℝ) (h : ∃ (n : ℕ) (m : ℤ), x = m / (10 ^ n)) : 
  ∃ (p q : ℤ), x = p / q ∧ q ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_finite_decimal_is_rational_l74_7459


namespace NUMINAMATH_CALUDE_f_symmetry_about_y_axis_l74_7442

def f (x : ℝ) : ℝ := |x|

theorem f_symmetry_about_y_axis : ∀ x : ℝ, f x = f (-x) := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_about_y_axis_l74_7442


namespace NUMINAMATH_CALUDE_coin_game_theorem_l74_7462

/-- Represents a pile of coins -/
structure CoinPile :=
  (count : ℕ)
  (hcount : count ≥ 2015)

/-- Represents the state of the three piles -/
structure GameState :=
  (pile1 : CoinPile)
  (pile2 : CoinPile)
  (pile3 : CoinPile)

/-- The polynomial f(x) = x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + 1 -/
def f (x : ℕ) : ℕ := x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + 1

/-- Represents a valid operation on the piles -/
inductive Operation
  | SplitEven (i : Fin 3) : Operation
  | RemoveOdd (i : Fin 3) : Operation

/-- Applies an operation to a game state -/
def applyOperation (state : GameState) (op : Operation) : GameState :=
  sorry

/-- Checks if a game state has reached the goal -/
def hasReachedGoal (state : GameState) : Prop :=
  ∃ (i : Fin 3), state.pile1.count ≥ 2017^2017 ∨ 
                 state.pile2.count ≥ 2017^2017 ∨ 
                 state.pile3.count ≥ 2017^2017

/-- The main theorem to prove -/
theorem coin_game_theorem (a b c : ℕ) 
  (ha : a ≥ 2015) (hb : b ≥ 2015) (hc : c ≥ 2015) :
  (∃ (ops : List Operation), 
    hasReachedGoal (ops.foldl applyOperation 
      { pile1 := ⟨a, ha⟩, pile2 := ⟨b, hb⟩, pile3 := ⟨c, hc⟩ })) ↔ 
  (f 2 = 2017 ∧ f 1 = 7) :=
sorry

end NUMINAMATH_CALUDE_coin_game_theorem_l74_7462


namespace NUMINAMATH_CALUDE_cubic_taylor_coefficient_l74_7470

theorem cubic_taylor_coefficient (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_taylor_coefficient_l74_7470


namespace NUMINAMATH_CALUDE_type_b_machine_time_l74_7417

def job_completion_time (machine_q : ℝ) (machine_b : ℝ) (combined_time : ℝ) : Prop :=
  2 / machine_q + 3 / machine_b = 1 / combined_time

theorem type_b_machine_time : 
  ∀ (machine_b : ℝ),
    job_completion_time 5 machine_b 1.2 →
    machine_b = 90 / 13 := by
  sorry

end NUMINAMATH_CALUDE_type_b_machine_time_l74_7417


namespace NUMINAMATH_CALUDE_min_value_expression_l74_7451

theorem min_value_expression (a b : ℤ) (h : a > b) :
  (2 : ℝ) ≤ ((2*a + 3*b : ℝ) / (a - 2*b : ℝ)) + ((a - 2*b : ℝ) / (2*a + 3*b : ℝ)) ∧
  ∃ (a' b' : ℤ), a' > b' ∧ ((2*a' + 3*b' : ℝ) / (a' - 2*b' : ℝ)) + ((a' - 2*b' : ℝ) / (2*a' + 3*b' : ℝ)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l74_7451


namespace NUMINAMATH_CALUDE_juans_number_l74_7489

theorem juans_number (j k : ℕ) (h1 : j > 0) (h2 : k > 0) 
  (h3 : 10^(k+1) + 10*j + 1 - j = 14789) : j = 532 := by
  sorry

end NUMINAMATH_CALUDE_juans_number_l74_7489


namespace NUMINAMATH_CALUDE_cube_diagonal_l74_7466

theorem cube_diagonal (V A : ℝ) (h1 : V = 384) (h2 : A = 384) :
  ∃ (s d : ℝ), s^3 = V ∧ 6 * s^2 = A ∧ d = s * Real.sqrt 3 ∧ d = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_diagonal_l74_7466


namespace NUMINAMATH_CALUDE_min_value_theorem_l74_7410

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  3 * x + y ≥ 1 + 8 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    3 * x₀ + y₀ = 1 + 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l74_7410


namespace NUMINAMATH_CALUDE_lucky_number_in_13_consecutive_l74_7418

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A number is lucky if the sum of its digits is divisible by 7 -/
def isLucky (n : ℕ) : Prop := sumOfDigits n % 7 = 0

/-- Any sequence of 13 consecutive numbers contains at least one lucky number -/
theorem lucky_number_in_13_consecutive (n : ℕ) : 
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 12 ∧ isLucky k := by sorry

end NUMINAMATH_CALUDE_lucky_number_in_13_consecutive_l74_7418


namespace NUMINAMATH_CALUDE_simplified_expression_l74_7480

theorem simplified_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  (108 * (Real.sqrt 10 + Real.sqrt 14 - Real.sqrt 6 - Real.sqrt 490)) / (-59) :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_l74_7480


namespace NUMINAMATH_CALUDE_unicorn_rope_problem_l74_7497

theorem unicorn_rope_problem (tower_radius : ℝ) (rope_length : ℝ) (rope_end_distance : ℝ)
  (a b c : ℕ) (h_radius : tower_radius = 10)
  (h_rope_length : rope_length = 25)
  (h_rope_end : rope_end_distance = 5)
  (h_c_prime : Nat.Prime c)
  (h_rope_touch : (a : ℝ) - Real.sqrt b = c * (rope_length - Real.sqrt ((tower_radius + rope_end_distance) ^ 2 + 5 ^ 2))) :
  a + b + c = 136 := by
sorry

end NUMINAMATH_CALUDE_unicorn_rope_problem_l74_7497


namespace NUMINAMATH_CALUDE_unique_charming_number_l74_7490

theorem unique_charming_number :
  ∃! (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 10 * a + b = 2 * a + b^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_charming_number_l74_7490


namespace NUMINAMATH_CALUDE_residue_calculation_l74_7435

theorem residue_calculation : (230 * 15 - 20 * 9 + 5) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l74_7435


namespace NUMINAMATH_CALUDE_track_length_l74_7478

/-- Represents a circular running track with two runners -/
structure CircularTrack where
  length : ℝ
  initial_distance : ℝ
  first_meeting_distance : ℝ
  second_meeting_additional_distance : ℝ

/-- The track satisfies the given conditions -/
def satisfies_conditions (track : CircularTrack) : Prop :=
  track.initial_distance = 120 ∧
  track.first_meeting_distance = 150 ∧
  track.second_meeting_additional_distance = 200

/-- The theorem stating the length of the track -/
theorem track_length (track : CircularTrack) 
  (h : satisfies_conditions track) : track.length = 450 := by
  sorry

end NUMINAMATH_CALUDE_track_length_l74_7478


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l74_7424

theorem contrapositive_equivalence (a b : ℝ) :
  (∀ a b, a > b → a - 1 > b - 1) ↔ (∀ a b, a - 1 ≤ b - 1 → a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l74_7424
