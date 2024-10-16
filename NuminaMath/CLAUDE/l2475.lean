import Mathlib

namespace NUMINAMATH_CALUDE_simplify_fraction_l2475_247536

theorem simplify_fraction (a b : ℝ) 
  (h1 : a ≠ -b) (h2 : a ≠ 2*b) (h3 : a ≠ b) (h4 : a ≠ -b) : 
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2475_247536


namespace NUMINAMATH_CALUDE_division_problem_l2475_247504

theorem division_problem : (250 : ℝ) / (15 + 13 * 3 - 4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2475_247504


namespace NUMINAMATH_CALUDE_number_order_l2475_247591

theorem number_order : 
  let a : ℝ := 30.5
  let b : ℝ := 0.53
  let c : ℝ := Real.log 0.53
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_number_order_l2475_247591


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2475_247545

/-- An increasing geometric sequence -/
def IsIncreasingGeometricSeq (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 1 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_incr_geom : IsIncreasingGeometricSeq a)
  (h_sum : a 4 + a 6 = 6)
  (h_prod : a 2 * a 8 = 8) :
  a 3 = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2475_247545


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_zero_one_l2475_247557

def A : Set ℝ := {x : ℝ | x^2 - 1 < 0}
def B : Set ℝ := {x : ℝ | x > 0}

theorem A_intersect_B_eq_open_zero_one : A ∩ B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_zero_one_l2475_247557


namespace NUMINAMATH_CALUDE_spinner_probability_F_l2475_247561

/-- Represents a spinner with three sections -/
structure Spinner :=
  (D : ℚ) (E : ℚ) (F : ℚ)

/-- The probability of landing on each section of the spinner -/
def probability (s : Spinner) : ℚ := s.D + s.E + s.F

theorem spinner_probability_F (s : Spinner) 
  (hD : s.D = 2/5) 
  (hE : s.E = 1/5) 
  (hP : probability s = 1) : 
  s.F = 2/5 := by
  sorry


end NUMINAMATH_CALUDE_spinner_probability_F_l2475_247561


namespace NUMINAMATH_CALUDE_union_A_B_when_a_is_one_value_set_of_a_when_intersection_empty_l2475_247547

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < a * x - 1 ∧ a * x - 1 ≤ 5}
def B : Set ℝ := {x | -1/2 < x ∧ x ≤ 2}

-- Theorem for part I
theorem union_A_B_when_a_is_one :
  A 1 ∪ B = {x : ℝ | -1/2 < x ∧ x ≤ 6} := by sorry

-- Theorem for part II
theorem value_set_of_a_when_intersection_empty :
  ∀ a : ℝ, a ≥ 0 → (A a ∩ B = ∅ ↔ 0 ≤ a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_is_one_value_set_of_a_when_intersection_empty_l2475_247547


namespace NUMINAMATH_CALUDE_circle_intersection_range_l2475_247508

theorem circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, (x - 2*a)^2 + (y - (a + 3))^2 = 4 ∧ x^2 + y^2 = 1) →
  -6/5 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l2475_247508


namespace NUMINAMATH_CALUDE_function_properties_l2475_247502

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 2*b

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x ≤ f a b (-1)) ∧
    (f a b (-1) = 2) ∧
    (a = 2 ∧ b = 1) ∧
    (∀ x ∈ Set.Icc (-1) 1, f a b x ≤ 6) ∧
    (∀ x ∈ Set.Icc (-1) 1, f a b x ≥ 50/27) ∧
    (∃ x ∈ Set.Icc (-1) 1, f a b x = 6) ∧
    (∃ x ∈ Set.Icc (-1) 1, f a b x = 50/27) :=
by
  sorry


end NUMINAMATH_CALUDE_function_properties_l2475_247502


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2475_247572

theorem trigonometric_identities (α : Real) 
  (h1 : (Real.tan α) / (Real.tan α - 1) = -1)
  (h2 : α ∈ Set.Icc (Real.pi) (3 * Real.pi / 2)) :
  (((Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α)) = -5/3) ∧
  ((Real.cos (-Real.pi + α) + Real.cos (Real.pi/2 + α)) = 3 * Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2475_247572


namespace NUMINAMATH_CALUDE_equation_solution_l2475_247540

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => (20 / (x^2 - 9)) - (3 / (x + 3)) - 2
  ∃ x₁ x₂ : ℝ, x₁ = (-3 + Real.sqrt 385) / 4 ∧ 
              x₂ = (-3 - Real.sqrt 385) / 4 ∧
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2475_247540


namespace NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l2475_247534

theorem no_integer_satisfies_conditions : ¬∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), n = 16 * k) ∧ 
  (23 < Real.sqrt n) ∧ 
  (Real.sqrt n < 23.2) := by
sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_conditions_l2475_247534


namespace NUMINAMATH_CALUDE_circle_area_theorem_l2475_247529

theorem circle_area_theorem (r : ℝ) (chord_length : ℝ) (intersection_distance : ℝ)
  (h1 : r = 42)
  (h2 : chord_length = 78)
  (h3 : intersection_distance = 18) :
  ∃ (m n d : ℕ), 
    (m * π - n * Real.sqrt d : ℝ) = 294 * π - 81 * Real.sqrt 3 ∧
    m + n + d = 378 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l2475_247529


namespace NUMINAMATH_CALUDE_same_speed_problem_l2475_247523

theorem same_speed_problem (x : ℝ) :
  let jack_speed := x^2 - 9*x - 18
  let jill_distance := x^2 - 5*x - 66
  let jill_time := x + 6
  let jill_speed := jill_distance / jill_time
  (x ≠ -6) →
  (jack_speed = jill_speed) →
  jack_speed = -4 :=
by sorry

end NUMINAMATH_CALUDE_same_speed_problem_l2475_247523


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2475_247505

theorem greatest_divisor_with_remainders : Nat.gcd (1557 - 7) (2037 - 5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2475_247505


namespace NUMINAMATH_CALUDE_lowest_degree_is_four_l2475_247553

/-- A polynomial with coefficients in ℤ -/
def IntPolynomial := Polynomial ℤ

/-- The set of coefficients of a polynomial -/
def coefficientSet (p : IntPolynomial) : Set ℤ :=
  {a : ℤ | ∃ (n : ℕ), p.coeff n = a}

/-- The property that a polynomial satisfies the given conditions -/
def satisfiesCondition (p : IntPolynomial) : Prop :=
  ∃ (b : ℤ),
    (∃ (a₁ : ℤ), a₁ ∈ coefficientSet p ∧ a₁ < b) ∧
    (∃ (a₂ : ℤ), a₂ ∈ coefficientSet p ∧ a₂ > b) ∧
    b ∉ coefficientSet p

/-- The theorem stating that the lowest degree of a polynomial satisfying the condition is 4 -/
theorem lowest_degree_is_four :
  ∃ (p : IntPolynomial),
    satisfiesCondition p ∧
    p.degree = 4 ∧
    ∀ (q : IntPolynomial), satisfiesCondition q → q.degree ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_lowest_degree_is_four_l2475_247553


namespace NUMINAMATH_CALUDE_vasya_distance_fraction_l2475_247513

/-- Represents the fraction of the total distance driven by each person -/
structure DistanceFractions where
  anton : ℚ
  vasya : ℚ
  sasha : ℚ
  dima : ℚ

/-- Theorem stating that given the conditions, Vasya drove 2/5 of the total distance -/
theorem vasya_distance_fraction 
  (df : DistanceFractions)
  (h1 : df.anton = df.vasya / 2)
  (h2 : df.sasha = df.anton + df.dima)
  (h3 : df.dima = 1 / 10)
  (h4 : df.anton + df.vasya + df.sasha + df.dima = 1) :
  df.vasya = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vasya_distance_fraction_l2475_247513


namespace NUMINAMATH_CALUDE_yard_to_stride_l2475_247526

-- Define the units of measurement
variable (step stride leap yard : ℚ)

-- Define the relationships between units
axiom step_stride_relation : 3 * step = 4 * stride
axiom leap_step_relation : 5 * leap = 2 * step
axiom leap_yard_relation : 7 * leap = 6 * yard

-- Theorem to prove
theorem yard_to_stride : yard = 28/45 * stride := by
  sorry

end NUMINAMATH_CALUDE_yard_to_stride_l2475_247526


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l2475_247510

/-- A circle with center (-1, 3) that is tangent to the line x - y = 0 -/
def tangentCircle (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 3)^2 = 8

/-- The line x - y = 0 -/
def tangentLine (x y : ℝ) : Prop :=
  x - y = 0

/-- The center of the circle -/
def circleCenter : ℝ × ℝ := (-1, 3)

theorem circle_tangent_to_line :
  ∃ (x₀ y₀ : ℝ), tangentCircle x₀ y₀ ∧ tangentLine x₀ y₀ ∧
  ∀ (x y : ℝ), tangentCircle x y ∧ tangentLine x y → (x, y) = (x₀, y₀) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l2475_247510


namespace NUMINAMATH_CALUDE_polygon_with_one_degree_exterior_angles_l2475_247549

/-- The number of sides in a polygon where each exterior angle measures 1 degree -/
def polygon_sides : ℕ := 360

/-- The measure of each exterior angle in degrees -/
def exterior_angle : ℝ := 1

/-- The sum of exterior angles in any polygon in degrees -/
def sum_exterior_angles : ℝ := 360

theorem polygon_with_one_degree_exterior_angles :
  (sum_exterior_angles / exterior_angle : ℝ) = polygon_sides := by sorry

end NUMINAMATH_CALUDE_polygon_with_one_degree_exterior_angles_l2475_247549


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l2475_247597

/-- A game on a circle where two players mark points -/
structure CircleGame where
  /-- The number of points each player marks -/
  p : ℕ
  /-- Condition that p is greater than 1 -/
  p_gt_one : p > 1

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins
  | Draw

/-- A strategy for playing the game -/
def Strategy := CircleGame → GameResult

/-- The theorem stating that the second player has a winning strategy -/
theorem second_player_winning_strategy (game : CircleGame) : 
  ∃ (s : Strategy), ∀ (opponent_strategy : Strategy), 
    s game = GameResult.SecondPlayerWins :=
sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l2475_247597


namespace NUMINAMATH_CALUDE_other_diagonal_length_l2475_247563

/-- A trapezoid with diagonals intersecting at a right angle -/
structure RightAngledTrapezoid where
  midline : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  diagonals_perpendicular : diagonal1 * diagonal2 = midline * midline * 2

/-- Theorem: In a right-angled trapezoid with midline 6.5 and one diagonal 12, the other diagonal is 5 -/
theorem other_diagonal_length (t : RightAngledTrapezoid) 
  (h1 : t.midline = 6.5) 
  (h2 : t.diagonal1 = 12) : 
  t.diagonal2 = 5 := by
sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l2475_247563


namespace NUMINAMATH_CALUDE_probability_of_correct_distribution_l2475_247566

/-- Represents the types of rolls -/
inductive RollType
  | Nut
  | Cheese
  | Fruit
  | Chocolate

/-- Represents a guest's set of rolls -/
def GuestRolls := Finset RollType

/-- The number of guests -/
def num_guests : Nat := 3

/-- The number of roll types -/
def num_roll_types : Nat := 4

/-- The total number of rolls -/
def total_rolls : Nat := num_guests * num_roll_types

/-- A function to calculate the probability of a specific distribution of rolls -/
noncomputable def probability_of_distribution (distribution : Finset GuestRolls) : ℚ := sorry

/-- The correct distribution where each guest has one of each roll type -/
def correct_distribution : Finset GuestRolls := sorry

/-- Theorem stating that the probability of the correct distribution is 24/1925 -/
theorem probability_of_correct_distribution :
  probability_of_distribution correct_distribution = 24 / 1925 := by sorry

end NUMINAMATH_CALUDE_probability_of_correct_distribution_l2475_247566


namespace NUMINAMATH_CALUDE_inequality_proof_l2475_247546

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2475_247546


namespace NUMINAMATH_CALUDE_evaluate_expression_l2475_247514

theorem evaluate_expression : 2022^3 - 2020 * 2022^2 - 2020^2 * 2022 + 2020^3 = 16168 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2475_247514


namespace NUMINAMATH_CALUDE_ball_attendees_l2475_247533

theorem ball_attendees :
  ∀ n m : ℕ,
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_attendees_l2475_247533


namespace NUMINAMATH_CALUDE_total_eggs_january_l2475_247538

/-- Represents a hen with a specific egg-laying frequency -/
structure Hen where
  frequency : ℕ  -- Number of days between each egg

/-- Calculates the number of eggs laid by a hen in a given number of days -/
def eggsLaid (h : Hen) (days : ℕ) : ℕ :=
  (days + h.frequency - 1) / h.frequency

/-- The three hens owned by Xiao Ming's family -/
def hens : List Hen := [
  { frequency := 1 },  -- First hen lays an egg every day
  { frequency := 2 },  -- Second hen lays an egg every two days
  { frequency := 3 }   -- Third hen lays an egg every three days
]

/-- The total number of eggs laid by all hens in January -/
def totalEggsInJanuary : ℕ :=
  (hens.map (eggsLaid · 31)).sum

theorem total_eggs_january : totalEggsInJanuary = 56 := by
  sorry

#eval totalEggsInJanuary  -- This should output 56

end NUMINAMATH_CALUDE_total_eggs_january_l2475_247538


namespace NUMINAMATH_CALUDE_roots_irrational_l2475_247598

theorem roots_irrational (p q : ℤ) (hp : Odd p) (hq : Odd q) 
  (h_real_roots : ∃ x y : ℝ, x ≠ y ∧ x^2 + 2*p*x + 2*q = 0 ∧ y^2 + 2*p*y + 2*q = 0) :
  ∀ z : ℝ, z^2 + 2*p*z + 2*q = 0 → Irrational z :=
sorry

end NUMINAMATH_CALUDE_roots_irrational_l2475_247598


namespace NUMINAMATH_CALUDE_f_sum_negative_l2475_247578

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_property_1 : ∀ x : ℝ, f (-x) = -f (x + 4)
axiom f_property_2 : ∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 2 → f x₁ > f x₂

-- Define the theorem
theorem f_sum_negative (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ < 4) 
  (h2 : (x₁ - 2) * (x₂ - 2) < 0) : 
  f x₁ + f x₂ < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_negative_l2475_247578


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2475_247503

theorem expand_and_simplify (x : ℝ) : (2 * x + 6) * (x + 9) = 2 * x^2 + 24 * x + 54 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2475_247503


namespace NUMINAMATH_CALUDE_average_marks_combined_l2475_247568

theorem average_marks_combined (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ) 
  (h₁ : n₁ = 12) (h₂ : n₂ = 28) (h₃ : avg₁ = 40) (h₄ : avg₂ = 60) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℚ) = 54 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_l2475_247568


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2475_247518

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- State the theorem
theorem set_intersection_theorem :
  (A ∩ B = {x | 0 < x ∧ x ≤ 2}) ∧
  (A ∩ (U \ B) = {x | x > 2}) := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2475_247518


namespace NUMINAMATH_CALUDE_g_difference_l2475_247521

-- Define the function g
def g (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 4

-- Theorem statement
theorem g_difference (x h : ℝ) : g (x + h) - g x = h * (12 * x + 6 * h - 3) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l2475_247521


namespace NUMINAMATH_CALUDE_complement_of_P_l2475_247565

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x : ℝ | x^2 ≤ 1}

-- State the theorem
theorem complement_of_P : 
  (Set.univ \ P) = {x : ℝ | x < -1 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_l2475_247565


namespace NUMINAMATH_CALUDE_abc_inequality_l2475_247582

theorem abc_inequality (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4*a*c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2475_247582


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2475_247571

theorem arithmetic_computation : 1325 + 572 / 52 - 225 + 2^3 = 1119 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2475_247571


namespace NUMINAMATH_CALUDE_hypotenuse_of_right_triangle_with_medians_l2475_247511

/-- A right triangle with specific median properties -/
structure RightTriangleWithMedians where
  -- The lengths of the two legs
  a : ℝ
  b : ℝ
  -- The medians from acute angles are both 6
  median_a : a^2 + (b/2)^2 = 36
  median_b : b^2 + (a/2)^2 = 36
  -- Ensure positivity of sides
  a_pos : a > 0
  b_pos : b > 0

/-- The hypotenuse of the right triangle with the given median properties is 2√57.6 -/
theorem hypotenuse_of_right_triangle_with_medians (t : RightTriangleWithMedians) :
  Real.sqrt ((2*t.a)^2 + (2*t.b)^2) = 2 * Real.sqrt 57.6 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_of_right_triangle_with_medians_l2475_247511


namespace NUMINAMATH_CALUDE_modulus_of_Z_l2475_247550

theorem modulus_of_Z (Z : ℂ) (h : (1 + Complex.I) * Z = Complex.I) : 
  Complex.abs Z = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_Z_l2475_247550


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2475_247519

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 8 / 27 →
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2475_247519


namespace NUMINAMATH_CALUDE_initial_speed_calculation_l2475_247530

/-- Proves that the initial speed satisfies the given equation under the problem conditions -/
theorem initial_speed_calculation (D T : ℝ) (hD : D > 0) (hT : T > 0) 
  (h_time_constraint : T/3 + (D/3) / 25 = T) : ∃ S : ℝ, S = 2*D/T ∧ S = 100 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_calculation_l2475_247530


namespace NUMINAMATH_CALUDE_prob_same_length_is_17_35_l2475_247524

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Set (ℕ × ℕ) := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of elements in set T -/
def total_elements : ℕ := num_sides + num_diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ :=
  (num_sides * (num_sides - 1) + num_diagonals * (num_diagonals - 1)) /
  (total_elements * (total_elements - 1))

theorem prob_same_length_is_17_35 : prob_same_length = 17 / 35 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_length_is_17_35_l2475_247524


namespace NUMINAMATH_CALUDE_intersection_of_M_and_S_l2475_247583

def M : Set ℕ := {x | 0 < x ∧ x < 4}
def S : Set ℕ := {2, 3, 5}

theorem intersection_of_M_and_S : M ∩ S = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_S_l2475_247583


namespace NUMINAMATH_CALUDE_quadratic_sum_equals_28_l2475_247506

theorem quadratic_sum_equals_28 (a b c : ℝ) 
  (h1 : a - b = 4) 
  (h2 : b + c = 2) : 
  a^2 + b^2 + c^2 - a*b + b*c + c*a = 28 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_equals_28_l2475_247506


namespace NUMINAMATH_CALUDE_adult_tickets_count_l2475_247552

theorem adult_tickets_count
  (adult_price : ℝ)
  (child_price : ℝ)
  (total_tickets : ℕ)
  (total_cost : ℝ)
  (h1 : adult_price = 5.5)
  (h2 : child_price = 3.5)
  (h3 : total_tickets = 21)
  (h4 : total_cost = 83.5) :
  ∃ (adult_count : ℕ) (child_count : ℕ),
    adult_count + child_count = total_tickets ∧
    adult_count * adult_price + child_count * child_price = total_cost ∧
    adult_count = 5 :=
by sorry

end NUMINAMATH_CALUDE_adult_tickets_count_l2475_247552


namespace NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l2475_247559

theorem square_area_to_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ > 0 ∧ s₂ > 0) :
  s₁^2 / s₂^2 = 16 / 49 → (4 * s₁) / (4 * s₂) = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l2475_247559


namespace NUMINAMATH_CALUDE_geometric_progression_quadratic_vertex_l2475_247548

/-- Given a geometric progression a, b, c, d and a quadratic function,
    prove that ad = 3 --/
theorem geometric_progression_quadratic_vertex (a b c d : ℝ) :
  (∃ (r : ℝ), b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric progression condition
  (2 * b^2 - 4 * b + 5 = c) →                      -- vertex condition
  a * d = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_quadratic_vertex_l2475_247548


namespace NUMINAMATH_CALUDE_function_properties_l2475_247555

variable (a b : ℝ × ℝ)

def f (x : ℝ) : ℝ := (x * a.1 + b.1) * (x * b.2 - a.2)

theorem function_properties
  (h1 : a ≠ (0, 0))
  (h2 : b ≠ (0, 0))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0)  -- perpendicular vectors
  (h4 : a.1^2 + a.2^2 ≠ b.1^2 + b.2^2)  -- different magnitudes
  : (∃ k : ℝ, ∀ x : ℝ, f a b x = k * x) ∧  -- first-order function
    (∀ x : ℝ, f a b x = -f a b (-x))  -- odd function
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l2475_247555


namespace NUMINAMATH_CALUDE_cost_price_of_cloth_l2475_247596

/-- Represents the cost price of one metre of cloth -/
def costPricePerMetre (totalMetres : ℕ) (sellingPrice : ℕ) (profitPerMetre : ℕ) : ℕ :=
  (sellingPrice - profitPerMetre * totalMetres) / totalMetres

/-- Theorem stating that the cost price of one metre of cloth is 85 rupees -/
theorem cost_price_of_cloth :
  costPricePerMetre 85 8925 20 = 85 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_cloth_l2475_247596


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l2475_247532

theorem arithmetic_sequence_sum_divisibility :
  ∀ (x c : ℕ+),
  ∃ (d : ℕ+),
  (d = 15) ∧
  (d ∣ (15 * x + 105 * c)) ∧
  (∀ (k : ℕ+), k > d → ¬(∀ (y z : ℕ+), k ∣ (15 * y + 105 * z))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l2475_247532


namespace NUMINAMATH_CALUDE_gcd_problem_l2475_247539

theorem gcd_problem : ∃ b : ℕ+, Nat.gcd (20 * b) (18 * 24) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2475_247539


namespace NUMINAMATH_CALUDE_min_bridges_correct_l2475_247574

/-- Represents a configuration of islands and bridges -/
structure IslandConfiguration where
  n : ℕ+  -- Number of islands (positive natural number)
  bridges : List (Fin n × Fin n)  -- List of bridges, each represented as a pair of island indices

/-- Checks if all islands are connected in the given configuration -/
def allConnected (config : IslandConfiguration) : Prop := sorry

/-- The minimum number of bridges required to connect n islands -/
def minBridges (n : ℕ+) : ℕ := n - 1

/-- Theorem stating that the minimum number of bridges to connect n islands is n - 1 -/
theorem min_bridges_correct (n : ℕ+) :
  ∃ (config : IslandConfiguration),
    config.n = n ∧
    config.bridges.length = minBridges n ∧
    allConnected config ∧
    ∀ (config' : IslandConfiguration),
      config'.n = n →
      config'.bridges.length < minBridges n →
      ¬allConnected config' :=
by sorry

end NUMINAMATH_CALUDE_min_bridges_correct_l2475_247574


namespace NUMINAMATH_CALUDE_expression_zero_iff_x_eq_three_l2475_247585

theorem expression_zero_iff_x_eq_three (x : ℝ) :
  (4 * x - 8 ≠ 0) →
  ((x^2 - 6*x + 9) / (4*x - 8) = 0 ↔ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_expression_zero_iff_x_eq_three_l2475_247585


namespace NUMINAMATH_CALUDE_green_upgrade_area_l2475_247558

/-- Proves that the actual average annual area of green upgrade is 90 million square meters --/
theorem green_upgrade_area (total_area : ℝ) (planned_years original_plan actual_plan : ℝ) :
  total_area = 180 →
  actual_plan = 2 * original_plan →
  planned_years - (total_area / actual_plan) = 2 →
  actual_plan = 90 := by
  sorry

end NUMINAMATH_CALUDE_green_upgrade_area_l2475_247558


namespace NUMINAMATH_CALUDE_max_sum_of_pairwise_sums_l2475_247556

/-- Given a set of four numbers with six pairwise sums, find the maximum value of x + y -/
theorem max_sum_of_pairwise_sums (a b c d : ℝ) : 
  let sums : Finset ℝ := {a + b, a + c, a + d, b + c, b + d, c + d}
  ∃ (x y : ℝ), x ∈ sums ∧ y ∈ sums ∧ 
    sums = {210, 345, 275, 255, x, y} →
    (∀ (u v : ℝ), u ∈ sums ∧ v ∈ sums → u + v ≤ 775) ∧
    (∃ (u v : ℝ), u ∈ sums ∧ v ∈ sums ∧ u + v = 775) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_pairwise_sums_l2475_247556


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l2475_247569

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : 4 * Real.pi * r₁^2 = (2/3) * (4 * Real.pi * r₂^2)) :
  (4/3) * Real.pi * r₁^3 = (2 * Real.sqrt 6 / 9) * ((4/3) * Real.pi * r₂^3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l2475_247569


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2475_247579

theorem triangle_perimeter : ∀ x : ℝ, 
  (x - 2) * (x - 4) = 0 →
  x + 3 > 6 →
  x + 3 + 6 = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2475_247579


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2475_247522

theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  bridge_length = 265 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2475_247522


namespace NUMINAMATH_CALUDE_loan_amount_to_c_l2475_247586

/-- Represents the loan details and interest calculation --/
structure LoanDetails where
  amount_b : ℝ  -- Amount lent to B
  amount_c : ℝ  -- Amount lent to C (to be determined)
  years_b : ℝ   -- Years for B's loan
  years_c : ℝ   -- Years for C's loan
  rate : ℝ      -- Annual interest rate
  total_interest : ℝ  -- Total interest received from both B and C

/-- Calculates the simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The main theorem to prove --/
theorem loan_amount_to_c 
  (loan : LoanDetails) 
  (h1 : loan.amount_b = 5000)
  (h2 : loan.years_b = 2)
  (h3 : loan.years_c = 4)
  (h4 : loan.rate = 0.09)
  (h5 : loan.total_interest = 1980)
  (h6 : simple_interest loan.amount_b loan.rate loan.years_b + 
        simple_interest loan.amount_c loan.rate loan.years_c = loan.total_interest) :
  loan.amount_c = 500 := by
  sorry


end NUMINAMATH_CALUDE_loan_amount_to_c_l2475_247586


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_3_l2475_247577

theorem sqrt_meaningful_iff_x_geq_3 (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_3_l2475_247577


namespace NUMINAMATH_CALUDE_complex_multiplication_l2475_247527

theorem complex_multiplication (i : ℂ) : i * i = -1 → (-1 + i) * (2 - i) = -1 + 3 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2475_247527


namespace NUMINAMATH_CALUDE_m_range_l2475_247592

-- Define the propositions
def P (t : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (t + 2) + y^2 / (t - 10) = 1

def Q (t m : ℝ) : Prop := 1 - m < t ∧ t < 1 + m ∧ m > 0

-- Define the relationship between P and Q
def relationship (m : ℝ) : Prop :=
  ∀ t, ¬(P t) → ¬(Q t m) ∧ ∃ t, ¬(Q t m) ∧ P t

-- State the theorem
theorem m_range :
  ∀ m, (∀ t, P t ↔ t ∈ Set.Ioo (-2) 10) →
       relationship m →
       m ∈ Set.Ioc 0 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2475_247592


namespace NUMINAMATH_CALUDE_martha_savings_l2475_247525

/-- Martha's daily allowance in dollars -/
def daily_allowance : ℚ := 12

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Number of days Martha saved half her allowance -/
def days_half_saved : ℕ := days_in_week - 1

/-- Amount saved when Martha saves half her allowance -/
def half_savings : ℚ := daily_allowance / 2

/-- Amount saved when Martha saves a quarter of her allowance -/
def quarter_savings : ℚ := daily_allowance / 4

/-- Martha's total savings for the week -/
def total_savings : ℚ := days_half_saved * half_savings + quarter_savings

theorem martha_savings : total_savings = 39 := by
  sorry

end NUMINAMATH_CALUDE_martha_savings_l2475_247525


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2475_247516

/-- A function f: ℝ → ℝ is quadratic if it can be expressed as f(x) = ax² + bx + c for some real constants a, b, and c, where a ≠ 0. -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = (x-1)(x-2) -/
def f (x : ℝ) : ℝ := (x - 1) * (x - 2)

/-- Theorem: The function f(x) = (x-1)(x-2) is a quadratic function -/
theorem f_is_quadratic : IsQuadratic f :=
sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2475_247516


namespace NUMINAMATH_CALUDE_initial_balance_was_800_liza_initial_balance_l2475_247554

/-- Represents the transactions in Liza's checking account --/
structure AccountTransactions where
  initial_balance : ℕ
  rent_payment : ℕ
  paycheck_deposit : ℕ
  electricity_bill : ℕ
  internet_bill : ℕ
  phone_bill : ℕ
  final_balance : ℕ

/-- Theorem stating that given the transactions and final balance, the initial balance was 800 --/
theorem initial_balance_was_800 (t : AccountTransactions) 
  (h1 : t.rent_payment = 450)
  (h2 : t.paycheck_deposit = 1500)
  (h3 : t.electricity_bill = 117)
  (h4 : t.internet_bill = 100)
  (h5 : t.phone_bill = 70)
  (h6 : t.final_balance = 1563)
  (h7 : t.initial_balance - t.rent_payment + t.paycheck_deposit - t.electricity_bill - t.internet_bill - t.phone_bill = t.final_balance) :
  t.initial_balance = 800 := by
  sorry

/-- Main theorem that proves Liza had $800 in her checking account on Tuesday --/
theorem liza_initial_balance : ∃ (t : AccountTransactions), t.initial_balance = 800 ∧ 
  t.rent_payment = 450 ∧
  t.paycheck_deposit = 1500 ∧
  t.electricity_bill = 117 ∧
  t.internet_bill = 100 ∧
  t.phone_bill = 70 ∧
  t.final_balance = 1563 ∧
  t.initial_balance - t.rent_payment + t.paycheck_deposit - t.electricity_bill - t.internet_bill - t.phone_bill = t.final_balance := by
  sorry

end NUMINAMATH_CALUDE_initial_balance_was_800_liza_initial_balance_l2475_247554


namespace NUMINAMATH_CALUDE_algebra_drafting_not_geography_algebra_drafting_not_geography_eq_25_l2475_247593

theorem algebra_drafting_not_geography (total_algebra : ℕ) (both_algebra_drafting : ℕ) 
  (drafting_only : ℕ) (total_geography : ℕ) (both_algebra_drafting_geography : ℕ) : ℕ :=
  let algebra_only := total_algebra - both_algebra_drafting
  let total_one_subject := algebra_only + drafting_only
  let result := total_one_subject - both_algebra_drafting_geography
  
  have h1 : total_algebra = 30 := by sorry
  have h2 : both_algebra_drafting = 15 := by sorry
  have h3 : drafting_only = 12 := by sorry
  have h4 : total_geography = 8 := by sorry
  have h5 : both_algebra_drafting_geography = 2 := by sorry

  result

theorem algebra_drafting_not_geography_eq_25 : 
  algebra_drafting_not_geography 30 15 12 8 2 = 25 := by sorry

end NUMINAMATH_CALUDE_algebra_drafting_not_geography_algebra_drafting_not_geography_eq_25_l2475_247593


namespace NUMINAMATH_CALUDE_bonsai_earnings_proof_l2475_247590

/-- Calculates the total earnings from selling bonsai. -/
def total_earnings (small_cost big_cost : ℕ) (small_sold big_sold : ℕ) : ℕ :=
  small_cost * small_sold + big_cost * big_sold

/-- Proves that the total earnings from selling 3 small bonsai at $30 each
    and 5 big bonsai at $20 each is equal to $190. -/
theorem bonsai_earnings_proof :
  total_earnings 30 20 3 5 = 190 := by
  sorry

end NUMINAMATH_CALUDE_bonsai_earnings_proof_l2475_247590


namespace NUMINAMATH_CALUDE_car_journey_time_l2475_247576

theorem car_journey_time (distance : ℝ) (new_speed : ℝ) (time_ratio : ℝ) 
  (h1 : distance = 450)
  (h2 : new_speed = 50)
  (h3 : time_ratio = 3/2)
  (h4 : distance = new_speed * (time_ratio * initial_time)) :
  initial_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_car_journey_time_l2475_247576


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l2475_247520

theorem intersection_point_of_lines (x y : ℚ) :
  (5 * x + 2 * y = 8) ∧ (11 * x - 5 * y = 1) ↔ x = 42/47 ∧ y = 83/47 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l2475_247520


namespace NUMINAMATH_CALUDE_partnership_profit_share_l2475_247560

/-- A partnership problem with four partners A, B, C, and D --/
theorem partnership_profit_share
  (capital_A : ℚ) (capital_B : ℚ) (capital_C : ℚ) (capital_D : ℚ) (total_profit : ℕ)
  (h1 : capital_A = 1 / 3)
  (h2 : capital_B = 1 / 4)
  (h3 : capital_C = 1 / 5)
  (h4 : capital_A + capital_B + capital_C + capital_D = 1)
  (h5 : total_profit = 2490) :
  ∃ (share_A : ℕ), share_A = 830 ∧ share_A = (capital_A * total_profit).num :=
sorry

end NUMINAMATH_CALUDE_partnership_profit_share_l2475_247560


namespace NUMINAMATH_CALUDE_no_equality_under_condition_l2475_247581

theorem no_equality_under_condition :
  ¬∃ (a b c : ℝ), (a^2 + b*c = (a + b)*(a + c)) ∧ (a + b + c = 2) :=
sorry

end NUMINAMATH_CALUDE_no_equality_under_condition_l2475_247581


namespace NUMINAMATH_CALUDE_box_comparison_l2475_247517

-- Define a structure for a box with three dimensions
structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

-- Define the comparison operation for boxes
def Box.lt (a b : Box) : Prop :=
  (a.x ≤ b.x ∧ a.y ≤ b.y ∧ a.z ≤ b.z) ∧
  (a.x < b.x ∨ a.y < b.y ∨ a.z < b.z)

-- Define boxes A, B, and C
def A : Box := ⟨6, 5, 3⟩
def B : Box := ⟨5, 4, 1⟩
def C : Box := ⟨3, 2, 2⟩

-- Theorem to prove A > B and C < A
theorem box_comparison :
  (Box.lt B A) ∧ (Box.lt C A) := by
  sorry

end NUMINAMATH_CALUDE_box_comparison_l2475_247517


namespace NUMINAMATH_CALUDE_vegetable_bins_l2475_247544

theorem vegetable_bins (total_bins soup_bins pasta_bins : ℚ)
  (h1 : total_bins = 0.75)
  (h2 : soup_bins = 0.12)
  (h3 : pasta_bins = 0.5)
  (h4 : total_bins = soup_bins + pasta_bins + (total_bins - soup_bins - pasta_bins)) :
  total_bins - soup_bins - pasta_bins = 0.13 := by
sorry

end NUMINAMATH_CALUDE_vegetable_bins_l2475_247544


namespace NUMINAMATH_CALUDE_jen_buys_50_candy_bars_l2475_247564

/-- The number of candy bars Jen buys -/
def num_candy_bars : ℕ := 50

/-- The cost of buying each candy bar in cents -/
def buy_price : ℕ := 80

/-- The selling price of each candy bar in cents -/
def sell_price : ℕ := 100

/-- The number of candy bars Jen sells -/
def num_sold : ℕ := 48

/-- Jen's profit in cents -/
def profit : ℕ := 800

/-- Theorem stating that given the conditions, Jen buys 50 candy bars -/
theorem jen_buys_50_candy_bars :
  (sell_price * num_sold) - (buy_price * num_candy_bars) = profit :=
by sorry

end NUMINAMATH_CALUDE_jen_buys_50_candy_bars_l2475_247564


namespace NUMINAMATH_CALUDE_locus_of_Q_is_ellipse_l2475_247512

/-- The ellipse C -/
def C (x y : ℝ) : Prop := x^2 / 24 + y^2 / 16 = 1

/-- The line l -/
def l (x y : ℝ) : Prop := x / 12 + y / 8 = 1

/-- Point R is on ellipse C -/
def R_on_C (xR yR : ℝ) : Prop := C xR yR

/-- Point P is on line l -/
def P_on_l (xP yP : ℝ) : Prop := l xP yP

/-- Q is on OP and satisfies |OQ| * |OP| = |OR|² -/
def Q_condition (xQ yQ xP yP xR yR : ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ xQ = t * xP ∧ yQ = t * yP ∧
  t * (xP^2 + yP^2) = xR^2 + yR^2

/-- The resulting ellipse for Q -/
def Q_ellipse (x y : ℝ) : Prop := (x - 1)^2 / (5/2) + (y - 1)^2 / (5/3) = 1

/-- Main theorem: The locus of Q is the ellipse (x-1)²/(5/2) + (y-1)²/(5/3) = 1 -/
theorem locus_of_Q_is_ellipse :
  ∀ (xQ yQ xP yP xR yR : ℝ),
    P_on_l xP yP →
    R_on_C xR yR →
    Q_condition xQ yQ xP yP xR yR →
    Q_ellipse xQ yQ :=
sorry

end NUMINAMATH_CALUDE_locus_of_Q_is_ellipse_l2475_247512


namespace NUMINAMATH_CALUDE_complement_characterization_l2475_247562

-- Define the universe of quadrilaterals
def Quadrilateral : Type := sorry

-- Define properties of quadrilaterals
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def has_right_angle (q : Quadrilateral) : Prop := sorry

-- Define sets A and B
def A : Set Quadrilateral := {q | is_rhombus q ∨ is_rectangle q}
def B : Set Quadrilateral := {q | is_rectangle q}

-- Define the complement of B with respect to A
def C_AB : Set Quadrilateral := A \ B

-- Theorem to prove
theorem complement_characterization :
  C_AB = {q : Quadrilateral | is_rhombus q ∧ ¬has_right_angle q} :=
sorry

end NUMINAMATH_CALUDE_complement_characterization_l2475_247562


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2475_247531

theorem sufficient_but_not_necessary (x : ℝ) :
  (((1 : ℝ) / x < 1) → (x > 1)) ∧ ¬((x > 1) → ((1 : ℝ) / x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2475_247531


namespace NUMINAMATH_CALUDE_town_shoppers_count_l2475_247500

/-- Represents the shopping scenario in the town. -/
structure ShoppingScenario where
  stores : Nat
  total_visits : Nat
  double_visitors : Nat
  max_visits_per_person : Nat

/-- The specific shopping scenario described in the problem. -/
def town_scenario : ShoppingScenario :=
  { stores := 8
  , total_visits := 22
  , double_visitors := 8
  , max_visits_per_person := 3 }

/-- The number of people who went shopping given a shopping scenario. -/
def shoppers (s : ShoppingScenario) : Nat :=
  s.double_visitors + (s.total_visits - 2 * s.double_visitors) / s.max_visits_per_person

/-- Theorem stating that the number of shoppers in the town scenario is 10. -/
theorem town_shoppers_count :
  shoppers town_scenario = 10 := by
  sorry

end NUMINAMATH_CALUDE_town_shoppers_count_l2475_247500


namespace NUMINAMATH_CALUDE_power_difference_l2475_247537

theorem power_difference (a : ℕ) (h : 5^a = 3125) : 5^(a-3) = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l2475_247537


namespace NUMINAMATH_CALUDE_dog_care_time_ratio_l2475_247551

/-- Proves the ratio of blow-drying time to bathing time for Marcus and his dog --/
theorem dog_care_time_ratio 
  (total_time : ℕ) 
  (bath_time : ℕ) 
  (walk_speed : ℚ) 
  (walk_distance : ℚ) 
  (h1 : total_time = 60) 
  (h2 : bath_time = 20) 
  (h3 : walk_speed = 6) 
  (h4 : walk_distance = 3) : 
  (total_time - bath_time - (walk_distance / walk_speed * 60).floor) * 2 = bath_time := by
sorry


end NUMINAMATH_CALUDE_dog_care_time_ratio_l2475_247551


namespace NUMINAMATH_CALUDE_square_difference_mental_calculation_l2475_247507

theorem square_difference (n : ℕ) : 
  ((n + 1) ^ 2 : ℕ) = n ^ 2 + 2 * n + 1 ∧ 
  ((n - 1) ^ 2 : ℕ) = n ^ 2 - 2 * n + 1 := by
  sorry

theorem mental_calculation : 
  (41 ^ 2 : ℕ) = 40 ^ 2 + 81 ∧ 
  (39 ^ 2 : ℕ) = 40 ^ 2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_mental_calculation_l2475_247507


namespace NUMINAMATH_CALUDE_coin_combination_l2475_247575

/-- Represents the number of different coin values that can be obtained -/
def different_values (five_cent : ℕ) (twenty_five_cent : ℕ) : ℕ :=
  75 - 4 * five_cent

theorem coin_combination (five_cent : ℕ) (twenty_five_cent : ℕ) :
  five_cent + twenty_five_cent = 15 →
  different_values five_cent twenty_five_cent = 27 →
  twenty_five_cent = 3 := by
sorry

end NUMINAMATH_CALUDE_coin_combination_l2475_247575


namespace NUMINAMATH_CALUDE_line_through_points_l2475_247584

/-- Given a line with slope 3 passing through points (3, 4) and (x, 7), prove that x = 4 -/
theorem line_through_points (x : ℝ) : 
  (7 - 4) / (x - 3) = 3 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2475_247584


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l2475_247595

theorem min_sum_absolute_values (a b c : ℝ) (h : |((a - b) * (b - c) * (c - a))| = 1) :
  ∃ (x y z : ℝ), |((x - y) * (y - z) * (z - x))| = 1 ∧ 
  (∀ (p q r : ℝ), |((p - q) * (q - r) * (r - p))| = 1 → |x| + |y| + |z| ≤ |p| + |q| + |r|) ∧
  |x| + |y| + |z| = Real.rpow 4 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l2475_247595


namespace NUMINAMATH_CALUDE_runner_speed_ratio_l2475_247501

/-- The runner's problem -/
theorem runner_speed_ratio (total_distance v₁ v₂ : ℝ) : 
  total_distance > 0 ∧
  v₁ > 0 ∧
  v₂ > 0 ∧
  total_distance / 2 / v₁ + 11 = total_distance / 2 / v₂ ∧
  total_distance / 2 / v₂ = 22 →
  v₁ / v₂ = 2 := by
  sorry

#check runner_speed_ratio

end NUMINAMATH_CALUDE_runner_speed_ratio_l2475_247501


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l2475_247509

-- Define the speeds and time
def alice_speed : ℚ := 1 / 12  -- miles per minute
def bob_speed : ℚ := 3 / 20    -- miles per minute
def duration : ℚ := 120        -- minutes (2 hours)

-- Theorem statement
theorem distance_after_two_hours :
  let alice_distance := alice_speed * duration
  let bob_distance := bob_speed * duration
  alice_distance + bob_distance = 28 := by
sorry


end NUMINAMATH_CALUDE_distance_after_two_hours_l2475_247509


namespace NUMINAMATH_CALUDE_triangle_problem_l2475_247588

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  c = 4 * Real.sqrt 2 →
  B = π / 4 →
  (1/2) * a * c * Real.sin B = 2 →
  a = 1 ∧ b = 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2475_247588


namespace NUMINAMATH_CALUDE_calculation_proof_equation_solutions_l2475_247543

-- Part 1: Prove the calculation
theorem calculation_proof :
  - Real.sqrt 12 + (2016 : ℝ) ^ 0 + |-3| + 4 * Real.cos (30 * π / 180) = 4 := by
  sorry

-- Part 2: Prove the equation solutions
theorem equation_solutions :
  ∀ x : ℝ, x^2 + 2*x - 8 = 0 ↔ x = -2 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_equation_solutions_l2475_247543


namespace NUMINAMATH_CALUDE_prop_1_false_prop_2_true_prop_3_false_l2475_247528

-- Proposition 1
theorem prop_1_false : ∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
  a * d = b * c ∧ ¬(∃ r : ℝ, (b = a * r ∧ c = b * r ∧ d = c * r) ∨ 
                             (a = b * r ∧ b = c * r ∧ c = d * r)) := by
  sorry

-- Proposition 2
theorem prop_2_true : ∀ (a : ℤ), 2 ∣ a → Even a := by
  sorry

-- Proposition 3
theorem prop_3_false : ∃ (A : ℝ), 
  30 * π / 180 < A ∧ A < π ∧ Real.sin A ≤ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prop_1_false_prop_2_true_prop_3_false_l2475_247528


namespace NUMINAMATH_CALUDE_fixed_point_existence_l2475_247515

theorem fixed_point_existence (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x : ℝ, x = a^(x - 2) - 3 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_existence_l2475_247515


namespace NUMINAMATH_CALUDE_only_caseD_has_two_solutions_l2475_247587

-- Define a structure for triangle cases
structure TriangleCase where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given cases
def caseA : TriangleCase := { a := 0, b := 10, c := 0, A := 45, B := 70, C := 0 }
def caseB : TriangleCase := { a := 60, b := 0, c := 48, A := 0, B := 100, C := 0 }
def caseC : TriangleCase := { a := 14, b := 16, c := 0, A := 45, B := 0, C := 0 }
def caseD : TriangleCase := { a := 7, b := 5, c := 0, A := 80, B := 0, C := 0 }

-- Function to determine if a case has two solutions
def hasTwoSolutions (t : TriangleCase) : Prop :=
  ∃ (B1 B2 : ℝ), B1 ≠ B2 ∧ 
    0 < B1 ∧ B1 < 180 ∧
    0 < B2 ∧ B2 < 180 ∧
    t.a / Real.sin t.A = t.b / Real.sin B1 ∧
    t.a / Real.sin t.A = t.b / Real.sin B2

-- Theorem stating that only case D has two solutions
theorem only_caseD_has_two_solutions :
  ¬(hasTwoSolutions caseA) ∧
  ¬(hasTwoSolutions caseB) ∧
  ¬(hasTwoSolutions caseC) ∧
  hasTwoSolutions caseD :=
sorry

end NUMINAMATH_CALUDE_only_caseD_has_two_solutions_l2475_247587


namespace NUMINAMATH_CALUDE_additional_investment_rate_l2475_247580

/-- Proves that the interest rate of an additional investment is 10% given specific conditions --/
theorem additional_investment_rate (initial_investment : ℝ) (initial_rate : ℝ) 
  (additional_investment : ℝ) (total_rate : ℝ) : 
  initial_investment = 2400 →
  initial_rate = 0.05 →
  additional_investment = 600 →
  total_rate = 0.06 →
  (initial_investment * initial_rate + additional_investment * 0.1) / 
    (initial_investment + additional_investment) = total_rate := by
  sorry

#check additional_investment_rate

end NUMINAMATH_CALUDE_additional_investment_rate_l2475_247580


namespace NUMINAMATH_CALUDE_ancient_chinese_car_problem_l2475_247589

/-- The number of cars in the ancient Chinese problem -/
def num_cars : ℕ := 15

/-- The number of people that can be accommodated when 3 people share a car -/
def people_three_per_car (x : ℕ) : ℕ := 3 * (x - 2)

/-- The number of people that can be accommodated when 2 people share a car -/
def people_two_per_car (x : ℕ) : ℕ := 2 * x

theorem ancient_chinese_car_problem :
  (people_three_per_car num_cars = people_two_per_car num_cars + 9) ∧
  (num_cars > 2) := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_car_problem_l2475_247589


namespace NUMINAMATH_CALUDE_intersection_point_l2475_247542

/-- The line equation y = x + 3 -/
def line_equation (x y : ℝ) : Prop := y = x + 3

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- Theorem: The point (0, 3) is the intersection of the line y = x + 3 and the y-axis -/
theorem intersection_point :
  line_equation 0 3 ∧ on_y_axis 0 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2475_247542


namespace NUMINAMATH_CALUDE_some_number_value_l2475_247535

theorem some_number_value (x : ℝ) : 65 + 5 * 12 / (180 / x) = 66 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2475_247535


namespace NUMINAMATH_CALUDE_sum_of_ages_l2475_247573

-- Define the ages of George, Christopher, and Ford
def christopher_age : ℕ := 18
def george_age : ℕ := christopher_age + 8
def ford_age : ℕ := christopher_age - 2

-- Theorem to prove
theorem sum_of_ages : george_age + christopher_age + ford_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2475_247573


namespace NUMINAMATH_CALUDE_solution_value_l2475_247567

/-- The function F as defined in the problem -/
def F (a b c : ℝ) : ℝ := a * b^2 + c

/-- Theorem stating that -1/8 is the solution to the equation F(a,3,8) = F(a,5,10) -/
theorem solution_value :
  ∃ a : ℝ, F a 3 8 = F a 5 10 ∧ a = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2475_247567


namespace NUMINAMATH_CALUDE_grocery_problem_l2475_247599

theorem grocery_problem (total_packs cookie_packs : ℕ) 
  (h1 : total_packs = 27)
  (h2 : cookie_packs = 23)
  (h3 : total_packs = cookie_packs + cake_packs) :
  cake_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_grocery_problem_l2475_247599


namespace NUMINAMATH_CALUDE_overlap_rectangle_area_l2475_247594

/-- Given two overlapping rectangles and the area of their intersection, 
    calculate the area of the non-overlapping part of the larger rectangle. -/
theorem overlap_rectangle_area (h w₁ w₂ : ℕ) (black_area : ℕ) 
    (h_pos : h > 0) (w₁_pos : w₁ > 0) (w₂_pos : w₂ > 0) (black_area_pos : black_area > 0)
    (h_le_w₁ : h ≤ w₁) (h_le_w₂ : h ≤ w₂) (w₁_le_w₂ : w₁ ≤ w₂) :
    h * w₂ - (h * w₁ - black_area) = 65 :=
by sorry

end NUMINAMATH_CALUDE_overlap_rectangle_area_l2475_247594


namespace NUMINAMATH_CALUDE_group_size_proof_l2475_247541

theorem group_size_proof (n : ℕ) 
  (h1 : (40 - 20 : ℝ) / n = 2.5) : n = 8 := by
  sorry

#check group_size_proof

end NUMINAMATH_CALUDE_group_size_proof_l2475_247541


namespace NUMINAMATH_CALUDE_function_range_theorem_l2475_247570

/-- A function f : ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f : ℝ → ℝ is decreasing on [0, +∞) if f(x) ≥ f(y) for all 0 ≤ x ≤ y -/
def IsDecreasingOnNonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≥ f y

theorem function_range_theorem (f : ℝ → ℝ) 
  (h_even : IsEven f) (h_decreasing : IsDecreasingOnNonnegatives f) :
  {x : ℝ | f (Real.log x) > f 1} = Set.Ioo (Real.exp (-1)) (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_function_range_theorem_l2475_247570
