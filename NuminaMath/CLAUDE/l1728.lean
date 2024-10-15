import Mathlib

namespace NUMINAMATH_CALUDE_single_point_equation_l1728_172830

/-- If the equation 3x^2 + y^2 + 6x - 12y + d = 0 represents a single point, then d = 39 -/
theorem single_point_equation (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + p.2^2 + 6 * p.1 - 12 * p.2 + d = 0) → d = 39 := by
  sorry

end NUMINAMATH_CALUDE_single_point_equation_l1728_172830


namespace NUMINAMATH_CALUDE_count_numbers_with_at_most_three_digits_is_900_l1728_172877

/-- Count of positive integers less than 1000 with at most three different digits -/
def count_numbers_with_at_most_three_digits : ℕ :=
  -- Single-digit numbers
  9 +
  -- Two-digit numbers
  (9 + 144 + 9) +
  -- Three-digit numbers
  (9 + 216 + 504)

/-- Theorem stating that the count of positive integers less than 1000
    with at most three different digits is 900 -/
theorem count_numbers_with_at_most_three_digits_is_900 :
  count_numbers_with_at_most_three_digits = 900 := by
  sorry

#eval count_numbers_with_at_most_three_digits

end NUMINAMATH_CALUDE_count_numbers_with_at_most_three_digits_is_900_l1728_172877


namespace NUMINAMATH_CALUDE_min_value_expression_l1728_172888

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x + 3 * y) * (y + 3 * z) * (x + 3 * z + 1) ≥ 24 * Real.sqrt 3 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (x₀ + 3 * y₀) * (y₀ + 3 * z₀) * (x₀ + 3 * z₀ + 1) = 24 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1728_172888


namespace NUMINAMATH_CALUDE_gcd_7163_209_l1728_172885

theorem gcd_7163_209 : Nat.gcd 7163 209 = 19 := by
  have h1 : 7163 = 209 * 34 + 57 := by sorry
  have h2 : 209 = 57 * 3 + 38 := by sorry
  have h3 : 57 = 38 * 1 + 19 := by sorry
  have h4 : 38 = 19 * 2 := by sorry
  sorry

end NUMINAMATH_CALUDE_gcd_7163_209_l1728_172885


namespace NUMINAMATH_CALUDE_ninety_degrees_to_radians_l1728_172823

theorem ninety_degrees_to_radians :
  let degrees_to_radians (d : ℝ) : ℝ := d * (π / 180)
  degrees_to_radians 90 = π / 2 := by sorry

end NUMINAMATH_CALUDE_ninety_degrees_to_radians_l1728_172823


namespace NUMINAMATH_CALUDE_m_returns_to_original_position_min_steps_to_return_l1728_172843

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the position of point M on side AB
def PositionM (t : Triangle) (a : ℝ) : ℝ × ℝ :=
  (a * t.A.1 + (1 - a) * t.B.1, a * t.A.2 + (1 - a) * t.B.2)

-- Define the movement of point M
def MoveM (t : Triangle) (pos : ℝ × ℝ) (step : ℕ) : ℝ × ℝ :=
  sorry

-- Theorem: M returns to its original position
theorem m_returns_to_original_position (t : Triangle) (a : ℝ) :
  ∃ n : ℕ, MoveM t (PositionM t a) n = PositionM t a :=
sorry

-- Theorem: Minimum number of steps for M to return
theorem min_steps_to_return (t : Triangle) (a : ℝ) :
  (a = 1/2 ∧ (∃ n : ℕ, n = 3 ∧ MoveM t (PositionM t a) n = PositionM t a ∧
    ∀ m : ℕ, m < n → MoveM t (PositionM t a) m ≠ PositionM t a)) ∨
  (a ≠ 1/2 ∧ (∃ n : ℕ, n = 6 ∧ MoveM t (PositionM t a) n = PositionM t a ∧
    ∀ m : ℕ, m < n → MoveM t (PositionM t a) m ≠ PositionM t a)) :=
sorry

end NUMINAMATH_CALUDE_m_returns_to_original_position_min_steps_to_return_l1728_172843


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_265_l1728_172828

-- Define the concept of angles having the same terminal side
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

-- State the theorem
theorem angle_with_same_terminal_side_as_negative_265 :
  same_terminal_side (-265) 95 :=
sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_265_l1728_172828


namespace NUMINAMATH_CALUDE_quadratic_fraction_value_l1728_172886

theorem quadratic_fraction_value (x : ℝ) (h : x^2 - 3*x + 1 = 0) : 
  x^2 / (x^4 + x^2 + 1) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_fraction_value_l1728_172886


namespace NUMINAMATH_CALUDE_village_population_theorem_l1728_172867

/-- Given a village with a total population and a subset of that population,
    calculate the percentage that the subset represents. -/
def village_population_percentage (total : ℕ) (subset : ℕ) : ℚ :=
  (subset : ℚ) / (total : ℚ) * 100

/-- Theorem stating that 45,000 is 90% of 50,000 -/
theorem village_population_theorem :
  village_population_percentage 50000 45000 = 90 := by
  sorry

end NUMINAMATH_CALUDE_village_population_theorem_l1728_172867


namespace NUMINAMATH_CALUDE_geometric_properties_l1728_172871

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of a line passing through a point
def passes_through (l : Line) (p : Point) : Prop := sorry

-- Define the concept of corresponding angles
def corresponding_angles (a1 a2 : Angle) : Prop := sorry

-- Define vertical angles
def vertical_angles (a1 a2 : Angle) : Prop := sorry

theorem geometric_properties :
  -- Statement 1
  (∀ a b c : Line, parallel a b → parallel b c → parallel a c) ∧
  -- Statement 2
  (∀ a1 a2 : Angle, corresponding_angles a1 a2 → a1 = a2) ∧
  -- Statement 3
  (∀ p : Point, ∀ l : Line, ∃! m : Line, passes_through m p ∧ parallel m l) ∧
  -- Statement 4
  (∀ a1 a2 : Angle, vertical_angles a1 a2 → a1 = a2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_properties_l1728_172871


namespace NUMINAMATH_CALUDE_parabola_directrix_l1728_172897

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x = -1/4 * y^2

/-- The directrix equation -/
def directrix (x : ℝ) : Prop := x = 1

/-- Theorem stating that the directrix of the given parabola is x = 1 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola x y → (∃ (f : ℝ), ∀ (x' y' : ℝ), 
    parabola x' y' → (x' - f)^2 + y'^2 = (x' - 1)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1728_172897


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_sum_l1728_172858

theorem sin_cos_sum_equals_sqrt_sum : 
  Real.sin (26 * π / 3) + Real.cos (-17 * π / 4) = (Real.sqrt 3 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_sum_l1728_172858


namespace NUMINAMATH_CALUDE_expression_evaluation_l1728_172811

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/5
  (2*x - 3)^2 - (x + 2*y)*(x - 2*y) - 3*y^2 + 3 = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1728_172811


namespace NUMINAMATH_CALUDE_triangle_extension_similarity_l1728_172873

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the similarity of triangles
def SimilarTriangles (t1 t2 : Triangle) : Prop := sorry

-- Define the extension of a line segment
def ExtendSegment (p1 p2 : ℝ × ℝ) (t : ℝ) : ℝ × ℝ := sorry

-- Define the length of a line segment
def SegmentLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem triangle_extension_similarity (ABC : Triangle) (P : ℝ × ℝ) :
  SegmentLength ABC.A ABC.B = 10 →
  SegmentLength ABC.B ABC.C = 9 →
  SegmentLength ABC.C ABC.A = 7 →
  P = ExtendSegment ABC.B ABC.C 1 →
  SimilarTriangles ⟨P, ABC.A, ABC.B⟩ ⟨P, ABC.C, ABC.A⟩ →
  SegmentLength P ABC.C = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_extension_similarity_l1728_172873


namespace NUMINAMATH_CALUDE_sandwich_jam_cost_l1728_172842

theorem sandwich_jam_cost :
  ∀ (N B J : ℕ),
  N > 1 →
  B > 0 →
  J > 0 →
  N * (3 * B + 7 * J) = 378 →
  (N * J * 7 : ℚ) / 100 = 2.52 := by
sorry

end NUMINAMATH_CALUDE_sandwich_jam_cost_l1728_172842


namespace NUMINAMATH_CALUDE_equation_solution_l1728_172836

theorem equation_solution (x : ℝ) : 
  (x = (-81 + Real.sqrt 5297) / 8 ∨ x = (-81 - Real.sqrt 5297) / 8) ↔ 
  (8 * x^2 + 89 * x + 3) / (3 * x + 41) = 4 * x + 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1728_172836


namespace NUMINAMATH_CALUDE_prob_at_least_one_karnataka_l1728_172840

/-- The probability of selecting at least one student from Karnataka -/
theorem prob_at_least_one_karnataka (total : ℕ) (karnataka : ℕ) (selected : ℕ)
  (h1 : total = 10)
  (h2 : karnataka = 3)
  (h3 : selected = 4) :
  (1 : ℚ) - (Nat.choose (total - karnataka) selected : ℚ) / (Nat.choose total selected : ℚ) = 5/6 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_karnataka_l1728_172840


namespace NUMINAMATH_CALUDE_total_tax_percentage_l1728_172820

-- Define the spending percentages
def clothing_percent : ℝ := 0.40
def food_percent : ℝ := 0.30
def other_percent : ℝ := 0.30

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.04
def food_tax_rate : ℝ := 0
def other_tax_rate : ℝ := 0.08

-- Theorem statement
theorem total_tax_percentage (total_spent : ℝ) (total_spent_pos : total_spent > 0) :
  let clothing_spent := clothing_percent * total_spent
  let food_spent := food_percent * total_spent
  let other_spent := other_percent * total_spent
  let clothing_tax := clothing_tax_rate * clothing_spent
  let food_tax := food_tax_rate * food_spent
  let other_tax := other_tax_rate * other_spent
  let total_tax := clothing_tax + food_tax + other_tax
  (total_tax / total_spent) * 100 = 4 := by
sorry

end NUMINAMATH_CALUDE_total_tax_percentage_l1728_172820


namespace NUMINAMATH_CALUDE_infinitely_many_squares_in_ap_l1728_172802

/-- An arithmetic progression of positive integers. -/
def ArithmeticProgression (a d : ℕ) : ℕ → ℕ
  | n => a + n * d

/-- Predicate to check if a number is a perfect square. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- The main theorem to be proved. -/
theorem infinitely_many_squares_in_ap (a d : ℕ) (h : d > 0) :
  (∃ n : ℕ, IsPerfectSquare (ArithmeticProgression a d n)) →
  (∀ m : ℕ, ∃ n : ℕ, n > m ∧ IsPerfectSquare (ArithmeticProgression a d n)) :=
by sorry


end NUMINAMATH_CALUDE_infinitely_many_squares_in_ap_l1728_172802


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l1728_172847

theorem largest_prime_factors_difference (n : Nat) (h : n = 242858) :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n ∧
  (∀ r : Nat, Nat.Prime r → r ∣ n → r ≤ p ∧ r ≤ q) ∧
  p ≠ q ∧ p - q = 80 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l1728_172847


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1728_172864

theorem hyperbola_asymptote (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (4 - m) + y^2 / (m - 2) = 1 → 
    (y = (1/3) * x ∨ y = -(1/3) * x)) → 
  m = 7/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1728_172864


namespace NUMINAMATH_CALUDE_middle_school_students_in_ganzhou_form_set_l1728_172846

-- Define the universe of discourse
def Universe : Type := Unit

-- Define the property of being a middle school student in Ganzhou
def IsMiddleSchoolStudentInGanzhou : Universe → Prop := sorry

-- Define what it means for a collection to have definite elements
def HasDefiniteElements (S : Set Universe) : Prop := sorry

-- Theorem: The set of all middle school students in Ganzhou has definite elements
theorem middle_school_students_in_ganzhou_form_set :
  HasDefiniteElements {x : Universe | IsMiddleSchoolStudentInGanzhou x} := by
  sorry

end NUMINAMATH_CALUDE_middle_school_students_in_ganzhou_form_set_l1728_172846


namespace NUMINAMATH_CALUDE_nested_average_equals_seven_eighteenths_l1728_172849

/-- Average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- Average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- The main theorem -/
theorem nested_average_equals_seven_eighteenths :
  avg3 (avg3 1 1 0) (avg2 0 1) 0 = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_nested_average_equals_seven_eighteenths_l1728_172849


namespace NUMINAMATH_CALUDE_train_crossing_time_l1728_172856

/-- The time taken for a train to cross a man walking in the same direction --/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 500 →
  train_speed = 75 * 1000 / 3600 →
  man_speed = 3 * 1000 / 3600 →
  train_length / (train_speed - man_speed) = 25 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1728_172856


namespace NUMINAMATH_CALUDE_sum_difference_1500_l1728_172831

/-- The sum of the first n odd counting numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The sum of the first n even counting numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even counting numbers
    and the sum of the first n odd counting numbers -/
def sumDifference (n : ℕ) : ℕ := sumEvenNumbers n - sumOddNumbers n

theorem sum_difference_1500 :
  sumDifference 1500 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_1500_l1728_172831


namespace NUMINAMATH_CALUDE_greg_earnings_l1728_172866

/-- Represents the rates for a dog size --/
structure DogRate where
  baseCharge : ℝ
  perMinuteCharge : ℝ

/-- Represents a group of dogs walked --/
structure DogGroup where
  count : ℕ
  minutes : ℕ

/-- Calculates the earnings for a group of dogs --/
def calculateEarnings (rate : DogRate) (group : DogGroup) : ℝ :=
  rate.baseCharge * group.count + rate.perMinuteCharge * group.count * group.minutes

/-- Theorem: Greg's total earnings for the day --/
theorem greg_earnings : 
  let extraSmallRate : DogRate := ⟨12, 0.80⟩
  let smallRate : DogRate := ⟨15, 1⟩
  let mediumRate : DogRate := ⟨20, 1.25⟩
  let largeRate : DogRate := ⟨25, 1.50⟩
  let extraLargeRate : DogRate := ⟨30, 1.75⟩

  let extraSmallGroup : DogGroup := ⟨2, 10⟩
  let smallGroup : DogGroup := ⟨3, 12⟩
  let mediumGroup : DogGroup := ⟨1, 18⟩
  let largeGroup : DogGroup := ⟨2, 25⟩
  let extraLargeGroup : DogGroup := ⟨1, 30⟩

  let totalEarnings := 
    calculateEarnings extraSmallRate extraSmallGroup +
    calculateEarnings smallRate smallGroup +
    calculateEarnings mediumRate mediumGroup +
    calculateEarnings largeRate largeGroup +
    calculateEarnings extraLargeRate extraLargeGroup

  totalEarnings = 371 := by sorry

end NUMINAMATH_CALUDE_greg_earnings_l1728_172866


namespace NUMINAMATH_CALUDE_exists_a_plus_ω_l1728_172874

noncomputable def f (ω a x : ℝ) : ℝ := Real.sin (ω * x) + a * Real.cos (ω * x)

theorem exists_a_plus_ω : ∃ (a ω : ℝ), 
  ω > 0 ∧ 
  (∀ x, f ω a x = f ω a (2 * Real.pi / 3 - x)) ∧ 
  (∀ x, f ω a (Real.pi / 6) ≤ f ω a x) ∧ 
  0 ≤ a + ω ∧ a + ω ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_exists_a_plus_ω_l1728_172874


namespace NUMINAMATH_CALUDE_square_area_with_side_5_l1728_172815

theorem square_area_with_side_5 :
  let side_length : ℝ := 5
  let area : ℝ := side_length * side_length
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_with_side_5_l1728_172815


namespace NUMINAMATH_CALUDE_triangles_in_decagon_l1728_172860

/-- The number of triangles that can be formed from the vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- A regular decagon has 10 sides -/
def decagonSides : ℕ := 10

/-- Theorem: The number of triangles that can be formed from the vertices of a regular decagon is equal to the number of ways to choose 3 vertices out of 10 -/
theorem triangles_in_decagon :
  trianglesInDecagon = Nat.choose decagonSides 3 := by
  sorry

#eval trianglesInDecagon -- Should output 120

end NUMINAMATH_CALUDE_triangles_in_decagon_l1728_172860


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1728_172813

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → x * f y - y * f x = x * y * f (x / y)

/-- Theorem stating that for any function satisfying the functional equation, f(50) = 0 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1728_172813


namespace NUMINAMATH_CALUDE_files_remaining_l1728_172851

theorem files_remaining (music_files video_files deleted_files : ℕ) :
  music_files = 4 →
  video_files = 21 →
  deleted_files = 23 →
  music_files + video_files - deleted_files = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l1728_172851


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_and_two_plus_two_not_zero_l1728_172891

theorem absolute_value_nonnegative_and_two_plus_two_not_zero :
  (∀ x : ℝ, |x| ≥ 0) ∧ ¬(2 + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_and_two_plus_two_not_zero_l1728_172891


namespace NUMINAMATH_CALUDE_average_book_width_l1728_172808

def book_widths : List ℝ := [4, 0.5, 1.2, 3, 7.5, 2, 5, 9]

theorem average_book_width : 
  (List.sum book_widths) / (List.length book_widths) = 4.025 := by
  sorry

end NUMINAMATH_CALUDE_average_book_width_l1728_172808


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1728_172899

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 4 / (x + y)^2 ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 4 / (x + y)^2 = 2 * Real.sqrt 2 ↔ x = y ∧ x = (Real.sqrt 2)^(3/4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1728_172899


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l1728_172824

/-- A triangle with two sides of length 3 and 6 can have a third side of length 6 -/
theorem triangle_third_side_length : ∃ (a b c : ℝ), 
  a = 3 ∧ b = 6 ∧ c = 6 ∧ 
  a + b > c ∧ b + c > a ∧ a + c > b ∧
  a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l1728_172824


namespace NUMINAMATH_CALUDE_inequality_problem_l1728_172834

theorem inequality_problem (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : x^2 < a^2) (h2 : y^3 < b^3) :
  ∃ (s : Finset (Fin 4)),
    s.card = 3 ∧
    (∀ i ∈ s, match i with
      | 0 => x^2 + y^2 < a^2 + b^2
      | 1 => x^2 - y^2 < a^2 - b^2
      | 2 => x^2 * y^3 < a^2 * b^3
      | 3 => x^2 / y^3 < a^2 / b^3
    ) ∧
    (∀ i ∉ s, ¬(match i with
      | 0 => x^2 + y^2 < a^2 + b^2
      | 1 => x^2 - y^2 < a^2 - b^2
      | 2 => x^2 * y^3 < a^2 * b^3
      | 3 => x^2 / y^3 < a^2 / b^3
    )) := by sorry

end NUMINAMATH_CALUDE_inequality_problem_l1728_172834


namespace NUMINAMATH_CALUDE_negative_three_times_b_minus_a_l1728_172832

theorem negative_three_times_b_minus_a (a b : ℚ) (h : a - b = 1/2) : -3 * (b - a) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_times_b_minus_a_l1728_172832


namespace NUMINAMATH_CALUDE_correct_number_of_workers_l1728_172829

/-- The number of workers in the team -/
def n : ℕ := 9

/-- The number of days it takes the full team to complete the task -/
def full_team_days : ℕ := 7

/-- The number of days it takes the team minus two workers to complete the task -/
def team_minus_two_days : ℕ := 14

/-- The number of days it takes the team minus six workers to complete the task -/
def team_minus_six_days : ℕ := 42

/-- The theorem stating that n is the correct number of workers -/
theorem correct_number_of_workers :
  n * full_team_days = (n - 2) * team_minus_two_days ∧
  n * full_team_days = (n - 6) * team_minus_six_days :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_workers_l1728_172829


namespace NUMINAMATH_CALUDE_reciprocal_and_inverse_sum_l1728_172852

theorem reciprocal_and_inverse_sum (a b : ℚ) (ha : a = 1 / a) (hb : b = -b) :
  a^2007 + b^2007 = 1 ∨ a^2007 + b^2007 = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_inverse_sum_l1728_172852


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1728_172825

theorem gcd_of_specific_numbers : Nat.gcd 333333 7777777 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1728_172825


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_50_l1728_172890

theorem four_digit_divisible_by_50 : 
  (Finset.filter 
    (fun n : ℕ => n ≥ 1000 ∧ n < 10000 ∧ n % 100 = 50 ∧ n % 50 = 0) 
    (Finset.range 10000)).card = 90 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_50_l1728_172890


namespace NUMINAMATH_CALUDE_expression_simplification_l1728_172862

theorem expression_simplification (a b : ℝ) : 
  (8 * a^3 * b) * (4 * a * b^2) * (1 / (2 * a * b)^3) = 4 * a := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1728_172862


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1728_172854

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1728_172854


namespace NUMINAMATH_CALUDE_existence_of_common_source_l1728_172845

/-- The type of positive integers. -/
def PositiveInt := { n : ℕ // n > 0 }

/-- Predicate to check if a number contains the digit 5. -/
def containsFive (n : PositiveInt) : Prop :=
  ∃ d, d ∈ n.val.digits 10 ∧ d = 5

/-- The process of replacing two consecutive digits with the last digit of their product. -/
def replaceDigits (n : PositiveInt) : PositiveInt :=
  sorry

/-- A number m is obtainable from n if there exists a finite sequence of replaceDigits operations. -/
def isObtainable (m n : PositiveInt) : Prop :=
  sorry

/-- Main theorem: For any finite set of positive integers without digit 5, 
    there exists a positive integer from which all elements are obtainable. -/
theorem existence_of_common_source (S : Finset PositiveInt) 
  (h : ∀ s ∈ S, ¬containsFive s) : 
  ∃ N : PositiveInt, ∀ s ∈ S, isObtainable s N :=
sorry

end NUMINAMATH_CALUDE_existence_of_common_source_l1728_172845


namespace NUMINAMATH_CALUDE_labor_costs_theorem_l1728_172857

/-- Calculates the overall labor costs for one day given the number of workers and their wages. -/
def overall_labor_costs (
  num_construction_workers : ℕ)
  (num_electricians : ℕ)
  (num_plumbers : ℕ)
  (construction_worker_wage : ℚ)
  (electrician_wage_multiplier : ℚ)
  (plumber_wage_multiplier : ℚ) : ℚ :=
  (num_construction_workers * construction_worker_wage) +
  (num_electricians * (electrician_wage_multiplier * construction_worker_wage)) +
  (num_plumbers * (plumber_wage_multiplier * construction_worker_wage))

/-- Proves that the overall labor costs for one day is $650 given the specified conditions. -/
theorem labor_costs_theorem :
  overall_labor_costs 2 1 1 100 2 (5/2) = 650 := by
  sorry

#eval overall_labor_costs 2 1 1 100 2 (5/2)

end NUMINAMATH_CALUDE_labor_costs_theorem_l1728_172857


namespace NUMINAMATH_CALUDE_photo_size_is_1_5_l1728_172881

/-- The size of a photo in kilobytes -/
def photo_size : ℝ := sorry

/-- The total storage space of the drive in kilobytes -/
def total_storage : ℝ := 2000 * photo_size

/-- The space used by 400 photos in kilobytes -/
def space_400_photos : ℝ := 400 * photo_size

/-- The space used by 12 200-kilobyte videos in kilobytes -/
def space_12_videos : ℝ := 12 * 200

/-- Theorem stating that the size of each photo is 1.5 kilobytes -/
theorem photo_size_is_1_5 : photo_size = 1.5 := by
  have h1 : total_storage = space_400_photos + space_12_videos := sorry
  sorry

end NUMINAMATH_CALUDE_photo_size_is_1_5_l1728_172881


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l1728_172848

/-- Proves that the speed of a stream is 12.6 kmph given specific boat travel conditions -/
theorem stream_speed_calculation (boat_speed : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : downstream_time = 1)
  (h3 : upstream_time = 11.5) :
  ∃ (stream_speed : ℝ), 
    stream_speed = 12.6 ∧ 
    (boat_speed + stream_speed) * downstream_time = (boat_speed - stream_speed) * upstream_time :=
by
  sorry


end NUMINAMATH_CALUDE_stream_speed_calculation_l1728_172848


namespace NUMINAMATH_CALUDE_polygon_sides_from_diagonals_l1728_172844

theorem polygon_sides_from_diagonals (d : ℕ) (h : d = 44) : ∃ n : ℕ, n ≥ 3 ∧ d = n * (n - 3) / 2 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_diagonals_l1728_172844


namespace NUMINAMATH_CALUDE_toy_store_revenue_l1728_172889

theorem toy_store_revenue (D : ℚ) (D_pos : D > 0) : 
  let N := (2 : ℚ) / 5 * D
  let J := (1 : ℚ) / 5 * N
  let F := (3 : ℚ) / 4 * D
  let avg := (N + J + F) / 3
  D / avg = 100 / 41 := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_l1728_172889


namespace NUMINAMATH_CALUDE_not_always_true_parallel_transitivity_l1728_172812

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel_lines : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem not_always_true_parallel_transitivity 
  (a b : Line) (α : Plane) : 
  ¬(∀ (a b : Line) (α : Plane), 
    parallel_lines a b → 
    parallel_line_plane a α → 
    parallel_line_plane b α) :=
by
  sorry


end NUMINAMATH_CALUDE_not_always_true_parallel_transitivity_l1728_172812


namespace NUMINAMATH_CALUDE_local_extrema_condition_l1728_172879

/-- A function f with parameter a that we want to analyze -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a + 6

theorem local_extrema_condition (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), IsLocalMin (f a) x₁ ∧ IsLocalMax (f a) x₂) ↔ (a ≤ -3 ∨ a > 6) :=
sorry

end NUMINAMATH_CALUDE_local_extrema_condition_l1728_172879


namespace NUMINAMATH_CALUDE_plane_contains_points_l1728_172807

def point1 : ℝ × ℝ × ℝ := (2, -1, 3)
def point2 : ℝ × ℝ × ℝ := (4, -1, 5)
def point3 : ℝ × ℝ × ℝ := (5, -3, 4)

def plane_equation (x y z : ℝ) : Prop := x + 2*y - 2*z + 6 = 0

theorem plane_contains_points :
  (plane_equation point1.1 point1.2.1 point1.2.2) ∧
  (plane_equation point2.1 point2.2.1 point2.2.2) ∧
  (plane_equation point3.1 point3.2.1 point3.2.2) ∧
  (1 > 0) ∧
  (Nat.gcd (Nat.gcd (Nat.gcd 1 2) 2) 6 = 1) :=
by sorry

end NUMINAMATH_CALUDE_plane_contains_points_l1728_172807


namespace NUMINAMATH_CALUDE_x_range_theorem_l1728_172806

theorem x_range_theorem (x : ℝ) : 
  (∀ (a b : ℝ), a > 0 → b > 0 → |x + 1| + |x - 2| ≤ (a + 1/b) * (1/a + b)) →
  -3/2 ≤ x ∧ x ≤ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_theorem_l1728_172806


namespace NUMINAMATH_CALUDE_eunji_shopping_l1728_172853

theorem eunji_shopping (initial_money : ℝ) : 
  initial_money * (1 - 1/4) * (1 - 1/3) = 1600 → initial_money = 3200 := by
  sorry

end NUMINAMATH_CALUDE_eunji_shopping_l1728_172853


namespace NUMINAMATH_CALUDE_sequence_kth_term_value_l1728_172895

/-- Given a sequence {a_n} with sum S_n = n^2 - 9n and 5 < a_k < 8, prove k = 8 -/
theorem sequence_kth_term_value (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ) :
  (∀ n, S n = n^2 - 9*n) →
  (∀ n, a n = 2*n - 10) →
  (5 < a k ∧ a k < 8) →
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_kth_term_value_l1728_172895


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_150_75_l1728_172810

theorem largest_two_digit_prime_factor_of_binom_150_75 :
  (∃ (p : ℕ), Prime p ∧ 10 ≤ p ∧ p < 100 ∧ p ∣ Nat.choose 150 75 ∧
    ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 150 75 → q ≤ p) →
  (73 : ℕ).Prime ∧ 73 ∣ Nat.choose 150 75 ∧
    ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 150 75 → q ≤ 73 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_150_75_l1728_172810


namespace NUMINAMATH_CALUDE_marias_carrots_l1728_172898

theorem marias_carrots : 
  ∃ (initial_carrots : ℕ), 
    initial_carrots - 11 + 15 = 52 ∧ 
    initial_carrots = 48 := by
  sorry

end NUMINAMATH_CALUDE_marias_carrots_l1728_172898


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_d_value_l1728_172827

/-- 
Given a polynomial of the form 3x^3 + dx + 9 with a factor x^2 + qx + 1,
prove that d = -24.
-/
theorem polynomial_factor_implies_d_value :
  ∀ d q : ℝ,
  (∃ c : ℝ, ∀ x : ℝ, 3*x^3 + d*x + 9 = (x^2 + q*x + 1) * (3*x + c)) →
  d = -24 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_d_value_l1728_172827


namespace NUMINAMATH_CALUDE_factor_cubic_expression_l1728_172859

theorem factor_cubic_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_factor_cubic_expression_l1728_172859


namespace NUMINAMATH_CALUDE_sally_coin_problem_l1728_172872

/-- Represents Sally's coin collection --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ

/-- Represents the changes in Sally's coin collection --/
def update_collection (initial : CoinCollection) (dad_nickels mom_nickels : ℕ) : CoinCollection :=
  { pennies := initial.pennies,
    nickels := initial.nickels + dad_nickels + mom_nickels }

theorem sally_coin_problem (initial : CoinCollection) (dad_nickels mom_nickels : ℕ) 
  (h1 : initial.nickels = 7)
  (h2 : dad_nickels = 9)
  (h3 : mom_nickels = 2)
  (h4 : (update_collection initial dad_nickels mom_nickels).nickels = 18) :
  initial.nickels = 7 ∧ ∀ (p : ℕ), ∃ (initial' : CoinCollection), 
    initial'.pennies = p ∧ 
    initial'.nickels = initial.nickels ∧
    (update_collection initial' dad_nickels mom_nickels).nickels = 18 :=
sorry

end NUMINAMATH_CALUDE_sally_coin_problem_l1728_172872


namespace NUMINAMATH_CALUDE_sandy_comic_books_l1728_172878

theorem sandy_comic_books :
  ∃ (initial : ℕ), (initial / 2 + 6 : ℕ) = 13 ∧ initial = 14 :=
by sorry

end NUMINAMATH_CALUDE_sandy_comic_books_l1728_172878


namespace NUMINAMATH_CALUDE_some_number_value_l1728_172816

theorem some_number_value (x : ℝ) (some_number : ℝ) 
  (h1 : x = 5)
  (h2 : (x / some_number) + 3 = 4) : 
  some_number = 5 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l1728_172816


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1728_172814

-- Define the original expression
def original_expression (x : ℝ) : ℝ := 4 * (x^2 - 2*x + 2) - 7 * (x^3 - 3*x + 1)

-- Define the simplified expression
def simplified_expression (x : ℝ) : ℝ := -7*x^3 + 4*x^2 + 13*x + 1

-- Theorem statement
theorem sum_of_squared_coefficients :
  ((-7)^2 + 4^2 + 13^2 + 1^2 = 235) ∧
  (∀ x : ℝ, original_expression x = simplified_expression x) :=
sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1728_172814


namespace NUMINAMATH_CALUDE_base_number_is_two_l1728_172865

theorem base_number_is_two (a : ℕ) (x : ℕ) (h1 : a^x - a^(x-2) = 3 * 2^11) (h2 : x = 13) :
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_is_two_l1728_172865


namespace NUMINAMATH_CALUDE_books_sold_correct_l1728_172894

/-- The number of books Paul sold in a garage sale -/
def books_sold : ℕ := 94

/-- The initial number of books Paul had -/
def initial_books : ℕ := 2

/-- The number of new books Paul bought -/
def new_books : ℕ := 150

/-- The final number of books Paul has -/
def final_books : ℕ := 58

/-- Theorem stating that the number of books Paul sold is correct -/
theorem books_sold_correct : 
  initial_books - books_sold + new_books = final_books :=
by sorry

end NUMINAMATH_CALUDE_books_sold_correct_l1728_172894


namespace NUMINAMATH_CALUDE_square_area_PS_l1728_172838

-- Define the triangles and their properties
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0

-- Define the configuration
def Configuration (P Q R S : ℝ × ℝ) : Prop :=
  Triangle P Q R ∧ Triangle P R S ∧
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 25 ∧
  (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = 4 ∧
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 49

-- State the theorem
theorem square_area_PS (P Q R S : ℝ × ℝ) 
  (h : Configuration P Q R S) : 
  (S.1 - P.1)^2 + (S.2 - P.2)^2 = 53 := by
  sorry

end NUMINAMATH_CALUDE_square_area_PS_l1728_172838


namespace NUMINAMATH_CALUDE_dacid_weighted_average_l1728_172841

/-- Calculates the weighted average grade given marks and weights -/
def weighted_average (marks : List ℝ) (weights : List ℝ) : ℝ :=
  (List.zip marks weights).map (fun (m, w) => m * w) |>.sum

/-- Theorem: Dacid's weighted average grade is 90.8 -/
theorem dacid_weighted_average :
  let marks := [96, 95, 82, 87, 92]
  let weights := [0.20, 0.25, 0.15, 0.25, 0.15]
  weighted_average marks weights = 90.8 := by
sorry

#eval weighted_average [96, 95, 82, 87, 92] [0.20, 0.25, 0.15, 0.25, 0.15]

end NUMINAMATH_CALUDE_dacid_weighted_average_l1728_172841


namespace NUMINAMATH_CALUDE_triangle_side_sum_l1728_172892

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  b * Real.cos C + c * Real.cos B = 3 * a * Real.cos B →
  b = 2 →
  (1 / 2) * a * c * Real.sin B = 3 * Real.sqrt 2 / 2 →
  a + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l1728_172892


namespace NUMINAMATH_CALUDE_first_number_in_proportion_l1728_172837

/-- Given a proportion a : 1.65 :: 5 : 11, prove that a = 0.75 -/
theorem first_number_in_proportion (a : ℝ) : 
  (a / 1.65 = 5 / 11) → a = 0.75 := by
sorry

end NUMINAMATH_CALUDE_first_number_in_proportion_l1728_172837


namespace NUMINAMATH_CALUDE_park_ant_count_l1728_172868

/-- Represents the dimensions and ant densities of a rectangular park with a special corner area -/
structure ParkInfo where
  width : ℝ  -- width of the park in feet
  length : ℝ  -- length of the park in feet
  normal_density : ℝ  -- average number of ants per square inch in most of the park
  corner_side : ℝ  -- side length of the square corner patch in feet
  corner_density : ℝ  -- average number of ants per square inch in the corner patch

/-- Calculates the total number of ants in the park -/
def totalAnts (park : ParkInfo) : ℝ :=
  let inches_per_foot : ℝ := 12
  let park_area := park.width * park.length * inches_per_foot^2
  let corner_area := park.corner_side^2 * inches_per_foot^2
  let normal_area := park_area - corner_area
  normal_area * park.normal_density + corner_area * park.corner_density

/-- Theorem stating that the total number of ants in the given park is approximately 73 million -/
theorem park_ant_count :
  let park : ParkInfo := {
    width := 200,
    length := 500,
    normal_density := 5,
    corner_side := 50,
    corner_density := 8
  }
  abs (totalAnts park - 73000000) < 100000 := by
  sorry


end NUMINAMATH_CALUDE_park_ant_count_l1728_172868


namespace NUMINAMATH_CALUDE_line_translation_l1728_172875

-- Define the original line
def original_line (x : ℝ) : ℝ := -2 * x + 3

-- Define the translation
def translation : ℝ := 2

-- Define the translated line
def translated_line (x : ℝ) : ℝ := -2 * x + 1

-- Theorem statement
theorem line_translation :
  ∀ x : ℝ, translated_line x = original_line x - translation := by
  sorry

end NUMINAMATH_CALUDE_line_translation_l1728_172875


namespace NUMINAMATH_CALUDE_sine_problem_l1728_172896

theorem sine_problem (α : ℝ) (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) :
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_problem_l1728_172896


namespace NUMINAMATH_CALUDE_correct_selection_schemes_l1728_172822

/-- Represents the number of translators for each language --/
structure TranslatorCounts where
  total : ℕ
  english : ℕ
  japanese : ℕ
  both : ℕ

/-- Represents the required team sizes --/
structure TeamSizes where
  english : ℕ
  japanese : ℕ

/-- Calculates the number of different selection schemes --/
def selectionSchemes (counts : TranslatorCounts) (sizes : TeamSizes) : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem correct_selection_schemes :
  let counts : TranslatorCounts := ⟨11, 5, 4, 2⟩
  let sizes : TeamSizes := ⟨4, 4⟩
  selectionSchemes counts sizes = 144 := by sorry

end NUMINAMATH_CALUDE_correct_selection_schemes_l1728_172822


namespace NUMINAMATH_CALUDE_solution_of_quadratic_equation_l1728_172882

theorem solution_of_quadratic_equation :
  {x : ℝ | 2 * (x + 1) = x * (x + 1)} = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_solution_of_quadratic_equation_l1728_172882


namespace NUMINAMATH_CALUDE_factors_of_M_l1728_172869

def M : ℕ := 2^5 * 3^4 * 5^3 * 7^2 * 11^1

theorem factors_of_M :
  (∃ (f : ℕ → ℕ), f M = 720 ∧ (∀ d : ℕ, d ∣ M ↔ d ∈ Finset.range (f M + 1))) ∧
  (∃ (g : ℕ → ℕ), g M = 120 ∧ (∀ d : ℕ, d ∣ M ∧ Odd d ↔ d ∈ Finset.range (g M + 1))) :=
by sorry

end NUMINAMATH_CALUDE_factors_of_M_l1728_172869


namespace NUMINAMATH_CALUDE_jackson_metropolitan_population_l1728_172817

theorem jackson_metropolitan_population :
  ∀ (average_population : ℝ),
  3200 ≤ average_population ∧ average_population ≤ 3600 →
  80000 ≤ 25 * average_population ∧ 25 * average_population ≤ 90000 :=
by sorry

end NUMINAMATH_CALUDE_jackson_metropolitan_population_l1728_172817


namespace NUMINAMATH_CALUDE_ice_cream_arrangement_count_l1728_172850

theorem ice_cream_arrangement_count : Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_arrangement_count_l1728_172850


namespace NUMINAMATH_CALUDE_exactly_two_clubs_l1728_172809

theorem exactly_two_clubs (S : ℕ) (A B C ABC : ℕ) : 
  S = 400 ∧
  A = S / 2 ∧
  B = S * 5 / 8 ∧
  C = S * 3 / 4 ∧
  ABC = S * 3 / 8 ∧
  A + B + C - 2 * ABC ≥ S →
  A + B + C - S - ABC = 500 := by
sorry

end NUMINAMATH_CALUDE_exactly_two_clubs_l1728_172809


namespace NUMINAMATH_CALUDE_max_tickets_purchasable_l1728_172803

theorem max_tickets_purchasable (ticket_price : ℚ) (budget : ℚ) : 
  ticket_price = 15.75 → budget = 200 → 
  ∃ n : ℕ, n * ticket_price ≤ budget ∧ 
           ∀ m : ℕ, m * ticket_price ≤ budget → m ≤ n ∧
           n = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_tickets_purchasable_l1728_172803


namespace NUMINAMATH_CALUDE_slower_pump_fill_time_l1728_172800

/-- Represents a water pump with a constant fill rate -/
structure Pump where
  rate : ℝ
  rate_positive : rate > 0

/-- Represents a swimming pool -/
structure Pool where
  volume : ℝ
  volume_positive : volume > 0

/-- Theorem stating the time taken by the slower pump to fill the pool alone -/
theorem slower_pump_fill_time (pool : Pool) (pump1 pump2 : Pump)
    (h1 : pump2.rate = 1.5 * pump1.rate)
    (h2 : (pump1.rate + pump2.rate) * 5 = pool.volume) :
    pump1.rate * 12.5 = pool.volume := by
  sorry

end NUMINAMATH_CALUDE_slower_pump_fill_time_l1728_172800


namespace NUMINAMATH_CALUDE_firewood_collection_l1728_172833

theorem firewood_collection (total kimberley ela : ℕ) (h1 : total = 35) (h2 : kimberley = 10) (h3 : ela = 13) :
  total - kimberley - ela = 12 := by
  sorry

end NUMINAMATH_CALUDE_firewood_collection_l1728_172833


namespace NUMINAMATH_CALUDE_range_of_m_l1728_172826

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) ↔ m ∈ Set.Icc (-3) 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1728_172826


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_expression_l1728_172887

theorem isosceles_right_triangle_expression (a : ℝ) (h : a > 0) :
  (a + 2 * a) / Real.sqrt (a^2 + a^2) = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_expression_l1728_172887


namespace NUMINAMATH_CALUDE_reciprocal_gp_sum_l1728_172861

/-- Given a geometric progression with n terms, first term 1, common ratio r^2 (r ≠ 0),
    and sum s^3, the sum of the geometric progression formed by the reciprocals of each term
    is s^3 / r^2 -/
theorem reciprocal_gp_sum (n : ℕ) (r s : ℝ) (hr : r ≠ 0) :
  let original_sum := (1 - r^(2*n)) / (1 - r^2)
  let reciprocal_sum := (1 - (1/r^2)^n) / (1 - 1/r^2)
  original_sum = s^3 → reciprocal_sum = s^3 / r^2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_gp_sum_l1728_172861


namespace NUMINAMATH_CALUDE_expo_volunteer_selection_l1728_172883

/-- The number of volunteers --/
def total_volunteers : ℕ := 5

/-- The number of volunteers to be selected --/
def selected_volunteers : ℕ := 4

/-- The number of tasks --/
def total_tasks : ℕ := 4

/-- The number of restricted tasks --/
def restricted_tasks : ℕ := 2

/-- The number of volunteers restricted to certain tasks --/
def restricted_volunteers : ℕ := 2

/-- The number of unrestricted volunteers --/
def unrestricted_volunteers : ℕ := total_volunteers - restricted_volunteers

theorem expo_volunteer_selection :
  (Nat.choose restricted_volunteers 1 * Nat.choose restricted_tasks 1 * (unrestricted_volunteers).factorial) +
  (Nat.choose restricted_volunteers 2 * (unrestricted_volunteers).factorial) = 36 := by
  sorry

end NUMINAMATH_CALUDE_expo_volunteer_selection_l1728_172883


namespace NUMINAMATH_CALUDE_prob_all_suits_in_five_draws_l1728_172855

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Number of cards drawn -/
def NumberOfDraws : ℕ := 5

/-- Probability of drawing a card from a specific suit -/
def ProbSingleSuit : ℚ := 1 / NumberOfSuits

/-- Theorem: Probability of getting at least one card from each suit in 5 draws with replacement -/
theorem prob_all_suits_in_five_draws : 
  let prob_different_suit (n : ℕ) := (NumberOfSuits - n) / NumberOfSuits
  (prob_different_suit 1) * (prob_different_suit 2) * (prob_different_suit 3) = 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_suits_in_five_draws_l1728_172855


namespace NUMINAMATH_CALUDE_sqrt_inequality_abc_inequality_l1728_172876

-- Problem 1
theorem sqrt_inequality : Real.sqrt 7 + Real.sqrt 13 < 3 + Real.sqrt 11 := by sorry

-- Problem 2
theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * (b^2 + c^2) + b * (c^2 + a^2) + c * (a^2 + b^2) ≥ 6 * a * b * c := by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_abc_inequality_l1728_172876


namespace NUMINAMATH_CALUDE_lottery_problem_l1728_172893

/-- Represents a lottery with prizes and blanks. -/
structure Lottery where
  prizes : ℕ
  blanks : ℕ
  prob_win : ℝ
  h_prob : prob_win = prizes / (prizes + blanks : ℝ)

/-- The lottery problem statement. -/
theorem lottery_problem (L : Lottery)
  (h_prizes : L.prizes = 10)
  (h_prob : L.prob_win = 0.2857142857142857) :
  L.blanks = 25 := by
  sorry

#check lottery_problem

end NUMINAMATH_CALUDE_lottery_problem_l1728_172893


namespace NUMINAMATH_CALUDE_carl_driving_hours_l1728_172819

/-- Calculates the total driving hours for Carl over two weeks -/
def total_driving_hours : ℕ :=
  let daily_hours : ℕ := 2
  let days_in_two_weeks : ℕ := 14
  let additional_weekly_hours : ℕ := 6
  let weeks : ℕ := 2
  (daily_hours * days_in_two_weeks) + (additional_weekly_hours * weeks)

/-- Theorem stating that Carl's total driving hours over two weeks is 40 -/
theorem carl_driving_hours : total_driving_hours = 40 := by
  sorry

end NUMINAMATH_CALUDE_carl_driving_hours_l1728_172819


namespace NUMINAMATH_CALUDE_trig_identity_l1728_172884

theorem trig_identity : 
  Real.sin (71 * π / 180) * Real.cos (26 * π / 180) - 
  Real.sin (19 * π / 180) * Real.sin (26 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_trig_identity_l1728_172884


namespace NUMINAMATH_CALUDE_quarters_indeterminate_l1728_172818

/-- Represents the number of coins Mike has --/
structure MikeCoins where
  quarters : ℕ
  nickels : ℕ

/-- Represents the state of Mike's coins before and after his dad's borrowing --/
structure CoinState where
  initial : MikeCoins
  borrowed_nickels : ℕ
  current : MikeCoins

/-- Theorem stating that the number of quarters cannot be uniquely determined --/
theorem quarters_indeterminate (state : CoinState) 
    (h1 : state.initial.nickels = 87)
    (h2 : state.borrowed_nickels = 75)
    (h3 : state.current.nickels = 12)
    (h4 : state.initial.nickels = state.borrowed_nickels + state.current.nickels) :
    ∀ q : ℕ, ∃ state' : CoinState, 
      state'.initial.nickels = state.initial.nickels ∧
      state'.borrowed_nickels = state.borrowed_nickels ∧
      state'.current.nickels = state.current.nickels ∧
      state'.initial.quarters = q :=
  sorry

end NUMINAMATH_CALUDE_quarters_indeterminate_l1728_172818


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1728_172835

theorem min_value_quadratic (x : ℝ) : 
  ∃ (min_z : ℝ), min_z = 5 ∧ ∀ z : ℝ, z = 5*x^2 + 20*x + 25 → z ≥ min_z := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1728_172835


namespace NUMINAMATH_CALUDE_revenue_change_specific_l1728_172801

/-- Calculates the change in revenue given price increase, sales decrease, loyalty discount, and sales tax -/
def revenueChange (priceIncrease salesDecrease loyaltyDiscount salesTax : ℝ) : ℝ :=
  let newPrice := 1 + priceIncrease
  let newSales := 1 - salesDecrease
  let discountedPrice := newPrice * (1 - loyaltyDiscount)
  let finalPrice := discountedPrice * (1 + salesTax)
  finalPrice * newSales - 1

/-- The revenue change given specific conditions -/
theorem revenue_change_specific : 
  revenueChange 0.9 0.3 0.1 0.15 = 0.37655 := by sorry

end NUMINAMATH_CALUDE_revenue_change_specific_l1728_172801


namespace NUMINAMATH_CALUDE_shorter_wall_area_l1728_172804

/-- Given a rectangular hall with specified dimensions, calculate the area of the shorter wall. -/
theorem shorter_wall_area (floor_area : ℝ) (longer_wall_area : ℝ) (height : ℝ) :
  floor_area = 20 →
  longer_wall_area = 10 →
  height = 40 →
  let length := longer_wall_area / height
  let width := floor_area / length
  width * height = 3200 := by
  sorry

#check shorter_wall_area

end NUMINAMATH_CALUDE_shorter_wall_area_l1728_172804


namespace NUMINAMATH_CALUDE_circle_m_range_l1728_172805

-- Define the equation of the circle
def circle_equation (x y m : ℝ) : Prop := x^2 + y^2 - x + y + m = 0

-- Define the condition for the equation to represent a circle
def is_circle (m : ℝ) : Prop := ∃ (x y : ℝ), circle_equation x y m

-- Theorem stating the range of m for which the equation represents a circle
theorem circle_m_range :
  ∀ m : ℝ, is_circle m ↔ m < (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_circle_m_range_l1728_172805


namespace NUMINAMATH_CALUDE_square_of_97_l1728_172839

theorem square_of_97 : 97 * 97 = 9409 := by
  sorry

end NUMINAMATH_CALUDE_square_of_97_l1728_172839


namespace NUMINAMATH_CALUDE_correct_additional_money_l1728_172870

/-- Calculates the additional money Jack needs to buy socks and shoes -/
def additional_money_needed (sock_price : ℝ) (shoe_price : ℝ) (jack_has : ℝ) : ℝ :=
  2 * sock_price + shoe_price - jack_has

/-- Proves that the additional money needed is correct -/
theorem correct_additional_money 
  (sock_price : ℝ) (shoe_price : ℝ) (jack_has : ℝ) 
  (h1 : sock_price = 9.5)
  (h2 : shoe_price = 92)
  (h3 : jack_has = 40) :
  additional_money_needed sock_price shoe_price jack_has = 71 :=
by sorry

end NUMINAMATH_CALUDE_correct_additional_money_l1728_172870


namespace NUMINAMATH_CALUDE_lydia_planting_age_l1728_172863

/-- Represents the age of Lydia when she planted the tree -/
def planting_age : ℕ := sorry

/-- The time it takes for an apple tree to bear fruit -/
def fruit_bearing_time : ℕ := 7

/-- Lydia's current age -/
def current_age : ℕ := 9

/-- Lydia's age when she first eats an apple from her tree -/
def first_apple_age : ℕ := 11

theorem lydia_planting_age : 
  planting_age = first_apple_age - fruit_bearing_time := by sorry

end NUMINAMATH_CALUDE_lydia_planting_age_l1728_172863


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l1728_172880

theorem pizza_toppings_combinations (n : ℕ) (h : n = 8) : 
  n + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l1728_172880


namespace NUMINAMATH_CALUDE_sequences_and_sum_theorem_l1728_172821

/-- Definition of sequence a_n -/
def a (n : ℕ+) : ℕ :=
  if n = 1 then 1 else 2 * n.val - 1

/-- Definition of sequence b_n -/
def b (n : ℕ+) : ℚ :=
  if n = 1 then 1 else 2^(2 - n.val)

/-- Definition of S_n (sum of first n terms of a_n) -/
def S (n : ℕ+) : ℕ := (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- Definition of T_n (sum of first n terms of a_n * b_n) -/
def T (n : ℕ+) : ℚ := 11 - (2 * n.val + 3) * 2^(2 - n.val)

theorem sequences_and_sum_theorem (n : ℕ+) :
  (∀ (k : ℕ+), k ≥ 2 → S (k + 1) + S (k - 1) = 2 * (S k + 1)) ∧
  (∀ (k : ℕ+), (Finset.range k.val).sum (λ i => 2^i * b ⟨i + 1, Nat.succ_pos i⟩) = a k) →
  (∀ (k : ℕ+), a k = 2 * k.val - 1) ∧
  (∀ (k : ℕ+), b k = if k = 1 then 1 else 2^(2 - k.val)) ∧
  (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩ * b ⟨i + 1, Nat.succ_pos i⟩) = T n :=
by sorry


end NUMINAMATH_CALUDE_sequences_and_sum_theorem_l1728_172821
