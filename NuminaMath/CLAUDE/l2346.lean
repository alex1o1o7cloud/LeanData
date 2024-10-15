import Mathlib

namespace NUMINAMATH_CALUDE_abs_a_minus_3_l2346_234611

theorem abs_a_minus_3 (a : ℝ) (h : ∀ x : ℝ, (a - 2) * x > a - 2 ↔ x < 1) : 
  |a - 3| = 3 - a := by
  sorry

end NUMINAMATH_CALUDE_abs_a_minus_3_l2346_234611


namespace NUMINAMATH_CALUDE_range_a_all_real_range_a_interval_l2346_234676

/-- The function f(x) = x^2 + ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

/-- Theorem for the range of 'a' when f(x) ≥ a for all real x -/
theorem range_a_all_real (a : ℝ) :
  (∀ x : ℝ, f a x ≥ a) ↔ a ≤ 3 :=
sorry

/-- Theorem for the range of 'a' when f(x) ≥ a for x in [-2, 2] -/
theorem range_a_interval (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f a x ≥ a) ↔ a ∈ Set.Icc (-6) 2 :=
sorry

end NUMINAMATH_CALUDE_range_a_all_real_range_a_interval_l2346_234676


namespace NUMINAMATH_CALUDE_calculation_proof_l2346_234628

theorem calculation_proof : 
  Real.sqrt 27 - |2 * Real.sqrt 3 - 9 * Real.tan (30 * π / 180)| + (1/2)⁻¹ - (1 - π)^0 = 2 * Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2346_234628


namespace NUMINAMATH_CALUDE_complex_sum_zero_l2346_234687

theorem complex_sum_zero : (1 - Complex.I) ^ 10 + (1 + Complex.I) ^ 10 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l2346_234687


namespace NUMINAMATH_CALUDE_strip_arrangement_area_l2346_234692

/-- Represents a rectangular paper strip -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℝ := s.length * s.width

/-- Calculates the overlap area between two perpendicular strips -/
def overlapArea (s1 s2 : Strip) : ℝ := s1.width * s2.width

/-- Represents the arrangement of strips on the table -/
structure StripArrangement where
  horizontalStrips : Fin 2 → Strip
  verticalStrips : Fin 2 → Strip

/-- Calculates the total area covered by the strips -/
def totalCoveredArea (arrangement : StripArrangement) : ℝ :=
  let totalStripArea := (Finset.sum (Finset.range 2) (λ i => stripArea (arrangement.horizontalStrips i))) +
                        (Finset.sum (Finset.range 2) (λ i => stripArea (arrangement.verticalStrips i)))
  let totalOverlapArea := Finset.sum (Finset.range 2) (λ i =>
                            Finset.sum (Finset.range 2) (λ j =>
                              overlapArea (arrangement.horizontalStrips i) (arrangement.verticalStrips j)))
  totalStripArea - totalOverlapArea

theorem strip_arrangement_area :
  ∀ (arrangement : StripArrangement),
    (∀ i : Fin 2, arrangement.horizontalStrips i = ⟨8, 1⟩) →
    (∀ i : Fin 2, arrangement.verticalStrips i = ⟨8, 1⟩) →
    totalCoveredArea arrangement = 28 := by
  sorry

end NUMINAMATH_CALUDE_strip_arrangement_area_l2346_234692


namespace NUMINAMATH_CALUDE_set_equality_l2346_234663

-- Define the universal set U as ℝ
def U := ℝ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l2346_234663


namespace NUMINAMATH_CALUDE_min_value_cube_root_plus_inverse_square_l2346_234666

theorem min_value_cube_root_plus_inverse_square (x : ℝ) (h : x > 0) :
  3 * x^(1/3) + 1 / x^2 ≥ 4 ∧
  (3 * x^(1/3) + 1 / x^2 = 4 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cube_root_plus_inverse_square_l2346_234666


namespace NUMINAMATH_CALUDE_divisor_problem_l2346_234607

theorem divisor_problem (d : ℕ) (h : d > 0) :
  (∃ n : ℤ, n % d = 3 ∧ (2 * n) % d = 2) → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2346_234607


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_l2346_234645

theorem largest_divisor_of_m (m : ℕ+) (h : 54 ∣ m ^ 2) :
  ∃ (d : ℕ), d ∣ m ∧ d = 18 ∧ ∀ (k : ℕ), k ∣ m → k ≤ d :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_l2346_234645


namespace NUMINAMATH_CALUDE_equation_solution_l2346_234691

theorem equation_solution : ∃! x : ℝ, 3 * x + 1 = x - 3 :=
  by
    use -2
    constructor
    · -- Prove that -2 satisfies the equation
      sorry
    · -- Prove uniqueness
      sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2346_234691


namespace NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l2346_234693

theorem cos_squared_alpha_minus_pi_fourth (α : ℝ) 
  (h : Real.sin (2 * α) = 1 / 3) : 
  Real.cos (α - π / 4) ^ 2 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l2346_234693


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2346_234667

/-- Simple interest calculation -/
theorem simple_interest_principal (interest : ℚ) (time : ℚ) (rate : ℚ) (principal : ℚ) :
  interest = principal * rate * time ∧
  interest = 10.92 ∧
  time = 6 ∧
  rate = 7 / 100 / 12 →
  principal = 26 := by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2346_234667


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l2346_234678

theorem inequality_holds_iff_a_in_range (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π/2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l2346_234678


namespace NUMINAMATH_CALUDE_union_complement_problem_l2346_234638

universe u

def I : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem union_complement_problem : A ∪ (I \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_problem_l2346_234638


namespace NUMINAMATH_CALUDE_power_zero_l2346_234698

theorem power_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_l2346_234698


namespace NUMINAMATH_CALUDE_expression_simplification_l2346_234662

theorem expression_simplification :
  1 / ((1 / (Real.sqrt 2 + 1)) + (2 / (Real.sqrt 3 - 1))) = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2346_234662


namespace NUMINAMATH_CALUDE_r_share_l2346_234610

/-- Given a total amount divided among three people P, Q, and R, with specified ratios,
    calculate R's share. -/
theorem r_share (total : ℕ) (p q r : ℕ) : 
  total = 1210 →
  5 * q = 4 * p →
  9 * r = 10 * q →
  r = 400 := by
  sorry


end NUMINAMATH_CALUDE_r_share_l2346_234610


namespace NUMINAMATH_CALUDE_container_capacity_l2346_234651

theorem container_capacity (initial_fill : Real) (added_water : Real) (final_fill : Real) :
  initial_fill = 0.3 →
  added_water = 45 →
  final_fill = 0.75 →
  ∃ (capacity : Real), capacity = 100 ∧
    final_fill * capacity = initial_fill * capacity + added_water :=
by sorry

end NUMINAMATH_CALUDE_container_capacity_l2346_234651


namespace NUMINAMATH_CALUDE_construct_quadrilateral_l2346_234647

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Checks if three sides of a quadrilateral are equal -/
def hasThreeEqualSides (q : Quadrilateral) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (M : Point) (A B : Point) : Prop := sorry

/-- Theorem: Given three points that are midpoints of three equal sides of a convex quadrilateral,
    a unique quadrilateral can be constructed -/
theorem construct_quadrilateral 
  (P Q R : Point) 
  (h_exists : ∃ (q : Quadrilateral), 
    isConvex q ∧ 
    hasThreeEqualSides q ∧
    isMidpoint P q.A q.B ∧
    isMidpoint Q q.B q.C ∧
    isMidpoint R q.C q.D) :
  ∃! (q : Quadrilateral), 
    isConvex q ∧ 
    hasThreeEqualSides q ∧
    isMidpoint P q.A q.B ∧
    isMidpoint Q q.B q.C ∧
    isMidpoint R q.C q.D :=
sorry

end NUMINAMATH_CALUDE_construct_quadrilateral_l2346_234647


namespace NUMINAMATH_CALUDE_emilys_skirt_cost_l2346_234680

theorem emilys_skirt_cost (art_supplies_cost shoes_original_price total_spent : ℝ)
  (skirt_count : ℕ) (shoe_discount_rate : ℝ) :
  art_supplies_cost = 20 →
  skirt_count = 2 →
  shoes_original_price = 30 →
  shoe_discount_rate = 0.15 →
  total_spent = 50 →
  let shoes_discounted_price := shoes_original_price * (1 - shoe_discount_rate)
  let skirts_total_cost := total_spent - art_supplies_cost - shoes_discounted_price
  let skirt_cost := skirts_total_cost / skirt_count
  skirt_cost = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_emilys_skirt_cost_l2346_234680


namespace NUMINAMATH_CALUDE_pens_purchased_l2346_234699

theorem pens_purchased (total_cost : ℝ) (num_pencils : ℕ) (pencil_price : ℝ) (pen_price : ℝ)
  (h1 : total_cost = 570)
  (h2 : num_pencils = 75)
  (h3 : pencil_price = 2)
  (h4 : pen_price = 14) :
  (total_cost - num_pencils * pencil_price) / pen_price = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_pens_purchased_l2346_234699


namespace NUMINAMATH_CALUDE_line_contains_point_l2346_234635

/-- The value of k for which the line 1 - 3kx + y = 7y contains the point (-1/3, -2) -/
theorem line_contains_point (k : ℝ) : 
  (1 - 3 * k * (-1/3) + (-2) = 7 * (-2)) ↔ k = -13 := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l2346_234635


namespace NUMINAMATH_CALUDE_min_value_inverse_sum_l2346_234614

theorem min_value_inverse_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) :
  1 / a + 3 / b ≥ 16 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ 1 / a + 3 / b = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inverse_sum_l2346_234614


namespace NUMINAMATH_CALUDE_old_clock_slow_l2346_234620

/-- Represents the number of minutes between hand overlaps on the old clock -/
def overlap_interval : ℕ := 66

/-- Represents the number of minutes in a standard day -/
def standard_day_minutes : ℕ := 24 * 60

/-- Represents the number of hand overlaps in a standard day -/
def overlaps_per_day : ℕ := 22

theorem old_clock_slow (old_clock_day : ℕ) 
  (h1 : old_clock_day = overlap_interval * overlaps_per_day) : 
  old_clock_day - standard_day_minutes = 12 := by
  sorry

end NUMINAMATH_CALUDE_old_clock_slow_l2346_234620


namespace NUMINAMATH_CALUDE_original_numbers_proof_l2346_234689

theorem original_numbers_proof (x y : ℤ) 
  (sum_condition : x + y = 2022)
  (modified_sum_condition : (x - 5) / 10 + 10 * y + 1 = 2252) :
  x = 1815 ∧ y = 207 := by
  sorry

end NUMINAMATH_CALUDE_original_numbers_proof_l2346_234689


namespace NUMINAMATH_CALUDE_power_equation_solution_l2346_234616

theorem power_equation_solution (m : ℕ) : 8^36 * 6^21 = 3 * 24^m → m = 43 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2346_234616


namespace NUMINAMATH_CALUDE_sample_size_calculation_l2346_234670

/-- Represents the sample size calculation for three communities --/
theorem sample_size_calculation 
  (pop_A pop_B pop_C : ℕ) 
  (sample_C : ℕ) 
  (h1 : pop_A = 600) 
  (h2 : pop_B = 1200) 
  (h3 : pop_C = 1500) 
  (h4 : sample_C = 15) : 
  ∃ n : ℕ, n * pop_C = sample_C * (pop_A + pop_B + pop_C) ∧ n = 33 :=
sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l2346_234670


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2346_234621

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 / x) + (1 / y) = 1) : x + 2*y ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2346_234621


namespace NUMINAMATH_CALUDE_divisors_not_div_by_seven_l2346_234686

def number_to_factorize : ℕ := 420

-- Define a function to count divisors not divisible by 7
def count_divisors_not_div_by_seven (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem divisors_not_div_by_seven :
  count_divisors_not_div_by_seven number_to_factorize = 12 := by sorry

end NUMINAMATH_CALUDE_divisors_not_div_by_seven_l2346_234686


namespace NUMINAMATH_CALUDE_inequality_solution_range_function_minimum_value_l2346_234682

-- Part 1
theorem inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| ≥ a) → a ≤ 3 := by sorry

-- Part 2
theorem function_minimum_value (a : ℝ) :
  (∃ m : ℝ, m = 5 ∧ ∀ x : ℝ, |x + 1| + 2 * |x - a| ≥ m) →
  a = 4 ∨ a = -6 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_function_minimum_value_l2346_234682


namespace NUMINAMATH_CALUDE_shelf_filling_theorem_l2346_234627

theorem shelf_filling_theorem (A H S M E : ℕ) 
  (distinct : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ E ∧ 
              H ≠ S ∧ H ≠ M ∧ H ≠ E ∧ 
              S ≠ M ∧ S ≠ E ∧ 
              M ≠ E)
  (positive : A > 0 ∧ H > 0 ∧ S > 0 ∧ M > 0 ∧ E > 0)
  (thicker : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y > x ∧ 
             A * x + H * y = S * x + M * y ∧ 
             A * x + H * y = E * x) : 
  E = (A * M - S * H) / (M - H) :=
by sorry

end NUMINAMATH_CALUDE_shelf_filling_theorem_l2346_234627


namespace NUMINAMATH_CALUDE_project_completion_time_l2346_234661

theorem project_completion_time
  (a b c d e : ℝ)
  (h1 : 1/a + 1/b + 1/c + 1/d = 1/6)
  (h2 : 1/b + 1/c + 1/d + 1/e = 1/8)
  (h3 : 1/a + 1/e = 1/12)
  : e = 48 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l2346_234661


namespace NUMINAMATH_CALUDE_ninety_seventh_rising_number_l2346_234685

/-- A rising number is a positive integer where each digit is larger than each of the digits to its left. -/
def IsRisingNumber (n : ℕ) : Prop := sorry

/-- The total count of five-digit rising numbers. -/
def TotalFiveDigitRisingNumbers : ℕ := 126

/-- The nth five-digit rising number when arranged from smallest to largest. -/
def NthFiveDigitRisingNumber (n : ℕ) : ℕ := sorry

theorem ninety_seventh_rising_number :
  NthFiveDigitRisingNumber 97 = 24678 := by sorry

end NUMINAMATH_CALUDE_ninety_seventh_rising_number_l2346_234685


namespace NUMINAMATH_CALUDE_EF_equals_5_sqrt_35_div_3_l2346_234644

/-- A rectangle ABCD with a point E inside -/
structure Rectangle :=
  (A B C D E : ℝ × ℝ)
  (is_rectangle : A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ C.2 = D.2)
  (AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 30)
  (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 40)
  (E_inside : E.1 > A.1 ∧ E.1 < C.1 ∧ E.2 > A.2 ∧ E.2 < C.2)
  (EA_length : Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 10)
  (EB_length : Real.sqrt ((E.1 - B.1)^2 + (E.2 - B.2)^2) = 30)

/-- The length of EF, where F is the foot of the perpendicular from E to AD -/
def EF_length (r : Rectangle) : ℝ := r.E.2 - r.A.2

theorem EF_equals_5_sqrt_35_div_3 (r : Rectangle) : 
  EF_length r = 5 * Real.sqrt 35 / 3 :=
sorry

end NUMINAMATH_CALUDE_EF_equals_5_sqrt_35_div_3_l2346_234644


namespace NUMINAMATH_CALUDE_derivative_at_one_l2346_234633

theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = x * Real.exp x) : 
  deriv f 1 = 2 * Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2346_234633


namespace NUMINAMATH_CALUDE_exists_fifteen_classmates_l2346_234660

/-- A type representing students. -/
def Student : Type := ℕ

/-- The total number of students. -/
def total_students : ℕ := 60

/-- A function that returns true if the given students are classmates. -/
def are_classmates : List Student → Prop := sorry

/-- The property that among any 10 students, there are always 3 classmates. -/
axiom three_classmates_in_ten : 
  ∀ (s : Finset Student), s.card = 10 → ∃ (t : Finset Student), t ⊆ s ∧ t.card = 3 ∧ are_classmates t.toList

/-- The theorem to be proved. -/
theorem exists_fifteen_classmates :
  ∃ (s : Finset Student), s.card ≥ 15 ∧ are_classmates s.toList :=
sorry

end NUMINAMATH_CALUDE_exists_fifteen_classmates_l2346_234660


namespace NUMINAMATH_CALUDE_star_op_and_comparison_l2346_234631

-- Define the * operation for non-zero integers
def star_op (a b : ℤ) : ℚ := (a⁻¹ : ℚ) + (b⁻¹ : ℚ)

-- Theorem statement
theorem star_op_and_comparison 
  (a b : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : a + b = 10) 
  (h4 : a * b = 24) : 
  star_op a b = 5 / 12 ∧ a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_star_op_and_comparison_l2346_234631


namespace NUMINAMATH_CALUDE_cubic_fraction_equality_l2346_234632

theorem cubic_fraction_equality : 
  let a : ℝ := 5
  let b : ℝ := 4
  (a^3 + b^3) / (a^2 - a*b + b^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equality_l2346_234632


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l2346_234615

/-- A positive geometric sequence satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a 1 / a 0
  condition : a 8 = a 6 + 2 * a 4

/-- The theorem statement -/
theorem geometric_sequence_minimum (seq : GeometricSequence) :
  (∃ m n : ℕ, Real.sqrt (seq.a m * seq.a n) = Real.sqrt 2 * seq.a 1) →
  (∀ m n : ℕ, 1 / m + 9 / n ≥ 4) ∧
  (∃ m n : ℕ, 1 / m + 9 / n = 4) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l2346_234615


namespace NUMINAMATH_CALUDE_two_flies_problem_l2346_234659

/-- Two flies crawling on a wall problem -/
theorem two_flies_problem (d v : ℝ) (h1 : d > 0) (h2 : v > 0) :
  let t1 := 2 * d / v
  let t2 := 5 * d / (2 * v)
  let avg_speed1 := 2 * d / t1
  let avg_speed2 := 2 * d / t2
  t1 < t2 ∧ avg_speed1 > avg_speed2 := by
  sorry

#check two_flies_problem

end NUMINAMATH_CALUDE_two_flies_problem_l2346_234659


namespace NUMINAMATH_CALUDE_system_solution_l2346_234684

theorem system_solution (x y : ℝ) : 
  (x^2 + x*y + y^2) / (x^2 - x*y + y^2) = 3 →
  x^3 + y^3 = 2 →
  x = 1 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2346_234684


namespace NUMINAMATH_CALUDE_equal_solutions_iff_n_eq_neg_one_third_l2346_234656

theorem equal_solutions_iff_n_eq_neg_one_third 
  (x y n : ℝ) : 
  (2 * x - 5 * y = 3 * n + 7 ∧ x - 3 * y = 4) → 
  (∃! (x y : ℝ), 2 * x - 5 * y = 3 * n + 7 ∧ x - 3 * y = 4) ↔ 
  n = -1/3 := by
sorry

end NUMINAMATH_CALUDE_equal_solutions_iff_n_eq_neg_one_third_l2346_234656


namespace NUMINAMATH_CALUDE_driveway_wheel_count_inconsistent_l2346_234669

/-- Represents the number of wheels on various vehicles and items in Jordan's driveway --/
structure DrivewayCounts where
  carCount : ℕ
  bikeCount : ℕ
  trashCanCount : ℕ
  tricycleCount : ℕ
  rollerSkatesPairCount : ℕ

/-- Calculates the total number of wheels based on the counts of vehicles and items --/
def totalWheels (counts : DrivewayCounts) : ℕ :=
  4 * counts.carCount +
  2 * counts.bikeCount +
  2 * counts.trashCanCount +
  3 * counts.tricycleCount +
  4 * counts.rollerSkatesPairCount

/-- Theorem stating that given the conditions, it's impossible to have 25 wheels in total --/
theorem driveway_wheel_count_inconsistent :
  ∀ (counts : DrivewayCounts),
    counts.carCount = 2 ∧
    counts.bikeCount = 2 ∧
    counts.trashCanCount = 1 ∧
    counts.tricycleCount = 1 ∧
    counts.rollerSkatesPairCount = 1 →
    totalWheels counts ≠ 25 :=
by
  sorry

end NUMINAMATH_CALUDE_driveway_wheel_count_inconsistent_l2346_234669


namespace NUMINAMATH_CALUDE_pentagon_y_coordinate_l2346_234641

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point

/-- Check if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop :=
  p.A.x = 0 ∧ p.B.x = 0 ∧ p.D.x = p.E.x ∧ p.C.x = (p.D.x / 2)

/-- Calculate the area of a pentagon -/
noncomputable def pentagonArea (p : Pentagon) : ℝ :=
  sorry -- Actual implementation would go here

theorem pentagon_y_coordinate (p : Pentagon) :
  p.A = ⟨0, 0⟩ →
  p.B = ⟨0, 5⟩ →
  p.D = ⟨6, 5⟩ →
  p.E = ⟨6, 0⟩ →
  p.C.x = 3 →
  hasVerticalSymmetry p →
  pentagonArea p = 50 →
  p.C.y = 35/3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_y_coordinate_l2346_234641


namespace NUMINAMATH_CALUDE_john_total_running_distance_l2346_234696

/-- The number of days from Monday to Saturday, inclusive -/
def days_ran : ℕ := 6

/-- The distance John ran each day in meters -/
def daily_distance : ℕ := 1700

/-- The total distance John ran before getting injured -/
def total_distance : ℕ := days_ran * daily_distance

/-- Theorem stating that the total distance John ran is 10200 meters -/
theorem john_total_running_distance :
  total_distance = 10200 := by sorry

end NUMINAMATH_CALUDE_john_total_running_distance_l2346_234696


namespace NUMINAMATH_CALUDE_modulus_of_complex_power_l2346_234606

theorem modulus_of_complex_power : Complex.abs ((2 : ℂ) + Complex.I) ^ 8 = 625 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_power_l2346_234606


namespace NUMINAMATH_CALUDE_solution_set_ax_gt_b_l2346_234619

theorem solution_set_ax_gt_b (a b : ℝ) :
  let S := {x : ℝ | a * x > b}
  (a > 0 → S = {x : ℝ | x > b / a}) ∧
  (a < 0 → S = {x : ℝ | x < b / a}) ∧
  (a = 0 ∧ b ≥ 0 → S = ∅) ∧
  (a = 0 ∧ b < 0 → S = Set.univ) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_ax_gt_b_l2346_234619


namespace NUMINAMATH_CALUDE_equation_two_complex_roots_l2346_234625

/-- The equation under consideration -/
def equation (x k : ℂ) : Prop :=
  x / (x + 2) + x / (x + 3) = k * x

/-- The equation has exactly two complex roots -/
def has_two_complex_roots (k : ℂ) : Prop :=
  ∃! (r₁ r₂ : ℂ), r₁ ≠ r₂ ∧ ∀ x, equation x k ↔ x = 0 ∨ x = r₁ ∨ x = r₂

/-- The main theorem stating the condition for the equation to have exactly two complex roots -/
theorem equation_two_complex_roots :
  ∀ k : ℂ, has_two_complex_roots k ↔ k = 2*I ∨ k = -2*I :=
sorry

end NUMINAMATH_CALUDE_equation_two_complex_roots_l2346_234625


namespace NUMINAMATH_CALUDE_allan_bought_two_balloons_l2346_234654

/-- The number of balloons Allan bought at the park -/
def balloons_bought (allan_initial jake_brought total : ℕ) : ℕ :=
  total - (allan_initial + jake_brought)

/-- Theorem: Allan bought 2 balloons at the park -/
theorem allan_bought_two_balloons : balloons_bought 3 5 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_allan_bought_two_balloons_l2346_234654


namespace NUMINAMATH_CALUDE_shanghai_population_equality_l2346_234697

/-- The population of Shanghai in millions -/
def shanghai_population : ℝ := 16.3

/-- Scientific notation representation of Shanghai's population -/
def shanghai_population_scientific : ℝ := 1.63 * 10^7

/-- Theorem stating that the population of Shanghai expressed in millions 
    is equal to its representation in scientific notation -/
theorem shanghai_population_equality : 
  shanghai_population * 10^6 = shanghai_population_scientific := by
  sorry

end NUMINAMATH_CALUDE_shanghai_population_equality_l2346_234697


namespace NUMINAMATH_CALUDE_shoes_cost_is_74_l2346_234652

-- Define the discount rate
def discount_rate : ℚ := 0.1

-- Define the cost of socks and bag
def socks_cost : ℚ := 2 * 2
def bag_cost : ℚ := 42

-- Define the discount threshold
def discount_threshold : ℚ := 100

-- Define the final payment amount
def final_payment : ℚ := 118

-- Theorem to prove
theorem shoes_cost_is_74 :
  ∃ (shoes_cost : ℚ),
    let total_cost := shoes_cost + socks_cost + bag_cost
    let discount := max (discount_rate * (total_cost - discount_threshold)) 0
    total_cost - discount = final_payment ∧ shoes_cost = 74 := by
  sorry

end NUMINAMATH_CALUDE_shoes_cost_is_74_l2346_234652


namespace NUMINAMATH_CALUDE_no_solution_exists_l2346_234623

theorem no_solution_exists (k : ℕ) (hk : k > 1) : ¬ ∃ n : ℕ+, ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → n / i = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2346_234623


namespace NUMINAMATH_CALUDE_interest_years_calculation_l2346_234664

/-- Given simple interest, compound interest, and interest rate, calculate the number of years -/
theorem interest_years_calculation (simple_interest compound_interest : ℝ) (rate : ℝ) 
  (h1 : simple_interest = 600)
  (h2 : compound_interest = 609)
  (h3 : rate = 0.03)
  (h4 : simple_interest = rate * (compound_interest / (rate * ((1 + rate)^2 - 1))))
  (h5 : compound_interest = (simple_interest / (rate * 2)) * ((1 + rate)^2 - 1)) :
  ∃ (n : ℕ), n = 2 ∧ 
    simple_interest = (compound_interest / ((1 + rate)^n - 1)) * rate * n ∧
    compound_interest = (simple_interest / (rate * n)) * ((1 + rate)^n - 1) :=
sorry

end NUMINAMATH_CALUDE_interest_years_calculation_l2346_234664


namespace NUMINAMATH_CALUDE_sharp_nested_30_l2346_234640

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem sharp_nested_30 : sharp (sharp (sharp (sharp 30))) = 8.24 := by sorry

end NUMINAMATH_CALUDE_sharp_nested_30_l2346_234640


namespace NUMINAMATH_CALUDE_composition_injective_implies_first_injective_l2346_234655

theorem composition_injective_implies_first_injective
  (f g : ℝ → ℝ) (h : Function.Injective (g ∘ f)) :
  Function.Injective f := by
  sorry

end NUMINAMATH_CALUDE_composition_injective_implies_first_injective_l2346_234655


namespace NUMINAMATH_CALUDE_equation_solution_l2346_234639

theorem equation_solution (x : ℝ) : 
  (8 / (Real.sqrt (x - 10) - 10) + 2 / (Real.sqrt (x - 10) - 5) + 
   10 / (Real.sqrt (x - 10) + 5) + 16 / (Real.sqrt (x - 10) + 10) = 0) ↔ 
  x = 60 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2346_234639


namespace NUMINAMATH_CALUDE_man_son_age_difference_l2346_234646

/-- Given a man and his son, where the son's present age is 16, and in two years
    the man's age will be twice the age of his son, prove that the man is 18 years
    older than his son. -/
theorem man_son_age_difference (man_age son_age : ℕ) : 
  son_age = 16 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_man_son_age_difference_l2346_234646


namespace NUMINAMATH_CALUDE_odd_function_increasing_function_symmetry_more_than_two_roots_l2346_234618

-- Define the function f
def f (b c x : ℝ) : ℝ := x * abs x + b * x + c

-- Theorem statements
theorem odd_function (b : ℝ) :
  ∀ x, f b 0 x = -f b 0 (-x) := by sorry

theorem increasing_function (c : ℝ) :
  ∀ x y, x < y → f 0 c x < f 0 c y := by sorry

theorem symmetry (b c : ℝ) :
  ∀ x, f b c x - c = -(f b c (-x) - c) := by sorry

theorem more_than_two_roots :
  ∃ b c : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  f b c x₁ = 0 ∧ f b c x₂ = 0 ∧ f b c x₃ = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_increasing_function_symmetry_more_than_two_roots_l2346_234618


namespace NUMINAMATH_CALUDE_xiaobing_jumps_189_ropes_per_minute_l2346_234602

/-- The number of ropes Xiaohan jumps per minute -/
def xiaohan_ropes_per_minute : ℕ := 168

/-- The number of ropes Xiaobing jumps per minute -/
def xiaobing_ropes_per_minute : ℕ := xiaohan_ropes_per_minute + 21

/-- The number of ropes Xiaobing jumps in the given time -/
def xiaobing_ropes : ℕ := 135

/-- The number of ropes Xiaohan jumps in the given time -/
def xiaohan_ropes : ℕ := 120

theorem xiaobing_jumps_189_ropes_per_minute :
  (xiaobing_ropes : ℚ) / xiaobing_ropes_per_minute = (xiaohan_ropes : ℚ) / xiaohan_ropes_per_minute →
  xiaobing_ropes_per_minute = 189 := by
  sorry

end NUMINAMATH_CALUDE_xiaobing_jumps_189_ropes_per_minute_l2346_234602


namespace NUMINAMATH_CALUDE_no_fishes_brought_home_l2346_234608

/-- Represents the number of fishes caught from a lake -/
def FishesCaught : Type := ℕ

/-- Represents whether all youngling fishes are returned -/
def ReturnedYounglings : Type := Bool

/-- Calculates the number of fishes brought home -/
def fishesBroughtHome (caught : List FishesCaught) (returned : ReturnedYounglings) : ℕ :=
  sorry

/-- Theorem: If all youngling fishes are returned, no fishes are brought home -/
theorem no_fishes_brought_home (caught : List FishesCaught) :
  fishesBroughtHome caught true = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_fishes_brought_home_l2346_234608


namespace NUMINAMATH_CALUDE_book_selection_combination_l2346_234600

theorem book_selection_combination : ∃ n : ℕ, n * 10^9 + 306249080 = Nat.choose 20 8 := by sorry

end NUMINAMATH_CALUDE_book_selection_combination_l2346_234600


namespace NUMINAMATH_CALUDE_cos_135_and_point_on_unit_circle_l2346_234665

theorem cos_135_and_point_on_unit_circle :
  let angle : Real := 135 * π / 180
  let Q : ℝ × ℝ := (Real.cos angle, Real.sin angle)
  (Real.cos angle = -Real.sqrt 2 / 2) ∧
  (Q = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_cos_135_and_point_on_unit_circle_l2346_234665


namespace NUMINAMATH_CALUDE_ceiling_times_x_equals_156_l2346_234609

theorem ceiling_times_x_equals_156 :
  ∃ x : ℝ, x > 0 ∧ ⌈x⌉ = 13 ∧ ⌈x⌉ * x = 156 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_times_x_equals_156_l2346_234609


namespace NUMINAMATH_CALUDE_g_neg_three_l2346_234683

def g (x : ℝ) : ℝ := x^2 - x + 2*x^3

theorem g_neg_three : g (-3) = -42 := by sorry

end NUMINAMATH_CALUDE_g_neg_three_l2346_234683


namespace NUMINAMATH_CALUDE_lemon_pie_degree_measure_l2346_234624

theorem lemon_pie_degree_measure (total_students : ℕ) (chocolate_pref : ℕ) (apple_pref : ℕ) (blueberry_pref : ℕ) 
  (h_total : total_students = 45)
  (h_chocolate : chocolate_pref = 15)
  (h_apple : apple_pref = 10)
  (h_blueberry : blueberry_pref = 9)
  (h_remaining : (total_students - (chocolate_pref + apple_pref + blueberry_pref)) % 2 = 0) :
  let remaining := total_students - (chocolate_pref + apple_pref + blueberry_pref)
  let lemon_pref := remaining / 2
  ↑lemon_pref / ↑total_students * 360 = 48 := by
sorry

end NUMINAMATH_CALUDE_lemon_pie_degree_measure_l2346_234624


namespace NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l2346_234637

theorem product_divisible_by_sum_implies_inequality (m n : ℕ) 
  (h : (m + n) ∣ (m * n)) : 
  m + n ≤ (Nat.gcd m n)^2 := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l2346_234637


namespace NUMINAMATH_CALUDE_balance_disruption_possible_l2346_234601

/-- Represents a coin with a weight of either 7 or 8 grams -/
inductive Coin
  | Light : Coin  -- 7 grams
  | Heavy : Coin  -- 8 grams

/-- Represents the state of the balance scale -/
structure BalanceState :=
  (left : List Coin)
  (right : List Coin)

/-- Checks if the balance scale is in equilibrium -/
def isBalanced (state : BalanceState) : Bool :=
  (state.left.length = state.right.length) &&
  (state.left.foldl (fun acc c => acc + match c with
    | Coin.Light => 7
    | Coin.Heavy => 8) 0 =
   state.right.foldl (fun acc c => acc + match c with
    | Coin.Light => 7
    | Coin.Heavy => 8) 0)

/-- Performs a swap operation on the balance scale -/
def swapCoins (state : BalanceState) (n : Nat) : BalanceState :=
  { left := state.right.take n ++ state.left.drop n,
    right := state.left.take n ++ state.right.drop n }

/-- The main theorem to be proved -/
theorem balance_disruption_possible :
  ∀ (initialState : BalanceState),
    initialState.left.length = 144 →
    initialState.right.length = 144 →
    isBalanced initialState →
    ∃ (finalState : BalanceState),
      ∃ (numOperations : Nat),
        numOperations ≤ 11 ∧
        ¬isBalanced finalState ∧
        (∃ (swaps : List Nat),
          swaps.length = numOperations ∧
          finalState = swaps.foldl swapCoins initialState) :=
sorry

end NUMINAMATH_CALUDE_balance_disruption_possible_l2346_234601


namespace NUMINAMATH_CALUDE_pear_trees_count_total_trees_sum_apple_tree_exists_pear_tree_exists_l2346_234617

/-- The number of trees in the garden -/
def total_trees : ℕ := 46

/-- The number of pear trees in the garden -/
def pear_trees : ℕ := 27

/-- The number of apple trees in the garden -/
def apple_trees : ℕ := total_trees - pear_trees

/-- Theorem stating that the number of pear trees is 27 -/
theorem pear_trees_count : pear_trees = 27 := by sorry

/-- Theorem stating that the sum of apple and pear trees equals the total number of trees -/
theorem total_trees_sum : apple_trees + pear_trees = total_trees := by sorry

/-- Theorem stating that among any 28 trees, there is at least one apple tree -/
theorem apple_tree_exists (subset : Finset ℕ) (h : subset.card = 28) (h2 : subset ⊆ Finset.range total_trees) : 
  ∃ (tree : ℕ), tree ∈ subset ∧ tree < apple_trees := by sorry

/-- Theorem stating that among any 20 trees, there is at least one pear tree -/
theorem pear_tree_exists (subset : Finset ℕ) (h : subset.card = 20) (h2 : subset ⊆ Finset.range total_trees) : 
  ∃ (tree : ℕ), tree ∈ subset ∧ tree ≥ apple_trees := by sorry

end NUMINAMATH_CALUDE_pear_trees_count_total_trees_sum_apple_tree_exists_pear_tree_exists_l2346_234617


namespace NUMINAMATH_CALUDE_square_perimeter_l2346_234643

/-- Given a square with side length 15 cm, prove that its perimeter is 60 cm. -/
theorem square_perimeter (side_length : ℝ) (h : side_length = 15) : 
  4 * side_length = 60 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2346_234643


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2346_234636

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence. -/
def CommonDifference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (p q : ℕ) (h_arith : ArithmeticSequence a)
  (h_p : a p = 4) (h_q : a q = 2) (h_pq : p = 4 + q) :
  ∃ d : ℝ, CommonDifference a d ∧ d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2346_234636


namespace NUMINAMATH_CALUDE_larger_number_proof_l2346_234605

theorem larger_number_proof (L S : ℕ) (hL : L > S) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2346_234605


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l2346_234603

theorem consecutive_integers_product_812_sum_57 :
  ∀ x y : ℕ+, 
    x.val + 1 = y.val →
    x * y = 812 →
    x + y = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l2346_234603


namespace NUMINAMATH_CALUDE_pentagonal_grid_toothpicks_l2346_234629

/-- The number of toothpicks in the base of the pentagonal grid -/
def base_toothpicks : ℕ := 10

/-- The number of toothpicks in each of the four non-base sides -/
def side_toothpicks : ℕ := 8

/-- The number of sides excluding the base -/
def num_sides : ℕ := 4

/-- The number of vertices in a pentagon -/
def num_vertices : ℕ := 5

/-- The total number of toothpicks needed for the framed pentagonal grid -/
def total_toothpicks : ℕ := base_toothpicks + num_sides * side_toothpicks + num_vertices

theorem pentagonal_grid_toothpicks : total_toothpicks = 47 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_grid_toothpicks_l2346_234629


namespace NUMINAMATH_CALUDE_f_properties_l2346_234658

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.tan x

def is_in_domain (x : ℝ) : Prop :=
  ∀ k : ℤ, x ≠ k * Real.pi + Real.pi / 2

def is_period (p : ℝ) : Prop :=
  ∀ x : ℝ, is_in_domain x → f (x + p) = f x

theorem f_properties :
  (∀ x : ℝ, is_in_domain x ↔ ∃ y : ℝ, f y = f x) ∧
  ¬ is_period Real.pi ∧
  is_period (2 * Real.pi) := by sorry

end NUMINAMATH_CALUDE_f_properties_l2346_234658


namespace NUMINAMATH_CALUDE_jims_taxi_charge_l2346_234694

/-- Proves that the additional charge per 2/5 of a mile is $0.30 for Jim's taxi service -/
theorem jims_taxi_charge (initial_fee : ℚ) (total_charge : ℚ) (trip_distance : ℚ) :
  initial_fee = 2.25 →
  total_charge = 4.95 →
  trip_distance = 3.6 →
  (total_charge - initial_fee) / (trip_distance / (2/5)) = 0.30 := by
sorry

end NUMINAMATH_CALUDE_jims_taxi_charge_l2346_234694


namespace NUMINAMATH_CALUDE_bryan_bus_time_l2346_234688

/-- Represents the travel time for Bryan's commute -/
structure CommuteTimes where
  walkToStation : ℕ  -- Time to walk from house to bus station
  walkToWork : ℕ     -- Time to walk from bus station to work
  totalYearlyTime : ℕ -- Total yearly commute time in hours
  daysWorked : ℕ     -- Number of days worked per year

/-- Calculates the one-way bus ride time in minutes -/
def onewayBusTime (c : CommuteTimes) : ℕ :=
  let totalDailyTime := (c.totalYearlyTime * 60) / c.daysWorked
  let totalWalkTime := 2 * (c.walkToStation + c.walkToWork)
  (totalDailyTime - totalWalkTime) / 2

/-- Theorem stating that Bryan's one-way bus ride time is 20 minutes -/
theorem bryan_bus_time :
  let c := CommuteTimes.mk 5 5 365 365
  onewayBusTime c = 20 := by
  sorry

end NUMINAMATH_CALUDE_bryan_bus_time_l2346_234688


namespace NUMINAMATH_CALUDE_encode_decode_natural_numbers_l2346_234649

/-- Given a list of 100 natural numbers, we can encode them into a single number. -/
theorem encode_decode_natural_numbers :
  ∃ (encode : (Fin 100 → ℕ) → ℕ) (decode : ℕ → (Fin 100 → ℕ)),
    ∀ (nums : Fin 100 → ℕ), decode (encode nums) = nums :=
by sorry

end NUMINAMATH_CALUDE_encode_decode_natural_numbers_l2346_234649


namespace NUMINAMATH_CALUDE_trailing_zeros_of_product_trailing_zeros_of_product_is_90_l2346_234668

/-- The number of trailing zeros in the product of 20^50 and 50^20 -/
theorem trailing_zeros_of_product : ℕ :=
  let a := 20^50
  let b := 50^20
  let product := a * b
  90

/-- Proof that the number of trailing zeros in the product of 20^50 and 50^20 is 90 -/
theorem trailing_zeros_of_product_is_90 :
  trailing_zeros_of_product = 90 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_product_trailing_zeros_of_product_is_90_l2346_234668


namespace NUMINAMATH_CALUDE_pentagonal_faces_count_l2346_234671

/-- A convex polyhedron with pentagon and hexagon faces -/
structure ConvexPolyhedron where
  -- Number of pentagonal faces
  n : ℕ
  -- Number of hexagonal faces
  k : ℕ
  -- The polyhedron is convex
  convex : True
  -- Faces are either pentagons or hexagons
  faces_pentagon_or_hexagon : True
  -- Exactly three edges meet at each vertex
  three_edges_per_vertex : True

/-- The number of pentagonal faces in a convex polyhedron with pentagon and hexagon faces -/
theorem pentagonal_faces_count (p : ConvexPolyhedron) : p.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_faces_count_l2346_234671


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2346_234650

theorem polynomial_division_remainder : ∃ q r : Polynomial ℚ, 
  (X : Polynomial ℚ)^4 = (X^2 + 4*X + 1) * q + r ∧ 
  r.degree < (X^2 + 4*X + 1).degree ∧ 
  r = -56*X - 15 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2346_234650


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l2346_234674

theorem consecutive_even_integers_sum (x : ℝ) :
  (x - 2) * x * (x + 2) = 48 * ((x - 2) + x + (x + 2)) →
  (x - 2) + x + (x + 2) = 6 * Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l2346_234674


namespace NUMINAMATH_CALUDE_largest_five_digit_sum_20_l2346_234690

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

def digit_sum (n : ℕ) : ℕ :=
  (digits n).sum

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem largest_five_digit_sum_20 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 20 → n ≤ 99200 :=
sorry

end NUMINAMATH_CALUDE_largest_five_digit_sum_20_l2346_234690


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l2346_234695

theorem subset_implies_a_values (a : ℝ) : 
  let A : Set ℝ := {-1, 1}
  let B : Set ℝ := {x | a * x + 2 = 0}
  B ⊆ A → a ∈ ({-2, 0, 2} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l2346_234695


namespace NUMINAMATH_CALUDE_shaded_area_grid_l2346_234653

/-- The area of the shaded region in a grid with specific properties -/
theorem shaded_area_grid (total_width total_height large_triangle_base large_triangle_height small_triangle_base small_triangle_height : ℝ) 
  (hw : total_width = 15)
  (hh : total_height = 5)
  (hlb : large_triangle_base = 15)
  (hlh : large_triangle_height = 3)
  (hsb : small_triangle_base = 3)
  (hsh : small_triangle_height = 4) :
  total_width * total_height - (large_triangle_base * large_triangle_height / 2) + (small_triangle_base * small_triangle_height / 2) = 58.5 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_grid_l2346_234653


namespace NUMINAMATH_CALUDE_starting_lineup_count_l2346_234613

/-- Represents a football team with its composition and eligibility rules. -/
structure FootballTeam where
  totalMembers : ℕ
  offensiveLinemenEligible : ℕ
  tightEndEligible : ℕ
  
/-- Calculates the number of ways to choose a starting lineup for a given football team. -/
def chooseStartingLineup (team : FootballTeam) : ℕ :=
  team.offensiveLinemenEligible * 
  team.tightEndEligible * 
  (team.totalMembers - 2) * 
  (team.totalMembers - 3) * 
  (team.totalMembers - 4)

/-- Theorem stating that for the given team composition, there are 5760 ways to choose a starting lineup. -/
theorem starting_lineup_count : 
  chooseStartingLineup ⟨12, 4, 2⟩ = 5760 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l2346_234613


namespace NUMINAMATH_CALUDE_small_cuboids_needed_for_large_l2346_234657

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℕ :=
  d.width * d.length * d.height

/-- The dimensions of the large cuboid -/
def largeCuboid : CuboidDimensions :=
  { width := 24, length := 15, height := 28 }

/-- The dimensions of the small cuboid -/
def smallCuboid : CuboidDimensions :=
  { width := 4, length := 5, height := 7 }

/-- Theorem stating that 72 small cuboids are needed to create the large cuboid -/
theorem small_cuboids_needed_for_large : 
  (cuboidVolume largeCuboid) / (cuboidVolume smallCuboid) = 72 := by
  sorry

end NUMINAMATH_CALUDE_small_cuboids_needed_for_large_l2346_234657


namespace NUMINAMATH_CALUDE_parallel_vector_scalar_l2346_234630

/-- Given two 2D vectors a and b, find the scalar m such that m*a + b is parallel to a - 2*b -/
theorem parallel_vector_scalar (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b = (-1, 2)) :
  ∃ m : ℝ, m * a.1 + b.1 = k * (a.1 - 2 * b.1) ∧ 
           m * a.2 + b.2 = k * (a.2 - 2 * b.2) ∧ 
           m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vector_scalar_l2346_234630


namespace NUMINAMATH_CALUDE_triangle_side_length_l2346_234673

-- Define the triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real
  x : Real
  y : Real
  z : Real

-- State the theorem
theorem triangle_side_length (t : Triangle) 
  (h1 : t.y = 7)
  (h2 : t.z = 6)
  (h3 : Real.cos (t.Y - t.Z) = 17/18) :
  t.x = Real.sqrt 65 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2346_234673


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l2346_234642

theorem initial_number_of_persons (n : ℕ) 
  (h1 : 4 * n = 48) : n = 12 := by
  sorry

#check initial_number_of_persons

end NUMINAMATH_CALUDE_initial_number_of_persons_l2346_234642


namespace NUMINAMATH_CALUDE_fraction_simplification_l2346_234634

theorem fraction_simplification (a b : ℝ) : (9 * b) / (6 * a + 3) = (3 * b) / (2 * a + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2346_234634


namespace NUMINAMATH_CALUDE_disjunction_false_implies_both_false_l2346_234679

theorem disjunction_false_implies_both_false (p q : Prop) :
  (¬(p ∨ q)) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_false_implies_both_false_l2346_234679


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2346_234626

theorem solution_set_inequality (x : ℝ) : 
  (abs (x - 1) + abs (x - 2) ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2346_234626


namespace NUMINAMATH_CALUDE_animal_biscuit_problem_l2346_234648

theorem animal_biscuit_problem :
  ∀ (dogs cats : ℕ),
  dogs + cats = 10 →
  6 * dogs + 5 * cats = 56 →
  dogs = 6 ∧ cats = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_animal_biscuit_problem_l2346_234648


namespace NUMINAMATH_CALUDE_min_soldiers_in_formation_l2346_234677

/-- Represents a rectangular formation of soldiers -/
structure SoldierFormation where
  columns : ℕ
  rows : ℕ
  new_uniforms : ℕ

/-- Checks if the formation satisfies the given conditions -/
def is_valid_formation (f : SoldierFormation) : Prop :=
  f.new_uniforms = (f.columns * f.rows) / 100 ∧
  f.new_uniforms ≥ (3 * f.columns) / 10 ∧
  f.new_uniforms ≥ (2 * f.rows) / 5

/-- The theorem stating the minimum number of soldiers -/
theorem min_soldiers_in_formation :
  ∀ f : SoldierFormation, is_valid_formation f → f.columns * f.rows ≥ 1200 :=
by sorry

end NUMINAMATH_CALUDE_min_soldiers_in_formation_l2346_234677


namespace NUMINAMATH_CALUDE_decreasing_quadratic_function_parameter_range_l2346_234612

/-- If f(x) = x^2 - 2(1-a)x + 2 is a decreasing function on (-∞, 4], then a ∈ (-∞, -3] -/
theorem decreasing_quadratic_function_parameter_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → (x^2 - 2*(1-a)*x + 2) > (y^2 - 2*(1-a)*y + 2)) →
  a ∈ Set.Iic (-3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_function_parameter_range_l2346_234612


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_greater_than_one_l2346_234681

-- Define the function f(x) = 2ax^2 - x - 1
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - x - 1

-- State the theorem
theorem unique_solution_implies_a_greater_than_one :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_greater_than_one_l2346_234681


namespace NUMINAMATH_CALUDE_cube_equation_solution_l2346_234672

theorem cube_equation_solution :
  ∃! x : ℝ, (12 - x)^3 = x^3 ∧ x = 12 := by sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l2346_234672


namespace NUMINAMATH_CALUDE_total_books_l2346_234675

/-- The number of books on a mystery shelf -/
def mystery_books_per_shelf : ℕ := 7

/-- The number of books on a picture book shelf -/
def picture_books_per_shelf : ℕ := 5

/-- The number of books on a science fiction shelf -/
def scifi_books_per_shelf : ℕ := 8

/-- The number of books on a biography shelf -/
def biography_books_per_shelf : ℕ := 6

/-- The number of mystery shelves -/
def mystery_shelves : ℕ := 8

/-- The number of picture book shelves -/
def picture_shelves : ℕ := 2

/-- The number of science fiction shelves -/
def scifi_shelves : ℕ := 3

/-- The number of biography shelves -/
def biography_shelves : ℕ := 4

/-- The total number of books on Megan's shelves -/
theorem total_books : 
  mystery_books_per_shelf * mystery_shelves + 
  picture_books_per_shelf * picture_shelves + 
  scifi_books_per_shelf * scifi_shelves + 
  biography_books_per_shelf * biography_shelves = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l2346_234675


namespace NUMINAMATH_CALUDE_rock_skipping_theorem_l2346_234604

/-- The number of times Bob can skip a rock -/
def bob_skips : ℕ := 12

/-- The number of times Jim can skip a rock -/
def jim_skips : ℕ := 15

/-- The number of rocks each person skipped -/
def rocks_skipped : ℕ := 10

/-- The total number of skips for both Bob and Jim -/
def total_skips : ℕ := bob_skips * rocks_skipped + jim_skips * rocks_skipped

theorem rock_skipping_theorem : total_skips = 270 := by
  sorry

end NUMINAMATH_CALUDE_rock_skipping_theorem_l2346_234604


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2346_234622

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  equation : (ℝ × ℝ) → Prop

-- Define our specific circle
def myCircle : Circle :=
  { center := (2, -1),
    equation := fun (x, y) => (x - 2)^2 + (y + 1)^2 = 3 }

-- Theorem statement
theorem circle_center_coordinates :
  myCircle.center = (2, -1) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2346_234622
