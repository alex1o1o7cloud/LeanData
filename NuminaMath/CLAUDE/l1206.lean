import Mathlib

namespace NUMINAMATH_CALUDE_two_times_choose_six_two_l1206_120628

theorem two_times_choose_six_two : 2 * (Nat.choose 6 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_times_choose_six_two_l1206_120628


namespace NUMINAMATH_CALUDE_miss_spelling_paper_sheets_l1206_120635

theorem miss_spelling_paper_sheets : ∃ (total_sheets : ℕ) (num_pupils : ℕ),
  total_sheets = 3 * num_pupils + 31 ∧
  total_sheets = 4 * num_pupils + 8 ∧
  total_sheets = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_miss_spelling_paper_sheets_l1206_120635


namespace NUMINAMATH_CALUDE_g_of_neg_three_eq_eight_l1206_120694

/-- Given functions f and g, prove that g(-3) = 8 -/
theorem g_of_neg_three_eq_eight
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = 4 * x - 7)
  (hg : ∀ x, g (f x) = 3 * x^2 + 4 * x + 1) :
  g (-3) = 8 := by
sorry

end NUMINAMATH_CALUDE_g_of_neg_three_eq_eight_l1206_120694


namespace NUMINAMATH_CALUDE_kindergarten_class_average_l1206_120614

theorem kindergarten_class_average (giraffe elephant rabbit : ℕ) : 
  giraffe = 225 →
  elephant = giraffe + 48 →
  rabbit = giraffe - 24 →
  (giraffe + elephant + rabbit) / 3 = 233 :=
by sorry

end NUMINAMATH_CALUDE_kindergarten_class_average_l1206_120614


namespace NUMINAMATH_CALUDE_polygon_interior_angles_increase_l1206_120619

theorem polygon_interior_angles_increase (n : ℕ) :
  (n + 1 - 2) * 180 - (n - 2) * 180 = 180 → n + 1 - n = 1 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_increase_l1206_120619


namespace NUMINAMATH_CALUDE_statue_model_ratio_l1206_120618

/-- Given a statue of height 75 feet and a model of height 5 inches,
    prove that one inch of the model represents 15 feet of the statue. -/
theorem statue_model_ratio :
  let statue_height : ℝ := 75  -- statue height in feet
  let model_height : ℝ := 5    -- model height in inches
  statue_height / model_height = 15 := by
sorry


end NUMINAMATH_CALUDE_statue_model_ratio_l1206_120618


namespace NUMINAMATH_CALUDE_total_volume_of_four_cubes_l1206_120610

theorem total_volume_of_four_cubes (edge_length : ℝ) (num_cubes : ℕ) :
  edge_length = 5 → num_cubes = 4 → (edge_length ^ 3) * num_cubes = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_four_cubes_l1206_120610


namespace NUMINAMATH_CALUDE_expression_evaluation_l1206_120671

theorem expression_evaluation (x y : ℤ) (hx : x = -1) (hy : y = 2) :
  2 * x * y + (3 * x * y - 2 * y^2) - 2 * (x * y - y^2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1206_120671


namespace NUMINAMATH_CALUDE_train_length_l1206_120676

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 63 → time_s = 20 → length_m = speed_kmh * (1000 / 3600) * time_s → length_m = 350 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1206_120676


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l1206_120649

theorem quadratic_equivalence :
  ∀ x y : ℝ, y = x^2 + 2*x + 4 ↔ y = (x + 1)^2 + 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l1206_120649


namespace NUMINAMATH_CALUDE_average_price_theorem_l1206_120677

theorem average_price_theorem (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  let p := (2 * a * b) / (a + b)
  a < p ∧ p < Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_average_price_theorem_l1206_120677


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_cubes_reciprocals_l1206_120667

theorem quadratic_roots_sum_of_cubes_reciprocals 
  (a b c r s : ℝ) 
  (h1 : 3 * a * r^2 + 5 * b * r + 7 * c = 0) 
  (h2 : 3 * a * s^2 + 5 * b * s + 7 * c = 0) 
  (h3 : r ≠ 0) 
  (h4 : s ≠ 0) 
  (h5 : c ≠ 0) : 
  1 / r^3 + 1 / s^3 = (-5 * b * (25 * b^2 - 63 * c)) / (343 * c^3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_cubes_reciprocals_l1206_120667


namespace NUMINAMATH_CALUDE_range_of_m_l1206_120609

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x - 2| ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - 1 - m) * (x - 1 + m) ≤ 0}

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (m > 0 ∧ A ⊂ B m) → m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1206_120609


namespace NUMINAMATH_CALUDE_inequality_proof_l1206_120630

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1206_120630


namespace NUMINAMATH_CALUDE_line_intercepts_minimum_minimum_sum_of_intercepts_l1206_120608

theorem line_intercepts_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a + b = a * b) : 
  (b / a) + (a / b) ≥ 2 ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = a * b ∧ x / a + y / b = 2) :=
by sorry

theorem minimum_sum_of_intercepts (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a + b = a * b) :
  a + b ≥ 4 ∧ (a + b = 4 ↔ a = 2 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_minimum_minimum_sum_of_intercepts_l1206_120608


namespace NUMINAMATH_CALUDE_odd_integer_not_divides_power_plus_one_l1206_120659

theorem odd_integer_not_divides_power_plus_one (n m : ℕ) : 
  n > 1 → Odd n → m ≥ 1 → ¬(n ∣ m^(n-1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_integer_not_divides_power_plus_one_l1206_120659


namespace NUMINAMATH_CALUDE_triangle_angle_and_area_l1206_120652

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the measure of angle B and the area of the triangle. -/
theorem triangle_angle_and_area 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 6)
  (h2 : b = 5)
  (h3 : Real.cos A = -4/5) :
  B = π/6 ∧ 
  (1/2 * a * b * Real.sin C = (9 * Real.sqrt 3 - 12) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_and_area_l1206_120652


namespace NUMINAMATH_CALUDE_no_division_for_all_n_l1206_120637

theorem no_division_for_all_n : ∀ n : ℕ, Nat.gcd (n + 2) (n^3 - 2*n^2 - 5*n + 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_division_for_all_n_l1206_120637


namespace NUMINAMATH_CALUDE_student_age_l1206_120686

theorem student_age (student_age man_age : ℕ) : 
  man_age = student_age + 26 →
  man_age + 2 = 2 * (student_age + 2) →
  student_age = 24 := by
sorry

end NUMINAMATH_CALUDE_student_age_l1206_120686


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_implies_a_eq_one_l1206_120693

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is nonzero. -/
def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number constructed from the real number a. -/
def complexNumber (a : ℝ) : ℂ :=
  ⟨a^2 - 3*a + 2, a - 2⟩

/-- If the complex number ((a^2 - 3a + 2) + (a - 2)i) is purely imaginary, then a = 1. -/
theorem complex_purely_imaginary_implies_a_eq_one (a : ℝ) :
  isPurelyImaginary (complexNumber a) → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_complex_purely_imaginary_implies_a_eq_one_l1206_120693


namespace NUMINAMATH_CALUDE_intersection_line_slope_is_one_third_l1206_120643

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 5 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 16*y + 24 = 0

/-- The slope of the line passing through the intersection points of two circles -/
def intersectionLineSlope (c1 c2 : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem intersection_line_slope_is_one_third :
  intersectionLineSlope circle1 circle2 = 1/3 := by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_is_one_third_l1206_120643


namespace NUMINAMATH_CALUDE_candies_distribution_proof_l1206_120680

def least_candies_to_remove (total_candies : ℕ) (num_friends : ℕ) : ℕ :=
  total_candies % num_friends

theorem candies_distribution_proof (total_candies : ℕ) (num_friends : ℕ) 
  (h1 : total_candies = 25) (h2 : num_friends = 4) :
  least_candies_to_remove total_candies num_friends = 1 := by
  sorry

end NUMINAMATH_CALUDE_candies_distribution_proof_l1206_120680


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1206_120665

theorem constant_term_expansion (x : ℝ) : 
  (x^4 + x + 5) * (x^5 + x^3 + 15) = x^9 + x^7 + 15*x^4 + x^6 + x^4 + 15*x + 5*x^5 + 5*x^3 + 75 := by
  sorry

#check constant_term_expansion

end NUMINAMATH_CALUDE_constant_term_expansion_l1206_120665


namespace NUMINAMATH_CALUDE_product_increase_theorem_l1206_120681

theorem product_increase_theorem :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ),
    (a₁ - 3) * (a₂ - 3) * (a₃ - 3) * (a₄ - 3) * (a₅ - 3) * (a₆ - 3) * (a₇ - 3) =
    13 * (a₁ * a₂ * a₃ * a₄ * a₅ * a₆ * a₇) :=
by
  sorry

end NUMINAMATH_CALUDE_product_increase_theorem_l1206_120681


namespace NUMINAMATH_CALUDE_optimal_price_reduction_l1206_120647

/-- Represents the daily profit function for a mall's product sales -/
def daily_profit (initial_sales : ℕ) (initial_profit : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_profit - price_reduction) * (initial_sales + 2 * price_reduction)

/-- Theorem stating that a price reduction of $12 results in a daily profit of $3572 -/
theorem optimal_price_reduction (initial_sales : ℕ) (initial_profit : ℝ)
    (h1 : initial_sales = 70)
    (h2 : initial_profit = 50) :
    daily_profit initial_sales initial_profit 12 = 3572 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_reduction_l1206_120647


namespace NUMINAMATH_CALUDE_work_completion_time_l1206_120664

/-- Given that 72 men can complete a piece of work in 18 days, and the number of men and days are inversely proportional, prove that 144 men will complete the same work in 9 days. -/
theorem work_completion_time 
  (men : ℕ → ℝ)
  (days : ℕ → ℝ)
  (h1 : men 1 = 72)
  (h2 : days 1 = 18)
  (h3 : ∀ k : ℕ, k > 0 → men k * days k = men 1 * days 1) :
  men 2 = 144 ∧ days 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1206_120664


namespace NUMINAMATH_CALUDE_lcm_ratio_implies_gcd_l1206_120615

theorem lcm_ratio_implies_gcd (A B : ℕ) (h1 : Nat.lcm A B = 180) (h2 : ∃ k : ℕ, A = 2 * k ∧ B = 3 * k) : 
  Nat.gcd A B = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_ratio_implies_gcd_l1206_120615


namespace NUMINAMATH_CALUDE_max_rotation_surface_area_l1206_120655

/-- Represents a triangle inscribed in a circle -/
structure InscribedTriangle where
  r : ℝ  -- radius of the circumscribed circle
  A : ℝ × ℝ  -- coordinates of point A
  B : ℝ × ℝ  -- coordinates of point B
  C : ℝ × ℝ  -- coordinates of point C

/-- Calculates the surface area generated by rotating side BC around the tangent at A -/
def rotationSurfaceArea (triangle : InscribedTriangle) : ℝ :=
  sorry

/-- Theorem: The maximum surface area generated by rotating side BC of an inscribed triangle
    around the tangent at A is achieved when the triangle is equilateral and equals 3r²π√3 -/
theorem max_rotation_surface_area (triangle : InscribedTriangle) :
  rotationSurfaceArea triangle ≤ 3 * triangle.r^2 * Real.pi * Real.sqrt 3 ∧
  (rotationSurfaceArea triangle = 3 * triangle.r^2 * Real.pi * Real.sqrt 3 ↔
   triangle.A.1^2 + triangle.A.2^2 = triangle.r^2 ∧
   triangle.B.1^2 + triangle.B.2^2 = triangle.r^2 ∧
   triangle.C.1^2 + triangle.C.2^2 = triangle.r^2 ∧
   (triangle.A.1 - triangle.B.1)^2 + (triangle.A.2 - triangle.B.2)^2 =
   (triangle.B.1 - triangle.C.1)^2 + (triangle.B.2 - triangle.C.2)^2 ∧
   (triangle.A.1 - triangle.C.1)^2 + (triangle.A.2 - triangle.C.2)^2 =
   (triangle.B.1 - triangle.C.1)^2 + (triangle.B.2 - triangle.C.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_max_rotation_surface_area_l1206_120655


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_for_ellipse_l1206_120607

/-- Predicate to determine if an equation represents an ellipse -/
def is_ellipse (a b : ℝ) : Prop := sorry

/-- Theorem stating that a > 0 and b > 0 is a necessary but not sufficient condition for ax^2 + by^2 = 1 to represent an ellipse -/
theorem necessary_not_sufficient_for_ellipse :
  (∀ a b : ℝ, is_ellipse a b → a > 0 ∧ b > 0) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ¬is_ellipse a b) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_for_ellipse_l1206_120607


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_l1206_120631

theorem consecutive_integers_product_812_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_l1206_120631


namespace NUMINAMATH_CALUDE_value_of_m_l1206_120623

theorem value_of_m (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 2*x + m
  let g : ℝ → ℝ := λ x => x^2 - 2*x + 2*m + 8
  3 * f 5 = g 5 → m = -22 := by
sorry

end NUMINAMATH_CALUDE_value_of_m_l1206_120623


namespace NUMINAMATH_CALUDE_polynomial_divisibility_and_divisor_l1206_120602

theorem polynomial_divisibility_and_divisor : ∃ m : ℤ,
  (∀ x : ℝ, (4 * x^2 - 6 * x + m) % (x - 3) = 0) ∧
  m = -18 ∧
  36 % m = 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_and_divisor_l1206_120602


namespace NUMINAMATH_CALUDE_batsman_performance_theorem_l1206_120640

/-- Represents a batsman's performance in a cricket tournament -/
structure BatsmanPerformance where
  innings : ℕ
  runsBeforeLastInning : ℕ
  runsInLastInning : ℕ
  averageIncrease : ℚ
  boundariesBeforeLastInning : ℕ
  boundariesInLastInning : ℕ

/-- Calculates the batting average after the last inning -/
def battingAverage (performance : BatsmanPerformance) : ℚ :=
  (performance.runsBeforeLastInning + performance.runsInLastInning) / performance.innings

/-- Calculates the batting efficiency factor -/
def battingEfficiencyFactor (performance : BatsmanPerformance) : ℚ :=
  (performance.boundariesBeforeLastInning + performance.boundariesInLastInning) / performance.innings

theorem batsman_performance_theorem (performance : BatsmanPerformance) 
  (h1 : performance.innings = 17)
  (h2 : performance.runsInLastInning = 84)
  (h3 : performance.averageIncrease = 5/2)
  (h4 : performance.boundariesInLastInning = 12)
  (h5 : performance.boundariesBeforeLastInning + performance.boundariesInLastInning = 72) :
  battingAverage performance = 44 ∧ battingEfficiencyFactor performance = 72/17 := by
  sorry

#eval (72 : ℚ) / 17

end NUMINAMATH_CALUDE_batsman_performance_theorem_l1206_120640


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1206_120605

theorem complex_expression_equality (z : ℂ) (h : z = 1 + Complex.I) :
  5 / z + z^2 = 5/2 - (1/2) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1206_120605


namespace NUMINAMATH_CALUDE_no_solution_for_system_l1206_120654

theorem no_solution_for_system :
  ¬∃ (x y : ℝ), (2 * x - 3 * y = 7) ∧ (4 * x - 6 * y = 20) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_system_l1206_120654


namespace NUMINAMATH_CALUDE_fathers_contribution_l1206_120613

/-- Given the costs of items, savings, and lacking amount, calculate the father's contribution --/
theorem fathers_contribution 
  (mp3_cost cd_cost savings lacking : ℕ) 
  (h1 : mp3_cost = 120)
  (h2 : cd_cost = 19)
  (h3 : savings = 55)
  (h4 : lacking = 64) :
  mp3_cost + cd_cost = savings + lacking + 148 := by
  sorry

#check fathers_contribution

end NUMINAMATH_CALUDE_fathers_contribution_l1206_120613


namespace NUMINAMATH_CALUDE_johnnys_hourly_wage_l1206_120682

/-- Johnny's hourly wage calculation --/
theorem johnnys_hourly_wage :
  let total_earned : ℚ := 11.75
  let hours_worked : ℕ := 5
  let hourly_wage : ℚ := total_earned / hours_worked
  hourly_wage = 2.35 := by
  sorry

end NUMINAMATH_CALUDE_johnnys_hourly_wage_l1206_120682


namespace NUMINAMATH_CALUDE_max_sum_with_negative_l1206_120674

def S : Finset Int := {-7, -5, -3, 0, 2, 4, 6}

def is_valid_selection (a b c : Int) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a < 0 ∨ b < 0 ∨ c < 0)

theorem max_sum_with_negative :
  ∃ (a b c : Int), is_valid_selection a b c ∧
    a + b + c = 7 ∧
    ∀ (x y z : Int), is_valid_selection x y z → x + y + z ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_negative_l1206_120674


namespace NUMINAMATH_CALUDE_partnership_profit_l1206_120601

/-- A partnership problem with four partners A, B, C, and D -/
theorem partnership_profit (total_capital : ℝ) (total_profit : ℝ) : 
  (1 / 3 : ℝ) * total_capital / total_capital = 810 / total_profit →
  (1 / 3 : ℝ) + (1 / 4 : ℝ) + (1 / 5 : ℝ) + 
    (1 - ((1 / 3 : ℝ) + (1 / 4 : ℝ) + (1 / 5 : ℝ))) = 1 →
  total_profit = 2430 := by
  sorry

#check partnership_profit

end NUMINAMATH_CALUDE_partnership_profit_l1206_120601


namespace NUMINAMATH_CALUDE_a_range_l1206_120692

def sequence_a (a : ℝ) : ℕ+ → ℝ
  | ⟨1, _⟩ => a
  | ⟨n+1, _⟩ => 4*(n+1) + (-1)^(n+1) * (8 - 2*a)

theorem a_range (a : ℝ) :
  (∀ n : ℕ+, sequence_a a n < sequence_a a (n + 1)) →
  (3 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_a_range_l1206_120692


namespace NUMINAMATH_CALUDE_family_gathering_arrangements_l1206_120658

theorem family_gathering_arrangements (n : ℕ) (h : n = 6) : 
  Nat.choose n (n / 2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_family_gathering_arrangements_l1206_120658


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l1206_120663

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l1206_120663


namespace NUMINAMATH_CALUDE_race_outcomes_five_participants_l1206_120699

/-- The number of different 1st-2nd-3rd place outcomes in a race -/
def raceOutcomes (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 1) * (n - 2)

/-- Theorem: In a race with 5 participants where one must finish first and there are no ties,
    the number of different 1st-2nd-3rd place outcomes is 12. -/
theorem race_outcomes_five_participants :
  raceOutcomes 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_five_participants_l1206_120699


namespace NUMINAMATH_CALUDE_kaleb_book_count_l1206_120633

theorem kaleb_book_count (initial_books sold_books new_books : ℕ) :
  initial_books = 34 →
  sold_books = 17 →
  new_books = 7 →
  initial_books - sold_books + new_books = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_kaleb_book_count_l1206_120633


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l1206_120650

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + base * acc) 0

theorem base_conversion_theorem :
  let base := 5
  let T := 0
  let P := 1
  let Q := 2
  let R := 3
  let S := 4
  let dividend_base5 := [P, Q, R, S, R, Q, P]
  let divisor_base5 := [Q, R, Q]
  (to_decimal dividend_base5 base = 24336) ∧
  (to_decimal divisor_base5 base = 67) := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l1206_120650


namespace NUMINAMATH_CALUDE_range_of_a_l1206_120632

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) a, f x ∈ Set.Icc (-4 : ℝ) 32) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) a, f x = -4) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) a, f x = 32) →
  a ∈ Set.Icc 2 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1206_120632


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1206_120683

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 2/3) : 
  Real.sin (2 * α) = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1206_120683


namespace NUMINAMATH_CALUDE_nested_sqrt_simplification_l1206_120675

theorem nested_sqrt_simplification : 
  Real.sqrt (9 * Real.sqrt (27 * Real.sqrt 81)) = 9 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_nested_sqrt_simplification_l1206_120675


namespace NUMINAMATH_CALUDE_quadratic_roots_nm_l1206_120622

theorem quadratic_roots_nm (m n : ℝ) : 
  (∀ x, 2 * x^2 + m * x + n = 0 ↔ x = -2 ∨ x = 1) → 
  n^m = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_nm_l1206_120622


namespace NUMINAMATH_CALUDE_min_p_value_l1206_120616

/-- The probability that Alex and Dylan are on the same team, given that Alex picks one of the cards a or a+7, and Dylan picks the other. -/
def p (a : ℕ) : ℚ :=
  let total_combinations := (52 - 2).choose 2
  let lower_team_combinations := (44 - a).choose 2
  let higher_team_combinations := (a - 1).choose 2
  (lower_team_combinations + higher_team_combinations : ℚ) / total_combinations

/-- The minimum value of a for which p(a) ≥ 1/2 -/
def min_a : ℕ := 8

theorem min_p_value :
  p min_a = 73 / 137 ∧ 
  p min_a ≥ 1 / 2 ∧
  ∀ a : ℕ, a < min_a → p a < 1 / 2 := by sorry

end NUMINAMATH_CALUDE_min_p_value_l1206_120616


namespace NUMINAMATH_CALUDE_lines_parallel_or_skew_l1206_120698

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for planes
variable (parallelPlanes : Plane → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subsetLinePlane : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (parallelLines : Line → Line → Prop)

-- Define the skew relation for lines
variable (skewLines : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_or_skew
  (a b : Line) (α β : Plane)
  (h_diff_lines : a ≠ b)
  (h_diff_planes : α ≠ β)
  (h_parallel_planes : parallelPlanes α β)
  (h_a_in_α : subsetLinePlane a α)
  (h_b_in_β : subsetLinePlane b β) :
  parallelLines a b ∨ skewLines a b :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_or_skew_l1206_120698


namespace NUMINAMATH_CALUDE_last_remaining_number_l1206_120604

/-- Represents the marking process on a list of numbers -/
def markingProcess (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else
  let m := markingProcess (n / 2)
  if m * 2 > n then 2 * m - 1 else 2 * m + 1

/-- The theorem stating that for 120 numbers, the last remaining number is 64 -/
theorem last_remaining_number :
  markingProcess 120 = 64 := by
  sorry

end NUMINAMATH_CALUDE_last_remaining_number_l1206_120604


namespace NUMINAMATH_CALUDE_gumball_probability_l1206_120627

/-- Given a jar with blue and pink gumballs, if the probability of drawing
    two blue gumballs with replacement is 25/49, then the probability of
    drawing a pink gumball is 2/7. -/
theorem gumball_probability (p_blue p_pink : ℝ) : 
  p_blue + p_pink = 1 →
  p_blue^2 = 25/49 →
  p_pink = 2/7 := by
sorry

end NUMINAMATH_CALUDE_gumball_probability_l1206_120627


namespace NUMINAMATH_CALUDE_factory_weekly_production_l1206_120678

/-- Represents the production of toys in a factory --/
structure ToyProduction where
  days_per_week : ℕ
  toys_per_day : ℕ
  constant_daily_production : Bool

/-- Calculates the weekly toy production --/
def weekly_production (tp : ToyProduction) : ℕ :=
  tp.days_per_week * tp.toys_per_day

/-- Theorem stating the weekly toy production for the given factory --/
theorem factory_weekly_production :
  ∀ (tp : ToyProduction),
    tp.days_per_week = 4 →
    tp.toys_per_day = 1500 →
    tp.constant_daily_production →
    weekly_production tp = 6000 := by
  sorry

end NUMINAMATH_CALUDE_factory_weekly_production_l1206_120678


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1206_120642

theorem quadratic_root_relation (a c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ y = 3 * x ∧ a * x^2 + 6 * x + c = 0 ∧ a * y^2 + 6 * y + c = 0) →
  c = 27 / (4 * a) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1206_120642


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1206_120657

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0}
def B : Set Int := {0, 1, 2}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1206_120657


namespace NUMINAMATH_CALUDE_cube_surface_area_l1206_120612

/-- Given a cube with side length x and distance d between non-intersecting diagonals
    of adjacent lateral faces, prove that its total surface area is 18d^2. -/
theorem cube_surface_area (d : ℝ) (h : d > 0) :
  let x := d * Real.sqrt 3
  6 * x^2 = 18 * d^2 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1206_120612


namespace NUMINAMATH_CALUDE_median_to_mean_l1206_120626

theorem median_to_mean (m : ℝ) : 
  let set := [m, m + 3, m + 7, m + 10, m + 12]
  m + 7 = 12 → 
  (set.sum / set.length : ℝ) = 11.4 := by
sorry

end NUMINAMATH_CALUDE_median_to_mean_l1206_120626


namespace NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l1206_120689

theorem at_least_one_leq_neg_two (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1/b ≤ -2) ∨ (b + 1/c ≤ -2) ∨ (c + 1/a ≤ -2) := by sorry

end NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l1206_120689


namespace NUMINAMATH_CALUDE_product_expansion_terms_count_l1206_120600

theorem product_expansion_terms_count :
  let a_terms := 3  -- number of terms in (a₁ + a₂ + a₃)
  let b_terms := 4  -- number of terms in (b₁ + b₂ + b₃ + b₄)
  let c_terms := 5  -- number of terms in (c₁ + c₂ + c₃ + c₄ + c₅)
  a_terms * b_terms * c_terms = 60 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_terms_count_l1206_120600


namespace NUMINAMATH_CALUDE_locus_of_midpoints_l1206_120672

-- Define the circles and their properties
def circle1_radius : ℝ := 1
def circle2_radius : ℝ := 3
def distance_between_centers : ℝ := 10

-- Define the locus
def locus_inner_radius : ℝ := 1
def locus_outer_radius : ℝ := 2

-- Theorem statement
theorem locus_of_midpoints (p : ℝ × ℝ) : 
  (∃ (p1 p2 : ℝ × ℝ), 
    (p1.1 - 0)^2 + (p1.2 - 0)^2 = circle1_radius^2 ∧ 
    (p2.1 - distance_between_centers)^2 + p2.2^2 = circle2_radius^2 ∧
    p = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)) ↔ 
  (locus_inner_radius^2 ≤ (p.1 - distance_between_centers / 2)^2 + p.2^2 ∧ 
   (p.1 - distance_between_centers / 2)^2 + p.2^2 ≤ locus_outer_radius^2) :=
sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_l1206_120672


namespace NUMINAMATH_CALUDE_class_size_l1206_120685

/-- Represents the number of students excelling in various combinations of sports -/
structure SportExcellence where
  sprint : ℕ
  swimming : ℕ
  basketball : ℕ
  sprint_swimming : ℕ
  swimming_basketball : ℕ
  sprint_basketball : ℕ
  all_three : ℕ

/-- The total number of students in the class -/
def total_students (se : SportExcellence) (non_excellent : ℕ) : ℕ :=
  se.sprint + se.swimming + se.basketball
  - se.sprint_swimming - se.swimming_basketball - se.sprint_basketball
  + se.all_three + non_excellent

/-- The theorem stating the total number of students in the class -/
theorem class_size (se : SportExcellence) (non_excellent : ℕ) : 
  se.sprint = 17 → se.swimming = 18 → se.basketball = 15 →
  se.sprint_swimming = 6 → se.swimming_basketball = 6 →
  se.sprint_basketball = 5 → se.all_three = 2 → non_excellent = 4 →
  total_students se non_excellent = 39 := by
  sorry

/-- Example usage of the theorem -/
example : ∃ (se : SportExcellence) (non_excellent : ℕ), 
  total_students se non_excellent = 39 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1206_120685


namespace NUMINAMATH_CALUDE_max_sum_cyclic_fraction_l1206_120684

open Real BigOperators

/-- The maximum value of the sum for positive real numbers with sum 1 -/
theorem max_sum_cyclic_fraction (n : ℕ) (a : ℕ → ℝ) 
  (hn : n ≥ 4)
  (ha_pos : ∀ k, a k > 0)
  (ha_sum : ∑ k in Finset.range n, a k = 1) :
  (∑ k in Finset.range n, (a k)^2 / (a k + a ((k + 1) % n) + a ((k + 2) % n))) ≤ 1/3 :=
sorry


end NUMINAMATH_CALUDE_max_sum_cyclic_fraction_l1206_120684


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l1206_120666

/-- Represents the number of people wearing each color of clothing -/
structure ClothingCounts where
  blue : Nat
  yellow : Nat
  red : Nat

/-- Calculates the number of valid arrangements for a given set of clothing counts -/
def validArrangements (counts : ClothingCounts) : Nat :=
  sorry

/-- The specific problem instance -/
def problemInstance : ClothingCounts :=
  { blue := 2, yellow := 2, red := 1 }

/-- The main theorem stating that the number of valid arrangements for the problem instance is 48 -/
theorem valid_arrangements_count :
  validArrangements problemInstance = 48 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l1206_120666


namespace NUMINAMATH_CALUDE_number_of_players_is_16_l1206_120669

def jersey_cost : ℚ := 25
def shorts_cost : ℚ := 15.20
def socks_cost : ℚ := 6.80
def total_cost : ℚ := 752

def equipment_cost_per_player : ℚ := jersey_cost + shorts_cost + socks_cost

theorem number_of_players_is_16 :
  (total_cost / equipment_cost_per_player : ℚ) = 16 := by sorry

end NUMINAMATH_CALUDE_number_of_players_is_16_l1206_120669


namespace NUMINAMATH_CALUDE_triangle_properties_l1206_120656

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the main theorem
theorem triangle_properties (t : Triangle)
  (h : t.a / (Real.cos t.C * Real.sin t.B) = t.b / Real.sin t.B + t.c / Real.cos t.C) :
  t.B = π / 4 ∧
  (t.b = Real.sqrt 2 → 
    ∀ (area : ℝ), area = 1 / 2 * t.a * t.c * Real.sin t.B → area ≤ (Real.sqrt 2 + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1206_120656


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l1206_120641

theorem cubic_inequality_solution (x : ℝ) :
  x^3 - 10*x^2 + 15*x > 0 ↔ x ∈ Set.Ioo 0 (5 - Real.sqrt 10) ∪ Set.Ioi (5 + Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l1206_120641


namespace NUMINAMATH_CALUDE_equation_solution_l1206_120670

theorem equation_solution : ∃ x : ℚ, (5 * x - 3 * (x + 2) = 450 - 9 * (x - 4)) ∧ (x = 492 / 11) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1206_120670


namespace NUMINAMATH_CALUDE_diamond_value_l1206_120611

theorem diamond_value (diamond : ℕ) : 
  diamond < 10 →  -- Ensuring diamond is a digit
  (9 * diamond + 3 = 10 * diamond + 2) →  -- Equivalent to ◇3_9 = ◇2_10
  diamond = 1 := by
sorry

end NUMINAMATH_CALUDE_diamond_value_l1206_120611


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1206_120617

theorem quadratic_equation_roots (p q r : ℝ) (h : p ≠ 0 ∧ q ≠ r) :
  let f : ℝ → ℝ := λ x => p * (q - r) * x^2 + q * (r - p) * x + r * (p - q)
  (f (-1) = 0) → (f (-r * (p - q) / (p * (q - r))) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1206_120617


namespace NUMINAMATH_CALUDE_twin_primes_with_prime_expression_l1206_120696

/-- Twin primes are prime numbers that differ by 2 -/
def TwinPrimes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ (p = q + 2 ∨ q = p + 2)

/-- The expression p^2 - pq + q^2 -/
def Expression (p q : ℕ) : ℕ :=
  p^2 - p*q + q^2

theorem twin_primes_with_prime_expression :
  ∀ p q : ℕ, TwinPrimes p q ∧ Nat.Prime (Expression p q) ↔ (p = 5 ∧ q = 3) ∨ (p = 3 ∧ q = 5) :=
sorry

end NUMINAMATH_CALUDE_twin_primes_with_prime_expression_l1206_120696


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_implies_m_less_than_two_l1206_120625

/-- A quadratic equation with parameter m -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 + 2*x + m = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  4 - 4*m

theorem quadratic_real_solutions_implies_m_less_than_two (m : ℝ) :
  (∃ x : ℝ, quadratic_equation x m) → m < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_implies_m_less_than_two_l1206_120625


namespace NUMINAMATH_CALUDE_new_athlete_rate_is_15_l1206_120634

/-- The rate at which new athletes arrived at the Ultimate Fitness Camp --/
def new_athlete_rate (initial_athletes : ℕ) (leaving_rate : ℕ) (leaving_hours : ℕ) 
  (arrival_hours : ℕ) (total_difference : ℕ) : ℕ :=
  let athletes_left := leaving_rate * leaving_hours
  let remaining_athletes := initial_athletes - athletes_left
  let final_athletes := initial_athletes - total_difference
  let new_athletes := final_athletes - remaining_athletes
  new_athletes / arrival_hours

/-- Theorem stating the rate at which new athletes arrived --/
theorem new_athlete_rate_is_15 : 
  new_athlete_rate 300 28 4 7 7 = 15 := by sorry

end NUMINAMATH_CALUDE_new_athlete_rate_is_15_l1206_120634


namespace NUMINAMATH_CALUDE_pants_price_calculation_l1206_120662

/-- Given 10 pairs of pants with a 20% discount, followed by a 10% tax,
    resulting in a final price of $396, prove that the original retail
    price of each pair of pants is $45. -/
theorem pants_price_calculation (quantity : Nat) (discount_rate : Real)
    (tax_rate : Real) (final_price : Real) :
  quantity = 10 →
  discount_rate = 0.20 →
  tax_rate = 0.10 →
  final_price = 396 →
  ∃ (original_price : Real),
    original_price = 45 ∧
    final_price = quantity * original_price * (1 - discount_rate) * (1 + tax_rate) := by
  sorry

end NUMINAMATH_CALUDE_pants_price_calculation_l1206_120662


namespace NUMINAMATH_CALUDE_grocery_shopping_theorem_l1206_120620

def initial_amount : ℝ := 100
def roast_price : ℝ := 17
def vegetables_price : ℝ := 11
def wine_price : ℝ := 12
def dessert_price : ℝ := 8
def bread_price : ℝ := 4
def milk_price : ℝ := 2
def discount_rate : ℝ := 0.15
def tax_rate : ℝ := 0.05

def total_purchase : ℝ := roast_price + vegetables_price + wine_price + dessert_price + bread_price + milk_price

def discounted_total : ℝ := total_purchase * (1 - discount_rate)

def final_amount : ℝ := discounted_total * (1 + tax_rate)

def remaining_amount : ℝ := initial_amount - final_amount

theorem grocery_shopping_theorem : 
  ∃ (ε : ℝ), abs (remaining_amount - 51.80) < ε ∧ ε > 0 :=
by sorry

end NUMINAMATH_CALUDE_grocery_shopping_theorem_l1206_120620


namespace NUMINAMATH_CALUDE_james_socks_l1206_120603

theorem james_socks (red_pairs : ℕ) (black : ℕ) (white : ℕ) : 
  black = red_pairs -- number of black socks is equal to the number of pairs of red socks
  → white = 2 * (2 * red_pairs + black) -- number of white socks is twice the number of red and black socks combined
  → 2 * red_pairs + black + white = 90 -- total number of socks is 90
  → red_pairs = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_socks_l1206_120603


namespace NUMINAMATH_CALUDE_art_gallery_theorem_l1206_120629

theorem art_gallery_theorem (total : ℕ) 
  (h1 : total > 0)
  (h2 : (total / 3 : ℚ) = total / 3)  -- Ensures division is exact
  (h3 : ((total / 3) / 6 : ℚ) = (total / 3) / 6)  -- Ensures division is exact
  (h4 : ((2 * total / 3) / 3 : ℚ) = (2 * total / 3) / 3)  -- Ensures division is exact
  (h5 : 2 * (2 * total / 3) / 3 = 1200) :
  total = 2700 := by
sorry

end NUMINAMATH_CALUDE_art_gallery_theorem_l1206_120629


namespace NUMINAMATH_CALUDE_employed_males_percentage_l1206_120673

/-- The percentage of employed people in the population -/
def employed_percentage : ℝ := 64

/-- The percentage of employed people who are female -/
def female_employed_percentage : ℝ := 25

/-- The theorem stating the percentage of the population that are employed males -/
theorem employed_males_percentage :
  (employed_percentage / 100) * (1 - female_employed_percentage / 100) * 100 = 48 := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l1206_120673


namespace NUMINAMATH_CALUDE_minimum_value_of_f_max_a_for_decreasing_f_properties_l1206_120691

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (4-a)*x^2 - 15*x + a

-- Theorem 1
theorem minimum_value_of_f (a : ℝ) :
  f a 0 = -2 → a = -2 ∧ ∃ x₀, ∀ x, f (-2) x ≥ f (-2) x₀ ∧ f (-2) x₀ = -10 :=
sorry

-- Theorem 2
theorem max_a_for_decreasing (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1) 1, ∀ y ∈ Set.Ioo (-1) 1, x < y → f a x > f a y) →
  a ≤ 10 :=
sorry

-- Theorem combining both results
theorem f_properties :
  (∃ a, f a 0 = -2 ∧ a = -2 ∧ ∃ x₀, ∀ x, f a x ≥ f a x₀ ∧ f a x₀ = -10) ∧
  (∃ a_max, a_max = 10 ∧ ∀ a > a_max, ¬(∀ x ∈ Set.Ioo (-1) 1, ∀ y ∈ Set.Ioo (-1) 1, x < y → f a x > f a y)) :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_max_a_for_decreasing_f_properties_l1206_120691


namespace NUMINAMATH_CALUDE_final_turtle_count_l1206_120624

/-- Number of turtle statues on Grandma Molly's lawn after four years -/
def turtle_statues : ℕ :=
  let year1 := 4
  let year2 := year1 * 4
  let year3_before_breakage := year2 + 12
  let year3_after_breakage := year3_before_breakage - 3
  let year4_new_statues := 3 * 2
  year3_after_breakage + year4_new_statues

theorem final_turtle_count : turtle_statues = 31 := by
  sorry

end NUMINAMATH_CALUDE_final_turtle_count_l1206_120624


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_one_fourth_l1206_120606

theorem arithmetic_square_root_of_one_fourth : Real.sqrt (1 / 4) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_one_fourth_l1206_120606


namespace NUMINAMATH_CALUDE_fiftyMeterDashIsSuitable_suitableSurveyIsCorrect_l1206_120687

/-- Represents a survey option -/
inductive SurveyOption
  | A
  | B
  | C
  | D

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  requiresPrecision : Bool
  easyToConduct : Bool
  nonDestructive : Bool
  manageableSubjects : Bool

/-- Defines the characteristics of a comprehensive survey method -/
def isComprehensiveSurvey (c : SurveyCharacteristics) : Prop :=
  c.requiresPrecision ∧ c.easyToConduct ∧ c.nonDestructive ∧ c.manageableSubjects

/-- Characteristics of the 50-meter dash survey -/
def fiftyMeterDashSurvey : SurveyCharacteristics :=
  { requiresPrecision := true
    easyToConduct := true
    nonDestructive := true
    manageableSubjects := true }

/-- Theorem stating that the 50-meter dash survey is suitable for a comprehensive survey method -/
theorem fiftyMeterDashIsSuitable : isComprehensiveSurvey fiftyMeterDashSurvey :=
  sorry

/-- Function to determine the suitable survey option -/
def suitableSurveyOption : SurveyOption :=
  SurveyOption.A

/-- Theorem stating that the suitable survey option is correct -/
theorem suitableSurveyIsCorrect : suitableSurveyOption = SurveyOption.A :=
  sorry

end NUMINAMATH_CALUDE_fiftyMeterDashIsSuitable_suitableSurveyIsCorrect_l1206_120687


namespace NUMINAMATH_CALUDE_glass_bowl_purchase_price_l1206_120688

theorem glass_bowl_purchase_price 
  (total_bowls : ℕ) 
  (sold_bowls : ℕ) 
  (selling_price : ℚ) 
  (percentage_gain : ℚ) :
  total_bowls = 118 →
  sold_bowls = 102 →
  selling_price = 15 →
  percentage_gain = 8050847457627118 / 100000000000000000 →
  ∃ (purchase_price : ℚ),
    purchase_price = 12 ∧
    sold_bowls * selling_price - total_bowls * purchase_price = 
      (percentage_gain / 100) * (total_bowls * purchase_price) := by
  sorry

end NUMINAMATH_CALUDE_glass_bowl_purchase_price_l1206_120688


namespace NUMINAMATH_CALUDE_last_digit_2008_2005_l1206_120690

theorem last_digit_2008_2005 : (2008^2005) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_2008_2005_l1206_120690


namespace NUMINAMATH_CALUDE_antibiotics_cost_l1206_120639

/-- The cost of Antibiotic A per dose in dollars -/
def cost_A : ℚ := 3

/-- The number of doses of Antibiotic A per day -/
def doses_per_day_A : ℕ := 2

/-- The number of days Antibiotic A is taken per week -/
def days_per_week_A : ℕ := 3

/-- The cost of Antibiotic B per dose in dollars -/
def cost_B : ℚ := 9/2

/-- The number of doses of Antibiotic B per day -/
def doses_per_day_B : ℕ := 1

/-- The number of days Antibiotic B is taken per week -/
def days_per_week_B : ℕ := 4

/-- The total cost of antibiotics for Archie for one week -/
def total_cost : ℚ := cost_A * doses_per_day_A * days_per_week_A + cost_B * doses_per_day_B * days_per_week_B

theorem antibiotics_cost : total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_antibiotics_cost_l1206_120639


namespace NUMINAMATH_CALUDE_min_cosine_sine_fraction_l1206_120644

open Real

theorem min_cosine_sine_fraction (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (cos x)^3 / sin x + (sin x)^3 / cos x ≥ 1 ∧
  ∃ y, 0 < y ∧ y < π / 2 ∧ (cos y)^3 / sin y + (sin y)^3 / cos y = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_cosine_sine_fraction_l1206_120644


namespace NUMINAMATH_CALUDE_abs_reciprocal_neg_six_l1206_120661

theorem abs_reciprocal_neg_six : |1 / (-6)| = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_reciprocal_neg_six_l1206_120661


namespace NUMINAMATH_CALUDE_stating_scale_theorem_l1206_120660

/-- Represents a curve in the xy-plane -/
structure Curve where
  equation : ℝ → ℝ → Prop

/-- Applies a scaling transformation to a curve in the y-axis direction -/
def scale_y (c : Curve) (k : ℝ) : Curve :=
  { equation := λ x y => c.equation x (y / k) }

/-- The original curve x^2 - 4y^2 = 16 -/
def original_curve : Curve :=
  { equation := λ x y => x^2 - 4*y^2 = 16 }

/-- The transformed curve x^2 - y^2 = 16 -/
def transformed_curve : Curve :=
  { equation := λ x y => x^2 - y^2 = 16 }

/-- 
Theorem stating that scaling the original curve by factor 2 in the y-direction 
results in the transformed curve
-/
theorem scale_theorem : scale_y original_curve 2 = transformed_curve := by
  sorry

end NUMINAMATH_CALUDE_stating_scale_theorem_l1206_120660


namespace NUMINAMATH_CALUDE_smallest_x_value_l1206_120621

theorem smallest_x_value (x : ℝ) : 
  ((((5 * x - 20) / (4 * x - 5)) ^ 2 + ((5 * x - 20) / (4 * x - 5))) = 6) →
  x ≥ 35 / 17 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1206_120621


namespace NUMINAMATH_CALUDE_smallest_k_sum_squares_multiple_360_l1206_120653

theorem smallest_k_sum_squares_multiple_360 : 
  ∃ k : ℕ+, (∀ m : ℕ+, m < k → ¬(∃ n : ℕ, m * (m + 1) * (2 * m + 1) = 6 * 360 * n)) ∧ 
  (∃ n : ℕ, k * (k + 1) * (2 * k + 1) = 6 * 360 * n) ∧ 
  k = 432 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_sum_squares_multiple_360_l1206_120653


namespace NUMINAMATH_CALUDE_picture_ratio_proof_l1206_120636

theorem picture_ratio_proof (total pictures : ℕ) (vertical_count : ℕ) (haphazard_count : ℕ) :
  total = 30 →
  vertical_count = 10 →
  haphazard_count = 5 →
  (total - vertical_count - haphazard_count) * 2 = total := by
  sorry

end NUMINAMATH_CALUDE_picture_ratio_proof_l1206_120636


namespace NUMINAMATH_CALUDE_product_difference_bound_l1206_120679

theorem product_difference_bound (n : ℕ+) (a b : ℕ+) 
  (h : (a : ℝ) * b = (n : ℝ)^2 + n + 1) : 
  |((a : ℝ) - b)| ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_bound_l1206_120679


namespace NUMINAMATH_CALUDE_system_solution_l1206_120697

theorem system_solution :
  let eq1 (x y : ℝ) := x * (y - 1) + y * (x + 1) = 6
  let eq2 (x y : ℝ) := (x - 1) * (y + 1) = 1
  (eq1 (4/3) 2 ∧ eq2 (4/3) 2) ∧ (eq1 (-2) (-4/3) ∧ eq2 (-2) (-4/3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1206_120697


namespace NUMINAMATH_CALUDE_binomial_sum_divides_power_of_two_l1206_120645

theorem binomial_sum_divides_power_of_two (n : ℕ) :
  n > 3 →
  (1 + Nat.choose n 1 + Nat.choose n 2 + Nat.choose n 3) ∣ 2^2000 ↔
  n = 7 ∨ n = 23 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_divides_power_of_two_l1206_120645


namespace NUMINAMATH_CALUDE_committee_selection_l1206_120638

theorem committee_selection (n : ℕ) (h : Nat.choose n 3 = 15) : Nat.choose n 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l1206_120638


namespace NUMINAMATH_CALUDE_fill_time_AB_is_2_4_hours_l1206_120668

-- Define the constants for the fill times
def fill_time_ABC : ℝ := 2
def fill_time_AC : ℝ := 3
def fill_time_BC : ℝ := 4

-- Define the rates of water flow for each valve
def rate_A : ℝ := sorry
def rate_B : ℝ := sorry
def rate_C : ℝ := sorry

-- Define the volume of the tank
def tank_volume : ℝ := sorry

-- Theorem to prove
theorem fill_time_AB_is_2_4_hours : 
  tank_volume / (rate_A + rate_B) = 2.4 := by sorry

end NUMINAMATH_CALUDE_fill_time_AB_is_2_4_hours_l1206_120668


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l1206_120651

theorem farmer_tomatoes (picked : ℕ) (left : ℕ) (h1 : picked = 83) (h2 : left = 14) :
  picked + left = 97 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l1206_120651


namespace NUMINAMATH_CALUDE_square_2004_content_l1206_120695

/-- Represents the content of a square in the sequence -/
inductive SquareContent
  | A
  | AB
  | ABCD
  | Number (n : ℕ)

/-- Returns the letter content of the nth square -/
def letterContent (n : ℕ) : SquareContent :=
  match n % 3 with
  | 0 => SquareContent.ABCD
  | 1 => SquareContent.A
  | 2 => SquareContent.AB
  | _ => SquareContent.A  -- This case is mathematically impossible, but needed for completeness

/-- Returns the number content of the nth square -/
def numberContent (n : ℕ) : SquareContent :=
  SquareContent.Number n

/-- Combines letter and number content for the nth square -/
def squareContent (n : ℕ) : (SquareContent × SquareContent) :=
  (letterContent n, numberContent n)

/-- The main theorem to prove -/
theorem square_2004_content :
  squareContent 2004 = (SquareContent.ABCD, SquareContent.Number 2004) := by
  sorry


end NUMINAMATH_CALUDE_square_2004_content_l1206_120695


namespace NUMINAMATH_CALUDE_lisa_spoon_count_l1206_120648

/-- The total number of spoons Lisa has after combining old and new sets -/
def total_spoons (num_children : ℕ) (baby_spoons_per_child : ℕ) (decorative_spoons : ℕ) 
                 (large_spoons : ℕ) (teaspoons : ℕ) : ℕ :=
  num_children * baby_spoons_per_child + decorative_spoons + large_spoons + teaspoons

/-- Theorem stating that Lisa has 39 spoons in total -/
theorem lisa_spoon_count : 
  total_spoons 4 3 2 10 15 = 39 := by
  sorry

end NUMINAMATH_CALUDE_lisa_spoon_count_l1206_120648


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l1206_120646

theorem no_real_roots_quadratic (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + (a - 5) * x + 2 ≠ 0) → 1 < a ∧ a < 9 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l1206_120646
