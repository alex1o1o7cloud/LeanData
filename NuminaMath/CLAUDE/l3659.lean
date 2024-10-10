import Mathlib

namespace economics_and_law_tournament_l3659_365968

theorem economics_and_law_tournament (n : ℕ) (m : ℕ) : 
  220 < n → n < 254 →
  m < n →
  (n - 2*m)^2 = n →
  ∀ k : ℕ, (220 < k ∧ k < 254 ∧ k < n ∧ (k - 2*(n-k))^2 = k) → n - m ≤ k - (n - k) →
  n - m = 105 :=
sorry

end economics_and_law_tournament_l3659_365968


namespace simplify_expression_l3659_365984

theorem simplify_expression (z : ℝ) : z - 2 + 4*z + 3 - 6*z + 5 - 8*z + 7 = -9*z + 13 := by
  sorry

end simplify_expression_l3659_365984


namespace cubic_sum_theorem_l3659_365924

theorem cubic_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
  a^3 + b^3 + c^3 = -36 := by
sorry

end cubic_sum_theorem_l3659_365924


namespace b_age_is_eight_l3659_365930

/-- Given three people a, b, and c, where:
    - a is two years older than b
    - b is twice as old as c
    - The total of their ages is 22
    Prove that b is 8 years old. -/
theorem b_age_is_eight (a b c : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : a + b + c = 22) : 
  b = 8 := by
  sorry

end b_age_is_eight_l3659_365930


namespace field_length_width_ratio_l3659_365932

/-- Proves that the ratio of length to width of a rectangular field is 2:1 given specific conditions --/
theorem field_length_width_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    pond_side = 8 →
    field_length = 112 →
    pond_side^2 = (1/98) * (field_length * field_width) →
    field_length / field_width = 2 := by
  sorry

end field_length_width_ratio_l3659_365932


namespace max_value_x_minus_2y_l3659_365952

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) :
  ∃ (M : ℝ), M = 10 ∧ ∀ (a b : ℝ), a^2 + b^2 - 2*a + 4*b = 0 → a - 2*b ≤ M :=
by sorry

end max_value_x_minus_2y_l3659_365952


namespace square_expression_is_perfect_square_l3659_365928

theorem square_expression_is_perfect_square (n k l : ℕ) 
  (h : n^2 + k^2 = 2 * l^2) : 
  ((2 * l - n - k) * (2 * l - n + k)) / 2 = (l - n)^2 :=
sorry

end square_expression_is_perfect_square_l3659_365928


namespace percentage_difference_l3659_365957

theorem percentage_difference (A B C x : ℝ) : 
  A > 0 → B > 0 → C > 0 → 
  A > C → C > B → 
  A = B * (1 + x / 100) → 
  C = 0.75 * A → 
  x > 100 / 3 := by
sorry

end percentage_difference_l3659_365957


namespace valid_systematic_sampling_l3659_365995

/-- Represents a systematic sampling selection -/
structure SystematicSampling where
  totalStudents : Nat
  sampleSize : Nat
  startingNumber : Nat

/-- Generates the set of selected numbers for a systematic sampling -/
def generateSelection (s : SystematicSampling) : Finset Nat :=
  let interval := s.totalStudents / s.sampleSize
  Finset.image (fun i => s.startingNumber + i * interval) (Finset.range s.sampleSize)

/-- Theorem stating that {3, 13, 23, 33, 43} is a valid systematic sampling selection -/
theorem valid_systematic_sampling :
  ∃ (s : SystematicSampling),
    s.totalStudents = 50 ∧
    s.sampleSize = 5 ∧
    1 ≤ s.startingNumber ∧
    s.startingNumber ≤ s.totalStudents ∧
    generateSelection s = {3, 13, 23, 33, 43} :=
sorry

end valid_systematic_sampling_l3659_365995


namespace ellipse_point_exists_l3659_365936

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the point P
def P : ℝ × ℝ := (0, 1)

-- Define the line x = 4
def line_x_4 (x y : ℝ) : Prop := x = 4

-- Define the distance ratio condition
def distance_ratio (A M : ℝ × ℝ) : Prop :=
  let (ax, ay) := A
  let (mx, my) := M
  (ax^2 + (ay - 1)^2) / ((mx - 0)^2 + (my - 1)^2) = 1/9

theorem ellipse_point_exists : 
  ∃ (A : ℝ × ℝ), 
    let (m, n) := A
    ellipse m n ∧ 
    m > 0 ∧
    ellipse P.1 P.2 ∧
    ∃ (M : ℝ × ℝ), 
      line_x_4 M.1 M.2 ∧
      (n - 1) / m * (M.1 - 0) + 1 = M.2 ∧
      distance_ratio A M :=
by sorry

end ellipse_point_exists_l3659_365936


namespace fruit_problem_solution_l3659_365990

/-- Represents the solution to the fruit buying problem -/
def FruitSolution : Type := ℕ × ℕ × ℕ

/-- The total number of fruits bought -/
def total_fruits : ℕ := 100

/-- The total cost in copper coins -/
def total_cost : ℕ := 100

/-- The cost of a single peach in copper coins -/
def peach_cost : ℕ := 3

/-- The cost of a single plum in copper coins -/
def plum_cost : ℕ := 4

/-- The number of olives that can be bought for 1 copper coin -/
def olives_per_coin : ℕ := 7

/-- Checks if a given solution satisfies all conditions of the problem -/
def is_valid_solution (solution : FruitSolution) : Prop :=
  let (peaches, plums, olives) := solution
  peaches + plums + olives = total_fruits ∧
  peach_cost * peaches + plum_cost * plums + (olives / olives_per_coin) = total_cost

/-- The correct solution to the problem -/
def correct_solution : FruitSolution := (3, 20, 77)

/-- Theorem stating that the correct_solution is the unique valid solution -/
theorem fruit_problem_solution :
  is_valid_solution correct_solution ∧
  ∀ (other : FruitSolution), is_valid_solution other → other = correct_solution :=
sorry

end fruit_problem_solution_l3659_365990


namespace max_value_z_minus_2i_l3659_365922

theorem max_value_z_minus_2i (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (max : ℝ), max = 3 ∧ ∀ w : ℂ, Complex.abs w = 1 → Complex.abs (w - 2*I) ≤ max :=
sorry

end max_value_z_minus_2i_l3659_365922


namespace bird_families_difference_l3659_365964

/-- Given the total number of bird families and the number that flew away,
    prove that the difference between those that stayed and those that flew away is 73. -/
theorem bird_families_difference (total : ℕ) (flew_away : ℕ) 
    (h1 : total = 87) (h2 : flew_away = 7) : total - flew_away - flew_away = 73 := by
  sorry

end bird_families_difference_l3659_365964


namespace brothers_age_in_6_years_l3659_365944

/-- The combined age of 4 brothers in a given number of years from now -/
def combined_age (years_from_now : ℕ) : ℕ :=
  sorry

theorem brothers_age_in_6_years :
  combined_age 15 = 107 → combined_age 6 = 71 :=
by sorry

end brothers_age_in_6_years_l3659_365944


namespace smallest_b_value_l3659_365904

theorem smallest_b_value (b : ℤ) (Q : ℤ → ℤ) : 
  b > 0 →
  (∀ x : ℤ, ∃ (a₀ a₁ a₂ : ℤ), Q x = a₀ * x^2 + a₁ * x + a₂) →
  Q 1 = b ∧ Q 4 = b ∧ Q 7 = b ∧ Q 10 = b →
  Q 2 = -b ∧ Q 5 = -b ∧ Q 8 = -b ∧ Q 11 = -b →
  (∀ c : ℤ, c > 0 ∧ 
    (∃ (P : ℤ → ℤ), (∀ x : ℤ, ∃ (a₀ a₁ a₂ : ℤ), P x = a₀ * x^2 + a₁ * x + a₂) ∧
      P 1 = c ∧ P 4 = c ∧ P 7 = c ∧ P 10 = c ∧
      P 2 = -c ∧ P 5 = -c ∧ P 8 = -c ∧ P 11 = -c) →
    c ≥ b) →
  b = 1260 :=
by sorry

end smallest_b_value_l3659_365904


namespace total_dress_designs_is_40_l3659_365988

/-- The number of color choices for a dress design. -/
def num_colors : ℕ := 4

/-- The number of pattern choices for a dress design. -/
def num_patterns : ℕ := 5

/-- The number of fabric type choices for a dress design. -/
def num_fabric_types : ℕ := 2

/-- The total number of possible dress designs. -/
def total_designs : ℕ := num_colors * num_patterns * num_fabric_types

/-- Theorem stating that the total number of possible dress designs is 40. -/
theorem total_dress_designs_is_40 : total_designs = 40 := by
  sorry

end total_dress_designs_is_40_l3659_365988


namespace min_M_n_value_l3659_365915

def M_n (n k : ℕ+) : ℚ :=
  max (40 / n) (max (80 / (k * n)) (60 / (200 - n - k * n)))

theorem min_M_n_value :
  ∀ k : ℕ+, (∃ n : ℕ+, n + k * n ≤ 200) →
    (∀ n : ℕ+, n + k * n ≤ 200 → M_n n k ≥ 10/11) ∧
    (∃ n : ℕ+, n + k * n ≤ 200 ∧ M_n n k = 10/11) :=
by sorry

end min_M_n_value_l3659_365915


namespace area_minimized_at_k_equals_one_l3659_365900

/-- Represents a planar region defined by a system of inequalities -/
def PlanarRegion := Set (ℝ × ℝ)

/-- Computes the area of a planar region -/
noncomputable def area (Ω : PlanarRegion) : ℝ := sorry

/-- The system of inequalities that defines Ω -/
def systemOfInequalities (k : ℝ) : PlanarRegion := sorry

theorem area_minimized_at_k_equals_one (k : ℝ) (hk : k ≥ 0) :
  let Ω := systemOfInequalities k
  ∀ k' ≥ 0, area Ω ≤ area (systemOfInequalities k') → k = 1 :=
sorry

end area_minimized_at_k_equals_one_l3659_365900


namespace consecutive_integers_sum_of_squares_l3659_365906

theorem consecutive_integers_sum_of_squares : 
  ∀ a : ℤ, (a - 2) * a * (a + 2) = 36 * a → 
  (a - 2)^2 + a^2 + (a + 2)^2 = 200 := by
sorry

end consecutive_integers_sum_of_squares_l3659_365906


namespace scientific_notation_of_10500_l3659_365996

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_10500 :
  toScientificNotation 10500 = ScientificNotation.mk 1.05 4 (by norm_num) :=
sorry

end scientific_notation_of_10500_l3659_365996


namespace read_book_series_l3659_365962

/-- The number of weeks needed to read a book series -/
def weeks_to_read (total_books : ℕ) (first_week : ℕ) (second_week : ℕ) (subsequent_weeks : ℕ) : ℕ :=
  let remaining_books := total_books - (first_week + second_week)
  2 + (remaining_books + subsequent_weeks - 1) / subsequent_weeks

/-- Proof that it takes 7 weeks to read the book series -/
theorem read_book_series : weeks_to_read 54 6 3 9 = 7 := by
  sorry

end read_book_series_l3659_365962


namespace inequality_system_integer_solutions_l3659_365963

theorem inequality_system_integer_solutions :
  ∀ x : ℤ, (3 * x + 6 > x + 8 ∧ x / 4 ≥ (x - 1) / 3) ↔ x ∈ ({2, 3, 4} : Set ℤ) := by
  sorry

end inequality_system_integer_solutions_l3659_365963


namespace quadratic_term_zero_l3659_365981

theorem quadratic_term_zero (a : ℝ) : 
  (∀ x : ℝ, (a * x + 3) * (6 * x^2 - 2 * x + 1) = 6 * a * x^3 + (18 - 2 * a) * x^2 + (a - 6) * x + 3) →
  (∀ x : ℝ, (a * x + 3) * (6 * x^2 - 2 * x + 1) = 6 * a * x^3 + (a - 6) * x + 3) →
  a = 9 := by
sorry

end quadratic_term_zero_l3659_365981


namespace number_puzzle_l3659_365938

theorem number_puzzle (x : ℝ) : (1/2 : ℝ) * x - 300 = 350 → (x + 200) * 2 = 3000 := by
  sorry

end number_puzzle_l3659_365938


namespace rick_ironing_theorem_l3659_365979

/-- Represents the number of dress shirts Rick can iron in an hour -/
def shirts_per_hour : ℕ := 4

/-- Represents the number of dress pants Rick can iron in an hour -/
def pants_per_hour : ℕ := 3

/-- Represents the number of hours Rick spends ironing dress shirts -/
def hours_ironing_shirts : ℕ := 3

/-- Represents the number of hours Rick spends ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- Calculates the total number of pieces of clothing Rick has ironed -/
def total_clothes_ironed : ℕ :=
  shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants

theorem rick_ironing_theorem :
  total_clothes_ironed = 27 := by
  sorry

end rick_ironing_theorem_l3659_365979


namespace log_equation_solution_l3659_365901

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution :
  ∃ y : ℝ, y > 0 ∧ log y 81 = 4/2 → y = 9 := by
  sorry

end log_equation_solution_l3659_365901


namespace place_value_ratios_l3659_365959

theorem place_value_ratios : 
  ∀ (d : ℕ), d > 0 → d < 10 →
  (d * 10000) / (d * 1000) = 10 ∧ 
  (d * 100000) / (d * 100) = 1000 := by
  sorry

end place_value_ratios_l3659_365959


namespace constant_term_expansion_l3659_365934

theorem constant_term_expansion (n : ℕ+) 
  (h : (2 : ℝ)^(n : ℝ) = 32) : 
  Nat.choose n.val 3 = 10 := by
  sorry

end constant_term_expansion_l3659_365934


namespace arithmetic_sequence_sum_l3659_365942

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  -- The sum function for the first n terms
  S : ℕ → ℝ
  -- Property: The sequence of differences forms an arithmetic sequence
  difference_is_arithmetic : ∀ (k : ℕ), S (k + 1) - S k = S (k + 2) - S (k + 1)

/-- Theorem: For an arithmetic sequence with S_n = 30 and S_{2n} = 100, S_{3n} = 170 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) (n : ℕ) 
    (h1 : a.S n = 30) (h2 : a.S (2 * n) = 100) : 
    a.S (3 * n) = 170 := by
  sorry

end arithmetic_sequence_sum_l3659_365942


namespace problem_solution_l3659_365966

theorem problem_solution : 2 * Real.sin (60 * π / 180) + |Real.sqrt 3 - 3| + (π - 1)^0 = 4 := by
  sorry

end problem_solution_l3659_365966


namespace isosceles_triangle_condition_l3659_365999

theorem isosceles_triangle_condition 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_condition : a^2 + a*b + c^2 - b*c = 2*a*c) : 
  a = c ∨ a = b :=
sorry

end isosceles_triangle_condition_l3659_365999


namespace abc_sum_product_bound_l3659_365951

theorem abc_sum_product_bound (a b c : ℝ) (h : a + b + c = 1) :
  0 ≤ a * b + a * c + b * c ∧ a * b + a * c + b * c ≤ 1/3 := by
  sorry

end abc_sum_product_bound_l3659_365951


namespace brick_width_is_10_cm_l3659_365971

/-- Prove that the width of a brick is 10 cm given the specified conditions -/
theorem brick_width_is_10_cm
  (brick_length : ℝ)
  (brick_height : ℝ)
  (wall_length : ℝ)
  (wall_width : ℝ)
  (wall_height : ℝ)
  (num_bricks : ℕ)
  (h1 : brick_length = 20)
  (h2 : brick_height = 7.5)
  (h3 : wall_length = 2600)
  (h4 : wall_width = 200)
  (h5 : wall_height = 75)
  (h6 : num_bricks = 26000)
  (h7 : wall_length * wall_width * wall_height = num_bricks * (brick_length * brick_height * brick_width)) :
  brick_width = 10 := by
  sorry


end brick_width_is_10_cm_l3659_365971


namespace josies_calculation_l3659_365982

theorem josies_calculation (a b c d e : ℤ) : 
  a = 2 → b = 1 → c = -1 → d = 3 → 
  (a - b + c^2 - d + e = a - (b - (c^2 - (d + e)))) → e = 0 := by
sorry

end josies_calculation_l3659_365982


namespace longest_segment_l3659_365953

-- Define the triangle ABD
structure TriangleABD where
  angleABD : ℝ
  angleADB : ℝ
  hab : angleABD = 30
  had : angleADB = 70

-- Define the triangle BCD
structure TriangleBCD where
  angleCBD : ℝ
  angleBDC : ℝ
  hcb : angleCBD = 45
  hbd : angleBDC = 60

-- Define the lengths of the segments
variables {AB AD BD BC CD : ℝ}

-- State the theorem
theorem longest_segment (abd : TriangleABD) (bcd : TriangleBCD) :
  CD > BC ∧ BC > BD ∧ BD > AB ∧ AB > AD :=
sorry

end longest_segment_l3659_365953


namespace additional_cakes_is_21_l3659_365927

/-- Represents the number of cakes baked in a week -/
structure CakeQuantities where
  cheesecakes : ℕ
  muffins : ℕ
  redVelvet : ℕ
  chocolateMoist : ℕ
  fruitcakes : ℕ
  carrotCakes : ℕ

/-- Carter's usual cake quantities -/
def usualQuantities : CakeQuantities := {
  cheesecakes := 6,
  muffins := 5,
  redVelvet := 8,
  chocolateMoist := 0,
  fruitcakes := 0,
  carrotCakes := 0
}

/-- Calculate the new quantities based on the given rates -/
def newQuantities (usual : CakeQuantities) : CakeQuantities := {
  cheesecakes := (usual.cheesecakes * 3 + 1) / 2,
  muffins := (usual.muffins * 6 + 2) / 5,
  redVelvet := (usual.redVelvet * 9 + 2) / 5,
  chocolateMoist := ((usual.redVelvet * 9 + 2) / 5) / 2,
  fruitcakes := (((usual.muffins * 6 + 2) / 5) * 2) / 3,
  carrotCakes := 0
}

/-- Calculate the total additional cakes -/
def additionalCakes (usual new : CakeQuantities) : ℕ :=
  (new.cheesecakes - usual.cheesecakes) +
  (new.muffins - usual.muffins) +
  (new.redVelvet - usual.redVelvet) +
  (new.chocolateMoist - usual.chocolateMoist) +
  (new.fruitcakes - usual.fruitcakes) +
  (new.carrotCakes - usual.carrotCakes)

theorem additional_cakes_is_21 :
  additionalCakes usualQuantities (newQuantities usualQuantities) = 21 := by
  sorry

end additional_cakes_is_21_l3659_365927


namespace point_distance_on_curve_l3659_365907

theorem point_distance_on_curve (e c d : ℝ) : 
  e > 0 →
  c ≠ d →
  c^2 + (Real.sqrt e)^6 = 3 * (Real.sqrt e)^3 * c + 1 →
  d^2 + (Real.sqrt e)^6 = 3 * (Real.sqrt e)^3 * d + 1 →
  |c - d| = |Real.sqrt (5 * e^3 + 4)| :=
by sorry

end point_distance_on_curve_l3659_365907


namespace factorial_of_factorial_div_factorial_l3659_365913

theorem factorial_of_factorial_div_factorial :
  (Nat.factorial (Nat.factorial 3)) / (Nat.factorial 3) = 120 := by
  sorry

end factorial_of_factorial_div_factorial_l3659_365913


namespace worksheets_graded_l3659_365933

/-- 
Given:
- There are 9 worksheets in total
- Each worksheet has 4 problems
- There are 16 problems left to grade

Prove that the number of worksheets already graded is 5.
-/
theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) :
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left = 16 →
  total_worksheets * problems_per_worksheet - problems_left = 5 * problems_per_worksheet :=
by
  sorry

#check worksheets_graded

end worksheets_graded_l3659_365933


namespace cubic_roots_sum_l3659_365908

theorem cubic_roots_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 25*a - 10 = 0 →
  b^3 - 15*b^2 + 25*b - 10 = 0 →
  c^3 - 15*c^2 + 25*c - 10 = 0 →
  (a / ((1/a) + b*c)) + (b / ((1/b) + c*a)) + (c / ((1/c) + a*b)) = 175/11 := by
sorry

end cubic_roots_sum_l3659_365908


namespace total_capacity_is_57600_l3659_365941

/-- The total capacity of James' fleet of vans -/
def total_capacity : ℕ := by
  -- Define the number of vans
  let total_vans : ℕ := 6
  let large_vans : ℕ := 2
  let medium_van : ℕ := 1
  let extra_large_vans : ℕ := 3

  -- Define the capacities
  let base_capacity : ℕ := 8000
  let medium_capacity : ℕ := base_capacity - (base_capacity * 30 / 100)
  let extra_large_capacity : ℕ := base_capacity + (base_capacity * 50 / 100)

  -- Calculate total capacity
  exact large_vans * base_capacity + 
        medium_van * medium_capacity + 
        extra_large_vans * extra_large_capacity

/-- Theorem stating that the total capacity is 57600 gallons -/
theorem total_capacity_is_57600 : total_capacity = 57600 := by
  sorry

end total_capacity_is_57600_l3659_365941


namespace expression_factorization_l3659_365973

theorem expression_factorization (y : ℝ) :
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) := by
sorry

end expression_factorization_l3659_365973


namespace arithmetic_sequence_fifth_term_l3659_365969

/-- Given an arithmetic sequence with first term -1 and third term 5, prove that the fifth term is 11. -/
theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
    a 1 = -1 →                                        -- first term
    a 3 = 5 →                                         -- third term
    a 5 = 11 :=                                       -- fifth term (to prove)
by
  sorry


end arithmetic_sequence_fifth_term_l3659_365969


namespace max_value_of_expression_l3659_365911

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  (∃ (m : ℝ), ∀ (a b : ℝ), a + b = 5 →
    x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 ≤ m ∧
    x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 = m) ∧
  (∀ (m : ℝ), (∀ (a b : ℝ), a + b = 5 →
    x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 ≤ m ∧
    x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 = m) →
  m = 441/2) := by
sorry

end max_value_of_expression_l3659_365911


namespace infinitely_many_terms_same_prime_factors_l3659_365912

/-- An arithmetic progression of natural numbers -/
def arithmeticProgression (a d : ℕ) : ℕ → ℕ := fun n => a + n * d

/-- The set of prime factors of a natural number -/
def primeFactors (n : ℕ) : Set ℕ := {p : ℕ | Nat.Prime p ∧ p ∣ n}

/-- There are infinitely many terms in an arithmetic progression with the same prime factors -/
theorem infinitely_many_terms_same_prime_factors (a d : ℕ) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, primeFactors (arithmeticProgression a d n) = primeFactors a :=
sorry

end infinitely_many_terms_same_prime_factors_l3659_365912


namespace rachel_apple_picking_l3659_365989

theorem rachel_apple_picking (num_trees : ℕ) (total_picked : ℕ) (h1 : num_trees = 4) (h2 : total_picked = 28) :
  total_picked / num_trees = 7 := by
  sorry

end rachel_apple_picking_l3659_365989


namespace bonnie_sticker_count_l3659_365943

/-- Calculates Bonnie's initial sticker count given the problem conditions -/
def bonnies_initial_stickers (june_initial : ℕ) (grandparents_gift : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (june_initial + 2 * grandparents_gift)

/-- Theorem stating that Bonnie's initial sticker count is 63 given the problem conditions -/
theorem bonnie_sticker_count :
  bonnies_initial_stickers 76 25 189 = 63 := by
  sorry

#eval bonnies_initial_stickers 76 25 189

end bonnie_sticker_count_l3659_365943


namespace meaningful_fraction_range_l3659_365931

theorem meaningful_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = (1 / (x - 1)) + 1) → x ≠ 1 := by
  sorry

end meaningful_fraction_range_l3659_365931


namespace bicycle_wheels_l3659_365992

theorem bicycle_wheels (num_bicycles : ℕ) (num_tricycles : ℕ) (tricycle_wheels : ℕ) (total_wheels : ℕ) :
  num_bicycles = 16 →
  num_tricycles = 7 →
  tricycle_wheels = 3 →
  total_wheels = 53 →
  ∃ (bicycle_wheels : ℕ), 
    bicycle_wheels = 2 ∧ 
    num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels = total_wheels :=
by sorry

end bicycle_wheels_l3659_365992


namespace simplify_polynomial_l3659_365965

/-- Proves that the simplified form of (9x^9+7x^8+4x^7) + (x^11+x^9+2x^7+3x^3+5x+8)
    is x^11+10x^9+7x^8+6x^7+3x^3+5x+8 -/
theorem simplify_polynomial (x : ℝ) :
  (9 * x^9 + 7 * x^8 + 4 * x^7) + (x^11 + x^9 + 2 * x^7 + 3 * x^3 + 5 * x + 8) =
  x^11 + 10 * x^9 + 7 * x^8 + 6 * x^7 + 3 * x^3 + 5 * x + 8 := by
  sorry

#check simplify_polynomial

end simplify_polynomial_l3659_365965


namespace impossible_to_transform_to_fives_l3659_365970

/-- Represents the three magician tricks -/
inductive MagicTrick
  | subtract_one
  | divide_by_two
  | multiply_by_three

/-- Represents the state of the transformation process -/
structure TransformState where
  numbers : List ℕ
  trick_counts : List ℕ
  deriving Repr

/-- Checks if a number is within the allowed range -/
def is_valid_number (n : ℕ) : Bool :=
  n ≤ 10

/-- Applies a magic trick to a number -/
def apply_trick (trick : MagicTrick) (n : ℕ) : Option ℕ :=
  match trick with
  | MagicTrick.subtract_one => if n > 0 then some (n - 1) else none
  | MagicTrick.divide_by_two => if n % 2 = 0 then some (n / 2) else none
  | MagicTrick.multiply_by_three => if n * 3 ≤ 10 then some (n * 3) else none

/-- Checks if the transformation is complete (all numbers are 5) -/
def is_transformation_complete (state : TransformState) : Bool :=
  state.numbers.all (· = 5)

/-- Checks if the transformation process is still valid -/
def is_valid_state (state : TransformState) : Bool :=
  state.numbers.all is_valid_number ∧
  state.trick_counts.all (· ≤ 5)

/-- The main theorem statement -/
theorem impossible_to_transform_to_fives :
  ¬ ∃ (final_state : TransformState),
    is_transformation_complete final_state ∧
    is_valid_state final_state ∧
    (∃ (initial_state : TransformState),
      initial_state.numbers = [3, 8, 9, 2, 4] ∧
      initial_state.trick_counts = [0, 0, 0] ∧
      -- There exists a sequence of valid transformations from initial_state to final_state
      True) :=
sorry

end impossible_to_transform_to_fives_l3659_365970


namespace debt_average_payment_l3659_365925

/-- Calculates the average payment for a debt paid in installments over a year. -/
theorem debt_average_payment
  (total_installments : ℕ)
  (first_payment_count : ℕ)
  (first_payment_amount : ℚ)
  (payment_increase : ℚ)
  (h1 : total_installments = 104)
  (h2 : first_payment_count = 24)
  (h3 : first_payment_amount = 520)
  (h4 : payment_increase = 95) :
  let remaining_payment_count := total_installments - first_payment_count
  let remaining_payment_amount := first_payment_amount + payment_increase
  let total_amount := first_payment_count * first_payment_amount +
                      remaining_payment_count * remaining_payment_amount
  total_amount / total_installments = 593.08 := by
sorry


end debt_average_payment_l3659_365925


namespace john_participation_count_l3659_365955

/-- Represents the possible point values in the archery competition -/
inductive ArcheryPoints
  | first : ArcheryPoints
  | second : ArcheryPoints
  | third : ArcheryPoints
  | fourth : ArcheryPoints

/-- Returns the point value for a given place -/
def pointValue (p : ArcheryPoints) : Nat :=
  match p with
  | ArcheryPoints.first => 11
  | ArcheryPoints.second => 7
  | ArcheryPoints.third => 5
  | ArcheryPoints.fourth => 2

/-- Represents John's participation in the archery competition -/
def JohnParticipation := List ArcheryPoints

/-- Calculates the product of points for a given participation list -/
def productOfPoints (participation : JohnParticipation) : Nat :=
  participation.foldl (fun acc p => acc * pointValue p) 1

/-- Theorem: John participated 7 times given the conditions -/
theorem john_participation_count :
  ∃ (participation : JohnParticipation),
    productOfPoints participation = 38500 ∧ participation.length = 7 :=
by sorry

end john_participation_count_l3659_365955


namespace a_less_than_b_l3659_365991

theorem a_less_than_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (h : (1 - a) * b > 1/4) : a < b := by
  sorry

end a_less_than_b_l3659_365991


namespace vector_equation_l3659_365972

/-- Given non-collinear points A, B, C, and a point O satisfying
    16*OA - 12*OB - 3*OC = 0, prove that OA = 12*AB + 3*AC -/
theorem vector_equation (A B C O : EuclideanSpace ℝ (Fin 3)) 
  (h_not_collinear : ¬Collinear ℝ {A, B, C})
  (h_equation : 16 • (O - A) - 12 • (O - B) - 3 • (O - C) = 0) :
  O - A = 12 • (B - A) + 3 • (C - A) := by
  sorry

end vector_equation_l3659_365972


namespace remainder_problem_l3659_365998

theorem remainder_problem (N : ℤ) : N % 221 = 43 → N % 17 = 9 := by
  sorry

end remainder_problem_l3659_365998


namespace circle_through_three_points_l3659_365958

theorem circle_through_three_points :
  let A : ℝ × ℝ := (1, 12)
  let B : ℝ × ℝ := (7, 10)
  let C : ℝ × ℝ := (-9, 2)
  let circle_equation (x y : ℝ) := x^2 + y^2 - 2*x - 4*y - 95 = 0
  (circle_equation A.1 A.2) ∧ 
  (circle_equation B.1 B.2) ∧ 
  (circle_equation C.1 C.2) := by
sorry

end circle_through_three_points_l3659_365958


namespace spherical_to_rectangular_conversion_l3659_365994

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 6
  let θ : ℝ := (7 * π) / 4
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (3 * Real.sqrt 6, -3 * Real.sqrt 6, 3) := by
  sorry

end spherical_to_rectangular_conversion_l3659_365994


namespace power_of_two_geq_n_l3659_365974

theorem power_of_two_geq_n (n : ℕ) (h : n ≥ 1) : 2^n ≥ n := by
  sorry

end power_of_two_geq_n_l3659_365974


namespace bag_production_l3659_365909

/-- Given that 15 machines produce 45 bags per minute, 
    prove that 150 machines will produce 3600 bags in 8 minutes. -/
theorem bag_production 
  (machines : ℕ) 
  (bags_per_minute : ℕ) 
  (time : ℕ) 
  (h1 : machines = 15) 
  (h2 : bags_per_minute = 45) 
  (h3 : time = 8) :
  (150 : ℕ) * bags_per_minute * time / machines = 3600 :=
sorry

end bag_production_l3659_365909


namespace triangle_inequality_l3659_365961

/-- Given a triangle with side lengths a, b, and c, 
    the inequality a^2 b(a-b) + b^2 c(b-c) + c^2 a(c-a) ≥ 0 holds. -/
theorem triangle_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
by sorry

end triangle_inequality_l3659_365961


namespace vector_sum_magnitude_l3659_365935

/-- The angle between two vectors in radians -/
def angle_between (a b : ℝ × ℝ) : ℝ := sorry

/-- The magnitude (length) of a vector -/
def magnitude (v : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)  -- 60° in radians
  (h2 : a = (2, 0))
  (h3 : magnitude b = 1) :
  magnitude (a + 2 • b) = 2 * Real.sqrt 3 := by
  sorry

end vector_sum_magnitude_l3659_365935


namespace product_equals_fraction_l3659_365954

/-- The decimal representation of the repeating decimal 0.456̄ -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of the repeating decimal and 11 -/
def product : ℚ := repeating_decimal * 11

/-- Theorem stating that the product of 0.456̄ and 11 is equal to 1672/333 -/
theorem product_equals_fraction : product = 1672 / 333 := by sorry

end product_equals_fraction_l3659_365954


namespace sphere_volume_from_surface_area_l3659_365919

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), (4 * π * r^2 = 144 * π) → ((4/3) * π * r^3 = 288 * π) :=
by
  sorry

end sphere_volume_from_surface_area_l3659_365919


namespace earth_sun_distance_calculation_l3659_365978

/-- The speed of light in vacuum (in m/s) -/
def speed_of_light : ℝ := 3 * 10^8

/-- The time taken for sunlight to reach Earth (in s) -/
def time_to_earth : ℝ := 5 * 10^2

/-- The distance between the Earth and the Sun (in m) -/
def earth_sun_distance : ℝ := 1.5 * 10^11

/-- Theorem stating that the distance between the Earth and the Sun
    is equal to the product of the speed of light and the time taken
    for sunlight to reach Earth -/
theorem earth_sun_distance_calculation :
  earth_sun_distance = speed_of_light * time_to_earth := by
  sorry

end earth_sun_distance_calculation_l3659_365978


namespace bbq_ice_cost_l3659_365937

/-- The cost of ice for Chad's BBQ --/
theorem bbq_ice_cost (people : ℕ) (ice_per_person : ℕ) (bags_per_pack : ℕ) (price_per_pack : ℚ) : 
  people = 15 →
  ice_per_person = 2 →
  bags_per_pack = 10 →
  price_per_pack = 3 →
  (people * ice_per_person : ℚ) / bags_per_pack * price_per_pack = 9 :=
by
  sorry

#check bbq_ice_cost

end bbq_ice_cost_l3659_365937


namespace jodi_walking_days_l3659_365976

def weekly_distance (days_per_week : ℕ) : ℕ := 
  1 * days_per_week + 2 * days_per_week + 3 * days_per_week + 4 * days_per_week

theorem jodi_walking_days : 
  ∃ (days_per_week : ℕ), weekly_distance days_per_week = 60 ∧ days_per_week = 6 := by
  sorry

end jodi_walking_days_l3659_365976


namespace remainder_of_a_83_mod_49_l3659_365918

theorem remainder_of_a_83_mod_49 : (6^83 + 8^83) % 49 = 35 := by
  sorry

end remainder_of_a_83_mod_49_l3659_365918


namespace crayon_count_l3659_365946

theorem crayon_count (blue : ℕ) (red : ℕ) (green : ℕ) : 
  blue = 3 → 
  red = 4 * blue → 
  green = 2 * red → 
  blue + red + green = 39 := by
sorry

end crayon_count_l3659_365946


namespace digit_add_sequence_contains_even_l3659_365903

/-- A sequence of natural numbers where each term is obtained from the previous term
    by adding one of its nonzero digits. -/
def DigitAddSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ d : ℕ, d > 0 ∧ d < 10 ∧ d ∣ a n ∧ a (n + 1) = a n + d

/-- The theorem stating that a DigitAddSequence contains an even number. -/
theorem digit_add_sequence_contains_even (a : ℕ → ℕ) (h : DigitAddSequence a) :
  ∃ n : ℕ, Even (a n) :=
sorry

end digit_add_sequence_contains_even_l3659_365903


namespace equality_from_fraction_equation_l3659_365987

theorem equality_from_fraction_equation (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b) → a = b := by
  sorry

end equality_from_fraction_equation_l3659_365987


namespace arithmetic_sequences_problem_l3659_365947

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequences_problem 
  (a b : ℕ → ℝ) (d₁ d₂ : ℝ) 
  (ha : arithmetic_sequence a d₁)
  (hb : arithmetic_sequence b d₂)
  (A : ℕ → ℝ)
  (B : ℕ → ℝ)
  (hA : ∀ n, A n = a n + b n)
  (hB : ∀ n, B n = a n * b n)
  (hA₁ : A 1 = 1)
  (hA₂ : A 2 = 3)
  (hB_arith : arithmetic_sequence B (B 2 - B 1)) :
  (∀ n, A n = 2 * n - 1) ∧ d₁ * d₂ = 0 := by
  sorry

end arithmetic_sequences_problem_l3659_365947


namespace quadratic_inequality_properties_l3659_365977

/-- Given a quadratic inequality with solution set (x₁, x₂), prove certain properties of the roots -/
theorem quadratic_inequality_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h_sol : ∀ x, a * (x - 1) * (x + 3) + 2 > 0 ↔ x ∈ Set.Ioo x₁ x₂) 
  (h_order : x₁ < x₂) :
  x₁ + x₂ + 2 = 0 ∧ |x₁ - x₂| > 4 ∧ x₁ * x₂ + 3 < 0 := by
sorry

end quadratic_inequality_properties_l3659_365977


namespace arithmetic_sequence_sum_l3659_365917

theorem arithmetic_sequence_sum (N : ℤ) : 
  (1001 : ℤ) + 1004 + 1007 + 1010 + 1013 = 5050 - N → N = 15 := by
  sorry

end arithmetic_sequence_sum_l3659_365917


namespace scaled_recipe_correct_l3659_365948

/-- Represents a cookie recipe -/
structure CookieRecipe where
  cookies : ℕ
  flour : ℕ
  eggs : ℕ

/-- Scales a cookie recipe by a given factor -/
def scaleRecipe (recipe : CookieRecipe) (factor : ℕ) : CookieRecipe :=
  { cookies := recipe.cookies * factor
  , flour := recipe.flour * factor
  , eggs := recipe.eggs * factor }

theorem scaled_recipe_correct (original : CookieRecipe) (scaled : CookieRecipe) :
  original.cookies = 40 ∧
  original.flour = 3 ∧
  original.eggs = 2 ∧
  scaled = scaleRecipe original 3 →
  scaled.cookies = 120 ∧
  scaled.flour = 9 ∧
  scaled.eggs = 6 := by
  sorry

#check scaled_recipe_correct

end scaled_recipe_correct_l3659_365948


namespace triangle_area_from_square_areas_l3659_365986

theorem triangle_area_from_square_areas (a b c : ℝ) (h1 : a^2 = 36) (h2 : b^2 = 64) (h3 : c^2 = 100) :
  (1/2) * a * b = 24 :=
by sorry

end triangle_area_from_square_areas_l3659_365986


namespace b_le_c_for_geometric_l3659_365950

/-- A geometric sequence of positive real numbers -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n ∧ a n > 0

/-- Definition of b_n -/
def b (a : ℕ → ℝ) (n : ℕ) : ℝ := a (n + 1) + a (n + 2)

/-- Definition of c_n -/
def c (a : ℕ → ℝ) (n : ℕ) : ℝ := a n + a (n + 3)

/-- Theorem: For a geometric sequence a, b_n ≤ c_n for all n -/
theorem b_le_c_for_geometric (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∀ n : ℕ, b a n ≤ c a n := by
  sorry

end b_le_c_for_geometric_l3659_365950


namespace parabola_transformation_sum_l3659_365945

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally -/
def translate (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
  , b := p.b - 2 * p.a * h
  , c := p.a * h^2 - p.b * h + p.c }

/-- Reflects a parabola about the x-axis -/
def reflect (p : Parabola) : Parabola :=
  { a := -p.a
  , b := -p.b
  , c := -p.c }

/-- Adds two parabolas coefficient-wise -/
def add (p q : Parabola) : Parabola :=
  { a := p.a + q.a
  , b := p.b + q.b
  , c := p.c + q.c }

theorem parabola_transformation_sum (p : Parabola) :
  let p1 := translate p 4
  let p2 := translate (reflect p) (-4)
  (add p1 p2).b = -16 * p.a ∧ (add p1 p2).a = 0 ∧ (add p1 p2).c = 0 := by
  sorry

#check parabola_transformation_sum

end parabola_transformation_sum_l3659_365945


namespace jake_sister_weight_ratio_l3659_365926

/-- Proves that the ratio of Jake's weight after losing 33 pounds to his sister's weight is 2:1 -/
theorem jake_sister_weight_ratio :
  let jakes_current_weight : ℕ := 113
  let combined_weight : ℕ := 153
  let weight_loss : ℕ := 33
  let jakes_new_weight : ℕ := jakes_current_weight - weight_loss
  let sisters_weight : ℕ := combined_weight - jakes_current_weight
  (jakes_new_weight : ℚ) / (sisters_weight : ℚ) = 2 / 1 :=
by sorry

end jake_sister_weight_ratio_l3659_365926


namespace range_of_a_for_two_integer_solutions_l3659_365920

/-- A system of inequalities has exactly two integer solutions -/
def has_two_integer_solutions (a : ℝ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧
    (↑x : ℝ)^2 - (↑x : ℝ) + a - a^2 < 0 ∧ (↑x : ℝ) + 2*a > 1 ∧
    (↑y : ℝ)^2 - (↑y : ℝ) + a - a^2 < 0 ∧ (↑y : ℝ) + 2*a > 1 ∧
    ∀ z : ℤ, z ≠ x → z ≠ y →
      ¬((↑z : ℝ)^2 - (↑z : ℝ) + a - a^2 < 0 ∧ (↑z : ℝ) + 2*a > 1)

/-- The range of a that satisfies the conditions -/
theorem range_of_a_for_two_integer_solutions :
  ∀ a : ℝ, has_two_integer_solutions a ↔ 1 < a ∧ a ≤ 2 :=
sorry

end range_of_a_for_two_integer_solutions_l3659_365920


namespace function_properties_l3659_365980

-- Define the properties of function f
def additive (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

def positive_for_positive (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x > 0

-- State the theorem
theorem function_properties (f : ℝ → ℝ) 
  (h_add : additive f) (h_pos : positive_for_positive f) : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x < f y) :=
by sorry

end function_properties_l3659_365980


namespace fan_sales_analysis_fan_sales_analysis_application_l3659_365940

/-- Represents the sales data for a week -/
structure WeeklySales where
  modelA : ℕ
  modelB : ℕ
  revenue : ℕ

/-- Represents the fan models and their prices -/
structure FanModels where
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceA : ℕ
  sellingPriceB : ℕ

/-- Represents the purchase constraints -/
structure PurchaseConstraints where
  totalUnits : ℕ
  maxBudget : ℕ

/-- Main theorem encompassing all parts of the problem -/
theorem fan_sales_analysis 
  (week1 : WeeklySales)
  (week2 : WeeklySales)
  (models : FanModels)
  (constraints : PurchaseConstraints) :
  (models.sellingPriceA = 200 ∧ models.sellingPriceB = 150) ∧
  (∀ m : ℕ, m ≤ 37 → m * models.purchasePriceA + (constraints.totalUnits - m) * models.purchasePriceB ≤ constraints.maxBudget) ∧
  (∃ m : ℕ, m ≤ 37 ∧ m * (models.sellingPriceA - models.purchasePriceA) + 
    (constraints.totalUnits - m) * (models.sellingPriceB - models.purchasePriceB) > 2850) :=
by
  sorry

/-- Given data for the problem -/
def problem_data : WeeklySales × WeeklySales × FanModels × PurchaseConstraints :=
  ({ modelA := 4, modelB := 3, revenue := 1250 },
   { modelA := 5, modelB := 5, revenue := 1750 },
   { purchasePriceA := 140, purchasePriceB := 100, sellingPriceA := 0, sellingPriceB := 0 },
   { totalUnits := 50, maxBudget := 6500 })

/-- Application of the main theorem to the given data -/
theorem fan_sales_analysis_application :
  let (week1, week2, models, constraints) := problem_data
  (models.sellingPriceA = 200 ∧ models.sellingPriceB = 150) ∧
  (∀ m : ℕ, m ≤ 37 → m * models.purchasePriceA + (constraints.totalUnits - m) * models.purchasePriceB ≤ constraints.maxBudget) ∧
  (∃ m : ℕ, m ≤ 37 ∧ m * (models.sellingPriceA - models.purchasePriceA) + 
    (constraints.totalUnits - m) * (models.sellingPriceB - models.purchasePriceB) > 2850) :=
by
  sorry

end fan_sales_analysis_fan_sales_analysis_application_l3659_365940


namespace rachels_math_homework_l3659_365929

/-- Rachel's homework problem -/
theorem rachels_math_homework (reading_pages : ℕ) (extra_math_pages : ℕ) : 
  reading_pages = 4 → extra_math_pages = 3 → reading_pages + extra_math_pages = 7 :=
by
  sorry

end rachels_math_homework_l3659_365929


namespace problem_statement_l3659_365997

theorem problem_statement (x y : ℝ) (h1 : 2 * x + 2 * y = 10) (h2 : x * y = -15) :
  4 * x^2 + 4 * y^2 = 220 := by
  sorry

end problem_statement_l3659_365997


namespace twentieth_number_in_twentieth_row_l3659_365960

/-- Calculates the first number in a given row of the triangular sequence -/
def first_number_in_row (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the kth number in the nth row of the triangular sequence -/
def number_in_sequence (n k : ℕ) : ℕ := first_number_in_row n + (k - 1)

/-- The 20th number in the 20th row of the triangular sequence is 381 -/
theorem twentieth_number_in_twentieth_row :
  number_in_sequence 20 20 = 381 := by sorry

end twentieth_number_in_twentieth_row_l3659_365960


namespace diagonal_cubes_120_270_300_l3659_365983

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def internal_diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd x z) + Nat.gcd x (Nat.gcd y z)

/-- The number of cubes a face diagonal passes through in a rectangular solid -/
def face_diagonal_cubes (x y : ℕ) : ℕ :=
  x + y - Nat.gcd x y

/-- Theorem about the number of cubes diagonals pass through in a 120 × 270 × 300 rectangular solid -/
theorem diagonal_cubes_120_270_300 :
  internal_diagonal_cubes 120 270 300 = 600 ∧
  face_diagonal_cubes 120 270 = 360 := by
  sorry


end diagonal_cubes_120_270_300_l3659_365983


namespace line_equation_through_point_with_angle_l3659_365923

/-- The equation of a line passing through a given point with a given angle -/
theorem line_equation_through_point_with_angle 
  (x₀ y₀ : ℝ) (θ : ℝ) :
  x₀ = Real.sqrt 3 →
  y₀ = -2 * Real.sqrt 3 →
  θ = 135 * π / 180 →
  ∃ (A B C : ℝ), A * x₀ + B * y₀ + C = 0 ∧
                 A * x + B * y + C = 0 ∧
                 A = 1 ∧ B = 1 ∧ C = Real.sqrt 3 :=
by sorry

end line_equation_through_point_with_angle_l3659_365923


namespace two_roots_implies_c_values_l3659_365921

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem two_roots_implies_c_values (c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f c x = 0 ∧ f c y = 0 ∧ ∀ z : ℝ, f c z = 0 → z = x ∨ z = y) →
  c = -2 ∨ c = 2 := by
  sorry

end two_roots_implies_c_values_l3659_365921


namespace ceiling_square_minus_fraction_l3659_365949

theorem ceiling_square_minus_fraction : ⌈((-7/4)^2 - 1/8)⌉ = 3 := by sorry

end ceiling_square_minus_fraction_l3659_365949


namespace not_p_sufficient_not_necessary_for_not_q_l3659_365939

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| < 2
def q (x : ℝ) : Prop := x^2 < 2 - x

-- Define the relationship between ¬p and ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∃ x : ℝ, ¬(p x) → ¬(q x)) ∧ 
  (∃ x : ℝ, ¬(q x) ∧ p x) := by
  sorry

end not_p_sufficient_not_necessary_for_not_q_l3659_365939


namespace no_real_solutions_for_square_rectangle_area_relation_l3659_365905

theorem no_real_solutions_for_square_rectangle_area_relation :
  ¬ ∃ x : ℝ, (x + 2) * (x - 5) = 2 * (x - 1)^2 :=
by sorry

end no_real_solutions_for_square_rectangle_area_relation_l3659_365905


namespace part_one_part_two_l3659_365914

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  t.a * Real.sin t.A + t.c * Real.sin t.C - Real.sqrt 2 * t.a * Real.sin t.C = t.b * Real.sin t.B

-- Theorem for part (I)
theorem part_one (t : Triangle) (h : given_condition t) : t.B = Real.pi / 4 := by
  sorry

-- Theorem for part (II)
theorem part_two (t : Triangle) (h1 : given_condition t) (h2 : t.A = 5 * Real.pi / 12) (h3 : t.b = 2) :
  t.a = 1 + Real.sqrt 3 ∧ t.c = Real.sqrt 6 := by
  sorry

end part_one_part_two_l3659_365914


namespace truncated_cone_height_l3659_365985

/-- The height of a circular truncated cone with given top and bottom surface areas and volume. -/
theorem truncated_cone_height (S₁ S₂ V : ℝ) (h : ℝ) 
    (hS₁ : S₁ = 4 * Real.pi)
    (hS₂ : S₂ = 9 * Real.pi)
    (hV : V = 19 * Real.pi)
    (h_def : V = (1/3) * h * (S₁ + Real.sqrt (S₁ * S₂) + S₂)) :
  h = 3 := by
  sorry

#check truncated_cone_height

end truncated_cone_height_l3659_365985


namespace total_insects_theorem_l3659_365910

/-- The number of geckos -/
def num_geckos : ℕ := 5

/-- The number of insects eaten by each gecko -/
def insects_per_gecko : ℕ := 6

/-- The number of lizards -/
def num_lizards : ℕ := 3

/-- The number of insects eaten by each lizard -/
def insects_per_lizard : ℕ := 2 * insects_per_gecko

/-- The total number of insects eaten by both geckos and lizards -/
def total_insects_eaten : ℕ := num_geckos * insects_per_gecko + num_lizards * insects_per_lizard

theorem total_insects_theorem : total_insects_eaten = 66 := by
  sorry

end total_insects_theorem_l3659_365910


namespace store_pricing_strategy_l3659_365956

/-- Calculates the marked price as a percentage of the list price given the purchase discount,
    selling discount, and desired profit percentage. -/
def markedPricePercentage (purchaseDiscount sellingDiscount profitPercentage : ℚ) : ℚ :=
  let costPrice := 1 - purchaseDiscount
  let markupFactor := (1 + profitPercentage) / (1 - sellingDiscount)
  costPrice * markupFactor * 100

/-- Theorem stating that under the given conditions, the marked price should be 121.⅓% of the list price -/
theorem store_pricing_strategy :
  markedPricePercentage (30/100) (25/100) (30/100) = 121 + 1/3 := by
  sorry

end store_pricing_strategy_l3659_365956


namespace pen_cost_calculation_l3659_365916

/-- Calculates the cost of each pen given the initial amount, notebook cost, and remaining amount --/
theorem pen_cost_calculation (initial_amount : ℚ) (notebook_cost : ℚ) (num_notebooks : ℕ) 
  (num_pens : ℕ) (remaining_amount : ℚ) : 
  initial_amount = 15 → 
  notebook_cost = 4 → 
  num_notebooks = 2 → 
  num_pens = 2 → 
  remaining_amount = 4 → 
  (initial_amount - remaining_amount - (num_notebooks : ℚ) * notebook_cost) / (num_pens : ℚ) = 1.5 := by
  sorry

#eval (15 : ℚ) - 4 - 2 * 4

#eval ((15 : ℚ) - 4 - 2 * 4) / 2

end pen_cost_calculation_l3659_365916


namespace mrs_hilt_friends_l3659_365967

/-- Mrs. Hilt's friends problem -/
theorem mrs_hilt_friends (friends_can_go : ℕ) (friends_cant_go : ℕ) 
  (h1 : friends_can_go = 8) (h2 : friends_cant_go = 7) : 
  friends_can_go + friends_cant_go = 15 := by
  sorry

end mrs_hilt_friends_l3659_365967


namespace train_length_l3659_365993

/-- Proves that a train moving at 40 kmph and passing a telegraph post in 7.199424046076314 seconds has a length of 80 meters. -/
theorem train_length (speed_kmph : ℝ) (time_seconds : ℝ) (length_meters : ℝ) : 
  speed_kmph = 40 →
  time_seconds = 7.199424046076314 →
  length_meters = speed_kmph * 1000 / 3600 * time_seconds →
  length_meters = 80 := by
  sorry

#check train_length

end train_length_l3659_365993


namespace train_length_calculation_l3659_365975

/-- Calculates the length of a train given the length of another train, their speeds, and the time they take to cross each other when traveling in opposite directions. -/
theorem train_length_calculation (length1 : ℝ) (speed1 speed2 : ℝ) (cross_time : ℝ) :
  length1 = 270 →
  speed1 = 120 →
  speed2 = 80 →
  cross_time = 9 →
  ∃ (length2 : ℝ), abs (length2 - 230.04) < 0.01 :=
by
  sorry

end train_length_calculation_l3659_365975


namespace mapping_has_output_l3659_365902

-- Define sets M and N
variable (M N : Type)

-- Define the mapping f from M to N
variable (f : M → N)

-- Theorem statement
theorem mapping_has_output : ∀ (x : M), ∃ (y : N), f x = y := by
  sorry

end mapping_has_output_l3659_365902
