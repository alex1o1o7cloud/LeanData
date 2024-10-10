import Mathlib

namespace positive_numbers_inequality_l2896_289693

theorem positive_numbers_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hnot_all_equal : ¬(a = b ∧ b = c)) : 
  ((a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0) ∧ 
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) :=
by sorry

end positive_numbers_inequality_l2896_289693


namespace max_difference_reversed_digits_l2896_289685

/-- Two-digit positive integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem max_difference_reversed_digits (q r : ℕ) :
  TwoDigitInt q ∧ TwoDigitInt r ∧
  r = reverseDigits q ∧
  (q > r → q - r < 20) ∧
  (r > q → r - q < 20) →
  (q > r → q - r ≤ 18) ∧
  (r > q → r - q ≤ 18) :=
sorry

end max_difference_reversed_digits_l2896_289685


namespace only_undergraduateGraduates2013_is_well_defined_set_l2896_289633

-- Define the universe of discourse
def Universe : Type := Set (Nat → Bool)

-- Define the options
def undergraduateGraduates2013 : Universe := sorry
def highWheatProductionCities2013 : Universe := sorry
def famousMathematicians : Universe := sorry
def numbersCloseToPI : Universe := sorry

-- Define a predicate for well-defined sets
def isWellDefinedSet (S : Universe) : Prop := sorry

-- Theorem statement
theorem only_undergraduateGraduates2013_is_well_defined_set :
  isWellDefinedSet undergraduateGraduates2013 ∧
  ¬isWellDefinedSet highWheatProductionCities2013 ∧
  ¬isWellDefinedSet famousMathematicians ∧
  ¬isWellDefinedSet numbersCloseToPI :=
sorry

end only_undergraduateGraduates2013_is_well_defined_set_l2896_289633


namespace geometric_subsequence_ratio_l2896_289692

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_d_nonzero : d ≠ 0

/-- The property that a_1, a_3, and a_7 form a geometric sequence -/
def IsGeometricSubsequence (seq : ArithmeticSequence) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ seq.a 3 = seq.a 1 * q ∧ seq.a 7 = seq.a 3 * q

/-- The theorem stating that the common ratio of the geometric subsequence is 2 -/
theorem geometric_subsequence_ratio (seq : ArithmeticSequence) 
  (h_geom : IsGeometricSubsequence seq) : 
  ∃ q : ℝ, q = 2 ∧ seq.a 3 = seq.a 1 * q ∧ seq.a 7 = seq.a 3 * q := by
  sorry

end geometric_subsequence_ratio_l2896_289692


namespace parabola_equation_l2896_289694

/-- A parabola with vertex at the origin and axis of symmetry x = -4 has the standard equation y^2 = 16x -/
theorem parabola_equation (y x : ℝ) : 
  (∀ p : ℝ, p > 0 → y^2 = 2*p*x) → -- Standard form of parabola equation
  (∀ p : ℝ, -p/2 = -4) →           -- Axis of symmetry condition
  y^2 = 16*x :=                    -- Conclusion: standard equation
by sorry

end parabola_equation_l2896_289694


namespace magnitude_of_complex_fourth_power_l2896_289622

theorem magnitude_of_complex_fourth_power : 
  Complex.abs ((5 - 2 * Complex.I * Real.sqrt 3) ^ 4) = 1369 := by
  sorry

end magnitude_of_complex_fourth_power_l2896_289622


namespace magnitude_of_2a_plus_b_l2896_289616

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (0, -1, 1)
def b : ℝ × ℝ × ℝ := (1, 0, 1)

-- Define the operation 2a + b
def result : ℝ × ℝ × ℝ := (2 * a.1 + b.1, 2 * a.2.1 + b.2.1, 2 * a.2.2 + b.2.2)

-- Theorem statement
theorem magnitude_of_2a_plus_b : 
  Real.sqrt ((result.1)^2 + (result.2.1)^2 + (result.2.2)^2) = Real.sqrt 14 :=
by
  sorry

end magnitude_of_2a_plus_b_l2896_289616


namespace shortest_path_in_sqrt2_octahedron_l2896_289636

/-- A regular octahedron -/
structure RegularOctahedron where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

/-- The shortest path between midpoints of non-adjacent edges -/
def shortestPathBetweenMidpoints (o : RegularOctahedron) : ℝ :=
  sorry

/-- Theorem: In a regular octahedron with edge length √2, the shortest path
    between midpoints of non-adjacent edges is √2 -/
theorem shortest_path_in_sqrt2_octahedron :
  let o : RegularOctahedron := ⟨ Real.sqrt 2, sorry ⟩
  shortestPathBetweenMidpoints o = Real.sqrt 2 := by
  sorry

end shortest_path_in_sqrt2_octahedron_l2896_289636


namespace harrys_annual_pet_feeding_cost_l2896_289649

/-- Calculates the annual cost of feeding pets given the number of each type and their monthly feeding costs. -/
def annual_pet_feeding_cost (num_geckos num_iguanas num_snakes : ℕ) 
                            (gecko_cost iguana_cost snake_cost : ℕ) : ℕ :=
  12 * (num_geckos * gecko_cost + num_iguanas * iguana_cost + num_snakes * snake_cost)

/-- Theorem stating that Harry's annual pet feeding cost is $1140. -/
theorem harrys_annual_pet_feeding_cost : 
  annual_pet_feeding_cost 3 2 4 15 5 10 = 1140 := by
  sorry

end harrys_annual_pet_feeding_cost_l2896_289649


namespace valid_combinations_for_elixir_l2896_289627

/-- Represents the number of different magical roots. -/
def num_roots : ℕ := 4

/-- Represents the number of different mystical minerals. -/
def num_minerals : ℕ := 6

/-- Represents the number of minerals incompatible with one root. -/
def minerals_incompatible_with_one_root : ℕ := 2

/-- Represents the number of roots incompatible with one mineral. -/
def roots_incompatible_with_one_mineral : ℕ := 2

/-- Represents the total number of incompatible combinations. -/
def total_incompatible_combinations : ℕ :=
  minerals_incompatible_with_one_root + roots_incompatible_with_one_mineral

/-- Theorem stating the number of valid combinations for the wizard's elixir. -/
theorem valid_combinations_for_elixir :
  num_roots * num_minerals - total_incompatible_combinations = 20 := by
  sorry

end valid_combinations_for_elixir_l2896_289627


namespace intersection_of_three_lines_l2896_289613

/-- Given three lines that intersect at the same point, prove the value of m -/
theorem intersection_of_three_lines (x y : ℝ) (m : ℝ) :
  (y = 4 * x + 2) ∧ 
  (y = -3 * x - 18) ∧ 
  (y = 2 * x + m) →
  m = -26 / 7 := by
  sorry

end intersection_of_three_lines_l2896_289613


namespace angle_bisector_ratio_l2896_289615

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define points D and E on AB and AC respectively
def D (triangle : Triangle) : ℝ × ℝ := sorry
def E (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the angle bisector AT
def T (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the intersection point F of AT and DE
def F (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the lengths
def AD (triangle : Triangle) : ℝ := 2
def DB (triangle : Triangle) : ℝ := 6
def AE (triangle : Triangle) : ℝ := 5
def EC (triangle : Triangle) : ℝ := 3

-- Define the ratio AF/AT
def AF_AT_ratio (triangle : Triangle) : ℝ := sorry

-- Theorem statement
theorem angle_bisector_ratio (triangle : Triangle) :
  AF_AT_ratio triangle = 2/5 := by sorry

end angle_bisector_ratio_l2896_289615


namespace solution_product_theorem_l2896_289678

theorem solution_product_theorem (a b : ℝ) : 
  a ≠ b → 
  (a^4 + a^3 - 1 = 0) → 
  (b^4 + b^3 - 1 = 0) → 
  ((a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0) := by
sorry

end solution_product_theorem_l2896_289678


namespace min_sum_a_b_l2896_289641

theorem min_sum_a_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 2 / b = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 := by
sorry

end min_sum_a_b_l2896_289641


namespace positive_A_value_l2896_289659

/-- The relation # is defined as A # B = A^2 + B^2 -/
def hash (A B : ℝ) : ℝ := A^2 + B^2

/-- Given A # 7 = 194, prove that the positive value of A is √145 -/
theorem positive_A_value (h : hash A 7 = 194) : A = Real.sqrt 145 := by
  sorry

end positive_A_value_l2896_289659


namespace hyperbola_intersection_ratio_difference_l2896_289695

/-- The hyperbola with equation x²/2 - y²/2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2/2 - y^2/2 = 1

/-- The right branch of the hyperbola -/
def right_branch (x y : ℝ) : Prop := hyperbola x y ∧ x > 0

/-- Point P lies on the right branch of the hyperbola -/
def P_on_right_branch (P : ℝ × ℝ) : Prop := right_branch P.1 P.2

/-- Point A is the intersection of PF₁ and the hyperbola -/
def A_is_intersection (P A : ℝ × ℝ) : Prop :=
  hyperbola A.1 A.2 ∧ ∃ t : ℝ, A = (t * (P.1 + 2) - 2, t * P.2)

/-- Point B is the intersection of PF₂ and the hyperbola -/
def B_is_intersection (P B : ℝ × ℝ) : Prop :=
  hyperbola B.1 B.2 ∧ ∃ t : ℝ, B = (t * (P.1 - 2) + 2, t * P.2)

/-- The main theorem -/
theorem hyperbola_intersection_ratio_difference (P A B : ℝ × ℝ) :
  P_on_right_branch P →
  A_is_intersection P A →
  B_is_intersection P B →
  ∃ (PF₁ AF₁ PF₂ BF₂ : ℝ),
    PF₁ / AF₁ - PF₂ / BF₂ = 6 :=
sorry

end hyperbola_intersection_ratio_difference_l2896_289695


namespace square_difference_equality_l2896_289611

theorem square_difference_equality : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end square_difference_equality_l2896_289611


namespace remainder_problem_l2896_289696

theorem remainder_problem (n a b c : ℕ) (hn : 0 < n) 
  (ha : n % 3 = a) (hb : n % 5 = b) (hc : n % 7 = c) 
  (heq : 4 * a + 3 * b + 2 * c = 30) : 
  n % 105 = 29 := by
sorry

end remainder_problem_l2896_289696


namespace gcd_of_powers_of_101_l2896_289657

theorem gcd_of_powers_of_101 : Nat.gcd (101^11 + 1) (101^11 + 101^3 + 1) = 1 := by
  sorry

end gcd_of_powers_of_101_l2896_289657


namespace nested_bracket_evaluation_l2896_289690

def bracket (a b c : ℚ) : ℚ := (a + b) / c

theorem nested_bracket_evaluation :
  let outer_bracket := bracket (bracket 72 36 108) (bracket 4 2 6) (bracket 12 6 18)
  outer_bracket = 2 := by sorry

end nested_bracket_evaluation_l2896_289690


namespace unique_solution_condition_l2896_289643

/-- The equation (3x+5)(x-3) = -55 + kx has exactly one real solution if and only if k = 18 or k = -26 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x + 5)*(x - 3) = -55 + k*x) ↔ (k = 18 ∨ k = -26) := by
  sorry

end unique_solution_condition_l2896_289643


namespace cos_plus_sin_value_l2896_289637

theorem cos_plus_sin_value (α : Real) (k : Real) :
  (∃ x y : Real, x * y = 1 ∧ 
    x^2 - k*x + k^2 - 3 = 0 ∧ 
    y^2 - k*y + k^2 - 3 = 0 ∧ 
    x = Real.tan α ∧ 
    y = 1 / Real.tan α) →
  3 * Real.pi < α ∧ α < 7/2 * Real.pi →
  Real.cos α + Real.sin α = -Real.sqrt 2 := by
sorry

end cos_plus_sin_value_l2896_289637


namespace parabola_decreasing_right_of_axis_l2896_289635

-- Define the parabola function
def f (b c x : ℝ) : ℝ := -x^2 + b*x + c

-- State the theorem
theorem parabola_decreasing_right_of_axis (b c : ℝ) :
  (∀ x, f b c x = f b c (6 - x)) →  -- Axis of symmetry at x = 3
  ∀ x > 3, ∀ y > x, f b c y < f b c x :=
sorry

end parabola_decreasing_right_of_axis_l2896_289635


namespace tim_and_linda_mowing_time_l2896_289673

/-- The time it takes for two people to complete a task together, given their individual rates -/
def combined_time (rate1 rate2 : ℚ) : ℚ :=
  1 / (rate1 + rate2)

/-- Proof that Tim and Linda can mow the lawn together in 6/7 hours -/
theorem tim_and_linda_mowing_time :
  let tim_rate : ℚ := 1 / (3/2)  -- Tim's rate: 1 lawn per 1.5 hours
  let linda_rate : ℚ := 1 / 2    -- Linda's rate: 1 lawn per 2 hours
  combined_time tim_rate linda_rate = 6/7 := by
  sorry

#eval (combined_time (1 / (3/2)) (1 / 2))

end tim_and_linda_mowing_time_l2896_289673


namespace messenger_speed_l2896_289609

/-- Messenger speed problem -/
theorem messenger_speed (team_length : ℝ) (team_speed : ℝ) (total_time : ℝ) :
  team_length = 6 →
  team_speed = 5 →
  total_time = 0.5 →
  ∃ messenger_speed : ℝ,
    messenger_speed > 0 ∧
    (team_length / (messenger_speed + team_speed) + team_length / (messenger_speed - team_speed) = total_time) ∧
    messenger_speed = 25 := by
  sorry

end messenger_speed_l2896_289609


namespace product_of_roots_l2896_289667

theorem product_of_roots (t : ℝ) : 
  let equation := fun t : ℝ => 18 * t^2 + 45 * t - 500
  let product_of_roots := -500 / 18
  product_of_roots = -250 / 9 := by
sorry

end product_of_roots_l2896_289667


namespace pie_chart_proportions_l2896_289604

theorem pie_chart_proportions :
  ∀ (white black gray blue : ℚ),
    white = 3 * black →
    black = 2 * gray →
    blue = gray →
    white + black + gray + blue = 1 →
    white = 3/5 ∧ black = 1/5 ∧ gray = 1/10 ∧ blue = 1/10 := by
  sorry

end pie_chart_proportions_l2896_289604


namespace min_value_cyclic_fraction_l2896_289675

theorem min_value_cyclic_fraction (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a / b + b / c + c / d + d / a ≥ 4 ∧ 
  (a / b + b / c + c / d + d / a = 4 ↔ a = b ∧ b = c ∧ c = d) := by
  sorry

end min_value_cyclic_fraction_l2896_289675


namespace travel_cost_theorem_l2896_289621

structure City where
  name : String

structure Triangle where
  D : City
  E : City
  F : City
  DE : ℝ
  EF : ℝ
  FD : ℝ
  right_angle_at_D : DE^2 + FD^2 = EF^2

def bus_fare_per_km : ℝ := 0.20

def plane_booking_fee (departure : City) : ℝ :=
  if departure.name = "E" then 150 else 120

def plane_fare_per_km : ℝ := 0.12

def travel_cost (t : Triangle) : ℝ :=
  t.DE * bus_fare_per_km +
  t.EF * plane_fare_per_km +
  plane_booking_fee t.E

theorem travel_cost_theorem (t : Triangle) :
  t.DE = 4000 ∧ t.EF = 4500 ∧ t.FD = 5000 →
  travel_cost t = 1490 := by
  sorry

end travel_cost_theorem_l2896_289621


namespace rectangle_x_satisfies_conditions_rectangle_x_unique_solution_l2896_289626

/-- The value of x for a rectangle with specific properties -/
def rectangle_x : ℝ := 1.924

/-- The length of the rectangle -/
def length (x : ℝ) : ℝ := 5 * x

/-- The width of the rectangle -/
def width (x : ℝ) : ℝ := 2 * x + 3

/-- The area of the rectangle -/
def area (x : ℝ) : ℝ := length x * width x

/-- The perimeter of the rectangle -/
def perimeter (x : ℝ) : ℝ := 2 * (length x + width x)

/-- Theorem stating that rectangle_x satisfies the given conditions -/
theorem rectangle_x_satisfies_conditions :
  area rectangle_x = 2 * perimeter rectangle_x ∧
  length rectangle_x > 0 ∧
  width rectangle_x > 0 := by
  sorry

/-- Theorem stating that rectangle_x is the unique solution -/
theorem rectangle_x_unique_solution :
  ∀ y : ℝ, (area y = 2 * perimeter y ∧ length y > 0 ∧ width y > 0) → y = rectangle_x := by
  sorry

end rectangle_x_satisfies_conditions_rectangle_x_unique_solution_l2896_289626


namespace sufficient_condition_l2896_289658

theorem sufficient_condition (a : ℝ) : a ≥ 0 → a^2 + a ≥ 0 := by sorry

end sufficient_condition_l2896_289658


namespace cycle_original_price_l2896_289605

/-- Proves that given a cycle sold for 1260 with a 40% gain, the original price was 900 --/
theorem cycle_original_price (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 1260)
  (h2 : gain_percentage = 40) : 
  (selling_price / (1 + gain_percentage / 100)) = 900 := by
  sorry

end cycle_original_price_l2896_289605


namespace system_solutions_l2896_289647

/-- The system of equations -/
def system (x₁ x₂ x₃ x₄ x₅ y : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₃

/-- The solutions to the system of equations -/
theorem system_solutions :
  ∀ x₁ x₂ x₃ x₄ x₅ y : ℝ,
  system x₁ x₂ x₃ x₄ x₅ y →
  (y = 2 ∧ x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅) ∨
  ((y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) ∧
   x₂ = y * x₁ ∧ x₃ = y * x₂ ∧ x₄ = y * x₃) :=
by sorry

end system_solutions_l2896_289647


namespace mistaken_calculation_l2896_289687

theorem mistaken_calculation (x : ℕ) : 
  423 - x = 421 → (423 * x) + 421 - 500 = 767 := by
  sorry

end mistaken_calculation_l2896_289687


namespace max_ratio_square_extension_l2896_289619

/-- Given a square ABCD with side length a, the ratio MA:MB is maximized 
    when M is positioned on the extension of CD such that MC = 2a / (1 + √5) -/
theorem max_ratio_square_extension (a : ℝ) (h : a > 0) :
  let square := {A : ℝ × ℝ | A.1 ∈ [0, a] ∧ A.2 ∈ [0, a]}
  let C := (a, 0)
  let D := (a, a)
  let M (x : ℝ) := (a + x, 0)
  let ratio (x : ℝ) := ‖M x - (0, a)‖ / ‖M x - (a, a)‖
  ∃ (x_max : ℝ), x_max = 2 * a / (1 + Real.sqrt 5) ∧
    ∀ (x : ℝ), x > 0 → ratio x ≤ ratio x_max :=
by
  sorry


end max_ratio_square_extension_l2896_289619


namespace modulo_31_problem_l2896_289651

theorem modulo_31_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ 81256 ≡ n [ZMOD 31] ∧ n = 16 := by
  sorry

end modulo_31_problem_l2896_289651


namespace jumping_competition_result_l2896_289684

/-- The difference in average jumps per minute between two competitors -/
def jump_difference (total_time : ℕ) (jumps_a : ℕ) (jumps_b : ℕ) : ℚ :=
  (jumps_a - jumps_b : ℚ) / total_time

theorem jumping_competition_result :
  jump_difference 5 480 420 = 12 := by
  sorry

end jumping_competition_result_l2896_289684


namespace water_level_rise_l2896_289620

/-- Calculates the rise in water level when a cube is immersed in a rectangular vessel -/
theorem water_level_rise
  (cube_edge : ℝ)
  (vessel_length : ℝ)
  (vessel_width : ℝ)
  (h_cube_edge : cube_edge = 10)
  (h_vessel_length : vessel_length = 20)
  (h_vessel_width : vessel_width = 15) :
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 10/3 :=
by sorry

end water_level_rise_l2896_289620


namespace farm_count_solution_l2896_289628

/-- Represents the count of animals in a farm -/
structure FarmCount where
  hens : ℕ
  cows : ℕ

/-- Checks if the given farm count satisfies the conditions -/
def isValidFarmCount (f : FarmCount) : Prop :=
  f.hens + f.cows = 46 ∧ 2 * f.hens + 4 * f.cows = 140

/-- Theorem stating that the farm with 22 hens satisfies the conditions -/
theorem farm_count_solution :
  ∃ (f : FarmCount), isValidFarmCount f ∧ f.hens = 22 := by
  sorry

#check farm_count_solution

end farm_count_solution_l2896_289628


namespace tan_product_pi_ninths_l2896_289610

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end tan_product_pi_ninths_l2896_289610


namespace sqrt_three_difference_of_squares_l2896_289603

theorem sqrt_three_difference_of_squares : (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) = 2 := by
  sorry

end sqrt_three_difference_of_squares_l2896_289603


namespace circle_equation_l2896_289662

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 2)

-- Define the radius of the circle
def radius : ℝ := 4

-- State the theorem
theorem circle_equation :
  ∀ (x y : ℝ), ((x + 1)^2 + (y - 2)^2 = 16) ↔ 
  ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end circle_equation_l2896_289662


namespace square_of_real_not_always_positive_l2896_289686

theorem square_of_real_not_always_positive : ¬ (∀ a : ℝ, a^2 > 0) := by
  sorry

end square_of_real_not_always_positive_l2896_289686


namespace minimum_m_value_l2896_289640

theorem minimum_m_value (m : ℕ) : 
  (∀ n : ℕ, n ≥ 2 → (n.factorial : ℝ) ^ (2 / (n * (n - 1))) < m) ↔ m ≥ 3 :=
by sorry

end minimum_m_value_l2896_289640


namespace rod_length_difference_l2896_289638

theorem rod_length_difference (L₁ L₂ : ℝ) : 
  L₁ + L₂ = 33 →
  (1 - 1/3) * L₁ = (1 - 1/5) * L₂ →
  L₁ - L₂ = 3 := by
sorry

end rod_length_difference_l2896_289638


namespace op_properties_l2896_289646

-- Define the @ operation
def op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Theorem statement
theorem op_properties :
  (op 1 (-2) = -8) ∧ 
  (∀ a b : ℝ, op a b = op b a) ∧
  (∀ a b : ℝ, a + b = 0 → op a a + op b b = 8 * a^2) := by
sorry

end op_properties_l2896_289646


namespace stating_distinguishable_triangles_l2896_289669

/-- Represents the number of colors available -/
def num_colors : ℕ := 8

/-- Represents the number of small triangles in the large triangle -/
def num_triangles : ℕ := 4

/-- 
Calculates the number of ways to color a large equilateral triangle 
made of 4 smaller triangles using 8 colors, where no adjacent triangles 
can have the same color.
-/
def count_colorings : ℕ := 
  num_colors * (num_colors - 1) * (num_colors - 2) * (num_colors - 3)

/-- 
Theorem stating that the number of distinguishable large equilateral triangles 
is equal to 1680.
-/
theorem distinguishable_triangles : count_colorings = 1680 := by
  sorry

end stating_distinguishable_triangles_l2896_289669


namespace isosceles_triangle_angle_l2896_289672

/-- Given an isosceles triangle ABC with AB = BC = a and AC = b, 
    if ax² - √2·bx + a = 0 has two real roots with absolute difference √2,
    then ∠ABC = 120° -/
theorem isosceles_triangle_angle (a b : ℝ) (u : ℝ) : 
  a > 0 → b > 0 →
  (∃ x y : ℝ, x ≠ y ∧ 
    a * x^2 - Real.sqrt 2 * b * x + a = 0 ∧ 
    a * y^2 - Real.sqrt 2 * b * y + a = 0 ∧
    |x - y| = Real.sqrt 2) →
  b^2 = 2 * a^2 * (1 - Real.cos u) →
  u = 120 * π / 180 := by
  sorry

end isosceles_triangle_angle_l2896_289672


namespace union_of_A_and_B_l2896_289617

def A : Set Nat := {1, 2, 4}
def B : Set Nat := {2, 4, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_of_A_and_B_l2896_289617


namespace replaced_person_weight_l2896_289629

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  new_person_weight - initial_count * average_increase

/-- Theorem stating the weight of the replaced person under the given conditions -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 6 93 = 45 := by
  sorry

end replaced_person_weight_l2896_289629


namespace ceiling_floor_sum_zero_l2896_289676

theorem ceiling_floor_sum_zero : 
  Int.ceil (7 / 3 : ℚ) + Int.floor (-(7 / 3) : ℚ) + 
  Int.ceil (4 / 5 : ℚ) + Int.floor (-(4 / 5) : ℚ) = 0 := by
sorry

end ceiling_floor_sum_zero_l2896_289676


namespace factorization_implies_c_value_l2896_289631

theorem factorization_implies_c_value (c : ℝ) :
  (∀ x : ℝ, x^2 + 3*x + c = (x + 1)*(x + 2)) → c = 2 := by
sorry

end factorization_implies_c_value_l2896_289631


namespace simone_apple_days_l2896_289607

theorem simone_apple_days (d : ℕ) : 
  (1/2 : ℚ) * d + (1/3 : ℚ) * 15 = 13 → d = 16 := by
  sorry

end simone_apple_days_l2896_289607


namespace triangle_area_product_l2896_289648

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ a * x + b * y = 6) → 
  (1/2 * (6/a) * (6/b) = 6) → a * b = 3 := by
sorry

end triangle_area_product_l2896_289648


namespace college_students_count_l2896_289666

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 175) : boys + girls = 455 := by
  sorry

end college_students_count_l2896_289666


namespace geometric_sequence_ratio_l2896_289660

/-- Given a geometric sequence {a_n} with a_2 = 2 and S_3 = 8, prove S_5 / a_3 = 11 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1)  -- Geometric sequence condition
  → a 2 = 2                           -- Given condition
  → S 3 = 8                           -- Given condition
  → S 5 / a 3 = 11 := by sorry

end geometric_sequence_ratio_l2896_289660


namespace solution_equation1_solution_equation2_l2896_289625

-- Define the equations
def equation1 (x : ℝ) : Prop := 4 - x = 3 * (2 - x)
def equation2 (x : ℝ) : Prop := (2 * x - 1) / 2 - (2 * x + 5) / 3 = (6 * x - 1) / 6 - 1

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1 := by sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -1.5 := by sorry

end solution_equation1_solution_equation2_l2896_289625


namespace parabola_line_intersection_length_l2896_289650

/-- Parabola represented by parametric equations x = 4t² and y = 4t -/
structure Parabola where
  t : ℝ
  x : ℝ := 4 * t^2
  y : ℝ := 4 * t

/-- Line with slope 1 passing through a point -/
structure Line where
  slope : ℝ := 1
  point : ℝ × ℝ

/-- Represents the intersection points of the line and the parabola -/
structure Intersection where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The focus of a parabola with equation y² = 4x -/
def focus : ℝ × ℝ := (1, 0)

theorem parabola_line_intersection_length 
  (p : Parabola) 
  (l : Line) 
  (i : Intersection) :
  l.point = focus → 
  (∃ t₁ t₂ : ℝ, 
    i.A = (4 * t₁^2, 4 * t₁) ∧ 
    i.B = (4 * t₂^2, 4 * t₂) ∧ 
    i.A.2 = l.slope * i.A.1 + (l.point.2 - l.slope * l.point.1) ∧
    i.B.2 = l.slope * i.B.1 + (l.point.2 - l.slope * l.point.1)) →
  Real.sqrt ((i.A.1 - i.B.1)^2 + (i.A.2 - i.B.2)^2) = 8 := by
  sorry

end parabola_line_intersection_length_l2896_289650


namespace vector_difference_magnitude_l2896_289652

/-- Given two vectors a and b in R^2, prove that their difference has magnitude 1 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (Real.cos (75 * π / 180), Real.sin (75 * π / 180)) →
  b = (Real.cos (15 * π / 180), Real.sin (15 * π / 180)) →
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1 := by
  sorry

end vector_difference_magnitude_l2896_289652


namespace symmetric_point_l2896_289679

/-- Given a point P(2,1) and a line x - y + 1 = 0, prove that the point Q(0,3) is symmetric to P with respect to the line. -/
theorem symmetric_point (P Q : ℝ × ℝ) (line : ℝ → ℝ → ℝ) : 
  P = (2, 1) → 
  Q = (0, 3) → 
  line x y = x - y + 1 →
  (Q.1 - P.1) * (Q.2 - P.2) = -1 ∧ 
  line ((P.1 + Q.1) / 2) ((P.2 + Q.2) / 2) = 0 :=
sorry


end symmetric_point_l2896_289679


namespace bottles_taken_back_l2896_289688

/-- The number of bottles Debby takes back home is equal to the number of bottles she brought minus the number of bottles drunk. -/
theorem bottles_taken_back (bottles_brought bottles_drunk : ℕ) :
  bottles_brought ≥ bottles_drunk →
  bottles_brought - bottles_drunk = bottles_brought - bottles_drunk :=
by sorry

end bottles_taken_back_l2896_289688


namespace course_size_l2896_289644

theorem course_size (total : ℕ) 
  (h1 : total / 5 + total / 4 + total / 2 + 40 = total) : total = 800 := by
  sorry

end course_size_l2896_289644


namespace trigonometric_identities_l2896_289614

theorem trigonometric_identities (θ : Real) 
  (h : (4 * Real.sin θ - 2 * Real.cos θ) / (3 * Real.sin θ + 5 * Real.cos θ) = 6 / 11) : 
  (Real.tan θ = 2) ∧ 
  ((5 * (Real.cos θ)^2) / (Real.sin (2*θ) + 2 * Real.sin θ * Real.cos θ - 3 * (Real.cos θ)^2) = 1) ∧ 
  (1 - 4 * Real.sin θ * Real.cos θ + 2 * (Real.cos θ)^2 = -1/5) := by
sorry

end trigonometric_identities_l2896_289614


namespace total_pages_read_l2896_289698

-- Define the book's properties
def total_pages : ℕ := 95
def total_chapters : ℕ := 8

-- Define Jake's reading
def initial_pages_read : ℕ := 37
def additional_pages_read : ℕ := 25

-- Theorem to prove
theorem total_pages_read :
  initial_pages_read + additional_pages_read = 62 :=
by sorry

end total_pages_read_l2896_289698


namespace password_probability_l2896_289677

-- Define the set of possible last digits
def LastDigits : Finset Char := {'A', 'a', 'B', 'b'}

-- Define the set of possible second-to-last digits
def SecondLastDigits : Finset Nat := {4, 5, 6}

-- Define the type for a password
def Password := Nat × Char

-- Define the set of all possible passwords
def AllPasswords : Finset Password :=
  SecondLastDigits.product LastDigits

-- Theorem statement
theorem password_probability :
  (Finset.card AllPasswords : ℚ) = 12 ∧
  (1 : ℚ) / (Finset.card AllPasswords : ℚ) = 1 / 12 :=
sorry

end password_probability_l2896_289677


namespace age_ratio_is_two_to_one_l2896_289681

def james_age_3_years_ago : ℕ := 27
def matt_current_age : ℕ := 65
def years_since_james_27 : ℕ := 3
def years_to_future : ℕ := 5

def james_future_age : ℕ := james_age_3_years_ago + years_since_james_27 + years_to_future
def matt_future_age : ℕ := matt_current_age + years_to_future

theorem age_ratio_is_two_to_one :
  (matt_future_age : ℚ) / james_future_age = 2 := by
  sorry

end age_ratio_is_two_to_one_l2896_289681


namespace parallel_vectors_m_value_l2896_289624

/-- Two 2D vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (-1, 2*m + 1)
  parallel a b → m = -1/3 := by
  sorry

end parallel_vectors_m_value_l2896_289624


namespace height_growth_l2896_289623

theorem height_growth (current_height : ℝ) (growth_rate : ℝ) (previous_height : ℝ) : 
  current_height = 126 ∧ 
  growth_rate = 0.05 ∧ 
  current_height = previous_height * (1 + growth_rate) → 
  previous_height = 120 := by
sorry

end height_growth_l2896_289623


namespace union_of_A_and_B_l2896_289665

def A : Set Int := {1, 2}
def B : Set Int := {-1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by
  sorry

end union_of_A_and_B_l2896_289665


namespace marks_theater_cost_l2896_289674

/-- The cost of Mark's theater visits over a given number of weeks -/
def theater_cost (weeks : ℕ) (hours_per_visit : ℕ) (price_per_hour : ℕ) : ℕ :=
  weeks * hours_per_visit * price_per_hour

/-- Theorem: Mark's theater visits cost $90 over 6 weeks -/
theorem marks_theater_cost :
  theater_cost 6 3 5 = 90 := by
  sorry

end marks_theater_cost_l2896_289674


namespace harmonic_division_in_circumscribed_square_l2896_289682

-- Define the square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define the circle
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the configuration
structure Configuration where
  square : Square
  circle : Circle
  tangent : Line
  P : Point
  Q : Point
  R : Point
  S : Point

-- Define the property of being circumscribed
def is_circumscribed (s : Square) (c : Circle) : Prop :=
  s.side = 2 * c.radius ∧ s.center = c.center

-- Define the property of being a tangent
def is_tangent (l : Line) (c : Circle) : Prop :=
  ∃ (p : Point), p.x^2 + p.y^2 = c.radius^2 ∧ l.a * p.x + l.b * p.y + l.c = 0

-- Define the property of points being on the square or its extensions
def on_square_or_extension (p : Point) (s : Square) : Prop :=
  (p.x = s.center.1 - s.side/2 ∨ p.x = s.center.1 + s.side/2) ∨
  (p.y = s.center.2 - s.side/2 ∨ p.y = s.center.2 + s.side/2)

-- Define the harmonic division property
def harmonic_division (a : Point) (b : Point) (c : Point) (d : Point) : Prop :=
  (a.x - c.x) / (b.x - c.x) = (a.x - d.x) / (b.x - d.x)

-- Main theorem
theorem harmonic_division_in_circumscribed_square (cfg : Configuration) 
  (h1 : is_circumscribed cfg.square cfg.circle)
  (h2 : is_tangent cfg.tangent cfg.circle)
  (h3 : on_square_or_extension cfg.P cfg.square)
  (h4 : on_square_or_extension cfg.Q cfg.square)
  (h5 : on_square_or_extension cfg.R cfg.square)
  (h6 : on_square_or_extension cfg.S cfg.square) :
  harmonic_division cfg.P cfg.R cfg.Q cfg.S ∧ harmonic_division cfg.Q cfg.S cfg.P cfg.R :=
sorry

end harmonic_division_in_circumscribed_square_l2896_289682


namespace percentage_unsold_bags_l2896_289645

/-- Given the initial stock and daily sales of bags in a bookshop,
    prove that the percentage of unsold bags is 25%. -/
theorem percentage_unsold_bags
  (initial_stock : ℕ)
  (monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ)
  (h_initial : initial_stock = 600)
  (h_monday : monday_sales = 25)
  (h_tuesday : tuesday_sales = 70)
  (h_wednesday : wednesday_sales = 100)
  (h_thursday : thursday_sales = 110)
  (h_friday : friday_sales = 145) :
  (initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) / initial_stock * 100 = 25 := by
  sorry

end percentage_unsold_bags_l2896_289645


namespace last_two_digits_sum_factorials_l2896_289670

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials : ℕ → ℕ
  | 0 => 0
  | n + 1 => factorial (5 * n + 3) + sum_factorials n

theorem last_two_digits_sum_factorials :
  last_two_digits (sum_factorials 20) = 26 := by sorry

end last_two_digits_sum_factorials_l2896_289670


namespace karabases_more_numerous_l2896_289634

/-- Represents the inhabitants of Perra-Terra -/
inductive Inhabitant
  | Karabas
  | Barabas

/-- The number of acquaintances each type of inhabitant has -/
def acquaintances (i : Inhabitant) : Nat × Nat :=
  match i with
  | Inhabitant.Karabas => (6, 9)  -- (Karabases, Barabases)
  | Inhabitant.Barabas => (10, 7) -- (Karabases, Barabases)

theorem karabases_more_numerous :
  ∃ (K B : Nat), K > B ∧
  K * (acquaintances Inhabitant.Karabas).2 = B * (acquaintances Inhabitant.Barabas).1 :=
by sorry

end karabases_more_numerous_l2896_289634


namespace remainder_a_squared_minus_3b_l2896_289699

theorem remainder_a_squared_minus_3b (a b : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 5) 
  (h_ineq : a^2 > 3*b) : 
  (a^2 - 3*b) % 7 = 3 := by sorry

end remainder_a_squared_minus_3b_l2896_289699


namespace point_coordinates_product_l2896_289656

theorem point_coordinates_product (y₁ y₂ : ℝ) : 
  (((4 : ℝ) - 7)^2 + (y₁ - (-3))^2 = 13^2) →
  (((4 : ℝ) - 7)^2 + (y₂ - (-3))^2 = 13^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -151 := by
sorry

end point_coordinates_product_l2896_289656


namespace calculation_21_implies_72_l2896_289664

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The calculation process described in the problem -/
def calculation (n : TwoDigitNumber) : Nat :=
  2 * (5 * n.units - 3) + n.tens

/-- Theorem stating that if the calculation result is 21, the original number is 72 -/
theorem calculation_21_implies_72 (n : TwoDigitNumber) :
  calculation n = 21 → n.tens = 7 ∧ n.units = 2 := by
  sorry

#eval calculation ⟨7, 2, by norm_num, by norm_num⟩

end calculation_21_implies_72_l2896_289664


namespace target_miss_probability_l2896_289689

theorem target_miss_probability 
  (p_I p_II p_III : ℝ) 
  (h_I : p_I = 0.35) 
  (h_II : p_II = 0.30) 
  (h_III : p_III = 0.25) : 
  1 - (p_I + p_II + p_III) = 0.1 := by
sorry

end target_miss_probability_l2896_289689


namespace total_jars_is_24_l2896_289653

/-- Represents the number of each type of jar -/
def num_each_jar : ℕ := 8

/-- Represents the total volume of water in gallons -/
def total_water : ℕ := 14

/-- Represents the volume of water in quarts held by all quart jars -/
def quart_jars_volume : ℕ := num_each_jar

/-- Represents the volume of water in quarts held by all half-gallon jars -/
def half_gallon_jars_volume : ℕ := 2 * num_each_jar

/-- Represents the volume of water in quarts held by all one-gallon jars -/
def gallon_jars_volume : ℕ := 4 * num_each_jar

/-- Theorem stating that the total number of water-filled jars is 24 -/
theorem total_jars_is_24 : 
  quart_jars_volume + half_gallon_jars_volume + gallon_jars_volume = total_water * 4 ∧
  3 * num_each_jar = 24 := by
  sorry

#check total_jars_is_24

end total_jars_is_24_l2896_289653


namespace girls_in_class_l2896_289655

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h_total : total = 35) (h_ratio : ratio_girls = 3 ∧ ratio_boys = 4) :
  ∃ (girls : ℕ), girls * ratio_boys = (total - girls) * ratio_girls ∧ girls = 15 := by
sorry

end girls_in_class_l2896_289655


namespace odd_function_with_minimum_l2896_289680

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem odd_function_with_minimum (a b c d : ℝ) :
  (∀ x, f a b c d x = -f a b c d (-x)) →  -- f is an odd function
  (∀ x, f a b c d x ≥ f a b c d (-1)) →   -- f(-1) is the minimum value
  f a b c d (-1) = -1 →                   -- f(-1) = -1
  (∀ x, f a b c d x = -x^3 + x) :=        -- Conclusion: f(x) = -x³ + x
by
  sorry


end odd_function_with_minimum_l2896_289680


namespace optimal_sampling_methods_l2896_289612

structure Community where
  high_income : Nat
  middle_income : Nat
  low_income : Nat

structure Survey where
  sample_size : Nat
  population_size : Nat

inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

def survey1 : Survey := {
  sample_size := 100,
  population_size := 125 + 280 + 95
}

def survey2 : Survey := {
  sample_size := 3,
  population_size := 12
}

def community : Community := {
  high_income := 125,
  middle_income := 280,
  low_income := 95
}

def optimal_sampling_method (s : Survey) (c : Option Community) : SamplingMethod :=
  sorry

theorem optimal_sampling_methods :
  optimal_sampling_method survey1 (some community) = SamplingMethod.Stratified ∧
  optimal_sampling_method survey2 none = SamplingMethod.SimpleRandom :=
sorry

end optimal_sampling_methods_l2896_289612


namespace quadratic_equation_unique_solution_l2896_289618

theorem quadratic_equation_unique_solution (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃! x, a * x^2 + 12 * x + 9 = 0) :
  ∃ x, a * x^2 + 12 * x + 9 = 0 ∧ x = -3/2 := by
  sorry

end quadratic_equation_unique_solution_l2896_289618


namespace sqrt_equation_solution_l2896_289632

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 + Real.sqrt x) = 4 → x = 121 := by
  sorry

end sqrt_equation_solution_l2896_289632


namespace rectangle_with_hole_area_l2896_289668

theorem rectangle_with_hole_area (x : ℝ) :
  let large_length : ℝ := 2*x + 9
  let large_width : ℝ := x + 6
  let hole_side : ℝ := x - 1
  let large_area : ℝ := large_length * large_width
  let hole_area : ℝ := hole_side * hole_side
  large_area - hole_area = x^2 + 23*x + 53 :=
by sorry

end rectangle_with_hole_area_l2896_289668


namespace problem_solution_l2896_289606

theorem problem_solution (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + e^2 + 1 = d + Real.sqrt (a + b + c + e - 2*d)) : 
  d = -23/8 := by
  sorry

end problem_solution_l2896_289606


namespace a_in_N_necessary_not_sufficient_for_a_in_M_l2896_289639

def M : Set ℝ := {x | 0 < x ∧ x < 1}
def N : Set ℝ := {x | -2 < x ∧ x < 1}

theorem a_in_N_necessary_not_sufficient_for_a_in_M :
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by sorry

end a_in_N_necessary_not_sufficient_for_a_in_M_l2896_289639


namespace cone_volume_ratio_l2896_289608

/-- Given two sectors of a circle with central angles in the ratio 3:4, 
    the ratio of the volumes of the cones formed by rolling these sectors is 27:64 -/
theorem cone_volume_ratio (r : ℝ) (θ : ℝ) (h₁ h₂ : ℝ) :
  r > 0 → θ > 0 →
  3 * θ + 4 * θ = 2 * π →
  h₁ = Real.sqrt (r^2 - (3 * θ * r / (2 * π))^2) →
  h₂ = Real.sqrt (r^2 - (4 * θ * r / (2 * π))^2) →
  (1/3 * π * (3 * θ * r / (2 * π))^2 * h₁) / (1/3 * π * (4 * θ * r / (2 * π))^2 * h₂) = 27/64 :=
by sorry

#check cone_volume_ratio

end cone_volume_ratio_l2896_289608


namespace consecutive_product_square_extension_l2896_289642

theorem consecutive_product_square_extension (n : ℕ) (h : n * (n + 1) > 12) : 
  ∃! k : ℕ, k < 100 ∧ ∃ m : ℕ, 100 * (n * (n + 1)) + k = m^2 :=
sorry

end consecutive_product_square_extension_l2896_289642


namespace player_b_wins_in_five_l2896_289601

/-- The probability that Player B wins a best-of-five series in exactly 5 matches,
    given that Player A wins each match with probability 3/4 -/
theorem player_b_wins_in_five (p : ℚ) (h : p = 3/4) :
  let q := 1 - p
  let prob_tied_after_four := 6 * q^2 * p^2
  let prob_b_wins_fifth := q
  prob_tied_after_four * prob_b_wins_fifth = 27/512 :=
by sorry

end player_b_wins_in_five_l2896_289601


namespace fruit_tree_count_l2896_289600

/-- Proves that given 18 streets, with every other tree being a fruit tree,
    and equal numbers of three types of fruit trees,
    the number of each type of fruit tree is 3. -/
theorem fruit_tree_count (total_streets : ℕ) (fruit_tree_types : ℕ) : 
  total_streets = 18 → 
  fruit_tree_types = 3 → 
  (total_streets / 2) / fruit_tree_types = 3 :=
by
  sorry

end fruit_tree_count_l2896_289600


namespace sum_of_circle_areas_l2896_289663

/-- Given a 6-8-10 right triangle with vertices as centers of three mutually externally tangent circles,
    the sum of the areas of these circles is 56π. -/
theorem sum_of_circle_areas (a b c : ℝ) (r s t : ℝ) : 
  a = 6 ∧ b = 8 ∧ c = 10 ∧  -- Triangle side lengths
  a^2 + b^2 = c^2 ∧         -- Right triangle condition
  r + s = a ∧              -- Circles are externally tangent
  r + t = b ∧
  s + t = c ∧
  r > 0 ∧ s > 0 ∧ t > 0 →   -- Radii are positive
  π * (r^2 + s^2 + t^2) = 56 * π :=
by sorry

end sum_of_circle_areas_l2896_289663


namespace complex_equation_solution_l2896_289654

theorem complex_equation_solution (a b : ℝ) 
  (h : (a - 1 : ℂ) + a * I = 3 + 2 * b * I) : b = 2 := by
  sorry

end complex_equation_solution_l2896_289654


namespace equation_solution_l2896_289630

theorem equation_solution : 
  {x : ℝ | x + 36 / (x - 5) = -12} = {-8, 3} := by
  sorry

end equation_solution_l2896_289630


namespace not_always_input_start_output_end_l2896_289683

/-- Represents the types of boxes in a program flowchart -/
inductive FlowchartBox
  | Start
  | Input
  | Process
  | Output
  | End

/-- Represents a program flowchart as a list of boxes -/
def Flowchart := List FlowchartBox

/-- Checks if the input box immediately follows the start box -/
def inputFollowsStart (f : Flowchart) : Prop :=
  match f with
  | FlowchartBox.Start :: FlowchartBox.Input :: _ => True
  | _ => False

/-- Checks if the output box immediately precedes the end box -/
def outputPrecedesEnd (f : Flowchart) : Prop :=
  match f.reverse with
  | FlowchartBox.End :: FlowchartBox.Output :: _ => True
  | _ => False

/-- Theorem stating that it's not always true that input must follow start
    and output must precede end in a flowchart -/
theorem not_always_input_start_output_end :
  ∃ (f : Flowchart), ¬(inputFollowsStart f ∧ outputPrecedesEnd f) :=
sorry

end not_always_input_start_output_end_l2896_289683


namespace bus_speed_l2896_289671

/-- The initial average speed of a bus given specific journey conditions -/
theorem bus_speed (D : ℝ) (h : D > 0) : ∃ v : ℝ,
  v > 0 ∧ 
  D = v * (65 / 60) ∧ 
  D = (v + 5) * 1 ∧
  v = 60 := by
  sorry

end bus_speed_l2896_289671


namespace problem_statement_l2896_289697

theorem problem_statement : 
  let p := (3 + 3 = 5)
  let q := (5 > 2)
  ¬(p ∧ q) ∧ ¬p := by sorry

end problem_statement_l2896_289697


namespace equation_decomposition_l2896_289661

-- Define the original equation
def original_equation (y z : ℝ) : Prop :=
  z^4 - 6*y^4 = 3*z^2 - 2

-- Define the hyperbola equation
def hyperbola_equation (y z : ℝ) : Prop :=
  z^2 - 3*y^2 = 2

-- Define the ellipse equation
def ellipse_equation (y z : ℝ) : Prop :=
  z^2 - 2*y^2 = 1

-- Theorem stating that the original equation can be decomposed into a hyperbola and an ellipse
theorem equation_decomposition :
  ∀ y z : ℝ, original_equation y z ↔ (hyperbola_equation y z ∨ ellipse_equation y z) :=
by sorry


end equation_decomposition_l2896_289661


namespace distributive_property_example_l2896_289691

theorem distributive_property_example :
  (3/4 + 7/12 - 5/9) * (-36) = 3/4 * (-36) + 7/12 * (-36) - 5/9 * (-36) := by
  sorry

end distributive_property_example_l2896_289691


namespace maria_trip_fraction_l2896_289602

theorem maria_trip_fraction (total_distance : ℝ) (first_stop_fraction : ℝ) (final_leg : ℝ) :
  total_distance = 480 →
  first_stop_fraction = 1/2 →
  final_leg = 180 →
  (total_distance - first_stop_fraction * total_distance - final_leg) / (total_distance - first_stop_fraction * total_distance) = 1/4 :=
by sorry

end maria_trip_fraction_l2896_289602
