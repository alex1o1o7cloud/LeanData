import Mathlib

namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2608_260813

-- Define the sets M and N
def M : Set ℕ := {0, 1}
def N : Set ℕ := {1, 2}

-- State the theorem
theorem union_of_M_and_N :
  M ∪ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2608_260813


namespace NUMINAMATH_CALUDE_birth_year_problem_l2608_260805

theorem birth_year_problem (x : ℕ) (h1 : x^2 - 2*x ≥ 1900) (h2 : x^2 - 2*x < 1950) : 
  (x^2 - 2*x + x = 1936) := by
  sorry

end NUMINAMATH_CALUDE_birth_year_problem_l2608_260805


namespace NUMINAMATH_CALUDE_base_conversion_equivalence_l2608_260800

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

def decimal_to_base_five (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

def base_three_number : List Nat := [2, 0, 1, 2, 1]
def base_five_number : List Nat := [1, 2, 0, 3]

theorem base_conversion_equivalence :
  decimal_to_base_five (base_three_to_decimal base_three_number) = base_five_number := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equivalence_l2608_260800


namespace NUMINAMATH_CALUDE_last_page_stamps_l2608_260824

/-- The number of stamp books Jenny originally has -/
def num_books : ℕ := 8

/-- The number of pages in each stamp book -/
def pages_per_book : ℕ := 42

/-- The number of stamps on each page originally -/
def stamps_per_page_original : ℕ := 6

/-- The number of stamps on each page after reorganization -/
def stamps_per_page_new : ℕ := 10

/-- The number of completely filled books after reorganization -/
def filled_books : ℕ := 4

/-- The number of completely filled pages in the partially filled book -/
def filled_pages_partial : ℕ := 33

theorem last_page_stamps :
  (num_books * pages_per_book * stamps_per_page_original) % stamps_per_page_new = 6 :=
sorry

end NUMINAMATH_CALUDE_last_page_stamps_l2608_260824


namespace NUMINAMATH_CALUDE_chess_tournament_theorem_l2608_260839

/-- Represents the number of participants from each city -/
structure Participants where
  moscow : ℕ
  saintPetersburg : ℕ
  kazan : ℕ

/-- Represents the number of games played between participants from different cities -/
structure Games where
  moscowSaintPetersburg : ℕ
  moscowKazan : ℕ
  saintPetersburgKazan : ℕ

/-- The theorem statement based on the chess tournament problem -/
theorem chess_tournament_theorem (p : Participants) (g : Games) : 
  (p.moscow * 9 = p.saintPetersburg * 6) ∧ 
  (p.saintPetersburg * 2 = p.kazan * 6) ∧ 
  (p.moscow * g.moscowKazan = p.kazan * 8) →
  g.moscowKazan = 4 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_theorem_l2608_260839


namespace NUMINAMATH_CALUDE_lcm_product_implies_hcf_l2608_260838

theorem lcm_product_implies_hcf (x y : ℕ+) 
  (h1 : Nat.lcm x y = 600) 
  (h2 : x * y = 18000) : 
  Nat.gcd x y = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_product_implies_hcf_l2608_260838


namespace NUMINAMATH_CALUDE_angle_abc_measure_l2608_260870

/-- A configuration of a regular hexagon with an inscribed square sharing a side. -/
structure HexagonSquareConfig where
  /-- The measure of an interior angle of the regular hexagon in degrees. -/
  hexagon_interior_angle : ℝ
  /-- The measure of an interior angle of the square in degrees. -/
  square_interior_angle : ℝ
  /-- Assumption that the hexagon is regular. -/
  hexagon_regular : hexagon_interior_angle = 120
  /-- Assumption that the inscribed shape is a square. -/
  square_regular : square_interior_angle = 90

/-- The theorem stating that the angle ABC in the given configuration is 45°. -/
theorem angle_abc_measure (config : HexagonSquareConfig) :
  let angle_bdc : ℝ := config.hexagon_interior_angle - config.square_interior_angle
  let angle_cbd : ℝ := (180 - angle_bdc) / 2
  let angle_abc : ℝ := config.hexagon_interior_angle - angle_cbd
  angle_abc = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_abc_measure_l2608_260870


namespace NUMINAMATH_CALUDE_inequality_solution_l2608_260834

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 5 / (x + 4) ≤ 2 - x) ↔ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2608_260834


namespace NUMINAMATH_CALUDE_complex_power_sum_l2608_260818

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^24 + 1/z^24 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2608_260818


namespace NUMINAMATH_CALUDE_largest_k_for_distinct_roots_l2608_260886

theorem largest_k_for_distinct_roots : 
  ∃ k : ℤ, k = 8 ∧ 
  (∀ m : ℤ, m > k → ¬(∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + m = 0 ∧ y^2 - 6*y + m = 0)) ∧
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + k = 0 ∧ y^2 - 6*y + k = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_distinct_roots_l2608_260886


namespace NUMINAMATH_CALUDE_tracy_candies_problem_l2608_260823

theorem tracy_candies_problem :
  ∃ (initial : ℕ) (brother_took : ℕ),
    initial > 0 ∧
    brother_took ≥ 2 ∧
    brother_took ≤ 6 ∧
    (3 * initial / 10 : ℚ) - 20 - brother_took = 6 ∧
    initial = 100 := by
  sorry

end NUMINAMATH_CALUDE_tracy_candies_problem_l2608_260823


namespace NUMINAMATH_CALUDE_count_valid_pairs_l2608_260828

def valid_pair (b c : ℕ) : Prop :=
  1 ≤ b ∧ b ≤ 5 ∧ 1 ≤ c ∧ c ≤ 5 ∧ 
  (b * b : ℤ) - 4 * c ≤ 0 ∧ 
  (c * c : ℤ) - 4 * b ≤ 0

theorem count_valid_pairs : 
  ∃ (S : Finset (ℕ × ℕ)), S.card = 15 ∧ ∀ p, p ∈ S ↔ valid_pair p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l2608_260828


namespace NUMINAMATH_CALUDE_decagon_perimeter_decagon_perimeter_is_35_l2608_260868

theorem decagon_perimeter : ℕ → ℕ → ℕ → ℕ
  | n, a, b =>
    if n = 10 ∧ a = 3 ∧ b = 4
    then 5 * a + 5 * b
    else 0

theorem decagon_perimeter_is_35 :
  decagon_perimeter 10 3 4 = 35 :=
by sorry

end NUMINAMATH_CALUDE_decagon_perimeter_decagon_perimeter_is_35_l2608_260868


namespace NUMINAMATH_CALUDE_crayons_per_child_l2608_260826

theorem crayons_per_child (total_crayons : ℕ) (num_children : ℕ) 
  (h1 : total_crayons = 72) (h2 : num_children = 12) :
  total_crayons / num_children = 6 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_child_l2608_260826


namespace NUMINAMATH_CALUDE_solution_interval_l2608_260895

theorem solution_interval (x₀ : ℝ) (k : ℤ) : 
  (Real.log x₀ + x₀ = 4) → 
  (x₀ > k ∧ x₀ < k + 1) → 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_interval_l2608_260895


namespace NUMINAMATH_CALUDE_divisibility_statements_l2608_260878

theorem divisibility_statements :
  (∃ n : ℤ, 24 = 4 * n) ∧
  (∃ m : ℤ, 152 = 19 * m) ∧ ¬(∃ k : ℤ, 96 = 19 * k) ∧
  ((∃ p : ℤ, 75 = 15 * p) ∨ (∃ q : ℤ, 90 = 15 * q)) ∧
  ((∃ r : ℤ, 28 = 14 * r) ∧ (∃ s : ℤ, 56 = 14 * s)) ∧
  (∃ t : ℤ, 180 = 6 * t) :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_statements_l2608_260878


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2608_260819

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  inningsPlayed : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the increase in average after a new innings -/
def averageIncrease (prevStats : BatsmanStats) (newInningRuns : ℕ) (newAverage : ℚ) : ℚ :=
  newAverage - prevStats.average

/-- Theorem: The increase in the batsman's average is 4 runs -/
theorem batsman_average_increase :
  ∀ (prevStats : BatsmanStats),
  prevStats.inningsPlayed = 16 →
  averageIncrease prevStats 87 23 = 4 := by
  sorry

#check batsman_average_increase

end NUMINAMATH_CALUDE_batsman_average_increase_l2608_260819


namespace NUMINAMATH_CALUDE_quadratic_roots_greater_than_one_l2608_260837

theorem quadratic_roots_greater_than_one (a : ℝ) :
  a ≠ -1 →
  (∀ x : ℝ, (1 + a) * x^2 - 3 * a * x + 4 * a = 0 → x > 1) ↔
  -16/7 < a ∧ a < -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_greater_than_one_l2608_260837


namespace NUMINAMATH_CALUDE_triangle_area_l2608_260854

/-- The area of a triangle with side lengths 3, 5, and 7 is equal to 15√3/4 -/
theorem triangle_area (a b c : ℝ) (h_a : a = 3) (h_b : b = 5) (h_c : c = 7) :
  (1/2) * b * c * Real.sqrt (1 - ((b^2 + c^2 - a^2) / (2*b*c))^2) = (15 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l2608_260854


namespace NUMINAMATH_CALUDE_tromino_tileable_tromino_area_div_by_three_l2608_260861

/-- Definition of a size-n tromino -/
def tromino (n : ℕ) := (2 * n) ^ 2 - n ^ 2

/-- The area of a size-n tromino -/
def tromino_area (n : ℕ) : ℕ := 3 * n ^ 2

/-- A size-n tromino can be tiled by size-1 trominos iff n ≢ 1 (mod 2) -/
theorem tromino_tileable (n : ℕ) (hn : n > 0) :
  (∃ k : ℕ, tromino n = 3 * k) ↔ n % 2 ≠ 1 := by sorry

/-- The area of a size-n tromino is divisible by 3 iff n ≢ 1 (mod 2) -/
theorem tromino_area_div_by_three (n : ℕ) (hn : n > 0) :
  3 ∣ tromino_area n ↔ n % 2 ≠ 1 := by sorry

end NUMINAMATH_CALUDE_tromino_tileable_tromino_area_div_by_three_l2608_260861


namespace NUMINAMATH_CALUDE_linear_function_composition_l2608_260851

theorem linear_function_composition (a b : ℝ) :
  (∀ x : ℝ, (fun x => 3 * ((fun x => a * x + b) x) + 2) = (fun x => 4 * x - 1)) →
  a + b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_linear_function_composition_l2608_260851


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2608_260857

theorem negative_fraction_comparison : -4/5 > -5/6 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2608_260857


namespace NUMINAMATH_CALUDE_vertical_shift_equivalence_l2608_260817

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the vertical shift transformation
def verticalShift (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f x + k

-- Theorem statement
theorem vertical_shift_equivalence :
  ∀ (x : ℝ), verticalShift f 2 x = f x + 2 := by sorry

end NUMINAMATH_CALUDE_vertical_shift_equivalence_l2608_260817


namespace NUMINAMATH_CALUDE_alison_initial_stamps_l2608_260874

/-- Represents the number of stamps each person has -/
structure StampCollection where
  anna : ℕ
  alison : ℕ
  jeff : ℕ

/-- The initial stamp collection -/
def initial : StampCollection where
  anna := 37
  alison := 26  -- This is what we want to prove
  jeff := 31

/-- The final stamp collection after exchanges -/
def final : StampCollection where
  anna := 50
  alison := initial.alison / 2
  jeff := initial.jeff + 1

theorem alison_initial_stamps :
  initial.anna + initial.alison / 2 = final.anna := by sorry

#check alison_initial_stamps

end NUMINAMATH_CALUDE_alison_initial_stamps_l2608_260874


namespace NUMINAMATH_CALUDE_complex_magnitude_l2608_260879

theorem complex_magnitude (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = 2*i + (9 - 3*i) / (1 + i) →
  Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2608_260879


namespace NUMINAMATH_CALUDE_equal_debt_days_l2608_260831

/-- The number of days for two borrowers to owe the same amount -/
def days_to_equal_debt (
  morgan_initial : ℚ)
  (morgan_rate : ℚ)
  (olivia_initial : ℚ)
  (olivia_rate : ℚ) : ℚ :=
  (olivia_initial - morgan_initial) / (morgan_rate * morgan_initial - olivia_rate * olivia_initial)

/-- Proof that Morgan and Olivia will owe the same amount after 25/3 days -/
theorem equal_debt_days :
  days_to_equal_debt 200 (12/100) 300 (4/100) = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_equal_debt_days_l2608_260831


namespace NUMINAMATH_CALUDE_red_cube_possible_l2608_260864

/-- Represents a small cube with colored faces -/
structure SmallCube where
  blue_faces : Nat
  red_faces : Nat

/-- Represents the arrangement of small cubes into a large cube -/
structure LargeCube where
  small_cubes : List SmallCube
  visible_red_faces : Nat
  visible_blue_faces : Nat

/-- Given conditions of the problem -/
def problem_conditions : Prop :=
  ∃ (cubes : List SmallCube) (large_cube : LargeCube),
    -- There are 8 identical cubes
    cubes.length = 8 ∧
    -- Each cube has 6 faces
    ∀ c ∈ cubes, c.blue_faces + c.red_faces = 6 ∧
    -- One-third of all faces are blue, the rest are red
    (cubes.map (λ c => c.blue_faces)).sum = 16 ∧
    (cubes.map (λ c => c.red_faces)).sum = 32 ∧
    -- When assembled into a larger cube, one-third of the visible faces are red
    large_cube.small_cubes = cubes ∧
    large_cube.visible_red_faces = 8 ∧
    large_cube.visible_blue_faces = 16

/-- The theorem to be proved -/
theorem red_cube_possible (h : problem_conditions) :
  ∃ (arrangement : List SmallCube),
    arrangement.length = 8 ∧
    (∀ c ∈ arrangement, c.red_faces ≥ 3) :=
  sorry

end NUMINAMATH_CALUDE_red_cube_possible_l2608_260864


namespace NUMINAMATH_CALUDE_profit_percentage_l2608_260822

/-- If the cost price of 72 articles equals the selling price of 60 articles, then the percent profit is 20%. -/
theorem profit_percentage (C S : ℝ) (h : 72 * C = 60 * S) : (S - C) / C * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l2608_260822


namespace NUMINAMATH_CALUDE_updated_mean_l2608_260830

/-- Given 50 observations with an original mean of 200 and a decrement of 47 from each observation,
    the updated mean is 153. -/
theorem updated_mean (n : ℕ) (original_mean decrement : ℚ) (h1 : n = 50) (h2 : original_mean = 200) (h3 : decrement = 47) :
  let total_sum := n * original_mean
  let total_decrement := n * decrement
  let updated_sum := total_sum - total_decrement
  let updated_mean := updated_sum / n
  updated_mean = 153 := by
sorry

end NUMINAMATH_CALUDE_updated_mean_l2608_260830


namespace NUMINAMATH_CALUDE_inequalities_in_quadrants_I_and_II_l2608_260827

-- Define the conditions
def satisfies_inequalities (x y : ℝ) : Prop :=
  y > x^2 ∧ y > 4 - x

-- Define the quadrants
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0
def in_quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem inequalities_in_quadrants_I_and_II :
  ∀ x y : ℝ, satisfies_inequalities x y →
    (in_quadrant_I x y ∨ in_quadrant_II x y) ∧
    ¬(in_quadrant_III x y) ∧ ¬(in_quadrant_IV x y) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_in_quadrants_I_and_II_l2608_260827


namespace NUMINAMATH_CALUDE_fraction_increase_l2608_260887

theorem fraction_increase (x y : ℝ) (h : x + y ≠ 0) :
  (3 * (2 * x) * (2 * y)) / ((2 * x) + (2 * y)) = 2 * ((3 * x * y) / (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_increase_l2608_260887


namespace NUMINAMATH_CALUDE_print_time_with_rate_change_l2608_260892

/-- Represents the printing scenario with given parameters -/
structure PrintingScenario where
  num_presses : ℕ
  initial_time : ℝ
  new_time : ℝ
  num_papers : ℕ

/-- Calculates the time taken to print papers given a printing scenario -/
def time_to_print (s : PrintingScenario) : ℝ :=
  s.new_time

/-- Theorem stating that the time to print remains the same as the new_time 
    when the printing rate changes but the number of presses remains constant -/
theorem print_time_with_rate_change (s : PrintingScenario) 
  (h1 : s.num_presses = 35)
  (h2 : s.initial_time = 15)
  (h3 : s.new_time = 21)
  (h4 : s.num_papers = 500000) :
  time_to_print s = s.new_time := by
  sorry


end NUMINAMATH_CALUDE_print_time_with_rate_change_l2608_260892


namespace NUMINAMATH_CALUDE_parabola_vertex_sum_max_l2608_260856

theorem parabola_vertex_sum_max (a T : ℤ) (h1 : T ≠ 0) : 
  (∃ b c : ℝ, ∀ x y : ℝ, 
    (y = a * x^2 + b * x + c ↔ 
      (x = 0 ∧ y = 0) ∨ 
      (x = 2 * T ∧ y = 0) ∨ 
      (x = 2 * T + 1 ∧ y = 36))) →
  (let N := T - a * T^2
   ∀ T' a' : ℤ, T' ≠ 0 → 
    (∃ b' c' : ℝ, ∀ x y : ℝ,
      (y = a' * x^2 + b' * x + c' ↔ 
        (x = 0 ∧ y = 0) ∨ 
        (x = 2 * T' ∧ y = 0) ∨ 
        (x = 2 * T' + 1 ∧ y = 36))) →
    T' - a' * T'^2 ≤ N) →
  N = -14 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_sum_max_l2608_260856


namespace NUMINAMATH_CALUDE_division_of_decimals_l2608_260809

theorem division_of_decimals : (0.45 : ℚ) / (0.005 : ℚ) = 90 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l2608_260809


namespace NUMINAMATH_CALUDE_max_points_in_specific_tournament_l2608_260849

/-- Represents a football tournament with given number of teams. -/
structure Tournament where
  num_teams : ℕ
  points_per_win : ℕ
  points_per_draw : ℕ
  points_per_loss : ℕ

/-- The maximum number of points each team can achieve in the tournament. -/
def max_points_per_team (t : Tournament) : ℕ :=
  sorry

/-- Theorem stating the maximum points per team in a specific tournament setup. -/
theorem max_points_in_specific_tournament :
  ∃ (t : Tournament),
    t.num_teams = 10 ∧
    t.points_per_win = 3 ∧
    t.points_per_draw = 1 ∧
    t.points_per_loss = 0 ∧
    max_points_per_team t = 13 :=
  sorry

end NUMINAMATH_CALUDE_max_points_in_specific_tournament_l2608_260849


namespace NUMINAMATH_CALUDE_equation_solution_l2608_260807

theorem equation_solution : ∃ x : ℚ, x - 1/2 = 2/5 - 1/4 ∧ x = 13/20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2608_260807


namespace NUMINAMATH_CALUDE_coefficient_x2_cube_polynomial_l2608_260860

/-- Given a polynomial q(x) = x^5 - 4x^3 + 5x^2 - 6x + 3, 
    this theorem states that the coefficient of x^2 in (q(x))^3 is 540. -/
theorem coefficient_x2_cube_polynomial :
  let q : Polynomial ℝ := X^5 - 4*X^3 + 5*X^2 - 6*X + 3
  (q^3).coeff 2 = 540 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2_cube_polynomial_l2608_260860


namespace NUMINAMATH_CALUDE_plates_for_matt_l2608_260848

/-- The number of plates needed for a week under specific dining conditions -/
def plates_needed (days_with_two : Nat) (days_with_four : Nat) (plates_per_person_two : Nat) (plates_per_person_four : Nat) : Nat :=
  (days_with_two * 2 * plates_per_person_two) + (days_with_four * 4 * plates_per_person_four)

theorem plates_for_matt : plates_needed 3 4 1 2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_plates_for_matt_l2608_260848


namespace NUMINAMATH_CALUDE_largest_operator_result_l2608_260890

theorem largest_operator_result : 
  let expr := (5 * Real.sqrt 2 - Real.sqrt 2)
  (expr * Real.sqrt 2 > expr + Real.sqrt 2) ∧
  (expr * Real.sqrt 2 > expr - Real.sqrt 2) ∧
  (expr * Real.sqrt 2 > expr / Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_largest_operator_result_l2608_260890


namespace NUMINAMATH_CALUDE_line_through_points_l2608_260882

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a line by its equation coefficients (ax + by + c = 0)
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

-- Theorem statement
theorem line_through_points : 
  ∃ (l : Line), 
    point_on_line (-3, 0) l ∧ 
    point_on_line (0, 4) l ∧ 
    l.a = 4 ∧ l.b = -3 ∧ l.c = 12 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2608_260882


namespace NUMINAMATH_CALUDE_fixed_point_on_all_lines_l2608_260802

/-- The fixed point through which all lines in the family pass -/
def fixed_point : ℝ × ℝ := (2, 2)

/-- The equation of the line for a given k -/
def line_equation (k x y : ℝ) : Prop :=
  (1 + 4*k)*x - (2 - 3*k)*y + (2 - 14*k) = 0

/-- Theorem stating that the fixed point satisfies the line equation for all k -/
theorem fixed_point_on_all_lines :
  ∀ k : ℝ, line_equation k fixed_point.1 fixed_point.2 :=
by
  sorry


end NUMINAMATH_CALUDE_fixed_point_on_all_lines_l2608_260802


namespace NUMINAMATH_CALUDE_company_income_analysis_l2608_260832

structure Company where
  employees : ℕ
  max_income : ℕ
  avg_income : ℕ
  min_income : ℕ
  mid_50_low : ℕ
  mid_50_high : ℕ

def is_high_income (c : Company) (income : ℕ) : Prop :=
  income > c.avg_income

def is_sufficient_info (c : Company) : Prop :=
  c.mid_50_low > 0 ∧ c.mid_50_high > c.mid_50_low

def estimate_median (c : Company) : ℕ :=
  (c.mid_50_low + c.mid_50_high) / 2

theorem company_income_analysis (c : Company) 
  (h1 : c.employees = 50)
  (h2 : c.max_income = 1000000)
  (h3 : c.avg_income = 35000)
  (h4 : c.min_income = 5000)
  (h5 : c.mid_50_low = 10000)
  (h6 : c.mid_50_high = 30000) :
  ¬is_high_income c 25000 ∧
  ¬is_sufficient_info {employees := c.employees, max_income := c.max_income, avg_income := c.avg_income, min_income := c.min_income, mid_50_low := 0, mid_50_high := 0} ∧
  is_sufficient_info c ∧
  estimate_median c < c.avg_income := by
  sorry

#check company_income_analysis

end NUMINAMATH_CALUDE_company_income_analysis_l2608_260832


namespace NUMINAMATH_CALUDE_number_ordering_l2608_260835

theorem number_ordering : (10 ^ 5 : ℝ) < 2 ^ 20 ∧ 2 ^ 20 < 5 ^ 10 := by sorry

end NUMINAMATH_CALUDE_number_ordering_l2608_260835


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2608_260865

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^4 + 14 * X^3 - 56 * X^2 - 72 * X + 88 = 
  (X^2 + 9 * X - 4) * q + (533 * X - 204) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2608_260865


namespace NUMINAMATH_CALUDE_max_value_theorem_l2608_260847

theorem max_value_theorem (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  ∃ (max : ℝ), max = 7/5 ∧ ∀ (x y : ℝ), 
    x + y - 2 ≥ 0 → y - x - 1 ≤ 0 → x ≤ 1 → 
    (x + 2*y) / (2*x + y) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2608_260847


namespace NUMINAMATH_CALUDE_solve_for_a_l2608_260862

theorem solve_for_a (x a : ℚ) (h1 : 3 * x + 2 * a = 2) (h2 : x = 1) : a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2608_260862


namespace NUMINAMATH_CALUDE_stating_minimum_red_cubes_correct_l2608_260843

/-- 
Given a positive integer n, we construct a cube of side length 3n using smaller 3x3x3 cubes.
Each 3x3x3 cube is made of 26 white unit cubes and 1 black unit cube.
This function returns the minimum number of white unit cubes that need to be painted red
so that every remaining white unit cube has at least one common point with at least one red unit cube.
-/
def minimum_red_cubes (n : ℕ+) : ℕ :=
  (n + 1) * n^2

/-- 
Theorem stating that the minimum number of white unit cubes that need to be painted red
is indeed (n+1)n^2, where n is the number of 3x3x3 cubes along each edge of the larger cube.
-/
theorem minimum_red_cubes_correct (n : ℕ+) : 
  minimum_red_cubes n = (n + 1) * n^2 := by sorry

end NUMINAMATH_CALUDE_stating_minimum_red_cubes_correct_l2608_260843


namespace NUMINAMATH_CALUDE_park_short_trees_l2608_260885

/-- The number of short trees in the park after planting -/
def total_short_trees (initial_short_trees new_short_trees : ℕ) : ℕ :=
  initial_short_trees + new_short_trees

/-- Theorem: The park will have 217 short trees after planting -/
theorem park_short_trees : 
  total_short_trees 112 105 = 217 := by
  sorry

end NUMINAMATH_CALUDE_park_short_trees_l2608_260885


namespace NUMINAMATH_CALUDE_flowers_ratio_proof_l2608_260808

/-- Proves that the ratio of flowers Ingrid gave to Collin to her initial flowers is 1:3 --/
theorem flowers_ratio_proof (collin_initial : ℕ) (ingrid_initial : ℕ) (petals_per_flower : ℕ) (collin_final_petals : ℕ) :
  collin_initial = 25 →
  ingrid_initial = 33 →
  petals_per_flower = 4 →
  collin_final_petals = 144 →
  ∃ (flowers_given : ℕ),
    flowers_given * petals_per_flower = collin_final_petals - collin_initial * petals_per_flower ∧
    flowers_given * 3 = ingrid_initial :=
by sorry

end NUMINAMATH_CALUDE_flowers_ratio_proof_l2608_260808


namespace NUMINAMATH_CALUDE_digit_2023_of_11_26_l2608_260875

/-- The repeating decimal representation of 11/26 -/
def repeating_decimal : List Nat := [4, 2, 3, 0, 7, 6]

/-- The length of the repeating decimal -/
def repeat_length : Nat := repeating_decimal.length

theorem digit_2023_of_11_26 : 
  -- The 2023rd digit past the decimal point in 11/26
  List.get! repeating_decimal ((2023 - 1) % repeat_length) = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_2023_of_11_26_l2608_260875


namespace NUMINAMATH_CALUDE_extended_tile_pattern_ratio_l2608_260872

theorem extended_tile_pattern_ratio :
  let initial_black : ℕ := 10
  let initial_white : ℕ := 15
  let initial_side_length : ℕ := (initial_black + initial_white).sqrt
  let extended_side_length : ℕ := initial_side_length + 2
  let border_black : ℕ := 4 * (extended_side_length - 1)
  let border_white : ℕ := 4 * (extended_side_length - 2)
  let total_black : ℕ := initial_black + border_black
  let total_white : ℕ := initial_white + border_white
  (total_black : ℚ) / (total_white : ℚ) = 26 / 23 := by
sorry

end NUMINAMATH_CALUDE_extended_tile_pattern_ratio_l2608_260872


namespace NUMINAMATH_CALUDE_inequality_solution_l2608_260846

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4*x - 12) / (x - 3) < 0 ↔ (x > -2 ∧ x < 3) ∨ (x > 3 ∧ x < 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2608_260846


namespace NUMINAMATH_CALUDE_even_product_square_sum_solution_l2608_260820

theorem even_product_square_sum_solution (a b : ℤ) (h : 2 ∣ (a * b)) :
  ∃ (x y : ℤ), a^2 + b^2 + x^2 = y^2 :=
by sorry

end NUMINAMATH_CALUDE_even_product_square_sum_solution_l2608_260820


namespace NUMINAMATH_CALUDE_age_difference_is_51_l2608_260811

/-- The age difference between Milena's cousin Alex and her grandfather -/
def age_difference : ℕ :=
  let milena_age : ℕ := 7
  let grandmother_age : ℕ := 9 * milena_age
  let grandfather_age : ℕ := grandmother_age + 2
  let alex_age : ℕ := 2 * milena_age
  grandfather_age - alex_age

theorem age_difference_is_51 : age_difference = 51 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_51_l2608_260811


namespace NUMINAMATH_CALUDE_sin_cos_eq_one_solutions_l2608_260883

theorem sin_cos_eq_one_solutions (x : Real) :
  x ∈ Set.Icc 0 Real.pi →
  (Real.sin x + Real.cos x = 1) ↔ (x = 0 ∨ x = Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_eq_one_solutions_l2608_260883


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2608_260816

/-- A quadratic polynomial q(x) satisfying specific conditions -/
def q (x : ℚ) : ℚ := (31/15) * x^2 - (27/5) * x - 289/15

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-1) = 7 ∧ q 2 = -3 ∧ q 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l2608_260816


namespace NUMINAMATH_CALUDE_f_properties_l2608_260841

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_properties (x₁ x₂ : ℝ) (h1 : x₂ > x₁) (h2 : x₁ > 1) :
  (|f x₁ - f x₂| < 1 / Real.exp 1) ∧ (f x₁ - f x₂ < x₂ - x₁) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2608_260841


namespace NUMINAMATH_CALUDE_fifth_fibonacci_is_eight_l2608_260863

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | k + 2 => fibonacci k + fibonacci (k + 1)

theorem fifth_fibonacci_is_eight :
  fibonacci 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fifth_fibonacci_is_eight_l2608_260863


namespace NUMINAMATH_CALUDE_tournament_games_played_21_teams_l2608_260855

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  no_ties : Bool

/-- The number of games played in a single-elimination tournament. -/
def games_played (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 21 teams and no ties,
    20 games are played before a winner is declared. -/
theorem tournament_games_played_21_teams :
  ∀ t : Tournament, t.num_teams = 21 → t.no_ties = true →
    games_played t = 20 := by
  sorry


end NUMINAMATH_CALUDE_tournament_games_played_21_teams_l2608_260855


namespace NUMINAMATH_CALUDE_tobacco_acreage_change_l2608_260884

/-- Calculates the change in tobacco acreage when crop ratios are adjusted -/
theorem tobacco_acreage_change 
  (total_land : ℝ) 
  (initial_ratio_corn initial_ratio_sugarcane initial_ratio_tobacco : ℕ)
  (new_ratio_corn new_ratio_sugarcane new_ratio_tobacco : ℕ) : 
  total_land = 1350 ∧ 
  initial_ratio_corn = 5 ∧ 
  initial_ratio_sugarcane = 2 ∧ 
  initial_ratio_tobacco = 2 ∧
  new_ratio_corn = 2 ∧ 
  new_ratio_sugarcane = 2 ∧ 
  new_ratio_tobacco = 5 →
  (new_ratio_tobacco * total_land / (new_ratio_corn + new_ratio_sugarcane + new_ratio_tobacco) -
   initial_ratio_tobacco * total_land / (initial_ratio_corn + initial_ratio_sugarcane + initial_ratio_tobacco)) = 450 :=
by sorry

end NUMINAMATH_CALUDE_tobacco_acreage_change_l2608_260884


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2608_260891

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A
def A : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2608_260891


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2608_260845

theorem min_value_quadratic (x : ℝ) : 
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 4*x^2 + 8*x + 10 → y ≥ y_min ∧ y_min = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2608_260845


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_x_equals_pi_half_l2608_260871

theorem integral_sqrt_minus_x_equals_pi_half :
  ∫ x in (-1)..(1), (Real.sqrt (1 - x^2) - x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_x_equals_pi_half_l2608_260871


namespace NUMINAMATH_CALUDE_triangle_two_solutions_l2608_260812

/-- Triangle ABC with given side lengths and angle --/
structure Triangle where
  a : ℝ
  b : ℝ
  A : ℝ

/-- The number of solutions for a triangle with given side lengths and angle --/
def numSolutions (t : Triangle) : ℕ :=
  sorry

/-- Theorem stating that a triangle with a = 18, b = 24, and A = 30° has exactly two solutions --/
theorem triangle_two_solutions :
  ∀ t : Triangle, t.a = 18 ∧ t.b = 24 ∧ t.A = 30 * π / 180 → numSolutions t = 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_two_solutions_l2608_260812


namespace NUMINAMATH_CALUDE_two_unit_circles_tangent_to_two_three_l2608_260815

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- A circle is externally tangent to two other circles -/
def externally_tangent_to_both (c : Circle) (c1 c2 : Circle) : Prop :=
  externally_tangent c c1 ∧ externally_tangent c c2

theorem two_unit_circles_tangent_to_two_three (c1 c2 : Circle)
  (h1 : c1.radius = 2)
  (h2 : c2.radius = 3)
  (h3 : externally_tangent c1 c2) :
  ∃! (s : Finset Circle), s.card = 2 ∧ ∀ c ∈ s, c.radius = 1 ∧ externally_tangent_to_both c c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_two_unit_circles_tangent_to_two_three_l2608_260815


namespace NUMINAMATH_CALUDE_correct_withdrawal_amount_withdrawal_amount_2016_l2608_260876

/-- Calculates the amount that can be withdrawn after a given number of years
    for a fixed-term deposit with annual compound interest. -/
def withdrawal_amount (initial_deposit : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  initial_deposit * (1 + interest_rate) ^ years

/-- Theorem stating the correct withdrawal amount after 14 years -/
theorem correct_withdrawal_amount (a : ℝ) (r : ℝ) :
  withdrawal_amount a r 14 = a * (1 + r)^14 := by
  sorry

/-- The number of years between January 1, 2002 and January 1, 2016 -/
def years_between_2002_and_2016 : ℕ := 14

/-- Theorem proving the correct withdrawal amount on January 1, 2016 -/
theorem withdrawal_amount_2016 (a : ℝ) (r : ℝ) :
  withdrawal_amount a r years_between_2002_and_2016 = a * (1 + r)^14 := by
  sorry

end NUMINAMATH_CALUDE_correct_withdrawal_amount_withdrawal_amount_2016_l2608_260876


namespace NUMINAMATH_CALUDE_selection_methods_l2608_260880

theorem selection_methods (boys girls : ℕ) (tasks : ℕ) : 
  boys = 5 → girls = 4 → tasks = 3 →
  (Nat.choose boys 2 * Nat.choose girls 1 + Nat.choose boys 1 * Nat.choose girls 2) * Nat.factorial tasks = 420 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_l2608_260880


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2608_260877

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ + a₄ = 120 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2608_260877


namespace NUMINAMATH_CALUDE_total_book_pairs_l2608_260803

-- Define the number of books in each genre
def mystery_books : Nat := 4
def fantasy_books : Nat := 4
def biography_books : Nat := 3

-- Define the function to calculate the number of book pairs
def book_pairs : Nat :=
  mystery_books * fantasy_books +
  mystery_books * biography_books +
  fantasy_books * biography_books

-- Theorem statement
theorem total_book_pairs : book_pairs = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_book_pairs_l2608_260803


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l2608_260821

theorem rectangle_shorter_side (a b d : ℝ) : 
  a > 0 → b > 0 → d > 0 →
  (a / b = 3 / 4) →
  (a^2 + b^2 = d^2) →
  d = 9 →
  a = 5.4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l2608_260821


namespace NUMINAMATH_CALUDE_einstein_snack_sale_l2608_260866

/-- The number of potato fries packs sold by Einstein --/
def potato_fries_packs : ℕ := sorry

theorem einstein_snack_sale :
  let goal : ℚ := 500
  let pizza_price : ℚ := 12
  let fries_price : ℚ := 0.30
  let soda_price : ℚ := 2
  let pizza_sold : ℕ := 15
  let soda_sold : ℕ := 25
  let remaining : ℚ := 258
  
  (pizza_price * pizza_sold + fries_price * potato_fries_packs + soda_price * soda_sold = goal - remaining) ∧
  (potato_fries_packs = 40) := by sorry

end NUMINAMATH_CALUDE_einstein_snack_sale_l2608_260866


namespace NUMINAMATH_CALUDE_parabola_vertex_l2608_260804

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 3)^2 + 1

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, 1)

/-- Theorem: The vertex of the parabola y = 2(x-3)^2 + 1 is (3, 1) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2608_260804


namespace NUMINAMATH_CALUDE_prob_at_least_one_goes_l2608_260836

/-- The probability that at least one person goes to a place given individual probabilities -/
theorem prob_at_least_one_goes (prob_A prob_B : ℚ) (h_prob_A : prob_A = 1/4) (h_prob_B : prob_B = 2/5) :
  1 - (1 - prob_A) * (1 - prob_B) = 11/20 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_goes_l2608_260836


namespace NUMINAMATH_CALUDE_reciprocal_of_eight_l2608_260801

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_eight : reciprocal 8 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_eight_l2608_260801


namespace NUMINAMATH_CALUDE_fireworks_display_l2608_260850

theorem fireworks_display (year_digits : ℕ) (phrase_letters : ℕ) 
  (additional_boxes : ℕ) (fireworks_per_box : ℕ) (fireworks_per_letter : ℕ) 
  (total_fireworks : ℕ) :
  year_digits = 4 →
  phrase_letters = 12 →
  additional_boxes = 50 →
  fireworks_per_box = 8 →
  fireworks_per_letter = 5 →
  total_fireworks = 484 →
  ∃ (fireworks_per_digit : ℕ),
    year_digits * fireworks_per_digit + 
    phrase_letters * fireworks_per_letter + 
    additional_boxes * fireworks_per_box = total_fireworks ∧
    fireworks_per_digit = 6 :=
by sorry

end NUMINAMATH_CALUDE_fireworks_display_l2608_260850


namespace NUMINAMATH_CALUDE_jorge_ticket_cost_l2608_260893

/-- Calculates the total cost of tickets after all discounts --/
def total_cost_after_discounts (adult_tickets senior_tickets child_tickets : ℕ)
  (adult_price senior_price child_price : ℚ)
  (tier1_threshold tier2_threshold tier3_threshold : ℚ)
  (tier1_adult_discount tier1_senior_discount : ℚ)
  (tier2_adult_discount tier2_senior_discount : ℚ)
  (tier3_adult_discount tier3_senior_discount : ℚ)
  (extra_discount_per_50 max_extra_discount : ℚ) : ℚ :=
  sorry

/-- The theorem to be proved --/
theorem jorge_ticket_cost :
  total_cost_after_discounts 10 8 6 12 8 6 100 200 300
    0.1 0.05 0.2 0.1 0.3 0.15 0.05 0.15 = 161.16 := by
  sorry

end NUMINAMATH_CALUDE_jorge_ticket_cost_l2608_260893


namespace NUMINAMATH_CALUDE_root_of_quadratic_l2608_260853

theorem root_of_quadratic (x : ℝ) : 
  x = (-25 + Real.sqrt 361) / 12 → 6 * x^2 + 25 * x + 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_of_quadratic_l2608_260853


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l2608_260867

theorem quadratic_root_k_value (k : ℝ) : 
  ((-2 : ℝ)^2 - k * (-2) - 6 = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l2608_260867


namespace NUMINAMATH_CALUDE_triangle_special_angle_l2608_260897

/-- In a triangle ABC, if 2b*cos(A) = 2c - sqrt(3)*a, then the measure of angle B is π/6 --/
theorem triangle_special_angle (a b c : ℝ) (A B : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A) (h5 : A < π) (h6 : 0 < B) (h7 : B < π)
  (h8 : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a) :
  B = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l2608_260897


namespace NUMINAMATH_CALUDE_julio_has_seven_grape_bottles_l2608_260899

-- Define the number of bottles and liters
def julio_orange_bottles : ℕ := 4
def mateo_orange_bottles : ℕ := 1
def mateo_grape_bottles : ℕ := 3
def liters_per_bottle : ℕ := 2
def julio_extra_liters : ℕ := 14

-- Define the function to calculate the number of grape bottles Julio has
def julio_grape_bottles : ℕ :=
  let mateo_total_liters := (mateo_orange_bottles + mateo_grape_bottles) * liters_per_bottle
  let julio_total_liters := mateo_total_liters + julio_extra_liters
  let julio_grape_liters := julio_total_liters - (julio_orange_bottles * liters_per_bottle)
  julio_grape_liters / liters_per_bottle

-- State the theorem
theorem julio_has_seven_grape_bottles :
  julio_grape_bottles = 7 := by sorry

end NUMINAMATH_CALUDE_julio_has_seven_grape_bottles_l2608_260899


namespace NUMINAMATH_CALUDE_dvd_book_problem_l2608_260889

theorem dvd_book_problem (total_capacity : ℕ) (empty_spaces : ℕ) (h1 : total_capacity = 126) (h2 : empty_spaces = 45) :
  total_capacity - empty_spaces = 81 := by
  sorry

end NUMINAMATH_CALUDE_dvd_book_problem_l2608_260889


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2608_260842

theorem quadratic_root_difference (r s : ℝ) (hr : r > 0) : 
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ 
    y1^2 - r*y1 - s = 0 ∧ 
    y2^2 - r*y2 - s = 0 ∧ 
    |y1 - y2| = 2) → 
  r = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2608_260842


namespace NUMINAMATH_CALUDE_sequence_decreasing_l2608_260806

theorem sequence_decreasing (a : ℕ → ℝ) (h1 : a 1 > 0) (h2 : ∀ n : ℕ, a (n + 1) / a n = 1 / 2) :
  ∀ n m : ℕ, n < m → a m < a n :=
sorry

end NUMINAMATH_CALUDE_sequence_decreasing_l2608_260806


namespace NUMINAMATH_CALUDE_min_prime_factorization_sum_l2608_260894

theorem min_prime_factorization_sum (x y a b : ℕ+) (e f : ℕ) :
  5 * x^7 = 13 * y^11 →
  x = a^e * b^f →
  a.val.Prime ∧ b.val.Prime →
  a ≠ b →
  a + b + e + f = 25 :=
sorry

end NUMINAMATH_CALUDE_min_prime_factorization_sum_l2608_260894


namespace NUMINAMATH_CALUDE_prob_at_least_two_women_l2608_260810

/-- The probability of selecting at least two women from a group of 9 men and 6 women when choosing 4 people at random -/
theorem prob_at_least_two_women (num_men : ℕ) (num_women : ℕ) (num_selected : ℕ) : 
  num_men = 9 → num_women = 6 → num_selected = 4 →
  (Nat.choose (num_men + num_women) num_selected - 
   (Nat.choose num_men num_selected + 
    num_women * Nat.choose num_men (num_selected - 1))) / 
  Nat.choose (num_men + num_women) num_selected = 7 / 13 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_women_l2608_260810


namespace NUMINAMATH_CALUDE_power_comparison_l2608_260869

theorem power_comparison : (2 : ℝ)^30 < 10^10 ∧ 10^10 < 5^15 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l2608_260869


namespace NUMINAMATH_CALUDE_min_value_of_a_plus_4b_l2608_260844

theorem min_value_of_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) : 
  a + 4*b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_plus_4b_l2608_260844


namespace NUMINAMATH_CALUDE_existence_of_coprime_sum_l2608_260898

theorem existence_of_coprime_sum (n k : ℕ) (hn : n > 0) (hk : Even (k * (n - 1))) :
  ∃ x y : ℤ, (Nat.gcd x.natAbs n = 1) ∧ (Nat.gcd y.natAbs n = 1) ∧ ((x + y) % n = k % n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_coprime_sum_l2608_260898


namespace NUMINAMATH_CALUDE_wanda_initial_blocks_l2608_260852

/-- The number of blocks Theresa gave to Wanda -/
def blocks_from_theresa : ℕ := 79

/-- The total number of blocks Wanda has after receiving blocks from Theresa -/
def total_blocks : ℕ := 83

/-- The number of blocks Wanda had initially -/
def initial_blocks : ℕ := total_blocks - blocks_from_theresa

theorem wanda_initial_blocks :
  initial_blocks = 4 :=
by sorry

end NUMINAMATH_CALUDE_wanda_initial_blocks_l2608_260852


namespace NUMINAMATH_CALUDE_tan_105_minus_one_over_tan_105_plus_one_equals_sqrt_three_l2608_260825

theorem tan_105_minus_one_over_tan_105_plus_one_equals_sqrt_three :
  (Real.tan (105 * π / 180) - 1) / (Real.tan (105 * π / 180) + 1) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_minus_one_over_tan_105_plus_one_equals_sqrt_three_l2608_260825


namespace NUMINAMATH_CALUDE_range_of_a_l2608_260840

def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def prop_q (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → (4 * x^2 - a * x) ≤ (4 * y^2 - a * y)

theorem range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) → ¬(prop_p a) → (a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2608_260840


namespace NUMINAMATH_CALUDE_T_is_three_intersecting_lines_l2608_260858

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ 5 ≤ y - 3) ∨
               (5 = y - 3 ∧ 5 ≤ x + 3) ∨
               (x + 3 = y - 3 ∧ 5 ≤ x + 3)}

-- Define the three lines
def line1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≥ 8}
def line2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 8 ∧ p.1 ≥ 2}
def line3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 6 ∧ p.1 ≤ 2}

-- Define the intersection points
def point1 : ℝ × ℝ := (2, 8)
def point2 : ℝ × ℝ := (2, 8)
def point3 : ℝ × ℝ := (2, 8)

-- Theorem statement
theorem T_is_three_intersecting_lines :
  T = line1 ∪ line2 ∪ line3 ∧
  (∃ (p1 p2 p3 : ℝ × ℝ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    p1 ∈ line1 ∩ line2 ∧ p2 ∈ line2 ∩ line3 ∧ p3 ∈ line1 ∩ line3) :=
by sorry

end NUMINAMATH_CALUDE_T_is_three_intersecting_lines_l2608_260858


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l2608_260881

def b (n : ℕ) : ℕ := (n + 2).factorial - n^2

theorem max_gcd_consecutive_terms :
  (∃ k : ℕ, Nat.gcd (b k) (b (k + 1)) = 5) ∧
  (∀ n : ℕ, Nat.gcd (b n) (b (n + 1)) ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l2608_260881


namespace NUMINAMATH_CALUDE_handshake_count_l2608_260814

theorem handshake_count (n : ℕ) (h : n = 10) : n * (n - 1) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l2608_260814


namespace NUMINAMATH_CALUDE_tina_change_probability_l2608_260888

/-- The number of toys in the machine -/
def num_toys : ℕ := 10

/-- The cost of the most expensive toy in cents -/
def max_cost : ℕ := 450

/-- The cost difference between consecutive toys in cents -/
def cost_diff : ℕ := 50

/-- The number of quarters Tina starts with -/
def initial_quarters : ℕ := 12

/-- The cost of Tina's favorite toy in cents -/
def favorite_toy_cost : ℕ := 400

/-- The probability that Tina needs to get change for her twenty-dollar bill -/
def change_probability : ℚ := 999802 / 1000000

theorem tina_change_probability :
  (1 : ℚ) - (Nat.factorial (num_toys - 4) : ℚ) / (Nat.factorial num_toys : ℚ) = change_probability :=
sorry

end NUMINAMATH_CALUDE_tina_change_probability_l2608_260888


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l2608_260829

def team_size : ℕ := 13
def lineup_size : ℕ := 6

theorem starting_lineup_combinations : 
  (team_size * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4) * (team_size - 5)) = 1027680 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l2608_260829


namespace NUMINAMATH_CALUDE_sin_cos_product_l2608_260873

theorem sin_cos_product (α : Real) 
  (h : (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7) : 
  Real.sin α * Real.cos α = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_product_l2608_260873


namespace NUMINAMATH_CALUDE_point_line_plane_relation_l2608_260833

-- Define the types for point, line, and plane
variable (Point Line Plane : Type)

-- Define the relations
variable (lies_on : Point → Line → Prop)
variable (lies_in : Line → Plane → Prop)

-- Define the set membership and subset relations
variable (mem : Point → Line → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem point_line_plane_relation 
  (A : Point) (b : Line) (β : Plane) 
  (h1 : lies_on A b) 
  (h2 : lies_in b β) :
  mem A b ∧ subset b β := by
  sorry

end NUMINAMATH_CALUDE_point_line_plane_relation_l2608_260833


namespace NUMINAMATH_CALUDE_first_brother_is_treljalya_l2608_260859

structure Brother where
  name : String
  card_color : String
  tells_truth : Bool

def first_brother_statement_1 (b : Brother) : Prop :=
  b.name = "Treljalya"

def second_brother_statement (b : Brother) : Prop :=
  b.name = "Treljalya"

def first_brother_statement_2 (b : Brother) : Prop :=
  b.card_color = "orange"

def same_suit_rule (b1 b2 : Brother) : Prop :=
  b1.card_color = b2.card_color → b1.tells_truth ≠ b2.tells_truth

def different_suit_rule (b1 b2 : Brother) : Prop :=
  b1.card_color ≠ b2.card_color → b1.tells_truth = b2.tells_truth

theorem first_brother_is_treljalya (b1 b2 : Brother) :
  same_suit_rule b1 b2 →
  different_suit_rule b1 b2 →
  first_brother_statement_1 b1 →
  second_brother_statement b2 →
  first_brother_statement_2 b2 →
  b1.name = "Treljalya" :=
sorry

end NUMINAMATH_CALUDE_first_brother_is_treljalya_l2608_260859


namespace NUMINAMATH_CALUDE_book_profit_rate_l2608_260896

/-- Calculates the rate of profit given cost price and selling price -/
def rate_of_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at 50 rupees and sold at 80 rupees is 60% -/
theorem book_profit_rate :
  let cost_price : ℚ := 50
  let selling_price : ℚ := 80
  rate_of_profit cost_price selling_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_profit_rate_l2608_260896
