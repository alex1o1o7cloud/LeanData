import Mathlib

namespace line_through_points_l1625_162580

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

end line_through_points_l1625_162580


namespace starting_lineup_combinations_l1625_162597

def team_size : ℕ := 13
def lineup_size : ℕ := 6

theorem starting_lineup_combinations : 
  (team_size * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4) * (team_size - 5)) = 1027680 := by
  sorry

end starting_lineup_combinations_l1625_162597


namespace decagon_perimeter_decagon_perimeter_is_35_l1625_162589

theorem decagon_perimeter : ℕ → ℕ → ℕ → ℕ
  | n, a, b =>
    if n = 10 ∧ a = 3 ∧ b = 4
    then 5 * a + 5 * b
    else 0

theorem decagon_perimeter_is_35 :
  decagon_perimeter 10 3 4 = 35 :=
by sorry

end decagon_perimeter_decagon_perimeter_is_35_l1625_162589


namespace min_value_of_a_plus_4b_l1625_162598

theorem min_value_of_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) : 
  a + 4*b ≥ 9 := by
  sorry

end min_value_of_a_plus_4b_l1625_162598


namespace two_unit_circles_tangent_to_two_three_l1625_162563

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

end two_unit_circles_tangent_to_two_three_l1625_162563


namespace vector_equality_transitivity_l1625_162524

variable {V : Type*} [AddCommGroup V]

theorem vector_equality_transitivity (a b c : V) :
  a = b → b = c → a = c := by
  sorry

end vector_equality_transitivity_l1625_162524


namespace triangle_two_solutions_l1625_162599

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

end triangle_two_solutions_l1625_162599


namespace polynomial_division_remainder_l1625_162550

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^4 + 14 * X^3 - 56 * X^2 - 72 * X + 88 = 
  (X^2 + 9 * X - 4) * q + (533 * X - 204) := by
  sorry

end polynomial_division_remainder_l1625_162550


namespace fake_coin_identification_l1625_162568

/-- Represents a weighing strategy for identifying a fake coin. -/
structure WeighingStrategy where
  /-- The number of weighings performed. -/
  num_weighings : ℕ
  /-- The maximum number of times any single coin is weighed. -/
  max_weighs_per_coin : ℕ

/-- Represents the problem of identifying a fake coin among a set of coins. -/
structure FakeCoinProblem where
  /-- The total number of coins. -/
  total_coins : ℕ
  /-- The number of fake coins. -/
  num_fake_coins : ℕ
  /-- Indicates whether the fake coin is lighter than the genuine coins. -/
  fake_is_lighter : Bool

/-- Theorem stating that the fake coin can be identified within the given constraints. -/
theorem fake_coin_identification
  (problem : FakeCoinProblem)
  (strategy : WeighingStrategy) :
  problem.total_coins = 99 →
  problem.num_fake_coins = 1 →
  problem.fake_is_lighter = true →
  strategy.num_weighings ≤ 7 →
  strategy.max_weighs_per_coin ≤ 2 →
  ∃ (identification_method : Unit), True :=
by
  sorry

end fake_coin_identification_l1625_162568


namespace dvd_book_problem_l1625_162533

theorem dvd_book_problem (total_capacity : ℕ) (empty_spaces : ℕ) (h1 : total_capacity = 126) (h2 : empty_spaces = 45) :
  total_capacity - empty_spaces = 81 := by
  sorry

end dvd_book_problem_l1625_162533


namespace pancakes_theorem_l1625_162592

/-- The number of pancakes left after Bobby and his dog eat some. -/
def pancakes_left (total : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) : ℕ :=
  total - (bobby_ate + dog_ate)

/-- Theorem: Given 21 pancakes, if Bobby eats 5 and his dog eats 7, there are 9 pancakes left. -/
theorem pancakes_theorem : pancakes_left 21 5 7 = 9 := by
  sorry

end pancakes_theorem_l1625_162592


namespace middle_number_is_seven_l1625_162518

/-- Given three consecutive integers where the sums of these integers taken in pairs are 18, 20, and 23, prove that the middle number is 7. -/
theorem middle_number_is_seven (x : ℤ) 
  (h1 : x + (x + 1) = 18) 
  (h2 : x + (x + 2) = 20) 
  (h3 : (x + 1) + (x + 2) = 23) : 
  x + 1 = 7 := by
  sorry

end middle_number_is_seven_l1625_162518


namespace marius_monica_difference_l1625_162516

/-- The number of subjects taken by students Millie, Monica, and Marius. -/
structure SubjectCounts where
  millie : ℕ
  monica : ℕ
  marius : ℕ

/-- The conditions of the problem. -/
def problem_conditions (counts : SubjectCounts) : Prop :=
  counts.millie = counts.marius + 3 ∧
  counts.marius > counts.monica ∧
  counts.monica = 10 ∧
  counts.millie + counts.monica + counts.marius = 41

/-- The theorem stating that Marius takes 4 more subjects than Monica. -/
theorem marius_monica_difference (counts : SubjectCounts) 
  (h : problem_conditions counts) : counts.marius - counts.monica = 4 := by
  sorry

end marius_monica_difference_l1625_162516


namespace equation_solution_l1625_162553

theorem equation_solution : ∃ x : ℚ, (5 * (x + 30) / 3 = (4 - 3 * x) / 7) ∧ (x = -519 / 22) := by
  sorry

end equation_solution_l1625_162553


namespace red_cube_possible_l1625_162549

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

end red_cube_possible_l1625_162549


namespace batsman_average_increase_l1625_162578

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

end batsman_average_increase_l1625_162578


namespace trig_identity_l1625_162557

open Real

theorem trig_identity (a b : ℝ) (θ : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (sin θ ^ 6 / a ^ 2 + cos θ ^ 6 / b ^ 2 = 1 / (a ^ 2 + b ^ 2)) →
  (sin θ ^ 12 / a ^ 5 + cos θ ^ 12 / b ^ 5 = 1 / a ^ 5) :=
by sorry

end trig_identity_l1625_162557


namespace division_of_decimals_l1625_162565

theorem division_of_decimals : (0.45 : ℚ) / (0.005 : ℚ) = 90 := by sorry

end division_of_decimals_l1625_162565


namespace ellipse_sum_bound_l1625_162523

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Theorem statement
theorem ellipse_sum_bound :
  ∀ x y : ℝ, ellipse x y → -4 ≤ x + 2*y ∧ x + 2*y ≤ 4 :=
by sorry

end ellipse_sum_bound_l1625_162523


namespace journey_time_proof_l1625_162508

/-- Proves that the total time to complete a 24 km journey is 8 hours, 
    given specific speed conditions. -/
theorem journey_time_proof (total_distance : ℝ) (initial_speed : ℝ) (initial_time : ℝ) 
    (remaining_speed : ℝ) : 
  total_distance = 24 →
  initial_speed = 4 →
  initial_time = 4 →
  remaining_speed = 2 →
  ∃ (total_time : ℝ), 
    total_time = 8 ∧ 
    total_distance = initial_speed * initial_time + 
      remaining_speed * (total_time - initial_time) :=
by
  sorry


end journey_time_proof_l1625_162508


namespace digit_2023_of_11_26_l1625_162528

/-- The repeating decimal representation of 11/26 -/
def repeating_decimal : List Nat := [4, 2, 3, 0, 7, 6]

/-- The length of the repeating decimal -/
def repeat_length : Nat := repeating_decimal.length

theorem digit_2023_of_11_26 : 
  -- The 2023rd digit past the decimal point in 11/26
  List.get! repeating_decimal ((2023 - 1) % repeat_length) = 4 := by
  sorry

end digit_2023_of_11_26_l1625_162528


namespace largest_operator_result_l1625_162587

theorem largest_operator_result : 
  let expr := (5 * Real.sqrt 2 - Real.sqrt 2)
  (expr * Real.sqrt 2 > expr + Real.sqrt 2) ∧
  (expr * Real.sqrt 2 > expr - Real.sqrt 2) ∧
  (expr * Real.sqrt 2 > expr / Real.sqrt 2) := by
  sorry

end largest_operator_result_l1625_162587


namespace inequality_solution_l1625_162535

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 5 / (x + 4) ≤ 2 - x) ↔ x ≤ 1 := by
  sorry

end inequality_solution_l1625_162535


namespace tobacco_acreage_change_l1625_162544

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

end tobacco_acreage_change_l1625_162544


namespace age_difference_is_51_l1625_162584

/-- The age difference between Milena's cousin Alex and her grandfather -/
def age_difference : ℕ :=
  let milena_age : ℕ := 7
  let grandmother_age : ℕ := 9 * milena_age
  let grandfather_age : ℕ := grandmother_age + 2
  let alex_age : ℕ := 2 * milena_age
  grandfather_age - alex_age

theorem age_difference_is_51 : age_difference = 51 := by
  sorry

end age_difference_is_51_l1625_162584


namespace M_properties_l1625_162525

-- Define the operation M
def M : ℚ → ℚ
| n => if (↑n : ℚ).den = 1 
       then (↑n : ℚ).num - 3 
       else -(1 / ((↑n : ℚ).den ^ 2))

-- Theorem statement
theorem M_properties : 
  (M 28 * M (1/5) = -1) ∧ 
  (-1 / M 39 / (-M (1/6)) = -1) := by
  sorry

end M_properties_l1625_162525


namespace crayons_per_child_l1625_162581

theorem crayons_per_child (total_crayons : ℕ) (num_children : ℕ) 
  (h1 : total_crayons = 72) (h2 : num_children = 12) :
  total_crayons / num_children = 6 := by
  sorry

end crayons_per_child_l1625_162581


namespace fixed_point_on_all_lines_l1625_162526

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


end fixed_point_on_all_lines_l1625_162526


namespace solve_for_a_l1625_162594

theorem solve_for_a (x a : ℚ) (h1 : 3 * x + 2 * a = 2) (h2 : x = 1) : a = -1/2 := by
  sorry

end solve_for_a_l1625_162594


namespace largest_prime_sum_of_digits_l1625_162509

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if a number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_prime_sum_of_digits :
  ∀ A B C D : ℕ,
    isSingleDigit A ∧ isSingleDigit B ∧ isSingleDigit C ∧ isSingleDigit D →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
    isPrime (A + B) ∧ isPrime (C + D) →
    (A + B) ≠ (C + D) →
    ∃ k : ℕ, k * (C + D) = A + B →
    ∀ E F : ℕ,
      isSingleDigit E ∧ isSingleDigit F →
      E ≠ F ∧ E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D ∧ F ≠ A ∧ F ≠ B ∧ F ≠ C ∧ F ≠ D →
      isPrime (E + F) →
      (E + F) ≠ (C + D) →
      ∃ m : ℕ, m * (C + D) = E + F →
      A + B ≥ E + F →
    A + B = 11
  := by sorry

end largest_prime_sum_of_digits_l1625_162509


namespace tan_105_minus_one_over_tan_105_plus_one_equals_sqrt_three_l1625_162547

theorem tan_105_minus_one_over_tan_105_plus_one_equals_sqrt_three :
  (Real.tan (105 * π / 180) - 1) / (Real.tan (105 * π / 180) + 1) = Real.sqrt 3 := by
  sorry

end tan_105_minus_one_over_tan_105_plus_one_equals_sqrt_three_l1625_162547


namespace triangle_area_l1625_162585

/-- The area of a triangle with side lengths 3, 5, and 7 is equal to 15√3/4 -/
theorem triangle_area (a b c : ℝ) (h_a : a = 3) (h_b : b = 5) (h_c : c = 7) :
  (1/2) * b * c * Real.sqrt (1 - ((b^2 + c^2 - a^2) / (2*b*c))^2) = (15 * Real.sqrt 3) / 4 := by
  sorry


end triangle_area_l1625_162585


namespace negative_fraction_comparison_l1625_162570

theorem negative_fraction_comparison : -4/5 > -5/6 := by
  sorry

end negative_fraction_comparison_l1625_162570


namespace handshake_count_l1625_162543

theorem handshake_count (n : ℕ) (h : n = 10) : n * (n - 1) / 2 = 45 := by
  sorry

end handshake_count_l1625_162543


namespace number_ordering_l1625_162536

theorem number_ordering : (10 ^ 5 : ℝ) < 2 ^ 20 ∧ 2 ^ 20 < 5 ^ 10 := by sorry

end number_ordering_l1625_162536


namespace angle_abc_measure_l1625_162559

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

end angle_abc_measure_l1625_162559


namespace sin_cos_eq_one_solutions_l1625_162586

theorem sin_cos_eq_one_solutions (x : Real) :
  x ∈ Set.Icc 0 Real.pi →
  (Real.sin x + Real.cos x = 1) ↔ (x = 0 ∨ x = Real.pi / 2) := by
  sorry

end sin_cos_eq_one_solutions_l1625_162586


namespace bryan_spent_1500_l1625_162513

/-- The total amount spent by Bryan on t-shirts and pants -/
def total_spent (num_tshirts : ℕ) (tshirt_price : ℕ) (num_pants : ℕ) (pants_price : ℕ) : ℕ :=
  num_tshirts * tshirt_price + num_pants * pants_price

/-- Theorem: Bryan spent $1500 on 5 t-shirts at $100 each and 4 pairs of pants at $250 each -/
theorem bryan_spent_1500 : total_spent 5 100 4 250 = 1500 := by
  sorry

end bryan_spent_1500_l1625_162513


namespace tree_growth_problem_l1625_162591

/-- A tree growth problem -/
theorem tree_growth_problem (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (target_height : ℝ) :
  initial_height = 5 →
  growth_rate = 3 →
  initial_age = 1 →
  target_height = 23 →
  ∃ (years : ℝ), 
    initial_height + growth_rate * years = target_height ∧
    years + initial_age = 7 :=
by sorry

end tree_growth_problem_l1625_162591


namespace fractional_equation_solution_l1625_162510

theorem fractional_equation_solution :
  ∃ (x : ℝ), (1 / x = 2 / (x + 3)) ∧ (x = 3) := by
  sorry

end fractional_equation_solution_l1625_162510


namespace fraction_increase_l1625_162548

theorem fraction_increase (x y : ℝ) (h : x + y ≠ 0) :
  (3 * (2 * x) * (2 * y)) / ((2 * x) + (2 * y)) = 2 * ((3 * x * y) / (x + y)) :=
by sorry

end fraction_increase_l1625_162548


namespace f_properties_l1625_162504

noncomputable section

def e : ℝ := Real.exp 1

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < e then -x^3 + x^2 else a * Real.log x

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z : ℝ, f a z = 0 → z = x ∨ z = y) ∧
  (∃ M N : ℝ × ℝ,
    let (xM, yM) := M
    let (xN, yN) := N
    xM > 0 ∧ xN < 0 ∧
    yM = f a xM ∧ yN = f a (-xN) ∧
    xM * xN + yM * yN = 0 ∧
    (xM - xN) * yM = xM * (yM - yN) ∧
    0 < a ∧ a ≤ 1 / (e + 1)) ∧
  (∀ a' : ℝ, a' ≤ 0 ∨ a' > 1 / (e + 1) →
    ¬∃ M N : ℝ × ℝ,
      let (xM, yM) := M
      let (xN, yN) := N
      xM > 0 ∧ xN < 0 ∧
      yM = f a' xM ∧ yN = f a' (-xN) ∧
      xM * xN + yM * yN = 0 ∧
      (xM - xN) * yM = xM * (yM - yN)) := by
  sorry

end f_properties_l1625_162504


namespace imaginary_part_of_z_l1625_162556

theorem imaginary_part_of_z (z : ℂ) : 
  z = Complex.I * (3 - 2 * Complex.I) * Complex.I ∧ z.re = 0 → z.im = 3 := by
  sorry

end imaginary_part_of_z_l1625_162556


namespace union_of_M_and_N_l1625_162542

-- Define the sets M and N
def M : Set ℕ := {0, 1}
def N : Set ℕ := {1, 2}

-- State the theorem
theorem union_of_M_and_N :
  M ∪ N = {0, 1, 2} := by sorry

end union_of_M_and_N_l1625_162542


namespace matrix_multiplication_l1625_162505

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 4]

theorem matrix_multiplication :
  A * B = !![17, -5; 16, -20] := by sorry

end matrix_multiplication_l1625_162505


namespace triangle_side_sum_l1625_162511

theorem triangle_side_sum (a b c : ℝ) (h_angles : a = 30 ∧ b = 45 ∧ c = 105) 
  (h_sum : a + b + c = 180) (h_side : ∃ side : ℝ, side = 6 * Real.sqrt 2) : 
  ∃ (x y : ℝ), x + y = 18 + 6 * Real.sqrt 3 :=
sorry

end triangle_side_sum_l1625_162511


namespace flowers_ratio_proof_l1625_162534

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

end flowers_ratio_proof_l1625_162534


namespace projection_implies_y_value_l1625_162541

/-- Given vectors v and w', if the projection of v on w' is proj_v_w, then y = -11/3 -/
theorem projection_implies_y_value (v w' proj_v_w : ℝ × ℝ) (y : ℝ) :
  v = (1, y) →
  w' = (-3, 1) →
  proj_v_w = (2, -2/3) →
  proj_v_w = (((v.1 * w'.1 + v.2 * w'.2) / (w'.1 ^ 2 + w'.2 ^ 2)) * w'.1,
              ((v.1 * w'.1 + v.2 * w'.2) / (w'.1 ^ 2 + w'.2 ^ 2)) * w'.2) →
  y = -11/3 := by
sorry

end projection_implies_y_value_l1625_162541


namespace complex_power_sum_l1625_162546

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^24 + 1/z^24 = -1 := by
  sorry

end complex_power_sum_l1625_162546


namespace f_properties_l1625_162531

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_properties (x₁ x₂ : ℝ) (h1 : x₂ > x₁) (h2 : x₁ > 1) :
  (|f x₁ - f x₂| < 1 / Real.exp 1) ∧ (f x₁ - f x₂ < x₂ - x₁) := by
  sorry

end f_properties_l1625_162531


namespace max_gcd_consecutive_terms_l1625_162573

def b (n : ℕ) : ℕ := (n + 2).factorial - n^2

theorem max_gcd_consecutive_terms :
  (∃ k : ℕ, Nat.gcd (b k) (b (k + 1)) = 5) ∧
  (∀ n : ℕ, Nat.gcd (b n) (b (n + 1)) ≤ 5) := by
  sorry

end max_gcd_consecutive_terms_l1625_162573


namespace town_population_males_l1625_162503

theorem town_population_males (total_population : ℕ) (num_segments : ℕ) (male_segments : ℕ) :
  total_population = 800 →
  num_segments = 4 →
  male_segments = 1 →
  2 * (total_population / num_segments * male_segments) = total_population →
  total_population / num_segments * male_segments = 400 :=
by sorry

end town_population_males_l1625_162503


namespace real_roots_iff_a_leq_two_l1625_162519

theorem real_roots_iff_a_leq_two (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 := by
  sorry

end real_roots_iff_a_leq_two_l1625_162519


namespace ryan_has_30_stickers_l1625_162502

/-- The number of stickers Ryan has -/
def ryan_stickers : ℕ := 30

/-- The number of stickers Steven has -/
def steven_stickers : ℕ := 3 * ryan_stickers

/-- The number of stickers Terry has -/
def terry_stickers : ℕ := steven_stickers + 20

/-- The total number of stickers -/
def total_stickers : ℕ := 230

theorem ryan_has_30_stickers :
  ryan_stickers + steven_stickers + terry_stickers = total_stickers ∧
  ryan_stickers = 30 := by sorry

end ryan_has_30_stickers_l1625_162502


namespace tournament_games_played_21_teams_l1625_162554

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


end tournament_games_played_21_teams_l1625_162554


namespace prob_at_least_two_women_l1625_162583

/-- The probability of selecting at least two women from a group of 9 men and 6 women when choosing 4 people at random -/
theorem prob_at_least_two_women (num_men : ℕ) (num_women : ℕ) (num_selected : ℕ) : 
  num_men = 9 → num_women = 6 → num_selected = 4 →
  (Nat.choose (num_men + num_women) num_selected - 
   (Nat.choose num_men num_selected + 
    num_women * Nat.choose num_men (num_selected - 1))) / 
  Nat.choose (num_men + num_women) num_selected = 7 / 13 :=
by sorry

end prob_at_least_two_women_l1625_162583


namespace prob_at_least_one_goes_l1625_162537

/-- The probability that at least one person goes to a place given individual probabilities -/
theorem prob_at_least_one_goes (prob_A prob_B : ℚ) (h_prob_A : prob_A = 1/4) (h_prob_B : prob_B = 2/5) :
  1 - (1 - prob_A) * (1 - prob_B) = 11/20 := by
  sorry

end prob_at_least_one_goes_l1625_162537


namespace inverse_variation_problem_l1625_162520

def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a * b = k

theorem inverse_variation_problem (a₁ a₂ b₁ b₂ : ℝ) 
  (h_inverse : inverse_relation a₁ b₁ ∧ inverse_relation a₂ b₂)
  (h_a₁ : a₁ = 1500)
  (h_b₁ : b₁ = 0.25)
  (h_a₂ : a₂ = 3000) :
  b₂ = 0.125 := by
sorry

end inverse_variation_problem_l1625_162520


namespace min_operations_to_500_l1625_162500

/-- Represents the available operations on the calculator --/
inductive Operation
  | addOne
  | subOne
  | mulTwo

/-- Applies an operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.addOne => n + 1
  | Operation.subOne => if n > 0 then n - 1 else 0
  | Operation.mulTwo => n * 2

/-- Checks if a sequence of operations contains all three operation types --/
def containsAllOperations (ops : List Operation) : Prop :=
  Operation.addOne ∈ ops ∧ Operation.subOne ∈ ops ∧ Operation.mulTwo ∈ ops

/-- Applies a sequence of operations to a starting number --/
def applyOperations (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Theorem: The minimum number of operations to reach 500 from 1 is 13 --/
theorem min_operations_to_500 :
  (∃ (ops : List Operation),
    applyOperations 1 ops = 500 ∧
    containsAllOperations ops ∧
    ops.length = 13) ∧
  (∀ (ops : List Operation),
    applyOperations 1 ops = 500 →
    containsAllOperations ops →
    ops.length ≥ 13) :=
  sorry

end min_operations_to_500_l1625_162500


namespace assignments_count_l1625_162567

/-- The number of assignments graded per hour initially -/
def initial_rate : ℕ := 6

/-- The number of assignments graded per hour after the change -/
def changed_rate : ℕ := 8

/-- The number of hours spent grading at the initial rate -/
def initial_hours : ℕ := 2

/-- The number of hours saved compared to the original plan -/
def hours_saved : ℕ := 3

/-- The total number of assignments in the batch -/
def total_assignments : ℕ := 84

/-- Theorem stating that the total number of assignments is 84 -/
theorem assignments_count :
  ∃ (x : ℕ), 
    (initial_rate * x = total_assignments) ∧ 
    (initial_rate * initial_hours + changed_rate * (x - initial_hours - hours_saved) = total_assignments) := by
  sorry

end assignments_count_l1625_162567


namespace odd_function_through_points_l1625_162593

/-- An odd function passing through two specific points -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem odd_function_through_points :
  (∀ x, f a b c (-x) = -(f a b c x)) →
  f a b c (-Real.sqrt 2) = Real.sqrt 2 →
  f a b c (2 * Real.sqrt 2) = 10 * Real.sqrt 2 →
  ∃ g : ℝ → ℝ, (∀ x, g x = x^3 - 3*x) ∧
              (∀ x, f a b c x = g x) ∧
              (∀ m, (∃! x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ g x₁ + m = 0 ∧ g x₂ + m = 0 ∧ g x₃ + m = 0) ↔
                    -2 < m ∧ m < 2) :=
sorry

end odd_function_through_points_l1625_162593


namespace independence_of_beta_l1625_162558

theorem independence_of_beta (α β : ℝ) : 
  ∃ (f : ℝ → ℝ), ∀ β, 
    (Real.sin (α + β))^2 + (Real.sin (β - α))^2 - 
    2 * Real.sin (α + β) * Real.sin (β - α) * Real.cos (2 * α) = f α :=
by sorry

end independence_of_beta_l1625_162558


namespace q_satisfies_conditions_l1625_162564

/-- A quadratic polynomial q(x) satisfying specific conditions -/
def q (x : ℚ) : ℚ := (31/15) * x^2 - (27/5) * x - 289/15

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-1) = 7 ∧ q 2 = -3 ∧ q 4 = 11 := by
  sorry

end q_satisfies_conditions_l1625_162564


namespace intersection_nonempty_implies_a_geq_neg_eight_l1625_162501

theorem intersection_nonempty_implies_a_geq_neg_eight (a : ℝ) : 
  (∃ x : ℝ, x ∈ {x | 1 ≤ x ∧ x ≤ 2} ∩ {x | x^2 + 2*x + a ≥ 0}) → a ≥ -8 := by
  sorry

end intersection_nonempty_implies_a_geq_neg_eight_l1625_162501


namespace marcus_pebbles_l1625_162552

theorem marcus_pebbles (initial_pebbles : ℕ) (current_pebbles : ℕ) 
  (h1 : initial_pebbles = 18)
  (h2 : current_pebbles = 39) :
  current_pebbles - (initial_pebbles - initial_pebbles / 2) = 30 := by
  sorry

end marcus_pebbles_l1625_162552


namespace ratio_nature_l1625_162551

theorem ratio_nature (x : ℝ) (m n : ℝ) (hx : x > 0) (hmn : m * n ≠ 0) (hineq : m * x > n * x + n) :
  m / (m + n) = (x + 1) / (2 * x + 1) := by
  sorry

end ratio_nature_l1625_162551


namespace limit_of_sequence_l1625_162522

def a (n : ℕ) : ℚ := (4 * n - 1) / (2 * n + 1)

theorem limit_of_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε :=
by
  sorry

end limit_of_sequence_l1625_162522


namespace complex_magnitude_l1625_162571

theorem complex_magnitude (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = 2*i + (9 - 3*i) / (1 + i) →
  Complex.abs z = 5 := by sorry

end complex_magnitude_l1625_162571


namespace inequality_proof_l1625_162514

theorem inequality_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  3 * ((x^2 / y^2) + (y^2 / x^2)) - 8 * ((x / y) + (y / x)) + 10 ≥ 0 := by
  sorry

end inequality_proof_l1625_162514


namespace vertical_shift_equivalence_l1625_162545

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the vertical shift transformation
def verticalShift (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f x + k

-- Theorem statement
theorem vertical_shift_equivalence :
  ∀ (x : ℝ), verticalShift f 2 x = f x + 2 := by sorry

end vertical_shift_equivalence_l1625_162545


namespace correct_withdrawal_amount_withdrawal_amount_2016_l1625_162529

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

end correct_withdrawal_amount_withdrawal_amount_2016_l1625_162529


namespace total_book_pairs_l1625_162527

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

end total_book_pairs_l1625_162527


namespace extended_tile_pattern_ratio_l1625_162561

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

end extended_tile_pattern_ratio_l1625_162561


namespace fireworks_display_l1625_162576

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

end fireworks_display_l1625_162576


namespace right_angled_triangle_sides_l1625_162517

theorem right_angled_triangle_sides : 
  (∃ (a b c : ℕ), (a = 5 ∧ b = 3 ∧ c = 4) ∧ a^2 = b^2 + c^2) ∧
  (∀ (a b c : ℕ), (a = 2 ∧ b = 3 ∧ c = 4) → a^2 ≠ b^2 + c^2) ∧
  (∀ (a b c : ℕ), (a = 4 ∧ b = 6 ∧ c = 9) → a^2 ≠ b^2 + c^2) ∧
  (∀ (a b c : ℕ), (a = 5 ∧ b = 11 ∧ c = 13) → a^2 ≠ b^2 + c^2) :=
by sorry

end right_angled_triangle_sides_l1625_162517


namespace alison_initial_stamps_l1625_162562

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

end alison_initial_stamps_l1625_162562


namespace quadratic_roots_sum_l1625_162538

theorem quadratic_roots_sum (m n : ℝ) : 
  (m^2 + m - 12 = 0) → (n^2 + n - 12 = 0) → m^2 + 2*m + n = 11 := by
  sorry

end quadratic_roots_sum_l1625_162538


namespace smallest_n_satisfying_condition_l1625_162521

/-- A function that calculates the probability of the given conditions for a given n -/
noncomputable def probability (n : ℕ) : ℝ :=
  ((n - 2)^3 + 3 * (n - 2) * (2 * n - 4)) / n^3

/-- The theorem stating that 12 is the smallest n satisfying the probability condition -/
theorem smallest_n_satisfying_condition :
  ∀ k : ℕ, k < 12 → probability k ≤ 3/4 ∧ probability 12 > 3/4 := by
  sorry

end smallest_n_satisfying_condition_l1625_162521


namespace point_line_plane_relation_l1625_162574

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

end point_line_plane_relation_l1625_162574


namespace power_comparison_l1625_162590

theorem power_comparison : (2 : ℝ)^30 < 10^10 ∧ 10^10 < 5^15 := by
  sorry

end power_comparison_l1625_162590


namespace employee_pays_216_l1625_162566

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the store markup percentage
def store_markup : ℝ := 0.20

-- Define the employee discount percentage
def employee_discount : ℝ := 0.10

-- Calculate the retail price
def retail_price : ℝ := wholesale_cost * (1 + store_markup)

-- Calculate the employee's final price
def employee_price : ℝ := retail_price * (1 - employee_discount)

-- Theorem to prove
theorem employee_pays_216 : employee_price = 216 := by sorry

end employee_pays_216_l1625_162566


namespace max_points_in_specific_tournament_l1625_162575

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

end max_points_in_specific_tournament_l1625_162575


namespace berry_theorem_l1625_162507

def berry_problem (total_needed : ℕ) (strawberry_cartons : ℕ) (blueberry_cartons : ℕ) : ℕ :=
  total_needed - (strawberry_cartons + blueberry_cartons)

theorem berry_theorem (total_needed strawberry_cartons blueberry_cartons : ℕ) :
  berry_problem total_needed strawberry_cartons blueberry_cartons =
  total_needed - (strawberry_cartons + blueberry_cartons) :=
by
  sorry

#eval berry_problem 42 2 7

end berry_theorem_l1625_162507


namespace even_product_square_sum_solution_l1625_162579

theorem even_product_square_sum_solution (a b : ℤ) (h : 2 ∣ (a * b)) :
  ∃ (x y : ℤ), a^2 + b^2 + x^2 = y^2 :=
by sorry

end even_product_square_sum_solution_l1625_162579


namespace basis_vectors_classification_l1625_162540

def is_basis (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.2 - v₁.2 * v₂.1 ≠ 0 ∧ v₁ ≠ (0, 0) ∧ v₂ ≠ (0, 0)

theorem basis_vectors_classification :
  let a₁ : ℝ × ℝ := (0, 0)
  let a₂ : ℝ × ℝ := (1, 2)
  let b₁ : ℝ × ℝ := (2, -1)
  let b₂ : ℝ × ℝ := (1, 2)
  let c₁ : ℝ × ℝ := (-1, -2)
  let c₂ : ℝ × ℝ := (1, 2)
  let d₁ : ℝ × ℝ := (1, 1)
  let d₂ : ℝ × ℝ := (1, 2)
  ¬(is_basis a₁ a₂) ∧
  ¬(is_basis c₁ c₂) ∧
  (is_basis b₁ b₂) ∧
  (is_basis d₁ d₂) :=
by sorry

end basis_vectors_classification_l1625_162540


namespace integral_sqrt_minus_x_equals_pi_half_l1625_162560

theorem integral_sqrt_minus_x_equals_pi_half :
  ∫ x in (-1)..(1), (Real.sqrt (1 - x^2) - x) = π / 2 := by sorry

end integral_sqrt_minus_x_equals_pi_half_l1625_162560


namespace parabola_vertex_sum_max_l1625_162555

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

end parabola_vertex_sum_max_l1625_162555


namespace parabola_line_intersection_l1625_162588

theorem parabola_line_intersection (α : Real) :
  (∃! x, 3 * x^2 + 1 = 4 * Real.sin α * x) →
  0 < α ∧ α < π / 2 →
  α = π / 3 := by
sorry

end parabola_line_intersection_l1625_162588


namespace tina_change_probability_l1625_162532

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

end tina_change_probability_l1625_162532


namespace base5_sum_theorem_l1625_162569

/-- Converts a base-10 number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a base-5 representation to base-10 -/
def fromBase5 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two base-5 numbers represented as lists -/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem base5_sum_theorem :
  let a := toBase5 259
  let b := toBase5 63
  addBase5 a b = [2, 2, 4, 2] := by sorry

end base5_sum_theorem_l1625_162569


namespace polynomial_coefficient_sum_l1625_162530

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ + a₄ = 120 := by
  sorry

end polynomial_coefficient_sum_l1625_162530


namespace linear_function_composition_l1625_162577

theorem linear_function_composition (a b : ℝ) :
  (∀ x : ℝ, (fun x => 3 * ((fun x => a * x + b) x) + 2) = (fun x => 4 * x - 1)) →
  a + b = 1 / 3 := by
sorry

end linear_function_composition_l1625_162577


namespace selection_methods_l1625_162572

theorem selection_methods (boys girls : ℕ) (tasks : ℕ) : 
  boys = 5 → girls = 4 → tasks = 3 →
  (Nat.choose boys 2 * Nat.choose girls 1 + Nat.choose boys 1 * Nat.choose girls 2) * Nat.factorial tasks = 420 := by
  sorry

end selection_methods_l1625_162572


namespace inequalities_in_quadrants_I_and_II_l1625_162582

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

end inequalities_in_quadrants_I_and_II_l1625_162582


namespace count_valid_pairs_l1625_162596

def valid_pair (b c : ℕ) : Prop :=
  1 ≤ b ∧ b ≤ 5 ∧ 1 ≤ c ∧ c ≤ 5 ∧ 
  (b * b : ℤ) - 4 * c ≤ 0 ∧ 
  (c * c : ℤ) - 4 * b ≤ 0

theorem count_valid_pairs : 
  ∃ (S : Finset (ℕ × ℕ)), S.card = 15 ∧ ∀ p, p ∈ S ↔ valid_pair p.1 p.2 :=
sorry

end count_valid_pairs_l1625_162596


namespace equilateral_hyperbola_properties_l1625_162506

/-- An equilateral hyperbola passing through point A(3,-1) with its axes of symmetry lying on the coordinate axes -/
def equilateral_hyperbola (x y : ℝ) : Prop :=
  x^2/8 - y^2/8 = 1

theorem equilateral_hyperbola_properties :
  -- The hyperbola passes through point A(3,-1)
  equilateral_hyperbola 3 (-1) ∧
  -- The axes of symmetry lie on the coordinate axes (implied by the equation form)
  ∀ (x y : ℝ), equilateral_hyperbola x y ↔ equilateral_hyperbola (-x) y ∧
  ∀ (x y : ℝ), equilateral_hyperbola x y ↔ equilateral_hyperbola x (-y) ∧
  -- The hyperbola is equilateral (asymptotes are perpendicular)
  ∃ (a : ℝ), a > 0 ∧ ∀ (x y : ℝ), equilateral_hyperbola x y ↔ x^2/a^2 - y^2/a^2 = 1 :=
by sorry

end equilateral_hyperbola_properties_l1625_162506


namespace sine_graph_shift_l1625_162539

theorem sine_graph_shift (x : ℝ) :
  3 * Real.sin (1/2 * (x - 4*π/5) + π/5) = 3 * Real.sin (1/2 * x - π/5) := by
  sorry

end sine_graph_shift_l1625_162539


namespace matrix_product_equality_l1625_162512

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 4]

theorem matrix_product_equality :
  A * B = !![17, -5; 16, -20] := by sorry

end matrix_product_equality_l1625_162512


namespace sin_cos_product_l1625_162595

theorem sin_cos_product (α : Real) 
  (h : (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7) : 
  Real.sin α * Real.cos α = 3 / 10 := by
sorry

end sin_cos_product_l1625_162595


namespace meena_cookie_sales_l1625_162515

/-- The number of dozens of cookies Meena sold to Mr. Stone -/
def cookies_sold_to_mr_stone (total_dozens : ℕ) (brock_cookies : ℕ) (katy_multiplier : ℕ) (cookies_left : ℕ) : ℕ :=
  let total_cookies := total_dozens * 12
  let katy_cookies := brock_cookies * katy_multiplier
  let sold_cookies := total_cookies - cookies_left
  let mr_stone_cookies := sold_cookies - (brock_cookies + katy_cookies)
  mr_stone_cookies / 12

theorem meena_cookie_sales : 
  cookies_sold_to_mr_stone 5 7 2 15 = 2 := by
  sorry

end meena_cookie_sales_l1625_162515
