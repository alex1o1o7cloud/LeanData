import Mathlib

namespace sector_tangent_problem_l2229_222930

theorem sector_tangent_problem (θ φ : Real) (h1 : 0 < θ) (h2 : θ < 2 * Real.pi) : 
  (1/2 * θ * 4^2 = 2 * Real.pi) → (Real.tan (θ + φ) = 3) → Real.tan φ = 1/2 := by
  sorry

end sector_tangent_problem_l2229_222930


namespace arithmetic_sequence_property_l2229_222988

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- Theorem: In an arithmetic sequence, if 3a₉ - a₁₅ - a₃ = 20, then 2a₈ - a₇ = 20 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h : arithmetic_sequence a) 
  (eq : 3 * a 9 - a 15 - a 3 = 20) : 
  2 * a 8 - a 7 = 20 := by
  sorry

end arithmetic_sequence_property_l2229_222988


namespace pizza_combinations_l2229_222913

def number_of_toppings : ℕ := 8
def toppings_per_pizza : ℕ := 3

theorem pizza_combinations :
  Nat.choose number_of_toppings toppings_per_pizza = 56 := by
  sorry

end pizza_combinations_l2229_222913


namespace sticker_difference_l2229_222987

theorem sticker_difference (belle_stickers carolyn_stickers : ℕ) 
  (h1 : belle_stickers = 97)
  (h2 : carolyn_stickers = 79)
  (h3 : carolyn_stickers < belle_stickers) : 
  belle_stickers - carolyn_stickers = 18 := by
  sorry

end sticker_difference_l2229_222987


namespace sqrt_equation_solution_l2229_222989

theorem sqrt_equation_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ Real.sqrt x + 2 * Real.sqrt (x^2 + 7*x) + Real.sqrt (x + 7) = 35 - 2*x ∧ x = 841/144 := by
  sorry

end sqrt_equation_solution_l2229_222989


namespace initial_average_problem_l2229_222952

theorem initial_average_problem (initial_count : Nat) (new_value : ℝ) (average_decrease : ℝ) :
  initial_count = 6 →
  new_value = 7 →
  average_decrease = 1 →
  ∃ initial_average : ℝ,
    initial_average * initial_count + new_value = 
    (initial_average - average_decrease) * (initial_count + 1) ∧
    initial_average = 14 := by
  sorry

end initial_average_problem_l2229_222952


namespace polynomial_division_theorem_l2229_222975

theorem polynomial_division_theorem (x : ℝ) : 
  ∃ (q r : ℝ), 8*x^4 + 7*x^3 + 3*x^2 - 5*x - 8 = (x - 1) * (8*x^3 + 15*x^2 + 18*x + 13) + 5 := by
  sorry

end polynomial_division_theorem_l2229_222975


namespace water_tank_capacity_l2229_222970

/-- The total capacity of a water tank in gallons -/
def tank_capacity : ℝ → Prop := λ T =>
  -- When the tank is 40% full, it contains 36 gallons less than when it is 70% full
  0.7 * T - 0.4 * T = 36

theorem water_tank_capacity : ∃ T : ℝ, tank_capacity T ∧ T = 120 := by
  sorry

end water_tank_capacity_l2229_222970


namespace arithmetic_sequence_sum_l2229_222955

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  d > 0 →  -- positive common difference
  a 1 + a 7 = 10 →  -- sum of roots condition
  a 1 * a 7 = 16 →  -- product of roots condition
  a 2 + a 4 + a 6 = 15 := by
  sorry

end arithmetic_sequence_sum_l2229_222955


namespace infinite_series_sum_l2229_222933

theorem infinite_series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series : ℕ → ℝ := fun n =>
    1 / ((n - 1) * a - (n - 3) * b) / (n * a - (2 * n - 3) * b)
  ∑' n, series n = 1 / ((a - b) * b) :=
by sorry

end infinite_series_sum_l2229_222933


namespace sequence_properties_l2229_222920

def a (n : ℤ) : ℤ := 30 + n - n^2

theorem sequence_properties :
  (a 10 = -60) ∧
  (∀ n : ℤ, a n = 0 ↔ n = 6) ∧
  (∀ n : ℤ, a n > 0 ↔ n > 6) ∧
  (∀ n : ℤ, a n < 0 ↔ n < 6) := by
  sorry

end sequence_properties_l2229_222920


namespace sam_apple_consumption_l2229_222922

/-- Calculates the number of apples Sam eats in a week -/
def apples_eaten_in_week (apples_per_sandwich : ℕ) (sandwiches_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  apples_per_sandwich * sandwiches_per_day * days_in_week

/-- Proves that Sam eats 280 apples in one week -/
theorem sam_apple_consumption : apples_eaten_in_week 4 10 7 = 280 := by
  sorry

#eval apples_eaten_in_week 4 10 7

end sam_apple_consumption_l2229_222922


namespace exactly_one_integer_satisfies_condition_l2229_222943

theorem exactly_one_integer_satisfies_condition : 
  ∃! (n : ℕ), n > 0 ∧ 20 - 5 * n ≥ 15 := by sorry

end exactly_one_integer_satisfies_condition_l2229_222943


namespace fraction_cube_equality_l2229_222912

theorem fraction_cube_equality : (81000 ^ 3) / (27000 ^ 3) = 27 := by sorry

end fraction_cube_equality_l2229_222912


namespace dave_tickets_proof_l2229_222903

/-- Represents the number of tickets Dave won initially -/
def initial_tickets : ℕ := 25

/-- Represents the number of tickets spent on a beanie -/
def spent_tickets : ℕ := 22

/-- Represents the number of additional tickets won -/
def additional_tickets : ℕ := 15

/-- Represents the final number of tickets Dave has -/
def final_tickets : ℕ := 18

/-- Proves that the initial number of tickets is correct given the problem conditions -/
theorem dave_tickets_proof :
  initial_tickets - spent_tickets + additional_tickets = final_tickets :=
by sorry

end dave_tickets_proof_l2229_222903


namespace jane_egg_income_l2229_222982

/-- Calculates the income from selling eggs given the number of chickens, eggs per chicken per week, 
    price per dozen eggs, and number of weeks. -/
def egg_income (num_chickens : ℕ) (eggs_per_chicken : ℕ) (price_per_dozen : ℚ) (num_weeks : ℕ) : ℚ :=
  (num_chickens * eggs_per_chicken * num_weeks : ℚ) / 12 * price_per_dozen

/-- Proves that Jane's income from selling eggs in 2 weeks is $20. -/
theorem jane_egg_income :
  egg_income 10 6 2 2 = 20 := by
  sorry

end jane_egg_income_l2229_222982


namespace ab_value_l2229_222980

theorem ab_value (a b : ℝ) 
  (h1 : (a + b)^2 + |b + 5| = b + 5)
  (h2 : 2 * a - b + 1 = 0) : 
  a * b = -1/9 := by
  sorry

end ab_value_l2229_222980


namespace inequality_proof_l2229_222937

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x * y + y * z + z * x = 6) : 
  (1 / (2 * Real.sqrt 2 + x^2 * (y + z))) + 
  (1 / (2 * Real.sqrt 2 + y^2 * (x + z))) + 
  (1 / (2 * Real.sqrt 2 + z^2 * (x + y))) ≤ 
  1 / (x * y * z) := by
sorry

end inequality_proof_l2229_222937


namespace total_vowels_written_l2229_222967

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 3

/-- Theorem: The total number of vowels written on the board is 15 -/
theorem total_vowels_written : num_vowels * times_written = 15 := by
  sorry

end total_vowels_written_l2229_222967


namespace vasya_has_more_placements_l2229_222966

/-- Represents a chessboard --/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a king placement on a board --/
def KingPlacement (b : Board) := Fin b.rows → Fin b.cols

/-- Predicate to check if a king placement is valid (no kings attack each other) --/
def IsValidPlacement (b : Board) (p : KingPlacement b) : Prop := sorry

/-- Number of valid king placements on a board --/
def NumValidPlacements (b : Board) (n : ℕ) : ℕ := sorry

/-- Petya's board --/
def PetyaBoard : Board := ⟨100, 50⟩

/-- Vasya's board (only white cells of a 100 × 100 checkerboard) --/
def VasyaBoard : Board := ⟨100, 50⟩

theorem vasya_has_more_placements :
  NumValidPlacements VasyaBoard 500 > NumValidPlacements PetyaBoard 500 := by
  sorry

end vasya_has_more_placements_l2229_222966


namespace triangle_count_is_36_l2229_222901

/-- A hexagon with diagonals and midpoint segments -/
structure HexagonWithDiagonalsAndMidpoints :=
  (vertices : Fin 6 → Point)
  (diagonals : List (Point × Point))
  (midpoint_segments : List (Point × Point))

/-- Count of triangles in the hexagon figure -/
def count_triangles (h : HexagonWithDiagonalsAndMidpoints) : ℕ :=
  sorry

/-- Theorem stating that the count of triangles is 36 -/
theorem triangle_count_is_36 (h : HexagonWithDiagonalsAndMidpoints) : 
  count_triangles h = 36 :=
sorry

end triangle_count_is_36_l2229_222901


namespace tetrahedron_section_theorem_l2229_222935

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron ABCD -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a plane defined by three points -/
structure Plane where
  P : Point3D
  Q : Point3D
  R : Point3D

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (M : Point3D) (A : Point3D) (D : Point3D) : Prop :=
  M.x = (A.x + D.x) / 2 ∧ M.y = (A.y + D.y) / 2 ∧ M.z = (A.z + D.z) / 2

/-- Check if a point is on the extension of a line segment -/
def isOnExtension (N : Point3D) (A : Point3D) (B : Point3D) : Prop :=
  ∃ t : ℝ, t > 1 ∧ N.x = A.x + t * (B.x - A.x) ∧
                 N.y = A.y + t * (B.y - A.y) ∧
                 N.z = A.z + t * (B.z - A.z)

/-- Calculate the ratio in which a plane divides a line segment -/
def divisionRatio (P : Plane) (A : Point3D) (B : Point3D) : ℝ × ℝ :=
  sorry

theorem tetrahedron_section_theorem (ABCD : Tetrahedron) (M N K : Point3D) :
  isMidpoint M ABCD.A ABCD.D →
  isOnExtension N ABCD.A ABCD.B →
  isOnExtension K ABCD.A ABCD.C →
  (N.x - ABCD.B.x)^2 + (N.y - ABCD.B.y)^2 + (N.z - ABCD.B.z)^2 =
    (ABCD.B.x - ABCD.A.x)^2 + (ABCD.B.y - ABCD.A.y)^2 + (ABCD.B.z - ABCD.A.z)^2 →
  (K.x - ABCD.C.x)^2 + (K.y - ABCD.C.y)^2 + (K.z - ABCD.C.z)^2 =
    4 * ((ABCD.C.x - ABCD.A.x)^2 + (ABCD.C.y - ABCD.A.y)^2 + (ABCD.C.z - ABCD.A.z)^2) →
  let P : Plane := {P := M, Q := N, R := K}
  divisionRatio P ABCD.D ABCD.B = (2, 1) ∧
  divisionRatio P ABCD.D ABCD.C = (3, 2) :=
by
  sorry

end tetrahedron_section_theorem_l2229_222935


namespace triangle_properties_l2229_222919

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  c * Real.sin B + (a + c^2 / a - b^2 / a) * Real.sin C = 2 * c * Real.sin A →
  (a * b * Real.sin C) / 2 = Real.sqrt 3 →
  a * Real.sin A / 2 = Real.sqrt 3 →
  b * Real.sin B / 2 = Real.sqrt 3 →
  c * Real.sin C / 2 = Real.sqrt 3 →
  C = π / 3 ∧ Real.cos (2 * A) - 2 * Real.sin B ^ 2 + 1 = -1 / 6 := by
sorry


end triangle_properties_l2229_222919


namespace pepper_spray_ratio_l2229_222968

theorem pepper_spray_ratio (total animals : ℕ) (raccoons : ℕ) : 
  total = 84 → raccoons = 12 → (total - raccoons) / raccoons = 6 := by
  sorry

end pepper_spray_ratio_l2229_222968


namespace sqrt_nine_equals_three_l2229_222932

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end sqrt_nine_equals_three_l2229_222932


namespace systematic_sampling_interval_example_l2229_222983

/-- The interval between segments in systematic sampling -/
def systematic_sampling_interval (N : ℕ) (n : ℕ) : ℕ :=
  N / n

/-- Theorem: For a population of 1500 and a sample size of 50, 
    the systematic sampling interval is 30 -/
theorem systematic_sampling_interval_example :
  systematic_sampling_interval 1500 50 = 30 := by
  sorry

end systematic_sampling_interval_example_l2229_222983


namespace paint_usage_l2229_222914

theorem paint_usage (total_paint : ℝ) (first_week_fraction : ℝ) (total_used : ℝ)
  (h1 : total_paint = 360)
  (h2 : first_week_fraction = 1 / 4)
  (h3 : total_used = 225) :
  let first_week_usage := first_week_fraction * total_paint
  let remaining_after_first_week := total_paint - first_week_usage
  let second_week_usage := total_used - first_week_usage
  second_week_usage / remaining_after_first_week = 1 / 2 := by
sorry

end paint_usage_l2229_222914


namespace complement_A_inter_B_when_m_is_one_one_in_A_union_B_iff_m_in_range_l2229_222954

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 9}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m + 4}

-- Part I
theorem complement_A_inter_B_when_m_is_one :
  (A ∩ B 1)ᶜ = {x | x < 3 ∨ x ≥ 6} := by sorry

-- Part II
theorem one_in_A_union_B_iff_m_in_range (m : ℝ) :
  (1 ∈ A ∪ B m) ↔ (-3/2 < m ∧ m < 0) := by sorry

end complement_A_inter_B_when_m_is_one_one_in_A_union_B_iff_m_in_range_l2229_222954


namespace largest_constant_inequality_l2229_222965

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), C = Real.sqrt (8 / 3) ∧
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z)) ∧
  (∀ (C' : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 2 ≥ C' * (x + y + z)) → C' ≤ C) :=
by
  sorry

end largest_constant_inequality_l2229_222965


namespace triangle_angle_A_l2229_222956

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Theorem statement
theorem triangle_angle_A (t : Triangle) :
  t.a = 3 ∧ t.b = 24/5 ∧ Real.cos t.B = 3/5 → t.A = 30 * π / 180 := by
  sorry

end triangle_angle_A_l2229_222956


namespace rain_difference_l2229_222997

/-- The amount of rain Greg experienced while camping, in millimeters. -/
def camping_rain : List ℝ := [3, 6, 5]

/-- The amount of rain at Greg's house during the same week, in millimeters. -/
def house_rain : ℝ := 26

/-- The difference in rainfall between Greg's house and his camping trip. -/
theorem rain_difference : house_rain - (camping_rain.sum) = 12 := by
  sorry

end rain_difference_l2229_222997


namespace geometric_sequence_seventh_term_l2229_222927

/-- Determinant of a 2x2 matrix --/
def det (a b c d : ℝ) : ℝ := a * d - b * c

/-- Geometric sequence --/
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geometric : isGeometric a)
  (h_third : a 3 = 1)
  (h_det : det (a 6) 8 8 (a 8) = 0) :
  a 7 = 8 := by
  sorry

end geometric_sequence_seventh_term_l2229_222927


namespace rollo_guinea_pigs_food_l2229_222962

/-- The amount of food needed to feed all guinea pigs -/
def total_food (first_pig_food second_pig_food third_pig_food : ℕ) : ℕ :=
  first_pig_food + second_pig_food + third_pig_food

/-- Theorem stating the total amount of food needed for Rollo's guinea pigs -/
theorem rollo_guinea_pigs_food :
  ∃ (first_pig_food second_pig_food third_pig_food : ℕ),
    first_pig_food = 2 ∧
    second_pig_food = 2 * first_pig_food ∧
    third_pig_food = second_pig_food + 3 ∧
    total_food first_pig_food second_pig_food third_pig_food = 13 :=
by
  sorry

#check rollo_guinea_pigs_food

end rollo_guinea_pigs_food_l2229_222962


namespace equation_solution_inequalities_solution_l2229_222908

-- Part 1: Equation solution
theorem equation_solution :
  ∃ x : ℝ, (2 * x / (x - 2) + 3 / (2 - x) = 1) ∧ (x = 1) := by sorry

-- Part 2: System of inequalities solution
theorem inequalities_solution :
  ∃ x : ℝ, (2 * x - 1 ≥ 3 * (x - 1)) ∧
           ((5 - x) / 2 < x + 3) ∧
           (-1/3 < x) ∧ (x ≤ 2) := by sorry

end equation_solution_inequalities_solution_l2229_222908


namespace presidency_meeting_arrangements_l2229_222931

/-- The number of schools --/
def num_schools : ℕ := 4

/-- The number of members per school --/
def members_per_school : ℕ := 5

/-- The number of representatives from the host school --/
def host_representatives : ℕ := 3

/-- The number of representatives from each non-host school --/
def non_host_representatives : ℕ := 1

/-- The total number of ways to arrange a presidency meeting --/
def meeting_arrangements : ℕ := num_schools * (Nat.choose members_per_school host_representatives) * (Nat.choose members_per_school non_host_representatives)^(num_schools - 1)

theorem presidency_meeting_arrangements :
  meeting_arrangements = 5000 :=
sorry

end presidency_meeting_arrangements_l2229_222931


namespace octal_minus_base9_equals_19559_l2229_222941

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem octal_minus_base9_equals_19559 : 
  let octal := [5, 4, 3, 2, 1]
  let base9 := [4, 3, 2, 1]
  base_to_decimal octal 8 - base_to_decimal base9 9 = 19559 := by
  sorry

end octal_minus_base9_equals_19559_l2229_222941


namespace pen_collection_l2229_222944

theorem pen_collection (initial_pens : ℕ) (mike_pens : ℕ) (sharon_pens : ℕ) : 
  initial_pens = 25 →
  mike_pens = 22 →
  sharon_pens = 19 →
  2 * (initial_pens + mike_pens) - sharon_pens = 75 := by
  sorry

end pen_collection_l2229_222944


namespace square_side_increase_l2229_222948

theorem square_side_increase (p : ℝ) : 
  (1 + p / 100)^2 = 1.21 → p = 10 := by
  sorry

end square_side_increase_l2229_222948


namespace original_bales_count_l2229_222924

/-- The number of bales Jason stacked today -/
def bales_stacked : ℕ := 23

/-- The total number of bales in the barn after Jason stacked -/
def total_bales : ℕ := 96

/-- The original number of bales in the barn -/
def original_bales : ℕ := total_bales - bales_stacked

theorem original_bales_count : original_bales = 73 := by
  sorry

end original_bales_count_l2229_222924


namespace plane_division_theorem_l2229_222951

/-- A line in the plane --/
structure Line where
  -- We don't need to define the actual properties of a line for this statement

/-- A set of lines in the plane --/
def LineSet := Set Line

/-- Predicate to check if all lines in a set are parallel to one of them --/
def allParallel (ls : LineSet) : Prop := sorry

/-- Number of regions formed by a set of lines --/
def numRegions (ls : LineSet) : ℕ := sorry

/-- Statement of the theorem --/
theorem plane_division_theorem :
  ∃ (k₀ : ℕ), ∀ (k : ℕ), k > k₀ →
    ∃ (ls : LineSet), ls.Finite ∧ ¬allParallel ls ∧ numRegions ls = k :=
by
  -- Let k₀ = 5
  use 5
  -- The rest of the proof
  sorry

end plane_division_theorem_l2229_222951


namespace oil_leak_height_l2229_222934

/-- The speed of oil leaking from a circular cylinder -/
def leak_speed (k : ℝ) (h : ℝ) : ℝ := k * h^2

theorem oil_leak_height (k : ℝ) (h' : ℝ) :
  (k > 0) →
  (leak_speed k 12 = 9 * leak_speed k h') →
  h' = 4 := by
sorry

end oil_leak_height_l2229_222934


namespace greatest_three_digit_multiple_of_17_l2229_222963

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 17 = 0 → n ≤ 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l2229_222963


namespace ball_probabilities_l2229_222976

structure Bag where
  red_balls : ℕ
  blue_balls : ℕ
  red_ones : ℕ
  blue_ones : ℕ

def total_balls (b : Bag) : ℕ := b.red_balls + b.blue_balls

def prob_one_red_sum_three (b : Bag) : ℚ := 16 / 81

def prob_first_red (b : Bag) : ℚ := b.red_balls / (total_balls b)

def prob_second_one (b : Bag) : ℚ := 1 / 3

theorem ball_probabilities (b : Bag) 
  (h1 : b.red_balls = 6) 
  (h2 : b.blue_balls = 3) 
  (h3 : b.red_ones = 2) 
  (h4 : b.blue_ones = 1) :
  prob_one_red_sum_three b = 16 / 81 ∧ 
  prob_first_red b = 2 / 3 ∧
  prob_second_one b = 1 / 3 ∧
  prob_first_red b * prob_second_one b = 
    (b.red_ones / (total_balls b)) * ((b.blue_ones) / (total_balls b - 1)) +
    ((b.red_balls - b.red_ones) / (total_balls b)) * (b.red_ones / (total_balls b - 1)) := by
  sorry

end ball_probabilities_l2229_222976


namespace shooter_probabilities_l2229_222971

/-- The probability of hitting the target in a single shot -/
def hit_probability : ℝ := 0.9

/-- The number of shots -/
def num_shots : ℕ := 4

/-- The probability of hitting the target on the third shot -/
def third_shot_probability : ℝ := hit_probability

/-- The probability of hitting the target at least once in four shots -/
def at_least_one_hit_probability : ℝ := 1 - (1 - hit_probability) ^ num_shots

/-- The number of correct statements -/
def correct_statements : ℕ := 2

theorem shooter_probabilities :
  (third_shot_probability = hit_probability) ∧
  (at_least_one_hit_probability = 1 - (1 - hit_probability) ^ num_shots) ∧
  (correct_statements = 2) := by sorry

end shooter_probabilities_l2229_222971


namespace dice_sum_probability_l2229_222972

theorem dice_sum_probability (n : ℕ) : n = 36 →
  ∃ (d1 d2 : Finset ℕ),
    d1.card = 6 ∧ d2.card = 6 ∧
    (∀ k : ℕ, k ∈ Finset.range (n + 1) →
      (∃! (x y : ℕ), x ∈ d1 ∧ y ∈ d2 ∧ x + y = k)) :=
by sorry

end dice_sum_probability_l2229_222972


namespace solve_system_l2229_222904

theorem solve_system (x y : ℤ) (h1 : x + y = 290) (h2 : x - y = 200) : y = 45 := by
  sorry

end solve_system_l2229_222904


namespace division_remainder_proof_l2229_222911

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 55053 →
  divisor = 456 →
  quotient = 120 →
  dividend = divisor * quotient + remainder →
  remainder = 333 := by
sorry

end division_remainder_proof_l2229_222911


namespace polynomial_properties_l2229_222926

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial (a b c : ℝ) (x : ℂ) : ℂ :=
  a * x^2 + b * x + c

/-- The polynomial we want to prove about -/
def our_polynomial (x : ℂ) : ℂ :=
  QuadraticPolynomial 2 (-12) 20 x

theorem polynomial_properties :
  (our_polynomial (3 + Complex.I) = 0) ∧
  (∀ x : ℂ, our_polynomial x = 2 * x^2 + (-12) * x + 20) :=
by sorry

end polynomial_properties_l2229_222926


namespace rectangle_width_l2229_222959

/-- Given a rectangular piece of metal with length 19 cm and perimeter 70 cm, 
    prove that its width is 16 cm. -/
theorem rectangle_width (length perimeter : ℝ) (h1 : length = 19) (h2 : perimeter = 70) :
  let width := (perimeter / 2) - length
  width = 16 := by
  sorry

end rectangle_width_l2229_222959


namespace inequality_solution_l2229_222999

theorem inequality_solution (x : ℝ) : 
  (x + 2) / (x + 3) > (4 * x + 5) / (3 * x + 10) ↔ 
  (x > -10/3 ∧ x < -1) ∨ x > 5 := by sorry

end inequality_solution_l2229_222999


namespace local_science_students_percentage_l2229_222960

/-- Proves that the percentage of local science students is 25% given the conditions of the problem -/
theorem local_science_students_percentage 
  (total_arts : ℕ) 
  (total_science : ℕ) 
  (total_commerce : ℕ) 
  (local_arts_percentage : ℚ) 
  (local_commerce_percentage : ℚ) 
  (total_local_percentage : ℚ) 
  (h1 : total_arts = 400) 
  (h2 : total_science = 100) 
  (h3 : total_commerce = 120) 
  (h4 : local_arts_percentage = 1/2) 
  (h5 : local_commerce_percentage = 17/20) 
  (h6 : total_local_percentage = 327/100) : 
  ∃ (local_science_percentage : ℚ), 
    local_science_percentage = 1/4 ∧ 
    (local_arts_percentage * total_arts + local_science_percentage * total_science + local_commerce_percentage * total_commerce) / (total_arts + total_science + total_commerce) = total_local_percentage := by
  sorry


end local_science_students_percentage_l2229_222960


namespace f_odd_and_increasing_l2229_222978

def f (x : ℝ) := x * |x|

theorem f_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x < f y) :=
sorry

end f_odd_and_increasing_l2229_222978


namespace least_five_digit_multiple_l2229_222909

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

theorem least_five_digit_multiple : ∃ (n : ℕ),
  n = 21000 ∧
  n ≥ 10000 ∧ n < 100000 ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m < n →
    ¬(is_divisible_by m 15 ∧
      is_divisible_by m 25 ∧
      is_divisible_by m 40 ∧
      is_divisible_by m 75 ∧
      is_divisible_by m 125 ∧
      is_divisible_by m 140)) ∧
  is_divisible_by n 15 ∧
  is_divisible_by n 25 ∧
  is_divisible_by n 40 ∧
  is_divisible_by n 75 ∧
  is_divisible_by n 125 ∧
  is_divisible_by n 140 :=
sorry

end least_five_digit_multiple_l2229_222909


namespace house_numbering_counts_l2229_222977

/-- Count of 9s in house numbers from 1 to n -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Total count of digits used in house numbers from 1 to n -/
def total_digits (n : ℕ) : ℕ := sorry

theorem house_numbering_counts :
  (count_nines 100 = 10) ∧ (total_digits 100 = 192) := by sorry

end house_numbering_counts_l2229_222977


namespace marjs_wallet_after_purchase_l2229_222994

/-- The amount of money left in Marj's wallet after buying a cake -/
def money_left_in_wallet (twenty_bills : ℕ) (five_bills : ℕ) (loose_coins : ℚ) (cake_cost : ℚ) : ℚ :=
  (twenty_bills * 20 + five_bills * 5 : ℚ) + loose_coins - cake_cost

/-- Theorem stating the amount of money left in Marj's wallet -/
theorem marjs_wallet_after_purchase :
  money_left_in_wallet 2 3 4.5 17.5 = 42 := by sorry

end marjs_wallet_after_purchase_l2229_222994


namespace sum_of_coefficients_eq_120_l2229_222925

def binomial_coefficient (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def sum_of_coefficients : ℕ :=
  (Finset.range 8).sum (fun i => binomial_coefficient (i + 2) 2)

theorem sum_of_coefficients_eq_120 : sum_of_coefficients = 120 := by
  sorry

end sum_of_coefficients_eq_120_l2229_222925


namespace digit_58_is_4_l2229_222929

/-- The repeating part of the decimal representation of 1/17 -/
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating part -/
def repeat_length : Nat := decimal_rep_1_17.length

/-- The 58th digit after the decimal point in the decimal representation of 1/17 -/
def digit_58 : Nat :=
  decimal_rep_1_17[(58 - 1) % repeat_length]

theorem digit_58_is_4 : digit_58 = 4 := by
  sorry

end digit_58_is_4_l2229_222929


namespace toms_floor_replacement_cost_l2229_222961

/-- The total cost to replace a floor given the room dimensions, removal cost, and new flooring cost per square foot. -/
def total_floor_replacement_cost (length width removal_cost cost_per_sqft : ℝ) : ℝ :=
  removal_cost + length * width * cost_per_sqft

/-- Theorem stating that the total cost to replace the floor in Tom's room is $120. -/
theorem toms_floor_replacement_cost :
  total_floor_replacement_cost 8 7 50 1.25 = 120 := by
  sorry

end toms_floor_replacement_cost_l2229_222961


namespace sin_cos_sum_greater_than_one_l2229_222915

theorem sin_cos_sum_greater_than_one (α : Real) (h : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin α + Real.cos α > 1 := by
  sorry

end sin_cos_sum_greater_than_one_l2229_222915


namespace mias_christmas_gifts_l2229_222993

/-- Proves that the amount spent on each parent's gift is $30 -/
theorem mias_christmas_gifts (total_spent : ℕ) (sibling_gift : ℕ) (num_siblings : ℕ) :
  total_spent = 150 ∧ sibling_gift = 30 ∧ num_siblings = 3 →
  ∃ (parent_gift : ℕ), 
    parent_gift * 2 + sibling_gift * num_siblings = total_spent ∧
    parent_gift = 30 :=
by sorry

end mias_christmas_gifts_l2229_222993


namespace complex_equality_implies_real_value_l2229_222906

theorem complex_equality_implies_real_value (a : ℝ) : 
  (Complex.re ((1 + 2*Complex.I) * (a + Complex.I)) = Complex.im ((1 + 2*Complex.I) * (a + Complex.I))) → 
  a = -3 := by
  sorry

end complex_equality_implies_real_value_l2229_222906


namespace spring_properties_l2229_222949

-- Define the spring's properties
def initial_length : ℝ := 20
def rate_of_change : ℝ := 0.5

-- Define the relationship between weight and length
def spring_length (weight : ℝ) : ℝ := initial_length + rate_of_change * weight

-- Theorem stating the properties of the spring
theorem spring_properties :
  (∀ w : ℝ, w ≥ 0 → spring_length w ≥ initial_length) ∧
  (∀ w1 w2 : ℝ, w1 < w2 → spring_length w1 < spring_length w2) ∧
  (∀ w : ℝ, (spring_length (w + 1) - spring_length w) = rate_of_change) :=
by sorry

end spring_properties_l2229_222949


namespace visibility_time_proof_l2229_222964

/-- The time when Jenny and Kenny become visible to each other after being blocked by a circular building -/
def visibilityTime (buildingRadius : ℝ) (pathDistance : ℝ) (jennySpeed : ℝ) (kennySpeed : ℝ) : ℝ :=
  120

theorem visibility_time_proof (buildingRadius : ℝ) (pathDistance : ℝ) (jennySpeed : ℝ) (kennySpeed : ℝ) 
    (h1 : buildingRadius = 60)
    (h2 : pathDistance = 240)
    (h3 : jennySpeed = 4)
    (h4 : kennySpeed = 2) :
  visibilityTime buildingRadius pathDistance jennySpeed kennySpeed = 120 :=
by
  sorry

#check visibility_time_proof

end visibility_time_proof_l2229_222964


namespace darnel_workout_l2229_222907

/-- Darnel's sprinting distances in miles -/
def sprint_distances : List ℝ := [0.8932, 0.7773, 0.9539, 0.5417, 0.6843]

/-- Darnel's jogging distances in miles -/
def jog_distances : List ℝ := [0.7683, 0.4231, 0.5733, 0.625, 0.6549]

/-- The difference between Darnel's total sprinting distance and total jogging distance -/
def sprint_jog_difference : ℝ := sprint_distances.sum - jog_distances.sum

theorem darnel_workout :
  sprint_jog_difference = 0.8058 := by sorry

end darnel_workout_l2229_222907


namespace pens_multiple_of_ten_l2229_222953

/-- Given that 920 pencils and some pens can be distributed equally among 10 students,
    prove that the number of pens must be a multiple of 10. -/
theorem pens_multiple_of_ten (num_pens : ℕ) (h : ∃ (pens_per_student : ℕ), num_pens = 10 * pens_per_student) :
  ∃ k : ℕ, num_pens = 10 * k := by
  sorry

end pens_multiple_of_ten_l2229_222953


namespace cubic_root_product_l2229_222905

theorem cubic_root_product (u v w : ℝ) : 
  (u^3 - 15*u^2 + 13*u - 6 = 0) →
  (v^3 - 15*v^2 + 13*v - 6 = 0) →
  (w^3 - 15*w^2 + 13*w - 6 = 0) →
  (1 + u) * (1 + v) * (1 + w) = 35 := by
sorry

end cubic_root_product_l2229_222905


namespace domino_placement_theorem_l2229_222945

/-- Represents a chessboard with dimensions n x n -/
structure Chessboard (n : ℕ) where
  size : n > 0

/-- Represents a domino with dimensions 1 x 2 -/
structure Domino where

/-- Represents a position on the chessboard -/
structure Position where
  x : ℝ
  y : ℝ

/-- Checks if a position is strictly within the chessboard boundaries -/
def Position.isWithinBoard (p : Position) (b : Chessboard n) : Prop :=
  0 < p.x ∧ p.x < n ∧ 0 < p.y ∧ p.y < n

/-- Represents a domino placement on the chessboard -/
structure DominoPlacement (b : Chessboard n) where
  center : Position
  isValid : center.isWithinBoard b

/-- Represents a configuration of domino placements on the chessboard -/
def Configuration (b : Chessboard n) := List (DominoPlacement b)

/-- Counts the number of dominoes in a configuration -/
def countDominoes (config : Configuration b) : ℕ := config.length

theorem domino_placement_theorem (b : Chessboard 8) :
  (∃ config : Configuration b, countDominoes config ≥ 40) ∧
  (∃ config : Configuration b, countDominoes config ≥ 41) ∧
  (∃ config : Configuration b, countDominoes config > 41) := by
  sorry

end domino_placement_theorem_l2229_222945


namespace expression_evaluation_l2229_222984

theorem expression_evaluation : 
  let x : ℚ := 3
  let y : ℚ := -3
  (1/2 * x - 2 * (x - 1/3 * y^2) + (-3/2 * x + 1/3 * y^2)) = 0 := by
  sorry

end expression_evaluation_l2229_222984


namespace cos_difference_angle_l2229_222991

theorem cos_difference_angle (α β : ℝ) : 
  Real.cos (α - β) = Real.cos α * Real.cos β + Real.sin α * Real.sin β := by
  sorry

end cos_difference_angle_l2229_222991


namespace abs_diff_eq_diff_abs_condition_l2229_222946

theorem abs_diff_eq_diff_abs_condition (a b : ℝ) :
  (∀ a b : ℝ, |a - b| = |a| - |b| → a * b ≥ 0) ∧
  (∃ a b : ℝ, a * b ≥ 0 ∧ |a - b| ≠ |a| - |b|) :=
by sorry

end abs_diff_eq_diff_abs_condition_l2229_222946


namespace fold_point_set_area_l2229_222942

/-- Triangle DEF with given side lengths and right angle -/
structure RightTriangle where
  de : ℝ
  df : ℝ
  angle_e_is_right : de^2 + ef^2 = df^2
  de_length : de = 24
  df_length : df = 48

/-- Set of fold points in the triangle -/
def FoldPointSet (t : RightTriangle) : Set (ℝ × ℝ) := sorry

/-- Area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Main theorem: Area of fold point set -/
theorem fold_point_set_area (t : RightTriangle) :
  area (FoldPointSet t) = 156 * Real.pi - 144 * Real.sqrt 3 := by sorry

end fold_point_set_area_l2229_222942


namespace marc_journey_fraction_l2229_222995

/-- Represents the time in minutes for a round trip journey -/
def roundTripTime (cyclingTime walkingTime : ℝ) : ℝ := cyclingTime + walkingTime

/-- Represents the time for Marc's modified journey -/
def modifiedJourneyTime (cyclingFraction : ℝ) : ℝ :=
  20 * cyclingFraction + 60 * (1 - cyclingFraction)

theorem marc_journey_fraction :
  ∃ (cyclingFraction : ℝ),
    roundTripTime 20 60 = 80 ∧
    modifiedJourneyTime cyclingFraction = 52 ∧
    cyclingFraction = 1/5 := by
  sorry

end marc_journey_fraction_l2229_222995


namespace wills_earnings_after_deductions_l2229_222902

/-- Calculates Will's earnings after tax deductions for a 5-day work week --/
def willsEarnings (monday_wage monday_hours tuesday_wage tuesday_hours
                   wednesday_wage wednesday_hours thursday_wage thursday_hours
                   friday_wage friday_hours tax_rate : ℝ) : ℝ :=
  let total_earnings := monday_wage * monday_hours +
                        tuesday_wage * tuesday_hours +
                        wednesday_wage * wednesday_hours +
                        thursday_wage * thursday_hours +
                        friday_wage * friday_hours
  let tax_deduction := total_earnings * tax_rate
  total_earnings - tax_deduction

/-- Theorem stating Will's earnings after deductions --/
theorem wills_earnings_after_deductions :
  willsEarnings 8 8 10 2 9 6 7 4 7 4 0.12 = 170.72 := by
  sorry

end wills_earnings_after_deductions_l2229_222902


namespace regular_polygon_exterior_angle_18_has_20_sides_l2229_222958

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_has_20_sides :
  ∀ n : ℕ, 
  n > 0 → 
  (360 : ℝ) / n = 18 → 
  n = 20 :=
by
  sorry

end regular_polygon_exterior_angle_18_has_20_sides_l2229_222958


namespace movie_admission_price_l2229_222947

theorem movie_admission_price (regular_price : ℝ) : 
  (∀ discounted_price : ℝ, 
    discounted_price = regular_price - 3 →
    6 * discounted_price = 30) →
  regular_price = 8 := by
sorry

end movie_admission_price_l2229_222947


namespace constant_term_expansion_l2229_222950

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function for the general term of the expansion
def generalTerm (r : ℕ) : ℤ :=
  (-1)^r * 2^(4 - r) * binomial 4 r

-- Theorem statement
theorem constant_term_expansion :
  generalTerm 2 = 24 :=
by sorry

end constant_term_expansion_l2229_222950


namespace complex_equation_solution_l2229_222986

theorem complex_equation_solution :
  ∀ (z : ℂ), z * (Complex.I - 1) = 2 * Complex.I → z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l2229_222986


namespace equation_solution_l2229_222973

theorem equation_solution : ∃ x : ℝ, x * 15 - x * (2/3) + 1.4 = 10 ∧ x = 0.6 := by
  sorry

end equation_solution_l2229_222973


namespace sum_equals_three_halves_l2229_222916

theorem sum_equals_three_halves : 
  let original_sum := (1 : ℚ) / 3 + 1 / 5 + 1 / 7 + 1 / 9 + 1 / 11 + 1 / 13 + 1 / 15
  let removed_terms := 1 / 13 + 1 / 15
  original_sum - removed_terms = 3 / 2 →
  (1 : ℚ) / 3 + 1 / 5 + 1 / 7 + 1 / 9 + 1 / 11 = 3 / 2 := by
  sorry

end sum_equals_three_halves_l2229_222916


namespace next_square_property_number_l2229_222921

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_square_property (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens_ones := n % 100
  is_perfect_square (hundreds * tens_ones)

theorem next_square_property_number :
  ∀ n : ℕ,
    1818 < n →
    n < 10000 →
    has_square_property n →
    (∀ m : ℕ, 1818 < m → m < n → ¬has_square_property m) →
    n = 1832 :=
sorry

end next_square_property_number_l2229_222921


namespace least_five_digit_congruent_to_7_mod_18_l2229_222923

theorem least_five_digit_congruent_to_7_mod_18 :
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ n % 18 = 7 → n ≥ 10015 :=
by
  sorry

end least_five_digit_congruent_to_7_mod_18_l2229_222923


namespace egg_distribution_l2229_222990

theorem egg_distribution (total_eggs : Nat) (num_students : Nat) 
  (h1 : total_eggs = 73) (h2 : num_students = 9) :
  ∃ (eggs_per_student : Nat) (leftover : Nat),
    total_eggs = num_students * eggs_per_student + leftover ∧
    eggs_per_student = 8 ∧
    leftover = 1 := by
  sorry

end egg_distribution_l2229_222990


namespace train_length_l2229_222985

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 → time_s = 12 → speed_kmh * (1000 / 3600) * time_s = 300 := by
  sorry

end train_length_l2229_222985


namespace solution_set_equals_plus_minus_one_l2229_222969

def solution_set : Set ℝ := {x | x^2 - 1 = 0}

theorem solution_set_equals_plus_minus_one : solution_set = {-1, 1} := by
  sorry

end solution_set_equals_plus_minus_one_l2229_222969


namespace equal_benefit_credit_debit_l2229_222979

/-- Represents the benefit of using a card for a purchase -/
structure CardBenefit where
  purchase_amount : ℝ
  cashback_rate : ℝ
  interest_rate : ℝ

/-- Calculates the net benefit of using a card after one month -/
def net_benefit (card : CardBenefit) : ℝ :=
  card.purchase_amount * card.cashback_rate + card.purchase_amount * card.interest_rate

/-- The purchase amount in rubles -/
def purchase_amount : ℝ := 10000

/-- Theorem stating that the net benefit is equal for both credit and debit cards -/
theorem equal_benefit_credit_debit :
  let credit_card := CardBenefit.mk purchase_amount 0.005 0.005
  let debit_card := CardBenefit.mk purchase_amount 0.01 0
  net_benefit credit_card = net_benefit debit_card :=
by sorry

end equal_benefit_credit_debit_l2229_222979


namespace largest_number_in_ratio_l2229_222940

theorem largest_number_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b / a = 5 / 4 →
  c / a = 6 / 4 →
  (a + b + c) / 3 = 20 →
  max a (max b c) = 24 := by
sorry

end largest_number_in_ratio_l2229_222940


namespace number_thought_of_l2229_222974

theorem number_thought_of : ∃ x : ℝ, (x / 4 + 9 = 15) ∧ (x = 24) := by
  sorry

end number_thought_of_l2229_222974


namespace unique_fifth_power_solution_l2229_222910

theorem unique_fifth_power_solution :
  ∀ x y : ℕ, x^5 = y^5 + 10*y^2 + 20*y + 1 → (x = 1 ∧ y = 0) := by
  sorry

end unique_fifth_power_solution_l2229_222910


namespace final_sum_after_operations_l2229_222981

theorem final_sum_after_operations (a b S : ℝ) : 
  a + b = S → 3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry

end final_sum_after_operations_l2229_222981


namespace board_crossing_area_l2229_222992

/-- The area of the parallelogram formed by two boards crossed at a 45-degree angle -/
theorem board_crossing_area (width1 width2 : ℝ) (angle : ℝ) : 
  width1 = 5 → width2 = 6 → angle = π/4 → 
  width2 * width1 = 30 := by sorry

end board_crossing_area_l2229_222992


namespace negation_of_all_x_squared_plus_one_positive_l2229_222918

theorem negation_of_all_x_squared_plus_one_positive :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by sorry

end negation_of_all_x_squared_plus_one_positive_l2229_222918


namespace area_of_ring_area_of_specific_ring_l2229_222957

/-- The area of a ring formed by two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h : r₁ > r₂) : 
  (π * r₁^2 - π * r₂^2 : ℝ) = π * (r₁^2 - r₂^2) :=
sorry

/-- The area of a ring formed by two concentric circles with radii 10 and 6 is 64π -/
theorem area_of_specific_ring : 
  (π * 10^2 - π * 6^2 : ℝ) = 64 * π :=
sorry

end area_of_ring_area_of_specific_ring_l2229_222957


namespace problem_solution_l2229_222939

theorem problem_solution (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 7) : 
  (x + y)^2 - x*y = 1183/36 := by
  sorry

end problem_solution_l2229_222939


namespace range_of_m_l2229_222996

def f (x : ℝ) : ℝ := sorry

theorem range_of_m (h1 : ∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≠ 0)
                   (h2 : ∀ x, f (-x) = -f x)
                   (h3 : ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ 2 → f x > f y)
                   (h4 : ∀ m, f (1 + m) + f m < 0) :
  ∀ m, (-1/2 < m ∧ m ≤ 1) ↔ (∃ x, -2 ≤ x ∧ x ≤ 2 ∧ f (1 + x) + f x < 0) :=
by sorry

end range_of_m_l2229_222996


namespace equation_solution_l2229_222917

theorem equation_solution (k x m n : ℝ) :
  (∃ x, ∀ k, 2 * k * x + 2 * m = 6 - 2 * x + n * k) →
  4 * m + 2 * n = 12 := by
sorry

end equation_solution_l2229_222917


namespace sequence_not_periodic_l2229_222928

theorem sequence_not_periodic (x : ℝ) (h1 : x > 1) (h2 : ¬ ∃ n : ℤ, x = n) : 
  ¬ ∃ p : ℕ, ∀ n : ℕ, (⌊x^(n+1)⌋ - x * ⌊x^n⌋) = (⌊x^(n+1+p)⌋ - x * ⌊x^(n+p)⌋) :=
by sorry

end sequence_not_periodic_l2229_222928


namespace ABC_reflection_collinear_l2229_222900

-- Define the basic structures
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ)

-- Define the triangle ABC
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry

-- Define point P and line γ
def P : Point := sorry
def γ : Line := sorry

-- Define the reflection of a line with respect to another line
def reflect (l₁ l₂ : Line) : Line := sorry

-- Define the intersection of two lines
def intersect (l₁ l₂ : Line) : Point := sorry

-- Define lines PA, PB, PC
def PA : Line := sorry
def PB : Line := sorry
def PC : Line := sorry

-- Define lines BC, AC, AB
def BC : Line := sorry
def AC : Line := sorry
def AB : Line := sorry

-- Define points A', B', C'
def A' : Point := intersect (reflect PA γ) BC
def B' : Point := intersect (reflect PB γ) AC
def C' : Point := intersect (reflect PC γ) AB

-- Define collinearity
def collinear (p q r : Point) : Prop := sorry

-- The theorem to be proved
theorem ABC_reflection_collinear : collinear A' B' C' := by sorry

end ABC_reflection_collinear_l2229_222900


namespace inverse_proportion_example_l2229_222936

/-- Two real numbers are inversely proportional if their product is constant. -/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ t : ℝ, x t * y t = c

theorem inverse_proportion_example :
  ∀ x y : ℝ → ℝ,
  InverselyProportional x y →
  x 5 = 40 →
  x 10 = 20 := by
sorry

end inverse_proportion_example_l2229_222936


namespace at_least_one_red_certain_l2229_222998

-- Define the total number of balls
def total_balls : ℕ := 8

-- Define the number of red balls
def red_balls : ℕ := 5

-- Define the number of white balls
def white_balls : ℕ := 3

-- Define the number of balls drawn
def drawn_balls : ℕ := 4

-- Theorem statement
theorem at_least_one_red_certain :
  ∀ (draw : Finset ℕ),
  draw.card = drawn_balls →
  draw ⊆ Finset.range total_balls →
  ∃ (x : ℕ), x ∈ draw ∧ x < red_balls :=
sorry

end at_least_one_red_certain_l2229_222998


namespace power_multiplication_l2229_222938

theorem power_multiplication (x : ℝ) : x^3 * x^4 = x^7 := by
  sorry

end power_multiplication_l2229_222938
