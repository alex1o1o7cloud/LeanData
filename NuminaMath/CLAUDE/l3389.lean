import Mathlib

namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l3389_338900

/-- Sum of first n positive even integers -/
def sum_even (n : ℕ) : ℕ := 2 * n * (n + 1)

/-- Sum of first n positive odd integers -/
def sum_odd (n : ℕ) : ℕ := n * n

/-- The positive difference between the sum of the first 30 positive even integers
    and the sum of the first 30 positive odd integers is 30 -/
theorem even_odd_sum_difference : sum_even 30 - sum_odd 30 = 30 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l3389_338900


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3389_338908

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3389_338908


namespace NUMINAMATH_CALUDE_bride_groom_age_difference_l3389_338994

theorem bride_groom_age_difference :
  ∀ (bride_age groom_age : ℕ),
    bride_age = 102 →
    bride_age + groom_age = 185 →
    bride_age - groom_age = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_bride_groom_age_difference_l3389_338994


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3389_338980

theorem arithmetic_geometric_mean_inequality 
  (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a + b > 2 * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3389_338980


namespace NUMINAMATH_CALUDE_constant_expression_l3389_338963

theorem constant_expression (x y : ℝ) (h : x + y = 1) :
  let a := Real.sqrt (1 + x^2)
  let b := Real.sqrt (1 + y^2)
  (a + b + 1) * (a + b - 1) * (a - b + 1) * (-a + b + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_constant_expression_l3389_338963


namespace NUMINAMATH_CALUDE_cone_height_from_lateral_surface_l3389_338962

/-- If the lateral surface of a cone, when unfolded, forms a semicircle with an area of 2π,
    then the height of the cone is √3. -/
theorem cone_height_from_lateral_surface (r h : ℝ) : 
  r > 0 → h > 0 → 2 * π = π * (r^2 + h^2) → h = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_from_lateral_surface_l3389_338962


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3389_338995

theorem trig_expression_equality :
  1 / Real.sin (70 * π / 180) - Real.sqrt 2 / Real.cos (70 * π / 180) =
  -2 * Real.sin (25 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3389_338995


namespace NUMINAMATH_CALUDE_valid_sequences_of_length_21_l3389_338902

/-- Counts valid sequences of 0s and 1s of given length -/
def countValidSequences (n : ℕ) : ℕ :=
  if n ≤ 4 then 0
  else if n = 5 then 1
  else if n = 6 then 2
  else if n = 7 then 2
  else countValidSequences (n - 4) + 2 * countValidSequences (n - 5) + countValidSequences (n - 6)

/-- Theorem stating the number of valid sequences of length 21 -/
theorem valid_sequences_of_length_21 :
  countValidSequences 21 = 114 := by
  sorry

end NUMINAMATH_CALUDE_valid_sequences_of_length_21_l3389_338902


namespace NUMINAMATH_CALUDE_mean_proportional_234_104_l3389_338916

theorem mean_proportional_234_104 : ∃ x : ℝ, x^2 = 234 * 104 ∧ x = 156 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_234_104_l3389_338916


namespace NUMINAMATH_CALUDE_division_problem_l3389_338936

theorem division_problem (A : ℕ) : A = 8 ↔ 41 = 5 * A + 1 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3389_338936


namespace NUMINAMATH_CALUDE_art_group_size_l3389_338909

/-- The number of students in the art interest group -/
def num_students : ℕ := 6

/-- The total number of colored papers when each student cuts 10 pieces -/
def total_papers_10 (x : ℕ) : ℕ := 10 * x + 6

/-- The total number of colored papers when each student cuts 12 pieces -/
def total_papers_12 (x : ℕ) : ℕ := 12 * x - 6

/-- Theorem stating that the number of students satisfies the given conditions -/
theorem art_group_size :
  total_papers_10 num_students = total_papers_12 num_students :=
by sorry

end NUMINAMATH_CALUDE_art_group_size_l3389_338909


namespace NUMINAMATH_CALUDE_geometric_arithmetic_geometric_progression_l3389_338906

theorem geometric_arithmetic_geometric_progression
  (a b c : ℝ) :
  (∃ q : ℝ, b = a * q ∧ c = a * q^2) →  -- Initial geometric progression
  (2 * (b + 2) = a + c) →               -- Arithmetic progression after increasing b by 2
  ((b + 2)^2 = a * (c + 9)) →           -- Geometric progression after increasing c by 9
  ((a = 4/25 ∧ b = -16/25 ∧ c = 64/25) ∨ (a = 4 ∧ b = 8 ∧ c = 16)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_geometric_progression_l3389_338906


namespace NUMINAMATH_CALUDE_roots_equation_l3389_338920

theorem roots_equation (α β : ℝ) : 
  α^2 - 3*α + 1 = 0 → 
  β^2 - 3*β + 1 = 0 → 
  7 * α^5 + 8 * β^4 = 1448 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_l3389_338920


namespace NUMINAMATH_CALUDE_sum_division_l3389_338954

/-- The problem of dividing a sum among four people with specific ratios -/
theorem sum_division (w x y z : ℝ) (total : ℝ) : 
  w > 0 ∧ 
  x = 0.8 * w ∧ 
  y = 0.65 * w ∧ 
  z = 0.45 * w ∧
  y = 78 →
  total = w + x + y + z ∧ total = 348 := by
  sorry

end NUMINAMATH_CALUDE_sum_division_l3389_338954


namespace NUMINAMATH_CALUDE_phil_wins_n_12_ellie_wins_n_2012_l3389_338941

/-- Represents a move on the chessboard -/
structure Move where
  x : Nat
  y : Nat
  shape : Fin 4 -- 4 possible L-shapes

/-- Represents the state of the chessboard -/
def Board (n : Nat) := Fin n → Fin n → Fin n

/-- Applies a move to the board -/
def applyMove (n : Nat) (board : Board n) (move : Move) : Board n :=
  sorry

/-- Checks if all numbers on the board are zero -/
def allZero (n : Nat) (board : Board n) : Prop :=
  sorry

/-- Sum of all numbers on the board -/
def boardSum (n : Nat) (board : Board n) : Nat :=
  sorry

theorem phil_wins_n_12 :
  ∃ (initial : Board 12),
    ∀ (moves : List Move),
      ¬(boardSum 12 (moves.foldl (applyMove 12) initial) % 3 = 0) :=
sorry

theorem ellie_wins_n_2012 :
  ∀ (initial : Board 2012),
    ∃ (moves : List Move),
      allZero 2012 (moves.foldl (applyMove 2012) initial) :=
sorry

end NUMINAMATH_CALUDE_phil_wins_n_12_ellie_wins_n_2012_l3389_338941


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l3389_338923

-- Define the equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (|m| - 1) - y^2 / (m - 2) = 1

-- Define the condition for the equation to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, hyperbola_equation x y m

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  (-1 < m ∧ m < 1) ∨ m > 2

-- Theorem statement
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ m_range m := by sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l3389_338923


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l3389_338991

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) :
  (2 * r) ^ 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l3389_338991


namespace NUMINAMATH_CALUDE_stream_speed_l3389_338951

/-- The speed of a stream given downstream and upstream speeds -/
theorem stream_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 14)
  (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3389_338951


namespace NUMINAMATH_CALUDE_locus_of_M_l3389_338990

-- Define the points A, B, and M
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the angle function
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define the condition for point M
def satisfies_angle_condition (M : ℝ × ℝ) : Prop :=
  angle M B A = 2 * angle M A B

-- Define the locus conditions
def on_hyperbola (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  3 * x^2 - y^2 = 3 ∧ x > -1

def on_segment (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  y = 0 ∧ -1 < x ∧ x < 2

-- State the theorem
theorem locus_of_M (M : ℝ × ℝ) :
  satisfies_angle_condition M ↔ (on_hyperbola M ∨ on_segment M) :=
sorry

end NUMINAMATH_CALUDE_locus_of_M_l3389_338990


namespace NUMINAMATH_CALUDE_washing_time_is_seven_hours_l3389_338989

/-- Calculates the number of cycles needed for a given number of items and capacity per cycle -/
def cycles_needed (items : ℕ) (capacity : ℕ) : ℕ :=
  (items + capacity - 1) / capacity

/-- Calculates the total washing time in minutes -/
def total_washing_time (shirts pants sweaters jeans socks scarves : ℕ) 
  (regular_capacity sock_capacity scarf_capacity : ℕ)
  (regular_time sock_time scarf_time : ℕ) : ℕ :=
  let regular_cycles := cycles_needed shirts regular_capacity + 
                        cycles_needed pants regular_capacity + 
                        cycles_needed sweaters regular_capacity + 
                        cycles_needed jeans regular_capacity
  let sock_cycles := cycles_needed socks sock_capacity
  let scarf_cycles := cycles_needed scarves scarf_capacity
  regular_cycles * regular_time + sock_cycles * sock_time + scarf_cycles * scarf_time

theorem washing_time_is_seven_hours :
  total_washing_time 18 12 17 13 10 8 15 10 5 45 30 60 = 7 * 60 := by
  sorry

end NUMINAMATH_CALUDE_washing_time_is_seven_hours_l3389_338989


namespace NUMINAMATH_CALUDE_triangle_side_length_l3389_338917

/-- A square with side length 10 cm is divided into two right trapezoids and a right triangle. -/
structure DividedSquare where
  /-- Side length of the square -/
  side_length : ℝ
  /-- Height of the trapezoids -/
  trapezoid_height : ℝ
  /-- Area difference between the trapezoids -/
  area_difference : ℝ
  /-- Length of one side of the right triangle -/
  triangle_side : ℝ
  /-- The side length is 10 cm -/
  side_length_eq : side_length = 10
  /-- The area difference between trapezoids is 10 cm² -/
  area_difference_eq : area_difference = 10
  /-- The trapezoids have equal height -/
  trapezoid_height_eq : trapezoid_height = side_length / 2

/-- The theorem to be proved -/
theorem triangle_side_length (s : DividedSquare) : s.triangle_side = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3389_338917


namespace NUMINAMATH_CALUDE_tree_balance_exists_l3389_338945

def LetterWeight : Char → ℕ
  | 'O' => 300
  | 'B' => 300
  | 'M' => 200
  | 'E' => 200
  | 'P' => 100
  | _ => 0

def InitialLeftWeight : ℕ := LetterWeight 'M' + LetterWeight 'B'
def InitialRightWeight : ℕ := LetterWeight 'P' + LetterWeight 'E'

def RemainingLetters : List Char := ['O', 'O', 'B', 'B', 'M', 'E', 'P']

theorem tree_balance_exists :
  ∃ (left right : List Char),
    left.length + right.length = RemainingLetters.length ∧
    (left.map LetterWeight).sum + InitialLeftWeight =
    (right.map LetterWeight).sum + InitialRightWeight :=
  sorry

end NUMINAMATH_CALUDE_tree_balance_exists_l3389_338945


namespace NUMINAMATH_CALUDE_doctor_lindsay_daily_income_is_2200_l3389_338914

/-- Calculates the total money Doctor Lindsay receives in a typical 8-hour day -/
def doctor_lindsay_daily_income : ℕ := by
  -- Define the number of adult patients per hour
  let adult_patients_per_hour : ℕ := 4
  -- Define the number of child patients per hour
  let child_patients_per_hour : ℕ := 3
  -- Define the cost for an adult's office visit
  let adult_visit_cost : ℕ := 50
  -- Define the cost for a child's office visit
  let child_visit_cost : ℕ := 25
  -- Define the number of working hours per day
  let working_hours_per_day : ℕ := 8
  
  -- Calculate the total income
  exact adult_patients_per_hour * adult_visit_cost * working_hours_per_day + 
        child_patients_per_hour * child_visit_cost * working_hours_per_day

/-- Theorem stating that Doctor Lindsay's daily income is $2200 -/
theorem doctor_lindsay_daily_income_is_2200 : 
  doctor_lindsay_daily_income = 2200 := by
  sorry

end NUMINAMATH_CALUDE_doctor_lindsay_daily_income_is_2200_l3389_338914


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3389_338984

/-- Given a cubic equation 5x^3 + 500x + 3005 = 0 with roots a, b, and c,
    prove that (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 -/
theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (5 * a^3 + 500 * a + 3005 = 0) →
  (5 * b^3 + 500 * b + 3005 = 0) →
  (5 * c^3 + 500 * c + 3005 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 := by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3389_338984


namespace NUMINAMATH_CALUDE_exhibition_solution_l3389_338932

/-- The number of paintings contributed by each grade in a school exhibition --/
structure PaintingExhibition where
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  fourth_grade : ℕ

/-- Properties of the painting exhibition --/
def ValidExhibition (e : PaintingExhibition) : Prop :=
  e.first_grade = 20 ∧
  e.second_grade = 45 ∧
  e.third_grade = e.first_grade + e.second_grade - 17 ∧
  e.fourth_grade = 2 * e.third_grade - 36

/-- Theorem stating the correct number of paintings for third and fourth grades --/
theorem exhibition_solution (e : PaintingExhibition) (h : ValidExhibition e) :
  e.third_grade = 48 ∧ e.fourth_grade = 60 := by
  sorry


end NUMINAMATH_CALUDE_exhibition_solution_l3389_338932


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3389_338925

theorem cube_volume_problem (a : ℝ) : 
  a > 0 →
  (a + 2) * (a + 2) * a - a^3 = 12 →
  a^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3389_338925


namespace NUMINAMATH_CALUDE_balloon_arrangements_l3389_338930

theorem balloon_arrangements : 
  let total_letters : ℕ := 7
  let repeated_letters : ℕ := 2
  let repetitions_per_letter : ℕ := 2
  (total_letters.factorial) / (repetitions_per_letter.factorial ^ repeated_letters) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l3389_338930


namespace NUMINAMATH_CALUDE_max_difference_two_digit_numbers_l3389_338955

theorem max_difference_two_digit_numbers :
  ∀ (A B : ℕ),
  (10 ≤ A ∧ A ≤ 99) →
  (10 ≤ B ∧ B ≤ 99) →
  (2 * A = 7 * B / 3) →
  (∀ (C D : ℕ), (10 ≤ C ∧ C ≤ 99) → (10 ≤ D ∧ D ≤ 99) → (2 * C = 7 * D / 3) → (C - D ≤ A - B)) →
  A - B = 56 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_two_digit_numbers_l3389_338955


namespace NUMINAMATH_CALUDE_isosceles_when_negative_one_is_root_roots_of_equilateral_triangle_l3389_338969

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadratic equation associated with the triangle -/
def quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 + 2 * t.b * x + (t.b - t.c)

theorem isosceles_when_negative_one_is_root (t : Triangle) :
  quadratic t (-1) = 0 → t.a = t.b :=
sorry

theorem roots_of_equilateral_triangle (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (quadratic t 0 = 0 ∧ quadratic t (-1) = 0) :=
sorry

end NUMINAMATH_CALUDE_isosceles_when_negative_one_is_root_roots_of_equilateral_triangle_l3389_338969


namespace NUMINAMATH_CALUDE_sharp_four_times_25_l3389_338949

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- State the theorem
theorem sharp_four_times_25 : sharp (sharp (sharp (sharp 25))) = 7.592 := by
  sorry

end NUMINAMATH_CALUDE_sharp_four_times_25_l3389_338949


namespace NUMINAMATH_CALUDE_parallelogram_area_18_10_l3389_338938

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 18 cm and height 10 cm is 180 cm² -/
theorem parallelogram_area_18_10 : parallelogram_area 18 10 = 180 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_18_10_l3389_338938


namespace NUMINAMATH_CALUDE_quadratic_equations_root_difference_l3389_338918

theorem quadratic_equations_root_difference (k : ℝ) : 
  (∀ x, x^2 + k*x + 6 = 0 → ∃ y, y^2 - k*y + 6 = 0 ∧ y = x + 5) →
  (∀ y, y^2 - k*y + 6 = 0 → ∃ x, x^2 + k*x + 6 = 0 ∧ y = x + 5) →
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_root_difference_l3389_338918


namespace NUMINAMATH_CALUDE_max_table_height_l3389_338981

/-- Given a triangle DEF with sides 25, 28, and 31, prove that the maximum possible height h'
    of a table constructed from this triangle is equal to 4√77 / 53. -/
theorem max_table_height (DE EF FD : ℝ) (h_DE : DE = 25) (h_EF : EF = 28) (h_FD : FD = 31) :
  let s := (DE + EF + FD) / 2
  let area := Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))
  let h_DE := 2 * area / DE
  let h_EF := 2 * area / EF
  ∃ h' : ℝ, h' = (h_DE * h_EF) / (h_DE + h_EF) ∧ h' = 4 * Real.sqrt 77 / 53 :=
by sorry


end NUMINAMATH_CALUDE_max_table_height_l3389_338981


namespace NUMINAMATH_CALUDE_descendant_divisibility_l3389_338979

theorem descendant_divisibility (N : ℕ) (h : N ≥ 10000 ∧ N < 100000) :
  N % 271 = 0 → (N * 10 + N / 10000 - (N / 10000) * 100000) % 271 = 0 := by
  sorry

end NUMINAMATH_CALUDE_descendant_divisibility_l3389_338979


namespace NUMINAMATH_CALUDE_expression_value_at_two_l3389_338966

theorem expression_value_at_two :
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x - 4)
  f 2 = 6 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_two_l3389_338966


namespace NUMINAMATH_CALUDE_modulus_of_z_l3389_338997

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 - Complex.I) :
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3389_338997


namespace NUMINAMATH_CALUDE_f_max_value_l3389_338939

def f (x : ℝ) := |x| - |x - 3|

theorem f_max_value :
  (∀ x, f x ≤ 3) ∧ (∃ x, f x = 3) := by sorry

end NUMINAMATH_CALUDE_f_max_value_l3389_338939


namespace NUMINAMATH_CALUDE_ten_factorial_minus_nine_factorial_l3389_338901

theorem ten_factorial_minus_nine_factorial : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_ten_factorial_minus_nine_factorial_l3389_338901


namespace NUMINAMATH_CALUDE_john_climbed_45_feet_l3389_338948

/-- Calculates the total distance climbed given the number of steps in three staircases and the height of each step -/
def total_distance_climbed (first_staircase : ℕ) (step_height : ℝ) : ℝ :=
  let second_staircase := 2 * first_staircase
  let third_staircase := second_staircase - 10
  let total_steps := first_staircase + second_staircase + third_staircase
  total_steps * step_height

/-- Theorem stating that John climbed 45 feet given the problem conditions -/
theorem john_climbed_45_feet :
  total_distance_climbed 20 0.5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_john_climbed_45_feet_l3389_338948


namespace NUMINAMATH_CALUDE_not_always_fifteen_different_l3389_338977

/-- Represents a student with a t-shirt color and a pants color -/
structure Student :=
  (tshirt : Fin 15)
  (pants : Fin 15)

/-- The theorem stating that it's not always possible to find 15 students
    with all different t-shirt and pants colors -/
theorem not_always_fifteen_different (n : Nat) (h : n = 30) :
  ∃ (students : Finset Student),
    students.card = n ∧
    ∀ (subset : Finset Student),
      subset ⊆ students →
      subset.card = 15 →
      ∃ (s1 s2 : Student),
        s1 ∈ subset ∧ s2 ∈ subset ∧ s1 ≠ s2 ∧
        (s1.tshirt = s2.tshirt ∨ s1.pants = s2.pants) :=
by sorry

end NUMINAMATH_CALUDE_not_always_fifteen_different_l3389_338977


namespace NUMINAMATH_CALUDE_library_book_purchase_ratio_l3389_338988

theorem library_book_purchase_ratio :
  ∀ (initial_books last_year_purchase current_total : ℕ),
  initial_books = 100 →
  last_year_purchase = 50 →
  current_total = 300 →
  ∃ (this_year_purchase : ℕ),
    this_year_purchase = 3 * last_year_purchase ∧
    current_total = initial_books + last_year_purchase + this_year_purchase :=
by
  sorry

end NUMINAMATH_CALUDE_library_book_purchase_ratio_l3389_338988


namespace NUMINAMATH_CALUDE_divisibility_property_l3389_338940

theorem divisibility_property (q : ℕ) (h_prime : Nat.Prime q) (h_odd : Odd q) :
  ∃ k : ℤ, (q - 1 : ℤ) ^ (q - 2) + 1 = k * q :=
sorry

end NUMINAMATH_CALUDE_divisibility_property_l3389_338940


namespace NUMINAMATH_CALUDE_pizza_delivery_solution_l3389_338982

/-- Represents the pizza delivery problem -/
def PizzaDelivery (total_pizzas : ℕ) (total_time : ℕ) (avg_time_per_stop : ℕ) : Prop :=
  ∃ (two_pizza_stops : ℕ),
    two_pizza_stops * 2 + (total_pizzas - two_pizza_stops * 2) = total_pizzas ∧
    (two_pizza_stops + (total_pizzas - two_pizza_stops * 2)) * avg_time_per_stop = total_time

/-- Theorem stating the solution to the pizza delivery problem -/
theorem pizza_delivery_solution :
  PizzaDelivery 12 40 4 → ∃ (two_pizza_stops : ℕ), two_pizza_stops = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_pizza_delivery_solution_l3389_338982


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_nine_count_l3389_338999

theorem four_digit_divisible_by_nine_count : 
  (Finset.filter (fun n => n % 9 = 0) (Finset.range 9000)).card = 1000 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_nine_count_l3389_338999


namespace NUMINAMATH_CALUDE_sunlovers_happy_days_l3389_338987

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2*D*(R^2 + 4) - 2*R*(D^2 + 4) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sunlovers_happy_days_l3389_338987


namespace NUMINAMATH_CALUDE_canyon_trail_length_l3389_338913

/-- Represents the hike on Canyon Trail -/
structure CanyonTrail where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the hike -/
def validHike (hike : CanyonTrail) : Prop :=
  hike.day1 + hike.day2 + hike.day3 = 36 ∧
  (hike.day2 + hike.day3 + hike.day4) / 3 = 14 ∧
  hike.day3 + hike.day4 + hike.day5 = 45 ∧
  hike.day1 + hike.day4 = 29

/-- The theorem stating the total length of the Canyon Trail -/
theorem canyon_trail_length (hike : CanyonTrail) (h : validHike hike) :
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 71 := by
  sorry

end NUMINAMATH_CALUDE_canyon_trail_length_l3389_338913


namespace NUMINAMATH_CALUDE_monic_polynomial_sum_l3389_338985

-- Define a monic polynomial of degree 4
def is_monic_degree_4 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

-- Define the conditions on p
def satisfies_conditions (p : ℝ → ℝ) : Prop :=
  p 1 = 17 ∧ p 2 = 34 ∧ p 3 = 51

-- Theorem statement
theorem monic_polynomial_sum (p : ℝ → ℝ) 
  (h1 : is_monic_degree_4 p) 
  (h2 : satisfies_conditions p) : 
  p 0 + p 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_monic_polynomial_sum_l3389_338985


namespace NUMINAMATH_CALUDE_total_cost_of_pens_and_pencils_l3389_338970

/-- The cost of buying multiple items given their individual prices -/
theorem total_cost_of_pens_and_pencils (x y : ℝ) : 
  5 * x + 3 * y = 5 * x + 3 * y := by sorry

end NUMINAMATH_CALUDE_total_cost_of_pens_and_pencils_l3389_338970


namespace NUMINAMATH_CALUDE_leadership_team_selection_l3389_338922

theorem leadership_team_selection (n : ℕ) (h : n = 20) :
  (n.choose 2) * ((n - 2).choose 1) = 3420 := by
  sorry

end NUMINAMATH_CALUDE_leadership_team_selection_l3389_338922


namespace NUMINAMATH_CALUDE_fraction_simplification_l3389_338973

theorem fraction_simplification (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3*x^2 - 2*x - 4) / ((x+2)*(x-3)) - (5+x) / ((x+2)*(x-3)) = 3*(x^2-x-3) / ((x+2)*(x-3)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3389_338973


namespace NUMINAMATH_CALUDE_range_of_a_l3389_338976

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 1| + |x - a| ≤ 2) ↔ a ∈ Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3389_338976


namespace NUMINAMATH_CALUDE_assembly_line_increased_rate_l3389_338965

/-- Represents the production rate of an assembly line -/
structure AssemblyLine where
  initial_rate : ℝ
  increased_rate : ℝ
  initial_order : ℝ
  second_order : ℝ
  average_output : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem assembly_line_increased_rate (a : AssemblyLine) 
  (h1 : a.initial_rate = 30)
  (h2 : a.initial_order = 60)
  (h3 : a.second_order = 60)
  (h4 : a.average_output = 40)
  (h5 : (a.initial_order + a.second_order) / 
        (a.initial_order / a.initial_rate + a.second_order / a.increased_rate) = a.average_output) :
  a.increased_rate = 60 := by
  sorry


end NUMINAMATH_CALUDE_assembly_line_increased_rate_l3389_338965


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3389_338928

theorem circle_area_ratio : 
  ∀ (r₁ r₂ : ℝ), r₁ > 0 → r₂ > 0 → r₂ = 3 * r₁ →
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3389_338928


namespace NUMINAMATH_CALUDE_correct_lunch_bill_l3389_338961

/-- The cost of Sara's lunch items and the total bill -/
def lunch_bill (hotdog_cost salad_cost : ℚ) : Prop :=
  hotdog_cost = 5.36 ∧ salad_cost = 5.10 ∧ hotdog_cost + salad_cost = 10.46

/-- Theorem stating that the total lunch bill is correct -/
theorem correct_lunch_bill :
  ∃ (hotdog_cost salad_cost : ℚ), lunch_bill hotdog_cost salad_cost :=
sorry

end NUMINAMATH_CALUDE_correct_lunch_bill_l3389_338961


namespace NUMINAMATH_CALUDE_smallest_added_number_l3389_338935

theorem smallest_added_number (n : ℤ) (x : ℕ) 
  (h1 : n % 25 = 4)
  (h2 : (n + x) % 5 = 4)
  (h3 : x > 0) :
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_added_number_l3389_338935


namespace NUMINAMATH_CALUDE_complex_inequality_l3389_338910

theorem complex_inequality (z₁ z₂ z₃ z₄ : ℂ) :
  ‖z₁ - z₃‖^2 + ‖z₂ - z₄‖^2 ≤ ‖z₁ - z₂‖^2 + ‖z₂ - z₃‖^2 + ‖z₃ - z₄‖^2 + ‖z₄ - z₁‖^2 ∧
  (‖z₁ - z₃‖^2 + ‖z₂ - z₄‖^2 = ‖z₁ - z₂‖^2 + ‖z₂ - z₃‖^2 + ‖z₃ - z₄‖^2 + ‖z₄ - z₁‖^2 ↔ z₁ + z₃ = z₂ + z₄) :=
by sorry

end NUMINAMATH_CALUDE_complex_inequality_l3389_338910


namespace NUMINAMATH_CALUDE_complex_modulus_range_l3389_338905

theorem complex_modulus_range (a : ℝ) (z : ℂ) (h1 : 0 < a) (h2 : a < 2) (h3 : z = a + Complex.I) :
  1 < Complex.abs z ∧ Complex.abs z < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l3389_338905


namespace NUMINAMATH_CALUDE_oliver_candy_boxes_l3389_338968

theorem oliver_candy_boxes : ∃ (initial : ℕ), initial + 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_oliver_candy_boxes_l3389_338968


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3389_338996

theorem complex_magnitude_problem (z : ℂ) : z = (2 - Complex.I) / (1 + 2 * Complex.I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3389_338996


namespace NUMINAMATH_CALUDE_integral_of_improper_rational_function_l3389_338956

noncomputable def F (x : ℝ) : ℝ :=
  x^3 / 3 + x^2 - x + 
  (1 / (4 * Real.sqrt 2)) * Real.log ((x^2 - Real.sqrt 2 * x + 1) / (x^2 + Real.sqrt 2 * x + 1)) + 
  (1 / (2 * Real.sqrt 2)) * (Real.arctan (Real.sqrt 2 * x + 1) + Real.arctan (Real.sqrt 2 * x - 1))

theorem integral_of_improper_rational_function (x : ℝ) :
  deriv F x = (x^6 + 2*x^5 - x^4 + x^2 + 2*x) / (x^4 + 1) := by sorry

end NUMINAMATH_CALUDE_integral_of_improper_rational_function_l3389_338956


namespace NUMINAMATH_CALUDE_cylinder_optimal_ratio_l3389_338946

/-- For a cylinder with fixed volume, the ratio of height to radius is 1 when surface area is minimized -/
theorem cylinder_optimal_ratio (V : ℝ) (h r : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  V = π * r^2 * h → -- Volume condition
  (∀ h' r' : ℝ, 0 < h' ∧ 0 < r' ∧ V = π * r'^2 * h' →
    2 * π * r^2 + 2 * π * r * h ≤ 2 * π * r'^2 + 2 * π * r' * h') → -- Surface area is minimized
  h / r = 1 := by sorry

end NUMINAMATH_CALUDE_cylinder_optimal_ratio_l3389_338946


namespace NUMINAMATH_CALUDE_value_of_x_l3389_338921

theorem value_of_x : (2011^3 - 2011^2) / 2011 = 2011 * 2010 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3389_338921


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3389_338944

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (1 / x + 1 / y) ≥ 1 / 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 20 ∧ 1 / x₀ + 1 / y₀ = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3389_338944


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_l3389_338960

/-- The function f is monotonic on the interval [1, 2] if and only if a is in the specified range -/
theorem monotonic_quadratic_function (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, Monotone (fun x => x^2 + (2*a + 1)*x + 1)) ↔ 
  a ∈ Set.Iic (-3/2) ∪ Set.Ioi (-5/2) :=
sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_l3389_338960


namespace NUMINAMATH_CALUDE_ladies_walking_distance_l3389_338919

theorem ladies_walking_distance (x y : ℝ) (h1 : x = 2 * y) (h2 : y = 4) :
  x + y = 12 := by sorry

end NUMINAMATH_CALUDE_ladies_walking_distance_l3389_338919


namespace NUMINAMATH_CALUDE_binomial_probability_equals_eight_twentyseven_l3389_338937

/-- A random variable following a binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- The probability mass function for a binomial distribution -/
def binomialPMF (dist : BinomialDistribution) (k : ℕ) : ℝ :=
  (dist.n.choose k) * (dist.p ^ k) * ((1 - dist.p) ^ (dist.n - k))

theorem binomial_probability_equals_eight_twentyseven :
  let ξ : BinomialDistribution := ⟨4, 1/3, by norm_num, by norm_num⟩
  binomialPMF ξ 2 = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_equals_eight_twentyseven_l3389_338937


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3389_338958

def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def in_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem z_in_first_quadrant (z₁ z₂ : ℂ) 
  (h₁ : complex_to_point z₁ = (2, 3))
  (h₂ : z₂ = -1 + 2*Complex.I) :
  in_first_quadrant (complex_to_point (z₁ - z₂)) := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3389_338958


namespace NUMINAMATH_CALUDE_number_equation_solution_l3389_338929

theorem number_equation_solution :
  ∃ x : ℝ, (3/4 * x + 3^2 = 1/5 * (x - 8 * x^(1/3))) ∧ x = -27 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3389_338929


namespace NUMINAMATH_CALUDE_trapezoid_triangle_area_l3389_338974

/-- A trapezoid with vertices A, B, C, and D -/
structure Trapezoid :=
  (A B C D : ℝ × ℝ)

/-- The area of a trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- The length of a line segment between two points -/
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem trapezoid_triangle_area (t : Trapezoid) :
  area t = 30 ∧ length t.C t.D = 3 * length t.A t.B →
  triangleArea t.A t.B t.C = 7.5 := by sorry

end NUMINAMATH_CALUDE_trapezoid_triangle_area_l3389_338974


namespace NUMINAMATH_CALUDE_largest_angle_right_triangle_l3389_338943

/-- A right triangle with acute angles in the ratio 8:1 has its largest angle measuring 90 degrees. -/
theorem largest_angle_right_triangle (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_acute_ratio : a / b = 8 ∨ b / a = 8) : max a (max b c) = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_right_triangle_l3389_338943


namespace NUMINAMATH_CALUDE_variance_of_letters_l3389_338927

def letters : List ℕ := [10, 6, 8, 5, 6]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let μ := mean l
  (l.map (fun x => ((x : ℚ) - μ)^2)).sum / l.length

theorem variance_of_letters :
  variance letters = 16/5 := by sorry

end NUMINAMATH_CALUDE_variance_of_letters_l3389_338927


namespace NUMINAMATH_CALUDE_article_cost_l3389_338957

/-- The cost of an article satisfying given profit conditions -/
theorem article_cost : ∃ (C : ℝ), 
  (C = 70) ∧ 
  (∃ (S : ℝ), S = 1.25 * C) ∧ 
  (∃ (S_new : ℝ), S_new = 0.8 * C + 0.3 * (0.8 * C) ∧ S_new = 1.25 * C - 14.70) :=
sorry

end NUMINAMATH_CALUDE_article_cost_l3389_338957


namespace NUMINAMATH_CALUDE_shaded_percentage_7x7_grid_l3389_338907

/-- The percentage of shaded squares in a 7x7 grid with 20 shaded squares -/
theorem shaded_percentage_7x7_grid (total_squares : Nat) (shaded_squares : Nat) :
  total_squares = 7 * 7 →
  shaded_squares = 20 →
  (shaded_squares : Real) / total_squares * 100 = 20 / 49 * 100 := by
  sorry

end NUMINAMATH_CALUDE_shaded_percentage_7x7_grid_l3389_338907


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3389_338971

/-- Given a boat that travels 10 km/hr downstream and 4 km/hr upstream, 
    its speed in still water is 7 km/hr. -/
theorem boat_speed_in_still_water 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h_downstream : downstream_speed = 10) 
  (h_upstream : upstream_speed = 4) : 
  (downstream_speed + upstream_speed) / 2 = 7 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3389_338971


namespace NUMINAMATH_CALUDE_expense_settlement_proof_l3389_338912

def expense_settlement (alice_paid bob_paid charlie_paid : ℚ) : Prop :=
  let total_paid := alice_paid + bob_paid + charlie_paid
  let share_per_person := total_paid / 3
  let alice_owes := share_per_person - alice_paid
  let bob_owes := share_per_person - bob_paid
  let charlie_owed := charlie_paid - share_per_person
  ∃ a b : ℚ, 
    a = alice_owes ∧ 
    b = bob_owes ∧ 
    a - b = 30

theorem expense_settlement_proof :
  expense_settlement 130 160 210 := by
  sorry

end NUMINAMATH_CALUDE_expense_settlement_proof_l3389_338912


namespace NUMINAMATH_CALUDE_cube_surface_area_l3389_338964

/-- Given a cube where the sum of all edge lengths is 180 cm, 
    prove that its surface area is 1350 cm². -/
theorem cube_surface_area (edge_sum : ℝ) (h_edge_sum : edge_sum = 180) :
  let edge_length := edge_sum / 12
  6 * edge_length^2 = 1350 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3389_338964


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3389_338992

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) / a -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → e = Real.sqrt 7 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3389_338992


namespace NUMINAMATH_CALUDE_max_positive_integer_solution_of_inequality_system_l3389_338986

theorem max_positive_integer_solution_of_inequality_system :
  ∃ (x : ℝ), (3 * x - 1 > x + 1) ∧ ((4 * x - 5) / 3 ≤ x) ∧
  (∀ (y : ℤ), (3 * y - 1 > y + 1) ∧ ((4 * y - 5) / 3 ≤ y) → y ≤ 5) ∧
  (3 * 5 - 1 > 5 + 1) ∧ ((4 * 5 - 5) / 3 ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_max_positive_integer_solution_of_inequality_system_l3389_338986


namespace NUMINAMATH_CALUDE_largest_number_divisible_by_sum_of_digits_eight_eight_eight_satisfies_property_l3389_338967

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the property we're looking for
def hasSumOfDigitsDivisibility (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

-- State the theorem
theorem largest_number_divisible_by_sum_of_digits :
  ∀ n : ℕ, n < 900 → hasSumOfDigitsDivisibility n → n ≤ 888 :=
by
  sorry

-- Prove that 888 satisfies the property
theorem eight_eight_eight_satisfies_property :
  hasSumOfDigitsDivisibility 888 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_divisible_by_sum_of_digits_eight_eight_eight_satisfies_property_l3389_338967


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3389_338993

theorem arithmetic_evaluation : 4 + 10 / 2 - 2 * 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3389_338993


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3389_338911

/-- The quadratic equation with coefficient m, (1/3), and 1 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 + (1/3) * x + 1 = 0

theorem quadratic_root_relation (m₁ m₂ x₁ x₂ x₃ x₄ : ℝ) :
  quadratic_equation m₁ x₁ →
  quadratic_equation m₁ x₂ →
  quadratic_equation m₂ x₃ →
  quadratic_equation m₂ x₄ →
  x₁ < x₃ →
  x₃ < x₄ →
  x₄ < x₂ →
  x₂ < 0 →
  m₂ > m₁ ∧ m₁ > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3389_338911


namespace NUMINAMATH_CALUDE_triangle_property_l3389_338947

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the properties of the triangle
def has_equal_roots (t : Triangle) : Prop :=
  ∃ x : ℝ, (t.b + t.c) * x^2 - 2 * t.a * x + t.c - t.b = 0

def angle_condition (t : Triangle) : Prop :=
  Real.sin t.B * Real.cos t.A - Real.cos t.B * Real.sin t.A = 0

def is_isosceles_right (t : Triangle) : Prop :=
  t.a = t.b ∧ t.a^2 + t.b^2 = t.c^2

-- State the theorem
theorem triangle_property (t : Triangle) :
  has_equal_roots t → angle_condition t → is_isosceles_right t :=
sorry

end NUMINAMATH_CALUDE_triangle_property_l3389_338947


namespace NUMINAMATH_CALUDE_meals_left_theorem_l3389_338978

/-- Given the initial number of meals, additional meals, and meals given away,
    calculate the number of meals left to be distributed. -/
def meals_left_to_distribute (initial_meals additional_meals meals_given_away : ℕ) : ℕ :=
  initial_meals + additional_meals - meals_given_away

/-- Theorem stating that for the given problem, 78 meals are left to be distributed. -/
theorem meals_left_theorem (initial_meals additional_meals meals_given_away : ℕ) 
  (h1 : initial_meals = 113)
  (h2 : additional_meals = 50)
  (h3 : meals_given_away = 85) :
  meals_left_to_distribute initial_meals additional_meals meals_given_away = 78 := by
  sorry

end NUMINAMATH_CALUDE_meals_left_theorem_l3389_338978


namespace NUMINAMATH_CALUDE_negative_expressions_l3389_338933

theorem negative_expressions (x : ℝ) (h : x < 0) : x^3 < 0 ∧ -x^4 < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_expressions_l3389_338933


namespace NUMINAMATH_CALUDE_min_value_cos_sum_l3389_338998

theorem min_value_cos_sum (x : ℝ) : 
  ∃ (m : ℝ), m = -Real.sqrt 2 ∧ ∀ y : ℝ, 
    Real.cos (3*y + π/6) + Real.cos (3*y - π/3) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_cos_sum_l3389_338998


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l3389_338959

/-- A function that represents the relationship between x and y -/
def f (x : ℝ) : ℝ := -2 * x + 6

/-- The proposition that f satisfies the given conditions -/
theorem f_satisfies_conditions :
  (∃ k : ℝ, ∀ x : ℝ, f x = k * (x - 3)) ∧  -- y is directly proportional to x-3
  (f 5 = -4)                               -- When x = 5, y = -4
  := by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l3389_338959


namespace NUMINAMATH_CALUDE_percentage_of_muslim_boys_l3389_338983

/-- Given a school with 850 boys, where 28% are Hindus, 10% are Sikhs, 
    and 136 boys belong to other communities, prove that 46% of the boys are Muslims. -/
theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percent : ℚ) (sikh_percent : ℚ) (other_boys : ℕ) : 
  total_boys = 850 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  other_boys = 136 →
  (↑(total_boys - (total_boys * hindu_percent).floor - (total_boys * sikh_percent).floor - other_boys) / total_boys : ℚ) = 46 / 100 :=
by
  sorry

#eval (850 : ℕ) - (850 * (28 / 100 : ℚ)).floor - (850 * (10 / 100 : ℚ)).floor - 136

end NUMINAMATH_CALUDE_percentage_of_muslim_boys_l3389_338983


namespace NUMINAMATH_CALUDE_julia_running_time_difference_l3389_338915

/-- Julia's running times with different shoes -/
theorem julia_running_time_difference (x : ℝ) : 
  let old_pace : ℝ := 10  -- minutes per mile in old shoes
  let new_pace : ℝ := 13  -- minutes per mile in new shoes
  let miles_for_known_difference : ℝ := 5
  let known_time_difference : ℝ := 15  -- minutes difference for 5 miles
  -- Prove that the time difference for x miles is 3x minutes
  (new_pace - old_pace) * x = 3 * x ∧
  -- Also prove that this is consistent with the given information for 5 miles
  (new_pace - old_pace) * miles_for_known_difference = known_time_difference
  := by sorry

end NUMINAMATH_CALUDE_julia_running_time_difference_l3389_338915


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3389_338972

theorem rectangle_perimeter (a b : ℕ) : 
  a ≠ b →  -- non-square condition
  a * b = 2 * (2 * a + 2 * b) - 8 →  -- area condition
  2 * (a + b) = 36 :=  -- perimeter conclusion
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3389_338972


namespace NUMINAMATH_CALUDE_cubic_odd_and_increasing_l3389_338934

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cubic_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_odd_and_increasing_l3389_338934


namespace NUMINAMATH_CALUDE_campaign_fundraising_l3389_338942

theorem campaign_fundraising (max_donation : ℕ) (max_donors : ℕ) (half_donors : ℕ) (percentage : ℚ) : 
  max_donation = 1200 →
  max_donors = 500 →
  half_donors = 3 * max_donors →
  percentage = 40 / 100 →
  (max_donation * max_donors + (max_donation / 2) * half_donors) / percentage = 3750000 := by
sorry

end NUMINAMATH_CALUDE_campaign_fundraising_l3389_338942


namespace NUMINAMATH_CALUDE_A_equality_l3389_338952

/-- The number of integer tuples (x₁, x₂, ..., xₖ) satisfying the given conditions -/
def A (n k r : ℕ+) : ℕ := sorry

/-- The theorem stating the equality of A for different arguments -/
theorem A_equality (s t : ℕ+) (hs : s ≥ 2) (ht : t ≥ 2) :
  A (s * t) s t = A (s * (t - 1)) s t ∧ A (s * t) s t = A ((s - 1) * t) s t :=
sorry

end NUMINAMATH_CALUDE_A_equality_l3389_338952


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l3389_338953

/-- The rate of a man rowing in still water, given his speeds with and against a stream. -/
theorem mans_rate_in_still_water
  (speed_with_stream : ℝ)
  (speed_against_stream : ℝ)
  (h_with : speed_with_stream = 20)
  (h_against : speed_against_stream = 4) :
  (speed_with_stream + speed_against_stream) / 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l3389_338953


namespace NUMINAMATH_CALUDE_not_equivalent_squared_and_equal_l3389_338903

variable {X : Type*}
variable (x : X)
variable (A B : X → ℝ)

theorem not_equivalent_squared_and_equal :
  ¬(∀ x, A x ^ 2 = B x ^ 2 ↔ A x = B x) :=
sorry

end NUMINAMATH_CALUDE_not_equivalent_squared_and_equal_l3389_338903


namespace NUMINAMATH_CALUDE_range_of_a_l3389_338926

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x | |x + a| < 1}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (B a ⊂ A) ∧ (B a ≠ A) → 0 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3389_338926


namespace NUMINAMATH_CALUDE_shopkeeper_loss_l3389_338924

/-- Represents the shopkeeper's fruit inventory and sales data --/
structure FruitShop where
  total_fruit : ℝ
  apples : ℝ
  oranges : ℝ
  bananas : ℝ
  apple_price : ℝ
  orange_price : ℝ
  banana_price : ℝ
  apple_increase : ℝ
  orange_increase : ℝ
  banana_increase : ℝ
  overhead : ℝ
  apple_morning_sales : ℝ
  orange_morning_sales : ℝ
  banana_morning_sales : ℝ

/-- Calculates the profit of the fruit shop --/
def calculate_profit (shop : FruitShop) : ℝ :=
  let morning_revenue := 
    shop.apple_price * shop.apples * shop.apple_morning_sales +
    shop.orange_price * shop.oranges * shop.orange_morning_sales +
    shop.banana_price * shop.bananas * shop.banana_morning_sales
  let afternoon_revenue := 
    shop.apple_price * (1 + shop.apple_increase) * shop.apples * (1 - shop.apple_morning_sales) +
    shop.orange_price * (1 + shop.orange_increase) * shop.oranges * (1 - shop.orange_morning_sales) +
    shop.banana_price * (1 + shop.banana_increase) * shop.bananas * (1 - shop.banana_morning_sales)
  let total_revenue := morning_revenue + afternoon_revenue
  let total_cost := 
    shop.apple_price * shop.apples +
    shop.orange_price * shop.oranges +
    shop.banana_price * shop.bananas +
    shop.overhead
  total_revenue - total_cost

/-- Theorem stating that the shopkeeper incurs a loss of $178.88 --/
theorem shopkeeper_loss (shop : FruitShop) 
  (h1 : shop.total_fruit = 700)
  (h2 : shop.apples = 280)
  (h3 : shop.oranges = 210)
  (h4 : shop.bananas = shop.total_fruit - shop.apples - shop.oranges)
  (h5 : shop.apple_price = 5)
  (h6 : shop.orange_price = 4)
  (h7 : shop.banana_price = 2)
  (h8 : shop.apple_increase = 0.12)
  (h9 : shop.orange_increase = 0.15)
  (h10 : shop.banana_increase = 0.08)
  (h11 : shop.overhead = 320)
  (h12 : shop.apple_morning_sales = 0.5)
  (h13 : shop.orange_morning_sales = 0.6)
  (h14 : shop.banana_morning_sales = 0.8) :
  calculate_profit shop = -178.88 := by
  sorry


end NUMINAMATH_CALUDE_shopkeeper_loss_l3389_338924


namespace NUMINAMATH_CALUDE_miles_and_davis_amount_l3389_338975

-- Define the conversion rate from tablespoons of kernels to cups of popcorn
def kernels_to_popcorn (tablespoons : ℚ) : ℚ := 2 * tablespoons

-- Define the amounts of popcorn wanted by Joanie, Mitchell, and Cliff
def joanie_amount : ℚ := 3
def mitchell_amount : ℚ := 4
def cliff_amount : ℚ := 3

-- Define the total amount of kernels needed
def total_kernels : ℚ := 8

-- Theorem to prove
theorem miles_and_davis_amount :
  kernels_to_popcorn total_kernels - (joanie_amount + mitchell_amount + cliff_amount) = 6 :=
by sorry

end NUMINAMATH_CALUDE_miles_and_davis_amount_l3389_338975


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l3389_338931

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (3 * x^2 + 2 - Real.sqrt (9 * x^4 + 4)) / x ≤ 12 / (5 + 3 * Real.sqrt 3) :=
sorry

theorem max_value_achievable :
  ∃ x : ℝ, x > 0 ∧ (3 * x^2 + 2 - Real.sqrt (9 * x^4 + 4)) / x = 12 / (5 + 3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l3389_338931


namespace NUMINAMATH_CALUDE_jony_walking_speed_l3389_338950

/-- Calculates the walking speed given distance and time -/
def walking_speed (distance : ℕ) (time : ℕ) : ℕ :=
  distance / time

/-- The number of blocks Jony walks -/
def blocks_walked : ℕ := (90 - 10) + (90 - 70)

/-- The length of each block in meters -/
def block_length : ℕ := 40

/-- The total distance Jony walks in meters -/
def total_distance : ℕ := blocks_walked * block_length

/-- The total time Jony spends walking in minutes -/
def total_time : ℕ := 40

theorem jony_walking_speed :
  walking_speed total_distance total_time = 100 := by
  sorry

end NUMINAMATH_CALUDE_jony_walking_speed_l3389_338950


namespace NUMINAMATH_CALUDE_solution_in_interval_l3389_338904

-- Define the function f(x) = x³ - x - 3
def f (x : ℝ) := x^3 - x - 3

-- State the theorem
theorem solution_in_interval :
  ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l3389_338904
