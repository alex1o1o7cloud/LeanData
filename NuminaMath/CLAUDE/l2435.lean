import Mathlib

namespace pillar_length_calculation_l2435_243595

/-- Given a formula for L and specific values for T, H, and K, prove that L equals 100. -/
theorem pillar_length_calculation (T H K L : ℝ) : 
  T = 2 * Real.sqrt 5 →
  H = 10 →
  K = 2 →
  L = (50 * T^4) / (H^2 * K) →
  L = 100 := by
sorry

end pillar_length_calculation_l2435_243595


namespace stewart_farm_sheep_count_l2435_243572

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
  sheep / horses = 3 / 7 →
  horses * 230 = 12880 →
  sheep = 24 :=
by
  sorry

end stewart_farm_sheep_count_l2435_243572


namespace total_rooms_in_hotel_l2435_243513

/-- Represents a wing of the hotel -/
structure Wing where
  floors : ℕ
  hallsPerFloor : ℕ
  singleRoomsPerHall : ℕ
  doubleRoomsPerHall : ℕ
  suitesPerHall : ℕ

/-- Calculates the total number of rooms in a wing -/
def totalRoomsInWing (w : Wing) : ℕ :=
  w.floors * w.hallsPerFloor * (w.singleRoomsPerHall + w.doubleRoomsPerHall + w.suitesPerHall)

/-- The first wing of the hotel -/
def wing1 : Wing :=
  { floors := 9
    hallsPerFloor := 6
    singleRoomsPerHall := 20
    doubleRoomsPerHall := 8
    suitesPerHall := 4 }

/-- The second wing of the hotel -/
def wing2 : Wing :=
  { floors := 7
    hallsPerFloor := 9
    singleRoomsPerHall := 25
    doubleRoomsPerHall := 10
    suitesPerHall := 5 }

/-- The third wing of the hotel -/
def wing3 : Wing :=
  { floors := 12
    hallsPerFloor := 4
    singleRoomsPerHall := 30
    doubleRoomsPerHall := 15
    suitesPerHall := 5 }

/-- Theorem stating the total number of rooms in the hotel -/
theorem total_rooms_in_hotel :
  totalRoomsInWing wing1 + totalRoomsInWing wing2 + totalRoomsInWing wing3 = 6648 := by
  sorry

end total_rooms_in_hotel_l2435_243513


namespace division_problem_l2435_243517

theorem division_problem : 
  (-1/42) / (1/6 - 3/14 + 2/3 - 2/7) = -1/14 := by
  sorry

end division_problem_l2435_243517


namespace cubic_inequality_implies_a_range_l2435_243568

theorem cubic_inequality_implies_a_range :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → a * x^3 - x^2 + 4*x + 3 ≥ 0) →
  a ∈ Set.Icc (-6) (-2) := by
  sorry

end cubic_inequality_implies_a_range_l2435_243568


namespace coin_pile_theorem_l2435_243525

/-- Represents the state of the three piles of coins -/
structure CoinState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Defines the allowed operations on the coin piles -/
inductive Operation
  | split_even : ℕ → Operation
  | remove_odd : ℕ → Operation

/-- Applies an operation to a CoinState -/
def apply_operation (state : CoinState) (op : Operation) : CoinState :=
  sorry

/-- Checks if a state has a pile with at least 2017^2017 coins -/
def has_large_pile (state : CoinState) : Prop :=
  sorry

/-- Theorem stating that any initial state except (2015, 2015, 2015) can reach a large pile -/
theorem coin_pile_theorem (initial : CoinState)
    (h1 : initial.a ≥ 2015)
    (h2 : initial.b ≥ 2015)
    (h3 : initial.c ≥ 2015)
    (h4 : ¬(initial.a = 2015 ∧ initial.b = 2015 ∧ initial.c = 2015)) :
    ∃ (ops : List Operation), has_large_pile (ops.foldl apply_operation initial) :=
  sorry

end coin_pile_theorem_l2435_243525


namespace symmetric_even_function_implies_odd_l2435_243585

/-- A function satisfying certain symmetry and evenness properties -/
def SymmetricEvenFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x - 1) + f (1 - x) = 0) ∧ 
  (∀ x, f (x + 1) = f (-x + 1)) ∧
  (f (-3/2) = 1)

/-- The main theorem stating that f(x-2) is an odd function -/
theorem symmetric_even_function_implies_odd (f : ℝ → ℝ) 
  (h : SymmetricEvenFunction f) : 
  ∀ x, f (-(x - 2)) = -f (x - 2) := by
sorry

end symmetric_even_function_implies_odd_l2435_243585


namespace range_of_b_l2435_243596

theorem range_of_b (a b : ℝ) (h1 : 0 ≤ a + b ∧ a + b < 1) (h2 : 2 ≤ a - b ∧ a - b < 3) :
  -3/2 < b ∧ b < -1/2 := by
  sorry

end range_of_b_l2435_243596


namespace mean_study_days_is_4_05_l2435_243514

/-- Represents the study data for Ms. Rossi's class -/
structure StudyData where
  oneDay : Nat
  twoDays : Nat
  fourDays : Nat
  fiveDays : Nat
  sixDays : Nat

/-- Calculates the mean number of study days for the given data -/
def calculateMean (data : StudyData) : Float :=
  let totalDays := data.oneDay * 1 + data.twoDays * 2 + data.fourDays * 4 + data.fiveDays * 5 + data.sixDays * 6
  let totalStudents := data.oneDay + data.twoDays + data.fourDays + data.fiveDays + data.sixDays
  (totalDays.toFloat) / (totalStudents.toFloat)

/-- Theorem stating that the mean number of study days for Ms. Rossi's class is 4.05 -/
theorem mean_study_days_is_4_05 (data : StudyData) 
  (h1 : data.oneDay = 2)
  (h2 : data.twoDays = 4)
  (h3 : data.fourDays = 5)
  (h4 : data.fiveDays = 7)
  (h5 : data.sixDays = 4) :
  calculateMean data = 4.05 := by
  sorry

end mean_study_days_is_4_05_l2435_243514


namespace cara_seating_arrangements_l2435_243543

/-- The number of people at the table -/
def total_people : ℕ := 8

/-- The number of people Cara can choose from for her other neighbor -/
def available_neighbors : ℕ := total_people - 2

/-- The number of different possible pairs of people Cara could be sitting between -/
def seating_arrangements : ℕ := available_neighbors

theorem cara_seating_arrangements :
  seating_arrangements = 6 :=
sorry

end cara_seating_arrangements_l2435_243543


namespace trivia_game_base_points_l2435_243534

/-- Calculates the base points per round for a trivia game -/
theorem trivia_game_base_points
  (total_rounds : ℕ)
  (total_score : ℕ)
  (bonus_points : ℕ)
  (penalty_points : ℕ)
  (h1 : total_rounds = 5)
  (h2 : total_score = 370)
  (h3 : bonus_points = 50)
  (h4 : penalty_points = 30) :
  (total_score + bonus_points - penalty_points) / total_rounds = 78 := by
  sorry

end trivia_game_base_points_l2435_243534


namespace expected_hits_value_l2435_243548

/-- The probability of hitting the target -/
def hit_probability : ℝ := 0.97

/-- The total number of shots -/
def total_shots : ℕ := 1000

/-- The expected number of hits -/
def expected_hits : ℝ := hit_probability * total_shots

theorem expected_hits_value : expected_hits = 970 := by
  sorry

end expected_hits_value_l2435_243548


namespace distance_to_leg_intersection_l2435_243518

/-- An isosceles trapezoid with specific diagonal properties -/
structure IsoscelesTrapezoid where
  /-- The length of the longer segment of each diagonal -/
  long_segment : ℝ
  /-- The length of the shorter segment of each diagonal -/
  short_segment : ℝ
  /-- The angle between the diagonals formed by the legs -/
  diagonal_angle : ℝ
  /-- Condition: The longer segment is 7 -/
  long_is_7 : long_segment = 7
  /-- Condition: The shorter segment is 3 -/
  short_is_3 : short_segment = 3
  /-- Condition: The angle between diagonals is 60° -/
  angle_is_60 : diagonal_angle = 60

/-- The theorem stating the distance from diagonal intersection to leg intersection -/
theorem distance_to_leg_intersection (t : IsoscelesTrapezoid) :
  (t.long_segment / t.short_segment) * t.short_segment = 21 / 4 := by
  sorry

end distance_to_leg_intersection_l2435_243518


namespace cousin_age_l2435_243561

/-- Given the ages of Rick and his brothers, prove the age of their cousin -/
theorem cousin_age (rick_age : ℕ) (oldest_brother_age : ℕ) (middle_brother_age : ℕ) 
  (smallest_brother_age : ℕ) (youngest_brother_age : ℕ) (cousin_age : ℕ) 
  (h1 : rick_age = 15)
  (h2 : oldest_brother_age = 2 * rick_age)
  (h3 : middle_brother_age = oldest_brother_age / 3)
  (h4 : smallest_brother_age = middle_brother_age / 2)
  (h5 : youngest_brother_age = smallest_brother_age - 2)
  (h6 : cousin_age = 5 * youngest_brother_age) :
  cousin_age = 15 := by
  sorry

#check cousin_age

end cousin_age_l2435_243561


namespace matrix_vector_product_plus_vector_l2435_243519

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; -5, 6]
def v : Matrix (Fin 2) (Fin 1) ℝ := !![5; -2]
def w : Matrix (Fin 2) (Fin 1) ℝ := !![1; -1]

theorem matrix_vector_product_plus_vector :
  A * v + w = !![25; -38] := by sorry

end matrix_vector_product_plus_vector_l2435_243519


namespace partition_condition_l2435_243530

theorem partition_condition (α β : ℕ+) : 
  (∃ (A B : Set ℕ+), 
    (A ∪ B = Set.univ) ∧ 
    (A ∩ B = ∅) ∧ 
    ({α * a | a ∈ A} = {β * b | b ∈ B})) ↔ 
  (α ∣ β ∧ α ≠ β) ∨ (β ∣ α ∧ α ≠ β) :=
sorry

end partition_condition_l2435_243530


namespace arithmetic_sequence_properties_l2435_243582

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a1 : a 1 = -3)
  (h_condition : 11 * a 5 = 5 * a 8 - 13) :
  ∃ (d : ℚ) (S : ℕ → ℚ),
    (d = 31 / 9) ∧
    (∀ n : ℕ, S n = n * (2 * a 1 + (n - 1) * d) / 2) ∧
    (∀ n : ℕ, S n ≥ -2401 / 840) ∧
    (S 1 = -2401 / 840) :=
by sorry

end arithmetic_sequence_properties_l2435_243582


namespace unique_solution_when_k_zero_no_unique_solution_when_k_nonzero_k_zero_only_unique_solution_l2435_243588

/-- The equation has exactly one solution when k = 0 -/
theorem unique_solution_when_k_zero : ∃! x : ℝ, (x + 3) / (0 * x - 2) = x :=
sorry

/-- For any k ≠ 0, the equation has either no solution or more than one solution -/
theorem no_unique_solution_when_k_nonzero (k : ℝ) (hk : k ≠ 0) :
  ¬(∃! x : ℝ, (x + 3) / (k * x - 2) = x) :=
sorry

/-- k = 0 is the only value for which the equation has exactly one solution -/
theorem k_zero_only_unique_solution :
  ∀ k : ℝ, (∃! x : ℝ, (x + 3) / (k * x - 2) = x) ↔ k = 0 :=
sorry

end unique_solution_when_k_zero_no_unique_solution_when_k_nonzero_k_zero_only_unique_solution_l2435_243588


namespace floor_ceiling_sum_seven_l2435_243556

theorem floor_ceiling_sum_seven (x : ℝ) :
  (Int.floor x + Int.ceil x = 7) ↔ (3 < x ∧ x < 4) ∨ x = 3.5 := by
  sorry

end floor_ceiling_sum_seven_l2435_243556


namespace watertown_marching_band_max_size_l2435_243505

theorem watertown_marching_band_max_size :
  ∀ n : ℕ,
  (25 * n < 1200) →
  (25 * n % 29 = 6) →
  (∀ m : ℕ, (25 * m < 1200) → (25 * m % 29 = 6) → m ≤ n) →
  25 * n = 1050 :=
by sorry

end watertown_marching_band_max_size_l2435_243505


namespace truncated_pyramid_volume_l2435_243515

/-- A regular truncated quadrilateral pyramid -/
structure TruncatedPyramid where
  upper_base : ℝ
  lower_base : ℝ

/-- The volume of a truncated pyramid -/
def volume (p : TruncatedPyramid) : ℝ := sorry

/-- A plane that divides the pyramid into two equal parts -/
structure DividingPlane where
  perpendicular_to_diagonal : Bool
  passes_through_upper_edge : Bool

theorem truncated_pyramid_volume
  (p : TruncatedPyramid)
  (d : DividingPlane)
  (h1 : p.upper_base = 1)
  (h2 : p.lower_base = 7)
  (h3 : d.perpendicular_to_diagonal = true)
  (h4 : d.passes_through_upper_edge = true)
  (h5 : ∃ (v : ℝ), volume { upper_base := p.upper_base, lower_base := v } = volume { upper_base := v, lower_base := p.lower_base }) :
  volume p = 38 / Real.sqrt 5 := by sorry

end truncated_pyramid_volume_l2435_243515


namespace cylinder_volume_equality_l2435_243531

/-- Given two identical cylinders with initial radius 5 inches and height 4 inches,
    prove that when the radius of one cylinder is increased by x inches and
    the height of the second cylinder is increased by 2 inches, resulting in
    equal volumes, x = (5(√6 - 2)) / 2. -/
theorem cylinder_volume_equality (x : ℝ) : 
  (π * (5 + x)^2 * 4 = π * 5^2 * (4 + 2)) → x = (5 * (Real.sqrt 6 - 2)) / 2 := by
sorry

end cylinder_volume_equality_l2435_243531


namespace pure_imaginary_condition_l2435_243526

theorem pure_imaginary_condition (k : ℝ) : 
  (2 * k^2 - 3 * k - 2 : ℂ) + (k^2 - 2 * k : ℂ) * Complex.I = Complex.I * (k^2 - 2 * k : ℂ) ↔ k = -1/2 := by
  sorry

end pure_imaginary_condition_l2435_243526


namespace percentage_problem_l2435_243504

theorem percentage_problem (p : ℝ) : p * 50 = 0.15 → p = 0.003 := by
  sorry

end percentage_problem_l2435_243504


namespace triangle_properties_l2435_243584

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a + b = 13 →
  c = 7 →
  4 * (Real.sin ((A + B) / 2))^2 - Real.cos (2 * C) = 7/2 →
  C = π/3 ∧ 
  π * (2 * (1/2 * a * b * Real.sin C) / (a + b + c))^2 = 3*π :=
by sorry

end triangle_properties_l2435_243584


namespace expression_value_at_1580_l2435_243589

theorem expression_value_at_1580 : 
  let a : ℝ := 1580
  let expr := 2*a - (((2*a - 3)/(a + 1)) - ((a + 1)/(2 - 2*a)) - ((a^2 + 3)/(2*a^(2-2)))) * ((a^3 + 1)/(a^2 - a)) + 2/a
  expr = 2 := by sorry

end expression_value_at_1580_l2435_243589


namespace binary_to_decimal_octal_conversion_l2435_243501

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_to_decimal_octal_conversion :
  (binary_to_decimal binary_101101 = 45) ∧
  (decimal_to_octal 45 = [5, 5]) := by
sorry

end binary_to_decimal_octal_conversion_l2435_243501


namespace triangle_properties_l2435_243527

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating properties of a specific triangle -/
theorem triangle_properties (t : Triangle) (m : ℝ) : 
  (Real.sqrt 2 * Real.sin t.A = Real.sqrt (3 * Real.cos t.A)) →
  (t.a^2 - t.c^2 = t.b^2 - m * t.b * t.c) →
  (t.a = Real.sqrt 3) →
  (m = 1 ∧ 
   (∀ S : ℝ, S ≤ (3 * Real.sqrt 3) / 4 ∨ 
    ¬(∃ t' : Triangle, S = (t'.b * t'.c * Real.sin t'.A) / 2))) := by
  sorry

end triangle_properties_l2435_243527


namespace tangent_slope_angle_at_point_one_three_l2435_243577

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_angle_at_point_one_three :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let k : ℝ := f' x₀
  Real.arctan k = π/4 :=
by sorry

end tangent_slope_angle_at_point_one_three_l2435_243577


namespace multiplicative_inverse_modulo_l2435_243516

def A : Nat := 111111
def B : Nat := 142857
def M : Nat := 1000000

theorem multiplicative_inverse_modulo :
  (63 * (A * B)) % M = 1 := by sorry

end multiplicative_inverse_modulo_l2435_243516


namespace sequence_problem_l2435_243528

/-- Given a sequence {a_n} and a geometric sequence {b_n}, prove a_10 = 64 -/
theorem sequence_problem (a b : ℕ → ℚ) : 
  a 1 = 1/8 →                           -- First term of sequence {a_n}
  b 5 = 2 →                             -- b_5 = 2 in geometric sequence {b_n}
  (∀ n, b n = a (n+1) / a n) →          -- Relation between a_n and b_n
  (∃ q, ∀ n, b n = 2 * q^(n-5)) →       -- b_n is a geometric sequence
  a 10 = 64 := by
sorry

end sequence_problem_l2435_243528


namespace diophantine_equation_solutions_l2435_243573

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 9*x*y - x^2 - 8*y^2 = 2005 ↔ 
    (x = 63 ∧ y = 58) ∨ (x = -63 ∧ y = -58) ∨ 
    (x = 459 ∧ y = 58) ∨ (x = -459 ∧ y = -58) := by
  sorry

end diophantine_equation_solutions_l2435_243573


namespace p_only_root_zero_l2435_243594

/-- Recursive definition of polynomial p_n(x) -/
def p : ℕ → ℝ → ℝ
| 0, x => 0
| 1, x => x
| (n+2), x => x * p (n+1) x + (1 - x) * p n x

/-- Theorem stating that 0 is the only real root of p_n(x) for n ≥ 1 -/
theorem p_only_root_zero (n : ℕ) (h : n ≥ 1) :
  ∀ x : ℝ, p n x = 0 ↔ x = 0 := by
  sorry

#check p_only_root_zero

end p_only_root_zero_l2435_243594


namespace sum_of_mod2_and_mod3_l2435_243502

theorem sum_of_mod2_and_mod3 (a b : ℤ) : 
  a % 4 = 2 → b % 4 = 3 → (a + b) % 4 = 1 := by
  sorry

end sum_of_mod2_and_mod3_l2435_243502


namespace intersection_of_A_and_B_l2435_243546

def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 - 2}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1}

theorem intersection_of_A_and_B : A ∩ B = {(1, 1)} := by
  sorry

end intersection_of_A_and_B_l2435_243546


namespace fraction_problem_l2435_243563

theorem fraction_problem :
  ∃ (x : ℝ) (a b : ℕ),
    x > 0 ∧
    x^2 = 25 ∧
    2*x = (a / b : ℝ)*x + 9 →
    a = 1 ∧ b = 5 := by
  sorry

end fraction_problem_l2435_243563


namespace student_arrangement_count_l2435_243523

def num_male_students : ℕ := 3
def num_female_students : ℕ := 3
def total_students : ℕ := num_male_students + num_female_students

def adjacent_female_students : ℕ := 2

def num_arrangements : ℕ := 432

theorem student_arrangement_count :
  (num_male_students = 3) →
  (num_female_students = 3) →
  (total_students = num_male_students + num_female_students) →
  (adjacent_female_students = 2) →
  (num_arrangements = 432) := by
  sorry

end student_arrangement_count_l2435_243523


namespace remainder_problem_l2435_243557

theorem remainder_problem (x : ℤ) : (x + 11) % 31 = 18 → x % 62 = 7 := by
  sorry

end remainder_problem_l2435_243557


namespace range_of_a_over_b_l2435_243591

def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

def line (a b x y : ℝ) : Prop := a * x + b * y = 2

theorem range_of_a_over_b (a b : ℝ) :
  a^2 + b^2 = 1 →
  b ≠ 0 →
  (∃ x y : ℝ, ellipse x y ∧ line a b x y) →
  (a / b < -1 ∨ a / b = -1 ∨ a / b = 1 ∨ a / b > 1) :=
sorry

end range_of_a_over_b_l2435_243591


namespace a_divisibility_characterization_l2435_243529

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 3
  | 1 => 9
  | (n + 2) => 4 * a (n + 1) - 3 * a n - 4 * (n + 2) + 2

/-- Predicate for n such that a_n is divisible by 9 -/
def is_divisible_by_9 (n : ℕ) : Prop :=
  n = 1 ∨ n % 9 = 7 ∨ n % 9 = 8

theorem a_divisibility_characterization :
  ∀ n : ℕ, 9 ∣ a n ↔ is_divisible_by_9 n :=
sorry

end a_divisibility_characterization_l2435_243529


namespace fraction_calculation_and_comparison_l2435_243539

theorem fraction_calculation_and_comparison : 
  let x := (1/6 - 1/7) / (1/3 - 1/5)
  x = 5/28 ∧ 
  x ≠ 1/4 ∧ 
  x ≠ 1/3 ∧ 
  x ≠ 1/2 ∧ 
  x ≠ 2/5 ∧ 
  x ≠ 3/5 ∧
  x ≠ 1 := by
  sorry

end fraction_calculation_and_comparison_l2435_243539


namespace number_of_pots_l2435_243552

/-- Given a collection of pots where each pot contains 71 flowers,
    and there are 10011 flowers in total, prove that there are 141 pots. -/
theorem number_of_pots (flowers_per_pot : ℕ) (total_flowers : ℕ) (h1 : flowers_per_pot = 71) (h2 : total_flowers = 10011) :
  total_flowers / flowers_per_pot = 141 := by
  sorry


end number_of_pots_l2435_243552


namespace arithmetic_sequence_property_l2435_243535

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∃ d, ∀ n, a (n + 1) = a n + d)

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  2 * a 6 + 2 * a 8 = (a 7) ^ 2 → a 7 = 4 := by
sorry

end arithmetic_sequence_property_l2435_243535


namespace parabolas_intersection_l2435_243574

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ := {-4, 5/2}

/-- The y-coordinates of the intersection points of two parabolas -/
def intersection_y : Set ℝ := {38, 31.5}

/-- First parabola equation -/
def parabola1 (x : ℝ) : ℝ := 4 * x^2 + 5 * x - 6

/-- Second parabola equation -/
def parabola2 (x : ℝ) : ℝ := 2 * x^2 + 14

theorem parabolas_intersection :
  ∀ x ∈ intersection_x, ∃ y ∈ intersection_y,
    parabola1 x = y ∧ parabola2 x = y :=
by sorry

end parabolas_intersection_l2435_243574


namespace workshop_workers_l2435_243579

/-- The total number of workers in the workshop -/
def total_workers : ℕ := 15

/-- The number of technicians in the workshop -/
def num_technicians : ℕ := 5

/-- The average salary of all workers in the workshop -/
def avg_salary_all : ℚ := 700

/-- The average salary of technicians -/
def avg_salary_technicians : ℚ := 800

/-- The average salary of non-technician workers -/
def avg_salary_rest : ℚ := 650

theorem workshop_workers :
  total_workers = num_technicians + 
    (avg_salary_all * total_workers - avg_salary_technicians * num_technicians) / 
    (avg_salary_rest - avg_salary_all) := by
  sorry

end workshop_workers_l2435_243579


namespace decimal_51_to_binary_l2435_243522

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Checks if a list of booleans represents the binary form of a given natural number -/
def is_binary_of (bits : List Bool) (n : ℕ) : Prop :=
  to_binary n = bits.reverse

theorem decimal_51_to_binary :
  is_binary_of [true, true, false, false, true, true] 51 := by
  sorry

end decimal_51_to_binary_l2435_243522


namespace least_positive_integer_with_remainders_l2435_243545

theorem least_positive_integer_with_remainders : ∃ n : ℕ,
  n > 0 ∧
  n % 4 = 1 ∧
  n % 3 = 2 ∧
  n % 5 = 3 ∧
  ∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 3 = 2 ∧ m % 5 = 3 → n ≤ m :=
by sorry

end least_positive_integer_with_remainders_l2435_243545


namespace sum_specific_repeating_decimals_l2435_243532

/-- Represents a repeating decimal with a whole number part and a repeating part -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ
  base : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.whole + (d.repeating : ℚ) / (d.base - 1 : ℚ)

/-- The sum of specific repeating decimals equals 3948/9999 -/
theorem sum_specific_repeating_decimals :
  let d1 := RepeatingDecimal.toRational ⟨0, 3, 10⟩
  let d2 := RepeatingDecimal.toRational ⟨0, 6, 100⟩
  let d3 := RepeatingDecimal.toRational ⟨0, 9, 10000⟩
  d1 + d2 + d3 = 3948 / 9999 := by
  sorry

#eval (3948 : ℚ) / 9999

end sum_specific_repeating_decimals_l2435_243532


namespace sqrt_three_squared_plus_one_l2435_243511

theorem sqrt_three_squared_plus_one : (Real.sqrt 3)^2 + 1 = 4 := by
  sorry

end sqrt_three_squared_plus_one_l2435_243511


namespace min_dials_for_equal_sums_l2435_243592

/-- A type representing a 12-sided dial with numbers from 1 to 12 -/
def Dial := Fin 12 → Fin 12

/-- A stack of dials -/
def Stack := ℕ → Dial

/-- The sum of numbers in a column of the stack -/
def columnSum (s : Stack) (col : Fin 12) (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => (s i col).val + 1)

/-- Whether all column sums have the same remainder modulo 12 -/
def allColumnSumsEqualMod12 (s : Stack) (n : ℕ) : Prop :=
  ∀ (c₁ c₂ : Fin 12), columnSum s c₁ n % 12 = columnSum s c₂ n % 12

/-- The theorem stating that 12 is the minimum number of dials required -/
theorem min_dials_for_equal_sums :
  ∀ (s : Stack), (∃ (n : ℕ), allColumnSumsEqualMod12 s n) →
  (∃ (n : ℕ), n ≥ 12 ∧ allColumnSumsEqualMod12 s n) :=
by sorry

end min_dials_for_equal_sums_l2435_243592


namespace fraction_addition_l2435_243555

theorem fraction_addition : (7 : ℚ) / 8 + (9 : ℚ) / 12 = (13 : ℚ) / 8 := by
  sorry

end fraction_addition_l2435_243555


namespace seventh_twentyninth_150th_digit_l2435_243567

/-- The decimal expansion of 7/29 has a repeating block of length 28 -/
def decimal_period : ℕ := 28

/-- The repeating block in the decimal expansion of 7/29 -/
def repeating_block : List ℕ := [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7, 2]

/-- The 150th digit after the decimal point in the decimal expansion of 7/29 -/
def digit_150 : ℕ := repeating_block[(150 - 1) % decimal_period]

theorem seventh_twentyninth_150th_digit :
  digit_150 = 8 := by sorry

end seventh_twentyninth_150th_digit_l2435_243567


namespace fourth_jeweler_bags_l2435_243510

def bags : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def total_gold : ℕ := bags.sum

theorem fourth_jeweler_bags (lost_bag : ℕ) 
  (h1 : lost_bag ∈ bags)
  (h2 : lost_bag ≠ 1 ∧ lost_bag ≠ 3 ∧ lost_bag ≠ 11)
  (h3 : (total_gold - lost_bag) % 4 = 0)
  (h4 : (bags.length - 1) % 4 = 0) :
  ∃ (jeweler1 jeweler2 jeweler3 jeweler4 : List ℕ),
    jeweler1.sum = jeweler2.sum ∧
    jeweler2.sum = jeweler3.sum ∧
    jeweler3.sum = jeweler4.sum ∧
    jeweler1.length = jeweler2.length ∧
    jeweler2.length = jeweler3.length ∧
    jeweler3.length = jeweler4.length ∧
    1 ∈ jeweler1 ∧
    3 ∈ jeweler2 ∧
    11 ∈ jeweler3 ∧
    jeweler4 = [2, 9, 10] :=
by
  sorry

end fourth_jeweler_bags_l2435_243510


namespace minimum_score_to_increase_average_l2435_243538

def scores : List ℕ := [84, 76, 89, 94, 67, 90]

def current_average : ℚ := (scores.sum : ℚ) / scores.length

def target_average : ℚ := current_average + 5

def required_score : ℕ := 118

theorem minimum_score_to_increase_average : 
  (((scores.sum + required_score : ℚ) / (scores.length + 1)) = target_average) ∧
  (∀ (s : ℕ), s < required_score → 
    ((scores.sum + s : ℚ) / (scores.length + 1)) < target_average) := by
  sorry

end minimum_score_to_increase_average_l2435_243538


namespace bluejay_female_fraction_l2435_243583

theorem bluejay_female_fraction (total_birds : ℝ) (total_birds_pos : 0 < total_birds) : 
  let robins := (2/5) * total_birds
  let bluejays := (3/5) * total_birds
  let female_robins := (1/3) * robins
  let male_birds := (7/15) * total_birds
  let female_bluejays := ((8/15) * total_birds) - female_robins
  (female_bluejays / bluejays) = (2/3) :=
by sorry

end bluejay_female_fraction_l2435_243583


namespace min_quadratic_l2435_243598

theorem min_quadratic (x : ℝ) : 
  (∀ y : ℝ, x^2 + 7*x + 3 ≤ y^2 + 7*y + 3) → x = -7/2 := by
  sorry

end min_quadratic_l2435_243598


namespace tv_screen_coverage_l2435_243576

theorem tv_screen_coverage (w1 h1 w2 h2 : ℚ) : 
  w1 / h1 = 16 / 9 →
  w2 / h2 = 4 / 3 →
  (h2 - h1 * (w2 / w1)) / h2 = 1 / 4 := by
sorry

end tv_screen_coverage_l2435_243576


namespace major_axis_endpoints_of_ellipse_l2435_243581

/-- An ellipse is defined by the equation 6x^2 + y^2 = 36 -/
def ellipse (x y : ℝ) : Prop := 6 * x^2 + y^2 = 36

/-- The endpoints of the major axis of the ellipse -/
def major_axis_endpoints : Set (ℝ × ℝ) := {(-6, 0), (6, 0)}

/-- Theorem: The coordinates of the endpoints of the major axis of the ellipse 6x^2 + y^2 = 36 are (0, -6) and (0, 6) -/
theorem major_axis_endpoints_of_ellipse :
  major_axis_endpoints = {(0, -6), (0, 6)} :=
sorry

end major_axis_endpoints_of_ellipse_l2435_243581


namespace symmetry_of_transformed_functions_l2435_243566

/-- Given a function f, prove that the graphs of f(x-1) and f(-x+1) are symmetric with respect to the line x = 1 -/
theorem symmetry_of_transformed_functions (f : ℝ → ℝ) : 
  ∀ (x y : ℝ), f (x - 1) = y ↔ f (-(x - 1)) = y :=
by sorry

end symmetry_of_transformed_functions_l2435_243566


namespace trapezoid_shorter_base_l2435_243507

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ

/-- The property of the trapezoid that the segment joining the midpoints of the diagonals
    is half the difference of the bases -/
def trapezoid_property (t : Trapezoid) : Prop :=
  t.midpoint_segment = (t.longer_base - t.shorter_base) / 2

theorem trapezoid_shorter_base (t : Trapezoid) 
  (h1 : t.longer_base = 113)
  (h2 : t.midpoint_segment = 5)
  (h3 : trapezoid_property t) :
  t.shorter_base = 103 := by
  sorry

#check trapezoid_shorter_base

end trapezoid_shorter_base_l2435_243507


namespace competition_participants_l2435_243558

theorem competition_participants (right_rank left_rank : ℕ) 
  (h1 : right_rank = 18) 
  (h2 : left_rank = 12) : 
  right_rank + left_rank - 1 = 29 := by
  sorry

end competition_participants_l2435_243558


namespace triangle_third_side_length_l2435_243562

theorem triangle_third_side_length (a b c : ℝ) (θ : ℝ) : 
  a = 10 → b = 12 → θ = π / 3 → c = Real.sqrt 124 := by
  sorry

end triangle_third_side_length_l2435_243562


namespace hyperbola_asymptotes_l2435_243500

/-- Represents a hyperbola -/
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_axes : Bool
  eccentricity : ℝ

/-- The equation of asymptotes for a hyperbola -/
def asymptote_equation (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = x ∨ y = -x}

theorem hyperbola_asymptotes (C : Hyperbola) 
  (h1 : C.center = (0, 0)) 
  (h2 : C.foci_on_axes = true) 
  (h3 : C.eccentricity = Real.sqrt 2) : 
  asymptote_equation C = {(x, y) | y = x ∨ y = -x} := by
  sorry

end hyperbola_asymptotes_l2435_243500


namespace customers_who_left_l2435_243564

theorem customers_who_left (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 13 → new = 4 → final = 9 → initial - (initial - final + new) = 8 := by
  sorry

end customers_who_left_l2435_243564


namespace book_costs_and_plans_l2435_243559

/-- Represents the cost and quantity of books --/
structure BookOrder where
  lao_she : ℕ
  classics : ℕ
  total_cost : ℕ

/-- Represents a purchasing plan --/
structure PurchasePlan where
  lao_she : ℕ
  classics : ℕ

def is_valid_plan (p : PurchasePlan) (lao_she_cost classics_cost : ℕ) : Prop :=
  p.lao_she + p.classics = 20 ∧
  p.lao_she ≤ 2 * p.classics ∧
  p.lao_she * lao_she_cost + p.classics * classics_cost ≤ 1720

theorem book_costs_and_plans : ∃ (lao_she_cost classics_cost : ℕ) (plans : List PurchasePlan),
  let order1 : BookOrder := ⟨4, 2, 480⟩
  let order2 : BookOrder := ⟨2, 3, 520⟩
  (order1.lao_she * lao_she_cost + order1.classics * classics_cost = order1.total_cost) ∧
  (order2.lao_she * lao_she_cost + order2.classics * classics_cost = order2.total_cost) ∧
  (lao_she_cost = 50) ∧
  (classics_cost = 140) ∧
  (plans.length = 2) ∧
  (∀ p ∈ plans, is_valid_plan p lao_she_cost classics_cost) ∧
  (∀ p : PurchasePlan, is_valid_plan p lao_she_cost classics_cost → p ∈ plans) :=
by sorry


end book_costs_and_plans_l2435_243559


namespace cube_sum_root_l2435_243521

theorem cube_sum_root : Real.sqrt (3^3 + 3^3 + 3^3) = 9 := by
  sorry

end cube_sum_root_l2435_243521


namespace line_increase_l2435_243554

/-- Given a line in the Cartesian plane where an increase of 2 units in x
    corresponds to an increase of 5 units in y, prove that an increase of 8 units
    in x will result in an increase of 20 units in y. -/
theorem line_increase (f : ℝ → ℝ) (h : ∀ x, f (x + 2) - f x = 5) :
  ∀ x, f (x + 8) - f x = 20 := by
  sorry

end line_increase_l2435_243554


namespace tenth_toss_probability_l2435_243547

/-- A fair coin is a coin with equal probability of landing heads or tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of getting heads on a single toss of a fair coin -/
def prob_heads (p : ℝ) : Prop := p = 1/2

/-- The number of times the coin has been tossed -/
def num_tosses : ℕ := 9

/-- The number of heads obtained in the previous tosses -/
def num_heads : ℕ := 7

/-- The number of tails obtained in the previous tosses -/
def num_tails : ℕ := 2

theorem tenth_toss_probability (p : ℝ) 
  (h_fair : fair_coin p) 
  (h_prev_tosses : num_tosses = num_heads + num_tails) :
  prob_heads p := by sorry

end tenth_toss_probability_l2435_243547


namespace third_term_base_l2435_243553

theorem third_term_base (x : ℝ) (some_number : ℝ) 
  (h1 : 625^(-x) + 25^(-2*x) + some_number^(-4*x) = 14)
  (h2 : x = 0.25) : 
  some_number = 125/1744 := by
sorry

end third_term_base_l2435_243553


namespace painter_workdays_l2435_243570

theorem painter_workdays (initial_painters : ℕ) (initial_days : ℚ) (new_painters : ℕ) :
  initial_painters = 5 →
  initial_days = 4/5 →
  new_painters = 4 →
  (initial_painters : ℚ) * initial_days = new_painters * 1 :=
by sorry

end painter_workdays_l2435_243570


namespace root_magnitude_l2435_243506

theorem root_magnitude (a b : ℝ) (z : ℂ) (h : z = 1 + b * Complex.I) 
  (h_root : z ^ 2 + a * z + 3 = 0) : Complex.abs z = Real.sqrt 3 := by
  sorry

end root_magnitude_l2435_243506


namespace triangle_abc_properties_l2435_243597

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_abc_properties (t : Triangle) 
  (ha : t.a = 3)
  (hb : t.b = 2)
  (hcosA : Real.cos t.A = 1/3) :
  Real.sin t.B = 4 * Real.sqrt 2 / 9 ∧ t.c = 3 := by
  sorry


end triangle_abc_properties_l2435_243597


namespace expression_simplification_l2435_243590

theorem expression_simplification (x y : ℝ) : 
  -(3*x*y - 2*x^2) - 2*(3*x^2 - x*y) = -4*x^2 - x*y := by
  sorry

end expression_simplification_l2435_243590


namespace parabola_focus_directrix_distance_l2435_243575

/-- Given a parabola y² = 2px with a point Q(6, y₀) on it, 
    if the distance from Q to the focus is 10, 
    then the distance from the focus to the directrix is 8. -/
theorem parabola_focus_directrix_distance 
  (p : ℝ) (y₀ : ℝ) :
  y₀^2 = 2*p*6 → -- Q(6, y₀) lies on the parabola y² = 2px
  (6 + p/2)^2 + y₀^2 = 10^2 → -- distance from Q to focus is 10
  p = 8 := -- distance from focus to directrix
by sorry

end parabola_focus_directrix_distance_l2435_243575


namespace decimal_division_remainder_l2435_243571

theorem decimal_division_remainder (n : ℕ) (N : ℕ) : 
  (N % (2^n) = (N % 10^n) % (2^n)) ∧ (N % (5^n) = (N % 10^n) % (5^n)) := by sorry

end decimal_division_remainder_l2435_243571


namespace smallest_valid_side_l2435_243565

-- Define the triangle sides
def a : ℝ := 7.5
def b : ℝ := 14.5

-- Define the property of s being a valid side length
def is_valid_side (s : ℕ) : Prop :=
  (a + s > b) ∧ (a + b > s) ∧ (b + s > a)

-- State the theorem
theorem smallest_valid_side :
  ∃ (s : ℕ), is_valid_side s ∧ ∀ (t : ℕ), t < s → ¬is_valid_side t :=
by sorry

end smallest_valid_side_l2435_243565


namespace change_parity_mismatch_l2435_243549

theorem change_parity_mismatch (bills : List ℕ) (denominations : List ℕ) :
  (bills.length = 10) →
  (∀ d ∈ denominations, d % 2 = 1) →
  (∀ b ∈ bills, b ∈ denominations) →
  (bills.sum ≠ 31) :=
sorry

end change_parity_mismatch_l2435_243549


namespace matrix_sum_theorem_l2435_243508

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; -1, 2]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, -3]

theorem matrix_sum_theorem :
  A + B = !![3, 4; 0, -1] := by sorry

end matrix_sum_theorem_l2435_243508


namespace two_lights_possible_l2435_243569

/-- Represents the state of light bulbs on an infinite integer line -/
def LightState := Int → Bool

/-- Applies the template set S to the light state at position p -/
def applyTemplate (S : Finset Int) (state : LightState) (p : Int) : LightState :=
  fun i => if (i - p) ∈ S then !state i else state i

/-- Counts the number of light bulbs that are on -/
def countOn (state : LightState) : Nat :=
  sorry

theorem two_lights_possible (S : Finset Int) :
  ∃ (ops : List Int), 
    let finalState := ops.foldl (fun st p => applyTemplate S st p) (fun _ => false)
    countOn finalState = 2 :=
  sorry

end two_lights_possible_l2435_243569


namespace selling_price_optimal_l2435_243533

/-- Represents the selling price of toy A in yuan -/
def selling_price : ℝ := 65

/-- Represents the purchase price of toy A in yuan -/
def purchase_price : ℝ := 60

/-- Represents the maximum allowed profit margin -/
def max_profit_margin : ℝ := 0.4

/-- Represents the daily profit target in yuan -/
def profit_target : ℝ := 2500

/-- Calculates the number of units sold per day based on the selling price -/
def units_sold (x : ℝ) : ℝ := 1800 - 20 * x

/-- Calculates the profit per unit based on the selling price -/
def profit_per_unit (x : ℝ) : ℝ := x - purchase_price

/-- Calculates the total daily profit based on the selling price -/
def daily_profit (x : ℝ) : ℝ := profit_per_unit x * units_sold x

/-- Theorem stating that the selling price of 65 yuan results in the target profit
    while satisfying the profit margin constraint -/
theorem selling_price_optimal :
  daily_profit selling_price = profit_target ∧
  profit_per_unit selling_price / selling_price ≤ max_profit_margin :=
by sorry

end selling_price_optimal_l2435_243533


namespace cody_money_calculation_l2435_243541

def final_money (initial : ℝ) (birthday_gift : ℝ) (game_cost : ℝ) (clothes_percentage : ℝ) (late_gift : ℝ) : ℝ :=
  let after_birthday := initial + birthday_gift
  let after_game := after_birthday - game_cost
  let clothes_cost := clothes_percentage * after_game
  let after_clothes := after_game - clothes_cost
  after_clothes + late_gift

theorem cody_money_calculation :
  final_money 45 9 19 0.4 4.5 = 25.5 := by
  sorry

end cody_money_calculation_l2435_243541


namespace union_problem_l2435_243509

def M : Set ℕ := {0, 1, 2}
def N (x : ℕ) : Set ℕ := {x}

theorem union_problem (x : ℕ) : M ∪ N x = {0, 1, 2, 3} → x = 3 := by
  sorry

end union_problem_l2435_243509


namespace abc_mod_five_l2435_243560

theorem abc_mod_five (a b c : ℕ) : 
  a < 5 → b < 5 → c < 5 →
  (a + 2*b + 3*c) % 5 = 3 →
  (2*a + 3*b + c) % 5 = 2 →
  (3*a + b + 2*c) % 5 = 1 →
  (a*b*c) % 5 = 3 := by
  sorry

#check abc_mod_five

end abc_mod_five_l2435_243560


namespace max_silver_tokens_l2435_243586

/-- Represents the number of tokens Alex has --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules --/
inductive ExchangeRule
  | RedToSilver : ExchangeRule  -- 3 red → 2 silver + 1 blue
  | BlueToSilver : ExchangeRule -- 2 blue → 1 silver + 1 red

/-- Applies an exchange rule to a token count --/
def applyExchange (tc : TokenCount) (rule : ExchangeRule) : Option TokenCount :=
  match rule with
  | ExchangeRule.RedToSilver =>
      if tc.red ≥ 3 then
        some ⟨tc.red - 3, tc.blue + 1, tc.silver + 2⟩
      else
        none
  | ExchangeRule.BlueToSilver =>
      if tc.blue ≥ 2 then
        some ⟨tc.red + 1, tc.blue - 2, tc.silver + 1⟩
      else
        none

/-- Checks if any exchange is possible --/
def canExchange (tc : TokenCount) : Bool :=
  tc.red ≥ 3 ∨ tc.blue ≥ 2

/-- The main theorem to prove --/
theorem max_silver_tokens (initialRed initialBlue : ℕ) 
    (h1 : initialRed = 100) (h2 : initialBlue = 50) :
    ∃ (finalTokens : TokenCount),
      finalTokens.red < 3 ∧ 
      finalTokens.blue < 2 ∧
      finalTokens.silver = 147 ∧
      (∃ (exchanges : List ExchangeRule), 
        finalTokens = exchanges.foldl 
          (fun acc rule => 
            match applyExchange acc rule with
            | some newCount => newCount
            | none => acc) 
          ⟨initialRed, initialBlue, 0⟩) := by
  sorry

end max_silver_tokens_l2435_243586


namespace cuboid_specific_surface_area_l2435_243542

/-- The surface area of a cuboid with given dimensions -/
def cuboid_surface_area (length breadth height : ℝ) : ℝ :=
  2 * (length * height + length * breadth + breadth * height)

/-- Theorem: The surface area of a cuboid with length 12 cm, breadth 6 cm, and height 10 cm is 504 cm² -/
theorem cuboid_specific_surface_area :
  cuboid_surface_area 12 6 10 = 504 := by
  sorry

end cuboid_specific_surface_area_l2435_243542


namespace inequality_proof_l2435_243536

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^4 + b^4 + c^4 - 2*(a^2*b^2 + a^2*c^2 + b^2*c^2) + a^2*b*c + b^2*a*c + c^2*a*b ≥ 0 :=
by sorry

end inequality_proof_l2435_243536


namespace function_definition_l2435_243599

-- Define the function property
def is_function {A B : Type} (f : A → B) : Prop :=
  ∀ x : A, ∃! y : B, f x = y

-- State the theorem
theorem function_definition {A B : Type} (f : A → B) :
  is_function f ↔ ∀ x : A, ∃! y : B, y = f x :=
by sorry

end function_definition_l2435_243599


namespace product_greater_than_sum_l2435_243512

theorem product_greater_than_sum (a b : ℝ) (ha : a ≥ 2) (hb : b > 2) : a * b > a + b := by
  sorry

end product_greater_than_sum_l2435_243512


namespace complete_collection_size_jerry_collection_l2435_243544

theorem complete_collection_size (initial_figures : ℕ) (figure_cost : ℕ) (additional_cost : ℕ) : ℕ :=
  let additional_figures := additional_cost / figure_cost
  initial_figures + additional_figures

theorem jerry_collection :
  complete_collection_size 7 8 72 = 16 := by
  sorry

end complete_collection_size_jerry_collection_l2435_243544


namespace log4_one_sixteenth_eq_neg_two_l2435_243580

-- Define the logarithm function for base 4
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

-- Theorem statement
theorem log4_one_sixteenth_eq_neg_two : log4 (1/16) = -2 := by
  sorry

end log4_one_sixteenth_eq_neg_two_l2435_243580


namespace right_triangle_hypotenuse_l2435_243537

/-- Given a right triangle ABC with angle C = 90°, side a = 12, and side b = 16, prove that the length of side c is 20. -/
theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 12 → b = 16 → c^2 = a^2 + b^2 → c = 20 := by
  sorry

end right_triangle_hypotenuse_l2435_243537


namespace equation_solution_l2435_243503

theorem equation_solution :
  let f (x : ℚ) := 1 - (3 + 2*x) / 4 = (x + 3) / 6
  ∃ (x : ℚ), f x ∧ x = -3/8 :=
by sorry

end equation_solution_l2435_243503


namespace difference_of_numbers_l2435_243593

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 275) : |x - y| = 14 := by
  sorry

end difference_of_numbers_l2435_243593


namespace smallest_consecutive_integer_sum_l2435_243520

/-- The sum of 15 consecutive positive integers that is a perfect square -/
def consecutiveIntegerSum (n : ℕ) : ℕ := 15 * (n + 7)

/-- The sum is a perfect square -/
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_consecutive_integer_sum :
  (∃ n : ℕ, isPerfectSquare (consecutiveIntegerSum n)) →
  (∀ m : ℕ, isPerfectSquare (consecutiveIntegerSum m) → consecutiveIntegerSum m ≥ 225) :=
by sorry

end smallest_consecutive_integer_sum_l2435_243520


namespace sequence_length_divisible_by_three_l2435_243524

theorem sequence_length_divisible_by_three (n : ℕ) (a : ℕ → ℝ) 
  (h1 : n ≥ 3)
  (h2 : ∀ i, a (i + n) = a i)
  (h3 : ∀ i, a i * a (i + 1) + 1 = a (i + 2)) :
  ∃ k : ℕ, n = 3 * k :=
sorry

end sequence_length_divisible_by_three_l2435_243524


namespace exists_coverable_prism_l2435_243540

/-- Represents a regular triangular prism -/
structure RegularTriangularPrism where
  base_side : ℝ
  lateral_edge : ℝ
  lateral_edge_eq : lateral_edge = Real.sqrt 3 * base_side

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ

/-- Predicate to check if a prism can be covered by equilateral triangles -/
def can_cover_with_equilateral_triangles (p : RegularTriangularPrism) (t : EquilateralTriangle) : Prop :=
  p.base_side = t.side ∧
  p.lateral_edge = Real.sqrt 3 * t.side

/-- Theorem stating that there exists a regular triangular prism that can be covered by equilateral triangles -/
theorem exists_coverable_prism : 
  ∃ (p : RegularTriangularPrism) (t : EquilateralTriangle), 
    can_cover_with_equilateral_triangles p t := by
  sorry

end exists_coverable_prism_l2435_243540


namespace three_roots_and_minimum_implies_ratio_l2435_243550

/-- Given positive real numbers a, b, c with a > c, if the equation |x²-ax+b| = cx 
    has exactly three distinct real roots, and the function f(x) = |x²-ax+b| + cx 
    has a minimum value of c², then a/c = 5 -/
theorem three_roots_and_minimum_implies_ratio (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hac : a > c)
  (h_three_roots : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    |x^2 - a*x + b| = c*x ∧ |y^2 - a*y + b| = c*y ∧ |z^2 - a*z + b| = c*z)
  (h_min : ∃ m : ℝ, ∀ x : ℝ, |x^2 - a*x + b| + c*x ≥ c^2 ∧ 
    ∃ x₀ : ℝ, |x₀^2 - a*x₀ + b| + c*x₀ = c^2) :
  a / c = 5 := by
sorry

end three_roots_and_minimum_implies_ratio_l2435_243550


namespace hyperbola_s_squared_l2435_243587

/-- A hyperbola passing through specific points -/
structure Hyperbola where
  /-- The hyperbola is centered at the origin -/
  center : (ℝ × ℝ) := (0, 0)
  /-- The hyperbola passes through (5, -6) -/
  point1 : (ℝ × ℝ) := (5, -6)
  /-- The hyperbola passes through (3, 0) -/
  point2 : (ℝ × ℝ) := (3, 0)
  /-- The hyperbola passes through (s, -3) for some real s -/
  point3 : (ℝ × ℝ)
  /-- The third point has y-coordinate -3 -/
  h_point3_y : point3.2 = -3

/-- The theorem stating that s² = 12 for the given hyperbola -/
theorem hyperbola_s_squared (h : Hyperbola) : h.point3.1 ^ 2 = 12 := by
  sorry

end hyperbola_s_squared_l2435_243587


namespace quadratic_roots_relation_l2435_243551

theorem quadratic_roots_relation (b c : ℚ) : 
  (∃ r₁ r₂ : ℚ, r₁ ≠ r₂ ∧ 
    (∀ x : ℚ, x^2 + b*x + c = 0 ↔ x = r₁ ∨ x = r₂) ∧
    (∃ s₁ s₂ : ℚ, s₁ ≠ s₂ ∧ 
      (∀ x : ℚ, 3*x^2 - 5*x - 7 = 0 ↔ x = s₁ ∨ x = s₂) ∧
      r₁ = s₁ + 3 ∧ r₂ = s₂ + 3)) →
  c = 35/3 := by
sorry

end quadratic_roots_relation_l2435_243551


namespace division_problem_l2435_243578

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 181 → 
  quotient = 9 → 
  remainder = 1 → 
  dividend = divisor * quotient + remainder →
  divisor = 20 := by sorry

end division_problem_l2435_243578
