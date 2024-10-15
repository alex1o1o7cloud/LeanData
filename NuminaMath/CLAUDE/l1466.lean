import Mathlib

namespace NUMINAMATH_CALUDE_M_intersect_N_equals_M_l1466_146653

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | |x - 1| < 1}
def N : Set ℝ := {x : ℝ | x * (x - 3) < 0}

-- State the theorem
theorem M_intersect_N_equals_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_M_l1466_146653


namespace NUMINAMATH_CALUDE_pie_eating_contest_l1466_146612

theorem pie_eating_contest (first_round first_second_round second_total : ℚ) 
  (h1 : first_round = 5/6)
  (h2 : first_second_round = 1/6)
  (h3 : second_total = 2/3) :
  first_round + first_second_round - second_total = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l1466_146612


namespace NUMINAMATH_CALUDE_angle_4_value_l1466_146691

theorem angle_4_value (angle1 angle2 angle3 angle4 : ℝ) : 
  angle1 + angle2 = 180 →
  angle3 = 2 * angle4 →
  angle1 = 50 →
  angle3 + angle4 = 130 →
  angle4 = 130 / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_4_value_l1466_146691


namespace NUMINAMATH_CALUDE_expand_product_l1466_146613

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x - 5) = 6 * x^2 - 7 * x - 20 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1466_146613


namespace NUMINAMATH_CALUDE_football_players_count_l1466_146625

theorem football_players_count (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 39)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 10) :
  ∃ football : ℕ, football = 26 ∧ (football - both) + (tennis - both) + both + neither = total :=
by sorry

end NUMINAMATH_CALUDE_football_players_count_l1466_146625


namespace NUMINAMATH_CALUDE_integer_count_between_negatives_l1466_146659

theorem integer_count_between_negatives (a : ℚ) : 
  (a > 0) → 
  (∃ n : ℕ, n = (⌊a⌋ - ⌈-a⌉ - 1) ∧ n = 2007) → 
  (1003 < a ∧ a ≤ 1004) :=
by
  sorry

end NUMINAMATH_CALUDE_integer_count_between_negatives_l1466_146659


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l1466_146637

theorem polynomial_product_expansion :
  ∀ x : ℝ, (x^2 - 2*x + 2) * (x^2 + 2*x + 2) = x^4 + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l1466_146637


namespace NUMINAMATH_CALUDE_pistachio_shell_percentage_l1466_146608

theorem pistachio_shell_percentage (total : ℕ) (shell_percent : ℚ) (opened_shells : ℕ) : 
  total = 80 →
  shell_percent = 95 / 100 →
  opened_shells = 57 →
  (opened_shells : ℚ) / (shell_percent * total) * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_pistachio_shell_percentage_l1466_146608


namespace NUMINAMATH_CALUDE_decimal_expansion_non_periodic_length_l1466_146671

/-- The length of the non-periodic part of the decimal expansion of 1/n -/
def nonPeriodicLength (n : ℕ) : ℕ :=
  max (Nat.factorization n 2) (Nat.factorization n 5)

/-- Theorem stating that for any natural number n > 1, the length of the non-periodic part
    of the decimal expansion of 1/n is equal to max[v₂(n), v₅(n)] -/
theorem decimal_expansion_non_periodic_length (n : ℕ) (h : n > 1) :
  nonPeriodicLength n = max (Nat.factorization n 2) (Nat.factorization n 5) := by
  sorry

#check decimal_expansion_non_periodic_length

end NUMINAMATH_CALUDE_decimal_expansion_non_periodic_length_l1466_146671


namespace NUMINAMATH_CALUDE_toms_beach_trip_l1466_146652

/-- Tom's beach trip problem -/
theorem toms_beach_trip (daily_seashells : ℕ) (total_seashells : ℕ) (days : ℕ) :
  daily_seashells = 7 →
  total_seashells = 35 →
  total_seashells = daily_seashells * days →
  days = 5 := by
  sorry

end NUMINAMATH_CALUDE_toms_beach_trip_l1466_146652


namespace NUMINAMATH_CALUDE_wire_cutting_theorem_l1466_146661

def wire1 : ℕ := 1008
def wire2 : ℕ := 1260
def wire3 : ℕ := 882
def wire4 : ℕ := 1134

def segment_length : ℕ := 126
def total_segments : ℕ := 34

theorem wire_cutting_theorem :
  (∃ (n : ℕ), n > 0 ∧ 
    wire1 % n = 0 ∧ 
    wire2 % n = 0 ∧ 
    wire3 % n = 0 ∧ 
    wire4 % n = 0 ∧
    ∀ (m : ℕ), m > n → 
      (wire1 % m ≠ 0 ∨ 
       wire2 % m ≠ 0 ∨ 
       wire3 % m ≠ 0 ∨ 
       wire4 % m ≠ 0)) ∧
  segment_length = 126 ∧
  total_segments = 34 ∧
  wire1 / segment_length + 
  wire2 / segment_length + 
  wire3 / segment_length + 
  wire4 / segment_length = total_segments :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_theorem_l1466_146661


namespace NUMINAMATH_CALUDE_divisor_of_q_l1466_146647

theorem divisor_of_q (p q r s : ℕ+) 
  (h1 : Nat.gcd p.val q.val = 30)
  (h2 : Nat.gcd q.val r.val = 42)
  (h3 : Nat.gcd r.val s.val = 66)
  (h4 : 80 < Nat.gcd s.val p.val ∧ Nat.gcd s.val p.val < 120) :
  5 ∣ q.val :=
by sorry

end NUMINAMATH_CALUDE_divisor_of_q_l1466_146647


namespace NUMINAMATH_CALUDE_product_simplification_l1466_146648

theorem product_simplification (x : ℝ) (h : x ≠ 0) :
  (12 * x^3) * (8 * x^2) * (1 / (4*x)^3) = (3/2) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l1466_146648


namespace NUMINAMATH_CALUDE_triangle_not_right_angle_l1466_146604

theorem triangle_not_right_angle (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : A / 3 = B / 4) (h3 : A / 3 = C / 5) : 
  A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_not_right_angle_l1466_146604


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1466_146656

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (given_line : Line) 
  (p : Point) 
  (h_given : given_line.a = 6 ∧ given_line.b = -5 ∧ given_line.c = 3) 
  (h_point : p.x = 1 ∧ p.y = 1) :
  ∃ (result_line : Line), 
    result_line.a = 6 ∧ 
    result_line.b = -5 ∧ 
    result_line.c = -1 ∧ 
    parallel result_line given_line ∧ 
    pointOnLine p result_line := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1466_146656


namespace NUMINAMATH_CALUDE_not_square_difference_l1466_146666

-- Define the square difference formula
def square_difference (a b : ℝ → ℝ) : ℝ → ℝ := λ x => (a x)^2 - (b x)^2

-- Define the expression we want to prove doesn't fit the square difference formula
def expression : ℝ → ℝ := λ x => (x + 1) * (1 + x)

-- Theorem statement
theorem not_square_difference :
  ¬ ∃ (a b : ℝ → ℝ), ∀ x, expression x = square_difference a b x :=
sorry

end NUMINAMATH_CALUDE_not_square_difference_l1466_146666


namespace NUMINAMATH_CALUDE_james_waiting_time_l1466_146667

/-- The number of days it took for James' pain to subside -/
def pain_subsided_days : ℕ := 3

/-- The factor by which the full healing time is longer than the pain subsidence time -/
def healing_factor : ℕ := 5

/-- The number of days James waits after healing before working out -/
def wait_before_workout_days : ℕ := 3

/-- The total number of days until James can lift heavy again -/
def total_days_until_heavy_lifting : ℕ := 39

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem james_waiting_time :
  (total_days_until_heavy_lifting - (pain_subsided_days * healing_factor + wait_before_workout_days)) / days_per_week = 3 := by
  sorry

end NUMINAMATH_CALUDE_james_waiting_time_l1466_146667


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1466_146674

theorem unique_solution_for_equation (m n p : ℕ+) (h_prime : Nat.Prime p) :
  2^(m : ℕ) * p^2 + 1 = n^5 ↔ m = 1 ∧ n = 3 ∧ p = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1466_146674


namespace NUMINAMATH_CALUDE_lower_right_is_one_l1466_146643

/-- Represents a 4x4 grid of integers -/
def Grid := Fin 4 → Fin 4 → Fin 4

/-- Checks if a given grid satisfies the Latin square property -/
def is_latin_square (g : Grid) : Prop :=
  (∀ i j k, i ≠ k → g i j ≠ g k j) ∧ 
  (∀ i j k, j ≠ k → g i j ≠ g i k)

/-- The initial configuration of the grid -/
def initial_config (g : Grid) : Prop :=
  g 0 0 = 0 ∧ g 0 3 = 3 ∧ g 1 1 = 1 ∧ g 2 2 = 2

theorem lower_right_is_one (g : Grid) 
  (h1 : is_latin_square g) 
  (h2 : initial_config g) : 
  g 3 3 = 0 := by sorry

end NUMINAMATH_CALUDE_lower_right_is_one_l1466_146643


namespace NUMINAMATH_CALUDE_cone_volume_from_semicircle_l1466_146602

theorem cone_volume_from_semicircle (r : ℝ) (h : r = 6) :
  let l := r  -- slant height
  let base_radius := r / 2  -- derived from circumference equality
  let height := Real.sqrt (l^2 - base_radius^2)
  let volume := (1/3) * Real.pi * base_radius^2 * height
  volume = 9 * Real.sqrt 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_semicircle_l1466_146602


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1466_146663

/-- An arithmetic sequence -/
def arithmeticSeq (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property that a₁ + a₇ + a₁₃ = 4 -/
def sumProperty (a : ℕ → ℚ) : Prop :=
  a 1 + a 7 + a 13 = 4

theorem arithmetic_sequence_sum (a : ℕ → ℚ) 
  (h1 : arithmeticSeq a) (h2 : sumProperty a) : 
  a 2 + a 12 = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1466_146663


namespace NUMINAMATH_CALUDE_female_students_count_l1466_146631

theorem female_students_count (total_students sample_size male_sample : ℕ) 
  (h1 : total_students = 2000)
  (h2 : sample_size = 200)
  (h3 : male_sample = 103) : 
  (total_students : ℚ) * ((sample_size - male_sample) : ℚ) / (sample_size : ℚ) = 970 := by
  sorry

end NUMINAMATH_CALUDE_female_students_count_l1466_146631


namespace NUMINAMATH_CALUDE_upper_limit_of_multiples_l1466_146682

def average_of_multiples_of_10 (n : ℕ) : ℚ :=
  (n * (10 + n)) / (2 * n)

theorem upper_limit_of_multiples (n : ℕ) :
  n ≥ 10 → average_of_multiples_of_10 n = 55 → n = 100 := by
  sorry

end NUMINAMATH_CALUDE_upper_limit_of_multiples_l1466_146682


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1466_146609

theorem solve_linear_equation (x : ℝ) (h : x - 3*x + 4*x = 120) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1466_146609


namespace NUMINAMATH_CALUDE_helmet_cost_l1466_146699

theorem helmet_cost (total_cost bicycle_cost helmet_cost : ℝ) : 
  total_cost = 240 →
  bicycle_cost = 5 * helmet_cost →
  total_cost = bicycle_cost + helmet_cost →
  helmet_cost = 40 := by
sorry

end NUMINAMATH_CALUDE_helmet_cost_l1466_146699


namespace NUMINAMATH_CALUDE_positive_A_value_l1466_146605

-- Define the relation #
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 218) : A = 13 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l1466_146605


namespace NUMINAMATH_CALUDE_mean_of_five_numbers_with_sum_two_thirds_l1466_146654

theorem mean_of_five_numbers_with_sum_two_thirds :
  ∀ (a b c d e : ℚ),
  a + b + c + d + e = 2/3 →
  (a + b + c + d + e) / 5 = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_five_numbers_with_sum_two_thirds_l1466_146654


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l1466_146692

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_less_than_10 : a < 10
  b_less_than_10 : b < 10
  c_less_than_10 : c < 10
  d_less_than_10 : d < 10

/-- The conditions given in the problem -/
def satisfiesConditions (n : FourDigitNumber) : Prop :=
  n.a + n.b + n.c + n.d = 26 ∧
  (n.b * n.d) / 10 = n.a + n.c ∧
  ∃ k : Nat, n.b * n.d - n.c * n.c = 2 * k

/-- The theorem to prove -/
theorem unique_four_digit_number :
  ∃! n : FourDigitNumber, satisfiesConditions n ∧ 
    n.a = 1 ∧ n.b = 9 ∧ n.c = 7 ∧ n.d = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l1466_146692


namespace NUMINAMATH_CALUDE_min_handshakes_coach_l1466_146639

/-- Represents the number of handshakes in a volleyball tournament --/
structure VolleyballTournament where
  n : ℕ  -- Total number of players
  m : ℕ  -- Number of players in the smaller team
  k₁ : ℕ -- Number of handshakes by the coach with fewer players
  h : ℕ  -- Total number of handshakes

/-- Conditions for the volleyball tournament --/
def tournament_conditions (t : VolleyballTournament) : Prop :=
  t.n = 3 * t.m ∧                                  -- Total players is 3 times the smaller team
  t.h = (t.n * (t.n - 1)) / 2 + 3 * t.k₁ ∧         -- Total handshakes equation
  t.h = 435                                        -- Given total handshakes

/-- Theorem stating the minimum number of handshakes for the coach with fewer players --/
theorem min_handshakes_coach (t : VolleyballTournament) :
  tournament_conditions t → t.k₁ ≥ 0 → t.k₁ = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_min_handshakes_coach_l1466_146639


namespace NUMINAMATH_CALUDE_book_distribution_l1466_146614

theorem book_distribution (people : ℕ) (books : ℕ) : 
  (5 * people = books + 2) →
  (4 * people + 3 = books) →
  (people = 5 ∧ books = 23) := by
sorry

end NUMINAMATH_CALUDE_book_distribution_l1466_146614


namespace NUMINAMATH_CALUDE_base_of_equation_l1466_146685

theorem base_of_equation (k x : ℝ) (h1 : (1/2)^23 * (1/81)^k = 1/x^23) (h2 : k = 11.5) : x = 18 := by
  sorry

end NUMINAMATH_CALUDE_base_of_equation_l1466_146685


namespace NUMINAMATH_CALUDE_point_position_l1466_146603

/-- An isosceles triangle with a point on its base satisfying certain conditions -/
structure IsoscelesTriangleWithPoint where
  -- The length of the base of the isosceles triangle
  a : ℝ
  -- The height of the isosceles triangle
  h : ℝ
  -- The distance from one endpoint of the base to the point on the base
  x : ℝ
  -- Condition: a > 0 (positive base length)
  a_pos : a > 0
  -- Condition: h > 0 (positive height)
  h_pos : h > 0
  -- Condition: 0 < x < a (point is on the base)
  x_on_base : 0 < x ∧ x < a
  -- Condition: BM + MA = 2h
  sum_condition : x + (2 * h - x) = 2 * h

/-- Theorem: The position of the point on the base satisfies the quadratic equation -/
theorem point_position (t : IsoscelesTriangleWithPoint) : 
  t.x = t.h + (Real.sqrt (t.a^2 - 8 * t.h^2)) / 4 ∨ 
  t.x = t.h - (Real.sqrt (t.a^2 - 8 * t.h^2)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_point_position_l1466_146603


namespace NUMINAMATH_CALUDE_final_brownies_count_l1466_146646

/-- The number of brownies in a dozen -/
def dozen : ℕ := 12

/-- The initial number of brownies made by Mother -/
def initial_brownies : ℕ := 2 * dozen

/-- The number of brownies Father ate -/
def father_ate : ℕ := 8

/-- The number of brownies Mooney ate -/
def mooney_ate : ℕ := 4

/-- The number of new brownies Mother made the next day -/
def new_brownies : ℕ := 2 * dozen

/-- Theorem stating the final number of brownies on the counter -/
theorem final_brownies_count :
  initial_brownies - father_ate - mooney_ate + new_brownies = 36 := by
  sorry

end NUMINAMATH_CALUDE_final_brownies_count_l1466_146646


namespace NUMINAMATH_CALUDE_triangle_properties_l1466_146665

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C

-- Define the theorem
theorem triangle_properties (ABC : Triangle) 
  (h1 : Real.cos (ABC.A / 2) = 2 * Real.sqrt 5 / 5)
  (h2 : ABC.b * ABC.c * Real.cos ABC.A = 15)
  (h3 : Real.tan ABC.B = 2) : 
  (1/2 * ABC.b * ABC.c * Real.sin ABC.A = 10) ∧ 
  (ABC.a = 2 * Real.sqrt 5) := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l1466_146665


namespace NUMINAMATH_CALUDE_vertical_angles_equal_l1466_146632

/-- Two lines in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- An angle formed by two lines -/
structure Angle :=
  (line1 : Line)
  (line2 : Line)

/-- Two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  l1.slope ≠ l2.slope

/-- Vertical angles are formed when two lines intersect -/
def vertical_angles (a1 a2 : Angle) : Prop :=
  ∃ (l1 l2 : Line), intersect l1 l2 ∧ 
    ((a1.line1 = l1 ∧ a1.line2 = l2 ∧ a2.line1 = l2 ∧ a2.line2 = l1) ∨
     (a1.line1 = l2 ∧ a1.line2 = l1 ∧ a2.line1 = l1 ∧ a2.line2 = l2))

/-- Angles are equal -/
def angle_equal (a1 a2 : Angle) : Prop :=
  sorry  -- Definition of angle equality

/-- Theorem: Vertical angles are equal -/
theorem vertical_angles_equal (a1 a2 : Angle) : 
  vertical_angles a1 a2 → angle_equal a1 a2 := by
  sorry

end NUMINAMATH_CALUDE_vertical_angles_equal_l1466_146632


namespace NUMINAMATH_CALUDE_valid_k_values_l1466_146686

/-- A function f: ℤ → ℤ satisfies the given property for a positive integer k -/
def satisfies_property (f : ℤ → ℤ) (k : ℕ+) : Prop :=
  ∀ (a b c : ℤ), a + b + c = 0 →
    f a + f b + f c = (f (a - b) + f (b - c) + f (c - a)) / k

/-- A function f: ℤ → ℤ is nonlinear -/
def is_nonlinear (f : ℤ → ℤ) : Prop :=
  ∃ (a b x y : ℤ), f (a + x) + f (b + y) ≠ f a + f b + f x + f y

theorem valid_k_values :
  {k : ℕ+ | ∃ (f : ℤ → ℤ), satisfies_property f k ∧ is_nonlinear f} = {1, 3, 9} := by
  sorry

end NUMINAMATH_CALUDE_valid_k_values_l1466_146686


namespace NUMINAMATH_CALUDE_tourist_journey_times_l1466_146627

-- Define the speeds of the tourists
variable (v1 v2 : ℝ)

-- Define the time (in minutes) it takes the second tourist to travel the distance the first tourist covers in 120 minutes
variable (x : ℝ)

-- Define the total journey times for each tourist
def first_tourist_time : ℝ := 120 + x + 28
def second_tourist_time : ℝ := 60 + x

-- State the theorem
theorem tourist_journey_times 
  (h1 : x * v2 = 120 * v1) -- Distance equality at meeting point
  (h2 : v2 * (x + 60) = 120 * v1 + v1 * (x + 28)) -- Total distance equality
  : first_tourist_time = 220 ∧ second_tourist_time = 132 := by
  sorry


end NUMINAMATH_CALUDE_tourist_journey_times_l1466_146627


namespace NUMINAMATH_CALUDE_min_rice_purchase_exact_min_rice_purchase_l1466_146693

/-- The minimum amount of rice Maria could purchase, given the constraints on oats and rice. -/
theorem min_rice_purchase (o r : ℝ) 
  (h1 : o ≥ 4 + r / 3)  -- Condition 1: oats ≥ 4 + 1/3 * rice
  (h2 : o ≤ 3 * r)      -- Condition 2: oats ≤ 3 * rice
  : r ≥ 3/2 := by sorry

/-- The exact minimum amount of rice Maria could purchase is 1.5 kg. -/
theorem exact_min_rice_purchase : 
  ∃ (o r : ℝ), r = 3/2 ∧ o = 4.5 ∧ o ≥ 4 + r / 3 ∧ o ≤ 3 * r := by sorry

end NUMINAMATH_CALUDE_min_rice_purchase_exact_min_rice_purchase_l1466_146693


namespace NUMINAMATH_CALUDE_imaginary_unit_cube_l1466_146673

theorem imaginary_unit_cube (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_cube_l1466_146673


namespace NUMINAMATH_CALUDE_quadratic_coeff_unequal_l1466_146678

/-- Given a quadratic equation 3x^2 + 7x + 2k = 0 with zero discriminant,
    prove that the coefficients 3, 7, and k are unequal -/
theorem quadratic_coeff_unequal (k : ℝ) :
  (7^2 - 4*3*(2*k) = 0) →
  (3 ≠ 7 ∧ 3 ≠ k ∧ 7 ≠ k) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coeff_unequal_l1466_146678


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1466_146616

theorem quadratic_root_problem (m : ℝ) : 
  ((-5)^2 + m*(-5) - 10 = 0) → (2^2 + m*2 - 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1466_146616


namespace NUMINAMATH_CALUDE_modular_inverse_72_l1466_146622

theorem modular_inverse_72 (h : (17⁻¹ : ZMod 89) = 53) : (72⁻¹ : ZMod 89) = 36 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_72_l1466_146622


namespace NUMINAMATH_CALUDE_paths_on_specific_grid_l1466_146635

/-- The number of paths on a rectangular grid from (0,0) to (m,n) moving only right or up -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The specific grid dimensions -/
def grid_width : ℕ := 7
def grid_height : ℕ := 3

theorem paths_on_specific_grid :
  grid_paths grid_width grid_height = 120 := by
  sorry

end NUMINAMATH_CALUDE_paths_on_specific_grid_l1466_146635


namespace NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l1466_146641

/-- A geometric sequence of positive integers -/
structure GeometricSequence where
  terms : ℕ → ℕ
  first_term : terms 1 = 6
  is_geometric : ∀ n : ℕ, n > 0 → ∃ r : ℚ, terms (n + 1) = (terms n : ℚ) * r

/-- The theorem stating the third term of the specific geometric sequence -/
theorem third_term_of_geometric_sequence
  (seq : GeometricSequence)
  (h_fourth : seq.terms 4 = 384) :
  seq.terms 3 = 96 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l1466_146641


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l1466_146683

/-- The area of a square with adjacent vertices at (0,3) and (3,-4) is 58 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (0, 3)
  let p2 : ℝ × ℝ := (3, -4)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 58 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l1466_146683


namespace NUMINAMATH_CALUDE_sixteen_cows_days_to_finish_l1466_146606

/-- Represents the grass consumption scenario in a pasture -/
structure GrassConsumption where
  /-- Daily grass growth rate -/
  daily_growth : ℝ
  /-- Amount of grass each cow eats per day -/
  cow_consumption : ℝ
  /-- Original amount of grass in the pasture -/
  initial_grass : ℝ

/-- Theorem stating that 16 cows will take 18 days to finish the grass -/
theorem sixteen_cows_days_to_finish (gc : GrassConsumption) : 
  gc.initial_grass + 6 * gc.daily_growth = 24 * 6 * gc.cow_consumption →
  gc.initial_grass + 8 * gc.daily_growth = 21 * 8 * gc.cow_consumption →
  gc.initial_grass + 18 * gc.daily_growth = 16 * 18 * gc.cow_consumption := by
  sorry

#check sixteen_cows_days_to_finish

end NUMINAMATH_CALUDE_sixteen_cows_days_to_finish_l1466_146606


namespace NUMINAMATH_CALUDE_memory_sequence_increment_prime_l1466_146644

def memory_sequence : ℕ → ℕ
  | 0 => 6
  | n + 1 => memory_sequence n + Nat.gcd (memory_sequence n) (n + 1)

theorem memory_sequence_increment_prime (n : ℕ) :
  n > 0 → (memory_sequence n - memory_sequence (n - 1) = 1) ∨
          Nat.Prime (memory_sequence n - memory_sequence (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_memory_sequence_increment_prime_l1466_146644


namespace NUMINAMATH_CALUDE_line_properties_l1466_146617

/-- Definition of line l₁ -/
def l₁ (m : ℝ) (x y : ℝ) : Prop := x + 2 * m * y + 6 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * m * y + 2 * m = 0

/-- Two lines are parallel if their slopes are equal -/
def parallel (m : ℝ) : Prop := (-1 / (2 * m)) = (-(m - 2) / (3 * m))

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m : ℝ) : Prop := (-1 / (2 * m)) * (-(m - 2) / (3 * m)) = -1

theorem line_properties (m : ℝ) :
  (parallel m ↔ m = 0 ∨ m = 7/2) ∧
  (perpendicular m ↔ m = -1/2 ∨ m = 2/3) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l1466_146617


namespace NUMINAMATH_CALUDE_percent_relation_l1466_146677

theorem percent_relation (a b c : ℝ) (h1 : c = 0.30 * a) (h2 : b = 1.20 * a) : 
  c = 0.25 * b := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l1466_146677


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1466_146615

/-- The speed of a train given another train passing in the opposite direction -/
theorem train_speed_calculation (passing_time : ℝ) (goods_train_length : ℝ) (goods_train_speed : ℝ) :
  passing_time = 9 →
  goods_train_length = 280 →
  goods_train_speed = 52 →
  ∃ (man_train_speed : ℝ), abs (man_train_speed - 60.16) < 0.01 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l1466_146615


namespace NUMINAMATH_CALUDE_new_regression_line_after_point_removal_l1466_146633

/-- Represents a sample point -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- Represents a regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the regression line from a list of sample points -/
def calculateRegressionLine (sample : List SamplePoint) : RegressionLine :=
  sorry

/-- Theorem stating the properties of the new regression line after removing two specific points -/
theorem new_regression_line_after_point_removal 
  (sample : List SamplePoint)
  (initial_line : RegressionLine)
  (mean_x : ℝ) :
  sample.length = 10 →
  initial_line = { slope := 2, intercept := -0.4 } →
  mean_x = 2 →
  let new_sample := sample.filter (λ p => ¬(p.x = -3 ∧ p.y = 1) ∧ ¬(p.x = 3 ∧ p.y = -1))
  let new_line := calculateRegressionLine new_sample
  new_line.slope = 3 →
  new_line = { slope := 3, intercept := -3 } :=
sorry

end NUMINAMATH_CALUDE_new_regression_line_after_point_removal_l1466_146633


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l1466_146696

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 4600

/-- The scientific notation representation of the number -/
def scientificForm : ScientificNotation :=
  { coefficient := 4.6
    exponent := 3
    property := by sorry }

/-- Theorem stating that the scientific notation form is correct -/
theorem scientific_notation_correct :
  (scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent) = number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l1466_146696


namespace NUMINAMATH_CALUDE_remainder_1234567891_div_98_l1466_146658

theorem remainder_1234567891_div_98 : 1234567891 % 98 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567891_div_98_l1466_146658


namespace NUMINAMATH_CALUDE_magic_money_box_l1466_146649

def tripleEachDay (initial : ℕ) (days : ℕ) : ℕ :=
  initial * (3 ^ days)

theorem magic_money_box (initial : ℕ) (days : ℕ) 
  (h1 : initial = 5) (h2 : days = 7) : 
  tripleEachDay initial days = 10935 := by
  sorry

end NUMINAMATH_CALUDE_magic_money_box_l1466_146649


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1466_146668

theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 7) (h2 : DF = 8) (h3 : EF = 9) :
  let s := (DE + DF + EF) / 2
  let A := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  A / s = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1466_146668


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l1466_146679

theorem mod_equivalence_problem : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 8173 [ZMOD 15] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l1466_146679


namespace NUMINAMATH_CALUDE_equation_solution_range_l1466_146690

theorem equation_solution_range (a : ℝ) (m : ℝ) :
  a > 0 ∧ a ≠ 1 →
  (∃ x : ℝ, a^(2*x) + (1 + 1/m)*a^x + 1 = 0) ↔
  -1/3 ≤ m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l1466_146690


namespace NUMINAMATH_CALUDE_janette_beef_jerky_dinner_l1466_146660

/-- Calculates the number of beef jerky pieces eaten for dinner each day during a camping trip. -/
def beef_jerky_for_dinner (days : ℕ) (total_pieces : ℕ) (breakfast_pieces : ℕ) (lunch_pieces : ℕ) (pieces_after_sharing : ℕ) : ℕ :=
  let pieces_before_sharing := 2 * pieces_after_sharing
  let pieces_eaten := total_pieces - pieces_before_sharing
  let pieces_for_breakfast_and_lunch := (breakfast_pieces + lunch_pieces) * days
  (pieces_eaten - pieces_for_breakfast_and_lunch) / days

/-- Theorem stating that Janette ate 2 pieces of beef jerky for dinner each day during her camping trip. -/
theorem janette_beef_jerky_dinner :
  beef_jerky_for_dinner 5 40 1 1 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_janette_beef_jerky_dinner_l1466_146660


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1466_146651

-- Define a circle in 2D plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if two circles intersect
def intersect (c1 c2 : Circle) : Prop := sorry

-- Define a function to get intersection points of two circles
def intersection_points (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

-- Define a function to check if points are concyclic or collinear
def concyclic_or_collinear (points : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem circle_intersection_theorem (S1 S2 S3 S4 : Circle) :
  intersect S1 S2 ∧ intersect S1 S4 ∧ intersect S3 S2 ∧ intersect S3 S4 →
  concyclic_or_collinear (intersection_points S1 S2 ∪ intersection_points S3 S4) →
  concyclic_or_collinear (intersection_points S1 S4 ∪ intersection_points S2 S3) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1466_146651


namespace NUMINAMATH_CALUDE_exists_coprime_linear_combination_divisible_l1466_146655

theorem exists_coprime_linear_combination_divisible (a b p : ℤ) :
  ∃ k l : ℤ, (Nat.gcd k.natAbs l.natAbs = 1) ∧ (∃ m : ℤ, a * k + b * l = p * m) := by
  sorry

end NUMINAMATH_CALUDE_exists_coprime_linear_combination_divisible_l1466_146655


namespace NUMINAMATH_CALUDE_intersection_forms_right_triangle_l1466_146620

/-- An ellipse with equation x²/m + y² = 1, where m > 1 -/
structure Ellipse where
  m : ℝ
  h_m : m > 1

/-- A hyperbola with equation x²/n - y² = 1, where n > 0 -/
structure Hyperbola where
  n : ℝ
  h_n : n > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents two curves (ellipse and hyperbola) with shared foci -/
structure SharedFociCurves where
  e : Ellipse
  h : Hyperbola
  f₁ : Point  -- First focus
  f₂ : Point  -- Second focus

/-- A point P that lies on both the ellipse and the hyperbola -/
structure IntersectionPoint (curves : SharedFociCurves) where
  p : Point
  on_ellipse : p.x ^ 2 / curves.e.m + p.y ^ 2 = 1
  on_hyperbola : p.x ^ 2 / curves.h.n - p.y ^ 2 = 1

/-- The main theorem: Triangle F₁PF₂ is always a right triangle -/
theorem intersection_forms_right_triangle (curves : SharedFociCurves) 
  (p : IntersectionPoint curves) : 
  (p.p.x - curves.f₁.x) ^ 2 + (p.p.y - curves.f₁.y) ^ 2 +
  (p.p.x - curves.f₂.x) ^ 2 + (p.p.y - curves.f₂.y) ^ 2 =
  (curves.f₁.x - curves.f₂.x) ^ 2 + (curves.f₁.y - curves.f₂.y) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_forms_right_triangle_l1466_146620


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1466_146607

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 3 = 1

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = (Real.sqrt 5 / 2) * x

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∀ x y : ℝ,
  (∃ f₁ f₂ : ℝ × ℝ, (∀ x y : ℝ, ellipse x y ↔ (x - f₁.1)^2 + y^2 = (x - f₂.1)^2 + y^2) ∧
                    (∀ x y : ℝ, hyperbola_C x y ↔ |(x - f₁.1)^2 + y^2| - |(x - f₂.1)^2 + y^2| = 2 * Real.sqrt 4)) →
  (∃ x y : ℝ, asymptote x y) →
  hyperbola_C x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1466_146607


namespace NUMINAMATH_CALUDE_min_distance_parallel_lines_l1466_146611

/-- The minimum distance between two parallel lines -/
theorem min_distance_parallel_lines :
  let line1 : ℝ → ℝ → Prop := λ x y => 3 * x + 4 * y - 6 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => 6 * x + 8 * y + 3 = 0
  ∀ (P Q : ℝ × ℝ), line1 P.1 P.2 → line2 Q.1 Q.2 →
  (∀ (P' Q' : ℝ × ℝ), line1 P'.1 P'.2 → line2 Q'.1 Q'.2 →
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2)) →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 3/2 :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_parallel_lines_l1466_146611


namespace NUMINAMATH_CALUDE_complex_square_l1466_146636

theorem complex_square (z : ℂ) (h : z = 5 + 6 * Complex.I) : z^2 = -11 + 60 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l1466_146636


namespace NUMINAMATH_CALUDE_inscribed_triangle_sum_l1466_146600

/-- An equilateral triangle inscribed in an ellipse -/
structure InscribedTriangle where
  /-- The x-coordinate of a vertex of the triangle -/
  x : ℝ
  /-- The y-coordinate of a vertex of the triangle -/
  y : ℝ
  /-- The condition that the vertex lies on the ellipse -/
  on_ellipse : x^2 + 9*y^2 = 9
  /-- The condition that one vertex is at (0, 1) -/
  vertex_at_origin : x = 0 ∧ y = 1
  /-- The condition that one altitude is aligned with the y-axis -/
  altitude_aligned : True  -- This condition is implicitly satisfied by the symmetry of the problem

/-- The theorem stating the result about the inscribed equilateral triangle -/
theorem inscribed_triangle_sum (t : InscribedTriangle) 
  (p q : ℕ) (h_coprime : Nat.Coprime p q) 
  (h_side_length : (12 * Real.sqrt 3 / 13)^2 = p / q) : 
  p + q = 601 := by sorry

end NUMINAMATH_CALUDE_inscribed_triangle_sum_l1466_146600


namespace NUMINAMATH_CALUDE_expression_simplification_l1466_146675

theorem expression_simplification (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  ((a^2 + 4*a + 4) / (a^2 - 4) - (a + 3) / (a - 2)) / ((a + 2) / (a - 2)) = -1 / (a + 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1466_146675


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l1466_146672

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | 2*x - 4 ≥ x - 2}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l1466_146672


namespace NUMINAMATH_CALUDE_yarn_theorem_l1466_146694

def yarn_problem (B1 : ℝ) : Prop :=
  let B2 := 2 * B1
  let B3 := 3 * B1
  let B4 := 2 * B2
  let B5 := B3 + B4
  B3 = 27 ∧ B2 = 18

theorem yarn_theorem : ∃ B1 : ℝ, yarn_problem B1 := by
  sorry

end NUMINAMATH_CALUDE_yarn_theorem_l1466_146694


namespace NUMINAMATH_CALUDE_triangle_radius_inequality_l1466_146650

/-- A structure representing a triangle with its circumradius and inradius -/
structure Triangle where
  R : ℝ  -- circumradius
  r : ℝ  -- inradius

/-- The theorem stating the relationship between circumradius and inradius of a triangle -/
theorem triangle_radius_inequality (t : Triangle) : 
  t.R ≥ 2 * t.r ∧ 
  (t.R = 2 * t.r ↔ ∃ (s : ℝ), s > 0 ∧ t.R = s * Real.sqrt 3 / 3 ∧ t.r = s / 3) ∧
  ∀ (R r : ℝ), R ≥ 2 * r → R > 0 → r > 0 → ∃ (t : Triangle), t.R = R ∧ t.r = r :=
sorry


end NUMINAMATH_CALUDE_triangle_radius_inequality_l1466_146650


namespace NUMINAMATH_CALUDE_solve_potato_problem_l1466_146662

def potato_problem (total_potatoes : ℕ) (time_per_potato : ℕ) (remaining_time : ℕ) : Prop :=
  let remaining_potatoes := remaining_time / time_per_potato
  let cooked_potatoes := total_potatoes - remaining_potatoes
  cooked_potatoes = total_potatoes - (remaining_time / time_per_potato)

theorem solve_potato_problem :
  potato_problem 12 6 36 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_potato_problem_l1466_146662


namespace NUMINAMATH_CALUDE_water_depth_for_specific_cylinder_l1466_146624

/-- Represents a cylindrical tower partially submerged in water -/
structure SubmergedCylinder where
  height : ℝ
  radius : ℝ
  aboveWaterRatio : ℝ

/-- Calculates the depth of water at the base of a partially submerged cylinder -/
def waterDepth (c : SubmergedCylinder) : ℝ :=
  c.height * (1 - c.aboveWaterRatio)

/-- Theorem stating the water depth for a specific cylinder -/
theorem water_depth_for_specific_cylinder :
  let c : SubmergedCylinder := {
    height := 1200,
    radius := 100,
    aboveWaterRatio := 1/3
  }
  waterDepth c = 400 := by sorry

end NUMINAMATH_CALUDE_water_depth_for_specific_cylinder_l1466_146624


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_112_l1466_146687

theorem smallest_four_digit_multiple_of_112 : ∃ n : ℕ, 
  (n = 1008) ∧ 
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 112 = 0) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 112 = 0 → m ≥ n) :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_112_l1466_146687


namespace NUMINAMATH_CALUDE_books_read_l1466_146657

theorem books_read (total : ℕ) (unread : ℕ) (h1 : total = 13) (h2 : unread = 4) :
  total - unread = 9 := by
  sorry

end NUMINAMATH_CALUDE_books_read_l1466_146657


namespace NUMINAMATH_CALUDE_function_composition_l1466_146610

theorem function_composition (f : ℝ → ℝ) :
  (∀ x, f (x - 1) = x^2 - 3*x) → (∀ x, f x = x^2 - x - 2) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l1466_146610


namespace NUMINAMATH_CALUDE_no_four_integers_with_odd_sum_and_product_l1466_146676

theorem no_four_integers_with_odd_sum_and_product : ¬∃ (a b c d : ℤ), 
  Odd (a + b + c + d) ∧ Odd (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_no_four_integers_with_odd_sum_and_product_l1466_146676


namespace NUMINAMATH_CALUDE_trapezoid_lines_parallel_or_concurrent_l1466_146688

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the Euclidean plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Trapezoid ABCD with diagonals intersecting at E -/
structure Trapezoid :=
  (A B C D E : Point)
  (AB_parallel_CD : Line)
  (AC_diagonal : Line)
  (BD_diagonal : Line)
  (E_on_AC_and_BD : Prop)

/-- P is the foot of altitude from A to BC -/
def altitude_foot_P (trap : Trapezoid) : Point :=
  sorry

/-- Q is the foot of altitude from B to AD -/
def altitude_foot_Q (trap : Trapezoid) : Point :=
  sorry

/-- F is the intersection of circumcircles of CEQ and DEP -/
def point_F (trap : Trapezoid) (P Q : Point) : Point :=
  sorry

/-- Line through two points -/
def line_through (P Q : Point) : Line :=
  sorry

/-- Check if three lines are parallel or concurrent -/
def parallel_or_concurrent (l₁ l₂ l₃ : Line) : Prop :=
  sorry

theorem trapezoid_lines_parallel_or_concurrent (trap : Trapezoid) :
  let P := altitude_foot_P trap
  let Q := altitude_foot_Q trap
  let F := point_F trap P Q
  let AP := line_through trap.A P
  let BQ := line_through trap.B Q
  let EF := line_through trap.E F
  parallel_or_concurrent AP BQ EF :=
sorry

end NUMINAMATH_CALUDE_trapezoid_lines_parallel_or_concurrent_l1466_146688


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_3_union_A_B_equals_A_iff_m_in_range_l1466_146634

-- Define sets A and B
def A : Set ℝ := {x | -3 < 2*x + 1 ∧ 2*x + 1 < 11}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2*m + 1}

-- Theorem 1
theorem intersection_A_complement_B_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | -2 < x ∧ x < 2} := by sorry

-- Theorem 2
theorem union_A_B_equals_A_iff_m_in_range (m : ℝ) :
  A ∪ B m = A ↔ m < -2 ∨ (-1 < m ∧ m < 2) := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_3_union_A_B_equals_A_iff_m_in_range_l1466_146634


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_third_l1466_146698

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1/3

/-- The reciprocal of the common fraction form of 0.333... is 3 --/
theorem reciprocal_of_repeating_third : (repeating_third⁻¹ : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_third_l1466_146698


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l1466_146623

theorem factorization_of_quadratic (x : ℝ) : 4 * x^2 - 2 * x = 2 * x * (2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l1466_146623


namespace NUMINAMATH_CALUDE_fraction_relation_l1466_146670

theorem fraction_relation (a b c d : ℚ) 
  (h1 : a / b = 8)
  (h2 : c / b = 5)
  (h3 : c / d = 1 / 3) :
  d / a = 15 / 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l1466_146670


namespace NUMINAMATH_CALUDE_priyanka_value_l1466_146684

/-- A system representing the values of individuals --/
structure ValueSystem where
  Neha : ℕ
  Sonali : ℕ
  Priyanka : ℕ
  Sadaf : ℕ
  Tanu : ℕ

/-- The theorem stating Priyanka's value in the given system --/
theorem priyanka_value (sys : ValueSystem) 
  (h1 : sys.Sonali = 15)
  (h2 : sys.Priyanka = 15)
  (h3 : sys.Sadaf = sys.Neha)
  (h4 : sys.Tanu = sys.Neha) :
  sys.Priyanka = 15 := by
    sorry

end NUMINAMATH_CALUDE_priyanka_value_l1466_146684


namespace NUMINAMATH_CALUDE_factorial_sum_unit_digit_l1466_146689

theorem factorial_sum_unit_digit : (Nat.factorial 25 + Nat.factorial 17 - Nat.factorial 18) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_unit_digit_l1466_146689


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_lower_bound_l1466_146695

theorem sum_of_reciprocals_lower_bound (a₁ a₂ a₃ : ℝ) 
  (pos₁ : a₁ > 0) (pos₂ : a₂ > 0) (pos₃ : a₃ > 0) 
  (sum_eq_one : a₁ + a₂ + a₃ = 1) : 
  1 / a₁ + 1 / a₂ + 1 / a₃ ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_lower_bound_l1466_146695


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1466_146680

theorem simplify_fraction_product : 15 * (18 / 5) * (-42 / 45) = -50.4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1466_146680


namespace NUMINAMATH_CALUDE_friend_symmetry_iff_d_mod_7_eq_2_l1466_146621

def isFriend (d : ℕ) (M N : ℕ) : Prop :=
  ∀ k, k < d → (M + 10^k * ((N / 10^k) % 10 - (M / 10^k) % 10)) % 7 = 0

theorem friend_symmetry_iff_d_mod_7_eq_2 (d : ℕ) :
  (∀ M N : ℕ, M < 10^d → N < 10^d → (isFriend d M N ↔ isFriend d N M)) ↔ d % 7 = 2 :=
sorry

end NUMINAMATH_CALUDE_friend_symmetry_iff_d_mod_7_eq_2_l1466_146621


namespace NUMINAMATH_CALUDE_species_decline_year_l1466_146642

def species_decrease_rate : ℝ := 0.3
def threshold : ℝ := 0.05
def base_year : ℕ := 2010

def species_count (n : ℕ) : ℝ := (1 - species_decrease_rate) ^ n

theorem species_decline_year :
  ∃ k : ℕ, (species_count k < threshold) ∧ (∀ m : ℕ, m < k → species_count m ≥ threshold) ∧ (base_year + k = 2019) :=
sorry

end NUMINAMATH_CALUDE_species_decline_year_l1466_146642


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1466_146628

theorem solution_set_inequality (x : ℝ) :
  (x + 2) * (x - 1) > 0 ↔ x < -2 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1466_146628


namespace NUMINAMATH_CALUDE_xy_squared_minus_x_squared_y_l1466_146638

theorem xy_squared_minus_x_squared_y (x y : ℝ) 
  (h1 : x - y = 1/2) 
  (h2 : x * y = 4/3) : 
  x * y^2 - x^2 * y = -2/3 := by
sorry

end NUMINAMATH_CALUDE_xy_squared_minus_x_squared_y_l1466_146638


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l1466_146645

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l1466_146645


namespace NUMINAMATH_CALUDE_T_properties_l1466_146629

-- Define the set T
def T : Set ℤ := {x | ∃ n : ℤ, x = n^2 + (n+2)^2 + (n+4)^2}

-- Statement to prove
theorem T_properties :
  (∀ x ∈ T, ¬(4 ∣ x)) ∧ (∃ x ∈ T, 13 ∣ x) := by sorry

end NUMINAMATH_CALUDE_T_properties_l1466_146629


namespace NUMINAMATH_CALUDE_exists_linear_approximation_l1466_146640

/-- Cyclic distance in Fp -/
def cyclic_distance (p : ℕ) (x : Fin p) : ℕ :=
  min x.val (p - x.val)

/-- Almost additive function property -/
def almost_additive (p : ℕ) (f : Fin p → Fin p) : Prop :=
  ∀ x y : Fin p, cyclic_distance p (f (x + y) - f x - f y) < 100

/-- Main theorem -/
theorem exists_linear_approximation
  (p : ℕ) (hp : Nat.Prime p) (f : Fin p → Fin p) (hf : almost_additive p f) :
  ∃ m : Fin p, ∀ x : Fin p, cyclic_distance p (f x - m * x) < 1000 :=
sorry

end NUMINAMATH_CALUDE_exists_linear_approximation_l1466_146640


namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangent_l1466_146630

theorem parabola_hyperbola_tangent (m : ℝ) : 
  (∀ x y : ℝ, y = x^2 + 4 ∧ y^2 - 4*m*x^2 = 4 → 
    (∃! u : ℝ, u^2 + (8 - 4*m)*u + 12 = 0)) →
  m = 2 + Real.sqrt 3 ∨ m = 2 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangent_l1466_146630


namespace NUMINAMATH_CALUDE_custom_mult_solution_l1466_146626

/-- Custom multiplication operation -/
def custom_mult (a b : ℚ) : ℚ := 3 * a - 2 * b^2

/-- Theorem stating that if a * 4 = -7 using the custom multiplication, then a = 25/3 -/
theorem custom_mult_solution :
  ∀ a : ℚ, custom_mult a 4 = -7 → a = 25/3 := by
sorry

end NUMINAMATH_CALUDE_custom_mult_solution_l1466_146626


namespace NUMINAMATH_CALUDE_homework_duration_equation_l1466_146601

-- Define the variables
variable (a b x : ℝ)

-- Define the conditions
variable (h1 : a > 0)  -- Initial average daily homework duration is positive
variable (h2 : b > 0)  -- Current average weekly homework duration is positive
variable (h3 : 0 < x ∧ x < 1)  -- Rate of decrease is between 0 and 1

-- Theorem statement
theorem homework_duration_equation : a * (1 - x)^2 = b := by
  sorry

end NUMINAMATH_CALUDE_homework_duration_equation_l1466_146601


namespace NUMINAMATH_CALUDE_wrapping_paper_problem_l1466_146619

theorem wrapping_paper_problem (x : ℝ) 
  (h1 : x + (3/4 * x) + (x + 3/4 * x) = 7) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_problem_l1466_146619


namespace NUMINAMATH_CALUDE_function_F_theorem_l1466_146669

theorem function_F_theorem (F : ℝ → ℝ) 
  (h_diff : Differentiable ℝ F) 
  (h_init : F 0 = -1)
  (h_deriv : ∀ x, deriv F x = Real.sin (Real.sin (Real.sin (Real.sin x))) * 
    Real.cos (Real.sin (Real.sin x)) * Real.cos (Real.sin x) * Real.cos x) :
  ∀ x, F x = -Real.cos (Real.sin (Real.sin (Real.sin x))) := by
sorry

end NUMINAMATH_CALUDE_function_F_theorem_l1466_146669


namespace NUMINAMATH_CALUDE_total_rice_weight_l1466_146618

-- Define the number of containers
def num_containers : ℕ := 4

-- Define the weight of rice in each container (in ounces)
def rice_per_container : ℚ := 25

-- Define the conversion rate from ounces to pounds
def ounces_per_pound : ℚ := 16

-- Theorem to prove
theorem total_rice_weight :
  (num_containers : ℚ) * rice_per_container / ounces_per_pound = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_total_rice_weight_l1466_146618


namespace NUMINAMATH_CALUDE_convention_handshakes_l1466_146681

theorem convention_handshakes (num_companies num_representatives_per_company : ℕ) :
  num_companies = 4 →
  num_representatives_per_company = 4 →
  let total_people := num_companies * num_representatives_per_company
  let handshakes_per_person := total_people - num_representatives_per_company
  (total_people * handshakes_per_person) / 2 = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_l1466_146681


namespace NUMINAMATH_CALUDE_fermat_little_theorem_l1466_146697

theorem fermat_little_theorem (N p : ℕ) (hp : Prime p) (hN : ¬ p ∣ N) :
  p ∣ (N^(p - 1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_fermat_little_theorem_l1466_146697


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1466_146664

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x - k + 1 = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y - k + 1 = 0 → y = x) → 
  k = 0 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1466_146664
