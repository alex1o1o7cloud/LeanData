import Mathlib

namespace NUMINAMATH_CALUDE_temperature_range_l2669_266948

/-- Given the highest and lowest temperatures on a certain day in Chengdu,
    prove that the temperature range is between these two values. -/
theorem temperature_range (highest lowest t : ℝ) 
  (h_highest : highest = 29)
  (h_lowest : lowest = 21)
  (h_range : lowest ≤ t ∧ t ≤ highest) : 
  21 ≤ t ∧ t ≤ 29 := by sorry

end NUMINAMATH_CALUDE_temperature_range_l2669_266948


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2669_266980

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 7 * b = 3 * c)  -- Seven bowling balls weigh the same as three canoes
  (h2 : 2 * c = 56)     -- Two canoes weigh 56 pounds
  : b = 12 :=           -- One bowling ball weighs 12 pounds
by
  sorry

#check bowling_ball_weight

end NUMINAMATH_CALUDE_bowling_ball_weight_l2669_266980


namespace NUMINAMATH_CALUDE_digit_distribution_proof_l2669_266954

theorem digit_distribution_proof (n : ℕ) 
  (h1 : n / 2 = n * (1 / 2 : ℚ))  -- 1/2 of all digits are 1
  (h2 : n / 5 = n * (1 / 5 : ℚ))  -- proportion of 2 and 5 are 1/5 each
  (h3 : n / 10 = n * (1 / 10 : ℚ))  -- proportion of other digits is 1/10
  (h4 : (1 / 2 : ℚ) + (1 / 5 : ℚ) + (1 / 5 : ℚ) + (1 / 10 : ℚ) = 1)  -- sum of all proportions is 1
  : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_digit_distribution_proof_l2669_266954


namespace NUMINAMATH_CALUDE_particular_number_proof_l2669_266999

theorem particular_number_proof : ∃! x : ℚ, ((x + 2 - 6) * 3) / 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_proof_l2669_266999


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l2669_266998

/-- Proves that the initial cost price of a bicycle is 112.5 given specific profit margins and final selling price -/
theorem bicycle_cost_price 
  (profit_a profit_b final_price : ℝ) 
  (h1 : profit_a = 0.6) 
  (h2 : profit_b = 0.25) 
  (h3 : final_price = 225) : 
  ∃ (initial_price : ℝ), 
    initial_price * (1 + profit_a) * (1 + profit_b) = final_price ∧ 
    initial_price = 112.5 := by
  sorry

#check bicycle_cost_price

end NUMINAMATH_CALUDE_bicycle_cost_price_l2669_266998


namespace NUMINAMATH_CALUDE_sort_table_in_99_moves_l2669_266919

/-- Represents a 10x10 table of distinct integers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Predicate to check if all numbers in the table are distinct -/
def all_distinct (t : Table) : Prop :=
  ∀ i j i' j', t i j = t i' j' → i = i' ∧ j = j'

/-- Predicate to check if the table is sorted in ascending order -/
def is_sorted (t : Table) : Prop :=
  (∀ i j j', j < j' → t i j < t i j') ∧
  (∀ i i' j, i < i' → t i j < t i' j)

/-- Represents a rectangular subset of the table -/
structure Rectangle where
  top_left : Fin 10 × Fin 10
  bottom_right : Fin 10 × Fin 10

/-- Represents a move (180° rotation of a rectangular subset) -/
def Move := Rectangle

/-- Applies a move to the table -/
def apply_move (t : Table) (m : Move) : Table :=
  sorry

/-- Theorem: It's always possible to sort the table in 99 or fewer moves -/
theorem sort_table_in_99_moves (t : Table) (h : all_distinct t) :
  ∃ (moves : List Move), moves.length ≤ 99 ∧ is_sorted (moves.foldl apply_move t) :=
  sorry

end NUMINAMATH_CALUDE_sort_table_in_99_moves_l2669_266919


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_l2669_266993

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

/-- Theorem: The simple interest rate for the given conditions is 1.25% -/
theorem simple_interest_rate_problem :
  let principal : ℚ := 750
  let amount : ℚ := 900
  let time : ℕ := 16
  simple_interest_rate principal amount time = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_l2669_266993


namespace NUMINAMATH_CALUDE_cube_cutting_problem_l2669_266927

theorem cube_cutting_problem :
  ∃! (n : ℕ), ∃ (s : ℕ), s < n ∧ n^3 - s^3 = 152 := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_problem_l2669_266927


namespace NUMINAMATH_CALUDE_inequality_proof_l2669_266981

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a + 1 / b ≥ 4 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2669_266981


namespace NUMINAMATH_CALUDE_union_of_overlapping_intervals_l2669_266950

open Set

theorem union_of_overlapping_intervals :
  let A : Set ℝ := {x | 1 < x ∧ x < 3}
  let B : Set ℝ := {x | 2 < x ∧ x < 4}
  A ∪ B = {x | 1 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_overlapping_intervals_l2669_266950


namespace NUMINAMATH_CALUDE_union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l2669_266935

variable {U : Type} -- Universe set
variable (A B C : Set U) -- Sets A, B, C in the universe U

-- Commutativity
theorem union_comm : A ∪ B = B ∪ A := by sorry
theorem inter_comm : A ∩ B = B ∩ A := by sorry

-- Associativity
theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := by sorry
theorem inter_assoc : A ∩ (B ∩ C) = (A ∩ B) ∩ C := by sorry

-- Distributivity
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := by sorry
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) := by sorry

-- Idempotence
theorem union_idem : A ∪ A = A := by sorry
theorem inter_idem : A ∩ A = A := by sorry

-- De Morgan's Laws
theorem de_morgan_union : (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ := by sorry
theorem de_morgan_inter : (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ := by sorry

end NUMINAMATH_CALUDE_union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l2669_266935


namespace NUMINAMATH_CALUDE_N_remainder_1000_l2669_266913

/-- A function that checks if a natural number has no repeating digits -/
def noRepeatingDigits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 8 with no repeating digits -/
def N : ℕ := sorry

/-- N is a multiple of 8 -/
axiom N_multiple_of_8 : 8 ∣ N

/-- N has no repeating digits -/
axiom N_no_repeating_digits : noRepeatingDigits N

/-- N is the greatest such number -/
axiom N_greatest : ∀ m : ℕ, m > N → ¬(8 ∣ m ∧ noRepeatingDigits m)

/-- The main theorem: the remainder when N is divided by 1000 is 120 -/
theorem N_remainder_1000 : N % 1000 = 120 := by sorry

end NUMINAMATH_CALUDE_N_remainder_1000_l2669_266913


namespace NUMINAMATH_CALUDE_binary_multiplication_subtraction_l2669_266922

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

def a : List Bool := [true, false, true, true, false, true, true]
def b : List Bool := [true, false, true, true, true]
def c : List Bool := [false, true, false, true, false, true]
def result : List Bool := [true, false, false, false, false, true, false, false, false, false, true]

theorem binary_multiplication_subtraction :
  nat_to_binary (binary_to_nat a * binary_to_nat b - binary_to_nat c) = result :=
sorry

end NUMINAMATH_CALUDE_binary_multiplication_subtraction_l2669_266922


namespace NUMINAMATH_CALUDE_proportional_function_k_l2669_266953

theorem proportional_function_k (k : ℝ) (h1 : k ≠ 0) (h2 : -5 = k * 3) : k = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_k_l2669_266953


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2669_266984

theorem inequality_system_solution (a : ℝ) :
  (∃ x : ℝ, (1 + x > a) ∧ (2 * x - 4 ≤ 0)) ↔ (a < 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2669_266984


namespace NUMINAMATH_CALUDE_ten_person_handshake_count_l2669_266903

/-- Represents a group of people with distinct heights -/
structure HeightGroup where
  n : ℕ
  heights : Fin n → ℕ
  distinct_heights : ∀ i j, i ≠ j → heights i ≠ heights j

/-- The number of handshakes in a height group -/
def handshake_count (group : HeightGroup) : ℕ :=
  (group.n * (group.n - 1)) / 2

/-- Theorem: In a group of 10 people with distinct heights, where each person
    only shakes hands with those taller than themselves, the total number of
    handshakes is 45. -/
theorem ten_person_handshake_count :
  ∀ (group : HeightGroup), group.n = 10 → handshake_count group = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_person_handshake_count_l2669_266903


namespace NUMINAMATH_CALUDE_check_problem_l2669_266908

/-- The check problem -/
theorem check_problem (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →
  (10 ≤ y ∧ y ≤ 99) →
  (100 * y + x) - (100 * x + y) = 2058 →
  (10 ≤ x ∧ x ≤ 78) ∧ y = x + 21 :=
by sorry

end NUMINAMATH_CALUDE_check_problem_l2669_266908


namespace NUMINAMATH_CALUDE_skew_and_parallel_imply_not_parallel_l2669_266949

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Two lines are skew if they are not coplanar and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

/-- Two lines are parallel if they have the same direction vector -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Definition of parallel lines
  sorry

theorem skew_and_parallel_imply_not_parallel (a b c : Line3D) :
  are_skew a b → are_parallel a c → ¬ are_parallel b c := by
  sorry

end NUMINAMATH_CALUDE_skew_and_parallel_imply_not_parallel_l2669_266949


namespace NUMINAMATH_CALUDE_parallelogram_area_32_22_l2669_266986

def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_32_22 :
  parallelogram_area 32 22 = 704 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_32_22_l2669_266986


namespace NUMINAMATH_CALUDE_left_of_kolya_l2669_266960

/-- The number of people in a physical education class line-up -/
def ClassSize : ℕ := 29

/-- The number of people to the right of Kolya -/
def RightOfKolya : ℕ := 12

/-- The number of people to the left of Sasha -/
def LeftOfSasha : ℕ := 20

/-- The number of people to the right of Sasha -/
def RightOfSasha : ℕ := 8

/-- Theorem: The number of people to the left of Kolya is 16 -/
theorem left_of_kolya : ClassSize - RightOfKolya - 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_left_of_kolya_l2669_266960


namespace NUMINAMATH_CALUDE_solution_set_l2669_266928

/-- A decreasing function f: ℝ → ℝ that passes through (0, 3) and (3, -1) -/
def f : ℝ → ℝ :=
  sorry

/-- f is a decreasing function -/
axiom f_decreasing : ∀ x y, x < y → f y < f x

/-- f(0) = 3 -/
axiom f_at_zero : f 0 = 3

/-- f(3) = -1 -/
axiom f_at_three : f 3 = -1

/-- The solution set of |f(x+1) - 1| < 2 is (-1, 2) -/
theorem solution_set : 
  {x : ℝ | |f (x + 1) - 1| < 2} = Set.Ioo (-1) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_l2669_266928


namespace NUMINAMATH_CALUDE_mersenne_prime_condition_l2669_266951

theorem mersenne_prime_condition (a n : ℕ) : 
  a > 1 → n > 1 → Nat.Prime (a^n - 1) → a = 2 ∧ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_mersenne_prime_condition_l2669_266951


namespace NUMINAMATH_CALUDE_parabola_minimum_distance_product_parabola_minimum_distance_product_achieved_l2669_266988

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the product of distances from A and B to F
def distance_product (x1 x2 : ℝ) : ℝ := (x1 + 1) * (x2 + 1)

theorem parabola_minimum_distance_product :
  ∀ k : ℝ, ∀ x1 x2 : ℝ,
  (∃ y1 y2 : ℝ, parabola x1 y1 ∧ parabola x2 y2 ∧ 
   line_through_focus k x1 y1 ∧ line_through_focus k x2 y2) →
  distance_product x1 x2 ≥ 4 :=
sorry

theorem parabola_minimum_distance_product_achieved :
  ∃ k x1 x2 : ℝ, ∃ y1 y2 : ℝ,
  parabola x1 y1 ∧ parabola x2 y2 ∧ 
  line_through_focus k x1 y1 ∧ line_through_focus k x2 y2 ∧
  distance_product x1 x2 = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_minimum_distance_product_parabola_minimum_distance_product_achieved_l2669_266988


namespace NUMINAMATH_CALUDE_alberto_clara_distance_difference_l2669_266975

/-- The difference in distance traveled between two bikers over a given time -/
def distance_difference (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 * time) - (speed2 * time)

/-- Theorem stating the difference in distance traveled between Alberto and Clara -/
theorem alberto_clara_distance_difference :
  distance_difference 16 12 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_alberto_clara_distance_difference_l2669_266975


namespace NUMINAMATH_CALUDE_stock_order_l2669_266944

def initial_investment : ℝ := 100

def apple_year1 : ℝ := 1.50
def apple_year2 : ℝ := 0.75
def banana_year1 : ℝ := 0.50
def banana_year2 : ℝ := 2.00
def cherry_year1 : ℝ := 1.30
def cherry_year2 : ℝ := 1.10
def date_year1 : ℝ := 1.00
def date_year2 : ℝ := 0.80

def final_value (year1 : ℝ) (year2 : ℝ) : ℝ :=
  initial_investment * year1 * year2

theorem stock_order :
  let A := final_value apple_year1 apple_year2
  let B := final_value banana_year1 banana_year2
  let C := final_value cherry_year1 cherry_year2
  let D := final_value date_year1 date_year2
  D < B ∧ B < A ∧ A < C := by
  sorry

end NUMINAMATH_CALUDE_stock_order_l2669_266944


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2669_266945

theorem quadratic_inequality_solution_set 
  (a b c m n : ℝ) 
  (h1 : a ≠ 0)
  (h2 : m > 0)
  (h3 : Set.Ioo m n = {x | a * x^2 + b * x + c > 0}) :
  {x | c * x^2 + b * x + a < 0} = Set.Iic (1/n) ∪ Set.Ioi (1/m) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2669_266945


namespace NUMINAMATH_CALUDE_intersection_M_N_l2669_266956

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2669_266956


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2669_266997

theorem absolute_value_equality (x : ℝ) : |x - 3| = |x + 1| → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2669_266997


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2669_266978

def A : Set ℝ := {x | x^2 - x - 6 > 0}
def B : Set ℝ := {x | x^2 - 3*x - 4 < 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2669_266978


namespace NUMINAMATH_CALUDE_no_perfect_squares_l2669_266977

def sequence_x : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 6 * sequence_x (n + 1) - sequence_x n

theorem no_perfect_squares (n : ℕ) (h : n ≥ 1) :
  ¬ ∃ m : ℤ, sequence_x n = m * m := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l2669_266977


namespace NUMINAMATH_CALUDE_sum_a_b_equals_31_l2669_266957

/-- The number of divisors of a positive integer -/
def num_divisors (x : ℕ+) : ℕ := sorry

/-- The product of the smallest ⌈n/2⌉ divisors of x -/
def f (x : ℕ+) : ℕ := sorry

/-- The least value of x such that f(x) is a multiple of x -/
def a : ℕ+ := sorry

/-- The least value of n such that there exists y with n factors and f(y) is a multiple of y -/
def b : ℕ := sorry

theorem sum_a_b_equals_31 : (a : ℕ) + b = 31 := by sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_31_l2669_266957


namespace NUMINAMATH_CALUDE_semicircle_pattern_area_l2669_266918

/-- Represents the pattern of alternating semicircles -/
structure SemicirclePattern where
  diameter : ℝ
  patternLength : ℝ

/-- Calculates the total shaded area of the semicircle pattern -/
def totalShadedArea (pattern : SemicirclePattern) : ℝ :=
  sorry

/-- Theorem stating that the total shaded area for the given pattern is 6.75π -/
theorem semicircle_pattern_area 
  (pattern : SemicirclePattern) 
  (h1 : pattern.diameter = 3)
  (h2 : pattern.patternLength = 10) : 
  totalShadedArea pattern = 6.75 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_semicircle_pattern_area_l2669_266918


namespace NUMINAMATH_CALUDE_elective_schemes_count_l2669_266991

/-- The number of elective courses available. -/
def total_courses : ℕ := 10

/-- The number of mutually exclusive courses. -/
def exclusive_courses : ℕ := 3

/-- The number of courses each student must elect. -/
def courses_to_choose : ℕ := 3

/-- Calculates the number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- 
Theorem: The number of ways to choose 3 courses out of 10, 
where 3 specific courses are mutually exclusive, is 98.
-/
theorem elective_schemes_count : 
  choose (total_courses - exclusive_courses) courses_to_choose + 
  exclusive_courses * choose (total_courses - exclusive_courses) (courses_to_choose - 1) = 98 := by
  sorry


end NUMINAMATH_CALUDE_elective_schemes_count_l2669_266991


namespace NUMINAMATH_CALUDE_eighth_group_frequency_l2669_266961

theorem eighth_group_frequency 
  (f1 f2 f3 f4 : ℝ) 
  (f5_to_7 : ℝ) 
  (h1 : f1 = 0.15)
  (h2 : f2 = 0.17)
  (h3 : f3 = 0.11)
  (h4 : f4 = 0.13)
  (h5 : f5_to_7 = 0.32)
  (h6 : ∀ f : ℝ, f ≥ 0 → f ≤ 1) -- Assumption: all frequencies are between 0 and 1
  (h7 : f1 + f2 + f3 + f4 + f5_to_7 + (1 - (f1 + f2 + f3 + f4 + f5_to_7)) = 1) -- Sum of all frequencies is 1
  : 1 - (f1 + f2 + f3 + f4 + f5_to_7) = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_eighth_group_frequency_l2669_266961


namespace NUMINAMATH_CALUDE_impossible_table_filling_l2669_266933

/-- Represents a table filled with digits -/
def Table := Matrix (Fin 5) (Fin 8) Nat

/-- Checks if a digit appears in exactly four rows of the table -/
def appearsInFourRows (t : Table) (d : Nat) : Prop :=
  (Finset.filter (fun i => ∃ j, t i j = d) Finset.univ).card = 4

/-- Checks if a digit appears in exactly four columns of the table -/
def appearsInFourCols (t : Table) (d : Nat) : Prop :=
  (Finset.filter (fun j => ∃ i, t i j = d) Finset.univ).card = 4

/-- A valid table satisfies the conditions for all digits -/
def isValidTable (t : Table) : Prop :=
  ∀ d, d ≤ 9 → appearsInFourRows t d ∧ appearsInFourCols t d

theorem impossible_table_filling : ¬ ∃ t : Table, isValidTable t := by
  sorry

end NUMINAMATH_CALUDE_impossible_table_filling_l2669_266933


namespace NUMINAMATH_CALUDE_ruby_height_l2669_266936

/-- Given the heights of various people, prove Ruby's height -/
theorem ruby_height
  (janet_height : ℕ)
  (charlene_height : ℕ)
  (pablo_height : ℕ)
  (ruby_height : ℕ)
  (h1 : janet_height = 62)
  (h2 : charlene_height = 2 * janet_height)
  (h3 : pablo_height = charlene_height + 70)
  (h4 : ruby_height = pablo_height - 2)
  : ruby_height = 192 := by
  sorry


end NUMINAMATH_CALUDE_ruby_height_l2669_266936


namespace NUMINAMATH_CALUDE_min_box_height_is_ten_l2669_266914

/-- Represents the side length of the square base of the box -/
def base_side : ℝ → ℝ := λ x => x

/-- Represents the height of the box -/
def box_height : ℝ → ℝ := λ x => x + 5

/-- Calculates the surface area of the box -/
def surface_area : ℝ → ℝ := λ x => 2 * x^2 + 4 * x * (x + 5)

/-- Theorem: The minimum height of the box satisfying the given conditions is 10 units -/
theorem min_box_height_is_ten :
  ∃ (x : ℝ), x > 0 ∧ 
             surface_area x ≥ 130 ∧ 
             box_height x = 10 ∧
             ∀ (y : ℝ), y > 0 ∧ surface_area y ≥ 130 → box_height y ≥ box_height x :=
by sorry


end NUMINAMATH_CALUDE_min_box_height_is_ten_l2669_266914


namespace NUMINAMATH_CALUDE_white_dandelions_on_saturday_l2669_266929

/-- Represents the state of dandelions on a given day -/
structure DandelionState :=
  (yellow : ℕ)
  (white : ℕ)

/-- Represents the lifecycle of a dandelion -/
def dandelionLifecycle : ℕ := 5

/-- The number of days a dandelion is yellow -/
def yellowDays : ℕ := 3

/-- The number of days a dandelion is white -/
def whiteDays : ℕ := 2

/-- Calculates the number of white dandelions on Saturday given the states on Monday and Wednesday -/
def whiteDandelionsOnSaturday (monday : DandelionState) (wednesday : DandelionState) : ℕ :=
  (wednesday.yellow + wednesday.white) - monday.yellow

theorem white_dandelions_on_saturday 
  (monday : DandelionState) 
  (wednesday : DandelionState) 
  (h1 : monday.yellow = 20)
  (h2 : monday.white = 14)
  (h3 : wednesday.yellow = 15)
  (h4 : wednesday.white = 11) :
  whiteDandelionsOnSaturday monday wednesday = 6 := by
  sorry

#check white_dandelions_on_saturday

end NUMINAMATH_CALUDE_white_dandelions_on_saturday_l2669_266929


namespace NUMINAMATH_CALUDE_trail_mix_nuts_l2669_266994

theorem trail_mix_nuts (walnuts almonds : ℚ) 
  (hw : walnuts = 0.25)
  (ha : almonds = 0.25) :
  walnuts + almonds = 0.50 := by sorry

end NUMINAMATH_CALUDE_trail_mix_nuts_l2669_266994


namespace NUMINAMATH_CALUDE_arithmetic_sum_10_terms_l2669_266932

def arithmetic_sum (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sum_10_terms : arithmetic_sum (-2) 7 10 = 295 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_10_terms_l2669_266932


namespace NUMINAMATH_CALUDE_time_per_bone_l2669_266902

def total_analysis_time : ℕ := 206
def total_bones : ℕ := 206

theorem time_per_bone : 
  (total_analysis_time : ℚ) / (total_bones : ℚ) = 1 := by sorry

end NUMINAMATH_CALUDE_time_per_bone_l2669_266902


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l2669_266968

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 2

/-- Represents the number of students selected -/
def num_selected : ℕ := 2

/-- Represents the event of selecting exactly one boy -/
def event_one_boy : Set (Fin num_boys × Fin num_girls) := sorry

/-- Represents the event of selecting exactly two boys -/
def event_two_boys : Set (Fin num_boys × Fin num_girls) := sorry

/-- The main theorem stating that the two events are mutually exclusive but not complementary -/
theorem events_mutually_exclusive_not_complementary :
  (event_one_boy ∩ event_two_boys = ∅) ∧ 
  (event_one_boy ∪ event_two_boys ≠ Set.univ) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l2669_266968


namespace NUMINAMATH_CALUDE_curve_not_hyperbola_l2669_266966

/-- The curve equation -/
def curve_equation (m : ℝ) (x y : ℝ) : Prop :=
  (m - 1) * x^2 + (3 - m) * y^2 = (m - 1) * (3 - m)

/-- Definition of a non-hyperbola based on the coefficient condition -/
def is_not_hyperbola (m : ℝ) : Prop :=
  (m - 1) * (3 - m) ≥ 0

/-- Theorem stating that for m in [1,3], the curve is not a hyperbola -/
theorem curve_not_hyperbola (m : ℝ) (h : 1 ≤ m ∧ m ≤ 3) : is_not_hyperbola m := by
  sorry

end NUMINAMATH_CALUDE_curve_not_hyperbola_l2669_266966


namespace NUMINAMATH_CALUDE_distance_scientific_notation_equivalence_l2669_266989

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The distance between two mountain peaks in meters -/
def distance : ℝ := 14000000

/-- The scientific notation representation of the distance -/
def distanceScientific : ScientificNotation := {
  coefficient := 1.4
  exponent := 7
  h1 := by sorry
}

theorem distance_scientific_notation_equivalence :
  distance = distanceScientific.coefficient * (10 : ℝ) ^ distanceScientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_distance_scientific_notation_equivalence_l2669_266989


namespace NUMINAMATH_CALUDE_forty_two_divisible_by_seven_l2669_266987

theorem forty_two_divisible_by_seven : ∃ k : ℤ, 42 = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_forty_two_divisible_by_seven_l2669_266987


namespace NUMINAMATH_CALUDE_magnitude_of_z_l2669_266937

theorem magnitude_of_z (z : ℂ) : z + Complex.I = 3 → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l2669_266937


namespace NUMINAMATH_CALUDE_student_count_l2669_266906

theorem student_count : ∃! n : ℕ, n < 50 ∧ n % 8 = 5 ∧ n % 4 = 1 ∧ n = 45 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2669_266906


namespace NUMINAMATH_CALUDE_work_completion_time_l2669_266963

/-- The number of days A takes to finish the work alone -/
def a_days : ℚ := 4

/-- The number of days B takes to finish the work alone -/
def b_days : ℚ := 10

/-- The number of days A and B work together before A leaves -/
def together_days : ℚ := 2

/-- The number of days B takes to finish the remaining work after A leaves -/
def remaining_days : ℚ := 3

theorem work_completion_time :
  (together_days * (1 / a_days + 1 / b_days)) + (remaining_days / b_days) = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2669_266963


namespace NUMINAMATH_CALUDE_closest_to_neg_sqrt_two_l2669_266905

theorem closest_to_neg_sqrt_two :
  let options : List ℝ := [-2, -1, 0, 1]
  ∀ x ∈ options, |(-1) - (-Real.sqrt 2)| ≤ |x - (-Real.sqrt 2)| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_to_neg_sqrt_two_l2669_266905


namespace NUMINAMATH_CALUDE_school_travel_time_l2669_266926

/-- Given a boy who walks at 7/6 of his usual rate and reaches school 6 minutes early,
    his usual time to reach the school is 42 minutes. -/
theorem school_travel_time (usual_rate : ℝ) (usual_time : ℝ) 
    (h1 : usual_rate > 0) 
    (h2 : usual_time > 0)
    (h3 : usual_rate * usual_time = (7/6 * usual_rate) * (usual_time - 6)) : 
  usual_time = 42 := by
sorry

end NUMINAMATH_CALUDE_school_travel_time_l2669_266926


namespace NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l2669_266910

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l2669_266910


namespace NUMINAMATH_CALUDE_equation_real_solutions_l2669_266985

theorem equation_real_solutions (a b x : ℝ) : 
  (∃ x : ℝ, Real.sqrt (2 * a + b + 2 * x) + Real.sqrt (10 * a + 9 * b - 6 * x) = 2 * Real.sqrt (2 * a + b - 2 * x)) ↔ 
  ((0 ≤ a ∧ -a ≤ b ∧ b ≤ 0 ∧ (x = Real.sqrt (a * (a + b)) ∨ x = -Real.sqrt (a * (a + b)))) ∨
   (a ≥ -8/9 * a ∧ -8/9 * a ≥ b ∧ b ≤ 0 ∧ x = -Real.sqrt (a * (a + b)))) :=
by sorry

end NUMINAMATH_CALUDE_equation_real_solutions_l2669_266985


namespace NUMINAMATH_CALUDE_small_cakes_needed_l2669_266946

/-- Prove that given the conditions, the number of small cakes needed is 630 --/
theorem small_cakes_needed (helpers : ℕ) (large_cakes_needed : ℕ) (hours : ℕ)
  (large_cakes_per_hour : ℕ) (small_cakes_per_hour : ℕ) :
  helpers = 10 →
  large_cakes_needed = 20 →
  hours = 3 →
  large_cakes_per_hour = 2 →
  small_cakes_per_hour = 35 →
  (helpers * hours * small_cakes_per_hour) - 
  (large_cakes_needed * small_cakes_per_hour * hours / large_cakes_per_hour) = 630 := by
  sorry

#check small_cakes_needed

end NUMINAMATH_CALUDE_small_cakes_needed_l2669_266946


namespace NUMINAMATH_CALUDE_four_spheres_cover_all_rays_l2669_266917

-- Define a point in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a ray in 3D space
structure Ray where
  origin : Point3D
  direction : Point3D

-- Function to check if a ray intersects a sphere
def ray_intersects_sphere (r : Ray) (s : Sphere) : Prop :=
  sorry

-- Theorem statement
theorem four_spheres_cover_all_rays :
  ∃ (s1 s2 s3 s4 : Sphere) (light_source : Point3D),
    ∀ (r : Ray),
      r.origin = light_source →
      ray_intersects_sphere r s1 ∨
      ray_intersects_sphere r s2 ∨
      ray_intersects_sphere r s3 ∨
      ray_intersects_sphere r s4 :=
sorry

end NUMINAMATH_CALUDE_four_spheres_cover_all_rays_l2669_266917


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l2669_266996

theorem smallest_dual_base_representation : ∃ (n : ℕ) (a b : ℕ), 
  a > 3 ∧ b > 3 ∧
  n = a + 3 ∧
  n = 3 * b + 1 ∧
  (∀ (m : ℕ) (c d : ℕ), c > 3 → d > 3 → m = c + 3 → m = 3 * d + 1 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l2669_266996


namespace NUMINAMATH_CALUDE_checker_arrangement_count_l2669_266907

/-- The number of ways to arrange white and black checkers on a chessboard -/
def checker_arrangements : ℕ := 
  let total_squares : ℕ := 32
  let white_checkers : ℕ := 12
  let black_checkers : ℕ := 12
  Nat.factorial total_squares / (Nat.factorial white_checkers * Nat.factorial black_checkers * Nat.factorial (total_squares - white_checkers - black_checkers))

/-- Theorem stating that the number of ways to arrange 12 white and 12 black checkers
    on 32 black squares of a chessboard is equal to (32! / (12! * 12! * 8!)) -/
theorem checker_arrangement_count : 
  checker_arrangements = Nat.factorial 32 / (Nat.factorial 12 * Nat.factorial 12 * Nat.factorial 8) :=
by sorry

end NUMINAMATH_CALUDE_checker_arrangement_count_l2669_266907


namespace NUMINAMATH_CALUDE_no_whole_number_57_times_less_l2669_266925

theorem no_whole_number_57_times_less : ¬ ∃ (N : ℕ) (n : ℕ) (a : Fin 10),
  N ≥ 10 ∧ 
  a.val ≠ 0 ∧
  N = a.val * 10^n + (N / 57) :=
sorry

end NUMINAMATH_CALUDE_no_whole_number_57_times_less_l2669_266925


namespace NUMINAMATH_CALUDE_intersection_product_l2669_266983

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 - 4*x + y^2 - 6*y + 9 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 6*y + 21 = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | circle1 p.1 p.2 ∧ circle2 p.1 p.2}

-- Theorem statement
theorem intersection_product : 
  ∀ p ∈ intersection_points, p.1 * p.2 = 12 := by sorry

end NUMINAMATH_CALUDE_intersection_product_l2669_266983


namespace NUMINAMATH_CALUDE_coffee_package_size_l2669_266921

/-- Given two types of coffee packages with the following properties:
    - The total amount of coffee is 70 ounces
    - There are two more packages of the first type than the second type
    - There are 4 packages of the second type, each containing 10 ounces
    Prove that the size of the first type of package is 5 ounces. -/
theorem coffee_package_size
  (total_coffee : ℕ)
  (package_type1 : ℕ)
  (package_type2 : ℕ)
  (size_type2 : ℕ)
  (h1 : total_coffee = 70)
  (h2 : package_type1 = package_type2 + 2)
  (h3 : package_type2 = 4)
  (h4 : size_type2 = 10)
  (h5 : package_type1 * (total_coffee - package_type2 * size_type2) / package_type1 = total_coffee - package_type2 * size_type2) :
  (total_coffee - package_type2 * size_type2) / package_type1 = 5 :=
sorry

end NUMINAMATH_CALUDE_coffee_package_size_l2669_266921


namespace NUMINAMATH_CALUDE_craig_total_commissions_l2669_266965

/-- Represents the commission structure for an appliance brand -/
structure CommissionStructure where
  refrigerator_base : ℝ
  refrigerator_rate : ℝ
  washing_machine_base : ℝ
  washing_machine_rate : ℝ
  oven_base : ℝ
  oven_rate : ℝ

/-- Represents the sales data for an appliance brand -/
structure SalesData where
  refrigerators : ℕ
  refrigerators_price : ℝ
  washing_machines : ℕ
  washing_machines_price : ℝ
  ovens : ℕ
  ovens_price : ℝ

/-- Calculates the commission for a single appliance type -/
def calculate_commission (base : ℝ) (rate : ℝ) (quantity : ℕ) (total_price : ℝ) : ℝ :=
  (base + rate * total_price) * quantity

/-- Calculates the total commission for a brand -/
def total_brand_commission (cs : CommissionStructure) (sd : SalesData) : ℝ :=
  calculate_commission cs.refrigerator_base cs.refrigerator_rate sd.refrigerators sd.refrigerators_price +
  calculate_commission cs.washing_machine_base cs.washing_machine_rate sd.washing_machines sd.washing_machines_price +
  calculate_commission cs.oven_base cs.oven_rate sd.ovens sd.ovens_price

/-- Main theorem: Craig's total commissions for the week -/
theorem craig_total_commissions :
  let brand_a_cs : CommissionStructure := {
    refrigerator_base := 75,
    refrigerator_rate := 0.08,
    washing_machine_base := 50,
    washing_machine_rate := 0.10,
    oven_base := 60,
    oven_rate := 0.12
  }
  let brand_b_cs : CommissionStructure := {
    refrigerator_base := 90,
    refrigerator_rate := 0.06,
    washing_machine_base := 40,
    washing_machine_rate := 0.14,
    oven_base := 70,
    oven_rate := 0.10
  }
  let brand_a_sales : SalesData := {
    refrigerators := 3,
    refrigerators_price := 5280,
    washing_machines := 4,
    washing_machines_price := 2140,
    ovens := 5,
    ovens_price := 4620
  }
  let brand_b_sales : SalesData := {
    refrigerators := 2,
    refrigerators_price := 3780,
    washing_machines := 3,
    washing_machines_price := 2490,
    ovens := 4,
    ovens_price := 3880
  }
  total_brand_commission brand_a_cs brand_a_sales + total_brand_commission brand_b_cs brand_b_sales = 9252.60 := by
  sorry

end NUMINAMATH_CALUDE_craig_total_commissions_l2669_266965


namespace NUMINAMATH_CALUDE_ladder_length_l2669_266930

theorem ladder_length (a b : ℝ) (ha : a = 20) (hb : b = 15) :
  Real.sqrt (a^2 + b^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ladder_length_l2669_266930


namespace NUMINAMATH_CALUDE_apples_sold_l2669_266923

/-- The amount of apples sold in a store --/
theorem apples_sold (kidney : ℕ) (golden : ℕ) (canada : ℕ) (left : ℕ) : 
  kidney + golden + canada - left = (kidney + golden + canada) - left :=
by sorry

end NUMINAMATH_CALUDE_apples_sold_l2669_266923


namespace NUMINAMATH_CALUDE_pure_imaginary_equation_l2669_266967

-- Define the complex number i
def i : ℂ := Complex.I

-- Define a pure imaginary number
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem pure_imaginary_equation (z : ℂ) (b : ℝ) 
  (h1 : isPureImaginary z) 
  (h2 : (2 - i) * z = 4 - b * i) : 
  b = -8 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_equation_l2669_266967


namespace NUMINAMATH_CALUDE_prob_at_least_two_successes_l2669_266952

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 3

/-- The probability of exactly k successes in n trials -/
def binomialProb (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of at least 2 successes in 3 trials -/
theorem prob_at_least_two_successes : 
  binomialProb 2 + binomialProb 3 = 81/125 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_successes_l2669_266952


namespace NUMINAMATH_CALUDE_charlie_missing_coins_l2669_266971

/-- Represents the fraction of coins Charlie has at different stages -/
structure CoinFraction where
  total : ℚ
  dropped : ℚ
  found : ℚ

/-- Calculates the fraction of coins still missing -/
def missing_fraction (cf : CoinFraction) : ℚ :=
  cf.total - (cf.total - cf.dropped + cf.found * cf.dropped)

/-- Theorem stating that the fraction of missing coins is 1/9 -/
theorem charlie_missing_coins :
  let cf : CoinFraction := { total := 1, dropped := 1/3, found := 2/3 }
  missing_fraction cf = 1/9 := by
  sorry

#check charlie_missing_coins

end NUMINAMATH_CALUDE_charlie_missing_coins_l2669_266971


namespace NUMINAMATH_CALUDE_triangle_property_l2669_266911

-- Define the binary operation ★
noncomputable def star (A B : ℂ) : ℂ := 
  let ζ : ℂ := Complex.exp (Complex.I * Real.pi / 3)
  ζ * (B - A) + A

-- Define the theorem
theorem triangle_property (I M O : ℂ) :
  star I (star M O) = star (star O I) M →
  -- Triangle IMO is positively oriented
  (Complex.arg ((I - O) / (M - O)) > 0) ∧
  -- Triangle IMO is isosceles with OI = OM
  Complex.abs (I - O) = Complex.abs (M - O) ∧
  -- ∠IOM = 2π/3
  Complex.arg ((I - O) / (M - O)) = 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l2669_266911


namespace NUMINAMATH_CALUDE_number_division_problem_l2669_266938

theorem number_division_problem : ∃ N : ℕ, N = (555 + 445) * (2 * (555 - 445)) + 30 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2669_266938


namespace NUMINAMATH_CALUDE_range_of_a_l2669_266920

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 9*x + a^2/x + 7
  else if x > 0 then 9*x + a^2/x - 7
  else 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) ∧  -- f is odd
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) →  -- condition for x ≥ 0
  a ≤ -8/7 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2669_266920


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2669_266970

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin of the coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- Given point P -/
def P : Point := ⟨-1, -2⟩

/-- Symmetry about the origin -/
def symmetricAboutOrigin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

/-- Theorem: The point symmetrical to P(-1, -2) about the origin has coordinates (1, 2) -/
theorem symmetric_point_coordinates :
  symmetricAboutOrigin P = Point.mk 1 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2669_266970


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l2669_266972

/-- Given two parallel lines, one passing through P(3,2m) and Q(m,2),
    and another passing through M(2,-1) and N(-3,4), prove that m = -1 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ,
  let P : ℝ × ℝ := (3, 2*m)
  let Q : ℝ × ℝ := (m, 2)
  let M : ℝ × ℝ := (2, -1)
  let N : ℝ × ℝ := (-3, 4)
  let slope_PQ := (Q.2 - P.2) / (Q.1 - P.1)
  let slope_MN := (N.2 - M.2) / (N.1 - M.1)
  slope_PQ = slope_MN →
  m = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l2669_266972


namespace NUMINAMATH_CALUDE_soccer_team_defenders_l2669_266964

theorem soccer_team_defenders (total_players : ℕ) (goalies : ℕ) (strikers : ℕ) (defenders : ℕ) :
  total_players = 40 →
  goalies = 3 →
  strikers = 7 →
  defenders + goalies + strikers + 2 * defenders = total_players →
  defenders = 10 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_defenders_l2669_266964


namespace NUMINAMATH_CALUDE_exists_n_good_not_n_plus_one_good_l2669_266909

/-- Sum of digits of a natural number -/
def S (k : ℕ) : ℕ := sorry

/-- A natural number is n-good if there exists a sequence satisfying the given condition -/
def is_n_good (a n : ℕ) : Prop :=
  ∃ (seq : Fin (n + 1) → ℕ), seq ⟨n, sorry⟩ = a ∧
    ∀ (i : Fin n), seq ⟨i.val + 1, sorry⟩ = seq i - S (seq i)

/-- For any n, there exists a number that is n-good but not (n+1)-good -/
theorem exists_n_good_not_n_plus_one_good :
  ∀ n : ℕ, ∃ a : ℕ, is_n_good a n ∧ ¬is_n_good a (n + 1) := by sorry

end NUMINAMATH_CALUDE_exists_n_good_not_n_plus_one_good_l2669_266909


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2669_266900

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), (1056 + n) % 25 = 0 ∧ 
  ∀ (m : ℕ), m < n → (1056 + m) % 25 ≠ 0 :=
by
  use 19
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2669_266900


namespace NUMINAMATH_CALUDE_shortest_ant_path_equals_slant_edge_l2669_266947

/-- Represents a regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  slantEdgeLength : ℝ
  dihedralAngle : ℝ

/-- The shortest path for an ant to visit all slant edges and return to the starting point -/
def shortestAntPath (pyramid : RegularHexagonalPyramid) : ℝ :=
  pyramid.slantEdgeLength

theorem shortest_ant_path_equals_slant_edge 
  (pyramid : RegularHexagonalPyramid) 
  (h1 : pyramid.dihedralAngle = 10) : 
  shortestAntPath pyramid = pyramid.slantEdgeLength :=
sorry

end NUMINAMATH_CALUDE_shortest_ant_path_equals_slant_edge_l2669_266947


namespace NUMINAMATH_CALUDE_count_valid_pairs_l2669_266959

def satisfies_equation (x y : ℤ) : Prop :=
  2 * x^2 - 2 * x * y + y^2 = 289

def valid_pair (p : ℤ × ℤ) : Prop :=
  satisfies_equation p.1 p.2 ∧ p.1 ≥ 0

theorem count_valid_pairs :
  ∃ (S : Finset (ℤ × ℤ)), (∀ p ∈ S, valid_pair p) ∧ S.card = 7 ∧
  ∀ p : ℤ × ℤ, valid_pair p → p ∈ S :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l2669_266959


namespace NUMINAMATH_CALUDE_recipe_total_cups_l2669_266940

/-- Calculates the total cups of ingredients in a recipe given the ratio and amount of sugar -/
def total_cups (butter_ratio : ℚ) (flour_ratio : ℚ) (sugar_ratio : ℚ) (sugar_cups : ℚ) : ℚ :=
  let total_ratio := butter_ratio + flour_ratio + sugar_ratio
  let part_size := sugar_cups / sugar_ratio
  (total_ratio * part_size)

/-- Proves that for a recipe with butter:flour:sugar ratio of 1:5:3 and 6 cups of sugar, 
    the total amount of ingredients is 18 cups -/
theorem recipe_total_cups : 
  total_cups 1 5 3 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l2669_266940


namespace NUMINAMATH_CALUDE_smallest_quotient_three_digit_number_l2669_266995

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem smallest_quotient_three_digit_number :
  ∀ n : ℕ, is_three_digit_number n →
    (n : ℚ) / (digit_sum n : ℚ) ≥ 199 / 19 :=
by sorry

end NUMINAMATH_CALUDE_smallest_quotient_three_digit_number_l2669_266995


namespace NUMINAMATH_CALUDE_min_value_expression_l2669_266958

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y)
  (h_sum : x + y + 1/x + 1/y = 2022) :
  (x + 1/y) * (x + 1/y - 2016) + (y + 1/x) * (y + 1/x - 2016) ≥ -2032188 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2669_266958


namespace NUMINAMATH_CALUDE_seventh_term_ratio_l2669_266979

/-- Two arithmetic sequences with sums S_n and T_n for the first n terms -/
def S (n : ℕ) : ℚ := sorry
def T (n : ℕ) : ℚ := sorry

/-- The ratio condition for all n -/
axiom ratio_condition (n : ℕ) : S n / T n = (7 * n + 3) / (4 * n + 30)

/-- The 7th term of each sequence -/
def a₇ : ℚ := sorry
def b₇ : ℚ := sorry

/-- The theorem to be proved -/
theorem seventh_term_ratio : a₇ / b₇ = 17 / 33 := by sorry

end NUMINAMATH_CALUDE_seventh_term_ratio_l2669_266979


namespace NUMINAMATH_CALUDE_bertha_descendants_without_daughters_l2669_266973

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The given conditions of Bertha's family -/
def bertha_family : BerthaFamily :=
  { daughters := 8,
    granddaughters := 20,
    total_descendants := 28,
    daughters_with_children := 5 }

/-- Theorem stating the number of Bertha's descendants without daughters -/
theorem bertha_descendants_without_daughters :
  bertha_family.total_descendants - bertha_family.daughters_with_children = 23 := by
  sorry

#check bertha_descendants_without_daughters

end NUMINAMATH_CALUDE_bertha_descendants_without_daughters_l2669_266973


namespace NUMINAMATH_CALUDE_parabola_equation_l2669_266904

/-- Represents a parabola with specific properties -/
structure Parabola where
  -- Equation coefficients
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  -- c is positive
  c_pos : c > 0
  -- GCD of absolute values of coefficients is 1
  gcd_one : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Nat.gcd (Int.natAbs d) (Nat.gcd (Int.natAbs e) (Int.natAbs f))))) = 1
  -- Passes through (2,6)
  passes_through : a * 2^2 + b * 2 * 6 + c * 6^2 + d * 2 + e * 6 + f = 0
  -- Focus y-coordinate is 2
  focus_y : ∃ (x : ℚ), a * x^2 + b * x * 2 + c * 2^2 + d * x + e * 2 + f = 0
  -- Axis of symmetry parallel to x-axis
  sym_axis_parallel : b = 0
  -- Vertex on y-axis
  vertex_on_y : ∃ (y : ℚ), a * 0^2 + b * 0 * y + c * y^2 + d * 0 + e * y + f = 0

/-- The parabola equation matches the given form -/
theorem parabola_equation (p : Parabola) : p.a = 0 ∧ p.b = 0 ∧ p.c = 1 ∧ p.d = -8 ∧ p.e = -4 ∧ p.f = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2669_266904


namespace NUMINAMATH_CALUDE_f_minimum_and_inequality_l2669_266990

def f (x : ℝ) := |2*x - 1| + |x - 3|

theorem f_minimum_and_inequality (x y : ℝ) :
  (∀ x, f x ≥ 5/2) ∧
  (∀ m, (∀ x y, f x > m * (|y + 1| - |y - 1|)) ↔ -5/4 < m ∧ m < 5/4) :=
by sorry

end NUMINAMATH_CALUDE_f_minimum_and_inequality_l2669_266990


namespace NUMINAMATH_CALUDE_line_angle_inclination_l2669_266943

/-- The angle of inclination of a line given its equation and a point it passes through -/
def angleOfInclination (a m : ℝ) (h1 : m ≠ 0) (h2 : a + m - 2*a = 0) : ℝ :=
  135

/-- Theorem: The angle of inclination of the line ax + my - 2a = 0 (m ≠ 0) passing through (1, 1) is 135° -/
theorem line_angle_inclination (a m : ℝ) (h1 : m ≠ 0) (h2 : a + m - 2*a = 0) :
  angleOfInclination a m h1 h2 = 135 := by
  sorry

end NUMINAMATH_CALUDE_line_angle_inclination_l2669_266943


namespace NUMINAMATH_CALUDE_first_number_in_ratio_l2669_266912

/-- Given two positive integers a and b with a ratio of 3:4 and LCM 180, prove that a = 45 -/
theorem first_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → 
  Nat.lcm a b = 180 → 
  a = 45 := by
sorry

end NUMINAMATH_CALUDE_first_number_in_ratio_l2669_266912


namespace NUMINAMATH_CALUDE_pool_width_l2669_266955

/-- Proves the width of a rectangular pool given its draining rate, dimensions, initial capacity, and time to drain. -/
theorem pool_width
  (drain_rate : ℝ)
  (length depth : ℝ)
  (initial_capacity : ℝ)
  (drain_time : ℝ)
  (h1 : drain_rate = 60)
  (h2 : length = 150)
  (h3 : depth = 10)
  (h4 : initial_capacity = 0.8)
  (h5 : drain_time = 800) :
  ∃ (width : ℝ), width = 40 ∧ 
    drain_rate * drain_time = initial_capacity * (length * width * depth) :=
by sorry

end NUMINAMATH_CALUDE_pool_width_l2669_266955


namespace NUMINAMATH_CALUDE_max_difference_second_largest_smallest_l2669_266974

theorem max_difference_second_largest_smallest (a b c d e f g h : ℕ) :
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 →
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g ∧ g < h →
  (a + b + c) / 3 = 9 →
  (a + b + c + d + e + f + g + h) / 8 = 19 →
  (f + g + h) / 3 = 29 →
  ∃ (a' b' c' d' e' f' g' h' : ℕ),
    a' ≠ 0 ∧ b' ≠ 0 ∧ c' ≠ 0 ∧ d' ≠ 0 ∧ e' ≠ 0 ∧ f' ≠ 0 ∧ g' ≠ 0 ∧ h' ≠ 0 ∧
    a' < b' ∧ b' < c' ∧ c' < d' ∧ d' < e' ∧ e' < f' ∧ f' < g' ∧ g' < h' ∧
    (a' + b' + c') / 3 = 9 ∧
    (a' + b' + c' + d' + e' + f' + g' + h') / 8 = 19 ∧
    (f' + g' + h') / 3 = 29 ∧
    g' - b' = 26 ∧
    ∀ (a'' b'' c'' d'' e'' f'' g'' h'' : ℕ),
      a'' ≠ 0 ∧ b'' ≠ 0 ∧ c'' ≠ 0 ∧ d'' ≠ 0 ∧ e'' ≠ 0 ∧ f'' ≠ 0 ∧ g'' ≠ 0 ∧ h'' ≠ 0 →
      a'' < b'' ∧ b'' < c'' ∧ c'' < d'' ∧ d'' < e'' ∧ e'' < f'' ∧ f'' < g'' ∧ g'' < h'' →
      (a'' + b'' + c'') / 3 = 9 →
      (a'' + b'' + c'' + d'' + e'' + f'' + g'' + h'') / 8 = 19 →
      (f'' + g'' + h'') / 3 = 29 →
      g'' - b'' ≤ 26 :=
by
  sorry

end NUMINAMATH_CALUDE_max_difference_second_largest_smallest_l2669_266974


namespace NUMINAMATH_CALUDE_work_completion_time_l2669_266962

theorem work_completion_time (a b c : ℝ) (h1 : b = 5) (h2 : c = 12) 
  (h3 : 1 / a + 1 / b + 1 / c = 9 / 10) : a = 60 / 37 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2669_266962


namespace NUMINAMATH_CALUDE_continuous_at_5_l2669_266942

def f (x : ℝ) : ℝ := 3 * x^2 - 2

theorem continuous_at_5 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 5| < δ → |f x - f 5| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuous_at_5_l2669_266942


namespace NUMINAMATH_CALUDE_pet_store_combinations_l2669_266934

/-- The number of puppies available in the pet store -/
def num_puppies : ℕ := 10

/-- The number of kittens available in the pet store -/
def num_kittens : ℕ := 6

/-- The number of hamsters available in the pet store -/
def num_hamsters : ℕ := 8

/-- The total number of ways Alice, Bob, and Charlie can buy pets and leave the store satisfied -/
def total_ways : ℕ := 960

/-- Theorem stating that the number of ways Alice, Bob, and Charlie can buy pets
    and leave the store satisfied is equal to total_ways -/
theorem pet_store_combinations :
  (num_puppies * num_kittens * num_hamsters) +
  (num_kittens * num_puppies * num_hamsters) = total_ways :=
by sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l2669_266934


namespace NUMINAMATH_CALUDE_crocodile_count_l2669_266992

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 20

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := 52

/-- The number of eyes each animal (frog or crocodile) has -/
def eyes_per_animal : ℕ := 2

/-- The number of crocodiles in the pond -/
def num_crocodiles : ℕ := 6

theorem crocodile_count :
  num_crocodiles * eyes_per_animal + num_frogs * eyes_per_animal = total_eyes :=
by sorry

end NUMINAMATH_CALUDE_crocodile_count_l2669_266992


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l2669_266924

theorem similar_triangle_perimeter (a b c d e : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for smaller triangle
  (d/a)^2 + (e/b)^2 = 1 →  -- Similar triangles condition
  2*c = 30 →  -- Hypotenuse of larger triangle
  d + e + 30 = 72 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l2669_266924


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2669_266941

theorem quadratic_equation_solution (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, k * x^2 + x - 3 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  (x₁ + x₂)^2 + x₁ * x₂ = 4 →
  k = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2669_266941


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2669_266916

theorem inequality_system_solution :
  let S := {x : ℝ | 2 * x - 2 > 0 ∧ 3 * (x - 1) - 7 < -2 * x}
  S = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2669_266916


namespace NUMINAMATH_CALUDE_divisor_exists_l2669_266915

def N : ℕ := 123456789101112131415161718192021222324252627282930313233343536373839404142434481

theorem divisor_exists : ∃ D : ℕ, D > 0 ∧ N % D = 36 := by
  sorry

end NUMINAMATH_CALUDE_divisor_exists_l2669_266915


namespace NUMINAMATH_CALUDE_sum_of_roots_l2669_266976

theorem sum_of_roots (a b : ℝ) : 
  (a^2 - 4*a - 2023 = 0) → (b^2 - 4*b - 2023 = 0) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2669_266976


namespace NUMINAMATH_CALUDE_cone_volume_and_surface_area_l2669_266969

/-- Represents a cone with given slant height and height --/
structure Cone where
  slant_height : ℝ
  height : ℝ

/-- Calculate the volume of a cone --/
def volume (c : Cone) : ℝ := sorry

/-- Calculate the surface area of a cone --/
def surface_area (c : Cone) : ℝ := sorry

/-- Theorem stating the volume and surface area of a specific cone --/
theorem cone_volume_and_surface_area :
  let c : Cone := { slant_height := 15, height := 9 }
  (volume c = 432 * Real.pi) ∧ (surface_area c = 324 * Real.pi) := by sorry

end NUMINAMATH_CALUDE_cone_volume_and_surface_area_l2669_266969


namespace NUMINAMATH_CALUDE_sin_2alpha_values_l2669_266939

theorem sin_2alpha_values (α : Real) 
  (h1 : 2 * (Real.tan α)^2 - 7 * Real.tan α + 3 = 0) :
  (π < α ∧ α < 5*π/4 → Real.sin (2*α) = 4/5) ∧
  (5*π/4 < α ∧ α < 3*π/2 → Real.sin (2*α) = 3/5) := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_values_l2669_266939


namespace NUMINAMATH_CALUDE_lemon_heads_package_count_l2669_266931

/-- The number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := 54

/-- The number of whole boxes Louis finished -/
def boxes_finished : ℕ := 9

/-- The number of Lemon Heads left after eating -/
def lemon_heads_left : ℕ := 0

/-- The number of Lemon Heads per package -/
def lemon_heads_per_package : ℕ := total_lemon_heads / boxes_finished

theorem lemon_heads_package_count : lemon_heads_per_package = 6 := by
  sorry

end NUMINAMATH_CALUDE_lemon_heads_package_count_l2669_266931


namespace NUMINAMATH_CALUDE_equation_solution_l2669_266901

theorem equation_solution : 
  ∃ r : ℚ, r = -7/4 ∧ 
  (r^2 - 6*r + 8) / (r^2 - 9*r + 20) = (r^2 - 3*r - 18) / (r^2 - 2*r - 24) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2669_266901


namespace NUMINAMATH_CALUDE_not_right_triangle_l2669_266982

/-- A predicate to check if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a)

/-- Theorem stating that (11, 40, 41) cannot form a right triangle --/
theorem not_right_triangle : ¬ is_right_triangle 11 40 41 := by
  sorry

#check not_right_triangle

end NUMINAMATH_CALUDE_not_right_triangle_l2669_266982
