import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_consecutive_naturals_with_lcm_168_l1017_101707

def consecutive_naturals (n : ℕ) : Fin 3 → ℕ := λ i => n + i.val

theorem sum_of_consecutive_naturals_with_lcm_168 :
  ∃ n : ℕ, (Nat.lcm (consecutive_naturals n 0) (Nat.lcm (consecutive_naturals n 1) (consecutive_naturals n 2)) = 168) ∧
  (consecutive_naturals n 0 + consecutive_naturals n 1 + consecutive_naturals n 2 = 21) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_naturals_with_lcm_168_l1017_101707


namespace NUMINAMATH_CALUDE_hyperbola_points_l1017_101787

def hyperbola (x y : ℝ) : Prop := y = -4 / x

theorem hyperbola_points :
  hyperbola (-2) 2 ∧
  ¬ hyperbola 1 4 ∧
  ¬ hyperbola (-1) (-4) ∧
  ¬ hyperbola (-2) (-2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_points_l1017_101787


namespace NUMINAMATH_CALUDE_race_result_l1017_101723

/-- The race between John and Steve --/
theorem race_result (initial_distance : ℝ) (john_speed steve_speed : ℝ) (time : ℝ) :
  initial_distance = 14 →
  john_speed = 4.2 →
  steve_speed = 3.7 →
  time = 32 →
  (john_speed * time + initial_distance) - (steve_speed * time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_race_result_l1017_101723


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1017_101754

theorem inequality_equivalence (x y z : ℝ) :
  x + 3 * y + 2 * z = 6 →
  (x^2 + 9 * y^2 - 2 * x - 6 * y + 4 * z ≤ 8 ↔
   z = 3 - 1/2 * x - 3/2 * y ∧ (x - 2)^2 + (3 * y - 2)^2 ≤ 4 ∧ 0 ≤ x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1017_101754


namespace NUMINAMATH_CALUDE_two_digit_number_representation_l1017_101735

/-- Represents a two-digit number -/
def two_digit_number (x y : ℕ) : ℕ := 10 * x + y

/-- The tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- The units digit of a two-digit number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_number_representation (x y : ℕ) (h1 : x < 10) (h2 : y < 10) :
  two_digit_number x y = 10 * x + y :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_representation_l1017_101735


namespace NUMINAMATH_CALUDE_popcorn_shrimp_orders_l1017_101774

theorem popcorn_shrimp_orders (catfish_price popcorn_price : ℚ)
  (total_orders : ℕ) (total_amount : ℚ)
  (h1 : catfish_price = 6)
  (h2 : popcorn_price = (7/2))
  (h3 : total_orders = 26)
  (h4 : total_amount = (267/2)) :
  ∃ (catfish_orders popcorn_orders : ℕ),
    catfish_orders + popcorn_orders = total_orders ∧
    catfish_price * catfish_orders + popcorn_price * popcorn_orders = total_amount ∧
    popcorn_orders = 9 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_shrimp_orders_l1017_101774


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l1017_101729

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l1017_101729


namespace NUMINAMATH_CALUDE_mod_23_equivalence_l1017_101736

theorem mod_23_equivalence :
  ∃! n : ℕ, 0 ≤ n ∧ n < 23 ∧ 123456 % 23 = n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_mod_23_equivalence_l1017_101736


namespace NUMINAMATH_CALUDE_problem_solution_l1017_101710

theorem problem_solution (m n : ℝ) 
  (h1 : m = (Real.sqrt (n^2 - 4) + Real.sqrt (4 - n^2) + 4) / (n - 2))
  (h2 : n^2 - 4 ≥ 0)
  (h3 : 4 - n^2 ≥ 0)
  (h4 : n ≠ 2) :
  |m - 2*n| + Real.sqrt (8*m*n) = 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1017_101710


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1017_101709

theorem arithmetic_geometric_mean_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c) / 3)^2 ≥ (a*b + b*c + c*a) / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1017_101709


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1017_101753

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 30 →
  a^2 + b^2 = c^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1017_101753


namespace NUMINAMATH_CALUDE_function_symmetry_and_periodicity_l1017_101762

/-- A function f: ℝ → ℝ is symmetric about the line x = a if f(2a - x) = f(x) for all x ∈ ℝ -/
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

/-- A function f: ℝ → ℝ is periodic with period p if f(x + p) = f(x) for all x ∈ ℝ -/
def Periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

/-- A function f: ℝ → ℝ is symmetric about the point (a, b) if f(2a - x) = 2b - f(x) for all x ∈ ℝ -/
def SymmetricAboutPoint (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = 2 * b - f x

theorem function_symmetry_and_periodicity (f : ℝ → ℝ) :
  (∀ x, f (2 - x) = f x) → SymmetricAboutLine f 1 ∧
  (∀ x, f (x - 1) = f (x + 1)) → Periodic f 2 ∧
  (∀ x, f (2 - x) = -f x) → SymmetricAboutPoint f 1 0 := by
  sorry


end NUMINAMATH_CALUDE_function_symmetry_and_periodicity_l1017_101762


namespace NUMINAMATH_CALUDE_car_push_distance_l1017_101721

/-- Proves that the total distance traveled is 10 miles given the conditions of the problem --/
theorem car_push_distance : 
  let segment1_distance : ℝ := 3
  let segment1_speed : ℝ := 6
  let segment2_distance : ℝ := 3
  let segment2_speed : ℝ := 3
  let segment3_distance : ℝ := 4
  let segment3_speed : ℝ := 8
  let total_time : ℝ := 2
  segment1_distance / segment1_speed + 
  segment2_distance / segment2_speed + 
  segment3_distance / segment3_speed = total_time →
  segment1_distance + segment2_distance + segment3_distance = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_car_push_distance_l1017_101721


namespace NUMINAMATH_CALUDE_min_buses_is_eleven_l1017_101780

/-- The maximum number of students a bus can hold -/
def bus_capacity : ℕ := 38

/-- The total number of students to be transported -/
def total_students : ℕ := 411

/-- The minimum number of buses needed is the ceiling of the division of total students by bus capacity -/
def min_buses : ℕ := (total_students + bus_capacity - 1) / bus_capacity

/-- Theorem stating that the minimum number of buses needed is 11 -/
theorem min_buses_is_eleven : min_buses = 11 := by sorry

end NUMINAMATH_CALUDE_min_buses_is_eleven_l1017_101780


namespace NUMINAMATH_CALUDE_green_peaches_count_l1017_101712

theorem green_peaches_count (red_peaches : ℕ) (green_peaches : ℕ) 
  (h1 : red_peaches = 17)
  (h2 : red_peaches = green_peaches + 1) : 
  green_peaches = 16 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l1017_101712


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_smallest_number_of_eggs_is_162_l1017_101724

theorem smallest_number_of_eggs (total_containers : ℕ) (eggs_per_full_container : ℕ) 
  (underfilled_containers : ℕ) (eggs_per_underfilled : ℕ) : ℕ :=
  let total_eggs := (total_containers - underfilled_containers) * eggs_per_full_container + 
                    underfilled_containers * eggs_per_underfilled
  have h1 : eggs_per_full_container = 15 := by sorry
  have h2 : underfilled_containers = 3 := by sorry
  have h3 : eggs_per_underfilled = 14 := by sorry
  have h4 : total_eggs > 150 := by sorry
  have h5 : ∀ n : ℕ, n < total_containers → 
            n * eggs_per_full_container - underfilled_containers ≤ 150 := by sorry
  total_eggs

theorem smallest_number_of_eggs_is_162 : 
  smallest_number_of_eggs 11 15 3 14 = 162 := by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_smallest_number_of_eggs_is_162_l1017_101724


namespace NUMINAMATH_CALUDE_fair_coin_probability_difference_l1017_101781

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def binomialProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

/-- The statement to prove -/
theorem fair_coin_probability_difference :
  (binomialProbability 3 2) - (binomialProbability 3 3) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_probability_difference_l1017_101781


namespace NUMINAMATH_CALUDE_exact_blue_marbles_probability_l1017_101767

def total_marbles : ℕ := 20
def blue_marbles : ℕ := 12
def red_marbles : ℕ := 8
def num_draws : ℕ := 8
def num_blue_draws : ℕ := 5

def prob_blue : ℚ := blue_marbles / total_marbles
def prob_red : ℚ := red_marbles / total_marbles

theorem exact_blue_marbles_probability :
  (Nat.choose num_draws num_blue_draws : ℚ) *
  (prob_blue ^ num_blue_draws) *
  (prob_red ^ (num_draws - num_blue_draws)) =
  108864 / 390625 :=
by sorry

end NUMINAMATH_CALUDE_exact_blue_marbles_probability_l1017_101767


namespace NUMINAMATH_CALUDE_sum_18_probability_l1017_101779

/-- The number of ways to distribute 10 points among 8 dice -/
def ways_to_distribute : ℕ := 19448

/-- The total number of possible outcomes when throwing 8 dice -/
def total_outcomes : ℕ := 6^8

/-- The probability of obtaining a sum of 18 when throwing 8 fair 6-sided dice -/
def probability_sum_18 : ℚ := ways_to_distribute / total_outcomes

theorem sum_18_probability :
  probability_sum_18 = 19448 / 6^8 :=
sorry

end NUMINAMATH_CALUDE_sum_18_probability_l1017_101779


namespace NUMINAMATH_CALUDE_fireworks_cost_and_remaining_l1017_101717

def small_firework_cost : ℕ := 12
def large_firework_cost : ℕ := 25

def henry_small : ℕ := 3
def henry_large : ℕ := 2
def friend_small : ℕ := 4
def friend_large : ℕ := 1

def saved_fireworks : ℕ := 6
def used_saved_fireworks : ℕ := 3

theorem fireworks_cost_and_remaining :
  (let total_cost := (henry_small + friend_small) * small_firework_cost +
                     (henry_large + friend_large) * large_firework_cost
   let remaining_fireworks := henry_small + henry_large + friend_small + friend_large +
                              (saved_fireworks - used_saved_fireworks)
   (total_cost = 159) ∧ (remaining_fireworks = 13)) := by
  sorry

end NUMINAMATH_CALUDE_fireworks_cost_and_remaining_l1017_101717


namespace NUMINAMATH_CALUDE_integer_in_3_rows_and_3_cols_l1017_101777

/-- Represents a 21x21 array of integers -/
def Array21x21 := Fin 21 → Fin 21 → Int

/-- Predicate to check if a row has at most 6 different integers -/
def row_at_most_6_different (arr : Array21x21) (row : Fin 21) : Prop :=
  (Finset.univ.image (fun col => arr row col)).card ≤ 6

/-- Predicate to check if a column has at most 6 different integers -/
def col_at_most_6_different (arr : Array21x21) (col : Fin 21) : Prop :=
  (Finset.univ.image (fun row => arr row col)).card ≤ 6

/-- Predicate to check if an integer appears in at least 3 rows -/
def in_at_least_3_rows (arr : Array21x21) (n : Int) : Prop :=
  (Finset.univ.filter (fun row => ∃ col, arr row col = n)).card ≥ 3

/-- Predicate to check if an integer appears in at least 3 columns -/
def in_at_least_3_cols (arr : Array21x21) (n : Int) : Prop :=
  (Finset.univ.filter (fun col => ∃ row, arr row col = n)).card ≥ 3

theorem integer_in_3_rows_and_3_cols (arr : Array21x21) 
  (h_rows : ∀ row, row_at_most_6_different arr row)
  (h_cols : ∀ col, col_at_most_6_different arr col) :
  ∃ n : Int, in_at_least_3_rows arr n ∧ in_at_least_3_cols arr n := by
  sorry

end NUMINAMATH_CALUDE_integer_in_3_rows_and_3_cols_l1017_101777


namespace NUMINAMATH_CALUDE_odd_digits_181_base4_l1017_101798

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a natural number from base 8 to base 4 --/
def base8ToBase4 (n : List ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers --/
def countOddDigits (n : List ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of odd digits in the base 4 representation of 181 (base 10),
    when converted through base 8, is equal to 5 --/
theorem odd_digits_181_base4 : 
  countOddDigits (base8ToBase4 (toBase8 181)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_odd_digits_181_base4_l1017_101798


namespace NUMINAMATH_CALUDE_find_divisor_find_divisor_proof_l1017_101761

theorem find_divisor (original : ℕ) (divisible : ℕ) (divisor : ℕ) : Prop :=
  (original = 859622) →
  (divisible = 859560) →
  (divisor = 62) →
  (original - divisible = divisor) ∧
  (divisible % divisor = 0)

/-- The proof of the theorem --/
theorem find_divisor_proof : ∃ (d : ℕ), find_divisor 859622 859560 d :=
  sorry

end NUMINAMATH_CALUDE_find_divisor_find_divisor_proof_l1017_101761


namespace NUMINAMATH_CALUDE_laser_reflection_theorem_l1017_101758

/-- Regular hexagon with side length 2 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- Point G on BC where the laser beam hits -/
def G (h : RegularHexagon) : ℝ × ℝ := sorry

/-- Midpoint of DE -/
def M (h : RegularHexagon) : ℝ × ℝ := sorry

/-- Length of BG -/
def BG_length (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating that BG length is 2/5 -/
theorem laser_reflection_theorem (h : RegularHexagon) :
  let g := G h
  let m := M h
  (∃ (t : ℝ), t • (g.1 - h.A.1, g.2 - h.A.2) = (m.1 - g.1, m.2 - g.2)) →
  BG_length h = 2/5 := by sorry

end NUMINAMATH_CALUDE_laser_reflection_theorem_l1017_101758


namespace NUMINAMATH_CALUDE_factorization_equivalence_l1017_101766

variable (a x y : ℝ)

theorem factorization_equivalence : 
  (2*a*x^2 - 8*a*x*y + 8*a*y^2 = 2*a*(x - 2*y)^2) ∧ 
  (6*x*y^2 - 9*x^2*y - y^3 = -y*(3*x - y)^2) := by sorry

end NUMINAMATH_CALUDE_factorization_equivalence_l1017_101766


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l1017_101764

/-- Given a cube with a face perimeter of 24 cm, prove its volume is 216 cubic cm. -/
theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 24) :
  let side_length := face_perimeter / 4
  side_length ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l1017_101764


namespace NUMINAMATH_CALUDE_equation_solution_l1017_101752

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 3
def g (x y : ℝ) : ℝ := 3 * x + y

-- State the theorem
theorem equation_solution (x y : ℝ) :
  2 * (f x) - 11 + g x y = f (x - 2) ↔ y = -5 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1017_101752


namespace NUMINAMATH_CALUDE_equation_solution_l1017_101778

theorem equation_solution (a b x : ℝ) (h : b ≠ 0) :
  a * (Real.cos (x / 2))^2 - (a + 2 * b) * (Real.sin (x / 2))^2 = a * Real.cos x - b * Real.sin x ↔
  (∃ n : ℤ, x = 2 * n * Real.pi) ∨ (∃ k : ℤ, x = Real.pi / 2 * (4 * k + 1)) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l1017_101778


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1017_101751

def M : Set Int := {m | -3 < m ∧ m < 2}
def N : Set Int := {n | -1 ≤ n ∧ n ≤ 3}

theorem union_of_M_and_N : M ∪ N = {-2, -1, 0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1017_101751


namespace NUMINAMATH_CALUDE_total_pencils_is_twelve_l1017_101746

/-- The number of pencils each child has -/
def pencils_per_child : ℕ := 6

/-- The number of children -/
def number_of_children : ℕ := 2

/-- The total number of pencils -/
def total_pencils : ℕ := pencils_per_child * number_of_children

theorem total_pencils_is_twelve : total_pencils = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_is_twelve_l1017_101746


namespace NUMINAMATH_CALUDE_intersection_is_empty_l1017_101711

-- Define set A
def A : Set ℝ := {x | x^2 + 4 ≤ 5*x}

-- Define set B
def B : Set (ℝ × ℝ) := {p | p.2 = 3^p.1 + 2}

-- Theorem statement
theorem intersection_is_empty : A ∩ (B.image Prod.fst) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l1017_101711


namespace NUMINAMATH_CALUDE_line_l_theorem_circle_M_theorem_l1017_101789

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 5 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, -1)

-- Define point Q
def point_Q : ℝ × ℝ := (0, 1)

-- Define the line l
def line_l (x y : ℝ) : Prop := x = -2 ∨ y = (15/8)*x + 11/4

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y - 7 = 0

-- Theorem for line l
theorem line_l_theorem : 
  ∃ (A B : ℝ × ℝ), 
    (∀ (x y : ℝ), line_l x y ↔ (∃ t : ℝ, (x, y) = (1-t) • point_P + t • A ∨ (x, y) = (1-t) • point_P + t • B)) ∧
    circle_C A.1 A.2 ∧ 
    circle_C B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16 :=
sorry

-- Theorem for circle M
theorem circle_M_theorem :
  (∀ x y : ℝ, circle_M x y → (x = point_P.1 ∧ y = point_P.2) ∨ (x = point_Q.1 ∧ y = point_Q.2)) ∧
  (∃ t : ℝ, ∀ x y : ℝ, circle_M x y → circle_C x y ∨ (x, y) = (1-t) • point_Q + t • point_P) :=
sorry

end NUMINAMATH_CALUDE_line_l_theorem_circle_M_theorem_l1017_101789


namespace NUMINAMATH_CALUDE_notP_set_equals_interval_l1017_101738

-- Define the proposition P
def P (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0

-- Define the set of x satisfying ¬P
def notP_set : Set ℝ := {x : ℝ | ¬(P x)}

-- Theorem stating that notP_set is equal to the closed interval [-1, 2]
theorem notP_set_equals_interval :
  notP_set = Set.Icc (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_notP_set_equals_interval_l1017_101738


namespace NUMINAMATH_CALUDE_bowling_team_weight_l1017_101700

theorem bowling_team_weight (original_avg : ℝ) : 
  (7 * original_avg + 110 + 60) / 9 = 92 → original_avg = 94 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_weight_l1017_101700


namespace NUMINAMATH_CALUDE_triangle_base_value_l1017_101727

theorem triangle_base_value (L R B : ℝ) : 
  L + R + B = 50 →
  R = L + 2 →
  L = 12 →
  B = 24 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_value_l1017_101727


namespace NUMINAMATH_CALUDE_drums_per_day_l1017_101737

/-- Given that 2916 drums of grapes are filled in 9 days, 
    prove that 324 drums of grapes are filled per day. -/
theorem drums_per_day (total_drums : ℕ) (total_days : ℕ) 
  (h1 : total_drums = 2916) (h2 : total_days = 9) :
  total_drums / total_days = 324 := by
  sorry

end NUMINAMATH_CALUDE_drums_per_day_l1017_101737


namespace NUMINAMATH_CALUDE_note_count_l1017_101782

theorem note_count (total_amount : ℕ) (denomination_1 : ℕ) (denomination_5 : ℕ) (denomination_10 : ℕ) :
  total_amount = 192 ∧
  denomination_1 = 1 ∧
  denomination_5 = 5 ∧
  denomination_10 = 10 ∧
  (∃ (x : ℕ), x * denomination_1 + x * denomination_5 + x * denomination_10 = total_amount) →
  (∃ (x : ℕ), x * 3 = 36 ∧ x * denomination_1 + x * denomination_5 + x * denomination_10 = total_amount) :=
by sorry

end NUMINAMATH_CALUDE_note_count_l1017_101782


namespace NUMINAMATH_CALUDE_spring_membership_decrease_l1017_101771

theorem spring_membership_decrease
  (fall_increase : Real)
  (total_decrease : Real)
  (h1 : fall_increase = 0.06)
  (h2 : total_decrease = 0.1414) :
  let fall_membership := 1 + fall_increase
  let spring_membership := 1 - total_decrease
  (fall_membership - spring_membership) / fall_membership = 0.19 := by
sorry

end NUMINAMATH_CALUDE_spring_membership_decrease_l1017_101771


namespace NUMINAMATH_CALUDE_specific_solid_volume_l1017_101773

/-- A solid with a square base and specific edge lengths -/
structure Solid where
  s : ℝ
  base_side_length : s > 0
  upper_edge_length : ℝ
  upper_edge_parallel : upper_edge_length = 3 * s
  other_edges_length : ℝ
  other_edges_equal_s : other_edges_length = s

/-- The volume of the solid -/
noncomputable def volume (solid : Solid) : ℝ := sorry

/-- Theorem stating the volume of the specific solid -/
theorem specific_solid_volume :
  ∀ (solid : Solid),
    solid.s = 4 * Real.sqrt 2 →
    volume solid = 144 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_solid_volume_l1017_101773


namespace NUMINAMATH_CALUDE_gas_tank_fill_level_l1017_101744

theorem gas_tank_fill_level (tank_capacity : ℚ) (initial_fill_fraction : ℚ) (added_amount : ℚ) : 
  tank_capacity = 42 → 
  initial_fill_fraction = 3/4 → 
  added_amount = 7 → 
  (initial_fill_fraction * tank_capacity + added_amount) / tank_capacity = 833/909 := by
  sorry

end NUMINAMATH_CALUDE_gas_tank_fill_level_l1017_101744


namespace NUMINAMATH_CALUDE_faye_earnings_proof_l1017_101745

/-- The number of bead necklaces Faye sold -/
def bead_necklaces : ℕ := 3

/-- The number of gem stone necklaces Faye sold -/
def gem_necklaces : ℕ := 7

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 7

/-- Faye's total earnings from selling necklaces -/
def faye_earnings : ℕ := (bead_necklaces + gem_necklaces) * necklace_cost

theorem faye_earnings_proof : faye_earnings = 70 := by
  sorry

end NUMINAMATH_CALUDE_faye_earnings_proof_l1017_101745


namespace NUMINAMATH_CALUDE_constant_term_of_expansion_l1017_101739

/-- The constant term in the expansion of (9x + 2/(3x))^8 -/
def constant_term : ℕ := 90720

/-- The binomial coefficient (8 choose 4) -/
def binomial_8_4 : ℕ := 70

theorem constant_term_of_expansion :
  constant_term = binomial_8_4 * 9^4 * 2^4 / 3^4 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_of_expansion_l1017_101739


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l1017_101790

theorem max_value_of_expression (x : ℝ) : 
  x^6 / (x^10 + 3*x^8 - 5*x^6 + 10*x^4 + 25) ≤ 1 / (5 + 2 * Real.sqrt 30) :=
sorry

theorem max_value_achievable : 
  ∃ x : ℝ, x^6 / (x^10 + 3*x^8 - 5*x^6 + 10*x^4 + 25) = 1 / (5 + 2 * Real.sqrt 30) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l1017_101790


namespace NUMINAMATH_CALUDE_min_sum_squares_l1017_101796

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + z = 1) :
  ∃ (m : ℝ), (∀ x' y' z' : ℝ, x' + 2*y' + z' = 1 → x'^2 + y'^2 + z'^2 ≥ m) ∧
             (∃ x₀ y₀ z₀ : ℝ, x₀ + 2*y₀ + z₀ = 1 ∧ x₀^2 + y₀^2 + z₀^2 = m) ∧
             m = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1017_101796


namespace NUMINAMATH_CALUDE_comparison_theorem_l1017_101701

theorem comparison_theorem :
  (-3 / 4 : ℚ) < -2 / 3 ∧ (3 : ℤ) > -|4| := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l1017_101701


namespace NUMINAMATH_CALUDE_rhombus_properties_l1017_101797

-- Define the rhombus ABCD
def Rhombus (A B C D : ℝ × ℝ) : Prop :=
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A B = 4 ∧ dist B C = 4 ∧ dist C D = 4 ∧ dist D A = 4

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the condition for point A on the semicircle
def OnSemicircle (A : ℝ × ℝ) : Prop :=
  (A.1 - 2)^2 + A.2^2 = 4 ∧ 2 ≤ A.1 ∧ A.1 ≤ 4

-- Main theorem
theorem rhombus_properties
  (A B C D : ℝ × ℝ)
  (h_rhombus : Rhombus A B C D)
  (h_OB : dist O B = 6)
  (h_OD : dist O D = 6)
  (h_A_semicircle : OnSemicircle A) :
  (∃ k, dist O A * dist O B = k) ∧
  (∃ y, -5 ≤ y ∧ y ≤ 5 ∧ C = (5, y)) :=
sorry

#check rhombus_properties

end NUMINAMATH_CALUDE_rhombus_properties_l1017_101797


namespace NUMINAMATH_CALUDE_problem_solution_l1017_101719

-- Define the set D
def D : Set ℝ := {x | x < -4 ∨ x > 0}

-- Define proposition p
def p (a : ℝ) : Prop := a ∈ D

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 - a*x₀ - a ≤ -3

-- Theorem statement
theorem problem_solution :
  (∀ a : ℝ, q a → p a) ∧ (∃ a : ℝ, p a ∧ ¬q a) →
  D = {x : ℝ | x < -4 ∨ x > 0} :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1017_101719


namespace NUMINAMATH_CALUDE_intersection_exists_l1017_101784

-- Define a structure for a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for a set of 5 points
def FivePoints := Fin 5 → Point3D

-- Define a predicate for 4 points being non-coplanar
def nonCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define a predicate for a line intersecting a triangle
def lineIntersectsTriangle (l1 l2 p1 p2 p3 : Point3D) : Prop := sorry

-- Main theorem
theorem intersection_exists (points : FivePoints) 
  (h : ∀ i j k l : Fin 5, i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → 
       nonCoplanar (points i) (points j) (points k) (points l)) :
  ∃ i j k l m : Fin 5, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ k ≠ l ∧ k ≠ m ∧ l ≠ m ∧
    lineIntersectsTriangle (points i) (points j) (points k) (points l) (points m) :=
  sorry

end NUMINAMATH_CALUDE_intersection_exists_l1017_101784


namespace NUMINAMATH_CALUDE_triangle_side_length_l1017_101799

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Conditions
  a = Real.sqrt 3 →
  b = 1 →
  A = 2 * B →
  -- Triangle properties
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  -- Sine law
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  -- Question/Conclusion
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1017_101799


namespace NUMINAMATH_CALUDE_marble_collection_total_l1017_101795

theorem marble_collection_total (b : ℝ) : 
  let r := 1.3 * b -- red marbles
  let g := 1.5 * b -- green marbles
  r + b + g = 3.8 * b := by sorry

end NUMINAMATH_CALUDE_marble_collection_total_l1017_101795


namespace NUMINAMATH_CALUDE_eulers_formula_3d_l1017_101783

/-- A space convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- Euler's formula for space convex polyhedra -/
theorem eulers_formula_3d (p : ConvexPolyhedron) : p.faces + p.vertices - p.edges = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_3d_l1017_101783


namespace NUMINAMATH_CALUDE_hot_water_bottle_price_is_six_l1017_101704

/-- The price of a hot-water bottle given the conditions of the problem -/
def hot_water_bottle_price (thermometer_price : ℚ) (total_sales : ℚ) 
  (thermometer_to_bottle_ratio : ℕ) (bottles_sold : ℕ) : ℚ :=
  (total_sales - thermometer_price * (thermometer_to_bottle_ratio * bottles_sold)) / bottles_sold

theorem hot_water_bottle_price_is_six :
  hot_water_bottle_price 2 1200 7 60 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hot_water_bottle_price_is_six_l1017_101704


namespace NUMINAMATH_CALUDE_max_a_for_decreasing_cosine_minus_sine_l1017_101768

theorem max_a_for_decreasing_cosine_minus_sine :
  let f : ℝ → ℝ := λ x ↦ Real.cos x - Real.sin x
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ a → f y < f x) →
    a ≤ 3 * Real.pi / 4 ∧ 
    ∃ b : ℝ, b > 3 * Real.pi / 4 ∧ ¬(∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ b → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_decreasing_cosine_minus_sine_l1017_101768


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1017_101792

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 5, 7]

def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]

theorem matrix_equation_solution (x y z w : ℝ) 
  (h1 : A * B x y z w = B x y z w * A)
  (h2 : 2 * z ≠ 5 * y) :
  ∃ x y z w, (x - w) / (z - 2 * y) = 0 :=
by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1017_101792


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1017_101731

theorem simplify_trig_expression (x : ℝ) :
  (1 + Real.sin x + Real.cos x) / (1 - Real.sin x + Real.cos x) = Real.tan (π / 4 + x / 2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1017_101731


namespace NUMINAMATH_CALUDE_collinear_points_sum_l1017_101716

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p q r : ℝ × ℝ × ℝ) : Prop := sorry

/-- The theorem states that if the given points are collinear, then a + b = 6. -/
theorem collinear_points_sum (a b : ℝ) : 
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l1017_101716


namespace NUMINAMATH_CALUDE_max_fleas_on_board_l1017_101742

/-- Represents a 10x10 board --/
def Board := Fin 10 → Fin 10 → Bool

/-- Represents the four possible directions of flea movement --/
inductive Direction
| Up
| Down
| Left
| Right

/-- Represents a flea's position and direction --/
structure Flea where
  pos : Fin 10 × Fin 10
  dir : Direction

/-- Represents the state of the board and fleas at a given time --/
structure BoardState where
  board : Board
  fleas : List Flea

/-- Simulates the movement of fleas for one hour (60 minutes) --/
def simulateMovement (initialState : BoardState) : BoardState :=
  sorry

/-- Checks if the simulation results in a valid state (no overlapping fleas) --/
def isValidSimulation (finalState : BoardState) : Bool :=
  sorry

/-- Theorem stating the maximum number of fleas --/
theorem max_fleas_on_board :
  ∀ (initialState : BoardState),
    isValidSimulation (simulateMovement initialState) →
    initialState.fleas.length ≤ 40 :=
  sorry

end NUMINAMATH_CALUDE_max_fleas_on_board_l1017_101742


namespace NUMINAMATH_CALUDE_inequality_proof_l1017_101740

theorem inequality_proof (a b : ℝ) (h : a + b > 0) :
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1017_101740


namespace NUMINAMATH_CALUDE_hiker_problem_l1017_101725

/-- A hiker's walking problem -/
theorem hiker_problem (h : ℕ) : 
  (3 * h) + (4 * (h - 1)) + 15 = 53 → 3 * h = 18 := by
  sorry

end NUMINAMATH_CALUDE_hiker_problem_l1017_101725


namespace NUMINAMATH_CALUDE_recipe_sugar_amount_l1017_101720

/-- The amount of sugar Katie has already put in the recipe -/
def sugar_already_added : ℝ := 0.5

/-- The amount of sugar Katie still needs to add to the recipe -/
def sugar_to_add : ℝ := 2.5

/-- The total amount of sugar required by the recipe -/
def total_sugar_needed : ℝ := sugar_already_added + sugar_to_add

theorem recipe_sugar_amount : total_sugar_needed = 3 := by
  sorry

end NUMINAMATH_CALUDE_recipe_sugar_amount_l1017_101720


namespace NUMINAMATH_CALUDE_problem_solution_l1017_101734

theorem problem_solution (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1017_101734


namespace NUMINAMATH_CALUDE_fraction_cube_two_thirds_l1017_101702

theorem fraction_cube_two_thirds : (2 / 3 : ℚ) ^ 3 = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cube_two_thirds_l1017_101702


namespace NUMINAMATH_CALUDE_loan_income_is_135_l1017_101713

/-- Calculates the yearly annual income from two parts of a loan at different interest rates -/
def yearly_income (total : ℚ) (part1 : ℚ) (rate1 : ℚ) (rate2 : ℚ) : ℚ :=
  let part2 := total - part1
  part1 * rate1 + part2 * rate2

/-- Theorem stating that the yearly income from the given loan parts is 135 -/
theorem loan_income_is_135 :
  yearly_income 2500 1500 (5/100) (6/100) = 135 := by
  sorry

end NUMINAMATH_CALUDE_loan_income_is_135_l1017_101713


namespace NUMINAMATH_CALUDE_find_M_l1017_101715

theorem find_M : ∃ M : ℕ, 995 + 997 + 999 + 1001 + 1003 = 5100 - M ∧ M = 104 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l1017_101715


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l1017_101759

/-- Given a line with slope m and y-intercept b, prove that their product mb equals -6 -/
theorem line_slope_intercept_product :
  ∀ (m b : ℝ), m = 2 → b = -3 → m * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l1017_101759


namespace NUMINAMATH_CALUDE_vector_orthogonality_l1017_101770

def a : ℝ × ℝ := (3, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)
def c : ℝ × ℝ := (0, 2)

theorem vector_orthogonality (x : ℝ) :
  a • (b x - c) = 0 → x = 4/3 := by sorry

end NUMINAMATH_CALUDE_vector_orthogonality_l1017_101770


namespace NUMINAMATH_CALUDE_reflected_polygon_area_equal_l1017_101743

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a polygon with n vertices -/
structure Polygon (n : ℕ) where
  vertices : Fin n → Point

/-- Calculates the area of a polygon -/
def area (p : Polygon n) : ℝ := sorry

/-- Reflects a point across the midpoint of two other points -/
def reflect (p : Point) (a : Point) (b : Point) : Point := sorry

/-- Creates a new polygon by reflecting each vertex of the given polygon
    across the midpoint of the corresponding side of the regular 2009-gon -/
def reflectedPolygon (p : Polygon 2009) (regularPolygon : Polygon 2009) : Polygon 2009 := sorry

/-- Theorem stating that the area of the reflected polygon is equal to the area of the original polygon -/
theorem reflected_polygon_area_equal (p : Polygon 2009) (regularPolygon : Polygon 2009) :
  area (reflectedPolygon p regularPolygon) = area p := by sorry

end NUMINAMATH_CALUDE_reflected_polygon_area_equal_l1017_101743


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l1017_101749

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_s

/-- Proof that a train with speed 30 km/h crossing a pole in 9 seconds has a length of approximately 75 meters -/
theorem train_length_proof (ε : ℝ) (h_ε : ε > 0) :
  ∃ (l : ℝ), abs (l - train_length 30 9) < ε ∧ l = 75 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l1017_101749


namespace NUMINAMATH_CALUDE_max_sum_abc_l1017_101718

/-- Definition of An as an n-digit number with all digits equal to a -/
def An (a : ℕ) (n : ℕ) : ℕ := a * (10^n - 1) / 9

/-- Definition of Bn as an n-digit number with all digits equal to b -/
def Bn (b : ℕ) (n : ℕ) : ℕ := b * (10^n - 1) / 9

/-- Definition of Cn as a 2n-digit number with all digits equal to c -/
def Cn (c : ℕ) (n : ℕ) : ℕ := c * (10^(2*n) - 1) / 9

/-- The main theorem stating that the maximum value of a + b + c is 18 -/
theorem max_sum_abc :
  ∃ (a b c : ℕ),
    (0 < a ∧ a ≤ 9) ∧
    (0 < b ∧ b ≤ 9) ∧
    (0 < c ∧ c ≤ 9) ∧
    (∃ (n₁ n₂ : ℕ), n₁ ≠ n₂ ∧ Cn c n₁ - Bn b n₁ = (An a n₁)^2 ∧ Cn c n₂ - Bn b n₂ = (An a n₂)^2) ∧
    a + b + c = 18 ∧
    ∀ (a' b' c' : ℕ),
      (0 < a' ∧ a' ≤ 9) →
      (0 < b' ∧ b' ≤ 9) →
      (0 < c' ∧ c' ≤ 9) →
      (∃ (n₁ n₂ : ℕ), n₁ ≠ n₂ ∧ Cn c' n₁ - Bn b' n₁ = (An a' n₁)^2 ∧ Cn c' n₂ - Bn b' n₂ = (An a' n₂)^2) →
      a' + b' + c' ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_abc_l1017_101718


namespace NUMINAMATH_CALUDE_bugs_meeting_point_l1017_101794

/-- Triangle DEF with given side lengths -/
structure Triangle where
  DE : ℝ
  EF : ℝ
  FD : ℝ

/-- Two bugs crawling on the triangle's perimeter -/
structure Bugs where
  speed1 : ℝ
  speed2 : ℝ
  direction : Bool -- True if same direction, False if opposite

/-- Point G where the bugs meet -/
def meetingPoint (t : Triangle) (b : Bugs) : ℝ := sorry

/-- Theorem stating that EG = 2 under given conditions -/
theorem bugs_meeting_point (t : Triangle) (b : Bugs) : 
  t.DE = 8 ∧ t.EF = 10 ∧ t.FD = 12 ∧ 
  b.speed1 = 1 ∧ b.speed2 = 2 ∧ b.direction = false → 
  meetingPoint t b = 2 := by sorry

end NUMINAMATH_CALUDE_bugs_meeting_point_l1017_101794


namespace NUMINAMATH_CALUDE_mixture_volume_proof_l1017_101750

/-- The initial volume of the mixture -/
def initial_volume : ℝ := 150

/-- The percentage of water in the initial mixture -/
def initial_water_percentage : ℝ := 0.15

/-- The volume of water added to the mixture -/
def added_water : ℝ := 20

/-- The percentage of water in the new mixture after adding water -/
def new_water_percentage : ℝ := 0.25

theorem mixture_volume_proof :
  initial_volume = 150 ∧
  initial_water_percentage * initial_volume + added_water = new_water_percentage * (initial_volume + added_water) :=
by sorry

end NUMINAMATH_CALUDE_mixture_volume_proof_l1017_101750


namespace NUMINAMATH_CALUDE_missing_sale_is_7225_l1017_101705

/-- Calculates the missing month's sale given the sales of other months and the target average --/
def calculate_missing_sale (sale1 sale2 sale3 sale5 sale6 target_average : ℕ) : ℕ :=
  6 * target_average - (sale1 + sale2 + sale3 + sale5 + sale6)

/-- Proves that the missing month's sale is 7225 given the problem conditions --/
theorem missing_sale_is_7225 :
  let sale1 : ℕ := 6235
  let sale2 : ℕ := 6927
  let sale3 : ℕ := 6855
  let sale5 : ℕ := 6562
  let sale6 : ℕ := 5191
  let target_average : ℕ := 6500
  calculate_missing_sale sale1 sale2 sale3 sale5 sale6 target_average = 7225 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_sale_is_7225_l1017_101705


namespace NUMINAMATH_CALUDE_word_game_possible_l1017_101757

structure WordDistribution where
  anya_only : ℕ
  borya_only : ℕ
  vasya_only : ℕ
  anya_borya : ℕ
  anya_vasya : ℕ
  borya_vasya : ℕ

def total_words (d : WordDistribution) : ℕ :=
  d.anya_only + d.borya_only + d.vasya_only + d.anya_borya + d.anya_vasya + d.borya_vasya

def anya_words (d : WordDistribution) : ℕ :=
  d.anya_only + d.anya_borya + d.anya_vasya

def borya_words (d : WordDistribution) : ℕ :=
  d.borya_only + d.anya_borya + d.borya_vasya

def vasya_words (d : WordDistribution) : ℕ :=
  d.vasya_only + d.anya_vasya + d.borya_vasya

def anya_score (d : WordDistribution) : ℕ :=
  2 * d.anya_only + d.anya_borya + d.anya_vasya

def borya_score (d : WordDistribution) : ℕ :=
  2 * d.borya_only + d.anya_borya + d.borya_vasya

def vasya_score (d : WordDistribution) : ℕ :=
  2 * d.vasya_only + d.anya_vasya + d.borya_vasya

theorem word_game_possible : ∃ d : WordDistribution,
  anya_words d > borya_words d ∧
  borya_words d > vasya_words d ∧
  vasya_score d > borya_score d ∧
  borya_score d > anya_score d :=
sorry

end NUMINAMATH_CALUDE_word_game_possible_l1017_101757


namespace NUMINAMATH_CALUDE_expression_evaluation_l1017_101769

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := -1
  (2*a - b)^2 + (a - b)*(a + b) - 5*a*(a - 2*b) = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1017_101769


namespace NUMINAMATH_CALUDE_smallest_n_for_special_function_l1017_101741

theorem smallest_n_for_special_function : ∃ (n : ℕ) (f : ℤ → Fin n),
  (∀ (A B : ℤ), |A - B| ∈ ({5, 7, 12} : Set ℤ) → f A ≠ f B) ∧
  (∀ (m : ℕ), m < n → ¬∃ (g : ℤ → Fin m), ∀ (A B : ℤ), |A - B| ∈ ({5, 7, 12} : Set ℤ) → g A ≠ g B) ∧
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_special_function_l1017_101741


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_seven_l1017_101775

theorem reciprocal_of_negative_seven :
  (1 : ℚ) / (-7 : ℚ) = -1/7 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_seven_l1017_101775


namespace NUMINAMATH_CALUDE_composition_f_equals_one_over_e_l1017_101730

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

-- State the theorem
theorem composition_f_equals_one_over_e :
  f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_composition_f_equals_one_over_e_l1017_101730


namespace NUMINAMATH_CALUDE_nancy_bead_purchase_cost_l1017_101788

/-- The total cost of Nancy's purchase given the prices of crystal and metal beads and the quantities she buys. -/
theorem nancy_bead_purchase_cost (crystal_price metal_price : ℕ) (crystal_qty metal_qty : ℕ) : 
  crystal_price = 9 → metal_price = 10 → crystal_qty = 1 → metal_qty = 2 →
  crystal_price * crystal_qty + metal_price * metal_qty = 29 := by
sorry

end NUMINAMATH_CALUDE_nancy_bead_purchase_cost_l1017_101788


namespace NUMINAMATH_CALUDE_bead_purchase_cost_l1017_101793

/-- Calculate the total cost of bead sets after discounts and taxes --/
theorem bead_purchase_cost (crystal_price metal_price glass_price : ℚ)
  (crystal_sets metal_sets glass_sets : ℕ)
  (crystal_discount metal_tax glass_discount : ℚ) :
  let crystal_cost := crystal_price * crystal_sets * (1 - crystal_discount)
  let metal_cost := metal_price * metal_sets * (1 + metal_tax)
  let glass_cost := glass_price * glass_sets * (1 - glass_discount)
  crystal_cost + metal_cost + glass_cost = 11028 / 100 →
  crystal_price = 12 →
  metal_price = 15 →
  glass_price = 8 →
  crystal_sets = 3 →
  metal_sets = 4 →
  glass_sets = 2 →
  crystal_discount = 1 / 10 →
  metal_tax = 1 / 20 →
  glass_discount = 7 / 100 →
  true := by sorry

end NUMINAMATH_CALUDE_bead_purchase_cost_l1017_101793


namespace NUMINAMATH_CALUDE_right_quadrilateral_area_area_is_twelve_l1017_101714

/-- A quadrilateral with right angles at B and D, diagonal AC of length 5, and sides AB and AD of lengths 3 and 4 respectively. -/
structure RightQuadrilateral where
  AC : ℝ
  AB : ℝ
  AD : ℝ
  ac_eq : AC = 5
  ab_eq : AB = 3
  ad_eq : AD = 4

/-- The area of the RightQuadrilateral is 12. -/
theorem right_quadrilateral_area (q : RightQuadrilateral) : ℝ :=
  12

/-- The area of a RightQuadrilateral is equal to 12. -/
theorem area_is_twelve (q : RightQuadrilateral) : right_quadrilateral_area q = 12 := by
  sorry

end NUMINAMATH_CALUDE_right_quadrilateral_area_area_is_twelve_l1017_101714


namespace NUMINAMATH_CALUDE_product_b_sample_size_l1017_101708

/-- Calculates the number of items drawn from a specific product
    using stratified sampling method. -/
def stratifiedSample (totalItems : ℕ) (ratio : List ℕ) (sampleSize : ℕ) (productIndex : ℕ) : ℕ :=
  (sampleSize * (ratio.get! productIndex)) / (ratio.sum)

/-- Theorem: Given 1200 total items with ratio 3:4:5 for products A, B, and C,
    when drawing 60 items using stratified sampling,
    the number of items drawn from product B is 20. -/
theorem product_b_sample_size :
  let totalItems : ℕ := 1200
  let ratio : List ℕ := [3, 4, 5]
  let sampleSize : ℕ := 60
  let productBIndex : ℕ := 1
  stratifiedSample totalItems ratio sampleSize productBIndex = 20 := by
  sorry

end NUMINAMATH_CALUDE_product_b_sample_size_l1017_101708


namespace NUMINAMATH_CALUDE_function_transformation_l1017_101726

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 3) : f (-(-1)) + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l1017_101726


namespace NUMINAMATH_CALUDE_fraction_problem_l1017_101776

theorem fraction_problem (N : ℚ) : (5 / 6 : ℚ) * N = (5 / 16 : ℚ) * N + 250 → N = 480 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1017_101776


namespace NUMINAMATH_CALUDE_patches_in_unit_l1017_101756

/-- The number of patches in a unit given cost price, selling price, and net profit -/
theorem patches_in_unit (cost_price selling_price net_profit : ℚ) : 
  cost_price = 1.25 → 
  selling_price = 12 → 
  net_profit = 1075 → 
  (net_profit / (selling_price - cost_price) : ℚ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_patches_in_unit_l1017_101756


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l1017_101785

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary (sum to 90°)
  a = 4 * b →   -- The angles are in a ratio of 4:1
  b = 18 :=     -- The smaller angle is 18°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l1017_101785


namespace NUMINAMATH_CALUDE_last_match_wickets_specific_case_l1017_101763

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  initialWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken in the last match -/
def lastMatchWickets (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the number of wickets in the last match is 8 -/
theorem last_match_wickets_specific_case :
  let stats : BowlerStats := {
    initialAverage := 12.4,
    initialWickets := 175,
    lastMatchRuns := 26,
    averageDecrease := 0.4
  }
  lastMatchWickets stats = 8 := by sorry

end NUMINAMATH_CALUDE_last_match_wickets_specific_case_l1017_101763


namespace NUMINAMATH_CALUDE_sons_age_l1017_101722

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 33 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1017_101722


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_perpendicular_lines_a_value_proof_l1017_101728

/-- Given two lines that are perpendicular, find the value of 'a' -/
theorem perpendicular_lines_a_value : ℝ → Prop :=
  fun a => 
    let line1 := fun x y : ℝ => 3 * y - x + 4 = 0
    let line2 := fun x y : ℝ => 4 * y + a * x + 5 = 0
    let slope1 := (1 : ℝ) / 3
    let slope2 := -a / 4
    (∀ x y : ℝ, line1 x y ∧ line2 x y → slope1 * slope2 = -1) →
    a = 12

/-- Proof of the theorem -/
theorem perpendicular_lines_a_value_proof : perpendicular_lines_a_value 12 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_perpendicular_lines_a_value_proof_l1017_101728


namespace NUMINAMATH_CALUDE_sports_club_membership_l1017_101732

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 42 →
  badminton = 20 →
  tennis = 23 →
  both = 7 →
  total - (badminton + tennis - both) = 6 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_membership_l1017_101732


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l1017_101747

theorem trigonometric_expression_equality : 
  4 * Real.sin (60 * π / 180) - Real.sqrt 12 + (-3)^2 - (1 / (2 - Real.sqrt 3)) = 7 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l1017_101747


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_300_l1017_101703

/-- The sum of the digits in the binary representation of a natural number -/
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- Theorem: The sum of the digits in the binary representation of 300 is 4 -/
theorem sum_of_binary_digits_300 : sum_of_binary_digits 300 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_300_l1017_101703


namespace NUMINAMATH_CALUDE_f_minus_two_equals_minus_twelve_l1017_101786

def symmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

theorem f_minus_two_equals_minus_twelve
  (f : ℝ → ℝ)
  (h_symmetric : symmetricAbout f 1)
  (h_def : ∀ x : ℝ, x ≥ 1 → f x = x * (1 - x)) :
  f (-2) = -12 := by
  sorry

end NUMINAMATH_CALUDE_f_minus_two_equals_minus_twelve_l1017_101786


namespace NUMINAMATH_CALUDE_farmer_randy_planting_l1017_101760

/-- Calculates the number of acres each tractor needs to plant per day -/
def acres_per_tractor_per_day (total_acres : ℕ) (total_days : ℕ) 
  (tractors_first_period : ℕ) (days_first_period : ℕ)
  (tractors_second_period : ℕ) (days_second_period : ℕ) : ℚ :=
  total_acres / (tractors_first_period * days_first_period + 
                 tractors_second_period * days_second_period)

theorem farmer_randy_planting (total_acres : ℕ) (total_days : ℕ) 
  (tractors_first_period : ℕ) (days_first_period : ℕ)
  (tractors_second_period : ℕ) (days_second_period : ℕ) 
  (h1 : total_acres = 1700)
  (h2 : total_days = 5)
  (h3 : tractors_first_period = 2)
  (h4 : days_first_period = 2)
  (h5 : tractors_second_period = 7)
  (h6 : days_second_period = 3)
  (h7 : total_days = days_first_period + days_second_period) :
  acres_per_tractor_per_day total_acres total_days 
    tractors_first_period days_first_period
    tractors_second_period days_second_period = 68 := by
  sorry

#eval acres_per_tractor_per_day 1700 5 2 2 7 3

end NUMINAMATH_CALUDE_farmer_randy_planting_l1017_101760


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1017_101791

theorem coefficient_of_x_cubed (x : ℝ) : 
  let expression := 2*(x^2 - 2*x^3 + x) + 4*(x + 3*x^3 - 2*x^2 + 2*x^5 + 2*x^3) - 3*(2 + x - 5*x^3 - x^2)
  ∃ (a b c d : ℝ), expression = a*x^5 + b*x^4 + 31*x^3 + c*x^2 + d*x + (2 * 1 - 3 * 2) :=
by sorry

#check coefficient_of_x_cubed

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1017_101791


namespace NUMINAMATH_CALUDE_cubic_with_repeated_root_l1017_101765

/-- Given a cubic polynomial 2x^3 + 8x^2 - 120x + k = 0 with a repeated root and positive k,
    prove that k = 6400/27 -/
theorem cubic_with_repeated_root (k : ℝ) : 
  (∃ x y : ℝ, (2 * x^3 + 8 * x^2 - 120 * x + k = 0) ∧ 
               (2 * y^3 + 8 * y^2 - 120 * y + k = 0) ∧ 
               (x ≠ y)) ∧
  (∃ z : ℝ, (2 * z^3 + 8 * z^2 - 120 * z + k = 0) ∧ 
            (∀ w : ℝ, 2 * w^3 + 8 * w^2 - 120 * w + k = 0 → w = z ∨ w = x ∨ w = y)) ∧
  (k > 0) →
  k = 6400 / 27 := by
sorry

end NUMINAMATH_CALUDE_cubic_with_repeated_root_l1017_101765


namespace NUMINAMATH_CALUDE_product_of_decimals_l1017_101772

theorem product_of_decimals : (0.5 : ℝ) * 0.3 = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1017_101772


namespace NUMINAMATH_CALUDE_alloy_mixture_l1017_101755

/-- Given two alloys with metal ratios m:n and p:q respectively, 
    this theorem proves the amounts of each alloy needed to create 1 kg 
    of a new alloy with equal parts of both metals. -/
theorem alloy_mixture (m n p q : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hq : q > 0) :
  let x := (1 : ℝ) / 2 + (m * p - n * q) / (2 * (n * p - m * q))
  x * (n / (m + n)) + (1 - x) * (p / (p + q)) = 
  x * (m / (m + n)) + (1 - x) * (q / (p + q)) :=
by sorry

end NUMINAMATH_CALUDE_alloy_mixture_l1017_101755


namespace NUMINAMATH_CALUDE_fruit_purchase_total_l1017_101748

/-- Calculates the total amount paid for fruits after discounts --/
def total_amount_paid (peach_price apple_price orange_price : ℚ)
                      (peach_count apple_count orange_count : ℕ)
                      (peach_discount apple_discount orange_discount : ℚ)
                      (peach_discount_threshold apple_discount_threshold orange_discount_threshold : ℚ) : ℚ :=
  let peach_total := peach_price * peach_count
  let apple_total := apple_price * apple_count
  let orange_total := orange_price * orange_count
  let peach_discount_applied := (peach_total / peach_discount_threshold).floor * peach_discount
  let apple_discount_applied := (apple_total / apple_discount_threshold).floor * apple_discount
  let orange_discount_applied := (orange_total / orange_discount_threshold).floor * orange_discount
  peach_total + apple_total + orange_total - peach_discount_applied - apple_discount_applied - orange_discount_applied

/-- Theorem stating the total amount paid for fruits after discounts --/
theorem fruit_purchase_total :
  total_amount_paid (40/100) (60/100) (50/100) 400 150 200 2 3 (3/2) 10 15 7 = 279 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_total_l1017_101748


namespace NUMINAMATH_CALUDE_division_theorem_specific_case_l1017_101706

theorem division_theorem_specific_case :
  ∀ (D d Q R : ℕ),
    D = d * Q + R →
    d * Q = 135 →
    R = 2 * d →
    R < d →
    Q > 0 →
    D = 165 ∧ d = 15 ∧ Q = 9 ∧ R = 30 :=
by sorry

end NUMINAMATH_CALUDE_division_theorem_specific_case_l1017_101706


namespace NUMINAMATH_CALUDE_minimum_red_chips_l1017_101733

theorem minimum_red_chips 
  (w b r : ℕ) 
  (blue_white : b ≥ w / 4)
  (blue_red : b ≤ r / 6)
  (white_blue_total : w + b ≥ 75) :
  r ≥ 90 ∧ ∀ r', (∃ w' b', 
    b' ≥ w' / 4 ∧ 
    b' ≤ r' / 6 ∧ 
    w' + b' ≥ 75 ∧ 
    r' < 90) → False :=
sorry

end NUMINAMATH_CALUDE_minimum_red_chips_l1017_101733
