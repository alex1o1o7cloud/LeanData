import Mathlib

namespace NUMINAMATH_CALUDE_fraction_simplification_l1704_170427

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1704_170427


namespace NUMINAMATH_CALUDE_quarter_squared_decimal_l1704_170452

theorem quarter_squared_decimal : (1 / 4 : ℚ) ^ 2 = 0.0625 := by
  sorry

end NUMINAMATH_CALUDE_quarter_squared_decimal_l1704_170452


namespace NUMINAMATH_CALUDE_inequalities_proof_l1704_170472

theorem inequalities_proof (a b : ℝ) (h : a + b > 0) : 
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1704_170472


namespace NUMINAMATH_CALUDE_simplify_expression_l1704_170435

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^3 + b^2) - 2 * b^3 = 9 * b^4 + b^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1704_170435


namespace NUMINAMATH_CALUDE_table_tennis_pairing_methods_l1704_170424

theorem table_tennis_pairing_methods (total_players : Nat) (male_players : Nat) (female_players : Nat) :
  total_players = male_players + female_players →
  male_players = 5 →
  female_players = 4 →
  (Nat.choose male_players 2) * (Nat.choose female_players 2) * 2 = 120 :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_pairing_methods_l1704_170424


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1704_170419

/-- Given a system of equations, prove that the maximum value of a^2 + b^2 + c^2 + d^2 is 82 -/
theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 16)
  (h2 : a * b + c + d = 81)
  (h3 : a * d + b * c = 168)
  (h4 : c * d = 100) :
  ∀ (w x y z : ℝ), 
  (w + x = 16) → 
  (w * x + y + z = 81) → 
  (w * z + x * y = 168) → 
  (y * z = 100) → 
  a^2 + b^2 + c^2 + d^2 ≥ w^2 + x^2 + y^2 + z^2 ∧
  a^2 + b^2 + c^2 + d^2 ≤ 82 :=
by
  sorry

#check max_sum_of_squares

end NUMINAMATH_CALUDE_max_sum_of_squares_l1704_170419


namespace NUMINAMATH_CALUDE_equation_solution_l1704_170471

theorem equation_solution : 
  let x : ℚ := -7/6
  (3 - x) / (x + 2) + (3*x - 9) / (3 - x) = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1704_170471


namespace NUMINAMATH_CALUDE_ball_size_ratio_l1704_170456

/-- Given three balls A, B, and C with different sizes, where A is three times bigger than B,
    and B is half the size of C, prove that A is 1.5 times the size of C. -/
theorem ball_size_ratio :
  ∀ (size_A size_B size_C : ℝ),
  size_A > 0 → size_B > 0 → size_C > 0 →
  size_A = 3 * size_B →
  size_B = (1 / 2) * size_C →
  size_A = (3 / 2) * size_C :=
by
  sorry

end NUMINAMATH_CALUDE_ball_size_ratio_l1704_170456


namespace NUMINAMATH_CALUDE_prism_with_five_faces_has_nine_edges_l1704_170433

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  faces : ℕ
  edges : ℕ

/-- Theorem: A prism with 5 faces has 9 edges. -/
theorem prism_with_five_faces_has_nine_edges (p : Prism) (h : p.faces = 5) : p.edges = 9 := by
  sorry


end NUMINAMATH_CALUDE_prism_with_five_faces_has_nine_edges_l1704_170433


namespace NUMINAMATH_CALUDE_percentage_problem_l1704_170415

theorem percentage_problem : ∃ p : ℝ, p = 25 ∧ 0.15 * 40 - (p / 100) * 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1704_170415


namespace NUMINAMATH_CALUDE_inequality_proof_l1704_170497

theorem inequality_proof (m n : ℝ) (hm : m < 0) (hn : n > 0) (hmn : m + n < 0) :
  m < -n ∧ -n < n ∧ n < -m :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1704_170497


namespace NUMINAMATH_CALUDE_f_range_theorem_l1704_170499

noncomputable def f (x : ℝ) : ℝ := x + (Real.exp x)⁻¹

theorem f_range_theorem :
  {a : ℝ | ∀ x, f x > a * x} = Set.Ioo (1 - Real.exp 1) 1 :=
sorry

end NUMINAMATH_CALUDE_f_range_theorem_l1704_170499


namespace NUMINAMATH_CALUDE_periodic_function_theorem_l1704_170487

/-- A function f: ℝ → ℝ is periodic if there exists a positive real number p such that
    for all x ∈ ℝ, f(x + p) = f(x) -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x

/-- The main theorem: if f satisfies the given functional equation,
    then f is periodic with period 2a -/
theorem periodic_function_theorem (f : ℝ → ℝ) (a : ℝ) 
    (h : a > 0) 
    (eq : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)) :
  IsPeriodic f ∧ ∃ p : ℝ, p = 2 * a ∧ ∀ x : ℝ, f (x + p) = f x :=
by sorry

end NUMINAMATH_CALUDE_periodic_function_theorem_l1704_170487


namespace NUMINAMATH_CALUDE_b_oxen_count_l1704_170448

/-- Represents the number of oxen-months for a person's contribution -/
def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

/-- Represents the total rent of the pasture -/
def total_rent : ℕ := 175

/-- Represents A's contribution in oxen-months -/
def a_contribution : ℕ := oxen_months 10 7

/-- Represents C's contribution in oxen-months -/
def c_contribution : ℕ := oxen_months 15 3

/-- Represents C's share of the rent -/
def c_share : ℕ := 45

/-- Represents the number of months B's oxen grazed -/
def b_months : ℕ := 5

/-- Theorem stating that B put 12 oxen for grazing -/
theorem b_oxen_count : 
  ∃ (b_oxen : ℕ), 
    b_oxen = 12 ∧ 
    (c_share : ℚ) / total_rent = 
      (c_contribution : ℚ) / (a_contribution + oxen_months b_oxen b_months + c_contribution) :=
sorry

end NUMINAMATH_CALUDE_b_oxen_count_l1704_170448


namespace NUMINAMATH_CALUDE_max_value_abc_max_value_abc_achievable_l1704_170460

theorem max_value_abc (A B C : ℕ) (h : A + B + C = 15) :
  (A * B * C + A * B + B * C + C * A) ≤ 200 :=
by sorry

theorem max_value_abc_achievable :
  ∃ (A B C : ℕ), A + B + C = 15 ∧ A * B * C + A * B + B * C + C * A = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_max_value_abc_achievable_l1704_170460


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1704_170490

theorem arithmetic_calculation : 4 * 6 * 8 + 18 / 3^2 = 194 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1704_170490


namespace NUMINAMATH_CALUDE_inequality_proof_l1704_170466

theorem inequality_proof (a d b c : ℝ) 
  (h1 : a ≥ 0) (h2 : d ≥ 0) (h3 : b > 0) (h4 : c > 0) (h5 : b + c ≥ a + d) : 
  (b / (c + d)) + (c / (b + a)) ≥ Real.sqrt 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1704_170466


namespace NUMINAMATH_CALUDE_angle_measure_when_supplement_is_four_times_complement_l1704_170422

theorem angle_measure_when_supplement_is_four_times_complement :
  ∀ x : ℝ,
  (0 < x) →
  (x < 180) →
  (180 - x = 4 * (90 - x)) →
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_when_supplement_is_four_times_complement_l1704_170422


namespace NUMINAMATH_CALUDE_walking_time_difference_l1704_170444

/-- The speed of person A in km/h -/
def speed_A : ℝ := 4

/-- The speed of person B in km/h -/
def speed_B : ℝ := 4.555555555555555

/-- The time in hours after which B overtakes A -/
def overtake_time : ℝ := 1.8

/-- The time in hours after A started that B starts walking -/
def time_diff : ℝ := 0.25

theorem walking_time_difference :
  speed_A * (time_diff + overtake_time) = speed_B * overtake_time := by
  sorry

end NUMINAMATH_CALUDE_walking_time_difference_l1704_170444


namespace NUMINAMATH_CALUDE_multiply_and_add_l1704_170480

theorem multiply_and_add : 45 * 21 + 45 * 79 = 4500 := by sorry

end NUMINAMATH_CALUDE_multiply_and_add_l1704_170480


namespace NUMINAMATH_CALUDE_product_of_distinct_numbers_l1704_170459

theorem product_of_distinct_numbers (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_numbers_l1704_170459


namespace NUMINAMATH_CALUDE_existence_of_multiple_2002_l1704_170418

theorem existence_of_multiple_2002 (a : Fin 41 → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ i j m n p q : Fin 41, i ≠ j ∧ m ≠ n ∧ p ≠ q ∧
    i ≠ m ∧ i ≠ n ∧ i ≠ p ∧ i ≠ q ∧
    j ≠ m ∧ j ≠ n ∧ j ≠ p ∧ j ≠ q ∧
    m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧
    (2002 ∣ (a i - a j) * (a m - a n) * (a p - a q)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_multiple_2002_l1704_170418


namespace NUMINAMATH_CALUDE_small_pizza_has_eight_slices_l1704_170491

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := sorry

/-- The number of people -/
def num_people : ℕ := 3

/-- The number of slices each person can eat -/
def slices_per_person : ℕ := 12

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 14

/-- The number of small pizzas ordered -/
def num_small_pizzas : ℕ := 1

/-- The number of large pizzas ordered -/
def num_large_pizzas : ℕ := 2

theorem small_pizza_has_eight_slices :
  small_pizza_slices = 8 ∧
  num_people * slices_per_person ≤ 
    num_small_pizzas * small_pizza_slices + num_large_pizzas * large_pizza_slices :=
by sorry

end NUMINAMATH_CALUDE_small_pizza_has_eight_slices_l1704_170491


namespace NUMINAMATH_CALUDE_f_3_range_l1704_170436

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem f_3_range (a b : ℝ) :
  (-1 ≤ f a b 1 ∧ f a b 1 ≤ 2) →
  (1 ≤ f a b 2 ∧ f a b 2 ≤ 3) →
  -3 ≤ f a b 3 ∧ f a b 3 ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_f_3_range_l1704_170436


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1704_170449

/-- The quadratic function f(x) = x^2 - kx - 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x - 1

/-- The interval [1, 4] -/
def interval : Set ℝ := Set.Icc 1 4

theorem quadratic_function_properties (k : ℝ) :
  /- Part 1: Monotonicity condition -/
  (∀ x ∈ interval, ∀ y ∈ interval, x < y → (f k x < f k y ∨ f k x > f k y)) ↔ 
  (k ≤ 2 ∨ k ≥ 8) ∧
  
  /- Part 2: Minimum value -/
  (∀ x ∈ interval, f k x ≥ 
    (if k ≤ 2 then -k
     else if k < 8 then -k^2/4 - 1
     else 15 - 4*k)) ∧
  
  /- The minimum value is attained in the interval -/
  (∃ x ∈ interval, f k x = 
    (if k ≤ 2 then -k
     else if k < 8 then -k^2/4 - 1
     else 15 - 4*k)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1704_170449


namespace NUMINAMATH_CALUDE_binary_ones_divisibility_l1704_170496

/-- The number of ones in the binary representation of a natural number -/
def binary_ones (n : ℕ) : ℕ := sorry

theorem binary_ones_divisibility (n : ℕ) (h : binary_ones n = 1995) :
  ∃ k : ℕ, n! = k * 2^(n - 1995) :=
sorry

end NUMINAMATH_CALUDE_binary_ones_divisibility_l1704_170496


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l1704_170405

theorem quadratic_perfect_square (x : ℝ) : 
  (∃ a : ℝ, x^2 + 10*x + 25 = (x + a)^2) ∧ 
  (∀ c : ℝ, c ≠ 25 → ¬∃ a : ℝ, x^2 + 10*x + c = (x + a)^2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l1704_170405


namespace NUMINAMATH_CALUDE_three_possible_values_for_d_l1704_170451

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Represents the equation AABC + CBBA = DCCD -/
def satisfies_equation (a b c d : Digit) : Prop :=
  1000 * a.val + 100 * a.val + 10 * b.val + c.val +
  1000 * c.val + 100 * b.val + 10 * b.val + a.val =
  1000 * d.val + 100 * c.val + 10 * c.val + d.val

/-- The main theorem stating there are exactly 3 possible values for D -/
theorem three_possible_values_for_d :
  ∃ (s : Finset Digit),
    s.card = 3 ∧
    (∀ d : Digit, d ∈ s ↔ 
      ∃ (a b c : Digit), distinct a b c d ∧ satisfies_equation a b c d) :=
sorry

end NUMINAMATH_CALUDE_three_possible_values_for_d_l1704_170451


namespace NUMINAMATH_CALUDE_sample_probability_l1704_170454

/-- Simple random sampling with given conditions -/
def SimpleRandomSampling (n : ℕ) : Prop :=
  n > 0 ∧ 
  (1 : ℚ) / n = 1 / 8 ∧
  ∀ i : ℕ, i ≤ n → (1 - (1 : ℚ) / n)^3 = (n - 1 : ℚ)^3 / n^3

theorem sample_probability (n : ℕ) (h : SimpleRandomSampling n) : 
  n = 8 ∧ (1 - (7 : ℚ) / 8^3) = 169 / 512 := by
  sorry

#check sample_probability

end NUMINAMATH_CALUDE_sample_probability_l1704_170454


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_plus_constant_l1704_170408

/-- For any positive real number a, the function f(x) = a^x + 4 passes through the point (0, 5) -/
theorem fixed_point_of_exponential_plus_constant (a : ℝ) (ha : a > 0) :
  let f := fun (x : ℝ) => a^x + 4
  f 0 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_plus_constant_l1704_170408


namespace NUMINAMATH_CALUDE_one_more_bird_than_storks_l1704_170421

/-- Given a fence with birds and storks, calculate the difference between the number of birds and storks -/
def bird_stork_difference (num_birds : ℕ) (num_storks : ℕ) : ℤ :=
  (num_birds : ℤ) - (num_storks : ℤ)

/-- Theorem: On a fence with 6 birds and 5 storks, there is 1 more bird than storks -/
theorem one_more_bird_than_storks :
  bird_stork_difference 6 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_more_bird_than_storks_l1704_170421


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1704_170407

/-- Given a workshop with workers, prove that the average salary of all workers is 8000 Rs. -/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (tech_salary : ℕ)
  (non_tech_salary : ℕ)
  (h1 : total_workers = 21)
  (h2 : technicians = 7)
  (h3 : tech_salary = 12000)
  (h4 : non_tech_salary = 6000) :
  (technicians * tech_salary + (total_workers - technicians) * non_tech_salary) / total_workers = 8000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l1704_170407


namespace NUMINAMATH_CALUDE_triangle_point_distance_l1704_170485

/-- Given a triangle ABC with AB = 8, BC = 20, CA = 16, and points D and E on BC
    such that CD = 8 and ∠BAE = ∠CAD, prove that BE = 2 -/
theorem triangle_point_distance (A B C D E : ℝ × ℝ) : 
  let dist := (fun (P Q : ℝ × ℝ) => Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))
  let angle := (fun (P Q R : ℝ × ℝ) => Real.arccos (((Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2)) / 
                (dist P Q * dist P R)))
  dist A B = 8 →
  dist B C = 20 →
  dist C A = 16 →
  D.1 = B.1 + 12 / 20 * (C.1 - B.1) ∧ D.2 = B.2 + 12 / 20 * (C.2 - B.2) →
  angle B A E = angle C A D →
  dist B E = 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_point_distance_l1704_170485


namespace NUMINAMATH_CALUDE_intersection_complement_M_and_N_l1704_170455

def U : Set ℕ := {x | x > 0 ∧ x < 9}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5, 6}

theorem intersection_complement_M_and_N :
  (U \ M) ∩ N = {4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_M_and_N_l1704_170455


namespace NUMINAMATH_CALUDE_number_of_arrangements_l1704_170443

/-- Represents a person in the group photo --/
inductive Person
  | StudentA
  | StudentB
  | StudentC
  | StudentD
  | StudentE
  | TeacherX
  | TeacherY

/-- Represents a valid arrangement of people in the group photo --/
def ValidArrangement : Type := List Person

/-- Checks if students A, B, and C are standing together in the arrangement --/
def studentsABCTogether (arrangement : ValidArrangement) : Prop := sorry

/-- Checks if teachers X and Y are not standing next to each other in the arrangement --/
def teachersNotAdjacent (arrangement : ValidArrangement) : Prop := sorry

/-- The set of all valid arrangements satisfying the given conditions --/
def validArrangements : Set ValidArrangement :=
  {arrangement | studentsABCTogether arrangement ∧ teachersNotAdjacent arrangement}

/-- The main theorem stating that the number of valid arrangements is 504 --/
theorem number_of_arrangements (h : Fintype validArrangements) :
  Fintype.card validArrangements = 504 := by sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l1704_170443


namespace NUMINAMATH_CALUDE_distance_to_school_l1704_170420

theorem distance_to_school (normal_time normal_speed light_time : ℚ) 
  (h1 : normal_time = 20 / 60)
  (h2 : light_time = 10 / 60)
  (h3 : normal_time * normal_speed = light_time * (normal_speed + 15)) :
  normal_time * normal_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_school_l1704_170420


namespace NUMINAMATH_CALUDE_assignFourFromTwentyFive_eq_303600_l1704_170450

/-- The number of ways to select and assign 4 people from a group of 25 to 4 distinct positions -/
def assignFourFromTwentyFive : ℕ := 25 * 24 * 23 * 22

/-- Theorem stating that the number of ways to select and assign 4 people from a group of 25 to 4 distinct positions is 303600 -/
theorem assignFourFromTwentyFive_eq_303600 : assignFourFromTwentyFive = 303600 := by
  sorry

end NUMINAMATH_CALUDE_assignFourFromTwentyFive_eq_303600_l1704_170450


namespace NUMINAMATH_CALUDE_money_needed_for_perfume_l1704_170486

def perfume_cost : ℕ := 50
def christian_initial : ℕ := 5
def sue_initial : ℕ := 7
def yards_mowed : ℕ := 4
def yard_charge : ℕ := 5
def dogs_walked : ℕ := 6
def dog_charge : ℕ := 2

theorem money_needed_for_perfume :
  perfume_cost - (christian_initial + sue_initial + yards_mowed * yard_charge + dogs_walked * dog_charge) = 6 := by
  sorry

end NUMINAMATH_CALUDE_money_needed_for_perfume_l1704_170486


namespace NUMINAMATH_CALUDE_total_bales_in_barn_l1704_170445

def initial_bales : ℕ := 54
def added_bales : ℕ := 28

theorem total_bales_in_barn : initial_bales + added_bales = 82 := by
  sorry

end NUMINAMATH_CALUDE_total_bales_in_barn_l1704_170445


namespace NUMINAMATH_CALUDE_sum_minimized_at_6_l1704_170464

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = -11
  sum_of_4th_and_6th : a 4 + a 6 = -6

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

/-- The value of n that minimizes the sum of first n terms -/
def minimizing_n (seq : ArithmeticSequence) : ℕ :=
  6

theorem sum_minimized_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, sum_n_terms seq (minimizing_n seq) ≤ sum_n_terms seq n :=
sorry

end NUMINAMATH_CALUDE_sum_minimized_at_6_l1704_170464


namespace NUMINAMATH_CALUDE_train_length_l1704_170409

theorem train_length (tree_time : ℝ) (platform_time : ℝ) (platform_length : ℝ) :
  tree_time = 120 →
  platform_time = 220 →
  platform_length = 1000 →
  let train_length := (platform_time * platform_length) / (platform_time - tree_time)
  train_length = 1200 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1704_170409


namespace NUMINAMATH_CALUDE_uncle_omar_parking_probability_l1704_170473

/-- The number of parking spaces -/
def total_spaces : ℕ := 18

/-- The number of cars already parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent empty spaces needed -/
def needed_spaces : ℕ := 2

/-- The probability of finding the required adjacent empty spaces -/
def parking_probability : ℚ := 16/51

theorem uncle_omar_parking_probability :
  (1 : ℚ) - (Nat.choose (total_spaces - needed_spaces + 1) parked_cars : ℚ) / 
  (Nat.choose total_spaces parked_cars : ℚ) = parking_probability := by
  sorry

end NUMINAMATH_CALUDE_uncle_omar_parking_probability_l1704_170473


namespace NUMINAMATH_CALUDE_circle_center_is_3_0_l1704_170477

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in standard form -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The given circle equation -/
def given_circle_equation (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 9

theorem circle_center_is_3_0 :
  ∃ (c : Circle), (∀ x y : ℝ, circle_equation c x y ↔ given_circle_equation x y) ∧ c.center = (3, 0) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_is_3_0_l1704_170477


namespace NUMINAMATH_CALUDE_product_of_square_roots_l1704_170447

theorem product_of_square_roots (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^3) * Real.sqrt (20 * p) * Real.sqrt (8 * p^5) = 20 * p^4 * Real.sqrt (6 * p) := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l1704_170447


namespace NUMINAMATH_CALUDE_tonya_needs_22_hamburgers_l1704_170483

/-- The weight of each hamburger in ounces -/
def hamburger_weight : ℕ := 4

/-- The total weight in ounces eaten by last year's winner -/
def last_year_winner_weight : ℕ := 84

/-- The number of hamburgers Tonya needs to eat to beat last year's winner -/
def hamburgers_to_beat_record : ℕ := 
  (last_year_winner_weight / hamburger_weight) + 1

/-- Theorem stating that Tonya needs to eat 22 hamburgers to beat last year's record -/
theorem tonya_needs_22_hamburgers : 
  hamburgers_to_beat_record = 22 := by sorry

end NUMINAMATH_CALUDE_tonya_needs_22_hamburgers_l1704_170483


namespace NUMINAMATH_CALUDE_greatest_difference_units_digit_l1704_170481

/-- Given a three-digit integer in the form 72x that is a multiple of 3,
    the greatest possible difference between two possibilities for the units digit is 9. -/
theorem greatest_difference_units_digit :
  ∀ x : ℕ,
  x < 10 →
  (720 + x) % 3 = 0 →
  ∃ y z : ℕ,
  y < 10 ∧ z < 10 ∧
  (720 + y) % 3 = 0 ∧
  (720 + z) % 3 = 0 ∧
  y - z = 9 ∧
  ∀ a b : ℕ,
  a < 10 → b < 10 →
  (720 + a) % 3 = 0 →
  (720 + b) % 3 = 0 →
  a - b ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_difference_units_digit_l1704_170481


namespace NUMINAMATH_CALUDE_min_value_expression_l1704_170425

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (m : ℝ), m = 2 ∧ (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x^2 + y^2 + 1/x^2 + 2*y/x ≥ m) ∧
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x^2 + y^2 + 1/x^2 + 2*y/x = m) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1704_170425


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l1704_170495

/-- Given two lines l₁ and l₂ with equations 3x + 2y - 2 = 0 and (2m-1)x + my + 1 = 0 respectively,
    if l₁ is parallel to l₂, then m = 2. -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, 3 * x + 2 * y - 2 = 0 ↔ (2 * m - 1) * x + m * y + 1 = 0) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l1704_170495


namespace NUMINAMATH_CALUDE_cube_sum_minus_product_2003_l1704_170489

theorem cube_sum_minus_product_2003 :
  {(x, y, z) : ℤ × ℤ × ℤ | x^3 + y^3 + z^3 - 3*x*y*z = 2003} =
  {(668, 668, 667), (668, 667, 668), (667, 668, 668)} := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_minus_product_2003_l1704_170489


namespace NUMINAMATH_CALUDE_scientific_notation_75500000_l1704_170476

theorem scientific_notation_75500000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 75500000 = a * (10 : ℝ) ^ n ∧ a = 7.55 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_75500000_l1704_170476


namespace NUMINAMATH_CALUDE_train_distance_in_three_hours_l1704_170428

-- Define the train's speed
def train_speed : ℚ := 1 / 2

-- Define the duration in hours
def duration : ℚ := 3

-- Define the number of minutes in an hour
def minutes_per_hour : ℚ := 60

-- Theorem statement
theorem train_distance_in_three_hours :
  train_speed * minutes_per_hour * duration = 90 := by
  sorry


end NUMINAMATH_CALUDE_train_distance_in_three_hours_l1704_170428


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1704_170469

/-- A quadratic function with positive leading coefficient and symmetry about x = 2 -/
def symmetric_quadratic (a b c : ℝ) : ℝ → ℝ :=
  fun x ↦ a * x^2 + b * x + c

theorem quadratic_inequality 
  (a b c : ℝ) 
  (ha : a > 0)
  (h_sym : ∀ x, symmetric_quadratic a b c (x + 2) = symmetric_quadratic a b c (2 - x)) :
  symmetric_quadratic a b c (Real.sqrt 2 / 2) > symmetric_quadratic a b c Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1704_170469


namespace NUMINAMATH_CALUDE_problem_solution_l1704_170462

theorem problem_solution (x y : ℤ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : x + y + x * y = 119) : 
  y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1704_170462


namespace NUMINAMATH_CALUDE_measure_gold_dust_l1704_170494

/-- Represents the available weights for measuring gold dust -/
inductive Weight
  | TwoHundredGram
  | FiftyGram

/-- Represents a case with different available weights -/
inductive Case
  | CaseA
  | CaseB

/-- Represents a weighing operation on a balance scale -/
def Weighing := ℝ → ℝ → Prop

/-- Represents the ability to measure a specific amount of gold dust -/
def CanMeasure (totalGold : ℝ) (targetAmount : ℝ) (weights : List Weight) (case : Case) : Prop :=
  ∃ (w1 w2 w3 : Weighing), 
    (w1 totalGold targetAmount) ∧ 
    (w2 totalGold targetAmount) ∧ 
    (w3 totalGold targetAmount)

/-- The main theorem stating that it's possible to measure 2 kg of gold dust in both cases -/
theorem measure_gold_dust : 
  ∀ (case : Case),
    CanMeasure 9 2 
      (match case with
        | Case.CaseA => [Weight.TwoHundredGram, Weight.FiftyGram]
        | Case.CaseB => [Weight.TwoHundredGram])
      case :=
by
  sorry

end NUMINAMATH_CALUDE_measure_gold_dust_l1704_170494


namespace NUMINAMATH_CALUDE_auntie_em_parking_probability_l1704_170404

def total_spaces : ℕ := 18
def parked_cars : ℕ := 12
def suv_spaces : ℕ := 2

theorem auntie_em_parking_probability :
  let total_configurations := Nat.choose total_spaces parked_cars
  let unfavorable_configurations := Nat.choose (parked_cars + 1) parked_cars
  (total_configurations - unfavorable_configurations : ℚ) / total_configurations = 1403 / 1546 :=
by sorry

end NUMINAMATH_CALUDE_auntie_em_parking_probability_l1704_170404


namespace NUMINAMATH_CALUDE_population_problem_l1704_170479

theorem population_problem : ∃ (n : ℕ), 
  (∃ (m k : ℕ), 
    (n^2 + 200 = m^2 + 1) ∧ 
    (n^2 + 500 = k^2) ∧ 
    (21 ∣ n^2) ∧ 
    (n^2 = 9801)) := by
  sorry

end NUMINAMATH_CALUDE_population_problem_l1704_170479


namespace NUMINAMATH_CALUDE_total_money_proof_l1704_170429

/-- The amount of money Beth currently has -/
def beth_money : ℕ := 70

/-- The amount of money Jan currently has -/
def jan_money : ℕ := 80

/-- The amount of money Tom currently has -/
def tom_money : ℕ := 210

theorem total_money_proof :
  (beth_money + 35 = 105) ∧
  (jan_money - 10 = beth_money) ∧
  (tom_money = 3 * (jan_money - 10)) →
  beth_money + jan_money + tom_money = 360 := by
sorry

end NUMINAMATH_CALUDE_total_money_proof_l1704_170429


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1704_170406

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 0 ∧ d ≤ 9 ∧ d % 3 ≠ 0 ∧ d % 7 ≠ 0

def has_valid_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_valid_digit d

theorem no_integer_solutions (p : ℕ) (hp : Prime p) (hp_gt : p > 5) (hp_digits : has_valid_digits p) :
  ¬∃ (x y : ℤ), x^4 + p = 3 * y^4 :=
sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1704_170406


namespace NUMINAMATH_CALUDE_integer_power_sum_l1704_170434

theorem integer_power_sum (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) :
  ∀ n : ℕ, ∃ m : ℤ, a^n + 1/(a^n) = m :=
sorry

end NUMINAMATH_CALUDE_integer_power_sum_l1704_170434


namespace NUMINAMATH_CALUDE_more_ones_than_zeros_mod_500_l1704_170414

/-- The number of positive integers less than or equal to 500 whose binary 
    representation contains more 1's than 0's -/
def N : ℕ := sorry

/-- Function to count 1's in the binary representation of a natural number -/
def count_ones (n : ℕ) : ℕ := sorry

/-- Function to count 0's in the binary representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem more_ones_than_zeros_mod_500 :
  N % 500 = 305 :=
sorry

end NUMINAMATH_CALUDE_more_ones_than_zeros_mod_500_l1704_170414


namespace NUMINAMATH_CALUDE_largest_multiple_proof_l1704_170416

/-- The largest three-digit number that is divisible by 6, 5, 8, and 9 -/
def largest_multiple : ℕ := 720

theorem largest_multiple_proof :
  (∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 6 ∣ n ∧ 5 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n → n ≤ largest_multiple) ∧
  100 ≤ largest_multiple ∧
  largest_multiple < 1000 ∧
  6 ∣ largest_multiple ∧
  5 ∣ largest_multiple ∧
  8 ∣ largest_multiple ∧
  9 ∣ largest_multiple :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_proof_l1704_170416


namespace NUMINAMATH_CALUDE_circle_area_l1704_170410

theorem circle_area (d : ℝ) (A : ℝ) (π : ℝ) (h1 : d = 10) (h2 : π = Real.pi) :
  A = π * 25 → A = (π * d^2) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_l1704_170410


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_less_than_four_l1704_170470

-- Define the system of inequalities
def has_solution (m : ℝ) : Prop :=
  ∃ x : ℝ, (2 * x - 6 + m < 0) ∧ (4 * x - m > 0)

-- State the theorem
theorem inequality_solution_implies_m_less_than_four :
  ∀ m : ℝ, has_solution m → m < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_less_than_four_l1704_170470


namespace NUMINAMATH_CALUDE_room_width_proof_l1704_170467

/-- Given a rectangular room with length 5.5 meters, prove that its width is 4 meters
    when the cost of paving is 850 rupees per square meter and the total cost is 18,700 rupees. -/
theorem room_width_proof (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 5.5 →
  cost_per_sqm = 850 →
  total_cost = 18700 →
  total_cost / cost_per_sqm / length = 4 := by
sorry

end NUMINAMATH_CALUDE_room_width_proof_l1704_170467


namespace NUMINAMATH_CALUDE_cubic_sum_l1704_170498

theorem cubic_sum (a b c : ℝ) 
  (h1 : a + b + c = 8) 
  (h2 : a * b + a * c + b * c = 9) 
  (h3 : a * b * c = -18) : 
  a^3 + b^3 + c^3 = 242 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_l1704_170498


namespace NUMINAMATH_CALUDE_impossibility_of_triangle_formation_l1704_170482

theorem impossibility_of_triangle_formation (n : ℕ) (h : n = 10) :
  ∃ (segments : Fin n → ℝ),
    ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k →
      ¬(segments i + segments j > segments k ∧
        segments j + segments k > segments i ∧
        segments k + segments i > segments j) :=
by sorry

end NUMINAMATH_CALUDE_impossibility_of_triangle_formation_l1704_170482


namespace NUMINAMATH_CALUDE_dodecagon_square_area_ratio_l1704_170400

theorem dodecagon_square_area_ratio :
  ∀ (square_side : ℝ) (dodecagon_area : ℝ),
    square_side = 2 →
    dodecagon_area = 3 →
    ∃ (shaded_area : ℝ),
      shaded_area = (square_side^2 - dodecagon_area) / 4 ∧
      shaded_area / dodecagon_area = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_square_area_ratio_l1704_170400


namespace NUMINAMATH_CALUDE_blanch_snack_slices_l1704_170401

/-- Calculates the number of pizza slices Blanch took as a snack -/
def snack_slices (initial : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) (left : ℕ) : ℕ :=
  initial - breakfast - lunch - dinner - left

theorem blanch_snack_slices :
  snack_slices 15 4 2 5 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_blanch_snack_slices_l1704_170401


namespace NUMINAMATH_CALUDE_average_age_proof_l1704_170417

def luke_age : ℕ := 20
def years_future : ℕ := 8

theorem average_age_proof :
  let bernard_future_age := 3 * luke_age
  let bernard_current_age := bernard_future_age - years_future
  let average_age := (luke_age + bernard_current_age) / 2
  average_age = 36 := by sorry

end NUMINAMATH_CALUDE_average_age_proof_l1704_170417


namespace NUMINAMATH_CALUDE_painting_price_increase_l1704_170438

theorem painting_price_increase (X : ℝ) : 
  (1 + X / 100) * (1 - 0.15) = 1.02 → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_increase_l1704_170438


namespace NUMINAMATH_CALUDE_equation_solution_l1704_170465

theorem equation_solution :
  ∃! x : ℤ, 45 - (28 - (x - (15 - 17))) = 56 :=
by
  -- The unique solution is x = 19
  use 19
  constructor
  · -- Prove that x = 19 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l1704_170465


namespace NUMINAMATH_CALUDE_sum_fraction_equality_l1704_170432

theorem sum_fraction_equality (x y z : ℝ) (h : x + y + z = 1) :
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2*(x^2 + y^2 + z^2)) := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_equality_l1704_170432


namespace NUMINAMATH_CALUDE_race_distance_l1704_170412

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  time_A : ℝ
  time_B : ℝ
  speed_A : ℝ
  speed_B : ℝ

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  r.time_A = 33 ∧
  r.speed_A = r.distance / r.time_A ∧
  r.speed_B = (r.distance - 35) / r.time_A ∧
  r.speed_B = 35 / 7 ∧
  r.time_B = r.time_A + 7

/-- The theorem stating that the race distance is 200 meters -/
theorem race_distance (r : Race) (h : race_conditions r) : r.distance = 200 :=
sorry

end NUMINAMATH_CALUDE_race_distance_l1704_170412


namespace NUMINAMATH_CALUDE_kekai_money_left_l1704_170423

def garage_sale_problem (num_shirts num_pants : ℕ) (price_shirt price_pants : ℚ) : ℚ :=
  let total_earned := num_shirts * price_shirt + num_pants * price_pants
  let amount_to_parents := total_earned / 2
  total_earned - amount_to_parents

theorem kekai_money_left :
  garage_sale_problem 5 5 1 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_kekai_money_left_l1704_170423


namespace NUMINAMATH_CALUDE_change_in_responses_l1704_170403

theorem change_in_responses (initial_yes initial_no final_yes final_no : ℚ) 
  (h1 : initial_yes = 50 / 100)
  (h2 : initial_no = 50 / 100)
  (h3 : final_yes = 70 / 100)
  (h4 : final_no = 30 / 100)
  (h5 : initial_yes + initial_no = 1)
  (h6 : final_yes + final_no = 1) :
  ∃ (min_change max_change : ℚ),
    min_change ≥ 0 ∧
    max_change ≤ 1 ∧
    min_change ≤ max_change ∧
    max_change - min_change = 30 / 100 :=
by sorry

end NUMINAMATH_CALUDE_change_in_responses_l1704_170403


namespace NUMINAMATH_CALUDE_cake_recipe_flour_l1704_170411

/-- The number of cups of flour in a cake recipe -/
def recipe_flour (sugar_cups : ℕ) (flour_added : ℕ) (flour_remaining : ℕ) : ℕ :=
  flour_added + flour_remaining

theorem cake_recipe_flour :
  ∀ (sugar_cups : ℕ) (flour_added : ℕ) (flour_remaining : ℕ),
    sugar_cups = 2 →
    flour_added = 7 →
    flour_remaining = sugar_cups + 1 →
    recipe_flour sugar_cups flour_added flour_remaining = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_recipe_flour_l1704_170411


namespace NUMINAMATH_CALUDE_equation_solution_l1704_170457

theorem equation_solution (α β : ℝ) : 
  (∀ x : ℝ, x ≠ -β → x ≠ 30 → x ≠ 70 → 
    (x - α) / (x + β) = (x^2 + 120*x + 1575) / (x^2 - 144*x + 1050)) →
  α + β = 5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1704_170457


namespace NUMINAMATH_CALUDE_pharmaceutical_optimization_l1704_170446

/-- Calculates the minimum number of experiments required for the fractional method -/
def min_experiments (lower_temp upper_temp accuracy : ℝ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of experiments for the given conditions -/
theorem pharmaceutical_optimization :
  min_experiments 29 63 1 = 7 := by sorry

end NUMINAMATH_CALUDE_pharmaceutical_optimization_l1704_170446


namespace NUMINAMATH_CALUDE_lcm_12_18_l1704_170453

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l1704_170453


namespace NUMINAMATH_CALUDE_specific_pyramid_sphere_radius_l1704_170475

/-- Pyramid with equilateral triangular base -/
structure Pyramid :=
  (base_side : ℝ)
  (height : ℝ)

/-- The radius of the circumscribed sphere around the pyramid -/
def circumscribed_sphere_radius (p : Pyramid) : ℝ :=
  sorry

/-- Theorem: The radius of the circumscribed sphere for a specific pyramid -/
theorem specific_pyramid_sphere_radius :
  let p : Pyramid := { base_side := 6, height := 4 }
  circumscribed_sphere_radius p = 4 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_sphere_radius_l1704_170475


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_a_range_l1704_170431

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The theorem states that if f(x) is decreasing on (-∞, 4], then a < -5 -/
theorem quadratic_decreasing_implies_a_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) →
  a < -5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_a_range_l1704_170431


namespace NUMINAMATH_CALUDE_tank_emptied_in_two_minutes_l1704_170493

/-- Represents the state and properties of a water tank system -/
structure WaterTank where
  initial_fill : ℚ
  fill_rate : ℚ
  empty_rate : ℚ

/-- Calculates the time to empty or fill the tank completely -/
def time_to_complete (tank : WaterTank) : ℚ :=
  let combined_rate := tank.fill_rate - tank.empty_rate
  let amount_to_change := 1 - tank.initial_fill
  amount_to_change / (-combined_rate)

/-- Theorem stating that the tank will be emptied in 2 minutes -/
theorem tank_emptied_in_two_minutes :
  let tank := WaterTank.mk (1/5) (1/15) (1/6)
  time_to_complete tank = 2 := by
  sorry

end NUMINAMATH_CALUDE_tank_emptied_in_two_minutes_l1704_170493


namespace NUMINAMATH_CALUDE_circle_equation_for_given_points_l1704_170474

/-- Given two points P and Q in a 2D plane, this function returns the standard equation
    of the circle with diameter PQ as a function from ℝ × ℝ → Prop -/
def circle_equation (P Q : ℝ × ℝ) : (ℝ × ℝ → Prop) :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  let h := (x₁ + x₂) / 2
  let k := (y₁ + y₂) / 2
  let r := Real.sqrt (((x₂ - x₁)^2 + (y₂ - y₁)^2) / 4)
  fun (x, y) ↦ (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the standard equation of the circle with diameter PQ,
    where P(3,4) and Q(-5,6), is (x + 1)^2 + (y - 5)^2 = 17 -/
theorem circle_equation_for_given_points :
  circle_equation (3, 4) (-5, 6) = fun (x, y) ↦ (x + 1)^2 + (y - 5)^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_for_given_points_l1704_170474


namespace NUMINAMATH_CALUDE_find_divisor_l1704_170461

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 14698 →
  quotient = 89 →
  remainder = 14 →
  dividend = divisor * quotient + remainder →
  divisor = 165 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1704_170461


namespace NUMINAMATH_CALUDE_snake_owners_count_l1704_170484

/-- Represents the number of pet owners for different combinations of pets --/
structure PetOwners where
  total : Nat
  onlyDogs : Nat
  onlyCats : Nat
  onlyBirds : Nat
  onlySnakes : Nat
  catsAndDogs : Nat
  dogsAndBirds : Nat
  catsAndBirds : Nat
  catsAndSnakes : Nat
  dogsAndSnakes : Nat
  allCategories : Nat

/-- Calculates the total number of snake owners --/
def totalSnakeOwners (po : PetOwners) : Nat :=
  po.onlySnakes + po.catsAndSnakes + po.dogsAndSnakes + po.allCategories

/-- Theorem stating that the total number of snake owners is 25 --/
theorem snake_owners_count (po : PetOwners) 
  (h1 : po.total = 75)
  (h2 : po.onlyDogs = 20)
  (h3 : po.onlyCats = 15)
  (h4 : po.onlyBirds = 8)
  (h5 : po.onlySnakes = 10)
  (h6 : po.catsAndDogs = 5)
  (h7 : po.dogsAndBirds = 4)
  (h8 : po.catsAndBirds = 3)
  (h9 : po.catsAndSnakes = 7)
  (h10 : po.dogsAndSnakes = 6)
  (h11 : po.allCategories = 2) :
  totalSnakeOwners po = 25 := by
  sorry

end NUMINAMATH_CALUDE_snake_owners_count_l1704_170484


namespace NUMINAMATH_CALUDE_simplify_fraction_l1704_170468

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) :
  ((3 * a / (a^2 - 1) - 1 / (a - 1)) / ((2 * a - 1) / (a + 1))) = 1 / (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1704_170468


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l1704_170413

theorem meaningful_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = 2 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l1704_170413


namespace NUMINAMATH_CALUDE_roots_of_quadratic_sum_of_fourth_powers_l1704_170463

theorem roots_of_quadratic_sum_of_fourth_powers (α β : ℝ) : 
  α^2 - 2*α - 8 = 0 → β^2 - 2*β - 8 = 0 → 3*α^4 + 4*β^4 = 1232 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_sum_of_fourth_powers_l1704_170463


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1704_170426

-- Define the constraint set
def ConstraintSet : Set (ℝ × ℝ) :=
  {(x, y) | 8 * x - y ≤ 4 ∧ x + y ≥ -1 ∧ y ≤ 4 * x}

-- Define the objective function
def ObjectiveFunction (a b : ℝ) (p : ℝ × ℝ) : ℝ :=
  a * p.1 + b * p.2

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (p : ℝ × ℝ), p ∈ ConstraintSet ∧ 
   ∀ (q : ℝ × ℝ), q ∈ ConstraintSet → ObjectiveFunction a b q ≤ ObjectiveFunction a b p) →
  (∀ (p : ℝ × ℝ), p ∈ ConstraintSet → ObjectiveFunction a b p ≤ 2) →
  1/a + 1/b ≥ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1704_170426


namespace NUMINAMATH_CALUDE_range_of_m_l1704_170478

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Iic 1, f (x + m) ≤ -f x) → m ∈ Set.Ici (-2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1704_170478


namespace NUMINAMATH_CALUDE_house_size_problem_l1704_170488

theorem house_size_problem (sara_house nada_house : ℝ) : 
  sara_house = 1000 ∧ 
  sara_house = 2 * nada_house + 100 → 
  nada_house = 450 := by
sorry

end NUMINAMATH_CALUDE_house_size_problem_l1704_170488


namespace NUMINAMATH_CALUDE_smallest_n_theorem_l1704_170439

/-- The smallest positive integer n for which the equation 15x^2 - nx + 630 = 0 has integral solutions -/
def smallest_n : ℕ := 195

/-- The equation 15x^2 - nx + 630 = 0 has integral solutions -/
def has_integral_solutions (n : ℕ) : Prop :=
  ∃ x : ℤ, 15 * x^2 - n * x + 630 = 0

theorem smallest_n_theorem :
  (has_integral_solutions smallest_n) ∧
  (∀ m : ℕ, m < smallest_n → ¬(has_integral_solutions m)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_theorem_l1704_170439


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1704_170492

open Set

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1704_170492


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_five_l1704_170430

theorem three_digit_divisible_by_five (n : ℕ) :
  300 ≤ n ∧ n < 400 →
  (n % 5 = 0 ↔ n % 100 = 5 ∧ n / 100 = 3) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_five_l1704_170430


namespace NUMINAMATH_CALUDE_trig_identity_l1704_170441

theorem trig_identity (α : ℝ) : 
  (1 - 2 * Real.sin (2 * α) ^ 2) / (1 - Real.sin (4 * α)) = 
  (1 + Real.tan (2 * α)) / (1 - Real.tan (2 * α)) := by sorry

end NUMINAMATH_CALUDE_trig_identity_l1704_170441


namespace NUMINAMATH_CALUDE_basketball_purchase_theorem_l1704_170442

/-- Represents the prices and quantities of basketballs --/
structure BasketballPurchase where
  priceA : ℕ  -- Price of brand A basketball
  priceB : ℕ  -- Price of brand B basketball
  quantityA : ℕ  -- Quantity of brand A basketballs
  quantityB : ℕ  -- Quantity of brand B basketballs

/-- Represents the conditions of the basketball purchase problem --/
def BasketballProblem (p : BasketballPurchase) : Prop :=
  p.priceB = p.priceA + 40 ∧
  4800 / p.priceA = (3/2) * (4000 / p.priceB) ∧
  p.quantityA + p.quantityB = 90 ∧
  p.quantityB ≥ 2 * p.quantityA ∧
  p.priceA * p.quantityA + p.priceB * p.quantityB ≤ 17200

/-- The theorem to be proved --/
theorem basketball_purchase_theorem (p : BasketballPurchase) 
  (h : BasketballProblem p) : 
  p.priceA = 160 ∧ 
  p.priceB = 200 ∧ 
  (∃ n : ℕ, n = 11 ∧ 
    ∀ m : ℕ, (20 ≤ m ∧ m ≤ 30) ↔ 
      BasketballProblem ⟨p.priceA, p.priceB, m, 90 - m⟩) ∧
  (∀ a : ℕ, 30 < a ∧ a < 50 → 
    (a < 40 → p.quantityA = 30) ∧ 
    (a > 40 → p.quantityA = 20)) :=
sorry


end NUMINAMATH_CALUDE_basketball_purchase_theorem_l1704_170442


namespace NUMINAMATH_CALUDE_river_speed_is_two_l1704_170402

/-- The speed of the river that satisfies the given conditions -/
def river_speed (mans_speed : ℝ) (distance : ℝ) (total_time : ℝ) : ℝ :=
  2

/-- Theorem stating that the river speed is 2 kmph given the conditions -/
theorem river_speed_is_two :
  let mans_speed : ℝ := 4
  let distance : ℝ := 2.25
  let total_time : ℝ := 1.5
  river_speed mans_speed distance total_time = 2 := by
  sorry

#check river_speed_is_two

end NUMINAMATH_CALUDE_river_speed_is_two_l1704_170402


namespace NUMINAMATH_CALUDE_problem_statement_l1704_170440

/-- Given a function f(x) = -ax^5 - x^3 + bx - 7 where f(2) = -9, prove that f(-2) = -5 -/
theorem problem_statement (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = -a * x^5 - x^3 + b * x - 7)
  (h2 : f 2 = -9) : 
  f (-2) = -5 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1704_170440


namespace NUMINAMATH_CALUDE_apple_orange_cost_l1704_170458

/-- The cost of oranges and apples in two scenarios -/
theorem apple_orange_cost (orange_cost apple_cost : ℝ) : 
  orange_cost = 29 →
  apple_cost = 29 →
  6 * orange_cost + 8 * apple_cost = 419 →
  5 * orange_cost + 7 * apple_cost = 488 →
  8 = ⌊(419 - 6 * orange_cost) / apple_cost⌋ := by
  sorry

end NUMINAMATH_CALUDE_apple_orange_cost_l1704_170458


namespace NUMINAMATH_CALUDE_intern_distribution_l1704_170437

/-- The number of ways to distribute n intern teachers to k freshman classes,
    with each class having at least 1 intern -/
def distribution_plans (n k : ℕ) : ℕ :=
  if n ≥ k then (n - k + 1) else 0

/-- Theorem: There are 4 ways to distribute 5 intern teachers to 4 freshman classes,
    with each class having at least 1 intern -/
theorem intern_distribution : distribution_plans 5 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_intern_distribution_l1704_170437
