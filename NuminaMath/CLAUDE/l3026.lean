import Mathlib

namespace problem_solution_l3026_302652

theorem problem_solution (a b : ℤ) 
  (h1 : 3015 * a + 3019 * b = 3023)
  (h2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := by
  sorry

end problem_solution_l3026_302652


namespace expected_distinct_faces_formula_l3026_302678

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of times the die is rolled -/
def numRolls : ℕ := 6

/-- The probability that a specific face does not appear in any of the rolls -/
def probNoAppearance : ℚ := (numFaces - 1 : ℚ) / numFaces ^ numRolls

/-- The expected number of distinct faces that appear when a die is rolled multiple times -/
def expectedDistinctFaces : ℚ := numFaces * (1 - probNoAppearance)

/-- Theorem: The expected number of distinct faces that appear when a die is rolled six times 
    is equal to (6^6 - 5^6) / 6^5 -/
theorem expected_distinct_faces_formula : 
  expectedDistinctFaces = (numFaces ^ numRolls - (numFaces - 1) ^ numRolls : ℚ) / numFaces ^ (numRolls - 1) := by
  sorry

end expected_distinct_faces_formula_l3026_302678


namespace problem_1_l3026_302637

theorem problem_1 (a : ℝ) : a^3 * a + (2*a^2)^2 = 5*a^4 := by
  sorry

end problem_1_l3026_302637


namespace perimeter_triangle_pst_l3026_302656

/-- Given a triangle PQR with points S on PQ, T on PR, and U on ST, 
    prove that the perimeter of triangle PST is 36 under specific conditions. -/
theorem perimeter_triangle_pst (P Q R S T U : ℝ × ℝ) : 
  dist P Q = 19 →
  dist Q R = 18 →
  dist P R = 17 →
  ∃ t₁ : ℝ, S = (1 - t₁) • P + t₁ • Q →
  ∃ t₂ : ℝ, T = (1 - t₂) • P + t₂ • R →
  ∃ t₃ : ℝ, U = (1 - t₃) • S + t₃ • T →
  dist Q S = dist S U →
  dist U T = dist T R →
  dist P S + dist S T + dist P T = 36 :=
sorry

end perimeter_triangle_pst_l3026_302656


namespace coefficient_x2y1_is_60_l3026_302680

/-- The coefficient of x^m y^n in the expansion of (1+x)^6(1+y)^4 -/
def f (m n : ℕ) : ℕ := Nat.choose 6 m * Nat.choose 4 n

/-- The theorem stating that the coefficient of x^2y^1 in the expansion of (1+x)^6(1+y)^4 is 60 -/
theorem coefficient_x2y1_is_60 : f 2 1 = 60 := by
  sorry

end coefficient_x2y1_is_60_l3026_302680


namespace ice_cream_bill_l3026_302641

/-- The cost of ice cream scoops for Pierre and his mom -/
theorem ice_cream_bill (cost_per_scoop : ℕ) (pierre_scoops : ℕ) (mom_scoops : ℕ) :
  cost_per_scoop = 2 → pierre_scoops = 3 → mom_scoops = 4 →
  cost_per_scoop * (pierre_scoops + mom_scoops) = 14 :=
by sorry

end ice_cream_bill_l3026_302641


namespace carol_first_six_probability_l3026_302690

/-- The probability of rolling a 6 on any single roll. -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on any single roll. -/
def prob_not_six : ℚ := 1 - prob_six

/-- The sequence of rolls, where 0 represents Alice, 1 represents Bob, and 2 represents Carol. -/
def roll_sequence : ℕ → Fin 3
  | n => n % 3

/-- The probability that Carol is the first to roll a 6. -/
def prob_carol_first_six : ℚ :=
  let a : ℚ := prob_not_six ^ 2 * prob_six
  let r : ℚ := prob_not_six ^ 3
  a / (1 - r)

theorem carol_first_six_probability :
  prob_carol_first_six = 25 / 91 := by
  sorry

end carol_first_six_probability_l3026_302690


namespace max_area_inscribed_rectangle_l3026_302627

/-- The maximum area of a rectangle inscribed in a circular segment -/
theorem max_area_inscribed_rectangle (r : ℝ) (α : ℝ) (h : 0 < α ∧ α ≤ π / 2) :
  ∃ (T_max : ℝ), T_max = (r^2 / 8) * (-3 * Real.cos α + Real.sqrt (8 + Real.cos α ^ 2)) *
    Real.sqrt (8 - 2 * Real.cos α ^ 2 - 2 * Real.cos α * Real.sqrt (8 + Real.cos α ^ 2)) ∧
  ∀ (T : ℝ), T ≤ T_max := by
  sorry

end max_area_inscribed_rectangle_l3026_302627


namespace unique_prime_digit_product_l3026_302691

def is_prime_digit (d : Nat) : Prop :=
  d ∈ [2, 3, 5, 7]

def all_prime_digits (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_prime_digit d

theorem unique_prime_digit_product : 
  ∃! (a b : Nat), 
    100 ≤ a ∧ a < 1000 ∧
    10 ≤ b ∧ b < 100 ∧
    all_prime_digits a ∧
    all_prime_digits b ∧
    1000 ≤ a * b ∧ a * b < 10000 ∧
    all_prime_digits (a * b) ∧
    a = 775 ∧ b = 33 :=
by sorry

end unique_prime_digit_product_l3026_302691


namespace system_solution_l3026_302665

theorem system_solution : 
  ∃ (x y : ℝ), 
    (6.751 * x + 3.249 * y = 26.751) ∧ 
    (3.249 * x + 6.751 * y = 23.249) ∧ 
    (x = 3) ∧ 
    (y = 2) := by
  sorry

end system_solution_l3026_302665


namespace pacific_ocean_area_rounded_l3026_302692

/-- Rounds a number to the nearest multiple of 10000 -/
def roundToNearestTenThousand (n : ℕ) : ℕ :=
  ((n + 5000) / 10000) * 10000

theorem pacific_ocean_area_rounded :
  roundToNearestTenThousand 17996800 = 18000000 := by sorry

end pacific_ocean_area_rounded_l3026_302692


namespace intersection_of_S_and_T_l3026_302686

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | x^2 + 2*x = 0}
def T : Set ℝ := {x : ℝ | x^2 - 2*x = 0}

-- State the theorem
theorem intersection_of_S_and_T : S ∩ T = {0} := by
  sorry

end intersection_of_S_and_T_l3026_302686


namespace ten_hash_four_l3026_302653

/-- Operation # defined on real numbers -/
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

/-- Properties of the hash operation -/
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 2

/-- Theorem stating that 10 # 4 = 58 -/
theorem ten_hash_four : hash 10 4 = 58 := by
  sorry

end ten_hash_four_l3026_302653


namespace original_number_is_five_l3026_302663

theorem original_number_is_five : ∃ x : ℚ, ((x / 4) * 12) - 6 = 9 ∧ x = 5 := by
  sorry

end original_number_is_five_l3026_302663


namespace arithmetic_sequence_first_term_l3026_302688

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with S₁₀ = 5 and a₇ = 1, a₁ = -1 -/
theorem arithmetic_sequence_first_term
  (seq : ArithmeticSequence)
  (h1 : seq.S 10 = 5)
  (h2 : seq.a 7 = 1) :
  seq.a 1 = -1 := by
  sorry

end arithmetic_sequence_first_term_l3026_302688


namespace solution_verification_l3026_302662

/-- Proves that (3, 2020, 4) and (-1, 2018, -2) are solutions to the given system of equations -/
theorem solution_verification :
  (∃ (x y z : ℤ), 
    (x + y - 2018 = (y - 2019) * x) ∧
    (y + z - 2017 = (y - 2019) * z) ∧
    (x + z + 5 = x * z) ∧
    ((x = 3 ∧ y = 2020 ∧ z = 4) ∨ (x = -1 ∧ y = 2018 ∧ z = -2))) := by
  sorry

end solution_verification_l3026_302662


namespace expression_value_l3026_302643

theorem expression_value : (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) = 470 := by
  sorry

end expression_value_l3026_302643


namespace min_values_l3026_302685

theorem min_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 3/b = 1 → x + 3*y ≤ a + 3*b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 3/b = 1 → x*y ≤ a*b) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 3/b = 1 ∧ x + 3*y = a + 3*b ∧ x*y = a*b) :=
by sorry

end min_values_l3026_302685


namespace triangle_area_theorem_l3026_302658

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define a point on a line segment
def PointOnSegment (P A B : ℝ × ℝ) : Prop := sorry

-- Define the midpoint of a line segment
def Midpoint (M A B : ℝ × ℝ) : Prop := sorry

-- Define the angle between two vectors
def Angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the length of a line segment
def Length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_theorem (A B C D E : ℝ × ℝ) :
  Triangle A B C →
  Midpoint E B C →
  PointOnSegment D A C →
  Length A C = 1 →
  Angle B A C = π / 3 →  -- 60°
  Angle A B C = 5 * π / 9 →  -- 100°
  Angle A C B = π / 9 →  -- 20°
  Angle D E C = 4 * π / 9 →  -- 80°
  TriangleArea A B C + 2 * TriangleArea C D E = Real.sqrt 3 / 8 := by sorry

end triangle_area_theorem_l3026_302658


namespace sqrt_meaningful_range_l3026_302651

theorem sqrt_meaningful_range (a : ℝ) : 
  (∃ x : ℝ, x^2 = 2 - a) ↔ a ≤ 2 := by
sorry

end sqrt_meaningful_range_l3026_302651


namespace drainpipe_time_l3026_302618

/-- Given a tank and three pipes:
    - Pipe1 fills the tank in 5 hours
    - Pipe2 fills the tank in 4 hours
    - Drainpipe empties the tank in x hours
    - All three pipes together fill the tank in 2.5 hours
    Prove that x = 20 -/
theorem drainpipe_time (pipe1_time pipe2_time all_pipes_time : ℝ) 
  (h1 : pipe1_time = 5)
  (h2 : pipe2_time = 4)
  (h3 : all_pipes_time = 2.5)
  (drain_time : ℝ) :
  (1 / pipe1_time + 1 / pipe2_time - 1 / drain_time = 1 / all_pipes_time) → 
  drain_time = 20 := by
  sorry

end drainpipe_time_l3026_302618


namespace certain_number_equation_l3026_302664

theorem certain_number_equation : ∃ x : ℚ, 1038 * x = 173 * 240 ∧ x = 40 := by
  sorry

end certain_number_equation_l3026_302664


namespace fredrickson_chickens_l3026_302699

/-- Given a total number of chickens, calculates the number of chickens that do not lay eggs. -/
def chickens_not_laying_eggs (total : ℕ) : ℕ :=
  let roosters := total / 4
  let hens := total - roosters
  let laying_hens := (hens * 3) / 4
  roosters + (hens - laying_hens)

/-- Theorem stating that for 80 chickens, where 1/4 are roosters and 3/4 of hens lay eggs,
    the number of chickens not laying eggs is 35. -/
theorem fredrickson_chickens :
  chickens_not_laying_eggs 80 = 35 := by
  sorry

#eval chickens_not_laying_eggs 80

end fredrickson_chickens_l3026_302699


namespace number_of_divisors_of_n_l3026_302645

def n : ℕ := 293601000

theorem number_of_divisors_of_n : Nat.card {d : ℕ | d ∣ n} = 32 := by
  sorry

end number_of_divisors_of_n_l3026_302645


namespace quadruple_sum_product_l3026_302684

theorem quadruple_sum_product : 
  ∀ (x₁ x₂ x₃ x₄ : ℝ),
  (x₁ + x₂ * x₃ * x₄ = 2 ∧
   x₂ + x₁ * x₃ * x₄ = 2 ∧
   x₃ + x₁ * x₂ * x₄ = 2 ∧
   x₄ + x₁ * x₂ * x₃ = 2) →
  ((x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = 3) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = 3 ∧ x₄ = -1) ∨
   (x₁ = -1 ∧ x₂ = 3 ∧ x₃ = -1 ∧ x₄ = -1) ∨
   (x₁ = 3 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = -1)) :=
by sorry


end quadruple_sum_product_l3026_302684


namespace angle_triple_supplement_measure_l3026_302687

theorem angle_triple_supplement_measure : 
  ∀ x : ℝ, (x = 3 * (180 - x)) → x = 135 := by
  sorry

end angle_triple_supplement_measure_l3026_302687


namespace mrs_sheridan_fish_count_l3026_302619

theorem mrs_sheridan_fish_count :
  let initial_fish : ℕ := 22
  let fish_from_sister : ℕ := 47
  initial_fish + fish_from_sister = 69 :=
by sorry

end mrs_sheridan_fish_count_l3026_302619


namespace abc_sum_problem_l3026_302682

theorem abc_sum_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b * c = 1) (h2 : a + 1 / c = 7) (h3 : b + 1 / a = 12) :
  c + 1 / b = 21 / 83 := by
sorry

end abc_sum_problem_l3026_302682


namespace theta_range_l3026_302670

theorem theta_range (θ : Real) : 
  θ ∈ Set.Icc 0 π ∧ 
  (∀ x ∈ Set.Icc (-1) 0, x^2 * Real.cos θ + (x+1)^2 * Real.sin θ + x^2 + x > 0) →
  θ ∈ Set.Ioo (π/12) (5*π/12) := by
sorry

end theta_range_l3026_302670


namespace ammonia_composition_l3026_302644

/-- The mass percentage of Nitrogen in Ammonia -/
def nitrogen_percentage : ℝ := 77.78

/-- The mass percentage of Hydrogen in Ammonia -/
def hydrogen_percentage : ℝ := 100 - nitrogen_percentage

theorem ammonia_composition :
  hydrogen_percentage = 22.22 := by sorry

end ammonia_composition_l3026_302644


namespace lines_properties_l3026_302615

/-- Two lines in 2D space -/
structure Lines where
  l1 : ℝ → ℝ → ℝ := fun x y => 2 * x + y + 4
  l2 : ℝ → ℝ → ℝ → ℝ := fun a x y => a * x + 4 * y + 1

/-- The intersection point of two lines when they are perpendicular -/
def intersection (lines : Lines) : ℝ × ℝ := sorry

/-- The distance between two lines when they are parallel -/
def distance (lines : Lines) : ℝ := sorry

/-- Main theorem about the properties of the two lines -/
theorem lines_properties (lines : Lines) :
  (intersection lines = (-3/2, -1) ∧ 
   distance lines = 3 * Real.sqrt 5 / 4) := by sorry

end lines_properties_l3026_302615


namespace matrix_product_l3026_302668

def A : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, 1], ![2, 1, 2], ![1, 2, 3]]
def B : Matrix (Fin 3) (Fin 3) ℤ := ![![1, 1, -1], ![2, -1, 1], ![1, 0, 1]]
def C : Matrix (Fin 3) (Fin 3) ℤ := ![![6, 2, -1], ![6, 1, 1], ![8, -1, 4]]

theorem matrix_product : A * B = C := by sorry

end matrix_product_l3026_302668


namespace x_power_2048_minus_reciprocal_l3026_302634

theorem x_power_2048_minus_reciprocal (x : ℂ) (h : x + 1/x = Complex.I * Real.sqrt 2) :
  x^2048 - 1/x^2048 = 14^512 - 1024 := by
  sorry

end x_power_2048_minus_reciprocal_l3026_302634


namespace complex_subtraction_simplification_l3026_302622

theorem complex_subtraction_simplification :
  (7 - 3*I) - (9 - 5*I) = -2 + 2*I :=
by sorry

end complex_subtraction_simplification_l3026_302622


namespace intersection_theorem_l3026_302613

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x > 0}
def N : Set ℝ := {x : ℝ | x^2 - 4 > 0}

-- State the theorem
theorem intersection_theorem :
  M ∩ (Set.univ \ N) = Set.Ioo 0 2 := by sorry

end intersection_theorem_l3026_302613


namespace complement_of_A_l3026_302628

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 6}

theorem complement_of_A : (Aᶜ : Set ℕ) = {1, 3, 5} := by sorry

end complement_of_A_l3026_302628


namespace equation_solution_l3026_302606

theorem equation_solution (a b c : ℤ) : 
  (∀ x, (x - a) * (x - 12) + 4 = (x + b) * (x + c)) → (a = 7 ∨ a = 17) :=
by sorry

end equation_solution_l3026_302606


namespace unique_six_digit_number_l3026_302621

theorem unique_six_digit_number : ∃! n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧  -- six-digit number
  n % 10 = 2 ∧                 -- ends in 2
  2000000 + (n / 10) = 3 * n ∧ -- moving 2 to first position triples the number
  n = 857142 := by
sorry

end unique_six_digit_number_l3026_302621


namespace binomial_15_4_l3026_302666

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binomial_15_4_l3026_302666


namespace zero_last_in_hundreds_l3026_302603

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Get the units digit of a number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- Get the hundreds digit of a number -/
def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

/-- Check if a digit has appeared in the units position up to the nth Fibonacci number -/
def digit_appeared_units (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ units_digit (fib k) = d

/-- Check if a digit has appeared in the hundreds position up to the nth Fibonacci number -/
def digit_appeared_hundreds (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ hundreds_digit (fib k) = d

/-- The main theorem: 0 is the last digit to appear in the hundreds position -/
theorem zero_last_in_hundreds :
  ∃ N : ℕ, ∀ d : ℕ, d < 10 →
    (∀ n ≥ N, digit_appeared_units d n → digit_appeared_hundreds d n) ∧
    (∃ n ≥ N, digit_appeared_units 0 n ∧ ¬digit_appeared_hundreds 0 n) :=
sorry

end zero_last_in_hundreds_l3026_302603


namespace total_books_read_l3026_302654

def books_may : ℕ := 2
def books_june : ℕ := 6
def books_july : ℕ := 10
def books_august : ℕ := 14
def books_september : ℕ := 18

theorem total_books_read : books_may + books_june + books_july + books_august + books_september = 50 := by
  sorry

end total_books_read_l3026_302654


namespace floor_sqrt_50_squared_l3026_302610

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end floor_sqrt_50_squared_l3026_302610


namespace book_arrangement_count_l3026_302620

def num_books : ℕ := 10
def num_calculus : ℕ := 3
def num_algebra : ℕ := 4
def num_statistics : ℕ := 3

theorem book_arrangement_count :
  (num_calculus.factorial * num_statistics.factorial * (num_books - num_algebra).factorial) = 25920 :=
sorry

end book_arrangement_count_l3026_302620


namespace sidorov_cash_calculation_l3026_302640

/-- The disposable cash of the Sidorov family as of June 1, 2018 -/
def sidorov_cash : ℝ := 724506.3

/-- The first component of the Sidorov family's cash -/
def cash_component1 : ℝ := 496941.3

/-- The second component of the Sidorov family's cash -/
def cash_component2 : ℝ := 227565.0

/-- Theorem stating that the sum of the two cash components equals the total disposable cash -/
theorem sidorov_cash_calculation : 
  cash_component1 + cash_component2 = sidorov_cash := by
  sorry

end sidorov_cash_calculation_l3026_302640


namespace building_stories_l3026_302602

/-- Represents the number of stories in the building -/
def n : ℕ := sorry

/-- Time taken by Lola to climb one story -/
def lola_time_per_story : ℕ := 10

/-- Time taken by the elevator to go up one story -/
def elevator_time_per_story : ℕ := 8

/-- Time the elevator stops on each floor -/
def elevator_stop_time : ℕ := 3

/-- Total time taken by the slower person to reach the top -/
def total_time : ℕ := 220

/-- Time taken by Lola to reach the top -/
def lola_total_time : ℕ := n * lola_time_per_story

/-- Time taken by Tara (using the elevator) to reach the top -/
def tara_total_time : ℕ := n * elevator_time_per_story + (n - 1) * elevator_stop_time

theorem building_stories :
  (tara_total_time ≥ lola_total_time) ∧ (tara_total_time = total_time) → n = 20 := by
  sorry

end building_stories_l3026_302602


namespace prob_select_AB_correct_l3026_302694

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of students to be selected
def selected_students : ℕ := 3

-- Define the probability of selecting both A and B
def prob_select_AB : ℚ := 3 / 10

-- Theorem statement
theorem prob_select_AB_correct :
  (Nat.choose (total_students - 2) (selected_students - 2)) / (Nat.choose total_students selected_students) = prob_select_AB :=
sorry

end prob_select_AB_correct_l3026_302694


namespace nested_fraction_evaluation_l3026_302673

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end nested_fraction_evaluation_l3026_302673


namespace largest_non_sum_is_correct_l3026_302649

/-- The largest natural number not exceeding 50 that cannot be expressed as a sum of 5s and 6s -/
def largest_non_sum : ℕ := 19

/-- A predicate that checks if a natural number can be expressed as a sum of 5s and 6s -/
def is_sum_of_5_and_6 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

theorem largest_non_sum_is_correct :
  (largest_non_sum ≤ 50) ∧
  ¬(is_sum_of_5_and_6 largest_non_sum) ∧
  ∀ (m : ℕ), m > largest_non_sum → m ≤ 50 → is_sum_of_5_and_6 m :=
by sorry

end largest_non_sum_is_correct_l3026_302649


namespace cassies_dogs_l3026_302607

/-- The number of parrots Cassie has -/
def num_parrots : ℕ := 8

/-- The number of nails per dog foot -/
def nails_per_dog_foot : ℕ := 4

/-- The number of feet a dog has -/
def dog_feet : ℕ := 4

/-- The number of claws per parrot leg -/
def claws_per_parrot_leg : ℕ := 3

/-- The number of legs a parrot has -/
def parrot_legs : ℕ := 2

/-- The total number of nails Cassie needs to cut -/
def total_nails : ℕ := 113

/-- The number of dogs Cassie has -/
def num_dogs : ℕ := 4

theorem cassies_dogs :
  num_dogs = 4 :=
by sorry

end cassies_dogs_l3026_302607


namespace double_burgers_count_l3026_302681

/-- Represents the purchase of hamburgers for the marching band. -/
structure HamburgerPurchase where
  total_cost : ℚ
  total_burgers : ℕ
  single_burger_price : ℚ
  double_burger_price : ℚ

/-- Calculates the number of double burgers purchased. -/
def number_of_double_burgers (purchase : HamburgerPurchase) : ℕ := 
  sorry

/-- Theorem stating that the number of double burgers purchased is 41. -/
theorem double_burgers_count (purchase : HamburgerPurchase) 
  (h1 : purchase.total_cost = 70.5)
  (h2 : purchase.total_burgers = 50)
  (h3 : purchase.single_burger_price = 1)
  (h4 : purchase.double_burger_price = 1.5) :
  number_of_double_burgers purchase = 41 := by
  sorry

end double_burgers_count_l3026_302681


namespace range_of_a_l3026_302639

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 2 3, x^2 - a ≥ 0) → a ∈ Set.Iic 4 :=
by sorry

end range_of_a_l3026_302639


namespace inequality_system_solution_l3026_302632

theorem inequality_system_solution (m : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x ≤ 2 ∧ x > m) → m < 2 := by
  sorry

end inequality_system_solution_l3026_302632


namespace hyperbola_properties_l3026_302675

noncomputable section

/-- Definition of the hyperbola C -/
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Definition of the focal length -/
def focal_length (a b : ℝ) : ℝ := 4 * Real.sqrt 2

/-- Definition of the point P on the hyperbola -/
def point_on_hyperbola (a b x₀ y₀ : ℝ) : Prop :=
  hyperbola a b x₀ y₀

/-- Definition of points P₁ and P₂ on the hyperbola -/
def points_on_hyperbola (a b x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  hyperbola a b x₁ y₁ ∧ hyperbola a b x₂ y₂

/-- Definition of the vector relation -/
def vector_relation (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  3 * x₀ = x₁ + 2 * x₂ ∧ 3 * y₀ = y₁ + 2 * y₂

/-- Definition of perpendicular lines through P -/
def perpendicular_lines (a x₀ : ℝ) : Prop :=
  ∃ (y : ℝ), x₀ * y = -a^2

/-- Main theorem -/
theorem hyperbola_properties
  (a b x₀ y₀ x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : focal_length a b = 4 * Real.sqrt 2)
  (h₄ : point_on_hyperbola a b x₀ y₀)
  (h₅ : points_on_hyperbola a b x₀ y₀ x₁ y₁ x₂ y₂)
  (h₆ : vector_relation x₀ y₀ x₁ y₁ x₂ y₂)
  (h₇ : perpendicular_lines a x₀) :
  (x₁ * x₂ - y₁ * y₂ = 9) ∧
  (∃ (S : ℝ), S ≤ 9/2 ∧ (S = 9/2 ↔ a = 2 * Real.sqrt 2 ∧ b = 2 * Real.sqrt 2)) ∧
  (∃ (A B : ℝ × ℝ), ∀ (x y : ℝ),
    (x - 2 * Real.sqrt 2)^2 + y^2 = (x + 2 * Real.sqrt 2)^2 + y^2 →
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2) :=
sorry

end hyperbola_properties_l3026_302675


namespace endpoint_coordinate_sum_l3026_302648

/-- Given a line segment with one endpoint (6, -1) and midpoint (3, 7),
    the sum of the coordinates of the other endpoint is 15. -/
theorem endpoint_coordinate_sum : ∀ (x y : ℝ),
  (6 + x) / 2 = 3 →
  (-1 + y) / 2 = 7 →
  x + y = 15 := by
  sorry

end endpoint_coordinate_sum_l3026_302648


namespace triangle_function_properties_l3026_302698

/-- Given a triangle ABC with side lengths a, b, c, where c > a > 0 and c > b > 0,
    and a function f(x) = a^x + b^x - c^x, prove that:
    1. For all x < 1, f(x) > 0
    2. There exists x > 0 such that xa^x, b^x, c^x cannot form a triangle
    3. If ABC is obtuse, then there exists x ∈ (1, 2) such that f(x) = 0 -/
theorem triangle_function_properties (a b c : ℝ) (h1 : c > a) (h2 : a > 0) (h3 : c > b) (h4 : b > 0)
  (h5 : a + b > c) (f : ℝ → ℝ) (hf : ∀ x, f x = a^x + b^x - c^x) :
  (∀ x < 1, f x > 0) ∧
  (∃ x > 0, ¬ (xa^x + b^x > c^x ∧ xa^x + c^x > b^x ∧ b^x + c^x > xa^x)) ∧
  (a^2 + b^2 < c^2 → ∃ x ∈ Set.Ioo 1 2, f x = 0) :=
by sorry

end triangle_function_properties_l3026_302698


namespace quadratic_inequality_solution_set_l3026_302633

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x - 1) * (x - 3) < 0} = {x : ℝ | 1 < x ∧ x < 3} := by
  sorry

end quadratic_inequality_solution_set_l3026_302633


namespace john_mary_chess_consecutive_l3026_302616

theorem john_mary_chess_consecutive (n : ℕ) : 
  ¬(n % 16 = 0 ∧ (n + 1) % 25 = 0) ∧ ¬((n + 1) % 16 = 0 ∧ n % 25 = 0) := by
  sorry

end john_mary_chess_consecutive_l3026_302616


namespace arithmetic_expression_equality_l3026_302655

theorem arithmetic_expression_equality : 
  (50 - (4050 - 450)) * (4050 - (450 - 50)) = -12957500 := by
  sorry

end arithmetic_expression_equality_l3026_302655


namespace bank_queue_properties_l3026_302636

/-- Represents a queue of people with different operation times -/
structure BankQueue where
  total_people : Nat
  simple_ops : Nat
  long_ops : Nat
  simple_time : Nat
  long_time : Nat

/-- Calculates the minimum wasted person-minutes -/
def min_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the maximum wasted person-minutes -/
def max_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the expected wasted person-minutes for a random order -/
def expected_wasted_time (q : BankQueue) : Rat :=
  sorry

/-- Theorem stating the properties of the bank queue problem -/
theorem bank_queue_properties (q : BankQueue)
  (h1 : q.total_people = 8)
  (h2 : q.simple_ops = 5)
  (h3 : q.long_ops = 3)
  (h4 : q.simple_time = 1)
  (h5 : q.long_time = 5) :
  min_wasted_time q = 40 ∧
  max_wasted_time q = 100 ∧
  expected_wasted_time q = 84 := by
  sorry

end bank_queue_properties_l3026_302636


namespace third_layer_sugar_l3026_302660

/-- The amount of sugar needed for each layer of the cake -/
def sugar_amount (layer : Nat) : ℕ :=
  match layer with
  | 1 => 2  -- First layer requires 2 cups of sugar
  | 2 => 2 * sugar_amount 1  -- Second layer is twice as big as the first
  | 3 => 3 * sugar_amount 2  -- Third layer is three times larger than the second
  | _ => 0  -- We only consider 3 layers in this problem

theorem third_layer_sugar : sugar_amount 3 = 12 := by
  sorry

end third_layer_sugar_l3026_302660


namespace money_distribution_l3026_302631

/-- Given three people A, B, and C with some amount of money, prove that B and C together have 340 rupees. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 500 →  -- Total money between A, B, and C
  A + C = 200 →      -- Money A and C have together
  C = 40 →           -- Money C has
  B + C = 340 :=     -- Prove that B and C have 340 together
by sorry

end money_distribution_l3026_302631


namespace remainder_2357912_div_8_l3026_302630

theorem remainder_2357912_div_8 : 2357912 % 8 = 0 := by
  sorry

end remainder_2357912_div_8_l3026_302630


namespace matching_shoes_probability_l3026_302612

/-- A box containing pairs of shoes -/
structure ShoeBox where
  pairs : ℕ
  total : ℕ
  total_eq_twice_pairs : total = 2 * pairs

/-- The probability of selecting two matching shoes from a ShoeBox -/
def matchingProbability (box : ShoeBox) : ℚ :=
  1 / (box.total - 1)

theorem matching_shoes_probability (box : ShoeBox) 
  (h : box.pairs = 100) : 
  matchingProbability box = 1 / 199 := by
  sorry

end matching_shoes_probability_l3026_302612


namespace first_month_sale_l3026_302659

theorem first_month_sale
  (sale2 : ℕ) (sale3 : ℕ) (sale4 : ℕ) (sale5 : ℕ) (sale6 : ℕ) (avg_sale : ℕ)
  (h1 : sale2 = 6927)
  (h2 : sale3 = 6855)
  (h3 : sale4 = 7230)
  (h4 : sale5 = 6562)
  (h5 : sale6 = 6191)
  (h6 : avg_sale = 6700) :
  6 * avg_sale - (sale2 + sale3 + sale4 + sale5 + sale6) = 6435 := by
  sorry

end first_month_sale_l3026_302659


namespace problem_solution_l3026_302623

theorem problem_solution (m n : ℝ) 
  (hm : m^2 - 2*m - 1 = 0) 
  (hn : n^2 + 2*n - 1 = 0) 
  (hmn : m*n ≠ 1) : 
  (m*n + n + 1) / n = 3 := by
  sorry

end problem_solution_l3026_302623


namespace complex_equation_solution_l3026_302614

theorem complex_equation_solution (z : ℂ) :
  z * (1 - Complex.I) = 2 + Complex.I → z = (1 / 2 : ℂ) + (3 / 2 : ℂ) * Complex.I := by
  sorry

end complex_equation_solution_l3026_302614


namespace replaced_person_weight_is_65_l3026_302657

/-- The weight of the replaced person when the average weight of 6 persons
    increases by 2.5 kg after replacing one person with a new 80 kg person -/
def replacedPersonWeight (initialCount : ℕ) (averageIncrease : ℝ) (newPersonWeight : ℝ) : ℝ :=
  newPersonWeight - (initialCount : ℝ) * averageIncrease

/-- Theorem stating that under the given conditions, the weight of the replaced person is 65 kg -/
theorem replaced_person_weight_is_65 :
  replacedPersonWeight 6 2.5 80 = 65 := by
  sorry

end replaced_person_weight_is_65_l3026_302657


namespace geometric_sequence_b_value_l3026_302669

theorem geometric_sequence_b_value (b : ℝ) (h₁ : b > 0) :
  (∃ r : ℝ, r ≠ 0 ∧
    b = 10 * r ∧
    10 / 9 = b * r ∧
    10 / 81 = (10 / 9) * r) →
  b = 10 := by
sorry

end geometric_sequence_b_value_l3026_302669


namespace shoe_probability_l3026_302617

def total_pairs : ℕ := 15
def black_pairs : ℕ := 8
def red_pairs : ℕ := 4
def white_pairs : ℕ := 3

def total_shoes : ℕ := total_pairs * 2

def favorable_outcomes : ℕ := black_pairs * black_pairs + red_pairs * red_pairs + white_pairs * white_pairs

theorem shoe_probability : 
  (favorable_outcomes : ℚ) / (total_shoes.choose 2) = 89 / 435 := by
  sorry

end shoe_probability_l3026_302617


namespace journal_pages_per_session_l3026_302650

/-- Given the number of journal-writing sessions per week and the total number of pages written
    in a certain number of weeks, calculate the number of pages written per session. -/
def pages_per_session (sessions_per_week : ℕ) (total_pages : ℕ) (num_weeks : ℕ) : ℕ :=
  total_pages / (sessions_per_week * num_weeks)

/-- Theorem stating that under the given conditions, each student writes 4 pages per session. -/
theorem journal_pages_per_session :
  pages_per_session 3 72 6 = 4 := by
  sorry

end journal_pages_per_session_l3026_302650


namespace consecutive_page_numbers_sum_l3026_302604

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n * (n + 1) = 20412 → n + (n + 1) = 285 := by
  sorry

end consecutive_page_numbers_sum_l3026_302604


namespace program_output_l3026_302611

def program (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
  let a' := b
  let b' := c
  let c' := a'
  (a', b', c')

theorem program_output : program 10 20 30 = (20, 30, 20) := by
  sorry

end program_output_l3026_302611


namespace add_10000_seconds_to_5_45_00_l3026_302661

def seconds_to_time (seconds : ℕ) : ℕ × ℕ × ℕ :=
  let total_minutes := seconds / 60
  let remaining_seconds := seconds % 60
  let hours := total_minutes / 60
  let minutes := total_minutes % 60
  (hours, minutes, remaining_seconds)

def add_time (start : ℕ × ℕ × ℕ) (duration : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (start_h, start_m, start_s) := start
  let (duration_h, duration_m, duration_s) := duration
  let total_seconds := start_s + start_m * 60 + start_h * 3600 +
                       duration_s + duration_m * 60 + duration_h * 3600
  seconds_to_time total_seconds

theorem add_10000_seconds_to_5_45_00 :
  add_time (5, 45, 0) (seconds_to_time 10000) = (8, 31, 40) :=
sorry

end add_10000_seconds_to_5_45_00_l3026_302661


namespace pool_capacity_after_addition_l3026_302646

/-- Proves that adding 300 gallons to a pool with given conditions results in 40.38% capacity filled -/
theorem pool_capacity_after_addition
  (total_capacity : ℝ)
  (additional_water : ℝ)
  (increase_percentage : ℝ)
  (h1 : total_capacity = 1529.4117647058824)
  (h2 : additional_water = 300)
  (h3 : increase_percentage = 30)
  (h4 : (additional_water / total_capacity) * 100 = increase_percentage) :
  let final_percentage := (((increase_percentage / 100) * total_capacity) / total_capacity) * 100
  ∃ ε > 0, |final_percentage - 40.38| < ε :=
by sorry

end pool_capacity_after_addition_l3026_302646


namespace min_value_reciprocal_sum_l3026_302672

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ (1/a + 1/b = 2 ↔ a = 1 ∧ b = 1) := by
  sorry

end min_value_reciprocal_sum_l3026_302672


namespace isosceles_triangle_base_length_l3026_302683

/-- An isosceles triangle with perimeter 13 and one side 3 has a base of 3 -/
theorem isosceles_triangle_base_length :
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 13 →
  (a = b ∧ c = 3) ∨ (a = c ∧ b = 3) ∨ (b = c ∧ a = 3) →
  (a = 3 ∨ b = 3 ∨ c = 3) :=
by sorry

end isosceles_triangle_base_length_l3026_302683


namespace multiply_decimal_l3026_302674

theorem multiply_decimal : (3.6 : ℝ) * 0.25 = 0.9 := by
  sorry

end multiply_decimal_l3026_302674


namespace john_total_pay_this_year_l3026_302697

/-- John's annual bonus calculation -/
def johnBonus (baseSalaryLastYear : ℝ) (firstBonusLastYear : ℝ) (baseSalaryThisYear : ℝ) 
              (bonusGrowthRate : ℝ) (projectBonus : ℝ) (projectsCompleted : ℕ) : ℝ :=
  let firstBonusThisYear := firstBonusLastYear * (1 + bonusGrowthRate)
  let secondBonus := projectBonus * projectsCompleted
  baseSalaryThisYear + firstBonusThisYear + secondBonus

theorem john_total_pay_this_year :
  johnBonus 100000 10000 200000 0.05 2000 8 = 226500 := by
  sorry

end john_total_pay_this_year_l3026_302697


namespace line_equation_l3026_302635

/-- Proves that the equation 4x + 3y - 13 = 0 represents the line passing through (1, 3)
    with a slope that is 1/3 of the slope of y = -4x -/
theorem line_equation (x y : ℝ) : 
  (∃ (k : ℝ), k = (-4 : ℝ) / 3 ∧ 
   y - 3 = k * (x - 1) ∧
   (∀ (x' y' : ℝ), y' = -4 * x' → k = (1 : ℝ) / 3 * (-4))) → 
  (4 * x + 3 * y - 13 = 0) := by
sorry

end line_equation_l3026_302635


namespace reflected_arcs_area_l3026_302600

/-- The area of the region bounded by 8 reflected arcs in a circle with an inscribed regular octagon -/
theorem reflected_arcs_area (s : ℝ) (h : s = 1) : 
  let r : ℝ := 1 / Real.sqrt (2 - Real.sqrt 2)
  let octagon_area : ℝ := 2 * (1 + Real.sqrt 2)
  let arc_area : ℝ := π * (2 + Real.sqrt 2) / 2 - 2 * Real.sqrt 3
  octagon_area - arc_area = 2 * (1 + Real.sqrt 2) - π * (2 + Real.sqrt 2) / 2 + 2 * Real.sqrt 3 :=
by sorry


end reflected_arcs_area_l3026_302600


namespace quadratic_equation_roots_l3026_302679

theorem quadratic_equation_roots : 
  let f : ℝ → ℝ := λ x => x^2 + 2*x - 3
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -3 := by
  sorry

end quadratic_equation_roots_l3026_302679


namespace cuboid_edge_length_l3026_302677

/-- Represents the length of an edge in centimeters -/
def Edge := ℝ

/-- Represents the volume in cubic centimeters -/
def Volume := ℝ

/-- Given a cuboid with edges a, x, and b, and volume v,
    prove that if a = 4, b = 6, and v = 96, then x = 4 -/
theorem cuboid_edge_length (a x b v : ℝ) :
  a = 4 → b = 6 → v = 96 → v = a * x * b → x = 4 := by
  sorry

end cuboid_edge_length_l3026_302677


namespace binomial_15_12_l3026_302626

theorem binomial_15_12 : Nat.choose 15 12 = 2730 := by
  sorry

end binomial_15_12_l3026_302626


namespace square_coloring_l3026_302695

/-- The number of triangles in the square -/
def n : ℕ := 18

/-- The number of triangles to be colored -/
def k : ℕ := 6

/-- Binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem square_coloring :
  binomial n k = 18564 := by
  sorry

end square_coloring_l3026_302695


namespace min_value_theorem_l3026_302609

theorem min_value_theorem (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 9) : 
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 := by
  sorry

end min_value_theorem_l3026_302609


namespace soccer_club_girls_l3026_302647

theorem soccer_club_girls (total_members : ℕ) (meeting_attendance : ℕ) 
  (h1 : total_members = 30)
  (h2 : meeting_attendance = 18)
  (h3 : ∃ (boys girls : ℕ), 
    boys + girls = total_members ∧ 
    boys + girls / 3 = meeting_attendance) :
  ∃ (girls : ℕ), girls = 18 ∧ 
    ∃ (boys : ℕ), boys + girls = total_members ∧ 
                   boys + girls / 3 = meeting_attendance :=
by sorry

end soccer_club_girls_l3026_302647


namespace magnitude_z_l3026_302689

theorem magnitude_z (w z : ℂ) (h1 : w * z = 16 - 30 * I) (h2 : Complex.abs w = 5) : 
  Complex.abs z = 6.8 := by
sorry

end magnitude_z_l3026_302689


namespace exists_initial_points_for_82_final_l3026_302624

/-- The number of points after applying the procedure once -/
def points_after_first_procedure (n : ℕ) : ℕ := 3 * n - 2

/-- The number of points after applying the procedure twice -/
def points_after_second_procedure (n : ℕ) : ℕ := 9 * n - 8

/-- Theorem stating that it's possible to have 82 points after the two procedures -/
theorem exists_initial_points_for_82_final : ∃ n : ℕ, points_after_second_procedure n = 82 := by
  sorry

#eval points_after_second_procedure 10

end exists_initial_points_for_82_final_l3026_302624


namespace hyperbola_C_eccentricity_l3026_302625

/-- Hyperbola C with foci F₁ and F₂, and points P and Q satisfying given conditions -/
structure HyperbolaC where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_P_on_C : P.1^2 / a^2 - P.2^2 / b^2 = 1
  h_Q_on_asymptote : Q.2 / Q.1 = b / a
  h_first_quadrant : P.1 > 0 ∧ P.2 > 0 ∧ Q.1 > 0 ∧ Q.2 > 0
  h_QP_eq_PF₂ : (Q.1 - P.1, Q.2 - P.2) = (P.1 - F₂.1, P.2 - F₂.2)
  h_QF₁_perp_QF₂ : (Q.1 - F₁.1) * (Q.1 - F₂.1) + (Q.2 - F₁.2) * (Q.2 - F₂.2) = 0

/-- The eccentricity of hyperbola C is √5 - 1 -/
theorem hyperbola_C_eccentricity (hC : HyperbolaC) : 
  ∃ e : ℝ, e = Real.sqrt 5 - 1 ∧ e^2 = (hC.a^2 + hC.b^2) / hC.a^2 :=
sorry

end hyperbola_C_eccentricity_l3026_302625


namespace complex_magnitude_theorem_l3026_302696

theorem complex_magnitude_theorem : 
  let i : ℂ := Complex.I
  let T : ℂ := 3 * ((1 + i)^15 - (1 - i)^15)
  Complex.abs T = 768 := by sorry

end complex_magnitude_theorem_l3026_302696


namespace bertha_family_no_daughters_bertha_family_no_daughters_is_32_l3026_302671

/-- Represents the family structure of Bertha and her descendants --/
structure BerthaFamily where
  daughters : Nat
  granddaughters : Nat
  daughters_with_children : Nat

/-- The properties of Bertha's family --/
def bertha_family : BerthaFamily where
  daughters := 8
  granddaughters := 32
  daughters_with_children := 8

theorem bertha_family_no_daughters : Nat :=
  let total := bertha_family.daughters + bertha_family.granddaughters
  let with_daughters := bertha_family.daughters_with_children
  total - with_daughters
  
#check bertha_family_no_daughters

theorem bertha_family_no_daughters_is_32 :
  bertha_family_no_daughters = 32 := by
  sorry

#check bertha_family_no_daughters_is_32

end bertha_family_no_daughters_bertha_family_no_daughters_is_32_l3026_302671


namespace no_collision_probability_correct_l3026_302642

/-- A regular icosahedron -/
structure Icosahedron :=
  (vertices : Fin 12 → Type)
  (adjacent : Fin 12 → Fin 5 → Fin 12)

/-- An ant on the icosahedron -/
structure Ant :=
  (position : Fin 12)

/-- The probability of an ant moving to a specific adjacent vertex -/
def move_probability : ℚ := 1 / 5

/-- The number of ants -/
def num_ants : ℕ := 12

/-- The probability that no two ants arrive at the same vertex -/
def no_collision_probability (i : Icosahedron) : ℚ :=
  (Nat.factorial num_ants : ℚ) / (5 ^ num_ants)

theorem no_collision_probability_correct (i : Icosahedron) :
  no_collision_probability i = (Nat.factorial num_ants : ℚ) / (5 ^ num_ants) :=
sorry

end no_collision_probability_correct_l3026_302642


namespace circumcenter_equidistant_closest_vertex_l3026_302601

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem: The circumcenter is equidistant from all vertices of the triangle
theorem circumcenter_equidistant (t : Triangle) :
  distance (circumcenter t) t.A = distance (circumcenter t) t.B ∧
  distance (circumcenter t) t.B = distance (circumcenter t) t.C :=
sorry

-- Theorem: Any point in the plane is closest to one of the three vertices
theorem closest_vertex (t : Triangle) (p : ℝ × ℝ) :
  (distance p t.A ≤ distance p t.B ∧ distance p t.A ≤ distance p t.C) ∨
  (distance p t.B ≤ distance p t.A ∧ distance p t.B ≤ distance p t.C) ∨
  (distance p t.C ≤ distance p t.A ∧ distance p t.C ≤ distance p t.B) :=
sorry

end circumcenter_equidistant_closest_vertex_l3026_302601


namespace expected_sides_after_cutting_l3026_302693

/-- The expected number of sides of a randomly picked polygon after cutting -/
def expected_sides (n : ℕ) : ℚ :=
  (n + 7200 : ℚ) / 3601

/-- Theorem stating the expected number of sides after cutting an n-sided polygon for 3600 seconds -/
theorem expected_sides_after_cutting (n : ℕ) :
  let initial_sides := n
  let num_cuts := 3600
  let total_sides := initial_sides + 2 * num_cuts
  let num_polygons := num_cuts + 1
  expected_sides n = total_sides / num_polygons :=
by
  sorry

#eval expected_sides 3  -- For a triangle
#eval expected_sides 4  -- For a quadrilateral

end expected_sides_after_cutting_l3026_302693


namespace original_number_proof_l3026_302629

theorem original_number_proof (N : ℕ) : 
  (∃ k : ℕ, N - 32 = 87 * k) ∧ 
  (∀ m : ℕ, m < 32 → ¬∃ j : ℕ, N - m = 87 * j) → 
  N = 119 :=
sorry

end original_number_proof_l3026_302629


namespace cookie_calculation_l3026_302605

theorem cookie_calculation (initial_cookies given_cookies received_cookies : ℕ) :
  initial_cookies ≥ given_cookies →
  initial_cookies - given_cookies + received_cookies =
    initial_cookies - given_cookies + received_cookies :=
by
  sorry

end cookie_calculation_l3026_302605


namespace factorization_1_factorization_2_l3026_302667

variable (a b x y : ℝ)

/-- Factorization of 3ax^2 - 6ax + 3a --/
theorem factorization_1 : 3*a*x^2 - 6*a*x + 3*a = 3*a*(x-1)^2 := by sorry

/-- Factorization of 9x^2(a-b) + 4y^3(b-a) --/
theorem factorization_2 : 9*x^2*(a-b) + 4*y^3*(b-a) = (a-b)*(9*x^2 - 4*y^3) := by sorry

end factorization_1_factorization_2_l3026_302667


namespace group_interval_calculation_l3026_302638

/-- Given a group [a,b) in a frequency distribution histogram with frequency 0.3 and height 0.06, |a-b| = 5 -/
theorem group_interval_calculation (a b : ℝ) 
  (frequency : ℝ) (height : ℝ) 
  (h1 : frequency = 0.3) 
  (h2 : height = 0.06) : 
  |a - b| = 5 := by sorry

end group_interval_calculation_l3026_302638


namespace caramel_apple_ice_cream_cost_difference_l3026_302608

/-- The cost difference between a caramel apple and an ice cream cone -/
theorem caramel_apple_ice_cream_cost_difference 
  (caramel_apple_cost : ℕ) 
  (ice_cream_cost : ℕ) 
  (h1 : caramel_apple_cost = 25)
  (h2 : ice_cream_cost = 15) : 
  caramel_apple_cost - ice_cream_cost = 10 := by
  sorry

end caramel_apple_ice_cream_cost_difference_l3026_302608


namespace rally_accident_probability_l3026_302676

/-- The probability of a car successfully completing the rally --/
def rally_success_probability : ℚ :=
  let bridge_success : ℚ := 4/5
  let turn_success : ℚ := 7/10
  let tunnel_success : ℚ := 9/10
  let sand_success : ℚ := 3/5
  bridge_success * turn_success * tunnel_success * sand_success

/-- The probability of a car being involved in an accident during the rally --/
def accident_probability : ℚ := 1 - rally_success_probability

theorem rally_accident_probability :
  accident_probability = 1756 / 2500 :=
by sorry

end rally_accident_probability_l3026_302676
