import Mathlib

namespace NUMINAMATH_CALUDE_remainder_theorem_l857_85744

theorem remainder_theorem (n m : ℤ) 
  (hn : n % 37 = 15) 
  (hm : m % 47 = 21) : 
  (3 * n + 2 * m) % 59 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l857_85744


namespace NUMINAMATH_CALUDE_display_rows_l857_85792

/-- Represents the number of cans in a row given its position from the top -/
def cans_in_row (n : ℕ) : ℕ := 3 * n - 2

/-- Calculates the total number of cans in the first n rows -/
def total_cans (n : ℕ) : ℕ := n * (3 * n - 1) / 2

theorem display_rows : ∃ n : ℕ, total_cans n = 225 ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_display_rows_l857_85792


namespace NUMINAMATH_CALUDE_percentage_increase_l857_85791

theorem percentage_increase (x y z : ℝ) : 
  y = 0.4 * z → x = 0.48 * z → (x - y) / y = 0.2 := by sorry

end NUMINAMATH_CALUDE_percentage_increase_l857_85791


namespace NUMINAMATH_CALUDE_prime_power_sum_square_l857_85731

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- The set of valid triples (p, q, r) -/
def validTriples : Set (ℕ × ℕ × ℕ) :=
  {(2, 5, 2), (2, 2, 5), (2, 3, 3), (3, 3, 2)} ∪ 
  {x | ∃ n : ℕ, x = (2, 2*n+1, 2*n+1)}

/-- The main theorem -/
theorem prime_power_sum_square (p q r : ℕ) :
  isPrime p ∧ isPrime q ∧ isPrime r ∧ 
  isPerfectSquare (p^q + p^r) ↔ 
  (p, q, r) ∈ validTriples := by sorry

end NUMINAMATH_CALUDE_prime_power_sum_square_l857_85731


namespace NUMINAMATH_CALUDE_min_average_of_four_integers_l857_85724

theorem min_average_of_four_integers (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  d = 90 ∧                 -- Largest is 90
  a ≥ 29 →                 -- Smallest is at least 29
  (a + b + c + d) / 4 ≥ 45 :=
sorry

end NUMINAMATH_CALUDE_min_average_of_four_integers_l857_85724


namespace NUMINAMATH_CALUDE_sum_of_cubes_consecutive_integers_l857_85789

theorem sum_of_cubes_consecutive_integers :
  ∃ n : ℤ, n^3 + (n + 1)^3 = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_consecutive_integers_l857_85789


namespace NUMINAMATH_CALUDE_student_line_count_l857_85790

theorem student_line_count :
  ∀ (n : ℕ),
    n > 0 →
    (∃ (eunjung_pos yoojung_pos : ℕ),
      eunjung_pos = 5 ∧
      yoojung_pos = n ∧
      yoojung_pos - eunjung_pos - 1 = 8) →
    n = 14 := by
  sorry

end NUMINAMATH_CALUDE_student_line_count_l857_85790


namespace NUMINAMATH_CALUDE_translation_not_equal_claim_l857_85750

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the original function
def original (x : ℝ) : ℝ := f (-x)

-- Define the function after translation to the right by 1 unit
def translated (x : ℝ) : ℝ := f (-(x - 1))

-- Define the function claimed in the problem statement
def claimed (x : ℝ) : ℝ := f (-x - 1)

-- Theorem stating that the translated function is not equal to the claimed function
theorem translation_not_equal_claim : translated f ≠ claimed f := by sorry

end NUMINAMATH_CALUDE_translation_not_equal_claim_l857_85750


namespace NUMINAMATH_CALUDE_right_triangle_geometric_sequence_l857_85732

theorem right_triangle_geometric_sequence (a b c q : ℝ) : 
  q > 1 →
  a > 0 →
  b > 0 →
  c > 0 →
  a * q = b →
  b * q = c →
  a^2 + b^2 = c^2 →
  q^2 = (Real.sqrt 5 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_geometric_sequence_l857_85732


namespace NUMINAMATH_CALUDE_box_volume_formula_l857_85776

/-- The volume of a box formed by cutting squares from corners of a metal sheet -/
def boxVolume (x : ℝ) : ℝ :=
  (16 - 2*x) * (12 - 2*x) * x

theorem box_volume_formula (x : ℝ) :
  boxVolume x = 192*x - 56*x^2 + 4*x^3 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_formula_l857_85776


namespace NUMINAMATH_CALUDE_equation_solutions_l857_85736

theorem equation_solutions : 
  {x : ℝ | Real.sqrt (6 * x - 5) + 12 / Real.sqrt (6 * x - 5) = 8} = {41/6, 3/2} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l857_85736


namespace NUMINAMATH_CALUDE_spiral_stripe_length_l857_85764

theorem spiral_stripe_length (c h : ℝ) (hc : c = 18) (hh : h = 8) :
  let stripe_length := Real.sqrt ((2 * c)^2 + h^2)
  stripe_length = Real.sqrt 1360 := by
  sorry

end NUMINAMATH_CALUDE_spiral_stripe_length_l857_85764


namespace NUMINAMATH_CALUDE_line_point_k_value_l857_85753

/-- A line contains the points (2, -1), (10, k), and (25, 4). The value of k is 17/23. -/
theorem line_point_k_value (k : ℚ) : 
  (∃ (line : ℝ → ℝ), 
    line 2 = -1 ∧ 
    line 10 = k ∧ 
    line 25 = 4) → 
  k = 17/23 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l857_85753


namespace NUMINAMATH_CALUDE_election_theorem_l857_85769

/-- Represents a ballot with candidate names -/
structure Ballot where
  candidates : Finset String
  constraint : candidates.card = 10

/-- Represents a ballot box containing ballots -/
structure BallotBox where
  ballots : Set Ballot
  nonempty : ballots.Nonempty

/-- The election setup -/
structure Election where
  boxes : Fin 11 → BallotBox
  common_candidate : ∀ (selection : Fin 11 → Ballot), 
    (∀ i, selection i ∈ (boxes i).ballots) → 
    ∃ c, ∀ i, c ∈ (selection i).candidates

theorem election_theorem (e : Election) :
  ∃ i : Fin 11, ∃ c : String, ∀ b ∈ (e.boxes i).ballots, c ∈ b.candidates :=
sorry

end NUMINAMATH_CALUDE_election_theorem_l857_85769


namespace NUMINAMATH_CALUDE_inequality_proof_l857_85748

theorem inequality_proof (x y z t : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : t ≥ 0)
  (h5 : x * y * z = 2) (h6 : y + z + t = 2 * Real.sqrt 2) :
  2 * x^2 + y^2 + z^2 + t^2 ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l857_85748


namespace NUMINAMATH_CALUDE_ant_probability_l857_85746

/-- Represents a vertex of a cube -/
inductive Vertex : Type
| A | B | C | D | E | F | G | H

/-- Represents the movement of an ant from one vertex to another -/
def Move : Type := Vertex → Vertex

/-- The set of all possible moves for all 8 ants -/
def AllMoves : Type := Fin 8 → Move

/-- Checks if a move is valid (i.e., to an adjacent vertex) -/
def isValidMove (m : Move) : Prop := sorry

/-- Checks if a set of moves results in no two ants on the same vertex -/
def noCollisions (moves : AllMoves) : Prop := sorry

/-- The total number of possible movement combinations -/
def totalMoves : ℕ := 3^8

/-- The number of valid movement combinations where no two ants collide -/
def validMoves : ℕ := 240

/-- The probability of no two ants arriving at the same vertex -/
theorem ant_probability : 
  (validMoves : ℚ) / totalMoves = 240 / 6561 := by sorry

end NUMINAMATH_CALUDE_ant_probability_l857_85746


namespace NUMINAMATH_CALUDE_yoongi_calculation_l857_85757

theorem yoongi_calculation : (30 + 5) / 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_calculation_l857_85757


namespace NUMINAMATH_CALUDE_custom_mult_solution_l857_85778

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := 2 * a - b^2

/-- Theorem stating that given the custom multiplication and the equation a * 7 = 16, a equals 32.5 -/
theorem custom_mult_solution :
  ∃ a : ℝ, custom_mult a 7 = 16 ∧ a = 32.5 := by sorry

end NUMINAMATH_CALUDE_custom_mult_solution_l857_85778


namespace NUMINAMATH_CALUDE_union_equality_implies_a_values_l857_85786

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {1, a}
def B (a : ℝ) : Set ℝ := {a^2}

-- State the theorem
theorem union_equality_implies_a_values (a : ℝ) :
  A a ∪ B a = A a → a = -1 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_values_l857_85786


namespace NUMINAMATH_CALUDE_fourth_person_truthful_l857_85788

/-- Represents a person who can be either a liar or truthful. -/
inductive Person
| Liar
| Truthful

/-- The statements made by each person. -/
def statement (p : Fin 4 → Person) : Prop :=
  (p 0 = Person.Liar ∧ p 1 = Person.Liar ∧ p 2 = Person.Liar ∧ p 3 = Person.Liar) ∨
  (∃! i, p i = Person.Liar) ∨
  (∃ i j, i ≠ j ∧ p i = Person.Liar ∧ p j = Person.Liar ∧ ∀ k, k ≠ i → k ≠ j → p k = Person.Truthful) ∨
  (p 3 = Person.Truthful)

/-- The main theorem stating that the fourth person must be truthful. -/
theorem fourth_person_truthful :
  ∀ p : Fin 4 → Person, statement p → p 3 = Person.Truthful :=
sorry

end NUMINAMATH_CALUDE_fourth_person_truthful_l857_85788


namespace NUMINAMATH_CALUDE_continuity_from_g_and_h_continuous_l857_85725

open Function Set Filter Topology

/-- Given a function f: ℝ → ℝ, if g(x) = f(x) + f(2x) and h(x) = f(x) + f(4x) are continuous,
    then f is continuous. -/
theorem continuity_from_g_and_h_continuous
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h : ℝ → ℝ)
  (hg : g = λ x => f x + f (2 * x))
  (hh : h = λ x => f x + f (4 * x))
  (hg_cont : Continuous g)
  (hh_cont : Continuous h) :
  Continuous f :=
sorry

end NUMINAMATH_CALUDE_continuity_from_g_and_h_continuous_l857_85725


namespace NUMINAMATH_CALUDE_entree_percentage_is_80_percent_l857_85795

/-- Calculates the percentage of total cost that went to entrees -/
def entree_percentage (total_cost appetizer_cost : ℚ) (num_appetizers : ℕ) : ℚ :=
  let appetizer_total := appetizer_cost * num_appetizers
  let entree_total := total_cost - appetizer_total
  (entree_total / total_cost) * 100

/-- Theorem stating that the percentage of total cost that went to entrees is 80% -/
theorem entree_percentage_is_80_percent :
  entree_percentage 50 5 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_entree_percentage_is_80_percent_l857_85795


namespace NUMINAMATH_CALUDE_decreasing_f_sufficient_not_necessary_for_increasing_g_l857_85734

open Real

theorem decreasing_f_sufficient_not_necessary_for_increasing_g
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x > a^y) →
  (∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3) ∧
  ¬(∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3 →
    a^x > a^y) :=
by sorry

end NUMINAMATH_CALUDE_decreasing_f_sufficient_not_necessary_for_increasing_g_l857_85734


namespace NUMINAMATH_CALUDE_greatest_prime_factor_sum_even_products_l857_85780

def double_factorial (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else n * double_factorial (n - 2)

def even_product (n : ℕ) : ℕ :=
  double_factorial (2 * (n / 2))

theorem greatest_prime_factor_sum_even_products :
  ∃ (p : ℕ), p.Prime ∧ p = 23 ∧
  ∀ (q : ℕ), q.Prime → q ∣ (even_product 22 + even_product 20) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_sum_even_products_l857_85780


namespace NUMINAMATH_CALUDE_R_final_coordinates_l857_85727

/-- Reflect a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflect a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflect a point over the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- The initial point R -/
def R : ℝ × ℝ := (0, -5)

/-- The sequence of reflections applied to R -/
def R_transformed : ℝ × ℝ :=
  reflect_y_eq_x (reflect_y (reflect_x R))

theorem R_final_coordinates :
  R_transformed = (5, 0) := by
  sorry

end NUMINAMATH_CALUDE_R_final_coordinates_l857_85727


namespace NUMINAMATH_CALUDE_second_person_speed_l857_85747

/-- Given two people traveling between points A and B, prove the speed of the second person. -/
theorem second_person_speed 
  (distance : ℝ) 
  (speed_first : ℝ) 
  (travel_time : ℝ) 
  (h1 : distance = 600) 
  (h2 : speed_first = 70) 
  (h3 : travel_time = 4) : 
  ∃ speed_second : ℝ, speed_second = 80 ∧ 
  speed_first * travel_time + speed_second * travel_time = distance :=
by
  sorry

#check second_person_speed

end NUMINAMATH_CALUDE_second_person_speed_l857_85747


namespace NUMINAMATH_CALUDE_number_value_l857_85739

theorem number_value (x : ℚ) (n : ℚ) : 
  x = 12 → n + 7 / x = 6 - 5 / x → n = 5 := by sorry

end NUMINAMATH_CALUDE_number_value_l857_85739


namespace NUMINAMATH_CALUDE_congruence_problem_l857_85701

theorem congruence_problem (x : ℤ) : 
  (5 * x + 9) % 20 = 3 → (3 * x + 14) % 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l857_85701


namespace NUMINAMATH_CALUDE_small_and_large_puzzle_cost_small_and_large_puzzle_cost_proof_l857_85770

/-- The cost of a small puzzle and a large puzzle together is $23 -/
theorem small_and_large_puzzle_cost : ℝ → ℝ → Prop :=
  fun (small_cost large_cost : ℝ) ↦
    large_cost = 15 ∧
    large_cost + 3 * small_cost = 39 →
    small_cost + large_cost = 23

/-- Proof of the theorem -/
theorem small_and_large_puzzle_cost_proof :
  ∃ (small_cost large_cost : ℝ),
    small_and_large_puzzle_cost small_cost large_cost :=
by
  sorry

end NUMINAMATH_CALUDE_small_and_large_puzzle_cost_small_and_large_puzzle_cost_proof_l857_85770


namespace NUMINAMATH_CALUDE_fraction_increase_l857_85700

theorem fraction_increase (x y : ℝ) (h : 2*x ≠ 3*y) : 
  (5*x * 5*y) / (2*(5*x) - 3*(5*y)) = 5 * (x*y / (2*x - 3*y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_increase_l857_85700


namespace NUMINAMATH_CALUDE_shirts_sold_l857_85761

/-- The number of shirts sold in a store -/
theorem shirts_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) : 
  initial = 49 → remaining = 28 → sold = initial - remaining → sold = 21 :=
by sorry

end NUMINAMATH_CALUDE_shirts_sold_l857_85761


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l857_85703

-- Define the sets A and B
def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l857_85703


namespace NUMINAMATH_CALUDE_final_painting_height_l857_85709

/-- Calculates the height of the final painting given the conditions -/
theorem final_painting_height :
  let total_paintings : ℕ := 5
  let total_area : ℝ := 200
  let small_painting_side : ℝ := 5
  let small_painting_count : ℕ := 3
  let large_painting_width : ℝ := 10
  let large_painting_height : ℝ := 8
  let final_painting_width : ℝ := 9
  
  let small_paintings_area : ℝ := small_painting_count * (small_painting_side * small_painting_side)
  let large_painting_area : ℝ := large_painting_width * large_painting_height
  let known_area : ℝ := small_paintings_area + large_painting_area
  let final_painting_area : ℝ := total_area - known_area
  
  final_painting_area / final_painting_width = 5 :=
by sorry

end NUMINAMATH_CALUDE_final_painting_height_l857_85709


namespace NUMINAMATH_CALUDE_combined_salaries_combined_salaries_proof_l857_85704

/-- The combined salaries of A, B, C, and E, given D's salary and the average salary of all five. -/
theorem combined_salaries (salary_D : ℕ) (average_salary : ℕ) : ℕ :=
  let total_salary := average_salary * 5
  total_salary - salary_D

/-- Proof that the combined salaries of A, B, C, and E is 38000, given the conditions. -/
theorem combined_salaries_proof (salary_D : ℕ) (average_salary : ℕ)
    (h1 : salary_D = 7000)
    (h2 : average_salary = 9000) :
    combined_salaries salary_D average_salary = 38000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_combined_salaries_proof_l857_85704


namespace NUMINAMATH_CALUDE_range_of_a_l857_85754

-- Define the conditions p and q as functions of x and a
def p (x : ℝ) : Prop := abs (4 * x - 1) ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the property that ¬p is a necessary but not sufficient condition for ¬q
def neg_p_necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, ¬(q x a) → ¬(p x)) ∧ ∃ x, ¬(p x) ∧ (q x a)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, neg_p_necessary_not_sufficient a ↔ -1/2 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l857_85754


namespace NUMINAMATH_CALUDE_punch_bowl_problem_l857_85735

/-- The capacity of the punch bowl in gallons -/
def bowl_capacity : ℝ := 16

/-- The amount of punch Sally drinks in gallons -/
def sally_drinks : ℝ := 2

/-- The amount of punch Mark adds to completely fill the bowl after Sally drinks -/
def final_addition : ℝ := 12

/-- The amount of punch Mark adds after his cousin drinks -/
def mark_addition : ℝ := 12

theorem punch_bowl_problem :
  ∃ (initial_amount : ℝ),
    initial_amount ≥ 0 ∧
    initial_amount ≤ bowl_capacity ∧
    (initial_amount / 2 + mark_addition - sally_drinks + final_addition = bowl_capacity) :=
by
  sorry

#check punch_bowl_problem

end NUMINAMATH_CALUDE_punch_bowl_problem_l857_85735


namespace NUMINAMATH_CALUDE_min_value_expression_l857_85719

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2)) / (a * b * c) ≥ 343 ∧
  ((1^2 + 5*1 + 2) * (1^2 + 5*1 + 2) * (1^2 + 5*1 + 2)) / (1 * 1 * 1) = 343 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l857_85719


namespace NUMINAMATH_CALUDE_nail_fractions_l857_85723

theorem nail_fractions (fraction_4d fraction_total : ℝ) 
  (h1 : fraction_4d = 0.5)
  (h2 : fraction_total = 0.75) : 
  fraction_total - fraction_4d = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_nail_fractions_l857_85723


namespace NUMINAMATH_CALUDE_rectangle_area_l857_85797

theorem rectangle_area (l w : ℝ) (h1 : l = 15) (h2 : (2 * l + 2 * w) / w = 5) :
  l * w = 150 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_l857_85797


namespace NUMINAMATH_CALUDE_pet_food_price_l857_85773

theorem pet_food_price (regular_discount_min regular_discount_max additional_discount lowest_price : Real) 
  (h1 : 0.1 ≤ regular_discount_min ∧ regular_discount_min ≤ regular_discount_max ∧ regular_discount_max ≤ 0.3)
  (h2 : additional_discount = 0.2)
  (h3 : lowest_price = 25.2)
  : ∃ (original_price : Real),
    original_price * (1 - regular_discount_max) * (1 - additional_discount) = lowest_price ∧
    original_price = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_food_price_l857_85773


namespace NUMINAMATH_CALUDE_comparison_of_powers_l857_85768

theorem comparison_of_powers : (2^40 : ℕ) < 3^28 ∧ (31^11 : ℕ) < 17^14 := by sorry

end NUMINAMATH_CALUDE_comparison_of_powers_l857_85768


namespace NUMINAMATH_CALUDE_f_composition_and_domain_l857_85715

def f (x : ℝ) : ℝ := x + 5

theorem f_composition_and_domain :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f x ∈ Set.Icc 2 7) →
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, ∀ y ∈ Set.Icc (-3 : ℝ) 2, x < y → f x < f y) →
  (∀ x ∈ Set.Icc (-8 : ℝ) (-3), f (f x) = x + 10) ∧
  (∀ x, f (f x) ∈ Set.Icc 2 7 ↔ x ∈ Set.Icc (-8 : ℝ) (-3)) := by
  sorry

#check f_composition_and_domain

end NUMINAMATH_CALUDE_f_composition_and_domain_l857_85715


namespace NUMINAMATH_CALUDE_rectangle_existence_theorem_l857_85751

/-- Represents a rectangle with given side lengths -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Checks if a rectangle B exists with half the perimeter and area of rectangle A -/
def exists_half_rectangle (A : Rectangle) : Prop :=
  ∃ x : ℝ, x * ((A.a + A.b) / 2 - x) = A.a * A.b / 2

theorem rectangle_existence_theorem (A : Rectangle) :
  (A.a = 6 ∧ A.b = 1 → exists_half_rectangle A) ∧
  (A.a = 2 ∧ A.b = 1 → ¬exists_half_rectangle A) := by
  sorry

#check rectangle_existence_theorem

end NUMINAMATH_CALUDE_rectangle_existence_theorem_l857_85751


namespace NUMINAMATH_CALUDE_unique_integer_pair_satisfying_equation_l857_85763

theorem unique_integer_pair_satisfying_equation : 
  ∃! (m n : ℤ), m + 2*n = m*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_pair_satisfying_equation_l857_85763


namespace NUMINAMATH_CALUDE_fraction_multiplication_l857_85772

theorem fraction_multiplication : (1/2 : ℚ) * (1/3 : ℚ) * (1/6 : ℚ) * 72 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l857_85772


namespace NUMINAMATH_CALUDE_equality_of_coefficients_l857_85784

theorem equality_of_coefficients (a b c : ℝ) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c ≥ b * x^2 + c * x + a ∧ 
                b * x^2 + c * x + a ≥ c * x^2 + a * x + b) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equality_of_coefficients_l857_85784


namespace NUMINAMATH_CALUDE_unfactorable_polynomial_l857_85771

theorem unfactorable_polynomial (b c d : ℤ) (h : Odd (b * d + c * d)) :
  ¬ ∃ (p q r : ℤ), ∀ (x : ℤ), x^3 + b*x^2 + c*x + d = (x + p) * (x^2 + q*x + r) :=
sorry

end NUMINAMATH_CALUDE_unfactorable_polynomial_l857_85771


namespace NUMINAMATH_CALUDE_phones_left_theorem_l857_85706

/-- Calculates the number of phones left in the factory after doubling production and selling a quarter --/
def phones_left_in_factory (last_year_production : ℕ) : ℕ :=
  let this_year_production := 2 * last_year_production
  let sold_phones := this_year_production / 4
  this_year_production - sold_phones

/-- Theorem stating that given last year's production of 5000 phones, 
    if this year's production is doubled and a quarter of it is sold, 
    then the number of phones left in the factory is 7500 --/
theorem phones_left_theorem : phones_left_in_factory 5000 = 7500 := by
  sorry

end NUMINAMATH_CALUDE_phones_left_theorem_l857_85706


namespace NUMINAMATH_CALUDE_cal_anthony_transaction_ratio_l857_85713

theorem cal_anthony_transaction_ratio :
  ∀ (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ),
    mabel_transactions = 90 →
    anthony_transactions = mabel_transactions + mabel_transactions / 10 →
    jade_transactions = 84 →
    jade_transactions = cal_transactions + 18 →
    cal_transactions * 3 = anthony_transactions * 2 := by
  sorry

end NUMINAMATH_CALUDE_cal_anthony_transaction_ratio_l857_85713


namespace NUMINAMATH_CALUDE_fair_coin_three_heads_probability_l857_85738

theorem fair_coin_three_heads_probability :
  let p_head : ℚ := 1/2  -- Probability of getting heads on a single flip
  let p_three_heads : ℚ := p_head * p_head * p_head  -- Probability of getting heads on all three flips
  p_three_heads = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_three_heads_probability_l857_85738


namespace NUMINAMATH_CALUDE_liam_bills_cost_liam_bills_proof_l857_85733

/-- Calculates the cost of Liam's bills given his savings and remaining money. -/
theorem liam_bills_cost (monthly_savings : ℕ) (savings_duration_months : ℕ) (money_left : ℕ) : ℕ :=
  let total_savings := monthly_savings * savings_duration_months
  total_savings - money_left

/-- Proves that Liam's bills cost $3,500 given the problem conditions. -/
theorem liam_bills_proof :
  liam_bills_cost 500 24 8500 = 3500 := by
  sorry

end NUMINAMATH_CALUDE_liam_bills_cost_liam_bills_proof_l857_85733


namespace NUMINAMATH_CALUDE_pencil_price_after_discount_l857_85718

/-- Given a pencil with an original cost and a discount, calculate the final price. -/
def final_price (original_cost discount : ℚ) : ℚ :=
  original_cost - discount

/-- Theorem stating that for a pencil with an original cost of 4 dollars and a discount of 0.63 dollars, the final price is 3.37 dollars. -/
theorem pencil_price_after_discount :
  final_price 4 0.63 = 3.37 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_after_discount_l857_85718


namespace NUMINAMATH_CALUDE_eighth_term_is_negative_one_thirty_second_l857_85707

/-- Sequence definition -/
def a (n : ℕ) : ℚ := (-1)^(n+1) * (n : ℚ) / 2^n

/-- Theorem: The 8th term of the sequence is -1/32 -/
theorem eighth_term_is_negative_one_thirty_second : a 8 = -1/32 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_negative_one_thirty_second_l857_85707


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l857_85756

/-- The slope angle of the line x + √3 * y - 5 = 0 is 150 degrees. -/
theorem slope_angle_of_line (x y : ℝ) : 
  x + Real.sqrt 3 * y - 5 = 0 → 
  ∃ α : ℝ, α = 150 * π / 180 ∧ (Real.tan α = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l857_85756


namespace NUMINAMATH_CALUDE_sin_alpha_value_l857_85716

theorem sin_alpha_value (α β : Real) (h_acute : 0 < α ∧ α < π / 2)
  (h1 : 2 * Real.tan (π - α) - 3 * Real.cos (π / 2 + β) + 5 = 0)
  (h2 : Real.tan (π + α) + 6 * Real.sin (π + β) = 1) :
  Real.sin α = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l857_85716


namespace NUMINAMATH_CALUDE_evaluate_expression_l857_85702

theorem evaluate_expression (x y : ℚ) (hx : x = 3) (hy : y = -3) :
  (4 + y * x * (y + x) - 4^2) / (y - 4 + y^2) = -6 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l857_85702


namespace NUMINAMATH_CALUDE_round_table_numbers_l857_85712

theorem round_table_numbers (n : Fin 10 → ℝ) 
  (h1 : (n 9 + n 1) / 2 = 1)
  (h2 : (n 0 + n 2) / 2 = 2)
  (h3 : (n 1 + n 3) / 2 = 3)
  (h4 : (n 2 + n 4) / 2 = 4)
  (h5 : (n 3 + n 5) / 2 = 5)
  (h6 : (n 4 + n 6) / 2 = 6)
  (h7 : (n 5 + n 7) / 2 = 7)
  (h8 : (n 6 + n 8) / 2 = 8)
  (h9 : (n 7 + n 9) / 2 = 9)
  (h10 : (n 8 + n 0) / 2 = 10) :
  n 5 = 7 := by
sorry

end NUMINAMATH_CALUDE_round_table_numbers_l857_85712


namespace NUMINAMATH_CALUDE_loan_repayment_proof_l857_85728

/-- Calculates the total amount to be repaid for a loan with simple interest -/
def total_repayment (initial_loan : ℝ) (additional_loan : ℝ) (initial_period : ℝ) (total_period : ℝ) (rate : ℝ) : ℝ :=
  let initial_with_interest := initial_loan * (1 + rate * initial_period)
  let total_loan := initial_with_interest + additional_loan
  total_loan * (1 + rate * (total_period - initial_period))

/-- Proves that the total repayment for the given loan scenario is 27376 Rs -/
theorem loan_repayment_proof :
  total_repayment 10000 12000 2 5 0.06 = 27376 := by
  sorry

#eval total_repayment 10000 12000 2 5 0.06

end NUMINAMATH_CALUDE_loan_repayment_proof_l857_85728


namespace NUMINAMATH_CALUDE_sunny_cakes_l857_85765

/-- Given that Sunny gives away 2 cakes, puts 6 candles on each remaining cake,
    and uses a total of 36 candles, prove that she initially baked 8 cakes. -/
theorem sunny_cakes (cakes_given_away : ℕ) (candles_per_cake : ℕ) (total_candles : ℕ) :
  cakes_given_away = 2 →
  candles_per_cake = 6 →
  total_candles = 36 →
  cakes_given_away + (total_candles / candles_per_cake) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sunny_cakes_l857_85765


namespace NUMINAMATH_CALUDE_twelve_tone_equal_temperament_l857_85740

theorem twelve_tone_equal_temperament (a : ℕ → ℝ) :
  (∀ n, 1 ≤ n → n < 13 → a (n + 1) / a n = a 2 / a 1) →  -- Equal ratio between adjacent terms
  a 13 = 2 * a 1 →                                      -- Last term is twice the first term
  a 8 / a 2 = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_twelve_tone_equal_temperament_l857_85740


namespace NUMINAMATH_CALUDE_cakes_donated_proof_l857_85708

/-- The number of slices per cake -/
def slices_per_cake : ℕ := 8

/-- The price of each slice in dollars -/
def price_per_slice : ℚ := 1

/-- The donation from the first business owner per slice in dollars -/
def donation1_per_slice : ℚ := 1/2

/-- The donation from the second business owner per slice in dollars -/
def donation2_per_slice : ℚ := 1/4

/-- The total amount raised in dollars -/
def total_raised : ℚ := 140

/-- The number of cakes donated -/
def num_cakes : ℕ := 10

theorem cakes_donated_proof :
  (num_cakes : ℚ) * slices_per_cake * (price_per_slice + donation1_per_slice + donation2_per_slice) = total_raised :=
by sorry

end NUMINAMATH_CALUDE_cakes_donated_proof_l857_85708


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l857_85730

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_2 + a_4 = 20 and a_3 + a_5 = 40, then q = 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 : a 2 + a 4 = 20) 
  (h3 : a 3 + a 5 = 40) : 
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l857_85730


namespace NUMINAMATH_CALUDE_parallelogram_area_l857_85782

/-- The area of a parallelogram with base 20 meters and height 4 meters is 80 square meters. -/
theorem parallelogram_area : 
  let base : ℝ := 20
  let height : ℝ := 4
  let area := base * height
  area = 80 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l857_85782


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l857_85767

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l857_85767


namespace NUMINAMATH_CALUDE_max_n_satisfying_condition_l857_85787

def sequence_a (n : ℕ) : ℕ := 2^n - 1

def sum_S (n : ℕ) : ℕ := 2 * sequence_a n - n

theorem max_n_satisfying_condition :
  (∀ n : ℕ, sum_S n = 2 * sequence_a n - n) →
  (∃ max_n : ℕ, (∀ n : ℕ, n ≤ max_n ↔ sequence_a n ≤ 10 * n) ∧ max_n = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_n_satisfying_condition_l857_85787


namespace NUMINAMATH_CALUDE_function_roots_imply_a_range_l857_85759

/-- The function f(x) = 2ln(x) - x^2 + a has two roots in [1/e, e] iff a ∈ (1, 2 + 1/e^2] -/
theorem function_roots_imply_a_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = 2 * Real.log x - x^2 + a) →
  (∃ x y, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ 
          y ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ 
          x ≠ y ∧ f x = 0 ∧ f y = 0) →
  a ∈ Set.Ioo 1 (2 + 1 / (Real.exp 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_function_roots_imply_a_range_l857_85759


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l857_85726

theorem imaginary_part_of_complex_fraction : Complex.im ((2 * Complex.I) / (1 - Complex.I) * Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l857_85726


namespace NUMINAMATH_CALUDE_class_size_is_36_l857_85783

/-- The number of students in a class, given boat seating conditions. -/
def number_of_students (b : ℕ) : Prop :=
  ∃ n : ℕ,
    n = 6 * (b + 1) ∧
    n = 9 * (b - 1)

/-- Theorem stating that the number of students is 36. -/
theorem class_size_is_36 :
  ∃ b : ℕ, number_of_students b ∧ (6 * (b + 1) = 36) :=
sorry

end NUMINAMATH_CALUDE_class_size_is_36_l857_85783


namespace NUMINAMATH_CALUDE_exponent_simplification_l857_85720

theorem exponent_simplification (x : ℝ) : (x^5 * x^2) * x^4 = x^11 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l857_85720


namespace NUMINAMATH_CALUDE_max_value_of_expression_l857_85745

theorem max_value_of_expression (y : ℝ) :
  (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) ≤ 15 ∧
  ∃ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l857_85745


namespace NUMINAMATH_CALUDE_arithmetic_mean_greater_than_geometric_mean_l857_85710

theorem arithmetic_mean_greater_than_geometric_mean
  (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a ≠ b) (ha_pos : a ≠ 0) (hb_pos : b ≠ 0) :
  (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_greater_than_geometric_mean_l857_85710


namespace NUMINAMATH_CALUDE_circle_number_placement_l857_85729

-- Define the type for circle positions
inductive CirclePosition
  | one | two | three | four | five | six | seven | eight

-- Define the neighborhood relation
def isNeighbor : CirclePosition → CirclePosition → Prop
  | CirclePosition.one, CirclePosition.two => True
  | CirclePosition.one, CirclePosition.four => True
  | CirclePosition.two, CirclePosition.three => True
  | CirclePosition.two, CirclePosition.four => True
  | CirclePosition.two, CirclePosition.six => True
  | CirclePosition.three, CirclePosition.four => True
  | CirclePosition.three, CirclePosition.seven => True
  | CirclePosition.four, CirclePosition.five => True
  | CirclePosition.five, CirclePosition.six => True
  | CirclePosition.six, CirclePosition.seven => True
  | CirclePosition.seven, CirclePosition.eight => True
  | _, _ => False

-- Define the valid numbers
def validNumbers : List Nat := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define a function to check if a number is a divisor of another
def isDivisor (a b : Nat) : Prop := b % a = 0 ∧ a ≠ 1 ∧ a ≠ b

-- Define the main theorem
theorem circle_number_placement :
  ∃ (f : CirclePosition → Nat),
    (∀ p, f p ∈ validNumbers) ∧
    (∀ p₁ p₂, p₁ ≠ p₂ → f p₁ ≠ f p₂) ∧
    (∀ p₁ p₂, isNeighbor p₁ p₂ → ¬isDivisor (f p₁) (f p₂)) := by
  sorry

end NUMINAMATH_CALUDE_circle_number_placement_l857_85729


namespace NUMINAMATH_CALUDE_chris_soccer_cards_l857_85774

/-- Chris has some soccer cards. His friend, Charlie, has 32 cards. 
    Chris has 14 fewer cards than Charlie. -/
theorem chris_soccer_cards 
  (charlie_cards : ℕ) 
  (chris_fewer : ℕ)
  (h1 : charlie_cards = 32)
  (h2 : chris_fewer = 14) :
  charlie_cards - chris_fewer = 18 := by
  sorry

end NUMINAMATH_CALUDE_chris_soccer_cards_l857_85774


namespace NUMINAMATH_CALUDE_ball_max_height_l857_85749

/-- The height function of a ball traveling along a parabolic path -/
def h (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 20

/-- The maximum height reached by the ball -/
def max_height : ℝ := 145

theorem ball_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l857_85749


namespace NUMINAMATH_CALUDE_expected_value_of_coins_l857_85714

/-- Represents a coin with its value in cents and probability of heads -/
structure Coin where
  value : ℝ
  prob_heads : ℝ

/-- The set of coins being flipped -/
def coins : Finset Coin := sorry

/-- The expected value of a single coin -/
def expected_value (c : Coin) : ℝ := c.value * c.prob_heads

/-- The total expected value of all coins -/
def total_expected_value : ℝ := (coins.sum expected_value)

/-- Theorem stating the expected value of coins coming up heads -/
theorem expected_value_of_coins : 
  (coins.card = 5) → 
  (∃ c ∈ coins, c.value = 1 ∧ c.prob_heads = 1/2) →
  (∃ c ∈ coins, c.value = 5 ∧ c.prob_heads = 1/2) →
  (∃ c ∈ coins, c.value = 10 ∧ c.prob_heads = 1/2) →
  (∃ c ∈ coins, c.value = 25 ∧ c.prob_heads = 1/2) →
  (∃ c ∈ coins, c.value = 50 ∧ c.prob_heads = 3/4) →
  total_expected_value = 58 :=
by sorry

end NUMINAMATH_CALUDE_expected_value_of_coins_l857_85714


namespace NUMINAMATH_CALUDE_leilas_savings_leilas_savings_proof_l857_85785

theorem leilas_savings : ℝ → Prop :=
  fun savings =>
    let makeup_fraction : ℝ := 3/5
    let sweater_fraction : ℝ := 1/3
    let sweater_cost : ℝ := 40
    let shoes_cost : ℝ := 30
    let remaining_fraction : ℝ := 1 - makeup_fraction - sweater_fraction
    
    (sweater_fraction * savings = sweater_cost) ∧
    (remaining_fraction * savings = shoes_cost) ∧
    (savings = 175)

-- The proof goes here
theorem leilas_savings_proof : ∃ (s : ℝ), leilas_savings s :=
sorry

end NUMINAMATH_CALUDE_leilas_savings_leilas_savings_proof_l857_85785


namespace NUMINAMATH_CALUDE_expand_expression_l857_85737

theorem expand_expression (x y z : ℝ) : 
  (x + 12) * (3 * y + 4 * z + 15) = 3 * x * y + 4 * x * z + 15 * x + 36 * y + 48 * z + 180 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l857_85737


namespace NUMINAMATH_CALUDE_hexagon_side_length_l857_85793

theorem hexagon_side_length (perimeter : ℝ) (h : perimeter = 48) : 
  perimeter / 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l857_85793


namespace NUMINAMATH_CALUDE_climb_nine_flights_l857_85798

/-- Calculates the number of steps climbed given the number of flights, height per flight, and height per step. -/
def steps_climbed (flights : ℕ) (feet_per_flight : ℕ) (inches_per_step : ℕ) : ℕ :=
  (flights * feet_per_flight * 12) / inches_per_step

/-- Proves that climbing 9 flights of 10-foot stairs with 18-inch steps results in 60 steps. -/
theorem climb_nine_flights : steps_climbed 9 10 18 = 60 := by
  sorry

end NUMINAMATH_CALUDE_climb_nine_flights_l857_85798


namespace NUMINAMATH_CALUDE_emily_quiz_score_l857_85752

theorem emily_quiz_score (scores : List ℝ) (target_mean : ℝ) : 
  scores = [92, 95, 87, 89, 100] →
  target_mean = 93 →
  let new_score := 95
  let all_scores := scores ++ [new_score]
  (all_scores.sum / all_scores.length : ℝ) = target_mean := by
sorry


end NUMINAMATH_CALUDE_emily_quiz_score_l857_85752


namespace NUMINAMATH_CALUDE_rachel_books_total_l857_85766

theorem rachel_books_total (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 9)
  (h2 : mystery_shelves = 6)
  (h3 : picture_shelves = 2) :
  books_per_shelf * (mystery_shelves + picture_shelves) = 72 :=
by sorry

end NUMINAMATH_CALUDE_rachel_books_total_l857_85766


namespace NUMINAMATH_CALUDE_sin_monotone_decreasing_l857_85796

theorem sin_monotone_decreasing (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (π / 3 - 2 * x)
  ∀ x y, x ∈ Set.Icc (k * π - π / 12) (k * π + 5 * π / 12) →
         y ∈ Set.Icc (k * π - π / 12) (k * π + 5 * π / 12) →
         x ≤ y → f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_sin_monotone_decreasing_l857_85796


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l857_85777

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

def min_value (b : ℕ → ℝ) : ℝ :=
  5 * b 1 + 6 * b 2

theorem geometric_sequence_min_value :
  ∀ b : ℕ → ℝ, geometric_sequence b → b 0 = 2 →
  ∃ m : ℝ, m = min_value b ∧ m = -25/12 ∧ ∀ b' : ℕ → ℝ, geometric_sequence b' → b' 0 = 2 → min_value b' ≥ m :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_value_l857_85777


namespace NUMINAMATH_CALUDE_emilys_glue_sticks_l857_85758

theorem emilys_glue_sticks (total : ℕ) (sisters : ℕ) (emilys : ℕ) : 
  total = 13 → sisters = 7 → emilys = total - sisters → emilys = 6 :=
by sorry

end NUMINAMATH_CALUDE_emilys_glue_sticks_l857_85758


namespace NUMINAMATH_CALUDE_probability_b_speaks_truth_l857_85717

theorem probability_b_speaks_truth (prob_a_truth : ℝ) (prob_both_truth : ℝ) :
  prob_a_truth = 0.75 →
  prob_both_truth = 0.45 →
  ∃ prob_b_truth : ℝ, prob_b_truth = 0.6 ∧ prob_a_truth * prob_b_truth = prob_both_truth :=
by sorry

end NUMINAMATH_CALUDE_probability_b_speaks_truth_l857_85717


namespace NUMINAMATH_CALUDE_fib_150_mod_9_l857_85711

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_150_mod_9 : fib 150 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_9_l857_85711


namespace NUMINAMATH_CALUDE_polynomial_simplification_l857_85721

theorem polynomial_simplification (x : ℝ) :
  (x^5 + 3*x^4 + x^2 + 13) + (x^5 - 4*x^4 + x^3 - x^2 + 15) = 2*x^5 - x^4 + x^3 + 28 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l857_85721


namespace NUMINAMATH_CALUDE_problem_statements_l857_85775

theorem problem_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (ab - a - 2*b = 0 → a + 2*b ≥ 8) ∧
  (a + b = 1 → Real.sqrt (2*a + 4) + Real.sqrt (b + 1) ≤ 2 * Real.sqrt 3) ∧
  (1 / (a + 1) + 1 / (b + 2) = 1 / 3 → a*b + a + b ≥ 14 + 6 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l857_85775


namespace NUMINAMATH_CALUDE_unique_solution_iff_in_set_l857_85762

/-- The set of real numbers m for which the equation 2√(1-m(x+2)) = x+4 has exactly one solution -/
def solution_set : Set ℝ :=
  {m : ℝ | m > -1/2 ∨ m = -1}

/-- The equation 2√(1-m(x+2)) = x+4 -/
def equation (m : ℝ) (x : ℝ) : Prop :=
  2 * Real.sqrt (1 - m * (x + 2)) = x + 4

theorem unique_solution_iff_in_set (m : ℝ) :
  (∃! x, equation m x) ↔ m ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_in_set_l857_85762


namespace NUMINAMATH_CALUDE_positive_real_solution_l857_85799

theorem positive_real_solution (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 - b*d)/(b + 2*c + d) + (b^2 - c*a)/(c + 2*d + a) + 
  (c^2 - d*b)/(d + 2*a + b) + (d^2 - a*c)/(a + 2*b + c) = 0 →
  a = c ∧ b = d := by
sorry

end NUMINAMATH_CALUDE_positive_real_solution_l857_85799


namespace NUMINAMATH_CALUDE_no_prime_solution_l857_85722

/-- Convert a number from base p to decimal --/
def baseP_to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.foldr (fun d acc => d + p * acc) 0

/-- The equation that needs to be satisfied --/
def equation (p : Nat) : Prop :=
  baseP_to_decimal [1, 0, 1, 3] p +
  baseP_to_decimal [2, 0, 7] p +
  baseP_to_decimal [2, 1, 4] p +
  baseP_to_decimal [1, 0, 0] p +
  baseP_to_decimal [1, 0] p =
  baseP_to_decimal [3, 2, 1] p +
  baseP_to_decimal [4, 0, 3] p +
  baseP_to_decimal [2, 1, 0] p

theorem no_prime_solution :
  ∀ p, Nat.Prime p → ¬(equation p) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l857_85722


namespace NUMINAMATH_CALUDE_total_insect_legs_l857_85781

/-- The number of insects in the laboratory -/
def num_insects : ℕ := 6

/-- The number of legs per insect -/
def legs_per_insect : ℕ := 6

/-- Theorem: The total number of insect legs in the laboratory is 36 -/
theorem total_insect_legs : num_insects * legs_per_insect = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_insect_legs_l857_85781


namespace NUMINAMATH_CALUDE_readers_of_both_l857_85760

theorem readers_of_both (total : ℕ) (science_fiction : ℕ) (literary : ℕ) 
  (h1 : total = 150) 
  (h2 : science_fiction = 120) 
  (h3 : literary = 90) :
  science_fiction + literary - total = 60 := by
  sorry

end NUMINAMATH_CALUDE_readers_of_both_l857_85760


namespace NUMINAMATH_CALUDE_no_five_digit_flippy_divisible_by_11_l857_85743

/-- A flippy number is a number whose digits alternate between two distinct digits. -/
def is_flippy (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≠ y ∧ x < 10 ∧ y < 10 ∧
  (n = x * 10000 + y * 1000 + x * 100 + y * 10 + x ∨
   n = y * 10000 + x * 1000 + y * 100 + x * 10 + y)

/-- A number is five digits long if it's between 10000 and 99999, inclusive. -/
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

theorem no_five_digit_flippy_divisible_by_11 :
  ¬∃ n : ℕ, is_flippy n ∧ is_five_digit n ∧ n % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_five_digit_flippy_divisible_by_11_l857_85743


namespace NUMINAMATH_CALUDE_x_minus_y_value_l857_85779

theorem x_minus_y_value (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 17) 
  (eq2 : x + 3 * y = 5) : 
  x - y = 73 / 13 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l857_85779


namespace NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l857_85794

/-- Calculates the toll for a truck given the number of axles -/
def toll (axles : ℕ) : ℚ :=
  1.5 + 0.5 * (axles - 2)

/-- Calculates the number of axles for a truck given the total number of wheels -/
def axles_count (total_wheels : ℕ) : ℕ :=
  1 + (total_wheels - 2) / 4

theorem eighteen_wheel_truck_toll :
  toll (axles_count 18) = 3 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l857_85794


namespace NUMINAMATH_CALUDE_cone_volume_relation_l857_85755

/-- Represents a cone with given dimensions and properties -/
structure Cone where
  r : ℝ  -- base radius
  h : ℝ  -- height
  l : ℝ  -- slant height
  d : ℝ  -- distance from center of base to slant height
  S : ℝ  -- lateral surface area
  V : ℝ  -- volume
  r_pos : 0 < r
  h_pos : 0 < h
  l_pos : 0 < l
  d_pos : 0 < d
  S_pos : 0 < S
  V_pos : 0 < V
  S_eq : S = π * r * l
  V_eq : V = (1/3) * π * r^2 * h

/-- The volume of a cone is one-third of the product of its lateral surface area and the distance from the center of the base to the slant height -/
theorem cone_volume_relation (c : Cone) : c.V = (1/3) * c.d * c.S := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_relation_l857_85755


namespace NUMINAMATH_CALUDE_exists_line_with_F_as_incenter_l857_85741

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The right focus F -/
def F : ℝ × ℝ := (1, 0)

/-- The upper vertex M -/
def M : ℝ × ℝ := (0, 1)

/-- The line l -/
def line_l (x y : ℝ) : Prop := x + (2 - Real.sqrt 6) * y + 6 - 3 * Real.sqrt 6 = 0

/-- P and Q are points on both the ellipse and line l -/
def P_Q_on_ellipse_and_line (P Q : ℝ × ℝ) : Prop :=
  ellipse P.1 P.2 ∧ ellipse Q.1 Q.2 ∧ line_l P.1 P.2 ∧ line_l Q.1 Q.2

/-- F is the incenter of triangle MPQ -/
def F_is_incenter (P Q : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
    (F.1 - P.1)^2 + (F.2 - P.2)^2 = r^2 ∧
    (F.1 - Q.1)^2 + (F.2 - Q.2)^2 = r^2 ∧
    (F.1 - M.1)^2 + (F.2 - M.2)^2 = r^2

/-- The main theorem -/
theorem exists_line_with_F_as_incenter :
  ∃ (P Q : ℝ × ℝ), P_Q_on_ellipse_and_line P Q ∧ F_is_incenter P Q := by sorry

end NUMINAMATH_CALUDE_exists_line_with_F_as_incenter_l857_85741


namespace NUMINAMATH_CALUDE_cord_length_proof_l857_85742

theorem cord_length_proof (n : ℕ) (longest shortest : ℝ) : 
  n = 19 → 
  longest = 8 → 
  shortest = 2 → 
  (n : ℝ) * (longest / 2 + shortest) = 114 :=
by
  sorry

end NUMINAMATH_CALUDE_cord_length_proof_l857_85742


namespace NUMINAMATH_CALUDE_initial_crayons_count_l857_85705

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := sorry

/-- The number of crayons Benny added to the drawer -/
def added_crayons : ℕ := 3

/-- The total number of crayons in the drawer after Benny's addition -/
def total_crayons : ℕ := 12

/-- Theorem stating that the initial number of crayons is 9 -/
theorem initial_crayons_count : initial_crayons = 9 := by sorry

end NUMINAMATH_CALUDE_initial_crayons_count_l857_85705
