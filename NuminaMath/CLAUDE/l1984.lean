import Mathlib

namespace NUMINAMATH_CALUDE_existence_of_person_with_few_amicable_foes_l1984_198466

structure Society where
  n : ℕ  -- number of persons
  q : ℕ  -- number of amicable pairs
  is_valid : q ≤ n * (n - 1) / 2  -- maximum possible number of pairs

def is_hostile (S : Society) (a b : Fin S.n) : Prop := sorry

def is_amicable (S : Society) (a b : Fin S.n) : Prop := ¬(is_hostile S a b)

axiom society_property (S : Society) :
  ∀ (a b c : Fin S.n), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    is_hostile S a b ∨ is_hostile S b c ∨ is_hostile S a c

def foes (S : Society) (a : Fin S.n) : Set (Fin S.n) :=
  {b | is_hostile S a b}

def amicable_pairs_among_foes (S : Society) (a : Fin S.n) : ℕ := sorry

theorem existence_of_person_with_few_amicable_foes (S : Society) :
  ∃ (a : Fin S.n), amicable_pairs_among_foes S a ≤ S.q * (1 - 4 * S.q / (S.n * S.n)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_person_with_few_amicable_foes_l1984_198466


namespace NUMINAMATH_CALUDE_cos_sum_plus_cos_diff_l1984_198498

theorem cos_sum_plus_cos_diff (x y : ℝ) : 
  Real.cos (x + y) + Real.cos (x - y) = 2 * Real.cos x * Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_plus_cos_diff_l1984_198498


namespace NUMINAMATH_CALUDE_intersection_M_N_l1984_198494

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x : ℕ | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1984_198494


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1984_198460

theorem complex_fraction_simplification :
  (7 + 8 * Complex.I) / (3 - 4 * Complex.I) = -11/25 + 52/25 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1984_198460


namespace NUMINAMATH_CALUDE_equal_salary_at_5000_sales_l1984_198458

/-- Represents the monthly salary options for Juliet --/
structure SalaryOptions where
  flat_salary : ℝ
  base_salary : ℝ
  commission_rate : ℝ

/-- Calculates the total salary for the commission-based option --/
def commission_salary (options : SalaryOptions) (sales : ℝ) : ℝ :=
  options.base_salary + options.commission_rate * sales

/-- The specific salary options given in the problem --/
def juliet_options : SalaryOptions :=
  { flat_salary := 1800
    base_salary := 1600
    commission_rate := 0.04 }

/-- Theorem stating that the sales amount for equal salaries is $5000 --/
theorem equal_salary_at_5000_sales (options : SalaryOptions := juliet_options) :
  ∃ (sales : ℝ), sales = 5000 ∧ options.flat_salary = commission_salary options sales :=
by
  sorry

end NUMINAMATH_CALUDE_equal_salary_at_5000_sales_l1984_198458


namespace NUMINAMATH_CALUDE_page_number_divisibility_l1984_198405

theorem page_number_divisibility (n : ℕ) (k : ℕ) : 
  n ≥ 52 → 
  52 ≤ n → 
  n % 13 = 0 → 
  n % k = 0 → 
  ∀ m, m < n → (m % 13 = 0 → m % k = 0) → m < 52 →
  k = 4 := by
  sorry

end NUMINAMATH_CALUDE_page_number_divisibility_l1984_198405


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l1984_198454

theorem purely_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (∃ (b : ℝ), (1 - 2*Complex.I)*(a + Complex.I) = b*Complex.I) ↔ a = -2 :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l1984_198454


namespace NUMINAMATH_CALUDE_function_minimum_value_l1984_198461

theorem function_minimum_value (x : ℝ) (h : x ≥ 5) :
  (x^2 - 4*x + 9) / (x - 4) ≥ 10 := by
  sorry

#check function_minimum_value

end NUMINAMATH_CALUDE_function_minimum_value_l1984_198461


namespace NUMINAMATH_CALUDE_mark_balloon_cost_l1984_198480

/-- Represents a bag of water balloons -/
structure BalloonBag where
  price : ℕ
  quantity : ℕ

/-- The available bag sizes -/
def availableBags : List BalloonBag := [
  { price := 4, quantity := 50 },
  { price := 6, quantity := 75 },
  { price := 12, quantity := 200 }
]

/-- The total number of balloons Mark wants to buy -/
def targetBalloons : ℕ := 400

/-- Calculates the minimum cost to buy the target number of balloons -/
def minCost (bags : List BalloonBag) (target : ℕ) : ℕ :=
  sorry

theorem mark_balloon_cost :
  minCost availableBags targetBalloons = 24 :=
sorry

end NUMINAMATH_CALUDE_mark_balloon_cost_l1984_198480


namespace NUMINAMATH_CALUDE_hex_to_binary_bits_l1984_198423

/-- The hexadecimal number A3F52 -/
def hex_number : ℕ := 0xA3F52

/-- The number of bits in the binary representation of a natural number -/
def num_bits (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

theorem hex_to_binary_bits :
  num_bits hex_number = 20 := by sorry

end NUMINAMATH_CALUDE_hex_to_binary_bits_l1984_198423


namespace NUMINAMATH_CALUDE_apple_division_l1984_198401

theorem apple_division (total_apples : ℕ) (total_weight : ℚ) (portions : ℕ) 
  (h1 : total_apples = 28)
  (h2 : total_weight = 3)
  (h3 : portions = 7) :
  (1 : ℚ) / portions = 1 / 7 ∧ total_weight / portions = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_apple_division_l1984_198401


namespace NUMINAMATH_CALUDE_square_sum_equals_eight_l1984_198403

theorem square_sum_equals_eight (a b : ℝ) 
  (h1 : (a + b)^2 = 11) 
  (h2 : (a - b)^2 = 5) : 
  a^2 + b^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_eight_l1984_198403


namespace NUMINAMATH_CALUDE_inverse_proportion_example_l1984_198447

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_example :
  ∀ x y : ℝ → ℝ,
  InverselyProportional x y →
  x 5 = 40 →
  y 5 = 5 →
  y 20 = 20 →
  x 20 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_example_l1984_198447


namespace NUMINAMATH_CALUDE_sum_P_eq_4477547_l1984_198456

/-- P(n) is the product of all non-zero digits of the positive integer n -/
def P (n : ℕ+) : ℕ := sorry

/-- The sum of P(n) for n from 1 to 2009 -/
def sum_P : ℕ := (Finset.range 2009).sum (fun i => P ⟨i + 1, Nat.succ_pos i⟩)

/-- Theorem stating that the sum of P(n) for n from 1 to 2009 is 4477547 -/
theorem sum_P_eq_4477547 : sum_P = 4477547 := by sorry

end NUMINAMATH_CALUDE_sum_P_eq_4477547_l1984_198456


namespace NUMINAMATH_CALUDE_inequality_proof_l1984_198427

theorem inequality_proof (a b : ℝ) : (a^2 - 1) * (b^2 - 1) ≥ 0 → a^2 + b^2 - 1 - a^2*b^2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1984_198427


namespace NUMINAMATH_CALUDE_mikes_stamp_collection_last_page_l1984_198450

/-- Represents the stamp collection problem --/
structure StampCollection where
  initial_books : ℕ
  pages_per_book : ℕ
  initial_stamps_per_page : ℕ
  new_stamps_per_page : ℕ
  filled_books : ℕ
  filled_pages_in_last_book : ℕ

/-- Calculates the number of stamps on the last page after reorganization --/
def stamps_on_last_page (sc : StampCollection) : ℕ :=
  let total_stamps := sc.initial_books * sc.pages_per_book * sc.initial_stamps_per_page
  let filled_pages := sc.filled_books * sc.pages_per_book + sc.filled_pages_in_last_book
  let stamps_on_filled_pages := filled_pages * sc.new_stamps_per_page
  total_stamps - stamps_on_filled_pages

/-- Theorem stating that for Mike's stamp collection, the last page contains 9 stamps --/
theorem mikes_stamp_collection_last_page :
  let sc : StampCollection := {
    initial_books := 6,
    pages_per_book := 30,
    initial_stamps_per_page := 7,
    new_stamps_per_page := 9,
    filled_books := 3,
    filled_pages_in_last_book := 26
  }
  stamps_on_last_page sc = 9 := by
  sorry


end NUMINAMATH_CALUDE_mikes_stamp_collection_last_page_l1984_198450


namespace NUMINAMATH_CALUDE_linear_function_y_axis_intersection_l1984_198441

/-- The coordinates of the intersection point of y = (1/2)x + 1 with the y-axis -/
theorem linear_function_y_axis_intersection :
  let f : ℝ → ℝ := λ x ↦ (1/2) * x + 1
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = f p.1 ∧ p = (0, 1) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_y_axis_intersection_l1984_198441


namespace NUMINAMATH_CALUDE_boys_camp_total_l1984_198433

theorem boys_camp_total (total : ℕ) 
  (h1 : (total : ℝ) * 0.2 * 0.7 = 42) : total = 300 := by
  sorry

#check boys_camp_total

end NUMINAMATH_CALUDE_boys_camp_total_l1984_198433


namespace NUMINAMATH_CALUDE_rectangle_square_ratio_l1984_198418

/-- Configuration of rectangles around a square -/
structure RectangleSquareConfig where
  /-- Side length of the inner square -/
  inner_side : ℝ
  /-- Shorter side of each rectangle -/
  rect_short : ℝ
  /-- Longer side of each rectangle -/
  rect_long : ℝ

/-- Theorem: If the area of the outer square is 9 times that of the inner square,
    then the ratio of the longer side to the shorter side of each rectangle is 2 -/
theorem rectangle_square_ratio (config : RectangleSquareConfig) 
    (h_area : (config.inner_side + 2 * config.rect_short)^2 = 9 * config.inner_side^2) :
    config.rect_long / config.rect_short = 2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_square_ratio_l1984_198418


namespace NUMINAMATH_CALUDE_prime_square_plus_2007p_minus_one_prime_l1984_198451

theorem prime_square_plus_2007p_minus_one_prime (p : ℕ) : 
  Prime p ∧ Prime (p^2 + 2007*p - 1) ↔ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_plus_2007p_minus_one_prime_l1984_198451


namespace NUMINAMATH_CALUDE_largest_prime_factor_l1984_198462

def numbers : List Nat := [85, 57, 119, 143, 169]

def has_largest_prime_factor (n : Nat) (ns : List Nat) : Prop :=
  ∀ m ∈ ns, ∀ p : Nat, p.Prime → p ∣ m → ∃ q : Nat, q.Prime ∧ q ∣ n ∧ q ≥ p

theorem largest_prime_factor :
  has_largest_prime_factor 57 numbers := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l1984_198462


namespace NUMINAMATH_CALUDE_dog_reach_area_l1984_198426

/-- The area outside a regular hexagonal doghouse that a dog can reach when tethered to a vertex --/
theorem dog_reach_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 2 → rope_length = 3 → 
  (area_outside_doghouse : ℝ) = (22 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_dog_reach_area_l1984_198426


namespace NUMINAMATH_CALUDE_numera_transaction_l1984_198424

/-- Represents a number in base s --/
def BaseS (digits : List Nat) (s : Nat) : Nat :=
  digits.foldr (fun d acc => d + s * acc) 0

/-- The transaction in the galaxy of Numera --/
theorem numera_transaction (s : Nat) : 
  s > 1 →  -- s must be greater than 1 to be a valid base
  BaseS [6, 3, 0] s + BaseS [2, 5, 0] s = BaseS [8, 8, 0] s →  -- cost of gadgets
  BaseS [4, 7, 0] s = BaseS [1, 0, 0, 0] s * 2 - BaseS [8, 8, 0] s →  -- change received
  s = 5 := by
  sorry

end NUMINAMATH_CALUDE_numera_transaction_l1984_198424


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l1984_198428

theorem factorial_fraction_simplification : 
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l1984_198428


namespace NUMINAMATH_CALUDE_company_average_service_l1984_198416

/-- Represents a department in the company -/
structure Department where
  employees : ℕ
  total_service : ℕ

/-- The company with two departments -/
structure Company where
  dept_a : Department
  dept_b : Department

/-- Average years of service for a department -/
def avg_service (d : Department) : ℚ :=
  d.total_service / d.employees

/-- Average years of service for the entire company -/
def company_avg_service (c : Company) : ℚ :=
  (c.dept_a.total_service + c.dept_b.total_service) / (c.dept_a.employees + c.dept_b.employees)

theorem company_average_service (k : ℕ) (h_k : k > 0) :
  let c : Company := {
    dept_a := { employees := 7 * k, total_service := 56 * k },
    dept_b := { employees := 5 * k, total_service := 30 * k }
  }
  avg_service c.dept_a = 8 ∧
  avg_service c.dept_b = 6 ∧
  company_avg_service c = 7 + 1/6 :=
by sorry

end NUMINAMATH_CALUDE_company_average_service_l1984_198416


namespace NUMINAMATH_CALUDE_simplify_expression_l1984_198477

theorem simplify_expression (a : ℝ) : (1 : ℝ) * (3 * a) * (5 * a^2) * (7 * a^3) * (9 * a^4) = 945 * a^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1984_198477


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1984_198406

theorem complex_fraction_simplification :
  ((1 - Complex.I) * (1 + 2 * Complex.I)) / (1 + Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1984_198406


namespace NUMINAMATH_CALUDE_not_p_false_sufficient_not_necessary_for_p_or_q_true_l1984_198410

theorem not_p_false_sufficient_not_necessary_for_p_or_q_true (p q : Prop) :
  (¬¬p → p ∨ q) ∧ ∃ (p q : Prop), (p ∨ q) ∧ ¬(¬¬p) := by sorry

end NUMINAMATH_CALUDE_not_p_false_sufficient_not_necessary_for_p_or_q_true_l1984_198410


namespace NUMINAMATH_CALUDE_unique_plane_for_skew_lines_l1984_198497

/-- Two lines in 3D space -/
structure Line3D where
  -- Define properties of a 3D line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a 3D plane

/-- Two lines are skew if they are not coplanar and do not intersect -/
def skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def contained_in (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def parallel_to (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

theorem unique_plane_for_skew_lines (a b : Line3D) 
  (h1 : skew a b) (h2 : ¬perpendicular a b) : 
  ∃! α : Plane3D, contained_in a α ∧ parallel_to b α :=
sorry

end NUMINAMATH_CALUDE_unique_plane_for_skew_lines_l1984_198497


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1984_198452

def A : Set Char := {'a', 'b', 'c', 'd'}
def B : Set Char := {'b', 'c', 'd', 'e'}

theorem intersection_of_A_and_B :
  A ∩ B = {'b', 'c', 'd'} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1984_198452


namespace NUMINAMATH_CALUDE_percentage_changes_l1984_198448

/-- Given an initial value of 950, prove that increasing it by 80% and then
    decreasing the result by 65% yields 598.5. -/
theorem percentage_changes (initial : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) :
  initial = 950 →
  increase_percent = 80 →
  decrease_percent = 65 →
  (initial * (1 + increase_percent / 100)) * (1 - decrease_percent / 100) = 598.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_changes_l1984_198448


namespace NUMINAMATH_CALUDE_prime_dividing_polynomial_congruence_l1984_198439

theorem prime_dividing_polynomial_congruence (n : ℕ) (p : ℕ) (hn : n > 0) (hp : Nat.Prime p) :
  p ∣ (5^(4*n) - 5^(3*n) + 5^(2*n) - 5^n + 1) → p % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_dividing_polynomial_congruence_l1984_198439


namespace NUMINAMATH_CALUDE_inequality_proof_l1984_198470

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  2 * a^2 / (b + c) + 2 * b^2 / (c + a) + 2 * c^2 / (a + b) ≥ 
  a + b + c + (2 * a - b - c)^2 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1984_198470


namespace NUMINAMATH_CALUDE_gcd_840_1764_l1984_198438

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l1984_198438


namespace NUMINAMATH_CALUDE_third_side_possible_length_l1984_198482

/-- Given a triangle with two sides of lengths 3 and 7, 
    prove that 6 is a possible length for the third side. -/
theorem third_side_possible_length :
  ∃ (a b c : ℝ), a = 3 ∧ b = 7 ∧ c = 6 ∧ 
  a + b > c ∧ b + c > a ∧ a + c > b ∧
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_third_side_possible_length_l1984_198482


namespace NUMINAMATH_CALUDE_only_B_is_random_event_l1984_198489

-- Define the type for a die roll
def DieRoll := Fin 6

-- Define the type for a pair of die rolls
def TwoDiceRoll := DieRoll × DieRoll

-- Define the sum of two dice
def diceSum (roll : TwoDiceRoll) : Nat := roll.1.val + roll.2.val + 2

-- Define the sample space
def Ω : Set TwoDiceRoll := Set.univ

-- Define the events
def A : Set TwoDiceRoll := {roll | diceSum roll = 1}
def B : Set TwoDiceRoll := {roll | diceSum roll = 6}
def C : Set TwoDiceRoll := {roll | diceSum roll > 12}
def D : Set TwoDiceRoll := {roll | diceSum roll < 13}

-- Theorem statement
theorem only_B_is_random_event :
  (A = ∅ ∧ B ≠ ∅ ∧ B ≠ Ω ∧ C = ∅ ∧ D = Ω) := by sorry

end NUMINAMATH_CALUDE_only_B_is_random_event_l1984_198489


namespace NUMINAMATH_CALUDE_polynomial_value_l1984_198421

theorem polynomial_value (x y : ℝ) (h : x - y = 5) :
  (x - y)^2 + 2*(x - y) - 10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l1984_198421


namespace NUMINAMATH_CALUDE_possible_values_of_e_l1984_198453

theorem possible_values_of_e :
  ∀ e : ℝ, |2 - e| = 5 → (e = 7 ∨ e = -3) :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_e_l1984_198453


namespace NUMINAMATH_CALUDE_initial_average_calculation_l1984_198442

theorem initial_average_calculation (n : ℕ) (correct_sum incorrect_sum : ℝ) 
  (h1 : n = 10)
  (h2 : correct_sum / n = 18)
  (h3 : incorrect_sum = correct_sum - 46 + 26) :
  incorrect_sum / n = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l1984_198442


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l1984_198449

-- Define the plane vectors
def a (m : ℝ) : Fin 2 → ℝ := ![1, m]
def b : Fin 2 → ℝ := ![2, 5]
def c (m : ℝ) : Fin 2 → ℝ := ![m, 3]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

-- Theorem statement
theorem vector_parallel_condition (m : ℝ) :
  parallel (a m + c m) (a m - b) →
  m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l1984_198449


namespace NUMINAMATH_CALUDE_combined_return_is_ten_percent_l1984_198422

/-- The combined yearly return percentage of two investments -/
def combined_return_percentage (investment1 investment2 return1 return2 : ℚ) : ℚ :=
  ((investment1 * return1 + investment2 * return2) / (investment1 + investment2)) * 100

/-- Theorem: The combined yearly return percentage of a $500 investment with 7% return
    and a $1500 investment with 11% return is 10% -/
theorem combined_return_is_ten_percent :
  combined_return_percentage 500 1500 (7/100) (11/100) = 10 := by
  sorry

end NUMINAMATH_CALUDE_combined_return_is_ten_percent_l1984_198422


namespace NUMINAMATH_CALUDE_first_group_weavers_l1984_198493

/-- The number of weavers in the first group -/
def num_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def mats_first_group : ℕ := 4

/-- The number of days taken by the first group -/
def days_first_group : ℕ := 4

/-- The number of weavers in the second group -/
def weavers_second_group : ℕ := 10

/-- The number of mats woven by the second group -/
def mats_second_group : ℕ := 25

/-- The number of days taken by the second group -/
def days_second_group : ℕ := 10

/-- The rate of weaving is constant across both groups -/
axiom constant_rate : (mats_first_group : ℚ) / (num_weavers * days_first_group) = 
                      (mats_second_group : ℚ) / (weavers_second_group * days_second_group)

theorem first_group_weavers : num_weavers = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_group_weavers_l1984_198493


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l1984_198411

theorem greatest_integer_less_than_negative_fraction :
  ⌊-22/5⌋ = -5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l1984_198411


namespace NUMINAMATH_CALUDE_original_number_proof_l1984_198465

theorem original_number_proof (x : ℝ) : 
  x * 16 = 3408 → 0.16 * 2.13 = 0.3408 → x = 213 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1984_198465


namespace NUMINAMATH_CALUDE_bike_rides_ratio_l1984_198412

/-- Proves that the ratio of John's bike rides to Billy's bike rides is 2:1 --/
theorem bike_rides_ratio : 
  ∀ (john_rides : ℕ),
  (17 : ℕ) + john_rides + (john_rides + 10) = 95 →
  (john_rides : ℚ) / 17 = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_bike_rides_ratio_l1984_198412


namespace NUMINAMATH_CALUDE_triangle_3_4_6_l1984_198415

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the line segments 3, 4, and 6 can form a triangle -/
theorem triangle_3_4_6 : can_form_triangle 3 4 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_3_4_6_l1984_198415


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l1984_198486

theorem arithmetic_evaluation : (7 - 6 * (-5)) - 4 * (-3) / (-2) = 31 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l1984_198486


namespace NUMINAMATH_CALUDE_add_9999_seconds_to_5_45_00_l1984_198444

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time -/
def initialTime : Time :=
  { hours := 5, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 9999

/-- The expected final time -/
def expectedFinalTime : Time :=
  { hours := 8, minutes := 31, seconds := 39 }

theorem add_9999_seconds_to_5_45_00 :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_add_9999_seconds_to_5_45_00_l1984_198444


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l1984_198473

theorem unique_four_digit_number : ∃! n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧
  (n / 10) % 10 = n % 10 + 2 ∧
  (n / 1000) = (n / 100) % 10 + 2 ∧
  n = 9742 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l1984_198473


namespace NUMINAMATH_CALUDE_max_xy_min_ratio_l1984_198469

theorem max_xy_min_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 4) : 
  (∀ a b, a > 0 → b > 0 → a + 2*b = 4 → a*b ≤ x*y) ∧ 
  (∀ a b, a > 0 → b > 0 → a + 2*b = 4 → y/x + 4/y ≤ a/b + 4/a) :=
sorry

end NUMINAMATH_CALUDE_max_xy_min_ratio_l1984_198469


namespace NUMINAMATH_CALUDE_shooting_probabilities_l1984_198478

-- Define probabilities
def prob_A_hit : ℚ := 2/3
def prob_B_hit : ℚ := 3/4

-- Define number of shots
def num_shots : ℕ := 4

-- Define the probability of A missing at least once in 4 shots
def prob_A_miss_at_least_once : ℚ := 1 - prob_A_hit^num_shots

-- Define the probability of A hitting exactly 2 times in 4 shots
def prob_A_hit_exactly_two : ℚ := 
  (num_shots.choose 2 : ℚ) * prob_A_hit^2 * (1 - prob_A_hit)^(num_shots - 2)

-- Define the probability of B hitting exactly 3 times in 4 shots
def prob_B_hit_exactly_three : ℚ :=
  (num_shots.choose 3 : ℚ) * prob_B_hit^3 * (1 - prob_B_hit)^(num_shots - 3)

-- Define the probability of B stopping after exactly 5 shots
def prob_B_stop_after_five : ℚ := 
  prob_B_hit^2 * (1 - prob_B_hit) * (1 - prob_B_hit^2)

theorem shooting_probabilities :
  prob_A_miss_at_least_once = 65/81 ∧
  prob_A_hit_exactly_two * prob_B_hit_exactly_three = 1/8 ∧
  prob_B_stop_after_five = 45/1024 := by sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l1984_198478


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1984_198490

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1984_198490


namespace NUMINAMATH_CALUDE_wills_remaining_money_l1984_198496

/-- Calculates the remaining money after a shopping trip with a refund -/
def remaining_money (initial_amount sweater_price tshirt_price shoes_price refund_percentage : ℚ) : ℚ :=
  initial_amount - sweater_price - tshirt_price + (shoes_price * refund_percentage)

/-- Theorem stating that Will's remaining money after the shopping trip is $81 -/
theorem wills_remaining_money :
  remaining_money 74 9 11 30 0.9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_wills_remaining_money_l1984_198496


namespace NUMINAMATH_CALUDE_vector_subtraction_l1984_198459

/-- Given two plane vectors a and b, prove that a - 2b equals the expected result -/
theorem vector_subtraction (a b : ℝ × ℝ) : 
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1984_198459


namespace NUMINAMATH_CALUDE_equation_solution_l1984_198499

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (x^3 + 2*x^2 + 3*x + 5) / (x + 2) = x + 4 ↔ 
  x = (-3 + Real.sqrt 13) / 4 ∨ x = (-3 - Real.sqrt 13) / 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1984_198499


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l1984_198436

theorem sum_of_squares_problem (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 52)
  (h_sum_products : x*y + y*z + z*x = 24) :
  x + y + z = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l1984_198436


namespace NUMINAMATH_CALUDE_seven_couples_handshakes_l1984_198471

/-- Represents a gathering of couples -/
structure Gathering where
  couples : ℕ
  deriving Repr

/-- Calculates the number of handshakes in a gathering under specific conditions -/
def handshakes (g : Gathering) : ℕ :=
  let total_people := 2 * g.couples
  let handshakes_per_person := total_people - 3  -- Excluding self, spouse, and one other
  (total_people * handshakes_per_person) / 2 - g.couples

/-- Theorem stating that in a gathering of 7 couples, 
    with the given handshake conditions, there are 77 handshakes -/
theorem seven_couples_handshakes :
  handshakes { couples := 7 } = 77 := by
  sorry

#eval handshakes { couples := 7 }

end NUMINAMATH_CALUDE_seven_couples_handshakes_l1984_198471


namespace NUMINAMATH_CALUDE_largest_n_value_l1984_198476

def base_8_to_10 (a b c : ℕ) : ℕ := 64 * a + 8 * b + c

def base_9_to_10 (c b a : ℕ) : ℕ := 81 * c + 9 * b + a

theorem largest_n_value (n : ℕ) (a b c : ℕ) :
  (n > 0) →
  (a < 8 ∧ b < 8 ∧ c < 8) →
  (a ≤ 8 ∧ b ≤ 8 ∧ c ≤ 8) →
  (n = base_8_to_10 a b c) →
  (n = base_9_to_10 c b a) →
  (∀ m, m > 0 ∧ 
    (∃ x y z, x < 8 ∧ y < 8 ∧ z < 8 ∧ m = base_8_to_10 x y z) ∧
    (∃ x y z, x ≤ 8 ∧ y ≤ 8 ∧ z ≤ 8 ∧ m = base_9_to_10 z y x) →
    m ≤ n) →
  n = 511 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_value_l1984_198476


namespace NUMINAMATH_CALUDE_difference_from_sum_and_difference_of_squares_l1984_198468

theorem difference_from_sum_and_difference_of_squares (x y : ℝ) 
  (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_difference_from_sum_and_difference_of_squares_l1984_198468


namespace NUMINAMATH_CALUDE_fraction_equality_l1984_198420

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a / 2 = b / 3) :
  3 / b = 2 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1984_198420


namespace NUMINAMATH_CALUDE_prime_power_sum_l1984_198445

theorem prime_power_sum (a b c d e : ℕ) :
  2^a * 3^b * 5^c * 7^d * 11^e = 27720 →
  2*a + 3*b + 5*c + 7*d + 11*e = 35 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l1984_198445


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l1984_198419

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)
  (distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (positive : a > 0 ∧ b > 0 ∧ c > 0)

-- Define the angle bisector property
def has_equal_angle_bisector_segments (t : Triangle) : Prop :=
  ∃ (d e f : ℝ), d > 0 ∧ e > 0 ∧ f > 0 ∧ d = e

-- Main theorem
theorem angle_bisector_theorem (t : Triangle) 
  (h : has_equal_angle_bisector_segments t) : 
  (t.a / (t.b + t.c) = t.b / (t.c + t.a) + t.c / (t.a + t.b)) ∧ 
  (Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c)) > Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_l1984_198419


namespace NUMINAMATH_CALUDE_tangent_condition_orthogonal_intersection_condition_l1984_198485

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

-- Define the line equation
def line_eq (m x y : ℝ) : Prop := x + m*y = 3

-- Define the tangency condition
def is_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y ∧
  ∀ (x' y' : ℝ), circle_eq x' y' ∧ line_eq m x' y' → (x', y') = (x, y)

-- Define the intersection condition
def intersects_at_orthogonal_points (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    line_eq m x₁ y₁ ∧ line_eq m x₂ y₂ ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧
    x₁ * x₂ + y₁ * y₂ = 0

-- Theorem statements
theorem tangent_condition :
  ∀ m : ℝ, is_tangent m ↔ m = 7/24 :=
sorry

theorem orthogonal_intersection_condition :
  ∀ m : ℝ, intersects_at_orthogonal_points m ↔ (m = 9 + 2*Real.sqrt 14 ∨ m = 9 - 2*Real.sqrt 14) :=
sorry

end NUMINAMATH_CALUDE_tangent_condition_orthogonal_intersection_condition_l1984_198485


namespace NUMINAMATH_CALUDE_solve_system_l1984_198400

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20)
  (eq2 : 6 * p + 5 * q = 29) :
  q = -25 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1984_198400


namespace NUMINAMATH_CALUDE_terminating_decimal_count_l1984_198463

theorem terminating_decimal_count : 
  let n_range := Finset.range 449
  let divisible_by_nine := n_range.filter (λ n => (n + 1) % 9 = 0)
  divisible_by_nine.card = 49 := by
  sorry

end NUMINAMATH_CALUDE_terminating_decimal_count_l1984_198463


namespace NUMINAMATH_CALUDE_four_objects_two_groups_l1984_198407

theorem four_objects_two_groups : ∃ (n : ℕ), n = 14 ∧ 
  n = (Nat.choose 4 1) + (Nat.choose 4 2) + (Nat.choose 4 3) :=
sorry

end NUMINAMATH_CALUDE_four_objects_two_groups_l1984_198407


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l1984_198474

theorem inverse_of_proposition : 
  (∀ x : ℝ, x < 0 → x^2 > 0) → 
  (∀ x : ℝ, x^2 > 0 → x < 0) := by sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l1984_198474


namespace NUMINAMATH_CALUDE_absolute_value_plus_tan_sixty_degrees_l1984_198475

theorem absolute_value_plus_tan_sixty_degrees : 
  |(-2 + Real.sqrt 3)| + Real.tan (π / 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_plus_tan_sixty_degrees_l1984_198475


namespace NUMINAMATH_CALUDE_donuts_left_for_coworkers_l1984_198464

def total_donuts : ℕ := 30
def gluten_free_donuts : ℕ := 12
def regular_donuts : ℕ := 18
def chocolate_gluten_free : ℕ := 6
def plain_gluten_free : ℕ := 6
def chocolate_regular : ℕ := 11
def plain_regular : ℕ := 7

def eaten_while_driving_gluten_free : ℕ := 1
def eaten_while_driving_regular : ℕ := 1

def afternoon_snack_regular : ℕ := 3
def afternoon_snack_gluten_free : ℕ := 3

theorem donuts_left_for_coworkers :
  total_donuts - 
  (eaten_while_driving_gluten_free + eaten_while_driving_regular + 
   afternoon_snack_regular + afternoon_snack_gluten_free) = 23 := by
  sorry

end NUMINAMATH_CALUDE_donuts_left_for_coworkers_l1984_198464


namespace NUMINAMATH_CALUDE_mitzi_food_expense_l1984_198429

/-- Proves that the amount spent on food is $13 given the conditions of Mitzi's amusement park expenses --/
theorem mitzi_food_expense (
  total_brought : ℕ)
  (ticket_cost : ℕ)
  (tshirt_cost : ℕ)
  (money_left : ℕ)
  (h1 : total_brought = 75)
  (h2 : ticket_cost = 30)
  (h3 : tshirt_cost = 23)
  (h4 : money_left = 9)
  : total_brought - money_left - (ticket_cost + tshirt_cost) = 13 := by
  sorry

end NUMINAMATH_CALUDE_mitzi_food_expense_l1984_198429


namespace NUMINAMATH_CALUDE_right_angled_triangle_exists_l1984_198432

/-- A color type with exactly three colors -/
inductive Color
  | Red
  | Green
  | Blue

/-- A point in the cartesian grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each grid point -/
def Coloring := GridPoint → Color

/-- Predicate to check if a triangle is right-angled -/
def isRightAngled (p1 p2 p3 : GridPoint) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0 ∨
  (p1.x - p2.x) * (p3.x - p2.x) + (p1.y - p2.y) * (p3.y - p2.y) = 0 ∨
  (p1.x - p3.x) * (p2.x - p3.x) + (p1.y - p3.y) * (p2.y - p3.y) = 0

/-- Main theorem: There always exists a right-angled triangle with vertices of different colors -/
theorem right_angled_triangle_exists (f : Coloring)
  (h1 : ∃ p : GridPoint, f p = Color.Red)
  (h2 : ∃ p : GridPoint, f p = Color.Green)
  (h3 : ∃ p : GridPoint, f p = Color.Blue) :
  ∃ p1 p2 p3 : GridPoint,
    isRightAngled p1 p2 p3 ∧
    f p1 ≠ f p2 ∧ f p2 ≠ f p3 ∧ f p1 ≠ f p3 :=
by
  sorry


end NUMINAMATH_CALUDE_right_angled_triangle_exists_l1984_198432


namespace NUMINAMATH_CALUDE_parabola_equation_l1984_198435

-- Define the parabola
def Parabola (p : ℝ) := {(x, y) : ℝ × ℝ | y^2 = 4 * p * x}

-- Define the focus of the parabola
def Focus (p : ℝ) : ℝ × ℝ := (p, 0)

-- Theorem statement
theorem parabola_equation (p : ℝ) (h : p = 2) :
  Parabola p = {(x, y) : ℝ × ℝ | y^2 = 8 * x} :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1984_198435


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1984_198425

theorem triangle_angle_measure (a b c A B C : Real) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  -- Given conditions
  (a = Real.sqrt 2) →
  (b = 2) →
  (B = π / 4) →
  -- Conclusion
  (A = π / 6) := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1984_198425


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1984_198481

def A : Set ℝ := {x | Real.tan x > Real.sqrt 3}
def B : Set ℝ := {x | x^2 - 4 < 0}

theorem intersection_of_A_and_B : 
  A ∩ B = Set.Ioo (-2) (-Real.pi/2) ∪ Set.Ioo (Real.pi/3) (Real.pi/2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1984_198481


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1984_198414

/-- The area of a square with side length 12 cm, minus the area of four quarter circles 
    with radius 4 cm (one-third of the square's side length) drawn at each corner, 
    is equal to 144 - 16π cm². -/
theorem shaded_area_calculation (π : Real) : 
  let square_side : Real := 12
  let circle_radius : Real := square_side / 3
  let square_area : Real := square_side ^ 2
  let quarter_circles_area : Real := π * circle_radius ^ 2
  square_area - quarter_circles_area = 144 - 16 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1984_198414


namespace NUMINAMATH_CALUDE_marathon_distance_yards_l1984_198472

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance :=
  (miles : ℕ)
  (yards : ℕ)

/-- Represents a total distance in miles and yards -/
structure TotalDistance :=
  (miles : ℕ)
  (yards : ℕ)

/-- The number of yards in a mile -/
def yardsPerMile : ℕ := 1760

/-- The distance of a single marathon -/
def marathonDistance : MarathonDistance :=
  { miles := 26, yards := 395 }

/-- The number of marathons Leila has run -/
def marathonCount : ℕ := 8

/-- Calculates the total distance run in multiple marathons -/
def totalMarathonDistance (marathonDist : MarathonDistance) (count : ℕ) : TotalDistance :=
  { miles := marathonDist.miles * count,
    yards := marathonDist.yards * count }

/-- Converts a TotalDistance to a normalized form where yards < yardsPerMile -/
def normalizeDistance (dist : TotalDistance) : TotalDistance :=
  { miles := dist.miles + dist.yards / yardsPerMile,
    yards := dist.yards % yardsPerMile }

theorem marathon_distance_yards :
  (normalizeDistance (totalMarathonDistance marathonDistance marathonCount)).yards = 1400 := by
  sorry

end NUMINAMATH_CALUDE_marathon_distance_yards_l1984_198472


namespace NUMINAMATH_CALUDE_sphere_volume_implies_pi_l1984_198483

theorem sphere_volume_implies_pi (D : ℝ) (h : D > 0) :
  (D^3 / 2 + 1 / 21 * D^3 / 2 = π * D^3 / 6) → π = 22 / 7 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_implies_pi_l1984_198483


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_gt_one_l1984_198409

theorem sqrt_meaningful_iff_x_gt_one (x : ℝ) : 
  (∃ y : ℝ, y * y = 1 / (x - 1)) ↔ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_gt_one_l1984_198409


namespace NUMINAMATH_CALUDE_parallel_line_slope_l1984_198413

/-- The slope of a line parallel to 5x - 3y = 12 is 5/3 -/
theorem parallel_line_slope : 
  ∀ (m : ℚ), (∃ b : ℚ, ∀ x y : ℚ, 5 * x - 3 * y = 12 ↔ y = m * x + b) → m = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l1984_198413


namespace NUMINAMATH_CALUDE_equation_solution_l1984_198467

theorem equation_solution : 
  ∃! x : ℝ, (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1984_198467


namespace NUMINAMATH_CALUDE_right_triangle_power_equality_l1984_198491

theorem right_triangle_power_equality (a b c : ℝ) (n : ℕ) 
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_n_gt_2 : n > 2)
  (h_equality : (a^n + b^n + c^n)^2 = 2*(a^(2*n) + b^(2*n) + c^(2*n))) :
  n = 4 := by sorry

end NUMINAMATH_CALUDE_right_triangle_power_equality_l1984_198491


namespace NUMINAMATH_CALUDE_lewis_harvest_weeks_l1984_198434

/-- The number of weeks Lewis works during the harvest -/
def harvest_weeks (total_earnings weekly_earnings : ℕ) : ℕ :=
  total_earnings / weekly_earnings

/-- Proof that Lewis works 5 weeks during the harvest -/
theorem lewis_harvest_weeks :
  harvest_weeks 460 92 = 5 := by
  sorry

end NUMINAMATH_CALUDE_lewis_harvest_weeks_l1984_198434


namespace NUMINAMATH_CALUDE_square_root_of_square_l1984_198437

theorem square_root_of_square (x : ℝ) : Real.sqrt (x^2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_square_l1984_198437


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l1984_198446

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if three numbers are consecutive primes -/
def areConsecutivePrimes (a b c : ℕ) : Prop := sorry

/-- A function that checks if three side lengths can form a triangle -/
def canFormTriangle (a b c : ℕ) : Prop := sorry

/-- The smallest perimeter of a scalene triangle with consecutive prime side lengths and a prime perimeter -/
theorem smallest_prime_perimeter_scalene_triangle :
  ∃ (a b c : ℕ),
    a < b ∧ b < c ∧
    areConsecutivePrimes a b c ∧
    canFormTriangle a b c ∧
    isPrime (a + b + c) ∧
    (a + b + c = 23) ∧
    (∀ (x y z : ℕ),
      x < y ∧ y < z ∧
      areConsecutivePrimes x y z ∧
      canFormTriangle x y z ∧
      isPrime (x + y + z) →
      (x + y + z ≥ 23)) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l1984_198446


namespace NUMINAMATH_CALUDE_guitar_picks_problem_l1984_198402

theorem guitar_picks_problem (total : ℕ) (red blue yellow : ℕ) : 
  total > 0 ∧ 
  red = total / 2 ∧ 
  blue = total / 3 ∧ 
  yellow = 6 ∧ 
  red + blue + yellow = total → 
  blue = 12 := by
sorry

end NUMINAMATH_CALUDE_guitar_picks_problem_l1984_198402


namespace NUMINAMATH_CALUDE_parabola_coefficient_l1984_198479

/-- Proves that for a parabola y = ax^2 + bx + c with vertex at (q,q) and y-intercept at (0, -2q), 
    where q ≠ 0, the coefficient b equals 6/q. -/
theorem parabola_coefficient (a b c q : ℝ) (h1 : q ≠ 0) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (q, q) = ((-b) / (2 * a), a * ((-b) / (2 * a))^2 + b * ((-b) / (2 * a)) + c) →
  -2 * q = c →
  b = 6 / q :=
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l1984_198479


namespace NUMINAMATH_CALUDE_problem_solution_l1984_198404

theorem problem_solution (a b : ℝ) 
  (h1 : 5 + a = 6 - b) 
  (h2 : 3 + b = 8 + a) : 
  5 - a = 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1984_198404


namespace NUMINAMATH_CALUDE_expression_simplification_l1984_198484

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1/m) / ((m^2 - 2*m + 1) / m) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1984_198484


namespace NUMINAMATH_CALUDE_abs_sum_lt_abs_diff_for_opposite_signs_l1984_198440

theorem abs_sum_lt_abs_diff_for_opposite_signs (a b : ℝ) (h : a * b < 0) :
  |a + b| < |a - b| := by sorry

end NUMINAMATH_CALUDE_abs_sum_lt_abs_diff_for_opposite_signs_l1984_198440


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l1984_198443

theorem last_two_digits_sum (n : ℕ) : n = 25 → (15^n + 5^n) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l1984_198443


namespace NUMINAMATH_CALUDE_lottery_prize_probability_l1984_198417

/-- The probability of getting a prize in a lottery with 10 prizes and 25 blanks -/
theorem lottery_prize_probability :
  let num_prizes : ℕ := 10
  let num_blanks : ℕ := 25
  let total_outcomes : ℕ := num_prizes + num_blanks
  let probability : ℚ := num_prizes / total_outcomes
  probability = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_lottery_prize_probability_l1984_198417


namespace NUMINAMATH_CALUDE_shape_e_not_in_square_pieces_l1984_198431

/-- Represents a shape in the diagram -/
structure Shape :=
  (id : String)

/-- Represents the set of shapes in the divided square -/
def SquarePieces : Finset Shape := sorry

/-- Represents the set of given shapes to check -/
def GivenShapes : Finset Shape := sorry

/-- Shape E is defined separately for the theorem -/
def ShapeE : Shape := { id := "E" }

theorem shape_e_not_in_square_pieces :
  ShapeE ∉ SquarePieces ∧
  ∀ s ∈ GivenShapes, s ≠ ShapeE → s ∈ SquarePieces :=
sorry

end NUMINAMATH_CALUDE_shape_e_not_in_square_pieces_l1984_198431


namespace NUMINAMATH_CALUDE_bouquet_cost_55_l1984_198408

/-- The cost of a bouquet of lilies given the number of lilies -/
def bouquet_cost (n : ℕ) : ℚ :=
  (30 : ℚ) * n / 24

theorem bouquet_cost_55 : bouquet_cost 55 = (68750 : ℚ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_cost_55_l1984_198408


namespace NUMINAMATH_CALUDE_equal_distribution_iff_even_total_l1984_198457

/-- Two piles of nuts with different numbers of nuts -/
structure NutPiles :=
  (pile1 : ℕ)
  (pile2 : ℕ)
  (different : pile1 ≠ pile2)

/-- The total number of nuts in both piles -/
def total_nuts (piles : NutPiles) : ℕ := piles.pile1 + piles.pile2

/-- A predicate indicating whether equal distribution is possible -/
def equal_distribution_possible (piles : NutPiles) : Prop :=
  ∃ (k : ℕ), piles.pile1 - k = piles.pile2 + k

/-- Theorem stating that equal distribution is possible if and only if the total number of nuts is even -/
theorem equal_distribution_iff_even_total (piles : NutPiles) :
  equal_distribution_possible piles ↔ Even (total_nuts piles) :=
sorry

end NUMINAMATH_CALUDE_equal_distribution_iff_even_total_l1984_198457


namespace NUMINAMATH_CALUDE_work_completion_rate_l1984_198495

/-- Given that A can finish a work in 12 days and B can do the same work in half the time taken by A,
    prove that working together, they can finish 1/4 of the work in a day. -/
theorem work_completion_rate (days_A : ℕ) (days_B : ℕ) : 
  days_A = 12 →
  days_B = days_A / 2 →
  (1 : ℚ) / days_A + (1 : ℚ) / days_B = (1 : ℚ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_rate_l1984_198495


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l1984_198430

/-- Parabola C₁ with focus F and equation y² = 2px (p > 0) -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  eq : (x y : ℝ) → Prop

/-- Hyperbola C₂ with equation y²/4 - x²/3 = 1 -/
structure Hyperbola where
  eq : (x y : ℝ) → Prop

/-- Two points A and B in the first quadrant -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  first_quadrant : Prop

/-- Area of triangle FAB -/
def triangleArea (F A B : ℝ × ℝ) : ℝ := sorry

/-- Dot product of vectors FA and FB -/
def dotProduct (F A B : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem parabola_hyperbola_intersection
  (C₁ : Parabola)
  (C₂ : Hyperbola)
  (points : IntersectionPoints)
  (h₁ : C₁.p > 0)
  (h₂ : C₁.eq = fun x y ↦ y^2 = 2 * C₁.p * x)
  (h₃ : C₂.eq = fun x y ↦ y^2 / 4 - x^2 / 3 = 1)
  (h₄ : C₁.focus = (C₁.p / 2, 0))
  (h₅ : triangleArea C₁.focus points.A points.B = 2/3 * dotProduct C₁.focus points.A points.B) :
  C₁.p = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l1984_198430


namespace NUMINAMATH_CALUDE_characterization_of_special_numbers_l1984_198487

def is_power_of_two (n : ℕ+) : Prop :=
  ∃ k : ℕ, n = 2^k

def greatest_odd_divisor (n : ℕ+) : ℕ+ :=
  sorry

def smallest_odd_divisor (n : ℕ+) : ℕ+ :=
  sorry

def is_odd_prime (p : ℕ+) : Prop :=
  Nat.Prime p.val ∧ p.val % 2 = 1

theorem characterization_of_special_numbers (n : ℕ+) :
  ¬is_power_of_two n →
  (n = 3 * greatest_odd_divisor n + 5 * smallest_odd_divisor n ↔
    (∃ p : ℕ+, is_odd_prime p ∧ n = 8 * p) ∨ n = 60 ∨ n = 100) :=
  sorry

end NUMINAMATH_CALUDE_characterization_of_special_numbers_l1984_198487


namespace NUMINAMATH_CALUDE_first_team_pies_l1984_198492

/-- Given a catering problem with three teams making pies, prove the number of pies made by the first team. -/
theorem first_team_pies (total_pies : ℕ) (team2_pies : ℕ) (team3_pies : ℕ)
  (h_total : total_pies = 750)
  (h_team2 : team2_pies = 275)
  (h_team3 : team3_pies = 240) :
  total_pies - team2_pies - team3_pies = 235 := by
  sorry

#check first_team_pies

end NUMINAMATH_CALUDE_first_team_pies_l1984_198492


namespace NUMINAMATH_CALUDE_max_sum_constraint_l1984_198455

theorem max_sum_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
  16 * x' * y' * z' = (x' + y')^2 * (x' + z')^2 ∧ x' + y' + z' = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_constraint_l1984_198455


namespace NUMINAMATH_CALUDE_incorrect_exponent_operation_l1984_198488

theorem incorrect_exponent_operation (a : ℝ) (h : a ≠ 0 ∧ a ≠ 1) : (a^2)^3 ≠ a^5 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_exponent_operation_l1984_198488
