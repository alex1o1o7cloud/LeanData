import Mathlib

namespace NUMINAMATH_CALUDE_complex_set_sum_l4026_402618

def is_closed_under_multiplication (S : Set ℂ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S

theorem complex_set_sum (a b c d : ℂ) :
  let S : Set ℂ := {a, b, c, d}
  is_closed_under_multiplication S →
  a^2 = 1 →
  b = 1 →
  c^2 = a →
  b + c + d = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_set_sum_l4026_402618


namespace NUMINAMATH_CALUDE_smallest_number_with_condition_condition_satisfied_by_725_l4026_402658

def ends_with_five (n : ℕ) : Prop := n % 10 = 5

def proper_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range (n - 1)).filter (fun d => d ≠ 1 ∧ n % d = 0)

def divisors_condition (n : ℕ) : Prop :=
  let divs := proper_divisors n
  let largest_sum := (Finset.max' divs (by sorry) + Finset.max' (divs.erase (Finset.max' divs (by sorry))) (by sorry))
  let smallest_sum := (Finset.min' divs (by sorry) + Finset.min' (divs.erase (Finset.min' divs (by sorry))) (by sorry))
  ¬(largest_sum % smallest_sum = 0)

theorem smallest_number_with_condition :
  ∀ n : ℕ, n < 725 → ¬(ends_with_five n ∧ divisors_condition n) :=
by sorry

theorem condition_satisfied_by_725 :
  ends_with_five 725 ∧ divisors_condition 725 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_condition_condition_satisfied_by_725_l4026_402658


namespace NUMINAMATH_CALUDE_coltons_stickers_coltons_initial_stickers_l4026_402637

theorem coltons_stickers (friends_count : ℕ) (stickers_per_friend : ℕ) 
  (extra_for_mandy : ℕ) (less_for_justin : ℕ) (stickers_left : ℕ) : ℕ :=
  let friends_total := friends_count * stickers_per_friend
  let mandy_stickers := friends_total + extra_for_mandy
  let justin_stickers := mandy_stickers - less_for_justin
  let given_away := friends_total + mandy_stickers + justin_stickers
  given_away + stickers_left

theorem coltons_initial_stickers : 
  coltons_stickers 3 4 2 10 42 = 72 := by sorry

end NUMINAMATH_CALUDE_coltons_stickers_coltons_initial_stickers_l4026_402637


namespace NUMINAMATH_CALUDE_probabilities_in_mathematics_l4026_402632

def word : String := "mathematics"

def is_vowel (c : Char) : Bool :=
  c ∈ ['a', 'e', 'i', 'o', 'u']

def count_char (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

def count_vowels (s : String) : Nat :=
  s.toList.filter is_vowel |>.length

theorem probabilities_in_mathematics :
  (count_char word 't' : ℚ) / word.length = 2 / 11 ∧
  (count_vowels word : ℚ) / word.length = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probabilities_in_mathematics_l4026_402632


namespace NUMINAMATH_CALUDE_only_zero_solution_l4026_402612

theorem only_zero_solution (n : ℕ) : 
  (∃ k : ℤ, (30 * n + 2) = k * (12 * n + 1)) ↔ n = 0 :=
by sorry

end NUMINAMATH_CALUDE_only_zero_solution_l4026_402612


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_range_of_a_l4026_402606

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 7| + 1

-- Theorem for the solution set of f(x) ≤ x
theorem solution_set_f_leq_x :
  {x : ℝ | f x ≤ x} = {x : ℝ | 8/3 ≤ x ∧ x ≤ 6} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x - 2*|x - 1| ≤ a) ↔ a ≥ -4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_range_of_a_l4026_402606


namespace NUMINAMATH_CALUDE_die_roll_probability_l4026_402651

theorem die_roll_probability (p_greater_than_four : ℚ) 
  (h : p_greater_than_four = 1/3) : 
  1 - p_greater_than_four = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l4026_402651


namespace NUMINAMATH_CALUDE_count_sequences_eq_fib_21_l4026_402671

/-- The number of increasing sequences satisfying the given conditions -/
def count_sequences : ℕ := sorry

/-- The 21st Fibonacci number -/
def fib_21 : ℕ := sorry

/-- Predicate for valid sequences -/
def valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ i, 1 ≤ a i ∧ a i ≤ 20) ∧
  (∀ i j, i < j → a i < a j) ∧
  (∀ i, a i % 2 = i % 2)

theorem count_sequences_eq_fib_21 : count_sequences = fib_21 := by
  sorry

end NUMINAMATH_CALUDE_count_sequences_eq_fib_21_l4026_402671


namespace NUMINAMATH_CALUDE_excellent_credit_prob_expectation_X_l4026_402689

/-- Credit score distribution --/
def credit_distribution : Finset (ℕ × ℕ) := {(150, 25), (120, 60), (100, 65), (80, 35), (0, 15)}

/-- Total population --/
def total_population : ℕ := 200

/-- Voucher allocation function --/
def voucher (score : ℕ) : ℕ :=
  if score > 150 then 100
  else if score > 100 then 50
  else 0

/-- Probability of selecting 2 people with excellent credit --/
theorem excellent_credit_prob : 
  (Nat.choose 25 2 : ℚ) / (Nat.choose total_population 2) = 3 / 199 := by sorry

/-- Distribution of total vouchers X for 2 randomly selected people --/
def voucher_distribution : Finset (ℕ × ℚ) := {(0, 1/16), (50, 5/16), (100, 29/64), (150, 5/32), (200, 1/64)}

/-- Expectation of X --/
theorem expectation_X : 
  (voucher_distribution.sum (λ (x, p) => x * p)) = 175 / 2 := by sorry

end NUMINAMATH_CALUDE_excellent_credit_prob_expectation_X_l4026_402689


namespace NUMINAMATH_CALUDE_unique_element_in_A_l4026_402685

/-- The set A defined by the quadratic equation ax^2 - x + 1 = 0 -/
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - x + 1 = 0}

/-- The theorem stating that if A contains only one element, then a = 0 or a = 1/4 -/
theorem unique_element_in_A (a : ℝ) : (∃! x, x ∈ A a) → a = 0 ∨ a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_element_in_A_l4026_402685


namespace NUMINAMATH_CALUDE_ms_hatcher_students_l4026_402676

def total_students (third_graders : ℕ) : ℕ :=
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  let sixth_graders := (third_graders + fourth_graders) * 3 / 4
  third_graders + fourth_graders + fifth_graders + sixth_graders

theorem ms_hatcher_students :
  total_students 20 = 115 := by
  sorry

end NUMINAMATH_CALUDE_ms_hatcher_students_l4026_402676


namespace NUMINAMATH_CALUDE_sequence_general_formula_l4026_402626

/-- Given a sequence {a_n} where a₁ = 6 and aₙ₊₁/aₙ = (n+3)/n for n ≥ 1,
    this theorem states that aₙ = n(n+1)(n+2) for all n ≥ 1 -/
theorem sequence_general_formula (a : ℕ → ℝ) 
    (h1 : a 1 = 6)
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = (n + 3) / n) :
  ∀ n : ℕ, n ≥ 1 → a n = n * (n + 1) * (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_formula_l4026_402626


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l4026_402611

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space represented by (x - h)² + (y - k)² = r² -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if a point (x, y) lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  ∃ (x y : ℝ), l.contains x y ∧ (x - c.h)^2 + (y - c.k)^2 = c.r^2 ∧
  ∀ (x' y' : ℝ), l.contains x' y' → (x' - c.h)^2 + (y' - c.k)^2 ≥ c.r^2

theorem tangent_lines_to_circle (c : Circle) (p : ℝ × ℝ) :
  c.h = 0 ∧ c.k = 0 ∧ c.r = 3 ∧ p = (3, 1) →
  ∃ (l1 l2 : Line),
    (l1.a = 4 ∧ l1.b = 3 ∧ l1.c = -15) ∧
    (l2.a = 1 ∧ l2.b = 0 ∧ l2.c = -3) ∧
    l1.contains p.1 p.2 ∧
    l2.contains p.1 p.2 ∧
    isTangent l1 c ∧
    isTangent l2 c ∧
    ∀ (l : Line), l.contains p.1 p.2 ∧ isTangent l c → l = l1 ∨ l = l2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l4026_402611


namespace NUMINAMATH_CALUDE_vet_donation_l4026_402609

theorem vet_donation (dog_fee : ℕ) (cat_fee : ℕ) (dog_adoptions : ℕ) (cat_adoptions : ℕ) 
  (h1 : dog_fee = 15)
  (h2 : cat_fee = 13)
  (h3 : dog_adoptions = 8)
  (h4 : cat_adoptions = 3) :
  (dog_fee * dog_adoptions + cat_fee * cat_adoptions) / 3 = 53 := by
  sorry

end NUMINAMATH_CALUDE_vet_donation_l4026_402609


namespace NUMINAMATH_CALUDE_water_depth_at_points_l4026_402695

/-- The depth function that calculates water depth based on Ron's height -/
def depth (x : ℝ) : ℝ := 16 * x

/-- Ron's height at point A -/
def ronHeightA : ℝ := 13

/-- Ron's height at point B -/
def ronHeightB : ℝ := ronHeightA + 4

/-- Theorem: The depth of water at points A and B -/
theorem water_depth_at_points : 
  depth ronHeightA = 208 ∧ depth ronHeightB = 272 := by
  sorry

/-- Dean's height relative to Ron -/
def deanHeight (ronHeight : ℝ) : ℝ := ronHeight + 9

/-- Alex's height relative to Dean -/
def alexHeight (deanHeight : ℝ) : ℝ := deanHeight - 5

end NUMINAMATH_CALUDE_water_depth_at_points_l4026_402695


namespace NUMINAMATH_CALUDE_curve_perimeter_ge_twice_diagonal_curve_perimeter_eq_twice_diagonal_l4026_402679

/-- A closed curve in 2D space -/
structure ClosedCurve where
  -- Add necessary fields/axioms for a closed curve

/-- A rectangle in 2D space -/
structure Rectangle where
  -- Add necessary fields for a rectangle (e.g., width, height, position)

/-- The perimeter of a closed curve -/
noncomputable def perimeter (c : ClosedCurve) : ℝ :=
  sorry

/-- The diagonal length of a rectangle -/
def diagonal (r : Rectangle) : ℝ :=
  sorry

/-- Predicate to check if a curve intersects all sides of a rectangle -/
def intersectsAllSides (c : ClosedCurve) (r : Rectangle) : Prop :=
  sorry

theorem curve_perimeter_ge_twice_diagonal 
  (c : ClosedCurve) (r : Rectangle) 
  (h : intersectsAllSides c r) : 
  perimeter c ≥ 2 * diagonal r :=
sorry

/-- Condition for equality -/
def equalityCondition (c : ClosedCurve) (r : Rectangle) : Prop :=
  sorry

theorem curve_perimeter_eq_twice_diagonal 
  (c : ClosedCurve) (r : Rectangle) 
  (h1 : intersectsAllSides c r)
  (h2 : equalityCondition c r) : 
  perimeter c = 2 * diagonal r :=
sorry

end NUMINAMATH_CALUDE_curve_perimeter_ge_twice_diagonal_curve_perimeter_eq_twice_diagonal_l4026_402679


namespace NUMINAMATH_CALUDE_part_one_part_two_l4026_402660

-- Define the function f
def f (x m : ℝ) : ℝ := 3 * x^2 + m * (m - 6) * x + 5

-- Theorem for part 1
theorem part_one (m : ℝ) : f 1 m > 0 ↔ m > 4 ∨ m < 2 := by sorry

-- Theorem for part 2
theorem part_two (m n : ℝ) : 
  (∀ x, f x m < n ↔ -1 < x ∧ x < 4) → m = 3 ∧ n = 17 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4026_402660


namespace NUMINAMATH_CALUDE_magnitude_of_sum_equals_five_l4026_402662

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b : Fin 2 → ℝ := ![2, 2]

theorem magnitude_of_sum_equals_five :
  ‖vector_a + vector_b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_equals_five_l4026_402662


namespace NUMINAMATH_CALUDE_angle_value_proof_l4026_402673

theorem angle_value_proof (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.cos (α + β) = Real.sin (α - β)) : α = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_proof_l4026_402673


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l4026_402610

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = -6) ∧ (6 * x - 5 * y = 9) ∧ (x = 22 / 3) ∧ (y = 7) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l4026_402610


namespace NUMINAMATH_CALUDE_regular_decagon_angles_l4026_402653

/-- Properties of a regular decagon -/
theorem regular_decagon_angles :
  let n : ℕ := 10  -- number of sides in a decagon
  let exterior_angle : ℝ := 360 / n
  let interior_angle : ℝ := (n - 2) * 180 / n
  exterior_angle = 36 ∧ interior_angle = 144 := by
  sorry

end NUMINAMATH_CALUDE_regular_decagon_angles_l4026_402653


namespace NUMINAMATH_CALUDE_outfit_cost_theorem_l4026_402634

/-- The cost of an outfit given the prices of individual items -/
def outfit_cost (pant_price t_shirt_price jacket_price : ℚ) : ℚ :=
  pant_price + 4 * t_shirt_price + jacket_price

/-- The theorem stating the cost of the outfit given the constraints -/
theorem outfit_cost_theorem (pant_price t_shirt_price jacket_price : ℚ) :
  (4 * pant_price + 8 * t_shirt_price + 2 * jacket_price = 2400) →
  (2 * pant_price + 14 * t_shirt_price + 3 * jacket_price = 2400) →
  (3 * pant_price + 6 * t_shirt_price = 1500) →
  outfit_cost pant_price t_shirt_price jacket_price = 860 := by
  sorry

#eval outfit_cost 340 80 200

end NUMINAMATH_CALUDE_outfit_cost_theorem_l4026_402634


namespace NUMINAMATH_CALUDE_inequality_proof_l4026_402646

theorem inequality_proof (a : ℝ) : 3 * (1 + a^2 + a^4) - (1 + a + a^2)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4026_402646


namespace NUMINAMATH_CALUDE_company_workshops_l4026_402624

/-- Given a total number of employees and a maximum workshop capacity,
    calculate the minimum number of workshops required. -/
def min_workshops (total_employees : ℕ) (max_capacity : ℕ) : ℕ :=
  (total_employees + max_capacity - 1) / max_capacity

/-- Theorem stating the minimum number of workshops required for the given problem -/
theorem company_workshops :
  let total_employees := 56
  let max_capacity := 15
  min_workshops total_employees max_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_company_workshops_l4026_402624


namespace NUMINAMATH_CALUDE_intersection_with_complement_l4026_402661

def U : Set ℕ := {1, 2, 3, 4}
def P : Set ℕ := {1, 2}
def Q : Set ℕ := {2, 3}

theorem intersection_with_complement : P ∩ (U \ Q) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l4026_402661


namespace NUMINAMATH_CALUDE_cafeteria_apples_l4026_402628

/-- The number of apples in the school cafeteria after using some for lunch and buying more. -/
def final_apples (initial : ℕ) (used : ℕ) (bought : ℕ) : ℕ :=
  initial - used + bought

/-- Theorem stating that given the specific numbers in the problem, the final number of apples is 9. -/
theorem cafeteria_apples : final_apples 23 20 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l4026_402628


namespace NUMINAMATH_CALUDE_A_D_independent_l4026_402615

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events A and D
def A : Set Ω := {ω | ω.1 = 0}
def D : Set Ω := {ω | ω.1.val + ω.2.val + 2 = 7}

-- Theorem statement
theorem A_D_independent : P (A ∩ D) = P A * P D := by sorry

end NUMINAMATH_CALUDE_A_D_independent_l4026_402615


namespace NUMINAMATH_CALUDE_expansion_properties_l4026_402620

/-- Represents the coefficient of x^(k/3) in the expansion of (∛x - 3/∛x)^n -/
def coeff (n : ℕ) (k : ℤ) : ℚ :=
  sorry

/-- The sixth term in the expansion -/
def sixth_term (n : ℕ) : ℚ := coeff n (n - 10)

/-- The coefficient of x² in the expansion -/
def x_squared_coeff (n : ℕ) : ℚ := coeff n 6

theorem expansion_properties (n : ℕ) :
  sixth_term n = 0 →
  n = 10 ∧ x_squared_coeff 10 = 405 := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l4026_402620


namespace NUMINAMATH_CALUDE_two_sin_plus_three_cos_l4026_402641

theorem two_sin_plus_three_cos (x : ℝ) : 
  2 * Real.cos x - 3 * Real.sin x = 4 → 
  (2 * Real.sin x + 3 * Real.cos x = 3) ∨ (2 * Real.sin x + 3 * Real.cos x = 1) := by
sorry

end NUMINAMATH_CALUDE_two_sin_plus_three_cos_l4026_402641


namespace NUMINAMATH_CALUDE_chef_pies_total_l4026_402688

theorem chef_pies_total (apple : ℕ) (pecan : ℕ) (pumpkin : ℕ) 
  (h1 : apple = 2) (h2 : pecan = 4) (h3 : pumpkin = 7) : 
  apple + pecan + pumpkin = 13 := by
  sorry

end NUMINAMATH_CALUDE_chef_pies_total_l4026_402688


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l4026_402604

theorem right_triangle_leg_length 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (leg_a : a = 7) 
  (hypotenuse : c = 25) : 
  b = 24 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l4026_402604


namespace NUMINAMATH_CALUDE_data_average_l4026_402638

theorem data_average (a : ℝ) : 
  (1 + 3 + 2 + 5 + a) / 5 = 3 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_data_average_l4026_402638


namespace NUMINAMATH_CALUDE_sphere_cylinder_ratio_l4026_402656

theorem sphere_cylinder_ratio (R : ℝ) (h : R > 0) : 
  let sphere_volume := (4 / 3) * Real.pi * R^3
  let cylinder_volume := 2 * Real.pi * R^3
  let empty_space := cylinder_volume - sphere_volume
  let total_empty_space := 5 * empty_space
  let total_occupied_space := 5 * sphere_volume
  (total_empty_space / total_occupied_space) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sphere_cylinder_ratio_l4026_402656


namespace NUMINAMATH_CALUDE_prob_same_color_l4026_402698

def box_prob (white : ℕ) (black : ℕ) : ℚ :=
  let total := white + black
  let same_color := (white.choose 3) + (black.choose 3)
  let total_combinations := total.choose 3
  same_color / total_combinations

theorem prob_same_color : box_prob 7 9 = 119 / 560 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_l4026_402698


namespace NUMINAMATH_CALUDE_coin_denominations_exist_l4026_402659

theorem coin_denominations_exist : ∃ (S : Finset ℕ), 
  (Finset.card S = 12) ∧ 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6543 → 
    ∃ (T : Finset ℕ), 
      (∀ m ∈ T, m ∈ S) ∧ 
      (Finset.card T ≤ 8) ∧ 
      (Finset.sum T id = n)) :=
sorry

end NUMINAMATH_CALUDE_coin_denominations_exist_l4026_402659


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4026_402633

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 7 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 →
  a + b + c = 5 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l4026_402633


namespace NUMINAMATH_CALUDE_exist_consecutive_lucky_tickets_l4026_402650

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Proposition: There exist two consecutive natural numbers whose sums of digits are both divisible by 7 -/
theorem exist_consecutive_lucky_tickets : ∃ n : ℕ, 7 ∣ sum_of_digits n ∧ 7 ∣ sum_of_digits (n + 1) :=
sorry

end NUMINAMATH_CALUDE_exist_consecutive_lucky_tickets_l4026_402650


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l4026_402684

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {1, 3}

-- Theorem stating that the complement of A in U is {2, 4}
theorem complement_of_A_in_U :
  (U \ A) = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l4026_402684


namespace NUMINAMATH_CALUDE_matrix_crossout_theorem_l4026_402652

theorem matrix_crossout_theorem (M : Matrix (Fin 1000) (Fin 1000) Bool) :
  (∃ (rows : Finset (Fin 1000)), rows.card = 10 ∧
    ∀ j, ∃ i ∈ rows, M i j = true) ∨
  (∃ (cols : Finset (Fin 1000)), cols.card = 10 ∧
    ∀ i, ∃ j ∈ cols, M i j = false) :=
sorry

end NUMINAMATH_CALUDE_matrix_crossout_theorem_l4026_402652


namespace NUMINAMATH_CALUDE_equation_solutions_l4026_402649

theorem equation_solutions :
  (∀ x : ℝ, 2 * (x - 1)^2 = 1 - x ↔ x = 1 ∨ x = 1/2) ∧
  (∀ x : ℝ, 4 * x^2 - 2 * Real.sqrt 3 * x - 1 = 0 ↔ 
    x = (Real.sqrt 3 + Real.sqrt 7) / 4 ∨ x = (Real.sqrt 3 - Real.sqrt 7) / 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4026_402649


namespace NUMINAMATH_CALUDE_sinusoidal_period_l4026_402655

theorem sinusoidal_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.sin (b * x + c) + d) →
  (∃ n : ℕ, n = 4 ∧ (2 * π) / b = (2 * π) / n) →
  b = 4 :=
by sorry

end NUMINAMATH_CALUDE_sinusoidal_period_l4026_402655


namespace NUMINAMATH_CALUDE_car_sales_third_day_l4026_402683

theorem car_sales_third_day 
  (total_sales : ℕ) 
  (first_day : ℕ) 
  (second_day : ℕ) 
  (h1 : total_sales = 57) 
  (h2 : first_day = 14) 
  (h3 : second_day = 16) : 
  total_sales - (first_day + second_day) = 27 := by
sorry

end NUMINAMATH_CALUDE_car_sales_third_day_l4026_402683


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_proof_l4026_402654

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def largest_even_digit_multiple_of_5 : ℕ := 8800

theorem largest_even_digit_multiple_of_5_proof :
  (has_only_even_digits largest_even_digit_multiple_of_5) ∧
  (largest_even_digit_multiple_of_5 < 10000) ∧
  (largest_even_digit_multiple_of_5 % 5 = 0) ∧
  (∀ n : ℕ, n > largest_even_digit_multiple_of_5 →
    ¬(has_only_even_digits n ∧ n < 10000 ∧ n % 5 = 0)) :=
by sorry

#check largest_even_digit_multiple_of_5_proof

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_proof_l4026_402654


namespace NUMINAMATH_CALUDE_twelve_sticks_need_two_breaks_fifteen_sticks_no_breaks_l4026_402625

/-- Given n sticks of lengths 1, 2, ..., n, this function returns the minimum number
    of sticks that need to be broken in half to form a square. If it's possible to form
    a square without breaking any sticks, it returns 0. -/
def minSticksToBreak (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for 12 sticks, we need to break 2 sticks to form a square -/
theorem twelve_sticks_need_two_breaks : minSticksToBreak 12 = 2 :=
  sorry

/-- Theorem stating that for 15 sticks, we can form a square without breaking any sticks -/
theorem fifteen_sticks_no_breaks : minSticksToBreak 15 = 0 :=
  sorry

end NUMINAMATH_CALUDE_twelve_sticks_need_two_breaks_fifteen_sticks_no_breaks_l4026_402625


namespace NUMINAMATH_CALUDE_train_length_l4026_402608

/-- Proves that a train with the given conditions has a length of 1500 meters -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  train_speed = 180 * (1000 / 3600) →  -- Convert 180 km/hr to m/s
  crossing_time = 60 →  -- Convert 1 minute to seconds
  train_length * 2 = train_speed * crossing_time →
  train_length = 1500 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l4026_402608


namespace NUMINAMATH_CALUDE_line_relationships_l4026_402619

/-- Definition of parallel lines based on slopes -/
def parallel (m1 m2 : ℚ) : Prop := m1 = m2

/-- Definition of perpendicular lines based on slopes -/
def perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

/-- The main theorem -/
theorem line_relationships :
  let slopes : List ℚ := [2, -3, 3, 4, -3/2]
  ∃! (pair : (ℚ × ℚ)), pair ∈ (slopes.product slopes) ∧
    (parallel pair.1 pair.2 ∨ perpendicular pair.1 pair.2) ∧
    pair.1 ≠ pair.2 :=
by sorry

end NUMINAMATH_CALUDE_line_relationships_l4026_402619


namespace NUMINAMATH_CALUDE_unique_solution_l4026_402640

/-- Function that calculates the product of digits of a positive integer -/
def product_of_digits (x : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that 12 is the only positive integer solution -/
theorem unique_solution :
  ∃! (x : ℕ+), product_of_digits x = x^2 - 10*x - 22 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l4026_402640


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l4026_402642

-- Define the sample space
def S : Set ℕ := {1, 2, 3, 4, 5}

-- Define event A
def A : Set ℕ := {n ∈ S | n % 2 = 0}

-- Define event B
def B : Set ℕ := {n ∈ S | n % 2 ≠ 0}

-- Theorem statement
theorem events_mutually_exclusive_and_complementary : 
  (A ∩ B = ∅) ∧ (A ∪ B = S) := by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l4026_402642


namespace NUMINAMATH_CALUDE_total_distance_traveled_l4026_402621

theorem total_distance_traveled (v1 v2 v3 : ℝ) (t : ℝ) (h1 : v1 = 2) (h2 : v2 = 6) (h3 : v3 = 6) (h4 : t = 11 / 60) :
  let d := t * (v1⁻¹ + v2⁻¹ + v3⁻¹)⁻¹
  3 * d = 33 / 50 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l4026_402621


namespace NUMINAMATH_CALUDE_simplification_to_5x_squared_l4026_402629

theorem simplification_to_5x_squared (k : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, (x - k*x) * (2*x - k*x) - 3*x * (2*x - k*x) = 5*x^2) ∧ 
  (∀ k : ℝ, (∀ x : ℝ, (x - k*x) * (2*x - k*x) - 3*x * (2*x - k*x) = 5*x^2) → (k = 3 ∨ k = -3)) :=
by sorry

end NUMINAMATH_CALUDE_simplification_to_5x_squared_l4026_402629


namespace NUMINAMATH_CALUDE_gcd_228_1995_l4026_402677

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l4026_402677


namespace NUMINAMATH_CALUDE_composite_sum_of_power_l4026_402601

theorem composite_sum_of_power (n : ℕ) (h : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 4^n = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_sum_of_power_l4026_402601


namespace NUMINAMATH_CALUDE_condition_relationship_l4026_402614

theorem condition_relationship (x : ℝ) : 
  (∀ x, x = Real.sqrt (x + 2) → x^2 = x + 2) ∧ 
  (∃ x, x^2 = x + 2 ∧ x ≠ Real.sqrt (x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l4026_402614


namespace NUMINAMATH_CALUDE_fraction_evaluation_l4026_402605

theorem fraction_evaluation (x : ℝ) (h : x = 3) : (x^6 + 8*x^3 + 16) / (x^3 + 4) = 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l4026_402605


namespace NUMINAMATH_CALUDE_min_value_ab_l4026_402647

theorem min_value_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b - a * b + 3 = 0) :
  9 ≤ a * b ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ - a₀ * b₀ + 3 = 0 ∧ a₀ * b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l4026_402647


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l4026_402600

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l4026_402600


namespace NUMINAMATH_CALUDE_buddy_met_66_boys_l4026_402668

/-- The number of girl students in the third grade -/
def num_girls : ℕ := 57

/-- The total number of third graders Buddy met -/
def total_students : ℕ := 123

/-- The number of boy students Buddy met -/
def num_boys : ℕ := total_students - num_girls

theorem buddy_met_66_boys : num_boys = 66 := by
  sorry

end NUMINAMATH_CALUDE_buddy_met_66_boys_l4026_402668


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l4026_402681

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l4026_402681


namespace NUMINAMATH_CALUDE_smallest_coprime_to_210_l4026_402675

theorem smallest_coprime_to_210 : 
  ∃ (x : ℕ), x > 1 ∧ Nat.gcd x 210 = 1 ∧ ∀ (y : ℕ), y > 1 ∧ y < x → Nat.gcd y 210 ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_coprime_to_210_l4026_402675


namespace NUMINAMATH_CALUDE_tangent_slope_determines_a_l4026_402639

/-- Given a function f(x) = (x^2 + a) / (x + 1), prove that if the slope of the tangent line
    at x = 1 is 1, then a = -1 -/
theorem tangent_slope_determines_a (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (x^2 + a) / (x + 1)
  (deriv f 1 = 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_determines_a_l4026_402639


namespace NUMINAMATH_CALUDE_faster_train_length_l4026_402670

/-- Proves that the length of a faster train is 340 meters given the specified conditions -/
theorem faster_train_length (faster_speed slower_speed : ℝ) (crossing_time : ℝ) : 
  faster_speed = 108 →
  slower_speed = 36 →
  crossing_time = 17 →
  (faster_speed - slower_speed) * crossing_time * (5/18) = 340 :=
by sorry

end NUMINAMATH_CALUDE_faster_train_length_l4026_402670


namespace NUMINAMATH_CALUDE_total_spokes_in_garage_l4026_402644

/-- The number of bicycles in the garage -/
def num_bicycles : ℕ := 4

/-- The number of spokes per wheel -/
def spokes_per_wheel : ℕ := 10

/-- The number of wheels per bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- Theorem: The total number of spokes in the garage is 80 -/
theorem total_spokes_in_garage : 
  num_bicycles * wheels_per_bicycle * spokes_per_wheel = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_spokes_in_garage_l4026_402644


namespace NUMINAMATH_CALUDE_pumpkin_contest_result_l4026_402697

/-- The weight of Brad's pumpkin in pounds -/
def brads_pumpkin : ℕ := 54

/-- The weight of Jessica's pumpkin in pounds -/
def jessicas_pumpkin : ℕ := brads_pumpkin / 2

/-- The weight of Betty's pumpkin in pounds -/
def bettys_pumpkin : ℕ := jessicas_pumpkin * 4

/-- The difference between the heaviest and lightest pumpkin in pounds -/
def pumpkin_weight_difference : ℕ := max brads_pumpkin (max jessicas_pumpkin bettys_pumpkin) - 
                                     min brads_pumpkin (min jessicas_pumpkin bettys_pumpkin)

theorem pumpkin_contest_result : pumpkin_weight_difference = 81 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_contest_result_l4026_402697


namespace NUMINAMATH_CALUDE_animal_sightings_l4026_402678

theorem animal_sightings (january : ℕ) (february : ℕ) (march : ℕ) 
  (h1 : february = 3 * january)
  (h2 : march = february / 2)
  (h3 : january + february + march = 143) :
  january = 26 := by
sorry

end NUMINAMATH_CALUDE_animal_sightings_l4026_402678


namespace NUMINAMATH_CALUDE_zero_product_property_l4026_402696

theorem zero_product_property (x : ℤ) : (∀ y : ℤ, x * y = 0) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_product_property_l4026_402696


namespace NUMINAMATH_CALUDE_direct_proportion_l4026_402666

theorem direct_proportion (x y : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y, y = k * x) ↔ (∃ k : ℝ, k ≠ 0 ∧ y = k * x) :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_l4026_402666


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l4026_402669

theorem simplify_and_evaluate (a : ℝ) (h : a = 19) :
  (1 + 2 / (a - 1)) / ((a^2 + 2*a + 1) / (a - 1)) = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l4026_402669


namespace NUMINAMATH_CALUDE_cos_equality_theorem_l4026_402645

theorem cos_equality_theorem (n : ℤ) :
  0 ≤ n ∧ n ≤ 360 →
  (Real.cos (n * π / 180) = Real.cos (812 * π / 180)) ↔ (n = 92 ∨ n = 268) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_theorem_l4026_402645


namespace NUMINAMATH_CALUDE_unique_sequence_l4026_402692

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n m : ℕ, a (n * m) = a n * a m) ∧
  (∀ k : ℕ, ∃ n > k, Finset.range n = Finset.image a (Finset.range n))

theorem unique_sequence (a : ℕ → ℕ) (h : is_valid_sequence a) : ∀ n : ℕ, a n = n := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_l4026_402692


namespace NUMINAMATH_CALUDE_y_intercept_of_line_y_intercept_specific_line_l4026_402623

/-- The y-intercept of a line with equation ax + by + c = 0 is -c/b when b ≠ 0 -/
theorem y_intercept_of_line (a b c : ℝ) (hb : b ≠ 0) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let y_intercept := {y : ℝ | (0, y) ∈ line}
  y_intercept = {-c/b} :=
by sorry

/-- The y-intercept of the line x + 2y + 1 = 0 is -1/2 -/
theorem y_intercept_specific_line :
  let line := {p : ℝ × ℝ | p.1 + 2 * p.2 + 1 = 0}
  let y_intercept := {y : ℝ | (0, y) ∈ line}
  y_intercept = {-1/2} :=
by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_y_intercept_specific_line_l4026_402623


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l4026_402631

-- Define the complex number z
def z : ℂ := (2 - Complex.I) ^ 2

-- Theorem stating that z is in the fourth quadrant
theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l4026_402631


namespace NUMINAMATH_CALUDE_binary_10111_equals_43_base_5_l4026_402693

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its base-5 representation -/
def to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The binary representation of 10111 -/
def binary_10111 : List Bool := [true, true, true, false, true]

theorem binary_10111_equals_43_base_5 :
  to_base_5 (binary_to_decimal binary_10111) = [4, 3] :=
sorry

end NUMINAMATH_CALUDE_binary_10111_equals_43_base_5_l4026_402693


namespace NUMINAMATH_CALUDE_mans_downstream_rate_l4026_402607

/-- The man's rate when rowing downstream, given his rate in still water and the current's rate -/
def downstream_rate (still_water_rate current_rate : ℝ) : ℝ :=
  still_water_rate + current_rate

/-- Theorem: The man's rate when rowing downstream is 32 kmph -/
theorem mans_downstream_rate :
  let still_water_rate : ℝ := 24.5
  let current_rate : ℝ := 7.5
  downstream_rate still_water_rate current_rate = 32 := by
  sorry

end NUMINAMATH_CALUDE_mans_downstream_rate_l4026_402607


namespace NUMINAMATH_CALUDE_speed_with_400_people_l4026_402667

/-- Represents the speed of a spaceship given the number of people on board. -/
def spaceshipSpeed (people : ℕ) : ℝ :=
  sorry

/-- The speed halves for every 100 additional people. -/
axiom speed_halves (n : ℕ) : spaceshipSpeed (n + 100) = (spaceshipSpeed n) / 2

/-- The speed of the spaceship with 200 people on board is 500 km/hr. -/
axiom initial_speed : spaceshipSpeed 200 = 500

/-- The speed of the spaceship with 400 people on board is 125 km/hr. -/
theorem speed_with_400_people : spaceshipSpeed 400 = 125 := by
  sorry

end NUMINAMATH_CALUDE_speed_with_400_people_l4026_402667


namespace NUMINAMATH_CALUDE_surface_area_ratio_of_cubes_l4026_402690

theorem surface_area_ratio_of_cubes (a b : ℝ) (h : a > 0) (k : b > 0) (ratio : a = 4 * b) :
  (6 * a^2) / (6 * b^2) = 16 := by sorry

end NUMINAMATH_CALUDE_surface_area_ratio_of_cubes_l4026_402690


namespace NUMINAMATH_CALUDE_initial_average_weight_l4026_402603

theorem initial_average_weight (n : ℕ) (A : ℝ) : 
  (n * A + 90 = (n + 1) * (A - 1)) ∧ 
  (n * A + 110 = (n + 1) * (A + 4)) →
  A = 94 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_weight_l4026_402603


namespace NUMINAMATH_CALUDE_special_parallelogram_existence_l4026_402699

/-- The existence of a special parallelogram for any point on an ellipse -/
theorem special_parallelogram_existence (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 →
    ∃ (p q r s : ℝ × ℝ),
      -- P is on the ellipse
      (x, y) = p ∧
      -- PQRS forms a parallelogram
      (p.1 - q.1 = r.1 - s.1 ∧ p.2 - q.2 = r.2 - s.2) ∧
      (p.1 - s.1 = q.1 - r.1 ∧ p.2 - s.2 = q.2 - r.2) ∧
      -- Parallelogram is tangent to the ellipse
      (∃ (t : ℝ × ℝ), t.1^2/a^2 + t.2^2/b^2 = 1 ∧
        ((t.1 - p.1) * (q.1 - p.1) + (t.2 - p.2) * (q.2 - p.2) = 0 ∨
         (t.1 - q.1) * (r.1 - q.1) + (t.2 - q.2) * (r.2 - q.2) = 0 ∨
         (t.1 - r.1) * (s.1 - r.1) + (t.2 - r.2) * (s.2 - r.2) = 0 ∨
         (t.1 - s.1) * (p.1 - s.1) + (t.2 - s.2) * (p.2 - s.2) = 0)) ∧
      -- Parallelogram is externally tangent to the unit circle
      (∃ (u : ℝ × ℝ), u.1^2 + u.2^2 = 1 ∧
        ((u.1 - p.1) * (q.1 - p.1) + (u.2 - p.2) * (q.2 - p.2) = 0 ∨
         (u.1 - q.1) * (r.1 - q.1) + (u.2 - q.2) * (r.2 - q.2) = 0 ∨
         (u.1 - r.1) * (s.1 - r.1) + (u.2 - r.2) * (s.2 - r.2) = 0 ∨
         (u.1 - s.1) * (p.1 - s.1) + (u.2 - s.2) * (p.2 - s.2) = 0))) ↔
  1/a^2 + 1/b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_special_parallelogram_existence_l4026_402699


namespace NUMINAMATH_CALUDE_division_remainder_proof_l4026_402622

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 1375 →
  divisor = 66 →
  quotient = 20 →
  dividend = divisor * quotient + remainder →
  remainder = 55 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l4026_402622


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruence_solution_l4026_402613

theorem smallest_four_digit_congruence_solution :
  let x : ℕ := 1011
  (∀ y : ℕ, y < x → y < 1000 ∨ ¬(5 * y ≡ 25 [ZMOD 20] ∧ 
                                 3 * y + 10 ≡ 19 [ZMOD 7] ∧ 
                                 y + 3 ≡ 2 * y [ZMOD 12])) ∧
  (5 * x ≡ 25 [ZMOD 20] ∧ 
   3 * x + 10 ≡ 19 [ZMOD 7] ∧ 
   x + 3 ≡ 2 * x [ZMOD 12]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruence_solution_l4026_402613


namespace NUMINAMATH_CALUDE_booster_club_tickets_l4026_402691

/-- Represents the ticket information for the Booster Club trip --/
structure TicketInfo where
  num_nine_dollar : Nat
  total_cost : Nat
  cost_seven : Nat
  cost_nine : Nat

/-- Calculates the total number of tickets bought given the ticket information --/
def total_tickets (info : TicketInfo) : Nat :=
  info.num_nine_dollar + (info.total_cost - info.num_nine_dollar * info.cost_nine) / info.cost_seven

/-- Theorem stating that given the specific ticket information, the total number of tickets is 29 --/
theorem booster_club_tickets :
  let info : TicketInfo := {
    num_nine_dollar := 11,
    total_cost := 225,
    cost_seven := 7,
    cost_nine := 9
  }
  total_tickets info = 29 := by sorry

end NUMINAMATH_CALUDE_booster_club_tickets_l4026_402691


namespace NUMINAMATH_CALUDE_survivor_same_tribe_probability_l4026_402694

/-- The probability that both quitters are from the same tribe in a Survivor-like game. -/
theorem survivor_same_tribe_probability :
  let total_contestants : ℕ := 18
  let tribe_size : ℕ := 9
  let immune_contestants : ℕ := 1
  let quitters : ℕ := 2
  let contestants_at_risk : ℕ := total_contestants - immune_contestants
  let same_tribe_quitters : ℕ := 2 * (tribe_size.choose quitters)
  let total_quitter_combinations : ℕ := contestants_at_risk.choose quitters
  (same_tribe_quitters : ℚ) / total_quitter_combinations = 9 / 17 :=
by sorry

end NUMINAMATH_CALUDE_survivor_same_tribe_probability_l4026_402694


namespace NUMINAMATH_CALUDE_sandwich_count_l4026_402657

theorem sandwich_count (billy_sandwiches : ℕ) (katelyn_extra : ℕ) :
  billy_sandwiches = 49 →
  katelyn_extra = 47 →
  (billy_sandwiches + katelyn_extra + billy_sandwiches + (billy_sandwiches + katelyn_extra) / 4 = 169) :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_count_l4026_402657


namespace NUMINAMATH_CALUDE_group_collection_theorem_l4026_402617

/-- Calculates the total collection amount in rupees for a group of students -/
def totalCollectionInRupees (groupSize : ℕ) : ℚ :=
  (groupSize * groupSize : ℚ) / 100

/-- Theorem: The total collection amount for a group of 45 students is 20.25 rupees -/
theorem group_collection_theorem :
  totalCollectionInRupees 45 = 20.25 := by
  sorry

#eval totalCollectionInRupees 45

end NUMINAMATH_CALUDE_group_collection_theorem_l4026_402617


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l4026_402616

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l4026_402616


namespace NUMINAMATH_CALUDE_only_B_and_C_participate_l4026_402680

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a type for the activity participation
def Activity := Person → Prop

-- Define the conditions
def condition1 (act : Activity) : Prop := act Person.A → act Person.B
def condition2 (act : Activity) : Prop := ¬act Person.C → ¬act Person.B
def condition3 (act : Activity) : Prop := act Person.C → ¬act Person.D

-- Define the property of exactly two people participating
def exactlyTwo (act : Activity) : Prop :=
  ∃ (p1 p2 : Person), p1 ≠ p2 ∧ act p1 ∧ act p2 ∧ ∀ (p : Person), act p → (p = p1 ∨ p = p2)

-- The main theorem
theorem only_B_and_C_participate :
  ∀ (act : Activity),
    condition1 act →
    condition2 act →
    condition3 act →
    exactlyTwo act →
    act Person.B ∧ act Person.C ∧ ¬act Person.A ∧ ¬act Person.D :=
by sorry

end NUMINAMATH_CALUDE_only_B_and_C_participate_l4026_402680


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l4026_402635

theorem trigonometric_equation_solution (t : ℝ) : 
  2 * (Real.sin t)^4 * (Real.sin (2 * t) - 3) - 2 * (Real.sin t)^2 * (Real.sin (2 * t) - 3) - 1 = 0 ↔ 
  (∃ k : ℤ, t = π/4 * (4 * k + 1)) ∨ 
  (∃ n : ℤ, t = (-1)^n * (1/2 * Real.arcsin (1 - Real.sqrt 3)) + π/2 * n) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l4026_402635


namespace NUMINAMATH_CALUDE_rhombus_prism_volume_l4026_402630

/-- A right prism with a rhombus base -/
structure RhombusPrism where
  /-- The acute angle of the rhombus base -/
  α : ℝ
  /-- The length of the larger diagonal of the rhombus base -/
  l : ℝ
  /-- The angle between the larger diagonal and the base plane -/
  β : ℝ
  /-- The acute angle condition -/
  h_α_acute : 0 < α ∧ α < π / 2
  /-- The positive length condition -/
  h_l_pos : l > 0
  /-- The angle β condition -/
  h_β_acute : 0 < β ∧ β < π / 2

/-- The volume of a rhombus-based right prism -/
noncomputable def volume (p : RhombusPrism) : ℝ :=
  1/2 * p.l^3 * Real.sin p.β * Real.cos p.β^2 * Real.tan (p.α/2)

theorem rhombus_prism_volume (p : RhombusPrism) :
  volume p = 1/2 * p.l^3 * Real.sin p.β * Real.cos p.β^2 * Real.tan (p.α/2) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_prism_volume_l4026_402630


namespace NUMINAMATH_CALUDE_bottles_recycled_l4026_402648

def bottle_deposit : ℚ := 10 / 100
def can_deposit : ℚ := 5 / 100
def cans_recycled : ℕ := 140
def total_earned : ℚ := 15

theorem bottles_recycled : 
  ∃ (bottles : ℕ), (bottles : ℚ) * bottle_deposit + (cans_recycled : ℚ) * can_deposit = total_earned ∧ bottles = 80 := by
  sorry

end NUMINAMATH_CALUDE_bottles_recycled_l4026_402648


namespace NUMINAMATH_CALUDE_internal_diagonal_cubes_l4026_402672

theorem internal_diagonal_cubes (a b c : ℕ) (ha : a = 200) (hb : b = 300) (hc : c = 350) :
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c) = 700 := by
  sorry

end NUMINAMATH_CALUDE_internal_diagonal_cubes_l4026_402672


namespace NUMINAMATH_CALUDE_two_preserving_transformations_l4026_402664

/-- Represents the regular, infinite pattern of squares and line segments along a line ℓ -/
structure RegularPattern :=
  (ℓ : Line)
  (square_size : ℝ)
  (diagonal_length : ℝ)

/-- Enumeration of the four types of rigid motion transformations -/
inductive RigidMotion
  | Rotation
  | Translation
  | ReflectionAcross
  | ReflectionPerpendicular

/-- Predicate to check if a rigid motion maps the pattern onto itself -/
def preserves_pattern (r : RegularPattern) (m : RigidMotion) : Prop :=
  sorry

/-- The main theorem stating that exactly two rigid motions preserve the pattern -/
theorem two_preserving_transformations (r : RegularPattern) :
  ∃! (s : Finset RigidMotion), s.card = 2 ∧ ∀ m ∈ s, preserves_pattern r m :=
sorry

end NUMINAMATH_CALUDE_two_preserving_transformations_l4026_402664


namespace NUMINAMATH_CALUDE_place_three_after_correct_l4026_402674

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : ℕ :=
  10 * n.tens + n.units

/-- The result of placing 3 after a two-digit number -/
def place_three_after (n : TwoDigitNumber) : ℕ :=
  100 * n.tens + 10 * n.units + 3

theorem place_three_after_correct (n : TwoDigitNumber) :
  place_three_after n = 100 * n.tens + 10 * n.units + 3 := by
  sorry

end NUMINAMATH_CALUDE_place_three_after_correct_l4026_402674


namespace NUMINAMATH_CALUDE_smallest_three_digit_twice_in_pascal_l4026_402663

/-- Represents a position in Pascal's triangle by row and column -/
structure PascalPosition where
  row : Nat
  col : Nat
  h : col ≤ row

/-- Returns the value at a given position in Pascal's triangle -/
def pascal_value (pos : PascalPosition) : Nat :=
  sorry

/-- Predicate to check if a number appears at least twice in Pascal's triangle -/
def appears_twice (n : Nat) : Prop :=
  ∃ (pos1 pos2 : PascalPosition), pos1 ≠ pos2 ∧ pascal_value pos1 = n ∧ pascal_value pos2 = n

/-- The smallest three-digit number is 100 -/
def smallest_three_digit : Nat := 100

theorem smallest_three_digit_twice_in_pascal :
  (appears_twice smallest_three_digit) ∧
  (∀ n : Nat, n < smallest_three_digit → ¬(appears_twice n ∧ n ≥ 100)) :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_twice_in_pascal_l4026_402663


namespace NUMINAMATH_CALUDE_unique_prime_factorization_l4026_402643

theorem unique_prime_factorization : 
  ∃! (d e f : ℕ), 
    d.Prime ∧ e.Prime ∧ f.Prime ∧ 
    d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
    d * e * f = 7902 ∧
    d + e + f = 1322 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_factorization_l4026_402643


namespace NUMINAMATH_CALUDE_change_calculation_l4026_402665

def shirt_price : ℕ := 5
def sandal_price : ℕ := 3
def num_shirts : ℕ := 10
def num_sandals : ℕ := 3
def payment : ℕ := 100

def total_cost : ℕ := shirt_price * num_shirts + sandal_price * num_sandals

theorem change_calculation : payment - total_cost = 41 := by
  sorry

end NUMINAMATH_CALUDE_change_calculation_l4026_402665


namespace NUMINAMATH_CALUDE_binomial_20_19_l4026_402602

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end NUMINAMATH_CALUDE_binomial_20_19_l4026_402602


namespace NUMINAMATH_CALUDE_cos_pi_plus_alpha_l4026_402686

theorem cos_pi_plus_alpha (α : Real) (h : Real.sin (π / 2 - α) = 3 / 5) :
  Real.cos (π + α) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_plus_alpha_l4026_402686


namespace NUMINAMATH_CALUDE_no_snow_no_fog_probability_l4026_402687

theorem no_snow_no_fog_probability
  (p_snow : ℝ)
  (p_fog_given_no_snow : ℝ)
  (h_p_snow : p_snow = 1/4)
  (h_p_fog_given_no_snow : p_fog_given_no_snow = 1/3) :
  (1 - p_snow) * (1 - p_fog_given_no_snow) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_no_fog_probability_l4026_402687


namespace NUMINAMATH_CALUDE_max_value_of_function_l4026_402636

/-- The function f(x) = -x - 9/x + 18 for x > 0 has a maximum value of 12 -/
theorem max_value_of_function (x : ℝ) (hx : x > 0) :
  ∃ (M : ℝ), M = 12 ∧ ∀ y, y > 0 → -y - 9/y + 18 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l4026_402636


namespace NUMINAMATH_CALUDE_lcm_48_147_l4026_402627

theorem lcm_48_147 : Nat.lcm 48 147 = 2352 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_147_l4026_402627


namespace NUMINAMATH_CALUDE_halfway_point_l4026_402682

theorem halfway_point (a b : ℚ) (ha : a = 1/8) (hb : b = 1/10) :
  (a + b) / 2 = 9/80 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_l4026_402682
