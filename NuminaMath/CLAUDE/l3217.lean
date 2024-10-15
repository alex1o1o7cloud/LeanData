import Mathlib

namespace NUMINAMATH_CALUDE_next_larger_perfect_square_l3217_321773

theorem next_larger_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, n = m^2) ∧ n = x + 4 * (x.sqrt) + 4 := by
  sorry

end NUMINAMATH_CALUDE_next_larger_perfect_square_l3217_321773


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3217_321793

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a < 1 → ∃ x : ℝ, x^2 - 2*x + a = 0) ∧
  ¬(∃ x : ℝ, x^2 - 2*x + a = 0 → a < 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3217_321793


namespace NUMINAMATH_CALUDE_triangle_area_l3217_321772

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  A = 5 * π / 6 → b = 2 → c = 4 →
  (1 / 2) * b * c * Real.sin A = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3217_321772


namespace NUMINAMATH_CALUDE_g_zero_at_neg_one_l3217_321790

-- Define the function g
def g (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + x^2 - 4 * x + s

-- Theorem statement
theorem g_zero_at_neg_one (s : ℝ) : g (-1) s = 0 ↔ s = -5 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_at_neg_one_l3217_321790


namespace NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l3217_321761

theorem square_sum_from_product_and_sum (x y : ℝ) 
  (h1 : x * y = 12) 
  (h2 : x + y = 10) : 
  x^2 + y^2 = 76 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l3217_321761


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l3217_321766

/-- Given three square regions I, II, and III, where the perimeter of region I is 16 units
    and the perimeter of region II is 32 units, the ratio of the area of region II
    to the area of region III is 1/4. -/
theorem area_ratio_of_squares (side_length_I side_length_II side_length_III : ℝ)
    (h1 : side_length_I * 4 = 16)
    (h2 : side_length_II * 4 = 32)
    (h3 : side_length_III = 2 * side_length_II) :
    (side_length_II ^ 2) / (side_length_III ^ 2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l3217_321766


namespace NUMINAMATH_CALUDE_vector_AB_equals_2_2_l3217_321711

def point := ℝ × ℝ

def A : point := (1, 0)
def B : point := (3, 2)

def vector_AB (p q : point) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

theorem vector_AB_equals_2_2 :
  vector_AB A B = (2, 2) := by sorry

end NUMINAMATH_CALUDE_vector_AB_equals_2_2_l3217_321711


namespace NUMINAMATH_CALUDE_boys_in_row_l3217_321730

/-- Represents the number of boys in a row with given conditions -/
def number_of_boys (left_position right_position between : ℕ) : ℕ :=
  left_position + between + right_position

/-- Theorem stating that under the given conditions, the number of boys in the row is 24 -/
theorem boys_in_row :
  let rajan_position := 6
  let vinay_position := 10
  let boys_between := 8
  number_of_boys rajan_position vinay_position boys_between = 24 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_row_l3217_321730


namespace NUMINAMATH_CALUDE_books_to_tables_ratio_l3217_321742

theorem books_to_tables_ratio 
  (num_tables : ℕ) 
  (total_books : ℕ) 
  (h1 : num_tables = 500) 
  (h2 : total_books = 100000) : 
  (total_books / num_tables : ℚ) = 200 := by
sorry

end NUMINAMATH_CALUDE_books_to_tables_ratio_l3217_321742


namespace NUMINAMATH_CALUDE_greatest_common_multiple_15_20_under_150_l3217_321792

def is_common_multiple (n m k : ℕ) : Prop := k % n = 0 ∧ k % m = 0

theorem greatest_common_multiple_15_20_under_150 : 
  (∀ k : ℕ, k < 150 → is_common_multiple 15 20 k → k ≤ 120) ∧ 
  is_common_multiple 15 20 120 ∧ 
  120 < 150 :=
sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_15_20_under_150_l3217_321792


namespace NUMINAMATH_CALUDE_johns_friends_l3217_321710

theorem johns_friends (num_pizzas : ℕ) (slices_per_pizza : ℕ) (slices_per_person : ℕ) :
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  slices_per_person = 4 →
  (num_pizzas * slices_per_pizza) / slices_per_person - 1 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_friends_l3217_321710


namespace NUMINAMATH_CALUDE_gold_bars_theorem_l3217_321794

/-- Represents the masses of five gold bars -/
structure GoldBars where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e
  h2 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e
  h3 : (a = 1 ∧ b = 2) ∨ (a = 1 ∧ c = 2) ∨ (a = 1 ∧ d = 2) ∨ (a = 1 ∧ e = 2) ∨
       (b = 1 ∧ c = 2) ∨ (b = 1 ∧ d = 2) ∨ (b = 1 ∧ e = 2) ∨
       (c = 1 ∧ d = 2) ∨ (c = 1 ∧ e = 2) ∨ (d = 1 ∧ e = 2)

/-- Condition for equal division of remaining bars -/
def canDivideEqually (bars : GoldBars) : Prop :=
  (bars.c + bars.d + bars.e = bars.a + bars.b) ∧
  (bars.b + bars.d + bars.e = bars.a + bars.c) ∧
  (bars.b + bars.c + bars.e = bars.a + bars.d) ∧
  (bars.a + bars.d + bars.e = bars.b + bars.c) ∧
  (bars.a + bars.c + bars.e = bars.b + bars.d) ∧
  (bars.a + bars.b + bars.e = bars.c + bars.d) ∧
  (bars.a + bars.c + bars.d = bars.b + bars.e) ∧
  (bars.a + bars.b + bars.d = bars.c + bars.e) ∧
  (bars.a + bars.b + bars.c = bars.d + bars.e)

/-- The main theorem -/
theorem gold_bars_theorem (bars : GoldBars) (h : canDivideEqually bars) :
  (bars.a = 1 ∧ bars.b = 1 ∧ bars.c = 2 ∧ bars.d = 2 ∧ bars.e = 2) ∨
  (bars.a = 1 ∧ bars.b = 2 ∧ bars.c = 3 ∧ bars.d = 3 ∧ bars.e = 3) ∨
  (bars.a = 1 ∧ bars.b = 1 ∧ bars.c = 1 ∧ bars.d = 1 ∧ bars.e = 2) :=
by sorry

end NUMINAMATH_CALUDE_gold_bars_theorem_l3217_321794


namespace NUMINAMATH_CALUDE_f_2012_eq_neg_2_l3217_321759

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2012_eq_neg_2 (f : ℝ → ℝ) 
  (h1 : is_odd (λ x => f (x + 1)))
  (h2 : is_even (λ x => f (x - 1)))
  (h3 : f 0 = 2) :
  f 2012 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2012_eq_neg_2_l3217_321759


namespace NUMINAMATH_CALUDE_overtake_time_l3217_321716

/-- The time it takes for person B to overtake person A given their speeds and start times -/
theorem overtake_time (speed_A speed_B : ℝ) (start_delay : ℝ) : 
  speed_A = 5 →
  speed_B = 5.555555555555555 →
  start_delay = 0.5 →
  speed_B > speed_A →
  (start_delay * speed_A) / (speed_B - speed_A) = 4.5 := by
  sorry

#check overtake_time

end NUMINAMATH_CALUDE_overtake_time_l3217_321716


namespace NUMINAMATH_CALUDE_a_9_value_l3217_321787

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_9_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 11) / 2 = 15 →
  a 1 + a 2 + a 3 = 9 →
  a 9 = 24 := by
sorry

end NUMINAMATH_CALUDE_a_9_value_l3217_321787


namespace NUMINAMATH_CALUDE_inscribed_circle_probability_l3217_321741

theorem inscribed_circle_probability (a b : ℝ) (h_right_triangle : a = 8 ∧ b = 15) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let r := (a * b) / (2 * s)
  1 - (π * r^2) / (a * b / 2) = 1 - 3 * π / 20 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_probability_l3217_321741


namespace NUMINAMATH_CALUDE_square_root_probability_l3217_321712

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def count_valid_numbers : ℕ := 71

def total_two_digit_numbers : ℕ := 90

theorem square_root_probability : 
  (count_valid_numbers : ℚ) / total_two_digit_numbers = 71 / 90 := by sorry

end NUMINAMATH_CALUDE_square_root_probability_l3217_321712


namespace NUMINAMATH_CALUDE_arrangement_count_l3217_321727

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem arrangement_count :
  let total_people : ℕ := 5
  let total_arrangements := factorial total_people
  let arrangements_with_A_first := factorial (total_people - 1)
  let arrangements_with_B_last := factorial (total_people - 1)
  let arrangements_with_A_first_and_B_last := factorial (total_people - 2)
  total_arrangements - arrangements_with_A_first - arrangements_with_B_last + arrangements_with_A_first_and_B_last = 78 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l3217_321727


namespace NUMINAMATH_CALUDE_union_complement_problem_l3217_321715

theorem union_complement_problem (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
    (hA : A = {3, 4}) (hB : B = {1, 4, 5}) :
  A ∪ (U \ B) = {2, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_union_complement_problem_l3217_321715


namespace NUMINAMATH_CALUDE_triangle_inequality_l3217_321751

theorem triangle_inequality (a b c : ℝ) (x y z : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0)
  (h4 : 0 ≤ x ∧ x ≤ π) (h5 : 0 ≤ y ∧ y ≤ π) (h6 : 0 ≤ z ∧ z ≤ π)
  (h7 : x + y + z = π) : 
  b * c + c * a - a * b < b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ∧
  b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ≤ (1/2) * (a^2 + b^2 + c^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3217_321751


namespace NUMINAMATH_CALUDE_solution_fraction_l3217_321731

theorem solution_fraction (initial_amount : ℝ) (first_day_fraction : ℝ) (second_day_addition : ℝ) : 
  initial_amount = 4 →
  first_day_fraction = 1/2 →
  second_day_addition = 1 →
  (initial_amount - first_day_fraction * initial_amount + second_day_addition) / initial_amount = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_solution_fraction_l3217_321731


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l3217_321718

theorem system_of_inequalities_solution (x : ℝ) :
  (4 * x^2 - 27 * x + 18 > 0 ∧ x^2 + 4 * x + 4 > 0) ↔ 
  ((x < 3/4 ∨ x > 6) ∧ x ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l3217_321718


namespace NUMINAMATH_CALUDE_predicted_distance_is_4km_l3217_321775

/-- Represents the cycling challenge scenario -/
structure CyclingChallenge where
  t : ℝ  -- Time taken to cycle first 1 km
  d : ℝ  -- Predicted distance for remaining time

/-- The cycling challenge satisfies the given conditions -/
def valid_challenge (c : CyclingChallenge) : Prop :=
  c.d = (60 - c.t) / c.t ∧  -- First prediction
  c.d = 384 / (c.t + 36)    -- Second prediction after cycling 15 km in 36 minutes

/-- The predicted distance is 4 km -/
theorem predicted_distance_is_4km (c : CyclingChallenge) 
  (h : valid_challenge c) : c.d = 4 := by
  sorry

#check predicted_distance_is_4km

end NUMINAMATH_CALUDE_predicted_distance_is_4km_l3217_321775


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3217_321795

theorem rationalize_denominator :
  ∃ (A B C D : ℚ),
    (1 / (2 - Real.rpow 7 (1/3 : ℚ)) = Real.rpow A (1/3 : ℚ) + Real.rpow B (1/3 : ℚ) + Real.rpow C (1/3 : ℚ)) ∧
    (A = 4) ∧ (B = 2) ∧ (C = 7) ∧ (D = 1) ∧
    (A + B + C + D = 14) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3217_321795


namespace NUMINAMATH_CALUDE_max_product_constraint_l3217_321700

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (hsum : 3 * a + 2 * b = 1) :
  a * b ≤ 1 / 24 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 1 ∧ a₀ * b₀ = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l3217_321700


namespace NUMINAMATH_CALUDE_price_restoration_l3217_321785

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) (h : reduced_price = 0.8 * original_price) :
  reduced_price * 1.25 = original_price := by
  sorry

end NUMINAMATH_CALUDE_price_restoration_l3217_321785


namespace NUMINAMATH_CALUDE_negation_of_quadratic_equation_l3217_321780

theorem negation_of_quadratic_equation :
  (¬ ∀ x : ℝ, x^2 + 2*x - 1 = 0) ↔ (∃ x : ℝ, x^2 + 2*x - 1 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_equation_l3217_321780


namespace NUMINAMATH_CALUDE_extremum_at_negative_one_l3217_321726

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extremum_at_negative_one (a : ℝ) : 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a (-1) ≤ f a x ∨ f a (-1) ≥ f a x) → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_extremum_at_negative_one_l3217_321726


namespace NUMINAMATH_CALUDE_class_trip_theorem_l3217_321760

/-- Represents the possible solutions for the class trip problem -/
inductive ClassTripSolution
  | five : ClassTripSolution
  | twentyFive : ClassTripSolution

/-- Checks if a given number of students and monthly contribution satisfy the problem conditions -/
def validSolution (numStudents : ℕ) (monthlyContribution : ℕ) : Prop :=
  numStudents * monthlyContribution * 9 = 22725

/-- The main theorem stating that only two solutions exist for the class trip problem -/
theorem class_trip_theorem : 
  ∀ (sol : ClassTripSolution), 
    (sol = ClassTripSolution.five ∧ validSolution 5 505) ∨
    (sol = ClassTripSolution.twentyFive ∧ validSolution 25 101) :=
by sorry

end NUMINAMATH_CALUDE_class_trip_theorem_l3217_321760


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3217_321770

theorem consecutive_even_numbers_sum (a b c d : ℤ) : 
  (∀ n : ℤ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6) →  -- Consecutive even numbers
  (a + b + c + d = 140) →                                        -- Sum condition
  (d = 38) :=                                                    -- Conclusion (largest number)
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3217_321770


namespace NUMINAMATH_CALUDE_max_triangle_area_max_triangle_area_is_156_l3217_321782

/-- The maximum area of the triangle formed by the intersections of three lines in a coordinate plane. -/
theorem max_triangle_area : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (8, 0)
  let C : ℝ × ℝ := (15, 0)
  let ℓ_A := {(x, y) : ℝ × ℝ | y = 2 * x}
  let ℓ_B := {(x, y) : ℝ × ℝ | x = 8}
  let ℓ_C := {(x, y) : ℝ × ℝ | y = -2 * (x - 15)}
  156

/-- The maximum area of the triangle is 156. -/
theorem max_triangle_area_is_156 : max_triangle_area = 156 := by
  sorry

end NUMINAMATH_CALUDE_max_triangle_area_max_triangle_area_is_156_l3217_321782


namespace NUMINAMATH_CALUDE_problem_solution_l3217_321798

theorem problem_solution (x y A : ℝ) 
  (h1 : 2^x = A) 
  (h2 : 7^(2*y) = A) 
  (h3 : 1/x + 1/y = 2) : 
  A = 7 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3217_321798


namespace NUMINAMATH_CALUDE_lamp_arrangement_theorem_l3217_321725

/-- The probability of a specific arrangement of lamps -/
def lamp_arrangement_probability (total_lamps green_lamps on_lamps : ℕ) : ℚ :=
  let favorable_arrangements := (Nat.choose 6 3) * (Nat.choose 7 3)
  let total_arrangements := (Nat.choose total_lamps green_lamps) * (Nat.choose total_lamps on_lamps)
  (favorable_arrangements : ℚ) / total_arrangements

/-- The specific lamp arrangement probability for 8 lamps, 4 green, 4 on -/
def specific_lamp_probability : ℚ := lamp_arrangement_probability 8 4 4

theorem lamp_arrangement_theorem : specific_lamp_probability = 10 / 49 := by
  sorry

end NUMINAMATH_CALUDE_lamp_arrangement_theorem_l3217_321725


namespace NUMINAMATH_CALUDE_haley_small_gardens_l3217_321791

def total_seeds : ℕ := 56
def big_garden_seeds : ℕ := 35
def seeds_per_small_garden : ℕ := 3

def small_gardens : ℕ := (total_seeds - big_garden_seeds) / seeds_per_small_garden

theorem haley_small_gardens : small_gardens = 7 := by
  sorry

end NUMINAMATH_CALUDE_haley_small_gardens_l3217_321791


namespace NUMINAMATH_CALUDE_inequality_implies_a_squared_gt_3b_l3217_321724

theorem inequality_implies_a_squared_gt_3b (a b c : ℝ) 
  (h : a^2 * b^2 + 18 * a * b * c > 4 * b^3 + 4 * a^3 * c + 27 * c^2) : 
  a^2 > 3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_squared_gt_3b_l3217_321724


namespace NUMINAMATH_CALUDE_bens_old_car_cost_l3217_321723

/-- The cost of Ben's old car in dollars -/
def old_car_cost : ℝ := 1900

/-- The cost of Ben's new car in dollars -/
def new_car_cost : ℝ := 3800

/-- The amount Ben received from selling his old car in dollars -/
def old_car_sale : ℝ := 1800

/-- The amount Ben still owes on his new car in dollars -/
def remaining_debt : ℝ := 2000

/-- Theorem stating that the cost of Ben's old car was $1900 -/
theorem bens_old_car_cost :
  old_car_cost = 1900 ∧
  new_car_cost = 2 * old_car_cost ∧
  new_car_cost = old_car_sale + remaining_debt :=
by sorry

end NUMINAMATH_CALUDE_bens_old_car_cost_l3217_321723


namespace NUMINAMATH_CALUDE_exam_candidates_l3217_321784

/-- Given an examination where the average marks obtained is 40 and the total marks are 2000,
    prove that the number of candidates who took the examination is 50. -/
theorem exam_candidates (average_marks : ℕ) (total_marks : ℕ) (h1 : average_marks = 40) (h2 : total_marks = 2000) :
  total_marks / average_marks = 50 := by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_l3217_321784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l3217_321756

theorem arithmetic_sequence_count : 
  let a₁ : ℝ := 4.5
  let aₙ : ℝ := 56.5
  let d : ℝ := 4
  let n := (aₙ - a₁) / d + 1
  n = 14 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l3217_321756


namespace NUMINAMATH_CALUDE_solve_for_y_l3217_321771

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3217_321771


namespace NUMINAMATH_CALUDE_rational_numbers_definition_l3217_321721

-- Define the set of rational numbers
def RationalNumbers : Set ℚ := {q : ℚ | true}

-- Define the set of integers as a subset of rational numbers
def Integers : Set ℚ := {q : ℚ | ∃ (n : ℤ), q = n}

-- Define the set of fractions as a subset of rational numbers
def Fractions : Set ℚ := {q : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b}

-- Theorem stating that rational numbers are the union of integers and fractions
theorem rational_numbers_definition : 
  RationalNumbers = Integers ∪ Fractions := by
  sorry

end NUMINAMATH_CALUDE_rational_numbers_definition_l3217_321721


namespace NUMINAMATH_CALUDE_infinite_series_equality_l3217_321729

theorem infinite_series_equality (a b : ℝ) 
  (h : ∑' n, a / b^n = 6) : 
  ∑' n, a / (a + b)^n = 6/7 := by
sorry

end NUMINAMATH_CALUDE_infinite_series_equality_l3217_321729


namespace NUMINAMATH_CALUDE_brenda_skittles_l3217_321774

/-- Calculates the total number of Skittles Brenda has after buying more. -/
def total_skittles (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem stating that Brenda ends up with 15 Skittles. -/
theorem brenda_skittles : total_skittles 7 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_brenda_skittles_l3217_321774


namespace NUMINAMATH_CALUDE_v2_equals_14_l3217_321709

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qin_jiushao (a b c d e : ℝ) (x : ℝ) : ℝ × ℝ × ℝ := 
  let v₀ := x
  let v₁ := a * x + b
  let v₂ := v₁ * x + c
  (v₀, v₁, v₂)

/-- The theorem stating that v₂ = 14 for the given function and x = 2 -/
theorem v2_equals_14 : 
  let (v₀, v₁, v₂) := qin_jiushao 2 3 0 5 (-4) 2
  v₂ = 14 := by
sorry

end NUMINAMATH_CALUDE_v2_equals_14_l3217_321709


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3217_321704

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

/-- The derivative of a function -/
def HasDerivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, deriv f x = f' x

/-- A quadratic equation has two equal real roots -/
def HasEqualRoots (f : ℝ → ℝ) : Prop :=
  ∃ r : ℝ, (∀ x, f x = 0 ↔ x = r) ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x - r| < δ → |f x| < ε)

/-- The main theorem -/
theorem quadratic_function_theorem (f : ℝ → ℝ) 
  (h1 : QuadraticFunction f)
  (h2 : HasEqualRoots f)
  (h3 : HasDerivative f (λ x ↦ 2 * x + 2)) :
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3217_321704


namespace NUMINAMATH_CALUDE_basketball_score_proof_l3217_321735

theorem basketball_score_proof (two_point_shots three_point_shots free_throws : ℕ) :
  (3 * three_point_shots = 2 * two_point_shots) →
  (free_throws = 2 * three_point_shots) →
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 80) →
  free_throws = 20 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l3217_321735


namespace NUMINAMATH_CALUDE_prob_sum_six_is_five_thirty_sixths_l3217_321722

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 6 * 6

/-- The number of ways to get a sum of 6 when rolling two dice -/
def favorable_outcomes : ℕ := 5

/-- The probability of getting a sum of 6 when rolling two fair dice -/
def prob_sum_six : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_six_is_five_thirty_sixths :
  prob_sum_six = 5 / 36 := by sorry

end NUMINAMATH_CALUDE_prob_sum_six_is_five_thirty_sixths_l3217_321722


namespace NUMINAMATH_CALUDE_balloon_difference_l3217_321708

theorem balloon_difference (allan_balloons jake_balloons : ℕ) : 
  allan_balloons = 5 →
  jake_balloons = 11 →
  jake_balloons > allan_balloons →
  jake_balloons - allan_balloons = 6 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l3217_321708


namespace NUMINAMATH_CALUDE_max_value_of_vector_sum_l3217_321781

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given unit vectors a and b satisfying |3a + 4b| = |4a - 3b|, and a vector c with |c| = 2,
    the maximum value of |a + b - c| is √2 + 2. -/
theorem max_value_of_vector_sum (a b c : E) 
    (ha : ‖a‖ = 1) 
    (hb : ‖b‖ = 1) 
    (hab : ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖) 
    (hc : ‖c‖ = 2) : 
  (‖a + b - c‖ : ℝ) ≤ Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_vector_sum_l3217_321781


namespace NUMINAMATH_CALUDE_problem_part1_problem_part2_l3217_321769

/-- Custom multiplication operation -/
def customMult (a b : ℤ) : ℤ := a^2 - b + a * b

/-- Theorem for the first part of the problem -/
theorem problem_part1 : customMult 2 (-5) = -1 := by sorry

/-- Theorem for the second part of the problem -/
theorem problem_part2 : customMult (-2) (customMult 2 (-3)) = 1 := by sorry

end NUMINAMATH_CALUDE_problem_part1_problem_part2_l3217_321769


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3217_321714

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (1 / (x^2 + y^2) + 1 / (x^2 + z^2) + 1 / (y^2 + z^2)) ≥ 9/2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3217_321714


namespace NUMINAMATH_CALUDE_todds_snow_cone_business_l3217_321728

/-- Todd's snow-cone business problem -/
theorem todds_snow_cone_business 
  (borrowed : ℝ) 
  (repay : ℝ) 
  (ingredients_cost : ℝ) 
  (num_sold : ℕ) 
  (price_per_cone : ℝ) 
  (h1 : borrowed = 100)
  (h2 : repay = 110)
  (h3 : ingredients_cost = 75)
  (h4 : num_sold = 200)
  (h5 : price_per_cone = 0.75)
  : borrowed - ingredients_cost + (num_sold : ℝ) * price_per_cone - repay = 65 :=
by
  sorry


end NUMINAMATH_CALUDE_todds_snow_cone_business_l3217_321728


namespace NUMINAMATH_CALUDE_shaded_fraction_of_large_rectangle_l3217_321754

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

theorem shaded_fraction_of_large_rectangle (large : Rectangle) (small : Rectangle) 
  (h1 : large.width = 15)
  (h2 : large.height = 20)
  (h3 : small.area = (1 / 5) * large.area)
  (h4 : small.area > 0) :
  (1 / 2) * small.area / large.area = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_large_rectangle_l3217_321754


namespace NUMINAMATH_CALUDE_square_to_rectangle_ratio_l3217_321705

theorem square_to_rectangle_ratio : 
  ∀ (square_side : ℝ) (rectangle_base rectangle_height : ℝ),
  square_side = 4 →
  rectangle_base = 2 * Real.sqrt 5 →
  rectangle_height * rectangle_base = square_side^2 →
  rectangle_height / rectangle_base = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_square_to_rectangle_ratio_l3217_321705


namespace NUMINAMATH_CALUDE_unique_solution_l3217_321713

/-- Represents the guesses made by the three friends --/
def friends_guesses : List Nat := [16, 19, 25]

/-- Represents the errors in the guesses --/
def guess_errors : List Nat := [2, 4, 5]

/-- Checks if a number satisfies all constraints --/
def satisfies_constraints (x : Nat) : Prop :=
  ∃ (perm : List Nat), perm.Perm guess_errors ∧
    (friends_guesses.zip perm).all (fun (guess, error) => 
      (guess + error = x) ∨ (guess - error = x))

/-- The theorem stating that 21 is the only number satisfying all constraints --/
theorem unique_solution : 
  satisfies_constraints 21 ∧ ∀ x : Nat, satisfies_constraints x → x = 21 := by
  sorry


end NUMINAMATH_CALUDE_unique_solution_l3217_321713


namespace NUMINAMATH_CALUDE_girls_attending_event_l3217_321748

theorem girls_attending_event (total_students : ℕ) (total_attending : ℕ) 
  (h_total : total_students = 1500)
  (h_attending : total_attending = 900)
  (h_girls_ratio : ∀ g : ℕ, g ≤ total_students → (3 * g) / 4 ≤ total_attending)
  (h_boys_ratio : ∀ b : ℕ, b ≤ total_students → (2 * b) / 5 ≤ total_attending)
  (h_all_students : ∀ g b : ℕ, g + b = total_students → (3 * g) / 4 + (2 * b) / 5 = total_attending) :
  ∃ g : ℕ, g ≤ total_students ∧ (3 * g) / 4 = 643 := by
sorry

end NUMINAMATH_CALUDE_girls_attending_event_l3217_321748


namespace NUMINAMATH_CALUDE_numbers_sum_l3217_321733

/-- Given the conditions about Mickey's, Jayden's, and Coraline's numbers, 
    prove that their sum is 180. -/
theorem numbers_sum (M J C : ℕ) : 
  M = J + 20 →  -- Mickey's number is greater than Jayden's by 20
  J = C - 40 →  -- Jayden's number is 40 less than Coraline's
  C = 80 →      -- Coraline's number is 80
  M + J + C = 180 := by
sorry

end NUMINAMATH_CALUDE_numbers_sum_l3217_321733


namespace NUMINAMATH_CALUDE_z_max_plus_z_min_l3217_321776

theorem z_max_plus_z_min (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 3) 
  (h2 : x + 2*y - 2*z = 4) : 
  ∃ (z_max z_min : ℝ), 
    (∀ z' : ℝ, (x^2 + y^2 + z'^2 = 3 ∧ x + 2*y - 2*z' = 4) → z' ≤ z_max ∧ z' ≥ z_min) ∧
    z_max + z_min = -4 :=
sorry

end NUMINAMATH_CALUDE_z_max_plus_z_min_l3217_321776


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3217_321762

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 3 * x^3 - 5 * x^2 + 6 * x - 8) + (-7 * x^4 - 4 * x^3 + 2 * x^2 - 6 * x + 15) =
  -5 * x^4 - x^3 - 3 * x^2 + 7 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3217_321762


namespace NUMINAMATH_CALUDE_initial_volume_calculation_l3217_321732

theorem initial_volume_calculation (initial_percentage : Real) 
  (final_percentage : Real) (pure_alcohol_added : Real) 
  (h1 : initial_percentage = 0.35)
  (h2 : final_percentage = 0.50)
  (h3 : pure_alcohol_added = 1.8) : 
  ∃ (initial_volume : Real), 
    initial_volume * initial_percentage + pure_alcohol_added = 
    (initial_volume + pure_alcohol_added) * final_percentage ∧ 
    initial_volume = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_volume_calculation_l3217_321732


namespace NUMINAMATH_CALUDE_field_length_proof_l3217_321749

theorem field_length_proof (width : ℝ) (length : ℝ) (pond_side : ℝ) :
  length = 2 * width →
  pond_side = 5 →
  pond_side^2 = (1/8) * (length * width) →
  length = 20 := by
  sorry

end NUMINAMATH_CALUDE_field_length_proof_l3217_321749


namespace NUMINAMATH_CALUDE_parallel_lines_alternate_angles_l3217_321786

/-- Two lines in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Angle between two lines -/
def angle (l1 l2 : Line) : ℝ := sorry

/-- Predicate for parallel lines -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Predicate for a line intersecting two other lines -/
def intersects (l : Line) (l1 l2 : Line) : Prop := sorry

/-- Predicate for alternate interior angles -/
def alternate_interior_angles (l : Line) (l1 l2 : Line) (α β : ℝ) : Prop := sorry

/-- Theorem: If two parallel lines are intersected by a third line, 
    then the alternate interior angles are equal -/
theorem parallel_lines_alternate_angles 
  (l1 l2 l : Line) (α β : ℝ) : 
  parallel l1 l2 → 
  intersects l l1 l2 → 
  alternate_interior_angles l l1 l2 α β →
  α = β := by sorry

end NUMINAMATH_CALUDE_parallel_lines_alternate_angles_l3217_321786


namespace NUMINAMATH_CALUDE_cruise_group_selection_l3217_321788

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem cruise_group_selection :
  choose 9 4 = 126 := by
  sorry

end NUMINAMATH_CALUDE_cruise_group_selection_l3217_321788


namespace NUMINAMATH_CALUDE_tile_perimeter_theorem_l3217_321706

/-- Represents the shape of the tile configuration -/
inductive TileShape
  | L

/-- Represents the possible perimeters after adding tiles -/
def PossiblePerimeters : Set ℕ := {12, 14, 16}

/-- The initial tile configuration -/
structure InitialConfig where
  shape : TileShape
  tileCount : ℕ
  tileSize : ℕ
  perimeter : ℕ

/-- The configuration after adding tiles -/
structure FinalConfig where
  initial : InitialConfig
  addedTiles : ℕ

/-- Predicate to check if a perimeter is possible after adding tiles -/
def IsValidPerimeter (config : FinalConfig) (p : ℕ) : Prop :=
  p ∈ PossiblePerimeters

/-- Main theorem statement -/
theorem tile_perimeter_theorem (config : FinalConfig)
  (h1 : config.initial.shape = TileShape.L)
  (h2 : config.initial.tileCount = 8)
  (h3 : config.initial.tileSize = 1)
  (h4 : config.initial.perimeter = 12)
  (h5 : config.addedTiles = 2) :
  ∃ (p : ℕ), IsValidPerimeter config p :=
sorry

end NUMINAMATH_CALUDE_tile_perimeter_theorem_l3217_321706


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l3217_321768

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_different (a b c d e f : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

theorem digit_sum_puzzle (a b c d e f : ℕ) :
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
  is_valid_digit d ∧ is_valid_digit e ∧ is_valid_digit f ∧
  are_different a b c d e f ∧
  100 * a + 10 * b + c +
  100 * d + 10 * e + a +
  100 * f + 10 * a + b = 1111 →
  a + b + c + d + e + f = 24 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l3217_321768


namespace NUMINAMATH_CALUDE_conference_seating_optimization_l3217_321755

theorem conference_seating_optimization
  (initial_chairs : ℕ)
  (chairs_per_row : ℕ)
  (expected_participants : ℕ)
  (h1 : initial_chairs = 144)
  (h2 : chairs_per_row = 12)
  (h3 : expected_participants = 100)
  : ∃ (chairs_to_remove : ℕ),
    chairs_to_remove = 36 ∧
    (initial_chairs - chairs_to_remove) % chairs_per_row = 0 ∧
    initial_chairs - chairs_to_remove ≥ expected_participants ∧
    ∀ (x : ℕ), x < chairs_to_remove →
      (initial_chairs - x) % chairs_per_row ≠ 0 ∨
      initial_chairs - x > expected_participants + chairs_per_row - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_conference_seating_optimization_l3217_321755


namespace NUMINAMATH_CALUDE_range_of_a_for_sufficient_not_necessary_l3217_321745

theorem range_of_a_for_sufficient_not_necessary (a : ℝ) : 
  (∀ x : ℝ, x < 1 → x < a) ∧ 
  (∃ x : ℝ, x < a ∧ x ≥ 1) ↔ 
  a > 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_sufficient_not_necessary_l3217_321745


namespace NUMINAMATH_CALUDE_fraction_order_l3217_321736

theorem fraction_order : 
  let f1 := 21 / 14
  let f2 := 25 / 18
  let f3 := 23 / 16
  let f4 := 27 / 19
  f2 < f4 ∧ f4 < f3 ∧ f3 < f1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l3217_321736


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3217_321740

/-- Conversion from spherical coordinates to rectangular coordinates -/
theorem spherical_to_rectangular_conversion 
  (ρ θ φ : Real) 
  (hρ : ρ = 8) 
  (hθ : θ = 5 * Real.pi / 4) 
  (hφ : φ = Real.pi / 4) : 
  (ρ * Real.sin φ * Real.cos θ, 
   ρ * Real.sin φ * Real.sin θ, 
   ρ * Real.cos φ) = (-4, -4, 4 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3217_321740


namespace NUMINAMATH_CALUDE_gcd_consecutive_b_terms_l3217_321778

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem gcd_consecutive_b_terms (n : ℕ) (hn : n ≥ 1) : Nat.gcd (b n) (b (n + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_consecutive_b_terms_l3217_321778


namespace NUMINAMATH_CALUDE_cube_difference_l3217_321744

theorem cube_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : a^3 - b^3 = 99 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l3217_321744


namespace NUMINAMATH_CALUDE_marble_selection_ways_l3217_321765

def total_marbles : ℕ := 15
def special_colors : ℕ := 3
def marbles_per_special_color : ℕ := 2
def marbles_to_choose : ℕ := 6
def special_marbles_to_choose : ℕ := 2

def remaining_marbles : ℕ := total_marbles - special_colors * marbles_per_special_color
def remaining_marbles_to_choose : ℕ := marbles_to_choose - special_marbles_to_choose

theorem marble_selection_ways :
  (special_colors * (marbles_per_special_color.choose special_marbles_to_choose)) *
  (remaining_marbles.choose remaining_marbles_to_choose) = 1485 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l3217_321765


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l3217_321707

theorem simultaneous_equations_solution :
  ∀ a b : ℚ,
  (a + b) * (a^2 - b^2) = 4 ∧
  (a - b) * (a^2 + b^2) = 5/2 →
  ((a = 3/2 ∧ b = 1/2) ∨ (a = -1/2 ∧ b = -3/2)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l3217_321707


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_in_binomial_expansion_l3217_321789

theorem coefficient_of_x_cubed_in_binomial_expansion (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ) :
  (∀ x : ℚ, (x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₃ = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_in_binomial_expansion_l3217_321789


namespace NUMINAMATH_CALUDE_existence_of_m_n_l3217_321767

theorem existence_of_m_n (p : Nat) (hp : p.Prime) (hp10 : p > 10) :
  ∃ m n : Nat, m > 0 ∧ n > 0 ∧ m + n < p ∧ (5^m * 7^n - 1) % p = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l3217_321767


namespace NUMINAMATH_CALUDE_ratio_problem_l3217_321702

theorem ratio_problem (a b c : ℚ) 
  (h1 : a / b = (-5/4) / (3/2))
  (h2 : b / c = (2/3) / (-5)) :
  a / c = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l3217_321702


namespace NUMINAMATH_CALUDE_trajectory_of_complex_point_l3217_321764

theorem trajectory_of_complex_point (z : ℂ) (h : Complex.abs z ≤ 1) :
  ∃ (P : ℝ × ℝ), P.1^2 + P.2^2 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_complex_point_l3217_321764


namespace NUMINAMATH_CALUDE_square_sum_and_product_l3217_321796

theorem square_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 49) : 
  x^2 + y^2 = 25 ∧ x * y = -12 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_and_product_l3217_321796


namespace NUMINAMATH_CALUDE_exists_fib_div_1000_l3217_321734

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: There exists a Fibonacci number divisible by 1000 -/
theorem exists_fib_div_1000 : ∃ n : ℕ, 1000 ∣ fib n := by
  sorry

end NUMINAMATH_CALUDE_exists_fib_div_1000_l3217_321734


namespace NUMINAMATH_CALUDE_correct_calculation_l3217_321750

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3217_321750


namespace NUMINAMATH_CALUDE_paint_bottle_cost_l3217_321719

theorem paint_bottle_cost (num_cars num_paintbrushes num_paint_bottles : ℕ)
                          (car_cost paintbrush_cost total_spent : ℚ)
                          (h1 : num_cars = 5)
                          (h2 : num_paintbrushes = 5)
                          (h3 : num_paint_bottles = 5)
                          (h4 : car_cost = 20)
                          (h5 : paintbrush_cost = 2)
                          (h6 : total_spent = 160)
                          : (total_spent - (num_cars * car_cost + num_paintbrushes * paintbrush_cost)) / num_paint_bottles = 10 := by
  sorry

end NUMINAMATH_CALUDE_paint_bottle_cost_l3217_321719


namespace NUMINAMATH_CALUDE_one_third_equals_six_l3217_321797

theorem one_third_equals_six (x : ℝ) : (1 / 3 : ℝ) * x = 6 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_one_third_equals_six_l3217_321797


namespace NUMINAMATH_CALUDE_min_value_expression_l3217_321746

theorem min_value_expression (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (|a - 3*b - 2| + |3*a - b|) / Real.sqrt (a^2 + (b + 1)^2) ≥ 2 ∧
  (a = 0 ∧ b = 0 → (|a - 3*b - 2| + |3*a - b|) / Real.sqrt (a^2 + (b + 1)^2) = 2) :=
by sorry

#check min_value_expression

end NUMINAMATH_CALUDE_min_value_expression_l3217_321746


namespace NUMINAMATH_CALUDE_composite_sequence_l3217_321703

theorem composite_sequence (a n : ℕ) (ha : a ≥ 2) (hn : n > 0) :
  ∃ k : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (a^k + i).Prime = false :=
sorry

end NUMINAMATH_CALUDE_composite_sequence_l3217_321703


namespace NUMINAMATH_CALUDE_circular_arrangement_students_l3217_321737

/-- Given a circular arrangement of students, if the 10th and 45th positions
    are opposite each other, then the total number of students is 70. -/
theorem circular_arrangement_students (n : ℕ) : 
  (10 + n / 2 ≡ 45 [MOD n]) → n = 70 := by sorry

end NUMINAMATH_CALUDE_circular_arrangement_students_l3217_321737


namespace NUMINAMATH_CALUDE_amy_candy_distribution_l3217_321720

/-- Proves that Amy puts 10 candies in each basket given the conditions of the problem -/
theorem amy_candy_distribution (chocolate_bars : ℕ) (num_baskets : ℕ) : 
  chocolate_bars = 5 →
  num_baskets = 25 →
  (chocolate_bars + 7 * chocolate_bars + 6 * (7 * chocolate_bars)) / num_baskets = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_candy_distribution_l3217_321720


namespace NUMINAMATH_CALUDE_perp_bisector_x_intercept_range_l3217_321717

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the perpendicular bisector intersection with x-axis
def perp_bisector_x_intercept (A B : PointOnParabola) : ℝ :=
  sorry -- Definition of x₀ in terms of A and B

-- Theorem statement
theorem perp_bisector_x_intercept_range (A B : PointOnParabola) :
  A ≠ B → perp_bisector_x_intercept A B > 1 :=
sorry

end NUMINAMATH_CALUDE_perp_bisector_x_intercept_range_l3217_321717


namespace NUMINAMATH_CALUDE_point_coordinates_given_distance_to_x_axis_l3217_321738

def distance_to_x_axis (y : ℝ) : ℝ := |y|

theorem point_coordinates_given_distance_to_x_axis (m : ℝ) :
  distance_to_x_axis m = 4 → m = 4 ∨ m = -4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_given_distance_to_x_axis_l3217_321738


namespace NUMINAMATH_CALUDE_special_circle_equation_midpoint_trajectory_l3217_321753

/-- A circle passing through two points with its center on a line -/
structure SpecialCircle where
  -- The circle passes through these two points
  A : ℝ × ℝ := (1, 0)
  B : ℝ × ℝ := (-1, -2)
  -- The center C lies on this line
  center_line : ℝ × ℝ → Prop := fun (x, y) ↦ x - y + 1 = 0

/-- The endpoint B of line segment AB -/
def endpointB : ℝ × ℝ := (4, 3)

theorem special_circle_equation (c : SpecialCircle) :
  ∃ (center : ℝ × ℝ),
    c.center_line center ∧
    ∀ (x y : ℝ), (x + 1)^2 + y^2 = 4 ↔ 
      ((x - center.1)^2 + (y - center.2)^2 = (c.A.1 - center.1)^2 + (c.A.2 - center.2)^2 ∧
       (x - center.1)^2 + (y - center.2)^2 = (c.B.1 - center.1)^2 + (c.B.2 - center.2)^2) :=
sorry

theorem midpoint_trajectory (c : SpecialCircle) :
  ∀ (x y : ℝ), (x - 1.5)^2 + (y - 1.5)^2 = 1 ↔
    ∃ (a : ℝ × ℝ), 
      (a.1 + 1)^2 + a.2^2 = 4 ∧
      x = (a.1 + endpointB.1) / 2 ∧
      y = (a.2 + endpointB.2) / 2 :=
sorry

end NUMINAMATH_CALUDE_special_circle_equation_midpoint_trajectory_l3217_321753


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l3217_321783

theorem log_equality_implies_ratio_one (p q : ℝ) 
  (hp : p > 0) (hq : q > 0)
  (h : Real.log p / Real.log 4 = Real.log q / Real.log 6 ∧ 
       Real.log q / Real.log 6 = Real.log (p * q) / Real.log 8) : 
  q / p = 1 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l3217_321783


namespace NUMINAMATH_CALUDE_second_year_interest_rate_l3217_321747

/-- Calculates the interest rate for the second year given the initial principal,
    first year interest rate, and final amount after two years. -/
theorem second_year_interest_rate
  (initial_principal : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_principal = 4000)
  (h2 : first_year_rate = 0.04)
  (h3 : final_amount = 4368) :
  let first_year_amount := initial_principal * (1 + first_year_rate)
  let second_year_rate := (final_amount / first_year_amount) - 1
  second_year_rate = 0.05 := by
sorry

end NUMINAMATH_CALUDE_second_year_interest_rate_l3217_321747


namespace NUMINAMATH_CALUDE_card_value_decrease_l3217_321739

theorem card_value_decrease (x : ℝ) :
  (1 - x/100) * (1 - x/100) = 0.64 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_card_value_decrease_l3217_321739


namespace NUMINAMATH_CALUDE_total_games_played_l3217_321779

-- Define the structure for the team's season statistics
structure SeasonStats where
  first100WinPercentage : ℝ
  homeWinPercentageAfter100 : ℝ
  awayWinPercentageAfter100 : ℝ
  overallWinPercentage : ℝ
  consecutiveWinStreak : ℕ

-- Define the theorem
theorem total_games_played (stats : SeasonStats) 
  (h1 : stats.first100WinPercentage = 0.85)
  (h2 : stats.homeWinPercentageAfter100 = 0.60)
  (h3 : stats.awayWinPercentageAfter100 = 0.45)
  (h4 : stats.overallWinPercentage = 0.70)
  (h5 : stats.consecutiveWinStreak = 15) :
  ∃ (totalGames : ℕ), totalGames = 186 ∧ 
  (∃ (remainingGames : ℕ), 
    remainingGames % 2 = 0 ∧
    totalGames = 100 + remainingGames ∧
    (85 + (stats.homeWinPercentageAfter100 + stats.awayWinPercentageAfter100) / 2 * remainingGames) / totalGames = stats.overallWinPercentage) :=
by
  sorry

end NUMINAMATH_CALUDE_total_games_played_l3217_321779


namespace NUMINAMATH_CALUDE_polynomial_roots_sum_l3217_321752

theorem polynomial_roots_sum (p q r : ℤ) (m : ℤ) : 
  (∀ x : ℤ, x^3 - 2024*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 104 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_sum_l3217_321752


namespace NUMINAMATH_CALUDE_part_one_part_two_l3217_321757

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2 * |x - 1|

-- Part I
theorem part_one :
  {x : ℝ | f 3 x ≥ 1} = Set.Icc 0 (4/3) := by sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x - |2*x - 5| ≤ 0) →
  a ∈ Set.Icc (-1) 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3217_321757


namespace NUMINAMATH_CALUDE_cube_face_sum_l3217_321777

/-- Represents the six face values of a cube -/
structure CubeFaces where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- Calculates the sum of vertex labels for a given set of cube faces -/
def vertexLabelSum (faces : CubeFaces) : ℕ :=
  faces.a * faces.b * faces.c +
  faces.a * faces.e * faces.c +
  faces.a * faces.b * faces.f +
  faces.a * faces.e * faces.f +
  faces.d * faces.b * faces.c +
  faces.d * faces.e * faces.c +
  faces.d * faces.b * faces.f +
  faces.d * faces.e * faces.f

/-- Theorem stating the sum of face values given the conditions -/
theorem cube_face_sum (faces : CubeFaces)
  (h1 : vertexLabelSum faces = 2002)
  (h2 : faces.a + faces.d = 22) :
  faces.a + faces.b + faces.c + faces.d + faces.e + faces.f = 42 := by
  sorry


end NUMINAMATH_CALUDE_cube_face_sum_l3217_321777


namespace NUMINAMATH_CALUDE_female_officers_count_l3217_321763

theorem female_officers_count (total_on_duty : ℕ) (male_percentage : ℚ) (female_on_duty_percentage : ℚ) :
  total_on_duty = 500 →
  male_percentage = 60 / 100 →
  female_on_duty_percentage = 10 / 100 →
  (female_on_duty_percentage * (total_female_officers : ℕ) : ℚ) = ((1 - male_percentage) * total_on_duty : ℚ) →
  total_female_officers = 2000 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l3217_321763


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3217_321701

theorem cubic_equation_roots (p : ℝ) : 
  (∃ x y z : ℤ, x > 0 ∧ y > 0 ∧ z > 0 ∧
   (∀ t : ℝ, 5*t^3 - 5*(p+1)*t^2 + (71*p - 1)*t + 1 = 66*p ↔ t = x ∨ t = y ∨ t = z))
  ↔ p = 76 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3217_321701


namespace NUMINAMATH_CALUDE_function_proof_l3217_321743

/-- Given a function f(x) = a^x + b, prove that if f(1) = 3 and f(0) = 2, then f(x) = 2^x + 1 -/
theorem function_proof (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a^x + b) 
  (h2 : f 1 = 3) (h3 : f 0 = 2) : ∀ x, f x = 2^x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_proof_l3217_321743


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l3217_321799

def total_crayons : ℕ := 15
def karls_selection : ℕ := 3
def friends_selection : ℕ := 4

def selection_ways : ℕ := Nat.choose total_crayons karls_selection * 
                           Nat.choose (total_crayons - karls_selection) friends_selection

theorem crayon_selection_theorem : 
  selection_ways = 225225 := by sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l3217_321799


namespace NUMINAMATH_CALUDE_f_1991_equals_1988_l3217_321758

/-- Represents the number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ := sorry

/-- Represents the cumulative sum of digits up to r-digit numbers -/
def g (r : ℕ) : ℕ := r * 10^r - (10^r - 1) / 9

/-- 
f(n) represents the number of digits in the number containing the 10^nth digit 
in the sequence of natural numbers written in order without spaces
-/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating that f(1991) = 1988 -/
theorem f_1991_equals_1988 : f 1991 = 1988 := by sorry

end NUMINAMATH_CALUDE_f_1991_equals_1988_l3217_321758
