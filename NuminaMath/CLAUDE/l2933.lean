import Mathlib

namespace NUMINAMATH_CALUDE_product_difference_square_equals_negative_one_l2933_293384

theorem product_difference_square_equals_negative_one :
  2021 * 2023 - 2022^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_square_equals_negative_one_l2933_293384


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2933_293349

/-- 
An arithmetic sequence is a sequence where the difference between 
consecutive terms is constant.
-/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- 
Theorem: In an arithmetic sequence where the sum of the first and fifth terms is 10, 
the third term is equal to 5.
-/
theorem arithmetic_sequence_third_term 
  (a : ℕ → ℚ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 1 + a 5 = 10) : 
  a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2933_293349


namespace NUMINAMATH_CALUDE_negation_of_existence_squared_greater_than_power_of_two_l2933_293352

theorem negation_of_existence_squared_greater_than_power_of_two :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_squared_greater_than_power_of_two_l2933_293352


namespace NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l2933_293335

theorem largest_whole_number_nine_times_less_than_150 :
  ∃ (x : ℕ), x = 16 ∧ (∀ y : ℕ, 9 * y < 150 → y ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_largest_whole_number_nine_times_less_than_150_l2933_293335


namespace NUMINAMATH_CALUDE_common_solution_y_values_l2933_293381

theorem common_solution_y_values : 
  ∀ x y : ℝ, 
  (x^2 + y^2 - 9 = 0 ∧ x^2 + 2*y - 7 = 0) ↔ 
  (y = 1 + Real.sqrt 3 ∨ y = 1 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_y_values_l2933_293381


namespace NUMINAMATH_CALUDE_problem_solution_l2933_293337

def proposition_p (m : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, 2 * x - 2 ≥ m^2 - 3 * m

def proposition_q (m : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-1) 1, x^2 - x - 1 + m ≤ 0

theorem problem_solution (m : ℝ) :
  (proposition_p m ↔ (1 ≤ m ∧ m ≤ 2)) ∧
  ((proposition_p m ∧ ¬proposition_q m) ∨ (¬proposition_p m ∧ proposition_q m) ↔
    (m < 1 ∨ (5/4 < m ∧ m ≤ 2))) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2933_293337


namespace NUMINAMATH_CALUDE_homework_problem_l2933_293332

theorem homework_problem (p t : ℕ) : 
  p ≥ 10 ∧ 
  p * t = (2 * p + 2) * (t + 1) →
  p * t = 60 :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_l2933_293332


namespace NUMINAMATH_CALUDE_max_area_of_rectangle_with_constraints_l2933_293382

/-- Represents a rectangle with sides x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- The perimeter of the rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.x + r.y)

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := r.x * r.y

/-- The condition that one side is at least twice as long as the other -/
def oneSideAtLeastTwiceOther (r : Rectangle) : Prop := r.x ≥ 2 * r.y

theorem max_area_of_rectangle_with_constraints :
  ∃ (r : Rectangle),
    perimeter r = 60 ∧
    oneSideAtLeastTwiceOther r ∧
    area r = 200 ∧
    ∀ (s : Rectangle),
      perimeter s = 60 →
      oneSideAtLeastTwiceOther s →
      area s ≤ area r :=
by sorry

end NUMINAMATH_CALUDE_max_area_of_rectangle_with_constraints_l2933_293382


namespace NUMINAMATH_CALUDE_collinear_points_iff_k_eq_neg_one_l2933_293333

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- The theorem stating the condition for collinearity of the given points -/
theorem collinear_points_iff_k_eq_neg_one (k : ℝ) :
  collinear ⟨3, 1⟩ ⟨6, 4⟩ ⟨10, k + 9⟩ ↔ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_iff_k_eq_neg_one_l2933_293333


namespace NUMINAMATH_CALUDE_balloons_in_park_l2933_293380

/-- The number of balloons Allan and Jake had in the park -/
def total_balloons (allan_initial : ℕ) (jake_balloons : ℕ) (allan_bought : ℕ) : ℕ :=
  (allan_initial + allan_bought) + jake_balloons

/-- Theorem stating the total number of balloons Allan and Jake had in the park -/
theorem balloons_in_park : total_balloons 3 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_balloons_in_park_l2933_293380


namespace NUMINAMATH_CALUDE_orange_business_profit_l2933_293353

/-- Represents the profit calculation for Mr. Smith's orange business --/
theorem orange_business_profit :
  let small_oranges : ℕ := 5
  let medium_oranges : ℕ := 3
  let large_oranges : ℕ := 3
  let small_buy_price : ℚ := 1
  let medium_buy_price : ℚ := 2
  let large_buy_price : ℚ := 3
  let small_sell_price : ℚ := 1.5
  let medium_sell_price : ℚ := 3
  let large_sell_price : ℚ := 4
  let transportation_cost : ℚ := 2
  let storage_fee : ℚ := 1
  
  let total_buy_cost : ℚ := 
    small_oranges * small_buy_price + 
    medium_oranges * medium_buy_price + 
    large_oranges * large_buy_price +
    transportation_cost + storage_fee
  
  let total_sell_revenue : ℚ :=
    small_oranges * small_sell_price +
    medium_oranges * medium_sell_price +
    large_oranges * large_sell_price
  
  let profit : ℚ := total_sell_revenue - total_buy_cost
  
  profit = 5.5 := by sorry

end NUMINAMATH_CALUDE_orange_business_profit_l2933_293353


namespace NUMINAMATH_CALUDE_average_score_three_subjects_l2933_293307

theorem average_score_three_subjects 
  (math_score : ℝ)
  (korean_english_avg : ℝ)
  (h1 : math_score = 100)
  (h2 : korean_english_avg = 88) : 
  (math_score + 2 * korean_english_avg) / 3 = 92 := by
  sorry

end NUMINAMATH_CALUDE_average_score_three_subjects_l2933_293307


namespace NUMINAMATH_CALUDE_divisibility_property_l2933_293397

theorem divisibility_property (n : ℕ) (hn : n > 1) : 
  ∃ k : ℤ, n^(n-1) - 1 = (n-1)^2 * k := by sorry

end NUMINAMATH_CALUDE_divisibility_property_l2933_293397


namespace NUMINAMATH_CALUDE_oscar_swag_bag_value_l2933_293324

/-- The total value of a swag bag with specified items -/
def swag_bag_value (earring_cost : ℕ) (iphone_cost : ℕ) (scarf_cost : ℕ) : ℕ :=
  2 * earring_cost + iphone_cost + 4 * scarf_cost

/-- Theorem: The total value of the Oscar swag bag is $20,000 -/
theorem oscar_swag_bag_value :
  swag_bag_value 6000 2000 1500 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_oscar_swag_bag_value_l2933_293324


namespace NUMINAMATH_CALUDE_lucius_weekly_earnings_l2933_293366

/-- Lucius's small business model --/
structure Business where
  daily_ingredient_cost : ℝ
  french_fries_price : ℝ
  poutine_price : ℝ
  tax_rate : ℝ
  daily_french_fries_sold : ℝ
  daily_poutine_sold : ℝ

/-- Calculate weekly earnings after taxes and expenses --/
def weekly_earnings_after_taxes_and_expenses (b : Business) : ℝ :=
  let daily_revenue := b.french_fries_price * b.daily_french_fries_sold + b.poutine_price * b.daily_poutine_sold
  let weekly_revenue := daily_revenue * 7
  let weekly_expenses := b.daily_ingredient_cost * 7
  let taxable_income := weekly_revenue
  let tax := taxable_income * b.tax_rate
  weekly_revenue - weekly_expenses - tax

/-- Theorem stating Lucius's weekly earnings --/
theorem lucius_weekly_earnings :
  ∃ (b : Business),
    b.daily_ingredient_cost = 10 ∧
    b.french_fries_price = 12 ∧
    b.poutine_price = 8 ∧
    b.tax_rate = 0.1 ∧
    b.daily_french_fries_sold = 1 ∧
    b.daily_poutine_sold = 1 ∧
    weekly_earnings_after_taxes_and_expenses b = 56 := by
  sorry


end NUMINAMATH_CALUDE_lucius_weekly_earnings_l2933_293366


namespace NUMINAMATH_CALUDE_min_white_points_l2933_293325

theorem min_white_points (total_points : ℕ) (h_total : total_points = 100) :
  ∃ (n : ℕ), n = 10 ∧ 
  (∀ (k : ℕ), k < n → k + (k.choose 3) < total_points) ∧
  (n + (n.choose 3) ≥ total_points) := by
  sorry

end NUMINAMATH_CALUDE_min_white_points_l2933_293325


namespace NUMINAMATH_CALUDE_train_length_l2933_293338

/-- Given a train that crosses an electric pole in 2.5 seconds at a speed of 144 km/hr,
    prove that its length is 100 meters. -/
theorem train_length (crossing_time : Real) (speed_kmh : Real) (length : Real) : 
  crossing_time = 2.5 →
  speed_kmh = 144 →
  length = speed_kmh * (1000 / 3600) * crossing_time →
  length = 100 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2933_293338


namespace NUMINAMATH_CALUDE_basketball_game_probability_basketball_game_probability_proof_l2933_293369

/-- The probability that at least 7 out of 8 people stay for an entire basketball game,
    given that 4 are certain to stay and 4 have a 1/3 probability of staying. -/
theorem basketball_game_probability : ℝ :=
  let total_people : ℕ := 8
  let certain_people : ℕ := 4
  let uncertain_people : ℕ := 4
  let stay_probability : ℝ := 1/3
  let at_least_stay : ℕ := 7

  1/9

/-- Proof of the basketball game probability theorem -/
theorem basketball_game_probability_proof :
  basketball_game_probability = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_probability_basketball_game_probability_proof_l2933_293369


namespace NUMINAMATH_CALUDE_line_equivalence_l2933_293326

/-- Given a line expressed in vector form, prove it's equivalent to a specific slope-intercept form -/
theorem line_equivalence (x y : ℝ) : 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y - (-4)) = 0 ↔ y = 2 * x - 10 := by sorry

end NUMINAMATH_CALUDE_line_equivalence_l2933_293326


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2933_293313

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  12 * x - 6 * y + 3 * y - 24 * x = -12 * x - 3 * y := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  (3/2) * (a^2 * b - 2 * a * b^2) - (1/2) * (a * b^2 - 4 * a^2 * b) + (a * b^2 / 2) = 
  (7/2) * a^2 * b - 3 * a * b^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2933_293313


namespace NUMINAMATH_CALUDE_physics_class_size_l2933_293375

theorem physics_class_size 
  (total_students : ℕ) 
  (math_only : ℕ) 
  (physics_only : ℕ) 
  (both : ℕ) :
  total_students = 53 →
  both = 7 →
  physics_only + both = 2 * (math_only + both) →
  total_students = math_only + physics_only + both →
  physics_only + both = 40 := by
  sorry

end NUMINAMATH_CALUDE_physics_class_size_l2933_293375


namespace NUMINAMATH_CALUDE_exactly_two_classical_models_l2933_293322

/-- Represents a random event model -/
structure RandomEventModel where
  is_finite : Bool
  has_equal_likelihood : Bool

/-- Checks if a random event model is a classical probability model -/
def is_classical_probability_model (model : RandomEventModel) : Bool :=
  model.is_finite && model.has_equal_likelihood

/-- The list of random event models given in the problem -/
def models : List RandomEventModel := [
  ⟨false, true⟩,   -- Model 1
  ⟨true, false⟩,   -- Model 2
  ⟨true, true⟩,    -- Model 3
  ⟨false, false⟩,  -- Model 4
  ⟨true, true⟩     -- Model 5
]

theorem exactly_two_classical_models : 
  (models.filter is_classical_probability_model).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_classical_models_l2933_293322


namespace NUMINAMATH_CALUDE_charity_donation_l2933_293367

theorem charity_donation (cassandra_pennies james_difference : ℕ) 
  (h1 : cassandra_pennies = 5000)
  (h2 : james_difference = 276) :
  cassandra_pennies + (cassandra_pennies - james_difference) = 9724 := by
  sorry

end NUMINAMATH_CALUDE_charity_donation_l2933_293367


namespace NUMINAMATH_CALUDE_students_in_one_activity_l2933_293331

/-- Given a school with students participating in two elective courses, 
    prove the number of students in exactly one course. -/
theorem students_in_one_activity 
  (total : ℕ) 
  (both : ℕ) 
  (none : ℕ) 
  (h1 : total = 317) 
  (h2 : both = 30) 
  (h3 : none = 20) : 
  total - both - none = 267 := by
  sorry

#check students_in_one_activity

end NUMINAMATH_CALUDE_students_in_one_activity_l2933_293331


namespace NUMINAMATH_CALUDE_smallest_m_no_real_roots_l2933_293311

theorem smallest_m_no_real_roots : 
  ∃ (m : ℤ), (∀ (n : ℤ), n < m → ∃ (x : ℝ), 3*x*(n*x-5) - x^2 + 8 = 0) ∧
             (∀ (x : ℝ), 3*x*(m*x-5) - x^2 + 8 ≠ 0) ∧
             m = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_no_real_roots_l2933_293311


namespace NUMINAMATH_CALUDE_triangle_inequality_l2933_293310

/-- The inequality for triangle sides and area -/
theorem triangle_inequality (a b c : ℝ) (Δ : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : Δ > 0) : 
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 ∧
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 ↔ a = b ∧ b = c) := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequality_l2933_293310


namespace NUMINAMATH_CALUDE_initial_student_count_l2933_293356

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ) :
  initial_avg = 15 →
  new_avg = 14.4 →
  new_student_weight = 3 →
  ∃ n : ℕ, n * initial_avg + new_student_weight = (n + 1) * new_avg ∧ n = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_student_count_l2933_293356


namespace NUMINAMATH_CALUDE_pet_store_combinations_l2933_293303

/-- The number of puppies available -/
def num_puppies : ℕ := 10

/-- The number of kittens available -/
def num_kittens : ℕ := 7

/-- The number of hamsters available -/
def num_hamsters : ℕ := 9

/-- The number of birds available -/
def num_birds : ℕ := 5

/-- The number of people buying pets -/
def num_people : ℕ := 4

/-- The number of ways to select one pet of each type and assign them to four different people -/
def num_ways : ℕ := num_puppies * num_kittens * num_hamsters * num_birds * Nat.factorial num_people

theorem pet_store_combinations : num_ways = 75600 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l2933_293303


namespace NUMINAMATH_CALUDE_geometric_sequence_statements_l2933_293357

def geometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def increasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

theorem geometric_sequence_statements (a : ℕ → ℝ) (q : ℝ) 
  (h : geometricSequence a q) : 
  (¬(q > 1 → increasingSequence a) ∧
   ¬(increasingSequence a → q > 1) ∧
   ¬(q ≤ 1 → ¬increasingSequence a) ∧
   ¬(¬increasingSequence a → q ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_statements_l2933_293357


namespace NUMINAMATH_CALUDE_triangle_inequality_expression_l2933_293362

theorem triangle_inequality_expression (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^4 + b^4 + c^4 - 2*a^2*b^2 - 2*b^2*c^2 - 2*c^2*a^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_expression_l2933_293362


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l2933_293360

/-- Represents a binary number as a list of bits (least significant bit first) -/
def BinaryNum := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNum) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : Nat) : BinaryNum :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : BinaryNum :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

theorem binary_multiplication_theorem :
  let a : BinaryNum := [true, false, true, true]  -- 1101₂
  let b : BinaryNum := [true, true, true]         -- 111₂
  let result : BinaryNum := [true, true, true, true, false, false, false, false, true] -- 100001111₂
  binary_to_decimal (decimal_to_binary (binary_to_decimal a * binary_to_decimal b)) = binary_to_decimal result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l2933_293360


namespace NUMINAMATH_CALUDE_cubic_function_continuous_l2933_293391

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- State the theorem that f is continuous for all real x
theorem cubic_function_continuous :
  ∀ x : ℝ, ContinuousAt f x :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_function_continuous_l2933_293391


namespace NUMINAMATH_CALUDE_teacher_distribution_l2933_293370

/-- The number of ways to distribute 4 teachers to 3 places -/
def distribute_teachers : ℕ := 36

/-- The number of ways to choose 2 teachers out of 4 -/
def choose_two_from_four : ℕ := Nat.choose 4 2

/-- The number of ways to arrange 3 groups into 3 places -/
def arrange_three_groups : ℕ := 6

theorem teacher_distribution :
  distribute_teachers = choose_two_from_four * arrange_three_groups :=
sorry

end NUMINAMATH_CALUDE_teacher_distribution_l2933_293370


namespace NUMINAMATH_CALUDE_circle_sum_l2933_293398

def Circle := Fin 12 → ℝ

def is_valid_circle (c : Circle) : Prop :=
  (∀ i, c i ≠ 0) ∧
  (∀ i, i % 2 = 0 → c i = c ((i + 11) % 12) + c ((i + 1) % 12)) ∧
  (∀ i, i % 2 = 1 → c i = c ((i + 11) % 12) * c ((i + 1) % 12))

theorem circle_sum (c : Circle) (h : is_valid_circle c) :
  (Finset.sum Finset.univ c) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_l2933_293398


namespace NUMINAMATH_CALUDE_number_comparison_l2933_293371

theorem number_comparison (A B : ℝ) (h : A = B + B / 4) : 
  B = A - A / 5 ∧ B ≠ A - A / 4 := by
  sorry

end NUMINAMATH_CALUDE_number_comparison_l2933_293371


namespace NUMINAMATH_CALUDE_sqrt_66_greater_than_8_l2933_293321

theorem sqrt_66_greater_than_8 : Real.sqrt 66 > 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_66_greater_than_8_l2933_293321


namespace NUMINAMATH_CALUDE_speed_is_pi_over_three_l2933_293315

/-- Represents a rectangular track with looped ends -/
structure Track where
  width : ℝ
  straightLength : ℝ

/-- Calculates the speed of a person walking around the track -/
def calculateSpeed (track : Track) (timeDifference : ℝ) : ℝ :=
  sorry

/-- Theorem stating that given the specific track conditions, the calculated speed is π/3 -/
theorem speed_is_pi_over_three (track : Track) (h1 : track.width = 8)
    (h2 : track.straightLength = 100) (timeDifference : ℝ) (h3 : timeDifference = 48) :
    calculateSpeed track timeDifference = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_speed_is_pi_over_three_l2933_293315


namespace NUMINAMATH_CALUDE_first_ten_digits_of_expression_l2933_293374

theorem first_ten_digits_of_expression (ε : ℝ) (h : ε > 0) :
  ∃ n : ℤ, (5 + Real.sqrt 26) ^ 100 = n - ε ∧ 0 < ε ∧ ε < 1e-10 :=
sorry

end NUMINAMATH_CALUDE_first_ten_digits_of_expression_l2933_293374


namespace NUMINAMATH_CALUDE_min_tablets_extraction_l2933_293301

/-- Represents the number of tablets for each medicine type -/
structure MedicineCount where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Represents the minimum number of tablets required for each medicine type -/
structure RequiredCount where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Calculates the minimum number of tablets to be extracted -/
def minTablets (total : MedicineCount) (required : RequiredCount) : Nat :=
  sorry

theorem min_tablets_extraction (total : MedicineCount) (required : RequiredCount) :
  total.A = 10 →
  total.B = 14 →
  total.C = 18 →
  total.D = 20 →
  required.A = 3 →
  required.B = 4 →
  required.C = 3 →
  required.D = 2 →
  minTablets total required = 55 := by
  sorry

end NUMINAMATH_CALUDE_min_tablets_extraction_l2933_293301


namespace NUMINAMATH_CALUDE_min_ships_proof_l2933_293387

/-- The number of passengers to accommodate -/
def total_passengers : ℕ := 792

/-- The maximum capacity of each cruise ship -/
def ship_capacity : ℕ := 55

/-- The minimum number of cruise ships required -/
def min_ships : ℕ := (total_passengers + ship_capacity - 1) / ship_capacity

theorem min_ships_proof : min_ships = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_ships_proof_l2933_293387


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2933_293354

/-- Represents a geometric sequence with first term a and common ratio q -/
structure GeometricSequence (α : Type*) [Field α] where
  a : α
  q : α

/-- Sum of first n terms of a geometric sequence -/
def sumGeometric {α : Type*} [Field α] (seq : GeometricSequence α) (n : ℕ) : α :=
  seq.a * (1 - seq.q ^ n) / (1 - seq.q)

theorem geometric_sequence_property {α : Type*} [Field α] (seq : GeometricSequence α) :
  (sumGeometric seq 3 + sumGeometric seq 6 = 2 * sumGeometric seq 9) →
  (seq.a * seq.q + seq.a * seq.q^4 = 4) →
  seq.a * seq.q^7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2933_293354


namespace NUMINAMATH_CALUDE_determinant_cosine_matrix_zero_l2933_293368

open Matrix Real

theorem determinant_cosine_matrix_zero (a b : ℝ) : 
  det !![1, cos (a - b), cos a; 
         cos (a - b), 1, cos b; 
         cos a, cos b, 1] = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_cosine_matrix_zero_l2933_293368


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2933_293358

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x + 2) * (x - 3) < 0} = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2933_293358


namespace NUMINAMATH_CALUDE_crease_length_is_twenty_thirds_l2933_293336

/-- Represents a right triangle with sides 6, 8, and 10 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2
  side_a : a = 6
  side_b : b = 8
  side_c : c = 10

/-- Represents the crease formed when point A is folded onto the midpoint of side BC -/
def crease_length (t : RightTriangle) : ℝ := sorry

/-- Theorem stating that the length of the crease is 20/3 inches -/
theorem crease_length_is_twenty_thirds (t : RightTriangle) :
  crease_length t = 20/3 := by sorry

end NUMINAMATH_CALUDE_crease_length_is_twenty_thirds_l2933_293336


namespace NUMINAMATH_CALUDE_variety_show_probability_variety_show_probability_proof_l2933_293350

/-- The probability of selecting 2 dance performances out of 3 for the first 3 slots
    in a randomly arranged program of 8 performances (5 singing, 3 dance) -/
theorem variety_show_probability : ℚ :=
  let total_performances : ℕ := 8
  let singing_performances : ℕ := 5
  let dance_performances : ℕ := 3
  let first_slots : ℕ := 3
  let required_dance : ℕ := 2

  3 / 28

theorem variety_show_probability_proof :
  variety_show_probability = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_variety_show_probability_variety_show_probability_proof_l2933_293350


namespace NUMINAMATH_CALUDE_bobby_average_increase_l2933_293383

/-- Represents Bobby's deadlift capabilities and progress --/
structure DeadliftProgress where
  initial_weight : ℕ  -- Initial deadlift weight at age 13
  final_weight : ℕ    -- Final deadlift weight at age 18
  initial_age : ℕ     -- Age when initial weight was lifted
  final_age : ℕ       -- Age when final weight was lifted

/-- Calculates the average yearly increase in deadlift weight --/
def average_yearly_increase (progress : DeadliftProgress) : ℚ :=
  (progress.final_weight - progress.initial_weight : ℚ) / (progress.final_age - progress.initial_age)

/-- Bobby's actual deadlift progress --/
def bobby_progress : DeadliftProgress := {
  initial_weight := 300,
  final_weight := 850,
  initial_age := 13,
  final_age := 18
}

/-- Theorem stating that Bobby's average yearly increase in deadlift weight is 110 pounds --/
theorem bobby_average_increase : 
  average_yearly_increase bobby_progress = 110 := by
  sorry

end NUMINAMATH_CALUDE_bobby_average_increase_l2933_293383


namespace NUMINAMATH_CALUDE_probability_is_one_over_sixtythree_l2933_293342

/-- Represents the color of a bead -/
inductive BeadColor
  | Red
  | White
  | Blue

/-- Represents a line of beads -/
def BeadLine := List BeadColor

/-- The total number of beads -/
def totalBeads : Nat := 9

/-- The number of red beads -/
def redBeads : Nat := 4

/-- The number of white beads -/
def whiteBeads : Nat := 3

/-- The number of blue beads -/
def blueBeads : Nat := 2

/-- Checks if no two neighboring beads in a line have the same color -/
def noAdjacentSameColor (line : BeadLine) : Bool :=
  sorry

/-- Generates all possible bead lines -/
def allBeadLines : List BeadLine :=
  sorry

/-- Counts the number of valid bead lines where no two neighboring beads have the same color -/
def countValidLines : Nat :=
  sorry

/-- The probability of no two neighboring beads being the same color -/
def probability : Rat :=
  sorry

theorem probability_is_one_over_sixtythree : 
  probability = 1 / 63 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_over_sixtythree_l2933_293342


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2933_293365

theorem quadratic_solution_sum (a b : ℕ+) (x : ℝ) :
  x^2 + 16*x = 100 ∧ x > 0 ∧ x = Real.sqrt a - b → a + b = 172 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2933_293365


namespace NUMINAMATH_CALUDE_apple_cost_is_21_cents_l2933_293339

/-- The cost of an apple and an orange satisfy the given conditions -/
def apple_orange_cost (apple_cost orange_cost : ℚ) : Prop :=
  6 * apple_cost + 3 * orange_cost = 177/100 ∧
  2 * apple_cost + 5 * orange_cost = 127/100

/-- The cost of an apple is 0.21 dollars -/
theorem apple_cost_is_21_cents :
  ∃ (orange_cost : ℚ), apple_orange_cost (21/100) orange_cost := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_is_21_cents_l2933_293339


namespace NUMINAMATH_CALUDE_subset_P_Q_l2933_293330

theorem subset_P_Q : Set.Subset {x : ℝ | x > 1} {x : ℝ | x^2 - x > 0} := by sorry

end NUMINAMATH_CALUDE_subset_P_Q_l2933_293330


namespace NUMINAMATH_CALUDE_orange_stock_proof_l2933_293394

/-- Represents the original stock of oranges in kg -/
def original_stock : ℝ := 2700

/-- Represents the percentage of stock remaining after sale -/
def remaining_percentage : ℝ := 0.25

/-- Represents the amount of oranges remaining after sale in kg -/
def remaining_stock : ℝ := 675

theorem orange_stock_proof :
  remaining_percentage * original_stock = remaining_stock :=
sorry

end NUMINAMATH_CALUDE_orange_stock_proof_l2933_293394


namespace NUMINAMATH_CALUDE_jasons_shopping_l2933_293304

theorem jasons_shopping (total_spent jacket_cost : ℝ) 
  (h1 : total_spent = 14.28)
  (h2 : jacket_cost = 4.74) :
  total_spent - jacket_cost = 9.54 := by
sorry

end NUMINAMATH_CALUDE_jasons_shopping_l2933_293304


namespace NUMINAMATH_CALUDE_exists_line_with_specified_length_l2933_293364

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the intersection points
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the circles
def S₁ : Circle := sorry
def S₂ : Circle := sorry

-- Specify that the circles intersect at A and B
axiom intersect_at_A : A ∈ {p | (p.1 - S₁.center.1)^2 + (p.2 - S₁.center.2)^2 = S₁.radius^2} ∩
                           {p | (p.1 - S₂.center.1)^2 + (p.2 - S₂.center.2)^2 = S₂.radius^2}
axiom intersect_at_B : B ∈ {p | (p.1 - S₁.center.1)^2 + (p.2 - S₁.center.2)^2 = S₁.radius^2} ∩
                           {p | (p.1 - S₂.center.1)^2 + (p.2 - S₂.center.2)^2 = S₂.radius^2}

-- Define a line passing through point A
def line_through_A (m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 - A.2 = m * (p.1 - A.1)}

-- Define the segment of a line contained within both circles
def segment_in_circles (l : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  l ∩ {p | (p.1 - S₁.center.1)^2 + (p.2 - S₁.center.2)^2 ≤ S₁.radius^2} ∩
       {p | (p.1 - S₂.center.1)^2 + (p.2 - S₂.center.2)^2 ≤ S₂.radius^2}

-- Define the length of a segment
def segment_length (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem exists_line_with_specified_length (length : ℝ) :
  ∃ m : ℝ, segment_length (segment_in_circles (line_through_A m)) = length :=
sorry

end NUMINAMATH_CALUDE_exists_line_with_specified_length_l2933_293364


namespace NUMINAMATH_CALUDE_day_shift_percentage_l2933_293306

theorem day_shift_percentage
  (excel_percentage : ℝ)
  (excel_and_night_percentage : ℝ)
  (h1 : excel_percentage = 0.20)
  (h2 : excel_and_night_percentage = 0.06) :
  1 - (excel_and_night_percentage / excel_percentage) = 0.70 := by
  sorry

end NUMINAMATH_CALUDE_day_shift_percentage_l2933_293306


namespace NUMINAMATH_CALUDE_nth_equation_l2933_293378

theorem nth_equation (n : ℕ+) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l2933_293378


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2933_293309

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 < 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2933_293309


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptote_l2933_293328

/-- The distance from the focus of the parabola x = (1/4)y^2 to the asymptote of the hyperbola x^2 - (y^2/3) = 1 is √3/2 -/
theorem distance_focus_to_asymptote :
  let focus : ℝ × ℝ := (1, 0)
  let asymptote (x : ℝ) : ℝ := Real.sqrt 3 * x
  let distance_point_to_line (p : ℝ × ℝ) (f : ℝ → ℝ) : ℝ :=
    |f p.1 - p.2| / Real.sqrt (1 + (Real.sqrt 3)^2)
  distance_point_to_line focus asymptote = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_distance_focus_to_asymptote_l2933_293328


namespace NUMINAMATH_CALUDE_base_conversion_l2933_293373

/-- Given that 132 in base k is equal to 42 in base 10, prove that k = 5 -/
theorem base_conversion (k : ℕ) : k ^ 2 + 3 * k + 2 = 42 → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l2933_293373


namespace NUMINAMATH_CALUDE_creeping_jennies_count_l2933_293318

/-- The number of creeping jennies per planter -/
def creeping_jennies : ℕ := sorry

/-- The cost of a palm fern -/
def palm_fern_cost : ℚ := 15

/-- The cost of a creeping jenny -/
def creeping_jenny_cost : ℚ := 4

/-- The cost of a geranium -/
def geranium_cost : ℚ := 3.5

/-- The number of geraniums per planter -/
def geraniums_per_planter : ℕ := 4

/-- The number of planters -/
def num_planters : ℕ := 4

/-- The total cost for all planters -/
def total_cost : ℚ := 180

theorem creeping_jennies_count : 
  creeping_jennies = 4 ∧ 
  (num_planters : ℚ) * (palm_fern_cost + creeping_jenny_cost * (creeping_jennies : ℚ) + 
    geranium_cost * (geraniums_per_planter : ℚ)) = total_cost :=
sorry

end NUMINAMATH_CALUDE_creeping_jennies_count_l2933_293318


namespace NUMINAMATH_CALUDE_max_k_value_l2933_293317

theorem max_k_value (k : ℤ) : 
  (∃ x : ℕ+, k * x.val - 5 = 2021 * x.val + 2 * k) → k ≤ 6068 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l2933_293317


namespace NUMINAMATH_CALUDE_sum_of_complex_roots_of_unity_l2933_293344

theorem sum_of_complex_roots_of_unity (a b c : ℂ) :
  (Complex.abs a = 1) →
  (Complex.abs b = 1) →
  (Complex.abs c = 1) →
  (a^2 / (b*c) + b^2 / (a*c) + c^2 / (a*b) = -1) →
  (Complex.abs (a + b + c) = 1 ∨ Complex.abs (a + b + c) = 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_complex_roots_of_unity_l2933_293344


namespace NUMINAMATH_CALUDE_fraction_equality_l2933_293314

theorem fraction_equality (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (1/x - 1/y) / (1/x + 1/y) = 2023 → (x + y) / (x - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2933_293314


namespace NUMINAMATH_CALUDE_perpendicular_bisector_correct_parallel_line_correct_l2933_293388

-- Define points A and B
def A : ℝ × ℝ := (7, -4)
def B : ℝ × ℝ := (-5, 6)

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + y - 8 = 0
def line2 (x y : ℝ) : Prop := x - 2*y + 1 = 0
def line3 (x y : ℝ) : Prop := 4*x - 3*y - 7 = 0

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 6*x - 5*y - 1 = 0

-- Define the parallel line equation
def parallel_line (x y : ℝ) : Prop := 4*x - 3*y - 6 = 0

-- Theorem for the perpendicular bisector
theorem perpendicular_bisector_correct : 
  perp_bisector = λ x y => (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

-- Theorem for the parallel line
theorem parallel_line_correct : 
  ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ parallel_line x y ∧
  ∃ k : ℝ, ∀ x' y' : ℝ, parallel_line x' y' ↔ line3 x' y' ∧ (y' - y = k * (x' - x)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_correct_parallel_line_correct_l2933_293388


namespace NUMINAMATH_CALUDE_a_range_l2933_293395

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, a ≥ 2^x

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 4*x + a = 0

-- Theorem statement
theorem a_range (a : ℝ) (hp : p a) (hq : q a) : 2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l2933_293395


namespace NUMINAMATH_CALUDE_line_slope_is_two_l2933_293386

/-- The slope of a line given by the equation 3y - 6x = 9 is 2 -/
theorem line_slope_is_two : 
  ∀ (x y : ℝ), 3 * y - 6 * x = 9 → (∃ b : ℝ, y = 2 * x + b) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l2933_293386


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2933_293347

-- Define the quadratic function f(x) = ax^2 + 2x + c
def f (a c x : ℝ) : ℝ := a * x^2 + 2 * x + c

-- Define the quadratic function g(x) = -cx^2 + 2x - a
def g (a c x : ℝ) : ℝ := -c * x^2 + 2 * x - a

-- Theorem statement
theorem solution_set_equivalence (a c : ℝ) :
  (∀ x, -1/3 < x ∧ x < 1/2 → f a c x > 0) →
  (∀ x, g a c x > 0 ↔ -2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2933_293347


namespace NUMINAMATH_CALUDE_tim_has_156_golf_balls_l2933_293316

/-- The number of units in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of golf balls Tim has -/
def tims_dozens : ℕ := 13

/-- The total number of golf balls Tim has -/
def tims_golf_balls : ℕ := tims_dozens * dozen

theorem tim_has_156_golf_balls : tims_golf_balls = 156 := by
  sorry

end NUMINAMATH_CALUDE_tim_has_156_golf_balls_l2933_293316


namespace NUMINAMATH_CALUDE_tan_fifteen_simplification_l2933_293359

theorem tan_fifteen_simplification :
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_simplification_l2933_293359


namespace NUMINAMATH_CALUDE_function_properties_l2933_293334

noncomputable def f (x : ℝ) : ℝ := 2^(Real.sin x)

theorem function_properties :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < π ∧ 0 < x₂ ∧ x₂ < π ∧ f x₁ + f x₂ = 2) ∨
  (∀ x₁ x₂ : ℝ, -π/2 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/2 → f x₁ < f x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2933_293334


namespace NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l2933_293305

/-- Given a hyperbola with equation y²/9 - x²/b² = 1 and eccentricity 2,
    the distance from its focus to its asymptote is 3√3 -/
theorem hyperbola_focus_to_asymptote_distance
  (b : ℝ) -- Parameter b of the hyperbola
  (h1 : ∀ x y, y^2/9 - x^2/b^2 = 1) -- Equation of the hyperbola
  (h2 : 2 = (Real.sqrt (9 + b^2)) / 3) -- Eccentricity is 2
  : ∃ (focus : ℝ × ℝ) (asymptote : ℝ → ℝ),
    (∀ x, asymptote x = (Real.sqrt 3 / 3) * x ∨ asymptote x = -(Real.sqrt 3 / 3) * x) ∧
    Real.sqrt ((asymptote (focus.1) - focus.2)^2 / (1 + (Real.sqrt 3 / 3)^2)) = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l2933_293305


namespace NUMINAMATH_CALUDE_lg_properties_l2933_293393

-- Define the base 10 logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_properties :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 →
    (lg (x₁ * x₂) = lg x₁ + lg x₂) ∧
    (x₁ ≠ x₂ → (lg x₁ - lg x₂) / (x₁ - x₂) > 0) :=
by sorry

end NUMINAMATH_CALUDE_lg_properties_l2933_293393


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l2933_293376

/-- Given a polynomial function g(x) = px^4 + qx^3 + rx^2 + sx + t,
    if g(-3) = 9, then 16p - 8q + 4r - 2s + t = -9 -/
theorem polynomial_value_theorem (p q r s t : ℝ) :
  let g : ℝ → ℝ := λ x => p * x^4 + q * x^3 + r * x^2 + s * x + t
  g (-3) = 9 → 16 * p - 8 * q + 4 * r - 2 * s + t = -9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l2933_293376


namespace NUMINAMATH_CALUDE_inequality_proof_l2933_293390

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - a)^2 / ((b + c)^2 + a^2) +
  (c + a - b)^2 / ((c + a)^2 + b^2) +
  (a + b - c)^2 / ((a + b)^2 + c^2) ≥ 3/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2933_293390


namespace NUMINAMATH_CALUDE_sqrt5_parts_sqrt2_plus_1_parts_sqrt3_plus_2_parts_l2933_293379

-- Define the irrational numbers
axiom sqrt2 : ℝ
axiom sqrt3 : ℝ
axiom sqrt5 : ℝ

-- Define the properties of these irrational numbers
axiom sqrt2_irrational : Irrational sqrt2
axiom sqrt3_irrational : Irrational sqrt3
axiom sqrt5_irrational : Irrational sqrt5

axiom sqrt2_bounds : 1 < sqrt2 ∧ sqrt2 < 2
axiom sqrt3_bounds : 1 < sqrt3 ∧ sqrt3 < 2
axiom sqrt5_bounds : 2 < sqrt5 ∧ sqrt5 < 3

-- Define the integer and decimal part functions
def intPart (x : ℝ) : ℤ := sorry
def decPart (x : ℝ) : ℝ := sorry

-- Theorem statements
theorem sqrt5_parts : intPart sqrt5 = 2 ∧ decPart sqrt5 = sqrt5 - 2 := by sorry

theorem sqrt2_plus_1_parts : intPart (1 + sqrt2) = 2 ∧ decPart (1 + sqrt2) = sqrt2 - 1 := by sorry

theorem sqrt3_plus_2_parts :
  let x := intPart (2 + sqrt3)
  let y := decPart (2 + sqrt3)
  x - sqrt3 * y = sqrt3 := by sorry

end NUMINAMATH_CALUDE_sqrt5_parts_sqrt2_plus_1_parts_sqrt3_plus_2_parts_l2933_293379


namespace NUMINAMATH_CALUDE_parabola_and_range_l2933_293302

-- Define the parabola G
def G (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 4)

-- Define point A
def point_A : ℝ × ℝ := (-4, 0)

-- Define the condition for points B and C
def intersect_points (p k x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  G p x₁ y₁ ∧ G p x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the condition AC = 1/4 * AB when k = 1/2
def vector_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = 4*y₂

-- Define the y-intercept of the perpendicular bisector
def perpendicular_bisector_y_intercept (k : ℝ) : ℝ :=
  2*(k + 1)^2

-- Theorem statement
theorem parabola_and_range :
  ∀ p k x₁ y₁ x₂ y₂,
    G p (-4) 0 →
    intersect_points p k x₁ y₁ x₂ y₂ →
    k = 1/2 →
    vector_condition x₁ y₁ x₂ y₂ →
    (p = 2 ∧ 
     ∀ b, b > 2 ↔ ∃ k', perpendicular_bisector_y_intercept k' = b) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_range_l2933_293302


namespace NUMINAMATH_CALUDE_restock_is_mode_l2933_293372

def shoe_sizes : List ℝ := [22, 22.5, 23, 23.5, 24, 24.5, 25]
def quantities : List ℕ := [3, 5, 10, 15, 8, 3, 2]
def restock_size : ℝ := 23.5

def mode (sizes : List ℝ) (quants : List ℕ) : ℝ :=
  let paired := List.zip sizes quants
  let max_quant := paired.map (λ p => p.2) |>.maximum?
  match paired.find? (λ p => p.2 = max_quant) with
  | some (size, _) => size
  | none => 0  -- This case should not occur if the lists are non-empty

theorem restock_is_mode :
  mode shoe_sizes quantities = restock_size :=
sorry

end NUMINAMATH_CALUDE_restock_is_mode_l2933_293372


namespace NUMINAMATH_CALUDE_complex_number_solution_l2933_293345

theorem complex_number_solution (z : ℂ) : (Complex.I * z = 1) → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_solution_l2933_293345


namespace NUMINAMATH_CALUDE_problem_solutions_l2933_293351

theorem problem_solutions :
  (∀ x : ℝ, 2 * x^2 - 3 * x + 4 > 0) ∧
  (∃ x : ℕ, x^2 ≤ x) ∧
  (∃ x : ℕ, 29 % x = 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_solutions_l2933_293351


namespace NUMINAMATH_CALUDE_valid_triangle_constructions_l2933_293399

-- Define the basic structure
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the given points
variable (A A₀ D E : ℝ × ℝ)

-- Define the midpoint property
def is_midpoint (M : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the median property
def is_median (M : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  is_midpoint M B C

-- Define the angle bisector property
def is_angle_bisector (D : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop := sorry

-- Define the perpendicular bisector property
def is_perpendicular_bisector (E : ℝ × ℝ) (A C : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem valid_triangle_constructions 
  (h1 : is_midpoint A₀ (Triangle.B t) (Triangle.C t))
  (h2 : is_median A₀ (Triangle.A t) (Triangle.B t) (Triangle.C t))
  (h3 : is_angle_bisector D (Triangle.A t) (Triangle.B t) (Triangle.C t))
  (h4 : is_perpendicular_bisector E (Triangle.A t) (Triangle.C t)) :
  ∃ (C₁ C₂ : ℝ × ℝ), C₁ ≠ C₂ ∧ 
    (∃ (t₁ t₂ : Triangle), 
      (t₁.A = A ∧ t₁.C = C₁) ∧ 
      (t₂.A = A ∧ t₂.C = C₂) ∧
      (is_midpoint A₀ t₁.B t₁.C) ∧
      (is_midpoint A₀ t₂.B t₂.C) ∧
      (is_median A₀ t₁.A t₁.B t₁.C) ∧
      (is_median A₀ t₂.A t₂.B t₂.C) ∧
      (is_angle_bisector D t₁.A t₁.B t₁.C) ∧
      (is_angle_bisector D t₂.A t₂.B t₂.C) ∧
      (is_perpendicular_bisector E t₁.A t₁.C) ∧
      (is_perpendicular_bisector E t₂.A t₂.C)) :=
sorry


end NUMINAMATH_CALUDE_valid_triangle_constructions_l2933_293399


namespace NUMINAMATH_CALUDE_opposite_face_of_one_is_three_l2933_293385

/-- Represents a face of a cube --/
inductive CubeFace
| One
| Two
| Three
| Four
| Five
| Six

/-- Represents a net of a cube --/
structure CubeNet :=
(faces : Finset CubeFace)
(valid : faces.card = 6)

/-- Represents a folded cube --/
structure Cube :=
(net : CubeNet)
(opposite : CubeFace → CubeFace)
(opposite_involutive : ∀ f, opposite (opposite f) = f)

/-- The theorem to be proved --/
theorem opposite_face_of_one_is_three (c : Cube) : c.opposite CubeFace.One = CubeFace.Three := by
  sorry

end NUMINAMATH_CALUDE_opposite_face_of_one_is_three_l2933_293385


namespace NUMINAMATH_CALUDE_max_cross_section_area_l2933_293329

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a square prism -/
structure SquarePrism where
  sideLength : ℝ
  baseVertices : List Point3D

/-- Calculates the area of the cross-section when a plane intersects a square prism -/
def crossSectionArea (prism : SquarePrism) (plane : Plane) : ℝ := sorry

/-- The main theorem stating the maximum area of the cross-section -/
theorem max_cross_section_area :
  let prism : SquarePrism := {
    sideLength := 12,
    baseVertices := [
      {x := 6, y := 6, z := 0},
      {x := -6, y := 6, z := 0},
      {x := -6, y := -6, z := 0},
      {x := 6, y := -6, z := 0}
    ]
  }
  let plane : Plane := {a := 5, b := -8, c := 3, d := 30}
  crossSectionArea prism plane = 252 := by sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l2933_293329


namespace NUMINAMATH_CALUDE_distinct_values_count_l2933_293312

def odd_integers_less_than_15 : Finset ℕ :=
  {1, 3, 5, 7, 9, 11, 13}

def expression (p q : ℕ) : ℤ :=
  p * q - (p + q)

theorem distinct_values_count :
  Finset.card (Finset.image₂ expression odd_integers_less_than_15 odd_integers_less_than_15) = 28 :=
by sorry

end NUMINAMATH_CALUDE_distinct_values_count_l2933_293312


namespace NUMINAMATH_CALUDE_equation_solution_l2933_293340

theorem equation_solution : 
  ∃! x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 :=
by
  use -13/4
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2933_293340


namespace NUMINAMATH_CALUDE_parabola_properties_l2933_293343

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
theorem parabola_properties (a b c m : ℝ) 
  (h_vertex : -b / (2 * a) = -1/2)
  (h_m_pos : m > 0)
  (h_vertex_y : parabola a b c (-1/2) = m)
  (h_intercept : ∃ x, 0 < x ∧ x < 1 ∧ parabola a b c x = 0) :
  (b < 0) ∧ 
  (∀ y₁ y₂, parabola a b c (-2) = y₁ → parabola a b c 2 = y₂ → y₁ > y₂) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2933_293343


namespace NUMINAMATH_CALUDE_marble_arrangement_l2933_293308

/-- Represents the color of a marble -/
inductive Color
| Blue
| Yellow

/-- Calculates the number of ways to arrange marbles -/
def arrange_marbles (blue : ℕ) (yellow : ℕ) : ℕ :=
  Nat.choose (yellow + blue - 1) (blue - 1)

/-- The main theorem -/
theorem marble_arrangement :
  let blue := 6
  let max_yellow := 17
  let arrangements := arrange_marbles blue max_yellow
  arrangements = 12376 ∧ arrangements % 1000 = 376 := by
  sorry


end NUMINAMATH_CALUDE_marble_arrangement_l2933_293308


namespace NUMINAMATH_CALUDE_eggs_needed_is_84_l2933_293348

/-- Represents the number of eggs in an omelette -/
inductive OmeletteType
  | threeEgg
  | fourEgg

/-- Represents an hour of operation at the cafe -/
structure Hour where
  customers : Nat
  omeletteType : OmeletteType

/-- Represents a day of operation at Theo's cafe -/
structure CafeDay where
  hours : List Hour

/-- Calculates the total number of eggs needed for a given hour -/
def eggsNeededForHour (hour : Hour) : Nat :=
  match hour.omeletteType with
  | OmeletteType.threeEgg => 3 * hour.customers
  | OmeletteType.fourEgg => 4 * hour.customers

/-- Calculates the total number of eggs needed for the entire day -/
def totalEggsNeeded (day : CafeDay) : Nat :=
  day.hours.foldl (fun acc hour => acc + eggsNeededForHour hour) 0

/-- Theorem stating that the total number of eggs needed is 84 -/
theorem eggs_needed_is_84 (day : CafeDay) 
    (h1 : day.hours = [
      { customers := 5, omeletteType := OmeletteType.threeEgg },
      { customers := 7, omeletteType := OmeletteType.fourEgg },
      { customers := 3, omeletteType := OmeletteType.threeEgg },
      { customers := 8, omeletteType := OmeletteType.fourEgg }
    ]) : 
    totalEggsNeeded day = 84 := by
  sorry


end NUMINAMATH_CALUDE_eggs_needed_is_84_l2933_293348


namespace NUMINAMATH_CALUDE_f_inequality_l2933_293346

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem f_inequality (a b c : ℝ) (h1 : a > 0) (h2 : ∀ x, f a b c (1 - x) = f a b c (1 + x)) :
  ∀ x, f a b c (2^x) > f a b c (3^x) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_l2933_293346


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2933_293363

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 + 4*i) / (1 + i) = (7:ℂ)/2 + (1:ℂ)/2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2933_293363


namespace NUMINAMATH_CALUDE_initial_native_trees_l2933_293320

theorem initial_native_trees (N : ℕ) : 
  (3 * N - N) + (3 * N - N) / 3 = 80 → N = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_native_trees_l2933_293320


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_500_l2933_293389

def hulk_jump (n : ℕ) : ℝ :=
  2 * (3 : ℝ) ^ (n - 1)

theorem hulk_jump_exceeds_500 :
  (∀ k < 7, hulk_jump k ≤ 500) ∧ hulk_jump 7 > 500 := by
  sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_500_l2933_293389


namespace NUMINAMATH_CALUDE_gcd_2703_1113_l2933_293327

theorem gcd_2703_1113 : Nat.gcd 2703 1113 = 159 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2703_1113_l2933_293327


namespace NUMINAMATH_CALUDE_width_length_ratio_l2933_293300

/-- A rectangle with given length and perimeter -/
structure Rectangle where
  length : ℝ
  perimeter : ℝ
  width : ℝ
  length_pos : length > 0
  perimeter_pos : perimeter > 0
  width_pos : width > 0
  perimeter_eq : perimeter = 2 * (length + width)

/-- The ratio of width to length for a rectangle with length 10 and perimeter 30 is 1:2 -/
theorem width_length_ratio (rect : Rectangle) 
    (h1 : rect.length = 10) 
    (h2 : rect.perimeter = 30) : 
    rect.width / rect.length = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_width_length_ratio_l2933_293300


namespace NUMINAMATH_CALUDE_hillarys_weekend_reading_l2933_293377

/-- The total reading time assigned for a weekend, given the reading times for Friday, Saturday, and Sunday. -/
def weekend_reading_time (friday_time saturday_time sunday_time : ℕ) : ℕ :=
  friday_time + saturday_time + sunday_time

/-- Theorem stating that the total reading time for Hillary's weekend assignment is 60 minutes. -/
theorem hillarys_weekend_reading : weekend_reading_time 16 28 16 = 60 := by
  sorry

end NUMINAMATH_CALUDE_hillarys_weekend_reading_l2933_293377


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2933_293341

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₂ + a₁₀ = 16, then a₄ + a₈ = 16 -/
theorem arithmetic_sequence_sum_property 
  (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 2 + a 10 = 16) : 
  a 4 + a 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2933_293341


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l2933_293323

-- Define the parabolas
def M1 (a c x : ℝ) : ℝ := a * x^2 + c
def M2 (a c x : ℝ) : ℝ := a * (x - 2)^2 + c - 5

-- Define the theorem
theorem parabola_triangle_area (a c : ℝ) :
  -- M2 passes through the vertex of M1
  (M2 a c 0 = M1 a c 0) →
  -- Point C on M2 has coordinates (2, c-5)
  (M2 a c 2 = c - 5) →
  -- The area of triangle ABC is 10
  ∃ (x_B y_B : ℝ), 
    x_B = 2 ∧ 
    y_B = M1 a c x_B ∧ 
    (1/2 * |x_B - 0| * |y_B - (c - 5)| = 10) :=
by sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l2933_293323


namespace NUMINAMATH_CALUDE_school_departments_l2933_293392

/-- Given a school with departments where each department has 20 teachers and there are 140 teachers in total, prove that the number of departments is 7. -/
theorem school_departments (total_teachers : ℕ) (teachers_per_dept : ℕ) (h1 : total_teachers = 140) (h2 : teachers_per_dept = 20) :
  total_teachers / teachers_per_dept = 7 := by
  sorry

end NUMINAMATH_CALUDE_school_departments_l2933_293392


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l2933_293361

theorem quadratic_two_roots (a b c : ℝ) 
  (h1 : 2016 + a^2 + a*c < a*b) 
  (h2 : a ≠ 0) : 
  ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l2933_293361


namespace NUMINAMATH_CALUDE_lomonosov_card_puzzle_l2933_293355

theorem lomonosov_card_puzzle :
  ∃ (L O M N C B : ℕ),
    L ≠ O ∧ L ≠ M ∧ L ≠ N ∧ L ≠ C ∧ L ≠ B ∧
    O ≠ M ∧ O ≠ N ∧ O ≠ C ∧ O ≠ B ∧
    M ≠ N ∧ M ≠ C ∧ M ≠ B ∧
    N ≠ C ∧ N ≠ B ∧
    C ≠ B ∧
    L < 10 ∧ O < 10 ∧ M < 10 ∧ N < 10 ∧ C < 10 ∧ B < 10 ∧
    O < M ∧ O < C ∧
    L + O / M + O + N + O / C = 10 * O + B :=
by sorry

end NUMINAMATH_CALUDE_lomonosov_card_puzzle_l2933_293355


namespace NUMINAMATH_CALUDE_park_area_l2933_293319

/-- Proves that a rectangular park with sides in ratio 3:2 and fencing cost of $225 at 90 ps per meter has an area of 3750 square meters -/
theorem park_area (length width perimeter cost_per_meter total_cost : ℝ) : 
  length / width = 3 / 2 →
  perimeter = 2 * (length + width) →
  cost_per_meter = 0.9 →
  total_cost = 225 →
  total_cost = perimeter * cost_per_meter →
  length * width = 3750 :=
by sorry

end NUMINAMATH_CALUDE_park_area_l2933_293319


namespace NUMINAMATH_CALUDE_circular_center_ratio_l2933_293396

/-- Represents a square flag with a symmetric cross design -/
structure SymmetricCrossFlag where
  side : ℝ
  cross_area_ratio : ℝ
  (cross_area_valid : cross_area_ratio = 1/4)

/-- The area of the circular center of the cross -/
noncomputable def circular_center_area (flag : SymmetricCrossFlag) : ℝ :=
  (flag.cross_area_ratio * flag.side^2) / 4

theorem circular_center_ratio (flag : SymmetricCrossFlag) :
  circular_center_area flag / flag.side^2 = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_circular_center_ratio_l2933_293396
