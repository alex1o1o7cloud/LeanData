import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2486_248636

/-- Given an arithmetic sequence with common difference 2 where a₁, a₃, a₄ form a geometric sequence, a₆ = 2 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →         -- a₁, a₃, a₄ form a geometric sequence
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2486_248636


namespace NUMINAMATH_CALUDE_prime_pair_fraction_integer_l2486_248604

theorem prime_pair_fraction_integer :
  ∀ p q : ℕ,
    Prime p → Prime q → p > q →
    (∃ n : ℤ, (((p + q : ℕ)^(p + q) * (p - q : ℕ)^(p - q) - 1) : ℤ) = 
              n * (((p + q : ℕ)^(p - q) * (p - q : ℕ)^(p + q) - 1) : ℤ)) →
    p = 3 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_pair_fraction_integer_l2486_248604


namespace NUMINAMATH_CALUDE_current_speed_l2486_248637

/-- Given a boat's upstream and downstream speeds, calculate the current's speed -/
theorem current_speed (upstream_time : ℝ) (downstream_time : ℝ) :
  upstream_time = 30 →
  downstream_time = 12 →
  let upstream_speed := 60 / upstream_time
  let downstream_speed := 60 / downstream_time
  (downstream_speed - upstream_speed) / 2 = 1.5 := by
  sorry

#check current_speed

end NUMINAMATH_CALUDE_current_speed_l2486_248637


namespace NUMINAMATH_CALUDE_lydia_apple_tree_age_l2486_248623

theorem lydia_apple_tree_age (tree_fruit_time : ℕ) (planting_age : ℕ) : 
  tree_fruit_time = 10 → planting_age = 6 → planting_age + tree_fruit_time = 16 := by
  sorry

end NUMINAMATH_CALUDE_lydia_apple_tree_age_l2486_248623


namespace NUMINAMATH_CALUDE_special_pair_example_special_pair_with_three_special_pair_negation_l2486_248675

/-- Definition of a special rational number pair -/
def is_special_pair (a b : ℚ) : Prop := a + b = a * b - 1

/-- Theorem 1: (5, 3/2) is a special rational number pair -/
theorem special_pair_example : is_special_pair 5 (3/2) := by sorry

/-- Theorem 2: If (a, 3) is a special rational number pair, then a = 2 -/
theorem special_pair_with_three (a : ℚ) : is_special_pair a 3 → a = 2 := by sorry

/-- Theorem 3: If (m, n) is a special rational number pair, then (-n, -m) is not a special rational number pair -/
theorem special_pair_negation (m n : ℚ) : is_special_pair m n → ¬ is_special_pair (-n) (-m) := by sorry

end NUMINAMATH_CALUDE_special_pair_example_special_pair_with_three_special_pair_negation_l2486_248675


namespace NUMINAMATH_CALUDE_exists_m_divides_sum_powers_l2486_248601

theorem exists_m_divides_sum_powers (n : ℕ+) :
  ∃ m : ℕ+, (7^n.val : ℤ) ∣ (3^m.val + 5^m.val - 1) := by sorry

end NUMINAMATH_CALUDE_exists_m_divides_sum_powers_l2486_248601


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2486_248618

/-- The surface area of a hemisphere given its base area -/
theorem hemisphere_surface_area (base_area : ℝ) (h : base_area = 3) :
  let r : ℝ := Real.sqrt (base_area / Real.pi)
  2 * Real.pi * r^2 + base_area = 9 := by
  sorry


end NUMINAMATH_CALUDE_hemisphere_surface_area_l2486_248618


namespace NUMINAMATH_CALUDE_jerry_added_eleven_action_figures_l2486_248674

/-- The number of action figures Jerry added to his shelf -/
def action_figures_added (initial : ℕ) (removed : ℕ) (final : ℕ) : ℤ :=
  final - initial + removed

/-- Proof that Jerry added 11 action figures to his shelf -/
theorem jerry_added_eleven_action_figures :
  action_figures_added 7 10 8 = 11 := by
  sorry

end NUMINAMATH_CALUDE_jerry_added_eleven_action_figures_l2486_248674


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l2486_248666

/-- Theorem: Given two cyclists on a 45-mile course, starting from opposite ends at the same time,
    where one cyclist rides at 14 mph and they meet after 1.5 hours, the speed of the second cyclist is 16 mph. -/
theorem cyclist_speed_problem (course_length : ℝ) (first_speed : ℝ) (meeting_time : ℝ) :
  course_length = 45 ∧ first_speed = 14 ∧ meeting_time = 1.5 →
  ∃ second_speed : ℝ, second_speed = 16 ∧ course_length = (first_speed + second_speed) * meeting_time :=
by
  sorry


end NUMINAMATH_CALUDE_cyclist_speed_problem_l2486_248666


namespace NUMINAMATH_CALUDE_james_tshirts_l2486_248693

/-- Calculates the number of t-shirts bought given the discount rate, original price, and total amount paid -/
def tshirts_bought (discount_rate : ℚ) (original_price : ℚ) (total_paid : ℚ) : ℚ :=
  total_paid / (original_price * (1 - discount_rate))

/-- Proves that James bought 6 t-shirts -/
theorem james_tshirts :
  let discount_rate : ℚ := 1/2
  let original_price : ℚ := 20
  let total_paid : ℚ := 60
  tshirts_bought discount_rate original_price total_paid = 6 := by
sorry

end NUMINAMATH_CALUDE_james_tshirts_l2486_248693


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l2486_248697

theorem min_value_parallel_vectors (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![4 - n, m]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  (∀ x y, x > 0 → y > 0 → x / y + y / x ≥ 2) →
  (n / m + 8 / n ≥ 6) ∧ (∃ m₀ n₀, m₀ > 0 ∧ n₀ > 0 ∧ n₀ / m₀ + 8 / n₀ = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l2486_248697


namespace NUMINAMATH_CALUDE_fraction_property_l2486_248622

theorem fraction_property (n : ℕ+) : 
  (∃ (a b c d e f : ℕ), 
    (1 : ℚ) / (2*n + 1) = (a*100000 + b*10000 + c*1000 + d*100 + e*10 + f) / 999999 ∧ 
    a + b + c + d + e + f = 999) ↔ 
  (2*n + 1 = 7 ∨ 2*n + 1 = 13) :=
sorry

end NUMINAMATH_CALUDE_fraction_property_l2486_248622


namespace NUMINAMATH_CALUDE_noemi_blackjack_loss_l2486_248639

/-- Calculates the amount lost on blackjack given initial amount, amount lost on roulette, and final amount -/
def blackjack_loss (initial : ℕ) (roulette_loss : ℕ) (final : ℕ) : ℕ :=
  initial - roulette_loss - final

/-- Proves that Noemi lost $500 on blackjack -/
theorem noemi_blackjack_loss :
  let initial := 1700
  let roulette_loss := 400
  let final := 800
  blackjack_loss initial roulette_loss final = 500 := by
  sorry

end NUMINAMATH_CALUDE_noemi_blackjack_loss_l2486_248639


namespace NUMINAMATH_CALUDE_largest_value_when_x_is_quarter_l2486_248694

theorem largest_value_when_x_is_quarter (x : ℝ) (h : x = 1/4) : 
  (1/x > x) ∧ (1/x > x^2) ∧ (1/x > (1/2)*x) ∧ (1/x > Real.sqrt x) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_when_x_is_quarter_l2486_248694


namespace NUMINAMATH_CALUDE_largest_unrepresentable_number_l2486_248643

theorem largest_unrepresentable_number (a b : ℤ) (ha : a > 1) (hb : b > 1) :
  ∃ n : ℤ, n = 47 ∧ (∀ m : ℤ, m > n → ∃ x y : ℤ, m = 7*a + 5*b + 7*x + 5*y) ∧
  (¬∃ x y : ℤ, n = 7*a + 5*b + 7*x + 5*y) := by
sorry

end NUMINAMATH_CALUDE_largest_unrepresentable_number_l2486_248643


namespace NUMINAMATH_CALUDE_age_relation_l2486_248690

/-- Proves that A was twice as old as B 10 years ago given the conditions -/
theorem age_relation (b_age : ℕ) (a_age : ℕ) (x : ℕ) : 
  b_age = 42 →
  a_age = b_age + 12 →
  a_age + 10 = 2 * (b_age - x) →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_relation_l2486_248690


namespace NUMINAMATH_CALUDE_four_letter_words_with_a_l2486_248633

/-- The number of letters in the alphabet we're using -/
def alphabet_size : ℕ := 5

/-- The length of the words we're forming -/
def word_length : ℕ := 4

/-- The number of letters in the alphabet excluding 'A' -/
def alphabet_size_without_a : ℕ := 4

/-- The total number of possible 4-letter words using all 5 letters -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of 4-letter words not containing 'A' -/
def words_without_a : ℕ := alphabet_size_without_a ^ word_length

/-- The number of 4-letter words containing at least one 'A' -/
def words_with_a : ℕ := total_words - words_without_a

theorem four_letter_words_with_a : words_with_a = 369 := by
  sorry

end NUMINAMATH_CALUDE_four_letter_words_with_a_l2486_248633


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2486_248647

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), p2 = p1 + t • (p3 - p1) ∨ p1 = p2 + t • (p3 - p2)

/-- 
If the points (2,x,y), (x,3,y), and (x,y,4) are collinear, then x + y = 6.
-/
theorem collinear_points_sum (x y : ℝ) :
  collinear (2, x, y) (x, 3, y) (x, y, 4) → x + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2486_248647


namespace NUMINAMATH_CALUDE_fraction_division_and_addition_l2486_248648

theorem fraction_division_and_addition : (3 / 7 : ℚ) / 4 + 1 / 28 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_and_addition_l2486_248648


namespace NUMINAMATH_CALUDE_slope_of_line_l2486_248673

theorem slope_of_line (x y : ℝ) :
  (4 * y = -5 * x + 8) → (y = (-5/4) * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l2486_248673


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2486_248651

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ∧
    C = 2 / Real.sqrt 3 ∧
    ∀ (D : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 1 ≥ D * (x + y + z)) → D ≤ C :=
by
  sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2486_248651


namespace NUMINAMATH_CALUDE_percentage_four_leaf_clovers_l2486_248677

/-- Proves that 20% of clovers have four leaves given the conditions -/
theorem percentage_four_leaf_clovers 
  (total_clovers : ℕ) 
  (purple_four_leaf : ℕ) 
  (h1 : total_clovers = 500)
  (h2 : purple_four_leaf = 25)
  (h3 : (4 : ℚ) * purple_four_leaf = total_clovers * (percentage_four_leaf / 100)) :
  percentage_four_leaf = 20 := by
  sorry

#check percentage_four_leaf_clovers

end NUMINAMATH_CALUDE_percentage_four_leaf_clovers_l2486_248677


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l2486_248628

theorem largest_n_for_equation : ∃ (x y z : ℕ+), 
  9^2 = 2*x^2 + 2*y^2 + 2*z^2 + 4*x*y + 4*y*z + 4*z*x + 6*x + 6*y + 6*z - 14 ∧ 
  ∀ (n : ℕ+), n > 9 → ¬∃ (a b c : ℕ+), 
    n^2 = 2*a^2 + 2*b^2 + 2*c^2 + 4*a*b + 4*b*c + 4*c*a + 6*a + 6*b + 6*c - 14 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l2486_248628


namespace NUMINAMATH_CALUDE_system_equations_and_inequality_l2486_248657

theorem system_equations_and_inequality (a x y : ℝ) : 
  x - y = 1 + 3 * a →
  x + y = -7 - a →
  x ≤ 0 →
  y < 0 →
  (-2 < a ∧ a ≤ 3) →
  (∀ x, 2 * a * x + x > 2 * a + 1 ↔ x < 1) →
  a = -1 := by sorry

end NUMINAMATH_CALUDE_system_equations_and_inequality_l2486_248657


namespace NUMINAMATH_CALUDE_sum_of_integers_l2486_248642

theorem sum_of_integers (x y : ℕ+) (h1 : x.val^2 + y.val^2 = 245) (h2 : x.val * y.val = 120) :
  (x.val : ℝ) + y.val = Real.sqrt 485 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2486_248642


namespace NUMINAMATH_CALUDE_f_not_monotonic_iff_t_in_range_l2486_248610

open Set
open Real

noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 4*x - 3 * Real.log x

def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a ≤ x ∧ x < y ∧ y ≤ b ∧ (f x < f y ∧ ∃ z, x < z ∧ z < y ∧ f z < f x) ∨
                               (f x > f y ∧ ∃ z, x < z ∧ z < y ∧ f z > f x)

theorem f_not_monotonic_iff_t_in_range (t : ℝ) :
  not_monotonic f t (t + 1) ↔ t ∈ Ioo 0 1 ∪ Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_f_not_monotonic_iff_t_in_range_l2486_248610


namespace NUMINAMATH_CALUDE_total_laundry_time_l2486_248607

/-- Represents the time in minutes for washing and drying a load of laundry -/
structure LaundryTime where
  washing : ℕ
  drying : ℕ

/-- Calculates the total time for a single load of laundry -/
def totalTimeForLoad (load : LaundryTime) : ℕ :=
  load.washing + load.drying

/-- The time for the whites load -/
def whites : LaundryTime := ⟨72, 50⟩

/-- The time for the darks load -/
def darks : LaundryTime := ⟨58, 65⟩

/-- The time for the colors load -/
def colors : LaundryTime := ⟨45, 54⟩

/-- Theorem stating that the total time for all three loads is 344 minutes -/
theorem total_laundry_time :
  totalTimeForLoad whites + totalTimeForLoad darks + totalTimeForLoad colors = 344 := by
  sorry

end NUMINAMATH_CALUDE_total_laundry_time_l2486_248607


namespace NUMINAMATH_CALUDE_ducks_in_lake_l2486_248668

theorem ducks_in_lake (initial_ducks joining_ducks : ℕ) 
  (h1 : initial_ducks = 13)
  (h2 : joining_ducks = 20) : 
  initial_ducks + joining_ducks = 33 := by
sorry

end NUMINAMATH_CALUDE_ducks_in_lake_l2486_248668


namespace NUMINAMATH_CALUDE_tent_capacity_l2486_248603

/-- The number of seating sections in the circus tent -/
def num_sections : ℕ := 4

/-- The number of people each section can accommodate -/
def people_per_section : ℕ := 246

/-- The total number of people the tent can accommodate -/
def total_capacity : ℕ := num_sections * people_per_section

theorem tent_capacity : total_capacity = 984 := by
  sorry

end NUMINAMATH_CALUDE_tent_capacity_l2486_248603


namespace NUMINAMATH_CALUDE_cricket_bat_weight_proof_l2486_248658

/-- The weight of one cricket bat in pounds -/
def cricket_bat_weight : ℝ := 18

/-- The weight of one basketball in pounds -/
def basketball_weight : ℝ := 36

/-- The number of cricket bats -/
def num_cricket_bats : ℕ := 8

/-- The number of basketballs -/
def num_basketballs : ℕ := 4

theorem cricket_bat_weight_proof :
  cricket_bat_weight * num_cricket_bats = basketball_weight * num_basketballs :=
by sorry

end NUMINAMATH_CALUDE_cricket_bat_weight_proof_l2486_248658


namespace NUMINAMATH_CALUDE_julieta_total_spend_l2486_248649

/-- Calculates the total amount spent by Julieta at the store -/
def total_amount_spent (
  backpack_original_price : ℕ)
  (ringbinder_original_price : ℕ)
  (backpack_price_increase : ℕ)
  (ringbinder_price_reduction : ℕ)
  (num_ringbinders : ℕ) : ℕ :=
  (backpack_original_price + backpack_price_increase) +
  num_ringbinders * (ringbinder_original_price - ringbinder_price_reduction)

/-- Theorem stating that Julieta's total spend is $109 -/
theorem julieta_total_spend :
  total_amount_spent 50 20 5 2 3 = 109 := by
  sorry

end NUMINAMATH_CALUDE_julieta_total_spend_l2486_248649


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2486_248672

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![1, 2, 0; 4, 5, -3; 7, 8, 6]
  Matrix.det A = -36 := by sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2486_248672


namespace NUMINAMATH_CALUDE_solve_equation_l2486_248638

theorem solve_equation : ∃ x : ℚ, (5*x + 8*x = 350 - 9*(x+8)) ∧ (x = 12 + 7/11) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2486_248638


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2486_248617

/-- A positive geometric sequence with specific sum conditions has a common ratio of 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- The sequence is positive
  (∃ q > 0, ∀ n, a (n + 1) = a n * q) →  -- The sequence is geometric with positive ratio
  a 3 + a 5 = 5 →  -- First condition
  a 5 + a 7 = 20 →  -- Second condition
  (∃ q > 0, ∀ n, a (n + 1) = a n * q ∧ q = 2) :=  -- The common ratio is 2
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2486_248617


namespace NUMINAMATH_CALUDE_elevator_problem_l2486_248634

def masses : List ℕ := [150, 60, 70, 71, 72, 100, 101, 102, 103]
def elevator_capacity : ℕ := 200

def is_valid_trip (trip : List ℕ) : Prop :=
  trip.sum ≤ elevator_capacity

def minimum_trips (m : List ℕ) (cap : ℕ) : ℕ :=
  sorry

theorem elevator_problem :
  minimum_trips masses elevator_capacity = 5 := by
  sorry

end NUMINAMATH_CALUDE_elevator_problem_l2486_248634


namespace NUMINAMATH_CALUDE_triangle_inequality_with_120_degree_angle_l2486_248685

/-- Given a triangle with sides a, b, and c, where an angle of 120 degrees lies opposite to side c,
    prove that a, c, and a + b satisfy the triangle inequality theorem. -/
theorem triangle_inequality_with_120_degree_angle 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (triangle_exists : a + b > c ∧ b + c > a ∧ c + a > b) 
  (angle_120 : a^2 = b^2 + c^2 - b*c) : 
  a + c > a + b ∧ a + (a + b) > c ∧ c + (a + b) > a :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_120_degree_angle_l2486_248685


namespace NUMINAMATH_CALUDE_green_face_probability_l2486_248625

/-- Probability of rolling a green face on an octahedron with 5 green faces out of 8 total faces -/
theorem green_face_probability (total_faces : ℕ) (green_faces : ℕ) 
  (h1 : total_faces = 8) 
  (h2 : green_faces = 5) : 
  (green_faces : ℚ) / total_faces = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_green_face_probability_l2486_248625


namespace NUMINAMATH_CALUDE_problem_statement_l2486_248654

theorem problem_statement (a b c : ℝ) 
  (h1 : a^2 + b^2 - 4*a ≤ 1)
  (h2 : b^2 + c^2 - 8*b ≤ -3)
  (h3 : c^2 + a^2 - 12*c ≤ -26) :
  (a + b)^c = 27 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2486_248654


namespace NUMINAMATH_CALUDE_largest_replacement_l2486_248653

def original_number : ℚ := -0.3168

def replace_digit (n : ℚ) (old_digit new_digit : ℕ) : ℚ := sorry

theorem largest_replacement :
  ∀ d : ℕ, d ≠ 0 → d ≠ 3 → d ≠ 1 → d ≠ 6 → d ≠ 8 →
    replace_digit original_number 6 4 ≥ replace_digit original_number d 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_replacement_l2486_248653


namespace NUMINAMATH_CALUDE_tshirts_per_package_l2486_248659

theorem tshirts_per_package (total_tshirts : ℕ) (num_packages : ℕ) 
  (h1 : total_tshirts = 70) 
  (h2 : num_packages = 14) : 
  total_tshirts / num_packages = 5 := by
  sorry

end NUMINAMATH_CALUDE_tshirts_per_package_l2486_248659


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2486_248664

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h3 : seq.S 3 = 9) (h6 : seq.S 6 = 36) : seq.S 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2486_248664


namespace NUMINAMATH_CALUDE_mac_preference_l2486_248682

theorem mac_preference (total : ℕ) (no_pref : ℕ) (windows_pref : ℕ) 
  (h_total : total = 210)
  (h_no_pref : no_pref = 90)
  (h_windows_pref : windows_pref = 40)
  : ∃ (mac_pref : ℕ), 
    mac_pref = 60 ∧ 
    (total - no_pref = mac_pref + windows_pref + mac_pref / 3) :=
by sorry

end NUMINAMATH_CALUDE_mac_preference_l2486_248682


namespace NUMINAMATH_CALUDE_mean_proportional_segment_l2486_248661

theorem mean_proportional_segment (a c : ℝ) (x : ℝ) 
  (ha : a = 9) (hc : c = 4) (hx : x^2 = a * c) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_segment_l2486_248661


namespace NUMINAMATH_CALUDE_odd_numbers_perfect_square_l2486_248695

/-- Sum of first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- The n-th odd number -/
def nthOddNumber (n : ℕ) : ℕ := 2*n - 1

theorem odd_numbers_perfect_square (K : ℕ) :
  K % 2 = 1 →  -- K is odd
  (∃ (N : ℕ), N < 50 ∧ sumOddNumbers N = N^2 ∧ nthOddNumber N = K) →
  1 ≤ K ∧ K ≤ 97 :=
by sorry

end NUMINAMATH_CALUDE_odd_numbers_perfect_square_l2486_248695


namespace NUMINAMATH_CALUDE_pi_digits_difference_l2486_248665

theorem pi_digits_difference (mina_digits : ℕ) (mina_carlos_ratio : ℕ) (sam_digits : ℕ)
  (h1 : mina_digits = 24)
  (h2 : mina_digits = mina_carlos_ratio * (sam_digits - 6))
  (h3 : sam_digits = 10) :
  sam_digits - (mina_digits / mina_carlos_ratio) = 6 := by
sorry

end NUMINAMATH_CALUDE_pi_digits_difference_l2486_248665


namespace NUMINAMATH_CALUDE_max_distance_between_functions_l2486_248680

theorem max_distance_between_functions (a : ℝ) : 
  let f (x : ℝ) := 2 * (Real.cos (π / 4 + x))^2
  let g (x : ℝ) := Real.sqrt 3 * Real.cos (2 * x)
  let distance := |f a - g a|
  ∃ (max_distance : ℝ), max_distance = 3 ∧ distance ≤ max_distance ∧
    ∀ (b : ℝ), |f b - g b| ≤ max_distance :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_functions_l2486_248680


namespace NUMINAMATH_CALUDE_jimmy_pizza_cost_per_slice_l2486_248656

/-- Represents the cost of a pizza with toppings -/
def pizza_cost (base_cost : ℚ) (num_slices : ℕ) (first_topping_cost : ℚ) 
  (next_two_toppings_cost : ℚ) (rest_toppings_cost : ℚ) (num_toppings : ℕ) : ℚ :=
  let total_cost := base_cost + first_topping_cost + 
    (if num_toppings > 1 then min (num_toppings - 1) 2 * next_two_toppings_cost else 0) +
    (if num_toppings > 3 then (num_toppings - 3) * rest_toppings_cost else 0)
  total_cost / num_slices

theorem jimmy_pizza_cost_per_slice :
  pizza_cost 10 8 2 1 0.5 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_pizza_cost_per_slice_l2486_248656


namespace NUMINAMATH_CALUDE_chicken_price_per_pound_l2486_248616

/-- The price per pound of chicken given the conditions of Alice's grocery shopping --/
theorem chicken_price_per_pound (min_spend : ℝ) (amount_needed : ℝ)
  (chicken_weight : ℝ) (lettuce_price : ℝ) (tomatoes_price : ℝ)
  (sweet_potato_price : ℝ) (sweet_potato_count : ℕ)
  (broccoli_price : ℝ) (broccoli_count : ℕ)
  (brussels_sprouts_price : ℝ) :
  min_spend = 35 →
  amount_needed = 11 →
  chicken_weight = 1.5 →
  lettuce_price = 3 →
  tomatoes_price = 2.5 →
  sweet_potato_price = 0.75 →
  sweet_potato_count = 4 →
  broccoli_price = 2 →
  broccoli_count = 2 →
  brussels_sprouts_price = 2.5 →
  (min_spend - amount_needed - (lettuce_price + tomatoes_price +
    sweet_potato_price * sweet_potato_count + broccoli_price * broccoli_count +
    brussels_sprouts_price)) / chicken_weight = 6 := by
  sorry

end NUMINAMATH_CALUDE_chicken_price_per_pound_l2486_248616


namespace NUMINAMATH_CALUDE_max_abs_z_is_one_l2486_248644

theorem max_abs_z_is_one (a b c d z : ℂ) 
  (h1 : Complex.abs a = Complex.abs b)
  (h2 : Complex.abs b = Complex.abs c)
  (h3 : Complex.abs c = Complex.abs d)
  (h4 : Complex.abs a > 0)
  (h5 : a * z^3 + b * z^2 + c * z + d = 0) :
  Complex.abs z ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_is_one_l2486_248644


namespace NUMINAMATH_CALUDE_benson_ticket_cost_l2486_248667

/-- Calculates the total cost of concert tickets for Mr. Benson -/
def concert_ticket_cost (base_price : ℝ) (general_count : ℕ) (vip_count : ℕ) (premium_count : ℕ) 
  (vip_markup : ℝ) (premium_markup : ℝ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_count := general_count + vip_count + premium_count
  let vip_price := base_price * (1 + vip_markup)
  let premium_price := base_price * (1 + premium_markup)
  let discounted_count := max (total_count - discount_threshold) 0
  let general_cost := base_price * general_count
  let vip_cost := if vip_count ≤ discounted_count
                  then vip_price * vip_count * (1 - discount_rate)
                  else vip_price * (vip_count - discounted_count) + 
                       vip_price * discounted_count * (1 - discount_rate)
  let premium_cost := if premium_count ≤ (discounted_count - vip_count)
                      then premium_price * premium_count * (1 - discount_rate)
                      else premium_price * (premium_count - (discounted_count - vip_count)) +
                           premium_price * (discounted_count - vip_count) * (1 - discount_rate)
  general_cost + vip_cost + premium_cost

/-- Theorem stating that the total cost for Mr. Benson's tickets is $650.80 -/
theorem benson_ticket_cost : 
  concert_ticket_cost 40 10 3 2 0.2 0.5 10 0.05 = 650.80 := by
  sorry


end NUMINAMATH_CALUDE_benson_ticket_cost_l2486_248667


namespace NUMINAMATH_CALUDE_attendance_difference_l2486_248605

/-- The attendance difference between this week and last week for baseball games --/
theorem attendance_difference : 
  let second_game : ℕ := 80
  let first_game : ℕ := second_game - 20
  let third_game : ℕ := second_game + 15
  let this_week_total : ℕ := first_game + second_game + third_game
  let last_week_total : ℕ := 200
  this_week_total - last_week_total = 35 := by
  sorry

end NUMINAMATH_CALUDE_attendance_difference_l2486_248605


namespace NUMINAMATH_CALUDE_product_148_152_l2486_248627

theorem product_148_152 : 148 * 152 = 22496 := by
  sorry

end NUMINAMATH_CALUDE_product_148_152_l2486_248627


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2486_248606

theorem hyperbola_eccentricity (a : ℝ) : 
  a > 0 → 
  (∃ x y : ℝ, x^2/a^2 - y^2/3 = 1) → 
  (∃ c : ℝ, c/a = 2) → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2486_248606


namespace NUMINAMATH_CALUDE_intersection_and_conditions_l2486_248632

-- Define the intersection point M
def M : ℝ × ℝ := (-1, 2)

-- Define the given lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def line2 (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0
def line3 (x y : ℝ) : Prop := 2 * x + y + 5 = 0

-- Define the resulting lines
def result_line1 (x y : ℝ) : Prop := x = -1
def result_line2 (x y : ℝ) : Prop := x - 2 * y + 5 = 0

theorem intersection_and_conditions :
  -- M is the intersection point of line1 and line2
  (line1 M.1 M.2 ∧ line2 M.1 M.2) ∧
  -- result_line1 passes through M and (-1, 0)
  (result_line1 M.1 M.2 ∧ result_line1 (-1) 0) ∧
  -- result_line2 passes through M
  result_line2 M.1 M.2 ∧
  -- result_line2 is perpendicular to line3
  (∃ (k : ℝ), k ≠ 0 ∧ 1 * 2 + (-2) * 1 = -k * k) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_conditions_l2486_248632


namespace NUMINAMATH_CALUDE_hard_hats_remaining_l2486_248681

/-- The number of hard hats remaining in a truck after some are removed --/
def remaining_hard_hats (pink_initial green_initial yellow_initial : ℕ)
  (pink_carl pink_john green_john : ℕ) : ℕ :=
  (pink_initial - pink_carl - pink_john) +
  (green_initial - green_john) +
  yellow_initial

theorem hard_hats_remaining :
  remaining_hard_hats 26 15 24 4 6 12 = 43 := by
  sorry

end NUMINAMATH_CALUDE_hard_hats_remaining_l2486_248681


namespace NUMINAMATH_CALUDE_jose_join_time_l2486_248609

/-- Proves that Jose joined 2 months after Tom opened the shop given the investment and profit information -/
theorem jose_join_time (tom_investment : ℕ) (jose_investment : ℕ) (total_profit : ℕ) (jose_profit : ℕ) :
  tom_investment = 30000 →
  jose_investment = 45000 →
  total_profit = 36000 →
  jose_profit = 20000 →
  ∃ x : ℕ, 
    (tom_investment * 12) / (jose_investment * (12 - x)) = (total_profit - jose_profit) / jose_profit ∧
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_jose_join_time_l2486_248609


namespace NUMINAMATH_CALUDE_some_number_value_l2486_248671

/-- Given that (3.242 * 12) / some_number = 0.038903999999999994, 
    prove that some_number is approximately equal to 1000 -/
theorem some_number_value (some_number : ℝ) 
  (h : (3.242 * 12) / some_number = 0.038903999999999994) : 
  ∃ ε > 0, |some_number - 1000| < ε :=
sorry

end NUMINAMATH_CALUDE_some_number_value_l2486_248671


namespace NUMINAMATH_CALUDE_transportation_cost_independent_of_order_l2486_248631

/-- Represents a destination with its distance from the city and the weight of goods to be delivered -/
structure Destination where
  distance : ℝ
  weight : ℝ
  weight_eq_distance : weight = distance

/-- Calculates the cost of transportation for a single trip -/
def transportCost (d : Destination) (extraDistance : ℝ) : ℝ :=
  d.weight * (d.distance + extraDistance)

/-- Theorem stating that the total transportation cost is independent of the order of visits -/
theorem transportation_cost_independent_of_order (m n : Destination) :
  transportCost m 0 + transportCost n m.distance =
  transportCost n 0 + transportCost m n.distance := by
  sorry

#check transportation_cost_independent_of_order

end NUMINAMATH_CALUDE_transportation_cost_independent_of_order_l2486_248631


namespace NUMINAMATH_CALUDE_outstanding_student_awards_l2486_248635

/-- The number of ways to distribute n identical awards among k classes,
    with each class receiving at least one award. -/
def distribution_schemes (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 10 identical awards among 8 classes,
    with each class receiving at least one award. -/
theorem outstanding_student_awards : distribution_schemes 10 8 = 36 := by
  sorry

end NUMINAMATH_CALUDE_outstanding_student_awards_l2486_248635


namespace NUMINAMATH_CALUDE_sqrt_36_times_sqrt_16_l2486_248676

theorem sqrt_36_times_sqrt_16 : Real.sqrt (36 * Real.sqrt 16) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_36_times_sqrt_16_l2486_248676


namespace NUMINAMATH_CALUDE_alpha_value_l2486_248698

/-- Given that α is inversely proportional to β and directly proportional to γ,
    prove that α = 2.5 when β = 30 and γ = 6, given that α = 5 when β = 15 and γ = 3 -/
theorem alpha_value (α β γ : ℝ) (h1 : ∃ k : ℝ, α * β = k)
    (h2 : ∃ j : ℝ, α * γ = j) (h3 : α = 5 ∧ β = 15 ∧ γ = 3) :
  α = 2.5 ∧ β = 30 ∧ γ = 6 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l2486_248698


namespace NUMINAMATH_CALUDE_inverse_statement_l2486_248652

theorem inverse_statement : 
  (∀ x : ℝ, x > 1 → x^2 - 2*x + 3 > 0) ↔ 
  (∀ x : ℝ, x^2 - 2*x + 3 > 0 → x > 1) :=
by sorry

end NUMINAMATH_CALUDE_inverse_statement_l2486_248652


namespace NUMINAMATH_CALUDE_solution_equals_answer_l2486_248699

/-- A perfect square is an integer that is the square of another integer. -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m^2

/-- The set of all integer pairs (a, b) satisfying the given conditions. -/
def solution_set : Set (ℤ × ℤ) :=
  {p : ℤ × ℤ | is_perfect_square (p.1^2 - 4*p.2) ∧ is_perfect_square (p.2^2 - 4*p.1)}

/-- The set described in the answer. -/
def answer_set : Set (ℤ × ℤ) :=
  {p : ℤ × ℤ | (∃ n : ℤ, p = (0, n^2) ∨ p = (n^2, 0)) ∨
               (p.1 > 0 ∧ p.2 = -1 - p.1) ∨
               (p.2 > 0 ∧ p.1 = -1 - p.2) ∨
               p = (4, 4) ∨ p = (5, 6) ∨ p = (6, 5)}

theorem solution_equals_answer : solution_set = answer_set :=
  sorry

end NUMINAMATH_CALUDE_solution_equals_answer_l2486_248699


namespace NUMINAMATH_CALUDE_present_age_of_B_present_age_of_B_proof_l2486_248615

/-- The present age of person B given the conditions -/
theorem present_age_of_B : ℕ → ℕ → Prop :=
  fun a b =>
    (a + 10 = 2 * (b - 10)) →  -- In 10 years, A will be twice as old as B was 10 years ago
    (a = b + 4) →              -- A is now 4 years older than B
    b = 34                     -- B's current age is 34

-- The proof is omitted
theorem present_age_of_B_proof : ∃ a b, present_age_of_B a b :=
  sorry

end NUMINAMATH_CALUDE_present_age_of_B_present_age_of_B_proof_l2486_248615


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l2486_248621

theorem at_least_one_greater_than_one (x y : ℝ) (h : x + y > 2) :
  x > 1 ∨ y > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l2486_248621


namespace NUMINAMATH_CALUDE_mani_pedi_regular_price_l2486_248613

/-- The regular price of a mani/pedi, given a 25% discount, 5 purchases, and $150 total spent. -/
theorem mani_pedi_regular_price :
  ∀ (regular_price : ℝ),
  (regular_price * 0.75 * 5 = 150) →
  regular_price = 40 := by
sorry

end NUMINAMATH_CALUDE_mani_pedi_regular_price_l2486_248613


namespace NUMINAMATH_CALUDE_number_problem_l2486_248614

theorem number_problem : ∃ x : ℚ, (x / 6) * 12 = 11 ∧ x = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2486_248614


namespace NUMINAMATH_CALUDE_N_minus_M_eq_six_l2486_248629

def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {2, 3, 6}

theorem N_minus_M_eq_six : N \ M = {6} := by sorry

end NUMINAMATH_CALUDE_N_minus_M_eq_six_l2486_248629


namespace NUMINAMATH_CALUDE_milly_science_homework_time_l2486_248689

/-- The time Milly spends studying various subjects -/
structure StudyTime where
  math : ℕ
  geography : ℕ
  science : ℕ
  total : ℕ

/-- Milly's study time satisfies the given conditions -/
def millysStudyTime : StudyTime where
  math := 60
  geography := 30
  science := 45
  total := 135

theorem milly_science_homework_time :
  ∀ (st : StudyTime),
    st.math = 60 →
    st.geography = st.math / 2 →
    st.total = 135 →
    st.science = st.total - st.math - st.geography →
    st.science = 45 := by
  sorry

end NUMINAMATH_CALUDE_milly_science_homework_time_l2486_248689


namespace NUMINAMATH_CALUDE_min_value_theorem_l2486_248655

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 5 * x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2486_248655


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2486_248600

theorem sum_with_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2486_248600


namespace NUMINAMATH_CALUDE_asha_win_probability_l2486_248688

theorem asha_win_probability (p_lose p_tie : ℚ) : 
  p_lose = 3/8 → p_tie = 1/4 → 1 - p_lose - p_tie = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_asha_win_probability_l2486_248688


namespace NUMINAMATH_CALUDE_total_ribbons_used_l2486_248602

def dresses_per_day_first_week : ℕ := 2
def days_first_week : ℕ := 7
def dresses_per_day_second_week : ℕ := 3
def days_second_week : ℕ := 2
def ribbons_per_dress : ℕ := 2

theorem total_ribbons_used : 
  (dresses_per_day_first_week * days_first_week + 
   dresses_per_day_second_week * days_second_week) * 
  ribbons_per_dress = 40 := by sorry

end NUMINAMATH_CALUDE_total_ribbons_used_l2486_248602


namespace NUMINAMATH_CALUDE_largest_non_square_sum_diff_smallest_n_with_square_digit_sum_n_66_has_square_digit_sum_l2486_248619

def is_sum_of_squares (x : ℕ) : Prop := ∃ a b : ℕ, x = a^2 + b^2

def is_diff_of_squares (x : ℕ) : Prop := ∃ a b : ℕ, x = a^2 - b^2

def sum_of_digit_squares (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.map (λ d => d^2)).sum

def largest_n_digit_number (n : ℕ) : ℕ := 10^n - 1

theorem largest_non_square_sum_diff (n : ℕ) (h : n > 2) :
  ∀ k : ℕ, k ≤ largest_n_digit_number n →
    (¬ is_sum_of_squares k ∧ ¬ is_diff_of_squares k) →
    k ≤ 10^n - 2 :=
sorry

theorem smallest_n_with_square_digit_sum :
  ∀ n : ℕ, n < 66 → ¬ ∃ k : ℕ, sum_of_digit_squares n = k^2 :=
sorry

theorem n_66_has_square_digit_sum :
  ∃ k : ℕ, sum_of_digit_squares 66 = k^2 :=
sorry

end NUMINAMATH_CALUDE_largest_non_square_sum_diff_smallest_n_with_square_digit_sum_n_66_has_square_digit_sum_l2486_248619


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2486_248626

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo 1 3 ↔ a * x^2 + b * x + 3 < 0) →
  a + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2486_248626


namespace NUMINAMATH_CALUDE_chicken_crossed_road_l2486_248646

/-- The number of cars dodged by a chicken crossing a road, given the initial and final feather counts. -/
def cars_dodged (initial_feathers final_feathers : ℕ) : ℕ :=
  (initial_feathers - final_feathers) / 2

/-- Theorem stating that the chicken dodged 23 cars given the problem conditions. -/
theorem chicken_crossed_road (initial_feathers final_feathers : ℕ) 
  (h1 : initial_feathers = 5263)
  (h2 : final_feathers = 5217) :
  cars_dodged initial_feathers final_feathers = 23 := by
  sorry

#eval cars_dodged 5263 5217

end NUMINAMATH_CALUDE_chicken_crossed_road_l2486_248646


namespace NUMINAMATH_CALUDE_quadratic_sum_equals_five_l2486_248612

theorem quadratic_sum_equals_five (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) : x^2 + y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_equals_five_l2486_248612


namespace NUMINAMATH_CALUDE_total_travel_methods_eq_thirteen_l2486_248687

/-- The number of bus services from A to B -/
def bus_services : ℕ := 8

/-- The number of train services from A to B -/
def train_services : ℕ := 3

/-- The number of ship services from A to B -/
def ship_services : ℕ := 2

/-- The total number of different methods to travel from A to B -/
def total_travel_methods : ℕ := bus_services + train_services + ship_services

theorem total_travel_methods_eq_thirteen :
  total_travel_methods = 13 := by sorry

end NUMINAMATH_CALUDE_total_travel_methods_eq_thirteen_l2486_248687


namespace NUMINAMATH_CALUDE_joes_steakhouse_wages_l2486_248678

/-- Proves that the hourly wage of a manager is $8.5 given the conditions from Joe's Steakhouse --/
theorem joes_steakhouse_wages (manager_wage dishwasher_wage chef_wage : ℝ) : 
  chef_wage = dishwasher_wage * 1.22 →
  dishwasher_wage = manager_wage / 2 →
  chef_wage = manager_wage - 3.315 →
  manager_wage = 8.5 := by
sorry

end NUMINAMATH_CALUDE_joes_steakhouse_wages_l2486_248678


namespace NUMINAMATH_CALUDE_petyas_friends_l2486_248670

theorem petyas_friends (total_stickers : ℕ) : 
  (∃ (friends : ℕ), total_stickers = 5 * friends + 8 ∧ total_stickers = 6 * friends - 11) →
  (∃ (friends : ℕ), friends = 19 ∧ total_stickers = 5 * friends + 8 ∧ total_stickers = 6 * friends - 11) :=
by sorry

end NUMINAMATH_CALUDE_petyas_friends_l2486_248670


namespace NUMINAMATH_CALUDE_clubsuit_equality_theorem_l2486_248645

-- Define the clubsuit operation
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the set of points (x, y) where x ⋆ y = y ⋆ x
def equality_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | clubsuit p.1 p.2 = clubsuit p.2 p.1}

-- Define the set of points on x-axis, y-axis, y = x, and y = -x
def target_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2}

theorem clubsuit_equality_theorem : equality_set = target_set := by
  sorry


end NUMINAMATH_CALUDE_clubsuit_equality_theorem_l2486_248645


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2486_248662

theorem purely_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (m^2 - m - 2) + (m + 1)*Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = 2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2486_248662


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2486_248624

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x ≥ 0}
def N : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2486_248624


namespace NUMINAMATH_CALUDE_simplify_expression_l2486_248696

theorem simplify_expression :
  (8 * 10^12) / (4 * 10^4) + 2 * 10^3 = 200002000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2486_248696


namespace NUMINAMATH_CALUDE_soccer_ball_surface_area_l2486_248692

theorem soccer_ball_surface_area (circumference : ℝ) (h : circumference = 69) :
  (4 * π * (circumference / (2 * π))^2) = 4761 / π := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_surface_area_l2486_248692


namespace NUMINAMATH_CALUDE_monster_count_is_thirteen_l2486_248669

/-- Represents the state of the battlefield --/
structure Battlefield where
  ultraman_heads : Nat
  ultraman_legs : Nat
  initial_monster_heads : Nat
  initial_monster_legs : Nat
  split_monster_heads : Nat
  split_monster_legs : Nat
  total_heads : Nat
  total_legs : Nat

/-- Calculates the number of monsters on the battlefield --/
def count_monsters (b : Battlefield) : Nat :=
  let remaining_heads := b.total_heads - b.ultraman_heads
  let remaining_legs := b.total_legs - b.ultraman_legs
  let initial_monsters := remaining_heads / b.initial_monster_heads
  let extra_legs := remaining_legs - (initial_monsters * b.initial_monster_legs)
  let splits := extra_legs / (2 * b.split_monster_legs - b.initial_monster_legs)
  initial_monsters + splits

/-- The main theorem stating that the number of monsters is 13 --/
theorem monster_count_is_thirteen (b : Battlefield) 
  (h1 : b.ultraman_heads = 1)
  (h2 : b.ultraman_legs = 2)
  (h3 : b.initial_monster_heads = 2)
  (h4 : b.initial_monster_legs = 5)
  (h5 : b.split_monster_heads = 1)
  (h6 : b.split_monster_legs = 6)
  (h7 : b.total_heads = 21)
  (h8 : b.total_legs = 73) :
  count_monsters b = 13 := by
  sorry

#eval count_monsters {
  ultraman_heads := 1,
  ultraman_legs := 2,
  initial_monster_heads := 2,
  initial_monster_legs := 5,
  split_monster_heads := 1,
  split_monster_legs := 6,
  total_heads := 21,
  total_legs := 73
}

end NUMINAMATH_CALUDE_monster_count_is_thirteen_l2486_248669


namespace NUMINAMATH_CALUDE_triangle_inradius_l2486_248608

/-- Given a triangle with perimeter 35 cm and area 78.75 cm², prove its inradius is 4.5 cm -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : p = 35)
  (h_area : A = 78.75)
  (h_inradius : A = r * p / 2) : 
  r = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l2486_248608


namespace NUMINAMATH_CALUDE_product_seven_reciprocal_squares_sum_l2486_248630

theorem product_seven_reciprocal_squares_sum (a b : ℕ) (h : a * b = 7) :
  (1 : ℚ) / (a^2 : ℚ) + (1 : ℚ) / (b^2 : ℚ) = 50 / 49 := by
  sorry

end NUMINAMATH_CALUDE_product_seven_reciprocal_squares_sum_l2486_248630


namespace NUMINAMATH_CALUDE_expression_evaluation_l2486_248684

theorem expression_evaluation (x : ℝ) (h : x = 1) : 
  (x - 1)^2 + (x + 1)*(x - 1) - 2*x^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2486_248684


namespace NUMINAMATH_CALUDE_johns_money_in_euros_johns_money_in_euros_proof_l2486_248679

/-- Proves that John's money in Euros is 612 given the conditions of the problem -/
theorem johns_money_in_euros : ℝ → Prop :=
  fun conversion_rate =>
    ∀ (darwin mia laura john : ℝ),
      darwin = 45 →
      mia = 2 * darwin + 20 →
      laura = 3 * (mia + darwin) - 30 →
      john = 1.5 * (laura + darwin) →
      conversion_rate = 0.85 →
      john * conversion_rate = 612

/-- Proof of the theorem -/
theorem johns_money_in_euros_proof : johns_money_in_euros 0.85 := by
  sorry

end NUMINAMATH_CALUDE_johns_money_in_euros_johns_money_in_euros_proof_l2486_248679


namespace NUMINAMATH_CALUDE_test_modes_l2486_248660

/-- Represents the frequency of each score in the test --/
def score_frequency : List (Nat × Nat) := [
  (65, 2), (73, 1), (82, 1), (88, 1),
  (91, 1), (96, 4), (102, 1), (104, 4), (110, 3)
]

/-- Finds the modes of a list of score frequencies --/
def find_modes (frequencies : List (Nat × Nat)) : List Nat :=
  sorry

/-- States that 96 and 104 are the modes of the given score frequencies --/
theorem test_modes : find_modes score_frequency = [96, 104] := by
  sorry

end NUMINAMATH_CALUDE_test_modes_l2486_248660


namespace NUMINAMATH_CALUDE_initial_speed_is_4_l2486_248620

/-- Represents the scenario of a person walking to a bus stand -/
structure BusScenario where
  distance : ℝ  -- Distance to the bus stand in km
  faster_speed : ℝ  -- Speed at which the person arrives early (km/h)
  early_time : ℝ  -- Time arrived early when walking at faster_speed (minutes)
  late_time : ℝ  -- Time arrived late when walking at initial speed (minutes)

/-- Calculates the initial walking speed given a BusScenario -/
def initial_speed (scenario : BusScenario) : ℝ :=
  sorry

/-- Theorem stating that the initial walking speed is 4 km/h for the given scenario -/
theorem initial_speed_is_4 (scenario : BusScenario) 
  (h1 : scenario.distance = 5)
  (h2 : scenario.faster_speed = 5)
  (h3 : scenario.early_time = 5)
  (h4 : scenario.late_time = 10) :
  initial_speed scenario = 4 :=
sorry

end NUMINAMATH_CALUDE_initial_speed_is_4_l2486_248620


namespace NUMINAMATH_CALUDE_rebus_puzzle_solution_l2486_248641

theorem rebus_puzzle_solution :
  ∀ (A B C : ℕ),
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A ≠ B → B ≠ C → A ≠ C →
    A < 10 → B < 10 → C < 10 →
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) = (100 * A + 10 * C + C) →
    (100 * A + 10 * C + C) = 1416 →
    A = 4 ∧ B = 7 ∧ C = 6 := by
  sorry

end NUMINAMATH_CALUDE_rebus_puzzle_solution_l2486_248641


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2486_248686

/-- A function is an "H function" if for any two distinct real numbers x₁ and x₂,
    it satisfies x₁f(x₁) + x₂f(x₂) > x₁f(x₂) + x₂f(x₁) -/
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function is strictly increasing if for any two real numbers x₁ < x₂,
    we have f(x₁) < f(x₂) -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ strictly_increasing f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2486_248686


namespace NUMINAMATH_CALUDE_store_socks_problem_l2486_248683

theorem store_socks_problem (x y w z : ℕ) : 
  x + y + w + z = 15 →
  x + 2*y + 3*w + 4*z = 36 →
  x ≥ 1 →
  y ≥ 1 →
  w ≥ 1 →
  z ≥ 1 →
  x = 5 :=
by sorry

end NUMINAMATH_CALUDE_store_socks_problem_l2486_248683


namespace NUMINAMATH_CALUDE_paper_tray_height_l2486_248640

/-- The height of a paper tray formed from a square sheet -/
theorem paper_tray_height (side_length : ℝ) (cut_start : ℝ) : 
  side_length = 120 →
  cut_start = Real.sqrt 20 →
  2 * Real.sqrt 5 = 
    (Real.sqrt 2 * cut_start) / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_paper_tray_height_l2486_248640


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l2486_248650

/-- If the sum of two monomials 3x^5y^m and -2x^ny^7 is still a monomial in terms of x and y, 
    then m - n = 2 -/
theorem monomial_sum_condition (m n : ℕ) : 
  (∃ (a : ℚ) (p q : ℕ), 3 * X^5 * Y^m + -2 * X^n * Y^7 = a * X^p * Y^q) → 
  m - n = 2 := by
  sorry

end NUMINAMATH_CALUDE_monomial_sum_condition_l2486_248650


namespace NUMINAMATH_CALUDE_length_difference_l2486_248611

/-- Represents a rectangular plot. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ

/-- The cost of fencing per meter. -/
def fencingCostPerMeter : ℝ := 26.50

/-- The total cost of fencing the plot. -/
def totalFencingCost : ℝ := 5300

/-- Calculates the perimeter of a rectangular plot. -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

theorem length_difference (plot : RectangularPlot) :
  plot.length = 57 →
  perimeter plot = totalFencingCost / fencingCostPerMeter →
  plot.length - plot.breadth = 14 := by
  sorry

end NUMINAMATH_CALUDE_length_difference_l2486_248611


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_504_l2486_248663

theorem factorial_ratio_equals_504 : ∃! n : ℕ, n > 0 ∧ n.factorial / (n - 3).factorial = 504 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_504_l2486_248663


namespace NUMINAMATH_CALUDE_continuity_at_two_l2486_248691

/-- The function f(x) = -2x^2 - 5 is continuous at x₀ = 2 -/
theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |(-2 * x^2 - 5) - (-2 * 2^2 - 5)| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_two_l2486_248691
