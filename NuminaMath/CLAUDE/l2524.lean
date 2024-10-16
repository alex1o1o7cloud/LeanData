import Mathlib

namespace NUMINAMATH_CALUDE_frustum_volume_l2524_252423

/-- Given a square pyramid and a smaller pyramid cut from it parallel to the base,
    calculate the volume of the resulting frustum. -/
theorem frustum_volume
  (base_edge : ℝ)
  (altitude : ℝ)
  (small_base_edge : ℝ)
  (small_altitude : ℝ)
  (h_base : base_edge = 15)
  (h_altitude : altitude = 10)
  (h_small_base : small_base_edge = 7.5)
  (h_small_altitude : small_altitude = 5) :
  (1 / 3 * base_edge^2 * altitude) - (1 / 3 * small_base_edge^2 * small_altitude) = 656.25 :=
by sorry

end NUMINAMATH_CALUDE_frustum_volume_l2524_252423


namespace NUMINAMATH_CALUDE_valid_seq_equals_fib_prob_no_consecutive_ones_l2524_252488

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Sequence of valid arrangements -/
def validSeq : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validSeq (n + 1) + validSeq n

theorem valid_seq_equals_fib (n : ℕ) : validSeq n = fib (n + 2) := by
  sorry

theorem prob_no_consecutive_ones : 
  (validSeq 12 : ℚ) / 2^12 = 377 / 4096 := by
  sorry

#eval fib 14 + 4096  -- Should output 4473


end NUMINAMATH_CALUDE_valid_seq_equals_fib_prob_no_consecutive_ones_l2524_252488


namespace NUMINAMATH_CALUDE_upstream_distance_calculation_l2524_252489

/-- Represents the problem of calculating the upstream distance rowed by a man --/
theorem upstream_distance_calculation
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (current_velocity : ℝ)
  (h1 : downstream_distance = 32)
  (h2 : downstream_time = 6)
  (h3 : upstream_time = 6)
  (h4 : current_velocity = 1.5)
  (h5 : downstream_time > 0)
  (h6 : upstream_time > 0)
  (h7 : current_velocity ≥ 0) :
  let still_water_speed := downstream_distance / downstream_time - current_velocity
  let upstream_distance := (still_water_speed - current_velocity) * upstream_time
  upstream_distance = 14 := by sorry


end NUMINAMATH_CALUDE_upstream_distance_calculation_l2524_252489


namespace NUMINAMATH_CALUDE_tyler_meal_choices_l2524_252463

def meat_options : ℕ := 3
def vegetable_options : ℕ := 5
def dessert_options : ℕ := 4
def drink_options : ℕ := 4
def vegetables_to_choose : ℕ := 3

def number_of_meals : ℕ := meat_options * Nat.choose vegetable_options vegetables_to_choose * dessert_options * drink_options

theorem tyler_meal_choices : number_of_meals = 480 := by
  sorry

end NUMINAMATH_CALUDE_tyler_meal_choices_l2524_252463


namespace NUMINAMATH_CALUDE_age_equals_birth_year_digit_sum_l2524_252446

theorem age_equals_birth_year_digit_sum :
  ∃! A : ℕ, 0 ≤ A ∧ A ≤ 99 ∧
  (∃ x y : ℕ, 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
    A = 1893 - (1800 + 10 * x + y) ∧
    A = 1 + 8 + x + y) ∧
  A = 24 :=
sorry

end NUMINAMATH_CALUDE_age_equals_birth_year_digit_sum_l2524_252446


namespace NUMINAMATH_CALUDE_largest_class_size_l2524_252434

/-- Proves that in a school with 5 classes, where each class has 2 students less than the previous class,
    and the total number of students is 140, the number of students in the largest class is 32. -/
theorem largest_class_size (num_classes : Nat) (student_difference : Nat) (total_students : Nat)
    (h1 : num_classes = 5)
    (h2 : student_difference = 2)
    (h3 : total_students = 140) :
    ∃ (x : Nat), x = 32 ∧ 
    (x + (x - student_difference) + (x - 2*student_difference) + 
     (x - 3*student_difference) + (x - 4*student_difference) = total_students) :=
  by sorry

end NUMINAMATH_CALUDE_largest_class_size_l2524_252434


namespace NUMINAMATH_CALUDE_f_inequality_l2524_252414

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the conditions
axiom period : ∀ x : ℝ, f (x + 3) = f (x - 3)
axiom even_shifted : ∀ x : ℝ, f (x + 3) = f (-x + 3)
axiom decreasing : ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 3 → f y < f x

-- State the theorem
theorem f_inequality : f 3.5 < f (-4.5) ∧ f (-4.5) < f 12.5 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2524_252414


namespace NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l2524_252402

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_reciprocal_sum
  (a : ℕ → ℝ)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_geometric : is_geometric_sequence a)
  (h_sum : a 0 + a 1 + a 2 + a 3 = 9)
  (h_product : a 0 * a 1 * a 2 * a 3 = 81 / 4) :
  (1 / a 0) + (1 / a 1) + (1 / a 2) + (1 / a 3) = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l2524_252402


namespace NUMINAMATH_CALUDE_expression_equals_half_y_l2524_252453

theorem expression_equals_half_y (y d : ℝ) (hy : y > 0) : 
  (4 * y) / 20 + (3 * y) / d = 0.5 * y → d = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_half_y_l2524_252453


namespace NUMINAMATH_CALUDE_correct_calculation_l2524_252403

theorem correct_calculation (x : ℝ) (h : 7 * x = 70) : 36 - x = 26 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2524_252403


namespace NUMINAMATH_CALUDE_no_common_terms_except_one_l2524_252455

def x : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n+2) => x (n+1) + 2 * x n

def y : ℕ → ℕ
  | 0 => 1
  | 1 => 7
  | (n+2) => 2 * y (n+1) + 3 * y n

theorem no_common_terms_except_one (n : ℕ) (m : ℕ) (h : n ≥ 1) :
  x n ≠ y m ∨ (x n = y m ∧ n = 0 ∧ m = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_common_terms_except_one_l2524_252455


namespace NUMINAMATH_CALUDE_percentage_relationship_z_less_than_y_l2524_252433

theorem percentage_relationship (w e y z : ℝ) 
  (hw : w = 0.6 * e)
  (he : e = 0.6 * y)
  (hz : z = w * 1.5000000000000002) :
  z = 0.54 * y :=
by sorry

-- The final result can be derived from this theorem
theorem z_less_than_y (w e y z : ℝ) 
  (hw : w = 0.6 * e)
  (he : e = 0.6 * y)
  (hz : z = w * 1.5000000000000002) :
  (y - z) / y = 0.46 :=
by sorry

end NUMINAMATH_CALUDE_percentage_relationship_z_less_than_y_l2524_252433


namespace NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l2524_252464

/-- Given an exponential function y = a^x, if the sum of its maximum and minimum values 
    on the interval [0,1] is 3, then a = 2 -/
theorem exponential_function_sum_of_extrema (a : ℝ) : 
  (a > 0) → 
  (∀ x ∈ Set.Icc 0 1, ∃ y, y = a^x) →
  (Real.exp a + 1 = 3 ∨ a + Real.exp a = 3) →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l2524_252464


namespace NUMINAMATH_CALUDE_intersection_k_range_l2524_252481

-- Define the line and hyperbola equations
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 2
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 6

-- Define the condition for intersection points
def intersects_right_branch (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
  hyperbola x₁ (line k x₁) ∧ hyperbola x₂ (line k x₂)

-- Theorem statement
theorem intersection_k_range :
  ∀ k : ℝ, intersects_right_branch k ↔ -Real.sqrt 15 / 3 < k ∧ k < -1 :=
sorry

end NUMINAMATH_CALUDE_intersection_k_range_l2524_252481


namespace NUMINAMATH_CALUDE_relief_supplies_total_l2524_252412

/-- The total amount of relief supplies in tons -/
def total_supplies : ℝ := 644

/-- Team A's daily transport capacity in tons -/
def team_a_capacity : ℝ := 64.4

/-- The percentage by which Team A's capacity exceeds Team B's -/
def capacity_difference_percentage : ℝ := 75

/-- The additional amount Team A has transported when it reaches half the total supplies -/
def additional_transport : ℝ := 138

/-- Theorem stating the total amount of relief supplies -/
theorem relief_supplies_total : 
  ∃ (team_b_capacity : ℝ),
    team_a_capacity = team_b_capacity * (1 + capacity_difference_percentage / 100) ∧
    (total_supplies / 2) - (total_supplies / 2 - additional_transport) = 
      (team_a_capacity - team_b_capacity) * (total_supplies / (2 * team_a_capacity)) ∧
    total_supplies = 644 :=
by sorry

end NUMINAMATH_CALUDE_relief_supplies_total_l2524_252412


namespace NUMINAMATH_CALUDE_h_equals_neg_f_of_six_minus_x_l2524_252460

-- Define a function that reflects a graph across the y-axis
def reflectY (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (-x)

-- Define a function that reflects a graph across the x-axis
def reflectX (f : ℝ → ℝ) : ℝ → ℝ := λ x => -f x

-- Define a function that shifts a graph to the right by a given amount
def shiftRight (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f (x - shift)

-- Define the composition of these transformations
def h (f : ℝ → ℝ) : ℝ → ℝ := shiftRight (reflectX (reflectY f)) 6

-- State the theorem
theorem h_equals_neg_f_of_six_minus_x (f : ℝ → ℝ) : 
  ∀ x : ℝ, h f x = -f (6 - x) := by sorry

end NUMINAMATH_CALUDE_h_equals_neg_f_of_six_minus_x_l2524_252460


namespace NUMINAMATH_CALUDE_man_in_well_l2524_252421

/-- The number of days required for a man to climb out of a well -/
def daysToClimbOut (wellDepth : ℕ) (climbUp : ℕ) (slipDown : ℕ) : ℕ :=
  let netClimbPerDay := climbUp - slipDown
  (wellDepth - 1) / netClimbPerDay + 1

/-- Theorem: It takes 30 days for a man to climb out of a 30-meter deep well
    when he climbs 4 meters up and slips 3 meters down each day -/
theorem man_in_well : daysToClimbOut 30 4 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_man_in_well_l2524_252421


namespace NUMINAMATH_CALUDE_second_smallest_packs_l2524_252443

/-- The number of hot dogs in each pack -/
def hot_dogs_per_pack : ℕ := 9

/-- The number of buns in each pack -/
def buns_per_pack : ℕ := 7

/-- The number of hot dogs left over after the barbecue -/
def leftover_hot_dogs : ℕ := 6

/-- 
Theorem: The second smallest number of packs of hot dogs that satisfies 
the conditions of the barbecue problem is 10.
-/
theorem second_smallest_packs : 
  (∃ m : ℕ, m < 10 ∧ hot_dogs_per_pack * m ≡ leftover_hot_dogs [MOD buns_per_pack]) ∧
  (∀ k : ℕ, k < 10 → hot_dogs_per_pack * k ≡ leftover_hot_dogs [MOD buns_per_pack] → 
    ∃ m : ℕ, m < k ∧ hot_dogs_per_pack * m ≡ leftover_hot_dogs [MOD buns_per_pack]) ∧
  hot_dogs_per_pack * 10 ≡ leftover_hot_dogs [MOD buns_per_pack] := by
  sorry


end NUMINAMATH_CALUDE_second_smallest_packs_l2524_252443


namespace NUMINAMATH_CALUDE_factorization_problems_l2524_252454

theorem factorization_problems :
  (∀ a b : ℝ, a^2 * b - a * b^2 = a * b * (a - b)) ∧
  (∀ x : ℝ, 2 * x^2 - 8 = 2 * (x + 2) * (x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l2524_252454


namespace NUMINAMATH_CALUDE_cos_inv_third_over_pi_irrational_l2524_252451

theorem cos_inv_third_over_pi_irrational : Irrational ((Real.arccos (1/3)) / Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cos_inv_third_over_pi_irrational_l2524_252451


namespace NUMINAMATH_CALUDE_ten_strikes_l2524_252475

/-- Represents the time it takes for a clock to strike a given number of times -/
def strike_time (strikes : ℕ) : ℝ :=
  sorry

/-- The clock takes 42 seconds to strike 7 times -/
axiom seven_strikes : strike_time 7 = 42

/-- Theorem: It takes 60 seconds for the clock to strike 10 times -/
theorem ten_strikes : strike_time 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ten_strikes_l2524_252475


namespace NUMINAMATH_CALUDE_power_seven_strictly_increasing_l2524_252459

theorem power_seven_strictly_increasing (m n : ℝ) (h : m < n) : m^7 < n^7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_strictly_increasing_l2524_252459


namespace NUMINAMATH_CALUDE_compare_fractions_l2524_252405

theorem compare_fractions : -3/8 > -4/9 := by sorry

end NUMINAMATH_CALUDE_compare_fractions_l2524_252405


namespace NUMINAMATH_CALUDE_beth_candies_theorem_l2524_252417

def total_candies : ℕ := 10

def is_valid_distribution (a b c : ℕ) : Prop :=
  a + b + c = total_candies ∧ a ≥ 3 ∧ b ≥ 2 ∧ c ≥ 2 ∧ c ≤ 3

def possible_beth_candies : Set ℕ := {2, 3, 4, 5}

theorem beth_candies_theorem :
  ∀ b : ℕ, (∃ a c : ℕ, is_valid_distribution a b c) ↔ b ∈ possible_beth_candies :=
sorry

end NUMINAMATH_CALUDE_beth_candies_theorem_l2524_252417


namespace NUMINAMATH_CALUDE_remaining_black_cards_l2524_252426

theorem remaining_black_cards (total_cards : Nat) (black_cards : Nat) (removed_cards : Nat) :
  total_cards = 52 →
  black_cards = 26 →
  removed_cards = 4 →
  black_cards - removed_cards = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_black_cards_l2524_252426


namespace NUMINAMATH_CALUDE_infinite_primes_4k_plus_3_l2524_252486

theorem infinite_primes_4k_plus_3 :
  ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4*k + 3) →
  ∃ q, Nat.Prime q ∧ (∃ m, q = 4*m + 3) ∧ q ∉ S :=
by sorry

end NUMINAMATH_CALUDE_infinite_primes_4k_plus_3_l2524_252486


namespace NUMINAMATH_CALUDE_circle_center_l2524_252496

/-- Given a circle with equation x^2 + y^2 - 2x + 4y - 4 = 0, 
    its center coordinates are (1, -2) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 4*y - 4 = 0) → 
  ∃ (h : ℝ), (x - 1)^2 + (y + 2)^2 = h^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l2524_252496


namespace NUMINAMATH_CALUDE_parallelogram_side_comparison_l2524_252466

structure Parallelogram where
  sides : Fin 4 → ℝ
  parallel : sides 0 = sides 2 ∧ sides 1 = sides 3

def inscribed (P Q : Parallelogram) : Prop :=
  ∀ i : Fin 4, P.sides i ≤ Q.sides i

theorem parallelogram_side_comparison 
  (P₁ P₂ P₃ : Parallelogram)
  (h₁ : inscribed P₂ P₁)
  (h₂ : inscribed P₃ P₂)
  (h₃ : ∀ i : Fin 4, P₃.sides i ≤ P₁.sides i) :
  ∃ i : Fin 4, P₁.sides i ≤ 2 * P₃.sides i :=
sorry

end NUMINAMATH_CALUDE_parallelogram_side_comparison_l2524_252466


namespace NUMINAMATH_CALUDE_art_math_supplies_cost_l2524_252479

-- Define the prices of items
def folder_price : ℚ := 3.5
def notebook_price : ℚ := 3
def binder_price : ℚ := 5
def pencil_price : ℚ := 1
def eraser_price : ℚ := 0.75
def highlighter_price : ℚ := 3.25
def marker_price : ℚ := 3.5
def sticky_note_price : ℚ := 2.5
def calculator_price : ℚ := 10.5
def sketchbook_price : ℚ := 4.5
def paint_set_price : ℚ := 18
def color_pencil_price : ℚ := 7

-- Define the quantities
def num_classes : ℕ := 12
def folders_per_class : ℕ := 1
def notebooks_per_class : ℕ := 2
def binders_per_class : ℕ := 1
def pencils_per_class : ℕ := 3
def erasers_per_6_pencils : ℕ := 2

-- Define the total spent
def total_spent : ℚ := 210

-- Theorem statement
theorem art_math_supplies_cost : 
  paint_set_price + color_pencil_price + calculator_price + sketchbook_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_art_math_supplies_cost_l2524_252479


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l2524_252400

theorem square_difference_of_sum_and_diff (x y : ℕ) 
  (sum_eq : x + y = 70) 
  (diff_eq : x - y = 20) 
  (pos_x : x > 0) 
  (pos_y : y > 0) : 
  x^2 - y^2 = 1400 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l2524_252400


namespace NUMINAMATH_CALUDE_eight_power_plus_six_divisible_by_seven_l2524_252480

theorem eight_power_plus_six_divisible_by_seven (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℤ, (8 : ℤ)^n + 6 = 7 * k :=
by sorry

end NUMINAMATH_CALUDE_eight_power_plus_six_divisible_by_seven_l2524_252480


namespace NUMINAMATH_CALUDE_factorization_proof_l2524_252435

theorem factorization_proof (x : ℝ) : 4 * x^2 - 1 = (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2524_252435


namespace NUMINAMATH_CALUDE_like_terms_exponent_l2524_252467

/-- 
Given two terms -3x^(2m)y^3 and 2x^4y^n are like terms,
prove that m^n = 8
-/
theorem like_terms_exponent (m n : ℕ) : 
  (∃ (x y : ℝ), -3 * x^(2*m) * y^3 = 2 * x^4 * y^n) → m^n = 8 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_l2524_252467


namespace NUMINAMATH_CALUDE_at_least_one_red_ball_probability_l2524_252408

theorem at_least_one_red_ball_probability 
  (prob_red_A : ℝ) 
  (prob_red_B : ℝ) 
  (h1 : prob_red_A = 1/3) 
  (h2 : prob_red_B = 1/2) :
  1 - (1 - prob_red_A) * (1 - prob_red_B) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_red_ball_probability_l2524_252408


namespace NUMINAMATH_CALUDE_exists_number_not_divisible_by_both_l2524_252468

def numbers : List Nat := [3654, 3664, 3674, 3684, 3694]

def divisible_by_4 (n : Nat) : Prop := n % 4 = 0

def divisible_by_3 (n : Nat) : Prop := n % 3 = 0

def units_digit (n : Nat) : Nat := n % 10

def tens_digit (n : Nat) : Nat := (n / 10) % 10

theorem exists_number_not_divisible_by_both :
  ∃ n ∈ numbers, ¬(divisible_by_4 n ∧ divisible_by_3 n) ∧
  (units_digit n * tens_digit n = 28 ∨ units_digit n * tens_digit n = 36) :=
by sorry

end NUMINAMATH_CALUDE_exists_number_not_divisible_by_both_l2524_252468


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_and_fixed_point_l2524_252415

noncomputable section

-- Define the ellipse Γ
def Γ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1

-- Define the circle E
def E (x y : ℝ) : Prop := x^2 + (y - 3/2)^2 = 4

-- Define point D
def D : ℝ × ℝ := (0, -1/2)

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define symmetry about y-axis
def symmetric_about_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem ellipse_eccentricity_and_fixed_point 
  (a : ℝ) (A B : ℝ × ℝ) 
  (h1 : a > 1)
  (h2 : Γ a A.1 A.2)
  (h3 : Γ a B.1 B.2)
  (h4 : E A.1 A.2)
  (h5 : E B.1 B.2)
  (h6 : distance A B = 2 * Real.sqrt 3) :
  (∃ (e : ℝ), e = Real.sqrt 3 / 2 ∧ 
    e^2 = 1 - 1/a^2) ∧ 
  (∀ (M N N' : ℝ × ℝ), 
    Γ a M.1 M.2 → Γ a N.1 N.2 → symmetric_about_y_axis N N' →
    (∃ (k : ℝ), M.2 - D.2 = k * (M.1 - D.1) ∧ 
                N.2 - D.2 = k * (N.1 - D.1)) →
    ∃ (t : ℝ), M.2 - N'.2 = (M.1 - N'.1) * (0 - M.1) / (t - M.1) ∧ 
               t = 0 ∧ M.2 - (0 - M.1) * (M.2 - N'.2) / (M.1 - N'.1) = -2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_and_fixed_point_l2524_252415


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2524_252431

/-- The polynomial f(x) = x^3 - 3x^2 - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2524_252431


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l2524_252497

-- Define the walking rates and duration
def jay_rate : ℚ := 1 / 12  -- miles per minute
def sarah_rate : ℚ := 3 / 36  -- miles per minute
def duration : ℚ := 2 * 60  -- 2 hours in minutes

-- Define the theorem
theorem distance_after_two_hours :
  let jay_distance := jay_rate * duration
  let sarah_distance := sarah_rate * duration
  jay_distance + sarah_distance = 20 := by sorry

end NUMINAMATH_CALUDE_distance_after_two_hours_l2524_252497


namespace NUMINAMATH_CALUDE_min_value_problem_1_l2524_252450

theorem min_value_problem_1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / (1 + y) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_1_l2524_252450


namespace NUMINAMATH_CALUDE_journey_distance_l2524_252411

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 20)
  (h2 : speed1 = 10)
  (h3 : speed2 = 15) : 
  ∃ (distance : ℝ), 
    distance / (2 * speed1) + distance / (2 * speed2) = total_time ∧ 
    distance = 240 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l2524_252411


namespace NUMINAMATH_CALUDE_money_distribution_l2524_252441

/-- The problem of distributing money among five people with specific conditions -/
theorem money_distribution (a b c d e : ℕ) : 
  a + b + c + d + e = 1010 ∧
  (a - 25) / 4 = (b - 10) / 3 ∧
  (a - 25) / 4 = (c - 15) / 6 ∧
  (a - 25) / 4 = (d - 20) / 2 ∧
  (a - 25) / 4 = (e - 30) / 5 →
  c = 288 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l2524_252441


namespace NUMINAMATH_CALUDE_equal_amount_after_15_days_l2524_252452

/-- The number of days it takes for Minjeong's and Soohyeok's piggy bank amounts to become equal -/
def days_to_equal_amount (minjeong_initial : ℕ) (soohyeok_initial : ℕ) 
                         (minjeong_daily : ℕ) (soohyeok_daily : ℕ) : ℕ :=
  15

theorem equal_amount_after_15_days 
  (minjeong_initial : ℕ) (soohyeok_initial : ℕ) 
  (minjeong_daily : ℕ) (soohyeok_daily : ℕ)
  (h1 : minjeong_initial = 8000)
  (h2 : soohyeok_initial = 5000)
  (h3 : minjeong_daily = 300)
  (h4 : soohyeok_daily = 500) :
  minjeong_initial + 15 * minjeong_daily = soohyeok_initial + 15 * soohyeok_daily :=
by
  sorry

#eval days_to_equal_amount 8000 5000 300 500

end NUMINAMATH_CALUDE_equal_amount_after_15_days_l2524_252452


namespace NUMINAMATH_CALUDE_fraction_of_puppies_sold_l2524_252436

/-- Proves that the fraction of puppies sold is 3/8 given the problem conditions --/
theorem fraction_of_puppies_sold (total_puppies : ℕ) (price_per_puppy : ℕ) (total_received : ℕ) :
  total_puppies = 20 →
  price_per_puppy = 200 →
  total_received = 3000 →
  (total_received / price_per_puppy : ℚ) / total_puppies = 3 / 8 := by
  sorry

#check fraction_of_puppies_sold

end NUMINAMATH_CALUDE_fraction_of_puppies_sold_l2524_252436


namespace NUMINAMATH_CALUDE_min_value_expression_l2524_252416

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : a * b = 2) :
  (a^2 + b^2 + 1) / (a - b) ≥ 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2524_252416


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_range_l2524_252406

/-- An isosceles triangle with perimeter 20 and base length x -/
structure IsoscelesTriangle where
  base : ℝ
  perimeter : ℝ
  is_isosceles : perimeter = 20
  base_definition : base > 0

/-- The range of possible base lengths for an isosceles triangle with perimeter 20 -/
theorem isosceles_triangle_base_range (t : IsoscelesTriangle) :
  5 < t.base ∧ t.base < 10 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_base_range_l2524_252406


namespace NUMINAMATH_CALUDE_sixteenth_occurrence_shift_l2524_252461

/-- Represents the number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- Calculates the sum of the first n even numbers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- Calculates the shift for the nth occurrence of a letter -/
def shift (n : ℕ) : ℕ := sum_even n % alphabet_size

/-- Theorem: The 16th occurrence of a letter is shifted by 16 places -/
theorem sixteenth_occurrence_shift :
  shift 16 = 16 := by sorry

end NUMINAMATH_CALUDE_sixteenth_occurrence_shift_l2524_252461


namespace NUMINAMATH_CALUDE_major_selection_ways_l2524_252407

-- Define the total number of majors
def total_majors : ℕ := 10

-- Define the number of majors to be chosen
def chosen_majors : ℕ := 3

-- Define the number of incompatible majors
def incompatible_majors : ℕ := 2

-- Theorem statement
theorem major_selection_ways :
  (total_majors.choose chosen_majors) - 
  ((total_majors - incompatible_majors).choose (chosen_majors - 1)) = 672 := by
  sorry

end NUMINAMATH_CALUDE_major_selection_ways_l2524_252407


namespace NUMINAMATH_CALUDE_sum_of_powers_of_two_l2524_252478

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 = 2^5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_two_l2524_252478


namespace NUMINAMATH_CALUDE_vector_magnitude_minimization_l2524_252490

/-- Given two unit vectors e₁ and e₂ with an angle of 60° between them,
    prove that |2e₁ + te₂| is minimized when t = -1 -/
theorem vector_magnitude_minimization (e₁ e₂ : ℝ × ℝ) :
  ‖e₁‖ = 1 →
  ‖e₂‖ = 1 →
  e₁ • e₂ = 1/2 →
  ∃ (t : ℝ), ∀ (s : ℝ), ‖2 • e₁ + t • e₂‖ ≤ ‖2 • e₁ + s • e₂‖ ∧ t = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_minimization_l2524_252490


namespace NUMINAMATH_CALUDE_difference_of_squares_l2524_252444

theorem difference_of_squares (m n : ℝ) : m^2 - 4*n^2 = (m + 2*n) * (m - 2*n) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2524_252444


namespace NUMINAMATH_CALUDE_inequality_proof_l2524_252427

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_eq_four : a + b + c + d = 4) :
  a * Real.sqrt (3*a + b + c) + b * Real.sqrt (3*b + c + d) + 
  c * Real.sqrt (3*c + d + a) + d * Real.sqrt (3*d + a + b) ≥ 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2524_252427


namespace NUMINAMATH_CALUDE_polygon_sides_l2524_252424

theorem polygon_sides (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2524_252424


namespace NUMINAMATH_CALUDE_nine_digit_divisibility_l2524_252413

theorem nine_digit_divisibility (A : ℕ) : 
  A < 10 →
  (457319808 * 10 + A) % 2 = 0 →
  (457319808 * 10 + A) % 5 = 0 →
  (457319808 * 10 + A) % 8 = 0 →
  (457319808 * 10 + A) % 10 = 0 →
  (457319808 * 10 + A) % 16 = 0 →
  A = 0 := by
sorry

end NUMINAMATH_CALUDE_nine_digit_divisibility_l2524_252413


namespace NUMINAMATH_CALUDE_student_assignment_l2524_252491

/-- The number of ways to assign n indistinguishable objects to k distinct containers,
    with each container receiving at least one object. -/
def assign_objects (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to assign 5 students to 3 towns -/
theorem student_assignment : assign_objects 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_student_assignment_l2524_252491


namespace NUMINAMATH_CALUDE_tan_alpha_minus_beta_l2524_252449

theorem tan_alpha_minus_beta (α β : ℝ) 
  (h1 : Real.tan (α + π / 5) = 2) 
  (h2 : Real.tan (β - 4 * π / 5) = -3) : 
  Real.tan (α - β) = -1 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_beta_l2524_252449


namespace NUMINAMATH_CALUDE_expression_bounds_l2524_252495

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) : 
  4 + 2 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
                        Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ∧
  Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
  Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l2524_252495


namespace NUMINAMATH_CALUDE_fraction_comparison_l2524_252439

theorem fraction_comparison : 
  (10 / 8 : ℚ) = 5 / 4 ∧ 
  (5 / 4 : ℚ) = 5 / 4 ∧ 
  (15 / 12 : ℚ) = 5 / 4 ∧ 
  (6 / 5 : ℚ) ≠ 5 / 4 ∧ 
  (50 / 40 : ℚ) = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2524_252439


namespace NUMINAMATH_CALUDE_unique_prime_pair_l2524_252472

theorem unique_prime_pair : ∃! (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p^3 - q^5 = (p + q)^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l2524_252472


namespace NUMINAMATH_CALUDE_perpendicular_planes_l2524_252448

-- Define the types for line and plane
variable (L : Type) [LinearOrder L]
variable (P : Type)

-- Define the relations
variable (perpendicular : L → P → Prop)
variable (contains : P → L → Prop)
variable (perp_planes : P → P → Prop)

-- State the theorem
theorem perpendicular_planes 
  (l : L) (α β : P) 
  (h1 : perpendicular l α) 
  (h2 : contains β l) : 
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l2524_252448


namespace NUMINAMATH_CALUDE_line_not_in_first_quadrant_l2524_252494

/-- Given that mx + 3 = 4 has the solution x = 1, prove that y = (m-2)x - 3 does not pass through the first quadrant -/
theorem line_not_in_first_quadrant (m : ℝ) (h : m * 1 + 3 = 4) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y ≠ (m - 2) * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_first_quadrant_l2524_252494


namespace NUMINAMATH_CALUDE_base_8_digits_of_2015_l2524_252428

-- Define the number we're working with
def n : ℕ := 2015

-- Define a function to calculate the number of digits in base 8
def num_digits_base_8 (x : ℕ) : ℕ :=
  if x = 0 then 1 else Nat.log 8 x + 1

-- Theorem statement
theorem base_8_digits_of_2015 : num_digits_base_8 n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_8_digits_of_2015_l2524_252428


namespace NUMINAMATH_CALUDE_intersection_implies_m_range_l2524_252458

theorem intersection_implies_m_range (m : ℝ) :
  (∃ x : ℝ, m * (4 : ℝ)^x - 3 * (2 : ℝ)^(x + 1) - 2 = 0) →
  m ≥ -9/2 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_m_range_l2524_252458


namespace NUMINAMATH_CALUDE_no_discount_on_backpacks_l2524_252485

/-- Proves that there is no discount on the backpacks given the problem conditions -/
theorem no_discount_on_backpacks 
  (num_backpacks : ℕ) 
  (monogram_cost : ℚ) 
  (total_cost : ℚ) 
  (h1 : num_backpacks = 5)
  (h2 : monogram_cost = 12)
  (h3 : total_cost = 140) :
  total_cost = num_backpacks * monogram_cost + (total_cost - num_backpacks * monogram_cost) := by
  sorry

#check no_discount_on_backpacks

end NUMINAMATH_CALUDE_no_discount_on_backpacks_l2524_252485


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2524_252425

theorem sufficient_not_necessary (x : ℝ) : 
  (x = 2 → (x - 2) * (x - 1) = 0) ∧ 
  ¬((x - 2) * (x - 1) = 0 → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2524_252425


namespace NUMINAMATH_CALUDE_jesses_room_carpet_area_l2524_252409

theorem jesses_room_carpet_area :
  let room_length : ℝ := 12
  let room_width : ℝ := 8
  let room_area := room_length * room_width
  room_area = 96 := by sorry

end NUMINAMATH_CALUDE_jesses_room_carpet_area_l2524_252409


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_is_one_tenth_l2524_252456

/-- A line passing through (10, 0) intersecting y = x^2 -/
structure IntersectingLine where
  /-- The slope of the line -/
  k : ℝ
  /-- The line passes through (10, 0) -/
  line_eq : ∀ x y : ℝ, y = k * (x - 10)
  /-- The line intersects y = x^2 at two distinct points -/
  intersects_parabola : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 = k * (x₁ - 10) ∧ x₂^2 = k * (x₂ - 10)

/-- The sum of reciprocals of intersection x-coordinates is 1/10 -/
theorem sum_of_reciprocals_is_one_tenth (L : IntersectingLine) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 = L.k * (x₁ - 10) ∧ 
    x₂^2 = L.k * (x₂ - 10) ∧
    1 / x₁ + 1 / x₂ = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_is_one_tenth_l2524_252456


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2524_252440

/-- Quadratic function f(x) = -2(x+1)^2 + k -/
def f (k : ℝ) (x : ℝ) : ℝ := -2 * (x + 1)^2 + k

/-- Theorem stating the relationship between f(x) values at x = 2, -3, and -0.5 -/
theorem quadratic_inequality (k : ℝ) : f k 2 < f k (-3) ∧ f k (-3) < f k (-0.5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2524_252440


namespace NUMINAMATH_CALUDE_sandy_carrots_l2524_252432

/-- Given that Sandy and Sam grew carrots together, with Sam growing 3 carrots
and a total of 9 carrots grown, prove that Sandy grew 6 carrots. -/
theorem sandy_carrots (total : ℕ) (sam : ℕ) (sandy : ℕ) 
  (h1 : total = 9)
  (h2 : sam = 3)
  (h3 : total = sam + sandy) :
  sandy = 6 := by
  sorry

end NUMINAMATH_CALUDE_sandy_carrots_l2524_252432


namespace NUMINAMATH_CALUDE_subtraction_equality_l2524_252471

theorem subtraction_equality : 25.52 - 3.248 - 1.004 = 21.268 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_equality_l2524_252471


namespace NUMINAMATH_CALUDE_system_solution_l2524_252482

theorem system_solution : 
  ∃! (x y : ℝ), x + y = 5 ∧ 2 * x + 5 * y = 28 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_l2524_252482


namespace NUMINAMATH_CALUDE_burger_combinations_l2524_252493

/-- The number of different toppings available. -/
def num_toppings : ℕ := 10

/-- The number of choices for meat patties. -/
def patty_choices : ℕ := 4

/-- Theorem stating the total number of burger combinations. -/
theorem burger_combinations :
  (2 ^ num_toppings) * patty_choices = 4096 :=
sorry

end NUMINAMATH_CALUDE_burger_combinations_l2524_252493


namespace NUMINAMATH_CALUDE_unique_solution_l2524_252473

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

/-- The main theorem stating the unique solution to the functional equation -/
theorem unique_solution :
  ∃! (α : ℝ) (f : ℝ → ℝ), SatisfiesEquation f α ∧ α = -1 ∧ ∀ x, f x = x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2524_252473


namespace NUMINAMATH_CALUDE_percentage_relation_l2524_252477

/-- Given that b as a percentage of x is equal to x as a percentage of (a + b),
    and this percentage is 61.80339887498949%, prove that a = b * (38.1966/61.8034) -/
theorem percentage_relation (a b x : ℝ) 
  (h1 : b / x = x / (a + b)) 
  (h2 : b / x = 61.80339887498949 / 100) : 
  a = b * (38.1966 / 61.8034) := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l2524_252477


namespace NUMINAMATH_CALUDE_remainder_problem_l2524_252476

theorem remainder_problem : ∃ x : ℕ, (71 * x) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2524_252476


namespace NUMINAMATH_CALUDE_number_of_persons_l2524_252483

theorem number_of_persons (total_amount : ℕ) (amount_per_person : ℕ) 
  (h1 : total_amount = 42900) 
  (h2 : amount_per_person = 1950) : 
  total_amount / amount_per_person = 22 := by
  sorry

end NUMINAMATH_CALUDE_number_of_persons_l2524_252483


namespace NUMINAMATH_CALUDE_class_size_proof_l2524_252462

/-- Represents the number of pupils in a class. -/
def num_pupils : ℕ := 56

/-- Represents the wrongly entered mark. -/
def wrong_mark : ℕ := 73

/-- Represents the correct mark. -/
def correct_mark : ℕ := 45

/-- Represents the increase in average marks due to the error. -/
def avg_increase : ℚ := 1/2

theorem class_size_proof :
  (wrong_mark - correct_mark : ℚ) / num_pupils = avg_increase :=
sorry

end NUMINAMATH_CALUDE_class_size_proof_l2524_252462


namespace NUMINAMATH_CALUDE_snow_probability_first_week_l2524_252470

def probability_of_snow (days : ℕ) (daily_prob : ℚ) : ℚ :=
  1 - (1 - daily_prob) ^ days

theorem snow_probability_first_week :
  let prob_first_four := probability_of_snow 4 (1/4)
  let prob_next_three := probability_of_snow 3 (1/3)
  let total_prob := 1 - (1 - prob_first_four) * (1 - prob_next_three)
  total_prob = 29/32 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_first_week_l2524_252470


namespace NUMINAMATH_CALUDE_sticker_cost_theorem_l2524_252465

def total_sticker_cost (allowance : ℕ) (card_cost : ℕ) (stickers_per_person : ℕ) : ℕ :=
  2 * allowance - card_cost

theorem sticker_cost_theorem (allowance : ℕ) (card_cost : ℕ) (stickers_per_person : ℕ)
  (h1 : allowance = 9)
  (h2 : card_cost = 10)
  (h3 : stickers_per_person = 2) :
  total_sticker_cost allowance card_cost stickers_per_person = 8 := by
sorry

#eval total_sticker_cost 9 10 2

end NUMINAMATH_CALUDE_sticker_cost_theorem_l2524_252465


namespace NUMINAMATH_CALUDE_train_crossing_lamppost_l2524_252430

/-- Calculates the time for a train to cross a lamp post given its length, bridge length, and time to cross the bridge -/
theorem train_crossing_lamppost 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (bridge_crossing_time : ℝ) 
  (h1 : train_length = 400)
  (h2 : bridge_length = 800)
  (h3 : bridge_crossing_time = 45)
  : (train_length * bridge_crossing_time) / bridge_length = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_lamppost_l2524_252430


namespace NUMINAMATH_CALUDE_area_bounded_by_function_and_double_tangent_l2524_252447

-- Define the function
def f (x : ℝ) : ℝ := -x^4 + 16*x^3 - 78*x^2 + 50*x - 2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -4*x^3 + 48*x^2 - 156*x + 50

-- Theorem statement
theorem area_bounded_by_function_and_double_tangent :
  ∃ (a b : ℝ),
    a < b ∧
    f' a = f' b ∧
    (f b - f a) / (b - a) = f' a ∧
    (∫ (x : ℝ) in a..b, (((f b - f a) / (b - a)) * (x - a) + f a) - f x) = 1296 / 5 :=
sorry

end NUMINAMATH_CALUDE_area_bounded_by_function_and_double_tangent_l2524_252447


namespace NUMINAMATH_CALUDE_complement_of_M_l2524_252457

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Finset Nat := {1, 2, 4}

-- Theorem statement
theorem complement_of_M : U \ M = {3, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l2524_252457


namespace NUMINAMATH_CALUDE_shaded_area_l2524_252484

/-- The area of the shaded region in a square with two non-overlapping rectangles --/
theorem shaded_area (total_area : ℝ) (rect1_area rect2_area : ℝ) :
  total_area = 16 →
  rect1_area = 6 →
  rect2_area = 2 →
  total_area - (rect1_area + rect2_area) = 8 := by
sorry


end NUMINAMATH_CALUDE_shaded_area_l2524_252484


namespace NUMINAMATH_CALUDE_log_sum_equality_l2524_252420

theorem log_sum_equality : 
  2 * Real.log 9 / Real.log 10 + 3 * Real.log 4 / Real.log 10 + 
  4 * Real.log 3 / Real.log 10 + 5 * Real.log 2 / Real.log 10 + 
  Real.log 16 / Real.log 10 = Real.log 215233856 / Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l2524_252420


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2524_252445

theorem quadratic_inequality_solution_set (a : ℝ) (h : a > 1) :
  let solution_set := {x : ℝ | (a - 1) * x^2 - a * x + 1 > 0}
  (a = 2 → solution_set = {x : ℝ | x ≠ 1}) ∧
  (1 < a ∧ a < 2 → solution_set = {x : ℝ | x < 1 ∨ x > 1 / (a - 1)}) ∧
  (a > 2 → solution_set = {x : ℝ | x < 1 / (a - 1) ∨ x > 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2524_252445


namespace NUMINAMATH_CALUDE_tangent_parallel_to_line_l2524_252429

-- Define the curve
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 6 * x + 2

-- Define the tangent line
def tangent_line (a : ℝ) (x y : ℝ) : Prop := 2 * a * x - y - 6 = 0

-- Theorem statement
theorem tangent_parallel_to_line (a : ℝ) : 
  (f 1 = 5) → -- The point (1, 5) is on the curve
  (tangent_line a 1 5) → -- The tangent line passes through (1, 5)
  (f' 1 = 2 * a) → -- The slope of the tangent line equals the derivative at x = 1
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_line_l2524_252429


namespace NUMINAMATH_CALUDE_triangle_problem_l2524_252418

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a = 3 →
  Real.cos B = 7/9 →
  a * c * Real.cos B = 7 →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) →
  b = 2 ∧ Real.sin (A - B) = 10 * Real.sqrt 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2524_252418


namespace NUMINAMATH_CALUDE_max_segment_length_through_centroid_l2524_252442

/-- Given a triangle ABC with vertex A at (0,0), B at (b, 0), and C at (c_x, c_y),
    the maximum length of a line segment starting from A and passing through the centroid
    is equal to the distance between A and the centroid. -/
theorem max_segment_length_through_centroid (b c_x c_y : ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (b, 0)
  let C : ℝ × ℝ := (c_x, c_y)
  let centroid : ℝ × ℝ := ((b + c_x) / 3, c_y / 3)
  let max_length := Real.sqrt (((b + c_x) / 3)^2 + (c_y / 3)^2)
  ∃ (segment : ℝ × ℝ → ℝ × ℝ),
    (segment 0 = A) ∧
    (∃ t, segment t = centroid) ∧
    (∀ t, ‖segment t - A‖ ≤ max_length) ∧
    (∃ t, ‖segment t - A‖ = max_length) :=
by sorry


end NUMINAMATH_CALUDE_max_segment_length_through_centroid_l2524_252442


namespace NUMINAMATH_CALUDE_expression_evaluation_l2524_252437

theorem expression_evaluation : (4 + 6 + 7) * 2 - 2 + 3 / 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2524_252437


namespace NUMINAMATH_CALUDE_sin_plus_cos_for_point_l2524_252419

/-- Given that the terminal side of angle θ passes through point P(-3,4),
    prove that sin θ + cos θ = 1/5 -/
theorem sin_plus_cos_for_point (θ : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos θ = -3 ∧ r * Real.sin θ = 4) →
  Real.sin θ + Real.cos θ = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_for_point_l2524_252419


namespace NUMINAMATH_CALUDE_kids_at_camp_l2524_252422

theorem kids_at_camp (total : ℕ) (stay_home : ℕ) (h1 : total = 898051) (h2 : stay_home = 268627) :
  total - stay_home = 629424 := by
  sorry

end NUMINAMATH_CALUDE_kids_at_camp_l2524_252422


namespace NUMINAMATH_CALUDE_complement_of_A_l2524_252469

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ 2}

-- State the theorem
theorem complement_of_A : Set.compl A = {x : ℝ | x < 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l2524_252469


namespace NUMINAMATH_CALUDE_sum_of_favorite_numbers_l2524_252404

def glorys_number : ℕ := 450

def mistys_number : ℕ := glorys_number / 3

theorem sum_of_favorite_numbers : glorys_number + mistys_number = 600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_favorite_numbers_l2524_252404


namespace NUMINAMATH_CALUDE_digit_equality_proof_l2524_252492

theorem digit_equality_proof (n : ℕ) (a : ℕ) (k : ℕ) :
  (n ≥ 4) →
  (a ≤ 9) →
  (k ≥ 1) →
  (∃ n k, n * (n + 1) / 2 = (10^k - 1) * a / 9) ↔ (a = 5 ∨ a = 6) :=
by sorry

end NUMINAMATH_CALUDE_digit_equality_proof_l2524_252492


namespace NUMINAMATH_CALUDE_min_value_theorem_l2524_252410

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b = 1 →
  (∀ x y : ℝ, 0 < x → x < 2 → y = 1 + Real.sin (π * x) → a * x + b * y = 1) →
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2524_252410


namespace NUMINAMATH_CALUDE_trapezoid_semicircle_area_l2524_252438

-- Define the trapezoid
def trapezoid : Set (ℝ × ℝ) :=
  {p | p = (5, 11) ∨ p = (16, 11) ∨ p = (16, -2) ∨ p = (5, -2)}

-- Define the semicircle
def semicircle : Set (ℝ × ℝ) :=
  {p | (p.1 - 10.5)^2 + (p.2 + 2)^2 ≤ 5.5^2 ∧ p.2 ≤ -2}

-- Define the area to be calculated
def bounded_area : ℝ := sorry

-- Theorem statement
theorem trapezoid_semicircle_area :
  bounded_area = 15.125 * Real.pi := by sorry

end NUMINAMATH_CALUDE_trapezoid_semicircle_area_l2524_252438


namespace NUMINAMATH_CALUDE_angle_value_proof_l2524_252474

theorem angle_value_proof (α : Real) (P : Real × Real) : 
  0 < α → α < Real.pi / 2 →
  P.1 = Real.sin (-50 * Real.pi / 180) →
  P.2 = Real.cos (130 * Real.pi / 180) →
  P ∈ {p : Real × Real | p.1 = Real.sin (5 * α) ∧ p.2 = Real.cos (5 * α)} →
  α = 44 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_value_proof_l2524_252474


namespace NUMINAMATH_CALUDE_banana_storage_l2524_252401

/-- The number of boxes needed to store bananas -/
def number_of_boxes (total_bananas : ℕ) (bananas_per_box : ℕ) : ℕ :=
  total_bananas / bananas_per_box

/-- Proof that 8 boxes are needed to store 40 bananas with 5 bananas per box -/
theorem banana_storage : number_of_boxes 40 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_banana_storage_l2524_252401


namespace NUMINAMATH_CALUDE_g_evaluation_l2524_252487

def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 7

theorem g_evaluation : 3 * g 2 + 4 * g (-2) = 113 := by
  sorry

end NUMINAMATH_CALUDE_g_evaluation_l2524_252487


namespace NUMINAMATH_CALUDE_digit_at_position_l2524_252499

/-- The fraction we're examining -/
def f : ℚ := 17 / 270

/-- The length of the repeating sequence in the decimal representation of f -/
def period : ℕ := 3

/-- The repeating sequence in the decimal representation of f -/
def repeating_sequence : List ℕ := [6, 2, 9]

/-- The position we're interested in -/
def target_position : ℕ := 145

theorem digit_at_position :
  (target_position - 1) % period = 0 →
  List.get! repeating_sequence (period - 1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_at_position_l2524_252499


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2524_252498

def A : Set ℕ := {2, 3, 5, 7}
def B : Set ℕ := {1, 2, 3, 5, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2524_252498
