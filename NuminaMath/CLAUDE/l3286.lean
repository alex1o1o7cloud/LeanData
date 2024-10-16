import Mathlib

namespace NUMINAMATH_CALUDE_system_solutions_l3286_328679

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  (x + y)^4 = 6*x^2*y^2 - 215 ∧ x*y*(x^2 + y^2) = -78

/-- The set of solutions -/
def solutions : Set (ℝ × ℝ) :=
  {(3, -2), (-2, 3), (-3, 2), (2, -3)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ (x y : ℝ), system x y ↔ (x, y) ∈ solutions := by sorry

end NUMINAMATH_CALUDE_system_solutions_l3286_328679


namespace NUMINAMATH_CALUDE_sum_of_digits_of_9n_l3286_328628

/-- A function that checks if each digit of a natural number is strictly greater than the digit to its left -/
def is_strictly_increasing_digits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n / 10^i) % 10 < (n / 10^j) % 10

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

/-- Theorem stating that for any natural number with strictly increasing digits,
    the sum of digits of 9 times that number is always 9 -/
theorem sum_of_digits_of_9n (N : ℕ) (h : is_strictly_increasing_digits N) :
  sum_of_digits (9 * N) = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_9n_l3286_328628


namespace NUMINAMATH_CALUDE_betty_doug_age_sum_l3286_328666

/-- Given the conditions about Betty's age and the cost of nuts, prove that the sum of Betty's and Doug's age is 90 years. -/
theorem betty_doug_age_sum :
  ∀ (betty_age : ℕ) (pack_cost : ℕ),
    (2 * betty_age = pack_cost) →  -- Twice Betty's age is the cost of a pack of nuts
    (20 * pack_cost = 2000) →      -- Betty pays $2000 for 20 packs
    (betty_age + 40 = 90)          -- The sum of Betty's and Doug's (40) age is 90
    := by sorry

end NUMINAMATH_CALUDE_betty_doug_age_sum_l3286_328666


namespace NUMINAMATH_CALUDE_magic_king_episodes_l3286_328636

/-- Calculates the total number of episodes for a TV show with the given parameters -/
def total_episodes (total_seasons : ℕ) (episodes_first_half : ℕ) (episodes_second_half : ℕ) : ℕ :=
  let half_seasons := total_seasons / 2
  half_seasons * episodes_first_half + half_seasons * episodes_second_half

/-- Proves that the TV show Magic King has 225 episodes in total -/
theorem magic_king_episodes : 
  total_episodes 10 20 25 = 225 := by
  sorry

end NUMINAMATH_CALUDE_magic_king_episodes_l3286_328636


namespace NUMINAMATH_CALUDE_graduating_class_boys_count_l3286_328699

theorem graduating_class_boys_count (total : ℕ) (difference : ℕ) (boys : ℕ) : 
  total = 466 → difference = 212 → boys + (boys + difference) = total → boys = 127 := by
  sorry

end NUMINAMATH_CALUDE_graduating_class_boys_count_l3286_328699


namespace NUMINAMATH_CALUDE_find_constant_b_l3286_328630

theorem find_constant_b (a b c : ℝ) : 
  (∀ x : ℝ, (3*x^2 - 4*x + 2)*(a*x^2 + b*x + c) = 9*x^4 - 10*x^3 + 5*x^2 - 8*x + 4) → 
  b = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_find_constant_b_l3286_328630


namespace NUMINAMATH_CALUDE_joes_lift_ratio_l3286_328682

/-- Joe's weight-lifting competition results -/
def JoesLifts (first second : ℕ) : Prop :=
  first + second = 600 ∧ first = 300 ∧ 2 * first = second + 300

theorem joes_lift_ratio :
  ∀ first second : ℕ, JoesLifts first second → first = second :=
by
  sorry

end NUMINAMATH_CALUDE_joes_lift_ratio_l3286_328682


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3286_328686

theorem sin_alpha_value (α : Real) :
  let point : Real × Real := (2 * Real.sin (60 * π / 180), -2 * Real.cos (60 * π / 180))
  (∃ k : Real, k > 0 ∧ k * point.1 = Real.cos α ∧ k * point.2 = Real.sin α) →
  Real.sin α = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3286_328686


namespace NUMINAMATH_CALUDE_sum_21_terms_arithmetic_sequence_l3286_328609

/-- Arithmetic sequence with first term 3 and common difference 10 -/
def arithmeticSequence (n : ℕ) : ℤ :=
  3 + (n - 1) * 10

/-- Sum of the first n terms of the arithmetic sequence -/
def sumArithmeticSequence (n : ℕ) : ℤ :=
  n * (3 + arithmeticSequence n) / 2

theorem sum_21_terms_arithmetic_sequence :
  sumArithmeticSequence 21 = 2163 := by
  sorry

end NUMINAMATH_CALUDE_sum_21_terms_arithmetic_sequence_l3286_328609


namespace NUMINAMATH_CALUDE_inequality_range_l3286_328672

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3286_328672


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3286_328656

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3286_328656


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3286_328622

theorem min_value_of_exponential_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  ∀ z, 3^x + 9^y ≥ z → z ≤ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3286_328622


namespace NUMINAMATH_CALUDE_horner_method_correctness_horner_poly_at_5_l3286_328698

def horner_poly (x : ℝ) : ℝ := (((((3*x - 4)*x + 6)*x - 2)*x - 5)*x - 2)

def original_poly (x : ℝ) : ℝ := 3*x^5 - 4*x^4 + 6*x^3 - 2*x^2 - 5*x - 2

theorem horner_method_correctness :
  ∀ x : ℝ, horner_poly x = original_poly x :=
sorry

theorem horner_poly_at_5 : horner_poly 5 = 7548 :=
sorry

end NUMINAMATH_CALUDE_horner_method_correctness_horner_poly_at_5_l3286_328698


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3286_328652

theorem right_triangle_inequality (a b c : ℝ) (n : ℕ) 
  (h_right_triangle : a^2 = b^2 + c^2)
  (h_order : a > b ∧ b > c)
  (h_n : n > 2) : 
  a^n > b^n + c^n := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l3286_328652


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l3286_328693

theorem rectangle_width_decrease (L W : ℝ) (L_new W_new A_new : ℝ) 
  (h1 : L_new = 1.6 * L)
  (h2 : A_new = 1.36 * (L * W))
  (h3 : A_new = L_new * W_new) :
  W_new = 0.85 * W := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l3286_328693


namespace NUMINAMATH_CALUDE_first_player_wins_l3286_328665

/-- Represents the game state -/
structure GameState where
  stones : ℕ
  last_move : ℕ

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (move : ℕ) : Prop :=
  move > 0 ∧ move ≤ state.stones ∧
  (state.last_move = 0 ∨ state.last_move % move = 0)

/-- Defines the winning condition -/
def is_winning_state (state : GameState) : Prop :=
  state.stones = 0

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  ∃ (first_move : ℕ),
    valid_move { stones := 1992, last_move := 0 } first_move ∧
    ∀ (second_move : ℕ),
      valid_move { stones := 1992 - first_move, last_move := first_move } second_move →
      ∃ (strategy : GameState → ℕ),
        (∀ (state : GameState),
          valid_move state (strategy state)) ∧
        (∀ (state : GameState),
          ¬is_winning_state state →
          is_winning_state { stones := state.stones - strategy state, last_move := strategy state }) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l3286_328665


namespace NUMINAMATH_CALUDE_ab_length_l3286_328654

-- Define the triangles
structure Triangle :=
  (a b c : ℝ)

-- Define similarity relation
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the triangles ABC and DEF
def ABC : Triangle := { a := 7, b := 14, c := 10 }
def DEF : Triangle := { a := 6, b := 3, c := 5 }

-- State the theorem
theorem ab_length :
  similar ABC DEF →
  ABC.b = 14 →
  DEF.a = 6 →
  DEF.b = 3 →
  ABC.a = 7 := by sorry

end NUMINAMATH_CALUDE_ab_length_l3286_328654


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l3286_328655

theorem unfair_coin_probability (n : ℕ) (k : ℕ) (p_head : ℚ) (p_tail : ℚ) :
  n = 8 →
  k = 3 →
  p_head = 1/3 →
  p_tail = 2/3 →
  p_head + p_tail = 1 →
  (n.choose k : ℚ) * p_tail^k * p_head^(n-k) = 448/177147 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l3286_328655


namespace NUMINAMATH_CALUDE_bob_water_usage_percentage_l3286_328673

-- Define the farmers
inductive Farmer
| Bob
| Brenda
| Bernie

-- Define the crop types
inductive Crop
| Corn
| Cotton
| Beans

-- Define the acreage for each farmer and crop
def acreage : Farmer → Crop → ℕ
  | Farmer.Bob, Crop.Corn => 3
  | Farmer.Bob, Crop.Cotton => 9
  | Farmer.Bob, Crop.Beans => 12
  | Farmer.Brenda, Crop.Corn => 6
  | Farmer.Brenda, Crop.Cotton => 7
  | Farmer.Brenda, Crop.Beans => 14
  | Farmer.Bernie, Crop.Corn => 2
  | Farmer.Bernie, Crop.Cotton => 12
  | Farmer.Bernie, Crop.Beans => 0

-- Define water requirements for each crop (in gallons per acre)
def waterPerAcre : Crop → ℕ
  | Crop.Corn => 20
  | Crop.Cotton => 80
  | Crop.Beans => 40  -- Twice as much as corn

-- Calculate total water used by a farmer
def farmerWaterUsage (f : Farmer) : ℕ :=
  (acreage f Crop.Corn * waterPerAcre Crop.Corn) +
  (acreage f Crop.Cotton * waterPerAcre Crop.Cotton) +
  (acreage f Crop.Beans * waterPerAcre Crop.Beans)

-- Calculate total water used by all farmers
def totalWaterUsage : ℕ :=
  farmerWaterUsage Farmer.Bob +
  farmerWaterUsage Farmer.Brenda +
  farmerWaterUsage Farmer.Bernie

-- Theorem: Bob's water usage is 36% of total water usage
theorem bob_water_usage_percentage :
  (farmerWaterUsage Farmer.Bob : ℚ) / totalWaterUsage * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bob_water_usage_percentage_l3286_328673


namespace NUMINAMATH_CALUDE_brothers_baskets_count_l3286_328660

/-- Represents the number of strawberries in each basket picked by Kimberly's brother -/
def strawberries_per_basket : ℕ := 15

/-- Represents the number of people sharing the strawberries -/
def number_of_people : ℕ := 4

/-- Represents the number of strawberries each person gets when divided equally -/
def strawberries_per_person : ℕ := 168

/-- Represents the number of baskets Kimberly's brother picked -/
def brothers_baskets : ℕ := 3

theorem brothers_baskets_count :
  ∃ (b : ℕ),
    b = brothers_baskets ∧
    (17 * b * strawberries_per_basket - 93 = number_of_people * strawberries_per_person) :=
by sorry

end NUMINAMATH_CALUDE_brothers_baskets_count_l3286_328660


namespace NUMINAMATH_CALUDE_total_earnings_proof_l3286_328615

def total_earnings (jermaine_earnings terrence_earnings emilee_earnings : ℕ) : ℕ :=
  jermaine_earnings + terrence_earnings + emilee_earnings

theorem total_earnings_proof (terrence_earnings emilee_earnings : ℕ) 
  (h1 : terrence_earnings = 30)
  (h2 : emilee_earnings = 25) :
  total_earnings (terrence_earnings + 5) terrence_earnings emilee_earnings = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_total_earnings_proof_l3286_328615


namespace NUMINAMATH_CALUDE_jon_points_l3286_328644

theorem jon_points (jon jack tom : ℕ) : 
  (jack = jon + 5) →
  (tom = jon + jack - 4) →
  (jon + jack + tom = 18) →
  (jon = 3) := by
sorry

end NUMINAMATH_CALUDE_jon_points_l3286_328644


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3286_328662

theorem geometric_sequence_sum (a : ℕ → ℚ) :
  a 0 = 4096 ∧ a 1 = 1024 ∧ a 2 = 256 ∧
  a 5 = 4 ∧ a 6 = 1 ∧ a 7 = (1/4 : ℚ) ∧
  (∀ n : ℕ, a (n + 1) = a n * (1/4 : ℚ)) →
  a 3 + a 4 = 80 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3286_328662


namespace NUMINAMATH_CALUDE_power_division_l3286_328637

theorem power_division (n : ℕ) : (16^3018) / 8 = 2^9032 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l3286_328637


namespace NUMINAMATH_CALUDE_soap_cost_l3286_328648

/-- The cost of a bar of soap given monthly usage and two-year expenditure -/
theorem soap_cost (monthly_usage : ℕ) (two_year_expenditure : ℚ) :
  monthly_usage = 1 →
  two_year_expenditure = 96 →
  two_year_expenditure / (24 : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_soap_cost_l3286_328648


namespace NUMINAMATH_CALUDE_power_function_through_point_l3286_328688

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  f 2 = 4 →
  f 9 = 81 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3286_328688


namespace NUMINAMATH_CALUDE_train_length_l3286_328618

/-- The length of a train given its speed and time to pass a pole -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 72) (h2 : time = 8) :
  speed * (1000 / 3600) * time = 160 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3286_328618


namespace NUMINAMATH_CALUDE_street_painting_cost_l3286_328659

/-- Calculates the total cost for painting house numbers on a street --/
def total_painting_cost (south_start : ℕ) (north_start : ℕ) (common_diff : ℕ) (houses_per_side : ℕ) : ℚ :=
  let south_end := south_start + common_diff * (houses_per_side - 1)
  let north_end := north_start + common_diff * (houses_per_side - 1)
  let south_two_digit := min houses_per_side (((99 - south_start) / common_diff) + 1)
  let north_two_digit := min houses_per_side (((99 - north_start) / common_diff) + 1)
  let south_three_digit := houses_per_side - south_two_digit
  let north_three_digit := houses_per_side - north_two_digit
  (2 * south_two_digit + 1.5 * south_three_digit + 2 * north_two_digit + 1.5 * north_three_digit : ℚ)

/-- The theorem stating the total cost for the given street configuration --/
theorem street_painting_cost :
  total_painting_cost 5 2 7 25 = 88.5 := by
  sorry

end NUMINAMATH_CALUDE_street_painting_cost_l3286_328659


namespace NUMINAMATH_CALUDE_magnitude_of_b_magnitude_of_c_and_area_l3286_328670

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 15 ∧ Real.sin t.A = 1/4

-- Theorem 1
theorem magnitude_of_b (t : Triangle) (h : triangle_conditions t) 
  (hcosB : Real.cos t.B = Real.sqrt 5 / 3) :
  t.b = 8 * Real.sqrt 15 / 3 :=
sorry

-- Theorem 2
theorem magnitude_of_c_and_area (t : Triangle) (h : triangle_conditions t) 
  (hb : t.b = 4 * t.a) :
  t.c = 15 ∧ (1/2 * t.b * t.c * Real.sin t.A = 15/2 * Real.sqrt 15) :=
sorry

end NUMINAMATH_CALUDE_magnitude_of_b_magnitude_of_c_and_area_l3286_328670


namespace NUMINAMATH_CALUDE_sum_of_flipped_digits_is_19_l3286_328696

/-- Function to flip a digit upside down -/
def flip_digit (d : ℕ) : ℕ := sorry

/-- Function to flip a number upside down -/
def flip_number (n : ℕ) : ℕ := sorry

/-- Function to sum the digits of a number -/
def sum_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of flipped digits is 19 -/
theorem sum_of_flipped_digits_is_19 :
  sum_digits (flip_number 340) +
  sum_digits (flip_number 24813) +
  sum_digits (flip_number 43323414) = 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_flipped_digits_is_19_l3286_328696


namespace NUMINAMATH_CALUDE_geq_one_necessary_not_sufficient_for_gt_one_l3286_328664

theorem geq_one_necessary_not_sufficient_for_gt_one :
  (∀ x : ℝ, x > 1 → x ≥ 1) ∧
  (∃ x : ℝ, x ≥ 1 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_geq_one_necessary_not_sufficient_for_gt_one_l3286_328664


namespace NUMINAMATH_CALUDE_matrix_cube_eq_matrix_plus_identity_det_positive_l3286_328625

open Matrix

theorem matrix_cube_eq_matrix_plus_identity_det_positive :
  ∀ (n : ℕ), ∃ (A : Matrix (Fin n) (Fin n) ℝ), A ^ 3 = A + 1 →
  ∀ (A : Matrix (Fin n) (Fin n) ℝ), A ^ 3 = A + 1 → 0 < det A :=
by sorry

end NUMINAMATH_CALUDE_matrix_cube_eq_matrix_plus_identity_det_positive_l3286_328625


namespace NUMINAMATH_CALUDE_three_digit_reverse_difference_theorem_l3286_328669

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_do_not_repeat (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

def reverse_number (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

def same_digits (m n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (m = 100 * a + 10 * b + c ∨ m = 100 * a + 10 * c + b ∨
     m = 100 * b + 10 * a + c ∨ m = 100 * b + 10 * c + a ∨
     m = 100 * c + 10 * a + b ∨ m = 100 * c + 10 * b + a) ∧
    (n = 100 * a + 10 * b + c ∨ n = 100 * a + 10 * c + b ∨
     n = 100 * b + 10 * a + c ∨ n = 100 * b + 10 * c + a ∨
     n = 100 * c + 10 * a + b ∨ n = 100 * c + 10 * b + a)

theorem three_digit_reverse_difference_theorem :
  ∀ x : ℕ,
    is_three_digit x ∧
    digits_do_not_repeat x ∧
    is_three_digit (x - reverse_number x) ∧
    same_digits x (x - reverse_number x) →
    x = 954 ∨ x = 459 := by
  sorry


end NUMINAMATH_CALUDE_three_digit_reverse_difference_theorem_l3286_328669


namespace NUMINAMATH_CALUDE_circles_intersection_sum_l3286_328612

/-- Given two circles intersecting at points (1, 3) and (m, 1), with their centers 
    on the line x - y + c/2 = 0, prove that m + c = 3 -/
theorem circles_intersection_sum (m c : ℝ) : 
  (∃ (circle1 circle2 : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ circle1 ∩ circle2 ↔ ((x = 1 ∧ y = 3) ∨ (x = m ∧ y = 1))) ∧
    (∃ (x1 y1 x2 y2 : ℝ), 
      (x1, y1) ∈ circle1 ∧ (x2, y2) ∈ circle2 ∧
      x1 - y1 + c/2 = 0 ∧ x2 - y2 + c/2 = 0)) →
  m + c = 3 := by
sorry

end NUMINAMATH_CALUDE_circles_intersection_sum_l3286_328612


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l3286_328640

def arithmetic_sequence (n : ℕ) : ℝ := 2 * n - 1

def kth_order_derivative_sequence (k m n : ℕ) : ℝ :=
  2^(k+2) * m - 2^(k+2) + 1

theorem geometric_sequence_proof (m : ℕ) (hm : m ≥ 2) :
  ∀ n : ℕ, n ≥ 1 → 
    (kth_order_derivative_sequence n m (n+1) - 1) / (kth_order_derivative_sequence n m n - 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l3286_328640


namespace NUMINAMATH_CALUDE_remainder_when_n_plus_3_and_n_plus_7_prime_l3286_328633

theorem remainder_when_n_plus_3_and_n_plus_7_prime (n : ℕ) 
  (h1 : Nat.Prime (n + 3)) 
  (h2 : Nat.Prime (n + 7)) : 
  n % 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_when_n_plus_3_and_n_plus_7_prime_l3286_328633


namespace NUMINAMATH_CALUDE_sum_and_count_integers_l3286_328695

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_integers : sum_integers 60 80 + count_even_integers 60 80 = 1481 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_integers_l3286_328695


namespace NUMINAMATH_CALUDE_jackson_flight_distance_l3286_328687

theorem jackson_flight_distance (beka_miles jackson_miles : ℕ) 
  (h1 : beka_miles = 873)
  (h2 : beka_miles = jackson_miles + 310) : 
  jackson_miles = 563 := by
sorry

end NUMINAMATH_CALUDE_jackson_flight_distance_l3286_328687


namespace NUMINAMATH_CALUDE_expression_simplification_l3286_328661

theorem expression_simplification (x : ℝ) (h : x^2 - x - 1 = 0) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2 * x^2 - x) / (x^2 + 2 * x + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3286_328661


namespace NUMINAMATH_CALUDE_max_a_cubic_function_l3286_328626

/-- Given a cubic function f(x) = a x^3 + b x^2 + c x + d where a ≠ 0,
    and |f'(x)| ≤ 1 for 0 ≤ x ≤ 1, the maximum value of a is 8/3. -/
theorem max_a_cubic_function (a b c d : ℝ) (h₁ : a ≠ 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |3 * a * x^2 + 2 * b * x + c| ≤ 1) →
  a ≤ 8/3 ∧ ∃ b c : ℝ, a = 8/3 ∧ 
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |3 * (8/3) * x^2 + 2 * b * x + c| ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_a_cubic_function_l3286_328626


namespace NUMINAMATH_CALUDE_original_polygon_sides_l3286_328632

theorem original_polygon_sides (n : ℕ) : 
  (∃ m : ℕ, (m - 2) * 180 = 1620 ∧ 
  (n = m + 1 ∨ n = m ∨ n = m - 1)) →
  (n = 10 ∨ n = 11 ∨ n = 12) :=
by sorry

end NUMINAMATH_CALUDE_original_polygon_sides_l3286_328632


namespace NUMINAMATH_CALUDE_jenny_recycling_l3286_328613

/-- Represents the recycling problem with given weights and prices -/
structure RecyclingProblem where
  bottle_weight : Nat
  can_weight : Nat
  jar_weight : Nat
  max_weight : Nat
  cans_collected : Nat
  bottle_price : Nat
  can_price : Nat
  jar_price : Nat

/-- Calculates the number of jars that can be carried given the remaining weight -/
def max_jars (p : RecyclingProblem) (remaining_weight : Nat) : Nat :=
  remaining_weight / p.jar_weight

/-- Calculates the total money earned from recycling -/
def total_money (p : RecyclingProblem) (cans : Nat) (jars : Nat) (bottles : Nat) : Nat :=
  cans * p.can_price + jars * p.jar_price + bottles * p.bottle_price

/-- States the theorem about Jenny's recycling problem -/
theorem jenny_recycling (p : RecyclingProblem) 
  (h1 : p.bottle_weight = 6)
  (h2 : p.can_weight = 2)
  (h3 : p.jar_weight = 8)
  (h4 : p.max_weight = 100)
  (h5 : p.cans_collected = 20)
  (h6 : p.bottle_price = 10)
  (h7 : p.can_price = 3)
  (h8 : p.jar_price = 12) :
  let remaining_weight := p.max_weight - (p.cans_collected * p.can_weight)
  let jars := max_jars p remaining_weight
  let bottles := 0
  (cans, jars, bottles) = (20, 7, 0) ∧ 
  total_money p p.cans_collected jars bottles = 144 := by
  sorry

end NUMINAMATH_CALUDE_jenny_recycling_l3286_328613


namespace NUMINAMATH_CALUDE_points_in_quadrant_I_l3286_328667

-- Define the set of points satisfying the given inequalities
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≥ 3 * p.1 ∧ p.2 ≥ 5 - p.1 ∧ p.2 < 7}

-- Define Quadrant I
def QuadrantI : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

-- Theorem statement
theorem points_in_quadrant_I : S ⊆ QuadrantI := by
  sorry

end NUMINAMATH_CALUDE_points_in_quadrant_I_l3286_328667


namespace NUMINAMATH_CALUDE_quadratic_equations_integer_roots_l3286_328642

theorem quadratic_equations_integer_roots :
  ∃ (a b c : ℕ),
    (∃ (x y : ℤ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ∧
    (∃ (x y : ℤ), x ≠ y ∧ a * x^2 + b * x - c = 0 ∧ a * y^2 + b * y - c = 0) ∧
    (∃ (x y : ℤ), x ≠ y ∧ a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) ∧
    (∃ (x y : ℤ), x ≠ y ∧ a * x^2 - b * x - c = 0 ∧ a * y^2 - b * y - c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_integer_roots_l3286_328642


namespace NUMINAMATH_CALUDE_book_profit_percentage_l3286_328629

/-- Calculates the profit percentage given purchase and selling prices in different currencies and their conversion rates to a common currency. -/
def profit_percentage (purchase_price_A : ℚ) (selling_price_B : ℚ) (rate_A_to_C : ℚ) (rate_B_to_C : ℚ) : ℚ :=
  let purchase_price_C := purchase_price_A * rate_A_to_C
  let selling_price_C := selling_price_B * rate_B_to_C
  let profit_C := selling_price_C - purchase_price_C
  (profit_C / purchase_price_C) * 100

/-- Theorem stating that under the given conditions, the profit percentage is 700/3%. -/
theorem book_profit_percentage :
  profit_percentage 50 100 (3/4) (5/4) = 700/3 := by
  sorry

end NUMINAMATH_CALUDE_book_profit_percentage_l3286_328629


namespace NUMINAMATH_CALUDE_alternate_arrangements_count_l3286_328690

/-- The number of ways to arrange two men and two women alternately in a row -/
def alternateArrangements : ℕ :=
  let menCount := 2
  let womenCount := 2
  let manFirstArrangements := menCount * womenCount
  let womanFirstArrangements := womenCount * menCount
  manFirstArrangements + womanFirstArrangements

theorem alternate_arrangements_count :
  alternateArrangements = 8 := by
  sorry

end NUMINAMATH_CALUDE_alternate_arrangements_count_l3286_328690


namespace NUMINAMATH_CALUDE_masons_father_age_l3286_328620

theorem masons_father_age :
  ∀ (mason_age sydney_age father_age : ℕ),
    mason_age = 20 →
    sydney_age = 3 * mason_age →
    father_age = sydney_age + 6 →
    father_age = 66 := by
  sorry

end NUMINAMATH_CALUDE_masons_father_age_l3286_328620


namespace NUMINAMATH_CALUDE_general_solution_second_order_recurrence_l3286_328651

/-- Second-order linear recurrence sequence -/
def RecurrenceSequence (a b : ℝ) (u : ℕ → ℝ) : Prop :=
  ∀ n, u (n + 2) = a * u (n + 1) + b * u n

/-- Characteristic polynomial of the recurrence sequence -/
def CharacteristicPolynomial (a b : ℝ) (X : ℝ) : ℝ :=
  X^2 - a*X - b

theorem general_solution_second_order_recurrence
  (a b : ℝ) (u : ℕ → ℝ) (r₁ r₂ : ℝ) :
  RecurrenceSequence a b u →
  r₁ ≠ r₂ →
  CharacteristicPolynomial a b r₁ = 0 →
  CharacteristicPolynomial a b r₂ = 0 →
  ∃ c d : ℝ, ∀ n, u n = c * r₁^n + d * r₂^n ∧
    c = (u 1 - u 0 * r₂) / (r₁ - r₂) ∧
    d = (u 0 * r₁ - u 1) / (r₁ - r₂) :=
sorry

end NUMINAMATH_CALUDE_general_solution_second_order_recurrence_l3286_328651


namespace NUMINAMATH_CALUDE_car_speed_theorem_l3286_328653

/-- Calculates the speed of a car in miles per hour -/
def car_speed (distance_yards : ℚ) (time_seconds : ℚ) (yards_per_mile : ℚ) : ℚ :=
  (distance_yards / yards_per_mile) * (3600 / time_seconds)

/-- Theorem stating that a car traveling 22 yards in 0.5 seconds has a speed of 90 miles per hour -/
theorem car_speed_theorem :
  car_speed 22 0.5 1760 = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_theorem_l3286_328653


namespace NUMINAMATH_CALUDE_ticket123123123_is_red_l3286_328639

-- Define the color type
inductive Color
| Red
| Green
| Blue

-- Define a ticket as a 9-digit number and a color
structure Ticket :=
  (number : Fin 9 → Fin 3)
  (color : Color)

-- Function to check if two tickets have no matching digits
def noMatchingDigits (t1 t2 : Ticket) : Prop :=
  ∀ i : Fin 9, t1.number i ≠ t2.number i

-- Define the given conditions
axiom different_colors (t1 t2 : Ticket) :
  noMatchingDigits t1 t2 → t1.color ≠ t2.color

-- Define the specific tickets mentioned in the problem
def ticket122222222 : Ticket :=
  { number := λ i => if i = 0 then 0 else 1,
    color := Color.Red }

def ticket222222222 : Ticket :=
  { number := λ _ => 1,
    color := Color.Green }

def ticket123123123 : Ticket :=
  { number := λ i => i % 3,
    color := Color.Red }  -- We'll prove this color

-- The theorem to prove
theorem ticket123123123_is_red :
  ticket123123123.color = Color.Red :=
sorry

end NUMINAMATH_CALUDE_ticket123123123_is_red_l3286_328639


namespace NUMINAMATH_CALUDE_bulb_cost_difference_l3286_328645

theorem bulb_cost_difference (lamp_cost : ℝ) (total_cost : ℝ) (bulb_cost : ℝ) : 
  lamp_cost = 7 → 
  2 * lamp_cost + 6 * bulb_cost = 32 → 
  bulb_cost < lamp_cost →
  lamp_cost - bulb_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_bulb_cost_difference_l3286_328645


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3286_328681

theorem arithmetic_calculations :
  (128 + 52 / 13 = 132) ∧
  (132 / 11 * 29 - 178 = 170) ∧
  (45 * (320 / (4 * 5)) = 720) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3286_328681


namespace NUMINAMATH_CALUDE_we_the_people_cows_l3286_328646

/-- The number of cows We the People has -/
def W : ℕ := 17

/-- The number of cows Happy Good Healthy Family has -/
def H : ℕ := 3 * W + 2

/-- The total number of cows when both groups are together -/
def total : ℕ := 70

theorem we_the_people_cows : W = 17 :=
  by sorry

end NUMINAMATH_CALUDE_we_the_people_cows_l3286_328646


namespace NUMINAMATH_CALUDE_same_solution_implies_b_value_l3286_328635

theorem same_solution_implies_b_value (x b : ℚ) : 
  (3 * x + 5 = 1) → 
  (b * x + 6 = 0) → 
  b = 9/2 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_b_value_l3286_328635


namespace NUMINAMATH_CALUDE_sidney_wednesday_jumping_jacks_l3286_328650

/-- The number of jumping jacks Sidney did on Wednesday -/
def sidney_wednesday : ℕ := sorry

/-- The total number of jumping jacks Sidney did -/
def sidney_total : ℕ := sorry

/-- The number of jumping jacks Brooke did -/
def brooke_total : ℕ := 438

theorem sidney_wednesday_jumping_jacks :
  sidney_wednesday = 40 ∧
  sidney_total = sidney_wednesday + 106 ∧
  brooke_total = 3 * sidney_total :=
by sorry

end NUMINAMATH_CALUDE_sidney_wednesday_jumping_jacks_l3286_328650


namespace NUMINAMATH_CALUDE_opposite_of_negative_2011_l3286_328692

theorem opposite_of_negative_2011 : 
  -((-2011) : ℤ) = (2011 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2011_l3286_328692


namespace NUMINAMATH_CALUDE_original_cost_price_correct_l3286_328627

/-- Represents the original cost price in euros -/
def original_cost_price : ℝ := 55.50

/-- Represents the selling price in dollars -/
def selling_price : ℝ := 100

/-- Represents the profit percentage -/
def profit_percentage : ℝ := 0.30

/-- Represents the exchange rate (dollars per euro) -/
def exchange_rate : ℝ := 1.2

/-- Represents the maintenance cost percentage -/
def maintenance_cost_percentage : ℝ := 0.05

/-- Represents the tax rate for the first 50 euros -/
def tax_rate_first_50 : ℝ := 0.10

/-- Represents the tax rate for amounts above 50 euros -/
def tax_rate_above_50 : ℝ := 0.15

/-- Represents the threshold for the tiered tax system -/
def tax_threshold : ℝ := 50

theorem original_cost_price_correct :
  let cost_price_dollars := selling_price / (1 + profit_percentage)
  let cost_price_euros := cost_price_dollars / exchange_rate
  let maintenance_cost := original_cost_price * maintenance_cost_percentage
  let tax_first_50 := min original_cost_price tax_threshold * tax_rate_first_50
  let tax_above_50 := max (original_cost_price - tax_threshold) 0 * tax_rate_above_50
  cost_price_euros = original_cost_price + maintenance_cost + tax_first_50 + tax_above_50 :=
by sorry

#check original_cost_price_correct

end NUMINAMATH_CALUDE_original_cost_price_correct_l3286_328627


namespace NUMINAMATH_CALUDE_det_A_equals_six_l3286_328685

theorem det_A_equals_six (a d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; -3, d]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2*a, 1; -1, d]
  A + B⁻¹ = 0 → Matrix.det A = 6 := by sorry

end NUMINAMATH_CALUDE_det_A_equals_six_l3286_328685


namespace NUMINAMATH_CALUDE_lcm_hcf_relation_l3286_328677

theorem lcm_hcf_relation (d c : ℕ) (h1 : d > 0) (h2 : Nat.lcm 76 d = 456) (h3 : Nat.gcd 76 d = c) : d = 24 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_relation_l3286_328677


namespace NUMINAMATH_CALUDE_squirrel_stockpiling_days_l3286_328641

/-- The number of busy squirrels -/
def busy_squirrels : ℕ := 2

/-- The number of nuts each busy squirrel stockpiles per day -/
def busy_squirrel_nuts_per_day : ℕ := 30

/-- The number of sleepy squirrels -/
def sleepy_squirrels : ℕ := 1

/-- The number of nuts the sleepy squirrel stockpiles per day -/
def sleepy_squirrel_nuts_per_day : ℕ := 20

/-- The total number of nuts found in Mason's car -/
def total_nuts : ℕ := 3200

/-- The number of days squirrels have been stockpiling nuts -/
def stockpiling_days : ℕ := 40

theorem squirrel_stockpiling_days :
  stockpiling_days * (busy_squirrels * busy_squirrel_nuts_per_day + sleepy_squirrels * sleepy_squirrel_nuts_per_day) = total_nuts :=
by sorry

end NUMINAMATH_CALUDE_squirrel_stockpiling_days_l3286_328641


namespace NUMINAMATH_CALUDE_cos_4theta_from_exp_l3286_328684

theorem cos_4theta_from_exp (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (1 - Complex.I * Real.sqrt 3) / 2) :
  Real.cos (4 * θ) = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_4theta_from_exp_l3286_328684


namespace NUMINAMATH_CALUDE_quadratic_root_one_iff_sum_coeffs_zero_l3286_328694

theorem quadratic_root_one_iff_sum_coeffs_zero (a b c : ℝ) :
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = 1) ↔ a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_one_iff_sum_coeffs_zero_l3286_328694


namespace NUMINAMATH_CALUDE_line_inclination_45_degrees_l3286_328638

/-- Given a line passing through points (-2, 1) and (m, 3) with an inclination angle of 45°, prove that m = 0 -/
theorem line_inclination_45_degrees (m : ℝ) : 
  (3 - 1) / (m + 2) = Real.tan (π / 4) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_45_degrees_l3286_328638


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l3286_328621

def f (x : ℝ) := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  let a := -3
  let b := 0
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 1 ∧ f x_min = -17 :=
sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l3286_328621


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3286_328603

noncomputable def z : ℂ := Complex.exp (-4 * Complex.I)

theorem z_in_second_quadrant : 
  z.re < 0 ∧ z.im > 0 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3286_328603


namespace NUMINAMATH_CALUDE_smallest_term_at_six_l3286_328691

/-- The general term of the sequence -/
def a (n : ℕ) : ℝ := 3 * n^2 - 38 * n + 12

/-- The index of the smallest term in the sequence -/
def smallest_term_index : ℕ := 6

/-- Theorem stating that the smallest term in the sequence occurs at index 6 -/
theorem smallest_term_at_six :
  ∀ (n : ℕ), n ≠ smallest_term_index → a n > a smallest_term_index :=
sorry

end NUMINAMATH_CALUDE_smallest_term_at_six_l3286_328691


namespace NUMINAMATH_CALUDE_solution_in_quadrant_II_l3286_328675

theorem solution_in_quadrant_II (k : ℝ) :
  (∃ x y : ℝ, 2 * x + y = 6 ∧ k * x - y = 4 ∧ x < 0 ∧ y > 0) ↔ k < -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_II_l3286_328675


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3286_328614

/-- Given a geometric sequence {a_n} with common ratio 2 and sum of first four terms equal to 1,
    prove that the sum of the first eight terms is 17. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  (a 1 + a 2 + a 3 + a 4 = 1) →  -- sum of first four terms is 1
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 17) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3286_328614


namespace NUMINAMATH_CALUDE_total_money_l3286_328602

/-- The amount of money Beth has -/
def beth_money : ℕ := 70

/-- The amount of money Jan has -/
def jan_money : ℕ := 80

/-- The condition that if Beth had $35 more, she would have $105 -/
axiom beth_condition : beth_money + 35 = 105

/-- The condition that if Jan had $10 less, he would have the same money as Beth -/
axiom jan_condition : jan_money - 10 = beth_money

/-- The theorem stating that Beth and Jan have $150 altogether -/
theorem total_money : beth_money + jan_money = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l3286_328602


namespace NUMINAMATH_CALUDE_probability_third_smallest_is_five_l3286_328607

def set_size : ℕ := 15
def selection_size : ℕ := 8
def target_number : ℕ := 5
def target_position : ℕ := 3

theorem probability_third_smallest_is_five :
  let total_combinations := Nat.choose set_size selection_size
  let favorable_combinations := 
    (Nat.choose (set_size - target_number) (selection_size - target_position)) *
    (Nat.choose (target_number - 1) (target_position - 1))
  (favorable_combinations : ℚ) / total_combinations = 4 / 21 := by
  sorry

end NUMINAMATH_CALUDE_probability_third_smallest_is_five_l3286_328607


namespace NUMINAMATH_CALUDE_paper_cups_pallets_l3286_328671

theorem paper_cups_pallets (total : ℕ) (towels tissues plates cups : ℕ) : 
  total = 20 ∧
  towels = total / 2 ∧
  tissues = total / 4 ∧
  plates = total / 5 ∧
  total = towels + tissues + plates + cups →
  cups = 1 := by
sorry

end NUMINAMATH_CALUDE_paper_cups_pallets_l3286_328671


namespace NUMINAMATH_CALUDE_divisibility_property_l3286_328610

theorem divisibility_property (n : ℕ) (hn : n > 0) :
  ∃ (a b : ℤ), (n : ℤ) ∣ (4 * a^2 + 9 * b^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3286_328610


namespace NUMINAMATH_CALUDE_angle_C_is_right_max_sum_CP_CB_l3286_328619

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the relationship between sides and angles
axiom sine_law : a / (Real.sin A) = b / (Real.sin B)

-- Given condition
axiom condition : 2 * c * Real.cos C + c = a * Real.cos B + b * Real.cos A

-- Point P on AB
variable (P : ℝ)

-- BP = 2
axiom BP_length : P = 2

-- sin∠PCA = 1/3
axiom sin_PCA : Real.sin (A - P) = 1/3

-- Theorem 1: Prove C = π/2
theorem angle_C_is_right : C = Real.pi / 2 := by sorry

-- Theorem 2: Prove CP + CB ≤ 2√3 for any valid P
theorem max_sum_CP_CB : ∀ x y : ℝ, x + y ≤ 2 * Real.sqrt 3 := by sorry

end

end NUMINAMATH_CALUDE_angle_C_is_right_max_sum_CP_CB_l3286_328619


namespace NUMINAMATH_CALUDE_ball_arrangements_count_l3286_328643

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of different arrangements of placing 5 numbered balls into 3 boxes,
    where two boxes contain 2 balls each and one box contains 1 ball --/
def ball_arrangements : ℕ :=
  choose 3 2 * choose 5 2 * choose 3 2

theorem ball_arrangements_count : ball_arrangements = 90 := by sorry

end NUMINAMATH_CALUDE_ball_arrangements_count_l3286_328643


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3286_328658

theorem smallest_n_for_roots_of_unity : ∃ (n : ℕ), n > 0 ∧ (∀ (z : ℂ), z^4 + z^3 + 1 = 0 → z^n = 1) ∧ (∀ (m : ℕ), m > 0 → (∀ (z : ℂ), z^4 + z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3286_328658


namespace NUMINAMATH_CALUDE_trains_crossing_time_l3286_328678

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (train_length : ℝ) (faster_speed : ℝ) : 
  train_length = 100 →
  faster_speed = 40 →
  (10 : ℝ) / 3 = (2 * train_length) / (faster_speed + faster_speed / 2) := by
  sorry

#check trains_crossing_time

end NUMINAMATH_CALUDE_trains_crossing_time_l3286_328678


namespace NUMINAMATH_CALUDE_equation_solutions_l3286_328617

theorem equation_solutions :
  ∀ n m : ℕ, m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3286_328617


namespace NUMINAMATH_CALUDE_additional_amount_needed_l3286_328600

def pencil_cost : ℚ := 6
def notebook_cost : ℚ := 7/2
def pen_cost : ℚ := 9/4
def initial_amount : ℚ := 5
def borrowed_amount : ℚ := 53/100

def total_cost : ℚ := pencil_cost + notebook_cost + pen_cost
def total_available : ℚ := initial_amount + borrowed_amount

theorem additional_amount_needed : total_cost - total_available = 311/50 := by
  sorry

end NUMINAMATH_CALUDE_additional_amount_needed_l3286_328600


namespace NUMINAMATH_CALUDE_compound_interest_period_l3286_328601

theorem compound_interest_period (P s k n : ℝ) (h_pos : k > -1) :
  P = s / (1 + k)^n →
  n = Real.log (s/P) / Real.log (1 + k) :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_period_l3286_328601


namespace NUMINAMATH_CALUDE_pony_speed_l3286_328689

/-- The average speed of a pony given specific conditions of a chase scenario. -/
theorem pony_speed (horse_speed : ℝ) (head_start : ℝ) (chase_time : ℝ) : 
  horse_speed = 35 → head_start = 3 → chase_time = 4 → 
  ∃ (pony_speed : ℝ), pony_speed = 20 ∧ 
  horse_speed * chase_time = pony_speed * (head_start + chase_time) := by
sorry

end NUMINAMATH_CALUDE_pony_speed_l3286_328689


namespace NUMINAMATH_CALUDE_billy_game_rounds_l3286_328697

def old_score : ℕ := 725
def min_points_per_round : ℕ := 3
def max_points_per_round : ℕ := 5
def target_score : ℕ := old_score + 1

theorem billy_game_rounds :
  let min_rounds := (target_score + max_points_per_round - 1) / max_points_per_round
  let max_rounds := target_score / min_points_per_round
  (min_rounds = 146 ∧ max_rounds = 242) := by
  sorry

end NUMINAMATH_CALUDE_billy_game_rounds_l3286_328697


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3286_328631

/-- Given a polynomial P with integer coefficients, 
    if P(2) and P(3) are both multiples of 6, 
    then P(5) is also a multiple of 6. -/
theorem polynomial_divisibility (P : ℤ → ℤ) 
  (h_poly : ∀ x y : ℤ, ∃ k : ℤ, P (x + y) = P x + P y + k * x * y)
  (h_p2 : ∃ m : ℤ, P 2 = 6 * m)
  (h_p3 : ∃ n : ℤ, P 3 = 6 * n) :
  ∃ l : ℤ, P 5 = 6 * l := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3286_328631


namespace NUMINAMATH_CALUDE_video_recorder_price_l3286_328676

def employee_price (wholesale_cost : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : ℝ :=
  let retail_price := wholesale_cost * (1 + markup_percentage)
  let discount := retail_price * discount_percentage
  retail_price - discount

theorem video_recorder_price :
  employee_price 200 0.2 0.2 = 192 := by
  sorry

end NUMINAMATH_CALUDE_video_recorder_price_l3286_328676


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3286_328606

theorem consecutive_integers_sum (n : ℤ) : n * (n + 1) = 20412 → n + (n + 1) = 287 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3286_328606


namespace NUMINAMATH_CALUDE_fencing_championship_medals_l3286_328605

/-- The number of ways to select first and second place winners from n fencers -/
def awardMedals (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 72 ways to award first and second place medals among 9 fencers -/
theorem fencing_championship_medals :
  awardMedals 9 = 72 := by
  sorry

end NUMINAMATH_CALUDE_fencing_championship_medals_l3286_328605


namespace NUMINAMATH_CALUDE_limit_hours_proof_l3286_328680

/-- The limit of hours per week for the regular rate -/
def limit_hours : ℕ := sorry

/-- The regular hourly rate in dollars -/
def regular_rate : ℚ := 16

/-- The overtime rate as a percentage increase over the regular rate -/
def overtime_rate_increase : ℚ := 75 / 100

/-- The total hours worked in a week -/
def total_hours : ℕ := 44

/-- The total compensation earned in dollars -/
def total_compensation : ℚ := 752

/-- Calculates the overtime rate based on the regular rate and overtime rate increase -/
def overtime_rate : ℚ := regular_rate * (1 + overtime_rate_increase)

theorem limit_hours_proof :
  regular_rate * limit_hours + 
  overtime_rate * (total_hours - limit_hours) = 
  total_compensation ∧ 
  limit_hours = 40 := by sorry

end NUMINAMATH_CALUDE_limit_hours_proof_l3286_328680


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l3286_328616

def S : Set Nat := {1, 2, 3, 4, 5}

def A : Set Nat := {x ∈ S | x % 2 = 0}

def B : Set Nat := {x ∈ S | x % 2 = 1}

theorem events_mutually_exclusive_and_complementary :
  (A ∩ B = ∅) ∧ (A ∪ B = S) := by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l3286_328616


namespace NUMINAMATH_CALUDE_distance_between_points_l3286_328657

/-- The distance between points (1, -3) and (-4, 7) is 5√5. -/
theorem distance_between_points : Real.sqrt ((1 - (-4))^2 + (-3 - 7)^2) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3286_328657


namespace NUMINAMATH_CALUDE_sample_size_proof_l3286_328623

theorem sample_size_proof (n : ℕ) 
  (h1 : ∃ k : ℕ, 2*k + 3*k + 4*k = 27) 
  (h2 : ∃ k : ℕ, n = 2*k + 3*k + 4*k + 6*k + 4*k + k) : n = 60 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_proof_l3286_328623


namespace NUMINAMATH_CALUDE_meal_contribution_proof_l3286_328668

/-- Calculates the individual contribution for a shared meal bill -/
def calculate_individual_contribution (total_price : ℚ) (coupon_value : ℚ) (num_people : ℕ) : ℚ :=
  (total_price - coupon_value) / num_people

/-- Proves that the individual contribution for the given scenario is $21 -/
theorem meal_contribution_proof (total_price : ℚ) (coupon_value : ℚ) (num_people : ℕ) 
  (h1 : total_price = 67)
  (h2 : coupon_value = 4)
  (h3 : num_people = 3) :
  calculate_individual_contribution total_price coupon_value num_people = 21 := by
  sorry

#eval calculate_individual_contribution 67 4 3

end NUMINAMATH_CALUDE_meal_contribution_proof_l3286_328668


namespace NUMINAMATH_CALUDE_f_properties_g_inequality_l3286_328683

/-- The function f(x) = a ln x + 1/x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1 / x

/-- The function g(x) = f(x) - 1/x -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 1 / x

theorem f_properties (a : ℝ) :
  (a > 0 → (∃ (x : ℝ), x > 0 ∧ f a x = a - a * Real.log a ∧ ∀ y > 0, f a y ≥ f a x) ∧
            (¬∃ (M : ℝ), ∀ x > 0, f a x ≤ M)) ∧
  (a ≤ 0 → ¬∃ (x : ℝ), x > 0 ∧ (∀ y > 0, f a y ≥ f a x ∨ ∀ y > 0, f a y ≤ f a x)) :=
sorry

theorem g_inequality (m n : ℝ) (h1 : 0 < m) (h2 : m < n) :
  (g 1 n - g 1 m) / 2 > (n - m) / (n + m) :=
sorry

end NUMINAMATH_CALUDE_f_properties_g_inequality_l3286_328683


namespace NUMINAMATH_CALUDE_retailer_pens_count_l3286_328634

theorem retailer_pens_count : ℕ :=
  let market_price : ℝ := 1  -- Arbitrary unit price
  let discount_rate : ℝ := 0.01
  let profit_rate : ℝ := 0.09999999999999996
  let cost_36_pens : ℝ := 36 * market_price
  let selling_price : ℝ := market_price * (1 - discount_rate)
  let n : ℕ := 40  -- Number of pens to be proven

  have h1 : n * selling_price - cost_36_pens = profit_rate * cost_36_pens := by sorry
  
  n


end NUMINAMATH_CALUDE_retailer_pens_count_l3286_328634


namespace NUMINAMATH_CALUDE_max_value_of_a_l3286_328604

theorem max_value_of_a (a : ℝ) : 
  (∀ k : ℝ, k ∈ Set.Icc (-1) 1 → 
    ∀ x : ℝ, x ∈ Set.Ioo 0 6 → 
      6 * Real.log x + x^2 - 8*x + a ≤ k*x) →
  a ≤ 6 - 6 * Real.log 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3286_328604


namespace NUMINAMATH_CALUDE_min_black_vertices_2016_gon_l3286_328647

/-- A regular polygon with n sides --/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A function that determines if three points form an acute triangle --/
def isAcuteTriangle (a b c : ℝ × ℝ) : Prop :=
  sorry

/-- The minimum number of vertices to paint black --/
def minBlackVertices (n : ℕ) : ℕ :=
  sorry

theorem min_black_vertices_2016_gon :
  ∀ (p : RegularPolygon 2016),
    minBlackVertices 2016 = 1008 ∧
    ∀ (S : Finset (Fin 2016)),
      S.card = 1008 →
      (∀ (a b c : Fin 2016), a ∈ S → b ∈ S → c ∈ S →
        ¬isAcuteTriangle (p.vertices a) (p.vertices b) (p.vertices c)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_black_vertices_2016_gon_l3286_328647


namespace NUMINAMATH_CALUDE_fifth_pythagorean_triple_l3286_328611

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def is_consecutive (m n : ℕ) : Prop :=
  m + 1 = n

theorem fifth_pythagorean_triple (a b c : ℕ) :
  is_pythagorean_triple 3 4 5 ∧
  is_pythagorean_triple 5 12 13 ∧
  is_pythagorean_triple 7 24 25 ∧
  is_pythagorean_triple 9 40 41 ∧
  (∀ x y z : ℕ, is_pythagorean_triple x y z → Odd x) ∧
  (∀ x y z : ℕ, is_pythagorean_triple x y z → is_consecutive y z) ∧
  (∀ x y z : ℕ, is_pythagorean_triple x y z → x * x = y + z) →
  is_pythagorean_triple 11 60 61 :=
by sorry

end NUMINAMATH_CALUDE_fifth_pythagorean_triple_l3286_328611


namespace NUMINAMATH_CALUDE_lg_sum_equals_two_l3286_328649

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_two : 2 * lg 2 + lg 25 = 2 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_two_l3286_328649


namespace NUMINAMATH_CALUDE_dwarf_truth_count_l3286_328663

/-- Represents the number of dwarfs who always tell the truth -/
def truthful_dwarfs : ℕ := sorry

/-- Represents the number of dwarfs who always lie -/
def lying_dwarfs : ℕ := sorry

/-- The total number of dwarfs -/
def total_dwarfs : ℕ := 10

/-- The number of times hands were raised for vanilla ice cream -/
def vanilla_hands : ℕ := 10

/-- The number of times hands were raised for chocolate ice cream -/
def chocolate_hands : ℕ := 5

/-- The number of times hands were raised for fruit ice cream -/
def fruit_hands : ℕ := 1

/-- The total number of times hands were raised -/
def total_hands_raised : ℕ := vanilla_hands + chocolate_hands + fruit_hands

theorem dwarf_truth_count :
  truthful_dwarfs + lying_dwarfs = total_dwarfs ∧
  truthful_dwarfs + 2 * lying_dwarfs = total_hands_raised ∧
  truthful_dwarfs = 4 := by sorry

end NUMINAMATH_CALUDE_dwarf_truth_count_l3286_328663


namespace NUMINAMATH_CALUDE_coterminal_angle_correct_l3286_328674

/-- The angle in degrees that is coterminal with 1000° and lies between 0° and 360° -/
def coterminal_angle : ℝ := 280

/-- Proof that the coterminal angle is correct -/
theorem coterminal_angle_correct :
  0 ≤ coterminal_angle ∧ 
  coterminal_angle < 360 ∧
  ∃ (k : ℤ), coterminal_angle = 1000 - 360 * k :=
by sorry

end NUMINAMATH_CALUDE_coterminal_angle_correct_l3286_328674


namespace NUMINAMATH_CALUDE_quadratic_root_form_n_l3286_328608

def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 8 * x - 5 = 0

def root_form (x m n p : ℝ) : Prop :=
  x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p

theorem quadratic_root_form_n :
  ∃ (m n p : ℕ+),
    (∀ x : ℝ, quadratic_equation x → root_form x m n p) ∧
    Nat.gcd m.val (Nat.gcd n.val p.val) = 1 ∧
    n = 124 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_form_n_l3286_328608


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3286_328624

theorem complex_equation_sum (a b : ℝ) : (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (2 + Complex.I) * (1 - b * Complex.I) = a + Complex.I →
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3286_328624
