import Mathlib

namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l1356_135688

theorem greatest_power_of_two_factor : ∃ k : ℕ, k = 502 ∧ 
  (∀ m : ℕ, 2^m ∣ (12^1002 - 6^501) → m ≤ k) ∧
  (2^k ∣ (12^1002 - 6^501)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l1356_135688


namespace NUMINAMATH_CALUDE_lucys_fish_count_l1356_135694

theorem lucys_fish_count (initial_fish : ℕ) 
  (h1 : initial_fish + 68 = 280) : initial_fish = 212 := by
  sorry

end NUMINAMATH_CALUDE_lucys_fish_count_l1356_135694


namespace NUMINAMATH_CALUDE_congruent_rectangle_perimeter_l1356_135619

/-- Given a rectangle with sides y and z, and a square with side x placed against
    the shorter side y, the perimeter of one of the four congruent rectangles
    formed in the remaining space is equal to 2y + 2z - 4x. -/
theorem congruent_rectangle_perimeter 
  (y z x : ℝ) 
  (h1 : y > 0) 
  (h2 : z > 0) 
  (h3 : x > 0) 
  (h4 : x < y) 
  (h5 : x < z) : 
  2*y + 2*z - 4*x = 2*((y - x) + (z - x)) := by
  sorry


end NUMINAMATH_CALUDE_congruent_rectangle_perimeter_l1356_135619


namespace NUMINAMATH_CALUDE_cross_section_area_fraction_l1356_135639

theorem cross_section_area_fraction (r : ℝ) (r_pos : r > 0) : 
  let sphere_surface_area := 4 * Real.pi * r^2
  let cross_section_radius := r / 2
  let cross_section_area := Real.pi * cross_section_radius^2
  cross_section_area / sphere_surface_area = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_fraction_l1356_135639


namespace NUMINAMATH_CALUDE_student_weight_average_l1356_135616

theorem student_weight_average (girls_avg : ℝ) (boys_avg : ℝ) 
  (h1 : girls_avg = 45) 
  (h2 : boys_avg = 55) : 
  (5 * girls_avg + 5 * boys_avg) / 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_student_weight_average_l1356_135616


namespace NUMINAMATH_CALUDE_partition_equivalence_l1356_135602

/-- Represents a partition of a positive integer -/
def Partition (n : ℕ) := Multiset ℕ

/-- The number of representations of n as a sum of distinct positive integers -/
def distinctSum (n : ℕ) : ℕ := sorry

/-- The number of representations of n as a sum of positive odd integers -/
def oddSum (n : ℕ) : ℕ := sorry

/-- The number of representations of n as a sum of positive integers, 
    where no term is repeated more than k-1 times -/
def limitedRepetitionSum (n k : ℕ) : ℕ := sorry

/-- The number of representations of n as a sum of positive integers, 
    where no term is divisible by k -/
def notDivisibleSum (n k : ℕ) : ℕ := sorry

/-- Main theorem stating the equality of representations -/
theorem partition_equivalence (n : ℕ) : 
  (∀ k : ℕ, k > 0 → limitedRepetitionSum n k = notDivisibleSum n k) ∧ 
  distinctSum n = oddSum n := by sorry

end NUMINAMATH_CALUDE_partition_equivalence_l1356_135602


namespace NUMINAMATH_CALUDE_solve_star_equation_l1356_135629

-- Define the star operation
def star (a b : ℝ) : ℝ := 4 * a + 2 * b

-- State the theorem
theorem solve_star_equation : 
  ∃ y : ℝ, star 3 (star 4 y) = 8 ∧ y = -9 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l1356_135629


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_equals_six_l1356_135676

theorem mean_equality_implies_y_equals_six :
  let mean1 := (4 + 8 + 16) / 3
  let mean2 := (10 + 12 + y) / 3
  mean1 = mean2 → y = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_equals_six_l1356_135676


namespace NUMINAMATH_CALUDE_f_properties_l1356_135666

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_properties (x₁ x₂ : ℝ) (h1 : x₂ > x₁) (h2 : x₁ > 1) :
  (|f x₁ - f x₂| < 1 / Real.exp 1) ∧ (f x₁ - f x₂ < x₂ - x₁) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1356_135666


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1356_135623

/-- The eccentricity of an ellipse with equation x²/4 + y²/9 = 1 is √5/3 -/
theorem ellipse_eccentricity : 
  let a : ℝ := 3
  let b : ℝ := 2
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let e : ℝ := c / a
  e = Real.sqrt 5 / 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1356_135623


namespace NUMINAMATH_CALUDE_factorial_square_root_square_l1356_135684

-- Define factorial function
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- State the theorem
theorem factorial_square_root_square : 
  (Real.sqrt (factorial 5 * factorial 4 : ℝ))^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_square_l1356_135684


namespace NUMINAMATH_CALUDE_helicopter_rental_cost_per_hour_l1356_135673

/-- Calculates the cost per hour for renting a helicopter --/
theorem helicopter_rental_cost_per_hour
  (hours_per_day : ℕ)
  (num_days : ℕ)
  (total_cost : ℕ)
  (h1 : hours_per_day = 2)
  (h2 : num_days = 3)
  (h3 : total_cost = 450) :
  total_cost / (hours_per_day * num_days) = 75 := by
sorry

end NUMINAMATH_CALUDE_helicopter_rental_cost_per_hour_l1356_135673


namespace NUMINAMATH_CALUDE_base6_addition_theorem_l1356_135653

/-- Converts a base 6 number to base 10 --/
def base6_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 6 --/
def base10_to_base6 (n : ℕ) : ℕ := sorry

/-- Addition in base 6 --/
def add_base6 (a b : ℕ) : ℕ :=
  base10_to_base6 (base6_to_base10 a + base6_to_base10 b)

theorem base6_addition_theorem :
  add_base6 1254 3452 = 5150 := by sorry

end NUMINAMATH_CALUDE_base6_addition_theorem_l1356_135653


namespace NUMINAMATH_CALUDE_balls_picked_proof_l1356_135622

def total_balls : ℕ := 9
def red_balls : ℕ := 3
def blue_balls : ℕ := 2
def green_balls : ℕ := 4

theorem balls_picked_proof (n : ℕ) : 
  total_balls = red_balls + blue_balls + green_balls →
  (red_balls.choose 2 : ℚ) / (total_balls.choose n) = 1 / 12 →
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_balls_picked_proof_l1356_135622


namespace NUMINAMATH_CALUDE_quadrilateral_ABCD_is_parallelogram_l1356_135664

-- Define the vertices of the quadrilateral
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, -1)
def C : ℝ × ℝ := (4, 2)
def D : ℝ × ℝ := (2, 3)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def DC : ℝ × ℝ := (C.1 - D.1, C.2 - D.2)

-- Define a function to check if two vectors are equal
def vectors_equal (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 = v2.1 ∧ v1.2 = v2.2

-- Define what it means for a quadrilateral to be a parallelogram
def is_parallelogram (a b c d : ℝ × ℝ) : Prop :=
  vectors_equal (b.1 - a.1, b.2 - a.2) (c.1 - d.1, c.2 - d.2) ∧
  vectors_equal (c.1 - b.1, c.2 - b.2) (d.1 - a.1, d.2 - a.2)

-- Theorem statement
theorem quadrilateral_ABCD_is_parallelogram :
  is_parallelogram A B C D :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_ABCD_is_parallelogram_l1356_135664


namespace NUMINAMATH_CALUDE_quadratic_maximum_l1356_135686

theorem quadratic_maximum : 
  (∀ s : ℝ, -3 * s^2 + 24 * s + 15 ≤ 63) ∧ 
  (∃ s : ℝ, -3 * s^2 + 24 * s + 15 = 63) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l1356_135686


namespace NUMINAMATH_CALUDE_expression_equals_3840_factorial_l1356_135665

/-- Custom factorial definition for positive p and b -/
def custom_factorial (p b : ℕ) : ℕ :=
  sorry

/-- The result of the expression 120₁₀!/20₃! + (10₂!)! -/
def expression_result : ℕ :=
  sorry

/-- Theorem stating that the expression equals (3840)! -/
theorem expression_equals_3840_factorial :
  expression_result = Nat.factorial 3840 :=
  sorry

end NUMINAMATH_CALUDE_expression_equals_3840_factorial_l1356_135665


namespace NUMINAMATH_CALUDE_inequality_proof_l1356_135632

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_prod : a * b * c * d = 1) :
  (1 / a) + (1 / b) + (1 / c) + (1 / d) + (12 / (a + b + c + d)) ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1356_135632


namespace NUMINAMATH_CALUDE_second_company_can_hire_three_geniuses_l1356_135637

/-- Represents a programmer --/
structure Programmer where
  id : Nat

/-- Represents a genius programmer --/
structure Genius extends Programmer

/-- Represents the hiring game between two companies --/
structure HiringGame where
  programmers : List Programmer
  geniuses : List Genius
  acquaintances : List (Programmer × Programmer)

/-- Represents a company's hiring strategy --/
structure HiringStrategy where
  nextHire : List Programmer → List Programmer → Option Programmer

/-- The result of the hiring game --/
inductive GameResult
  | FirstCompanyWins
  | SecondCompanyWins

/-- Simulates the hiring game given two strategies --/
def playGame (game : HiringGame) (strategy1 strategy2 : HiringStrategy) : GameResult :=
  sorry

/-- Theorem stating that there exists a winning strategy for the second company --/
theorem second_company_can_hire_three_geniuses :
  ∃ (game : HiringGame) (strategy : HiringStrategy),
    (game.geniuses.length = 4) →
    ∀ (opponent_strategy : HiringStrategy),
      playGame game opponent_strategy strategy = GameResult.SecondCompanyWins :=
sorry

end NUMINAMATH_CALUDE_second_company_can_hire_three_geniuses_l1356_135637


namespace NUMINAMATH_CALUDE_original_number_proof_l1356_135640

/-- Given a number n formed by adding a digit h in the 10's place of 284,
    where n is divisible by 6 and h = 1, prove that the original number
    without the 10's digit is 284. -/
theorem original_number_proof (n : ℕ) (h : ℕ) :
  n = 2000 + h * 100 + 84 →
  h = 1 →
  n % 6 = 0 →
  2000 + 84 = 284 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l1356_135640


namespace NUMINAMATH_CALUDE_sector_central_angle_l1356_135603

theorem sector_central_angle (arc_length : Real) (radius : Real) (central_angle : Real) :
  arc_length = 4 * Real.pi ∧ radius = 8 →
  arc_length = (central_angle * Real.pi * radius) / 180 →
  central_angle = 90 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1356_135603


namespace NUMINAMATH_CALUDE_range_of_a_l1356_135683

/-- Given sets A and B, where A is [-2, 4) and B is {x | x^2 - ax - 4 ≤ 0},
    if B is a subset of A, then a is in the range [0, 3). -/
theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := {x | -2 ≤ x ∧ x < 4}
  let B : Set ℝ := {x | x^2 - a*x - 4 ≤ 0}
  B ⊆ A → 0 ≤ a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1356_135683


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1356_135691

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = -8 ∧ x₂ = -4 ∧
  ∀ x : ℝ, x^2 + 6*x + 8 = -2*(x + 4)*(x + 5) ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1356_135691


namespace NUMINAMATH_CALUDE_player1_receives_57_coins_l1356_135690

/-- Represents the number of players and sectors on the table -/
def n : ℕ := 9

/-- Represents the total number of rotations -/
def total_rotations : ℕ := 11

/-- Represents the coins received by player 4 -/
def player4_coins : ℕ := 90

/-- Represents the coins received by player 8 -/
def player8_coins : ℕ := 35

/-- Represents the coins received by player 1 -/
def player1_coins : ℕ := 57

/-- Theorem stating that given the conditions, player 1 receives 57 coins -/
theorem player1_receives_57_coins :
  n = 9 →
  total_rotations = 11 →
  player4_coins = 90 →
  player8_coins = 35 →
  player1_coins = 57 :=
by sorry

end NUMINAMATH_CALUDE_player1_receives_57_coins_l1356_135690


namespace NUMINAMATH_CALUDE_hypotenuse_length_l1356_135668

/-- Given a right-angled triangle with sides a, b, and c (hypotenuse),
    where the sum of squares of all sides is 2500,
    prove that the length of the hypotenuse is 25√2. -/
theorem hypotenuse_length (a b c : ℝ) 
  (right_angle : a^2 + b^2 = c^2)  -- right-angled triangle condition
  (sum_squares : a^2 + b^2 + c^2 = 2500)  -- sum of squares condition
  : c = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l1356_135668


namespace NUMINAMATH_CALUDE_y_share_is_36_l1356_135677

/-- Given a sum divided among x, y, and z, where y gets 45 paisa and z gets 30 paisa for each rupee x gets,
    and the total amount is Rs. 140, prove that y's share is Rs. 36. -/
theorem y_share_is_36 
  (total : ℝ) 
  (x_share : ℝ) 
  (y_share : ℝ) 
  (z_share : ℝ) 
  (h1 : total = 140) 
  (h2 : y_share = 0.45 * x_share) 
  (h3 : z_share = 0.30 * x_share) 
  (h4 : total = x_share + y_share + z_share) : 
  y_share = 36 := by
sorry


end NUMINAMATH_CALUDE_y_share_is_36_l1356_135677


namespace NUMINAMATH_CALUDE_expression_equality_l1356_135609

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem expression_equality : (-8)^(1/3) + π^0 + lg 4 + lg 25 = 1 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l1356_135609


namespace NUMINAMATH_CALUDE_robin_water_consumption_l1356_135699

def bottles_morning : ℕ := sorry
def bottles_afternoon : ℕ := sorry
def total_bottles : ℕ := 14

theorem robin_water_consumption :
  (bottles_morning = bottles_afternoon) →
  (bottles_morning + bottles_afternoon = total_bottles) →
  bottles_morning = 7 := by
  sorry

end NUMINAMATH_CALUDE_robin_water_consumption_l1356_135699


namespace NUMINAMATH_CALUDE_solve_age_ratio_l1356_135650

def age_ratio_problem (p q x : ℕ) : Prop :=
  -- P's current age is 15
  p = 15 ∧
  -- Some years ago, the ratio of P's age to Q's age was 4:3
  (p - x) * 3 = (q - x) * 4 ∧
  -- 6 years from now, the ratio of their ages will be 7:6
  (p + 6) * 6 = (q + 6) * 7

theorem solve_age_ratio : ∃ q x, age_ratio_problem 15 q x ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_age_ratio_l1356_135650


namespace NUMINAMATH_CALUDE_total_stick_length_l1356_135697

/-- The length of Jazel's sticks -/
def stick_length (n : Nat) : ℝ :=
  match n with
  | 1 => 3
  | 2 => 2 * stick_length 1
  | 3 => stick_length 2 - 1
  | _ => 0

/-- The theorem stating the total length of Jazel's sticks -/
theorem total_stick_length :
  stick_length 1 + stick_length 2 + stick_length 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_stick_length_l1356_135697


namespace NUMINAMATH_CALUDE_probability_two_heads_two_tails_prove_probability_two_heads_two_tails_l1356_135634

/-- The probability of getting exactly two heads and two tails when tossing four fair coins -/
theorem probability_two_heads_two_tails : ℚ :=
  3/8

/-- Proof that the probability of getting exactly two heads and two tails
    when tossing four fair coins is 3/8 -/
theorem prove_probability_two_heads_two_tails :
  probability_two_heads_two_tails = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_heads_two_tails_prove_probability_two_heads_two_tails_l1356_135634


namespace NUMINAMATH_CALUDE_opposite_signs_sum_and_max_difference_l1356_135681

theorem opposite_signs_sum_and_max_difference (m n : ℤ) : 
  (|m| = 1 ∧ |n| = 4) → 
  ((m > 0 ∧ n < 0) ∨ (m < 0 ∧ n > 0) → (m + n = -3 ∨ m + n = 3)) ∧
  (∀ (a b : ℤ), |a| = 1 ∧ |b| = 4 → m - n ≥ a - b) :=
by sorry

end NUMINAMATH_CALUDE_opposite_signs_sum_and_max_difference_l1356_135681


namespace NUMINAMATH_CALUDE_gear_speed_ratio_l1356_135645

structure Gear where
  teeth : ℕ
  speed : ℚ

def meshed (g1 g2 : Gear) : Prop :=
  g1.teeth * g1.speed = g2.teeth * g2.speed

theorem gear_speed_ratio 
  (A B C D : Gear)
  (h_mesh_AB : meshed A B)
  (h_mesh_BC : meshed B C)
  (h_mesh_CD : meshed C D)
  (h_prime_p : Nat.Prime A.teeth)
  (h_prime_q : Nat.Prime B.teeth)
  (h_prime_r : Nat.Prime C.teeth)
  (h_prime_s : Nat.Prime D.teeth)
  (h_distinct : A.teeth ≠ B.teeth ∧ A.teeth ≠ C.teeth ∧ A.teeth ≠ D.teeth ∧
                B.teeth ≠ C.teeth ∧ B.teeth ≠ D.teeth ∧ C.teeth ≠ D.teeth)
  (h_speed_ratio : A.speed / A.teeth = B.speed / B.teeth ∧
                   B.speed / B.teeth = C.speed / C.teeth ∧
                   C.speed / C.teeth = D.speed / D.teeth) :
  ∃ (k : ℚ), A.speed = k * D.teeth * C.teeth ∧
             B.speed = k * D.teeth * A.teeth ∧
             C.speed = k * D.teeth * C.teeth ∧
             D.speed = k * C.teeth * A.teeth := by
  sorry

end NUMINAMATH_CALUDE_gear_speed_ratio_l1356_135645


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l1356_135625

theorem arithmetic_mean_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = (a + b + c) / 3 + 5)
  (h2 : (a + c) / 2 = (a + b + c) / 3 - 8) :
  (b + c) / 2 = (a + b + c) / 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l1356_135625


namespace NUMINAMATH_CALUDE_brendas_age_l1356_135620

/-- Proves that Brenda's age is 3 years old given the conditions from the problem. -/
theorem brendas_age :
  ∀ (addison_age brenda_age janet_age : ℕ),
  addison_age = 4 * brenda_age →
  janet_age = brenda_age + 9 →
  addison_age = janet_age →
  brenda_age = 3 := by
sorry

end NUMINAMATH_CALUDE_brendas_age_l1356_135620


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_not_1991_l1356_135654

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Theorem statement
theorem sum_of_digits_of_square_not_1991 :
  ∀ n : ℕ, sumOfDigits (n^2) ≠ 1991 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_not_1991_l1356_135654


namespace NUMINAMATH_CALUDE_postman_return_speed_l1356_135663

/-- Proves that given a round trip with specified conditions, the return speed is 6 miles/hour -/
theorem postman_return_speed 
  (total_distance : ℝ) 
  (first_half_time : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_distance = 4) 
  (h2 : first_half_time = 1) 
  (h3 : average_speed = 3) : 
  (total_distance / 2) / (total_distance / average_speed - first_half_time) = 6 := by
  sorry

end NUMINAMATH_CALUDE_postman_return_speed_l1356_135663


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_five_l1356_135604

theorem smallest_five_digit_mod_five : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  n % 5 = 4 ∧ 
  ∀ m : ℕ, (m ≥ 10000 ∧ m < 100000 ∧ m % 5 = 4) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_five_l1356_135604


namespace NUMINAMATH_CALUDE_correct_addition_l1356_135679

def original_sum : ℕ := 2002
def correct_sum : ℕ := 2502
def num1 : ℕ := 736
def num2 : ℕ := 941
def num3 : ℕ := 825

def smallest_digit_change (d : ℕ) : Prop :=
  d ≤ 9 ∧ 
  (num1 - d * 100 + num2 + num3 = correct_sum) ∧
  ∀ e, e < d → (num1 - e * 100 + num2 + num3 ≠ correct_sum)

theorem correct_addition :
  smallest_digit_change 5 :=
sorry

end NUMINAMATH_CALUDE_correct_addition_l1356_135679


namespace NUMINAMATH_CALUDE_cos_300_degrees_l1356_135642

theorem cos_300_degrees : Real.cos (300 * π / 180) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l1356_135642


namespace NUMINAMATH_CALUDE_vector_equality_implies_norm_equality_l1356_135638

theorem vector_equality_implies_norm_equality 
  {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] 
  (a b : n) (ha : a ≠ 0) (hb : b ≠ 0) :
  a + 2 • b = 0 → ‖a - b‖ = ‖a‖ + ‖b‖ := by
sorry

end NUMINAMATH_CALUDE_vector_equality_implies_norm_equality_l1356_135638


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l1356_135605

theorem simultaneous_equations_solution (k : ℝ) :
  (k ≠ 1) ↔ (∃ x y : ℝ, (y = k * x + 2) ∧ (y = (3 * k - 2) * x + 5)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l1356_135605


namespace NUMINAMATH_CALUDE_reciprocal_comparison_l1356_135614

theorem reciprocal_comparison : 
  (let numbers := [-1/2, -3, 1/3, 3, 3/2]
   ∀ x ∈ numbers, x < 1/x ↔ (x = -3 ∨ x = 1/3)) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_comparison_l1356_135614


namespace NUMINAMATH_CALUDE_find_n_l1356_135682

theorem find_n (x y n : ℝ) (h1 : x = 3) (h2 : y = 27) (h3 : n^(n / (2 + x)) = y) : n = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1356_135682


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_sum_of_digits_15_l1356_135649

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isFirstYearAfter2020WithSumOfDigits15 (year : ℕ) : Prop :=
  year > 2020 ∧ 
  sumOfDigits year = 15 ∧
  ∀ y, 2020 < y ∧ y < year → sumOfDigits y ≠ 15

theorem first_year_after_2020_with_sum_of_digits_15 :
  isFirstYearAfter2020WithSumOfDigits15 2049 := by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2020_with_sum_of_digits_15_l1356_135649


namespace NUMINAMATH_CALUDE_middle_terms_equal_l1356_135618

/-- Given two geometric progressions with positive terms satisfying certain conditions,
    prove that the middle terms are equal. -/
theorem middle_terms_equal (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
    (h_pos_a : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0)
    (h_pos_b : b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0)
    (h_geom_a : ∃ q : ℝ, q > 0 ∧ a₂ = a₁ * q ∧ a₃ = a₂ * q)
    (h_geom_b : ∃ r : ℝ, r > 0 ∧ b₂ = b₁ * r ∧ b₃ = b₂ * r)
    (h_sum_eq : a₁ + a₂ + a₃ = b₁ + b₂ + b₃)
    (h_arith_prog : ∃ d : ℝ, a₂ * b₂ - a₁ * b₁ = d ∧ a₃ * b₃ - a₂ * b₂ = d) :
  a₂ = b₂ := by
  sorry

end NUMINAMATH_CALUDE_middle_terms_equal_l1356_135618


namespace NUMINAMATH_CALUDE_angle_between_points_after_one_second_l1356_135696

/-- Represents the angular velocity of a rotating point. -/
structure AngularVelocity where
  value : ℝ
  positive : value > 0

/-- Represents a rotating point on a circle. -/
structure RotatingPoint where
  velocity : AngularVelocity

/-- Calculates the angle between two rotating points after 1 second. -/
def angleBetweenPoints (p1 p2 : RotatingPoint) : ℝ := sorry

/-- Theorem stating the angle between two rotating points after 1 second. -/
theorem angle_between_points_after_one_second 
  (p1 p2 : RotatingPoint) 
  (h1 : p1.velocity.value - p2.velocity.value = 2 * Real.pi / 60)  -- Two more revolutions per minute
  (h2 : 1 / p1.velocity.value - 1 / p2.velocity.value = 5)  -- 5 seconds faster revolution
  : angleBetweenPoints p1 p2 = 12 * Real.pi / 180 ∨ 
    angleBetweenPoints p1 p2 = 60 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_points_after_one_second_l1356_135696


namespace NUMINAMATH_CALUDE_parallelogram_area_l1356_135661

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Checks if a line is perpendicular to another line -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (m p1 p2 : Point) : Prop := sorry

/-- Main theorem: Area of parallelogram ABCD is √35 -/
theorem parallelogram_area (ABCD : Parallelogram) (E : Point) :
  area ABCD = Real.sqrt 35 :=
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1356_135661


namespace NUMINAMATH_CALUDE_convex_quadrilateral_triangle_angles_l1356_135667

theorem convex_quadrilateral_triangle_angles 
  (α β γ θ₁ θ₂ θ₃ θ₄ : Real) : 
  (α + β + γ = Real.pi) →  -- Sum of angles in a triangle is π radians (180°)
  (θ₁ + θ₂ + θ₃ + θ₄ = 2 * Real.pi) →  -- Sum of angles in a quadrilateral is 2π radians (360°)
  (θ₁ = α) → (θ₂ = β) → (θ₃ = γ) →  -- Three angles of quadrilateral equal to triangle angles
  ¬(θ₁ < Real.pi ∧ θ₂ < Real.pi ∧ θ₃ < Real.pi ∧ θ₄ < Real.pi)  -- Negation of convexity condition
  := by sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_triangle_angles_l1356_135667


namespace NUMINAMATH_CALUDE_fraction_decomposition_l1356_135687

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 5/3) :
  (7 * x - 15) / (3 * x^2 - x - 10) = (29/11) / (x + 2) + (-9/11) / (3*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l1356_135687


namespace NUMINAMATH_CALUDE_planes_parallel_from_parallel_perp_lines_l1356_135656

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_from_parallel_perp_lines 
  (m n : Line) (α β : Plane) :
  m ≠ n →
  parallel m n →
  perp_line_plane m α →
  perp_line_plane n β →
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_from_parallel_perp_lines_l1356_135656


namespace NUMINAMATH_CALUDE_four_digit_sum_product_divisible_by_11_l1356_135633

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Converts four digits to a four-digit number -/
def toNumber (w x y z : Digit) : ℕ :=
  1000 * w.val + 100 * x.val + 10 * y.val + z.val

theorem four_digit_sum_product_divisible_by_11 
  (w x y z : Digit) 
  (hw : w ≠ x) (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ w) 
  (hwx : w ≠ y) (hwy : w ≠ z) (hxy : x ≠ z) : 
  11 ∣ (toNumber w x y z + toNumber z y x w + toNumber w x y z * toNumber z y x w) :=
sorry

end NUMINAMATH_CALUDE_four_digit_sum_product_divisible_by_11_l1356_135633


namespace NUMINAMATH_CALUDE_tomato_price_is_five_l1356_135610

/-- Represents the price per pound of tomatoes -/
def tomato_price : ℝ := sorry

/-- The number of pounds of tomatoes bought -/
def tomato_pounds : ℝ := 2

/-- The number of pounds of apples bought -/
def apple_pounds : ℝ := 5

/-- The price per pound of apples -/
def apple_price : ℝ := 6

/-- The total amount spent -/
def total_spent : ℝ := 40

/-- Theorem stating that the price per pound of tomatoes is $5 -/
theorem tomato_price_is_five :
  tomato_price * tomato_pounds + apple_price * apple_pounds = total_spent →
  tomato_price = 5 := by sorry

end NUMINAMATH_CALUDE_tomato_price_is_five_l1356_135610


namespace NUMINAMATH_CALUDE_cosine_product_equals_quarter_l1356_135630

theorem cosine_product_equals_quarter : 
  (1 + Real.cos (π/4)) * (1 + Real.cos (3*π/4)) * (1 + Real.cos (π/2)) * (1 - Real.cos (π/4)^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_equals_quarter_l1356_135630


namespace NUMINAMATH_CALUDE_range_of_n_l1356_135626

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 2*x + 2

-- Define the point P
structure Point where
  m : ℝ
  n : ℝ

-- Define the condition that P lies on the graph of f
def on_graph (P : Point) : Prop := P.n = f P.m

-- Define the condition that the circle intersects the y-axis
def circle_intersects_y_axis (P : Point) : Prop := abs P.m ≤ 2

-- Theorem statement
theorem range_of_n (P : Point) 
  (h1 : on_graph P) 
  (h2 : circle_intersects_y_axis P) : 
  1 ≤ P.n ∧ P.n < 10 := by sorry

end NUMINAMATH_CALUDE_range_of_n_l1356_135626


namespace NUMINAMATH_CALUDE_domain_of_g_is_closed_unit_interval_l1356_135607

-- Define the function f with domain [0,1]
def f : Set ℝ := Set.Icc 0 1

-- Define the function g(x) = f(x^2)
def g (x : ℝ) : Prop := x^2 ∈ f

-- Theorem statement
theorem domain_of_g_is_closed_unit_interval :
  {x : ℝ | g x} = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_domain_of_g_is_closed_unit_interval_l1356_135607


namespace NUMINAMATH_CALUDE_lawrence_marbles_l1356_135635

theorem lawrence_marbles (num_friends : ℕ) (marbles_per_friend : ℕ) 
  (h1 : num_friends = 64) (h2 : marbles_per_friend = 86) : 
  num_friends * marbles_per_friend = 5504 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_marbles_l1356_135635


namespace NUMINAMATH_CALUDE_jqk_count_l1356_135660

/-- Given a pack of 52 cards, if the probability of drawing a jack, queen, or king
    is 0.23076923076923078, then the number of jacks, queens, and kings in the pack is 12. -/
theorem jqk_count (total_cards : ℕ) (prob : ℝ) (jqk_count : ℕ) : 
  total_cards = 52 →
  prob = 0.23076923076923078 →
  prob = (jqk_count : ℝ) / total_cards →
  jqk_count = 12 :=
by sorry

end NUMINAMATH_CALUDE_jqk_count_l1356_135660


namespace NUMINAMATH_CALUDE_max_isosceles_triangles_2017gon_l1356_135657

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- Represents a triangulation of a regular polygon -/
structure Triangulation (n : ℕ) where
  polygon : RegularPolygon n
  num_diagonals : ℕ
  num_triangles : ℕ
  diagonals_non_intersecting : Bool
  triangle_count_valid : num_triangles = n - 2

/-- Counts the maximum number of isosceles triangles in a given triangulation -/
def max_isosceles_triangles (t : Triangulation 2017) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of isosceles triangles -/
theorem max_isosceles_triangles_2017gon :
  ∀ (t : Triangulation 2017),
    t.num_diagonals = 2014 ∧ 
    t.diagonals_non_intersecting = true →
    max_isosceles_triangles t = 2010 :=
  sorry

end NUMINAMATH_CALUDE_max_isosceles_triangles_2017gon_l1356_135657


namespace NUMINAMATH_CALUDE_estimate_fish_population_l1356_135627

/-- Estimates the number of fish in a pond using the catch-mark-recapture method. -/
theorem estimate_fish_population (initial_catch : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  initial_catch > 0 →
  second_catch > 0 →
  marked_in_second > 0 →
  marked_in_second ≤ second_catch →
  marked_in_second ≤ initial_catch →
  (initial_catch * second_catch) / marked_in_second = 1200 →
  ∃ (estimated_population : ℕ), estimated_population = 1200 :=
by sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l1356_135627


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l1356_135669

/-- Given a paint mixture ratio of 3:2:4 for blue:green:white paint, 
    if 12 quarts of white paint are used, then 6 quarts of green paint should be used. -/
theorem paint_mixture_ratio (blue green white : ℚ) : 
  blue / green = 3 / 2 ∧ 
  green / white = 2 / 4 ∧ 
  white = 12 → 
  green = 6 := by
sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l1356_135669


namespace NUMINAMATH_CALUDE_tim_tetrises_l1356_135648

/-- The number of tetrises Tim scored -/
def num_tetrises (single_points tetris_points num_singles total_points : ℕ) : ℕ :=
  (total_points - num_singles * single_points) / tetris_points

/-- Theorem: Tim scored 4 tetrises -/
theorem tim_tetrises :
  let single_points : ℕ := 1000
  let tetris_points : ℕ := 8 * single_points
  let num_singles : ℕ := 6
  let total_points : ℕ := 38000
  num_tetrises single_points tetris_points num_singles total_points = 4 := by
  sorry

end NUMINAMATH_CALUDE_tim_tetrises_l1356_135648


namespace NUMINAMATH_CALUDE_marco_strawberry_weight_l1356_135606

theorem marco_strawberry_weight (total_weight : ℕ) (weight_difference : ℕ) :
  total_weight = 47 →
  weight_difference = 13 →
  ∃ (marco_weight dad_weight : ℕ),
    marco_weight + dad_weight = total_weight ∧
    marco_weight = dad_weight + weight_difference ∧
    marco_weight = 30 :=
by sorry

end NUMINAMATH_CALUDE_marco_strawberry_weight_l1356_135606


namespace NUMINAMATH_CALUDE_negative_representation_is_spending_l1356_135655

-- Define a type for monetary transactions
inductive MonetaryTransaction
| Receive (amount : ℤ)
| Spend (amount : ℤ)

-- Define a function to represent transactions as integers
def represent (t : MonetaryTransaction) : ℤ :=
  match t with
  | MonetaryTransaction.Receive amount => amount
  | MonetaryTransaction.Spend amount => -amount

-- State the theorem
theorem negative_representation_is_spending :
  (represent (MonetaryTransaction.Receive 100) = 100) →
  (represent (MonetaryTransaction.Spend 80) = -80) :=
by sorry

end NUMINAMATH_CALUDE_negative_representation_is_spending_l1356_135655


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_41_l1356_135689

/-- Represents the pricing model of a caterer -/
structure CatererPricing where
  basicFee : ℕ
  pricePerPerson : ℕ → ℕ

/-- The pricing model for the first caterer -/
def firstCaterer : CatererPricing :=
  { basicFee := 150,
    pricePerPerson := fun _ => 17 }

/-- The pricing model for the second caterer -/
def secondCaterer : CatererPricing :=
  { basicFee := 250,
    pricePerPerson := fun x => if x ≤ 40 then 15 else 13 }

/-- Calculate the total price for a caterer given the number of people -/
def totalPrice (c : CatererPricing) (people : ℕ) : ℕ :=
  c.basicFee + c.pricePerPerson people * people

/-- The theorem stating that 41 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_41 :
  (∀ n < 41, totalPrice firstCaterer n ≤ totalPrice secondCaterer n) ∧
  (totalPrice secondCaterer 41 < totalPrice firstCaterer 41) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_41_l1356_135689


namespace NUMINAMATH_CALUDE_intersection_empty_iff_not_p_sufficient_not_necessary_l1356_135674

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - a^2 ≥ 0}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- Theorem 1: A ∩ B = ∅ if and only if a ≥ 5
theorem intersection_empty_iff (a : ℝ) : A ∩ B a = ∅ ↔ a ≥ 5 := by sorry

-- Theorem 2: ¬p is a sufficient but not necessary condition for q if and only if 0 < a ≤ 2
theorem not_p_sufficient_not_necessary (a : ℝ) : 
  (∀ x, ¬p x → q a x) ∧ (∃ x, q a x ∧ p x) ↔ 0 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_not_p_sufficient_not_necessary_l1356_135674


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1356_135670

/-- The lateral surface area of a cone with base radius 2 cm and slant height 5 cm is 10π cm². -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 2  -- radius in cm
  let l : ℝ := 5  -- slant height in cm
  let lateral_area := (1/2) * l * (2 * Real.pi * r)
  lateral_area = 10 * Real.pi
  := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1356_135670


namespace NUMINAMATH_CALUDE_spherical_triangle_smallest_angle_l1356_135646

/-- 
Theorem: In a spherical triangle where the interior angles are in a 4:5:6 ratio 
and their sum is 270 degrees, the smallest angle measures 72 degrees.
-/
theorem spherical_triangle_smallest_angle 
  (a b c : ℝ) 
  (ratio : a = 4 * (b / 5) ∧ b = 5 * (c / 6)) 
  (sum_270 : a + b + c = 270) : 
  a = 72 := by
sorry

end NUMINAMATH_CALUDE_spherical_triangle_smallest_angle_l1356_135646


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l1356_135624

theorem fraction_to_zero_power :
  let a : ℤ := -573293
  let b : ℕ := 7903827
  (a : ℚ) / b ^ (0 : ℕ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l1356_135624


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1356_135658

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  inningsPlayed : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the increase in average after a new innings -/
def averageIncrease (prevStats : BatsmanStats) (newInningRuns : ℕ) (newAverage : ℚ) : ℚ :=
  newAverage - prevStats.average

/-- Theorem: The increase in the batsman's average is 4 runs -/
theorem batsman_average_increase :
  ∀ (prevStats : BatsmanStats),
  prevStats.inningsPlayed = 16 →
  averageIncrease prevStats 87 23 = 4 := by
  sorry

#check batsman_average_increase

end NUMINAMATH_CALUDE_batsman_average_increase_l1356_135658


namespace NUMINAMATH_CALUDE_painting_area_calculation_l1356_135695

theorem painting_area_calculation (price_per_sqft : ℝ) (total_cost : ℝ) (area : ℝ) :
  price_per_sqft = 15 →
  total_cost = 840 →
  area * price_per_sqft = total_cost →
  area = 56 := by
sorry

end NUMINAMATH_CALUDE_painting_area_calculation_l1356_135695


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l1356_135611

theorem gcd_lcm_product (a b : ℕ) (ha : a = 180) (hb : b = 225) :
  (Nat.gcd a b) * (Nat.lcm a b) = 40500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l1356_135611


namespace NUMINAMATH_CALUDE_zoey_holidays_per_month_l1356_135613

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total number of holidays Zoey took in a year -/
def total_holidays : ℕ := 24

/-- Zoey took holidays every month for an entire year -/
axiom holidays_every_month : ∀ (month : ℕ), month ≤ months_in_year → ∃ (holidays : ℕ), holidays > 0

/-- The number of holidays Zoey took each month -/
def holidays_per_month : ℚ := total_holidays / months_in_year

theorem zoey_holidays_per_month : holidays_per_month = 2 := by sorry

end NUMINAMATH_CALUDE_zoey_holidays_per_month_l1356_135613


namespace NUMINAMATH_CALUDE_project_men_count_l1356_135698

/-- The number of men originally working on the project -/
def original_men : ℕ := 110

/-- The number of days it takes the original number of men to complete the work -/
def original_days : ℕ := 100

/-- The reduction in the number of men -/
def men_reduction : ℕ := 10

/-- The increase in days when the number of men is reduced -/
def days_increase : ℕ := 10

theorem project_men_count :
  (original_men * original_days = (original_men - men_reduction) * (original_days + days_increase)) →
  original_men = 110 := by
  sorry

end NUMINAMATH_CALUDE_project_men_count_l1356_135698


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l1356_135600

/-- Emma's current age -/
def emma_age : ℕ := sorry

/-- Sarah's current age -/
def sarah_age : ℕ := sorry

/-- The number of years until the ratio of their ages is 3:2 -/
def years_until_ratio : ℕ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem age_ratio_theorem :
  (emma_age - 3 = 2 * (sarah_age - 3)) ∧
  (emma_age - 5 = 3 * (sarah_age - 5)) →
  years_until_ratio = 1 ∧
  (emma_age + years_until_ratio) * 2 = (sarah_age + years_until_ratio) * 3 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_theorem_l1356_135600


namespace NUMINAMATH_CALUDE_even_product_square_sum_solution_l1356_135659

theorem even_product_square_sum_solution (a b : ℤ) (h : 2 ∣ (a * b)) :
  ∃ (x y : ℤ), a^2 + b^2 + x^2 = y^2 :=
by sorry

end NUMINAMATH_CALUDE_even_product_square_sum_solution_l1356_135659


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1356_135685

theorem simplify_fraction_product : (144 : ℚ) / 1296 * 72 = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1356_135685


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1356_135615

theorem polynomial_factorization (x : ℝ) : 
  75 * x^7 - 175 * x^13 = 25 * x^7 * (3 - 7 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1356_135615


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l1356_135693

/-- A polynomial is a perfect square trinomial if it can be expressed as (x + a)^2 for some real number a. -/
def IsPerfectSquareTrinomial (p : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, p x = (x + a)^2

/-- Given that x^2 + kx + 25 is a perfect square trinomial, prove that k = 10 or k = -10. -/
theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial (fun x => x^2 + k*x + 25) →
  k = 10 ∨ k = -10 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l1356_135693


namespace NUMINAMATH_CALUDE_product_of_fractions_l1356_135692

theorem product_of_fractions : (1 / 3 : ℚ) * (4 / 7 : ℚ) * (5 / 8 : ℚ) = 5 / 42 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1356_135692


namespace NUMINAMATH_CALUDE_black_haired_girls_l1356_135631

/-- Represents the number of girls in the choir -/
def initial_total : ℕ := 80

/-- Represents the number of blonde-haired girls added -/
def blonde_added : ℕ := 10

/-- Represents the initial number of blonde-haired girls -/
def initial_blonde : ℕ := 30

/-- Theorem stating the number of black-haired girls in the choir -/
theorem black_haired_girls : 
  initial_total - (initial_blonde + blonde_added) = 50 := by
  sorry

end NUMINAMATH_CALUDE_black_haired_girls_l1356_135631


namespace NUMINAMATH_CALUDE_length_of_AB_l1356_135628

-- Define the circle Γ
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 3}

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - m * p.2 - 1 = 0}
def l₂ (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | m * p.1 + p.2 - m = 0}

-- Define the intersection points
def A (m : ℝ) : ℝ × ℝ := sorry
def B (m : ℝ) : ℝ × ℝ := sorry
def C (m : ℝ) : ℝ × ℝ := sorry
def D (m : ℝ) : ℝ × ℝ := sorry

-- State the theorem
theorem length_of_AB (m : ℝ) : 
  A m ∈ Γ ∧ A m ∈ l₁ m ∧
  B m ∈ Γ ∧ B m ∈ l₂ m ∧
  C m ∈ Γ ∧ C m ∈ l₁ m ∧
  D m ∈ Γ ∧ D m ∈ l₂ m ∧
  (A m).2 > 0 ∧ (B m).2 > 0 ∧
  (C m).2 < 0 ∧ (D m).2 < 0 ∧
  (D m).2 - (C m).2 = (D m).1 - (C m).1 →
  Real.sqrt ((A m).1 - (B m).1)^2 + ((A m).2 - (B m).2)^2 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AB_l1356_135628


namespace NUMINAMATH_CALUDE_exists_special_box_l1356_135641

/-- A rectangular box with integer dimensions (a, b, c) where the volume is four times the surface area -/
def SpecialBox (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = 8 * (a * b + b * c + c * a)

/-- There exists at least one ordered triple (a, b, c) satisfying the SpecialBox conditions -/
theorem exists_special_box : ∃ (a b c : ℕ), SpecialBox a b c := by
  sorry

end NUMINAMATH_CALUDE_exists_special_box_l1356_135641


namespace NUMINAMATH_CALUDE_max_y_value_l1356_135647

theorem max_y_value (x y : ℝ) : 
  3 * x^2 - x * y = 1 →
  9 * x * y + y^2 = 22 →
  y ≤ 5.5 :=
by sorry

end NUMINAMATH_CALUDE_max_y_value_l1356_135647


namespace NUMINAMATH_CALUDE_max_digit_sum_l1356_135651

/-- A_n is an n-digit integer with all digits equal to a -/
def A_n (a : ℕ) (n : ℕ) : ℕ := a * (10^n - 1) / 9

/-- B_n is a 2n-digit integer with all digits equal to b -/
def B_n (b : ℕ) (n : ℕ) : ℕ := b * (10^(2*n) - 1) / 9

/-- C_n is a 3n-digit integer with all digits equal to c -/
def C_n (c : ℕ) (n : ℕ) : ℕ := c * (10^(3*n) - 1) / 9

/-- The theorem statement -/
theorem max_digit_sum (a b c : ℕ) (ha : 0 < a ∧ a < 10) (hb : 0 < b ∧ b < 10) (hc : 0 < c ∧ c < 10) :
  (∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ 0 < n₁ ∧ 0 < n₂ ∧ 
    C_n c n₁ - A_n a n₁ = (B_n b n₁)^2 ∧
    C_n c n₂ - A_n a n₂ = (B_n b n₂)^2) →
  a + b + c ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_l1356_135651


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l1356_135608

-- Define the function f(x) = -x^3
def f (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l1356_135608


namespace NUMINAMATH_CALUDE_dvd_pack_discounted_price_l1356_135643

/-- The price of a DVD pack after discount -/
def price_after_discount (original_price discount : ℕ) : ℕ :=
  original_price - discount

/-- Theorem: The price of a DVD pack after a $25 discount is $51, given that the original price is $76 -/
theorem dvd_pack_discounted_price :
  price_after_discount 76 25 = 51 := by
  sorry

end NUMINAMATH_CALUDE_dvd_pack_discounted_price_l1356_135643


namespace NUMINAMATH_CALUDE_one_fourth_of_8_2_l1356_135671

theorem one_fourth_of_8_2 : (8.2 : ℚ) / 4 = 41 / 20 := by sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_2_l1356_135671


namespace NUMINAMATH_CALUDE_spinner_probability_l1356_135601

theorem spinner_probability (P : Finset (Fin 4) → ℚ) 
  (h_total : P {0, 1, 2, 3} = 1)
  (h_A : P {0} = 1/4)
  (h_B : P {1} = 1/3)
  (h_D : P {3} = 1/6) :
  P {2} = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l1356_135601


namespace NUMINAMATH_CALUDE_no_statement_implies_p_and_not_q_l1356_135612

theorem no_statement_implies_p_and_not_q (p q : Prop) : 
  ¬((p → q) → (p ∧ ¬q)) ∧ 
  ¬((p ∨ ¬q) → (p ∧ ¬q)) ∧ 
  ¬((¬p ∧ q) → (p ∧ ¬q)) ∧ 
  ¬((¬p ∨ q) → (p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_no_statement_implies_p_and_not_q_l1356_135612


namespace NUMINAMATH_CALUDE_train_length_l1356_135675

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) : 
  speed = 36 * (1000 / 3600) → 
  time = 27.997760179185665 → 
  bridge_length = 180 → 
  speed * time - bridge_length = 99.97760179185665 := by
sorry

#eval (36 * (1000 / 3600) * 27.997760179185665 - 180)

end NUMINAMATH_CALUDE_train_length_l1356_135675


namespace NUMINAMATH_CALUDE_experimental_fields_yield_l1356_135617

theorem experimental_fields_yield (x : ℝ) : 
  x > 0 →
  (900 : ℝ) / x = (1500 : ℝ) / (x + 300) ↔
  (∃ (area : ℝ), 
    area > 0 ∧
    area * x = 900 ∧
    area * (x + 300) = 1500) :=
by sorry

end NUMINAMATH_CALUDE_experimental_fields_yield_l1356_135617


namespace NUMINAMATH_CALUDE_chess_team_probability_l1356_135652

def chess_club_size : ℕ := 20
def num_boys : ℕ := 12
def num_girls : ℕ := 8
def team_size : ℕ := 4

theorem chess_team_probability :
  let total_combinations := Nat.choose chess_club_size team_size
  let all_boys_combinations := Nat.choose num_boys team_size
  let all_girls_combinations := Nat.choose num_girls team_size
  let probability_at_least_one_each := 1 - (all_boys_combinations + all_girls_combinations : ℚ) / total_combinations
  probability_at_least_one_each = 4280 / 4845 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_probability_l1356_135652


namespace NUMINAMATH_CALUDE_certain_value_problem_l1356_135644

theorem certain_value_problem (x y : ℝ) : x = 69 ∧ x - 18 = 3 * (y - x) → y = 86 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_problem_l1356_135644


namespace NUMINAMATH_CALUDE_euler_number_proof_l1356_135672

def gauss_number : ℂ := Complex.mk 6 4

theorem euler_number_proof (product : ℂ) (h1 : product = Complex.mk 48 (-18)) :
  ∃ (euler_number : ℂ), euler_number * gauss_number = product ∧ euler_number = Complex.mk 4 (-6) := by
  sorry

end NUMINAMATH_CALUDE_euler_number_proof_l1356_135672


namespace NUMINAMATH_CALUDE_jasons_points_theorem_l1356_135662

/-- Calculates the total points Jason has from seashells and starfish -/
def jasons_total_points (initial_seashells : ℕ) (initial_starfish : ℕ) 
  (seashell_points : ℕ) (starfish_points : ℕ)
  (seashells_given_tim : ℕ) (seashells_given_lily : ℕ)
  (seashells_found : ℕ) (seashells_lost : ℕ) : ℕ :=
  let initial_points := initial_seashells * seashell_points + initial_starfish * starfish_points
  let points_given_away := (seashells_given_tim + seashells_given_lily) * seashell_points
  let net_points_found_lost := (seashells_found - seashells_lost) * seashell_points
  initial_points - points_given_away + net_points_found_lost

theorem jasons_points_theorem :
  jasons_total_points 49 48 2 3 13 7 15 5 = 222 := by
  sorry

end NUMINAMATH_CALUDE_jasons_points_theorem_l1356_135662


namespace NUMINAMATH_CALUDE_six_valid_configurations_l1356_135621

/-- Represents a square piece of the figure -/
structure Square :=
  (label : Char)

/-- Represents the T-shaped figure -/
structure TShape :=
  (squares : Fin 4 → Square)

/-- Represents the set of additional squares -/
structure AdditionalSquares :=
  (squares : Fin 8 → Square)

/-- Represents a configuration of the figure with an additional square -/
structure Configuration :=
  (base : TShape)
  (additional : Square)

/-- Predicate to check if a configuration can be folded into a cubical box -/
def is_foldable (c : Configuration) : Prop :=
  sorry

/-- The main theorem stating that there are exactly 6 valid configurations -/
theorem six_valid_configurations (t : TShape) (extras : AdditionalSquares) :
  ∃! (valid : Finset Configuration),
    (∀ c ∈ valid, is_foldable c) ∧
    (∀ c : Configuration, c ∈ valid ↔ is_foldable c) ∧
    (valid.card = 6) :=
  sorry

end NUMINAMATH_CALUDE_six_valid_configurations_l1356_135621


namespace NUMINAMATH_CALUDE_factor_calculation_l1356_135636

theorem factor_calculation : ∃ (f : ℚ), 
  let initial_number := 10
  let doubled_plus_eight := 2 * initial_number + 8
  f * doubled_plus_eight = 84 ∧ f = 3 := by sorry

end NUMINAMATH_CALUDE_factor_calculation_l1356_135636


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l1356_135678

theorem arithmetic_mean_geq_geometric_mean {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l1356_135678


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1356_135680

theorem sum_of_squares_of_roots : 
  ∃ (r s t : ℝ), 
    (∀ x : ℝ, x ≥ 0 → (x * Real.sqrt x - 8 * x + 9 * Real.sqrt x - 2 = 0 ↔ x = r ∨ x = s ∨ x = t)) →
    r ≥ 0 ∧ s ≥ 0 ∧ t ≥ 0 →
    r^2 + s^2 + t^2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1356_135680
