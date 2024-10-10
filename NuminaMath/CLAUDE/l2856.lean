import Mathlib

namespace parallel_vectors_magnitude_l2856_285622

/-- Given two vectors a and b in ℝ³, where a is (1,1,2) and b is (2,x,y),
    and a is parallel to b, prove that the magnitude of b is 2√6. -/
theorem parallel_vectors_magnitude (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (1, 1, 2)
  let b : ℝ × ℝ × ℝ := (2, x, y)
  (∃ (k : ℝ), b.1 = k * a.1 ∧ b.2.1 = k * a.2.1 ∧ b.2.2 = k * a.2.2) →
  ‖(b.1, b.2.1, b.2.2)‖ = 2 * Real.sqrt 6 := by
  sorry

end parallel_vectors_magnitude_l2856_285622


namespace direct_square_variation_problem_l2856_285607

/-- A function representing direct variation with the square of x -/
def direct_square_variation (k : ℝ) (x : ℝ) : ℝ := k * x^2

theorem direct_square_variation_problem (y : ℝ → ℝ) :
  (∃ k : ℝ, ∀ x, y x = direct_square_variation k x) →
  y 3 = 18 →
  y 6 = 72 := by sorry

end direct_square_variation_problem_l2856_285607


namespace total_handshakes_at_gathering_l2856_285609

def number_of_couples : ℕ := 15

def men_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

def men_women_handshakes (n : ℕ) : ℕ := n * (n - 1)

def women_subset_handshakes : ℕ := 3

theorem total_handshakes_at_gathering :
  men_handshakes number_of_couples +
  men_women_handshakes number_of_couples +
  women_subset_handshakes = 318 := by sorry

end total_handshakes_at_gathering_l2856_285609


namespace not_p_or_q_false_implies_p_or_q_true_l2856_285657

theorem not_p_or_q_false_implies_p_or_q_true (p q : Prop) :
  ¬(¬(p ∨ q)) → (p ∨ q) := by
  sorry

end not_p_or_q_false_implies_p_or_q_true_l2856_285657


namespace difference_of_squares_502_498_l2856_285655

theorem difference_of_squares_502_498 : 502^2 - 498^2 = 4000 := by
  sorry

end difference_of_squares_502_498_l2856_285655


namespace sum_of_remaining_numbers_l2856_285666

theorem sum_of_remaining_numbers
  (n : ℕ)
  (total_sum : ℝ)
  (subset_sum : ℝ)
  (h1 : n = 5)
  (h2 : total_sum / n = 20)
  (h3 : subset_sum / 2 = 26) :
  total_sum - subset_sum = 48 := by
  sorry

end sum_of_remaining_numbers_l2856_285666


namespace divisibility_implication_l2856_285616

theorem divisibility_implication (a b : ℤ) : (17 ∣ (2*a + 3*b)) → (17 ∣ (9*a + 5*b)) := by
  sorry

end divisibility_implication_l2856_285616


namespace shaded_cubes_count_l2856_285626

/-- Represents a 4x4x4 cube with shaded faces -/
structure ShadedCube where
  /-- The number of smaller cubes along each edge of the large cube -/
  size : Nat
  /-- The number of shaded cubes in the central area of each face -/
  centralShaded : Nat
  /-- The number of shaded corner cubes per face -/
  cornerShaded : Nat

/-- Calculates the total number of uniquely shaded cubes -/
def totalShadedCubes (cube : ShadedCube) : Nat :=
  sorry

/-- Theorem stating that the total number of shaded cubes is 16 -/
theorem shaded_cubes_count (cube : ShadedCube) 
  (h1 : cube.size = 4)
  (h2 : cube.centralShaded = 4)
  (h3 : cube.cornerShaded = 1) : 
  totalShadedCubes cube = 16 := by
  sorry

end shaded_cubes_count_l2856_285626


namespace kevin_bought_three_muffins_l2856_285667

/-- The number of muffins Kevin bought -/
def num_muffins : ℕ := 3

/-- The cost of juice in dollars -/
def juice_cost : ℚ := 145/100

/-- The total amount paid in dollars -/
def total_paid : ℚ := 370/100

/-- The cost of each muffin in dollars -/
def muffin_cost : ℚ := 75/100

/-- Theorem stating that the number of muffins Kevin bought is 3 -/
theorem kevin_bought_three_muffins :
  num_muffins = 3 ∧
  juice_cost + (num_muffins : ℚ) * muffin_cost = total_paid :=
sorry

end kevin_bought_three_muffins_l2856_285667


namespace zoo_visitors_l2856_285645

/-- Proves that the number of adults who went to the zoo is 51, given the total number of people,
    ticket prices, and total sales. -/
theorem zoo_visitors (total_people : ℕ) (adult_price kid_price : ℕ) (total_sales : ℕ)
    (h_total : total_people = 254)
    (h_adult_price : adult_price = 28)
    (h_kid_price : kid_price = 12)
    (h_sales : total_sales = 3864) :
    ∃ (adults : ℕ), adults = 51 ∧
    ∃ (kids : ℕ), adults + kids = total_people ∧
    adult_price * adults + kid_price * kids = total_sales :=
  sorry

end zoo_visitors_l2856_285645


namespace range_of_x_l2856_285694

def p (x : ℝ) : Prop := x^2 - 5*x + 6 ≥ 0

def q (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) :
  (∀ x, p x ∨ q x) → (∀ x, ¬q x) → x ≤ 0 ∨ x ≥ 4 := by
  sorry

end range_of_x_l2856_285694


namespace right_triangle_30_60_90_properties_l2856_285620

/-- A right triangle with one leg of 15 inches and the angle opposite that leg being 30° --/
structure RightTriangle30_60_90 where
  /-- The length of one leg of the triangle --/
  leg : ℝ
  /-- The angle opposite the given leg in radians --/
  angle : ℝ
  /-- The triangle is a right triangle --/
  is_right_triangle : leg > 0
  /-- The length of the given leg is 15 inches --/
  leg_length : leg = 15
  /-- The angle opposite the given leg is 30° (π/6 radians) --/
  angle_measure : angle = π / 6

/-- The length of the hypotenuse in the given right triangle --/
def hypotenuse_length (t : RightTriangle30_60_90) : ℝ := 30

/-- The length of the altitude from the hypotenuse to the right angle in the given triangle --/
def altitude_length (t : RightTriangle30_60_90) : ℝ := 22.5

/-- Theorem stating the length of the hypotenuse and altitude in the given right triangle --/
theorem right_triangle_30_60_90_properties (t : RightTriangle30_60_90) :
  hypotenuse_length t = 30 ∧ altitude_length t = 22.5 := by
  sorry

end right_triangle_30_60_90_properties_l2856_285620


namespace complex_distance_and_midpoint_l2856_285623

/-- Given two complex numbers, prove the distance between them and their midpoint -/
theorem complex_distance_and_midpoint (z1 z2 : ℂ) 
  (hz1 : z1 = 3 + 4*I) (hz2 : z2 = -2 - 3*I) : 
  Complex.abs (z1 - z2) = Real.sqrt 74 ∧ 
  (z1 + z2) / 2 = (1/2 : ℂ) + (1/2 : ℂ)*I := by
  sorry

end complex_distance_and_midpoint_l2856_285623


namespace gcd_2024_1728_l2856_285617

theorem gcd_2024_1728 : Nat.gcd 2024 1728 = 8 := by sorry

end gcd_2024_1728_l2856_285617


namespace geometric_progression_common_ratio_l2856_285625

theorem geometric_progression_common_ratio
  (q : ℝ)
  (h1 : |q| < 1)
  (h2 : ∀ (a : ℝ), a ≠ 0 → a = 4 * (a / (1 - q) - a)) :
  q = 1/5 := by
sorry

end geometric_progression_common_ratio_l2856_285625


namespace unique_modular_equivalent_in_range_l2856_285671

theorem unique_modular_equivalent_in_range : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [ZMOD 10] := by
  sorry

end unique_modular_equivalent_in_range_l2856_285671


namespace system_solution_l2856_285601

theorem system_solution : ∃ (x y z : ℝ), 
  (x^2 - y - z = 8) ∧ 
  (4*x + y^2 + 3*z = -11) ∧ 
  (2*x - 3*y + z^2 = -11) ∧ 
  (x = -3) ∧ (y = 2) ∧ (z = -1) := by
  sorry

end system_solution_l2856_285601


namespace acute_angle_range_characterization_l2856_285682

/-- The angle between two vectors is acute if and only if their dot product is positive and they are not collinear -/
def is_acute_angle (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2 > 0) ∧ (a.1 * b.2 ≠ a.2 * b.1)

/-- The set of real numbers m for which the angle between vectors a and b is acute -/
def acute_angle_range : Set ℝ :=
  {m | is_acute_angle (m - 2, m + 3) (2*m + 1, m - 2)}

theorem acute_angle_range_characterization :
  acute_angle_range = {m | m > 2 ∨ (m < (-11 - 5*Real.sqrt 5) / 2) ∨ 
    (((-11 + 5*Real.sqrt 5) / 2 < m) ∧ (m < -4/3))} := by
  sorry

end acute_angle_range_characterization_l2856_285682


namespace strawberry_milk_probability_l2856_285603

theorem strawberry_milk_probability : 
  let n : ℕ := 7  -- number of trials
  let k : ℕ := 5  -- number of successes
  let p : ℚ := 3/4  -- probability of success in each trial
  Nat.choose n k * p^k * (1-p)^(n-k) = 5103/16384 := by
  sorry

end strawberry_milk_probability_l2856_285603


namespace pythagorean_theorem_special_case_l2856_285680

/-- A right triangle with legs of lengths 1 and 2 -/
structure RightTriangle :=
  (leg1 : ℝ)
  (leg2 : ℝ)
  (is_right : leg1 = 1 ∧ leg2 = 2)

/-- The square of the hypotenuse of a right triangle -/
def hypotenuse_squared (t : RightTriangle) : ℝ :=
  t.leg1^2 + t.leg2^2

/-- Theorem: The square of the hypotenuse of a right triangle with legs 1 and 2 is 5 -/
theorem pythagorean_theorem_special_case (t : RightTriangle) :
  hypotenuse_squared t = 5 := by
  sorry

end pythagorean_theorem_special_case_l2856_285680


namespace gcd_180_294_l2856_285660

theorem gcd_180_294 : Nat.gcd 180 294 = 6 := by
  sorry

end gcd_180_294_l2856_285660


namespace triangle_cosine_theorem_l2856_285613

theorem triangle_cosine_theorem (X Y Z : Real) :
  -- Triangle XYZ
  X + Y + Z = Real.pi →
  -- sin X = 4/5
  Real.sin X = 4/5 →
  -- cos Y = 12/13
  Real.cos Y = 12/13 →
  -- Then cos Z = -16/65
  Real.cos Z = -16/65 := by
  sorry

end triangle_cosine_theorem_l2856_285613


namespace greatest_sum_consecutive_even_integers_l2856_285649

theorem greatest_sum_consecutive_even_integers (n : ℕ) :
  n % 2 = 0 →  -- n is even
  n * (n + 2) < 800 →  -- product is less than 800
  ∀ m : ℕ, m % 2 = 0 →  -- for all even m
    m * (m + 2) < 800 →  -- whose product with its consecutive even is less than 800
    n + (n + 2) ≥ m + (m + 2) →  -- n and n+2 have the greatest sum
  n + (n + 2) = 54  -- the greatest sum is 54
:= by sorry

end greatest_sum_consecutive_even_integers_l2856_285649


namespace savings_calculation_l2856_285688

theorem savings_calculation (income expenditure : ℕ) 
  (h1 : income = 36000)
  (h2 : income * 8 = expenditure * 9) : 
  income - expenditure = 4000 :=
sorry

end savings_calculation_l2856_285688


namespace function_identity_l2856_285651

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : 
  ∀ n : ℕ+, f n = n := by
  sorry

end function_identity_l2856_285651


namespace solution_negative_l2856_285691

-- Define the equation
def equation (x a : ℝ) : Prop :=
  (x - 1) / (x - 2) - (x - 2) / (x + 1) = (2 * x + a) / ((x - 2) * (x + 1))

-- Define the theorem
theorem solution_negative (a : ℝ) :
  (∃ x : ℝ, equation x a ∧ x < 0) ↔ (a < -5 ∧ a ≠ -7) :=
sorry

end solution_negative_l2856_285691


namespace quadratic_inequality_solution_condition_l2856_285636

theorem quadratic_inequality_solution_condition (k : ℝ) : 
  (k > 0) → 
  (∃ x : ℝ, x^2 - 8*x + k < 0) ↔ 
  (k > 0 ∧ k < 16) :=
by sorry

end quadratic_inequality_solution_condition_l2856_285636


namespace complement_union_equals_d_l2856_285653

universe u

def U : Set (Fin 4) := {0, 1, 2, 3}
def A : Set (Fin 4) := {0, 1}
def B : Set (Fin 4) := {2}

theorem complement_union_equals_d : 
  (U \ (A ∪ B)) = {3} := by sorry

end complement_union_equals_d_l2856_285653


namespace rectangular_plot_dimensions_l2856_285685

theorem rectangular_plot_dimensions (length breadth : ℝ) : 
  length = 55 →
  breadth + (length - breadth) = length →
  4 * breadth + 2 * (length - breadth) = 5300 / 26.5 →
  length - breadth = 10 := by
sorry

end rectangular_plot_dimensions_l2856_285685


namespace negation_of_proposition_l2856_285634

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x + 3 ≤ 0) :=
by sorry

end negation_of_proposition_l2856_285634


namespace blown_out_dune_probability_l2856_285640

/-- The probability that a sand dune remains after being formed -/
def prob_dune_remains : ℚ := 1 / 3

/-- The probability that a blown-out sand dune contains treasure -/
def prob_treasure : ℚ := 1 / 5

/-- The probability that a formed sand dune has a lucky coupon -/
def prob_lucky_coupon : ℚ := 2 / 3

/-- The probability that a blown-out sand dune contains both treasure and a lucky coupon -/
def prob_both : ℚ := prob_treasure * prob_lucky_coupon

theorem blown_out_dune_probability : prob_both = 2 / 15 := by
  sorry

end blown_out_dune_probability_l2856_285640


namespace seating_arrangement_l2856_285693

/-- The number of students per row and total number of students in a seating arrangement problem. -/
theorem seating_arrangement (S R : ℕ) 
  (h1 : S = 5 * R + 6)  -- When 5 students sit in a row, 6 are left without seats
  (h2 : S = 12 * (R - 3))  -- When 12 students sit in a row, 3 rows are empty
  : R = 6 ∧ S = 36 := by
  sorry

end seating_arrangement_l2856_285693


namespace investment_interest_proof_l2856_285624

/-- Calculates the total annual interest for a two-fund investment --/
def total_annual_interest (total_investment : ℝ) (rate1 rate2 : ℝ) (amount_in_fund1 : ℝ) : ℝ :=
  let amount_in_fund2 := total_investment - amount_in_fund1
  let interest1 := amount_in_fund1 * rate1
  let interest2 := amount_in_fund2 * rate2
  interest1 + interest2

/-- Proves that the total annual interest for the given investment scenario is $4,120 --/
theorem investment_interest_proof :
  total_annual_interest 50000 0.08 0.085 26000 = 4120 := by
  sorry

end investment_interest_proof_l2856_285624


namespace grinder_price_correct_l2856_285669

/-- The purchase price of the grinder -/
def grinder_price : ℝ := 15000

/-- The purchase price of the mobile phone -/
def mobile_price : ℝ := 8000

/-- The selling price of the grinder -/
def grinder_sell_price : ℝ := 0.98 * grinder_price

/-- The selling price of the mobile phone -/
def mobile_sell_price : ℝ := 1.1 * mobile_price

/-- The total profit -/
def total_profit : ℝ := 500

theorem grinder_price_correct :
  grinder_sell_price + mobile_sell_price = grinder_price + mobile_price + total_profit :=
by sorry

end grinder_price_correct_l2856_285669


namespace solve_problem_l2856_285659

-- Define the sets A and B as functions of m
def A (m : ℤ) : Set ℤ := {-4, 2*m-1, m^2}
def B (m : ℤ) : Set ℤ := {9, m-5, 1-m}

-- Define the universal set U
def U : Set ℤ := Set.univ

-- State the theorem
theorem solve_problem (m : ℤ) 
  (h_intersection : A m ∩ B m = {9}) : 
  m = -3 ∧ A m ∩ (U \ B m) = {-4, -7} := by
  sorry


end solve_problem_l2856_285659


namespace circumcircle_diameter_perpendicular_to_DK_l2856_285618

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0

-- Define the altitude CD
def altitude (A B C D : ℝ × ℝ) : Prop :=
  (D.1 - C.1) * (B.1 - A.1) + (D.2 - C.2) * (B.2 - A.2) = 0

-- Define point K such that |AK| = |AC|
def point_K (A C K : ℝ × ℝ) : Prop :=
  (K.1 - A.1)^2 + (K.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

-- Define the circumcircle of triangle ABK
def circumcircle (A B K O : ℝ × ℝ) : Prop :=
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (B.1 - O.1)^2 + (B.2 - O.2)^2 ∧
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (K.1 - O.1)^2 + (K.2 - O.2)^2

-- Define perpendicularity
def perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (S.1 - R.1) + (Q.2 - P.2) * (S.2 - R.2) = 0

-- Theorem statement
theorem circumcircle_diameter_perpendicular_to_DK 
  (A B C D K O : ℝ × ℝ) 
  (h1 : Triangle A B C)
  (h2 : altitude A B C D)
  (h3 : point_K A C K)
  (h4 : circumcircle A B K O) :
  perpendicular A O D K :=
sorry

end circumcircle_diameter_perpendicular_to_DK_l2856_285618


namespace consecutive_even_numbers_sum_l2856_285610

theorem consecutive_even_numbers_sum (x : ℤ) : 
  (x % 2 = 0) →                   -- x is even
  (x + (x + 2) + (x + 4) = 1194) →  -- sum of three consecutive even numbers is 1194
  x = 396 :=                      -- the first even number is 396
by
  sorry

end consecutive_even_numbers_sum_l2856_285610


namespace zoo_escape_zoo_escape_proof_l2856_285644

theorem zoo_escape (lions : ℕ) (recovery_time : ℕ) (total_time : ℕ) : ℕ :=
  let rhinos := (total_time / recovery_time) - lions
  rhinos

theorem zoo_escape_proof :
  zoo_escape 3 2 10 = 2 := by
  sorry

end zoo_escape_zoo_escape_proof_l2856_285644


namespace rectangular_box_surface_area_l2856_285696

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 180) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 1400 := by
  sorry

end rectangular_box_surface_area_l2856_285696


namespace base_conversion_problem_l2856_285611

theorem base_conversion_problem : ∃! (n : ℕ), ∃ (A C : ℕ), 
  (A < 8 ∧ C < 8) ∧
  (A < 6 ∧ C < 6) ∧
  (n = 8 * A + C) ∧
  (n = 6 * C + A) ∧
  (n = 47) := by
sorry

end base_conversion_problem_l2856_285611


namespace rectangle_triangle_length_l2856_285648

/-- Given a rectangle PQRS with PQ = 4 cm, QR = 10 cm, and PM = MQ,
    if the area of triangle PMQ is half the area of rectangle PQRS,
    then the length of segment MQ is 2√10 cm. -/
theorem rectangle_triangle_length (P Q R S M : ℝ × ℝ) : 
  let pq := dist P Q
  let qr := dist Q R
  let pm := dist P M
  let mq := dist M Q
  let area_rect := pq * qr
  let area_tri := (1/2) * pm * mq
  pq = 4 →
  qr = 10 →
  pm = mq →
  area_tri = (1/2) * area_rect →
  mq = 2 * Real.sqrt 10 := by
  sorry

end rectangle_triangle_length_l2856_285648


namespace paint_calculation_l2856_285664

/-- Given three people painting a wall with a work ratio and total area,
    calculate the area painted by the third person. -/
theorem paint_calculation (ratio_a ratio_b ratio_c total_area : ℕ) 
    (ratio_positive : ratio_a > 0 ∧ ratio_b > 0 ∧ ratio_c > 0)
    (total_positive : total_area > 0) :
    let total_ratio := ratio_a + ratio_b + ratio_c
    ratio_c * total_area / total_ratio = 60 :=
by
  sorry

end paint_calculation_l2856_285664


namespace unique_solution_system_l2856_285631

theorem unique_solution_system : 
  ∃! (x y : ℝ), (x + y = (7 - x) + (7 - y)) ∧ (x - y = (x - 2) + (y - 2)) ∧ x = 5 ∧ y = 2 := by
  sorry

end unique_solution_system_l2856_285631


namespace opponent_score_l2856_285661

/-- Given UF's previous game scores and championship game performance, 
    calculate their opponent's score. -/
theorem opponent_score (total_points : ℕ) (num_games : ℕ) (half_reduction : ℕ) (point_difference : ℕ) : 
  total_points = 720 →
  num_games = 24 →
  half_reduction = 2 →
  point_difference = 2 →
  (total_points / num_games / 2 - half_reduction) - point_difference = 11 := by
  sorry


end opponent_score_l2856_285661


namespace gcd_8917_4273_l2856_285656

theorem gcd_8917_4273 : Nat.gcd 8917 4273 = 1 := by
  sorry

end gcd_8917_4273_l2856_285656


namespace cut_length_of_divided_square_cake_l2856_285699

/-- Represents a square cake divided into four equal pieces -/
structure DividedSquareCake where
  side_length : ℝ
  cut_length : ℝ

/-- The perimeter of the original square cake -/
def square_perimeter (cake : DividedSquareCake) : ℝ :=
  4 * cake.side_length

/-- The perimeter of each piece after division -/
def piece_perimeter (cake : DividedSquareCake) : ℝ :=
  2 * cake.side_length + 2 * cake.cut_length

/-- Theorem: The length of each cut in a divided square cake -/
theorem cut_length_of_divided_square_cake :
  ∀ (cake : DividedSquareCake),
    square_perimeter cake = 100 →
    piece_perimeter cake = 56 →
    cake.cut_length = 3 :=
by sorry

end cut_length_of_divided_square_cake_l2856_285699


namespace gcd_lcm_2970_1722_l2856_285628

theorem gcd_lcm_2970_1722 : 
  (Nat.gcd 2970 1722 = 6) ∧ (Nat.lcm 2970 1722 = 856170) := by
  sorry

end gcd_lcm_2970_1722_l2856_285628


namespace set_intersection_theorem_l2856_285641

def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem set_intersection_theorem :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end set_intersection_theorem_l2856_285641


namespace regular_polygon_perimeter_l2856_285638

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧
  side_length = 7 ∧
  exterior_angle = 90 ∧
  (360 : ℝ) / n = exterior_angle →
  n * side_length = 28 := by
  sorry

end regular_polygon_perimeter_l2856_285638


namespace intersection_M_N_l2856_285684

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l2856_285684


namespace faye_pencils_l2856_285643

/-- The number of rows of pencils and crayons -/
def num_rows : ℕ := 30

/-- The number of pencils in each row -/
def pencils_per_row : ℕ := 24

/-- The total number of pencils -/
def total_pencils : ℕ := num_rows * pencils_per_row

theorem faye_pencils : total_pencils = 720 := by
  sorry

end faye_pencils_l2856_285643


namespace average_age_increase_l2856_285686

theorem average_age_increase (n : ℕ) (m : ℕ) (avg_29 : ℝ) (age_30 : ℕ) :
  n = 30 →
  m = 29 →
  avg_29 = 12 →
  age_30 = 80 →
  let total_29 := m * avg_29
  let new_total := total_29 + age_30
  let new_avg := new_total / n
  abs (new_avg - avg_29 - 2.27) < 0.01 := by
sorry


end average_age_increase_l2856_285686


namespace avg_price_goat_l2856_285662

def num_goats : ℕ := 5
def num_hens : ℕ := 10
def total_cost : ℕ := 2500
def avg_price_hen : ℕ := 50

theorem avg_price_goat :
  (total_cost - num_hens * avg_price_hen) / num_goats = 400 :=
sorry

end avg_price_goat_l2856_285662


namespace cube_difference_eq_108_l2856_285670

/-- Given two real numbers x and y, if x - y = 3 and x^2 + y^2 = 27, then x^3 - y^3 = 108 -/
theorem cube_difference_eq_108 (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) :
  x^3 - y^3 = 108 := by
  sorry

end cube_difference_eq_108_l2856_285670


namespace count_valid_integers_eq_44_l2856_285683

def digit_set : List Nat := [2, 3, 5, 5, 6, 6, 6]

def is_valid_integer (n : Nat) : Bool :=
  let digits := n.digits 10
  digits.length == 3 ∧ 
  digits.all (λ d => d ∈ digit_set) ∧
  digits.count 2 ≤ 1 ∧
  digits.count 3 ≤ 1 ∧
  digits.count 5 ≤ 2 ∧
  digits.count 6 ≤ 3

def count_valid_integers : Nat :=
  (List.range 900).map (λ n => n + 100)
    |>.filter is_valid_integer
    |>.length

theorem count_valid_integers_eq_44 : count_valid_integers = 44 := by
  sorry

end count_valid_integers_eq_44_l2856_285683


namespace tangent_slope_at_zero_l2856_285642

-- Define the function representing the curve
def f (x : ℝ) : ℝ := -2 * x^2 + 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -4 * x

-- Theorem statement
theorem tangent_slope_at_zero : f' 0 = 0 := by
  sorry

end tangent_slope_at_zero_l2856_285642


namespace work_completion_time_l2856_285633

/-- The time it takes for worker C to complete the work alone -/
def time_C : ℕ := 36

/-- The time it takes for workers A, B, and C to complete the work together -/
def time_ABC : ℕ := 4

/-- The time it takes for worker A to complete the work alone -/
def time_A : ℕ := 6

/-- The time it takes for worker B to complete the work alone -/
def time_B : ℕ := 18

theorem work_completion_time :
  (1 : ℚ) / time_ABC = (1 : ℚ) / time_A + (1 : ℚ) / time_B + (1 : ℚ) / time_C :=
by sorry


end work_completion_time_l2856_285633


namespace inference_is_analogical_l2856_285663

/-- Inductive reasoning is the process of reasoning from specific instances to a general conclusion. -/
def inductive_reasoning : Prop := sorry

/-- Deductive reasoning is the process of reasoning from a general premise to a specific conclusion. -/
def deductive_reasoning : Prop := sorry

/-- Analogical reasoning is the process of reasoning from one specific instance to another specific instance. -/
def analogical_reasoning : Prop := sorry

/-- The inference from "If a > b, then a + c > b + c" to "If a > b, then ac > bc" -/
def inference : Prop := sorry

/-- The inference is an example of analogical reasoning -/
theorem inference_is_analogical : inference → analogical_reasoning := by sorry

end inference_is_analogical_l2856_285663


namespace equation_roots_l2856_285612

/-- The equation a²(x-2) + a(39-20x) + 20 = 0 has at least two distinct roots if and only if a = 20 -/
theorem equation_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20*x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20*y) + 20 = 0) ↔ 
  a = 20 := by
sorry

end equation_roots_l2856_285612


namespace daniel_water_bottles_l2856_285672

/-- The number of bottles Daniel filled for the rugby team -/
def rugby_bottles : ℕ := by sorry

theorem daniel_water_bottles :
  let total_bottles : ℕ := 254
  let football_players : ℕ := 11
  let football_bottles_per_player : ℕ := 6
  let soccer_bottles : ℕ := 53
  let lacrosse_extra_bottles : ℕ := 12
  let coach_bottles : ℕ := 2
  let num_teams : ℕ := 4

  let football_bottles := football_players * football_bottles_per_player
  let lacrosse_bottles := football_bottles + lacrosse_extra_bottles
  let total_coach_bottles := coach_bottles * num_teams

  rugby_bottles = total_bottles - (football_bottles + soccer_bottles + lacrosse_bottles + total_coach_bottles) :=
by sorry

end daniel_water_bottles_l2856_285672


namespace quadratic_always_positive_l2856_285675

theorem quadratic_always_positive (b : ℝ) :
  (∀ x : ℝ, x^2 + b*x + b > 0) ↔ (0 < b ∧ b < 4) := by sorry

end quadratic_always_positive_l2856_285675


namespace shirt_cost_theorem_l2856_285606

theorem shirt_cost_theorem :
  let total_shirts : ℕ := 5
  let cheap_shirts : ℕ := 3
  let expensive_shirts : ℕ := total_shirts - cheap_shirts
  let cheap_price : ℕ := 15
  let expensive_price : ℕ := 20
  
  (cheap_shirts * cheap_price + expensive_shirts * expensive_price : ℕ) = 85
  := by sorry

end shirt_cost_theorem_l2856_285606


namespace intersection_points_count_l2856_285637

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  -- Add necessary fields here
  nonconcurrent : Bool  -- Represents that no three diagonals are concurrent

/-- The number of intersection points of diagonals inside a convex polygon -/
def intersectionPoints (p : ConvexPolygon n) : ℕ := sorry

/-- Theorem: The number of intersection points of diagonals inside a convex n-gon
    where no three diagonals are concurrent is equal to (n choose 4) -/
theorem intersection_points_count (n : ℕ) (p : ConvexPolygon n) 
    (h : p.nonconcurrent = true) : 
  intersectionPoints p = Nat.choose n 4 := by sorry

end intersection_points_count_l2856_285637


namespace number_of_cows_bought_l2856_285621

/-- Prove that the number of cows bought is 2 -/
theorem number_of_cows_bought (total_cost : ℕ) (num_goats : ℕ) (goat_price : ℕ) (cow_price : ℕ) :
  total_cost = 1400 →
  num_goats = 8 →
  goat_price = 60 →
  cow_price = 460 →
  (total_cost - num_goats * goat_price) / cow_price = 2 := by
  sorry

#check number_of_cows_bought

end number_of_cows_bought_l2856_285621


namespace dot_product_equals_four_l2856_285602

def a : ℝ × ℝ := (1, 2)

theorem dot_product_equals_four (b : ℝ × ℝ) 
  (h : (2 • a) - b = (4, 1)) : a • b = 4 := by
  sorry

end dot_product_equals_four_l2856_285602


namespace negation_equivalence_l2856_285646

theorem negation_equivalence :
  (¬ (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end negation_equivalence_l2856_285646


namespace fraction_sum_l2856_285608

theorem fraction_sum : (1 : ℚ) / 4 + 2 / 9 + 3 / 6 = 35 / 36 := by sorry

end fraction_sum_l2856_285608


namespace incircle_tangent_inequality_l2856_285627

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the incircle points
variable (A₁ B₁ : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h_triangle : Triangle A B C)
variable (h_incircle : IsIncircle A₁ B₁ A B C)
variable (h_AC_gt_BC : dist A C > dist B C)

-- State the theorem
theorem incircle_tangent_inequality :
  dist A A₁ > dist B B₁ := by sorry

end incircle_tangent_inequality_l2856_285627


namespace max_sum_constrained_l2856_285687

theorem max_sum_constrained (x y : ℝ) 
  (h1 : 5 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 5 * y = 15) : 
  x + y ≤ 47 / 16 := by
  sorry

end max_sum_constrained_l2856_285687


namespace solve_linear_equation_l2856_285654

theorem solve_linear_equation (x : ℝ) : 3*x - 5*x + 8*x = 240 → x = 40 := by
  sorry

end solve_linear_equation_l2856_285654


namespace sqrt_equation_solutions_l2856_285605

theorem sqrt_equation_solutions :
  {x : ℝ | Real.sqrt (3 * x^2 + 2 * x + 1) = 3} = {4/3, -2} := by
  sorry

end sqrt_equation_solutions_l2856_285605


namespace lines_parallel_iff_m_eq_one_l2856_285697

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b ∧ l1.a * l2.c ≠ l2.a * l1.c

/-- The first line: x + (1+m)y = 2-m -/
def line1 (m : ℝ) : Line :=
  { a := 1, b := 1 + m, c := m - 2 }

/-- The second line: 2mx + 4y = -16 -/
def line2 (m : ℝ) : Line :=
  { a := 2 * m, b := 4, c := 16 }

/-- The theorem stating that the lines are parallel iff m = 1 -/
theorem lines_parallel_iff_m_eq_one :
  ∀ m : ℝ, parallel (line1 m) (line2 m) ↔ m = 1 := by
  sorry

end lines_parallel_iff_m_eq_one_l2856_285697


namespace one_third_minus_decimal_l2856_285665

theorem one_third_minus_decimal : (1 : ℚ) / 3 - 333 / 1000 = 1 / (3 * 1000) := by sorry

end one_third_minus_decimal_l2856_285665


namespace lucky_sock_pairs_l2856_285692

/-- The probability of all pairs being lucky given n pairs of socks --/
def prob_all_lucky (n : ℕ) : ℚ :=
  (2^n * n.factorial) / (2*n).factorial

/-- The expected number of lucky pairs given n pairs of socks --/
def expected_lucky_pairs (n : ℕ) : ℚ :=
  n / (2*n - 1)

/-- Theorem stating the properties of lucky sock pairs --/
theorem lucky_sock_pairs (n : ℕ) (h : n > 0) : 
  prob_all_lucky n = (2^n * n.factorial) / (2*n).factorial ∧ 
  expected_lucky_pairs n > 1/2 := by
  sorry

#check lucky_sock_pairs

end lucky_sock_pairs_l2856_285692


namespace circle_Q_equation_no_perpendicular_bisector_l2856_285650

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 0)

-- Define line l₁ passing through P and intersecting circle C at M and N
def line_l₁ (x y : ℝ) : Prop := ∃ (t : ℝ), x = 2 + t ∧ y = t ∧ circle_C x y

-- Define the length of MN
def MN_length : ℝ := 4

-- Define line ax - y + 1 = 0
def line_AB (a x y : ℝ) : Prop := a*x - y + 1 = 0

-- Theorem 1: Equation of circle Q
theorem circle_Q_equation : 
  ∀ x y : ℝ, (∃ M N : ℝ × ℝ, line_l₁ M.1 M.2 ∧ line_l₁ N.1 N.2 ∧ 
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = MN_length^2) →
  ((x - 2)^2 + y^2 = 4) := 
sorry

-- Theorem 2: Non-existence of a
theorem no_perpendicular_bisector :
  ¬ ∃ a : ℝ, ∀ A B : ℝ × ℝ, 
    (line_AB a A.1 A.2 ∧ circle_C A.1 A.2 ∧ 
     line_AB a B.1 B.2 ∧ circle_C B.1 B.2 ∧ A ≠ B) →
    (∃ l₂ : ℝ → ℝ → Prop, 
      l₂ point_P.1 point_P.2 ∧
      l₂ ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧
      (B.2 - A.2) * (point_P.1 - A.1) = (point_P.2 - A.2) * (B.1 - A.1)) :=
sorry

end circle_Q_equation_no_perpendicular_bisector_l2856_285650


namespace reciprocal_of_negative_2023_l2856_285600

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by
  sorry

end reciprocal_of_negative_2023_l2856_285600


namespace problem_solution_l2856_285632

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- State the theorem
theorem problem_solution (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, ¬(p x a) → ¬(q x)) 
  (h3 : ∃ x, ¬(p x a) ∧ (q x)) :
  (a = 1 → ∃ x, x > 2 ∧ x < 3 ∧ p x a ∧ q x) ∧
  (a > 1 ∧ a ≤ 2) :=
sorry

end problem_solution_l2856_285632


namespace terminating_decimal_expansion_13_200_l2856_285681

theorem terminating_decimal_expansion_13_200 : 
  ∃ (n : ℕ) (a : ℤ), (13 : ℚ) / 200 = (a : ℚ) / (10 ^ n) ∧ (a : ℚ) / (10 ^ n) = 0.052 :=
by
  sorry

end terminating_decimal_expansion_13_200_l2856_285681


namespace perfect_cube_units_digits_l2856_285635

theorem perfect_cube_units_digits :
  ∀ d : Fin 10, ∃ n : ℤ, (n^3) % 10 = d.val :=
by sorry

end perfect_cube_units_digits_l2856_285635


namespace total_balls_l2856_285698

def ball_count (red blue yellow : ℕ) : Prop :=
  red + blue + yellow > 0 ∧ 2 * blue = 3 * red ∧ 4 * red = 2 * yellow

theorem total_balls (red blue yellow : ℕ) :
  ball_count red blue yellow → yellow = 40 → red + blue + yellow = 90 := by
  sorry

end total_balls_l2856_285698


namespace problem_solution_l2856_285615

theorem problem_solution (a b : ℤ) : 
  (5 + a = 6 - b) → (6 + b = 9 + a) → (5 - a = 6) := by
  sorry

end problem_solution_l2856_285615


namespace jack_book_pages_l2856_285639

/-- Calculates the total number of pages in a book given the daily reading rate and the number of days to finish. -/
def total_pages (pages_per_day : ℕ) (days_to_finish : ℕ) : ℕ :=
  pages_per_day * days_to_finish

/-- Proves that the book Jack is reading has 299 pages. -/
theorem jack_book_pages :
  let pages_per_day : ℕ := 23
  let days_to_finish : ℕ := 13
  total_pages pages_per_day days_to_finish = 299 := by
  sorry

end jack_book_pages_l2856_285639


namespace stick_pieces_l2856_285614

def stick_length : ℕ := 60

def marks_10 : List ℕ := [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
def marks_12 : List ℕ := [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
def marks_15 : List ℕ := [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]

def all_marks : List ℕ := marks_10 ++ marks_12 ++ marks_15

theorem stick_pieces : 
  (all_marks.toFinset.card) + 1 = 28 := by sorry

end stick_pieces_l2856_285614


namespace eldorado_license_plates_l2856_285668

/-- The number of vowels that can be used as the first letter of a license plate. -/
def numVowels : ℕ := 5

/-- The number of letters in the alphabet. -/
def numLetters : ℕ := 26

/-- The number of digits (0-9). -/
def numDigits : ℕ := 10

/-- The total number of valid license plates in Eldorado. -/
def totalLicensePlates : ℕ := numVowels * numLetters * numLetters * numDigits * numDigits

theorem eldorado_license_plates :
  totalLicensePlates = 338000 :=
by sorry

end eldorado_license_plates_l2856_285668


namespace sequence_sum_expression_l2856_285674

/-- Given a sequence {a_n} with sum of first n terms S_n, where a_1 = 1 and S_n = 2a_{n+1},
    prove that S_n = (3/2)^(n-1) for n > 1 -/
theorem sequence_sum_expression (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  a 1 = 1 →
  (∀ k, S k = 2 * a (k + 1)) →
  n > 1 →
  S n = (3/2)^(n-1) := by
sorry

end sequence_sum_expression_l2856_285674


namespace special_determinant_l2856_285630

open Matrix

/-- The determinant of an n×n matrix with diagonal elements b and all other elements a
    is equal to [b+(n-1)a](b-a)^(n-1) -/
theorem special_determinant (n : ℕ) (a b : ℝ) :
  let M : Matrix (Fin n) (Fin n) ℝ := λ i j => if i = j then b else a
  det M = (b + (n - 1) * a) * (b - a) ^ (n - 1) := by
  sorry

end special_determinant_l2856_285630


namespace absolute_value_of_complex_product_l2856_285695

open Complex

theorem absolute_value_of_complex_product : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + i) * (1 + 3*i)
  Complex.abs z = 2 * Real.sqrt 5 := by sorry

end absolute_value_of_complex_product_l2856_285695


namespace sum_of_solutions_eq_zero_l2856_285652

theorem sum_of_solutions_eq_zero : 
  ∃ (S : Finset ℤ), (∀ x ∈ S, x^4 - 13*x^2 + 36 = 0) ∧ 
                    (∀ x : ℤ, x^4 - 13*x^2 + 36 = 0 → x ∈ S) ∧ 
                    (S.sum id = 0) := by
  sorry

end sum_of_solutions_eq_zero_l2856_285652


namespace lcm_of_prime_and_nonmultiple_lcm_1227_40_l2856_285658

theorem lcm_of_prime_and_nonmultiple (p n : ℕ) (h_prime : Nat.Prime p) (h_not_dvd : ¬p ∣ n) :
  Nat.lcm p n = p * n :=
by sorry

theorem lcm_1227_40 :
  Nat.lcm 1227 40 = 49080 :=
by sorry

end lcm_of_prime_and_nonmultiple_lcm_1227_40_l2856_285658


namespace sum_six_consecutive_integers_l2856_285679

theorem sum_six_consecutive_integers (m : ℤ) : 
  m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5) = 6 * m + 15 := by
  sorry

end sum_six_consecutive_integers_l2856_285679


namespace vector_magnitude_proof_l2856_285673

/-- Given two vectors HK and AE in a vector space, prove that if HK = 1/4 * AE and 
    the magnitude of 4 * HK is 4.8, then the magnitude of AE is 4.8. -/
theorem vector_magnitude_proof 
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (HK AE : V) 
  (h1 : HK = (1/4 : ℝ) • AE) 
  (h2 : ‖(4 : ℝ) • HK‖ = 4.8) : 
  ‖AE‖ = 4.8 := by
  sorry

#check vector_magnitude_proof

end vector_magnitude_proof_l2856_285673


namespace function_domain_range_l2856_285689

theorem function_domain_range (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (m * x^2 + m * x + 1)) ↔ 0 ≤ m ∧ m ≤ 4 := by
  sorry

end function_domain_range_l2856_285689


namespace geometric_sequence_ratio_l2856_285690

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  sum_condition : a 3 + a 5 = 8
  product_condition : a 1 * a 5 = 4

/-- The ratio of the 13th term to the 9th term is 9 -/
theorem geometric_sequence_ratio
  (seq : GeometricSequence) :
  seq.a 13 / seq.a 9 = 9 := by
  sorry

end geometric_sequence_ratio_l2856_285690


namespace students_not_participating_l2856_285678

theorem students_not_participating (total : ℕ) (football : ℕ) (tennis : ℕ) (basketball : ℕ)
  (football_tennis : ℕ) (football_basketball : ℕ) (tennis_basketball : ℕ) (all_three : ℕ) :
  total = 50 →
  football = 30 →
  tennis = 25 →
  basketball = 18 →
  football_tennis = 12 →
  football_basketball = 10 →
  tennis_basketball = 8 →
  all_three = 5 →
  total - (football + tennis + basketball - football_tennis - football_basketball - tennis_basketball + all_three) = 2 :=
by sorry

end students_not_participating_l2856_285678


namespace existence_of_non_divisible_pair_l2856_285647

theorem existence_of_non_divisible_pair (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧
    ¬(p^2 ∣ a^(p-1) - 1) ∧ ¬(p^2 ∣ (a+1)^(p-1) - 1) := by
  sorry

end existence_of_non_divisible_pair_l2856_285647


namespace grant_total_earnings_l2856_285619

/-- Grant's earnings as a freelance math worker over three months -/
def grant_earnings : ℕ → ℕ
| 0 => 350  -- First month
| 1 => 2 * 350 + 50  -- Second month
| 2 => 4 * (grant_earnings 0 + grant_earnings 1)  -- Third month
| _ => 0  -- Other months (not relevant for this problem)

/-- The total earnings for the first three months -/
def total_earnings : ℕ := grant_earnings 0 + grant_earnings 1 + grant_earnings 2

theorem grant_total_earnings : total_earnings = 5500 := by
  sorry

end grant_total_earnings_l2856_285619


namespace xy_equals_nine_l2856_285677

theorem xy_equals_nine (x y : ℝ) (h : x * (x + 2*y) = x^2 + 18) : x * y = 9 := by
  sorry

end xy_equals_nine_l2856_285677


namespace sqrt_equation_solution_l2856_285629

theorem sqrt_equation_solution : ∃ x : ℝ, x = 2209 / 64 ∧ Real.sqrt x + Real.sqrt (x + 3) = 12 := by
  sorry

end sqrt_equation_solution_l2856_285629


namespace same_hours_october_september_l2856_285676

/-- Represents Julie's landscaping business earnings --/
structure LandscapingEarnings where
  mowing_rate : ℕ
  weeding_rate : ℕ
  sept_mowing_hours : ℕ
  sept_weeding_hours : ℕ
  total_earnings : ℕ

/-- Theorem stating that Julie worked the same hours in October as in September --/
theorem same_hours_october_september (j : LandscapingEarnings)
  (h1 : j.mowing_rate = 4)
  (h2 : j.weeding_rate = 8)
  (h3 : j.sept_mowing_hours = 25)
  (h4 : j.sept_weeding_hours = 3)
  (h5 : j.total_earnings = 248) :
  j.mowing_rate * j.sept_mowing_hours + j.weeding_rate * j.sept_weeding_hours =
  j.total_earnings - (j.mowing_rate * j.sept_mowing_hours + j.weeding_rate * j.sept_weeding_hours) :=
by
  sorry

end same_hours_october_september_l2856_285676


namespace fib_arithmetic_seq_solution_l2856_285604

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Property of three consecutive Fibonacci numbers forming an arithmetic sequence -/
def is_fib_arithmetic_seq (a b c : ℕ) : Prop :=
  fib b - fib a = fib c - fib b ∧ fib a < fib b ∧ fib b < fib c

theorem fib_arithmetic_seq_solution :
  ∃ a b c : ℕ, is_fib_arithmetic_seq a b c ∧ a + b + c = 3000 ∧ a = 998 := by
  sorry


end fib_arithmetic_seq_solution_l2856_285604
