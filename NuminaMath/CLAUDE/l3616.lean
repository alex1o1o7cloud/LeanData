import Mathlib

namespace NUMINAMATH_CALUDE_b_minus_c_subscription_l3616_361651

/-- Represents the business subscription problem -/
structure BusinessSubscription where
  total_subscription : ℕ
  total_profit : ℕ
  c_profit : ℕ
  a_more_than_b : ℕ

/-- Theorem stating the difference between B's and C's subscriptions -/
theorem b_minus_c_subscription (bs : BusinessSubscription)
  (h1 : bs.total_subscription = 50000)
  (h2 : bs.total_profit = 35000)
  (h3 : bs.c_profit = 8400)
  (h4 : bs.a_more_than_b = 4000) :
  ∃ (b_sub c_sub : ℕ), b_sub - c_sub = 10000 ∧
    ∃ (a_sub : ℕ), a_sub + b_sub + c_sub = bs.total_subscription ∧
    a_sub = b_sub + bs.a_more_than_b ∧
    bs.c_profit * bs.total_subscription = c_sub * bs.total_profit :=
by sorry

end NUMINAMATH_CALUDE_b_minus_c_subscription_l3616_361651


namespace NUMINAMATH_CALUDE_ratio_problem_l3616_361632

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : 
  x / y = 13 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3616_361632


namespace NUMINAMATH_CALUDE_elimination_failure_l3616_361693

theorem elimination_failure (x y : ℝ) : 
  (2 * x - 3 * y = 5) → 
  (3 * x - 2 * y = 7) → 
  (2 * (2 * x - 3 * y) - (-3) * (3 * x - 2 * y) ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_elimination_failure_l3616_361693


namespace NUMINAMATH_CALUDE_john_laptop_savings_l3616_361690

def octal_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem john_laptop_savings :
  octal_to_decimal 5555 - 1500 = 1425 := by
  sorry

end NUMINAMATH_CALUDE_john_laptop_savings_l3616_361690


namespace NUMINAMATH_CALUDE_days_to_complete_correct_l3616_361696

/-- Represents the number of days required to complete the work -/
def days_to_complete : ℕ := 9

/-- Represents the total number of family members -/
def total_members : ℕ := 15

/-- Represents the number of days it takes for a woman to complete the work -/
def woman_days : ℕ := 180

/-- Represents the number of days it takes for a man to complete the work -/
def man_days : ℕ := 120

/-- Represents the number of women in the family -/
def num_women : ℕ := 3

/-- Represents the number of men in the family -/
def num_men : ℕ := total_members - num_women

/-- Represents the fraction of work done by women in one day -/
def women_work_per_day : ℚ := (1 / woman_days : ℚ) * num_women / 3

/-- Represents the fraction of work done by men in one day -/
def men_work_per_day : ℚ := (1 / man_days : ℚ) * num_men / 2

/-- Represents the total fraction of work done by the family in one day -/
def total_work_per_day : ℚ := women_work_per_day + men_work_per_day

/-- Theorem stating that the calculated number of days to complete the work is correct -/
theorem days_to_complete_correct : 
  ⌈(1 : ℚ) / total_work_per_day⌉ = days_to_complete := by sorry

end NUMINAMATH_CALUDE_days_to_complete_correct_l3616_361696


namespace NUMINAMATH_CALUDE_verna_haley_weight_difference_l3616_361673

/-- Given the weights of Verna, Haley, and Sherry, prove that Verna weighs 17 pounds more than Haley -/
theorem verna_haley_weight_difference :
  ∀ (verna_weight haley_weight sherry_weight : ℕ),
    verna_weight > haley_weight →
    verna_weight = sherry_weight / 2 →
    haley_weight = 103 →
    verna_weight + sherry_weight = 360 →
    verna_weight - haley_weight = 17 := by
  sorry

end NUMINAMATH_CALUDE_verna_haley_weight_difference_l3616_361673


namespace NUMINAMATH_CALUDE_opposite_of_neg_two_l3616_361627

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- Theorem: The opposite of -2 is 2 -/
theorem opposite_of_neg_two : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_two_l3616_361627


namespace NUMINAMATH_CALUDE_permutation_144_is_216543_l3616_361676

/-- Represents a permutation of the digits 1, 2, 3, 4, 5, 6 -/
def Permutation := Fin 6 → Fin 6

/-- Returns true if the permutation is valid (bijective) -/
def isValidPermutation (p : Permutation) : Prop :=
  Function.Bijective p

/-- Converts a permutation to an integer -/
def permutationToInt (p : Permutation) : ℕ :=
  (p 0) * 100000 + (p 1) * 10000 + (p 2) * 1000 + (p 3) * 100 + (p 4) * 10 + (p 5)

/-- Returns true if the permutation corresponds to the given integer -/
def permutationMatchesInt (p : Permutation) (n : ℕ) : Prop :=
  permutationToInt p = n

/-- Represents the ascending order of permutations -/
def isAscendingOrder (perms : Fin 720 → Permutation) : Prop :=
  ∀ i j : Fin 720, i < j → permutationToInt (perms i) < permutationToInt (perms j)

/-- The main theorem -/
theorem permutation_144_is_216543 
  (perms : Fin 720 → Permutation)
  (h_valid : ∀ i, isValidPermutation (perms i))
  (h_ascending : isAscendingOrder perms)
  (h_first : permutationMatchesInt (perms 0) 123456)
  (h_last : permutationMatchesInt (perms 719) 654321) :
  permutationMatchesInt (perms 143) 216543 :=
sorry

end NUMINAMATH_CALUDE_permutation_144_is_216543_l3616_361676


namespace NUMINAMATH_CALUDE_beach_trip_seashells_l3616_361681

/-- Calculates the total number of seashells found during a beach trip -/
def total_seashells (days : ℕ) (shells_per_day : ℕ) : ℕ :=
  days * shells_per_day

theorem beach_trip_seashells :
  let days : ℕ := 5
  let shells_per_day : ℕ := 7
  total_seashells days shells_per_day = 35 := by
  sorry

end NUMINAMATH_CALUDE_beach_trip_seashells_l3616_361681


namespace NUMINAMATH_CALUDE_triangle_problem_l3616_361699

/-- Triangle ABC with angles A, B, C opposite sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions and theorem -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.A = π / 6)
  (h2 : (1 + Real.sqrt 3) * t.c = 2 * t.b) :
  t.C = π / 4 ∧ 
  (t.b * t.a * Real.cos t.C = 1 + Real.sqrt 3 → 
    t.a = Real.sqrt 2 ∧ t.b = 1 + Real.sqrt 3 ∧ t.c = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3616_361699


namespace NUMINAMATH_CALUDE_index_cards_per_pack_l3616_361620

-- Define the given conditions
def cards_per_student : ℕ := 10
def periods_per_day : ℕ := 6
def students_per_class : ℕ := 30
def cost_per_pack : ℕ := 3
def total_spent : ℕ := 108

-- Define the theorem
theorem index_cards_per_pack :
  (cards_per_student * periods_per_day * students_per_class) / (total_spent / cost_per_pack) = 50 := by
  sorry

end NUMINAMATH_CALUDE_index_cards_per_pack_l3616_361620


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l3616_361698

def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

theorem f_strictly_increasing :
  (∀ x y, x < y ∧ y < -1 → f x < f y) ∧
  (∀ x y, 11 < x ∧ x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l3616_361698


namespace NUMINAMATH_CALUDE_triangle_is_right_angled_l3616_361638

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def complementary (t : Triangle) : Prop :=
  t.A + t.B = 90

def pythagorean (t : Triangle) : Prop :=
  (t.a + t.b) * (t.a - t.b) = t.c^2

def angle_ratio (t : Triangle) : Prop :=
  t.A / t.B = 1 / 2 ∧ t.A / t.C = 1

-- Theorem statement
theorem triangle_is_right_angled (t : Triangle) 
  (h1 : complementary t) 
  (h2 : pythagorean t) 
  (h3 : angle_ratio t) : 
  t.A = 45 ∧ t.B = 90 ∧ t.C = 45 :=
sorry

end NUMINAMATH_CALUDE_triangle_is_right_angled_l3616_361638


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l3616_361683

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, 3; 4, 5, 6; 7, 8, 9]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 1; 1, 1, 0; 0, 1, 1]
  A * B = !![3, 5, 4; 9, 11, 10; 15, 17, 16] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l3616_361683


namespace NUMINAMATH_CALUDE_equation_solutions_l3616_361642

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x - 2)^6 + (x - 6)^6 = 432

-- Define the approximate solutions
def solution1 : ℝ := 4.795
def solution2 : ℝ := 3.205

-- State the theorem
theorem equation_solutions :
  ∃ (ε : ℝ), ε > 0 ∧ 
  (∀ (x : ℝ), equation x → (|x - solution1| < ε ∨ |x - solution2| < ε)) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3616_361642


namespace NUMINAMATH_CALUDE_f_plus_g_at_one_l3616_361660

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_plus_g_at_one
  (h_even : is_even f)
  (h_odd : is_odd g)
  (h_diff : ∀ x, f x - g x = x^3 + x^2 + 1) :
  f 1 + g 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_plus_g_at_one_l3616_361660


namespace NUMINAMATH_CALUDE_money_sharing_l3616_361689

theorem money_sharing (total : ℚ) (per_person : ℚ) (num_people : ℕ) : 
  total = 3.75 ∧ per_person = 1.25 → num_people = 3 ∧ total = num_people * per_person :=
by sorry

end NUMINAMATH_CALUDE_money_sharing_l3616_361689


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3616_361675

theorem complex_fraction_equality : 
  (2 + 5/8 - 2/3 * (2 + 5/14)) / ((3 + 1/12 + 4.375) / (19 + 8/9)) = 2 + 17/21 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3616_361675


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3616_361619

theorem arithmetic_expression_equality : 2 + 3 * 4 - 5 + 6 * (2 - 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3616_361619


namespace NUMINAMATH_CALUDE_total_wage_proof_l3616_361695

/-- The weekly payment for employee B -/
def wage_B : ℝ := 249.99999999999997

/-- The weekly payment for employee A -/
def wage_A : ℝ := 1.2 * wage_B

/-- The total weekly payment for both employees -/
def total_wage : ℝ := wage_A + wage_B

theorem total_wage_proof : total_wage = 549.9999999999999 := by sorry

end NUMINAMATH_CALUDE_total_wage_proof_l3616_361695


namespace NUMINAMATH_CALUDE_x0_value_l3616_361635

def f (x : ℝ) : ℝ := x^3 + x - 1

theorem x0_value (x₀ : ℝ) (h : (deriv f) x₀ = 4) : x₀ = 1 ∨ x₀ = -1 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l3616_361635


namespace NUMINAMATH_CALUDE_A_three_times_faster_than_B_l3616_361686

/-- The work rate of A -/
def work_rate_A : ℚ := 1 / 16

/-- The work rate of B -/
def work_rate_B : ℚ := 1 / 12 - 1 / 16

/-- The theorem stating that A is 3 times faster than B -/
theorem A_three_times_faster_than_B : work_rate_A / work_rate_B = 3 := by
  sorry

end NUMINAMATH_CALUDE_A_three_times_faster_than_B_l3616_361686


namespace NUMINAMATH_CALUDE_fraction_identity_l3616_361658

theorem fraction_identity (M N a b x : ℝ) (h1 : x ≠ a) (h2 : x ≠ b) (h3 : a ≠ b) :
  (M * x + N) / ((x - a) * (x - b)) = 
  (M * a + N) / (a - b) * (1 / (x - a)) - (M * b + N) / (a - b) * (1 / (x - b)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_identity_l3616_361658


namespace NUMINAMATH_CALUDE_multiple_of_nine_problem_l3616_361633

theorem multiple_of_nine_problem (N : ℕ) : 
  (∃ k : ℕ, N = 9 * k) →
  (∃ Q : ℕ, N = 9 * Q ∧ Q = 9 * 25 + 7) →
  N = 2088 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_nine_problem_l3616_361633


namespace NUMINAMATH_CALUDE_expression_equality_l3616_361674

theorem expression_equality (x : ℝ) (h : x^2 + 2*x - 3 = 7) : 2*x^2 + 4*x + 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3616_361674


namespace NUMINAMATH_CALUDE_shifted_line_equation_l3616_361684

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line horizontally -/
def shift_line (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift * l.slope }

/-- The original line y = x -/
def original_line : Line := { slope := 1, intercept := 0 }

theorem shifted_line_equation :
  let shifted := shift_line original_line (-1)
  shifted.slope = 1 ∧ shifted.intercept = 1 :=
sorry

end NUMINAMATH_CALUDE_shifted_line_equation_l3616_361684


namespace NUMINAMATH_CALUDE_root_product_cubic_polynomial_l3616_361694

theorem root_product_cubic_polynomial :
  let p := fun (x : ℝ) => 3 * x^3 - 4 * x^2 + x - 10
  ∃ a b c : ℝ, p a = 0 ∧ p b = 0 ∧ p c = 0 ∧ a * b * c = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_root_product_cubic_polynomial_l3616_361694


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3616_361631

-- Define the quadratic function
def f (k x : ℝ) : ℝ := 2*k*x^2 - 2*x - 3*k - 2

-- Define the property of having two real roots
def has_two_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0

-- Define the property of roots being on opposite sides of 1
def roots_around_one (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 1 ∧ x₂ > 1 ∧ f k x₁ = 0 ∧ f k x₂ = 0

-- Theorem statement
theorem quadratic_roots_range (k : ℝ) :
  has_two_real_roots k ∧ roots_around_one k ↔ k < -4 ∨ k > 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3616_361631


namespace NUMINAMATH_CALUDE_expand_and_complete_square_l3616_361652

theorem expand_and_complete_square (x : ℝ) : 
  -2 * (x - 3) * (x + 1/2) = -2 * (x - 5/4)^2 + 49/8 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_complete_square_l3616_361652


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3616_361691

theorem area_between_concentric_circles 
  (r : Real) -- radius of inner circle
  (h1 : r > 0) -- inner radius is positive
  (h2 : 3 * r - r = 4) -- difference between outer and inner radii is 4
  : π * (3 * r)^2 - π * r^2 = 32 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3616_361691


namespace NUMINAMATH_CALUDE_game_tie_fraction_l3616_361666

theorem game_tie_fraction (max_win_rate sara_win_rate postponed_rate : ℚ)
  (h_max : max_win_rate = 2/5)
  (h_sara : sara_win_rate = 1/4)
  (h_postponed : postponed_rate = 5/100) :
  let total_win_rate := max_win_rate + sara_win_rate
  let non_postponed_rate := 1 - postponed_rate
  let win_rate_of_non_postponed := total_win_rate / non_postponed_rate
  1 - win_rate_of_non_postponed = 6/19 := by
sorry

end NUMINAMATH_CALUDE_game_tie_fraction_l3616_361666


namespace NUMINAMATH_CALUDE_senior_mean_score_senior_mean_score_is_88_l3616_361622

/-- The mean score of seniors in a math competition --/
theorem senior_mean_score (total_students : ℕ) (overall_mean : ℝ) 
  (junior_ratio : ℝ) (senior_score_ratio : ℝ) : ℝ :=
  let senior_count := (total_students : ℝ) / (1 + junior_ratio)
  let junior_count := senior_count * junior_ratio
  let junior_mean := overall_mean * (total_students : ℝ) / (senior_count * senior_score_ratio + junior_count)
  junior_mean * senior_score_ratio

/-- The mean score of seniors is approximately 88 --/
theorem senior_mean_score_is_88 : 
  ∃ ε > 0, |senior_mean_score 150 80 1.2 1.2 - 88| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_senior_mean_score_senior_mean_score_is_88_l3616_361622


namespace NUMINAMATH_CALUDE_peters_horse_food_l3616_361608

/-- Calculates the total food required for horses over a given number of days -/
def total_food_required (num_horses : ℕ) (oats_per_meal : ℕ) (oats_meals_per_day : ℕ) 
                        (grain_per_day : ℕ) (num_days : ℕ) : ℕ :=
  let total_oats := num_horses * oats_per_meal * oats_meals_per_day * num_days
  let total_grain := num_horses * grain_per_day * num_days
  total_oats + total_grain

/-- Theorem: Peter needs 132 pounds of food to feed his horses for 3 days -/
theorem peters_horse_food : total_food_required 4 4 2 3 3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_peters_horse_food_l3616_361608


namespace NUMINAMATH_CALUDE_star_product_scaling_l3616_361600

/-- Given that 2994 ã · 14.5 = 179, prove that 29.94 ã · 1.45 = 0.179 -/
theorem star_product_scaling (h : 2994 * 14.5 = 179) : 29.94 * 1.45 = 0.179 := by
  sorry

end NUMINAMATH_CALUDE_star_product_scaling_l3616_361600


namespace NUMINAMATH_CALUDE_inequality_proof_l3616_361672

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a * b * c ≤ (a + b) * (b + c) * (c + a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3616_361672


namespace NUMINAMATH_CALUDE_line_through_point_with_given_slope_l3616_361697

/-- Given a line L1: 2x + y - 10 = 0 and a point P(1, 0),
    prove that the line L2 passing through P with the same slope as L1
    has the equation 2x + y - 2 = 0 -/
theorem line_through_point_with_given_slope (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 2 * x + y - 10 = 0
  let P : ℝ × ℝ := (1, 0)
  let L2 : ℝ → ℝ → Prop := λ x y => 2 * x + y - 2 = 0
  (∀ x y, L1 x y ↔ 2 * x + y = 10) →
  (L2 (P.1) (P.2)) →
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = (y - P.2) / (x - P.1)) →
  ∀ x y, L2 x y ↔ 2 * x + y = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_given_slope_l3616_361697


namespace NUMINAMATH_CALUDE_prob_rolling_six_is_five_thirty_sixths_l3616_361617

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of ways to roll a sum of 6 with two dice -/
def waysToRollSix : ℕ := 5

/-- The probability of rolling a sum of 6 with two fair dice -/
def probRollingSix : ℚ := waysToRollSix / totalOutcomes

theorem prob_rolling_six_is_five_thirty_sixths :
  probRollingSix = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_rolling_six_is_five_thirty_sixths_l3616_361617


namespace NUMINAMATH_CALUDE_raj_snow_volume_l3616_361602

/-- The total volume of snow on a rectangular driveway with two distinct layers -/
def total_snow_volume (length width depth1 depth2 : ℝ) : ℝ :=
  length * width * depth1 + length * width * depth2

/-- Theorem: The total volume of snow on Raj's driveway is 96 cubic feet -/
theorem raj_snow_volume :
  total_snow_volume 30 4 0.5 0.3 = 96 := by
  sorry

end NUMINAMATH_CALUDE_raj_snow_volume_l3616_361602


namespace NUMINAMATH_CALUDE_senior_citizens_average_age_l3616_361610

theorem senior_citizens_average_age
  (total_members : ℕ)
  (overall_average_age : ℚ)
  (num_women : ℕ)
  (num_men : ℕ)
  (num_seniors : ℕ)
  (women_average_age : ℚ)
  (men_average_age : ℚ)
  (h1 : total_members = 60)
  (h2 : overall_average_age = 30)
  (h3 : num_women = 25)
  (h4 : num_men = 20)
  (h5 : num_seniors = 15)
  (h6 : women_average_age = 28)
  (h7 : men_average_age = 35)
  (h8 : total_members = num_women + num_men + num_seniors) :
  (total_members * overall_average_age - num_women * women_average_age - num_men * men_average_age) / num_seniors = 80 / 3 :=
by sorry

end NUMINAMATH_CALUDE_senior_citizens_average_age_l3616_361610


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3616_361650

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the angle between its two asymptotes is 60°, then its eccentricity e
    is either 2 or 2√3/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let asymptote_angle := Real.pi / 3
  let eccentricity := Real.sqrt (1 + b^2 / a^2)
  asymptote_angle = Real.arctan (b / a) * 2 →
  eccentricity = 2 ∨ eccentricity = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3616_361650


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3616_361616

/-- A geometric sequence with the given first four terms has a common ratio of -3/2 -/
theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℚ), 
    a 0 = 32 ∧ a 1 = -48 ∧ a 2 = 72 ∧ a 3 = -108 →
    ∃ (r : ℚ), r = -3/2 ∧ ∀ n, a (n + 1) = r * a n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3616_361616


namespace NUMINAMATH_CALUDE_divisibility_by_24_l3616_361653

theorem divisibility_by_24 (n : ℕ) (h_odd : Odd n) (h_not_div_3 : ¬3 ∣ n) :
  24 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l3616_361653


namespace NUMINAMATH_CALUDE_train_length_l3616_361646

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 12 → ∃ length : ℝ, 
  (length ≥ 399) ∧ (length ≤ 401) ∧ (length = speed * time * 1000 / 3600) := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3616_361646


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3616_361613

def P : Set ℤ := {-1, 1}
def Q : Set ℤ := {0, 1, 2}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3616_361613


namespace NUMINAMATH_CALUDE_only_one_implies_negation_l3616_361685

theorem only_one_implies_negation (p q : Prop) : 
  (∃! x : Fin 4, match x with
    | 0 => (p ∨ q) → ¬(p ∨ q)
    | 1 => (p ∧ ¬q) → ¬(p ∨ q)
    | 2 => (¬p ∧ q) → ¬(p ∨ q)
    | 3 => (¬p ∧ ¬q) → ¬(p ∨ q)
  ) := by sorry

end NUMINAMATH_CALUDE_only_one_implies_negation_l3616_361685


namespace NUMINAMATH_CALUDE_lowest_possible_price_l3616_361636

def manufacturer_price : ℝ := 45.00
def max_regular_discount : ℝ := 0.30
def additional_sale_discount : ℝ := 0.20

theorem lowest_possible_price :
  let regular_discounted_price := manufacturer_price * (1 - max_regular_discount)
  let final_price := regular_discounted_price * (1 - additional_sale_discount)
  final_price = 25.20 := by
sorry

end NUMINAMATH_CALUDE_lowest_possible_price_l3616_361636


namespace NUMINAMATH_CALUDE_translated_line_through_point_l3616_361604

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically -/
def translate_line (l : Line) (units : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + units }

/-- Check if a point lies on a line -/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem translated_line_through_point (m : ℝ) :
  let original_line : Line := { slope := 1, intercept := 0 }
  let translated_line := translate_line original_line 3
  point_on_line translated_line 2 m → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_translated_line_through_point_l3616_361604


namespace NUMINAMATH_CALUDE_train_speed_l3616_361606

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 160)
  (h2 : bridge_length = 215)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3616_361606


namespace NUMINAMATH_CALUDE_coin_distribution_l3616_361668

def sum_of_integers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem coin_distribution (n : ℕ) (h : n = 20) :
  ∃ k : ℕ, sum_of_integers n = 3 * k ∧ ¬∃ m : ℕ, sum_of_integers n + 100 = 3 * m := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_l3616_361668


namespace NUMINAMATH_CALUDE_two_planes_parallel_to_same_line_are_parallel_two_planes_parallel_to_same_line_not_always_parallel_l3616_361678

-- Define the concept of a plane in 3D space
variable (Plane : Type)

-- Define the concept of a line in 3D space
variable (Line : Type)

-- Define the parallel relation between planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the parallel relation between a plane and a line
variable (parallel_plane_line : Plane → Line → Prop)

-- Theorem to be proven false
theorem two_planes_parallel_to_same_line_are_parallel 
  (p1 p2 : Plane) (l : Line) : 
  parallel_plane_line p1 l → parallel_plane_line p2 l → parallel_planes p1 p2 := by
  sorry

-- The actual theorem should be that the above statement is false
theorem two_planes_parallel_to_same_line_not_always_parallel : 
  ¬∀ (p1 p2 : Plane) (l : Line), 
    parallel_plane_line p1 l → parallel_plane_line p2 l → parallel_planes p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_two_planes_parallel_to_same_line_are_parallel_two_planes_parallel_to_same_line_not_always_parallel_l3616_361678


namespace NUMINAMATH_CALUDE_wall_height_proof_l3616_361656

/-- Given a wall and a painting, proves that the wall height is 5 feet -/
theorem wall_height_proof (wall_width painting_width painting_height painting_area_percentage : ℝ) :
  wall_width = 10 ∧ 
  painting_width = 2 ∧ 
  painting_height = 4 ∧ 
  painting_area_percentage = 0.16 ∧
  painting_width * painting_height = painting_area_percentage * (wall_width * (wall_width * painting_height / (painting_width * painting_height))) →
  wall_width * painting_height / (painting_width * painting_height) = 5 := by
sorry

end NUMINAMATH_CALUDE_wall_height_proof_l3616_361656


namespace NUMINAMATH_CALUDE_family_savings_by_end_of_2019_l3616_361612

/-- Proves that the family's savings by 31.12.2019 will be 1340840 rubles given their income, expenses, and initial savings. -/
theorem family_savings_by_end_of_2019 
  (income : ℕ) 
  (expenses : ℕ) 
  (initial_savings : ℕ) 
  (h1 : income = (55000 + 45000 + 10000 + 17400) * 4)
  (h2 : expenses = (40000 + 20000 + 5000 + 2000 + 2000) * 4)
  (h3 : initial_savings = 1147240) : 
  initial_savings + income - expenses = 1340840 :=
by sorry

end NUMINAMATH_CALUDE_family_savings_by_end_of_2019_l3616_361612


namespace NUMINAMATH_CALUDE_polygon_symmetry_l3616_361629

-- Define a convex polygon
def ConvexPolygon : Type := sorry

-- Define a point inside a polygon
def PointInside (P : ConvexPolygon) : Type := sorry

-- Define a line passing through a point
def LineThroughPoint (P : ConvexPolygon) (O : PointInside P) : Type := sorry

-- Define the property of a line dividing the polygon area in half
def DividesAreaInHalf (P : ConvexPolygon) (O : PointInside P) (l : LineThroughPoint P O) : Prop := sorry

-- Define central symmetry of a polygon
def CentrallySymmetric (P : ConvexPolygon) : Prop := sorry

-- Define center of symmetry
def CenterOfSymmetry (P : ConvexPolygon) (O : PointInside P) : Prop := sorry

-- The main theorem
theorem polygon_symmetry (P : ConvexPolygon) (O : PointInside P) 
  (h : ∀ (l : LineThroughPoint P O), DividesAreaInHalf P O l) : 
  CentrallySymmetric P ∧ CenterOfSymmetry P O := by
  sorry

end NUMINAMATH_CALUDE_polygon_symmetry_l3616_361629


namespace NUMINAMATH_CALUDE_robie_chocolates_l3616_361659

/-- Calculates the number of chocolate bags left after a series of purchases and giveaways. -/
def chocolates_left (initial_purchase : ℕ) (given_away : ℕ) (additional_purchase : ℕ) : ℕ :=
  initial_purchase - given_away + additional_purchase

/-- Proves that given the specific scenario, 4 bags of chocolates are left. -/
theorem robie_chocolates : chocolates_left 3 2 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_robie_chocolates_l3616_361659


namespace NUMINAMATH_CALUDE_alpha_values_l3616_361624

theorem alpha_values (α : ℂ) 
  (h1 : α ≠ Complex.I ∧ α ≠ -Complex.I)
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : (Complex.abs (α^4 - 1))^2 = 9 * (Complex.abs (α - 1))^2) :
  α = (1/2 : ℂ) + Complex.I * (Real.sqrt 35 / 2) ∨ 
  α = (1/2 : ℂ) - Complex.I * (Real.sqrt 35 / 2) :=
sorry

end NUMINAMATH_CALUDE_alpha_values_l3616_361624


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3616_361665

theorem smallest_number_satisfying_conditions :
  ∃ (n : ℕ), n > 0 ∧
  (∀ k : ℕ, (21 ^ k ∣ n) → 7 ^ k - k ^ 7 = 1) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, (21 ^ k ∣ m) → 7 ^ k - k ^ 7 = 1) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3616_361665


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3616_361634

theorem decimal_to_fraction : 
  ∃ (n d : ℕ), d ≠ 0 ∧ 3.36 = (n : ℚ) / (d : ℚ) ∧ (n.gcd d = 1) ∧ n = 84 ∧ d = 25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3616_361634


namespace NUMINAMATH_CALUDE_record_storage_cost_l3616_361645

/-- A record storage problem -/
theorem record_storage_cost (box_length box_width box_height : ℝ)
  (total_volume : ℝ) (cost_per_box : ℝ) :
  box_length = 15 →
  box_width = 12 →
  box_height = 10 →
  total_volume = 1080000 →
  cost_per_box = 0.2 →
  (total_volume / (box_length * box_width * box_height)) * cost_per_box = 120 := by
  sorry

end NUMINAMATH_CALUDE_record_storage_cost_l3616_361645


namespace NUMINAMATH_CALUDE_trapezoid_EFGH_area_l3616_361623

/-- Trapezoid with vertices E(0,0), F(0,3), G(5,0), and H(5,7) -/
structure Trapezoid where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

/-- The area of a trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ := sorry

/-- The theorem stating the area of the specific trapezoid EFGH -/
theorem trapezoid_EFGH_area :
  let t : Trapezoid := {
    E := (0, 0),
    F := (0, 3),
    G := (5, 0),
    H := (5, 7)
  }
  trapezoidArea t = 25 := by sorry

end NUMINAMATH_CALUDE_trapezoid_EFGH_area_l3616_361623


namespace NUMINAMATH_CALUDE_johnsonville_marching_band_max_members_l3616_361671

theorem johnsonville_marching_band_max_members :
  ∀ n : ℕ,
  (∃ k : ℕ, 15 * n = 30 * k + 6) →
  15 * n < 900 →
  (∀ m : ℕ, (∃ j : ℕ, 15 * m = 30 * j + 6) → 15 * m < 900 → 15 * m ≤ 15 * n) →
  15 * n = 810 :=
by sorry

end NUMINAMATH_CALUDE_johnsonville_marching_band_max_members_l3616_361671


namespace NUMINAMATH_CALUDE_inverse_function_equality_f_equals_f_inverse_l3616_361601

def f (x : ℝ) : ℝ := 4 * x - 5

theorem inverse_function_equality (f : ℝ → ℝ) (h : Function.Bijective f) :
  ∃ x : ℝ, f x = Function.invFun f x :=
by
  sorry

theorem f_equals_f_inverse :
  ∃ x : ℝ, f x = Function.invFun f x ∧ x = 5/3 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_function_equality_f_equals_f_inverse_l3616_361601


namespace NUMINAMATH_CALUDE_greatest_power_of_two_l3616_361657

theorem greatest_power_of_two (n : ℕ) : 
  (∃ k : ℕ, 2^k ∣ (10^1006 - 6^503) ∧ 
   ∀ m : ℕ, 2^m ∣ (10^1006 - 6^503) → m ≤ k) → 
  (∃ k : ℕ, k = 503 ∧ 2^k ∣ (10^1006 - 6^503) ∧ 
   ∀ m : ℕ, 2^m ∣ (10^1006 - 6^503) → m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_l3616_361657


namespace NUMINAMATH_CALUDE_profit_maximized_at_70_l3616_361609

/-- Represents the store's helmet sales scenario -/
structure HelmetStore where
  initialPrice : ℝ
  initialSales : ℝ
  priceReductionEffect : ℝ
  costPrice : ℝ

/-- Calculates the monthly profit for a given selling price -/
def monthlyProfit (store : HelmetStore) (sellingPrice : ℝ) : ℝ :=
  let salesVolume := store.initialSales + (store.initialPrice - sellingPrice) * store.priceReductionEffect
  (sellingPrice - store.costPrice) * salesVolume

/-- Theorem stating that 70 yuan maximizes the monthly profit -/
theorem profit_maximized_at_70 (store : HelmetStore) 
    (h1 : store.initialPrice = 80)
    (h2 : store.initialSales = 200)
    (h3 : store.priceReductionEffect = 20)
    (h4 : store.costPrice = 50) :
    ∀ x, monthlyProfit store 70 ≥ monthlyProfit store x := by
  sorry

#check profit_maximized_at_70

end NUMINAMATH_CALUDE_profit_maximized_at_70_l3616_361609


namespace NUMINAMATH_CALUDE_ratio_equality_l3616_361692

theorem ratio_equality (x y : ℝ) (h : 1.5 * x = 0.04 * y) :
  (y - x) / (y + x) = 73 / 77 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3616_361692


namespace NUMINAMATH_CALUDE_intersection_A_notB_when_a_is_neg_two_union_A_B_equals_B_implies_a_range_l3616_361679

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Define the complement of B
def notB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Theorem for part I
theorem intersection_A_notB_when_a_is_neg_two : 
  A (-2) ∩ notB = {x | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part II
theorem union_A_B_equals_B_implies_a_range (a : ℝ) : 
  A a ∪ B = B → a < -4 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_notB_when_a_is_neg_two_union_A_B_equals_B_implies_a_range_l3616_361679


namespace NUMINAMATH_CALUDE_polynomial_sum_simplification_l3616_361647

/-- Given two polynomials over ℝ, prove their sum equals a specific polynomial -/
theorem polynomial_sum_simplification (x : ℝ) :
  (3 * x^4 - 2 * x^3 + 5 * x^2 - 8 * x + 10) + 
  (7 * x^5 - 3 * x^4 + x^3 - 7 * x^2 + 2 * x - 2) = 
  7 * x^5 - x^3 - 2 * x^2 - 6 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_simplification_l3616_361647


namespace NUMINAMATH_CALUDE_odd_function_inequality_l3616_361688

/-- An odd, differentiable function satisfying certain conditions -/
structure OddFunction where
  f : ℝ → ℝ
  odd : ∀ x, f (-x) = -f x
  diff : Differentiable ℝ f
  cond : ∀ x < 0, 2 * f x + x * deriv f x < 0

/-- Theorem stating the relationship between f(1), 2016f(√2016), and 2017f(√2017) -/
theorem odd_function_inequality (f : OddFunction) :
  f.f 1 < 2016 * f.f (Real.sqrt 2016) ∧
  2016 * f.f (Real.sqrt 2016) < 2017 * f.f (Real.sqrt 2017) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l3616_361688


namespace NUMINAMATH_CALUDE_equiangular_equilateral_parallelogram_is_square_l3616_361654

-- Define a parallelogram
class Parallelogram (P : Type) where
  -- Add any necessary properties of a parallelogram

-- Define the property of being equiangular
class Equiangular (P : Type) where
  -- All angles are equal

-- Define the property of being equilateral
class Equilateral (P : Type) where
  -- All sides have equal length

-- Define a square
class Square (P : Type) extends Parallelogram P where
  -- A square is a parallelogram with additional properties

-- Theorem statement
theorem equiangular_equilateral_parallelogram_is_square 
  (P : Type) [Parallelogram P] [Equiangular P] [Equilateral P] : Square P :=
by sorry

end NUMINAMATH_CALUDE_equiangular_equilateral_parallelogram_is_square_l3616_361654


namespace NUMINAMATH_CALUDE_symmetric_quadratic_property_symmetric_quadratic_comparison_l3616_361628

/-- A quadratic function with a positive leading coefficient and symmetric about x = 2 -/
def symmetric_quadratic (a b c : ℝ) (h : a > 0) : ℝ → ℝ :=
  fun x ↦ a * x^2 + b * x + c

theorem symmetric_quadratic_property {a b c : ℝ} (h : a > 0) :
  ∀ x, symmetric_quadratic a b c h (2 + x) = symmetric_quadratic a b c h (2 - x) :=
by sorry

theorem symmetric_quadratic_comparison {a b c : ℝ} (h : a > 0) :
  symmetric_quadratic a b c h 0.5 > symmetric_quadratic a b c h π :=
by sorry

end NUMINAMATH_CALUDE_symmetric_quadratic_property_symmetric_quadratic_comparison_l3616_361628


namespace NUMINAMATH_CALUDE_september_births_percentage_l3616_361611

theorem september_births_percentage 
  (total_people : ℕ) 
  (september_births : ℕ) 
  (h1 : total_people = 120) 
  (h2 : september_births = 12) : 
  (september_births : ℚ) / total_people * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_september_births_percentage_l3616_361611


namespace NUMINAMATH_CALUDE_axes_of_symmetry_coincide_l3616_361615

/-- Two quadratic functions with their coefficients -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  r : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  f₁ : QuadraticFunction
  f₂ : QuadraticFunction
  A : Point
  B : Point
  p₁_positive : f₁.p > 0
  p₂_negative : f₂.p < 0
  distinct_intersections : A ≠ B
  intersections_on_curves : 
    A.y = f₁.p * A.x^2 + f₁.q * A.x + f₁.r ∧
    A.y = f₂.p * A.x^2 + f₂.q * A.x + f₂.r ∧
    B.y = f₁.p * B.x^2 + f₁.q * B.x + f₁.r ∧
    B.y = f₂.p * B.x^2 + f₂.q * B.x + f₂.r
  tangents_form_cyclic_quad : True  -- This is a placeholder for the cyclic quadrilateral condition

/-- The main theorem stating that the axes of symmetry coincide -/
theorem axes_of_symmetry_coincide (setup : ProblemSetup) : 
  setup.f₁.q / setup.f₁.p = setup.f₂.q / setup.f₂.p := by
  sorry

end NUMINAMATH_CALUDE_axes_of_symmetry_coincide_l3616_361615


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3616_361661

theorem smallest_integer_with_remainders (k : ℕ) : k = 61 ↔ 
  (k > 1) ∧ 
  (∀ m : ℕ, m < k → 
    (m % 12 ≠ 1 ∨ m % 5 ≠ 1 ∨ m % 3 ≠ 1)) ∧
  (k % 12 = 1 ∧ k % 5 = 1 ∧ k % 3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3616_361661


namespace NUMINAMATH_CALUDE_center_distance_of_isosceles_triangle_l3616_361626

/-- An isosceles triangle with two sides of length 6 and one side of length 10 -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2
  side_lengths : side1 = 6 ∧ base = 10

/-- The distance between the centers of the circumscribed and inscribed circles of the triangle -/
def center_distance (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem stating the distance between the centers of the circumscribed and inscribed circles -/
theorem center_distance_of_isosceles_triangle (t : IsoscelesTriangle) :
  center_distance t = (5 * Real.sqrt 110) / 11 := by sorry

end NUMINAMATH_CALUDE_center_distance_of_isosceles_triangle_l3616_361626


namespace NUMINAMATH_CALUDE_roots_equation_value_l3616_361648

theorem roots_equation_value (α β : ℝ) :
  α^2 - 3*α - 2 = 0 →
  β^2 - 3*β - 2 = 0 →
  5 * α^4 + 12 * β^3 = 672.5 + 31.5 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_value_l3616_361648


namespace NUMINAMATH_CALUDE_patsy_guests_l3616_361655

-- Define the problem parameters
def appetizers_per_guest : ℕ := 6
def initial_dozens : ℕ := 3 + 2 + 2
def additional_dozens : ℕ := 8

-- Define the theorem
theorem patsy_guests :
  (((initial_dozens + additional_dozens) * 12) / appetizers_per_guest) = 30 := by
  sorry

end NUMINAMATH_CALUDE_patsy_guests_l3616_361655


namespace NUMINAMATH_CALUDE_homothety_circle_transformation_l3616_361625

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℚ

/-- Applies a homothety transformation to a point -/
def homothety (center : Point) (scale : ℚ) (p : Point) : Point :=
  { x := center.x + scale * (p.x - center.x)
  , y := center.y + scale * (p.y - center.y) }

theorem homothety_circle_transformation :
  let O : Point := { x := 3, y := 4 }
  let originalCircle : Circle := { center := O, radius := 8 }
  let P : Point := { x := 11, y := 12 }
  let scale : ℚ := 2/3
  let newCenter : Point := homothety P scale O
  let newRadius : ℚ := scale * originalCircle.radius
  newCenter.x = 17/3 ∧ newCenter.y = 20/3 ∧ newRadius = 16/3 :=
by sorry

end NUMINAMATH_CALUDE_homothety_circle_transformation_l3616_361625


namespace NUMINAMATH_CALUDE_parallelogram_longer_side_length_l3616_361677

/-- Given a parallelogram with adjacent sides in the ratio 3:2 and perimeter 20,
    prove that the length of the longer side is 6. -/
theorem parallelogram_longer_side_length
  (a b : ℝ)
  (ratio : a / b = 3 / 2)
  (perimeter : 2 * (a + b) = 20)
  : a = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_longer_side_length_l3616_361677


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3616_361643

theorem triangle_angle_proof 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : (a + b + c) * (a - b + c) = a * c)
  (h2 : Real.sin A * Real.sin C = (Real.sqrt 3 - 1) / 4) : 
  B = 2 * π / 3 ∧ (C = π / 12 ∨ C = π / 4) := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_proof_l3616_361643


namespace NUMINAMATH_CALUDE_two_number_problem_l3616_361664

theorem two_number_problem : ∃ x y : ℕ, 
  x ≠ y ∧ 
  x ≥ 10 ∧ 
  y ≥ 10 ∧ 
  (x + y) + (max x y - min x y) + (x * y) + (max x y / min x y) = 576 := by
  sorry

end NUMINAMATH_CALUDE_two_number_problem_l3616_361664


namespace NUMINAMATH_CALUDE_profit_achieved_l3616_361605

/-- The number of disks in a buy package -/
def buy_package : ℕ := 3

/-- The cost of a buy package in cents -/
def buy_cost : ℕ := 400

/-- The number of disks in a sell package -/
def sell_package : ℕ := 4

/-- The price of a sell package in cents -/
def sell_price : ℕ := 600

/-- The target profit in cents -/
def target_profit : ℕ := 15000

/-- The minimum number of disks to be sold to achieve the target profit -/
def min_disks_to_sell : ℕ := 883

theorem profit_achieved : 
  ∃ (n : ℕ), n ≥ min_disks_to_sell ∧ 
  (n * sell_price / sell_package - n * buy_cost / buy_package) ≥ target_profit ∧
  ∀ (m : ℕ), m < min_disks_to_sell → 
  (m * sell_price / sell_package - m * buy_cost / buy_package) < target_profit :=
sorry

end NUMINAMATH_CALUDE_profit_achieved_l3616_361605


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l3616_361639

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l3616_361639


namespace NUMINAMATH_CALUDE_surface_area_comparison_l3616_361607

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  
/-- Represents a point on the parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Represents a chord of the parabola -/
structure Chord where
  p1 : ParabolaPoint
  p2 : ParabolaPoint

/-- Represents the projection of a chord onto the directrix -/
def projection (c : Chord) : ℝ := sorry

/-- Surface area formed by rotating a chord around the directrix -/
def surfaceAreaRotation (c : Chord) : ℝ := sorry

/-- Surface area of a sphere with given diameter -/
def surfaceAreaSphere (diameter : ℝ) : ℝ := sorry

/-- Theorem stating that the surface area of rotation is greater than or equal to
    the surface area of the sphere formed by the projection -/
theorem surface_area_comparison
  (para : Parabola) (c : Chord) 
  (h1 : c.p1.y^2 = 2 * para.p * c.p1.x)
  (h2 : c.p2.y^2 = 2 * para.p * c.p2.x)
  (h3 : c.p1.x + c.p2.x = 2 * para.p) -- chord passes through focus
  : surfaceAreaRotation c ≥ surfaceAreaSphere (projection c) := by
  sorry

end NUMINAMATH_CALUDE_surface_area_comparison_l3616_361607


namespace NUMINAMATH_CALUDE_cube_decomposition_largest_number_l3616_361630

theorem cube_decomposition_largest_number :
  let n : ℕ := 10
  let sum_of_terms : ℕ → ℕ := λ k => k * (k + 1) / 2
  let total_terms : ℕ := sum_of_terms n - sum_of_terms 1
  2 * total_terms + 1 = 109 :=
by sorry

end NUMINAMATH_CALUDE_cube_decomposition_largest_number_l3616_361630


namespace NUMINAMATH_CALUDE_conference_left_handed_fraction_l3616_361663

theorem conference_left_handed_fraction :
  ∀ (total : ℕ) (red blue left_handed_red left_handed_blue : ℕ),
    red + blue = total →
    red = 7 * (total / 10) →
    blue = 3 * (total / 10) →
    left_handed_red = red / 3 →
    left_handed_blue = 2 * blue / 3 →
    (left_handed_red + left_handed_blue : ℚ) / total = 13 / 30 :=
by sorry

end NUMINAMATH_CALUDE_conference_left_handed_fraction_l3616_361663


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3616_361649

/-- Calculates the molecular weight of a compound given the atomic weights and number of atoms of each element. -/
def molecular_weight (al_weight o_weight h_weight : ℝ) (al_count o_count h_count : ℕ) : ℝ :=
  al_weight * al_count + o_weight * o_count + h_weight * h_count

/-- Theorem stating that the molecular weight of a compound with 1 Aluminium, 3 Oxygen, and 3 Hydrogen atoms is 78.001 g/mol. -/
theorem compound_molecular_weight :
  let al_weight : ℝ := 26.98
  let o_weight : ℝ := 15.999
  let h_weight : ℝ := 1.008
  let al_count : ℕ := 1
  let o_count : ℕ := 3
  let h_count : ℕ := 3
  molecular_weight al_weight o_weight h_weight al_count o_count h_count = 78.001 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3616_361649


namespace NUMINAMATH_CALUDE_total_production_all_companies_l3616_361687

/-- Represents the production of cars by a company in different continents -/
structure CarProduction where
  northAmerica : ℕ
  europe : ℕ
  asia : ℕ

/-- Calculates the total production for a company -/
def totalProduction (p : CarProduction) : ℕ :=
  p.northAmerica + p.europe + p.asia

/-- The production data for Car Company A -/
def companyA : CarProduction :=
  { northAmerica := 3884
    europe := 2871
    asia := 1529 }

/-- The production data for Car Company B -/
def companyB : CarProduction :=
  { northAmerica := 4357
    europe := 3690
    asia := 1835 }

/-- The production data for Car Company C -/
def companyC : CarProduction :=
  { northAmerica := 2937
    europe := 4210
    asia := 977 }

/-- Theorem stating that the total production of all companies is 26,290 -/
theorem total_production_all_companies :
  totalProduction companyA + totalProduction companyB + totalProduction companyC = 26290 := by
  sorry

end NUMINAMATH_CALUDE_total_production_all_companies_l3616_361687


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3616_361614

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

-- Define an increasing sequence
def increasing_sequence (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : increasing_sequence a)
  (h3 : a 2 = 2)
  (h4 : a 4 - a 3 = 4) :
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3616_361614


namespace NUMINAMATH_CALUDE_cannot_achieve_multiple_100s_l3616_361618

/-- Represents the scores for Russian, Physics, and Mathematics exams -/
structure Scores where
  russian : ℕ
  physics : ℕ
  math : ℕ

/-- Defines the initial relationship between scores -/
def initial_score_relationship (s : Scores) : Prop :=
  s.russian = s.physics - 5 ∧ s.physics = s.math - 9

/-- Represents the two types of operations allowed -/
inductive Operation
  | add_one_to_all
  | decrease_one_increase_two

/-- Applies an operation to the scores -/
def apply_operation (s : Scores) (op : Operation) : Scores :=
  match op with
  | Operation.add_one_to_all => 
      { russian := s.russian + 1, physics := s.physics + 1, math := s.math + 1 }
  | Operation.decrease_one_increase_two => 
      { russian := s.russian - 3, physics := s.physics + 1, math := s.math + 1 }
      -- Note: This is just one possible application of the second operation

/-- Checks if any score exceeds 100 -/
def exceeds_100 (s : Scores) : Prop :=
  s.russian > 100 ∨ s.physics > 100 ∨ s.math > 100

/-- Checks if more than one score is equal to 100 -/
def more_than_one_100 (s : Scores) : Prop :=
  (s.russian = 100 ∧ s.physics = 100) ∨
  (s.russian = 100 ∧ s.math = 100) ∨
  (s.physics = 100 ∧ s.math = 100)

/-- The main theorem to be proved -/
theorem cannot_achieve_multiple_100s (s : Scores) 
  (h : initial_score_relationship s) : 
  ¬ ∃ (ops : List Operation), 
    let final_scores := ops.foldl apply_operation s
    ¬ exceeds_100 final_scores ∧ more_than_one_100 final_scores :=
sorry


end NUMINAMATH_CALUDE_cannot_achieve_multiple_100s_l3616_361618


namespace NUMINAMATH_CALUDE_oven_temperature_l3616_361682

/-- Represents the temperature of the steak at time t -/
noncomputable def T (t : ℝ) : ℝ := sorry

/-- The constant oven temperature -/
def T_o : ℝ := sorry

/-- The initial temperature of the steak -/
def T_i : ℝ := 5

/-- The constant of proportionality in Newton's Law of Cooling -/
noncomputable def k : ℝ := sorry

/-- Newton's Law of Cooling: The rate of change of the steak's temperature
    is proportional to the difference between the steak's temperature and the oven temperature -/
axiom newtons_law_cooling : ∀ t, (deriv T t) = k * (T_o - T t)

/-- The solution to Newton's Law of Cooling -/
axiom cooling_solution : ∀ t, T t = T_o + (T_i - T_o) * Real.exp (-k * t)

/-- The temperature after 15 minutes is 45°C -/
axiom temp_at_15 : T 15 = 45

/-- The temperature after 30 minutes is 77°C -/
axiom temp_at_30 : T 30 = 77

/-- The theorem stating that the oven temperature is 205°C -/
theorem oven_temperature : T_o = 205 := by sorry

end NUMINAMATH_CALUDE_oven_temperature_l3616_361682


namespace NUMINAMATH_CALUDE_min_value_expression_l3616_361667

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a*b*c) ≥ 216 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3616_361667


namespace NUMINAMATH_CALUDE_slope_angle_l3616_361680

def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 2 = 0

theorem slope_angle (x y : ℝ) (h : line_equation x y) : 
  ∃ (θ : ℝ), θ = 120 * Real.pi / 180 ∧ Real.tan θ = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_slope_angle_l3616_361680


namespace NUMINAMATH_CALUDE_function_expression_l3616_361670

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (x + 2) = x^2 - x + 1) :
  ∀ x, f x = x^2 - 5*x + 7 := by
sorry

end NUMINAMATH_CALUDE_function_expression_l3616_361670


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3616_361641

theorem complex_magnitude_problem (z : ℂ) (h : 3 * z * Complex.I = -6 + 2 * Complex.I) :
  Complex.abs z = 2 * Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3616_361641


namespace NUMINAMATH_CALUDE_simplify_expression_find_value_evaluate_compound_l3616_361603

-- Part 1
theorem simplify_expression (a b : ℝ) :
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 := by sorry

-- Part 2
theorem find_value (x y : ℝ) (h : x^2 - 2*y = 4) :
  3 * x^2 - 6 * y - 21 = -9 := by sorry

-- Part 3
theorem evaluate_compound (a b c d : ℝ) 
  (h1 : a - 2*b = 6) (h2 : 2*b - c = -8) (h3 : c - d = 9) :
  (a - c) + (2*b - d) - (2*b - c) = 7 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_find_value_evaluate_compound_l3616_361603


namespace NUMINAMATH_CALUDE_cube_root_bound_l3616_361637

theorem cube_root_bound (n : ℕ) (hn : n ≥ 2) :
  (n : ℝ) + 0.6 < (((n : ℝ)^3 + 2*(n : ℝ)^2 + (n : ℝ))^(1/3 : ℝ)) ∧
  (((n : ℝ)^3 + 2*(n : ℝ)^2 + (n : ℝ))^(1/3 : ℝ)) < (n : ℝ) + 0.7 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_bound_l3616_361637


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_fifteen_fourths_l3616_361640

theorem floor_plus_self_eq_fifteen_fourths :
  ∃! (x : ℚ), ⌊x⌋ + x = 15/4 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_fifteen_fourths_l3616_361640


namespace NUMINAMATH_CALUDE_f_not_valid_mapping_l3616_361662

-- Define the sets M and P
def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 6}
def P : Set ℝ := {y | 0 ≤ y ∧ y ≤ 3}

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x

-- Theorem stating that f is not a valid mapping from M to P
theorem f_not_valid_mapping : ¬(∀ x ∈ M, f x ∈ P) := by
  sorry


end NUMINAMATH_CALUDE_f_not_valid_mapping_l3616_361662


namespace NUMINAMATH_CALUDE_sum_of_solutions_l3616_361669

noncomputable def solution_sum : ℝ → Prop :=
  fun x ↦ (x^2 - 6*x - 3 = 0) ∧ (x ≠ 1) ∧ (x ≠ -1)

theorem sum_of_solutions :
  ∃ (a b : ℝ), solution_sum a ∧ solution_sum b ∧ a + b = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l3616_361669


namespace NUMINAMATH_CALUDE_daily_wage_calculation_l3616_361644

/-- Proves the daily wage for a worker given total days, idle days, total pay, and idle day deduction --/
theorem daily_wage_calculation (total_days idle_days : ℕ) (total_pay idle_day_deduction : ℚ) :
  total_days = 60 →
  idle_days = 40 →
  total_pay = 280 →
  idle_day_deduction = 3 →
  ∃ (daily_wage : ℚ), 
    daily_wage * (total_days - idle_days : ℚ) - idle_day_deduction * idle_days = total_pay ∧
    daily_wage = 20 := by
  sorry

end NUMINAMATH_CALUDE_daily_wage_calculation_l3616_361644


namespace NUMINAMATH_CALUDE_a_greater_than_b_l3616_361621

theorem a_greater_than_b (n : ℕ) (a b : ℝ) 
  (h_n : n ≥ 2) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) 
  (h_a : a^n = a + 1) 
  (h_b : b^(2*n) = b + 3*a) : 
  a > b := by sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l3616_361621
