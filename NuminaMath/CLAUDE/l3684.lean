import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3684_368436

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define the property of being a positive sequence
def IsPositive (a : Sequence) : Prop :=
  ∀ n : ℕ, a n > 0

-- Define the recurrence relation
def SatisfiesRecurrence (a : Sequence) : Prop :=
  ∀ n : ℕ, a (n + 2) = a n * a (n + 1)

-- Define the property of being a geometric sequence
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_condition (a : Sequence) 
  (h_pos : IsPositive a) (h_rec : SatisfiesRecurrence a) :
  IsGeometric a ↔ a 1 = 1 ∧ a 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l3684_368436


namespace NUMINAMATH_CALUDE_max_product_sum_l3684_368494

theorem max_product_sum (X Y Z : ℕ) (sum_constraint : X + Y + Z = 15) :
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 200 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l3684_368494


namespace NUMINAMATH_CALUDE_worker_a_time_l3684_368482

theorem worker_a_time (b_time : ℝ) (combined_time : ℝ) (a_time : ℝ) : 
  b_time = 10 →
  combined_time = 4.444444444444445 →
  (1 / a_time + 1 / b_time = 1 / combined_time) →
  a_time = 8 := by
    sorry

end NUMINAMATH_CALUDE_worker_a_time_l3684_368482


namespace NUMINAMATH_CALUDE_sarah_initial_trucks_l3684_368401

/-- The number of trucks Sarah gave to Jeff -/
def trucks_given_to_jeff : ℕ := 13

/-- The number of trucks Sarah has left -/
def trucks_left : ℕ := 38

/-- The initial number of trucks Sarah had -/
def initial_trucks : ℕ := trucks_given_to_jeff + trucks_left

theorem sarah_initial_trucks : initial_trucks = 51 := by sorry

end NUMINAMATH_CALUDE_sarah_initial_trucks_l3684_368401


namespace NUMINAMATH_CALUDE_third_month_sales_l3684_368463

def sales_1 : ℕ := 3435
def sales_2 : ℕ := 3927
def sales_4 : ℕ := 4230
def sales_5 : ℕ := 3562
def sales_6 : ℕ := 1991
def target_average : ℕ := 3500
def num_months : ℕ := 6

theorem third_month_sales :
  sales_1 + sales_2 + sales_4 + sales_5 + sales_6 + 3855 = target_average * num_months :=
by sorry

end NUMINAMATH_CALUDE_third_month_sales_l3684_368463


namespace NUMINAMATH_CALUDE_perimeter_ABCDE_l3684_368461

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom AB_eq_BC : dist A B = dist B C
axiom AB_eq_4 : dist A B = 4
axiom AE_eq_8 : dist A E = 8
axiom ED_eq_7 : dist E D = 7
axiom angle_AED_right : (E.1 - A.1) * (E.1 - D.1) + (E.2 - A.2) * (E.2 - D.2) = 0
axiom angle_ABC_right : (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0

-- Define the theorem
theorem perimeter_ABCDE :
  dist A B + dist B C + dist C D + dist D E + dist E A = 28 :=
sorry

end NUMINAMATH_CALUDE_perimeter_ABCDE_l3684_368461


namespace NUMINAMATH_CALUDE_iphone_price_drop_l3684_368413

/-- Calculates the final price of an iPhone after two consecutive price drops -/
theorem iphone_price_drop (initial_price : ℝ) (first_drop : ℝ) (second_drop : ℝ) :
  initial_price = 1000 ∧ first_drop = 0.1 ∧ second_drop = 0.2 →
  initial_price * (1 - first_drop) * (1 - second_drop) = 720 := by
  sorry


end NUMINAMATH_CALUDE_iphone_price_drop_l3684_368413


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3684_368479

/-- The equations of the asymptotes of the hyperbola y²/9 - x²/4 = 1 are y = ±(3/2)x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → ℝ := λ x y => y^2/9 - x^2/4 - 1
  ∀ x y : ℝ, h x y = 0 →
  ∃ k : ℝ, k = 3/2 ∧
  (∀ ε > 0, ∃ M : ℝ, ∀ x y : ℝ, h x y = 0 ∧ |x| > M →
    (|y - k*x| < ε*|x| ∨ |y + k*x| < ε*|x|)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3684_368479


namespace NUMINAMATH_CALUDE_min_PM_AB_implies_AB_equation_l3684_368476

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x + y + 2 = 0

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define tangent points A and B on circle M
def tangent_points (xA yA xB yB : ℝ) : Prop :=
  circle_M xA yA ∧ circle_M xB yB

-- Define the minimization condition
def min_condition (xP yP xM yM xA yA xB yB : ℝ) : Prop :=
  ∀ x y, point_P x y →
    (x - xM)^2 + (y - yM)^2 ≤ (xP - xM)^2 + (yP - yM)^2

-- Theorem statement
theorem min_PM_AB_implies_AB_equation :
  ∀ xP yP xM yM xA yA xB yB,
    point_P xP yP →
    tangent_points xA yA xB yB →
    min_condition xP yP xM yM xA yA xB yB →
    2*xA + yA + 1 = 0 ∧ 2*xB + yB + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_PM_AB_implies_AB_equation_l3684_368476


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3684_368451

def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {1, 2}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3684_368451


namespace NUMINAMATH_CALUDE_cupcake_ratio_l3684_368430

/-- Proves that the ratio of gluten-free cupcakes to total cupcakes is 3/20 given the specified conditions --/
theorem cupcake_ratio : 
  ∀ (total vegan non_vegan gluten_free : ℕ),
    total = 80 →
    vegan = 24 →
    non_vegan = 28 →
    gluten_free = vegan / 2 →
    (gluten_free : ℚ) / total = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_ratio_l3684_368430


namespace NUMINAMATH_CALUDE_family_ages_sum_l3684_368423

/-- Represents the ages of family members -/
structure FamilyAges where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- The sum of ages 5 years ago -/
def sumAgesFiveYearsAgo (ages : FamilyAges) : ℕ :=
  ages.a + ages.b + ages.c + ages.d

/-- The sum of ages today with daughter-in-law -/
def sumAgesToday (ages : FamilyAges) (daughterInLawAge : ℕ) : ℕ :=
  ages.a + 5 + ages.b + 5 + ages.c + 5 + daughterInLawAge + 5

theorem family_ages_sum (ages : FamilyAges) :
  sumAgesFiveYearsAgo ages = 94 →
  ∃ (daughterInLawAge : ℕ), 
    daughterInLawAge = ages.d - 14 ∧
    sumAgesToday ages daughterInLawAge = 120 :=
by
  sorry


end NUMINAMATH_CALUDE_family_ages_sum_l3684_368423


namespace NUMINAMATH_CALUDE_price_difference_l3684_368425

/-- The original price of the toy rabbit -/
def original_price : ℝ := 25

/-- The price increase percentage for Store A -/
def increase_percentage : ℝ := 0.1

/-- The price decrease percentage for Store A -/
def decrease_percentage_A : ℝ := 0.2

/-- The price decrease percentage for Store B -/
def decrease_percentage_B : ℝ := 0.1

/-- The final price of the toy rabbit in Store A -/
def price_A : ℝ := original_price * (1 + increase_percentage) * (1 - decrease_percentage_A)

/-- The final price of the toy rabbit in Store B -/
def price_B : ℝ := original_price * (1 - decrease_percentage_B)

/-- Theorem stating that the price in Store A is 0.5 yuan less than in Store B -/
theorem price_difference : price_B - price_A = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_l3684_368425


namespace NUMINAMATH_CALUDE_olivia_coins_left_l3684_368443

/-- The number of coins Olivia has left after buying a soda -/
def coins_left (initial_quarters : ℕ) (spent_quarters : ℕ) : ℕ :=
  initial_quarters - spent_quarters

/-- Theorem: Olivia has 7 coins left after buying a soda -/
theorem olivia_coins_left : coins_left 11 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_olivia_coins_left_l3684_368443


namespace NUMINAMATH_CALUDE_discount_is_twenty_percent_l3684_368408

/-- Calculates the discount percentage given the original price, quantity, tax rate, and final price --/
def calculate_discount_percentage (original_price quantity : ℕ) (tax_rate final_price : ℚ) : ℚ :=
  let discounted_price := final_price / (1 + tax_rate) / quantity
  let discount_amount := original_price - discounted_price
  (discount_amount / original_price) * 100

/-- The discount percentage is 20% given the problem conditions --/
theorem discount_is_twenty_percent :
  calculate_discount_percentage 45 10 (1/10) 396 = 20 := by
  sorry

end NUMINAMATH_CALUDE_discount_is_twenty_percent_l3684_368408


namespace NUMINAMATH_CALUDE_square_root_of_four_l3684_368448

theorem square_root_of_four : ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l3684_368448


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l3684_368452

theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo (π / 6) (π / 3), 
    Monotone (fun x => (a - Real.sin x) / Real.cos x)) →
  a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l3684_368452


namespace NUMINAMATH_CALUDE_exist_two_N_l3684_368465

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the condition for point M
def M_condition (x y : ℝ) : Prop :=
  Real.sqrt ((x+1)^2 + y^2) + Real.sqrt ((x-1)^2 + y^2) = 2 * Real.sqrt 2

-- Define the line l
def line_l (x : ℝ) : Prop := x = -1/2

-- Define the property that N is the midpoint of AB
def is_midpoint (N A B : ℝ × ℝ) : Prop :=
  N.1 = (A.1 + B.1) / 2 ∧ N.2 = (A.2 + B.2) / 2

-- Define the property that PQ is perpendicular bisector of AB
def is_perp_bisector (P Q A B : ℝ × ℝ) : Prop :=
  (P.1 - Q.1) * (A.1 - B.1) + (P.2 - Q.2) * (A.2 - B.2) = 0 ∧
  (P.1 + Q.1) / 2 = (A.1 + B.1) / 2 ∧
  (P.2 + Q.2) / 2 = (A.2 + B.2) / 2

-- Define the property that (1,0) is on the circle with diameter PQ
def on_circle_PQ (P Q : ℝ × ℝ) : Prop :=
  (1 - P.1) * (1 - Q.1) + (-P.2) * (-Q.2) = 0

-- Main theorem
theorem exist_two_N :
  ∃ N1 N2 : ℝ × ℝ,
    N1 ≠ N2 ∧
    line_l N1.1 ∧ line_l N2.1 ∧
    (∃ A B P Q : ℝ × ℝ,
      E A.1 A.2 ∧ E B.1 B.2 ∧ E P.1 P.2 ∧ E Q.1 Q.2 ∧
      is_midpoint N1 A B ∧
      is_perp_bisector P Q A B ∧
      on_circle_PQ P Q) ∧
    (∃ A B P Q : ℝ × ℝ,
      E A.1 A.2 ∧ E B.1 B.2 ∧ E P.1 P.2 ∧ E Q.1 Q.2 ∧
      is_midpoint N2 A B ∧
      is_perp_bisector P Q A B ∧
      on_circle_PQ P Q) ∧
    N1 = (-1/2, Real.sqrt 19 / 19) ∧
    N2 = (-1/2, -Real.sqrt 19 / 19) ∧
    (∀ N : ℝ × ℝ,
      line_l N.1 →
      (∃ A B P Q : ℝ × ℝ,
        E A.1 A.2 ∧ E B.1 B.2 ∧ E P.1 P.2 ∧ E Q.1 Q.2 ∧
        is_midpoint N A B ∧
        is_perp_bisector P Q A B ∧
        on_circle_PQ P Q) →
      N = N1 ∨ N = N2) :=
sorry

end NUMINAMATH_CALUDE_exist_two_N_l3684_368465


namespace NUMINAMATH_CALUDE_widest_opening_is_f₃_l3684_368467

/-- The quadratic function with the widest opening -/
def widest_opening (f₁ f₂ f₃ f₄ : ℝ → ℝ) : Prop :=
  ∃ (a₁ a₂ a₃ a₄ : ℝ),
    (∀ x, f₁ x = -10 * x^2) ∧
    (∀ x, f₂ x = 2 * x^2) ∧
    (∀ x, f₃ x = (1/100) * x^2) ∧
    (∀ x, f₄ x = -x^2) ∧
    (abs a₃ < abs a₄ ∧ abs a₄ < abs a₂ ∧ abs a₂ < abs a₁)

/-- Theorem stating that f₃ has the widest opening -/
theorem widest_opening_is_f₃ (f₁ f₂ f₃ f₄ : ℝ → ℝ) :
  widest_opening f₁ f₂ f₃ f₄ → (∀ x, f₃ x = (1/100) * x^2) := by
  sorry

end NUMINAMATH_CALUDE_widest_opening_is_f₃_l3684_368467


namespace NUMINAMATH_CALUDE_not_square_p_cubed_plus_p_plus_one_l3684_368468

theorem not_square_p_cubed_plus_p_plus_one (p : ℕ) (hp : Prime p) :
  ¬ ∃ (n : ℕ), n^2 = p^3 + p + 1 := by
  sorry

end NUMINAMATH_CALUDE_not_square_p_cubed_plus_p_plus_one_l3684_368468


namespace NUMINAMATH_CALUDE_evaluate_expression_l3684_368459

theorem evaluate_expression : (27^24) / (81^12) = 3^24 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3684_368459


namespace NUMINAMATH_CALUDE_weight_sum_proof_l3684_368444

/-- Given the weights of four people in pairs, prove that the sum of two specific people's weights can be determined. -/
theorem weight_sum_proof (e f g h : ℝ) 
  (ef_sum : e + f = 280)
  (fg_sum : f + g = 230)
  (gh_sum : g + h = 260) :
  e + h = 310 := by sorry

end NUMINAMATH_CALUDE_weight_sum_proof_l3684_368444


namespace NUMINAMATH_CALUDE_power_function_m_value_l3684_368435

theorem power_function_m_value : ∃! m : ℝ, m^2 - 9*m + 19 = 1 ∧ 2*m^2 - 7*m - 9 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_power_function_m_value_l3684_368435


namespace NUMINAMATH_CALUDE_subtraction_problem_l3684_368457

theorem subtraction_problem (x : ℤ) : x - 46 = 15 → x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3684_368457


namespace NUMINAMATH_CALUDE_water_balloon_count_l3684_368470

/-- The total number of filled water balloons Max and Zach have -/
def total_filled_balloons (max_time max_rate zach_time zach_rate popped : ℕ) : ℕ :=
  max_time * max_rate + zach_time * zach_rate - popped

/-- Theorem: The total number of filled water balloons Max and Zach have is 170 -/
theorem water_balloon_count : total_filled_balloons 30 2 40 3 10 = 170 := by
  sorry

end NUMINAMATH_CALUDE_water_balloon_count_l3684_368470


namespace NUMINAMATH_CALUDE_workshop_workers_l3684_368499

/-- The total number of workers in a workshop -/
def total_workers : ℕ := 49

/-- The average salary of all workers -/
def avg_salary_all : ℕ := 8000

/-- The number of technicians -/
def num_technicians : ℕ := 7

/-- The average salary of technicians -/
def avg_salary_technicians : ℕ := 20000

/-- The average salary of non-technicians -/
def avg_salary_others : ℕ := 6000

/-- Theorem stating that the total number of workers is 49 -/
theorem workshop_workers :
  total_workers = 49 ∧
  avg_salary_all * total_workers = 
    avg_salary_technicians * num_technicians + 
    avg_salary_others * (total_workers - num_technicians) :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l3684_368499


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l3684_368472

theorem coefficient_x_squared (x y : ℝ) : 
  let expansion := (x - 2 * y^3) * (x + 1/y)^5
  ∃ (a b c : ℝ), expansion = a * x^3 + (-20) * x^2 + b * x + c :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_l3684_368472


namespace NUMINAMATH_CALUDE_deepak_age_l3684_368439

theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / (deepak_age : ℚ) = 4 / 3 →
  rahul_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l3684_368439


namespace NUMINAMATH_CALUDE_problem_solution_l3684_368410

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 9| - |x - 5|

-- Define the function y(x)
def y (x : ℝ) : ℝ := f x + 3*|x - 5|

theorem problem_solution :
  -- Part 1: Solution set of f(x) ≥ 2x-1
  (∀ x : ℝ, f x ≥ 2*x - 1 ↔ x ≤ 5/3) ∧
  -- Part 2: Minimum value of y(x)
  (∀ x : ℝ, y x ≥ 1) ∧
  (∃ x : ℝ, y x = 1) ∧
  -- Part 3: Minimum value of a + 3b given 1/a + 3/b = 1
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 3/b = 1 → a + 3*b ≥ 16) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 3/b = 1 ∧ a + 3*b = 16) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3684_368410


namespace NUMINAMATH_CALUDE_factorization_proof_l3684_368416

theorem factorization_proof (x y : ℝ) : x * y^2 - 6 * x * y + 9 * x = x * (y - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3684_368416


namespace NUMINAMATH_CALUDE_binomial_expansion_largest_coefficient_l3684_368491

theorem binomial_expansion_largest_coefficient (n : ℕ) : 
  (∃ k, k = 5 ∧ 
    (∀ j, j ≠ k → Nat.choose n k > Nat.choose n j) ∧
    (∀ j, j < k → Nat.choose n j < Nat.choose n (j+1)) ∧
    (∀ j, k < j ∧ j ≤ n → Nat.choose n j < Nat.choose n (j-1))) →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_largest_coefficient_l3684_368491


namespace NUMINAMATH_CALUDE_crop_yield_growth_l3684_368424

theorem crop_yield_growth (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 1) 
  (h3 : 300 * (1 + x)^2 = 363) : 
  ∃ (initial_yield final_yield : ℝ),
    initial_yield = 300 ∧ 
    final_yield = 363 ∧ 
    final_yield = initial_yield * (1 + x)^2 :=
sorry

end NUMINAMATH_CALUDE_crop_yield_growth_l3684_368424


namespace NUMINAMATH_CALUDE_quarters_to_nickels_difference_l3684_368487

/-- The difference in money (in nickels) between two people given their quarter amounts -/
def money_difference_in_nickels (charles_quarters richard_quarters : ℕ) : ℤ :=
  5 * (charles_quarters - richard_quarters)

theorem quarters_to_nickels_difference (q : ℕ) :
  money_difference_in_nickels (5 * q + 3) (q + 7) = 20 * (q - 1) := by
  sorry

end NUMINAMATH_CALUDE_quarters_to_nickels_difference_l3684_368487


namespace NUMINAMATH_CALUDE_square_circle_area_l3684_368427

theorem square_circle_area (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  let r := d / 2
  s^2 + π * r^2 = 144 + 72 * π := by sorry

end NUMINAMATH_CALUDE_square_circle_area_l3684_368427


namespace NUMINAMATH_CALUDE_function_inequality_relation_l3684_368495

theorem function_inequality_relation (a b : ℝ) (h_pos : a > 0 ∧ b > 0) :
  (∀ x : ℝ, |x + 1| < b → |2*x + 3 - 1| < a) →
  b ≥ a / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_relation_l3684_368495


namespace NUMINAMATH_CALUDE_hillary_activities_lcm_l3684_368449

theorem hillary_activities_lcm : Nat.lcm (Nat.lcm 6 4) 16 = 48 := by sorry

end NUMINAMATH_CALUDE_hillary_activities_lcm_l3684_368449


namespace NUMINAMATH_CALUDE_sum_of_first_2015_digits_l3684_368496

/-- The repeating decimal 0.0142857 -/
def repeatingDecimal : ℚ := 1 / 7

/-- The length of the repeating part of the decimal -/
def repeatLength : ℕ := 6

/-- The sum of digits in one complete cycle of the repeating part -/
def cycleSum : ℕ := 27

/-- The number of complete cycles in the first 2015 digits -/
def completeCycles : ℕ := 2015 / repeatLength

/-- The number of remaining digits after complete cycles -/
def remainingDigits : ℕ := 2015 % repeatLength

/-- The sum of the remaining digits -/
def remainingSum : ℕ := 20

theorem sum_of_first_2015_digits : 
  (cycleSum * completeCycles + remainingSum : ℕ) = 9065 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_2015_digits_l3684_368496


namespace NUMINAMATH_CALUDE_star_symmetry_l3684_368400

/-- The star operation for real numbers -/
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

/-- Theorem: For all real x and y, (x² - y²) ⋆ (y² - x²) = 0 -/
theorem star_symmetry (x y : ℝ) : star (x^2 - y^2) (y^2 - x^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_symmetry_l3684_368400


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l3684_368483

theorem largest_n_with_unique_k : ∀ n : ℕ, n > 24 →
  ¬(∃! k : ℤ, (3 : ℚ)/7 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/19) ∧
  (∃! k : ℤ, (3 : ℚ)/7 < (24 : ℚ)/(24 + k) ∧ (24 : ℚ)/(24 + k) < 8/19) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l3684_368483


namespace NUMINAMATH_CALUDE_cakes_destroyed_or_stolen_proof_l3684_368445

def total_cakes : ℕ := 36
def num_stacks : ℕ := 2

def cakes_per_stack : ℕ := total_cakes / num_stacks

def crow_knocked_percentage : ℚ := 60 / 100
def mischievous_squirrel_stole_fraction : ℚ := 1 / 3
def red_squirrel_took_percentage : ℚ := 25 / 100
def red_squirrel_dropped_fraction : ℚ := 1 / 2
def dog_ate : ℕ := 4

def cakes_destroyed_or_stolen : ℕ := 19

theorem cakes_destroyed_or_stolen_proof :
  let crow_knocked := (crow_knocked_percentage * cakes_per_stack).floor
  let mischievous_squirrel_stole := (mischievous_squirrel_stole_fraction * crow_knocked).floor
  let red_squirrel_took := (red_squirrel_took_percentage * cakes_per_stack).floor
  let red_squirrel_destroyed := (red_squirrel_dropped_fraction * red_squirrel_took).floor
  crow_knocked + mischievous_squirrel_stole + red_squirrel_destroyed + dog_ate = cakes_destroyed_or_stolen :=
by sorry

end NUMINAMATH_CALUDE_cakes_destroyed_or_stolen_proof_l3684_368445


namespace NUMINAMATH_CALUDE_function_shift_l3684_368460

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_shift (x : ℝ) :
  (∀ y : ℝ, f (y + 1) = y^2 + 2*y) →
  f (x - 1) = x^2 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_function_shift_l3684_368460


namespace NUMINAMATH_CALUDE_smallest_divisible_by_11_with_remainders_l3684_368421

theorem smallest_divisible_by_11_with_remainders :
  ∃! n : ℕ, n > 0 ∧ 
    11 ∣ n ∧
    n % 2 = 1 ∧
    n % 3 = 1 ∧
    n % 4 = 1 ∧
    n % 5 = 1 ∧
    ∀ m : ℕ, m > 0 ∧ 
      11 ∣ m ∧
      m % 2 = 1 ∧
      m % 3 = 1 ∧
      m % 4 = 1 ∧
      m % 5 = 1
    → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_11_with_remainders_l3684_368421


namespace NUMINAMATH_CALUDE_princess_daphne_jewelry_cost_l3684_368437

/-- The cost of Princess Daphne's jewelry purchase -/
def total_cost : ℕ := 240000

/-- The cost of a single necklace -/
def necklace_cost : ℕ := 40000

/-- The cost of the earrings -/
def earrings_cost : ℕ := 3 * necklace_cost

theorem princess_daphne_jewelry_cost :
  3 * necklace_cost + earrings_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_princess_daphne_jewelry_cost_l3684_368437


namespace NUMINAMATH_CALUDE_coefficient_x6_sum_binomial_expansions_l3684_368404

theorem coefficient_x6_sum_binomial_expansions :
  let f (n : ℕ) := (1 + X : Polynomial ℚ)^n
  let expansion := f 5 + f 6 + f 7
  (expansion.coeff 6 : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x6_sum_binomial_expansions_l3684_368404


namespace NUMINAMATH_CALUDE_ratio_equality_l3684_368417

theorem ratio_equality {a₁ a₂ a₃ b₁ b₂ b₃ p₁ p₂ p₃ : ℝ} (h1 : a₁ / b₁ = a₂ / b₂) (h2 : a₁ / b₁ = a₃ / b₃)
    (h3 : ¬(p₁ = 0 ∧ p₂ = 0 ∧ p₃ = 0)) :
  ∀ n : ℕ, (a₁ / b₁) ^ n = (p₁ * a₁^n + p₂ * a₂^n + p₃ * a₃^n) / (p₁ * b₁^n + p₂ * b₂^n + p₃ * b₃^n) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3684_368417


namespace NUMINAMATH_CALUDE_dartboard_probability_l3684_368481

/-- The probability of hitting a specific region on a square dartboard -/
theorem dartboard_probability : 
  ∃ (square_side_length : ℝ) (region_area : ℝ → ℝ),
    square_side_length = 2 ∧
    (∀ x, x > 0 → region_area x = (π * x^2) / 4 - x^2 / 2) ∧
    region_area square_side_length / square_side_length^2 = (π - 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_probability_l3684_368481


namespace NUMINAMATH_CALUDE_mean_median_difference_l3684_368493

/-- Represents the score distribution in the exam -/
structure ScoreDistribution where
  score_60 : Rat
  score_75 : Rat
  score_80 : Rat
  score_85 : Rat
  score_90 : Rat
  score_100 : Rat

/-- The given score distribution from the problem -/
def exam_distribution : ScoreDistribution :=
  { score_60 := 5/100
    score_75 := 10/100
    score_80 := 30/100
    score_85 := 25/100
    score_90 := 15/100
    score_100 := 15/100 }

/-- Calculate the mean score given a score distribution -/
def mean_score (d : ScoreDistribution) : Rat :=
  (60 * d.score_60 + 75 * d.score_75 + 80 * d.score_80 +
   85 * d.score_85 + 90 * d.score_90 + 100 * d.score_100)

/-- Calculate the median score given a score distribution -/
def median_score (d : ScoreDistribution) : Rat := 85

/-- Theorem: The absolute difference between the mean and median score is 0.75 -/
theorem mean_median_difference (d : ScoreDistribution) :
  d = exam_distribution →
  |mean_score d - median_score d| = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l3684_368493


namespace NUMINAMATH_CALUDE_triangle_parallelogram_area_relation_l3684_368480

theorem triangle_parallelogram_area_relation (x : ℝ) :
  let triangle_base := x - 2
  let triangle_height := x - 2
  let parallelogram_base := x - 3
  let parallelogram_height := x + 4
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let parallelogram_area := parallelogram_base * parallelogram_height
  parallelogram_area = 3 * triangle_area →
  (∀ y : ℝ, (y - 8) * (y - 3) = 0 ↔ y = x) →
  8 + 3 = 11 :=
by sorry

end NUMINAMATH_CALUDE_triangle_parallelogram_area_relation_l3684_368480


namespace NUMINAMATH_CALUDE_biased_coin_expected_value_l3684_368441

def coin_flip_expected_value (p_heads : ℚ) (p_tails : ℚ) (win_heads : ℚ) (win_tails : ℚ) : ℚ :=
  p_heads * win_heads + p_tails * win_tails

theorem biased_coin_expected_value :
  let p_heads : ℚ := 2/5
  let p_tails : ℚ := 3/5
  let win_heads : ℚ := 5
  let win_tails : ℚ := -4
  coin_flip_expected_value p_heads p_tails win_heads win_tails = -2/5 := by
sorry

end NUMINAMATH_CALUDE_biased_coin_expected_value_l3684_368441


namespace NUMINAMATH_CALUDE_gordons_second_restaurant_meals_l3684_368403

/-- Given Gordon's restaurants and their meal serving information, prove that the second restaurant serves 40 meals per day. -/
theorem gordons_second_restaurant_meals (total_weekly_meals : ℕ)
  (first_restaurant_daily_meals : ℕ) (third_restaurant_daily_meals : ℕ)
  (h1 : total_weekly_meals = 770)
  (h2 : first_restaurant_daily_meals = 20)
  (h3 : third_restaurant_daily_meals = 50) :
  ∃ (second_restaurant_daily_meals : ℕ),
    second_restaurant_daily_meals = 40 ∧
    total_weekly_meals = 7 * (first_restaurant_daily_meals + second_restaurant_daily_meals + third_restaurant_daily_meals) :=
by sorry

end NUMINAMATH_CALUDE_gordons_second_restaurant_meals_l3684_368403


namespace NUMINAMATH_CALUDE_complex_magnitude_range_l3684_368486

theorem complex_magnitude_range (z : ℂ) (h : Complex.abs z = 1) :
  4 * Real.sqrt 2 ≤ Complex.abs ((z + 1) + Complex.I * (7 - z)) ∧
  Complex.abs ((z + 1) + Complex.I * (7 - z)) ≤ 6 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_range_l3684_368486


namespace NUMINAMATH_CALUDE_hundredth_digit_of_17_over_99_l3684_368411

/-- The 100th digit after the decimal point in the decimal representation of 17/99 is 7 -/
theorem hundredth_digit_of_17_over_99 : ∃ (d : ℕ), d = 7 ∧ 
  (∃ (a b : ℕ) (s : List ℕ), 
    (17 : ℚ) / 99 = (a : ℚ) + (b : ℚ) / 10 + (s.foldr (λ x acc => acc / 10 + (x : ℚ) / 10) 0) ∧
    s.length = 99 ∧
    d = s.reverse.head!) :=
sorry

end NUMINAMATH_CALUDE_hundredth_digit_of_17_over_99_l3684_368411


namespace NUMINAMATH_CALUDE_checkerboard_sums_l3684_368453

/-- Represents a 10x10 checkerboard filled with numbers 1 to 100 -/
def Checkerboard := Fin 10 → Fin 10 → Nat

/-- The checkerboard filled with numbers 1 to 100 in order -/
def filledCheckerboard : Checkerboard :=
  fun i j => i.val * 10 + j.val + 1

/-- The sum of the corner numbers on the checkerboard -/
def cornerSum (board : Checkerboard) : Nat :=
  board 0 0 + board 0 9 + board 9 0 + board 9 9

/-- The sum of the main diagonal numbers on the checkerboard -/
def diagonalSum (board : Checkerboard) : Nat :=
  board 0 0 + board 9 9

theorem checkerboard_sums :
  cornerSum filledCheckerboard = 202 ∧
  diagonalSum filledCheckerboard = 101 := by
  sorry


end NUMINAMATH_CALUDE_checkerboard_sums_l3684_368453


namespace NUMINAMATH_CALUDE_parabola_through_point_l3684_368419

/-- The value of 'a' for a parabola y = ax^2 passing through (-1, 2) -/
theorem parabola_through_point (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) → 2 = a * (-1)^2 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_l3684_368419


namespace NUMINAMATH_CALUDE_point_on_line_l3684_368478

/-- 
Given two points (m, n) and (m + 2, n + some_value) that lie on the line x = (y/2) - (2/5),
prove that some_value must equal 4.
-/
theorem point_on_line (m n some_value : ℝ) : 
  (m = n / 2 - 2 / 5) ∧ (m + 2 = (n + some_value) / 2 - 2 / 5) → some_value = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l3684_368478


namespace NUMINAMATH_CALUDE_cube_folding_preserves_adjacency_l3684_368475

/-- Represents a face of the cube -/
inductive Face : Type
| One
| Two
| Three
| Four
| Five
| Six

/-- Represents the net of the cube -/
structure CubeNet :=
(faces : List Face)
(adjacent : Face → Face → Bool)

/-- Represents the folded cube -/
structure FoldedCube :=
(faces : List Face)
(adjacent : Face → Face → Bool)

/-- Theorem stating that the face adjacencies in the folded cube
    must match the adjacencies in the original net -/
theorem cube_folding_preserves_adjacency (net : CubeNet) (cube : FoldedCube) :
  (net.faces = cube.faces) →
  (∀ (f1 f2 : Face), net.adjacent f1 f2 = cube.adjacent f1 f2) :=
sorry

end NUMINAMATH_CALUDE_cube_folding_preserves_adjacency_l3684_368475


namespace NUMINAMATH_CALUDE_circle_equation_l3684_368412

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The equation of a line ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem circle_equation (C : Circle) (L : Line) (P1 P2 : Point) :
  L.a = 2 ∧ L.b = 1 ∧ L.c = -1 →  -- Line equation: 2x + y - 1 = 0
  C.h * L.a + C.k * L.b + L.c = 0 →  -- Center is on the line
  (0 - C.h)^2 + (0 - C.k)^2 = C.r^2 →  -- Circle passes through origin
  (P1.x - C.h)^2 + (P1.y - C.k)^2 = C.r^2 →  -- Circle passes through P1
  P1.x = -1 ∧ P1.y = -5 →  -- P1 coordinates
  C.h = 2 ∧ C.k = -3 ∧ C.r^2 = 13 →  -- Circle equation coefficients
  ∀ x y : ℝ, (x - C.h)^2 + (y - C.k)^2 = C.r^2 ↔ (x - 2)^2 + (y + 3)^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3684_368412


namespace NUMINAMATH_CALUDE_basketball_score_proof_l3684_368431

theorem basketball_score_proof (total_score : ℕ) (two_point_shots : ℕ) (three_point_shots : ℕ) :
  total_score = 16 ∧
  two_point_shots = three_point_shots + 3 ∧
  2 * two_point_shots + 3 * three_point_shots = total_score →
  three_point_shots = 2 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l3684_368431


namespace NUMINAMATH_CALUDE_ratio_from_equation_l3684_368442

theorem ratio_from_equation (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_from_equation_l3684_368442


namespace NUMINAMATH_CALUDE_rational_sequence_to_integer_l3684_368456

theorem rational_sequence_to_integer (x : ℚ) : 
  ∃ (f : ℕ → ℚ), 
    f 0 = x ∧ 
    (∀ n : ℕ, n ≥ 1 → (f n = 2 * f (n - 1) ∨ f n = 2 * f (n - 1) + 1 / n)) ∧
    (∃ k : ℕ, ∃ m : ℤ, f k = m) := by
  sorry

end NUMINAMATH_CALUDE_rational_sequence_to_integer_l3684_368456


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_integers_l3684_368450

/-- Two-digit positive integer -/
def TwoDigitPositiveInteger (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem max_ratio_two_digit_integers (x y : ℕ) :
  TwoDigitPositiveInteger x →
  TwoDigitPositiveInteger y →
  (x + y) / 2 = 55 →
  ∀ a b : ℕ, TwoDigitPositiveInteger a → TwoDigitPositiveInteger b → (a + b) / 2 = 55 →
    (a : ℚ) / b ≤ 79 / 31 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_integers_l3684_368450


namespace NUMINAMATH_CALUDE_expansion_coefficients_l3684_368497

theorem expansion_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ = 1 ∧ a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -128) := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l3684_368497


namespace NUMINAMATH_CALUDE_slope_from_sin_cos_sum_l3684_368426

theorem slope_from_sin_cos_sum (θ : Real) 
  (h : Real.sin θ + Real.cos θ = Real.sqrt 5 / 5) : 
  Real.tan θ = -2 := by
  sorry

end NUMINAMATH_CALUDE_slope_from_sin_cos_sum_l3684_368426


namespace NUMINAMATH_CALUDE_polynomial_problem_l3684_368484

/-- Given polynomial P = 2(ax-3) - 3(bx+5) -/
def P (a b x : ℝ) : ℝ := 2*(a*x - 3) - 3*(b*x + 5)

theorem polynomial_problem (a b : ℝ) (h1 : P a b 2 = -31) (h2 : a + b = 0) :
  (a = -1 ∧ b = 1) ∧ 
  (∀ x : ℤ, P a b x > 0 → x ≤ -5) ∧
  (P a b (-5 : ℝ) > 0) :=
sorry

end NUMINAMATH_CALUDE_polynomial_problem_l3684_368484


namespace NUMINAMATH_CALUDE_triangle_circumradius_l3684_368469

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = √3 and c - 2b + 2√3 cos C = 0, then the radius of the circumcircle is 1. -/
theorem triangle_circumradius (a b c A B C : ℝ) : 
  a = Real.sqrt 3 →
  c - 2*b + 2*(Real.sqrt 3)*(Real.cos C) = 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  (a / (2 * Real.sin A)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l3684_368469


namespace NUMINAMATH_CALUDE_complex_fraction_equals_negative_i_negative_i_coordinates_l3684_368446

theorem complex_fraction_equals_negative_i :
  let z : ℂ := (1 - 2*I) / (2 + I)
  z = -I :=
by sorry

theorem negative_i_coordinates :
  let z : ℂ := -I
  Complex.re z = 0 ∧ Complex.im z = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_negative_i_negative_i_coordinates_l3684_368446


namespace NUMINAMATH_CALUDE_inequality_always_true_l3684_368477

theorem inequality_always_true (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l3684_368477


namespace NUMINAMATH_CALUDE_geometric_sum_five_terms_l3684_368455

/-- Given a geometric sequence with first term a and common ratio r,
    find n such that the sum of the first n terms is equal to s. -/
def find_n_for_geometric_sum (a r s : ℚ) : ℕ :=
  sorry

theorem geometric_sum_five_terms
  (a r : ℚ)
  (h_a : a = 1/3)
  (h_r : r = 1/3)
  (h_sum : (a * (1 - r^5)) / (1 - r) = 80/243) :
  find_n_for_geometric_sum a r (80/243) = 5 :=
sorry

end NUMINAMATH_CALUDE_geometric_sum_five_terms_l3684_368455


namespace NUMINAMATH_CALUDE_expenditure_savings_ratio_l3684_368474

/-- Given an income, expenditure, and savings, proves that the ratio of expenditure to savings is 1.5:1 -/
theorem expenditure_savings_ratio 
  (income expenditure savings : ℝ) 
  (h1 : income = expenditure + savings)
  (h2 : 1.15 * income = 1.21 * expenditure + 1.06 * savings) : 
  expenditure = 1.5 * savings := by
  sorry

end NUMINAMATH_CALUDE_expenditure_savings_ratio_l3684_368474


namespace NUMINAMATH_CALUDE_a7_not_prime_l3684_368434

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Defines the sequence a_n -/
def a : ℕ → ℕ
  | 0 => 170  -- Initial value
  | n + 1 => a n + reverseDigits (a n)

/-- States that a_7 is not prime -/
theorem a7_not_prime : ¬ Nat.Prime (a 7) := by sorry

end NUMINAMATH_CALUDE_a7_not_prime_l3684_368434


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3684_368432

def polynomial (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 9 * x^3 - 6) + 4 * (7 * x^6 - 2 * x^3 + 8)

theorem sum_of_coefficients : 
  polynomial 1 = 62 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3684_368432


namespace NUMINAMATH_CALUDE_percentage_increase_l3684_368438

theorem percentage_increase (x y : ℝ) (h : x > y) :
  (x - y) / y * 100 = 50 → x = 132 ∧ y = 88 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3684_368438


namespace NUMINAMATH_CALUDE_carlos_initial_blocks_l3684_368485

/-- The number of blocks Carlos gave to Rachel -/
def blocks_given : ℕ := 21

/-- The number of blocks Carlos had left -/
def blocks_left : ℕ := 37

/-- The initial number of blocks Carlos had -/
def initial_blocks : ℕ := blocks_given + blocks_left

theorem carlos_initial_blocks : initial_blocks = 58 := by sorry

end NUMINAMATH_CALUDE_carlos_initial_blocks_l3684_368485


namespace NUMINAMATH_CALUDE_belle_biscuits_l3684_368402

/-- The number of dog biscuits Belle eats every evening -/
def num_biscuits : ℕ := 4

/-- The number of rawhide bones Belle eats every evening -/
def num_bones : ℕ := 2

/-- The cost of one rawhide bone in dollars -/
def cost_bone : ℚ := 1

/-- The cost of one dog biscuit in dollars -/
def cost_biscuit : ℚ := 1/4

/-- The total cost to feed Belle these treats for a week in dollars -/
def total_cost : ℚ := 21

theorem belle_biscuits :
  num_biscuits = 4 ∧
  (7 : ℚ) * (num_bones * cost_bone + num_biscuits * cost_biscuit) = total_cost :=
sorry

end NUMINAMATH_CALUDE_belle_biscuits_l3684_368402


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l3684_368466

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l3684_368466


namespace NUMINAMATH_CALUDE_absolute_value_of_z_l3684_368422

theorem absolute_value_of_z (r : ℝ) (z : ℂ) (h1 : |r| < 3) (h2 : z + 1/z = r) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_z_l3684_368422


namespace NUMINAMATH_CALUDE_perimeter_difference_rectangle_and_squares_l3684_368464

/-- The perimeter of a rectangle with given length and width -/
def rectangle_perimeter (length width : ℕ) : ℕ := 2 * (length + width)

/-- The perimeter of a single unit square -/
def unit_square_perimeter : ℕ := 4

/-- The perimeter of n non-overlapping unit squares arranged in a straight line -/
def n_unit_squares_perimeter (n : ℕ) : ℕ := n * unit_square_perimeter

theorem perimeter_difference_rectangle_and_squares : 
  (rectangle_perimeter 6 1) - (n_unit_squares_perimeter 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_rectangle_and_squares_l3684_368464


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l3684_368414

/-- The number of ways to distribute n indistinguishable objects into k distinct categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of combinations of n items taken k at a time -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem ice_cream_combinations : 
  distribute 5 4 = combinations 8 3 := by sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l3684_368414


namespace NUMINAMATH_CALUDE_evaluate_expression_l3684_368433

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 + 2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3684_368433


namespace NUMINAMATH_CALUDE_sum_of_ratios_is_four_l3684_368462

/-- Given two nonconstant geometric sequences with common first term,
    if the difference of their third terms is four times the difference of their second terms,
    then the sum of their common ratios is 4. -/
theorem sum_of_ratios_is_four (k p r : ℝ) (h_k : k ≠ 0) (h_p : p ≠ 1) (h_r : r ≠ 1) 
    (h_eq : k * p^2 - k * r^2 = 4 * (k * p - k * r)) :
  p + r = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ratios_is_four_l3684_368462


namespace NUMINAMATH_CALUDE_number_of_divisors_of_m_l3684_368489

def m : ℕ := 2^5 * 3^6 * 5^7 * 7^8

theorem number_of_divisors_of_m : 
  (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 3024 :=
sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_m_l3684_368489


namespace NUMINAMATH_CALUDE_negative_point_two_fifth_times_five_fifth_equals_negative_one_l3684_368492

theorem negative_point_two_fifth_times_five_fifth_equals_negative_one :
  (-0.2)^5 * 5^5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_point_two_fifth_times_five_fifth_equals_negative_one_l3684_368492


namespace NUMINAMATH_CALUDE_q_necessary_not_sufficient_for_p_l3684_368429

theorem q_necessary_not_sufficient_for_p :
  (∀ x : ℝ, |x| < 1 → x^2 + x - 6 < 0) ∧
  (∃ x : ℝ, x^2 + x - 6 < 0 ∧ |x| ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_q_necessary_not_sufficient_for_p_l3684_368429


namespace NUMINAMATH_CALUDE_equation_roots_and_expression_l3684_368498

open Real

theorem equation_roots_and_expression (α m : ℝ) : 
  0 < α → α < π →
  (∃ x : ℝ, x^2 + 4 * x * sin (α/2) + m * tan (α/2) = 0 ∧ 
   ∀ y : ℝ, y^2 + 4 * y * sin (α/2) + m * tan (α/2) = 0 → y = x) →
  m + 2 * cos α = 4/3 →
  (0 < m ∧ m ≤ 2) ∧ 
  (1 + sin (2*α) - cos (2*α)) / (1 + tan α) = -5/9 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_and_expression_l3684_368498


namespace NUMINAMATH_CALUDE_hexagram_arrangements_l3684_368409

/-- A regular six-pointed star -/
structure HexagramStar :=
  (points : Fin 6 → Type)

/-- The group of symmetries of a regular six-pointed star -/
def hexagramSymmetries : ℕ := 12

/-- The number of ways to arrange 6 distinct objects -/
def totalArrangements : ℕ := 720

/-- The number of distinct arrangements of 6 objects on a hexagram star -/
def distinctArrangements (star : HexagramStar) : ℕ :=
  totalArrangements / hexagramSymmetries

theorem hexagram_arrangements (star : HexagramStar) :
  distinctArrangements star = 60 := by
  sorry

end NUMINAMATH_CALUDE_hexagram_arrangements_l3684_368409


namespace NUMINAMATH_CALUDE_max_c_trees_l3684_368473

/-- Represents the types of scenic trees -/
inductive TreeType
| A
| B
| C

/-- The price of a tree given its type -/
def price (t : TreeType) : ℕ :=
  match t with
  | TreeType.A => 200
  | TreeType.B => 200
  | TreeType.C => 300

/-- The total budget for purchasing trees -/
def total_budget : ℕ := 220120

/-- The total number of trees to be purchased -/
def total_trees : ℕ := 1000

/-- Theorem stating the maximum number of C-type trees that can be purchased -/
theorem max_c_trees :
  (∃ (a b c : ℕ), a + b + c = total_trees ∧
                   a * price TreeType.A + b * price TreeType.B + c * price TreeType.C ≤ total_budget) →
  (∀ (a b c : ℕ), a + b + c = total_trees →
                   a * price TreeType.A + b * price TreeType.B + c * price TreeType.C ≤ total_budget →
                   c ≤ 201) ∧
  (∃ (a b : ℕ), a + b + 201 = total_trees ∧
                 a * price TreeType.A + b * price TreeType.B + 201 * price TreeType.C ≤ total_budget) :=
by sorry


end NUMINAMATH_CALUDE_max_c_trees_l3684_368473


namespace NUMINAMATH_CALUDE_odd_square_minus_one_multiple_of_24_and_101_case_l3684_368405

theorem odd_square_minus_one_multiple_of_24_and_101_case : 
  (∀ n : ℕ, n > 1 → (2*n + 1)^2 - 1 = 24 * (n * (n + 1) / 2)) ∧ 
  (101^2 - 1 = 10200) := by
  sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_multiple_of_24_and_101_case_l3684_368405


namespace NUMINAMATH_CALUDE_snack_bar_employees_l3684_368415

theorem snack_bar_employees (total : ℕ) (buffet dining : ℕ) (two_restaurants all_restaurants : ℕ) :
  total = 39 →
  buffet = 17 →
  dining = 18 →
  two_restaurants = 4 →
  all_restaurants = 2 →
  ∃ (snack_bar : ℕ),
    snack_bar = 8 ∧
    total = buffet + dining + snack_bar - two_restaurants - 2 * all_restaurants + all_restaurants :=
by sorry

end NUMINAMATH_CALUDE_snack_bar_employees_l3684_368415


namespace NUMINAMATH_CALUDE_triangle_inequality_l3684_368407

/-- Given a triangle ABC with point P inside, prove the inequality involving
    sides and distances from P to the sides. -/
theorem triangle_inequality (a b c d₁ d₂ d₃ S_ABC : ℝ) 
    (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
    (h₄ : d₁ > 0) (h₅ : d₂ > 0) (h₆ : d₃ > 0)
    (h₇ : S_ABC > 0)
    (h₈ : S_ABC = (1/2) * (a * d₁ + b * d₂ + c * d₃)) :
  (a / d₁) + (b / d₂) + (c / d₃) ≥ (a + b + c)^2 / (2 * S_ABC) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3684_368407


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3684_368440

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 75 → 
  E = 4 * F + 15 → 
  D + E + F = 180 → 
  F = 18 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3684_368440


namespace NUMINAMATH_CALUDE_quadratic_not_always_two_roots_l3684_368454

theorem quadratic_not_always_two_roots :
  ∃ (a b c : ℝ), b - c > a ∧ a ≠ 0 ∧ ¬(∃ (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_not_always_two_roots_l3684_368454


namespace NUMINAMATH_CALUDE_floor_sum_example_l3684_368471

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l3684_368471


namespace NUMINAMATH_CALUDE_function_value_at_three_l3684_368490

/-- Given a function f : ℝ → ℝ satisfying f(x) + 2f(1 - x) = 3x^2 for all real x,
    prove that f(3) = -1 -/
theorem function_value_at_three (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 3 * x^2) : 
    f 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_three_l3684_368490


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l3684_368406

/-- Calculates the average speed of a round trip flight with wind effects -/
theorem round_trip_average_speed
  (up_airspeed : ℝ)
  (up_tailwind : ℝ)
  (down_airspeed : ℝ)
  (down_headwind : ℝ)
  (h1 : up_airspeed = 110)
  (h2 : up_tailwind = 20)
  (h3 : down_airspeed = 88)
  (h4 : down_headwind = 15) :
  (up_airspeed + up_tailwind + (down_airspeed - down_headwind)) / 2 = 101.5 := by
sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l3684_368406


namespace NUMINAMATH_CALUDE_negation_equivalence_l3684_368488

theorem negation_equivalence :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3684_368488


namespace NUMINAMATH_CALUDE_union_covers_reals_a_equals_complement_b_l3684_368447

open Set Real

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | -m ≤ x - 2 ∧ x - 2 ≤ m}
def B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 4}

-- Part 1: A ∪ B = ℝ iff m ≥ 4
theorem union_covers_reals (m : ℝ) : A m ∪ B = univ ↔ m ≥ 4 := by sorry

-- Part 2: A = ℝ\B iff 0 < m < 2
theorem a_equals_complement_b (m : ℝ) : A m = Bᶜ ↔ 0 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_union_covers_reals_a_equals_complement_b_l3684_368447


namespace NUMINAMATH_CALUDE_beach_trip_duration_l3684_368420

-- Define the variables
def seashells_per_day : ℕ := 7
def total_seashells : ℕ := 35

-- Define the function to calculate the number of days
def days_at_beach : ℕ := total_seashells / seashells_per_day

-- Theorem statement
theorem beach_trip_duration : days_at_beach = 5 := by
  sorry

end NUMINAMATH_CALUDE_beach_trip_duration_l3684_368420


namespace NUMINAMATH_CALUDE_vehicle_speed_problem_l3684_368418

/-- Proves that the average speed of vehicle X is 36 miles per hour given the conditions -/
theorem vehicle_speed_problem (initial_distance : ℝ) (y_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  initial_distance = 22 →
  y_speed = 45 →
  time = 5 →
  final_distance = 23 →
  let x_distance := y_speed * time - (initial_distance + final_distance)
  let x_speed := x_distance / time
  x_speed = 36 := by sorry

end NUMINAMATH_CALUDE_vehicle_speed_problem_l3684_368418


namespace NUMINAMATH_CALUDE_lab_budget_calculation_l3684_368428

theorem lab_budget_calculation (flask_cost safety_gear_cost test_tube_cost remaining_budget : ℝ) 
  (h1 : flask_cost = 150)
  (h2 : test_tube_cost = 2/3 * flask_cost)
  (h3 : safety_gear_cost = 1/2 * test_tube_cost)
  (h4 : remaining_budget = 25) :
  flask_cost + test_tube_cost + safety_gear_cost + remaining_budget = 325 := by
sorry

end NUMINAMATH_CALUDE_lab_budget_calculation_l3684_368428


namespace NUMINAMATH_CALUDE_power_of_two_divisibility_l3684_368458

theorem power_of_two_divisibility (n : ℕ+) :
  (∀ (n : ℕ+), ∃ (m : ℤ), (2^n.val - 1) ∣ (m^2 + 9)) ↔
  ∃ (s : ℕ), n.val = 2^s :=
sorry

end NUMINAMATH_CALUDE_power_of_two_divisibility_l3684_368458
