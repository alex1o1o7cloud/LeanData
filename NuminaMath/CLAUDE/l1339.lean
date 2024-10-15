import Mathlib

namespace NUMINAMATH_CALUDE_product_mod_five_l1339_133930

theorem product_mod_five : 2011 * 2012 * 2013 * 2014 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_five_l1339_133930


namespace NUMINAMATH_CALUDE_fraction_simplification_l1339_133946

theorem fraction_simplification : 3 / (2 - 3/4) = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1339_133946


namespace NUMINAMATH_CALUDE_trig_ratio_simplification_l1339_133967

theorem trig_ratio_simplification :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_ratio_simplification_l1339_133967


namespace NUMINAMATH_CALUDE_find_number_l1339_133944

theorem find_number : ∃ x : ℝ, x = 800 ∧ 0.4 * x = 0.2 * 650 + 190 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1339_133944


namespace NUMINAMATH_CALUDE_qinJiushao_v3_value_l1339_133908

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 2*x^5 + 5*x^4 + 8*x^3 + 7*x^2 - 6*x + 11

-- Define Qin Jiushao's algorithm for this specific polynomial
def qinJiushao (x : ℝ) : ℝ × ℝ × ℝ × ℝ := 
  let v1 := 2*x + 5
  let v2 := v1*x + 8
  let v3 := v2*x + 7
  let v4 := v3*x - 6
  (v1, v2, v3, v4)

-- Theorem statement
theorem qinJiushao_v3_value : 
  (qinJiushao 3).2.2 = 130 :=
by sorry

end NUMINAMATH_CALUDE_qinJiushao_v3_value_l1339_133908


namespace NUMINAMATH_CALUDE_circumcircle_diameter_l1339_133998

theorem circumcircle_diameter (a b c : ℝ) (θ : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : Real.cos θ = 1/3) :
  let d := max a (max b c)
  2 * d / Real.sin θ = 9 * Real.sqrt 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_diameter_l1339_133998


namespace NUMINAMATH_CALUDE_square_area_ratio_l1339_133962

theorem square_area_ratio (r : ℝ) (hr : r > 0) :
  let s1 := Real.sqrt ((4 / 5) * r ^ 2)
  let s2 := Real.sqrt (2 * r ^ 2)
  (s1 ^ 2) / (s2 ^ 2) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1339_133962


namespace NUMINAMATH_CALUDE_exists_face_sum_gt_25_l1339_133983

/-- Represents a cube with labeled edges -/
structure LabeledCube where
  edges : Fin 12 → ℕ
  edge_sum : ∀ i : Fin 12, edges i ∈ Finset.range 13 \ {0}

/-- Represents a face of the cube -/
def Face := Fin 4 → Fin 12

/-- The sum of the numbers on the edges of a face -/
def face_sum (c : LabeledCube) (f : Face) : ℕ :=
  (Finset.range 4).sum (λ i => c.edges (f i))

/-- Theorem: There exists a face with sum greater than 25 -/
theorem exists_face_sum_gt_25 (c : LabeledCube) : 
  ∃ f : Face, face_sum c f > 25 := by
  sorry


end NUMINAMATH_CALUDE_exists_face_sum_gt_25_l1339_133983


namespace NUMINAMATH_CALUDE_coin_flip_expected_value_l1339_133940

/-- Calculates the expected value of a coin flip experiment -/
def expected_value (coin_values : List ℚ) (probability : ℚ) : ℚ :=
  (coin_values.sum * probability)

/-- The main theorem: expected value of the coin flip experiment -/
theorem coin_flip_expected_value :
  let coin_values : List ℚ := [1, 5, 10, 50, 100]
  let probability : ℚ := 1/2
  expected_value coin_values probability = 83 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_expected_value_l1339_133940


namespace NUMINAMATH_CALUDE_three_integer_solutions_l1339_133916

theorem three_integer_solutions (n : ℕ) (x₁ y₁ : ℤ) 
  (h : x₁^3 - 3*x₁*y₁^2 + y₁^3 = n) : 
  ∃ (x₂ y₂ x₃ y₃ : ℤ), 
    (x₂^3 - 3*x₂*y₂^2 + y₂^3 = n) ∧ 
    (x₃^3 - 3*x₃*y₃^2 + y₃^3 = n) ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ 
    (x₂ ≠ x₃ ∨ y₂ ≠ y₃) := by
  sorry

end NUMINAMATH_CALUDE_three_integer_solutions_l1339_133916


namespace NUMINAMATH_CALUDE_roy_sports_hours_l1339_133900

/-- Calculates the total hours spent on sports in school for a week with missed days -/
def sports_hours_in_week (daily_hours : ℕ) (school_days : ℕ) (missed_days : ℕ) : ℕ :=
  (school_days - missed_days) * daily_hours

/-- Proves that Roy spent 6 hours on sports in school for the given week -/
theorem roy_sports_hours :
  let daily_hours : ℕ := 2
  let school_days : ℕ := 5
  let missed_days : ℕ := 2
  sports_hours_in_week daily_hours school_days missed_days = 6 := by
  sorry

end NUMINAMATH_CALUDE_roy_sports_hours_l1339_133900


namespace NUMINAMATH_CALUDE_train_length_l1339_133911

/-- The length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 55 * (1000 / 3600) →
  platform_length = 520 →
  crossing_time = 64.79481641468682 →
  (train_speed * crossing_time) - platform_length = 470 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1339_133911


namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l1339_133917

theorem inequality_holds_iff_m_in_range (m : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3*m) ↔ 
  (m ≤ -1 ∨ m ≥ 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l1339_133917


namespace NUMINAMATH_CALUDE_find_number_l1339_133932

theorem find_number (x : ℕ) : 102 * 102 + x * x = 19808 → x = 97 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1339_133932


namespace NUMINAMATH_CALUDE_gcd_cube_plus_27_l1339_133958

theorem gcd_cube_plus_27 (n : ℕ) (h : n > 27) :
  Nat.gcd (n^3 + 3^3) (n + 3) = n + 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_27_l1339_133958


namespace NUMINAMATH_CALUDE_dual_colored_cubes_count_l1339_133970

/-- Represents a cube with colored faces -/
structure ColoredCube where
  size : ℕ
  blue_faces : Fin 3
  red_faces : Fin 3

/-- Counts the number of smaller cubes with both colors when a colored cube is sliced -/
def count_dual_colored_cubes (cube : ColoredCube) : ℕ :=
  sorry

/-- The main theorem stating that a 4x4x4 cube with two opposite blue faces and four red faces
    will have exactly 24 smaller cubes with both colors when sliced into 1x1x1 cubes -/
theorem dual_colored_cubes_count :
  let cube : ColoredCube := ⟨4, 2, 4⟩
  count_dual_colored_cubes cube = 24 := by sorry

end NUMINAMATH_CALUDE_dual_colored_cubes_count_l1339_133970


namespace NUMINAMATH_CALUDE_income_problem_l1339_133929

theorem income_problem (m n o : ℕ) : 
  (m + n) / 2 = 5050 →
  (n + o) / 2 = 6250 →
  (m + o) / 2 = 5200 →
  m = 4000 := by
sorry

end NUMINAMATH_CALUDE_income_problem_l1339_133929


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l1339_133979

/-- The volume of a rectangular prism with given edge lengths and space diagonal --/
theorem rectangular_prism_volume (AB AD AC1 : ℝ) :
  AB = 2 →
  AD = 2 →
  AC1 = 3 →
  ∃ (AA1 : ℝ), AA1 > 0 ∧ AB * AD * AA1 = 4 ∧ AC1^2 = AB^2 + AD^2 + AA1^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l1339_133979


namespace NUMINAMATH_CALUDE_calculate_interest_rate_loan_interest_rate_proof_l1339_133978

/-- Calculates the rate of interest for a loan with simple interest -/
theorem calculate_interest_rate (principal : ℝ) (interest_paid : ℝ) : ℝ :=
  let rate_squared := (100 * interest_paid) / (principal)
  Real.sqrt rate_squared

/-- Proves that the rate of interest for the given loan conditions is approximately 8.888% -/
theorem loan_interest_rate_proof 
  (principal : ℝ) 
  (interest_paid : ℝ) 
  (h1 : principal = 800) 
  (h2 : interest_paid = 632) : 
  ∃ (ε : ℝ), ε > 0 ∧ |calculate_interest_rate principal interest_paid - 8.888| < ε :=
sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_loan_interest_rate_proof_l1339_133978


namespace NUMINAMATH_CALUDE_grocery_problem_l1339_133938

theorem grocery_problem (total_packs cookie_packs : ℕ) 
  (h1 : total_packs = 27)
  (h2 : cookie_packs = 23)
  (h3 : total_packs = cookie_packs + cake_packs) :
  cake_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_grocery_problem_l1339_133938


namespace NUMINAMATH_CALUDE_apple_production_solution_l1339_133947

/-- Represents the apple production of a tree over three years -/
structure AppleProduction where
  first_year : ℕ
  second_year : ℕ := 2 * first_year + 8
  third_year : ℕ := (3 * second_year) / 4

/-- Theorem stating the solution to the apple production problem -/
theorem apple_production_solution :
  ∃ (prod : AppleProduction),
    prod.first_year + prod.second_year + prod.third_year = 194 ∧
    prod.first_year = 40 := by
  sorry

end NUMINAMATH_CALUDE_apple_production_solution_l1339_133947


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angles_l1339_133992

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b) ∨ (b = c) ∨ (a = c)
  sumOfAngles : a + b + c = 180

-- Define the condition of angle ratio
def hasAngleRatio (t : IsoscelesTriangle) : Prop :=
  (t.a = 2 * t.b) ∨ (t.b = 2 * t.c) ∨ (t.a = 2 * t.c)

-- Theorem statement
theorem isosceles_triangle_base_angles 
  (t : IsoscelesTriangle) 
  (h : hasAngleRatio t) : 
  (t.a = 45 ∧ t.b = 45) ∨ (t.b = 72 ∧ t.c = 72) ∨ (t.a = 72 ∧ t.c = 72) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angles_l1339_133992


namespace NUMINAMATH_CALUDE_binary_10010_is_18_l1339_133980

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10010_is_18 :
  binary_to_decimal [false, true, false, false, true] = 18 := by
  sorry

end NUMINAMATH_CALUDE_binary_10010_is_18_l1339_133980


namespace NUMINAMATH_CALUDE_multiple_with_four_digits_l1339_133943

theorem multiple_with_four_digits (k : ℕ) (h : k > 1) :
  ∃ w : ℕ, w > 0 ∧ k ∣ w ∧ w < k^4 ∧ 
  ∃ (a b c d : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    (a = 0 ∨ a = 1 ∨ a = 8 ∨ a = 9) ∧
    (b = 0 ∨ b = 1 ∨ b = 8 ∨ b = 9) ∧
    (c = 0 ∨ c = 1 ∨ c = 8 ∨ c = 9) ∧
    (d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 9) ∧
    w = a * 1000 + b * 100 + c * 10 + d := by
  sorry

end NUMINAMATH_CALUDE_multiple_with_four_digits_l1339_133943


namespace NUMINAMATH_CALUDE_combined_return_percentage_l1339_133904

theorem combined_return_percentage 
  (investment1 : ℝ) 
  (investment2 : ℝ) 
  (return1 : ℝ) 
  (return2 : ℝ) 
  (h1 : investment1 = 500)
  (h2 : investment2 = 1500)
  (h3 : return1 = 0.07)
  (h4 : return2 = 0.09) :
  (investment1 * return1 + investment2 * return2) / (investment1 + investment2) = 0.085 := by
sorry

end NUMINAMATH_CALUDE_combined_return_percentage_l1339_133904


namespace NUMINAMATH_CALUDE_shirt_ratio_l1339_133984

theorem shirt_ratio : 
  ∀ (steven andrew brian : ℕ),
  steven = 4 * andrew →
  brian = 3 →
  steven = 72 →
  andrew / brian = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_shirt_ratio_l1339_133984


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_simplify_sqrt_difference_squares_simplify_sqrt_fraction_product_simplify_sqrt_decimal_l1339_133974

-- Part 1
theorem simplify_sqrt_fraction : (1/2) * Real.sqrt (4/7) = Real.sqrt 7 / 7 := by sorry

-- Part 2
theorem simplify_sqrt_difference_squares : Real.sqrt (20^2 - 15^2) = 5 * Real.sqrt 7 := by sorry

-- Part 3
theorem simplify_sqrt_fraction_product : Real.sqrt ((32 * 9) / 25) = (12 * Real.sqrt 2) / 5 := by sorry

-- Part 4
theorem simplify_sqrt_decimal : Real.sqrt 22.5 = (3 * Real.sqrt 10) / 2 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_simplify_sqrt_difference_squares_simplify_sqrt_fraction_product_simplify_sqrt_decimal_l1339_133974


namespace NUMINAMATH_CALUDE_parabola_vertex_l1339_133921

/-- The vertex coordinates of the parabola y = x^2 - 6x + 1 are (3, -8) -/
theorem parabola_vertex : 
  let f : ℝ → ℝ := λ x => x^2 - 6*x + 1
  ∃ a b : ℝ, a = 3 ∧ b = -8 ∧ ∀ x : ℝ, f x = (x - a)^2 + b :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1339_133921


namespace NUMINAMATH_CALUDE_multiplicative_inverse_of_PQ_l1339_133990

theorem multiplicative_inverse_of_PQ (P Q : ℕ) (M : ℕ) : 
  P = 123321 → 
  Q = 246642 → 
  M = 69788 → 
  (P * Q * M) % 1000003 = 1 := by
sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_of_PQ_l1339_133990


namespace NUMINAMATH_CALUDE_adjacent_chair_subsets_theorem_l1339_133994

/-- The number of subsets containing at least three adjacent chairs in a circular arrangement of 12 chairs -/
def adjacent_chair_subsets : ℕ := 1634

/-- The number of chairs arranged in a circle -/
def num_chairs : ℕ := 12

/-- A function that calculates the number of subsets containing at least three adjacent chairs -/
def calculate_subsets (n : ℕ) : ℕ := sorry

theorem adjacent_chair_subsets_theorem :
  calculate_subsets num_chairs = adjacent_chair_subsets :=
by sorry

end NUMINAMATH_CALUDE_adjacent_chair_subsets_theorem_l1339_133994


namespace NUMINAMATH_CALUDE_square_hexagon_area_l1339_133963

theorem square_hexagon_area (s : ℝ) (square_area : ℝ) (hexagon_area : ℝ) :
  square_area = Real.sqrt 3 →
  square_area = s ^ 2 →
  hexagon_area = 3 * Real.sqrt 3 * s ^ 2 / 2 →
  hexagon_area = 9 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_square_hexagon_area_l1339_133963


namespace NUMINAMATH_CALUDE_negative_x_sqrt_squared_diff_l1339_133909

theorem negative_x_sqrt_squared_diff (x : ℝ) (h : x < 0) : x - Real.sqrt ((x - 1)^2) = 2*x - 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_sqrt_squared_diff_l1339_133909


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l1339_133995

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water (km/h)
  stream : ℝ   -- Speed of the stream (km/h)

/-- Calculates the effective speed when swimming downstream. -/
def downstreamSpeed (s : SwimmerSpeed) : ℝ := s.swimmer + s.stream

/-- Calculates the effective speed when swimming upstream. -/
def upstreamSpeed (s : SwimmerSpeed) : ℝ := s.swimmer - s.stream

/-- Theorem stating that given the conditions of the swimming problem, 
    the swimmer's speed in still water is 7 km/h. -/
theorem swimmer_speed_in_still_water : 
  ∀ s : SwimmerSpeed, 
    downstreamSpeed s = 36 / 6 → 
    upstreamSpeed s = 48 / 6 → 
    s.swimmer = 7 := by
  sorry

#check swimmer_speed_in_still_water

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l1339_133995


namespace NUMINAMATH_CALUDE_no_solution_iff_m_less_than_neg_two_l1339_133993

theorem no_solution_iff_m_less_than_neg_two (m : ℝ) :
  (∀ x : ℝ, ¬(x - m ≤ 2*m + 3 ∧ (x - 1)/2 ≥ m)) ↔ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_less_than_neg_two_l1339_133993


namespace NUMINAMATH_CALUDE_books_remaining_l1339_133975

theorem books_remaining (initial_books yard_sale_books day1_books day2_books day3_books : ℕ) :
  initial_books = 75 →
  yard_sale_books = 33 →
  day1_books = 15 →
  day2_books = 8 →
  day3_books = 12 →
  initial_books - (yard_sale_books + day1_books + day2_books + day3_books) = 7 :=
by sorry

end NUMINAMATH_CALUDE_books_remaining_l1339_133975


namespace NUMINAMATH_CALUDE_smallest_square_cover_l1339_133948

/-- The width of the rectangle -/
def rectangle_width : ℕ := 3

/-- The height of the rectangle -/
def rectangle_height : ℕ := 4

/-- The area of a single rectangle -/
def rectangle_area : ℕ := rectangle_width * rectangle_height

/-- The side length of the smallest square that can be covered exactly by the rectangles -/
def square_side : ℕ := lcm rectangle_width rectangle_height

/-- The area of the square -/
def square_area : ℕ := square_side * square_side

/-- The number of rectangles needed to cover the square -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem smallest_square_cover :
  num_rectangles = 12 ∧
  ∀ n : ℕ, n < square_side → ¬(n * n % rectangle_area = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_cover_l1339_133948


namespace NUMINAMATH_CALUDE_smallest_valid_number_l1339_133977

-- Define the property of being a nonprime integer greater than 1 with no prime factor less than 15
def is_valid_number (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n ∧ ∀ p : ℕ, Nat.Prime p → p < 15 → ¬ p ∣ n

-- State the theorem
theorem smallest_valid_number :
  ∃ n : ℕ, is_valid_number n ∧ ∀ m : ℕ, is_valid_number m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l1339_133977


namespace NUMINAMATH_CALUDE_greeting_card_distribution_four_l1339_133920

def greeting_card_distribution (n : ℕ) : ℕ :=
  if n = 4 then 9 else 0

theorem greeting_card_distribution_four :
  greeting_card_distribution 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_greeting_card_distribution_four_l1339_133920


namespace NUMINAMATH_CALUDE_spider_journey_l1339_133928

theorem spider_journey (r : ℝ) (final_leg : ℝ) : r = 75 ∧ final_leg = 90 →
  2 * r + r + final_leg = 315 := by
  sorry

end NUMINAMATH_CALUDE_spider_journey_l1339_133928


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1339_133959

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x^2 + 2*x < 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {-2, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1339_133959


namespace NUMINAMATH_CALUDE_total_seashells_l1339_133954

theorem total_seashells (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) 
  (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l1339_133954


namespace NUMINAMATH_CALUDE_chewing_gum_price_l1339_133961

def currency_denominations : List Nat := [1, 5, 10, 20, 50, 100]

def is_valid_payment (price : Nat) (payment1 payment2 : Nat) : Prop :=
  payment1 > price ∧ payment2 > price ∧
  ∃ (exchange : Nat), exchange ≤ payment1 ∧ exchange ≤ payment2 ∧
    payment1 - exchange + (payment2 - price) = price ∧
    payment2 - (payment2 - price) + exchange = price

def exists_valid_payments (price : Nat) : Prop :=
  ∃ (payment1 payment2 : Nat),
    payment1 ∈ currency_denominations ∧
    payment2 ∈ currency_denominations ∧
    is_valid_payment price payment1 payment2

theorem chewing_gum_price :
  ¬ exists_valid_payments 2 ∧
  ¬ exists_valid_payments 6 ∧
  ¬ exists_valid_payments 7 ∧
  exists_valid_payments 8 :=
by sorry

end NUMINAMATH_CALUDE_chewing_gum_price_l1339_133961


namespace NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l1339_133913

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^3 + 7*x

-- Theorem statement
theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 582 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l1339_133913


namespace NUMINAMATH_CALUDE_prob_blue_face_four_blue_two_red_l1339_133907

/-- A cube with blue and red faces -/
structure ColoredCube where
  blue_faces : ℕ
  red_faces : ℕ

/-- The probability of rolling a blue face on a colored cube -/
def prob_blue_face (cube : ColoredCube) : ℚ :=
  cube.blue_faces / (cube.blue_faces + cube.red_faces)

/-- Theorem: The probability of rolling a blue face on a cube with 4 blue faces and 2 red faces is 2/3 -/
theorem prob_blue_face_four_blue_two_red :
  prob_blue_face ⟨4, 2⟩ = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_blue_face_four_blue_two_red_l1339_133907


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_l1339_133950

theorem rectangular_prism_surface_area
  (r : ℝ) (l w h : ℝ) 
  (h_r : r = 3 * (36 / Real.pi))
  (h_l : l = 6)
  (h_w : w = 4)
  (h_vol_eq : (4 / 3) * Real.pi * r^3 = l * w * h) :
  2 * (l * w + l * h + w * h) = 88 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_l1339_133950


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l1339_133971

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l1339_133971


namespace NUMINAMATH_CALUDE_three_people_seven_steps_l1339_133969

def staircase_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose k 3 * (Nat.choose n 3 + Nat.choose 3 1 * Nat.choose n 2)

theorem three_people_seven_steps :
  staircase_arrangements 7 7 = 336 := by
  sorry

end NUMINAMATH_CALUDE_three_people_seven_steps_l1339_133969


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l1339_133966

theorem concentric_circles_area_ratio :
  ∀ (r₁ r₂ : ℝ), r₁ > 0 → r₂ > 0 → r₂ = 3 * r₁ →
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l1339_133966


namespace NUMINAMATH_CALUDE_points_per_treasure_l1339_133988

theorem points_per_treasure (treasures_level1 treasures_level2 total_score : ℕ) 
  (h1 : treasures_level1 = 6)
  (h2 : treasures_level2 = 2)
  (h3 : total_score = 32) :
  total_score / (treasures_level1 + treasures_level2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_points_per_treasure_l1339_133988


namespace NUMINAMATH_CALUDE_farm_harvest_after_26_days_l1339_133981

/-- Represents the daily harvest rates for a fruit farm -/
structure HarvestRates where
  ripeOrangesOdd : ℕ
  unripeOrangesOdd : ℕ
  ripeOrangesEven : ℕ
  unripeOrangesEven : ℕ
  ripeApples : ℕ
  unripeApples : ℕ

/-- Calculates the total harvest for a given number of days -/
def totalHarvest (rates : HarvestRates) (days : ℕ) :
  ℕ × ℕ × ℕ × ℕ :=
  let oddDays := (days + 1) / 2
  let evenDays := days / 2
  ( oddDays * rates.ripeOrangesOdd + evenDays * rates.ripeOrangesEven
  , oddDays * rates.unripeOrangesOdd + evenDays * rates.unripeOrangesEven
  , days * rates.ripeApples
  , days * rates.unripeApples
  )

/-- The main theorem stating the total harvest after 26 days -/
theorem farm_harvest_after_26_days (rates : HarvestRates)
  (h1 : rates.ripeOrangesOdd = 32)
  (h2 : rates.unripeOrangesOdd = 46)
  (h3 : rates.ripeOrangesEven = 28)
  (h4 : rates.unripeOrangesEven = 52)
  (h5 : rates.ripeApples = 50)
  (h6 : rates.unripeApples = 30) :
  totalHarvest rates 26 = (780, 1274, 1300, 780) := by
  sorry

end NUMINAMATH_CALUDE_farm_harvest_after_26_days_l1339_133981


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1339_133914

theorem rectangle_perimeter (square_side : ℝ) (rect_length rect_breadth : ℝ) :
  square_side = 8 →
  rect_length = 8 →
  rect_breadth = 4 →
  let new_length := square_side + rect_length
  let new_breadth := square_side
  2 * (new_length + new_breadth) = 48 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1339_133914


namespace NUMINAMATH_CALUDE_exp_13pi_over_3_rectangular_form_l1339_133924

open Complex

theorem exp_13pi_over_3_rectangular_form :
  exp (13 * π * I / 3) = (1 / 2 : ℂ) + (I * (Real.sqrt 3 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_exp_13pi_over_3_rectangular_form_l1339_133924


namespace NUMINAMATH_CALUDE_average_height_combined_l1339_133910

theorem average_height_combined (group1_count group2_count : ℕ) 
  (group1_avg group2_avg : ℝ) (total_count : ℕ) :
  group1_count = 20 →
  group2_count = 11 →
  group1_avg = 20 →
  group2_avg = 20 →
  total_count = group1_count + group2_count →
  (group1_count * group1_avg + group2_count * group2_avg) / total_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_average_height_combined_l1339_133910


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1339_133957

theorem sqrt_equation_solution :
  ∀ x : ℝ, x > 0 → (6 * Real.sqrt (4 + x) + 6 * Real.sqrt (4 - x) = 9 * Real.sqrt 2) → x = Real.sqrt 255 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1339_133957


namespace NUMINAMATH_CALUDE_marks_per_correct_answer_l1339_133955

/-- Proves that the number of marks scored for each correct answer is 4 -/
theorem marks_per_correct_answer 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (total_marks : ℤ) 
  (wrong_answer_penalty : ℤ) 
  (h1 : total_questions = 50)
  (h2 : correct_answers = 36)
  (h3 : total_marks = 130)
  (h4 : wrong_answer_penalty = -1)
  (h5 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (marks_per_correct : ℤ), 
    marks_per_correct * correct_answers + 
    wrong_answer_penalty * (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 :=
by sorry

end NUMINAMATH_CALUDE_marks_per_correct_answer_l1339_133955


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_proof_l1339_133939

/-- The smallest number of eggs given the conditions -/
def smallest_number_of_eggs : ℕ := 137

/-- The number of containers with 9 eggs -/
def containers_with_nine : ℕ := 3

/-- The capacity of a full container -/
def container_capacity : ℕ := 10

theorem smallest_number_of_eggs_proof :
  ∀ n : ℕ,
  n > 130 ∧
  n = container_capacity * (n / container_capacity) - containers_with_nine →
  n ≥ smallest_number_of_eggs :=
by
  sorry

#check smallest_number_of_eggs_proof

end NUMINAMATH_CALUDE_smallest_number_of_eggs_proof_l1339_133939


namespace NUMINAMATH_CALUDE_like_terms_difference_l1339_133915

def are_like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∃ (c d : ℚ) (m n : ℕ), ∀ (x y : ℕ), a x y = c * x^m * y^3 ∧ b x y = d * x^4 * y^n

theorem like_terms_difference (m n : ℕ) :
  are_like_terms (λ x y => -3 * x^m * y^3) (λ x y => 2 * x^4 * y^n) → m - n = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_difference_l1339_133915


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1339_133952

/-- A geometric sequence with first term 3 and the sum of 1st, 3rd, and 5th terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 
    a 1 = 3 ∧ 
    (∀ n : ℕ, a (n + 1) = a n * q) ∧
    a 1 + a 3 + a 5 = 21

/-- The product of the 2nd and 6th terms of the geometric sequence is 72 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 2 * a 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1339_133952


namespace NUMINAMATH_CALUDE_total_candles_in_small_boxes_l1339_133999

theorem total_candles_in_small_boxes 
  (small_boxes_per_big_box : ℕ) 
  (num_big_boxes : ℕ) 
  (candles_per_small_box : ℕ) : 
  small_boxes_per_big_box = 4 → 
  num_big_boxes = 50 → 
  candles_per_small_box = 40 → 
  small_boxes_per_big_box * num_big_boxes * candles_per_small_box = 8000 :=
by sorry

end NUMINAMATH_CALUDE_total_candles_in_small_boxes_l1339_133999


namespace NUMINAMATH_CALUDE_r_value_when_m_is_3_l1339_133926

theorem r_value_when_m_is_3 :
  let m : ℕ := 3
  let t : ℕ := 3^m + 2
  let r : ℕ := 5^t - 2*t
  r = 5^29 - 58 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_m_is_3_l1339_133926


namespace NUMINAMATH_CALUDE_platform_length_l1339_133906

/-- Given a train of length 300 meters that takes 42 seconds to cross a platform
    and 18 seconds to cross a signal pole, prove that the length of the platform
    is approximately 400.14 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 42)
  (h3 : pole_crossing_time = 18) :
  ∃ (platform_length : ℝ), abs (platform_length - 400.14) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l1339_133906


namespace NUMINAMATH_CALUDE_total_balls_l1339_133923

/-- Given the number of basketballs, volleyballs, and soccer balls in a school,
    prove that the total number of balls is 94. -/
theorem total_balls (b v s : ℕ) : 
  b = 32 →
  b = v + 5 →
  b = s - 3 →
  b + v + s = 94 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_l1339_133923


namespace NUMINAMATH_CALUDE_marco_marie_age_difference_l1339_133912

theorem marco_marie_age_difference (marie_age : ℕ) (total_age : ℕ) : 
  marie_age = 12 → 
  total_age = 37 → 
  ∃ (marco_age : ℕ), marco_age + marie_age = total_age ∧ marco_age = 2 * marie_age + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_marco_marie_age_difference_l1339_133912


namespace NUMINAMATH_CALUDE_unique_A_value_l1339_133976

/-- A function that checks if a number is a four-digit number -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- Theorem stating that 2 is the only single-digit value for A that satisfies the equation -/
theorem unique_A_value : 
  ∃! (A : ℕ), isSingleDigit A ∧ 
    (∃ (B C D : ℕ), isSingleDigit B ∧ isSingleDigit C ∧ isSingleDigit D ∧
      isFourDigit (A * 1000 + 2 * 100 + B * 10 + 2) ∧
      isFourDigit (1000 + C * 100 + 10 + D) ∧
      (A * 1000 + 2 * 100 + B * 10 + 2) + (1000 + C * 100 + 10 + D) = 3333) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_A_value_l1339_133976


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l1339_133925

theorem angle_sum_around_point (y : ℝ) : 
  3 * y + 6 * y + 2 * y + 4 * y + y = 360 → y = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l1339_133925


namespace NUMINAMATH_CALUDE_christinas_age_problem_l1339_133934

theorem christinas_age_problem (C : ℝ) (Y : ℝ) :
  (C + 5 = Y / 2) →
  (21 = (3 / 5) * C) →
  Y = 80 := by
sorry

end NUMINAMATH_CALUDE_christinas_age_problem_l1339_133934


namespace NUMINAMATH_CALUDE_amy_picture_files_l1339_133973

theorem amy_picture_files (music_files : ℝ) (video_files : ℝ) (total_files : ℕ) : 
  music_files = 4.0 →
  video_files = 21.0 →
  total_files = 48 →
  (total_files : ℝ) - (music_files + video_files) = 23 := by
sorry

end NUMINAMATH_CALUDE_amy_picture_files_l1339_133973


namespace NUMINAMATH_CALUDE_root_sum_relation_l1339_133987

/-- The polynomial x^3 - 4x^2 + 7x - 10 -/
def p (x : ℝ) : ℝ := x^3 - 4*x^2 + 7*x - 10

/-- The sum of the k-th powers of the roots of p -/
def t (k : ℕ) : ℝ := sorry

theorem root_sum_relation :
  ∃ (u v w : ℝ), p u = 0 ∧ p v = 0 ∧ p w = 0 ∧
  (∀ k, t k = u^k + v^k + w^k) ∧
  t 0 = 3 ∧ t 1 = 4 ∧ t 2 = 10 ∧
  (∃ (d e f : ℝ), ∀ k ≥ 2, t (k+1) = d * t k + e * t (k-1) + f * t (k-2)) →
  ∃ (d e f : ℝ), (∀ k ≥ 2, t (k+1) = d * t k + e * t (k-1) + f * t (k-2)) ∧ d + e + f = 3 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_relation_l1339_133987


namespace NUMINAMATH_CALUDE_competition_results_l1339_133905

/-- Represents the categories of safety questions -/
inductive Category
  | TrafficSafety
  | FireSafety
  | WaterSafety

/-- Represents the scoring system for the competition -/
structure ScoringSystem where
  correct_points : ℕ
  incorrect_points : ℕ

/-- Represents the correct rates for each category -/
def correct_rates : Category → ℚ
  | Category.TrafficSafety => 2/3
  | Category.FireSafety => 1/2
  | Category.WaterSafety => 1/3

/-- The scoring system used in the competition -/
def competition_scoring : ScoringSystem :=
  { correct_points := 5, incorrect_points := 1 }

/-- Calculates the probability of scoring at least 6 points for two questions -/
def prob_at_least_6_points (s : ScoringSystem) : ℚ :=
  let p_traffic := correct_rates Category.TrafficSafety
  let p_fire := correct_rates Category.FireSafety
  p_traffic * p_fire + p_traffic * (1 - p_fire) + (1 - p_traffic) * p_fire

/-- Calculates the expected value of the total score for three questions from different categories -/
def expected_score_three_questions (s : ScoringSystem) : ℚ :=
  let p_traffic := correct_rates Category.TrafficSafety
  let p_fire := correct_rates Category.FireSafety
  let p_water := correct_rates Category.WaterSafety
  let p_all_correct := p_traffic * p_fire * p_water
  let p_two_correct := p_traffic * p_fire * (1 - p_water) +
                       p_traffic * (1 - p_fire) * p_water +
                       (1 - p_traffic) * p_fire * p_water
  let p_one_correct := p_traffic * (1 - p_fire) * (1 - p_water) +
                       (1 - p_traffic) * p_fire * (1 - p_water) +
                       (1 - p_traffic) * (1 - p_fire) * p_water
  let p_all_incorrect := (1 - p_traffic) * (1 - p_fire) * (1 - p_water)
  3 * s.correct_points * p_all_correct +
  (2 * s.correct_points + s.incorrect_points) * p_two_correct +
  (s.correct_points + 2 * s.incorrect_points) * p_one_correct +
  3 * s.incorrect_points * p_all_incorrect

theorem competition_results :
  prob_at_least_6_points competition_scoring = 5/6 ∧
  expected_score_three_questions competition_scoring = 9 := by
  sorry

end NUMINAMATH_CALUDE_competition_results_l1339_133905


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l1339_133972

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 11*x + 15

-- Define the partial fraction decomposition
def pfd (x A B C : ℝ) : Prop :=
  1 / p x = A / (x - 5) + B / (x + 3) + C / ((x + 3)^2)

-- State the theorem
theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x, pfd x A B C) → (∀ x, p x = (x - 5) * (x + 3)^2) → A = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l1339_133972


namespace NUMINAMATH_CALUDE_even_function_symmetry_l1339_133949

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def has_min_value_on (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

theorem even_function_symmetry (f : ℝ → ℝ) :
  is_even_function f →
  is_increasing_on f 3 7 →
  has_min_value_on f 3 7 2 →
  is_decreasing_on f (-7) (-3) ∧ has_min_value_on f (-7) (-3) 2 :=
sorry

end NUMINAMATH_CALUDE_even_function_symmetry_l1339_133949


namespace NUMINAMATH_CALUDE_root_equation_problem_l1339_133953

theorem root_equation_problem (c d n r s : ℝ) : 
  (c^2 - n*c + 3 = 0) → 
  (d^2 - n*d + 3 = 0) → 
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) → 
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) → 
  s = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l1339_133953


namespace NUMINAMATH_CALUDE_election_result_theorem_l1339_133986

/-- Represents the result of an election with five candidates -/
structure ElectionResult where
  total_votes : ℕ
  candidate1_votes : ℕ
  candidate2_votes : ℕ
  candidate3_votes : ℕ
  candidate4_votes : ℕ
  candidate5_votes : ℕ

/-- Theorem stating the election result given the conditions -/
theorem election_result_theorem (er : ElectionResult) : 
  er.candidate1_votes = (30 * er.total_votes) / 100 ∧
  er.candidate2_votes = (20 * er.total_votes) / 100 ∧
  er.candidate3_votes = 3000 ∧
  er.candidate3_votes = (15 * er.total_votes) / 100 ∧
  er.candidate4_votes = (25 * er.total_votes) / 100 ∧
  er.candidate5_votes = 2 * er.candidate3_votes →
  er.total_votes = 20000 ∧
  er.candidate1_votes = 6000 ∧
  er.candidate2_votes = 4000 ∧
  er.candidate3_votes = 3000 ∧
  er.candidate4_votes = 5000 ∧
  er.candidate5_votes = 6000 :=
by
  sorry

end NUMINAMATH_CALUDE_election_result_theorem_l1339_133986


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l1339_133936

/-- A game on a circle where two players mark points -/
structure CircleGame where
  /-- The number of points each player marks -/
  p : ℕ
  /-- Condition that p is greater than 1 -/
  p_gt_one : p > 1

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins
  | Draw

/-- A strategy for playing the game -/
def Strategy := CircleGame → GameResult

/-- The theorem stating that the second player has a winning strategy -/
theorem second_player_winning_strategy (game : CircleGame) : 
  ∃ (s : Strategy), ∀ (opponent_strategy : Strategy), 
    s game = GameResult.SecondPlayerWins :=
sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l1339_133936


namespace NUMINAMATH_CALUDE_percentage_against_proposal_l1339_133922

def total_votes : ℕ := 290
def vote_difference : ℕ := 58

theorem percentage_against_proposal :
  let votes_against := (total_votes - vote_difference) / 2
  (votes_against : ℚ) / total_votes * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_against_proposal_l1339_133922


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l1339_133989

/-- Given two 2D vectors a and b, prove that the magnitude of their difference is 5. -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l1339_133989


namespace NUMINAMATH_CALUDE_cone_hemisphere_intersection_volume_l1339_133956

/-- The volume of the common part of a right circular cone and an inscribed hemisphere -/
theorem cone_hemisphere_intersection_volume 
  (m r : ℝ) 
  (h : m > r) 
  (h_pos : r > 0) : 
  ∃ V : ℝ, V = (2 * Real.pi * r^3 / 3) * (1 - (2 * m * r^3) / (m^2 + r^2)^2) := by
  sorry

end NUMINAMATH_CALUDE_cone_hemisphere_intersection_volume_l1339_133956


namespace NUMINAMATH_CALUDE_f_triple_composition_equals_self_l1339_133931

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 3 else n / 2

theorem f_triple_composition_equals_self (k : ℤ) :
  k % 2 = 1 → (f (f (f k)) = k ↔ k = 1) := by
  sorry

end NUMINAMATH_CALUDE_f_triple_composition_equals_self_l1339_133931


namespace NUMINAMATH_CALUDE_election_votes_total_l1339_133918

theorem election_votes_total (winner_percentage : ℚ) (vote_majority : ℕ) : 
  winner_percentage = 7/10 →
  vote_majority = 280 →
  ∃ (total_votes : ℕ), 
    (winner_percentage * total_votes : ℚ) - ((1 - winner_percentage) * total_votes : ℚ) = vote_majority ∧
    total_votes = 700 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_total_l1339_133918


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_of_80_factorial_l1339_133996

theorem last_two_nonzero_digits_of_80_factorial (n : ℕ) : n = 80 → 
  ∃ k : ℕ, n.factorial = 100 * k + 12 ∧ k % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_of_80_factorial_l1339_133996


namespace NUMINAMATH_CALUDE_binomial_and_permutation_60_3_l1339_133942

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the permutation
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Theorem statement
theorem binomial_and_permutation_60_3 :
  binomial 60 3 = 34220 ∧ permutation 60 3 = 205320 :=
by sorry

end NUMINAMATH_CALUDE_binomial_and_permutation_60_3_l1339_133942


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1339_133951

/-- Given that i is the imaginary unit and z is a complex number defined as
    z = ((1+i)^2 + 3(1-i)) / (2+i), prove that if z^2 + az + b = 1 + i
    where a and b are real numbers, then a = -3 and b = 4. -/
theorem complex_equation_solution (i : ℂ) (a b : ℝ) :
  i^2 = -1 →
  let z : ℂ := ((1 + i)^2 + 3*(1 - i)) / (2 + i)
  z^2 + a*z + b = 1 + i →
  a = -3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1339_133951


namespace NUMINAMATH_CALUDE_existence_of_x0_l1339_133919

theorem existence_of_x0 (a b : ℝ) : ∃ x0 : ℝ, x0 ∈ Set.Icc 1 9 ∧ |a * x0 + b + 9 / x0| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x0_l1339_133919


namespace NUMINAMATH_CALUDE_range_of_m_l1339_133933

-- Define set A
def A : Set ℝ := {x | x^2 + 3*x - 10 ≤ 0}

-- Define set B (parametrized by m)
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem statement
theorem range_of_m (m : ℝ) : B m ⊆ A → m < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1339_133933


namespace NUMINAMATH_CALUDE_restaurant_ratio_change_l1339_133903

theorem restaurant_ratio_change (initial_cooks : ℕ) (initial_waiters : ℕ) 
  (hired_waiters : ℕ) :
  initial_cooks = 9 →
  initial_cooks * 11 = initial_waiters * 3 →
  hired_waiters = 12 →
  initial_cooks * 5 = (initial_waiters + hired_waiters) * 1 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_ratio_change_l1339_133903


namespace NUMINAMATH_CALUDE_no_eight_consecutive_odd_exponent_primes_l1339_133985

theorem no_eight_consecutive_odd_exponent_primes :
  ∀ n : ℕ, ∃ k : ℕ, k ∈ Finset.range 8 ∧
  ∃ p : ℕ, Prime p ∧ ∃ m : ℕ, m > 0 ∧ 2 ∣ m ∧ p ^ m ∣ (n + k) := by
  sorry

end NUMINAMATH_CALUDE_no_eight_consecutive_odd_exponent_primes_l1339_133985


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l1339_133965

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n > 0 → a (n + 1) / a n = r

theorem geometric_sequence_ninth_term (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, n > 0 → a (n + 1) / a n = 2^n) →
  a 1 = 1 →
  a 9 = 2^36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l1339_133965


namespace NUMINAMATH_CALUDE_odd_function_extension_l1339_133927

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the properties of the function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_extension {f : ℝ → ℝ} 
  (h_odd : is_odd_function f)
  (h_pos : ∀ x > 0, f x = lg (x + 1)) :
  ∀ x < 0, f x = -lg (-x + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1339_133927


namespace NUMINAMATH_CALUDE_q_components_l1339_133941

/-- The rank of a rational number -/
def rank (q : ℚ) : ℕ :=
  sorry

/-- The largest rational number less than 1/4 with rank 3 -/
def q : ℚ :=
  sorry

/-- The components of q when expressed as a sum of three unit fractions -/
def a₁ : ℕ := sorry
def a₂ : ℕ := sorry
def a₃ : ℕ := sorry

/-- q is less than 1/4 -/
axiom q_lt_quarter : q < 1/4

/-- q has rank 3 -/
axiom q_rank : rank q = 3

/-- q is the largest such number -/
axiom q_largest (r : ℚ) : r < 1/4 → rank r = 3 → r ≤ q

/-- q is expressed as the sum of three unit fractions -/
axiom q_sum : q = 1/a₁ + 1/a₂ + 1/a₃

/-- Each aᵢ is the smallest positive integer satisfying the condition -/
axiom a₁_minimal : ∀ n : ℕ, n > 0 → q ≥ 1/n → n ≥ a₁
axiom a₂_minimal : ∀ n : ℕ, n > 0 → q ≥ 1/a₁ + 1/n → n ≥ a₂
axiom a₃_minimal : ∀ n : ℕ, n > 0 → q ≥ 1/a₁ + 1/a₂ + 1/n → n ≥ a₃

theorem q_components : a₁ = 5 ∧ a₂ = 21 ∧ a₃ = 421 :=
  sorry

end NUMINAMATH_CALUDE_q_components_l1339_133941


namespace NUMINAMATH_CALUDE_readers_overlap_l1339_133945

theorem readers_overlap (total : ℕ) (sci_fi : ℕ) (literary : ℕ) (h1 : total = 250) (h2 : sci_fi = 180) (h3 : literary = 88) :
  sci_fi + literary - total = 18 := by
  sorry

end NUMINAMATH_CALUDE_readers_overlap_l1339_133945


namespace NUMINAMATH_CALUDE_subset_condition_nonempty_intersection_l1339_133960

-- Define the sets A and B
def A : Set ℝ := {x | -x^2 + 12*x - 20 > 0}
def B (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Theorem for part (1)
theorem subset_condition (a : ℝ) : B a ⊆ A → a ∈ Set.Iic 3 := by sorry

-- Theorem for part (2)
theorem nonempty_intersection (a : ℝ) : (A ∩ B a).Nonempty → a ∈ Set.Ioi (5/2) := by sorry

end NUMINAMATH_CALUDE_subset_condition_nonempty_intersection_l1339_133960


namespace NUMINAMATH_CALUDE_expressions_same_type_l1339_133964

/-- Two expressions are of the same type if they have the same variables with the same exponents -/
def same_type (e1 e2 : ℕ → ℕ → ℕ → ℚ) : Prop :=
  ∀ a b c : ℕ, ∃ k1 k2 : ℚ, e1 a b c = k1 * a * b^3 * c ∧ e2 a b c = k2 * a * b^3 * c

/-- The original expression -/
def original (a b c : ℕ) : ℚ := -↑a * ↑b^3 * ↑c

/-- The expression to compare -/
def to_compare (a b c : ℕ) : ℚ := (1/3) * ↑a * ↑c * ↑b^3

/-- Theorem stating that the two expressions are of the same type -/
theorem expressions_same_type : same_type original to_compare := by
  sorry

end NUMINAMATH_CALUDE_expressions_same_type_l1339_133964


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1339_133901

/-- Given a triangle ABC with the following properties:
  1. sin C + sin(B-A) = 3 sin(2A)
  2. c = 2
  3. ∠C = π/3
  Prove that the area of triangle ABC is either 2√3/3 or 3√3/7 -/
theorem triangle_area_theorem (A B C : ℝ) (h1 : Real.sin C + Real.sin (B - A) = 3 * Real.sin (2 * A))
    (h2 : 2 = 2) (h3 : C = π / 3) :
  let S := Real.sqrt 3 / 3 * 2
  let S' := Real.sqrt 3 * 3 / 7
  let area := (Real.sin C) * 2 / 2
  area = S ∨ area = S' := by
sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1339_133901


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l1339_133902

/-- A rectangular parallelepiped with face diagonals √3, √5, and 2 has volume √6 -/
theorem parallelepiped_volume (a b c : ℝ) 
  (h1 : a^2 + b^2 = 3)
  (h2 : a^2 + c^2 = 5)
  (h3 : b^2 + c^2 = 4) :
  a * b * c = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l1339_133902


namespace NUMINAMATH_CALUDE_perimeter_diagonal_ratio_bounds_l1339_133997

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Add appropriate convexity condition

/-- The perimeter of a convex quadrilateral -/
def perimeter (q : ConvexQuadrilateral) : ℝ := sorry

/-- The sum of diagonal lengths of a convex quadrilateral -/
def diagonalSum (q : ConvexQuadrilateral) : ℝ := sorry

/-- Theorem: The ratio of perimeter to diagonal sum is strictly between 1 and 2 -/
theorem perimeter_diagonal_ratio_bounds (q : ConvexQuadrilateral) :
  1 < perimeter q / diagonalSum q ∧ perimeter q / diagonalSum q < 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_diagonal_ratio_bounds_l1339_133997


namespace NUMINAMATH_CALUDE_milk_carton_volume_l1339_133991

/-- The volume of a rectangular prism with given dimensions -/
def rectangular_prism_volume (width length height : ℝ) : ℝ :=
  width * length * height

/-- Theorem: The volume of a milk carton with given dimensions is 252 cubic centimeters -/
theorem milk_carton_volume :
  rectangular_prism_volume 9 4 7 = 252 := by
  sorry

end NUMINAMATH_CALUDE_milk_carton_volume_l1339_133991


namespace NUMINAMATH_CALUDE_roots_irrational_l1339_133937

theorem roots_irrational (p q : ℤ) (hp : Odd p) (hq : Odd q) 
  (h_real_roots : ∃ x y : ℝ, x ≠ y ∧ x^2 + 2*p*x + 2*q = 0 ∧ y^2 + 2*p*y + 2*q = 0) :
  ∀ z : ℝ, z^2 + 2*p*z + 2*q = 0 → Irrational z :=
sorry

end NUMINAMATH_CALUDE_roots_irrational_l1339_133937


namespace NUMINAMATH_CALUDE_simplify_expression_l1339_133935

theorem simplify_expression (m : ℝ) (h1 : m ≠ -1) (h2 : m ≠ -2) :
  ((4 * m + 5) / (m + 1) + m - 1) / ((m + 2) / (m + 1)) = m + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1339_133935


namespace NUMINAMATH_CALUDE_vector_relationships_l1339_133968

/-- Given vector a and unit vector b, prove their parallel and perpendicular relationships -/
theorem vector_relationships (a b : ℝ × ℝ) : 
  a = (3, 4) → 
  (b.1^2 + b.2^2 = 1) → 
  ((∃ k : ℝ, b = k • a) → (b = (3/5, 4/5) ∨ b = (-3/5, -4/5))) ∧ 
  ((a.1 * b.1 + a.2 * b.2 = 0) → (b = (-4/5, 3/5) ∨ b = (4/5, -3/5))) := by
  sorry

end NUMINAMATH_CALUDE_vector_relationships_l1339_133968


namespace NUMINAMATH_CALUDE_probability_after_removal_l1339_133982

/-- Represents a deck of cards -/
structure Deck :=
  (cards : Finset ℕ)
  (card_counts : ℕ → ℕ)
  (total_cards : ℕ)

/-- Initial deck configuration -/
def initial_deck : Deck :=
  { cards := Finset.range 13,
    card_counts := λ _ => 4,
    total_cards := 52 }

/-- Deck after removing two pairs -/
def deck_after_removal (d : Deck) : Deck :=
  { cards := d.cards,
    card_counts := λ n => if d.card_counts n ≥ 2 then d.card_counts n - 2 else d.card_counts n,
    total_cards := d.total_cards - 4 }

/-- Number of ways to choose 2 cards from n cards -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Probability of selecting a pair from the remaining deck -/
def pair_probability (d : Deck) : ℚ :=
  let total_choices := choose_two d.total_cards
  let pair_choices := d.cards.sum (λ n => choose_two (d.card_counts n))
  pair_choices / total_choices

/-- Main theorem -/
theorem probability_after_removal :
  pair_probability (deck_after_removal initial_deck) = 17 / 282 := by
  sorry

end NUMINAMATH_CALUDE_probability_after_removal_l1339_133982
