import Mathlib

namespace NUMINAMATH_CALUDE_trig_expression_simplification_l173_17356

theorem trig_expression_simplification :
  (Real.sin (15 * π / 180) + Real.sin (30 * π / 180) + Real.sin (45 * π / 180) + 
   Real.sin (60 * π / 180) + Real.sin (75 * π / 180)) / 
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) =
  (Real.sqrt 2 * (4 * Real.cos (22.5 * π / 180) * Real.cos (7.5 * π / 180) + 1)) / 
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l173_17356


namespace NUMINAMATH_CALUDE_min_even_integers_l173_17397

theorem min_even_integers (a b c d e f g : ℤ) : 
  a + b + c = 30 →
  a + b + c + d + e = 48 →
  a + b + c + d + e + f + g = 60 →
  ∃ (a' b' c' d' e' f' g' : ℤ), 
    a' + b' + c' = 30 ∧
    a' + b' + c' + d' + e' = 48 ∧
    a' + b' + c' + d' + e' + f' + g' = 60 ∧
    Even a' ∧ Even b' ∧ Even c' ∧ Even d' ∧ Even e' ∧ Even f' ∧ Even g' :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l173_17397


namespace NUMINAMATH_CALUDE_partnership_investment_l173_17351

/-- Represents a partnership investment. -/
structure Partnership where
  a_investment : ℚ
  b_investment : ℚ
  c_investment : ℚ
  b_profit : ℚ
  a_profit : ℚ

/-- Theorem stating the relationship between investments and profits in a partnership. -/
theorem partnership_investment (p : Partnership) 
  (hb : p.b_investment = 11000)
  (hc : p.c_investment = 18000)
  (hbp : p.b_profit = 880)
  (hap : p.a_profit = 560) :
  p.a_investment = 7700 := by
  sorry


end NUMINAMATH_CALUDE_partnership_investment_l173_17351


namespace NUMINAMATH_CALUDE_ru_length_l173_17398

/-- Triangle PQR with given side lengths -/
structure Triangle (P Q R : ℝ × ℝ) where
  pq_length : dist P Q = 13
  qr_length : dist Q R = 30
  rp_length : dist R P = 26

/-- Point S on PR such that QS bisects angle PQR -/
def S (P Q R : ℝ × ℝ) (tri : Triangle P Q R) : ℝ × ℝ :=
  sorry

/-- Point T on the circumcircle of PQR, different from Q, such that QT bisects angle PQR -/
def T (P Q R : ℝ × ℝ) (tri : Triangle P Q R) : ℝ × ℝ :=
  sorry

/-- Point U on PQ, different from P, such that U is on the circumcircle of PTS -/
def U (P Q R : ℝ × ℝ) (tri : Triangle P Q R) : ℝ × ℝ :=
  sorry

/-- The main theorem stating that RU = 34 -/
theorem ru_length (P Q R : ℝ × ℝ) (tri : Triangle P Q R) :
  dist R (U P Q R tri) = 34 :=
sorry

end NUMINAMATH_CALUDE_ru_length_l173_17398


namespace NUMINAMATH_CALUDE_intersection_M_N_l173_17367

def M : Set ℝ := {x | x / (x - 1) ≥ 0}

def N : Set ℝ := {y | ∃ x, y = 3 * x^2 + 1}

theorem intersection_M_N : M ∩ N = {x | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l173_17367


namespace NUMINAMATH_CALUDE_x_minus_reciprocal_equals_one_l173_17374

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- Define the fractional part function
noncomputable def fracPart (x : ℝ) : ℝ :=
  x - intPart x

-- Main theorem
theorem x_minus_reciprocal_equals_one (x : ℝ) 
  (h1 : x > 0)
  (h2 : (intPart x : ℝ)^2 = x * fracPart x) : 
  x - 1/x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_reciprocal_equals_one_l173_17374


namespace NUMINAMATH_CALUDE_terrell_lifting_equivalence_l173_17388

/-- The number of times Terrell lifts the weights in the initial setup -/
def initial_lifts : ℕ := 10

/-- The weight of each dumbbell in the initial setup (in pounds) -/
def initial_weight : ℕ := 25

/-- The number of dumbbells used in the initial setup -/
def initial_dumbbells : ℕ := 2

/-- The weight of the single dumbbell in the new setup (in pounds) -/
def new_weight : ℕ := 20

/-- The total weight lifted in the initial setup (in pounds) -/
def total_weight : ℕ := initial_dumbbells * initial_weight * initial_lifts

/-- The number of times Terrell must lift the new weight to achieve the same total weight -/
def required_lifts : ℕ := total_weight / new_weight

theorem terrell_lifting_equivalence :
  required_lifts = 25 := by sorry

end NUMINAMATH_CALUDE_terrell_lifting_equivalence_l173_17388


namespace NUMINAMATH_CALUDE_two_hundred_fiftieth_letter_l173_17384

def repeating_pattern : ℕ → Char
  | n => match n % 3 with
         | 0 => 'C'
         | 1 => 'A'
         | _ => 'B'

theorem two_hundred_fiftieth_letter : repeating_pattern 250 = 'A' := by
  sorry

end NUMINAMATH_CALUDE_two_hundred_fiftieth_letter_l173_17384


namespace NUMINAMATH_CALUDE_min_value_of_f_l173_17380

/-- Given a function f(x) = (a + x^2) / x, where a > 0 and x ∈ (0, b),
    prove that the minimum value of f(x) is 2√a when b > √a. -/
theorem min_value_of_f (a b : ℝ) (ha : a > 0) (hb : b > Real.sqrt a) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt a ∧
    ∀ x ∈ Set.Ioo 0 b, (a + x^2) / x ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l173_17380


namespace NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l173_17306

/-- Proves the equivalence between a polar equation and its Cartesian form --/
theorem polar_to_cartesian_equivalence (x y ρ θ : ℝ) :
  (ρ = -4 * Real.cos θ + Real.sin θ) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) →
  x^2 + y^2 + 4*x - y = 0 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l173_17306


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l173_17346

/-- Acme's cost function -/
def acme_cost (x : ℕ) : ℕ := 75 + 10 * x

/-- Beta's cost function -/
def beta_cost (x : ℕ) : ℕ :=
  if x < 30 then 15 * x else 14 * x

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_for_acme : ℕ := 20

theorem acme_cheaper_at_min_shirts :
  (∀ x < min_shirts_for_acme, beta_cost x ≤ acme_cost x) ∧
  (beta_cost min_shirts_for_acme > acme_cost min_shirts_for_acme) := by
  sorry

#check acme_cheaper_at_min_shirts

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l173_17346


namespace NUMINAMATH_CALUDE_quadratic_value_l173_17307

/-- A quadratic function with specific properties -/
def f (a b c : ℚ) (x : ℚ) : ℚ := a * x^2 + b * x + c

/-- Theorem stating the properties of the quadratic function and its value at x = 5 -/
theorem quadratic_value (a b c : ℚ) :
  (f a b c (-2) = 10) →  -- Maximum value is 10 at x = -2
  ((2 * a * (-2) + b) = 0) →  -- Derivative is 0 at x = -2 (maximum condition)
  (f a b c 0 = -8) →  -- Passes through (0, -8)
  (f a b c 1 = 0) →  -- Passes through (1, 0)
  (f a b c 5 = -400/9) :=  -- Value at x = 5
by sorry

end NUMINAMATH_CALUDE_quadratic_value_l173_17307


namespace NUMINAMATH_CALUDE_large_number_arithmetic_l173_17359

/-- The result of a series of arithmetic operations on large numbers. -/
theorem large_number_arithmetic :
  let start : ℕ := 1500000000000
  let subtract : ℕ := 877888888888
  let add : ℕ := 123456789012
  (start - subtract + add : ℕ) = 745567900124 := by
  sorry

end NUMINAMATH_CALUDE_large_number_arithmetic_l173_17359


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l173_17333

def a : ℕ := 10^2023 - 1

def b : ℕ := 7 * (10^2023 - 1) / 9

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_9ab : sum_of_digits (9 * a * b) = 36410 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l173_17333


namespace NUMINAMATH_CALUDE_potato_distribution_l173_17338

theorem potato_distribution (num_people : ℕ) (bag_weight : ℝ) (bag_cost : ℝ) (total_cost : ℝ) :
  num_people = 40 →
  bag_weight = 20 →
  bag_cost = 5 →
  total_cost = 15 →
  (total_cost / bag_cost * bag_weight) / num_people = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_potato_distribution_l173_17338


namespace NUMINAMATH_CALUDE_fraction_simplification_l173_17350

theorem fraction_simplification : (5 * 8) / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l173_17350


namespace NUMINAMATH_CALUDE_merchant_salt_price_l173_17323

/-- Represents the price per pound of the unknown salt in cents -/
def unknown_price : ℝ := 50

/-- The weight of the unknown salt in pounds -/
def unknown_weight : ℝ := 20

/-- The weight of the known salt in pounds -/
def known_weight : ℝ := 40

/-- The price per pound of the known salt in cents -/
def known_price : ℝ := 35

/-- The selling price per pound of the mixture in cents -/
def selling_price : ℝ := 48

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.2

theorem merchant_salt_price :
  unknown_price = 50 ∧
  (unknown_price * unknown_weight + known_price * known_weight) * (1 + profit_percentage) =
    selling_price * (unknown_weight + known_weight) :=
by sorry

end NUMINAMATH_CALUDE_merchant_salt_price_l173_17323


namespace NUMINAMATH_CALUDE_overlapping_triangles_area_l173_17373

/-- The area common to two overlapping right triangles -/
theorem overlapping_triangles_area :
  let triangle1_hypotenuse : ℝ := 10
  let triangle1_angle1 : ℝ := 30 * π / 180
  let triangle1_angle2 : ℝ := 60 * π / 180
  let triangle2_hypotenuse : ℝ := 15
  let triangle2_angle1 : ℝ := 45 * π / 180
  let triangle2_angle2 : ℝ := 45 * π / 180
  let overlap_length : ℝ := 5
  ∃ (common_area : ℝ), common_area = (25 * Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_triangles_area_l173_17373


namespace NUMINAMATH_CALUDE_course_selection_theorem_l173_17330

def total_course_selection_plans (n : ℕ) (k₁ k₂ : ℕ) : ℕ :=
  (n.choose k₁) * (n.choose k₂) * (n.choose k₂)

theorem course_selection_theorem :
  total_course_selection_plans 4 2 3 = 96 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l173_17330


namespace NUMINAMATH_CALUDE_food_values_l173_17316

def A : Set ℝ := {-1, 1/2, 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 = 1 ∧ a ≥ 0}

def full_food (X Y : Set ℝ) : Prop := X ⊆ Y ∨ Y ⊆ X

def partial_food (X Y : Set ℝ) : Prop :=
  (∃ x, x ∈ X ∩ Y) ∧ ¬(X ⊆ Y) ∧ ¬(Y ⊆ X)

theorem food_values :
  ∀ a : ℝ, (full_food A (B a) ∨ partial_food A (B a)) ↔ (a = 0 ∨ a = 1 ∨ a = 4) :=
sorry

end NUMINAMATH_CALUDE_food_values_l173_17316


namespace NUMINAMATH_CALUDE_servant_served_nine_months_l173_17345

/-- Represents the compensation and service time of a servant --/
structure ServantCompensation where
  fullYearSalary : ℕ  -- Salary for a full year in Rupees
  uniformPrice : ℕ    -- Price of the uniform in Rupees
  receivedSalary : ℕ  -- Salary actually received in Rupees
  monthsServed : ℕ    -- Number of months served

/-- Calculates the total compensation for a full year --/
def fullYearCompensation (s : ServantCompensation) : ℕ :=
  s.fullYearSalary + s.uniformPrice

/-- Calculates the total compensation received --/
def totalReceived (s : ServantCompensation) : ℕ :=
  s.receivedSalary + s.uniformPrice

/-- Theorem stating that under given conditions, the servant served for 9 months --/
theorem servant_served_nine_months (s : ServantCompensation)
  (h1 : s.fullYearSalary = 600)
  (h2 : s.uniformPrice = 200)
  (h3 : s.receivedSalary = 400)
  : s.monthsServed = 9 := by
  sorry


end NUMINAMATH_CALUDE_servant_served_nine_months_l173_17345


namespace NUMINAMATH_CALUDE_fraction_irreducible_l173_17369

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l173_17369


namespace NUMINAMATH_CALUDE_five_digit_number_counts_l173_17399

def digits := [0, 1, 2, 3, 4]

/-- Count of five-digit numbers without repeated digits using 0, 1, 2, 3, and 4
    that are greater than 21035 and even -/
def count_greater_than_21035_and_even : ℕ := 39

/-- Count of five-digit even numbers without repeated digits using 0, 1, 2, 3, and 4
    with the second and fourth digits from the left being odd numbers -/
def count_even_with_odd_second_and_fourth : ℕ := 8

/-- Theorem stating the counts of numbers satisfying the given conditions -/
theorem five_digit_number_counts :
  (count_greater_than_21035_and_even = 39) ∧
  (count_even_with_odd_second_and_fourth = 8) := by
  sorry

end NUMINAMATH_CALUDE_five_digit_number_counts_l173_17399


namespace NUMINAMATH_CALUDE_average_rate_of_change_l173_17302

def f (x : ℝ) : ℝ := x^2 + x

theorem average_rate_of_change (a b : ℝ) (h : a = 1 ∧ b = 2) :
  (f b - f a) / (b - a) = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_of_change_l173_17302


namespace NUMINAMATH_CALUDE_max_product_of_tangent_circles_l173_17300

/-- Two circles C₁ and C₂ are externally tangent -/
def externally_tangent (a b : ℝ) : Prop :=
  a + b = 3

/-- The product of a and b -/
def product (a b : ℝ) : ℝ := a * b

/-- The theorem stating the maximum value of ab -/
theorem max_product_of_tangent_circles (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_tangent : externally_tangent a b) :
  product a b ≤ 9/4 :=
sorry

end NUMINAMATH_CALUDE_max_product_of_tangent_circles_l173_17300


namespace NUMINAMATH_CALUDE_intersection_point_slope_l173_17358

/-- Given three lines in a plane, if two of them intersect at a point on the third line, 
    then the slope of one of the intersecting lines is 4. -/
theorem intersection_point_slope (k : ℝ) : 
  (∃ x y : ℝ, y = -2*x + 4 ∧ y = k*x ∧ y = x + 2) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_slope_l173_17358


namespace NUMINAMATH_CALUDE_min_value_a_l173_17393

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, |x + a| - |x + 1| ≤ 2*a) ↔ a ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l173_17393


namespace NUMINAMATH_CALUDE_positive_rationals_decomposition_l173_17368

-- Define the set of positive integers
def PositiveIntegers : Set ℚ := {x : ℚ | x > 0 ∧ x.den = 1}

-- Define the set of positive fractions
def PositiveFractions : Set ℚ := {x : ℚ | x > 0 ∧ x.den ≠ 1}

-- Define the set of positive rational numbers
def PositiveRationals : Set ℚ := {x : ℚ | x > 0}

-- Theorem statement
theorem positive_rationals_decomposition :
  PositiveRationals = PositiveIntegers ∪ PositiveFractions :=
by sorry

end NUMINAMATH_CALUDE_positive_rationals_decomposition_l173_17368


namespace NUMINAMATH_CALUDE_box_calories_l173_17387

/-- Calculates the total calories in a box of cookies -/
def total_calories (cookies_per_bag : ℕ) (bags_per_box : ℕ) (calories_per_cookie : ℕ) : ℕ :=
  cookies_per_bag * bags_per_box * calories_per_cookie

/-- Theorem: The total calories in a box of cookies is 1600 -/
theorem box_calories :
  total_calories 20 4 20 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_box_calories_l173_17387


namespace NUMINAMATH_CALUDE_combined_age_theorem_l173_17328

/-- The combined age of Jane and John after 12 years -/
def combined_age_after_12_years (justin_age : ℕ) (jessica_age_diff : ℕ) (james_age_diff : ℕ) (julia_age_diff : ℕ) (jane_age_diff : ℕ) (john_age_diff : ℕ) : ℕ :=
  let jessica_age := justin_age + jessica_age_diff
  let james_age := jessica_age + james_age_diff
  let jane_age := james_age + jane_age_diff
  let john_age := jane_age + john_age_diff
  (jane_age + 12) + (john_age + 12)

/-- Theorem stating the combined age of Jane and John after 12 years -/
theorem combined_age_theorem :
  combined_age_after_12_years 26 6 7 8 25 3 = 155 := by
  sorry

end NUMINAMATH_CALUDE_combined_age_theorem_l173_17328


namespace NUMINAMATH_CALUDE_complex_number_equality_l173_17337

theorem complex_number_equality : (1 - Complex.I)^2 * (1 + Complex.I) = 2 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l173_17337


namespace NUMINAMATH_CALUDE_goods_train_speed_l173_17355

theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (passing_time : ℝ) 
  (goods_train_length : ℝ) :
  man_train_speed = 100 →
  passing_time = 9 →
  goods_train_length = 280 →
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 12 ∧
    goods_train_length = (man_train_speed + goods_train_speed) * (5/18) * passing_time :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l173_17355


namespace NUMINAMATH_CALUDE_remaining_payment_l173_17394

/-- Given a 10% deposit of $80, prove that the remaining amount to be paid is $720 -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (total_price : ℝ) : 
  deposit = 80 →
  deposit_percentage = 0.1 →
  deposit = deposit_percentage * total_price →
  total_price - deposit = 720 := by
sorry

end NUMINAMATH_CALUDE_remaining_payment_l173_17394


namespace NUMINAMATH_CALUDE_valid_arrangements_equal_catalan_number_l173_17336

/-- Represents a person in the queue with their bill denomination -/
inductive Person
  | fiveYuan : Person
  | tenYuan : Person

/-- Checks if a sequence of people is a valid queue arrangement -/
def isValidArrangement (queue : List Person) : Bool :=
  let rec check (acc : Int) (remaining : List Person) : Bool :=
    match remaining with
    | [] => true
    | Person.fiveYuan :: rest => check (acc + 1) rest
    | Person.tenYuan :: rest => if acc > 0 then check (acc - 1) rest else false
  check 0 queue

/-- Counts the number of valid queue arrangements for n people with 5 yuan and n people with 10 yuan -/
def countValidArrangements (n : Nat) : Nat :=
  let people := List.replicate n Person.fiveYuan ++ List.replicate n Person.tenYuan
  (List.permutations people).filter isValidArrangement |>.length

/-- The nth Catalan number -/
def catalanNumber (n : Nat) : Nat :=
  (Nat.choose (2 * n) n) / (n + 1)

theorem valid_arrangements_equal_catalan_number (n : Nat) :
  countValidArrangements n = catalanNumber n := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_equal_catalan_number_l173_17336


namespace NUMINAMATH_CALUDE_line_equation_through_points_l173_17319

/-- The equation of a line passing through two given points -/
theorem line_equation_through_points (x y : ℝ) : 
  (2 * x - y - 2 = 0) ↔ 
  ((x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = -2) ∨ 
   (∃ t : ℝ, x = 1 - t ∧ y = 0 + 2*t)) :=
sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l173_17319


namespace NUMINAMATH_CALUDE_rational_sum_and_sum_of_squares_coprime_to_six_l173_17305

theorem rational_sum_and_sum_of_squares_coprime_to_six (a b : ℚ) :
  let S := a + b
  (S = a + b) → (S = a^2 + b^2) → ∃ (m k : ℤ), S = m / k ∧ k ≠ 0 ∧ Nat.Coprime k.natAbs 6 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_and_sum_of_squares_coprime_to_six_l173_17305


namespace NUMINAMATH_CALUDE_initial_white_lights_correct_l173_17372

/-- The number of white lights Malcolm had initially -/
def initial_white_lights : ℕ := 59

/-- The number of red lights Malcolm bought -/
def red_lights : ℕ := 12

/-- The number of blue lights Malcolm bought -/
def blue_lights : ℕ := 3 * red_lights

/-- The number of green lights Malcolm bought -/
def green_lights : ℕ := 6

/-- The number of colored lights Malcolm still needs to buy -/
def remaining_lights : ℕ := 5

/-- Theorem stating that the initial number of white lights is correct -/
theorem initial_white_lights_correct : 
  initial_white_lights = red_lights + blue_lights + green_lights + remaining_lights :=
sorry

end NUMINAMATH_CALUDE_initial_white_lights_correct_l173_17372


namespace NUMINAMATH_CALUDE_collage_glue_drops_l173_17325

/-- Calculates the total number of glue drops needed for a collage -/
def total_glue_drops (num_friends : ℕ) (clippings_per_friend : ℕ) (glue_drops_per_clipping : ℕ) : ℕ :=
  num_friends * clippings_per_friend * glue_drops_per_clipping

/-- Proves that for 7 friends, 3 clippings per friend, and 6 drops of glue per clipping, 
    the total number of glue drops needed is 126 -/
theorem collage_glue_drops : 
  total_glue_drops 7 3 6 = 126 := by
  sorry

end NUMINAMATH_CALUDE_collage_glue_drops_l173_17325


namespace NUMINAMATH_CALUDE_distance_PQ_l173_17347

/-- The distance between two points P and Q, where P is on the line 6y = 13x + 7, 
    Q is on the line 7y = 2x - 5, and (10, 4) is the midpoint of PQ. -/
theorem distance_PQ : ∃ (P Q : ℝ × ℝ),
  (6 * P.2 = 13 * P.1 + 7) ∧
  (7 * Q.2 = 2 * Q.1 - 5) ∧
  ((10 : ℝ), 4) = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) ∧
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 
    Real.sqrt ((21401/1087)^2 + (1141/7609)^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_PQ_l173_17347


namespace NUMINAMATH_CALUDE_square_area_increase_l173_17366

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let original_area := s^2
  let new_side := 1.1 * s
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l173_17366


namespace NUMINAMATH_CALUDE_smallest_quotient_is_seven_l173_17382

/-- A type representing a division of numbers 1 to 10 into two groups -/
def Division := (Finset Nat) × (Finset Nat)

/-- Checks if a division is valid (contains all numbers from 1 to 10 exactly once) -/
def is_valid_division (d : Division) : Prop :=
  d.1 ∪ d.2 = Finset.range 10 ∧ d.1 ∩ d.2 = ∅

/-- Calculates the product of numbers in a Finset -/
def product (s : Finset Nat) : Nat :=
  s.prod id

/-- Checks if the division satisfies the divisibility condition -/
def satisfies_condition (d : Division) : Prop :=
  (product d.1) % (product d.2) = 0

/-- The main theorem stating the smallest possible quotient is 7 -/
theorem smallest_quotient_is_seven :
  ∀ d : Division, 
    is_valid_division d → 
    satisfies_condition d → 
    (product d.1) / (product d.2) ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_quotient_is_seven_l173_17382


namespace NUMINAMATH_CALUDE_not_all_even_P_true_l173_17322

/-- A proposition P on even natural numbers -/
def P : ℕ → Prop := sorry

/-- Theorem stating that we cannot conclude P holds for all even natural numbers -/
theorem not_all_even_P_true :
  (∀ n : ℕ, n ≤ 1001 → P (2 * n)) →
  ¬(∀ k : ℕ, Even k → P k) :=
by sorry

end NUMINAMATH_CALUDE_not_all_even_P_true_l173_17322


namespace NUMINAMATH_CALUDE_a_range_when_A_B_disjoint_l173_17378

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 1| ≤ a ∧ a > 0}
def B : Set ℝ := {x : ℝ | x^2 - 6*x - 7 > 0}

-- State the theorem
theorem a_range_when_A_B_disjoint :
  ∀ a : ℝ, (A a ∩ B = ∅) ↔ (0 < a ∧ a ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_a_range_when_A_B_disjoint_l173_17378


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l173_17362

/-- Given a quadratic function f(x) = mx² - 2x - 3 where the solution set of f(x) ≤ 0 is (-1, 3),
    prove that for 0 < a < 1, the minimum value of y = f(a^x) - 3a^(x+1) for x ∈ [1, 2]
    is -5 if and only if a = (√5 - 1) / 2 -/
theorem quadratic_function_minimum (m : ℝ) (a : ℝ) 
    (hm : m > 0)
    (ha : 0 < a ∧ a < 1)
    (hf : ∀ x, (m * x^2 - 2*x - 3 ≤ 0) ↔ (-1 < x ∧ x < 3)) :
  (∀ x ∈ Set.Icc 1 2, m * (a^x)^2 - 2*(a^x) - 3 - 3*a^(x+1) ≥ -5) ∧
  (∃ x ∈ Set.Icc 1 2, m * (a^x)^2 - 2*(a^x) - 3 - 3*a^(x+1) = -5) ↔
  a = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l173_17362


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l173_17311

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_sum_theorem (a b c : ℕ) 
  (ha : isPrime a) (hb : isPrime b) (hc : isPrime c)
  (h1 : b + c = 13) (h2 : c^2 - a^2 = 72) : a + b + c = 15 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l173_17311


namespace NUMINAMATH_CALUDE_lcm_gcd_difference_times_min_l173_17320

theorem lcm_gcd_difference_times_min (a b : ℕ) (ha : a = 8) (hb : b = 12) :
  (Nat.lcm a b - Nat.gcd a b) * min a b = 160 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_difference_times_min_l173_17320


namespace NUMINAMATH_CALUDE_complex_power_210_deg_30_l173_17354

theorem complex_power_210_deg_30 :
  (Complex.exp (210 * Real.pi / 180 * Complex.I)) ^ 30 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_210_deg_30_l173_17354


namespace NUMINAMATH_CALUDE_power_equation_solution_l173_17329

theorem power_equation_solution : ∃ m : ℤ, 2^4 - 3 = 5^2 + m ∧ m = -12 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l173_17329


namespace NUMINAMATH_CALUDE_greatest_display_groups_l173_17332

theorem greatest_display_groups (plates spoons glasses bowls : ℕ) 
  (h_plates : plates = 3219)
  (h_spoons : spoons = 5641)
  (h_glasses : glasses = 1509)
  (h_bowls : bowls = 2387) :
  Nat.gcd (Nat.gcd (Nat.gcd plates spoons) glasses) bowls = 1 := by
  sorry

end NUMINAMATH_CALUDE_greatest_display_groups_l173_17332


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_l173_17396

/-- A cubic polynomial with specific values at 0, 1, and -1 -/
structure CubicPolynomial (k : ℝ) where
  P : ℝ → ℝ
  is_cubic : ∃ a b c : ℝ, ∀ x, P x = a * x^3 + b * x^2 + c * x + k
  value_at_zero : P 0 = k
  value_at_one : P 1 = 2 * k
  value_at_neg_one : P (-1) = 3 * k

/-- The sum of the polynomial evaluated at 2 and -2 equals 14k -/
theorem cubic_polynomial_sum (k : ℝ) (p : CubicPolynomial k) :
  p.P 2 + p.P (-2) = 14 * k := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_l173_17396


namespace NUMINAMATH_CALUDE_function_equivalence_l173_17312

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x, f (2 * x + 1) = 2 * x^2 + 1) :
  ∀ x, f x = 1/2 * x^2 - x + 3/2 := by
sorry

end NUMINAMATH_CALUDE_function_equivalence_l173_17312


namespace NUMINAMATH_CALUDE_elephant_giraffe_jade_ratio_l173_17365

/-- The amount of jade (in grams) needed for a giraffe statue -/
def giraffe_jade : ℝ := 120

/-- The selling price of a giraffe statue -/
def giraffe_price : ℝ := 150

/-- The selling price of an elephant statue -/
def elephant_price : ℝ := 350

/-- The total amount of jade Nancy has -/
def total_jade : ℝ := 1920

/-- The additional revenue from making all elephants instead of giraffes -/
def additional_revenue : ℝ := 400

/-- The ratio of jade used for an elephant statue to a giraffe statue -/
def jade_ratio : ℝ := 2

theorem elephant_giraffe_jade_ratio :
  let elephant_jade := giraffe_jade * jade_ratio
  let giraffe_count := total_jade / giraffe_jade
  let elephant_count := total_jade / elephant_jade
  giraffe_count * giraffe_price + additional_revenue = elephant_count * elephant_price :=
sorry

end NUMINAMATH_CALUDE_elephant_giraffe_jade_ratio_l173_17365


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l173_17392

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- Definition of Line 1: ax + 4y - 2 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 4 * y - 2 = 0

/-- Definition of Line 2: 2x - 5y + b = 0 -/
def line2 (b : ℝ) (x y : ℝ) : Prop := 2 * x - 5 * y + b = 0

/-- The foot of the perpendicular (1, c) lies on both lines -/
def foot_on_lines (a b c : ℝ) : Prop := line1 a 1 c ∧ line2 b 1 c

theorem perpendicular_lines_sum (a b c : ℝ) : 
  perpendicular (-a/4) (2/5) → foot_on_lines a b c → a + b + c = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l173_17392


namespace NUMINAMATH_CALUDE_bombay_express_speed_l173_17348

/-- The speed of Bombay Express in km/h -/
def speed_bombay_express : ℝ := 60

/-- The speed of Rajdhani Express in km/h -/
def speed_rajdhani_express : ℝ := 80

/-- The distance at which the trains meet in km -/
def meeting_distance : ℝ := 480

/-- The time difference between the departures of the two trains in hours -/
def time_difference : ℝ := 2

theorem bombay_express_speed :
  speed_bombay_express = meeting_distance / (meeting_distance / speed_rajdhani_express + time_difference) :=
by sorry

end NUMINAMATH_CALUDE_bombay_express_speed_l173_17348


namespace NUMINAMATH_CALUDE_zuca_win_probability_l173_17357

/-- The Game played on a regular hexagon --/
structure TheGame where
  /-- Number of vertices in the hexagon --/
  vertices : Nat
  /-- Number of players --/
  players : Nat
  /-- Probability of Bamal and Halvan moving to adjacent vertices --/
  prob_adjacent : ℚ
  /-- Probability of Zuca moving to adjacent or opposite vertices --/
  prob_zuca_move : ℚ

/-- The specific instance of The Game as described in the problem --/
def gameInstance : TheGame :=
  { vertices := 6
  , players := 3
  , prob_adjacent := 1/2
  , prob_zuca_move := 1/3 }

/-- The probability that Zuca hasn't lost when The Game ends --/
def probZucaWins (g : TheGame) : ℚ :=
  29/90

/-- Theorem stating that the probability of Zuca not losing is 29/90 --/
theorem zuca_win_probability (g : TheGame) :
  g = gameInstance → probZucaWins g = 29/90 := by
  sorry

end NUMINAMATH_CALUDE_zuca_win_probability_l173_17357


namespace NUMINAMATH_CALUDE_simplify_expression_l173_17310

theorem simplify_expression (x : ℝ) : x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -5*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l173_17310


namespace NUMINAMATH_CALUDE_shooting_probabilities_l173_17308

/-- Probability of hitting a specific ring in one shot -/
def ring_probability : Fin 3 → ℝ
| 0 => 0.13  -- 10-ring
| 1 => 0.28  -- 9-ring
| 2 => 0.31  -- 8-ring

/-- The sum of probabilities for 10-ring and 9-ring -/
def prob_10_or_9 : ℝ := ring_probability 0 + ring_probability 1

/-- The probability of hitting less than 9 rings -/
def prob_less_than_9 : ℝ := 1 - prob_10_or_9

theorem shooting_probabilities :
  prob_10_or_9 = 0.41 ∧ prob_less_than_9 = 0.59 := by sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l173_17308


namespace NUMINAMATH_CALUDE_mary_cut_ten_roses_l173_17364

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Proof that Mary cut 10 roses from her garden -/
theorem mary_cut_ten_roses (initial_roses final_roses : ℕ)
  (h1 : initial_roses = 6)
  (h2 : final_roses = 16) :
  roses_cut initial_roses final_roses = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_cut_ten_roses_l173_17364


namespace NUMINAMATH_CALUDE_cousins_distribution_l173_17352

-- Define the number of cousins and rooms
def num_cousins : ℕ := 5
def num_rooms : ℕ := 3

-- Function to calculate the number of ways to distribute cousins
def distribute_cousins (n : ℕ) (k : ℕ) : ℕ := sorry

-- Theorem stating the result
theorem cousins_distribution :
  distribute_cousins num_cousins num_rooms = 66 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_l173_17352


namespace NUMINAMATH_CALUDE_chess_game_probability_l173_17391

theorem chess_game_probability (p_draw p_B_win p_A_win : ℚ) :
  p_draw = 1/2 →
  p_B_win = 1/3 →
  p_draw + p_B_win + p_A_win = 1 →
  p_A_win = 1/6 := by
sorry

end NUMINAMATH_CALUDE_chess_game_probability_l173_17391


namespace NUMINAMATH_CALUDE_chef_lunch_meals_l173_17339

theorem chef_lunch_meals (meals_sold_lunch : ℕ) (meals_prepared_dinner : ℕ) (total_dinner_meals : ℕ)
  (h1 : meals_sold_lunch = 12)
  (h2 : meals_prepared_dinner = 5)
  (h3 : total_dinner_meals = 10) :
  meals_sold_lunch + (total_dinner_meals - meals_prepared_dinner) = 17 :=
by sorry

end NUMINAMATH_CALUDE_chef_lunch_meals_l173_17339


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l173_17318

theorem cubic_roots_relation (a b c : ℂ) : 
  (a^3 - 3*a^2 + 5*a - 8 = 0) → 
  (b^3 - 3*b^2 + 5*b - 8 = 0) → 
  (c^3 - 3*c^2 + 5*c - 8 = 0) → 
  (∃ r s : ℂ, (a-b)^3 + r*(a-b)^2 + s*(a-b) + 243 = 0 ∧ 
               (b-c)^3 + r*(b-c)^2 + s*(b-c) + 243 = 0 ∧ 
               (c-a)^3 + r*(c-a)^2 + s*(c-a) + 243 = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l173_17318


namespace NUMINAMATH_CALUDE_pool_perimeter_is_20_l173_17309

/-- Represents the dimensions and properties of a garden with a rectangular pool -/
structure GardenPool where
  garden_length : ℝ
  garden_width : ℝ
  pool_area : ℝ
  walkway_width : ℝ

/-- Calculates the perimeter of the pool given the garden dimensions and pool properties -/
def pool_perimeter (g : GardenPool) : ℝ :=
  2 * ((g.garden_length - 2 * g.walkway_width) + (g.garden_width - 2 * g.walkway_width))

/-- Theorem stating that the perimeter of the pool is 20 meters under the given conditions -/
theorem pool_perimeter_is_20 (g : GardenPool) 
    (h1 : g.garden_length = 8)
    (h2 : g.garden_width = 6)
    (h3 : g.pool_area = 24)
    (h4 : (g.garden_length - 2 * g.walkway_width) * (g.garden_width - 2 * g.walkway_width) = g.pool_area) :
  pool_perimeter g = 20 := by
  sorry

#check pool_perimeter_is_20

end NUMINAMATH_CALUDE_pool_perimeter_is_20_l173_17309


namespace NUMINAMATH_CALUDE_find_p_value_l173_17342

theorem find_p_value (x y z p : ℝ) 
  (h1 : 8 / (x + y) = p / (x + z)) 
  (h2 : p / (x + z) = 12 / (z - y)) : p = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_p_value_l173_17342


namespace NUMINAMATH_CALUDE_remainder_252_power_252_mod_13_l173_17370

theorem remainder_252_power_252_mod_13 : 252^252 ≡ 1 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_remainder_252_power_252_mod_13_l173_17370


namespace NUMINAMATH_CALUDE_x_squared_minus_one_necessary_not_sufficient_l173_17324

theorem x_squared_minus_one_necessary_not_sufficient :
  (∀ x : ℝ, x - 1 = 0 → x^2 - 1 = 0) ∧
  ¬(∀ x : ℝ, x^2 - 1 = 0 → x - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_minus_one_necessary_not_sufficient_l173_17324


namespace NUMINAMATH_CALUDE_marks_additional_height_l173_17377

/-- Proves that Mark is 3 inches tall in addition to his height in feet given the conditions -/
theorem marks_additional_height :
  -- Define constants
  let feet_to_inches : ℕ := 12
  let marks_feet : ℕ := 5
  let mikes_feet : ℕ := 6
  let mikes_additional_inches : ℕ := 1
  let height_difference : ℕ := 10

  -- Calculate Mike's height in inches
  let mikes_height : ℕ := mikes_feet * feet_to_inches + mikes_additional_inches

  -- Calculate Mark's height in inches
  let marks_height : ℕ := mikes_height - height_difference

  -- Calculate Mark's additional inches
  let marks_additional_inches : ℕ := marks_height - (marks_feet * feet_to_inches)

  -- Theorem statement
  marks_additional_inches = 3 := by
  sorry

end NUMINAMATH_CALUDE_marks_additional_height_l173_17377


namespace NUMINAMATH_CALUDE_infinite_series_not_computable_l173_17383

/-- An infinite series of natural numbers -/
def infinite_series (n : ℕ) : ℕ := n

/-- A predicate indicating whether a series can be computed algorithmically -/
def is_algorithmically_computable (f : ℕ → ℕ) : Prop :=
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → f n = 0

/-- The theorem stating that the infinite series cannot be computed algorithmically -/
theorem infinite_series_not_computable :
  ¬ (is_algorithmically_computable infinite_series) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_not_computable_l173_17383


namespace NUMINAMATH_CALUDE_cross_section_area_unit_cube_l173_17343

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  edge_length : ℝ

/-- Theorem: Area of cross-section in a unit cube -/
theorem cross_section_area_unit_cube (c : Cube) (X Y Z : Point3D) :
  c.edge_length = 1 →
  X = ⟨1/2, 1/2, 0⟩ →
  Y = ⟨1, 1/2, 1/2⟩ →
  Z = ⟨3/4, 3/4, 3/4⟩ →
  let sphere_radius := Real.sqrt 3 / 2
  let plane_distance := Real.sqrt ((1/4)^2 + (1/4)^2 + (3/4)^2)
  let cross_section_radius := Real.sqrt (sphere_radius^2 - plane_distance^2)
  let cross_section_area := π * cross_section_radius^2
  cross_section_area = 5 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_unit_cube_l173_17343


namespace NUMINAMATH_CALUDE_quadratic_equation_transform_l173_17331

theorem quadratic_equation_transform (x : ℝ) :
  25 * x^2 - 10 * x - 1000 = 0 →
  ∃ (r : ℝ), (x + r)^2 = 40.04 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_transform_l173_17331


namespace NUMINAMATH_CALUDE_inequality_proof_l173_17313

theorem inequality_proof (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) (h₄ : a₄ > 1) (h₅ : a₅ > 1) : 
  16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) > (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l173_17313


namespace NUMINAMATH_CALUDE_balloon_problem_l173_17321

/-- Represents the balloon counts for a person -/
structure BalloonCount where
  red : Nat
  blue : Nat

/-- Calculates the total cost of balloons -/
def totalCost (redCount blue_count : Nat) (redCost blueCost : Nat) : Nat :=
  redCount * redCost + blue_count * blueCost

/-- Theorem statement for the balloon problem -/
theorem balloon_problem 
  (fred sam dan : BalloonCount)
  (redCost blueCost : Nat)
  (h1 : fred = ⟨10, 5⟩)
  (h2 : sam = ⟨46, 20⟩)
  (h3 : dan = ⟨16, 12⟩)
  (h4 : redCost = 10)
  (h5 : blueCost = 5) :
  let totalRed := fred.red + sam.red + dan.red
  let totalBlue := fred.blue + sam.blue + dan.blue
  let totalCost := totalCost totalRed totalBlue redCost blueCost
  totalRed = 72 ∧ totalBlue = 37 ∧ totalCost = 905 := by
  sorry


end NUMINAMATH_CALUDE_balloon_problem_l173_17321


namespace NUMINAMATH_CALUDE_keith_attended_games_l173_17361

def total_games : ℕ := 8
def missed_games : ℕ := 4

theorem keith_attended_games :
  total_games - missed_games = 4 := by sorry

end NUMINAMATH_CALUDE_keith_attended_games_l173_17361


namespace NUMINAMATH_CALUDE_max_cubes_from_seven_points_l173_17379

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Determines if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Determines if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (plane : Plane3D), pointOnPlane p1 plane ∧ pointOnPlane p2 plane ∧ 
                       pointOnPlane p3 plane ∧ pointOnPlane p4 plane

/-- Represents a cube determined by 7 points -/
structure Cube where
  a1 : Point3D
  a2 : Point3D
  f1 : Point3D
  f2 : Point3D
  e : Point3D
  h : Point3D
  j : Point3D
  lowerFace : Plane3D
  upperFace : Plane3D
  frontFace : Plane3D
  backFace : Plane3D
  rightFace : Plane3D

/-- The main theorem to prove -/
theorem max_cubes_from_seven_points 
  (a1 a2 f1 f2 e h j : Point3D)
  (h1 : pointOnPlane a1 (Cube.lowerFace cube))
  (h2 : pointOnPlane a2 (Cube.lowerFace cube))
  (h3 : pointOnPlane f1 (Cube.upperFace cube))
  (h4 : pointOnPlane f2 (Cube.upperFace cube))
  (h5 : ¬ areCoplanar a1 a2 f1 f2)
  (h6 : pointOnPlane e (Cube.frontFace cube))
  (h7 : pointOnPlane h (Cube.backFace cube))
  (h8 : pointOnPlane j (Cube.rightFace cube))
  : ∃ (n : ℕ), n ≤ 2 ∧ ∀ (m : ℕ), (∃ (cubes : Fin m → Cube), 
    (∀ (i : Fin m), 
      Cube.a1 (cubes i) = a1 ∧
      Cube.a2 (cubes i) = a2 ∧
      Cube.f1 (cubes i) = f1 ∧
      Cube.f2 (cubes i) = f2 ∧
      Cube.e (cubes i) = e ∧
      Cube.h (cubes i) = h ∧
      Cube.j (cubes i) = j) ∧
    (∀ (i j : Fin m), i ≠ j → cubes i ≠ cubes j)) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_cubes_from_seven_points_l173_17379


namespace NUMINAMATH_CALUDE_complex_calculation_l173_17376

def a : ℂ := 3 + 2*Complex.I
def b : ℂ := 1 - 2*Complex.I

theorem complex_calculation : 3*a - 4*b = 5 + 14*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l173_17376


namespace NUMINAMATH_CALUDE_different_color_probability_l173_17390

def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 3
def green_chips : ℕ := 2

def total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

theorem different_color_probability :
  let p_blue : ℚ := blue_chips / total_chips
  let p_red : ℚ := red_chips / total_chips
  let p_yellow : ℚ := yellow_chips / total_chips
  let p_green : ℚ := green_chips / total_chips
  
  p_blue * (1 - p_blue) + p_red * (1 - p_red) + 
  p_yellow * (1 - p_yellow) + p_green * (1 - p_green) = 91 / 128 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l173_17390


namespace NUMINAMATH_CALUDE_f_always_above_y_l173_17335

/-- The function f(x) = mx^2 - 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 3

/-- The line y = mx - m -/
def y (m : ℝ) (x : ℝ) : ℝ := m * x - m

/-- Theorem stating that f(x) is always above y for all real x if and only if m > 4 -/
theorem f_always_above_y (m : ℝ) : 
  (∀ x : ℝ, f m x > y m x) ↔ m > 4 := by
  sorry

end NUMINAMATH_CALUDE_f_always_above_y_l173_17335


namespace NUMINAMATH_CALUDE_last_digit_fifth_power_l173_17314

theorem last_digit_fifth_power (R : ℤ) : 10 ∣ (R^5 - R) := by
  sorry

end NUMINAMATH_CALUDE_last_digit_fifth_power_l173_17314


namespace NUMINAMATH_CALUDE_mixed_number_sum_l173_17360

theorem mixed_number_sum : 
  (2 + 1/10) + (3 + 11/100) + (4 + 111/1000) = 9321/1000 := by sorry

end NUMINAMATH_CALUDE_mixed_number_sum_l173_17360


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l173_17326

-- Problem 1
theorem problem_1 (a b : ℝ) :
  a^2 * (2*a*b - 1) + (a - 3*b) * (a + b) = 2*a^3*b - 2*a*b - 3*b^2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) :
  (2*x - 3)^2 - (x + 2)^2 = 3*x^2 - 16*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l173_17326


namespace NUMINAMATH_CALUDE_ellipse_m_value_l173_17340

/-- Represents an ellipse with equation x^2 + my^2 = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ x y : ℝ, x^2 + m * y^2 = 1

/-- Indicates that the foci of the ellipse are on the y-axis -/
def foci_on_y_axis (e : Ellipse m) : Prop :=
  ∃ c : ℝ, c^2 = 1/m - 1 ∧ c ≥ 0

/-- The length of the major axis is twice the length of the minor axis -/
def major_axis_twice_minor (e : Ellipse m) : Prop :=
  2 * (1 : ℝ) = Real.sqrt (1/m)

/-- Theorem stating that m = 1/4 for the given ellipse properties -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m)
  (h1 : foci_on_y_axis e)
  (h2 : major_axis_twice_minor e) :
  m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l173_17340


namespace NUMINAMATH_CALUDE_jelly_beans_solution_l173_17381

/-- The number of jelly beans in jar Y -/
def jelly_beans_Y : ℕ := sorry

/-- The number of jelly beans in jar X -/
def jelly_beans_X : ℕ := 3 * jelly_beans_Y - 400

/-- The total number of jelly beans -/
def total_jelly_beans : ℕ := 1200

theorem jelly_beans_solution :
  jelly_beans_X + jelly_beans_Y = total_jelly_beans ∧ jelly_beans_Y = 400 := by sorry

end NUMINAMATH_CALUDE_jelly_beans_solution_l173_17381


namespace NUMINAMATH_CALUDE_january_first_is_monday_l173_17315

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with its properties -/
structure Month where
  days : Nat
  firstDay : DayOfWeek
  mondayCount : Nat
  thursdayCount : Nat

/-- Theorem stating that a month with 31 days, 5 Mondays, and 4 Thursdays must start on a Monday -/
theorem january_first_is_monday (m : Month) :
  m.days = 31 ∧ m.mondayCount = 5 ∧ m.thursdayCount = 4 →
  m.firstDay = DayOfWeek.Monday := by
  sorry


end NUMINAMATH_CALUDE_january_first_is_monday_l173_17315


namespace NUMINAMATH_CALUDE_factorial_ratio_l173_17334

theorem factorial_ratio : (12 : ℕ).factorial / (11 : ℕ).factorial = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l173_17334


namespace NUMINAMATH_CALUDE_isosceles_triangle_removal_l173_17327

/-- Given a square with side length x, from which isosceles right triangles
    with leg length s are removed from each corner to form a rectangle
    with longer side 15 units, prove that the combined area of the four
    removed triangles is 225 square units. -/
theorem isosceles_triangle_removal (x s : ℝ) : 
  x > 0 →
  s > 0 →
  x - 2*s = 15 →
  (x - s)^2 + (x - s)^2 = x^2 →
  4 * (1/2 * s^2) = 225 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_removal_l173_17327


namespace NUMINAMATH_CALUDE_system_solution_implies_k_value_l173_17349

theorem system_solution_implies_k_value (k : ℝ) (x y : ℝ) : 
  x + y = 5 * k →
  x - y = 9 * k →
  x - 2 * y = 22 →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_implies_k_value_l173_17349


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l173_17385

theorem polynomial_evaluation : (3 : ℝ)^3 + (3 : ℝ)^2 + (3 : ℝ) + 1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l173_17385


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l173_17344

theorem max_value_sum_of_roots (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -2/3)
  (z_ge : z ≥ -2) :
  (∀ a b c : ℝ, a + b + c = 3 → a ≥ -1 → b ≥ -2/3 → c ≥ -2 →
    Real.sqrt (3*a + 3) + Real.sqrt (3*b + 2) + Real.sqrt (3*c + 6) ≤ 
    Real.sqrt (3*x + 3) + Real.sqrt (3*y + 2) + Real.sqrt (3*z + 6)) ∧
  Real.sqrt (3*x + 3) + Real.sqrt (3*y + 2) + Real.sqrt (3*z + 6) = 2 * Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l173_17344


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l173_17371

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width height : ℝ) : ℝ :=
  length * width + 2 * length * height + 2 * width * height

/-- Theorem: The wet surface area of a cistern with given dimensions is 49 square meters -/
theorem cistern_wet_surface_area :
  wetSurfaceArea 6 4 1.25 = 49 := by sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l173_17371


namespace NUMINAMATH_CALUDE_log_equation_solution_l173_17386

theorem log_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ x - 2 > 0 ∧ x + 2 > 0 ∧
  Real.log x + Real.log (x - 2) = Real.log 3 + Real.log (x + 2) ∧
  x = 6 :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l173_17386


namespace NUMINAMATH_CALUDE_last_date_with_sum_property_l173_17304

/-- Represents a date in DD.MM.YYYY format -/
structure Date where
  day : Nat
  month : Nat
  year : Nat
  deriving Repr

/-- Checks if a given date is valid in the year 2008 -/
def isValidDate (d : Date) : Prop :=
  d.year = 2008 ∧
  d.month ≥ 1 ∧ d.month ≤ 12 ∧
  d.day ≥ 1 ∧ d.day ≤ 31 ∧
  (d.month ∈ [4, 6, 9, 11] → d.day ≤ 30) ∧
  (d.month = 2 → d.day ≤ 29)

/-- Extracts individual digits from a number -/
def digits (n : Nat) : List Nat :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

/-- Calculates the sum of the first four digits of a date -/
def sumFirstFour (d : Date) : Nat :=
  List.sum (List.take 4 (digits d.day ++ digits d.month))

/-- Calculates the sum of the last four digits of a date -/
def sumLastFour (d : Date) : Nat :=
  List.sum (List.take 4 (digits d.year).reverse)

/-- Checks if the sum of the first four digits equals the sum of the last four digits -/
def hasSumProperty (d : Date) : Prop :=
  sumFirstFour d = sumLastFour d

/-- States that December 25, 2008 is the last date in 2008 with the sum property -/
theorem last_date_with_sum_property :
  ∀ d : Date, isValidDate d → hasSumProperty d →
  d.year = 2008 → d.month ≤ 12 → d.day ≤ 25 :=
sorry

end NUMINAMATH_CALUDE_last_date_with_sum_property_l173_17304


namespace NUMINAMATH_CALUDE_expression_simplification_l173_17363

theorem expression_simplification :
  ((3 + 4 + 6 + 7) / 4) + ((2 * 6 + 10) / 4) = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l173_17363


namespace NUMINAMATH_CALUDE_biased_coin_probability_l173_17303

def probability_of_heads (h : ℝ) : ℝ := h

def probability_of_tails (h : ℝ) : ℝ := 1 - h

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

theorem biased_coin_probability 
  (h : ℝ) 
  (h_prob : 0 < h ∧ h < 1) 
  (h_equal_prob : binomial_coefficient 6 2 * (probability_of_heads h)^2 * (probability_of_tails h)^4 = 
                  binomial_coefficient 6 3 * (probability_of_heads h)^3 * (probability_of_tails h)^3) :
  binomial_coefficient 6 4 * (probability_of_heads h)^4 * (probability_of_tails h)^2 = 19440 / 117649 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l173_17303


namespace NUMINAMATH_CALUDE_power_order_l173_17395

theorem power_order : 
  let p := (2 : ℕ) ^ 3009
  let q := (3 : ℕ) ^ 2006
  let r := (5 : ℕ) ^ 1003
  r < p ∧ p < q := by sorry

end NUMINAMATH_CALUDE_power_order_l173_17395


namespace NUMINAMATH_CALUDE_no_solutions_to_inequality_l173_17353

theorem no_solutions_to_inequality :
  ¬∃ x : ℝ, (6 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 4) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_to_inequality_l173_17353


namespace NUMINAMATH_CALUDE_problem_solution_l173_17301

/-- Represents the contents of a box of colored balls -/
structure Box where
  red : Nat
  yellow : Nat
  blue : Nat

/-- Calculates the probability of Person B winning given the contents of both boxes -/
def probability_b_wins (box_a box_b : Box) : Rat :=
  let total_a := box_a.red + box_a.yellow + box_a.blue
  let total_b := box_b.red + box_b.yellow + box_b.blue
  ((box_a.red * box_b.red) + (box_a.yellow * box_b.yellow) + (box_a.blue * box_b.blue)) / (total_a * total_b)

/-- Calculates the average score for Person B given the contents of both boxes -/
def average_score_b (box_a box_b : Box) : Rat :=
  let total_a := box_a.red + box_a.yellow + box_a.blue
  let total_b := box_b.red + box_b.yellow + box_b.blue
  ((box_a.red * box_b.red * 1) + (box_a.yellow * box_b.yellow * 2) + (box_a.blue * box_b.blue * 3)) / (total_a * total_b)

theorem problem_solution :
  let box_a : Box := ⟨3, 2, 1⟩
  let box_b1 : Box := ⟨1, 2, 3⟩
  let box_b2 : Box := ⟨1, 4, 1⟩
  (probability_b_wins box_a box_b1 = 5/18) ∧
  (average_score_b box_a box_b2 = 11/18) ∧
  (∀ (x y z : Nat), x + y + z = 6 → average_score_b box_a ⟨x, y, z⟩ ≤ 11/18) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l173_17301


namespace NUMINAMATH_CALUDE_candy_distribution_l173_17375

theorem candy_distribution (a b d : ℕ) : 
  (4 * b = 3 * a) →  -- While Andrey eats 4 candies, Boris eats 3
  (6 * d = 7 * a) →  -- While Andrey eats 6 candies, Denis eats 7
  (a + b + d = 70) → -- Total candies eaten
  (a = 24 ∧ b = 18 ∧ d = 28) := by sorry

end NUMINAMATH_CALUDE_candy_distribution_l173_17375


namespace NUMINAMATH_CALUDE_triangle_side_length_l173_17317

theorem triangle_side_length (a : ℤ) : 
  (6 + 2 > a) ∧ (6 + a > 2) ∧ (a + 2 > 6) → a = 6 :=
by
  sorry

#check triangle_side_length

end NUMINAMATH_CALUDE_triangle_side_length_l173_17317


namespace NUMINAMATH_CALUDE_sum_of_zero_seven_representable_l173_17341

/-- A function that checks if a real number can be written using only 0 and 7 in decimal notation -/
def uses_only_zero_and_seven (x : ℝ) : Prop :=
  ∃ (digits : ℕ → ℕ), (∀ n, digits n ∈ ({0, 7} : Set ℕ)) ∧
    x = ∑' n, (digits n : ℝ) / 10^n

/-- Theorem stating that any positive real number can be represented as the sum of nine numbers,
    each of which in decimal notation consists of the digits 0 and 7 -/
theorem sum_of_zero_seven_representable (x : ℝ) (hx : 0 < x) :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ),
    x = a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ ∧
    (uses_only_zero_and_seven a₁) ∧
    (uses_only_zero_and_seven a₂) ∧
    (uses_only_zero_and_seven a₃) ∧
    (uses_only_zero_and_seven a₄) ∧
    (uses_only_zero_and_seven a₅) ∧
    (uses_only_zero_and_seven a₆) ∧
    (uses_only_zero_and_seven a₇) ∧
    (uses_only_zero_and_seven a₈) ∧
    (uses_only_zero_and_seven a₉) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_zero_seven_representable_l173_17341


namespace NUMINAMATH_CALUDE_scavenger_hunting_students_l173_17389

theorem scavenger_hunting_students (total : ℕ) (skiing : ℕ → ℕ) (scavenger : ℕ) :
  total = 12000 →
  skiing scavenger = 2 * scavenger →
  total = skiing scavenger + scavenger →
  scavenger = 4000 := by
sorry

end NUMINAMATH_CALUDE_scavenger_hunting_students_l173_17389
