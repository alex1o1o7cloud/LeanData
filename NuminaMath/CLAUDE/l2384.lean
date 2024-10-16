import Mathlib

namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l2384_238460

theorem no_positive_integer_solution :
  ¬∃ (x y : ℕ+), x^5 = y^2 + 4 := by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l2384_238460


namespace NUMINAMATH_CALUDE_pants_cost_is_correct_l2384_238462

/-- The cost of pants given total payment, cost of shirt, and change received -/
def cost_of_pants (total_payment shirt_cost change : ℚ) : ℚ :=
  total_payment - shirt_cost - change

/-- Theorem stating the cost of pants is $9.24 given the problem conditions -/
theorem pants_cost_is_correct :
  let total_payment : ℚ := 20
  let shirt_cost : ℚ := 8.25
  let change : ℚ := 2.51
  cost_of_pants total_payment shirt_cost change = 9.24 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_is_correct_l2384_238462


namespace NUMINAMATH_CALUDE_crackers_distribution_l2384_238447

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_person : ℕ) :
  total_crackers = 22 →
  num_friends = 11 →
  crackers_per_person = total_crackers / num_friends →
  crackers_per_person = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_crackers_distribution_l2384_238447


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l2384_238486

theorem inequality_and_minimum_value 
  (m n : ℝ) 
  (h_diff : m ≠ n) 
  (h_pos_m : m > 0) 
  (h_pos_n : n > 0) 
  (x y : ℝ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) :
  (m^2 / x + n^2 / y > (m + n)^2 / (x + y)) ∧
  (∃ (min_val : ℝ) (min_x : ℝ), 
    min_val = 64 ∧ 
    min_x = 1/8 ∧ 
    (∀ x, 0 < x ∧ x < 1/5 → 5/x + 9/(1-5*x) ≥ min_val) ∧
    (5/min_x + 9/(1-5*min_x) = min_val)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l2384_238486


namespace NUMINAMATH_CALUDE_quadratic_equation_has_two_distinct_real_roots_l2384_238476

/-- The quadratic equation x^2 - 3x - 1 = 0 has two distinct real roots -/
theorem quadratic_equation_has_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ - 1 = 0 ∧ x₂^2 - 3*x₂ - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_has_two_distinct_real_roots_l2384_238476


namespace NUMINAMATH_CALUDE_stating_smallest_n_with_constant_term_l2384_238489

/-- 
Given a positive integer n and the expression (4x^3 + 1/x^2)^n,
this function returns true if there exists a constant term in the expansion,
and false otherwise.
-/
def has_constant_term (n : ℕ+) : Prop :=
  ∃ r : ℕ, r ≤ n ∧ 3 * n = 5 * r

/-- 
Theorem stating that 5 is the smallest positive integer n 
for which there exists a constant term in the expansion of (4x^3 + 1/x^2)^n.
-/
theorem smallest_n_with_constant_term : 
  (∀ m : ℕ+, m < 5 → ¬has_constant_term m) ∧ has_constant_term 5 :=
sorry

end NUMINAMATH_CALUDE_stating_smallest_n_with_constant_term_l2384_238489


namespace NUMINAMATH_CALUDE_more_students_than_rabbits_l2384_238418

theorem more_students_than_rabbits : 
  let num_classrooms : ℕ := 6
  let students_per_classroom : ℕ := 22
  let rabbits_per_classroom : ℕ := 4
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_rabbits : ℕ := num_classrooms * rabbits_per_classroom
  total_students - total_rabbits = 108 := by
sorry

end NUMINAMATH_CALUDE_more_students_than_rabbits_l2384_238418


namespace NUMINAMATH_CALUDE_zero_points_property_l2384_238430

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x

theorem zero_points_property (a : ℝ) :
  (∃ x₂ : ℝ, x₂ ≠ sqrt e ∧ f a (sqrt e) = 0 ∧ f a x₂ = 0) →
  a = sqrt e / (2 * e) ∧ ∀ x₂ : ℝ, x₂ ≠ sqrt e ∧ f a x₂ = 0 → x₂ > e^(3/2) :=
by sorry

end NUMINAMATH_CALUDE_zero_points_property_l2384_238430


namespace NUMINAMATH_CALUDE_cos_2017pi_over_3_l2384_238401

theorem cos_2017pi_over_3 : Real.cos (2017 * Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2017pi_over_3_l2384_238401


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l2384_238433

theorem fraction_subtraction_equality : 
  (4 + 6 + 8 + 10) / (3 + 5 + 7) - (3 + 5 + 7 + 9) / (4 + 6 + 8) = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l2384_238433


namespace NUMINAMATH_CALUDE_eraser_ratio_l2384_238485

/-- The number of erasers each person has -/
structure EraserCounts where
  hanna : ℕ
  rachel : ℕ
  tanya : ℕ
  tanya_red : ℕ

/-- The conditions of the problem -/
def problem_conditions (counts : EraserCounts) : Prop :=
  counts.hanna = 2 * counts.rachel ∧
  counts.rachel = counts.tanya_red - 3 ∧
  counts.tanya = 20 ∧
  counts.tanya_red = counts.tanya / 2 ∧
  counts.hanna = 4

/-- The theorem to be proved -/
theorem eraser_ratio (counts : EraserCounts) 
  (h : problem_conditions counts) : 
  counts.rachel / counts.tanya_red = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_eraser_ratio_l2384_238485


namespace NUMINAMATH_CALUDE_cereal_eating_time_l2384_238417

theorem cereal_eating_time (fat_rate mr_thin_rate : ℚ) (total_cereal : ℚ) : 
  fat_rate = 1 / 25 →
  mr_thin_rate = 1 / 40 →
  total_cereal = 5 →
  (total_cereal / (fat_rate + mr_thin_rate) : ℚ) = 1000 / 13 := by
  sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l2384_238417


namespace NUMINAMATH_CALUDE_only_30_40_50_is_pythagorean_triple_l2384_238453

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem only_30_40_50_is_pythagorean_triple :
  (is_pythagorean_triple 30 40 50) ∧
  ¬(is_pythagorean_triple 1 1 2) ∧
  ¬(is_pythagorean_triple 1 2 2) ∧
  ¬(is_pythagorean_triple 7 14 15) :=
by sorry

end NUMINAMATH_CALUDE_only_30_40_50_is_pythagorean_triple_l2384_238453


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l2384_238465

theorem weekend_rain_probability (p_sat p_sun : ℝ) 
  (h_sat : p_sat = 0.6) 
  (h_sun : p_sun = 0.7) 
  (h_independent : True) -- We don't need to express independence in the statement
  : 1 - (1 - p_sat) * (1 - p_sun) = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l2384_238465


namespace NUMINAMATH_CALUDE_pizzeria_small_pizzas_sold_l2384_238429

/-- Calculates the number of small pizzas sold given the prices, total sales, and number of large pizzas sold. -/
def small_pizzas_sold (small_price large_price total_sales : ℕ) (large_pizzas_sold : ℕ) : ℕ :=
  (total_sales - large_price * large_pizzas_sold) / small_price

/-- Theorem stating that the number of small pizzas sold is 8 under the given conditions. -/
theorem pizzeria_small_pizzas_sold :
  small_pizzas_sold 2 8 40 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizzeria_small_pizzas_sold_l2384_238429


namespace NUMINAMATH_CALUDE_circle_radius_from_square_perimeter_area_equality_l2384_238441

theorem circle_radius_from_square_perimeter_area_equality (r : ℝ) : 
  (4 * (r * Real.sqrt 2)) = (Real.pi * r^2) → r = (4 * Real.sqrt 2) / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_square_perimeter_area_equality_l2384_238441


namespace NUMINAMATH_CALUDE_rotation_result_l2384_238480

-- Define the shapes
inductive Shape
  | Triangle
  | SmallCircle
  | Square
  | InvertedTriangle

-- Define the initial configuration
def initial_config : List Shape :=
  [Shape.Triangle, Shape.SmallCircle, Shape.Square, Shape.InvertedTriangle]

-- Define the rotation function
def rotate (angle : ℕ) (config : List Shape) : List Shape :=
  let shift := angle / 30  -- 150° = 5 * 30°
  config.rotateLeft shift

-- Theorem statement
theorem rotation_result :
  rotate 150 initial_config = [Shape.Square, Shape.InvertedTriangle, Shape.Triangle, Shape.SmallCircle] :=
by sorry

end NUMINAMATH_CALUDE_rotation_result_l2384_238480


namespace NUMINAMATH_CALUDE_range_of_a_l2384_238497

def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}

def B : Set ℝ := {x | x^2 - 5*x + 4 > 0}

theorem range_of_a (a : ℝ) : A a ∩ B = ∅ → a ∈ Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2384_238497


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l2384_238469

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∃ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) = 11) ∧
  (∀ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 11) :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l2384_238469


namespace NUMINAMATH_CALUDE_unique_root_implies_a_equals_negative_one_l2384_238499

theorem unique_root_implies_a_equals_negative_one (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 2 * a * x - 1 = 0) → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_implies_a_equals_negative_one_l2384_238499


namespace NUMINAMATH_CALUDE_original_number_proof_l2384_238420

theorem original_number_proof (x : ℝ) : 
  x - 25 = 0.75 * x + 25 → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2384_238420


namespace NUMINAMATH_CALUDE_positive_numbers_l2384_238444

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (sum_products_positive : a * b + b * c + c * a > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_l2384_238444


namespace NUMINAMATH_CALUDE_class_survey_l2384_238470

theorem class_survey (total_students : ℕ) (green_students : ℕ) (yellow_students : ℕ) (girls : ℕ) : 
  total_students = 30 →
  green_students = total_students / 2 →
  yellow_students = 9 →
  girls * 3 = (total_students - green_students - yellow_students) * 3 + girls →
  girls = 18 := by
sorry

end NUMINAMATH_CALUDE_class_survey_l2384_238470


namespace NUMINAMATH_CALUDE_square_root_sum_equals_eight_l2384_238457

theorem square_root_sum_equals_eight (x : ℝ) : 
  (Real.sqrt (49 - x^2) - Real.sqrt (25 - x^2) = 3) → 
  (Real.sqrt (49 - x^2) + Real.sqrt (25 - x^2) = 8) := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_eight_l2384_238457


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2384_238442

theorem polynomial_factorization (a : ℝ) : 
  -3*a + 12*a^2 - 12*a^3 = -3*a*(1 - 2*a)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2384_238442


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2384_238424

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 1, P x) ↔ (∀ x > 1, ¬ P x) :=
by
  sorry

theorem negation_of_proposition :
  (¬ ∃ x > 1, x^2 - 1 > 0) ↔ (∀ x > 1, x^2 - 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2384_238424


namespace NUMINAMATH_CALUDE_melanie_trout_count_l2384_238454

/-- Prove that Melanie caught 8 trouts given the conditions -/
theorem melanie_trout_count (tom_count : ℕ) (melanie_count : ℕ) 
  (h1 : tom_count = 16) 
  (h2 : tom_count = 2 * melanie_count) : 
  melanie_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_melanie_trout_count_l2384_238454


namespace NUMINAMATH_CALUDE_total_pages_read_three_weeks_l2384_238477

/-- Represents the reading statistics for a week --/
structure WeeklyReading where
  books : Nat
  pages_per_book : Nat
  magazines : Nat
  pages_per_magazine : Nat
  newspapers : Nat
  pages_per_newspaper : Nat

/-- Calculates the total pages read in a week --/
def total_pages_read (w : WeeklyReading) : Nat :=
  w.books * w.pages_per_book +
  w.magazines * w.pages_per_magazine +
  w.newspapers * w.pages_per_newspaper

/-- The reading statistics for the first week --/
def week1 : WeeklyReading :=
  { books := 5
    pages_per_book := 300
    magazines := 3
    pages_per_magazine := 120
    newspapers := 2
    pages_per_newspaper := 50 }

/-- The reading statistics for the second week --/
def week2 : WeeklyReading :=
  { books := 2 * week1.books
    pages_per_book := 350
    magazines := 4
    pages_per_magazine := 150
    newspapers := 1
    pages_per_newspaper := 60 }

/-- The reading statistics for the third week --/
def week3 : WeeklyReading :=
  { books := 3 * week1.books
    pages_per_book := 400
    magazines := 5
    pages_per_magazine := 125
    newspapers := 1
    pages_per_newspaper := 70 }

/-- Theorem: The total number of pages read over three weeks is 12815 --/
theorem total_pages_read_three_weeks :
  total_pages_read week1 + total_pages_read week2 + total_pages_read week3 = 12815 := by
  sorry


end NUMINAMATH_CALUDE_total_pages_read_three_weeks_l2384_238477


namespace NUMINAMATH_CALUDE_equilateral_triangle_exists_l2384_238421

-- Define the plane S parallel to x₁,₂ axis
structure Plane :=
  (s₁ : ℝ)
  (s₂ : ℝ)

-- Define a point in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the trace lines of the plane
def traceLine1 (S : Plane) : Set Point3D :=
  {p : Point3D | p.y = S.s₁}

def traceLine2 (S : Plane) : Set Point3D :=
  {p : Point3D | p.z = S.s₂}

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (A : Point3D)
  (B : Point3D)
  (C : Point3D)

-- State the theorem
theorem equilateral_triangle_exists (S : Plane) (A : Point3D) 
  (h : A.y = S.s₁ ∧ A.z = S.s₂) : 
  ∃ (t : EquilateralTriangle), 
    t.A = A ∧ 
    t.B ∈ traceLine1 S ∧ 
    t.C ∈ traceLine2 S ∧
    (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 + (t.A.z - t.B.z)^2 = 
    (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 + (t.B.z - t.C.z)^2 ∧
    (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 + (t.A.z - t.B.z)^2 = 
    (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 + (t.A.z - t.C.z)^2 := by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_exists_l2384_238421


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l2384_238451

/-- Given two parabolas that intersect at four points, prove that these points lie on a circle with radius squared equal to 5/2 -/
theorem intersection_points_on_circle (x y : ℝ) : 
  (y = (x - 2)^2) ∧ (x - 3 = (y + 1)^2) →
  ∃ (center : ℝ × ℝ), 
    (x - center.1)^2 + (y - center.2)^2 = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l2384_238451


namespace NUMINAMATH_CALUDE_correct_fraction_l2384_238419

/-- The number of quarters Roger has -/
def total_quarters : ℕ := 22

/-- The number of states that joined the union during 1800-1809 -/
def states_1800_1809 : ℕ := 5

/-- The fraction of quarters representing states that joined during 1800-1809 -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_quarters

theorem correct_fraction :
  fraction_1800_1809 = 5 / 22 :=
by sorry

end NUMINAMATH_CALUDE_correct_fraction_l2384_238419


namespace NUMINAMATH_CALUDE_no_solution_floor_plus_x_l2384_238492

theorem no_solution_floor_plus_x :
  ¬ ∃ x : ℝ, ⌊x⌋ + x = 15.3 := by sorry

end NUMINAMATH_CALUDE_no_solution_floor_plus_x_l2384_238492


namespace NUMINAMATH_CALUDE_candy_distribution_l2384_238483

theorem candy_distribution (x : ℚ) 
  (h1 : 3 * x = mia_candies)
  (h2 : 4 * mia_candies = noah_candies)
  (h3 : 6 * noah_candies = olivia_candies)
  (h4 : x + mia_candies + noah_candies + olivia_candies = 468) :
  x = 117 / 22 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2384_238483


namespace NUMINAMATH_CALUDE_samuel_money_left_l2384_238466

/-- Calculates the amount Samuel has left after receiving a share of the total amount and spending on drinks -/
def samuel_remaining_money (total : ℝ) (share_fraction : ℝ) (spend_fraction : ℝ) : ℝ :=
  total * share_fraction - total * spend_fraction

/-- Theorem stating that given the conditions in the problem, Samuel has $132 left -/
theorem samuel_money_left :
  let total : ℝ := 240
  let share_fraction : ℝ := 3/4
  let spend_fraction : ℝ := 1/5
  samuel_remaining_money total share_fraction spend_fraction = 132 := by
  sorry


end NUMINAMATH_CALUDE_samuel_money_left_l2384_238466


namespace NUMINAMATH_CALUDE_total_distance_flown_l2384_238409

theorem total_distance_flown (trip_distance : ℝ) (num_trips : ℝ) 
  (h1 : trip_distance = 256.0) 
  (h2 : num_trips = 32.0) : 
  trip_distance * num_trips = 8192.0 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_flown_l2384_238409


namespace NUMINAMATH_CALUDE_luxury_car_price_l2384_238449

def initial_price : ℝ := 80000

def discounts : List ℝ := [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price : ℝ := discounts.foldl apply_discount initial_price

theorem luxury_car_price : final_price = 24418.80 := by
  sorry

end NUMINAMATH_CALUDE_luxury_car_price_l2384_238449


namespace NUMINAMATH_CALUDE_cone_height_l2384_238416

/-- Given a cone with slant height 13 cm and lateral area 65π cm², prove its height is 12 cm -/
theorem cone_height (s : ℝ) (l : ℝ) (h : ℝ) : 
  s = 13 →  -- slant height
  l = 65 * Real.pi →  -- lateral area
  l = Real.pi * s * (s^2 - h^2).sqrt →  -- formula for lateral area
  h = 12 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l2384_238416


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_equivalence_negation_equivalence_l2384_238427

-- Proposition 1
theorem sufficient_not_necessary :
  (∃ x : ℝ, x ≠ 1 ∧ x^2 - 3*x + 2 = 0) ∧
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) := by sorry

-- Proposition 2
theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) := by sorry

-- Proposition 3
theorem negation_equivalence :
  (¬∃ x : ℝ, x > 0 ∧ x^2 + x + 1 < 0) ↔
  (∀ x : ℝ, x > 0 → x^2 + x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_equivalence_negation_equivalence_l2384_238427


namespace NUMINAMATH_CALUDE_prob_sum_three_is_half_l2384_238411

/-- A fair coin toss outcome -/
inductive CoinToss
  | heads
  | tails

/-- The numeric value associated with a coin toss -/
def coinValue (t : CoinToss) : ℕ :=
  match t with
  | CoinToss.heads => 1
  | CoinToss.tails => 2

/-- The sample space of two coin tosses -/
def sampleSpace : List (CoinToss × CoinToss) :=
  [(CoinToss.heads, CoinToss.heads),
   (CoinToss.heads, CoinToss.tails),
   (CoinToss.tails, CoinToss.heads),
   (CoinToss.tails, CoinToss.tails)]

/-- The event where the sum of two coin tosses is 3 -/
def sumThreeEvent (t : CoinToss × CoinToss) : Bool :=
  coinValue t.1 + coinValue t.2 = 3

/-- Theorem: The probability of obtaining a sum of 3 when tossing a fair coin twice is 1/2 -/
theorem prob_sum_three_is_half :
  (sampleSpace.filter sumThreeEvent).length / sampleSpace.length = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_three_is_half_l2384_238411


namespace NUMINAMATH_CALUDE_marble_ratio_l2384_238432

theorem marble_ratio (b y r : ℚ) 
  (h1 : b / y = 1.2) 
  (h2 : y / r = 5 / 6) 
  (h3 : b > 0) 
  (h4 : y > 0) 
  (h5 : r > 0) : 
  b / r = 1 := by
sorry

end NUMINAMATH_CALUDE_marble_ratio_l2384_238432


namespace NUMINAMATH_CALUDE_base_conversion_2345_to_base_7_l2384_238406

theorem base_conversion_2345_to_base_7 :
  (2345 : ℕ) = 6 * (7 : ℕ)^3 + 5 * (7 : ℕ)^2 + 6 * (7 : ℕ)^1 + 0 * (7 : ℕ)^0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_2345_to_base_7_l2384_238406


namespace NUMINAMATH_CALUDE_restaurant_meal_cost_l2384_238426

theorem restaurant_meal_cost 
  (total_people : ℕ) 
  (num_kids : ℕ) 
  (total_cost : ℕ) 
  (h1 : total_people = 11) 
  (h2 : num_kids = 2) 
  (h3 : total_cost = 72) :
  (total_cost : ℚ) / (total_people - num_kids : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_meal_cost_l2384_238426


namespace NUMINAMATH_CALUDE_monotone_increasing_implies_m_leq_half_m_leq_half_not_implies_monotone_increasing_l2384_238413

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x

-- Define the property of f being monotonically increasing on (0, +∞)
def is_monotone_increasing_on_positive (m : ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f m x < f m y

-- Theorem stating that if f is monotonically increasing on (0, +∞), then m ≤ 1/2
theorem monotone_increasing_implies_m_leq_half (m : ℝ) :
  is_monotone_increasing_on_positive m → m ≤ 1/2 :=
sorry

-- Theorem stating that m ≤ 1/2 does not necessarily imply f is monotonically increasing on (0, +∞)
theorem m_leq_half_not_implies_monotone_increasing :
  ∃ m, m ≤ 1/2 ∧ ¬is_monotone_increasing_on_positive m :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_implies_m_leq_half_m_leq_half_not_implies_monotone_increasing_l2384_238413


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2384_238473

/-- 
Given three numbers forming an arithmetic sequence where the first number is 3,
and when the middle term is reduced by 6 it forms a geometric sequence,
prove that the third number (the unknown number) is either 3 or 27.
-/
theorem arithmetic_geometric_sequence (a b : ℝ) : 
  (2 * a = 3 + b) →  -- arithmetic sequence condition
  ((a - 6)^2 = 3 * b) →  -- geometric sequence condition after reduction
  (b = 3 ∨ b = 27) :=  -- conclusion: the unknown number is either 3 or 27
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2384_238473


namespace NUMINAMATH_CALUDE_mod_power_thirteen_six_eleven_l2384_238404

theorem mod_power_thirteen_six_eleven : 13^6 % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_mod_power_thirteen_six_eleven_l2384_238404


namespace NUMINAMATH_CALUDE_problem_solution_l2384_238414

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.cos y = 3005)
  (h2 : x + 3005 * Real.sin y = 3004)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 3004 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2384_238414


namespace NUMINAMATH_CALUDE_grape_juice_concentration_l2384_238422

/-- Given an initial mixture and added grape juice, calculate the final grape juice concentration -/
theorem grape_juice_concentration 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (added_juice : ℝ) 
  (h1 : initial_volume = 40)
  (h2 : initial_concentration = 0.1)
  (h3 : added_juice = 10) : 
  (initial_volume * initial_concentration + added_juice) / (initial_volume + added_juice) = 0.28 := by
sorry

end NUMINAMATH_CALUDE_grape_juice_concentration_l2384_238422


namespace NUMINAMATH_CALUDE_complex_number_properties_l2384_238487

/-- Given a complex number z where z + 1/z is real, this theorem proves:
    1. The value of z that minimizes |z + 2 - i|
    2. The minimum value of |z + 2 - i|
    3. u = (1 - z) / (1 + z) is purely imaginary -/
theorem complex_number_properties (z : ℂ) 
    (h : (z + z⁻¹).im = 0) : 
    ∃ (min_z : ℂ) (min_val : ℝ),
    (min_z = -2 * Real.sqrt 5 / 5 + (Real.sqrt 5 / 5) * Complex.I) ∧
    (min_val = Real.sqrt 5 - 1) ∧
    (∀ w : ℂ, Complex.abs (w + 2 - Complex.I) ≥ min_val) ∧
    (Complex.abs (min_z + 2 - Complex.I) = min_val) ∧
    ((1 - z) / (1 + z)).re = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2384_238487


namespace NUMINAMATH_CALUDE_new_person_weight_l2384_238459

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 10 →
  replaced_weight = 50 →
  avg_increase = 2.5 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * avg_increase + replaced_weight ∧
    new_weight = 75 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2384_238459


namespace NUMINAMATH_CALUDE_reading_ratio_two_to_three_nights_l2384_238436

/-- Represents the number of pages read on each night -/
structure ReadingPattern where
  threeNightsAgo : ℕ
  twoNightsAgo : ℕ
  lastNight : ℕ
  tonight : ℕ

/-- Theorem stating the ratio of pages read two nights ago to three nights ago -/
theorem reading_ratio_two_to_three_nights (r : ReadingPattern) : 
  r.threeNightsAgo = 15 →
  r.lastNight = r.twoNightsAgo + 5 →
  r.tonight = 20 →
  r.threeNightsAgo + r.twoNightsAgo + r.lastNight + r.tonight = 100 →
  r.twoNightsAgo / r.threeNightsAgo = 2 := by
  sorry

#check reading_ratio_two_to_three_nights

end NUMINAMATH_CALUDE_reading_ratio_two_to_three_nights_l2384_238436


namespace NUMINAMATH_CALUDE_fourth_day_temperature_l2384_238440

theorem fourth_day_temperature
  (temp1 temp2 temp3 : ℤ)
  (avg_temp : ℚ)
  (h1 : temp1 = -36)
  (h2 : temp2 = 13)
  (h3 : temp3 = -15)
  (h4 : avg_temp = -12)
  (h5 : (temp1 + temp2 + temp3 + temp4 : ℚ) / 4 = avg_temp) :
  temp4 = -10 :=
sorry

end NUMINAMATH_CALUDE_fourth_day_temperature_l2384_238440


namespace NUMINAMATH_CALUDE_polygon_with_44_diagonals_has_11_sides_l2384_238463

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 44 diagonals has 11 sides -/
theorem polygon_with_44_diagonals_has_11_sides :
  ∃ (n : ℕ), n > 2 ∧ diagonals n = 44 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_44_diagonals_has_11_sides_l2384_238463


namespace NUMINAMATH_CALUDE_cubic_two_roots_l2384_238415

/-- The cubic function we're analyzing -/
def f (d : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + d

/-- A function has exactly two roots -/
def has_exactly_two_roots (g : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧ g a = 0 ∧ g b = 0 ∧ ∀ x, g x = 0 → x = a ∨ x = b

/-- If f(x) = x^3 - 3x + d has exactly two roots, then d = 2 or d = -2 -/
theorem cubic_two_roots (d : ℝ) : has_exactly_two_roots (f d) → d = 2 ∨ d = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_two_roots_l2384_238415


namespace NUMINAMATH_CALUDE_inequality_proof_l2384_238443

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (1 + a^2) + Real.sqrt (1 + b^2) ≥ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2384_238443


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l2384_238439

def is_valid_cryptarithm (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ 
  ∃ (a b c d : ℕ), n * n = 1000 * c + 100 * d + n ∧ 
  c ≠ 0

theorem cryptarithm_solution :
  ∃! (S : Set ℕ), 
    (∀ n ∈ S, is_valid_cryptarithm n ∧ Odd n) ∧ 
    (∀ n, is_valid_cryptarithm n → Odd n → n ∈ S) ∧
    (∃ m, m ∈ S) ∧
    (∀ T : Set ℕ, (∀ n ∈ T, is_valid_cryptarithm n ∧ Even n) → T.Nonempty → ¬(∃! x, x ∈ T)) :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l2384_238439


namespace NUMINAMATH_CALUDE_multiplication_equations_l2384_238482

theorem multiplication_equations : 
  (30 * 30 = 900) ∧
  (30 * 40 = 1200) ∧
  (40 * 70 = 2800) ∧
  (50 * 70 = 3500) ∧
  (60 * 70 = 4200) ∧
  (4 * 90 = 360) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equations_l2384_238482


namespace NUMINAMATH_CALUDE_triangle_shape_l2384_238452

theorem triangle_shape (a b : ℝ) (A B : ℝ) :
  0 < A → A < π →
  0 < B → B < π →
  a^2 * Real.tan B = b^2 * Real.tan A →
  (A = B ∨ A + B = π / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_l2384_238452


namespace NUMINAMATH_CALUDE_product_equals_eight_l2384_238407

theorem product_equals_eight :
  (1 + 1/1) * (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_eight_l2384_238407


namespace NUMINAMATH_CALUDE_equality_from_fraction_equality_l2384_238461

theorem equality_from_fraction_equality (a b c d : ℝ) :
  (a + b) / (c + d) = (b + c) / (a + d) ∧ 
  (a + b) / (c + d) ≠ -1 →
  a = c :=
by sorry

end NUMINAMATH_CALUDE_equality_from_fraction_equality_l2384_238461


namespace NUMINAMATH_CALUDE_quadratic_function_monotonicity_l2384_238479

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem quadratic_function_monotonicity :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → f x₁ > f x₂) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_monotonicity_l2384_238479


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2384_238496

theorem sum_of_xyz (x y z : ℕ+) 
  (h1 : (x * y * z : ℕ) = 240)
  (h2 : (x * y + z : ℕ) = 46)
  (h3 : (x + y * z : ℕ) = 64) :
  (x + y + z : ℕ) = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2384_238496


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2384_238431

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) / z = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2384_238431


namespace NUMINAMATH_CALUDE_solution_set_equals_interval_l2384_238412

-- Define the ⊕ operation
def circle_plus (a b : ℝ) : ℝ := a * b + 2 * a + b

-- Define the set of x satisfying the inequality
def solution_set : Set ℝ := {x | circle_plus x (x - 2) < 0}

-- Theorem statement
theorem solution_set_equals_interval :
  solution_set = Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_interval_l2384_238412


namespace NUMINAMATH_CALUDE_pair_and_triplet_count_two_pairs_count_l2384_238425

/- Define the structure of a deck of cards -/
def numSuits : Nat := 4
def numRanks : Nat := 13
def deckSize : Nat := numSuits * numRanks

/- Define the combination function -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/- Theorem for part 1 -/
theorem pair_and_triplet_count :
  choose numRanks 1 * choose numSuits 2 * choose (numRanks - 1) 1 * choose numSuits 3 = 3744 :=
by sorry

/- Theorem for part 2 -/
theorem two_pairs_count :
  choose numRanks 2 * (choose numSuits 2)^2 * choose (numRanks - 2) 1 * choose numSuits 1 = 123552 :=
by sorry

end NUMINAMATH_CALUDE_pair_and_triplet_count_two_pairs_count_l2384_238425


namespace NUMINAMATH_CALUDE_tangent_slope_of_circle_l2384_238490

/-- Given a circle with center (1,3) and a point (4,7) on the circle,
    the slope of the line tangent to the circle at (4,7) is -3/4 -/
theorem tangent_slope_of_circle (center : ℝ × ℝ) (point : ℝ × ℝ) :
  center = (1, 3) →
  point = (4, 7) →
  (let slope_tangent := -(((point.2 - center.2) / (point.1 - center.1))⁻¹)
   slope_tangent = -3/4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_of_circle_l2384_238490


namespace NUMINAMATH_CALUDE_unique_sum_value_l2384_238434

theorem unique_sum_value (n m : ℤ) 
  (h1 : 3*n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3*m - 2*n < 46) :
  2*n + m = 36 := by
  sorry

end NUMINAMATH_CALUDE_unique_sum_value_l2384_238434


namespace NUMINAMATH_CALUDE_smallest_multiple_eight_is_solution_eight_is_smallest_l2384_238491

theorem smallest_multiple (x : ℕ+) : (450 * x : ℕ) % 720 = 0 → x ≥ 8 := by
  sorry

theorem eight_is_solution : (450 * 8 : ℕ) % 720 = 0 := by
  sorry

theorem eight_is_smallest : ∀ (x : ℕ+), (450 * x : ℕ) % 720 = 0 → x ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_eight_is_solution_eight_is_smallest_l2384_238491


namespace NUMINAMATH_CALUDE_expansion_terms_count_l2384_238400

theorem expansion_terms_count :
  let n : ℕ := 10
  let num_terms : ℕ := Nat.choose n 5
  num_terms = 252 := by sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l2384_238400


namespace NUMINAMATH_CALUDE_escalator_length_is_160_l2384_238437

/-- The length of an escalator given its speed, a person's walking speed on it, and the time taken to cover the entire length. -/
def escalatorLength (escalatorSpeed personSpeed : ℝ) (timeTaken : ℝ) : ℝ :=
  (escalatorSpeed + personSpeed) * timeTaken

/-- Theorem stating that the length of the escalator is 160 feet under the given conditions. -/
theorem escalator_length_is_160 :
  escalatorLength 12 8 8 = 160 := by
  sorry

end NUMINAMATH_CALUDE_escalator_length_is_160_l2384_238437


namespace NUMINAMATH_CALUDE_mass_of_compound_l2384_238481

/-- Molar mass of potassium in g/mol -/
def molar_mass_K : ℝ := 39.10

/-- Molar mass of aluminum in g/mol -/
def molar_mass_Al : ℝ := 26.98

/-- Molar mass of sulfur in g/mol -/
def molar_mass_S : ℝ := 32.07

/-- Molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Molar mass of hydrogen in g/mol -/
def molar_mass_H : ℝ := 1.01

/-- Number of moles of the compound -/
def num_moles : ℝ := 15

/-- Molar mass of potassium aluminum sulfate dodecahydrate (KAl(SO4)2·12H2O) in g/mol -/
def molar_mass_compound : ℝ := 
  molar_mass_K + molar_mass_Al + 2 * molar_mass_S + 32 * molar_mass_O + 24 * molar_mass_H

/-- Mass of the compound in grams -/
def mass_compound : ℝ := num_moles * molar_mass_compound

theorem mass_of_compound : mass_compound = 9996.9 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_compound_l2384_238481


namespace NUMINAMATH_CALUDE_range_m_for_f_negative_solution_inequality_range_m_for_f_geq_quadratic_l2384_238478

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + (m - 1)

/-- The range of m for which f(x) < 0 has solution set ℝ -/
theorem range_m_for_f_negative (m : ℝ) : 
  (∀ x, f m x < 0) ↔ m < -5/3 := by sorry

/-- The solution to f(x) ≥ 3x + m - 2 when m < 0 -/
theorem solution_inequality (m : ℝ) (hm : m < 0) :
  (∀ x, f m x ≥ 3*x + m - 2) ↔ 
    ((-1 < m ∧ (∀ x, x ≤ 1 ∨ x ≥ 1/(m+1))) ∨
     (m = -1 ∧ (∀ x, x ≤ 1)) ∨
     (m < -1 ∧ (∀ x, 1/(m+1) ≤ x ∧ x ≤ 1))) := by sorry

/-- The range of m for which f(x) ≥ x^2 + 2x holds for all x ∈ [0,2] -/
theorem range_m_for_f_geq_quadratic (m : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f m x ≥ x^2 + 2*x) ↔ m ≥ 2*Real.sqrt 3/3 + 1 := by sorry

end NUMINAMATH_CALUDE_range_m_for_f_negative_solution_inequality_range_m_for_f_geq_quadratic_l2384_238478


namespace NUMINAMATH_CALUDE_max_parrots_in_zoo_l2384_238410

/-- Represents the number of parrots of each color in the zoo -/
structure ParrotCount where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- The conditions of the zoo problem -/
def ZooConditions (p : ParrotCount) : Prop :=
  p.red > 0 ∧ p.yellow > 0 ∧ p.green > 0 ∧
  ∀ (s : Finset ℕ), s.card = 10 → (∃ i ∈ s, i < p.red) ∧
  ∀ (s : Finset ℕ), s.card = 12 → (∃ i ∈ s, i < p.yellow)

/-- The theorem stating the maximum number of parrots in the zoo -/
theorem max_parrots_in_zoo :
  ∃ (max : ℕ), max = 19 ∧
  (∃ (p : ParrotCount), ZooConditions p ∧ p.red + p.yellow + p.green = max) ∧
  ∀ (p : ParrotCount), ZooConditions p → p.red + p.yellow + p.green ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_parrots_in_zoo_l2384_238410


namespace NUMINAMATH_CALUDE_vector_proof_l2384_238445

/-- Given two vectors a and b in ℝ², prove that if they form an angle of 180° and b has a specific magnitude, then b equals a specific value. -/
theorem vector_proof (a b : ℝ × ℝ) : 
  a = (2, -1) → 
  (a.1 * b.1 + a.2 * b.2 = -Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2))) → 
  (b.1^2 + b.2^2 = (3 * Real.sqrt 5)^2) → 
  b = (-6, 3) := by
sorry

end NUMINAMATH_CALUDE_vector_proof_l2384_238445


namespace NUMINAMATH_CALUDE_snack_distribution_probability_l2384_238403

/-- The number of students and snack types -/
def n : ℕ := 4

/-- The total number of snacks -/
def total_snacks : ℕ := n * n

/-- The number of ways to distribute snacks to one student -/
def ways_per_student (k : ℕ) : ℕ := n^n

/-- The number of ways to choose snacks for one student from remaining snacks -/
def choose_from_remaining (k : ℕ) : ℕ := Nat.choose (total_snacks - (k - 1) * n) n

/-- The probability of correct distribution for the k-th student -/
def prob_for_student (k : ℕ) : ℚ := ways_per_student k / choose_from_remaining k

/-- The probability that each student gets one of each type of snack -/
def prob_correct_distribution : ℚ :=
  prob_for_student 1 * prob_for_student 2 * prob_for_student 3

theorem snack_distribution_probability :
  prob_correct_distribution = 64 / 1225 :=
sorry

end NUMINAMATH_CALUDE_snack_distribution_probability_l2384_238403


namespace NUMINAMATH_CALUDE_pair_one_six_least_restricted_l2384_238455

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a license plate ending pair -/
structure LicensePlatePair :=
  (first : Nat)
  (second : Nat)

/-- The restriction schedule for each license plate ending pair -/
def restrictionSchedule : LicensePlatePair → List DayOfWeek
  | ⟨1, 6⟩ => [DayOfWeek.Monday, DayOfWeek.Tuesday]
  | ⟨2, 7⟩ => [DayOfWeek.Tuesday, DayOfWeek.Wednesday]
  | ⟨3, 8⟩ => [DayOfWeek.Wednesday, DayOfWeek.Thursday]
  | ⟨4, 9⟩ => [DayOfWeek.Thursday, DayOfWeek.Friday]
  | ⟨5, 0⟩ => [DayOfWeek.Friday, DayOfWeek.Monday]
  | _ => []

/-- Calculate the number of restricted days for a given license plate pair in January 2014 -/
def restrictedDays (pair : LicensePlatePair) : Nat :=
  sorry

/-- All possible license plate ending pairs -/
def allPairs : List LicensePlatePair :=
  [⟨1, 6⟩, ⟨2, 7⟩, ⟨3, 8⟩, ⟨4, 9⟩, ⟨5, 0⟩]

/-- Theorem: The license plate pair (1,6) has the fewest restricted days in January 2014 -/
theorem pair_one_six_least_restricted :
  ∀ pair ∈ allPairs, restrictedDays ⟨1, 6⟩ ≤ restrictedDays pair := by
  sorry

end NUMINAMATH_CALUDE_pair_one_six_least_restricted_l2384_238455


namespace NUMINAMATH_CALUDE_jenny_max_earnings_l2384_238448

def neighborhood_A_homes : ℕ := 10
def neighborhood_A_boxes_per_home : ℕ := 2
def neighborhood_B_homes : ℕ := 5
def neighborhood_B_boxes_per_home : ℕ := 5
def price_per_box : ℕ := 2

def total_boxes_A : ℕ := neighborhood_A_homes * neighborhood_A_boxes_per_home
def total_boxes_B : ℕ := neighborhood_B_homes * neighborhood_B_boxes_per_home

def max_earnings : ℕ := max total_boxes_A total_boxes_B * price_per_box

theorem jenny_max_earnings :
  max_earnings = 50 := by
  sorry

end NUMINAMATH_CALUDE_jenny_max_earnings_l2384_238448


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2384_238484

-- Problem 1
theorem problem_1 : 24 - |(-2)| + (-16) - 8 = -2 := by sorry

-- Problem 2
theorem problem_2 : (-2) * (3/2) / (-3/4) * 4 = 4 := by sorry

-- Problem 3
theorem problem_3 : (-1)^2016 - (1 - 0.5) / 3 * (2 - (-3)^2) = 1/6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2384_238484


namespace NUMINAMATH_CALUDE_bauble_painting_friends_l2384_238458

/-- The number of friends needed to complete the bauble painting task -/
def friends_needed (total_baubles : ℕ) (total_colors : ℕ) (first_group_colors : ℕ) 
  (second_group_colors : ℕ) (baubles_per_hour : ℕ) (available_hours : ℕ) : ℕ :=
  let first_group_baubles_per_color := total_baubles / (first_group_colors + 2 * second_group_colors)
  let second_group_baubles_per_color := 2 * first_group_baubles_per_color
  let baubles_per_hour_needed := total_baubles / available_hours
  baubles_per_hour_needed / baubles_per_hour

theorem bauble_painting_friends (total_baubles : ℕ) (total_colors : ℕ) (first_group_colors : ℕ) 
  (second_group_colors : ℕ) (baubles_per_hour : ℕ) (available_hours : ℕ) 
  (h1 : total_baubles = 1000)
  (h2 : total_colors = 20)
  (h3 : first_group_colors = 15)
  (h4 : second_group_colors = 5)
  (h5 : baubles_per_hour = 10)
  (h6 : available_hours = 50)
  (h7 : first_group_colors + second_group_colors = total_colors) :
  friends_needed total_baubles total_colors first_group_colors second_group_colors baubles_per_hour available_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_bauble_painting_friends_l2384_238458


namespace NUMINAMATH_CALUDE_double_root_condition_l2384_238467

/-- 
For a quadratic equation ax^2 + bx + c = 0, if one root is double the other, 
then 2b^2 = 9ac.
-/
theorem double_root_condition (a b c : ℝ) (x₁ x₂ : ℝ) : 
  a ≠ 0 → 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) → 
  x₂ = 2 * x₁ → 
  2 * b^2 = 9 * a * c := by
sorry

end NUMINAMATH_CALUDE_double_root_condition_l2384_238467


namespace NUMINAMATH_CALUDE_unknown_number_is_six_l2384_238435

theorem unknown_number_is_six : ∃ x : ℚ, (2 / 3) * x + 6 = 10 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_is_six_l2384_238435


namespace NUMINAMATH_CALUDE_canteen_banana_units_l2384_238456

/-- Represents the number of bananas in a unit -/
def bananas_per_unit (daily_units : ℕ) (total_bananas : ℕ) (weeks : ℕ) : ℕ :=
  (total_bananas / (weeks * 7)) / daily_units

/-- Theorem stating that given the conditions, each unit consists of 12 bananas -/
theorem canteen_banana_units :
  bananas_per_unit 13 9828 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_canteen_banana_units_l2384_238456


namespace NUMINAMATH_CALUDE_quadratic_sum_l2384_238428

/-- A quadratic function passing through (-3,0) and (3,0) with maximum value 36 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x ≤ 36) ∧ 
  QuadraticFunction a b c (-3) = 0 ∧
  QuadraticFunction a b c 3 = 0 →
  a + b + c = 32 := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2384_238428


namespace NUMINAMATH_CALUDE_min_intersection_points_l2384_238475

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of circles -/
structure CircleConfiguration where
  n : ℕ+
  circles : Fin (4 * n) → Circle
  same_radius : ∀ i j, (circles i).radius = (circles j).radius
  no_tangent : ∀ i j, i ≠ j → (circles i).center ≠ (circles j).center ∨ 
               dist (circles i).center (circles j).center ≠ (circles i).radius + (circles j).radius
  intersect_at_least_three : ∀ i, ∃ j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
                             dist (circles i).center (circles j).center < (circles i).radius + (circles j).radius ∧
                             dist (circles i).center (circles k).center < (circles i).radius + (circles k).radius ∧
                             dist (circles i).center (circles l).center < (circles i).radius + (circles l).radius

/-- The number of intersection points in a circle configuration -/
def num_intersection_points (config : CircleConfiguration) : ℕ :=
  sorry

/-- The main theorem: the minimum number of intersection points is 4n -/
theorem min_intersection_points (config : CircleConfiguration) :
  num_intersection_points config ≥ 4 * config.n :=
sorry

end NUMINAMATH_CALUDE_min_intersection_points_l2384_238475


namespace NUMINAMATH_CALUDE_gym_class_group_sizes_l2384_238474

/-- Given a gym class with two groups of students, prove that if the total number of students is 71 and one group has 37 students, then the other group must have 34 students. -/
theorem gym_class_group_sizes (total_students : ℕ) (group1_size : ℕ) (group2_size : ℕ) 
  (h1 : total_students = 71)
  (h2 : group2_size = 37)
  (h3 : total_students = group1_size + group2_size) :
  group1_size = 34 := by
  sorry

end NUMINAMATH_CALUDE_gym_class_group_sizes_l2384_238474


namespace NUMINAMATH_CALUDE_number_problem_l2384_238423

theorem number_problem (N : ℕ) (h1 : ∃ k : ℕ, N = 5 * k) (h2 : N / 5 = 25) :
  (N - 17) / 6 = 18 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l2384_238423


namespace NUMINAMATH_CALUDE_jeans_purchase_savings_l2384_238402

/-- Calculates the total savings on a purchase with multiple discounts and a rebate --/
theorem jeans_purchase_savings 
  (original_price : ℝ)
  (sale_discount_percent : ℝ)
  (coupon_discount : ℝ)
  (credit_card_discount_percent : ℝ)
  (voucher_discount_percent : ℝ)
  (rebate : ℝ)
  (sales_tax_percent : ℝ)
  (h1 : original_price = 200)
  (h2 : sale_discount_percent = 30)
  (h3 : coupon_discount = 15)
  (h4 : credit_card_discount_percent = 15)
  (h5 : voucher_discount_percent = 10)
  (h6 : rebate = 20)
  (h7 : sales_tax_percent = 8.25) :
  ∃ (savings : ℝ), abs (savings - 116.49) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_jeans_purchase_savings_l2384_238402


namespace NUMINAMATH_CALUDE_fibonacci_sum_l2384_238408

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of F_n / 10^n from n = 0 to infinity -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / (10 : ℝ) ^ n

/-- Theorem: The sum of F_n / 10^n from n = 0 to infinity equals 10/89 -/
theorem fibonacci_sum : fibSum = 10 / 89 := by sorry

end NUMINAMATH_CALUDE_fibonacci_sum_l2384_238408


namespace NUMINAMATH_CALUDE_inequality_proof_l2384_238405

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a + b + c = 1) : 
  (a^4 + b^4)/(a^6 + b^6) + (b^4 + c^4)/(b^6 + c^6) + (c^4 + a^4)/(c^6 + a^6) ≤ 1/(a*b*c) := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l2384_238405


namespace NUMINAMATH_CALUDE_farmer_apples_l2384_238468

theorem farmer_apples (initial_apples given_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : given_apples = 88) :
  initial_apples - given_apples = 39 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l2384_238468


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l2384_238471

theorem min_value_expression (a b c : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 4) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (4/c - 1)^2 ≥ 12 - 8 * Real.sqrt 2 :=
by sorry

theorem min_value_attainable :
  ∃ a b c : ℝ, 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 4 ∧
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (4/c - 1)^2 = 12 - 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l2384_238471


namespace NUMINAMATH_CALUDE_min_cost_achieved_funds_sufficient_l2384_238498

def total_desks : ℕ := 36
def cost_per_desk : ℕ := 200
def shipping_fee : ℕ := 40
def available_funds : ℕ := 480

def f (x : ℕ) : ℚ := 144 / x + 4 * x

theorem min_cost_achieved (x : ℕ) (hx : 0 < x ∧ x ≤ 36) : 
  f x ≥ 48 ∧ (f x = 48 ↔ x = 6) := by
  sorry

theorem funds_sufficient : ∃ x : ℕ, 0 < x ∧ x ≤ 36 ∧ f x ≤ available_funds := by
  sorry

end NUMINAMATH_CALUDE_min_cost_achieved_funds_sufficient_l2384_238498


namespace NUMINAMATH_CALUDE_exists_increasing_perfect_squares_sequence_l2384_238450

theorem exists_increasing_perfect_squares_sequence : 
  ∃ (a : ℕ → ℕ), 
    (∀ k : ℕ, k > 0 → ∃ n : ℕ, a k = n ^ 2) ∧ 
    (∀ k : ℕ, k > 0 → a k < a (k + 1)) ∧
    (∀ k : ℕ, k > 0 → (13 ^ k) ∣ (a k + 1)) := by
  sorry

end NUMINAMATH_CALUDE_exists_increasing_perfect_squares_sequence_l2384_238450


namespace NUMINAMATH_CALUDE_points_collinear_if_linear_combination_l2384_238446

/-- Four points in space are collinear if one is a linear combination of the others -/
theorem points_collinear_if_linear_combination (P A B C : EuclideanSpace ℝ (Fin 3)) :
  (C - P) = (1/4 : ℝ) • (A - P) + (3/4 : ℝ) • (B - P) →
  ∃ (t : ℝ), C - A = t • (B - A) :=
by sorry

end NUMINAMATH_CALUDE_points_collinear_if_linear_combination_l2384_238446


namespace NUMINAMATH_CALUDE_public_swimming_pool_attendance_l2384_238493

/-- Proves the total number of people who used the public swimming pool -/
theorem public_swimming_pool_attendance 
  (child_price : ℚ) 
  (adult_price : ℚ) 
  (total_receipts : ℚ) 
  (num_children : ℕ) : 
  child_price = 3/2 →
  adult_price = 9/4 →
  total_receipts = 1422 →
  num_children = 388 →
  ∃ (num_adults : ℕ), 
    num_adults * adult_price + num_children * child_price = total_receipts ∧
    num_adults + num_children = 761 := by
  sorry

end NUMINAMATH_CALUDE_public_swimming_pool_attendance_l2384_238493


namespace NUMINAMATH_CALUDE_burgers_remaining_l2384_238494

theorem burgers_remaining (total_burgers : ℕ) (slices_per_burger : ℕ) 
  (friend1 friend2 friend3 friend4 friend5 : ℚ) : 
  total_burgers = 5 →
  slices_per_burger = 8 →
  friend1 = 3 / 8 →
  friend2 = 8 / 8 →
  friend3 = 5 / 8 →
  friend4 = 11 / 8 →
  friend5 = 6 / 8 →
  (total_burgers * slices_per_burger : ℚ) - (friend1 + friend2 + friend3 + friend4 + friend5) * slices_per_burger = 7 := by
  sorry

end NUMINAMATH_CALUDE_burgers_remaining_l2384_238494


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l2384_238495

theorem arctan_equation_solution (y : ℝ) :
  2 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/3 →
  y = 1005/97 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l2384_238495


namespace NUMINAMATH_CALUDE_expression_simplification_and_value_l2384_238464

theorem expression_simplification_and_value (a : ℤ) 
  (h1 : 0 < a) (h2 : a < Int.floor (Real.sqrt 5)) : 
  (((a^2 - 1) / (a^2 + 2*a)) / ((a - 1) / a) - a / (a + 2) : ℚ) = 1 / (a + 2) ∧
  (((1^2 - 1) / (1^2 + 2*1)) / ((1 - 1) / 1) - 1 / (1 + 2) : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_value_l2384_238464


namespace NUMINAMATH_CALUDE_pegboard_empty_holes_l2384_238472

/-- Represents a square pegboard -/
structure Pegboard :=
  (size : ℕ)

/-- Calculates the total number of holes on the pegboard -/
def total_holes (p : Pegboard) : ℕ := (p.size + 1) ^ 2

/-- Calculates the number of holes with pegs (on diagonals) -/
def holes_with_pegs (p : Pegboard) : ℕ := 2 * (p.size + 1) - 1

/-- Calculates the number of empty holes on the pegboard -/
def empty_holes (p : Pegboard) : ℕ := total_holes p - holes_with_pegs p

theorem pegboard_empty_holes :
  ∃ (p : Pegboard), p.size = 10 ∧ empty_holes p = 100 :=
sorry

end NUMINAMATH_CALUDE_pegboard_empty_holes_l2384_238472


namespace NUMINAMATH_CALUDE_tom_climbing_time_l2384_238438

/-- Tom and Elizabeth's hill climbing competition -/
theorem tom_climbing_time (elizabeth_time : ℕ) (tom_factor : ℕ) : elizabeth_time = 30 → tom_factor = 4 → (elizabeth_time * tom_factor) / 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tom_climbing_time_l2384_238438


namespace NUMINAMATH_CALUDE_power_of_ten_plus_one_divisibility_not_always_divisible_by_nine_l2384_238488

theorem power_of_ten_plus_one_divisibility (n : ℕ) :
  (9 ∣ 10^n + 1) → (9 ∣ 10^(n+1) + 1) :=
by sorry

theorem not_always_divisible_by_nine :
  ∃ n : ℕ, ¬(9 ∣ 10^n + 1) :=
by sorry

end NUMINAMATH_CALUDE_power_of_ten_plus_one_divisibility_not_always_divisible_by_nine_l2384_238488
