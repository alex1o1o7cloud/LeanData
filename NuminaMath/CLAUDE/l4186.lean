import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_for_2015_divisibility_l4186_418697

theorem smallest_n_for_2015_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → (2^k - 1) % 2015 = 0 → k ≥ n) ∧ 
  (2^n - 1) % 2015 = 0 ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_2015_divisibility_l4186_418697


namespace NUMINAMATH_CALUDE_min_a_for_inequality_l4186_418601

theorem min_a_for_inequality (a : ℝ) : 
  (∀ x > a, 2 * x + 3 ≥ 7) ↔ a < 2 := by sorry

end NUMINAMATH_CALUDE_min_a_for_inequality_l4186_418601


namespace NUMINAMATH_CALUDE_regular_triangular_prism_edge_length_l4186_418647

/-- A regular triangular prism with edge length a and volume 16√3 has a = 4 -/
theorem regular_triangular_prism_edge_length (a : ℝ) : 
  a > 0 →  -- Ensure a is positive
  (1/4 : ℝ) * a^3 * Real.sqrt 3 = 16 * Real.sqrt 3 → 
  a = 4 := by
  sorry

#check regular_triangular_prism_edge_length

end NUMINAMATH_CALUDE_regular_triangular_prism_edge_length_l4186_418647


namespace NUMINAMATH_CALUDE_nancy_count_l4186_418674

theorem nancy_count (a b c d e f : ℕ) (h_mean : (a + b + c + d + e + f) / 6 = 7)
  (h_a : a = 6) (h_b : b = 12) (h_c : c = 1) (h_d : d = 12) (h_f : f = 8) :
  e = 3 := by
  sorry

end NUMINAMATH_CALUDE_nancy_count_l4186_418674


namespace NUMINAMATH_CALUDE_square_area_proof_l4186_418654

theorem square_area_proof (x : ℚ) : 
  (5 * x - 22 : ℚ) = (34 - 4 * x) → 
  (5 * x - 22 : ℚ) > 0 →
  ((5 * x - 22) ^ 2 : ℚ) = 6724 / 81 := by
sorry

end NUMINAMATH_CALUDE_square_area_proof_l4186_418654


namespace NUMINAMATH_CALUDE_joes_remaining_money_l4186_418639

/-- Represents the problem of calculating Joe's remaining money after shopping --/
theorem joes_remaining_money (initial_amount : ℕ) (notebooks : ℕ) (books : ℕ) 
  (notebook_cost : ℕ) (book_cost : ℕ) : 
  initial_amount = 56 →
  notebooks = 7 →
  books = 2 →
  notebook_cost = 4 →
  book_cost = 7 →
  initial_amount - (notebooks * notebook_cost + books * book_cost) = 14 :=
by sorry

end NUMINAMATH_CALUDE_joes_remaining_money_l4186_418639


namespace NUMINAMATH_CALUDE_power_difference_l4186_418611

theorem power_difference (a m n : ℝ) (hm : a^m = 9) (hn : a^n = 3) :
  a^(m - n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l4186_418611


namespace NUMINAMATH_CALUDE_ruby_candy_sharing_l4186_418624

theorem ruby_candy_sharing (total_candies : ℕ) (candies_per_friend : ℕ) 
  (h1 : total_candies = 36)
  (h2 : candies_per_friend = 4) :
  total_candies / candies_per_friend = 9 := by
  sorry

end NUMINAMATH_CALUDE_ruby_candy_sharing_l4186_418624


namespace NUMINAMATH_CALUDE_stone_distance_l4186_418681

theorem stone_distance (n : ℕ) (total_distance : ℝ) : 
  n = 31 → 
  n % 2 = 1 → 
  total_distance = 4.8 → 
  (2 * (n / 2) * (n / 2 + 1) / 2) * (total_distance / (2 * (n / 2) * (n / 2 + 1) / 2)) = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_stone_distance_l4186_418681


namespace NUMINAMATH_CALUDE_distance_XY_proof_l4186_418660

/-- The distance between points X and Y -/
def distance_XY : ℝ := 52

/-- Yolanda's walking speed in miles per hour -/
def yolanda_speed : ℝ := 3

/-- Bob's walking speed in miles per hour -/
def bob_speed : ℝ := 4

/-- The time difference between Yolanda's and Bob's start in hours -/
def time_difference : ℝ := 1

/-- The distance Bob has walked when they meet -/
def bob_distance : ℝ := 28

theorem distance_XY_proof :
  distance_XY = yolanda_speed * (bob_distance / bob_speed + time_difference) + bob_distance :=
by sorry

end NUMINAMATH_CALUDE_distance_XY_proof_l4186_418660


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4186_418680

theorem trigonometric_identity (α : ℝ) :
  Real.cos (5 / 2 * Real.pi - 6 * α) * Real.sin (Real.pi - 2 * α)^3 -
  Real.cos (6 * α - Real.pi) * Real.sin (Real.pi / 2 - 2 * α)^3 =
  Real.cos (4 * α)^3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4186_418680


namespace NUMINAMATH_CALUDE_cube_and_fifth_power_sum_l4186_418628

theorem cube_and_fifth_power_sum (a : ℝ) (h : (a + 1/a)^2 = 11) :
  (a^3 + 1/a^3, a^5 + 1/a^5) = (8 * Real.sqrt 11, 71 * Real.sqrt 11) ∨
  (a^3 + 1/a^3, a^5 + 1/a^5) = (-8 * Real.sqrt 11, -71 * Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_cube_and_fifth_power_sum_l4186_418628


namespace NUMINAMATH_CALUDE_line_through_circle_center_l4186_418691

theorem line_through_circle_center (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y = 0 ∧ 
                 3*x + y + a = 0 ∧ 
                 x = -1 ∧ y = 2) → 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l4186_418691


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l4186_418627

theorem polynomial_division_theorem (x : ℚ) :
  (4 * x^2 - 4/3 * x + 2) * (3 * x + 4) + 10/3 = 12 * x^3 + 24 * x^2 - 10 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l4186_418627


namespace NUMINAMATH_CALUDE_inequality_proof_l4186_418605

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a + 4 / (a^2 - 2*a*b + b^2) ≥ b + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4186_418605


namespace NUMINAMATH_CALUDE_a_is_arithmetic_sequence_b_max_min_l4186_418675

-- Define the sequence a_n implicitly through S_n
def S (n : ℕ+) : ℚ := (1 / 2) * n^2 - 2 * n

-- Define a_n as the difference of consecutive S_n terms
def a (n : ℕ+) : ℚ := S n - S (n - 1)

-- Define b_n
def b (n : ℕ+) : ℚ := (a n + 1) / (a n)

-- Theorem 1: a_n is an arithmetic sequence with common difference 1
theorem a_is_arithmetic_sequence : ∀ n : ℕ+, n > 1 → a (n + 1) - a n = 1 :=
sorry

-- Theorem 2: Maximum and minimum values of b_n
theorem b_max_min :
  (∀ n : ℕ+, b n ≤ b 3) ∧
  (∀ n : ℕ+, b n ≥ b 2) ∧
  (b 3 = 3) ∧
  (b 2 = -1) :=
sorry

end NUMINAMATH_CALUDE_a_is_arithmetic_sequence_b_max_min_l4186_418675


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l4186_418617

theorem at_least_one_not_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l4186_418617


namespace NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l4186_418612

theorem coefficient_x5_in_expansion : 
  let n : ℕ := 9
  let k : ℕ := 4
  let a : ℝ := 3 * Real.sqrt 2
  (Nat.choose n k) * a^k = 40824 := by sorry

end NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l4186_418612


namespace NUMINAMATH_CALUDE_geometric_sequence_a2_a4_condition_l4186_418699

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_a2_a4_condition (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (is_monotonically_increasing a → a 2 < a 4) ∧
  ¬(a 2 < a 4 → is_monotonically_increasing a) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a2_a4_condition_l4186_418699


namespace NUMINAMATH_CALUDE_parallel_condition_l4186_418651

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line -/
def l1 (a : ℝ) : Line2D :=
  { a := a, b := 1, c := -a + 1 }

/-- The second line -/
def l2 (a : ℝ) : Line2D :=
  { a := 4, b := a, c := -2 }

theorem parallel_condition (a : ℝ) :
  (parallel (l1 a) (l2 a) → a = 2 ∨ a = -2) ∧
  ¬(a = 2 ∨ a = -2 → parallel (l1 a) (l2 a)) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l4186_418651


namespace NUMINAMATH_CALUDE_factorization_proof_l4186_418676

theorem factorization_proof (a : ℝ) : 180 * a^2 + 45 * a = 45 * a * (4 * a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l4186_418676


namespace NUMINAMATH_CALUDE_z_ninth_power_l4186_418625

theorem z_ninth_power (z : ℂ) : z = (-Real.sqrt 3 + Complex.I) / 2 → z^9 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_z_ninth_power_l4186_418625


namespace NUMINAMATH_CALUDE_domain_transformation_l4186_418649

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem domain_transformation (h : Set.Icc (-3 : ℝ) 3 = {x | ∃ y, f (2*y - 1) = x}) :
  {x | ∃ y, f y = x} = Set.Icc (-7 : ℝ) 5 := by
  sorry

end NUMINAMATH_CALUDE_domain_transformation_l4186_418649


namespace NUMINAMATH_CALUDE_number_of_students_l4186_418657

theorem number_of_students (group_size : ℕ) (num_groups : ℕ) (h1 : group_size = 5) (h2 : num_groups = 6) :
  group_size * num_groups = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l4186_418657


namespace NUMINAMATH_CALUDE_min_shared_side_length_l4186_418619

/-- Given three triangles ABC, DBC, and EBC sharing side BC, prove that the minimum possible
    integer length of BC is 8 cm. -/
theorem min_shared_side_length
  (AB : ℝ) (AC : ℝ) (DC : ℝ) (BD : ℝ) (EC : ℝ)
  (h_AB : AB = 7)
  (h_AC : AC = 15)
  (h_DC : DC = 9)
  (h_BD : BD = 12)
  (h_EC : EC = 11)
  : ∃ (BC : ℕ), BC ≥ 8 ∧ 
    (∀ (BC' : ℕ), BC' ≥ 8 → BC' > AC - AB) ∧
    (∀ (BC' : ℕ), BC' ≥ 8 → BC' > BD - DC) ∧
    (∀ (BC' : ℕ), BC' ≥ 8 → BC' > 0) ∧
    (∀ (BC'' : ℕ), BC'' < BC → 
      (BC'' ≤ AC - AB ∨ BC'' ≤ BD - DC ∨ BC'' ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_min_shared_side_length_l4186_418619


namespace NUMINAMATH_CALUDE_ravi_overall_profit_l4186_418644

/-- Calculates the overall profit for Ravi's purchases and sales -/
theorem ravi_overall_profit (refrigerator_cost mobile_cost : ℝ)
  (refrigerator_loss_percent mobile_profit_percent : ℝ) :
  refrigerator_cost = 15000 →
  mobile_cost = 8000 →
  refrigerator_loss_percent = 4 →
  mobile_profit_percent = 10 →
  let refrigerator_selling_price := refrigerator_cost * (1 - refrigerator_loss_percent / 100)
  let mobile_selling_price := mobile_cost * (1 + mobile_profit_percent / 100)
  let total_cost := refrigerator_cost + mobile_cost
  let total_selling_price := refrigerator_selling_price + mobile_selling_price
  total_selling_price - total_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_ravi_overall_profit_l4186_418644


namespace NUMINAMATH_CALUDE_train_length_l4186_418640

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 4 → ∃ length : ℝ, abs (length - 66.68) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l4186_418640


namespace NUMINAMATH_CALUDE_triangle_with_pi_power_sum_is_acute_l4186_418618

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

-- Define the property of being an acute triangle
def IsAcute (t : Triangle) : Prop :=
  t.a^2 + t.b^2 > t.c^2

-- State the theorem
theorem triangle_with_pi_power_sum_is_acute (t : Triangle) 
  (h : t.a^Real.pi + t.b^Real.pi = t.c^Real.pi) : IsAcute t := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_pi_power_sum_is_acute_l4186_418618


namespace NUMINAMATH_CALUDE_donut_combinations_l4186_418677

/-- The number of ways to choose k items from n types with repetition. -/
def choose_with_repetition (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- The number of donut types available. -/
def num_donut_types : ℕ := 6

/-- The number of remaining donuts to be chosen. -/
def remaining_donuts : ℕ := 2

/-- The total number of donuts in the order. -/
def total_donuts : ℕ := 8

/-- The number of donuts already accounted for (2 each of 3 specific kinds). -/
def accounted_donuts : ℕ := 6

theorem donut_combinations :
  choose_with_repetition num_donut_types remaining_donuts = 21 ∧
  total_donuts = accounted_donuts + remaining_donuts :=
by sorry

end NUMINAMATH_CALUDE_donut_combinations_l4186_418677


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l4186_418662

theorem five_digit_divisible_by_nine (B : ℕ) : 
  B < 10 →
  (40000 + 10000 * B + 500 + 20 + B) % 9 = 0 →
  B = 8 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l4186_418662


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l4186_418642

theorem part_to_whole_ratio (N P : ℚ) 
  (h1 : (1/4) * P = 17)
  (h2 : (2/5) * N = P)
  (h3 : (40/100) * N = 204) :
  P / N = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l4186_418642


namespace NUMINAMATH_CALUDE_irrational_minus_two_implies_irrational_l4186_418641

theorem irrational_minus_two_implies_irrational (a : ℝ) :
  Irrational (a - 2) → Irrational a := by
  sorry

end NUMINAMATH_CALUDE_irrational_minus_two_implies_irrational_l4186_418641


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_value_l4186_418682

/-- Given functions f and g, prove that if f(x) ≥ g(x) holds exactly for x ∈ [-1, 2], then m = 2 -/
theorem function_inequality_implies_m_value (m : ℝ) :
  (∀ x : ℝ, (x^2 - 3*x + m ≥ 2*x^2 - 4*x) ↔ (-1 ≤ x ∧ x ≤ 2)) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_value_l4186_418682


namespace NUMINAMATH_CALUDE_c_share_is_63_l4186_418678

/-- Represents a person renting the pasture -/
structure Renter where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent for a given renter -/
def calculateShare (renter : Renter) (totalRent : ℕ) (totalOxMonths : ℕ) : ℚ :=
  (renter.oxen * renter.months : ℚ) / totalOxMonths * totalRent

theorem c_share_is_63 (a b c : Renter) (totalRent : ℕ) :
  a.oxen = 10 →
  a.months = 7 →
  b.oxen = 12 →
  b.months = 5 →
  c.oxen = 15 →
  c.months = 3 →
  totalRent = 245 →
  calculateShare c totalRent (a.oxen * a.months + b.oxen * b.months + c.oxen * c.months) = 63 := by
  sorry

#eval calculateShare (Renter.mk 15 3) 245 175

end NUMINAMATH_CALUDE_c_share_is_63_l4186_418678


namespace NUMINAMATH_CALUDE_elephant_donkey_weight_l4186_418694

/-- Calculates the combined weight of an elephant and a donkey in pounds -/
theorem elephant_donkey_weight (elephant_tons : ℝ) (donkey_percent_less : ℝ) : 
  elephant_tons = 3 ∧ donkey_percent_less = 90 →
  elephant_tons * 2000 + (elephant_tons * 2000 * (1 - donkey_percent_less / 100)) = 6600 := by
  sorry

end NUMINAMATH_CALUDE_elephant_donkey_weight_l4186_418694


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l4186_418622

-- Define the days of the week
inductive Day : Type
  | Sunday : Day
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day

def next_day (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

def days_after (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => next_day (days_after d m)

theorem tomorrow_is_saturday 
  (day_before_yesterday : Day)
  (h : days_after day_before_yesterday 5 = Day.Monday) :
  days_after day_before_yesterday 3 = Day.Saturday :=
by sorry

end NUMINAMATH_CALUDE_tomorrow_is_saturday_l4186_418622


namespace NUMINAMATH_CALUDE_nine_digit_integers_count_l4186_418686

theorem nine_digit_integers_count : 
  (Finset.range 8).card * (10 ^ 8) = 800000000 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_integers_count_l4186_418686


namespace NUMINAMATH_CALUDE_flowers_in_pot_l4186_418688

theorem flowers_in_pot (chrysanthemums : ℕ) (roses : ℕ) : 
  chrysanthemums = 5 → roses = 2 → chrysanthemums + roses = 7 := by
  sorry

end NUMINAMATH_CALUDE_flowers_in_pot_l4186_418688


namespace NUMINAMATH_CALUDE_odd_function_value_l4186_418613

def f (a : ℝ) (x : ℝ) : ℝ := a * x + a + 3

theorem odd_function_value (a : ℝ) :
  (∀ x : ℝ, f a x = -(f a (-x))) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l4186_418613


namespace NUMINAMATH_CALUDE_triangle_construction_theorem_l4186_418603

-- Define the types for points and triangles
def Point := ℝ × ℝ
def Triangle := Point × Point × Point

-- Define a function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Define a predicate for parallel lines
def parallel (p1 p2 q1 q2 : Point) : Prop := sorry

-- Define a predicate for congruent triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

theorem triangle_construction_theorem 
  (ABC A₁B₁C₁ : Triangle) 
  (h_equal_area : triangleArea ABC = triangleArea A₁B₁C₁) :
  ∃ (A₂B₂C₂ : Triangle),
    congruent A₂B₂C₂ A₁B₁C₁ ∧ 
    parallel (ABC.1) (A₂B₂C₂.1) (ABC.2.1) (A₂B₂C₂.2.1) ∧
    parallel (ABC.2.1) (A₂B₂C₂.2.1) (ABC.2.2) (A₂B₂C₂.2.2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_construction_theorem_l4186_418603


namespace NUMINAMATH_CALUDE_max_fraction_sum_l4186_418645

def DigitSet : Set Nat := {2, 3, 4, 5, 6, 7, 8, 9}

def ValidOptions : Set Rat := {2/17, 3/17, 17/72, 25/72, 13/36}

theorem max_fraction_sum (A B C D : Nat) :
  A ∈ DigitSet → B ∈ DigitSet → C ∈ DigitSet → D ∈ DigitSet →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (A : Rat) / B + (C : Rat) / D ∈ ValidOptions →
  ∀ (X Y Z W : Nat), X ∈ DigitSet → Y ∈ DigitSet → Z ∈ DigitSet → W ∈ DigitSet →
    X ≠ Y → X ≠ Z → X ≠ W → Y ≠ Z → Y ≠ W → Z ≠ W →
    (X : Rat) / Y + (Z : Rat) / W ∈ ValidOptions →
    (X : Rat) / Y + (Z : Rat) / W ≤ (A : Rat) / B + (C : Rat) / D →
  (A : Rat) / B + (C : Rat) / D = 25 / 72 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l4186_418645


namespace NUMINAMATH_CALUDE_stream_speed_l4186_418698

/-- 
Proves that given a boat with a speed of 51 kmph in still water, 
if the time taken to row upstream is twice the time taken to row downstream, 
then the speed of the stream is 17 kmph.
-/
theorem stream_speed (D : ℝ) (v : ℝ) : 
  (D / (51 - v) = 2 * (D / (51 + v))) → v = 17 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l4186_418698


namespace NUMINAMATH_CALUDE_sum_of_roots_l4186_418667

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 14*p*x - 15*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 14*r*x - 15*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 3150 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l4186_418667


namespace NUMINAMATH_CALUDE_greater_than_implies_greater_than_scaled_and_shifted_l4186_418670

theorem greater_than_implies_greater_than_scaled_and_shifted {a b : ℝ} (h : a > b) : 3*a + 5 > 3*b + 5 := by
  sorry

end NUMINAMATH_CALUDE_greater_than_implies_greater_than_scaled_and_shifted_l4186_418670


namespace NUMINAMATH_CALUDE_sugar_calculation_l4186_418637

theorem sugar_calculation (initial_sugar : ℕ) (used_sugar : ℕ) (bought_sugar : ℕ) 
  (h1 : initial_sugar = 65)
  (h2 : used_sugar = 18)
  (h3 : bought_sugar = 50) :
  initial_sugar - used_sugar + bought_sugar = 97 := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l4186_418637


namespace NUMINAMATH_CALUDE_largest_common_number_l4186_418616

def is_in_first_sequence (n : ℕ) : Prop := ∃ k : ℕ, n = 1 + 8 * k

def is_in_second_sequence (n : ℕ) : Prop := ∃ m : ℕ, n = 4 + 9 * m

def is_in_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 250

theorem largest_common_number :
  (is_in_first_sequence 193 ∧ is_in_second_sequence 193 ∧ is_in_range 193) ∧
  ∀ n : ℕ, is_in_first_sequence n → is_in_second_sequence n → is_in_range n → n ≤ 193 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_number_l4186_418616


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l4186_418696

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * Real.sqrt 3 * x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 2 * Real.sqrt 3 * y + k = 0 → y = x) → 
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l4186_418696


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4186_418608

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4186_418608


namespace NUMINAMATH_CALUDE_range_of_m_l4186_418663

theorem range_of_m (a b m : ℝ) (h1 : 3 * a + 4 / b = 1) (h2 : a > 0) (h3 : b > 0)
  (h4 : ∀ (a b : ℝ), a > 0 → b > 0 → 3 * a + 4 / b = 1 → 1 / a + 3 * b > m) :
  m < 27 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l4186_418663


namespace NUMINAMATH_CALUDE_min_sum_m_n_l4186_418626

theorem min_sum_m_n (m n : ℕ+) (h : 300 * m = n^3) : 
  ∀ (m' n' : ℕ+), 300 * m' = n'^3 → m + n ≤ m' + n' :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l4186_418626


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l4186_418658

/-- In a right-angled triangle XYZ, given the following conditions:
  - ∠X = 90°
  - YZ = 20
  - tan Z = 3 sin Y
  Prove that XY = (40√2) / 3 -/
theorem right_triangle_side_length (X Y Z : ℝ) (h1 : X^2 + Y^2 = Z^2) 
  (h2 : Z = 20) (h3 : Real.tan X = 3 * Real.sin Y) : 
  Y = (40 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l4186_418658


namespace NUMINAMATH_CALUDE_sum_base5_equals_l4186_418684

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec convert (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else convert (m / 5) ((m % 5) :: acc)
  convert n []

/-- The theorem to be proved -/
theorem sum_base5_equals : 
  decimalToBase5 (base5ToDecimal [1, 2, 3] + base5ToDecimal [4, 3, 2] + base5ToDecimal [2, 1, 4]) = 
  [1, 3, 2, 4] := by
  sorry

end NUMINAMATH_CALUDE_sum_base5_equals_l4186_418684


namespace NUMINAMATH_CALUDE_jane_current_age_l4186_418673

/-- Represents Jane's age when she stopped babysitting -/
def jane_stop_age : ℕ := 30

/-- Represents the number of years since Jane stopped babysitting -/
def years_since_stop : ℕ := 10

/-- Represents the current age of the oldest person Jane could have babysat -/
def oldest_babysat_current_age : ℕ := 25

/-- Represents Jane's starting age for babysitting -/
def jane_start_age : ℕ := 18

theorem jane_current_age :
  jane_stop_age + years_since_stop = 40 ∧
  2 * (oldest_babysat_current_age - years_since_stop) ≤ jane_stop_age ∧
  jane_stop_age ≥ jane_start_age :=
by sorry

end NUMINAMATH_CALUDE_jane_current_age_l4186_418673


namespace NUMINAMATH_CALUDE_statues_painted_l4186_418630

theorem statues_painted (total_paint : ℚ) (paint_per_statue : ℚ) :
  total_paint = 3/6 ∧ paint_per_statue = 1/6 → total_paint / paint_per_statue = 3 :=
by sorry

end NUMINAMATH_CALUDE_statues_painted_l4186_418630


namespace NUMINAMATH_CALUDE_weight_of_new_person_l4186_418633

theorem weight_of_new_person (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 12 →
  weight_increase = 3.7 →
  replaced_weight = 57.3 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 101.7 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l4186_418633


namespace NUMINAMATH_CALUDE_spurs_team_size_l4186_418661

theorem spurs_team_size :
  ∀ (num_players : ℕ) (basketballs_per_player : ℕ) (total_basketballs : ℕ),
    basketballs_per_player = 11 →
    total_basketballs = 242 →
    total_basketballs = num_players * basketballs_per_player →
    num_players = 22 := by
  sorry

end NUMINAMATH_CALUDE_spurs_team_size_l4186_418661


namespace NUMINAMATH_CALUDE_max_pq_plus_r_for_primes_l4186_418689

theorem max_pq_plus_r_for_primes (p q r : ℕ) : 
  Prime p → Prime q → Prime r → 
  p * q + q * r + r * p = 2016 → 
  p * q + r ≤ 1008 := by
sorry

end NUMINAMATH_CALUDE_max_pq_plus_r_for_primes_l4186_418689


namespace NUMINAMATH_CALUDE_number_problem_l4186_418609

theorem number_problem (x n : ℝ) : x = 4 ∧ n * x + 3 = 10 * x - 17 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4186_418609


namespace NUMINAMATH_CALUDE_problem_solution_l4186_418646

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : 
  x = (3 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4186_418646


namespace NUMINAMATH_CALUDE_chi_squared_relationship_confidence_l4186_418664

-- Define the chi-squared statistic
def chi_squared : ℝ := 4.073

-- Define the critical values and their corresponding p-values
def critical_value_1 : ℝ := 3.841
def p_value_1 : ℝ := 0.05

def critical_value_2 : ℝ := 5.024
def p_value_2 : ℝ := 0.025

-- Define the confidence level we want to prove
def target_confidence : ℝ := 0.95

-- Theorem statement
theorem chi_squared_relationship_confidence :
  chi_squared > critical_value_1 ∧ chi_squared < critical_value_2 →
  ∃ (confidence : ℝ), confidence ≥ target_confidence ∧
    confidence ≤ 1 - p_value_1 ∧
    confidence > 1 - p_value_2 :=
by sorry

end NUMINAMATH_CALUDE_chi_squared_relationship_confidence_l4186_418664


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l4186_418665

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l4186_418665


namespace NUMINAMATH_CALUDE_arithmetic_sequence_value_l4186_418687

/-- 
Given that -7, a, and 1 form an arithmetic sequence, prove that a = -3.
-/
theorem arithmetic_sequence_value (a : ℝ) : 
  (∃ d : ℝ, a - (-7) = d ∧ 1 - a = d) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_value_l4186_418687


namespace NUMINAMATH_CALUDE_max_k_inequality_l4186_418621

theorem max_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ k : ℝ, k = 100 ∧ 
  (∀ k' : ℝ, (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → 
    k' * a' * b' * c' / (a' + b' + c') ≤ (a' + b')^2 + (a' + b' + 4*c')^2) → 
  k' ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_max_k_inequality_l4186_418621


namespace NUMINAMATH_CALUDE_negation_equivalence_l4186_418636

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, (x₀ + 1 < 0) ∨ (x₀^2 - x₀ > 0)) ↔ 
  (∀ x : ℝ, (x + 1 ≥ 0) ∧ (x^2 - x ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4186_418636


namespace NUMINAMATH_CALUDE_least_months_to_double_debt_l4186_418652

def initial_amount : ℝ := 1500
def monthly_rate : ℝ := 0.06

def amount_owed (t : ℕ) : ℝ :=
  initial_amount * (1 + monthly_rate) ^ t

theorem least_months_to_double_debt :
  (∀ n < 12, amount_owed n ≤ 2 * initial_amount) ∧
  amount_owed 12 > 2 * initial_amount :=
sorry

end NUMINAMATH_CALUDE_least_months_to_double_debt_l4186_418652


namespace NUMINAMATH_CALUDE_floor_difference_equals_five_l4186_418690

theorem floor_difference_equals_five (n : ℤ) : (⌊(n^2 : ℚ) / 4⌋ - ⌊(n : ℚ) / 2⌋^2 = 5) ↔ (n = 11) :=
sorry

end NUMINAMATH_CALUDE_floor_difference_equals_five_l4186_418690


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l4186_418653

-- Define the set A
def A : Set ℝ := {a | ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0}

-- Define the set B
def B : Set ℝ := {x | ∀ a ∈ Set.Icc (-2 : ℝ) 2, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l4186_418653


namespace NUMINAMATH_CALUDE_first_seven_primes_sum_mod_eighth_prime_l4186_418610

theorem first_seven_primes_sum_mod_eighth_prime : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17) % 19 = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_seven_primes_sum_mod_eighth_prime_l4186_418610


namespace NUMINAMATH_CALUDE_sum_of_fifth_and_eighth_term_l4186_418671

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_fifth_and_eighth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_roots : a 3 * a 10 = 5 ∧ a 3 + a 10 = 3) :
  a 5 + a 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_and_eighth_term_l4186_418671


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_even_numbers_l4186_418685

theorem sum_of_four_consecutive_even_numbers : 
  let n : ℕ := 32
  let sum := n + (n + 2) + (n + 4) + (n + 6)
  sum = 140 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_even_numbers_l4186_418685


namespace NUMINAMATH_CALUDE_angle_at_5pm_l4186_418648

/-- The angle between the hour and minute hands of a clock at a given hour -/
def clockAngle (hour : ℝ) : ℝ := 30 * hour

/-- Proposition: The angle between the minute hand and hour hand is 150° at 5 pm -/
theorem angle_at_5pm : clockAngle 5 = 150 := by sorry

end NUMINAMATH_CALUDE_angle_at_5pm_l4186_418648


namespace NUMINAMATH_CALUDE_friday_zoo_visitors_l4186_418600

/-- The number of people who visited the zoo on Saturday -/
def saturday_visitors : ℕ := 3750

/-- The number of people who visited the zoo on Friday -/
def friday_visitors : ℕ := saturday_visitors / 3

/-- Theorem stating that 1250 people visited the zoo on Friday -/
theorem friday_zoo_visitors : friday_visitors = 1250 := by
  sorry

end NUMINAMATH_CALUDE_friday_zoo_visitors_l4186_418600


namespace NUMINAMATH_CALUDE_equation_three_solutions_l4186_418692

theorem equation_three_solutions (a : ℝ) :
  (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℝ, x^2 * a - 2*x + 1 = 3 * |x| ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) ↔
  a = (1 : ℝ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_three_solutions_l4186_418692


namespace NUMINAMATH_CALUDE_range_of_a_l4186_418632

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 1 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, Real.exp (2*x) - 2 * Real.exp x + a ≥ 0

-- Define the theorem
theorem range_of_a : 
  ∀ a : ℝ, (p a ∧ q a) → a ∈ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l4186_418632


namespace NUMINAMATH_CALUDE_max_product_roots_l4186_418604

theorem max_product_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 2 * x^2 - m * x + m^2 = 0 ∧ 2 * y^2 - m * y + m^2 = 0) →
  (∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ 2 * x^2 - k * x + k^2 = 0 ∧ 2 * y^2 - k * y + k^2 = 0) →
    (m^2 / 2 ≥ k^2 / 2)) →
  m = 0 :=
sorry

end NUMINAMATH_CALUDE_max_product_roots_l4186_418604


namespace NUMINAMATH_CALUDE_expression_evaluation_l4186_418634

theorem expression_evaluation (c d : ℝ) (hc : c = 3) (hd : d = 2) :
  (c^2 + d)^2 - (c^2 - d)^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4186_418634


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l4186_418631

theorem permutation_equation_solution (m : ℕ) : 
  (m * (m - 1) * (m - 2) * (m - 3) * (m - 4) = 2 * m * (m - 1) * (m - 2)) → m = 5 :=
by sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l4186_418631


namespace NUMINAMATH_CALUDE_cubic_equation_root_problem_l4186_418623

theorem cubic_equation_root_problem (c d : ℚ) : 
  (∃ x : ℝ, x^3 + c*x^2 + d*x + 15 = 0 ∧ x = 3 + Real.sqrt 5) → d = -37/2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_root_problem_l4186_418623


namespace NUMINAMATH_CALUDE_popsicle_stick_sum_l4186_418655

/-- The sum of popsicle sticks owned by two people -/
theorem popsicle_stick_sum (gino_sticks : ℕ) (your_sticks : ℕ) 
  (h1 : gino_sticks = 63) (h2 : your_sticks = 50) : 
  gino_sticks + your_sticks = 113 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_stick_sum_l4186_418655


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l4186_418629

theorem simplify_and_evaluate (a : ℚ) (h : a = -3) : 
  (a - 2) / ((1 + 2*a + a^2) * (a - 3*a/(a+1))) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l4186_418629


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_three_l4186_418615

theorem opposite_of_sqrt_three : -(Real.sqrt 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_three_l4186_418615


namespace NUMINAMATH_CALUDE_period_of_cosine_l4186_418643

theorem period_of_cosine (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.cos ((3 * x) / 4)
  ∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x ∧ T = (8 * Real.pi) / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_period_of_cosine_l4186_418643


namespace NUMINAMATH_CALUDE_power_mod_50_l4186_418656

theorem power_mod_50 : 11^1501 % 50 = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_50_l4186_418656


namespace NUMINAMATH_CALUDE_product_increase_l4186_418672

theorem product_increase (A B : ℝ) (h : A * B = 1.6) : (5 * A) * (5 * B) = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_increase_l4186_418672


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_eq_2_subset_complement_iff_m_geq_3_l4186_418602

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x : ℝ | x > 2^m}
def B : Set ℝ := {x : ℝ | -4 < x - 4 ∧ x - 4 < 4}

-- Theorem for part (1)
theorem intersection_and_union_when_m_eq_2 :
  (A 2 ∩ B = {x : ℝ | 4 < x ∧ x < 8}) ∧
  (A 2 ∪ B = {x : ℝ | x > 0}) := by sorry

-- Theorem for part (2)
theorem subset_complement_iff_m_geq_3 (m : ℝ) :
  A m ⊆ (Set.univ \ B) ↔ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_eq_2_subset_complement_iff_m_geq_3_l4186_418602


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4186_418607

theorem polynomial_factorization :
  ∀ x : ℝ, x^12 + x^6 + 1 = (x^4 - x^2 + 1) * (x^8 + x^4 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4186_418607


namespace NUMINAMATH_CALUDE_can_generate_x_squared_and_xy_l4186_418650

/-- A type representing the allowed operations on numbers -/
inductive Operation
  | Add : ℝ → ℝ → Operation
  | Sub : ℝ → ℝ → Operation
  | Mul : ℝ → ℝ → Operation
  | Div : ℝ → ℝ → Operation
  | Recip : ℝ → Operation

/-- A function that applies an operation to a set of real numbers -/
def applyOperation (s : Set ℝ) (op : Operation) : Set ℝ :=
  match op with
  | Operation.Add a b => s ∪ {a + b}
  | Operation.Sub a b => s ∪ {a - b}
  | Operation.Mul a b => s ∪ {a * b}
  | Operation.Div a b => s ∪ {a / b}
  | Operation.Recip a => s ∪ {1 / a}

/-- The main theorem stating that x^2 and xy can be generated -/
theorem can_generate_x_squared_and_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ (ops : List Operation), 
    let final_set := (ops.foldl applyOperation {x, y, 1})
    x^2 ∈ final_set ∧ x*y ∈ final_set :=
  sorry

end NUMINAMATH_CALUDE_can_generate_x_squared_and_xy_l4186_418650


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l4186_418668

theorem average_of_a_and_b (a b : ℝ) : 
  (5 + a + b) / 3 = 33 → (a + b) / 2 = 47 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l4186_418668


namespace NUMINAMATH_CALUDE_wages_calculation_l4186_418635

def total_budget : ℚ := 3000

def food_fraction : ℚ := 1/3
def supplies_fraction : ℚ := 1/4

def food_expense : ℚ := food_fraction * total_budget
def supplies_expense : ℚ := supplies_fraction * total_budget

def wages_expense : ℚ := total_budget - (food_expense + supplies_expense)

theorem wages_calculation : wages_expense = 1250 := by
  sorry

end NUMINAMATH_CALUDE_wages_calculation_l4186_418635


namespace NUMINAMATH_CALUDE_wolf_sheep_eating_time_l4186_418679

/-- If 7 wolves eat 7 sheep in 7 days, then 9 wolves will eat 9 sheep in 7 days. -/
theorem wolf_sheep_eating_time (initial_wolves initial_sheep initial_days : ℕ) 
  (new_wolves new_sheep : ℕ) : 
  initial_wolves = 7 → initial_sheep = 7 → initial_days = 7 →
  new_wolves = 9 → new_sheep = 9 →
  initial_wolves * initial_sheep * new_days = new_wolves * new_sheep * initial_days →
  new_days = 7 :=
by
  sorry

#check wolf_sheep_eating_time

end NUMINAMATH_CALUDE_wolf_sheep_eating_time_l4186_418679


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l4186_418683

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → 
  a + h + k = -6 := by sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l4186_418683


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l4186_418659

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (5678 * 10 + N) % 6 = 0 → N ≤ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l4186_418659


namespace NUMINAMATH_CALUDE_judys_shopping_cost_l4186_418666

/-- Represents Judy's shopping trip cost calculation --/
theorem judys_shopping_cost :
  let carrot_cost : ℕ := 5 * 1
  let milk_cost : ℕ := 3 * 3
  let pineapple_cost : ℕ := 2 * (4 / 2)
  let flour_cost : ℕ := 2 * 5
  let ice_cream_cost : ℕ := 7
  let total_before_coupon := carrot_cost + milk_cost + pineapple_cost + flour_cost + ice_cream_cost
  let coupon_value : ℕ := 5
  let coupon_threshold : ℕ := 25
  total_before_coupon ≥ coupon_threshold →
  total_before_coupon - coupon_value = 30 :=
by sorry

end NUMINAMATH_CALUDE_judys_shopping_cost_l4186_418666


namespace NUMINAMATH_CALUDE_expression_evaluation_l4186_418693

theorem expression_evaluation :
  let x : ℚ := 3
  let f (y : ℚ) := (y + 3) / (y - 2)
  3 * (f (f x) + 3) / (f (f x) - 2) = 27 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4186_418693


namespace NUMINAMATH_CALUDE_stamp_purchase_problem_l4186_418638

theorem stamp_purchase_problem :
  ∀ (x y z : ℕ),
  (x : ℤ) + 2 * y + 5 * z = 100 →  -- Total cost in cents
  y = 10 * x →                    -- Relation between 1-cent and 2-cent stamps
  x > 0 ∧ y > 0 ∧ z > 0 →         -- All stamp quantities are positive
  x = 5 ∧ y = 50 ∧ z = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_stamp_purchase_problem_l4186_418638


namespace NUMINAMATH_CALUDE_limit_proof_l4186_418606

theorem limit_proof : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ,
  0 < |x - 1/3| ∧ |x - 1/3| < δ →
  |(15*x^2 - 2*x - 1) / (x - 1/3) - 8| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_proof_l4186_418606


namespace NUMINAMATH_CALUDE_sunset_colors_l4186_418695

/-- The duration of the sunset in hours -/
def sunset_duration : ℕ := 2

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The interval between color changes in minutes -/
def color_change_interval : ℕ := 10

/-- The number of colors the sky turns during the sunset -/
def number_of_colors : ℕ := sunset_duration * minutes_per_hour / color_change_interval

theorem sunset_colors :
  number_of_colors = 12 := by
  sorry

end NUMINAMATH_CALUDE_sunset_colors_l4186_418695


namespace NUMINAMATH_CALUDE_drive_duration_proof_l4186_418669

def podcast1_duration : ℕ := 45
def podcast2_duration : ℕ := 2 * podcast1_duration
def podcast3_duration : ℕ := 105
def podcast4_duration : ℕ := 60
def podcast5_duration : ℕ := 60

def total_duration : ℕ := podcast1_duration + podcast2_duration + podcast3_duration + podcast4_duration + podcast5_duration

theorem drive_duration_proof :
  total_duration / 60 = 6 := by sorry

end NUMINAMATH_CALUDE_drive_duration_proof_l4186_418669


namespace NUMINAMATH_CALUDE_larger_number_problem_l4186_418614

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 10) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l4186_418614


namespace NUMINAMATH_CALUDE_two_in_A_l4186_418620

def A : Set ℝ := {x | x > 1}

theorem two_in_A : 2 ∈ A := by
  sorry

end NUMINAMATH_CALUDE_two_in_A_l4186_418620
