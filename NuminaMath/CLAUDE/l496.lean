import Mathlib

namespace NUMINAMATH_CALUDE_universal_set_equality_l496_49651

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5, 7}

-- Define set A
def A : Finset Nat := {1, 3, 5, 7}

-- Define set B
def B : Finset Nat := {3, 5}

-- Theorem statement
theorem universal_set_equality : U = A ∪ (U \ B) := by
  sorry

end NUMINAMATH_CALUDE_universal_set_equality_l496_49651


namespace NUMINAMATH_CALUDE_expand_expression_l496_49698

theorem expand_expression (x : ℝ) : 20 * (3 * x + 4) - 10 = 60 * x + 70 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l496_49698


namespace NUMINAMATH_CALUDE_coin_puzzle_solution_l496_49665

/-- Represents the number of coins in each pile -/
structure CoinPiles :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Represents the coin movement operation -/
def moveCoins (piles : CoinPiles) : CoinPiles :=
  { first := piles.first - piles.second + piles.third,
    second := 2 * piles.second - piles.third,
    third := piles.third + piles.second - piles.first }

/-- Theorem stating that if after moving coins each pile has 16 coins, 
    then the initial number in the first pile was 22 -/
theorem coin_puzzle_solution (initial : CoinPiles) :
  (moveCoins initial).first = 16 ∧
  (moveCoins initial).second = 16 ∧
  (moveCoins initial).third = 16 →
  initial.first = 22 :=
sorry

end NUMINAMATH_CALUDE_coin_puzzle_solution_l496_49665


namespace NUMINAMATH_CALUDE_angle_D_measure_l496_49627

-- Define a convex hexagon
structure ConvexHexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ
  sum_of_angles : A + B + C + D + E + F = 720
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E ∧ 0 < F

-- Define the specific conditions of the hexagon
def SpecialHexagon (h : ConvexHexagon) : Prop :=
  h.A = h.B ∧ h.B = h.C ∧   -- Angles A, B, and C are congruent
  h.D = h.E ∧ h.E = h.F ∧   -- Angles D, E, and F are congruent
  h.A + 30 = h.D            -- Angle A is 30° less than angle D

-- State the theorem
theorem angle_D_measure (h : ConvexHexagon) (special : SpecialHexagon h) : h.D = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l496_49627


namespace NUMINAMATH_CALUDE_part_one_part_two_l496_49693

-- Define the sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | x < 2*m - 3}

-- Statement for part 1
theorem part_one (m : ℝ) (h : m = 5) : 
  A ∩ B m = A ∧ (Aᶜ ∪ B m) = Set.univ := by sorry

-- Statement for part 2
theorem part_two (m : ℝ) : 
  A ⊆ B m ↔ m > 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l496_49693


namespace NUMINAMATH_CALUDE_first_hundred_complete_l496_49644

/-- Represents a color of a number in the sequence -/
inductive Color
| Blue
| Red

/-- Represents the properties of the sequence of 200 numbers -/
structure NumberSequence :=
  (numbers : Fin 200 → ℕ)
  (colors : Fin 200 → Color)
  (blue_ascending : ∀ i j, i < j → colors i = Color.Blue → colors j = Color.Blue → numbers i < numbers j)
  (red_descending : ∀ i j, i < j → colors i = Color.Red → colors j = Color.Red → numbers i > numbers j)
  (blue_range : ∀ n, n ∈ Finset.range 100 → ∃ i, colors i = Color.Blue ∧ numbers i = n + 1)
  (red_range : ∀ n, n ∈ Finset.range 100 → ∃ i, colors i = Color.Red ∧ numbers i = 100 - n)

/-- The main theorem stating that the first 100 numbers contain all natural numbers from 1 to 100 -/
theorem first_hundred_complete (seq : NumberSequence) :
  ∀ n, n ∈ Finset.range 100 → ∃ i, i < 100 ∧ seq.numbers i = n + 1 :=
sorry

end NUMINAMATH_CALUDE_first_hundred_complete_l496_49644


namespace NUMINAMATH_CALUDE_profit_and_max_profit_l496_49614

/-- Initial profit per visitor in yuan -/
def initial_profit_per_visitor : ℝ := 10

/-- Initial daily visitor count -/
def initial_visitor_count : ℝ := 500

/-- Visitor loss per yuan of price increase -/
def visitor_loss_per_yuan : ℝ := 20

/-- Calculate profit based on price increase -/
def profit (price_increase : ℝ) : ℝ :=
  (initial_profit_per_visitor + price_increase) * (initial_visitor_count - visitor_loss_per_yuan * price_increase)

/-- Ticket price increase for 6000 yuan daily profit -/
def price_increase_for_target_profit : ℝ := 10

/-- Ticket price increase for maximum profit -/
def price_increase_for_max_profit : ℝ := 7.5

theorem profit_and_max_profit :
  (profit price_increase_for_target_profit = 6000) ∧
  (∀ x : ℝ, profit x ≤ profit price_increase_for_max_profit) := by
  sorry

end NUMINAMATH_CALUDE_profit_and_max_profit_l496_49614


namespace NUMINAMATH_CALUDE_valid_plate_count_l496_49675

/-- Represents a license plate with 4 characters -/
structure LicensePlate :=
  (first : Char) (second : Char) (third : Char) (fourth : Char)

/-- Checks if a character is a letter -/
def isLetter (c : Char) : Bool := c.isAlpha

/-- Checks if a character is a digit -/
def isDigit (c : Char) : Bool := c.isDigit

/-- Checks if a license plate is valid according to the given conditions -/
def isValidPlate (plate : LicensePlate) : Bool :=
  (isLetter plate.first) &&
  (isDigit plate.second) &&
  (isDigit plate.third) &&
  (isLetter plate.fourth) &&
  (plate.first == plate.fourth || plate.second == plate.third)

/-- The total number of possible characters for a letter position -/
def numLetters : Nat := 26

/-- The total number of possible characters for a digit position -/
def numDigits : Nat := 10

/-- Counts the number of valid license plates -/
def countValidPlates : Nat :=
  (numLetters * numDigits * 1 * numLetters) +  -- Same digits
  (numLetters * numDigits * numDigits * 1) -   -- Same letters
  (numLetters * numDigits * 1 * 1)             -- Both pairs same

theorem valid_plate_count :
  countValidPlates = 9100 := by
  sorry

#eval countValidPlates  -- Should output 9100

end NUMINAMATH_CALUDE_valid_plate_count_l496_49675


namespace NUMINAMATH_CALUDE_range_equality_odd_decreasing_function_l496_49641

-- Statement 1
theorem range_equality (f : ℝ → ℝ) : Set.range f = Set.range (fun x ↦ f (x + 1)) := by sorry

-- Statement 3
theorem odd_decreasing_function (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_decreasing_neg : ∀ x y, x < y → y < 0 → f y < f x) : 
  ∀ x y, 0 < x → x < y → f y < f x := by sorry

end NUMINAMATH_CALUDE_range_equality_odd_decreasing_function_l496_49641


namespace NUMINAMATH_CALUDE_owlHootsPerMinute_l496_49667

/-- The number of hoot sounds one barnyard owl makes per minute, given that 3 owls together make 5 less than 20 hoots per minute. -/
def owlHoots : ℕ :=
  let totalHoots : ℕ := 20 - 5
  let numOwls : ℕ := 3
  totalHoots / numOwls

/-- Theorem stating that one barnyard owl makes 5 hoot sounds per minute under the given conditions. -/
theorem owlHootsPerMinute : owlHoots = 5 := by
  sorry

end NUMINAMATH_CALUDE_owlHootsPerMinute_l496_49667


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l496_49600

theorem simplify_and_evaluate (x : ℝ) (h : x = 1 / (Real.sqrt 3 - 2)) :
  x^2 + 4*x - 4 = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l496_49600


namespace NUMINAMATH_CALUDE_cos_A_minus_B_l496_49604

theorem cos_A_minus_B (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2)
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = -3/8 := by
sorry

end NUMINAMATH_CALUDE_cos_A_minus_B_l496_49604


namespace NUMINAMATH_CALUDE_translation_problem_l496_49645

def complex_translation (z w : ℂ) : ℂ := z + w

theorem translation_problem (t : ℂ → ℂ) :
  (t (1 + 3*I) = -2 + 4*I) →
  (∃ w : ℂ, ∀ z : ℂ, t z = complex_translation z w) →
  (t (3 + 7*I) = 8*I) :=
by
  sorry

end NUMINAMATH_CALUDE_translation_problem_l496_49645


namespace NUMINAMATH_CALUDE_base_three_five_digits_l496_49663

theorem base_three_five_digits : ∃! b : ℕ, b ≥ 2 ∧ b^4 ≤ 200 ∧ 200 < b^5 := by sorry

end NUMINAMATH_CALUDE_base_three_five_digits_l496_49663


namespace NUMINAMATH_CALUDE_sports_competition_theorem_l496_49683

-- Part a
def highest_average_rank (num_athletes : ℕ) (num_judges : ℕ) (max_rank_diff : ℕ) : ℚ :=
  8/3

-- Part b
def highest_winner_rank (num_players : ℕ) (max_rank_diff : ℕ) : ℕ :=
  21

theorem sports_competition_theorem :
  (highest_average_rank 20 9 3 = 8/3) ∧
  (highest_winner_rank 1024 2 = 21) :=
by sorry

end NUMINAMATH_CALUDE_sports_competition_theorem_l496_49683


namespace NUMINAMATH_CALUDE_arccos_sin_five_equals_five_minus_pi_over_two_l496_49605

theorem arccos_sin_five_equals_five_minus_pi_over_two :
  Real.arccos (Real.sin 5) = (5 - Real.pi) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_five_equals_five_minus_pi_over_two_l496_49605


namespace NUMINAMATH_CALUDE_plastic_for_one_ruler_l496_49662

/-- The amount of plastic needed to make one ruler, given the total amount of plastic and the number of rulers that can be made. -/
def plastic_per_ruler (total_plastic : ℕ) (num_rulers : ℕ) : ℚ :=
  (total_plastic : ℚ) / (num_rulers : ℚ)

/-- Theorem stating that 8 grams of plastic are needed to make one ruler. -/
theorem plastic_for_one_ruler :
  plastic_per_ruler 828 103 = 8 := by
  sorry

end NUMINAMATH_CALUDE_plastic_for_one_ruler_l496_49662


namespace NUMINAMATH_CALUDE_kendra_shirts_theorem_l496_49648

/-- Represents the number of shirts Kendra needs for various activities --/
structure ShirtRequirements where
  weekdaySchool : Nat
  afterSchoolClub : Nat
  spiritDay : Nat
  saturday : Nat
  sunday : Nat
  familyReunion : Nat

/-- Calculates the total number of shirts needed for a given number of weeks --/
def totalShirtsNeeded (req : ShirtRequirements) (weeks : Nat) : Nat :=
  (req.weekdaySchool + req.afterSchoolClub + req.spiritDay + req.saturday + req.sunday) * weeks + req.familyReunion

/-- Theorem stating that Kendra needs 61 shirts for 4 weeks --/
theorem kendra_shirts_theorem (req : ShirtRequirements) 
    (h1 : req.weekdaySchool = 5)
    (h2 : req.afterSchoolClub = 3)
    (h3 : req.spiritDay = 1)
    (h4 : req.saturday = 3)
    (h5 : req.sunday = 3)
    (h6 : req.familyReunion = 1) :
  totalShirtsNeeded req 4 = 61 := by
  sorry

#eval totalShirtsNeeded ⟨5, 3, 1, 3, 3, 1⟩ 4

end NUMINAMATH_CALUDE_kendra_shirts_theorem_l496_49648


namespace NUMINAMATH_CALUDE_creature_dressing_order_l496_49625

-- Define the number of arms
def num_arms : ℕ := 6

-- Define the number of items per arm
def items_per_arm : ℕ := 3

-- Define the total number of items
def total_items : ℕ := num_arms * items_per_arm

-- Define the number of valid permutations per arm (1 out of 6)
def valid_perm_per_arm : ℕ := 1

-- Define the total number of permutations per arm
def total_perm_per_arm : ℕ := Nat.factorial items_per_arm

-- Theorem statement
theorem creature_dressing_order :
  (Nat.factorial total_items) / (total_perm_per_arm ^ num_arms) =
  (Nat.factorial total_items) / (3 ^ num_arms) :=
sorry

end NUMINAMATH_CALUDE_creature_dressing_order_l496_49625


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l496_49682

/-- The volume of a tetrahedron given its edge lengths -/
def tetrahedron_volume (PQ PR PS QR QS RS : ℝ) : ℝ :=
  -- Define the volume calculation here
  sorry

/-- Theorem: The volume of tetrahedron PQRS with given edge lengths is 15√2 / 2 -/
theorem volume_of_specific_tetrahedron :
  let PQ : ℝ := 6
  let PR : ℝ := 4
  let PS : ℝ := 5
  let QR : ℝ := 5
  let QS : ℝ := 3
  let RS : ℝ := 15 / 4 * Real.sqrt 2
  tetrahedron_volume PQ PR PS QR QS RS = 15 / 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l496_49682


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l496_49619

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the sum of the first and third terms is 8,
    prove that the second term is 4. -/
theorem arithmetic_sequence_second_term
  (a : ℕ → ℝ)
  (h_arithmetic : IsArithmeticSequence a)
  (h_sum : a 0 + a 2 = 8) :
  a 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l496_49619


namespace NUMINAMATH_CALUDE_irreducible_fractions_l496_49611

theorem irreducible_fractions (n : ℕ) : 
  (Nat.gcd (2*n + 13) (n + 7) = 1) ∧ 
  (Nat.gcd (2*n^2 - 1) (n + 1) = 1) ∧ 
  (Nat.gcd (n^2 - n + 1) (n^2 + 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fractions_l496_49611


namespace NUMINAMATH_CALUDE_angle_Q_measure_l496_49628

-- Define a scalene triangle PQR
structure ScaleneTriangle where
  P : ℝ
  Q : ℝ
  R : ℝ
  scalene : P ≠ Q ∧ Q ≠ R ∧ R ≠ P
  sum_180 : P + Q + R = 180

-- Theorem statement
theorem angle_Q_measure (t : ScaleneTriangle) 
  (h1 : t.Q = 2 * t.P) 
  (h2 : t.R = 3 * t.P) : 
  t.Q = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_Q_measure_l496_49628


namespace NUMINAMATH_CALUDE_adult_meals_calculation_l496_49626

/-- Given a ratio of kids meals to adult meals and the number of kids meals sold,
    calculate the number of adult meals sold. -/
def adult_meals_sold (kids_ratio : ℕ) (adult_ratio : ℕ) (kids_meals : ℕ) : ℕ :=
  (adult_ratio * kids_meals) / kids_ratio

/-- Theorem stating that given the specific ratio and number of kids meals,
    the number of adult meals sold is 49. -/
theorem adult_meals_calculation :
  adult_meals_sold 10 7 70 = 49 := by
  sorry

#eval adult_meals_sold 10 7 70

end NUMINAMATH_CALUDE_adult_meals_calculation_l496_49626


namespace NUMINAMATH_CALUDE_student_pet_difference_l496_49661

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 24

/-- The number of rabbits in each classroom -/
def rabbits_per_classroom : ℕ := 2

/-- The number of hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 3

/-- Theorem: The difference between the total number of students and the total number of pets
    in all fourth-grade classrooms is 95 -/
theorem student_pet_difference :
  num_classrooms * students_per_classroom - 
  (num_classrooms * rabbits_per_classroom + num_classrooms * hamsters_per_classroom) = 95 := by
  sorry

end NUMINAMATH_CALUDE_student_pet_difference_l496_49661


namespace NUMINAMATH_CALUDE_park_fencing_cost_l496_49646

/-- Represents the dimensions and fencing costs of a park with a flower bed -/
structure ParkWithFlowerBed where
  park_ratio : Rat -- Ratio of park's length to width
  park_area : ℝ -- Area of the park in square meters
  park_fence_cost : ℝ -- Cost of fencing the park per meter
  flowerbed_fence_cost : ℝ -- Cost of fencing the flower bed per meter

/-- Calculates the total fencing cost for a park with a flower bed -/
def total_fencing_cost (p : ParkWithFlowerBed) : ℝ :=
  sorry

/-- Theorem stating the total fencing cost for the given park configuration -/
theorem park_fencing_cost :
  let p : ParkWithFlowerBed := {
    park_ratio := 3/2,
    park_area := 3750,
    park_fence_cost := 0.70,
    flowerbed_fence_cost := 0.90
  }
  total_fencing_cost p = 245.65 := by
  sorry

end NUMINAMATH_CALUDE_park_fencing_cost_l496_49646


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l496_49606

theorem inequality_and_equality_condition (a b c : ℝ) :
  (5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * a * c + 4 * b * c) ∧
  (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * a * c + 4 * b * c ↔ a = 0 ∧ b = 0 ∧ c = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l496_49606


namespace NUMINAMATH_CALUDE_beidou_usage_scientific_notation_l496_49647

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℕ) : ℝ × ℤ :=
  sorry

theorem beidou_usage_scientific_notation :
  scientific_notation 360000000000 = (3.6, 11) :=
sorry

end NUMINAMATH_CALUDE_beidou_usage_scientific_notation_l496_49647


namespace NUMINAMATH_CALUDE_last_digit_of_large_prime_l496_49671

theorem last_digit_of_large_prime (h : 859433 = 214858 * 4 + 1) :
  (2^859433 - 1) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_large_prime_l496_49671


namespace NUMINAMATH_CALUDE_percentage_difference_l496_49659

theorem percentage_difference (x y : ℝ) (h : x = 7 * y) :
  (x - y) / x * 100 = (6 / 7) * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l496_49659


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_18_l496_49635

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_18 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 18 → n ≤ 990 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_18_l496_49635


namespace NUMINAMATH_CALUDE_books_sold_l496_49615

theorem books_sold (initial_books : ℕ) (remaining_books : ℕ) : initial_books = 136 → remaining_books = 27 → initial_books - remaining_books = 109 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l496_49615


namespace NUMINAMATH_CALUDE_complex_multiplication_l496_49610

/-- Given that i is the imaginary unit, prove that (2+i)(1-3i) = 5-5i -/
theorem complex_multiplication (i : ℂ) (hi : i * i = -1) :
  (2 + i) * (1 - 3*i) = 5 - 5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l496_49610


namespace NUMINAMATH_CALUDE_six_and_negative_six_are_opposite_l496_49630

/-- Two real numbers are opposite if one is the negative of the other -/
def are_opposite (a b : ℝ) : Prop := b = -a

/-- 6 and -6 are opposite numbers -/
theorem six_and_negative_six_are_opposite : are_opposite 6 (-6) := by
  sorry

end NUMINAMATH_CALUDE_six_and_negative_six_are_opposite_l496_49630


namespace NUMINAMATH_CALUDE_box_balls_problem_l496_49681

theorem box_balls_problem (balls : ℕ) (x : ℕ) : 
  balls = 57 → 
  (balls - x = 70 - balls) →
  x = 44 := by
  sorry

end NUMINAMATH_CALUDE_box_balls_problem_l496_49681


namespace NUMINAMATH_CALUDE_absolute_value_inequality_implies_a_geq_two_l496_49636

theorem absolute_value_inequality_implies_a_geq_two :
  (∀ x : ℝ, |x + 3| - |x + 1| - 2*a + 2 < 0) → a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_implies_a_geq_two_l496_49636


namespace NUMINAMATH_CALUDE_cookies_eaten_difference_l496_49689

theorem cookies_eaten_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) :
  initial_sweet = 8 →
  initial_salty = 6 →
  eaten_sweet = 20 →
  eaten_salty = 34 →
  eaten_salty - eaten_sweet = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_difference_l496_49689


namespace NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l496_49618

-- Define the circles and lines
def circle1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 10
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 11 = 0
def line1 (x y : ℝ) : Prop := y = 2*x - 3
def line2 (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0
def circle3 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle4 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Theorem for the first circle
theorem circle1_properties :
  (∀ x y : ℝ, circle1 x y → ((x = 5 ∧ y = 2) ∨ (x = 3 ∧ y = 2))) ∧
  (∃ x y : ℝ, circle1 x y ∧ line1 x y) :=
sorry

-- Theorem for the second circle
theorem circle2_properties :
  (∀ x y : ℝ, (circle3 x y ∧ circle4 x y) → circle2 x y) ∧
  (∃ x y : ℝ, circle2 x y ∧ line2 x y) :=
sorry

end NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l496_49618


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l496_49680

theorem matrix_multiplication_result :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 0; 7, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -1; 0, 2]
  A * B = !![15, -3; 35, -11] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l496_49680


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l496_49609

theorem quadratic_no_real_roots : ∀ x : ℝ, x^2 - 2*x + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l496_49609


namespace NUMINAMATH_CALUDE_ac_over_b_squared_range_l496_49616

/-- Given an obtuse triangle ABC with sides a, b, c satisfying a < b < c
    and internal angles forming an arithmetic sequence,
    the value of ac/b^2 is strictly between 0 and 2/3. -/
theorem ac_over_b_squared_range (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a < b → b < c →
  0 < A → A < π → 0 < B → B < π → 0 < C → C < π →
  A + B + C = π →
  C > π / 2 →
  ∃ (k : ℝ), B - A = C - B ∧ B = k * A ∧ C = (k + 1) * A →
  0 < a * c / (b * b) ∧ a * c / (b * b) < 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ac_over_b_squared_range_l496_49616


namespace NUMINAMATH_CALUDE_football_players_l496_49607

theorem football_players (total : ℕ) (cricket : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 420)
  (h2 : cricket = 175)
  (h3 : both = 130)
  (h4 : neither = 50) :
  total - neither - (cricket - both) = 325 :=
sorry

end NUMINAMATH_CALUDE_football_players_l496_49607


namespace NUMINAMATH_CALUDE_infimum_of_function_over_D_l496_49694

-- Define the set D
def D : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≠ p.2 ∧ p.1 ^ p.2 = p.2 ^ p.1}

-- State the theorem
theorem infimum_of_function_over_D (α β : ℝ) (hα : α > 0) (hβ : β > 0) (hαβ : α ≤ β) :
  ∃ (inf : ℝ), inf = Real.exp (α + β) ∧
    ∀ (x y : ℝ), (x, y) ∈ D → inf ≤ x^α * y^β :=
sorry

end NUMINAMATH_CALUDE_infimum_of_function_over_D_l496_49694


namespace NUMINAMATH_CALUDE_discounted_shirt_price_l496_49634

/-- Given a shirt sold at a 30% discount for 560 units of currency,
    prove that the original price was 800 units of currency. -/
theorem discounted_shirt_price (discount_percent : ℝ) (discounted_price : ℝ) :
  discount_percent = 30 →
  discounted_price = 560 →
  (1 - discount_percent / 100) * 800 = discounted_price := by
sorry

end NUMINAMATH_CALUDE_discounted_shirt_price_l496_49634


namespace NUMINAMATH_CALUDE_final_state_is_green_l496_49699

/-- Represents the colors of chameleons -/
inductive Color
  | Yellow
  | Red
  | Green

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  yellow : Nat
  red : Nat
  green : Nat

/-- The initial state of chameleons -/
def initialState : ChameleonState :=
  { yellow := 7, red := 10, green := 17 }

/-- The total number of chameleons -/
def totalChameleons : Nat := 34

/-- Represents a color change event between two chameleons -/
def colorChange (c1 c2 : Color) : Color :=
  match c1, c2 with
  | Color.Yellow, Color.Red => Color.Green
  | Color.Yellow, Color.Green => Color.Red
  | Color.Red, Color.Yellow => Color.Green
  | Color.Red, Color.Green => Color.Yellow
  | Color.Green, Color.Yellow => Color.Red
  | Color.Green, Color.Red => Color.Yellow
  | _, _ => c1  -- No change if colors are the same

/-- Theorem: The only possible final state is all chameleons being green -/
theorem final_state_is_green (finalState : ChameleonState) :
  (finalState.yellow + finalState.red + finalState.green = totalChameleons) →
  (∀ (c1 c2 : Color), colorChange c1 c2 = colorChange c2 c1) →
  (finalState.yellow = 0 ∧ finalState.red = 0 ∧ finalState.green = totalChameleons) :=
by sorry

#check final_state_is_green

end NUMINAMATH_CALUDE_final_state_is_green_l496_49699


namespace NUMINAMATH_CALUDE_complex_simplification_l496_49669

theorem complex_simplification : (4 - 3*I)^2 + (1 + 2*I) = 8 - 22*I := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l496_49669


namespace NUMINAMATH_CALUDE_total_dress_designs_l496_49633

/-- Represents the number of fabric color choices --/
def num_colors : Nat := 5

/-- Represents the number of pattern choices --/
def num_patterns : Nat := 4

/-- Represents the number of sleeve length choices --/
def num_sleeve_lengths : Nat := 2

/-- Theorem stating the total number of possible dress designs --/
theorem total_dress_designs : num_colors * num_patterns * num_sleeve_lengths = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l496_49633


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l496_49624

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ 0}
def N : Set ℝ := {y | ∃ x, y = -x^2 + 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l496_49624


namespace NUMINAMATH_CALUDE_sixth_triangular_number_l496_49602

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := (n * (n + 1)) / 2

/-- The 6th triangular number is 21 -/
theorem sixth_triangular_number : triangular 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sixth_triangular_number_l496_49602


namespace NUMINAMATH_CALUDE_different_color_probability_l496_49692

/-- The probability of drawing two balls of different colors from a box with 2 red and 3 black balls -/
theorem different_color_probability : 
  let total_balls : ℕ := 2 + 3
  let red_balls : ℕ := 2
  let black_balls : ℕ := 3
  let different_color_draws : ℕ := red_balls * black_balls
  let total_draws : ℕ := (total_balls * (total_balls - 1)) / 2
  (different_color_draws : ℚ) / total_draws = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l496_49692


namespace NUMINAMATH_CALUDE_add_negative_and_positive_l496_49658

theorem add_negative_and_positive : -3 + 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_add_negative_and_positive_l496_49658


namespace NUMINAMATH_CALUDE_algebraic_identities_l496_49603

theorem algebraic_identities (a b c : ℝ) : 
  (a^4 * (a^2)^3 = a^10) ∧ 
  (2*a^3*b^2*c / ((1/3)*a^2*b) = 6*a*b*c) ∧ 
  (6*a*((1/3)*a*b - b) - (2*a*b + b)*(a - 1) = -5*a*b + b) ∧ 
  ((a - 2)^2 - (3*a + 2*b)*(3*a - 2*b) = -8*a^2 - 4*a + 4 + 4*b^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l496_49603


namespace NUMINAMATH_CALUDE_exactly_two_sunny_days_probability_l496_49668

theorem exactly_two_sunny_days_probability 
  (num_days : ℕ) 
  (rain_prob : ℝ) 
  (sunny_prob : ℝ) :
  num_days = 3 →
  rain_prob = 0.6 →
  sunny_prob = 1 - rain_prob →
  (num_days.choose 2 : ℝ) * sunny_prob^2 * rain_prob = 54/125 :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_sunny_days_probability_l496_49668


namespace NUMINAMATH_CALUDE_max_k_minus_m_is_neg_sqrt_two_l496_49688

/-- A point on a parabola with complementary lines intersecting the parabola -/
structure ParabolaPoint where
  m : ℝ
  k : ℝ
  h1 : m > 0  -- First quadrant condition
  h2 : k = 1 / (-2 * m)  -- Derived from the problem

/-- The maximum value of k - m for a point on the parabola -/
def max_k_minus_m (p : ParabolaPoint) : ℝ := p.k - p.m

/-- Theorem: The maximum value of k - m is -√2 -/
theorem max_k_minus_m_is_neg_sqrt_two :
  ∃ (p : ParabolaPoint), ∀ (q : ParabolaPoint), max_k_minus_m p ≥ max_k_minus_m q ∧ 
  max_k_minus_m p = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_k_minus_m_is_neg_sqrt_two_l496_49688


namespace NUMINAMATH_CALUDE_water_transfer_theorem_l496_49672

/-- Represents a water canister with a given capacity and current water level. -/
structure Canister where
  capacity : ℝ
  water : ℝ
  h_water_nonneg : 0 ≤ water
  h_water_le_capacity : water ≤ capacity

/-- The result of pouring water from one canister to another. -/
structure PourResult where
  source : Canister
  target : Canister

theorem water_transfer_theorem (c d : Canister) 
  (h_c_half_full : c.water = c.capacity / 2)
  (h_d_capacity : d.capacity = 2 * c.capacity)
  (h_d_third_full : d.water = d.capacity / 3)
  : ∃ (result : PourResult), 
    result.target.water = result.target.capacity ∧ 
    result.source.water = result.source.capacity / 12 := by
  sorry

end NUMINAMATH_CALUDE_water_transfer_theorem_l496_49672


namespace NUMINAMATH_CALUDE_binary_101101_equals_45_l496_49621

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_45_l496_49621


namespace NUMINAMATH_CALUDE_triangle_area_l496_49653

theorem triangle_area (a b : ℝ) (cos_theta : ℝ) : 
  a = 3 → 
  b = 5 → 
  5 * cos_theta^2 - 7 * cos_theta - 6 = 0 →
  abs (1/2 * a * b * cos_theta) = 4.5 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l496_49653


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l496_49687

/-- The number of ways to choose 4 socks from 6 socks (where one is blue and the rest are different colors), 
    such that at least one chosen sock is blue. -/
def choose_socks (total_socks : ℕ) (blue_socks : ℕ) (choose : ℕ) : ℕ :=
  Nat.choose (total_socks - blue_socks) (choose - 1)

/-- Theorem stating that there are 10 ways to choose 4 socks from 6 socks, 
    where at least one is blue. -/
theorem sock_selection_theorem :
  choose_socks 6 1 4 = 10 := by
  sorry

#eval choose_socks 6 1 4

end NUMINAMATH_CALUDE_sock_selection_theorem_l496_49687


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l496_49638

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 20) :
  let side_length := face_perimeter / 4
  (side_length ^ 3) = 125 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l496_49638


namespace NUMINAMATH_CALUDE_yellow_beads_proof_l496_49617

theorem yellow_beads_proof (green_beads : ℕ) (yellow_fraction : ℚ) : 
  green_beads = 4 → 
  yellow_fraction = 4/5 → 
  (yellow_fraction * (green_beads + 16 : ℚ)).num = 16 := by
  sorry

end NUMINAMATH_CALUDE_yellow_beads_proof_l496_49617


namespace NUMINAMATH_CALUDE_problem_solution_l496_49677

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1/p + 1/q = 1) 
  (h4 : p*q = 12) : 
  q = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l496_49677


namespace NUMINAMATH_CALUDE_Φ_is_connected_l496_49652

-- Define the set Φ
def Φ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               Real.sqrt (y^2 - 8*x^2 - 6*y + 9) ≤ 3*y - 1 ∧
               x^2 + y^2 ≤ 9}

-- Theorem statement
theorem Φ_is_connected : IsConnected Φ := by
  sorry

end NUMINAMATH_CALUDE_Φ_is_connected_l496_49652


namespace NUMINAMATH_CALUDE_apple_rate_problem_l496_49643

theorem apple_rate_problem (apple_rate : ℕ) : 
  (8 * apple_rate + 9 * 75 = 1235) → apple_rate = 70 := by
  sorry

end NUMINAMATH_CALUDE_apple_rate_problem_l496_49643


namespace NUMINAMATH_CALUDE_seventh_term_is_64_l496_49613

/-- A geometric sequence with given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) * a n = (a (n + 1))^2
  sum_first_two : a 1 + a 2 = 3
  sum_second_third : a 2 + a 3 = 6

/-- The 7th term of the geometric sequence is 64 -/
theorem seventh_term_is_64 (seq : GeometricSequence) : seq.a 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_64_l496_49613


namespace NUMINAMATH_CALUDE_xy_value_given_condition_l496_49632

theorem xy_value_given_condition (x y : ℝ) : 
  |x - 2| + Real.sqrt (y + 3) = 0 → x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_given_condition_l496_49632


namespace NUMINAMATH_CALUDE_logger_productivity_l496_49660

/-- Represents the number of trees one logger can cut down per day -/
def trees_per_logger_per_day (forest_length : ℕ) (forest_width : ℕ) (trees_per_square_mile : ℕ) 
  (days_per_month : ℕ) (num_loggers : ℕ) (num_months : ℕ) : ℕ :=
  let total_trees := forest_length * forest_width * trees_per_square_mile
  let total_days := num_months * days_per_month
  total_trees / (num_loggers * total_days)

theorem logger_productivity : 
  trees_per_logger_per_day 4 6 600 30 8 10 = 6 := by
  sorry

#eval trees_per_logger_per_day 4 6 600 30 8 10

end NUMINAMATH_CALUDE_logger_productivity_l496_49660


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l496_49686

theorem z_in_first_quadrant (z : ℂ) : 
  (z - 2*I) * (1 + I) = Complex.abs (1 - Real.sqrt 3 * I) → 
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l496_49686


namespace NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l496_49654

/-- The number of distinct arrangements of n distinct beads on a necklace, 
    considering rotations and reflections as identical -/
def necklace_arrangements (n : ℕ) : ℕ := Nat.factorial n / (n * 2)

/-- Theorem stating that for 8 distinct beads, the number of distinct necklace arrangements is 2520 -/
theorem eight_bead_necklace_arrangements : 
  necklace_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l496_49654


namespace NUMINAMATH_CALUDE_new_person_weight_l496_49695

theorem new_person_weight (n : ℕ) (initial_avg : ℝ) (weight_decrease : ℝ) :
  n = 20 →
  initial_avg = 58 →
  weight_decrease = 5 →
  let total_weight := n * initial_avg
  let new_avg := initial_avg - weight_decrease
  let new_person_weight := total_weight - (n + 1) * new_avg
  new_person_weight = 47 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l496_49695


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l496_49664

def U : Set Nat := {x | x > 0 ∧ x < 9}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {3, 4, 5, 6}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l496_49664


namespace NUMINAMATH_CALUDE_geometric_mean_inequality_l496_49629

theorem geometric_mean_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let g := Real.sqrt (x * y)
  (g ≥ 3 → 1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≥ 2 / Real.sqrt (1 + g)) ∧
  (g ≤ 2 → 1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≤ 2 / Real.sqrt (1 + g)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_inequality_l496_49629


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l496_49684

theorem arithmetic_mean_of_fractions :
  let a := 3 / 5
  let b := 5 / 7
  let c := 9 / 14
  let arithmetic_mean := (a + b) / 2
  arithmetic_mean = 23 / 35 ∧ arithmetic_mean ≠ c := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l496_49684


namespace NUMINAMATH_CALUDE_same_color_probability_l496_49678

/-- The probability of drawing two balls of the same color from a bag with green and white balls -/
theorem same_color_probability (green white : ℕ) (h : green = 5 ∧ white = 9) :
  let total := green + white
  let p_green := green / total
  let p_white := white / total
  let p_same_color := p_green * ((green - 1) / (total - 1)) + p_white * ((white - 1) / (total - 1))
  p_same_color = 46 / 91 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l496_49678


namespace NUMINAMATH_CALUDE_container_volume_ratio_l496_49655

theorem container_volume_ratio (A B C : ℚ) 
  (h1 : (3 : ℚ) / 4 * A = (2 : ℚ) / 3 * B) 
  (h2 : (2 : ℚ) / 3 * B = (1 : ℚ) / 2 * C) : 
  A / C = (2 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l496_49655


namespace NUMINAMATH_CALUDE_norma_cards_l496_49685

/-- 
Given that Norma loses 70 cards and has 18 cards left,
prove that she initially had 88 cards.
-/
theorem norma_cards : 
  ∀ (initial_cards : ℕ),
  (initial_cards - 70 = 18) → initial_cards = 88 := by
  sorry

end NUMINAMATH_CALUDE_norma_cards_l496_49685


namespace NUMINAMATH_CALUDE_brownie_division_l496_49612

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the pan of brownies -/
def pan : Dimensions := ⟨24, 15⟩

/-- Represents a single piece of brownie -/
def piece : Dimensions := ⟨3, 2⟩

/-- Theorem stating that the pan can be divided into exactly 60 pieces -/
theorem brownie_division :
  (area pan) / (area piece) = 60 := by sorry

end NUMINAMATH_CALUDE_brownie_division_l496_49612


namespace NUMINAMATH_CALUDE_root_implies_u_value_l496_49674

theorem root_implies_u_value (u : ℝ) : 
  (6 * ((-25 - Real.sqrt 421) / 12)^2 + 25 * ((-25 - Real.sqrt 421) / 12) + u = 0) → 
  u = 8.5 := by
sorry

end NUMINAMATH_CALUDE_root_implies_u_value_l496_49674


namespace NUMINAMATH_CALUDE_infinite_sum_equals_five_twentyfourths_l496_49620

/-- The sum of the infinite series n / (n^4 - 4n^2 + 8) from n = 1 to infinity is equal to 5/24. -/
theorem infinite_sum_equals_five_twentyfourths :
  ∑' n : ℕ+, (n : ℝ) / ((n : ℝ)^4 - 4*(n : ℝ)^2 + 8) = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_five_twentyfourths_l496_49620


namespace NUMINAMATH_CALUDE_colored_tape_length_l496_49601

theorem colored_tape_length : 
  ∀ (original_length : ℝ),
  (1 / 5 : ℝ) * original_length + -- Used for art
  (3 / 4 : ℝ) * (4 / 5 : ℝ) * original_length + -- Given away
  1.5 = original_length → -- Remaining length
  original_length = 7.5 := by
sorry

end NUMINAMATH_CALUDE_colored_tape_length_l496_49601


namespace NUMINAMATH_CALUDE_roses_planted_l496_49650

theorem roses_planted (day1 day2 day3 : ℕ) : 
  day2 = day1 + 20 →
  day3 = 2 * day1 →
  day1 + day2 + day3 = 220 →
  day1 = 50 := by
sorry

end NUMINAMATH_CALUDE_roses_planted_l496_49650


namespace NUMINAMATH_CALUDE_a_power_b_minus_a_power_neg_b_l496_49691

theorem a_power_b_minus_a_power_neg_b (a b : ℝ) (ha : a > 1) (hb : b > 0) 
  (h : a^b + a^(-b) = 2 * Real.sqrt 2) : a^b - a^(-b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_power_b_minus_a_power_neg_b_l496_49691


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_l496_49649

theorem similar_triangles_perimeter (p_small p_large : ℝ) : 
  p_small > 0 → 
  p_large > 0 → 
  p_small / p_large = 2 / 3 → 
  p_small + p_large = 20 → 
  p_small = 8 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_l496_49649


namespace NUMINAMATH_CALUDE_factorization_equality_l496_49656

theorem factorization_equality (a b : ℝ) : a^3 - 2*a^2*b + a*b^2 = a*(a-b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l496_49656


namespace NUMINAMATH_CALUDE_residue_calculation_l496_49657

theorem residue_calculation : (222 * 15 - 35 * 9 + 2^3) % 18 = 17 := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l496_49657


namespace NUMINAMATH_CALUDE_root_equation_m_value_l496_49697

theorem root_equation_m_value (x m : ℝ) : 
  (3 / x = m / (x - 3)) → (x = 6) → (m = 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_root_equation_m_value_l496_49697


namespace NUMINAMATH_CALUDE_bus_overlap_count_l496_49673

-- Define the bus schedules
def busA_interval : ℕ := 6
def busB_interval : ℕ := 10
def busC_interval : ℕ := 14

-- Define the time range in minutes (5:00 PM to 10:00 PM)
def start_time : ℕ := 240  -- 4 hours after 1:00 PM
def end_time : ℕ := 540    -- 9 hours after 1:00 PM

-- Function to calculate the number of overlaps between two buses
def count_overlaps (interval1 interval2 start_time end_time : ℕ) : ℕ :=
  (end_time - start_time) / Nat.lcm interval1 interval2 + 1

-- Function to calculate the total number of distinct overlaps
def total_distinct_overlaps (start_time end_time : ℕ) : ℕ :=
  let ab_overlaps := count_overlaps busA_interval busB_interval start_time end_time
  let bc_overlaps := count_overlaps busB_interval busC_interval start_time end_time
  let ac_overlaps := count_overlaps busA_interval busC_interval start_time end_time
  ab_overlaps + bc_overlaps + ac_overlaps - 2  -- Subtracting 2 for common overlaps

-- The main theorem
theorem bus_overlap_count : 
  total_distinct_overlaps start_time end_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_bus_overlap_count_l496_49673


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l496_49696

/-- Given an arithmetic sequence where the fifth term is 15 and the common difference is 2,
    prove that the product of the first two terms is 63. -/
theorem arithmetic_sequence_product (a : ℕ → ℕ) :
  (∀ n, a (n + 1) = a n + 2) →  -- Common difference is 2
  a 5 = 15 →                    -- Fifth term is 15
  a 1 * a 2 = 63 :=              -- Product of first two terms is 63
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l496_49696


namespace NUMINAMATH_CALUDE_membership_change_l496_49639

theorem membership_change (initial_members : ℝ) : 
  let fall_increase := 0.07
  let spring_decrease := 0.19
  let fall_members := initial_members * (1 + fall_increase)
  let spring_members := fall_members * (1 - spring_decrease)
  let total_change_percentage := (spring_members / initial_members - 1) * 100
  total_change_percentage = -13.33 := by
sorry

end NUMINAMATH_CALUDE_membership_change_l496_49639


namespace NUMINAMATH_CALUDE_opposite_of_five_l496_49622

theorem opposite_of_five : 
  -(5 : ℤ) = -5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_five_l496_49622


namespace NUMINAMATH_CALUDE_complex_number_simplification_l496_49676

theorem complex_number_simplification (i : ℂ) : 
  i * i = -1 → 
  (4 * i) / ((1 - i)^2 + 2) + i^2018 = -2 + i := by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l496_49676


namespace NUMINAMATH_CALUDE_mean_home_runs_l496_49690

theorem mean_home_runs : 
  let total_players : ℕ := 2 + 3 + 2 + 1 + 1
  let total_home_runs : ℕ := 2 * 5 + 3 * 6 + 2 * 8 + 1 * 9 + 1 * 11
  (total_home_runs : ℚ) / total_players = 64 / 9 := by sorry

end NUMINAMATH_CALUDE_mean_home_runs_l496_49690


namespace NUMINAMATH_CALUDE_f_inequality_solutions_l496_49670

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (m^2 + 1) * x + m

theorem f_inequality_solutions :
  (∀ x, f 2 x ≤ 0 ↔ 1/2 ≤ x ∧ x ≤ 2) ∧
  (∀ m, m > 0 →
    (0 < m ∧ m < 1 →
      (∀ x, f m x > 0 ↔ x < m ∨ x > 1/m)) ∧
    (m = 1 →
      (∀ x, f m x > 0 ↔ x ≠ 1)) ∧
    (m > 1 →
      (∀ x, f m x > 0 ↔ x < 1/m ∨ x > m))) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_solutions_l496_49670


namespace NUMINAMATH_CALUDE_six_digit_number_theorem_l496_49637

def is_valid_six_digit_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def extract_digits (n : ℕ) : Fin 6 → ℕ
| 0 => n / 100000
| 1 => (n / 10000) % 10
| 2 => (n / 1000) % 10
| 3 => (n / 100) % 10
| 4 => (n / 10) % 10
| 5 => n % 10

theorem six_digit_number_theorem (n : ℕ) (hn : is_valid_six_digit_number n) :
  (extract_digits n 0 = 1) →
  (3 * n = (n % 100000) * 10 + 1) →
  (extract_digits n 1 + extract_digits n 2 + extract_digits n 3 + 
   extract_digits n 4 + extract_digits n 5 = 26) := by
  sorry

end NUMINAMATH_CALUDE_six_digit_number_theorem_l496_49637


namespace NUMINAMATH_CALUDE_solution_set_for_m_eq_3_min_m_value_l496_49640

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |m * x + 1| + |2 * x - 1|

-- Part I
theorem solution_set_for_m_eq_3 :
  {x : ℝ | f 3 x > 4} = {x : ℝ | x < -4/5 ∨ x > 4/5} :=
sorry

-- Part II
theorem min_m_value (m : ℝ) (h1 : 0 < m) (h2 : m < 2) 
  (h3 : ∀ x : ℝ, f m x ≥ 3 / (2 * m)) :
  m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_m_eq_3_min_m_value_l496_49640


namespace NUMINAMATH_CALUDE_train_length_l496_49631

/-- Calculates the length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 108 →
  platform_length = 300.06 →
  crossing_time = 25 →
  (train_speed * 1000 / 3600 * crossing_time) - platform_length = 449.94 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l496_49631


namespace NUMINAMATH_CALUDE_inequality_proof_l496_49623

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l496_49623


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l496_49642

theorem junk_mail_distribution (total : ℕ) (blocks : ℕ) (first : ℕ) (second : ℕ) 
  (h1 : total = 2758)
  (h2 : blocks = 5)
  (h3 : first = 365)
  (h4 : second = 421) :
  (total - first - second) / (blocks - 2) = 657 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l496_49642


namespace NUMINAMATH_CALUDE_sachin_age_l496_49679

theorem sachin_age (sachin_age rahul_age : ℕ) 
  (age_difference : rahul_age = sachin_age + 7)
  (age_ratio : sachin_age * 9 = rahul_age * 6) :
  sachin_age = 14 := by sorry

end NUMINAMATH_CALUDE_sachin_age_l496_49679


namespace NUMINAMATH_CALUDE_removed_player_height_l496_49666

/-- The height of the removed player given the initial and final average heights -/
def height_of_removed_player (initial_avg : ℝ) (final_avg : ℝ) : ℝ :=
  11 * initial_avg - 10 * final_avg

/-- Theorem stating the height of the removed player -/
theorem removed_player_height :
  height_of_removed_player 182 181 = 192 := by
  sorry

#eval height_of_removed_player 182 181

end NUMINAMATH_CALUDE_removed_player_height_l496_49666


namespace NUMINAMATH_CALUDE_pencil_sharpening_l496_49608

/-- Given the initial and final lengths of a pencil, calculate the length sharpened off. -/
theorem pencil_sharpening (initial_length final_length : ℕ) : 
  initial_length ≥ final_length → 
  initial_length - final_length = initial_length - final_length :=
by sorry

end NUMINAMATH_CALUDE_pencil_sharpening_l496_49608
