import Mathlib

namespace NUMINAMATH_CALUDE_parabola_intersection_point_l2354_235406

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.evaluate (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_intersection_point (p : Parabola) (h1 : p.a = 1 ∧ p.b = -2 ∧ p.c = -3)
    (h2 : p.evaluate (-1) = 0) :
    p.evaluate 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_point_l2354_235406


namespace NUMINAMATH_CALUDE_f_properties_l2354_235410

noncomputable def f (x : ℝ) := x * Real.log x

theorem f_properties :
  (∀ x > 0, f x ≥ -1 / Real.exp 1) ∧
  (∀ t > 0, (∀ x ∈ Set.Icc t (t + 2), f x ≥ min (-1 / Real.exp 1) (f t))) ∧
  (∀ x > 0, Real.log x > 1 / (Real.exp x) - 2 / (Real.exp 1 * x)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2354_235410


namespace NUMINAMATH_CALUDE_no_real_solutions_l2354_235471

theorem no_real_solutions : 
  ∀ x : ℝ, (2 * x^2 - 3 * x + 5)^2 + 1 ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2354_235471


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l2354_235485

-- Define the cubic equation
def cubic_equation (x : ℝ) : Prop := x^3 - 6*x^2 + 11*x - 6 = 0

-- Define the roots of the equation
def roots (a b c : ℝ) : Prop := cubic_equation a ∧ cubic_equation b ∧ cubic_equation c

-- Theorem statement
theorem sum_of_reciprocal_squares (a b c : ℝ) :
  roots a b c → 1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l2354_235485


namespace NUMINAMATH_CALUDE_joseph_cards_l2354_235448

theorem joseph_cards (initial_cards : ℕ) (cards_to_friend : ℕ) (remaining_fraction : ℚ) : 
  initial_cards = 16 →
  cards_to_friend = 2 →
  remaining_fraction = 1/2 →
  (initial_cards - cards_to_friend - (remaining_fraction * initial_cards)) / initial_cards = 3/8 := by
sorry

end NUMINAMATH_CALUDE_joseph_cards_l2354_235448


namespace NUMINAMATH_CALUDE_carnival_tickets_l2354_235462

/-- The total number of tickets bought by a group of friends at a carnival. -/
def total_tickets (num_friends : ℕ) (tickets_per_friend : ℕ) : ℕ :=
  num_friends * tickets_per_friend

/-- Theorem stating that 6 friends buying 39 tickets each results in 234 total tickets. -/
theorem carnival_tickets : total_tickets 6 39 = 234 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_l2354_235462


namespace NUMINAMATH_CALUDE_five_digit_number_proof_l2354_235440

theorem five_digit_number_proof (x : ℕ) : 
  x ≥ 10000 ∧ x < 100000 ∧ 10 * x + 1 = 3 * (100000 + x) → x = 42857 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_number_proof_l2354_235440


namespace NUMINAMATH_CALUDE_cubic_polynomial_relation_l2354_235477

/-- Given a cubic polynomial f(x) = x^3 + 3x^2 + x + 1, and another cubic polynomial h
    such that h(0) = 1 and the roots of h are the cubes of the roots of f,
    prove that h(-8) = -115. -/
theorem cubic_polynomial_relation (f h : ℝ → ℝ) : 
  (∀ x, f x = x^3 + 3*x^2 + x + 1) →
  (∃ a b c : ℝ, ∀ x, h x = (x - a^3) * (x - b^3) * (x - c^3)) →
  h 0 = 1 →
  (∀ x, f x = 0 ↔ h (x^3) = 0) →
  h (-8) = -115 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_relation_l2354_235477


namespace NUMINAMATH_CALUDE_sara_oranges_l2354_235496

/-- Given that Joan picked 37 oranges, 47 oranges were picked in total, 
    and Alyssa picked 30 pears, prove that Sara picked 10 oranges. -/
theorem sara_oranges (joan_oranges : ℕ) (total_oranges : ℕ) (alyssa_pears : ℕ) 
    (h1 : joan_oranges = 37)
    (h2 : total_oranges = 47)
    (h3 : alyssa_pears = 30) : 
  total_oranges - joan_oranges = 10 := by
  sorry

end NUMINAMATH_CALUDE_sara_oranges_l2354_235496


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l2354_235429

/-- Calculates the time taken for a monkey to climb a tree given the tree height,
    distance hopped up per hour, and distance slipped back per hour. -/
def monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) : ℕ :=
  let net_distance := hop_distance - slip_distance
  let full_climb_distance := tree_height - hop_distance
  full_climb_distance / net_distance + 1

theorem monkey_climb_theorem :
  monkey_climb_time 19 3 2 = 17 :=
sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l2354_235429


namespace NUMINAMATH_CALUDE_decrypt_ciphertext_l2354_235451

/-- Represents the encryption rule --/
def encrypt (a b c d : ℤ) : ℤ × ℤ × ℤ × ℤ :=
  (a + 2*b, 2*b + c, 2*c + 3*d, 4*d)

/-- Represents the given ciphertext --/
def ciphertext : ℤ × ℤ × ℤ × ℤ := (14, 9, 23, 28)

/-- Theorem stating that the plaintext (6, 4, 1, 7) corresponds to the given ciphertext --/
theorem decrypt_ciphertext :
  encrypt 6 4 1 7 = ciphertext := by sorry

end NUMINAMATH_CALUDE_decrypt_ciphertext_l2354_235451


namespace NUMINAMATH_CALUDE_average_speed_to_destination_l2354_235488

/-- Proves that given a round trip with a total one-way distance of 150 km,
    a return speed of 30 km/hr, and an average speed for the whole journey of 37.5 km/hr,
    the average speed while traveling to the place is 50 km/hr. -/
theorem average_speed_to_destination (v : ℝ) : 
  (150 : ℝ) / v + (150 : ℝ) / 30 = 300 / (37.5 : ℝ) → v = 50 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_to_destination_l2354_235488


namespace NUMINAMATH_CALUDE_twin_primes_divisibility_l2354_235432

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem twin_primes_divisibility (a : ℤ) 
  (h1 : is_prime (a - 1).natAbs) 
  (h2 : is_prime (a + 1).natAbs) 
  (h3 : (a - 1).natAbs > 10) 
  (h4 : (a + 1).natAbs > 10) : 
  120 ∣ (a^3 - 4*a) :=
sorry

end NUMINAMATH_CALUDE_twin_primes_divisibility_l2354_235432


namespace NUMINAMATH_CALUDE_power_three_405_mod_13_l2354_235417

theorem power_three_405_mod_13 : 3^405 ≡ 1 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_power_three_405_mod_13_l2354_235417


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2354_235431

-- Define the quadratic function
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  (∀ x ∈ Set.Ioo 2 3, f b c x ≤ 1) →  -- Maximum value of 1 in (2,3]
  (∃ x ∈ Set.Ioo 2 3, f b c x = 1) →  -- Maximum value of 1 is achieved in (2,3]
  (∀ x : ℝ, abs x > 2 → f b c x > 0) →  -- f(x) > 0 when |x| > 2
  (c = 4 → b = -4) ∧  -- Part 1: When c = 4, b = -4
  (Set.Icc (-34/7) (-15/4) = {x | ∃ b c : ℝ, b + 1/c = x}) -- Part 2: Range of b + 1/c
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2354_235431


namespace NUMINAMATH_CALUDE_negative_integer_square_plus_self_equals_twelve_l2354_235413

theorem negative_integer_square_plus_self_equals_twelve (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_square_plus_self_equals_twelve_l2354_235413


namespace NUMINAMATH_CALUDE_equal_numbers_product_l2354_235484

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 ∧ 
  a = 12 ∧ 
  b = 22 ∧ 
  c = d → 
  c * d = 529 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l2354_235484


namespace NUMINAMATH_CALUDE_no_sequences_exist_l2354_235453

theorem no_sequences_exist : ¬ ∃ (a b : ℕ → ℝ), 
  (∀ n : ℕ, (3/2) * Real.pi ≤ a n ∧ a n ≤ b n) ∧
  (∀ n : ℕ, ∀ x : ℝ, 0 < x ∧ x < 1 → Real.cos (a n * x) + Real.cos (b n * x) ≥ -1 / n) := by
  sorry

end NUMINAMATH_CALUDE_no_sequences_exist_l2354_235453


namespace NUMINAMATH_CALUDE_net_marble_change_l2354_235430

def marble_transactions (initial : Int) (lost : Int) (found : Int) (traded_out : Int) (traded_in : Int) (gave_away : Int) (received : Int) : Int :=
  initial - lost + found - traded_out + traded_in - gave_away + received

theorem net_marble_change : 
  marble_transactions 20 16 8 5 9 3 4 = -3 := by
  sorry

end NUMINAMATH_CALUDE_net_marble_change_l2354_235430


namespace NUMINAMATH_CALUDE_economy_class_seats_count_l2354_235438

/-- Represents the seating configuration of an airplane -/
structure AirplaneSeating where
  first_class_seats : ℕ
  business_class_seats : ℕ
  economy_class_seats : ℕ
  first_class_occupied : ℕ
  business_class_occupied : ℕ

/-- Theorem stating the number of economy class seats in the given airplane configuration -/
theorem economy_class_seats_count (a : AirplaneSeating) 
  (h1 : a.first_class_seats = 10)
  (h2 : a.business_class_seats = 30)
  (h3 : a.first_class_occupied = 3)
  (h4 : a.business_class_occupied = 22)
  (h5 : a.first_class_occupied + a.business_class_occupied = a.economy_class_seats / 2)
  : a.economy_class_seats = 50 := by
  sorry

#check economy_class_seats_count

end NUMINAMATH_CALUDE_economy_class_seats_count_l2354_235438


namespace NUMINAMATH_CALUDE_algebra_test_average_l2354_235425

theorem algebra_test_average (total_average : ℝ) (male_count : ℕ) (female_average : ℝ) (female_count : ℕ) :
  total_average = 90 →
  male_count = 8 →
  female_average = 92 →
  female_count = 32 →
  (total_average * (male_count + female_count) - female_average * female_count) / male_count = 82 := by
  sorry

end NUMINAMATH_CALUDE_algebra_test_average_l2354_235425


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2354_235411

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (x : ℂ), (3 - 2 * i * x = 5 + 4 * i * x) ∧ (x = i / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2354_235411


namespace NUMINAMATH_CALUDE_candy_ratio_l2354_235469

theorem candy_ratio (kitkat : ℕ) (nerds : ℕ) (lollipops : ℕ) (babyruths : ℕ) (reeses : ℕ) (remaining : ℕ) :
  kitkat = 5 →
  nerds = 8 →
  lollipops = 11 →
  babyruths = 10 →
  reeses = babyruths / 2 →
  remaining = 49 →
  ∃ (hershey : ℕ),
    hershey + kitkat + nerds + (lollipops - 5) + babyruths + reeses = remaining ∧
    hershey / kitkat = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_l2354_235469


namespace NUMINAMATH_CALUDE_trisector_triangle_angles_l2354_235498

/-- Given a triangle ABC with angles α, β, and γ, if the triangle formed by the first angle trisectors
    has two angles of 45° and 55°, then the triangle formed by the second angle trisectors
    has angles of 40°, 65°, and 75°. -/
theorem trisector_triangle_angles 
  (α β γ : Real) 
  (h_sum : α + β + γ = 180)
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_first_trisector : 
    ((β + 2*γ)/3 = 45 ∧ (γ + 2*α)/3 = 55) ∨ 
    ((β + 2*γ)/3 = 55 ∧ (γ + 2*α)/3 = 45) ∨
    ((γ + 2*α)/3 = 45 ∧ (α + 2*β)/3 = 55) ∨
    ((γ + 2*α)/3 = 55 ∧ (α + 2*β)/3 = 45) ∨
    ((α + 2*β)/3 = 45 ∧ (β + 2*γ)/3 = 55) ∨
    ((α + 2*β)/3 = 55 ∧ (β + 2*γ)/3 = 45)) :
  (2*β + γ)/3 = 65 ∧ (2*γ + α)/3 = 40 ∧ (2*α + β)/3 = 75 := by
  sorry


end NUMINAMATH_CALUDE_trisector_triangle_angles_l2354_235498


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l2354_235465

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l2354_235465


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l2354_235476

theorem rectangle_measurement_error (x : ℝ) : 
  (1 + x / 100) * 0.95 = 102.6 / 100 → x = 8 := by sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l2354_235476


namespace NUMINAMATH_CALUDE_john_bought_three_puzzles_l2354_235443

/-- Represents the number of puzzles John bought -/
def num_puzzles : ℕ := 3

/-- Represents the number of pieces in the first puzzle -/
def first_puzzle_pieces : ℕ := 1000

/-- Represents the number of pieces in the second and third puzzles -/
def other_puzzle_pieces : ℕ := (3 * first_puzzle_pieces) / 2

/-- The total number of pieces in all puzzles -/
def total_pieces : ℕ := 4000

/-- Theorem stating that the number of puzzles John bought is 3 -/
theorem john_bought_three_puzzles :
  num_puzzles = 3 ∧
  first_puzzle_pieces = 1000 ∧
  other_puzzle_pieces = (3 * first_puzzle_pieces) / 2 ∧
  total_pieces = first_puzzle_pieces + 2 * other_puzzle_pieces :=
by sorry

end NUMINAMATH_CALUDE_john_bought_three_puzzles_l2354_235443


namespace NUMINAMATH_CALUDE_carols_birthday_invitations_l2354_235474

/-- Given that Carol buys invitation packages with 2 invitations each and needs 5 packs,
    prove that she is inviting 10 friends. -/
theorem carols_birthday_invitations
  (invitations_per_pack : ℕ)
  (packs_needed : ℕ)
  (h1 : invitations_per_pack = 2)
  (h2 : packs_needed = 5) :
  invitations_per_pack * packs_needed = 10 := by
  sorry

end NUMINAMATH_CALUDE_carols_birthday_invitations_l2354_235474


namespace NUMINAMATH_CALUDE_total_money_from_tshirts_l2354_235416

/-- The amount of money made from each t-shirt -/
def money_per_tshirt : ℕ := 215

/-- The number of t-shirts sold -/
def tshirts_sold : ℕ := 20

/-- The total money made from selling t-shirts -/
def total_money : ℕ := money_per_tshirt * tshirts_sold

theorem total_money_from_tshirts : total_money = 4300 := by
  sorry

end NUMINAMATH_CALUDE_total_money_from_tshirts_l2354_235416


namespace NUMINAMATH_CALUDE_range_of_m_for_nonempty_solution_l2354_235456

theorem range_of_m_for_nonempty_solution (m : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) → -5 ≤ m ∧ m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_nonempty_solution_l2354_235456


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l2354_235414

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

/-- A geometric sequence with positive terms -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q ∧ b n > 0

theorem arithmetic_geometric_inequality (a b : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) (h_geom : GeometricSequence b)
    (h_eq1 : a 1 = b 1) (h_eq2 : a 2 = b 2) (h_neq : a 1 ≠ a 2) :
    ∀ n : ℕ, n ≥ 3 → a n < b n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l2354_235414


namespace NUMINAMATH_CALUDE_cube_triangle_area_sum_l2354_235407

/-- Represents a 2x2x2 cube -/
structure Cube :=
  (side : ℝ)
  (is_two_by_two_by_two : side = 2)

/-- Represents a triangle with vertices on the cube -/
structure CubeTriangle :=
  (cube : Cube)
  (vertices : Fin 3 → Fin 8)

/-- The area of a triangle on the cube -/
noncomputable def triangleArea (t : CubeTriangle) : ℝ := sorry

/-- The sum of areas of all triangles on the cube -/
noncomputable def totalArea (c : Cube) : ℝ := sorry

/-- The representation of the total area in the form m + √n + √p -/
structure AreaRepresentation (c : Cube) :=
  (m n p : ℕ)
  (total_area_eq : totalArea c = m + Real.sqrt n + Real.sqrt p)

theorem cube_triangle_area_sum (c : Cube) (rep : AreaRepresentation c) :
  rep.m + rep.n + rep.p = 11584 := by sorry

end NUMINAMATH_CALUDE_cube_triangle_area_sum_l2354_235407


namespace NUMINAMATH_CALUDE_favorite_number_ratio_l2354_235401

theorem favorite_number_ratio :
  ∀ (misty_number glory_number : ℕ),
    glory_number = 450 →
    misty_number + glory_number = 600 →
    glory_number / misty_number = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_favorite_number_ratio_l2354_235401


namespace NUMINAMATH_CALUDE_geese_migration_rate_ratio_l2354_235422

/-- Given a population of geese where 50% are male and 20% of migrating geese are male,
    the ratio of migration rates between male and female geese is 1:4. -/
theorem geese_migration_rate_ratio :
  ∀ (total_geese male_geese migrating_geese male_migrating : ℕ),
  male_geese = total_geese / 2 →
  male_migrating = migrating_geese / 5 →
  (male_migrating : ℚ) / male_geese = (migrating_geese - male_migrating : ℚ) / (total_geese - male_geese) / 4 :=
by sorry

end NUMINAMATH_CALUDE_geese_migration_rate_ratio_l2354_235422


namespace NUMINAMATH_CALUDE_total_fruit_punch_l2354_235461

def orange_punch : Real := 4.5
def cherry_punch : Real := 2 * orange_punch
def apple_juice : Real := cherry_punch - 1.5
def pineapple_juice : Real := 3
def grape_punch : Real := apple_juice + 0.5 * apple_juice

theorem total_fruit_punch :
  orange_punch + cherry_punch + apple_juice + pineapple_juice + grape_punch = 35.25 := by
  sorry

end NUMINAMATH_CALUDE_total_fruit_punch_l2354_235461


namespace NUMINAMATH_CALUDE_optimal_profit_l2354_235412

-- Define the profit function
def profit (x : ℕ) : ℝ :=
  (500 - 10 * x) * (50 + x) - (500 - 10 * x) * 40

-- Define the optimal price increase
def optimal_price_increase : ℕ := 20

-- Define the optimal selling price
def optimal_selling_price : ℕ := 50 + optimal_price_increase

-- Define the maximum profit
def max_profit : ℝ := 9000

-- Theorem statement
theorem optimal_profit :
  (∀ x : ℕ, profit x ≤ profit optimal_price_increase) ∧
  (profit optimal_price_increase = max_profit) ∧
  (optimal_selling_price = 70) :=
sorry

end NUMINAMATH_CALUDE_optimal_profit_l2354_235412


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2354_235490

/-- If 4x^2 + mxy + y^2 is a perfect square, then m = ±4 -/
theorem perfect_square_condition (x y m : ℝ) : 
  (∃ (k : ℝ), 4*x^2 + m*x*y + y^2 = k^2) → (m = 4 ∨ m = -4) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2354_235490


namespace NUMINAMATH_CALUDE_salary_calculation_l2354_235478

theorem salary_calculation (food_fraction rent_fraction clothes_fraction remainder : ℚ) 
  (h1 : food_fraction = 1/5)
  (h2 : rent_fraction = 1/10)
  (h3 : clothes_fraction = 3/5)
  (h4 : remainder = 18000) :
  let total_spent_fraction := food_fraction + rent_fraction + clothes_fraction
  let remaining_fraction := 1 - total_spent_fraction
  let salary := remainder / remaining_fraction
  salary = 180000 := by sorry

end NUMINAMATH_CALUDE_salary_calculation_l2354_235478


namespace NUMINAMATH_CALUDE_sqrt5_and_sequences_l2354_235473

-- Define p-arithmetic
structure PArithmetic where
  p : ℕ
  -- Add more properties as needed

-- Define the concept of "extracting √5"
def can_extract_sqrt5 (pa : PArithmetic) : Prop := sorry

-- Define a sequence type
def Sequence (α : Type) := ℕ → α

-- Define properties for Fibonacci and geometric sequences
def is_fibonacci {α : Type} [Add α] (seq : Sequence α) : Prop := 
  ∀ n, seq (n + 2) = seq (n + 1) + seq n

def is_geometric {α : Type} [Mul α] (seq : Sequence α) : Prop := 
  ∃ r, ∀ n, seq (n + 1) = r * seq n

-- Main theorem
theorem sqrt5_and_sequences (pa : PArithmetic) :
  (¬ can_extract_sqrt5 pa → 
    ¬ ∃ (seq : Sequence ℚ), is_fibonacci seq ∧ is_geometric seq) ∧
  (can_extract_sqrt5 pa → 
    (∃ (seq : Sequence ℚ), is_fibonacci seq ∧ is_geometric seq) ∧
    (∀ (fib : Sequence ℚ), is_fibonacci fib → 
      ∃ (seq1 seq2 : Sequence ℚ), 
        is_fibonacci seq1 ∧ is_geometric seq1 ∧
        is_fibonacci seq2 ∧ is_geometric seq2 ∧
        (∀ n, fib n = seq1 n + seq2 n))) :=
by sorry

end NUMINAMATH_CALUDE_sqrt5_and_sequences_l2354_235473


namespace NUMINAMATH_CALUDE_randys_remaining_biscuits_l2354_235486

/-- Calculates the number of biscuits Randy is left with after receiving biscuits from his parents and his brother eating some. -/
theorem randys_remaining_biscuits
  (initial_biscuits : ℕ)
  (father_biscuits_kg : ℚ)
  (biscuit_weight_g : ℕ)
  (mother_biscuits : ℕ)
  (brother_eaten_percent : ℚ)
  (h1 : initial_biscuits = 32)
  (h2 : father_biscuits_kg = 1/2)
  (h3 : biscuit_weight_g = 50)
  (h4 : mother_biscuits = 15)
  (h5 : brother_eaten_percent = 30/100)
  : ℕ :=
by
  sorry

#check randys_remaining_biscuits

end NUMINAMATH_CALUDE_randys_remaining_biscuits_l2354_235486


namespace NUMINAMATH_CALUDE_proposition_A_necessary_not_sufficient_l2354_235479

/-- Proposition A: The inequality x^2 + 2ax + 4 ≤ 0 has solutions -/
def proposition_A (a : ℝ) : Prop := a ≤ -2 ∨ a ≥ 2

/-- Proposition B: The function f(x) = log_a(x + a - 2) is always positive on the interval (1, +∞) -/
def proposition_B (a : ℝ) : Prop := a ≥ 2

theorem proposition_A_necessary_not_sufficient :
  (∀ a : ℝ, proposition_B a → proposition_A a) ∧
  ¬(∀ a : ℝ, proposition_A a → proposition_B a) := by
  sorry

end NUMINAMATH_CALUDE_proposition_A_necessary_not_sufficient_l2354_235479


namespace NUMINAMATH_CALUDE_current_algae_count_l2354_235468

/-- The number of algae plants originally in Milford Lake -/
def original_algae : ℕ := 809

/-- The number of additional algae plants in Milford Lake -/
def additional_algae : ℕ := 2454

/-- Theorem stating the current total number of algae plants in Milford Lake -/
theorem current_algae_count : original_algae + additional_algae = 3263 := by
  sorry

end NUMINAMATH_CALUDE_current_algae_count_l2354_235468


namespace NUMINAMATH_CALUDE_missing_bricks_count_l2354_235427

/-- Represents a brick wall -/
structure BrickWall where
  total_positions : ℕ
  filled_positions : ℕ
  h_filled_le_total : filled_positions ≤ total_positions

/-- The number of missing bricks in a wall -/
def missing_bricks (wall : BrickWall) : ℕ :=
  wall.total_positions - wall.filled_positions

/-- Theorem stating that the number of missing bricks in the given wall is 26 -/
theorem missing_bricks_count (wall : BrickWall) 
  (h_total : wall.total_positions = 60)
  (h_filled : wall.filled_positions = 34) : 
  missing_bricks wall = 26 := by
sorry


end NUMINAMATH_CALUDE_missing_bricks_count_l2354_235427


namespace NUMINAMATH_CALUDE_exists_equitable_non_symmetric_polygon_l2354_235497

-- Define a polygon as a list of points in 2D space
def Polygon := List (ℝ × ℝ)

-- Function to check if a polygon has no self-intersections
def hasNoSelfIntersections (p : Polygon) : Prop :=
  sorry

-- Function to check if a line through the origin divides a polygon into two regions of equal area
def dividesEquallyThroughOrigin (p : Polygon) (line : ℝ → ℝ) : Prop :=
  sorry

-- Function to check if a polygon is equitable
def isEquitable (p : Polygon) : Prop :=
  ∀ line : ℝ → ℝ, dividesEquallyThroughOrigin p line

-- Function to check if a polygon is centrally symmetric about the origin
def isCentrallySymmetric (p : Polygon) : Prop :=
  sorry

-- Theorem statement
theorem exists_equitable_non_symmetric_polygon :
  ∃ p : Polygon, hasNoSelfIntersections p ∧ isEquitable p ∧ ¬(isCentrallySymmetric p) :=
sorry

end NUMINAMATH_CALUDE_exists_equitable_non_symmetric_polygon_l2354_235497


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_squared_times_one_plus_i_l2354_235452

/-- The imaginary part of i²(1+i) is -1 -/
theorem imaginary_part_of_i_squared_times_one_plus_i :
  Complex.im (Complex.I^2 * (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_squared_times_one_plus_i_l2354_235452


namespace NUMINAMATH_CALUDE_number_difference_l2354_235402

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 21800)
  (b_div_100 : 100 ∣ b)
  (a_eq_b_div_100 : a = b / 100) :
  b - a = 21384 := by sorry

end NUMINAMATH_CALUDE_number_difference_l2354_235402


namespace NUMINAMATH_CALUDE_total_donation_l2354_235458

def cassandra_pennies : ℕ := 5000
def james_pennies : ℕ := cassandra_pennies - 276
def stephanie_pennies : ℕ := 2 * james_pennies

theorem total_donation :
  cassandra_pennies + james_pennies + stephanie_pennies = 19172 :=
by sorry

end NUMINAMATH_CALUDE_total_donation_l2354_235458


namespace NUMINAMATH_CALUDE_pizza_area_increase_l2354_235423

/-- Theorem: Percent increase in pizza area
    If the radius of a large pizza is 60% larger than that of a medium pizza,
    then the percent increase in area between a medium and a large pizza is 156%. -/
theorem pizza_area_increase (r : ℝ) (h : r > 0) : 
  let large_radius := 1.6 * r
  let medium_area := π * r^2
  let large_area := π * large_radius^2
  (large_area - medium_area) / medium_area * 100 = 156 :=
by
  sorry

#check pizza_area_increase

end NUMINAMATH_CALUDE_pizza_area_increase_l2354_235423


namespace NUMINAMATH_CALUDE_tan_4305_degrees_l2354_235491

theorem tan_4305_degrees : Real.tan (4305 * π / 180) = -2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_4305_degrees_l2354_235491


namespace NUMINAMATH_CALUDE_long_jump_ratio_l2354_235463

/-- Given the conditions of a long jump event, prove the ratio of Margarita's jump to Ricciana's jump -/
theorem long_jump_ratio (ricciana_run : ℕ) (ricciana_jump : ℕ) (margarita_run : ℕ) (total_difference : ℕ) :
  ricciana_run = 20 →
  ricciana_jump = 4 →
  margarita_run = 18 →
  total_difference = 1 →
  (margarita_run + (ricciana_run + ricciana_jump + total_difference - margarita_run)) / ricciana_jump = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_long_jump_ratio_l2354_235463


namespace NUMINAMATH_CALUDE_problem_statement_l2354_235481

theorem problem_statement (a : ℝ) (h : a^2 - 2*a = 1) : 3*a^2 - 6*a - 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2354_235481


namespace NUMINAMATH_CALUDE_sphere_ratios_l2354_235434

/-- Given two spheres with radii in the ratio 2:3, prove that the ratio of their surface areas is 4:9 and the ratio of their volumes is 8:27 -/
theorem sphere_ratios (r₁ r₂ : ℝ) (h : r₁ / r₂ = 2 / 3) :
  (4 * π * r₁^2) / (4 * π * r₂^2) = 4 / 9 ∧
  ((4 / 3) * π * r₁^3) / ((4 / 3) * π * r₂^3) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_ratios_l2354_235434


namespace NUMINAMATH_CALUDE_height_inscribed_circle_inequality_l2354_235437

/-- For a right triangle, the height dropped to the hypotenuse is at most (1 + √2) times the radius of the inscribed circle. -/
theorem height_inscribed_circle_inequality (a b c h r : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 → r > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  h = (a * b) / c →  -- Definition of height
  r = (a + b - c) / 2 →  -- Definition of inscribed circle radius
  h ≤ r * (1 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_height_inscribed_circle_inequality_l2354_235437


namespace NUMINAMATH_CALUDE_hostel_expenditure_l2354_235442

theorem hostel_expenditure (original_students : ℕ) (new_students : ℕ) (expense_increase : ℚ) (average_decrease : ℚ) 
  (h1 : original_students = 35)
  (h2 : new_students = 7)
  (h3 : expense_increase = 42)
  (h4 : average_decrease = 1) :
  ∃ (original_expenditure : ℚ),
    original_expenditure = original_students * 
      ((expense_increase + (original_students + new_students) * average_decrease) / new_students) := by
  sorry

end NUMINAMATH_CALUDE_hostel_expenditure_l2354_235442


namespace NUMINAMATH_CALUDE_simplify_expression_l2354_235482

theorem simplify_expression (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ 1) :
  (x - 5 + 16 / (x + 3)) / ((x - 1) / (x^2 - 9)) = x^2 - 4*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2354_235482


namespace NUMINAMATH_CALUDE_incompatible_food_probability_l2354_235439

-- Define the set of foods
def Food : Type := Fin 5

-- Define the incompatibility relation
def incompatible : Food → Food → Prop := sorry

-- Define the number of incompatible pairs
def num_incompatible_pairs : ℕ := 3

-- Define the total number of possible pairs
def total_pairs : ℕ := Nat.choose 5 2

-- State the theorem
theorem incompatible_food_probability :
  (num_incompatible_pairs : ℚ) / (total_pairs : ℚ) = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_incompatible_food_probability_l2354_235439


namespace NUMINAMATH_CALUDE_vector_angle_condition_l2354_235493

-- Define the vectors a and b as functions of x
def a (x : ℝ) : Fin 2 → ℝ := ![2, x + 1]
def b (x : ℝ) : Fin 2 → ℝ := ![x + 2, 6]

-- Define the dot product of a and b
def dot_product (x : ℝ) : ℝ := (a x 0) * (b x 0) + (a x 1) * (b x 1)

-- Define the cross product of a and b
def cross_product (x : ℝ) : ℝ := (a x 0) * (b x 1) - (a x 1) * (b x 0)

-- Theorem statement
theorem vector_angle_condition (x : ℝ) :
  (dot_product x > 0 ∧ cross_product x ≠ 0) ↔ (x > -5/4 ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_vector_angle_condition_l2354_235493


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2354_235483

theorem max_value_of_expression (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum : x + y + z = 1) :
  x + y^2 + z^3 ≤ 1 ∧ ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ z₀ ≥ 0 ∧ x₀ + y₀ + z₀ = 1 ∧ x₀ + y₀^2 + z₀^3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2354_235483


namespace NUMINAMATH_CALUDE_new_england_population_l2354_235487

/-- The population of New England -/
def population_new_england : ℕ := sorry

/-- The population of New York -/
def population_new_york : ℕ := sorry

/-- New York's population is two-thirds of New England's -/
axiom new_york_population_ratio : population_new_york = (2 * population_new_england) / 3

/-- The combined population of New York and New England is 3,500,000 -/
axiom combined_population : population_new_york + population_new_england = 3500000

/-- Theorem: The population of New England is 2,100,000 -/
theorem new_england_population : population_new_england = 2100000 := by sorry

end NUMINAMATH_CALUDE_new_england_population_l2354_235487


namespace NUMINAMATH_CALUDE_parabola_translation_l2354_235420

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
    b := p.b - 2 * p.a * h
    c := p.c + p.a * h^2 - p.b * h - v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk (-1) (-2) 0
  let translated := translate (translate original 2 0) 0 3
  y = -(x * (x + 2)) → y = -(x - 1)^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l2354_235420


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l2354_235418

def A (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![x, 2, x^2],
    ![3, y, 4],
    ![z, 3, z^2]]

def B (x y z k l m n : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![-8, k, -x^3],
    ![l, -y^2, m],
    ![3, n, z^3]]

theorem inverse_matrices_sum (x y z k l m n : ℝ) :
  A x y z * B x y z k l m n = 1 →
  x + y + z + k + l + m + n = -1/3 := by sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l2354_235418


namespace NUMINAMATH_CALUDE_fraction_subtraction_addition_l2354_235436

theorem fraction_subtraction_addition : 
  (1 : ℚ) / 12 - 5 / 6 + 1 / 3 = -5 / 12 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_addition_l2354_235436


namespace NUMINAMATH_CALUDE_retail_price_decrease_percentage_l2354_235426

/-- Proves that the retail price decrease percentage is equal to 44.000000000000014% 
    given the conditions in the problem. -/
theorem retail_price_decrease_percentage 
  (wholesale_price : ℝ) 
  (retail_price : ℝ) 
  (decrease_percentage : ℝ) : 
  retail_price = wholesale_price * 1.80 →
  retail_price * (1 - decrease_percentage) = 
    (wholesale_price * 1.44000000000000014) * 1.80 →
  decrease_percentage = 0.44000000000000014 := by
sorry

end NUMINAMATH_CALUDE_retail_price_decrease_percentage_l2354_235426


namespace NUMINAMATH_CALUDE_faster_train_speed_l2354_235409

/-- Calculates the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed (train_length : ℝ) (crossing_time : ℝ) : 
  train_length = 100 →
  crossing_time = 10 →
  (2 * train_length) / crossing_time = 40 / 3 :=
by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_faster_train_speed_l2354_235409


namespace NUMINAMATH_CALUDE_rhombus_diagonals_l2354_235480

/-- Given a rhombus with perimeter 100 cm and sum of diagonals 62 cm, 
    prove that its diagonals are 48 cm and 14 cm. -/
theorem rhombus_diagonals (s : ℝ) (d₁ d₂ : ℝ) 
  (h_perimeter : 4 * s = 100)
  (h_diag_sum : d₁ + d₂ = 62)
  (h_pythag : s^2 = (d₁/2)^2 + (d₂/2)^2) :
  (d₁ = 48 ∧ d₂ = 14) ∨ (d₁ = 14 ∧ d₂ = 48) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_l2354_235480


namespace NUMINAMATH_CALUDE_line_plane_relationship_l2354_235475

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line being parallel to a plane
variable (parallel_to_plane : Line → Plane → Prop)

-- Define the possible relationships between a line and a plane
inductive LineToPlaneRelation
  | Parallel
  | Intersecting
  | Within

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (α : Plane) 
  (h1 : skew a b) 
  (h2 : parallel_to_plane a α) :
  ∃ (r : LineToPlaneRelation), 
    (r = LineToPlaneRelation.Parallel) ∨ 
    (r = LineToPlaneRelation.Intersecting) ∨ 
    (r = LineToPlaneRelation.Within) :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l2354_235475


namespace NUMINAMATH_CALUDE_discount_percentage_l2354_235415

theorem discount_percentage (M : ℝ) (C : ℝ) (SP : ℝ) 
  (h1 : C = 0.64 * M) 
  (h2 : SP = C * 1.171875) : 
  (M - SP) / M * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l2354_235415


namespace NUMINAMATH_CALUDE_correct_answer_l2354_235404

theorem correct_answer (x : ℤ) (h : x + 5 = 35) : x - 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l2354_235404


namespace NUMINAMATH_CALUDE_smallest_n_for_quadratic_inequality_six_satisfies_inequality_smallest_n_is_six_l2354_235444

theorem smallest_n_for_quadratic_inequality :
  ∀ n : ℤ, n^2 - 9*n + 20 > 0 → n ≥ 6 :=
by
  sorry

theorem six_satisfies_inequality : (6 : ℤ)^2 - 9*(6 : ℤ) + 20 > 0 :=
by
  sorry

theorem smallest_n_is_six :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 9*m + 20 > 0 → m ≥ n) ∧ n^2 - 9*n + 20 > 0 ∧ n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_quadratic_inequality_six_satisfies_inequality_smallest_n_is_six_l2354_235444


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2354_235433

theorem fraction_subtraction : (18 : ℚ) / 42 - 2 / 9 = 13 / 63 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2354_235433


namespace NUMINAMATH_CALUDE_log_inequality_l2354_235495

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + 2*x) < 2*x := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2354_235495


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2354_235460

theorem complex_equation_solution (a : ℝ) : (a + Complex.I) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2354_235460


namespace NUMINAMATH_CALUDE_work_fraction_after_twenty_days_l2354_235405

/-- Proves that the fraction of work completed after 20 days is 15/64 -/
theorem work_fraction_after_twenty_days 
  (W : ℝ) -- Total work to be done
  (initial_workers : ℕ := 10) -- Initial number of workers
  (initial_duration : ℕ := 100) -- Initial planned duration in days
  (work_time : ℕ := 20) -- Time worked before firing workers
  (fired_workers : ℕ := 2) -- Number of workers fired
  (remaining_time : ℕ := 75) -- Time to complete the remaining work
  (F : ℝ) -- Fraction of work completed after 20 days
  (h1 : initial_workers * (W / initial_duration) = work_time * (F * W / work_time)) -- Work rate equality for first 20 days
  (h2 : (initial_workers - fired_workers) * ((1 - F) * W / remaining_time) = initial_workers * (W / initial_duration)) -- Work rate equality for remaining work
  : F = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_work_fraction_after_twenty_days_l2354_235405


namespace NUMINAMATH_CALUDE_integral_condition_implies_b_value_l2354_235472

open MeasureTheory Measure Set Real
open intervalIntegral

theorem integral_condition_implies_b_value (b : ℝ) :
  (∫ x in (-1)..0, (2 * x + b)) = 2 →
  b = 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_condition_implies_b_value_l2354_235472


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2354_235464

-- Define the universal set U
def U : Set ℕ := {x : ℕ | x ≥ 2}

-- Define set A
def A : Set ℕ := {x : ℕ | x^2 ≥ 5}

-- Theorem statement
theorem complement_of_A_in_U : (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2354_235464


namespace NUMINAMATH_CALUDE_lcm_problem_l2354_235403

theorem lcm_problem (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 28) :
  ∃ (a' c' : ℕ), a' ∣ a ∧ c' ∣ c ∧ Nat.lcm a' c' = 35 ∧ 
  ∀ (x y : ℕ), x ∣ a → y ∣ c → Nat.lcm x y ≥ 35 :=
sorry

end NUMINAMATH_CALUDE_lcm_problem_l2354_235403


namespace NUMINAMATH_CALUDE_apple_picking_fraction_l2354_235494

theorem apple_picking_fraction (total_apples : ℕ) (remaining_apples : ℕ) : 
  total_apples = 200 →
  remaining_apples = 20 →
  ∃ f : ℚ, 
    f > 0 ∧ 
    f < 1 ∧
    (f * total_apples : ℚ) + (2 * f * total_apples : ℚ) + (f * total_apples + 20 : ℚ) = total_apples - remaining_apples ∧
    f = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_apple_picking_fraction_l2354_235494


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2354_235424

-- Define a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function
def f (x : ℝ) : ℝ := 11 * x^2 + 29 * x

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2354_235424


namespace NUMINAMATH_CALUDE_seventh_term_ratio_l2354_235457

/-- Two arithmetic sequences with sums S and T for their first n terms. -/
def ArithmeticSequences (S T : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n / T n = (5 * n + 10) / (2 * n - 1)

/-- The 7th term of an arithmetic sequence. -/
def seventhTerm (seq : ℕ → ℚ) : ℚ := seq 7

theorem seventh_term_ratio (S T : ℕ → ℚ) (h : ArithmeticSequences S T) :
  seventhTerm S / seventhTerm T = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_ratio_l2354_235457


namespace NUMINAMATH_CALUDE_x_not_greater_than_one_l2354_235455

theorem x_not_greater_than_one (x : ℝ) (h : |x - 1| + x = 1) : x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_x_not_greater_than_one_l2354_235455


namespace NUMINAMATH_CALUDE_good_numbers_in_set_l2354_235419

-- Define what a "good number" is
def isGoodNumber (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), Function.Bijective a ∧
    ∀ k : Fin n, ∃ m : ℕ, (k.val + 1 + (a k).val + 1 : ℕ) = m * m

-- Theorem statement
theorem good_numbers_in_set :
  isGoodNumber 11 = false ∧
  isGoodNumber 13 = true ∧
  isGoodNumber 15 = true ∧
  isGoodNumber 17 = true ∧
  isGoodNumber 19 = true :=
by sorry

end NUMINAMATH_CALUDE_good_numbers_in_set_l2354_235419


namespace NUMINAMATH_CALUDE_solve_inequality_max_a_value_l2354_235446

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

-- Theorem for part I
theorem solve_inequality :
  ∀ x : ℝ, f x > 4 ↔ x < -1.5 ∨ x > 2.5 := by sorry

-- Theorem for part II
theorem max_a_value :
  ∃ a : ℝ, (∀ x : ℝ, f x ≥ a) ∧ (∀ b : ℝ, (∀ x : ℝ, f x ≥ b) → b ≤ a) ∧ a = 3 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_max_a_value_l2354_235446


namespace NUMINAMATH_CALUDE_pet_store_cages_theorem_l2354_235408

/-- Given a number of initial puppies, sold puppies, and puppies per cage,
    calculate the number of cages needed. -/
def cagesNeeded (initialPuppies soldPuppies puppiesPerCage : ℕ) : ℕ :=
  ((initialPuppies - soldPuppies) + puppiesPerCage - 1) / puppiesPerCage

theorem pet_store_cages_theorem :
  cagesNeeded 36 7 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_theorem_l2354_235408


namespace NUMINAMATH_CALUDE_select_five_from_fifteen_l2354_235450

theorem select_five_from_fifteen (n : Nat) (r : Nat) : n = 15 ∧ r = 5 →
  Nat.choose n r = 3003 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_fifteen_l2354_235450


namespace NUMINAMATH_CALUDE_radical_product_simplification_l2354_235421

theorem radical_product_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (48 * x) * Real.sqrt (3 * x) * Real.sqrt (50 * x) = 60 * x * Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l2354_235421


namespace NUMINAMATH_CALUDE_triangle_problem_l2354_235400

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with side lengths a, b, c opposite to angles A, B, C
  c * Real.cos A - 2 * b * Real.cos B + a * Real.cos C = 0 →
  a + c = 13 →
  c > a →
  a * c * Real.cos B = 20 →
  B = Real.pi / 3 ∧ Real.sin A = 5 * Real.sqrt 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2354_235400


namespace NUMINAMATH_CALUDE_rectangle_to_cylinders_volume_ratio_l2354_235470

theorem rectangle_to_cylinders_volume_ratio :
  let rectangle_width : ℝ := 7
  let rectangle_length : ℝ := 10
  let cylinder1_radius : ℝ := rectangle_width / (2 * Real.pi)
  let cylinder1_height : ℝ := rectangle_length
  let cylinder1_volume : ℝ := Real.pi * cylinder1_radius^2 * cylinder1_height
  let cylinder2_radius : ℝ := rectangle_length / (2 * Real.pi)
  let cylinder2_height : ℝ := rectangle_width
  let cylinder2_volume : ℝ := Real.pi * cylinder2_radius^2 * cylinder2_height
  let larger_volume : ℝ := max cylinder1_volume cylinder2_volume
  let smaller_volume : ℝ := min cylinder1_volume cylinder2_volume
  larger_volume / smaller_volume = 10 / 7 := by sorry

end NUMINAMATH_CALUDE_rectangle_to_cylinders_volume_ratio_l2354_235470


namespace NUMINAMATH_CALUDE_billy_wins_l2354_235441

/-- Represents the swimming times for Billy and Margaret -/
structure SwimmingTimes where
  billy_first_5_laps : ℕ  -- in minutes
  billy_next_3_laps : ℕ  -- in minutes
  billy_next_lap : ℕ     -- in minutes
  billy_final_lap : ℕ    -- in seconds
  margaret_total : ℕ     -- in minutes

/-- Calculates the time difference between Billy and Margaret's finish times -/
def timeDifference (times : SwimmingTimes) : ℕ :=
  let billy_total_seconds := 
    (times.billy_first_5_laps + times.billy_next_3_laps + times.billy_next_lap) * 60 + times.billy_final_lap
  let margaret_total_seconds := times.margaret_total * 60
  margaret_total_seconds - billy_total_seconds

/-- Theorem stating that Billy finishes 30 seconds before Margaret -/
theorem billy_wins (times : SwimmingTimes) 
    (h1 : times.billy_first_5_laps = 2)
    (h2 : times.billy_next_3_laps = 4)
    (h3 : times.billy_next_lap = 1)
    (h4 : times.billy_final_lap = 150)
    (h5 : times.margaret_total = 10) : 
  timeDifference times = 30 := by
  sorry


end NUMINAMATH_CALUDE_billy_wins_l2354_235441


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l2354_235445

theorem boat_speed_ratio (b r : ℝ) (h1 : b > 0) (h2 : r > 0) 
  (h3 : (b - r)⁻¹ = 2 * (b + r)⁻¹) 
  (s1 s2 : ℝ) (h4 : s1 > 0) (h5 : s2 > 0)
  (h6 : b * (1/4) + b * (3/4) = b) :
  b / (s1 + s2) = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l2354_235445


namespace NUMINAMATH_CALUDE_inequality_proof_l2354_235467

theorem inequality_proof (a : ℝ) (h : a ≠ 2) :
  (1 : ℝ) / (a^2 - 4*a + 4) > 2 / (a^3 - 8) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2354_235467


namespace NUMINAMATH_CALUDE_max_a_for_monotonic_cubic_l2354_235466

/-- Given a function f(x) = x^3 - ax that is monotonically increasing on [1, +∞),
    the maximum value of a is 3. -/
theorem max_a_for_monotonic_cubic (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ x, (x^3 - a*x) ≤ (y^3 - a*y)) → a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_a_for_monotonic_cubic_l2354_235466


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2354_235499

theorem arithmetic_calculations : 
  (12 - (-18) + (-7) - 15 = 8) ∧ 
  ((-1/2) * (-8) + (-6) / (-1/3)^2 = -50) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2354_235499


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2354_235492

def a (n : ℕ) : ℝ := 2 * n - 8

theorem arithmetic_sequence_properties :
  (∀ n : ℕ, a (n + 1) > a n) ∧
  (∀ n : ℕ, n > 0 → a (n + 1) / (n + 1) > a n / n) ∧
  (∃ n : ℕ, (n + 1) * a (n + 1) ≤ n * a n) ∧
  (∃ n : ℕ, a (n + 1)^2 ≤ a n^2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2354_235492


namespace NUMINAMATH_CALUDE_sqrt_of_sixteen_l2354_235454

theorem sqrt_of_sixteen : ∃ (x : ℝ), x^2 = 16 ∧ (x = 4 ∨ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sixteen_l2354_235454


namespace NUMINAMATH_CALUDE_workshop_workers_count_workshop_workers_count_is_49_l2354_235459

/-- Proves that the total number of workers in a workshop is 49, given the following conditions:
  * The average salary of all workers is 8000
  * There are 7 technicians with an average salary of 20000
  * The average salary of the non-technicians is 6000
-/
theorem workshop_workers_count : ℕ → Prop :=
  fun (total_workers : ℕ) =>
    let avg_salary : ℚ := 8000
    let technician_count : ℕ := 7
    let technician_avg_salary : ℚ := 20000
    let non_technician_avg_salary : ℚ := 6000
    let non_technician_count : ℕ := total_workers - technician_count
    (↑total_workers * avg_salary = 
      ↑technician_count * technician_avg_salary + 
      ↑non_technician_count * non_technician_avg_salary) →
    total_workers = 49

theorem workshop_workers_count_is_49 : workshop_workers_count 49 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_workshop_workers_count_is_49_l2354_235459


namespace NUMINAMATH_CALUDE_seventh_roots_of_unity_product_l2354_235435

theorem seventh_roots_of_unity_product (z : ℂ) (h : z = Complex.exp (2 * Real.pi * Complex.I / 7)) :
  (3 - z) * (3 - z^2) * (3 - z^3) * (3 - z^4) * (3 - z^5) * (3 - z^6) = 1093 := by
  sorry

end NUMINAMATH_CALUDE_seventh_roots_of_unity_product_l2354_235435


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l2354_235447

/-- Given a polynomial division where:
    - p(x) is a polynomial of degree 17
    - g(x) is the divisor polynomial
    - The quotient polynomial has degree 9
    - The remainder polynomial has degree 5
    Then the degree of g(x) is 8. -/
theorem polynomial_division_degree (p g q r : Polynomial ℝ) : 
  Polynomial.degree p = 17 →
  p = g * q + r →
  Polynomial.degree q = 9 →
  Polynomial.degree r = 5 →
  Polynomial.degree g = 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l2354_235447


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l2354_235489

def f (x : ℝ) : ℝ := x^3 + 2*x

theorem f_sum_symmetric : f 5 + f (-5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l2354_235489


namespace NUMINAMATH_CALUDE_collinear_probability_is_1_182_l2354_235449

/-- Represents a 4x4 square array of dots -/
def SquareArray : Type := Fin 4 × Fin 4

/-- The total number of dots in the array -/
def totalDots : Nat := 16

/-- The number of ways to choose 4 dots from the array -/
def totalChoices : Nat := Nat.choose totalDots 4

/-- The number of sets of 4 collinear dots in the array -/
def collinearSets : Nat := 10

/-- The probability of choosing 4 collinear dots -/
def collinearProbability : Rat := collinearSets / totalChoices

theorem collinear_probability_is_1_182 :
  collinearProbability = 1 / 182 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_is_1_182_l2354_235449


namespace NUMINAMATH_CALUDE_min_value_expression_l2354_235428

theorem min_value_expression (x y z : ℝ) 
  (hx : -0.5 ≤ x ∧ x ≤ 1) 
  (hy : -0.5 ≤ y ∧ y ≤ 1) 
  (hz : -0.5 ≤ z ∧ z ≤ 1) : 
  3 / ((1 - x) * (1 - y) * (1 - z)) + 3 / ((1 + x) * (1 + y) * (1 + z)) ≥ 6 ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → 3 / ((1 - x) * (1 - y) * (1 - z)) + 3 / ((1 + x) * (1 + y) * (1 + z)) = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2354_235428
