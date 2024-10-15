import Mathlib

namespace NUMINAMATH_CALUDE_set_cardinality_relation_l2362_236208

theorem set_cardinality_relation (a b : ℕ+) (A B : Finset ℕ+) :
  (A ∩ B = ∅) →
  (∀ i ∈ A ∪ B, (i + a) ∈ A ∨ (i - b) ∈ B) →
  a * A.card = b * B.card :=
sorry

end NUMINAMATH_CALUDE_set_cardinality_relation_l2362_236208


namespace NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l2362_236271

theorem midpoint_of_complex_line_segment :
  let z₁ : ℂ := -5 + 7*I
  let z₂ : ℂ := 9 - 3*I
  let midpoint := (z₁ + z₂) / 2
  midpoint = 2 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l2362_236271


namespace NUMINAMATH_CALUDE_tree_height_difference_l2362_236281

-- Define the heights of the trees
def birch_height : ℚ := 12 + 1/4
def maple_height : ℚ := 20 + 2/5

-- Define the height difference
def height_difference : ℚ := maple_height - birch_height

-- Theorem to prove
theorem tree_height_difference :
  height_difference = 8 + 3/20 := by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l2362_236281


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l2362_236224

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  let x := sum_integers 40 60
  let y := count_even_integers 40 60
  x + y = 1061 := by sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l2362_236224


namespace NUMINAMATH_CALUDE_cos_graph_transformation_l2362_236230

theorem cos_graph_transformation (x : ℝ) :
  let original_point := (x, Real.cos x)
  let transformed_point := (4 * x, Real.cos (x / 4))
  transformed_point.2 = original_point.2 := by
sorry

end NUMINAMATH_CALUDE_cos_graph_transformation_l2362_236230


namespace NUMINAMATH_CALUDE_min_stamps_for_50_cents_l2362_236241

/-- Represents the number of stamps and their total value -/
structure StampCombination :=
  (threes : ℕ)
  (fours : ℕ)

/-- Calculates the total value of stamps in cents -/
def total_value (s : StampCombination) : ℕ :=
  3 * s.threes + 4 * s.fours

/-- Checks if a stamp combination is valid (totals 50 cents) -/
def is_valid (s : StampCombination) : Prop :=
  total_value s = 50

/-- Theorem: The minimum number of stamps to make 50 cents using 3 cent and 4 cent stamps is 13 -/
theorem min_stamps_for_50_cents :
  ∃ (s : StampCombination), is_valid s ∧
    (∀ (t : StampCombination), is_valid t → s.threes + s.fours ≤ t.threes + t.fours) ∧
    s.threes + s.fours = 13 :=
  sorry

end NUMINAMATH_CALUDE_min_stamps_for_50_cents_l2362_236241


namespace NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l2362_236280

/-- Given a paper with a certain number of pages and a number of days to complete it,
    calculate the number of pages that need to be written per day. -/
def pagesPerDay (totalPages : ℕ) (days : ℕ) : ℕ :=
  totalPages / days

theorem stacy_paper_pages_per_day :
  pagesPerDay 33 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l2362_236280


namespace NUMINAMATH_CALUDE_positive_correlation_missing_data_point_l2362_236253

-- Define the regression line
def regression_line (x : ℝ) : ℝ := 6.5 * x + 17.5

-- Define the data points
def data_points : List (ℝ × ℝ) := [(2, 30), (4, 40), (5, 60), (6, 50), (8, 70)]

-- Theorem 1: Positive correlation
theorem positive_correlation : 
  ∀ x₁ x₂, x₁ < x₂ → regression_line x₁ < regression_line x₂ :=
by sorry

-- Theorem 2: Missing data point
theorem missing_data_point : 
  ∃ y, (2, y) ∈ data_points ∧ y = 30 :=
by sorry

end NUMINAMATH_CALUDE_positive_correlation_missing_data_point_l2362_236253


namespace NUMINAMATH_CALUDE_volume_range_l2362_236202

/-- Pyramid S-ABCD with square base ABCD and isosceles right triangle side face SAD -/
structure Pyramid where
  /-- Side length of the square base ABCD -/
  base_side : ℝ
  /-- Length of SC -/
  sc_length : ℝ
  /-- The base ABCD is a square with side length 2 -/
  base_side_eq_two : base_side = 2
  /-- The side face SAD is an isosceles right triangle with SD as the hypotenuse -/
  sad_isosceles_right : True  -- This condition is implied by the structure
  /-- 2√2 ≤ SC ≤ 4 -/
  sc_range : 2 * Real.sqrt 2 ≤ sc_length ∧ sc_length ≤ 4

/-- Volume of the pyramid -/
def volume (p : Pyramid) : ℝ := sorry

/-- Theorem stating the range of the pyramid's volume -/
theorem volume_range (p : Pyramid) : 
  (4 * Real.sqrt 3) / 3 ≤ volume p ∧ volume p ≤ 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_range_l2362_236202


namespace NUMINAMATH_CALUDE_parabola_circle_theorem_l2362_236207

/-- Given a parabola y = ax^2 + bx + c (a ≠ 0) intersecting the x-axis at points A and B,
    the equation of the circle with AB as diameter is ax^2 + bx + c + ay^2 = 0. -/
theorem parabola_circle_theorem (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ →
  ∀ x y : ℝ, a * x^2 + b * x + c + a * y^2 = 0 ↔ 
    ∃ t : ℝ, x = (1 - t) * x₁ + t * x₂ ∧ 
             y^2 = t * (1 - t) * (x₂ - x₁)^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_circle_theorem_l2362_236207


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l2362_236205

/-- Given a geometric sequence {a_n} where a_4 = 7 and a_8 = 63, prove that a_6 = 21 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_a4 : a 4 = 7) (h_a8 : a 8 = 63) : a 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l2362_236205


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2362_236229

/-- Given a complex number i such that i^2 = -1, 
    prove that (2-i)/(1+4i) = -2/17 - (9/17)i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + 4*i) = -2/17 - (9/17)*i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2362_236229


namespace NUMINAMATH_CALUDE_cubic_function_minimum_l2362_236228

/-- The function f(x) = x³ - 3x² + 1 reaches its global minimum at x = 2 -/
theorem cubic_function_minimum (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 3*x^2 + 1
  ∀ y : ℝ, f 2 ≤ f y := by sorry

end NUMINAMATH_CALUDE_cubic_function_minimum_l2362_236228


namespace NUMINAMATH_CALUDE_zero_rational_others_irrational_l2362_236203

-- Define rational numbers
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- State the theorem
theorem zero_rational_others_irrational :
  IsRational 0 ∧ ¬IsRational (-Real.pi) ∧ ¬IsRational (Real.sqrt 3) ∧ ¬IsRational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_zero_rational_others_irrational_l2362_236203


namespace NUMINAMATH_CALUDE_d17_value_l2362_236239

def is_divisor_of (d n : ℕ) : Prop := n % d = 0

theorem d17_value (n : ℕ) (d : ℕ → ℕ) :
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 17 → d i < d j) →
  (∀ i, 1 ≤ i ∧ i ≤ 17 → is_divisor_of (d i) n) →
  d 1 = 1 →
  (d 7)^2 + (d 15)^2 = (d 16)^2 →
  d 17 = 28 :=
sorry

end NUMINAMATH_CALUDE_d17_value_l2362_236239


namespace NUMINAMATH_CALUDE_robot_gloves_rings_arrangements_l2362_236237

/-- Represents the number of arms of the robot -/
def num_arms : ℕ := 6

/-- Represents the total number of items (gloves and rings) -/
def total_items : ℕ := 2 * num_arms

/-- Represents the number of valid arrangements for putting on gloves and rings -/
def valid_arrangements : ℕ := (Nat.factorial total_items) / (2^num_arms)

/-- Theorem stating the number of valid arrangements for the robot to put on gloves and rings -/
theorem robot_gloves_rings_arrangements :
  valid_arrangements = (Nat.factorial total_items) / (2^num_arms) :=
by sorry

end NUMINAMATH_CALUDE_robot_gloves_rings_arrangements_l2362_236237


namespace NUMINAMATH_CALUDE_fraction_comparison_l2362_236274

theorem fraction_comparison (a b m : ℝ) (ha : a > b) (hb : b > 0) (hm : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2362_236274


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l2362_236254

/-- Proves that the weight of a zinc-copper mixture is 74 kg given the specified conditions -/
theorem zinc_copper_mixture_weight :
  ∀ (zinc copper total : ℝ),
  zinc = 33.3 →
  zinc / copper = 9 / 11 →
  total = zinc + copper →
  total = 74 := by
sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l2362_236254


namespace NUMINAMATH_CALUDE_prime_pairs_sum_50_l2362_236295

/-- A function that returns the number of unordered pairs of prime numbers that sum to a given natural number. -/
def count_prime_pairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p) ∧ 2 * p ≤ n) (Finset.range (n / 2 + 1))).card

/-- The theorem stating that there are exactly 4 unordered pairs of prime numbers that sum to 50. -/
theorem prime_pairs_sum_50 : count_prime_pairs 50 = 4 := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_sum_50_l2362_236295


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_45_l2362_236206

theorem four_digit_divisible_by_45 : ∃ (a b : ℕ), 
  a < 10 ∧ b < 10 ∧ 
  (1000 * a + 520 + b) % 45 = 0 ∧
  (∃ (c d : ℕ), c < 10 ∧ d < 10 ∧ c ≠ a ∧ d ≠ b ∧ (1000 * c + 520 + d) % 45 = 0) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_45_l2362_236206


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2362_236201

theorem quadratic_equation_solution (m : ℝ) 
  (x₁ x₂ : ℝ) -- Two real roots
  (h1 : x₁^2 - m*x₁ + 2*m - 1 = 0) -- x₁ satisfies the equation
  (h2 : x₂^2 - m*x₂ + 2*m - 1 = 0) -- x₂ satisfies the equation
  (h3 : x₁^2 + x₂^2 = 7) -- Given condition
  : m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2362_236201


namespace NUMINAMATH_CALUDE_pencils_count_l2362_236261

/-- The number of pencils originally in the jar -/
def original_pencils : ℕ := 87

/-- The number of pencils removed from the jar -/
def removed_pencils : ℕ := 4

/-- The number of pencils left in the jar after removal -/
def remaining_pencils : ℕ := 83

/-- Theorem stating that the original number of pencils equals the sum of removed and remaining pencils -/
theorem pencils_count : original_pencils = removed_pencils + remaining_pencils := by
  sorry

end NUMINAMATH_CALUDE_pencils_count_l2362_236261


namespace NUMINAMATH_CALUDE_shadow_problem_l2362_236204

/-- Given a cube with edge length 2 cm and a light source y cm above an upper vertex
    casting a shadow with area 98 sq cm (excluding the area beneath the cube),
    prove that the greatest integer not exceeding 1000y is 8100. -/
theorem shadow_problem (y : ℝ) : 
  (2 : ℝ) > 0 ∧ 
  y > 0 ∧ 
  (y / (Real.sqrt 102 - 2) = 1) ∧ 
  (98 : ℝ) = (Real.sqrt 102)^2 - 2^2 →
  Int.floor (1000 * y) = 8100 := by
sorry

end NUMINAMATH_CALUDE_shadow_problem_l2362_236204


namespace NUMINAMATH_CALUDE_second_tree_height_l2362_236213

/-- Given two trees casting shadows under the same conditions, 
    this theorem calculates the height of the second tree. -/
theorem second_tree_height
  (h1 : ℝ) -- Height of the first tree
  (s1 : ℝ) -- Shadow length of the first tree
  (s2 : ℝ) -- Shadow length of the second tree
  (h1_positive : h1 > 0)
  (s1_positive : s1 > 0)
  (s2_positive : s2 > 0)
  (h1_value : h1 = 28)
  (s1_value : s1 = 30)
  (s2_value : s2 = 45) :
  ∃ (h2 : ℝ), h2 = 42 ∧ h2 / s2 = h1 / s1 := by
  sorry


end NUMINAMATH_CALUDE_second_tree_height_l2362_236213


namespace NUMINAMATH_CALUDE_consumption_increase_after_tax_reduction_l2362_236246

/-- 
Given a commodity with tax and consumption, prove that if the tax is reduced by 20% 
and the revenue decreases by 8%, then the consumption must have increased by 15%.
-/
theorem consumption_increase_after_tax_reduction (T C : ℝ) 
  (h1 : T > 0) (h2 : C > 0) : 
  (0.80 * T) * (C * (1 + 15/100)) = 0.92 * (T * C) := by
  sorry

end NUMINAMATH_CALUDE_consumption_increase_after_tax_reduction_l2362_236246


namespace NUMINAMATH_CALUDE_total_schedules_l2362_236287

/-- Represents the number of classes to be scheduled -/
def num_classes : ℕ := 4

/-- Represents the number of classes that can be scheduled in the first period -/
def first_period_options : ℕ := 3

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: The total number of different possible schedules is 18 -/
theorem total_schedules : 
  first_period_options * factorial (num_classes - 1) = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_schedules_l2362_236287


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2362_236294

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 1415 ∧ 
  (2500000 - n) % 1423 = 0 ∧ 
  ∀ (m : ℕ), m < n → (2500000 - m) % 1423 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2362_236294


namespace NUMINAMATH_CALUDE_min_value_C_squared_minus_D_squared_l2362_236284

theorem min_value_C_squared_minus_D_squared
  (x y z : ℝ)
  (hx : x ≥ 0)
  (hy : y ≥ 0)
  (hz : z ≥ 0)
  (C : ℝ := Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 11))
  (D : ℝ := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2)) :
  C^2 - D^2 ≥ 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_C_squared_minus_D_squared_l2362_236284


namespace NUMINAMATH_CALUDE_inequality_solution_l2362_236244

def linear_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

def negative_for_positive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x < 0

theorem inequality_solution 
  (f : ℝ → ℝ) 
  (h_linear : linear_function f) 
  (h_neg : negative_for_positive f) 
  (n : ℕ) 
  (hn : n > 0) 
  (a : ℝ) 
  (ha : a < 0) :
  ∀ x : ℝ, 
    (1 / n : ℝ) * f (a * x^2) - f x > (1 / n : ℝ) * f (a^2 * x) - f a ↔ 
      (a < -Real.sqrt n ∧ (x > n / a ∨ x < a)) ∨ 
      (a = -Real.sqrt n ∧ x ≠ -Real.sqrt n) ∨ 
      (-Real.sqrt n < a ∧ (x > a ∨ x < n / a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2362_236244


namespace NUMINAMATH_CALUDE_donut_distribution_l2362_236269

/-- The number of ways to distribute n indistinguishable objects among k distinct boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := sorry

theorem donut_distribution :
  let n : ℕ := 3  -- number of additional donuts to distribute
  let k : ℕ := 5  -- number of donut kinds
  distribute n k = choose (n + k - 1) n ∧
  choose (n + k - 1) n = 35 :=
by sorry

end NUMINAMATH_CALUDE_donut_distribution_l2362_236269


namespace NUMINAMATH_CALUDE_annie_initial_money_l2362_236291

/-- The amount of money Annie had initially -/
def initial_money : ℕ := 132

/-- The price of a hamburger -/
def hamburger_price : ℕ := 4

/-- The price of a milkshake -/
def milkshake_price : ℕ := 5

/-- The number of hamburgers Annie bought -/
def hamburgers_bought : ℕ := 8

/-- The number of milkshakes Annie bought -/
def milkshakes_bought : ℕ := 6

/-- The amount of money Annie had left -/
def money_left : ℕ := 70

theorem annie_initial_money :
  initial_money = 
    hamburger_price * hamburgers_bought + 
    milkshake_price * milkshakes_bought + 
    money_left :=
by sorry

end NUMINAMATH_CALUDE_annie_initial_money_l2362_236291


namespace NUMINAMATH_CALUDE_half_plus_five_equals_fifteen_l2362_236247

theorem half_plus_five_equals_fifteen (n : ℝ) : (1/2) * n + 5 = 15 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_fifteen_l2362_236247


namespace NUMINAMATH_CALUDE_helen_cookies_l2362_236293

/-- The total number of chocolate chip cookies Helen baked -/
def total_cookies (yesterday today : ℕ) : ℕ := yesterday + today

/-- Theorem stating that Helen baked 1081 chocolate chip cookies in total -/
theorem helen_cookies : total_cookies 527 554 = 1081 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_l2362_236293


namespace NUMINAMATH_CALUDE_complex_magnitude_l2362_236200

theorem complex_magnitude (w : ℂ) (h : w^2 = -48 + 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2362_236200


namespace NUMINAMATH_CALUDE_hex_B1F_to_dec_l2362_236248

def hex_to_dec (hex : String) : ℕ :=
  hex.foldr (fun c acc => 16 * acc + 
    match c with
    | 'A' => 10
    | 'B' => 11
    | 'C' => 12
    | 'D' => 13
    | 'E' => 14
    | 'F' => 15
    | _ => c.toNat - '0'.toNat
  ) 0

theorem hex_B1F_to_dec : hex_to_dec "B1F" = 2847 := by
  sorry

end NUMINAMATH_CALUDE_hex_B1F_to_dec_l2362_236248


namespace NUMINAMATH_CALUDE_frankie_pet_count_l2362_236282

/-- Represents the number of pets Frankie has of each type -/
structure PetCounts where
  dogs : Nat
  cats : Nat
  parrots : Nat
  snakes : Nat

/-- Calculates the total number of pets -/
def totalPets (p : PetCounts) : Nat :=
  p.dogs + p.cats + p.parrots + p.snakes

/-- Represents the conditions given in the problem -/
structure PetConditions (p : PetCounts) : Prop where
  dog_count : p.dogs = 2
  four_legged : p.dogs + p.cats = 6
  parrot_count : p.parrots = p.cats - 1
  snake_count : p.snakes = p.cats + 6

/-- Theorem stating that given the conditions, Frankie has 19 pets in total -/
theorem frankie_pet_count (p : PetCounts) (h : PetConditions p) : totalPets p = 19 := by
  sorry


end NUMINAMATH_CALUDE_frankie_pet_count_l2362_236282


namespace NUMINAMATH_CALUDE_jane_is_26_l2362_236270

/-- Given Danny's current age and the age difference between Danny and Jane 19 years ago,
    calculates Jane's current age. -/
def janes_current_age (dannys_current_age : ℕ) (years_ago : ℕ) : ℕ :=
  let dannys_age_then := dannys_current_age - years_ago
  let janes_age_then := dannys_age_then / 3
  janes_age_then + years_ago

/-- Proves that Jane's current age is 26, given the problem conditions. -/
theorem jane_is_26 :
  janes_current_age 40 19 = 26 := by
  sorry


end NUMINAMATH_CALUDE_jane_is_26_l2362_236270


namespace NUMINAMATH_CALUDE_equation_solution_l2362_236278

theorem equation_solution (x : ℝ) : 
  (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0 → x = 0 ∨ x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2362_236278


namespace NUMINAMATH_CALUDE_sequence_ratio_theorem_l2362_236266

theorem sequence_ratio_theorem (d : ℝ) (q : ℚ) :
  d ≠ 0 →
  q > 0 →
  let a : ℕ → ℝ := λ n => d * n
  let b : ℕ → ℝ := λ n => d^2 * q^(n-1)
  ∃ k : ℕ+, (a 1)^2 + (a 2)^2 + (a 3)^2 = k * ((b 1) + (b 2) + (b 3)) →
  q = 2 ∨ q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_theorem_l2362_236266


namespace NUMINAMATH_CALUDE_F_of_2_f_of_3_equals_15_l2362_236252

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 2*a
def F (a b : ℝ) : ℝ := b^2 + a*b

-- State the theorem
theorem F_of_2_f_of_3_equals_15 : F 2 (f 3) = 15 := by
  sorry

end NUMINAMATH_CALUDE_F_of_2_f_of_3_equals_15_l2362_236252


namespace NUMINAMATH_CALUDE_order_of_abc_l2362_236296

theorem order_of_abc (a b c : ℝ) 
  (h1 : Real.sqrt (1 + 2*a) = Real.exp b)
  (h2 : Real.exp b = 1 / (1 - c))
  (h3 : 1 / (1 - c) = 1.01) : 
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l2362_236296


namespace NUMINAMATH_CALUDE_length_of_bd_l2362_236242

-- Define the equilateral triangle
def EquilateralTriangle (side_length : ℝ) : Prop :=
  side_length > 0

-- Define points A and C on the sides of the triangle
def PointA (a1 a2 : ℝ) (side_length : ℝ) : Prop :=
  a1 > 0 ∧ a2 > 0 ∧ a1 + a2 = side_length

def PointC (c1 c2 : ℝ) (side_length : ℝ) : Prop :=
  c1 > 0 ∧ c2 > 0 ∧ c1 + c2 = side_length

-- Define the line segment AB and BD
def LineSegments (ab bd : ℝ) : Prop :=
  ab > 0 ∧ bd > 0

-- Theorem statement
theorem length_of_bd
  (side_length : ℝ)
  (a1 a2 c1 c2 ab : ℝ)
  (h1 : EquilateralTriangle side_length)
  (h2 : PointA a1 a2 side_length)
  (h3 : PointC c1 c2 side_length)
  (h4 : LineSegments ab bd)
  (h5 : side_length = 26)
  (h6 : a1 = 3 ∧ a2 = 22)
  (h7 : c1 = 3 ∧ c2 = 23)
  (h8 : ab = 6)
  : bd = 3 := by
  sorry

end NUMINAMATH_CALUDE_length_of_bd_l2362_236242


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l2362_236225

theorem cube_sum_given_sum_and_product (a b : ℝ) 
  (h1 : a + b = 11) (h2 : a * b = 20) : 
  a^3 + b^3 = 671 := by sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l2362_236225


namespace NUMINAMATH_CALUDE_M_greater_than_N_l2362_236234

theorem M_greater_than_N : ∀ a : ℝ, 2 * a * (a - 2) > (a + 1) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_M_greater_than_N_l2362_236234


namespace NUMINAMATH_CALUDE_unique_positive_zero_iff_a_lt_neg_two_l2362_236289

/-- The function f(x) = ax³ - 3x² + 1 has a unique positive zero if and only if a ∈ (-∞, -2) -/
theorem unique_positive_zero_iff_a_lt_neg_two (a : ℝ) :
  (∃! x : ℝ, x > 0 ∧ a * x^3 - 3 * x^2 + 1 = 0) ↔ a < -2 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_zero_iff_a_lt_neg_two_l2362_236289


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2362_236245

/-- Given a triangle ABC with the following properties:
  - b = √2
  - c = 3
  - B + C = 3A
  Prove the following:
  1. a = √5
  2. sin(B + 3π/4) = √10/10
-/
theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  b = Real.sqrt 2 →
  c = 3 →
  B + C = 3 * A →
  a = Real.sqrt 5 ∧ Real.sin (B + 3 * Real.pi / 4) = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2362_236245


namespace NUMINAMATH_CALUDE_calculation_result_l2362_236221

theorem calculation_result : (0.0077 : ℝ) * 4.5 / (0.05 * 0.1 * 0.007) = 990 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l2362_236221


namespace NUMINAMATH_CALUDE_cubic_root_product_theorem_l2362_236251

/-- The cubic polynomial x^3 - 2x^2 + x + k -/
def cubic (k : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + x + k

/-- The condition that the product of roots equals the square of the difference between max and min real roots -/
def root_product_condition (k : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (∀ x, cubic k x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
    a * b * c = (max a (max b c) - min a (min b c))^2

theorem cubic_root_product_theorem : 
  ∀ k : ℝ, root_product_condition k ↔ k = -2 :=
sorry

end NUMINAMATH_CALUDE_cubic_root_product_theorem_l2362_236251


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l2362_236235

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

theorem even_function_implies_a_zero :
  (∀ x, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l2362_236235


namespace NUMINAMATH_CALUDE_inequality_theorem_l2362_236209

theorem inequality_theorem (p q r : ℝ) (n : ℕ) 
  (h_pos_p : p > 0) (h_pos_q : q > 0) (h_pos_r : r > 0) (h_pqr : p * q * r = 1) : 
  (1 / (p^n + q^n + 1)) + (1 / (q^n + r^n + 1)) + (1 / (r^n + p^n + 1)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2362_236209


namespace NUMINAMATH_CALUDE_subset_implies_a_geq_two_l2362_236257

def A : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - (a+1)*x + a ≤ 0}

theorem subset_implies_a_geq_two (a : ℝ) (h : A ⊆ B a) : a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_geq_two_l2362_236257


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l2362_236263

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4} := by sorry

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l2362_236263


namespace NUMINAMATH_CALUDE_sum_of_altitudes_l2362_236283

/-- The sum of altitudes of a triangle formed by the line 10x + 8y = 80 and the coordinate axes --/
theorem sum_of_altitudes (x y : ℝ) : 
  (10 * x + 8 * y = 80) →
  (∃ (a b c : ℝ), 
    (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) ∧
    (10 * a + 8 * b = 80) ∧
    (a + b + c = (18 * Real.sqrt 41 + 40) / Real.sqrt 41) ∧
    (c = 40 / Real.sqrt 41)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_l2362_236283


namespace NUMINAMATH_CALUDE_franks_money_l2362_236299

theorem franks_money (initial_money : ℚ) : 
  (3/4 : ℚ) * ((4/5 : ℚ) * initial_money) = 360 → initial_money = 600 := by
  sorry

end NUMINAMATH_CALUDE_franks_money_l2362_236299


namespace NUMINAMATH_CALUDE_cat_food_consumption_l2362_236250

/-- Represents the amount of food eaten by the cat each day -/
def daily_consumption : ℚ := 1/3 + 1/4

/-- Represents the total number of cans available -/
def total_cans : ℚ := 6

/-- Represents the day on which the cat finishes all the food -/
def finish_day : ℕ := 4

theorem cat_food_consumption :
  ∃ (n : ℕ), n * daily_consumption > total_cans ∧ (n - 1) * daily_consumption ≤ total_cans ∧ n = finish_day :=
by sorry

end NUMINAMATH_CALUDE_cat_food_consumption_l2362_236250


namespace NUMINAMATH_CALUDE_prime_factor_difference_l2362_236243

theorem prime_factor_difference (a b : ℕ) : 
  Prime a → Prime b → b > a → 
  456456 = 2^3 * a * 7 * 11 * 13 * b → 
  b - a = 16 := by
sorry

end NUMINAMATH_CALUDE_prime_factor_difference_l2362_236243


namespace NUMINAMATH_CALUDE_sum_of_seven_thirds_l2362_236218

theorem sum_of_seven_thirds (x : ℚ) : 
  x = 1 / 3 → x + x + x + x + x + x + x = 7 * (1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_thirds_l2362_236218


namespace NUMINAMATH_CALUDE_zeros_imply_a_range_l2362_236288

theorem zeros_imply_a_range (a : ℝ) : 
  (∃ x y, x ∈ (Set.Ioo 0 1) ∧ y ∈ (Set.Ioo 1 2) ∧ 
    x^2 - 2*a*x + 1 = 0 ∧ y^2 - 2*a*y + 1 = 0) → 
  a ∈ (Set.Ioo 1 (5/4)) := by
sorry

end NUMINAMATH_CALUDE_zeros_imply_a_range_l2362_236288


namespace NUMINAMATH_CALUDE_three_white_balls_probability_l2362_236265

/-- The number of white balls in the urn -/
def white_balls : ℕ := 6

/-- The total number of balls in the urn -/
def total_balls : ℕ := 21

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

/-- Probability of drawing 3 white balls without replacement -/
def prob_without_replacement : ℚ := 2 / 133

/-- Probability of drawing 3 white balls with replacement -/
def prob_with_replacement : ℚ := 8 / 343

/-- Probability of drawing 3 white balls simultaneously -/
def prob_simultaneous : ℚ := 2 / 133

/-- Theorem stating the probabilities of drawing 3 white balls under different conditions -/
theorem three_white_balls_probability :
  (Nat.choose white_balls drawn_balls / Nat.choose total_balls drawn_balls : ℚ) = prob_without_replacement ∧
  ((white_balls : ℚ) / total_balls) ^ drawn_balls = prob_with_replacement ∧
  (Nat.choose white_balls drawn_balls / Nat.choose total_balls drawn_balls : ℚ) = prob_simultaneous :=
sorry

end NUMINAMATH_CALUDE_three_white_balls_probability_l2362_236265


namespace NUMINAMATH_CALUDE_prohibited_items_most_suitable_for_census_l2362_236226

/-- Represents a survey type -/
inductive SurveyType
  | CrashResistance
  | ProhibitedItems
  | AppleSweetness
  | WetlandSpecies

/-- Determines if a survey type is suitable for a census -/
def isSuitableForCensus (survey : SurveyType) : Prop :=
  match survey with
  | .ProhibitedItems => true
  | _ => false

/-- Theorem: The survey about prohibited items on high-speed trains is the most suitable for a census -/
theorem prohibited_items_most_suitable_for_census :
  ∀ (survey : SurveyType), isSuitableForCensus survey → survey = SurveyType.ProhibitedItems :=
by
  sorry

#check prohibited_items_most_suitable_for_census

end NUMINAMATH_CALUDE_prohibited_items_most_suitable_for_census_l2362_236226


namespace NUMINAMATH_CALUDE_point_distributive_l2362_236264

/-- Addition of two points in the plane -/
noncomputable def point_add (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Multiplication (midpoint) of two points in the plane -/
noncomputable def point_mul (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: A × (B + C) = (B + A) × (A + C) for any three points A, B, C in the plane -/
theorem point_distributive (A B C : ℝ × ℝ) :
  point_mul A (point_add B C) = point_mul (point_add B A) (point_add A C) :=
sorry

end NUMINAMATH_CALUDE_point_distributive_l2362_236264


namespace NUMINAMATH_CALUDE_angle_C_measure_l2362_236216

/-- Given a triangle ABC, if 3 sin A + 4 cos B = 6 and 3 cos A + 4 sin B = 1, then ∠C = π/6 -/
theorem angle_C_measure (A B C : ℝ) (hsum : A + B + C = π) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) (h2 : 3 * Real.cos A + 4 * Real.sin B = 1) : C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l2362_236216


namespace NUMINAMATH_CALUDE_sequence_contains_intermediate_value_l2362_236212

theorem sequence_contains_intermediate_value 
  (n : ℕ) 
  (a : ℕ → ℤ) 
  (A B : ℤ) 
  (h1 : a 1 < A ∧ A < B ∧ B < a n) 
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i < n → a (i + 1) - a i ≤ 1) :
  ∀ C : ℤ, A ≤ C ∧ C ≤ B → ∃ i₀ : ℕ, 1 < i₀ ∧ i₀ < n ∧ a i₀ = C := by
  sorry

end NUMINAMATH_CALUDE_sequence_contains_intermediate_value_l2362_236212


namespace NUMINAMATH_CALUDE_total_pay_is_550_l2362_236255

/-- The total weekly pay for two employees, where one is paid 150% of the other -/
def total_weekly_pay (b_pay : ℚ) : ℚ :=
  b_pay + (150 / 100) * b_pay

/-- Theorem: Given B is paid 220 per week, the total pay for both employees is 550 -/
theorem total_pay_is_550 : total_weekly_pay 220 = 550 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_is_550_l2362_236255


namespace NUMINAMATH_CALUDE_equation_transformation_l2362_236232

theorem equation_transformation (x : ℝ) (y : ℝ) (h1 : x ≠ 0) (h2 : x^2 ≠ 2) :
  y = (x^2 - 2) / x ∧ (x^2 - 2) / x + 2 * x / (x^2 - 2) = 5 → y^2 - 5*y + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l2362_236232


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_fraction_inequality_solution_l2362_236215

-- Part 1
def quadratic_inequality (x : ℝ) : Prop := x^2 + 3*x - 4 > 0

theorem quadratic_inequality_solution :
  ∀ x : ℝ, quadratic_inequality x ↔ (x > 1 ∨ x < -4) :=
by sorry

-- Part 2
def fraction_inequality (x : ℝ) : Prop := x ≠ 5 ∧ (1 - x) / (x - 5) ≥ 1

theorem fraction_inequality_solution :
  ∀ x : ℝ, fraction_inequality x ↔ (3 ≤ x ∧ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_fraction_inequality_solution_l2362_236215


namespace NUMINAMATH_CALUDE_line_slope_through_points_l2362_236210

/-- The slope of a line passing through points (1,3) and (5,7) is 1 -/
theorem line_slope_through_points : 
  let x1 : ℝ := 1
  let y1 : ℝ := 3
  let x2 : ℝ := 5
  let y2 : ℝ := 7
  let slope := (y2 - y1) / (x2 - x1)
  slope = 1 := by sorry

end NUMINAMATH_CALUDE_line_slope_through_points_l2362_236210


namespace NUMINAMATH_CALUDE_distribute_researchers_count_l2362_236238

/-- The number of ways to distribute 4 researchers to 3 schools -/
def distribute_researchers : ℕ :=
  -- Number of ways to divide 4 researchers into 3 groups (one group of 2, two groups of 1)
  (Nat.choose 4 2) *
  -- Number of ways to assign 3 groups to 3 schools
  (Nat.factorial 3)

/-- Theorem stating that the number of distribution schemes is 36 -/
theorem distribute_researchers_count :
  distribute_researchers = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_researchers_count_l2362_236238


namespace NUMINAMATH_CALUDE_complex_multiplication_sum_l2362_236217

theorem complex_multiplication_sum (z : ℂ) (a b : ℝ) :
  z = 5 + 3 * I →
  I * z = a + b * I →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_multiplication_sum_l2362_236217


namespace NUMINAMATH_CALUDE_f_minimum_and_g_inequality_l2362_236267

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (a * x + x * Real.cos x + 1)

theorem f_minimum_and_g_inequality :
  (∃! x : ℝ, ∀ y : ℝ, f y ≥ f x) ∧ f 0 = 0 ∧
  ∀ a : ℝ, a > -1 → ∀ x : ℝ, x > 0 ∧ x < 1 → g a x > 1 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_and_g_inequality_l2362_236267


namespace NUMINAMATH_CALUDE_geometric_sequence_11th_term_l2362_236214

/-- Given a geometric sequence where a₅ = 16 and a₈ = 4√2, prove that a₁₁ = 2√2 -/
theorem geometric_sequence_11th_term 
  (a : ℕ → ℝ) 
  (is_geometric : ∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n) ^ m)
  (a5 : a 5 = 16)
  (a8 : a 8 = 4 * Real.sqrt 2) :
  a 11 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_11th_term_l2362_236214


namespace NUMINAMATH_CALUDE_max_valid_sequence_length_l2362_236298

/-- A sequence of integers satisfying the given property -/
def ValidSequence (x : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i ≤ j → j ≤ n → i + j ≤ n →
    (3 ∣ x i - x j) → (3 ∣ x (i + j) + x i + x j + 1)

/-- The maximum length of a valid sequence -/
def MaxValidSequenceLength : ℕ := 8

/-- Theorem stating that the maximum length of a valid sequence is 8 -/
theorem max_valid_sequence_length :
  (∃ x, ValidSequence x MaxValidSequenceLength) ∧
  (∀ n > MaxValidSequenceLength, ¬∃ x, ValidSequence x n) :=
sorry

end NUMINAMATH_CALUDE_max_valid_sequence_length_l2362_236298


namespace NUMINAMATH_CALUDE_nala_seashells_l2362_236231

/-- The number of seashells Nala found on the first day -/
def first_day : ℕ := 5

/-- The number of seashells Nala found on the second day -/
def second_day : ℕ := 7

/-- The number of seashells Nala found on the third day is twice the sum of the first two days -/
def third_day : ℕ := 2 * (first_day + second_day)

/-- The total number of seashells Nala has -/
def total_seashells : ℕ := first_day + second_day + third_day

theorem nala_seashells : total_seashells = 36 := by
  sorry

end NUMINAMATH_CALUDE_nala_seashells_l2362_236231


namespace NUMINAMATH_CALUDE_final_position_l2362_236277

-- Define the ant's position type
def Position := ℤ × ℤ

-- Define the direction type
inductive Direction
| East
| North
| West
| South

-- Define the function to get the next direction
def nextDirection (d : Direction) : Direction :=
  match d with
  | Direction.East => Direction.North
  | Direction.North => Direction.West
  | Direction.West => Direction.South
  | Direction.South => Direction.East

-- Define the function to move in a direction
def move (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.East => (p.1 + 1, p.2)
  | Direction.North => (p.1, p.2 + 1)
  | Direction.West => (p.1 - 1, p.2)
  | Direction.South => (p.1, p.2 - 1)

-- Define the function to calculate the position after n steps
def positionAfterSteps (n : ℕ) : Position :=
  sorry -- Proof implementation goes here

-- The main theorem
theorem final_position : positionAfterSteps 2015 = (13, -22) := by
  sorry -- Proof implementation goes here

end NUMINAMATH_CALUDE_final_position_l2362_236277


namespace NUMINAMATH_CALUDE_no_valid_operation_l2362_236259

-- Define the set of standard arithmetic operations
inductive ArithOp
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply an arithmetic operation
def applyOp (op : ArithOp) (a b : ℤ) : ℚ :=
  match op with
  | ArithOp.Add => a + b
  | ArithOp.Sub => a - b
  | ArithOp.Mul => a * b
  | ArithOp.Div => (a : ℚ) / (b : ℚ)

-- Theorem statement
theorem no_valid_operation : ∀ (op : ArithOp), 
  (applyOp op 9 3 : ℚ) + 5 - (4 - 2) ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_operation_l2362_236259


namespace NUMINAMATH_CALUDE_square_root_of_121_l2362_236249

theorem square_root_of_121 : ∀ x : ℝ, x^2 = 121 ↔ x = 11 ∨ x = -11 := by sorry

end NUMINAMATH_CALUDE_square_root_of_121_l2362_236249


namespace NUMINAMATH_CALUDE_sum_of_abs_and_square_zero_l2362_236268

theorem sum_of_abs_and_square_zero (x y : ℝ) :
  |x + 3| + (2 * y - 5)^2 = 0 → x + 2 * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abs_and_square_zero_l2362_236268


namespace NUMINAMATH_CALUDE_percentage_within_one_std_dev_l2362_236292

-- Define a symmetric distribution
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percentage_less_than_mean_plus_std : ℝ

-- Theorem statement
theorem percentage_within_one_std_dev 
  (d : SymmetricDistribution) 
  (h1 : d.is_symmetric = true) 
  (h2 : d.percentage_less_than_mean_plus_std = 84) : 
  (d.percentage_less_than_mean_plus_std - (100 - d.percentage_less_than_mean_plus_std)) = 68 := by
  sorry

end NUMINAMATH_CALUDE_percentage_within_one_std_dev_l2362_236292


namespace NUMINAMATH_CALUDE_product_of_primes_sum_99_l2362_236260

theorem product_of_primes_sum_99 (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 99 → p * q = 194 := by sorry

end NUMINAMATH_CALUDE_product_of_primes_sum_99_l2362_236260


namespace NUMINAMATH_CALUDE_triangle_area_bounds_l2362_236223

def triangle_area (s : ℝ) : ℝ := (s + 2)^(4/3)

theorem triangle_area_bounds :
  ∀ s : ℝ, (2^(1/2) * 3^(1/4) ≤ s ∧ s ≤ 3^(2/3) * 2^(1/3)) ↔
    (12 ≤ triangle_area s ∧ triangle_area s ≤ 72) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_bounds_l2362_236223


namespace NUMINAMATH_CALUDE_integer_quotient_problem_l2362_236258

theorem integer_quotient_problem (x y : ℤ) : 
  1996 * x + y / 96 = x + y → y / x = 2016 ∨ x / y = 1 / 2016 := by
  sorry

end NUMINAMATH_CALUDE_integer_quotient_problem_l2362_236258


namespace NUMINAMATH_CALUDE_solution_existence_conditions_l2362_236211

theorem solution_existence_conditions (a b : ℝ) :
  (∃ x y : ℝ, (Real.tan x) * (Real.tan y) = a ∧ (Real.sin x)^2 + (Real.sin y)^2 = b^2) ↔
  (1 < b^2 ∧ b^2 < 2*a/(a+1)) ∨ (1 < b^2 ∧ b^2 < 2*a/(a-1)) := by
  sorry

end NUMINAMATH_CALUDE_solution_existence_conditions_l2362_236211


namespace NUMINAMATH_CALUDE_bulls_and_heat_wins_l2362_236222

/-- The number of games won by the Chicago Bulls and Miami Heat combined in 2010 -/
theorem bulls_and_heat_wins (bulls_wins : ℕ) (heat_wins : ℕ) : 
  bulls_wins = 70 →
  heat_wins = bulls_wins + 5 →
  bulls_wins + heat_wins = 145 := by
  sorry

end NUMINAMATH_CALUDE_bulls_and_heat_wins_l2362_236222


namespace NUMINAMATH_CALUDE_population_after_two_years_l2362_236279

def initial_population : ℕ := 1000
def year1_increase : ℚ := 20 / 100
def year2_increase : ℚ := 30 / 100

theorem population_after_two_years :
  let year1_population := initial_population * (1 + year1_increase)
  let year2_population := year1_population * (1 + year2_increase)
  ↑(round year2_population) = 1560 := by sorry

end NUMINAMATH_CALUDE_population_after_two_years_l2362_236279


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2362_236233

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e,
    if there exists a point P on the ellipse such that the angle F₁PF₂ is 60°,
    then the eccentricity e is in the range [1/2, 1). -/
theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∃ (e : ℝ), e = Real.sqrt (1 - b^2 / a^2) ∧
    ∃ (F1 F2 : ℝ × ℝ),
      F1 = (-a * e, 0) ∧ F2 = (a * e, 0) ∧
      Real.cos (60 * π / 180) = ((x - F1.1)^2 + (y - F1.2)^2 + (x - F2.1)^2 + (y - F2.2)^2 - 4 * a^2 * e^2) /
        (2 * Real.sqrt ((x - F1.1)^2 + (y - F1.2)^2) * Real.sqrt ((x - F2.1)^2 + (y - F2.2)^2))) →
  1/2 ≤ e ∧ e < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2362_236233


namespace NUMINAMATH_CALUDE_evaluate_expression_l2362_236227

theorem evaluate_expression : (1023 : ℕ) * 1023 - 1022 * 1024 = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2362_236227


namespace NUMINAMATH_CALUDE_min_students_l2362_236272

/-- Represents a student in the math competition -/
structure Student where
  solved : Finset (Fin 6)

/-- Represents the math competition -/
structure MathCompetition where
  students : Finset Student
  problem_count : Nat
  students_per_problem : Nat

/-- The conditions of the math competition -/
def validCompetition (c : MathCompetition) : Prop :=
  c.problem_count = 6 ∧
  c.students_per_problem = 500 ∧
  (∀ p : Fin 6, (c.students.filter (fun s => p ∈ s.solved)).card = c.students_per_problem) ∧
  (∀ s₁ s₂ : Student, s₁ ∈ c.students → s₂ ∈ c.students → s₁ ≠ s₂ → 
    ∃ p : Fin 6, p ∉ s₁.solved ∧ p ∉ s₂.solved)

/-- The theorem to be proved -/
theorem min_students (c : MathCompetition) (h : validCompetition c) : 
  c.students.card ≥ 1000 := by
  sorry

end NUMINAMATH_CALUDE_min_students_l2362_236272


namespace NUMINAMATH_CALUDE_combined_resistance_of_parallel_resistors_l2362_236240

def parallel_resistance (r1 r2 r3 : ℚ) : ℚ := 1 / (1 / r1 + 1 / r2 + 1 / r3)

theorem combined_resistance_of_parallel_resistors :
  let r1 : ℚ := 2
  let r2 : ℚ := 5
  let r3 : ℚ := 6
  let r : ℚ := parallel_resistance r1 r2 r3
  r = 15 / 13 := by sorry

end NUMINAMATH_CALUDE_combined_resistance_of_parallel_resistors_l2362_236240


namespace NUMINAMATH_CALUDE_consecutive_integers_base_equation_l2362_236276

/-- Given two consecutive positive integers A and B that satisfy the equation
    132_A + 43_B = 69_(A+B), prove that A + B = 13 -/
theorem consecutive_integers_base_equation (A B : ℕ) : 
  A > 0 ∧ B > 0 ∧ (B = A + 1 ∨ A = B + 1) →
  (A^2 + 3*A + 2) + (4*B + 3) = 6*(A + B) + 9 →
  A + B = 13 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_base_equation_l2362_236276


namespace NUMINAMATH_CALUDE_function_inequality_implies_b_bound_l2362_236275

open Real

theorem function_inequality_implies_b_bound (b : ℝ) :
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, exp x * (x - b) + x * exp x * (x + 1 - b) > 0) →
  b < 8/3 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_b_bound_l2362_236275


namespace NUMINAMATH_CALUDE_symmetric_points_sqrt_l2362_236236

/-- Given that point P(3, -1) is symmetric to point Q(a+b, 1-b) about the y-axis,
    prove that the square root of -ab equals √10. -/
theorem symmetric_points_sqrt (a b : ℝ) : 
  (3 = -(a + b) ∧ -1 = 1 - b) → Real.sqrt (-a * b) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sqrt_l2362_236236


namespace NUMINAMATH_CALUDE_sheep_purchase_equation_l2362_236256

/-- Represents a group of people jointly buying sheep -/
structure SheepPurchase where
  x : ℕ  -- number of people
  price : ℕ  -- price of the sheep

/-- The equation holds for the given conditions -/
theorem sheep_purchase_equation (sp : SheepPurchase) : 
  (5 * sp.x + 45 = sp.price) ∧ (7 * sp.x - 3 = sp.price) → 5 * sp.x + 45 = 7 * sp.x + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sheep_purchase_equation_l2362_236256


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l2362_236297

/-- Calculates the distance traveled downstream by a boat given its own speed, speed against current, and time. -/
def distance_downstream (boat_speed : ℝ) (speed_against_current : ℝ) (time : ℝ) : ℝ :=
  let current_speed : ℝ := boat_speed - speed_against_current
  let downstream_speed : ℝ := boat_speed + current_speed
  downstream_speed * time

/-- Proves that a boat with given specifications travels 255 km downstream in 6 hours. -/
theorem boat_downstream_distance :
  distance_downstream 40 37.5 6 = 255 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l2362_236297


namespace NUMINAMATH_CALUDE_hyperbola_standard_form_l2362_236286

theorem hyperbola_standard_form (x y : ℝ) :
  x^2 - 15 * y^2 = 15 ↔ x^2 / 15 - y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_form_l2362_236286


namespace NUMINAMATH_CALUDE_sanitizer_dilution_l2362_236285

/-- Proves that adding 6 ounces of water to 12 ounces of hand sanitizer with 60% alcohol
    concentration results in a solution with 40% alcohol concentration. -/
theorem sanitizer_dilution (initial_volume : ℝ) (initial_concentration : ℝ)
    (water_added : ℝ) (final_concentration : ℝ)
    (h1 : initial_volume = 12)
    (h2 : initial_concentration = 0.6)
    (h3 : water_added = 6)
    (h4 : final_concentration = 0.4) :
  initial_concentration * initial_volume =
    final_concentration * (initial_volume + water_added) :=
by sorry

end NUMINAMATH_CALUDE_sanitizer_dilution_l2362_236285


namespace NUMINAMATH_CALUDE_locus_is_extended_rectangle_l2362_236290

/-- A line in a plane --/
structure Line where
  -- We assume some representation of a line
  mk :: (dummy : Unit)

/-- Distance between a point and a line --/
noncomputable def dist (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- The locus of points with constant difference of distances from two lines --/
def locus (l₁ l₂ : Line) (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |dist p l₁ - dist p l₂| = a}

/-- A rectangle in a plane --/
structure Rectangle where
  -- We assume some representation of a rectangle
  mk :: (dummy : Unit)

/-- The sides of a rectangle extended infinitely --/
def extended_sides (r : Rectangle) : Set (ℝ × ℝ) := sorry

/-- Construct a rectangle from two lines and a distance --/
def construct_rectangle (l₁ l₂ : Line) (a : ℝ) : Rectangle := sorry

/-- The main theorem --/
theorem locus_is_extended_rectangle (l₁ l₂ : Line) (a : ℝ) :
  locus l₁ l₂ a = extended_sides (construct_rectangle l₁ l₂ a) := by sorry

end NUMINAMATH_CALUDE_locus_is_extended_rectangle_l2362_236290


namespace NUMINAMATH_CALUDE_square_difference_identity_simplify_expression_l2362_236219

theorem square_difference_identity (a b : ℝ) : (a - b)^2 = a^2 + b^2 - 2*a*b := by sorry

theorem simplify_expression : 2021^2 - 2021 * 4034 + 2017^2 = 16 := by
  have h : ∀ (x y : ℝ), (x - y)^2 = x^2 + y^2 - 2*x*y := square_difference_identity
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_simplify_expression_l2362_236219


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2362_236262

theorem water_tank_capacity (w c : ℝ) (h1 : w / c = 1 / 5) (h2 : (w + 3) / c = 1 / 4) : c = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l2362_236262


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2362_236273

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 64

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := x + Real.sqrt 3 * y = 0

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 36 - y^2 / 12 = 1

-- Theorem statement
theorem hyperbola_equation 
  (h1 : ∀ x y, ellipse x y ↔ hyperbola x y)  -- Same foci condition
  (h2 : ∃ x y, hyperbola x y ∧ asymptote x y)  -- Asymptote condition
  : ∀ x y, hyperbola x y ↔ x^2 / 36 - y^2 / 12 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2362_236273


namespace NUMINAMATH_CALUDE_max_a_inequality_max_a_equality_l2362_236220

theorem max_a_inequality (a : ℝ) : 
  (∀ x > 0, Real.log (a * x) + a * x ≤ x + Real.exp x) → a ≤ Real.exp 1 := by
  sorry

theorem max_a_equality : 
  ∃ a : ℝ, a = Real.exp 1 ∧ (∀ x > 0, Real.log (a * x) + a * x ≤ x + Real.exp x) := by
  sorry

end NUMINAMATH_CALUDE_max_a_inequality_max_a_equality_l2362_236220
