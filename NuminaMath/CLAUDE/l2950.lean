import Mathlib

namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l2950_295020

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel 
  (l : Line) (m : Line) (α : Plane) (β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β) :
  (parallel α β → perpendicular_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l2950_295020


namespace NUMINAMATH_CALUDE_smallest_with_ten_divisors_l2950_295097

/-- A function that returns the number of positive integer divisors of a given natural number. -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 10 positive integer divisors. -/
def has_ten_divisors (n : ℕ) : Prop := num_divisors n = 10

/-- Theorem stating that 48 is the smallest positive integer with exactly 10 positive integer divisors. -/
theorem smallest_with_ten_divisors : 
  has_ten_divisors 48 ∧ ∀ m : ℕ, m < 48 → ¬(has_ten_divisors m) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_ten_divisors_l2950_295097


namespace NUMINAMATH_CALUDE_system_solution_l2950_295021

theorem system_solution (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 - y^2 + x + y = 44) :
  x^2 + y^2 = 109 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2950_295021


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2950_295064

def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_properties (f : ℝ → ℝ) 
  (h1 : quadratic_function f) 
  (h2 : f 0 = 1) 
  (h3 : ∀ x, f (x + 1) - f x = 2 * x) :
  (∀ x, f x = x^2 - x + 1) ∧ 
  Set.Icc (3/4 : ℝ) 3 = {y | ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2950_295064


namespace NUMINAMATH_CALUDE_quadratic_rational_roots_l2950_295083

/-- Given positive prime numbers p and q, the equation x^2 + p^2x + q^3 = 0 has rational roots if and only if p = 3 and q = 2 -/
theorem quadratic_rational_roots (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ x : ℚ, x^2 + p^2*x + q^3 = 0) ↔ (p = 3 ∧ q = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_roots_l2950_295083


namespace NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l2950_295078

theorem video_recorder_wholesale_cost :
  ∀ (wholesale : ℝ),
  let retail := wholesale * 1.20
  let employee_price := retail * 0.75
  employee_price = 180 →
  wholesale = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l2950_295078


namespace NUMINAMATH_CALUDE_base_of_equation_l2950_295049

theorem base_of_equation (k x : ℝ) (h1 : (1/2)^23 * (1/81)^k = 1/x^23) (h2 : k = 11.5) : x = 18 := by
  sorry

end NUMINAMATH_CALUDE_base_of_equation_l2950_295049


namespace NUMINAMATH_CALUDE_min_sum_product_l2950_295035

theorem min_sum_product (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 9/n = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 1 → m + n ≤ a + b) →
  m * n = 48 := by
sorry

end NUMINAMATH_CALUDE_min_sum_product_l2950_295035


namespace NUMINAMATH_CALUDE_fraction_equality_l2950_295067

theorem fraction_equality : (2015 : ℚ) / (2015^2 - 2016 * 2014) = 2015 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2950_295067


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2950_295092

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 < 0) ↔ (∃ x : ℝ, x^2 - x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2950_295092


namespace NUMINAMATH_CALUDE_parker_dumbbells_l2950_295075

/-- Given an initial number of dumbbells, the weight of each dumbbell, 
    and the total weight being used, calculate the number of additional dumbbells needed. -/
def additional_dumbbells (initial_count : ℕ) (weight_per_dumbbell : ℕ) (total_weight : ℕ) : ℕ :=
  ((total_weight - initial_count * weight_per_dumbbell) / weight_per_dumbbell)

/-- Theorem stating that given 4 initial dumbbells of 20 pounds each, 
    and a total weight of 120 pounds, the number of additional dumbbells needed is 2. -/
theorem parker_dumbbells : 
  additional_dumbbells 4 20 120 = 2 := by
  sorry

end NUMINAMATH_CALUDE_parker_dumbbells_l2950_295075


namespace NUMINAMATH_CALUDE_words_with_e_count_l2950_295089

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E'}
def word_length : Nat := 4

def total_words : Nat := alphabet.card ^ word_length

def words_without_e : Nat := (alphabet.card - 1) ^ word_length

theorem words_with_e_count : 
  total_words - words_without_e = 369 := by sorry

end NUMINAMATH_CALUDE_words_with_e_count_l2950_295089


namespace NUMINAMATH_CALUDE_price_per_chicken_l2950_295000

/-- Given Alan's market purchases, prove the price per chicken --/
theorem price_per_chicken (num_eggs : ℕ) (price_per_egg : ℕ) (num_chickens : ℕ) (total_spent : ℕ) :
  num_eggs = 20 →
  price_per_egg = 2 →
  num_chickens = 6 →
  total_spent = 88 →
  (total_spent - num_eggs * price_per_egg) / num_chickens = 8 := by
  sorry

end NUMINAMATH_CALUDE_price_per_chicken_l2950_295000


namespace NUMINAMATH_CALUDE_count_numbers_with_at_least_two_zeros_l2950_295044

/-- The number of digits in the numbers we're considering -/
def n : ℕ := 6

/-- The total number of n-digit numbers -/
def total_n_digit_numbers : ℕ := 9 * 10^(n-1)

/-- The number of n-digit numbers with no zeros -/
def numbers_with_no_zeros : ℕ := 9^n

/-- The number of n-digit numbers with exactly one zero -/
def numbers_with_one_zero : ℕ := n * 9^(n-1)

/-- The number of n-digit numbers with at least two zeros -/
def numbers_with_at_least_two_zeros : ℕ :=
  total_n_digit_numbers - numbers_with_no_zeros - numbers_with_one_zero

theorem count_numbers_with_at_least_two_zeros :
  numbers_with_at_least_two_zeros = 14265 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_at_least_two_zeros_l2950_295044


namespace NUMINAMATH_CALUDE_min_blocks_for_wall_l2950_295085

/-- Represents the dimensions of a block -/
structure Block where
  length : Nat
  height : Nat

/-- Represents the dimensions of the wall -/
structure Wall where
  length : Nat
  height : Nat

/-- Calculates the minimum number of blocks needed to build the wall -/
def minBlocksNeeded (wall : Wall) (blocks : List Block) : Nat :=
  sorry

/-- The theorem to be proven -/
theorem min_blocks_for_wall :
  let wall : Wall := { length := 120, height := 9 }
  let blocks : List Block := [
    { length := 3, height := 1 },
    { length := 2, height := 1 },
    { length := 1, height := 1 }
  ]
  minBlocksNeeded wall blocks = 365 := by sorry

end NUMINAMATH_CALUDE_min_blocks_for_wall_l2950_295085


namespace NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l2950_295047

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- Main theorem
theorem smallest_prime_12_less_than_square : 
  ∀ n : ℕ, n > 0 → is_prime n → (∃ m : ℕ, is_perfect_square m ∧ n = m - 12) → n ≥ 13 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l2950_295047


namespace NUMINAMATH_CALUDE_brown_eyes_light_brown_skin_l2950_295087

/-- Represents the characteristics of the group of girls -/
structure GirlGroup where
  total : Nat
  blue_eyes_fair_skin : Nat
  light_brown_skin : Nat
  brown_eyes : Nat

/-- Theorem stating the number of girls with brown eyes and light brown skin -/
theorem brown_eyes_light_brown_skin (g : GirlGroup) 
  (h1 : g.total = 50)
  (h2 : g.blue_eyes_fair_skin = 14)
  (h3 : g.light_brown_skin = 31)
  (h4 : g.brown_eyes = 18) :
  g.brown_eyes - (g.total - g.light_brown_skin - g.blue_eyes_fair_skin) = 13 := by
  sorry

#check brown_eyes_light_brown_skin

end NUMINAMATH_CALUDE_brown_eyes_light_brown_skin_l2950_295087


namespace NUMINAMATH_CALUDE_inequality_proof_l2950_295036

theorem inequality_proof (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > a*x ∧ a*x > a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2950_295036


namespace NUMINAMATH_CALUDE_valid_k_values_l2950_295050

/-- A function f: ℤ → ℤ satisfies the given property for a positive integer k -/
def satisfies_property (f : ℤ → ℤ) (k : ℕ+) : Prop :=
  ∀ (a b c : ℤ), a + b + c = 0 →
    f a + f b + f c = (f (a - b) + f (b - c) + f (c - a)) / k

/-- A function f: ℤ → ℤ is nonlinear -/
def is_nonlinear (f : ℤ → ℤ) : Prop :=
  ∃ (a b x y : ℤ), f (a + x) + f (b + y) ≠ f a + f b + f x + f y

theorem valid_k_values :
  {k : ℕ+ | ∃ (f : ℤ → ℤ), satisfies_property f k ∧ is_nonlinear f} = {1, 3, 9} := by
  sorry

end NUMINAMATH_CALUDE_valid_k_values_l2950_295050


namespace NUMINAMATH_CALUDE_mitzel_allowance_left_l2950_295084

/-- Proves that the amount left in Mitzel's allowance is $26, given that she spent 35% of her allowance, which amounts to $14. -/
theorem mitzel_allowance_left (spent_percentage : ℝ) (spent_amount : ℝ) (total_allowance : ℝ) :
  spent_percentage = 0.35 →
  spent_amount = 14 →
  spent_amount = spent_percentage * total_allowance →
  total_allowance - spent_amount = 26 :=
by sorry

end NUMINAMATH_CALUDE_mitzel_allowance_left_l2950_295084


namespace NUMINAMATH_CALUDE_arithmetic_trapezoid_third_largest_angle_l2950_295095

/-- Represents a trapezoid with angles in arithmetic sequence -/
structure ArithmeticTrapezoid where
  /-- The smallest angle of the trapezoid -/
  smallest_angle : ℝ
  /-- The common difference between consecutive angles -/
  angle_diff : ℝ

/-- The theorem statement -/
theorem arithmetic_trapezoid_third_largest_angle
  (trap : ArithmeticTrapezoid)
  (sum_smallest_largest : trap.smallest_angle + (trap.smallest_angle + 3 * trap.angle_diff) = 200)
  (second_smallest : trap.smallest_angle + trap.angle_diff = 70) :
  trap.smallest_angle + 2 * trap.angle_diff = 130 := by
  sorry

#check arithmetic_trapezoid_third_largest_angle

end NUMINAMATH_CALUDE_arithmetic_trapezoid_third_largest_angle_l2950_295095


namespace NUMINAMATH_CALUDE_annalise_purchase_l2950_295070

/-- Represents the purchase of tissue boxes -/
structure TissuePurchase where
  packs_per_box : ℕ
  tissues_per_pack : ℕ
  tissue_cost_cents : ℕ
  total_spent_dollars : ℕ

/-- Calculates the number of boxes bought given a TissuePurchase -/
def boxes_bought (purchase : TissuePurchase) : ℕ :=
  (purchase.total_spent_dollars * 100) /
  (purchase.packs_per_box * purchase.tissues_per_pack * purchase.tissue_cost_cents)

/-- Theorem stating that Annalise bought 10 boxes -/
theorem annalise_purchase : 
  let purchase := TissuePurchase.mk 20 100 5 1000
  boxes_bought purchase = 10 := by
  sorry

end NUMINAMATH_CALUDE_annalise_purchase_l2950_295070


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l2950_295098

/-- A linear function f(x) = mx + b passes through a quadrant if there exists a point (x, y) in that quadrant such that y = f(x) -/
def passes_through_quadrant (m b : ℝ) (q : Nat) : Prop :=
  match q with
  | 1 => ∃ x y, x > 0 ∧ y > 0 ∧ y = m * x + b
  | 2 => ∃ x y, x < 0 ∧ y > 0 ∧ y = m * x + b
  | 3 => ∃ x y, x < 0 ∧ y < 0 ∧ y = m * x + b
  | 4 => ∃ x y, x > 0 ∧ y < 0 ∧ y = m * x + b
  | _ => False

/-- The linear function y = -2x + 1 passes through Quadrants I, II, and IV -/
theorem linear_function_quadrants :
  passes_through_quadrant (-2) 1 1 ∧
  passes_through_quadrant (-2) 1 2 ∧
  passes_through_quadrant (-2) 1 4 :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l2950_295098


namespace NUMINAMATH_CALUDE_solution_exists_l2950_295082

theorem solution_exists (x y : ℝ) : (2*x - 3*y + 5)^2 + |x - y + 2| = 0 → x = -1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l2950_295082


namespace NUMINAMATH_CALUDE_crayon_purchase_l2950_295063

def half_dozen : ℕ := 6

theorem crayon_purchase (total_cost : ℕ) (cost_per_crayon : ℕ) (half_dozens : ℕ) : 
  total_cost = 48 ∧ 
  cost_per_crayon = 2 ∧ 
  total_cost = half_dozens * half_dozen * cost_per_crayon →
  half_dozens = 4 := by
sorry

end NUMINAMATH_CALUDE_crayon_purchase_l2950_295063


namespace NUMINAMATH_CALUDE_dance_troupe_members_l2950_295034

theorem dance_troupe_members : ∃! n : ℕ, 
  150 < n ∧ n < 300 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 3 ∧ 
  n % 9 = 4 ∧ 
  n = 176 := by
  sorry

end NUMINAMATH_CALUDE_dance_troupe_members_l2950_295034


namespace NUMINAMATH_CALUDE_complex_fourth_power_problem_l2950_295001

theorem complex_fourth_power_problem : ∃ (d : ℤ), (1 + 3*I : ℂ)^4 = 82 + d*I := by sorry

end NUMINAMATH_CALUDE_complex_fourth_power_problem_l2950_295001


namespace NUMINAMATH_CALUDE_integer_representation_l2950_295016

theorem integer_representation (n : ℤ) : ∃ x y z : ℤ, n = x^2 + y^2 - z^2 := by
  sorry

end NUMINAMATH_CALUDE_integer_representation_l2950_295016


namespace NUMINAMATH_CALUDE_circle_m_range_l2950_295017

-- Define the equation as a function of x, y, and m
def circle_equation (x y m : ℝ) : ℝ := x^2 + y^2 - 2*m*x + 2*m^2 + 2*m - 3

-- Define what it means for the equation to represent a circle
def is_circle (m : ℝ) : Prop :=
  ∃ (a b r : ℝ), r > 0 ∧ ∀ (x y : ℝ),
    circle_equation x y m = 0 ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem stating the range of m for which the equation represents a circle
theorem circle_m_range :
  ∀ m : ℝ, is_circle m ↔ -3 < m ∧ m < 1/2 := by sorry

end NUMINAMATH_CALUDE_circle_m_range_l2950_295017


namespace NUMINAMATH_CALUDE_van_distance_proof_l2950_295031

/-- Proves that the distance covered by a van is 180 km given specific conditions -/
theorem van_distance_proof (D : ℝ) (original_time : ℝ) (new_time : ℝ) (new_speed : ℝ) :
  original_time = 6 →
  new_time = 3/2 * original_time →
  new_speed = 20 →
  D = new_speed * new_time →
  D = 180 := by
  sorry

#check van_distance_proof

end NUMINAMATH_CALUDE_van_distance_proof_l2950_295031


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l2950_295048

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular solid given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- Represents the box dimensions -/
def box : Dimensions := ⟨3, 4, 2⟩

/-- Represents the block dimensions -/
def block : Dimensions := ⟨2, 1, 2⟩

/-- Theorem stating that the maximum number of blocks that can fit in the box is 6 -/
theorem max_blocks_in_box :
  ∃ (n : ℕ), n = 6 ∧ 
  n * volume block ≤ volume box ∧
  ∀ m : ℕ, m * volume block ≤ volume box → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_blocks_in_box_l2950_295048


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_sum_l2950_295045

theorem smallest_n_for_integer_sum : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 6 + (1 : ℚ) / n = k) ∧
  (∀ (m : ℕ), m > 0 → m < n → 
    ¬∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 6 + (1 : ℚ) / m = k) ∧
  n = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_sum_l2950_295045


namespace NUMINAMATH_CALUDE_sequence_term_l2950_295011

theorem sequence_term (n : ℕ) (a : ℕ → ℝ) : 
  (∀ k, a k = Real.sqrt (2 * k - 1)) → 
  a 23 = 3 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_term_l2950_295011


namespace NUMINAMATH_CALUDE_sum_ge_sum_of_abs_div_three_l2950_295068

theorem sum_ge_sum_of_abs_div_three (a b c : ℝ) 
  (hab : a + b ≥ 0) (hbc : b + c ≥ 0) (hca : c + a ≥ 0) :
  a + b + c ≥ (|a| + |b| + |c|) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_ge_sum_of_abs_div_three_l2950_295068


namespace NUMINAMATH_CALUDE_identity_iff_annihilator_l2950_295022

variable (R : Type) [Fintype R] [CommRing R]

def has_multiplicative_identity (R : Type) [Ring R] : Prop :=
  ∃ e : R, ∀ x : R, e * x = x ∧ x * e = x

def annihilator_is_zero (R : Type) [Ring R] : Prop :=
  ∀ a : R, (∀ x : R, a * x = 0) → a = 0

theorem identity_iff_annihilator (R : Type) [Fintype R] [CommRing R] :
  has_multiplicative_identity R ↔ annihilator_is_zero R :=
sorry

end NUMINAMATH_CALUDE_identity_iff_annihilator_l2950_295022


namespace NUMINAMATH_CALUDE_seedling_problem_l2950_295002

-- Define variables for seedling prices
variable (x y : ℚ)

-- Define conditions
def condition1 : Prop := 3 * x + 2 * y = 12
def condition2 : Prop := x + 3 * y = 11

-- Define total number of seedlings
def total_seedlings : ℕ := 200

-- Define value multiplier
def value_multiplier : ℕ := 100

-- Define minimum total value
def min_total_value : ℕ := 50000

-- Theorem to prove
theorem seedling_problem (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 2 ∧ y = 3 ∧
  ∃ m : ℕ, m ≥ 100 ∧
  m ≤ total_seedlings ∧
  2 * value_multiplier * (total_seedlings - m) + 3 * value_multiplier * m ≥ min_total_value ∧
  ∀ n : ℕ, n < m →
    2 * value_multiplier * (total_seedlings - n) + 3 * value_multiplier * n < min_total_value :=
by
  sorry

end NUMINAMATH_CALUDE_seedling_problem_l2950_295002


namespace NUMINAMATH_CALUDE_ellipse_equation_l2950_295037

/-- Given a hyperbola and an ellipse with specific properties, prove the equation of the ellipse -/
theorem ellipse_equation (h : ℝ → ℝ → Prop) (e : ℝ → ℝ → Prop) :
  (∀ x y, h x y ↔ y^2/12 - x^2/4 = 1) →  -- Definition of the hyperbola
  (∃ a b, ∀ x y, e x y ↔ x^2/a + y^2/b = 1) →  -- General form of the ellipse
  (∃ v₁ v₂, v₁ ≠ v₂ ∧ h 0 v₁ ∧ h 0 (-v₁) ∧ 
    ∀ x y, e x y → (x - 0)^2 + (y - v₁)^2 + (x - 0)^2 + (y + v₁)^2 = 16) →  -- Vertices of hyperbola as foci of ellipse
  (∀ x y, e x y ↔ x^2/4 + y^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2950_295037


namespace NUMINAMATH_CALUDE_prob_spade_seven_red_l2950_295018

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of sevens in a standard deck -/
def NumSevens : ℕ := 4

/-- Number of red cards in a standard deck -/
def NumRed : ℕ := 26

/-- Probability of drawing a spade, then a 7, then a red card from a standard 52-card deck -/
theorem prob_spade_seven_red (deck : ℕ) (spades : ℕ) (sevens : ℕ) (red : ℕ) :
  deck = StandardDeck →
  spades = NumSpades →
  sevens = NumSevens →
  red = NumRed →
  (spades / deck) * (sevens / (deck - 1)) * (red / (deck - 2)) = 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_prob_spade_seven_red_l2950_295018


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2950_295024

theorem necessary_not_sufficient_condition : 
  (∀ x : ℝ, x - 1 = 0 → (x - 1) * (x - 2) = 0) ∧ 
  (∃ x : ℝ, (x - 1) * (x - 2) = 0 ∧ x - 1 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2950_295024


namespace NUMINAMATH_CALUDE_roberta_initial_records_l2950_295053

/-- The number of records Roberta initially had -/
def initial_records : ℕ := sorry

/-- The number of records Roberta received as gifts -/
def gifted_records : ℕ := 12

/-- The number of records Roberta bought at a garage sale -/
def bought_records : ℕ := 30

/-- The number of days it takes Roberta to listen to one record -/
def days_per_record : ℕ := 2

/-- The total number of days it will take Roberta to listen to her entire collection -/
def total_listening_days : ℕ := 100

theorem roberta_initial_records :
  initial_records = 8 :=
by sorry

end NUMINAMATH_CALUDE_roberta_initial_records_l2950_295053


namespace NUMINAMATH_CALUDE_book_donations_mode_l2950_295091

/-- Represents the distribution of book donations -/
def book_donations : List (ℕ × ℕ) := [
  (30, 40), (22, 30), (16, 25), (8, 50), (6, 20), (4, 35)
]

/-- Calculates the mode of a list of pairs (value, frequency) -/
def mode (l : List (ℕ × ℕ)) : ℕ :=
  let max_frequency := l.map Prod.snd |>.maximum?
  match max_frequency with
  | none => 0
  | some max => (l.filter (fun p => p.2 = max)).map Prod.fst |>.minimum?
                |>.getD 0

/-- Theorem: The mode of the book donations is 8 -/
theorem book_donations_mode :
  mode book_donations = 8 := by
  sorry

end NUMINAMATH_CALUDE_book_donations_mode_l2950_295091


namespace NUMINAMATH_CALUDE_bobby_cars_after_seven_years_l2950_295081

def initial_cars : ℕ := 30

def double (n : ℕ) : ℕ := 2 * n

def donate (n : ℕ) : ℕ := n - (n / 10)

def update_cars (year : ℕ) (cars : ℕ) : ℕ :=
  if year % 2 = 0 ∧ year ≠ 0 then
    donate (double cars)
  else
    double cars

def cars_after_years (years : ℕ) : ℕ :=
  match years with
  | 0 => initial_cars
  | n + 1 => update_cars n (cars_after_years n)

theorem bobby_cars_after_seven_years :
  cars_after_years 7 = 2792 := by sorry

end NUMINAMATH_CALUDE_bobby_cars_after_seven_years_l2950_295081


namespace NUMINAMATH_CALUDE_joyce_suitable_land_l2950_295056

/-- Given Joyce's property information, calculate the suitable land for growing vegetables -/
theorem joyce_suitable_land (previous_property : ℝ) (new_property_multiplier : ℝ) (non_arable_land : ℝ) :
  previous_property = 2 ∧ 
  new_property_multiplier = 8 ∧ 
  non_arable_land = 6 →
  previous_property * new_property_multiplier - non_arable_land = 10 := by
  sorry

#check joyce_suitable_land

end NUMINAMATH_CALUDE_joyce_suitable_land_l2950_295056


namespace NUMINAMATH_CALUDE_darker_tile_fraction_is_three_fourths_l2950_295030

/-- Represents a floor with a repeating tile pattern -/
structure Floor :=
  (pattern_size : Nat)
  (corner_size : Nat)
  (dark_tiles_in_corner : Nat)

/-- The fraction of darker tiles in the floor -/
def darker_tile_fraction (f : Floor) : Rat :=
  let total_tiles := f.pattern_size * f.pattern_size
  let corner_tiles := f.corner_size * f.corner_size
  let num_corners := (f.pattern_size / f.corner_size) ^ 2
  let total_dark_tiles := f.dark_tiles_in_corner * num_corners
  total_dark_tiles / total_tiles

/-- Theorem stating that for a floor with a 4x4 repeating pattern and 3 darker tiles in each 2x2 corner,
    the fraction of darker tiles is 3/4 -/
theorem darker_tile_fraction_is_three_fourths (f : Floor)
  (h1 : f.pattern_size = 4)
  (h2 : f.corner_size = 2)
  (h3 : f.dark_tiles_in_corner = 3) :
  darker_tile_fraction f = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_darker_tile_fraction_is_three_fourths_l2950_295030


namespace NUMINAMATH_CALUDE_f_properties_l2950_295013

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) + (2 - a) * Real.exp x - a * x + a * Real.exp 1 / 2

theorem f_properties (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ∨
  (a > 0 ∧ ∃ x_min, ∀ x, f a x ≥ f a x_min) ∧
  (∀ x, f a x ≥ 0 ↔ a ∈ Set.Icc 0 (2 * Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2950_295013


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2950_295073

theorem cubic_equation_root (a b : ℚ) : 
  ((-2 - 3 * Real.sqrt 3) ^ 3 + a * (-2 - 3 * Real.sqrt 3) ^ 2 + b * (-2 - 3 * Real.sqrt 3) + 54 = 0) → 
  a = 38 / 23 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2950_295073


namespace NUMINAMATH_CALUDE_archie_marbles_l2950_295015

/-- Represents the number of marbles Archie has at different stages --/
structure MarbleCount where
  initial : ℕ
  after_street : ℕ
  after_sewer : ℕ
  after_bush : ℕ
  final : ℕ

/-- Represents the number of Glacier marbles Archie has at different stages --/
structure GlacierMarbleCount where
  initial : ℕ
  final : ℕ

/-- Calculates the number of marbles lost at each stage and the total Glacier marbles lost --/
def calculate_marbles (m : MarbleCount) (g : GlacierMarbleCount) : Prop :=
  m.after_street = m.initial * 40 / 100 ∧
  m.after_sewer = m.after_street / 2 ∧
  m.after_bush = m.after_sewer * 3 / 4 ∧
  m.final = m.after_bush + 5 ∧
  m.final = 15 ∧
  g.initial = m.initial * 30 / 100 ∧
  g.final = 4 ∧
  g.initial - g.final = 16

theorem archie_marbles :
  ∃ (m : MarbleCount) (g : GlacierMarbleCount),
    calculate_marbles m g ∧ m.initial = 67 := by sorry

#check archie_marbles

end NUMINAMATH_CALUDE_archie_marbles_l2950_295015


namespace NUMINAMATH_CALUDE_forty_students_not_enrolled_l2950_295026

/-- The number of students not enrolled in any language course -/
def students_not_enrolled (total students_french students_german students_spanish
  students_french_german students_french_spanish students_german_spanish
  students_all_three : ℕ) : ℕ :=
  total - (students_french + students_german + students_spanish
           - students_french_german - students_french_spanish - students_german_spanish
           + students_all_three)

/-- Theorem stating that 40 students are not enrolled in any language course -/
theorem forty_students_not_enrolled :
  students_not_enrolled 150 60 50 40 20 15 10 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_forty_students_not_enrolled_l2950_295026


namespace NUMINAMATH_CALUDE_dollar_equality_l2950_295052

-- Define the $ operation
def dollar (a b : ℝ) : ℝ := (a - b)^2

-- Theorem statement
theorem dollar_equality (x y : ℝ) : 
  dollar ((2*x + y)^2) ((x - 2*y)^2) = (3*x^2 + 8*x*y - 3*y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_dollar_equality_l2950_295052


namespace NUMINAMATH_CALUDE_complement_of_A_l2950_295027

def A : Set ℝ := {x | x ≥ 3} ∪ {x | x < -1}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | -1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2950_295027


namespace NUMINAMATH_CALUDE_course_selection_proof_l2950_295028

def total_courses : ℕ := 9
def courses_to_choose : ℕ := 4
def conflicting_courses : ℕ := 3
def other_courses : ℕ := total_courses - conflicting_courses

def selection_schemes : ℕ := 
  (conflicting_courses.choose 1 * other_courses.choose (courses_to_choose - 1)) +
  (other_courses.choose courses_to_choose)

theorem course_selection_proof : selection_schemes = 75 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_proof_l2950_295028


namespace NUMINAMATH_CALUDE_calculate_expression_l2950_295074

theorem calculate_expression : 5 * 423 + 4 * 423 + 3 * 423 + 421 = 5497 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2950_295074


namespace NUMINAMATH_CALUDE_inequality_proof_l2950_295055

theorem inequality_proof (x : ℝ) (h1 : 3/2 ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2950_295055


namespace NUMINAMATH_CALUDE_sum_and_interval_l2950_295043

theorem sum_and_interval : 
  let sum := 3 + 1/6 + 4 + 3/8 + 6 + 1/12
  sum = 13.625 ∧ 13.5 < sum ∧ sum < 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_interval_l2950_295043


namespace NUMINAMATH_CALUDE_tangent_line_condition_max_ab_value_l2950_295006

noncomputable section

/-- The function f(x) = ln(ax + b) + x^2 -/
def f (a b x : ℝ) : ℝ := Real.log (a * x + b) + x^2

/-- The derivative of f with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := a / (a * x + b) + 2 * x

theorem tangent_line_condition (a b : ℝ) (h1 : a ≠ 0) :
  (f_deriv a b 1 = 1 ∧ f a b 1 = 1) → (a = -1 ∧ b = 2) :=
sorry

theorem max_ab_value (a b : ℝ) (h1 : a ≠ 0) :
  (∀ x, f a b x ≤ x^2 + x) → (a * b ≤ Real.exp 1 / 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_condition_max_ab_value_l2950_295006


namespace NUMINAMATH_CALUDE_shifted_line_not_in_third_quadrant_l2950_295088

/-- The original line equation -/
def original_line (x : ℝ) : ℝ := -2 * x - 1

/-- The shifted line equation -/
def shifted_line (x : ℝ) : ℝ := -2 * x + 5

/-- The shift amount -/
def shift : ℝ := 3

/-- Theorem: The shifted line does not intersect the third quadrant -/
theorem shifted_line_not_in_third_quadrant :
  ∀ x y : ℝ, y = shifted_line x → ¬(x < 0 ∧ y < 0) :=
sorry

end NUMINAMATH_CALUDE_shifted_line_not_in_third_quadrant_l2950_295088


namespace NUMINAMATH_CALUDE_burger_distribution_theorem_l2950_295012

/-- Represents the burger distribution problem --/
def burger_distribution (total_burgers : ℕ) (num_friends : ℕ) (slices_per_burger : ℕ) 
  (slices_friend3 : ℕ) (slices_friend4 : ℕ) (slices_era : ℕ) : Prop :=
  let total_slices := total_burgers * slices_per_burger
  let slices_for_friends12 := total_slices - (slices_friend3 + slices_friend4 + slices_era)
  slices_for_friends12 = 3

/-- Theorem stating that under the given conditions, the first and second friends get 3 slices combined --/
theorem burger_distribution_theorem : 
  burger_distribution 5 4 2 3 3 1 := by sorry

end NUMINAMATH_CALUDE_burger_distribution_theorem_l2950_295012


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_l2950_295004

/-- The function f(x) = x^2 + x - 1 -/
def f (x : ℝ) : ℝ := x^2 + x - 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2*x + 1

theorem tangent_line_at_point_one :
  ∃ (m b : ℝ), 
    (f 1 = 1) ∧ 
    (f' 1 = m) ∧ 
    (∀ x y : ℝ, y = m * (x - 1) + 1 ↔ m * x - y + b = 0) ∧
    (3 * 1 - 1 + b = 0) ∧
    (∀ x y : ℝ, y = m * (x - 1) + 1 ↔ 3 * x - y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_l2950_295004


namespace NUMINAMATH_CALUDE_log_equation_solution_l2950_295046

theorem log_equation_solution :
  ∃ x : ℝ, (Real.log (3 * x) - 4 * Real.log 9 = 3) ∧ (x = 2187000) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2950_295046


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2950_295005

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y) = f x * f y - 2 * x * y) →
  (∀ x : ℝ, f x = 2 * x) ∨ (∀ x : ℝ, f x = -x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2950_295005


namespace NUMINAMATH_CALUDE_function_equality_l2950_295008

theorem function_equality :
  (∀ x : ℝ, x^2 = (x^6)^(1/3)) ∧
  (∀ x : ℝ, x = (x^3)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2950_295008


namespace NUMINAMATH_CALUDE_inequality_and_uniqueness_l2950_295079

theorem inequality_and_uniqueness 
  (a b c d : ℝ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (pos_c : 0 < c) 
  (pos_d : 0 < d) 
  (sum_eq : a + b = 4) 
  (prod_eq : c * d = 4) : 
  (a * b ≤ c + d) ∧ 
  (a * b = c + d → 
    ∀ (a' b' c' d' : ℝ), 
      0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 
      a' + b' = 4 ∧ c' * d' = 4 ∧ 
      a' * b' = c' + d' → 
      a' = a ∧ b' = b ∧ c' = c ∧ d' = d) := by
sorry

end NUMINAMATH_CALUDE_inequality_and_uniqueness_l2950_295079


namespace NUMINAMATH_CALUDE_specific_arithmetic_series_sum_l2950_295069

/-- The sum of an arithmetic series with given first term, last term, and common difference -/
def arithmetic_series_sum (a l d : ℤ) : ℤ :=
  let n : ℤ := (l - a) / d + 1
  (n * (a + l)) / 2

/-- Theorem stating that the sum of the specific arithmetic series is -576 -/
theorem specific_arithmetic_series_sum :
  arithmetic_series_sum (-47) (-1) 2 = -576 := by
  sorry

end NUMINAMATH_CALUDE_specific_arithmetic_series_sum_l2950_295069


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2950_295071

/-- Given two points P and Q that are symmetric with respect to the origin,
    prove that the sum of their x-coordinates plus the difference of their y-coordinates is zero. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (∃ (P Q : ℝ × ℝ), P = (m - 1, 5) ∧ Q = (3, 2 - n) ∧ P = (-Q.1, -Q.2)) →
  m + n = 5 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2950_295071


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l2950_295062

theorem correct_quotient_proof (N : ℕ) : 
  N % 21 = 0 →  -- remainder is 0 when divided by 21
  N / 12 = 56 → -- quotient is 56 when divided by 12
  N / 21 = 32   -- correct quotient when divided by 21
:= by sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l2950_295062


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l2950_295076

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_less_than_10 : a < 10
  b_less_than_10 : b < 10
  c_less_than_10 : c < 10
  d_less_than_10 : d < 10

/-- The conditions given in the problem -/
def satisfiesConditions (n : FourDigitNumber) : Prop :=
  n.a + n.b + n.c + n.d = 26 ∧
  (n.b * n.d) / 10 = n.a + n.c ∧
  ∃ k : Nat, n.b * n.d - n.c * n.c = 2 * k

/-- The theorem to prove -/
theorem unique_four_digit_number :
  ∃! n : FourDigitNumber, satisfiesConditions n ∧ 
    n.a = 1 ∧ n.b = 9 ∧ n.c = 7 ∧ n.d = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l2950_295076


namespace NUMINAMATH_CALUDE_committee_probability_l2950_295019

def totalStudents : ℕ := 18
def numBoys : ℕ := 8
def numGirls : ℕ := 10
def committeeSize : ℕ := 4

theorem committee_probability : 
  let totalCommittees := Nat.choose totalStudents committeeSize
  let allBoysCommittees := Nat.choose numBoys committeeSize
  let allGirlsCommittees := Nat.choose numGirls committeeSize
  let probabilityAtLeastOneBoyOneGirl := 1 - (allBoysCommittees + allGirlsCommittees : ℚ) / totalCommittees
  probabilityAtLeastOneBoyOneGirl = 139 / 153 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l2950_295019


namespace NUMINAMATH_CALUDE_robbery_participants_l2950_295029

-- Define the suspects
variable (A B V G : Prop)

-- A: Alexey is guilty
-- B: Boris is guilty
-- V: Veniamin is guilty
-- G: Grigory is guilty

-- Define the conditions
variable (h1 : ¬G → (B ∧ ¬A))
variable (h2 : V → (¬A ∧ ¬B))
variable (h3 : G → B)
variable (h4 : B → (A ∨ V))

-- Theorem statement
theorem robbery_participants : A ∧ B ∧ G ∧ ¬V := by
  sorry

end NUMINAMATH_CALUDE_robbery_participants_l2950_295029


namespace NUMINAMATH_CALUDE_apples_picked_total_l2950_295077

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- The total number of apples picked -/
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_picked_total :
  total_apples = 11 :=
by sorry

end NUMINAMATH_CALUDE_apples_picked_total_l2950_295077


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2950_295010

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2 - x) ↔ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2950_295010


namespace NUMINAMATH_CALUDE_sum_of_divisible_by_four_l2950_295057

theorem sum_of_divisible_by_four : 
  (Finset.filter (fun n => n > 10 ∧ n < 30 ∧ n % 4 = 0) (Finset.range 30)).sum id = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisible_by_four_l2950_295057


namespace NUMINAMATH_CALUDE_sam_distance_l2950_295007

/-- Given that Harvey runs 8 miles more than Sam and their total distance is 32 miles,
    prove that Sam runs 12 miles. -/
theorem sam_distance (sam : ℝ) (harvey : ℝ) : 
  harvey = sam + 8 → sam + harvey = 32 → sam = 12 := by
  sorry

end NUMINAMATH_CALUDE_sam_distance_l2950_295007


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2950_295090

theorem polynomial_division_remainder :
  ∃ (q r : Polynomial ℤ),
    (X^5 - X^4 + X^3 - X + 1 : Polynomial ℤ) = (X^3 - X + 1) * q + r ∧
    r = -X^2 + 4*X - 1 ∧
    r.degree < (X^3 - X + 1).degree :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2950_295090


namespace NUMINAMATH_CALUDE_game_result_l2950_295066

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 7
  else if n % 2 = 0 then 3
  else if Nat.Prime n then 5
  else 0

def allie_rolls : List ℕ := [2, 3, 4, 5, 6]
def betty_rolls : List ℕ := [6, 3, 4, 2, 1]

theorem game_result :
  (List.sum (List.map f allie_rolls)) * (List.sum (List.map f betty_rolls)) = 500 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l2950_295066


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_two_l2950_295025

theorem smallest_integer_with_remainder_two : ∃ n : ℕ, 
  (n > 20) ∧ 
  (∀ m : ℕ, m > 20 → 
    ((m % 3 = 2) ∧ (m % 4 = 2) ∧ (m % 5 = 2) ∧ (m % 6 = 2)) → 
    (n ≤ m)) ∧
  (n % 3 = 2) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) :=
by
  -- The proof goes here
  sorry

#eval Nat.lcm (Nat.lcm 3 4) (Nat.lcm 5 6)  -- This should output 60

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_two_l2950_295025


namespace NUMINAMATH_CALUDE_steel_tin_mass_ratio_l2950_295039

theorem steel_tin_mass_ratio : 
  ∀ (steel_mass tin_mass copper_mass : ℝ),
  steel_mass > 0 ∧ tin_mass > 0 ∧ copper_mass > 0 →
  steel_mass = copper_mass + 20 →
  copper_mass = 90 →
  20 * steel_mass + 20 * copper_mass + 20 * tin_mass = 5100 →
  steel_mass / tin_mass = 2 := by
sorry

end NUMINAMATH_CALUDE_steel_tin_mass_ratio_l2950_295039


namespace NUMINAMATH_CALUDE_eat_porridge_together_l2950_295032

/-- Masha's time to eat one bowl of porridge in minutes -/
def mashaTime : ℝ := 12

/-- The Bear's eating speed relative to Masha's -/
def bearRelativeSpeed : ℝ := 2

/-- Number of bowls to eat together -/
def totalBowls : ℝ := 6

/-- Time for Masha and the Bear to eat all bowls together -/
def totalTime : ℝ := 24

theorem eat_porridge_together :
  (totalBowls * mashaTime) / (1 + bearRelativeSpeed) = totalTime := by
  sorry

end NUMINAMATH_CALUDE_eat_porridge_together_l2950_295032


namespace NUMINAMATH_CALUDE_power_function_through_point_and_value_l2950_295061

/-- A power function that passes through the point (2,8) -/
def f : ℝ → ℝ := fun x ↦ x^3

theorem power_function_through_point_and_value :
  f 2 = 8 ∧ f 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_and_value_l2950_295061


namespace NUMINAMATH_CALUDE_monotone_iff_bound_l2950_295093

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

/-- The derivative of f with respect to x -/
def f' (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 4*x + m

/-- f is monotonically increasing -/
def is_monotone_increasing (m : ℝ) : Prop :=
  ∀ x : ℝ, f' m x ≥ 0

theorem monotone_iff_bound (m : ℝ) :
  is_monotone_increasing m ↔ m ≥ 4/3 :=
sorry

end NUMINAMATH_CALUDE_monotone_iff_bound_l2950_295093


namespace NUMINAMATH_CALUDE_smallest_multiple_congruence_l2950_295040

theorem smallest_multiple_congruence : 
  ∃ (n : ℕ), n = 494 ∧ 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(19 ∣ m ∧ m % 97 = 3)) ∧
  19 ∣ n ∧ n % 97 = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_congruence_l2950_295040


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_base_6_number_l2950_295080

/-- Represents the number 100111001 in base 6 -/
def base_6_number : ℕ := 6^8 + 6^5 + 6^4 + 6^3 + 6 + 1

/-- The largest prime divisor of base_6_number -/
def largest_prime_divisor : ℕ := 43

theorem largest_prime_divisor_of_base_6_number :
  (∀ p : ℕ, Prime p → p ∣ base_6_number → p ≤ largest_prime_divisor) ∧
  (Prime largest_prime_divisor ∧ largest_prime_divisor ∣ base_6_number) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_base_6_number_l2950_295080


namespace NUMINAMATH_CALUDE_complementary_angle_of_60_13_25_l2950_295014

/-- Represents an angle in degrees, minutes, and seconds -/
structure DMS where
  degrees : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the complementary angle of a given angle in DMS format -/
def complementaryAngle (angle : DMS) : DMS :=
  sorry

/-- Theorem stating that the complementary angle of 60°13'25" is 29°46'35" -/
theorem complementary_angle_of_60_13_25 :
  let givenAngle : DMS := ⟨60, 13, 25⟩
  complementaryAngle givenAngle = ⟨29, 46, 35⟩ := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_of_60_13_25_l2950_295014


namespace NUMINAMATH_CALUDE_optimalPlan_is_most_cost_effective_l2950_295038

/-- Represents a vehicle type with its capacity and cost -/
structure VehicleType where
  peopleCapacity : ℕ
  luggageCapacity : ℕ
  cost : ℕ

/-- Represents a rental plan -/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

def totalStudents : ℕ := 290
def totalLuggage : ℕ := 100
def totalVehicles : ℕ := 8

def typeA : VehicleType := ⟨40, 10, 2000⟩
def typeB : VehicleType := ⟨30, 20, 1800⟩

def isValidPlan (plan : RentalPlan) : Prop :=
  plan.typeA + plan.typeB = totalVehicles ∧
  plan.typeA * typeA.peopleCapacity + plan.typeB * typeB.peopleCapacity ≥ totalStudents ∧
  plan.typeA * typeA.luggageCapacity + plan.typeB * typeB.luggageCapacity ≥ totalLuggage

def planCost (plan : RentalPlan) : ℕ :=
  plan.typeA * typeA.cost + plan.typeB * typeB.cost

def optimalPlan : RentalPlan := ⟨5, 3⟩

theorem optimalPlan_is_most_cost_effective :
  isValidPlan optimalPlan ∧
  ∀ plan, isValidPlan plan → planCost optimalPlan ≤ planCost plan :=
sorry

end NUMINAMATH_CALUDE_optimalPlan_is_most_cost_effective_l2950_295038


namespace NUMINAMATH_CALUDE_ceiling_sqrt_244_l2950_295041

theorem ceiling_sqrt_244 : ⌈Real.sqrt 244⌉ = 16 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_244_l2950_295041


namespace NUMINAMATH_CALUDE_D_72_equals_38_l2950_295072

/-- D(n) represents the number of ways to factor n into integers greater than 1, counting permutations -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem: D(72) equals 38 -/
theorem D_72_equals_38 : D 72 = 38 := by sorry

end NUMINAMATH_CALUDE_D_72_equals_38_l2950_295072


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2950_295096

def data : List ℝ := [2, 4, 6, 8]

def median (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem min_value_reciprocal_sum 
  (m : ℝ) 
  (n : ℝ) 
  (hm : m = median data) 
  (hn : n = variance data) 
  (a : ℝ) 
  (b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (heq : m * a + n * b = 1) : 
  (1 / a + 1 / b) ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2950_295096


namespace NUMINAMATH_CALUDE_distributive_property_l2950_295009

theorem distributive_property (a b c : ℝ) : -2 * (a + b - 3 * c) = -2 * a - 2 * b + 6 * c := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_l2950_295009


namespace NUMINAMATH_CALUDE_dave_white_tshirt_packs_l2950_295023

/-- The number of T-shirts in a pack of white T-shirts -/
def white_pack_size : ℕ := 6

/-- The number of T-shirts in a pack of blue T-shirts -/
def blue_pack_size : ℕ := 4

/-- The number of packs of blue T-shirts Dave bought -/
def blue_packs : ℕ := 2

/-- The total number of T-shirts Dave bought -/
def total_tshirts : ℕ := 26

/-- The number of packs of white T-shirts Dave bought -/
def white_packs : ℕ := 3

theorem dave_white_tshirt_packs :
  white_packs * white_pack_size + blue_packs * blue_pack_size = total_tshirts :=
by sorry

end NUMINAMATH_CALUDE_dave_white_tshirt_packs_l2950_295023


namespace NUMINAMATH_CALUDE_number_equation_proof_l2950_295059

theorem number_equation_proof : ∃ x : ℚ, x + (2/3) * x + 1 = 10 ∧ x = 27/5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_proof_l2950_295059


namespace NUMINAMATH_CALUDE_min_value_a_l2950_295086

theorem min_value_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, |x - a| + |1 - x| ≥ 1) : 
  (∀ b : ℝ, b > 0 ∧ (∀ x : ℝ, |x - b| + |1 - x| ≥ 1) → b ≥ 2) ∧ 
  (∃ c : ℝ, c > 0 ∧ (∀ x : ℝ, |x - c| + |1 - x| ≥ 1) ∧ c = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l2950_295086


namespace NUMINAMATH_CALUDE_limit_special_function_l2950_295058

/-- The limit of ((1 + sin(x) * cos(2x)) / (1 + sin(x) * cos(3x)))^(1 / sin(x)^3) as x approaches 0 is e^(-5/2) -/
theorem limit_special_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → 
    |((1 + Real.sin x * Real.cos (2*x)) / (1 + Real.sin x * Real.cos (3*x)))^(1 / Real.sin x^3) - Real.exp (-5/2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_special_function_l2950_295058


namespace NUMINAMATH_CALUDE_circles_symmetric_line_l2950_295065

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y + 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y + 7 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Theorem statement
theorem circles_symmetric_line :
  ∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → line_l x y :=
by sorry

end NUMINAMATH_CALUDE_circles_symmetric_line_l2950_295065


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2950_295099

theorem geometric_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, a k > 0) →  -- positive terms
  (∀ k, a (k + 1) / a k = a (k + 2) / a (k + 1)) →  -- geometric sequence
  (a 3 = 4) →
  (a 4 * a 5 * a 6 = 2^12) →
  (S n = 2^10 - 1) →
  (S n = (a 1 * (1 - (a 2 / a 1)^n)) / (1 - (a 2 / a 1))) →  -- sum formula for geometric sequence
  (a 1 = 1 ∧ a 2 / a 1 = 2 ∧ n = 10) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2950_295099


namespace NUMINAMATH_CALUDE_marias_sister_bottles_l2950_295003

/-- Given the initial number of water bottles, the number Maria drank, and the number left,
    calculate the number of bottles Maria's sister drank. -/
theorem marias_sister_bottles (initial : ℝ) (maria_drank : ℝ) (left : ℝ) :
  initial = 45.0 →
  maria_drank = 14.0 →
  left = 23.0 →
  initial - maria_drank - left = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_marias_sister_bottles_l2950_295003


namespace NUMINAMATH_CALUDE_social_media_time_ratio_l2950_295042

/-- Proves that the ratio of daily time spent on social media to total daily time spent on phone is 1:2 -/
theorem social_media_time_ratio 
  (daily_phone_time : ℝ) 
  (weekly_social_media_time : ℝ) 
  (h1 : daily_phone_time = 6)
  (h2 : weekly_social_media_time = 21) :
  (weekly_social_media_time / 7) / daily_phone_time = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_social_media_time_ratio_l2950_295042


namespace NUMINAMATH_CALUDE_union_equality_implies_m_values_l2950_295094

def A : Set ℝ := {-1, 2}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem union_equality_implies_m_values (m : ℝ) :
  A ∪ B m = A → m = 0 ∨ m = 1 ∨ m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_m_values_l2950_295094


namespace NUMINAMATH_CALUDE_car_distance_l2950_295033

/-- Given a car that travels 180 miles in 4 hours, prove that it will travel 135 miles in the next 3 hours at the same rate. -/
theorem car_distance (initial_distance : ℝ) (initial_time : ℝ) (next_time : ℝ) :
  initial_distance = 180 ∧ initial_time = 4 ∧ next_time = 3 →
  (initial_distance / initial_time) * next_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l2950_295033


namespace NUMINAMATH_CALUDE_computer_game_cost_l2950_295051

/-- The cost of a computer game, given the total cost of movie tickets and the total spent on entertainment. -/
theorem computer_game_cost (movie_tickets_cost total_spent : ℕ) : 
  movie_tickets_cost = 36 → total_spent = 102 → total_spent - movie_tickets_cost = 66 := by
  sorry

end NUMINAMATH_CALUDE_computer_game_cost_l2950_295051


namespace NUMINAMATH_CALUDE_extremum_condition_l2950_295054

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, (fun x => Real.exp x + a * x) x > 0 ∧
   ∀ y : ℝ, (fun x => Real.exp x + a * x) y ≤ (fun x => Real.exp x + a * x) x) →
  a < -1 := by
  sorry

end NUMINAMATH_CALUDE_extremum_condition_l2950_295054


namespace NUMINAMATH_CALUDE_remainder_theorem_l2950_295060

theorem remainder_theorem (n : ℤ) (h : ∃ (a : ℤ), n = 100 * a - 1) : 
  (n^2 + 2*n + 3) % 100 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2950_295060
