import Mathlib

namespace NUMINAMATH_CALUDE_ice_pop_probability_l2199_219931

/-- Represents the number of ice pops of each flavor --/
structure IcePops where
  cherry : ℕ
  orange : ℕ
  lemonLime : ℕ

/-- Calculates the probability of selecting two ice pops of different flavors --/
def probDifferentFlavors (pops : IcePops) : ℚ :=
  let total := pops.cherry + pops.orange + pops.lemonLime
  1 - (pops.cherry * (pops.cherry - 1) + pops.orange * (pops.orange - 1) + pops.lemonLime * (pops.lemonLime - 1)) / (total * (total - 1))

/-- The main theorem stating that for the given ice pop distribution, 
    the probability of selecting two different flavors is 8/11 --/
theorem ice_pop_probability : 
  let pops : IcePops := ⟨4, 3, 4⟩
  probDifferentFlavors pops = 8/11 := by
  sorry

end NUMINAMATH_CALUDE_ice_pop_probability_l2199_219931


namespace NUMINAMATH_CALUDE_square_perimeter_l2199_219953

/-- Given a rectangle with length 50 cm and width 10 cm, and a square with an area
    five times that of the rectangle, prove that the perimeter of the square is 200 cm. -/
theorem square_perimeter (rectangle_length : ℝ) (rectangle_width : ℝ) (square_area : ℝ) :
  rectangle_length = 50 ∧ 
  rectangle_width = 10 ∧ 
  square_area = 5 * (rectangle_length * rectangle_width) →
  4 * Real.sqrt square_area = 200 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l2199_219953


namespace NUMINAMATH_CALUDE_age_difference_l2199_219939

theorem age_difference (a b c d : ℕ) : 
  (a + b = b + c + 13) →
  (b + d = c + d + 7) →
  (a + d = 2 * c - 12) →
  (a = c + 13) :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2199_219939


namespace NUMINAMATH_CALUDE_cows_count_l2199_219965

/-- Represents the number of animals in the farm -/
structure FarmAnimals where
  ducks : ℕ
  cows : ℕ
  spiders : ℕ

/-- Checks if the given farm animals satisfy all the conditions -/
def satisfiesConditions (animals : FarmAnimals) : Prop :=
  let totalLegs := 2 * animals.ducks + 4 * animals.cows + 8 * animals.spiders
  let totalHeads := animals.ducks + animals.cows + animals.spiders
  totalLegs = 2 * totalHeads + 72 ∧
  animals.spiders = 2 * animals.ducks ∧
  totalHeads ≤ 40

/-- Theorem stating that the number of cows is 30 given the conditions -/
theorem cows_count (animals : FarmAnimals) :
  satisfiesConditions animals → animals.cows = 30 := by
  sorry

end NUMINAMATH_CALUDE_cows_count_l2199_219965


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2199_219980

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2199_219980


namespace NUMINAMATH_CALUDE_total_decorations_count_l2199_219916

/-- The number of decorations in each box -/
def decorations_per_box : ℕ := 4 + 1 + 5

/-- The number of families receiving a box -/
def number_of_families : ℕ := 11

/-- The number of boxes given to the community center -/
def community_center_boxes : ℕ := 1

/-- The total number of decorations handed out -/
def total_decorations : ℕ := decorations_per_box * (number_of_families + community_center_boxes)

theorem total_decorations_count : total_decorations = 120 := by
  sorry


end NUMINAMATH_CALUDE_total_decorations_count_l2199_219916


namespace NUMINAMATH_CALUDE_factorization_equalities_l2199_219955

theorem factorization_equalities (x y a : ℝ) : 
  (x^4 - 9*x^2 = x^2*(x+3)*(x-3)) ∧ 
  (25*x^2*y + 20*x*y^2 + 4*y^3 = y*(5*x+2*y)^2) ∧ 
  (x^2*(a-1) + y^2*(1-a) = (a-1)*(x+y)*(x-y)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equalities_l2199_219955


namespace NUMINAMATH_CALUDE_rulers_in_drawer_l2199_219995

/-- Given an initial number of rulers and an additional number of rulers,
    calculate the total number of rulers in the drawer. -/
def total_rulers (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that with 46 rulers initially and 25 rulers added,
    the total number of rulers in the drawer is 71. -/
theorem rulers_in_drawer : total_rulers 46 25 = 71 := by
  sorry

end NUMINAMATH_CALUDE_rulers_in_drawer_l2199_219995


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2199_219973

theorem min_value_reciprocal_sum (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2199_219973


namespace NUMINAMATH_CALUDE_binary_101101110_equals_octal_556_l2199_219982

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_natural (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to its octal representation as a list of digits. -/
def natural_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: natural_to_octal (n / 8)

theorem binary_101101110_equals_octal_556 :
  let binary : List Bool := [true, false, true, true, false, true, true, true, false]
  let octal : List ℕ := [6, 5, 5]
  binary_to_natural binary = (natural_to_octal (binary_to_natural binary)).reverse.foldl (fun acc d => acc * 8 + d) 0 ∧
  natural_to_octal (binary_to_natural binary) = octal.reverse :=
by sorry

end NUMINAMATH_CALUDE_binary_101101110_equals_octal_556_l2199_219982


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_six_l2199_219986

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem angle_A_is_pi_over_six (t : Triangle) :
  (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A = Real.sqrt 3 * t.a * Real.cos t.C →
  t.A = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_six_l2199_219986


namespace NUMINAMATH_CALUDE_robot_staircase_l2199_219979

theorem robot_staircase (a b : ℕ+) : 
  ∃ n : ℕ, n = a + b - Nat.gcd a b ∧ 
  (∀ m : ℕ, m < n → ¬∃ (k l : ℕ), k * a = m + l * b) ∧
  (∃ (k l : ℕ), k * a = n + l * b) := by
  sorry

end NUMINAMATH_CALUDE_robot_staircase_l2199_219979


namespace NUMINAMATH_CALUDE_parabola_coeff_sum_l2199_219943

/-- A parabola with equation y = ax^2 + bx + c, vertex at (-3, 2), and passing through (1, 6) -/
def Parabola (a b c : ℝ) : Prop :=
  (∀ x y : ℝ, y = a * x^2 + b * x + c) ∧
  (2 = a * (-3)^2 + b * (-3) + c) ∧
  (6 = a * 1^2 + b * 1 + c)

/-- The sum of coefficients a, b, and c equals 6 -/
theorem parabola_coeff_sum (a b c : ℝ) (h : Parabola a b c) : a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coeff_sum_l2199_219943


namespace NUMINAMATH_CALUDE_b_over_a_is_real_l2199_219996

variable (a b x y : ℂ)

theorem b_over_a_is_real
  (h1 : a * b ≠ 0)
  (h2 : Complex.abs x = Complex.abs y)
  (h3 : x + y = a)
  (h4 : x * y = b) :
  ∃ (r : ℝ), b / a = r :=
sorry

end NUMINAMATH_CALUDE_b_over_a_is_real_l2199_219996


namespace NUMINAMATH_CALUDE_residue_625_mod_17_l2199_219994

theorem residue_625_mod_17 : 625 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_residue_625_mod_17_l2199_219994


namespace NUMINAMATH_CALUDE_rent_utilities_percentage_l2199_219925

-- Define the previous monthly income
def previous_income : ℝ := 1000

-- Define the salary increase
def salary_increase : ℝ := 600

-- Define the percentage spent on rent and utilities after the increase
def new_percentage : ℝ := 0.25

-- Define the function to calculate the amount spent on rent and utilities
def rent_utilities (income : ℝ) (percentage : ℝ) : ℝ := income * percentage

-- Theorem statement
theorem rent_utilities_percentage :
  ∃ (old_percentage : ℝ),
    rent_utilities previous_income old_percentage = 
    rent_utilities (previous_income + salary_increase) new_percentage ∧
    old_percentage = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_rent_utilities_percentage_l2199_219925


namespace NUMINAMATH_CALUDE_max_self_intersections_correct_max_self_intersections_13_max_self_intersections_1950_l2199_219921

/-- The maximum number of self-intersection points in a closed polygonal line with n segments -/
def max_self_intersections (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    n * (n - 3) / 2
  else
    n * (n - 4) / 2 + 1

theorem max_self_intersections_correct (n : ℕ) (h : n ≥ 3) :
  max_self_intersections n =
    if n % 2 = 1 then
      n * (n - 3) / 2
    else
      n * (n - 4) / 2 + 1 :=
by sorry

-- Specific cases
theorem max_self_intersections_13 :
  max_self_intersections 13 = 65 :=
by sorry

theorem max_self_intersections_1950 :
  max_self_intersections 1950 = 1897851 :=
by sorry

end NUMINAMATH_CALUDE_max_self_intersections_correct_max_self_intersections_13_max_self_intersections_1950_l2199_219921


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_over_f_at_zero_l2199_219932

noncomputable def f (x : ℝ) : ℝ := (x + Real.sqrt (1 + x^2))^10

theorem f_derivative_at_zero_over_f_at_zero : 
  (deriv f 0) / (f 0) = 10 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_over_f_at_zero_l2199_219932


namespace NUMINAMATH_CALUDE_probability_twelve_no_consecutive_ones_l2199_219992

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def validSequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

/-- Probability of a valid sequence of length n -/
def probability (n : ℕ) : ℚ :=
  (validSequences n : ℚ) / (totalSequences n : ℚ)

theorem probability_twelve_no_consecutive_ones :
  probability 12 = 377 / 4096 :=
sorry

end NUMINAMATH_CALUDE_probability_twelve_no_consecutive_ones_l2199_219992


namespace NUMINAMATH_CALUDE_hotel_halls_first_wing_l2199_219952

/-- Represents the number of halls on each floor of the first wing -/
def halls_first_wing : ℕ := sorry

/-- Represents the total number of rooms in the hotel -/
def total_rooms : ℕ := 4248

/-- Represents the number of floors in the first wing -/
def floors_first_wing : ℕ := 9

/-- Represents the number of rooms in each hall of the first wing -/
def rooms_per_hall_first_wing : ℕ := 32

/-- Represents the number of floors in the second wing -/
def floors_second_wing : ℕ := 7

/-- Represents the number of halls on each floor of the second wing -/
def halls_second_wing : ℕ := 9

/-- Represents the number of rooms in each hall of the second wing -/
def rooms_per_hall_second_wing : ℕ := 40

theorem hotel_halls_first_wing : 
  halls_first_wing * floors_first_wing * rooms_per_hall_first_wing + 
  floors_second_wing * halls_second_wing * rooms_per_hall_second_wing = total_rooms ∧ 
  halls_first_wing = 6 := by sorry

end NUMINAMATH_CALUDE_hotel_halls_first_wing_l2199_219952


namespace NUMINAMATH_CALUDE_common_sum_in_square_matrix_l2199_219900

theorem common_sum_in_square_matrix : 
  let n : ℕ := 36
  let a : ℤ := -15
  let l : ℤ := 20
  let total_sum : ℤ := n * (a + l) / 2
  let matrix_size : ℕ := 6
  total_sum / matrix_size = 15 := by sorry

end NUMINAMATH_CALUDE_common_sum_in_square_matrix_l2199_219900


namespace NUMINAMATH_CALUDE_combination_problem_l2199_219991

/-- Given that 1/C_5^m - 1/C_6^m = 7/(10C_7^m), prove that C_{21}^m = 210 -/
theorem combination_problem (m : ℕ) : 
  (1 / (Nat.choose 5 m : ℚ) - 1 / (Nat.choose 6 m : ℚ) = 7 / (10 * (Nat.choose 7 m : ℚ))) → 
  Nat.choose 21 m = 210 := by
sorry

end NUMINAMATH_CALUDE_combination_problem_l2199_219991


namespace NUMINAMATH_CALUDE_trash_can_ratio_l2199_219964

/-- Represents the number of trash cans added to the streets -/
def street_cans : ℕ := 14

/-- Represents the total number of trash cans -/
def total_cans : ℕ := 42

/-- Represents the number of trash cans added to the back of stores -/
def store_cans : ℕ := total_cans - street_cans

/-- The ratio of trash cans added to the back of stores to trash cans added to the streets -/
theorem trash_can_ratio : (store_cans : ℚ) / street_cans = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_trash_can_ratio_l2199_219964


namespace NUMINAMATH_CALUDE_product_of_roots_l2199_219968

theorem product_of_roots (x : ℝ) : 
  (∃ y : ℝ, (x^2 + 90*x + 2027) / 3 = Real.sqrt (x^2 + 90*x + 2055)) → 
  (∃ x₁ x₂ : ℝ, x₁ * x₂ = 2006 ∧ 
    ((x₁^2 + 90*x₁ + 2027) / 3 = Real.sqrt (x₁^2 + 90*x₁ + 2055)) ∧
    ((x₂^2 + 90*x₂ + 2027) / 3 = Real.sqrt (x₂^2 + 90*x₂ + 2055))) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2199_219968


namespace NUMINAMATH_CALUDE_ship_speed_theorem_l2199_219967

/-- Represents the speed of a ship in different conditions -/
structure ShipSpeed where
  downstream : ℝ  -- Speed downstream
  waterFlow : ℝ   -- Speed of water flow
  upstream : ℝ    -- Speed upstream

/-- 
Theorem: Given a ship traveling downstream at 26 km/h and a water flow speed of v km/h, 
the speed of the ship traveling upstream is 26 - 2v km/h.
-/
theorem ship_speed_theorem (v : ℝ) : 
  let s : ShipSpeed := { downstream := 26, waterFlow := v, upstream := 26 - 2*v }
  s.upstream = s.downstream - 2 * s.waterFlow := by
  sorry

end NUMINAMATH_CALUDE_ship_speed_theorem_l2199_219967


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achievable_l2199_219935

theorem min_value_sum_reciprocals (x y z : ℕ+) (h : x + y + z = 12) :
  (x + y + z : ℚ) * (1 / (x + y : ℚ) + 1 / (x + z : ℚ) + 1 / (y + z : ℚ)) ≥ 9 / 2 :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℕ+), x + y + z = 12 ∧
    (x + y + z : ℚ) * (1 / (x + y : ℚ) + 1 / (x + z : ℚ) + 1 / (y + z : ℚ)) = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achievable_l2199_219935


namespace NUMINAMATH_CALUDE_closest_to_product_l2199_219905

def options : List ℝ := [7, 42, 74, 84, 737]

def product : ℝ := 1.8 * (40.3 + 0.07)

theorem closest_to_product : 
  ∃ (x : ℝ), x ∈ options ∧ 
  ∀ (y : ℝ), y ∈ options → |x - product| ≤ |y - product| ∧ 
  x = 74 :=
by sorry

end NUMINAMATH_CALUDE_closest_to_product_l2199_219905


namespace NUMINAMATH_CALUDE_greatest_power_of_three_in_40_factorial_l2199_219919

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_multiples (n k : ℕ) : ℕ := n / k

theorem greatest_power_of_three_in_40_factorial :
  (∀ m : ℕ, m ≤ 18 → is_factor (3^m) (factorial 40)) ∧
  ¬(is_factor (3^19) (factorial 40)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_in_40_factorial_l2199_219919


namespace NUMINAMATH_CALUDE_power_property_iff_square_property_l2199_219957

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, a * b ≠ 0 → f (a * b) ≥ f a + f b

/-- The property that f(a^n) = n * f(a) for all non-zero a and natural n -/
def SatisfiesPowerProperty (f : ℤ → ℤ) : Prop :=
  ∀ a : ℤ, a ≠ 0 → ∀ n : ℕ, f (a ^ n) = n * f a

/-- The property that f(a^2) = 2 * f(a) for all non-zero a -/
def SatisfiesSquareProperty (f : ℤ → ℤ) : Prop :=
  ∀ a : ℤ, a ≠ 0 → f (a ^ 2) = 2 * f a

theorem power_property_iff_square_property (f : ℤ → ℤ) (h : SatisfiesInequality f) :
  SatisfiesPowerProperty f ↔ SatisfiesSquareProperty f :=
sorry

end NUMINAMATH_CALUDE_power_property_iff_square_property_l2199_219957


namespace NUMINAMATH_CALUDE_perimeter_of_specific_pentagon_l2199_219938

/-- An irregular pentagon with given side lengths -/
structure IrregularPentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ

/-- The perimeter of an irregular pentagon -/
def perimeter (p : IrregularPentagon) : ℝ :=
  p.side1 + p.side2 + p.side3 + p.side4 + p.side5

/-- Theorem: The perimeter of the given irregular pentagon is 40 -/
theorem perimeter_of_specific_pentagon :
  let p : IrregularPentagon := {
    side1 := 6,
    side2 := 7,
    side3 := 8,
    side4 := 9,
    side5 := 10
  }
  perimeter p = 40 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_specific_pentagon_l2199_219938


namespace NUMINAMATH_CALUDE_local_minimum_at_zero_l2199_219929

/-- The function f(x) = (x^2 - 1)^3 + 1 has a local minimum at x = 0 -/
theorem local_minimum_at_zero (f : ℝ → ℝ) (h : f = λ x => (x^2 - 1)^3 + 1) :
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f 0 ≤ f x :=
sorry

end NUMINAMATH_CALUDE_local_minimum_at_zero_l2199_219929


namespace NUMINAMATH_CALUDE_total_hours_played_l2199_219944

def football_minutes : ℕ := 60
def basketball_minutes : ℕ := 30

def total_minutes : ℕ := football_minutes + basketball_minutes

def minutes_per_hour : ℕ := 60

theorem total_hours_played :
  (total_minutes : ℚ) / minutes_per_hour = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_played_l2199_219944


namespace NUMINAMATH_CALUDE_distinct_collections_is_125_l2199_219907

/-- Represents the word "COMPUTATIONS" -/
def word : String := "COMPUTATIONS"

/-- The number of vowels in the word -/
def num_vowels : Nat := 5

/-- The number of consonants in the word, excluding T's -/
def num_consonants_without_t : Nat := 5

/-- The number of T's in the word -/
def num_t : Nat := 2

/-- The number of vowels to select -/
def vowels_to_select : Nat := 4

/-- The number of consonants to select -/
def consonants_to_select : Nat := 4

/-- Calculates the number of distinct collections of letters -/
def distinct_collections : Nat :=
  (Nat.choose num_vowels vowels_to_select) * 
  ((Nat.choose num_consonants_without_t consonants_to_select) + 
   (Nat.choose num_consonants_without_t (consonants_to_select - 1)) +
   (Nat.choose num_consonants_without_t (consonants_to_select - 2)))

/-- Theorem stating that the number of distinct collections is 125 -/
theorem distinct_collections_is_125 : distinct_collections = 125 := by
  sorry

end NUMINAMATH_CALUDE_distinct_collections_is_125_l2199_219907


namespace NUMINAMATH_CALUDE_total_crayons_l2199_219926

theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) : 
  crayons_per_child = 5 → num_children = 10 → crayons_per_child * num_children = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l2199_219926


namespace NUMINAMATH_CALUDE_train_length_calculation_l2199_219937

/-- The length of a train given the speeds of two trains, the length of one train, and the time they take to clear each other. -/
theorem train_length_calculation (v1 v2 : ℝ) (l2 t : ℝ) (h1 : v1 = 80) (h2 : v2 = 65) (h3 : l2 = 165) (h4 : t = 7.100121645440779) : 
  ∃ l1 : ℝ, abs (l1 - 121.197) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2199_219937


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l2199_219999

/-- The focal length of an ellipse with equation 2x^2 + 3y^2 = 1 is √6/3 -/
theorem ellipse_focal_length : 
  let a : ℝ := 1 / Real.sqrt 2
  let b : ℝ := 1 / Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  c = Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l2199_219999


namespace NUMINAMATH_CALUDE_inequality_proof_l2199_219941

theorem inequality_proof (m n : ℝ) (h1 : m > n) (h2 : n > 0) : 
  m * Real.exp n + n < n * Real.exp m + m := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2199_219941


namespace NUMINAMATH_CALUDE_alexey_min_banks_l2199_219917

/-- The minimum number of banks needed to fully insure a given amount of money -/
def min_banks (total_amount : ℕ) (max_payout : ℕ) : ℕ :=
  (total_amount + max_payout - 1) / max_payout

/-- Theorem stating the minimum number of banks needed for Alexey's case -/
theorem alexey_min_banks :
  min_banks 10000000 1400000 = 8 := by
  sorry

end NUMINAMATH_CALUDE_alexey_min_banks_l2199_219917


namespace NUMINAMATH_CALUDE_identity_function_satisfies_conditions_l2199_219972

theorem identity_function_satisfies_conditions (f : ℕ → ℕ) 
  (h1 : ∀ n, f (2 * n) = 2 * f n) 
  (h2 : ∀ n, f (2 * n + 1) = 2 * f n + 1) : 
  ∀ n, f n = n := by sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_conditions_l2199_219972


namespace NUMINAMATH_CALUDE_lineup_probability_probability_no_more_than_five_girls_between_boys_l2199_219945

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem lineup_probability :
  (C 14 9 + 6 * C 13 8) / C 20 9 =
  (↑(C 14 9 + 6 * C 13 8) : ℚ) / (C 20 9 : ℚ) :=
by sorry

theorem probability_no_more_than_five_girls_between_boys :
  (↑(C 14 9 + 6 * C 13 8) : ℚ) / (C 20 9 : ℚ) =
  9724 / 167960 :=
by sorry

end NUMINAMATH_CALUDE_lineup_probability_probability_no_more_than_five_girls_between_boys_l2199_219945


namespace NUMINAMATH_CALUDE_x_value_proof_l2199_219984

theorem x_value_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 4) (h2 : y^2 / z = 9) (h3 : z^2 / x = 16) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2199_219984


namespace NUMINAMATH_CALUDE_sum_of_roots_l2199_219927

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x

-- State the theorem
theorem sum_of_roots (h k : ℝ) (h_root : p h = 1) (k_root : p k = 5) : h + k = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2199_219927


namespace NUMINAMATH_CALUDE_ages_sum_l2199_219904

theorem ages_sum (a b c : ℕ+) (h1 : b = c) (h2 : a * b * c = 72) : a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l2199_219904


namespace NUMINAMATH_CALUDE_tank_bucket_ratio_l2199_219961

theorem tank_bucket_ratio (tank bucket : ℝ) 
  (h1 : 3/5 * tank - (1/2 - 1/4) * bucket = 2/3 * tank) 
  (h2 : tank > 0) 
  (h3 : bucket > 0) : 
  tank / bucket = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_tank_bucket_ratio_l2199_219961


namespace NUMINAMATH_CALUDE_vector_combination_equality_l2199_219989

/-- Given vectors a, b, and c in ℝ³, prove that 2a - 3b + 4c equals (16, 0, -19) -/
theorem vector_combination_equality (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (3, 5, 1)) 
  (hb : b = (2, 2, 3)) 
  (hc : c = (4, -1, -3)) : 
  (2 : ℝ) • a - (3 : ℝ) • b + (4 : ℝ) • c = (16, 0, -19) := by
  sorry

end NUMINAMATH_CALUDE_vector_combination_equality_l2199_219989


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2199_219949

theorem quadratic_inequality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) :
  ∀ x : ℝ, c * x^2 - b * x + a > c * x - b :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2199_219949


namespace NUMINAMATH_CALUDE_midpoint_coordinate_relation_l2199_219966

/-- Given two points D and E in the plane, if F is their midpoint,
    then 3 times the x-coordinate of F minus 5 times the y-coordinate of F equals 9. -/
theorem midpoint_coordinate_relation :
  let D : ℝ × ℝ := (30, 10)
  let E : ℝ × ℝ := (6, 8)
  let F : ℝ × ℝ := ((D.1 + E.1) / 2, (D.2 + E.2) / 2)
  3 * F.1 - 5 * F.2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_relation_l2199_219966


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2199_219930

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 2 * x^2 - 3 * x - 5
  ∃ x₁ x₂ : ℝ, x₁ = 5/2 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2199_219930


namespace NUMINAMATH_CALUDE_function_shift_l2199_219951

theorem function_shift (f : ℝ → ℝ) :
  (∀ x, f (x - 1) = x^2 + 4*x - 5) →
  (∀ x, f (x + 1) = x^2 + 8*x + 7) :=
by
  sorry

end NUMINAMATH_CALUDE_function_shift_l2199_219951


namespace NUMINAMATH_CALUDE_select_blocks_count_l2199_219908

/-- The number of ways to select 4 blocks from a 6x6 grid with no two in the same row or column -/
def select_blocks : ℕ :=
  (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)

/-- Theorem stating that the number of ways to select 4 blocks from a 6x6 grid
    with no two in the same row or column is 5400 -/
theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end NUMINAMATH_CALUDE_select_blocks_count_l2199_219908


namespace NUMINAMATH_CALUDE_restaurant_tax_calculation_l2199_219906

-- Define the tax calculation function
def calculate_tax (turnover : ℕ) : ℕ :=
  if turnover ≤ 1000 then
    300
  else
    300 + (turnover - 1000) * 4 / 100

-- Theorem statement
theorem restaurant_tax_calculation :
  calculate_tax 35000 = 1660 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_tax_calculation_l2199_219906


namespace NUMINAMATH_CALUDE_thomas_salary_l2199_219974

/-- Given the average salaries of two groups, prove Thomas's salary --/
theorem thomas_salary (raj roshan thomas : ℕ) : 
  (raj + roshan) / 2 = 4000 →
  (raj + roshan + thomas) / 3 = 5000 →
  thomas = 7000 := by
sorry

end NUMINAMATH_CALUDE_thomas_salary_l2199_219974


namespace NUMINAMATH_CALUDE_goods_payment_calculation_l2199_219933

/-- Calculates the final amount to be paid for goods after rebate and sales tax. -/
def final_amount (total_cost rebate_percent sales_tax_percent : ℚ) : ℚ :=
  let rebate_amount := (rebate_percent / 100) * total_cost
  let amount_after_rebate := total_cost - rebate_amount
  let sales_tax := (sales_tax_percent / 100) * amount_after_rebate
  amount_after_rebate + sales_tax

/-- Proves that given a total cost of 6650, a rebate of 6%, and a sales tax of 10%,
    the final amount to be paid is 6876.10. -/
theorem goods_payment_calculation :
  final_amount 6650 6 10 = 6876.1 := by
  sorry

end NUMINAMATH_CALUDE_goods_payment_calculation_l2199_219933


namespace NUMINAMATH_CALUDE_sale_final_prices_correct_l2199_219983

/-- Calculates the final price of an item after a series of percentage discounts and flat discounts --/
def finalPrice (originalPrice : ℝ) (percentDiscounts : List ℝ) (flatDiscounts : List ℝ) : ℝ :=
  let applyPercentDiscount (price : ℝ) (discount : ℝ) := price * (1 - discount)
  let applyFlatDiscount (price : ℝ) (discount : ℝ) := price - discount
  let priceAfterPercentDiscounts := percentDiscounts.foldl applyPercentDiscount originalPrice
  flatDiscounts.foldl applyFlatDiscount priceAfterPercentDiscounts

/-- Proves that the final prices of the electronic item and clothing item are correct after the 4-day sale --/
theorem sale_final_prices_correct (electronicOriginalPrice clothingOriginalPrice : ℝ) 
  (h1 : electronicOriginalPrice = 480)
  (h2 : clothingOriginalPrice = 260) : 
  let electronicFinalPrice := finalPrice electronicOriginalPrice [0.1, 0.14, 0.12, 0.08] []
  let clothingFinalPrice := finalPrice clothingOriginalPrice [0.1, 0.12, 0.05] [20]
  (electronicFinalPrice = 300.78 ∧ clothingFinalPrice = 176.62) := by
  sorry


end NUMINAMATH_CALUDE_sale_final_prices_correct_l2199_219983


namespace NUMINAMATH_CALUDE_box_volume_solutions_l2199_219976

def box_volume (x : ℤ) : ℤ :=
  (x + 3) * (x - 3) * (x^3 - 5*x + 25)

def satisfies_condition (x : ℤ) : Prop :=
  x > 0 ∧ box_volume x < 1500

theorem box_volume_solutions :
  (∃ (S : Finset ℤ), (∀ x ∈ S, satisfies_condition x) ∧
                     (∀ x : ℤ, satisfies_condition x → x ∈ S) ∧
                     Finset.card S = 4) := by
  sorry

end NUMINAMATH_CALUDE_box_volume_solutions_l2199_219976


namespace NUMINAMATH_CALUDE_midpoint_specific_segment_l2199_219960

/-- The midpoint of a line segment in polar coordinates -/
def polar_midpoint (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

theorem midpoint_specific_segment :
  let (r, θ) := polar_midpoint 10 (π/4) 10 (3*π/4)
  r = 5 * Real.sqrt 2 ∧ θ = π/2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π := by
  sorry

end NUMINAMATH_CALUDE_midpoint_specific_segment_l2199_219960


namespace NUMINAMATH_CALUDE_sum_of_cubes_equation_l2199_219922

theorem sum_of_cubes_equation (x y : ℝ) : 
  x^3 + 21*x*y + y^3 = 343 → (x + y = 7 ∨ x + y = -14) := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equation_l2199_219922


namespace NUMINAMATH_CALUDE_total_commission_proof_l2199_219985

def commission_rate : ℚ := 2 / 100

def house_prices : List ℚ := [157000, 499000, 125000]

def calculate_commission (price : ℚ) : ℚ :=
  price * commission_rate

theorem total_commission_proof :
  (house_prices.map calculate_commission).sum = 15620 := by
  sorry

end NUMINAMATH_CALUDE_total_commission_proof_l2199_219985


namespace NUMINAMATH_CALUDE_smallest_z_for_inequality_l2199_219963

theorem smallest_z_for_inequality : ∃! z : ℕ, (∀ w : ℕ, 27^w > 3^24 → w ≥ z) ∧ 27^z > 3^24 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_z_for_inequality_l2199_219963


namespace NUMINAMATH_CALUDE_equation_solution_l2199_219969

theorem equation_solution : 
  ∀ x : ℝ, (2 / ((x - 1) * (x - 2)) + 2 / ((x - 2) * (x - 3)) + 2 / ((x - 3) * (x - 4)) = 1 / 3) ↔ 
  (x = 8 ∨ x = -5/2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2199_219969


namespace NUMINAMATH_CALUDE_distance_from_P_to_y_axis_l2199_219910

def point_to_y_axis_distance (x y : ℝ) : ℝ := |x|

theorem distance_from_P_to_y_axis :
  let P : ℝ × ℝ := (-3, -4)
  point_to_y_axis_distance P.1 P.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_P_to_y_axis_l2199_219910


namespace NUMINAMATH_CALUDE_nine_more_likely_than_ten_l2199_219936

def roll_combinations (sum : ℕ) : Finset (ℕ × ℕ) :=
  (Finset.range 6 ×ˢ Finset.range 6).filter (fun (a, b) => a + b + 2 = sum)

theorem nine_more_likely_than_ten :
  (roll_combinations 9).card > (roll_combinations 10).card := by
  sorry

end NUMINAMATH_CALUDE_nine_more_likely_than_ten_l2199_219936


namespace NUMINAMATH_CALUDE_max_vertex_product_sum_l2199_219962

/-- The set of numbers to be assigned to the cube faces -/
def CubeNumbers : Finset ℕ := {1, 2, 3, 8, 9, 10}

/-- A valid assignment of numbers to cube faces -/
structure CubeAssignment where
  assignment : Fin 6 → ℕ
  valid : ∀ i, assignment i ∈ CubeNumbers
  distinct : Function.Injective assignment

/-- The sum of products at vertices for a given assignment -/
def vertexProductSum (a : CubeAssignment) : ℕ :=
  let faces := a.assignment
  (faces 0 + faces 1) * (faces 2 + faces 3) * (faces 4 + faces 5)

/-- Theorem stating the maximum sum of vertex products -/
theorem max_vertex_product_sum :
  ∀ a : CubeAssignment, vertexProductSum a ≤ 1331 :=
sorry

end NUMINAMATH_CALUDE_max_vertex_product_sum_l2199_219962


namespace NUMINAMATH_CALUDE_unique_integer_with_conditions_l2199_219975

theorem unique_integer_with_conditions : ∃! n : ℤ,
  50 ≤ n ∧ n ≤ 100 ∧
  n % 7 = 0 ∧
  n % 9 = 3 ∧
  n % 6 = 3 ∧
  n = 84 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_with_conditions_l2199_219975


namespace NUMINAMATH_CALUDE_circle_center_and_difference_l2199_219977

/-- 
Given a circle described by the equation x^2 + y^2 - 10x + 4y + 13 = 0,
prove that its center is (5, -2) and x - y = 7.
-/
theorem circle_center_and_difference (x y : ℝ) :
  x^2 + y^2 - 10*x + 4*y + 13 = 0 →
  (∃ (r : ℝ), (x - 5)^2 + (y + 2)^2 = r^2) ∧
  x - y = 7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_and_difference_l2199_219977


namespace NUMINAMATH_CALUDE_distinct_arrangements_of_three_letters_l2199_219958

/-- The number of distinct arrangements of 3 unique letters -/
def arrangements_of_three_letters : ℕ := 6

/-- The word consists of 3 distinct letters -/
def number_of_letters : ℕ := 3

theorem distinct_arrangements_of_three_letters : 
  arrangements_of_three_letters = Nat.factorial number_of_letters := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_of_three_letters_l2199_219958


namespace NUMINAMATH_CALUDE_equation_value_l2199_219954

theorem equation_value (x y : ℝ) (h : x + 2 * y = 30) :
  x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l2199_219954


namespace NUMINAMATH_CALUDE_viewing_time_theorem_l2199_219934

/-- Represents the duration of the show in minutes -/
def show_duration : ℕ := 30

/-- Represents the number of days Max watches the show -/
def days_watched : ℕ := 4

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℕ) : ℚ :=
  (minutes : ℚ) / 60

/-- Theorem stating that watching a 30-minute show for 4 days results in 2 hours of viewing time -/
theorem viewing_time_theorem :
  minutes_to_hours (show_duration * days_watched) = 2 := by
  sorry

end NUMINAMATH_CALUDE_viewing_time_theorem_l2199_219934


namespace NUMINAMATH_CALUDE_godzilla_stitches_proof_l2199_219920

/-- The number of stitches Carolyn can sew per minute -/
def stitches_per_minute : ℕ := 4

/-- The number of stitches required to embroider a flower -/
def stitches_per_flower : ℕ := 60

/-- The number of stitches required to embroider a unicorn -/
def stitches_per_unicorn : ℕ := 180

/-- The number of unicorns in the embroidery -/
def num_unicorns : ℕ := 3

/-- The number of flowers in the embroidery -/
def num_flowers : ℕ := 50

/-- The total time Carolyn spends embroidering (in minutes) -/
def total_time : ℕ := 1085

/-- The number of stitches required to embroider Godzilla -/
def stitches_for_godzilla : ℕ := 800

theorem godzilla_stitches_proof : 
  stitches_for_godzilla = 
    total_time * stitches_per_minute - 
    (num_unicorns * stitches_per_unicorn + num_flowers * stitches_per_flower) := by
  sorry

end NUMINAMATH_CALUDE_godzilla_stitches_proof_l2199_219920


namespace NUMINAMATH_CALUDE_chinese_paper_probability_l2199_219911

/-- The number of Chinese exam papers in the bag -/
def chinese_papers : ℕ := 2

/-- The number of Tibetan exam papers in the bag -/
def tibetan_papers : ℕ := 3

/-- The number of English exam papers in the bag -/
def english_papers : ℕ := 1

/-- The total number of exam papers in the bag -/
def total_papers : ℕ := chinese_papers + tibetan_papers + english_papers

/-- The probability of drawing a Chinese exam paper -/
def prob_chinese : ℚ := chinese_papers / total_papers

theorem chinese_paper_probability : prob_chinese = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_chinese_paper_probability_l2199_219911


namespace NUMINAMATH_CALUDE_star_sum_squared_l2199_219987

/-- The star operation defined on real numbers -/
def star (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating the result of (x+y)² ⋆ (y+x)² -/
theorem star_sum_squared (x y : ℝ) : star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end NUMINAMATH_CALUDE_star_sum_squared_l2199_219987


namespace NUMINAMATH_CALUDE_stating_bus_passenger_count_l2199_219912

/-- 
Calculates the final number of passengers on a bus given the initial number
and the changes at various stops.
-/
def final_passengers (initial : ℕ) (first_stop_on : ℕ) (other_stops_off : ℕ) (other_stops_on : ℕ) : ℕ :=
  initial + first_stop_on - other_stops_off + other_stops_on

/-- 
Theorem stating that given the specific passenger changes described in the problem,
the final number of passengers on the bus is 49.
-/
theorem bus_passenger_count : final_passengers 50 16 22 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_stating_bus_passenger_count_l2199_219912


namespace NUMINAMATH_CALUDE_competition_sequences_eq_choose_l2199_219942

/-- Represents the number of players in each team -/
def n : ℕ := 7

/-- Represents the total number of players from both teams -/
def total_players : ℕ := 2 * n

/-- The number of different possible competition sequences -/
def competition_sequences : ℕ := Nat.choose total_players n

theorem competition_sequences_eq_choose :
  competition_sequences = 3432 := by sorry

end NUMINAMATH_CALUDE_competition_sequences_eq_choose_l2199_219942


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l2199_219997

/-- Given that the line x + y = c is the perpendicular bisector of the line segment
    from (2, 4) to (6, 8), prove that c = 10. -/
theorem perpendicular_bisector_c_value :
  ∀ c : ℝ,
  (∀ x y : ℝ, x + y = c ↔ 
    ((x - 4)^2 + (y - 6)^2 = 8) ∧ 
    ((x - 2) * (8 - 4) = (y - 4) * (6 - 2))) →
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l2199_219997


namespace NUMINAMATH_CALUDE_melanie_brownies_batches_l2199_219959

/-- Represents the number of brownies in each batch -/
def brownies_per_batch : ℕ := 20

/-- Represents the fraction of brownies set aside for the bake sale -/
def bake_sale_fraction : ℚ := 3/4

/-- Represents the fraction of remaining brownies put in a container -/
def container_fraction : ℚ := 3/5

/-- Represents the number of brownies given out -/
def brownies_given_out : ℕ := 20

/-- Proves that Melanie baked 10 batches of brownies -/
theorem melanie_brownies_batches :
  ∃ (batches : ℕ),
    batches = 10 ∧
    (brownies_per_batch * batches : ℚ) * (1 - bake_sale_fraction) * (1 - container_fraction) =
      brownies_given_out :=
by sorry

end NUMINAMATH_CALUDE_melanie_brownies_batches_l2199_219959


namespace NUMINAMATH_CALUDE_expression_value_at_negative_two_l2199_219902

theorem expression_value_at_negative_two :
  ∀ x : ℝ, x = -2 → x * x^2 * (1/x) = 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_at_negative_two_l2199_219902


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2199_219923

theorem divisibility_equivalence :
  (∀ n : ℤ, 6 ∣ n → 2 ∣ n) ↔ (∀ n : ℤ, ¬(2 ∣ n) → ¬(6 ∣ n)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2199_219923


namespace NUMINAMATH_CALUDE_chosen_number_proof_l2199_219913

theorem chosen_number_proof (x : ℝ) : (x / 4) - 175 = 10 → x = 740 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l2199_219913


namespace NUMINAMATH_CALUDE_select_students_specific_selection_l2199_219903

/-- The number of ways to select 4 students with exactly 1 female student from two groups -/
theorem select_students (group_a_male : Nat) (group_a_female : Nat) 
  (group_b_male : Nat) (group_b_female : Nat) : Nat :=
  let total_ways := 
    (group_a_male.choose 1 * group_a_female.choose 1 * group_b_male.choose 2) +
    (group_a_male.choose 2 * group_b_male.choose 1 * group_b_female.choose 1)
  total_ways

/-- The specific problem instance -/
theorem specific_selection : select_students 5 3 6 2 = 345 := by
  sorry

end NUMINAMATH_CALUDE_select_students_specific_selection_l2199_219903


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l2199_219971

def sum_of_digits (n : ℕ) : ℕ := n % 10 + n / 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_fourth_number :
  let known_numbers := [34, 56, 45]
  let sum_known := known_numbers.sum
  let sum_digits_known := (known_numbers.map sum_of_digits).sum
  ∃ x : ℕ,
    is_two_digit x ∧
    (∀ y : ℕ, is_two_digit y →
      sum_digits_known + sum_of_digits x + sum_digits_known + sum_of_digits y = (sum_known + x + sum_known + y) / 3
      → x ≤ y) ∧
    x = 35 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l2199_219971


namespace NUMINAMATH_CALUDE_distance_between_points_l2199_219956

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (6, 5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l2199_219956


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2199_219947

-- Define set A as the domain of y = lg x
def A : Set ℝ := {x | x > 0}

-- Define set B
def B : Set ℝ := {0, 1, 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2199_219947


namespace NUMINAMATH_CALUDE_min_value_theorem_l2199_219998

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 8) :
  (2/x + 3/y) ≥ 25/8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 8 ∧ 2/x₀ + 3/y₀ = 25/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2199_219998


namespace NUMINAMATH_CALUDE_warehouse_space_theorem_l2199_219990

/-- Represents the warehouse with two floors and some occupied space -/
structure Warehouse :=
  (second_floor : ℝ)
  (first_floor : ℝ)
  (occupied_space : ℝ)

/-- The remaining available space in the warehouse -/
def remaining_space (w : Warehouse) : ℝ :=
  w.first_floor + w.second_floor - w.occupied_space

/-- The theorem stating the remaining available space in the warehouse -/
theorem warehouse_space_theorem (w : Warehouse) 
  (h1 : w.first_floor = 2 * w.second_floor)
  (h2 : w.occupied_space = w.second_floor / 4)
  (h3 : w.occupied_space = 5000) : 
  remaining_space w = 55000 := by
  sorry

#check warehouse_space_theorem

end NUMINAMATH_CALUDE_warehouse_space_theorem_l2199_219990


namespace NUMINAMATH_CALUDE_unique_square_of_divisors_l2199_219970

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- n is a positive integer that equals the square of its number of positive divisors -/
def is_square_of_divisors (n : ℕ) : Prop :=
  n > 0 ∧ n = (num_divisors n) ^ 2

theorem unique_square_of_divisors :
  ∃! n : ℕ, is_square_of_divisors n ∧ n = 9 := by sorry

end NUMINAMATH_CALUDE_unique_square_of_divisors_l2199_219970


namespace NUMINAMATH_CALUDE_vector_problem_l2199_219909

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 3]

def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (c : ℝ), v = fun i => c * (w i)

theorem vector_problem :
  (∃ k : ℝ, perpendicular (fun i => k * (a i) + (b i)) (fun i => (a i) - 3 * (b i)) ∧ k = -2.5) ∧
  (∃ k : ℝ, parallel (fun i => k * (a i) + (b i)) (fun i => (a i) - 3 * (b i)) ∧ k = -1/3) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l2199_219909


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_3_l2199_219981

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x := by sorry

theorem negation_of_greater_than_3 :
  (¬ ∃ x : ℝ, x^2 > 3) ↔ (∀ x : ℝ, x^2 ≤ 3) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_3_l2199_219981


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2199_219940

theorem sum_of_fractions : (2 : ℚ) / 7 + 8 / 10 = 38 / 35 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2199_219940


namespace NUMINAMATH_CALUDE_gcd_7488_12467_l2199_219948

theorem gcd_7488_12467 : Nat.gcd 7488 12467 = 39 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7488_12467_l2199_219948


namespace NUMINAMATH_CALUDE_x_range_given_equation_l2199_219993

theorem x_range_given_equation (x y : ℝ) (h : x - 4 * Real.sqrt y = 2 * Real.sqrt (x - y)) :
  x = 0 ∨ (4 ≤ x ∧ x ≤ 20) := by sorry

end NUMINAMATH_CALUDE_x_range_given_equation_l2199_219993


namespace NUMINAMATH_CALUDE_units_digit_of_p_plus_two_l2199_219928

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Predicate for a number being even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem units_digit_of_p_plus_two (p : ℕ) 
  (h1 : p > 0)
  (h2 : isEven p)
  (h3 : unitsDigit p > 0)
  (h4 : unitsDigit (p^3) - unitsDigit (p^2) = 0) :
  unitsDigit (p + 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_p_plus_two_l2199_219928


namespace NUMINAMATH_CALUDE_infinite_special_integers_l2199_219978

theorem infinite_special_integers :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, (n ∣ 2^(2^n + 1) + 1) ∧ ¬(n ∣ 2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_infinite_special_integers_l2199_219978


namespace NUMINAMATH_CALUDE_base_conversion_l2199_219914

theorem base_conversion (b : ℝ) : b > 0 → (3 * 5 + 2 = b^2 + 2) → b = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l2199_219914


namespace NUMINAMATH_CALUDE_not_passes_third_quadrant_l2199_219946

/-- A linear function f(x) = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- The quadrants of the Cartesian plane -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.first  => x > 0 ∧ y > 0
  | Quadrant.second => x < 0 ∧ y > 0
  | Quadrant.third  => x < 0 ∧ y < 0
  | Quadrant.fourth => x > 0 ∧ y < 0

/-- A linear function passes through a quadrant if there exists a point (x, y) in that quadrant satisfying the function equation -/
def passesThroughQuadrant (f : LinearFunction) (q : Quadrant) : Prop :=
  ∃ x y : ℝ, y = f.m * x + f.b ∧ inQuadrant x y q

/-- The main theorem: the graph of y = -3x + 2 does not pass through the third quadrant -/
theorem not_passes_third_quadrant :
  ¬ passesThroughQuadrant { m := -3, b := 2 } Quadrant.third := by
  sorry

end NUMINAMATH_CALUDE_not_passes_third_quadrant_l2199_219946


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2199_219924

theorem quadratic_equation_roots (a : ℝ) :
  a = 1 →
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
    x₁^2 + (1 - a) * x₁ - 1 = 0 ∧
    x₂^2 + (1 - a) * x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2199_219924


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2199_219950

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℤ, (x < 0 ∧ -4 * x - k ≤ 0) ↔ (x = -1 ∨ x = -2)) →
  (8 ≤ k ∧ k < 12) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2199_219950


namespace NUMINAMATH_CALUDE_expression_value_l2199_219918

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  3 * x - 4 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2199_219918


namespace NUMINAMATH_CALUDE_perfect_square_factors_450_l2199_219915

/-- The number of perfect square factors of 450 -/
def num_perfect_square_factors : ℕ := 4

/-- The prime factorization of 450 -/
def factorization_450 : List (ℕ × ℕ) := [(2, 1), (3, 2), (5, 2)]

/-- Theorem stating that the number of perfect square factors of 450 is 4 -/
theorem perfect_square_factors_450 :
  (List.prod (List.map (fun (p : ℕ × ℕ) => p.1 ^ p.2) factorization_450) = 450) →
  (∀ (n : ℕ), n * n ∣ 450 ↔ n ∈ [1, 3, 5, 15]) →
  num_perfect_square_factors = 4 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_450_l2199_219915


namespace NUMINAMATH_CALUDE_max_A_value_l2199_219901

def digits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_valid_arrangement (a b c d e f g h i : Nat) : Prop :=
  {a, b, c, d, e, f, g, h, i} = digits

def A (a b c d e f g h i : Nat) : Nat :=
  (100*a + 10*b + c) + (100*b + 10*c + d) + (100*c + 10*d + e) +
  (100*d + 10*e + f) + (100*e + 10*f + g) + (100*f + 10*g + h) +
  (100*g + 10*h + i)

theorem max_A_value :
  ∃ (a b c d e f g h i : Nat),
    is_valid_arrangement a b c d e f g h i ∧
    A a b c d e f g h i = 4648 ∧
    ∀ (a' b' c' d' e' f' g' h' i' : Nat),
      is_valid_arrangement a' b' c' d' e' f' g' h' i' →
      A a' b' c' d' e' f' g' h' i' ≤ 4648 :=
by sorry

end NUMINAMATH_CALUDE_max_A_value_l2199_219901


namespace NUMINAMATH_CALUDE_tangent_line_problem_l2199_219988

theorem tangent_line_problem (k a : ℝ) : 
  (∃ b : ℝ, (3 = 4 + a / 2 + 1) ∧ 
             (3 = 2 * k + b) ∧ 
             (k = 2 * 2 - a / 4)) → 
  (∃ b : ℝ, (3 = 4 + a / 2 + 1) ∧ 
             (3 = 2 * k + b) ∧ 
             (k = 2 * 2 - a / 4) ∧ 
             b = -7) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l2199_219988
