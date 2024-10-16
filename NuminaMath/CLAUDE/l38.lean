import Mathlib

namespace NUMINAMATH_CALUDE_area_increase_6_to_7_l38_3823

/-- Calculates the increase in area of a square when its side length is increased by 1 unit -/
def area_increase (side_length : ℝ) : ℝ :=
  (side_length + 1)^2 - side_length^2

/-- Theorem: The increase in area of a square with side length 6 units, 
    when increased by 1 unit, is 13 square units -/
theorem area_increase_6_to_7 : area_increase 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_area_increase_6_to_7_l38_3823


namespace NUMINAMATH_CALUDE_P_divisibility_l38_3847

/-- The polynomial P(x) -/
def P (a x : ℝ) : ℝ := a^3 * x^5 + (1 - a) * x^4 + (1 + a^3) * x^2 + (1 - 3*a) * x - a^3

/-- The set of values of a for which P(x) is divisible by (x-1) -/
def A : Set ℝ := {a | ∃ q : ℝ → ℝ, ∀ x, P a x = (x - 1) * q x}

theorem P_divisibility :
  A = {1, (-1 + Real.sqrt 13) / 2, (-1 - Real.sqrt 13) / 2} :=
sorry

end NUMINAMATH_CALUDE_P_divisibility_l38_3847


namespace NUMINAMATH_CALUDE_total_green_peaches_l38_3815

/-- Represents a basket of peaches -/
structure Basket :=
  (red : ℕ)
  (green : ℕ)

/-- Proves that the total number of green peaches is 9 given the conditions -/
theorem total_green_peaches
  (b1 b2 b3 : Basket)
  (h1 : b1.red = 4)
  (h2 : b2.red = 4)
  (h3 : b3.red = 3)
  (h_total : b1.red + b1.green + b2.red + b2.green + b3.red + b3.green = 20) :
  b1.green + b2.green + b3.green = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_green_peaches_l38_3815


namespace NUMINAMATH_CALUDE_mark_travel_distance_l38_3826

/-- Represents the time in minutes to travel one mile on day 1 -/
def initial_time : ℕ := 3

/-- Calculates the time in minutes to travel one mile on a given day -/
def time_for_mile (day : ℕ) : ℕ :=
  initial_time + 3 * (day - 1)

/-- Calculates the distance traveled in miles on a given day -/
def distance_per_day (day : ℕ) : ℕ :=
  if 60 % (time_for_mile day) = 0 then 60 / (time_for_mile day) else 0

/-- Calculates the total distance traveled over 6 days -/
def total_distance : ℕ :=
  (List.range 6).map (fun i => distance_per_day (i + 1)) |> List.sum

theorem mark_travel_distance :
  total_distance = 39 := by sorry

end NUMINAMATH_CALUDE_mark_travel_distance_l38_3826


namespace NUMINAMATH_CALUDE_marias_number_l38_3834

theorem marias_number : ∃ x : ℚ, (((3 * x - 6) * 5) / 2 = 94) ∧ (x = 218 / 15) := by
  sorry

end NUMINAMATH_CALUDE_marias_number_l38_3834


namespace NUMINAMATH_CALUDE_odd_function_properties_l38_3881

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_properties (f : ℝ → ℝ) 
    (h_odd : is_odd f) 
    (h_shift : ∀ x, f (x - 2) = -f x) : 
    (f 2 = 0) ∧ 
    (periodic f 4) ∧ 
    (∀ x, f (x + 2) = f (-x)) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_properties_l38_3881


namespace NUMINAMATH_CALUDE_amount_for_c_l38_3829

def total_amount : ℕ := 2000
def ratio_b : ℕ := 4
def ratio_c : ℕ := 16

theorem amount_for_c (total : ℕ) (rb : ℕ) (rc : ℕ) (h1 : total = total_amount) (h2 : rb = ratio_b) (h3 : rc = ratio_c) :
  (rc * total) / (rb + rc) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_amount_for_c_l38_3829


namespace NUMINAMATH_CALUDE_odd_prime_property_l38_3809

/-- P(x) is the smallest prime factor of x^2 + 1 -/
noncomputable def P (x : ℕ) : ℕ := Nat.minFac (x^2 + 1)

theorem odd_prime_property (p a : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) 
  (ha : a < p) (ha_cong : a^2 + 1 ≡ 0 [MOD p]) : 
  a ≠ p - a ∧ P a = p ∧ P (p - a) = p :=
sorry

end NUMINAMATH_CALUDE_odd_prime_property_l38_3809


namespace NUMINAMATH_CALUDE_factorial_ratio_l38_3813

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l38_3813


namespace NUMINAMATH_CALUDE_factor_twoOnesWithZeros_l38_3804

/-- Creates a number with two ones and n zeros between them -/
def twoOnesWithZeros (n : ℕ) : ℕ :=
  10^(n + 1) + 1

/-- The other factor in the decomposition -/
def otherFactor (k : ℕ) : ℕ :=
  (10^(3*k + 3) - 1) / 9999

theorem factor_twoOnesWithZeros (k : ℕ) :
  ∃ (m : ℕ), twoOnesWithZeros (3*k + 2) = 73 * 137 * m :=
sorry

end NUMINAMATH_CALUDE_factor_twoOnesWithZeros_l38_3804


namespace NUMINAMATH_CALUDE_systematic_sampling_solution_l38_3825

/-- Represents a systematic sampling problem -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ

/-- Represents a solution to a systematic sampling problem -/
structure SystematicSamplingSolution where
  excluded : ℕ
  interval : ℕ

/-- Checks if a solution is valid for a given systematic sampling problem -/
def is_valid_solution (problem : SystematicSampling) (solution : SystematicSamplingSolution) : Prop :=
  (problem.population_size - solution.excluded) % problem.sample_size = 0 ∧
  (problem.population_size - solution.excluded) / problem.sample_size = solution.interval

theorem systematic_sampling_solution 
  (problem : SystematicSampling) 
  (h_pop : problem.population_size = 102) 
  (h_sample : problem.sample_size = 9) :
  ∃ (solution : SystematicSamplingSolution), 
    solution.excluded = 3 ∧ 
    solution.interval = 11 ∧ 
    is_valid_solution problem solution :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_solution_l38_3825


namespace NUMINAMATH_CALUDE_streetlight_problem_l38_3856

/-- The number of ways to select k non-adjacent items from a sequence of n items,
    excluding the first and last items. -/
def non_adjacent_selections (n k : ℕ) : ℕ :=
  Nat.choose (n - k - 1) k

/-- The problem statement -/
theorem streetlight_problem :
  non_adjacent_selections 12 3 = Nat.choose 7 3 := by
  sorry

end NUMINAMATH_CALUDE_streetlight_problem_l38_3856


namespace NUMINAMATH_CALUDE_fibonacci_tetrahedron_volume_zero_l38_3837

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def tetrahedron_vertex (n : ℕ) : ℕ × ℕ × ℕ :=
  (fibonacci n, fibonacci (n + 1), fibonacci (n + 2))

def tetrahedron_volume (n : ℕ) : ℝ :=
  let v1 := tetrahedron_vertex n
  let v2 := tetrahedron_vertex (n + 3)
  let v3 := tetrahedron_vertex (n + 6)
  let v4 := tetrahedron_vertex (n + 9)
  -- Volume calculation would go here
  0  -- Placeholder for the actual volume calculation

theorem fibonacci_tetrahedron_volume_zero (n : ℕ) :
  tetrahedron_volume n = 0 := by
  sorry

#check fibonacci_tetrahedron_volume_zero

end NUMINAMATH_CALUDE_fibonacci_tetrahedron_volume_zero_l38_3837


namespace NUMINAMATH_CALUDE_bella_galya_distance_l38_3820

/-- The distance between two houses -/
def distance (house1 house2 : ℕ) : ℕ := sorry

/-- The order of houses along the road -/
def house_order : List String := ["Alya", "Bella", "Valya", "Galya", "Dilya"]

/-- The total distance from a house to all other houses -/
def total_distance (house : String) : ℕ := sorry

theorem bella_galya_distance :
  distance 1 3 = 150 ∧
  house_order = ["Alya", "Bella", "Valya", "Galya", "Dilya"] ∧
  total_distance "Bella" = 700 ∧
  total_distance "Valya" = 600 ∧
  total_distance "Galya" = 650 :=
by sorry

end NUMINAMATH_CALUDE_bella_galya_distance_l38_3820


namespace NUMINAMATH_CALUDE_maddie_total_cost_l38_3893

/-- Calculates the total cost of Maddie's beauty products purchase --/
def total_cost (palette_price : ℚ) (palette_quantity : ℕ) 
               (lipstick_price : ℚ) (lipstick_quantity : ℕ)
               (hair_color_price : ℚ) (hair_color_quantity : ℕ) : ℚ :=
  palette_price * palette_quantity + 
  lipstick_price * lipstick_quantity + 
  hair_color_price * hair_color_quantity

/-- Theorem stating that Maddie's total cost is $67 --/
theorem maddie_total_cost : 
  total_cost 15 3 (5/2) 4 4 3 = 67 := by
  sorry

end NUMINAMATH_CALUDE_maddie_total_cost_l38_3893


namespace NUMINAMATH_CALUDE_three_digit_sum_24_count_l38_3810

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_sum_24_count :
  (∃ (s : Finset ℕ), (∀ n ∈ s, is_three_digit n ∧ digit_sum n = 24) ∧ s.card = 4 ∧
    ∀ n, is_three_digit n → digit_sum n = 24 → n ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_24_count_l38_3810


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l38_3808

/-- A trinomial ax^2 + bx + c is a perfect square if there exists r such that ax^2 + bx + c = (rx + s)^2 for all x -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ r s : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (r * x + s)^2

/-- If x^2 + kx + 81 is a perfect square trinomial, then k = 18 or k = -18 -/
theorem perfect_square_trinomial_condition (k : ℝ) :
  IsPerfectSquareTrinomial 1 k 81 → k = 18 ∨ k = -18 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l38_3808


namespace NUMINAMATH_CALUDE_jason_added_erasers_l38_3827

/-- Given an initial number of erasers and a final number of erasers after Jason adds some,
    calculate how many erasers Jason placed in the drawer. -/
def erasers_added (initial_erasers final_erasers : ℕ) : ℕ :=
  final_erasers - initial_erasers

/-- Theorem stating that Jason added 131 erasers to the drawer. -/
theorem jason_added_erasers :
  erasers_added 139 270 = 131 := by sorry

end NUMINAMATH_CALUDE_jason_added_erasers_l38_3827


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l38_3833

/-- Given vectors a and b, if ka + b is perpendicular to a, then k = 2/5 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : b = (-2, 0)) 
  (h3 : (k • a.1 + b.1) * a.1 + (k • a.2 + b.2) * a.2 = 0) : 
  k = 2/5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l38_3833


namespace NUMINAMATH_CALUDE_triangle_area_l38_3865

theorem triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : b = 3) (h3 : c = 4) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l38_3865


namespace NUMINAMATH_CALUDE_cloth_trimming_l38_3886

theorem cloth_trimming (x : ℝ) :
  x > 0 →
  (x - 6) * (x - 5) = 120 →
  x = 15 :=
by sorry

end NUMINAMATH_CALUDE_cloth_trimming_l38_3886


namespace NUMINAMATH_CALUDE_solution_inequality1_solution_inequality2_l38_3844

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 2 * x^2 + x - 3 < 0
def inequality2 (x : ℝ) : Prop := x * (9 - x) > 0

-- Define the solution sets
def solution_set1 : Set ℝ := {x | -3/2 < x ∧ x < 1}
def solution_set2 : Set ℝ := {x | 0 < x ∧ x < 9}

-- Theorem statements
theorem solution_inequality1 : {x : ℝ | inequality1 x} = solution_set1 := by sorry

theorem solution_inequality2 : {x : ℝ | inequality2 x} = solution_set2 := by sorry

end NUMINAMATH_CALUDE_solution_inequality1_solution_inequality2_l38_3844


namespace NUMINAMATH_CALUDE_inequality_proof_l38_3891

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  1 / (x^2 + y + z) + 1 / (x + y^2 + z) + 1 / (x + y + z^2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l38_3891


namespace NUMINAMATH_CALUDE_festival_lineup_solution_valid_l38_3897

/-- Represents the minimum number of Gennadys required for the festival lineup -/
def min_gennadys (alexanders borises vasilies : ℕ) : ℕ :=
  max 0 (borises - 1 - alexanders - vasilies)

/-- Theorem stating the minimum number of Gennadys required for the given problem -/
theorem festival_lineup (alexanders borises vasilies : ℕ) 
  (h_alex : alexanders = 45)
  (h_boris : borises = 122)
  (h_vasily : vasilies = 27) :
  min_gennadys alexanders borises vasilies = 49 := by
  sorry

/-- Verifies that the solution satisfies the problem constraints -/
theorem solution_valid (alexanders borises vasilies gennadys : ℕ)
  (h_alex : alexanders = 45)
  (h_boris : borises = 122)
  (h_vasily : vasilies = 27)
  (h_gennady : gennadys = min_gennadys alexanders borises vasilies) :
  alexanders + borises + vasilies + gennadys ≥ borises + (borises - 1) := by
  sorry

end NUMINAMATH_CALUDE_festival_lineup_solution_valid_l38_3897


namespace NUMINAMATH_CALUDE_divisible_by_6_up_to_88_eq_l38_3843

def divisible_by_6_up_to_88 : Set ℕ :=
  {n | 1 < n ∧ n ≤ 88 ∧ n % 6 = 0}

theorem divisible_by_6_up_to_88_eq :
  divisible_by_6_up_to_88 = {6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84} := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_6_up_to_88_eq_l38_3843


namespace NUMINAMATH_CALUDE_f_neg_three_gt_f_neg_pi_l38_3824

/-- A function f satisfying the given condition -/
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (x₁ - x₂) * (f x₁ - f x₂) > 0

/-- Theorem stating that f(-3) > f(-π) given the condition -/
theorem f_neg_three_gt_f_neg_pi (f : ℝ → ℝ) (h : StrictlyIncreasing f) :
  f (-3) > f (-Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_f_neg_three_gt_f_neg_pi_l38_3824


namespace NUMINAMATH_CALUDE_ab_minus_one_lt_a_minus_b_l38_3890

theorem ab_minus_one_lt_a_minus_b (a b : ℝ) (ha : a > 0) (hb : b < 1) :
  a * b - 1 < a - b := by
  sorry

end NUMINAMATH_CALUDE_ab_minus_one_lt_a_minus_b_l38_3890


namespace NUMINAMATH_CALUDE_os_value_l38_3841

/-- Square with center and points on its diagonals -/
structure SquareWithPoints where
  /-- Side length of the square -/
  a : ℝ
  /-- Center of the square -/
  O : ℝ × ℝ
  /-- Point P on OA -/
  P : ℝ × ℝ
  /-- Point Q on OB -/
  Q : ℝ × ℝ
  /-- Point R on OC -/
  R : ℝ × ℝ
  /-- Point S on OD -/
  S : ℝ × ℝ
  /-- A is a vertex of the square -/
  A : ℝ × ℝ
  /-- B is a vertex of the square -/
  B : ℝ × ℝ
  /-- C is a vertex of the square -/
  C : ℝ × ℝ
  /-- D is a vertex of the square -/
  D : ℝ × ℝ
  /-- O is the center of the square ABCD -/
  h_center : O = (0, 0)
  /-- ABCD is a square with side length 2a -/
  h_square : A = (-a, a) ∧ B = (a, a) ∧ C = (a, -a) ∧ D = (-a, -a)
  /-- P is on OA with OP = 3 -/
  h_P : P = (-3*a/Real.sqrt 2, 3*a/Real.sqrt 2) ∧ Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = 3
  /-- Q is on OB with OQ = 5 -/
  h_Q : Q = (5*a/Real.sqrt 2, 5*a/Real.sqrt 2) ∧ Real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2) = 5
  /-- R is on OC with OR = 4 -/
  h_R : R = (4*a/Real.sqrt 2, -4*a/Real.sqrt 2) ∧ Real.sqrt ((R.1 - O.1)^2 + (R.2 - O.2)^2) = 4
  /-- S is on OD -/
  h_S : ∃ x : ℝ, S = (-x*a/Real.sqrt 2, -x*a/Real.sqrt 2)
  /-- X is the intersection of AB and PQ -/
  h_X : ∃ X : ℝ × ℝ, X.2 = a ∧ X.2 = (1/4)*X.1 + 15*a/(4*Real.sqrt 2)
  /-- Y is the intersection of BC and QR -/
  h_Y : ∃ Y : ℝ × ℝ, Y.1 = a ∧ Y.2 = 9*Y.1 - 40*a/Real.sqrt 2
  /-- Z is the intersection of CD and RS -/
  h_Z : ∃ Z : ℝ × ℝ, Z.2 = -a ∧ Z.2 + 4*a/Real.sqrt 2 = (4*a - S.1)/(-(4*a + S.1)) * (Z.1 - 4*a/Real.sqrt 2)
  /-- X, Y, and Z are collinear -/
  h_collinear : ∀ X Y Z : ℝ × ℝ, 
    (X.2 = a ∧ X.2 = (1/4)*X.1 + 15*a/(4*Real.sqrt 2)) →
    (Y.1 = a ∧ Y.2 = 9*Y.1 - 40*a/Real.sqrt 2) →
    (Z.2 = -a ∧ Z.2 + 4*a/Real.sqrt 2 = (4*a - S.1)/(-(4*a + S.1)) * (Z.1 - 4*a/Real.sqrt 2)) →
    (Y.2 - X.2)*(Z.1 - X.1) = (Z.2 - X.2)*(Y.1 - X.1)

/-- The main theorem: OS = 60/23 -/
theorem os_value (sq : SquareWithPoints) : 
  Real.sqrt ((sq.S.1 - sq.O.1)^2 + (sq.S.2 - sq.O.2)^2) = 60/23 := by
  sorry

end NUMINAMATH_CALUDE_os_value_l38_3841


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l38_3832

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 5 ∧ Real.sqrt (x + 1) + Real.sqrt (y + 3) ≥ Real.sqrt (a + 1) + Real.sqrt (b + 3)) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 5 → Real.sqrt (x + 1) + Real.sqrt (y + 3) ≤ 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l38_3832


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l38_3853

/-- An isosceles triangle with sides a, b, and c, where a and b satisfy |a-2|+(b-5)^2=0 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a)
  satisfiesEquation : |a - 2| + (b - 5)^2 = 0

/-- The perimeter of an isosceles triangle is 12 if it satisfies the given condition -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : t.a + t.b + t.c = 12 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l38_3853


namespace NUMINAMATH_CALUDE_laptop_discount_l38_3866

theorem laptop_discount (original_price : ℝ) 
  (first_discount second_discount : ℝ) 
  (h1 : first_discount = 0.25) 
  (h2 : second_discount = 0.10) : 
  (original_price * (1 - first_discount) * (1 - second_discount)) / original_price = 0.675 := by
  sorry

end NUMINAMATH_CALUDE_laptop_discount_l38_3866


namespace NUMINAMATH_CALUDE_first_set_video_count_l38_3819

/-- The cost of a video cassette in rupees -/
def video_cost : ℕ := 300

/-- The total cost of the first set in rupees -/
def first_set_cost : ℕ := 1110

/-- The total cost of the second set in rupees -/
def second_set_cost : ℕ := 1350

/-- The number of audio cassettes in the first set -/
def first_set_audio : ℕ := 7

/-- The number of audio cassettes in the second set -/
def second_set_audio : ℕ := 5

/-- The number of video cassettes in the second set -/
def second_set_video : ℕ := 4

/-- Theorem stating that the number of video cassettes in the first set is 3 -/
theorem first_set_video_count : 
  ∃ (audio_cost : ℕ) (first_set_video : ℕ),
    first_set_audio * audio_cost + first_set_video * video_cost = first_set_cost ∧
    second_set_audio * audio_cost + second_set_video * video_cost = second_set_cost ∧
    first_set_video = 3 :=
by sorry

end NUMINAMATH_CALUDE_first_set_video_count_l38_3819


namespace NUMINAMATH_CALUDE_smallest_max_sum_l38_3822

theorem smallest_max_sum (p q r s t : ℕ+) (h : p + q + r + s + t = 2022) :
  let N := max (p + q) (max (q + r) (max (r + s) (s + t)))
  506 ≤ N ∧ ∃ (p' q' r' s' t' : ℕ+), p' + q' + r' + s' + t' = 2022 ∧ 
    max (p' + q') (max (q' + r') (max (r' + s') (s' + t'))) = 506 :=
by sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l38_3822


namespace NUMINAMATH_CALUDE_al2s3_weight_l38_3867

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight (al_weight s_weight : ℝ) : ℝ :=
  2 * al_weight + 3 * s_weight

/-- The total weight of a given number of moles of a compound -/
def total_weight (moles : ℝ) (mol_weight : ℝ) : ℝ :=
  moles * mol_weight

/-- Theorem: The molecular weight of 3 moles of Al2S3 is 450.51 grams -/
theorem al2s3_weight : 
  let al_weight := 26.98
  let s_weight := 32.07
  let mol_weight := molecular_weight al_weight s_weight
  total_weight 3 mol_weight = 450.51 := by
sorry


end NUMINAMATH_CALUDE_al2s3_weight_l38_3867


namespace NUMINAMATH_CALUDE_sequence_values_bound_l38_3877

theorem sequence_values_bound (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 2) = -f x) :
  let a : ℕ → ℝ := λ n => f n
  ∃ S : Finset ℝ, S.card ≤ 4 ∧ ∀ n : ℕ, a n ∈ S :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_values_bound_l38_3877


namespace NUMINAMATH_CALUDE_dog_roaming_area_l38_3805

/-- The area available for a dog to roam when tied to the corner of an L-shaped garden wall. -/
theorem dog_roaming_area (wall_length : ℝ) (rope_length : ℝ) : wall_length = 16 ∧ rope_length = 8 → 
  (2 * (1/4 * Real.pi * rope_length^2)) = 32 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_dog_roaming_area_l38_3805


namespace NUMINAMATH_CALUDE_pastor_prayer_difference_l38_3828

/-- Pastor Paul's daily prayer count on weekdays -/
def paul_weekday : ℕ := 20

/-- Pastor Paul's Sunday prayer count -/
def paul_sunday : ℕ := 2 * paul_weekday

/-- Pastor Bruce's weekday prayer count -/
def bruce_weekday : ℕ := paul_weekday / 2

/-- Pastor Bruce's Sunday prayer count -/
def bruce_sunday : ℕ := 2 * paul_sunday

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Number of weekdays in a week -/
def weekdays : ℕ := 6

theorem pastor_prayer_difference :
  paul_weekday * weekdays + paul_sunday - (bruce_weekday * weekdays + bruce_sunday) = 20 := by
  sorry

end NUMINAMATH_CALUDE_pastor_prayer_difference_l38_3828


namespace NUMINAMATH_CALUDE_max_value_of_sum_and_reciprocal_l38_3873

theorem max_value_of_sum_and_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_and_reciprocal_l38_3873


namespace NUMINAMATH_CALUDE_train_length_proof_l38_3863

/-- Calculates the length of a train given bridge length, crossing time, and train speed. -/
def train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) : ℝ :=
  train_speed * crossing_time - bridge_length

/-- Proves that given the specific conditions, the train length is 844 meters. -/
theorem train_length_proof (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ)
  (h1 : bridge_length = 200)
  (h2 : crossing_time = 36)
  (h3 : train_speed = 29) :
  train_length bridge_length crossing_time train_speed = 844 := by
  sorry

#eval train_length 200 36 29

end NUMINAMATH_CALUDE_train_length_proof_l38_3863


namespace NUMINAMATH_CALUDE_michaels_regular_hours_l38_3806

/-- Proves that given the conditions of Michael's work schedule and earnings,
    the number of regular hours worked before overtime is 40. -/
theorem michaels_regular_hours
  (regular_rate : ℝ)
  (total_earnings : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 7)
  (h2 : total_earnings = 320)
  (h3 : total_hours = 42.857142857142854)
  : ∃ (regular_hours : ℝ),
    regular_hours = 40 ∧
    regular_hours * regular_rate +
    (total_hours - regular_hours) * (2 * regular_rate) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_michaels_regular_hours_l38_3806


namespace NUMINAMATH_CALUDE_area_between_quartic_and_line_l38_3854

/-- The area between a quartic function and a line that touch at two points -/
theorem area_between_quartic_and_line 
  (a b c d e p q α β : ℝ) 
  (ha : a ≠ 0) 
  (hαβ : α < β) : 
  let f := fun (x : ℝ) ↦ a * x^4 + b * x^3 + c * x^2 + d * x + e
  let g := fun (x : ℝ) ↦ p * x + q
  (∃ (x : ℝ), x = α ∨ x = β → f x = g x ∧ (deriv f) x = (deriv g) x) →
  ∫ x in α..β, |f x - g x| = a * (β - α)^5 / 30 := by
sorry

end NUMINAMATH_CALUDE_area_between_quartic_and_line_l38_3854


namespace NUMINAMATH_CALUDE_valid_sequences_count_l38_3876

/-- The number of distinct coin flip sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

/-- The number of distinct coin flip sequences of length n starting with two heads -/
def sequences_starting_with_two_heads (n : ℕ) : ℕ := 2^(n-2)

/-- The number of valid coin flip sequences of length 10, excluding those starting with two heads -/
def valid_sequences : ℕ := total_sequences 10 - sequences_starting_with_two_heads 10

theorem valid_sequences_count : valid_sequences = 768 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_count_l38_3876


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l38_3816

/-- The value of m for a hyperbola with given equation and asymptote form -/
theorem hyperbola_asymptote_slope (x y : ℝ) :
  (y^2 / 16 - x^2 / 9 = 1) →
  (∃ (m : ℝ), ∀ (x y : ℝ), y = m * x ∨ y = -m * x) →
  (∃ (m : ℝ), m = 4/3 ∧ (∀ (x y : ℝ), y = m * x ∨ y = -m * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l38_3816


namespace NUMINAMATH_CALUDE_cherry_pie_pitting_time_l38_3892

/-- Represents the time needed to pit cherries for each pound -/
structure PittingTime where
  first : ℕ  -- Time in minutes for the first pound
  second : ℕ -- Time in minutes for the second pound
  third : ℕ  -- Time in minutes for the third pound

/-- Calculates the total time in hours to pit cherries for a cherry pie -/
def total_pitting_time (pt : PittingTime) : ℚ :=
  (pt.first + pt.second + pt.third) / 60

/-- Theorem: Given the conditions, it takes 2 hours to pit all cherries for the pie -/
theorem cherry_pie_pitting_time :
  ∀ (pt : PittingTime),
    (∃ (n : ℕ), pt.first = 10 * (80 / 20) ∧
                pt.second = 8 * (80 / 20) ∧
                pt.third = 12 * (80 / 20) ∧
                n = 3) →
    total_pitting_time pt = 2 := by
  sorry


end NUMINAMATH_CALUDE_cherry_pie_pitting_time_l38_3892


namespace NUMINAMATH_CALUDE_line_slope_through_origin_and_one_neg_one_l38_3888

/-- The slope of a line passing through points (0,0) and (1,-1) is -1. -/
theorem line_slope_through_origin_and_one_neg_one : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, -1)
  (B.2 - A.2) / (B.1 - A.1) = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_through_origin_and_one_neg_one_l38_3888


namespace NUMINAMATH_CALUDE_andrews_dog_foreign_objects_l38_3894

/-- Calculates the total number of foreign objects on a dog given the number of burrs,
    the ratio of ticks to burrs, and the ratio of fleas to ticks. -/
def total_foreign_objects (burrs : ℕ) (ticks_to_burrs_ratio : ℕ) (fleas_to_ticks_ratio : ℕ) : ℕ :=
  let ticks := burrs * ticks_to_burrs_ratio
  let fleas := ticks * fleas_to_ticks_ratio
  burrs + ticks + fleas

/-- Theorem stating that for a dog with 12 burrs, 6 times as many ticks as burrs,
    and 3 times as many fleas as ticks, the total number of foreign objects is 300. -/
theorem andrews_dog_foreign_objects : 
  total_foreign_objects 12 6 3 = 300 := by
  sorry

end NUMINAMATH_CALUDE_andrews_dog_foreign_objects_l38_3894


namespace NUMINAMATH_CALUDE_cube_sum_over_product_l38_3803

theorem cube_sum_over_product (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10)
  (h_eq : (x - y)^2 + (x - z)^2 + (y - z)^2 + 6 = x*y*z) :
  (x^3 + y^3 + z^3 - 3*x*y*z) / (x*y*z) = 5 - 30 / (x*y*z) := by
sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_l38_3803


namespace NUMINAMATH_CALUDE_second_cat_blue_eyes_l38_3840

/-- The number of blue-eyed kittens the first cat has -/
def first_cat_blue : ℕ := 3

/-- The number of brown-eyed kittens the first cat has -/
def first_cat_brown : ℕ := 7

/-- The number of brown-eyed kittens the second cat has -/
def second_cat_brown : ℕ := 6

/-- The percentage of kittens with blue eyes -/
def blue_eye_percentage : ℚ := 35 / 100

/-- The number of blue-eyed kittens the second cat has -/
def second_cat_blue : ℕ := 4

theorem second_cat_blue_eyes :
  (first_cat_blue + second_cat_blue : ℚ) / 
  (first_cat_blue + first_cat_brown + second_cat_blue + second_cat_brown) = 
  blue_eye_percentage := by
  sorry

#check second_cat_blue_eyes

end NUMINAMATH_CALUDE_second_cat_blue_eyes_l38_3840


namespace NUMINAMATH_CALUDE_gan_is_axisymmetric_l38_3898

/-- A figure is axisymmetric if it can be folded along a line so that the parts on both sides of the line coincide. -/
def is_axisymmetric (figure : Type*) : Prop :=
  ∃ (line : Set figure), ∀ (point : figure), 
    ∃ (reflected_point : figure), 
      point ≠ reflected_point ∧ 
      (point ∈ line ∨ reflected_point ∈ line) ∧
      (∀ (p : figure), p ∉ line → (p = point ↔ p = reflected_point))

/-- The Chinese character "干" -/
def gan : Type* := sorry

/-- Theorem: The Chinese character "干" is an axisymmetric figure -/
theorem gan_is_axisymmetric : is_axisymmetric gan := by sorry

end NUMINAMATH_CALUDE_gan_is_axisymmetric_l38_3898


namespace NUMINAMATH_CALUDE_eightieth_number_is_eighty_l38_3838

def game_sequence (n : ℕ) : ℕ := n

theorem eightieth_number_is_eighty : game_sequence 80 = 80 := by sorry

end NUMINAMATH_CALUDE_eightieth_number_is_eighty_l38_3838


namespace NUMINAMATH_CALUDE_horner_method_v3_l38_3830

/-- The polynomial f(x) = 2 + 0.35x + 1.8x² - 3.66x³ + 6x⁴ - 5.2x⁵ + x⁶ -/
def f (x : ℝ) : ℝ := 2 + 0.35*x + 1.8*x^2 - 3.66*x^3 + 6*x^4 - 5.2*x^5 + x^6

/-- Horner's method for calculating v₃ -/
def horner_v3 (x : ℝ) : ℝ :=
  let v0 : ℝ := 1
  let v1 : ℝ := v0 * x - 5.2
  let v2 : ℝ := v1 * x + 6
  v2 * x - 3.66

theorem horner_method_v3 :
  horner_v3 (-1) = -15.86 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l38_3830


namespace NUMINAMATH_CALUDE_minimum_balls_in_box_l38_3872

theorem minimum_balls_in_box (blue : ℕ) (white : ℕ) (total : ℕ) : 
  white = 8 * blue →
  total = blue + white →
  (∀ drawn : ℕ, drawn = 100 → drawn > white) →
  total ≥ 108 := by
sorry

end NUMINAMATH_CALUDE_minimum_balls_in_box_l38_3872


namespace NUMINAMATH_CALUDE_factor_expression_l38_3878

theorem factor_expression (b : ℝ) : 49 * b^2 + 98 * b = 49 * b * (b + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l38_3878


namespace NUMINAMATH_CALUDE_percentage_not_sold_is_66_l38_3801

def initial_stock : ℕ := 800
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

theorem percentage_not_sold_is_66 : 
  (books_not_sold : ℚ) / (initial_stock : ℚ) * 100 = 66 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_sold_is_66_l38_3801


namespace NUMINAMATH_CALUDE_intersection_points_range_l38_3859

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem intersection_points_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f x₁ = a ∧ f x₂ = a ∧ f x₃ = a) ↔ 
  -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_range_l38_3859


namespace NUMINAMATH_CALUDE_complex_cube_root_l38_3870

theorem complex_cube_root (x y : ℕ+) :
  (↑x + ↑y * I : ℂ)^3 = 2 + 11 * I →
  ↑x + ↑y * I = 2 + I :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_root_l38_3870


namespace NUMINAMATH_CALUDE_work_completion_time_l38_3858

/-- 
Given that:
- A and B can do the same work
- B can do the work in 16 days
- A and B together can do the work in 16/3 days
Prove that A can do the work alone in 8 days
-/
theorem work_completion_time (b_time a_and_b_time : ℚ) 
  (hb : b_time = 16)
  (hab : a_and_b_time = 16 / 3) : 
  ∃ (a_time : ℚ), a_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l38_3858


namespace NUMINAMATH_CALUDE_largest_B_term_l38_3800

def B (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.1 ^ k)

theorem largest_B_term : 
  ∀ k ∈ Finset.range 501, B 45 ≥ B k :=
sorry

end NUMINAMATH_CALUDE_largest_B_term_l38_3800


namespace NUMINAMATH_CALUDE_basketball_cards_cost_l38_3836

/-- The cost of Mary's sunglasses -/
def sunglasses_cost : ℕ := 50

/-- The number of sunglasses Mary bought -/
def num_sunglasses : ℕ := 2

/-- The cost of Mary's jeans -/
def jeans_cost : ℕ := 100

/-- The cost of Rose's shoes -/
def shoes_cost : ℕ := 150

/-- The number of basketball card decks Rose bought -/
def num_card_decks : ℕ := 2

/-- Mary's total spending -/
def mary_total : ℕ := num_sunglasses * sunglasses_cost + jeans_cost

/-- Rose's total spending -/
def rose_total : ℕ := shoes_cost + num_card_decks * (mary_total - shoes_cost) / num_card_decks

theorem basketball_cards_cost (h : mary_total = rose_total) : 
  (mary_total - shoes_cost) / num_card_decks = 25 := by
  sorry

end NUMINAMATH_CALUDE_basketball_cards_cost_l38_3836


namespace NUMINAMATH_CALUDE_total_amount_correct_l38_3860

/-- The total amount earned from selling notebooks -/
def total_amount (a b : ℝ) : ℝ :=
  70 * (1 + 0.2) * a + 30 * (a - b)

/-- Proof that the total amount is correct -/
theorem total_amount_correct (a b : ℝ) :
  let total_notebooks : ℕ := 100
  let first_batch : ℕ := 70
  let price_increase : ℝ := 0.2
  total_amount a b = first_batch * (1 + price_increase) * a + (total_notebooks - first_batch) * (a - b) :=
by sorry

end NUMINAMATH_CALUDE_total_amount_correct_l38_3860


namespace NUMINAMATH_CALUDE_log_sin_in_terms_of_m_n_l38_3855

open Real

theorem log_sin_in_terms_of_m_n (α m n : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2)
  (h3 : log (1 + cos α) = m)
  (h4 : log (1 / (1 - cos α)) = n) :
  log (sin α) = (1 / 2) * (m - 1 / n) := by
  sorry

end NUMINAMATH_CALUDE_log_sin_in_terms_of_m_n_l38_3855


namespace NUMINAMATH_CALUDE_exists_number_satisfying_equation_l38_3885

theorem exists_number_satisfying_equation : ∃ x : ℝ, (x * 7) / (10 * 17) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_satisfying_equation_l38_3885


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l38_3887

/-- An arithmetic progression with first three terms x - 1, x + 1, and 2x + 3 -/
def arithmetic_progression (x : ℝ) : ℕ → ℝ
| 0 => x - 1
| 1 => x + 1
| 2 => 2*x + 3
| _ => 0  -- We only care about the first three terms

/-- The common difference of the arithmetic progression -/
def common_difference (x : ℝ) : ℝ := arithmetic_progression x 1 - arithmetic_progression x 0

theorem arithmetic_progression_x_value :
  ∀ x : ℝ, 
  (arithmetic_progression x 1 - arithmetic_progression x 0 = common_difference x) ∧
  (arithmetic_progression x 2 - arithmetic_progression x 1 = common_difference x) →
  x = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l38_3887


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l38_3862

/-- The intersection point of f(x) = log_a(x) and g(x) = (1-a)x is in the fourth quadrant when a > 1 -/
theorem intersection_in_fourth_quadrant (a : ℝ) (h : a > 1) :
  ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ Real.log x / Real.log a = (1 - a) * x := by
  sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l38_3862


namespace NUMINAMATH_CALUDE_solution_set_not_three_elements_l38_3868

noncomputable section

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a^(|x - b|)

-- Theorem statement
theorem solution_set_not_three_elements
  (a b m n p : ℝ)
  (ha : a > 0)
  (ha_neq : a ≠ 1)
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (hp : p ≠ 0) :
  ¬ ∃ (x y z : ℝ),
    (x ≠ y ∧ x ≠ z ∧ y ≠ z) ∧
    (∀ w, m * (f a b w)^2 + n * (f a b w) + p = 0 ↔ w = x ∨ w = y ∨ w = z) :=
sorry

end

end NUMINAMATH_CALUDE_solution_set_not_three_elements_l38_3868


namespace NUMINAMATH_CALUDE_total_area_form_and_sum_l38_3848

/-- Represents a rectangular prism with dimensions 1 × 1 × 2 -/
structure RectangularPrism :=
  (length : ℝ := 1)
  (width : ℝ := 1)
  (height : ℝ := 2)

/-- Represents a triangle with vertices from the rectangular prism -/
structure PrismTriangle :=
  (vertices : Fin 3 → Fin 8)

/-- Calculates the area of a PrismTriangle -/
def triangleArea (prism : RectangularPrism) (triangle : PrismTriangle) : ℝ :=
  sorry

/-- The sum of areas of all triangles whose vertices are vertices of the prism -/
def totalTriangleArea (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem stating the form of the total area and the sum of m, n, and p -/
theorem total_area_form_and_sum (prism : RectangularPrism) :
  ∃ (m n p : ℕ), totalTriangleArea prism = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 100 :=
sorry

end NUMINAMATH_CALUDE_total_area_form_and_sum_l38_3848


namespace NUMINAMATH_CALUDE_crayons_division_l38_3871

/-- Given 24 crayons equally divided among 3 people, prove that each person gets 8 crayons. -/
theorem crayons_division (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) :
  total_crayons = 24 →
  num_people = 3 →
  crayons_per_person = total_crayons / num_people →
  crayons_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_crayons_division_l38_3871


namespace NUMINAMATH_CALUDE_triangle_area_change_l38_3845

theorem triangle_area_change (h b : ℝ) (h_pos : h > 0) (b_pos : b > 0) :
  let new_height := 0.6 * h
  let new_base := b * (1 + 40 / 100)
  let original_area := (1 / 2) * b * h
  let new_area := (1 / 2) * new_base * new_height
  new_area = 0.84 * original_area :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_change_l38_3845


namespace NUMINAMATH_CALUDE_oscar_class_count_l38_3807

/-- The number of questions per student on the final exam -/
def questions_per_student : ℕ := 10

/-- The number of students per class -/
def students_per_class : ℕ := 35

/-- The total number of questions to review -/
def total_questions : ℕ := 1750

/-- The number of classes Professor Oscar has -/
def number_of_classes : ℕ := total_questions / (questions_per_student * students_per_class)

theorem oscar_class_count :
  number_of_classes = 5 := by
  sorry

end NUMINAMATH_CALUDE_oscar_class_count_l38_3807


namespace NUMINAMATH_CALUDE_centroid_distance_relation_l38_3812

/-- Given a triangle ABC with centroid G and any point P in the plane, 
    prove that the sum of squared distances from P to the vertices of the triangle 
    is equal to the sum of squared distances from G to the vertices 
    plus three times the squared distance from G to P. -/
theorem centroid_distance_relation (A B C G P : ℝ × ℝ) : 
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) →
  (A.1 - P.1)^2 + (A.2 - P.2)^2 + 
  (B.1 - P.1)^2 + (B.2 - P.2)^2 + 
  (C.1 - P.1)^2 + (C.2 - P.2)^2 = 
  (A.1 - G.1)^2 + (A.2 - G.2)^2 + 
  (B.1 - G.1)^2 + (B.2 - G.2)^2 + 
  (C.1 - G.1)^2 + (C.2 - G.2)^2 + 
  3 * ((G.1 - P.1)^2 + (G.2 - P.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_centroid_distance_relation_l38_3812


namespace NUMINAMATH_CALUDE_Al2O3_weight_and_H2_volume_l38_3857

/-- Molar mass of Aluminum in g/mol -/
def molar_mass_Al : ℝ := 26.98

/-- Molar mass of Oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Volume occupied by 1 mole of gas at STP in liters -/
def molar_volume_STP : ℝ := 22.4

/-- Molar mass of Al2O3 in g/mol -/
def molar_mass_Al2O3 : ℝ := 2 * molar_mass_Al + 3 * molar_mass_O

/-- Number of moles of Al2O3 -/
def moles_Al2O3 : ℝ := 6

/-- Theorem stating the weight of Al2O3 and volume of H2 produced -/
theorem Al2O3_weight_and_H2_volume :
  (moles_Al2O3 * molar_mass_Al2O3 = 611.76) ∧
  (moles_Al2O3 * 3 * molar_volume_STP = 403.2) := by
  sorry

end NUMINAMATH_CALUDE_Al2O3_weight_and_H2_volume_l38_3857


namespace NUMINAMATH_CALUDE_horner_method_multiplications_for_degree_5_l38_3879

def horner_multiplications (n : ℕ) : ℕ := n

theorem horner_method_multiplications_for_degree_5 :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  let f : ℝ → ℝ := λ x => a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀
  horner_multiplications 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_multiplications_for_degree_5_l38_3879


namespace NUMINAMATH_CALUDE_train_speed_l38_3818

theorem train_speed (length time : ℝ) (h1 : length = 300) (h2 : time = 15) :
  length / time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l38_3818


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l38_3896

theorem complex_number_in_third_quadrant (z : ℂ) (h : (z + Complex.I) * Complex.I = 1 + z) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l38_3896


namespace NUMINAMATH_CALUDE_enchanted_creatures_gala_handshakes_l38_3814

/-- The number of handshakes at the Enchanted Creatures Gala -/
theorem enchanted_creatures_gala_handshakes : 
  let num_goblins : ℕ := 30
  let num_trolls : ℕ := 20
  let goblin_handshakes := num_goblins * (num_goblins - 1) / 2
  let goblin_troll_handshakes := num_goblins * num_trolls
  goblin_handshakes + goblin_troll_handshakes = 1035 := by
  sorry

#check enchanted_creatures_gala_handshakes

end NUMINAMATH_CALUDE_enchanted_creatures_gala_handshakes_l38_3814


namespace NUMINAMATH_CALUDE_larger_number_is_42_l38_3835

theorem larger_number_is_42 (x y : ℝ) (sum_eq : x + y = 77) (ratio_eq : 5 * x = 6 * y) :
  max x y = 42 := by
sorry

end NUMINAMATH_CALUDE_larger_number_is_42_l38_3835


namespace NUMINAMATH_CALUDE_subtracted_number_l38_3864

theorem subtracted_number (x : ℤ) : 88 - x = 54 → x = 34 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l38_3864


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l38_3849

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : x^2 + y^2 = 20) 
  (h2 : x * y = 6) : 
  (x + y)^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l38_3849


namespace NUMINAMATH_CALUDE_solution_set_equality_l38_3851

theorem solution_set_equality : {x : ℤ | (3*x - 1)*(x + 3) = 0} = {-3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l38_3851


namespace NUMINAMATH_CALUDE_picture_frame_length_l38_3874

/-- Given a rectangular frame with perimeter 30 cm and width 10 cm, its length is 5 cm. -/
theorem picture_frame_length (perimeter width : ℝ) (h1 : perimeter = 30) (h2 : width = 10) :
  let length := (perimeter - 2 * width) / 2
  length = 5 := by sorry

end NUMINAMATH_CALUDE_picture_frame_length_l38_3874


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l38_3846

theorem expression_simplification_and_evaluation :
  ∀ a : ℤ, -Real.sqrt 2 < a ∧ a < Real.sqrt 5 →
  (∃ x : ℚ, (a - 1 - 3 / (a + 1)) / ((a^2 - 4*a + 4) / (a + 1)) = x) →
  (∃ x : ℚ, (a + 2) / (a - 2) = x) ∧
  (a = 0 ∨ a = 1) ∧
  ((a = 0 → (a + 2) / (a - 2) = -1) ∧
   (a = 1 → (a + 2) / (a - 2) = -3)) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l38_3846


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l38_3883

/-- The time taken by two cyclists meeting on the road -/
theorem cyclists_meeting_time (x y : ℝ) 
  (h1 : x - 4 = y - 9)  -- Time before meeting is equal for both cyclists
  (h2 : 4 / (y - 9) = (x - 4) / 9)  -- Proportion of speeds based on distances
  : x = 10 ∧ y = 15 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l38_3883


namespace NUMINAMATH_CALUDE_systematic_sampling_l38_3889

/-- Systematic sampling for a given population and sample size -/
theorem systematic_sampling
  (population : ℕ)
  (sample_size : ℕ)
  (h_pop : population = 1650)
  (h_sample : sample_size = 35)
  : ∃ (removed : ℕ) (segments : ℕ),
    removed = 5 ∧
    segments = 35 ∧
    (population - removed) % segments = 0 ∧
    (population - removed) / segments = sample_size :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l38_3889


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l38_3831

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 6 > 0}
def B : Set ℝ := {x | x - 1 > 0}

-- Define the complement of A in ℝ
def C_R_A : Set ℝ := (Set.univ : Set ℝ) \ A

-- State the theorem
theorem complement_A_intersect_B : C_R_A ∩ B = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l38_3831


namespace NUMINAMATH_CALUDE_equation_solution_l38_3850

theorem equation_solution :
  ∃ y : ℝ, y ≠ 2 ∧ (7 * y / (y - 2) - 5 / (y - 2) = 2 / (y - 2)) ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l38_3850


namespace NUMINAMATH_CALUDE_power_function_inequality_l38_3852

-- Define the power function
def f (x : ℝ) : ℝ := x^(4/5)

-- State the theorem
theorem power_function_inequality (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) : 
  f ((x₁ + x₂)/2) > (f x₁ + f x₂)/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_inequality_l38_3852


namespace NUMINAMATH_CALUDE_martha_points_l38_3882

/-- Represents Martha's shopping trip and point system. -/
structure ShoppingTrip where
  /-- Points earned per $10 spent -/
  pointsPerTen : ℕ
  /-- Bonus points for spending over $100 -/
  overHundredBonus : ℕ
  /-- Bonus points for 5th visit -/
  fifthVisitBonus : ℕ
  /-- Price of beef per pound -/
  beefPrice : ℚ
  /-- Quantity of beef in pounds -/
  beefQuantity : ℕ
  /-- Discount on beef as a percentage -/
  beefDiscount : ℚ
  /-- Price of fruits and vegetables per pound -/
  fruitVegPrice : ℚ
  /-- Quantity of fruits and vegetables in pounds -/
  fruitVegQuantity : ℕ
  /-- Discount on fruits and vegetables as a percentage -/
  fruitVegDiscount : ℚ
  /-- Price of spices per jar -/
  spicePrice : ℚ
  /-- Quantity of spice jars -/
  spiceQuantity : ℕ
  /-- Discount on spices as a percentage -/
  spiceDiscount : ℚ
  /-- Price of other groceries before coupon -/
  otherGroceriesPrice : ℚ
  /-- Coupon value for other groceries -/
  otherGroceriesCoupon : ℚ

/-- Calculates the total points earned during the shopping trip. -/
def calculatePoints (trip : ShoppingTrip) : ℕ :=
  sorry

/-- Theorem stating that Martha earns 850 points given the specific shopping conditions. -/
theorem martha_points : ∃ (trip : ShoppingTrip),
  trip.pointsPerTen = 50 ∧
  trip.overHundredBonus = 250 ∧
  trip.fifthVisitBonus = 100 ∧
  trip.beefPrice = 11 ∧
  trip.beefQuantity = 3 ∧
  trip.beefDiscount = 1/10 ∧
  trip.fruitVegPrice = 4 ∧
  trip.fruitVegQuantity = 8 ∧
  trip.fruitVegDiscount = 2/25 ∧
  trip.spicePrice = 6 ∧
  trip.spiceQuantity = 3 ∧
  trip.spiceDiscount = 1/20 ∧
  trip.otherGroceriesPrice = 37 ∧
  trip.otherGroceriesCoupon = 3 ∧
  calculatePoints trip = 850 :=
sorry

end NUMINAMATH_CALUDE_martha_points_l38_3882


namespace NUMINAMATH_CALUDE_max_salary_320000_l38_3842

/-- Represents a baseball team with salary constraints -/
structure BaseballTeam where
  num_players : ℕ
  min_salary : ℕ
  total_salary_cap : ℕ

/-- Calculates the maximum possible salary for a single player in a baseball team -/
def max_single_player_salary (team : BaseballTeam) : ℕ :=
  team.total_salary_cap - (team.num_players - 1) * team.min_salary

/-- Theorem stating the maximum possible salary for a single player in a specific baseball team -/
theorem max_salary_320000 :
  let team : BaseballTeam := ⟨25, 20000, 800000⟩
  max_single_player_salary team = 320000 := by
  sorry

#eval max_single_player_salary ⟨25, 20000, 800000⟩

end NUMINAMATH_CALUDE_max_salary_320000_l38_3842


namespace NUMINAMATH_CALUDE_division_remainder_proof_l38_3869

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 2944) (h2 : divisor = 72) (h3 : quotient = 40) :
  dividend - divisor * quotient = 64 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l38_3869


namespace NUMINAMATH_CALUDE_correct_num_tables_l38_3899

/-- The number of tables in the lunchroom -/
def num_tables : ℕ := 6

/-- The initial number of students per table -/
def initial_students_per_table : ℚ := 6

/-- The desired number of students per table -/
def desired_students_per_table : ℚ := 17 / 3

/-- Theorem stating that the number of tables is correct -/
theorem correct_num_tables :
  (initial_students_per_table * num_tables : ℚ) =
  (desired_students_per_table * num_tables : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_correct_num_tables_l38_3899


namespace NUMINAMATH_CALUDE_investment_change_investment_change_specific_l38_3839

theorem investment_change (initial_investment : ℝ) 
                          (first_year_loss_percent : ℝ) 
                          (second_year_gain_percent : ℝ) : ℝ :=
  let first_year_amount := initial_investment * (1 - first_year_loss_percent / 100)
  let second_year_amount := first_year_amount * (1 + second_year_gain_percent / 100)
  let total_change_percent := (second_year_amount - initial_investment) / initial_investment * 100
  total_change_percent

theorem investment_change_specific : 
  investment_change 200 10 25 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_investment_change_investment_change_specific_l38_3839


namespace NUMINAMATH_CALUDE_female_officers_count_l38_3821

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 170 →
  female_on_duty_ratio = 17 / 100 →
  female_ratio = 1 / 2 →
  ∃ (total_female : ℕ), 
    (female_on_duty_ratio * total_female = female_ratio * total_on_duty) ∧
    total_female = 500 :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l38_3821


namespace NUMINAMATH_CALUDE_water_poured_out_l38_3817

-- Define the initial and final amounts of water
def initial_amount : ℝ := 0.8
def final_amount : ℝ := 0.6

-- Define the amount of water poured out
def poured_out : ℝ := initial_amount - final_amount

-- Theorem to prove
theorem water_poured_out : poured_out = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_water_poured_out_l38_3817


namespace NUMINAMATH_CALUDE_radical_equality_implies_c_equals_six_l38_3802

theorem radical_equality_implies_c_equals_six 
  (a b c : ℕ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) 
  (h : ∀ M : ℝ, M ≠ 1 → M^(1/a + 1/(a*b) + 3/(a*b*c)) = M^(14/24)) : 
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_radical_equality_implies_c_equals_six_l38_3802


namespace NUMINAMATH_CALUDE_median_sum_ge_four_times_circumradius_l38_3880

/-- A triangle in a 2D Euclidean space --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The radius of the circumscribed circle of a triangle --/
def circumradius (t : Triangle) : ℝ := sorry

/-- The length of a median of a triangle --/
def median_length (t : Triangle) (vertex : Fin 3) : ℝ := sorry

/-- Predicate to check if a triangle is not obtuse --/
def is_not_obtuse (t : Triangle) : Prop := sorry

/-- Theorem: For any non-obtuse triangle, the sum of its three medians
    is greater than or equal to four times the radius of its circumscribed circle --/
theorem median_sum_ge_four_times_circumradius (t : Triangle) :
  is_not_obtuse t →
  (median_length t 0) + (median_length t 1) + (median_length t 2) ≥ 4 * (circumradius t) := by
  sorry

end NUMINAMATH_CALUDE_median_sum_ge_four_times_circumradius_l38_3880


namespace NUMINAMATH_CALUDE_expected_condition_sufferers_l38_3861

theorem expected_condition_sufferers (total_sample : ℕ) (condition_rate : ℚ) : 
  total_sample = 450 → condition_rate = 1/3 → 
  (condition_rate * total_sample : ℚ) = 150 := by
sorry

end NUMINAMATH_CALUDE_expected_condition_sufferers_l38_3861


namespace NUMINAMATH_CALUDE_student_calculation_error_l38_3875

theorem student_calculation_error (N : ℝ) : (5/4)*N - (4/5)*N = 36 → N = 80 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_error_l38_3875


namespace NUMINAMATH_CALUDE_first_player_always_wins_l38_3884

/-- Represents a position on the rectangular table -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents the state of the game -/
structure GameState :=
  (table : Set Position)
  (occupied : Set Position)
  (currentPlayer : Bool)

/-- The winning strategy for the first player -/
def firstPlayerStrategy (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → Position),
    ∀ (state : GameState),
      state.currentPlayer = true →
      strategy state ∉ state.occupied →
      strategy state ∈ state.table

/-- The main theorem stating that the first player always has a winning strategy -/
theorem first_player_always_wins :
  ∀ (initialState : GameState),
    initialState.occupied = ∅ →
    initialState.table.Nonempty →
    ∃ (center : Position), center ∈ initialState.table →
      firstPlayerStrategy initialState :=
sorry

end NUMINAMATH_CALUDE_first_player_always_wins_l38_3884


namespace NUMINAMATH_CALUDE_donut_selections_l38_3895

/-- The number of ways to select n items from k types with unlimited supply -/
def selections (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of donut types -/
def donut_types : ℕ := 3

/-- The number of donuts to be selected -/
def donuts_to_select : ℕ := 5

theorem donut_selections :
  selections donuts_to_select donut_types = 21 := by
  sorry

end NUMINAMATH_CALUDE_donut_selections_l38_3895


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l38_3811

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 8200 →
  candidate_percentage = 35/100 →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 2460 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l38_3811
