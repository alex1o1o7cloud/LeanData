import Mathlib

namespace NUMINAMATH_CALUDE_probability_same_length_l1365_136549

/-- The set of all sides and diagonals of a regular hexagon -/
def S : Finset ℕ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of segments in S -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The number of ways to choose 2 segments from S -/
def total_choices : ℕ := (total_segments.choose 2)

/-- The number of ways to choose 2 sides -/
def side_choices : ℕ := (num_sides.choose 2)

/-- The number of ways to choose 2 diagonals -/
def diagonal_choices : ℕ := (num_diagonals.choose 2)

/-- The total number of favorable outcomes (choosing two segments of the same length) -/
def favorable_outcomes : ℕ := side_choices + diagonal_choices

/-- The probability of selecting two segments of the same length from S -/
theorem probability_same_length : 
  (favorable_outcomes : ℚ) / total_choices = 17 / 35 :=
sorry

end NUMINAMATH_CALUDE_probability_same_length_l1365_136549


namespace NUMINAMATH_CALUDE_intersection_eq_interval_l1365_136506

open Set

-- Define the sets M and N
def M : Set ℝ := {x | 2 - x > 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the interval [1, 2)
def interval_1_2 : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- Theorem statement
theorem intersection_eq_interval : M ∩ N = interval_1_2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_eq_interval_l1365_136506


namespace NUMINAMATH_CALUDE_prom_services_cost_l1365_136589

/-- Calculate the total cost of prom services for Keesha --/
theorem prom_services_cost : 
  let hair_cost : ℚ := 50
  let hair_discount : ℚ := 0.1
  let manicure_cost : ℚ := 30
  let pedicure_cost : ℚ := 35
  let pedicure_discount : ℚ := 0.5
  let makeup_cost : ℚ := 40
  let makeup_tax : ℚ := 0.05
  let tip_percentage : ℚ := 0.2

  let hair_total := (hair_cost * (1 - hair_discount)) * (1 + tip_percentage)
  let nails_total := (manicure_cost + pedicure_cost * pedicure_discount) * (1 + tip_percentage)
  let makeup_total := (makeup_cost * (1 + makeup_tax)) * (1 + tip_percentage)

  hair_total + nails_total + makeup_total = 161.4 := by
    sorry

end NUMINAMATH_CALUDE_prom_services_cost_l1365_136589


namespace NUMINAMATH_CALUDE_yoongi_hoseok_age_sum_l1365_136538

/-- Given the ages of Yoongi's aunt, the age difference between Yoongi and his aunt,
    and the age difference between Yoongi and Hoseok, prove that the sum of
    Yoongi and Hoseok's ages is 26 years. -/
theorem yoongi_hoseok_age_sum :
  ∀ (aunt_age : ℕ) (yoongi_aunt_diff : ℕ) (yoongi_hoseok_diff : ℕ),
  aunt_age = 38 →
  yoongi_aunt_diff = 23 →
  yoongi_hoseok_diff = 4 →
  (aunt_age - yoongi_aunt_diff) + (aunt_age - yoongi_aunt_diff - yoongi_hoseok_diff) = 26 :=
by sorry

end NUMINAMATH_CALUDE_yoongi_hoseok_age_sum_l1365_136538


namespace NUMINAMATH_CALUDE_cat_toy_cost_l1365_136559

/-- The cost of a cat toy given the total amount paid, the cost of a cage, and the change received. -/
theorem cat_toy_cost (total_paid : ℚ) (cage_cost : ℚ) (change : ℚ) :
  total_paid = 20 →
  cage_cost = 10.97 →
  change = 0.26 →
  total_paid - change - cage_cost = 8.77 := by
sorry

end NUMINAMATH_CALUDE_cat_toy_cost_l1365_136559


namespace NUMINAMATH_CALUDE_addition_point_value_l1365_136537

def optimal_range_lower : ℝ := 628
def optimal_range_upper : ℝ := 774
def good_point : ℝ := 718
def golden_ratio : ℝ := 0.618

def addition_point : ℝ := optimal_range_upper + optimal_range_lower - good_point

theorem addition_point_value :
  addition_point = 684 :=
sorry

end NUMINAMATH_CALUDE_addition_point_value_l1365_136537


namespace NUMINAMATH_CALUDE_cubic_sum_prime_power_l1365_136507

theorem cubic_sum_prime_power (a b p n : ℕ) : 
  0 < a ∧ 0 < b ∧ 0 < p ∧ 0 < n ∧ 
  Nat.Prime p ∧ 
  a^3 + b^3 = p^n →
  (∃ k : ℕ, (a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨
             (a = 2*(3^k) ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
             (a = 3^k ∧ b = 2*(3^k) ∧ p = 3 ∧ n = 3*k + 2)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_prime_power_l1365_136507


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l1365_136557

theorem gold_coin_distribution (x y : ℕ) (h1 : x > y) (h2 : x + y = 49) :
  ∃ (k : ℕ), x^2 - y^2 = k * (x - y) → k = 49 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l1365_136557


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1365_136587

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1365_136587


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1365_136522

/-- An odd function -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- An even function -/
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

/-- F(x) is increasing on (-∞, 0) -/
def increasing_on_neg (F : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → F x₁ < F x₂

theorem solution_set_equivalence (f g : ℝ → ℝ) 
  (hf : odd_function f) (hg : even_function g)
  (hF : increasing_on_neg (λ x => f x * g x))
  (hg2 : g 2 = 0) :
  {x : ℝ | f x * g x < 0} = {x : ℝ | x < -2 ∨ (0 < x ∧ x < 2)} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1365_136522


namespace NUMINAMATH_CALUDE_group_size_l1365_136597

theorem group_size (total_paise : ℕ) (h : total_paise = 7744) : 
  ∃ n : ℕ, n * n = total_paise ∧ n = 88 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l1365_136597


namespace NUMINAMATH_CALUDE_total_daisies_l1365_136548

/-- Calculates the total number of daisies in Jack's flower crowns --/
theorem total_daisies (white pink red : ℕ) : 
  white = 6 ∧ 
  pink = 9 * white ∧ 
  red = 4 * pink - 3 → 
  white + pink + red = 273 := by
sorry


end NUMINAMATH_CALUDE_total_daisies_l1365_136548


namespace NUMINAMATH_CALUDE_min_area_is_zero_l1365_136528

/-- Represents a rectangle with one integer dimension and one half-integer dimension -/
structure Rectangle where
  x : ℕ  -- Integer dimension
  y : ℚ  -- Half-integer dimension
  y_half_int : ∃ (n : ℕ), y = n + 1/2
  perimeter_150 : 2 * (x + y) = 150

/-- The area of a rectangle -/
def area (r : Rectangle) : ℚ :=
  r.x * r.y

/-- Theorem stating that the minimum area of a rectangle with the given conditions is 0 -/
theorem min_area_is_zero :
  ∃ (r : Rectangle), ∀ (s : Rectangle), area r ≤ area s :=
sorry

end NUMINAMATH_CALUDE_min_area_is_zero_l1365_136528


namespace NUMINAMATH_CALUDE_diamond_two_seven_l1365_136590

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 3 * y

-- Theorem statement
theorem diamond_two_seven : diamond 2 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_seven_l1365_136590


namespace NUMINAMATH_CALUDE_composition_result_l1365_136573

/-- Given two functions f and g, prove that f(g(f(3))) = 119 -/
theorem composition_result :
  let f (x : ℝ) := 2 * x + 5
  let g (x : ℝ) := 5 * x + 2
  f (g (f 3)) = 119 := by sorry

end NUMINAMATH_CALUDE_composition_result_l1365_136573


namespace NUMINAMATH_CALUDE_min_distance_point_l1365_136518

def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (2, 2)

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def sum_of_distances (p : ℝ × ℝ) : ℝ :=
  distance_squared p A + distance_squared p B

def is_on_line (p : ℝ × ℝ) : Prop :=
  p.1 = p.2

theorem min_distance_point :
  ∃ (p : ℝ × ℝ), is_on_line p ∧
    ∀ (q : ℝ × ℝ), is_on_line q → sum_of_distances p ≤ sum_of_distances q :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_l1365_136518


namespace NUMINAMATH_CALUDE_sqrt_two_thirds_less_than_half_l1365_136530

theorem sqrt_two_thirds_less_than_half : (Real.sqrt 2) / 3 < 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_thirds_less_than_half_l1365_136530


namespace NUMINAMATH_CALUDE_partnership_investment_l1365_136531

/-- Partnership investment problem -/
theorem partnership_investment (x : ℝ) (m : ℝ) : 
  x > 0 ∧ m > 0 →
  18900 * (12 * x) / (12 * x + 2 * x * (12 - m) + 3 * x * 4) = 6300 →
  m = 6 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_l1365_136531


namespace NUMINAMATH_CALUDE_angle_equality_l1365_136581

theorem angle_equality (C : Real) (h1 : 0 < C) (h2 : C < π) 
  (h3 : Real.cos C = Real.sin C) : C = π/4 ∨ C = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l1365_136581


namespace NUMINAMATH_CALUDE_three_digit_puzzle_l1365_136552

theorem three_digit_puzzle :
  ∀ (A B C : ℕ),
  (A ≥ 1 ∧ A ≤ 9) →
  (B ≥ 0 ∧ B ≤ 9) →
  (C ≥ 0 ∧ C ≤ 9) →
  (100 * A + 10 * B + B ≥ 100 ∧ 100 * A + 10 * B + B ≤ 999) →
  (A * B * B ≥ 10 ∧ A * B * B ≤ 99) →
  A * B * B = 10 * A + C →
  A * C = C →
  100 * A + 10 * B + B = 144 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_puzzle_l1365_136552


namespace NUMINAMATH_CALUDE_milk_pumping_rate_l1365_136565

/-- Calculates the rate of milk pumped into a tanker given initial conditions --/
theorem milk_pumping_rate 
  (initial_milk : ℝ) 
  (pumping_time : ℝ) 
  (add_rate : ℝ) 
  (add_time : ℝ) 
  (milk_left : ℝ) 
  (h1 : initial_milk = 30000)
  (h2 : pumping_time = 4)
  (h3 : add_rate = 1500)
  (h4 : add_time = 7)
  (h5 : milk_left = 28980) :
  (initial_milk + add_rate * add_time - milk_left) / pumping_time = 2880 := by
  sorry

#check milk_pumping_rate

end NUMINAMATH_CALUDE_milk_pumping_rate_l1365_136565


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_sequence_arithmetic_mean_of_sixty_integers_from_three_l1365_136534

theorem arithmetic_mean_of_sequence (n : ℕ) (start : ℕ) (count : ℕ) :
  let sequence := fun i => start + i - 1
  let sum := (count * (2 * start + count - 1)) / 2
  (sum : ℚ) / count = (2 * start + count - 1 : ℚ) / 2 :=
by sorry

theorem arithmetic_mean_of_sixty_integers_from_three :
  let n := 60
  let start := 3
  let sequence := fun i => start + i - 1
  let sum := (n * (2 * start + n - 1)) / 2
  (sum : ℚ) / n = 32.5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_sequence_arithmetic_mean_of_sixty_integers_from_three_l1365_136534


namespace NUMINAMATH_CALUDE_angle_U_measure_l1365_136580

/-- A hexagon with specific angle properties -/
structure SpecialHexagon where
  -- Define the angles of the hexagon
  F : ℝ
  G : ℝ
  I : ℝ
  R : ℝ
  U : ℝ
  E : ℝ
  -- Conditions from the problem
  angle_sum : F + G + I + R + U + E = 720
  angle_congruence : F = I ∧ I = U
  supplementary_GR : G + R = 180
  supplementary_EU : E + U = 180

/-- The measure of angle U in the special hexagon is 120 degrees -/
theorem angle_U_measure (h : SpecialHexagon) : h.U = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_U_measure_l1365_136580


namespace NUMINAMATH_CALUDE_smallest_number_proof_l1365_136560

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 20) % 15 = 0 ∧ (n - 20) % 30 = 0 ∧ (n - 20) % 45 = 0 ∧ (n - 20) % 60 = 0

theorem smallest_number_proof :
  is_divisible_by_all 200 ∧ ∀ m : ℕ, m < 200 → ¬is_divisible_by_all m :=
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l1365_136560


namespace NUMINAMATH_CALUDE_mangoes_purchased_correct_mango_kg_l1365_136502

theorem mangoes_purchased (grape_kg : ℕ) (grape_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : ℕ :=
  let mango_kg := (total_paid - grape_kg * grape_rate) / mango_rate
  mango_kg

theorem correct_mango_kg : mangoes_purchased 14 54 62 1376 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_purchased_correct_mango_kg_l1365_136502


namespace NUMINAMATH_CALUDE_equation_solution_l1365_136525

theorem equation_solution (x : ℝ) : 
  x ≠ 3 → (-x^2 = (3*x - 3) / (x - 3)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1365_136525


namespace NUMINAMATH_CALUDE_simplify_expression_l1365_136544

theorem simplify_expression : 
  2 - (2 / (2 + Real.sqrt 5)) + (2 / (2 - Real.sqrt 5)) = 2 - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1365_136544


namespace NUMINAMATH_CALUDE_expression_evaluation_l1365_136504

theorem expression_evaluation (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1365_136504


namespace NUMINAMATH_CALUDE_prob_three_even_dice_l1365_136509

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of sides on each die -/
def num_sides : ℕ := 12

/-- The number of even outcomes on each die -/
def num_even_sides : ℕ := 6

/-- The number of dice showing even numbers -/
def num_even_dice : ℕ := 3

/-- The probability of exactly three dice showing even numbers when six fair 12-sided dice are rolled -/
theorem prob_three_even_dice : 
  (num_dice.choose num_even_dice * (num_even_sides / num_sides) ^ num_even_dice * 
  ((num_sides - num_even_sides) / num_sides) ^ (num_dice - num_even_dice) : ℚ) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_dice_l1365_136509


namespace NUMINAMATH_CALUDE_exists_pentagon_with_similar_subpentagon_l1365_136594

/-- A convex pentagon with specific angles and side lengths -/
structure ConvexPentagon where
  -- Sides of the pentagon
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  -- Two angles of the pentagon (in radians)
  angle1 : ℝ
  angle2 : ℝ
  -- Convexity condition
  convex : angle1 > 0 ∧ angle2 > 0 ∧ angle1 < π ∧ angle2 < π

/-- Similarity between two pentagons -/
def isSimilar (p1 p2 : ConvexPentagon) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    p2.side1 = k * p1.side1 ∧
    p2.side2 = k * p1.side2 ∧
    p2.side3 = k * p1.side3 ∧
    p2.side4 = k * p1.side4 ∧
    p2.side5 = k * p1.side5 ∧
    p2.angle1 = p1.angle1 ∧
    p2.angle2 = p1.angle2

/-- Theorem stating the existence of a specific convex pentagon with a similar sub-pentagon -/
theorem exists_pentagon_with_similar_subpentagon :
  ∃ (p : ConvexPentagon) (q : ConvexPentagon),
    p.side1 = 2 ∧ p.side2 = 4 ∧ p.side3 = 8 ∧ p.side4 = 6 ∧ p.side5 = 12 ∧
    p.angle1 = π / 3 ∧ p.angle2 = 2 * π / 3 ∧
    isSimilar p q :=
sorry

end NUMINAMATH_CALUDE_exists_pentagon_with_similar_subpentagon_l1365_136594


namespace NUMINAMATH_CALUDE_parallel_lines_planes_l1365_136554

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the "not contained in" relation for a line and a plane
variable (not_contained_in : Line → Plane → Prop)

-- State the theorem
theorem parallel_lines_planes 
  (l m : Line) 
  (α β : Plane) 
  (h_distinct_lines : l ≠ m)
  (h_distinct_planes : α ≠ β)
  (h_alpha_beta_parallel : parallel_plane α β)
  (h_l_alpha_parallel : parallel_line_plane l α)
  (h_l_m_parallel : parallel l m)
  (h_m_not_in_beta : not_contained_in m β) :
  parallel_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_planes_l1365_136554


namespace NUMINAMATH_CALUDE_smallest_bottom_right_corner_l1365_136595

/-- Represents a 3x3 grid of natural numbers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Checks if all numbers in the grid are different -/
def all_different (g : Grid) : Prop :=
  ∀ i j k l, g i j = g k l → (i = k ∧ j = l)

/-- Checks if the sum condition is satisfied for rows -/
def row_sum_condition (g : Grid) : Prop :=
  ∀ i, g i 0 + g i 1 = g i 2

/-- Checks if the sum condition is satisfied for columns -/
def col_sum_condition (g : Grid) : Prop :=
  ∀ j, g 0 j + g 1 j = g 2 j

/-- The main theorem stating the smallest possible value for the bottom right corner -/
theorem smallest_bottom_right_corner (g : Grid) 
  (h1 : all_different g) 
  (h2 : row_sum_condition g) 
  (h3 : col_sum_condition g) : 
  g 2 2 ≥ 12 := by
  sorry


end NUMINAMATH_CALUDE_smallest_bottom_right_corner_l1365_136595


namespace NUMINAMATH_CALUDE_min_consecutive_sum_36_proof_l1365_136519

/-- The sum of N consecutive integers starting from a -/
def sum_consecutive (a : ℤ) (N : ℕ) : ℤ := N * (2 * a + N - 1) / 2

/-- Predicate to check if a sequence of N consecutive integers starting from a sums to 36 -/
def is_valid_sequence (a : ℤ) (N : ℕ) : Prop := sum_consecutive a N = 36

/-- The minimum number of consecutive integers that sum to 36 -/
def min_consecutive_sum_36 : ℕ := 3

theorem min_consecutive_sum_36_proof :
  (∃ a : ℤ, is_valid_sequence a min_consecutive_sum_36) ∧
  (∀ N : ℕ, N < min_consecutive_sum_36 → ∀ a : ℤ, ¬is_valid_sequence a N) :=
sorry

end NUMINAMATH_CALUDE_min_consecutive_sum_36_proof_l1365_136519


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1365_136527

def inequality (x : ℝ) := x^2 - 3*x - 10 > 0

theorem inequality_solution_set :
  {x : ℝ | inequality x} = {x : ℝ | x > 5 ∨ x < -2} :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1365_136527


namespace NUMINAMATH_CALUDE_smallest_number_remainder_l1365_136582

theorem smallest_number_remainder (n : ℕ) : 
  (n = 197) → 
  (∀ m : ℕ, m < n → m % 13 ≠ 2 ∨ m % 16 ≠ 5) → 
  n % 13 = 2 → 
  n % 16 = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_remainder_l1365_136582


namespace NUMINAMATH_CALUDE_triangle_ratio_l1365_136596

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_ratio (t : Triangle) 
  (h1 : t.A = π / 3)
  (h2 : Real.sin (t.B + t.C) = 6 * Real.cos t.B * Real.sin t.C) :
  t.b / t.c = (1 + Real.sqrt 21) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_ratio_l1365_136596


namespace NUMINAMATH_CALUDE_boys_average_age_l1365_136520

theorem boys_average_age (a b c : ℕ) (h1 : a = 15) (h2 : b = 3 * a) (h3 : c = 4 * a) :
  (a + b + c) / 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_boys_average_age_l1365_136520


namespace NUMINAMATH_CALUDE_negative_integer_square_plus_self_twelve_l1365_136583

theorem negative_integer_square_plus_self_twelve (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N % 3 = 0 → N = -3 := by sorry

end NUMINAMATH_CALUDE_negative_integer_square_plus_self_twelve_l1365_136583


namespace NUMINAMATH_CALUDE_subset_of_any_set_implies_zero_l1365_136588

theorem subset_of_any_set_implies_zero (a : ℝ) : 
  (∀ S : Set ℝ, {x : ℝ | a * x = 1} ⊆ S) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_of_any_set_implies_zero_l1365_136588


namespace NUMINAMATH_CALUDE_student_A_more_stable_l1365_136576

/-- Represents a student with their score variance -/
structure Student where
  name : String
  variance : ℝ

/-- Defines the concept of score stability based on variance -/
def moreStable (s1 s2 : Student) : Prop :=
  s1.variance < s2.variance

/-- Theorem stating that student A has more stable scores than student B -/
theorem student_A_more_stable :
  let studentA : Student := ⟨"A", 3.6⟩
  let studentB : Student := ⟨"B", 4.4⟩
  moreStable studentA studentB := by
  sorry

end NUMINAMATH_CALUDE_student_A_more_stable_l1365_136576


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_m_range_l1365_136598

theorem complex_in_second_quadrant_m_range (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 2) (m - 1)
  (z.re < 0 ∧ z.im > 0) → (1 < m ∧ m < Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_m_range_l1365_136598


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l1365_136561

/-- Represents the time (in hours) when two cyclists A and B are 32.5 km apart -/
def time_when_apart (initial_distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) (final_distance : ℝ) : Set ℝ :=
  {t : ℝ | t * (speed_A + speed_B) = initial_distance - final_distance ∨ 
           t * (speed_A + speed_B) = initial_distance + final_distance}

/-- Theorem stating that the time when cyclists A and B are 32.5 km apart is either 1 or 3 hours -/
theorem cyclists_meeting_time :
  time_when_apart 65 17.5 15 32.5 = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l1365_136561


namespace NUMINAMATH_CALUDE_both_correct_count_l1365_136593

theorem both_correct_count (total : ℕ) (set_correct : ℕ) (func_correct : ℕ) (both_incorrect : ℕ) :
  total = 50 →
  set_correct = 40 →
  func_correct = 31 →
  both_incorrect = 4 →
  total - both_incorrect = set_correct + func_correct - (set_correct + func_correct - (total - both_incorrect)) :=
by
  sorry

#check both_correct_count

end NUMINAMATH_CALUDE_both_correct_count_l1365_136593


namespace NUMINAMATH_CALUDE_expand_expression_l1365_136511

theorem expand_expression (y : ℝ) : 5 * (3 * y^3 + 4 * y^2 - 7 * y + 2) = 15 * y^3 + 20 * y^2 - 35 * y + 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1365_136511


namespace NUMINAMATH_CALUDE_no_perfect_square_polynomial_l1365_136556

theorem no_perfect_square_polynomial (n : ℕ) : ∃ (m : ℕ), n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3 ≠ m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_polynomial_l1365_136556


namespace NUMINAMATH_CALUDE_adoption_time_proof_l1365_136515

/-- The number of days required to adopt all puppies -/
def adoption_days (initial_puppies : ℕ) (new_puppies : ℕ) (adopted_per_day : ℕ) : ℕ :=
  (initial_puppies + new_puppies) / adopted_per_day

/-- Theorem stating that it takes 9 days to adopt all puppies under given conditions -/
theorem adoption_time_proof :
  adoption_days 2 34 4 = 9 :=
by sorry

end NUMINAMATH_CALUDE_adoption_time_proof_l1365_136515


namespace NUMINAMATH_CALUDE_triangle_angle_bounds_l1365_136541

/-- Given a positive real number R, if R, 1, and R+1/2 form a triangle with θ as the angle between R and R+1/2, then 1 < 2Rθ < π. -/
theorem triangle_angle_bounds (R : ℝ) (θ : ℝ) (h_pos : R > 0) 
  (h_triangle : R + 1 > R + 1/2 ∧ R + (R + 1/2) > 1 ∧ 1 + (R + 1/2) > R) 
  (h_angle : θ = Real.arccos ((R^2 + (R + 1/2)^2 - 1) / (2 * R * (R + 1/2)))) :
  1 < 2 * R * θ ∧ 2 * R * θ < π :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_bounds_l1365_136541


namespace NUMINAMATH_CALUDE_probability_of_selecting_specific_animals_l1365_136542

theorem probability_of_selecting_specific_animals :
  let total_animals : ℕ := 7
  let animals_to_select : ℕ := 2
  let specific_animals : ℕ := 2

  let total_combinations := Nat.choose total_animals animals_to_select
  let favorable_combinations := total_combinations - Nat.choose (total_animals - specific_animals) animals_to_select

  (favorable_combinations : ℚ) / total_combinations = 11 / 21 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_selecting_specific_animals_l1365_136542


namespace NUMINAMATH_CALUDE_coefficient_x4_sum_binomials_l1365_136516

theorem coefficient_x4_sum_binomials : 
  (Finset.sum (Finset.range 3) (fun i => Nat.choose (i + 5) 4)) = 55 := by sorry

end NUMINAMATH_CALUDE_coefficient_x4_sum_binomials_l1365_136516


namespace NUMINAMATH_CALUDE_special_number_theorem_l1365_136540

def is_smallest_nontrivial_divisor (a n : ℕ) : Prop :=
  a ≠ 1 ∧ a ∣ n ∧ ∀ d, 1 < d → d < a → ¬(d ∣ n)

theorem special_number_theorem :
  ∀ n : ℕ, n ≥ 2 →
  (∃ a b : ℕ, is_smallest_nontrivial_divisor a n ∧ b ∣ n ∧ n = a^2 + b^2) →
  (n = 8 ∨ n = 20) :=
sorry

end NUMINAMATH_CALUDE_special_number_theorem_l1365_136540


namespace NUMINAMATH_CALUDE_total_rainfall_is_23_l1365_136524

/-- Rainfall data for three days --/
structure RainfallData :=
  (monday_hours : ℕ)
  (monday_rate : ℕ)
  (tuesday_hours : ℕ)
  (tuesday_rate : ℕ)
  (wednesday_hours : ℕ)

/-- Calculate total rainfall for three days --/
def total_rainfall (data : RainfallData) : ℕ :=
  data.monday_hours * data.monday_rate +
  data.tuesday_hours * data.tuesday_rate +
  data.wednesday_hours * (2 * data.tuesday_rate)

/-- Theorem: The total rainfall for the given conditions is 23 inches --/
theorem total_rainfall_is_23 (data : RainfallData)
  (h1 : data.monday_hours = 7)
  (h2 : data.monday_rate = 1)
  (h3 : data.tuesday_hours = 4)
  (h4 : data.tuesday_rate = 2)
  (h5 : data.wednesday_hours = 2) :
  total_rainfall data = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_is_23_l1365_136524


namespace NUMINAMATH_CALUDE_additional_water_for_two_tanks_l1365_136539

/-- Calculates the additional water needed to fill two tanks with equal capacity -/
theorem additional_water_for_two_tanks
  (capacity : ℝ)  -- Capacity of each tank
  (filled1 : ℝ)   -- Amount of water in the first tank
  (filled2 : ℝ)   -- Amount of water in the second tank
  (h1 : filled1 = 300)  -- First tank has 300 liters
  (h2 : filled2 = 450)  -- Second tank has 450 liters
  (h3 : filled2 / capacity = 0.45)  -- Second tank is 45% filled
  : capacity - filled1 + capacity - filled2 = 1250 :=
by sorry

end NUMINAMATH_CALUDE_additional_water_for_two_tanks_l1365_136539


namespace NUMINAMATH_CALUDE_garden_width_l1365_136543

/-- Proves that the width of a rectangular garden with given conditions is 120 feet -/
theorem garden_width :
  ∀ (width : ℝ),
  (width > 0) →
  (220 * width > 0) →
  (220 * width / 2 > 0) →
  (220 * width / 2 * 2 / 3 > 0) →
  (220 * width / 2 * 2 / 3 = 8800) →
  (width = 120) := by
sorry

end NUMINAMATH_CALUDE_garden_width_l1365_136543


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1365_136564

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  (2 - x < 0) ∧ (-2 * x < 6)

-- Define the solution set
def solution_set : Set ℝ :=
  {x | x > 2}

-- Theorem statement
theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1365_136564


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1365_136510

theorem angle_measure_proof (x : ℝ) : 
  (x + (3 * x - 2) = 180) → x = 45.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1365_136510


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l1365_136517

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (5 : ℚ) / 8 ∧ 
  (∀ p' q' : ℕ+, (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (5 : ℚ) / 8 → q ≤ q') →
  q - p = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l1365_136517


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l1365_136503

theorem degree_to_radian_conversion (π : Real) (h : π * 1 = 180) :
  -885 * (π / 180) = -59 / 12 * π := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l1365_136503


namespace NUMINAMATH_CALUDE_min_value_of_y_l1365_136547

theorem min_value_of_y (x : ℝ) (h1 : x > 3) :
  let y := x + 1 / (x - 3)
  ∀ z, y ≥ z → z ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_y_l1365_136547


namespace NUMINAMATH_CALUDE_intersection_length_l1365_136555

/-- The length of the circular intersection between a sphere and a plane -/
theorem intersection_length (x y z : ℝ) : 
  x + y + z = 8 → 
  x * y + y * z + x * z = 14 → 
  (2 * Real.pi : ℝ) * (2 * Real.sqrt (11 / 3)) = 4 * Real.pi * Real.sqrt (11 / 3) := by
  sorry

#check intersection_length

end NUMINAMATH_CALUDE_intersection_length_l1365_136555


namespace NUMINAMATH_CALUDE_veranda_area_l1365_136514

/-- Calculates the area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) : 
  room_length = 18 ∧ room_width = 12 ∧ veranda_width = 2 →
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 136 := by
  sorry

end NUMINAMATH_CALUDE_veranda_area_l1365_136514


namespace NUMINAMATH_CALUDE_c_range_l1365_136572

-- Define the functions
def f (c : ℝ) (x : ℝ) := x^2 - 2*c*x + 1

-- State the theorem
theorem c_range (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) :
  (((∀ x y : ℝ, x < y → c^x > c^y) ∨
    (∀ x y : ℝ, x > y → x > 1/2 → y > 1/2 → f c x > f c y)) ∧
   ¬((∀ x y : ℝ, x < y → c^x > c^y) ∧
     (∀ x y : ℝ, x > y → x > 1/2 → y > 1/2 → f c x > f c y))) →
  (1/2 < c ∧ c < 1) :=
by sorry

end NUMINAMATH_CALUDE_c_range_l1365_136572


namespace NUMINAMATH_CALUDE_shekar_biology_score_l1365_136505

/-- Represents a student's scores in five subjects -/
structure StudentScores where
  mathematics : ℕ
  science : ℕ
  socialStudies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average score for a student -/
def averageScore (scores : StudentScores) : ℚ :=
  (scores.mathematics + scores.science + scores.socialStudies + scores.english + scores.biology) / 5

/-- Theorem: Given Shekar's scores in four subjects and the average, his Biology score must be 75 -/
theorem shekar_biology_score :
  ∀ (scores : StudentScores),
    scores.mathematics = 76 →
    scores.science = 65 →
    scores.socialStudies = 82 →
    scores.english = 67 →
    averageScore scores = 73 →
    scores.biology = 75 := by
  sorry

end NUMINAMATH_CALUDE_shekar_biology_score_l1365_136505


namespace NUMINAMATH_CALUDE_jerry_shelf_theorem_l1365_136585

/-- The difference between action figures and books on Jerry's shelf -/
def shelf_difference (books : ℕ) (initial_figures : ℕ) (added_figures : ℕ) : ℕ :=
  (initial_figures + added_figures) - books

/-- Theorem stating the difference between action figures and books on Jerry's shelf -/
theorem jerry_shelf_theorem :
  shelf_difference 3 4 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_theorem_l1365_136585


namespace NUMINAMATH_CALUDE_multiple_of_nine_three_l1365_136584

theorem multiple_of_nine_three (S : ℤ) : 
  (∀ x : ℤ, 9 ∣ x → 3 ∣ x) →  -- All multiples of 9 are multiples of 3
  (Odd S) →                   -- S is an odd number
  (9 ∣ S) →                   -- S is a multiple of 9
  (3 ∣ S) :=                  -- S is a multiple of 3
by sorry

end NUMINAMATH_CALUDE_multiple_of_nine_three_l1365_136584


namespace NUMINAMATH_CALUDE_bernardo_winning_number_l1365_136513

theorem bernardo_winning_number : ∃ N : ℕ, 
  (N ≤ 1999) ∧ 
  (8 * N + 600 < 2000) ∧ 
  (8 * N + 700 ≥ 2000) ∧ 
  (∀ M : ℕ, M < N → 
    (M ≤ 1999 → 8 * M + 700 < 2000) ∨ 
    (8 * M + 600 ≥ 2000)) := by
  sorry

#eval Nat.find bernardo_winning_number

end NUMINAMATH_CALUDE_bernardo_winning_number_l1365_136513


namespace NUMINAMATH_CALUDE_regular_polygon_with_108_degree_interior_angles_l1365_136529

theorem regular_polygon_with_108_degree_interior_angles (n : ℕ) : 
  (n ≥ 3) →  -- ensuring it's a valid polygon
  (((n - 2) * 180) / n = 108) →  -- interior angle formula
  (n = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_108_degree_interior_angles_l1365_136529


namespace NUMINAMATH_CALUDE_root_existence_l1365_136546

theorem root_existence : ∃ x : ℝ, 3 < x ∧ x < 4 ∧ Real.log x = 8 - 2 * x := by
  sorry

end NUMINAMATH_CALUDE_root_existence_l1365_136546


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l1365_136550

/-- Two vectors in ℝ² are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (x, 1)
  collinear a b → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l1365_136550


namespace NUMINAMATH_CALUDE_line_circle_intersection_l1365_136553

theorem line_circle_intersection (a b : ℝ) (h : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x/a + y/b = 1) :
  1/a^2 + 1/b^2 ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l1365_136553


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l1365_136562

theorem quadratic_roots_difference (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*a*x₁ + 5*a^2 - 6*a = 0 ∧ 
    x₂^2 - 4*a*x₂ + 5*a^2 - 6*a = 0 ∧
    |x₁ - x₂| = 6) → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l1365_136562


namespace NUMINAMATH_CALUDE_choir_size_proof_l1365_136577

theorem choir_size_proof (n : ℕ) : 
  (∃ (p : ℕ), p > 10 ∧ Prime p ∧ p ∣ n) ∧ 
  9 ∣ n ∧ 10 ∣ n ∧ 12 ∣ n →
  n ≥ 1980 :=
by sorry

end NUMINAMATH_CALUDE_choir_size_proof_l1365_136577


namespace NUMINAMATH_CALUDE_contrapositive_proof_l1365_136545

theorem contrapositive_proof : 
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔ 
  (∀ x : ℝ, x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_contrapositive_proof_l1365_136545


namespace NUMINAMATH_CALUDE_sqrt_real_iff_geq_two_l1365_136521

theorem sqrt_real_iff_geq_two (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_real_iff_geq_two_l1365_136521


namespace NUMINAMATH_CALUDE_problem_statement_l1365_136591

-- Define proposition p
def p : Prop := ∀ x : ℝ, (|x| = x ↔ x > 0)

-- Define proposition q
def q : Prop := (¬∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)

-- Theorem to prove
theorem problem_statement : ¬(p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1365_136591


namespace NUMINAMATH_CALUDE_fraction_order_l1365_136558

theorem fraction_order : (6/17)^2 < 8/25 ∧ 8/25 < 10/31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l1365_136558


namespace NUMINAMATH_CALUDE_david_squats_l1365_136551

/-- Fitness competition between David and Zachary -/
theorem david_squats (zachary_pushups zachary_crunches zachary_squats : ℕ) 
  (h1 : zachary_pushups = 68)
  (h2 : zachary_crunches = 130)
  (h3 : zachary_squats = 58) :
  ∃ (x : ℕ),
    x = zachary_squats ∧
    2 * zachary_pushups = zachary_pushups + x ∧
    zachary_crunches = (zachary_crunches - x / 2) + x / 2 ∧
    3 * x = 174 :=
by sorry

end NUMINAMATH_CALUDE_david_squats_l1365_136551


namespace NUMINAMATH_CALUDE_largest_divisor_power_l1365_136586

-- Define the expression A
def A : ℕ := 1990^(1991^1992) + 1991^(1990^1992) + 1992^(1991^1990)

-- State the theorem
theorem largest_divisor_power (k : ℕ) : (∀ m : ℕ, m > k → ¬(1991^m ∣ A)) ∧ (1991^k ∣ A) ↔ k = 1991 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_power_l1365_136586


namespace NUMINAMATH_CALUDE_one_non_prime_expression_l1365_136512

def expressions : List (ℕ → ℕ) := [
  (λ n => n^2 + (n+1)^2),
  (λ n => (n+1)^2 + (n+2)^2),
  (λ n => (n+2)^2 + (n+3)^2),
  (λ n => (n+3)^2 + (n+4)^2),
  (λ n => (n+4)^2 + (n+5)^2)
]

theorem one_non_prime_expression :
  (expressions.filter (λ f => ¬ Nat.Prime (f 1))).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_non_prime_expression_l1365_136512


namespace NUMINAMATH_CALUDE_money_problem_l1365_136592

theorem money_problem (a b : ℝ) 
  (h1 : 4 * a + 2 * b > 110)
  (h2 : 2 * a + 3 * b = 105) :
  a > 15 ∧ b < 25 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l1365_136592


namespace NUMINAMATH_CALUDE_matrix_product_equality_l1365_136579

theorem matrix_product_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A - B = A * B)
  (h2 : A * B = ![![7, -2], ![4, -3]]) :
  B * A = ![![6, -2], ![4, -4]] := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l1365_136579


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1365_136570

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ m : ℝ, m = 5 ∧ ∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → x + 1/x + y + 1/y ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1365_136570


namespace NUMINAMATH_CALUDE_business_investment_problem_l1365_136500

/-- Represents the investment and profit share of a business partner -/
structure Partner where
  investment : ℕ
  profitShare : ℕ

/-- Proves that given the conditions of the business problem, partner a's investment is 16000 -/
theorem business_investment_problem 
  (a b c : Partner)
  (h1 : b.profitShare = 1800)
  (h2 : a.profitShare - c.profitShare = 720)
  (h3 : b.investment = 10000)
  (h4 : c.investment = 12000)
  (h5 : a.profitShare * b.investment = b.profitShare * a.investment)
  (h6 : b.profitShare * c.investment = c.profitShare * b.investment)
  (h7 : a.profitShare * c.investment = c.profitShare * a.investment) :
  a.investment = 16000 := by
  sorry


end NUMINAMATH_CALUDE_business_investment_problem_l1365_136500


namespace NUMINAMATH_CALUDE_british_flag_theorem_expected_value_zero_l1365_136566

/-- A rectangle in a 2D plane -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: For any rectangle ABCD and any point P, AP^2 + CP^2 - BP^2 - DP^2 = 0 -/
theorem british_flag_theorem (rect : Rectangle) (P : ℝ × ℝ) :
  distanceSquared rect.A P + distanceSquared rect.C P
  = distanceSquared rect.B P + distanceSquared rect.D P := by
  sorry

/-- Corollary: The expected value of AP^2 + CP^2 - BP^2 - DP^2 is always 0 -/
theorem expected_value_zero (rect : Rectangle) :
  ∃ E : ℝ, E = 0 ∧ ∀ P : ℝ × ℝ,
    E = distanceSquared rect.A P + distanceSquared rect.C P
      - distanceSquared rect.B P - distanceSquared rect.D P := by
  sorry

end NUMINAMATH_CALUDE_british_flag_theorem_expected_value_zero_l1365_136566


namespace NUMINAMATH_CALUDE_basketballs_with_holes_l1365_136567

/-- Given the number of soccer balls and basketballs, the number of soccer balls with holes,
    and the total number of balls without holes, calculate the number of basketballs with holes. -/
theorem basketballs_with_holes
  (total_soccer : ℕ)
  (total_basketball : ℕ)
  (soccer_with_holes : ℕ)
  (total_without_holes : ℕ)
  (h1 : total_soccer = 40)
  (h2 : total_basketball = 15)
  (h3 : soccer_with_holes = 30)
  (h4 : total_without_holes = 18) :
  total_basketball - (total_without_holes - (total_soccer - soccer_with_holes)) = 7 := by
  sorry


end NUMINAMATH_CALUDE_basketballs_with_holes_l1365_136567


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_smallest_number_of_eggs_is_162_l1365_136568

theorem smallest_number_of_eggs : ℕ → Prop :=
  fun n =>
    (n > 150) ∧
    (∃ c : ℕ, c > 0 ∧ n = 15 * c - 3) ∧
    (∀ m : ℕ, m > 150 ∧ (∃ d : ℕ, d > 0 ∧ m = 15 * d - 3) → m ≥ n) →
    n = 162

theorem smallest_number_of_eggs_is_162 : smallest_number_of_eggs 162 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_smallest_number_of_eggs_is_162_l1365_136568


namespace NUMINAMATH_CALUDE_eulers_formula_l1365_136563

/-- A convex polyhedron is a structure with faces, vertices, and edges. -/
structure ConvexPolyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- Euler's formula for convex polyhedra states that V + F - E = 2 -/
theorem eulers_formula (P : ConvexPolyhedron) : 
  P.vertices + P.faces - P.edges = 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l1365_136563


namespace NUMINAMATH_CALUDE_equation_solutions_l1365_136532

theorem equation_solutions (a b : ℝ) (h : a + b = 0) :
  (∃! x : ℝ, a * x + b = 0) ∨ (∀ x : ℝ, a * x + b = 0) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1365_136532


namespace NUMINAMATH_CALUDE_students_not_playing_sports_l1365_136571

theorem students_not_playing_sports (total : ℕ) (soccer : ℕ) (volleyball : ℕ) (one_sport : ℕ) : 
  total = 40 → soccer = 20 → volleyball = 19 → one_sport = 15 → 
  ∃ (both : ℕ), 
    both = soccer + volleyball - one_sport ∧
    total - (soccer + volleyball - both) = 13 := by
  sorry

end NUMINAMATH_CALUDE_students_not_playing_sports_l1365_136571


namespace NUMINAMATH_CALUDE_normal_prob_equal_zero_l1365_136574

-- Define a normally distributed random variable
def normal_dist (μ σ : ℝ) : Type := ℝ

-- Define the probability density function for a normal distribution
noncomputable def pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

-- Define the probability of a continuous random variable being equal to a specific value
def prob_equal (X : Type) (a : ℝ) : ℝ := 0

-- Theorem statement
theorem normal_prob_equal_zero (μ σ : ℝ) (a : ℝ) :
  prob_equal (normal_dist μ σ) a = 0 :=
sorry

end NUMINAMATH_CALUDE_normal_prob_equal_zero_l1365_136574


namespace NUMINAMATH_CALUDE_total_amount_paid_l1365_136526

/-- Represents the purchase of a fruit with its quantity and price per kg -/
structure FruitPurchase where
  quantity : ℕ
  price_per_kg : ℕ

/-- Calculates the total cost of a fruit purchase -/
def total_cost (purchase : FruitPurchase) : ℕ :=
  purchase.quantity * purchase.price_per_kg

/-- Represents Tom's fruit shopping -/
def fruit_shopping : List FruitPurchase :=
  [
    { quantity := 8, price_per_kg := 70 },  -- Apples
    { quantity := 9, price_per_kg := 65 },  -- Mangoes
    { quantity := 5, price_per_kg := 50 },  -- Oranges
    { quantity := 3, price_per_kg := 30 }   -- Bananas
  ]

/-- Theorem: The total amount Tom paid for all fruits is $1485 -/
theorem total_amount_paid : (fruit_shopping.map total_cost).sum = 1485 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l1365_136526


namespace NUMINAMATH_CALUDE_closest_value_to_sqrt_difference_l1365_136536

theorem closest_value_to_sqrt_difference : 
  let diff := Real.sqrt 101 - Real.sqrt 99
  let candidates := [0.10, 0.12, 0.14, 0.16, 0.18]
  ∀ x ∈ candidates, x ≠ 0.10 → |diff - 0.10| < |diff - x| := by
  sorry

end NUMINAMATH_CALUDE_closest_value_to_sqrt_difference_l1365_136536


namespace NUMINAMATH_CALUDE_motorcycles_in_anytown_l1365_136569

/-- Given the ratio of vehicles in Anytown and the number of sedans, 
    prove the number of motorcycles. -/
theorem motorcycles_in_anytown 
  (truck_ratio : ℕ) 
  (sedan_ratio : ℕ) 
  (motorcycle_ratio : ℕ) 
  (num_sedans : ℕ) 
  (h1 : truck_ratio = 3)
  (h2 : sedan_ratio = 7)
  (h3 : motorcycle_ratio = 2)
  (h4 : num_sedans = 9100) : 
  (num_sedans / sedan_ratio) * motorcycle_ratio = 2600 := by
  sorry

end NUMINAMATH_CALUDE_motorcycles_in_anytown_l1365_136569


namespace NUMINAMATH_CALUDE_janet_initial_clips_l1365_136578

/-- The number of paper clips Janet had in the morning -/
def initial_clips : ℕ := sorry

/-- The number of paper clips Janet used during the day -/
def used_clips : ℕ := 59

/-- The number of paper clips Janet had left at the end of the day -/
def remaining_clips : ℕ := 26

/-- Theorem: Janet had 85 paper clips in the morning -/
theorem janet_initial_clips : initial_clips = 85 := by sorry

end NUMINAMATH_CALUDE_janet_initial_clips_l1365_136578


namespace NUMINAMATH_CALUDE_light_travel_distance_l1365_136501

/-- The distance light travels in one year (in miles) -/
def light_year_distance : ℝ := 6000000000000

/-- The number of years we're calculating for -/
def years : ℕ := 50

/-- The distance light travels in the given number of years -/
def total_distance : ℝ := light_year_distance * years

theorem light_travel_distance : total_distance = 3 * (10 ^ 14) := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l1365_136501


namespace NUMINAMATH_CALUDE_afternoon_rowing_count_l1365_136508

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 35

/-- The total number of campers who went rowing -/
def total_campers : ℕ := 62

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := total_campers - morning_campers

theorem afternoon_rowing_count : afternoon_campers = 27 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_rowing_count_l1365_136508


namespace NUMINAMATH_CALUDE_smallest_square_sum_20_consecutive_l1365_136533

/-- The sum of 20 consecutive positive integers starting from n -/
def sum_20_consecutive (n : ℕ) : ℕ := 10 * (2 * n + 19)

/-- A number is a perfect square -/
def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_square_sum_20_consecutive :
  (∃ n : ℕ, sum_20_consecutive n = 250) ∧
  (∀ m : ℕ, m < 250 → ¬∃ n : ℕ, sum_20_consecutive n = m ∧ is_perfect_square m) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_sum_20_consecutive_l1365_136533


namespace NUMINAMATH_CALUDE_toy_cost_l1365_136599

theorem toy_cost (price_A price_B price_C : ℝ) 
  (h1 : 2 * price_A + price_B + 3 * price_C = 24)
  (h2 : 3 * price_A + 4 * price_B + 2 * price_C = 36) :
  price_A + price_B + price_C = 12 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_l1365_136599


namespace NUMINAMATH_CALUDE_exponential_sum_l1365_136575

theorem exponential_sum (a x : ℝ) (ha : a > 0) 
  (h : a^(x/2) + a^(-x/2) = 5) : a^x + a^(-x) = 23 := by
  sorry

end NUMINAMATH_CALUDE_exponential_sum_l1365_136575


namespace NUMINAMATH_CALUDE_tan_22_5_deg_l1365_136535

theorem tan_22_5_deg (h1 : Real.pi / 4 = 2 * (22.5 * Real.pi / 180)) 
  (h2 : Real.tan (Real.pi / 4) = 1) :
  Real.tan (22.5 * Real.pi / 180) / (1 - Real.tan (22.5 * Real.pi / 180)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_l1365_136535


namespace NUMINAMATH_CALUDE_red_green_peaches_count_l1365_136523

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := 6

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 16

/-- The total number of red and green peaches in the basket -/
def total_red_green : ℕ := red_peaches + green_peaches

theorem red_green_peaches_count : total_red_green = 22 := by
  sorry

end NUMINAMATH_CALUDE_red_green_peaches_count_l1365_136523
