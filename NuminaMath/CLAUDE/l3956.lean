import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3956_395605

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + a - 1 < 0) ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3956_395605


namespace NUMINAMATH_CALUDE_sum_of_constants_l3956_395659

/-- Given a function y = a + b/x, prove that a + b = 11 under specific conditions -/
theorem sum_of_constants (a b : ℝ) : 
  (2 = a + b/(-2)) → 
  (6 = a + b/(-6)) → 
  (10 = a + b/(-3)) → 
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_constants_l3956_395659


namespace NUMINAMATH_CALUDE_senior_sports_solution_l3956_395658

def senior_sports_problem (total_seniors : ℕ) 
  (football : Finset ℕ) (baseball : Finset ℕ) (lacrosse : Finset ℕ) : Prop :=
  (total_seniors = 85) ∧
  (football.card = 74) ∧
  (baseball.card = 26) ∧
  ((football ∩ lacrosse).card = 17) ∧
  ((baseball ∩ football).card = 18) ∧
  ((baseball ∩ lacrosse).card = 13) ∧
  (lacrosse.card = 2 * (football ∩ baseball ∩ lacrosse).card) ∧
  (∀ s, s ∈ football ∪ baseball ∪ lacrosse) ∧
  ((football ∪ baseball ∪ lacrosse).card = total_seniors)

theorem senior_sports_solution 
  {total_seniors : ℕ} {football baseball lacrosse : Finset ℕ} 
  (h : senior_sports_problem total_seniors football baseball lacrosse) :
  (football ∩ baseball ∩ lacrosse).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_senior_sports_solution_l3956_395658


namespace NUMINAMATH_CALUDE_balloon_distribution_l3956_395601

theorem balloon_distribution (yellow_balloons : ℕ) (black_balloons_diff : ℕ) (balloons_per_school : ℕ) : 
  yellow_balloons = 3414 →
  black_balloons_diff = 1762 →
  balloons_per_school = 859 →
  (yellow_balloons + (yellow_balloons + black_balloons_diff)) / balloons_per_school = 10 := by
  sorry

end NUMINAMATH_CALUDE_balloon_distribution_l3956_395601


namespace NUMINAMATH_CALUDE_geometry_propositions_l3956_395621

-- Define the concept of vertical angles
def are_vertical_angles (α β : Real) : Prop := sorry

-- Define the concept of complementary angles
def are_complementary (α β : Real) : Prop := α + β = 90

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

theorem geometry_propositions :
  -- Proposition 1: Vertical angles are equal
  ∀ (α β : Real), are_vertical_angles α β → α = β
  
  -- Proposition 2: Complementary angles of equal angles are equal
  ∧ ∀ (α β γ δ : Real), α = β ∧ are_complementary α γ ∧ are_complementary β δ → γ = δ
  
  -- Proposition 3: If b is parallel to a and c is parallel to a, then b is parallel to c
  ∧ ∀ (a b c : Line), parallel b a ∧ parallel c a → parallel b c :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l3956_395621


namespace NUMINAMATH_CALUDE_unique_solution_condition_inequality_condition_l3956_395635

/-- The function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The function g(x) = a|x-1| -/
def g (a x : ℝ) : ℝ := a * |x - 1|

/-- If |f(x)| = g(x) has only one real solution, then a < 0 -/
theorem unique_solution_condition (a : ℝ) :
  (∃! x : ℝ, |f x| = g a x) → a < 0 := by sorry

/-- If f(x) ≥ g(x) for all x ∈ ℝ, then a ≤ -2 -/
theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) → a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_inequality_condition_l3956_395635


namespace NUMINAMATH_CALUDE_factorization_left_to_right_l3956_395627

theorem factorization_left_to_right : 
  ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_factorization_left_to_right_l3956_395627


namespace NUMINAMATH_CALUDE_odd_function_max_to_min_l3956_395602

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has a maximum value on [a, b] if there exists a point c in [a, b] 
    such that f(c) ≥ f(x) for all x in [a, b] -/
def HasMaxOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c

/-- A function f has a minimum value on [a, b] if there exists a point c in [a, b] 
    such that f(c) ≤ f(x) for all x in [a, b] -/
def HasMinOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f c ≤ f x

theorem odd_function_max_to_min (f : ℝ → ℝ) (a b : ℝ) (h1 : a < b) 
  (h2 : IsOdd f) (h3 : HasMaxOn f a b) : HasMinOn f (-b) (-a) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_max_to_min_l3956_395602


namespace NUMINAMATH_CALUDE_roses_distribution_l3956_395699

def total_money : ℕ := 300
def jenna_price : ℕ := 2
def imma_price : ℕ := 3
def ravi_price : ℕ := 4
def leila_price : ℕ := 5

def jenna_budget : ℕ := 100
def imma_budget : ℕ := 100
def ravi_budget : ℕ := 50
def leila_budget : ℕ := 50

def jenna_fraction : ℚ := 1/3
def imma_fraction : ℚ := 1/4
def ravi_fraction : ℚ := 1/6

theorem roses_distribution (jenna_roses imma_roses ravi_roses leila_roses : ℕ) :
  jenna_roses = ⌊(jenna_fraction * (jenna_budget / jenna_price : ℚ))⌋ ∧
  imma_roses = ⌊(imma_fraction * (imma_budget / imma_price : ℚ))⌋ ∧
  ravi_roses = ⌊(ravi_fraction * (ravi_budget / ravi_price : ℚ))⌋ ∧
  leila_roses = leila_budget / leila_price →
  jenna_roses + imma_roses + ravi_roses + leila_roses = 36 := by
  sorry

end NUMINAMATH_CALUDE_roses_distribution_l3956_395699


namespace NUMINAMATH_CALUDE_renu_work_time_l3956_395641

/-- The number of days it takes Renu to complete the work alone -/
def renu_days : ℝ := 8

/-- The number of days it takes Suma to complete the work alone -/
def suma_days : ℝ := 4.8

/-- The number of days it takes Renu and Suma to complete the work together -/
def combined_days : ℝ := 3

theorem renu_work_time :
  (1 / renu_days) + (1 / suma_days) = (1 / combined_days) :=
sorry

end NUMINAMATH_CALUDE_renu_work_time_l3956_395641


namespace NUMINAMATH_CALUDE_total_volume_of_cubes_l3956_395660

def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

def carl_cubes : ℕ := 6
def carl_side_length : ℝ := 1

def kate_cubes : ℕ := 4
def kate_side_length : ℝ := 3

theorem total_volume_of_cubes :
  (carl_cubes : ℝ) * cube_volume carl_side_length +
  (kate_cubes : ℝ) * cube_volume kate_side_length = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_cubes_l3956_395660


namespace NUMINAMATH_CALUDE_faster_speed_problem_l3956_395665

/-- Proves that the faster speed is 15 km/hr given the conditions of the problem -/
theorem faster_speed_problem (actual_distance : ℝ) (original_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 10)
  (h2 : original_speed = 5)
  (h3 : additional_distance = 20) :
  let time := actual_distance / original_speed
  let faster_speed := (actual_distance + additional_distance) / time
  faster_speed = 15 := by sorry

end NUMINAMATH_CALUDE_faster_speed_problem_l3956_395665


namespace NUMINAMATH_CALUDE_parabola_y_relationship_l3956_395622

theorem parabola_y_relationship (x₁ x₂ y₁ y₂ : ℝ) : 
  (y₁ = x₁^2 - 3) →  -- Point A lies on the parabola
  (y₂ = x₂^2 - 3) →  -- Point B lies on the parabola
  (0 < x₁) →         -- x₁ is positive
  (x₁ < x₂) →        -- x₁ is less than x₂
  y₁ < y₂ :=         -- Conclusion: y₁ is less than y₂
by sorry

end NUMINAMATH_CALUDE_parabola_y_relationship_l3956_395622


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3956_395681

theorem fraction_to_decimal : (3 : ℚ) / 24 = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3956_395681


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l3956_395624

theorem largest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > n → ¬((m + 12) ∣ (m^3 + 160))) ∧ 
  ((n + 12) ∣ (n^3 + 160)) ∧ n = 1748 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l3956_395624


namespace NUMINAMATH_CALUDE_simultaneous_work_time_l3956_395684

/-- The time taken for two workers to fill a truck when working simultaneously -/
theorem simultaneous_work_time (rate1 rate2 : ℚ) (h1 : rate1 = 1 / 6) (h2 : rate2 = 1 / 8) :
  1 / (rate1 + rate2) = 24 / 7 := by sorry

end NUMINAMATH_CALUDE_simultaneous_work_time_l3956_395684


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3956_395686

theorem quadratic_equation_roots (x : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (2 * x₁^2 - 3 * x₁ - (3/2) = 0) ∧ (2 * x₂^2 - 3 * x₂ - (3/2) = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3956_395686


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3956_395648

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def is_mersenne_prime (m : Nat) : Prop :=
  ∃ n : Nat, is_prime n ∧ m = 2^n - 1 ∧ is_prime m

theorem largest_mersenne_prime_under_500 :
  ∀ m : Nat, is_mersenne_prime m → m < 500 → m ≤ 127 :=
by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3956_395648


namespace NUMINAMATH_CALUDE_number_divisibility_problem_l3956_395671

theorem number_divisibility_problem : 
  ∃ x : ℚ, (x / 3) * 12 = 9 ∧ x = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_number_divisibility_problem_l3956_395671


namespace NUMINAMATH_CALUDE_line_through_points_sum_of_slope_and_intercept_l3956_395618

/-- Given a line passing through points (1,2) and (4,11) with equation y = mx + b, prove that m + b = 2 -/
theorem line_through_points_sum_of_slope_and_intercept :
  ∀ (m b : ℝ),
  (2 : ℝ) = m * 1 + b →  -- Point (1,2) satisfies the equation
  (11 : ℝ) = m * 4 + b →  -- Point (4,11) satisfies the equation
  m + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_sum_of_slope_and_intercept_l3956_395618


namespace NUMINAMATH_CALUDE_binomial_plus_four_l3956_395680

theorem binomial_plus_four : (Nat.choose 18 17) + 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_binomial_plus_four_l3956_395680


namespace NUMINAMATH_CALUDE_pufferfish_count_swordfish_to_pufferfish_ratio_total_fish_count_l3956_395695

/-- The number of pufferfish in an aquarium exhibit -/
def num_pufferfish : ℕ := 15

/-- The number of swordfish in the aquarium exhibit -/
def num_swordfish : ℕ := 5 * num_pufferfish

/-- The total number of fish in the aquarium exhibit -/
def total_fish : ℕ := 90

/-- Theorem stating that the number of pufferfish is 15 -/
theorem pufferfish_count : num_pufferfish = 15 := by sorry

/-- Theorem stating that the number of swordfish is five times the number of pufferfish -/
theorem swordfish_to_pufferfish_ratio : num_swordfish = 5 * num_pufferfish := by sorry

/-- Theorem stating that the total number of fish is 90 -/
theorem total_fish_count : total_fish = num_swordfish + num_pufferfish := by sorry

end NUMINAMATH_CALUDE_pufferfish_count_swordfish_to_pufferfish_ratio_total_fish_count_l3956_395695


namespace NUMINAMATH_CALUDE_proposition_q_undetermined_l3956_395674

theorem proposition_q_undetermined (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (q ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_proposition_q_undetermined_l3956_395674


namespace NUMINAMATH_CALUDE_cylinder_radius_l3956_395690

theorem cylinder_radius (length width : Real) (h1 : length = 3 * Real.pi) (h2 : width = Real.pi) :
  ∃ (r : Real), (r = 3/2 ∨ r = 1/2) ∧ 
  (2 * Real.pi * r = length ∨ 2 * Real.pi * r = width) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_l3956_395690


namespace NUMINAMATH_CALUDE_book_reading_competition_l3956_395600

/-- Represents the number of pages read by each girl -/
structure PageCount where
  ivana : ℕ
  majka : ℕ
  lucka : ℕ
  sasa : ℕ
  zuzka : ℕ

/-- Checks if all values in the PageCount are distinct -/
def allDistinct (p : PageCount) : Prop :=
  p.ivana ≠ p.majka ∧ p.ivana ≠ p.lucka ∧ p.ivana ≠ p.sasa ∧ p.ivana ≠ p.zuzka ∧
  p.majka ≠ p.lucka ∧ p.majka ≠ p.sasa ∧ p.majka ≠ p.zuzka ∧
  p.lucka ≠ p.sasa ∧ p.lucka ≠ p.zuzka ∧
  p.sasa ≠ p.zuzka

/-- The theorem representing the book reading competition -/
theorem book_reading_competition :
  ∃! (p : PageCount),
    p.lucka = 32 ∧
    p.lucka = (p.sasa + p.zuzka) / 2 ∧
    p.ivana = p.zuzka + 5 ∧
    p.majka = p.sasa - 8 ∧
    allDistinct p ∧
    (∀ x ∈ [p.ivana, p.majka, p.lucka, p.sasa, p.zuzka], x ≥ 27) ∧
    p.ivana = 34 ∧ p.majka = 27 ∧ p.lucka = 32 ∧ p.sasa = 35 ∧ p.zuzka = 29 :=
by sorry

end NUMINAMATH_CALUDE_book_reading_competition_l3956_395600


namespace NUMINAMATH_CALUDE_simplify_expression_l3956_395632

theorem simplify_expression :
  (Real.sqrt 10 + Real.sqrt 15) / (Real.sqrt 3 + Real.sqrt 5 - Real.sqrt 2) =
  (2 * Real.sqrt 30 + 5 * Real.sqrt 2 + 11 * Real.sqrt 5 + 5 * Real.sqrt 3) / 6 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3956_395632


namespace NUMINAMATH_CALUDE_problem_statement_l3956_395619

theorem problem_statement (a b : ℝ) (h : |a - 3| + (b + 2)^2 = 0) :
  (a + b)^2023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3956_395619


namespace NUMINAMATH_CALUDE_pokemon_cards_total_l3956_395696

theorem pokemon_cards_total (jenny : ℕ) (orlando : ℕ) (richard : ℕ) : 
  jenny = 6 →
  orlando = jenny + 2 →
  richard = 3 * orlando →
  jenny + orlando + richard = 38 := by
sorry

end NUMINAMATH_CALUDE_pokemon_cards_total_l3956_395696


namespace NUMINAMATH_CALUDE_new_boarders_count_l3956_395628

theorem new_boarders_count (initial_boarders : ℕ) (initial_ratio_boarders initial_ratio_day_scholars : ℕ) 
  (final_ratio_boarders final_ratio_day_scholars : ℕ) :
  initial_boarders = 560 →
  initial_ratio_boarders = 7 →
  initial_ratio_day_scholars = 16 →
  final_ratio_boarders = 1 →
  final_ratio_day_scholars = 2 →
  ∃ (new_boarders : ℕ),
    (initial_boarders + new_boarders) * final_ratio_day_scholars = 
    (initial_boarders * initial_ratio_day_scholars / initial_ratio_boarders) * final_ratio_boarders ∧
    new_boarders = 80 :=
by sorry

end NUMINAMATH_CALUDE_new_boarders_count_l3956_395628


namespace NUMINAMATH_CALUDE_revenue_change_l3956_395675

theorem revenue_change 
  (original_tax : ℝ) 
  (original_consumption : ℝ) 
  (tax_reduction_rate : ℝ) 
  (consumption_increase_rate : ℝ) 
  (h1 : tax_reduction_rate = 0.19) 
  (h2 : consumption_increase_rate = 0.15) : 
  let new_tax := original_tax * (1 - tax_reduction_rate)
  let new_consumption := original_consumption * (1 + consumption_increase_rate)
  let original_revenue := original_tax * original_consumption
  let new_revenue := new_tax * new_consumption
  (new_revenue - original_revenue) / original_revenue = -0.0685 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l3956_395675


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3956_395697

theorem quadratic_equation_solution : ∃ x1 x2 : ℝ, 
  x1 = 95 ∧ 
  x2 = -105 ∧ 
  x1^2 + 10*x1 - 9975 = 0 ∧ 
  x2^2 + 10*x2 - 9975 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3956_395697


namespace NUMINAMATH_CALUDE_sqrt_33_between_5_and_6_l3956_395638

theorem sqrt_33_between_5_and_6 : 5 < Real.sqrt 33 ∧ Real.sqrt 33 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_33_between_5_and_6_l3956_395638


namespace NUMINAMATH_CALUDE_livestock_theorem_l3956_395615

/-- Represents the value of livestock in taels of silver -/
structure LivestockValue where
  cow : ℕ
  sheep : ℕ
  total : ℕ

/-- Represents a purchase of livestock -/
structure Purchase where
  cows : ℕ
  sheep : ℕ

/-- The main theorem about livestock values and purchases -/
theorem livestock_theorem 
  (eq1 : LivestockValue)
  (eq2 : LivestockValue)
  (h1 : eq1.cow = 5 ∧ eq1.sheep = 2 ∧ eq1.total = 19)
  (h2 : eq2.cow = 2 ∧ eq2.sheep = 5 ∧ eq2.total = 16) :
  (∃ (cow_value sheep_value : ℕ),
    cow_value = 3 ∧ sheep_value = 2 ∧
    eq1.cow * cow_value + eq1.sheep * sheep_value = eq1.total ∧
    eq2.cow * cow_value + eq2.sheep * sheep_value = eq2.total) ∧
  (∃ (purchases : List Purchase),
    purchases.length = 3 ∧
    purchases.all (λ p => p.cows > 0 ∧ p.sheep > 0 ∧ p.cows * 3 + p.sheep * 2 = 20) ∧
    ∀ p : Purchase, p.cows > 0 → p.sheep > 0 → p.cows * 3 + p.sheep * 2 = 20 → p ∈ purchases) :=
by sorry

end NUMINAMATH_CALUDE_livestock_theorem_l3956_395615


namespace NUMINAMATH_CALUDE_no_safe_numbers_l3956_395617

def is_p_safe (n p : ℕ+) : Prop :=
  ∀ k : ℤ, |n.val - k * p.val| > 3

theorem no_safe_numbers : 
  ¬ ∃ n : ℕ+, n.val ≤ 15000 ∧ 
    is_p_safe n 5 ∧ 
    is_p_safe n 7 ∧ 
    is_p_safe n 11 :=
sorry

end NUMINAMATH_CALUDE_no_safe_numbers_l3956_395617


namespace NUMINAMATH_CALUDE_equal_differences_l3956_395683

theorem equal_differences (x : Fin 102 → ℕ) 
  (h_increasing : ∀ i j : Fin 102, i < j → x i < x j)
  (h_upper_bound : ∀ i : Fin 102, x i < 255) :
  ∃ (S : Finset (Fin 101)) (d : ℕ), 
    S.card ≥ 26 ∧ ∀ i ∈ S, x (i + 1) - x i = d := by
  sorry

end NUMINAMATH_CALUDE_equal_differences_l3956_395683


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3956_395651

-- Define the quadratic expression
def quadratic (x : ℝ) : ℝ := 6 * x^2 - 12 * x + 4

-- Define the transformed expression
def transformed (x a h k : ℝ) : ℝ := a * (x - h)^2 + k

-- Theorem statement
theorem quadratic_transformation :
  ∃ (a h k : ℝ), (∀ x, quadratic x = transformed x a h k) ∧ (a + h + k = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3956_395651


namespace NUMINAMATH_CALUDE_complement_of_B_relative_to_A_l3956_395640

def A : Set Int := {-1, 0, 1, 2, 3}
def B : Set Int := {-1, 1}

theorem complement_of_B_relative_to_A :
  {x : Int | x ∈ A ∧ x ∉ B} = {0, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_relative_to_A_l3956_395640


namespace NUMINAMATH_CALUDE_inequality_solution_and_function_property_l3956_395669

def f (x : ℝ) : ℝ := |x - 1|

theorem inequality_solution_and_function_property :
  (∃ (S : Set ℝ), S = {x : ℝ | x ≤ -2 ∨ x ≥ 4/3} ∧
    ∀ x : ℝ, x ∈ S ↔ f (2*x) + f (x+4) ≥ 6) ∧
  (∀ a b : ℝ, |a| < 1 → |b| < 1 → f (a*b) > f (a-b+1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_function_property_l3956_395669


namespace NUMINAMATH_CALUDE_largest_sum_is_three_fourths_l3956_395609

theorem largest_sum_is_three_fourths : 
  let sums : List ℚ := [1/4 + 1/9, 1/4 + 1/10, 1/4 + 1/2, 1/4 + 1/12, 1/4 + 1/11]
  (∀ s ∈ sums, s ≤ 3/4) ∧ (3/4 ∈ sums) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_is_three_fourths_l3956_395609


namespace NUMINAMATH_CALUDE_orchid_rose_difference_l3956_395667

/-- Given the initial and final counts of roses and orchids in a vase,
    prove that there are 10 more orchids than roses after adding new flowers. -/
theorem orchid_rose_difference (initial_roses initial_orchids final_roses final_orchids : ℕ) : 
  initial_roses = 9 →
  initial_orchids = 6 →
  final_roses = 3 →
  final_orchids = 13 →
  final_orchids - final_roses = 10 := by
  sorry

end NUMINAMATH_CALUDE_orchid_rose_difference_l3956_395667


namespace NUMINAMATH_CALUDE_ram_ravi_selection_probability_l3956_395629

theorem ram_ravi_selection_probability 
  (p_ram : ℝ) 
  (p_both : ℝ) 
  (h1 : p_ram = 1/7)
  (h2 : p_both = 0.02857142857142857) :
  ∃ (p_ravi : ℝ), p_ravi = 0.2 ∧ p_both = p_ram * p_ravi :=
by sorry

end NUMINAMATH_CALUDE_ram_ravi_selection_probability_l3956_395629


namespace NUMINAMATH_CALUDE_infinite_triangles_with_side_ten_l3956_395692

/-- A function that checks if three positive integers can form a triangle -/
def can_form_triangle (a b c : ℕ+) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating that there are infinitely many triangles with sides x, y, and 10 -/
theorem infinite_triangles_with_side_ten :
  ∀ n : ℕ, ∃ x y : ℕ+, x > n ∧ y > n ∧ can_form_triangle x y 10 :=
sorry

end NUMINAMATH_CALUDE_infinite_triangles_with_side_ten_l3956_395692


namespace NUMINAMATH_CALUDE_max_value_implies_m_l3956_395679

noncomputable def f (x m : ℝ) : ℝ :=
  2 * (Real.cos x) * (Real.cos x) + Real.sqrt 3 * Real.sin (2 * x) + m

theorem max_value_implies_m (h : ∀ x ∈ Set.Icc 0 (Real.pi / 6), f x 1 ≤ 4) :
  ∃ x ∈ Set.Icc 0 (Real.pi / 6), f x 1 = 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l3956_395679


namespace NUMINAMATH_CALUDE_expression_value_l3956_395642

theorem expression_value : (19 + 43 / 151) * 151 = 2910 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3956_395642


namespace NUMINAMATH_CALUDE_triangle_sine_relations_l3956_395698

theorem triangle_sine_relations (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧
  b = 7 * a * Real.sin B →
  Real.sin A = 1/7 ∧ 
  (B = π/3 → Real.sin C = 13/14) := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_relations_l3956_395698


namespace NUMINAMATH_CALUDE_imaginary_part_z2_l3956_395643

theorem imaginary_part_z2 (z₁ z₂ : ℂ) : 
  z₁ = 2 - 3*I → z₁ * z₂ = 1 + 2*I → z₂.im = 7/13 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_z2_l3956_395643


namespace NUMINAMATH_CALUDE_no_common_solution_l3956_395614

theorem no_common_solution : ¬ ∃ (x y : ℝ), (x^2 - 6*x + y + 9 = 0) ∧ (x^2 + 4*y + 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l3956_395614


namespace NUMINAMATH_CALUDE_distance_is_60km_l3956_395647

/-- A ship's journey relative to a lighthouse -/
structure ShipJourney where
  speed : ℝ
  time : ℝ
  initial_angle : ℝ
  final_angle : ℝ

/-- Calculate the distance between the ship and the lighthouse at the end of the journey -/
def distance_to_lighthouse (journey : ShipJourney) : ℝ :=
  sorry

/-- Theorem stating that for the given journey parameters, the distance to the lighthouse is 60 km -/
theorem distance_is_60km (journey : ShipJourney) 
  (h1 : journey.speed = 15)
  (h2 : journey.time = 4)
  (h3 : journey.initial_angle = π / 3)
  (h4 : journey.final_angle = π / 6) :
  distance_to_lighthouse journey = 60 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_60km_l3956_395647


namespace NUMINAMATH_CALUDE_similar_triangles_ab_length_l3956_395685

/-- Two triangles are similar -/
def similar_triangles (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : Prop := sorry

theorem similar_triangles_ab_length :
  ∀ (P Q R X Y Z A B C : ℝ × ℝ),
  let pqr : Set (Fin 3 → ℝ × ℝ) := {![P, Q, R]}
  let xyz : Set (Fin 3 → ℝ × ℝ) := {![X, Y, Z]}
  let abc : Set (Fin 3 → ℝ × ℝ) := {![A, B, C]}
  similar_triangles pqr xyz →
  similar_triangles xyz abc →
  dist P Q = 8 →
  dist Q R = 16 →
  dist B C = 24 →
  dist Y Z = 12 →
  dist A B = 12 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_ab_length_l3956_395685


namespace NUMINAMATH_CALUDE_trigonometric_equality_l3956_395649

theorem trigonometric_equality : 
  Real.sqrt (1 - 2 * Real.cos (π / 2 + 3) * Real.sin (π / 2 - 3)) = Real.sin 3 + Real.cos 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l3956_395649


namespace NUMINAMATH_CALUDE_june_decrease_percentage_l3956_395636

-- Define the price changes for each month
def january_change : ℝ := 0.15
def february_change : ℝ := -0.10
def march_change : ℝ := 0.20
def april_change : ℝ := -0.30
def may_change : ℝ := 0.10

-- Function to calculate the price after a change
def apply_change (price : ℝ) (change : ℝ) : ℝ :=
  price * (1 + change)

-- Theorem stating the required decrease in June
theorem june_decrease_percentage (initial_price : ℝ) (initial_price_pos : initial_price > 0) :
  let final_price := apply_change (apply_change (apply_change (apply_change (apply_change initial_price january_change) february_change) march_change) april_change) may_change
  ∃ (june_decrease : ℝ), 
    (apply_change final_price june_decrease = initial_price) ∧ 
    (abs (june_decrease + 0.0456) < 0.0001) := by
  sorry

end NUMINAMATH_CALUDE_june_decrease_percentage_l3956_395636


namespace NUMINAMATH_CALUDE_second_group_count_l3956_395661

theorem second_group_count (total : ℕ) (avg : ℚ) (sum_three : ℕ) (avg_others : ℚ) :
  total = 5 ∧ 
  avg = 20 ∧ 
  sum_three = 48 ∧ 
  avg_others = 26 →
  (total - 3 : ℕ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_second_group_count_l3956_395661


namespace NUMINAMATH_CALUDE_original_expenditure_problem_l3956_395662

/-- The original expenditure problem -/
theorem original_expenditure_problem :
  ∀ (E A : ℕ),
  -- Initial conditions
  E = 35 * A →
  -- After first admission
  E + 84 = 42 * (A - 1) →
  -- After second change
  E + 124 = 37 * (A + 1) →
  -- Conclusion
  E = 630 := by
  sorry

end NUMINAMATH_CALUDE_original_expenditure_problem_l3956_395662


namespace NUMINAMATH_CALUDE_sum_congruence_mod_nine_l3956_395655

theorem sum_congruence_mod_nine :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_mod_nine_l3956_395655


namespace NUMINAMATH_CALUDE_no_perfect_square_300_ones_l3956_395607

/-- Represents the count of digits '1' in the decimal representation of a number -/
def count_ones (n : ℕ) : ℕ := sorry

/-- Checks if a number's decimal representation contains only '0' and '1' -/
def only_zero_and_one (n : ℕ) : Prop := sorry

/-- Theorem: There does not exist a perfect square integer with exactly 300 digits of '1' 
    and no other digits except '0' in its decimal representation -/
theorem no_perfect_square_300_ones : 
  ¬ ∃ (n : ℕ), count_ones n = 300 ∧ only_zero_and_one n ∧ ∃ (k : ℕ), n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_300_ones_l3956_395607


namespace NUMINAMATH_CALUDE_polynomial_characterization_l3956_395694

-- Define the polynomial type
def RealPolynomial := ℝ → ℝ

-- Define the condition for a, b, c
def SumProductZero (a b c : ℝ) : Prop := a * b + b * c + c * a = 0

-- Define the equality condition for the polynomial
def PolynomialCondition (P : RealPolynomial) : Prop :=
  ∀ (a b c : ℝ), SumProductZero a b c →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

-- Define the form of the polynomial we want to prove
def QuarticQuadraticForm (P : RealPolynomial) : Prop :=
  ∃ (α β : ℝ), ∀ (x : ℝ), P x = α * x^4 + β * x^2

-- The main theorem
theorem polynomial_characterization (P : RealPolynomial) :
  PolynomialCondition P → QuarticQuadraticForm P :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l3956_395694


namespace NUMINAMATH_CALUDE_eqLength_is_53_l3956_395637

/-- Represents a trapezoid with a circle inscribed in it. -/
structure InscribedTrapezoid where
  /-- Length of side EF -/
  ef : ℝ
  /-- Length of side FG -/
  fg : ℝ
  /-- Length of side GH -/
  gh : ℝ
  /-- Length of side HE -/
  he : ℝ
  /-- EF is parallel to GH -/
  parallel : ef > gh

/-- The length of EQ in the inscribed trapezoid. -/
def eqLength (t : InscribedTrapezoid) : ℝ := sorry

/-- Theorem stating that for the given trapezoid, EQ = 53 -/
theorem eqLength_is_53 (t : InscribedTrapezoid) 
  (h1 : t.ef = 84) 
  (h2 : t.fg = 58) 
  (h3 : t.gh = 27) 
  (h4 : t.he = 64) : 
  eqLength t = 53 := by sorry

end NUMINAMATH_CALUDE_eqLength_is_53_l3956_395637


namespace NUMINAMATH_CALUDE_airplane_passenger_ratio_l3956_395664

/-- Given an airplane with 80 passengers, of which 30 are men, prove that the ratio of men to women is 3:5. -/
theorem airplane_passenger_ratio :
  let total_passengers : ℕ := 80
  let num_men : ℕ := 30
  let num_women : ℕ := total_passengers - num_men
  (num_men : ℚ) / (num_women : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_airplane_passenger_ratio_l3956_395664


namespace NUMINAMATH_CALUDE_original_number_proof_l3956_395688

theorem original_number_proof (x : ℚ) :
  1 + (1 / x) = 5 / 2 → x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3956_395688


namespace NUMINAMATH_CALUDE_complex_number_sum_l3956_395673

theorem complex_number_sum (z : ℂ) : z = (2 + Complex.I) / (1 - 2 * Complex.I) → 
  ∃ (a b : ℝ), z = a + b * Complex.I ∧ a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_sum_l3956_395673


namespace NUMINAMATH_CALUDE_relationship_between_A_B_C_l3956_395639

-- Define the variables and functions
variable (a : ℝ)
def A : ℝ := 2 * a - 7
def B : ℝ := a^2 - 4 * a + 3
def C : ℝ := a^2 + 6 * a - 28

-- Theorem statement
theorem relationship_between_A_B_C (h : a > 2) :
  (B a - A a > 0) ∧
  (∀ x, 2 < x ∧ x < 3 → C x - A x < 0) ∧
  (C 3 - A 3 = 0) ∧
  (∀ y, y > 3 → C y - A y > 0) := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_A_B_C_l3956_395639


namespace NUMINAMATH_CALUDE_flower_bee_difference_l3956_395633

def number_of_flowers : ℕ := 5
def number_of_bees : ℕ := 3

theorem flower_bee_difference : 
  number_of_flowers - number_of_bees = 2 := by sorry

end NUMINAMATH_CALUDE_flower_bee_difference_l3956_395633


namespace NUMINAMATH_CALUDE_room_space_is_400_l3956_395626

/-- The total space of a room with bookshelves and reserved space for desk and walking -/
def room_space (num_shelves : ℕ) (shelf_space desk_space : ℝ) : ℝ :=
  num_shelves * shelf_space + desk_space

/-- Theorem: The room space is 400 square feet -/
theorem room_space_is_400 :
  room_space 3 80 160 = 400 :=
by sorry

end NUMINAMATH_CALUDE_room_space_is_400_l3956_395626


namespace NUMINAMATH_CALUDE_solution_set_when_a_neg_three_a_range_when_subset_condition_l3956_395691

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x + a| + |x - 2|

-- Theorem 1
theorem solution_set_when_a_neg_three :
  {x : ℝ | f x (-3) ≥ 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

-- Theorem 2
theorem a_range_when_subset_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f x a ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_neg_three_a_range_when_subset_condition_l3956_395691


namespace NUMINAMATH_CALUDE_inequality_proof_l3956_395689

theorem inequality_proof (a b c : ℝ) : 
  a = 4/5 → b = Real.sin (2/3) → c = Real.cos (1/3) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3956_395689


namespace NUMINAMATH_CALUDE_average_of_p_and_q_l3956_395676

theorem average_of_p_and_q (p q : ℝ) (h : (5 / 4) * (p + q) = 15) : (p + q) / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_of_p_and_q_l3956_395676


namespace NUMINAMATH_CALUDE_stating_tree_structure_equation_l3956_395652

/-- Represents a tree structure with a trunk, branches, and small branches. -/
structure TreeStructure where
  x : ℕ  -- number of branches grown by the trunk
  total : ℕ  -- total count of trunk, branches, and small branches
  h_total : total = x^2 + x + 1  -- relation between x and total

/-- 
Theorem stating that for a tree structure with 43 total elements,
the equation x^2 + x + 1 = 43 correctly represents the structure.
-/
theorem tree_structure_equation (t : TreeStructure) (h : t.total = 43) :
  t.x^2 + t.x + 1 = 43 := by
  sorry

end NUMINAMATH_CALUDE_stating_tree_structure_equation_l3956_395652


namespace NUMINAMATH_CALUDE_total_marbles_l3956_395630

/-- Given a bag of marbles with only red, blue, and yellow marbles, where the ratio of
    red:blue:yellow is 2:3:4, and there are 24 blue marbles, prove that the total number
    of marbles is 72. -/
theorem total_marbles (red blue yellow total : ℕ) : 
  red + blue + yellow = total →
  red = 2 * n ∧ blue = 3 * n ∧ yellow = 4 * n →
  blue = 24 →
  total = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l3956_395630


namespace NUMINAMATH_CALUDE_tangent_line_circle_minimum_l3956_395623

theorem tangent_line_circle_minimum (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, x + y + a = 0 ∧ (x - b)^2 + (y - 1)^2 = 2 ∧ 
    ∀ x' y' : ℝ, (x' - b)^2 + (y' - 1)^2 ≤ 2 → (x' + y' + a)^2 > 0) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x y : ℝ, x + y + a' = 0 ∧ (x - b')^2 + (y - 1)^2 = 2 ∧ 
      ∀ x' y' : ℝ, (x' - b')^2 + (y' - 1)^2 ≤ 2 → (x' + y' + a')^2 > 0) → 
    (3 - 2*b')^2 / (2*a') ≥ (3 - 2*b)^2 / (2*a)) →
  (3 - 2*b)^2 / (2*a) = 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_circle_minimum_l3956_395623


namespace NUMINAMATH_CALUDE_hcf_of_36_and_84_l3956_395653

theorem hcf_of_36_and_84 : Nat.gcd 36 84 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_36_and_84_l3956_395653


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l3956_395654

theorem complex_modulus_equality (n : ℝ) :
  n > 0 → Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 13 → n = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l3956_395654


namespace NUMINAMATH_CALUDE_shaded_sectors_ratio_l3956_395670

/-- Given three semicircular protractors with radii 1, 3, and 5,
    whose centers coincide and diameters align,
    prove that the ratio of the areas of the shaded sectors is 48 : 40 : 15 -/
theorem shaded_sectors_ratio (r₁ r₂ r₃ : ℝ) (S_A S_B S_C : ℝ) :
  r₁ = 1 → r₂ = 3 → r₃ = 5 →
  S_A = (π / 10) * (r₃^2 - r₂^2) →
  S_B = (π / 6) * (r₂^2 - r₁^2) →
  S_C = (π / 2) * r₁^2 →
  ∃ (k : ℝ), k > 0 ∧ S_A = 48 * k ∧ S_B = 40 * k ∧ S_C = 15 * k :=
by sorry

end NUMINAMATH_CALUDE_shaded_sectors_ratio_l3956_395670


namespace NUMINAMATH_CALUDE_calculate_expression_l3956_395631

theorem calculate_expression : (42 / (3^2 + 2 * 3 - 1)) * 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3956_395631


namespace NUMINAMATH_CALUDE_max_value_sum_and_reciprocal_l3956_395625

theorem max_value_sum_and_reciprocal (nums : Finset ℝ) (x : ℝ) :
  (Finset.card nums = 11) →
  (∀ y ∈ nums, y > 0) →
  (x ∈ nums) →
  (Finset.sum nums id = 102) →
  (Finset.sum nums (λ y => 1 / y) = 102) →
  (x + 1 / x ≤ 10304 / 102) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_and_reciprocal_l3956_395625


namespace NUMINAMATH_CALUDE_james_earnings_l3956_395613

/-- Represents the amount of water collected per inch of rain -/
def water_per_inch : ℝ := 15

/-- Represents the amount of rain on Monday in inches -/
def monday_rain : ℝ := 4

/-- Represents the amount of rain on Tuesday in inches -/
def tuesday_rain : ℝ := 3

/-- Represents the price per gallon of water in dollars -/
def price_per_gallon : ℝ := 1.2

/-- Calculates the total amount of money James made from selling all the water -/
def total_money : ℝ :=
  (monday_rain * water_per_inch + tuesday_rain * water_per_inch) * price_per_gallon

/-- Theorem stating that James made $126 from selling all the water -/
theorem james_earnings : total_money = 126 := by
  sorry

end NUMINAMATH_CALUDE_james_earnings_l3956_395613


namespace NUMINAMATH_CALUDE_binary_253_ones_minus_zeros_l3956_395663

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Count the number of true values in a list of booleans -/
def countOnes (l : List Bool) : ℕ :=
  l.filter id |>.length

/-- Count the number of false values in a list of booleans -/
def countZeros (l : List Bool) : ℕ :=
  l.filter not |>.length

theorem binary_253_ones_minus_zeros :
  let binary := toBinary 253
  let y := countOnes binary
  let x := countZeros binary
  y - x = 6 := by sorry

end NUMINAMATH_CALUDE_binary_253_ones_minus_zeros_l3956_395663


namespace NUMINAMATH_CALUDE_probability_at_least_one_heart_in_top_three_l3956_395668

-- Define the total number of cards in a standard deck
def totalCards : ℕ := 52

-- Define the number of hearts in a standard deck
def numHearts : ℕ := 13

-- Define the number of cards we're considering (top three)
def topCards : ℕ := 3

-- Theorem statement
theorem probability_at_least_one_heart_in_top_three :
  let prob : ℚ := 1 - (totalCards - numHearts).descFactorial topCards / totalCards.descFactorial topCards
  prob = 325 / 425 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_heart_in_top_three_l3956_395668


namespace NUMINAMATH_CALUDE_cube_root_of_64_l3956_395682

theorem cube_root_of_64 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 64) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l3956_395682


namespace NUMINAMATH_CALUDE_f_6_equals_21_l3956_395657

def f (x : ℝ) : ℝ := (x - 1)^2 - 4

theorem f_6_equals_21 : f 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_f_6_equals_21_l3956_395657


namespace NUMINAMATH_CALUDE_power_function_below_identity_l3956_395650

theorem power_function_below_identity (α : ℝ) : 
  (∀ x : ℝ, x > 1 → x^α < x) → α < 1 := by sorry

end NUMINAMATH_CALUDE_power_function_below_identity_l3956_395650


namespace NUMINAMATH_CALUDE_black_ball_probability_l3956_395603

theorem black_ball_probability 
  (n₁ n₂ k₁ k₂ : ℕ) 
  (h_total : n₁ + n₂ = 25)
  (h_white_prob : (k₁ : ℚ) / n₁ * k₂ / n₂ = 54 / 100) :
  (n₁ - k₁ : ℚ) / n₁ * (n₂ - k₂) / n₂ = 4 / 100 := by
sorry

end NUMINAMATH_CALUDE_black_ball_probability_l3956_395603


namespace NUMINAMATH_CALUDE_unique_divisor_sequence_l3956_395611

theorem unique_divisor_sequence : ∃! (x y z w : ℕ), 
  x > 1 ∧ y > 1 ∧ z > 1 ∧ w > 1 ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
  x % y = 0 ∧ y % z = 0 ∧ z % w = 0 ∧
  x + y + z + w = 329 ∧
  x = 231 ∧ y = 77 ∧ z = 14 ∧ w = 7 := by
sorry

end NUMINAMATH_CALUDE_unique_divisor_sequence_l3956_395611


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3956_395666

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 5) :
  (a - 1)^2 - 2*a*(a - 1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3956_395666


namespace NUMINAMATH_CALUDE_circumcenter_rational_coords_l3956_395612

/-- If the coordinates of the vertices of a triangle are rational, 
    then the coordinates of the center of its circumscribed circle are also rational. -/
theorem circumcenter_rational_coords 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℚ) : ∃ x y : ℚ, 
  (x - a₁)^2 + (y - b₁)^2 = (x - a₂)^2 + (y - b₂)^2 ∧
  (x - a₁)^2 + (y - b₁)^2 = (x - a₃)^2 + (y - b₃)^2 := by
  sorry

end NUMINAMATH_CALUDE_circumcenter_rational_coords_l3956_395612


namespace NUMINAMATH_CALUDE_weighted_sum_inequality_l3956_395620

theorem weighted_sum_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (sum_geq_one : a + b + c + d ≥ 1) :
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_weighted_sum_inequality_l3956_395620


namespace NUMINAMATH_CALUDE_train_length_l3956_395646

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 180 → time = 9 → speed * time * (1000 / 3600) = 450 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3956_395646


namespace NUMINAMATH_CALUDE_trig_identity_l3956_395677

theorem trig_identity (α : Real) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * (Real.cos (π / 6 + α / 2))^2 - 1 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3956_395677


namespace NUMINAMATH_CALUDE_fraction_of_108_l3956_395616

theorem fraction_of_108 : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 108 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_108_l3956_395616


namespace NUMINAMATH_CALUDE_harkamal_fruit_payment_l3956_395672

/-- Calculates the total amount Harkamal had to pay for fruits with given quantities, prices, discount, and tax rates. -/
def calculate_total_payment (grape_kg : ℝ) (grape_price : ℝ) (mango_kg : ℝ) (mango_price : ℝ)
                            (apple_kg : ℝ) (apple_price : ℝ) (orange_kg : ℝ) (orange_price : ℝ)
                            (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let grape_total := grape_kg * grape_price
  let mango_total := mango_kg * mango_price
  let apple_total := apple_kg * apple_price
  let orange_total := orange_kg * orange_price
  let total_before_discount := grape_total + mango_total + apple_total + orange_total
  let discount := discount_rate * (grape_total + apple_total)
  let price_after_discount := total_before_discount - discount
  let tax := tax_rate * price_after_discount
  price_after_discount + tax

/-- Theorem stating that the total payment for Harkamal's fruit purchase is $1507.32. -/
theorem harkamal_fruit_payment :
  calculate_total_payment 9 70 9 55 5 40 6 30 0.1 0.06 = 1507.32 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_fruit_payment_l3956_395672


namespace NUMINAMATH_CALUDE_A_equals_Z_l3956_395644

-- Define the set A
def A : Set Int :=
  {n | ∃ (a b : Nat), a ≥ 1 ∧ b ≥ 1 ∧ n = 2^a - 2^b}

-- Define the closure property of A
axiom A_closure (a b : Int) : a ∈ A → b ∈ A → (a + b) ∈ A

-- Axiom stating that A contains at least one odd number
axiom A_contains_odd : ∃ (n : Int), n ∈ A ∧ n % 2 ≠ 0

-- Theorem to prove
theorem A_equals_Z : A = Set.univ := by sorry

end NUMINAMATH_CALUDE_A_equals_Z_l3956_395644


namespace NUMINAMATH_CALUDE_intersection_when_a_half_subset_iff_a_range_l3956_395610

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

-- Theorem 1: When a = 1/2, A ∩ B = B
theorem intersection_when_a_half : A (1/2) ∩ B = B := by sorry

-- Theorem 2: B ⊆ A if and only if 0 ≤ a ≤ 1
theorem subset_iff_a_range : B ⊆ A a ↔ 0 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_half_subset_iff_a_range_l3956_395610


namespace NUMINAMATH_CALUDE_average_of_c_and_d_l3956_395606

theorem average_of_c_and_d (c d : ℝ) : 
  (4 + 6 + 8 + c + d) / 5 = 18 → (c + d) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_average_of_c_and_d_l3956_395606


namespace NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_l3956_395687

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x² + a(b+1)x + a + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*(b+1)*x + a + b

/-- "a = 0" is a sufficient but not necessary condition for "f is an even function" -/
theorem a_zero_sufficient_not_necessary (a b : ℝ) :
  (a = 0 → IsEven (f a b)) ∧ ¬(IsEven (f a b) → a = 0) := by
  sorry

end NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_l3956_395687


namespace NUMINAMATH_CALUDE_wage_percentage_difference_l3956_395656

/-- Proves that the percentage difference between chef's and dishwasher's hourly wage is 20% -/
theorem wage_percentage_difference
  (manager_wage : ℝ)
  (chef_wage_difference : ℝ)
  (h_manager_wage : manager_wage = 6.50)
  (h_chef_wage_difference : chef_wage_difference = 2.60)
  (h_dishwasher_wage : dishwasher_wage = manager_wage / 2)
  (h_chef_wage : chef_wage = manager_wage - chef_wage_difference) :
  (chef_wage - dishwasher_wage) / dishwasher_wage * 100 = 20 :=
by sorry


end NUMINAMATH_CALUDE_wage_percentage_difference_l3956_395656


namespace NUMINAMATH_CALUDE_ellipse_and_range_of_m_l3956_395604

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the square formed by foci and vertices of minor axis -/
def square_perimeter (a b : ℝ) : Prop := 4 * a = 4 * Real.sqrt 2 ∧ b = Real.sqrt (a^2 - b^2)

/-- Definition of point B -/
def point_B (m : ℝ) : ℝ × ℝ := (0, m)

/-- Definition of point D symmetric to B with respect to origin -/
def point_D (m : ℝ) : ℝ × ℝ := (0, -m)

/-- Definition of line l passing through B -/
def line_l (x y m k : ℝ) : Prop := y = k * x + m

/-- Definition of intersection points E and F -/
def intersection_points (x y m k : ℝ) : Prop :=
  ellipse_C x y (Real.sqrt 2) 1 ∧ line_l x y m k

/-- Definition of D being inside circle with diameter EF -/
def D_inside_circle (m : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ,
    intersection_points x₁ y₁ m k ∧
    intersection_points x₂ y₂ m k ∧
    (0 - (x₁ + x₂)/2)^2 + (-m - (y₁ + y₂)/2)^2 < ((x₁ - x₂)^2 + (y₁ - y₂)^2) / 4

/-- Main theorem -/
theorem ellipse_and_range_of_m (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) 
  (h₃ : square_perimeter a b) :
  (ellipse_C x y (Real.sqrt 2) 1 ↔ ellipse_C x y a b) ∧
  (∀ m : ℝ, m > 0 → D_inside_circle m → 0 < m ∧ m < Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_range_of_m_l3956_395604


namespace NUMINAMATH_CALUDE_indifferent_passengers_adjacent_probability_l3956_395693

/-- The number of seats on each sofa -/
def seats_per_sofa : ℕ := 5

/-- The total number of passengers -/
def total_passengers : ℕ := 10

/-- The number of passengers who prefer to sit facing the locomotive -/
def facing_passengers : ℕ := 4

/-- The number of passengers who prefer to sit with their backs to the locomotive -/
def back_passengers : ℕ := 3

/-- The number of passengers who do not care where they sit -/
def indifferent_passengers : ℕ := 3

/-- The probability that two of the three indifferent passengers sit next to each other -/
theorem indifferent_passengers_adjacent_probability :
  (seats_per_sofa = 5) →
  (total_passengers = 10) →
  (facing_passengers = 4) →
  (back_passengers = 3) →
  (indifferent_passengers = 3) →
  (Nat.factorial seats_per_sofa * Nat.factorial 3 * 2 * 4) / 
  (3 * Nat.factorial seats_per_sofa * Nat.factorial seats_per_sofa) = 2 / 15 := by
  sorry


end NUMINAMATH_CALUDE_indifferent_passengers_adjacent_probability_l3956_395693


namespace NUMINAMATH_CALUDE_max_perimeter_l3956_395634

-- Define the triangle sides
def side1 : ℝ := 7
def side2 : ℝ := 8

-- Define the third side as a natural number
def x : ℕ → ℝ := λ n => n

-- Define the triangle inequality
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- The theorem to prove
theorem max_perimeter :
  ∃ n : ℕ, (is_valid_triangle side1 side2 (x n)) ∧
  (∀ m : ℕ, is_valid_triangle side1 side2 (x m) →
    perimeter side1 side2 (x n) ≥ perimeter side1 side2 (x m)) ∧
  perimeter side1 side2 (x n) = 29 :=
sorry

end NUMINAMATH_CALUDE_max_perimeter_l3956_395634


namespace NUMINAMATH_CALUDE_smallest_integer_S_n_l3956_395608

def K' : ℚ := 137 / 60

def S (n : ℕ) : ℚ := n * 5^(n-1) * K' + 1

theorem smallest_integer_S_n : 
  (∀ m : ℕ, m > 0 → m < 12 → ¬(S m).isInt) ∧ (S 12).isInt :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_S_n_l3956_395608


namespace NUMINAMATH_CALUDE_fraction_addition_l3956_395645

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3956_395645


namespace NUMINAMATH_CALUDE_square_less_than_triple_l3956_395678

theorem square_less_than_triple (n : ℕ) : n > 0 → (n^2 < 3*n ↔ n = 1 ∨ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l3956_395678
