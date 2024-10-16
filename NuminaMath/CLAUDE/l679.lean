import Mathlib

namespace NUMINAMATH_CALUDE_union_intersection_result_intersection_complement_result_l679_67936

-- Define the universe set U
def U : Set ℤ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define sets A, B, and C
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-1, 0, 1}
def C : Set ℤ := {-2, 0, 2}

-- Theorem for the first part
theorem union_intersection_result : A ∪ (B ∩ C) = {0, 1, 2, 3} := by
  sorry

-- Theorem for the second part
theorem intersection_complement_result : A ∩ (U \ (B ∪ C)) = {3} := by
  sorry

end NUMINAMATH_CALUDE_union_intersection_result_intersection_complement_result_l679_67936


namespace NUMINAMATH_CALUDE_consecutive_integers_product_210_l679_67971

theorem consecutive_integers_product_210 (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 210) → (a + b + c = 18) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_210_l679_67971


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l679_67988

theorem difference_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l679_67988


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l679_67984

-- Define the total number of balls and the number of each color
def totalBalls : ℕ := 6
def redBalls : ℕ := 3
def whiteBalls : ℕ := 3

-- Define the number of balls drawn
def ballsDrawn : ℕ := 3

-- Define the events
def atLeastTwoWhite (w : ℕ) : Prop := w ≥ 2
def allRed (r : ℕ) : Prop := r = 3

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : Prop) : Prop :=
  ¬(e1 ∧ e2)

-- Define opposite events
def oppositeEvents (e1 e2 : Prop) : Prop :=
  ∀ (outcome : ℕ × ℕ), (e1 ∨ e2) ∧ ¬(e1 ∧ e2)

-- Theorem statement
theorem events_mutually_exclusive_but_not_opposite :
  (mutuallyExclusive (atLeastTwoWhite whiteBalls) (allRed redBalls)) ∧
  ¬(oppositeEvents (atLeastTwoWhite whiteBalls) (allRed redBalls)) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l679_67984


namespace NUMINAMATH_CALUDE_sin_240_degrees_l679_67967

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l679_67967


namespace NUMINAMATH_CALUDE_inequality_solution_and_function_property_l679_67922

def f (x : ℝ) : ℝ := |x - 1|

theorem inequality_solution_and_function_property :
  (∃ (S : Set ℝ), S = {x : ℝ | x ≤ -2 ∨ x ≥ 4/3} ∧
    ∀ x : ℝ, x ∈ S ↔ f (2*x) + f (x+4) ≥ 6) ∧
  (∀ a b : ℝ, |a| < 1 → |b| < 1 → f (a*b) > f (a-b+1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_function_property_l679_67922


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l679_67945

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℚ, 
  (∀ x : ℚ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 36 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l679_67945


namespace NUMINAMATH_CALUDE_books_bought_from_first_shop_l679_67916

theorem books_bought_from_first_shop
  (total_first_shop : ℕ)
  (books_second_shop : ℕ)
  (total_second_shop : ℕ)
  (average_price : ℕ)
  (h1 : total_first_shop = 600)
  (h2 : books_second_shop = 20)
  (h3 : total_second_shop = 240)
  (h4 : average_price = 14)
  : ∃ (books_first_shop : ℕ),
    (total_first_shop + total_second_shop) / (books_first_shop + books_second_shop) = average_price ∧
    books_first_shop = 40 :=
by sorry

end NUMINAMATH_CALUDE_books_bought_from_first_shop_l679_67916


namespace NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l679_67914

/-- Hyperbola type representing xy = 1 -/
structure Hyperbola where
  C₁ : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.1 * p.2 = 1}
  C₂ : Set (ℝ × ℝ) := {p | p.1 < 0 ∧ p.1 * p.2 = 1}

/-- Predicate to check if three points form an equilateral triangle -/
def IsEquilateralTriangle (p q r : ℝ × ℝ) : Prop :=
  let d₁ := (p.1 - q.1)^2 + (p.2 - q.2)^2
  let d₂ := (q.1 - r.1)^2 + (q.2 - r.2)^2
  let d₃ := (r.1 - p.1)^2 + (r.2 - p.2)^2
  d₁ = d₂ ∧ d₂ = d₃

/-- Main theorem statement -/
theorem hyperbola_equilateral_triangle (h : Hyperbola) (p q r : ℝ × ℝ) 
  (hp : p = (-1, -1) ∧ p ∈ h.C₂)
  (hq : q ∈ h.C₁)
  (hr : r ∈ h.C₁)
  (heq : IsEquilateralTriangle p q r) :
  (¬ (p ∈ h.C₁ ∧ q ∈ h.C₁ ∧ r ∈ h.C₁)) ∧
  (q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ r = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l679_67914


namespace NUMINAMATH_CALUDE_eleventh_number_is_137_l679_67915

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits add up to 11 -/
def nth_number_with_digit_sum_11 (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 11th number in the sequence is 137 -/
theorem eleventh_number_is_137 : nth_number_with_digit_sum_11 11 = 137 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_is_137_l679_67915


namespace NUMINAMATH_CALUDE_probability_theorem_l679_67970

def is_valid_pair (b c : Int) : Prop :=
  (b.natAbs ≤ 6) ∧ (c.natAbs ≤ 6)

def has_non_real_or_non_positive_roots (b c : Int) : Prop :=
  (b^2 < 4*c) ∨ (b ≥ 0) ∨ (b^2 ≤ 4*c)

def total_pairs : Nat := 13 * 13

def valid_pairs : Nat := 150

theorem probability_theorem :
  (Nat.cast valid_pairs / Nat.cast total_pairs : ℚ) = 150 / 169 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l679_67970


namespace NUMINAMATH_CALUDE_hyperbola_condition_equivalence_l679_67902

/-- The equation represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 1) + y^2 / (k + 2) = 1 ∧ (k - 1) * (k + 2) < 0

/-- The condition 0 < k < 1 -/
def condition (k : ℝ) : Prop := 0 < k ∧ k < 1

theorem hyperbola_condition_equivalence :
  ∀ k : ℝ, is_hyperbola k ↔ condition k := by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_equivalence_l679_67902


namespace NUMINAMATH_CALUDE_adblock_interesting_ads_l679_67911

theorem adblock_interesting_ads 
  (total_ads : ℝ) 
  (unblocked_ratio : ℝ) 
  (uninteresting_unblocked_ratio : ℝ) 
  (h1 : unblocked_ratio = 0.2) 
  (h2 : uninteresting_unblocked_ratio = 0.16) : 
  (unblocked_ratio * total_ads - uninteresting_unblocked_ratio * total_ads) / (unblocked_ratio * total_ads) = 0.2 := by
sorry

end NUMINAMATH_CALUDE_adblock_interesting_ads_l679_67911


namespace NUMINAMATH_CALUDE_fifth_term_is_fifteen_l679_67938

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 2 + a 4 = 16
  first_term : a 1 = 1

/-- The fifth term of the arithmetic sequence is 15 -/
theorem fifth_term_is_fifteen (seq : ArithmeticSequence) : seq.a 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_fifteen_l679_67938


namespace NUMINAMATH_CALUDE_max_product_sum_2020_l679_67965

theorem max_product_sum_2020 : 
  (∃ (a b : ℤ), a + b = 2020 ∧ a * b = 1020100) ∧ 
  (∀ (x y : ℤ), x + y = 2020 → x * y ≤ 1020100) := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_2020_l679_67965


namespace NUMINAMATH_CALUDE_expected_votes_for_candidate_a_l679_67953

-- Define the percentage of Democrat and Republican voters
def democrat_percentage : ℝ := 0.60
def republican_percentage : ℝ := 1 - democrat_percentage

-- Define the percentage of Democrats and Republicans voting for candidate A
def democrat_vote_for_a : ℝ := 0.85
def republican_vote_for_a : ℝ := 0.20

-- Define the theorem
theorem expected_votes_for_candidate_a :
  democrat_percentage * democrat_vote_for_a + republican_percentage * republican_vote_for_a = 0.59 := by
  sorry

end NUMINAMATH_CALUDE_expected_votes_for_candidate_a_l679_67953


namespace NUMINAMATH_CALUDE_move_point_right_l679_67924

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point horizontally by a given distance -/
def moveHorizontally (p : Point) (distance : ℝ) : Point :=
  { x := p.x + distance, y := p.y }

theorem move_point_right : 
  let A : Point := { x := -2, y := 3 }
  let movedA : Point := moveHorizontally A 2
  movedA = { x := 0, y := 3 } := by
  sorry


end NUMINAMATH_CALUDE_move_point_right_l679_67924


namespace NUMINAMATH_CALUDE_pizza_combinations_l679_67985

theorem pizza_combinations (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 3) :
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l679_67985


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l679_67939

/-- The perimeter of a semi-circle with radius 21.005164601010506 cm is 108.01915941002101 cm -/
theorem semicircle_perimeter : 
  let r : ℝ := 21.005164601010506
  (π * r + 2 * r) = 108.01915941002101 := by
sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l679_67939


namespace NUMINAMATH_CALUDE_locus_of_point_P_l679_67997

/-- Given two rays OA and OB, and a point P inside the angle AOx, prove the equation of the locus of P and its domain --/
theorem locus_of_point_P (k : ℝ) (h_k : k > 0) :
  ∃ (f : ℝ → ℝ) (domain : Set ℝ),
    (∀ x y, (y = k * x ∧ x > 0) → (y = f x → x ∈ domain)) ∧
    (∀ x y, (y = -k * x ∧ x > 0) → (y = f x → x ∈ domain)) ∧
    (∀ x y, x ∈ domain → 0 < y ∧ y < k * x ∧ y < (1/k) * x) ∧
    (∀ x y, x ∈ domain → y = f x → y = Real.sqrt (x^2 - (1 + k^2))) ∧
    (0 < k ∧ k < 1 →
      domain = {x | Real.sqrt (k^2 + 1) < x ∧ x < Real.sqrt ((k^2 + 1)/(1 - k^2))}) ∧
    (k = 1 →
      domain = {x | Real.sqrt 2 < x}) ∧
    (k > 1 →
      domain = {x | Real.sqrt (k^2 + 1) < x ∧ x < k * Real.sqrt ((k^2 + 1)/(k^2 - 1))}) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_point_P_l679_67997


namespace NUMINAMATH_CALUDE_target_hit_probability_l679_67977

theorem target_hit_probability (p_a p_b : ℝ) (h_a : p_a = 0.9) (h_b : p_b = 0.8) 
  (h_independent : True) : 1 - (1 - p_a) * (1 - p_b) = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l679_67977


namespace NUMINAMATH_CALUDE_ferry_problem_l679_67934

/-- The ferry problem -/
theorem ferry_problem (speed_p speed_q : ℝ) (time_p : ℝ) (distance_q : ℝ) :
  speed_p = 8 →
  time_p = 3 →
  speed_q = speed_p + 4 →
  distance_q = 2 * speed_p * time_p →
  distance_q / speed_q - time_p = 1 := by
  sorry

end NUMINAMATH_CALUDE_ferry_problem_l679_67934


namespace NUMINAMATH_CALUDE_fraction_equality_l679_67975

theorem fraction_equality : 
  (1 * 2 + 2 * 4 - 3 * 8 + 4 * 16 + 5 * 32 - 6 * 64) / 
  (2 * 4 + 4 * 8 - 6 * 16 + 8 * 32 + 10 * 64 - 12 * 128) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l679_67975


namespace NUMINAMATH_CALUDE_same_terminal_side_characterization_l679_67978

def has_same_terminal_side (α : ℝ) : Prop :=
  ∃ k : ℤ, α = 30 + k * 360

theorem same_terminal_side_characterization (α : ℝ) :
  has_same_terminal_side α ↔ (α = -30 ∨ α = 390) :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_characterization_l679_67978


namespace NUMINAMATH_CALUDE_range_of_f_l679_67955

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l679_67955


namespace NUMINAMATH_CALUDE_factorization_sum_l679_67925

theorem factorization_sum (a b c d e f g : ℤ) :
  (∀ x y : ℝ, 16 * x^4 - 81 * y^4 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y)) →
  a + b + c + d + e + f + g = 17 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l679_67925


namespace NUMINAMATH_CALUDE_increased_chickens_sum_l679_67960

/-- The number of increased chickens since the beginning -/
def increased_chickens (original : ℕ) (first_day : ℕ) (second_day : ℕ) : ℕ :=
  first_day + second_day

/-- Theorem stating that the number of increased chickens is the sum of chickens brought on the first and second day -/
theorem increased_chickens_sum (original : ℕ) (first_day : ℕ) (second_day : ℕ) :
  increased_chickens original first_day second_day = first_day + second_day :=
by sorry

#eval increased_chickens 45 18 12

end NUMINAMATH_CALUDE_increased_chickens_sum_l679_67960


namespace NUMINAMATH_CALUDE_scientists_born_in_july_percentage_l679_67983

theorem scientists_born_in_july_percentage :
  let total_scientists : ℕ := 120
  let july_born_scientists : ℕ := 15
  (july_born_scientists : ℚ) / total_scientists * 100 = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_scientists_born_in_july_percentage_l679_67983


namespace NUMINAMATH_CALUDE_tomato_types_salad_bar_problem_l679_67917

theorem tomato_types (lettuce_types : Nat) (olive_types : Nat) (soup_types : Nat) 
  (total_options : Nat) : Nat :=
  let tomato_types := total_options / (lettuce_types * olive_types * soup_types)
  tomato_types

theorem salad_bar_problem :
  let lettuce_types := 2
  let olive_types := 4
  let soup_types := 2
  let total_options := 48
  tomato_types lettuce_types olive_types soup_types total_options = 3 := by
  sorry

end NUMINAMATH_CALUDE_tomato_types_salad_bar_problem_l679_67917


namespace NUMINAMATH_CALUDE_first_day_is_sunday_l679_67961

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of week after a given number of days -/
def afterDays (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | Nat.succ m => nextDay (afterDays start m)

/-- Theorem: If the 21st day of a month is a Saturday, then the 1st day of that month is a Sunday -/
theorem first_day_is_sunday (d : DayOfWeek) :
  afterDays d 20 = DayOfWeek.Saturday → d = DayOfWeek.Sunday :=
by
  sorry


end NUMINAMATH_CALUDE_first_day_is_sunday_l679_67961


namespace NUMINAMATH_CALUDE_mango_ratio_l679_67994

def alexis_mangoes : ℕ := 60
def total_mangoes : ℕ := 75

def others_mangoes : ℕ := total_mangoes - alexis_mangoes

theorem mango_ratio : 
  (alexis_mangoes : ℚ) / (others_mangoes : ℚ) = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_mango_ratio_l679_67994


namespace NUMINAMATH_CALUDE_min_sum_with_log_condition_l679_67943

theorem min_sum_with_log_condition (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_log : Real.log a + Real.log b = Real.log (a + b)) :
  ∀ x y : ℝ, x > 0 → y > 0 → Real.log x + Real.log y = Real.log (x + y) → a + b ≤ x + y ∧ a + b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_log_condition_l679_67943


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l679_67954

def U : Set ℝ := Set.univ
def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x ≤ -1 ∨ x > 2}

theorem complement_intersection_theorem :
  (Set.compl B ∩ A) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l679_67954


namespace NUMINAMATH_CALUDE_cartesian_to_polar_conversion_l679_67968

theorem cartesian_to_polar_conversion (x y ρ θ : Real) :
  x = -1 ∧ y = Real.sqrt 3 →
  ρ = 2 ∧ θ = 2 * Real.pi / 3 →
  ρ * Real.cos θ = x ∧ ρ * Real.sin θ = y ∧ ρ^2 = x^2 + y^2 :=
by sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_conversion_l679_67968


namespace NUMINAMATH_CALUDE_coprime_20172019_l679_67948

theorem coprime_20172019 :
  (Nat.gcd 20172019 20172017 = 1) ∧
  (Nat.gcd 20172019 20172018 = 1) ∧
  (Nat.gcd 20172019 20172020 = 1) ∧
  (Nat.gcd 20172019 20172021 = 1) :=
by sorry

end NUMINAMATH_CALUDE_coprime_20172019_l679_67948


namespace NUMINAMATH_CALUDE_network_connections_l679_67996

/-- Calculates the number of unique connections in a network of switches -/
def calculate_connections (num_switches : ℕ) (connections_per_switch : ℕ) : ℕ :=
  (num_switches * connections_per_switch) / 2

/-- Theorem: In a network of 30 switches, where each switch is connected to exactly 5 other switches,
    the total number of unique connections is 75. -/
theorem network_connections :
  calculate_connections 30 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l679_67996


namespace NUMINAMATH_CALUDE_unique_solution_system_l679_67972

theorem unique_solution_system :
  ∃! (x y z : ℝ), 
    x * y + y * z + z * x = 1 ∧ 
    5 * x + 8 * y + 9 * z = 12 ∧
    x = 1 ∧ y = (1 : ℝ) / 2 ∧ z = (1 : ℝ) / 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l679_67972


namespace NUMINAMATH_CALUDE_pine_percentage_correct_l679_67947

/-- Represents the number of trees of each type in the forest -/
structure ForestComposition where
  oak : ℕ
  pine : ℕ
  spruce : ℕ
  birch : ℕ

/-- The total number of trees in the forest -/
def total_trees : ℕ := 4000

/-- The actual composition of the forest -/
def forest : ForestComposition := {
  oak := 720,
  pine := 520,
  spruce := 400,
  birch := 2160
}

/-- The percentage of pine trees in the forest -/
def pine_percentage : ℚ := 13 / 100

theorem pine_percentage_correct :
  (forest.oak + forest.pine + forest.spruce + forest.birch = total_trees) ∧
  (forest.spruce = total_trees / 10) ∧
  (forest.oak = forest.spruce + forest.pine) ∧
  (forest.birch = 2160) →
  (forest.pine : ℚ) / total_trees = pine_percentage :=
by sorry

end NUMINAMATH_CALUDE_pine_percentage_correct_l679_67947


namespace NUMINAMATH_CALUDE_planes_parallel_to_line_are_parallel_planes_parallel_to_plane_are_parallel_l679_67900

-- Define a type for planes
variable (Plane : Type)

-- Define a type for lines
variable (Line : Type)

-- Define a relation for parallelism between planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define a relation for parallelism between a plane and a line
variable (parallel_plane_line : Plane → Line → Prop)

-- Define a relation for parallelism between a plane and another plane
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Theorem 1: Two planes parallel to the same line are parallel
theorem planes_parallel_to_line_are_parallel 
  (P Q : Plane) (L : Line) 
  (h1 : parallel_plane_line P L) 
  (h2 : parallel_plane_line Q L) : 
  parallel_planes P Q :=
sorry

-- Theorem 2: Two planes parallel to the same plane are parallel
theorem planes_parallel_to_plane_are_parallel 
  (P Q R : Plane) 
  (h1 : parallel_plane_plane P R) 
  (h2 : parallel_plane_plane Q R) : 
  parallel_planes P Q :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_line_are_parallel_planes_parallel_to_plane_are_parallel_l679_67900


namespace NUMINAMATH_CALUDE_saltwater_volume_l679_67909

/-- Proves that the initial volume of a saltwater solution is 160 gallons --/
theorem saltwater_volume : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (0.20 * x / x = 1/5) ∧ 
  ((0.20 * x + 16) / (3/4 * x + 24) = 1/3) ∧ 
  (x = 160) := by
  sorry

end NUMINAMATH_CALUDE_saltwater_volume_l679_67909


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l679_67949

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  have h1 : 8251 = 6105 * 1 + 2146 := by sorry
  have h2 : 6105 = 2146 * 2 + 1813 := by sorry
  have h3 : 2146 = 1813 * 1 + 333 := by sorry
  have h4 : 333 = 148 * 2 + 37 := by sorry
  have h5 : 148 = 37 * 4 := by sorry
  sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l679_67949


namespace NUMINAMATH_CALUDE_range_of_m_when_a_is_zero_x_minus_one_times_f_nonpositive_l679_67956

noncomputable section

-- Define the function f
def f (m a x : ℝ) : ℝ := -m * (a * x + 1) * Real.log x + x - a

-- Part 1
theorem range_of_m_when_a_is_zero (m : ℝ) :
  (∀ x > 1, f m 0 x ≥ 0) ↔ m ∈ Set.Iic (Real.exp 1) :=
sorry

-- Part 2
theorem x_minus_one_times_f_nonpositive (x : ℝ) (hx : x > 0) :
  (x - 1) * f 1 1 x ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_when_a_is_zero_x_minus_one_times_f_nonpositive_l679_67956


namespace NUMINAMATH_CALUDE_disk_space_calculation_l679_67928

/-- The total space on Mike's disk drive in GB. -/
def total_space : ℕ := 28

/-- The space taken by Mike's files in GB. -/
def file_space : ℕ := 26

/-- The space left over after backing up Mike's files in GB. -/
def space_left : ℕ := 2

/-- Theorem stating that the total space on Mike's disk drive is equal to
    the sum of the space taken by his files and the space left over. -/
theorem disk_space_calculation :
  total_space = file_space + space_left := by sorry

end NUMINAMATH_CALUDE_disk_space_calculation_l679_67928


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_one_range_of_a_l679_67910

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| - |x + 1|

-- Part I
theorem solution_set_for_a_eq_one :
  {x : ℝ | f 1 x ≤ x^2 - x} = {x : ℝ | x ≤ -1 ∨ x ≥ 0} :=
sorry

-- Part II
theorem range_of_a (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : 2*m + n = 1) :
  (∀ x, f a x ≤ 1/m + 2/n) → -9 ≤ a ∧ a ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_one_range_of_a_l679_67910


namespace NUMINAMATH_CALUDE_binary_101_eq_5_l679_67913

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 5 -/
def binary_5 : List Bool := [true, false, true]

/-- Theorem stating that the binary number 101 (represented as [true, false, true]) is equal to 5 in decimal -/
theorem binary_101_eq_5 : binary_to_decimal binary_5 = 5 := by sorry

end NUMINAMATH_CALUDE_binary_101_eq_5_l679_67913


namespace NUMINAMATH_CALUDE_cost_2005_l679_67935

/-- Represents the number of songs downloaded in 2004 -/
def songs_2004 : ℕ := 200

/-- Represents the number of songs downloaded in 2005 -/
def songs_2005 : ℕ := 360

/-- Represents the difference in cost per song between 2004 and 2005 in cents -/
def cost_difference : ℕ := 32

/-- Theorem stating that the cost of downloading 360 songs in 2005 was $144.00 -/
theorem cost_2005 (c : ℚ) : 
  (songs_2005 : ℚ) * c = (songs_2004 : ℚ) * (c + cost_difference) → 
  songs_2005 * c = 14400 := by
  sorry

end NUMINAMATH_CALUDE_cost_2005_l679_67935


namespace NUMINAMATH_CALUDE_exists_n_with_special_digit_sum_l679_67995

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_n_with_special_digit_sum : 
  ∃ (n : ℕ), n > 0 ∧ sumOfDigits n = 1000 ∧ sumOfDigits (n^2) = 1000^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_with_special_digit_sum_l679_67995


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l679_67957

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + (y + 2)^2) - Real.sqrt ((x - 6)^2 + (y + 2)^2) = 4

-- Define the distance between foci
def distance_between_foci : ℝ := 5

-- Define the semi-major axis
def semi_major_axis : ℝ := 2

-- Define the positive slope of an asymptote
def positive_asymptote_slope : ℝ := 0.75

-- Theorem statement
theorem hyperbola_asymptote_slope :
  positive_asymptote_slope = (Real.sqrt (((distance_between_foci / 2)^2) - semi_major_axis^2)) / semi_major_axis :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l679_67957


namespace NUMINAMATH_CALUDE_seven_possible_D_values_l679_67952

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the addition of two 5-digit numbers ABBCB + BCAIA = DBDDD -/
def ValidAddition (A B C D : Digit) : Prop :=
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D) ∧
  (10000 * A.val + 1000 * B.val + 100 * B.val + 10 * C.val + B.val) +
  (10000 * B.val + 1000 * C.val + 100 * A.val + 10 * 1 + A.val) =
  (10000 * D.val + 1000 * B.val + 100 * D.val + 10 * D.val + D.val)

/-- The theorem stating that there are exactly 7 possible values for D -/
theorem seven_possible_D_values :
  ∃ (S : Finset Digit), S.card = 7 ∧
  (∀ D, D ∈ S ↔ ∃ A B C, ValidAddition A B C D) :=
sorry

end NUMINAMATH_CALUDE_seven_possible_D_values_l679_67952


namespace NUMINAMATH_CALUDE_initial_crayons_l679_67926

theorem initial_crayons (C : ℕ) : 
  (C : ℚ) * (3/4) * (1/2) = 18 → C = 48 := by
  sorry

end NUMINAMATH_CALUDE_initial_crayons_l679_67926


namespace NUMINAMATH_CALUDE_ordering_of_trig_functions_l679_67963

theorem ordering_of_trig_functions (a b c d : ℝ) : 
  a = Real.sin (Real.cos (2015 * π / 180)) →
  b = Real.sin (Real.sin (2015 * π / 180)) →
  c = Real.cos (Real.sin (2015 * π / 180)) →
  d = Real.cos (Real.cos (2015 * π / 180)) →
  c > d ∧ d > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ordering_of_trig_functions_l679_67963


namespace NUMINAMATH_CALUDE_catch_up_theorem_l679_67980

/-- The time it takes for me to walk from home to school in minutes -/
def my_walk_time : ℝ := 30

/-- The time it takes for my brother to walk from home to school in minutes -/
def brother_walk_time : ℝ := 40

/-- The time my brother left before me in minutes -/
def brother_head_start : ℝ := 5

/-- The time it takes for me to catch up with my brother in minutes -/
def catch_up_time : ℝ := 15

/-- Theorem stating that I will catch up with my brother after 15 minutes -/
theorem catch_up_theorem :
  let my_speed := 1 / my_walk_time
  let brother_speed := 1 / brother_walk_time
  let relative_speed := my_speed - brother_speed
  let head_start_distance := brother_speed * brother_head_start
  head_start_distance / relative_speed = catch_up_time := by
  sorry


end NUMINAMATH_CALUDE_catch_up_theorem_l679_67980


namespace NUMINAMATH_CALUDE_addition_preserves_inequality_l679_67929

theorem addition_preserves_inequality (a b c d : ℝ) : a < b → c < d → a + c < b + d := by
  sorry

end NUMINAMATH_CALUDE_addition_preserves_inequality_l679_67929


namespace NUMINAMATH_CALUDE_hole_perimeter_formula_l679_67951

/-- Represents an isosceles trapezium -/
structure IsoscelesTrapezium where
  a : ℝ  -- Length of non-parallel sides
  b : ℝ  -- Length of longer parallel side

/-- Represents an equilateral triangle with a hole formed by three congruent isosceles trapeziums -/
structure TriangleWithHole where
  trapezium : IsoscelesTrapezium
  -- Assumption that three of these trapeziums form an equilateral triangle with a hole

/-- The perimeter of the hole in a TriangleWithHole -/
def holePerimeter (t : TriangleWithHole) : ℝ :=
  6 * t.trapezium.a - 3 * t.trapezium.b

/-- Theorem stating that the perimeter of the hole is 6a - 3b -/
theorem hole_perimeter_formula (t : TriangleWithHole) :
  holePerimeter t = 6 * t.trapezium.a - 3 * t.trapezium.b :=
by
  sorry

end NUMINAMATH_CALUDE_hole_perimeter_formula_l679_67951


namespace NUMINAMATH_CALUDE_heather_walk_distance_l679_67966

/-- The distance Heather walked from the carnival rides back to the car -/
def carnival_to_car : ℝ := 0.08333333333333333

/-- The total distance Heather walked -/
def total_distance : ℝ := 0.75

/-- The distance from the car to the entrance (and from the entrance to the carnival rides) -/
def car_to_entrance : ℝ := 0.33333333333333335

theorem heather_walk_distance :
  2 * car_to_entrance + carnival_to_car = total_distance :=
by sorry

end NUMINAMATH_CALUDE_heather_walk_distance_l679_67966


namespace NUMINAMATH_CALUDE_tan_neg_five_pi_fourth_l679_67946

theorem tan_neg_five_pi_fourth : Real.tan (-5 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_five_pi_fourth_l679_67946


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l679_67987

theorem simplify_sqrt_expression :
  (Real.sqrt 450 / Real.sqrt 200) - (Real.sqrt 175 / Real.sqrt 75) = (9 - 2 * Real.sqrt 21) / 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l679_67987


namespace NUMINAMATH_CALUDE_meetings_on_elliptical_track_l679_67979

/-- Represents the elliptical track --/
structure EllipticalTrack where
  majorAxis : ℝ
  minorAxis : ℝ

/-- Represents a boy running on the track --/
structure Runner where
  speed : ℝ

/-- Calculates the number of meetings between two runners on an elliptical track --/
def numberOfMeetings (track : EllipticalTrack) (runner1 runner2 : Runner) : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem meetings_on_elliptical_track :
  let track := EllipticalTrack.mk 100 60
  let runner1 := Runner.mk 7
  let runner2 := Runner.mk 11
  numberOfMeetings track runner1 runner2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_meetings_on_elliptical_track_l679_67979


namespace NUMINAMATH_CALUDE_circle_tangent_triangle_l679_67990

/-- Given a circle with radius R externally tangent to triangle ABC, 
    prove that angle C is π/6 and the maximum area is (√3 + 2)/4 * R^2 -/
theorem circle_tangent_triangle (R a b c : ℝ) (A B C : ℝ) :
  R > 0 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  2 * R * (Real.sin A ^ 2 - Real.sin C ^ 2) = (Real.sqrt 3 * a - b) * Real.sin B →
  C = π / 6 ∧ 
  ∃ (S : ℝ), S ≤ (Real.sqrt 3 + 2) / 4 * R^2 ∧ 
    (∀ (A' B' C' : ℝ), A' + B' + C' = π → 
      1 / 2 * 2 * R * Real.sin A' * 2 * R * Real.sin B' * Real.sin C' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_triangle_l679_67990


namespace NUMINAMATH_CALUDE_school_teacher_student_ratio_l679_67999

theorem school_teacher_student_ratio 
  (b c k h : ℕ) 
  (h_positive : h > 0) 
  (k_ge_two : k ≥ 2) 
  (c_ge_two : c ≥ 2) :
  (b : ℚ) / h = (c * (c - 1)) / (k * (k - 1)) := by
sorry

end NUMINAMATH_CALUDE_school_teacher_student_ratio_l679_67999


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l679_67982

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l679_67982


namespace NUMINAMATH_CALUDE_skew_to_common_line_relationships_l679_67919

-- Define the concept of a line in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- or any other suitable representation
  -- This is just a placeholder structure

-- Define the concept of skew lines
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are not parallel and do not intersect
  sorry

-- Define the possible positional relationships
inductive PositionalRelationship
  | Parallel
  | Intersecting
  | Skew

-- Theorem statement
theorem skew_to_common_line_relationships 
  (a b l : Line3D) 
  (ha : are_skew a l) 
  (hb : are_skew b l) : 
  ∃ (r : PositionalRelationship), 
    (r = PositionalRelationship.Parallel) ∨ 
    (r = PositionalRelationship.Intersecting) ∨ 
    (r = PositionalRelationship.Skew) :=
sorry

end NUMINAMATH_CALUDE_skew_to_common_line_relationships_l679_67919


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l679_67901

theorem hyperbola_a_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (b / a = 2) →
  (a^2 + b^2 = 20) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l679_67901


namespace NUMINAMATH_CALUDE_remaining_budget_theorem_l679_67992

/-- Represents the annual budget of Centerville --/
def annual_budget : ℝ := 20000

/-- Represents the percentage of the budget spent on the public library --/
def library_percentage : ℝ := 0.15

/-- Represents the amount spent on the public library --/
def library_spending : ℝ := 3000

/-- Represents the percentage of the budget spent on public parks --/
def parks_percentage : ℝ := 0.24

/-- Theorem stating the remaining amount of the budget after library and parks spending --/
theorem remaining_budget_theorem :
  annual_budget * (1 - library_percentage - parks_percentage) = 12200 := by
  sorry


end NUMINAMATH_CALUDE_remaining_budget_theorem_l679_67992


namespace NUMINAMATH_CALUDE_increasing_function_domain_l679_67937

/-- The function f(x) = x^2 + 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

/-- The set A on which f is defined -/
def A : Set ℝ := {x | ∃ y, f x = y}

theorem increasing_function_domain (h : StrictMono f) : A = Set.Ici (-1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_domain_l679_67937


namespace NUMINAMATH_CALUDE_max_value_sin_cos_l679_67944

theorem max_value_sin_cos (a b : ℝ) (h : a^2 + b^2 ≥ 1) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → a * Real.sin θ + b * Real.cos θ ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ a * Real.sin θ + b * Real.cos θ = Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_l679_67944


namespace NUMINAMATH_CALUDE_coins_probability_theorem_l679_67950

def total_coins : ℕ := 15
def num_quarters : ℕ := 3
def num_dimes : ℕ := 5
def num_nickels : ℕ := 7
def coins_drawn : ℕ := 8

def value_quarter : ℚ := 25 / 100
def value_dime : ℚ := 10 / 100
def value_nickel : ℚ := 5 / 100

def target_value : ℚ := 3 / 2

def probability_at_least_target : ℚ := 316 / 6435

theorem coins_probability_theorem :
  let total_outcomes := Nat.choose total_coins coins_drawn
  let successful_outcomes := 
    Nat.choose num_quarters 3 * Nat.choose num_dimes 5 +
    Nat.choose num_quarters 2 * Nat.choose num_dimes 4 * Nat.choose num_nickels 2
  (successful_outcomes : ℚ) / total_outcomes = probability_at_least_target :=
sorry

end NUMINAMATH_CALUDE_coins_probability_theorem_l679_67950


namespace NUMINAMATH_CALUDE_cherry_pie_count_l679_67962

theorem cherry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h1 : total_pies = 36)
  (h2 : apple_ratio = 2)
  (h3 : blueberry_ratio = 5)
  (h4 : cherry_ratio = 4) :
  (cherry_ratio : ℚ) * total_pies / (apple_ratio + blueberry_ratio + cherry_ratio) = 144 / 11 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pie_count_l679_67962


namespace NUMINAMATH_CALUDE_problem_solution_l679_67903

theorem problem_solution (x : ℝ) : (400 * 7000 : ℝ) = 28000 * (100 ^ x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l679_67903


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l679_67907

theorem quadratic_root_implies_k (p k : ℝ) : 
  (∃ x : ℂ, 3 * x^2 + p * x + k = 0 ∧ x = 4 + 3*I) → k = 75 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l679_67907


namespace NUMINAMATH_CALUDE_least_integer_in_ratio_l679_67906

theorem least_integer_in_ratio (a b c : ℕ+) : 
  a.val + b.val + c.val = 90 →
  2 * a = 3 * a →
  5 * a = 3 * b →
  a ≤ b ∧ a ≤ c →
  a.val = 9 := by
sorry

end NUMINAMATH_CALUDE_least_integer_in_ratio_l679_67906


namespace NUMINAMATH_CALUDE_min_M_is_two_thirds_l679_67974

-- Define the set of quadratic polynomials satisfying the conditions
def QuadraticPolynomials : Set (ℝ → ℝ) :=
  {p | ∀ x ∈ Set.Icc (-1 : ℝ) 1,
       (∃ a b c : ℝ, p = fun x ↦ a * x^2 + b * x + c) ∧
       p x ≥ 0 ∧
       (∫ x in Set.Icc (-1 : ℝ) 1, p x) = 1}

-- Define M(x) as the maximum value of polynomials in QuadraticPolynomials at x
noncomputable def M (x : ℝ) : ℝ :=
  ⨆ (p ∈ QuadraticPolynomials), p x

theorem min_M_is_two_thirds :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, M x ≥ 2/3) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, M x = 2/3) :=
sorry

end NUMINAMATH_CALUDE_min_M_is_two_thirds_l679_67974


namespace NUMINAMATH_CALUDE_expand_polynomial_product_l679_67933

theorem expand_polynomial_product : ∀ t : ℝ,
  (3 * t^2 - 4 * t + 3) * (-2 * t^2 + 3 * t - 4) = 
  -6 * t^4 + 17 * t^3 - 30 * t^2 + 25 * t - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_product_l679_67933


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l679_67904

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 < 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l679_67904


namespace NUMINAMATH_CALUDE_expected_smallest_seven_from_sixtythree_l679_67931

/-- The expected value of the smallest number when randomly selecting r numbers from a set of n numbers. -/
def expected_smallest (n : ℕ) (r : ℕ) : ℚ :=
  (n + 1 : ℚ) / (r + 1 : ℚ)

/-- The set size -/
def n : ℕ := 63

/-- The sample size -/
def r : ℕ := 7

theorem expected_smallest_seven_from_sixtythree :
  expected_smallest n r = 8 := by
  sorry

end NUMINAMATH_CALUDE_expected_smallest_seven_from_sixtythree_l679_67931


namespace NUMINAMATH_CALUDE_A_D_mutually_exclusive_not_complementary_l679_67959

/-- Represents the possible outcomes of a fair die toss -/
inductive DieFace
  | one
  | two
  | three
  | four
  | five
  | six

/-- Event A: an odd number is facing up -/
def event_A (face : DieFace) : Prop :=
  face = DieFace.one ∨ face = DieFace.three ∨ face = DieFace.five

/-- Event D: either 2 or 4 is facing up -/
def event_D (face : DieFace) : Prop :=
  face = DieFace.two ∨ face = DieFace.four

/-- The sample space of a fair die toss -/
def sample_space : Set DieFace :=
  {DieFace.one, DieFace.two, DieFace.three, DieFace.four, DieFace.five, DieFace.six}

theorem A_D_mutually_exclusive_not_complementary :
  (∀ (face : DieFace), ¬(event_A face ∧ event_D face)) ∧
  (∃ (face : DieFace), ¬event_A face ∧ ¬event_D face) :=
by sorry

end NUMINAMATH_CALUDE_A_D_mutually_exclusive_not_complementary_l679_67959


namespace NUMINAMATH_CALUDE_probability_at_least_one_heart_in_top_three_l679_67921

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

end NUMINAMATH_CALUDE_probability_at_least_one_heart_in_top_three_l679_67921


namespace NUMINAMATH_CALUDE_bakery_outdoor_tables_l679_67969

/-- Given a bakery setup with indoor and outdoor tables, prove the number of outdoor tables. -/
theorem bakery_outdoor_tables
  (indoor_tables : ℕ)
  (indoor_chairs_per_table : ℕ)
  (outdoor_chairs_per_table : ℕ)
  (total_chairs : ℕ)
  (h1 : indoor_tables = 8)
  (h2 : indoor_chairs_per_table = 3)
  (h3 : outdoor_chairs_per_table = 3)
  (h4 : total_chairs = 60) :
  (total_chairs - indoor_tables * indoor_chairs_per_table) / outdoor_chairs_per_table = 12 := by
  sorry

end NUMINAMATH_CALUDE_bakery_outdoor_tables_l679_67969


namespace NUMINAMATH_CALUDE_magazines_read_in_five_hours_l679_67918

/-- 
Proves that given a reading rate of 1 magazine per 20 minutes, 
the number of magazines that can be read in 5 hours is equal to 15.
-/
theorem magazines_read_in_five_hours 
  (reading_rate : ℚ) -- Reading rate in magazines per minute
  (hours : ℕ) -- Number of hours
  (h1 : reading_rate = 1 / 20) -- Reading rate is 1 magazine per 20 minutes
  (h2 : hours = 5) -- Time period is 5 hours
  : ⌊hours * 60 * reading_rate⌋ = 15 := by
  sorry

#check magazines_read_in_five_hours

end NUMINAMATH_CALUDE_magazines_read_in_five_hours_l679_67918


namespace NUMINAMATH_CALUDE_similar_triangles_shortest_side_l679_67998

theorem similar_triangles_shortest_side 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a = 24) 
  (h3 : c = 37) 
  (h4 : ∃ k, k > 0 ∧ k * c = 74) : 
  ∃ x, x > 0 ∧ x^2 = 793 ∧ 2 * x = min (2 * a) (2 * b) := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_shortest_side_l679_67998


namespace NUMINAMATH_CALUDE_sergey_ndfl_calculation_l679_67912

/-- Calculates the personal income tax (НДФЛ) for a Russian resident --/
def calculate_ndfl (monthly_income : ℕ) (bonus : ℕ) (car_sale : ℕ) (land_purchase : ℕ) : ℕ :=
  let annual_income := monthly_income * 12
  let total_income := annual_income + bonus + car_sale
  let total_deductions := car_sale + land_purchase
  let taxable_income := total_income - total_deductions
  let tax_rate := 13
  (taxable_income * tax_rate) / 100

/-- Theorem stating that the calculated НДФЛ for Sergey is 10400 rubles --/
theorem sergey_ndfl_calculation :
  calculate_ndfl 30000 20000 250000 300000 = 10400 := by
  sorry

end NUMINAMATH_CALUDE_sergey_ndfl_calculation_l679_67912


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l679_67905

theorem fraction_to_decimal : (29 : ℚ) / 160 = 0.18125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l679_67905


namespace NUMINAMATH_CALUDE_slope_relationship_l679_67958

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Definition of point A₁ -/
def A₁ : ℝ × ℝ := (-2, 0)

/-- Definition of point A₂ -/
def A₂ : ℝ × ℝ := (2, 0)

/-- Definition of the line PQ -/
def line_PQ (x y : ℝ) : Prop := ∃ m : ℝ, x = m * y + 1/2

/-- Theorem stating the relationship between slopes -/
theorem slope_relationship (P Q : ℝ × ℝ) :
  ellipse_C P.1 P.2 →
  ellipse_C Q.1 Q.2 →
  line_PQ P.1 P.2 →
  line_PQ Q.1 Q.2 →
  P ≠ A₁ →
  P ≠ A₂ →
  Q ≠ A₁ →
  Q ≠ A₂ →
  (P.2 - A₁.2) / (P.1 - A₁.1) = 3/5 * (Q.2 - A₂.2) / (Q.1 - A₂.1) :=
sorry

end NUMINAMATH_CALUDE_slope_relationship_l679_67958


namespace NUMINAMATH_CALUDE_cost_price_calculation_l679_67920

/-- Proves that the cost price of an article is $975, given that it was sold at $1170 with a 20% profit. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 1170 →
  profit_percentage = 20 →
  selling_price = (100 + profit_percentage) / 100 * 975 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l679_67920


namespace NUMINAMATH_CALUDE_seats_needed_l679_67908

/-- Given 58 children and 2 children per seat, prove that 29 seats are needed. -/
theorem seats_needed (total_children : ℕ) (children_per_seat : ℕ) (h1 : total_children = 58) (h2 : children_per_seat = 2) :
  total_children / children_per_seat = 29 := by
  sorry

end NUMINAMATH_CALUDE_seats_needed_l679_67908


namespace NUMINAMATH_CALUDE_max_value_S_l679_67976

theorem max_value_S (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) 
  (hsum : a + b + c + d = 100) : 
  let S := (a / (b + 7)) ^ (1/3) + (b / (c + 7)) ^ (1/3) + (c / (d + 7)) ^ (1/3) + (d / (a + 7)) ^ (1/3)
  S ≤ 8 / 7^(1/3) := by
sorry

end NUMINAMATH_CALUDE_max_value_S_l679_67976


namespace NUMINAMATH_CALUDE_indifferent_passengers_adjacent_probability_l679_67930

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


end NUMINAMATH_CALUDE_indifferent_passengers_adjacent_probability_l679_67930


namespace NUMINAMATH_CALUDE_blood_cell_count_l679_67986

theorem blood_cell_count (total : ℕ) (second : ℕ) (first : ℕ) : 
  total = 7341 → second = 3120 → first = total - second → first = 4221 := by
  sorry

end NUMINAMATH_CALUDE_blood_cell_count_l679_67986


namespace NUMINAMATH_CALUDE_min_value_expression_l679_67941

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (((x^2 + y^2 + z^2) * (4 * x^2 + y^2 + z^2)).sqrt) / (x * y * z) ≥ 4 ∧
  (∃ (a : ℝ), a > 0 ∧ (((a^2 + a^2 + a^2) * (4 * a^2 + a^2 + a^2)).sqrt) / (a * a * a) = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l679_67941


namespace NUMINAMATH_CALUDE_pinedale_bus_distance_l679_67964

theorem pinedale_bus_distance (average_speed : ℝ) (stop_interval : ℝ) (num_stops : ℕ) 
  (h1 : average_speed = 60) 
  (h2 : stop_interval = 5 / 60) 
  (h3 : num_stops = 8) : 
  average_speed * (stop_interval * num_stops) = 40 := by
  sorry

end NUMINAMATH_CALUDE_pinedale_bus_distance_l679_67964


namespace NUMINAMATH_CALUDE_no_natural_square_diff_2018_l679_67932

theorem no_natural_square_diff_2018 : ¬∃ (a b : ℕ), a^2 - b^2 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_square_diff_2018_l679_67932


namespace NUMINAMATH_CALUDE_expression_value_l679_67973

theorem expression_value : 
  (2020^4 - 3 * 2020^3 * 2021 + 4 * 2020 * 2021^3 - 2021^4 + 1) / (2020 * 2021) = 4096046 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l679_67973


namespace NUMINAMATH_CALUDE_max_substitutions_is_fifty_l679_67942

/-- A type representing a fifth-degree polynomial -/
def FifthDegreePolynomial := ℕ → ℕ

/-- Given a list of ten fifth-degree polynomials, returns the maximum number of consecutive
    natural numbers that can be substituted to produce an arithmetic progression -/
def max_consecutive_substitutions (polynomials : List FifthDegreePolynomial) : ℕ :=
  sorry

/-- The main theorem stating that the maximum number of consecutive substitutions is 50 -/
theorem max_substitutions_is_fifty :
  ∀ (polynomials : List FifthDegreePolynomial),
    polynomials.length = 10 →
    max_consecutive_substitutions polynomials = 50 :=
  sorry

end NUMINAMATH_CALUDE_max_substitutions_is_fifty_l679_67942


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l679_67991

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ p q : ℝ, p ≠ 0 ∧ q ≠ 0 ∧ p / q = 3 / 2 ∧ 
   p + q = -10 ∧ p * q = k ∧ 
   ∀ x : ℝ, x^2 + 10*x + k = 0 ↔ (x = p ∨ x = q)) → 
  k = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l679_67991


namespace NUMINAMATH_CALUDE_mortgage_payment_months_l679_67940

theorem mortgage_payment_months (first_payment : ℝ) (ratio : ℝ) (total_amount : ℝ) : 
  first_payment = 100 →
  ratio = 3 →
  total_amount = 2952400 →
  (∃ n : ℕ, first_payment * (1 - ratio^n) / (1 - ratio) = total_amount ∧ n = 10) :=
by sorry

end NUMINAMATH_CALUDE_mortgage_payment_months_l679_67940


namespace NUMINAMATH_CALUDE_inequality_proof_l679_67989

theorem inequality_proof (a b c : ℝ) (ha : a = 31/32) (hb : b = Real.cos (1/4))
  (hc : c = 4 * Real.sin (1/4)) : c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l679_67989


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l679_67981

-- Problem 1
theorem factorization_problem_1 (a b x y : ℝ) :
  a * (x + y) - 2 * b * (x + y) = (x + y) * (a - 2 * b) := by sorry

-- Problem 2
theorem factorization_problem_2 (a b : ℝ) :
  a^3 + 2*a^2*b + a*b^2 = a * (a + b)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l679_67981


namespace NUMINAMATH_CALUDE_hindi_speakers_count_l679_67927

/-- Represents the number of students who can speak a certain number of languages -/
structure LanguageSpeakers where
  total : ℕ  -- Total number of students in the class
  gujarati : ℕ  -- Number of students who can speak Gujarati
  marathi : ℕ  -- Number of students who can speak Marathi
  twoLanguages : ℕ  -- Number of students who can speak two languages
  allThree : ℕ  -- Number of students who can speak all three languages

/-- Calculates the number of Hindi speakers given the language distribution in the class -/
def numHindiSpeakers (ls : LanguageSpeakers) : ℕ :=
  ls.total - (ls.gujarati + ls.marathi - ls.twoLanguages + ls.allThree)

/-- Theorem stating that the number of Hindi speakers is 10 given the problem conditions -/
theorem hindi_speakers_count (ls : LanguageSpeakers) 
  (h_total : ls.total = 22)
  (h_gujarati : ls.gujarati = 6)
  (h_marathi : ls.marathi = 6)
  (h_two : ls.twoLanguages = 2)
  (h_all : ls.allThree = 1) :
  numHindiSpeakers ls = 10 := by
  sorry


end NUMINAMATH_CALUDE_hindi_speakers_count_l679_67927


namespace NUMINAMATH_CALUDE_large_rectangle_ratio_l679_67993

/-- Represents the side length of a square in the arrangement -/
def square_side : ℝ := sorry

/-- Represents the length of the large rectangle -/
def large_rectangle_length : ℝ := 3 * square_side

/-- Represents the width of the large rectangle -/
def large_rectangle_width : ℝ := 3 * square_side

/-- Represents the length of the smaller rectangle -/
def small_rectangle_length : ℝ := 3 * square_side

/-- Represents the width of the smaller rectangle -/
def small_rectangle_width : ℝ := square_side

theorem large_rectangle_ratio :
  large_rectangle_length / large_rectangle_width = 3 := by sorry

end NUMINAMATH_CALUDE_large_rectangle_ratio_l679_67993


namespace NUMINAMATH_CALUDE_collinear_points_t_value_l679_67923

/-- Given three points A, B, and C in a 2D plane, this function checks if they are collinear --/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem stating that if points A(1, 2), B(-3, 4), and C(2, t) are collinear, then t = 3/2 --/
theorem collinear_points_t_value :
  ∀ t : ℝ, are_collinear (1, 2) (-3, 4) (2, t) → t = 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_points_t_value_l679_67923
