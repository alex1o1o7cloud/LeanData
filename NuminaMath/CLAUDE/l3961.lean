import Mathlib

namespace NUMINAMATH_CALUDE_fraction_evaluation_l3961_396191

theorem fraction_evaluation (x : ℝ) (h : x = 3) : (x^6 + 8*x^3 + 16) / (x^3 + 4) = 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3961_396191


namespace NUMINAMATH_CALUDE_ajay_dal_gain_l3961_396131

/-- Calculates the total gain from a dal transaction -/
def calculate_gain (quantity1 : ℕ) (price1 : ℚ) (quantity2 : ℕ) (price2 : ℚ) (selling_price : ℚ) : ℚ :=
  let total_cost := quantity1 * price1 + quantity2 * price2
  let total_quantity := quantity1 + quantity2
  let total_revenue := total_quantity * selling_price
  total_revenue - total_cost

/-- Proves that Ajay's total gain in the dal transaction is Rs 27.50 -/
theorem ajay_dal_gain : calculate_gain 15 (14.5) 10 13 15 = (27.5) := by
  sorry

end NUMINAMATH_CALUDE_ajay_dal_gain_l3961_396131


namespace NUMINAMATH_CALUDE_fine_on_fifth_day_l3961_396130

/-- Calculates the fine for a given day -/
def dailyFine (previousFine : ℚ) : ℚ :=
  min (previousFine * 2) (previousFine + 0.15)

/-- Calculates the total fine up to a given day -/
def totalFine (day : ℕ) : ℚ :=
  match day with
  | 0 => 0
  | 1 => 0.05
  | n + 1 => totalFine n + dailyFine (dailyFine (totalFine n))

/-- The theorem to be proved -/
theorem fine_on_fifth_day :
  totalFine 5 = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_fine_on_fifth_day_l3961_396130


namespace NUMINAMATH_CALUDE_estate_value_l3961_396140

/-- Represents the estate distribution problem --/
structure EstateDistribution where
  total : ℝ
  daughter1 : ℝ
  daughter2 : ℝ
  son : ℝ
  husband : ℝ
  gardener : ℝ

/-- The estate distribution satisfies the given conditions --/
def validDistribution (e : EstateDistribution) : Prop :=
  -- The two daughters and son receive 3/5 of the estate
  e.daughter1 + e.daughter2 + e.son = 3/5 * e.total
  -- The daughters and son share in the ratio of 5:3:2
  ∧ e.daughter1 = 5/10 * (e.daughter1 + e.daughter2 + e.son)
  ∧ e.daughter2 = 3/10 * (e.daughter1 + e.daughter2 + e.son)
  ∧ e.son = 2/10 * (e.daughter1 + e.daughter2 + e.son)
  -- The husband gets three times as much as the son
  ∧ e.husband = 3 * e.son
  -- The gardener receives $600
  ∧ e.gardener = 600
  -- The total estate is the sum of all shares
  ∧ e.total = e.daughter1 + e.daughter2 + e.son + e.husband + e.gardener

/-- The estate value is $15000 --/
theorem estate_value (e : EstateDistribution) (h : validDistribution e) : e.total = 15000 := by
  sorry

end NUMINAMATH_CALUDE_estate_value_l3961_396140


namespace NUMINAMATH_CALUDE_problem_statement_l3961_396115

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of f(x) + x^2 being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + x^2 = -(f (-x) + (-x)^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem problem_statement (f : ℝ → ℝ) 
  (h1 : is_odd_function f) 
  (h2 : f 1 = 1) : 
  g f (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3961_396115


namespace NUMINAMATH_CALUDE_seven_pow_minus_three_times_two_pow_eq_one_l3961_396107

theorem seven_pow_minus_three_times_two_pow_eq_one
  (m n : ℕ+) : 7^(m:ℕ) - 3 * 2^(n:ℕ) = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_seven_pow_minus_three_times_two_pow_eq_one_l3961_396107


namespace NUMINAMATH_CALUDE_exists_special_function_l3961_396118

/-- A function from pairs of positive integers to positive integers -/
def PositiveIntegerFunction := ℕ+ → ℕ+ → ℕ+

/-- Predicate for a function being a polynomial in one variable when the other is fixed -/
def IsPolynomialInOneVariable (f : PositiveIntegerFunction) : Prop :=
  (∀ x : ℕ+, ∃ Px : ℕ+ → ℕ+, ∀ y : ℕ+, f x y = Px y) ∧
  (∀ y : ℕ+, ∃ Qy : ℕ+ → ℕ+, ∀ x : ℕ+, f x y = Qy x)

/-- Predicate for a function not being a polynomial in both variables -/
def IsNotPolynomialInBothVariables (f : PositiveIntegerFunction) : Prop :=
  ¬∃ P : ℕ+ → ℕ+ → ℕ+, ∀ x y : ℕ+, f x y = P x y

/-- The main theorem stating the existence of a function with the required properties -/
theorem exists_special_function : 
  ∃ f : PositiveIntegerFunction, 
    IsPolynomialInOneVariable f ∧ IsNotPolynomialInBothVariables f := by
  sorry

end NUMINAMATH_CALUDE_exists_special_function_l3961_396118


namespace NUMINAMATH_CALUDE_locus_is_ray_l3961_396104

/-- The locus of point P satisfying |PM| - |PN| = 4 is a ray -/
theorem locus_is_ray (M N P : ℝ × ℝ) :
  M = (-2, 0) →
  N = (2, 0) →
  abs (P.1 - M.1) + abs (P.2 - M.2) - (abs (P.1 - N.1) + abs (P.2 - N.2)) = 4 →
  P.1 ≥ 2 ∧ P.2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_is_ray_l3961_396104


namespace NUMINAMATH_CALUDE_smallest_digit_correction_l3961_396187

def original_sum : ℕ := 356 + 781 + 492
def incorrect_sum : ℕ := 1529
def corrected_number : ℕ := 256

theorem smallest_digit_correction :
  (original_sum = incorrect_sum + 100) ∧
  (corrected_number + 781 + 492 = incorrect_sum) ∧
  (∀ n : ℕ, n < 356 → n > corrected_number → n + 781 + 492 ≠ incorrect_sum) := by
  sorry

end NUMINAMATH_CALUDE_smallest_digit_correction_l3961_396187


namespace NUMINAMATH_CALUDE_rotating_triangle_path_length_l3961_396121

/-- The total path length of point A in a rotating triangle -/
theorem rotating_triangle_path_length 
  (a : Real) 
  (h1 : 0 < a) 
  (h2 : a < π / 3) : 
  ∃ (s : Real), 
    s = 22 * π * (1 + Real.sin a) - 66 * a ∧ 
    s = (100 - 1) / 3 * (2 / 3 * π * (1 + Real.sin a) - 2 * a) := by
  sorry

end NUMINAMATH_CALUDE_rotating_triangle_path_length_l3961_396121


namespace NUMINAMATH_CALUDE_f_increasing_when_x_greater_than_one_l3961_396151

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

theorem f_increasing_when_x_greater_than_one :
  ∀ x : ℝ, x > 1 → (deriv f) x > 0 := by sorry

end NUMINAMATH_CALUDE_f_increasing_when_x_greater_than_one_l3961_396151


namespace NUMINAMATH_CALUDE_sphere_volume_calculation_l3961_396153

-- Define the sphere and plane
def Sphere : Type := Unit
def Plane : Type := Unit

-- Define the properties of the intersection
def intersection_diameter (s : Sphere) (p : Plane) : ℝ := 6

-- Define the distance from the center of the sphere to the plane
def center_to_plane_distance (s : Sphere) (p : Plane) : ℝ := 4

-- Define the volume of a sphere
def sphere_volume (s : Sphere) : ℝ := sorry

-- Theorem statement
theorem sphere_volume_calculation (s : Sphere) (p : Plane) :
  sphere_volume s = (500 * Real.pi) / 3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_calculation_l3961_396153


namespace NUMINAMATH_CALUDE_prob_four_blue_before_three_yellow_l3961_396197

/-- The probability of drawing all blue marbles before all yellow marbles -/
def blue_before_yellow_prob (blue : ℕ) (yellow : ℕ) : ℚ :=
  if blue = 0 then 0
  else if yellow = 0 then 1
  else (blue : ℚ) / (blue + yellow : ℚ)

/-- The theorem stating the probability of drawing all 4 blue marbles before all 3 yellow marbles -/
theorem prob_four_blue_before_three_yellow :
  blue_before_yellow_prob 4 3 = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_blue_before_three_yellow_l3961_396197


namespace NUMINAMATH_CALUDE_factory_weekly_production_l3961_396168

/-- Represents a toy production line with a daily production rate -/
structure ProductionLine where
  dailyRate : ℕ

/-- Represents a factory with multiple production lines -/
structure Factory where
  lines : List ProductionLine
  daysPerWeek : ℕ

/-- Calculates the total weekly production of a factory -/
def weeklyProduction (factory : Factory) : ℕ :=
  (factory.lines.map (λ line => line.dailyRate * factory.daysPerWeek)).sum

/-- The theorem stating the total weekly production of the given factory -/
theorem factory_weekly_production :
  let lineA : ProductionLine := ⟨1500⟩
  let lineB : ProductionLine := ⟨1800⟩
  let lineC : ProductionLine := ⟨2200⟩
  let factory : Factory := ⟨[lineA, lineB, lineC], 5⟩
  weeklyProduction factory = 27500 := by
  sorry


end NUMINAMATH_CALUDE_factory_weekly_production_l3961_396168


namespace NUMINAMATH_CALUDE_f_of_g_of_three_l3961_396174

def f (x : ℝ) : ℝ := 2 * x - 5

def g (x : ℝ) : ℝ := x + 2

theorem f_of_g_of_three : f (1 + g 3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_of_three_l3961_396174


namespace NUMINAMATH_CALUDE_sally_quarters_remaining_l3961_396128

def initial_quarters : ℕ := 760
def first_purchase : ℕ := 418
def second_purchase : ℕ := 192

theorem sally_quarters_remaining :
  initial_quarters - first_purchase - second_purchase = 150 := by sorry

end NUMINAMATH_CALUDE_sally_quarters_remaining_l3961_396128


namespace NUMINAMATH_CALUDE_artist_paint_usage_l3961_396106

/-- The amount of paint used for all paintings --/
def total_paint_used (large_paint small_paint large_count small_count : ℕ) : ℕ :=
  large_paint * large_count + small_paint * small_count

/-- Proof that the artist used 17 ounces of paint --/
theorem artist_paint_usage : total_paint_used 3 2 3 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_artist_paint_usage_l3961_396106


namespace NUMINAMATH_CALUDE_quadratic_function_a_range_l3961_396164

/-- Given a quadratic function y = (ax - 1)(x - a), this theorem proves that the range of a
    satisfying specific conditions about its roots and axis of symmetry is (0, 1). -/
theorem quadratic_function_a_range :
  ∀ a : ℝ,
  (∀ x : ℝ, (a * x - 1) * (x - a) > 0 ↔ x < a ∨ x > 1/a) ∧
  ((a^2 + 1) / (2 * a) > 0) ∧
  ¬(∀ x : ℝ, (a * x - 1) * (x - a) < 0 ↔ x < a ∨ x > 1/a)
  ↔ 0 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_a_range_l3961_396164


namespace NUMINAMATH_CALUDE_special_arithmetic_progression_all_integer_l3961_396129

/-- An arithmetic progression with the property that the product of any two distinct terms is also a term. -/
structure SpecialArithmeticProgression where
  seq : ℕ → ℤ
  is_arithmetic : ∃ d : ℤ, ∀ n : ℕ, seq (n + 1) = seq n + d
  is_increasing : ∀ n : ℕ, seq (n + 1) > seq n
  product_property : ∀ m n : ℕ, m ≠ n → ∃ k : ℕ, seq m * seq n = seq k

/-- All terms in a SpecialArithmeticProgression are integers. -/
theorem special_arithmetic_progression_all_integer (ap : SpecialArithmeticProgression) : 
  ∀ n : ℕ, ∃ k : ℤ, ap.seq n = k :=
sorry

end NUMINAMATH_CALUDE_special_arithmetic_progression_all_integer_l3961_396129


namespace NUMINAMATH_CALUDE_fraction_simplification_l3961_396145

theorem fraction_simplification 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (h_sum : a^2 + b^2 + c^2 ≠ 0) :
  (a^2*b^2 + 2*a^2*b*c + a^2*c^2 - b^4) / (a^4 - b^2*c^2 + 2*a*b*c^2 + c^4) = 
  ((a*b+a*c+b^2)*(a*b+a*c-b^2)) / ((a^2 + b^2 - c^2)*(a^2 - b^2 + c^2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3961_396145


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l3961_396134

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- A diagonal of a regular nonagon is a line segment connecting two non-adjacent vertices -/
def Diagonal (n : RegularNonagon) : Type := Unit

/-- The probability that two randomly chosen diagonals of a regular nonagon intersect inside the nonagon -/
def intersectionProbability (n : RegularNonagon) : ℚ :=
  14 / 39

/-- Theorem: The probability that two randomly chosen diagonals of a regular nonagon intersect inside the nonagon is 14/39 -/
theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersectionProbability n = 14 / 39 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l3961_396134


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3961_396180

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (11^3 + 3^3 : ℝ)^(1/3) → 
  ∃ (n : ℤ), n = 11 ∧ ∀ (m : ℤ), |x - n| ≤ |x - m| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3961_396180


namespace NUMINAMATH_CALUDE_value_of_b_l3961_396101

theorem value_of_b (m a b d : ℝ) (h : m = (d * a * b) / (a + b)) :
  b = (m * a) / (d * a - m) := by sorry

end NUMINAMATH_CALUDE_value_of_b_l3961_396101


namespace NUMINAMATH_CALUDE_not_right_triangle_condition_l3961_396126

/-- Triangle ABC with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of a right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

/-- The theorem to be proved -/
theorem not_right_triangle_condition (t : Triangle) 
  (h1 : t.a = 3^2)
  (h2 : t.b = 4^2)
  (h3 : t.c = 5^2) : 
  ¬ is_right_triangle t := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_condition_l3961_396126


namespace NUMINAMATH_CALUDE_scientific_notation_of_nanometers_l3961_396117

theorem scientific_notation_of_nanometers : 
  ∃ (a : ℝ) (n : ℤ), 0.000000007 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7 ∧ n = -9 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_nanometers_l3961_396117


namespace NUMINAMATH_CALUDE_honey_servings_l3961_396184

def total_honey : ℚ := 37 + 1/3
def serving_size : ℚ := 1 + 1/2

theorem honey_servings : (total_honey / serving_size) = 24 + 8/9 := by
  sorry

end NUMINAMATH_CALUDE_honey_servings_l3961_396184


namespace NUMINAMATH_CALUDE_functional_equation_implies_identity_l3961_396103

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- Theorem stating that any function satisfying the functional equation is the identity function -/
theorem functional_equation_implies_identity (f : ℝ → ℝ) 
  (h : FunctionalEquation f) : ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_identity_l3961_396103


namespace NUMINAMATH_CALUDE_license_plate_theorem_l3961_396177

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of consonants -/
def consonant_count : ℕ := alphabet_size - vowel_count

/-- The number of possible digits -/
def digit_count : ℕ := 10

/-- The number of license plate combinations -/
def license_plate_count : ℕ := consonant_count * consonant_count * vowel_count * vowel_count * digit_count

theorem license_plate_theorem : license_plate_count = 110250 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l3961_396177


namespace NUMINAMATH_CALUDE_num_teams_is_nine_l3961_396190

/-- The number of games in a round-robin tournament with n teams -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of games played in the tournament -/
def total_games : ℕ := 36

/-- Theorem: The number of teams in the tournament is 9 -/
theorem num_teams_is_nine : ∃ (n : ℕ), n > 0 ∧ num_games n = total_games ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_num_teams_is_nine_l3961_396190


namespace NUMINAMATH_CALUDE_los_angeles_women_ratio_l3961_396192

/-- The ratio of women to the total population in Los Angeles -/
def women_ratio (total_population women_in_retail : ℕ) (retail_fraction : ℚ) : ℚ :=
  (women_in_retail / retail_fraction) / total_population

/-- Proof that the ratio of women to the total population in Los Angeles is 1/2 -/
theorem los_angeles_women_ratio :
  women_ratio 6000000 1000000 (1/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_los_angeles_women_ratio_l3961_396192


namespace NUMINAMATH_CALUDE_item_sale_ratio_l3961_396102

theorem item_sale_ratio (x y c : ℝ) (hx : x = 0.9 * c) (hy : y = 1.2 * c) :
  y / x = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_item_sale_ratio_l3961_396102


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l3961_396178

theorem difference_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l3961_396178


namespace NUMINAMATH_CALUDE_ab_equals_two_l3961_396144

theorem ab_equals_two (a b : ℝ) (h : (a + 1)^2 + (b + 2)^2 = 0) : a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_two_l3961_396144


namespace NUMINAMATH_CALUDE_probability_different_groups_l3961_396150

/-- The number of study groups -/
def num_groups : ℕ := 6

/-- The number of members in each study group -/
def members_per_group : ℕ := 3

/-- The total number of people -/
def total_people : ℕ := num_groups * members_per_group

/-- The number of people to be selected -/
def people_to_select : ℕ := 3

/-- The probability of selecting 3 people from different study groups -/
theorem probability_different_groups : 
  (Nat.choose num_groups people_to_select : ℚ) / 
  (Nat.choose total_people people_to_select : ℚ) = 5 / 204 :=
sorry

end NUMINAMATH_CALUDE_probability_different_groups_l3961_396150


namespace NUMINAMATH_CALUDE_total_tickets_sold_l3961_396159

/-- Given the number of tickets sold in section A and section B, prove that the total number of tickets sold is their sum. -/
theorem total_tickets_sold (section_a_tickets : ℕ) (section_b_tickets : ℕ) :
  section_a_tickets = 2900 →
  section_b_tickets = 1600 →
  section_a_tickets + section_b_tickets = 4500 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l3961_396159


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3961_396166

theorem triangle_angle_A (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  b = 4 →
  Real.sin B = 2/3 →
  a < b →
  (Real.sin A) * b = a * (Real.sin B) →
  A = π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3961_396166


namespace NUMINAMATH_CALUDE_gcd_2023_2048_l3961_396148

theorem gcd_2023_2048 : Nat.gcd 2023 2048 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2023_2048_l3961_396148


namespace NUMINAMATH_CALUDE_negation_of_all_is_some_not_l3961_396169

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (M : U → Prop)  -- M x means "x is a member of the math club"
variable (E : U → Prop)  -- E x means "x enjoys puzzles"

-- State the theorem
theorem negation_of_all_is_some_not :
  (¬ ∀ x, M x → E x) ↔ (∃ x, M x ∧ ¬ E x) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_is_some_not_l3961_396169


namespace NUMINAMATH_CALUDE_rowing_speed_problem_l3961_396119

/-- Represents the rowing speed problem -/
theorem rowing_speed_problem (v c : ℝ) : 
  c = 1.4 → 
  (v + c) = 2 * (v - c) → 
  v = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_rowing_speed_problem_l3961_396119


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_and_formula_l3961_396173

/-- A geometric sequence with first term 1 and third term 4 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 3 = 4 ∧ ∀ n : ℕ, n ≥ 1 → ∃ q : ℝ, a (n + 1) = a n * q

theorem geometric_sequence_ratio_and_formula (a : ℕ → ℝ) (h : GeometricSequence a) :
  (∃ q : ℝ, q = 2 ∨ q = -2) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 2^(n-1) ∨ a n = (-2)^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_and_formula_l3961_396173


namespace NUMINAMATH_CALUDE_relay_schemes_count_l3961_396136

/-- The number of segments in the Olympic torch relay route -/
def num_segments : ℕ := 6

/-- The number of torchbearers -/
def num_torchbearers : ℕ := 6

/-- The set of possible first runners -/
inductive FirstRunner
| A
| B
| C

/-- The set of possible last runners -/
inductive LastRunner
| A
| B

/-- A function to calculate the number of relay schemes -/
def count_relay_schemes : ℕ := sorry

/-- Theorem stating that the number of relay schemes is 96 -/
theorem relay_schemes_count :
  count_relay_schemes = 96 := by sorry

end NUMINAMATH_CALUDE_relay_schemes_count_l3961_396136


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l3961_396162

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the cost of insulating a rectangular tank -/
def insulationCost (l w h costPerSqFt : ℝ) : ℝ :=
  surfaceArea l w h * costPerSqFt

/-- Theorem: The cost of insulating a rectangular tank with given dimensions is $1240 -/
theorem tank_insulation_cost :
  insulationCost 5 3 2 20 = 1240 := by
  sorry

end NUMINAMATH_CALUDE_tank_insulation_cost_l3961_396162


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3961_396108

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  a 5 - (1/2) * a 7 = (1/2) * a 7 - a 6 →
  (a 1 + a 2 + a 3) / (a 2 + a 3 + a 4) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3961_396108


namespace NUMINAMATH_CALUDE_sqrt5_irrational_l3961_396194

theorem sqrt5_irrational : Irrational (Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt5_irrational_l3961_396194


namespace NUMINAMATH_CALUDE_max_value_constraint_l3961_396193

theorem max_value_constraint (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  2*x*y*Real.sqrt 6 + 9*y*z ≤ Real.sqrt 87 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3961_396193


namespace NUMINAMATH_CALUDE_pencils_per_box_l3961_396133

theorem pencils_per_box (total_pencils : ℕ) (num_boxes : ℕ) 
  (h1 : total_pencils = 648) 
  (h2 : num_boxes = 162) 
  (h3 : total_pencils % num_boxes = 0) : 
  total_pencils / num_boxes = 4 := by
sorry

end NUMINAMATH_CALUDE_pencils_per_box_l3961_396133


namespace NUMINAMATH_CALUDE_average_fish_is_75_l3961_396105

/-- The number of fish in Boast Pool -/
def boast_pool : ℕ := 75

/-- The number of fish in Onum Lake -/
def onum_lake : ℕ := boast_pool + 25

/-- The number of fish in Riddle Pond -/
def riddle_pond : ℕ := onum_lake / 2

/-- The total number of fish in all three bodies of water -/
def total_fish : ℕ := boast_pool + onum_lake + riddle_pond

/-- The number of bodies of water -/
def num_bodies : ℕ := 3

/-- Theorem stating that the average number of fish in all three bodies of water is 75 -/
theorem average_fish_is_75 : total_fish / num_bodies = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_fish_is_75_l3961_396105


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_point_circle_tangent_conditions_l3961_396157

/-- The equation of the given circle -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 4*a*x + 2*a*y + 20*(a - 1) = 0

/-- The equation of the fixed circle -/
def fixed_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

theorem circle_passes_through_fixed_point :
  ∀ a : ℝ, circle_equation 4 (-2) a := by sorry

theorem circle_tangent_conditions :
  ∀ a : ℝ, (∃ x y : ℝ, circle_equation x y a ∧ fixed_circle_equation x y ∧
    (∀ x' y' : ℝ, circle_equation x' y' a ∧ fixed_circle_equation x' y' → (x', y') = (x, y))) ↔
  (a = 1 - Real.sqrt 5 ∨ a = 1 + Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_circle_passes_through_fixed_point_circle_tangent_conditions_l3961_396157


namespace NUMINAMATH_CALUDE_afternoon_absences_l3961_396172

theorem afternoon_absences (morning_registered : ℕ) (morning_absent : ℕ) 
  (afternoon_registered : ℕ) (total_students : ℕ) 
  (h1 : morning_registered = 25)
  (h2 : morning_absent = 3)
  (h3 : afternoon_registered = 24)
  (h4 : total_students = 42)
  (h5 : total_students = (morning_registered - morning_absent) + (afternoon_registered - afternoon_absent)) :
  afternoon_absent = 4 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_absences_l3961_396172


namespace NUMINAMATH_CALUDE_coin_arrangement_count_l3961_396120

/-- Represents the number of ways to arrange 5 gold coins in 6 gaps between silver coins -/
def gold_arrangements : ℕ := 6

/-- Represents the number of valid orientations satisfying the engraving conditions -/
def valid_orientations : ℕ := 8

/-- The total number of distinguishable arrangements satisfying all conditions -/
def total_arrangements : ℕ := gold_arrangements * valid_orientations

/-- Theorem stating that the number of distinguishable arrangements is 48 -/
theorem coin_arrangement_count :
  total_arrangements = 48 := by sorry

end NUMINAMATH_CALUDE_coin_arrangement_count_l3961_396120


namespace NUMINAMATH_CALUDE_pencil_boxes_per_carton_l3961_396156

/-- The number of boxes in a carton of pencils -/
def boxes_per_pencil_carton : ℕ := sorry

/-- The number of cartons of pencils bought -/
def pencil_cartons : ℕ := 20

/-- The cost of one box of pencils in dollars -/
def pencil_box_cost : ℕ := 2

/-- The number of cartons of markers bought -/
def marker_cartons : ℕ := 10

/-- The number of boxes in a carton of markers -/
def boxes_per_marker_carton : ℕ := 5

/-- The cost of one box of markers in dollars -/
def marker_box_cost : ℕ := 4

/-- The total amount spent by the school in dollars -/
def total_spent : ℕ := 600

theorem pencil_boxes_per_carton :
  boxes_per_pencil_carton = 10 ∧
  pencil_cartons * boxes_per_pencil_carton * pencil_box_cost +
  marker_cartons * boxes_per_marker_carton * marker_box_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_pencil_boxes_per_carton_l3961_396156


namespace NUMINAMATH_CALUDE_point_zero_three_on_graph_minimum_at_minus_two_point_one_seven_not_on_graph_l3961_396176

/-- Quadratic function f(x) = (x + 2)^2 - 1 -/
def f (x : ℝ) : ℝ := (x + 2)^2 - 1

/-- The point (0, 3) lies on the graph of f -/
theorem point_zero_three_on_graph : f 0 = 3 := by sorry

/-- The function f has a minimum value of -1 when x = -2 -/
theorem minimum_at_minus_two : 
  (∀ x : ℝ, f x ≥ -1) ∧ f (-2) = -1 := by sorry

/-- The point P(1, 7) does not lie on the graph of f -/
theorem point_one_seven_not_on_graph : f 1 ≠ 7 := by sorry

end NUMINAMATH_CALUDE_point_zero_three_on_graph_minimum_at_minus_two_point_one_seven_not_on_graph_l3961_396176


namespace NUMINAMATH_CALUDE_reciprocal_sum_contains_two_l3961_396147

theorem reciprocal_sum_contains_two (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = 1 →
  a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_contains_two_l3961_396147


namespace NUMINAMATH_CALUDE_joes_bath_shop_soap_sales_l3961_396188

theorem joes_bath_shop_soap_sales : ∃ n : ℕ, n > 0 ∧ n % 7 = 0 ∧ n % 23 = 0 ∧ ∀ m : ℕ, m > 0 → m % 7 = 0 → m % 23 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_joes_bath_shop_soap_sales_l3961_396188


namespace NUMINAMATH_CALUDE_probability_of_purple_marble_l3961_396152

theorem probability_of_purple_marble (blue_prob green_prob purple_prob : ℝ) :
  blue_prob = 0.25 →
  green_prob = 0.35 →
  blue_prob + green_prob + purple_prob = 1 →
  purple_prob = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_purple_marble_l3961_396152


namespace NUMINAMATH_CALUDE_circle_condition_m_set_l3961_396116

/-- A set in R² represents a circle if it can be expressed as 
    {(x, y) | (x - h)² + (y - k)² = r²} for some h, k, and r > 0 -/
def IsCircle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ h k r, r > 0 ∧ S = {p : ℝ × ℝ | (p.1 - h)^2 + (p.2 - k)^2 = r^2}

/-- The set of points (x, y) satisfying the given equation -/
def S (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*m*p.1 - 2*m*p.2 + 2*m^2 + m - 1 = 0}

theorem circle_condition (m : ℝ) : IsCircle (S m) → m < 1 := by
  sorry

theorem m_set : {m : ℝ | IsCircle (S m)} = {m : ℝ | m < 1} := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_m_set_l3961_396116


namespace NUMINAMATH_CALUDE_least_positive_period_is_30_l3961_396124

/-- A function satisfying the given condition -/
def PeriodicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least positive period of a function -/
def IsLeastPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ IsPeriod f p ∧ ∀ q : ℝ, 0 < q ∧ q < p → ¬IsPeriod f q

theorem least_positive_period_is_30 :
  ∀ f : ℝ → ℝ, PeriodicFunction f → IsLeastPositivePeriod f 30 :=
sorry

end NUMINAMATH_CALUDE_least_positive_period_is_30_l3961_396124


namespace NUMINAMATH_CALUDE_perpendicular_edges_count_l3961_396125

/-- A cube is a three-dimensional shape with 6 square faces -/
structure Cube where
  -- Add necessary fields here

/-- An edge of a cube -/
structure Edge (c : Cube) where
  -- Add necessary fields here

/-- Predicate to check if two edges are perpendicular -/
def perpendicular (c : Cube) (e1 e2 : Edge c) : Prop :=
  sorry

theorem perpendicular_edges_count (c : Cube) (e : Edge c) :
  (∃ (s : Finset (Edge c)), s.card = 8 ∧ ∀ e' ∈ s, perpendicular c e e') ∧
  ¬∃ (s : Finset (Edge c)), s.card > 8 ∧ ∀ e' ∈ s, perpendicular c e e' :=
sorry

end NUMINAMATH_CALUDE_perpendicular_edges_count_l3961_396125


namespace NUMINAMATH_CALUDE_opposite_to_Y_is_Z_l3961_396189

-- Define the faces of the cube
inductive Face : Type
  | V | W | X | Y | Z

-- Define the cube structure
structure Cube where
  faces : List Face
  bottom : Face
  right_of_bottom : Face

-- Define the opposite face relation
def opposite_face (c : Cube) (f : Face) : Face :=
  sorry

-- Theorem statement
theorem opposite_to_Y_is_Z (c : Cube) :
  c.bottom = Face.X →
  c.right_of_bottom = Face.W →
  c.faces = [Face.V, Face.W, Face.X, Face.Y, Face.Z] →
  opposite_face c Face.Y = Face.Z := by
  sorry

end NUMINAMATH_CALUDE_opposite_to_Y_is_Z_l3961_396189


namespace NUMINAMATH_CALUDE_pascal_sum_29_l3961_396158

/-- Number of elements in a row of Pascal's Triangle -/
def pascal_row_count (n : ℕ) : ℕ := n + 1

/-- Sum of elements in Pascal's Triangle from row 0 to row n -/
def pascal_sum (n : ℕ) : ℕ :=
  (n + 1) * (n + 2) / 2

theorem pascal_sum_29 : pascal_sum 29 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_sum_29_l3961_396158


namespace NUMINAMATH_CALUDE_lola_baked_eight_pies_l3961_396112

/-- Represents the number of pastries baked by Lola and Lulu -/
structure Pastries where
  lola_cupcakes : ℕ
  lola_poptarts : ℕ
  lola_pies : ℕ
  lulu_cupcakes : ℕ
  lulu_poptarts : ℕ
  lulu_pies : ℕ

/-- The total number of pastries baked by both Lola and Lulu -/
def total_pastries (p : Pastries) : ℕ :=
  p.lola_cupcakes + p.lola_poptarts + p.lola_pies +
  p.lulu_cupcakes + p.lulu_poptarts + p.lulu_pies

/-- Theorem stating that Lola baked 8 blueberry pies -/
theorem lola_baked_eight_pies (p : Pastries)
  (h1 : p.lola_cupcakes = 13)
  (h2 : p.lola_poptarts = 10)
  (h3 : p.lulu_cupcakes = 16)
  (h4 : p.lulu_poptarts = 12)
  (h5 : p.lulu_pies = 14)
  (h6 : total_pastries p = 73) :
  p.lola_pies = 8 := by
  sorry

end NUMINAMATH_CALUDE_lola_baked_eight_pies_l3961_396112


namespace NUMINAMATH_CALUDE_probability_for_given_box_l3961_396160

/-- Represents the contents of the box -/
structure Box where
  blue : Nat
  red : Nat
  green : Nat

/-- The probability of drawing all blue chips before both green chips -/
def probability_all_blue_before_both_green (box : Box) : Rat :=
  17/36

/-- Theorem stating the probability for the given box configuration -/
theorem probability_for_given_box :
  let box : Box := { blue := 4, red := 3, green := 2 }
  probability_all_blue_before_both_green box = 17/36 := by
  sorry

end NUMINAMATH_CALUDE_probability_for_given_box_l3961_396160


namespace NUMINAMATH_CALUDE_smallest_binary_multiple_of_15_l3961_396138

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_multiple_of_15 :
  ∀ T : ℕ, 
    T > 0 → 
    is_binary_number T → 
    T % 15 = 0 → 
    ∀ X : ℕ, 
      X = T / 15 → 
      X ≥ 74 :=
sorry

end NUMINAMATH_CALUDE_smallest_binary_multiple_of_15_l3961_396138


namespace NUMINAMATH_CALUDE_final_cell_count_l3961_396113

/-- Calculates the number of cells after a given number of days, 
    given an initial population and a tripling period. -/
def cell_count (initial_cells : ℕ) (tripling_period : ℕ) (total_days : ℕ) : ℕ :=
  initial_cells * (3 ^ (total_days / tripling_period))

/-- Theorem stating that given 5 initial cells, tripling every 3 days for 9 days, 
    the final cell count is 135. -/
theorem final_cell_count : cell_count 5 3 9 = 135 := by
  sorry

#eval cell_count 5 3 9

end NUMINAMATH_CALUDE_final_cell_count_l3961_396113


namespace NUMINAMATH_CALUDE_jane_farm_eggs_l3961_396132

/-- Calculates the number of eggs laid per chicken per week -/
def eggs_per_chicken_per_week (num_chickens : ℕ) (price_per_dozen : ℚ) (total_revenue : ℚ) (num_weeks : ℕ) : ℚ :=
  (total_revenue / price_per_dozen * 12) / (num_chickens * num_weeks)

theorem jane_farm_eggs : 
  let num_chickens : ℕ := 10
  let price_per_dozen : ℚ := 2
  let total_revenue : ℚ := 20
  let num_weeks : ℕ := 2
  eggs_per_chicken_per_week num_chickens price_per_dozen total_revenue num_weeks = 6 := by
  sorry

#eval eggs_per_chicken_per_week 10 2 20 2

end NUMINAMATH_CALUDE_jane_farm_eggs_l3961_396132


namespace NUMINAMATH_CALUDE_find_liar_in_17_questions_l3961_396149

/-- Represents a person who can be either a knight or a liar -/
inductive Person
  | knight : Person
  | liar : Person

/-- Represents the response to a question -/
inductive Response
  | yes : Response
  | no : Response

/-- A function that simulates asking a question to a person -/
def ask (p : Person) (cardNumber : Nat) (askedNumber : Nat) : Response :=
  match p with
  | Person.knight => if cardNumber = askedNumber then Response.yes else Response.no
  | Person.liar => if cardNumber ≠ askedNumber then Response.yes else Response.no

/-- The main theorem statement -/
theorem find_liar_in_17_questions 
  (people : Fin 10 → Person) 
  (cards : Fin 10 → Nat) 
  (h1 : ∃! i, people i = Person.liar) 
  (h2 : ∀ i j, i ≠ j → cards i ≠ cards j) 
  (h3 : ∀ i, cards i ∈ Set.range (fun n : Nat => n + 1) ∩ Set.range (fun n : Nat => 11 - n)) :
  ∃ (strategy : Nat → Fin 10 × Nat), 
    (∀ n, n < 17 → (strategy n).2 ∈ Set.range (fun n : Nat => n + 1) ∩ Set.range (fun n : Nat => 11 - n)) →
    ∃ (result : Fin 10), 
      (∀ i, i ≠ result → people i = Person.knight) ∧ 
      (people result = Person.liar) :=
sorry

end NUMINAMATH_CALUDE_find_liar_in_17_questions_l3961_396149


namespace NUMINAMATH_CALUDE_shorter_tank_radius_l3961_396154

/-- Given two cylindrical tanks with equal volumes, where one tank is twice as tall as the other,
    and the radius of the taller tank is 10 units, the radius of the shorter tank is 10√2 units. -/
theorem shorter_tank_radius (h : ℝ) (h_pos : h > 0) : 
  let v := π * (10^2) * (2*h)  -- Volume of the taller tank
  let r := Real.sqrt 200       -- Radius of the shorter tank
  v = π * r^2 * h              -- Volumes are equal
  → r = 10 * Real.sqrt 2       -- Radius of the shorter tank is 10√2
  := by sorry

end NUMINAMATH_CALUDE_shorter_tank_radius_l3961_396154


namespace NUMINAMATH_CALUDE_college_students_count_l3961_396165

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 210) :
  boys + girls = 546 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l3961_396165


namespace NUMINAMATH_CALUDE_min_lines_for_200_intersections_l3961_396142

theorem min_lines_for_200_intersections :
  ∃ n : ℕ,
    n > 0 ∧
    n * (n - 1) / 2 = 200 ∧
    ∀ m : ℕ, m > 0 → m * (m - 1) / 2 = 200 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_min_lines_for_200_intersections_l3961_396142


namespace NUMINAMATH_CALUDE_managers_salary_l3961_396199

theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (avg_increase : ℕ) (managers_salary : ℕ) : 
  num_employees = 18 → 
  avg_salary = 2000 → 
  avg_increase = 200 → 
  (num_employees * avg_salary + managers_salary) / (num_employees + 1) = avg_salary + avg_increase →
  managers_salary = 5800 := by
sorry

end NUMINAMATH_CALUDE_managers_salary_l3961_396199


namespace NUMINAMATH_CALUDE_lewis_weekly_rent_l3961_396137

/-- Calculates the weekly rent given the total rent and number of weeks -/
def weekly_rent (total_rent : ℕ) (num_weeks : ℕ) : ℚ :=
  (total_rent : ℚ) / (num_weeks : ℚ)

/-- Theorem: The weekly rent for Lewis during harvest season -/
theorem lewis_weekly_rent :
  weekly_rent 527292 1359 = 388 := by
  sorry

end NUMINAMATH_CALUDE_lewis_weekly_rent_l3961_396137


namespace NUMINAMATH_CALUDE_largest_digit_sum_l3961_396170

theorem largest_digit_sum (a b c : ℕ) (y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, c are digits
  (100 * a + 10 * b + c = 800 / y) →  -- 0.abc = 1/y
  (0 < y ∧ y ≤ 10) →  -- 0 < y ≤ 10
  (∃ (a' b' c' : ℕ), a' < 10 ∧ b' < 10 ∧ c' < 10 ∧ 
    100 * a' + 10 * b' + c' = 800 / y ∧ 
    a' + b' + c' = 8 ∧
    ∀ (x y z : ℕ), x < 10 → y < 10 → z < 10 → 
      100 * x + 10 * y + z = 800 / y → x + y + z ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l3961_396170


namespace NUMINAMATH_CALUDE_a_4_value_l3961_396163

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem a_4_value (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 4 / a 2 - a 3 = 0 →
  a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_a_4_value_l3961_396163


namespace NUMINAMATH_CALUDE_watch_cost_price_l3961_396114

/-- The cost price of a watch satisfying certain conditions -/
theorem watch_cost_price : ∃ (cp : ℝ), 
  cp = 687.5 ∧ 
  (cp * 0.725 + 275 = cp * 1.125) := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l3961_396114


namespace NUMINAMATH_CALUDE_shipping_cost_correct_l3961_396167

/-- The cost function for shipping packages -/
def shipping_cost (W : ℕ) : ℝ :=
  5 + 4 * (W - 1)

/-- Theorem stating the correctness of the shipping cost formula -/
theorem shipping_cost_correct (W : ℕ) (h : W ≥ 2) :
  shipping_cost W = 5 + 4 * (W - 1) :=
by sorry

end NUMINAMATH_CALUDE_shipping_cost_correct_l3961_396167


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3961_396185

theorem system_solution_ratio (x y c d : ℝ) : 
  x ≠ 0 → y ≠ 0 → d ≠ 0 →
  8 * x - 6 * y = c →
  12 * y - 18 * x = d →
  c / d = -4 / 9 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3961_396185


namespace NUMINAMATH_CALUDE_max_value_implies_ratio_l3961_396135

/-- Given a cubic function f(x) with a maximum at x=1, prove that a/b = -2/3 --/
theorem max_value_implies_ratio (a b : ℝ) :
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x - a^2 - 7*a
  (∀ x, f x ≤ f 1) ∧ (f 1 = 10) →
  a / b = -2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_ratio_l3961_396135


namespace NUMINAMATH_CALUDE_smallest_product_l3961_396183

def S : Finset ℕ := {3, 5, 7, 9, 11, 13}

theorem smallest_product (a b c d : ℕ) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  (a + b) * (c + d) ≥ 128 := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_l3961_396183


namespace NUMINAMATH_CALUDE_pi_is_real_l3961_396110

-- Define π as a real number representing the ratio of a circle's circumference to its diameter
noncomputable def π : ℝ := Real.pi

-- Theorem stating that π is a real number
theorem pi_is_real : π ∈ Set.univ := by sorry

end NUMINAMATH_CALUDE_pi_is_real_l3961_396110


namespace NUMINAMATH_CALUDE_complex_number_modulus_l3961_396175

theorem complex_number_modulus (i : ℂ) (Z : ℂ) (a : ℝ) :
  i^2 = -1 →
  Z = i * (3 - a * i) →
  Complex.abs Z = 5 →
  a = 4 ∨ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l3961_396175


namespace NUMINAMATH_CALUDE_quadratic_vertex_not_minus_one_minus_three_a_l3961_396111

/-- Given a quadratic function y = ax^2 + 2ax - 3a where a > 0,
    prove that its vertex coordinates are not (-1, -3a) -/
theorem quadratic_vertex_not_minus_one_minus_three_a (a : ℝ) (h : a > 0) :
  ∃ (x y : ℝ), (y = a*x^2 + 2*a*x - 3*a) ∧ 
  (∀ x' : ℝ, a*x'^2 + 2*a*x' - 3*a ≥ y) ∧
  (x ≠ -1 ∨ y ≠ -3*a) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_not_minus_one_minus_three_a_l3961_396111


namespace NUMINAMATH_CALUDE_min_effort_for_mop_l3961_396143

/-- Represents the effort and points for each exam --/
structure ExamEffort :=
  (effort : ℕ)
  (points : ℕ)

/-- Defines the problem of Alex making MOP --/
def MakeMOP (amc : ExamEffort) (aime : ExamEffort) (usamo : ExamEffort) : Prop :=
  let total_points := amc.points + aime.points
  let total_effort := amc.effort + aime.effort + usamo.effort
  total_points ≥ 200 ∧ usamo.points ≥ 21 ∧ total_effort = 320

/-- Theorem stating the minimum effort required for Alex to make MOP --/
theorem min_effort_for_mop :
  ∃ (amc aime usamo : ExamEffort),
    amc.effort = 3 * (amc.points / 6) ∧
    aime.effort = 7 * (aime.points / 10) ∧
    usamo.effort = 10 * usamo.points ∧
    MakeMOP amc aime usamo ∧
    ∀ (amc' aime' usamo' : ExamEffort),
      amc'.effort = 3 * (amc'.points / 6) →
      aime'.effort = 7 * (aime'.points / 10) →
      usamo'.effort = 10 * usamo'.points →
      MakeMOP amc' aime' usamo' →
      amc'.effort + aime'.effort + usamo'.effort ≥ 320 :=
by
  sorry

end NUMINAMATH_CALUDE_min_effort_for_mop_l3961_396143


namespace NUMINAMATH_CALUDE_ellipse_intersection_property_l3961_396186

/-- An ellipse with semi-major axis 2 and semi-minor axis √3 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + (p.2^2 / 3) = 1}

/-- Check if two points are symmetric about the x-axis -/
def symmetric_about_x (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- The intersection point of two lines -/
def intersection (p₁ p₂ q₁ q₂ : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: For the given ellipse, if points satisfy the specified conditions,
    then the x-coordinates of A and B multiply to give 4 -/
theorem ellipse_intersection_property
  (d e : ℝ × ℝ)
  (h_d : d ∈ Ellipse)
  (h_e : e ∈ Ellipse)
  (h_sym : symmetric_about_x d e)
  (x₁ x₂ : ℝ)
  (h_not_tangent : ∀ y, (x₁, y) ≠ d)
  (c : ℝ × ℝ)
  (h_c_intersection : c = intersection d (x₁, 0) e (x₂, 0))
  (h_c_on_ellipse : c ∈ Ellipse) :
  x₁ * x₂ = 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_property_l3961_396186


namespace NUMINAMATH_CALUDE_unique_intercept_line_l3961_396109

/-- A line passing through a point with equal absolute horizontal and vertical intercepts -/
structure InterceptLine where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept
  point_condition : 4 = m * 1 + b  -- line passes through (1, 4)
  intercept_condition : |m| = |b|  -- equal absolute intercepts

/-- There exists a unique line passing through (1, 4) with equal absolute horizontal and vertical intercepts -/
theorem unique_intercept_line : ∃! l : InterceptLine, True :=
  sorry

end NUMINAMATH_CALUDE_unique_intercept_line_l3961_396109


namespace NUMINAMATH_CALUDE_age_ratio_in_one_year_l3961_396123

/-- Represents the current ages of Jack and Alex -/
structure Ages where
  jack : ℕ
  alex : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  (ages.jack - 3 = 2 * (ages.alex - 3)) ∧ 
  (ages.jack - 5 = 3 * (ages.alex - 5))

/-- The future ratio of their ages will be 3:2 -/
def future_ratio (ages : Ages) (years : ℕ) : Prop :=
  3 * (ages.alex + years) = 2 * (ages.jack + years)

/-- The theorem to be proved -/
theorem age_ratio_in_one_year (ages : Ages) :
  age_conditions ages → ∃ (y : ℕ), y = 1 ∧ future_ratio ages y :=
sorry

end NUMINAMATH_CALUDE_age_ratio_in_one_year_l3961_396123


namespace NUMINAMATH_CALUDE_school_children_count_l3961_396155

theorem school_children_count :
  let absent_children : ℕ := 160
  let total_bananas : ℕ → ℕ → ℕ := λ present absent => 2 * present + 2 * absent
  let extra_bananas : ℕ → ℕ → ℕ := λ present absent => 2 * present
  let boys_bananas : ℕ → ℕ := λ total => 3 * (total / 4)
  let girls_bananas : ℕ → ℕ := λ total => total / 4
  ∃ (present_children : ℕ),
    total_bananas present_children absent_children = 
      total_bananas present_children present_children + extra_bananas present_children absent_children ∧
    boys_bananas (total_bananas present_children absent_children) + 
      girls_bananas (total_bananas present_children absent_children) = 
      total_bananas present_children absent_children ∧
    present_children + absent_children = 6560 :=
by sorry

end NUMINAMATH_CALUDE_school_children_count_l3961_396155


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3961_396146

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  first_sum : a 1 + a 8 = 10
  second_sum : a 2 + a 9 = 18

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) :
  common_difference seq = 4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3961_396146


namespace NUMINAMATH_CALUDE_largest_prime_form_is_seven_l3961_396171

def is_largest_prime_form (p : ℕ) : Prop :=
  Prime p ∧
  (∃ n : ℕ, Prime n ∧ p = 2^n + n^2 - 1) ∧
  p < 100 ∧
  ∀ q : ℕ, q ≠ p → Prime q → (∃ m : ℕ, Prime m ∧ q = 2^m + m^2 - 1) → q < 100 → q < p

theorem largest_prime_form_is_seven : is_largest_prime_form 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_form_is_seven_l3961_396171


namespace NUMINAMATH_CALUDE_no_positive_integers_satisfy_divisibility_l3961_396179

theorem no_positive_integers_satisfy_divisibility : ¬ ∃ (a b c : ℕ+), (3 * (a * b + b * c + c * a)) ∣ (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integers_satisfy_divisibility_l3961_396179


namespace NUMINAMATH_CALUDE_sequence_matches_first_10_terms_l3961_396100

/-- The sequence defined by a(n) = n(n-1) -/
def a (n : ℕ) : ℕ := n * (n - 1)

/-- The first 10 terms of the sequence -/
def first_10_terms : List ℕ := [0, 2, 6, 12, 20, 30, 42, 56, 72, 90]

theorem sequence_matches_first_10_terms :
  (List.range 10).map (fun i => a (i + 1)) = first_10_terms := by sorry

end NUMINAMATH_CALUDE_sequence_matches_first_10_terms_l3961_396100


namespace NUMINAMATH_CALUDE_johns_hats_cost_l3961_396182

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of weeks John can wear a different hat each day -/
def weeks_of_different_hats : ℕ := 2

/-- The cost of each hat in dollars -/
def cost_per_hat : ℕ := 50

/-- The total cost of John's hats -/
def total_cost : ℕ := weeks_of_different_hats * days_in_week * cost_per_hat

theorem johns_hats_cost : total_cost = 700 := by
  sorry

end NUMINAMATH_CALUDE_johns_hats_cost_l3961_396182


namespace NUMINAMATH_CALUDE_angle_measure_when_complement_is_half_supplement_l3961_396161

theorem angle_measure_when_complement_is_half_supplement :
  ∀ x : ℝ,
  (x > 0) →
  (x ≤ 180) →
  (90 - x = (180 - x) / 2) →
  x = 90 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_when_complement_is_half_supplement_l3961_396161


namespace NUMINAMATH_CALUDE_speed_ratio_is_one_third_l3961_396196

/-- The problem setup for two moving objects A and B -/
structure MovementProblem where
  vA : ℝ  -- Speed of A
  vB : ℝ  -- Speed of B
  initialDistance : ℝ  -- Initial distance of B from O

/-- The conditions of the problem -/
def satisfiesConditions (p : MovementProblem) : Prop :=
  p.initialDistance = 300 ∧
  p.vA = |p.initialDistance - p.vB| ∧
  7 * p.vA = |p.initialDistance - 7 * p.vB|

/-- The theorem to be proved -/
theorem speed_ratio_is_one_third (p : MovementProblem) 
  (h : satisfiesConditions p) : p.vA / p.vB = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_is_one_third_l3961_396196


namespace NUMINAMATH_CALUDE_division_remainder_l3961_396198

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 11 →
  divisor = 4 →
  quotient = 2 →
  dividend = divisor * quotient + remainder →
  remainder = 3 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l3961_396198


namespace NUMINAMATH_CALUDE_percentage_increase_problem_l3961_396139

theorem percentage_increase_problem : 
  let initial := 100
  let after_first_increase := initial * (1 + 0.2)
  let final := after_first_increase * (1 + 0.5)
  final = 180 := by sorry

end NUMINAMATH_CALUDE_percentage_increase_problem_l3961_396139


namespace NUMINAMATH_CALUDE_not_prime_23021_pow_377_minus_1_l3961_396122

theorem not_prime_23021_pow_377_minus_1 : ¬ Nat.Prime (23021^377 - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_23021_pow_377_minus_1_l3961_396122


namespace NUMINAMATH_CALUDE_logical_conditions_l3961_396181

-- Define a proposition type to represent logical statements
variable (A B : Prop)

-- Define sufficient condition
def is_sufficient_condition (A B : Prop) : Prop :=
  A → B

-- Define necessary condition
def is_necessary_condition (A B : Prop) : Prop :=
  B → A

-- Define necessary and sufficient condition
def is_necessary_and_sufficient_condition (A B : Prop) : Prop :=
  (A → B) ∧ (B → A)

-- Theorem statement
theorem logical_conditions :
  (is_sufficient_condition A B ↔ (A → B)) ∧
  (is_necessary_condition A B ↔ (B → A)) ∧
  (is_necessary_and_sufficient_condition A B ↔ ((A → B) ∧ (B → A))) :=
by sorry

end NUMINAMATH_CALUDE_logical_conditions_l3961_396181


namespace NUMINAMATH_CALUDE_min_q_for_min_a2016_l3961_396141

theorem min_q_for_min_a2016 (a : ℕ → ℕ) (q : ℚ) :
  (∀ n, 1 ≤ n ∧ n ≤ 2016 → a n = a 1 * q ^ (n - 1)) →
  1 < q ∧ q < 2 →
  (∀ r, 1 < r ∧ r < 2 → a 2016 ≤ (a 1 : ℚ) * r ^ 2015) →
  q = 6/5 :=
sorry

end NUMINAMATH_CALUDE_min_q_for_min_a2016_l3961_396141


namespace NUMINAMATH_CALUDE_red_rows_in_specific_grid_l3961_396195

/-- Represents the grid coloring problem -/
structure GridColoring where
  total_rows : ℕ
  squares_per_row : ℕ
  blue_rows : ℕ
  green_squares : ℕ
  red_squares_per_row : ℕ

/-- Calculates the number of red rows in the grid -/
def red_rows (g : GridColoring) : ℕ :=
  let total_squares := g.total_rows * g.squares_per_row
  let blue_squares := g.blue_rows * g.squares_per_row
  let red_squares := total_squares - blue_squares - g.green_squares
  red_squares / g.red_squares_per_row

/-- Theorem stating the number of red rows in the specific problem -/
theorem red_rows_in_specific_grid :
  let g : GridColoring := {
    total_rows := 10,
    squares_per_row := 15,
    blue_rows := 4,
    green_squares := 66,
    red_squares_per_row := 6
  }
  red_rows g = 4 := by sorry

end NUMINAMATH_CALUDE_red_rows_in_specific_grid_l3961_396195


namespace NUMINAMATH_CALUDE_square_diff_fourth_power_l3961_396127

theorem square_diff_fourth_power : (7^2 - 5^2)^4 = 331776 := by sorry

end NUMINAMATH_CALUDE_square_diff_fourth_power_l3961_396127
