import Mathlib

namespace NUMINAMATH_CALUDE_negative_difference_l2777_277768

theorem negative_difference (P Q R S T : ℝ) 
  (h1 : P < Q) (h2 : Q < R) (h3 : R < S) (h4 : S < T) 
  (h5 : P ≠ 0) (h6 : Q ≠ 0) (h7 : R ≠ 0) (h8 : S ≠ 0) (h9 : T ≠ 0) : 
  P - Q < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l2777_277768


namespace NUMINAMATH_CALUDE_line_slope_l2777_277702

theorem line_slope (m n p K : ℝ) (h1 : p = 0.3333333333333333) : 
  (m = K * n + 5 ∧ m + 2 = K * (n + p) + 5) → K = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l2777_277702


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l2777_277742

theorem cuboid_surface_area 
  (x y z : ℝ) 
  (edge_sum : 4*x + 4*y + 4*z = 160) 
  (diagonal : x^2 + y^2 + z^2 = 25^2) : 
  2*(x*y + y*z + z*x) = 975 := by
sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l2777_277742


namespace NUMINAMATH_CALUDE_largest_package_size_l2777_277731

theorem largest_package_size (alex_markers becca_markers charlie_markers : ℕ) 
  (h_alex : alex_markers = 36)
  (h_becca : becca_markers = 45)
  (h_charlie : charlie_markers = 60) :
  Nat.gcd alex_markers (Nat.gcd becca_markers charlie_markers) = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l2777_277731


namespace NUMINAMATH_CALUDE_sara_letters_ratio_l2777_277796

theorem sara_letters_ratio (january february total : ℕ) 
  (h1 : january = 6)
  (h2 : february = 9)
  (h3 : total = 33) :
  (total - january - february) / january = 3 := by
  sorry

end NUMINAMATH_CALUDE_sara_letters_ratio_l2777_277796


namespace NUMINAMATH_CALUDE_sum_difference_equality_l2777_277777

theorem sum_difference_equality : 291 + 503 - 91 + 492 - 103 - 392 = 700 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equality_l2777_277777


namespace NUMINAMATH_CALUDE_bulbs_needed_l2777_277752

/-- Represents the number of bulbs required for each type of ceiling light. -/
structure BulbRequirement where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of each type of ceiling light. -/
structure CeilingLights where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of bulbs needed given the requirements and quantities. -/
def totalBulbs (req : BulbRequirement) (lights : CeilingLights) : Nat :=
  req.small * lights.small + req.medium * lights.medium + req.large * lights.large

/-- The main theorem stating the total number of bulbs needed. -/
theorem bulbs_needed :
  ∀ (req : BulbRequirement) (lights : CeilingLights),
    req.small = 1 ∧ req.medium = 2 ∧ req.large = 3 ∧
    lights.medium = 12 ∧
    lights.large = 2 * lights.medium ∧
    lights.small = lights.medium + 10 →
    totalBulbs req lights = 118 := by
  sorry


end NUMINAMATH_CALUDE_bulbs_needed_l2777_277752


namespace NUMINAMATH_CALUDE_series_sum_l2777_277740

theorem series_sum : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l2777_277740


namespace NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l2777_277748

/-- The set A -/
def A : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}

/-- The set B -/
def B : Set (ℝ × ℝ) := {p | (p.1 - 5) / (p.1 - 4) = 1}

/-- The symmetric difference of two sets -/
def symmetricDifference (X Y : Set α) : Set α :=
  (X \ Y) ∪ (Y \ X)

/-- Theorem: The symmetric difference of A and B -/
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {p : ℝ × ℝ | p.2 = p.1 + 1 ∧ p.1 ≠ 4} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l2777_277748


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2777_277715

theorem parallel_vectors_k_value (k : ℝ) :
  let a : Fin 2 → ℝ := ![k, Real.sqrt 2]
  let b : Fin 2 → ℝ := ![2, -2]
  (∃ (c : ℝ), a = c • b) →
  k = -Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2777_277715


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_l2777_277704

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_series : ℕ → ℕ
  | 0 => 0
  | n + 1 => sum_series n + if (n + 1) % 3 = 0 ∧ n + 1 ≤ 9 then 2 * factorial (n + 1) else 0

theorem last_two_digits_of_sum : last_two_digits (sum_series 99) = 24 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_l2777_277704


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2777_277787

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = (3125 : ℚ) / 10000 :=
by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2777_277787


namespace NUMINAMATH_CALUDE_coin_difference_is_ten_l2777_277789

def coin_values : List ℕ := [5, 10, 25, 50]
def target_amount : ℕ := 60

def min_coins (values : List ℕ) (target : ℕ) : ℕ := sorry
def max_coins (values : List ℕ) (target : ℕ) : ℕ := sorry

theorem coin_difference_is_ten :
  max_coins coin_values target_amount - min_coins coin_values target_amount = 10 := by sorry

end NUMINAMATH_CALUDE_coin_difference_is_ten_l2777_277789


namespace NUMINAMATH_CALUDE_xyz_value_l2777_277795

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19)
  (h3 : x + y + z = 7) : 
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2777_277795


namespace NUMINAMATH_CALUDE_h2o_required_for_reaction_l2777_277700

-- Define the chemical reaction
def chemical_reaction (NaH H2O NaOH H2 : ℕ) : Prop :=
  NaH = H2O ∧ NaH = NaOH ∧ NaH = H2

-- Define the problem statement
theorem h2o_required_for_reaction (NaH : ℕ) (h : NaH = 2) :
  ∃ H2O : ℕ, chemical_reaction NaH H2O NaH NaH ∧ H2O = 2 :=
by sorry

end NUMINAMATH_CALUDE_h2o_required_for_reaction_l2777_277700


namespace NUMINAMATH_CALUDE_geometric_sequence_a2_l2777_277744

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_a2 (a : ℕ → ℚ) 
  (h1 : geometric_sequence a) 
  (h2 : a 1 + a 3 = 4) : 
  a 2 = 8/5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a2_l2777_277744


namespace NUMINAMATH_CALUDE_peanut_butter_servings_l2777_277728

theorem peanut_butter_servings 
  (jar_content : ℚ) 
  (serving_size : ℚ) 
  (h1 : jar_content = 35 + 4/5)
  (h2 : serving_size = 5/2) : 
  jar_content / serving_size = 14 + 8/25 := by
sorry

end NUMINAMATH_CALUDE_peanut_butter_servings_l2777_277728


namespace NUMINAMATH_CALUDE_union_equality_implies_range_l2777_277774

def A : Set ℝ := {x | |x| > 1}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem union_equality_implies_range (a : ℝ) : A ∪ B a = A → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_range_l2777_277774


namespace NUMINAMATH_CALUDE_range_of_a_l2777_277718

-- Define the propositions P and Q as functions of a
def P (a : ℝ) : Prop := ∀ x > 0, Monotone (fun x => Real.log x / Real.log a)

def Q (a : ℝ) : Prop := ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : P a ∨ Q a) 
  (h2 : ¬(P a ∧ Q a)) : 
  a > 2 ∨ (-2 < a ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2777_277718


namespace NUMINAMATH_CALUDE_f_2011_is_zero_l2777_277716

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2011_is_zero (f : ℝ → ℝ) (h_odd : is_odd f) (h_period : ∀ x, f (x + 1) = -f x) : f 2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2011_is_zero_l2777_277716


namespace NUMINAMATH_CALUDE_largest_square_and_rectangle_in_right_triangle_l2777_277771

/-- Given a right triangle ABC with legs AC = a and CB = b, prove:
    1. The side length of the largest square (with vertex C) that lies entirely within the triangle ABC is ab/(a+b)
    2. The dimensions of the largest rectangle (with vertex C) that lies entirely within the triangle ABC are a/2 and b/2 -/
theorem largest_square_and_rectangle_in_right_triangle 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let square_side := a * b / (a + b)
  let rectangle_width := a / 2
  let rectangle_height := b / 2
  (∀ s, s > 0 → s * s ≤ square_side * square_side) ∧
  (∀ w h, w > 0 → h > 0 → w * h ≤ rectangle_width * rectangle_height) := by
  sorry

end NUMINAMATH_CALUDE_largest_square_and_rectangle_in_right_triangle_l2777_277771


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l2777_277781

theorem quadratic_root_m_value :
  ∀ m : ℝ, (1 : ℝ)^2 + m * 1 + 2 = 0 → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l2777_277781


namespace NUMINAMATH_CALUDE_part1_part2_l2777_277780

-- Define the quadratic inequality
def quadratic_inequality (a b x : ℝ) : Prop := a * x^2 + b * x - 1 ≥ 0

-- Part 1
theorem part1 (a b : ℝ) :
  (∀ x, quadratic_inequality a b x ↔ (3 ≤ x ∧ x ≤ 4)) →
  a + b = 1/2 := by sorry

-- Part 2
theorem part2 (b : ℝ) :
  let solution_set := {x : ℝ | quadratic_inequality (-1) b x}
  if -2 < b ∧ b < 2 then
    solution_set = ∅
  else if b = -2 then
    solution_set = {-1}
  else if b = 2 then
    solution_set = {1}
  else
    ∃ (l u : ℝ), l = (b - Real.sqrt (b^2 - 4)) / 2 ∧
                 u = (b + Real.sqrt (b^2 - 4)) / 2 ∧
                 solution_set = {x | l ≤ x ∧ x ≤ u} := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2777_277780


namespace NUMINAMATH_CALUDE_total_amount_is_70_l2777_277739

/-- Represents the distribution of money among three people -/
structure Distribution where
  x : ℚ  -- x's share in rupees
  y : ℚ  -- y's share in rupees
  z : ℚ  -- z's share in rupees

/-- Checks if a distribution satisfies the given conditions -/
def is_valid_distribution (d : Distribution) : Prop :=
  d.y = 0.45 * d.x ∧ d.z = 0.3 * d.x ∧ d.y = 18

/-- The theorem to prove -/
theorem total_amount_is_70 (d : Distribution) :
  is_valid_distribution d → d.x + d.y + d.z = 70 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_70_l2777_277739


namespace NUMINAMATH_CALUDE_uninsured_part_time_percentage_l2777_277782

/-- Represents the survey data and calculates the percentage of uninsured part-time employees -/
def survey_data (total : ℕ) (uninsured : ℕ) (part_time : ℕ) (neither_prob : ℚ) : ℚ :=
  let uninsured_part_time := total - (neither_prob * total).num - uninsured - part_time
  (uninsured_part_time / uninsured) * 100

/-- Theorem stating that given the survey conditions, the percentage of uninsured employees
    who work part-time is approximately 12.5% -/
theorem uninsured_part_time_percentage :
  let result := survey_data 330 104 54 (559606060606060606 / 1000000000000000000)
  ∃ (ε : ℚ), abs (result - 125/10) < ε ∧ ε < 1/10 := by
  sorry

end NUMINAMATH_CALUDE_uninsured_part_time_percentage_l2777_277782


namespace NUMINAMATH_CALUDE_elena_bouquet_petals_l2777_277775

/-- Represents the number of flowers of each type in Elena's garden -/
structure FlowerCounts where
  lilies : ℕ
  tulips : ℕ
  roses : ℕ
  daisies : ℕ

/-- Represents the number of petals for each type of flower -/
structure PetalCounts where
  lily_petals : ℕ
  tulip_petals : ℕ
  rose_petals : ℕ
  daisy_petals : ℕ

/-- Calculates the number of flowers to take for the bouquet -/
def bouquet_flowers (garden : FlowerCounts) : FlowerCounts :=
  let min_count := min (garden.lilies / 2) (min (garden.tulips / 2) (min (garden.roses / 2) (garden.daisies / 2)))
  { lilies := min_count
    tulips := min_count
    roses := min_count
    daisies := min_count }

/-- Calculates the total number of petals in the bouquet -/
def total_petals (flowers : FlowerCounts) (petals : PetalCounts) : ℕ :=
  flowers.lilies * petals.lily_petals +
  flowers.tulips * petals.tulip_petals +
  flowers.roses * petals.rose_petals +
  flowers.daisies * petals.daisy_petals

/-- Elena's garden and petal counts -/
def elena_garden : FlowerCounts := { lilies := 8, tulips := 5, roses := 4, daisies := 3 }
def elena_petals : PetalCounts := { lily_petals := 6, tulip_petals := 3, rose_petals := 5, daisy_petals := 12 }

theorem elena_bouquet_petals :
  total_petals (bouquet_flowers elena_garden) elena_petals = 52 := by
  sorry


end NUMINAMATH_CALUDE_elena_bouquet_petals_l2777_277775


namespace NUMINAMATH_CALUDE_determinant_problem_l2777_277747

theorem determinant_problem (x y z w : ℝ) 
  (h : x * w - y * z = 7) : 
  x * (8 * z + 4 * w) - z * (8 * x + 4 * y) = 28 := by
  sorry

end NUMINAMATH_CALUDE_determinant_problem_l2777_277747


namespace NUMINAMATH_CALUDE_employee_pay_calculation_l2777_277773

def total_pay : ℝ := 550
def a_percentage : ℝ := 1.2

theorem employee_pay_calculation (b_pay : ℝ) (h1 : b_pay > 0) 
  (h2 : b_pay + a_percentage * b_pay = total_pay) : b_pay = 250 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_calculation_l2777_277773


namespace NUMINAMATH_CALUDE_greatest_q_plus_r_l2777_277714

theorem greatest_q_plus_r : ∃ (q r : ℕ+), 
  927 = 21 * q + r ∧ 
  ∀ (q' r' : ℕ+), 927 = 21 * q' + r' → q + r ≥ q' + r' :=
by sorry

end NUMINAMATH_CALUDE_greatest_q_plus_r_l2777_277714


namespace NUMINAMATH_CALUDE_slope_product_on_hyperbola_l2777_277788

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 - (2 * y^2) / (Real.sqrt 5 + 1) = 1

theorem slope_product_on_hyperbola 
  (M N P : ℝ × ℝ) 
  (hM : hyperbola M.1 M.2) 
  (hN : hyperbola N.1 N.2) 
  (hP : hyperbola P.1 P.2) 
  (hMN : N = (-M.1, -M.2)) :
  let k_PM := (P.2 - M.2) / (P.1 - M.1)
  let k_PN := (P.2 - N.2) / (P.1 - N.1)
  k_PM * k_PN = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_product_on_hyperbola_l2777_277788


namespace NUMINAMATH_CALUDE_break_room_capacity_l2777_277790

/-- The number of people that can be seated at each table -/
def people_per_table : ℕ := 8

/-- The number of tables in the break room -/
def number_of_tables : ℕ := 4

/-- The total number of people that can be seated in the break room -/
def total_seating_capacity : ℕ := people_per_table * number_of_tables

theorem break_room_capacity : total_seating_capacity = 32 := by
  sorry

end NUMINAMATH_CALUDE_break_room_capacity_l2777_277790


namespace NUMINAMATH_CALUDE_power_mod_37_l2777_277734

theorem power_mod_37 (n : ℕ) (h1 : n < 37) (h2 : (6 * n) % 37 = 1) :
  (Nat.pow (Nat.pow 2 n) 4 - 3) % 37 = 35 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_37_l2777_277734


namespace NUMINAMATH_CALUDE_midpoint_complex_plane_l2777_277723

theorem midpoint_complex_plane (A B C : ℂ) : 
  A = 6 + 5*I ∧ B = -2 + 3*I ∧ C = (A + B) / 2 → C = 2 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_midpoint_complex_plane_l2777_277723


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2777_277758

/-- Proves that for a quadratic function y = ax^2 + bx + c with integer coefficients,
    if the vertex is at (2, 5) and the point (3, 8) lies on the parabola, then a = 3. -/
theorem quadratic_coefficient (a b c : ℤ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = a * x^2 + b * x + c) →
  (∃ y : ℝ, 5 = a * 2^2 + b * 2 + c ∧ 5 ≥ y) →
  (8 = a * 3^2 + b * 3 + c) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2777_277758


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2777_277772

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (right_triangle : a^2 + b^2 = c^2) (b_larger : b > a) (tan_condition : b/a < 2) :
  (a^2 / (b^2 + c^2)) + (b^2 / (a^2 + c^2)) > 4/9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2777_277772


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_lengths_l2777_277785

def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∧ a + b + c = 5 ∧ (a = 2 ∨ b = 2 ∨ c = 2)) ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

theorem isosceles_triangle_base_lengths :
  ∃ (a b c : ℝ), IsoscelesTriangle a b c ∧ (c = 1.5 ∨ c = 2) := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_lengths_l2777_277785


namespace NUMINAMATH_CALUDE_age_difference_proof_l2777_277783

theorem age_difference_proof (hurley_age richard_age : ℕ) : 
  hurley_age = 14 →
  hurley_age + 40 + (richard_age + 40) = 128 →
  richard_age - hurley_age = 20 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2777_277783


namespace NUMINAMATH_CALUDE_parallel_lines_iff_a_eq_two_l2777_277760

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

theorem parallel_lines_iff_a_eq_two (a : ℝ) :
  parallel (2 / a) ((a - 1) / 1) ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_iff_a_eq_two_l2777_277760


namespace NUMINAMATH_CALUDE_probability_of_green_ball_l2777_277756

/-- The probability of drawing a green ball from a bag with specified conditions -/
theorem probability_of_green_ball (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h_total : total = 10)
  (h_red : red = 3)
  (h_blue : blue = 2) :
  (total - red - blue) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_green_ball_l2777_277756


namespace NUMINAMATH_CALUDE_win_probability_comparison_l2777_277738

theorem win_probability_comparison :
  let p : ℝ := 1 / 2  -- Probability of winning a single game
  let n₁ : ℕ := 4     -- Total number of games in scenario 1
  let k₁ : ℕ := 3     -- Number of wins needed in scenario 1
  let n₂ : ℕ := 8     -- Total number of games in scenario 2
  let k₂ : ℕ := 5     -- Number of wins needed in scenario 2
  
  -- Probability of winning exactly k₁ out of n₁ games
  let prob₁ : ℝ := (n₁.choose k₁ : ℝ) * p ^ k₁ * (1 - p) ^ (n₁ - k₁)
  
  -- Probability of winning exactly k₂ out of n₂ games
  let prob₂ : ℝ := (n₂.choose k₂ : ℝ) * p ^ k₂ * (1 - p) ^ (n₂ - k₂)
  
  prob₁ > prob₂ := by sorry

end NUMINAMATH_CALUDE_win_probability_comparison_l2777_277738


namespace NUMINAMATH_CALUDE_transylvanian_vampire_statement_l2777_277736

-- Define the possible species
inductive Species
| Human
| Vampire

-- Define the possible mental states
inductive MentalState
| Sane
| Insane

-- Define a person
structure Person where
  species : Species
  mentalState : MentalState

-- Define the statement made by the person
def madeVampireStatement (p : Person) : Prop :=
  p.mentalState = MentalState.Insane

-- Theorem statement
theorem transylvanian_vampire_statement 
  (p : Person) 
  (h : madeVampireStatement p) : 
  (∃ (s : Species), p.species = s) ∧ 
  (p.mentalState = MentalState.Insane) :=
sorry

end NUMINAMATH_CALUDE_transylvanian_vampire_statement_l2777_277736


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l2777_277707

theorem constant_ratio_problem (x y : ℚ) (k : ℚ) : 
  (k = (5 * x - 3) / (y + 20)) → 
  (y = 2 ∧ x = 1 → k = 1/11) → 
  (y = 5 → x = 58/55) := by
  sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l2777_277707


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l2777_277764

theorem polynomial_value_theorem (x : ℝ) : 
  x^2 - (5/2)*x = 6 → 2*x^2 - 5*x + 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l2777_277764


namespace NUMINAMATH_CALUDE_solution_set_implies_ab_l2777_277792

theorem solution_set_implies_ab (a b : ℝ) : 
  (∀ x, x^2 + a*x + b ≤ 0 ↔ -1 ≤ x ∧ x ≤ 3) → a*b = 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_ab_l2777_277792


namespace NUMINAMATH_CALUDE_student_failed_marks_l2777_277770

def total_marks : ℕ := 400
def passing_percentage : ℚ := 45 / 100
def obtained_marks : ℕ := 150

theorem student_failed_marks :
  (total_marks * passing_percentage).floor - obtained_marks = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_failed_marks_l2777_277770


namespace NUMINAMATH_CALUDE_vector_angle_problem_l2777_277713

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_angle_problem (b : ℝ × ℝ) :
  let a : ℝ × ℝ := (1, Real.sqrt 3)
  (Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 1) →
  (Real.sqrt (((a.1 + b.1) ^ 2) + ((a.2 + b.2) ^ 2)) = Real.sqrt 3) →
  angle_between_vectors a b = (2 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_problem_l2777_277713


namespace NUMINAMATH_CALUDE_circle_properties_l2777_277729

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the line equation for the center
def center_line (x y : ℝ) : Prop := y = -2 * x

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

theorem circle_properties :
  -- The circle passes through point A(2, -1)
  circle_equation 2 (-1) ∧
  -- The center of the circle is on the line y = -2x
  ∃ (cx cy : ℝ), center_line cx cy ∧ 
    ∀ (x y : ℝ), circle_equation x y ↔ (x - cx)^2 + (y - cy)^2 = 2 ∧
  -- The circle is tangent to the line x + y - 1 = 0
  ∃ (tx ty : ℝ), tangent_line tx ty ∧
    circle_equation tx ty ∧
    ∀ (x y : ℝ), tangent_line x y → 
      ((x - tx)^2 + (y - ty)^2 < 2 ∨ (x = tx ∧ y = ty))
  := by sorry

end NUMINAMATH_CALUDE_circle_properties_l2777_277729


namespace NUMINAMATH_CALUDE_somu_age_relation_somu_age_relation_past_somu_present_age_l2777_277732

/-- Somu's present age -/
def somu_age : ℕ := sorry

/-- Somu's father's present age -/
def father_age : ℕ := sorry

/-- Theorem stating the relationship between Somu's age and his father's age -/
theorem somu_age_relation : somu_age = father_age / 3 := by sorry

/-- Theorem stating the relationship between Somu's and his father's ages 10 years ago -/
theorem somu_age_relation_past : somu_age - 10 = (father_age - 10) / 5 := by sorry

/-- Main theorem proving Somu's present age -/
theorem somu_present_age : somu_age = 20 := by sorry

end NUMINAMATH_CALUDE_somu_age_relation_somu_age_relation_past_somu_present_age_l2777_277732


namespace NUMINAMATH_CALUDE_joels_age_when_dad_twice_as_old_l2777_277799

theorem joels_age_when_dad_twice_as_old (joel_current_age dad_current_age : ℕ) 
  (h1 : joel_current_age = 12) 
  (h2 : dad_current_age = 47) : 
  ∃ (years : ℕ), dad_current_age + years = 2 * (joel_current_age + years) ∧ 
                 joel_current_age + years = 35 := by
  sorry

end NUMINAMATH_CALUDE_joels_age_when_dad_twice_as_old_l2777_277799


namespace NUMINAMATH_CALUDE_jessica_rearrangements_time_l2777_277743

def name_length : ℕ := 7
def repeated_s : ℕ := 2
def repeated_a : ℕ := 2
def rearrangements_per_minute : ℕ := 15

def total_rearrangements : ℕ := name_length.factorial / (repeated_s.factorial * repeated_a.factorial)

theorem jessica_rearrangements_time :
  (total_rearrangements : ℚ) / rearrangements_per_minute / 60 = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_jessica_rearrangements_time_l2777_277743


namespace NUMINAMATH_CALUDE_cost_of_two_pans_l2777_277730

/-- The cost of a single pot -/
def pot_cost : ℕ := 20

/-- The number of pots purchased -/
def num_pots : ℕ := 3

/-- The number of pans purchased -/
def num_pans : ℕ := 4

/-- The total cost of all items -/
def total_cost : ℕ := 100

/-- The cost of a single pan -/
def pan_cost : ℕ := (total_cost - num_pots * pot_cost) / num_pans

theorem cost_of_two_pans :
  2 * pan_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_two_pans_l2777_277730


namespace NUMINAMATH_CALUDE_inequalities_theorem_l2777_277757

theorem inequalities_theorem (a b c d : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > -a) (h4 : c < d) (h5 : d < 0) : 
  (a / b + b / c < 0) ∧ (a - c > b - d) ∧ (a * (d - c) > b * (d - c)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l2777_277757


namespace NUMINAMATH_CALUDE_inequality_proof_l2777_277791

theorem inequality_proof (a b c : ℝ) : 
  a = Real.sin (14 * π / 180) + Real.cos (14 * π / 180) →
  b = 2 * Real.sqrt 2 * Real.sin (30.5 * π / 180) * Real.cos (30.5 * π / 180) →
  c = Real.sqrt 6 / 2 →
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2777_277791


namespace NUMINAMATH_CALUDE_sauteTimeRatio_l2777_277741

/-- Represents the time spent on various tasks in making calzones -/
structure CalzoneTime where
  total : ℕ
  sauteOnions : ℕ
  kneadDough : ℕ

/-- Calculates the time spent sauteing garlic and peppers -/
def sauteGarlicPeppers (ct : CalzoneTime) : ℕ :=
  ct.total - (ct.sauteOnions + ct.kneadDough + 2 * ct.kneadDough + (ct.kneadDough + 2 * ct.kneadDough) / 10)

/-- Theorem stating the ratio of time spent sauteing garlic and peppers to time spent sauteing onions -/
theorem sauteTimeRatio (ct : CalzoneTime) 
    (h1 : ct.total = 124)
    (h2 : ct.sauteOnions = 20)
    (h3 : ct.kneadDough = 30) :
  4 * (sauteGarlicPeppers ct) = ct.sauteOnions := by
  sorry

#eval sauteGarlicPeppers { total := 124, sauteOnions := 20, kneadDough := 30 }

end NUMINAMATH_CALUDE_sauteTimeRatio_l2777_277741


namespace NUMINAMATH_CALUDE_cricket_team_size_l2777_277769

/-- Represents a cricket team with its age-related properties -/
structure CricketTeam where
  n : ℕ  -- number of team members
  captain_age : ℕ
  wicket_keeper_age : ℕ
  team_avg_age : ℚ
  remaining_avg_age : ℚ

/-- The cricket team satisfies the given conditions -/
def satisfies_conditions (team : CricketTeam) : Prop :=
  team.captain_age = 27 ∧
  team.wicket_keeper_age = team.captain_age + 3 ∧
  team.team_avg_age = 24 ∧
  team.remaining_avg_age = team.team_avg_age - 1 ∧
  team.n * team.team_avg_age = team.captain_age + team.wicket_keeper_age + (team.n - 2) * team.remaining_avg_age

/-- The number of members in the cricket team that satisfies the conditions is 11 -/
theorem cricket_team_size :
  ∃ (team : CricketTeam), satisfies_conditions team ∧ team.n = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l2777_277769


namespace NUMINAMATH_CALUDE_happy_valley_kennel_arrangement_l2777_277710

/-- The number of chickens -/
def num_chickens : ℕ := 4

/-- The number of dogs -/
def num_dogs : ℕ := 3

/-- The number of cats -/
def num_cats : ℕ := 5

/-- The number of empty cages -/
def num_empty_cages : ℕ := 2

/-- The total number of entities (animals + empty cages) -/
def total_entities : ℕ := num_chickens + num_dogs + num_cats + num_empty_cages

/-- The number of animal groups -/
def num_groups : ℕ := 3

/-- The number of possible positions for empty cages -/
def num_positions : ℕ := num_groups + 2

theorem happy_valley_kennel_arrangement :
  (Nat.factorial num_groups) * (Nat.choose num_positions num_empty_cages) *
  (Nat.factorial num_chickens) * (Nat.factorial num_dogs) * (Nat.factorial num_cats) = 1036800 := by
  sorry

end NUMINAMATH_CALUDE_happy_valley_kennel_arrangement_l2777_277710


namespace NUMINAMATH_CALUDE_sum_odd_product_even_l2777_277751

theorem sum_odd_product_even (a b : ℤ) : 
  Odd (a + b) → Even (a * b) := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_product_even_l2777_277751


namespace NUMINAMATH_CALUDE_total_notes_is_133_l2777_277727

/-- Calculates the sum of integers from 1 to n -/
def triangleSum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the distribution of notes on the board -/
structure NoteDistribution where
  redRowCount : ℕ
  redPerRow : ℕ
  redScattered : ℕ
  blueRowCount : ℕ
  bluePerRow : ℕ
  blueScattered : ℕ
  greenTriangleBases : List ℕ
  yellowDiagonal1 : ℕ
  yellowDiagonal2 : ℕ
  yellowHexagon : ℕ

/-- Calculates the total number of notes based on the given distribution -/
def totalNotes (dist : NoteDistribution) : ℕ :=
  let redNotes := dist.redRowCount * dist.redPerRow + dist.redScattered
  let blueNotes := dist.blueRowCount * dist.bluePerRow + dist.blueScattered
  let greenNotes := (dist.greenTriangleBases.map triangleSum).sum
  let yellowNotes := dist.yellowDiagonal1 + dist.yellowDiagonal2 + dist.yellowHexagon
  redNotes + blueNotes + greenNotes + yellowNotes

/-- The actual distribution of notes on the board -/
def actualDistribution : NoteDistribution := {
  redRowCount := 5
  redPerRow := 6
  redScattered := 3
  blueRowCount := 4
  bluePerRow := 7
  blueScattered := 12
  greenTriangleBases := [4, 5, 6]
  yellowDiagonal1 := 5
  yellowDiagonal2 := 3
  yellowHexagon := 6
}

/-- Theorem stating that the total number of notes is 133 -/
theorem total_notes_is_133 : totalNotes actualDistribution = 133 := by
  sorry

end NUMINAMATH_CALUDE_total_notes_is_133_l2777_277727


namespace NUMINAMATH_CALUDE_magician_marbles_left_l2777_277762

theorem magician_marbles_left (red_initial : Nat) (blue_initial : Nat) 
  (red_taken : Nat) (blue_taken_multiplier : Nat) : 
  red_initial = 20 → 
  blue_initial = 30 → 
  red_taken = 3 → 
  blue_taken_multiplier = 4 → 
  (red_initial - red_taken) + (blue_initial - (blue_taken_multiplier * red_taken)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_magician_marbles_left_l2777_277762


namespace NUMINAMATH_CALUDE_range_of_m_l2777_277709

theorem range_of_m (x m : ℝ) : 
  (∀ x, (4 * x - m < 0 → 1 ≤ 3 - x ∧ 3 - x ≤ 4) ∧ 
  ∃ x, (1 ≤ 3 - x ∧ 3 - x ≤ 4 ∧ ¬(4 * x - m < 0))) →
  m > 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2777_277709


namespace NUMINAMATH_CALUDE_sams_recycling_cans_l2777_277750

theorem sams_recycling_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : 
  saturday_bags = 4 → sunday_bags = 3 → cans_per_bag = 6 →
  (saturday_bags + sunday_bags) * cans_per_bag = 42 := by
  sorry

end NUMINAMATH_CALUDE_sams_recycling_cans_l2777_277750


namespace NUMINAMATH_CALUDE_robbie_rice_solution_l2777_277798

/-- Robbie's daily rice consumption and fat intake --/
def robbie_rice_problem (x : ℝ) : Prop :=
  let morning_rice := x
  let afternoon_rice := 2
  let evening_rice := 5
  let fat_per_cup := 10
  let weekly_fat := 700
  let daily_rice := morning_rice + afternoon_rice + evening_rice
  let daily_fat := daily_rice * fat_per_cup
  daily_fat * 7 = weekly_fat

/-- The solution to Robbie's rice consumption problem --/
theorem robbie_rice_solution :
  ∃ x : ℝ, robbie_rice_problem x ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_robbie_rice_solution_l2777_277798


namespace NUMINAMATH_CALUDE_volleyball_tournament_winner_l2777_277797

/-- Represents a volleyball tournament -/
structure VolleyballTournament where
  /-- The number of teams in the tournament -/
  num_teams : ℕ
  /-- The number of games each team plays -/
  games_per_team : ℕ
  /-- The total number of games played in the tournament -/
  total_games : ℕ
  /-- There are no draws in the tournament -/
  no_draws : Bool

/-- Theorem stating that in a volleyball tournament with 6 teams, 
    where each team plays against every other team once and there are no draws, 
    at least one team must win 3 or more games -/
theorem volleyball_tournament_winner (t : VolleyballTournament) 
  (h1 : t.num_teams = 6)
  (h2 : t.games_per_team = 5)
  (h3 : t.total_games = t.num_teams * t.games_per_team / 2)
  (h4 : t.no_draws = true) :
  ∃ (team : ℕ), team ≤ t.num_teams ∧ (∃ (wins : ℕ), wins ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_volleyball_tournament_winner_l2777_277797


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_theorem_l2777_277784

noncomputable def inscribed_circle_radius (side_length : ℝ) (A₁ A₂ : ℝ) : ℝ :=
  sorry

theorem inscribed_circle_radius_theorem :
  let side_length : ℝ := 4
  let A₁ : ℝ := 8
  let A₂ : ℝ := 8
  -- Square circumscribes both circles
  side_length ^ 2 = A₁ + A₂ →
  -- Arithmetic progression condition
  A₁ + A₂ / 2 = (A₁ + (A₁ + A₂)) / 2 →
  -- Radius calculation
  inscribed_circle_radius side_length A₁ A₂ = 2 * Real.sqrt (2 / Real.pi)
  := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_theorem_l2777_277784


namespace NUMINAMATH_CALUDE_area_of_special_trapezoid_l2777_277754

/-- An isosceles trapezoid with a circle inscribed in it -/
structure InscribedCircleTrapezoid where
  /-- Length of the shorter base -/
  a : ℝ
  /-- Length of the longer base -/
  b : ℝ
  /-- Length of the leg -/
  c : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : c > 0
  /-- A circle can be inscribed in the trapezoid -/
  hasInscribedCircle : a + b = 2 * c

/-- The area of an isosceles trapezoid with bases 2 and 8, in which a circle can be inscribed, is 20 -/
theorem area_of_special_trapezoid :
  ∀ t : InscribedCircleTrapezoid, t.a = 2 ∧ t.b = 8 → 
  (1/2 : ℝ) * (t.a + t.b) * Real.sqrt (t.c^2 - ((t.b - t.a)/2)^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_area_of_special_trapezoid_l2777_277754


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2777_277794

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2777_277794


namespace NUMINAMATH_CALUDE_monkey_peach_problem_l2777_277746

theorem monkey_peach_problem :
  ∀ (num_monkeys num_peaches : ℕ),
    (num_peaches = 14 * num_monkeys + 48) →
    (num_peaches = 18 * num_monkeys - 64) →
    (num_monkeys = 28 ∧ num_peaches = 440) :=
by
  sorry

end NUMINAMATH_CALUDE_monkey_peach_problem_l2777_277746


namespace NUMINAMATH_CALUDE_equation_roots_l2777_277761

theorem equation_roots : ∃ (x y : ℝ), x < 0 ∧ y = 0 ∧
  3^x + x^2 + 2*x - 1 = 0 ∧
  3^y + y^2 + 2*y - 1 = 0 ∧
  ∀ (z : ℝ), (3^z + z^2 + 2*z - 1 = 0) → (z = x ∨ z = y) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l2777_277761


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2777_277719

-- Define a geometric sequence with positive terms
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Main theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 10 + a 3 * a 5 = 26 →
  a 5 * a 7 = 5 →
  a 4 + a 8 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2777_277719


namespace NUMINAMATH_CALUDE_five_x_minus_six_greater_than_one_l2777_277737

theorem five_x_minus_six_greater_than_one (x : ℝ) :
  (5 * x - 6 > 1) ↔ (5 * x - 6 > 1) :=
by sorry

end NUMINAMATH_CALUDE_five_x_minus_six_greater_than_one_l2777_277737


namespace NUMINAMATH_CALUDE_sum_of_complex_roots_l2777_277701

theorem sum_of_complex_roots (a b c : ℂ) 
  (eq1 : a^2 = b - c) 
  (eq2 : b^2 = c - a) 
  (eq3 : c^2 = a - b) : 
  (a + b + c = 0) ∨ (a + b + c = Complex.I * Real.sqrt 6) ∨ (a + b + c = -Complex.I * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_roots_l2777_277701


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_primes_l2777_277786

/-- The five smallest prime numbers -/
def smallest_primes : List Nat := [2, 3, 5, 7, 11]

/-- A number is four-digit if it's between 1000 and 9999 inclusive -/
def is_four_digit (n : Nat) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A number is divisible by all numbers in a list if it's divisible by their product -/
def divisible_by_all (n : Nat) (list : List Nat) : Prop :=
  n % (list.prod) = 0

theorem smallest_four_digit_divisible_by_smallest_primes :
  (2310 = (smallest_primes.prod)) ∧
  (is_four_digit 2310) ∧
  (divisible_by_all 2310 smallest_primes) ∧
  (∀ m : Nat, m < 2310 → ¬(is_four_digit m ∧ divisible_by_all m smallest_primes)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_primes_l2777_277786


namespace NUMINAMATH_CALUDE_bowling_balls_count_l2777_277708

/-- The number of red bowling balls -/
def red_balls : ℕ := 30

/-- The difference between green and red bowling balls -/
def green_red_difference : ℕ := 6

/-- The total number of bowling balls -/
def total_balls : ℕ := red_balls + (red_balls + green_red_difference)

theorem bowling_balls_count : total_balls = 66 := by
  sorry

end NUMINAMATH_CALUDE_bowling_balls_count_l2777_277708


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_222_l2777_277767

/-- The sum of the digits in the binary representation of a natural number -/
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- Theorem: The sum of the digits in the binary representation of 222 is 6 -/
theorem sum_of_binary_digits_222 :
  sum_of_binary_digits 222 = 6 := by
  sorry

#eval sum_of_binary_digits 222  -- This should output 6

end NUMINAMATH_CALUDE_sum_of_binary_digits_222_l2777_277767


namespace NUMINAMATH_CALUDE_longest_crafting_pattern_length_l2777_277779

/-- Represents the lengths of ribbons in inches -/
structure RibbonLengths where
  red : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ
  purple : ℕ

/-- Calculates the remaining lengths of ribbons -/
def remainingLengths (initial used : RibbonLengths) : RibbonLengths :=
  { red := initial.red - used.red,
    blue := initial.blue - used.blue,
    green := initial.green - used.green,
    yellow := initial.yellow - used.yellow,
    purple := initial.purple - used.purple }

/-- Finds the minimum length among all ribbon colors -/
def minLength (lengths : RibbonLengths) : ℕ :=
  min lengths.red (min lengths.blue (min lengths.green (min lengths.yellow lengths.purple)))

/-- The initial lengths of ribbons -/
def initialLengths : RibbonLengths :=
  { red := 84, blue := 96, green := 112, yellow := 54, purple := 120 }

/-- The used lengths of ribbons -/
def usedLengths : RibbonLengths :=
  { red := 46, blue := 58, green := 72, yellow := 30, purple := 90 }

theorem longest_crafting_pattern_length :
  minLength (remainingLengths initialLengths usedLengths) = 24 := by
  sorry

#eval minLength (remainingLengths initialLengths usedLengths)

end NUMINAMATH_CALUDE_longest_crafting_pattern_length_l2777_277779


namespace NUMINAMATH_CALUDE_product_of_numbers_l2777_277706

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 30) (sum_cubes_eq : x^3 + y^3 = 9450) : x * y = -585 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2777_277706


namespace NUMINAMATH_CALUDE_positive_number_problem_l2777_277755

theorem positive_number_problem : ∃ (n : ℕ), n > 0 ∧ 3 * n + n^2 = 300 ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_problem_l2777_277755


namespace NUMINAMATH_CALUDE_relationship_abc_l2777_277726

theorem relationship_abc : 
  let a : ℝ := Real.rpow 0.7 0.6
  let b : ℝ := Real.rpow 0.6 (-0.6)
  let c : ℝ := Real.rpow 0.6 0.7
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l2777_277726


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l2777_277749

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 48 → (∀ x y : ℕ, x * y = 48 → heart + club ≥ x + y) → heart + club = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l2777_277749


namespace NUMINAMATH_CALUDE_soccer_match_handshakes_l2777_277725

def soccer_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : ℕ :=
  let player_handshakes := team_size * team_size * (num_teams - 1) / 2
  let referee_handshakes := team_size * num_teams * num_referees
  player_handshakes + referee_handshakes

theorem soccer_match_handshakes :
  soccer_handshakes 6 2 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_soccer_match_handshakes_l2777_277725


namespace NUMINAMATH_CALUDE_paint_cans_for_house_l2777_277724

/-- Calculates the number of paint cans needed for a house painting job. -/
def paint_cans_needed (num_bedrooms : ℕ) (paint_per_room : ℕ) 
  (color_can_size : ℕ) (white_can_size : ℕ) : ℕ :=
  let num_other_rooms := 2 * num_bedrooms
  let total_color_paint := num_bedrooms * paint_per_room
  let total_white_paint := num_other_rooms * paint_per_room
  let color_cans := (total_color_paint + color_can_size - 1) / color_can_size
  let white_cans := (total_white_paint + white_can_size - 1) / white_can_size
  color_cans + white_cans

/-- Theorem stating that the number of paint cans needed for the given conditions is 10. -/
theorem paint_cans_for_house : paint_cans_needed 3 2 1 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_for_house_l2777_277724


namespace NUMINAMATH_CALUDE_periodic_function_l2777_277735

theorem periodic_function (f : ℝ → ℝ) (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)) →
  ∀ x : ℝ, f x = f (x + 2*a) :=
by sorry

end NUMINAMATH_CALUDE_periodic_function_l2777_277735


namespace NUMINAMATH_CALUDE_divisibility_of_2_pow_62_plus_1_l2777_277721

theorem divisibility_of_2_pow_62_plus_1 :
  ∃ k : ℕ, 2^62 + 1 = k * (2^31 + 2^16 + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_2_pow_62_plus_1_l2777_277721


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_zeros_l2777_277765

/-- A quadratic function of the form y = ax^2 + 4x - 2 has two distinct zeros if and only if a > -2 and a ≠ 0 -/
theorem quadratic_two_distinct_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 4 * x₁ - 2 = 0 ∧ a * x₂^2 + 4 * x₂ - 2 = 0) ↔
  (a > -2 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_zeros_l2777_277765


namespace NUMINAMATH_CALUDE_abc_value_l2777_277745

theorem abc_value (a b c : ℝ) 
  (h1 : a^2 * (b + c) = 2011)
  (h2 : b^2 * (a + c) = 2011)
  (h3 : a ≠ b) : 
  a * b * c = -2011 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l2777_277745


namespace NUMINAMATH_CALUDE_equation_solution_l2777_277705

theorem equation_solution : 
  ∃ x : ℝ, (3 / (2 * x + 1) = 5 / (4 * x)) ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2777_277705


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_negation_greater_than_neg_200_l2777_277722

theorem largest_multiple_of_8_negation_greater_than_neg_200 :
  ∀ n : ℤ, (n % 8 = 0 ∧ -n > -200) → n ≤ 192 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_negation_greater_than_neg_200_l2777_277722


namespace NUMINAMATH_CALUDE_no_linear_function_satisfies_inequality_l2777_277766

theorem no_linear_function_satisfies_inequality :
  ¬ ∃ (a b : ℝ), ∀ x ∈ Set.Icc 0 (2 * Real.pi),
    (a * x + b)^2 - Real.cos x * (a * x + b) < (1/4) * Real.sin x^2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_function_satisfies_inequality_l2777_277766


namespace NUMINAMATH_CALUDE_lower_interest_rate_l2777_277778

/-- Calculates simple interest given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem lower_interest_rate 
  (principal : ℚ) 
  (time : ℚ) 
  (higher_rate : ℚ) 
  (interest_difference : ℚ) : 
  principal = 5000 → 
  time = 2 → 
  higher_rate = 18 → 
  interest_difference = 600 → 
  ∃ (lower_rate : ℚ), 
    simpleInterest principal higher_rate time - simpleInterest principal lower_rate time = interest_difference ∧ 
    lower_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_lower_interest_rate_l2777_277778


namespace NUMINAMATH_CALUDE_walking_time_calculation_walk_two_miles_time_l2777_277703

/-- Calculates the time taken to walk a given distance at a constant pace -/
theorem walking_time_calculation (distance : ℝ) (total_distance : ℝ) (total_time : ℝ) :
  (distance > 0) → (total_distance > 0) → (total_time > 0) →
  (total_distance / total_time * total_time = total_distance) →
  (distance / (total_distance / total_time) = distance * total_time / total_distance) := by
  sorry

/-- Proves that walking 2 miles takes 1 hour given the conditions -/
theorem walk_two_miles_time :
  ∃ (pace : ℝ),
    (2 : ℝ) / pace = 1 ∧
    pace * 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_calculation_walk_two_miles_time_l2777_277703


namespace NUMINAMATH_CALUDE_difference_of_squares_l2777_277753

theorem difference_of_squares (m : ℝ) : m^2 - 1 = (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2777_277753


namespace NUMINAMATH_CALUDE_cone_volume_divided_by_pi_l2777_277793

/-- The volume of a cone formed from a 270-degree sector of a circle with radius 20, when divided by π, equals 375√7 -/
theorem cone_volume_divided_by_pi (r : Real) (h : Real) :
  r = 15 →
  h = 5 * Real.sqrt 7 →
  (1 / 3 * π * r^2 * h) / π = 375 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_divided_by_pi_l2777_277793


namespace NUMINAMATH_CALUDE_wife_account_percentage_l2777_277720

def total_income : ℝ := 200000
def children_count : ℕ := 3
def children_percentage : ℝ := 0.15
def orphan_house_percentage : ℝ := 0.05
def final_amount : ℝ := 40000

theorem wife_account_percentage :
  let children_total := children_count * children_percentage * total_income
  let remaining_after_children := total_income - children_total
  let orphan_house_amount := orphan_house_percentage * remaining_after_children
  let remaining_after_orphan := remaining_after_children - orphan_house_amount
  let wife_amount := remaining_after_orphan - final_amount
  (wife_amount / total_income) * 100 = 32.25 := by sorry

end NUMINAMATH_CALUDE_wife_account_percentage_l2777_277720


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l2777_277712

/-- If a line x - y - 1 = 0 is tangent to a parabola y = ax², then a = 1/4 -/
theorem tangent_line_to_parabola (a : ℝ) : 
  (∃ x y : ℝ, x - y - 1 = 0 ∧ y = a * x^2 ∧ 
   ∀ x' y' : ℝ, x' - y' - 1 = 0 → y' ≥ a * x'^2) → 
  a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l2777_277712


namespace NUMINAMATH_CALUDE_f_properties_l2777_277733

-- Define the function f(x) = x|x - 2|
def f (x : ℝ) : ℝ := x * abs (x - 2)

-- Theorem for the monotonicity intervals and inequality solution
theorem f_properties :
  (∀ x y, x ≤ y ∧ y ≤ 1 → f x ≤ f y) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f x ≤ f y) ∧
  (∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f y ≤ f x) ∧
  (∀ x, f x < 3 ↔ x < 3) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2777_277733


namespace NUMINAMATH_CALUDE_min_variance_sum_l2777_277759

theorem min_variance_sum (a b c : ℕ) : 
  70 ≤ a ∧ a < 80 →
  80 ≤ b ∧ b < 90 →
  90 ≤ c ∧ c ≤ 100 →
  let variance := (a - (a + b + c) / 3)^2 + (b - (a + b + c) / 3)^2 + (c - (a + b + c) / 3)^2
  (∀ a' b' c' : ℕ, 
    70 ≤ a' ∧ a' < 80 →
    80 ≤ b' ∧ b' < 90 →
    90 ≤ c' ∧ c' ≤ 100 →
    variance ≤ (a' - (a' + b' + c') / 3)^2 + (b' - (a' + b' + c') / 3)^2 + (c' - (a' + b' + c') / 3)^2) →
  a + b + c = 253 ∨ a + b + c = 254 :=
sorry

end NUMINAMATH_CALUDE_min_variance_sum_l2777_277759


namespace NUMINAMATH_CALUDE_tyrone_gave_fifteen_marbles_l2777_277763

/-- Represents the marble redistribution problem between Tyrone and Eric -/
def marble_redistribution (x : ℕ) : Prop :=
  let tyrone_initial : ℕ := 150
  let eric_initial : ℕ := 30
  let tyrone_final : ℕ := tyrone_initial - x
  let eric_final : ℕ := eric_initial + x
  (tyrone_final = 3 * eric_final) ∧ (x > 0) ∧ (x < tyrone_initial)

/-- The theorem stating that Tyrone gave 15 marbles to Eric -/
theorem tyrone_gave_fifteen_marbles : 
  ∃ (x : ℕ), marble_redistribution x ∧ x = 15 := by
sorry

end NUMINAMATH_CALUDE_tyrone_gave_fifteen_marbles_l2777_277763


namespace NUMINAMATH_CALUDE_system_solution_unique_l2777_277717

theorem system_solution_unique (x y : ℝ) : 
  x + 3 * y = -1 ∧ 2 * x + y = 3 ↔ x = 2 ∧ y = -1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2777_277717


namespace NUMINAMATH_CALUDE_range_of_a_l2777_277711

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

theorem range_of_a (h : ¬(∃ a : ℝ, p a ∨ q a)) :
  ∃ a : ℝ, 1 < a ∧ a < 2 ∧ ∀ b : ℝ, (1 < b ∧ b < 2) → (¬(p b) ∧ ¬(q b)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2777_277711


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2777_277776

theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 600) : 
  (0.8 * L) * (1.15 * W) = 552 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2777_277776
