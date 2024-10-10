import Mathlib

namespace max_gcd_consecutive_terms_l2143_214359

def a (n : ℕ) : ℕ := n.factorial + 3 * n

theorem max_gcd_consecutive_terms :
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ 3 ∧
  (∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = 3) :=
sorry

end max_gcd_consecutive_terms_l2143_214359


namespace sum_of_angles_two_triangles_l2143_214390

/-- The sum of interior angles of a triangle in degrees -/
def triangle_angle_sum : ℝ := 180

/-- The number of triangles in the diagram -/
def number_of_triangles : ℕ := 2

/-- Theorem: The sum of all interior angles in two triangles is 360° -/
theorem sum_of_angles_two_triangles : 
  (↑number_of_triangles : ℝ) * triangle_angle_sum = 360 := by
  sorry

end sum_of_angles_two_triangles_l2143_214390


namespace common_root_divisibility_l2143_214392

theorem common_root_divisibility (a b c : ℤ) (h1 : c ≠ b) :
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0) →
  ∃ k : ℤ, a + b + 2*c = 3*k := by
sorry

end common_root_divisibility_l2143_214392


namespace gcd_problems_l2143_214326

theorem gcd_problems : 
  (Nat.gcd 91 49 = 7) ∧ (Nat.gcd (Nat.gcd 319 377) 116 = 29) := by
  sorry

end gcd_problems_l2143_214326


namespace arithmetic_calculations_l2143_214370

theorem arithmetic_calculations :
  ((-15) + 4 + (-6) - (-11) = -6) ∧
  (-1^2024 + (-3)^2 * |(-1/18)| - 1 / (-2) = 0) := by
  sorry

end arithmetic_calculations_l2143_214370


namespace number_exists_l2143_214383

theorem number_exists : ∃ x : ℝ, (x^2 * 9^2) / 356 = 51.193820224719104 := by
  sorry

end number_exists_l2143_214383


namespace ac_squared_gt_bc_squared_sufficient_not_necessary_l2143_214374

theorem ac_squared_gt_bc_squared_sufficient_not_necessary (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) :=
by sorry

end ac_squared_gt_bc_squared_sufficient_not_necessary_l2143_214374


namespace dragons_total_games_dragons_total_games_is_90_l2143_214316

theorem dragons_total_games : ℕ → Prop :=
  fun total_games =>
    ∃ (pre_tournament_games : ℕ) (pre_tournament_wins : ℕ),
      -- Condition 1: 60% win rate before tournament
      pre_tournament_wins = (6 * pre_tournament_games) / 10 ∧
      -- Condition 2: 9 wins and 3 losses in tournament
      total_games = pre_tournament_games + 12 ∧
      -- Condition 3: 62% overall win rate after tournament
      (pre_tournament_wins + 9) = (62 * total_games) / 100 ∧
      -- Prove that total games is 90
      total_games = 90

theorem dragons_total_games_is_90 : dragons_total_games 90 := by
  sorry

end dragons_total_games_dragons_total_games_is_90_l2143_214316


namespace root_of_two_equations_l2143_214321

theorem root_of_two_equations (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * k^3 + b * k^2 - c * k + d = 0)
  (h2 : -b * k^3 + c * k^2 - d * k + a = 0) :
  k^4 = -1 := by
sorry

end root_of_two_equations_l2143_214321


namespace jed_card_collection_l2143_214348

def cards_after_weeks (initial_cards : ℕ) (cards_per_week : ℕ) (cards_given_biweekly : ℕ) (weeks : ℕ) : ℕ :=
  initial_cards + weeks * cards_per_week - (weeks / 2) * cards_given_biweekly

theorem jed_card_collection (target_cards : ℕ) : 
  cards_after_weeks 20 6 2 4 = target_cards ∧ target_cards = 40 :=
by sorry

end jed_card_collection_l2143_214348


namespace gcd_of_390_455_546_l2143_214344

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end gcd_of_390_455_546_l2143_214344


namespace base7_to_base10_65432_l2143_214330

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (init := 0) fun acc (i, d) => acc + d * (7 ^ (digits.length - 1 - i))

/-- The base-7 representation of the number --/
def base7Number : List Nat := [6, 5, 4, 3, 2]

/-- Theorem: The base-10 equivalent of 65432 in base-7 is 16340 --/
theorem base7_to_base10_65432 : base7ToBase10 base7Number = 16340 := by
  sorry

end base7_to_base10_65432_l2143_214330


namespace perfect_square_trinomial_l2143_214327

theorem perfect_square_trinomial (m : ℚ) : 
  m > 0 → 
  (∃ a : ℚ, ∀ x : ℚ, x^2 - 2*m*x + 36 = (x - a)^2) → 
  m = 6 := by
sorry

end perfect_square_trinomial_l2143_214327


namespace seven_classes_matches_l2143_214302

/-- 
Given a number of classes, calculates the total number of matches 
when each class plays against every other class exactly once.
-/
def totalMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- 
Theorem: When 7 classes play against each other once, 
the total number of matches is 21.
-/
theorem seven_classes_matches : totalMatches 7 = 21 := by
  sorry

end seven_classes_matches_l2143_214302


namespace sin_2alpha_value_l2143_214396

theorem sin_2alpha_value (α : Real) (h1 : α ∈ (Set.Ioo 0 Real.pi)) (h2 : Real.tan (Real.pi / 4 - α) = 1 / 3) : 
  Real.sin (2 * α) = 4 / 5 := by
sorry

end sin_2alpha_value_l2143_214396


namespace ellipse_axes_sum_l2143_214395

-- Define the cylinder and spheres
def cylinder_radius : ℝ := 6
def sphere_radius : ℝ := 6
def sphere_centers_distance : ℝ := 13

-- Define the ellipse axes
def minor_axis : ℝ := 2 * cylinder_radius
def major_axis : ℝ := sphere_centers_distance

-- Theorem statement
theorem ellipse_axes_sum :
  minor_axis + major_axis = 25 := by sorry

end ellipse_axes_sum_l2143_214395


namespace min_value_sqrt_plus_reciprocal_l2143_214378

theorem min_value_sqrt_plus_reciprocal (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 2 / x ≥ 5 ∧
  (3 * Real.sqrt x + 2 / x = 5 ↔ x = 1) :=
by sorry

end min_value_sqrt_plus_reciprocal_l2143_214378


namespace extreme_points_imply_a_l2143_214319

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b * x^2 + x

theorem extreme_points_imply_a (a b : ℝ) :
  (∀ x, x > 0 → (deriv (f a b)) x = 0 ↔ x = 1 ∨ x = 2) →
  a = -2/3 := by
  sorry

end extreme_points_imply_a_l2143_214319


namespace necklace_calculation_l2143_214356

theorem necklace_calculation (spools : Nat) (spool_length : Nat) (feet_per_necklace : Nat) : 
  spools = 3 → spool_length = 20 → feet_per_necklace = 4 → 
  (spools * spool_length) / feet_per_necklace = 15 := by
  sorry

end necklace_calculation_l2143_214356


namespace exactly_four_pairs_l2143_214388

/-- The number of ordered pairs (m,n) of positive integers satisfying the given conditions -/
def count_pairs : ℕ := 4

/-- Predicate to check if a pair (m,n) satisfies the required conditions -/
def is_valid_pair (m n : ℕ) : Prop :=
  m ≥ n ∧ m * m - n * n = 144

/-- The theorem stating that there are exactly 4 valid pairs -/
theorem exactly_four_pairs :
  (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = count_pairs ∧
    ∀ (p : ℕ × ℕ), p ∈ pairs ↔ is_valid_pair p.1 p.2) :=
sorry

end exactly_four_pairs_l2143_214388


namespace gcd_of_sum_and_lcm_l2143_214310

theorem gcd_of_sum_and_lcm (a b : ℕ+) (h1 : a + b = 33) (h2 : Nat.lcm a b = 90) : 
  Nat.gcd a b = 3 := by
  sorry

end gcd_of_sum_and_lcm_l2143_214310


namespace quadratic_equation_c_value_l2143_214360

theorem quadratic_equation_c_value (b c : ℝ) : 
  (∀ x : ℝ, x^2 - b*x + c = 0 → 
    ∃ y : ℝ, y^2 - b*y + c = 0 ∧ x ≠ y ∧ x * y = 20 ∧ x + y = 12) →
  c = 20 := by
sorry

end quadratic_equation_c_value_l2143_214360


namespace correct_password_contains_one_and_seven_l2143_214375

/-- Represents a four-digit password -/
def Password := Fin 4 → Fin 10

/-- Checks if two passwords have exactly two matching digits in different positions -/
def hasTwoMatchingDigits (p1 p2 : Password) : Prop :=
  (∃ i j : Fin 4, i ≠ j ∧ p1 i = p2 i ∧ p1 j = p2 j) ∧
  (∀ i j k : Fin 4, i ≠ j → j ≠ k → k ≠ i → ¬(p1 i = p2 i ∧ p1 j = p2 j ∧ p1 k = p2 k))

/-- The first four incorrect attempts -/
def attempts : Fin 4 → Password
| 0 => λ i => [3, 4, 0, 6].get i
| 1 => λ i => [1, 6, 3, 0].get i
| 2 => λ i => [7, 3, 6, 4].get i
| 3 => λ i => [6, 1, 7, 3].get i

/-- The theorem stating that the correct password must contain 1 and 7 -/
theorem correct_password_contains_one_and_seven 
  (correct : Password)
  (h1 : ∀ i : Fin 4, hasTwoMatchingDigits (attempts i) correct)
  (h2 : correct ≠ attempts 0 ∧ correct ≠ attempts 1 ∧ correct ≠ attempts 2 ∧ correct ≠ attempts 3) :
  (∃ i j : Fin 4, i ≠ j ∧ correct i = 1 ∧ correct j = 7) :=
sorry

end correct_password_contains_one_and_seven_l2143_214375


namespace divisible_by_fifteen_l2143_214339

theorem divisible_by_fifteen (a : ℤ) : ∃ k : ℤ, 9 * a^5 - 5 * a^3 - 4 * a = 15 * k := by
  sorry

end divisible_by_fifteen_l2143_214339


namespace trail_mix_nuts_l2143_214364

theorem trail_mix_nuts (walnuts almonds : ℚ) 
  (h1 : walnuts = 0.25)
  (h2 : almonds = 0.25) : 
  walnuts + almonds = 0.50 := by
sorry

end trail_mix_nuts_l2143_214364


namespace solve_determinant_equation_l2143_214365

-- Define the determinant operation
def det (a b c d : ℚ) : ℚ := a * d - b * c

-- Theorem statement
theorem solve_determinant_equation :
  ∀ x : ℚ, det 2 4 (1 - x) 5 = 18 → x = 3 := by
  sorry

end solve_determinant_equation_l2143_214365


namespace total_acorns_formula_l2143_214399

/-- The total number of acorns for Shawna, Sheila, and Danny -/
def total_acorns (x y : ℝ) : ℝ :=
  let shawna_acorns := x
  let sheila_acorns := 5.3 * x
  let danny_acorns := sheila_acorns + y
  shawna_acorns + sheila_acorns + danny_acorns

/-- Theorem stating that the total number of acorns is 11.6x + y -/
theorem total_acorns_formula (x y : ℝ) : total_acorns x y = 11.6 * x + y := by
  sorry

end total_acorns_formula_l2143_214399


namespace arc_length_for_specific_circle_l2143_214352

theorem arc_length_for_specific_circle (r : ℝ) (α : ℝ) (l : ℝ) : 
  r = π → α = 2 * π / 3 → l = r * α → l = 2 * π^2 / 3 :=
by sorry

end arc_length_for_specific_circle_l2143_214352


namespace negation_of_universal_proposition_l2143_214397

theorem negation_of_universal_proposition (a b : ℝ) :
  ¬(a < b → ∀ c : ℝ, a * c^2 < b * c^2) ↔ (a < b → ∃ c : ℝ, a * c^2 ≥ b * c^2) := by
  sorry

end negation_of_universal_proposition_l2143_214397


namespace not_power_of_two_l2143_214317

theorem not_power_of_two (m n : ℕ+) : ¬∃ k : ℕ, (36 * m.val + n.val) * (m.val + 36 * n.val) = 2^k := by
  sorry

end not_power_of_two_l2143_214317


namespace water_boiling_point_l2143_214349

/-- The temperature in Fahrenheit at which water boils -/
def boiling_point_f : ℝ := 212

/-- The temperature in Fahrenheit at which water melts -/
def melting_point_f : ℝ := 32

/-- The temperature in Celsius at which water melts -/
def melting_point_c : ℝ := 0

/-- A function to convert Celsius to Fahrenheit -/
def celsius_to_fahrenheit (c : ℝ) : ℝ := sorry

/-- A function to convert Fahrenheit to Celsius -/
def fahrenheit_to_celsius (f : ℝ) : ℝ := sorry

/-- The boiling point of water in Celsius -/
def boiling_point_c : ℝ := 100

theorem water_boiling_point :
  ∃ (temp_c temp_f : ℝ),
    celsius_to_fahrenheit temp_c = temp_f ∧
    temp_c = 35 ∧
    temp_f = 95 →
  fahrenheit_to_celsius boiling_point_f = boiling_point_c :=
sorry

end water_boiling_point_l2143_214349


namespace milburg_population_l2143_214385

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := 5256

/-- The number of children in Milburg -/
def children : ℕ := 2987

/-- The total population of Milburg -/
def total_population : ℕ := grown_ups + children

/-- Theorem stating that the total population of Milburg is 8243 -/
theorem milburg_population : total_population = 8243 := by
  sorry

end milburg_population_l2143_214385


namespace watches_synchronize_after_1600_days_l2143_214368

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- The rate at which Glafira's watch gains time (in seconds per day) -/
def glafira_gain : ℕ := 36

/-- The rate at which Gavrila's watch loses time (in seconds per day) -/
def gavrila_loss : ℕ := 18

/-- The theorem stating that the watches will display the correct time simultaneously after 1600 days -/
theorem watches_synchronize_after_1600_days :
  (seconds_per_day * 1600) % (glafira_gain + gavrila_loss) = 0 := by
  sorry

end watches_synchronize_after_1600_days_l2143_214368


namespace cuboid_distance_theorem_l2143_214384

/-- Given a cuboid with edges a, b, and c, and a vertex P, 
    the distance m from P to the plane passing through the vertices adjacent to P 
    satisfies the equation: 1/m² = 1/a² + 1/b² + 1/c² -/
theorem cuboid_distance_theorem (a b c m : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hm : m > 0) :
  (1 / m^2) = (1 / a^2) + (1 / b^2) + (1 / c^2) :=
sorry

end cuboid_distance_theorem_l2143_214384


namespace range_of_a_l2143_214347

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (h_odd : is_odd f) 
  (h_domain : ∀ x ∈ Set.Ioo (-1) 1, f x ≠ 0) 
  (h_ineq : ∀ a : ℝ, f (1 - a) + f (2 * a - 1) < 0) :
  Set.Ioo 0 1 = {a : ℝ | 0 < a ∧ a < 1} :=
sorry

end range_of_a_l2143_214347


namespace distance_to_circle_center_l2143_214372

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the problem
theorem distance_to_circle_center 
  (ABC : Triangle)
  (circle : Circle)
  (M : ℝ × ℝ)
  (h1 : ABC.C.1 = ABC.A.1 ∧ ABC.C.2 = ABC.B.2) -- Right triangle condition
  (h2 : (ABC.A.1 - ABC.B.1)^2 + (ABC.A.2 - ABC.B.2)^2 = 
        (ABC.C.1 - ABC.B.1)^2 + (ABC.C.2 - ABC.B.2)^2) -- Equal legs condition
  (h3 : circle.radius = (ABC.C.1 - ABC.A.1) / 2) -- Circle diameter is AC
  (h4 : circle.center = ((ABC.A.1 + ABC.C.1) / 2, (ABC.A.2 + ABC.C.2) / 2)) -- Circle center is midpoint of AC
  (h5 : (M.1 - ABC.A.1)^2 + (M.2 - ABC.A.2)^2 = circle.radius^2) -- M is on the circle
  (h6 : (M.1 - ABC.B.1)^2 + (M.2 - ABC.B.2)^2 = 2) -- BM = √2
  : (ABC.B.1 - circle.center.1)^2 + (ABC.B.2 - circle.center.2)^2 = 5 := by
  sorry

end distance_to_circle_center_l2143_214372


namespace complement_of_M_in_U_l2143_214306

def U : Set Nat := {0, 1, 2, 3, 4, 5}
def M : Set Nat := {0, 1}

theorem complement_of_M_in_U : 
  (U \ M) = {2, 3, 4, 5} := by sorry

end complement_of_M_in_U_l2143_214306


namespace fraction_of_powers_equals_five_fourths_l2143_214358

theorem fraction_of_powers_equals_five_fourths :
  (3^1007 + 3^1005) / (3^1007 - 3^1005) = 5/4 := by sorry

end fraction_of_powers_equals_five_fourths_l2143_214358


namespace third_factorial_is_seven_l2143_214308

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def gcd_of_three (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

theorem third_factorial_is_seven (b : ℕ) (x : ℕ) 
  (h1 : b = 9) 
  (h2 : gcd_of_three (factorial (b - 2)) (factorial (b + 1)) (factorial x) = 5040) : 
  x = 7 := by
  sorry

end third_factorial_is_seven_l2143_214308


namespace sum_of_x_and_y_equals_two_l2143_214337

theorem sum_of_x_and_y_equals_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 20) : x + y = 2 := by
  sorry

end sum_of_x_and_y_equals_two_l2143_214337


namespace complementary_sets_imply_a_eq_two_subset_implies_a_range_l2143_214300

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}

-- Theorem 1: If A ∩ B = ∅ and A ∪ B = ℝ, then a = 2
theorem complementary_sets_imply_a_eq_two (a : ℝ) :
  A a ∩ B = ∅ ∧ A a ∪ B = Set.univ → a = 2 := by sorry

-- Theorem 2: If A ⊆ B, then a ∈ (-∞, 0] ∪ [4, +∞)
theorem subset_implies_a_range (a : ℝ) :
  A a ⊆ B → a ≤ 0 ∨ a ≥ 4 := by sorry

end complementary_sets_imply_a_eq_two_subset_implies_a_range_l2143_214300


namespace exist_positive_reals_satisfying_inequalities_l2143_214311

theorem exist_positive_reals_satisfying_inequalities :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 + c^2 > 2 ∧
    a^3 + b^3 + c^3 < 2 ∧
    a^4 + b^4 + c^4 > 2 := by
  sorry

end exist_positive_reals_satisfying_inequalities_l2143_214311


namespace base8_45_equals_base10_37_l2143_214334

/-- Converts a two-digit base-eight number to base-ten -/
def base8_to_base10 (tens : Nat) (units : Nat) : Nat :=
  tens * 8 + units

/-- The base-eight number 45 is equal to the base-ten number 37 -/
theorem base8_45_equals_base10_37 : base8_to_base10 4 5 = 37 := by
  sorry

end base8_45_equals_base10_37_l2143_214334


namespace digit_2023_is_7_l2143_214331

/-- The sequence of digits obtained by writing integers 1 through 9999 in ascending order -/
def digit_sequence : ℕ → ℕ := sorry

/-- The real number x defined as .123456789101112...99989999 -/
noncomputable def x : ℝ := sorry

/-- The nth digit to the right of the decimal point in x -/
def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_2023_is_7 : nth_digit 2023 = 7 := by sorry

end digit_2023_is_7_l2143_214331


namespace arithmetic_geometric_sequence_l2143_214333

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 ∧
  (∀ n : ℕ, a (n + 1) = a n + d) ∧
  ∃ r : ℝ, r ≠ 0 ∧ a 3 = r * a 1 ∧ a 6 = r * a 3
  →
  ∃ r : ℝ, r = 3/2 ∧ a 3 = r * a 1 ∧ a 6 = r * a 3 :=
by sorry

end arithmetic_geometric_sequence_l2143_214333


namespace range_of_a_for_meaningful_sqrt_l2143_214371

theorem range_of_a_for_meaningful_sqrt (a : ℝ) : 
  (∃ x : ℝ, x^2 = 4 - a) ↔ a ≤ 4 := by sorry

end range_of_a_for_meaningful_sqrt_l2143_214371


namespace gravitational_force_calculation_l2143_214398

/-- Gravitational force calculation -/
theorem gravitational_force_calculation 
  (k : ℝ) -- Gravitational constant
  (d₁ d₂ : ℝ) -- Distances
  (f₁ : ℝ) -- Initial force
  (h₁ : d₁ = 5000) -- Initial distance
  (h₂ : d₂ = 300000) -- New distance
  (h₃ : f₁ = 400) -- Initial force value
  (h₄ : k = f₁ * d₁^2) -- Inverse square law
  : (k / d₂^2) = 1/9 := by
  sorry

end gravitational_force_calculation_l2143_214398


namespace regular_polygon_perimeter_l2143_214366

/-- A regular polygon with side length 7 and exterior angle 90 degrees has perimeter 28. -/
theorem regular_polygon_perimeter :
  ∀ (n : ℕ) (s : ℝ) (θ : ℝ),
    n > 0 →
    s = 7 →
    θ = 90 →
    (360 : ℝ) / n = θ →
    n * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l2143_214366


namespace license_plate_count_l2143_214380

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of odd digits -/
def num_odd_digits : ℕ := 5

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of prime digits under 10 -/
def num_prime_digits : ℕ := 4

/-- The total number of license plates -/
def total_license_plates : ℕ := num_letters ^ 3 * num_odd_digits * num_digits * num_prime_digits

theorem license_plate_count :
  total_license_plates = 351520 :=
by sorry

end license_plate_count_l2143_214380


namespace cost_price_is_41_l2143_214307

/-- Calculates the cost price per metre of cloth given the total length,
    total selling price, and loss per metre. -/
def cost_price_per_metre (total_length : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  (total_selling_price + total_length * loss_per_metre) / total_length

/-- Proves that the cost price per metre of cloth is 41 rupees given the specified conditions. -/
theorem cost_price_is_41 :
  cost_price_per_metre 500 18000 5 = 41 := by
  sorry

#eval cost_price_per_metre 500 18000 5

end cost_price_is_41_l2143_214307


namespace stream_speed_relationship_l2143_214320

-- Define the boat speeds and distances
def low_speed : ℝ := 20
def high_speed : ℝ := 40
def downstream_distance : ℝ := 26
def upstream_distance : ℝ := 14

-- Define the stream speeds as variables
variable (x y : ℝ)

-- Define the theorem
theorem stream_speed_relationship :
  (downstream_distance / (low_speed + x) = upstream_distance / (high_speed - y)) →
  380 = 7 * x + 13 * y :=
by
  sorry


end stream_speed_relationship_l2143_214320


namespace infinitely_many_solutions_l2143_214387

/-- Represents the number of animals in the farm -/
structure FarmAnimals where
  pigs : ℕ
  hens : ℕ

/-- The total number of legs in the farm -/
def totalLegs (animals : FarmAnimals) : ℕ :=
  4 * animals.pigs + 2 * animals.hens

/-- The total number of heads in the farm -/
def totalHeads (animals : FarmAnimals) : ℕ :=
  animals.pigs + animals.hens

/-- The condition given in the problem -/
def satisfiesCondition (animals : FarmAnimals) : Prop :=
  totalLegs animals = 3 * totalHeads animals + 36

theorem infinitely_many_solutions : 
  ∀ n : ℕ, ∃ animals : FarmAnimals, satisfiesCondition animals ∧ animals.pigs = n :=
sorry

end infinitely_many_solutions_l2143_214387


namespace arrangement_count_proof_l2143_214303

/-- The number of ways to arrange 8 athletes on 8 tracks with 3 specified athletes in consecutive tracks -/
def arrangement_count : ℕ := 4320

/-- The number of tracks in the stadium -/
def num_tracks : ℕ := 8

/-- The total number of athletes -/
def num_athletes : ℕ := 8

/-- The number of specified athletes that must be in consecutive tracks -/
def num_specified : ℕ := 3

/-- The number of ways to arrange the specified athletes in consecutive tracks -/
def consecutive_arrangements : ℕ := num_tracks - num_specified + 1

/-- The number of permutations of the specified athletes -/
def specified_permutations : ℕ := Nat.factorial num_specified

/-- The number of permutations of the remaining athletes -/
def remaining_permutations : ℕ := Nat.factorial (num_athletes - num_specified)

theorem arrangement_count_proof : 
  arrangement_count = consecutive_arrangements * specified_permutations * remaining_permutations :=
by sorry

end arrangement_count_proof_l2143_214303


namespace inverse_as_linear_combination_l2143_214379

/-- Given a 2x2 matrix N, prove that its inverse can be expressed as c * N + d * I -/
theorem inverse_as_linear_combination (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h₁ : N 0 0 = 3) (h₂ : N 0 1 = 1) (h₃ : N 1 0 = -2) (h₄ : N 1 1 = 4) :
  ∃ (c d : ℝ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ 
  c = -1/14 ∧ d = 3/7 := by
  sorry

end inverse_as_linear_combination_l2143_214379


namespace recreation_area_tents_l2143_214393

/-- Represents the number of tents in different parts of the campsite -/
structure CampsiteTents where
  north : ℕ
  east : ℕ
  center : ℕ
  south : ℕ

/-- Calculates the total number of tents in the campsite -/
def total_tents (c : CampsiteTents) : ℕ :=
  c.north + c.east + c.center + c.south

/-- Theorem stating the total number of tents in the recreation area -/
theorem recreation_area_tents : 
  ∀ c : CampsiteTents, 
  c.north = 100 → 
  c.east = 2 * c.north → 
  c.center = 4 * c.north → 
  c.south = 200 → 
  total_tents c = 900 := by
  sorry


end recreation_area_tents_l2143_214393


namespace average_problem_l2143_214314

theorem average_problem (t b c d e : ℝ) : 
  (t + b + c + 14 + 15) / 5 = 12 →
  t = 2 * b →
  (t + b + c + d + e + 14 + 15) / 7 = 12 →
  (t + b + c + 29) / 4 = 15 := by
sorry

end average_problem_l2143_214314


namespace triangle_properties_l2143_214324

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about properties of a specific triangle --/
theorem triangle_properties (t : Triangle) 
  (h1 : t.B = π / 3) :
  (t.a = 2 ∧ t.b = 2 * Real.sqrt 3 → t.c = 4) ∧
  (Real.tan t.A = 2 * Real.sqrt 3 → Real.tan t.C = 3 * Real.sqrt 3 / 5) := by
  sorry

end triangle_properties_l2143_214324


namespace perpendicular_parallel_implication_parallel_perpendicular_transitivity_l2143_214301

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_parallel_implication 
  (m n : Line) (α : Plane) : 
  perpendicular m α → parallel_line_plane n α → perpendicular_lines m n :=
sorry

-- Theorem 2
theorem parallel_perpendicular_transitivity 
  (m : Line) (α β γ : Plane) :
  parallel_plane α β → parallel_plane β γ → perpendicular m α → perpendicular m γ :=
sorry

end perpendicular_parallel_implication_parallel_perpendicular_transitivity_l2143_214301


namespace shortest_tree_height_l2143_214357

/-- Given three trees in a town square, this theorem proves the height of the shortest tree. -/
theorem shortest_tree_height (tallest middle shortest : ℝ) : 
  tallest = 150 →
  middle = 2/3 * tallest →
  shortest = 1/2 * middle →
  shortest = 50 := by
sorry

end shortest_tree_height_l2143_214357


namespace sum_of_quadratic_solutions_l2143_214346

/-- The sum of the solutions to the quadratic equation x² - 6x - 8 = 2x + 18 is 8 -/
theorem sum_of_quadratic_solutions : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x - 8 - (2*x + 18)
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 8 := by
sorry

end sum_of_quadratic_solutions_l2143_214346


namespace arithmetic_evaluation_l2143_214373

theorem arithmetic_evaluation : 1523 + 180 / 60 - 223 = 1303 := by
  sorry

end arithmetic_evaluation_l2143_214373


namespace common_chord_of_circles_l2143_214354

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the common chord
def common_chord (x y : ℝ) : Prop := 2*x - 4*y + 1 = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
by sorry

end common_chord_of_circles_l2143_214354


namespace first_quadrant_is_well_defined_set_l2143_214376

-- Define the first quadrant
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 > 0 ∧ p.2 > 0}

-- Theorem stating that the FirstQuadrant is a well-defined set
theorem first_quadrant_is_well_defined_set : 
  ∀ p : ℝ × ℝ, Decidable (p ∈ FirstQuadrant) :=
by
  sorry


end first_quadrant_is_well_defined_set_l2143_214376


namespace solution_set_f_greater_than_two_min_value_f_l2143_214342

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x < -7 ∨ x > 5/3} := by sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (x : ℝ), f x = -9/2 ∧ ∀ (y : ℝ), f y ≥ -9/2 := by sorry

end solution_set_f_greater_than_two_min_value_f_l2143_214342


namespace adam_ferris_wheel_cost_l2143_214363

/-- The amount of money Adam spent on the ferris wheel ride -/
def ferris_wheel_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * ticket_price

/-- Theorem: Adam spent 81 dollars on the ferris wheel ride -/
theorem adam_ferris_wheel_cost :
  ferris_wheel_cost 13 4 9 = 81 := by
  sorry

end adam_ferris_wheel_cost_l2143_214363


namespace function_property_l2143_214391

theorem function_property (f : ℕ+ → ℕ) 
  (h1 : ∀ (x y : ℕ+), f (x * y) = f x + f y)
  (h2 : f 30 = 21)
  (h3 : f 90 = 27) :
  f 270 = 33 := by
sorry

end function_property_l2143_214391


namespace contrapositive_equivalence_l2143_214340

theorem contrapositive_equivalence (p q : Prop) :
  (p → q) → (¬q → ¬p) := by sorry

end contrapositive_equivalence_l2143_214340


namespace train_length_l2143_214377

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 30 → time = 9 → ∃ length : ℝ, 
  (length ≥ 74.96 ∧ length ≤ 74.98) ∧ length = speed * (5/18) * time := by
  sorry

end train_length_l2143_214377


namespace circle_center_on_line_l2143_214329

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_center_on_line (x y : ℚ) : 
  (5 * x - 4 * y = 40) ∧ 
  (5 * x - 4 * y = -20) ∧ 
  (3 * x - y = 0) →
  x = -10/7 ∧ y = -30/7 := by
sorry

end circle_center_on_line_l2143_214329


namespace max_homework_time_l2143_214362

/-- The time Max spent on biology homework -/
def biology_time : ℕ := 20

/-- The time Max spent on history homework -/
def history_time : ℕ := 2 * biology_time

/-- The time Max spent on geography homework -/
def geography_time : ℕ := 3 * history_time

/-- The total time Max spent on homework -/
def total_time : ℕ := 180

theorem max_homework_time : 
  biology_time + history_time + geography_time = total_time ∧ 
  biology_time = 20 := by sorry

end max_homework_time_l2143_214362


namespace quadratic_sum_l2143_214332

/-- Given a quadratic expression 4x^2 - 8x - 3, when expressed in the form a(x - h)^2 + k,
    the sum a + h + k equals -2. -/
theorem quadratic_sum (a h k : ℝ) : 
  (∀ x, 4*x^2 - 8*x - 3 = a*(x - h)^2 + k) → a + h + k = -2 := by
  sorry

end quadratic_sum_l2143_214332


namespace A_greater_than_B_l2143_214325

def A : ℕ → ℕ
  | 0 => 3
  | n+1 => 3^(A n)

def B : ℕ → ℕ
  | 0 => 8
  | n+1 => 8^(B n)

theorem A_greater_than_B (n : ℕ) : A (n + 1) > B n := by
  sorry

end A_greater_than_B_l2143_214325


namespace equation_solution_l2143_214328

theorem equation_solution : 
  ∃ x : ℚ, (1 / 4 : ℚ) + 5 / x = 12 / x + (1 / 15 : ℚ) → x = 420 / 11 := by
  sorry

end equation_solution_l2143_214328


namespace square_garden_perimeter_l2143_214312

/-- A square garden with an area of 9 square meters has a perimeter of 12 meters. -/
theorem square_garden_perimeter : 
  ∀ (side : ℝ), side > 0 → side^2 = 9 → 4 * side = 12 := by
  sorry

end square_garden_perimeter_l2143_214312


namespace field_reduction_l2143_214322

theorem field_reduction (L W : ℝ) (h : L > 0 ∧ W > 0) :
  ∃ x : ℝ, x > 0 ∧ x < 100 ∧ 
  (1 - x / 100) * (1 - x / 100) * (L * W) = (1 - 0.64) * (L * W) →
  x = 40 := by
sorry

end field_reduction_l2143_214322


namespace total_houses_l2143_214343

theorem total_houses (garage : ℕ) (pool : ℕ) (both : ℕ) 
  (h1 : garage = 50) 
  (h2 : pool = 40) 
  (h3 : both = 35) : 
  garage + pool - both = 55 := by
  sorry

end total_houses_l2143_214343


namespace max_x_value_l2143_214341

theorem max_x_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x - 2 * Real.sqrt y = Real.sqrt (2 * x - y)) : 
  (∀ z : ℝ, z > 0 ∧ ∃ w : ℝ, w > 0 ∧ z - 2 * Real.sqrt w = Real.sqrt (2 * z - w) → z ≤ 10) :=
sorry

end max_x_value_l2143_214341


namespace remainder_theorem_l2143_214367

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^4 - 4*x^2 + 7*x - 8

-- State the theorem
theorem remainder_theorem :
  ∃ (Q : ℝ → ℝ), P = λ x => (x - 3) * Q x + 50 :=
sorry

end remainder_theorem_l2143_214367


namespace train_length_calculation_l2143_214351

/-- The length of a train that passes a tree in 12 seconds while traveling at 90 km/hr is 300 meters. -/
theorem train_length_calculation (passing_time : ℝ) (speed_kmh : ℝ) : 
  passing_time = 12 → speed_kmh = 90 → passing_time * (speed_kmh * (1000 / 3600)) = 300 := by
  sorry

end train_length_calculation_l2143_214351


namespace sqrt_difference_equality_l2143_214361

theorem sqrt_difference_equality : Real.sqrt (64 + 81) - Real.sqrt (49 - 36) = Real.sqrt 145 - Real.sqrt 13 := by
  sorry

end sqrt_difference_equality_l2143_214361


namespace equation_equivalence_l2143_214305

theorem equation_equivalence (x : ℝ) : 
  (x - 1) / 2 - (2 * x + 3) / 3 = 1 ↔ 3 * (x - 1) - 2 * (2 * x + 3) = 6 := by
  sorry

end equation_equivalence_l2143_214305


namespace factorial_fraction_is_integer_l2143_214386

/-- Given that m and n are non-negative integers and 0! = 1, 
    prove that (2m)!(2n)! / (m!n!(m+n)!) is an integer. -/
theorem factorial_fraction_is_integer (m n : ℕ) : 
  ∃ k : ℤ, (↑((2*m).factorial * (2*n).factorial) : ℚ) / 
    (↑(m.factorial * n.factorial * (m+n).factorial)) = ↑k :=
sorry

end factorial_fraction_is_integer_l2143_214386


namespace ab_difference_l2143_214315

theorem ab_difference (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a - b = 5 := by
  sorry

end ab_difference_l2143_214315


namespace milk_mixture_theorem_l2143_214389

/-- Proves that adding 24 gallons of 10% butterfat milk to 8 gallons of 50% butterfat milk 
    results in a mixture with 20% butterfat. -/
theorem milk_mixture_theorem :
  let initial_volume : ℝ := 8
  let initial_butterfat_percent : ℝ := 50
  let added_volume : ℝ := 24
  let added_butterfat_percent : ℝ := 10
  let desired_butterfat_percent : ℝ := 20
  let total_volume := initial_volume + added_volume
  let total_butterfat := (initial_volume * initial_butterfat_percent / 100) + 
                         (added_volume * added_butterfat_percent / 100)
  (total_butterfat / total_volume) * 100 = desired_butterfat_percent :=
by sorry

end milk_mixture_theorem_l2143_214389


namespace integer_fractional_parts_theorem_l2143_214338

theorem integer_fractional_parts_theorem : ∃ (x y : ℝ), 
  (x = ⌊8 - Real.sqrt 11⌋) ∧ 
  (y = 8 - Real.sqrt 11 - ⌊8 - Real.sqrt 11⌋) ∧ 
  (2 * x * y - y^2 = 5) := by
  sorry

end integer_fractional_parts_theorem_l2143_214338


namespace fraction_ordering_l2143_214336

theorem fraction_ordering : (6 : ℚ) / 29 < 8 / 25 ∧ 8 / 25 < 11 / 31 := by
  sorry

end fraction_ordering_l2143_214336


namespace sphere_to_hemisphere_radius_l2143_214350

/-- Given a sphere that transforms into a hemisphere, this theorem relates the radius of the 
    hemisphere to the radius of the original sphere. -/
theorem sphere_to_hemisphere_radius (r : ℝ) (h : r = 5 * Real.rpow 2 (1/3)) : 
  ∃ R : ℝ, R = 5 ∧ (4/3) * Real.pi * R^3 = (2/3) * Real.pi * r^3 :=
by sorry

end sphere_to_hemisphere_radius_l2143_214350


namespace geometric_difference_ratio_l2143_214323

def geometric_difference (a : ℕ+ → ℝ) (d : ℝ) :=
  ∀ n : ℕ+, a (n + 2) / a (n + 1) - a (n + 1) / a n = d

theorem geometric_difference_ratio 
  (a : ℕ+ → ℝ) 
  (h1 : geometric_difference a 2)
  (h2 : a 1 = 1)
  (h3 : a 2 = 1)
  (h4 : a 3 = 3) :
  a 12 / a 10 = 399 := by
sorry

end geometric_difference_ratio_l2143_214323


namespace base_conversion_digit_sum_l2143_214345

theorem base_conversion_digit_sum : 
  (∃ (d_min d_max : ℕ), 
    (∀ n : ℕ, 
      (9^3 ≤ n ∧ n < 9^4) → 
      (d_min ≤ Nat.log2 (n + 1) ∧ Nat.log2 (n + 1) ≤ d_max)) ∧
    (d_max - d_min = 2) ∧
    (d_min + (d_min + 1) + d_max = 33)) := by
  sorry

end base_conversion_digit_sum_l2143_214345


namespace not_always_prime_l2143_214394

theorem not_always_prime : ∃ n : ℕ, ¬ Nat.Prime (n^2 - n + 11) := by
  sorry

end not_always_prime_l2143_214394


namespace problem_solution_l2143_214369

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem problem_solution :
  (∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3)) ∧
  (∀ a : ℝ, (a > 0 ∧ (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x)) ↔ (1 ≤ a ∧ a ≤ 2)) :=
by sorry

end problem_solution_l2143_214369


namespace inequality_solution_set_l2143_214309

theorem inequality_solution_set (x : ℝ) : (-2 * x - 1 < -1) ↔ (x > 0) := by
  sorry

end inequality_solution_set_l2143_214309


namespace triangle_abc_problem_l2143_214335

theorem triangle_abc_problem (a b c A B C : ℝ) 
  (h1 : a * Real.sin A = 4 * b * Real.sin B)
  (h2 : a * c = Real.sqrt 5 * (a^2 - b^2 - c^2)) :
  Real.cos A = -(Real.sqrt 5) / 5 ∧ 
  Real.sin (2 * B - A) = -2 * (Real.sqrt 5) / 5 := by
sorry

end triangle_abc_problem_l2143_214335


namespace smaller_bedroom_size_l2143_214304

/-- Given two bedrooms with a total area of 300 square feet, where one bedroom
    is 60 square feet larger than the other, prove that the smaller bedroom
    is 120 square feet. -/
theorem smaller_bedroom_size (total_area : ℝ) (difference : ℝ) (smaller : ℝ) :
  total_area = 300 →
  difference = 60 →
  total_area = smaller + (smaller + difference) →
  smaller = 120 := by
  sorry

end smaller_bedroom_size_l2143_214304


namespace min_packs_for_144_cans_l2143_214355

/-- Represents the number of cans in each pack size --/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans for a given pack size --/
def cansInPack (s : PackSize) : Nat :=
  match s with
  | PackSize.small => 8
  | PackSize.medium => 18
  | PackSize.large => 30

/-- Represents a combination of packs --/
structure PackCombination where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of cans in a pack combination --/
def totalCans (c : PackCombination) : Nat :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a combination --/
def totalPacks (c : PackCombination) : Nat :=
  c.small + c.medium + c.large

/-- Defines what it means for a pack combination to be valid --/
def isValidCombination (c : PackCombination) : Prop :=
  totalCans c = 144

/-- Theorem: The minimum number of packs to buy 144 cans is 6 --/
theorem min_packs_for_144_cans :
  ∃ (c : PackCombination),
    isValidCombination c ∧
    totalPacks c = 6 ∧
    (∀ (c' : PackCombination), isValidCombination c' → totalPacks c' ≥ 6) :=
  sorry

end min_packs_for_144_cans_l2143_214355


namespace initial_carrots_count_l2143_214313

/-- Proves that the initial number of carrots is 300 given the problem conditions --/
theorem initial_carrots_count : ℕ :=
  let initial_carrots : ℕ := 300
  let before_lunch_fraction : ℚ := 2/5
  let after_lunch_fraction : ℚ := 3/5
  let unused_carrots : ℕ := 72

  have h1 : (1 - before_lunch_fraction) * (1 - after_lunch_fraction) * initial_carrots = unused_carrots := by sorry

  initial_carrots


end initial_carrots_count_l2143_214313


namespace max_stamps_is_125_l2143_214381

/-- The maximum number of stamps that can be purchased with a given budget. -/
def max_stamps (budget : ℕ) (price_low : ℕ) (price_high : ℕ) (threshold : ℕ) : ℕ :=
  max (min (budget / price_high) threshold) (budget / price_low)

/-- Proof that 125 stamps is the maximum number that can be purchased with 5000 cents,
    given the pricing conditions. -/
theorem max_stamps_is_125 :
  max_stamps 5000 40 45 100 = 125 := by
  sorry

end max_stamps_is_125_l2143_214381


namespace perpendicular_bisector_of_intersecting_circles_l2143_214353

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the intersection points A and B
def intersectionPoints (A B : ℝ × ℝ) : Prop :=
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
  circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
  A ≠ B

-- Define the perpendicular bisector
def perpendicularBisector (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersecting_circles
  (A B : ℝ × ℝ) (h : intersectionPoints A B) :
  ∀ x y, perpendicularBisector x y ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

end perpendicular_bisector_of_intersecting_circles_l2143_214353


namespace coin_probability_impossibility_l2143_214318

theorem coin_probability_impossibility : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
sorry

end coin_probability_impossibility_l2143_214318


namespace second_quadrant_complex_l2143_214382

theorem second_quadrant_complex (a b : ℝ) : 
  (Complex.ofReal a + Complex.I * Complex.ofReal b).im > 0 ∧ 
  (Complex.ofReal a + Complex.I * Complex.ofReal b).re < 0 → 
  a < 0 ∧ b > 0 := by sorry

end second_quadrant_complex_l2143_214382
