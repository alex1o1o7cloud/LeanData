import Mathlib

namespace NUMINAMATH_CALUDE_expression_equality_l1375_137548

theorem expression_equality : 
  (Real.sqrt 3 - Real.sqrt 2) * (-Real.sqrt 3 - Real.sqrt 2) + (3 + 2 * Real.sqrt 5)^2 = 28 + 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1375_137548


namespace NUMINAMATH_CALUDE_power_difference_equality_l1375_137537

theorem power_difference_equality : 4^(2+4+6) - (4^2 + 4^4 + 4^6) = 16772848 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equality_l1375_137537


namespace NUMINAMATH_CALUDE_only_coordinates_specific_l1375_137595

/-- Represents a location description --/
inductive LocationDescription
  | CinemaRow (row : Nat)
  | StreetAddress (street : String) (city : String)
  | Direction (angle : Float) (direction : String)
  | Coordinates (longitude : Float) (latitude : Float)

/-- Determines if a location description provides a specific, unique location --/
def isSpecificLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.Coordinates _ _ => True
  | _ => False

/-- Theorem stating that only the coordinates option provides a specific location --/
theorem only_coordinates_specific (desc : LocationDescription) :
  isSpecificLocation desc ↔ ∃ (long lat : Float), desc = LocationDescription.Coordinates long lat :=
sorry

#check only_coordinates_specific

end NUMINAMATH_CALUDE_only_coordinates_specific_l1375_137595


namespace NUMINAMATH_CALUDE_triangle_ABC_dot_product_l1375_137560

def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (5, 6)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem triangle_ABC_dot_product :
  dot_product vector_AB vector_AC = 9 := by sorry

end NUMINAMATH_CALUDE_triangle_ABC_dot_product_l1375_137560


namespace NUMINAMATH_CALUDE_marbles_remainder_l1375_137594

theorem marbles_remainder (n m k : ℤ) : (8*n + 5 + 7*m + 2 + 7*k + 4) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_marbles_remainder_l1375_137594


namespace NUMINAMATH_CALUDE_ella_coin_value_l1375_137561

/-- Represents the number of coins Ella has -/
def total_coins : ℕ := 18

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the number of nickels Ella has -/
def nickels : ℕ := sorry

/-- Represents the number of dimes Ella has -/
def dimes : ℕ := sorry

/-- The total number of coins is the sum of nickels and dimes -/
axiom coin_sum : nickels + dimes = total_coins

/-- If Ella had two more dimes, she would have an equal number of nickels and dimes -/
axiom equal_with_two_more : nickels = dimes + 2

/-- The theorem to be proved -/
theorem ella_coin_value : 
  nickels * nickel_value + dimes * dime_value = 130 :=
sorry

end NUMINAMATH_CALUDE_ella_coin_value_l1375_137561


namespace NUMINAMATH_CALUDE_collinear_vectors_cos_2theta_l1375_137551

/-- 
Given vectors AB and BC in 2D space and the condition that points A, B, and C are collinear,
prove that cos(2θ) = 7/9.
-/
theorem collinear_vectors_cos_2theta (θ : ℝ) :
  let AB : Fin 2 → ℝ := ![- 1, - 3]
  let BC : Fin 2 → ℝ := ![2 * Real.sin θ, 2]
  (∃ (k : ℝ), AB = k • BC) →
  Real.cos (2 * θ) = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_cos_2theta_l1375_137551


namespace NUMINAMATH_CALUDE_square_circle_union_area_l1375_137512

/-- The area of the union of a square with side length 8 and a circle with radius 12
    centered at the center of the square is equal to 144π. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let union_area : ℝ := max square_area circle_area
  union_area = 144 * π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l1375_137512


namespace NUMINAMATH_CALUDE_expected_distinct_faces_proof_l1375_137506

/-- A fair six-sided die is rolled six times. -/
def roll_die (n : ℕ) : Type := Fin 6 → Fin n

/-- The probability of a specific face not appearing in a single roll. -/
def prob_not_appear : ℚ := 5 / 6

/-- The expected number of distinct faces appearing in six rolls of a fair die. -/
def expected_distinct_faces : ℚ := (6^6 - 5^6) / 6^5

/-- Theorem stating that the expected number of distinct faces appearing when a fair
    six-sided die is rolled six times is equal to (6^6 - 5^6) / 6^5. -/
theorem expected_distinct_faces_proof :
  expected_distinct_faces = (6^6 - 5^6) / 6^5 :=
by sorry

end NUMINAMATH_CALUDE_expected_distinct_faces_proof_l1375_137506


namespace NUMINAMATH_CALUDE_three_roots_range_l1375_137549

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem three_roots_range (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = a ∧ f y = a ∧ f z = a) →
  -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_roots_range_l1375_137549


namespace NUMINAMATH_CALUDE_outfit_combinations_l1375_137596

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) : 
  shirts = 3 → pants = 4 → shirts * pants = 12 :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1375_137596


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_l1375_137553

theorem polynomial_equality_sum (a b c d : ℤ) : 
  (∀ x, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + x^3 - 2*x^2 + 17*x - 5) → 
  a + b + c + d = 5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_l1375_137553


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1375_137513

def M : Set ℤ := {-1, 0, 1, 2}

def f (x : ℤ) : ℤ := Int.natAbs x

def N : Set ℤ := f '' M

theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1375_137513


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l1375_137510

/-- Given a complex number z where |z - 3 + 2i| = 3, 
    the minimum value of |z + 1 - i|^2 + |z - 7 + 3i|^2 is 86. -/
theorem min_value_complex_expression (z : ℂ) 
  (h : Complex.abs (z - (3 - 2*Complex.I)) = 3) : 
  (Complex.abs (z + (1 - Complex.I)))^2 + (Complex.abs (z - (7 - 3*Complex.I)))^2 ≥ 86 ∧ 
  ∃ w : ℂ, Complex.abs (w - (3 - 2*Complex.I)) = 3 ∧ 
    (Complex.abs (w + (1 - Complex.I)))^2 + (Complex.abs (w - (7 - 3*Complex.I)))^2 = 86 :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l1375_137510


namespace NUMINAMATH_CALUDE_mike_speaker_cost_l1375_137562

/-- The amount Mike spent on speakers -/
def speaker_cost (total_cost new_tire_cost : ℚ) : ℚ :=
  total_cost - new_tire_cost

/-- Theorem: Mike spent $118.54 on speakers -/
theorem mike_speaker_cost : 
  speaker_cost 224.87 106.33 = 118.54 := by sorry

end NUMINAMATH_CALUDE_mike_speaker_cost_l1375_137562


namespace NUMINAMATH_CALUDE_chocolate_division_l1375_137518

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_given : ℕ) : 
  total_chocolate = 60 / 7 →
  num_piles = 5 →
  piles_given = 2 →
  (total_chocolate / num_piles) * piles_given = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l1375_137518


namespace NUMINAMATH_CALUDE_crystal_meal_options_l1375_137502

/-- Represents the available options for each part of the meal -/
structure MealOptions where
  entrees : Nat
  drinks : Nat
  desserts : Nat

/-- Represents the special condition for Apple Pie -/
structure SpecialCondition where
  icedTeaDesserts : Nat
  otherDrinksDesserts : Nat

/-- Calculates the number of distinct possible meals -/
def countMeals (options : MealOptions) (condition : SpecialCondition) : Nat :=
  options.entrees * (1 * condition.icedTeaDesserts + 
    (options.drinks - 1) * condition.otherDrinksDesserts)

/-- Theorem stating the number of distinct possible meals Crystal can buy -/
theorem crystal_meal_options : 
  let options := MealOptions.mk 4 3 3
  let condition := SpecialCondition.mk 1 2
  countMeals options condition = 20 := by
  sorry

end NUMINAMATH_CALUDE_crystal_meal_options_l1375_137502


namespace NUMINAMATH_CALUDE_smallest_max_sum_l1375_137573

theorem smallest_max_sum (a b c d e : ℕ+) 
  (sum_constraint : a + b + c + d + e = 3015) : 
  (∃ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (d + e))) ∧ 
   ∀ N : ℕ, (N = max (a + b) (max (b + c) (max (c + d) (d + e))) → M ≤ N) ∧
   M = 755) := by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l1375_137573


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l1375_137554

theorem sin_two_alpha_value (α : Real) 
  (h1 : 0 < α ∧ α < π) 
  (h2 : (1/2) * Real.cos (2*α) = Real.sin (π/4 + α)) : 
  Real.sin (2*α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l1375_137554


namespace NUMINAMATH_CALUDE_basketball_tournament_l1375_137569

/-- Number of classes in the tournament -/
def num_classes : ℕ := 10

/-- Total number of matches in the tournament -/
def total_matches : ℕ := 45

/-- Points earned for winning a game -/
def win_points : ℕ := 2

/-- Points earned for losing a game -/
def lose_points : ℕ := 1

/-- Minimum points target for a class -/
def min_points : ℕ := 14

/-- Theorem stating the number of classes and minimum wins required -/
theorem basketball_tournament :
  (num_classes * (num_classes - 1)) / 2 = total_matches ∧
  ∃ (min_wins : ℕ), 
    min_wins * win_points + (num_classes - 1 - min_wins) * lose_points ≥ min_points ∧
    ∀ (wins : ℕ), wins < min_wins → 
      wins * win_points + (num_classes - 1 - wins) * lose_points < min_points :=
by sorry

end NUMINAMATH_CALUDE_basketball_tournament_l1375_137569


namespace NUMINAMATH_CALUDE_quadratic_roots_coefficients_relation_l1375_137523

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Represents the roots of a quadratic equation -/
structure QuadraticRoots where
  α : ℝ
  β : ℝ

/-- Theorem stating the relationship between roots and coefficients of a quadratic equation -/
theorem quadratic_roots_coefficients_relation 
  (eq : QuadraticEquation) 
  (roots : QuadraticRoots) 
  (h1 : eq.a * roots.α^2 + eq.b * roots.α + eq.c = 0)
  (h2 : eq.a * roots.β^2 + eq.b * roots.β + eq.c = 0) :
  roots.α + roots.β = -eq.b / eq.a ∧ roots.α * roots.β = eq.c / eq.a := by
  sorry

#check quadratic_roots_coefficients_relation

end NUMINAMATH_CALUDE_quadratic_roots_coefficients_relation_l1375_137523


namespace NUMINAMATH_CALUDE_banker_cannot_guarantee_2kg_l1375_137555

/-- Represents the state of the banker's sand and exchange rates -/
structure SandState where
  g : ℕ -- Exchange rate for gold
  p : ℕ -- Exchange rate for platinum
  G : ℚ -- Amount of gold sand in kg
  P : ℚ -- Amount of platinum sand in kg

/-- Calculates the invariant metric S for a given SandState -/
def calcMetric (state : SandState) : ℚ :=
  state.G * state.p + state.P * state.g

/-- Represents the daily change in exchange rates -/
inductive DailyChange
  | decreaseG
  | decreaseP

/-- Applies a daily change to the SandState -/
def applyDailyChange (state : SandState) (change : DailyChange) : SandState :=
  match change with
  | DailyChange.decreaseG => { state with g := state.g - 1 }
  | DailyChange.decreaseP => { state with p := state.p - 1 }

/-- Theorem stating that the banker cannot guarantee 2 kg of each sand type after 2000 days -/
theorem banker_cannot_guarantee_2kg (initialState : SandState)
  (h_initial_g : initialState.g = 1001)
  (h_initial_p : initialState.p = 1001)
  (h_initial_G : initialState.G = 1)
  (h_initial_P : initialState.P = 1) :
  ¬ ∃ (finalState : SandState),
    (∃ (changes : List DailyChange),
      changes.length = 2000 ∧
      finalState = changes.foldl applyDailyChange initialState ∧
      finalState.g = 1 ∧ finalState.p = 1) ∧
    finalState.G ≥ 2 ∧ finalState.P ≥ 2 :=
  sorry

#check banker_cannot_guarantee_2kg

end NUMINAMATH_CALUDE_banker_cannot_guarantee_2kg_l1375_137555


namespace NUMINAMATH_CALUDE_tea_cost_price_l1375_137565

/-- Represents the cost price per kg of the 80 kg tea -/
def x : ℝ := 15

/-- The total amount of tea in kg -/
def total_tea : ℝ := 100

/-- The amount of tea with known cost price in kg -/
def known_tea : ℝ := 20

/-- The amount of tea with unknown cost price in kg -/
def unknown_tea : ℝ := 80

/-- The cost price per kg of the known tea -/
def known_tea_price : ℝ := 20

/-- The sale price per kg of the mixed tea -/
def sale_price : ℝ := 21.6

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.35

theorem tea_cost_price : 
  (unknown_tea * x + known_tea * known_tea_price) * (1 + profit_percentage) = 
  total_tea * sale_price := by sorry

end NUMINAMATH_CALUDE_tea_cost_price_l1375_137565


namespace NUMINAMATH_CALUDE_flower_beds_count_l1375_137542

theorem flower_beds_count (seeds_per_bed : ℕ) (total_seeds : ℕ) (num_beds : ℕ) : 
  seeds_per_bed = 10 → total_seeds = 60 → num_beds * seeds_per_bed = total_seeds → num_beds = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_flower_beds_count_l1375_137542


namespace NUMINAMATH_CALUDE_correct_average_marks_l1375_137588

theorem correct_average_marks (n : ℕ) (incorrect_avg : ℚ) (incorrect_mark correct_mark : ℚ) :
  n = 10 ∧ incorrect_avg = 100 ∧ incorrect_mark = 50 ∧ correct_mark = 10 →
  (n * incorrect_avg - (incorrect_mark - correct_mark)) / n = 96 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_marks_l1375_137588


namespace NUMINAMATH_CALUDE_path_count_l1375_137519

/-- The number of paths from A to each blue arrow -/
def paths_to_blue : ℕ := 2

/-- The number of blue arrows -/
def num_blue_arrows : ℕ := 2

/-- The number of distinct ways from each blue arrow to each green arrow -/
def paths_blue_to_green : ℕ := 3

/-- The number of green arrows -/
def num_green_arrows : ℕ := 2

/-- The number of distinct final approaches from each green arrow to C -/
def paths_green_to_C : ℕ := 2

/-- The total number of paths from A to C -/
def total_paths : ℕ := 
  paths_to_blue * num_blue_arrows * 
  (paths_blue_to_green * num_blue_arrows) * num_green_arrows * 
  paths_green_to_C

theorem path_count : total_paths = 288 := by
  sorry

end NUMINAMATH_CALUDE_path_count_l1375_137519


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1375_137585

theorem geometric_sequence_sixth_term
  (a : ℕ+) -- first term
  (r : ℝ) -- common ratio
  (h1 : a = 3)
  (h2 : a * r^3 = 243) -- fourth term condition
  : a * r^5 = 729 := by -- sixth term
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1375_137585


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1375_137505

theorem inequality_equivalence :
  (∀ x : ℝ, |x + 1| + |x - 1| ≥ a) ↔ (a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1375_137505


namespace NUMINAMATH_CALUDE_inequalities_theorem_l1375_137566

theorem inequalities_theorem :
  (∀ a b c d : ℝ, a > b → c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a > b ∧ c > d ∧ a * c ≤ b * d) ∧
  (∃ a b c : ℝ, a < b ∧ a * c^2 ≥ b * c^2) ∧
  (∀ a b c : ℝ, a > b → b > 0 → c < 0 → c / a > c / b) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l1375_137566


namespace NUMINAMATH_CALUDE_toy_car_cost_l1375_137507

theorem toy_car_cost (initial_amount : ℕ) (num_cars : ℕ) (scarf_cost : ℕ) (beanie_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 53 →
  num_cars = 2 →
  scarf_cost = 10 →
  beanie_cost = 14 →
  remaining_amount = 7 →
  (initial_amount - remaining_amount - scarf_cost - beanie_cost) / num_cars = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_car_cost_l1375_137507


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1375_137525

/-- Represents a repeating decimal with a single digit repetend -/
def repeating_decimal (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with a two-digit repetend -/
def repeating_decimal_two_digits (n : ℕ) : ℚ := n / 99

theorem repeating_decimal_sum :
  repeating_decimal 6 + repeating_decimal_two_digits 12 - repeating_decimal 4 = 34 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1375_137525


namespace NUMINAMATH_CALUDE_expression_evaluation_l1375_137540

theorem expression_evaluation : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1375_137540


namespace NUMINAMATH_CALUDE_max_distance_point_to_line_l1375_137577

noncomputable def point_to_line_distance (θ : ℝ) : ℝ :=
  |3 * Real.cos θ + 4 * Real.sin θ - 4| / 5

theorem max_distance_point_to_line :
  ∃ (θ : ℝ), ∀ (φ : ℝ), point_to_line_distance θ ≥ point_to_line_distance φ ∧
  point_to_line_distance θ = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_point_to_line_l1375_137577


namespace NUMINAMATH_CALUDE_hyperbola_equation_with_eccentricity_hyperbola_equation_with_asymptote_l1375_137530

-- Define the hyperbola C
structure Hyperbola where
  center : ℝ × ℝ
  right_focus : ℝ × ℝ

-- Define the standard form of a hyperbola equation
def standard_form (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x^2 / a^2 - y^2 / b^2 = 1

-- Theorem for the first part of the problem
theorem hyperbola_equation_with_eccentricity 
  (C : Hyperbola) 
  (h_center : C.center = (0, 0))
  (h_focus : C.right_focus = (Real.sqrt 3, 0))
  (h_eccentricity : ∃ e, e = Real.sqrt 3) :
  ∃ a b, standard_form a b = λ x y => x^2 - y^2 / 2 = 1 :=
sorry

-- Theorem for the second part of the problem
theorem hyperbola_equation_with_asymptote
  (C : Hyperbola)
  (h_center : C.center = (0, 0))
  (h_focus : C.right_focus = (Real.sqrt 3, 0))
  (h_asymptote : ∃ X Y, X + Real.sqrt 2 * Y = 0) :
  ∃ a b, standard_form a b = λ x y => x^2 / 2 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_with_eccentricity_hyperbola_equation_with_asymptote_l1375_137530


namespace NUMINAMATH_CALUDE_max_handshakes_l1375_137527

theorem max_handshakes (n : ℕ) (h : n = 25) : 
  (n * (n - 1)) / 2 = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_max_handshakes_l1375_137527


namespace NUMINAMATH_CALUDE_translated_function_eq_l1375_137583

-- Define the original function
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 1

-- Define the translated function
def g (x : ℝ) : ℝ := f (x + 1) + 3

-- Theorem stating that the translated function is equal to 3x^2 - 1
theorem translated_function_eq (x : ℝ) : g x = 3 * x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_translated_function_eq_l1375_137583


namespace NUMINAMATH_CALUDE_triangle_angle_max_value_l1375_137531

theorem triangle_angle_max_value (A B C : ℝ) : 
  A + B + C = π →
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  (Real.sqrt 3 * Real.cos A + Real.sin A) / (Real.sqrt 3 * Real.sin A - Real.cos A) = Real.tan (-7 * π / 12) →
  ∃ (x : ℝ), x = 2 * Real.cos B + Real.sin (2 * C) ∧ x ≤ 3 / 2 ∧ 
  ∀ (y : ℝ), y = 2 * Real.cos B + Real.sin (2 * C) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_max_value_l1375_137531


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l1375_137526

theorem quadratic_root_zero (m : ℝ) : 
  (∃ x, (m - 3) * x^2 + x + m^2 - 9 = 0) ∧ 
  ((m - 3) * 0^2 + 0 + m^2 - 9 = 0) →
  m = -3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l1375_137526


namespace NUMINAMATH_CALUDE_puzzle_pieces_missing_l1375_137536

theorem puzzle_pieces_missing (total : ℕ) (border : ℕ) (trevor : ℕ) (joe : ℕ) 
  (h1 : total = 500)
  (h2 : border = 75)
  (h3 : trevor = 105)
  (h4 : joe = 3 * trevor) :
  total - (border + trevor + joe) = 5 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_pieces_missing_l1375_137536


namespace NUMINAMATH_CALUDE_power_equation_solution_l1375_137581

theorem power_equation_solution :
  ∃ x : ℕ, (1000 : ℝ)^7 / (10 : ℝ)^x = 10000 ∧ x = 17 := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1375_137581


namespace NUMINAMATH_CALUDE_circle_power_theorem_l1375_137544

/-- The power of a point with respect to a circle -/
def power (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : ℝ :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 - radius^2

theorem circle_power_theorem (k : ℝ) (hk : k < 0) :
  (∃ p : ℝ × ℝ, power (0, 0) 1 p = k) ∧
  ¬(∀ k : ℝ, k < 0 → ∃ q : ℝ × ℝ, power (0, 0) 1 q = -k) :=
by sorry

end NUMINAMATH_CALUDE_circle_power_theorem_l1375_137544


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l1375_137580

def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (a+2)*x + 2*a^2 - a + 1 = 0}

theorem set_intersection_and_union (a : ℝ) :
  (A ∩ B a = {2} → a = 1/2) ∧
  (A ∪ B a = A → a ≤ 0 ∨ a = 1 ∨ a > 8/7) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l1375_137580


namespace NUMINAMATH_CALUDE_sock_ratio_proof_l1375_137538

theorem sock_ratio_proof (green_socks red_socks : ℕ) (price_red : ℚ) :
  green_socks = 6 →
  (6 * (3 * price_red) + red_socks * price_red + 15 : ℚ) * (9/5) = 
    red_socks * (3 * price_red) + 6 * price_red + 15 →
  (green_socks : ℚ) / red_socks = 6 / 23 :=
by
  sorry

end NUMINAMATH_CALUDE_sock_ratio_proof_l1375_137538


namespace NUMINAMATH_CALUDE_max_coverage_of_two_inch_card_l1375_137511

/-- A checkerboard square -/
structure CheckerboardSquare where
  size : Real
  (size_positive : size > 0)

/-- A square card -/
structure SquareCard where
  side_length : Real
  (side_length_positive : side_length > 0)

/-- Represents the coverage of a card on a checkerboard -/
def Coverage (card : SquareCard) (square : CheckerboardSquare) : Nat :=
  sorry

/-- Theorem: The maximum number of one-inch squares on a checkerboard 
    that can be covered by a 2-inch square card is 12 -/
theorem max_coverage_of_two_inch_card : 
  ∀ (board_square : CheckerboardSquare) (card : SquareCard),
    board_square.size = 1 → 
    card.side_length = 2 → 
    ∃ (n : Nat), Coverage card board_square = n ∧ 
      ∀ (m : Nat), Coverage card board_square ≤ m → n ≤ m ∧ n = 12 :=
sorry

end NUMINAMATH_CALUDE_max_coverage_of_two_inch_card_l1375_137511


namespace NUMINAMATH_CALUDE_unique_quadratic_with_real_roots_l1375_137529

/-- A geometric progression of length 2016 -/
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i ∈ Finset.range 2015, a (i + 1) = r * a i

/-- An arithmetic progression of length 2016 -/
def arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i ∈ Finset.range 2015, b (i + 1) = b i + d

/-- The quadratic trinomial P_i(x) = x^2 + a_i * x + b_i -/
def P (a b : ℕ → ℝ) (i : ℕ) (x : ℝ) : ℝ :=
  x^2 + a i * x + b i

/-- P_k(x) has real roots iff its discriminant is non-negative -/
def has_real_roots (a b : ℕ → ℝ) (k : ℕ) : Prop :=
  (a k)^2 - 4 * b k ≥ 0

theorem unique_quadratic_with_real_roots
  (a b : ℕ → ℝ)
  (h_geom : geometric_progression a)
  (h_arith : arithmetic_progression b)
  (h_unique : ∃! k : ℕ, k ∈ Finset.range 2016 ∧ has_real_roots a b k) :
  ∃ k : ℕ, (k = 1 ∨ k = 2016) ∧ k ∈ Finset.range 2016 ∧ has_real_roots a b k :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_with_real_roots_l1375_137529


namespace NUMINAMATH_CALUDE_joans_initial_balloons_count_l1375_137576

/-- The number of blue balloons Joan had initially -/
def joans_initial_balloons : ℕ := 9

/-- The number of balloons Sally popped -/
def popped_balloons : ℕ := 5

/-- The number of blue balloons Jessica has -/
def jessicas_balloons : ℕ := 2

/-- The total number of blue balloons they have now -/
def total_balloons_now : ℕ := 6

theorem joans_initial_balloons_count : 
  joans_initial_balloons = popped_balloons + (total_balloons_now - jessicas_balloons) :=
by sorry

end NUMINAMATH_CALUDE_joans_initial_balloons_count_l1375_137576


namespace NUMINAMATH_CALUDE_three_people_seven_steps_l1375_137514

/-- The number of ways to arrange n people on m steps with at most k people per step -/
def arrange (n m k : ℕ) : ℕ :=
  sorry

/-- The number of ways to arrange 3 people on 7 steps with at most 2 people per step -/
theorem three_people_seven_steps : arrange 3 7 2 = 336 := by
  sorry

end NUMINAMATH_CALUDE_three_people_seven_steps_l1375_137514


namespace NUMINAMATH_CALUDE_louisa_travel_speed_l1375_137575

/-- The average speed of Louisa's travel -/
def average_speed : ℝ := 37.5

/-- The distance traveled on the first day -/
def distance_day1 : ℝ := 375

/-- The distance traveled on the second day -/
def distance_day2 : ℝ := 525

/-- The time difference between the two trips -/
def time_difference : ℝ := 4

theorem louisa_travel_speed :
  (distance_day2 / average_speed) = (distance_day1 / average_speed) + time_difference :=
sorry

end NUMINAMATH_CALUDE_louisa_travel_speed_l1375_137575


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_l1375_137547

theorem smallest_sum_of_factors (x y z w : ℕ+) : 
  x * y * z * w = 362880 → 
  ∀ a b c d : ℕ+, a * b * c * d = 362880 → x + y + z + w ≤ a + b + c + d →
  x + y + z + w = 69 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_l1375_137547


namespace NUMINAMATH_CALUDE_square_plaza_area_l1375_137578

/-- The area of a square plaza with side length 5 × 10^2 m is 2.5 × 10^5 m^2. -/
theorem square_plaza_area :
  let side_length : ℝ := 5 * 10^2
  let area : ℝ := side_length^2
  area = 2.5 * 10^5 := by sorry

end NUMINAMATH_CALUDE_square_plaza_area_l1375_137578


namespace NUMINAMATH_CALUDE_rainfall_problem_l1375_137597

/-- Rainfall problem -/
theorem rainfall_problem (total_time : ℕ) (total_rainfall : ℕ) 
  (storm1_rate : ℕ) (storm1_duration : ℕ) :
  total_time = 45 →
  total_rainfall = 975 →
  storm1_rate = 30 →
  storm1_duration = 20 →
  ∃ storm2_rate : ℕ, 
    storm2_rate * (total_time - storm1_duration) = 
      total_rainfall - (storm1_rate * storm1_duration) ∧
    storm2_rate = 15 := by
  sorry


end NUMINAMATH_CALUDE_rainfall_problem_l1375_137597


namespace NUMINAMATH_CALUDE_daily_income_ratio_l1375_137589

/-- The ratio of daily income in a business where:
    - Initial income on day 1 is 3
    - Income on day 15 is 36
    - Each day's income is a multiple of the previous day's income
-/
theorem daily_income_ratio : ∃ (r : ℝ), 
  r > 0 ∧ 
  3 * r^14 = 36 ∧ 
  r = 2^(1/7) * 3^(1/14) := by
  sorry

end NUMINAMATH_CALUDE_daily_income_ratio_l1375_137589


namespace NUMINAMATH_CALUDE_median_exists_for_seven_prices_l1375_137521

theorem median_exists_for_seven_prices (prices : List ℝ) (h : prices.length = 7) :
  ∃ (median : ℝ), median ∈ prices ∧ 
    (prices.filter (λ x => x ≤ median)).length ≥ 4 ∧
    (prices.filter (λ x => x ≥ median)).length ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_median_exists_for_seven_prices_l1375_137521


namespace NUMINAMATH_CALUDE_probability_is_sqrt_two_over_fifteen_l1375_137584

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability of x^2 < y for a point (x,y) randomly picked from the given rectangle --/
def probability_x_squared_less_than_y (rect : Rectangle) : ℝ :=
  sorry

/-- The specific rectangle in the problem --/
def problem_rectangle : Rectangle :=
  { x_min := 0
  , x_max := 5
  , y_min := 0
  , y_max := 2
  , h_x := by norm_num
  , h_y := by norm_num
  }

theorem probability_is_sqrt_two_over_fifteen :
  probability_x_squared_less_than_y problem_rectangle = Real.sqrt 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_sqrt_two_over_fifteen_l1375_137584


namespace NUMINAMATH_CALUDE_accessories_count_proof_l1375_137500

/-- The number of accessories included for each doll in a factory production. -/
def accessories_per_doll : ℕ := 11

/-- The number of dolls produced. -/
def total_dolls : ℕ := 12000

/-- The time in seconds to make one doll. -/
def time_per_doll : ℕ := 45

/-- The time in seconds to make one accessory. -/
def time_per_accessory : ℕ := 10

/-- The total machine operation time in seconds. -/
def total_time : ℕ := 1860000

theorem accessories_count_proof :
  total_dolls * (time_per_doll + accessories_per_doll * time_per_accessory) = total_time :=
by sorry

end NUMINAMATH_CALUDE_accessories_count_proof_l1375_137500


namespace NUMINAMATH_CALUDE_rotate90_of_4_minus_2i_l1375_137587

def rotate90 (z : ℂ) : ℂ := z * Complex.I

theorem rotate90_of_4_minus_2i : 
  rotate90 (4 - 2 * Complex.I) = 2 + 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_rotate90_of_4_minus_2i_l1375_137587


namespace NUMINAMATH_CALUDE_watermelon_seeds_l1375_137550

theorem watermelon_seeds (total_slices : ℕ) (black_seeds_per_slice : ℕ) (total_seeds : ℕ) :
  total_slices = 40 →
  black_seeds_per_slice = 20 →
  total_seeds = 1600 →
  (total_seeds - total_slices * black_seeds_per_slice) / total_slices = 20 :=
by sorry

end NUMINAMATH_CALUDE_watermelon_seeds_l1375_137550


namespace NUMINAMATH_CALUDE_prime_arithmetic_seq_large_diff_l1375_137557

/-- A sequence of 15 different positive prime numbers in arithmetic progression -/
structure PrimeArithmeticSequence where
  terms : Fin 15 → ℕ
  is_prime : ∀ i, Nat.Prime (terms i)
  is_arithmetic : ∀ i j k, i.val + k.val = j.val → terms i + terms k = 2 * terms j
  is_distinct : ∀ i j, i ≠ j → terms i ≠ terms j

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : PrimeArithmeticSequence) : ℕ :=
  seq.terms 1 - seq.terms 0

/-- Theorem: The common difference of a sequence of 15 different positive primes
    in arithmetic progression is greater than 30000 -/
theorem prime_arithmetic_seq_large_diff (seq : PrimeArithmeticSequence) :
  common_difference seq > 30000 := by
  sorry

end NUMINAMATH_CALUDE_prime_arithmetic_seq_large_diff_l1375_137557


namespace NUMINAMATH_CALUDE_shirt_cost_l1375_137515

theorem shirt_cost (total_cost pants_cost tie_cost : ℕ) 
  (h1 : total_cost = 198)
  (h2 : pants_cost = 140)
  (h3 : tie_cost = 15) :
  total_cost - pants_cost - tie_cost = 43 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l1375_137515


namespace NUMINAMATH_CALUDE_ram_pairs_sold_l1375_137501

/-- Represents the sales and earnings of a hardware store for a week. -/
structure StoreSales where
  graphics_cards : Nat
  hard_drives : Nat
  cpus : Nat
  ram_pairs : Nat
  graphics_card_price : Nat
  hard_drive_price : Nat
  cpu_price : Nat
  ram_pair_price : Nat
  total_earnings : Nat

/-- Calculates the total earnings from a given StoreSales. -/
def calculate_earnings (sales : StoreSales) : Nat :=
  sales.graphics_cards * sales.graphics_card_price +
  sales.hard_drives * sales.hard_drive_price +
  sales.cpus * sales.cpu_price +
  sales.ram_pairs * sales.ram_pair_price

/-- Theorem stating that the number of RAM pairs sold is 4. -/
theorem ram_pairs_sold (sales : StoreSales) :
  sales.graphics_cards = 10 →
  sales.hard_drives = 14 →
  sales.cpus = 8 →
  sales.graphics_card_price = 600 →
  sales.hard_drive_price = 80 →
  sales.cpu_price = 200 →
  sales.ram_pair_price = 60 →
  sales.total_earnings = 8960 →
  calculate_earnings sales = sales.total_earnings →
  sales.ram_pairs = 4 := by
  sorry


end NUMINAMATH_CALUDE_ram_pairs_sold_l1375_137501


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l1375_137572

/-- The number of sides in a regular pentadecagon -/
def n : ℕ := 15

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem pentadecagon_triangles : num_triangles = 455 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l1375_137572


namespace NUMINAMATH_CALUDE_sale_markdown_l1375_137558

theorem sale_markdown (regular_price sale_price : ℝ) 
  (h : sale_price * (1 + 0.25) = regular_price) :
  (regular_price - sale_price) / regular_price = 0.2 := by
sorry

end NUMINAMATH_CALUDE_sale_markdown_l1375_137558


namespace NUMINAMATH_CALUDE_probability_in_given_scenario_l1375_137598

/-- Represents the probability of drawing a genuine product after drawing a defective one -/
def probability_genuine_after_defective (total : ℕ) (genuine : ℕ) (defective : ℕ) : ℚ :=
  if total = genuine + defective ∧ defective > 0 then
    genuine / (total - 1)
  else
    0

/-- The main theorem about the probability in the given scenario -/
theorem probability_in_given_scenario :
  probability_genuine_after_defective 7 4 3 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_given_scenario_l1375_137598


namespace NUMINAMATH_CALUDE_newspaper_photos_l1375_137532

/-- The total number of photos in a newspaper with specified page types -/
def total_photos (pages_with_two_photos pages_with_three_photos : ℕ) : ℕ :=
  2 * pages_with_two_photos + 3 * pages_with_three_photos

/-- Theorem stating that the total number of photos in the newspaper is 51 -/
theorem newspaper_photos : total_photos 12 9 = 51 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_photos_l1375_137532


namespace NUMINAMATH_CALUDE_car_count_total_l1375_137556

/-- Given the car counting scenario, prove the total count of cars. -/
theorem car_count_total (jared_count : ℕ) (ann_count : ℕ) (alfred_count : ℕ) :
  jared_count = 300 →
  ann_count = (115 * jared_count) / 100 →
  alfred_count = ann_count - 7 →
  jared_count + ann_count + alfred_count = 983 :=
by sorry

end NUMINAMATH_CALUDE_car_count_total_l1375_137556


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1375_137564

/-- The standard equation of a hyperbola with specific properties -/
theorem hyperbola_equation : ∃ (a b : ℝ), 
  (a > 0 ∧ b > 0) ∧
  (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ 
    -- Same asymptotes as x^2/3 - y^2 = 1
    (∀ t : ℝ, (y = t*x ↔ t^2 = 1/3)) ∧
    -- Focus distance from asymptote is 2
    (∃ c : ℝ, c / Real.sqrt (1 + (1/Real.sqrt 3)^2) = 2 ∧
              c^2 = a^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1375_137564


namespace NUMINAMATH_CALUDE_sum_reciprocals_S_l1375_137508

def S : Set ℕ+ := {n : ℕ+ | ∀ p : ℕ, Nat.Prime p → p ∣ n → p = 2 ∨ p = 2017}

theorem sum_reciprocals_S : ∑' (s : S), (1 : ℝ) / (s : ℝ) = 2017 / 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_S_l1375_137508


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1375_137590

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ)^4 * (3 : ℝ)^x = 81 ∧ x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1375_137590


namespace NUMINAMATH_CALUDE_mabel_tomatoes_l1375_137567

/-- The number of tomato plants Mabel planted -/
def num_plants : ℕ := 4

/-- The number of tomatoes on the first plant -/
def first_plant_tomatoes : ℕ := 8

/-- The number of additional tomatoes on the second plant compared to the first -/
def second_plant_additional : ℕ := 4

/-- The factor by which the remaining plants' tomatoes exceed the sum of the first two plants -/
def remaining_plants_factor : ℕ := 3

/-- The total number of tomatoes Mabel has -/
def total_tomatoes : ℕ := 140

theorem mabel_tomatoes :
  let second_plant_tomatoes := first_plant_tomatoes + second_plant_additional
  let first_two_plants := first_plant_tomatoes + second_plant_tomatoes
  let remaining_plants_tomatoes := 2 * (remaining_plants_factor * first_two_plants)
  first_plant_tomatoes + second_plant_tomatoes + remaining_plants_tomatoes = total_tomatoes :=
by sorry

end NUMINAMATH_CALUDE_mabel_tomatoes_l1375_137567


namespace NUMINAMATH_CALUDE_remainder_sum_powers_mod_seven_l1375_137503

theorem remainder_sum_powers_mod_seven :
  (9^6 + 8^7 + 7^8) % 7 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_powers_mod_seven_l1375_137503


namespace NUMINAMATH_CALUDE_cube_root_fraction_equivalence_l1375_137563

theorem cube_root_fraction_equivalence :
  let x : ℝ := 12.75
  let y : ℚ := 51 / 4
  x = y →
  (6 / x) ^ (1/3 : ℝ) = 2 / (17 ^ (1/3 : ℝ)) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_fraction_equivalence_l1375_137563


namespace NUMINAMATH_CALUDE_suv_highway_mpg_l1375_137582

/-- The average miles per gallon (mpg) on the highway for an SUV -/
def highway_mpg : ℝ := 12.2

/-- The maximum distance in miles that the SUV can travel on 25 gallons of gasoline -/
def max_distance : ℝ := 305

/-- The amount of gasoline in gallons used to calculate the maximum distance -/
def gasoline_amount : ℝ := 25

theorem suv_highway_mpg :
  highway_mpg = max_distance / gasoline_amount :=
by sorry

end NUMINAMATH_CALUDE_suv_highway_mpg_l1375_137582


namespace NUMINAMATH_CALUDE_a_values_l1375_137520

def A : Set ℝ := {x | 2 * x^2 - 7 * x - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem a_values (h : B a ⊆ A) : a = 0 ∨ a = -2 ∨ a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_a_values_l1375_137520


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l1375_137559

theorem square_perimeter_relation (C D : Real) : 
  (C = 16) → -- perimeter of square C is 16 cm
  (D^2 = (C/4)^2 / 3) → -- area of D is one-third the area of C
  (4 * D = 16 * Real.sqrt 3 / 3) -- perimeter of D is 16√3/3 cm
  := by sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l1375_137559


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1375_137579

theorem sum_of_three_numbers (x y z : ℝ) 
  (sum_xy : x + y = 29)
  (sum_yz : y + z = 46)
  (sum_zx : z + x = 53) :
  x + y + z = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1375_137579


namespace NUMINAMATH_CALUDE_distribute_negative_five_l1375_137522

theorem distribute_negative_five (x y : ℝ) : -5 * (x - y) = -5 * x + 5 * y := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_five_l1375_137522


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1375_137570

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1375_137570


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l1375_137586

theorem cube_volume_ratio : 
  let cube1_side_length : ℝ := 2  -- in meters
  let cube2_side_length : ℝ := 100 / 100  -- 100 cm converted to meters
  let cube1_volume := cube1_side_length ^ 3
  let cube2_volume := cube2_side_length ^ 3
  cube1_volume / cube2_volume = 8 := by sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l1375_137586


namespace NUMINAMATH_CALUDE_complex_magnitude_l1375_137591

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I) = 4 - 2 * Complex.I) :
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1375_137591


namespace NUMINAMATH_CALUDE_range_of_3a_minus_b_l1375_137545

theorem range_of_3a_minus_b (a b : ℝ) 
  (h1 : -1 < a + b ∧ a + b < 3) 
  (h2 : 2 < a - b ∧ a - b < 4) : 
  (∃ (x y : ℝ), (x = a ∧ y = b) ∧ 3*x - y = 3) ∧ 
  (∃ (x y : ℝ), (x = a ∧ y = b) ∧ 3*x - y = 11) ∧
  (∀ (x y : ℝ), (x = a ∧ y = b) → 3 ≤ 3*x - y ∧ 3*x - y ≤ 11) :=
sorry

end NUMINAMATH_CALUDE_range_of_3a_minus_b_l1375_137545


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1375_137535

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) :
  x₁^2 - 5*x₁ + 4 = 0 →
  x₂^2 - 5*x₂ + 4 = 0 →
  x₁ + x₂ = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1375_137535


namespace NUMINAMATH_CALUDE_tan_value_for_special_condition_l1375_137593

theorem tan_value_for_special_condition (a : Real) 
  (h1 : 0 < a ∧ a < π / 2) 
  (h2 : Real.sin a ^ 2 + Real.cos (2 * a) = 1) : 
  Real.tan a = 0 := by
sorry

end NUMINAMATH_CALUDE_tan_value_for_special_condition_l1375_137593


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_thirty_l1375_137504

theorem prime_square_minus_one_divisible_by_thirty {p : ℕ} (hp : Prime p) (hp_ge_7 : p ≥ 7) :
  30 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_thirty_l1375_137504


namespace NUMINAMATH_CALUDE_probability_both_win_is_one_third_l1375_137552

/-- Represents the three types of lottery tickets -/
inductive Ticket
  | FirstPrize
  | SecondPrize
  | NonPrize

/-- Represents a draw of two tickets without replacement -/
def Draw := (Ticket × Ticket)

/-- The set of all possible draws -/
def allDraws : Finset Draw := sorry

/-- Predicate to check if a draw results in both people winning a prize -/
def bothWinPrize (draw : Draw) : Prop := 
  draw.1 ≠ Ticket.NonPrize ∧ draw.2 ≠ Ticket.NonPrize

/-- The set of draws where both people win a prize -/
def winningDraws : Finset Draw := sorry

/-- The probability of both people winning a prize -/
def probabilityBothWin : ℚ := (winningDraws.card : ℚ) / (allDraws.card : ℚ)

theorem probability_both_win_is_one_third : 
  probabilityBothWin = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_both_win_is_one_third_l1375_137552


namespace NUMINAMATH_CALUDE_apple_cost_l1375_137516

/-- Given that apples cost m yuan per kilogram, prove that the cost of purchasing 3 kilograms of apples is 3m yuan. -/
theorem apple_cost (m : ℝ) : m * 3 = 3 * m := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_l1375_137516


namespace NUMINAMATH_CALUDE_max_subjects_per_teacher_l1375_137533

theorem max_subjects_per_teacher (maths_teachers physics_teachers chemistry_teachers min_teachers : ℕ)
  (h1 : maths_teachers = 11)
  (h2 : physics_teachers = 8)
  (h3 : chemistry_teachers = 5)
  (h4 : min_teachers = 8) :
  ∃ (max_subjects : ℕ), max_subjects = 3 ∧
    min_teachers * max_subjects ≥ maths_teachers + physics_teachers + chemistry_teachers ∧
    ∀ (x : ℕ), x > max_subjects → min_teachers * x > maths_teachers + physics_teachers + chemistry_teachers :=
by
  sorry

end NUMINAMATH_CALUDE_max_subjects_per_teacher_l1375_137533


namespace NUMINAMATH_CALUDE_sixteen_to_power_divided_by_eight_l1375_137592

theorem sixteen_to_power_divided_by_eight (n : ℕ) : n = 16^1024 → n / 8 = 2^4093 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_to_power_divided_by_eight_l1375_137592


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1375_137509

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + a ≥ 0) → a ∈ Set.Icc 0 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1375_137509


namespace NUMINAMATH_CALUDE_carl_reach_probability_l1375_137574

-- Define the lily pad setup
def num_pads : ℕ := 16
def predator_pads : List ℕ := [4, 7, 12]
def start_pad : ℕ := 0
def goal_pad : ℕ := 14

-- Define Carl's movement probabilities
def hop_prob : ℚ := 1/2
def leap_prob : ℚ := 1/2

-- Define a function to calculate the probability of reaching a specific pad
def reach_prob (pad : ℕ) : ℚ :=
  sorry

-- State the theorem
theorem carl_reach_probability :
  reach_prob goal_pad = 3/512 :=
sorry

end NUMINAMATH_CALUDE_carl_reach_probability_l1375_137574


namespace NUMINAMATH_CALUDE_student_sister_weight_ratio_l1375_137517

/-- Proves that the ratio of a student's weight after losing 5 kg to his sister's weight is 2:1 -/
theorem student_sister_weight_ratio 
  (student_weight : ℕ) 
  (total_weight : ℕ) 
  (weight_loss : ℕ) :
  student_weight = 75 →
  total_weight = 110 →
  weight_loss = 5 →
  (student_weight - weight_loss) / (total_weight - student_weight) = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_sister_weight_ratio_l1375_137517


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l1375_137599

/-- Proves that for a square plot with an area of 289 sq ft and a total fencing cost of Rs. 3876, the price per foot of fencing is Rs. 57. -/
theorem fence_cost_per_foot (area : ℝ) (total_cost : ℝ) (h1 : area = 289) (h2 : total_cost = 3876) :
  (total_cost / (4 * Real.sqrt area)) = 57 := by
sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l1375_137599


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1375_137534

theorem negation_of_proposition :
  (¬ ∀ n : ℕ, n^2 ≤ 2*n + 5) ↔ (∃ n : ℕ, n^2 > 2*n + 5) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1375_137534


namespace NUMINAMATH_CALUDE_inequality_proof_l1375_137546

theorem inequality_proof (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x^2 + y^2 + z^2 = 3) : 
  (x^2009 - 2008*(x-1))/(y+z) + (y^2009 - 2008*(y-1))/(x+z) + (z^2009 - 2008*(z-1))/(x+y) ≥ (1/2)*(x+y+z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1375_137546


namespace NUMINAMATH_CALUDE_sequence_sum_l1375_137568

theorem sequence_sum (a b c d : ℕ) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d →
  b - a = c - b →
  c * c = b * d →
  d - a = 30 →
  a + b + c + d = 129 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1375_137568


namespace NUMINAMATH_CALUDE_function_graph_relationship_l1375_137541

theorem function_graph_relationship (a : ℝ) (h1 : a > 0) :
  (∀ x : ℝ, x > 0 → Real.log x < a * x^2 - 1/2) → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_graph_relationship_l1375_137541


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l1375_137539

/-- The number of meters of cloth sold by a trader -/
def meters_of_cloth : ℕ := 85

/-- The total selling price in dollars -/
def total_selling_price : ℕ := 8925

/-- The profit per meter of cloth in dollars -/
def profit_per_meter : ℕ := 15

/-- The cost price per meter of cloth in dollars -/
def cost_price_per_meter : ℕ := 90

/-- Theorem stating that the number of meters of cloth sold is correct -/
theorem cloth_sale_calculation :
  meters_of_cloth * (cost_price_per_meter + profit_per_meter) = total_selling_price :=
by sorry

end NUMINAMATH_CALUDE_cloth_sale_calculation_l1375_137539


namespace NUMINAMATH_CALUDE_inductive_reasoning_classification_l1375_137524

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| NonInductive

-- Define the types of inductive reasoning
inductive InductiveReasoningType
| Generalization
| Analogy

-- Define a structure for an inference
structure Inference where
  description : String
  reasoning_type : ReasoningType
  inductive_type : Option InductiveReasoningType

-- Define the four inferences
def inference1 : Inference :=
  { description := "Inferring properties of a ball by analogy with properties of a circle",
    reasoning_type := ReasoningType.Inductive,
    inductive_type := some InductiveReasoningType.Analogy }

def inference2 : Inference :=
  { description := "Inferring that the sum of the internal angles of all triangles is 180° by induction from the sum of the internal angles of right triangles, isosceles triangles, and equilateral triangles",
    reasoning_type := ReasoningType.Inductive,
    inductive_type := some InductiveReasoningType.Generalization }

def inference3 : Inference :=
  { description := "Inferring that all students in the class scored 100 points because Zhang Jun scored 100 points in an exam",
    reasoning_type := ReasoningType.NonInductive,
    inductive_type := none }

def inference4 : Inference :=
  { description := "Inferring the formula of each term of the sequence 1, 0, 1, 0, ... as a_n = 1/2 + (-1)^(n+1) · 1/2",
    reasoning_type := ReasoningType.Inductive,
    inductive_type := some InductiveReasoningType.Generalization }

-- Theorem to prove
theorem inductive_reasoning_classification :
  (inference1.reasoning_type = ReasoningType.Inductive) ∧
  (inference2.reasoning_type = ReasoningType.Inductive) ∧
  (inference4.reasoning_type = ReasoningType.Inductive) :=
sorry

end NUMINAMATH_CALUDE_inductive_reasoning_classification_l1375_137524


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l1375_137543

theorem no_real_solution_for_log_equation :
  ¬ ∃ x : ℝ, (Real.log (x + 5) + Real.log (2 * x - 2) = Real.log (2 * x^2 + x - 10)) ∧ 
  (x + 5 > 0) ∧ (2 * x - 2 > 0) ∧ (2 * x^2 + x - 10 > 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l1375_137543


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1375_137528

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 1 / a^2 → a^2 > 1 / a) ∧ 
  (∃ a, a^2 > 1 / a ∧ ¬(a > 1 / a^2)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1375_137528


namespace NUMINAMATH_CALUDE_tangent_line_sin_at_pi_l1375_137571

theorem tangent_line_sin_at_pi :
  let f (x : ℝ) := Real.sin x
  let x₀ : ℝ := Real.pi
  let y₀ : ℝ := f x₀
  let m : ℝ := Real.cos x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x + y - Real.pi = 0) := by sorry

end NUMINAMATH_CALUDE_tangent_line_sin_at_pi_l1375_137571
