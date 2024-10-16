import Mathlib

namespace NUMINAMATH_CALUDE_least_positive_integer_exceeding_million_l2065_206507

theorem least_positive_integer_exceeding_million (n : ℕ) : 
  (∀ k < n, (8 : ℝ) ^ ((k * (k + 3)) / 22) ≤ 1000000) ∧
  (8 : ℝ) ^ ((n * (n + 3)) / 22) > 1000000 →
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_exceeding_million_l2065_206507


namespace NUMINAMATH_CALUDE_f_unique_zero_x1_minus_2x2_bound_l2065_206545

noncomputable section

variables (a : ℝ) (h : a ≥ 0)

def f (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

def g (x : ℝ) : ℝ := a * Real.exp x + x

theorem f_unique_zero :
  ∃! x, f a x = 0 :=
sorry

theorem x1_minus_2x2_bound (x₁ x₂ : ℝ) 
  (h₁ : x₁ > -1) (h₂ : x₂ > -1) 
  (h₃ : f a x₁ = g a x₁ - g a x₂) :
  x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_f_unique_zero_x1_minus_2x2_bound_l2065_206545


namespace NUMINAMATH_CALUDE_cell_growth_theorem_l2065_206538

def cell_growth (initial : ℕ) (hours : ℕ) : ℕ :=
  if hours = 0 then
    initial
  else
    2 * (cell_growth initial (hours - 1) - 2)

theorem cell_growth_theorem :
  cell_growth 9 8 = 1284 := by
  sorry

end NUMINAMATH_CALUDE_cell_growth_theorem_l2065_206538


namespace NUMINAMATH_CALUDE_number_of_observations_proof_l2065_206586

theorem number_of_observations_proof (initial_mean : ℝ) (incorrect_obs : ℝ) (correct_obs : ℝ) (new_mean : ℝ) :
  initial_mean = 32 →
  incorrect_obs = 23 →
  correct_obs = 48 →
  new_mean = 32.5 →
  ∃ n : ℕ, n > 0 ∧ n * initial_mean + (correct_obs - incorrect_obs) = n * new_mean ∧ n = 50 :=
by
  sorry

#check number_of_observations_proof

end NUMINAMATH_CALUDE_number_of_observations_proof_l2065_206586


namespace NUMINAMATH_CALUDE_power_four_five_l2065_206573

theorem power_four_five : (4 : ℕ) ^ 4 * (5 : ℕ) ^ 4 = 160000 := by sorry

end NUMINAMATH_CALUDE_power_four_five_l2065_206573


namespace NUMINAMATH_CALUDE_winter_spending_calculation_l2065_206542

/-- The amount spent by the Surf City government at the end of November 1988, in millions of dollars. -/
def spent_end_november : ℝ := 3.3

/-- The amount spent by the Surf City government at the end of February 1989, in millions of dollars. -/
def spent_end_february : ℝ := 7.0

/-- The amount spent during December, January, and February, in millions of dollars. -/
def winter_spending : ℝ := spent_end_february - spent_end_november

theorem winter_spending_calculation : winter_spending = 3.7 := by
  sorry

end NUMINAMATH_CALUDE_winter_spending_calculation_l2065_206542


namespace NUMINAMATH_CALUDE_range_of_x_l2065_206591

theorem range_of_x (x y : ℝ) (h1 : x + y = 1) (h2 : y ≤ 2) : x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2065_206591


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l2065_206599

/-- A line passing through two points intersects the x-axis at a specific point -/
theorem line_intersection_x_axis (x₁ y₁ x₂ y₂ : ℝ) (h : y₁ ≠ y₂) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  let x_intercept := b / m
  (x₁ = 8 ∧ y₁ = 2 ∧ x₂ = 4 ∧ y₂ = 6) →
  x_intercept = 10 ∧ m * x_intercept + b = 0 :=
by
  sorry

#check line_intersection_x_axis

end NUMINAMATH_CALUDE_line_intersection_x_axis_l2065_206599


namespace NUMINAMATH_CALUDE_bakers_pastries_l2065_206537

/-- Baker's pastry problem -/
theorem bakers_pastries 
  (total_cakes : ℕ) 
  (sold_cakes : ℕ) 
  (sold_pastries : ℕ) 
  (remaining_pastries : ℕ) 
  (h1 : total_cakes = 124) 
  (h2 : sold_cakes = 104) 
  (h3 : sold_pastries = 29) 
  (h4 : remaining_pastries = 27) : 
  sold_pastries + remaining_pastries = 56 := by
  sorry

end NUMINAMATH_CALUDE_bakers_pastries_l2065_206537


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_quadratic_equation_specific_roots_l2065_206531

theorem quadratic_equation_roots (m : ℝ) :
  let equation := fun x => m * x^2 - 2 * x + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0) ↔ (m ≤ 1 ∧ m ≠ 0) :=
sorry

theorem quadratic_equation_specific_roots (m : ℝ) :
  let equation := fun x => m * x^2 - 2 * x + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0 ∧ x₁ * x₂ - x₁ - x₂ = 1/2) →
  m = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_quadratic_equation_specific_roots_l2065_206531


namespace NUMINAMATH_CALUDE_condition_relation_l2065_206555

theorem condition_relation (p q : Prop) 
  (h : (p → ¬q) ∧ ¬(¬q → p)) : 
  (q → ¬p) ∧ ¬(¬p → q) := by
  sorry

end NUMINAMATH_CALUDE_condition_relation_l2065_206555


namespace NUMINAMATH_CALUDE_sqrt_72_simplification_l2065_206590

theorem sqrt_72_simplification : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_72_simplification_l2065_206590


namespace NUMINAMATH_CALUDE_urn_problem_l2065_206506

theorem urn_problem (N : ℕ) : 
  (6 : ℚ) / 10 * 20 / (20 + N) + (4 : ℚ) / 10 * N / (20 + N) = 13 / 20 → N = 4 := by
  sorry

end NUMINAMATH_CALUDE_urn_problem_l2065_206506


namespace NUMINAMATH_CALUDE_kyler_won_zero_games_l2065_206554

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Peter : Player
| Emma : Player
| Kyler : Player

/-- Represents the number of games won by each player -/
def games_won (p : Player) : ℕ :=
  match p with
  | Player.Peter => 5
  | Player.Emma => 4
  | Player.Kyler => 0  -- This is what we want to prove

/-- Represents the number of games lost by each player -/
def games_lost (p : Player) : ℕ :=
  match p with
  | Player.Peter => 3
  | Player.Emma => 4
  | Player.Kyler => 4

/-- The total number of games played in the tournament -/
def total_games : ℕ := (games_won Player.Peter + games_won Player.Emma + games_won Player.Kyler +
                        games_lost Player.Peter + games_lost Player.Emma + games_lost Player.Kyler) / 2

theorem kyler_won_zero_games :
  games_won Player.Kyler = 0 ∧
  2 * total_games = games_won Player.Peter + games_won Player.Emma + games_won Player.Kyler +
                    games_lost Player.Peter + games_lost Player.Emma + games_lost Player.Kyler :=
by sorry

end NUMINAMATH_CALUDE_kyler_won_zero_games_l2065_206554


namespace NUMINAMATH_CALUDE_gold_cube_comparison_l2065_206579

/-- Represents the properties of a cube of gold -/
structure GoldCube where
  side_length : ℝ
  weight : ℝ
  value : ℝ

/-- Theorem stating the relationship between two gold cubes of different sizes -/
theorem gold_cube_comparison (small_cube large_cube : GoldCube) :
  small_cube.side_length = 4 →
  small_cube.weight = 5 →
  small_cube.value = 1200 →
  large_cube.side_length = 6 →
  (large_cube.weight = 16.875 ∧ large_cube.value = 4050) :=
by
  sorry

#check gold_cube_comparison

end NUMINAMATH_CALUDE_gold_cube_comparison_l2065_206579


namespace NUMINAMATH_CALUDE_circles_equal_radii_l2065_206564

/-- Proves that the radii of circles A, B, and C are equal -/
theorem circles_equal_radii (r_A : ℝ) (d_B : ℝ) (c_C : ℝ) : 
  r_A = 5 → d_B = 10 → c_C = 10 * Real.pi → r_A = d_B / 2 ∧ r_A = c_C / (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_circles_equal_radii_l2065_206564


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l2065_206589

theorem sqrt_equation_solutions (x : ℝ) :
  x ≥ 2 →
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 10 - 8 * Real.sqrt (x - 2)) = 3) ↔
  (x = 32.25 ∨ x = 8.25) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l2065_206589


namespace NUMINAMATH_CALUDE_rollercoaster_interval_l2065_206540

/-- Given that 7 students ride a rollercoaster every certain minutes,
    and 21 students rode the rollercoaster in 15 minutes,
    prove that the time interval for 7 students to ride the rollercoaster is 5 minutes. -/
theorem rollercoaster_interval (students_per_ride : ℕ) (total_students : ℕ) (total_time : ℕ) :
  students_per_ride = 7 →
  total_students = 21 →
  total_time = 15 →
  (total_time / (total_students / students_per_ride) : ℚ) = 5 :=
by sorry

end NUMINAMATH_CALUDE_rollercoaster_interval_l2065_206540


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l2065_206502

theorem sum_of_three_squares (a k : ℕ) :
  ¬ ∃ x y z : ℤ, (4^a * (8*k + 7) : ℤ) = x^2 + y^2 + z^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l2065_206502


namespace NUMINAMATH_CALUDE_undominated_implies_favorite_toy_l2065_206597

/-- A type representing children -/
def Child : Type := Nat

/-- A type representing toys -/
def Toy : Type := Nat

/-- A type representing a preference ordering of toys for a child -/
def Preference := Toy → Toy → Prop

/-- A type representing a distribution of toys to children -/
def Distribution := Child → Toy

/-- Predicate indicating if a toy is preferred over another for a given child's preference -/
def IsPreferred (pref : Preference) (t1 t2 : Toy) : Prop := pref t1 t2 ∧ ¬pref t2 t1

/-- Predicate indicating if a distribution is dominated by another -/
def Dominates (prefs : Child → Preference) (d1 d2 : Distribution) : Prop :=
  ∀ c : Child, IsPreferred (prefs c) (d1 c) (d2 c) ∨ d1 c = d2 c

/-- Predicate indicating if a toy is the favorite for a child -/
def IsFavorite (pref : Preference) (t : Toy) : Prop :=
  ∀ t' : Toy, t ≠ t' → IsPreferred pref t t'

theorem undominated_implies_favorite_toy
  (n : Nat)
  (prefs : Child → Preference)
  (d : Distribution)
  (h_strict : ∀ c : Child, ∀ t1 t2 : Toy, t1 ≠ t2 → (IsPreferred (prefs c) t1 t2 ∨ IsPreferred (prefs c) t2 t1))
  (h_undominated : ∀ d' : Distribution, ¬Dominates prefs d' d) :
  ∃ c : Child, IsFavorite (prefs c) (d c) :=
sorry

end NUMINAMATH_CALUDE_undominated_implies_favorite_toy_l2065_206597


namespace NUMINAMATH_CALUDE_incircle_tangent_bisects_altitude_median_l2065_206594

/-- Triangle with incircle -/
structure TriangleWithIncircle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Positivity of sides
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  -- Triangle inequality
  hab : a + b > c
  hbc : b + c > a
  hca : c + a > b
  -- Existence of incircle (implied by above conditions)

/-- Point on a line segment -/
def PointOnSegment (A B T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ T = (1 - t) • A + t • B

/-- Midpoint of a line segment -/
def Midpoint (A B M : ℝ × ℝ) : Prop :=
  M = (A + B) / 2

/-- Foot of altitude from a point to a line -/
def AltitudeFoot (C H : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  H ∈ l ∧ (∀ P ∈ l, ‖C - H‖ ≤ ‖C - P‖)

/-- Tangent point of incircle -/
def TangentPoint (T : ℝ × ℝ) (circle : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Prop :=
  T ∈ circle ∧ T ∈ l ∧ (∀ P ∈ circle ∩ l, P = T)

theorem incircle_tangent_bisects_altitude_median 
  (triangle : TriangleWithIncircle) 
  (A B C T H M : ℝ × ℝ) 
  (l : Set (ℝ × ℝ)) 
  (circle : Set (ℝ × ℝ)) :
  (PointOnSegment A B T ∧ 
   Midpoint A B M ∧ 
   AltitudeFoot C H l ∧
   TangentPoint T circle l) →
  (T = (H + M) / 2 ↔ triangle.c = (triangle.a + triangle.b) / 2) :=
sorry

end NUMINAMATH_CALUDE_incircle_tangent_bisects_altitude_median_l2065_206594


namespace NUMINAMATH_CALUDE_min_abs_w_l2065_206583

theorem min_abs_w (w : ℂ) (h : Complex.abs (w + 2 - 2*I) + Complex.abs (w - 5*I) = 7) :
  Complex.abs w ≥ 10/7 ∧ ∃ w₀ : ℂ, Complex.abs (w₀ + 2 - 2*I) + Complex.abs (w₀ - 5*I) = 7 ∧ Complex.abs w₀ = 10/7 :=
sorry

end NUMINAMATH_CALUDE_min_abs_w_l2065_206583


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2065_206547

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let e := c / a
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → 
    ((x + c)^2 + y^2 = (a + e * x)^2) ∧
    (0 - b)^2 / ((-c) - 0)^2 + (b - 0)^2 / (0 - a)^2 = 1) →
  e = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2065_206547


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2065_206550

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 36*x^2 + 215*x - 470

-- State the theorem
theorem root_sum_reciprocal (a b c : ℝ) (D E F : ℝ) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  p a = 0 ∧ p b = 0 ∧ p c = 0 →
  (∀ t : ℝ, t ≠ a ∧ t ≠ b ∧ t ≠ c →
    1 / (t^3 - 36*t^2 + 215*t - 470) = D / (t - a) + E / (t - b) + F / (t - c)) →
  1 / D + 1 / E + 1 / F = 105 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2065_206550


namespace NUMINAMATH_CALUDE_max_dishes_l2065_206576

theorem max_dishes (main_ingredients : Nat) (secondary_ingredients : Nat) (cooking_methods : Nat)
  (select_main : Nat) (select_secondary : Nat) :
  main_ingredients = 5 →
  secondary_ingredients = 8 →
  cooking_methods = 5 →
  select_main = 2 →
  select_secondary = 3 →
  (Nat.choose main_ingredients select_main) *
  (Nat.choose secondary_ingredients select_secondary) *
  cooking_methods = 2800 :=
by sorry

end NUMINAMATH_CALUDE_max_dishes_l2065_206576


namespace NUMINAMATH_CALUDE_minibus_children_count_l2065_206596

theorem minibus_children_count (total_seats : ℕ) (full_seats : ℕ) (children_per_full_seat : ℕ) (children_per_remaining_seat : ℕ) :
  total_seats = 7 →
  full_seats = 5 →
  children_per_full_seat = 3 →
  children_per_remaining_seat = 2 →
  full_seats * children_per_full_seat + (total_seats - full_seats) * children_per_remaining_seat = 19 :=
by sorry

end NUMINAMATH_CALUDE_minibus_children_count_l2065_206596


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l2065_206521

def f (x : ℝ) : ℝ := x^3 + x^2 + 2*x + 4

def g (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_roots_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧
    g (r₁^3) b c d = 0 ∧ g (r₂^3) b c d = 0 ∧ g (r₃^3) b c d = 0) →
  b = 24 ∧ c = 32 ∧ d = 64 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l2065_206521


namespace NUMINAMATH_CALUDE_fraction_equality_l2065_206569

theorem fraction_equality : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2065_206569


namespace NUMINAMATH_CALUDE_age_difference_l2065_206556

-- Define the ages of A and B
def A : ℕ := sorry
def B : ℕ := 95

-- State the theorem
theorem age_difference : A - B = 5 := by
  -- The condition that in 30 years, A will be twice as old as B was 30 years ago
  have h : A + 30 = 2 * (B - 30) := by sorry
  sorry

end NUMINAMATH_CALUDE_age_difference_l2065_206556


namespace NUMINAMATH_CALUDE_division_problem_l2065_206536

theorem division_problem : (-1/24) / (1/3 - 1/6 + 3/8) = -1/13 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2065_206536


namespace NUMINAMATH_CALUDE_mia_darwin_money_multiple_l2065_206516

theorem mia_darwin_money_multiple (darwin_money mia_money : ℕ) (multiple : ℚ) : 
  darwin_money = 45 →
  mia_money = 110 →
  mia_money = multiple * darwin_money + 20 →
  multiple = 2 := by sorry

end NUMINAMATH_CALUDE_mia_darwin_money_multiple_l2065_206516


namespace NUMINAMATH_CALUDE_range_of_sin_minus_cos_l2065_206525

open Real

theorem range_of_sin_minus_cos (x : ℝ) : 
  -Real.sqrt 3 ≤ sin (x + 18 * π / 180) - cos (x + 48 * π / 180) ∧
  sin (x + 18 * π / 180) - cos (x + 48 * π / 180) ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sin_minus_cos_l2065_206525


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2065_206568

/-- Given two vectors a and b in ℝ², where a = (3, -2) and b = (x, 1),
    prove that if a ⊥ b, then x = 2/3 -/
theorem perpendicular_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![3, -2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∀ i j, i ≠ j → a i * a j + b i * b j = 0) →
  x = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2065_206568


namespace NUMINAMATH_CALUDE_constant_function_invariant_l2065_206585

theorem constant_function_invariant (g : ℝ → ℝ) (h : ∀ x : ℝ, g x = -3) :
  ∀ x : ℝ, g (3 * x - 5) = -3 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_invariant_l2065_206585


namespace NUMINAMATH_CALUDE_pill_supply_lasts_eight_months_l2065_206566

/-- Calculates the duration in months that a pill supply will last -/
def pill_supply_duration (total_pills : ℕ) (days_per_pill : ℕ) (days_per_month : ℕ) : ℕ :=
  (total_pills * days_per_pill) / days_per_month

/-- Proves that a supply of 120 pills, taken every two days, lasts 8 months -/
theorem pill_supply_lasts_eight_months :
  pill_supply_duration 120 2 30 = 8 := by
  sorry

#eval pill_supply_duration 120 2 30

end NUMINAMATH_CALUDE_pill_supply_lasts_eight_months_l2065_206566


namespace NUMINAMATH_CALUDE_prime_divisibility_l2065_206515

theorem prime_divisibility (p a b : ℤ) (hp : Prime p) (hp_not_3 : p ≠ 3)
  (h_sum : p ∣ (a + b)) (h_cube_sum : p^2 ∣ (a^3 + b^3)) :
  p^2 ∣ (a + b) ∨ p^3 ∣ (a^3 + b^3) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l2065_206515


namespace NUMINAMATH_CALUDE_iains_old_pennies_l2065_206552

/-- The number of pennies older than Iain -/
def oldPennies : ℕ := 30

/-- The initial number of pennies Iain has -/
def initialPennies : ℕ := 200

/-- The number of pennies Iain has left after removing old pennies and 20% of the remaining -/
def remainingPennies : ℕ := 136

/-- The percentage of remaining pennies Iain throws out -/
def throwOutPercentage : ℚ := 1/5

theorem iains_old_pennies :
  oldPennies = initialPennies - (remainingPennies / (1 - throwOutPercentage)) := by
  sorry

end NUMINAMATH_CALUDE_iains_old_pennies_l2065_206552


namespace NUMINAMATH_CALUDE_forty_percent_of_two_l2065_206512

theorem forty_percent_of_two : (40 / 100) * 2 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_two_l2065_206512


namespace NUMINAMATH_CALUDE_triangle_equality_l2065_206522

theorem triangle_equality (a b c : ℝ) 
  (h1 : |a| ≥ |b + c|) 
  (h2 : |b| ≥ |c + a|) 
  (h3 : |c| ≥ |a + b|) : 
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_l2065_206522


namespace NUMINAMATH_CALUDE_element_in_two_sets_l2065_206505

variable {n : ℕ}
variable (A : Fin n → Finset (Fin n))
variable (a : Fin n → Fin n)

-- A₁, A₂, ..., Aₙ are all two-element subsets of the set {a₁, a₂, ..., aₙ}
axiom two_element_subsets : ∀ i : Fin n, (A i).card = 2 ∧ (A i) ⊆ (Finset.univ : Finset (Fin n))

-- For any i, j = 1, 2, 3, ..., n and i ≠ j, if Aᵢ ∩ Aⱼ ≠ ∅, then one of Aᵢ or Aⱼ must be {aᵢ, aⱼ}
axiom intersection_condition : ∀ i j : Fin n, i ≠ j → (A i ∩ A j).Nonempty → 
  (A i = {a i, a j} ∨ A j = {a i, a j})

-- Theorem: Each element in the set {a₁, a₂, ..., aₙ} belongs to exactly two Aᵢ
theorem element_in_two_sets : ∀ k : Fin n, (Finset.filter (fun i => k ∈ A i) Finset.univ).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_element_in_two_sets_l2065_206505


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2065_206548

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_problem : Nat.gcd (factorial 7) ((factorial 12) / (factorial 5)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2065_206548


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2065_206520

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number we want to express in scientific notation -/
def original_number : ℕ := 12060000

/-- The scientific notation representation of the original number -/
def scientific_representation : ScientificNotation := {
  coefficient := 1.206
  exponent := 7
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (original_number : ℝ) = scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2065_206520


namespace NUMINAMATH_CALUDE_time_sum_after_duration_l2065_206577

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time and returns the result on a 12-hour clock -/
def addDuration (startTime : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

theorem time_sum_after_duration (startTime : Time) (durationHours durationMinutes durationSeconds : Nat) :
  let endTime := addDuration startTime durationHours durationMinutes durationSeconds
  startTime.hours = 3 ∧ startTime.minutes = 0 ∧ startTime.seconds = 0 ∧
  durationHours = 315 ∧ durationMinutes = 58 ∧ durationSeconds = 16 →
  endTime.hours + endTime.minutes + endTime.seconds = 77 := by
  sorry

end NUMINAMATH_CALUDE_time_sum_after_duration_l2065_206577


namespace NUMINAMATH_CALUDE_stock_investment_percentage_l2065_206571

theorem stock_investment_percentage (investment : ℝ) (earnings : ℝ) (percentage : ℝ) :
  investment = 5760 →
  earnings = 1900 →
  percentage = (earnings * 100) / investment →
  percentage = 33 :=
by sorry

end NUMINAMATH_CALUDE_stock_investment_percentage_l2065_206571


namespace NUMINAMATH_CALUDE_chocolate_box_bars_l2065_206504

def chocolate_problem (total_bars : ℕ) : Prop :=
  let bar_price : ℚ := 3
  let remaining_bars : ℕ := 4
  let sales : ℚ := 9
  bar_price * (total_bars - remaining_bars) = sales

theorem chocolate_box_bars : ∃ (x : ℕ), chocolate_problem x ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_bars_l2065_206504


namespace NUMINAMATH_CALUDE_sphere_radius_from_hole_l2065_206551

theorem sphere_radius_from_hole (hole_diameter : ℝ) (hole_depth : ℝ) (sphere_radius : ℝ) :
  hole_diameter = 30 →
  hole_depth = 12 →
  sphere_radius = (27 / 8 + 12) →
  sphere_radius = 15.375 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_from_hole_l2065_206551


namespace NUMINAMATH_CALUDE_average_books_theorem_l2065_206567

/-- Represents the number of books borrowed by a student -/
structure BooksBorrowed where
  count : ℕ
  is_valid : count ≤ 6

/-- Represents the distribution of books borrowed in the class -/
structure ClassDistribution where
  total_students : ℕ
  zero_books : ℕ
  one_book : ℕ
  two_books : ℕ
  at_least_three : ℕ
  is_valid : total_students = zero_books + one_book + two_books + at_least_three

def average_books (dist : ClassDistribution) : ℚ :=
  let total_books := dist.one_book + 2 * dist.two_books + 3 * dist.at_least_three
  total_books / dist.total_students

theorem average_books_theorem (dist : ClassDistribution) 
  (h1 : dist.total_students = 40)
  (h2 : dist.zero_books = 2)
  (h3 : dist.one_book = 12)
  (h4 : dist.two_books = 13)
  (h5 : dist.at_least_three = dist.total_students - (dist.zero_books + dist.one_book + dist.two_books)) :
  average_books dist = 77 / 40 := by
  sorry

#eval (77 : ℚ) / 40

end NUMINAMATH_CALUDE_average_books_theorem_l2065_206567


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2065_206517

/-- The parabola function y = x^2 - 6x + 8 -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- A square inscribed in the region bounded by the parabola and x-axis -/
structure InscribedSquare where
  center : ℝ
  sideLength : ℝ
  top_touches_parabola : parabola (center + sideLength/2) = sideLength
  bottom_on_x_axis : center - sideLength/2 ≥ 0

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area :
  ∀ (s : InscribedSquare), s.sideLength^2 = 12 - 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2065_206517


namespace NUMINAMATH_CALUDE_no_negative_roots_and_positive_root_exists_l2065_206546

def f (x : ℝ) : ℝ := x^6 - 3*x^5 - 6*x^3 - x + 8

theorem no_negative_roots_and_positive_root_exists :
  (∀ x < 0, f x ≠ 0) ∧ (∃ x > 0, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_and_positive_root_exists_l2065_206546


namespace NUMINAMATH_CALUDE_total_cleaning_time_is_114_l2065_206562

/-- Calculates the total time spent by Andy and Dawn on house cleaning tasks -/
def total_cleaning_time (dawn_dish_time : ℕ) : ℕ :=
  let andy_laundry_time := 2 * dawn_dish_time + 6
  let andy_vacuum_time := andy_laundry_time - 8
  let dawn_window_time := dawn_dish_time / 2
  let andy_total := andy_laundry_time + andy_vacuum_time
  let dawn_total := dawn_dish_time + dawn_window_time
  andy_total + dawn_total

/-- Theorem stating that the total cleaning time is 114 minutes -/
theorem total_cleaning_time_is_114 : total_cleaning_time 20 = 114 := by
  sorry


end NUMINAMATH_CALUDE_total_cleaning_time_is_114_l2065_206562


namespace NUMINAMATH_CALUDE_quadratic_linear_third_quadrant_l2065_206539

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a linear function y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Checks if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Checks if a quadratic equation has no real roots -/
def hasNoRealRoots (eq : QuadraticEquation) : Prop :=
  eq.b^2 - 4*eq.a*eq.c < 0

/-- Checks if a linear function passes through a point -/
def passesThrough (f : LinearFunction) (p : Point) : Prop :=
  p.y = f.m * p.x + f.b

theorem quadratic_linear_third_quadrant 
  (b : ℝ) 
  (quad : QuadraticEquation) 
  (lin : LinearFunction) :
  quad = QuadraticEquation.mk 1 2 (b - 3) →
  hasNoRealRoots quad →
  lin = LinearFunction.mk (-2) b →
  ¬ ∃ (p : Point), isInThirdQuadrant p ∧ passesThrough lin p :=
sorry

end NUMINAMATH_CALUDE_quadratic_linear_third_quadrant_l2065_206539


namespace NUMINAMATH_CALUDE_soccer_tournament_games_l2065_206561

def soccer_tournament (n : ℕ) (m : ℕ) (tie_breaker : ℕ) : ℕ :=
  let first_stage := n * (n - 1) / 2
  let second_stage := 2 * (m * (m - 1) / 2)
  first_stage + second_stage + tie_breaker

theorem soccer_tournament_games :
  soccer_tournament 25 10 1 = 391 := by
  sorry

end NUMINAMATH_CALUDE_soccer_tournament_games_l2065_206561


namespace NUMINAMATH_CALUDE_group_formation_count_l2065_206527

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of officers --/
def totalOfficers : ℕ := 10

/-- The total number of jawans --/
def totalJawans : ℕ := 15

/-- The number of officers in each group --/
def officersPerGroup : ℕ := 3

/-- The number of jawans in each group --/
def jawansPerGroup : ℕ := 5

/-- The number of ways to form groups --/
def numberOfGroups : ℕ := 
  totalOfficers * 
  (choose (totalOfficers - 1) (officersPerGroup - 1)) * 
  (choose totalJawans jawansPerGroup)

theorem group_formation_count : numberOfGroups = 1081080 := by
  sorry

end NUMINAMATH_CALUDE_group_formation_count_l2065_206527


namespace NUMINAMATH_CALUDE_ordered_pairs_count_l2065_206524

theorem ordered_pairs_count : 
  { p : ℤ × ℤ | (p.1 : ℤ) ^ 2019 + (p.2 : ℤ) ^ 2 = 2 * (p.2 : ℤ) }.Finite ∧ 
  { p : ℤ × ℤ | (p.1 : ℤ) ^ 2019 + (p.2 : ℤ) ^ 2 = 2 * (p.2 : ℤ) }.ncard = 3 :=
by sorry

end NUMINAMATH_CALUDE_ordered_pairs_count_l2065_206524


namespace NUMINAMATH_CALUDE_area_XYZA_is_four_thirds_l2065_206560

/-- Right trapezoid PQRS with the given properties -/
structure RightTrapezoid where
  PQ : ℝ
  RS : ℝ
  PR : ℝ
  trisectPQ : ℝ → ℝ → ℝ  -- Function to represent trisection points on PQ
  trisectRS : ℝ → ℝ → ℝ  -- Function to represent trisection points on RS
  midpoint : ℝ → ℝ → ℝ   -- Function to calculate midpoint

/-- The area of quadrilateral XYZA in the right trapezoid -/
def areaXYZA (t : RightTrapezoid) : ℝ :=
  let X := t.midpoint 0 (t.trisectPQ 0 1)
  let Y := t.midpoint (t.trisectPQ 0 1) (t.trisectPQ 1 2)
  let Z := t.midpoint (t.trisectRS 1 2) (t.trisectRS 0 1)
  let A := t.midpoint (t.trisectRS 0 1) t.RS
  -- Area calculation would go here
  sorry

/-- Theorem stating that the area of XYZA is 4/3 -/
theorem area_XYZA_is_four_thirds (t : RightTrapezoid) 
    (h1 : t.PQ = 2) 
    (h2 : t.RS = 6) 
    (h3 : t.PR = 4) : 
  areaXYZA t = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_area_XYZA_is_four_thirds_l2065_206560


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2065_206544

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  x^2 + 2*x*y ≤ 25 ∧ ∃ x y : ℝ, x + y = 5 ∧ x^2 + 2*x*y = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2065_206544


namespace NUMINAMATH_CALUDE_negation_of_implication_l2065_206509

theorem negation_of_implication (m : ℝ) : 
  (¬(m > 0 → ∃ x : ℝ, x^2 + x - m = 0)) ↔ 
  (m ≤ 0 → ¬∃ x : ℝ, x^2 + x - m = 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2065_206509


namespace NUMINAMATH_CALUDE_inequality_proof_l2065_206529

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 / (a^2 + a*b + b^2) + b^3 / (b^2 + b*c + c^2) + c^3 / (c^2 + c*a + a^2) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2065_206529


namespace NUMINAMATH_CALUDE_remainder_problem_l2065_206513

theorem remainder_problem (k : ℕ) 
  (h1 : k % 6 = 5) 
  (h2 : k % 7 = 3) 
  (h3 : k < 41) : 
  k % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2065_206513


namespace NUMINAMATH_CALUDE_workshop_workers_l2065_206578

/-- The number of technicians in the workshop -/
def num_technicians : ℕ := 7

/-- The average salary of all workers in the workshop -/
def avg_salary_all : ℕ := 8000

/-- The average salary of technicians in the workshop -/
def avg_salary_tech : ℕ := 12000

/-- The average salary of non-technicians in the workshop -/
def avg_salary_nontech : ℕ := 6000

/-- The total number of workers in the workshop -/
def total_workers : ℕ := 21

theorem workshop_workers :
  (total_workers * avg_salary_all = 
   num_technicians * avg_salary_tech + 
   (total_workers - num_technicians) * avg_salary_nontech) ∧
  (total_workers = 21) := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l2065_206578


namespace NUMINAMATH_CALUDE_rooster_ratio_l2065_206588

theorem rooster_ratio (total : ℕ) (roosters : ℕ) (hens : ℕ) :
  total = 80 →
  total = roosters + hens →
  roosters + (1/4 : ℚ) * hens = 35 →
  (roosters : ℚ) / total = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_rooster_ratio_l2065_206588


namespace NUMINAMATH_CALUDE_tires_cost_calculation_l2065_206563

def speakers_cost : ℚ := 136.01
def cd_player_cost : ℚ := 139.38
def total_spent : ℚ := 387.85

theorem tires_cost_calculation :
  total_spent - (speakers_cost + cd_player_cost) = 112.46 :=
by sorry

end NUMINAMATH_CALUDE_tires_cost_calculation_l2065_206563


namespace NUMINAMATH_CALUDE_umbrella_arrangement_count_l2065_206501

/-- The number of ways to arrange 7 people in an umbrella shape -/
def umbrella_arrangements : ℕ := sorry

/-- The binomial coefficient (n choose k) -/
def choose (n k : ℕ) : ℕ := sorry

theorem umbrella_arrangement_count : umbrella_arrangements = 20 := by
  sorry

end NUMINAMATH_CALUDE_umbrella_arrangement_count_l2065_206501


namespace NUMINAMATH_CALUDE_f_composition_result_l2065_206580

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1/2 else 2^x

theorem f_composition_result : f (f (5/6)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_result_l2065_206580


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2065_206533

def p (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2065_206533


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2065_206557

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, |x^2 - a*x + 4*a| ≤ 3) ↔ (a = 8 + 2*Real.sqrt 13 ∨ a = 8 - 2*Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2065_206557


namespace NUMINAMATH_CALUDE_min_same_score_competition_l2065_206549

/-- Represents a math competition with fill-in-the-blank and short-answer questions. -/
structure MathCompetition where
  fill_in_blank_count : Nat
  fill_in_blank_points : Nat
  short_answer_count : Nat
  short_answer_points : Nat
  participant_count : Nat

/-- Calculates the minimum number of participants with the same score. -/
def min_same_score (comp : MathCompetition) : Nat :=
  let max_score := comp.fill_in_blank_count * comp.fill_in_blank_points +
                   comp.short_answer_count * comp.short_answer_points
  let distinct_scores := (comp.fill_in_blank_count + 1) * (comp.short_answer_count + 1)
  (comp.participant_count + distinct_scores - 1) / distinct_scores

/-- Theorem stating the minimum number of participants with the same score
    in the given competition configuration. -/
theorem min_same_score_competition :
  let comp := MathCompetition.mk 8 4 6 7 400
  min_same_score comp = 8 := by
  sorry


end NUMINAMATH_CALUDE_min_same_score_competition_l2065_206549


namespace NUMINAMATH_CALUDE_cindy_pens_l2065_206535

theorem cindy_pens (initial_pens : ℕ) (mike_gives : ℕ) (sharon_receives : ℕ) (final_pens : ℕ)
  (h1 : initial_pens = 5)
  (h2 : mike_gives = 20)
  (h3 : sharon_receives = 19)
  (h4 : final_pens = 31) :
  final_pens = initial_pens + mike_gives - sharon_receives + 25 :=
by sorry

end NUMINAMATH_CALUDE_cindy_pens_l2065_206535


namespace NUMINAMATH_CALUDE_rectangle_configuration_CG_length_l2065_206584

/-- A configuration of two rectangles ABCD and EFGH with parallel sides -/
structure RectangleConfiguration where
  /-- The length of segment AE -/
  AE : ℝ
  /-- The length of segment BF -/
  BF : ℝ
  /-- The length of segment DH -/
  DH : ℝ
  /-- The length of segment CG -/
  CG : ℝ

/-- The theorem stating the length of CG given the other lengths -/
theorem rectangle_configuration_CG_length 
  (config : RectangleConfiguration) 
  (h1 : config.AE = 10) 
  (h2 : config.BF = 20) 
  (h3 : config.DH = 30) : 
  config.CG = 20 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_configuration_CG_length_l2065_206584


namespace NUMINAMATH_CALUDE_grains_per_teaspoon_l2065_206570

/-- Represents the number of grains of rice in a cup -/
def grains_per_cup : ℕ := 480

/-- Represents the number of tablespoons in half a cup -/
def tablespoons_per_half_cup : ℕ := 8

/-- Represents the number of teaspoons in a tablespoon -/
def teaspoons_per_tablespoon : ℕ := 3

/-- Theorem stating that there are 10 grains of rice in a teaspoon -/
theorem grains_per_teaspoon : 
  (grains_per_cup : ℚ) / ((2 * tablespoons_per_half_cup) * teaspoons_per_tablespoon) = 10 := by
  sorry

end NUMINAMATH_CALUDE_grains_per_teaspoon_l2065_206570


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l2065_206582

theorem smallest_gcd_multiple (a b : ℕ+) (h : Nat.gcd a b = 10) :
  (∃ (x y : ℕ+), Nat.gcd x y = 10 ∧ Nat.gcd (8 * x) (14 * y) = 20) ∧
  ∀ (c d : ℕ+), Nat.gcd c d = 10 → Nat.gcd (8 * c) (14 * d) ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l2065_206582


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2065_206595

theorem arithmetic_calculation : 4 * (8 - 3) / 2 - 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2065_206595


namespace NUMINAMATH_CALUDE_eight_digit_integers_count_l2065_206553

theorem eight_digit_integers_count : 
  (Finset.range 9).card * (Finset.range 10).card ^ 7 = 90000000 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_integers_count_l2065_206553


namespace NUMINAMATH_CALUDE_equation_solution_l2065_206526

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => 1/((x - 3)*(x - 4)) + 1/((x - 4)*(x - 5)) + 1/((x - 5)*(x - 6))
  ∀ x : ℝ, f x = 1/8 ↔ x = (9 + Real.sqrt 57)/2 ∨ x = (9 - Real.sqrt 57)/2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2065_206526


namespace NUMINAMATH_CALUDE_maddy_classes_per_semester_l2065_206528

/-- The number of classes Maddy needs to take per semester to graduate -/
def classes_per_semester (total_semesters : ℕ) (total_credits : ℕ) (credits_per_class : ℕ) : ℕ :=
  (total_credits / credits_per_class) / total_semesters

/-- Proof that Maddy needs to take 5 classes per semester -/
theorem maddy_classes_per_semester :
  classes_per_semester 8 120 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_maddy_classes_per_semester_l2065_206528


namespace NUMINAMATH_CALUDE_equality_sum_l2065_206572

theorem equality_sum (M N : ℚ) : 
  (3 : ℚ) / 5 = M / 30 ∧ (3 : ℚ) / 5 = 90 / N → M + N = 168 := by
  sorry

end NUMINAMATH_CALUDE_equality_sum_l2065_206572


namespace NUMINAMATH_CALUDE_field_trip_difference_proof_l2065_206593

/-- Calculates the difference in number of people traveling by bus versus van on a field trip. -/
def field_trip_difference (num_vans : Real) (num_buses : Real) (people_per_van : Real) (people_per_bus : Real) : Real :=
  num_buses * people_per_bus - num_vans * people_per_van

/-- Proves that the difference in number of people traveling by bus versus van is 108.0 for the given conditions. -/
theorem field_trip_difference_proof :
  field_trip_difference 6.0 8.0 6.0 18.0 = 108.0 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_difference_proof_l2065_206593


namespace NUMINAMATH_CALUDE_exactly_one_true_l2065_206587

-- Define the polynomials
def A (x : ℝ) : ℝ := 2 * x^2
def B (x : ℝ) : ℝ := x + 1
def C (x : ℝ) : ℝ := -2 * x
def D (y : ℝ) : ℝ := y^2
def E (x y : ℝ) : ℝ := 2 * x - y

-- Define the three statements
def statement1 : Prop :=
  ∀ y : ℕ+, ∀ x : ℝ, B x * C x + A x + D y + E x y > 0

def statement2 : Prop :=
  ∃ x y : ℝ, A x + D y + 2 * E x y = -2

def statement3 : Prop :=
  ∀ x : ℝ, ∀ m : ℝ,
    (∃ k : ℝ, 3 * (A x - B x) + m * B x * C x = k * x^2 + (3 : ℝ)) →
    3 * (A x - B x) + m * B x * C x > -3

theorem exactly_one_true : (statement1 ∧ ¬statement2 ∧ ¬statement3) ∨
                           (¬statement1 ∧ statement2 ∧ ¬statement3) ∨
                           (¬statement1 ∧ ¬statement2 ∧ statement3) :=
  sorry

end NUMINAMATH_CALUDE_exactly_one_true_l2065_206587


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l2065_206541

/-- A quadratic equation is of the form ax² + bx + c = 0, where a ≠ 0 --/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² + 2x = 0 is a quadratic equation --/
theorem equation_is_quadratic :
  is_quadratic_equation (λ x => x^2 + 2*x) :=
sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l2065_206541


namespace NUMINAMATH_CALUDE_speed_equivalence_l2065_206523

/-- Proves that a speed of 12/36 m/s is equivalent to 1.2 km/h -/
theorem speed_equivalence : ∀ (x : ℚ), x = 12 / 36 → x * (3600 / 1000) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_speed_equivalence_l2065_206523


namespace NUMINAMATH_CALUDE_area_of_PQRS_l2065_206503

/-- Reflect a point (x, y) in the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflect a point (x, y) in the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- Reflect a point (x, y) in the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Calculate the area of a quadrilateral given its four vertices -/
def quadrilateral_area (a b c d : ℝ × ℝ) : ℝ := sorry

theorem area_of_PQRS : 
  let P : ℝ × ℝ := (-1, 4)
  let Q := reflect_y P
  let R := reflect_y_eq_x Q
  let S := reflect_x R
  quadrilateral_area P Q R S = 8 := by sorry

end NUMINAMATH_CALUDE_area_of_PQRS_l2065_206503


namespace NUMINAMATH_CALUDE_root_sum_ratio_l2065_206518

theorem root_sum_ratio (k : ℝ) (a b : ℝ) : 
  (k * (a^2 - 2*a) + 3*a + 7 = 0) →
  (k * (b^2 - 2*b) + 3*b + 7 = 0) →
  (a / b + b / a = 3 / 4) →
  ∃ (k₁ k₂ : ℝ), k₁ / k₂ + k₂ / k₁ = 433.42 := by sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l2065_206518


namespace NUMINAMATH_CALUDE_cow_count_is_ten_l2065_206558

/-- Represents the number of animals in a group -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (group : AnimalGroup) : ℕ :=
  2 * group.ducks + 4 * group.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (group : AnimalGroup) : ℕ :=
  group.ducks + group.cows

/-- Theorem stating that the number of cows is 10 -/
theorem cow_count_is_ten (group : AnimalGroup) 
    (h : totalLegs group = 2 * totalHeads group + 20) : 
    group.cows = 10 := by
  sorry

#check cow_count_is_ten

end NUMINAMATH_CALUDE_cow_count_is_ten_l2065_206558


namespace NUMINAMATH_CALUDE_dice_diff_three_prob_l2065_206581

-- Define a die as having 6 sides
def die := Finset.range 6

-- Define the favorable outcomes
def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 4), (4, 1), (2, 5), (5, 2), (3, 6), (6, 3)}

-- Define the total number of outcomes
def total_outcomes : ℕ := die.card * die.card

-- Define the number of favorable outcomes
def num_favorable_outcomes : ℕ := favorable_outcomes.card

-- Theorem statement
theorem dice_diff_three_prob :
  (num_favorable_outcomes : ℚ) / total_outcomes = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_dice_diff_three_prob_l2065_206581


namespace NUMINAMATH_CALUDE_percentage_b_grades_l2065_206530

def scores : List Nat := [92, 81, 68, 88, 82, 63, 79, 70, 85, 99, 59, 67, 84, 90, 75, 61, 87, 65, 86]

def is_b_grade (score : Nat) : Bool :=
  80 ≤ score ∧ score ≤ 84

def count_b_grades (scores : List Nat) : Nat :=
  scores.filter is_b_grade |>.length

theorem percentage_b_grades : 
  (count_b_grades scores : Rat) / (scores.length : Rat) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_b_grades_l2065_206530


namespace NUMINAMATH_CALUDE_max_quotient_value_l2065_206500

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) :
  (∀ x y, 100 ≤ x ∧ x ≤ 300 → 500 ≤ y ∧ y ≤ 1500 → y^2 / x^2 ≤ 225) ∧
  (∃ x y, 100 ≤ x ∧ x ≤ 300 ∧ 500 ≤ y ∧ y ≤ 1500 ∧ y^2 / x^2 = 225) :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l2065_206500


namespace NUMINAMATH_CALUDE_temp_difference_changtai_beijing_l2065_206519

/-- The difference in temperature between two locations -/
def temperature_difference (temp1 : Int) (temp2 : Int) : Int :=
  temp1 - temp2

/-- The lowest temperature in Beijing (in °C) -/
def beijing_temp : Int := -6

/-- The lowest temperature in Changtai County (in °C) -/
def changtai_temp : Int := 15

/-- Theorem: The temperature difference between Changtai County and Beijing is 21°C -/
theorem temp_difference_changtai_beijing :
  temperature_difference changtai_temp beijing_temp = 21 := by
  sorry

end NUMINAMATH_CALUDE_temp_difference_changtai_beijing_l2065_206519


namespace NUMINAMATH_CALUDE_min_colors_tessellation_l2065_206574

/-- Represents a tile in the tessellation -/
inductive Tile
| Triangle
| Trapezoid

/-- Represents a color used in the tessellation -/
inductive Color
| Red
| Green
| Blue

/-- Represents the tessellation as a function from coordinates to tiles -/
def Tessellation := ℕ → ℕ → Tile

/-- A valid tessellation alternates between rows of triangles and trapezoids -/
def isValidTessellation (t : Tessellation) : Prop :=
  ∀ i j, t i j = if i % 2 = 0 then Tile.Triangle else Tile.Trapezoid

/-- A coloring of the tessellation -/
def Coloring := ℕ → ℕ → Color

/-- Checks if two tiles are adjacent -/
def isAdjacent (i1 j1 i2 j2 : ℕ) : Prop :=
  (i1 = i2 ∧ j1 + 1 = j2) ∨ 
  (i1 + 1 = i2 ∧ j1 = j2) ∨ 
  (i1 + 1 = i2 ∧ j1 + 1 = j2)

/-- A valid coloring ensures no adjacent tiles have the same color -/
def isValidColoring (t : Tessellation) (c : Coloring) : Prop :=
  ∀ i1 j1 i2 j2, isAdjacent i1 j1 i2 j2 → c i1 j1 ≠ c i2 j2

/-- The main theorem: 3 colors are sufficient and necessary -/
theorem min_colors_tessellation (t : Tessellation) (h : isValidTessellation t) :
  (∃ c : Coloring, isValidColoring t c) ∧ 
  (∀ c : Coloring, isValidColoring t c → ∃ (x y z : Color), 
    (∀ i j, c i j = x ∨ c i j = y ∨ c i j = z)) :=
sorry

end NUMINAMATH_CALUDE_min_colors_tessellation_l2065_206574


namespace NUMINAMATH_CALUDE_cicely_hundredth_birthday_l2065_206508

def cicely_birthday_problem (birth_year : ℕ) (twenty_first_year : ℕ) (hundredth_year : ℕ) : Prop :=
  (twenty_first_year - birth_year = 21) ∧ 
  (twenty_first_year = 1939) ∧
  (hundredth_year - birth_year = 100)

theorem cicely_hundredth_birthday : 
  ∃ (birth_year : ℕ), cicely_birthday_problem birth_year 1939 2018 := by
  sorry

end NUMINAMATH_CALUDE_cicely_hundredth_birthday_l2065_206508


namespace NUMINAMATH_CALUDE_molecular_weight_8_moles_Al2O3_l2065_206532

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Aluminum atoms in one molecule of Al2O3 -/
def num_Al_atoms : ℕ := 2

/-- The number of Oxygen atoms in one molecule of Al2O3 -/
def num_O_atoms : ℕ := 3

/-- The number of moles of Al2O3 -/
def num_moles : ℕ := 8

/-- The molecular weight of Al2O3 in g/mol -/
def molecular_weight_Al2O3 : ℝ :=
  num_Al_atoms * atomic_weight_Al + num_O_atoms * atomic_weight_O

/-- Theorem: The molecular weight of 8 moles of Al2O3 is 815.68 grams -/
theorem molecular_weight_8_moles_Al2O3 :
  num_moles * molecular_weight_Al2O3 = 815.68 := by
  sorry


end NUMINAMATH_CALUDE_molecular_weight_8_moles_Al2O3_l2065_206532


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2065_206511

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 2 → x^2 - 1 > 0)) ↔ (∃ x : ℝ, x > 2 ∧ x^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2065_206511


namespace NUMINAMATH_CALUDE_polar_bear_daily_fish_consumption_l2065_206510

/-- The amount of trout eaten daily by the polar bear in buckets -/
def trout_amount : ℝ := 0.2

/-- The amount of salmon eaten daily by the polar bear in buckets -/
def salmon_amount : ℝ := 0.4

/-- The total amount of fish eaten daily by the polar bear in buckets -/
def total_fish_amount : ℝ := trout_amount + salmon_amount

theorem polar_bear_daily_fish_consumption :
  total_fish_amount = 0.6 := by sorry

end NUMINAMATH_CALUDE_polar_bear_daily_fish_consumption_l2065_206510


namespace NUMINAMATH_CALUDE_solve_for_y_l2065_206534

theorem solve_for_y (x y : ℝ) (h1 : x^(2*y) = 4) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2065_206534


namespace NUMINAMATH_CALUDE_quadratic_solution_value_l2065_206559

theorem quadratic_solution_value (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0 → 2023 - a - 2 * b = 2024 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_value_l2065_206559


namespace NUMINAMATH_CALUDE_quarter_percent_of_160_l2065_206565

theorem quarter_percent_of_160 : (1 / 4 : ℚ) / 100 * 160 = (0.4 : ℚ) := by sorry

end NUMINAMATH_CALUDE_quarter_percent_of_160_l2065_206565


namespace NUMINAMATH_CALUDE_condition_relationship_l2065_206543

theorem condition_relationship (x : ℝ) :
  ¬(∀ x, (1 / x ≤ 1 → (1/3)^x ≥ (1/2)^x)) ∧
  ¬(∀ x, ((1/3)^x ≥ (1/2)^x → 1 / x ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2065_206543


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_y_coordinates_l2065_206514

-- Define the function
def f (x : ℝ) : ℝ := x * (x - 4)^3

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 4 * (x - 4)^2 * (x - 1)

-- Theorem statement
theorem tangent_parallel_to_x_axis :
  ∀ x : ℝ, f' x = 0 ↔ x = 4 ∨ x = 1 :=
sorry

-- Verify the y-coordinates
theorem y_coordinates :
  f 4 = 0 ∧ f 1 = -27 :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_y_coordinates_l2065_206514


namespace NUMINAMATH_CALUDE_consecutive_root_count_l2065_206592

/-- A function that checks if a number is divisible by 5 -/
def divisible_by_five (m : ℤ) : Prop := ∃ k : ℤ, m = 5 * k

/-- A function that checks if two integers are consecutive -/
def consecutive (a b : ℤ) : Prop := b = a + 1

/-- A function that checks if a number is a positive integer -/
def is_positive_integer (x : ℤ) : Prop := x > 0

/-- The main theorem -/
theorem consecutive_root_count :
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, n < 50 ∧ is_positive_integer n) ∧ 
    (∀ n ∈ S, ∃ m : ℤ, 
      divisible_by_five m ∧
      ∃ a b : ℤ, is_positive_integer a ∧ is_positive_integer b ∧ consecutive a b ∧
      a * b = m ∧ a + b = n) ∧
    Finset.card S = 5 := by sorry

end NUMINAMATH_CALUDE_consecutive_root_count_l2065_206592


namespace NUMINAMATH_CALUDE_initial_value_proof_l2065_206598

theorem initial_value_proof : 
  (∃ x : ℕ, (x + 294) % 456 = 0 ∧ ∀ y : ℕ, y < x → (y + 294) % 456 ≠ 0) → 
  (∃! x : ℕ, (x + 294) % 456 = 0 ∧ ∀ y : ℕ, y < x → (y + 294) % 456 ≠ 0) ∧
  (∃ x : ℕ, (x + 294) % 456 = 0 ∧ ∀ y : ℕ, y < x → (y + 294) % 456 ≠ 0 ∧ x = 162) :=
by sorry

end NUMINAMATH_CALUDE_initial_value_proof_l2065_206598


namespace NUMINAMATH_CALUDE_surface_area_difference_l2065_206575

/-- The difference in surface area between 8 unit cubes and a cube with volume 8 -/
theorem surface_area_difference (large_cube_volume : ℝ) (small_cube_volume : ℝ) 
  (num_small_cubes : ℕ) (h1 : large_cube_volume = 8) (h2 : small_cube_volume = 1) 
  (h3 : num_small_cubes = 8) : 
  (num_small_cubes : ℝ) * (6 * small_cube_volume ^ (2/3)) - 
  (6 * large_cube_volume ^ (2/3)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_difference_l2065_206575
