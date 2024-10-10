import Mathlib

namespace henrys_cd_collection_l3763_376386

theorem henrys_cd_collection (country rock classical : ℕ) : 
  country = rock + 3 →
  rock = 2 * classical →
  country = 23 →
  classical = 10 := by
sorry

end henrys_cd_collection_l3763_376386


namespace problem_statement_l3763_376390

theorem problem_statement : ¬(
  (∀ (p q : Prop), (p → ¬p) ↔ (q → ¬p)) ∧
  ((∀ x : ℝ, x ∈ Set.Icc 0 1 → Real.exp x ≥ 1) ∧ 
   (∃ x : ℝ, x^2 + x + 1 < 0)) ∧
  (¬∀ (a b m : ℝ), a * m^2 < b * m^2 → a < b) ∧
  (∀ (a b : ℝ), (a + b) / 2 ≥ Real.sqrt (a * b) → (a > 0 ∧ b > 0))
) := by sorry

end problem_statement_l3763_376390


namespace new_person_weight_is_85_l3763_376376

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + n * avg_increase

/-- Theorem stating that the weight of the new person is 85 kg -/
theorem new_person_weight_is_85 :
  new_person_weight 8 2.5 65 = 85 := by
  sorry

end new_person_weight_is_85_l3763_376376


namespace pirate_treasure_sum_l3763_376391

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The value of sapphires in base 7 -/
def sapphires : List Nat := [2, 3, 5, 6]

/-- The value of silverware in base 7 -/
def silverware : List Nat := [0, 5, 6, 1]

/-- The value of spices in base 7 -/
def spices : List Nat := [0, 5, 2]

/-- The theorem stating the sum of the treasures in base 10 -/
theorem pirate_treasure_sum :
  base7ToBase10 sapphires + base7ToBase10 silverware + base7ToBase10 spices = 3131 := by
  sorry


end pirate_treasure_sum_l3763_376391


namespace inequality_implies_a_value_l3763_376308

theorem inequality_implies_a_value (a : ℝ) 
  (h : ∀ x : ℝ, x > 0 → (x^2 + a*x - 5)*(a*x - 1) ≥ 0) : 
  a = 1/2 := by
sorry

end inequality_implies_a_value_l3763_376308


namespace problem_statement_l3763_376373

open Real

-- Define the propositions
def p : Prop := ∀ x, cos (2*x - π/5) = cos (2*(x - π/5))

def q : Prop := ∀ α, tan α = 2 → (cos α)^2 - 2*(sin α)^2 = -7/4 * sin (2*α)

-- State the theorem
theorem problem_statement : (¬p) ∧ q := by sorry

end problem_statement_l3763_376373


namespace division_problem_l3763_376398

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2 / 5) : 
  c / a = 5 / 6 := by sorry

end division_problem_l3763_376398


namespace complement_intersection_theorem_l3763_376306

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set M
def M : Set Int := {-1, 0, 1, 3}

-- Define set N
def N : Set Int := {-2, 0, 2, 3}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl M ∩ N) = {-2, 2} := by sorry

end complement_intersection_theorem_l3763_376306


namespace cone_lateral_surface_area_l3763_376371

theorem cone_lateral_surface_area (r : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = 120) :
  (θ / 360) * π * r^2 = 12 * π :=
sorry

end cone_lateral_surface_area_l3763_376371


namespace largest_view_angle_point_l3763_376348

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an angle -/
structure Angle where
  vertex : Point
  side1 : Point
  side2 : Point

/-- Checks if an angle is acute -/
def isAcute (α : Angle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Checks if a point is on one side of an angle -/
def isOnSide (p : Point) (α : Angle) : Prop := sorry

/-- Checks if a point is on the other side of an angle -/
def isOnOtherSide (p : Point) (α : Angle) : Prop := sorry

/-- Calculates the angle at which a segment is seen from a point -/
def viewAngle (p : Point) (a b : Point) : ℝ := sorry

/-- States that a point maximizes the view angle of a segment -/
def maximizesViewAngle (c : Point) (a b : Point) (α : Angle) : Prop :=
  ∀ p, isOnOtherSide p α → viewAngle c a b ≥ viewAngle p a b

theorem largest_view_angle_point (α : Angle) (a b c : Point) :
  isAcute α →
  isOnSide a α →
  isOnSide b α →
  isOnOtherSide c α →
  maximizesViewAngle c a b α →
  (distance α.vertex c)^2 = distance α.vertex a * distance α.vertex b := by
  sorry

end largest_view_angle_point_l3763_376348


namespace polar_to_cartesian_equivalence_l3763_376355

/-- Given a curve C in the Cartesian coordinate system with polar equation ρ = 2cosθ - 4sinθ,
    prove that its Cartesian equation is (x - 2)² - 15y² = 68 - (y + 8)² -/
theorem polar_to_cartesian_equivalence (x y ρ θ : ℝ) :
  ρ = 2 * Real.cos θ - 4 * Real.sin θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  (x - 2)^2 - 15 * y^2 = 68 - (y + 8)^2 :=
by sorry

end polar_to_cartesian_equivalence_l3763_376355


namespace wedding_guest_ratio_l3763_376325

def wedding_guests (bridgette_guests : ℕ) (extra_plates : ℕ) (spears_per_plate : ℕ) (total_spears : ℕ) : Prop :=
  ∃ (alex_guests : ℕ),
    (bridgette_guests + alex_guests + extra_plates) * spears_per_plate = total_spears ∧
    alex_guests * 3 = bridgette_guests * 2

theorem wedding_guest_ratio :
  wedding_guests 84 10 8 1200 :=
sorry

end wedding_guest_ratio_l3763_376325


namespace parallel_vectors_m_value_l3763_376385

/-- Given two vectors a and b in ℝ², where a = (2, m) and b = (l, -2),
    if a is parallel to a + 2b, then m = -4. -/
theorem parallel_vectors_m_value
  (m l : ℝ)
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h1 : a = (2, m))
  (h2 : b = (l, -2))
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ a = k • (a + 2 • b)) :
  m = -4 := by
  sorry

end parallel_vectors_m_value_l3763_376385


namespace hyperbola_equation_l3763_376383

/-- The equation of a hyperbola with given parameters -/
theorem hyperbola_equation (a c : ℝ) (h1 : a > 0) (h2 : c > a) :
  let e := c / a
  let b := Real.sqrt (c^2 - a^2)
  ∀ x y : ℝ, 2 * a = 8 → e = 5/4 →
    (x^2 / a^2 - y^2 / b^2 = 1) ↔ (x^2 / 16 - y^2 / 9 = 1) :=
by sorry

end hyperbola_equation_l3763_376383


namespace value_of_A_l3763_376319

-- Define the letter values as variables
variable (M A T H E : ℤ)

-- State the theorem
theorem value_of_A 
  (h_H : H = 8)
  (h_MATH : M + A + T + H = 32)
  (h_TEAM : T + E + A + M = 40)
  (h_MEET : M + E + E + T = 36) :
  A = 20 := by
sorry

end value_of_A_l3763_376319


namespace production_days_calculation_l3763_376300

/-- Given the average production and a new day's production, find the number of previous days. -/
theorem production_days_calculation (avg_n : ℝ) (new_prod : ℝ) (avg_n_plus_1 : ℝ) :
  avg_n = 50 →
  new_prod = 100 →
  avg_n_plus_1 = 55 →
  (avg_n * n + new_prod) / (n + 1) = avg_n_plus_1 →
  n = 9 :=
by
  sorry

end production_days_calculation_l3763_376300


namespace least_period_scaled_least_period_sum_sine_cosine_least_period_sin_cos_least_period_cos_sin_l3763_376342

-- Definition of periodic function
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Definition of least period
def least_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  is_periodic f T ∧ ∀ T', 0 < T' ∧ T' < T → ¬ is_periodic f T'

-- Theorem 1
theorem least_period_scaled (g : ℝ → ℝ) :
  least_period g π → least_period (fun x ↦ g (x / 3)) (3 * π) := by sorry

-- Theorem 2
theorem least_period_sum_sine_cosine :
  least_period (fun x ↦ Real.sin (8 * x) + Real.cos (4 * x)) (π / 2) := by sorry

-- Theorem 3
theorem least_period_sin_cos :
  least_period (fun x ↦ Real.sin (Real.cos x)) (2 * π) := by sorry

-- Theorem 4
theorem least_period_cos_sin :
  least_period (fun x ↦ Real.cos (Real.sin x)) π := by sorry

end least_period_scaled_least_period_sum_sine_cosine_least_period_sin_cos_least_period_cos_sin_l3763_376342


namespace greatest_common_multiple_10_15_under_150_greatest_common_multiple_10_15_under_150_is_120_l3763_376365

theorem greatest_common_multiple_10_15_under_150 : ℕ → Prop :=
  fun n =>
    (∃ k : ℕ, n = 10 * k) ∧
    (∃ k : ℕ, n = 15 * k) ∧
    n < 150 ∧
    ∀ m : ℕ, (∃ k : ℕ, m = 10 * k) ∧ (∃ k : ℕ, m = 15 * k) ∧ m < 150 → m ≤ n →
    n = 120

-- The proof goes here
theorem greatest_common_multiple_10_15_under_150_is_120 :
  greatest_common_multiple_10_15_under_150 120 :=
sorry

end greatest_common_multiple_10_15_under_150_greatest_common_multiple_10_15_under_150_is_120_l3763_376365


namespace halloween_cleanup_time_l3763_376388

theorem halloween_cleanup_time (
  egg_cleanup_time : ℕ)
  (tp_cleanup_time : ℕ)
  (graffiti_cleanup_time : ℕ)
  (pumpkin_cleanup_time : ℕ)
  (num_eggs : ℕ)
  (num_tp_rolls : ℕ)
  (sq_ft_graffiti : ℕ)
  (num_pumpkins : ℕ)
  (h1 : egg_cleanup_time = 15)
  (h2 : tp_cleanup_time = 30)
  (h3 : graffiti_cleanup_time = 45)
  (h4 : pumpkin_cleanup_time = 10)
  (h5 : num_eggs = 60)
  (h6 : num_tp_rolls = 7)
  (h7 : sq_ft_graffiti = 8)
  (h8 : num_pumpkins = 5) :
  (num_eggs * egg_cleanup_time) / 60 +
  num_tp_rolls * tp_cleanup_time +
  sq_ft_graffiti * graffiti_cleanup_time +
  num_pumpkins * pumpkin_cleanup_time = 635 := by
sorry

end halloween_cleanup_time_l3763_376388


namespace algebraic_expression_value_l3763_376324

theorem algebraic_expression_value (a b : ℝ) (h : 2 * a - b = 5) :
  2 * b - 4 * a + 8 = -2 := by
  sorry

end algebraic_expression_value_l3763_376324


namespace sqrt_seven_minus_fraction_greater_than_reciprocal_l3763_376302

theorem sqrt_seven_minus_fraction_greater_than_reciprocal 
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : Real.sqrt 7 - m / n > 0) : 
  Real.sqrt 7 - m / n > 1 / (m * n) := by
sorry

end sqrt_seven_minus_fraction_greater_than_reciprocal_l3763_376302


namespace base8_573_equals_379_l3763_376387

/-- Converts a base-8 number to base 10 --/
def base8_to_base10 (a b c : ℕ) : ℕ := a * 8^2 + b * 8^1 + c * 8^0

/-- The base-8 number 573₈ is equal to 379 in base 10 --/
theorem base8_573_equals_379 : base8_to_base10 5 7 3 = 379 := by
  sorry

end base8_573_equals_379_l3763_376387


namespace intersection_nonempty_implies_m_range_l3763_376384

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x^2 - 4*m*x + 2*m + 6 = 0}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem intersection_nonempty_implies_m_range (m : ℝ) :
  (A m ∩ B).Nonempty → m ≤ -1 := by
  sorry

end intersection_nonempty_implies_m_range_l3763_376384


namespace ali_seashells_l3763_376367

theorem ali_seashells (initial : ℕ) (given_to_friends : ℕ) (left_after_selling : ℕ) :
  initial = 180 →
  given_to_friends = 40 →
  left_after_selling = 55 →
  ∃ (given_to_brothers : ℕ),
    given_to_brothers = 30 ∧
    2 * left_after_selling = initial - given_to_friends - given_to_brothers :=
by sorry

end ali_seashells_l3763_376367


namespace function_range_iff_a_ge_one_l3763_376331

/-- Given a real number a, the function f(x) = √((a-1)x² + ax + 1) has range [0, +∞) if and only if a ≥ 1 -/
theorem function_range_iff_a_ge_one (a : ℝ) :
  (Set.range (fun x => Real.sqrt ((a - 1) * x^2 + a * x + 1)) = Set.Ici 0) ↔ a ≥ 1 := by
  sorry

end function_range_iff_a_ge_one_l3763_376331


namespace equation_solution_l3763_376341

theorem equation_solution : ∃! x : ℚ, 2 * x - 5/6 = 7/18 + 1/2 ∧ x = 31/36 := by
  sorry

end equation_solution_l3763_376341


namespace income_expenditure_ratio_l3763_376350

/-- Given a person's income and savings, calculate the ratio of income to expenditure -/
theorem income_expenditure_ratio 
  (income : ℕ) 
  (savings : ℕ) 
  (h1 : income = 10000) 
  (h2 : savings = 4000) : 
  (income : ℚ) / (income - savings) = 5 / 3 := by
  sorry

end income_expenditure_ratio_l3763_376350


namespace log_equation_implies_sum_l3763_376382

theorem log_equation_implies_sum (x y : ℝ) 
  (h1 : x > 1) (h2 : y > 1) 
  (h3 : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 6 = 
        6 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) : 
  x^Real.sqrt 3 + y^Real.sqrt 3 = 189 := by
  sorry

end log_equation_implies_sum_l3763_376382


namespace star_op_identity_l3763_376352

/-- Define the * operation on ordered pairs of real numbers -/
def star_op (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, a * d + b * c)

/-- Theorem: If (a, b) * (x, y) = (a, b) and a² ≠ b², then (x, y) = (1, 0) -/
theorem star_op_identity {a b x y : ℝ} (h : a^2 ≠ b^2) :
  star_op a b x y = (a, b) → (x, y) = (1, 0) := by
  sorry

end star_op_identity_l3763_376352


namespace simplify_sqrt_one_third_l3763_376323

theorem simplify_sqrt_one_third : Real.sqrt (1/3) = Real.sqrt 3 / 3 := by
  sorry

end simplify_sqrt_one_third_l3763_376323


namespace susan_strawberry_picking_l3763_376321

theorem susan_strawberry_picking (basket_capacity : ℕ) (total_picked : ℕ) (eaten_per_handful : ℕ) :
  basket_capacity = 60 →
  total_picked = 75 →
  eaten_per_handful = 1 →
  ∃ (strawberries_per_handful : ℕ),
    strawberries_per_handful * (total_picked / strawberries_per_handful) = total_picked ∧
    (strawberries_per_handful - eaten_per_handful) * (total_picked / strawberries_per_handful) = basket_capacity ∧
    strawberries_per_handful = 5 :=
by sorry

end susan_strawberry_picking_l3763_376321


namespace trick_or_treat_total_l3763_376353

/-- Calculates the total number of treats received by children while trick-or-treating. -/
def total_treats (num_children : ℕ) (hours_out : ℕ) (houses_per_hour : ℕ) (treats_per_child_per_house : ℕ) : ℕ :=
  num_children * hours_out * houses_per_hour * treats_per_child_per_house

/-- Proves that given the specific conditions, the total number of treats is 180. -/
theorem trick_or_treat_total (num_children : ℕ) (hours_out : ℕ) (houses_per_hour : ℕ) (treats_per_child_per_house : ℕ) 
    (h1 : num_children = 3)
    (h2 : hours_out = 4)
    (h3 : houses_per_hour = 5)
    (h4 : treats_per_child_per_house = 3) :
  total_treats num_children hours_out houses_per_hour treats_per_child_per_house = 180 := by
  sorry

end trick_or_treat_total_l3763_376353


namespace circles_intersect_l3763_376307

theorem circles_intersect : ∃ (x y : ℝ),
  (x^2 + y^2 - 8*x + 6*y - 11 = 0) ∧ (x^2 + y^2 = 100) := by
  sorry

end circles_intersect_l3763_376307


namespace equation_system_solution_l3763_376399

/-- Given a system of equations, prove the values of x and y, and the expression for 2p + q -/
theorem equation_system_solution (p q r x y : ℚ) 
  (eq1 : p / q = 6 / 7)
  (eq2 : p / r = 8 / 9)
  (eq3 : q / r = x / y) :
  x = 28 ∧ y = 27 ∧ 2 * p + q = 19 / 6 * p := by
  sorry

end equation_system_solution_l3763_376399


namespace all_subjects_identified_l3763_376310

theorem all_subjects_identified (num_colors : ℕ) (num_subjects : ℕ) : 
  num_colors = 5 → num_subjects = 16 → num_colors ^ 2 ≥ num_subjects := by
  sorry

#check all_subjects_identified

end all_subjects_identified_l3763_376310


namespace valid_paths_count_l3763_376333

/-- Represents a point on a 2D grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points on a grid --/
def numPaths (start finish : Point) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- Calculates the number of paths between two points passing through an intermediate point --/
def numPathsThrough (start mid finish : Point) : ℕ :=
  (numPaths start mid) * (numPaths mid finish)

/-- The main theorem stating the number of valid paths --/
theorem valid_paths_count :
  let start := Point.mk 0 0
  let finish := Point.mk 5 3
  let risky := Point.mk 2 2
  (numPaths start finish) - (numPathsThrough start risky finish) = 32 := by
  sorry

end valid_paths_count_l3763_376333


namespace money_division_l3763_376394

theorem money_division (a b c : ℕ) (h1 : a = b / 2) (h2 : b = c / 2) (h3 : c = 224) :
  a + b + c = 392 := by
  sorry

end money_division_l3763_376394


namespace min_value_of_f_l3763_376361

def f (x : ℝ) := x^2 + 2

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 2 := by
  sorry

end min_value_of_f_l3763_376361


namespace cubic_fraction_factorization_l3763_376330

theorem cubic_fraction_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) 
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end cubic_fraction_factorization_l3763_376330


namespace g_of_6_eq_0_l3763_376396

/-- The polynomial g(x) = 3x^4 - 18x^3 + 31x^2 - 29x - 72 -/
def g (x : ℝ) : ℝ := 3*x^4 - 18*x^3 + 31*x^2 - 29*x - 72

/-- Theorem: g(6) = 0 -/
theorem g_of_6_eq_0 : g 6 = 0 := by sorry

end g_of_6_eq_0_l3763_376396


namespace commute_time_difference_l3763_376346

theorem commute_time_difference (x y : ℝ) : 
  (x + y + 10 + 11 + 9) / 5 = 10 →
  ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2 →
  |x - y| = 4 := by
sorry

end commute_time_difference_l3763_376346


namespace max_intersections_fifth_degree_polynomials_l3763_376379

-- Define a fifth degree polynomial with leading coefficient 1
def FifthDegreePolynomial (a b c d e : ℝ) : ℝ → ℝ := λ x => x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

-- Theorem statement
theorem max_intersections_fifth_degree_polynomials 
  (p q : ℝ → ℝ) 
  (hp : ∃ a b c d e, p = FifthDegreePolynomial a b c d e) 
  (hq : ∃ a' b' c' d' e', q = FifthDegreePolynomial a' b' c' d' e') 
  (hpq_diff : p ≠ q) :
  ∃ S : Finset ℝ, (∀ x ∈ S, p x = q x) ∧ S.card ≤ 4 :=
sorry

end max_intersections_fifth_degree_polynomials_l3763_376379


namespace plant_species_numbering_impossibility_l3763_376326

theorem plant_species_numbering_impossibility :
  ∃ (a b : ℕ), 2 ≤ a ∧ a < b ∧ b ≤ 20000 ∧
  (∀ (x : ℕ), 2 ≤ x ∧ x ≤ 20000 →
    (Nat.gcd a x = Nat.gcd b x)) :=
sorry

end plant_species_numbering_impossibility_l3763_376326


namespace wheel_probability_l3763_376359

theorem wheel_probability (p_D p_E p_F : ℚ) : 
  p_D = 2/5 → p_E = 1/3 → p_D + p_E + p_F = 1 → p_F = 4/15 := by
  sorry

end wheel_probability_l3763_376359


namespace unique_triple_solution_l3763_376357

theorem unique_triple_solution : 
  ∃! (x y z : ℕ+), 
    (¬(3 ∣ z ∧ y ∣ z)) ∧ 
    (Nat.Prime y) ∧ 
    (x^3 - y^3 = z^2) ∧
    x = 8 ∧ y = 7 ∧ z = 13 := by
  sorry

end unique_triple_solution_l3763_376357


namespace equation_solution_set_l3763_376392

theorem equation_solution_set : ∃ (S : Set ℝ),
  S = {x : ℝ | Real.sqrt (x + 3 - 4 * Real.sqrt (x - 1)) + Real.sqrt (x + 8 - 6 * Real.sqrt (x - 1)) = 1 ∧
                x ≥ 5 ∧ x ≤ 10} ∧
  ∀ x : ℝ, x ∈ S ↔ (Real.sqrt (x + 3 - 4 * Real.sqrt (x - 1)) + Real.sqrt (x + 8 - 6 * Real.sqrt (x - 1)) = 1 ∧
                    x ≥ 5 ∧ x ≤ 10) :=
by
  sorry

end equation_solution_set_l3763_376392


namespace right_triangle_trig_sum_l3763_376328

theorem right_triangle_trig_sum (A B C : Real) : 
  -- Conditions
  A = π / 2 →  -- A = 90° in radians
  0 < B → B < π / 2 →  -- B is acute angle
  C = π / 2 - B →  -- C is complementary to B in right triangle
  -- Theorem
  Real.sin A + Real.sin B ^ 2 + Real.sin C ^ 2 = 2 := by
  sorry

end right_triangle_trig_sum_l3763_376328


namespace bar_chart_clarity_l3763_376309

/-- Represents a bar chart --/
structure BarChart where
  data : List (String × ℝ)

/-- Represents the clarity of quantity representation in a chart --/
def ClearQuantityRepresentation : Prop := True

/-- Theorem: A bar chart clearly shows the amount of each quantity it represents --/
theorem bar_chart_clarity (chart : BarChart) : ClearQuantityRepresentation := by
  sorry

end bar_chart_clarity_l3763_376309


namespace quadratic_inequality_solution_l3763_376318

theorem quadratic_inequality_solution (x : ℝ) : 
  (2 * x^2 + x < 6) ↔ (-2 < x ∧ x < 3/2) := by sorry

end quadratic_inequality_solution_l3763_376318


namespace square_field_area_l3763_376372

theorem square_field_area (diagonal : ℝ) (h : diagonal = 16) : 
  let side := diagonal / Real.sqrt 2
  let area := side ^ 2
  area = 128 := by sorry

end square_field_area_l3763_376372


namespace square_greater_than_negative_l3763_376303

theorem square_greater_than_negative (x : ℝ) : x < 0 → x^2 > x := by
  sorry

end square_greater_than_negative_l3763_376303


namespace total_sugar_third_layer_is_correct_l3763_376375

/-- The amount of sugar needed for the smallest layer of the cake -/
def smallest_layer_sugar : ℝ := 2

/-- The size multiplier for the second layer compared to the first -/
def second_layer_multiplier : ℝ := 1.5

/-- The size multiplier for the third layer compared to the second -/
def third_layer_multiplier : ℝ := 2.5

/-- The percentage of sugar loss while baking each layer -/
def sugar_loss_percentage : ℝ := 0.15

/-- Calculates the total cups of sugar needed for the third layer -/
def total_sugar_third_layer : ℝ :=
  smallest_layer_sugar * second_layer_multiplier * third_layer_multiplier * (1 + sugar_loss_percentage)

/-- Theorem stating that the total sugar needed for the third layer is 8.625 cups -/
theorem total_sugar_third_layer_is_correct :
  total_sugar_third_layer = 8.625 := by
  sorry

end total_sugar_third_layer_is_correct_l3763_376375


namespace non_constant_geometric_sequence_exists_l3763_376356

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence is non-constant -/
def NonConstant (a : ℕ → ℝ) : Prop :=
  ∃ m n : ℕ, a m ≠ a n

theorem non_constant_geometric_sequence_exists :
  ∃ a : ℕ → ℝ, GeometricSequence a ∧ NonConstant a ∧
  ∃ r s : ℕ, r ≠ s ∧ a r = a s :=
by sorry

end non_constant_geometric_sequence_exists_l3763_376356


namespace points_on_line_or_circle_l3763_376369

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in a 2D plane -/
structure Circle2D where
  center : Point2D
  radius : ℝ

/-- Function to check if three points are collinear -/
def areCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

/-- Function to generate points based on the described process -/
def generatePoints (p1 p2 p3 : Point2D) : Set Point2D :=
  sorry

/-- The main theorem -/
theorem points_on_line_or_circle (p1 p2 p3 : Point2D) :
  ∃ (l : Line2D) (c : Circle2D), 
    (areCollinear p1 p2 p3 ∧ generatePoints p1 p2 p3 ⊆ {p | p.x * l.a + p.y * l.b + l.c = 0}) ∨
    (¬areCollinear p1 p2 p3 ∧ generatePoints p1 p2 p3 ⊆ {p | (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2}) :=
  sorry

end points_on_line_or_circle_l3763_376369


namespace f_properties_l3763_376336

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 * Real.exp x

theorem f_properties :
  (∃ x, f x = 0) ∧
  (∃ x₁ x₂, IsLocalMax f x₁ ∧ IsLocalMin f x₂) ∧
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = 1 ∧ f x₂ = 1 ∧ f x₃ = 1 ∧
    ∀ x, f x = 1 → x = x₁ ∨ x = x₂ ∨ x = x₃) :=
by
  sorry

#check f_properties

end f_properties_l3763_376336


namespace robin_gum_count_l3763_376311

theorem robin_gum_count (initial : Real) (additional : Real) (total : Real) : 
  initial = 18.0 → additional = 44.0 → total = initial + additional → total = 62.0 := by
  sorry

end robin_gum_count_l3763_376311


namespace divisibility_by_three_l3763_376364

theorem divisibility_by_three (a : ℤ) : ¬(3 ∣ a) → (3 ∣ (5 * a^2 + 1)) := by sorry

end divisibility_by_three_l3763_376364


namespace smallest_number_of_guesses_l3763_376335

def is_determinable (guesses : List Nat) : Prop :=
  ∀ N : Nat, 1 < N → N < 100 → 
    ∃! N', 1 < N' → N' < 100 → 
      ∀ g ∈ guesses, g % N = g % N'

theorem smallest_number_of_guesses :
  ∃ guesses : List Nat,
    guesses.length = 6 ∧
    is_determinable guesses ∧
    ∀ guesses' : List Nat, guesses'.length < 6 → ¬is_determinable guesses' :=
sorry

end smallest_number_of_guesses_l3763_376335


namespace kiran_currency_notes_l3763_376397

/-- Represents the currency denominations in Rupees --/
inductive Denomination
  | fifty : Denomination
  | hundred : Denomination

/-- Represents the total amount and number of notes for each denomination --/
structure CurrencyNotes where
  total_amount : ℕ
  fifty_amount : ℕ
  fifty_count : ℕ
  hundred_count : ℕ

/-- Calculates the total number of currency notes --/
def total_notes (c : CurrencyNotes) : ℕ :=
  c.fifty_count + c.hundred_count

/-- Theorem stating that given the conditions, Kiran has 85 currency notes in total --/
theorem kiran_currency_notes :
  ∀ (c : CurrencyNotes),
    c.total_amount = 5000 →
    c.fifty_amount = 3500 →
    c.fifty_count = c.fifty_amount / 50 →
    c.hundred_count = (c.total_amount - c.fifty_amount) / 100 →
    total_notes c = 85 := by
  sorry

end kiran_currency_notes_l3763_376397


namespace shoe_probability_l3763_376332

/-- Represents the total number of shoe pairs -/
def total_pairs : ℕ := 16

/-- Represents the number of black shoe pairs -/
def black_pairs : ℕ := 8

/-- Represents the number of brown shoe pairs -/
def brown_pairs : ℕ := 5

/-- Represents the number of white shoe pairs -/
def white_pairs : ℕ := 3

/-- The probability of picking two shoes of the same color with one being left and the other right -/
theorem shoe_probability : 
  (black_pairs * black_pairs + brown_pairs * brown_pairs + white_pairs * white_pairs) / 
  (total_pairs * (2 * total_pairs - 1)) = 49 / 248 := by
  sorry

end shoe_probability_l3763_376332


namespace geometric_sequence_theorem_l3763_376347

/-- A geometric sequence with positive common ratio -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_theorem (a : ℕ → ℝ) :
  GeometricSequence a →
  a 2 * a 10 = 2 * (a 5)^2 →
  a 2 = 1 →
  ∀ n : ℕ, a n = 2^((n - 2) / 2) :=
by sorry

end geometric_sequence_theorem_l3763_376347


namespace ab_nonpositive_l3763_376368

theorem ab_nonpositive (a b : ℝ) (h : 2011 * a + 2012 * b = 0) : a * b ≤ 0 := by
  sorry

end ab_nonpositive_l3763_376368


namespace intersection_P_Q_l3763_376370

def P : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def Q : Set ℝ := {1, 2, 3, 4}

theorem intersection_P_Q : P ∩ Q = {1, 2} := by sorry

end intersection_P_Q_l3763_376370


namespace problems_solved_l3763_376362

theorem problems_solved (first last : ℕ) (h : first = 70 ∧ last = 125) : last - first + 1 = 56 := by
  sorry

end problems_solved_l3763_376362


namespace isosceles_trapezoid_theorem_l3763_376315

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoid (a b : ℝ) :=
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (a_lt_b : a < b)

/-- Properties of the isosceles trapezoid -/
def trapezoid_properties (a b : ℝ) (t : IsoscelesTrapezoid a b) :=
  let AB := (a + b) / 2
  let BH := Real.sqrt (a * b)
  let BP := 2 * a * b / (a + b)
  let DF := Real.sqrt ((a^2 + b^2) / 2)
  (AB = (a + b) / 2) ∧
  (BH = Real.sqrt (a * b)) ∧
  (BP = 2 * a * b / (a + b)) ∧
  (DF = Real.sqrt ((a^2 + b^2) / 2)) ∧
  (BP < BH) ∧ (BH < AB) ∧ (AB < DF)

/-- Theorem stating the properties of the isosceles trapezoid -/
theorem isosceles_trapezoid_theorem (a b : ℝ) (t : IsoscelesTrapezoid a b) :
  trapezoid_properties a b t := by
  sorry

end isosceles_trapezoid_theorem_l3763_376315


namespace union_of_A_and_B_l3763_376374

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by
  sorry

end union_of_A_and_B_l3763_376374


namespace quadratic_symmetry_l3763_376366

/-- A quadratic function with specific properties -/
def p (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_symmetry (d e f : ℝ) :
  (∀ x : ℝ, p d e f (10.5 + x) = p d e f (10.5 - x)) →  -- axis of symmetry at x = 10.5
  p d e f 3 = -5 →                                      -- passes through (3, -5)
  p d e f 12 = -5 :=                                    -- conclusion: p(12) = -5
by sorry

end quadratic_symmetry_l3763_376366


namespace fourth_day_jumps_l3763_376334

def jump_count (day : ℕ) : ℕ :=
  match day with
  | 0 => 0  -- day 0 is not defined in the problem, so we set it to 0
  | 1 => 15 -- first day
  | n + 1 => 2 * jump_count n -- subsequent days

theorem fourth_day_jumps :
  jump_count 4 = 120 :=
by sorry

end fourth_day_jumps_l3763_376334


namespace tangent_line_to_parabola_l3763_376381

/-- The equation of the tangent line to the parabola y = x² that is parallel to y = 2x -/
theorem tangent_line_to_parabola (x y : ℝ) : 
  (∀ t, y = t^2 → (2 * t = 2 → x = t ∧ y = t^2)) →
  (2 * x - y - 1 = 0) :=
sorry

end tangent_line_to_parabola_l3763_376381


namespace calculate_total_profit_l3763_376389

/-- Given the investments of three partners and the profit share of one partner,
    calculate the total profit of the business. -/
theorem calculate_total_profit
  (a_investment b_investment c_investment : ℕ)
  (c_profit_share : ℕ)
  (h1 : a_investment = 5000)
  (h2 : b_investment = 15000)
  (h3 : c_investment = 30000)
  (h4 : c_profit_share = 3000) :
  (a_investment + b_investment + c_investment) * c_profit_share
  / c_investment = 5000 :=
sorry

end calculate_total_profit_l3763_376389


namespace triangle_inequality_expression_l3763_376327

theorem triangle_inequality_expression (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : 
  (a - b)^2 - c^2 < 0 := by
  sorry

end triangle_inequality_expression_l3763_376327


namespace exists_easy_a_difficult_b_l3763_376344

structure TestConfiguration where
  variants : Type
  students : Type
  problems : Type
  solved : variants → students → problems → Prop

def easy_a (tc : TestConfiguration) : Prop :=
  ∀ v : tc.variants, ∀ p : tc.problems, ∃ s : tc.students, tc.solved v s p

def difficult_b (tc : TestConfiguration) : Prop :=
  ∀ v : tc.variants, ¬∃ s : tc.students, ∀ p : tc.problems, tc.solved v s p

theorem exists_easy_a_difficult_b :
  ∃ tc : TestConfiguration, easy_a tc ∧ difficult_b tc := by
  sorry

end exists_easy_a_difficult_b_l3763_376344


namespace fraction_simplification_l3763_376339

theorem fraction_simplification :
  (30 : ℚ) / 35 * 21 / 45 * 70 / 63 - 2 / 3 = -8 / 15 := by
sorry

end fraction_simplification_l3763_376339


namespace josh_marbles_l3763_376313

/-- The number of marbles Josh had earlier -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Josh lost -/
def lost_marbles : ℕ := 11

/-- The number of marbles Josh has now -/
def current_marbles : ℕ := 8

/-- Theorem stating that the initial number of marbles is 19 -/
theorem josh_marbles : initial_marbles = lost_marbles + current_marbles := by sorry

end josh_marbles_l3763_376313


namespace min_running_time_l3763_376316

/-- Proves the minimum running time to cover a given distance within a time limit -/
theorem min_running_time 
  (total_distance : ℝ) 
  (time_limit : ℝ) 
  (walking_speed : ℝ) 
  (running_speed : ℝ) 
  (h1 : total_distance = 2.1) 
  (h2 : time_limit = 18) 
  (h3 : walking_speed = 90) 
  (h4 : running_speed = 210) :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ time_limit ∧ 
  running_speed * x + walking_speed * (time_limit - x) ≥ total_distance * 1000 :=
sorry

end min_running_time_l3763_376316


namespace danny_bottle_caps_count_l3763_376320

/-- The number of bottle caps Danny has in his collection now -/
def danny_bottle_caps : ℕ := 56

/-- The number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 46

/-- The number of bottle caps Danny found at the park -/
def bottle_caps_found : ℕ := 50

/-- The number of wrappers Danny has in his collection now -/
def wrappers_in_collection : ℕ := 52

theorem danny_bottle_caps_count :
  danny_bottle_caps = wrappers_in_collection + (bottle_caps_found - wrappers_found) :=
by sorry

end danny_bottle_caps_count_l3763_376320


namespace fifty_cent_items_count_l3763_376395

theorem fifty_cent_items_count (a b c : ℕ) : 
  a + b + c = 50 →
  50 * a + 400 * b + 500 * c = 10000 →
  a = 40 :=
by sorry

end fifty_cent_items_count_l3763_376395


namespace cycle_cost_price_l3763_376380

-- Define the cost price and selling price
def cost_price : ℝ := 1600
def selling_price : ℝ := 1360

-- Define the loss percentage
def loss_percentage : ℝ := 15

-- Theorem statement
theorem cycle_cost_price : 
  selling_price = cost_price * (1 - loss_percentage / 100) := by
  sorry

end cycle_cost_price_l3763_376380


namespace alice_bob_meet_l3763_376377

/-- The number of points on the circle -/
def n : ℕ := 15

/-- Alice's movement per turn (clockwise) -/
def alice_move : ℕ := 7

/-- Bob's movement per turn (counterclockwise) -/
def bob_move : ℕ := 11

/-- The starting position for both Alice and Bob -/
def start_pos : ℕ := n

/-- The number of turns after which Alice and Bob meet -/
def meeting_turns : ℕ := 5

/-- The position on the circle after a given number of clockwise moves -/
def position_after_moves (start : ℕ) (moves : ℕ) : ℕ :=
  (start + moves - 1) % n + 1

theorem alice_bob_meet :
  position_after_moves start_pos (meeting_turns * alice_move) =
  position_after_moves start_pos (meeting_turns * (n - bob_move)) :=
sorry

end alice_bob_meet_l3763_376377


namespace no_k_exists_for_prime_and_binomial_cong_l3763_376378

theorem no_k_exists_for_prime_and_binomial_cong (k : ℕ+) (p : ℕ) : 
  p = 6 * k + 1 → 
  Nat.Prime p → 
  (Nat.choose (3 * k) k : ZMod p) = 1 → 
  False := by sorry

end no_k_exists_for_prime_and_binomial_cong_l3763_376378


namespace line_passes_through_fixed_point_l3763_376322

/-- A line defined by the equation ax + (2-a)y + 1 = 0 -/
def line (a : ℝ) (x y : ℝ) : Prop := a * x + (2 - a) * y + 1 = 0

/-- The theorem states that for any real number a, 
    the line ax + (2-a)y + 1 = 0 passes through the point (-1/2, -1/2) -/
theorem line_passes_through_fixed_point :
  ∀ a : ℝ, line a (-1/2) (-1/2) :=
by
  sorry

end line_passes_through_fixed_point_l3763_376322


namespace puppy_price_calculation_l3763_376317

/-- Calculates the price per puppy in John's puppy selling scenario -/
theorem puppy_price_calculation (initial_puppies : ℕ) (stud_fee profit : ℚ) : 
  initial_puppies = 8 →
  stud_fee = 300 →
  profit = 1500 →
  (initial_puppies / 2 - 1) > 0 →
  (profit + stud_fee) / (initial_puppies / 2 - 1) = 600 := by
sorry

end puppy_price_calculation_l3763_376317


namespace condition_C_necessary_for_A_l3763_376354

-- Define the conditions as propositions
variable (A B C D : Prop)

-- Define the relationship between the conditions
variable (h : (C → D) → (A → B))

-- Theorem to prove
theorem condition_C_necessary_for_A (h : (C → D) → (A → B)) : A → C :=
  sorry

end condition_C_necessary_for_A_l3763_376354


namespace evaluate_expression_l3763_376337

theorem evaluate_expression : 225 + 2 * 15 * 8 + 64 = 529 := by
  sorry

end evaluate_expression_l3763_376337


namespace probability_x_lt_2y_is_one_sixth_l3763_376363

/-- A rectangle in the 2D plane -/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  x_min_le_x_max : x_min ≤ x_max
  y_min_le_y_max : y_min ≤ y_max

/-- The probability that a randomly chosen point (x,y) from the given rectangle satisfies x < 2y -/
def probability_x_lt_2y (r : Rectangle) : ℝ :=
  sorry

/-- The specific rectangle with vertices (0,0), (6,0), (6,1), and (0,1) -/
def specific_rectangle : Rectangle :=
  { x_min := 0
  , x_max := 6
  , y_min := 0
  , y_max := 1
  , x_min_le_x_max := by norm_num
  , y_min_le_y_max := by norm_num
  }

/-- Theorem stating that the probability of x < 2y for a randomly chosen point
    in the specific rectangle is 1/6 -/
theorem probability_x_lt_2y_is_one_sixth :
  probability_x_lt_2y specific_rectangle = 1/6 := by
  sorry

end probability_x_lt_2y_is_one_sixth_l3763_376363


namespace inscribed_squares_ratio_l3763_376314

/-- Right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_a : a = 5
  side_b : b = 12
  side_c : c = 13

/-- Square inscribed with one vertex at the right angle -/
def inscribed_square_x (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x < t.a ∧ x < t.b ∧ x / t.a = x / t.b

/-- Square inscribed with one side along the hypotenuse -/
def inscribed_square_y (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y < t.c ∧ (t.a / t.c) * y / t.a = y / t.c

/-- The main theorem -/
theorem inscribed_squares_ratio (t : RightTriangle) 
  (x y : ℝ) (hx : inscribed_square_x t x) (hy : inscribed_square_y t y) : 
  x / y = 4320 / 2873 := by
  sorry

end inscribed_squares_ratio_l3763_376314


namespace max_balls_in_cube_l3763_376349

theorem max_balls_in_cube (cube_side : ℝ) (ball_radius : ℝ) :
  cube_side = 9 →
  ball_radius = 3 →
  ⌊(cube_side^3) / ((4/3) * π * ball_radius^3)⌋ = 6 := by
  sorry

end max_balls_in_cube_l3763_376349


namespace divisors_of_ten_factorial_greater_than_nine_factorial_l3763_376338

theorem divisors_of_ten_factorial_greater_than_nine_factorial : 
  (Finset.filter (fun d => d > Nat.factorial 9 ∧ Nat.factorial 10 % d = 0) 
    (Finset.range (Nat.factorial 10 + 1))).card = 9 := by
  sorry

end divisors_of_ten_factorial_greater_than_nine_factorial_l3763_376338


namespace polynomial_never_33_l3763_376304

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end polynomial_never_33_l3763_376304


namespace inequality_group_solution_set_l3763_376358

theorem inequality_group_solution_set :
  ∀ x : ℝ, (x > -3 ∧ x < 5) ↔ (-3 < x ∧ x < 5) :=
by sorry

end inequality_group_solution_set_l3763_376358


namespace walk_distance_proof_l3763_376351

/-- Calculates the distance traveled given a constant speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that walking at 4 miles per hour for 2 hours results in a distance of 8 miles -/
theorem walk_distance_proof :
  let speed : ℝ := 4
  let time : ℝ := 2
  distance_traveled speed time = 8 := by sorry

end walk_distance_proof_l3763_376351


namespace circle_M_equation_l3763_376329

/-- The equation of a line passing through the center of circle M -/
def center_line (x y : ℝ) : Prop := x - y - 4 = 0

/-- The equation of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0

/-- The equation of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

/-- The equation of circle M -/
def circle_M (x y : ℝ) : Prop := (x - 1/2)^2 + (y + 7/2)^2 = 89/2

/-- Theorem stating that the given conditions imply the equation of circle M -/
theorem circle_M_equation (x y : ℝ) :
  (∃ (xc yc : ℝ), center_line xc yc ∧ 
    (∀ (xi yi : ℝ), (circle1 xi yi ∧ circle2 xi yi) → 
      (x - xc)^2 + (y - yc)^2 = (xi - xc)^2 + (yi - yc)^2)) →
  circle_M x y :=
sorry

end circle_M_equation_l3763_376329


namespace dog_speed_is_16_l3763_376305

/-- Represents the scenario of a man and a dog walking on a path -/
structure WalkingScenario where
  path_length : Real
  man_speed : Real
  dog_trips : Nat
  remaining_distance : Real
  dog_speed : Real

/-- Checks if the given scenario is valid -/
def is_valid_scenario (s : WalkingScenario) : Prop :=
  s.path_length = 0.625 ∧
  s.man_speed = 4 ∧
  s.dog_trips = 4 ∧
  s.remaining_distance = 0.081 ∧
  s.dog_speed > s.man_speed

/-- Theorem: Given the conditions, the dog's speed is 16 km/h -/
theorem dog_speed_is_16 (s : WalkingScenario) 
  (h : is_valid_scenario s) : s.dog_speed = 16 := by
  sorry

#check dog_speed_is_16

end dog_speed_is_16_l3763_376305


namespace odd_function_sum_l3763_376301

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_property : ∀ x, f (2 + x) + f (2 - x) = 0)
  (h_f1 : f 1 = 9) :
  f 2010 + f 2011 + f 2012 = -9 := by
  sorry

end odd_function_sum_l3763_376301


namespace tangent_line_at_one_monotonicity_condition_l3763_376393

/-- The function f(x) = √x - a ln(x+1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt x - a * Real.log (x + 1)

/-- The derivative of f with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / (2 * Real.sqrt x) - a / (x + 1)

theorem tangent_line_at_one (a : ℝ) :
  a = -1 → (fun x => x + Real.log 2) = fun x => f (-1) 1 + f_deriv (-1) 1 * (x - 1) := by sorry

theorem monotonicity_condition (a : ℝ) :
  a ≤ 1 → ∀ x y, 0 ≤ x ∧ x < y → f a x < f a y := by sorry

end tangent_line_at_one_monotonicity_condition_l3763_376393


namespace total_dolls_count_l3763_376343

def grandmother_dolls : ℕ := 50

def sister_dolls (grandmother_dolls : ℕ) : ℕ := grandmother_dolls + 2

def rene_dolls (sister_dolls : ℕ) : ℕ := 3 * sister_dolls

def total_dolls (grandmother_dolls sister_dolls rene_dolls : ℕ) : ℕ :=
  grandmother_dolls + sister_dolls + rene_dolls

theorem total_dolls_count :
  total_dolls grandmother_dolls (sister_dolls grandmother_dolls) (rene_dolls (sister_dolls grandmother_dolls)) = 258 := by
  sorry

end total_dolls_count_l3763_376343


namespace triangle_angle_sum_impossibility_l3763_376360

theorem triangle_angle_sum_impossibility (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬((α + β < 120 ∧ β + γ < 120 ∧ γ + α < 120) ∨
    (α + β > 120 ∧ β + γ > 120 ∧ γ + α > 120)) := by
  sorry

end triangle_angle_sum_impossibility_l3763_376360


namespace rhombus_area_from_square_midpoints_l3763_376312

/-- The area of a rhombus formed by connecting the midpoints of a square with side length 4 is 8 -/
theorem rhombus_area_from_square_midpoints (s : ℝ) (h : s = 4) : 
  let rhombus_area := s^2 / 2
  rhombus_area = 8 := by
  sorry

end rhombus_area_from_square_midpoints_l3763_376312


namespace journey_satisfies_equations_l3763_376340

/-- Represents Li Hai's journey from point A to point B -/
structure Journey where
  totalDistance : ℝ
  uphillSpeed : ℝ
  downhillSpeed : ℝ
  totalTime : ℝ
  uphillTime : ℝ
  downhillTime : ℝ

/-- Checks if the given journey satisfies the system of equations -/
def satisfiesEquations (j : Journey) : Prop :=
  j.uphillTime + j.downhillTime = j.totalTime ∧
  (j.uphillSpeed * j.uphillTime / 60 + j.downhillSpeed * j.downhillTime / 60) * 1000 = j.totalDistance

/-- Theorem stating that Li Hai's journey satisfies the system of equations -/
theorem journey_satisfies_equations :
  ∀ (j : Journey),
    j.totalDistance = 1200 ∧
    j.uphillSpeed = 3 ∧
    j.downhillSpeed = 5 ∧
    j.totalTime = 16 →
    satisfiesEquations j :=
  sorry

#check journey_satisfies_equations

end journey_satisfies_equations_l3763_376340


namespace birdseed_mix_proportion_l3763_376345

/-- Proves that the proportion of Brand A in a birdseed mix is 60% when the mix is 50% sunflower -/
theorem birdseed_mix_proportion :
  ∀ (x : ℝ), 
  x ≥ 0 ∧ x ≤ 1 →  -- x represents the proportion of Brand A in the mix
  0.60 * x + 0.35 * (1 - x) = 0.50 →  -- The mix is 50% sunflower
  x = 0.60 := by
  sorry

end birdseed_mix_proportion_l3763_376345
