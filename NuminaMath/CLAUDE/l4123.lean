import Mathlib

namespace NUMINAMATH_CALUDE_tree_walk_properties_l4123_412344

/-- Represents a random walk on a line of trees. -/
structure TreeWalk where
  n : ℕ
  trees : Fin (2 * n + 1) → ℕ
  start : Fin (2 * n + 1)
  prob_left : ℚ
  prob_stay : ℚ
  prob_right : ℚ

/-- The probability of ending at a specific tree after the walk. -/
def end_probability (w : TreeWalk) (i : Fin (2 * w.n + 1)) : ℚ :=
  (Nat.choose (2 * w.n) (i - 1)) / (2 ^ (2 * w.n))

/-- The expected distance from the starting point after the walk. -/
def expected_distance (w : TreeWalk) : ℚ :=
  (w.n * Nat.choose (2 * w.n) w.n) / (2 ^ (2 * w.n))

/-- Theorem stating the properties of the random walk. -/
theorem tree_walk_properties (w : TreeWalk) 
  (h1 : w.n > 0)
  (h2 : w.start = ⟨w.n + 1, by sorry⟩)
  (h3 : w.prob_left = 1/4)
  (h4 : w.prob_stay = 1/2)
  (h5 : w.prob_right = 1/4) :
  (∀ i : Fin (2 * w.n + 1), end_probability w i = (Nat.choose (2 * w.n) (i - 1)) / (2 ^ (2 * w.n))) ∧
  expected_distance w = (w.n * Nat.choose (2 * w.n) w.n) / (2 ^ (2 * w.n)) := by
  sorry


end NUMINAMATH_CALUDE_tree_walk_properties_l4123_412344


namespace NUMINAMATH_CALUDE_no_solutions_condition_l4123_412386

theorem no_solutions_condition (b : ℝ) : 
  (∀ a x : ℝ, a > 1 → a^(2-2*x^2) + (b+4)*a^(1-x^2) + 3*b + 4 ≠ 0) ↔ 
  (b ∈ Set.Ioc (-4/3) 0 ∪ Set.Ici 4) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_condition_l4123_412386


namespace NUMINAMATH_CALUDE_geometric_series_property_l4123_412324

theorem geometric_series_property (b₁ q : ℝ) (h_q : |q| < 1) :
  (b₁ / (1 - q)) / (b₁^3 / (1 - q^3)) = 1/12 →
  (b₁^4 / (1 - q^4)) / (b₁^2 / (1 - q^2)) = 36/5 →
  (b₁ = 3 ∨ b₁ = -3) ∧ q = -1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_property_l4123_412324


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4123_412393

/-- Given a geometric sequence {a_n} where a_1 = 1 and 4a_2, 2a_3, a_4 form an arithmetic sequence,
    prove that the sum a_2 + a_3 + a_4 equals 14. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →                            -- a_1 = 1
  4 * a 2 - 2 * a 3 = 2 * a 3 - a 4 →  -- arithmetic sequence condition
  a 2 + a 3 + a 4 = 14 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4123_412393


namespace NUMINAMATH_CALUDE_initial_barking_dogs_l4123_412390

theorem initial_barking_dogs (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  additional = 10 → total = 40 → initial + additional = total → initial = 30 := by
sorry

end NUMINAMATH_CALUDE_initial_barking_dogs_l4123_412390


namespace NUMINAMATH_CALUDE_vip_tickets_count_l4123_412306

theorem vip_tickets_count (initial_savings : ℕ) (vip_ticket_cost : ℕ) (regular_ticket_cost : ℕ) (regular_tickets_count : ℕ) (remaining_money : ℕ) : 
  initial_savings = 500 →
  vip_ticket_cost = 100 →
  regular_ticket_cost = 50 →
  regular_tickets_count = 3 →
  remaining_money = 150 →
  ∃ vip_tickets_count : ℕ, 
    vip_tickets_count * vip_ticket_cost + regular_tickets_count * regular_ticket_cost = initial_savings - remaining_money ∧
    vip_tickets_count = 2 :=
by sorry

end NUMINAMATH_CALUDE_vip_tickets_count_l4123_412306


namespace NUMINAMATH_CALUDE_gray_squares_33_l4123_412304

/-- The number of squares in the n-th figure of the series -/
def total_squares (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The number of black squares in the n-th figure -/
def black_squares (n : ℕ) : ℕ := n ^ 2

/-- The number of white squares in the n-th figure -/
def white_squares (n : ℕ) : ℕ := (n - 1) ^ 2

/-- The number of gray squares in the n-th figure -/
def gray_squares (n : ℕ) : ℕ := total_squares n - black_squares n - white_squares n

theorem gray_squares_33 : gray_squares 33 = 2112 := by
  sorry

end NUMINAMATH_CALUDE_gray_squares_33_l4123_412304


namespace NUMINAMATH_CALUDE_difference_of_squares_l4123_412395

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4123_412395


namespace NUMINAMATH_CALUDE_total_pies_eq_750_l4123_412321

/-- The number of mini meat pies made by the first team -/
def team1_pies : ℕ := 235

/-- The number of mini meat pies made by the second team -/
def team2_pies : ℕ := 275

/-- The number of mini meat pies made by the third team -/
def team3_pies : ℕ := 240

/-- The total number of teams -/
def num_teams : ℕ := 3

/-- The total number of mini meat pies made by all teams -/
def total_pies : ℕ := team1_pies + team2_pies + team3_pies

theorem total_pies_eq_750 : total_pies = 750 := by
  sorry

end NUMINAMATH_CALUDE_total_pies_eq_750_l4123_412321


namespace NUMINAMATH_CALUDE_minimal_area_circle_equation_l4123_412388

/-- The standard equation of a circle with minimal area, given that its center is on the curve y² = x
    and it is tangent to the line x + 2y + 6 = 0 -/
theorem minimal_area_circle_equation (x y : ℝ) : 
  (∃ (cx cy : ℝ), cy^2 = cx ∧ (x - cx)^2 + (y - cy)^2 = ((x + 2*y + 6) / Real.sqrt 5)^2) →
  (x - 1)^2 + (y + 1)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_minimal_area_circle_equation_l4123_412388


namespace NUMINAMATH_CALUDE_peter_bought_five_large_glasses_l4123_412380

/-- The number of large glasses Peter bought -/
def num_large_glasses (small_cost large_cost total_money num_small change : ℕ) : ℕ :=
  (total_money - change - small_cost * num_small) / large_cost

/-- Proof that Peter bought 5 large glasses -/
theorem peter_bought_five_large_glasses :
  num_large_glasses 3 5 50 8 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_peter_bought_five_large_glasses_l4123_412380


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l4123_412362

theorem subtraction_of_fractions : 
  (16 : ℚ) / 24 - (1 + 2 / 9) = -5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l4123_412362


namespace NUMINAMATH_CALUDE_buckets_required_l4123_412345

/-- The number of buckets required to fill a tank with the original bucket size,
    given that 62.5 buckets are needed when the bucket capacity is reduced to two-fifths. -/
theorem buckets_required (original_buckets : ℝ) : 
  (62.5 * (2/5) * original_buckets = original_buckets) → original_buckets = 25 := by
  sorry

end NUMINAMATH_CALUDE_buckets_required_l4123_412345


namespace NUMINAMATH_CALUDE_alyssa_picked_32_limes_l4123_412301

/-- The number of limes Alyssa picked -/
def alyssas_limes (total_limes fred_limes nancy_limes : ℕ) : ℕ :=
  total_limes - (fred_limes + nancy_limes)

/-- Proof that Alyssa picked 32 limes -/
theorem alyssa_picked_32_limes :
  alyssas_limes 103 36 35 = 32 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_picked_32_limes_l4123_412301


namespace NUMINAMATH_CALUDE_matrix_not_invertible_sum_l4123_412366

def matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x + y, x, y],
    ![x, y + z, y],
    ![y, x, x + z]]

theorem matrix_not_invertible_sum (x y z : ℝ) :
  ¬(IsUnit (Matrix.det (matrix x y z))) →
  x + y + z = 0 →
  x / (y + z) + y / (x + z) + z / (x + y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_sum_l4123_412366


namespace NUMINAMATH_CALUDE_score_change_effect_l4123_412320

/-- Proves that changing one student's score from 86 to 74 in a group of 8 students
    with an initial average of 82.5 decreases the average by 1.5 points -/
theorem score_change_effect (n : ℕ) (initial_avg : ℚ) (old_score new_score : ℚ) :
  n = 8 →
  initial_avg = 82.5 →
  old_score = 86 →
  new_score = 74 →
  initial_avg - (n * initial_avg - old_score + new_score) / n = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_score_change_effect_l4123_412320


namespace NUMINAMATH_CALUDE_square_root_of_square_l4123_412396

theorem square_root_of_square (n : ℝ) (h : n = 36) : Real.sqrt (n ^ 2) = n := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_square_l4123_412396


namespace NUMINAMATH_CALUDE_segment_division_ratio_l4123_412377

/-- Given a line segment AC and points B and D on it, this theorem proves
    that if B divides AC in a 2:1 ratio and D divides AB in a 3:2 ratio,
    then D divides AC in a 2:3 ratio. -/
theorem segment_division_ratio (A B C D : ℝ) :
  (B - A) / (C - B) = 2 / 1 →
  (D - A) / (B - D) = 3 / 2 →
  (D - A) / (C - D) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_segment_division_ratio_l4123_412377


namespace NUMINAMATH_CALUDE_solve_for_q_l4123_412348

theorem solve_for_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_q_l4123_412348


namespace NUMINAMATH_CALUDE_horatio_sonnets_l4123_412311

/-- Represents the number of lines in a sonnet -/
def lines_per_sonnet : ℕ := 14

/-- Represents the number of sonnets the lady heard before telling Horatio to leave -/
def sonnets_heard : ℕ := 7

/-- Represents the number of romantic lines Horatio wrote that were never heard -/
def unheard_lines : ℕ := 70

/-- Calculates the total number of sonnets Horatio wrote -/
def total_sonnets : ℕ := sonnets_heard + (unheard_lines / lines_per_sonnet)

theorem horatio_sonnets : total_sonnets = 12 := by sorry

end NUMINAMATH_CALUDE_horatio_sonnets_l4123_412311


namespace NUMINAMATH_CALUDE_decimal_to_binary_88_l4123_412352

theorem decimal_to_binary_88 : 
  (88 : ℕ).digits 2 = [0, 0, 0, 1, 1, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_decimal_to_binary_88_l4123_412352


namespace NUMINAMATH_CALUDE_sports_school_distribution_l4123_412326

theorem sports_school_distribution (total : ℕ) (skiing : ℕ) (speed_skating : ℕ) (hockey : ℕ) :
  total = 96 →
  speed_skating = (skiing * 4) / 5 →
  hockey = (skiing + speed_skating) / 3 →
  skiing + speed_skating + hockey = total →
  (skiing = 40 ∧ speed_skating = 32 ∧ hockey = 24) := by
  sorry

end NUMINAMATH_CALUDE_sports_school_distribution_l4123_412326


namespace NUMINAMATH_CALUDE_chord_quadrilateral_probability_l4123_412374

/-- Given 7 points on a circle, the probability that 4 randomly selected chords
    form a convex quadrilateral is 1/171. -/
theorem chord_quadrilateral_probability :
  let n : ℕ := 7  -- number of points on the circle
  let k : ℕ := 4  -- number of chords selected
  let total_chords : ℕ := n.choose 2  -- total number of possible chords
  let total_selections : ℕ := total_chords.choose k  -- ways to select k chords
  let convex_quads : ℕ := n.choose k  -- number of convex quadrilaterals
  (convex_quads : ℚ) / total_selections = 1 / 171 := by
sorry

end NUMINAMATH_CALUDE_chord_quadrilateral_probability_l4123_412374


namespace NUMINAMATH_CALUDE_toms_age_ratio_l4123_412391

theorem toms_age_ratio (T N : ℕ) : T > 0 → N > 0 → T = 7 * N := by
  sorry

#check toms_age_ratio

end NUMINAMATH_CALUDE_toms_age_ratio_l4123_412391


namespace NUMINAMATH_CALUDE_x_range_theorem_l4123_412316

-- Define the condition from the original problem
def satisfies_equation (x y : ℝ) : Prop :=
  x - 4 * Real.sqrt y = 2 * Real.sqrt (x - y)

-- Define the range of x
def x_range (x : ℝ) : Prop :=
  x ∈ Set.Icc 4 20 ∪ {0}

-- Theorem statement
theorem x_range_theorem :
  ∀ x y : ℝ, satisfies_equation x y → x_range x :=
by
  sorry

end NUMINAMATH_CALUDE_x_range_theorem_l4123_412316


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l4123_412350

theorem modulus_of_complex_fraction : 
  let z : ℂ := (2 - I) / (1 + 2*I)
  Complex.abs z = 1 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l4123_412350


namespace NUMINAMATH_CALUDE_vector_dot_product_l4123_412369

/-- Given vectors a, b, c in ℝ², if a is parallel to b, then b · c = 10 -/
theorem vector_dot_product (a b c : ℝ × ℝ) : 
  a = (-1, 2) → b.1 = 2 → c = (7, 1) → (∃ k : ℝ, b = k • a) → b.1 * c.1 + b.2 * c.2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l4123_412369


namespace NUMINAMATH_CALUDE_dot_product_range_l4123_412361

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4 ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8

-- Define points M and N on the hypotenuse
def OnHypotenuse (M N : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ (t s : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧
  M = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2) ∧
  N = (s * A.1 + (1 - s) * B.1, s * A.2 + (1 - s) * B.2)

-- Define the distance between M and N
def MNDistance (M N : ℝ × ℝ) : Prop :=
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 2

-- Define the dot product of CM and CN
def DotProduct (C M N : ℝ × ℝ) : ℝ :=
  (M.1 - C.1) * (N.1 - C.1) + (M.2 - C.2) * (N.2 - C.2)

theorem dot_product_range (A B C M N : ℝ × ℝ) :
  Triangle A B C →
  OnHypotenuse M N A B →
  MNDistance M N →
  (3/2 : ℝ) ≤ DotProduct C M N ∧ DotProduct C M N ≤ 2 := by sorry

end NUMINAMATH_CALUDE_dot_product_range_l4123_412361


namespace NUMINAMATH_CALUDE_preceding_number_in_base_3_l4123_412334

def base_3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3^i)) 0

def decimal_to_base_3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
  aux n []

theorem preceding_number_in_base_3 (N : Nat) (h : base_3_to_decimal [2, 1, 0, 1] = N) :
  decimal_to_base_3 (N - 1) = [2, 1, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_preceding_number_in_base_3_l4123_412334


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4123_412347

/-- Given a function y = log_a(x + 3) - 1 where a > 0 and a ≠ 1, 
    its graph always passes through a fixed point A.
    If point A lies on the line mx + ny + 2 = 0 where mn > 0,
    then the minimum value of 1/m + 2/n is 4. -/
theorem min_value_sum_reciprocals (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : m > 0) (h4 : n > 0) (h5 : m * n > 0) :
  let f : ℝ → ℝ := λ x => (Real.log x) / (Real.log a) - 1
  let A : ℝ × ℝ := (-2, -1)
  (f (A.1 + 3) = A.2) →
  (m * A.1 + n * A.2 + 2 = 0) →
  (∀ x y, f y = x → m * x + n * y + 2 = 0) →
  (1 / m + 2 / n) ≥ 4 ∧ ∃ m₀ n₀, 1 / m₀ + 2 / n₀ = 4 := by
  sorry


end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4123_412347


namespace NUMINAMATH_CALUDE_rectangle_perpendicular_point_theorem_l4123_412315

/-- Given a rectangle ABCD with point E on diagonal BD such that AE is perpendicular to BD -/
structure RectangleWithPerpendicularPoint where
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side BC -/
  BC : ℝ
  /-- Distance from E to DC -/
  n : ℝ
  /-- Distance from E to BC -/
  EC : ℝ
  /-- Distance from E to AB -/
  x : ℝ
  /-- Length of diagonal BD -/
  d : ℝ
  /-- EC is 1 -/
  h_EC : EC = 1
  /-- ABCD is a rectangle -/
  h_rectangle : AB > 0 ∧ BC > 0
  /-- E is on diagonal BD -/
  h_E_on_BD : d > 0
  /-- AE is perpendicular to BD -/
  h_AE_perp_BD : True

/-- The main theorem about the rectangle with perpendicular point -/
theorem rectangle_perpendicular_point_theorem (r : RectangleWithPerpendicularPoint) :
  /- Part a -/
  (r.d - r.x * Real.sqrt (1 + r.x^2))^2 = r.x^4 * (1 + r.x^2) ∧
  /- Part b -/
  r.n = r.x^3 ∧
  /- Part c -/
  r.d^(2/3) - r.x^(2/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perpendicular_point_theorem_l4123_412315


namespace NUMINAMATH_CALUDE_marias_savings_l4123_412360

/-- Represents the cost of the bike in dollars -/
def bike_cost : ℕ := 600

/-- Represents the amount Maria's mother offered in dollars -/
def mother_offer : ℕ := 250

/-- Represents the amount Maria needs to earn in dollars -/
def amount_to_earn : ℕ := 230

/-- Represents Maria's initial savings in dollars -/
def initial_savings : ℕ := 120

theorem marias_savings :
  initial_savings + mother_offer + amount_to_earn = bike_cost :=
sorry

end NUMINAMATH_CALUDE_marias_savings_l4123_412360


namespace NUMINAMATH_CALUDE_max_value_sum_of_squares_l4123_412317

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem max_value_sum_of_squares (u v w : V) 
  (hu : ‖u‖ = 3) (hv : ‖v‖ = 1) (hw : ‖w‖ = 2) : 
  ‖u - 3 • v‖^2 + ‖v - 3 • w‖^2 + ‖w - 3 • u‖^2 ≤ 224 := by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_squares_l4123_412317


namespace NUMINAMATH_CALUDE_fraction_simplification_l4123_412365

theorem fraction_simplification :
  ((2^1010)^2 - (2^1008)^2) / ((2^1009)^2 - (2^1007)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4123_412365


namespace NUMINAMATH_CALUDE_bowTie_seven_eq_nine_impl_g_eq_two_l4123_412392

/-- The bow-tie operation defined as a + √(b + √(b + √(b + ...))) -/
noncomputable def bowTie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

/-- Theorem stating that if 7 ⋈ g = 9, then g = 2 -/
theorem bowTie_seven_eq_nine_impl_g_eq_two :
  ∀ g : ℝ, bowTie 7 g = 9 → g = 2 := by
  sorry

end NUMINAMATH_CALUDE_bowTie_seven_eq_nine_impl_g_eq_two_l4123_412392


namespace NUMINAMATH_CALUDE_completing_square_sum_l4123_412322

theorem completing_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 9 = 0 ↔ (x + b)^2 = c) → b + c = -3 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_sum_l4123_412322


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l4123_412336

/-- Proves that given a simple interest of 4025.25, an interest rate of 9% per annum, 
and a time period of 5 years, the principal sum is 8950. -/
theorem simple_interest_principal_calculation :
  let simple_interest : ℝ := 4025.25
  let rate : ℝ := 9
  let time : ℝ := 5
  let principal : ℝ := simple_interest / (rate * time / 100)
  principal = 8950 := by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l4123_412336


namespace NUMINAMATH_CALUDE_combined_molecular_weight_l4123_412343

/-- Atomic weight of Nitrogen in g/mol -/
def N_weight : ℝ := 14.01

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- Molecular weight of N2O3 in g/mol -/
def N2O3_weight : ℝ := 2 * N_weight + 3 * O_weight

/-- Molecular weight of H2O in g/mol -/
def H2O_weight : ℝ := 2 * H_weight + O_weight

/-- Molecular weight of CO2 in g/mol -/
def CO2_weight : ℝ := C_weight + 2 * O_weight

/-- Combined molecular weight of 4 moles of N2O3, 3.5 moles of H2O, and 2 moles of CO2 in grams -/
theorem combined_molecular_weight :
  4 * N2O3_weight + 3.5 * H2O_weight + 2 * CO2_weight = 455.17 := by
  sorry

end NUMINAMATH_CALUDE_combined_molecular_weight_l4123_412343


namespace NUMINAMATH_CALUDE_two_extreme_points_l4123_412351

noncomputable section

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2 - x + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

/-- Theorem stating the range of a for which f(x) has two extreme value points -/
theorem two_extreme_points (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
    f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0) ↔ 
  (0 < a ∧ a < Real.exp (-1)) :=
sorry

end

end NUMINAMATH_CALUDE_two_extreme_points_l4123_412351


namespace NUMINAMATH_CALUDE_range_of_p_characterization_l4123_412381

def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

def range_of_p : Set ℝ := {p | B p ⊆ A}

theorem range_of_p_characterization :
  range_of_p = 
    {p | B p = ∅} ∪ 
    {p | B p ≠ ∅ ∧ ∀ x ∈ B p, x ∈ A} :=
by sorry

end NUMINAMATH_CALUDE_range_of_p_characterization_l4123_412381


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l4123_412308

theorem quadratic_vertex_form (x : ℝ) : ∃ (a h k : ℝ), 
  x^2 - 6*x + 1 = a*(x - h)^2 + k ∧ k = -8 := by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l4123_412308


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l4123_412394

theorem like_terms_exponent_sum (x y : ℝ) (m n : ℕ) :
  (∃ (k : ℝ), -x^6 * y^(2*m) = k * x^(n+2) * y^4) →
  n + m = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l4123_412394


namespace NUMINAMATH_CALUDE_fraction_simplification_l4123_412339

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4123_412339


namespace NUMINAMATH_CALUDE_breath_holding_improvement_l4123_412383

theorem breath_holding_improvement (initial_time : ℝ) : 
  initial_time = 10 → 
  (((initial_time * 2) * 2) * 1.5) = 60 := by
sorry

end NUMINAMATH_CALUDE_breath_holding_improvement_l4123_412383


namespace NUMINAMATH_CALUDE_new_oarsman_weight_l4123_412376

/-- Given a crew of 10 oarsmen, proves that replacing a 53 kg member with a new member
    that increases the average weight by 1.8 kg results in the new member weighing 71 kg. -/
theorem new_oarsman_weight (crew_size : Nat) (old_weight : ℝ) (avg_increase : ℝ) :
  crew_size = 10 →
  old_weight = 53 →
  avg_increase = 1.8 →
  (crew_size : ℝ) * avg_increase + old_weight = 71 :=
by sorry

end NUMINAMATH_CALUDE_new_oarsman_weight_l4123_412376


namespace NUMINAMATH_CALUDE_david_drive_distance_david_drive_distance_proof_l4123_412389

theorem david_drive_distance : ℝ → Prop :=
  fun distance =>
    ∀ (initial_speed : ℝ) (increased_speed : ℝ) (on_time_duration : ℝ),
      initial_speed = 40 ∧
      increased_speed = initial_speed + 20 ∧
      distance = initial_speed * (on_time_duration + 1.5) ∧
      distance - initial_speed = increased_speed * (on_time_duration - 2) →
      distance = 340

-- The proof is omitted
theorem david_drive_distance_proof : david_drive_distance 340 := by sorry

end NUMINAMATH_CALUDE_david_drive_distance_david_drive_distance_proof_l4123_412389


namespace NUMINAMATH_CALUDE_line_properties_l4123_412357

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0
def l₂ (a x y : ℝ) : Prop := 3 * x + (a - 1) * y + 3 - a = 0

-- Define perpendicularity of two lines
def perpendicular (a : ℝ) : Prop := (-a / 2) * (-3 / (a - 1)) = -1

theorem line_properties (a : ℝ) :
  (l₂ a (-2/3) 1) ∧
  (perpendicular a → a = 2/5) := by sorry

end NUMINAMATH_CALUDE_line_properties_l4123_412357


namespace NUMINAMATH_CALUDE_students_in_davids_grade_l4123_412354

/-- Given that David is both the 75th best and 75th worst student in his grade,
    prove that there are 149 students in total. -/
theorem students_in_davids_grade (n : ℕ) 
  (h1 : n ≥ 75)  -- David's grade has at least 75 students
  (h2 : ∃ (better worse : ℕ), better = 74 ∧ worse = 74 ∧ n = better + worse + 1) 
  : n = 149 := by
  sorry

end NUMINAMATH_CALUDE_students_in_davids_grade_l4123_412354


namespace NUMINAMATH_CALUDE_backyard_area_l4123_412340

theorem backyard_area (length width : ℝ) 
  (h1 : 40 * length = 1000) 
  (h2 : 8 * (2 * (length + width)) = 1000) : 
  length * width = 937.5 := by
sorry

end NUMINAMATH_CALUDE_backyard_area_l4123_412340


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l4123_412328

theorem fraction_inequality_solution_set (x : ℝ) : 
  (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l4123_412328


namespace NUMINAMATH_CALUDE_last_three_average_l4123_412325

theorem last_three_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 60 →
  (list.take 4).sum / 4 = 55 →
  (list.drop 4).sum / 3 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_last_three_average_l4123_412325


namespace NUMINAMATH_CALUDE_special_polynomial_max_value_l4123_412309

/-- A polynomial with real coefficients satisfying the given condition -/
def SpecialPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, P t = P 1 * t^2 + P (P 1) * t + P (P (P 1))

/-- The theorem stating the maximum value of P(P(P(P(1)))) -/
theorem special_polynomial_max_value (P : ℝ → ℝ) (h : SpecialPolynomial P) :
    ∃ M : ℝ, M = (1 : ℝ) / 9 ∧ P (P (P (P 1))) ≤ M ∧ 
    ∃ P₀ : ℝ → ℝ, SpecialPolynomial P₀ ∧ P₀ (P₀ (P₀ (P₀ 1))) = M :=
by sorry

end NUMINAMATH_CALUDE_special_polynomial_max_value_l4123_412309


namespace NUMINAMATH_CALUDE_range_of_expression_l4123_412353

theorem range_of_expression (x y : ℝ) (h : x^2 - y^2 = 4) :
  ∃ (z : ℝ), z = (1/x^2) - (y/x) ∧ -1 ≤ z ∧ z ≤ 5/4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l4123_412353


namespace NUMINAMATH_CALUDE_probability_sum_ten_l4123_412302

/-- Represents an octahedral die with 8 faces -/
def OctahedralDie := Fin 8

/-- The set of possible outcomes when rolling two octahedral dice -/
def DiceOutcomes := OctahedralDie × OctahedralDie

/-- The total number of possible outcomes when rolling two octahedral dice -/
def totalOutcomes : ℕ := 64

/-- Predicate to check if a pair of dice rolls sums to 10 -/
def sumsToTen (roll : DiceOutcomes) : Prop :=
  (roll.1.val + 1) + (roll.2.val + 1) = 10

/-- The number of favorable outcomes (sum of 10) -/
def favorableOutcomes : ℕ := 5

/-- Theorem stating the probability of rolling a sum of 10 -/
theorem probability_sum_ten :
  (favorableOutcomes : ℚ) / totalOutcomes = 5 / 64 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_ten_l4123_412302


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l4123_412359

def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def IsIncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) :
  IsGeometricSequence a →
  (a 1 < a 2 ∧ a 2 < a 3) ↔ IsIncreasingSequence a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l4123_412359


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l4123_412379

theorem sqrt_sum_equality : 
  Real.sqrt 9 + Real.sqrt (9 + 11) + Real.sqrt (9 + 11 + 13) + 
  Real.sqrt (9 + 11 + 13 + 15) + Real.sqrt (9 + 11 + 13 + 15 + 17) = 
  3 + 2 * Real.sqrt 5 + Real.sqrt 33 + 4 * Real.sqrt 3 + Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l4123_412379


namespace NUMINAMATH_CALUDE_fixed_points_are_corresponding_l4123_412303

/-- A type representing a geometric figure -/
structure Figure where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A type representing a point in a geometric figure -/
structure Point where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Predicate to check if three figures are similar -/
def are_similar (f1 f2 f3 : Figure) : Prop :=
  sorry

/-- Predicate to check if a point is fixed (invariant) in a figure -/
def is_fixed_point (p : Point) (f : Figure) : Prop :=
  sorry

/-- Predicate to check if two points are corresponding in two figures -/
def are_corresponding_points (p1 p2 : Point) (f1 f2 : Figure) : Prop :=
  sorry

/-- Theorem stating that fixed points of three similar figures are corresponding points -/
theorem fixed_points_are_corresponding
  (f1 f2 f3 : Figure)
  (h_similar : are_similar f1 f2 f3)
  (p1 : Point)
  (h_fixed1 : is_fixed_point p1 f1)
  (p2 : Point)
  (h_fixed2 : is_fixed_point p2 f2)
  (p3 : Point)
  (h_fixed3 : is_fixed_point p3 f3) :
  are_corresponding_points p1 p2 f1 f2 ∧
  are_corresponding_points p2 p3 f2 f3 ∧
  are_corresponding_points p1 p3 f1 f3 :=
by
  sorry

end NUMINAMATH_CALUDE_fixed_points_are_corresponding_l4123_412303


namespace NUMINAMATH_CALUDE_smallest_n_is_correct_l4123_412341

/-- The smallest positive integer n such that all roots of z^6 - z^3 + 1 = 0 are n-th roots of unity -/
def smallest_n : ℕ := 18

/-- The polynomial z^6 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^6 - z^3 + 1

theorem smallest_n_is_correct :
  smallest_n = 18 ∧
  (∀ z : ℂ, f z = 0 → z^smallest_n = 1) ∧
  (∀ m : ℕ, m < smallest_n → ∃ z : ℂ, f z = 0 ∧ z^m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_is_correct_l4123_412341


namespace NUMINAMATH_CALUDE_equation_solution_l4123_412319

theorem equation_solution : ∃ x : ℝ, (x - 3) ^ 4 = (1 / 16)⁻¹ ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4123_412319


namespace NUMINAMATH_CALUDE_base_number_proof_l4123_412387

theorem base_number_proof (n : ℕ) (x : ℝ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = x^26) 
  (h2 : n = 25) : 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l4123_412387


namespace NUMINAMATH_CALUDE_denis_neighbors_l4123_412367

-- Define the set of children
inductive Child : Type
| Anya : Child
| Borya : Child
| Vera : Child
| Gena : Child
| Denis : Child

-- Define the line as a function from position (1 to 5) to Child
def Line := Fin 5 → Child

-- Define the conditions
def is_valid_line (l : Line) : Prop :=
  -- Borya is at the beginning of the line
  l 1 = Child.Borya ∧
  -- Vera is next to Anya but not next to Gena
  (∃ i : Fin 4, (l i = Child.Vera ∧ l (i+1) = Child.Anya) ∨ (l (i+1) = Child.Vera ∧ l i = Child.Anya)) ∧
  (∀ i : Fin 4, ¬(l i = Child.Vera ∧ l (i+1) = Child.Gena) ∧ ¬(l (i+1) = Child.Vera ∧ l i = Child.Gena)) ∧
  -- Among Anya, Borya, and Gena, no two are standing next to each other
  (∀ i : Fin 4, ¬((l i = Child.Anya ∨ l i = Child.Borya ∨ l i = Child.Gena) ∧
                 (l (i+1) = Child.Anya ∨ l (i+1) = Child.Borya ∨ l (i+1) = Child.Gena)))

-- Theorem statement
theorem denis_neighbors (l : Line) (h : is_valid_line l) :
  (∃ i : Fin 4, (l i = Child.Anya ∧ l (i+1) = Child.Denis) ∨ (l (i+1) = Child.Anya ∧ l i = Child.Denis)) ∧
  (∃ j : Fin 4, (l j = Child.Gena ∧ l (j+1) = Child.Denis) ∨ (l (j+1) = Child.Gena ∧ l j = Child.Denis)) :=
by sorry

end NUMINAMATH_CALUDE_denis_neighbors_l4123_412367


namespace NUMINAMATH_CALUDE_greatest_power_of_three_l4123_412385

def w : ℕ := (List.range 30).foldl (· * ·) 1

theorem greatest_power_of_three (p : ℕ) : 
  (3^p ∣ w) ∧ ∀ q, q > p → ¬(3^q ∣ w) ↔ p = 15 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_l4123_412385


namespace NUMINAMATH_CALUDE_total_pints_picked_l4123_412364

def annie_pints : ℕ := 8

def kathryn_pints (annie : ℕ) : ℕ := annie + 2

def ben_pints (kathryn : ℕ) : ℕ := kathryn - 3

theorem total_pints_picked :
  annie_pints + kathryn_pints annie_pints + ben_pints (kathryn_pints annie_pints) = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_pints_picked_l4123_412364


namespace NUMINAMATH_CALUDE_third_number_proof_l4123_412342

theorem third_number_proof :
  ∃ x : ℝ, 12.1212 + 17.0005 - x = 20.011399999999995 ∧ x = 9.110300000000005 := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l4123_412342


namespace NUMINAMATH_CALUDE_kyle_pe_laps_l4123_412329

/-- The number of laps Kyle jogged during track practice -/
def track_laps : ℝ := 2.12

/-- The total number of laps Kyle jogged -/
def total_laps : ℝ := 3.25

/-- The number of laps Kyle jogged in P.E. class -/
def pe_laps : ℝ := total_laps - track_laps

theorem kyle_pe_laps : pe_laps = 1.13 := by
  sorry

end NUMINAMATH_CALUDE_kyle_pe_laps_l4123_412329


namespace NUMINAMATH_CALUDE_tree_planting_problem_l4123_412398

theorem tree_planting_problem (total_trees : ℕ) (a : ℕ) (b : ℕ) : 
  total_trees = 2013 →
  a * b = total_trees →
  (a - 5) * (b + 2) < total_trees →
  (a - 5) * (b + 3) > total_trees →
  a = 61 := by
sorry

end NUMINAMATH_CALUDE_tree_planting_problem_l4123_412398


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l4123_412330

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {x | x^2 - 3*x > 0}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l4123_412330


namespace NUMINAMATH_CALUDE_college_student_count_l4123_412370

/-- The total number of students at the college -/
def total_students : ℕ := 880

/-- The percentage of students enrolled in biology classes -/
def biology_enrollment_percentage : ℚ := 47.5 / 100

/-- The number of students not enrolled in biology classes -/
def students_not_in_biology : ℕ := 462

/-- Theorem stating the total number of students at the college -/
theorem college_student_count :
  total_students = students_not_in_biology / (1 - biology_enrollment_percentage) := by
  sorry

end NUMINAMATH_CALUDE_college_student_count_l4123_412370


namespace NUMINAMATH_CALUDE_mixed_fruit_juice_cost_l4123_412305

/-- The cost per litre of the superfruit juice cocktail -/
def cocktail_cost_per_litre : ℝ := 1399.45

/-- The cost per litre of açaí berry juice -/
def acai_cost_per_litre : ℝ := 3104.35

/-- The volume of mixed fruit juice used -/
def mixed_fruit_volume : ℝ := 32

/-- The volume of açaí berry juice used -/
def acai_volume : ℝ := 21.333333333333332

/-- The cost per litre of mixed fruit juice -/
def mixed_fruit_cost_per_litre : ℝ := 262.8125

theorem mixed_fruit_juice_cost : 
  cocktail_cost_per_litre * (mixed_fruit_volume + acai_volume) = 
  mixed_fruit_cost_per_litre * mixed_fruit_volume + acai_cost_per_litre * acai_volume := by
  sorry

end NUMINAMATH_CALUDE_mixed_fruit_juice_cost_l4123_412305


namespace NUMINAMATH_CALUDE_correct_operation_l4123_412384

theorem correct_operation (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l4123_412384


namespace NUMINAMATH_CALUDE_question_1_question_2_l4123_412337

-- Define the given conditions
def p (x : ℝ) : Prop := -x^2 + 2*x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0
def s (x : ℝ) : Prop := -x^2 + 8*x + 20 ≥ 0

-- Define the sufficient but not necessary conditions
def sufficient_not_necessary (P Q : ℝ → Prop) : Prop :=
  (∀ x, P x → Q x) ∧ ∃ x, Q x ∧ ¬(P x)

-- Theorem for the first question
theorem question_1 (m : ℝ) :
  m > 0 → (sufficient_not_necessary (p) (q m) → m ≥ 3) :=
sorry

-- Theorem for the second question
theorem question_2 (m : ℝ) :
  m > 0 → (sufficient_not_necessary (fun x => ¬(s x)) (fun x => ¬(q m x)) → False) :=
sorry

end NUMINAMATH_CALUDE_question_1_question_2_l4123_412337


namespace NUMINAMATH_CALUDE_vector_problem_l4123_412363

/-- Given two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_problem (x : ℝ) :
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (1/2, 1)
  let c : ℝ × ℝ := (a.1 + 2*b.1, a.2 + 2*b.2)
  let d : ℝ × ℝ := (2*a.1 - b.1, 2*a.2 - b.2)
  are_parallel c d →
  (c.1 - 2*d.1, c.2 - 2*d.2) = (-1, -2) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l4123_412363


namespace NUMINAMATH_CALUDE_arithmetic_mean_first_n_odd_l4123_412356

/-- The sum of the first n odd positive integers -/
def sum_first_n_odd (n : ℕ) : ℕ := n^2

/-- The arithmetic mean of a list of numbers -/
def arithmetic_mean (sum : ℕ) (count : ℕ) : ℚ := sum / count

theorem arithmetic_mean_first_n_odd (n : ℕ) :
  arithmetic_mean (sum_first_n_odd n) n = n := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_first_n_odd_l4123_412356


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l4123_412310

theorem quadratic_root_implies_coefficient (b : ℝ) : 
  (2^2 + b*2 - 10 = 0) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l4123_412310


namespace NUMINAMATH_CALUDE_winnie_lollipops_theorem_l4123_412332

/-- The number of lollipops Winnie keeps for herself -/
def lollipops_kept (total_lollipops friends : ℕ) : ℕ :=
  total_lollipops % friends

theorem winnie_lollipops_theorem (cherry wintergreen grape shrimp friends : ℕ) :
  let total_lollipops := cherry + wintergreen + grape + shrimp
  lollipops_kept total_lollipops friends = 
    total_lollipops - friends * (total_lollipops / friends) := by
  sorry

#eval lollipops_kept (67 + 154 + 23 + 312) 17

end NUMINAMATH_CALUDE_winnie_lollipops_theorem_l4123_412332


namespace NUMINAMATH_CALUDE_triangle_inequality_l4123_412346

theorem triangle_inequality (a b c p q r : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- triangle side lengths are positive
  a + b > c ∧ b + c > a ∧ c + a > b ∧  -- triangle inequality
  p + q + r = 0 →  -- sum of p, q, r is zero
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4123_412346


namespace NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus3_l4123_412382

theorem quadratic_root_sqrt5_minus3 : ∃ (a b c : ℚ), 
  a = 1 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus3_l4123_412382


namespace NUMINAMATH_CALUDE_ln_concave_l4123_412371

/-- The natural logarithm function is concave on the positive real numbers. -/
theorem ln_concave : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 →
  Real.log ((x₁ + x₂) / 2) ≥ (Real.log x₁ + Real.log x₂) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ln_concave_l4123_412371


namespace NUMINAMATH_CALUDE_female_officers_count_female_officers_count_proof_l4123_412312

/-- The number of female officers on a police force, given:
  * 10% of female officers were on duty
  * 200 officers were on duty in total
  * Half of the officers on duty were female
-/
theorem female_officers_count : ℕ :=
  let total_on_duty : ℕ := 200
  let female_ratio_on_duty : ℚ := 1/2
  let female_on_duty_ratio : ℚ := 1/10
  1000

/-- Proof that the number of female officers is correct -/
theorem female_officers_count_proof :
  let total_on_duty : ℕ := 200
  let female_ratio_on_duty : ℚ := 1/2
  let female_on_duty_ratio : ℚ := 1/10
  female_officers_count = 1000 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_female_officers_count_proof_l4123_412312


namespace NUMINAMATH_CALUDE_people_born_in_country_l4123_412372

theorem people_born_in_country (immigrants : ℕ) (new_residents : ℕ) 
  (h1 : immigrants = 16320) 
  (h2 : new_residents = 106491) : 
  new_residents - immigrants = 90171 := by
sorry

end NUMINAMATH_CALUDE_people_born_in_country_l4123_412372


namespace NUMINAMATH_CALUDE_larger_number_problem_l4123_412378

theorem larger_number_problem (x y : ℤ) : 
  x + y = 96 → y = x + 12 → y = 54 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l4123_412378


namespace NUMINAMATH_CALUDE_derivative_x_minus_reciprocal_l4123_412333

/-- The derivative of f(x) = x - 1/x is f'(x) = 1 + 1/x^2 -/
theorem derivative_x_minus_reciprocal (x : ℝ) (hx : x ≠ 0) :
  deriv (fun x => x - 1 / x) x = 1 + 1 / x^2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_minus_reciprocal_l4123_412333


namespace NUMINAMATH_CALUDE_crane_sling_diameter_l4123_412327

/-- Represents the problem of determining the smallest safe rope diameter for a crane sling. -/
theorem crane_sling_diameter
  (M : ℝ) -- Mass of the load in tons
  (n : ℕ) -- Number of slings
  (α : ℝ) -- Angle of each sling with vertical in radians
  (k : ℝ) -- Safety factor
  (q : ℝ) -- Maximum load per thread in N/mm²
  (g : ℝ) -- Free fall acceleration in m/s²
  (h : M = 20)
  (hn : n = 3)
  (hα : α = Real.pi / 6) -- 30° in radians
  (hk : k = 6)
  (hq : q = 1000)
  (hg : g = 10)
  : ∃ (D : ℕ), D = 26 ∧ 
    D = ⌈(4 * (k * M * g) / (n * q * π * Real.cos α))^(1/2) * 1000⌉ ∧
    ∀ (D' : ℕ), D' < D → 
      D' < ⌈(4 * (k * M * g) / (n * q * π * Real.cos α))^(1/2) * 1000⌉ :=
sorry

end NUMINAMATH_CALUDE_crane_sling_diameter_l4123_412327


namespace NUMINAMATH_CALUDE_distance_point_to_line_l4123_412318

/-- The distance between a point and a horizontal line is the absolute difference
    between their y-coordinates. -/
def distance_point_to_horizontal_line (point : ℝ × ℝ) (line_y : ℝ) : ℝ :=
  |point.2 - line_y|

/-- Theorem: The distance between the point (3, 0) and the line y = 1 is 1. -/
theorem distance_point_to_line : distance_point_to_horizontal_line (3, 0) 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l4123_412318


namespace NUMINAMATH_CALUDE_davids_age_l4123_412323

theorem davids_age (david : ℕ) (yuan : ℕ) : 
  yuan = david + 7 → yuan = 2 * david → david = 7 := by
  sorry

end NUMINAMATH_CALUDE_davids_age_l4123_412323


namespace NUMINAMATH_CALUDE_server_data_processing_l4123_412368

/-- Represents the data processing rate in megabytes per minute -/
def processing_rate : ℝ := 150

/-- Represents the time period in hours -/
def time_period : ℝ := 12

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℝ := 60

/-- Represents the number of megabytes in a gigabyte -/
def mb_per_gb : ℝ := 1000

/-- Theorem stating that the server processes 108 gigabytes in 12 hours -/
theorem server_data_processing :
  (processing_rate * time_period * minutes_per_hour) / mb_per_gb = 108 := by
  sorry

end NUMINAMATH_CALUDE_server_data_processing_l4123_412368


namespace NUMINAMATH_CALUDE_income_percentage_increase_l4123_412313

/-- Calculates the percentage increase in monthly income given initial and new weekly incomes -/
theorem income_percentage_increase 
  (initial_job_income initial_freelance_income : ℚ)
  (new_job_income new_freelance_income : ℚ)
  (weeks_per_month : ℕ)
  (h1 : initial_job_income = 60)
  (h2 : initial_freelance_income = 40)
  (h3 : new_job_income = 120)
  (h4 : new_freelance_income = 60)
  (h5 : weeks_per_month = 4) :
  let initial_monthly_income := (initial_job_income + initial_freelance_income) * weeks_per_month
  let new_monthly_income := (new_job_income + new_freelance_income) * weeks_per_month
  (new_monthly_income - initial_monthly_income) / initial_monthly_income * 100 = 80 :=
by sorry


end NUMINAMATH_CALUDE_income_percentage_increase_l4123_412313


namespace NUMINAMATH_CALUDE_cos_difference_l4123_412373

theorem cos_difference (A B : Real) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 2) : 
  Real.cos (A - B) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_l4123_412373


namespace NUMINAMATH_CALUDE_complex_equality_l4123_412335

theorem complex_equality (z : ℂ) : 
  z = -1 + I ↔ Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
               Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l4123_412335


namespace NUMINAMATH_CALUDE_cat_dog_positions_after_365_moves_l4123_412331

/-- Represents the positions on the 3x3 grid --/
inductive GridPosition
  | TopLeft | TopCenter | TopRight
  | MiddleLeft | Center | MiddleRight
  | BottomLeft | BottomCenter | BottomRight

/-- Represents the edge positions on the 3x3 grid --/
inductive EdgePosition
  | LeftTop | LeftMiddle | LeftBottom
  | BottomLeft | BottomCenter | BottomRight
  | RightBottom | RightMiddle | RightTop
  | TopRight | TopCenter | TopLeft

/-- Calculates the cat's position after a given number of moves --/
def catPosition (moves : ℕ) : GridPosition :=
  match moves % 9 with
  | 0 => GridPosition.TopLeft
  | 1 => GridPosition.TopCenter
  | 2 => GridPosition.TopRight
  | 3 => GridPosition.MiddleRight
  | 4 => GridPosition.BottomRight
  | 5 => GridPosition.BottomCenter
  | 6 => GridPosition.BottomLeft
  | 7 => GridPosition.MiddleLeft
  | _ => GridPosition.Center

/-- Calculates the dog's position after a given number of moves --/
def dogPosition (moves : ℕ) : EdgePosition :=
  match moves % 16 with
  | 0 => EdgePosition.LeftMiddle
  | 1 => EdgePosition.LeftTop
  | 2 => EdgePosition.TopLeft
  | 3 => EdgePosition.TopCenter
  | 4 => EdgePosition.TopRight
  | 5 => EdgePosition.RightTop
  | 6 => EdgePosition.RightMiddle
  | 7 => EdgePosition.RightBottom
  | 8 => EdgePosition.BottomRight
  | 9 => EdgePosition.BottomCenter
  | 10 => EdgePosition.BottomLeft
  | 11 => EdgePosition.LeftBottom
  | 12 => EdgePosition.LeftMiddle
  | 13 => EdgePosition.LeftTop
  | 14 => EdgePosition.TopLeft
  | _ => EdgePosition.TopCenter

theorem cat_dog_positions_after_365_moves :
  catPosition 365 = GridPosition.Center ∧ dogPosition 365 = EdgePosition.LeftMiddle :=
sorry

end NUMINAMATH_CALUDE_cat_dog_positions_after_365_moves_l4123_412331


namespace NUMINAMATH_CALUDE_find_A_l4123_412338

theorem find_A : ∃ A : ℤ, A + 10 = 15 → A = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l4123_412338


namespace NUMINAMATH_CALUDE_binomial_sum_unique_l4123_412397

theorem binomial_sum_unique (m : ℤ) : 
  (Nat.choose 25 m.toNat + Nat.choose 25 12 = Nat.choose 26 13) ↔ m = 13 :=
sorry

end NUMINAMATH_CALUDE_binomial_sum_unique_l4123_412397


namespace NUMINAMATH_CALUDE_vector_equality_l4123_412355

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-2, 4)

theorem vector_equality : c = a - 3 • b := by sorry

end NUMINAMATH_CALUDE_vector_equality_l4123_412355


namespace NUMINAMATH_CALUDE_total_votes_l4123_412349

theorem total_votes (votes_for votes_against total : ℕ) : 
  votes_for = votes_against + 66 →
  votes_against = (40 * total) / 100 →
  votes_for + votes_against = total →
  total = 330 := by
sorry

end NUMINAMATH_CALUDE_total_votes_l4123_412349


namespace NUMINAMATH_CALUDE_angle_120_in_second_quadrant_l4123_412307

/-- An angle in the Cartesian plane -/
structure CartesianAngle where
  /-- The measure of the angle in degrees -/
  measure : ℝ
  /-- The angle's vertex is at the origin -/
  vertex_at_origin : Bool
  /-- The angle's initial side is along the positive x-axis -/
  initial_side_positive_x : Bool

/-- Definition of the second quadrant -/
def is_in_second_quadrant (angle : CartesianAngle) : Prop :=
  angle.measure > 90 ∧ angle.measure < 180

/-- Theorem: An angle of 120° with vertex at origin and initial side along positive x-axis is in the second quadrant -/
theorem angle_120_in_second_quadrant :
  ∀ (angle : CartesianAngle),
    angle.measure = 120 ∧
    angle.vertex_at_origin = true ∧
    angle.initial_side_positive_x = true →
    is_in_second_quadrant angle :=
by sorry

end NUMINAMATH_CALUDE_angle_120_in_second_quadrant_l4123_412307


namespace NUMINAMATH_CALUDE_mixture_weight_is_3_64_l4123_412300

-- Define the weights of brands in grams per liter
def weight_a : ℚ := 950
def weight_b : ℚ := 850

-- Define the ratio of volumes
def ratio_a : ℚ := 3
def ratio_b : ℚ := 2

-- Define the total volume in liters
def total_volume : ℚ := 4

-- Define the function to calculate the weight of the mixture in kg
def mixture_weight : ℚ :=
  ((ratio_a / (ratio_a + ratio_b)) * total_volume * weight_a +
   (ratio_b / (ratio_a + ratio_b)) * total_volume * weight_b) / 1000

-- Theorem statement
theorem mixture_weight_is_3_64 : mixture_weight = 3.64 := by
  sorry

end NUMINAMATH_CALUDE_mixture_weight_is_3_64_l4123_412300


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l4123_412358

theorem quadratic_inequality_condition (x : ℝ) : 
  (((x < 1) ∨ (x > 4)) → (x^2 - 3*x + 2 > 0)) ∧ 
  (∃ x, (x^2 - 3*x + 2 > 0) ∧ ¬((x < 1) ∨ (x > 4))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l4123_412358


namespace NUMINAMATH_CALUDE_marble_probability_l4123_412399

theorem marble_probability (total : ℕ) (red : ℕ) (white : ℕ) (green : ℕ)
  (h_total : total = 100)
  (h_red : red = 35)
  (h_white : white = 30)
  (h_green : green = 10) :
  (red + white + green : ℚ) / total = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l4123_412399


namespace NUMINAMATH_CALUDE_leak_drain_time_l4123_412375

-- Define the pump fill rate
def pump_rate : ℚ := 1 / 2

-- Define the time it takes to fill the tank with the leak
def fill_time_with_leak : ℚ := 17 / 8

-- Define the leak rate
def leak_rate : ℚ := pump_rate - (1 / fill_time_with_leak)

-- Theorem to prove
theorem leak_drain_time (pump_rate : ℚ) (fill_time_with_leak : ℚ) (leak_rate : ℚ) :
  pump_rate = 1 / 2 →
  fill_time_with_leak = 17 / 8 →
  leak_rate = pump_rate - (1 / fill_time_with_leak) →
  (1 / leak_rate) = 34 := by
  sorry

end NUMINAMATH_CALUDE_leak_drain_time_l4123_412375


namespace NUMINAMATH_CALUDE_johns_remaining_money_l4123_412314

/-- Calculates the remaining money for John after transactions --/
def remaining_money (initial_amount : ℚ) (sister_fraction : ℚ) (groceries_cost : ℚ) (gift_cost : ℚ) : ℚ :=
  initial_amount - (sister_fraction * initial_amount) - groceries_cost - gift_cost

/-- Theorem stating that John's remaining money is $11.67 --/
theorem johns_remaining_money :
  remaining_money 100 (1/3) 40 15 = 35/3 :=
by sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l4123_412314
