import Mathlib

namespace fraction_inequality_l3283_328341

theorem fraction_inequality (a b t : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : t > 0) :
  a / b > (a + t) / (b + t) := by
  sorry

end fraction_inequality_l3283_328341


namespace parallelogram_fourth_vertex_l3283_328331

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
def Vector2D := Point2D

/-- Addition of two vectors -/
def vectorAdd (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Subtraction of two vectors -/
def vectorSub (v w : Vector2D) : Vector2D :=
  ⟨v.x - w.x, v.y - w.y⟩

/-- Negation of a vector -/
def vectorNeg (v : Vector2D) : Vector2D :=
  ⟨-v.x, -v.y⟩

theorem parallelogram_fourth_vertex 
  (A B C : Point2D)
  (h : A = ⟨-1, -2⟩)
  (h' : B = ⟨3, 1⟩)
  (h'' : C = ⟨0, 2⟩) :
  let D := vectorAdd C (vectorAdd A (vectorNeg B))
  D = ⟨-4, -1⟩ := by
  sorry

end parallelogram_fourth_vertex_l3283_328331


namespace eight_bead_bracelet_arrangements_l3283_328383

/-- The number of distinct arrangements of n distinct beads on a bracelet,
    considering rotational and reflectional symmetries --/
def braceletArrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of distinct arrangements of 8 distinct beads
    on a bracelet, considering rotational and reflectional symmetries, is 2520 --/
theorem eight_bead_bracelet_arrangements :
  braceletArrangements 8 = 2520 := by
  sorry

end eight_bead_bracelet_arrangements_l3283_328383


namespace first_player_wins_l3283_328399

/-- Represents a position on the 8x8 grid --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Defines the possible moves --/
inductive Move
  | Right
  | Up
  | UpRight

/-- Applies a move to a position --/
def applyMove (p : Position) (m : Move) : Position :=
  match m with
  | Move.Right => ⟨p.x + 1, p.y⟩
  | Move.Up => ⟨p.x, p.y + 1⟩
  | Move.UpRight => ⟨p.x + 1, p.y + 1⟩

/-- Checks if a position is within the 8x8 grid --/
def isValidPosition (p : Position) : Prop :=
  1 ≤ p.x ∧ p.x ≤ 8 ∧ 1 ≤ p.y ∧ p.y ≤ 8

/-- Defines a winning position --/
def isWinningPosition (p : Position) : Prop :=
  p.x = 8 ∧ p.y = 8

/-- Theorem: The first player has a winning strategy --/
theorem first_player_wins :
  ∃ (m : Move), isValidPosition (applyMove ⟨1, 1⟩ m) ∧
  ∀ (p : Position),
    isValidPosition p →
    ¬isWinningPosition p →
    (p.x % 2 = 0 ∧ p.y % 2 = 0) →
    ∃ (m : Move),
      isValidPosition (applyMove p m) ∧
      ¬(applyMove p m).x % 2 = 0 ∧
      ¬(applyMove p m).y % 2 = 0 :=
by sorry

#check first_player_wins

end first_player_wins_l3283_328399


namespace binomial_product_l3283_328373

theorem binomial_product (x : ℝ) : (4 * x - 3) * (x + 7) = 4 * x^2 + 25 * x - 21 := by
  sorry

end binomial_product_l3283_328373


namespace complex_multiplication_l3283_328365

theorem complex_multiplication (z : ℂ) (h : z = 1 + 2 * I) : I * z = -2 + I := by sorry

end complex_multiplication_l3283_328365


namespace arithmetic_simplification_l3283_328315

theorem arithmetic_simplification : (4 + 4 + 6) / 2 - 2 / 2 = 6 := by
  sorry

end arithmetic_simplification_l3283_328315


namespace sequence_minus_two_is_geometric_l3283_328371

/-- Given a sequence a and its partial sums s, prove {a n - 2} is geometric -/
theorem sequence_minus_two_is_geometric
  (a : ℕ+ → ℝ)  -- The sequence a_n
  (s : ℕ+ → ℝ)  -- The sequence of partial sums s_n
  (h : ∀ n : ℕ+, s n + a n = 2 * n)  -- The given condition
  : ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) - 2 = r * (a n - 2) :=
sorry

end sequence_minus_two_is_geometric_l3283_328371


namespace inequality_proof_l3283_328339

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  a*b/c + b*c/a + c*a/b ≥ Real.sqrt 3 := by
  sorry

end inequality_proof_l3283_328339


namespace indeterminate_157th_digit_l3283_328385

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ :=
  sorry

/-- The nth digit after the decimal point in the decimal representation of q -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem indeterminate_157th_digit :
  ∀ (d : ℕ),
  (∃ (q : ℚ), q = 525 / 2027 ∧ 
    (∀ (i : ℕ), i < 6 → nth_digit_after_decimal q (i + 1) = nth_digit_after_decimal (0.258973 : ℚ) (i + 1))) →
  (∃ (r : ℚ), r ≠ q ∧ 
    (∀ (i : ℕ), i < 6 → nth_digit_after_decimal r (i + 1) = nth_digit_after_decimal (0.258973 : ℚ) (i + 1)) ∧
    nth_digit_after_decimal r 157 ≠ d) :=
by
  sorry


end indeterminate_157th_digit_l3283_328385


namespace orthocenter_locus_l3283_328355

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  (ha : a > 0)
  (hb : b > 0)
  (hba : b ≤ a)

/-- A triangle inscribed in an ellipse -/
structure InscribedTriangle (e : Ellipse a b) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  hA : A = (-a, 0)
  hB : B = (a, 0)
  hC : (C.1^2 / a^2) + (C.2^2 / b^2) = 1

/-- The orthocenter of a triangle -/
def orthocenter (t : InscribedTriangle e) : ℝ × ℝ :=
  sorry

/-- The locus of the orthocenter is an ellipse -/
theorem orthocenter_locus (e : Ellipse a b) :
  ∀ t : InscribedTriangle e,
  let M := orthocenter t
  ((M.1^2 / a^2) + (M.2^2 / (a^2/b)^2) = 1) :=
sorry

end orthocenter_locus_l3283_328355


namespace min_value_theorem_l3283_328348

theorem min_value_theorem (a : ℝ) (h : a > 1) :
  a + 1 / (a - 1) ≥ 3 ∧ (a + 1 / (a - 1) = 3 ↔ a = 2) := by
sorry

end min_value_theorem_l3283_328348


namespace horse_sale_problem_l3283_328337

theorem horse_sale_problem (x : ℝ) : 
  (x - x^2 / 100 = 24) → (x = 40 ∨ x = 60) :=
by
  sorry

end horse_sale_problem_l3283_328337


namespace physics_marks_l3283_328306

theorem physics_marks (P C M : ℝ) 
  (total_avg : (P + C + M) / 3 = 85)
  (phys_math_avg : (P + M) / 2 = 90)
  (phys_chem_avg : (P + C) / 2 = 70) :
  P = 65 := by
sorry

end physics_marks_l3283_328306


namespace curve_transformation_l3283_328351

theorem curve_transformation (x y : ℝ) :
  x^2 + y^2 = 1 → 4 * (x/2)^2 + (2*y)^2 / 4 = 1 := by
  sorry

end curve_transformation_l3283_328351


namespace log_sum_equation_l3283_328342

theorem log_sum_equation (x y z : ℝ) (hx : x = 625) (hy : y = 5) (hz : z = 1/25) :
  Real.log x / Real.log 5 + Real.log y / Real.log 5 - Real.log z / Real.log 5 = 7 := by
  sorry

end log_sum_equation_l3283_328342


namespace f_three_l3283_328321

/-- A function satisfying the given property -/
def f (x : ℝ) : ℝ := sorry

/-- The property of the function f -/
axiom f_property (x y : ℝ) : f (x + y) = f x + f y + x * y

/-- The given condition that f(1) = 1 -/
axiom f_one : f 1 = 1

/-- Theorem stating that f(3) = 6 -/
theorem f_three : f 3 = 6 := by sorry

end f_three_l3283_328321


namespace matrix_fourth_power_l3283_328319

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_fourth_power :
  A ^ 4 = !![0, -9; 9, -9] := by sorry

end matrix_fourth_power_l3283_328319


namespace milk_box_width_l3283_328308

/-- Represents a rectangular milk box -/
structure MilkBox where
  length : Real
  width : Real

/-- Calculates the volume of milk removed when lowering the level by a certain height -/
def volumeRemoved (box : MilkBox) (height : Real) : Real :=
  box.length * box.width * height

theorem milk_box_width (box : MilkBox) 
  (h1 : box.length = 50)
  (h2 : volumeRemoved box 0.5 = 4687.5 / 7.5) : 
  box.width = 25 := by
  sorry

#check milk_box_width

end milk_box_width_l3283_328308


namespace wilfred_carrots_l3283_328374

/-- The number of carrots Wilfred ate on Tuesday, Wednesday, and Thursday -/
def carrots_tuesday : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun tuesday wednesday thursday total =>
    tuesday + wednesday + thursday = total

theorem wilfred_carrots :
  ∃ (tuesday : ℕ),
    carrots_tuesday tuesday 6 5 15 ∧ tuesday = 4 :=
by
  sorry

end wilfred_carrots_l3283_328374


namespace brian_drove_200_miles_more_l3283_328396

/-- Represents the driving scenario of Mike, Steve, and Brian --/
structure DrivingScenario where
  t : ℝ  -- Mike's driving time
  s : ℝ  -- Mike's driving speed
  d : ℝ  -- Mike's driving distance
  steve_distance : ℝ  -- Steve's driving distance
  brian_distance : ℝ  -- Brian's driving distance

/-- The conditions of the driving scenario --/
def scenario_conditions (scenario : DrivingScenario) : Prop :=
  scenario.d = scenario.s * scenario.t ∧  -- Mike's distance equation
  scenario.steve_distance = (scenario.s + 6) * (scenario.t + 1.5) ∧  -- Steve's distance equation
  scenario.brian_distance = (scenario.s + 12) * (scenario.t + 3) ∧  -- Brian's distance equation
  scenario.steve_distance = scenario.d + 90  -- Steve drove 90 miles more than Mike

/-- The theorem stating that Brian drove 200 miles more than Mike --/
theorem brian_drove_200_miles_more (scenario : DrivingScenario) 
  (h : scenario_conditions scenario) : 
  scenario.brian_distance = scenario.d + 200 := by
  sorry


end brian_drove_200_miles_more_l3283_328396


namespace binomial_probability_two_successes_l3283_328395

-- Define the parameters of the binomial distribution
def n : ℕ := 6
def p : ℚ := 1/3

-- Define the probability mass function for the binomial distribution
def binomial_pmf (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- State the theorem
theorem binomial_probability_two_successes :
  binomial_pmf 2 = 80/243 := by
  sorry

end binomial_probability_two_successes_l3283_328395


namespace biased_coin_probability_l3283_328349

theorem biased_coin_probability (p : ℝ) : 
  p < (1 : ℝ) / 2 →
  (20 : ℝ) * p^3 * (1 - p)^3 = (5 : ℝ) / 32 →
  p = (1 - Real.sqrt ((32 - 4 * Real.rpow 5 (1/3)) / 8)) / 2 := by
sorry

end biased_coin_probability_l3283_328349


namespace preimage_of_4_neg2_l3283_328318

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

theorem preimage_of_4_neg2 :
  ∃ (p : ℝ × ℝ), f p = (4, -2) ∧ p = (1, 3) :=
sorry

end preimage_of_4_neg2_l3283_328318


namespace mothers_offer_l3283_328369

def bike_cost : ℕ := 600
def maria_savings : ℕ := 120
def maria_earnings : ℕ := 230

theorem mothers_offer :
  bike_cost - (maria_savings + maria_earnings) = 250 :=
by sorry

end mothers_offer_l3283_328369


namespace print_shop_charge_l3283_328304

/-- The charge per color copy at print shop X -/
def charge_X : ℝ := 1.25

/-- The charge per color copy at print shop Y -/
def charge_Y : ℝ := 2.75

/-- The number of color copies -/
def num_copies : ℕ := 60

/-- The difference in total charge between print shops Y and X -/
def charge_difference : ℝ := 90

theorem print_shop_charge : 
  charge_X * num_copies + charge_difference = charge_Y * num_copies := by
  sorry

end print_shop_charge_l3283_328304


namespace f_properties_l3283_328354

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (a / (a^2 - 1)) * (a^x - a^(-x))

theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- 1. f is an increasing odd function on ℝ
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  (∀ x y : ℝ, x < y → f a x < f a y) ∧
  -- 2. For x ∈ (-1, 1), f(1-m) + f(1-m^2) < 0 implies m ∈ (1, √2)
  (∀ m : ℝ, (∀ x : ℝ, -1 < x ∧ x < 1 → f a (1-m) + f a (1-m^2) < 0) → 
    1 < m ∧ m < Real.sqrt 2) ∧
  -- 3. For x ∈ (-∞, 2), f(x) - 4 < 0 implies a ∈ (2 - √3, 2 + √3) \ {1}
  ((∀ x : ℝ, x < 2 → f a x - 4 < 0) → 
    2 - Real.sqrt 3 < a ∧ a < 2 + Real.sqrt 3 ∧ a ≠ 1) :=
by sorry

end f_properties_l3283_328354


namespace cuboid_surface_area_l3283_328344

/-- The surface area of a rectangular parallelepiped with given dimensions -/
theorem cuboid_surface_area (w : ℝ) (h l : ℝ) : 
  w = 4 →
  l = w + 6 →
  h = l + 5 →
  2 * l * w + 2 * l * h + 2 * w * h = 500 := by
  sorry

end cuboid_surface_area_l3283_328344


namespace imaginary_roots_condition_l3283_328334

/-- The quadratic equation kx^2 + mx + k = 0 (where k ≠ 0) has imaginary roots
    if and only if m^2 < 4k^2 -/
theorem imaginary_roots_condition (k m : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, k * x^2 + m * x + k ≠ 0) ↔ m^2 < 4 * k^2 := by sorry

end imaginary_roots_condition_l3283_328334


namespace ceiling_negative_three_point_seven_l3283_328362

theorem ceiling_negative_three_point_seven : ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end ceiling_negative_three_point_seven_l3283_328362


namespace two_by_one_parallelepiped_removals_l3283_328329

/-- Represents a position on the net of a parallelepiped --/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Represents the net of a parallelepiped --/
structure ParallelepipedNet :=
  (squares : List Position)
  (width : ℕ)
  (height : ℕ)

/-- Checks if a given position is valid for removal --/
def is_valid_removal (net : ParallelepipedNet) (pos : Position) : Prop := sorry

/-- Counts the number of valid removal positions --/
def count_valid_removals (net : ParallelepipedNet) : ℕ := sorry

/-- Creates a 2x1 parallelepiped net --/
def create_2x1_net : ParallelepipedNet := sorry

theorem two_by_one_parallelepiped_removals :
  count_valid_removals (create_2x1_net) = 5 := by sorry

end two_by_one_parallelepiped_removals_l3283_328329


namespace compare_powers_l3283_328361

theorem compare_powers : 5^333 < 3^555 ∧ 3^555 < 4^444 := by
  sorry

end compare_powers_l3283_328361


namespace miss_grayson_class_size_l3283_328384

/-- The number of students in Miss Grayson's class -/
def num_students : ℕ := sorry

/-- The amount raised by the students -/
def amount_raised : ℕ := sorry

/-- The cost of the trip -/
def trip_cost : ℕ := sorry

/-- The remaining fund after the trip -/
def remaining_fund : ℕ := sorry

theorem miss_grayson_class_size :
  (amount_raised = num_students * 5) →
  (trip_cost = num_students * 7) →
  (amount_raised - trip_cost = remaining_fund) →
  (remaining_fund = 10) →
  (num_students = 5) := by sorry

end miss_grayson_class_size_l3283_328384


namespace arc_length_120_degrees_l3283_328378

/-- The arc length of a sector in a circle with radius π and central angle 120° -/
theorem arc_length_120_degrees (r : Real) (θ : Real) : 
  r = π → θ = 2 * π / 3 → 2 * π * r * (θ / (2 * π)) = 2 * π^2 / 3 := by
  sorry

#check arc_length_120_degrees

end arc_length_120_degrees_l3283_328378


namespace intersection_equality_implies_a_value_l3283_328398

def M (a : ℝ) : Set ℝ := {x | x - a = 0}
def N (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem intersection_equality_implies_a_value (a : ℝ) :
  M a ∩ N a = N a → a = 0 ∨ a = 1 ∨ a = -1 := by
  sorry

end intersection_equality_implies_a_value_l3283_328398


namespace unique_digit_product_l3283_328389

theorem unique_digit_product (n : ℕ) : n ≤ 9 → (n * (10 * n + n) = 176) ↔ n = 4 := by
  sorry

end unique_digit_product_l3283_328389


namespace tony_school_years_l3283_328310

/-- The number of years Tony spent getting his initial science degree -/
def initial_degree_years : ℕ := 4

/-- The number of additional degrees Tony obtained -/
def additional_degrees : ℕ := 2

/-- The number of years each additional degree took -/
def years_per_additional_degree : ℕ := 4

/-- The number of years Tony spent getting his graduate degree in physics -/
def graduate_degree_years : ℕ := 2

/-- The total number of years Tony spent in school to become an astronaut -/
def total_years : ℕ := initial_degree_years + additional_degrees * years_per_additional_degree + graduate_degree_years

theorem tony_school_years : total_years = 14 := by
  sorry

end tony_school_years_l3283_328310


namespace solution_set_correct_l3283_328316

-- Define the solution set
def solution_set : Set ℝ := {x | x < 0 ∨ x ≥ 1}

-- Define the inequality
def inequality (x : ℝ) : Prop := (1 - x) / x ≤ 0

-- Theorem stating that the solution set is correct
theorem solution_set_correct :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
by sorry

end solution_set_correct_l3283_328316


namespace ammonium_chloride_formed_l3283_328390

-- Define the reaction components
variable (NH3 : ℝ) -- Moles of Ammonia
variable (HCl : ℝ) -- Moles of Hydrochloric acid
variable (NH4Cl : ℝ) -- Moles of Ammonium chloride

-- Define the conditions
axiom ammonia_moles : NH3 = 3
axiom total_product : NH4Cl = 3

-- Theorem to prove
theorem ammonium_chloride_formed : NH4Cl = 3 :=
by sorry

end ammonium_chloride_formed_l3283_328390


namespace inequality_proof_l3283_328332

-- Define the set M
def M : Set ℝ := {x | 0 < |x + 2| - |1 - x| ∧ |x + 2| - |1 - x| < 2}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|a + 1/2 * b| < 3/4) ∧ (|4 * a * b - 1| > 2 * |b - a|) := by
  sorry

end inequality_proof_l3283_328332


namespace power_inequality_l3283_328360

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^b * b^c * c^a ≤ a^a * b^b * c^c := by
  sorry

end power_inequality_l3283_328360


namespace solve_linear_equation_l3283_328311

theorem solve_linear_equation (x : ℝ) (h : 3*x - 4*x + 7*x = 120) : x = 20 := by
  sorry

end solve_linear_equation_l3283_328311


namespace business_value_calculation_l3283_328340

/-- Calculates the total value of a business given partial ownership and sale information. -/
theorem business_value_calculation (total_ownership : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) :
  total_ownership = 2 / 3 →
  sold_fraction = 3 / 4 →
  sale_price = 30000 →
  (total_ownership * sold_fraction * sale_price) / (total_ownership * sold_fraction) = 60000 := by
  sorry

end business_value_calculation_l3283_328340


namespace f_is_quadratic_l3283_328350

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation_in_one_variable (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 + 15x - 7 = 0 -/
def f (x : ℝ) : ℝ := x^2 + 15*x - 7

/-- Theorem: f is a quadratic equation in one variable -/
theorem f_is_quadratic : is_quadratic_equation_in_one_variable f := by
  sorry

end f_is_quadratic_l3283_328350


namespace unique_triangle_arrangement_l3283_328320

/-- Represents the arrangement of numbers in the triangle --/
structure TriangleArrangement where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Checks if the given arrangement is valid according to the problem conditions --/
def is_valid_arrangement (arr : TriangleArrangement) : Prop :=
  -- All numbers are between 6 and 9
  (arr.A ≥ 6 ∧ arr.A ≤ 9) ∧
  (arr.B ≥ 6 ∧ arr.B ≤ 9) ∧
  (arr.C ≥ 6 ∧ arr.C ≤ 9) ∧
  (arr.D ≥ 6 ∧ arr.D ≤ 9) ∧
  -- All numbers are different
  arr.A ≠ arr.B ∧ arr.A ≠ arr.C ∧ arr.A ≠ arr.D ∧
  arr.B ≠ arr.C ∧ arr.B ≠ arr.D ∧
  arr.C ≠ arr.D ∧
  -- Sum of numbers on each side is equal
  arr.A + arr.C + 3 + 4 = 5 + arr.D + 2 + 4 ∧
  5 + 1 + arr.B + arr.A = 5 + arr.D + 2 + 4 ∧
  arr.A + arr.C + 3 + 4 = 5 + 1 + arr.B + arr.A

theorem unique_triangle_arrangement :
  ∃! arr : TriangleArrangement, is_valid_arrangement arr ∧
    arr.A = 6 ∧ arr.B = 8 ∧ arr.C = 7 ∧ arr.D = 9 :=
by sorry

end unique_triangle_arrangement_l3283_328320


namespace equidistant_point_l3283_328377

theorem equidistant_point (x y : ℝ) : 
  let d_y_axis := |x|
  let d_line1 := |x + y - 1| / Real.sqrt 2
  let d_line2 := |y - 3*x| / Real.sqrt 10
  (d_y_axis = d_line1 ∧ d_y_axis = d_line2) → 
  x = 1 / (4 + Real.sqrt 10 - Real.sqrt 2) := by
sorry

end equidistant_point_l3283_328377


namespace construction_team_problem_l3283_328397

/-- Represents the possible solutions for the original number of people in the second group -/
inductive Solution : Type
  | fiftySeven : Solution
  | twentyOne : Solution

/-- Checks if a given number satisfies the conditions of the problem -/
def satisfiesConditions (x : ℕ) : Prop :=
  ∃ (k : ℕ+), 96 - 16 = k * (x + 16) + 6

/-- The theorem stating that the only solutions are 58 and 21 -/
theorem construction_team_problem :
  ∀ x : ℕ, satisfiesConditions x ↔ (x = 58 ∨ x = 21) :=
sorry

end construction_team_problem_l3283_328397


namespace fraction_simplification_l3283_328324

theorem fraction_simplification :
  (1 / 5 + 1 / 7) / ((2 / 3 - 1 / 4) * 2 / 5) = 72 / 35 := by
  sorry

end fraction_simplification_l3283_328324


namespace rectangle_problem_l3283_328347

theorem rectangle_problem (a b k l : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < k) (h4 : 0 < l) :
  (13 * (a + b) = a * k) →
  (13 * (a + b) = b * l) →
  (k > l) →
  (k = 182) ∧ (l = 14) := by
sorry

end rectangle_problem_l3283_328347


namespace ratio_from_mean_ratio_l3283_328359

theorem ratio_from_mean_ratio (x y : ℝ) (h : x > 0 ∧ y > 0) :
  (x + y) / 2 / Real.sqrt (x * y) = 25 / 24 →
  (x / y = 16 / 9 ∨ x / y = 9 / 16) := by
  sorry

end ratio_from_mean_ratio_l3283_328359


namespace cube_preserves_order_l3283_328357

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end cube_preserves_order_l3283_328357


namespace abs_x_squared_plus_abs_x_minus_six_roots_sum_l3283_328391

theorem abs_x_squared_plus_abs_x_minus_six_roots_sum (x : ℝ) :
  (|x|^2 + |x| - 6 = 0) → (∃ a b : ℝ, a + b = 0 ∧ |a|^2 + |a| - 6 = 0 ∧ |b|^2 + |b| - 6 = 0) :=
by sorry

end abs_x_squared_plus_abs_x_minus_six_roots_sum_l3283_328391


namespace min_y_value_l3283_328376

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 30*x + 20*y) :
  ∃ (y_min : ℝ), y_min = 10 - 5 * Real.sqrt 13 ∧ ∀ (y' : ℝ), ∃ (x' : ℝ), x'^2 + y'^2 = 30*x' + 20*y' → y' ≥ y_min :=
sorry

end min_y_value_l3283_328376


namespace fourth_plus_fifth_sum_l3283_328345

/-- A geometric sequence with a negative common ratio satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  q_neg : q < 0
  second_term : a 2 = 1 - a 1
  fourth_term : a 4 = 4 - a 3
  geom_seq : ∀ n : ℕ, a (n + 1) = a n * q

/-- The sum of the fourth and fifth terms of the geometric sequence is -8 -/
theorem fourth_plus_fifth_sum (seq : GeometricSequence) : seq.a 4 + seq.a 5 = -8 := by
  sorry

end fourth_plus_fifth_sum_l3283_328345


namespace six_couples_handshakes_l3283_328380

/-- The number of handshakes exchanged at a gathering of couples -/
def handshakes_at_gathering (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem stating that for 6 couples, the number of handshakes is 54 -/
theorem six_couples_handshakes :
  handshakes_at_gathering 6 = 54 := by
  sorry

end six_couples_handshakes_l3283_328380


namespace image_fixed_point_l3283_328386

variable {S : Type*} [Finite S]

-- Define the set of all functions from S to S
def AllFunctions (S : Type*) := S → S

-- Define the image of a set under a function
def Image (f : S → S) (A : Set S) : Set S := {y | ∃ x ∈ A, f x = y}

-- Main theorem
theorem image_fixed_point
  (f : S → S)
  (h : ∀ g : S → S, g ≠ f → (f ∘ g ∘ f) ≠ (g ∘ f ∘ g)) :
  Image f (Image f (Set.univ : Set S)) = Image f (Set.univ : Set S) :=
sorry

end image_fixed_point_l3283_328386


namespace journey_time_difference_l3283_328379

theorem journey_time_difference 
  (speed : ℝ) 
  (distance1 : ℝ) 
  (distance2 : ℝ) 
  (h1 : speed = 40) 
  (h2 : distance1 = 360) 
  (h3 : distance2 = 400) :
  (distance2 - distance1) / speed * 60 = 60 := by
  sorry

end journey_time_difference_l3283_328379


namespace ipads_sold_l3283_328326

/-- Proves that the number of iPads sold is 20 given the conditions of the problem -/
theorem ipads_sold (iphones : ℕ) (ipads : ℕ) (apple_tvs : ℕ) 
  (iphone_cost : ℝ) (ipad_cost : ℝ) (apple_tv_cost : ℝ) (average_cost : ℝ) :
  iphones = 100 →
  apple_tvs = 80 →
  iphone_cost = 1000 →
  ipad_cost = 900 →
  apple_tv_cost = 200 →
  average_cost = 670 →
  (iphones * iphone_cost + ipads * ipad_cost + apple_tvs * apple_tv_cost) / 
    (iphones + ipads + apple_tvs : ℝ) = average_cost →
  ipads = 20 := by
  sorry

#check ipads_sold

end ipads_sold_l3283_328326


namespace orc_sword_weight_l3283_328352

/-- Given a total weight of swords, number of squads, and orcs per squad,
    calculates the weight each orc must carry. -/
def weight_per_orc (total_weight : ℕ) (num_squads : ℕ) (orcs_per_squad : ℕ) : ℚ :=
  total_weight / (num_squads * orcs_per_squad)

/-- Proves that given 1200 pounds of swords, 10 squads, and 8 orcs per squad,
    each orc must carry 15 pounds of swords. -/
theorem orc_sword_weight :
  weight_per_orc 1200 10 8 = 15 := by
  sorry

#eval weight_per_orc 1200 10 8

end orc_sword_weight_l3283_328352


namespace common_chord_of_circles_l3283_328314

/-- Represents a circle in the 2D plane -/
structure Circle where
  a : ℝ  -- x^2 coefficient
  b : ℝ  -- y^2 coefficient
  c : ℝ  -- x coefficient
  d : ℝ  -- y coefficient
  e : ℝ  -- constant term

/-- Represents a line in the 2D plane -/
structure Line where
  a : ℝ  -- x coefficient
  b : ℝ  -- y coefficient
  c : ℝ  -- constant term

/-- Definition of the first circle -/
def circle1 : Circle := { a := 1, b := 1, c := -2, d := 0, e := -4 }

/-- Definition of the second circle -/
def circle2 : Circle := { a := 1, b := 1, c := 0, d := 2, e := -6 }

/-- The common chord line -/
def commonChord : Line := { a := 1, b := 1, c := -1 }

/-- Theorem: The given line is the common chord of the two circles -/
theorem common_chord_of_circles :
  commonChord = Line.mk 1 1 (-1) ∧
  (∀ x y : ℝ, x + y - 1 = 0 →
    (x^2 + y^2 - 2*x - 4 = 0 ↔ x^2 + y^2 + 2*y - 6 = 0)) :=
by sorry

end common_chord_of_circles_l3283_328314


namespace annual_mischief_convention_handshakes_l3283_328328

/-- The number of handshakes at the Annual Mischief Convention -/
def total_handshakes (num_gremlins num_imps num_friendly_imps : ℕ) : ℕ :=
  let gremlin_handshakes := num_gremlins * (num_gremlins - 1) / 2
  let imp_gremlin_handshakes := num_imps * num_gremlins
  let friendly_imp_handshakes := num_friendly_imps * (num_friendly_imps - 1) / 2
  gremlin_handshakes + imp_gremlin_handshakes + friendly_imp_handshakes

/-- Theorem stating the total number of handshakes at the Annual Mischief Convention -/
theorem annual_mischief_convention_handshakes :
  total_handshakes 30 20 5 = 1045 := by
  sorry

end annual_mischief_convention_handshakes_l3283_328328


namespace product_of_three_numbers_l3283_328364

theorem product_of_three_numbers (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 32) (hac : a * c = 48) (hbc : b * c = 80) :
  a * b * c = 64 * Real.sqrt 30 := by
  sorry

end product_of_three_numbers_l3283_328364


namespace min_value_fraction_l3283_328375

theorem min_value_fraction (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 2011) 
  (hb : 1 ≤ b ∧ b ≤ 2011) 
  (hc : 1 ≤ c ∧ c ≤ 2011) : 
  (a * b + c : ℚ) / (a + b + c) ≥ 2/3 := by
  sorry

end min_value_fraction_l3283_328375


namespace pascal_triangle_45th_number_l3283_328363

theorem pascal_triangle_45th_number (n : ℕ) : n = 46 →
  Nat.choose n 2 = 1035 :=
by sorry

end pascal_triangle_45th_number_l3283_328363


namespace prove_new_average_weight_l3283_328338

def average_weight_problem (num_boys num_girls : ℕ) 
                           (avg_weight_boys avg_weight_girls : ℚ)
                           (lightest_boy_weight lightest_girl_weight : ℚ) : Prop :=
  let total_weight_boys := num_boys * avg_weight_boys
  let total_weight_girls := num_girls * avg_weight_girls
  let remaining_weight_boys := total_weight_boys - lightest_boy_weight
  let remaining_weight_girls := total_weight_girls - lightest_girl_weight
  let total_remaining_weight := remaining_weight_boys + remaining_weight_girls
  let remaining_children := num_boys + num_girls - 2
  let new_average_weight := total_remaining_weight / remaining_children
  new_average_weight = 161.5

theorem prove_new_average_weight : 
  average_weight_problem 8 5 155 125 140 110 := by
  sorry

end prove_new_average_weight_l3283_328338


namespace p_sufficient_but_not_necessary_for_q_l3283_328366

-- Define the propositions
variable (p q r s : Prop)

-- Define the relationships between the propositions
variable (h1 : p → r)  -- p is sufficient for r
variable (h2 : r → s)  -- s is necessary for r
variable (h3 : s → q)  -- q is necessary for s

-- State the theorem
theorem p_sufficient_but_not_necessary_for_q :
  (p → q) ∧ ¬(q → p) :=
sorry

end p_sufficient_but_not_necessary_for_q_l3283_328366


namespace expected_attacked_squares_theorem_l3283_328313

/-- The number of squares on a chessboard -/
def chessboardSize : ℕ := 64

/-- The number of rooks placed on the board -/
def numRooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook -/
def probNotAttacked : ℚ := (49 : ℚ) / 64

/-- The expected number of squares under attack by rooks on a chessboard -/
def expectedAttackedSquares : ℚ :=
  chessboardSize * (1 - probNotAttacked ^ numRooks)

/-- Theorem stating the expected number of squares under attack -/
theorem expected_attacked_squares_theorem :
  expectedAttackedSquares = 64 * (1 - (49/64)^3) :=
sorry

end expected_attacked_squares_theorem_l3283_328313


namespace max_distance_to_line_l3283_328356

theorem max_distance_to_line (a b c : ℝ) (h : a - b - c = 0) :
  ∃ (x y : ℝ), a * x + b * y + c = 0 ∧
  ∀ (x' y' : ℝ), a * x' + b * y' + c = 0 →
  (x' ^ 2 + y' ^ 2 : ℝ) ≤ 2 :=
sorry

end max_distance_to_line_l3283_328356


namespace intersection_implies_a_value_l3283_328327

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {-3} → a = -1 := by
  sorry

end intersection_implies_a_value_l3283_328327


namespace net_population_change_l3283_328325

def population_change (initial : ℝ) : ℝ :=
  initial * (1.2 * 0.9 * 1.3 * 0.85)

theorem net_population_change :
  ∀ initial : ℝ, initial > 0 →
  let final := population_change initial
  let percent_change := (final - initial) / initial * 100
  round percent_change = 51 := by
  sorry

#check net_population_change

end net_population_change_l3283_328325


namespace simplify_sqrt_expression_simplify_rational_expression_l3283_328370

-- Problem 1
theorem simplify_sqrt_expression :
  3 * Real.sqrt 12 - 6 * Real.sqrt (1/3) + Real.sqrt 48 = 8 * Real.sqrt 3 := by sorry

-- Problem 2
theorem simplify_rational_expression (m : ℝ) (h : m^2 + 3*m - 4 = 0) :
  (m - 3) / (3 * m^2 - 6 * m) / (m + 2 - 5 / (m - 2)) = 1 / 12 := by sorry

end simplify_sqrt_expression_simplify_rational_expression_l3283_328370


namespace multiple_of_9_digit_sum_possible_digits_for_multiple_of_9_l3283_328381

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.repr.data.map (λ c => c.toNat - '0'.toNat)
  digits.sum

theorem multiple_of_9_digit_sum (n : ℕ) : is_multiple_of_9 n ↔ is_multiple_of_9 (digit_sum n) := by sorry

theorem possible_digits_for_multiple_of_9 :
  ∀ d : ℕ, d < 10 →
    (is_multiple_of_9 (86300 + d * 10 + 7) ↔ d = 3 ∨ d = 9) := by sorry

end multiple_of_9_digit_sum_possible_digits_for_multiple_of_9_l3283_328381


namespace unique_row_with_29_l3283_328336

def pascal_coeff (n k : ℕ) : ℕ := Nat.choose n k

def contains_29 (row : ℕ) : Prop :=
  ∃ k, k ≤ row ∧ pascal_coeff row k = 29

theorem unique_row_with_29 :
  ∃! row, contains_29 row :=
sorry

end unique_row_with_29_l3283_328336


namespace diophantine_equation_solutions_l3283_328346

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 9*x*y - x^2 - 8*y^2 = 2005 ↔ 
  (x = 63 ∧ y = 58) ∨ (x = 459 ∧ y = 58) ∨ (x = -63 ∧ y = -58) ∨ (x = -459 ∧ y = -58) := by
  sorry

end diophantine_equation_solutions_l3283_328346


namespace trip_time_difference_l3283_328330

/-- Proves that the difference in time between a 600-mile trip and a 540-mile trip,
    when traveling at a constant speed of 60 miles per hour, is 60 minutes. -/
theorem trip_time_difference (speed : ℝ) (distance1 : ℝ) (distance2 : ℝ) 
    (h1 : speed = 60) 
    (h2 : distance1 = 600) 
    (h3 : distance2 = 540) : 
  (distance1 / speed - distance2 / speed) * 60 = 60 := by
  sorry

#check trip_time_difference

end trip_time_difference_l3283_328330


namespace expression_simplification_and_evaluation_l3283_328307

theorem expression_simplification_and_evaluation :
  let a : ℝ := Real.sqrt 3
  let b : ℝ := Real.sqrt 3 - 1
  ((3 * a) / (2 * a - b) - 1) / ((a + b) / (4 * a^2 - b^2)) = 3 * Real.sqrt 3 - 1 := by
  sorry

end expression_simplification_and_evaluation_l3283_328307


namespace distance_between_points_l3283_328300

/-- The distance between points (0, 8) and (6, 0) is 10. -/
theorem distance_between_points : Real.sqrt ((6 - 0)^2 + (0 - 8)^2) = 10 := by
  sorry

end distance_between_points_l3283_328300


namespace janet_action_figures_l3283_328353

def action_figure_count (initial : ℕ) (sold : ℕ) (bought : ℕ) (brother_factor : ℕ) : ℕ :=
  let remaining := initial - sold
  let after_buying := remaining + bought
  let brother_collection := after_buying * brother_factor
  after_buying + brother_collection

theorem janet_action_figures :
  action_figure_count 10 6 4 2 = 24 := by
  sorry

end janet_action_figures_l3283_328353


namespace two_valid_numbers_l3283_328392

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000 ∧ n < 1000000) ∧  -- six-digit number
  (n % 72 = 0) ∧  -- divisible by 72
  (∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = a * 100000 + 2016 * 10 + b)  -- formed by adding digits to 2016

theorem two_valid_numbers :
  {n : ℕ | is_valid_number n} = {920160, 120168} := by sorry

end two_valid_numbers_l3283_328392


namespace fifth_term_is_negative_one_l3283_328388

/-- An arithmetic sequence with specific first four terms -/
def arithmetic_sequence (x y : ℚ) : ℕ → ℚ
  | 0 => x + 2*y
  | 1 => x - y
  | 2 => 2*x*y
  | 3 => x / (2*y)
  | n + 4 => arithmetic_sequence x y 3 + (n + 1) * (arithmetic_sequence x y 1 - arithmetic_sequence x y 0)

/-- The theorem stating that the fifth term of the specific arithmetic sequence is -1 -/
theorem fifth_term_is_negative_one :
  let x : ℚ := 4
  let y : ℚ := 1
  arithmetic_sequence x y 4 = -1 := by sorry

end fifth_term_is_negative_one_l3283_328388


namespace root_ratio_sum_l3283_328372

theorem root_ratio_sum (k₁ k₂ : ℝ) : 
  (∃ a b : ℝ, k₁ * (a^2 - 2*a) + 3*a + 7 = 0 ∧ 
              k₁ * (b^2 - 2*b) + 3*b + 7 = 0 ∧ 
              a/b + b/a = 9/7) →
  (∃ a b : ℝ, k₂ * (a^2 - 2*a) + 3*a + 7 = 0 ∧ 
              k₂ * (b^2 - 2*b) + 3*b + 7 = 0 ∧ 
              a/b + b/a = 9/7) →
  k₁/k₂ + k₂/k₁ = 64/9 := by
sorry

end root_ratio_sum_l3283_328372


namespace linda_savings_l3283_328368

theorem linda_savings (tv_cost : ℝ) (tv_fraction : ℝ) : 
  tv_cost = 300 → tv_fraction = 1/2 → tv_cost / tv_fraction = 600 := by
  sorry

end linda_savings_l3283_328368


namespace magician_works_two_weeks_l3283_328302

/-- Calculates the number of weeks a magician works given their hourly rate, daily hours, and total payment. -/
def magician_weeks_worked (hourly_rate : ℚ) (daily_hours : ℚ) (total_payment : ℚ) : ℚ :=
  total_payment / (hourly_rate * daily_hours * 7)

/-- Theorem stating that a magician charging $60 per hour, working 3 hours per day, and receiving $2520 in total works for 2 weeks. -/
theorem magician_works_two_weeks :
  magician_weeks_worked 60 3 2520 = 2 := by
  sorry

end magician_works_two_weeks_l3283_328302


namespace balls_per_bag_l3283_328312

theorem balls_per_bag (total_balls : ℕ) (num_bags : ℕ) (balls_per_bag : ℕ) : 
  total_balls = 36 → num_bags = 9 → total_balls = num_bags * balls_per_bag → balls_per_bag = 4 := by
  sorry

end balls_per_bag_l3283_328312


namespace range_of_squared_sum_l3283_328367

theorem range_of_squared_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 1) :
  1/2 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 1 := by
sorry

end range_of_squared_sum_l3283_328367


namespace sum_of_divisors_3k_plus_2_multiple_of_3_l3283_328394

/-- The sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- A number is of the form 3k + 2 -/
def is_3k_plus_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k + 2

theorem sum_of_divisors_3k_plus_2_multiple_of_3 (n : ℕ) (h : is_3k_plus_2 n) :
  3 ∣ sum_of_divisors n :=
sorry

end sum_of_divisors_3k_plus_2_multiple_of_3_l3283_328394


namespace cube_diff_of_squares_l3283_328335

theorem cube_diff_of_squares (a : ℕ+) : ∃ x y : ℤ, x^2 - y^2 = (a : ℤ)^3 := by
  sorry

end cube_diff_of_squares_l3283_328335


namespace checkerboard_probability_l3283_328387

/-- The size of one side of the checkerboard -/
def board_size : ℕ := 9

/-- The total number of squares on the checkerboard -/
def total_squares : ℕ := board_size * board_size

/-- The number of squares on the perimeter of the checkerboard -/
def perimeter_squares : ℕ := 4 * board_size - 4

/-- The number of squares not on the perimeter of the checkerboard -/
def non_perimeter_squares : ℕ := total_squares - perimeter_squares

/-- The probability of choosing a square not on the perimeter -/
def prob_non_perimeter : ℚ := non_perimeter_squares / total_squares

theorem checkerboard_probability :
  prob_non_perimeter = 49 / 81 := by sorry

end checkerboard_probability_l3283_328387


namespace sector_radius_l3283_328323

/-- Given a circular sector with area 11.25 cm² and arc length 4.5 cm, 
    the radius of the circle is 5 cm. -/
theorem sector_radius (area : ℝ) (arc_length : ℝ) (radius : ℝ) : 
  area = 11.25 → arc_length = 4.5 → area = (1/2) * radius * arc_length → radius = 5 := by
  sorry

end sector_radius_l3283_328323


namespace simplify_and_evaluate_l3283_328343

theorem simplify_and_evaluate (a : ℚ) (h : a = -1/3) :
  (a + 1) * (a - 1) - a * (a + 3) = 0 := by sorry

end simplify_and_evaluate_l3283_328343


namespace parabola_translation_l3283_328303

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_translation :
  let p : Parabola := { a := 1, b := 2, c := -1 }
  let translated_p := translate p 2 1
  translated_p = { a := 1, b := -2, c := -3 } :=
by sorry

end parabola_translation_l3283_328303


namespace oranges_per_box_l3283_328382

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (h1 : total_oranges = 35) (h2 : num_boxes = 7) :
  total_oranges / num_boxes = 5 := by
  sorry

end oranges_per_box_l3283_328382


namespace solution_set_part1_min_value_condition_l3283_328305

-- Define the function f
def f (x a b : ℝ) : ℝ := 2 * abs (x + a) + abs (3 * x - b)

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 0 ≥ 3 * abs x + 1} = {x : ℝ | x ≥ -1/2 ∨ x ≤ -3/2} := by sorry

-- Part 2
theorem min_value_condition :
  ∀ a b : ℝ, a > 0 → b > 0 → (∀ x : ℝ, f x a b ≥ 2) → (∃ x : ℝ, f x a b = 2) →
  3 * a + b = 3 := by sorry

end solution_set_part1_min_value_condition_l3283_328305


namespace eve_gift_cost_is_135_l3283_328333

/-- The cost of Eve's gifts for her nieces --/
def eve_gift_cost : ℝ :=
  let hand_mitts : ℝ := 14
  let apron : ℝ := 16
  let utensils : ℝ := 10
  let knife : ℝ := 2 * utensils
  let cost_per_niece : ℝ := hand_mitts + apron + utensils + knife
  let total_cost : ℝ := 3 * cost_per_niece
  let discount_rate : ℝ := 0.25
  let discounted_cost : ℝ := total_cost * (1 - discount_rate)
  discounted_cost

theorem eve_gift_cost_is_135 : eve_gift_cost = 135 := by
  sorry

end eve_gift_cost_is_135_l3283_328333


namespace linear_systems_solutions_l3283_328393

theorem linear_systems_solutions :
  -- First system
  (∃ x y : ℝ, x + y = 5 ∧ 4*x - 2*y = 2 ∧ x = 2 ∧ y = 3) ∧
  -- Second system
  (∃ x y : ℝ, 3*x - 2*y = 13 ∧ 4*x + 3*y = 6 ∧ x = 3 ∧ y = -2) :=
by sorry

end linear_systems_solutions_l3283_328393


namespace rancher_animals_count_l3283_328301

theorem rancher_animals_count : ∀ (horses cows total : ℕ),
  cows = 5 * horses →
  cows = 140 →
  total = cows + horses →
  total = 168 :=
by
  sorry

end rancher_animals_count_l3283_328301


namespace solution_for_E_l3283_328309

/-- Definition of the function E --/
def E (a b c : ℚ) : ℚ := a * b^2 + c

/-- Theorem stating that -5/8 is the solution to E(a,3,1) = E(a,5,11) --/
theorem solution_for_E : ∃ a : ℚ, E a 3 1 = E a 5 11 ∧ a = -5/8 := by
  sorry

end solution_for_E_l3283_328309


namespace mean_score_is_215_div_11_l3283_328358

def points : List ℕ := [15, 20, 25, 30]
def players : List ℕ := [5, 3, 2, 1]

theorem mean_score_is_215_div_11 : 
  (List.sum (List.zipWith (· * ·) points players)) / (List.sum players) = 215 / 11 := by
  sorry

end mean_score_is_215_div_11_l3283_328358


namespace grocery_solution_l3283_328322

/-- Represents the grocery shopping problem --/
def grocery_problem (mustard_oil_price : ℝ) (pasta_price : ℝ) (pasta_amount : ℝ) 
  (sauce_price : ℝ) (sauce_amount : ℝ) (initial_money : ℝ) (remaining_money : ℝ) : Prop :=
  ∃ (mustard_oil_amount : ℝ),
    mustard_oil_amount ≥ 0 ∧
    mustard_oil_price > 0 ∧
    pasta_price > 0 ∧
    pasta_amount > 0 ∧
    sauce_price > 0 ∧
    sauce_amount > 0 ∧
    initial_money > 0 ∧
    remaining_money ≥ 0 ∧
    initial_money - remaining_money = 
      mustard_oil_amount * mustard_oil_price + pasta_amount * pasta_price + sauce_amount * sauce_price ∧
    mustard_oil_amount = 2

/-- Theorem stating the solution to the grocery problem --/
theorem grocery_solution : 
  grocery_problem 13 4 3 5 1 50 7 :=
by sorry

end grocery_solution_l3283_328322


namespace third_term_of_specific_series_l3283_328317

/-- Represents an infinite geometric series -/
structure InfiniteGeometricSeries where
  firstTerm : ℝ
  commonRatio : ℝ
  sum : ℝ
  hSum : sum = firstTerm / (1 - commonRatio)
  hRatio : abs commonRatio < 1

/-- The third term of a geometric sequence -/
def thirdTerm (s : InfiniteGeometricSeries) : ℝ :=
  s.firstTerm * s.commonRatio ^ 2

/-- Theorem: In an infinite geometric series with common ratio 1/4 and sum 40, the third term is 15/8 -/
theorem third_term_of_specific_series :
  ∃ s : InfiniteGeometricSeries, 
    s.commonRatio = 1/4 ∧ 
    s.sum = 40 ∧ 
    thirdTerm s = 15/8 := by
  sorry

end third_term_of_specific_series_l3283_328317
