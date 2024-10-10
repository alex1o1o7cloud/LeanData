import Mathlib

namespace least_value_theorem_l3663_366326

theorem least_value_theorem (x y z : ℕ+) (h : 2 * x.val = 5 * y.val ∧ 5 * y.val = 6 * z.val) :
  ∃ n : ℤ, x.val + y.val + n = 26 ∧ ∀ m : ℤ, x.val + y.val + m = 26 → n ≤ m :=
by sorry

end least_value_theorem_l3663_366326


namespace tenth_angle_measure_l3663_366380

/-- The sum of interior angles of a decagon -/
def decagon_angle_sum : ℝ := 1440

/-- The number of angles in a decagon that are 150° -/
def num_150_angles : ℕ := 9

/-- The measure of each of the known angles -/
def known_angle_measure : ℝ := 150

theorem tenth_angle_measure (decagon_sum : ℝ) (num_known : ℕ) (known_measure : ℝ) 
  (h1 : decagon_sum = decagon_angle_sum) 
  (h2 : num_known = num_150_angles) 
  (h3 : known_measure = known_angle_measure) :
  decagon_sum - num_known * known_measure = 90 := by
  sorry

end tenth_angle_measure_l3663_366380


namespace surfers_count_l3663_366390

/-- The number of surfers on Santa Monica beach -/
def santa_monica_surfers : ℕ := 20

/-- The number of surfers on Malibu beach -/
def malibu_surfers : ℕ := 2 * santa_monica_surfers

/-- The total number of surfers on both beaches -/
def total_surfers : ℕ := malibu_surfers + santa_monica_surfers

theorem surfers_count : total_surfers = 60 := by
  sorry

end surfers_count_l3663_366390


namespace sequence_and_sum_properties_l3663_366307

def sequence_a (n : ℕ) : ℤ :=
  4 * n - 25

def sum_S (n : ℕ) : ℤ :=
  n * (sequence_a 1 + sequence_a n) / 2

theorem sequence_and_sum_properties :
  (sequence_a 3 = -13) ∧
  (∀ n > 1, sequence_a n = sequence_a (n - 1) + 4) ∧
  (sequence_a 1 = -21) ∧
  (sequence_a 2 = -17) ∧
  (∀ n, sequence_a n = 4 * n - 25) ∧
  (∀ k, sum_S 6 ≤ sum_S k) ∧
  (sum_S 6 = -66) := by
  sorry

end sequence_and_sum_properties_l3663_366307


namespace bingo_first_column_count_l3663_366322

/-- The number of ways to choose 5 distinct numbers from 1 to 15 -/
def bingo_first_column : ℕ :=
  (15 * 14 * 13 * 12 * 11)

/-- Theorem: The number of distinct possibilities for the first column
    of a MODIFIED SHORT BINGO card is 360360 -/
theorem bingo_first_column_count : bingo_first_column = 360360 := by
  sorry

end bingo_first_column_count_l3663_366322


namespace rectangle_shorter_side_l3663_366395

/-- A rectangle with perimeter 60 feet and area 130 square feet has a shorter side of approximately 5 feet -/
theorem rectangle_shorter_side (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≥ b)
  (h_perimeter : 2*a + 2*b = 60) (h_area : a*b = 130) :
  ∃ ε > 0, abs (b - 5) < ε :=
sorry

end rectangle_shorter_side_l3663_366395


namespace max_pencils_is_seven_l3663_366317

/-- The maximum number of pencils Alice can purchase given the conditions --/
def max_pencils : ℕ :=
  let pin_cost : ℕ := 3
  let pen_cost : ℕ := 4
  let pencil_cost : ℕ := 9
  let total_budget : ℕ := 72
  let min_purchase : ℕ := pin_cost + pen_cost
  let remaining_budget : ℕ := total_budget - min_purchase
  remaining_budget / pencil_cost

/-- Theorem stating that the maximum number of pencils Alice can purchase is 7 --/
theorem max_pencils_is_seven : max_pencils = 7 := by
  sorry

#eval max_pencils -- This will evaluate to 7

end max_pencils_is_seven_l3663_366317


namespace special_pentagon_angles_l3663_366372

/-- A pentagon that is a cross-section of a parallelepiped with side ratio constraints -/
structure SpecialPentagon where
  -- The pentagon is a cross-section of a parallelepiped
  is_cross_section : Bool
  -- The ratio of any two sides is either 1, 2, or 1/2
  side_ratio_constraint : ∀ (s1 s2 : ℝ), s1 > 0 → s2 > 0 → s1 / s2 ∈ ({1, 2, 1/2} : Set ℝ)

/-- The interior angles of the special pentagon -/
def interior_angles (p : SpecialPentagon) : List ℝ := sorry

/-- Theorem stating the interior angles of the special pentagon -/
theorem special_pentagon_angles (p : SpecialPentagon) :
  ∃ (angles : List ℝ), angles = interior_angles p ∧ angles.length = 5 ∧
  (angles.count 120 = 4 ∧ angles.count 60 = 1) := by sorry

end special_pentagon_angles_l3663_366372


namespace gcd_of_powers_of_three_l3663_366369

theorem gcd_of_powers_of_three :
  Nat.gcd (3^1007 - 1) (3^1018 - 1) = 3^11 - 1 := by sorry

end gcd_of_powers_of_three_l3663_366369


namespace intersection_A_complement_B_l3663_366321

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {2, 3, 4}

-- Define set B
def B : Finset Nat := {4, 5}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {2, 3} := by
  sorry

end intersection_A_complement_B_l3663_366321


namespace arc_square_region_area_coefficients_sum_l3663_366328

/-- Represents a circular arc --/
structure CircularArc where
  radius : ℝ
  centralAngle : ℝ

/-- Represents the region formed by three circular arcs and a square --/
structure ArcSquareRegion where
  arcs : Fin 3 → CircularArc
  squareSideLength : ℝ

/-- The area of the region inside the arcs but outside the square --/
noncomputable def regionArea (r : ArcSquareRegion) : ℝ :=
  sorry

/-- Coefficients of the area expression a√b + cπ - d --/
structure AreaCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem arc_square_region_area_coefficients_sum :
  ∀ r : ArcSquareRegion,
  (∀ i : Fin 3, (r.arcs i).radius = 6 ∧ (r.arcs i).centralAngle = 45 * π / 180) →
  r.squareSideLength = 12 →
  ∃ coeff : AreaCoefficients,
    regionArea r = coeff.c * π - coeff.d ∧
    coeff.a + coeff.b + coeff.c + coeff.d = 174 :=
sorry

end arc_square_region_area_coefficients_sum_l3663_366328


namespace circle_area_with_diameter_10_l3663_366383

theorem circle_area_with_diameter_10 (π : ℝ) :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end circle_area_with_diameter_10_l3663_366383


namespace square_of_complex_is_real_implies_m_is_plus_minus_one_l3663_366316

theorem square_of_complex_is_real_implies_m_is_plus_minus_one (m : ℝ) :
  (∃ (r : ℝ), (m + Complex.I)^2 = r) → (m = 1 ∨ m = -1) := by
  sorry

end square_of_complex_is_real_implies_m_is_plus_minus_one_l3663_366316


namespace right_triangle_hypotenuse_l3663_366362

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 :=
by
  sorry

end right_triangle_hypotenuse_l3663_366362


namespace base_k_equivalence_l3663_366337

/-- 
Given a natural number k, this function converts a number from base k to decimal.
The input is a list of digits in reverse order (least significant digit first).
-/
def baseKToDecimal (k : ℕ) (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * k^i) 0

/-- 
This theorem states that if 26 in decimal is equal to 32 in base-k, then k must be 8.
-/
theorem base_k_equivalence :
  ∀ k : ℕ, k > 1 → baseKToDecimal k [2, 3] = 26 → k = 8 := by
  sorry

end base_k_equivalence_l3663_366337


namespace division_problem_l3663_366367

theorem division_problem (n : ℕ) : n / 4 = 5 ∧ n % 4 = 3 → n = 23 := by
  sorry

end division_problem_l3663_366367


namespace champion_determination_races_l3663_366386

/-- The number of races needed to determine a champion sprinter -/
def races_needed (initial_sprinters : ℕ) (sprinters_per_race : ℕ) (eliminated_per_race : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 50 races are needed for the given conditions -/
theorem champion_determination_races :
  races_needed 400 10 8 = 50 := by sorry

end champion_determination_races_l3663_366386


namespace no_such_function_l3663_366368

theorem no_such_function : ¬∃ f : ℝ → ℝ, f 0 > 0 ∧ ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x) := by
  sorry

end no_such_function_l3663_366368


namespace complement_intersection_equals_set_l3663_366365

def U : Set Nat := {0, 1, 2, 3}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {1, 2, 3}

theorem complement_intersection_equals_set : (U \ (M ∩ N)) = {0, 3} := by
  sorry

end complement_intersection_equals_set_l3663_366365


namespace simplify_sqrt_450_l3663_366329

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l3663_366329


namespace smallest_integer_above_sum_of_roots_l3663_366374

theorem smallest_integer_above_sum_of_roots : ∃ n : ℕ, n = 2703 ∧ 
  (∀ m : ℕ, (m : ℝ) > (Real.sqrt 4 + Real.sqrt 3)^6 → m ≥ n) ∧
  ((n : ℝ) - 1 ≤ (Real.sqrt 4 + Real.sqrt 3)^6) := by
  sorry

end smallest_integer_above_sum_of_roots_l3663_366374


namespace inverse_variation_solution_l3663_366335

/-- Inverse variation relation between three quantities -/
def inverse_variation (r s t : ℝ) : Prop :=
  ∃ (k₁ k₂ : ℝ), k₁ ≠ 0 ∧ k₂ ≠ 0 ∧ r * s = k₁ ∧ r * t = k₂

theorem inverse_variation_solution (r₁ s₁ t₁ r₂ s₂ t₂ : ℝ) :
  inverse_variation r₁ s₁ t₁ →
  inverse_variation r₂ s₂ t₂ →
  r₁ = 1500 →
  s₁ = 0.25 →
  t₁ = 0.5 →
  r₂ = 3000 →
  s₂ = 0.125 ∧ t₂ = 0.25 := by
  sorry


end inverse_variation_solution_l3663_366335


namespace consistency_comparison_l3663_366344

/-- Represents a student's performance in a series of games -/
structure StudentPerformance where
  numGames : ℕ
  avgScore : ℝ
  stdDev : ℝ

/-- Defines what it means for a student to perform more consistently -/
def MoreConsistent (a b : StudentPerformance) : Prop :=
  a.numGames = b.numGames ∧ a.avgScore = b.avgScore ∧ a.stdDev < b.stdDev

theorem consistency_comparison (a b : StudentPerformance) 
  (h1 : a.numGames = b.numGames) 
  (h2 : a.avgScore = b.avgScore) 
  (h3 : a.stdDev < b.stdDev) : 
  MoreConsistent a b :=
sorry

end consistency_comparison_l3663_366344


namespace base6_multiplication_l3663_366325

/-- Converts a base-6 number to base-10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base-10 number to base-6 --/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

theorem base6_multiplication :
  toBase6 (toBase10 [6] * toBase10 [1, 2]) = [2, 1, 0] := by sorry

end base6_multiplication_l3663_366325


namespace company_employees_l3663_366343

theorem company_employees (total : ℕ) 
  (h1 : (60 : ℚ) / 100 * total = (total : ℚ).floor)
  (h2 : (20 : ℚ) / 100 * total = ((40 : ℚ) / 100 * total).floor / 2)
  (h3 : (60 : ℚ) / 100 * total = (20 : ℚ) / 100 * total + 40) :
  total = 100 := by
sorry

end company_employees_l3663_366343


namespace hcd_4760_280_minus_12_l3663_366309

theorem hcd_4760_280_minus_12 : Nat.gcd 4760 280 - 12 = 268 := by
  sorry

end hcd_4760_280_minus_12_l3663_366309


namespace quadratic_equation_solutions_l3663_366308

theorem quadratic_equation_solutions :
  let equation := fun y : ℝ => 3 * y * (y - 1) = 2 * (y - 1)
  (equation (2/3) ∧ equation 1) ∧
  ∀ y : ℝ, equation y → (y = 2/3 ∨ y = 1) :=
by sorry

end quadratic_equation_solutions_l3663_366308


namespace stadium_seats_count_l3663_366311

/-- The number of seats in a stadium is equal to the sum of occupied and empty seats -/
theorem stadium_seats_count
  (children : ℕ)
  (adults : ℕ)
  (empty_seats : ℕ)
  (h1 : children = 52)
  (h2 : adults = 29)
  (h3 : empty_seats = 14) :
  children + adults + empty_seats = 95 := by
  sorry

#check stadium_seats_count

end stadium_seats_count_l3663_366311


namespace green_ball_probability_l3663_366312

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- Calculates the total probability of selecting a green ball from three containers -/
def totalGreenProbability (c1 c2 c3 : Container) : ℚ :=
  (1 / 3) * (greenProbability c1 + greenProbability c2 + greenProbability c3)

/-- Theorem stating the probability of selecting a green ball -/
theorem green_ball_probability :
  let c1 : Container := ⟨8, 4⟩
  let c2 : Container := ⟨2, 4⟩
  let c3 : Container := ⟨2, 6⟩
  totalGreenProbability c1 c2 c3 = 7 / 12 := by
  sorry


end green_ball_probability_l3663_366312


namespace a_and_b_know_own_results_a_and_b_dont_know_each_others_results_l3663_366318

-- Define the possible results
inductive Result
| Excellent
| Good

-- Define the students
inductive Student
| A
| B
| C
| D

-- Define the function that assigns results to students
def result : Student → Result := sorry

-- Define the knowledge state of each student
structure Knowledge where
  knows_b : Bool
  knows_c : Bool
  knows_d : Bool

-- Define the initial knowledge state
def initial_knowledge : Student → Knowledge
| Student.A => { knows_b := false, knows_c := false, knows_d := true }
| Student.B => { knows_b := false, knows_c := true,  knows_d := false }
| Student.C => { knows_b := false, knows_c := false, knows_d := false }
| Student.D => { knows_b := true,  knows_c := true,  knows_d := false }

-- Theorem stating that A and B can know their own results
theorem a_and_b_know_own_results :
  (∃ (s₁ s₂ : Student), result s₁ = Result.Excellent ∧ result s₂ = Result.Excellent) ∧
  (∃ (s₃ s₄ : Student), result s₃ = Result.Good ∧ result s₄ = Result.Good) ∧
  (result Student.B ≠ result Student.C) ∧
  (¬ (result Student.D = Result.Excellent ∧ result Student.B = Result.Excellent ∧ result Student.C = Result.Good)) ∧
  (¬ (result Student.D = Result.Excellent ∧ result Student.B = Result.Good ∧ result Student.C = Result.Excellent)) ∧
  (¬ (result Student.D = Result.Good ∧ result Student.B = Result.Good ∧ result Student.C = Result.Excellent)) ∧
  (¬ (result Student.D = Result.Good ∧ result Student.B = Result.Excellent ∧ result Student.C = Result.Good)) →
  (∃ (f : Student → Result),
    (f Student.A = result Student.A) ∧
    (f Student.B = result Student.B) ∧
    (f Student.C ≠ result Student.C ∨ f Student.D ≠ result Student.D)) :=
sorry

-- Theorem stating that A and B cannot know each other's results
theorem a_and_b_dont_know_each_others_results :
  (∃ (s₁ s₂ : Student), result s₁ = Result.Excellent ∧ result s₂ = Result.Excellent) ∧
  (∃ (s₃ s₄ : Student), result s₃ = Result.Good ∧ result s₄ = Result.Good) ∧
  (result Student.B ≠ result Student.C) ∧
  (¬ (result Student.D = Result.Excellent ∧ result Student.B = Result.Excellent ∧ result Student.C = Result.Good)) ∧
  (¬ (result Student.D = Result.Excellent ∧ result Student.B = Result.Good ∧ result Student.C = Result.Excellent)) ∧
  (¬ (result Student.D = Result.Good ∧ result Student.B = Result.Good ∧ result Student.C = Result.Excellent)) ∧
  (¬ (result Student.D = Result.Good ∧ result Student.B = Result.Excellent ∧ result Student.C = Result.Good)) →
  ¬(∃ (f : Student → Result),
    (f Student.A = result Student.A) ∧
    (f Student.B = result Student.B) ∧
    (f Student.C = result Student.C) ∧
    (f Student.D = result Student.D)) :=
sorry

end a_and_b_know_own_results_a_and_b_dont_know_each_others_results_l3663_366318


namespace cucumber_equivalent_to_16_apples_l3663_366336

/-- The cost of fruits in an arbitrary unit -/
structure FruitCost where
  apple : ℕ → ℚ
  banana : ℕ → ℚ
  cucumber : ℕ → ℚ

/-- The given conditions about fruit costs -/
def fruit_cost_conditions (c : FruitCost) : Prop :=
  c.apple 8 = c.banana 4 ∧ c.banana 2 = c.cucumber 3

/-- The theorem to prove -/
theorem cucumber_equivalent_to_16_apples (c : FruitCost) 
  (h : fruit_cost_conditions c) : 
  ∃ n : ℕ, c.apple 16 = c.cucumber n ∧ n = 12 := by
  sorry

end cucumber_equivalent_to_16_apples_l3663_366336


namespace base_nine_calculation_l3663_366304

/-- Represents a number in base 9 --/
def BaseNine : Type := Nat

/-- Addition operation for base 9 numbers --/
def add_base_nine : BaseNine → BaseNine → BaseNine
| a, b => sorry

/-- Multiplication operation for base 9 numbers --/
def mul_base_nine : BaseNine → BaseNine → BaseNine
| a, b => sorry

/-- Converts a natural number to its base 9 representation --/
def to_base_nine : Nat → BaseNine
| n => sorry

theorem base_nine_calculation :
  let a : BaseNine := to_base_nine 35
  let b : BaseNine := to_base_nine 273
  let c : BaseNine := to_base_nine 2
  let result : BaseNine := to_base_nine 620
  mul_base_nine (add_base_nine a b) c = result := by sorry

end base_nine_calculation_l3663_366304


namespace imaginary_part_of_z_l3663_366306

theorem imaginary_part_of_z (z : ℂ) (h : z - Complex.I = (4 - 2 * Complex.I) / (1 + 2 * Complex.I)) : 
  z.im = -1 := by
  sorry

end imaginary_part_of_z_l3663_366306


namespace fifty_third_term_is_2_to_53_l3663_366315

def double_sequence : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * double_sequence n

theorem fifty_third_term_is_2_to_53 :
  double_sequence 52 = 2^53 := by
  sorry

end fifty_third_term_is_2_to_53_l3663_366315


namespace arithmetic_simplification_l3663_366338

theorem arithmetic_simplification : 2 - (-3) * 2 - 4 - (-5) - 6 - (-7) * 2 = 17 := by
  sorry

end arithmetic_simplification_l3663_366338


namespace firecracker_explosion_speed_l3663_366332

/-- The speed of a fragment after an explosion, given initial conditions of a firecracker. -/
theorem firecracker_explosion_speed 
  (v₀ : ℝ)           -- Initial upward speed of firecracker
  (t : ℝ)            -- Time of explosion
  (m₁ m₂ : ℝ)        -- Masses of fragments
  (v_small : ℝ)      -- Horizontal speed of smaller fragment after explosion
  (g : ℝ)            -- Acceleration due to gravity
  (h : v₀ = 20)      -- Initial speed is 20 m/s
  (h_t : t = 3)      -- Explosion occurs at 3 seconds
  (h_m : m₂ = 2 * m₁) -- Mass ratio is 1:2
  (h_v : v_small = 16) -- Smaller fragment's horizontal speed is 16 m/s
  (h_g : g = 10)     -- Acceleration due to gravity is 10 m/s^2
  : ∃ v : ℝ, v = 17 ∧ v = 
    Real.sqrt ((2 * m₁ * v_small / (m₁ + m₂))^2 + (v₀ - g * t)^2) :=
by sorry

end firecracker_explosion_speed_l3663_366332


namespace total_cars_produced_l3663_366353

/-- Given that a car company produced 3,884 cars in North America and 2,871 cars in Europe,
    prove that the total number of cars produced is 6,755. -/
theorem total_cars_produced (north_america : ℕ) (europe : ℕ) 
  (h1 : north_america = 3884) (h2 : europe = 2871) : 
  north_america + europe = 6755 := by
  sorry

end total_cars_produced_l3663_366353


namespace specific_hexagon_perimeter_l3663_366394

/-- A hexagon with specific side lengths and right angles -/
structure RightAngleHexagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  FA : ℝ
  right_angles : Bool

/-- The perimeter of a hexagon -/
def perimeter (h : RightAngleHexagon) : ℝ :=
  h.AB + h.BC + h.CD + h.DE + h.EF + h.FA

/-- Theorem: The perimeter of the specific hexagon is 6 -/
theorem specific_hexagon_perimeter :
  ∃ (h : RightAngleHexagon),
    h.AB = 1 ∧ h.BC = 1 ∧ h.CD = 2 ∧ h.DE = 1 ∧ h.EF = 1 ∧ h.right_angles = true ∧
    perimeter h = 6 := by
  sorry

end specific_hexagon_perimeter_l3663_366394


namespace polar_line_theorem_l3663_366384

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  rho : ℝ
  theta : ℝ

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on a line in polar coordinates -/
def lies_on (p : PolarPoint) (l : PolarLine) : Prop :=
  l.equation p.rho p.theta

/-- Checks if a line is parallel to the polar axis -/
def parallel_to_polar_axis (l : PolarLine) : Prop :=
  ∀ (rho theta : ℝ), l.equation rho theta ↔ l.equation rho 0

theorem polar_line_theorem (p : PolarPoint) (l : PolarLine) 
  (h1 : p.rho = 2 ∧ p.theta = π/3)
  (h2 : lies_on p l)
  (h3 : parallel_to_polar_axis l) :
  ∀ (rho theta : ℝ), l.equation rho theta ↔ rho * Real.sin theta = Real.sqrt 3 := by
  sorry

end polar_line_theorem_l3663_366384


namespace football_team_throwers_l3663_366396

/-- Proves the number of throwers on a football team given specific conditions -/
theorem football_team_throwers 
  (total_players : ℕ) 
  (total_right_handed : ℕ) 
  (h_total : total_players = 70)
  (h_right : total_right_handed = 62)
  (h_throwers_right : ∀ t : ℕ, t ≤ total_players → t ≤ total_right_handed)
  (h_non_throwers_division : ∀ n : ℕ, n < total_players → 
    3 * (total_players - n) = 2 * (total_right_handed - n) + (total_players - total_right_handed)) :
  ∃ throwers : ℕ, throwers = 46 ∧ throwers ≤ total_players ∧ throwers ≤ total_right_handed :=
sorry

end football_team_throwers_l3663_366396


namespace tangent_slope_at_A_l3663_366331

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + x

-- Define the point A
def point_A : ℝ × ℝ := (2, 6)

-- Theorem statement
theorem tangent_slope_at_A :
  (deriv f) point_A.1 = 5 := by sorry

end tangent_slope_at_A_l3663_366331


namespace fair_coin_tails_probability_l3663_366339

-- Define a fair coin
def FairCoin : Type := Unit

-- Define the possible outcomes of a coin flip
inductive CoinOutcome : Type
| Heads : CoinOutcome
| Tails : CoinOutcome

-- Define the probability of getting tails for a fair coin
def probTails (coin : FairCoin) : ℚ := 1 / 2

-- Theorem statement
theorem fair_coin_tails_probability (coin : FairCoin) (previous_flips : List CoinOutcome) :
  probTails coin = 1 / 2 := by
  sorry

end fair_coin_tails_probability_l3663_366339


namespace town_x_employment_l3663_366303

structure TownPopulation where
  total_employed : Real
  employed_20_35 : Real
  employed_36_50 : Real
  employed_51_65 : Real
  employed_males : Real
  males_high_school : Real
  males_college : Real
  males_postgrad : Real

def employed_females (pop : TownPopulation) : Real :=
  pop.total_employed - pop.employed_males

theorem town_x_employment (pop : TownPopulation)
  (h1 : pop.total_employed = 0.96)
  (h2 : pop.employed_20_35 = 0.40 * pop.total_employed)
  (h3 : pop.employed_36_50 = 0.50 * pop.total_employed)
  (h4 : pop.employed_51_65 = 0.10 * pop.total_employed)
  (h5 : pop.employed_males = 0.24)
  (h6 : pop.males_high_school = 0.45 * pop.employed_males)
  (h7 : pop.males_college = 0.35 * pop.employed_males)
  (h8 : pop.males_postgrad = 0.20 * pop.employed_males) :
  let females := employed_females pop
  ∃ (f_20_35 f_36_50 f_51_65 f_high_school f_college f_postgrad : Real),
    f_20_35 = 0.288 ∧
    f_36_50 = 0.36 ∧
    f_51_65 = 0.072 ∧
    f_high_school = 0.324 ∧
    f_college = 0.252 ∧
    f_postgrad = 0.144 ∧
    f_20_35 = 0.40 * females ∧
    f_36_50 = 0.50 * females ∧
    f_51_65 = 0.10 * females ∧
    f_high_school = 0.45 * females ∧
    f_college = 0.35 * females ∧
    f_postgrad = 0.20 * females :=
by sorry

end town_x_employment_l3663_366303


namespace brownie_problem_l3663_366351

theorem brownie_problem (initial_brownies : ℕ) 
  (h1 : initial_brownies = 16) 
  (children_ate_percent : ℚ) 
  (h2 : children_ate_percent = 1/4) 
  (family_ate_percent : ℚ) 
  (h3 : family_ate_percent = 1/2) 
  (lorraine_ate : ℕ) 
  (h4 : lorraine_ate = 1) : 
  initial_brownies - 
  (initial_brownies * children_ate_percent).floor - 
  ((initial_brownies - (initial_brownies * children_ate_percent).floor) * family_ate_percent).floor - 
  lorraine_ate = 5 := by
sorry


end brownie_problem_l3663_366351


namespace sum_equals_zero_l3663_366391

theorem sum_equals_zero (m n p : ℝ) 
  (h1 : m * n + p^2 + 4 = 0) 
  (h2 : m - n = 4) : 
  m + n = 0 := by
sorry

end sum_equals_zero_l3663_366391


namespace f_always_positive_l3663_366313

def f (x : ℝ) : ℝ := x^2 + 3*x + 4

theorem f_always_positive : ∀ x : ℝ, f x > 0 := by
  sorry

end f_always_positive_l3663_366313


namespace shirt_cost_difference_l3663_366373

/-- The difference in cost between two shirts -/
def cost_difference (total_cost first_shirt_cost : ℕ) : ℕ :=
  first_shirt_cost - (total_cost - first_shirt_cost)

/-- Proof that the difference in cost between two shirts is $6 -/
theorem shirt_cost_difference :
  let total_cost : ℕ := 24
  let first_shirt_cost : ℕ := 15
  first_shirt_cost > total_cost - first_shirt_cost →
  cost_difference total_cost first_shirt_cost = 6 := by
  sorry

end shirt_cost_difference_l3663_366373


namespace quadratic_roots_sum_and_product_problem_1_l3663_366376

theorem quadratic_roots_sum_and_product (a b c x₁ x₂ : ℝ) (ha : a ≠ 0) :
  a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 →
  x₁ + x₂ = -b / a ∧ x₁ * x₂ = c / a :=
by sorry

theorem problem_1 (x₁ x₂ : ℝ) :
  5 * x₁^2 + 10 * x₁ - 1 = 0 ∧ 5 * x₂^2 + 10 * x₂ - 1 = 0 →
  x₁ + x₂ = -2 ∧ x₁ * x₂ = -1/5 :=
by sorry

end quadratic_roots_sum_and_product_problem_1_l3663_366376


namespace f_min_value_l3663_366346

def f (x : ℝ) : ℝ := |x - 1| + |x + 4| - 5

theorem f_min_value :
  ∀ x : ℝ, f x ≥ 0 ∧ ∃ y : ℝ, f y = 0 :=
by sorry

end f_min_value_l3663_366346


namespace repeating_decimal_as_fraction_l3663_366341

/-- The repeating decimal 0.5̄10 as a rational number -/
def repeating_decimal : ℚ := 0.5 + 0.01 / (1 - 1/100)

/-- The theorem stating that 0.5̄10 is equal to 101/198 -/
theorem repeating_decimal_as_fraction : repeating_decimal = 101 / 198 := by
  sorry

end repeating_decimal_as_fraction_l3663_366341


namespace total_pencils_l3663_366370

/-- Given that each child has 2 pencils and there are 9 children, 
    prove that the total number of pencils is 18. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 2) (h2 : num_children = 9) : 
  pencils_per_child * num_children = 18 := by
sorry

end total_pencils_l3663_366370


namespace weight_replacement_l3663_366375

theorem weight_replacement (initial_count : Nat) (weight_increase : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 5 →
  new_weight = 105 →
  (new_weight - (initial_count * weight_increase)) = 65 := by
  sorry

end weight_replacement_l3663_366375


namespace min_value_of_expression_l3663_366397

theorem min_value_of_expression (x : ℚ) : (2*x - 5)^2 + 18 ≥ 18 := by
  sorry

end min_value_of_expression_l3663_366397


namespace remaining_walk_time_l3663_366385

theorem remaining_walk_time (total_distance : ℝ) (speed : ℝ) (walked_distance : ℝ) : 
  total_distance = 2.5 → 
  speed = 1 / 20 → 
  walked_distance = 1 → 
  (total_distance - walked_distance) / speed = 30 := by
sorry

end remaining_walk_time_l3663_366385


namespace skittles_shared_l3663_366324

theorem skittles_shared (starting_amount ending_amount : ℕ) 
  (h1 : starting_amount = 76)
  (h2 : ending_amount = 4) :
  starting_amount - ending_amount = 72 := by
  sorry

end skittles_shared_l3663_366324


namespace midpoint_polar_specific_points_l3663_366350

/-- The midpoint of a line segment in polar coordinates --/
def midpoint_polar (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

theorem midpoint_polar_specific_points :
  let A : ℝ × ℝ := (9, π/3)
  let B : ℝ × ℝ := (9, 2*π/3)
  let M := midpoint_polar A.1 A.2 B.1 B.2
  (0 < A.1 ∧ 0 ≤ A.2 ∧ A.2 < 2*π) ∧
  (0 < B.1 ∧ 0 ≤ B.2 ∧ B.2 < 2*π) →
  M = (9 * Real.sqrt 3 / 2, π/2) :=
by sorry

end midpoint_polar_specific_points_l3663_366350


namespace function_non_negative_implies_k_range_l3663_366333

/-- The function f(x) defined in the problem -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + k * x + k + 3

/-- The theorem statement -/
theorem function_non_negative_implies_k_range (k : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f k x ≥ 0) → k ≥ -3/13 := by
  sorry

end function_non_negative_implies_k_range_l3663_366333


namespace share_distribution_l3663_366305

theorem share_distribution (total : ℝ) (maya annie saiji : ℝ) : 
  total = 900 →
  maya = (1/2) * annie →
  annie = (1/2) * saiji →
  total = maya + annie + saiji →
  saiji = 900 * (4/7) :=
by sorry

end share_distribution_l3663_366305


namespace min_value_exponential_sum_l3663_366348

theorem min_value_exponential_sum (a b : ℝ) (h : 2 * a + b = 6) :
  ∃ (min : ℝ), min = 4 * Real.sqrt 2 ∧ ∀ (x y : ℝ), 2 * x + y = 6 → 2^x + Real.sqrt 2^y ≥ min := by
  sorry

end min_value_exponential_sum_l3663_366348


namespace tangent_point_min_value_on_interval_l3663_366334

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem tangent_point (a : ℝ) :
  (∃ x > 0, f a x = 0 ∧ (deriv (f a)) x = 0) → a = 1 / Real.exp 1 :=
sorry

theorem min_value_on_interval (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Icc 1 2, f a x ≥ (if a < Real.log 2 then -a else Real.log 2 - 2 * a)) ∧
  (∃ x ∈ Set.Icc 1 2, f a x = (if a < Real.log 2 then -a else Real.log 2 - 2 * a)) :=
sorry

end tangent_point_min_value_on_interval_l3663_366334


namespace at_least_one_good_part_l3663_366327

theorem at_least_one_good_part (total : ℕ) (good : ℕ) (defective : ℕ) (pick : ℕ) :
  total = 20 →
  good = 16 →
  defective = 4 →
  pick = 3 →
  total = good + defective →
  (Nat.choose total pick) - (Nat.choose defective pick) = 1136 :=
by sorry

end at_least_one_good_part_l3663_366327


namespace smallest_six_digit_negative_congruent_to_5_mod_17_l3663_366371

theorem smallest_six_digit_negative_congruent_to_5_mod_17 :
  ∀ n : ℤ, -999999 ≤ n ∧ n < -99999 ∧ n ≡ 5 [ZMOD 17] → n ≥ -100011 :=
by sorry

end smallest_six_digit_negative_congruent_to_5_mod_17_l3663_366371


namespace route_comparison_l3663_366392

-- Define the circular tram line
structure TramLine where
  circumference : ℝ
  park_zoo : ℝ
  zoo_circus : ℝ
  circus_park : ℝ

-- Define the conditions
def valid_tram_line (t : TramLine) : Prop :=
  t.park_zoo + t.zoo_circus + t.circus_park = t.circumference ∧
  t.park_zoo + t.circus_park = 3 * t.park_zoo ∧
  t.zoo_circus = 2 * (t.park_zoo + t.circus_park)

-- Theorem statement
theorem route_comparison (t : TramLine) (h : valid_tram_line t) :
  t.circus_park = 11 * t.zoo_circus :=
sorry

end route_comparison_l3663_366392


namespace system_solution_l3663_366347

theorem system_solution (x y : ℝ) (eq1 : x + y = 2) (eq2 : 3 * x - y = 8) : x - y = 3 := by
  sorry

end system_solution_l3663_366347


namespace ice_cream_sundaes_l3663_366378

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) :
  Nat.choose n 2 = 28 := by
  sorry

end ice_cream_sundaes_l3663_366378


namespace cyclist_round_trip_time_l3663_366355

/-- Calculates the total time for a cyclist's round trip given specific conditions. -/
theorem cyclist_round_trip_time 
  (total_distance : ℝ)
  (first_segment_distance : ℝ)
  (second_segment_distance : ℝ)
  (first_segment_speed : ℝ)
  (second_segment_speed : ℝ)
  (return_speed : ℝ)
  (h1 : total_distance = first_segment_distance + second_segment_distance)
  (h2 : first_segment_distance = 12)
  (h3 : second_segment_distance = 24)
  (h4 : first_segment_speed = 8)
  (h5 : second_segment_speed = 12)
  (h6 : return_speed = 9)
  : (first_segment_distance / first_segment_speed + 
     second_segment_distance / second_segment_speed + 
     total_distance / return_speed) = 7.5 := by
  sorry

end cyclist_round_trip_time_l3663_366355


namespace volunteer_selection_l3663_366387

theorem volunteer_selection (n_boys n_girls n_selected : ℕ) 
  (h_boys : n_boys = 4)
  (h_girls : n_girls = 3)
  (h_selected : n_selected = 3) : 
  (Nat.choose n_boys 2 * Nat.choose n_girls 1) + 
  (Nat.choose n_girls 2 * Nat.choose n_boys 1) = 30 :=
sorry

end volunteer_selection_l3663_366387


namespace shaded_area_of_partitioned_isosceles_right_triangle_l3663_366354

theorem shaded_area_of_partitioned_isosceles_right_triangle 
  (leg_length : ℝ) 
  (num_partitions : ℕ) 
  (num_shaded : ℕ) : 
  leg_length = 8 → 
  num_partitions = 16 → 
  num_shaded = 10 → 
  (1 / 2 * leg_length * leg_length) * (num_shaded / num_partitions) = 20 := by
sorry

end shaded_area_of_partitioned_isosceles_right_triangle_l3663_366354


namespace function_composition_difference_l3663_366330

/-- Given functions f and g, prove that f(g(x)) - g(f(x)) = 5/2 for all x. -/
theorem function_composition_difference (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 5 * x - 3
  let g : ℝ → ℝ := λ x ↦ x / 2 + 1
  f (g x) - g (f x) = 5 / 2 := by
  sorry

end function_composition_difference_l3663_366330


namespace point_ordering_l3663_366359

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem point_ordering :
  let y₁ := f (-3)
  let y₂ := f 1
  let y₃ := f (-1/2)
  y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end point_ordering_l3663_366359


namespace equation_represents_hyperbola_l3663_366319

/-- The equation (x-3)^2 = (3y+4)^2 - 75 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
  ∀ (x y : ℝ), (x - 3)^2 = (3*y + 4)^2 - 75 ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0 :=
by sorry

end equation_represents_hyperbola_l3663_366319


namespace complex_equation_solution_l3663_366379

theorem complex_equation_solution : ∃ (z : ℂ), (Complex.I + 1) * z = Complex.abs (2 * Complex.I) ∧ z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l3663_366379


namespace complement_of_M_wrt_U_l3663_366393

open Set

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}

theorem complement_of_M_wrt_U : (U \ M) = {4} := by sorry

end complement_of_M_wrt_U_l3663_366393


namespace det_equation_roots_l3663_366323

/-- The determinant equation has either one or three real roots -/
theorem det_equation_roots (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let det := fun x => x * (x * x + a * a) + c * (b * x + a * b) - b * (a * c - b * x)
  ∃ (n : Fin 2), (n = 0 ∧ (∃! x, det x = d)) ∨ (n = 1 ∧ (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ det x = d ∧ det y = d ∧ det z = d)) :=
by sorry

end det_equation_roots_l3663_366323


namespace championship_outcomes_l3663_366349

theorem championship_outcomes (n : ℕ) (m : ℕ) : 
  n = 5 → m = 3 → n ^ m = 125 := by
  sorry

end championship_outcomes_l3663_366349


namespace scientific_notation_of_34000_l3663_366381

theorem scientific_notation_of_34000 :
  (34000 : ℝ) = 3.4 * (10 : ℝ)^4 := by sorry

end scientific_notation_of_34000_l3663_366381


namespace daycare_toddlers_l3663_366360

/-- Given a day care center with toddlers and infants, prove that under certain conditions, 
    the number of toddlers is 42. -/
theorem daycare_toddlers (T I : ℕ) : 
  T / I = 7 / 3 →  -- Initial ratio of toddlers to infants
  T / (I + 12) = 7 / 5 →  -- New ratio after 12 infants join
  T = 42 := by
  sorry

end daycare_toddlers_l3663_366360


namespace solution_set1_solution_set2_l3663_366361

-- Part 1
def system1 (x : ℝ) : Prop :=
  3 * x - (x - 2) ≥ 6 ∧ x + 1 > (4 * x - 1) / 3

theorem solution_set1 : 
  ∀ x : ℝ, system1 x ↔ 1 ≤ x ∧ x < 4 := by sorry

-- Part 2
def system2 (x : ℝ) : Prop :=
  2 * x + 1 > 0 ∧ x > 2 * x - 5

def is_positive_integer (x : ℝ) : Prop :=
  ∃ n : ℕ, x = n ∧ n > 0

theorem solution_set2 :
  {x : ℝ | system2 x ∧ is_positive_integer x} = {1, 2, 3, 4} := by sorry

end solution_set1_solution_set2_l3663_366361


namespace orthocenter_PQR_l3663_366320

/-- The orthocenter of a triangle PQR in 3D space. -/
def orthocenter (P Q R : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle PQR with given coordinates is (1/2, 13/2, 15/2). -/
theorem orthocenter_PQR :
  let P : ℝ × ℝ × ℝ := (2, 3, 4)
  let Q : ℝ × ℝ × ℝ := (6, 4, 2)
  let R : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter P Q R = (1/2, 13/2, 15/2) := by sorry

end orthocenter_PQR_l3663_366320


namespace sqrt_meaningful_range_l3663_366363

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) → x ≥ 4 := by sorry

end sqrt_meaningful_range_l3663_366363


namespace operation_result_l3663_366377

theorem operation_result (x : ℕ) (h : x = 40) : (((x / 4) * 5) + 10) - 12 = 48 := by
  sorry

end operation_result_l3663_366377


namespace option_C_most_suitable_for_comprehensive_survey_l3663_366356

/-- Represents a survey option -/
inductive SurveyOption
  | A  -- Understanding the sleep time of middle school students nationwide
  | B  -- Understanding the water quality of a river
  | C  -- Surveying the vision of all classmates
  | D  -- Understanding the service life of a batch of light bulbs

/-- Defines what makes a survey comprehensive -/
def isComprehensive (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.C => true
  | _ => false

/-- Theorem stating that option C is the most suitable for a comprehensive survey -/
theorem option_C_most_suitable_for_comprehensive_survey :
  ∀ (option : SurveyOption), isComprehensive option → option = SurveyOption.C :=
by sorry

end option_C_most_suitable_for_comprehensive_survey_l3663_366356


namespace wedding_attendance_l3663_366301

/-- The number of people Laura invited to her wedding. -/
def invited : ℕ := 220

/-- The percentage of people who typically don't show up. -/
def no_show_percentage : ℚ := 5 / 100

/-- The number of people expected to attend Laura's wedding. -/
def expected_attendance : ℕ := 209

/-- Proves that the expected attendance at Laura's wedding is 209 people. -/
theorem wedding_attendance : 
  (invited : ℚ) * (1 - no_show_percentage) = expected_attendance := by
  sorry

end wedding_attendance_l3663_366301


namespace g_values_l3663_366340

/-- The real-valued function f -/
def f (x : ℝ) : ℝ := (x - 3) * (x + 4)

/-- The complex-valued function g -/
def g (x y : ℝ) : ℂ := (f (2 * x + 3) : ℂ) + Complex.I * y

/-- Theorem stating the values of g(29,k) for k = 1, 2, 3 -/
theorem g_values : ∀ k ∈ ({1, 2, 3} : Set ℕ), g 29 k = (858 : ℂ) + k * Complex.I :=
sorry

end g_values_l3663_366340


namespace linear_function_properties_l3663_366358

/-- A linear function y = ax + b satisfying specific conditions -/
def LinearFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x + b

theorem linear_function_properties (a b : ℝ) :
  (LinearFunction a b 1 = 1 ∧ LinearFunction a b 2 = -5) →
  (a = -6 ∧ b = 7 ∧
   LinearFunction a b 0 = 7 ∧
   ∀ x, LinearFunction a b x > 0 ↔ x < 7/6) :=
by sorry

end linear_function_properties_l3663_366358


namespace parallelogram_area_l3663_366357

/-- Proves that a parallelogram with base 16 cm and where 2 times the sum of its base and height is 56, has an area of 192 square centimeters. -/
theorem parallelogram_area (b h : ℝ) : 
  b = 16 → 2 * (b + h) = 56 → b * h = 192 := by
  sorry

end parallelogram_area_l3663_366357


namespace no_solution_for_inequality_l3663_366398

theorem no_solution_for_inequality :
  ¬ ∃ x : ℝ, 3 * x^2 + 9 * x ≤ -12 := by sorry

end no_solution_for_inequality_l3663_366398


namespace apples_distribution_l3663_366310

/-- The number of people who received apples -/
def num_people (total_apples : ℕ) (apples_per_person : ℚ) : ℚ :=
  total_apples / apples_per_person

/-- Proof that 3 people received apples -/
theorem apples_distribution (total_apples : ℕ) (apples_per_person : ℚ) 
  (h1 : total_apples = 45)
  (h2 : apples_per_person = 15.0) : 
  num_people total_apples apples_per_person = 3 := by
  sorry


end apples_distribution_l3663_366310


namespace seed_germination_requires_water_l3663_366300

-- Define a seed
structure Seed where
  water_content : ℝ
  germinated : Bool

-- Define the germination process
def germinate (s : Seed) : Prop :=
  s.germinated = true

-- Theorem: A seed cannot germinate without water
theorem seed_germination_requires_water (s : Seed) :
  germinate s → s.water_content > 0 :=
by
  sorry


end seed_germination_requires_water_l3663_366300


namespace max_distance_between_vectors_l3663_366352

theorem max_distance_between_vectors (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  (∀ a b : ℝ × ℝ, a = (x, y) ∧ b = (1, 2) → 
    ‖a - b‖ ≤ Real.sqrt 5 + 1) ∧
  (∃ a b : ℝ × ℝ, a = (x, y) ∧ b = (1, 2) ∧ 
    ‖a - b‖ = Real.sqrt 5 + 1) := by
  sorry

end max_distance_between_vectors_l3663_366352


namespace complex_equation_solution_l3663_366314

theorem complex_equation_solution (a b : ℝ) :
  (1 + 2*I : ℂ)*a + b = 2*I → a = 1 ∧ b = -1 := by
  sorry

end complex_equation_solution_l3663_366314


namespace sphere_cube_volume_ratio_l3663_366302

/-- Given a cube with its vertices on a spherical surface, 
    the ratio of the sphere's volume to the cube's volume is √3π/2 -/
theorem sphere_cube_volume_ratio : 
  ∀ (cube_edge : ℝ) (sphere_radius : ℝ),
  cube_edge > 0 →
  sphere_radius > 0 →
  sphere_radius = cube_edge * (Real.sqrt 3) / 2 →
  (4 / 3 * Real.pi * sphere_radius^3) / cube_edge^3 = Real.sqrt 3 * Real.pi / 2 :=
by sorry


end sphere_cube_volume_ratio_l3663_366302


namespace class_size_l3663_366366

theorem class_size (total_budget : ℕ) (souvenir_cost : ℕ) (remaining : ℕ) : 
  total_budget = 730 →
  souvenir_cost = 17 →
  remaining = 16 →
  (total_budget - remaining) / souvenir_cost = 42 :=
by
  sorry

end class_size_l3663_366366


namespace line_intersects_parabola_once_l3663_366345

/-- The line x = k intersects the parabola x = -2y^2 - 3y + 5 at exactly one point if and only if k = 49/8 -/
theorem line_intersects_parabola_once (k : ℝ) : 
  (∃! y : ℝ, k = -2 * y^2 - 3 * y + 5) ↔ k = 49/8 := by
  sorry

end line_intersects_parabola_once_l3663_366345


namespace sector_area_l3663_366388

/-- The area of a circular sector with central angle π/3 and radius 4 is 8π/3 -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 4) :
  (1 / 2) * r * r * θ = (8 * π) / 3 := by
  sorry

end sector_area_l3663_366388


namespace arithmetic_geometric_mean_inequality_l3663_366342

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a * b ≤ ((a + b) / 2) * Real.sqrt (a * b) := by
  sorry

end arithmetic_geometric_mean_inequality_l3663_366342


namespace g_of_3_equals_12_l3663_366382

def g (x : ℝ) : ℝ := x^3 - 2*x^2 + x

theorem g_of_3_equals_12 : g 3 = 12 := by sorry

end g_of_3_equals_12_l3663_366382


namespace lines_parallel_to_same_line_are_parallel_l3663_366364

-- Define a type for lines
def Line := Type

-- Define a relation for parallel lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem lines_parallel_to_same_line_are_parallel 
  (l1 l2 l3 : Line) : 
  Parallel l1 l3 → Parallel l2 l3 → Parallel l1 l2 := by sorry

end lines_parallel_to_same_line_are_parallel_l3663_366364


namespace geometric_sequence_sum_l3663_366399

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  q > 1 →  -- common ratio > 1
  4 * (a 2010)^2 - 8 * (a 2010) + 3 = 0 →  -- a_2010 is a root
  4 * (a 2011)^2 - 8 * (a 2011) + 3 = 0 →  -- a_2011 is a root
  a 2012 + a 2013 = 18 :=
by sorry

end geometric_sequence_sum_l3663_366399


namespace fence_rods_count_l3663_366389

/-- Calculates the total number of metal rods needed for a fence --/
def total_rods (panels : ℕ) (sheets_per_panel : ℕ) (beams_per_panel : ℕ) 
                (rods_per_sheet : ℕ) (rods_per_beam : ℕ) : ℕ :=
  panels * (sheets_per_panel * rods_per_sheet + beams_per_panel * rods_per_beam)

/-- Proves that the total number of metal rods needed for the fence is 380 --/
theorem fence_rods_count : total_rods 10 3 2 10 4 = 380 := by
  sorry

end fence_rods_count_l3663_366389
