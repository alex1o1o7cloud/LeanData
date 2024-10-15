import Mathlib

namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1783_178305

theorem simplify_and_evaluate : 
  let x : ℚ := 1/2
  5 * x^2 - (x^2 - 2*(2*x - 3)) = -3 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1783_178305


namespace NUMINAMATH_CALUDE_laptop_price_after_discount_l1783_178395

/-- Calculates the final price of a laptop after a percentage discount --/
def final_price (original_price : ℕ) (discount_percent : ℕ) : ℕ :=
  original_price - (original_price * discount_percent / 100)

/-- Theorem: The final price of a laptop originally costing $800 with a 15% discount is $680 --/
theorem laptop_price_after_discount :
  final_price 800 15 = 680 := by
  sorry

end NUMINAMATH_CALUDE_laptop_price_after_discount_l1783_178395


namespace NUMINAMATH_CALUDE_tangent_length_to_circle_l1783_178323

/-- The length of the tangent from a point to a circle -/
theorem tangent_length_to_circle (x y : ℝ) : 
  let p : ℝ × ℝ := (2, 3)
  let center : ℝ × ℝ := (1, 1)
  let radius : ℝ := 1
  let dist_squared : ℝ := (p.1 - center.1)^2 + (p.2 - center.2)^2
  (x - 1)^2 + (y - 1)^2 = 1 →  -- Circle equation
  dist_squared > radius^2 →    -- P is outside the circle
  Real.sqrt (dist_squared - radius^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_length_to_circle_l1783_178323


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_holes_value_l1783_178307

/-- The surface area of a sphere with diameter 10 inches, after drilling three holes each with a radius of 0.5 inches -/
def sphere_surface_area_with_holes : ℝ := sorry

/-- The diameter of the bowling ball in inches -/
def ball_diameter : ℝ := 10

/-- The number of finger holes -/
def num_holes : ℕ := 3

/-- The radius of each finger hole in inches -/
def hole_radius : ℝ := 0.5

theorem sphere_surface_area_with_holes_value :
  sphere_surface_area_with_holes = (197 / 2) * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_with_holes_value_l1783_178307


namespace NUMINAMATH_CALUDE_no_real_roots_iff_k_gt_two_l1783_178326

theorem no_real_roots_iff_k_gt_two (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 2 * x + (1/2) ≠ 0) ↔ k > 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_k_gt_two_l1783_178326


namespace NUMINAMATH_CALUDE_simple_random_for_small_population_l1783_178335

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Other

/-- Determines the appropriate sampling method based on population size -/
def appropriateSamplingMethod (populationSize : ℕ) (sampleSize : ℕ) : SamplingMethod :=
  if populationSize ≤ 10 ∧ sampleSize = 1 then
    SamplingMethod.SimpleRandom
  else
    SamplingMethod.Other

/-- Theorem: For a population of 10 items with 1 item randomly selected,
    the appropriate sampling method is simple random sampling -/
theorem simple_random_for_small_population :
  appropriateSamplingMethod 10 1 = SamplingMethod.SimpleRandom :=
by sorry

end NUMINAMATH_CALUDE_simple_random_for_small_population_l1783_178335


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1783_178363

/-- A function f(x) = ax^2 + bx + 1 that is even and has domain [2a, 1-a] -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The domain of f is [2a, 1-a] -/
def domain (a : ℝ) : Set ℝ := Set.Icc (2 * a) (1 - a)

/-- f is an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem sum_of_coefficients (a b : ℝ) :
  (∃ x, x ∈ domain a) →
  is_even (f a b) →
  a + b = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1783_178363


namespace NUMINAMATH_CALUDE_ratio_equality_l1783_178366

theorem ratio_equality (a b : ℝ) (h1 : 7 * a = 8 * b) (h2 : a * b ≠ 0) :
  (a / 8) / (b / 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1783_178366


namespace NUMINAMATH_CALUDE_intersection_M_N_l1783_178342

def M : Set ℤ := {-1, 1}
def N : Set ℤ := {x | -1 < x ∧ x < 4}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1783_178342


namespace NUMINAMATH_CALUDE_area_between_curves_l1783_178334

theorem area_between_curves : 
  let f (x : ℝ) := Real.exp x
  let g (x : ℝ) := Real.exp (-x)
  let a := 0
  let b := 1
  ∫ x in a..b, (f x - g x) = Real.exp 1 + Real.exp (-1) - 2 := by
  sorry

end NUMINAMATH_CALUDE_area_between_curves_l1783_178334


namespace NUMINAMATH_CALUDE_hide_and_seek_l1783_178337

-- Define the players
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
variable (h1 : Andrew → (Boris ∧ ¬Vasya))
variable (h2 : Boris → (Gena ∨ Denis))
variable (h3 : ¬Vasya → (¬Boris ∧ ¬Denis))
variable (h4 : ¬Andrew → (Boris ∧ ¬Gena))

-- Theorem to prove
theorem hide_and_seek :
  Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena :=
sorry

end NUMINAMATH_CALUDE_hide_and_seek_l1783_178337


namespace NUMINAMATH_CALUDE_repair_labor_hours_l1783_178386

/-- Calculates the number of labor hours given the labor cost per hour, part cost, and total repair cost. -/
def labor_hours (labor_cost_per_hour : ℕ) (part_cost : ℕ) (total_cost : ℕ) : ℕ :=
  (total_cost - part_cost) / labor_cost_per_hour

/-- Proves that given the specified costs, the number of labor hours is 16. -/
theorem repair_labor_hours :
  labor_hours 75 1200 2400 = 16 := by
  sorry

end NUMINAMATH_CALUDE_repair_labor_hours_l1783_178386


namespace NUMINAMATH_CALUDE_symmetry_of_point_l1783_178313

def symmetric_point_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem symmetry_of_point :
  symmetric_point_x_axis (1, 2) = (1, -2) := by
sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l1783_178313


namespace NUMINAMATH_CALUDE_expression_simplification_l1783_178319

theorem expression_simplification (m : ℝ) (h : m^2 - 2*m - 1 = 0) :
  (m + 2) / (2*m^2 - 6*m) / (m + 3 + 5 / (m - 3)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1783_178319


namespace NUMINAMATH_CALUDE_code_cracking_probabilities_l1783_178371

/-- The probability of person i cracking the code -/
def P (i : Fin 3) : ℚ :=
  match i with
  | 0 => 1/5
  | 1 => 1/4
  | 2 => 1/3

/-- The probability that exactly two people crack the code -/
def prob_two_crack : ℚ :=
  P 0 * P 1 * (1 - P 2) + P 0 * (1 - P 1) * P 2 + (1 - P 0) * P 1 * P 2

/-- The probability that no one cracks the code -/
def prob_none_crack : ℚ :=
  (1 - P 0) * (1 - P 1) * (1 - P 2)

theorem code_cracking_probabilities :
  prob_two_crack = 3/20 ∧ 
  (1 - prob_none_crack) > prob_none_crack := by
  sorry


end NUMINAMATH_CALUDE_code_cracking_probabilities_l1783_178371


namespace NUMINAMATH_CALUDE_BC_time_proof_l1783_178367

-- Define work rates for A, B, and C
def A_rate : ℚ := 1 / 4
def B_rate : ℚ := 1 / 12
def AC_rate : ℚ := 1 / 2

-- Define the time taken by B and C together
def BC_time : ℚ := 3

-- Theorem statement
theorem BC_time_proof :
  let C_rate : ℚ := AC_rate - A_rate
  let BC_rate : ℚ := B_rate + C_rate
  BC_time = 1 / BC_rate :=
by sorry

end NUMINAMATH_CALUDE_BC_time_proof_l1783_178367


namespace NUMINAMATH_CALUDE_gcd_problem_l1783_178360

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 1729 * k) : 
  Int.gcd (b^2 + 11*b + 28) (b + 5) = 2 := by
sorry

end NUMINAMATH_CALUDE_gcd_problem_l1783_178360


namespace NUMINAMATH_CALUDE_remainder_theorem_l1783_178397

/-- The remainder when x³ - 3x + 5 is divided by x + 2 is 3 -/
theorem remainder_theorem (x : ℝ) : 
  (x^3 - 3*x + 5) % (x + 2) = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1783_178397


namespace NUMINAMATH_CALUDE_committee_formation_count_l1783_178318

def club_size : ℕ := 12
def committee_size : ℕ := 5
def president_count : ℕ := 1

theorem committee_formation_count :
  (club_size.choose committee_size) - ((club_size - president_count).choose committee_size) = 330 :=
by sorry

end NUMINAMATH_CALUDE_committee_formation_count_l1783_178318


namespace NUMINAMATH_CALUDE_pick_school_supply_l1783_178382

/-- The number of pencils in the pencil case -/
def num_pencils : ℕ := 2

/-- The number of erasers in the pencil case -/
def num_erasers : ℕ := 4

/-- The total number of school supplies in the pencil case -/
def total_supplies : ℕ := num_pencils + num_erasers

/-- Theorem stating that the number of ways to pick up a school supply is 6 -/
theorem pick_school_supply : total_supplies = 6 := by
  sorry

end NUMINAMATH_CALUDE_pick_school_supply_l1783_178382


namespace NUMINAMATH_CALUDE_negative_product_cube_squared_l1783_178338

theorem negative_product_cube_squared (a b : ℝ) : (-a * b^3)^2 = a^2 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_product_cube_squared_l1783_178338


namespace NUMINAMATH_CALUDE_last_ball_is_white_l1783_178384

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Black

/-- Represents the state of the box -/
structure BoxState :=
  (white : Nat)
  (black : Nat)

/-- The process of drawing balls and applying rules -/
def drawProcess (state : BoxState) : BoxState :=
  sorry

/-- The final state of the box after the process ends -/
def finalState (initial : BoxState) : BoxState :=
  sorry

/-- Theorem stating that the last ball is always white -/
theorem last_ball_is_white (initial : BoxState) :
  initial.white = 2011 → initial.black = 2012 →
  (finalState initial).white = 1 ∧ (finalState initial).black = 0 :=
sorry

end NUMINAMATH_CALUDE_last_ball_is_white_l1783_178384


namespace NUMINAMATH_CALUDE_phd_team_combinations_setup_correct_l1783_178383

def total_engineers : ℕ := 8
def phd_engineers : ℕ := 3
def ms_bs_engineers : ℕ := 5
def team_size : ℕ := 3

-- Function to calculate combinations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem phd_team_combinations : 
  choose phd_engineers 1 * choose ms_bs_engineers 2 + 
  choose phd_engineers 2 * choose ms_bs_engineers 1 + 
  choose phd_engineers 3 = 46 := by
  sorry

-- Additional theorem to ensure the setup is correct
theorem setup_correct : 
  total_engineers = phd_engineers + ms_bs_engineers ∧ 
  team_size ≤ total_engineers := by
  sorry

end NUMINAMATH_CALUDE_phd_team_combinations_setup_correct_l1783_178383


namespace NUMINAMATH_CALUDE_same_terminal_side_as_405_degrees_l1783_178378

theorem same_terminal_side_as_405_degrees :
  ∀ θ : ℝ, (∃ k : ℤ, θ = k * 360 + 45) ↔ (∃ n : ℤ, θ = 405 + n * 360) :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_as_405_degrees_l1783_178378


namespace NUMINAMATH_CALUDE_triangle_tangent_ratio_l1783_178396

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that if a*cos(B) - b*cos(A) = (3/5)*c, then tan(A) / tan(B) = 4 -/
theorem triangle_tangent_ratio (a b c : ℝ) (A B C : ℝ) 
    (h_triangle : A + B + C = Real.pi)
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
    (h_angles : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi)
    (h_law_of_sines : a / Real.sin A = b / Real.sin B)
    (h_given : a * Real.cos B - b * Real.cos A = (3/5) * c) :
  Real.tan A / Real.tan B = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_ratio_l1783_178396


namespace NUMINAMATH_CALUDE_greatest_integer_not_divisible_by_1111_l1783_178372

theorem greatest_integer_not_divisible_by_1111 :
  (∃ (N : ℕ), N > 0 ∧
    (∃ (x : Fin N → ℤ), ∀ (i j : Fin N), i ≠ j →
      ¬(1111 ∣ (x i)^2 - (x i) * (x j))) ∧
    (∀ (M : ℕ), M > N →
      ¬(∃ (y : Fin M → ℤ), ∀ (i j : Fin M), i ≠ j →
        ¬(1111 ∣ (y i)^2 - (y i) * (y j)))) ∧
  N = 1000) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_not_divisible_by_1111_l1783_178372


namespace NUMINAMATH_CALUDE_cindys_calculation_l1783_178348

theorem cindys_calculation (x : ℤ) (h : (x - 7) / 5 = 51) : (x - 5) / 7 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l1783_178348


namespace NUMINAMATH_CALUDE_expression_simplification_l1783_178370

theorem expression_simplification (y : ℝ) : 
  y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1783_178370


namespace NUMINAMATH_CALUDE_simplify_expression_l1783_178309

theorem simplify_expression (m : ℝ) (h : m < 1) :
  (m - 1) * Real.sqrt (-1 / (m - 1)) = -Real.sqrt (1 - m) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1783_178309


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l1783_178380

theorem smallest_whole_number_above_sum : ℕ := by
  let sum := 3 + 1/3 + 4 + 1/4 + 5 + 1/6 + 6 + 1/7
  have h1 : sum < 19 := by sorry
  have h2 : sum > 18 := by sorry
  exact 19

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l1783_178380


namespace NUMINAMATH_CALUDE_fraction_equality_l1783_178300

theorem fraction_equality : (25 + 15) / (5 - 3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1783_178300


namespace NUMINAMATH_CALUDE_mathland_license_plate_probability_l1783_178379

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The number of possible two-digit numbers --/
def two_digit_numbers : ℕ := 100

/-- The total number of possible license plates in Mathland --/
def total_license_plates : ℕ := alphabet_size * (alphabet_size - 1) * (alphabet_size - 2) * two_digit_numbers

/-- The probability of a specific license plate configuration in Mathland --/
def specific_plate_probability : ℚ := 1 / total_license_plates

theorem mathland_license_plate_probability :
  specific_plate_probability = 1 / 1560000 := by sorry

end NUMINAMATH_CALUDE_mathland_license_plate_probability_l1783_178379


namespace NUMINAMATH_CALUDE_solve_system_for_q_l1783_178357

theorem solve_system_for_q : 
  ∀ p q : ℚ, 3 * p + 4 * q = 8 → 4 * p + 3 * q = 13 → q = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_for_q_l1783_178357


namespace NUMINAMATH_CALUDE_fourth_term_is_five_l1783_178320

/-- An arithmetic sequence where the sum of the third and fifth terms is 10 -/
def ArithmeticSequence (a : ℝ) (d : ℝ) : Prop :=
  a + (a + 2*d) = 10

/-- The fourth term of the arithmetic sequence -/
def FourthTerm (a : ℝ) (d : ℝ) : ℝ := a + d

theorem fourth_term_is_five {a d : ℝ} (h : ArithmeticSequence a d) : FourthTerm a d = 5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_five_l1783_178320


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l1783_178311

-- Define the polyhedron P
structure Polyhedron :=
  (vertices : Finset (Fin 9))
  (edges : Finset (Fin 9 × Fin 9))
  (base : Finset (Fin 7))
  (apex1 : Fin 9)
  (apex2 : Fin 9)

-- Define the coloring of edges
def Coloring (P : Polyhedron) := (Fin 9 × Fin 9) → Bool

-- Define a valid coloring
def ValidColoring (P : Polyhedron) (c : Coloring P) : Prop :=
  ∀ e ∈ P.edges, c e = true ∨ c e = false

-- Define a monochromatic triangle
def MonochromaticTriangle (P : Polyhedron) (c : Coloring P) : Prop :=
  ∃ (v1 v2 v3 : Fin 9), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
    (v1, v2) ∈ P.edges ∧ (v2, v3) ∈ P.edges ∧ (v1, v3) ∈ P.edges ∧
    c (v1, v2) = c (v2, v3) ∧ c (v2, v3) = c (v1, v3)

-- The main theorem
theorem monochromatic_triangle_exists (P : Polyhedron) (c : Coloring P)
    (h_valid : ValidColoring P c) :
    MonochromaticTriangle P c := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l1783_178311


namespace NUMINAMATH_CALUDE_hattie_jumps_l1783_178361

theorem hattie_jumps (H : ℚ) 
  (total_jumps : H + (3/4 * H) + (2/3 * H) + (2/3 * H + 50) = 605) : 
  H = 180 := by
sorry

end NUMINAMATH_CALUDE_hattie_jumps_l1783_178361


namespace NUMINAMATH_CALUDE_rectangle_area_l1783_178341

theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let p := 2 * (l + b)
  p = 112 → l * b = 588 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1783_178341


namespace NUMINAMATH_CALUDE_min_value_ab_l1783_178377

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 4/b = Real.sqrt (a*b)) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 4/y = Real.sqrt (x*y) → a*b ≤ x*y :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l1783_178377


namespace NUMINAMATH_CALUDE_gigi_remaining_pieces_l1783_178304

/-- The number of remaining mushroom pieces after cutting and using some -/
def remaining_pieces (total_mushrooms : ℕ) (pieces_per_mushroom : ℕ) 
  (used_by_kenny : ℕ) (used_by_karla : ℕ) : ℕ :=
  total_mushrooms * pieces_per_mushroom - (used_by_kenny + used_by_karla)

/-- Theorem stating the number of remaining mushroom pieces in GiGi's scenario -/
theorem gigi_remaining_pieces : 
  remaining_pieces 22 4 38 42 = 8 := by sorry

end NUMINAMATH_CALUDE_gigi_remaining_pieces_l1783_178304


namespace NUMINAMATH_CALUDE_external_tangent_lines_of_circles_l1783_178356

-- Define the circles
def circle_A (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 9
def circle_B (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the external common tangent lines
def external_tangent_lines (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x - 3) ∨ y = -(Real.sqrt 3 / 3) * (x - 3)

-- Theorem statement
theorem external_tangent_lines_of_circles :
  ∀ x y : ℝ, (circle_A x y ∨ circle_B x y) → external_tangent_lines x y :=
by
  sorry

end NUMINAMATH_CALUDE_external_tangent_lines_of_circles_l1783_178356


namespace NUMINAMATH_CALUDE_shortest_player_height_l1783_178344

theorem shortest_player_height (tallest_height shortest_height height_difference : ℝ) 
  (h1 : tallest_height = 77.75)
  (h2 : height_difference = 9.5)
  (h3 : tallest_height = shortest_height + height_difference) :
  shortest_height = 68.25 := by
  sorry

end NUMINAMATH_CALUDE_shortest_player_height_l1783_178344


namespace NUMINAMATH_CALUDE_rancher_corn_cost_l1783_178358

/-- Represents the rancher's situation --/
structure RancherSituation where
  sheep : Nat
  cattle : Nat
  grassAcres : Nat
  grassPerCowPerMonth : Nat
  grassPerSheepPerMonth : Nat
  monthsPerBagForCow : Nat
  monthsPerBagForSheep : Nat
  cornBagPrice : Nat

/-- Calculates the yearly cost of feed corn for the rancher --/
def yearlyCornCost (s : RancherSituation) : Nat :=
  let totalGrassPerMonth := s.cattle * s.grassPerCowPerMonth + s.sheep * s.grassPerSheepPerMonth
  let grazingMonths := s.grassAcres / totalGrassPerMonth
  let cornMonths := 12 - grazingMonths
  let cornForSheep := (cornMonths * s.sheep + s.monthsPerBagForSheep - 1) / s.monthsPerBagForSheep
  let cornForCattle := cornMonths * s.cattle / s.monthsPerBagForCow
  (cornForSheep + cornForCattle) * s.cornBagPrice

/-- Theorem stating that the rancher needs to spend $360 on feed corn each year --/
theorem rancher_corn_cost :
  let s : RancherSituation := {
    sheep := 8,
    cattle := 5,
    grassAcres := 144,
    grassPerCowPerMonth := 2,
    grassPerSheepPerMonth := 1,
    monthsPerBagForCow := 1,
    monthsPerBagForSheep := 2,
    cornBagPrice := 10
  }
  yearlyCornCost s = 360 := by sorry

end NUMINAMATH_CALUDE_rancher_corn_cost_l1783_178358


namespace NUMINAMATH_CALUDE_product_value_l1783_178302

theorem product_value (x : ℝ) (h : Real.sqrt (6 + x) + Real.sqrt (21 - x) = 8) :
  (6 + x) * (21 - x) = 1369 / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_value_l1783_178302


namespace NUMINAMATH_CALUDE_total_lost_words_l1783_178310

/-- Represents the number of letters in the language --/
def total_letters : ℕ := 100

/-- Represents the number of forbidden letters --/
def forbidden_letters : ℕ := 6

/-- Calculates the number of lost one-letter words --/
def lost_one_letter_words : ℕ := forbidden_letters

/-- Calculates the number of lost two-letter words with forbidden first letter --/
def lost_two_letter_first : ℕ := forbidden_letters * total_letters

/-- Calculates the number of lost two-letter words with forbidden second letter --/
def lost_two_letter_second : ℕ := total_letters * forbidden_letters

/-- Calculates the number of lost two-letter words with both letters forbidden --/
def lost_two_letter_both : ℕ := forbidden_letters * forbidden_letters

/-- Calculates the total number of lost two-letter words --/
def lost_two_letter_words : ℕ := lost_two_letter_first + lost_two_letter_second - lost_two_letter_both

/-- Theorem stating the total number of lost words --/
theorem total_lost_words :
  lost_one_letter_words + lost_two_letter_words = 1170 := by sorry

end NUMINAMATH_CALUDE_total_lost_words_l1783_178310


namespace NUMINAMATH_CALUDE_maurice_age_l1783_178355

theorem maurice_age (ron_age : ℕ) (maurice_age : ℕ) : 
  ron_age = 43 → 
  ron_age + 5 = 4 * (maurice_age + 5) → 
  maurice_age = 7 := by
sorry

end NUMINAMATH_CALUDE_maurice_age_l1783_178355


namespace NUMINAMATH_CALUDE_fifth_number_in_first_set_l1783_178332

theorem fifth_number_in_first_set (x : ℝ) (fifth_number : ℝ) : 
  ((28 + x + 70 + 88 + fifth_number) / 5 = 67) →
  ((50 + 62 + 97 + 124 + x) / 5 = 75.6) →
  fifth_number = 104 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_in_first_set_l1783_178332


namespace NUMINAMATH_CALUDE_power_equality_l1783_178374

theorem power_equality (p : ℕ) : 81^6 = 3^p → p = 24 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1783_178374


namespace NUMINAMATH_CALUDE_jerrys_breakfast_calories_l1783_178375

/-- Represents the number of pancakes in Jerry's breakfast -/
def num_pancakes : ℕ := 6

/-- Represents the calories per pancake -/
def calories_per_pancake : ℕ := 120

/-- Represents the number of bacon strips in Jerry's breakfast -/
def num_bacon_strips : ℕ := 2

/-- Represents the calories per bacon strip -/
def calories_per_bacon_strip : ℕ := 100

/-- Represents the calories in the bowl of cereal -/
def cereal_calories : ℕ := 200

/-- Theorem stating that the total calories in Jerry's breakfast is 1120 -/
theorem jerrys_breakfast_calories : 
  num_pancakes * calories_per_pancake + 
  num_bacon_strips * calories_per_bacon_strip + 
  cereal_calories = 1120 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_breakfast_calories_l1783_178375


namespace NUMINAMATH_CALUDE_marble_bag_total_l1783_178328

/-- Given a bag of marbles with red:blue:green ratio of 2:4:5 and 40 blue marbles,
    the total number of marbles is 110. -/
theorem marble_bag_total (red blue green total : ℕ) : 
  red + blue + green = total →
  red = 2 * n ∧ blue = 4 * n ∧ green = 5 * n →
  blue = 40 →
  total = 110 := by
  sorry

end NUMINAMATH_CALUDE_marble_bag_total_l1783_178328


namespace NUMINAMATH_CALUDE_trig_identity_l1783_178343

theorem trig_identity : 
  Real.sin (40 * π / 180) * Real.cos (20 * π / 180) - 
  Real.cos (220 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l1783_178343


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l1783_178321

theorem triangle_angle_inequality (α β γ s R : Real) : 
  α > 0 → β > 0 → γ > 0 → 
  α + β + γ = π →
  s > 0 → R > 0 →
  (α + β) * (β + γ) * (γ + α) ≤ 4 * (π / Real.sqrt 3)^3 * R / s := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l1783_178321


namespace NUMINAMATH_CALUDE_unique_score_above_90_l1783_178333

/-- Represents the scoring system for the exam -/
structure ScoringSystem where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ

/-- Calculates the score given the number of correct and incorrect answers -/
def calculate_score (system : ScoringSystem) (correct : ℕ) (incorrect : ℕ) : ℤ :=
  system.correct_points * correct + system.incorrect_points * incorrect

/-- Checks if a score uniquely determines the number of correct and incorrect answers -/
def is_unique_score (system : ScoringSystem) (score : ℤ) : Prop :=
  ∃! (correct incorrect : ℕ),
    correct + incorrect ≤ system.total_questions ∧
    calculate_score system correct incorrect = score

/-- The main theorem to prove -/
theorem unique_score_above_90 (system : ScoringSystem) : 
  system.total_questions = 35 →
  system.correct_points = 5 →
  system.incorrect_points = -2 →
  (∀ s, s > 90 ∧ s < 116 → ¬is_unique_score system s) →
  is_unique_score system 116 := 
by sorry

end NUMINAMATH_CALUDE_unique_score_above_90_l1783_178333


namespace NUMINAMATH_CALUDE_min_cost_theorem_min_cost_value_l1783_178330

def volleyball_price : ℕ := 50
def basketball_price : ℕ := 80

def total_balls : ℕ := 60
def max_cost : ℕ := 3800
def max_volleyballs : ℕ := 38

def cost_function (m : ℕ) : ℕ := volleyball_price * m + basketball_price * (total_balls - m)

theorem min_cost_theorem (m : ℕ) (h1 : m ≤ max_volleyballs) (h2 : cost_function m ≤ max_cost) :
  cost_function max_volleyballs ≤ cost_function m :=
sorry

theorem min_cost_value : cost_function max_volleyballs = 3660 :=
sorry

end NUMINAMATH_CALUDE_min_cost_theorem_min_cost_value_l1783_178330


namespace NUMINAMATH_CALUDE_calculation_problems_l1783_178399

theorem calculation_problems :
  (((1 : ℚ) / 2 - 5 / 9 + 7 / 12) * (-36 : ℚ) = -19) ∧
  ((-199 - 24 / 25) * (5 : ℚ) = -999 - 4 / 5) := by
  sorry

end NUMINAMATH_CALUDE_calculation_problems_l1783_178399


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1783_178350

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem not_sufficient_not_necessary 
  (m : Line) (α β : Plane) 
  (h_perp_planes : perpendicular_planes α β) :
  ¬(∀ m α β, parallel m α → perpendicular m β) ∧ 
  ¬(∀ m α β, perpendicular m β → parallel m α) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1783_178350


namespace NUMINAMATH_CALUDE_complex_multiplication_l1783_178327

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1783_178327


namespace NUMINAMATH_CALUDE_absolute_difference_sum_product_l1783_178301

theorem absolute_difference_sum_product (x y : ℝ) (hx : x = 12) (hy : y = 18) :
  |x - y| * (x + y) = 180 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_sum_product_l1783_178301


namespace NUMINAMATH_CALUDE_ab_inequality_l1783_178317

theorem ab_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = 2) :
  (a + b) * (a^5 + b^5) ≥ 4 ∧ a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_inequality_l1783_178317


namespace NUMINAMATH_CALUDE_expected_girls_left_10_7_l1783_178387

/-- The expected number of girls standing to the left of all boys in a random arrangement -/
def expected_girls_left (num_boys num_girls : ℕ) : ℚ :=
  num_girls / (num_boys + 1 : ℚ)

/-- Theorem: In a random arrangement of 10 boys and 7 girls, 
    the expected number of girls standing to the left of all boys is 7/11 -/
theorem expected_girls_left_10_7 :
  expected_girls_left 10 7 = 7 / 11 := by sorry

end NUMINAMATH_CALUDE_expected_girls_left_10_7_l1783_178387


namespace NUMINAMATH_CALUDE_triangle_existence_l1783_178303

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A B C : Point)

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Calculates the angle between three points -/
noncomputable def angle (p q r : Point) : ℝ :=
  sorry

/-- Checks if an angle is obtuse -/
def isObtuse (θ : ℝ) : Prop :=
  θ > Real.pi / 2

/-- Constructs a triangle given A₀, A₁, and A₂ -/
noncomputable def constructTriangle (A₀ A₁ A₂ : Point) : Option Triangle :=
  sorry

theorem triangle_existence (A₀ A₁ A₂ : Point) :
  ¬collinear A₀ A₁ A₂ →
  isObtuse (angle A₀ A₁ A₂) →
  ∃! (t : Triangle),
    (constructTriangle A₀ A₁ A₂ = some t) ∧
    (A₀.x = (t.B.x + t.C.x) / 2 ∧ A₀.y = (t.B.y + t.C.y) / 2) ∧
    collinear A₁ t.B t.C ∧
    (let midAlt := Point.mk ((t.A.x + A₀.x) / 2) ((t.A.y + A₀.y) / 2);
     A₂ = midAlt) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_l1783_178303


namespace NUMINAMATH_CALUDE_tetrahedron_surface_area_l1783_178308

/-- The surface area of a regular tetrahedron inscribed in a sphere, 
    which is itself inscribed in a cube with a surface area of 54 square meters. -/
theorem tetrahedron_surface_area : ℝ := by
  -- Define the surface area of the cube
  let cube_surface_area : ℝ := 54

  -- Define the relationship between the cube and the inscribed sphere
  let sphere_inscribed_in_cube : Prop := sorry

  -- Define the relationship between the sphere and the inscribed tetrahedron
  let tetrahedron_inscribed_in_sphere : Prop := sorry

  -- State that the surface area of the inscribed regular tetrahedron is 12√3
  have h : ∃ (area : ℝ), area = 12 * Real.sqrt 3 := sorry

  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_tetrahedron_surface_area_l1783_178308


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1783_178373

/-- An ellipse with equation x²/a² + y²/9 = 1, where a > 3 -/
structure Ellipse where
  a : ℝ
  h_a : a > 3

/-- The foci of the ellipse -/
structure Foci (e : Ellipse) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  h_dist : dist F₁ F₂ = 8

/-- A chord AB passing through F₁ -/
structure Chord (e : Ellipse) (f : Foci e) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_pass : A.1 = f.F₁.1 ∧ A.2 = f.F₁.2

/-- The theorem stating that the perimeter of triangle ABF₂ is 20 -/
theorem triangle_perimeter (e : Ellipse) (f : Foci e) (c : Chord e f) :
  dist c.A c.B + dist c.B f.F₂ + dist c.A f.F₂ = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1783_178373


namespace NUMINAMATH_CALUDE_book_club_hardcover_cost_l1783_178394

/-- Proves that the cost of each hardcover book is $30 given the book club fee structure --/
theorem book_club_hardcover_cost :
  let members : ℕ := 6
  let snack_fee : ℕ := 150
  let hardcover_count : ℕ := 6
  let paperback_count : ℕ := 6
  let paperback_cost : ℕ := 12
  let total_collected : ℕ := 2412
  ∃ (hardcover_cost : ℕ),
    members * (snack_fee + hardcover_count * hardcover_cost + paperback_count * paperback_cost) = total_collected ∧
    hardcover_cost = 30 :=
by sorry

end NUMINAMATH_CALUDE_book_club_hardcover_cost_l1783_178394


namespace NUMINAMATH_CALUDE_sin_tan_inequality_l1783_178381

theorem sin_tan_inequality (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) :
  2 * Real.sin α + Real.tan α > 3 * α := by
  sorry

end NUMINAMATH_CALUDE_sin_tan_inequality_l1783_178381


namespace NUMINAMATH_CALUDE_multiple_of_nine_implies_multiple_of_three_l1783_178339

theorem multiple_of_nine_implies_multiple_of_three (n : ℤ) :
  (∀ k : ℤ, 9 ∣ k → 3 ∣ k) →
  9 ∣ n →
  3 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_nine_implies_multiple_of_three_l1783_178339


namespace NUMINAMATH_CALUDE_dormitory_problem_l1783_178349

theorem dormitory_problem : ∃! x : ℕ+, ∃ n : ℕ+, 
  (x = 4 * n + 20) ∧ 
  (↑(n - 1) < (↑x : ℚ) / 8 ∧ (↑x : ℚ) / 8 < ↑n) := by
  sorry

end NUMINAMATH_CALUDE_dormitory_problem_l1783_178349


namespace NUMINAMATH_CALUDE_restaurant_bill_rounding_l1783_178322

theorem restaurant_bill_rounding (people : ℕ) (bill : ℚ) : 
  people = 9 → 
  bill = 314.16 → 
  ∃ (rounded_total : ℚ), 
    rounded_total = (people : ℚ) * (⌈(bill / people) * 100⌉ / 100) ∧ 
    rounded_total = 314.19 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_rounding_l1783_178322


namespace NUMINAMATH_CALUDE_remainder_of_prime_powers_l1783_178346

theorem remainder_of_prime_powers (p q : Nat) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  (p^(q - 1) + q^(p - 1)) % (p * q) = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_prime_powers_l1783_178346


namespace NUMINAMATH_CALUDE_min_value_expression_l1783_178353

theorem min_value_expression (a b : ℝ) (h : a^2 * b^2 + 2*a*b + 2*a + 1 = 0) :
  ∃ (x : ℝ), x = a*b*(a*b+2) + (b+1)^2 + 2*a ∧ 
  (∀ (y : ℝ), y = a*b*(a*b+2) + (b+1)^2 + 2*a → x ≤ y) ∧
  x = -3/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1783_178353


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1783_178314

theorem polynomial_remainder (x : ℝ) : 
  let p : ℝ → ℝ := λ x => 5*x^8 - 3*x^7 + 2*x^6 - 4*x^3 + x^2 - 9
  let d : ℝ → ℝ := λ x => 3*x - 9
  ∃ q : ℝ → ℝ, p = λ x => d x * q x + 39594 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1783_178314


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2379_l1783_178324

theorem smallest_prime_factor_of_2379 : Nat.minFac 2379 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2379_l1783_178324


namespace NUMINAMATH_CALUDE_water_in_bucket_A_l1783_178340

/-- Given two buckets A and B, prove that the original amount of water in bucket A is 20 kg. -/
theorem water_in_bucket_A (bucket_A bucket_B : ℝ) : 
  (0.2 * bucket_A = 0.4 * bucket_B) → 
  (0.6 * bucket_B = 6) → 
  bucket_A = 20 := by sorry

end NUMINAMATH_CALUDE_water_in_bucket_A_l1783_178340


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1783_178352

theorem cyclic_sum_inequality (x y z : ℝ) (α : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) (hα : α ≥ 0) :
  ((x^(α+3) + y^(α+3)) / (x^2 + x*y + y^2) +
   (y^(α+3) + z^(α+3)) / (y^2 + y*z + z^2) +
   (z^(α+3) + x^(α+3)) / (z^2 + z*x + x^2)) ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1783_178352


namespace NUMINAMATH_CALUDE_extremum_point_of_f_l1783_178351

def f (x : ℝ) := x^2 - 2*x

theorem extremum_point_of_f :
  ∃ (c : ℝ), c = 1 ∧ ∀ (x : ℝ), f x ≤ f c ∨ f x ≥ f c :=
sorry

end NUMINAMATH_CALUDE_extremum_point_of_f_l1783_178351


namespace NUMINAMATH_CALUDE_time_after_classes_l1783_178388

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hLt60 : minutes < 60

/-- Adds a duration in minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60,
    hLt60 := by sorry }

/-- The starting time of classes -/
def startTime : Time := { hours := 12, minutes := 0, hLt60 := by simp }

/-- The number of completed classes -/
def completedClasses : ℕ := 4

/-- The duration of each class in minutes -/
def classDuration : ℕ := 45

/-- Theorem: After 4 classes of 45 minutes each, starting at 12 pm, the time is 3 pm -/
theorem time_after_classes :
  (addMinutes startTime (completedClasses * classDuration)).hours = 15 := by sorry

end NUMINAMATH_CALUDE_time_after_classes_l1783_178388


namespace NUMINAMATH_CALUDE_highest_score_is_179_l1783_178376

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  totalInnings : ℕ
  averageScore : ℝ
  highestScore : ℝ
  lowestScore : ℝ
  averageExcludingExtremes : ℝ

/-- Theorem: Given the batsman's statistics, prove that the highest score is 179 runs -/
theorem highest_score_is_179 (stats : BatsmanStats)
  (h1 : stats.totalInnings = 46)
  (h2 : stats.averageScore = 60)
  (h3 : stats.highestScore - stats.lowestScore = 150)
  (h4 : stats.averageExcludingExtremes = 58) :
  stats.highestScore = 179 := by
  sorry

#check highest_score_is_179

end NUMINAMATH_CALUDE_highest_score_is_179_l1783_178376


namespace NUMINAMATH_CALUDE_square_of_one_plus_i_l1783_178316

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem square_of_one_plus_i : (1 + i)^2 = 2*i := by sorry

end NUMINAMATH_CALUDE_square_of_one_plus_i_l1783_178316


namespace NUMINAMATH_CALUDE_percentage_of_flowering_plants_l1783_178312

/-- Proves that the percentage of flowering plants is 40% given the conditions --/
theorem percentage_of_flowering_plants 
  (total_plants : ℕ)
  (porch_fraction : ℚ)
  (flowers_per_plant : ℕ)
  (total_porch_flowers : ℕ)
  (h1 : total_plants = 80)
  (h2 : porch_fraction = 1 / 4)
  (h3 : flowers_per_plant = 5)
  (h4 : total_porch_flowers = 40) :
  (total_porch_flowers : ℚ) / (porch_fraction * flowers_per_plant * total_plants) = 40 / 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_flowering_plants_l1783_178312


namespace NUMINAMATH_CALUDE_exponent_manipulation_l1783_178390

theorem exponent_manipulation (x y : ℝ) :
  (x - y)^4 * (y - x)^3 / (y - x)^2 = (x - y)^5 :=
by sorry

end NUMINAMATH_CALUDE_exponent_manipulation_l1783_178390


namespace NUMINAMATH_CALUDE_lineup_count_is_636_l1783_178392

/-- Represents a basketball team with specified number of players and positions -/
structure BasketballTeam where
  total_players : ℕ
  forwards : ℕ
  guards : ℕ
  versatile_players : ℕ
  lineup_forwards : ℕ
  lineup_guards : ℕ

/-- Calculates the number of different lineups for a given basketball team -/
def count_lineups (team : BasketballTeam) : ℕ :=
  sorry

/-- Theorem stating that the number of different lineups is 636 for the given team configuration -/
theorem lineup_count_is_636 : 
  let team : BasketballTeam := {
    total_players := 12,
    forwards := 6,
    guards := 4,
    versatile_players := 2,
    lineup_forwards := 3,
    lineup_guards := 2
  }
  count_lineups team = 636 := by sorry

end NUMINAMATH_CALUDE_lineup_count_is_636_l1783_178392


namespace NUMINAMATH_CALUDE_roses_picked_l1783_178364

theorem roses_picked (initial : ℕ) (sold : ℕ) (final : ℕ) 
  (h1 : initial = 37) 
  (h2 : sold = 16) 
  (h3 : final = 40) : 
  final - (initial - sold) = 19 := by
sorry

end NUMINAMATH_CALUDE_roses_picked_l1783_178364


namespace NUMINAMATH_CALUDE_cubic_factorization_l1783_178389

theorem cubic_factorization (a : ℝ) : a^3 - 4*a^2 + 4*a = a*(a-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1783_178389


namespace NUMINAMATH_CALUDE_john_leftover_percentage_l1783_178336

/-- The percentage of earnings John spent on rent -/
def rent_percentage : ℝ := 40

/-- The percentage less than rent that John spent on the dishwasher -/
def dishwasher_percentage_less : ℝ := 30

/-- Theorem stating that the percentage of John's earnings left over is 48% -/
theorem john_leftover_percentage : 
  100 - (rent_percentage + (100 - dishwasher_percentage_less) / 100 * rent_percentage) = 48 := by
  sorry

end NUMINAMATH_CALUDE_john_leftover_percentage_l1783_178336


namespace NUMINAMATH_CALUDE_min_square_value_l1783_178354

theorem min_square_value (a b : ℕ+) 
  (h1 : ∃ x : ℕ+, (15 * a.val + 16 * b.val : ℕ) = x * x)
  (h2 : ∃ y : ℕ+, (16 * a.val - 15 * b.val : ℕ) = y * y) :
  min (15 * a.val + 16 * b.val) (16 * a.val - 15 * b.val) ≥ 231361 :=
by sorry

end NUMINAMATH_CALUDE_min_square_value_l1783_178354


namespace NUMINAMATH_CALUDE_sum_max_min_f_l1783_178329

def f (x : ℝ) := -x^2 + 2*x + 3

theorem sum_max_min_f : 
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧ 
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max + min = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_sum_max_min_f_l1783_178329


namespace NUMINAMATH_CALUDE_role_assignment_count_l1783_178325

/-- The number of ways to assign roles in a play. -/
def assign_roles (num_men : ℕ) (num_women : ℕ) (male_roles : ℕ) (female_roles : ℕ) (either_roles : ℕ) : ℕ :=
  (num_men.descFactorial male_roles) *
  (num_women.descFactorial female_roles) *
  ((num_men + num_women - male_roles - female_roles).descFactorial either_roles)

/-- Theorem stating the number of ways to assign roles in the given scenario. -/
theorem role_assignment_count :
  assign_roles 7 8 3 3 4 = 213955200 :=
by sorry

end NUMINAMATH_CALUDE_role_assignment_count_l1783_178325


namespace NUMINAMATH_CALUDE_sqrt_division_minus_abs_l1783_178306

theorem sqrt_division_minus_abs : Real.sqrt 63 / Real.sqrt 7 - |(-4)| = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_minus_abs_l1783_178306


namespace NUMINAMATH_CALUDE_cos_sin_eighteen_degrees_identity_l1783_178398

theorem cos_sin_eighteen_degrees_identity : 
  4 * (Real.cos (18 * π / 180))^2 - 1 = 1 / (4 * (Real.sin (18 * π / 180))^2) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_eighteen_degrees_identity_l1783_178398


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1783_178368

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) :
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1783_178368


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1783_178315

def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 < x ∧ x < 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1783_178315


namespace NUMINAMATH_CALUDE_gingerbread_red_hat_percentage_l1783_178385

/-- Calculates the percentage of gingerbread men with red hats -/
theorem gingerbread_red_hat_percentage
  (red_hats : ℕ)
  (blue_boots : ℕ)
  (both : ℕ)
  (h1 : red_hats = 6)
  (h2 : blue_boots = 9)
  (h3 : both = 3) :
  (red_hats : ℚ) / ((red_hats + blue_boots - both) : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_gingerbread_red_hat_percentage_l1783_178385


namespace NUMINAMATH_CALUDE_fourth_fifth_supplier_cars_l1783_178347

/-- The number of cars each of the fourth and fifth suppliers receive -/
def cars_per_last_supplier (total_cars : ℕ) (first_supplier : ℕ) (additional_second : ℕ) : ℕ :=
  let second_supplier := first_supplier + additional_second
  let third_supplier := first_supplier + second_supplier
  let remaining_cars := total_cars - (first_supplier + second_supplier + third_supplier)
  remaining_cars / 2

/-- Proof that given the conditions, the fourth and fifth suppliers each receive 325,000 cars -/
theorem fourth_fifth_supplier_cars :
  cars_per_last_supplier 5650000 1000000 500000 = 325000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_fifth_supplier_cars_l1783_178347


namespace NUMINAMATH_CALUDE_usual_time_calculation_l1783_178331

/-- Given a man who walks at P% of his usual speed and takes T minutes more than usual,
    his usual time U (in minutes) to cover the distance is (P * T) / (100 - P). -/
theorem usual_time_calculation (P T : ℝ) (h1 : 0 < P) (h2 : P < 100) (h3 : 0 < T) :
  ∃ U : ℝ, U > 0 ∧ U = (P * T) / (100 - P) :=
sorry

end NUMINAMATH_CALUDE_usual_time_calculation_l1783_178331


namespace NUMINAMATH_CALUDE_football_team_addition_l1783_178362

theorem football_team_addition : 36 + 14 = 50 := by
  sorry

end NUMINAMATH_CALUDE_football_team_addition_l1783_178362


namespace NUMINAMATH_CALUDE_tripod_height_after_damage_l1783_178365

/-- Represents the height of a tripod after one leg is shortened -/
def tripod_height (leg_length : ℝ) (initial_height : ℝ) (shortened_length : ℝ) : ℝ :=
  -- Define the function to calculate the new height
  sorry

theorem tripod_height_after_damage :
  let leg_length : ℝ := 6
  let initial_height : ℝ := 5
  let shortened_length : ℝ := 1
  tripod_height leg_length initial_height shortened_length = 5 := by
  sorry

#check tripod_height_after_damage

end NUMINAMATH_CALUDE_tripod_height_after_damage_l1783_178365


namespace NUMINAMATH_CALUDE_unique_coprime_squares_l1783_178369

theorem unique_coprime_squares (m n : ℕ+) : 
  m.val.Coprime n.val ∧ 
  ∃ x y : ℕ, (m.val^2 - 5*n.val^2 = x^2) ∧ (m.val^2 + 5*n.val^2 = y^2) →
  m.val = 41 ∧ n.val = 12 :=
by sorry

end NUMINAMATH_CALUDE_unique_coprime_squares_l1783_178369


namespace NUMINAMATH_CALUDE_line_equation_of_points_on_parabola_l1783_178393

/-- Given a parabola y² = 4x and two points on it with midpoint (2, 2), 
    the line through these points has equation x - y = 0 -/
theorem line_equation_of_points_on_parabola (A B : ℝ × ℝ) : 
  (A.2^2 = 4 * A.1) →  -- A is on the parabola
  (B.2^2 = 4 * B.1) →  -- B is on the parabola
  ((A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 2) →  -- midpoint is (2, 2)
  ∃ (k : ℝ), ∀ (x y : ℝ), (x - A.1) = k * (y - A.2) ∧ x - y = 0 :=
sorry

end NUMINAMATH_CALUDE_line_equation_of_points_on_parabola_l1783_178393


namespace NUMINAMATH_CALUDE_sugar_difference_l1783_178391

theorem sugar_difference (brown_sugar white_sugar : ℝ) 
  (h1 : brown_sugar = 0.62)
  (h2 : white_sugar = 0.25) :
  brown_sugar - white_sugar = 0.37 := by
  sorry

end NUMINAMATH_CALUDE_sugar_difference_l1783_178391


namespace NUMINAMATH_CALUDE_product_closest_to_105_l1783_178345

def product : ℝ := 2.1 * (50.2 + 0.09)

def options : List ℝ := [100, 105, 106, 110]

theorem product_closest_to_105 : 
  ∀ x ∈ options, |product - 105| ≤ |product - x| := by
  sorry

end NUMINAMATH_CALUDE_product_closest_to_105_l1783_178345


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_difference_l1783_178359

theorem triangle_square_perimeter_difference (d : ℤ) : 
  (∃ (t s : ℝ), 3 * t - 4 * s = 1575 ∧ t - s = d ∧ s > 0) ↔ d > 525 :=
sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_difference_l1783_178359
