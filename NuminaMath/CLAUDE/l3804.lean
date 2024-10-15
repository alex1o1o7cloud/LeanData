import Mathlib

namespace NUMINAMATH_CALUDE_max_quarters_and_dimes_l3804_380491

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The total amount Eva has in cents -/
def total_amount : ℕ := 480

/-- 
Given $4.80 in U.S. coins with an equal number of quarters and dimes,
prove that the maximum number of quarters (and dimes) is 13.
-/
theorem max_quarters_and_dimes :
  ∃ (n : ℕ), n * (quarter_value + dime_value) ≤ total_amount ∧
             ∀ (m : ℕ), m * (quarter_value + dime_value) ≤ total_amount → m ≤ n ∧
             n = 13 :=
sorry

end NUMINAMATH_CALUDE_max_quarters_and_dimes_l3804_380491


namespace NUMINAMATH_CALUDE_prob_grad_degree_is_three_nineteenths_l3804_380454

/-- Represents the ratio of two quantities -/
structure Ratio :=
  (antecedent : ℕ)
  (consequent : ℕ)

/-- Represents the company's employee composition -/
structure Company :=
  (grad_ratio : Ratio)     -- Ratio of graduates with graduate degree to non-graduates
  (nongrad_ratio : Ratio)  -- Ratio of graduates without graduate degree to non-graduates

/-- Calculates the probability of a randomly picked college graduate having a graduate degree -/
def probability_grad_degree (c : Company) : ℚ :=
  let lcm := Nat.lcm c.grad_ratio.consequent c.nongrad_ratio.consequent
  let grad_scaled := c.grad_ratio.antecedent * (lcm / c.grad_ratio.consequent)
  let nongrad_scaled := c.nongrad_ratio.antecedent * (lcm / c.nongrad_ratio.consequent)
  grad_scaled / (grad_scaled + nongrad_scaled)

/-- The main theorem to be proved -/
theorem prob_grad_degree_is_three_nineteenths :
  ∀ c : Company,
    c.grad_ratio = ⟨1, 8⟩ →
    c.nongrad_ratio = ⟨2, 3⟩ →
    probability_grad_degree c = 3 / 19 :=
by
  sorry


end NUMINAMATH_CALUDE_prob_grad_degree_is_three_nineteenths_l3804_380454


namespace NUMINAMATH_CALUDE_lion_path_theorem_l3804_380471

/-- A broken line path within a circle -/
structure BrokenLinePath where
  points : List (Real × Real)
  inside_circle : ∀ p ∈ points, p.1^2 + p.2^2 ≤ 100

/-- The total length of a broken line path -/
def pathLength (path : BrokenLinePath) : Real :=
  sorry

/-- The sum of turning angles in a broken line path -/
def sumTurningAngles (path : BrokenLinePath) : Real :=
  sorry

/-- Main theorem: If a broken line path within a circle of radius 10 meters
    has a total length of 30,000 meters, then the sum of all turning angles
    along the path is at least 2998 radians -/
theorem lion_path_theorem (path : BrokenLinePath) 
    (h : pathLength path = 30000) :
  sumTurningAngles path ≥ 2998 := by
  sorry

end NUMINAMATH_CALUDE_lion_path_theorem_l3804_380471


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_a_value_l3804_380427

theorem hyperbola_asymptote_a_value :
  ∀ (a : ℝ), a > 0 →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / 9 = 1 → 
    ((3 * x + 2 * y = 0) ∨ (3 * x - 2 * y = 0))) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_a_value_l3804_380427


namespace NUMINAMATH_CALUDE_jerry_weller_votes_l3804_380450

theorem jerry_weller_votes 
  (total_votes : ℕ) 
  (vote_difference : ℕ) 
  (h1 : total_votes = 196554)
  (h2 : vote_difference = 20196) :
  ∃ (jerry_votes john_votes : ℕ),
    jerry_votes = 108375 ∧ 
    john_votes + vote_difference = jerry_votes ∧
    jerry_votes + john_votes = total_votes :=
by
  sorry

end NUMINAMATH_CALUDE_jerry_weller_votes_l3804_380450


namespace NUMINAMATH_CALUDE_inequality_chain_l3804_380408

theorem inequality_chain (b a x : ℝ) (h1 : b > a) (h2 : a > x) (h3 : x > 0) :
  x^2 < x*a ∧ x*a < a^2 ∧ a^2 < x*b := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l3804_380408


namespace NUMINAMATH_CALUDE_streamers_for_confetti_l3804_380489

/-- The price relationship between streamers and confetti packages -/
def price_relationship (p q : ℝ) : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  x * (1 + p / 100) = y ∧
  y * (1 - q / 100) = x

/-- The theorem stating the number of streamer packages that can be bought for 10 confetti packages -/
theorem streamers_for_confetti (p q : ℝ) :
  price_relationship p q →
  |p - q| = 90 →
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  x * (1 + p / 100) = y ∧
  y * (1 - q / 100) = x ∧
  10 * x = 4 * y :=
by sorry

end NUMINAMATH_CALUDE_streamers_for_confetti_l3804_380489


namespace NUMINAMATH_CALUDE_max_value_of_a_l3804_380425

-- Define the condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, |x - 2| + |x - a| ≥ a

-- State the theorem
theorem max_value_of_a :
  ∃ a_max : ℝ, a_max = 1 ∧
  inequality_holds a_max ∧
  ∀ a : ℝ, inequality_holds a → a ≤ a_max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3804_380425


namespace NUMINAMATH_CALUDE_fraction_transformation_l3804_380403

theorem fraction_transformation (x y : ℝ) (h1 : x / y = 2 / 5) (h2 : x + y = 5.25) :
  (x + 3) / (2 * y) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l3804_380403


namespace NUMINAMATH_CALUDE_nonagon_trapezium_existence_l3804_380424

/-- A type representing the vertices of a regular nonagon -/
inductive Vertex : Type
  | A | B | C | D | E | F | G | H | I

/-- A function to determine if four vertices form a trapezium -/
def is_trapezium (v1 v2 v3 v4 : Vertex) : Prop :=
  sorry -- The actual implementation would depend on the geometry of the nonagon

/-- Main theorem: Given any five vertices of a regular nonagon, 
    there always exists a subset of four vertices among them that form a trapezium -/
theorem nonagon_trapezium_existence 
  (chosen : Finset Vertex) 
  (h : chosen.card = 5) : 
  ∃ (v1 v2 v3 v4 : Vertex), v1 ∈ chosen ∧ v2 ∈ chosen ∧ v3 ∈ chosen ∧ v4 ∈ chosen ∧
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4 ∧
    is_trapezium v1 v2 v3 v4 :=
  sorry


end NUMINAMATH_CALUDE_nonagon_trapezium_existence_l3804_380424


namespace NUMINAMATH_CALUDE_unique_prime_for_equal_sets_l3804_380406

theorem unique_prime_for_equal_sets (p : Nat) (g : Nat) : 
  Nat.Prime p → 
  p % 2 = 1 → 
  (∀ a : Nat, 1 ≤ a → a < p → g^a % p ≠ 1) → 
  g^(p-1) % p = 1 → 
  (∀ k : Nat, 1 ≤ k → k ≤ (p-1)/2 → ∃ m : Nat, 1 ≤ m ∧ m ≤ (p-1)/2 ∧ (k^2 + 1) % p = g^m % p) → 
  (∀ m : Nat, 1 ≤ m → m ≤ (p-1)/2 → ∃ k : Nat, 1 ≤ k ∧ k ≤ (p-1)/2 ∧ g^m % p = (k^2 + 1) % p) → 
  p = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_for_equal_sets_l3804_380406


namespace NUMINAMATH_CALUDE_six_b_equals_twenty_l3804_380409

theorem six_b_equals_twenty (a b : ℚ) 
  (h1 : 10 * a = b) 
  (h2 : b = 20) 
  (h3 : 120 * a * b = 800) : 
  6 * b = 20 := by
sorry

end NUMINAMATH_CALUDE_six_b_equals_twenty_l3804_380409


namespace NUMINAMATH_CALUDE_remainder_problem_l3804_380407

theorem remainder_problem (n : ℤ) (h : n % 22 = 12) : (2 * n) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3804_380407


namespace NUMINAMATH_CALUDE_lucy_fish_count_l3804_380467

theorem lucy_fish_count (initial_fish : ℝ) (bought_fish : ℝ) : 
  initial_fish = 212.0 → bought_fish = 280.0 → initial_fish + bought_fish = 492.0 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l3804_380467


namespace NUMINAMATH_CALUDE_mario_moving_sidewalk_time_l3804_380465

/-- The time it takes Mario to walk from A to B on a moving sidewalk -/
theorem mario_moving_sidewalk_time (d : ℝ) (w : ℝ) (v : ℝ) : 
  d > 0 ∧ w > 0 ∧ v > 0 →  -- distances and speeds are positive
  d / w = 90 →             -- time to walk when sidewalk is off
  d / v = 45 →             -- time to be carried without walking
  d / (w + v) = 30 :=      -- time to walk on moving sidewalk
by sorry

end NUMINAMATH_CALUDE_mario_moving_sidewalk_time_l3804_380465


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3804_380461

theorem fraction_to_decimal : (45 : ℚ) / (2^3 * 5^4) = 0.0090 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3804_380461


namespace NUMINAMATH_CALUDE_choose_starters_count_l3804_380488

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 7 starters from a team of 16 players,
    where exactly one player must be chosen from a set of 4 quadruplets -/
def choose_starters : ℕ :=
  4 * binomial 12 6

theorem choose_starters_count : choose_starters = 3696 := by sorry

end NUMINAMATH_CALUDE_choose_starters_count_l3804_380488


namespace NUMINAMATH_CALUDE_teacher_grading_problem_l3804_380420

/-- Calculates the number of problems left to grade given the total number of worksheets,
    the number of graded worksheets, and the number of problems per worksheet. -/
def problems_left_to_grade (total_worksheets : ℕ) (graded_worksheets : ℕ) (problems_per_worksheet : ℕ) : ℕ :=
  (total_worksheets - graded_worksheets) * problems_per_worksheet

/-- Proves that given 9 total worksheets, 5 graded worksheets, and 4 problems per worksheet,
    there are 16 problems left to grade. -/
theorem teacher_grading_problem :
  problems_left_to_grade 9 5 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_teacher_grading_problem_l3804_380420


namespace NUMINAMATH_CALUDE_unique_valid_number_l3804_380455

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ 
  n = (n / 10)^3 + (n % 10)^3 - 3

theorem unique_valid_number : 
  ∃! n : ℕ, is_valid_number n ∧ n = 32 := by sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3804_380455


namespace NUMINAMATH_CALUDE_urea_formation_proof_l3804_380470

-- Define the chemical species
inductive Species
  | NH3
  | CO2
  | H2O
  | NH4_2CO3
  | NH4OH
  | NH2CONH2

-- Define the reaction equations
inductive Reaction
  | ammonium_carbonate_formation
  | ammonium_carbonate_hydrolysis
  | urea_formation

-- Define the initial quantities
def initial_quantities : Species → ℚ
  | Species.NH3 => 2
  | Species.CO2 => 1
  | Species.H2O => 1
  | _ => 0

-- Define the stoichiometric coefficients for each reaction
def stoichiometry : Reaction → Species → ℚ
  | Reaction.ammonium_carbonate_formation, Species.NH3 => -2
  | Reaction.ammonium_carbonate_formation, Species.CO2 => -1
  | Reaction.ammonium_carbonate_formation, Species.NH4_2CO3 => 1
  | Reaction.ammonium_carbonate_hydrolysis, Species.NH4_2CO3 => -1
  | Reaction.ammonium_carbonate_hydrolysis, Species.H2O => -1
  | Reaction.ammonium_carbonate_hydrolysis, Species.NH4OH => 2
  | Reaction.ammonium_carbonate_hydrolysis, Species.CO2 => 1
  | Reaction.urea_formation, Species.NH4OH => -1
  | Reaction.urea_formation, Species.CO2 => -1
  | Reaction.urea_formation, Species.NH2CONH2 => 1
  | Reaction.urea_formation, Species.H2O => 1
  | _, _ => 0

-- Define the function to calculate the amount of Urea formed
def urea_formed (reactions : List Reaction) : ℚ :=
  sorry

-- Theorem statement
theorem urea_formation_proof :
  urea_formed [Reaction.ammonium_carbonate_formation,
               Reaction.ammonium_carbonate_hydrolysis,
               Reaction.urea_formation] = 1 :=
sorry

end NUMINAMATH_CALUDE_urea_formation_proof_l3804_380470


namespace NUMINAMATH_CALUDE_remainder_problem_l3804_380451

theorem remainder_problem (n : ℕ) : 
  n % 68 = 0 ∧ n / 68 = 269 → n % 67 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3804_380451


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3804_380435

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 - x + 1 ≥ 0) ↔ (∃ x₀ : ℝ, 2 * x₀^2 - x₀ + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3804_380435


namespace NUMINAMATH_CALUDE_f_minimum_properties_l3804_380479

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (exp x - 1) + x

theorem f_minimum_properties {x_0 : ℝ} (h_pos : x_0 > 0) 
  (h_min : ∀ x > 0, f x ≥ f x_0) : 
  f x_0 = x_0 + 1 ∧ f x_0 < 3 := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_properties_l3804_380479


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3804_380434

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = Real.sqrt (1 - b^2 / a^2)

/-- A point on the ellipse -/
structure EllipsePoint (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The upper vertex of the ellipse -/
def upperVertex (E : Ellipse) : EllipsePoint E where
  x := 0
  y := E.b
  h_on_ellipse := by sorry

/-- A focus of the ellipse -/
structure Focus (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_major_axis : y = 0
  h_distance_from_center : x^2 = E.a^2 * E.e^2

/-- A line perpendicular to the line connecting a focus and the upper vertex -/
structure PerpendicularLine (E : Ellipse) (F : Focus E) where
  slope : ℝ
  h_perpendicular : slope * (F.x / E.b) = -1

/-- The intersection points of the perpendicular line with the ellipse -/
structure IntersectionPoints (E : Ellipse) (F : Focus E) (L : PerpendicularLine E F) where
  D : EllipsePoint E
  E : EllipsePoint E
  h_on_line_D : D.y = L.slope * (D.x - F.x)
  h_on_line_E : E.y = L.slope * (E.x - F.x)
  h_distance : (D.x - E.x)^2 + (D.y - E.y)^2 = 36

/-- The main theorem -/
theorem ellipse_triangle_perimeter
  (E : Ellipse)
  (h_e : E.e = 1/2)
  (F₁ F₂ : Focus E)
  (L : PerpendicularLine E F₁)
  (I : IntersectionPoints E F₁ L) :
  let A := upperVertex E
  let D := I.D
  let E := I.E
  (Real.sqrt ((A.x - D.x)^2 + (A.y - D.y)^2) +
   Real.sqrt ((A.x - E.x)^2 + (A.y - E.y)^2) +
   Real.sqrt ((D.x - E.x)^2 + (D.y - E.y)^2)) = 13 := by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3804_380434


namespace NUMINAMATH_CALUDE_solutions_of_z_sixth_power_eq_neg_64_l3804_380417

-- Define the complex number z
variable (z : ℂ)

-- Define the equation
def equation (z : ℂ) : Prop := z^6 = -64

-- State the theorem
theorem solutions_of_z_sixth_power_eq_neg_64 :
  (∀ z : ℂ, equation z ↔ z = 2*I ∨ z = -2*I) :=
sorry

end NUMINAMATH_CALUDE_solutions_of_z_sixth_power_eq_neg_64_l3804_380417


namespace NUMINAMATH_CALUDE_parabola_axis_symmetry_l3804_380416

/-- 
Given a parabola defined by y = a * x^2 with axis of symmetry y = -2,
prove that a = 1/8.
-/
theorem parabola_axis_symmetry (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) → 
  (∀ x : ℝ, -2 = a * x^2) → 
  a = 1/8 := by sorry

end NUMINAMATH_CALUDE_parabola_axis_symmetry_l3804_380416


namespace NUMINAMATH_CALUDE_decimal_multiplication_l3804_380494

theorem decimal_multiplication (a b c : ℚ) : 
  a = 8/10 → b = 25/100 → c = 2/10 → a * b * c = 4/100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l3804_380494


namespace NUMINAMATH_CALUDE_pi_is_monomial_l3804_380492

-- Define what a monomial is
def is_monomial (e : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℕ), ∀ x, e x = a * x^n

-- State the theorem
theorem pi_is_monomial : is_monomial (λ _ => Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_pi_is_monomial_l3804_380492


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3804_380414

theorem complex_fraction_equality : Complex.I * 5 / (1 - Complex.I * 2) = -2 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3804_380414


namespace NUMINAMATH_CALUDE_total_bathing_suits_l3804_380453

def one_piece : ℕ := 8500
def two_piece : ℕ := 12750
def trunks : ℕ := 5900
def shorts : ℕ := 7250
def children : ℕ := 1100

theorem total_bathing_suits :
  one_piece + two_piece + trunks + shorts + children = 35500 := by
  sorry

end NUMINAMATH_CALUDE_total_bathing_suits_l3804_380453


namespace NUMINAMATH_CALUDE_complex_number_problem_l3804_380459

-- Define the complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- Define the conditions
def condition1 : Prop := (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I
def condition2 : Prop := z₂.im = 2
def condition3 : Prop := (z₁ * z₂).im = 0

-- State the theorem
theorem complex_number_problem :
  condition1 z₁ → condition2 z₂ → condition3 z₁ z₂ → z₂ = 4 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3804_380459


namespace NUMINAMATH_CALUDE_expression_evaluation_l3804_380464

/-- Proves that the given expression evaluates to the specified value -/
theorem expression_evaluation (x y z : ℝ) (hx : x = 3) (hy : y = 4) (hz : z = 5) :
  3500 - (1000 / (20.50 + x * 10)) / (y^2 - 2*z) = 3496.6996699669967 := by
  sorry

#eval (3500 - (1000 / (20.50 + 3 * 10)) / (4^2 - 2*5) : Float)

end NUMINAMATH_CALUDE_expression_evaluation_l3804_380464


namespace NUMINAMATH_CALUDE_ellipse_focus_m_value_l3804_380445

/-- Given an ellipse with equation x²/25 + y²/m² = 1 where m > 0,
    if the left focus is at (-4,0), then m = 3 -/
theorem ellipse_focus_m_value (m : ℝ) :
  m > 0 →
  (∀ x y : ℝ, x^2/25 + y^2/m^2 = 1 → (x + 4)^2 + y^2 = (5 + m)^2) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_m_value_l3804_380445


namespace NUMINAMATH_CALUDE_difference_of_squares_l3804_380405

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3804_380405


namespace NUMINAMATH_CALUDE_equation_solution_l3804_380469

theorem equation_solution : ∃ x : ℝ, 0.4 * x + (0.6 * 0.8) = 0.56 ∧ x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3804_380469


namespace NUMINAMATH_CALUDE_division_problem_l3804_380476

theorem division_problem : 
  ∃ (q r : ℕ), 253 = (15 + 13 * 3 - 5) * q + r ∧ r < (15 + 13 * 3 - 5) ∧ q = 5 ∧ r = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3804_380476


namespace NUMINAMATH_CALUDE_ruby_pizza_tip_l3804_380474

/-- Represents the pizza order scenario --/
structure PizzaOrder where
  base_price : ℕ        -- Price of a pizza without toppings
  topping_price : ℕ     -- Price of each topping
  num_pizzas : ℕ        -- Number of pizzas ordered
  num_toppings : ℕ      -- Total number of toppings
  total_with_tip : ℕ    -- Total cost including tip

/-- Calculates the tip amount for a given pizza order --/
def calculate_tip (order : PizzaOrder) : ℕ :=
  order.total_with_tip - (order.base_price * order.num_pizzas + order.topping_price * order.num_toppings)

/-- Theorem stating that the tip for Ruby's pizza order is $5 --/
theorem ruby_pizza_tip :
  let order : PizzaOrder := {
    base_price := 10,
    topping_price := 1,
    num_pizzas := 3,
    num_toppings := 4,
    total_with_tip := 39
  }
  calculate_tip order = 5 := by
  sorry


end NUMINAMATH_CALUDE_ruby_pizza_tip_l3804_380474


namespace NUMINAMATH_CALUDE_max_cake_boxes_in_carton_l3804_380468

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the carton dimensions -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- Represents the cake box dimensions -/
def cakeBoxDimensions : BoxDimensions :=
  { length := 8, width := 7, height := 5 }

/-- Theorem stating the maximum number of cake boxes that can fit in the carton -/
theorem max_cake_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume cakeBoxDimensions) = 225 := by
  sorry

end NUMINAMATH_CALUDE_max_cake_boxes_in_carton_l3804_380468


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l3804_380418

/-- The slope angle of a line passing through points (0,√3) and (2,3√3) is π/3 -/
theorem slope_angle_of_line (A B : ℝ × ℝ) : 
  A = (0, Real.sqrt 3) → 
  B = (2, 3 * Real.sqrt 3) → 
  let slope := (B.2 - A.2) / (B.1 - A.1)
  Real.arctan slope = π / 3 := by sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l3804_380418


namespace NUMINAMATH_CALUDE_select_three_from_eight_l3804_380442

theorem select_three_from_eight : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_three_from_eight_l3804_380442


namespace NUMINAMATH_CALUDE_sugar_amount_l3804_380433

/-- The total amount of sugar the store owner started with, given the conditions. -/
theorem sugar_amount (num_packs : ℕ) (pack_weight : ℕ) (remaining_sugar : ℕ) 
  (h1 : num_packs = 12)
  (h2 : pack_weight = 250)
  (h3 : remaining_sugar = 20) :
  num_packs * pack_weight + remaining_sugar = 3020 :=
by sorry

end NUMINAMATH_CALUDE_sugar_amount_l3804_380433


namespace NUMINAMATH_CALUDE_disjunction_false_implies_negation_true_l3804_380466

variable (p q : Prop)

theorem disjunction_false_implies_negation_true :
  (¬(p ∨ q) → ¬p) ∧ ¬(¬p → ¬(p ∨ q)) := by sorry

end NUMINAMATH_CALUDE_disjunction_false_implies_negation_true_l3804_380466


namespace NUMINAMATH_CALUDE_jokes_count_l3804_380436

/-- The total number of jokes told by Jessy and Alan over two Saturdays -/
def total_jokes (jessy_first : ℕ) (alan_first : ℕ) : ℕ :=
  let first_saturday := jessy_first + alan_first
  let second_saturday := 2 * jessy_first + 2 * alan_first
  first_saturday + second_saturday

/-- Theorem stating the total number of jokes told by Jessy and Alan -/
theorem jokes_count : total_jokes 11 7 = 54 := by
  sorry

end NUMINAMATH_CALUDE_jokes_count_l3804_380436


namespace NUMINAMATH_CALUDE_total_fish_caught_l3804_380473

/-- The total number of fish caught by Jason, Ryan, and Jeffery is 100 -/
theorem total_fish_caught (jeffery_fish : ℕ) (h1 : jeffery_fish = 60) 
  (h2 : ∃ ryan_fish : ℕ, jeffery_fish = 2 * ryan_fish) 
  (h3 : ∃ jason_fish : ℕ, ∃ ryan_fish : ℕ, ryan_fish = 3 * jason_fish) : 
  ∃ total : ℕ, total = jeffery_fish + ryan_fish + jason_fish ∧ total = 100 :=
by sorry

end NUMINAMATH_CALUDE_total_fish_caught_l3804_380473


namespace NUMINAMATH_CALUDE_original_houses_count_l3804_380411

-- Define the given conditions
def houses_built_during_boom : ℕ := 97741
def current_total_houses : ℕ := 118558

-- Define the theorem to prove
theorem original_houses_count : 
  current_total_houses - houses_built_during_boom = 20817 := by
  sorry

end NUMINAMATH_CALUDE_original_houses_count_l3804_380411


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3804_380498

theorem solution_set_inequality (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (x - a) * (x - 1/a) < 0} = {x : ℝ | a < x ∧ x < 1/a} := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3804_380498


namespace NUMINAMATH_CALUDE_stan_boxes_count_l3804_380412

theorem stan_boxes_count (john jules joseph stan : ℕ) : 
  john = (120 * jules) / 100 →
  jules = joseph + 5 →
  joseph = (20 * stan) / 100 →
  john = 30 →
  stan = 100 := by
sorry

end NUMINAMATH_CALUDE_stan_boxes_count_l3804_380412


namespace NUMINAMATH_CALUDE_log_inequality_l3804_380444

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Main theorem
theorem log_inequality (a m n : ℝ) (ha : a > 1) (hm : 0 < m) (hmn : m < 1) (hn : 1 < n) :
  f a m < 0 ∧ 0 < f a n := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3804_380444


namespace NUMINAMATH_CALUDE_log_exponent_sum_l3804_380402

theorem log_exponent_sum (a : ℝ) (h : a = Real.log 3 / Real.log 4) : 
  2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_log_exponent_sum_l3804_380402


namespace NUMINAMATH_CALUDE_set_intersection_complement_problem_l3804_380478

theorem set_intersection_complement_problem :
  let U : Type := ℝ
  let A : Set U := {x | x ≤ 3}
  let B : Set U := {x | x ≤ 6}
  (Aᶜ ∩ B) = {x : U | 3 < x ∧ x ≤ 6} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_complement_problem_l3804_380478


namespace NUMINAMATH_CALUDE_distance_to_nearest_town_l3804_380429

theorem distance_to_nearest_town (d : ℝ) 
  (h1 : ¬(d ≥ 8))  -- Alice's statement is false
  (h2 : ¬(d ≤ 7))  -- Bob's statement is false
  (h3 : d ≠ 5)     -- Charlie's statement is false
  : 7 < d ∧ d < 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_nearest_town_l3804_380429


namespace NUMINAMATH_CALUDE_club_members_count_l3804_380462

def sock_cost : ℝ := 6
def tshirt_cost : ℝ := sock_cost + 7
def cap_cost : ℝ := tshirt_cost - 3
def total_cost_per_member : ℝ := 2 * (sock_cost + tshirt_cost + cap_cost)
def total_club_cost : ℝ := 3630

theorem club_members_count : 
  ∃ n : ℕ, n = 63 ∧ (n : ℝ) * total_cost_per_member = total_club_cost :=
sorry

end NUMINAMATH_CALUDE_club_members_count_l3804_380462


namespace NUMINAMATH_CALUDE_distance_between_points_l3804_380432

/-- The distance between two points (5, -3) and (9, 6) in a 2D plane is √97 units. -/
theorem distance_between_points : Real.sqrt 97 = Real.sqrt ((9 - 5)^2 + (6 - (-3))^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3804_380432


namespace NUMINAMATH_CALUDE_third_column_second_row_l3804_380400

/-- Represents a position in a classroom grid -/
structure Position :=
  (column : ℕ)
  (row : ℕ)

/-- The coordinate system for the classroom -/
def classroom_coordinate_system : Position → Bool
  | ⟨1, 2⟩ => true  -- This represents the condition that (1,2) is a valid position
  | _ => false

/-- Theorem: In the given coordinate system, (3,2) represents the 3rd column and 2nd row -/
theorem third_column_second_row :
  classroom_coordinate_system ⟨1, 2⟩ → 
  (∃ p : Position, p.column = 3 ∧ p.row = 2 ∧ classroom_coordinate_system p) :=
sorry

end NUMINAMATH_CALUDE_third_column_second_row_l3804_380400


namespace NUMINAMATH_CALUDE_chicken_cost_is_40_cents_l3804_380452

/-- The cost of chicken per plate given the total number of plates, 
    cost of rice per plate, and total spent on food. -/
def chicken_cost_per_plate (total_plates : ℕ) (rice_cost_per_plate : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent - (total_plates : ℚ) * rice_cost_per_plate) / (total_plates : ℚ)

/-- Theorem stating that the cost of chicken per plate is $0.40 
    given the specific conditions of the problem. -/
theorem chicken_cost_is_40_cents :
  chicken_cost_per_plate 100 (1/10) 50 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_chicken_cost_is_40_cents_l3804_380452


namespace NUMINAMATH_CALUDE_exact_location_determination_l3804_380480

-- Define a type for location descriptors
inductive LocationDescriptor
  | CinemaLocation (row : ℕ) (hall : ℕ) (cinema : String)
  | Direction (angle : ℝ)
  | StreetSection (street : String)
  | Coordinates (longitude : ℝ) (latitude : ℝ)

-- Define a function to check if a location descriptor can determine an exact location
def canDetermineExactLocation (descriptor : LocationDescriptor) : Prop :=
  match descriptor with
  | LocationDescriptor.Coordinates _ _ => True
  | _ => False

-- Theorem statement
theorem exact_location_determination
  (cinema_loc : LocationDescriptor)
  (direction : LocationDescriptor)
  (street_section : LocationDescriptor)
  (coordinates : LocationDescriptor)
  (h1 : cinema_loc = LocationDescriptor.CinemaLocation 2 3 "Pacific Cinema")
  (h2 : direction = LocationDescriptor.Direction 40)
  (h3 : street_section = LocationDescriptor.StreetSection "Middle section of Tianfu Avenue")
  (h4 : coordinates = LocationDescriptor.Coordinates 116 42) :
  canDetermineExactLocation coordinates ∧
  ¬canDetermineExactLocation cinema_loc ∧
  ¬canDetermineExactLocation direction ∧
  ¬canDetermineExactLocation street_section :=
sorry

end NUMINAMATH_CALUDE_exact_location_determination_l3804_380480


namespace NUMINAMATH_CALUDE_amanda_weekly_earnings_l3804_380482

def amanda_hourly_rate : ℝ := 20.00

def monday_appointments : ℕ := 5
def monday_appointment_duration : ℝ := 1.5

def tuesday_appointment_duration : ℝ := 3

def thursday_appointments : ℕ := 2
def thursday_appointment_duration : ℝ := 2

def saturday_appointment_duration : ℝ := 6

def total_hours : ℝ :=
  monday_appointments * monday_appointment_duration +
  tuesday_appointment_duration +
  thursday_appointments * thursday_appointment_duration +
  saturday_appointment_duration

theorem amanda_weekly_earnings :
  amanda_hourly_rate * total_hours = 410.00 := by
  sorry

end NUMINAMATH_CALUDE_amanda_weekly_earnings_l3804_380482


namespace NUMINAMATH_CALUDE_decimal_difference_l3804_380441

-- Define the repeating decimal 0.2̅4̅
def repeating_decimal : ℚ := 8 / 33

-- Define the terminating decimal 0.24
def terminating_decimal : ℚ := 24 / 100

-- Theorem statement
theorem decimal_difference :
  repeating_decimal - terminating_decimal = 2 / 825 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l3804_380441


namespace NUMINAMATH_CALUDE_valid_speaking_orders_l3804_380438

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n items -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 7

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of special students (A and B) -/
def special_students : ℕ := 2

theorem valid_speaking_orders : 
  (choose special_students 1 * choose (total_students - special_students) (selected_students - 1) * arrange selected_students selected_students) +
  (choose special_students 2 * choose (total_students - special_students) (selected_students - 2) * arrange selected_students selected_students) -
  (choose special_students 2 * choose (total_students - special_students) (selected_students - 2) * arrange (selected_students - 1) (selected_students - 1) * arrange 2 2) = 600 := by
  sorry

end NUMINAMATH_CALUDE_valid_speaking_orders_l3804_380438


namespace NUMINAMATH_CALUDE_sum_of_ABC_values_l3804_380431

/-- A function that represents the number A5B79C given digits A, B, and C -/
def number (A B C : ℕ) : ℕ := A * 100000 + 5 * 10000 + B * 1000 + 7 * 100 + 9 * 10 + C

/-- Predicate to check if a number is a single digit -/
def is_single_digit (n : ℕ) : Prop := n ≤ 9

/-- The sum of all possible values of A+B+C given the conditions -/
def sum_of_possible_values : ℕ := 29

/-- The main theorem -/
theorem sum_of_ABC_values (A B C : ℕ) 
  (hA : is_single_digit A) (hB : is_single_digit B) (hC : is_single_digit C)
  (h_div : (number A B C) % 11 = 0) : 
  sum_of_possible_values = 29 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ABC_values_l3804_380431


namespace NUMINAMATH_CALUDE_smallest_among_four_l3804_380460

theorem smallest_among_four (a b c d : ℚ) (h1 : a = -2) (h2 : b = -1) (h3 : c = 0) (h4 : d = 1) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_among_four_l3804_380460


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l3804_380472

theorem largest_x_sqrt_3x_eq_5x : 
  ∃ (x_max : ℚ), x_max = 3/25 ∧ 
  (∀ x : ℚ, x ≥ 0 → (Real.sqrt (3 * x) = 5 * x) → x ≤ x_max) ∧
  (Real.sqrt (3 * x_max) = 5 * x_max) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l3804_380472


namespace NUMINAMATH_CALUDE_sandy_initial_money_l3804_380493

/-- Sandy's initial amount of money before buying the pie -/
def initial_money : ℕ := sorry

/-- The cost of the pie -/
def pie_cost : ℕ := 6

/-- The amount of money Sandy has left after buying the pie -/
def remaining_money : ℕ := 57

/-- Theorem stating that Sandy's initial amount of money was 63 dollars -/
theorem sandy_initial_money : initial_money = 63 := by sorry

end NUMINAMATH_CALUDE_sandy_initial_money_l3804_380493


namespace NUMINAMATH_CALUDE_marbles_per_friend_l3804_380481

theorem marbles_per_friend (total_marbles : ℕ) (num_friends : ℕ) 
  (h1 : total_marbles = 72) (h2 : num_friends = 9) :
  total_marbles / num_friends = 8 :=
by sorry

end NUMINAMATH_CALUDE_marbles_per_friend_l3804_380481


namespace NUMINAMATH_CALUDE_square_number_divisible_by_5_between_20_and_110_l3804_380475

theorem square_number_divisible_by_5_between_20_and_110 (y : ℕ) :
  (∃ n : ℕ, y = n^2) →
  y % 5 = 0 →
  20 < y →
  y < 110 →
  (y = 25 ∨ y = 100) :=
by sorry

end NUMINAMATH_CALUDE_square_number_divisible_by_5_between_20_and_110_l3804_380475


namespace NUMINAMATH_CALUDE_f_properties_l3804_380443

noncomputable section

def f (x : ℝ) : ℝ := (Real.log x) / x - 1

def e : ℝ := Real.exp 1

theorem f_properties :
  (∀ x > 0, f x ≤ f e) ∧ 
  (∀ ε > 0, ∃ x > 0, f x < -1/ε) ∧
  (∀ m > 0, 
    (m ≤ e/2 → (∀ x ∈ Set.Icc m (2*m), f x ≤ f (2*m))) ∧
    (e/2 < m ∧ m < e → (∀ x ∈ Set.Icc m (2*m), f x ≤ f e)) ∧
    (m ≥ e → (∀ x ∈ Set.Icc m (2*m), f x ≤ f m))) :=
sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l3804_380443


namespace NUMINAMATH_CALUDE_tan_double_angle_problem_l3804_380484

open Real

theorem tan_double_angle_problem (θ : ℝ) 
  (h1 : tan (2 * θ) = -2 * sqrt 2) 
  (h2 : π < 2 * θ ∧ 2 * θ < 2 * π) : 
  tan θ = -sqrt 2 / 2 ∧ 
  (2 * (cos (θ / 2))^2 - sin θ - 1) / (sqrt 2 * sin (θ + π / 4)) = 3 + 2 * sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_problem_l3804_380484


namespace NUMINAMATH_CALUDE_elevator_time_to_bottom_l3804_380495

/-- Proves that the elevator takes 2 hours to reach the bottom floor given the specified conditions. -/
theorem elevator_time_to_bottom (total_floors : ℕ) (first_half_time : ℕ) (mid_floors_time : ℕ) (last_floors_time : ℕ) :
  total_floors = 20 →
  first_half_time = 15 →
  mid_floors_time = 5 →
  last_floors_time = 16 →
  (first_half_time + 5 * mid_floors_time + 5 * last_floors_time) / 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_elevator_time_to_bottom_l3804_380495


namespace NUMINAMATH_CALUDE_position_2025_l3804_380426

/-- Represents the possible positions of the square -/
inductive SquarePosition
  | ABCD
  | CDAB
  | BADC
  | DCBA

/-- Applies the transformation pattern to a given position -/
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.CDAB
  | SquarePosition.CDAB => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

/-- Returns the position after n transformations -/
def nthPosition (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.CDAB
  | 2 => SquarePosition.BADC
  | _ => SquarePosition.DCBA

theorem position_2025 : nthPosition 2025 = SquarePosition.ABCD := by
  sorry


end NUMINAMATH_CALUDE_position_2025_l3804_380426


namespace NUMINAMATH_CALUDE_rectangle_division_l3804_380477

theorem rectangle_division (a b c d e f : ℕ) : 
  (∀ a b, 39 ≠ 5 * a + 11 * b) ∧ 
  (∃ c d, 27 = 5 * c + 11 * d) ∧ 
  (∃ e f, 55 = 5 * e + 11 * f) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l3804_380477


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3804_380430

/-- Given vectors a and b in ℝ², and c = a + k*b, prove that if a ⊥ c, then k = -10/3 -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (3, 1))
  (h2 : b = (1, 0))
  (h3 : c = a + k • b)
  (h4 : a.1 * c.1 + a.2 * c.2 = 0) : 
  k = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3804_380430


namespace NUMINAMATH_CALUDE_impossible_to_face_up_all_coins_l3804_380419

/-- Represents the state of all coins -/
def CoinState := List Bool

/-- Represents a flip operation on 6 coins -/
def Flip := List Nat

/-- The initial state of the coins -/
def initialState : CoinState := 
  (List.replicate 1000 true) ++ (List.replicate 997 false)

/-- Applies a flip to a coin state -/
def applyFlip (state : CoinState) (flip : Flip) : CoinState :=
  sorry

/-- Checks if all coins are facing up -/
def allFacingUp (state : CoinState) : Bool :=
  state.all id

/-- Theorem stating that it's impossible to make all coins face up -/
theorem impossible_to_face_up_all_coins :
  ∀ (flips : List Flip), 
    ¬(allFacingUp (flips.foldl applyFlip initialState)) :=
by
  sorry

end NUMINAMATH_CALUDE_impossible_to_face_up_all_coins_l3804_380419


namespace NUMINAMATH_CALUDE_function_characterization_l3804_380485

theorem function_characterization (f : ℝ → ℝ) (C : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x ≥ 0) →
  (∀ x : ℝ, x ≥ 0 → f (f x) = x^4) →
  (∀ x : ℝ, x ≥ 0 → f x ≤ C * x^2) →
  C ≥ 1 →
  (∀ x : ℝ, x ≥ 0 → f x = x^2) :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l3804_380485


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l3804_380423

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (edge_sum : a + b + c = 35) 
  (diagonal : a^2 + b^2 + c^2 = 21^2) : 
  2 * (a*b + b*c + c*a) = 784 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l3804_380423


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3804_380439

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Two vectors are parallel -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : parallel (a 2, 2) (a 3, 3)) :
  (a 2 + a 4) / (a 3 + a 5) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3804_380439


namespace NUMINAMATH_CALUDE_yuna_survey_l3804_380497

theorem yuna_survey (math_lovers : ℕ) (korean_lovers : ℕ) (both_lovers : ℕ)
  (h1 : math_lovers = 27)
  (h2 : korean_lovers = 28)
  (h3 : both_lovers = 22) :
  math_lovers + korean_lovers - both_lovers = 33 := by
  sorry

end NUMINAMATH_CALUDE_yuna_survey_l3804_380497


namespace NUMINAMATH_CALUDE_mika_stickers_l3804_380496

/-- The number of stickers Mika has left after various additions and subtractions -/
def stickers_left (initial : ℕ) (bought : ℕ) (birthday : ℕ) (given : ℕ) (used : ℕ) : ℕ :=
  initial + bought + birthday - given - used

/-- Theorem stating that Mika is left with 2 stickers -/
theorem mika_stickers :
  stickers_left 20 26 20 6 58 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_l3804_380496


namespace NUMINAMATH_CALUDE_max_students_distribution_l3804_380456

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 2010) (h2 : pencils = 1050) : 
  (∃ (notebooks : ℕ), notebooks ≥ 30 ∧ 
    (∃ (distribution : ℕ → ℕ × ℕ × ℕ), 
      (∀ i j, i ≠ j → (distribution i).2.2 ≠ (distribution j).2.2) ∧
      (∀ i, i < 30 → (distribution i).1 = pens / 30 ∧ (distribution i).2.1 = pencils / 30))) ∧
  (∀ n : ℕ, n > 30 → 
    ¬(∃ (notebooks : ℕ), notebooks ≥ n ∧ 
      (∃ (distribution : ℕ → ℕ × ℕ × ℕ), 
        (∀ i j, i ≠ j → (distribution i).2.2 ≠ (distribution j).2.2) ∧
        (∀ i, i < n → (distribution i).1 = pens / n ∧ (distribution i).2.1 = pencils / n)))) :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3804_380456


namespace NUMINAMATH_CALUDE_max_perimeter_right_triangle_l3804_380413

theorem max_perimeter_right_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) (h5 : c = 5) :
  a + b + c ≤ 5 + 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_perimeter_right_triangle_l3804_380413


namespace NUMINAMATH_CALUDE_percentage_problem_l3804_380490

theorem percentage_problem (P : ℝ) : 
  (0.15 * P * (0.5 * 5600) = 126) → P = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3804_380490


namespace NUMINAMATH_CALUDE_fraction_denominator_l3804_380448

theorem fraction_denominator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (9 * y) / x + (3 * y) / 10 = 0.75 * y) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_l3804_380448


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l3804_380483

/-- The number of people sitting at the round table -/
def total_people : ℕ := 10

/-- The number of math majors -/
def math_majors : ℕ := 4

/-- The number of ways to choose seats for math majors -/
def total_ways : ℕ := Nat.choose total_people math_majors

/-- The number of ways math majors can sit consecutively -/
def consecutive_ways : ℕ := total_people

/-- The probability that all math majors sit in consecutive seats -/
def probability : ℚ := consecutive_ways / total_ways

theorem math_majors_consecutive_probability :
  probability = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l3804_380483


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3804_380463

def is_between (x a b : ℕ) : Prop := a < x ∧ x < b

def is_single_digit (x : ℕ) : Prop := x < 10

theorem unique_number_satisfying_conditions :
  ∃! x : ℕ, is_between x 5 9 ∧ is_single_digit x ∧ x > 7 :=
sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3804_380463


namespace NUMINAMATH_CALUDE_retailer_items_sold_l3804_380499

/-- The problem of determining the number of items sold by a retailer -/
theorem retailer_items_sold 
  (profit_per_item : ℝ) 
  (profit_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (min_items_with_discount : ℝ) : 
  profit_per_item = 30 →
  profit_percentage = 0.16 →
  discount_percentage = 0.05 →
  min_items_with_discount = 156.86274509803923 →
  ∃ (items_sold : ℕ), items_sold = 100 := by
  sorry

end NUMINAMATH_CALUDE_retailer_items_sold_l3804_380499


namespace NUMINAMATH_CALUDE_expression_value_l3804_380446

theorem expression_value : 
  (0.02^2 + 0.52^2 + 0.035^2) / (0.002^2 + 0.052^2 + 0.0035^2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3804_380446


namespace NUMINAMATH_CALUDE_monomial_count_is_four_l3804_380421

/-- A monomial is an algebraic expression consisting of one term. It can be a constant, a variable, or a product of constants and variables raised to whole number powers. -/
def is_monomial (expr : String) : Bool := sorry

/-- The list of algebraic expressions given in the problem -/
def expressions : List String := ["-2/3*a^3*b", "xy/2", "-4", "-2/a", "0", "x-y"]

/-- Count the number of monomials in a list of expressions -/
def count_monomials (exprs : List String) : Nat :=
  exprs.filter is_monomial |>.length

/-- The theorem to be proved -/
theorem monomial_count_is_four : count_monomials expressions = 4 := by sorry

end NUMINAMATH_CALUDE_monomial_count_is_four_l3804_380421


namespace NUMINAMATH_CALUDE_harmonic_is_T_sequence_T_sequence_property_T_sequence_property_2_l3804_380422

/-- Definition of a T sequence -/
def is_T_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) > a (n + 1) - a n

/-- The sequence A_n(n, 1/n) is a T sequence -/
theorem harmonic_is_T_sequence :
  is_T_sequence (fun n ↦ 1 / (n : ℝ)) := by sorry

/-- Property of T sequences for certain index relationships -/
theorem T_sequence_property (a : ℕ → ℝ) (h : is_T_sequence a) 
    (m n p q : ℕ) (hm : 1 ≤ m) (hmn : m < n) (hnp : n < p) (hpq : p < q) 
    (hsum : m + q = n + p) :
  a q - a p ≥ (q - p : ℝ) * (a (p + 1) - a p) := by sorry

/-- Another property of T sequences for certain index relationships -/
theorem T_sequence_property_2 (a : ℕ → ℝ) (h : is_T_sequence a) 
    (m n p q : ℕ) (hm : 1 ≤ m) (hmn : m < n) (hnp : n < p) (hpq : p < q) 
    (hsum : m + q = n + p) :
  a q - a n > a p - a m := by sorry

end NUMINAMATH_CALUDE_harmonic_is_T_sequence_T_sequence_property_T_sequence_property_2_l3804_380422


namespace NUMINAMATH_CALUDE_range_of_m_l3804_380437

-- Define propositions p and q as functions of x and m
def p (x m : ℝ) : Prop := (x - m)^2 > 3*(x - m)

def q (x : ℝ) : Prop := x^2 + 3*x - 4 < 0

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (m : ℝ) : Prop :=
  (∀ x, q x → p x m) ∧ (∃ x, p x m ∧ ¬q x)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, necessary_but_not_sufficient m ↔ (m ≥ 1 ∨ m ≤ -7) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3804_380437


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3804_380401

theorem train_bridge_crossing_time
  (train_length : ℝ)
  (bridge_length : ℝ)
  (train_speed_kmph : ℝ)
  (h1 : train_length = 165)
  (h2 : bridge_length = 660)
  (h3 : train_speed_kmph = 90) :
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 33 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3804_380401


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l3804_380415

theorem cos_pi_third_minus_alpha (α : Real) 
  (h : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5) : 
  Real.cos (π / 3 - α) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l3804_380415


namespace NUMINAMATH_CALUDE_number_divisibility_l3804_380457

theorem number_divisibility (N : ℕ) (h1 : N % 68 = 0) (h2 : N % 67 = 1) : N = 68 := by
  sorry

end NUMINAMATH_CALUDE_number_divisibility_l3804_380457


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3804_380449

theorem circle_center_coordinate_sum :
  ∀ (x y h k : ℝ),
  (∀ (x' y' : ℝ), x'^2 + y'^2 = 4*x' - 6*y' + 9 ↔ (x' - h)^2 + (y' - k)^2 = (h^2 + k^2 - 9 + 4*h - 6*k)) →
  h + k = -1 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3804_380449


namespace NUMINAMATH_CALUDE_min_value_expression_l3804_380487

theorem min_value_expression (x y : ℝ) : 3 * x^2 + 3 * x * y + y^2 - 6 * x + 4 * y + 5 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3804_380487


namespace NUMINAMATH_CALUDE_polynomial_roots_l3804_380440

def polynomial (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem polynomial_roots : 
  ∃ (a b c d e : ℝ), 
    (a = -1 - Real.sqrt 3) ∧
    (b = -1 + Real.sqrt 3) ∧
    (c = -1) ∧
    (d = 1) ∧
    (e = 2) ∧
    (∀ x : ℝ, polynomial x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3804_380440


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l3804_380428

theorem theater_ticket_difference :
  ∀ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = 370 →
    12 * orchestra_tickets + 8 * balcony_tickets = 3320 →
    balcony_tickets - orchestra_tickets = 190 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l3804_380428


namespace NUMINAMATH_CALUDE_product_inequality_l3804_380486

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4*a/(b+c)) * (1 + 4*b/(c+a)) * (1 + 4*c/(a+b)) > 25 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3804_380486


namespace NUMINAMATH_CALUDE_modulus_of_z_is_two_l3804_380447

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := by sorry

-- State the theorem
theorem modulus_of_z_is_two :
  z * (2 - 3 * i) = 6 + 4 * i → Complex.abs z = 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_is_two_l3804_380447


namespace NUMINAMATH_CALUDE_brain_info_scientific_notation_l3804_380404

/-- The number of pieces of information the human brain can record per day -/
def brain_info_capacity : ℕ := 86000000

/-- Scientific notation representation of brain_info_capacity -/
def brain_info_scientific : ℝ := 8.6 * (10 ^ 7)

theorem brain_info_scientific_notation :
  (brain_info_capacity : ℝ) = brain_info_scientific := by
  sorry

end NUMINAMATH_CALUDE_brain_info_scientific_notation_l3804_380404


namespace NUMINAMATH_CALUDE_sweet_potato_problem_l3804_380410

-- Define the problem parameters
def total_harvested : ℕ := 80
def sold_to_adams : ℕ := 20
def sold_to_lenon : ℕ := 15
def traded_for_pumpkins : ℕ := 10
def pumpkins_received : ℕ := 5
def pumpkin_weight : ℕ := 3
def donation_percentage : Rat := 5 / 100

-- Define the theorem
theorem sweet_potato_problem :
  let remaining_before_donation := total_harvested - (sold_to_adams + sold_to_lenon + traded_for_pumpkins)
  let donation := (remaining_before_donation : Rat) * donation_percentage
  let remaining_after_donation := remaining_before_donation - ⌈donation⌉
  remaining_after_donation = 33 ∧ pumpkins_received * pumpkin_weight = 15 := by
  sorry


end NUMINAMATH_CALUDE_sweet_potato_problem_l3804_380410


namespace NUMINAMATH_CALUDE_smaller_prime_l3804_380458

theorem smaller_prime (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y)
  (h1 : x + y = 36) (h2 : 4 * x + y = 87) : x < y := by
  sorry

end NUMINAMATH_CALUDE_smaller_prime_l3804_380458
