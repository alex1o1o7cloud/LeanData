import Mathlib

namespace NUMINAMATH_CALUDE_inequality_transformation_l1760_176014

theorem inequality_transformation (a b : ℝ) (h : a > b) : 2 * a + 1 > 2 * b + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l1760_176014


namespace NUMINAMATH_CALUDE_monomial_count_l1760_176058

/-- An algebraic expression is a monomial if it is a single number, a single variable, or a product of numbers and variables without variables in the denominator. -/
def is_monomial (expr : String) : Bool :=
  sorry

/-- The set of given algebraic expressions -/
def expressions : List String := ["2x^2", "-3", "x-2y", "t", "6m^2/π", "1/a", "m^3+2m^2-m"]

/-- Count the number of monomials in a list of expressions -/
def count_monomials (exprs : List String) : Nat :=
  sorry

/-- The main theorem: The number of monomials in the given set of expressions is 4 -/
theorem monomial_count : count_monomials expressions = 4 := by
  sorry

end NUMINAMATH_CALUDE_monomial_count_l1760_176058


namespace NUMINAMATH_CALUDE_integral_identity_l1760_176097

theorem integral_identity : ∫ x in (2 * Real.arctan 2)..(2 * Real.arctan 3), 
  1 / (Real.cos x * (1 - Real.cos x)) = 1/6 + Real.log 2 - Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_identity_l1760_176097


namespace NUMINAMATH_CALUDE_randy_theorem_l1760_176099

def randy_problem (initial_amount : ℕ) (smith_contribution : ℕ) (sally_gift : ℕ) : Prop :=
  let total := initial_amount + smith_contribution
  let remaining := total - sally_gift
  remaining = 2000

theorem randy_theorem : randy_problem 3000 200 1200 := by
  sorry

end NUMINAMATH_CALUDE_randy_theorem_l1760_176099


namespace NUMINAMATH_CALUDE_mowing_problem_l1760_176035

/-- Represents the time it takes to mow a lawn together -/
def mowing_time (mary_rate tom_rate : ℚ) (tom_alone_time : ℚ) : ℚ :=
  let remaining_lawn := 1 - tom_rate * tom_alone_time
  remaining_lawn / (mary_rate + tom_rate)

theorem mowing_problem :
  let mary_rate : ℚ := 1 / 3  -- Mary's mowing rate (lawn per hour)
  let tom_rate : ℚ := 1 / 6   -- Tom's mowing rate (lawn per hour)
  let tom_alone_time : ℚ := 2 -- Time Tom mows alone (hours)
  mowing_time mary_rate tom_rate tom_alone_time = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_mowing_problem_l1760_176035


namespace NUMINAMATH_CALUDE_team_C_most_uniform_l1760_176085

-- Define the teams
inductive Team : Type
| A : Team
| B : Team
| C : Team
| D : Team

-- Define the variance for each team
def variance : Team → ℝ
| Team.A => 0.13
| Team.B => 0.11
| Team.C => 0.09
| Team.D => 0.15

-- Define a function to determine if a team has the most uniform height
def has_most_uniform_height (t : Team) : Prop :=
  ∀ other : Team, variance t ≤ variance other

-- Theorem: Team C has the most uniform height
theorem team_C_most_uniform : has_most_uniform_height Team.C := by
  sorry


end NUMINAMATH_CALUDE_team_C_most_uniform_l1760_176085


namespace NUMINAMATH_CALUDE_contrapositive_example_l1760_176084

theorem contrapositive_example : 
  (∀ x : ℝ, x ≤ -3 → x < 0) ↔ (∀ x : ℝ, x ≥ 0 → x > -3) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l1760_176084


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1760_176002

-- Define the sets A and B
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2 - 2*x}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x + 1)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1 : ℝ) (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1760_176002


namespace NUMINAMATH_CALUDE_printer_price_ratio_l1760_176054

/-- Given the price of a basic computer and printer setup, prove the ratio of the printer price
    to the total price of an enhanced computer and printer setup. -/
theorem printer_price_ratio (basic_computer_price printer_price enhanced_computer_price : ℕ) : 
  basic_computer_price + printer_price = 2500 →
  enhanced_computer_price = basic_computer_price + 500 →
  basic_computer_price = 2125 →
  printer_price / (enhanced_computer_price + printer_price) = 1 / 8 := by
  sorry

#check printer_price_ratio

end NUMINAMATH_CALUDE_printer_price_ratio_l1760_176054


namespace NUMINAMATH_CALUDE_three_quarters_of_48_minus_12_l1760_176047

theorem three_quarters_of_48_minus_12 : (3 / 4 : ℚ) * 48 - 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_three_quarters_of_48_minus_12_l1760_176047


namespace NUMINAMATH_CALUDE_existence_of_basis_vectors_l1760_176076

-- Define the set of points
variable (n : ℕ)
variable (O : ℝ × ℝ)
variable (A : Fin n → ℝ × ℝ)

-- Define the distance condition
variable (h : ∀ (i j : Fin n), ∃ (m : ℕ), ‖A i - A j‖ = Real.sqrt m)
variable (h' : ∀ (i : Fin n), ∃ (m : ℕ), ‖A i - O‖ = Real.sqrt m)

-- The theorem to be proved
theorem existence_of_basis_vectors :
  ∃ (x y : ℝ × ℝ), ∀ (i : Fin n), ∃ (k l : ℤ), A i - O = k • x + l • y :=
sorry

end NUMINAMATH_CALUDE_existence_of_basis_vectors_l1760_176076


namespace NUMINAMATH_CALUDE_procedure_cost_l1760_176081

theorem procedure_cost (insurance_coverage : Real) (amount_saved : Real) :
  insurance_coverage = 0.80 →
  amount_saved = 3520 →
  ∃ (cost : Real), cost = 4400 ∧ insurance_coverage * cost = amount_saved :=
by sorry

end NUMINAMATH_CALUDE_procedure_cost_l1760_176081


namespace NUMINAMATH_CALUDE_age_difference_l1760_176009

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 16) : a - c = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1760_176009


namespace NUMINAMATH_CALUDE_integer_equation_proof_l1760_176006

theorem integer_equation_proof (m n : ℤ) (h : 3 * m * n + 3 * m = n + 2) : 3 * m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_integer_equation_proof_l1760_176006


namespace NUMINAMATH_CALUDE_function_inequality_l1760_176019

-- Define the function F
noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x / Real.exp x

-- State the theorem
theorem function_inequality
  (f : ℝ → ℝ)
  (f_diff : Differentiable ℝ f)
  (h : ∀ x, deriv f x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2012 < Real.exp 2012 * f 0 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l1760_176019


namespace NUMINAMATH_CALUDE_radical_simplification_l1760_176016

theorem radical_simplification (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (15 * q) * Real.sqrt (10 * q^3) * Real.sqrt (14 * q^5) = 10 * q^4 * Real.sqrt (21 * q) :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l1760_176016


namespace NUMINAMATH_CALUDE_deepak_age_l1760_176051

theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 2 →
  rahul_age + 10 = 26 →
  deepak_age = 8 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1760_176051


namespace NUMINAMATH_CALUDE_chess_tournament_points_inequality_l1760_176011

theorem chess_tournament_points_inequality (boys girls : ℕ) (boys_points girls_points : ℚ) : 
  boys = 9 → 
  girls = 3 → 
  boys_points = 36 + (9 * 3 - boys_points) → 
  girls_points = 3 + (9 * 3 - girls_points) → 
  boys_points ≠ girls_points :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_points_inequality_l1760_176011


namespace NUMINAMATH_CALUDE_perimeter_marbles_12_l1760_176026

/-- A square made of marbles -/
structure MarbleSquare where
  side_length : ℕ
  
/-- The number of marbles on the perimeter of a square -/
def perimeter_marbles (square : MarbleSquare) : ℕ :=
  4 * square.side_length - 4

theorem perimeter_marbles_12 :
  ∀ (square : MarbleSquare),
    square.side_length = 12 →
    perimeter_marbles square = 44 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_marbles_12_l1760_176026


namespace NUMINAMATH_CALUDE_probability_three_girls_in_six_children_l1760_176087

theorem probability_three_girls_in_six_children :
  let n : ℕ := 6  -- Total number of children
  let k : ℕ := 3  -- Number of girls we're interested in
  let p : ℚ := 1/2  -- Probability of having a girl
  Nat.choose n k * p^k * (1-p)^(n-k) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_girls_in_six_children_l1760_176087


namespace NUMINAMATH_CALUDE_zephyrian_word_count_l1760_176040

/-- The number of letters in the Zephyrian alphabet -/
def zephyrian_alphabet_size : ℕ := 8

/-- The maximum word length in the Zephyrian language -/
def max_word_length : ℕ := 3

/-- Calculate the number of possible words in the Zephyrian language -/
def count_zephyrian_words : ℕ :=
  zephyrian_alphabet_size +
  zephyrian_alphabet_size ^ 2 +
  zephyrian_alphabet_size ^ 3

theorem zephyrian_word_count :
  count_zephyrian_words = 584 :=
sorry

end NUMINAMATH_CALUDE_zephyrian_word_count_l1760_176040


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_is_15_l1760_176064

/-- An isosceles triangle with side lengths 3, 6, and 6 has a perimeter of 15 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c p =>
    (a = 3 ∧ b = 6 ∧ c = 6) →  -- Two sides are 6, one side is 3
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (b = c) →  -- Isosceles condition
    (p = a + b + c) →  -- Definition of perimeter
    p = 15

theorem isosceles_triangle_perimeter_is_15 : 
  ∃ (a b c p : ℝ), isosceles_triangle_perimeter a b c p :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_is_15_l1760_176064


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1760_176082

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_7 : a + b + c + d + e + f = 7) :
  (1/a) + (4/b) + (9/c) + (16/d) + (25/e) + (36/f) ≥ 63 ∧
  ((1/a) + (4/b) + (9/c) + (16/d) + (25/e) + (36/f) = 63 ↔ 
   a = 1/3 ∧ b = 2/3 ∧ c = 1 ∧ d = 4/3 ∧ e = 5/3 ∧ f = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1760_176082


namespace NUMINAMATH_CALUDE_remainder_71_73_div_9_l1760_176022

theorem remainder_71_73_div_9 : (71 * 73) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_71_73_div_9_l1760_176022


namespace NUMINAMATH_CALUDE_distance_between_points_is_2_5_km_l1760_176065

/-- Represents the running scenario with given parameters -/
structure RunningScenario where
  initialStandingTime : Real
  constantRunningRate : Real
  averageRate1 : Real
  averageRate2 : Real

/-- Calculates the distance run between two average rate points -/
def distanceBetweenPoints (scenario : RunningScenario) : Real :=
  sorry

/-- Theorem stating the distance run between the two average rate points -/
theorem distance_between_points_is_2_5_km (scenario : RunningScenario) 
  (h1 : scenario.initialStandingTime = 15 / 60) -- 15 seconds in minutes
  (h2 : scenario.constantRunningRate = 7)
  (h3 : scenario.averageRate1 = 7.5)
  (h4 : scenario.averageRate2 = 85 / 12) : -- 7 minutes 5 seconds in minutes
  distanceBetweenPoints scenario = 2.5 :=
  sorry

#check distance_between_points_is_2_5_km

end NUMINAMATH_CALUDE_distance_between_points_is_2_5_km_l1760_176065


namespace NUMINAMATH_CALUDE_quadratic_root_reciprocal_relation_l1760_176067

/-- Given two quadratic equations ax² + bx + c = 0 and cx² + bx + a = 0,
    this theorem states that the roots of the second equation
    are the reciprocals of the roots of the first equation. -/
theorem quadratic_root_reciprocal_relation (a b c : ℝ) (x₁ x₂ : ℝ) :
  (a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) →
  (c * (1/x₁)^2 + b * (1/x₁) + a = 0 ∧ c * (1/x₂)^2 + b * (1/x₂) + a = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_reciprocal_relation_l1760_176067


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1760_176045

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 5*x + 5 = 16) → (∃ y : ℝ, y^2 - 5*y + 5 = 16 ∧ x + y = 5) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1760_176045


namespace NUMINAMATH_CALUDE_quadratic_other_x_intercept_l1760_176008

/-- A quadratic function with vertex (5, 8) and one x-intercept at (1, 0) has its other x-intercept at x = 9 -/
theorem quadratic_other_x_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 8 - a * (x - 5)^2) →  -- vertex form of quadratic with vertex (5, 8)
  (a * 1^2 + b * 1 + c = 0) →                       -- (1, 0) is an x-intercept
  (∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_other_x_intercept_l1760_176008


namespace NUMINAMATH_CALUDE_inequality_equivalent_to_interval_l1760_176072

-- Define the inequality
def inequality (x : ℝ) : Prop := |8 - x| / 4 < 3

-- Define the interval
def interval (x : ℝ) : Prop := -4 < x ∧ x < 20

-- Theorem statement
theorem inequality_equivalent_to_interval :
  ∀ x : ℝ, inequality x ↔ interval x :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalent_to_interval_l1760_176072


namespace NUMINAMATH_CALUDE_albert_pizza_consumption_l1760_176033

/-- The number of large pizzas Albert buys -/
def large_pizzas : ℕ := 2

/-- The number of small pizzas Albert buys -/
def small_pizzas : ℕ := 2

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 16

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := 8

/-- The total number of pizza slices Albert eats in one day -/
def total_slices : ℕ := large_pizzas * large_pizza_slices + small_pizzas * small_pizza_slices

theorem albert_pizza_consumption :
  total_slices = 48 := by
  sorry

end NUMINAMATH_CALUDE_albert_pizza_consumption_l1760_176033


namespace NUMINAMATH_CALUDE_abigail_fence_count_l1760_176069

/-- The number of fences Abigail builds in total -/
def total_fences (initial_fences : ℕ) (build_time_per_fence : ℕ) (additional_hours : ℕ) : ℕ :=
  initial_fences + (60 / build_time_per_fence) * additional_hours

theorem abigail_fence_count :
  total_fences 10 30 8 = 26 := by
  sorry

end NUMINAMATH_CALUDE_abigail_fence_count_l1760_176069


namespace NUMINAMATH_CALUDE_trig_simplification_l1760_176075

theorem trig_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l1760_176075


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1760_176029

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptote y = 2x -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b = 2*a) :
  let e := Real.sqrt (1 + b^2 / a^2)
  e = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1760_176029


namespace NUMINAMATH_CALUDE_transitivity_of_greater_than_l1760_176050

theorem transitivity_of_greater_than {a b c : ℝ} (h1 : a > b) (h2 : b > c) : a > c := by
  sorry

end NUMINAMATH_CALUDE_transitivity_of_greater_than_l1760_176050


namespace NUMINAMATH_CALUDE_wire_forms_perpendicular_segments_l1760_176089

/-- Represents a wire configuration -/
structure WireConfiguration where
  semicircles : ℕ
  straight_segments : ℕ
  segment_length : ℝ

/-- Represents a figure formed by the wire -/
inductive Figure
  | TwoPerpendicularSegments
  | Other

/-- Checks if a wire configuration can form two perpendicular segments -/
def can_form_perpendicular_segments (w : WireConfiguration) : Prop :=
  w.semicircles = 3 ∧ w.straight_segments = 4

/-- Theorem stating that a specific wire configuration can form two perpendicular segments -/
theorem wire_forms_perpendicular_segments (w : WireConfiguration) 
  (h : can_form_perpendicular_segments w) : 
  ∃ (f : Figure), f = Figure.TwoPerpendicularSegments :=
sorry

end NUMINAMATH_CALUDE_wire_forms_perpendicular_segments_l1760_176089


namespace NUMINAMATH_CALUDE_unique_pair_l1760_176036

def is_valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ a + b = 28 ∧ (Even a ∨ Even b)

theorem unique_pair : ∀ a b : ℕ, is_valid_pair a b → (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_l1760_176036


namespace NUMINAMATH_CALUDE_genetic_events_in_both_divisions_l1760_176057

-- Define cell division processes
inductive CellDivision
| mitosis
| meiosis

-- Define genetic events
inductive GeneticEvent
| mutation
| chromosomalVariation

-- Define cellular processes during division
structure CellularProcess where
  chromosomeReplication : Bool
  centromereSplitting : Bool

-- Define the occurrence of genetic events during cell division
def geneticEventOccurs (event : GeneticEvent) (division : CellDivision) : Prop :=
  ∃ (process : CellularProcess), 
    process.chromosomeReplication ∧ 
    process.centromereSplitting

-- Theorem statement
theorem genetic_events_in_both_divisions :
  (∀ (event : GeneticEvent) (division : CellDivision), 
    geneticEventOccurs event division) :=
sorry

end NUMINAMATH_CALUDE_genetic_events_in_both_divisions_l1760_176057


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l1760_176091

/-- The probability of drawing a red ball from a bag with white and red balls -/
theorem probability_of_red_ball (total_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) 
  (h1 : total_balls = white_balls + red_balls)
  (h2 : white_balls = 3)
  (h3 : red_balls = 7) :
  (red_balls : ℚ) / total_balls = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l1760_176091


namespace NUMINAMATH_CALUDE_theater_eye_color_ratio_l1760_176044

theorem theater_eye_color_ratio :
  let total_people : ℕ := 100
  let blue_eyes : ℕ := 19
  let brown_eyes : ℕ := total_people / 2
  let green_eyes : ℕ := 6
  let black_eyes : ℕ := total_people - (blue_eyes + brown_eyes + green_eyes)
  (black_eyes : ℚ) / total_people = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_theater_eye_color_ratio_l1760_176044


namespace NUMINAMATH_CALUDE_correct_freshmen_sample_l1760_176004

/-- Represents a stratified sampling scenario in a college -/
structure CollegeSampling where
  total_students : ℕ
  freshmen : ℕ
  sample_size : ℕ

/-- Calculates the number of freshmen to be sampled in a stratified sampling -/
def freshmen_in_sample (cs : CollegeSampling) : ℕ :=
  cs.sample_size * cs.freshmen / cs.total_students

/-- Theorem stating the correct number of freshmen to be sampled -/
theorem correct_freshmen_sample (cs : CollegeSampling) 
  (h1 : cs.total_students = 3000)
  (h2 : cs.freshmen = 800)
  (h3 : cs.sample_size = 300) :
  freshmen_in_sample cs = 80 :=
sorry

end NUMINAMATH_CALUDE_correct_freshmen_sample_l1760_176004


namespace NUMINAMATH_CALUDE_final_state_l1760_176066

/-- Represents the state of variables a, b, and c --/
structure State where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Executes the program statements and returns the final state --/
def execute : State := 
  let s1 : State := ⟨1, 2, 3⟩  -- Initial assignment: a=1, b=2, c=3
  let s2 : State := ⟨s1.a, s1.b, s1.b⟩  -- c = b
  let s3 : State := ⟨s2.a, s2.a, s2.c⟩  -- b = a
  ⟨s3.c, s3.b, s3.c⟩  -- a = c

/-- The theorem stating the final values of a, b, and c --/
theorem final_state : execute = ⟨2, 1, 2⟩ := by
  sorry


end NUMINAMATH_CALUDE_final_state_l1760_176066


namespace NUMINAMATH_CALUDE_inequality_proof_l1760_176062

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : 0 ≤ b ∧ b ≤ 1) 
  (h3 : 0 ≤ c ∧ c ≤ 1) 
  (h4 : a + b + c = 1 + Real.sqrt (2 * (1 - a) * (1 - b) * (1 - c))) :
  Real.sqrt (1 - a^2) + Real.sqrt (1 - b^2) + Real.sqrt (1 - c^2) ≤ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1760_176062


namespace NUMINAMATH_CALUDE_trig_function_properties_l1760_176083

open Real

theorem trig_function_properties :
  (∀ x, cos (x + π/3) = cos (π/3 - x)) ∧
  (∀ x, 3 * sin (2 * (x - π/6) + π/3) = 3 * sin (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_trig_function_properties_l1760_176083


namespace NUMINAMATH_CALUDE_sum_with_abs_zero_implies_triple_l1760_176049

theorem sum_with_abs_zero_implies_triple (a : ℝ) : a + |a| = 0 → a - |2*a| = 3*a := by
  sorry

end NUMINAMATH_CALUDE_sum_with_abs_zero_implies_triple_l1760_176049


namespace NUMINAMATH_CALUDE_train_late_speed_l1760_176071

/-- Proves that the late average speed is 35 kmph given the conditions of the train problem -/
theorem train_late_speed (distance : ℝ) (on_time_speed : ℝ) (late_time : ℝ) :
  distance = 70 →
  on_time_speed = 40 →
  late_time = (distance / on_time_speed) + (15 / 60) →
  ∃ (late_speed : ℝ), late_speed = distance / late_time ∧ late_speed = 35 := by
  sorry

end NUMINAMATH_CALUDE_train_late_speed_l1760_176071


namespace NUMINAMATH_CALUDE_initial_cards_equals_sum_l1760_176003

/-- The number of Pokemon cards Jason had initially --/
def initial_cards : ℕ := 13

/-- The number of Pokemon cards Jason gave away --/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left --/
def cards_left : ℕ := 4

/-- Theorem stating that the initial number of cards equals the sum of cards given away and cards left --/
theorem initial_cards_equals_sum : initial_cards = cards_given_away + cards_left := by
  sorry

end NUMINAMATH_CALUDE_initial_cards_equals_sum_l1760_176003


namespace NUMINAMATH_CALUDE_two_digit_number_decimal_sum_l1760_176063

theorem two_digit_number_decimal_sum (a b : ℕ) (h1 : a ≥ 1 ∧ a ≤ 9) (h2 : b ≥ 0 ∧ b ≤ 9) :
  let n := 10 * a + b
  (n : ℚ) + (a : ℚ) + (b : ℚ) / 10 = 869 / 10 → n = 79 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_decimal_sum_l1760_176063


namespace NUMINAMATH_CALUDE_max_y_value_l1760_176012

theorem max_y_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = -14) : y ≤ 21 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l1760_176012


namespace NUMINAMATH_CALUDE_max_p_value_l1760_176020

theorem max_p_value (p q r s : ℕ+) 
  (h1 : p < 3 * q)
  (h2 : q < 4 * r)
  (h3 : r < 5 * s)
  (h4 : s < 90) :
  p ≤ 5324 ∧ ∃ (p' q' r' s' : ℕ+), 
    p' = 5324 ∧ 
    p' < 3 * q' ∧ 
    q' < 4 * r' ∧ 
    r' < 5 * s' ∧ 
    s' < 90 :=
by sorry

end NUMINAMATH_CALUDE_max_p_value_l1760_176020


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1760_176043

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 2)
  (h3 : ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  ∀ n : ℕ, a n = 2 * n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1760_176043


namespace NUMINAMATH_CALUDE_coordinates_of_D_l1760_176017

-- Define the points
def C : ℝ × ℝ := (5, -1)
def M : ℝ × ℝ := (3, 7)

-- Define D as a point that satisfies the midpoint condition
def D : ℝ × ℝ := (2 * M.1 - C.1, 2 * M.2 - C.2)

-- Theorem statement
theorem coordinates_of_D :
  D.1 * D.2 = 15 ∧ D.1 + D.2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_D_l1760_176017


namespace NUMINAMATH_CALUDE_mike_tire_change_l1760_176077

def total_tires_changed (num_motorcycles num_cars tires_per_motorcycle tires_per_car : ℕ) : ℕ :=
  num_motorcycles * tires_per_motorcycle + num_cars * tires_per_car

theorem mike_tire_change :
  let num_motorcycles : ℕ := 12
  let num_cars : ℕ := 10
  let tires_per_motorcycle : ℕ := 2
  let tires_per_car : ℕ := 4
  total_tires_changed num_motorcycles num_cars tires_per_motorcycle tires_per_car = 64 := by
  sorry

end NUMINAMATH_CALUDE_mike_tire_change_l1760_176077


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1760_176015

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 3*x)^9 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 4^9 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1760_176015


namespace NUMINAMATH_CALUDE_kennel_problem_l1760_176039

/-- Represents the number of dogs in a kennel that don't like either watermelon or salmon. -/
def dogs_not_liking_either (total : ℕ) (watermelon : ℕ) (salmon : ℕ) (both : ℕ) : ℕ :=
  total - (watermelon + salmon - both)

/-- Theorem stating that in a kennel of 60 dogs, where 9 like watermelon, 
    48 like salmon, and 5 like both, 8 dogs don't like either. -/
theorem kennel_problem : dogs_not_liking_either 60 9 48 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_kennel_problem_l1760_176039


namespace NUMINAMATH_CALUDE_ratio_equality_l1760_176068

theorem ratio_equality (a b c d : ℚ) 
  (h1 : a = 4 * b) 
  (h2 : b = 5 * c) 
  (h3 : c = 3 * d) : 
  a * d / (b * c) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1760_176068


namespace NUMINAMATH_CALUDE_expression_evaluation_l1760_176037

theorem expression_evaluation : 6 * 5 * ((-1) ^ (2 ^ (3 ^ 5))) + ((-1) ^ (5 ^ (3 ^ 2))) = 29 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1760_176037


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l1760_176030

/-- Given a man who can row with the stream at 16 km/h and against the stream at 8 km/h,
    his rate in still water is 12 km/h. -/
theorem mans_rate_in_still_water
  (speed_with_stream : ℝ)
  (speed_against_stream : ℝ)
  (h_with : speed_with_stream = 16)
  (h_against : speed_against_stream = 8) :
  (speed_with_stream + speed_against_stream) / 2 = 12 := by
  sorry

#check mans_rate_in_still_water

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l1760_176030


namespace NUMINAMATH_CALUDE_junior_score_l1760_176041

theorem junior_score (total_students : ℕ) (junior_percentage senior_percentage : ℚ)
  (class_average senior_average : ℚ) (h1 : junior_percentage = 1/5)
  (h2 : senior_percentage = 4/5) (h3 : junior_percentage + senior_percentage = 1)
  (h4 : class_average = 85) (h5 : senior_average = 84) :
  let junior_count := (junior_percentage * total_students).num
  let senior_count := (senior_percentage * total_students).num
  let total_score := class_average * total_students
  let senior_total_score := senior_average * senior_count
  let junior_total_score := total_score - senior_total_score
  junior_total_score / junior_count = 89 := by
sorry


end NUMINAMATH_CALUDE_junior_score_l1760_176041


namespace NUMINAMATH_CALUDE_part_one_part_two_l1760_176000

-- Define the sets M and N
def M : Set ℝ := {x | (2*x - 2)/(x + 3) > 1}
def N (a : ℝ) : Set ℝ := {x | x^2 + (a - 8)*x - 8*a ≤ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ M
def q (a x : ℝ) : Prop := x ∈ N a

-- Part I: Relationship when a = -6
theorem part_one : 
  (∀ x, q (-6) x → p x) ∧ 
  (∃ x, p x ∧ ¬(q (-6) x)) := by sorry

-- Part II: Range of a where p is necessary but not sufficient for q
theorem part_two : 
  (∀ a, (∀ x, q a x → p x) ∧ (∃ x, p x ∧ ¬(q a x))) ↔ 
  a < -5 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1760_176000


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l1760_176038

theorem division_multiplication_equality : (1100 / 25) * 4 / 11 = 16 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l1760_176038


namespace NUMINAMATH_CALUDE_email_sending_ways_l1760_176093

/-- The number of ways to send emails given the number of email addresses and the number of emails to be sent. -/
def number_of_ways (num_addresses : ℕ) (num_emails : ℕ) : ℕ :=
  num_addresses ^ num_emails

/-- Theorem stating that the number of ways to send 5 emails using 3 email addresses is 3^5. -/
theorem email_sending_ways : number_of_ways 3 5 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_email_sending_ways_l1760_176093


namespace NUMINAMATH_CALUDE_min_value_of_product_l1760_176042

theorem min_value_of_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 47 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_product_l1760_176042


namespace NUMINAMATH_CALUDE_cos_330_degrees_l1760_176088

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l1760_176088


namespace NUMINAMATH_CALUDE_triangle_inequality_1_triangle_inequality_2_l1760_176034

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_A : A > 0
  pos_B : B > 0
  pos_C : C > 0
  angle_sum : A + B + C = π

-- State the theorems
theorem triangle_inequality_1 (t : Triangle) :
  1 / t.a^3 + 1 / t.b^3 + 1 / t.c^3 + t.a * t.b * t.c ≥ 2 * Real.sqrt 3 := by
  sorry

theorem triangle_inequality_2 (t : Triangle) :
  1 / t.A + 1 / t.B + 1 / t.C ≥ 9 / π := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_1_triangle_inequality_2_l1760_176034


namespace NUMINAMATH_CALUDE_total_campers_is_71_l1760_176031

/-- The total number of campers who went rowing and hiking -/
def total_campers (morning_rowing : ℕ) (morning_hiking : ℕ) (afternoon_rowing : ℕ) : ℕ :=
  morning_rowing + morning_hiking + afternoon_rowing

/-- Theorem stating that the total number of campers who went rowing and hiking is 71 -/
theorem total_campers_is_71 :
  total_campers 41 4 26 = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_is_71_l1760_176031


namespace NUMINAMATH_CALUDE_origami_distribution_l1760_176018

theorem origami_distribution (total_papers : ℕ) (num_cousins : ℕ) (papers_per_cousin : ℕ) : 
  total_papers = 48 → 
  num_cousins = 6 → 
  total_papers = num_cousins * papers_per_cousin → 
  papers_per_cousin = 8 := by
sorry

end NUMINAMATH_CALUDE_origami_distribution_l1760_176018


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1760_176021

theorem divisibility_theorem (a b c x y z : ℝ) :
  (a * y - b * x)^2 + (b * z - c * y)^2 + (c * x - a * z)^2 + (a * x + b * y + c * z)^2 =
  (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1760_176021


namespace NUMINAMATH_CALUDE_game_points_sum_l1760_176048

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allieRolls : List ℕ := [6, 3, 2, 4]
def charlieRolls : List ℕ := [5, 3, 1, 6]

theorem game_points_sum : 
  (List.sum (List.map g allieRolls)) + (List.sum (List.map g charlieRolls)) = 38 := by
  sorry

end NUMINAMATH_CALUDE_game_points_sum_l1760_176048


namespace NUMINAMATH_CALUDE_four_groups_four_spots_l1760_176070

/-- The number of ways to arrange tour groups among scenic spots with one spot unvisited -/
def tourArrangements (numGroups numSpots : ℕ) : ℕ :=
  (numGroups.choose 2) * (numSpots.factorial / (numSpots - 3).factorial)

/-- Theorem stating the number of arrangements for 4 groups and 4 spots -/
theorem four_groups_four_spots :
  tourArrangements 4 4 = 144 := by
  sorry

end NUMINAMATH_CALUDE_four_groups_four_spots_l1760_176070


namespace NUMINAMATH_CALUDE_abs_sum_inequalities_l1760_176027

theorem abs_sum_inequalities (a b : ℝ) (h : a * b > 0) : 
  (abs (a + b) > abs a) ∧ (abs (a + b) > abs (a - b)) := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequalities_l1760_176027


namespace NUMINAMATH_CALUDE_pet_store_cages_l1760_176095

/-- The number of bird cages in a pet store -/
def num_cages : ℕ := 9

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 6

/-- The total number of birds in the pet store -/
def total_birds : ℕ := 72

/-- Theorem stating that the number of bird cages is correct -/
theorem pet_store_cages :
  num_cages * (parrots_per_cage + parakeets_per_cage) = total_birds :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l1760_176095


namespace NUMINAMATH_CALUDE_billy_coins_l1760_176025

theorem billy_coins (quarter_piles dime_piles coins_per_pile : ℕ) 
  (h1 : quarter_piles = 2)
  (h2 : dime_piles = 3)
  (h3 : coins_per_pile = 4) :
  quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 20 :=
by sorry

end NUMINAMATH_CALUDE_billy_coins_l1760_176025


namespace NUMINAMATH_CALUDE_last_day_of_month_l1760_176086

/-- Days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

/-- Theorem: If the 24th day of a 31-day month is a Wednesday, 
    then the last day of the month (31st) is also a Wednesday -/
theorem last_day_of_month (d : DayOfWeek) (h : d = DayOfWeek.Wednesday) :
  advanceDay d 7 = DayOfWeek.Wednesday :=
by
  sorry

end NUMINAMATH_CALUDE_last_day_of_month_l1760_176086


namespace NUMINAMATH_CALUDE_at_least_ten_same_weight_l1760_176092

/-- Represents the weight of a coin as measured by the scale -/
structure MeasuredWeight where
  value : ℝ
  is_valid : value > 11

/-- Represents the actual weight of a coin -/
structure ActualWeight where
  value : ℝ
  is_valid : value > 10

/-- The scale's measurement is always off by exactly 1 gram -/
def scale_error (actual : ActualWeight) (measured : MeasuredWeight) : Prop :=
  (measured.value = actual.value + 1) ∨ (measured.value = actual.value - 1)

/-- A collection of 12 coin measurements -/
def CoinMeasurements := Fin 12 → MeasuredWeight

/-- The actual weights corresponding to the measurements -/
def ActualWeights := Fin 12 → ActualWeight

theorem at_least_ten_same_weight 
  (measurements : CoinMeasurements) 
  (actual_weights : ActualWeights) 
  (h : ∀ i, scale_error (actual_weights i) (measurements i)) :
  ∃ (w : ℝ) (s : Finset (Fin 12)), s.card ≥ 10 ∧ ∀ i ∈ s, (actual_weights i).value = w :=
sorry

end NUMINAMATH_CALUDE_at_least_ten_same_weight_l1760_176092


namespace NUMINAMATH_CALUDE_tenth_number_in_sixteenth_group_l1760_176090

/-- The sequence a_n defined by a_n = 2n - 3 -/
def a (n : ℕ) : ℤ := 2 * n - 3

/-- The first number in the kth group -/
def first_in_group (k : ℕ) : ℤ := k^2 - k - 1

/-- The mth number in the kth group -/
def number_in_group (k m : ℕ) : ℤ := first_in_group k + 2 * (m - 1)

theorem tenth_number_in_sixteenth_group :
  number_in_group 16 10 = 257 := by sorry

end NUMINAMATH_CALUDE_tenth_number_in_sixteenth_group_l1760_176090


namespace NUMINAMATH_CALUDE_right_triangle_partition_l1760_176073

-- Define the set of points on the sides of an equilateral triangle
def TrianglePoints : Type := Set (ℝ × ℝ)

-- Define a property that a set of points contains a right triangle
def ContainsRightTriangle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ × ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    (a.1 - b.1) * (a.1 - c.1) + (a.2 - b.2) * (a.2 - c.2) = 0

-- State the theorem
theorem right_triangle_partition (T : TrianglePoints) :
  ∀ (S₁ S₂ : Set (ℝ × ℝ)), S₁ ∪ S₂ = T ∧ S₁ ∩ S₂ = ∅ →
    ContainsRightTriangle S₁ ∨ ContainsRightTriangle S₂ :=
sorry

end NUMINAMATH_CALUDE_right_triangle_partition_l1760_176073


namespace NUMINAMATH_CALUDE_inequality_preserved_division_l1760_176078

theorem inequality_preserved_division (x y a : ℝ) (h : x > y) :
  x / (a^2 + 1) > y / (a^2 + 1) := by sorry

end NUMINAMATH_CALUDE_inequality_preserved_division_l1760_176078


namespace NUMINAMATH_CALUDE_quartic_polynomial_satisfies_conditions_l1760_176001

def q (x : ℚ) : ℚ := -1/6 * x^4 + 4/3 * x^3 - 4/3 * x^2 - 8/3 * x

theorem quartic_polynomial_satisfies_conditions :
  q 1 = -3 ∧ q 2 = -5 ∧ q 3 = -9 ∧ q 4 = -17 ∧ q 5 = -35 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_satisfies_conditions_l1760_176001


namespace NUMINAMATH_CALUDE_haris_joining_time_l1760_176024

theorem haris_joining_time (praveen_investment hari_investment : ℝ) 
  (profit_ratio_praveen profit_ratio_hari : ℕ) (x : ℝ) :
  praveen_investment = 3780 →
  hari_investment = 9720 →
  profit_ratio_praveen = 2 →
  profit_ratio_hari = 3 →
  (praveen_investment * 12) / (hari_investment * (12 - x)) = 
    profit_ratio_praveen / profit_ratio_hari →
  x = 5 := by sorry

end NUMINAMATH_CALUDE_haris_joining_time_l1760_176024


namespace NUMINAMATH_CALUDE_commutative_property_demonstration_l1760_176059

theorem commutative_property_demonstration :
  (2 + 1 + 5 - 1 = 2 - 1 + 1 + 5) →
  ∃ (a b c d : ℤ), a + b + c + d = b + c + d + a :=
by sorry

end NUMINAMATH_CALUDE_commutative_property_demonstration_l1760_176059


namespace NUMINAMATH_CALUDE_jack_book_loss_l1760_176052

/-- Calculates the amount of money Jack lost in a year buying and selling books. -/
theorem jack_book_loss (books_per_month : ℕ) (book_cost : ℕ) (selling_price : ℕ) (months_per_year : ℕ) : 
  books_per_month = 3 →
  book_cost = 20 →
  selling_price = 500 →
  months_per_year = 12 →
  (books_per_month * months_per_year * book_cost) - selling_price = 220 := by
sorry

end NUMINAMATH_CALUDE_jack_book_loss_l1760_176052


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1760_176023

theorem inequality_solution_set :
  ∀ x : ℝ, abs (2*x - 1) - abs (x - 2) < 0 ↔ -1 < x ∧ x < 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1760_176023


namespace NUMINAMATH_CALUDE_inequality_proof_l1760_176053

theorem inequality_proof (a b x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x / (a*y + b*z)) + (y / (a*z + b*x)) + (z / (a*x + b*y)) ≥ 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1760_176053


namespace NUMINAMATH_CALUDE_cards_found_l1760_176010

def initial_cards : ℕ := 7
def final_cards : ℕ := 54

theorem cards_found (initial : ℕ) (final : ℕ) (h1 : initial = initial_cards) (h2 : final = final_cards) :
  final - initial = 47 := by sorry

end NUMINAMATH_CALUDE_cards_found_l1760_176010


namespace NUMINAMATH_CALUDE_nonagon_triangles_l1760_176046

/-- The number of triangles formed by vertices of a regular nonagon -/
def triangles_in_nonagon : ℕ := Nat.choose 9 3

/-- Theorem stating that the number of triangles in a regular nonagon is 84 -/
theorem nonagon_triangles : triangles_in_nonagon = 84 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_triangles_l1760_176046


namespace NUMINAMATH_CALUDE_tan_ratio_inequality_l1760_176005

theorem tan_ratio_inequality (α β : Real) (h1 : 0 < α) (h2 : α < β) (h3 : β < π/2) : 
  (Real.tan α) / α < (Real.tan β) / β := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_inequality_l1760_176005


namespace NUMINAMATH_CALUDE_coffee_decaf_percentage_l1760_176013

/-- Given initial coffee stock, percentages, and additional purchase,
    calculate the percentage of decaffeinated coffee in the new batch. -/
theorem coffee_decaf_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (additional_purchase : ℝ)
  (final_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 0.3)
  (h3 : additional_purchase = 100)
  (h4 : final_decaf_percent = 0.36)
  (h5 : initial_stock > 0)
  (h6 : additional_purchase > 0) :
  let total_stock := initial_stock + additional_purchase
  let initial_decaf := initial_stock * initial_decaf_percent
  let total_decaf := total_stock * final_decaf_percent
  let new_decaf := total_decaf - initial_decaf
  new_decaf / additional_purchase = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_coffee_decaf_percentage_l1760_176013


namespace NUMINAMATH_CALUDE_half_of_expression_l1760_176094

theorem half_of_expression : (2^12 + 3 * 2^10) / 2 = 2^9 * 7 := by sorry

end NUMINAMATH_CALUDE_half_of_expression_l1760_176094


namespace NUMINAMATH_CALUDE_curtis_farm_chickens_l1760_176074

/-- The number of chickens on Mr. Curtis's farm -/
theorem curtis_farm_chickens :
  let roosters : ℕ := 28
  let non_egg_laying_hens : ℕ := 20
  let egg_laying_hens : ℕ := 277
  roosters + non_egg_laying_hens + egg_laying_hens = 325 :=
by sorry

end NUMINAMATH_CALUDE_curtis_farm_chickens_l1760_176074


namespace NUMINAMATH_CALUDE_perimeter_of_figure_C_l1760_176055

/-- Given a large rectangle composed of 20 identical small rectangles,
    prove that the perimeter of figure C is 40 cm given the perimeters of figures A and B. -/
theorem perimeter_of_figure_C (x y : ℝ) : 
  (x > 0) → 
  (y > 0) → 
  (6 * x + 2 * y = 56) →  -- Perimeter of figure A
  (4 * x + 6 * y = 56) →  -- Perimeter of figure B
  (2 * x + 6 * y = 40)    -- Perimeter of figure C
  := by sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_C_l1760_176055


namespace NUMINAMATH_CALUDE_quotient_problem_l1760_176028

theorem quotient_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 139)
  (h2 : divisor = 19)
  (h3 : remainder = 6)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 7 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l1760_176028


namespace NUMINAMATH_CALUDE_sin_sum_equality_l1760_176098

theorem sin_sum_equality : 
  Real.sin (30 * π / 180) * Real.sin (75 * π / 180) + 
  Real.sin (60 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_equality_l1760_176098


namespace NUMINAMATH_CALUDE_remainder_274_pow_274_mod_13_l1760_176007

theorem remainder_274_pow_274_mod_13 : 274^274 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_274_pow_274_mod_13_l1760_176007


namespace NUMINAMATH_CALUDE_fraction_valid_for_all_reals_l1760_176096

theorem fraction_valid_for_all_reals :
  ∀ x : ℝ, (x^2 + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_fraction_valid_for_all_reals_l1760_176096


namespace NUMINAMATH_CALUDE_next_friday_birthday_l1760_176060

/-- Represents the day of the week --/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Checks if a given year is a leap year --/
def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)

/-- Calculates the day of the week for May 27 in a given year, 
    assuming May 27, 2013 was a Monday --/
def dayOfWeekMay27 (year : Nat) : DayOfWeek :=
  sorry

/-- Theorem: The next year after 2013 when May 27 falls on a Friday is 2016 --/
theorem next_friday_birthday : 
  (dayOfWeekMay27 2013 = DayOfWeek.Monday) → 
  (∀ y : Nat, 2013 < y ∧ y < 2016 → dayOfWeekMay27 y ≠ DayOfWeek.Friday) ∧
  (dayOfWeekMay27 2016 = DayOfWeek.Friday) :=
sorry

end NUMINAMATH_CALUDE_next_friday_birthday_l1760_176060


namespace NUMINAMATH_CALUDE_total_paths_a_to_d_l1760_176061

/-- The number of paths between two adjacent points -/
def paths_between_adjacent : ℕ := 2

/-- The number of direct paths from A to D -/
def direct_paths : ℕ := 1

/-- Theorem: The total number of paths from A to D is 9 -/
theorem total_paths_a_to_d : 
  paths_between_adjacent^3 + direct_paths = 9 := by sorry

end NUMINAMATH_CALUDE_total_paths_a_to_d_l1760_176061


namespace NUMINAMATH_CALUDE_jump_rope_cost_is_seven_l1760_176080

/-- The cost of Dalton's jump rope --/
def jump_rope_cost (board_game_cost playground_ball_cost allowance_savings uncle_gift additional_needed : ℕ) : ℕ :=
  (allowance_savings + uncle_gift + additional_needed) - (board_game_cost + playground_ball_cost)

/-- Theorem stating that the jump rope costs $7 --/
theorem jump_rope_cost_is_seven :
  jump_rope_cost 12 4 6 13 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jump_rope_cost_is_seven_l1760_176080


namespace NUMINAMATH_CALUDE_set_B_characterization_l1760_176079

def U : Set ℕ := {x | x > 0 ∧ Real.log x < 1}

def A : Set ℕ := {x | x ∈ U ∧ ∃ n : ℕ, n ≤ 4 ∧ x = 2*n + 1}

def B : Set ℕ := {x | x ∈ U ∧ x % 2 = 0}

theorem set_B_characterization :
  B = {2, 4, 6, 8} :=
sorry

end NUMINAMATH_CALUDE_set_B_characterization_l1760_176079


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l1760_176056

/-- Prove that for points on an inverse proportion function with k < 0,
    the y-coordinates have a specific ordering. -/
theorem inverse_proportion_ordering (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (hk : k < 0)
  (h1 : y₁ = k / (-2))
  (h2 : y₂ = k / 1)
  (h3 : y₃ = k / 2) :
  y₂ < y₃ ∧ y₃ < y₁ :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l1760_176056


namespace NUMINAMATH_CALUDE_one_fourth_x_equals_nine_l1760_176032

theorem one_fourth_x_equals_nine (x : ℝ) (h : (1 / 3) * x = 12) : (1 / 4) * x = 9 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_x_equals_nine_l1760_176032
