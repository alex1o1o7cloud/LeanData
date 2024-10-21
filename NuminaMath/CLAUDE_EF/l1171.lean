import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_after_reflection_l1171_117187

noncomputable def P : ℝ × ℝ := (-1, -3)
noncomputable def Q : ℝ × ℝ := (2, 5)
noncomputable def R : ℝ × ℝ := (-4, 1)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem segment_length_after_reflection :
  distance R (reflect_over_x_axis R) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_after_reflection_l1171_117187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_subset_l1171_117189

theorem divisibility_in_subset (n : ℕ) (S : Finset ℕ) :
  S ⊆ Finset.range (2 * n + 1) →
  S.card = n + 1 →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_subset_l1171_117189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_rotation_60_is_6_l1171_117186

noncomputable def rotation_matrix (θ : Real) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

theorem smallest_power_rotation_60_is_6 :
  (∃ n : ℕ+, rotation_matrix (π/3)^(n : ℕ) = 1 ∧ ∀ m : ℕ+, m < n → rotation_matrix (π/3)^(m : ℕ) ≠ 1) ∧
  (rotation_matrix (π/3)^(6 : ℕ) = 1) := by
  sorry

#check smallest_power_rotation_60_is_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_rotation_60_is_6_l1171_117186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_l1171_117101

theorem trigonometric_sum (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : Real.sin α = 4/5) : 
  Real.sin (α + π/4) + Real.cos (α + π/4) = -3*Real.sqrt 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_l1171_117101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_g_range_l1171_117110

noncomputable def m (x : Real) : Real × Real := (Real.sin x, -1/2)

noncomputable def n (x : Real) : Real × Real := (Real.sqrt 3 * Real.cos x, Real.cos (2*x))

noncomputable def f (x : Real) : Real := (m x).1 * (n x).1 + (m x).2 * (n x).2

noncomputable def g (x : Real) : Real := f (x + Real.pi/6)

theorem f_properties :
  (∀ x, f x ≤ 1) ∧
  (∃ x, f x = 1) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ p, p > 0 → (∀ x, f (x + p) = f x) → p ≥ Real.pi) :=
sorry

theorem g_range :
  ∀ x ∈ Set.Icc 0 (Real.pi/2), -1/2 ≤ g x ∧ g x ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_g_range_l1171_117110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_micchell_classes_needed_l1171_117159

/-- Represents the submissiveness rating of a class -/
def SubmissivenessRating := Float

/-- Represents the number of classes -/
def ClassCount := Nat

/-- Calculates the college potential given the total submissiveness rating and the number of classes -/
def collegePotential (totalRating : Float) (classCount : Nat) : Float :=
  totalRating / classCount.toFloat

/-- The minimum college potential required for admission -/
def minRequiredPotential : Float := 3.995

/-- Theorem: Given the conditions, Micchell needs 160 more classes to achieve the required college potential -/
theorem micchell_classes_needed 
  (initialClasses : Nat)
  (initialPotential : Float)
  (futureRating : Float)
  (h1 : initialClasses = 40)
  (h2 : initialPotential = 3.975)
  (h3 : futureRating = 4.0) :
  let additionalClasses : Nat := 160
  let totalClasses := initialClasses + additionalClasses
  let totalRating := initialClasses.toFloat * initialPotential + additionalClasses.toFloat * futureRating
  collegePotential totalRating totalClasses ≥ minRequiredPotential ∧ 
  ∀ x : Nat, x < additionalClasses → 
    collegePotential (initialClasses.toFloat * initialPotential + x.toFloat * futureRating) (initialClasses + x) < minRequiredPotential :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_micchell_classes_needed_l1171_117159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_pi_fourth_l1171_117104

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sqrt 2 * Real.sin (x + Real.pi / 4)) - 1 / 2

theorem tangent_slope_at_pi_fourth :
  deriv f (Real.pi / 4) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_pi_fourth_l1171_117104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1171_117157

def range : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 60) (Finset.range 61)

def multiples_of_four : Finset ℕ := Finset.filter (λ n => n % 4 = 0) range

def probability_at_least_one_multiple_of_four : ℚ :=
  1 - (1 - (multiples_of_four.card : ℚ) / (range.card : ℚ))^2

theorem probability_theorem :
  probability_at_least_one_multiple_of_four = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l1171_117157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hclo2_weight_3_moles_l1171_117191

/-- The molecular weight of a substance in grams per mole -/
def molecularWeight (substance : String) : ℝ := sorry

/-- The number of atoms of an element in a molecule -/
def atomCount (element : String) (molecule : String) : ℕ := sorry

/-- The atomic weight of an element in grams per mole -/
def atomicWeight (element : String) : ℝ := sorry

axiom hydrogen_weight : atomicWeight "H" = 1.008
axiom chlorine_weight : atomicWeight "Cl" = 35.453
axiom oxygen_weight : atomicWeight "O" = 15.999

axiom hclo2_composition : 
  (atomCount "H" "HClO2" = 1) ∧ 
  (atomCount "Cl" "HClO2" = 1) ∧ 
  (atomCount "O" "HClO2" = 2)

axiom molecular_weight_sum (molecule : String) :
  molecularWeight molecule = 
    (atomCount "H" molecule * atomicWeight "H") +
    (atomCount "Cl" molecule * atomicWeight "Cl") +
    (atomCount "O" molecule * atomicWeight "O")

theorem hclo2_weight_3_moles : 
  3 * molecularWeight "HClO2" = 205.377 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hclo2_weight_3_moles_l1171_117191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1171_117123

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add a case for 0
  | 1 => 1
  | (n + 2) => (n + 1) / (n + 2) * sequence_a (n + 1)

theorem sequence_a_formula (n : ℕ) : sequence_a n = 1 / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1171_117123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l1171_117152

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 2 * n - 1

-- Define the sequence b_n
noncomputable def b (n : ℕ) : ℝ := 2^n + a n

-- Define the sum of the first n terms of b_n
noncomputable def T (n : ℕ) : ℝ := 2^(n+1) - 2 + n^2

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def S (n : ℕ) (a₁ d : ℝ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_proof :
  a 3 = 5 ∧ S 7 (a 1) (a 2 - a 1) = 49 →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = 2^(n+1) - 2 + n^2) := by
  sorry

#check arithmetic_sequence_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l1171_117152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_balls_loaded_l1171_117146

theorem tennis_balls_loaded (first_set : ℕ) (second_set : ℕ) 
  (hit_ratio_first : ℚ) (hit_ratio_second : ℚ) (total_not_hit : ℕ) :
  first_set = 100 →
  second_set = 75 →
  hit_ratio_first = 2 / 5 →
  hit_ratio_second = 1 / 3 →
  total_not_hit = 110 →
  (↑first_set * hit_ratio_first).floor + (↑second_set * hit_ratio_second).floor + total_not_hit = first_set + second_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_balls_loaded_l1171_117146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1171_117182

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * (x - 1)

theorem function_properties (a : ℝ) :
  (∀ x > 1, f a x < 0) →
  (a ≥ 2 ∧
   (∀ x ∈ Set.Icc 2 (Real.exp 1), Real.exp (x - 2) ≤ x^(Real.exp 1 - 2)) ∧
   (∀ x > Real.exp 1, Real.exp (x - 2) > x^(Real.exp 1 - 2))) :=
by sorry

lemma exp_pow_compare_aux (x : ℝ) :
  x ≥ 2 →
  (Real.exp (x - 2) < x^(Real.exp 1 - 2) ↔ x < Real.exp 1) ∧
  (Real.exp (x - 2) = x^(Real.exp 1 - 2) ↔ x = Real.exp 1) ∧
  (Real.exp (x - 2) > x^(Real.exp 1 - 2) ↔ x > Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1171_117182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_product_l1171_117138

theorem tan_half_product (α β : Real) 
  (h1 : Real.sin α + Real.sin β = 2 * Real.sin (α + β))
  (h2 : ∀ n : ℤ, α + β ≠ 2 * Real.pi * ↑n) : 
  Real.tan (α / 2) * Real.tan (β / 2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_product_l1171_117138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_counts_l1171_117171

def num_girls : ℕ := 3
def num_boys : ℕ := 5
def total_people : ℕ := num_girls + num_boys

def arrangements_girls_together : ℕ := (Nat.factorial (total_people - num_girls + 1)) * (Nat.factorial num_girls)
def arrangements_girls_separated : ℕ := (Nat.factorial (num_boys + 1)) * (Nat.factorial (total_people - num_boys))
def arrangements_girls_not_both_ends : ℕ := (Nat.choose num_boys 2) * (Nat.factorial (total_people - 2))
def arrangements_girls_not_simultaneously_both_ends : ℕ := (Nat.factorial total_people) - (Nat.choose num_girls 2) * (Nat.factorial (total_people - 2))

theorem arrangement_counts :
  arrangements_girls_together = 4320 ∧
  arrangements_girls_separated = 14400 ∧
  arrangements_girls_not_both_ends = 14400 ∧
  arrangements_girls_not_simultaneously_both_ends = 36000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_counts_l1171_117171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_amount_in_altered_solution_l1171_117124

/-- Represents the ratio of components in a cleaning solution -/
structure CleaningSolution where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- Calculates the new ratio after altering the solution -/
def alter_solution (s : CleaningSolution) : CleaningSolution :=
  { bleach := 3 * s.bleach,
    detergent := s.detergent,
    water := 2 * s.water }

theorem detergent_amount_in_altered_solution 
  (original : CleaningSolution)
  (h_original_ratio : original = ⟨2, 40, 100⟩)
  (h_water_amount : (alter_solution original).water = 200) :
  (alter_solution original).detergent = 80 := by
  sorry

-- Remove the #eval line as it was causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_amount_in_altered_solution_l1171_117124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1171_117172

open Real

theorem trigonometric_identities :
  (∀ α : ℝ, tan α = 1/3 → 1 / (2 * sin α * cos α + cos α ^ 2) = 2/3) ∧
  (∀ α : ℝ, (tan (π - α) * cos (2*π - α) * sin (-α + 3*π/2)) / (cos (-α - π) * sin (-π - α)) = 1) :=
by
  constructor
  · intro α h
    sorry -- Proof for the first part
  · intro α
    sorry -- Proof for the second part

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1171_117172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2012_equals_3_l1171_117121

def sequence_a : ℕ → ℚ
  | 0 => -2
  | n + 1 => (1 + sequence_a n) / (1 - sequence_a n)

theorem a_2012_equals_3 : sequence_a 2012 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2012_equals_3_l1171_117121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1171_117173

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  ((-a) * c + b^2 = 0) →
  c / a = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1171_117173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_five_seconds_l1171_117174

/-- The time (in seconds) it takes for a train to cross a pole --/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  train_length / (train_speed_kmph * 1000 / 3600)

/-- Proof that a train with given length and speed takes approximately 5 seconds to cross a pole --/
theorem train_crossing_approx_five_seconds :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |train_crossing_time 250.02 180 - 5| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_five_seconds_l1171_117174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_in_open_interval_l1171_117196

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2)^x - Real.log x

-- State the theorem
theorem x0_in_open_interval :
  ∃ (x₀ : ℝ), f x₀ > Real.log (Real.sin (π/8)) / Real.log (1/8) + Real.log (Real.cos (π/8)) / Real.log (1/8) →
  ∃ (a b : ℝ), a = 0 ∧ b = 1 ∧ a < x₀ ∧ x₀ < b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_in_open_interval_l1171_117196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_perpendicular_lines_parallel_lines_perpendicular_planes_l1171_117144

-- Define the basic structures
structure Line where

structure Plane where

-- Define the relationships
def perpendicular (l : Line) (p : Plane) : Prop :=
  sorry

def parallel (p1 p2 : Plane) : Prop :=
  sorry

def contained_in (l : Line) (p : Plane) : Prop :=
  sorry

def line_perpendicular (l1 l2 : Line) : Prop :=
  sorry

def line_parallel (l1 l2 : Line) : Prop :=
  sorry

def plane_perpendicular (p1 p2 : Plane) : Prop :=
  sorry

-- Define the given conditions
variable (l m : Line) (α β : Plane)
variable (h1 : perpendicular l α)
variable (h2 : contained_in m β)

-- State the theorems to be proved
theorem parallel_planes_perpendicular_lines 
  (h : parallel α β) : line_perpendicular l m := by
  sorry

theorem parallel_lines_perpendicular_planes 
  (h : line_parallel l m) : plane_perpendicular α β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_perpendicular_lines_parallel_lines_perpendicular_planes_l1171_117144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_probability_after_removal_l1171_117158

/-- Represents a deck of cards -/
structure Deck where
  cards : Finset (Fin 10)

/-- Represents a pair of cards -/
structure Pair where
  card1 : Fin 10
  card2 : Fin 10

/-- The initial deck of 40 cards -/
def initialDeck : Deck where
  cards := Finset.univ

/-- Removes a pair from the deck -/
def removePair (d : Deck) (p : Pair) : Deck where
  cards := d.cards.erase p.card1 \ {p.card2}

/-- Calculates the probability of selecting a pair from a deck -/
noncomputable def pairProbability (d : Deck) : ℚ :=
  -- Implementation details omitted
  0 -- Placeholder value

theorem pair_probability_after_removal :
  ∃ (d : Deck) (p1 p2 : Pair),
    p1 ≠ p2 ∧
    d = removePair (removePair initialDeck p1) p2 ∧
    pairProbability d = 25 / 315 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_probability_after_removal_l1171_117158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_burger_cost_l1171_117165

/-- Proves that the cost of a double burger is $1.50 given the problem conditions -/
theorem double_burger_cost 
  (total_spent : ℚ)
  (total_burgers : ℕ)
  (single_burger_cost : ℚ)
  (double_burgers : ℕ)
  (h1 : total_spent = 74.5)
  (h2 : total_burgers = 50)
  (h3 : single_burger_cost = 1)
  (h4 : double_burgers = 49) :
  let single_burgers := total_burgers - double_burgers
  let double_burger_cost := (total_spent - single_burger_cost * single_burgers) / double_burgers
  double_burger_cost = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_burger_cost_l1171_117165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l1171_117133

theorem tan_ratio_from_sin_sum_diff (a b : ℝ) 
  (h1 : Real.sin (a + b) = 1/2) 
  (h2 : Real.sin (a - b) = 1/4) : 
  Real.tan a / Real.tan b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l1171_117133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l1171_117170

theorem complex_magnitude_problem :
  Complex.abs ((1 - Complex.I) / (1 + Complex.I) + 2 * Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_problem_l1171_117170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l1171_117132

/-- Calculates the final amount for a compound interest investment -/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem investment_difference : 
  let principal := (40000 : ℝ)
  let rate := (0.05 : ℝ)
  let years := (3 : ℕ)
  let alice_amount := compound_interest principal rate years
  let bob_amount := compound_interest principal (rate / 2) (years * 2)
  round_to_nearest (bob_amount - alice_amount) = 66 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l1171_117132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payroll_amount_for_given_tax_l1171_117166

/-- Represents the special municipal payroll tax structure -/
noncomputable def payroll_tax (payroll : ℝ) : ℝ :=
  if payroll < 200000 then 0
  else if payroll ≤ 500000 then 0.002 * payroll
  else 0.004 * (payroll - 500000) + 2000

/-- Theorem: Given the tax structure and conditions, a company paying $600 in tax
    after a $1,000 deduction must have a payroll of $900,000 -/
theorem payroll_amount_for_given_tax : ∀ (payroll : ℝ),
  payroll > 500000 →
  payroll_tax payroll - 1000 = 600 →
  payroll = 900000 := by
  sorry

#check payroll_amount_for_given_tax

end NUMINAMATH_CALUDE_ERRORFEEDBACK_payroll_amount_for_given_tax_l1171_117166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_is_cos_l1171_117100

open Real

/-- Sequence of functions defined by f₀(x) = cos x and fₙ₊₁(x) = fₙ'(x) -/
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => cos
  | n + 1 => deriv (f n)

/-- Theorem stating that the 2016th function in the sequence is cosine -/
theorem f_2016_is_cos : f 2016 = cos := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_is_cos_l1171_117100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_sum_l1171_117142

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (3, 0)

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define that P is the midpoint of AB
def is_midpoint (A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_distance_sum (A B : ℝ × ℝ) :
  parabola A.1 A.2 → parabola B.1 B.2 → is_midpoint A B →
  distance A focus + distance B focus = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_sum_l1171_117142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_area_EFGH_l1171_117162

-- Define the circle F₁
def circle_F₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define the fixed point F₂
def F₂ : ℝ × ℝ := (1, 0)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the quadrilateral EFGH
structure Quadrilateral where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

-- Define the condition that EFGH has vertices on curve C
def on_curve_C (q : Quadrilateral) : Prop :=
  curve_C q.E.1 q.E.2 ∧ curve_C q.F.1 q.F.2 ∧ curve_C q.G.1 q.G.2 ∧ curve_C q.H.1 q.H.2

-- Define the condition that diagonals pass through origin
def diagonals_through_origin (q : Quadrilateral) : Prop :=
  ∃ t₁ t₂ : ℝ, (t₁ • q.E.1, t₁ • q.E.2) + ((1 - t₁) • q.G.1, (1 - t₁) • q.G.2) = (0, 0) ∧
              (t₂ • q.F.1, t₂ • q.F.2) + ((1 - t₂) • q.H.1, (1 - t₂) • q.H.2) = (0, 0)

-- Define the slope product condition
def slope_product_condition (q : Quadrilateral) : Prop :=
  let kₑₒ := q.E.2 / q.E.1
  let kₒₕ := q.H.2 / q.H.1
  kₑₒ * kₒₕ = -3/4

-- Define area function (placeholder)
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Main theorem
theorem curve_C_and_area_EFGH :
  ∀ (q : Quadrilateral),
    on_curve_C q →
    diagonals_through_origin q →
    slope_product_condition q →
    (∀ x y, curve_C x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
    (∃ S : ℝ, S = 4 * Real.sqrt 3 ∧ ∀ q', on_curve_C q' → diagonals_through_origin q' → slope_product_condition q' → area q' = S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_area_EFGH_l1171_117162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_b_value_l1171_117109

/-- A lattice point is a point with integer coordinates -/
def LatticePoint (x y : ℚ) : Prop := x.isInt ∧ y.isInt

/-- The line equation y = mx + 3 -/
def LineEquation (m x : ℚ) : ℚ := m * x + 3

/-- The condition that the line passes through no lattice points in the given range -/
def NoLatticePoints (m : ℚ) : Prop :=
  ∀ x y, 0 < x → x ≤ 150 → LatticePoint x y → y ≠ LineEquation m x

/-- The theorem statement -/
theorem largest_b_value :
  (∀ m, (1/2 : ℚ) < m → m < 76/151 → NoLatticePoints m) ∧
  ¬(∀ b, b > 76/151 → ∀ m, (1/2 : ℚ) < m → m < b → NoLatticePoints m) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_b_value_l1171_117109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_30_factorial_l1171_117114

theorem distinct_prime_factors_of_30_factorial : 
  (Finset.filter (fun p => Nat.Prime p ∧ p ≤ 30) (Finset.range 31)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_30_factorial_l1171_117114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_pure_imaginary_z_in_second_quadrant_l1171_117147

/-- Given m ∈ ℝ, z is a complex number defined as z = (m-2)/(m-1) + (m^2+2m-3)i -/
noncomputable def z (m : ℝ) : ℂ := (m - 2) / (m - 1) + (m^2 + 2*m - 3) * Complex.I

/-- z belongs to ℝ if and only if m = -3 -/
theorem z_is_real (m : ℝ) : z m ∈ Set.range Complex.ofReal ↔ m = -3 := by sorry

/-- z is a pure imaginary number if and only if m = 2 -/
theorem z_is_pure_imaginary (m : ℝ) : z m ∈ {w : ℂ | w.re = 0 ∧ w.im ≠ 0} ↔ m = 2 := by sorry

/-- z is in the second quadrant if and only if 1 < m < 2 -/
theorem z_in_second_quadrant (m : ℝ) : z m ∈ {w : ℂ | w.re < 0 ∧ w.im > 0} ↔ 1 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_pure_imaginary_z_in_second_quadrant_l1171_117147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_log_l1171_117135

/-- Given a function y = 4a^(x-9) - 1 where a > 0 and a ≠ 1, 
    which always passes through point A (m, n), prove that logₘn = 1/2 -/
theorem exponential_function_log (a : ℝ) (m n : ℝ) 
  (ha : a > 0 ∧ a ≠ 1)
  (h_point : ∀ x : ℝ, 4 * a^(x - 9) - 1 = n ↔ x = m) : 
  Real.log n / Real.log m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_log_l1171_117135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_domain_l1171_117140

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.exp (x * Real.log 2) - 4 else Real.exp (-x * Real.log 2) - 4

-- State the theorem
theorem f_negative_domain :
  (∀ x : ℝ, f (-x) = f x) →  -- f is even
  (∀ x ≥ 0, f x = Real.exp (x * Real.log 2) - 4) →  -- f(x) = 2^x - 4 for x ≥ 0
  {x : ℝ | f x < 0 ∧ x < 0} = Set.Ioo (-2) 0 := by 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_domain_l1171_117140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_angle_satisfying_conditions_l1171_117117

theorem unique_angle_satisfying_conditions :
  ∃! x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi ∧ Real.sin x = -0.65 ∧ Real.cos x < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_angle_satisfying_conditions_l1171_117117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1171_117176

/-- Curve C defined by parametric equations --/
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sin α + Real.cos α, Real.sin α - Real.cos α)

/-- Line l defined by Cartesian equation --/
def line_l (x y : ℝ) : Prop := x - y + 1/2 = 0

/-- Theorem stating the length of chord AB --/
theorem chord_length :
  ∃ (A B : ℝ × ℝ) (α₁ α₂ : ℝ),
    curve_C α₁ = A ∧
    curve_C α₂ = B ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 30/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1171_117176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1171_117108

/-- Given a hyperbola and a line with no intersection, prove the eccentricity range -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (no_intersection : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → y ≠ 2*x) :
  let e := Real.sqrt (1 + (b/a)^2)
  1 < e ∧ e ≤ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1171_117108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_2AB_plus_AC_cosine_angle_AB_AC_l1171_117183

-- Define the points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)
def C : ℝ × ℝ := (2, 5)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- Define vector operations
def vectorAdd (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vectorScale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define dot product
def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem 1: Magnitude of 2AB + AC
theorem magnitude_2AB_plus_AC :
  magnitude (vectorAdd (vectorScale 2 AB) AC) = 5 * Real.sqrt 2 := by sorry

-- Theorem 2: Cosine of angle between AB and AC
theorem cosine_angle_AB_AC :
  dotProduct AB AC / (magnitude AB * magnitude AC) = 2 * Real.sqrt 13 / 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_2AB_plus_AC_cosine_angle_AB_AC_l1171_117183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_distribution_count_l1171_117198

/-- Represents a valid distribution of tickets to people -/
structure TicketDistribution where
  tickets : Fin 5 → Fin 4
  valid : ∀ i : Fin 4, ∃ t : Fin 5, tickets t = i
  consecutive : ∀ i : Fin 4, ∀ t1 t2 : Fin 5, 
    tickets t1 = i → tickets t2 = i → t1.val + 1 = t2.val ∨ t2.val + 1 = t1.val ∨ t1 = t2
  at_most_two : ∀ i : Fin 4, (Finset.filter (λ t : Fin 5 => tickets t = i) Finset.univ).card ≤ 2

/-- The number of valid ticket distributions -/
def num_distributions : Nat := sorry

theorem ticket_distribution_count : num_distributions = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_distribution_count_l1171_117198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_greater_than_negative_one_by_two_l1171_117192

theorem number_greater_than_negative_one_by_two : 
  ∃ x : ℤ, x > -1 ∧ x - (-1) = 2 := by
  use 1
  constructor
  · simp
  · simp


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_greater_than_negative_one_by_two_l1171_117192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_flowers_killed_l1171_117151

/-- Represents the number of flowers of each color --/
structure FlowerCount where
  red : ℕ
  yellow : ℕ
  orange : ℕ
  purple : ℕ

/-- Represents the problem setup --/
structure FloristProblem where
  seedsPerColor : ℕ
  flowersPerBouquet : ℕ
  totalBouquets : ℕ
  killedFlowers : FlowerCount

/-- The main theorem to prove --/
theorem red_flowers_killed (problem : FloristProblem)
    (h1 : problem.seedsPerColor = 125)
    (h2 : problem.flowersPerBouquet = 9)
    (h3 : problem.totalBouquets = 36)
    (h4 : problem.killedFlowers.yellow = 61)
    (h5 : problem.killedFlowers.orange = 30)
    (h6 : problem.killedFlowers.purple = 40) :
    problem.killedFlowers.red = 45 := by
  sorry

#check red_flowers_killed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_flowers_killed_l1171_117151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_product_l1171_117164

theorem power_of_two_product (x y : ℕ) : 
  (∃ k : ℕ, (x + y) * (x * y + 1) = 2^k) ↔ 
  (∃ a : ℕ, (x = 1 ∧ y = 2^a - 1)) ∨
  (∃ b : ℕ, (x = 2^b - 1 ∧ y = 2^b + 1)) ∨
  (∃ c : ℕ, (x = 2^c + 1 ∧ y = 2^c - 1)) ∨
  (∃ d : ℕ, (x = 2^d - 1 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_product_l1171_117164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_round_trip_time_l1171_117163

/-- Calculates the time taken to row to a place and back given the rowing speed in still water, 
    river speed, and total distance traveled. -/
noncomputable def time_to_row_round_trip (rowing_speed : ℝ) (river_speed : ℝ) (total_distance : ℝ) : ℝ :=
  let upstream_speed := rowing_speed - river_speed
  let downstream_speed := rowing_speed + river_speed
  let one_way_distance := total_distance / 2
  (one_way_distance / upstream_speed) + (one_way_distance / downstream_speed)

/-- Theorem stating that given specific conditions, the time to row round trip is 1 hour. -/
theorem row_round_trip_time :
  time_to_row_round_trip 8 2 7.5 = 1 := by
  -- Unfold the definition of time_to_row_round_trip
  unfold time_to_row_round_trip
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_round_trip_time_l1171_117163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1171_117149

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add this case for n = 0
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) / (1 + 2 * sequence_a (n + 1))

theorem sequence_a_formula (n : ℕ) (h : n ≥ 1) : 
  sequence_a n = 1 / (2 * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1171_117149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_quadratic_equation_solution_l1171_117193

-- Problem 1
theorem simplify_expression (x : ℝ) (hx : x > 0) :
  (2/3) * Real.sqrt (9*x) + 6 * Real.sqrt (x/4) - x * Real.sqrt (1/x) = 4 * Real.sqrt x :=
sorry

-- Problem 2
theorem quadratic_equation_solution :
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 3 ∧ x₂ = 2 - Real.sqrt 3 ∧
  x₁^2 - 4*x₁ + 1 = 0 ∧ x₂^2 - 4*x₂ + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_quadratic_equation_solution_l1171_117193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1171_117168

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 2

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ k : ℤ, ∃ c : ℝ, c = k * Real.pi / 2 + Real.pi / 6 ∧ ∀ x : ℝ, f (c + x) = f (c - x)) ∧
  (∀ k : ℤ, ∀ x : ℝ, k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12 →
    ∀ y : ℝ, k * Real.pi - Real.pi / 12 ≤ y ∧ y < x → f y < f x) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ 1) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ -Real.sqrt 3 / 2) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 1) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1171_117168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_zero_floor_l1171_117139

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - Real.cos x + 4 / Real.cos x

theorem smallest_zero_floor (s : ℝ) : 
  (∀ x, 0 < x → x < s → g x ≠ 0) →
  g s = 0 →
  s > 0 →
  ⌊s⌋ = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_zero_floor_l1171_117139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_fund_interest_cents_l1171_117148

/-- Represents the simple interest scenario for the Local Community Library Fund --/
structure LibraryFund where
  total_after_interest : ℚ
  annual_rate : ℚ
  time_months : ℕ

/-- Calculates the interest credited in cents --/
def interest_cents (fund : LibraryFund) : ℕ :=
  let principal : ℚ := fund.total_after_interest / (1 + fund.annual_rate * (fund.time_months / 12))
  let interest : ℚ := fund.total_after_interest - principal
  (interest * 100).floor.toNat

/-- Theorem stating that the interest credited in cents is 43 --/
theorem library_fund_interest_cents :
  let fund := LibraryFund.mk (307.80) (6/100) 3
  interest_cents fund = 43 := by sorry

#eval interest_cents (LibraryFund.mk (307.80) (6/100) 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_fund_interest_cents_l1171_117148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_statements_l1171_117194

-- Define the basic geometric objects
variable (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] [CompleteSpace P]
variable (L : Set P) -- Line
variable (Pl : Set P) -- Plane

-- Define the statements
def interior_angles_equal (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] (L : Set P) : Prop :=
  ∀ (a b : P), a ∈ L → b ∈ L → ∃ (c : P), c ∉ L ∧ (∃ (angle : ℝ), True) -- Placeholder for angle equality

def shortest_segment_perpendicular (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] (L : Set P) (p : P) : Prop :=
  p ∉ L → ∃ (q : P), q ∈ L ∧ ∀ (r : P), r ∈ L → dist p q ≤ dist p r →
    (∃ (angle : ℝ), angle = Real.pi / 2) -- Placeholder for perpendicularity

def vertical_angles_equal (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] : Prop :=
  ∀ (l₁ l₂ : Set P), (∃ (p : P), p ∈ l₁ ∧ p ∈ l₂) → ∀ (a b c d : P),
    a ∈ l₁ ∧ b ∈ l₁ ∧ c ∈ l₂ ∧ d ∈ l₂ →
    (∃ (angle : ℝ), True) -- Placeholder for angle equality

def non_intersecting_parallel (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] (Pl : Set P) : Prop :=
  ∀ (l₁ l₂ : Set P), l₁ ⊆ Pl ∧ l₂ ⊆ Pl → (¬∃ (p : P), p ∈ l₁ ∧ p ∈ l₂) → 
    (∃ (v : P), ∀ (p q : P), p ∈ l₁ → q ∈ l₂ → ∃ (t : ℝ), q = p + t • v) -- Placeholder for parallelism

-- Theorem stating that only the first statement is not always true
theorem geometric_statements :
  (¬ ∀ (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] (L : Set P), interior_angles_equal P L) ∧
  (∀ (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] (L : Set P), ∀ p, shortest_segment_perpendicular P L p) ∧
  (∀ (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P], vertical_angles_equal P) ∧
  (∀ (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] (Pl : Set P), non_intersecting_parallel P Pl) :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_statements_l1171_117194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_sequences_sum_ten_l1171_117177

/-- Number of sequences with sum 10 -/
def num_sequences (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else num_sequences (n - 1) + num_sequences (n - 2)

/-- The sequence elements are either 1 or 2 -/
def valid_sequence (s : List ℕ) : Bool :=
  s.all (λ x => x = 1 ∨ x = 2)

/-- The sum of the sequence elements is 10 -/
def sum_is_ten (s : List ℕ) : Bool :=
  s.sum = 10

theorem num_sequences_sum_ten :
  num_sequences 10 = (List.filter (λ s => valid_sequence s ∧ sum_is_ten s) (List.sublists (List.range 11))).length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_sequences_sum_ten_l1171_117177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tyler_aquarium_animals_l1171_117130

theorem tyler_aquarium_animals
  (total_aquariums : ℕ)
  (first_group_aquariums : ℕ)
  (second_group_aquariums : ℕ)
  (third_group_aquariums : ℕ)
  (first_group_animals : ℕ)
  (second_group_animals : ℕ)
  (third_group_animals : ℕ)
  (h1 : total_aquariums = first_group_aquariums + second_group_aquariums + third_group_aquariums)
  (h2 : total_aquariums = 15)
  (h3 : first_group_aquariums = 8)
  (h4 : second_group_aquariums = 5)
  (h5 : third_group_aquariums = 2)
  (h6 : first_group_animals = 128)
  (h7 : second_group_animals = 85)
  (h8 : third_group_animals = 155) :
  first_group_aquariums * first_group_animals +
  second_group_aquariums * second_group_animals +
  third_group_aquariums * third_group_animals = 1759 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tyler_aquarium_animals_l1171_117130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waffle_bowl_banana_split_difference_total_scoops_correct_l1171_117125

/-- Ice cream order types -/
inductive IceCreamOrder
  | BananaSplit
  | WaffleBowl
  | SingleCone
  | DoubleCone

/-- Number of scoops for each ice cream order -/
def scoops : IceCreamOrder → ℕ
  | IceCreamOrder.SingleCone => 1
  | IceCreamOrder.BananaSplit => 3
  | IceCreamOrder.DoubleCone => 2
  | IceCreamOrder.WaffleBowl => 4

/-- Total number of scoops served -/
def totalScoops : ℕ := 10

theorem waffle_bowl_banana_split_difference :
  scoops IceCreamOrder.WaffleBowl - scoops IceCreamOrder.BananaSplit = 1 :=
by
  -- Unfold the definitions
  rw [scoops, scoops]
  -- Perform the subtraction
  norm_num

theorem total_scoops_correct :
  totalScoops = scoops IceCreamOrder.BananaSplit + scoops IceCreamOrder.WaffleBowl +
                scoops IceCreamOrder.SingleCone + scoops IceCreamOrder.DoubleCone :=
by
  -- Unfold the definitions
  rw [totalScoops, scoops, scoops, scoops, scoops]
  -- Perform the addition
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_waffle_bowl_banana_split_difference_total_scoops_correct_l1171_117125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_problem_l1171_117188

theorem floor_ceil_fraction_problem : 
  ⌊⌈((12:ℝ)/5)^2⌉ + 19/5⌋ = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_problem_l1171_117188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1171_117122

def a : ℕ → ℚ
| 0 => 1
| n + 1 => a n / (1 + a n)

theorem sequence_properties :
  (a 3 = 1/4) ∧
  (∀ n : ℕ, n > 0 → a (n - 1) = 1 / n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1171_117122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_numerator_is_m_l1171_117105

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a stack configuration -/
structure StackConfig where
  num_2ft : ℕ
  num_3ft : ℕ
  num_5ft : ℕ

def crate_dimensions : CrateDimensions := ⟨2, 3, 5⟩

def total_crates : ℕ := 15

def target_height : ℕ := 50

def min_5ft_crates : ℕ := 2

/-- Calculates the total number of possible stack arrangements -/
def total_arrangements : ℕ := 3^total_crates

/-- Checks if a stack configuration is valid -/
def is_valid_config (config : StackConfig) : Prop :=
  config.num_2ft + config.num_3ft + config.num_5ft = total_crates ∧
  2 * config.num_2ft + 3 * config.num_3ft + 5 * config.num_5ft = target_height ∧
  config.num_5ft ≥ min_5ft_crates

/-- Calculates the number of arrangements for a given configuration -/
def arrangements_for_config (config : StackConfig) : ℕ :=
  Nat.choose total_crates config.num_5ft *
  Nat.choose (total_crates - config.num_5ft) config.num_3ft

/-- Calculates the total number of valid arrangements -/
noncomputable def valid_arrangements : ℕ :=
  sorry -- Sum of arrangements_for_config for all valid configurations

/-- The probability of the stack being exactly 50ft tall -/
noncomputable def probability : ℚ :=
  ↑valid_arrangements / ↑total_arrangements

/-- m is the numerator of the probability in its simplest form -/
noncomputable def m : ℕ := 
  (probability * ↑probability.den).num.natAbs

theorem probability_numerator_is_m :
  (probability * ↑probability.den).num.natAbs = m := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_numerator_is_m_l1171_117105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_true_q_false_l1171_117112

-- Define the sine function
noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

-- Statement for proposition p
theorem p_true : ∀ x : ℝ, f (x - Real.pi / 6) = f (-x - Real.pi / 6) := by sorry

-- Statement for proposition q
theorem q_false : ∃ a b : ℝ, (2 : ℝ) ^ a < (2 : ℝ) ^ b ∧ Real.log a ≥ Real.log b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_true_q_false_l1171_117112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_correct_l1171_117181

/-- Returns the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + digitSum (n / 10)

/-- Checks if a number satisfies the condition for all multipliers -/
def satisfiesCondition (n : ℕ) : Prop :=
  ∀ m : Fin 8, digitSum (((m : ℕ) + 2) * n) = digitSum n

/-- The set of two-digit numbers satisfying the condition -/
def solutionSet : Set ℕ := {18, 45, 90}

/-- Theorem stating that the solution set is correct -/
theorem solution_is_correct :
  ∀ n : ℕ, n ≥ 10 ∧ n ≤ 99 →
    (n ∈ solutionSet ↔ satisfiesCondition n) := by sorry

#check solution_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_correct_l1171_117181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_extreme_points_l1171_117126

open Real

-- Define the function f(x)
noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 1/2 * x^2 + x - x * log x

-- Define the derivative of f(x)
noncomputable def f_derivative (a x : ℝ) : ℝ := 3 * a * x^2 - x - log x

-- Define the function g(x) used in the analysis
noncomputable def g (x : ℝ) : ℝ := (x + log x) / x^2

-- State the theorem
theorem f_has_two_extreme_points (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_derivative a x₁ = 0 ∧ f_derivative a x₂ = 0) ↔ 0 < a ∧ a < 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_extreme_points_l1171_117126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfying_equation_l1171_117116

theorem function_satisfying_equation (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f (x*y - x)) + f (x + y) = y * f x + f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfying_equation_l1171_117116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_micheal_work_time_l1171_117107

/-- Represents the time (in days) it takes for a person to complete the work alone -/
structure WorkTime where
  days : ℝ
  days_pos : days > 0

/-- Represents the portion of work completed -/
abbrev WorkPortion : Type := ℝ

/-- The rate at which work is completed (portion per day) -/
noncomputable def workRate (wt : WorkTime) : ℝ := 1 / wt.days

theorem micheal_work_time 
  (michael_adam : WorkTime) -- Time for Michael and Adam to complete work together
  (adam : WorkTime) -- Time for Adam to complete work alone
  (h1 : michael_adam.days = 20)
  (h2 : workRate michael_adam * 15 + workRate adam * 10 = 1)
  : ∃ (michael : WorkTime), michael.days = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_micheal_work_time_l1171_117107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_result_l1171_117169

def initial_number : ℕ := 100000

def operation (n : ℚ) : ℚ :=
  ((n / 2) / 2) / 2 * 3 * 3

def sequence_step (n : ℕ) : ℚ :=
  match n with
  | 0 => initial_number
  | n + 1 => operation (sequence_step n)

theorem final_result :
  sequence_step 15 = (15 : ℚ)^5 * 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_result_l1171_117169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l1171_117106

theorem cosine_identity (α : ℝ) : 
  (Real.cos α)^2 + (Real.cos (α + π/3))^2 - (Real.cos α) * (Real.cos (α + π/3)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l1171_117106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorizable_b_l1171_117155

-- Define x as a polynomial variable
variable (x : polynomial ℤ)

def is_factorizable (b : ℤ) : Prop :=
  ∃ (p q : ℤ), x^2 + b*x + (2052 : ℤ) = (x + p) * (x + q)

theorem smallest_factorizable_b :
  (is_factorizable 132) ∧
  (∀ b : ℤ, b < 132 → ¬(is_factorizable b)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorizable_b_l1171_117155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_orthocenter_product_l1171_117113

/-- Helper function to calculate the area of a triangle given its side lengths using Heron's formula -/
noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Given a triangle with sides a, b, and c, where m is an altitude and m₁ is the segment of this
    altitude from the vertex to the orthocenter, prove that m · m₁ = (b² + c² - a²) / 2 -/
theorem triangle_altitude_orthocenter_product (a b c m m₁ : ℝ) 
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
    (h_m : m > 0)
    (h_m₁ : m₁ > 0)
    (h_altitude : m = 2 * (area a b c) / a)
    (h_orthocenter : m₁ = a * (b^2 + c^2 - a^2) / (4 * area a b c)) :
  m * m₁ = (b^2 + c^2 - a^2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitude_orthocenter_product_l1171_117113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_distribution_l1171_117160

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 891) (h2 : pencils = 810) : 
  Nat.gcd pens pencils = 81 :=
by
  rw [h1, h2]
  norm_num

#eval Nat.gcd 891 810

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_distribution_l1171_117160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1171_117179

theorem range_of_m (α β m : ℝ) : 
  α ∈ Set.Icc (-π/2) (π/2) →
  β ∈ Set.Icc (-π/2) (π/2) →
  α + β < 0 →
  Real.sin α = 1 - m →
  Real.sin β = 1 - m^2 →
  m ∈ Set.Ioo 1 (Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1171_117179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_planes_from_parallel_lines_l1171_117115

/-- A line in 3D space -/
structure Line3D where
  -- You might define this structure with appropriate fields,
  -- but for this problem, we only need it as an abstract type

/-- A plane in 3D space -/
structure Plane3D where
  -- Similarly, we only need this as an abstract type for our problem

/-- Predicate to check if two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of parallel lines

/-- Function to create a plane from two lines -/
def plane_from_lines (l1 l2 : Line3D) : Plane3D :=
  sorry -- Definition of how to create a plane from two lines

theorem max_planes_from_parallel_lines (l1 l2 l3 : Line3D) 
  (h12 : parallel l1 l2) (h23 : parallel l2 l3) (h13 : parallel l1 l3) :
  ∃ (p1 p2 p3 : Plane3D), 
    (p1 = plane_from_lines l1 l2 ∧ 
     p2 = plane_from_lines l2 l3 ∧ 
     p3 = plane_from_lines l1 l3) ∧
    ∀ (p : Plane3D), p = p1 ∨ p = p2 ∨ p = p3 ∨ 
      (∀ (i j : Line3D), (i = l1 ∨ i = l2 ∨ i = l3) → (j = l1 ∨ j = l2 ∨ j = l3) → 
        p ≠ plane_from_lines i j) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_planes_from_parallel_lines_l1171_117115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_N_for_sequence_inequality_l1171_117120

noncomputable def sequence_a (a₀ : ℝ) : ℕ → ℝ
  | 0 => a₀
  | n + 1 => (sequence_a a₀ n + 1) / 2

noncomputable def sequence_b (b₀ k : ℝ) : ℕ → ℝ
  | 0 => b₀
  | n + 1 => (sequence_b b₀ k n) ^ k

theorem exists_N_for_sequence_inequality (k a₀ b₀ : ℝ) 
  (h_k : 0 < k ∧ k < 1/2) 
  (h_a₀ : 0 < a₀ ∧ a₀ < 1) 
  (h_b₀ : 0 < b₀ ∧ b₀ < 1) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → sequence_a a₀ n < sequence_b b₀ k n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_N_for_sequence_inequality_l1171_117120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_plot_ratio_l1171_117141

/-- Represents a rectangular plot with given area and width. -/
structure RectangularPlot where
  area : ℝ
  width : ℝ

/-- Calculates the length of a rectangular plot. -/
noncomputable def length (plot : RectangularPlot) : ℝ := plot.area / plot.width

/-- Calculates the ratio of length to width for a rectangular plot. -/
noncomputable def lengthToWidthRatio (plot : RectangularPlot) : ℝ := length plot / plot.width

/-- Theorem: For a rectangular plot with area 432 sq meters and width 12 meters, 
    the ratio of its length to its width is 3:1. -/
theorem rectangular_plot_ratio : 
  let plot := RectangularPlot.mk 432 12
  lengthToWidthRatio plot = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_plot_ratio_l1171_117141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_select_defective_items_l1171_117111

/-- The number of ways to select at least 1 defective item from a box of 16 items (14 qualified and 2 defective) when 3 items are selected. -/
theorem select_defective_items (total : Nat) (qualified : Nat) (defective : Nat) (select : Nat)
  (h_total : total = 16)
  (h_qualified : qualified = 14)
  (h_defective : defective = 2)
  (h_select : select = 3)
  (h_sum : total = qualified + defective) :
  Nat.choose defective 1 * Nat.choose qualified 2 + Nat.choose defective 2 * Nat.choose qualified 1 =
  Nat.choose total select - Nat.choose qualified select ∧
  Nat.choose total select - Nat.choose qualified select =
  Nat.choose defective 1 * Nat.choose (total - 1) (select - 1) - Nat.choose defective 2 * Nat.choose qualified 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_select_defective_items_l1171_117111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_returns_to_start_l1171_117153

/-- Represents the rotation angle of 45 degrees counterclockwise -/
noncomputable def ω : ℂ := Complex.exp (Complex.I * Real.pi / 4)

/-- The position of the bee after n steps -/
noncomputable def bee_position : ℕ → ℂ
  | 0 => 0
  | n + 1 => bee_position n + (n + 1 : ℂ) * ω^n

/-- The theorem stating that the bee returns to the starting point after 7 steps -/
theorem bee_returns_to_start : Complex.abs (bee_position 7) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_returns_to_start_l1171_117153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_l1171_117156

theorem trigonometric_values (θ : Real) 
  (h1 : θ > Real.pi / 2) 
  (h2 : θ < Real.pi) 
  (h3 : Real.sin θ = 4 / 5) : 
  Real.cos θ = -3 / 5 ∧ Real.sin (θ + Real.pi / 3) = (4 - 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_l1171_117156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bandit_with_many_showdowns_l1171_117136

/-- Represents a group of bandits and their showdowns -/
structure BanditGroup where
  size : ℕ
  has_met : Fin size → Fin size → Bool
  met_once : ∀ i j, i ≠ j → has_met i j = true
  no_self_meet : ∀ i, has_met i i = false

/-- The number of showdowns a bandit has participated in -/
def showdown_count (bg : BanditGroup) (i : Fin bg.size) : ℕ :=
  (Finset.univ.filter (λ j => bg.has_met i j)).card

theorem bandit_with_many_showdowns (bg : BanditGroup) (h : bg.size = 50) :
  ∃ i : Fin bg.size, showdown_count bg i ≥ 8 := by
  sorry

#check bandit_with_many_showdowns

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bandit_with_many_showdowns_l1171_117136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_a3_value_l1171_117199

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the relationship between a_3 and the sum of a_1, a_3, and a_5 -/
theorem cos_a3_value (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) 
    (h_sum : a 1 + a 3 + a 5 = 2 * Real.pi) : 
  ∃ k : ℝ, Real.cos (a 3) = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_a3_value_l1171_117199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AH_length_l1171_117102

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem setup
variable (Γ Ω O₁ O₂ : Circle)
variable (P A B X Y F T M H : Point)

-- Define the given conditions
axiom internally_tangent : Γ.center = (P.x, P.y) ∧ Ω.center = (P.x, P.y) ∧ Γ.radius < Ω.radius
axiom A_B_on_Ω : (A.x - Ω.center.1)^2 + (A.y - Ω.center.2)^2 = Ω.radius^2 ∧
                 (B.x - Ω.center.1)^2 + (B.y - Ω.center.2)^2 = Ω.radius^2
axiom X_Y_on_Γ : (X.x - Γ.center.1)^2 + (X.y - Γ.center.2)^2 = Γ.radius^2 ∧
                 (Y.x - Γ.center.1)^2 + (Y.y - Γ.center.2)^2 = Γ.radius^2
axiom O₁_diameter : O₁.radius = Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) / 2
axiom O₂_diameter : O₂.radius = Real.sqrt ((X.x - Y.x)^2 + (X.y - Y.y)^2) / 2
axiom F_foot_of_Y : (F.x - X.x) * (Y.x - X.x) + (F.y - X.y) * (Y.y - X.y) = 0
axiom TM_common_tangent : (T.x - O₁.center.1)^2 + (T.y - O₁.center.2)^2 = O₁.radius^2 ∧
                          (M.x - O₂.center.1)^2 + (M.y - O₂.center.2)^2 = O₂.radius^2
axiom H_orthocenter : (H.x - A.x) * (B.y - A.y) = (H.y - A.y) * (B.x - A.x) ∧
                      (H.x - B.x) * (P.y - B.y) = (H.y - B.y) * (P.x - B.x)
axiom PF_length : Real.sqrt ((P.x - F.x)^2 + (P.y - F.y)^2) = 12
axiom FX_length : Real.sqrt ((F.x - X.x)^2 + (F.y - X.y)^2) = 15
axiom TM_length : Real.sqrt ((T.x - M.x)^2 + (T.y - M.y)^2) = 18
axiom PB_length : Real.sqrt ((P.x - B.x)^2 + (P.y - B.y)^2) = 50

-- Theorem to prove
theorem AH_length (Γ Ω O₁ O₂ : Circle) (P A B X Y F T M H : Point) :
  Real.sqrt ((A.x - H.x)^2 + (A.y - H.y)^2) = 750 / Real.sqrt 481 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AH_length_l1171_117102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_lines_condition_l1171_117154

/-- Two lines in 3D space -/
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- Check if two lines are coplanar -/
def are_coplanar (l1 l2 : Line3D) : Prop :=
  ∃ (t u : ℝ), ∀ i : Fin 3, l1.point i + t * l1.direction i = l2.point i + u * l2.direction i

/-- The main theorem -/
theorem coplanar_lines_condition (p : ℝ) : 
  let l1 : Line3D := ⟨(![3, 2, 6]), (![- p, 1, 2])⟩
  let l2 : Line3D := ⟨(![2, 5, 7]), (![1, p, 3])⟩
  are_coplanar l1 l2 ↔ p = 1/2 ∨ p = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_lines_condition_l1171_117154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l1171_117195

theorem trigonometric_expression_value (m : ℝ) (α : ℝ) :
  m < 0 →
  ∃ (x y : ℝ), x = m ∧ y = -2*m ∧ (x * Real.cos α = y * Real.sin α) →
  1 / (2 * Real.sin α * Real.cos α + (Real.cos α) ^ 2) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l1171_117195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l1171_117197

-- Define the point P on the positive half of the x-axis
noncomputable def P (x : ℝ) : ℝ × ℝ × ℝ := (x, 0, 0)

-- Define the point Q
noncomputable def Q : ℝ × ℝ × ℝ := (0, Real.sqrt 2, 3)

-- State the theorem
theorem point_coordinates (x : ℝ) :
  x > 0 →  -- P is on the positive half of the x-axis
  Real.sqrt ((x - 0)^2 + (0 - Real.sqrt 2)^2 + (0 - 3)^2) = 2 * Real.sqrt 3 →  -- Distance between P and Q is 2√3
  P x = (1, 0, 0) :=  -- Conclusion: coordinates of P are (1, 0, 0)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l1171_117197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1171_117145

theorem inequality_solution_set (x : ℝ) : 
  x + 2 / (x + 1) > 2 ↔ x ∈ Set.Ioi (-1) ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1171_117145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_triple_volume_l1171_117127

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

def sphere_diameter (r : ℝ) : ℝ := 2 * r

theorem sphere_diameter_triple_volume (r₁ r₂ : ℝ) :
  r₁ = 12 →
  sphere_volume r₂ = 3 * sphere_volume r₁ →
  sphere_diameter r₂ = 24 * (3 : ℝ) ^ (1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_triple_volume_l1171_117127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_side_significant_digits_l1171_117180

-- Define the area of the square garden
def garden_area : ℝ := 2.3049

-- Define the side length of the square garden
noncomputable def side_length : ℝ := Real.sqrt garden_area

-- Function to count significant digits
-- This is a placeholder function and would need a proper implementation
def count_significant_digits (x : ℝ) : ℕ := sorry

-- Theorem statement
theorem garden_side_significant_digits :
  count_significant_digits side_length = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_side_significant_digits_l1171_117180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_relation_l1171_117119

/-- Given a triangle ABC with centroid G and an arbitrary point P,
    prove that PA² + PB² + PC² = 3 · PG² + GA² + GB² + GC² -/
theorem centroid_distance_relation (A B C P : EuclideanSpace ℝ (Fin 2)) :
  let G := (1/3 : ℝ) • (A + B + C)
  ‖P - A‖^2 + ‖P - B‖^2 + ‖P - C‖^2 =
  3 * ‖P - G‖^2 + ‖G - A‖^2 + ‖G - B‖^2 + ‖G - C‖^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_relation_l1171_117119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1171_117118

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side length opposite to A
  b : ℝ  -- Side length opposite to B
  c : ℝ  -- Side length opposite to C

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.a * Real.sin t.B + t.b * Real.cos t.A = 0 ∧
  t.a = Real.sqrt 2 ∧
  t.b = 1

-- Helper function to calculate area (not part of the proof, just for completeness)
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem triangle_proof (t : Triangle) (h : TriangleProperties t) :
  t.A = 3 * π / 4 ∧
  area t = (Real.sqrt 3 - 1) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l1171_117118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1171_117143

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (a*x^2 - x + a) / Real.log a

-- Define the theorem
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : (p a ∧ ¬(q a)) ∨ (¬(p a) ∧ q a)) : 
  a ∈ Set.Ioc 0 (1/2) ∪ Set.Ioi 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1171_117143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_union_B_l1171_117184

-- Define the sets A and B
def A : Set ℝ := {x | (x - 2) * (x + 1) ≤ 0}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 3}

-- State the theorem
theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = Set.Iic (-1) ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_union_B_l1171_117184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_monochromatic_coloring_l1171_117137

def M : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 2010}

def is_arithmetic_progression (s : List ℕ) : Prop :=
  s.length = 9 ∧ ∃ a d : ℕ, ∀ i : Fin 9, s[i.val]? = some (a + i.val * d)

def is_monochromatic (f : ℕ → Fin 5) (s : List ℕ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → f x = f y

theorem exists_non_monochromatic_coloring :
  ∃ f : ℕ → Fin 5, ∀ s : List ℕ,
    (∀ x, x ∈ s → x ∈ M) →
    is_arithmetic_progression s →
    ¬is_monochromatic f s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_monochromatic_coloring_l1171_117137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_reduced_price_l1171_117150

/-- The reduced price of a shirt, given the original price and discount percentage -/
noncomputable def reduced_price (original_price : ℚ) (discount_percentage : ℚ) : ℚ :=
  original_price * (discount_percentage / 100)

/-- Theorem stating that the reduced price of a shirt is $6 -/
theorem shirt_reduced_price :
  reduced_price 24 25 = 6 := by
  -- Unfold the definition of reduced_price
  unfold reduced_price
  -- Perform the calculation
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_reduced_price_l1171_117150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_is_ten_minutes_l1171_117131

/-- Represents the race scenario between a tortoise and a hare -/
structure RaceScenario where
  raceDistance : ℚ
  tortoiseSpeed : ℚ
  hareInitialDistance : ℚ
  hareInitialTimeFraction : ℚ

/-- Calculates the total race time given a race scenario -/
def calculateRaceTime (scenario : RaceScenario) : ℚ :=
  scenario.raceDistance / scenario.tortoiseSpeed

/-- Theorem stating that the race time is 10 minutes for the given scenario -/
theorem race_time_is_ten_minutes (scenario : RaceScenario) 
  (h1 : scenario.raceDistance = 100)
  (h2 : scenario.tortoiseSpeed = 10)
  (h3 : scenario.hareInitialDistance = 50)
  (h4 : scenario.hareInitialTimeFraction = 1/4) : 
  calculateRaceTime scenario = 10 := by
  sorry

def main : IO Unit := do
  let scenario : RaceScenario := { 
    raceDistance := 100, 
    tortoiseSpeed := 10, 
    hareInitialDistance := 50, 
    hareInitialTimeFraction := 1/4 
  }
  IO.println s!"Race time: {calculateRaceTime scenario}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_is_ten_minutes_l1171_117131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_2010_l1171_117161

/-- Three strictly increasing sequences of positive integers -/
def a : ℕ → ℕ := sorry
def b : ℕ → ℕ := sorry
def c : ℕ → ℕ := sorry

/-- Properties of the sequences -/
axiom strictly_increasing_a : ∀ n, a (n + 1) > a n
axiom strictly_increasing_b : ∀ n, b (n + 1) > b n
axiom strictly_increasing_c : ∀ n, c (n + 1) > c n

/-- Every positive integer belongs to exactly one of the three sequences -/
axiom partition : ∀ m : ℕ, (∃ n, a n = m) ∨ (∃ n, b n = m) ∨ (∃ n, c n = m)
axiom unique_partition : ∀ m n, a m ≠ b n ∧ a m ≠ c n ∧ b m ≠ c n

/-- Conditions for every positive integer n -/
axiom condition_a : ∀ n, c (a n) = b n + 1
axiom condition_b : ∀ n, a (n + 1) > b n
axiom condition_c : ∀ n, Even (c (n + 1) * c n - (n + 1) * c (n + 1) - n * c n)

/-- The theorem to prove -/
theorem sequences_2010 :
  a 2010 = 4040100 ∧ b 2010 = 4044119 ∧ c 2010 = 2099 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequences_2010_l1171_117161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1171_117175

theorem problem_solution :
  (Real.sqrt 16 + ((-27) ^ (1/3 : ℝ)) + abs (Real.sqrt 3 - 2) = 3 - Real.sqrt 3) ∧
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → (16 * a^4 * b^3 - 12 * a^3 * b^2 + 4 * a * b) / (4 * a * b) = 4 * a^3 * b^2 - 3 * a^2 * b + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1171_117175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extremum_at_neg_one_g_minimum_value_l1171_117185

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (-x) + a * x - 1 / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (-x) + 2 * x

-- Theorem for part (I)
theorem f_extremum_at_neg_one (a : ℝ) :
  (∀ x < 0, HasDerivAt (f a) ((1 / x) + a + (1 / x^2)) x) →
  HasDerivAt (f a) 0 (-1) →
  a = 0 := by
  sorry

-- Theorem for part (II)
theorem g_minimum_value (a : ℝ) :
  a = 0 →
  ∀ x > 0, g a x ≥ 3 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extremum_at_neg_one_g_minimum_value_l1171_117185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l1171_117134

noncomputable def variance (s : Finset ℝ) (f : ℝ → ℝ) : ℝ :=
  let μ := (s.sum f) / s.card
  (s.sum (λ x => (f x - μ)^2)) / s.card

theorem variance_transformation (s : Finset ℝ) (f : ℝ → ℝ) (h : variance s f = 3) :
  variance s (λ x => 3 * (f x - 2)) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l1171_117134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_tangent_lines_l1171_117129

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x + 1

noncomputable def f_derivative (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem extreme_values :
  (∃ x : ℝ, f x = 8/3 ∧ f_derivative x = 0) ∧
  (∃ x : ℝ, f x = -8 ∧ f_derivative x = 0) := by
  sorry

theorem tangent_lines :
  ∃ (m₁ m₂ b₁ b₂ : ℝ),
    (m₁ * 0 + b₁ = 1 ∧ ∀ x, f x = m₁ * x + b₁ → f_derivative x = m₁) ∧
    (m₂ * 0 + b₂ = 1 ∧ ∀ x, f x = m₂ * x + b₂ → f_derivative x = m₂) ∧
    ((m₁ = 3 ∧ b₁ = 1) ∨ (m₁ = 15/4 ∧ b₁ = 1)) ∧
    ((m₂ = 3 ∧ b₂ = 1) ∨ (m₂ = 15/4 ∧ b₂ = 1)) ∧
    m₁ ≠ m₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_tangent_lines_l1171_117129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_f_dot_product_range_l1171_117128

-- Define the vectors OP and OQ
noncomputable def OP (x : ℝ) : ℝ × ℝ := (2 * Real.cos x + 1, Real.cos (2 * x) - Real.sin x + 1)
noncomputable def OQ (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

-- Define the dot product function f(x)
noncomputable def f (x : ℝ) : ℝ := (OP x).1 * (OQ x).1 + (OP x).2 * (OQ x).2

-- Statement 1: The smallest positive period of f(x) is 2π
theorem smallest_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi := by
  sorry

-- Statement 2: For x ∈ (0, 2π), OP · OQ < -1 if and only if x ∈ (π, 3π/2)
theorem dot_product_range :
  ∀ (x : ℝ), 0 < x → x < 2 * Real.pi →
  (f x < -1 ↔ Real.pi < x ∧ x < 3 * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_f_dot_product_range_l1171_117128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l1171_117190

-- Define the function for problem 1
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (3 - x)) / (x - 1)

-- Define the function for problem 2
def g (x : ℝ) : ℝ := -x^2 + 4*x - 2

-- Theorem for problem 1
theorem domain_of_f : 
  Set.range f = {y | ∃ x, x ≤ 3 ∧ x ≠ 1 ∧ y = f x} := by
  sorry

-- Theorem for problem 2
theorem range_of_g : 
  Set.image g (Set.Icc 1 4) = Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l1171_117190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_product_equals_63700992_l1171_117178

/-- The radius of the circle -/
def radius : ℝ := 3

/-- The number of equal arcs the circle is divided into -/
def num_arcs : ℕ := 8

/-- The number of division points (excluding A and B) -/
def num_points : ℕ := num_arcs - 1

/-- The complex number representing the rotation by one arc -/
noncomputable def θ : ℂ := Complex.exp (2 * Real.pi * Complex.I / num_arcs)

/-- The complex number representing point A -/
def A : ℂ := radius

/-- The complex number representing point B -/
def B : ℂ := -radius

/-- The complex number representing point Pₖ -/
noncomputable def P (k : ℕ) : ℂ := radius * θ^k

/-- The length of the chord APₖ -/
noncomputable def chord_AP (k : ℕ) : ℝ := Complex.abs (A - P k)

/-- The length of the chord BPₖ -/
noncomputable def chord_BP (k : ℕ) : ℝ := Complex.abs (B - P k)

/-- The product of all chord lengths -/
noncomputable def chord_product : ℝ :=
  (Finset.range num_points).prod (λ k => chord_AP k * chord_BP k)

theorem chord_product_equals_63700992 : chord_product = 63700992 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_product_equals_63700992_l1171_117178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_charge_equals_6_75_l1171_117167

/-- Taxi service pricing structure and trip details -/
structure TaxiTrip where
  initialFee : ℚ
  firstTwoMileRate : ℚ
  afterTwoMileRate : ℚ
  peakHourSurcharge : ℚ
  tripDistance : ℚ
  isPeakHour : Bool

/-- Calculate the total charge for a taxi trip -/
def calculateTotalCharge (trip : TaxiTrip) : ℚ :=
  let firstTwoMileCharge := 4 * trip.firstTwoMileRate
  let remainingDistance := max (trip.tripDistance - 2) 0
  let remainingCharge := (remainingDistance / (2/5)) * trip.afterTwoMileRate
  let peakHourCharge := if trip.isPeakHour then trip.peakHourSurcharge else 0
  trip.initialFee + firstTwoMileCharge + remainingCharge + peakHourCharge

/-- Theorem: The total charge for the given trip is $6.75 -/
theorem total_charge_equals_6_75 (trip : TaxiTrip) 
  (h1 : trip.initialFee = 205/100)
  (h2 : trip.firstTwoMileRate = 45/100)
  (h3 : trip.afterTwoMileRate = 35/100)
  (h4 : trip.peakHourSurcharge = 3/2)
  (h5 : trip.tripDistance = 18/5)
  (h6 : trip.isPeakHour = true) :
  calculateTotalCharge trip = 27/4 := by
  sorry

#eval calculateTotalCharge {
  initialFee := 205/100,
  firstTwoMileRate := 45/100,
  afterTwoMileRate := 35/100,
  peakHourSurcharge := 3/2,
  tripDistance := 18/5,
  isPeakHour := true
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_charge_equals_6_75_l1171_117167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_iff_l1171_117103

-- Define the function f(x) = ln x - x
noncomputable def f (x : ℝ) : ℝ := Real.log x - x

-- Theorem statement
theorem f_strictly_increasing_iff (x : ℝ) (h : x > 0) :
  StrictMono (fun y => f y) ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_iff_l1171_117103
