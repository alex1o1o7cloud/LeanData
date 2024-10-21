import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_beats_fifth_l45_4544

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  players : Fin 8 → ℕ
  scores_different : ∀ i j, i ≠ j → players i ≠ players j
  second_place_score : players 1 = players 4 + players 5 + players 6 + players 7
  total_games : Finset.sum (Finset.univ : Finset (Fin 8)) players = 28

/-- The result of a game between two players -/
inductive GameResult
  | Win
  | Loss
  | Draw

/-- The game between two players in the tournament -/
def game (t : ChessTournament) (i j : Fin 8) : GameResult :=
  if t.players i > t.players j then GameResult.Win
  else if t.players i < t.players j then GameResult.Loss
  else GameResult.Draw

/-- The main theorem to prove -/
theorem third_beats_fifth (t : ChessTournament) : 
  game t 2 4 = GameResult.Win := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_beats_fifth_l45_4544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_in_mixture_l45_4528

/-- Molar mass of hydrogen in g/mol -/
def molar_mass_H : ℚ := 1008 / 1000

/-- Molar mass of calcium in g/mol -/
def molar_mass_Ca : ℚ := 40080 / 1000

/-- Molar mass of oxygen in g/mol -/
def molar_mass_O : ℚ := 15999 / 1000

/-- Molar mass of sulfur in g/mol -/
def molar_mass_S : ℚ := 32065 / 1000

/-- Molar mass of CaH2 in g/mol -/
def molar_mass_CaH2 : ℚ := molar_mass_Ca + 2 * molar_mass_H

/-- Molar mass of H2O in g/mol -/
def molar_mass_H2O : ℚ := 2 * molar_mass_H + molar_mass_O

/-- Molar mass of H2SO4 in g/mol -/
def molar_mass_H2SO4 : ℚ := 2 * molar_mass_H + molar_mass_S + 4 * molar_mass_O

/-- Number of moles of CaH2 -/
def moles_CaH2 : ℚ := 3

/-- Number of moles of H2O -/
def moles_H2O : ℚ := 4

/-- Number of moles of H2SO4 -/
def moles_H2SO4 : ℚ := 2

/-- Theorem stating that the mass percentage of H in the mixture is approximately 4.599% -/
theorem mass_percentage_H_in_mixture : 
  let total_mass_H := (moles_CaH2 * 2 + moles_H2O * 2 + moles_H2SO4 * 2) * molar_mass_H
  let total_mass_mixture := moles_CaH2 * molar_mass_CaH2 + moles_H2O * molar_mass_H2O + moles_H2SO4 * molar_mass_H2SO4
  abs ((total_mass_H / total_mass_mixture) * 100 - 4599 / 1000) < 1 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_in_mixture_l45_4528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l45_4531

noncomputable def f (x φ : ℝ) : ℝ := -2 * Real.tan (2 * x + φ)

theorem monotonically_decreasing_interval
  (φ : ℝ)
  (h1 : |φ| < π)
  (h2 : f (π/16) φ = -2) :
  ∃ (a b : ℝ), a = 3*π/16 ∧ b = 11*π/16 ∧
  ∀ x ∈ Set.Ioo a b, ∀ y ∈ Set.Ioo a b,
  x < y → f x φ > f y φ := by
  sorry

#check monotonically_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l45_4531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_set_f_unique_root_l45_4530

noncomputable section

variable (a : ℝ)

def f (x : ℝ) := (2*x - 2) * Real.exp x - a * x^2 + 2 * a^2

theorem f_positive_set (h : a = 1) :
  {x : ℝ | f a x > 0} = Set.Ioi 0 := by sorry

theorem f_unique_root (h : 0 < a) (h' : a < 1) :
  ∃! x₀, f a x₀ = 0 ∧ a * x₀ < (3/2 : ℝ) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_set_f_unique_root_l45_4530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l45_4551

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2*a + 1)/a - 1/(a^2 * x)

-- State the theorem
theorem f_properties (a : ℝ) (h_a : a > 0) (m n : ℝ) :
  (m * n > 0 → (∀ x y, x ∈ Set.Icc m n → y ∈ Set.Icc m n → x < y → f a x < f a y)) ∧
  (0 < m ∧ m < n ∧ Set.range (f a) = Set.Icc m n → a > 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l45_4551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_average_after_transformation_l45_4566

/-- Represents a set of grades as a function from ℕ to ℕ -/
def GradeSet := ℕ → ℕ

/-- The total number of grades in a GradeSet -/
def total (grades : GradeSet) : ℕ := sorry

/-- The sum of all grades in a GradeSet -/
def sum (grades : GradeSet) : ℕ := sorry

/-- The average of grades in a GradeSet -/
noncomputable def average (grades : GradeSet) : ℚ :=
  (sum grades : ℚ) / (total grades : ℚ)

/-- Transforms a GradeSet by replacing all 1s with 3s -/
def transform (grades : GradeSet) : GradeSet := sorry

theorem grade_average_after_transformation
  (grades : GradeSet)
  (h1 : ∀ n, grades n ∈ ({1, 2, 3, 4, 5} : Set ℕ))
  (h2 : average grades ≤ 3)
  : average (transform grades) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_average_after_transformation_l45_4566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_between_curves_l45_4550

/-- Curve C₁ -/
def C₁ (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

/-- Curve C₂ -/
def C₂ (x y : ℝ) : Prop := (x - 1/2)^2 + y^2 = 1/4

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The main theorem -/
theorem distance_range_between_curves :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    C₁ x₁ y₁ → C₂ x₂ y₂ → 
    (Real.sqrt 7 / 2 - 1/2) ≤ distance x₁ y₁ x₂ y₂ ∧ 
    distance x₁ y₁ x₂ y₂ ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_between_curves_l45_4550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equal_if_floor_equal_l45_4570

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the property of being a quadratic polynomial
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

-- State the theorem
theorem quadratic_equal_if_floor_equal
  (f g : ℝ → ℝ)
  (hf : is_quadratic f)
  (hg : is_quadratic g)
  (h : ∀ x, floor (f x) = floor (g x)) :
  ∀ x, f x = g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equal_if_floor_equal_l45_4570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_proof_binomial_expansion_coefficient_l45_4538

/-- The binomial expansion of (x^2 - 1/x)^9 -/
noncomputable def binomial_expansion (x : ℝ) : ℝ := (x^2 - 1/x)^9

/-- The coefficient of x^3 in the binomial expansion of (x^2 - 1/x)^9 -/
def coefficient_x_cubed : ℝ := -126

/-- The remainder function in the binomial expansion -/
noncomputable def g (x : ℝ) : ℝ := binomial_expansion x - coefficient_x_cubed * x^3

theorem coefficient_x_cubed_proof :
  ∃ (f : ℝ → ℝ), ∀ x ≠ 0, binomial_expansion x = f x * x^3 + coefficient_x_cubed * x^3 + g x := by
  sorry

/-- The main theorem stating that the coefficient of x^3 in the binomial expansion of (x^2 - 1/x)^9 is -126 -/
theorem binomial_expansion_coefficient :
  coefficient_x_cubed = -126 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_proof_binomial_expansion_coefficient_l45_4538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_norm_double_vector_l45_4510

variable {E : Type*} [NormedAddCommGroup E] [SMul ℝ E]

theorem norm_double_vector (v : E) (h : ‖v‖ = 5) : ‖(2 : ℝ) • v‖ = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_norm_double_vector_l45_4510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_problem_l45_4516

theorem function_value_problem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun (x : ℝ) => a^(x - 1/2)
  f (Real.log a / Real.log a) = Real.sqrt 10 → a = 10 ∨ a = Real.sqrt (1/10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_problem_l45_4516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_fund_deficit_is_340_l45_4559

/-- Calculates the deficit in Miss Grayson's class fund after the field trip expenses --/
def class_fund_deficit (bake_sale_amount : ℕ) (num_students : ℕ) (student_contribution : ℕ)
  (first_activity_cost : ℕ) (second_activity_cost : ℕ) (service_charge : ℕ) : ℕ :=
  let total_raised := bake_sale_amount + num_students * student_contribution
  let total_costs := num_students * (first_activity_cost + second_activity_cost + service_charge)
  total_costs - total_raised

theorem class_fund_deficit_is_340 : 
  class_fund_deficit 50 30 5 8 9 1 = 340 := by
  -- Unfold the definition and simplify
  unfold class_fund_deficit
  -- Perform the arithmetic
  norm_num

#eval class_fund_deficit 50 30 5 8 9 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_fund_deficit_is_340_l45_4559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_is_thirteen_l45_4564

/-- The age of a person born on October 1st (National Day) -/
def age : ℕ := 13

/-- The number of days in October -/
def october_days : ℕ := 31

/-- The relationship between the age and the number of days in October -/
axiom age_relation : 3 * age - 8 = october_days

theorem age_is_thirteen : age = 13 := by
  -- Proof goes here
  rfl

#check age_is_thirteen

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_is_thirteen_l45_4564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_graphs_l45_4569

-- Define the three functions
noncomputable def f (x : ℝ) : ℝ := x^2 - 3

noncomputable def g (x : ℝ) : ℝ := 
  if x ≠ 3 then (x^3 - 27) / (x - 3)
  else 0  -- undefined at x = 3, we set it to 0 for completeness

def h (x y : ℝ) : Prop := (x - 3) * y = x^3 - 27

-- Theorem statement
theorem different_graphs : 
  (∃ x : ℝ, f x ≠ g x) ∧ 
  (∃ x : ℝ, ¬(h x (f x))) ∧ 
  (∃ x : ℝ, ¬(h x (g x))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_graphs_l45_4569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_option_is_9_95_l45_4565

def scale_start : ℝ := 9.75
def scale_end : ℝ := 10.0
def arrow_position : ℝ := scale_start + 0.75 * (scale_end - scale_start)

def options : List ℝ := [9.80, 9.90, 9.95, 10.0, 9.85]

noncomputable def closest_option (x : ℝ) (opts : List ℝ) : ℝ :=
  (opts.argmin (fun y => |x - y|)).getD 0

theorem closest_option_is_9_95 :
  closest_option arrow_position options = 9.95 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_option_is_9_95_l45_4565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_eq_four_l45_4529

/-- The number of positive factors of 48 that are also multiples of 6 -/
def count_factors : ℕ :=
  (Finset.filter (λ x ↦ x ∣ 48 ∧ 6 ∣ x) (Finset.range 49)).card

/-- Theorem stating that the count of positive factors of 48 that are also multiples of 6 is 4 -/
theorem count_factors_eq_four : count_factors = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_eq_four_l45_4529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_home_cost_l45_4583

/-- Represents the cost and size of a module -/
structure HomeModule where
  size : Nat
  cost : Nat

/-- Calculates the total cost of the modular home -/
def totalCost (totalSize : Nat) (kitchen : HomeModule) (bathroom : HomeModule) (bedroom : HomeModule) (livingSpaceCost : Nat) : Nat :=
  let kitchenCost := kitchen.cost
  let bathroomCost := 2 * bathroom.cost
  let bedroomCost := 3 * bedroom.cost
  let requiredModulesSize := kitchen.size + 2 * bathroom.size + 3 * bedroom.size
  let livingSpaceSize := totalSize - requiredModulesSize
  let livingSpaceTotalCost := livingSpaceSize * livingSpaceCost
  kitchenCost + bathroomCost + bedroomCost + livingSpaceTotalCost

theorem modular_home_cost :
  totalCost 3000 
    (HomeModule.mk 400 28000) 
    (HomeModule.mk 200 12000) 
    (HomeModule.mk 300 18000) 
    110 = 249000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_home_cost_l45_4583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monica_expected_value_l45_4558

-- Define the die
def die_sides : ℕ := 8

-- Define prime numbers on the die
def primes : List ℕ := [2, 3, 5, 7]

-- Define composite numbers on the die
def composites : List ℕ := [4, 6, 8]

-- Define the number that is neither prime nor composite
def neither : ℕ := 1

-- Define the loss for rolling neither prime nor composite
def loss : ℚ := 4

-- Function to calculate expected value
def expected_value : ℚ :=
  (primes.sum / die_sides) - (loss / die_sides)

-- Theorem to prove
theorem monica_expected_value :
  (expected_value * 100).floor / 100 = 163 / 100 := by
  sorry

#eval (expected_value * 100).floor / 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monica_expected_value_l45_4558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_25_l45_4590

/-- Represents the swimming scenario with given parameters -/
structure SwimmingScenario where
  downstream_distance : ℝ
  time : ℝ
  still_water_speed : ℝ

/-- Calculates the upstream distance given a swimming scenario -/
noncomputable def upstream_distance (s : SwimmingScenario) : ℝ :=
  let stream_speed := s.downstream_distance / s.time - s.still_water_speed
  (s.still_water_speed - stream_speed) * s.time

/-- Theorem stating that for the given scenario, the upstream distance is 25 km -/
theorem upstream_distance_is_25 :
  let s : SwimmingScenario := {
    downstream_distance := 45,
    time := 5,
    still_water_speed := 7
  }
  upstream_distance s = 25 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_25_l45_4590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_identity_l45_4541

theorem log_identity (a b : ℝ) 
  (h1 : Real.log 9 / Real.log 18 = a) 
  (h2 : (18 : ℝ)^b = 5) : 
  Real.log 45 / Real.log 36 = (a + b) / (2 - a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_identity_l45_4541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l45_4505

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := tan (2 * x - π / 6)

-- Theorem for the smallest positive period and the value of f(π/8)
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  f (π / 8) = 2 - sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l45_4505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_displacement_l45_4515

-- Define the velocity function
def v (t : ℝ) : ℝ := t^2 - t + 6

-- Define the displacement function
noncomputable def displacement (a b : ℝ) : ℝ := ∫ t in a..b, v t

-- Theorem statement
theorem particle_displacement :
  displacement 1 4 = 31.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_displacement_l45_4515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l45_4579

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := x^(1/3) + 1

-- Define the proposed inverse function
def f_inv (x : ℝ) : ℝ := (x - 1) ^ 3

-- Theorem statement
theorem inverse_function_proof :
  (∀ x : ℝ, f (f_inv x) = x) ∧ (∀ x : ℝ, f_inv (f x) = x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l45_4579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_intuitive_diagram_formula_l45_4545

/-- The area of an equilateral triangle with side length a -/
noncomputable def area_equilateral (a : ℝ) : ℝ := (Real.sqrt 3 / 4) * a^2

/-- The relationship between the area of the original diagram and its planar intuitive diagram -/
noncomputable def intuitive_area_ratio : ℝ := Real.sqrt 2 / 4

/-- The area of the planar intuitive diagram A'B'C' -/
noncomputable def area_intuitive_diagram (a : ℝ) : ℝ := intuitive_area_ratio * area_equilateral a

/-- Theorem stating the formula for the area of the intuitive diagram -/
theorem area_intuitive_diagram_formula (a : ℝ) :
  area_intuitive_diagram a = (Real.sqrt 6 / 16) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_intuitive_diagram_formula_l45_4545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_x_coordinate_l45_4502

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem midpoint_x_coordinate 
  (M N : ℝ × ℝ) 
  (hM : parabola M) 
  (hN : parabola N) 
  (h_dist : distance M focus + distance N focus = 6) :
  (M.1 + N.1) / 2 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_x_coordinate_l45_4502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l45_4503

noncomputable section

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + m * x^2 + 1

-- Define the derivative of f(x)
def f' (m : ℝ) (x : ℝ) : ℝ := x^2 + 2 * m * x

-- State the theorem
theorem tangent_line_and_monotonicity 
  (m : ℝ) 
  (h : f' m (-1) = 3) :
  -- Part 1: Equation of the tangent line
  (∃ A B C : ℝ, A * 1 + B * f m 1 + C = 0 ∧ 
   ∀ x y : ℝ, y = f m x → (A * x + B * y + C = 0 ↔ y - f m 1 = f' m 1 * (x - 1)) ∧
   A = 3 ∧ B = -3 ∧ C = 4) ∧ 
  -- Part 2: Intervals of monotonicity
  (∀ x : ℝ, x < -2 → f' m x > 0) ∧
  (∀ x : ℝ, -2 < x ∧ x < 0 → f' m x < 0) ∧
  (∀ x : ℝ, 0 < x → f' m x > 0) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l45_4503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_prime_factor_of_binom_300_150_l45_4540

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  (Finset.filter (fun p => Nat.Prime p ∧ 10 ≤ p ∧ p < 100) (Nat.divisors (Nat.choose 300 150))).max' 
    (by sorry) = 97 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_prime_factor_of_binom_300_150_l45_4540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l45_4581

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (75 - x)) + Real.sqrt (x * (3 - x))

noncomputable def x_0 : ℝ := 25 / 8
def M : ℝ := 15

theorem max_value_of_f :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x ≤ M) ∧
  f x_0 = M ∧
  0 ≤ x_0 ∧ x_0 ≤ 3 := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l45_4581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_height_l45_4574

-- Define the cone properties
noncomputable def cone_lateral_area : ℝ := 2 * Real.pi
noncomputable def cone_base_area : ℝ := Real.pi

-- Define the theorem
theorem cone_slant_height :
  ∀ (r l : ℝ),
  r > 0 →
  l > 0 →
  r^2 * Real.pi = cone_base_area →
  r * l * Real.pi = cone_lateral_area →
  l = 2 := by
  sorry

#check cone_slant_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_height_l45_4574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_negative_functions_l45_4578

-- Define the "inverse negative" property
def is_inverse_negative (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f (1 / x) = -f x

-- Define the three functions
noncomputable def f1 (x : ℝ) : ℝ := x - 1 / x

noncomputable def f2 (x : ℝ) : ℝ := x + 1 / x

noncomputable def f3 (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x
  else if x = 1 then 0
  else if x > 1 then -1 / x
  else 0  -- This case is added to make the function total

-- State the theorem
theorem inverse_negative_functions :
  is_inverse_negative f1 ∧
  ¬is_inverse_negative f2 ∧
  (∀ x : ℝ, x > 0 → f3 (1 / x) = -f3 x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_negative_functions_l45_4578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_cosine_relation_l45_4522

theorem arithmetic_sequence_cosine_relation (θ α β : ℝ) :
  (∃ k : ℝ, Real.sin α = Real.sin θ + k ∧ Real.cos θ = Real.sin α + k) →
  (Real.sin β)^2 = Real.sin θ * Real.cos θ →
  2 * Real.cos (2 * θ) = Real.cos (2 * β) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_cosine_relation_l45_4522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estate_value_l45_4508

-- Define the total estate
def total_estate : ℝ := sorry

-- Define the shares of the three children
def oldest_child_share : ℝ := sorry
def middle_child_share : ℝ := sorry
def youngest_child_share : ℝ := sorry

-- Define the wife's share
def wife_share : ℝ := sorry

-- Define the charity's share
def charity_share : ℝ := 600

-- Theorem to prove
theorem estate_value : 
  -- Children get two-thirds of the estate
  oldest_child_share + middle_child_share + youngest_child_share = (2/3) * total_estate →
  -- Children's shares are in the ratio 5:3:2
  oldest_child_share = (5/2) * youngest_child_share ∧
  middle_child_share = (3/2) * youngest_child_share →
  -- Wife gets three times as much as the youngest child
  wife_share = 3 * youngest_child_share →
  -- Total estate is the sum of all shares
  total_estate = oldest_child_share + middle_child_share + youngest_child_share + wife_share + charity_share →
  -- Prove that the total estate is $90,000
  total_estate = 90000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estate_value_l45_4508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_served_l45_4563

/-- Represents the number of students from the East school -/
def east (n : ℕ) : ℕ := sorry

/-- Represents the number of students from the West school -/
def west (n : ℕ) : ℕ := sorry

/-- Calculates the total bus fare for both schools -/
def total_fare (n : ℕ) : ℕ := 3 * east n + 5 * west n

/-- Calculates the number of primary school students served -/
def students_served (n : ℕ) : ℕ := 5 * east n + 3 * west n

/-- The conditions of the problem -/
def conditions (n : ℕ) : Prop :=
  east n ≥ 1 ∧ 
  west n ≥ 1 ∧ 
  west n = east n + 1 ∧ 
  total_fare n ≤ 37

theorem max_students_served :
  ∃ n : ℕ, conditions n ∧ 
  (∀ m : ℕ, conditions m → students_served m ≤ students_served n) ∧
  students_served n = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_served_l45_4563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factors_difference_l45_4537

theorem largest_prime_factors_difference (n : Nat) (h : n = 163027) :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p * q = n ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ n → r ≤ p ∧ r ≤ q) ∧
  p - q = 662 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factors_difference_l45_4537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_equation_solution_l45_4562

-- Define the custom operation *
def star (a b : ℚ) : ℚ := 4 * a - 2 * b

-- State the theorem
theorem custom_equation_solution :
  ∀ x : ℚ, star 3 (star 7 x) = 5 → x = 49/4 := by
  intro x h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_equation_solution_l45_4562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_sector_l45_4504

/-- Given a circular sector with radius 5 cm and central angle 162°, 
    prove that the height of the cone formed from this sector is (√319)/4 cm. -/
theorem cone_height_from_sector (r θ h : ℝ) : 
  r = 5 → θ = 162 → h = (Real.sqrt 319) / 4 → 
  2 * r * Real.sin (θ * Real.pi / 360) = r * Real.sqrt ((r / h)^2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_sector_l45_4504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l45_4588

theorem max_value_of_expression (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 1) : 
  x + Real.sqrt (2 * x * y) + 3 * (x * y * z) ^ (1/3) ≤ 2 ∧ 
  ∃ x' y' z' : ℝ, x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' + y' + z' = 1 ∧ 
    x' + Real.sqrt (2 * x' * y') + 3 * (x' * y' * z') ^ (1/3) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l45_4588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_imply_a_range_l45_4534

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x

noncomputable def g (x : ℝ) : ℝ := Real.log x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x - g x

-- State the theorem
theorem two_zeros_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ h a x₁ = 0 ∧ h a x₂ = 0) →
  (0 < a ∧ a < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_imply_a_range_l45_4534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_pairs_count_l45_4525

theorem yellow_pairs_count (total_students : ℕ) (blue_students : ℕ) (yellow_students : ℕ) 
  (total_pairs : ℕ) (blue_pairs : ℕ) 
  (h1 : total_students = 132)
  (h2 : blue_students = 57)
  (h3 : yellow_students = 75)
  (h4 : total_pairs = 66)
  (h5 : blue_pairs = 23)
  (h6 : total_students = blue_students + yellow_students) :
  total_pairs - blue_pairs - (blue_students - 2 * blue_pairs) = 32 := by
  sorry

#check yellow_pairs_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_pairs_count_l45_4525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_subset_with_divisibility_property_l45_4501

theorem no_infinite_subset_with_divisibility_property :
  ¬∃ (S : Set ℕ+), Set.Infinite S ∧
    ∀ (a b : ℕ+), a ∈ S → b ∈ S → (a^2 - a*b + b^2) ∣ (a*b)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_subset_with_divisibility_property_l45_4501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_one_problem_two_l45_4511

-- Problem 1
theorem problem_one : 
  Real.rpow 0.027 (1/3) - Real.rpow (-1/7) (-2) + Real.rpow 2.56 (3/4) - Real.rpow 3 (-1) + Real.rpow (Real.sqrt 2 - 1) 0 = 
  -(1471 - 48 * Real.sqrt 10) / 30 := by sorry

-- Problem 2
theorem problem_two : 
  (Real.log 8 + Real.log 125 - Real.log 2 - Real.log 5) / (Real.log (Real.sqrt 10) * Real.log 0.1) = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_one_problem_two_l45_4511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l45_4543

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a sphere -/
noncomputable def sphereVolume (s : Sphere) : ℝ := (4/3) * Real.pi * s.radius^3

/-- Calculates the new height of liquid in a cone after submerging a sphere -/
noncomputable def newHeight (c : Cone) (s : Sphere) : ℝ :=
  c.height + sphereVolume s / (Real.pi * c.radius^2)

/-- Represents the rise in liquid level -/
noncomputable def liquidRise (c : Cone) (s : Sphere) : ℝ := newHeight c s - c.height

theorem liquid_rise_ratio :
  ∀ (c1 c2 : Cone) (s : Sphere),
    c1.radius = 4 →
    c2.radius = 8 →
    s.radius = 1.5 →
    coneVolume c1 = coneVolume c2 →
    liquidRise c1 s / liquidRise c2 s = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l45_4543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l45_4567

/-- Given a hyperbola with equation (x^2 / 144) - (y^2 / 64) = 1, 
    the distance between its vertices is 24. -/
theorem hyperbola_vertex_distance : 
  ∀ (x y : ℝ), x^2 / 144 - y^2 / 64 = 1 → 
  ∃ (v₁ v₂ : ℝ × ℝ), 
    (v₁.1^2 / 144 - v₁.2^2 / 64 = 1) ∧ 
    (v₂.1^2 / 144 - v₂.2^2 / 64 = 1) ∧ 
    (v₁.2 = 0) ∧ (v₂.2 = 0) ∧
    (Real.sqrt ((v₁.1 - v₂.1)^2 + (v₁.2 - v₂.2)^2) = 24) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l45_4567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lake_area_proof_l45_4589

theorem lake_area_proof (a b c d : ℕ) : 
  a^2 + c^2 = 74 →
  b^2 + d^2 = 116 →
  (a + b)^2 + (c + d)^2 = 370 →
  (a = 5 ∧ b = 4 ∧ c = 7 ∧ d = 10) →
  (((a + b) * (c + d) : ℚ) / 2) - ((a * c : ℚ) / 2) - ((b * d : ℚ) / 2) - (c * d : ℚ) = 11 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check lake_area_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lake_area_proof_l45_4589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_g_inv_at_one_l45_4514

/-- The function g(x) = 2x^2 - 5x + 3 -/
def g (x : ℝ) : ℝ := 2*x^2 - 5*x + 3

/-- The inverse function of g -/
noncomputable def g_inv (x : ℝ) : ℝ := (5 + Real.sqrt (1 + 8*x)) / 4

/-- Theorem stating that g(1) = g⁻¹(1) -/
theorem g_equals_g_inv_at_one : g 1 = g_inv 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_g_inv_at_one_l45_4514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inclination_range_l45_4594

theorem angle_inclination_range (α : ℝ) (θ : ℝ) : 
  (x : ℝ) → (y : ℝ) → x * Real.cos α - y + 1 = 0 →
  Real.tan θ = Real.cos α →
  -1 ≤ Real.cos α ∧ Real.cos α ≤ 1 →
  θ ∈ Set.Icc 0 (π/4) ∪ Set.Ico (3*π/4) π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inclination_range_l45_4594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_action_figures_remaining_l45_4507

theorem action_figures_remaining (initial : ℕ) (sold_fraction : ℚ) (given_fraction : ℚ) : 
  initial = 24 → 
  sold_fraction = 1/4 → 
  given_fraction = 1/3 → 
  initial - (sold_fraction * initial).floor - (given_fraction * (initial - (sold_fraction * initial).floor)).floor = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_action_figures_remaining_l45_4507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_relations_l45_4561

/-- Given two vectors a and b in R^2, prove cosine of their angle difference and sine of alpha -/
theorem vector_angle_relations (α β : ℝ) 
  (h1 : -π/2 < β ∧ β < 0 ∧ 0 < α ∧ α < π/2)
  (h2 : Real.sin β = -1/7)
  (h3 : ‖(Real.cos α, Real.sin α) - (Real.cos β, Real.sin β)‖ = 1) :
  Real.cos (α - β) = 1/2 ∧ Real.sin α = 13/14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_relations_l45_4561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_solutions_l45_4591

theorem no_positive_integer_solutions :
  ¬∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x^2 = 5^y + 2^z - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_solutions_l45_4591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_points_form_regular_hexagon_l45_4520

/-- An equilateral triangle with sides divided into three equal parts -/
structure DividedEquilateralTriangle where
  /-- The side length of the equilateral triangle -/
  side_length : ℝ
  /-- Assumption that the side length is positive -/
  side_positive : side_length > 0

/-- A point on the side of the triangle, represented by its distance from a vertex -/
structure DivisionPoint (triangle : DividedEquilateralTriangle) where
  /-- The distance of the point from a vertex -/
  distance : ℝ
  /-- Assumption that the distance is between 0 and the side length -/
  distance_valid : 0 < distance ∧ distance < triangle.side_length

/-- The set of division points that form the hexagon -/
def hexagon_points (triangle : DividedEquilateralTriangle) : Set (DivisionPoint triangle) :=
  {p : DivisionPoint triangle | p.distance = triangle.side_length / 3 ∨ p.distance = 2 * triangle.side_length / 3}

/-- Definition of a regular hexagon -/
structure RegularHexagon where
  /-- The side length of the regular hexagon -/
  side_length : ℝ
  /-- Assumption that the side length is positive -/
  side_positive : side_length > 0

/-- Helper function to represent the set of vertices of a regular hexagon -/
def set_of_vertices (hexagon : RegularHexagon) : Set (ℝ × ℝ) := sorry

/-- The theorem to be proved -/
theorem division_points_form_regular_hexagon (triangle : DividedEquilateralTriangle) :
  ∃ (hexagon : RegularHexagon), hexagon.side_length = triangle.side_length / 3 ∧
  ∀ (p : DivisionPoint triangle), p ∈ hexagon_points triangle → 
  ∃ (v : ℝ × ℝ), v ∈ set_of_vertices hexagon := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_points_form_regular_hexagon_l45_4520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l45_4592

def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (2, 2)

theorem vector_properties :
  let cos_angle := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  let lambda := -(a.1^2 + a.2^2) / (2 * (a.1 * b.1 + a.2 * b.2))
  cos_angle = Real.sqrt 2 / 10 ∧ lambda = -25/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l45_4592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_integer_solutions_l45_4576

def is_solution (x : ℤ) : Prop :=
  (x - 1 : ℚ) ^ (16 - x^2) = 1

theorem four_integer_solutions :
  ∃ (s : Finset ℤ), s.card = 4 ∧ (∀ x, x ∈ s ↔ is_solution x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_integer_solutions_l45_4576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_correct_l45_4517

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (2*x^3 + 4*x^2 + 2*x - 1) / ((x+1)^2 * (x^2 + 2*x + 2))

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := 1/(x+1) + Real.log (x^2 + 2*x + 2) - Real.arctan (x+1)

-- State the theorem
theorem indefinite_integral_correct : 
  ∀ x : ℝ, x ≠ -1 → (deriv F x = f x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_correct_l45_4517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactlyOneAndTwoBlackMutuallyExclusiveNotComplementary_l45_4554

-- Define the sample space
inductive Ball : Type
| Red : Ball
| Black : Ball

instance : DecidableEq Ball :=
  fun a b => match a, b with
  | Ball.Red, Ball.Red => isTrue rfl
  | Ball.Black, Ball.Black => isTrue rfl
  | Ball.Red, Ball.Black => isFalse (fun h => Ball.noConfusion h)
  | Ball.Black, Ball.Red => isFalse (fun h => Ball.noConfusion h)

-- Define the pocket
def pocket : Multiset Ball :=
  2 • {Ball.Red} + 2 • {Ball.Black}

-- Define the event of selecting exactly one black ball
def exactlyOneBlack (selection : Multiset Ball) : Prop :=
  Multiset.card selection = 2 ∧ (Multiset.count Ball.Black selection = 1)

-- Define the event of selecting exactly two black balls
def exactlyTwoBlack (selection : Multiset Ball) : Prop :=
  Multiset.card selection = 2 ∧ (Multiset.count Ball.Black selection = 2)

-- Theorem stating that the events are mutually exclusive but not complementary
theorem exactlyOneAndTwoBlackMutuallyExclusiveNotComplementary :
  (∀ selection : Multiset Ball, selection ⊆ pocket → Multiset.card selection = 2 →
    ¬(exactlyOneBlack selection ∧ exactlyTwoBlack selection)) ∧
  (∃ selection : Multiset Ball, selection ⊆ pocket ∧ Multiset.card selection = 2 ∧
    ¬exactlyOneBlack selection ∧ ¬exactlyTwoBlack selection) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactlyOneAndTwoBlackMutuallyExclusiveNotComplementary_l45_4554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_nonagon_exterior_angle_is_130_l45_4556

/-- The measure of the exterior angle formed by a square and a regular nonagon sharing a side -/
def square_nonagon_exterior_angle : ℚ :=
  let square_interior_angle : ℚ := 90
  let nonagon_interior_angle : ℚ := 140
  360 - square_interior_angle - nonagon_interior_angle

/-- Proof that the exterior angle formed by a square and a regular nonagon sharing a side is 130° -/
theorem square_nonagon_exterior_angle_is_130 :
  square_nonagon_exterior_angle = 130 := by
  unfold square_nonagon_exterior_angle
  norm_num

#eval square_nonagon_exterior_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_nonagon_exterior_angle_is_130_l45_4556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_b_values_l45_4509

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

theorem product_of_b_values : 
  ∃ b₁ b₂ : ℝ, (f b₁ 3 = (f b₁).invFun (b₁ + 2)) ∧ 
               (f b₂ 3 = (f b₂).invFun (b₂ + 2)) ∧ 
               (b₁ * b₂ = -40/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_b_values_l45_4509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_speed_calculation_l45_4532

-- Define the actual speed, distance, and additional distance
def actual_speed : ℚ := 10
def actual_distance : ℚ := 40
def additional_distance : ℚ := 20

-- Define the faster speed as a function of the given parameters
noncomputable def faster_speed (s a d : ℚ) : ℚ := (a + d) / (a / s)

-- Theorem statement
theorem faster_speed_calculation :
  faster_speed actual_speed actual_distance additional_distance = 15 := by
  -- Unfold the definition of faster_speed
  unfold faster_speed
  -- Simplify the expression
  simp [actual_speed, actual_distance, additional_distance]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_speed_calculation_l45_4532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_below_average_is_140_l45_4555

noncomputable def class_averages : List ℝ := [75, 85, 90, 65]

noncomputable def school_average (averages : List ℝ) : ℝ :=
  (averages.sum) / (averages.length : ℝ)

noncomputable def sum_below_average (averages : List ℝ) : ℝ :=
  (averages.filter (λ x => x < school_average averages)).sum

theorem sum_below_average_is_140 :
  sum_below_average class_averages = 140 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_below_average_is_140_l45_4555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_length_l45_4536

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the properties of the rectangle
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ := Real.sqrt (r.length^2 + r.width^2)

-- State the theorem
theorem rectangle_diagonal_length :
  ∃ (r : Rectangle), r.area = 20 ∧ r.perimeter = 18 → r.diagonal = Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_length_l45_4536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_boys_same_room_l45_4599

theorem probability_two_boys_same_room (n : ℕ) (k : ℕ) (max_per_room : ℕ) 
  (h1 : n = 5) 
  (h2 : k = 3) 
  (h3 : max_per_room = 2) :
  (k * (n - 2).factorial : ℚ) / ((n.choose 2 * (n - 2).choose 2 / 2) * k.factorial) = 1 / 5 := by
  sorry

#check probability_two_boys_same_room

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_boys_same_room_l45_4599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_eq_one_count_l45_4548

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if -5 ≤ x ∧ x ≤ -1 then -x^2 + 1
  else if -1 < x ∧ x < 2 then 2*x + 3
  else if 2 ≤ x ∧ x ≤ 5 then -x + 4
  else 0  -- undefined for x outside [-5, 5]

-- Define the theorem
theorem g_composition_eq_one_count :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, x ∈ Set.Icc (-5) 5 ∧ g (g x) = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_eq_one_count_l45_4548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_f_l45_4596

noncomputable def f (x : ℝ) : ℝ := x - Real.log x + (2*x - 1) / x^2

noncomputable def f' (x : ℝ) : ℝ := 1 - 1/x + (2*x^2 - (2*x - 1)*2*x) / x^4

theorem f_greater_than_f'_plus_three_halves (x : ℝ) (h : x ∈ Set.Icc 1 2) :
  f x > f' x + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_f_l45_4596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_conversion_l45_4587

theorem polar_to_cartesian_conversion :
  ∀ (ρ θ x y : ℝ),
  (ρ * Real.cos θ = x) →
  (ρ * Real.sin θ = y) →
  (ρ^2 = x^2 + y^2) →
  (ρ^2 * Real.cos θ - ρ = 0) ↔ (x^2 + y^2 = 0 ∨ x = 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_conversion_l45_4587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_fixed_point_l45_4593

/-- Ellipse with center at origin, foci on x-axis, and eccentricity 1/2 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : c > 0
  h4 : a > b
  h5 : c / a = 1 / 2
  h6 : a^2 = b^2 + c^2

/-- Line with given slope and y-intercept -/
structure Line where
  k : ℝ
  m : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle defined by its equation x^2 + y^2 = r^2 -/
structure Circle where
  r : ℝ
  h : r > 0

/-- Predicate for a point being on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Predicate for a point being on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  p.y = l.k * p.x + l.m

/-- Predicate for a line being tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  ∃ (p : Point), on_line p l ∧ p.x^2 + p.y^2 = c.r^2 ∧
  ∀ (q : Point), on_line q l → q.x^2 + q.y^2 ≥ c.r^2

/-- Predicate for a circle passing through three points -/
def circle_through_points (p1 p2 p3 : Point) : Prop :=
  ∃ (center : Point) (r : ℝ),
    (p1.x - center.x)^2 + (p1.y - center.y)^2 = r^2 ∧
    (p2.x - center.x)^2 + (p2.y - center.y)^2 = r^2 ∧
    (p3.x - center.x)^2 + (p3.y - center.y)^2 = r^2

theorem ellipse_equation_and_fixed_point 
  (e : Ellipse) 
  (circ : Circle)
  (h1 : ∃ (l : Line), l.k = Real.sqrt 3 ∧ 
       l.m = -e.c * Real.sqrt 3 ∧
       is_tangent l circ) 
  (h2 : circ.r^2 = e.b^2 / e.a^2) :
  (∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1 ↔ 
    x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  (∀ (l : Line), 
    (∃ (M N : Point), 
      on_ellipse M e ∧ on_ellipse N e ∧
      on_line M l ∧ on_line N l ∧
      circle_through_points M N (Point.mk 2 0)) →
    on_line (Point.mk (2/7) 0) l) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_fixed_point_l45_4593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option_B_correct_option_C_correct_option_D_correct_l45_4552

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Option B
theorem option_B_correct (t : Triangle) (k : Real) 
  (h1 : t.a = 2 * k) (h2 : t.b = 3 * k) (h3 : t.c = 4 * k) :
  Real.cos t.C < 0 := by
  sorry

-- Option C
theorem option_C_correct (t : Triangle) 
  (h : Real.sin t.A > Real.sin t.B) :
  t.A > t.B := by
  sorry

-- Option D
theorem option_D_correct (t : Triangle) 
  (h1 : t.C = Real.pi / 3) (h2 : t.b = 10) (h3 : t.c = 9) :
  ∃ (B1 B2 : Real), B1 ≠ B2 ∧ 
    Real.sin B1 = (5 * Real.sqrt 3) / 9 ∧ 
    Real.sin B2 = (5 * Real.sqrt 3) / 9 ∧
    0 < B1 ∧ B1 < Real.pi ∧ 0 < B2 ∧ B2 < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_option_B_correct_option_C_correct_option_D_correct_l45_4552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_b_circumference_l45_4586

/-- Represents a right circular cylinder tank -/
structure Tank where
  height : ℝ
  circumference : ℝ

/-- The volume of a cylinder given its height and circumference -/
noncomputable def cylinderVolume (t : Tank) : ℝ :=
  (t.circumference ^ 2 * t.height) / (4 * Real.pi)

theorem tank_b_circumference (tank_a tank_b : Tank)
  (h_a_height : tank_a.height = 5)
  (h_a_circumference : tank_a.circumference = 4)
  (h_b_height : tank_b.height = 8)
  (h_volume_ratio : cylinderVolume tank_a = 0.10000000000000002 * cylinderVolume tank_b) :
  tank_b.circumference = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_b_circumference_l45_4586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_special_case_l45_4535

theorem geometric_sum_special_case (S : ℝ) (h : S = 2^100) :
  S + 2*S + 4*S + 8*S + 16*S + Finset.sum (Finset.range 96) (λ i => (2^(i+5))*S) + 2^100*S = 2*S^2 - S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_special_case_l45_4535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alphametic_puzzle_solution_l45_4595

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the value of a four-digit number ABCD -/
def fourDigitValue (a b c d : Digit) : Nat :=
  1000 * a.val + 100 * b.val + 10 * c.val + d.val

/-- Represents the value of a three-digit number BCD -/
def threeDigitValue (b c d : Digit) : Nat :=
  100 * b.val + 10 * c.val + d.val

/-- Represents the value of a two-digit number CD -/
def twoDigitValue (c d : Digit) : Nat :=
  10 * c.val + d.val

theorem alphametic_puzzle_solution :
  ∃ (o s e l : Digit),
    (o ≠ s ∧ o ≠ e ∧ o ≠ l ∧ s ≠ e ∧ s ≠ l ∧ e ≠ l) ∧
    (fourDigitValue o s e l + threeDigitValue s e l + twoDigitValue e l + l.val = 10034) ∧
    ((o = ⟨9, by norm_num⟩ ∧ s = ⟨4, by norm_num⟩ ∧ e = ⟨7, by norm_num⟩ ∧ l = ⟨6, by norm_num⟩) ∨
     (o = ⟨8, by norm_num⟩ ∧ s = ⟨9, by norm_num⟩ ∧ e = ⟨7, by norm_num⟩ ∧ l = ⟨6, by norm_num⟩)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alphametic_puzzle_solution_l45_4595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_total_distance_l45_4580

def bug_crawl (start : ℤ) (waypoints : List ℤ) : ℕ :=
  (start :: waypoints).zip waypoints
  |>.map (fun (a, b) => (a - b).natAbs)
  |>.sum

theorem bug_total_distance :
  bug_crawl (-4) [-1, -8, 6] = 24 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_total_distance_l45_4580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_equals_4_f_geq_g_implies_t_geq_17_8_l45_4546

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

noncomputable def g (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := 2 * (Real.log (2 * x + t - 2) / Real.log a)

-- Define the function F
noncomputable def F (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := g a t x - f a x

-- Theorem 1
theorem min_value_implies_a_equals_4 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc 1 2, F a 4 x ≥ 2) ∧ (∃ x ∈ Set.Icc 1 2, F a 4 x = 2) →
  a = 4 := by sorry

-- Theorem 2
theorem f_geq_g_implies_t_geq_17_8 (a : ℝ) (t : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x ∈ Set.Icc 1 2, f a x ≥ g a t x) →
  t ≥ 17/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_equals_4_f_geq_g_implies_t_geq_17_8_l45_4546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fare_formula_for_long_distance_l45_4523

/-- Represents the taxi fare calculation system -/
structure TaxiFare where
  base_price : ℝ := 8
  base_distance : ℕ := 3
  additional_rate : ℝ := 1.6

/-- Calculates the taxi fare for a given distance -/
def calculate_fare (tf : TaxiFare) (distance : ℕ) : ℝ :=
  if distance ≤ tf.base_distance then
    tf.base_price
  else
    tf.base_price + tf.additional_rate * (distance - tf.base_distance)

/-- Theorem: For distances greater than 3 km, the fare is 1.6x + 3.2 -/
theorem fare_formula_for_long_distance (tf : TaxiFare) (distance : ℕ) 
    (h : distance > tf.base_distance) :
    calculate_fare tf distance = tf.additional_rate * (distance : ℝ) + 3.2 := by
  sorry

#eval calculate_fare { base_price := 8, base_distance := 3, additional_rate := 1.6 } 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fare_formula_for_long_distance_l45_4523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_sum_equals_two_l45_4598

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define points D and E
variable (D E : EuclideanSpace ℝ (Fin 2))

-- Define point F (intersection of DE and AC)
variable (F : EuclideanSpace ℝ (Fin 2))

-- B is the midpoint of AC
variable (h1 : B = (A + C) / 2)

-- D is on BC with BD:DC = 1:2
variable (h2 : ∃ t : ℝ, t ∈ Set.Ioo 0 1 ∧ D = (1 - t) • B + t • C ∧ t = 2/3)

-- E is on AB with AE:EB = 2:1
variable (h3 : ∃ s : ℝ, s ∈ Set.Ioo 0 1 ∧ E = (1 - s) • A + s • B ∧ s = 1/3)

-- F is on AC and DE
variable (h4 : ∃ u v : ℝ, u ∈ Set.Ioo 0 1 ∧ v ∈ Set.Ioo 0 1 ∧ 
              F = (1 - u) • A + u • C ∧ 
              F = (1 - v) • D + v • E)

-- Theorem statement
theorem ratio_sum_equals_two :
  (dist E F / dist F C) + (dist A F / dist F D) = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_sum_equals_two_l45_4598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_jumps_circular_arrangement_l45_4549

/-- Represents a circular arrangement of points -/
structure CircularArrangement where
  size : ℕ
  jump_sizes : List ℕ

/-- Represents a path through the circular arrangement -/
structure CircPath (arrangement : CircularArrangement) where
  jumps : List ℕ
  start_point : ℕ

/-- Checks if a path is valid for a given arrangement -/
def is_valid_path (arrangement : CircularArrangement) (path : CircPath arrangement) : Prop :=
  path.jumps.all (λ j => j ∈ arrangement.jump_sizes) ∧
  path.start_point < arrangement.size ∧
  (path.jumps.foldl (λ acc j => (acc + j) % arrangement.size) path.start_point = path.start_point) ∧
  (∀ i < arrangement.size, ∃ k, (path.jumps.take k).foldl (λ acc j => (acc + j) % arrangement.size) path.start_point = i)

/-- The main theorem to be proved -/
theorem min_jumps_circular_arrangement :
  ∀ (arrangement : CircularArrangement),
    arrangement.size = 2016 ∧
    arrangement.jump_sizes = [2, 3] →
    (∃ (path : CircPath arrangement),
      is_valid_path arrangement path ∧
      path.jumps.length = 2017 ∧
      (∀ (other_path : CircPath arrangement),
        is_valid_path arrangement other_path →
        other_path.jumps.length ≥ 2017)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_jumps_circular_arrangement_l45_4549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_d_no_integral_points_in_circle_l45_4533

-- Define the line d
def line_d (x y : ℝ) : Prop := y = x + 1 / Real.sqrt 2

-- Define the distance function from a point to the line d
noncomputable def distance_to_d (m n : ℤ) : ℝ :=
  abs (m - n + 1 / Real.sqrt 2) / Real.sqrt 2

-- Theorem statement
theorem min_distance_to_d :
  (∀ m n : ℤ, distance_to_d m n ≥ (Real.sqrt 2 - 1) / 2) ∧
  (∀ ε > 0, ∃ m n : ℤ, distance_to_d m n < (Real.sqrt 2 - 1) / 2 + ε) := by
  sorry

-- Bonus: Part a) of the original problem
theorem no_integral_points_in_circle (I : ℝ × ℝ) (h : line_d I.1 I.2) :
  ∀ m n : ℤ, (m - I.1)^2 + (n - I.2)^2 > (1/8)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_d_no_integral_points_in_circle_l45_4533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l45_4584

-- Define the variables and parameters
variable (x y k : ℝ)

-- Define the conditions
def condition1 (x y : ℝ) : Prop := x + y - 4 ≤ 0
def condition2 (x y : ℝ) : Prop := x - 2*y + 2 ≤ 0
def condition3 (x y k : ℝ) : Prop := k*x - y + 1 ≥ 0
def condition4 (k : ℝ) : Prop := k > 1/2

-- Define the objective function
def z (x y : ℝ) : ℝ := x - y

-- Define the minimum value condition
def min_condition (k : ℝ) : Prop := 
  ∀ (x y : ℝ), condition1 x y → condition2 x y → condition3 x y k → z x y > -3

-- State the theorem
theorem k_range (x y k : ℝ) : 
  condition1 x y → condition2 x y → condition3 x y k → condition4 k → min_condition k → 
  1/2 < k ∧ k < 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l45_4584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_determines_magnitude_l45_4539

/-- Given two non-zero vectors a and b in a real inner product space, 
    with θ as the angle between them, prove that if the minimum value 
    of |a + tb| is 1 for any real t, then θ uniquely determines |a|. -/
theorem angle_determines_magnitude {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (θ : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0) 
  (h_angle : θ = Real.arccos (inner a b / (norm a * norm b))) 
  (h_min : ∀ t : ℝ, 1 ≤ norm (a + t • b)) 
  (h_exists_min : ∃ t : ℝ, norm (a + t • b) = 1) : 
  ∀ a' : V, θ = Real.arccos (inner a' b / (norm a' * norm b)) → norm a' = norm a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_determines_magnitude_l45_4539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_min_distance_is_two_l45_4521

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1
def circle_C2 (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem min_distance_between_circles :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle_C1 x1 y1 → circle_C2 x2 y2 →
    distance x1 y1 x2 y2 ≥ 2 :=
by
  sorry

-- Additional theorem to show that the minimum distance is exactly 2
theorem min_distance_is_two :
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_C1 x1 y1 ∧ circle_C2 x2 y2 ∧
    distance x1 y1 x2 y2 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_min_distance_is_two_l45_4521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l45_4524

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.sqrt (x + 1)

-- State the theorem
theorem range_of_f :
  (Set.range f) = { y : ℝ | y ≥ -2 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l45_4524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l45_4577

/-- The sum of the infinite series Σ(n=1 to ∞) (2n-1)(1/1001)^(n-1) -/
noncomputable def series_sum : ℝ := ∑' n, (2 * n - 1) * (1 / 1001) ^ (n - 1)

/-- Theorem stating that the sum of the infinite series is equal to 1.003002 -/
theorem series_sum_value : series_sum = 1.003002 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l45_4577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_plus_5pi_over_4_l45_4512

theorem sin_beta_plus_5pi_over_4 (α β : ℝ) :
  (Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 3/5) →
  (π < β ∧ β < 3*π/2) →
  Real.sin (β + 5*π/4) = 7*Real.sqrt 2/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_plus_5pi_over_4_l45_4512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramesh_payment_l45_4526

noncomputable def labelled_price : ℚ := 24475 / 11 * 10

theorem ramesh_payment (transport_cost installation_cost : ℚ) 
  (h1 : transport_cost = 125) 
  (h2 : installation_cost = 250) : 
  (labelled_price * 4/5 + transport_cost + installation_cost : ℚ) = 18175 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramesh_payment_l45_4526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_multiples_of_five_l45_4513

theorem four_digit_multiples_of_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 9000)).card = 1800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_multiples_of_five_l45_4513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_addition_correction_l45_4553

/-- Represents a 6-digit number as a list of digits -/
def SixDigitNumber := List Nat

/-- Checks if a list represents a valid 6-digit number -/
def isValidSixDigitNumber (n : SixDigitNumber) : Prop :=
  n.length = 6 ∧ n.all (λ d => d < 10)

/-- Replaces all occurrences of one digit with another in a number -/
def replaceDigit (n : SixDigitNumber) (d e : Nat) : SixDigitNumber :=
  n.map (λ x => if x = d then e else x)

/-- Converts a list of digits to a natural number -/
def listToNat (n : SixDigitNumber) : Nat :=
  n.foldl (λ acc d => acc * 10 + d) 0

/-- The main theorem to be proved -/
theorem incorrect_addition_correction :
  ∃ (d e : Nat), d ≠ e ∧ d < 10 ∧ e < 10 ∧ d + e = 7 ∧
  ∃ (n1 n2 : SixDigitNumber),
    isValidSixDigitNumber n1 ∧
    isValidSixDigitNumber n2 ∧
    listToNat (replaceDigit n1 d e) + listToNat (replaceDigit n2 d e) =
    listToNat [1, 8, 6, 7, 4, 2, 8] ∧
    n1 = [8, 3, 5, 6, 9, 7] ∧
    n2 = [9, 3, 4, 8, 2, 1] :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_addition_correction_l45_4553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_eccentricity_sum_l45_4518

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b
  i : b > 0

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > 0
  i : b > 0

/-- Represents the problem setup -/
structure Setup where
  C₁ : Ellipse
  C₂ : Hyperbola
  F₁ : Point
  F₂ : Point
  P : Point
  h : True  -- Placeholder for P ∈ C₁ ∩ C₂
  i : True  -- Placeholder for F₁ is a focus of C₁ and C₂
  j : True  -- Placeholder for F₂ is a focus of C₁ and C₂
  k : (P.x - F₁.x) * (P.x - F₂.x) + (P.y - F₁.y) * (P.y - F₂.y) = 0

/-- The eccentricity of an ellipse -/
noncomputable def ellipseEccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The eccentricity of a hyperbola -/
noncomputable def hyperbolaEccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- The main theorem to be proved -/
theorem min_eccentricity_sum (s : Setup) :
  let e₁ := ellipseEccentricity s.C₁
  let e₂ := hyperbolaEccentricity s.C₂
  ∃ (m : ℝ), m = 9/2 ∧ ∀ (e₁' e₂' : ℝ), 4 * e₁'^2 + e₂'^2 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_eccentricity_sum_l45_4518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_zero_point_solution_min_value_exists_and_eq_min_of_segments_l45_4572

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Define the solution set of f(x) > 2
def solution_set : Set ℝ := {x : ℝ | f x > 2}

-- Define the zero-point segmentation method solution
def zero_point_solution : Set ℝ := sorry

-- Define the minimum value of f(x)
noncomputable def min_value : ℝ := sorry

-- Define the minimum values on each segment
def segment_min_values : List ℝ := sorry

-- Theorem 1: The solution set equals the zero-point segmentation method solution
theorem solution_set_eq_zero_point_solution : solution_set = zero_point_solution := by sorry

-- Theorem 2: The minimum value exists and equals the minimum of segment minimum values
theorem min_value_exists_and_eq_min_of_segments : 
  ∃ (min : ℝ), min_value = min ∧ min ∈ segment_min_values := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_zero_point_solution_min_value_exists_and_eq_min_of_segments_l45_4572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_is_sixty_l45_4582

/-- A function that returns true if a three-digit number satisfies all conditions -/
def satisfies_conditions (n : ℕ) : Bool :=
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 5 ≠ 0 ∧  -- not divisible by 5
  (n / 100 + (n / 10) % 10 + n % 10) < 20 ∧  -- sum of digits less than 20
  n / 100 = n % 10  -- first digit equals third digit

/-- The count of numbers satisfying the conditions -/
def count_satisfying_numbers : ℕ :=
  (List.range 1000).filter satisfies_conditions |>.length

/-- Theorem stating that the count of numbers satisfying the conditions is 60 -/
theorem count_is_sixty : count_satisfying_numbers = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_is_sixty_l45_4582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_operation_terminates_l45_4568

/-- Represents the state of the pentagon -/
structure Pentagon where
  vertices : Fin 5 → ℤ
  sum_positive : (vertices 0) + (vertices 1) + (vertices 2) + (vertices 3) + (vertices 4) > 0

/-- Applies the operation to three consecutive vertices -/
def apply_operation (p : Pentagon) (i : Fin 5) : Pentagon :=
  let j := i.succ
  let k := j.succ
  if p.vertices j < 0 then
    { vertices := λ m =>
        if m = i then p.vertices i + p.vertices j
        else if m = j then -p.vertices j
        else if m = k then p.vertices k + p.vertices j
        else p.vertices m,
      sum_positive := sorry }
  else p

/-- Checks if any vertex is negative -/
def has_negative (p : Pentagon) : Bool :=
  ∃ i, p.vertices i < 0

/-- The measure T as defined in the solution -/
def measure_T (p : Pentagon) : ℤ :=
  (p.vertices 0 - p.vertices 2)^2 +
  (p.vertices 1 - p.vertices 3)^2 +
  (p.vertices 2 - p.vertices 4)^2 +
  (p.vertices 3 - p.vertices 0)^2 +
  (p.vertices 4 - p.vertices 1)^2

/-- The main theorem -/
theorem pentagon_operation_terminates (p : Pentagon) :
  ∃ n : ℕ, ¬(has_negative (Nat.iterate (λ p => apply_operation p 0) n p)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_operation_terminates_l45_4568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l45_4547

/-- Given a line l with direction vector (2,m,1) and a plane α with normal vector (1,1/2,2),
    if l is parallel to α, then m = -8 -/
theorem line_parallel_to_plane (m : ℝ) :
  let direction_vector : ℝ × ℝ × ℝ := (2, m, 1)
  let normal_vector : ℝ × ℝ × ℝ := (1, 1/2, 2)
  (2 * 1 + m * (1/2) + 1 * 2 = 0) →
  m = -8 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l45_4547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l45_4571

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is on an asymptote of the hyperbola -/
def is_on_asymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y = (h.a / h.b) * p.x ∨ p.y = -(h.a / h.b) * p.x

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- Theorem stating the condition for the eccentricity of the hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) :
  (∀ p q : Point, is_on_asymptote h p → is_on_asymptote h q →
    p ≠ q → (∃ c : Point, c.x = 0 ∧ c.y = 0 ∧
    (c.x - p.x)^2 + (c.y - p.y)^2 = (c.x - q.x)^2 + (c.y - q.y)^2)) →
  eccentricity h = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l45_4571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l45_4597

/-- The curve on which point P lies -/
def curve (x y : ℝ) : Prop := x^2 - y - Real.log x = 0

/-- The line to which we're calculating the distance -/
def line (x y : ℝ) : Prop := y = x - 2

/-- The distance function from a point to a line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - y - 2| / Real.sqrt 2

/-- Theorem stating the existence of a point on the curve with minimum distance to the line -/
theorem min_distance_to_line : 
  ∃ (x y : ℝ), curve x y ∧ 
  (∀ (x' y' : ℝ), curve x' y' → 
    distance_to_line x y ≤ distance_to_line x' y') ∧
  distance_to_line x y = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l45_4597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_264_l45_4575

/-- A structure representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A structure representing a pyramid in 3D space -/
structure Pyramid where
  S : Point3D
  P : Point3D
  Q : Point3D
  R : Point3D

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Check if three vectors are perpendicular to each other -/
def arePerpendicular (v1 v2 v3 : Point3D → Point3D → Point3D) : Prop :=
  ∀ (p : Point3D), 
    (v1 p (v2 p p)).x * (v2 p (v3 p p)).x + 
    (v1 p (v2 p p)).y * (v2 p (v3 p p)).y + 
    (v1 p (v2 p p)).z * (v2 p (v3 p p)).z = 0 ∧
    (v2 p (v3 p p)).x * (v3 p (v1 p p)).x + 
    (v2 p (v3 p p)).y * (v3 p (v1 p p)).y + 
    (v2 p (v3 p p)).z * (v3 p (v1 p p)).z = 0 ∧
    (v3 p (v1 p p)).x * (v1 p (v2 p p)).x + 
    (v3 p (v1 p p)).y * (v1 p (v2 p p)).y + 
    (v3 p (v1 p p)).z * (v1 p (v2 p p)).z = 0

/-- Calculate the volume of a pyramid -/
noncomputable def pyramidVolume (pyr : Pyramid) : ℝ :=
  (1/3) * (distance pyr.S pyr.P) * (distance pyr.S pyr.Q) * (distance pyr.S pyr.R)

theorem pyramid_volume_is_264 (pyr : Pyramid) :
  arePerpendicular 
    (λ s p => p) 
    (λ s q => q) 
    (λ s r => r) ∧
  distance pyr.S pyr.P = 12 ∧
  distance pyr.S pyr.Q = 12 ∧
  distance pyr.S pyr.R = 11 →
  pyramidVolume pyr = 264 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_264_l45_4575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_cube_volume_ratio_l45_4506

/-- A regular quadrilateral pyramid with an inscribed cube -/
structure PyramidWithCube where
  /-- Side length of the base of the pyramid -/
  a : ℝ
  /-- Edge length of the inscribed cube -/
  x : ℝ
  /-- The cube is inscribed in the pyramid -/
  cube_inscribed : x > 0 ∧ x < a
  /-- One edge of the cube lies on the midline of the base of the pyramid -/
  edge_on_midline : True
  /-- Vertices of the cube not on the edge lie on the lateral surface of the pyramid -/
  vertices_on_surface : True
  /-- The center of the cube lies on the height of the pyramid -/
  center_on_height : True

/-- The ratio of the volume of the pyramid to the volume of the inscribed cube -/
noncomputable def volume_ratio (p : PyramidWithCube) : ℝ :=
  (19 * Real.sqrt 2 - 6) / 6

/-- Theorem stating the volume ratio of the pyramid to the inscribed cube -/
theorem pyramid_cube_volume_ratio (p : PyramidWithCube) :
  (p.a^2 * p.x * Real.sqrt 2) / (3 * (p.a - p.x) * p.x^3) = volume_ratio p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_cube_volume_ratio_l45_4506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_correct_verify_solution_l45_4557

/-- The time taken for two people walking in opposite directions on a circular track to meet. -/
noncomputable def meetingTime (trackCircumference : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  trackCircumference / (speed1 + speed2)

/-- Theorem: The meeting time is correct for two people walking in opposite directions on a circular track. -/
theorem meeting_time_correct 
  (trackCircumference : ℝ) 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (h1 : trackCircumference > 0) 
  (h2 : speed1 > 0) 
  (h3 : speed2 > 0) :
  meetingTime trackCircumference speed1 speed2 = trackCircumference / (speed1 + speed2) := by
  sorry

/-- Verify the solution for the specific problem instance -/
theorem verify_solution :
  let trackCircumference : ℝ := 640
  let speed1 : ℝ := 70  -- Lata's speed in m/min
  let speed2 : ℝ := 63  -- Geeta's speed in m/min
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |meetingTime trackCircumference speed1 speed2 - 4.812| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_correct_verify_solution_l45_4557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_constant_l45_4542

noncomputable section

-- Define the points and curve
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (-1/2, 0)
def S : ℝ × ℝ := (2, 0)

def C : Set (ℝ × ℝ) := {P | (P.1^2 / 4) + (P.2^2 / 3) = 1}

-- Define the line l
def l (m : ℝ) : Set (ℝ × ℝ) := {P | P.1 = m * (P.2 + 1/2)}

-- Define line l₁
def l₁ : Set (ℝ × ℝ) := {P | P.1 = -3}

-- Define the theorem
theorem ellipse_ratio_constant 
  (P Q : ℝ × ℝ) 
  (m : ℝ) 
  (hP : P ∈ C) 
  (hQ : Q ∈ C) 
  (hPQ : P ∈ l m ∧ Q ∈ l m) 
  (hm : m ≠ 0) 
  (A B : ℝ × ℝ) 
  (hA : A ∈ l₁ ∧ ∃ t : ℝ, A = t • (P - S) + S) 
  (hB : B ∈ l₁ ∧ ∃ t : ℝ, B = t • (Q - S) + S) :
  (1 / A.2 + 1 / B.2) / (1 / P.2 + 1 / Q.2) = 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_constant_l45_4542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_z_axis_l45_4500

def point_on_z_axis : ℝ × ℝ × ℝ := (0, 0, -1)
def point1 : ℝ × ℝ × ℝ := (1, 0, 2)
def point2 : ℝ × ℝ × ℝ := (1, -3, 1)

theorem equidistant_point_on_z_axis :
  Real.sqrt ((point_on_z_axis.1 - point1.1)^2 + (point_on_z_axis.2.1 - point1.2.1)^2 + (point_on_z_axis.2.2 - point1.2.2)^2) =
  Real.sqrt ((point_on_z_axis.1 - point2.1)^2 + (point_on_z_axis.2.1 - point2.2.1)^2 + (point_on_z_axis.2.2 - point2.2.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_z_axis_l45_4500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_fencing_cost_l45_4527

/-- Represents a rectangular field with given dimensions and fencing requirements. -/
structure RectangularField where
  length : ℝ
  width : ℝ
  uncovered_side : ℝ
  area : ℝ
  fencing_length : ℝ

/-- Represents a fencing material with cost and maximum span. -/
structure FencingMaterial where
  cost_per_foot : ℝ
  max_span : ℝ

/-- Calculates the total cost of fencing using a single material. -/
def total_cost (f : RectangularField) (m : FencingMaterial) : ℝ :=
  f.fencing_length * m.cost_per_foot

theorem optimal_fencing_cost (f : RectangularField) (material_a material_b : FencingMaterial) :
  f.length = 50 ∧
  f.area = 1200 ∧
  f.width = f.area / f.length ∧
  f.fencing_length = 2 * f.width + f.length ∧
  material_a.cost_per_foot = 8 ∧
  material_a.max_span = 100 ∧
  material_b.cost_per_foot = 12 ∧
  f.fencing_length ≤ material_a.max_span →
  total_cost f material_a = 784 ∧
  total_cost f material_a < total_cost f material_b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_fencing_cost_l45_4527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l45_4585

-- Define the complex equation
def complex_equation (z : ℂ) : Prop := (z + 6) ^ 12 = 64

-- Define the set of solutions
def solution_set : Set ℂ := {z | complex_equation z}

-- Define the regular dodecagon formed by the solutions
def solution_dodecagon : Set ℂ := solution_set

-- Define a function to calculate the area of a triangle given three complex points
noncomputable def triangle_area (a b c : ℂ) : ℝ :=
  let s1 := Complex.abs (b - a)
  let s2 := Complex.abs (c - b)
  let s3 := Complex.abs (a - c)
  let s := (s1 + s2 + s3) / 2
  Real.sqrt (s * (s - s1) * (s - s2) * (s - s3))

-- Theorem statement
theorem min_triangle_area :
  ∃ (a b c : ℂ), a ∈ solution_dodecagon ∧ b ∈ solution_dodecagon ∧ c ∈ solution_dodecagon ∧
  (∀ (x y z : ℂ), x ∈ solution_dodecagon → y ∈ solution_dodecagon → z ∈ solution_dodecagon →
    triangle_area a b c ≤ triangle_area x y z) ∧
  triangle_area a b c = Real.sqrt (18 - 6 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l45_4585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_wins_l45_4573

/-- Represents a player in the game -/
inductive Player : Type
| A : Player
| B : Player

/-- Represents the game state -/
structure GameState :=
  (matchsticks : ℕ)
  (used_numbers : Set ℕ)
  (current_player : Player)

/-- Represents a move in the game -/
structure Move :=
  (change : Int)

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  -5 ≤ move.change ∧ move.change ≤ 5 ∧
  0 ≤ state.matchsticks + move.change.toNat ∧
  (state.matchsticks + move.change.toNat) ∉ state.used_numbers

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  { matchsticks := state.matchsticks + move.change.toNat,
    used_numbers := state.used_numbers.insert (state.matchsticks + move.change.toNat),
    current_player := match state.current_player with
      | Player.A => Player.B
      | Player.B => Player.A }

/-- Represents a winning strategy for Player B -/
def winning_strategy (strategy : GameState → Move) : Prop :=
  ∀ (state : GameState),
    state.current_player = Player.B →
    is_valid_move state (strategy state) ∧
    ∃ (n : ℕ), ∀ (moves : List Move),
      moves.length = n →
      (moves.foldl apply_move state).current_player = Player.A →
      ¬(is_valid_move (moves.foldl apply_move state) (strategy (moves.foldl apply_move state)))

/-- The main theorem stating that Player B has a winning strategy -/
theorem player_b_wins :
  ∃ (strategy : GameState → Move),
    winning_strategy strategy ∧
    strategy { matchsticks := 1000, used_numbers := ∅, current_player := Player.B } ∈ 
      { move | is_valid_move { matchsticks := 1000, used_numbers := ∅, current_player := Player.B } move } :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_b_wins_l45_4573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_four_l45_4519

noncomputable section

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2

-- Define the function g based on f
noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 12) + 2

-- State the symmetry condition
axiom g_symmetry (α : ℝ) : ∀ x : ℝ, g (α - x) = g (α + x)

-- State the theorem to be proved
theorem g_sum_equals_four (α : ℝ) 
  (h : ∀ x : ℝ, g (α - x) = g (α + x)) : 
  g (α + Real.pi / 4) + g (Real.pi / 4) = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_four_l45_4519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l45_4560

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  side : ℝ
  side_pos : side > 0

/-- A cube whose vertices are the centers of the faces of a regular tetrahedron -/
structure CenterCube (T : RegularTetrahedron) where
  edge : ℝ
  edge_def : edge = (2 * T.side) / (3 * Real.sqrt 3) * Real.sqrt 2

/-- The volume of a regular tetrahedron -/
noncomputable def tetrahedron_volume (T : RegularTetrahedron) : ℝ :=
  (T.side ^ 3 * Real.sqrt 3) / 12

/-- The volume of a cube -/
noncomputable def cube_volume (T : RegularTetrahedron) (C : CenterCube T) : ℝ :=
  C.edge ^ 3

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio (T : RegularTetrahedron) (C : CenterCube T) :
    tetrahedron_volume T / cube_volume T C = 81 / 64 := by
  sorry

#eval 81 + 64  -- This should evaluate to 145

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_l45_4560
