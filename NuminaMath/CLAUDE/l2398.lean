import Mathlib

namespace NUMINAMATH_CALUDE_characterize_u_l2398_239826

/-- A function is strictly monotonic if it preserves the order relation -/
def StrictlyMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The main theorem statement -/
theorem characterize_u (u : ℝ → ℝ) :
  (∃ f : ℝ → ℝ, StrictlyMonotonic f ∧
    (∀ x y : ℝ, f (x + y) = f x * u y + f y)) →
  (∃ a : ℝ, ∀ x : ℝ, u x = Real.exp (a * x)) :=
by sorry

end NUMINAMATH_CALUDE_characterize_u_l2398_239826


namespace NUMINAMATH_CALUDE_cheaper_module_cost_l2398_239817

-- Define the total number of modules
def total_modules : ℕ := 22

-- Define the number of cheaper modules
def cheaper_modules : ℕ := 21

-- Define the cost of the expensive module
def expensive_module_cost : ℚ := 10

-- Define the total stock value
def total_stock_value : ℚ := 62.5

-- Theorem to prove
theorem cheaper_module_cost :
  ∃ (x : ℚ), x > 0 ∧ x < expensive_module_cost ∧
  x * cheaper_modules + expensive_module_cost = total_stock_value ∧
  x = 2.5 := by sorry

end NUMINAMATH_CALUDE_cheaper_module_cost_l2398_239817


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l2398_239832

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns true if all numbers in the range [a, b] are nonprime, false otherwise -/
def allNonPrime (a b : ℕ) : Prop := sorry

theorem smallest_prime_after_six_nonprimes : 
  ∃ (k : ℕ), 
    isPrime 97 ∧ 
    (∀ p < 97, isPrime p → ¬(allNonPrime (p + 1) (p + 6))) ∧
    allNonPrime 91 96 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l2398_239832


namespace NUMINAMATH_CALUDE_point_quadrant_relation_l2398_239803

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Checks if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem stating that if P(a+b, ab) is in the second quadrant, 
    then Q(-a, b) is in the fourth quadrant -/
theorem point_quadrant_relation (a b : ℝ) :
  isInSecondQuadrant (Point.mk (a + b) (a * b)) →
  isInFourthQuadrant (Point.mk (-a) b) :=
by
  sorry


end NUMINAMATH_CALUDE_point_quadrant_relation_l2398_239803


namespace NUMINAMATH_CALUDE_battery_life_is_19_5_hours_l2398_239883

/-- Represents the tablet's battery and usage characteristics -/
structure TabletBattery where
  passive_life : ℝ  -- Battery life in hours when not actively used
  active_life : ℝ   -- Battery life in hours when actively used
  used_time : ℝ     -- Total time the tablet has been on since last charge
  gaming_time : ℝ   -- Time spent gaming since last charge
  charge_rate_passive : ℝ  -- Additional passive battery life gained per hour of charging
  charge_rate_active : ℝ   -- Additional active battery life gained per hour of charging
  charge_time : ℝ   -- Time spent charging the tablet

/-- Calculates the remaining battery life after usage and charging -/
def remaining_battery_life (tb : TabletBattery) : ℝ :=
  sorry

/-- Theorem stating that the remaining battery life is 19.5 hours -/
theorem battery_life_is_19_5_hours (tb : TabletBattery) 
  (h1 : tb.passive_life = 36)
  (h2 : tb.active_life = 6)
  (h3 : tb.used_time = 15)
  (h4 : tb.gaming_time = 1.5)
  (h5 : tb.charge_rate_passive = 2)
  (h6 : tb.charge_rate_active = 0.5)
  (h7 : tb.charge_time = 3) :
  remaining_battery_life tb = 19.5 :=
sorry

end NUMINAMATH_CALUDE_battery_life_is_19_5_hours_l2398_239883


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l2398_239886

/-- The value of m for which the ellipse x^2 + 9y^2 = 9 is tangent to the hyperbola x^2 - m(y + 3)^2 = 1 -/
theorem ellipse_hyperbola_tangent : ∃ (m : ℝ), 
  (∀ (x y : ℝ), x^2 + 9*y^2 = 9 ∧ x^2 - m*(y + 3)^2 = 1) →
  (∃! (x y : ℝ), x^2 + 9*y^2 = 9 ∧ x^2 - m*(y + 3)^2 = 1) →
  m = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l2398_239886


namespace NUMINAMATH_CALUDE_number_plus_19_equals_47_l2398_239870

theorem number_plus_19_equals_47 (x : ℤ) : x + 19 = 47 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_19_equals_47_l2398_239870


namespace NUMINAMATH_CALUDE_some_number_value_l2398_239874

theorem some_number_value (some_number : ℝ) : 
  (some_number * 10) / 100 = 0.032420000000000004 → 
  some_number = 0.32420000000000004 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l2398_239874


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2398_239852

/-- The function f(x) = -9x^2 + 27x + 15 has a maximum value of 141/4. -/
theorem max_value_quadratic : ∃ (M : ℝ), M = (141 : ℝ) / 4 ∧ 
  ∀ (x : ℝ), -9 * x^2 + 27 * x + 15 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2398_239852


namespace NUMINAMATH_CALUDE_fred_grew_nine_onions_l2398_239833

/-- The number of onions Sally grew -/
def sally_onions : ℕ := 5

/-- The number of onions Sally and Fred gave away -/
def onions_given_away : ℕ := 4

/-- The number of onions Sally and Fred have remaining -/
def onions_remaining : ℕ := 10

/-- The number of onions Fred grew -/
def fred_onions : ℕ := sally_onions + onions_given_away + onions_remaining - sally_onions - onions_given_away

theorem fred_grew_nine_onions : fred_onions = 9 := by
  sorry

end NUMINAMATH_CALUDE_fred_grew_nine_onions_l2398_239833


namespace NUMINAMATH_CALUDE_vectors_form_basis_l2398_239871

theorem vectors_form_basis (a b : ℝ × ℝ) : 
  a = (1, 2) → b = (3, 1) → LinearIndependent ℝ ![a, b] :=
by
  sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l2398_239871


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l2398_239810

def P : Set ℝ := {x | x ≤ 0 ∨ x > 3}
def Q : Set ℝ := {0, 1, 2, 3}

theorem complement_P_intersect_Q :
  (Set.compl P) ∩ Q = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l2398_239810


namespace NUMINAMATH_CALUDE_no_real_roots_l2398_239820

-- Define the sequence of polynomials P_n(x)
def P : ℕ → (ℝ → ℝ)
  | 0 => λ _ => 1
  | n + 1 => λ x => x^(17*(n+1)) - P n x

-- Theorem statement
theorem no_real_roots : ∀ n : ℕ, ∀ x : ℝ, P n x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2398_239820


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2398_239825

/-- A geometric sequence with third term 5 and sixth term 40 has first term 5/4 -/
theorem geometric_sequence_first_term (a : ℝ) (r : ℝ) : 
  a * r^2 = 5 → a * r^5 = 40 → a = 5/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2398_239825


namespace NUMINAMATH_CALUDE_candy_solution_l2398_239850

def candy_problem (f b j : ℕ) : Prop :=
  f = 12 ∧ b = f + 6 ∧ j = 10 * (f + b)

theorem candy_solution : 
  ∀ f b j : ℕ, candy_problem f b j → (40 * j^2) / 100 = 36000 := by
  sorry

end NUMINAMATH_CALUDE_candy_solution_l2398_239850


namespace NUMINAMATH_CALUDE_max_value_7b_5c_l2398_239823

/-- The function f(x) = ax^2 + bx + c -/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the maximum value of 7b+5c given the conditions -/
theorem max_value_7b_5c (a b c : ℝ) : 
  (∃ a' ∈ Set.Icc 1 2, ∀ x ∈ Set.Icc 1 2, f a' b c x ≤ 1) →
  (∀ y : ℝ, 7 * b + 5 * c ≤ y) → y = -6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_7b_5c_l2398_239823


namespace NUMINAMATH_CALUDE_max_y_value_l2398_239855

theorem max_y_value (x y : ℤ) (h : 2*x*y + 8*x + 2*y = -14) : 
  ∃ (max_y : ℤ), (∃ (x' : ℤ), 2*x'*max_y + 8*x' + 2*max_y = -14) ∧ 
  (∀ (y' : ℤ), (∃ (x'' : ℤ), 2*x''*y' + 8*x'' + 2*y' = -14) → y' ≤ max_y) ∧
  max_y = 5 := by
sorry

end NUMINAMATH_CALUDE_max_y_value_l2398_239855


namespace NUMINAMATH_CALUDE_multiple_is_two_l2398_239842

-- Define the variables
def mother_age : ℕ := 40
def daughter_age : ℕ := 30 -- This is derived, not given directly
def multiple : ℚ := 2 -- This is what we want to prove

-- Define the conditions
def condition1 (m : ℕ) (d : ℕ) (x : ℚ) : Prop :=
  m + x * d = 70

def condition2 (m : ℕ) (d : ℕ) (x : ℚ) : Prop :=
  d + x * m = 95

-- Theorem statement
theorem multiple_is_two :
  condition1 mother_age daughter_age multiple ∧
  condition2 mother_age daughter_age multiple ∧
  multiple = 2 := by sorry

end NUMINAMATH_CALUDE_multiple_is_two_l2398_239842


namespace NUMINAMATH_CALUDE_perp_para_implies_perp_line_perp_para_planes_implies_perp_perp_two_planes_implies_para_l2398_239869

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (paraPlane : Plane → Plane → Prop)

-- Axioms for the relations
axiom perp_antisymm {l : Line} {p : Plane} : perp l p ↔ perp l p
axiom para_antisymm {l : Line} {p : Plane} : para l p ↔ para l p
axiom perpLine_antisymm {l1 l2 : Line} : perpLine l1 l2 ↔ perpLine l2 l1
axiom paraPlane_antisymm {p1 p2 : Plane} : paraPlane p1 p2 ↔ paraPlane p2 p1

-- Theorem 1
theorem perp_para_implies_perp_line {m n : Line} {α : Plane} 
  (h1 : perp m α) (h2 : para n α) : perpLine m n := by sorry

-- Theorem 2
theorem perp_para_planes_implies_perp {m : Line} {α β : Plane}
  (h1 : perp m α) (h2 : paraPlane α β) : perp m β := by sorry

-- Theorem 3
theorem perp_two_planes_implies_para {m : Line} {α β : Plane}
  (h1 : perp m α) (h2 : perp m β) : paraPlane α β := by sorry

end NUMINAMATH_CALUDE_perp_para_implies_perp_line_perp_para_planes_implies_perp_perp_two_planes_implies_para_l2398_239869


namespace NUMINAMATH_CALUDE_arrangements_with_specific_people_at_ends_l2398_239844

/-- The number of permutations of n distinct objects. -/
def permutations (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange m objects out of n distinct objects. -/
def arrangements (n m : ℕ) : ℕ := 
  if m ≤ n then
    permutations n / permutations (n - m)
  else
    0

theorem arrangements_with_specific_people_at_ends (total_people : ℕ) 
  (specific_people : ℕ) (h : total_people = 6 ∧ specific_people = 2) : 
  permutations total_people - 
  (arrangements (total_people - 2) specific_people * permutations (total_people - specific_people)) = 432 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_specific_people_at_ends_l2398_239844


namespace NUMINAMATH_CALUDE_mistaken_division_l2398_239860

theorem mistaken_division (n : ℕ) : 
  (n / 9 = 8 ∧ n % 9 = 6) → n / 6 = 13 := by
sorry

end NUMINAMATH_CALUDE_mistaken_division_l2398_239860


namespace NUMINAMATH_CALUDE_sin_arctan_equality_l2398_239816

theorem sin_arctan_equality : ∃ (x : ℝ), x > 0 ∧ Real.sin (Real.arctan x) = x := by
  let x := Real.sqrt ((-1 + Real.sqrt 5) / 2)
  use x
  have h1 : x > 0 := sorry
  have h2 : Real.sin (Real.arctan x) = x := sorry
  exact ⟨h1, h2⟩

#check sin_arctan_equality

end NUMINAMATH_CALUDE_sin_arctan_equality_l2398_239816


namespace NUMINAMATH_CALUDE_count_hundredths_in_half_l2398_239898

theorem count_hundredths_in_half : (0.5 : ℚ) / (0.01 : ℚ) = 50 := by sorry

end NUMINAMATH_CALUDE_count_hundredths_in_half_l2398_239898


namespace NUMINAMATH_CALUDE_self_inverse_sum_zero_l2398_239811

def is_self_inverse (a d : ℝ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, 5; -12, d]
  M * M = 1

theorem self_inverse_sum_zero (a d : ℝ) (h : is_self_inverse a d) : a + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_self_inverse_sum_zero_l2398_239811


namespace NUMINAMATH_CALUDE_complex_arithmetic_result_l2398_239848

theorem complex_arithmetic_result : 
  ((2 - 3*I) + (4 + 6*I)) * (-1 + 2*I) = -12 + 9*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_result_l2398_239848


namespace NUMINAMATH_CALUDE_double_root_values_l2398_239892

/-- A polynomial with integer coefficients of the form x^4 + b₃x³ + b₂x² + b₁x + 50 -/
def IntPolynomial (b₃ b₂ b₁ : ℤ) (x : ℝ) : ℝ := x^4 + b₃*x^3 + b₂*x^2 + b₁*x + 50

/-- s is a double root of the polynomial if both the polynomial and its derivative evaluate to 0 at s -/
def IsDoubleRoot (p : ℝ → ℝ) (s : ℝ) : Prop :=
  p s = 0 ∧ (deriv p) s = 0

theorem double_root_values (b₃ b₂ b₁ : ℤ) (s : ℤ) :
  IsDoubleRoot (IntPolynomial b₃ b₂ b₁) s → s = -5 ∨ s = -1 ∨ s = 1 ∨ s = 5 := by
  sorry

end NUMINAMATH_CALUDE_double_root_values_l2398_239892


namespace NUMINAMATH_CALUDE_correct_system_of_equations_l2398_239806

/-- Represents the number of workers in the workshop -/
def total_workers : ℕ := 26

/-- Represents the number of screws a worker can produce per day -/
def screws_per_worker : ℕ := 800

/-- Represents the number of nuts a worker can produce per day -/
def nuts_per_worker : ℕ := 1000

/-- Represents the number of nuts needed to match one screw -/
def nuts_per_screw : ℕ := 2

/-- Theorem stating the correct system of equations for matching screws and nuts -/
theorem correct_system_of_equations (x y : ℕ) :
  (x + y = total_workers) ∧
  (nuts_per_worker * y = nuts_per_screw * screws_per_worker * x) →
  (x + y = total_workers) ∧
  (1000 * y = 2 * 800 * x) :=
by sorry

end NUMINAMATH_CALUDE_correct_system_of_equations_l2398_239806


namespace NUMINAMATH_CALUDE_trapezoid_area_l2398_239802

/-- The area of a trapezoid with given vertices in a standard rectangular coordinate system -/
theorem trapezoid_area (E F G H : ℝ × ℝ) : 
  E = (2, -3) → 
  F = (2, 2) → 
  G = (7, 8) → 
  H = (7, 3) → 
  (1/2 : ℝ) * ((F.2 - E.2) + (G.2 - H.2)) * (G.1 - E.1) = 25 := by
  sorry

#check trapezoid_area

end NUMINAMATH_CALUDE_trapezoid_area_l2398_239802


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l2398_239830

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (length width : ℝ),
    r = 6 →
    length = 3 * width →
    width = 2 * r →
    length * width = 432 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l2398_239830


namespace NUMINAMATH_CALUDE_mod_twelve_six_nine_l2398_239872

theorem mod_twelve_six_nine (n : ℕ) : 12^6 ≡ n [ZMOD 9] → 0 ≤ n → n < 9 → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_twelve_six_nine_l2398_239872


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_and_fraction_l2398_239854

noncomputable def α : ℝ := Real.arctan 3 * 2

theorem tan_alpha_plus_pi_third_and_fraction (h : Real.tan (α/2) = 3) :
  Real.tan (α + π/3) = (48 - 25 * Real.sqrt 3) / 11 ∧
  (Real.sin α + 2 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = -5/17 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_and_fraction_l2398_239854


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2398_239887

def A : Set ℝ := {x : ℝ | (2*x - 5)*(x + 3) > 0}
def B : Set ℝ := {1, 2, 3, 4, 5}

theorem complement_A_intersect_B : 
  (Set.compl A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2398_239887


namespace NUMINAMATH_CALUDE_negation_of_implication_l2398_239867

theorem negation_of_implication :
  (¬(∀ x : ℝ, x > 5 → x > 0)) ↔ (∀ x : ℝ, x ≤ 5 → x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2398_239867


namespace NUMINAMATH_CALUDE_volume_of_sphere_wedge_l2398_239863

/-- The volume of a wedge from a sphere cut into six congruent parts, given the sphere's circumference --/
theorem volume_of_sphere_wedge (circumference : ℝ) :
  circumference = 18 * Real.pi →
  (1 / 6) * (4 / 3) * Real.pi * (circumference / (2 * Real.pi))^3 = 162 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_volume_of_sphere_wedge_l2398_239863


namespace NUMINAMATH_CALUDE_hamburger_combinations_l2398_239838

/-- The number of different condiments available. -/
def num_condiments : ℕ := 10

/-- The number of patty options available. -/
def num_patty_options : ℕ := 4

/-- The total number of different hamburger combinations. -/
def total_combinations : ℕ := 2^num_condiments * num_patty_options

/-- Theorem stating that the total number of different hamburger combinations is 4096. -/
theorem hamburger_combinations : total_combinations = 4096 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l2398_239838


namespace NUMINAMATH_CALUDE_suv_city_mpg_l2398_239880

/-- The average miles per gallon (mpg) for an SUV in the city. -/
def city_mpg : ℝ := 12.2

/-- The maximum distance in miles that the SUV can travel on 20 gallons of gasoline. -/
def max_distance : ℝ := 244

/-- The amount of gasoline in gallons used for the maximum distance. -/
def gas_amount : ℝ := 20

/-- Theorem stating that the average mpg in the city for the SUV is 12.2,
    given the maximum distance on 20 gallons of gasoline is 244 miles. -/
theorem suv_city_mpg :
  city_mpg = max_distance / gas_amount :=
by sorry

end NUMINAMATH_CALUDE_suv_city_mpg_l2398_239880


namespace NUMINAMATH_CALUDE_land_area_decreases_l2398_239862

theorem land_area_decreases (a : ℝ) (h : a > 4) : a^2 > (a+4)*(a-4) := by
  sorry

end NUMINAMATH_CALUDE_land_area_decreases_l2398_239862


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2398_239849

theorem pie_eating_contest (erik_pie frank_pie : Float) 
  (h1 : erik_pie = 0.6666666666666666)
  (h2 : frank_pie = 0.3333333333333333) :
  erik_pie - frank_pie = 0.3333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2398_239849


namespace NUMINAMATH_CALUDE_no_consecutive_prime_roots_l2398_239864

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if two numbers are consecutive primes -/
def areConsecutivePrimes (p q : ℕ) : Prop := sorry

/-- The theorem stating that there are no values of k satisfying the conditions -/
theorem no_consecutive_prime_roots :
  ¬ ∃ (k : ℤ) (p q : ℕ), 
    p < q ∧ 
    areConsecutivePrimes p q ∧ 
    p + q = 65 ∧ 
    p * q = k ∧
    ∀ (x : ℤ), x^2 - 65*x + k = 0 ↔ (x = p ∨ x = q) :=
sorry

end NUMINAMATH_CALUDE_no_consecutive_prime_roots_l2398_239864


namespace NUMINAMATH_CALUDE_matrix_N_property_l2398_239843

open Matrix

theorem matrix_N_property (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec ![3, -2] = ![4, 1])
  (h2 : N.mulVec ![-4, 6] = ![-2, 0]) :
  N.mulVec ![7, 0] = ![6, 2] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_property_l2398_239843


namespace NUMINAMATH_CALUDE_developed_countries_completed_transformation_l2398_239846

-- Define the different stages of population growth patterns
inductive PopulationGrowthStage
| Traditional
| Transitional
| Modern

-- Define the types of countries
inductive CountryType
| Developed
| Developing

-- Define the world population distribution
def worldPopulationDistribution : CountryType → Bool
| CountryType.Developing => true
| CountryType.Developed => false

-- Define the population growth stage for each country type
def populationGrowthStage : CountryType → PopulationGrowthStage
| CountryType.Developing => PopulationGrowthStage.Traditional
| CountryType.Developed => PopulationGrowthStage.Modern

-- Define the overall global population growth stage
def globalPopulationGrowthStage : PopulationGrowthStage :=
  PopulationGrowthStage.Transitional

-- Theorem statement
theorem developed_countries_completed_transformation :
  (∀ c : CountryType, worldPopulationDistribution c → populationGrowthStage c = PopulationGrowthStage.Traditional) →
  globalPopulationGrowthStage = PopulationGrowthStage.Transitional →
  populationGrowthStage CountryType.Developed = PopulationGrowthStage.Modern :=
by
  sorry

end NUMINAMATH_CALUDE_developed_countries_completed_transformation_l2398_239846


namespace NUMINAMATH_CALUDE_M_eq_real_l2398_239837

/-- The set M of complex numbers z where (z-1)^2 = |z-1|^2 -/
def M : Set ℂ := {z : ℂ | (z - 1)^2 = Complex.abs (z - 1)^2}

/-- Theorem stating that M is equal to the set of real numbers -/
theorem M_eq_real : M = {z : ℂ | z.im = 0} := by sorry

end NUMINAMATH_CALUDE_M_eq_real_l2398_239837


namespace NUMINAMATH_CALUDE_pencil_distribution_l2398_239822

theorem pencil_distribution (num_pens : ℕ) (num_pencils : ℕ) (num_students : ℕ) :
  num_pens = 781 →
  num_students = 71 →
  num_pens % num_students = 0 →
  num_pencils % num_students = 0 →
  ∃ k : ℕ, num_pencils = 71 * k :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2398_239822


namespace NUMINAMATH_CALUDE_tree_height_after_three_good_years_l2398_239847

/-- Represents the growth factor of a tree in different conditions -/
inductive GrowthCondition
| Good
| Bad

/-- Calculates the height of a tree after a given number of years -/
def treeHeight (initialHeight : ℝ) (years : ℕ) (conditions : List GrowthCondition) : ℝ :=
  match years, conditions with
  | 0, _ => initialHeight
  | n+1, [] => initialHeight  -- Default to initial height if no conditions are specified
  | n+1, c::cs => 
    let newHeight := 
      match c with
      | GrowthCondition.Good => 3 * initialHeight
      | GrowthCondition.Bad => 2 * initialHeight
    treeHeight newHeight n cs

/-- Theorem stating the height of the tree after 3 years of good growth -/
theorem tree_height_after_three_good_years :
  let initialHeight : ℝ := treeHeight 1458 3 [GrowthCondition.Bad, GrowthCondition.Bad, GrowthCondition.Bad]
  treeHeight initialHeight 3 [GrowthCondition.Good, GrowthCondition.Good, GrowthCondition.Good] = 1458 :=
by sorry

#eval treeHeight 1458 3 [GrowthCondition.Bad, GrowthCondition.Bad, GrowthCondition.Bad]

end NUMINAMATH_CALUDE_tree_height_after_three_good_years_l2398_239847


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_zero_l2398_239866

theorem sum_of_a_and_b_is_zero (a b : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a + i) / i = 1 + b * i) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_zero_l2398_239866


namespace NUMINAMATH_CALUDE_vertices_form_hyperbola_branch_l2398_239813

/-- Given a real number k and a constant c, the set of vertices (x_t, y_t) of the parabola
    y = t^2 x^2 + 2ktx + c for varying t forms one branch of a hyperbola. -/
theorem vertices_form_hyperbola_branch (k : ℝ) (c : ℝ) :
  ∃ (A B C D : ℝ), A ≠ 0 ∧
    (∀ x_t y_t : ℝ, x_t ≠ 0 →
      (∃ t : ℝ, y_t = t^2 * x_t^2 + 2*k*t*x_t + c ∧
                x_t = -k/t) →
      A * x_t * y_t + B * x_t + C * y_t + D = 0) :=
sorry

end NUMINAMATH_CALUDE_vertices_form_hyperbola_branch_l2398_239813


namespace NUMINAMATH_CALUDE_fibonacci_geometric_sequence_l2398_239804

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the theorem
theorem fibonacci_geometric_sequence (a k p : ℕ) :
  (∃ r : ℚ, r > 1 ∧ fib k = r * fib a ∧ fib p = r * fib k) →  -- Geometric sequence condition
  (a < k ∧ k < p) →  -- Increasing order condition
  (a + k + p = 2010) →  -- Sum condition
  a = 669 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_geometric_sequence_l2398_239804


namespace NUMINAMATH_CALUDE_lawn_area_l2398_239845

/-- Calculates the area of a lawn in a rectangular park with crossroads -/
theorem lawn_area (park_length park_width road_width : ℝ) 
  (h1 : park_length = 60)
  (h2 : park_width = 40)
  (h3 : road_width = 3) : 
  park_length * park_width - 
  (park_length * road_width + park_width * road_width - road_width * road_width) = 2109 :=
by sorry

end NUMINAMATH_CALUDE_lawn_area_l2398_239845


namespace NUMINAMATH_CALUDE_percentage_increase_theorem_l2398_239815

theorem percentage_increase_theorem (initial_value : ℝ) 
  (first_increase_percent : ℝ) (second_increase_percent : ℝ) :
  let first_increase := initial_value * (1 + first_increase_percent / 100)
  let final_value := first_increase * (1 + second_increase_percent / 100)
  initial_value = 5000 ∧ first_increase_percent = 65 ∧ second_increase_percent = 45 →
  final_value = 11962.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_theorem_l2398_239815


namespace NUMINAMATH_CALUDE_shirt_fixing_time_l2398_239894

/-- Proves that the time to fix a shirt is 1.5 hours given the problem conditions --/
theorem shirt_fixing_time (num_shirts : ℕ) (num_pants : ℕ) (hourly_rate : ℚ) (total_cost : ℚ) :
  num_shirts = 10 →
  num_pants = 12 →
  hourly_rate = 30 →
  total_cost = 1530 →
  ∃ (time_per_shirt : ℚ),
    time_per_shirt = 3/2 ∧
    total_cost = hourly_rate * (num_shirts * time_per_shirt + num_pants * (2 * time_per_shirt)) :=
by sorry

end NUMINAMATH_CALUDE_shirt_fixing_time_l2398_239894


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l2398_239814

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |x^2 + a*x + 2| ≤ 4) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l2398_239814


namespace NUMINAMATH_CALUDE_room_length_proof_l2398_239861

/-- The length of a rectangular room given its width, number of tiles, and tile size. -/
theorem room_length_proof (width : ℝ) (num_tiles : ℕ) (tile_size : ℝ) 
  (h1 : width = 12)
  (h2 : num_tiles = 6)
  (h3 : tile_size = 4)
  : width * (num_tiles * tile_size / width) = 2 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l2398_239861


namespace NUMINAMATH_CALUDE_area_of_region_S_l2398_239808

/-- A rhombus with side length 3 and angle B = 150° --/
structure Rhombus :=
  (side_length : ℝ)
  (angle_B : ℝ)
  (h_side : side_length = 3)
  (h_angle : angle_B = 150)

/-- The region S inside the rhombus closer to vertex B than to A, C, or D --/
def region_S (r : Rhombus) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² --/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the area of region S is approximately 1.1 --/
theorem area_of_region_S (r : Rhombus) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |area (region_S r) - 1.1| < ε :=
sorry

end NUMINAMATH_CALUDE_area_of_region_S_l2398_239808


namespace NUMINAMATH_CALUDE_triangle_theorem_l2398_239885

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (Real.cos t.A * Real.cos t.B - Real.sin t.A * Real.sin t.B) = Real.cos (2 * t.C))
  (h2 : 2 * t.c = t.a + t.b)
  (h3 : t.a * t.b * Real.cos t.C = 18) :
  t.C = Real.pi / 3 ∧ t.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2398_239885


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l2398_239884

/-- The parabola equation -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2/3 - x^2 = 1

/-- The directrix of the parabola -/
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

/-- Point F is the focus of the parabola -/
def focus (p : ℝ) (F : ℝ × ℝ) : Prop := F.1 = p/2 ∧ F.2 = 0

/-- Points M and N are the intersections of the directrix and hyperbola -/
def intersection_points (p : ℝ) (M N : ℝ × ℝ) : Prop :=
  directrix p M.1 ∧ hyperbola M.1 M.2 ∧
  directrix p N.1 ∧ hyperbola N.1 N.2

/-- Triangle MNF is a right-angled triangle with F as the right angle vertex -/
def right_triangle (F M N : ℝ × ℝ) : Prop :=
  (M.1 - F.1)^2 + (M.2 - F.2)^2 + (N.1 - F.1)^2 + (N.2 - F.2)^2 =
  (M.1 - N.1)^2 + (M.2 - N.2)^2

theorem parabola_hyperbola_intersection (p : ℝ) (F M N : ℝ × ℝ) :
  parabola p F.1 F.2 →
  focus p F →
  intersection_points p M N →
  right_triangle F M N →
  p = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l2398_239884


namespace NUMINAMATH_CALUDE_jack_stair_step_height_l2398_239835

/-- Given Jack's stair climbing scenario, prove the height of each step. -/
theorem jack_stair_step_height :
  -- Net flights descended
  ∀ (net_flights : ℕ),
  -- Steps per flight
  ∀ (steps_per_flight : ℕ),
  -- Total descent in inches
  ∀ (total_descent : ℕ),
  -- Given conditions
  net_flights = 3 →
  steps_per_flight = 12 →
  total_descent = 288 →
  -- Prove that the height of each step is 8 inches
  (total_descent : ℚ) / (net_flights * steps_per_flight : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_stair_step_height_l2398_239835


namespace NUMINAMATH_CALUDE_melissa_games_played_l2398_239891

theorem melissa_games_played (points_per_game : ℕ) (total_points : ℕ) (h1 : points_per_game = 12) (h2 : total_points = 36) :
  total_points / points_per_game = 3 := by
  sorry

end NUMINAMATH_CALUDE_melissa_games_played_l2398_239891


namespace NUMINAMATH_CALUDE_jogger_distance_ahead_l2398_239824

/-- Calculates the distance a jogger is ahead of a train given their speeds and the time it takes for the train to pass the jogger. -/
theorem jogger_distance_ahead (jogger_speed train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  passing_time = 38 →
  (train_speed - jogger_speed) * passing_time - train_length = 260 :=
by sorry

end NUMINAMATH_CALUDE_jogger_distance_ahead_l2398_239824


namespace NUMINAMATH_CALUDE_x_percent_of_z_l2398_239809

theorem x_percent_of_z (x y z : ℝ) (h1 : x = 1.30 * y) (h2 : y = 0.50 * z) : x = 0.65 * z := by
  sorry

end NUMINAMATH_CALUDE_x_percent_of_z_l2398_239809


namespace NUMINAMATH_CALUDE_bankers_gain_interest_rate_l2398_239839

/-- Given a banker's gain, present worth, and time period, 
    calculate the annual interest rate. -/
theorem bankers_gain_interest_rate 
  (bankers_gain : ℝ) 
  (present_worth : ℝ) 
  (time_period : ℕ) 
  (h1 : bankers_gain = 36) 
  (h2 : present_worth = 400) 
  (h3 : time_period = 3) :
  ∃ r : ℝ, bankers_gain = present_worth * (1 + r)^time_period - present_worth :=
sorry

end NUMINAMATH_CALUDE_bankers_gain_interest_rate_l2398_239839


namespace NUMINAMATH_CALUDE_dispersion_measures_l2398_239812

/-- A sample of data points -/
def Sample : Type := List ℝ

/-- Standard deviation of a sample -/
noncomputable def standardDeviation (s : Sample) : ℝ := sorry

/-- Range of a sample -/
noncomputable def range (s : Sample) : ℝ := sorry

/-- Median of a sample -/
noncomputable def median (s : Sample) : ℝ := sorry

/-- Mean of a sample -/
noncomputable def mean (s : Sample) : ℝ := sorry

/-- A measure of dispersion is a function that quantifies the spread of a sample -/
def isDispersionMeasure (f : Sample → ℝ) : Prop := sorry

theorem dispersion_measures (s : Sample) :
  isDispersionMeasure standardDeviation ∧
  isDispersionMeasure range ∧
  ¬isDispersionMeasure median ∧
  ¬isDispersionMeasure mean :=
sorry

end NUMINAMATH_CALUDE_dispersion_measures_l2398_239812


namespace NUMINAMATH_CALUDE_dodecagon_vertex_product_l2398_239890

/-- Regular dodecagon in the complex plane -/
structure RegularDodecagon where
  center : ℂ
  vertex : ℂ

/-- The product of the complex representations of all vertices of a regular dodecagon -/
def vertexProduct (d : RegularDodecagon) : ℂ :=
  (d.center + 1)^12 - 1

/-- Theorem: The product of vertices of a regular dodecagon with center (2,1) and a vertex at (3,1) -/
theorem dodecagon_vertex_product :
  let d : RegularDodecagon := { center := 2 + 1*I, vertex := 3 + 1*I }
  vertexProduct d = -2926 - 3452*I :=
by
  sorry

end NUMINAMATH_CALUDE_dodecagon_vertex_product_l2398_239890


namespace NUMINAMATH_CALUDE_paving_stones_required_l2398_239858

theorem paving_stones_required (courtyard_length courtyard_width stone_length stone_width : ℝ) 
  (h1 : courtyard_length = 75)
  (h2 : courtyard_width = 20 + 3/4)
  (h3 : stone_length = 3 + 1/4)
  (h4 : stone_width = 2 + 1/2) : 
  ⌈(courtyard_length * courtyard_width) / (stone_length * stone_width)⌉ = 192 := by
  sorry

end NUMINAMATH_CALUDE_paving_stones_required_l2398_239858


namespace NUMINAMATH_CALUDE_f_monotone_iff_a_range_f_lower_bound_l2398_239840

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - x^2 - a*x

theorem f_monotone_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 2 - 2 * Real.log 2 :=
sorry

theorem f_lower_bound (x : ℝ) (hx : x > 0) :
  f 1 x > 1 - (Real.log 2) / 2 - ((Real.log 2) / 2)^2 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_iff_a_range_f_lower_bound_l2398_239840


namespace NUMINAMATH_CALUDE_range_of_a_for_subset_intersection_when_a_is_4_l2398_239834

/-- The set A depending on the parameter a -/
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - 2*a - 5) < 0}

/-- The set B depending on the parameter a -/
def B (a : ℝ) : Set ℝ := {x | x > 2*a ∧ x < a^2 + 2}

/-- The theorem stating the range of a -/
theorem range_of_a_for_subset (a : ℝ) : 
  (a > -3/2) → (B a ⊆ A a) → (1 ≤ a ∧ a ≤ 3) :=
sorry

/-- The theorem for the specific case when a = 4 -/
theorem intersection_when_a_is_4 : 
  A 4 ∩ B 4 = {x | 8 < x ∧ x < 13} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_subset_intersection_when_a_is_4_l2398_239834


namespace NUMINAMATH_CALUDE_square_root_three_expansion_l2398_239882

theorem square_root_three_expansion 
  (a b m n : ℕ+) 
  (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n := by
  sorry

end NUMINAMATH_CALUDE_square_root_three_expansion_l2398_239882


namespace NUMINAMATH_CALUDE_work_completion_time_l2398_239828

theorem work_completion_time (b_days : ℝ) (a_wage_ratio : ℝ) (a_days : ℝ) : 
  b_days = 15 →
  a_wage_ratio = 3/5 →
  a_wage_ratio = (1/a_days) / (1/a_days + 1/b_days) →
  a_days = 10 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2398_239828


namespace NUMINAMATH_CALUDE_exists_m_with_x_squared_leq_eight_l2398_239878

theorem exists_m_with_x_squared_leq_eight : ∃ m : ℝ, m ≤ 2 ∧ ∃ x > m, x^2 ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_with_x_squared_leq_eight_l2398_239878


namespace NUMINAMATH_CALUDE_units_digit_plus_two_l2398_239868

/-- Given a positive even integer with a positive units digit, 
    if the units digit of its cube minus the units digit of its square is 0, 
    then the units digit of the number plus 2 is 8. -/
theorem units_digit_plus_two (p : ℕ) : 
  p > 0 → 
  Even p → 
  (p % 10 > 0) → 
  ((p^3 % 10) - (p^2 % 10) = 0) → 
  ((p + 2) % 10 = 8) := by
sorry

end NUMINAMATH_CALUDE_units_digit_plus_two_l2398_239868


namespace NUMINAMATH_CALUDE_soccer_team_wins_l2398_239851

theorem soccer_team_wins (total_games : ℕ) (win_percentage : ℚ) (games_won : ℕ) : 
  total_games = 158 →
  win_percentage = 40 / 100 →
  games_won = (total_games : ℚ) * win_percentage →
  games_won = 63 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_wins_l2398_239851


namespace NUMINAMATH_CALUDE_counterfeit_coin_strategy_exists_l2398_239879

/-- Represents a weighing operation that can compare two groups of coins. -/
def Weighing := List Nat → List Nat → Ordering

/-- Represents a strategy for finding the counterfeit coin. -/
def Strategy := List Nat → List Weighing → Option Nat

/-- The number of coins. -/
def n : Nat := 81

/-- The maximum number of weighings allowed. -/
def max_weighings : Nat := 4

/-- Theorem stating that there exists a strategy to find the counterfeit coin. -/
theorem counterfeit_coin_strategy_exists :
  ∃ (s : Strategy),
    ∀ (counterfeit : Nat),
      counterfeit < n →
      ∃ (weighings : List Weighing),
        weighings.length ≤ max_weighings ∧
        s (List.range n) weighings = some counterfeit :=
by sorry

end NUMINAMATH_CALUDE_counterfeit_coin_strategy_exists_l2398_239879


namespace NUMINAMATH_CALUDE_floor_painting_theorem_l2398_239859

/-- The number of integer solutions to the floor painting problem -/
def floor_painting_solutions : Nat :=
  (Finset.filter
    (fun p : Nat × Nat =>
      let a := p.1
      let b := p.2
      b > a ∧ (a - 4) * (b - 4) = a * b / 2)
    (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- The floor painting problem has exactly 3 solutions -/
theorem floor_painting_theorem : floor_painting_solutions = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_theorem_l2398_239859


namespace NUMINAMATH_CALUDE_unique_valid_number_l2398_239873

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000 ∧ n < 10000000) ∧
  (∀ d : ℕ, d < 7 → (∃! i : ℕ, i < 7 ∧ (n / 10^i) % 10 = d)) ∧
  (n % 100 % 2 = 0 ∧ (n / 100000) % 100 % 2 = 0) ∧
  (n % 1000 % 3 = 0 ∧ (n / 10000) % 1000 % 3 = 0) ∧
  (n % 10000 % 4 = 0 ∧ (n / 1000) % 10000 % 4 = 0) ∧
  (n % 100000 % 5 = 0 ∧ (n / 100) % 100000 % 5 = 0) ∧
  (n % 1000000 % 6 = 0 ∧ (n / 10) % 1000000 % 6 = 0)

theorem unique_valid_number : ∃! n : ℕ, is_valid_number n ∧ n = 3216540 := by
  sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2398_239873


namespace NUMINAMATH_CALUDE_sin_2alpha_in_terms_of_y_l2398_239807

theorem sin_2alpha_in_terms_of_y (α y : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : y > 0) 
  (h3 : Real.cos (α/2) = Real.sqrt ((y+1)/(2*y))) : 
  Real.sin (2*α) = (2 * Real.sqrt (y^2 - 1)) / y := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_in_terms_of_y_l2398_239807


namespace NUMINAMATH_CALUDE_parallel_segments_between_parallel_planes_l2398_239881

/-- Two planes are parallel if they do not intersect -/
def ParallelPlanes (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry

/-- A line segment between two planes -/
def LineSegmentBetweenPlanes (p q : Set (ℝ × ℝ × ℝ)) (s : Set (ℝ × ℝ × ℝ)) : Prop := sorry

/-- Two line segments are parallel -/
def ParallelLineSegments (s t : Set (ℝ × ℝ × ℝ)) : Prop := sorry

/-- The length of a line segment -/
def LengthOfLineSegment (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

theorem parallel_segments_between_parallel_planes 
  (p q : Set (ℝ × ℝ × ℝ)) 
  (s t : Set (ℝ × ℝ × ℝ)) :
  ParallelPlanes p q →
  LineSegmentBetweenPlanes p q s →
  LineSegmentBetweenPlanes p q t →
  ParallelLineSegments s t →
  LengthOfLineSegment s = LengthOfLineSegment t := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_between_parallel_planes_l2398_239881


namespace NUMINAMATH_CALUDE_vector_equation_l2398_239889

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (4, 2)

theorem vector_equation : c = 3 • a - b := by sorry

end NUMINAMATH_CALUDE_vector_equation_l2398_239889


namespace NUMINAMATH_CALUDE_real_roots_condition_l2398_239819

theorem real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - x + 3 = 0) ↔ (k ≤ 1/12 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_real_roots_condition_l2398_239819


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_ratio_l2398_239829

/-- Given two circles A and B where A is inside B, this theorem proves the diameter of A
    given the diameter of B and the ratio of areas. -/
theorem circle_diameter_from_area_ratio (dB : ℝ) (r : ℝ) :
  dB = 20 →  -- Diameter of circle B is 20 cm
  r = 1/7 →  -- Ratio of area of A to shaded area is 1:7
  ∃ dA : ℝ,  -- There exists a diameter for circle A
    (π * (dA/2)^2) / (π * (dB/2)^2 - π * (dA/2)^2) = r ∧  -- Area ratio condition
    abs (dA - 7.08) < 0.01  -- Diameter of A is approximately 7.08 cm
    := by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_ratio_l2398_239829


namespace NUMINAMATH_CALUDE_power_product_equality_l2398_239876

theorem power_product_equality : (3^3 * 5^3) * (3^8 * 5^8) = 15^11 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2398_239876


namespace NUMINAMATH_CALUDE_double_then_half_sixteen_l2398_239827

theorem double_then_half_sixteen : 
  let initial_number := 16
  let doubled := initial_number * 2
  let halved := doubled / 2
  halved = 2^4 := by sorry

end NUMINAMATH_CALUDE_double_then_half_sixteen_l2398_239827


namespace NUMINAMATH_CALUDE_tank_capacity_calculation_l2398_239895

/-- Represents a tank with a leak and an inlet pipe. -/
structure Tank where
  capacity : ℝ
  leak_empty_time : ℝ
  inlet_rate : ℝ
  combined_empty_time : ℝ

/-- Theorem stating the relationship between tank properties and its capacity. -/
theorem tank_capacity_calculation (t : Tank)
  (h1 : t.leak_empty_time = 6)
  (h2 : t.inlet_rate = 4.5)
  (h3 : t.combined_empty_time = 8) :
  t.capacity = 6480 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_calculation_l2398_239895


namespace NUMINAMATH_CALUDE_vector_projection_l2398_239865

/-- Given two vectors a and b in ℝ², and a vector c such that a + c = 0,
    prove that the projection of c onto b is -√65/5 -/
theorem vector_projection (a b c : ℝ × ℝ) :
  a = (2, 3) →
  b = (-4, 7) →
  a + c = (0, 0) →
  (c.1 * b.1 + c.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -Real.sqrt 65 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l2398_239865


namespace NUMINAMATH_CALUDE_jeffreys_farm_chickens_total_chickens_is_76_l2398_239818

/-- Calculates the total number of chickens on Jeffrey's farm -/
theorem jeffreys_farm_chickens (num_hens : ℕ) (hen_rooster_ratio : ℕ) (chicks_per_hen : ℕ) : ℕ :=
  let num_roosters := num_hens / hen_rooster_ratio
  let num_chicks := num_hens * chicks_per_hen
  num_hens + num_roosters + num_chicks

/-- Proves that the total number of chickens on Jeffrey's farm is 76 -/
theorem total_chickens_is_76 :
  jeffreys_farm_chickens 12 3 5 = 76 := by
  sorry

end NUMINAMATH_CALUDE_jeffreys_farm_chickens_total_chickens_is_76_l2398_239818


namespace NUMINAMATH_CALUDE_tangent_points_focus_slope_l2398_239836

/-- The slope of the line connecting the tangent points and the focus of a parabola -/
theorem tangent_points_focus_slope (x₀ y₀ : ℝ) : 
  x₀ = -1 → y₀ = 2 → 
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- Tangent points satisfy the parabola equation
    y₁^2 = 4*x₁ ∧ y₂^2 = 4*x₂ ∧
    -- Tangent lines pass through (x₀, y₀)
    (∃ k₁ k₂ : ℝ, y₁ - y₀ = k₁*(x₁ - x₀) ∧ y₂ - y₀ = k₂*(x₂ - x₀)) →
    -- Slope of the line connecting tangent points and focus
    (y₁ - 1/4) / (x₁ - 1/4) = 1 ∧ (y₂ - 1/4) / (x₂ - 1/4) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_points_focus_slope_l2398_239836


namespace NUMINAMATH_CALUDE_increasing_cubic_function_l2398_239800

/-- A function f(x) = x^3 - ax^2 - 3x is increasing on [1, +∞) if and only if a ≤ 0 -/
theorem increasing_cubic_function (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → (deriv (fun x => x^3 - a*x^2 - 3*x)) x ≥ 0) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_l2398_239800


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l2398_239821

/-- The ratio of the area of a rectangle to the area of a triangle formed with one side of the rectangle as base --/
theorem rectangle_triangle_area_ratio 
  (L W : ℝ) 
  (θ : ℝ) 
  (h_pos : L > 0 ∧ W > 0)
  (h_angle : 0 < θ ∧ θ < π / 2) :
  (L * W) / ((1/2) * L * W * Real.sin θ) = 2 / Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l2398_239821


namespace NUMINAMATH_CALUDE_sum_b_m_is_neg_eleven_fifths_l2398_239888

/-- Two lines intersecting at a point -/
structure IntersectingLines where
  m : ℚ
  b : ℚ
  x : ℚ
  y : ℚ
  h1 : y = m * x + 3
  h2 : y = 2 * x + b
  h3 : x = 5
  h4 : y = 7

/-- The sum of b and m for the intersecting lines -/
def sum_b_m (l : IntersectingLines) : ℚ := l.b + l.m

/-- Theorem stating that the sum of b and m is -11/5 -/
theorem sum_b_m_is_neg_eleven_fifths (l : IntersectingLines) : 
  sum_b_m l = -11/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_b_m_is_neg_eleven_fifths_l2398_239888


namespace NUMINAMATH_CALUDE_rearrange_3622_l2398_239853

def digits : List ℕ := [3, 6, 2, 2]

theorem rearrange_3622 : (List.permutations digits).length = 12 := by
  sorry

end NUMINAMATH_CALUDE_rearrange_3622_l2398_239853


namespace NUMINAMATH_CALUDE_equation_solution_l2398_239801

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (x - 20))) = 59 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2398_239801


namespace NUMINAMATH_CALUDE_rita_calculation_l2398_239831

theorem rita_calculation (a b c : ℝ) 
  (h1 : a - (2*b - 3*c) = 23) 
  (h2 : a - 2*b - 3*c = 5) : 
  a - 2*b = 14 := by
sorry

end NUMINAMATH_CALUDE_rita_calculation_l2398_239831


namespace NUMINAMATH_CALUDE_four_students_line_arrangement_l2398_239896

/-- The number of ways to arrange 4 students in a line with restrictions -/
def restricted_arrangements : ℕ := 12

/-- The total number of unrestricted arrangements of 4 students -/
def total_arrangements : ℕ := 24

/-- The number of arrangements where the fourth student is next to at least one other -/
def invalid_arrangements : ℕ := 12

theorem four_students_line_arrangement :
  restricted_arrangements = total_arrangements - invalid_arrangements :=
by sorry

end NUMINAMATH_CALUDE_four_students_line_arrangement_l2398_239896


namespace NUMINAMATH_CALUDE_smallest_positive_linear_combination_l2398_239875

theorem smallest_positive_linear_combination : 
  ∃ (k : ℕ), k > 0 ∧ (∃ (m n : ℤ), k = 1205 * m + 27090 * n) ∧ 
  (∀ (j : ℕ), j > 0 → (∃ (x y : ℤ), j = 1205 * x + 27090 * y) → j ≥ k) ∧
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_linear_combination_l2398_239875


namespace NUMINAMATH_CALUDE_min_red_vertices_l2398_239856

/-- Given a square partitioned into n^2 unit squares, each divided into two triangles,
    the minimum number of red vertices needed to ensure each triangle has a red vertex is ⌈n^2/2⌉ -/
theorem min_red_vertices (n : ℕ) (h : n > 0) :
  ∃ (red_vertices : Finset (ℕ × ℕ)),
    (∀ i j : ℕ, i < n → j < n →
      (∃ k l : ℕ, (k = i ∨ k = i + 1) ∧ (l = j ∨ l = j + 1) ∧ (k, l) ∈ red_vertices)) ∧
    red_vertices.card = ⌈(n^2 : ℝ) / 2⌉ ∧
    (∀ rv : Finset (ℕ × ℕ), 
      (∀ i j : ℕ, i < n → j < n →
        (∃ k l : ℕ, (k = i ∨ k = i + 1) ∧ (l = j ∨ l = j + 1) ∧ (k, l) ∈ rv)) →
      rv.card ≥ ⌈(n^2 : ℝ) / 2⌉) := by
  sorry


end NUMINAMATH_CALUDE_min_red_vertices_l2398_239856


namespace NUMINAMATH_CALUDE_original_cube_side_length_l2398_239805

/-- Given a cube of side length s that is painted and cut into smaller cubes of side 3,
    if there are exactly 12 smaller cubes with paint on 2 sides, then s = 6 -/
theorem original_cube_side_length (s : ℕ) : 
  s > 0 →  -- ensure the side length is positive
  (12 * (s / 3 - 1) = 12) →  -- condition for 12 smaller cubes with paint on 2 sides
  s = 6 := by
sorry

end NUMINAMATH_CALUDE_original_cube_side_length_l2398_239805


namespace NUMINAMATH_CALUDE_sixth_root_of_1061520150601_l2398_239841

theorem sixth_root_of_1061520150601 :
  let n : ℕ := 1061520150601
  ∃ (m : ℕ), m = 101 ∧ m^6 = n :=
by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_1061520150601_l2398_239841


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solutions_l2398_239877

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first_fourth_sum : a 1 + a 4 = 4
  second_third_product : a 2 * a 3 = 3
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_solutions (seq : ArithmeticSequence) :
  (seq.a 1 = -1 ∧ seq.d = 2 ∧ (∀ n, seq.a n = 2 * n - 3) ∧ (∀ n, S seq n = n^2 - 2*n)) ∨
  (seq.a 1 = 5 ∧ seq.d = -2 ∧ (∀ n, seq.a n = 7 - 2 * n) ∧ (∀ n, S seq n = 6*n - n^2)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solutions_l2398_239877


namespace NUMINAMATH_CALUDE_marble_probability_l2398_239897

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) :
  total = 100 →
  p_white = 1/4 →
  p_green = 1/5 →
  ∃ (p_red_blue : ℚ), p_red_blue = 11/20 ∧ 
    p_white + p_green + p_red_blue = 1 :=
sorry

end NUMINAMATH_CALUDE_marble_probability_l2398_239897


namespace NUMINAMATH_CALUDE_beatrice_tv_shopping_l2398_239893

theorem beatrice_tv_shopping (first_store : ℕ) (online_store : ℕ) (auction_site : ℕ) :
  first_store = 8 →
  online_store = 3 * first_store →
  first_store + online_store + auction_site = 42 →
  auction_site = 10 := by
sorry

end NUMINAMATH_CALUDE_beatrice_tv_shopping_l2398_239893


namespace NUMINAMATH_CALUDE_cube_root_equation_l2398_239899

theorem cube_root_equation (a b : ℝ) :
  let z : ℝ := (a + (a^2 + b^3)^(1/2))^(1/3) - ((a^2 + b^3)^(1/2) - a)^(1/3)
  z^3 + 3*b*z - 2*a = 0 := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_l2398_239899


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l2398_239857

/-- Given a right triangle with sides 9, 12, and 15, the diameter of its inscribed circle is 6. -/
theorem inscribed_circle_diameter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_sides : a = 9 ∧ b = 12 ∧ c = 15) : 
  2 * ((a + b - c) / 2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_diameter_l2398_239857
