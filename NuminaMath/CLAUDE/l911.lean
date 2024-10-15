import Mathlib

namespace NUMINAMATH_CALUDE_labourer_income_is_78_l911_91175

/-- Represents the financial situation of a labourer over a 10-month period. -/
structure LabourerFinances where
  monthly_income : ℝ
  initial_debt : ℝ
  first_period_months : ℕ := 6
  second_period_months : ℕ := 4
  first_period_monthly_expense : ℝ := 85
  second_period_monthly_expense : ℝ := 60
  final_savings : ℝ := 30

/-- The labourer's financial situation satisfies the given conditions. -/
def satisfies_conditions (f : LabourerFinances) : Prop :=
  f.first_period_months * f.monthly_income - f.initial_debt = 
    f.first_period_months * f.first_period_monthly_expense ∧
  f.second_period_months * f.monthly_income = 
    f.second_period_months * f.second_period_monthly_expense + f.initial_debt + f.final_savings

/-- The labourer's monthly income is 78 given the conditions. -/
theorem labourer_income_is_78 (f : LabourerFinances) 
  (h : satisfies_conditions f) : f.monthly_income = 78 := by
  sorry

end NUMINAMATH_CALUDE_labourer_income_is_78_l911_91175


namespace NUMINAMATH_CALUDE_lcm_gcd_equation_solutions_l911_91145

def solution_pairs : List (Nat × Nat) := [(3, 6), (4, 6), (4, 4), (6, 4), (6, 3)]

theorem lcm_gcd_equation_solutions :
  ∀ a b : Nat,
    a > 0 ∧ b > 0 →
    (Nat.lcm a b + Nat.gcd a b + a + b = a * b) ↔ (a, b) ∈ solution_pairs := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_equation_solutions_l911_91145


namespace NUMINAMATH_CALUDE_jorge_corn_yield_l911_91134

/-- Represents the yield calculation for Jorge's corn plantation --/
def jorge_yield (total_acres : ℝ) (clay_rich_fraction : ℝ) (total_yield : ℝ) (other_soil_yield : ℝ) : Prop :=
  let clay_rich_acres := clay_rich_fraction * total_acres
  let other_soil_acres := (1 - clay_rich_fraction) * total_acres
  let clay_rich_yield := (other_soil_yield / 2) * clay_rich_acres
  let other_soil_total_yield := other_soil_yield * other_soil_acres
  clay_rich_yield + other_soil_total_yield = total_yield

theorem jorge_corn_yield :
  jorge_yield 60 (1/3) 20000 400 := by
  sorry

end NUMINAMATH_CALUDE_jorge_corn_yield_l911_91134


namespace NUMINAMATH_CALUDE_cookie_making_time_l911_91103

/-- Proves that the time to make dough and cool cookies is equal to the total time minus the sum of baking time and icing hardening times. -/
theorem cookie_making_time (total_time baking_time white_icing_time chocolate_icing_time : ℕ)
  (h1 : total_time = 120)
  (h2 : baking_time = 15)
  (h3 : white_icing_time = 30)
  (h4 : chocolate_icing_time = 30) :
  total_time - (baking_time + white_icing_time + chocolate_icing_time) = 45 :=
by sorry

end NUMINAMATH_CALUDE_cookie_making_time_l911_91103


namespace NUMINAMATH_CALUDE_jason_grass_cutting_time_l911_91109

/-- The time Jason spends cutting grass over a weekend -/
def time_cutting_grass (time_per_lawn : ℕ) (lawns_per_day : ℕ) (days : ℕ) : ℕ :=
  time_per_lawn * lawns_per_day * days

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

theorem jason_grass_cutting_time :
  let time_per_lawn := 30
  let lawns_per_day := 8
  let days := 2
  minutes_to_hours (time_cutting_grass time_per_lawn lawns_per_day days) = 8 := by
  sorry

end NUMINAMATH_CALUDE_jason_grass_cutting_time_l911_91109


namespace NUMINAMATH_CALUDE_smallest_fraction_proof_l911_91130

def is_natural_number (q : ℚ) : Prop := ∃ (n : ℕ), q = n

theorem smallest_fraction_proof (f : ℚ) : 
  (f ≥ 42/5) →
  (is_natural_number (f / (21/25))) →
  (is_natural_number (f / (14/15))) →
  (∀ g : ℚ, g < f → ¬(is_natural_number (g / (21/25)) ∧ is_natural_number (g / (14/15)))) →
  f = 42/5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_proof_l911_91130


namespace NUMINAMATH_CALUDE_find_other_number_l911_91199

-- Define the given conditions
def n : ℕ := 48
def lcm_nm : ℕ := 56
def gcf_nm : ℕ := 12

-- Define the theorem
theorem find_other_number (m : ℕ) : 
  (Nat.lcm n m = lcm_nm) → 
  (Nat.gcd n m = gcf_nm) → 
  m = 14 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l911_91199


namespace NUMINAMATH_CALUDE_pet_shop_dogs_l911_91184

/-- Given a pet shop with dogs, cats, and bunnies, where the ratio of dogs to cats to bunnies
    is 3:7:12 and the total number of dogs and bunnies is 375, prove that there are 75 dogs. -/
theorem pet_shop_dogs (dogs cats bunnies : ℕ) : 
  dogs + cats + bunnies > 0 →
  dogs * 7 = cats * 3 →
  dogs * 12 = bunnies * 3 →
  dogs + bunnies = 375 →
  dogs = 75 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_dogs_l911_91184


namespace NUMINAMATH_CALUDE_x_range_for_sqrt_equality_l911_91162

theorem x_range_for_sqrt_equality (x : ℝ) : 
  (Real.sqrt (x / (1 - x)) = Real.sqrt x / Real.sqrt (1 - x)) → 
  (0 ≤ x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_sqrt_equality_l911_91162


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l911_91106

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 < x ∧ x < 1) ↔ ((x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l911_91106


namespace NUMINAMATH_CALUDE_max_k_inequality_l911_91172

theorem max_k_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (k : ℝ), k > 0 ∧ 
  (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    Real.sqrt (x^2 + k*y^2) + Real.sqrt (y^2 + k*x^2) ≥ x + y + (k-1) * Real.sqrt (x*y)) ∧
  (∀ (k' : ℝ), k' > k → 
    ∃ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
      Real.sqrt (x^2 + k'*y^2) + Real.sqrt (y^2 + k'*x^2) < x + y + (k'-1) * Real.sqrt (x*y)) ∧
  k = 3 :=
sorry

end NUMINAMATH_CALUDE_max_k_inequality_l911_91172


namespace NUMINAMATH_CALUDE_loan_amount_proof_l911_91186

/-- The interest rate at which A lends to B (as a decimal) -/
def rate_A_to_B : ℚ := 15 / 100

/-- The interest rate at which B lends to C (as a decimal) -/
def rate_B_to_C : ℚ := 185 / 1000

/-- The number of years for which the loan is given -/
def years : ℕ := 3

/-- The gain of B in the given period -/
def gain_B : ℕ := 294

/-- The amount lent by A to B -/
def amount_lent : ℕ := 2800

theorem loan_amount_proof :
  ∃ (P : ℕ), 
    (P : ℚ) * rate_B_to_C * years - (P : ℚ) * rate_A_to_B * years = gain_B ∧
    P = amount_lent :=
by sorry

end NUMINAMATH_CALUDE_loan_amount_proof_l911_91186


namespace NUMINAMATH_CALUDE_b3f_hex_to_decimal_l911_91197

/-- Converts a single hexadecimal digit to its decimal value -/
def hexToDecimal (c : Char) : ℕ :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => c.toString.toNat!

/-- Converts a hexadecimal string to its decimal value -/
def hexStringToDecimal (s : String) : ℕ :=
  s.foldr (fun c acc => 16 * acc + hexToDecimal c) 0

theorem b3f_hex_to_decimal :
  hexStringToDecimal "B3F" = 2879 := by
  sorry

end NUMINAMATH_CALUDE_b3f_hex_to_decimal_l911_91197


namespace NUMINAMATH_CALUDE_noah_holidays_l911_91154

/-- The number of holidays Noah takes per month -/
def holidays_per_month : ℕ := 3

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total number of holidays Noah takes in a year -/
def total_holidays : ℕ := holidays_per_month * months_in_year

theorem noah_holidays : total_holidays = 36 := by
  sorry

end NUMINAMATH_CALUDE_noah_holidays_l911_91154


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_l911_91101

open Set

theorem fraction_inequality_solution (x : ℝ) :
  (x - 5) / ((x - 3)^2) < 0 ↔ x ∈ Iio 3 ∪ Ioo 3 5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_l911_91101


namespace NUMINAMATH_CALUDE_zoo_animals_l911_91169

theorem zoo_animals (b r m : ℕ) : 
  b + r + m = 300 →
  2 * b + 3 * r + 4 * m = 798 →
  r = 102 :=
by sorry

end NUMINAMATH_CALUDE_zoo_animals_l911_91169


namespace NUMINAMATH_CALUDE_dihedral_angle_line_relationship_l911_91146

/-- A dihedral angle with edge l and planes α and β -/
structure DihedralAngle where
  l : Line
  α : Plane
  β : Plane

/-- A right dihedral angle -/
def is_right_dihedral (d : DihedralAngle) : Prop := sorry

/-- A line a in plane α -/
def line_in_plane_α (d : DihedralAngle) (a : Line) : Prop := sorry

/-- A line b in plane β -/
def line_in_plane_β (d : DihedralAngle) (b : Line) : Prop := sorry

/-- Line not perpendicular to edge l -/
def not_perp_to_edge (d : DihedralAngle) (m : Line) : Prop := sorry

/-- Two lines are parallel -/
def are_parallel (m n : Line) : Prop := sorry

/-- Two lines are perpendicular -/
def are_perpendicular (m n : Line) : Prop := sorry

theorem dihedral_angle_line_relationship (d : DihedralAngle) (a b : Line) 
  (h_right : is_right_dihedral d)
  (h_a_in_α : line_in_plane_α d a)
  (h_b_in_β : line_in_plane_β d b)
  (h_a_not_perp : not_perp_to_edge d a)
  (h_b_not_perp : not_perp_to_edge d b) :
  (∃ (a' b' : Line), line_in_plane_α d a' ∧ line_in_plane_β d b' ∧ 
    not_perp_to_edge d a' ∧ not_perp_to_edge d b' ∧ are_parallel a' b') ∧ 
  (∀ (a' b' : Line), line_in_plane_α d a' → line_in_plane_β d b' → 
    not_perp_to_edge d a' → not_perp_to_edge d b' → ¬ are_perpendicular a' b') :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_line_relationship_l911_91146


namespace NUMINAMATH_CALUDE_coupon_discount_percentage_l911_91113

theorem coupon_discount_percentage (original_price increased_price final_price : ℝ) 
  (h1 : original_price = 200)
  (h2 : increased_price = original_price * 1.3)
  (h3 : final_price = 182) : 
  (increased_price - final_price) / increased_price = 0.3 := by
sorry

end NUMINAMATH_CALUDE_coupon_discount_percentage_l911_91113


namespace NUMINAMATH_CALUDE_final_K_value_l911_91165

/-- Represents the state of the program at each iteration -/
structure ProgramState :=
  (S : ℕ)
  (K : ℕ)

/-- Defines a single iteration of the loop -/
def iterate (state : ProgramState) : ProgramState :=
  { S := state.S^2 + 1,
    K := state.K + 1 }

/-- Defines the condition for continuing the loop -/
def loopCondition (state : ProgramState) : Prop :=
  state.S < 100

/-- Theorem: The final value of K is 4 -/
theorem final_K_value :
  ∃ (n : ℕ), ∃ (finalState : ProgramState),
    (finalState.K = 4) ∧
    (¬loopCondition finalState) ∧
    (finalState = (iterate^[n] ⟨1, 1⟩)) :=
  sorry

end NUMINAMATH_CALUDE_final_K_value_l911_91165


namespace NUMINAMATH_CALUDE_hyperbola_equation_l911_91148

/-- Definition of a hyperbola with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  focus : ℝ × ℝ
  intersection_line : ℝ → ℝ
  midpoint_x : ℝ

/-- Theorem stating the equation of the hyperbola with given properties -/
theorem hyperbola_equation (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_focus : h.focus = (Real.sqrt 7, 0))
  (h_line : h.intersection_line = fun x ↦ x - 1)
  (h_midpoint : h.midpoint_x = -2/3) :
  ∃ (x y : ℝ), x^2/2 - y^2/5 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l911_91148


namespace NUMINAMATH_CALUDE_lesser_number_problem_l911_91110

theorem lesser_number_problem (x y : ℝ) 
  (sum_eq : x + y = 50) 
  (diff_eq : x - y = 7) : 
  y = 21.5 := by
sorry

end NUMINAMATH_CALUDE_lesser_number_problem_l911_91110


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l911_91117

theorem sqrt_sum_equality : 
  Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) + 1 = 8 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l911_91117


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l911_91182

theorem complex_fraction_simplification :
  (Complex.I + 1) / (1 - Complex.I) = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l911_91182


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l911_91194

theorem simplify_and_evaluate : 
  let a : ℚ := -2
  let b : ℚ := 1/5
  2 * a * b^2 - (6 * a^3 * b + 2 * (a * b^2 - 1/2 * a^3 * b)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l911_91194


namespace NUMINAMATH_CALUDE_correct_average_proof_l911_91135

/-- The number of students in the class -/
def num_students : ℕ := 60

/-- The incorrect average marks -/
def incorrect_average : ℚ := 82

/-- Reema's correct mark -/
def reema_correct : ℕ := 78

/-- Reema's incorrect mark -/
def reema_incorrect : ℕ := 68

/-- Mark's correct mark -/
def mark_correct : ℕ := 95

/-- Mark's incorrect mark -/
def mark_incorrect : ℕ := 91

/-- Jenny's correct mark -/
def jenny_correct : ℕ := 84

/-- Jenny's incorrect mark -/
def jenny_incorrect : ℕ := 74

/-- The correct average marks -/
def correct_average : ℚ := 82.40

theorem correct_average_proof :
  let incorrect_total := (incorrect_average * num_students : ℚ)
  let mark_difference := (reema_correct - reema_incorrect) + (mark_correct - mark_incorrect) + (jenny_correct - jenny_incorrect)
  let correct_total := incorrect_total + mark_difference
  (correct_total / num_students : ℚ) = correct_average := by sorry

end NUMINAMATH_CALUDE_correct_average_proof_l911_91135


namespace NUMINAMATH_CALUDE_sum_of_alan_and_bob_ages_l911_91140

-- Define the set of possible ages
def Ages : Set ℕ := {3, 8, 12, 14}

-- Define the cousins' ages as natural numbers
variables (alan_age bob_age carl_age dan_age : ℕ)

-- Define the conditions
def conditions (alan_age bob_age carl_age dan_age : ℕ) : Prop :=
  alan_age ∈ Ages ∧ bob_age ∈ Ages ∧ carl_age ∈ Ages ∧ dan_age ∈ Ages ∧
  alan_age ≠ bob_age ∧ alan_age ≠ carl_age ∧ alan_age ≠ dan_age ∧
  bob_age ≠ carl_age ∧ bob_age ≠ dan_age ∧ carl_age ≠ dan_age ∧
  alan_age < carl_age ∧
  (alan_age + dan_age) % 5 = 0 ∧
  (carl_age + dan_age) % 5 = 0

-- Theorem statement
theorem sum_of_alan_and_bob_ages 
  (h : conditions alan_age bob_age carl_age dan_age) :
  alan_age + bob_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_alan_and_bob_ages_l911_91140


namespace NUMINAMATH_CALUDE_quadratic_factorization_l911_91179

theorem quadratic_factorization (a b c : ℤ) :
  (∀ x : ℚ, x^2 + 16*x + 63 = (x + a) * (x + b)) →
  (∀ x : ℚ, x^2 + 6*x - 72 = (x + b) * (x - c)) →
  a + b + c = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l911_91179


namespace NUMINAMATH_CALUDE_uncle_zhang_revenue_l911_91191

/-- Uncle Zhang's newspaper selling problem -/
theorem uncle_zhang_revenue
  (a b : ℕ)  -- a and b are natural numbers representing the number of newspapers
  (purchase_price sell_price return_price : ℚ)  -- prices are rational numbers
  (h1 : purchase_price = 0.4)  -- purchase price is 0.4 yuan
  (h2 : sell_price = 0.5)  -- selling price is 0.5 yuan
  (h3 : return_price = 0.2)  -- return price is 0.2 yuan
  (h4 : b ≤ a)  -- number of sold newspapers cannot exceed purchased newspapers
  : (sell_price * b + return_price * (a - b) - purchase_price * a : ℚ) = 0.3 * b - 0.2 * a :=
by sorry

end NUMINAMATH_CALUDE_uncle_zhang_revenue_l911_91191


namespace NUMINAMATH_CALUDE_monopolist_optimal_quantity_l911_91163

/-- Represents the demand function for a monopolist's product -/
def demand (P : ℝ) : ℝ := 10 - P

/-- Represents the revenue function for the monopolist -/
def revenue (Q : ℝ) : ℝ := Q * (10 - Q)

/-- Represents the profit function for the monopolist -/
def profit (Q : ℝ) : ℝ := revenue Q

/-- The maximum quantity of goods the monopolist can sell -/
def max_quantity : ℝ := 10

/-- Theorem: The monopolist maximizes profit by selling 5 units -/
theorem monopolist_optimal_quantity :
  ∃ (Q : ℝ), Q = 5 ∧ 
  Q ≤ max_quantity ∧
  ∀ (Q' : ℝ), Q' ≤ max_quantity → profit Q' ≤ profit Q :=
sorry

end NUMINAMATH_CALUDE_monopolist_optimal_quantity_l911_91163


namespace NUMINAMATH_CALUDE_solution_system_equations_l911_91168

theorem solution_system_equations (A : ℤ) (hA : A ≠ 0) :
  ∀ x y z : ℤ,
    x + y^2 + z^3 = A ∧
    (1 : ℚ) / x + (1 : ℚ) / y^2 + (1 : ℚ) / z^3 = (1 : ℚ) / A ∧
    x * y^2 * z^3 = A^2 →
    ∃ k : ℤ, A = -k^12 ∧
      ((x = -k^12 ∧ (y = k^3 ∨ y = -k^3) ∧ z = -k^2) ∨
       (x = -k^3 ∧ (y = k^3 ∨ y = -k^3) ∧ z = -k^4)) :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l911_91168


namespace NUMINAMATH_CALUDE_correct_sums_l911_91107

theorem correct_sums (total : ℕ) (wrong_ratio : ℕ) (correct : ℕ) : 
  total = 36 → 
  wrong_ratio = 2 → 
  total = correct + wrong_ratio * correct → 
  correct = 12 := by sorry

end NUMINAMATH_CALUDE_correct_sums_l911_91107


namespace NUMINAMATH_CALUDE_inequality_system_solution_l911_91159

theorem inequality_system_solution (x : ℝ) : 
  (abs (x - 1) > 1 ∧ 1 / (4 - x) ≤ 1) ↔ (x < 0 ∨ (2 < x ∧ x ≤ 3) ∨ x > 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l911_91159


namespace NUMINAMATH_CALUDE_trapezoid_solutions_l911_91144

def is_trapezoid_solution (b₁ b₂ : ℕ) : Prop :=
  b₁ % 8 = 0 ∧ b₂ % 8 = 0 ∧ (b₁ + b₂) * 50 / 2 = 1400 ∧ b₁ > 0 ∧ b₂ > 0

theorem trapezoid_solutions :
  ∃! (solutions : List (ℕ × ℕ)), solutions.length = 3 ∧
    ∀ (b₁ b₂ : ℕ), (b₁, b₂) ∈ solutions ↔ is_trapezoid_solution b₁ b₂ :=
sorry

end NUMINAMATH_CALUDE_trapezoid_solutions_l911_91144


namespace NUMINAMATH_CALUDE_range_of_p_l911_91149

def h (x : ℝ) : ℝ := 4 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_of_p :
  ∀ y ∈ Set.range p, -1 ≤ y ∧ y ≤ 1023 ∧
  ∀ z, -1 ≤ z ∧ z ≤ 1023 → ∃ x, -1 ≤ x ∧ x ≤ 3 ∧ p x = z :=
sorry

end NUMINAMATH_CALUDE_range_of_p_l911_91149


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l911_91118

theorem arithmetic_calculation : 90 + 5 * 12 / (180 / 3) = 91 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l911_91118


namespace NUMINAMATH_CALUDE_cookie_difference_l911_91116

theorem cookie_difference (paul_cookies : ℕ) (total_cookies : ℕ) (paula_cookies : ℕ) : 
  paul_cookies = 45 → 
  total_cookies = 87 → 
  paula_cookies < paul_cookies →
  paul_cookies + paula_cookies = total_cookies →
  paul_cookies - paula_cookies = 3 := by
sorry

end NUMINAMATH_CALUDE_cookie_difference_l911_91116


namespace NUMINAMATH_CALUDE_domain_equals_range_l911_91193

-- Define the function f(x) = |x-2| - 2
def f (x : ℝ) : ℝ := |x - 2| - 2

-- Define the domain set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the range set N
def N : Set ℝ := f '' M

-- Theorem stating that M equals N
theorem domain_equals_range : M = N := by sorry

end NUMINAMATH_CALUDE_domain_equals_range_l911_91193


namespace NUMINAMATH_CALUDE_function_inequality_l911_91100

open Set

-- Define the interval [a, b]
variable (a b : ℝ) (hab : a < b)

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State that f and g are differentiable on [a, b]
variable (hf : DifferentiableOn ℝ f (Icc a b))
variable (hg : DifferentiableOn ℝ g (Icc a b))

-- State that f'(x) < g'(x) for all x in [a, b]
variable (h_deriv : ∀ x ∈ Icc a b, deriv f x < deriv g x)

-- State the theorem
theorem function_inequality (x : ℝ) (hx : a < x ∧ x < b) :
  f x + g a < g x + f a :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l911_91100


namespace NUMINAMATH_CALUDE_intersection_minimum_distance_l911_91198

/-- Given a line y = b intersecting f(x) = 2x + 3 and g(x) = ax + ln x at points A and B respectively,
    if the minimum value of |AB| is 2, then a + b = 2 -/
theorem intersection_minimum_distance (a b : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    2 * x₁ + 3 = b ∧ 
    a * x₂ + Real.log x₂ = b ∧
    (∀ (y₁ y₂ : ℝ), 2 * y₁ + 3 = b → a * y₂ + Real.log y₂ = b → |y₂ - y₁| ≥ 2) ∧
    |x₂ - x₁| = 2) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_minimum_distance_l911_91198


namespace NUMINAMATH_CALUDE_twelve_chairs_adjacent_subsets_l911_91102

/-- The number of subsets containing at least three adjacent chairs 
    when n chairs are arranged in a circle. -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs arranged in a circle, 
    the number of subsets containing at least three adjacent chairs is 2040. -/
theorem twelve_chairs_adjacent_subsets : 
  subsets_with_adjacent_chairs 12 = 2040 := by sorry

end NUMINAMATH_CALUDE_twelve_chairs_adjacent_subsets_l911_91102


namespace NUMINAMATH_CALUDE_f_surjective_and_unique_l911_91150

def f (x y : ℕ) : ℕ := (x + y - 1) * (x + y - 2) / 2 + y

theorem f_surjective_and_unique :
  ∀ n : ℕ, ∃! (x y : ℕ), f x y = n :=
by sorry

end NUMINAMATH_CALUDE_f_surjective_and_unique_l911_91150


namespace NUMINAMATH_CALUDE_runner_problem_l911_91155

/-- Proves that for a 40-mile run where the speed is halved halfway through,
    and the second half takes 12 hours longer than the first half,
    the time to complete the second half is 24 hours. -/
theorem runner_problem (v : ℝ) (h1 : v > 0) : 
  (40 / v = 20 / v + 12) → (40 / (v / 2) = 24) :=
by sorry

end NUMINAMATH_CALUDE_runner_problem_l911_91155


namespace NUMINAMATH_CALUDE_cinnamon_swirls_distribution_l911_91122

theorem cinnamon_swirls_distribution (total_pieces : ℕ) (num_people : ℕ) (pieces_per_person : ℕ) : 
  total_pieces = 12 → num_people = 3 → total_pieces = num_people * pieces_per_person → pieces_per_person = 4 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_swirls_distribution_l911_91122


namespace NUMINAMATH_CALUDE_fifteenSidedFigureArea_l911_91119

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon defined by a list of vertices -/
def Polygon := List Point

/-- The 15-sided figure described in the problem -/
def fifteenSidedFigure : Polygon := [
  ⟨1, 2⟩, ⟨2, 2⟩, ⟨2, 3⟩, ⟨3, 4⟩, ⟨4, 4⟩, ⟨5, 5⟩, ⟨6, 5⟩, ⟨7, 4⟩,
  ⟨6, 3⟩, ⟨6, 2⟩, ⟨5, 1⟩, ⟨4, 1⟩, ⟨3, 1⟩, ⟨2, 1⟩, ⟨1, 2⟩
]

/-- Calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℝ :=
  sorry

/-- Theorem stating that the area of the 15-sided figure is 15 cm² -/
theorem fifteenSidedFigureArea :
  calculateArea fifteenSidedFigure = 15 := by sorry

end NUMINAMATH_CALUDE_fifteenSidedFigureArea_l911_91119


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l911_91131

/-- Given a quadratic equation (k-1)x^2 + 6x + k^2 - k = 0 with a root of 0, prove that k = 0 -/
theorem quadratic_root_zero (k : ℝ) : 
  (∃ x, (k - 1) * x^2 + 6 * x + k^2 - k = 0) ∧ 
  ((k - 1) * 0^2 + 6 * 0 + k^2 - k = 0) → 
  k = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l911_91131


namespace NUMINAMATH_CALUDE_linear_function_properties_l911_91174

-- Define the linear function
def f (x : ℝ) : ℝ := -2 * x + 1

-- Define the properties to be proven
theorem linear_function_properties :
  (∀ x y : ℝ, f x - f y = -2 * (x - y)) ∧  -- Slope is -2
  (f 0 = 1) ∧                              -- y-intercept is (0, 1)
  (∃ x y z : ℝ, f x > 0 ∧ x > 0 ∧          -- Passes through first quadrant
               f y < 0 ∧ y > 0 ∧           -- Passes through second quadrant
               f z < 0 ∧ z < 0) ∧          -- Passes through fourth quadrant
  (∀ x y : ℝ, x < y → f x > f y)           -- Slope is negative
  := by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l911_91174


namespace NUMINAMATH_CALUDE_function_has_infinitely_many_extreme_points_l911_91180

/-- The function f(x) = x^2 - 2x cos(x) has infinitely many extreme points -/
theorem function_has_infinitely_many_extreme_points :
  ∃ (f : ℝ → ℝ), (∀ x, f x = x^2 - 2*x*(Real.cos x)) ∧
  (∀ n : ℕ, ∃ (S : Finset ℝ), S.card ≥ n ∧ 
    (∀ x ∈ S, ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x)) :=
by sorry


end NUMINAMATH_CALUDE_function_has_infinitely_many_extreme_points_l911_91180


namespace NUMINAMATH_CALUDE_meeting_time_prove_meeting_time_l911_91141

/-- The time it takes for Petya and Vasya to meet under the given conditions -/
theorem meeting_time : ℝ → ℝ → ℝ → Prop :=
  fun (x : ℝ) (v_g : ℝ) (t : ℝ) =>
    x > 0 ∧ v_g > 0 ∧  -- Positive distance and speed
    x = 3 * v_g ∧  -- Petya reaches the bridge in 1 hour
    t = 1 + (2 * x - 2 * v_g) / (2 * v_g) ∧  -- Total time calculation
    t = 2  -- The meeting time is 2 hours

/-- Proof of the meeting time theorem -/
theorem prove_meeting_time : ∃ (x v_g : ℝ), meeting_time x v_g 2 := by
  sorry


end NUMINAMATH_CALUDE_meeting_time_prove_meeting_time_l911_91141


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l911_91143

theorem arithmetic_calculations :
  ((1 : ℝ) - 12 + (-6) - (-28) = 10) ∧
  ((2 : ℝ) - 3^2 + (7/8 - 1) * (-2)^2 = -9.5) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l911_91143


namespace NUMINAMATH_CALUDE_gcd_2728_1575_l911_91188

theorem gcd_2728_1575 : Nat.gcd 2728 1575 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2728_1575_l911_91188


namespace NUMINAMATH_CALUDE_one_diagonal_polygon_has_four_edges_edges_equal_vertices_one_diagonal_polygon_four_edges_l911_91181

/-- A polygon is a shape with straight sides and angles. -/
structure Polygon where
  vertices : ℕ
  vertices_positive : vertices > 0

/-- A diagonal in a polygon is a line segment that connects two non-adjacent vertices. -/
def diagonals_from_vertex (p : Polygon) : ℕ := p.vertices - 3

/-- A polygon where only one diagonal can be drawn from a single vertex has 4 edges. -/
theorem one_diagonal_polygon_has_four_edges (p : Polygon) 
  (h : diagonals_from_vertex p = 1) : p.vertices = 4 := by
  sorry

/-- The number of edges in a polygon is equal to its number of vertices. -/
theorem edges_equal_vertices (p : Polygon) : 
  (number_of_edges : ℕ) → number_of_edges = p.vertices := by
  sorry

/-- A polygon where only one diagonal can be drawn from a single vertex has 4 edges. -/
theorem one_diagonal_polygon_four_edges (p : Polygon) 
  (h : diagonals_from_vertex p = 1) : (number_of_edges : ℕ) → number_of_edges = 4 := by
  sorry

end NUMINAMATH_CALUDE_one_diagonal_polygon_has_four_edges_edges_equal_vertices_one_diagonal_polygon_four_edges_l911_91181


namespace NUMINAMATH_CALUDE_function_value_proof_l911_91160

theorem function_value_proof (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f ((1/2) * x - 1) = 2 * x - 5) →
  f a = 6 →
  a = 7/4 := by
sorry

end NUMINAMATH_CALUDE_function_value_proof_l911_91160


namespace NUMINAMATH_CALUDE_finite_good_numbers_not_divisible_by_l911_91132

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- n is a good number if τ(m) < τ(n) for all m < n -/
def is_good (n : ℕ+) : Prop :=
  ∀ m : ℕ+, m < n → tau m < tau n

/-- The set of good numbers not divisible by k is finite -/
theorem finite_good_numbers_not_divisible_by (k : ℕ+) :
  {n : ℕ+ | is_good n ∧ ¬k ∣ n}.Finite := by sorry

end NUMINAMATH_CALUDE_finite_good_numbers_not_divisible_by_l911_91132


namespace NUMINAMATH_CALUDE_inequality_proof_l911_91138

theorem inequality_proof (x a : ℝ) (h1 : x < a) (h2 : a < -1) (h3 : x < 0) (h4 : a < 0) :
  x^2 > a*x ∧ a*x > a^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l911_91138


namespace NUMINAMATH_CALUDE_no_triples_satisfying_lcm_conditions_l911_91183

theorem no_triples_satisfying_lcm_conditions :
  ¬∃ (x y z : ℕ+), 
    (Nat.lcm x.val y.val = 48) ∧ 
    (Nat.lcm x.val z.val = 900) ∧ 
    (Nat.lcm y.val z.val = 180) :=
by sorry

end NUMINAMATH_CALUDE_no_triples_satisfying_lcm_conditions_l911_91183


namespace NUMINAMATH_CALUDE_cube_painting_cost_l911_91177

/-- The cost of painting a cube's surface area given its volume and paint cost per area unit -/
theorem cube_painting_cost (volume : ℝ) (cost_per_area : ℝ) : 
  volume = 9261 → 
  cost_per_area = 13 / 100 →
  6 * (volume ^ (1/3))^2 * cost_per_area = 344.98 := by
sorry

end NUMINAMATH_CALUDE_cube_painting_cost_l911_91177


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l911_91136

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l911_91136


namespace NUMINAMATH_CALUDE_infinitely_many_square_sum_averages_l911_91121

theorem infinitely_many_square_sum_averages :
  ∀ k : ℕ, ∃ n > k, ∃ m : ℕ, ((n + 1) * (2 * n + 1)) / 6 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_square_sum_averages_l911_91121


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l911_91164

def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_in_second_quadrant :
  let z : ℂ := -1 + 3*Complex.I
  second_quadrant z :=
by
  sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l911_91164


namespace NUMINAMATH_CALUDE_pyramid_angles_theorem_l911_91105

/-- Represents the angles formed by the lateral faces of a pyramid with its square base -/
structure PyramidAngles where
  α : Real
  β : Real
  γ : Real
  δ : Real

/-- Theorem: Given a pyramid with a square base, if the angles formed by the lateral faces 
    with the base are in the ratio 1:2:4:2, then these angles are π/6, π/3, 2π/3, and π/3. -/
theorem pyramid_angles_theorem (angles : PyramidAngles) : 
  (angles.α : Real) / (angles.β : Real) = 1 / 2 ∧
  (angles.α : Real) / (angles.γ : Real) = 1 / 4 ∧
  (angles.α : Real) / (angles.δ : Real) = 1 / 2 ∧
  angles.α + angles.β + angles.γ + angles.δ = 2 * Real.pi →
  angles.α = Real.pi / 6 ∧
  angles.β = Real.pi / 3 ∧
  angles.γ = 2 * Real.pi / 3 ∧
  angles.δ = Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_angles_theorem_l911_91105


namespace NUMINAMATH_CALUDE_prob_same_color_proof_l911_91156

def total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2

def prob_same_color : ℚ := 1 / 3

theorem prob_same_color_proof :
  let prob_red := red_balls / total_balls * (red_balls - 1) / (total_balls - 1)
  let prob_white := white_balls / total_balls * (white_balls - 1) / (total_balls - 1)
  prob_red + prob_white = prob_same_color :=
sorry

end NUMINAMATH_CALUDE_prob_same_color_proof_l911_91156


namespace NUMINAMATH_CALUDE_intersection_of_lines_l911_91185

theorem intersection_of_lines :
  ∃! (x y : ℚ), (3 * y = -2 * x + 6) ∧ (-2 * y = 4 * x - 3) ∧ (x = 3/8) ∧ (y = 7/4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l911_91185


namespace NUMINAMATH_CALUDE_park_area_is_1500000_l911_91126

/-- Represents the scale of the map in miles per inch -/
def scale : ℝ := 250

/-- Represents the length of the park on the map in inches -/
def map_length : ℝ := 6

/-- Represents the width of the park on the map in inches -/
def map_width : ℝ := 4

/-- Calculates the actual area of the park in square miles -/
def park_area : ℝ := (map_length * scale) * (map_width * scale)

/-- Theorem stating that the actual area of the park is 1500000 square miles -/
theorem park_area_is_1500000 : park_area = 1500000 := by
  sorry

end NUMINAMATH_CALUDE_park_area_is_1500000_l911_91126


namespace NUMINAMATH_CALUDE_cubic_root_product_l911_91189

theorem cubic_root_product (a b c : ℝ) : 
  a^3 - 15*a^2 + 22*a - 8 = 0 →
  b^3 - 15*b^2 + 22*b - 8 = 0 →
  c^3 - 15*c^2 + 22*c - 8 = 0 →
  (2+a)*(2+b)*(2+c) = 120 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_product_l911_91189


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l911_91133

def M : ℕ := 24 * 36 * 49 * 125

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 62 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l911_91133


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l911_91123

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (3 + Real.sqrt (4 * y - 5)) = Real.sqrt 8 → y = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l911_91123


namespace NUMINAMATH_CALUDE_moving_point_on_line_segment_l911_91170

/-- Two fixed points in a plane -/
structure FixedPoints where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  distance : dist F₁ F₂ = 16

/-- A moving point M satisfying the condition |MF₁| + |MF₂| = 16 -/
def MovingPoint (fp : FixedPoints) (M : ℝ × ℝ) : Prop :=
  dist M fp.F₁ + dist M fp.F₂ = 16

/-- The theorem stating that any moving point M lies on the line segment F₁F₂ -/
theorem moving_point_on_line_segment (fp : FixedPoints) (M : ℝ × ℝ) 
    (h : MovingPoint fp M) : 
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • fp.F₁ + t • fp.F₂ :=
  sorry

end NUMINAMATH_CALUDE_moving_point_on_line_segment_l911_91170


namespace NUMINAMATH_CALUDE_product_expansion_l911_91114

theorem product_expansion (x : ℝ) (h : x ≠ 0) :
  (3 / 7) * ((7 / x^3) - 14 * x^4) = 3 / x^3 - 6 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l911_91114


namespace NUMINAMATH_CALUDE_quadrilateral_cosine_sum_l911_91111

theorem quadrilateral_cosine_sum (α β γ δ : Real) :
  (α + β + γ + δ = 2 * Real.pi) →
  (Real.cos α + Real.cos β + Real.cos γ + Real.cos δ = 0) →
  (α + β = Real.pi) ∨ (α + γ = Real.pi) ∨ (α + δ = Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_cosine_sum_l911_91111


namespace NUMINAMATH_CALUDE_night_day_crew_ratio_l911_91171

theorem night_day_crew_ratio (D N : ℝ) (h1 : D > 0) (h2 : N > 0) : 
  (D / (D + 3/4 * N) = 0.64) → (N / D = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_night_day_crew_ratio_l911_91171


namespace NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l911_91190

theorem six_digit_multiple_of_nine :
  ∃! d : Nat, d < 10 ∧ (456780 + d) % 9 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l911_91190


namespace NUMINAMATH_CALUDE_banana_permutations_count_l911_91176

/-- The number of unique permutations of a multiset with 6 elements,
    where one element appears 3 times, another appears 2 times,
    and the third appears once. -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)

/-- Theorem stating that the number of unique permutations of "BANANA" is 60. -/
theorem banana_permutations_count : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_count_l911_91176


namespace NUMINAMATH_CALUDE_circle_focus_at_center_l911_91195

/-- An ellipse with equal major and minor axes is a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The focus of a circle is at its center -/
def Circle.focus (c : Circle) : ℝ × ℝ := c.center

theorem circle_focus_at_center (h_center : ℝ × ℝ) (h_radius : ℝ) :
  let c : Circle := { center := h_center, radius := h_radius }
  c.focus = c.center := by sorry

end NUMINAMATH_CALUDE_circle_focus_at_center_l911_91195


namespace NUMINAMATH_CALUDE_y_sixth_power_root_l911_91129

theorem y_sixth_power_root (y : ℝ) (hy : y > 0) (h : Real.sin (Real.arctan y) = y^3) :
  ∃ (z : ℝ), z > 0 ∧ z^3 + z^2 - 1 = 0 ∧ y^6 = z := by
  sorry

end NUMINAMATH_CALUDE_y_sixth_power_root_l911_91129


namespace NUMINAMATH_CALUDE_polygon_internal_external_angles_equal_l911_91139

theorem polygon_internal_external_angles_equal (n : ℕ) : 
  (n : ℝ) ≥ 3 → ((n - 2) * 180 = 360) → n = 4 := by sorry

end NUMINAMATH_CALUDE_polygon_internal_external_angles_equal_l911_91139


namespace NUMINAMATH_CALUDE_diagonal_shorter_than_midpoint_distance_l911_91187

-- Define the quadrilateral ABCD
variables {A B C D : EuclideanSpace ℝ (Fin 2)}

-- Define the property that a circle through three points is tangent to a line segment
def is_tangent_circle (P Q R S T : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ),
    dist center P = radius ∧ dist center Q = radius ∧ dist center R = radius ∧
    dist center S = radius ∧ dist S T = dist center S + dist center T

-- State the theorem
theorem diagonal_shorter_than_midpoint_distance
  (h1 : is_tangent_circle A B C C D)
  (h2 : is_tangent_circle A C D A B) :
  dist A C < (dist A D + dist B C) / 2 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_shorter_than_midpoint_distance_l911_91187


namespace NUMINAMATH_CALUDE_amount_difference_l911_91192

theorem amount_difference (p q r : ℝ) : 
  p = 47.99999999999999 →
  q = p / 6 →
  r = p / 6 →
  p - (q + r) = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_amount_difference_l911_91192


namespace NUMINAMATH_CALUDE_inverse_relationship_R_squared_residuals_l911_91161

/-- Represents the coefficient of determination in regression analysis -/
def R_squared : ℝ := sorry

/-- Represents the sum of squares of residuals in regression analysis -/
def sum_of_squares_residuals : ℝ := sorry

/-- States that there is an inverse relationship between R² and the sum of squares of residuals -/
theorem inverse_relationship_R_squared_residuals :
  ∀ (R₁ R₂ : ℝ) (SSR₁ SSR₂ : ℝ),
    R₁ < R₂ → SSR₁ > SSR₂ :=
by sorry

end NUMINAMATH_CALUDE_inverse_relationship_R_squared_residuals_l911_91161


namespace NUMINAMATH_CALUDE_quadratic_factorization_l911_91158

theorem quadratic_factorization (x : ℝ) : -2 * x^2 + 2 * x - (1/2) = -2 * (x - 1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l911_91158


namespace NUMINAMATH_CALUDE_inequality_solution_set_l911_91108

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (a + 1)*x + a > 0}
  if a < 1 then S = {x : ℝ | x < a ∨ x > 1}
  else if a = 1 then S = {x : ℝ | x ≠ 1}
  else S = {x : ℝ | x < 1 ∨ x > a} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l911_91108


namespace NUMINAMATH_CALUDE_radical_simplification_l911_91147

theorem radical_simplification (y : ℝ) (h : y > 0) :
  Real.sqrt (50 * y) * Real.sqrt (5 * y) * Real.sqrt (45 * y) = 15 * y * Real.sqrt (10 * y) :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l911_91147


namespace NUMINAMATH_CALUDE_base_3_to_decimal_l911_91151

/-- Converts a list of digits in base k to its decimal representation -/
def to_decimal (digits : List Nat) (k : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * k^i) 0

/-- The base-3 representation of a number -/
def base_3_number : List Nat := [1, 0, 2]

/-- Theorem stating that the base-3 number (102)₃ is equal to 11 in decimal -/
theorem base_3_to_decimal :
  to_decimal base_3_number 3 = 11 := by sorry

end NUMINAMATH_CALUDE_base_3_to_decimal_l911_91151


namespace NUMINAMATH_CALUDE_candy_store_food_colouring_l911_91137

/-- The amount of food colouring used by a candy store in one day -/
def total_food_colouring (lollipop_count : ℕ) (hard_candy_count : ℕ) 
  (lollipop_colouring : ℕ) (hard_candy_colouring : ℕ) : ℕ :=
  lollipop_count * lollipop_colouring + hard_candy_count * hard_candy_colouring

/-- Theorem stating the total amount of food colouring used by the candy store -/
theorem candy_store_food_colouring : 
  total_food_colouring 100 5 5 20 = 600 := by
  sorry

end NUMINAMATH_CALUDE_candy_store_food_colouring_l911_91137


namespace NUMINAMATH_CALUDE_equation_solution_l911_91115

theorem equation_solution : ∃ x : ℚ, (x^2 + 4*x + 7) / (x + 5) = x + 6 ∧ x = -23/7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l911_91115


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l911_91124

/-- Given complex numbers z₁, z₂, z₃ that satisfy certain conditions,
    prove that z₁z₂/z₃ = -5. -/
theorem complex_ratio_theorem (z₁ z₂ z₃ : ℂ)
  (h1 : Complex.abs z₁ = Complex.abs z₂)
  (h2 : Complex.abs z₁ = Real.sqrt 3 * Complex.abs z₃)
  (h3 : z₁ + z₃ = z₂) :
  z₁ * z₂ / z₃ = -5 := by
  sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l911_91124


namespace NUMINAMATH_CALUDE_quiz_goal_achievement_l911_91167

theorem quiz_goal_achievement (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (current_as : ℕ) : 
  total_quizzes = 40 →
  goal_percentage = 85 / 100 →
  completed_quizzes = 25 →
  current_as = 20 →
  ∃ (max_non_as : ℕ), 
    max_non_as = 1 ∧ 
    (current_as + (total_quizzes - completed_quizzes - max_non_as)) / total_quizzes ≥ goal_percentage ∧
    ∀ (x : ℕ), x > max_non_as → 
      (current_as + (total_quizzes - completed_quizzes - x)) / total_quizzes < goal_percentage :=
by sorry

end NUMINAMATH_CALUDE_quiz_goal_achievement_l911_91167


namespace NUMINAMATH_CALUDE_division_problem_l911_91120

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2 / 5) : 
  c / a = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l911_91120


namespace NUMINAMATH_CALUDE_circle_radius_is_three_l911_91178

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y - 7 = 0

-- State the theorem
theorem circle_radius_is_three :
  ∃ (h k r : ℝ), r = 3 ∧ ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_three_l911_91178


namespace NUMINAMATH_CALUDE_board_length_l911_91112

-- Define the lengths of the two pieces
def shorter_piece : ℝ := 2
def longer_piece : ℝ := 2 * shorter_piece

-- Define the total length of the board
def total_length : ℝ := shorter_piece + longer_piece

-- Theorem to prove
theorem board_length : total_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_board_length_l911_91112


namespace NUMINAMATH_CALUDE_inequality_proof_l911_91128

theorem inequality_proof (a b : ℝ) (h : a * b > 0) : b / a + a / b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l911_91128


namespace NUMINAMATH_CALUDE_train_speed_problem_l911_91142

/-- Proves that given the conditions of the train problem, the average speed of Train B is 43 miles per hour. -/
theorem train_speed_problem (initial_gap : ℝ) (train_a_speed : ℝ) (overtake_time : ℝ) (final_gap : ℝ) :
  initial_gap = 13 →
  train_a_speed = 37 →
  overtake_time = 5 →
  final_gap = 17 →
  (initial_gap + train_a_speed * overtake_time + final_gap) / overtake_time = 43 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l911_91142


namespace NUMINAMATH_CALUDE_triangle_side_length_l911_91166

theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  b = 2 * Real.sqrt 3 →
  a = 2 →
  B = π / 3 →  -- 60° in radians
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos B →
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l911_91166


namespace NUMINAMATH_CALUDE_power_multiplication_l911_91152

theorem power_multiplication (a : ℝ) : (-a^2)^3 * a^3 = -a^9 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_l911_91152


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l911_91157

theorem decimal_sum_to_fraction : 
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 733 / 12500 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l911_91157


namespace NUMINAMATH_CALUDE_impossible_three_coin_piles_l911_91153

/-- Represents the coin removal and division process -/
def coin_process (initial_coins : ℕ) (steps : ℕ) : Prop :=
  ∃ (final_piles : ℕ),
    (initial_coins - steps = 3 * final_piles) ∧
    (final_piles = steps + 1)

/-- Theorem stating the impossibility of ending with only piles of three coins -/
theorem impossible_three_coin_piles : ¬∃ (steps : ℕ), coin_process 2013 steps :=
  sorry

end NUMINAMATH_CALUDE_impossible_three_coin_piles_l911_91153


namespace NUMINAMATH_CALUDE_garden_area_difference_l911_91127

def alice_length : ℝ := 15
def alice_width : ℝ := 30
def bob_length : ℝ := 18
def bob_width : ℝ := 28

theorem garden_area_difference :
  bob_length * bob_width - alice_length * alice_width = 54 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_difference_l911_91127


namespace NUMINAMATH_CALUDE_vehicle_inspection_is_systematic_l911_91173

/-- Represents a vehicle's license plate -/
structure LicensePlate where
  number : Nat

/-- Represents a sampling method -/
inductive SamplingMethod
  | Systematic
  | Other

/-- The criterion for selecting a vehicle based on its license plate -/
def selectionCriterion (plate : LicensePlate) : Bool :=
  plate.number % 10 = 5

/-- The sampling method used in the vehicle inspection process -/
def vehicleInspectionSampling : SamplingMethod :=
  SamplingMethod.Systematic

/-- Theorem stating that the vehicle inspection sampling method is systematic sampling -/
theorem vehicle_inspection_is_systematic :
  vehicleInspectionSampling = SamplingMethod.Systematic :=
sorry

end NUMINAMATH_CALUDE_vehicle_inspection_is_systematic_l911_91173


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l911_91196

/-- Given a two-digit number, prove that if the difference between the original number
    and the number with interchanged digits is 54, then the difference between its two digits is 6. -/
theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 54 → x - y = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l911_91196


namespace NUMINAMATH_CALUDE_hyperbola_focus_k_value_l911_91125

/-- Theorem: For a hyperbola with equation 8kx^2 - ky^2 = 8 and one focus at (0, -3), the value of k is -1. -/
theorem hyperbola_focus_k_value (k : ℝ) : 
  (∀ x y : ℝ, 8 * k * x^2 - k * y^2 = 8) → -- hyperbola equation
  (∃ x : ℝ, (x, -3) ∈ {(x, y) | x^2 / (8 / k) + y^2 / (8 / k + 1) = 1}) → -- focus at (0, -3)
  k = -1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_k_value_l911_91125


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l911_91104

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 18 = -6) →
  (a 2 * a 18 = 4) →
  a 4 * a 16 + a 10 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l911_91104
