import Mathlib

namespace NUMINAMATH_CALUDE_sine_function_properties_l3284_328404

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem sine_function_properties (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : |φ| < π/2) 
  (h_max : f ω φ (π/4) = 1) 
  (h_min : f ω φ (7*π/12) = -1) 
  (h_period : ∃ T > 0, ∀ x, f ω φ (x + T) = f ω φ x) :
  ω = 3 ∧ 
  φ = -π/4 ∧ 
  ∀ k : ℤ, ∀ x ∈ Set.Icc (2*k*π/3 + π/4) (2*k*π/3 + 7*π/12), 
    ∀ y ∈ Set.Icc (2*k*π/3 + π/4) (2*k*π/3 + 7*π/12), 
      x ≤ y → f ω φ x ≥ f ω φ y :=
by sorry

end NUMINAMATH_CALUDE_sine_function_properties_l3284_328404


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3284_328409

-- Define the quadratic function f(x)
def f (x : ℝ) : ℝ := -x^2 + 2*x + 15

-- Define g(x) in terms of f(x)
def g (x : ℝ) : ℝ := f x + (-2)*x

-- Theorem statement
theorem quadratic_function_properties :
  -- Vertex of f(x) is at (1, 16)
  (f 1 = 16) ∧
  -- Roots of f(x) are 8 units apart
  (∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ r₂ - r₁ = 8) →
  -- Conclusion 1: f(x) = -x^2 + 2x + 15
  (∀ x : ℝ, f x = -x^2 + 2*x + 15) ∧
  -- Conclusion 2: Maximum value of g(x) on [0, 2] is 7
  (∀ x : ℝ, x ≥ 0 ∧ x ≤ 2 → g x ≤ 7) ∧ (∃ x : ℝ, x ≥ 0 ∧ x ≤ 2 ∧ g x = 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3284_328409


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3284_328462

-- Define the hyperbola
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2^2 / b^2) - (p.1^2 / a^2) = 1}

-- Define the asymptotes
def Asymptotes (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 ∨ p.2 = -m * p.1}

theorem hyperbola_equation 
  (h : (3, 2 * Real.sqrt 2) ∈ Hyperbola 3 2) 
  (a : Asymptotes (2/3) = Asymptotes (2/3)) :
  Hyperbola 3 2 = {p : ℝ × ℝ | (p.2^2 / 4) - (p.1^2 / 9) = 1} :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3284_328462


namespace NUMINAMATH_CALUDE_square_of_five_power_plus_four_l3284_328463

theorem square_of_five_power_plus_four (n : ℕ) : 
  (∃ m : ℕ, 5^n + 4 = m^2) ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_square_of_five_power_plus_four_l3284_328463


namespace NUMINAMATH_CALUDE_three_positions_from_six_people_l3284_328416

/-- The number of ways to choose three distinct positions from a group of people -/
def choose_three_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- The number of people in the group -/
def group_size : ℕ := 6

/-- Theorem: The number of ways to choose a President, Vice-President, and Secretary 
    from a group of 6 people, where all positions must be filled by different individuals, 
    is equal to 120. -/
theorem three_positions_from_six_people : 
  choose_three_positions group_size = 120 := by sorry

end NUMINAMATH_CALUDE_three_positions_from_six_people_l3284_328416


namespace NUMINAMATH_CALUDE_specific_log_stack_count_l3284_328443

/-- Represents a stack of logs -/
structure LogStack where
  bottomCount : ℕ
  topCount : ℕ
  decreaseRate : ℕ

/-- Calculates the total number of logs in the stack -/
def totalLogs (stack : LogStack) : ℕ :=
  let rowCount := stack.bottomCount - stack.topCount + 1
  let avgRowCount := (stack.bottomCount + stack.topCount) / 2
  rowCount * avgRowCount

/-- Theorem stating that the specific log stack has 110 logs -/
theorem specific_log_stack_count :
  ∃ (stack : LogStack),
    stack.bottomCount = 15 ∧
    stack.topCount = 5 ∧
    stack.decreaseRate = 1 ∧
    totalLogs stack = 110 := by
  sorry

end NUMINAMATH_CALUDE_specific_log_stack_count_l3284_328443


namespace NUMINAMATH_CALUDE_extremum_implies_zero_derivative_zero_derivative_not_implies_extremum_l3284_328449

/-- A function f : ℝ → ℝ attains an extremum at x₀ -/
def AttainsExtremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  (∀ x, f x ≤ f x₀) ∨ (∀ x, f x ≥ f x₀)

/-- The derivative of f at x₀ is 0 -/
def DerivativeZero (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  deriv f x₀ = 0

theorem extremum_implies_zero_derivative
  (f : ℝ → ℝ) (x₀ : ℝ) (h : Differentiable ℝ f) :
  AttainsExtremum f x₀ → DerivativeZero f x₀ :=
sorry

theorem zero_derivative_not_implies_extremum :
  ∃ (f : ℝ → ℝ) (x₀ : ℝ), Differentiable ℝ f ∧ DerivativeZero f x₀ ∧ ¬AttainsExtremum f x₀ :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_zero_derivative_zero_derivative_not_implies_extremum_l3284_328449


namespace NUMINAMATH_CALUDE_movie_ticket_price_difference_l3284_328446

theorem movie_ticket_price_difference (regular_price children_price : ℕ) : 
  regular_price = 9 →
  2 * regular_price + 3 * children_price = 39 →
  regular_price - children_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_movie_ticket_price_difference_l3284_328446


namespace NUMINAMATH_CALUDE_alex_dresses_theorem_l3284_328483

/-- Calculates the maximum number of complete dresses Alex can make --/
def max_dresses (initial_silk initial_satin initial_chiffon : ℕ) 
                (silk_per_dress satin_per_dress chiffon_per_dress : ℕ) 
                (friends : ℕ) (silk_per_friend satin_per_friend chiffon_per_friend : ℕ) : ℕ :=
  let remaining_silk := initial_silk - friends * silk_per_friend
  let remaining_satin := initial_satin - friends * satin_per_friend
  let remaining_chiffon := initial_chiffon - friends * chiffon_per_friend
  min (remaining_silk / silk_per_dress) 
      (min (remaining_satin / satin_per_dress) (remaining_chiffon / chiffon_per_dress))

theorem alex_dresses_theorem : 
  max_dresses 600 400 350 5 3 2 8 15 10 5 = 96 := by
  sorry

end NUMINAMATH_CALUDE_alex_dresses_theorem_l3284_328483


namespace NUMINAMATH_CALUDE_christine_savings_l3284_328427

/-- Calculates the amount saved given a commission rate, total sales, and personal needs allocation. -/
def amount_saved (commission_rate : ℝ) (total_sales : ℝ) (personal_needs_rate : ℝ) : ℝ :=
  let commission_earned := commission_rate * total_sales
  let savings_rate := 1 - personal_needs_rate
  savings_rate * commission_earned

/-- Proves that given a 12% commission rate on $24000 worth of sales, 
    and allocating 60% of earnings to personal needs, the amount saved is $1152. -/
theorem christine_savings : 
  amount_saved 0.12 24000 0.60 = 1152 := by
sorry

end NUMINAMATH_CALUDE_christine_savings_l3284_328427


namespace NUMINAMATH_CALUDE_polynomial_proofs_l3284_328420

theorem polynomial_proofs (x : ℝ) : 
  (x^2 + 2*x - 3 = (x + 3)*(x - 1)) ∧ 
  (x^2 + 8*x + 7 = (x + 7)*(x + 1)) ∧ 
  (-x^2 + 2/3*x + 1 < 4/3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_proofs_l3284_328420


namespace NUMINAMATH_CALUDE_machine_output_for_26_l3284_328474

def machine_operation (input : ℕ) : ℕ :=
  (input + 15) - 6

theorem machine_output_for_26 :
  machine_operation 26 = 35 := by
  sorry

end NUMINAMATH_CALUDE_machine_output_for_26_l3284_328474


namespace NUMINAMATH_CALUDE_four_positions_l3284_328434

/-- Represents a cell in the 4x4 grid -/
structure Cell :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the value in a cell -/
inductive CellValue
  | One
  | Two
  | Three
  | Four

/-- Represents the 4x4 grid -/
def Grid := Cell → Option CellValue

/-- Check if a 2x2 square is valid (contains 1, 2, 3, 4 exactly once) -/
def isValidSquare (g : Grid) (topLeft : Cell) : Prop := sorry

/-- Check if the entire grid is valid -/
def isValidGrid (g : Grid) : Prop := sorry

/-- The given partial grid -/
def partialGrid : Grid := sorry

/-- Theorem stating the positions of fours in the grid -/
theorem four_positions (g : Grid) 
  (h1 : isValidGrid g) 
  (h2 : g = partialGrid) : 
  g ⟨0, 2⟩ = some CellValue.Four ∧ 
  g ⟨1, 0⟩ = some CellValue.Four ∧ 
  g ⟨2, 1⟩ = some CellValue.Four ∧ 
  g ⟨3, 3⟩ = some CellValue.Four := by
  sorry

end NUMINAMATH_CALUDE_four_positions_l3284_328434


namespace NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l3284_328408

theorem and_sufficient_not_necessary_for_or :
  (∃ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_and_sufficient_not_necessary_for_or_l3284_328408


namespace NUMINAMATH_CALUDE_reebok_cost_is_35_l3284_328488

/-- The cost of a pair of Reebok shoes -/
def reebok_cost : ℚ := 35

/-- Alice's sales quota -/
def quota : ℚ := 1000

/-- The cost of a pair of Adidas shoes -/
def adidas_cost : ℚ := 45

/-- The cost of a pair of Nike shoes -/
def nike_cost : ℚ := 60

/-- The number of Nike shoes sold -/
def nike_sold : ℕ := 8

/-- The number of Adidas shoes sold -/
def adidas_sold : ℕ := 6

/-- The number of Reebok shoes sold -/
def reebok_sold : ℕ := 9

/-- The amount by which Alice exceeded her quota -/
def excess : ℚ := 65

theorem reebok_cost_is_35 :
  reebok_cost * reebok_sold + nike_cost * nike_sold + adidas_cost * adidas_sold = quota + excess :=
by sorry

end NUMINAMATH_CALUDE_reebok_cost_is_35_l3284_328488


namespace NUMINAMATH_CALUDE_subset_condition_l3284_328467

def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B (a : ℝ) : Set ℝ := {x | 0 < x ∧ x < a}

theorem subset_condition (a : ℝ) : A ⊆ B a ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l3284_328467


namespace NUMINAMATH_CALUDE_sixth_graders_percentage_l3284_328407

theorem sixth_graders_percentage (seventh_graders : ℕ) (seventh_graders_percentage : ℚ) (sixth_graders : ℕ) :
  seventh_graders = 64 →
  seventh_graders_percentage = 32 / 100 →
  sixth_graders = 76 →
  (sixth_graders : ℚ) / ((seventh_graders : ℚ) / seventh_graders_percentage) = 38 / 100 := by
  sorry

end NUMINAMATH_CALUDE_sixth_graders_percentage_l3284_328407


namespace NUMINAMATH_CALUDE_max_value_of_S_l3284_328493

theorem max_value_of_S (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let S := min x (min (y + 1/x) (1/y))
  ∃ (max_S : ℝ), max_S = Real.sqrt 2 ∧ 
    (∀ x' y' : ℝ, x' > 0 → y' > 0 → 
      min x' (min (y' + 1/x') (1/y')) ≤ max_S) ∧
    (S = max_S ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_S_l3284_328493


namespace NUMINAMATH_CALUDE_system_two_solutions_l3284_328479

/-- The system of equations has exactly two solutions if and only if a = 49 or a = 169 -/
theorem system_two_solutions (a : ℝ) :
  (∃! (s : Set (ℝ × ℝ)), s.ncard = 2 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ s ↔ 
      (|x + y + 5| + |y - x + 5| = 10 ∧
       (|x| - 12)^2 + (|y| - 5)^2 = a))) ↔
  (a = 49 ∨ a = 169) :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_l3284_328479


namespace NUMINAMATH_CALUDE_base6_addition_l3284_328441

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The main theorem to prove -/
theorem base6_addition :
  base6ToDecimal [4, 5, 3, 5] + base6ToDecimal [2, 3, 2, 4, 3] =
  base6ToDecimal [3, 2, 2, 2, 2] := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_l3284_328441


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_nine_l3284_328447

theorem smallest_four_digit_mod_nine : ∃ n : ℕ, 
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 9 = 5) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 9 = 5 → m ≥ n) ∧ 
  (n = 1004) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_nine_l3284_328447


namespace NUMINAMATH_CALUDE_min_value_2a_plus_1_l3284_328466

theorem min_value_2a_plus_1 (a : ℝ) (h : 6 * a^2 + 5 * a + 4 = 3) :
  ∃ (m : ℝ), (2 * a + 1 ≥ m) ∧ (∀ x, 6 * x^2 + 5 * x + 4 = 3 → 2 * x + 1 ≥ m) ∧ m = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_1_l3284_328466


namespace NUMINAMATH_CALUDE_raffle_ticket_cost_l3284_328444

theorem raffle_ticket_cost (total_amount : ℕ) (num_tickets : ℕ) (cost_per_ticket : ℚ) : 
  total_amount = 620 → num_tickets = 155 → cost_per_ticket = 4 → 
  (total_amount : ℚ) / num_tickets = cost_per_ticket :=
by sorry

end NUMINAMATH_CALUDE_raffle_ticket_cost_l3284_328444


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3284_328456

theorem quadratic_inequality (x : ℝ) : x^2 + 3*x - 18 < 0 ↔ -6 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3284_328456


namespace NUMINAMATH_CALUDE_min_value_quadratic_roots_l3284_328426

theorem min_value_quadratic_roots (k : ℝ) (α β : ℝ) : 
  (α ^ 2 - 2 * k * α + k + 20 = 0) →
  (β ^ 2 - 2 * k * β + k + 20 = 0) →
  (k ≤ -4 ∨ k ≥ 5) →
  (∀ k', k' ≤ -4 ∨ k' ≥ 5 → (α + 1) ^ 2 + (β + 1) ^ 2 ≥ 18) ∧
  ((α + 1) ^ 2 + (β + 1) ^ 2 = 18 ↔ k = -4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_roots_l3284_328426


namespace NUMINAMATH_CALUDE_problem_statement_l3284_328473

theorem problem_statement (m : ℝ) : 
  let U : Set ℝ := Set.univ
  let A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
  let B : Set ℝ := {x | x^2 + (m+1)*x + m = 0}
  (Set.compl A ∩ B = ∅) → (m = 1 ∨ m = 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3284_328473


namespace NUMINAMATH_CALUDE_pool_water_rates_l3284_328480

/-- Represents the water delivery rates for two pools -/
structure PoolRates :=
  (first : ℝ)
  (second : ℝ)

/-- Proves that the water delivery rates for two pools satisfy the given conditions -/
theorem pool_water_rates :
  ∃ (rates : PoolRates),
    rates.first = 90 ∧
    rates.second = 60 ∧
    rates.first = rates.second + 30 ∧
    ∃ (t : ℝ),
      (rates.first * t + rates.second * t = 2 * rates.first * t) ∧
      (rates.first * (t + 8/3) = rates.first * t) ∧
      (rates.second * (t + 10/3) = rates.second * t) :=
by sorry

end NUMINAMATH_CALUDE_pool_water_rates_l3284_328480


namespace NUMINAMATH_CALUDE_minerals_found_today_l3284_328486

def minerals_yesterday (gemstones_yesterday : ℕ) : ℕ := 2 * gemstones_yesterday

theorem minerals_found_today 
  (gemstones_today minerals_today : ℕ) 
  (h1 : minerals_today = 48) 
  (h2 : gemstones_today = 21) : 
  minerals_today - minerals_yesterday gemstones_today = 6 := by
  sorry

end NUMINAMATH_CALUDE_minerals_found_today_l3284_328486


namespace NUMINAMATH_CALUDE_donation_to_third_home_l3284_328454

/-- Proves that the donation to the third home is $230.00 -/
theorem donation_to_third_home 
  (total_donation : ℝ) 
  (first_home_donation : ℝ) 
  (second_home_donation : ℝ)
  (h1 : total_donation = 700)
  (h2 : first_home_donation = 245)
  (h3 : second_home_donation = 225) :
  total_donation - first_home_donation - second_home_donation = 230 := by
  sorry

end NUMINAMATH_CALUDE_donation_to_third_home_l3284_328454


namespace NUMINAMATH_CALUDE_inscribed_parallelepiped_surface_area_l3284_328494

/-- A parallelepiped inscribed in a sphere -/
structure InscribedParallelepiped where
  /-- The radius of the circumscribed sphere -/
  sphere_radius : ℝ
  /-- The volume of the parallelepiped -/
  volume : ℝ
  /-- The edges of the parallelepiped -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- The sphere radius is √3 -/
  sphere_radius_eq : sphere_radius = Real.sqrt 3
  /-- The volume is 8 -/
  volume_eq : volume = 8
  /-- The volume is the product of the edges -/
  volume_product : volume = a * b * c
  /-- The diagonal of the parallelepiped equals the diameter of the sphere -/
  diagonal_eq : a^2 + b^2 + c^2 = 4 * sphere_radius^2

/-- The theorem stating that the surface area of the inscribed parallelepiped is 24 -/
theorem inscribed_parallelepiped_surface_area
  (p : InscribedParallelepiped) : 2 * (p.a * p.b + p.b * p.c + p.c * p.a) = 24 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_parallelepiped_surface_area_l3284_328494


namespace NUMINAMATH_CALUDE_symmetric_points_sum_sum_of_symmetric_point_l3284_328461

/-- Given two points A and B in a 2D plane, where A is symmetric to B with respect to the origin,
    prove that the sum of B's coordinates equals the negative sum of A's coordinates. -/
theorem symmetric_points_sum (A B : ℝ × ℝ) (hSymmetric : B = (-A.1, -A.2)) :
  B.1 + B.2 = -(A.1 + A.2) := by
  sorry

/-- Prove that if point A(-2022, -1) is symmetric with respect to the origin O to point B(a, b),
    then a + b = 2023. -/
theorem sum_of_symmetric_point :
  ∃ (a b : ℝ), (a, b) = (-(-2022), -(-1)) → a + b = 2023 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_sum_of_symmetric_point_l3284_328461


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3284_328475

theorem cubic_root_sum_cubes (a b c : ℂ) : 
  (5 * a^3 + 2014 * a + 4027 = 0) →
  (5 * b^3 + 2014 * b + 4027 = 0) →
  (5 * c^3 + 2014 * c + 4027 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 2416.2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3284_328475


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l3284_328497

def is_hyperbola (m : ℝ) : Prop :=
  (16 - m) * (9 - m) < 0

theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ 9 < m ∧ m < 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l3284_328497


namespace NUMINAMATH_CALUDE_original_bill_calculation_l3284_328419

theorem original_bill_calculation (num_friends : ℕ) (discount_rate : ℚ) (individual_payment : ℚ) :
  num_friends = 5 →
  discount_rate = 6 / 100 →
  individual_payment = 188 / 10 →
  ∃ (original_bill : ℚ), 
    (1 - discount_rate) * original_bill = num_friends * individual_payment ∧
    original_bill = 100 := by
  sorry

end NUMINAMATH_CALUDE_original_bill_calculation_l3284_328419


namespace NUMINAMATH_CALUDE_solution_value_l3284_328421

theorem solution_value (a : ℝ) (h : a^2 - 2*a - 1 = 0) : a^2 - 2*a + 2022 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3284_328421


namespace NUMINAMATH_CALUDE_unique_abcd_l3284_328418

def is_valid_abcd (abcd : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    abcd = a * 1000 + b * 100 + c * 10 + d ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    0 < a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 < d ∧ d < 10 ∧
    abcd / d = d * 100 + b * 10 + a

theorem unique_abcd :
  ∃! abcd : ℕ, is_valid_abcd abcd ∧ abcd = 1964 :=
sorry

end NUMINAMATH_CALUDE_unique_abcd_l3284_328418


namespace NUMINAMATH_CALUDE_jose_play_time_l3284_328417

/-- Calculates the total hours played given the time spent on football and basketball in minutes -/
def total_hours_played (football_minutes : ℕ) (basketball_minutes : ℕ) : ℚ :=
  (football_minutes + basketball_minutes : ℚ) / 60

/-- Theorem stating that playing football for 30 minutes and basketball for 60 minutes results in 1.5 hours of total play time -/
theorem jose_play_time : total_hours_played 30 60 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_jose_play_time_l3284_328417


namespace NUMINAMATH_CALUDE_association_ticket_sales_l3284_328413

/-- Represents an association with male and female members selling raffle tickets -/
structure Association where
  male_members : ℕ
  female_members : ℕ
  male_avg_tickets : ℝ
  female_avg_tickets : ℝ
  overall_avg_tickets : ℝ

/-- The theorem stating the conditions and the result to be proved -/
theorem association_ticket_sales (a : Association) 
  (h1 : a.female_members = 2 * a.male_members)
  (h2 : a.overall_avg_tickets = 66)
  (h3 : a.male_avg_tickets = 58) :
  a.female_avg_tickets = 70 := by
  sorry


end NUMINAMATH_CALUDE_association_ticket_sales_l3284_328413


namespace NUMINAMATH_CALUDE_sequence_sum_equals_642_l3284_328425

def a (n : ℕ) : ℤ := (-2) ^ n
def b (n : ℕ) : ℤ := (-2) ^ n + 2
def c (n : ℕ) : ℚ := ((-2) ^ n : ℚ) / 2

theorem sequence_sum_equals_642 :
  ∃! n : ℕ, (a n : ℚ) + (b n : ℚ) + c n = 642 ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_642_l3284_328425


namespace NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l3284_328453

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Theorem stating that one cubic foot is equal to 1728 cubic inches -/
theorem cubic_foot_to_cubic_inches : (1 : ℝ)^3 * feet_to_inches^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l3284_328453


namespace NUMINAMATH_CALUDE_function_sum_equals_four_l3284_328411

/-- Given a function f(x) = ax^7 - bx^5 + cx^3 + 2, prove that f(5) + f(-5) = 4 -/
theorem function_sum_equals_four (a b c m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^7 - b * x^5 + c * x^3 + 2
  f (-5) = m →
  f 5 + f (-5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_sum_equals_four_l3284_328411


namespace NUMINAMATH_CALUDE_circular_arrangement_equality_l3284_328487

/-- Given a circular arrangement of n people numbered 1 to n,
    if the distance from person 31 to person 7 is equal to
    the distance from person 31 to person 14, then n = 41. -/
theorem circular_arrangement_equality (n : ℕ) : n > 30 →
  (n - 31 + 7) % n = (14 - 31 + n) % n →
  n = 41 := by
  sorry


end NUMINAMATH_CALUDE_circular_arrangement_equality_l3284_328487


namespace NUMINAMATH_CALUDE_mobile_phone_price_l3284_328431

theorem mobile_phone_price (x : ℝ) : 
  (0.8 * (1.4 * x)) - x = 270 → x = 2250 := by
  sorry

end NUMINAMATH_CALUDE_mobile_phone_price_l3284_328431


namespace NUMINAMATH_CALUDE_units_digit_sum_l3284_328412

theorem units_digit_sum (n : ℕ) : (35^87 + 3^45) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_l3284_328412


namespace NUMINAMATH_CALUDE_sam_bank_total_l3284_328490

def initial_dimes : ℕ := 9
def initial_quarters : ℕ := 5
def initial_nickels : ℕ := 3

def dad_dimes : ℕ := 7
def dad_quarters : ℕ := 2

def mom_nickels : ℕ := 1
def mom_dimes : ℕ := 2

def grandma_dollars : ℕ := 3

def sister_quarters : ℕ := 4
def sister_nickels : ℕ := 2

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5
def dollar_value : ℕ := 100

theorem sam_bank_total :
  (initial_dimes * dime_value +
   initial_quarters * quarter_value +
   initial_nickels * nickel_value +
   dad_dimes * dime_value +
   dad_quarters * quarter_value -
   mom_nickels * nickel_value -
   mom_dimes * dime_value +
   grandma_dollars * dollar_value +
   sister_quarters * quarter_value +
   sister_nickels * nickel_value) = 735 := by
  sorry

end NUMINAMATH_CALUDE_sam_bank_total_l3284_328490


namespace NUMINAMATH_CALUDE_selling_price_ratio_l3284_328451

/-- Given an item with cost price c, prove that the ratio of selling prices
    y (at 20% profit) to x (at 10% loss) is 4/3 -/
theorem selling_price_ratio (c x y : ℝ) (hx : x = 0.9 * c) (hy : y = 1.2 * c) :
  y / x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l3284_328451


namespace NUMINAMATH_CALUDE_river_round_trip_time_l3284_328450

/-- Calculates the total time for a round trip on a river -/
theorem river_round_trip_time 
  (river_current : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_current = 8) 
  (h2 : boat_speed = 20) 
  (h3 : distance = 84) : 
  (distance / (boat_speed - river_current)) + (distance / (boat_speed + river_current)) = 10 := by
  sorry

#check river_round_trip_time

end NUMINAMATH_CALUDE_river_round_trip_time_l3284_328450


namespace NUMINAMATH_CALUDE_ratio_problem_l3284_328459

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y + 1) = 4 / 5) :
  x / y = 22 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3284_328459


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3284_328455

theorem ellipse_eccentricity (a b m n c : ℝ) : 
  a > b ∧ b > 0 ∧ m > 0 ∧ n > 0 →
  c^2 = a^2 - b^2 →
  c^2 = m^2 + n^2 →
  c^2 = a * m →
  n^2 = (2 * m^2 + c^2) / 2 →
  c / a = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3284_328455


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l3284_328476

theorem least_number_with_remainder (n : ℕ) : n ≥ 261 ∧ n % 37 = 2 ∧ n % 7 = 2 → n = 261 :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l3284_328476


namespace NUMINAMATH_CALUDE_transaction_fraction_l3284_328465

theorem transaction_fraction :
  let mabel_transactions : ℕ := 90
  let anthony_transactions : ℕ := mabel_transactions + mabel_transactions / 10
  let jade_transactions : ℕ := 82
  let cal_transactions : ℕ := jade_transactions - 16
  (cal_transactions : ℚ) / (anthony_transactions : ℚ) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_transaction_fraction_l3284_328465


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l3284_328499

-- Define the fuel efficiency of the car
def fuel_efficiency : ℝ := 32

-- Define the distance the car can travel
def distance : ℝ := 368

-- Define the total cost of gas
def total_cost : ℝ := 46

-- Theorem to prove the cost of gas per gallon
theorem gas_cost_per_gallon :
  (total_cost / (distance / fuel_efficiency)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l3284_328499


namespace NUMINAMATH_CALUDE_at_least_one_passes_l3284_328481

/-- The probability that at least one of three independent events occurs, given their individual probabilities -/
theorem at_least_one_passes (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_passes_l3284_328481


namespace NUMINAMATH_CALUDE_solve_for_T_l3284_328491

theorem solve_for_T : ∃ T : ℚ, (3/4 : ℚ) * (1/6 : ℚ) * T = (2/5 : ℚ) * (1/4 : ℚ) * 200 ∧ T = 80 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_T_l3284_328491


namespace NUMINAMATH_CALUDE_right_triangle_angle_calculation_l3284_328448

theorem right_triangle_angle_calculation (A B C : Real) : 
  A = 35 → C = 90 → A + B + C = 180 → B = 55 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_calculation_l3284_328448


namespace NUMINAMATH_CALUDE_class_composition_l3284_328492

/-- Represents a pair of numbers written by a child -/
structure Response :=
  (boys : ℕ)
  (girls : ℕ)

/-- Checks if a response is valid given the actual number of boys and girls -/
def is_valid_response (r : Response) (actual_boys : ℕ) (actual_girls : ℕ) : Prop :=
  (r.boys = actual_boys - 1 ∧ (r.girls = actual_girls - 1 + 4 ∨ r.girls = actual_girls - 1 - 4)) ∨
  (r.girls = actual_girls - 1 ∧ (r.boys = actual_boys - 1 + 4 ∨ r.boys = actual_boys - 1 - 4))

/-- The theorem to be proved -/
theorem class_composition :
  ∃ (boys girls : ℕ),
    boys = 14 ∧ girls = 15 ∧
    is_valid_response ⟨10, 14⟩ boys girls ∧
    is_valid_response ⟨13, 11⟩ boys girls ∧
    is_valid_response ⟨13, 19⟩ boys girls ∧
    ∀ (b g : ℕ),
      (is_valid_response ⟨10, 14⟩ b g ∧
       is_valid_response ⟨13, 11⟩ b g ∧
       is_valid_response ⟨13, 19⟩ b g) →
      b = boys ∧ g = girls :=
sorry

end NUMINAMATH_CALUDE_class_composition_l3284_328492


namespace NUMINAMATH_CALUDE_rem_evaluation_l3284_328429

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_evaluation :
  rem (7/12 : ℚ) (-3/4 : ℚ) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_rem_evaluation_l3284_328429


namespace NUMINAMATH_CALUDE_square_plus_one_greater_than_one_l3284_328440

theorem square_plus_one_greater_than_one (a : ℝ) (h : a ≠ 0) : a^2 + 1 > 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_greater_than_one_l3284_328440


namespace NUMINAMATH_CALUDE_alien_mineral_conversion_l3284_328452

/-- Converts a three-digit number from base 7 to base 10 -/
def base7ToBase10 (a b c : ℕ) : ℕ :=
  a * 7^2 + b * 7^1 + c * 7^0

/-- The base 7 number 365₇ is equal to 194 in base 10 -/
theorem alien_mineral_conversion :
  base7ToBase10 3 6 5 = 194 := by
  sorry

end NUMINAMATH_CALUDE_alien_mineral_conversion_l3284_328452


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l3284_328405

theorem sqrt_difference_equality : 2 * (Real.sqrt (49 + 81) - Real.sqrt (36 - 25)) = 2 * (Real.sqrt 130 - Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l3284_328405


namespace NUMINAMATH_CALUDE_product_real_implies_a_value_l3284_328415

theorem product_real_implies_a_value (z₁ z₂ : ℂ) (a : ℝ) :
  z₁ = 2 + I →
  z₂ = 1 + a * I →
  (z₁ * z₂).im = 0 →
  a = -1/2 := by sorry

end NUMINAMATH_CALUDE_product_real_implies_a_value_l3284_328415


namespace NUMINAMATH_CALUDE_prob_not_all_same_l3284_328422

-- Define a fair 6-sided die
def fair_die : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the probability of all dice showing the same number
def prob_all_same : ℚ := 1 / 1296

-- Theorem statement
theorem prob_not_all_same (d : ℕ) (n : ℕ) (p : ℚ) 
  (hd : d = fair_die) (hn : n = num_dice) (hp : p = prob_all_same) : 
  1 - p = 1295 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_all_same_l3284_328422


namespace NUMINAMATH_CALUDE_inequality_theorem_l3284_328489

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c ≥ a * b * c) :
  (2 / a + 3 / b + 6 / c ≥ 6 ∧ 2 / b + 3 / c + 6 / a ≥ 6) ∨
  (2 / a + 3 / b + 6 / c ≥ 6 ∧ 2 / c + 3 / a + 6 / b ≥ 6) ∨
  (2 / b + 3 / c + 6 / a ≥ 6 ∧ 2 / c + 3 / a + 6 / b ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3284_328489


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l3284_328428

variables (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)

theorem polynomial_coefficients :
  (x + 2) * (x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 →
  a₂ = 8 ∧ a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l3284_328428


namespace NUMINAMATH_CALUDE_integer_root_of_polynomial_l3284_328400

-- Define the polynomial
def P (x b c : ℚ) : ℚ := x^4 + 7*x^3 + b*x + c

-- State the theorem
theorem integer_root_of_polynomial (b c : ℚ) :
  (∃ (r : ℚ), r^2 = 5 ∧ P (2 + r) b c = 0) →  -- 2 + √5 is a root
  (∃ (n : ℤ), P n b c = 0) →                  -- There exists an integer root
  P 0 b c = 0                                 -- 0 is a root
:= by sorry

end NUMINAMATH_CALUDE_integer_root_of_polynomial_l3284_328400


namespace NUMINAMATH_CALUDE_root_equation_solution_l3284_328471

theorem root_equation_solution (y : ℝ) : 
  (y * (y^5)^(1/3))^(1/7) = 4 → y = 2^(21/4) := by
sorry

end NUMINAMATH_CALUDE_root_equation_solution_l3284_328471


namespace NUMINAMATH_CALUDE_problem_statement_l3284_328468

theorem problem_statement (x : ℝ) :
  (Real.sqrt x - 5) / 7 = 7 →
  ((x - 14)^2) / 10 = 842240.4 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3284_328468


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_boat_speed_proof_l3284_328439

/-- The speed of a boat in still water, given stream speed and downstream travel information -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 120)
  (h3 : downstream_time = 4)
  : ℝ :=
  let downstream_speed := downstream_distance / downstream_time
  let boat_speed := downstream_speed - stream_speed
  25

/-- Proof that the boat's speed in still water is 25 km/hr -/
theorem boat_speed_proof 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 120)
  (h3 : downstream_time = 4)
  : boat_speed_in_still_water stream_speed downstream_distance downstream_time h1 h2 h3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_boat_speed_proof_l3284_328439


namespace NUMINAMATH_CALUDE_expression_value_l3284_328495

theorem expression_value : 
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 4
  x^2 * y * z - x * y * z^2 = 48 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3284_328495


namespace NUMINAMATH_CALUDE_ceiling_minus_x_value_l3284_328406

theorem ceiling_minus_x_value (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ δ : ℝ, 0 < δ ∧ δ < 1 ∧ ⌈x⌉ - x = 1 - δ := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_value_l3284_328406


namespace NUMINAMATH_CALUDE_ratio_of_fractions_l3284_328460

theorem ratio_of_fractions : (1 : ℚ) / 6 / ((5 : ℚ) / 8) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_fractions_l3284_328460


namespace NUMINAMATH_CALUDE_coldness_probability_l3284_328442

def word1 := "CART"
def word2 := "BLEND"
def word3 := "SHOW"
def target_word := "COLDNESS"

def select_letters (word : String) (n : Nat) : Nat := Nat.choose word.length n

theorem coldness_probability :
  let p1 := (1 : ℚ) / select_letters word1 2
  let p2 := (1 : ℚ) / select_letters word2 4
  let p3 := (1 : ℚ) / 2
  p1 * p2 * p3 = 1 / 60 := by sorry

end NUMINAMATH_CALUDE_coldness_probability_l3284_328442


namespace NUMINAMATH_CALUDE_grid_value_l3284_328410

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first + seq.diff * (n - 1 : ℝ)

theorem grid_value (row : ArithmeticSequence) (col : ArithmeticSequence) : 
  row.first = 25 ∧ 
  row.nthTerm 4 = 11 ∧ 
  col.nthTerm 2 = 11 ∧ 
  col.nthTerm 3 = 11 ∧
  row.nthTerm 7 = col.nthTerm 1 →
  row.nthTerm 7 = -3 := by
  sorry


end NUMINAMATH_CALUDE_grid_value_l3284_328410


namespace NUMINAMATH_CALUDE_sugar_price_correct_l3284_328464

/-- The price of a kilogram of sugar -/
def sugar_price : ℝ := 1.50

/-- The price of a kilogram of salt -/
noncomputable def salt_price : ℝ := 5 - 3 * sugar_price

theorem sugar_price_correct : sugar_price = 1.50 := by
  have h1 : 2 * sugar_price + 5 * salt_price = 5.50 := by sorry
  have h2 : 3 * sugar_price + salt_price = 5 := by sorry
  sorry

end NUMINAMATH_CALUDE_sugar_price_correct_l3284_328464


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3284_328498

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  P.1 ≥ 0 →
  P.2 ≥ 0 →
  P.1^2 / a^2 - P.2^2 / b^2 = 1 →
  P.1^2 + P.2^2 = a^2 + b^2 →
  F₁.1 < 0 →
  F₂.1 > 0 →
  F₁.2 = 0 →
  F₂.2 = 0 →
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 3 * Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) →
  Real.sqrt ((F₂.1 - F₁.1)^2 / (2*a)^2) = Real.sqrt 10 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3284_328498


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_evaluate_trigonometric_fraction_l3284_328430

-- Part 1
theorem simplify_trigonometric_expression :
  (Real.sqrt (1 - 2 * Real.sin (20 * π / 180) * Real.cos (20 * π / 180))) /
  (Real.sin (160 * π / 180) - Real.sqrt (1 - Real.sin (20 * π / 180) ^ 2)) = -1 := by
  sorry

-- Part 2
theorem evaluate_trigonometric_fraction (α : Real) (h : Real.tan α = 1 / 3) :
  1 / (4 * Real.cos α ^ 2 - 6 * Real.sin α * Real.cos α) = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_evaluate_trigonometric_fraction_l3284_328430


namespace NUMINAMATH_CALUDE_triangle_side_ratio_max_l3284_328469

theorem triangle_side_ratio_max (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (1/2) * a * b * Real.sin C = c^2 / 4 →
  (∃ (x : ℝ), a / b + b / a ≤ x) ∧ 
  (a / b + b / a ≤ 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_max_l3284_328469


namespace NUMINAMATH_CALUDE_largest_corner_sum_l3284_328403

-- Define the face values of the cube
def face_values : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the property that opposite faces sum to 9
def opposite_sum_9 (faces : List ℕ) : Prop :=
  ∀ x ∈ faces, (9 - x) ∈ faces

-- Define a function to check if three numbers can be on adjacent faces
def can_be_adjacent (a b c : ℕ) : Prop :=
  a + b ≠ 9 ∧ b + c ≠ 9 ∧ a + c ≠ 9

-- Theorem statement
theorem largest_corner_sum :
  ∀ (cube : List ℕ),
  cube = face_values →
  opposite_sum_9 cube →
  (∃ (a b c : ℕ),
    a ∈ cube ∧ b ∈ cube ∧ c ∈ cube ∧
    can_be_adjacent a b c ∧
    (∀ (x y z : ℕ),
      x ∈ cube → y ∈ cube → z ∈ cube →
      can_be_adjacent x y z →
      x + y + z ≤ a + b + c)) →
  (∃ (a b c : ℕ),
    a ∈ cube ∧ b ∈ cube ∧ c ∈ cube ∧
    can_be_adjacent a b c ∧
    a + b + c = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_corner_sum_l3284_328403


namespace NUMINAMATH_CALUDE_last_digit_of_sum_l3284_328435

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Swaps the last two digits of a three-digit number -/
def ThreeDigitNumber.swap_last_two (n : ThreeDigitNumber) : ThreeDigitNumber :=
  { hundreds := n.hundreds
  , tens := n.ones
  , ones := n.tens
  , is_valid := by sorry }

theorem last_digit_of_sum (n : ThreeDigitNumber) :
  (n.value + (n.swap_last_two).value ≥ 1000) →
  (n.value + (n.swap_last_two).value < 2000) →
  (n.value + (n.swap_last_two).value) / 10 = 195 →
  (n.value + (n.swap_last_two).value) % 10 = 4 := by
  sorry


end NUMINAMATH_CALUDE_last_digit_of_sum_l3284_328435


namespace NUMINAMATH_CALUDE_simplify_expression_l3284_328437

theorem simplify_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x) * ((y^3 + 2) / y) + ((x^3 - 2) / y) * ((y^3 - 2) / x) = 2 * x^2 * y^2 + 8 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3284_328437


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3284_328423

theorem max_value_quadratic (f : ℝ → ℝ) (h : ∀ x, f x = -x^2 + 2*x + 3) :
  (∀ x ∈ Set.Icc 2 3, f x ≤ 3) ∧ (∃ x ∈ Set.Icc 2 3, f x = 3) := by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3284_328423


namespace NUMINAMATH_CALUDE_student_village_arrangements_l3284_328484

theorem student_village_arrangements :
  let num_students : ℕ := 3
  let num_villages : ℕ := 2
  let arrangements : ℕ := (num_students.choose (num_students - 1)) * (num_villages.factorial)
  arrangements = 6 := by
  sorry

end NUMINAMATH_CALUDE_student_village_arrangements_l3284_328484


namespace NUMINAMATH_CALUDE_sum_three_numbers_l3284_328436

theorem sum_three_numbers (x y z M : ℝ) : 
  x + y + z = 90 ∧ 
  x - 5 = M ∧ 
  y + 5 = M ∧ 
  5 * z = M → 
  M = 450 / 11 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l3284_328436


namespace NUMINAMATH_CALUDE_real_m_values_l3284_328457

theorem real_m_values (m : ℝ) : 
  let z : ℂ := m^2 * (1 + Complex.I) - m * (m + Complex.I)
  Complex.im z = 0 → m = 0 ∨ m = 1 := by
sorry

end NUMINAMATH_CALUDE_real_m_values_l3284_328457


namespace NUMINAMATH_CALUDE_ellipse_equation_l3284_328401

/-- The equation of an ellipse given its foci and the sum of distances from any point on the ellipse to the foci -/
theorem ellipse_equation (F₁ F₂ M : ℝ × ℝ) (d : ℝ) : 
  F₁ = (-4, 0) →
  F₂ = (4, 0) →
  d = 10 →
  (dist M F₁ + dist M F₂ = d) →
  (M.1^2 / 25 + M.2^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3284_328401


namespace NUMINAMATH_CALUDE_man_rowing_speed_l3284_328432

/-- The speed of a man rowing a boat against the stream, given his speed with the stream and his rate in still water. -/
def speed_against_stream (speed_with_stream : ℝ) (rate_still_water : ℝ) : ℝ :=
  abs (2 * rate_still_water - speed_with_stream)

/-- Theorem: Given a man's speed with the stream of 22 km/h and his rate in still water of 6 km/h, his speed against the stream is 10 km/h. -/
theorem man_rowing_speed 
  (h1 : speed_with_stream = 22)
  (h2 : rate_still_water = 6) :
  speed_against_stream speed_with_stream rate_still_water = 10 := by
  sorry

#eval speed_against_stream 22 6

end NUMINAMATH_CALUDE_man_rowing_speed_l3284_328432


namespace NUMINAMATH_CALUDE_impossible_cover_l3284_328414

/-- Represents an L-trimino piece -/
structure LTrimino where
  covers : Nat
  covers_eq : covers = 3

/-- Represents a 3x5 board with special squares -/
structure Board where
  total_squares : Nat
  total_squares_eq : total_squares = 15
  special_squares : Nat
  special_squares_eq : special_squares = 6

/-- States that it's impossible to cover the board with L-triminos -/
theorem impossible_cover (b : Board) (l : LTrimino) : 
  ¬∃ (n : Nat), n * l.covers = b.total_squares ∧ n ≥ b.special_squares :=
sorry

end NUMINAMATH_CALUDE_impossible_cover_l3284_328414


namespace NUMINAMATH_CALUDE_additional_men_problem_l3284_328433

/-- Calculates the number of additional men given initial conditions and new duration -/
def additional_men (initial_men : ℕ) (initial_days : ℚ) (new_days : ℚ) : ℚ :=
  (initial_men * initial_days / new_days) - initial_men

theorem additional_men_problem :
  let initial_men : ℕ := 1000
  let initial_days : ℚ := 17
  let new_days : ℚ := 11.333333333333334
  additional_men initial_men initial_days new_days = 500 := by
sorry

end NUMINAMATH_CALUDE_additional_men_problem_l3284_328433


namespace NUMINAMATH_CALUDE_prob_red_blue_black_l3284_328478

/-- Represents the color of a marble -/
inductive MarbleColor
  | Red
  | Green
  | Blue
  | White
  | Black
  | Yellow

/-- Represents a bag of marbles -/
structure MarbleBag where
  total : ℕ
  colors : List MarbleColor
  probs : MarbleColor → ℚ

/-- The probability of drawing a marble of a specific color or set of colors -/
def prob (bag : MarbleBag) (colors : List MarbleColor) : ℚ :=
  colors.map bag.probs |>.sum

/-- Theorem stating the probability of drawing a red, blue, or black marble -/
theorem prob_red_blue_black (bag : MarbleBag) :
  bag.total = 120 ∧
  bag.colors = [MarbleColor.Red, MarbleColor.Green, MarbleColor.Blue,
                MarbleColor.White, MarbleColor.Black, MarbleColor.Yellow] ∧
  bag.probs MarbleColor.White = 1/5 ∧
  bag.probs MarbleColor.Green = 3/10 ∧
  bag.probs MarbleColor.Yellow = 1/6 →
  prob bag [MarbleColor.Red, MarbleColor.Blue, MarbleColor.Black] = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_blue_black_l3284_328478


namespace NUMINAMATH_CALUDE_value_of_shares_theorem_l3284_328438

/-- Represents the value of shares bought by an investor -/
def value_of_shares (N : ℝ) : ℝ := 0.5 * N * 25

/-- Theorem stating the relationship between the value of shares and the number of shares -/
theorem value_of_shares_theorem (N : ℝ) (dividend_rate : ℝ) (return_rate : ℝ) (share_price : ℝ)
  (h1 : dividend_rate = 0.125)
  (h2 : return_rate = 0.25)
  (h3 : share_price = 25) :
  value_of_shares N = return_rate * (value_of_shares N) / dividend_rate := by
sorry

end NUMINAMATH_CALUDE_value_of_shares_theorem_l3284_328438


namespace NUMINAMATH_CALUDE_hedge_cost_and_quantity_l3284_328472

/-- Represents the cost and quantity of concrete blocks for a hedge --/
structure HedgeBlocks where
  cost_a : ℕ  -- Cost of Type A blocks
  cost_b : ℕ  -- Cost of Type B blocks
  cost_c : ℕ  -- Cost of Type C blocks
  qty_a : ℕ   -- Quantity of Type A blocks per section
  qty_b : ℕ   -- Quantity of Type B blocks per section
  qty_c : ℕ   -- Quantity of Type C blocks per section
  sections : ℕ -- Number of sections in the hedge

/-- Calculates the total cost and quantity of blocks for the entire hedge --/
def hedge_totals (h : HedgeBlocks) : ℕ × ℕ × ℕ × ℕ :=
  let total_cost := h.sections * (h.cost_a * h.qty_a + h.cost_b * h.qty_b + h.cost_c * h.qty_c)
  let total_a := h.sections * h.qty_a
  let total_b := h.sections * h.qty_b
  let total_c := h.sections * h.qty_c
  (total_cost, total_a, total_b, total_c)

theorem hedge_cost_and_quantity (h : HedgeBlocks) 
  (h_cost_a : h.cost_a = 2)
  (h_cost_b : h.cost_b = 3)
  (h_cost_c : h.cost_c = 4)
  (h_qty_a : h.qty_a = 20)
  (h_qty_b : h.qty_b = 10)
  (h_qty_c : h.qty_c = 5)
  (h_sections : h.sections = 8) :
  hedge_totals h = (720, 160, 80, 40) := by
  sorry

end NUMINAMATH_CALUDE_hedge_cost_and_quantity_l3284_328472


namespace NUMINAMATH_CALUDE_problem_solution_l3284_328496

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x^3 - x^2 + 5*x + (1 - a) * Real.log x

theorem problem_solution :
  (∃ a : ℝ, (∀ x : ℝ, (deriv (f a)) x = 0 ↔ x = 1) ∧ a = 1) ∧
  (∃ x : ℝ, (deriv (f 0)) x = -1) ∧
  (¬∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧
    ∃ d : ℝ, x₂ = x₁ + d ∧ x₃ = x₂ + d ∧
    (deriv (f 2)) x₂ = (f 2 x₃ - f 2 x₁) / (x₃ - x₁)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3284_328496


namespace NUMINAMATH_CALUDE_church_full_capacity_l3284_328445

/-- Calculates the total number of people that can be seated in a church with three sections -/
def church_capacity (section1_rows section1_chairs_per_row section2_rows section2_chairs_per_row section3_rows section3_chairs_per_row : ℕ) : ℕ :=
  section1_rows * section1_chairs_per_row +
  section2_rows * section2_chairs_per_row +
  section3_rows * section3_chairs_per_row

/-- Theorem stating that the church capacity is 490 given the specified section configurations -/
theorem church_full_capacity :
  church_capacity 15 8 20 6 25 10 = 490 := by
  sorry

end NUMINAMATH_CALUDE_church_full_capacity_l3284_328445


namespace NUMINAMATH_CALUDE_intersection_M_N_l3284_328470

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2 * x ∧ x > 0}
def N : Set ℝ := {x | ∃ y, y = Real.log (2 * x - x^2) ∧ x > 0 ∧ x < 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3284_328470


namespace NUMINAMATH_CALUDE_milk_remaining_l3284_328402

theorem milk_remaining (initial : ℚ) (given_away : ℚ) (remaining : ℚ) : 
  initial = 5 → given_away = 18/7 → remaining = initial - given_away → remaining = 17/7 := by
  sorry

end NUMINAMATH_CALUDE_milk_remaining_l3284_328402


namespace NUMINAMATH_CALUDE_gum_per_nickel_l3284_328477

/-- 
Given:
- initial_nickels: The number of nickels Quentavious started with
- remaining_nickels: The number of nickels Quentavious had left
- total_gum: The total number of gum pieces Quentavious received

Prove: The number of gum pieces per nickel is 2
-/
theorem gum_per_nickel 
  (initial_nickels : ℕ) 
  (remaining_nickels : ℕ) 
  (total_gum : ℕ) 
  (h1 : initial_nickels = 5)
  (h2 : remaining_nickels = 2)
  (h3 : total_gum = 6)
  : (total_gum : ℚ) / (initial_nickels - remaining_nickels : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gum_per_nickel_l3284_328477


namespace NUMINAMATH_CALUDE_cryptarithmetic_problem_l3284_328485

theorem cryptarithmetic_problem (A B C D : ℕ) : 
  (A + B + C = 11) →
  (B + A + D = 10) →
  (A + D = 4) →
  (A ≠ B) → (A ≠ C) → (A ≠ D) → (B ≠ C) → (B ≠ D) → (C ≠ D) →
  (A < 10) → (B < 10) → (C < 10) → (D < 10) →
  C = 4 := by
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_problem_l3284_328485


namespace NUMINAMATH_CALUDE_bus_passengers_specific_case_l3284_328424

def passengers (m n : ℕ) : ℕ := m - 12 + n

theorem bus_passengers (m n : ℕ) (h : m ≥ 12) : 
  passengers m n = m - 12 + n :=
by sorry

theorem specific_case : passengers 26 6 = 20 :=
by sorry

end NUMINAMATH_CALUDE_bus_passengers_specific_case_l3284_328424


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l3284_328458

theorem complementary_angles_ratio (x y : ℝ) : 
  x + y = 90 →  -- The angles are complementary (sum to 90°)
  x = 4 * y →   -- The ratio of the angles is 4:1
  y = 18 :=     -- The smaller angle is 18°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l3284_328458


namespace NUMINAMATH_CALUDE_parabola_h_value_l3284_328482

/-- Represents a parabola of the form y = a(x-h)^2 + c -/
structure Parabola where
  a : ℝ
  h : ℝ
  c : ℝ

/-- The y-intercept of a parabola -/
def y_intercept (p : Parabola) : ℝ := p.a * p.h^2 + p.c

/-- Checks if a parabola has two positive integer x-intercepts -/
def has_two_positive_integer_x_intercepts (p : Parabola) : Prop :=
  ∃ x1 x2 : ℤ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ 
    p.a * (x1 - p.h)^2 + p.c = 0 ∧ 
    p.a * (x2 - p.h)^2 + p.c = 0

theorem parabola_h_value 
  (p1 p2 : Parabola)
  (h1 : p1.a = 4)
  (h2 : p2.a = 5)
  (h3 : p1.h = p2.h)
  (h4 : y_intercept p1 = 4027)
  (h5 : y_intercept p2 = 4028)
  (h6 : has_two_positive_integer_x_intercepts p1)
  (h7 : has_two_positive_integer_x_intercepts p2) :
  p1.h = 36 := by
  sorry

end NUMINAMATH_CALUDE_parabola_h_value_l3284_328482
