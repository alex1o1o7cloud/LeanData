import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seventh_power_expansion_sum_of_squares_coefficients_l975_97561

theorem cos_seventh_power_expansion (θ : ℝ) :
  (Real.cos θ) ^ 7 = (35/64) * Real.cos θ + (21/64) * Real.cos (3*θ) + (7/64) * Real.cos (5*θ) + (1/64) * Real.cos (7*θ) :=
by sorry

theorem sum_of_squares_coefficients :
  (35/64)^2 + (21/64)^2 + (7/64)^2 + (1/64)^2 = 1687/4096 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seventh_power_expansion_sum_of_squares_coefficients_l975_97561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_abc_l975_97505

/-- Represents the rate of water flow for a valve in liters per hour -/
def rate (valve : Char) : ℝ := sorry

/-- The volume of the tank in liters -/
def tank_volume : ℝ := 1

/-- Time to fill the tank with all valves open -/
def time_all : ℝ := 1.2

/-- Time to fill the tank with valves A, B, and D open -/
def time_abd : ℝ := 2

/-- Time to fill the tank with valves A, C, and D open -/
def time_acd : ℝ := 1.5

/-- The rate of valve D is half the rate of valve C -/
axiom rate_d_half_c : rate 'D' = (1/2) * rate 'C'

/-- The flow rates of all valves sum to fill the tank in time_all -/
axiom fill_all : rate 'A' + rate 'B' + rate 'C' + rate 'D' = tank_volume / time_all

/-- The flow rates of A, B, D sum to fill the tank in time_abd -/
axiom fill_abd : rate 'A' + rate 'B' + rate 'D' = tank_volume / time_abd

/-- The flow rates of A, C, D sum to fill the tank in time_acd -/
axiom fill_acd : rate 'A' + rate 'C' + rate 'D' = tank_volume / time_acd

/-- The time to fill the tank with valves A, B, and C open is 1.5 hours -/
theorem fill_time_abc : tank_volume / (rate 'A' + rate 'B' + rate 'C') = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_abc_l975_97505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_boarders_correct_l975_97544

/-- The number of new boarders that joined the school -/
def new_boarders : ℕ := 21

/-- The initial number of boarders -/
def initial_boarders : ℕ := 240

/-- The initial ratio of boarders to day students -/
def initial_ratio : Rat := 8 / 17

/-- The final ratio of boarders to day students -/
def final_ratio : Rat := 3 / 7

/-- The theorem stating that the number of new boarders is correct given the conditions -/
theorem new_boarders_correct :
  let initial_day_students : ℚ := (initial_boarders : ℚ) * (17 / 8)
  (initial_boarders + new_boarders : ℚ) / initial_day_students = 3 / 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_boarders_correct_l975_97544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_range_l975_97535

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 4^x - k * 2^x + k + 3

noncomputable def f_derivative (k : ℝ) (x : ℝ) : ℝ := Real.log 4 * 4^x - k * Real.log 2 * 2^x

theorem unique_solution_range (k : ℝ) :
  (∃! x : ℝ, f k x = 0) ↔ k ∈ Set.Ioi (-3) ∪ {6} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_range_l975_97535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arthur_wallet_remaining_l975_97568

noncomputable def initial_amount : ℚ := 200
def spent_fraction : ℚ := 4/5

theorem arthur_wallet_remaining (spent : ℚ) (remaining : ℚ) 
  (h1 : spent = initial_amount * spent_fraction)
  (h2 : remaining = initial_amount - spent) : 
  remaining = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arthur_wallet_remaining_l975_97568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l975_97531

noncomputable def original_expression (x : ℝ) : ℝ :=
  (x^3 + 2*x^2) / (x^2 - 4*x + 4) / ((4*x + 8) / (x - 2)) - 1 / (x - 2)

noncomputable def simplified_expression (x : ℝ) : ℝ :=
  (x + 2) / 4

theorem expression_simplification :
  original_expression (-6) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l975_97531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_minor_axis_zero_l975_97516

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  focus : Point
  semiMajorEndpoint : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the semi-minor axis of an ellipse -/
noncomputable def semiMinorAxis (e : Ellipse) : ℝ :=
  let a := distance e.center e.semiMajorEndpoint
  let c := distance e.center e.focus
  Real.sqrt (a^2 - c^2)

theorem ellipse_semi_minor_axis_zero (e : Ellipse) 
  (h1 : e.center = ⟨2, -4⟩) 
  (h2 : e.focus = ⟨2, -7⟩) 
  (h3 : e.semiMajorEndpoint = ⟨2, -1⟩) : 
  semiMinorAxis e = 0 := by
  sorry

#check ellipse_semi_minor_axis_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_minor_axis_zero_l975_97516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gold_tokens_l975_97595

/-- Represents the number of tokens Alex has --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  gold : ℕ

/-- Represents an exchange booth --/
structure Booth where
  redCost : ℕ
  blueCost : ℕ
  redGain : ℕ
  blueGain : ℕ
  goldGain : ℕ

/-- Defines if an exchange is possible given a token count and a booth --/
def canExchange (tokens : TokenCount) (booth : Booth) : Prop :=
  tokens.red ≥ booth.redCost ∧ tokens.blue ≥ booth.blueCost

/-- Performs an exchange if possible --/
def exchange (tokens : TokenCount) (booth : Booth) : TokenCount :=
  if tokens.red ≥ booth.redCost ∧ tokens.blue ≥ booth.blueCost then
    { red := tokens.red - booth.redCost + booth.redGain,
      blue := tokens.blue - booth.blueCost + booth.blueGain,
      gold := tokens.gold + booth.goldGain }
  else
    tokens

/-- Theorem: The maximum number of gold tokens Alex can obtain is 78 --/
theorem max_gold_tokens : ∃ (finalTokens : TokenCount),
  let initialTokens : TokenCount := ⟨100, 60, 0⟩
  let booth1 : Booth := ⟨3, 0, 0, 2, 1⟩
  let booth2 : Booth := ⟨0, 4, 1, 0, 1⟩
  finalTokens.gold = 78 ∧
  (∀ (t : TokenCount),
    (∃ (n1 n2 : ℕ), t = (exchange (exchange initialTokens booth1) booth2)) →
    t.gold ≤ finalTokens.gold) ∧
  ¬(canExchange finalTokens booth1) ∧
  ¬(canExchange finalTokens booth2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gold_tokens_l975_97595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_additional_spheres_is_two_l975_97503

/-- Represents a truncated cone -/
structure TruncatedCone where
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

/-- Represents the configuration of spheres in the truncated cone -/
structure SphereConfiguration where
  cone : TruncatedCone
  O₁ : Sphere
  O₂ : Sphere

/-- Predicate to check if a sphere configuration is valid -/
def is_valid_configuration (config : SphereConfiguration) : Prop :=
  config.cone.height = 8 ∧
  config.O₁.radius = 2 ∧
  config.O₂.radius = 3 ∧
  -- Additional conditions for tangency would be defined here
  True -- Placeholder for additional conditions

/-- The maximum number of additional spheres that can fit -/
def max_additional_spheres (config : SphereConfiguration) : ℕ := 2

/-- Theorem stating the maximum number of additional spheres -/
theorem max_additional_spheres_is_two (config : SphereConfiguration) 
  (h : is_valid_configuration config) :
  max_additional_spheres config = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_additional_spheres_is_two_l975_97503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l975_97546

noncomputable def f (x : ℝ) : ℝ := -8/9 * x^2 + 32/9 * x + 40/9

theorem quadratic_function_properties :
  f (-1) = 0 ∧ f 5 = 0 ∧ f 2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l975_97546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_powers_of_three_l975_97596

theorem two_digit_powers_of_three : 
  (Finset.filter (λ n : ℕ => 10 ≤ 3^n ∧ 3^n ≤ 99) (Finset.range 5)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_powers_of_three_l975_97596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_rate_approx_l975_97598

/-- The rate of markup given profit and expense percentages -/
noncomputable def rate_of_markup (selling_price profit_percent expense_percent : ℝ) : ℝ :=
  let cost := selling_price * (1 - profit_percent - expense_percent)
  (selling_price - cost) / cost * 100

/-- Theorem stating that the rate of markup is approximately 53.85% under given conditions -/
theorem markup_rate_approx (ε : ℝ) (h_ε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ 
    |rate_of_markup 10 0.20 0.15 - 53.85| < δ ∧ δ < ε :=
by
  sorry

/-- Compute an approximation of the rate of markup -/
def approx_markup : ℚ :=
  (10 - 10 * (1 - 0.20 - 0.15)) / (10 * (1 - 0.20 - 0.15)) * 100

#eval approx_markup

end NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_rate_approx_l975_97598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_expression_l975_97524

/-- Given a function f : ℝ → ℝ such that f(x+1) = x^2 - x + 1 for all x,
    prove that f(x) = x^2 - 3x + 3 for all x. -/
theorem function_expression (f : ℝ → ℝ) 
    (h : ∀ x, f (x + 1) = x^2 - x + 1) : 
    ∀ x, f x = x^2 - 3*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_expression_l975_97524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_throws_to_return_l975_97534

/-- The number of positions in the circle -/
def n : ℕ := 15

/-- The number of positions to skip in the first phase -/
def skip1 : ℕ := 5

/-- The number of positions to skip in the second phase -/
def skip2 : ℕ := 3

/-- The number of throws in the first phase -/
def throws1 : ℕ := 10

/-- Function to calculate the next position after a throw -/
def nextPos (curr : ℕ) (skip : ℕ) : ℕ :=
  (curr + skip - 1) % n + 1

/-- Function to simulate throws and count total throws -/
def simulateThrows : ℕ :=
  let rec loop (pos : ℕ) (throwCount : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then throwCount
    else if pos = 1 ∧ throwCount > throws1 then throwCount
    else if throwCount ≤ throws1 then
      loop (nextPos pos skip1) (throwCount + 1) (fuel - 1)
    else
      loop (nextPos pos skip2) (throwCount + 1) (fuel - 1)
  loop 1 0 (2 * n) -- Use 2*n as an upper bound for the number of throws

theorem total_throws_to_return : simulateThrows = 8 := by sorry

#eval simulateThrows

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_throws_to_return_l975_97534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sine_cosine_l975_97504

theorem geometric_sequence_sine_cosine (α β γ : ℝ) : 
  (β = 2 * α ∧ γ = 2 * β) →  -- Geometric sequence condition
  ((Real.sin β) / (Real.sin α) = (Real.sin γ) / (Real.sin β)) →  -- Sine geometric sequence condition
  Real.cos α = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sine_cosine_l975_97504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_x_values_l975_97532

noncomputable def list (x : ℝ) : List ℝ := [4, 9, x, 4, 9, 4, 11, x]

noncomputable def mean (x : ℝ) : ℝ := (41 + 2*x) / 8

def mode : ℝ := 4

noncomputable def median (x : ℝ) : ℝ :=
  if x ≤ 9 then 4.5
  else if x < 11 then 9
  else x

def is_geometric_progression (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ b^2 = a*c

theorem sum_of_possible_x_values :
  ∃ (x₁ x₂ : ℝ),
    (∀ x : ℝ, is_geometric_progression mode (median x) (mean x) →
      (x = x₁ ∨ x = x₂)) ∧
    x₁ + x₂ = 15.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_x_values_l975_97532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_additives_percentage_l975_97556

/-- Represents the budget allocation for Megatech Corporation --/
structure BudgetAllocation where
  microphotonics : ℝ
  home_electronics : ℝ
  genetically_modified_microorganisms : ℝ
  industrial_lubricants : ℝ
  basic_astrophysics : ℝ
  food_additives : ℝ

/-- The total budget percentage should equal 100% --/
def total_budget_constraint (b : BudgetAllocation) : Prop :=
  b.microphotonics + b.home_electronics + b.genetically_modified_microorganisms +
  b.industrial_lubricants + b.basic_astrophysics + b.food_additives = 100

/-- The given percentages for specific categories --/
def given_percentages (b : BudgetAllocation) : Prop :=
  b.microphotonics = 13 ∧
  b.home_electronics = 24 ∧
  b.genetically_modified_microorganisms = 29 ∧
  b.industrial_lubricants = 8

/-- The relation between degrees and percentage for basic astrophysics --/
def basic_astrophysics_constraint (b : BudgetAllocation) : Prop :=
  b.basic_astrophysics / 100 = 39.6 / 360

/-- The main theorem: proving the percentage allocated to food additives --/
theorem food_additives_percentage (b : BudgetAllocation) 
  (h1 : total_budget_constraint b)
  (h2 : given_percentages b)
  (h3 : basic_astrophysics_constraint b) :
  b.food_additives = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_additives_percentage_l975_97556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l975_97523

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2^(1/(x-1))

-- Define the domain
def domain (x : ℝ) : Prop := x < 1 ∨ x > 1

-- State the theorem
theorem f_strictly_decreasing :
  ∀ x₁ x₂, domain x₁ ∧ domain x₂ ∧ x₁ < x₂ → f x₁ > f x₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l975_97523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_of_women_l975_97574

theorem average_age_of_women (n : ℕ) (a : ℝ) (age_increase : ℝ) (man1_age man2_age : ℝ) :
  n = 9 →
  age_increase = 4 →
  man1_age = 36 →
  man2_age = 32 →
  (n * age_increase + man1_age + man2_age) / 2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_of_women_l975_97574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_is_ten_percent_l975_97567

/-- Represents the tax system of Country X -/
structure TaxSystem where
  base_rate : ℚ  -- Tax rate for the first $40,000
  excess_rate : ℚ := 1/5  -- Tax rate for income over $40,000
  base_income : ℚ := 40000  -- The threshold for the base rate

/-- Calculates the total tax for a given income -/
def calculate_tax (sys : TaxSystem) (income : ℚ) : ℚ :=
  let base_tax := sys.base_rate * sys.base_income
  let excess_income := max (income - sys.base_income) 0
  let excess_tax := sys.excess_rate * excess_income
  base_tax + excess_tax

/-- Theorem stating that the base tax rate is 10% -/
theorem tax_rate_is_ten_percent (sys : TaxSystem) :
  calculate_tax sys 60000 = 8000 → sys.base_rate = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_is_ten_percent_l975_97567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_range_l975_97513

-- Define the function f(x) = (1/3)x³ - x²
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := x^2 - 2*x

-- Define the range of the slope angle
def slope_angle_range : Set ℝ := 
  {α | 0 ≤ α ∧ α < Real.pi ∧ (0 ≤ α ∧ α < Real.pi/2 ∨ 3*Real.pi/4 ≤ α ∧ α < Real.pi)}

-- Theorem statement
theorem tangent_slope_angle_range :
  ∀ x : ℝ, ∃ α ∈ slope_angle_range, Real.tan α = f_derivative x := by
  sorry

#check tangent_slope_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_range_l975_97513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt3_div_2_l975_97593

open Real

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/3) -/
noncomputable def circle_radius (θ : ℝ) : ℝ :=
  sqrt ((sin (π/3) * cos θ)^2 + (sin (π/3) * sin θ)^2)

theorem circle_radius_is_sqrt3_div_2 :
  ∀ θ : ℝ, circle_radius θ = sqrt 3 / 2 := by
  intro θ
  unfold circle_radius
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt3_div_2_l975_97593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_midpoint_theorem_l975_97565

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: In an isosceles triangle ABC where AB = BC, with D being the midpoint of both BC and AC,
    and E on BC such that BE = 9 units, if AC = 16 units, then BD = 8 units -/
theorem isosceles_triangle_midpoint_theorem (t : Triangle) (D E : Point) :
  distance t.A t.B = distance t.B t.C →  -- AB = BC
  D = ⟨(t.B.x + t.C.x) / 2, (t.B.y + t.C.y) / 2⟩ →  -- D is midpoint of BC
  D = ⟨(t.A.x + t.C.x) / 2, (t.A.y + t.C.y) / 2⟩ →  -- D is midpoint of AC
  E.x = t.B.x + 9 * (t.C.x - t.B.x) / distance t.B t.C →  -- E is on BC and BE = 9
  E.y = t.B.y + 9 * (t.C.y - t.B.y) / distance t.B t.C →
  distance t.A t.C = 16 →  -- AC = 16
  distance t.B D = 8 :=  -- BD = 8
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_midpoint_theorem_l975_97565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l975_97553

-- Define the points
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, -4)
def C : ℝ × ℝ := (9, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem triangle_perimeter :
  distance A B + distance B C + distance C A = 7 + Real.sqrt 74 + Real.sqrt 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l975_97553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contact_loss_time_l975_97572

/-- The time it takes for two teams to lose radio contact -/
noncomputable def time_to_lose_contact (speed1 speed2 radio_range : ℝ) : ℝ :=
  radio_range / (speed1 + speed2)

/-- Theorem: The time to lose contact for the given scenario is 2.5 hours -/
theorem contact_loss_time :
  time_to_lose_contact 20 30 125 = 2.5 := by
  -- Unfold the definition of time_to_lose_contact
  unfold time_to_lose_contact
  -- Simplify the expression
  simp
  -- Check that 125 / (20 + 30) = 2.5
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_contact_loss_time_l975_97572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_terms_bound_product_fewer_terms_l975_97597

/-- A polynomial with coefficients in ring R -/
def MyPolynomial (R : Type*) [Ring R] := List (R × ℕ)

/-- The number of terms in a polynomial -/
def num_terms {R : Type*} [Ring R] (p : MyPolynomial R) : ℕ := p.length

/-- The product of two polynomials -/
noncomputable def poly_mul {R : Type*} [CommRing R] (p q : MyPolynomial R) : MyPolynomial R :=
  sorry -- Implementation of polynomial multiplication

theorem product_terms_bound {R : Type*} [CommRing R] (p q : MyPolynomial R) :
  ∃ (r : MyPolynomial R), r = poly_mul p q ∧ 
    num_terms r ≤ (num_terms p) * (num_terms q) := by
  sorry

/-- The main theorem: product can have fewer terms than m*n -/
theorem product_fewer_terms {R : Type*} [CommRing R] :
  ∃ (p q : MyPolynomial R), 
    num_terms (poly_mul p q) < (num_terms p) * (num_terms q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_terms_bound_product_fewer_terms_l975_97597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_m_l975_97518

/-- Given a set M of integers from 1 to m, if M has 4 subsets, then m equals 2 -/
theorem subsets_of_m (m : ℕ) : 
  let M := Finset.range m
  Finset.powerset M |>.card = 4 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_m_l975_97518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_triangle_area_l975_97570

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x - Real.pi/6)

-- Theorem for the range of f(x)
theorem f_range : 
  ∀ x ∈ Set.Icc 0 (Real.pi/2), f x ∈ Set.Icc (-1/2) (1/4) := by sorry

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Theorem for the area of the triangle
theorem triangle_area (t : Triangle) 
  (h1 : f t.A = 1/4)
  (h2 : t.a = Real.sqrt 3)
  (h3 : Real.sin t.B = 2 * Real.sin t.C) :
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_triangle_area_l975_97570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defined_necessary_not_sufficient_for_continuity_l975_97566

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be defined at a point
def DefinedAt (f : RealFunction) (x : ℝ) : Prop := ∃ y : ℝ, f x = y

-- We don't need to redefine ContinuousAt as it's already in Mathlib
-- Instead, we'll use the existing definition

-- Theorem statement
theorem defined_necessary_not_sufficient_for_continuity :
  (∀ f : RealFunction, ∀ x : ℝ, ContinuousAt f x → DefinedAt f x) ∧
  (∃ f : RealFunction, ∃ x : ℝ, DefinedAt f x ∧ ¬ContinuousAt f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_defined_necessary_not_sufficient_for_continuity_l975_97566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l975_97551

-- Problem 1
theorem problem_1 : ((-27) ^ (1/3 : ℝ)) + abs (Real.sqrt 3 - 2) - Real.sqrt (9/4) = -5/2 - Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 (a b : ℝ) (h1 : Real.sqrt (2*a - 1) = 3) (h2 : (3*a + 6*b) ^ (1/3 : ℝ) = 3) :
  Real.sqrt (a + b) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l975_97551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_theorem_l975_97509

/-- The number of days it takes for two workers to complete a job together,
    given the individual completion times for each worker. -/
noncomputable def combined_work_time (x_time y_time : ℝ) : ℝ :=
  1 / (1 / x_time + 1 / y_time)

/-- Theorem stating that if worker x completes a job in 15 days and worker y
    completes the same job in 45 days, then together they will complete
    the job in 11.25 days. -/
theorem combined_work_theorem :
  combined_work_time 15 45 = 11.25 := by
  -- Expand the definition of combined_work_time
  unfold combined_work_time
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_theorem_l975_97509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_50_between_consecutive_integers_l975_97500

theorem log_50_between_consecutive_integers :
  ∃ (c d : ℤ), c + 1 = d ∧ c < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < d ∧ c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_50_between_consecutive_integers_l975_97500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_squared_plus_z_squared_l975_97564

theorem y_squared_plus_z_squared (x y z a b c : ℝ) 
  (h1 : x * y = a) 
  (h2 : x + y = b) 
  (h3 : x * z = c) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  y^2 + z^2 = (2*b^2 - 4*a + 4*c^2) / ((b + Real.sqrt (b^2 - 4*a))^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_squared_plus_z_squared_l975_97564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_cells_sum_l975_97502

/-- Represents a 2x3 grid with digits 1, 2, and 3 --/
def Grid := Fin 2 → Fin 3 → Fin 3

/-- A valid grid satisfies the row and column constraints --/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i : Fin 2, Function.Injective (g i)) ∧
  (∀ j : Fin 3, Function.Injective (λ i ↦ g i j))

/-- Theorem stating that the sum of any two cells in a row of a valid grid is 4 --/
theorem shaded_cells_sum (g : Grid) (h : is_valid_grid g) :
  ∀ i : Fin 2, ∀ j k : Fin 3, j ≠ k → (g i j).val + (g i k).val = 4 :=
by
  sorry

#check shaded_cells_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_cells_sum_l975_97502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_last_four_digits_l975_97542

theorem same_last_four_digits : ∃ (N : ℕ), 
  N > 0 ∧ 
  N % 8000 = (N^2 : ℕ) % 8000 ∧ 
  N % 10000 ≥ 1000 ∧
  N = 3625 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_last_four_digits_l975_97542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l975_97549

/-- Proves that given two loans with the same interest rate, the total interest over a specific time period results in the correct annual interest rate. -/
theorem interest_rate_calculation (loan1 loan2 total_interest time_period : ℝ) 
  (h1 : loan1 = 1000)
  (h2 : loan2 = 1400)
  (h3 : total_interest = 350)
  (h4 : time_period = 4.861111111111111) :
  ∃ r : ℝ, (r ≥ 0.029 ∧ r ≤ 0.031) ∧ r * time_period * (loan1 + loan2) = total_interest := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l975_97549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l975_97548

/-- The circle centered at (-1, 2) with radius 1 -/
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

/-- The line 3x - 4y - 9 = 0 -/
def my_line (x y : ℝ) : Prop := 3*x - 4*y - 9 = 0

/-- The shortest distance from a point on the circle to the line is 3 -/
theorem shortest_distance_circle_to_line :
  ∀ (x y : ℝ), my_circle x y →
  ∃ (d : ℝ), d = 3 ∧ 
  (∀ (x' y' : ℝ), my_line x' y' → 
    d ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l975_97548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_needed_to_fill_glasses_ten_glasses_water_needed_l975_97517

theorem water_needed_to_fill_glasses (n : ℕ) (capacity : ℚ) (current_fill : ℚ) :
  n > 0 →
  capacity > 0 →
  0 < current_fill ∧ current_fill < 1 →
  (n * capacity) - (n * (capacity * current_fill)) = n * capacity * (1 - current_fill) :=
by sorry

theorem ten_glasses_water_needed :
  (10 : ℚ) * 6 * (1 - 4/5) = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_needed_to_fill_glasses_ten_glasses_water_needed_l975_97517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_triangles_theorem_l975_97550

/-- Given a triangle ABC with external triangles BPC, CQA, and ARB, prove that ∠PRQ = 90° and QR = PR -/
theorem external_triangles_theorem (A B C P Q R : ℂ) : 
  (Complex.arg (P - B) - Complex.arg (C - B) = 45 * π / 180) →
  (Complex.arg (Q - A) - Complex.arg (C - A) = 45 * π / 180) →
  (Complex.arg (C - B) - Complex.arg (P - B) = 30 * π / 180) →
  (Complex.arg (C - A) - Complex.arg (Q - A) = 30 * π / 180) →
  (Complex.arg (R - B) - Complex.arg (A - B) = 15 * π / 180) →
  (Complex.arg (R - A) - Complex.arg (B - A) = 15 * π / 180) →
  (Complex.arg (Q - R) - Complex.arg (P - R) = 90 * π / 180) ∧ (Complex.abs (Q - R) = Complex.abs (P - R)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_triangles_theorem_l975_97550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_intersection_l975_97590

-- Define a parallelogram type
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ

-- Theorem statement
theorem parallelogram_diagonal_intersection
  (p : Parallelogram)
  (h1 : p.v1 = (2, -3))
  (h2 : p.v2 = (14, 9)) :
  ((p.v1.1 + p.v2.1) / 2, (p.v1.2 + p.v2.2) / 2) = (8, 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_intersection_l975_97590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_painting_time_equation_l975_97540

/-- Represents the time taken to paint a room -/
structure PaintingTime where
  hannah : ℝ  -- Time for Hannah to paint the room alone
  sarah : ℝ   -- Time for Sarah to paint the room alone
  temp : ℝ    -- Current temperature
  breakTime : ℝ    -- Duration of the break
  decreaseRate : ℝ -- Rate decrease percentage above 25°C

/-- The equation for the total time to paint the room -/
def totalTimeEquation (pt : PaintingTime) (t : ℝ) : Prop :=
  let combinedRate := (1 / pt.hannah + 1 / pt.sarah) * (1 - pt.decreaseRate)
  combinedRate * (t - pt.breakTime) = 1

/-- Theorem stating that the given equation correctly expresses the total painting time -/
theorem correct_painting_time_equation (pt : PaintingTime) (t : ℝ) : 
  pt.hannah = 6 → 
  pt.sarah = 8 → 
  pt.temp > 25 →
  pt.decreaseRate = 0.1 →
  pt.breakTime = 1.5 →
  totalTimeEquation pt t ↔ (21/80) * (t - 1.5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_painting_time_equation_l975_97540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_equation_l975_97538

/-- Given an ellipse C and a line l, prove the trajectory equation of the midpoint of AB --/
theorem midpoint_trajectory_equation (C : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) 
  (h_C : C = {p : ℝ × ℝ | p.1^2 + 2*p.2^2 = 4})
  (h_l : ∃ k b, l = {p : ℝ × ℝ | p.2 = k*p.1 + b ∧ 6 = k*4 + b})
  (h_intersect : ∃ A B, A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l ∧ A ≠ B) :
  ∃ M : Set (ℝ × ℝ), M = {p : ℝ × ℝ | (p.1-2)^2/22 + (p.2-3)^2/11 = 1 ∧ 
    ∃ A B, A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l ∧ A ≠ B ∧ 
    p.1 = (A.1 + B.1)/2 ∧ p.2 = (A.2 + B.2)/2} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_equation_l975_97538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_intersection_l975_97533

-- Define the point A
def A : ℝ × ℝ := (1, 7)

-- Define the two lines
def line1 (x y : ℝ) : Prop := x - y - 1 = 0
def line2 (x y : ℝ) : Prop := x + 3*y - 12 = 0

-- Define the intersection point B
noncomputable def B : ℝ × ℝ := (15/4, 11/4)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem distance_A_to_intersection :
  line1 B.1 B.2 ∧ line2 B.1 B.2 ∧ distance A B = Real.sqrt 410 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_intersection_l975_97533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_fraction_l975_97514

theorem matrix_sum_fraction (x y z : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![x, x+y, x+z; x+y, y, y+z; x+z, y+z, z]
  ¬(IsUnit (Matrix.det M)) → 
  (x / (x+y+z) + y / (x+y+z) + z / (x+y+z) = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_fraction_l975_97514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_verify_median_coin_l975_97511

/-- A device that can find the median among 2n+1 coins -/
def MedianDevice (n : ℕ) := (Fin (2*n + 1) → ℕ) → Fin (2*n + 1)

/-- The main theorem stating that it's possible to verify the median coin using the device at most n+2 times -/
theorem verify_median_coin (n : ℕ) (h : n > 1) :
  ∃ (verify : (Fin (4*n + 1) → ℕ) → MedianDevice n → Fin (4*n + 1) → Bool),
    ∀ (weights : Fin (4*n + 1) → ℕ) (device : MedianDevice n) (claimed_median : Fin (4*n + 1)),
      (∀ i j, i ≠ j → weights i ≠ weights j) →
      (verify weights device claimed_median = true ↔ 
        ∃ (S : Finset (Fin (4*n + 1))), S.card = 2*n ∧ 
          (∀ i ∈ S, weights i < weights claimed_median) ∧
          (∀ i ∈ (Finset.univ \ S).erase claimed_median, weights claimed_median < weights i)) ∧
      (∃ num_uses : ℕ, num_uses ≤ n + 2 ∧ 
        (verify weights device claimed_median = true ∨ 
         verify weights device claimed_median = false)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_verify_median_coin_l975_97511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_chips_probability_l975_97555

/-- The probability of drawing all red chips before all green chips -/
theorem red_chips_probability (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) : 
  total_chips = red_chips + green_chips →
  red_chips = 4 →
  green_chips = 3 →
  (Nat.choose total_chips green_chips : ℚ) ≠ 0 →
  (Nat.choose (total_chips - 1) (green_chips - 1) : ℚ) / (Nat.choose total_chips green_chips : ℚ) = 3/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_chips_probability_l975_97555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_increasing_intervals_not_always_increasing_l975_97536

/-- A function for which the union of multiple increasing intervals is not an increasing interval -/
noncomputable def counterexample_function : ℝ → ℝ := sorry

/-- The first increasing interval of the counterexample function -/
def increasing_interval1 : Set ℝ := sorry

/-- The second increasing interval of the counterexample function -/
def increasing_interval2 : Set ℝ := sorry

/-- The union of the two increasing intervals -/
def union_intervals : Set ℝ := increasing_interval1 ∪ increasing_interval2

theorem union_of_increasing_intervals_not_always_increasing :
  (∀ x y, x ∈ increasing_interval1 → y ∈ increasing_interval1 → x < y → counterexample_function x < counterexample_function y) ∧
  (∀ x y, x ∈ increasing_interval2 → y ∈ increasing_interval2 → x < y → counterexample_function x < counterexample_function y) ∧
  ¬(∀ x y, x ∈ union_intervals → y ∈ union_intervals → x < y → counterexample_function x < counterexample_function y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_increasing_intervals_not_always_increasing_l975_97536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_owners_without_motorcycle_l975_97575

theorem car_owners_without_motorcycle (total : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ)
  (h1 : total = 500)
  (h2 : car_owners = 450)
  (h3 : motorcycle_owners = 80)
  (h4 : car_owners + motorcycle_owners - total ≤ car_owners)
  : car_owners - (car_owners + motorcycle_owners - total) = 420 := by
  sorry

#check car_owners_without_motorcycle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_owners_without_motorcycle_l975_97575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_a_time_l975_97519

/-- The time it takes for worker a to complete the work alone -/
noncomputable def time_a : ℝ := 4.8

/-- The time it takes for worker b to complete the work alone -/
def time_b : ℝ := 12

/-- The total time taken to complete the work with alternating shifts -/
def total_time : ℕ := 6

/-- The fraction of work completed by worker a in one hour -/
noncomputable def work_rate_a : ℝ := 1 / time_a

/-- The fraction of work completed by worker b in one hour -/
noncomputable def work_rate_b : ℝ := 1 / time_b

/-- Theorem stating that the time for worker a to complete the work alone is 4.8 hours -/
theorem worker_a_time : 
  (4 * work_rate_a + 2 * work_rate_b = 1) → time_a = 4.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_a_time_l975_97519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_selling_price_l975_97525

/-- The original purchase price of the product -/
def P : ℝ := 900

/-- The original selling price -/
def SP₁ : ℝ := 1.1 * P

/-- The new purchase price if bought for 10% less -/
def P_new : ℝ := 0.9 * P

/-- The new selling price with 30% profit on the new purchase price -/
def SP₂ : ℝ := 1.17 * P

/-- The difference between the new and original selling prices is $63 -/
axiom price_difference : SP₂ - SP₁ = 63

/-- The original selling price is $990 -/
theorem original_selling_price : SP₁ = 990 := by
  -- Unfold the definitions
  unfold SP₁ P
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_selling_price_l975_97525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_minus_cot_l975_97530

theorem cos_double_minus_cot (α : ℝ) : 
  0 < α ∧ α < Real.pi / 2 → 
  Real.cos α = 2 * Real.sqrt 5 / 5 →
  Real.cos (2 * α) - Real.cos α / Real.sin α = -7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_minus_cot_l975_97530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_repeating_decimal_to_thousandth_l975_97506

/-- The repeating decimal 67.326326... -/
noncomputable def repeating_decimal : ℝ := 67 + 326 / 999

/-- Rounding a real number to the nearest thousandth -/
noncomputable def round_to_thousandth (x : ℝ) : ℝ := 
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

/-- Theorem stating that rounding the repeating decimal to the nearest thousandth equals 67.326 -/
theorem round_repeating_decimal_to_thousandth : 
  round_to_thousandth repeating_decimal = 67.326 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_repeating_decimal_to_thousandth_l975_97506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l975_97557

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {2, 3, a^2 + 2*a - 3}

-- Define set A
def A (a : ℝ) : Set ℝ := {|2*a - 1|, 2}

-- Theorem statement
theorem find_a : ∃ (a : ℝ), 
  (U a).Finite ∧ 
  (A a).Finite ∧
  ((U a) \ (A a)) = {5} → 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l975_97557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l975_97583

open Real

theorem inequality_condition (l : ℝ) : 
  (l > 0) → 
  (∀ x > 0, exp (l * x) - log x / l ≥ 0) ↔ 
  (l ≥ 1 / exp 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l975_97583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_when_m_3_b_subset_a_iff_l975_97547

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 8*x + 12 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 5 - 2*m ≤ x ∧ x ≤ m + 1}

-- Theorem for part 1
theorem intersection_and_union_when_m_3 :
  (A ∩ B 3 = Set.Icc 2 4) ∧ (A ∪ B 3 = Set.Icc (-1) 6) := by sorry

-- Theorem for part 2
theorem b_subset_a_iff (m : ℝ) :
  B m ⊆ A ↔ m ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_when_m_3_b_subset_a_iff_l975_97547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_finger_value_l975_97573

def g : ℕ → ℕ
  | 0 => 8
  | 1 => 7
  | 2 => 6
  | 3 => 5
  | 4 => 4
  | 5 => 3
  | 6 => 2
  | 7 => 1
  | 8 => 0
  | _ => 0  -- For completeness, though not used in the problem

def apply_g_n_times : ℕ → ℕ → ℕ
  | 0, x => x
  | (n+1), x => apply_g_n_times n (g x)

theorem eighth_finger_value :
  apply_g_n_times 7 2 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_finger_value_l975_97573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_coprime_characterization_l975_97581

def isPrimePower (n : ℕ) : Prop :=
  ∃ p k, Nat.Prime p ∧ n = p^k

def allEntriesCoprime (n k : ℕ) : Prop :=
  ∀ i, i ≤ k → Nat.Coprime n (Nat.choose k i)

theorem pascal_triangle_coprime_characterization (n : ℕ) :
  (∃ f : ℕ → ℕ, StrictMono f ∧ ∀ k, allEntriesCoprime n (f k)) ↔ isPrimePower n :=
sorry

#check pascal_triangle_coprime_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_coprime_characterization_l975_97581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_ngon_vertex_sequence_l975_97563

/-- A regular n-gon with n odd -/
structure RegularNGon (n : ℕ) :=
  (odd : Odd n)

/-- The sequence of vertices in a regular n-gon -/
def VertexSequence (n : ℕ) := ℕ → Fin n

/-- The property that each vertex occurs in the sequence -/
def EachVertexOccurs (n : ℕ) (seq : VertexSequence n) : Prop :=
  ∀ v : Fin n, ∃ k : ℕ, seq k = v

/-- The sequence follows the given rule -/
def FollowsRule (n : ℕ) (seq : VertexSequence n) : Prop :=
  ∀ k : ℕ, k ≥ 3 → 
    seq k = ⟨(2 * (seq (k-1)).val + (seq (k-2)).val) % n, by sorry⟩

/-- The main theorem -/
theorem regular_ngon_vertex_sequence (n : ℕ) (ngon : RegularNGon n) :
  (∃ seq : VertexSequence n, FollowsRule n seq ∧ EachVertexOccurs n seq) ↔ 
  (∃ k : ℕ, n = 3^k) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_ngon_vertex_sequence_l975_97563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l975_97515

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1))

-- State the theorem
theorem f_derivative_at_2 : 
  deriv f 2 = 2/5 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l975_97515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_function_unique_l975_97591

def PositiveIntegers : Type := {n : ℕ // n > 0}

theorem zero_function_unique
  (f : PositiveIntegers → ℝ)
  (h : ∀ (m n : PositiveIntegers), (n.val : ℕ) ≥ (m.val : ℕ) → 
       f ⟨n.val + m.val, sorry⟩ + f ⟨n.val - m.val, sorry⟩ = f ⟨3 * n.val, sorry⟩) :
  ∀ (n : PositiveIntegers), f n = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_function_unique_l975_97591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_traces_different_circle_l975_97599

/-- A triangle with vertex C moving on a circular path -/
structure MovingTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- The path traced by a point as the triangle's vertex C moves -/
inductive PathType
  | Circle
  | DifferentCircle
  | Ellipse
  | Parabola
  | StraightLine

/-- The centroid of a triangle -/
def centroid (t : MovingTriangle) : ℝ × ℝ := sorry

/-- The path traced by the centroid as C moves -/
def centroidPath (t : MovingTriangle) : PathType := sorry

theorem centroid_traces_different_circle (t : MovingTriangle) 
  (h1 : (t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2 = 100) -- AB length is 10
  (h2 : t.center = ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2)) -- center is midpoint of AB
  (h3 : t.radius = 5) -- radius of C's path is 5
  : centroidPath t = PathType.DifferentCircle := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_traces_different_circle_l975_97599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_a_value_l975_97526

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that a = 4 under the given conditions. -/
theorem triangle_side_a_value (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- Positive angles
  A + B + C = π →  -- Angle sum in a triangle
  Real.sin B = (1/4) * Real.sin A + Real.sin C →  -- Given condition
  2 * b = 3 * c →  -- Given condition
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 15) / 4 →  -- Area condition
  a = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_a_value_l975_97526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_purchase_cost_l975_97558

theorem candy_purchase_cost
  (price1 : ℝ) (price2 : ℝ) (extra_amount : ℝ) (price_ratio : ℝ) :
  price1 = 1.80 →
  price2 = 1.50 →
  extra_amount = 0.5 →
  price_ratio = 1.5 →
  ∃ (x y : ℝ),
    y = x + extra_amount ∧
    price1 * y = price_ratio * (price2 * x) ∧
    price1 * y + price2 * x = 7.50 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_purchase_cost_l975_97558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l975_97510

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (2*Real.pi - α) * Real.cos (Real.pi + α) * Real.cos (Real.pi/2 + α) * Real.cos (Real.pi/2 - α)) /
  (Real.cos (Real.pi - α) * Real.sin (3*Real.pi - α) * Real.sin (-α) * Real.sin (Real.pi/2 + α))

theorem sin_cos_identity (α : ℝ) :
  f α = -2 → Real.sin α^2 - Real.sin α * Real.cos α - 2 * Real.cos α^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l975_97510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l975_97578

theorem trigonometric_equation_solution :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 360 ∧
  Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x) ∧
  x = 270 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l975_97578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_within_radius_of_lattice_point_l975_97594

-- Define the probability
noncomputable def probability : ℝ := 1/6

-- Define the radius as a function of the probability
noncomputable def radius (p : ℝ) : ℝ := Real.sqrt (1 / (6 * Real.pi))

-- Theorem statement
theorem probability_within_radius_of_lattice_point :
  probability = (Real.pi * (radius probability)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_within_radius_of_lattice_point_l975_97594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_composite_l975_97508

theorem sum_of_squares_composite (a b c d n : ℕ) (h_composite : ¬ Nat.Prime n) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_n_eq_ab : n = a * b) (h_n_eq_cd : n = c * d) : 
  ¬ Nat.Prime (a^2 + b^2 + c^2 + d^2) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_composite_l975_97508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_triangle_to_square_l975_97588

/-- Predicate to check if four points form a square -/
def is_square (A B C D : ℝ × ℝ) : Prop := sorry

/-- Helper function to calculate the area of a triangle given three points -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Helper function to calculate the area of a square given four points -/
def area_square (A B C D : ℝ × ℝ) : ℝ := sorry

/-- Given a square ABCD, point M on AB, and point N on BC, prove that the ratio of the area of triangle AMN to the area of square ABCD is 1/9 -/
theorem area_ratio_triangle_to_square (A B C D M N : ℝ × ℝ) : 
  is_square A B C D →
  M = (2/3 • A + 1/3 • B : ℝ × ℝ) →
  N = (1/3 • B + 2/3 • C : ℝ × ℝ) →
  area_triangle A M N / area_square A B C D = 1/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_triangle_to_square_l975_97588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shell_trajectory_domain_l975_97587

/-- The height function of a shell's trajectory -/
def h (t : ℝ) : ℝ := 130 * t - 5 * t^2

/-- The time it takes for the shell to hit the ground -/
def impact_time : ℝ := 26

/-- The initial height of the shell -/
def initial_height : ℝ := 845

/-- The domain of the height function -/
def height_domain : Set ℝ := Set.Icc 0 impact_time

theorem shell_trajectory_domain :
  ∀ t : ℝ, t ∈ height_domain ↔ ∃ y : ℝ, 0 ≤ y ∧ y ≤ initial_height ∧ y = h t :=
sorry

#check shell_trajectory_domain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shell_trajectory_domain_l975_97587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_cost_minimized_l975_97586

/-- The transportation cost function -/
noncomputable def f (c x : ℝ) : ℝ := c * (Real.sqrt (1089 + x^2) + 183/2 - x/2)

/-- Theorem stating that the transportation cost is minimized at x = 11√3 -/
theorem transportation_cost_minimized :
  ∃ (c : ℝ), c > 0 ∧
  (∀ (x : ℝ), f c (11 * Real.sqrt 3) ≤ f c x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_cost_minimized_l975_97586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_intersection_point_l975_97569

/-- Two circles in a plane -/
structure TwoCircles where
  γ₁ : Set (Real × Real)
  γ₂ : Set (Real × Real)
  O₁ : Real × Real
  O₂ : Real × Real
  h₁ : ∃ r₁ : ℝ, ∀ p : ℝ × ℝ, p ∈ γ₁ ↔ (p.1 - O₁.1)^2 + (p.2 - O₁.2)^2 = r₁^2
  h₂ : ∃ r₂ : ℝ, ∀ p : ℝ × ℝ, p ∈ γ₂ ↔ (p.1 - O₂.1)^2 + (p.2 - O₂.2)^2 = r₂^2

/-- Predicate to check if two circles touch at a point -/
def TouchesAt (γ₁ γ₂ : Set (Real × Real)) (p : Real × Real) : Prop :=
  p ∈ γ₁ ∧ p ∈ γ₂ ∧ ∃ ε > 0, ∀ q ∈ γ₁, q ≠ p → ‖q - p‖ ≥ ε

/-- Configuration of circles and points -/
structure CircleConfiguration (tc : TwoCircles) where
  K : Real × Real
  A : Real × Real
  B : Real × Real
  γ₃ : Set (Real × Real)
  γ₄ : Set (Real × Real)
  h_K_on_γ₂ : K ∈ tc.γ₂
  h_γ₃_touches_γ₂ : TouchesAt γ₃ tc.γ₂ K
  h_γ₃_touches_γ₁ : TouchesAt γ₃ tc.γ₁ A
  h_γ₄_touches_γ₂ : TouchesAt γ₄ tc.γ₂ K
  h_γ₄_touches_γ₁ : TouchesAt γ₄ tc.γ₁ B

/-- Predicate to check if a point lies on the line through two other points -/
def LineThroughPoints (A B P : Real × Real) : Prop :=
  ∃ t : Real, P = (1 - t) • A + t • B

/-- The theorem to be proved -/
theorem fixed_intersection_point (tc : TwoCircles) :
  ∃ P : Real × Real, ∀ (config : CircleConfiguration tc),
    LineThroughPoints config.A config.B P :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_intersection_point_l975_97569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_5_equals_23_3_l975_97529

noncomputable def f (x : ℝ) : ℝ := 5 / (4 - x)

noncomputable def f_inv (x : ℝ) : ℝ := 4 - 5 / x

noncomputable def g (x : ℝ) : ℝ := 2 / (f_inv x) + 7

theorem g_5_equals_23_3 : g 5 = 23 / 3 := by
  -- Expand the definition of g
  unfold g
  -- Expand the definition of f_inv
  unfold f_inv
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_5_equals_23_3_l975_97529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_error_approx_l975_97520

/-- The true diameter of the circle in centimeters -/
noncomputable def true_diameter : ℝ := 30

/-- The maximum error in diameter measurement as a percentage -/
noncomputable def max_error_percentage : ℝ := 30

/-- The maximum measured diameter -/
noncomputable def max_measured_diameter : ℝ := true_diameter * (1 + max_error_percentage / 100)

/-- The true area of the circle -/
noncomputable def true_area : ℝ := Real.pi * (true_diameter / 2) ^ 2

/-- The maximum computed area based on the maximum measured diameter -/
noncomputable def max_computed_area : ℝ := Real.pi * (max_measured_diameter / 2) ^ 2

/-- The maximum percent error in the computed area -/
noncomputable def max_area_error_percentage : ℝ := (max_computed_area - true_area) / true_area * 100

/-- Theorem stating that the maximum percent error in the computed area is approximately 68.9% -/
theorem max_area_error_approx :
  abs (max_area_error_percentage - 68.9) < 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_error_approx_l975_97520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_af_b_range_l975_97512

noncomputable def f (x : ℝ) : ℝ := if x < 0 then x + 2 else 2^x

theorem af_b_range (a b : ℝ) (ha : a < 0) (hb : 0 ≤ b) (hf : f a = f b) :
  ∃ y, y ∈ Set.Icc (-1 : ℝ) 0 ∧ y = a * f b ∧
  ∀ z, z = a * f b → -1 ≤ z ∧ z < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_af_b_range_l975_97512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_cycle_l975_97507

/-- Represents the daily requirement of food additives in kg -/
def daily_requirement : ℚ := 200

/-- Represents the price of food additives in yuan/kg -/
def additive_price : ℚ := 9/5

/-- Represents the shipping fee per purchase in yuan -/
def shipping_fee : ℚ := 236

/-- Represents the storage fee for the first 7 days in yuan/day -/
def storage_fee_first_week : ℚ := 10

/-- Represents the storage fee after 7 days in yuan/kg/day -/
def storage_fee_after_week : ℚ := 3/100

/-- Represents the total cost function for x days between purchases -/
def total_cost (x : ℚ) : ℚ :=
  if x ≤ 7 then 370 * x + 236
  else 3 * x^2 + 321 * x + 432

/-- Represents the average daily cost function -/
noncomputable def average_daily_cost (x : ℚ) : ℚ := total_cost x / x

/-- Theorem stating that the optimal purchase cycle is 12 days -/
theorem optimal_purchase_cycle :
  ∃ (x : ℚ), x > 0 ∧ ∀ (y : ℚ), y > 0 → average_daily_cost x ≤ average_daily_cost y ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_cycle_l975_97507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_height_l975_97592

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  area : ℝ
  cot_angle : ℝ
  height : ℝ

/-- The height of an isosceles trapezoid is 4, given its area is 32 and the cotangent of the angle between the diagonal and the base is 2 -/
theorem isosceles_trapezoid_height 
  (t : IsoscelesTrapezoid) 
  (h_area : t.area = 32) 
  (h_cot : t.cot_angle = 2) : 
  t.height = 4 := by
  sorry

#check isosceles_trapezoid_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_height_l975_97592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_emptying_time_l975_97559

/-- Represents the tank system with a leak and an inlet pipe. -/
structure TankSystem where
  capacity : ℝ
  leakTime : ℝ
  inletRate : ℝ

/-- Calculates the time to empty the tank with both inlet and leak working. -/
noncomputable def timeToEmpty (ts : TankSystem) : ℝ :=
  ts.capacity / (ts.capacity / ts.leakTime - ts.inletRate * 60)

/-- Theorem stating the time to empty the tank under given conditions. -/
theorem tank_emptying_time (ts : TankSystem) 
  (h_capacity : ts.capacity = 5760)
  (h_leak_time : ts.leakTime = 6)
  (h_inlet_rate : ts.inletRate = 4) :
  timeToEmpty ts = 8 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_emptying_time_l975_97559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_value_l975_97554

theorem cos_2x_value (x : ℝ) (h1 : x > π/4 ∧ x < π/2) 
  (h2 : Real.sin (π/4 - x) = -3/5) : Real.cos (2*x) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_value_l975_97554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_quadratic_l975_97585

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

-- Define the property of monotonically decreasing on an interval
def monotonically_decreasing_on (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ x y, x ∈ interval → y ∈ interval → x < y → f x > f y

-- Theorem statement
theorem monotonically_decreasing_quadratic (a : ℝ) :
  monotonically_decreasing_on (f a) {x | x ≥ -1} → a = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_quadratic_l975_97585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_select_A_and_B_l975_97522

/-- The probability of selecting both A and B when randomly choosing 3 students from a group of 5 students (including A and B) -/
theorem probability_select_A_and_B (n : ℕ) (k : ℕ) (total_students : ℕ) (selected_students : ℕ) :
  n = 5 →
  k = 3 →
  total_students = n →
  selected_students = k →
  (Nat.choose n k : ℚ)⁻¹ * (Nat.choose (n - 2) (k - 2) : ℚ) = 3 / 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_select_A_and_B_l975_97522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_replacement_l975_97560

/-- Calculates the sugar percentage of a replacement solution -/
noncomputable def replacement_sugar_percentage (original_percentage : ℝ) (final_percentage : ℝ) 
  (replaced_fraction : ℝ) : ℝ :=
  (final_percentage - original_percentage * (1 - replaced_fraction)) / replaced_fraction

theorem sugar_solution_replacement 
  (original_percentage : ℝ) (final_percentage : ℝ) (replaced_fraction : ℝ)
  (h_original : original_percentage = 0.1)
  (h_final : final_percentage = 0.17)
  (h_replaced : replaced_fraction = 0.25) :
  replacement_sugar_percentage original_percentage final_percentage replaced_fraction = 0.38 := by
  sorry

#eval (0.17 - 0.1 * (1 - 0.25)) / 0.25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_replacement_l975_97560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_cos_sum_diff_l975_97545

theorem tan_ratio_from_cos_sum_diff (a b : ℝ) 
  (h1 : Real.cos (a + b) = 1/3) 
  (h2 : Real.cos (a - b) = 1/2) : 
  Real.tan a / Real.tan b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_cos_sum_diff_l975_97545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_identity_l975_97528

open Function

theorem functional_equation_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f (x + 1) + y - 1) = f x + y) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_identity_l975_97528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_similar_stars_l975_97527

-- Define number_of_non_similar_stars as a function
def number_of_non_similar_stars (p : ℕ+) : ℕ := 
  sorry -- We'll leave this as sorry for now, as the implementation is not provided

theorem count_non_similar_stars (p a b : ℕ+) (h1 : p = a * b) (h2 : a < b) 
  (h3 : Nat.Prime a.val) (h4 : Nat.Prime b.val) : 
  ∃ n : ℕ, n = (p - a - b + 1) / 2 ∧ n = number_of_non_similar_stars p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_similar_stars_l975_97527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_zeros_F_min_value_l975_97541

-- Use noncomputable section for definitions involving real numbers
noncomputable section

def f (x : ℝ) : ℝ := (1/2) * (x + 1/x)
def g (x : ℝ) : ℝ := (1/2) * (x - 1/x)
def h (x : ℝ) : ℝ := f x + 2 * g x
def F (n : ℕ) (x : ℝ) : ℝ := (f x)^(2*n) - (g x)^(2*n)

-- Fix the syntax error in the theorem statement
theorem h_zeros :
  ∀ x : ℝ, h x = 0 ↔ x = Real.sqrt 3 / 3 ∨ x = -(Real.sqrt 3 / 3) :=
sorry

theorem F_min_value (n : ℕ) (hn : n > 0) :
  ∀ x : ℝ, F n x ≥ 1 ∧ (F n x = 1 ↔ x = 1 ∨ x = -1) :=
sorry

end -- Close the noncomputable section

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_zeros_F_min_value_l975_97541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_sine_l975_97539

/-- If the terminal side of angle α passes through the point (2cos120°, √2sin225°), then sin α = -√2/2 -/
theorem angle_terminal_side_sine (α : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = 2 * (Real.cos (120 * π / 180)) ∧
                           r * (Real.sin α) = Real.sqrt 2 * (Real.sin (225 * π / 180))) →
  Real.sin α = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_sine_l975_97539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_d_equals_two_l975_97571

/-- A function f satisfying the given conditions -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + c) / (d * x + b)

/-- Theorem stating that under specific conditions, a + d = 2 -/
theorem a_plus_d_equals_two
  (a b c d : ℝ)
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h2 : ∀ x, f a b c d (f a b c d x) = x)
  (h3 : c = -a * b)
  (h4 : a = d) :
  a + d = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_d_equals_two_l975_97571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_of_data_with_given_average_and_variance_l975_97543

noncomputable def average (xs : List ℝ) : ℝ := xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := average xs
  (xs.map (fun x => (x - μ)^2)).sum / xs.length

theorem abs_diff_of_data_with_given_average_and_variance
  (x y : ℝ)
  (h_avg : average [x, y, 30, 29, 31] = 30)
  (h_var : variance [x, y, 30, 29, 31] = 2) :
  |x - y| = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_of_data_with_given_average_and_variance_l975_97543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_no_m_satisfies_union_condition_l975_97582

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((6 / (x + 1)) - 1)
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := Real.log (-x^2 + m*x + 4)

-- Define the domains A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | 0 < -x^2 + m*x + 4}

-- Statement 1
theorem intersection_of_A_and_complement_of_B :
  A ∩ (Set.univ \ B 3) = Set.Icc 4 5 := by sorry

-- Statement 2
theorem no_m_satisfies_union_condition :
  ¬∃ m, A ∪ B m = A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_no_m_satisfies_union_condition_l975_97582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_theorem_l975_97577

/-- Represents a moving walkway scenario -/
structure MovingWalkway where
  length : ℚ
  time_with : ℚ
  time_against : ℚ

/-- Calculates the time to walk the walkway if it were not moving -/
def time_without_movement (w : MovingWalkway) : ℚ :=
  (8 * w.length) / 15

/-- Theorem stating the time to walk the walkway if it were not moving -/
theorem walkway_time_theorem (w : MovingWalkway) 
  (h1 : w.length = 90)
  (h2 : w.time_with = 30)
  (h3 : w.time_against = 120) :
  time_without_movement w = 48 := by
  sorry

def example_walkway : MovingWalkway := ⟨90, 30, 120⟩

#eval time_without_movement example_walkway

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_time_theorem_l975_97577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_unique_property_l975_97584

-- Define the properties of quadrilaterals
class Quadrilateral (α : Type*) :=
  (diagonals_equal : α → Prop)
  (diagonals_bisect : α → Prop)
  (diagonals_perpendicular : α → Prop)
  (opposite_sides_equal_parallel : α → Prop)

-- Define rhombus
class Rhombus (α : Type*) extends Quadrilateral α :=
  (is_rhombus : α → Prop)
  (rhombus_diagonals_perpendicular : ∀ r : α, is_rhombus r → diagonals_perpendicular r)

-- Define rectangle
class Rectangle (α : Type*) extends Quadrilateral α :=
  (is_rectangle : α → Prop)

-- Theorem statement
theorem rhombus_unique_property {α : Type*} [Rhombus α] [Rectangle α] :
  ∃ r : α, Rhombus.is_rhombus r ∧ Quadrilateral.diagonals_perpendicular r ∧
  ¬(∀ rect : α, Rectangle.is_rectangle rect → Quadrilateral.diagonals_perpendicular rect) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_unique_property_l975_97584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donut_students_count_l975_97579

/-- Represents the fundraiser scenario -/
structure Fundraiser where
  brownie_students : ℕ
  brownie_per_student : ℕ
  cookie_students : ℕ
  cookie_per_student : ℕ
  donut_per_student : ℕ
  price_per_item : ℚ
  total_raised : ℚ

/-- Calculates the number of students asked to bring donuts -/
def donut_students (f : Fundraiser) : ℕ :=
  let brownie_count := f.brownie_students * f.brownie_per_student
  let cookie_count := f.cookie_students * f.cookie_per_student
  let non_donut_revenue := (brownie_count + cookie_count : ℚ) * f.price_per_item
  let donut_revenue := f.total_raised - non_donut_revenue
  let donut_count := donut_revenue / f.price_per_item
  (donut_count / f.donut_per_student).floor.toNat

/-- Theorem stating that the number of students asked to bring donuts is 15 -/
theorem donut_students_count (f : Fundraiser) : 
  f.brownie_students = 30 →
  f.brownie_per_student = 12 →
  f.cookie_students = 20 →
  f.cookie_per_student = 24 →
  f.donut_per_student = 12 →
  f.price_per_item = 2 →
  f.total_raised = 2040 →
  donut_students f = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donut_students_count_l975_97579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l975_97537

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * Real.sin (x - Real.pi / 4) ^ 2

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x, f (x + Real.pi) = f x) ∧  -- f has period π
  (∀ p, 0 < p → p < Real.pi → ∃ x, f (x + p) ≠ f x) :=  -- π is the smallest positive period
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l975_97537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_and_angle_range_l975_97501

/-- Given a line passing through points A(2, 1) and B(1, m^2) where m ∈ ℝ,
    this theorem proves the range of the slope and angle of inclination. -/
theorem line_slope_and_angle_range (m : ℝ) :
  ∃ (slope α : ℝ), 
    slope = 1 - m^2 ∧
    0 ≤ α ∧ α < Real.pi ∧
    (slope ≤ 1) ∧
    (α ∈ Set.Icc 0 (Real.pi/4) ∪ Set.Ioo (Real.pi/2) Real.pi) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_and_angle_range_l975_97501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_seven_equality_l975_97521

theorem power_seven_equality (x : ℝ) : (7 : ℝ)^(4*x) = 343 → (7 : ℝ)^(4*x - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_seven_equality_l975_97521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l975_97589

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents the game state -/
structure GameState where
  unused : Finset Nat
  top_row : List Nat
  bottom_row : List Nat

/-- Defines a valid move in the game -/
def ValidMove (s : GameState) (n : Nat) (row : Bool) : Prop :=
  n ∈ s.unused ∧ (row → s.top_row.length < 4) ∧ (¬row → s.bottom_row.length < 4)

/-- Defines the winning condition for a player -/
def Wins (p : Player) (s : GameState) : Prop :=
  match p with
  | Player.First => s.top_row.prod > s.bottom_row.prod
  | Player.Second => s.bottom_row.prod > s.top_row.prod

/-- Function to compute the final state of the game given strategies for both players -/
def final_state (initial : GameState) 
  (strategy : GameState → Nat × Bool)
  (opponent_strategy : GameState → Nat × Bool) : GameState :=
sorry

/-- The main theorem stating that the first player can always win -/
theorem first_player_wins :
  ∀ (initial_state : GameState),
  initial_state.unused = {1, 2, 3, 4, 5, 6, 7, 8} ∧
  initial_state.top_row = [] ∧
  initial_state.bottom_row = [] →
  ∃ (strategy : GameState → Nat × Bool),
  ∀ (opponent_strategy : GameState → Nat × Bool),
  Wins Player.First (final_state initial_state strategy opponent_strategy) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l975_97589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_MC₂N_l975_97576

-- Define the curve C₁
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, Real.sin α)

-- Define the circle C₂
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the ray l
def l (θ₀ ρ : ℝ) : Prop := ρ ≥ 0 ∧ ρ * Real.cos θ₀ = ρ * Real.sin θ₀

-- Define the intersection points M and N
noncomputable def M (θ₀ : ℝ) : ℝ × ℝ := (2 * Real.cos θ₀, Real.sin θ₀)
noncomputable def N (θ₀ : ℝ) : ℝ × ℝ := (4 * Real.cos θ₀ * Real.cos θ₀, 4 * Real.cos θ₀ * Real.sin θ₀)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem area_of_triangle_MC₂N (θ₀ : ℝ) :
  0 < θ₀ ∧ θ₀ < Real.pi / 2 →
  C₂ (N θ₀).1 (N θ₀).2 →
  distance (0, 0) (N θ₀) = 2 * distance (0, 0) (M θ₀) →
  let C₂_center : ℝ × ℝ := (2, 0)
  let triangle_area := (1 / 2) * distance C₂_center (N θ₀) * (distance (0, 0) (N θ₀) - distance (0, 0) (M θ₀)) * Real.sin θ₀
  triangle_area = 2 * Real.sqrt 2 / 3 := by
  sorry

#check area_of_triangle_MC₂N

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_MC₂N_l975_97576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_102nd_term_l975_97562

theorem geometric_sequence_102nd_term 
  (a₁ : ℝ) 
  (a₂ : ℝ) 
  (h₁ : a₁ = 12) 
  (h₂ : a₂ = -24) : 
  let r := a₂ / a₁
  let aₙ := fun n ↦ a₁ * r^(n-1)
  aₙ 102 = -2^101 * 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_102nd_term_l975_97562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_leg_ratios_l975_97552

/-- Represents a right triangle with legs a and b, and hypotenuse c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2
  positive_a : a > 0
  positive_b : b > 0
  positive_c : c > 0

/-- Surface areas of solids formed by rotating the triangle -/
noncomputable def surface_areas (t : RightTriangle) : ℝ × ℝ × ℝ :=
  (t.b * Real.pi * (2*t.a + t.b + t.c),
   t.a * Real.pi * (t.a + 2*t.b + t.c),
   Real.pi * (t.a * t.b / t.c) * (t.a + t.b + 2*t.c))

/-- Volumes of solids formed by rotating the triangle -/
noncomputable def volumes (t : RightTriangle) : ℝ × ℝ × ℝ :=
  ((2/3) * Real.pi * t.b^2 * t.a,
   (2/3) * Real.pi * t.a^2 * t.b,
   (2/3) * Real.pi * (t.a * t.b)^2 / t.c)

/-- Theorem stating the ratios of surface areas and volumes when a = b -/
theorem equal_leg_ratios (t : RightTriangle) (h : t.a = t.b) :
  let (F_a, F_b, F_c) := surface_areas t
  let (K_a, K_b, K_c) := volumes t
  (F_a / F_a : ℝ) = 1 ∧
  (F_b / F_a : ℝ) = 1 ∧
  (F_c / F_a : ℝ) = (4 - Real.sqrt 2) / 7 ∧
  (K_a / K_a : ℝ) = 1 ∧
  (K_b / K_a : ℝ) = 1 ∧
  (K_c / K_a : ℝ) = Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_leg_ratios_l975_97552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l975_97580

/-- A quadrilateral with specific properties -/
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)
  (right_angle_Q : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0)
  (PR_perp_RS : (R.1 - P.1) * (S.1 - R.1) + (R.2 - P.2) * (S.2 - R.2) = 0)
  (PQ_length : ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 15^2)
  (QR_length : ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 20^2)
  (RS_length : ((S.1 - R.1)^2 + (S.2 - R.2)^2) = 9^2)

/-- The perimeter of a quadrilateral -/
noncomputable def perimeter (q : Quadrilateral) : ℝ :=
  Real.sqrt ((q.Q.1 - q.P.1)^2 + (q.Q.2 - q.P.2)^2) +
  Real.sqrt ((q.R.1 - q.Q.1)^2 + (q.R.2 - q.Q.2)^2) +
  Real.sqrt ((q.S.1 - q.R.1)^2 + (q.S.2 - q.R.2)^2) +
  Real.sqrt ((q.P.1 - q.S.1)^2 + (q.P.2 - q.S.2)^2)

/-- The main theorem -/
theorem quadrilateral_perimeter (q : Quadrilateral) :
  perimeter q = 44 + Real.sqrt 706 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l975_97580
