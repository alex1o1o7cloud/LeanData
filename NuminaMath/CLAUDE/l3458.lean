import Mathlib

namespace proper_subset_of_A_l3458_345812

def A : Set ℝ := {x | x^2 < 5*x}

theorem proper_subset_of_A : Set.Subset (Set.Ioo 1 5) A ∧ (Set.Ioo 1 5) ≠ A := by
  sorry

end proper_subset_of_A_l3458_345812


namespace normal_distribution_probability_l3458_345837

def normal_distribution (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  ∃ f : ℝ → ℝ, ∀ x, f x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

theorem normal_distribution_probability (X : ℝ → ℝ) (μ σ : ℝ) :
  normal_distribution μ σ X →
  (∫ x in Set.Ioo (μ - 2*σ) (μ + 2*σ), X x) = 0.9544 →
  (∫ x in Set.Ioo (μ - σ) (μ + σ), X x) = 0.6826 →
  μ = 4 →
  σ = 1 →
  (∫ x in Set.Ioo 5 6, X x) = 0.1359 :=
by sorry

end normal_distribution_probability_l3458_345837


namespace cos_one_sufficient_not_necessary_l3458_345833

theorem cos_one_sufficient_not_necessary (x : ℝ) : 
  (∀ x, Real.cos x = 1 → Real.sin x = 0) ∧ 
  (∃ x, Real.sin x = 0 ∧ Real.cos x ≠ 1) := by
  sorry

end cos_one_sufficient_not_necessary_l3458_345833


namespace real_part_of_z_l3458_345890

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  (z.re : ℝ) = 1 := by
  sorry

end real_part_of_z_l3458_345890


namespace integer_pair_property_l3458_345835

theorem integer_pair_property (x y : ℤ) (h : x > y) :
  (x * y - (x + y) = Nat.gcd x.natAbs y.natAbs + Nat.lcm x.natAbs y.natAbs) ↔
  ((x = 6 ∧ y = 3) ∨ 
   (x = 6 ∧ y = 4) ∨ 
   (∃ t : ℕ, x = 1 + t ∧ y = -t) ∨
   (∃ t : ℕ, x = 2 ∧ y = -2 * t)) := by
sorry

end integer_pair_property_l3458_345835


namespace quadratic_function_theorem_l3458_345839

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the quadratic function at a given x -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Checks if a point (x, y) lies on the quadratic function -/
def QuadraticFunction.passesThrough (f : QuadraticFunction) (x y : ℝ) : Prop :=
  f.evaluate x = y

theorem quadratic_function_theorem (f : QuadraticFunction) :
  f.c = -3 →
  f.passesThrough 2 (-3) →
  f.passesThrough (-1) 0 →
  (f.a = 1 ∧ f.b = -2) ∧
  (∃ k : ℝ, k = 4 ∧
    (∀ x : ℝ, (f.evaluate x + k = 0) → (∀ y : ℝ, y ≠ x → f.evaluate y + k ≠ 0))) :=
by sorry

end quadratic_function_theorem_l3458_345839


namespace electricity_consumption_for_2_75_yuan_l3458_345889

-- Define the relationship between electricity consumption and charges
def electricity_charge (consumption : ℝ) : ℝ := 0.55 * consumption

-- Theorem statement
theorem electricity_consumption_for_2_75_yuan :
  ∃ (consumption : ℝ), electricity_charge consumption = 2.75 ∧ consumption = 5 :=
sorry

end electricity_consumption_for_2_75_yuan_l3458_345889


namespace trapezoid_perimeters_l3458_345875

/-- A trapezoid with given measurements -/
structure Trapezoid where
  longerBase : ℝ
  height : ℝ
  leg1 : ℝ
  leg2 : ℝ

/-- The set of possible perimeters for a given trapezoid -/
def possiblePerimeters (t : Trapezoid) : Set ℝ :=
  {p | ∃ shorterBase : ℝ, 
    p = t.longerBase + t.leg1 + t.leg2 + shorterBase ∧
    shorterBase > 0 ∧
    (shorterBase = t.longerBase - Real.sqrt (t.leg1^2 - t.height^2) - Real.sqrt (t.leg2^2 - t.height^2) ∨
     shorterBase = t.longerBase + Real.sqrt (t.leg1^2 - t.height^2) - Real.sqrt (t.leg2^2 - t.height^2))}

/-- The theorem to be proved -/
theorem trapezoid_perimeters (t : Trapezoid) 
  (h1 : t.longerBase = 30)
  (h2 : t.height = 24)
  (h3 : t.leg1 = 25)
  (h4 : t.leg2 = 30) :
  possiblePerimeters t = {90, 104} := by
  sorry

end trapezoid_perimeters_l3458_345875


namespace mary_anne_sparkling_water_l3458_345842

-- Define the cost per bottle
def cost_per_bottle : ℚ := 2

-- Define the total spent per year
def total_spent_per_year : ℚ := 146

-- Define the number of days in a year
def days_per_year : ℕ := 365

-- Define the fraction of a bottle drunk each night
def fraction_per_night : ℚ := 1 / 5

-- Theorem statement
theorem mary_anne_sparkling_water :
  fraction_per_night * (days_per_year : ℚ) = total_spent_per_year / cost_per_bottle :=
sorry

end mary_anne_sparkling_water_l3458_345842


namespace shortest_tangent_length_l3458_345821

/-- Given two circles C₁ and C₂ defined by equations (x-12)²+y²=25 and (x+18)²+y²=64 respectively,
    the length of the shortest line segment RS tangent to C₁ at R and C₂ at S is 339/13. -/
theorem shortest_tangent_length (C₁ C₂ : Set (ℝ × ℝ)) (R S : ℝ × ℝ) :
  C₁ = {p : ℝ × ℝ | (p.1 - 12)^2 + p.2^2 = 25} →
  C₂ = {p : ℝ × ℝ | (p.1 + 18)^2 + p.2^2 = 64} →
  R ∈ C₁ →
  S ∈ C₂ →
  (∀ p ∈ C₁, (R.1 - p.1) * (R.1 - 12) + (R.2 - p.2) * R.2 = 0) →
  (∀ p ∈ C₂, (S.1 - p.1) * (S.1 + 18) + (S.2 - p.2) * S.2 = 0) →
  (∀ T U : ℝ × ℝ, T ∈ C₁ → U ∈ C₂ → 
    (∀ q ∈ C₁, (T.1 - q.1) * (T.1 - 12) + (T.2 - q.2) * T.2 = 0) →
    (∀ q ∈ C₂, (U.1 - q.1) * (U.1 + 18) + (U.2 - q.2) * U.2 = 0) →
    Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) ≤ Real.sqrt ((T.1 - U.1)^2 + (T.2 - U.2)^2)) →
  Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 339 / 13 :=
by sorry

end shortest_tangent_length_l3458_345821


namespace alicia_tax_deduction_l3458_345853

/-- Represents Alicia's hourly wage in dollars -/
def hourly_wage : ℚ := 25

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℚ := 25 / 1000

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

/-- Calculates the tax deduction in cents -/
def tax_deduction (wage : ℚ) (rate : ℚ) : ℚ :=
  dollars_to_cents (wage * rate)

theorem alicia_tax_deduction :
  tax_deduction hourly_wage tax_rate = 62.5 := by
  sorry

end alicia_tax_deduction_l3458_345853


namespace roots_of_polynomial_l3458_345824

def p (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) :=
by sorry

end roots_of_polynomial_l3458_345824


namespace largest_non_representable_integer_l3458_345894

/-- 
Given positive integers a, b, and c with no two having a common divisor greater than 1,
2abc-ab-bc-ca is the largest integer that cannot be expressed as xbc+yca+zab 
for non-negative integers x, y, z
-/
theorem largest_non_representable_integer (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : Nat.gcd a b = 1) (hbc : Nat.gcd b c = 1) (hca : Nat.gcd c a = 1) :
  (∀ x y z : ℕ, 2*a*b*c - a*b - b*c - c*a ≠ x*b*c + y*c*a + z*a*b) ∧
  (∀ n : ℕ, n > 2*a*b*c - a*b - b*c - c*a → 
    ∃ x y z : ℕ, n = x*b*c + y*c*a + z*a*b) := by
  sorry

end largest_non_representable_integer_l3458_345894


namespace rectangle_center_line_slope_l3458_345867

/-- The slope of a line passing through the origin and the center of a rectangle
    with vertices (1, 0), (5, 0), (1, 2), and (5, 2) is 1/3. -/
theorem rectangle_center_line_slope :
  let vertices : List (ℝ × ℝ) := [(1, 0), (5, 0), (1, 2), (5, 2)]
  let center : ℝ × ℝ := (
    (vertices.map Prod.fst).sum / vertices.length,
    (vertices.map Prod.snd).sum / vertices.length
  )
  let slope : ℝ := (center.2 - 0) / (center.1 - 0)
  slope = 1 / 3 := by
sorry


end rectangle_center_line_slope_l3458_345867


namespace doors_per_apartment_l3458_345841

/-- Proves that the number of doors per apartment is 7, given the specifications of the apartment buildings and total doors needed. -/
theorem doors_per_apartment 
  (num_buildings : ℕ) 
  (floors_per_building : ℕ) 
  (apartments_per_floor : ℕ) 
  (total_doors : ℕ) 
  (h1 : num_buildings = 2)
  (h2 : floors_per_building = 12)
  (h3 : apartments_per_floor = 6)
  (h4 : total_doors = 1008) :
  total_doors / (num_buildings * floors_per_building * apartments_per_floor) = 7 := by
  sorry

#check doors_per_apartment

end doors_per_apartment_l3458_345841


namespace combined_temp_range_l3458_345878

-- Define the temperature ranges for each vegetable type
def type_a_range : Set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }
def type_b_range : Set ℝ := { x | 3 ≤ x ∧ x ≤ 8 }

-- Define the combined suitable temperature range
def combined_range : Set ℝ := type_a_range ∩ type_b_range

-- Theorem stating the combined suitable temperature range
theorem combined_temp_range : 
  combined_range = { x | 3 ≤ x ∧ x ≤ 5 } := by sorry

end combined_temp_range_l3458_345878


namespace total_is_600_l3458_345880

/-- Represents the shares of money for three individuals -/
structure Shares :=
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)

/-- The conditions of the money division problem -/
def SatisfiesConditions (s : Shares) : Prop :=
  s.a = (2/3) * (s.b + s.c) ∧
  s.b = (6/9) * (s.a + s.c) ∧
  s.a = 240

/-- The theorem stating that the total amount is 600 given the conditions -/
theorem total_is_600 (s : Shares) (h : SatisfiesConditions s) :
  s.a + s.b + s.c = 600 := by
  sorry

#check total_is_600

end total_is_600_l3458_345880


namespace B_2_1_equals_12_l3458_345832

-- Define the function B using the given recurrence relation
def B : ℕ → ℕ → ℕ
| 0, n => n + 2
| m + 1, 0 => B m 2
| m + 1, n + 1 => B m (B (m + 1) n)

-- Theorem statement
theorem B_2_1_equals_12 : B 2 1 = 12 := by
  sorry

end B_2_1_equals_12_l3458_345832


namespace frog_jump_probability_l3458_345807

-- Define the square
def Square := {(x, y) : ℝ × ℝ | 0 ≤ x ∧ x ≤ 6 ∧ 0 ≤ y ∧ y ≤ 6}

-- Define the vertical sides of the square
def VerticalSides := {(x, y) : ℝ × ℝ | (x = 0 ∨ x = 6) ∧ 0 ≤ y ∧ y ≤ 6}

-- Define the possible jump directions
inductive Direction
| Up
| Down
| Left
| Right

-- Define a function to represent a single jump
def jump (pos : ℝ × ℝ) (dir : Direction) : ℝ × ℝ :=
  match dir with
  | Direction.Up => (pos.1, pos.2 + 2)
  | Direction.Down => (pos.1, pos.2 - 2)
  | Direction.Left => (pos.1 - 2, pos.2)
  | Direction.Right => (pos.1 + 2, pos.2)

-- Define the probability function
noncomputable def P (pos : ℝ × ℝ) : ℝ :=
  sorry  -- The actual implementation would go here

-- State the theorem
theorem frog_jump_probability :
  P (1, 3) = 2/3 :=
sorry

end frog_jump_probability_l3458_345807


namespace inequality_solution_set_l3458_345854

theorem inequality_solution_set (a b : ℝ) (h : b ≠ 0) :
  ¬(∀ x : ℝ, ax > b ↔ x < -b/a) :=
by sorry

end inequality_solution_set_l3458_345854


namespace sum_of_first_45_terms_l3458_345825

def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := 3*n - 1

def c (n : ℕ) : ℕ := a n + b n

def S (n : ℕ) : ℕ := (2^n - 1) + n * (3*n + 1) / 2 - (2 + 8 + 32)

theorem sum_of_first_45_terms : S 45 = 2^45 - 3017 := by sorry

end sum_of_first_45_terms_l3458_345825


namespace two_digit_primes_from_set_l3458_345852

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def digit_set : Finset ℕ := {3, 5, 8, 9}

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def formed_from_set (n : ℕ) : Prop :=
  is_two_digit n ∧
  (n / 10) ∈ digit_set ∧
  (n % 10) ∈ digit_set ∧
  (n / 10) ≠ (n % 10)

theorem two_digit_primes_from_set :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, is_prime n ∧ formed_from_set n) ∧
    (∀ n, is_prime n ∧ formed_from_set n → n ∈ s) ∧
    s.card = 2 :=
sorry

end two_digit_primes_from_set_l3458_345852


namespace arithmetic_sequence_squares_l3458_345844

theorem arithmetic_sequence_squares (a b c : ℝ) 
  (h : ∃ (d : ℝ), (1 / (b + c)) - (1 / (a + b)) = (1 / (c + a)) - (1 / (b + c))) :
  ∃ (k : ℝ), b^2 - a^2 = c^2 - b^2 :=
sorry

end arithmetic_sequence_squares_l3458_345844


namespace range_of_y_minus_x_l3458_345866

-- Define the triangle ABC and points D and P
variable (A B C D P : ℝ × ℝ)

-- Define vectors
def vec (X Y : ℝ × ℝ) : ℝ × ℝ := (Y.1 - X.1, Y.2 - X.2)

-- Conditions
variable (h1 : vec D C = (2 * (vec A D).1, 2 * (vec A D).2))
variable (h2 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * B.1 + (1 - t) * D.1, t * B.2 + (1 - t) * D.2))
variable (h3 : ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ vec A P = (x * (vec A B).1 + y * (vec A C).1, x * (vec A B).2 + y * (vec A C).2))

-- Theorem statement
theorem range_of_y_minus_x :
  ∃ S : Set ℝ, S = {z | ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    vec A P = (x * (vec A B).1 + y * (vec A C).1, x * (vec A B).2 + y * (vec A C).2) ∧
    z = y - x} ∧
  S = {z | -1 < z ∧ z < 1/3} :=
sorry

end range_of_y_minus_x_l3458_345866


namespace min_sum_with_constraint_l3458_345868

theorem min_sum_with_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = a * b) :
  a + b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4 * a₀ + b₀ = a₀ * b₀ ∧ a₀ + b₀ = 9 := by
  sorry

end min_sum_with_constraint_l3458_345868


namespace bus_speed_l3458_345873

/-- Calculates the speed of a bus in kilometers per hour (kmph) given distance and time -/
theorem bus_speed (distance : Real) (time : Real) (conversion_factor : Real) : 
  distance = 900.072 ∧ time = 30 ∧ conversion_factor = 3.6 →
  (distance / time) * conversion_factor = 108.00864 := by
  sorry

#check bus_speed

end bus_speed_l3458_345873


namespace concert_ticket_cost_l3458_345806

/-- Calculates the total cost of concert tickets for a group of friends --/
theorem concert_ticket_cost :
  let normal_price : ℚ := 50
  let website_tickets : ℕ := 3
  let scalper_tickets : ℕ := 4
  let scalper_price_multiplier : ℚ := 2.5
  let scalper_discount : ℚ := 15
  let service_fee_rate : ℚ := 0.1
  let discount_ticket1_rate : ℚ := 0.6
  let discount_ticket2_rate : ℚ := 0.75

  let website_cost : ℚ := normal_price * website_tickets
  let website_fee : ℚ := website_cost * service_fee_rate
  let total_website_cost : ℚ := website_cost + website_fee

  let scalper_cost : ℚ := normal_price * scalper_tickets * scalper_price_multiplier - scalper_discount
  let scalper_fee : ℚ := scalper_cost * service_fee_rate
  let total_scalper_cost : ℚ := scalper_cost + scalper_fee

  let discount_ticket1_cost : ℚ := normal_price * discount_ticket1_rate
  let discount_ticket2_cost : ℚ := normal_price * discount_ticket2_rate
  let total_discount_cost : ℚ := discount_ticket1_cost + discount_ticket2_cost

  let total_cost : ℚ := total_website_cost + total_scalper_cost + total_discount_cost

  total_cost = 766
  := by sorry

end concert_ticket_cost_l3458_345806


namespace inequality_proof_l3458_345847

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  1 / (b - c) > 1 / (a - c) := by
  sorry

end inequality_proof_l3458_345847


namespace no_x_squared_term_l3458_345822

theorem no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + 5) * (-2*x) - 6*x^2 = -2*x^3 - 10*x) ↔ a = -3 :=
by sorry

end no_x_squared_term_l3458_345822


namespace work_completion_time_l3458_345888

/-- Given that two workers A and B can complete a work together in a certain number of days,
    and worker A can complete the work alone in a certain number of days,
    this function calculates the number of days worker B needs to complete the work alone. -/
def days_for_b_alone (days_together : ℚ) (days_a_alone : ℚ) : ℚ :=
  (days_together * days_a_alone) / (days_a_alone - days_together)

/-- Theorem stating that if A and B together can complete a work in 4 days,
    and A alone can complete the same work in 12 days,
    then B alone can complete the work in 6 days. -/
theorem work_completion_time :
  days_for_b_alone 4 12 = 6 := by
  sorry


end work_completion_time_l3458_345888


namespace brendas_weight_multiple_l3458_345819

theorem brendas_weight_multiple (brenda_weight mel_weight : ℕ) (multiple : ℚ) : 
  brenda_weight = 220 →
  mel_weight = 70 →
  brenda_weight = mel_weight * multiple + 10 →
  multiple = 3 := by
  sorry

end brendas_weight_multiple_l3458_345819


namespace circle_diameter_from_triangle_l3458_345891

/-- Theorem: The diameter of a circle inscribing a right triangle with area 150 and one leg 30 is 10√10 -/
theorem circle_diameter_from_triangle (triangle_area : ℝ) (leg : ℝ) (diameter : ℝ) : 
  triangle_area = 150 →
  leg = 30 →
  diameter = 10 * Real.sqrt 10 :=
by sorry

end circle_diameter_from_triangle_l3458_345891


namespace min_c_value_l3458_345881

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! (x y : ℝ), 2 * x + y = 2025 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1013 :=
by sorry

end min_c_value_l3458_345881


namespace quadratic_function_theorem_l3458_345879

-- Define the quadratic function f
def f (x : ℝ) : ℝ := x^2 + x - 2

-- Define the theorem
theorem quadratic_function_theorem :
  (∀ x : ℝ, f x < 0 ↔ -2 < x ∧ x < 1) ∧ 
  f 0 = -2 ∧
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (∀ x : ℝ, f x = x^2 + x - 2) ∧
  (∀ m : ℝ, (∀ θ : ℝ, f (Real.cos θ) ≤ Real.sqrt 2 * Real.sin (θ + Real.pi / 4) + m * Real.sin θ) ↔ 
    -3 ≤ m ∧ m ≤ 1) := by
  sorry

end quadratic_function_theorem_l3458_345879


namespace cubic_inequality_l3458_345886

theorem cubic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 * b + b^3 * c + c^3 * a - a^2 * b * c - b^2 * c * a - c^2 * a * b ≥ 0 := by
  sorry

end cubic_inequality_l3458_345886


namespace surface_area_unchanged_l3458_345898

/-- The surface area of a cube after removing smaller cubes from its corners --/
def surface_area_after_removal (cube_size : ℝ) (corner_size : ℝ) : ℝ :=
  6 * cube_size^2

/-- Theorem: The surface area remains unchanged after corner removal --/
theorem surface_area_unchanged
  (cube_size : ℝ)
  (corner_size : ℝ)
  (h1 : cube_size = 4)
  (h2 : corner_size = 1.5)
  : surface_area_after_removal cube_size corner_size = 96 := by
  sorry

#check surface_area_unchanged

end surface_area_unchanged_l3458_345898


namespace arithmetic_progression_non_prime_existence_l3458_345882

theorem arithmetic_progression_non_prime_existence 
  (a d : ℕ+) : 
  ∃ K : ℕ+, ∀ n : ℕ, ∃ i : Fin K, ¬ Nat.Prime (a + (n + i) * d) :=
sorry

end arithmetic_progression_non_prime_existence_l3458_345882


namespace greatest_power_sum_l3458_345855

/-- Given positive integers c and d where d > 1, if c^d is the greatest possible value less than 800, then c + d = 30 -/
theorem greatest_power_sum (c d : ℕ) (hc : c > 0) (hd : d > 1) 
  (h : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 800 → c^d ≥ x^y) : c + d = 30 := by
  sorry

end greatest_power_sum_l3458_345855


namespace china_gdp_scientific_notation_l3458_345802

def trillion : ℝ := 10^12

theorem china_gdp_scientific_notation :
  11.69 * trillion = 1.169 * 10^14 := by sorry

end china_gdp_scientific_notation_l3458_345802


namespace multiply_and_distribute_l3458_345857

theorem multiply_and_distribute (m : ℝ) : (4*m + 1) * (2*m) = 8*m^2 + 2*m := by
  sorry

end multiply_and_distribute_l3458_345857


namespace jamie_yellow_balls_l3458_345870

theorem jamie_yellow_balls (initial_red : ℕ) (total_after : ℕ) : 
  initial_red = 16 →
  total_after = 74 →
  (initial_red - 6) + (2 * initial_red) + (total_after - ((initial_red - 6) + (2 * initial_red))) = total_after :=
by
  sorry

end jamie_yellow_balls_l3458_345870


namespace sum_expression_value_l3458_345887

theorem sum_expression_value (a b c : ℝ) 
  (h1 : a + b = 8) 
  (h2 : a * b = c^2 + 16) : 
  a + 2*b + 3*c = 12 := by
sorry

end sum_expression_value_l3458_345887


namespace rotate_180_equals_optionC_l3458_345846

/-- Represents a geometric shape --/
structure Shape :=
  (id : ℕ)

/-- Represents a rotation operation --/
def rotate (s : Shape) (angle : ℝ) : Shape :=
  { id := s.id }

/-- The original T-like shape --/
def original : Shape :=
  { id := 0 }

/-- Option C from the problem --/
def optionC : Shape :=
  { id := 1 }

/-- Theorem stating that rotating the original shape 180 degrees results in option C --/
theorem rotate_180_equals_optionC : 
  rotate original 180 = optionC := by
  sorry

end rotate_180_equals_optionC_l3458_345846


namespace philippe_can_win_l3458_345827

/-- Represents a game state with cards remaining and sums for each player -/
structure GameState :=
  (remaining : Finset Nat)
  (philippe_sum : Nat)
  (emmanuel_sum : Nat)

/-- The initial game state -/
def initial_state : GameState :=
  { remaining := Finset.range 2018,
    philippe_sum := 0,
    emmanuel_sum := 0 }

/-- A strategy is a function that selects a card from the remaining set -/
def Strategy := (GameState → Nat)

/-- Applies a strategy to a game state, returning the new state -/
def apply_strategy (s : Strategy) (g : GameState) : GameState :=
  let card := s g
  { remaining := g.remaining.erase card,
    philippe_sum := g.philippe_sum + card,
    emmanuel_sum := g.emmanuel_sum }

/-- Plays the game to completion using the given strategies -/
def play_game (philippe_strategy : Strategy) (emmanuel_strategy : Strategy) : GameState :=
  sorry

/-- Theorem stating that Philippe can always win -/
theorem philippe_can_win :
  ∃ (philippe_strategy : Strategy),
    ∀ (emmanuel_strategy : Strategy),
      let final_state := play_game philippe_strategy emmanuel_strategy
      Even final_state.philippe_sum ∧ Odd final_state.emmanuel_sum :=
sorry

end philippe_can_win_l3458_345827


namespace triangle_cosine_problem_l3458_345897

theorem triangle_cosine_problem (A B C : ℝ) (a b c : ℝ) (D : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sqrt 3 * Real.sin (2018 * Real.pi - x) * Real.sin (3 * Real.pi / 2 + x) - Real.cos x ^ 2 + 1
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- D is on angle bisector of A
  2 * Real.cos (A / 2) * Real.sin (B / 2) = Real.sin (C / 2) →
  -- f(A) = 3/2
  f A = 3 / 2 →
  -- AD = √2 BD = 2
  2 * Real.sin (B / 2) = Real.sqrt 2 * Real.sin (C / 2) ∧
  2 * Real.sin (B / 2) = 2 * Real.sin ((B + C) / 2) →
  -- Conclusion
  Real.cos C = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end triangle_cosine_problem_l3458_345897


namespace field_length_is_96_l3458_345801

/-- Proves that the length of a rectangular field is 96 meters given the specified conditions. -/
theorem field_length_is_96 (l w : ℝ) (h1 : l = 2 * w) (h2 : 64 = (1 / 72) * (l * w)) : l = 96 := by
  sorry

end field_length_is_96_l3458_345801


namespace remainder_theorem_l3458_345858

theorem remainder_theorem : ∃ q : ℕ, 3^303 + 303 = (3^151 + 3^76 + 1) * q + 303 := by
  sorry

end remainder_theorem_l3458_345858


namespace arithmetic_sequence_iff_constant_difference_l3458_345813

/-- A sequence is arithmetic if and only if the difference between consecutive terms is constant -/
theorem arithmetic_sequence_iff_constant_difference (a : ℕ → ℝ) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) ↔ 
  (∃ a₀ d : ℝ, ∀ n : ℕ, a n = a₀ + n • d) :=
sorry

end arithmetic_sequence_iff_constant_difference_l3458_345813


namespace max_value_is_12_l3458_345820

/-- Represents an arithmetic expression using the given operations and numbers -/
inductive Expr
  | num : ℕ → Expr
  | add : Expr → Expr → Expr
  | div : Expr → Expr → Expr
  | mul : Expr → Expr → Expr

/-- Evaluates an arithmetic expression -/
def eval : Expr → ℚ
  | Expr.num n => n
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.div e1 e2 => eval e1 / eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2

/-- Checks if an expression uses the given numbers in order -/
def usesNumbers (e : Expr) (nums : List ℕ) : Prop := sorry

/-- Counts the number of times each operation is used in an expression -/
def countOps (e : Expr) : (ℕ × ℕ × ℕ) := sorry

/-- Checks if an expression uses at most one pair of parentheses -/
def atMostOneParenthesis (e : Expr) : Prop := sorry

/-- The main theorem statement -/
theorem max_value_is_12 :
  ∀ e : Expr,
    usesNumbers e [7, 2, 3, 4] →
    countOps e = (1, 1, 1) →
    atMostOneParenthesis e →
    eval e ≤ 12 :=
by sorry

end max_value_is_12_l3458_345820


namespace encyclopedia_interest_percentage_l3458_345814

/-- Calculates the interest paid as a percentage of the amount borrowed for a set of encyclopedias. -/
theorem encyclopedia_interest_percentage (cost : ℝ) (down_payment : ℝ) (monthly_payment : ℝ) (num_months : ℕ) (final_payment : ℝ) :
  cost = 1200 →
  down_payment = 500 →
  monthly_payment = 70 →
  num_months = 12 →
  final_payment = 45 →
  let total_paid := down_payment + (monthly_payment * num_months) + final_payment
  let amount_borrowed := cost - down_payment
  let interest_paid := total_paid - cost
  let interest_percentage := (interest_paid / amount_borrowed) * 100
  ∃ ε > 0, |interest_percentage - 26.43| < ε :=
by sorry

end encyclopedia_interest_percentage_l3458_345814


namespace worker_idle_days_l3458_345851

/-- Proves that given the specified conditions, the number of idle days is 38 --/
theorem worker_idle_days 
  (total_days : ℕ) 
  (pay_per_working_day : ℕ) 
  (forfeit_per_idle_day : ℕ) 
  (total_amount : ℕ) 
  (h1 : total_days = 60)
  (h2 : pay_per_working_day = 30)
  (h3 : forfeit_per_idle_day = 5)
  (h4 : total_amount = 500) :
  ∃ (idle_days : ℕ), 
    idle_days = 38 ∧ 
    idle_days + (total_days - idle_days) = total_days ∧
    pay_per_working_day * (total_days - idle_days) - forfeit_per_idle_day * idle_days = total_amount :=
by
  sorry

end worker_idle_days_l3458_345851


namespace gcd_140_396_l3458_345823

theorem gcd_140_396 : Nat.gcd 140 396 = 4 := by
  sorry

end gcd_140_396_l3458_345823


namespace exponent_multiplication_l3458_345883

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l3458_345883


namespace female_officers_on_duty_percentage_l3458_345893

/-- Calculates the percentage of female officers on duty -/
def percentage_female_officers_on_duty (total_on_duty : ℕ) (female_ratio_on_duty : ℚ) (total_female_officers : ℕ) : ℚ :=
  (female_ratio_on_duty * total_on_duty : ℚ) / total_female_officers * 100

/-- Theorem stating that the percentage of female officers on duty is 20% -/
theorem female_officers_on_duty_percentage 
  (total_on_duty : ℕ) 
  (female_ratio_on_duty : ℚ) 
  (total_female_officers : ℕ) 
  (h1 : total_on_duty = 100)
  (h2 : female_ratio_on_duty = 1/2)
  (h3 : total_female_officers = 250) :
  percentage_female_officers_on_duty total_on_duty female_ratio_on_duty total_female_officers = 20 :=
sorry

end female_officers_on_duty_percentage_l3458_345893


namespace vanessa_missed_days_l3458_345831

theorem vanessa_missed_days (total : ℕ) (vanessa_mike : ℕ) (mike_sarah : ℕ)
  (h1 : total = 17)
  (h2 : vanessa_mike = 14)
  (h3 : mike_sarah = 12) :
  ∃ (vanessa mike sarah : ℕ),
    vanessa + mike + sarah = total ∧
    vanessa + mike = vanessa_mike ∧
    mike + sarah = mike_sarah ∧
    vanessa = 5 := by
  sorry

end vanessa_missed_days_l3458_345831


namespace sum_and_ratio_to_difference_l3458_345871

theorem sum_and_ratio_to_difference (a b : ℝ) 
  (h1 : a + b = 500) 
  (h2 : a / b = 0.8) : 
  b - a = 100 / 1.8 := by
sorry

end sum_and_ratio_to_difference_l3458_345871


namespace product_set_sum_l3458_345884

theorem product_set_sum (a₁ a₂ a₃ a₄ : ℚ) :
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Finset ℚ) =
  {-24, -2, -3/2, -1/8, 1, 3} →
  (a₁ + a₂ + a₃ + a₄ = 9/4) ∨ (a₁ + a₂ + a₃ + a₄ = -9/4) := by
  sorry

end product_set_sum_l3458_345884


namespace john_ray_difference_l3458_345845

/-- The number of chickens each person took -/
structure ChickenCount where
  john : ℕ
  mary : ℕ
  ray : ℕ

/-- The conditions of the chicken distribution -/
def valid_distribution (c : ChickenCount) : Prop :=
  c.john = c.mary + 5 ∧
  c.ray = c.mary - 6 ∧
  c.ray = 10

/-- The theorem stating the difference between John's and Ray's chicken count -/
theorem john_ray_difference (c : ChickenCount) (h : valid_distribution c) : 
  c.john - c.ray = 11 := by
  sorry

end john_ray_difference_l3458_345845


namespace no_rational_roots_odd_coefficients_l3458_345803

theorem no_rational_roots_odd_coefficients (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := by
  sorry

end no_rational_roots_odd_coefficients_l3458_345803


namespace sqrt_seven_to_sixth_l3458_345865

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_sixth_l3458_345865


namespace no_identical_lines_l3458_345836

-- Define the equations of the lines
def line1 (a d x y : ℝ) : Prop := 4 * x + a * y + d = 0
def line2 (d x y : ℝ) : Prop := d * x - 3 * y + 15 = 0

-- Define what it means for the lines to be identical
def identical_lines (a d : ℝ) : Prop :=
  ∀ x y : ℝ, line1 a d x y ↔ line2 d x y

-- Theorem statement
theorem no_identical_lines : ¬∃ a d : ℝ, identical_lines a d :=
sorry

end no_identical_lines_l3458_345836


namespace cubic_root_sum_l3458_345874

theorem cubic_root_sum (a b c : ℕ+) :
  let x : ℝ := (Real.rpow a (1/3 : ℝ) + Real.rpow b (1/3 : ℝ) + 2) / c
  27 * x^3 - 6 * x^2 - 6 * x - 2 = 0 →
  a + b + c = 75 := by
  sorry

end cubic_root_sum_l3458_345874


namespace blue_garden_yield_l3458_345892

/-- Calculates the expected potato yield from a rectangular garden --/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (feet_per_step : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (length_steps : ℝ) * feet_per_step * (width_steps : ℝ) * feet_per_step * yield_per_sqft

theorem blue_garden_yield :
  expected_potato_yield 18 25 3 (3/4) = 3037.5 := by
  sorry

end blue_garden_yield_l3458_345892


namespace volume_cylinder_from_square_rotation_l3458_345818

/-- The volume of a cylinder formed by rotating a square about its horizontal line of symmetry. -/
theorem volume_cylinder_from_square_rotation (side_length : ℝ) (h_positive : side_length > 0) :
  let radius : ℝ := side_length / 2
  let height : ℝ := side_length
  let volume : ℝ := π * radius ^ 2 * height
  side_length = 10 → volume = 250 * π := by sorry

end volume_cylinder_from_square_rotation_l3458_345818


namespace probability_theorem_l3458_345843

def team_sizes : List Nat := [6, 9, 10]
def co_captains_per_team : Nat := 3
def members_selected : Nat := 3

def probability_all_co_captains (sizes : List Nat) (co_captains : Nat) (selected : Nat) : ℚ :=
  let team_probabilities := sizes.map (λ n => (co_captains.factorial * (n - co_captains).choose (selected - co_captains)) / n.choose selected)
  (1 / sizes.length) * team_probabilities.sum

theorem probability_theorem :
  probability_all_co_captains team_sizes co_captains_per_team members_selected = 177 / 12600 := by
  sorry

end probability_theorem_l3458_345843


namespace geometric_to_arithmetic_sequence_l3458_345877

theorem geometric_to_arithmetic_sequence (a b c : ℝ) (x y z : ℝ) :
  (10 ^ a = x) →
  (10 ^ b = y) →
  (10 ^ c = z) →
  (∃ r : ℝ, y = x * r ∧ z = y * r) →  -- geometric sequence condition
  ∃ d : ℝ, b - a = d ∧ c - b = d  -- arithmetic sequence condition
:= by sorry

end geometric_to_arithmetic_sequence_l3458_345877


namespace razorback_tshirt_sales_l3458_345861

theorem razorback_tshirt_sales 
  (revenue_per_tshirt : ℕ) 
  (total_tshirts : ℕ) 
  (revenue_one_game : ℕ) 
  (h1 : revenue_per_tshirt = 98)
  (h2 : total_tshirts = 163)
  (h3 : revenue_one_game = 8722) :
  ∃ (arkansas_tshirts : ℕ), 
    arkansas_tshirts * revenue_per_tshirt = revenue_one_game ∧
    arkansas_tshirts ≤ total_tshirts ∧
    arkansas_tshirts = 89 :=
by sorry

end razorback_tshirt_sales_l3458_345861


namespace polynomial_multiplication_l3458_345885

theorem polynomial_multiplication :
  ∀ x : ℝ, (5 * x + 3) * (2 * x - 4 + x^2) = 5 * x^3 + 13 * x^2 - 14 * x - 12 := by
  sorry

end polynomial_multiplication_l3458_345885


namespace reflection_sum_coordinates_l3458_345800

/-- Given a point C with coordinates (x, 8), when reflected over the y-axis to point D,
    the sum of all coordinate values of C and D is 16. -/
theorem reflection_sum_coordinates (x : ℝ) : 
  let C : ℝ × ℝ := (x, 8)
  let D : ℝ × ℝ := (-x, 8)
  x + 8 + (-x) + 8 = 16 := by sorry

end reflection_sum_coordinates_l3458_345800


namespace square_plus_reciprocal_square_l3458_345811

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 := by
  sorry

end square_plus_reciprocal_square_l3458_345811


namespace nonagon_side_length_l3458_345895

/-- A regular nonagon with perimeter 171 cm has sides of length 19 cm -/
theorem nonagon_side_length : ∀ (perimeter side_length : ℝ),
  perimeter = 171 →
  side_length * 9 = perimeter →
  side_length = 19 :=
by
  sorry

end nonagon_side_length_l3458_345895


namespace expression_equals_x_power_44_l3458_345860

def numerator_sequence (n : ℕ) : ℕ := 2 * n + 1

def denominator_sequence (n : ℕ) : ℕ := 4 * n

def numerator_sum (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => numerator_sequence (i + 1))

def denominator_sum (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => denominator_sequence (i + 1))

theorem expression_equals_x_power_44 (x : ℝ) (hx : x = 3) :
  (x ^ numerator_sum 14) / (x ^ denominator_sum 9) = x ^ 44 := by
  sorry

end expression_equals_x_power_44_l3458_345860


namespace teaching_position_allocation_l3458_345830

theorem teaching_position_allocation :
  let total_positions : ℕ := 8
  let num_schools : ℕ := 3
  let min_positions_per_school : ℕ := 1
  let min_positions_school_a : ℕ := 2
  let remaining_positions : ℕ := total_positions - (min_positions_school_a + min_positions_per_school * (num_schools - 1))
  (remaining_positions.choose (num_schools - 1)) = 6 :=
by sorry

end teaching_position_allocation_l3458_345830


namespace final_output_is_four_l3458_345828

def program_output (initial : ℕ) (increment1 : ℕ) (increment2 : ℕ) : ℕ :=
  initial + increment1 + increment2

theorem final_output_is_four :
  program_output 1 1 2 = 4 := by
  sorry

end final_output_is_four_l3458_345828


namespace import_tax_percentage_l3458_345810

theorem import_tax_percentage 
  (total_value : ℝ) 
  (non_taxed_portion : ℝ) 
  (import_tax_amount : ℝ) 
  (h1 : total_value = 2580) 
  (h2 : non_taxed_portion = 1000) 
  (h3 : import_tax_amount = 110.60) : 
  (import_tax_amount / (total_value - non_taxed_portion)) * 100 = 7 := by
  sorry

end import_tax_percentage_l3458_345810


namespace valid_solution_l3458_345859

/-- A number is a perfect square if it's the square of an integer. -/
def IsPerfectSquare (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

/-- A function to check if a number is prime. -/
def IsPrime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

/-- The theorem stating that 900 is a valid solution for n. -/
theorem valid_solution :
  ∃ m : ℕ, IsPerfectSquare m ∧ IsPerfectSquare 900 ∧ IsPrime (m - 900) :=
by sorry

end valid_solution_l3458_345859


namespace trigonometric_expression_equality_l3458_345862

theorem trigonometric_expression_equality : 
  4 * Real.sin (80 * π / 180) - Real.cos (10 * π / 180) / Real.sin (10 * π / 180) = -Real.sqrt 3 := by
  sorry

end trigonometric_expression_equality_l3458_345862


namespace kimikos_age_l3458_345849

theorem kimikos_age (kimiko omi arlette : ℝ) 
  (h1 : omi = 2 * kimiko)
  (h2 : arlette = 3/4 * kimiko)
  (h3 : (kimiko + omi + arlette) / 3 = 35) :
  kimiko = 28 := by
  sorry

end kimikos_age_l3458_345849


namespace cost_per_candy_bar_l3458_345850

-- Define the given conditions
def boxes_sold : ℕ := 5
def candy_bars_per_box : ℕ := 10
def selling_price_per_bar : ℚ := 3/2  -- $1.50 as a rational number
def total_profit : ℚ := 25

-- Define the theorem
theorem cost_per_candy_bar :
  let total_bars := boxes_sold * candy_bars_per_box
  let total_revenue := total_bars * selling_price_per_bar
  let total_cost := total_revenue - total_profit
  total_cost / total_bars = 1 := by sorry

end cost_per_candy_bar_l3458_345850


namespace image_of_two_is_five_l3458_345809

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- State the theorem
theorem image_of_two_is_five : f 2 = 5 := by sorry

end image_of_two_is_five_l3458_345809


namespace absolute_value_integral_l3458_345817

theorem absolute_value_integral : ∫ x in (-1)..2, |x| = 5/2 := by sorry

end absolute_value_integral_l3458_345817


namespace max_value_constraint_l3458_345816

theorem max_value_constraint (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) :
  x + 2*y + 2*z ≤ 15 := by sorry

end max_value_constraint_l3458_345816


namespace neil_initial_games_l3458_345826

theorem neil_initial_games (henry_initial : ℕ) (games_given : ℕ) (neil_initial : ℕ) :
  henry_initial = 58 →
  games_given = 6 →
  henry_initial - games_given = 4 * (neil_initial + games_given) →
  neil_initial = 7 := by
sorry

end neil_initial_games_l3458_345826


namespace rainwater_farm_l3458_345815

theorem rainwater_farm (cows goats chickens : ℕ) : 
  cows = 9 →
  goats = 4 * cows →
  goats = 2 * chickens →
  chickens = 18 := by
sorry

end rainwater_farm_l3458_345815


namespace cubic_function_uniqueness_l3458_345899

/-- Given a cubic function f(x) = ax^3 - 3x^2 + x + b with a ≠ 0, 
    if the tangent line at x = 1 is 2x + y + 1 = 0, 
    then f(x) = x^3 - 3x^2 + x - 2 -/
theorem cubic_function_uniqueness (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - 3 * x^2 + x + b
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 - 6 * x + 1
  (f' 1 = -2 ∧ f 1 = -3) → f = λ x ↦ x^3 - 3 * x^2 + x - 2 := by
  sorry

end cubic_function_uniqueness_l3458_345899


namespace complex_product_simplification_l3458_345864

theorem complex_product_simplification :
  let i : ℂ := Complex.I
  ((4 - 3*i) - (2 + 5*i)) * (2*i) = 16 + 4*i := by sorry

end complex_product_simplification_l3458_345864


namespace two_lines_theorem_l3458_345863

/-- Two lines in a 2D plane -/
structure TwoLines where
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → ℝ → Prop

/-- The given lines -/
def given_lines : TwoLines where
  l₁ := fun x y ↦ 2 * x + y + 4 = 0
  l₂ := fun a x y ↦ a * x + 4 * y + 1 = 0

/-- Perpendicularity condition -/
def perpendicular (lines : TwoLines) (a : ℝ) : Prop :=
  ∃ x y, lines.l₁ x y ∧ lines.l₂ a x y

/-- Parallelism condition -/
def parallel (lines : TwoLines) (a : ℝ) : Prop :=
  ∀ x y, lines.l₁ x y ↔ ∃ k, lines.l₂ a (x + k) (y + k)

/-- Main theorem -/
theorem two_lines_theorem (lines : TwoLines) :
  (∃ a, perpendicular lines a → 
    ∃ x y, lines.l₁ x y ∧ lines.l₂ a x y ∧ x = -3/2 ∧ y = -1) ∧
  (∃ a, parallel lines a → 
    ∃ d, d = (3 * Real.sqrt 5) / 4 ∧ 
      ∀ x₁ y₁ x₂ y₂, lines.l₁ x₁ y₁ → lines.l₂ a x₂ y₂ → 
        ((x₂ - x₁)^2 + (y₂ - y₁)^2 : ℝ) ≥ d^2) :=
by sorry

end two_lines_theorem_l3458_345863


namespace correct_survey_order_l3458_345805

/-- Represents the steps in conducting a survey --/
inductive SurveyStep
  | CreateQuestionnaire
  | OrganizeResults
  | DrawPieChart
  | AnalyzeResults

/-- Defines the correct order of survey steps --/
def correct_order : List SurveyStep :=
  [SurveyStep.CreateQuestionnaire, SurveyStep.OrganizeResults, 
   SurveyStep.DrawPieChart, SurveyStep.AnalyzeResults]

/-- Theorem stating that the defined order is correct for determining the most popular club activity --/
theorem correct_survey_order : 
  correct_order = [SurveyStep.CreateQuestionnaire, SurveyStep.OrganizeResults, 
                   SurveyStep.DrawPieChart, SurveyStep.AnalyzeResults] := by
  sorry

end correct_survey_order_l3458_345805


namespace price_reduction_equation_l3458_345804

/-- Given an original price and a final price after two equal percentage reductions,
    this theorem states the equation relating the reduction percentage to the prices. -/
theorem price_reduction_equation (original_price final_price : ℝ) (x : ℝ) 
  (h1 : original_price = 60)
  (h2 : final_price = 48.6)
  (h3 : x > 0 ∧ x < 1) :
  original_price * (1 - x)^2 = final_price := by
sorry

end price_reduction_equation_l3458_345804


namespace opposite_solutions_value_of_m_l3458_345829

theorem opposite_solutions_value_of_m :
  ∀ (x y m : ℝ),
  (3 * x + 4 * y = 7) →
  (5 * x - 4 * y = m) →
  (x + y = 0) →
  m = -63 := by
sorry

end opposite_solutions_value_of_m_l3458_345829


namespace geometric_sequence_third_term_l3458_345869

/-- Given a geometric sequence {aₙ} with sum of first n terms Sₙ, 
    if S₆/S₃ = -19/8 and a₄ - a₂ = -15/8, then a₃ = 9/4 -/
theorem geometric_sequence_third_term
  (a : ℕ → ℚ)  -- The geometric sequence
  (S : ℕ → ℚ)  -- The sum function
  (h1 : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1)))  -- Definition of sum for geometric sequence
  (h2 : S 6 / S 3 = -19/8)  -- Given condition
  (h3 : a 4 - a 2 = -15/8)  -- Given condition
  : a 3 = 9/4 :=
sorry

end geometric_sequence_third_term_l3458_345869


namespace sum_bounds_and_range_l3458_345876

open Real

theorem sum_bounds_and_range (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let S := a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)
  (1 < S ∧ S < 2) ∧ ∀ x, 1 < x → x < 2 → ∃ a' b' c' d', 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 
    x = a' / (a' + b' + d') + b' / (a' + b' + c') + c' / (b' + c' + d') + d' / (a' + c' + d') :=
by sorry

end sum_bounds_and_range_l3458_345876


namespace intersection_of_A_and_B_l3458_345872

def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

theorem intersection_of_A_and_B : 
  A ∩ B = {x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4} := by sorry

end intersection_of_A_and_B_l3458_345872


namespace absolute_value_five_minus_sqrt_eleven_l3458_345856

theorem absolute_value_five_minus_sqrt_eleven : |5 - Real.sqrt 11| = 1.683 := by
  sorry

end absolute_value_five_minus_sqrt_eleven_l3458_345856


namespace number_operation_l3458_345834

theorem number_operation (x : ℝ) : (x - 5) / 7 = 7 → (x - 4) / 10 = 5 := by
  sorry

end number_operation_l3458_345834


namespace A_closed_under_mult_l3458_345896

/-- The set A of quadratic forms over integers -/
def A : Set ℤ := {n : ℤ | ∃ (a b k : ℤ), n = a^2 + k*a*b + b^2}

/-- A is closed under multiplication -/
theorem A_closed_under_mult :
  ∀ (x y : ℤ), x ∈ A → y ∈ A → (x * y) ∈ A := by
  sorry

end A_closed_under_mult_l3458_345896


namespace symmetric_points_difference_l3458_345838

/-- Two points are symmetric with respect to the origin if their coordinates are negations of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

theorem symmetric_points_difference (a b : ℝ) :
  let A : ℝ × ℝ := (-2, b)
  let B : ℝ × ℝ := (a, 3)
  symmetric_wrt_origin A B → a - b = 5 := by
  sorry

end symmetric_points_difference_l3458_345838


namespace loss_percentage_calculation_l3458_345840

theorem loss_percentage_calculation (CP : ℝ) :
  CP > 0 ∧ 
  240 = CP * (1 + 0.20) ∧ 
  170 < CP 
  → 
  (CP - 170) / CP * 100 = 15 := by
sorry

end loss_percentage_calculation_l3458_345840


namespace proctoring_arrangements_l3458_345808

theorem proctoring_arrangements (n : ℕ) (k : ℕ) (h : n = 12 ∧ k = 8) :
  (Nat.choose n k) * ((n - k) * (n - k - 1)) = 4455 :=
sorry

end proctoring_arrangements_l3458_345808


namespace senate_democrats_count_l3458_345848

/-- Given the conditions of the House of Representatives and Senate composition,
    prove that the number of Democrats in the Senate is 55. -/
theorem senate_democrats_count : 
  ∀ (house_total house_dem house_rep senate_total senate_dem senate_rep : ℕ),
  house_total = 434 →
  house_total = house_dem + house_rep →
  house_rep = house_dem + 30 →
  senate_total = 100 →
  senate_total = senate_dem + senate_rep →
  5 * senate_rep = 4 * senate_dem →
  senate_dem = 55 := by
  sorry

end senate_democrats_count_l3458_345848
