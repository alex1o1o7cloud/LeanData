import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_pairs_l648_64842

theorem count_ordered_pairs : 
  let S := {p : ℕ × ℕ | p.fst + p.snd = 50 ∧ p.fst > 0 ∧ p.snd > 0}
  Finset.card (Finset.filter (fun p => p.fst + p.snd = 50 ∧ p.fst > 0 ∧ p.snd > 0) (Finset.range 50 ×ˢ Finset.range 50)) = 49 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_pairs_l648_64842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_c_zero_c_range_for_given_f_range_l648_64834

-- Define the function f
noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ c then x^2 + x
  else if c < x ∧ x ≤ 3 then 1/x
  else 0  -- Default value for x outside the defined intervals

-- Theorem for the first case
theorem range_when_c_zero :
  ∀ y : ℝ, y ≥ -1/4 → ∃ x : ℝ, f 0 x = y :=
by
  sorry

-- Theorem for the second case
theorem c_range_for_given_f_range (c : ℝ) :
  (∀ y : ℝ, (∃ x : ℝ, f c x = y) ↔ -1/4 ≤ y ∧ y ≤ 2) →
  1/2 ≤ c ∧ c ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_c_zero_c_range_for_given_f_range_l648_64834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_empty_l648_64830

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = 2 * x + 3}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = 4 * x + 1}

theorem A_intersect_B_empty : A ∩ B = ∅ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_empty_l648_64830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l648_64883

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x - Real.pi / 3)

theorem function_properties :
  -- Conditions
  ∃ (ω φ : ℝ),
    ω > 0 ∧
    -Real.pi/2 < φ ∧ φ < 0 ∧
    (∀ x, f x = 2 * Real.sin (ω * x + φ)) ∧
    Real.tan φ = -Real.sqrt 3 ∧
    (∀ x₁ x₂, |f x₁ - f x₂| = 4 → |x₁ - x₂| ≥ Real.pi/3) ∧
    (∃ x₁ x₂, |f x₁ - f x₂| = 4 ∧ |x₁ - x₂| = Real.pi/3) →
  -- Conclusions
  (∀ x, f x = 2 * Real.sin (3 * x - Real.pi / 3)) ∧
  (∀ m, (∀ x, 0 ≤ x ∧ x ≤ Real.pi/6 → m * f x + 2 * m ≥ f x) ↔ m ≥ 1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l648_64883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_upright_equals_flat_depth_l648_64840

/-- Represents a right cylindrical water tank -/
structure WaterTank where
  height : ℝ
  baseDiameter : ℝ

/-- Calculates the volume of water in the tank when lying flat -/
noncomputable def volumeWhenFlat (tank : WaterTank) (flatDepth : ℝ) : ℝ :=
  (Real.pi * (tank.baseDiameter / 2)^2) * flatDepth

/-- Calculates the total volume of the tank -/
noncomputable def totalVolume (tank : WaterTank) : ℝ :=
  (Real.pi * (tank.baseDiameter / 2)^2) * tank.height

/-- Theorem: Water depth when upright equals flat depth for specific tank dimensions -/
theorem water_depth_upright_equals_flat_depth 
  (tank : WaterTank) 
  (flatDepth : ℝ) 
  (h1 : tank.height = 18) 
  (h2 : tank.baseDiameter = 6) 
  (h3 : flatDepth = 4) : 
  (volumeWhenFlat tank flatDepth) / (totalVolume tank) * tank.height = flatDepth := by
  sorry

#check water_depth_upright_equals_flat_depth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_upright_equals_flat_depth_l648_64840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_striped_has_eight_legs_l648_64877

-- Define the types of octopuses
inductive Octopus : Type
  | Green : Octopus
  | DarkBlue : Octopus
  | Violet : Octopus
  | Striped : Octopus

-- Define a function to represent the number of legs for each octopus
def legs : Octopus → ℕ := sorry

-- Define a function to determine if an octopus tells the truth
def tellsTruth (o : Octopus) : Prop := Even (legs o)

-- Define the statements made by each octopus
axiom green_statement : legs Octopus.Green = 8 ∧ legs Octopus.DarkBlue = 6
axiom darkblue_statement : legs Octopus.DarkBlue = 8 ∧ legs Octopus.Green = 7
axiom violet_statement : legs Octopus.DarkBlue = 8 ∧ legs Octopus.Violet = 9
axiom striped_statement : (∀ o : Octopus, o ≠ Octopus.Striped → legs o ≠ 8) ∧ legs Octopus.Striped = 8

-- Theorem to prove
theorem striped_has_eight_legs :
  legs Octopus.Striped = 8 ∧
  (∀ o : Octopus, o ≠ Octopus.Striped → legs o ≠ 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_striped_has_eight_legs_l648_64877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_range_l648_64856

theorem triangle_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ x : ℝ, (x > |a - b| ∧ x < a + b) ↔ x > 0 ∧ x + a > b ∧ x + b > a :=
by sorry

theorem third_side_range :
  ∀ x : ℝ, (x > |7 - 3| ∧ x < 7 + 3) ↔ (4 < x ∧ x < 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_range_l648_64856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tripleSum_eq_reciprocal_1836_l648_64886

/-- The sum of 1 / (2^a * 5^b * 7^c) over all positive integers a, b, c where 1 ≤ a < b < c -/
noncomputable def tripleSum : ℝ :=
  ∑' (a : ℕ), ∑' (b : ℕ), ∑' (c : ℕ),
    if 1 ≤ a ∧ a < b ∧ b < c then (1 : ℝ) / ((2 : ℝ)^a * (5 : ℝ)^b * (7 : ℝ)^c) else 0

/-- The sum of 1 / (2^a * 5^b * 7^c) over all positive integers a, b, c where 1 ≤ a < b < c equals 1/1836 -/
theorem tripleSum_eq_reciprocal_1836 : tripleSum = (1 : ℝ) / 1836 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tripleSum_eq_reciprocal_1836_l648_64886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_0_to_99_sum_of_digits_18_to_21_l648_64887

-- Define a function to calculate the sum of digits of a number
def sumOfDigits (n : Nat) : Nat := sorry

-- Define a function to calculate the sum of digits for a range of numbers
def sumOfDigitsRange (start finish : Nat) : Nat := sorry

-- Theorem stating that the sum of all digits from 0 to 99 is 900
theorem sum_of_digits_0_to_99 : sumOfDigitsRange 0 99 = 900 := by sorry

-- Additional theorem for the given condition in the problem
theorem sum_of_digits_18_to_21 : sumOfDigitsRange 18 21 = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_0_to_99_sum_of_digits_18_to_21_l648_64887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_side_length_l648_64857

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ
  area : ℝ

/-- Calculate the area of a trapezium -/
noncomputable def trapeziumArea (t : Trapezium) : ℝ :=
  (1/2) * (t.side1 + t.side2) * t.height

/-- Theorem: Given a trapezium with one side 20 cm, height 16 cm, and area 304 cm², the other side is 18 cm -/
theorem other_side_length (t : Trapezium) 
    (h1 : t.side1 = 20)
    (h2 : t.height = 16)
    (h3 : t.area = 304)
    (h4 : t.area = trapeziumArea t) :
  t.side2 = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_side_length_l648_64857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_wig_cost_l648_64847

/-- Calculates the total cost of wigs for John's plays --/
def total_wig_cost (num_plays : ℕ) (acts_per_play : ℕ) (wigs_per_act : ℕ) 
  (wig_costs : List ℚ) (discount_rate : ℚ) (refund_rate : ℚ) : ℚ :=
  let play1_cost := wigs_per_act * acts_per_play * wig_costs[0]!
  let play2_cost := wigs_per_act * acts_per_play * wig_costs[1]!
  let play3_cost := wigs_per_act * acts_per_play * wig_costs[2]!
  let play4_cost := wigs_per_act * acts_per_play * wig_costs[3]!
  let play2_discount := play2_cost * discount_rate
  let play4_refund := play4_cost * refund_rate
  play1_cost + play3_cost + (play4_cost - play4_refund) - (play2_cost - play2_discount)

/-- Theorem stating the total cost of wigs for John's plays --/
theorem john_wig_cost : 
  total_wig_cost 4 8 3 [5, 6, 7, 8] (25/100) (10/100) = 352.8 := by
  sorry

#eval total_wig_cost 4 8 3 [5, 6, 7, 8] (25/100) (10/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_wig_cost_l648_64847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bernoulli_equation_solution_l648_64837

/-- Bernoulli equation solution -/
theorem bernoulli_equation_solution (x : ℝ) (C₁ : ℝ) (h : x ≠ 0) :
  let y : ℝ → ℝ := λ x => (3 / (2 * x) + C₁ / x^3) ^ (1/3)
  deriv y x + y x / x = 1 / (x^2 * (y x)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bernoulli_equation_solution_l648_64837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_m_properties_l648_64817

/-- Circle M passing through two points and with center on a line -/
structure CircleM where
  -- Center of the circle is on the line x + y - 2 = 0
  center_on_line : ∀ (a b : ℝ), a + b - 2 = 0 → (a, b) ∈ Set.range (λ t : ℝ ↦ (t, 2 - t))
  -- Circle passes through (1, -1) and (-1, 1)
  passes_through_points : ∀ (a b r : ℝ), (1 - a)^2 + (-1 - b)^2 = r^2 ∧ (-1 - a)^2 + (1 - b)^2 = r^2

/-- Theorem about the equation and properties of Circle M -/
theorem circle_m_properties (m : CircleM) :
  -- The equation of circle M is (x-1)^2 + (y-1)^2 = 4
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 4 ↔ (x, y) ∈ Set.range (λ t : ℝ ↦ (Real.cos t + 1, Real.sin t + 1))) ∧
  -- For any point P(x,y) on circle M, (4-√7)/3 ≤ (y+3)/(x+3) ≤ (4+√7)/3
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 4 →
    (4 - Real.sqrt 7) / 3 ≤ (y + 3) / (x + 3) ∧ (y + 3) / (x + 3) ≤ (4 + Real.sqrt 7) / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_m_properties_l648_64817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l648_64838

def S (n : ℕ) : ℤ := n^2 - 2*n + 2

def a : ℕ → ℤ
| 0 => 1  -- Added case for 0
| 1 => 1
| (n+2) => 2*(n+2) - 3

theorem sequence_properties :
  (a 1 = 1) ∧
  (∀ n : ℕ, n ≥ 2 → a n = 2*n - 3) ∧
  (∀ n : ℕ, n ≥ 1 → S n = S (n-1) + a n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l648_64838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l648_64805

theorem trig_problem (θ : ℝ) (h1 : θ ∈ Set.Ioo (π / 2) π) (h2 : Real.sin θ = 3 / 5) :
  (Real.tan θ = -3 / 4) ∧ (Real.cos (θ + π / 3) = -(4 + 3 * Real.sqrt 3) / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l648_64805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l648_64844

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Define the base case for 0
  | n + 1 => sequence_a n + 2 * (n + 1)

theorem a_100_value : sequence_a 99 = 9902 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l648_64844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_weekly_profit_l648_64807

/-- Represents the weekly sales and profit model for a component -/
structure ComponentSales where
  cost : ℝ                    -- Cost per piece in yuan
  initial_price : ℝ           -- Initial selling price per piece in yuan
  initial_volume : ℝ          -- Initial weekly sales volume
  volume_increase_rate : ℝ    -- Increase in volume per yuan of price decrease
  price_decrease : ℝ          -- Amount of price decrease in yuan (x)

/-- Calculates the weekly sales volume given a price decrease -/
def sales_volume (model : ComponentSales) : ℝ :=
  model.initial_volume + model.volume_increase_rate * model.price_decrease

/-- Calculates the weekly profit given a price decrease -/
def weekly_profit (model : ComponentSales) : ℝ :=
  let new_price := model.initial_price - model.price_decrease
  let volume := sales_volume model
  (new_price - model.cost) * volume

/-- The main theorem stating the maximum weekly profit -/
theorem max_weekly_profit (model : ComponentSales) 
  (h1 : model.cost = 30)
  (h2 : model.initial_price = 80)
  (h3 : model.initial_volume = 600)
  (h4 : model.volume_increase_rate = 15)
  (h5 : model.price_decrease = 4 ∨ model.price_decrease = 6) :
  weekly_profit model = 30360 := by
  sorry

#check max_weekly_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_weekly_profit_l648_64807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_b_equal_one_l648_64890

noncomputable def f (x a b : ℝ) : ℝ :=
  if x < 0 then x + a else b * x - 1

theorem odd_function_implies_a_b_equal_one (a b : ℝ) :
  (∀ x, f (-x) a b = -f x a b) → a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_b_equal_one_l648_64890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baba_yaga_journey_baba_yaga_solution_l648_64878

/-- Represents the problem of Baba Yaga's journey to Bald Mountain -/
structure BabaYagaJourney where
  T : ℝ  -- Time required in hours for the journey
  d : ℝ  -- Distance to Bald Mountain in km

/-- The conditions of Baba Yaga's journey -/
def journey_conditions (j : BabaYagaJourney) : Prop :=
  j.d = 50 * (j.T + 2) ∧ j.d = 150 * (j.T - 2)

/-- The theorem stating the correct departure time and speed -/
theorem baba_yaga_journey (j : BabaYagaJourney) 
  (h : journey_conditions j) : 
  j.T = 4 ∧ j.d / j.T = 75 := by
  sorry

/-- The departure time in hours before midnight -/
noncomputable def departure_time (j : BabaYagaJourney) : ℝ := j.T

/-- The speed of the broom in km/h -/
noncomputable def broom_speed (j : BabaYagaJourney) : ℝ := j.d / j.T

/-- The main theorem stating the correct departure time and broom speed -/
theorem baba_yaga_solution (j : BabaYagaJourney) 
  (h : journey_conditions j) : 
  departure_time j = 4 ∧ broom_speed j = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baba_yaga_journey_baba_yaga_solution_l648_64878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_side_l648_64833

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

theorem triangle_area_and_side (t : Triangle) 
  (h1 : 3 * t.a * Real.sin t.C = 4 * t.c * Real.cos t.A) 
  (h2 : t.b * t.c * Real.cos t.A = 3) : 
  area t = 2 ∧ (t.c = 1 → t.a = 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_side_l648_64833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l648_64804

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x + 1/(x^2 + 1/x)

theorem f_minimum_value (x : ℝ) (h : x > 0) : f x ≥ 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l648_64804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_and_equation_l648_64826

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define point P₀
def P₀ : ℝ × ℝ := (-1, 2)

-- Define chord AB passing through P₀
def chordAB (α : ℝ) (x y : ℝ) : Prop :=
  y - P₀.2 = Real.tan α * (x - P₀.1)

theorem chord_length_and_equation :
  -- Part 1: Length of AB when α = 3π/4
  (∀ x y : ℝ, circle_eq x y ∧ chordAB (3 * Real.pi / 4) x y →
    ∃ A B : ℝ × ℝ, (A.1 - B.1)^2 + (A.2 - B.2)^2 = 30) ∧
  -- Part 2: Equation of AB when P₀ bisects it
  (∀ x y : ℝ, circle_eq x y ∧ 
    (∃ A B : ℝ × ℝ, ∃ α : ℝ, chordAB α x y ∧ 
      P₀.1 = (A.1 + B.1) / 2 ∧ P₀.2 = (A.2 + B.2) / 2) →
    x - 2*y + 5 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_and_equation_l648_64826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l648_64867

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 1
  | n + 1 => 3 * a n + 4

/-- Theorem stating the closed form of a_n -/
theorem a_closed_form (n : ℕ) : a n = 3^n - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l648_64867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_side_length_of_squares_l648_64885

theorem average_side_length_of_squares : 
  let areas : List ℝ := [25, 64, 100, 144]
  let side_lengths := areas.map Real.sqrt
  (side_lengths.sum / side_lengths.length) = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_side_length_of_squares_l648_64885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_formula_l648_64831

/-- The volume of a tetrahedron with congruent faces and side lengths a, b, and c -/
noncomputable def tetrahedronVolume (a b c : ℝ) : ℝ :=
  (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (a^2 + c^2 - b^2) * (b^2 + c^2 - a^2))

/-- Predicate to represent that a tetrahedron has congruent faces with side lengths a, b, and c -/
def CongruentFaces (a b c : ℝ) : Prop := sorry

/-- Function to represent a tetrahedron with side lengths a, b, and c -/
def Tetrahedron (a b c : ℝ) : Type := sorry

/-- Function to calculate the volume of a tetrahedron -/
noncomputable def Volume (t : Type) : ℝ := sorry

/-- Theorem: The volume of a tetrahedron with congruent faces and side lengths a, b, and c
    is given by the formula V = (1 / (6 * √2)) * √((a² + b² - c²)(a² + c² - b²)(b² + c² - a²)) -/
theorem tetrahedron_volume_formula (a b c : ℝ) (h : CongruentFaces a b c) :
  Volume (Tetrahedron a b c) = tetrahedronVolume a b c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_formula_l648_64831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l648_64896

theorem cos_beta_value (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.sin α = 4 / 5) (h4 : Real.cos (α + β) = -12 / 13) : Real.cos β = -16 / 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l648_64896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_mixture_volume_constant_l648_64812

/-- Represents the properties of an oil mixture -/
structure OilMixture where
  hot_volume : ℝ
  cold_volume : ℝ
  hot_temp : ℝ
  cold_temp : ℝ
  expansion_coeff : ℝ

/-- Calculates the equilibrium temperature of the oil mixture -/
noncomputable def equilibrium_temp (mix : OilMixture) : ℝ :=
  (mix.hot_volume * mix.hot_temp + mix.cold_volume * mix.cold_temp) / (mix.hot_volume + mix.cold_volume)

/-- Theorem: The final volume of the oil mixture remains constant after reaching thermal equilibrium -/
theorem oil_mixture_volume_constant (mix : OilMixture) 
  (h1 : mix.hot_volume > 0)
  (h2 : mix.cold_volume > 0)
  (h3 : mix.hot_temp > mix.cold_temp)
  (h4 : mix.expansion_coeff > 0) :
  let init_volume := mix.hot_volume + mix.cold_volume
  let eq_temp := equilibrium_temp mix
  let final_volume := init_volume * (1 + mix.expansion_coeff * (eq_temp - mix.cold_temp))
  final_volume = init_volume :=
by
  sorry

#check oil_mixture_volume_constant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_mixture_volume_constant_l648_64812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inscribed_iff_opposite_angles_sum_180_l648_64850

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the property of being inscribed in a circle
def is_inscribed (q : Quadrilateral) : Prop := sorry

-- Define the angle measure function
noncomputable def angle_measure (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_inscribed_iff_opposite_angles_sum_180 (q : Quadrilateral) :
  is_inscribed q ↔ 
    angle_measure q.A q.B q.C + angle_measure q.C q.D q.A = 180 ∧
    angle_measure q.B q.C q.D + angle_measure q.D q.A q.B = 180 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inscribed_iff_opposite_angles_sum_180_l648_64850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_bought_l648_64893

def apples : ℕ := 5
def apple_price : ℕ := 1
def orange_price : ℕ := 2
def total_spent : ℕ := 9

theorem oranges_bought : (total_spent - apples * apple_price) / orange_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_bought_l648_64893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_f_l648_64863

noncomputable def f (x : ℝ) : ℝ := -5 * Real.cos (2 * x + Real.pi / 3) + 4

theorem properties_of_f :
  (∃ A : ℝ, A > 0 ∧ ∀ x, |f x - 4| ≤ A ∧ ∃ x₀, |f x₀ - 4| = A) ∧
  (∀ x, f (x - Real.pi / 6) = -5 * Real.cos (2 * x) + 4) ∧
  (∀ x, f x - (-5 * Real.cos (2 * x + Real.pi / 3)) = 4) := by
  sorry

#check properties_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_f_l648_64863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_40_l648_64851

/-- Geometric sequence sum -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

/-- Conditions for the geometric sequence -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  sum_10 : geometric_sum a r 10 = 10
  sum_30 : geometric_sum a r 30 = 70

/-- Theorem: Given the conditions, S_40 is either 150 or 110 -/
theorem geometric_sequence_sum_40 (seq : GeometricSequence) :
  geometric_sum seq.a seq.r 40 = 150 ∨ geometric_sum seq.a seq.r 40 = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_40_l648_64851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_composite_as_sum_of_consecutive_odds_l648_64806

theorem even_composite_as_sum_of_consecutive_odds (m : Nat) :
  (∃ n : Nat, m = n * (2 * n.succ - 1)) ↔ m % 4 = 0 ∧ m > 4 := by
  sorry

#check even_composite_as_sum_of_consecutive_odds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_composite_as_sum_of_consecutive_odds_l648_64806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ratio_l648_64827

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Theorem statement
theorem min_distance_ratio (p₁ p₂ p₃ p₄ : Point) 
  (h : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄) :
  let dist_sum := distance p₁ p₂ + distance p₁ p₃ + distance p₁ p₄ + 
                  distance p₂ p₃ + distance p₂ p₄ + distance p₃ p₄
  let min_dist := min (min (min (min (min (distance p₁ p₂) (distance p₁ p₃)) (distance p₁ p₄)) 
                  (distance p₂ p₃)) (distance p₂ p₄)) (distance p₃ p₄)
  dist_sum / min_dist ≥ 5 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ratio_l648_64827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carls_dads_contribution_value_l648_64835

/-- Calculate the amount Carl's dad gave him to buy the coat -/
noncomputable def carls_dads_contribution (savings : ℝ) (coat_price_eur : ℝ) (discount_rate : ℝ) 
  (exchange_rate : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let discounted_price := coat_price_eur * (1 - discount_rate)
  let taxed_price := discounted_price * (1 + sales_tax_rate)
  let total_cost_usd := taxed_price / exchange_rate
  total_cost_usd - savings

/-- The amount Carl's dad gave him is $14.25 -/
theorem carls_dads_contribution_value : 
  carls_dads_contribution 235 220 0.1 0.85 0.07 = 14.25 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and might cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carls_dads_contribution_value_l648_64835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l648_64814

theorem inequality_solution (x : ℝ) : 
  x ≠ 5 → x * (x + 2) / (x - 5)^2 ≥ 15 → x ∈ Set.Icc 3.790 5 ∪ Set.Ioc 5 7.067 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l648_64814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_simplification_l648_64873

theorem absolute_value_simplification : 
  |(-(4^2) + 5 * (2⁻¹:ℝ))| = (13.5 : ℝ) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_simplification_l648_64873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l648_64882

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 4 = 1

noncomputable def distance_between_foci (h : ∃ x y, hyperbola_equation x y) : ℝ :=
  2 * Real.sqrt 29

theorem hyperbola_foci_distance :
  ∀ h : ∃ x y, hyperbola_equation x y,
  distance_between_foci h = 2 * Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l648_64882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l648_64874

/-- Represents a spinner with a given number of parts -/
structure Spinner :=
  (parts : ℕ)

/-- Calculates the probability of getting a multiple of 3 on a single spinner -/
def prob_multiple_of_three (s : Spinner) : ℚ :=
  (Finset.filter (fun n => n % 3 = 0) (Finset.range s.parts)).card / s.parts

/-- Calculates the probability of not getting a multiple of 3 on a single spinner -/
def prob_not_multiple_of_three (s : Spinner) : ℚ :=
  1 - prob_multiple_of_three s

/-- The probability of getting a product that is a multiple of 3 when spinning two spinners -/
def prob_product_multiple_of_three (s1 s2 : Spinner) : ℚ :=
  1 - (prob_not_multiple_of_three s1 * prob_not_multiple_of_three s2)

theorem spinner_probability :
  let spinner1 : Spinner := ⟨4⟩
  let spinner2 : Spinner := ⟨5⟩
  prob_product_multiple_of_three spinner1 spinner2 = 11 / 20 := by
  sorry

#eval prob_product_multiple_of_three ⟨4⟩ ⟨5⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l648_64874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebecca_pies_l648_64870

/-- The number of pies Rebecca bought -/
def P : ℕ := 2

/-- The total number of slices in all pies -/
def total_slices : ℕ := 8 * P

/-- The number of slices Rebecca ate initially -/
def rebecca_initial : ℕ := P

/-- The number of slices remaining after Rebecca's initial eating -/
def remaining_after_rebecca : ℕ := total_slices - rebecca_initial

/-- The number of slices eaten by family and friends -/
def family_friends : ℕ := (remaining_after_rebecca / 2 : ℕ)

/-- The number of slices Rebecca and her husband ate on Sunday -/
def sunday_slices : ℕ := 2

/-- The number of slices remaining at the end -/
def remaining_slices : ℕ := 5

theorem rebecca_pies :
  rebecca_initial + family_friends + sunday_slices + remaining_slices = total_slices ∧ P = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebecca_pies_l648_64870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_proof_l648_64810

noncomputable def A (S : List ℝ) : List ℝ :=
  List.zipWith (λ a b => (a + b) / 2) S (List.tail S)

noncomputable def A_power (S : List ℝ) : ℕ → List ℝ
  | 0 => S
  | n + 1 => A (A_power S n)

theorem x_value_proof (x : ℝ) (h1 : x > 0) :
  let S := List.range 51 |>.map (λ i => x^i)
  A_power S 50 = [1 / 2^25] →
  x = Real.sqrt 2 - 1 := by
  sorry

#check x_value_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_proof_l648_64810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l648_64822

/-- The function f(x) = x + 1/(x - 2) for x > 2 -/
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

/-- The domain of the function -/
def domain (x : ℝ) : Prop := x > 2

theorem minimum_value_of_f (a : ℝ) :
  domain a →
  (∀ x, domain x → f a ≤ f x) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l648_64822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l648_64858

theorem angle_sum_theorem (α β : Real) : 
  0 < α ∧ α < Real.pi/2 →
  0 < β ∧ β < Real.pi/2 →
  Real.sin α = 2 * Real.sqrt 5 / 5 →
  Real.sin β = 3 * Real.sqrt 10 / 10 →
  α + β = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l648_64858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l648_64869

def sequenceA (n : ℕ) : ℤ := n^2 - 5*n + 4

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sequenceA n ≥ -2) ∧
  sequenceA 7 = 18 ∧
  sequenceA 2 = -2 ∧
  sequenceA 3 = -2 := by
  sorry

#eval sequenceA 7
#eval sequenceA 2
#eval sequenceA 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l648_64869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l648_64889

-- Define set A
def A : Set ℝ := {x | -1 < x - 3 ∧ x - 3 ≤ 2}

-- Define set B
def B : Set ℝ := {x | 3 ≤ x ∧ x < 6}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 3 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l648_64889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_similar_right_triangles_l648_64843

-- Define the triangles and their properties
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = 180

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

-- Theorem statement
theorem angle_in_similar_right_triangles 
  (ABC DEF : Triangle) 
  (h_similar : similar ABC DEF) 
  (h_right_ABC : ABC.B = 90) 
  (h_right_DEF : DEF.B = 90) 
  (h_angle_A : ABC.A = 30) 
  (h_angle_D : DEF.A = 60) : 
  DEF.C = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_similar_right_triangles_l648_64843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_bound_l648_64895

/-- A function that checks if a positive integer contains the consecutive digits "729" -/
def contains_729 (n : ℕ+) : Prop :=
  ∃ k : ℕ, (n : ℕ) / 10^k % 1000 = 729

/-- The set of all positive integers that do not contain the consecutive digits "729" -/
def S : Set ℕ+ :=
  {n : ℕ+ | ¬contains_729 n}

/-- The theorem to be proved -/
theorem sum_reciprocals_bound (T : Set ℕ+) (hT : T ⊆ S) :
  ∑' (n : T), (1 : ℝ) / (n : ℝ) ≤ 30000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_bound_l648_64895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l648_64836

theorem log_equation_solution (x : ℝ) (h1 : Real.cos x ≠ 0) (h2 : Real.sin x ≠ 0) :
  (Real.log (Real.sin x) / Real.log (Real.cos x)) - 2 * (Real.log (Real.cos x) / Real.log (Real.sin x)) + 1 = 0 →
  ∃ k : ℤ, x = π/4 + k*π :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l648_64836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_seating_theorem_l648_64879

/-- Represents a person in the meeting -/
structure Person where
  id : Nat

/-- Represents the meeting -/
structure Meeting where
  n : Nat
  people : Finset Person
  knows : Person → Person → Bool

/-- Defines what it means for a seating arrangement to be valid -/
def ValidSeating (m : Meeting) (seated : Finset Person) : Prop :=
  seated.card = 4 ∧
  ∀ p ∈ seated, ∃ p1 p2, p1 ∈ seated ∧ p2 ∈ seated ∧ p ≠ p1 ∧ p ≠ p2 ∧ p1 ≠ p2 ∧ m.knows p p1 ∧ m.knows p p2

/-- The main theorem -/
theorem meeting_seating_theorem (m : Meeting) 
    (h1 : m.people.card = 2 * m.n)
    (h2 : ∀ p ∈ m.people, (m.people.filter (fun q => m.knows p q)).card ≥ m.n) :
    ∃ seated : Finset Person, ValidSeating m seated := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_seating_theorem_l648_64879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_eq_neg_two_l648_64823

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {a b c d e f : ℝ} (h : b ≠ 0 ∧ d ≠ 0) : 
  (∀ x y : ℝ, a*x + b*y + e = 0 ↔ c*x + d*y + f = 0) ↔ a/b = c/d

/-- Definition of line l₁ -/
def l₁ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (m+1)*x + 2*y + 2*m - 2 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ 2*x + (m-2)*y + 2 = 0

theorem parallel_lines_m_eq_neg_two :
  ∀ m : ℝ, (∀ x y : ℝ, l₁ m x y ↔ l₂ m x y) → m = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_eq_neg_two_l648_64823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_area_quadrilateral_is_maximal_maximal_area_value_l648_64872

/-- A quadrilateral inside a semicircle of radius r -/
structure QuadrilateralInSemicircle (r : ℝ) where
  vertices : Fin 4 → ℝ × ℝ
  inside_semicircle : ∀ i, (vertices i).1^2 + (vertices i).2^2 ≤ r^2 ∧ (vertices i).2 ≥ 0

/-- The area of a quadrilateral -/
noncomputable def area (q : QuadrilateralInSemicircle r) : ℝ := sorry

/-- The maximal area quadrilateral -/
noncomputable def maximal_area_quadrilateral (r : ℝ) : QuadrilateralInSemicircle r where
  vertices := λ i => match i with
    | 0 => (-r, 0)
    | 1 => (r, 0)
    | 2 => (r/2, Real.sqrt 3 * r/2)
    | 3 => (-r/2, Real.sqrt 3 * r/2)
  inside_semicircle := sorry

/-- Theorem stating that the maximal area quadrilateral has the largest area -/
theorem maximal_area_quadrilateral_is_maximal (r : ℝ) (h : r > 0) :
  ∀ q : QuadrilateralInSemicircle r, area q ≤ area (maximal_area_quadrilateral r) := by
  sorry

/-- Theorem stating the value of the maximal area -/
theorem maximal_area_value (r : ℝ) (h : r > 0) :
  area (maximal_area_quadrilateral r) = (3 * Real.sqrt 3 / 4) * r^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_area_quadrilateral_is_maximal_maximal_area_value_l648_64872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheaper_rock_cost_l648_64839

/-- Represents the cost of rock per ton -/
structure RockCost where
  cost : ℚ
  deriving Repr

/-- Represents a mixture of rocks -/
structure RockMixture where
  totalWeight : ℚ
  totalCost : ℚ
  cheaperRock : RockCost
  expensiveRock : RockCost
  initialCheaperWeight : ℚ
  initialExpensiveWeight : ℚ
  deriving Repr

/-- The theorem to prove the cost of cheaper rock -/
theorem cheaper_rock_cost (mixture : RockMixture) : 
  mixture.totalWeight = 24 ∧ 
  mixture.totalCost = 800 ∧ 
  mixture.expensiveRock.cost = 40 ∧
  mixture.initialCheaperWeight = 8 ∧
  mixture.initialExpensiveWeight = 8 → 
  mixture.cheaperRock.cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheaper_rock_cost_l648_64839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_average_height_l648_64819

/-- The number of boys in the class -/
def num_boys : ℕ := 35

/-- The initially calculated average height in centimeters -/
def initial_avg : ℚ := 182

/-- The incorrectly recorded height of one boy in centimeters -/
def incorrect_height : ℚ := 166

/-- The actual height of the boy with the incorrectly recorded height in centimeters -/
def actual_height : ℚ := 106

/-- The actual average height of the boys in the class -/
def actual_avg : ℚ := (num_boys * initial_avg - (incorrect_height - actual_height)) / num_boys

/-- Rounds a rational number to two decimal places -/
def round_to_two_decimals (x : ℚ) : ℚ := 
  (x * 100).floor / 100

theorem actual_average_height :
  round_to_two_decimals actual_avg = 18029 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_average_height_l648_64819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_expression_l648_64820

-- Define the integer part function
noncomputable def integerPart (x : ℝ) : ℤ := 
  Int.floor x

-- Define the decimal part function
noncomputable def decimalPart (x : ℝ) : ℝ := 
  x - (integerPart x : ℝ)

-- Main theorem
theorem cube_root_of_expression (a b : ℝ) 
  (h1 : decimalPart (Real.sqrt 5) = a)
  (h2 : (integerPart (Real.sqrt 101) : ℝ) = b) :
  (a + b - Real.sqrt 5) ^ (1/3 : ℝ) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_expression_l648_64820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_showcase_l648_64868

theorem stratified_sampling_showcase (class1_size class2_size showcase_size : ℕ) 
  (h1 : class1_size = 54)
  (h2 : class2_size = 42)
  (h3 : showcase_size = 16) :
  (class1_size * showcase_size) / (class1_size + class2_size) = 9 ∧
  (class2_size * showcase_size) / (class1_size + class2_size) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_showcase_l648_64868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_tan_cot_l648_64801

/-- The area of the region bounded by r = tan θ, r = cot θ, the x-axis, and the y-axis in the first quadrant -/
noncomputable def boundedArea : ℝ := 1/2

/-- The maximum value of r for both r = tan θ and r = cot θ in the first quadrant -/
noncomputable def maxR : ℝ := 1

/-- The angle at which both r = tan θ and r = cot θ reach their maximum in the first quadrant -/
noncomputable def maxAngle : ℝ := Real.pi/4

/-- The point where r = tan θ and r = cot θ intersect in the first quadrant -/
noncomputable def intersectionPoint : ℝ × ℝ := (1, 1)

theorem area_bounded_by_tan_cot (θ : ℝ) :
  0 < θ → θ < Real.pi/2 →
  boundedArea = (1/2) * maxR * maxR := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_tan_cot_l648_64801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_is_ten_l648_64898

/-- An isosceles right triangle with hypotenuse XY and area 25 -/
structure IsoscelesRightTriangle where
  -- The length of a leg
  a : ℝ
  -- The area is 25
  area_eq : a * a / 2 = 25
  -- XY is the hypotenuse
  xy_is_hypotenuse : a * Real.sqrt 2 > a

/-- The length of the hypotenuse XY in the isosceles right triangle -/
noncomputable def hypotenuseLength (t : IsoscelesRightTriangle) : ℝ :=
  t.a * Real.sqrt 2

/-- Theorem: The length of XY in the given isosceles right triangle is 10 -/
theorem hypotenuse_length_is_ten (t : IsoscelesRightTriangle) :
  hypotenuseLength t = 10 := by
  sorry

#check hypotenuse_length_is_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_is_ten_l648_64898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_on_AE_l648_64855

-- Define the triangle ABC
variable (A B C : ℂ)

-- Define the foot of the altitude from A
noncomputable def D (A B C : ℂ) : ℂ := sorry

-- Define the midpoints of BC and AC
noncomputable def M (B C : ℂ) : ℂ := (B + C) / 2
noncomputable def N (A C : ℂ) : ℂ := (A + C) / 2

-- Define the reflection of D over the line passing through M and N
noncomputable def E (A B C : ℂ) : ℂ := 2 * ((M B C + N A C) / 2) - D A B C

-- Define the circumcenter of triangle ABC
noncomputable def O (A B C : ℂ) : ℂ := sorry

-- Theorem statement
theorem circumcenter_on_AE (A B C : ℂ) : 
  ∃ (t : ℝ), O A B C = (1 - t) • A + t • (E A B C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_on_AE_l648_64855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_sequences_return_triangle_l648_64859

-- Define the triangle
structure Triangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

-- Define the isometries
inductive Isometry'
  | rot90 : Isometry'
  | rot180 : Isometry'
  | reflectX : Isometry'
  | reflectY : Isometry'
  | reflectYEqX : Isometry'

def T : Triangle :=
  { v1 := (0, 0),
    v2 := (6, 0),
    v3 := (0, 4) }

-- Apply an isometry to a point
def applyIsometry (i : Isometry') (p : ℝ × ℝ) : ℝ × ℝ :=
  match i with
  | Isometry'.rot90 => (-p.2, p.1)
  | Isometry'.rot180 => (-p.1, -p.2)
  | Isometry'.reflectX => (p.1, -p.2)
  | Isometry'.reflectY => (-p.1, p.2)
  | Isometry'.reflectYEqX => (p.2, p.1)

-- Apply an isometry to a triangle
def applyIsometryToTriangle (i : Isometry') (t : Triangle) : Triangle :=
  { v1 := applyIsometry i t.v1,
    v2 := applyIsometry i t.v2,
    v3 := applyIsometry i t.v3 }

-- Apply a sequence of three isometries to a triangle
def applyThreeIsometries (i1 i2 i3 : Isometry') (t : Triangle) : Triangle :=
  applyIsometryToTriangle i3 (applyIsometryToTriangle i2 (applyIsometryToTriangle i1 t))

-- Check if two triangles are equal
def trianglesEqual (t1 t2 : Triangle) : Prop :=
  t1.v1 = t2.v1 ∧ t1.v2 = t2.v2 ∧ t1.v3 = t2.v3

-- Theorem statement
theorem eighteen_sequences_return_triangle :
  (∃ (s : List (Isometry' × Isometry' × Isometry')),
    (∀ (i1 i2 i3 : Isometry'), trianglesEqual (applyThreeIsometries i1 i2 i3 T) T ↔ (i1, i2, i3) ∈ s) ∧
    s.length = 18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_sequences_return_triangle_l648_64859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_distribution_l648_64821

/-- Represents the total number of players --/
def total_players : ℕ := 1000

/-- Represents the number of players aged below 20 --/
def A : ℕ := sorry

/-- Represents the number of players aged between 20 and 25 years --/
def B : ℕ := sorry

/-- Represents the number of players aged between 25 and 35 years --/
def C : ℕ := sorry

/-- Represents the number of players aged above 35 --/
def D : ℕ := sorry

/-- The sum of all age groups equals the total number of players --/
theorem player_distribution : A + B + C + D = total_players := by
  sorry

#check player_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_distribution_l648_64821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solutions_l648_64880

theorem floor_equation_solutions : 
  (∃ (s : Finset ℕ), s.card = 110 ∧ 
    (∀ x : ℕ, x ∈ s ↔ (⌊(x : ℚ) / 10⌋ : ℤ) = ⌊(x : ℚ) / 11⌋ + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solutions_l648_64880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_rate_problem_l648_64813

/-- The rate of the current given a man's rowing speed and time ratio -/
noncomputable def current_rate (rowing_speed : ℝ) (time_ratio : ℝ) : ℝ :=
  rowing_speed / (time_ratio + 1)

/-- Theorem stating the rate of the current given the problem conditions -/
theorem current_rate_problem :
  let rowing_speed : ℝ := 3.9
  let time_ratio : ℝ := 2
  current_rate rowing_speed time_ratio = 1.3 := by
  -- Unfold the definition of current_rate
  unfold current_rate
  -- Simplify the expression
  simp
  -- Check that the equation holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_rate_problem_l648_64813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_45_degrees_l648_64884

-- Define a right triangle with a 45-degree angle
structure RightTriangle45 where
  opposite : ℝ
  adjacent : ℝ
  hypotenuse : ℝ
  is_right_triangle : opposite^2 + adjacent^2 = hypotenuse^2
  is_45_degree : opposite = adjacent

-- Define tangent for a right triangle
noncomputable def tangent (t : RightTriangle45) : ℝ := t.opposite / t.adjacent

-- State the theorem
theorem tan_45_degrees :
  ∀ (t : RightTriangle45), tangent t = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_45_degrees_l648_64884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_point_is_unique_l648_64803

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (4 - 3*t, Real.sqrt 3 * t)

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 + Real.cos θ, Real.sin θ)

noncomputable def common_point : ℝ × ℝ := (5/2, Real.sqrt 3 / 2)

theorem common_point_is_unique :
  (∃ t : ℝ, line_l t = common_point) ∧
  (∃ θ : ℝ, curve_C θ = common_point) ∧
  (∀ p : ℝ × ℝ, (∃ t : ℝ, line_l t = p) ∧ (∃ θ : ℝ, curve_C θ = p) → p = common_point) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_point_is_unique_l648_64803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_shaded_areas_different_l648_64860

-- Define the side length of the square
variable (s : ℝ) (h : s > 0)

-- Define the shaded areas for each figure
noncomputable def shaded_area_circle (s : ℝ) : ℝ := s^2 * (1 - Real.pi/4)
noncomputable def shaded_area_diagonals (s : ℝ) : ℝ := s^2 / 2
noncomputable def shaded_area_triangle (s : ℝ) : ℝ := 3 * s^2 / 4

-- Theorem stating that all shaded areas are different
theorem all_shaded_areas_different (s : ℝ) (h : s > 0) :
  shaded_area_circle s ≠ shaded_area_diagonals s ∧
  shaded_area_circle s ≠ shaded_area_triangle s ∧
  shaded_area_diagonals s ≠ shaded_area_triangle s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_shaded_areas_different_l648_64860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_log_inequality_l648_64852

theorem x_range_for_log_inequality (x : ℝ) :
  (Real.log (|x - 5| + |x + 3|) / Real.log 10 ≥ 1) ↔ (x ≤ -4 ∨ x ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_log_inequality_l648_64852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isometric_drawing_area_l648_64881

/-- An equilateral triangle with side length a -/
def equilateral_triangle (a : ℝ) : Set (Fin 3 → ℝ × ℝ) :=
  sorry

/-- The isometric drawing of a triangle -/
def isometric_drawing (t : Set (Fin 3 → ℝ × ℝ)) : Set (Fin 3 → ℝ × ℝ) :=
  sorry

/-- The area of a triangle -/
def area (t : Set (Fin 3 → ℝ × ℝ)) : ℝ :=
  sorry

/-- Given an equilateral triangle ABC with side length a, 
    the area of its isometric drawing A'B'C' is (√6/16)a² -/
theorem isometric_drawing_area (a : ℝ) (h : a > 0) :
  let ABC := equilateral_triangle a
  let A'B'C' := isometric_drawing ABC
  area A'B'C' = (Real.sqrt 6 / 16) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isometric_drawing_area_l648_64881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_parameters_l648_64892

/-- A function f(x) = (mx + 1) / (x + n) that is symmetric about the point (2, 4) -/
noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := (m * x + 1) / (x + n)

/-- The symmetry condition for f about the point (2, 4) -/
def is_symmetric (m n : ℝ) : Prop :=
  ∀ x y : ℝ, f m n x = y → f m n (4 - x) = 8 - y

/-- Theorem stating that if f is symmetric about (2, 4), then m = 4 and n = -2 -/
theorem symmetric_function_parameters (m n : ℝ) :
  is_symmetric m n → m = 4 ∧ n = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_parameters_l648_64892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_theorem_l648_64824

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Get the x-intercept of a line -/
noncomputable def xIntercept (l : Line) : ℝ :=
  -l.c / l.a

/-- Get the y-intercept of a line -/
noncomputable def yIntercept (l : Line) : ℝ :=
  -l.c / l.b

/-- Check if a line has equal intercepts on both axes -/
def hasEqualIntercepts (l : Line) : Prop :=
  xIntercept l = yIntercept l

/-- The given point P(2,3) -/
def P : Point :=
  { x := 2, y := 3 }

/-- The two lines that satisfy the conditions -/
def line1 : Line :=
  { a := 3, b := -2, c := 0 }

def line2 : Line :=
  { a := 1, b := 1, c := -5 }

theorem two_lines_theorem :
  ∀ l : Line,
    pointOnLine P l ∧ hasEqualIntercepts l →
    l = line1 ∨ l = line2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_theorem_l648_64824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_on_unit_circle_l648_64853

def A : ℂ := -2
def B : ℂ := 2

theorem max_product_on_unit_circle :
  ∀ P : ℂ, Complex.abs P = 1 → Complex.abs (P - A) * Complex.abs (P - B) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_on_unit_circle_l648_64853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sums_count_l648_64888

def bag_A : Finset ℕ := {1, 2, 5, 7}
def bag_B : Finset ℕ := {3, 4, 6, 8}

def possible_sums : Finset ℕ :=
  Finset.biUnion bag_A (fun a => Finset.image (fun b => a + b) bag_B)

theorem unique_sums_count : Finset.card possible_sums = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sums_count_l648_64888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_set_structure_l648_64871

theorem valid_set_structure (X : Set ℕ+) : 
  (∃ a b : ℕ+, a ∈ X ∧ b ∈ X ∧ a ≠ b) →
  (∀ m n : ℕ+, m ∈ X → n ∈ X → n > m → ∃ k : ℕ+, k ∈ X ∧ n = m * k^2) →
  ∃ m : ℕ+, m > 1 ∧ X = {m, m^3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_set_structure_l648_64871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_six_l648_64899

/-- Represents a rational number with a periodic decimal representation. -/
structure PeriodicDecimal where
  numerator : ℤ
  period : ℕ

/-- Given two PeriodicDecimals with period 30 whose difference has period 15,
    find the smallest k such that their sum with k times the second also has period 15. -/
def smallestK (a b : PeriodicDecimal) : ℕ :=
  6 -- We're stating the result directly here

/-- The main theorem statement -/
theorem smallest_k_is_six (a b : PeriodicDecimal) 
  (ha : a.period = 30) 
  (hb : b.period = 30) 
  (hab : (PeriodicDecimal.mk ((a.numerator - b.numerator) : ℤ) 15).period = 15) : 
  smallestK a b = 6 := by
  sorry

/-- Helper lemma to show that the sum a + k*b has period 15 -/
lemma sum_has_period_15 (a b : PeriodicDecimal) (k : ℕ) 
  (ha : a.period = 30) (hb : b.period = 30) (hk : k = 6) :
  (PeriodicDecimal.mk ((a.numerator + k * b.numerator) : ℤ) 15).period = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_six_l648_64899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_encyclopedia_sorting_l648_64846

/-- A swap operation that exchanges elements at positions i and j in a permutation -/
def swap (π : Fin 10 → Fin 10) (i j : Fin 10) : Fin 10 → Fin 10 :=
  fun k => if k = i then π j else if k = j then π i else π k

/-- Predicate to check if a swap is valid (at least 4 elements between) -/
def valid_swap (i j : Fin 10) : Prop :=
  (i.val < j.val ∧ j.val - i.val > 4) ∨ (j.val < i.val ∧ i.val - j.val > 4)

/-- Predicate to check if a permutation is sorted in ascending order -/
def is_sorted (π : Fin 10 → Fin 10) : Prop :=
  ∀ i j : Fin 10, i.val < j.val → (π i).val < (π j).val

/-- Apply a sequence of swaps to a permutation -/
def apply_swaps (π : Fin 10 → Fin 10) (swaps : List (Fin 10 × Fin 10)) : Fin 10 → Fin 10 :=
  swaps.foldl (fun acc (pair : Fin 10 × Fin 10) => swap acc pair.1 pair.2) π

theorem encyclopedia_sorting :
  ∀ π : Fin 10 → Fin 10,
  ∃ swaps : List (Fin 10 × Fin 10),
  (∀ pair ∈ swaps, valid_swap pair.1 pair.2) ∧
  is_sorted (apply_swaps π swaps) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_encyclopedia_sorting_l648_64846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l648_64832

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State that f is differentiable with derivative f'
variable (hf : Differentiable ℝ f)
variable (hf' : ∀ x, HasDerivAt f (f' x) x)

-- State the condition f'(x) < 2f(x) for all x
variable (h : ∀ x, f' x < 2 * f x)

-- Theorem to prove
theorem function_inequality : Real.exp 2 * f 0 > f 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l648_64832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_to_banana_ratio_l648_64800

/-- The cost of a single banana -/
def b : ℚ := sorry

/-- The cost of a single muffin -/
def m : ℚ := 2 * b

/-- The total cost of Susie's purchase -/
def susie_cost : ℚ := 3 * m + 5 * b

/-- The total cost of Rick's purchase -/
def rick_cost : ℚ := 6 * m + 10 * b

theorem muffin_to_banana_ratio :
  rick_cost = 3 * susie_cost →
  m / b = 2 :=
by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_to_banana_ratio_l648_64800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_squares_no_shared_edge_l648_64849

/-- Represents a square in a 3x3 grid -/
inductive Square
| TopLeft | TopCenter | TopRight
| MiddleLeft | Center | MiddleRight
| BottomLeft | BottomCenter | BottomRight
deriving Repr, DecidableEq, Fintype

/-- Checks if two squares share an edge -/
def shares_edge (s1 s2 : Square) : Bool :=
  match s1, s2 with
  | Square.TopLeft, Square.TopCenter => true
  | Square.TopLeft, Square.MiddleLeft => true
  | Square.TopCenter, Square.TopLeft => true
  | Square.TopCenter, Square.TopRight => true
  | Square.TopCenter, Square.Center => true
  | Square.TopRight, Square.TopCenter => true
  | Square.TopRight, Square.MiddleRight => true
  | Square.MiddleLeft, Square.TopLeft => true
  | Square.MiddleLeft, Square.Center => true
  | Square.MiddleLeft, Square.BottomLeft => true
  | Square.Center, Square.TopCenter => true
  | Square.Center, Square.MiddleLeft => true
  | Square.Center, Square.MiddleRight => true
  | Square.Center, Square.BottomCenter => true
  | Square.MiddleRight, Square.TopRight => true
  | Square.MiddleRight, Square.Center => true
  | Square.MiddleRight, Square.BottomRight => true
  | Square.BottomLeft, Square.MiddleLeft => true
  | Square.BottomLeft, Square.BottomCenter => true
  | Square.BottomCenter, Square.BottomLeft => true
  | Square.BottomCenter, Square.Center => true
  | Square.BottomCenter, Square.BottomRight => true
  | Square.BottomRight, Square.MiddleRight => true
  | Square.BottomRight, Square.BottomCenter => true
  | _, _ => false

/-- The number of ways to select two squares that do not share an edge -/
def num_ways_select_two_squares : ℕ := 24

theorem two_squares_no_shared_edge :
  (Finset.univ.filter (fun p : Square × Square => p.1 ≠ p.2 ∧ ¬shares_edge p.1 p.2)).card = num_ways_select_two_squares :=
by sorry

#eval (Finset.univ.filter (fun p : Square × Square => p.1 ≠ p.2 ∧ ¬shares_edge p.1 p.2)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_squares_no_shared_edge_l648_64849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l648_64841

noncomputable def f (x : ℝ) := 1 / Real.sqrt (6 - x - x^2)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -3 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l648_64841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gaming_marathon_problem_l648_64875

/-- Represents the game rules for a level --/
structure LevelRules where
  duration : ℕ
  lossRate : ℚ
  gainRate : ℚ

/-- Calculates the net lives lost in a level --/
def netLivesLost (rules : LevelRules) : ℚ :=
  rules.duration * rules.lossRate - rules.duration * rules.gainRate

/-- Theorem: Victor's final lives after the gaming marathon --/
def final_lives_count (initialLives : ℕ) 
  (level1 level2 level3 : LevelRules) : ℕ :=
  let l1Loss := netLivesLost level1
  let l2Loss := netLivesLost level2
  let l3Loss := netLivesLost level3
  (initialLives : ℤ) - ⌊l1Loss⌋ - ⌊l2Loss⌋ - ⌊l3Loss⌋ |>.toNat

/-- The gaming marathon problem --/
theorem gaming_marathon_problem : 
  final_lives_count 320 
    (LevelRules.mk 50 (7/10) (2/25))
    (LevelRules.mk 210 (1/8) (4/35))
    (LevelRules.mk 120 (8/30) (5/40)) = 270 := by
  sorry

#eval final_lives_count 320 
  (LevelRules.mk 50 (7/10) (2/25))
  (LevelRules.mk 210 (1/8) (4/35))
  (LevelRules.mk 120 (8/30) (5/40))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gaming_marathon_problem_l648_64875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_equation_sum_l648_64809

/-- Represents a digit in base ten -/
def Digit := Fin 10

/-- Checks if all elements in a list are unique -/
def all_unique (l : List Digit) : Prop :=
  l.Nodup

/-- Converts a two-digit number represented by two digits into a natural number -/
def two_digit_to_nat (tens : Digit) (ones : Digit) : ℕ :=
  (tens.val : ℕ) * 10 + (ones.val : ℕ)

/-- Converts a three-digit number represented by three identical digits into a natural number -/
def three_digit_to_nat (d : Digit) : ℕ :=
  (d.val : ℕ) * 100 + (d.val : ℕ) * 10 + (d.val : ℕ)

theorem digit_equation_sum :
  ∀ (r y b n : Digit),
    all_unique [r, y, b, n] →
    two_digit_to_nat r y * two_digit_to_nat b y = three_digit_to_nat n →
    (y.val : ℕ) + (n.val : ℕ) + (r.val : ℕ) + (b.val : ℕ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_equation_sum_l648_64809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_hemisphere_cylinder_volume_ratio_l648_64811

/-- The ratio of the volume of a sphere to the combined volume of a hemisphere and cylinder -/
theorem sphere_to_hemisphere_cylinder_volume_ratio
  (r : ℝ) (hr : r > 0) :
  (4 / 3 * Real.pi * r^3) / ((1 / 2 * 4 / 3 * Real.pi * (2 * r)^3) + (Real.pi * (2 * r)^2 * (2 * r))) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_hemisphere_cylinder_volume_ratio_l648_64811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_edges_length_eq_five_shortest_visible_edges_length_eq_twenty_l648_64854

/-- A polygon with perpendicular adjacent sides -/
structure PerpendicularPolygon where
  /-- The length of the shortest side -/
  shortest_side : ℝ
  /-- The length of the longest side -/
  longest_side : ℝ
  /-- The length of a removed side -/
  removed_side : ℝ
  /-- The length of the longest side is four times the shortest side -/
  longest_side_eq : longest_side = 4 * shortest_side
  /-- The length of a removed side is twice the shortest side -/
  removed_side_eq : removed_side = 2 * shortest_side
  /-- The removed side is half of the longest side -/
  removed_side_half_longest : removed_side = longest_side / 2
  /-- All lengths are positive -/
  shortest_side_pos : shortest_side > 0

variable (P : PerpendicularPolygon)

/-- The total length of visible edges in the new shape -/
noncomputable def visible_edges_length : ℝ :=
  P.shortest_side + P.longest_side / 2 + P.removed_side

/-- The theorem stating that the total length of visible edges is 5 times the shortest side -/
theorem visible_edges_length_eq_five_shortest :
  visible_edges_length P = 5 * P.shortest_side := by
  unfold visible_edges_length
  rw [P.longest_side_eq, P.removed_side_eq]
  ring

/-- The theorem stating that the total length of visible edges is 20 units when the shortest side is 4 units -/
theorem visible_edges_length_eq_twenty (h : P.shortest_side = 4) :
  visible_edges_length P = 20 := by
  rw [visible_edges_length_eq_five_shortest]
  rw [h]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_edges_length_eq_five_shortest_visible_edges_length_eq_twenty_l648_64854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_photo_problem_l648_64891

/-- The minimal number of attractions needed for a group of tourists to have photos with everyone else -/
def min_attractions (n : ℕ) (k : ℕ) : ℕ :=
  (n * (n - 1) / 2 + k * (n - k) - 1) / (k * (n - k))

/-- Represents a photography session at an attraction -/
structure PhotoSession where
  photographers : Finset ℕ
  subjects : Finset ℕ

/-- Checks if all tourists have photos with everyone else -/
def all_combinations_covered (sessions : List PhotoSession) (n : ℕ) : Prop :=
  ∀ i j, i < n → j < n → i ≠ j →
    ∃ s ∈ sessions, (i ∈ s.photographers ∧ j ∈ s.subjects) ∨ (j ∈ s.photographers ∧ i ∈ s.subjects)

theorem tourist_photo_problem :
  ∃ (sessions : List PhotoSession),
    sessions.length = 4 ∧
    (∀ s ∈ sessions, s.photographers.card = 3 ∧ s.subjects.card = 3) ∧
    all_combinations_covered sessions 6 ∧
    min_attractions 6 3 = 4 := by
  sorry

#eval min_attractions 6 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_photo_problem_l648_64891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_race_lead_is_24_meters_l648_64808

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- The race scenario -/
structure RaceScenario where
  alex : Runner
  blake : Runner
  race_distance : ℝ
  first_race_lead : ℝ
  second_race_headstart : ℝ

/-- Calculates the lead of Alex in the second race -/
noncomputable def second_race_lead (scenario : RaceScenario) : ℝ :=
  scenario.race_distance - 
  (scenario.blake.speed * (scenario.race_distance + scenario.second_race_headstart) / scenario.alex.speed)

/-- The main theorem to prove -/
theorem second_race_lead_is_24_meters 
  (scenario : RaceScenario)
  (h1 : scenario.race_distance = 200)
  (h2 : scenario.first_race_lead = 40)
  (h3 : scenario.second_race_headstart = 20)
  (h4 : scenario.alex.speed / scenario.blake.speed = 5 / 4) :
  second_race_lead scenario = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_race_lead_is_24_meters_l648_64808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_coordinates_l648_64894

def a : ℝ × ℝ := (4, 2)
def b : ℝ × ℝ := (1, 1)

theorem projection_vector_coordinates :
  let proj := ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b
  proj = (3, 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_coordinates_l648_64894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l648_64865

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) / (2*x - 2)

-- State the theorem
theorem f_max_value :
  ∃ (x : ℝ), -4 < x ∧ x < 1 ∧ f x = -1 ∧
  ∀ (y : ℝ), -4 < y → y < 1 → f y ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l648_64865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_from_parallel_planes_l648_64876

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (parallelPlanes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (m n : Line) (α β : Plane) :
  m ≠ n →  -- m and n are distinct
  α ≠ β →  -- α and β are distinct
  perpendicularToPlane m α →
  parallelToPlane n β →
  parallelPlanes α β →
  perpendicular m n :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_from_parallel_planes_l648_64876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l648_64861

/-- The length of a train given its speed, the speed of a man running in the same direction, and the time it takes for the train to cross the man. -/
noncomputable def train_length (train_speed man_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  let relative_speed := train_speed - man_speed
  let relative_speed_ms := relative_speed * (5/18)
  relative_speed_ms * crossing_time

/-- Theorem stating that under the given conditions, the train length is approximately 1259.9 meters. -/
theorem train_length_approx :
  let train_speed := (30 : ℝ) -- km/hr
  let man_speed := (6 : ℝ) -- km/hr
  let crossing_time := (62.99496040316775 : ℝ) -- seconds
  abs (train_length train_speed man_speed crossing_time - 1259.9) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l648_64861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_ratios_l648_64818

/-- Properties of an equilateral triangle -/
theorem equilateral_triangle_ratios (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3 / 4) / (3 * s) = Real.sqrt 3 / 2 ∧
  (s * Real.sqrt 3 / 2) / s = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_ratios_l648_64818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cameron_lead_second_race_l648_64816

/-- Represents the race setup and results -/
structure RaceSetup where
  initialRaceDistance : ℝ
  cameronLeadFirstRace : ℝ
  secondRaceDistance : ℝ
  cameronStartOffset : ℝ

/-- Calculates the distance Cameron is ahead at the end of the second race -/
noncomputable def cameronLeadSecondRace (setup : RaceSetup) : ℝ :=
  let cameronDistance := setup.secondRaceDistance + setup.cameronStartOffset
  let caseyDistance := setup.initialRaceDistance * cameronDistance / setup.initialRaceDistance
  cameronDistance - caseyDistance

/-- Theorem stating that Cameron finishes 29.625 meters ahead in the second race -/
theorem cameron_lead_second_race :
  let setup : RaceSetup := {
    initialRaceDistance := 200
    cameronLeadFirstRace := 15
    secondRaceDistance := 400
    cameronStartOffset := 15
  }
  cameronLeadSecondRace setup = 29.625 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cameron_lead_second_race_l648_64816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l648_64845

theorem solution_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 3 4 ∧ 2^(x - 2) + x = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l648_64845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_class_cyclic_permutations_l648_64829

/-- The number of first-class circular permutations for 2 a's, 2 b's, and 2 c's -/
def T : ℕ := 16

/-- The number of second-class cyclic permutations for n_a a's, n_b b's, and n_c c's -/
def M (n_a n_b n_c : ℕ) : ℕ :=
  (T / 2) + ((((n_a + n_b + n_c) / 2).factorial / (n_a / 2).factorial / (n_b / 2).factorial / (n_c / 2).factorial) / 2)

/-- Theorem: The number of second-class cyclic permutations formed using 2 a's, 2 b's, and 2 c's is 11 -/
theorem second_class_cyclic_permutations : M 2 2 2 = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_class_cyclic_permutations_l648_64829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_magnitude_relationship_l648_64802

theorem trig_magnitude_relationship : Real.sin (168 * π / 180) < Real.cos (10 * π / 180) ∧ Real.cos (10 * π / 180) < Real.tan (58 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_magnitude_relationship_l648_64802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l648_64825

noncomputable def z : ℂ := (Complex.I + 1)^2 + 2*(5 - Complex.I) / (3 + Complex.I)

theorem z_properties : 
  z = 3 - Complex.I ∧ 
  Complex.abs z = Real.sqrt 10 ∧ 
  z.re > 0 ∧ z.im < 0 ∧
  ∃ (a b : ℝ), z * (z + a) = b + Complex.I ∧ a = -7 ∧ b = -13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l648_64825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_23_exists_valid_school_23_no_valid_school_less_than_23_l648_64897

/-- A structure representing a school with students and clubs -/
structure School where
  students : Finset ℕ
  clubs : Finset ℕ
  membership : ℕ → Finset ℕ  -- For each student, the set of clubs they belong to

/-- The properties that the school must satisfy -/
def ValidSchool (s : School) (k : ℕ) : Prop :=
  -- There are 1200 students
  (s.students.card = 1200) ∧
  -- Each student is part of exactly k clubs
  (∀ student ∈ s.students, (s.membership student).card = k) ∧
  -- Any 23 students share at least one common club
  (∀ group : Finset ℕ, group ⊆ s.students → group.card = 23 →
    ∃ club ∈ s.clubs, ∀ student ∈ group, club ∈ s.membership student) ∧
  -- No club includes all 1200 students
  (∀ club ∈ s.clubs, ∃ student ∈ s.students, club ∉ s.membership student)

/-- The main theorem stating that 23 is the smallest possible value for k -/
theorem smallest_k_is_23 :
  ∃ (s : School), ValidSchool s 23 ∧ ∀ k < 23, ¬∃ (s : School), ValidSchool s k := by
  sorry

/-- Auxiliary theorem: There exists a valid school configuration for k = 23 -/
theorem exists_valid_school_23 :
  ∃ (s : School), ValidSchool s 23 := by
  sorry

/-- Auxiliary theorem: No valid school configuration exists for k < 23 -/
theorem no_valid_school_less_than_23 :
  ∀ k < 23, ¬∃ (s : School), ValidSchool s k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_23_exists_valid_school_23_no_valid_school_less_than_23_l648_64897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonreal_root_sum_l648_64862

theorem nonreal_root_sum (ω : ℂ) (h : ω ^ 2 = 1) (h_nonreal : ω ≠ 1 ∧ ω ≠ -1) :
  (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 730 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonreal_root_sum_l648_64862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_telegraph_post_l648_64866

/-- The time (in seconds) it takes for a train to pass a stationary point -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  train_length / (train_speed_kmph * 1000 / 3600)

/-- Theorem: A train of length 80 meters moving at 40 km/h takes approximately 7.2 seconds to pass a stationary point -/
theorem train_passing_telegraph_post :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |train_passing_time 80 40 - 7.2| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_telegraph_post_l648_64866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_factors_of_n_l648_64828

/-- The number of distinct, natural-number factors of 4^4 * 5^3 * 7^2 -/
def num_factors : ℕ := 108

/-- The given number -/
def n : ℕ := 4^4 * 5^3 * 7^2

theorem num_factors_of_n : (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = num_factors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_factors_of_n_l648_64828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l648_64815

theorem negation_of_proposition :
  (∀ x : ℝ, x > 0 → x^2 - 1 ≥ 0) ↔ ¬(∃ x : ℝ, x > 0 ∧ x^2 - 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l648_64815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_equals_one_l648_64848

noncomputable def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

noncomputable def slope1 (a : ℝ) : ℝ := -a / 2

def slope2 : ℝ := 2

theorem perpendicular_lines_a_equals_one :
  ∀ a : ℝ, perpendicular (slope1 a) slope2 → a = 1 := by
  intro a h
  unfold perpendicular at h
  unfold slope1 at h
  unfold slope2 at h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_equals_one_l648_64848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l648_64864

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (2 * a + 1) / x

-- State the theorem
theorem m_range (a : ℝ) (m : ℝ) 
  (h1 : a > 0) 
  (h2 : f a (m^2 + 1) > f a (m^2 - m + 3)) : 
  m > 2 := by
  sorry

#check m_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l648_64864
