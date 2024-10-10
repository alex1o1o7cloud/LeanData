import Mathlib

namespace range_of_a_l1472_147282

/-- A linear function y = mx + b where m = -3a + 1 and b = a -/
def linear_function (a : ℝ) (x : ℝ) : ℝ := (-3 * a + 1) * x + a

/-- Condition that the function is increasing -/
def is_increasing (a : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > x₂ → linear_function a x₁ > linear_function a x₂

/-- Condition that the graph does not pass through the fourth quadrant -/
def not_in_fourth_quadrant (a : ℝ) : Prop :=
  ∀ x y, linear_function a x = y → (x ≥ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≤ 0)

/-- The main theorem stating the range of a -/
theorem range_of_a (a : ℝ) 
  (h1 : is_increasing a) 
  (h2 : not_in_fourth_quadrant a) : 
  0 ≤ a ∧ a < 1/3 := by
  sorry

end range_of_a_l1472_147282


namespace line_equation_correct_l1472_147268

/-- A line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The equation of a line in slope-intercept form -/
def line_equation (l : Line) : ℝ → ℝ := fun x => l.slope * x + (l.point.2 - l.slope * l.point.1)

theorem line_equation_correct (l : Line) : 
  l.slope = 2 ∧ l.point = (0, 3) → line_equation l = fun x => 2 * x + 3 := by
  sorry

end line_equation_correct_l1472_147268


namespace matrix_multiplication_l1472_147284

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -1; 4, 2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 6; -2, 3]

theorem matrix_multiplication :
  A * B = !![2, 15; -4, 30] := by sorry

end matrix_multiplication_l1472_147284


namespace find_divisor_l1472_147278

theorem find_divisor : ∃ (x : ℕ), x > 0 ∧ 190 = 9 * x + 1 :=
by
  use 21
  sorry

end find_divisor_l1472_147278


namespace books_from_library_l1472_147216

/-- The number of books initially obtained from the library -/
def initial_books : ℕ := 54

/-- The number of additional books obtained from the library -/
def additional_books : ℕ := 23

/-- The total number of books obtained from the library -/
def total_books : ℕ := initial_books + additional_books

theorem books_from_library : total_books = 77 := by
  sorry

end books_from_library_l1472_147216


namespace parabola_sum_l1472_147265

/-- Represents a parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℚ) : ℚ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum (p : Parabola) :
  p.x_coord (-6) = 7 →  -- vertex condition
  p.x_coord 0 = 1 →     -- point condition
  p.a + p.b + p.c = -43/6 := by
sorry

end parabola_sum_l1472_147265


namespace max_triangles_is_eleven_l1472_147279

/-- Represents an equilateral triangle with a line segment connecting midpoints of two sides -/
structure EquilateralTriangleWithMidline where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents the configuration of two overlapping equilateral triangles -/
structure OverlappingTriangles where
  triangle1 : EquilateralTriangleWithMidline
  triangle2 : EquilateralTriangleWithMidline
  overlap : ℝ -- Represents the degree of overlap between the triangles

/-- Counts the number of triangles formed in a given configuration -/
def countTriangles (config : OverlappingTriangles) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of triangles is 11 -/
theorem max_triangles_is_eleven :
  ∃ (config : OverlappingTriangles),
    (∀ (other : OverlappingTriangles), countTriangles other ≤ countTriangles config) ∧
    countTriangles config = 11 :=
  sorry

end max_triangles_is_eleven_l1472_147279


namespace four_tangent_circles_l1472_147224

-- Define a line in 2D space
def Line2D := (ℝ × ℝ) → Prop

-- Define a circle in 2D space
structure Circle2D where
  center : ℝ × ℝ
  radius : ℝ

-- Define tangency between a circle and a line
def isTangent (c : Circle2D) (l : Line2D) : Prop := sorry

-- Main theorem
theorem four_tangent_circles 
  (l1 l2 : Line2D) 
  (intersect : ∃ p : ℝ × ℝ, l1 p ∧ l2 p) 
  (r : ℝ) 
  (h : r > 0) : 
  ∃! (cs : Finset Circle2D), 
    cs.card = 4 ∧ 
    (∀ c ∈ cs, c.radius = r ∧ isTangent c l1 ∧ isTangent c l2) :=
sorry

end four_tangent_circles_l1472_147224


namespace vanessa_missed_days_l1472_147218

/-- Represents the number of days missed by each student -/
structure MissedDays where
  vanessa : ℕ
  mike : ℕ
  sarah : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (d : MissedDays) : Prop :=
  d.vanessa + d.mike + d.sarah = 17 ∧
  d.vanessa + d.mike = 14 ∧
  d.mike + d.sarah = 12

/-- The theorem to prove -/
theorem vanessa_missed_days (d : MissedDays) (h : satisfiesConditions d) : d.vanessa = 5 := by
  sorry

end vanessa_missed_days_l1472_147218


namespace max_bar_weight_example_l1472_147267

/-- Calculates the maximum weight that can be put on a weight bench bar given the bench's maximum support weight, safety margin percentage, and the weights of two people using the bench. -/
def maxBarWeight (benchMax : ℝ) (safetyMargin : ℝ) (weight1 : ℝ) (weight2 : ℝ) : ℝ :=
  benchMax * (1 - safetyMargin) - (weight1 + weight2)

/-- Theorem stating that for a 1000-pound bench with 20% safety margin and two people weighing 250 and 180 pounds, the maximum weight on the bar is 370 pounds. -/
theorem max_bar_weight_example :
  maxBarWeight 1000 0.2 250 180 = 370 := by
  sorry

end max_bar_weight_example_l1472_147267


namespace silverware_cost_l1472_147209

/-- The cost of silverware given the conditions in the problem -/
theorem silverware_cost : 
  ∀ (silverware_cost dinner_plates_cost : ℝ),
  dinner_plates_cost = 0.5 * silverware_cost →
  silverware_cost + dinner_plates_cost = 30 →
  silverware_cost = 20 := by
sorry

end silverware_cost_l1472_147209


namespace equal_fractions_imply_one_third_l1472_147217

theorem equal_fractions_imply_one_third (x : ℝ) (h1 : x > 0) 
  (h2 : (2/3) * x = (16/216) * (1/x)) : x = 1/3 := by
  sorry

end equal_fractions_imply_one_third_l1472_147217


namespace triangle_trig_identity_l1472_147254

theorem triangle_trig_identity (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = Real.pi) (h5 : A ≤ B) (h6 : B ≤ C)
  (h7 : (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) = Real.sqrt 3) :
  Real.sin B + Real.sin (2 * B) = Real.sqrt 3 := by
sorry

end triangle_trig_identity_l1472_147254


namespace circle_ratio_l1472_147241

theorem circle_ratio (R r : ℝ) (h1 : R > 0) (h2 : r > 0) (h3 : R > r) :
  (π * R^2 - π * r^2) = 3 * (π * r^2) → R = 2 * r := by
  sorry

end circle_ratio_l1472_147241


namespace inverse_of_f_is_neg_g_neg_l1472_147280

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the symmetry condition
def symmetric_wrt_x_plus_y_eq_0 (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g (-y) = -x

-- Define the inverse function
def has_inverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x

-- Theorem statement
theorem inverse_of_f_is_neg_g_neg (hf : has_inverse f) (h_sym : symmetric_wrt_x_plus_y_eq_0 f g) :
  ∃ f_inv : ℝ → ℝ, (∀ x, f_inv (f x) = x ∧ f (f_inv x) = x) ∧ (∀ x, f_inv x = -g (-x)) :=
sorry

end inverse_of_f_is_neg_g_neg_l1472_147280


namespace rectangle_length_from_square_perimeter_l1472_147272

theorem rectangle_length_from_square_perimeter (square_side : ℝ) (rect_width : ℝ) :
  square_side = 12 →
  rect_width = 6 →
  4 * square_side = 2 * (rect_width + (18 : ℝ)) :=
by
  sorry

#check rectangle_length_from_square_perimeter

end rectangle_length_from_square_perimeter_l1472_147272


namespace fair_coin_heads_prob_equals_frequency_l1472_147230

/-- Represents the outcome of a coin toss experiment -/
structure CoinTossExperiment where
  total_tosses : ℕ
  heads_count : ℕ
  heads_frequency : ℝ

/-- Defines what it means for an experiment to be valid -/
def is_valid_experiment (e : CoinTossExperiment) : Prop :=
  e.total_tosses > 0 ∧ 
  e.heads_count ≤ e.total_tosses ∧ 
  e.heads_frequency = (e.heads_count : ℝ) / (e.total_tosses : ℝ)

/-- The probability of a fair coin landing heads up -/
def fair_coin_heads_probability : ℝ := 0.5005

/-- Pearson's experiment data -/
def pearson_experiment : CoinTossExperiment := {
  total_tosses := 24000,
  heads_count := 12012,
  heads_frequency := 0.5005
}

/-- Theorem stating that the probability of a fair coin landing heads up
    is equal to the frequency observed in Pearson's large-scale experiment -/
theorem fair_coin_heads_prob_equals_frequency 
  (h_valid : is_valid_experiment pearson_experiment)
  (h_large : pearson_experiment.total_tosses ≥ 10000) :
  fair_coin_heads_probability = pearson_experiment.heads_frequency := by
  sorry


end fair_coin_heads_prob_equals_frequency_l1472_147230


namespace matching_color_probability_l1472_147264

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.yellow

/-- Abe's jelly bean distribution -/
def abe : JellyBeans :=
  { green := 2, red := 2, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob : JellyBeans :=
  { green := 2, red := 3, yellow := 2 }

/-- Calculates the probability of selecting a specific color -/
def probColor (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / jb.total

/-- Calculates the probability of both selecting the same color -/
def probMatchingColor (jb1 jb2 : JellyBeans) : ℚ :=
  probColor jb1 jb1.green * probColor jb2 jb2.green +
  probColor jb1 jb1.red * probColor jb2 jb2.red

theorem matching_color_probability :
  probMatchingColor abe bob = 5 / 14 := by
  sorry

end matching_color_probability_l1472_147264


namespace noodles_and_pirates_total_l1472_147297

theorem noodles_and_pirates_total (pirates : ℕ) (noodles : ℕ) : 
  pirates = 45 → noodles = pirates - 7 → noodles + pirates = 83 :=
by sorry

end noodles_and_pirates_total_l1472_147297


namespace probability_of_card_sequence_l1472_147222

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards of each suit in a standard deck -/
def CardsPerSuit : ℕ := 13

/-- Calculates the probability of the specified card sequence -/
def probabilityOfSequence : ℚ :=
  (CardsPerSuit : ℚ) / StandardDeck *
  (CardsPerSuit - 1) / (StandardDeck - 1) *
  CardsPerSuit / (StandardDeck - 2) *
  CardsPerSuit / (StandardDeck - 3)

/-- Theorem stating that the probability of drawing two hearts, 
    followed by one diamond, and then one club from a standard 
    52-card deck is equal to 39/63875 -/
theorem probability_of_card_sequence :
  probabilityOfSequence = 39 / 63875 := by
  sorry

end probability_of_card_sequence_l1472_147222


namespace chord_length_in_unit_circle_l1472_147238

theorem chord_length_in_unit_circle (chord1 chord2 chord3 : Real) : 
  -- Unit circle condition
  ∀ (r : Real), r = 1 →
  -- Three distinct diameters
  ∃ (α θ : Real), α ≠ θ ∧ α + θ + (180 - α - θ) = 180 →
  -- One chord has length √2
  chord1 = Real.sqrt 2 →
  -- The other two chords have equal lengths
  chord2 = chord3 →
  -- Length of chord2 and chord3
  chord2 = Real.sqrt (2 - Real.sqrt 2) := by
sorry


end chord_length_in_unit_circle_l1472_147238


namespace sqrt_meaningful_range_l1472_147205

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 2) ↔ x ≥ -2 := by sorry

end sqrt_meaningful_range_l1472_147205


namespace total_product_weight_is_correct_l1472_147219

/-- Represents a chemical element or compound -/
structure Chemical where
  formula : String
  molarMass : Float

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List (Chemical × Float)
  products : List (Chemical × Float)

def CaCO3 : Chemical := ⟨"CaCO3", 100.09⟩
def CaO : Chemical := ⟨"CaO", 56.08⟩
def CO2 : Chemical := ⟨"CO2", 44.01⟩
def HCl : Chemical := ⟨"HCl", 36.46⟩
def CaCl2 : Chemical := ⟨"CaCl2", 110.98⟩
def H2O : Chemical := ⟨"H2O", 18.02⟩

def reaction1 : Reaction := ⟨[(CaCO3, 1)], [(CaO, 1), (CO2, 1)]⟩
def reaction2 : Reaction := ⟨[(HCl, 2), (CaCO3, 1)], [(CaCl2, 1), (CO2, 1), (H2O, 1)]⟩

def initialCaCO3 : Float := 8
def initialHCl : Float := 12

/-- Calculates the total weight of products from both reactions -/
def totalProductWeight (r1 : Reaction) (r2 : Reaction) (initCaCO3 : Float) (initHCl : Float) : Float :=
  sorry

theorem total_product_weight_is_correct :
  totalProductWeight reaction1 reaction2 initialCaCO3 initialHCl = 800.72 := by sorry

end total_product_weight_is_correct_l1472_147219


namespace problem_solution_l1472_147274

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/2 = 6*y) : x = 48 := by
  sorry

end problem_solution_l1472_147274


namespace opposite_blue_is_black_l1472_147229

-- Define the colors
inductive Color
| Blue | Yellow | Orange | Black | Silver | Gold

-- Define a cube
structure Cube where
  faces : Fin 6 → Color

-- Define the views
structure View where
  top : Color
  front : Color
  right : Color

-- Define the problem setup
def cube_problem (c : Cube) (v1 v2 v3 : View) : Prop :=
  -- All faces have different colors
  (∀ i j : Fin 6, i ≠ j → c.faces i ≠ c.faces j) ∧
  -- The views are consistent with the cube
  (v1.top = Color.Gold ∧ v1.front = Color.Black ∧ v1.right = Color.Orange) ∧
  (v2.top = Color.Gold ∧ v2.front = Color.Yellow ∧ v2.right = Color.Orange) ∧
  (v3.top = Color.Gold ∧ v3.front = Color.Silver ∧ v3.right = Color.Orange)

-- The theorem to prove
theorem opposite_blue_is_black (c : Cube) (v1 v2 v3 : View) 
  (h : cube_problem c v1 v2 v3) : 
  ∃ (i j : Fin 6), c.faces i = Color.Blue ∧ c.faces j = Color.Black ∧ 
  (i.val + j.val = 5 ∨ i.val + j.val = 7) :=
sorry

end opposite_blue_is_black_l1472_147229


namespace sum_prime_factors_2_pow_22_minus_4_l1472_147242

/-- SPF(n) denotes the sum of the prime factors of n, where the prime factors are not necessarily distinct. -/
def SPF (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of prime factors of 2^22 - 4 is 100. -/
theorem sum_prime_factors_2_pow_22_minus_4 : SPF (2^22 - 4) = 100 := by sorry

end sum_prime_factors_2_pow_22_minus_4_l1472_147242


namespace power_of_power_l1472_147266

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l1472_147266


namespace hypotenuse_length_l1472_147239

/-- A point on the parabola y = -x^2 --/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = -x^2

/-- Triangle ABO with A and B on the parabola y = -x^2 and ∠AOB = 45° --/
structure TriangleABO where
  A : ParabolaPoint
  B : ParabolaPoint
  angle_AOB : Real.pi / 4 = Real.arctan (A.y / A.x) + Real.arctan (B.y / B.x)

/-- The length of the hypotenuse of triangle ABO is 2 --/
theorem hypotenuse_length (t : TriangleABO) : 
  Real.sqrt ((t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2) = 2 := by
  sorry

end hypotenuse_length_l1472_147239


namespace log_system_solution_l1472_147251

theorem log_system_solution :
  ∀ x y : ℝ, x > 0 → y > 0 →
  (Real.log x / Real.log 4 - Real.log y / Real.log 2 = 0) →
  (x^2 - 5*y^2 + 4 = 0) →
  ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 2)) :=
by sorry

end log_system_solution_l1472_147251


namespace appropriate_mass_units_l1472_147253

-- Define the mass units
inductive MassUnit
| Gram
| Ton
| Kilogram

-- Define a structure for an item with its mass value
structure MassItem where
  value : ℕ
  unit : MassUnit

-- Define the function to check if a mass unit is appropriate for a given item
def isAppropriateUnit (item : MassItem) : Prop :=
  match item with
  | ⟨1, MassUnit.Gram⟩ => true     -- Peanut kernel
  | ⟨8, MassUnit.Ton⟩ => true      -- Truck loading capacity
  | ⟨30, MassUnit.Kilogram⟩ => true -- Xiao Ming's weight
  | ⟨580, MassUnit.Gram⟩ => true   -- Basketball mass
  | _ => false

-- Theorem statement
theorem appropriate_mass_units :
  let peanut := MassItem.mk 1 MassUnit.Gram
  let truck := MassItem.mk 8 MassUnit.Ton
  let xiaoMing := MassItem.mk 30 MassUnit.Kilogram
  let basketball := MassItem.mk 580 MassUnit.Gram
  isAppropriateUnit peanut ∧
  isAppropriateUnit truck ∧
  isAppropriateUnit xiaoMing ∧
  isAppropriateUnit basketball :=
by sorry


end appropriate_mass_units_l1472_147253


namespace identity_is_unique_solution_l1472_147226

/-- The set of positive integers -/
def PositiveIntegers := {n : ℕ | n > 0}

/-- A function from positive integers to positive integers -/
def PositiveIntegerFunction := PositiveIntegers → PositiveIntegers

/-- The functional equation that f must satisfy -/
def SatisfiesEquation (f : PositiveIntegerFunction) : Prop :=
  ∀ m n : PositiveIntegers,
    Nat.gcd (f m) n + Nat.lcm m (f n) = Nat.gcd m (f n) + Nat.lcm (f m) n

/-- The theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∃! f : PositiveIntegerFunction, SatisfiesEquation f ∧ (∀ n, f n = n) :=
sorry

end identity_is_unique_solution_l1472_147226


namespace least_number_with_remainder_one_l1472_147240

theorem least_number_with_remainder_one (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 115 → (m % 38 ≠ 1 ∨ m % 3 ≠ 1)) ∧ 
  (115 % 38 = 1 ∧ 115 % 3 = 1) := by
  sorry

end least_number_with_remainder_one_l1472_147240


namespace min_three_digit_divisible_by_seven_l1472_147259

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def remove_middle_digit (n : ℕ) : ℕ :=
  (n / 100) * 10 + (n % 10)

theorem min_three_digit_divisible_by_seven :
  ∃ (N : ℕ),
    is_three_digit N ∧
    N % 7 = 0 ∧
    (remove_middle_digit N) % 7 = 0 ∧
    (∀ (M : ℕ), 
      is_three_digit M ∧ 
      M % 7 = 0 ∧ 
      (remove_middle_digit M) % 7 = 0 → 
      N ≤ M) ∧
    N = 154 := by
  sorry

end min_three_digit_divisible_by_seven_l1472_147259


namespace range_of_m_l1472_147295

theorem range_of_m (f : ℝ → ℝ → ℝ) (x₀ : ℝ) (h_nonzero : x₀ ≠ 0) :
  (∀ m : ℝ, f m x = 9*x - m) →
  f x₀ x₀ = f 0 x₀ →
  ∃ m : ℝ, 0 < m ∧ m < 1/2 :=
sorry

end range_of_m_l1472_147295


namespace basketball_rim_height_l1472_147257

/-- Represents the height of a basketball rim above the ground -/
def rim_height : ℕ := sorry

/-- Represents the player's height in feet -/
def player_height_feet : ℕ := 6

/-- Represents the player's reach above their head in inches -/
def player_reach : ℕ := 22

/-- Represents the player's jump height in inches -/
def player_jump : ℕ := 32

/-- Represents how far above the rim the player can reach when jumping, in inches -/
def above_rim : ℕ := 6

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

theorem basketball_rim_height : 
  rim_height = player_height_feet * feet_to_inches + player_reach + player_jump - above_rim :=
by sorry

end basketball_rim_height_l1472_147257


namespace sector_central_angle_sine_l1472_147206

theorem sector_central_angle_sine (r : ℝ) (arc_length : ℝ) (h1 : r = 2) (h2 : arc_length = 8 * Real.pi / 3) :
  Real.sin (arc_length / r) = -Real.sqrt 3 / 2 := by
  sorry

end sector_central_angle_sine_l1472_147206


namespace temperature_data_inconsistency_l1472_147214

theorem temperature_data_inconsistency 
  (x_bar : ℝ) 
  (m : ℝ) 
  (S_squared : ℝ) 
  (hx : x_bar = 0) 
  (hm : m = 4) 
  (hS : S_squared = 15.917) : 
  ¬(|x_bar - m| ≤ Real.sqrt S_squared) := by
sorry

end temperature_data_inconsistency_l1472_147214


namespace circle_radius_l1472_147220

/-- Given a circle with area P and circumference Q, if P/Q = 25, then the radius is 50 -/
theorem circle_radius (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) (h3 : P / Q = 25) :
  ∃ (r : ℝ), P = π * r^2 ∧ Q = 2 * π * r ∧ r = 50 := by
sorry

end circle_radius_l1472_147220


namespace arithmetic_sequence_ratio_l1472_147293

/-- An arithmetic sequence with first term 5 and sum of first 31 terms equal to 390 -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 5 ∧ 
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (Finset.sum (Finset.range 31) (λ i => a (i + 1)) = 390)

/-- The ratio of sum of odd-indexed terms to sum of even-indexed terms -/
def ratio (a : ℕ → ℚ) : ℚ :=
  (Finset.sum (Finset.filter (λ i => i % 2 = 1) (Finset.range 31)) (λ i => a (i + 1))) /
  (Finset.sum (Finset.filter (λ i => i % 2 = 0) (Finset.range 31)) (λ i => a (i + 1)))

theorem arithmetic_sequence_ratio (a : ℕ → ℚ) :
  arithmetic_sequence a → ratio a = 16 / 15 := by
  sorry

end arithmetic_sequence_ratio_l1472_147293


namespace least_k_divisible_by_1680_l1472_147202

theorem least_k_divisible_by_1680 :
  ∃ (k : ℕ), k > 0 ∧
  (∃ (a b c d : ℕ), k = 2^a * 3^b * 5^c * 7^d) ∧
  (1680 ∣ k^4) ∧
  (∀ (m : ℕ), m > 0 →
    (∃ (x y z w : ℕ), m = 2^x * 3^y * 5^z * 7^w) →
    (1680 ∣ m^4) →
    m ≥ k) ∧
  k = 210 :=
sorry

end least_k_divisible_by_1680_l1472_147202


namespace equation_roots_and_expression_l1472_147294

open Real

theorem equation_roots_and_expression (α m : ℝ) : 
  0 < α → α < π →
  (∃ x : ℝ, x^2 + 4 * x * sin (α/2) + m * tan (α/2) = 0 ∧ 
   ∀ y : ℝ, y^2 + 4 * y * sin (α/2) + m * tan (α/2) = 0 → y = x) →
  m + 2 * cos α = 4/3 →
  (0 < m ∧ m ≤ 2) ∧ 
  (1 + sin (2*α) - cos (2*α)) / (1 + tan α) = -5/9 := by
sorry

end equation_roots_and_expression_l1472_147294


namespace muffin_banana_cost_ratio_l1472_147299

/-- The cost ratio of a muffin to a banana given purchase information -/
theorem muffin_banana_cost_ratio :
  ∀ (m b : ℝ), 
    m > 0 →  -- m is positive (cost of muffin)
    b > 0 →  -- b is positive (cost of banana)
    5 * m + 2 * b > 0 →  -- Susie's purchase is positive
    3 * (5 * m + 2 * b) = 4 * m + 10 * b →  -- Jason's purchase is 3 times Susie's
    m / b = 4 / 11 :=
by
  sorry

end muffin_banana_cost_ratio_l1472_147299


namespace hyperbola_focus_l1472_147228

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  (x - 5)^2 / 9^2 - (y - 20)^2 / 15^2 = 1

def is_focus (x y : ℝ) : Prop :=
  hyperbola_equation x y ∧ 
  ∃ (x' y' : ℝ), hyperbola_equation x' y' ∧ 
  (x - 5)^2 + (y - 20)^2 = (x' - 5)^2 + (y' - 20)^2 ∧ 
  (x ≠ x' ∨ y ≠ y')

theorem hyperbola_focus :
  ∃ (x y : ℝ), is_focus x y ∧ 
  (∀ (x' y' : ℝ), is_focus x' y' → x' ≤ x) ∧
  x = 5 + Real.sqrt 306 ∧ y = 20 := by sorry

end hyperbola_focus_l1472_147228


namespace hyperbola_range_of_b_squared_l1472_147232

/-- Given a hyperbola M: x^2 - y^2/b^2 = 1 (b > 0) with foci F1(-c, 0) and F2(c, 0),
    if a line parallel to one asymptote passes through F1 and intersects the other asymptote at P(-c/2, bc/2),
    and P is inside the circle x^2 + y^2 = 4b^2, then 7 - 4√3 < b^2 < 7 + 4√3 -/
theorem hyperbola_range_of_b_squared (b c : ℝ) (hb : b > 0) (hc : c^2 = b^2 + 1) :
  let P : ℝ × ℝ := (-c/2, b*c/2)
  (P.1^2 + P.2^2 < 4*b^2) → (7 - 4*Real.sqrt 3 < b^2 ∧ b^2 < 7 + 4*Real.sqrt 3) :=
by sorry

end hyperbola_range_of_b_squared_l1472_147232


namespace binary_101101_to_decimal_l1472_147237

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (λ acc (i, b) => acc + if b then 2^i else 0) 0

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_to_decimal :
  binary_to_decimal binary_101101 = 45 := by
  sorry

end binary_101101_to_decimal_l1472_147237


namespace abc_relationship_l1472_147211

theorem abc_relationship : ∀ (a b c : ℕ),
  a = 3^44 → b = 4^33 → c = 5^22 → c < b ∧ b < a := by
  sorry

end abc_relationship_l1472_147211


namespace least_value_ba_l1472_147255

/-- Given a number in the form 11,0ab that is divisible by 115, 
    the least possible value of b × a is 0 -/
theorem least_value_ba (a b : ℕ) : 
  a < 10 → b < 10 → (11000 + 100 * a + b) % 115 = 0 → 
  ∀ (c d : ℕ), c < 10 → d < 10 → (11000 + 100 * c + d) % 115 = 0 → 
  b * a ≤ d * c := by
  sorry

end least_value_ba_l1472_147255


namespace complex_magnitude_l1472_147273

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 30)
  (h2 : Complex.abs (z + 3 * w) = 15)
  (h3 : Complex.abs (z - w) = 10) :
  Complex.abs z = 9 := by
  sorry

end complex_magnitude_l1472_147273


namespace isosceles_triangle_x_values_l1472_147204

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the square of the distance between two points in 3D space -/
def distanceSquared (p1 p2 : Point3D) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

/-- Theorem: In an isosceles triangle ABC with vertices A(4, 1, 9), B(10, -1, 6), 
    and C(x, 4, 3), where BC is the base, the possible values of x are 2 and 6 -/
theorem isosceles_triangle_x_values :
  let A : Point3D := ⟨4, 1, 9⟩
  let B : Point3D := ⟨10, -1, 6⟩
  let C : Point3D := ⟨x, 4, 3⟩
  (distanceSquared A B = distanceSquared A C) → (x = 2 ∨ x = 6) :=
by sorry

end isosceles_triangle_x_values_l1472_147204


namespace base7_to_base10_conversion_l1472_147247

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def base7Number : List Nat := [0, 1, 2, 3, 4]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 10738 := by
  sorry

end base7_to_base10_conversion_l1472_147247


namespace games_lost_calculation_l1472_147287

def total_games : ℕ := 12
def games_won : ℕ := 8

theorem games_lost_calculation : total_games - games_won = 4 := by
  sorry

end games_lost_calculation_l1472_147287


namespace reflection_across_y_axis_l1472_147236

/-- Given a point A with coordinates (-4, 8, 6), 
    prove that its reflection across the y-axis has coordinates (4, 8, 6) -/
theorem reflection_across_y_axis :
  let A : ℝ × ℝ × ℝ := (-4, 8, 6)
  let reflection : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ := fun (x, y, z) ↦ (-x, y, z)
  reflection A = (4, 8, 6) := by
  sorry

end reflection_across_y_axis_l1472_147236


namespace complex_expression_value_l1472_147234

theorem complex_expression_value (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end complex_expression_value_l1472_147234


namespace angle_H_measure_l1472_147208

/-- Pentagon MATHS with specific angle conditions -/
structure Pentagon where
  M : ℝ  -- Measure of angle M
  A : ℝ  -- Measure of angle A
  T : ℝ  -- Measure of angle T
  H : ℝ  -- Measure of angle H
  S : ℝ  -- Measure of angle S
  angles_sum : M + A + T + H + S = 540
  equal_angles : M = T ∧ T = H
  supplementary : A + S = 180

/-- The measure of angle H in the specified pentagon is 120° -/
theorem angle_H_measure (p : Pentagon) : p.H = 120 := by
  sorry

end angle_H_measure_l1472_147208


namespace rectangle_height_twice_square_side_l1472_147262

/-- Given a square with side length s and a rectangle with base s and area twice that of the square,
    prove that the height of the rectangle is 2s. -/
theorem rectangle_height_twice_square_side (s : ℝ) (h : s > 0) : 
  let square_area := s^2
  let rectangle_base := s
  let rectangle_area := 2 * square_area
  rectangle_area / rectangle_base = 2 * s := by
  sorry

end rectangle_height_twice_square_side_l1472_147262


namespace problem_solution_l1472_147288

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 - 1/2

theorem problem_solution :
  ∃ (A B C a b c : ℝ),
    0 < A ∧ A < π ∧
    0 < B ∧ B < π ∧
    0 < C ∧ C < π ∧
    Real.sin B - 2 * Real.sin A = 0 ∧
    c = 3 ∧
    f C = 0 ∧
    (∀ x, f x ≥ -2) ∧
    (∀ ε > 0, ∃ x, f x < -2 + ε) ∧
    (∀ x, f (x + π) = f x) ∧
    (∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ π) ∧
    a = Real.sqrt 3 ∧
    b = 2 * Real.sqrt 3 := by
  sorry

end problem_solution_l1472_147288


namespace subtraction_of_large_numbers_l1472_147258

theorem subtraction_of_large_numbers :
  1000000000000 - 888777888777 = 111222111223 := by
  sorry

end subtraction_of_large_numbers_l1472_147258


namespace quadrilateral_angle_sum_l1472_147285

theorem quadrilateral_angle_sum (a b c d : ℝ) (α β γ δ : ℝ) 
  (ha : a = 15) (hb : b = 20) (hc : c = 25) (hd : d = 33)
  (hα : α = 100) (hβ : β = 80) (hγ : γ = 105) (hδ : δ = 75) :
  α + β + γ + δ = 360 := by
sorry

end quadrilateral_angle_sum_l1472_147285


namespace system_solution_unique_l1472_147271

theorem system_solution_unique :
  ∃! (x y : ℚ), (3 * (x - 1) = y + 6) ∧ (x / 2 + y / 3 = 2) ∧ (x = 10 / 3) ∧ (y = 1) := by
  sorry

end system_solution_unique_l1472_147271


namespace least_positive_angle_phi_l1472_147203

theorem least_positive_angle_phi : 
  ∃ φ : ℝ, φ > 0 ∧ φ ≤ π/2 ∧ 
  (∀ ψ : ℝ, ψ > 0 → ψ < φ → Real.cos (10 * π/180) ≠ Real.sin (15 * π/180) + Real.sin ψ) ∧
  Real.cos (10 * π/180) = Real.sin (15 * π/180) + Real.sin φ ∧
  φ = 42.5 * π/180 :=
sorry

end least_positive_angle_phi_l1472_147203


namespace fraction_order_l1472_147246

theorem fraction_order : (25 : ℚ) / 21 < 23 / 19 ∧ 23 / 19 < 21 / 17 := by
  sorry

end fraction_order_l1472_147246


namespace egg_price_calculation_l1472_147201

/-- Proves that the price of each egg is $0.20 given the conditions of the problem --/
theorem egg_price_calculation (total_eggs : ℕ) (crate_cost : ℚ) (eggs_left : ℕ) : 
  total_eggs = 30 → crate_cost = 5 → eggs_left = 5 → 
  (crate_cost / (total_eggs - eggs_left : ℚ)) = 0.20 := by
  sorry

#check egg_price_calculation

end egg_price_calculation_l1472_147201


namespace systematic_sampling_problem_l1472_147250

/-- Systematic sampling function -/
def systematic_sampling (population : ℕ) (sample_size : ℕ) : 
  (ℕ × ℕ × ℕ) :=
  let remaining := population % sample_size
  let eliminated := remaining
  let segment_size := (population - eliminated) / sample_size
  (eliminated, sample_size, segment_size)

/-- Theorem for the given systematic sampling problem -/
theorem systematic_sampling_problem :
  systematic_sampling 1650 35 = (5, 35, 47) := by
  sorry

end systematic_sampling_problem_l1472_147250


namespace solution_set_x_squared_minus_one_lt_zero_l1472_147292

theorem solution_set_x_squared_minus_one_lt_zero :
  Set.Ioo (-1 : ℝ) 1 = {x : ℝ | x^2 - 1 < 0} := by sorry

end solution_set_x_squared_minus_one_lt_zero_l1472_147292


namespace part_one_part_two_l1472_147223

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Part 1
theorem part_one : (Aᶜ ∪ B 1) = {x : ℝ | x ≤ -2 ∨ x > 1} := by sorry

-- Part 2
theorem part_two : A ⊆ B a → a ≤ -2 := by sorry

end part_one_part_two_l1472_147223


namespace min_value_squared_sum_l1472_147243

theorem min_value_squared_sum (a b t k : ℝ) (hk : k > 0) (ht : a + k * b = t) :
  a^2 + k^2 * b^2 ≥ ((1 + k^2) * t^2) / (1 + k)^2 := by
  sorry

end min_value_squared_sum_l1472_147243


namespace bracket_mult_example_bracket_mult_equation_roots_l1472_147298

-- Define the operation for real numbers
def bracket_mult (a b c d : ℝ) : ℝ := a * c - b * d

-- Theorem 1
theorem bracket_mult_example : bracket_mult (-4) 3 2 (-6) = 10 := by sorry

-- Theorem 2
theorem bracket_mult_equation_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ bracket_mult x (2*x - 1) (m*x + 1) m = 0) ↔ 
  (m ≤ 1/4 ∧ m ≠ 0) := by sorry

end bracket_mult_example_bracket_mult_equation_roots_l1472_147298


namespace coplanar_vectors_lambda_l1472_147269

/-- Given vectors a, b, and c in ℝ³, prove that if they are coplanar and have the specified coordinates, then the third component of c equals 65/7. -/
theorem coplanar_vectors_lambda (a b c : ℝ × ℝ × ℝ) : 
  a = (2, -1, 3) →
  b = (-1, 4, -2) →
  c.1 = 7 ∧ c.2.1 = 5 →
  (∃ (x y : ℝ), c = x • a + y • b) →
  c.2.2 = 65/7 := by
  sorry

end coplanar_vectors_lambda_l1472_147269


namespace smallest_factorizable_b_l1472_147270

/-- 
A function that checks if a quadratic expression x^2 + bx + c 
can be factored into two binomials with integer coefficients
-/
def is_factorizable (b : ℤ) (c : ℤ) : Prop :=
  ∃ (r s : ℤ), c = r * s ∧ b = r + s

/-- 
The smallest positive integer b for which x^2 + bx + 1890 
factors into a product of two binomials with integer coefficients
-/
theorem smallest_factorizable_b : 
  (∀ b : ℤ, b > 0 ∧ b < 141 → ¬(is_factorizable b 1890)) ∧ 
  (is_factorizable 141 1890) := by
  sorry

#check smallest_factorizable_b

end smallest_factorizable_b_l1472_147270


namespace quadratic_equations_solutions_l1472_147221

theorem quadratic_equations_solutions :
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
    (x₁^2 - 6*x₁ + 5 = 0 ∧ x₂^2 - 6*x₂ + 5 = 0 ∧ x₁ = 5 ∧ x₂ = 1) ∧
    (3*x₃*(2*x₃ - 1) = 4*x₃ - 2 ∧ 3*x₄*(2*x₄ - 1) = 4*x₄ - 2 ∧ x₃ = 1/2 ∧ x₄ = 2/3) ∧
    (x₅^2 - 2*Real.sqrt 2*x₅ - 2 = 0 ∧ x₆^2 - 2*Real.sqrt 2*x₆ - 2 = 0 ∧ 
     x₅ = Real.sqrt 2 + 2 ∧ x₆ = Real.sqrt 2 - 2) := by
  sorry

end quadratic_equations_solutions_l1472_147221


namespace rosie_pies_l1472_147286

/-- Represents the number of pies Rosie can make given the number of apples and pears -/
def total_pies (apples : ℕ) (pears : ℕ) : ℕ :=
  let apple_pies := (apples / 9) * 2
  let pear_pies := (pears / 15) * 3
  apple_pies + pear_pies

/-- Theorem stating that Rosie can make 12 pies with 27 apples and 30 pears -/
theorem rosie_pies : total_pies 27 30 = 12 := by
  sorry

end rosie_pies_l1472_147286


namespace intersection_point_after_rotation_l1472_147233

theorem intersection_point_after_rotation (θ : Real) : 
  0 < θ ∧ θ < π / 2 → 
  (fun φ ↦ (Real.cos φ, Real.sin φ)) (θ + π / 2) = (-Real.sin θ, Real.cos θ) := by
  sorry

end intersection_point_after_rotation_l1472_147233


namespace cone_lateral_surface_area_l1472_147225

theorem cone_lateral_surface_area 
  (r : ℝ) 
  (V : ℝ) 
  (h : ℝ) 
  (l : ℝ) : 
  r = 6 →
  V = 30 * Real.pi →
  V = (1/3) * Real.pi * r^2 * h →
  l^2 = r^2 + h^2 →
  r * l * Real.pi = 39 * Real.pi :=
by sorry

end cone_lateral_surface_area_l1472_147225


namespace find_k_l1472_147235

theorem find_k (x y z k : ℝ) 
  (h1 : 7 / (x + y) = k / (x + z)) 
  (h2 : k / (x + z) = 11 / (z - y)) : k = 18 := by
  sorry

end find_k_l1472_147235


namespace trig_identity_l1472_147277

theorem trig_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (π / 6 + α / 2) ^ 2 = 2 / 3 := by
  sorry

end trig_identity_l1472_147277


namespace algebraic_notation_correctness_l1472_147207

/-- Rules for algebraic notation --/
structure AlgebraicNotationRules where
  no_multiplication_sign : Bool
  number_before_variable : Bool
  proper_fraction : Bool
  correct_negative_placement : Bool

/-- Check if an expression follows algebraic notation rules --/
def follows_algebraic_notation (expr : String) (rules : AlgebraicNotationRules) : Bool :=
  rules.no_multiplication_sign ∧ 
  rules.number_before_variable ∧ 
  rules.proper_fraction ∧ 
  rules.correct_negative_placement

/-- Given expressions --/
def expr_A : String := "a×5"
def expr_B : String := "a7"
def expr_C : String := "3½x"
def expr_D : String := "-⅞x"

theorem algebraic_notation_correctness :
  follows_algebraic_notation expr_D 
    {no_multiplication_sign := true, 
     number_before_variable := true, 
     proper_fraction := true, 
     correct_negative_placement := true} ∧
  ¬follows_algebraic_notation expr_A 
    {no_multiplication_sign := false, 
     number_before_variable := false, 
     proper_fraction := true, 
     correct_negative_placement := true} ∧
  ¬follows_algebraic_notation expr_B
    {no_multiplication_sign := true, 
     number_before_variable := false, 
     proper_fraction := true, 
     correct_negative_placement := true} ∧
  ¬follows_algebraic_notation expr_C
    {no_multiplication_sign := true, 
     number_before_variable := true, 
     proper_fraction := false, 
     correct_negative_placement := true} :=
by sorry

end algebraic_notation_correctness_l1472_147207


namespace inequality_solution_l1472_147252

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 2) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) :=
sorry

end inequality_solution_l1472_147252


namespace first_ball_odd_given_two_odd_one_even_l1472_147275

/-- The probability of selecting an odd-numbered ball from a box of 100 balls numbered 1 to 100 -/
def prob_odd_ball : ℚ := 1/2

/-- The probability of selecting an even-numbered ball from a box of 100 balls numbered 1 to 100 -/
def prob_even_ball : ℚ := 1/2

/-- The probability of selecting two odd-numbered balls and one even-numbered ball in any order when selecting 3 balls with replacement -/
def prob_two_odd_one_even : ℚ := 3 * prob_odd_ball * prob_odd_ball * prob_even_ball

theorem first_ball_odd_given_two_odd_one_even :
  let prob_first_odd := prob_odd_ball * (prob_odd_ball * prob_even_ball + prob_even_ball * prob_odd_ball)
  prob_first_odd / prob_two_odd_one_even = 1/4 := by sorry

end first_ball_odd_given_two_odd_one_even_l1472_147275


namespace parabola_through_point_l1472_147260

def is_parabola_equation (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, f x y ↔ y^2 = 4*a*x

theorem parabola_through_point (f : ℝ → ℝ → Prop) : 
  is_parabola_equation f →
  (∀ x y : ℝ, f x y → y^2 = x) →
  f 4 (-2) →
  ∀ x y : ℝ, f x y ↔ y^2 = x :=
sorry

end parabola_through_point_l1472_147260


namespace misha_current_money_l1472_147200

/-- The amount of money Misha needs to earn -/
def additional_money : ℕ := 13

/-- The total amount Misha would have after earning the additional money -/
def total_money : ℕ := 47

/-- Misha's current money amount -/
def current_money : ℕ := total_money - additional_money

theorem misha_current_money : current_money = 34 := by
  sorry

end misha_current_money_l1472_147200


namespace distinct_collections_l1472_147245

/-- Represents the count of each letter in MATHEMATICSH -/
def letter_count : Finset (Char × ℕ) :=
  {('A', 3), ('E', 1), ('I', 1), ('T', 2), ('M', 2), ('H', 2), ('C', 1), ('S', 1)}

/-- The set of vowels in MATHEMATICSH -/
def vowels : Finset Char := {'A', 'E', 'I'}

/-- The set of consonants in MATHEMATICSH -/
def consonants : Finset Char := {'T', 'M', 'H', 'C', 'S'}

/-- The number of distinct vowel combinations -/
def vowel_combinations : ℕ := 5

/-- The number of distinct consonant combinations -/
def consonant_combinations : ℕ := 48

/-- Theorem stating the number of distinct possible collections -/
theorem distinct_collections :
  vowel_combinations * consonant_combinations = 240 :=
by sorry

end distinct_collections_l1472_147245


namespace gcd_1234_2047_l1472_147290

theorem gcd_1234_2047 : Nat.gcd 1234 2047 = 1 := by
  sorry

end gcd_1234_2047_l1472_147290


namespace min_value_expression_l1472_147261

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  ∃ (x₀ y₀ : ℝ), 2*x₀*y₀ - 2*x₀ - y₀ = 8 ∧ ∀ x y, x > 0 → y > 0 → 1/x + 2/y = 1 → 2*x*y - 2*x - y ≥ 8 :=
sorry

end min_value_expression_l1472_147261


namespace puzzle_solution_l1472_147291

theorem puzzle_solution :
  ∀ (E H O Y A : ℕ),
    (10 ≤ E * 10 + H) ∧ (E * 10 + H < 100) ∧
    (10 ≤ O * 10 + Y) ∧ (O * 10 + Y < 100) ∧
    (10 ≤ A * 10 + Y) ∧ (A * 10 + Y < 100) ∧
    (10 ≤ O * 10 + H) ∧ (O * 10 + H < 100) ∧
    (E * 10 + H = 4 * (O * 10 + Y)) ∧
    (A * 10 + Y = 4 * (O * 10 + H)) →
    (E * 10 + H) + (O * 10 + Y) + (A * 10 + Y) + (O * 10 + H) = 150 :=
by sorry


end puzzle_solution_l1472_147291


namespace inequality_range_l1472_147244

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2*x - 1| + |x + 1| > a) → a < (3/2) := by
sorry

end inequality_range_l1472_147244


namespace festival_attendance_l1472_147281

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (h1 : total_students = 1500)
  (h2 : festival_attendees = 900)
  (h3 : ∃ (girls boys : ℕ), 
    girls + boys = total_students ∧ 
    (2 * girls) / 3 + boys / 2 = festival_attendees) :
  ∃ (girls : ℕ), (2 * girls) / 3 = 600 := by
  sorry

end festival_attendance_l1472_147281


namespace line_passes_through_fixed_point_l1472_147283

/-- The line mx-y+2m+1=0 passes through the point (-2, 1) for all values of m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), m * (-2) - 1 + 2 * m + 1 = 0 := by sorry

end line_passes_through_fixed_point_l1472_147283


namespace trees_represents_41225_l1472_147212

-- Define the type for our digit mapping
def DigitMapping := Char → Option Nat

-- Define our specific mapping
def greatSuccessMapping : DigitMapping := fun c =>
  match c with
  | 'G' => some 0
  | 'R' => some 1
  | 'E' => some 2
  | 'A' => some 3
  | 'T' => some 4
  | 'S' => some 5
  | 'U' => some 6
  | 'C' => some 7
  | _ => none

-- Function to convert a string to a number using the mapping
def stringToNumber (s : String) (m : DigitMapping) : Option Nat :=
  s.foldr (fun c acc =>
    match acc, m c with
    | some n, some d => some (n * 10 + d)
    | _, _ => none
  ) (some 0)

-- Theorem statement
theorem trees_represents_41225 :
  stringToNumber "TREES" greatSuccessMapping = some 41225 := by
  sorry

end trees_represents_41225_l1472_147212


namespace line_contains_point_l1472_147231

/-- A line in the xy-plane is represented by the equation 2 - kx = -4y,
    where k is a real number. The line contains the point (3,-2) if and only if k = -2. -/
theorem line_contains_point (k : ℝ) : 2 - k * 3 = -4 * (-2) ↔ k = -2 := by
  sorry

end line_contains_point_l1472_147231


namespace parking_space_savings_l1472_147215

/-- The cost of renting a parking space for one week in dollars -/
def weekly_cost : ℕ := 10

/-- The cost of renting a parking space for one month in dollars -/
def monthly_cost : ℕ := 24

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The savings in dollars when renting a parking space by the month instead of by the week for a year -/
theorem parking_space_savings : 
  weeks_per_year * weekly_cost - months_per_year * monthly_cost = 232 := by
  sorry

end parking_space_savings_l1472_147215


namespace f_f_has_four_distinct_roots_l1472_147289

-- Define the function f
def f (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

-- State the theorem
theorem f_f_has_four_distinct_roots :
  ∃! d : ℝ, ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄) ∧
    (∀ x : ℝ, f (f d x) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
    d = 2 :=
sorry

end f_f_has_four_distinct_roots_l1472_147289


namespace exists_acute_triangle_configuration_l1472_147248

/-- A configuration of n points on a plane. -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ

/-- A triangle formed by three points. -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Predicate to check if a triangle is acute. -/
def is_acute (t : Triangle) : Prop :=
  sorry  -- Definition of acute triangle

/-- Function to get the i-th triangle from a point configuration. -/
def get_triangle (config : PointConfiguration n) (i : Fin n) : Triangle :=
  sorry  -- Definition to extract triangle from configuration

/-- Theorem stating the existence of a configuration with all acute triangles. -/
theorem exists_acute_triangle_configuration (n : ℕ) (h_odd : Odd n) (h_gt_3 : n > 3) :
  ∃ (config : PointConfiguration n), ∀ (i : Fin n), is_acute (get_triangle config i) :=
sorry

end exists_acute_triangle_configuration_l1472_147248


namespace quadratic_roots_sum_product_l1472_147227

theorem quadratic_roots_sum_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → 
  x₂^2 - 3*x₂ - 1 = 0 → 
  x₁^2 * x₂ + x₁ * x₂^2 = -3 := by
  sorry

end quadratic_roots_sum_product_l1472_147227


namespace extra_red_pencil_packs_l1472_147249

theorem extra_red_pencil_packs (total_packs : ℕ) (normal_red_per_pack : ℕ) (total_red_pencils : ℕ) :
  total_packs = 15 →
  normal_red_per_pack = 1 →
  total_red_pencils = 21 →
  ∃ (extra_packs : ℕ),
    extra_packs * 2 + total_packs * normal_red_per_pack = total_red_pencils ∧
    extra_packs = 3 :=
by sorry

end extra_red_pencil_packs_l1472_147249


namespace marble_problem_l1472_147213

theorem marble_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ) (emma : ℚ)
  (h1 : angela = a)
  (h2 : brian = 2 * a)
  (h3 : caden = 3 * brian)
  (h4 : daryl = 5 * caden)
  (h5 : emma = 2 * daryl)
  (h6 : angela + brian + caden + daryl + emma = 212) :
  a = 212 / 99 := by
  sorry

end marble_problem_l1472_147213


namespace profit_percentage_without_discount_l1472_147263

theorem profit_percentage_without_discount
  (cost_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate_with_discount : ℝ)
  (h_positive_cost : cost_price > 0)
  (h_discount : discount_rate = 0.05)
  (h_profit_with_discount : profit_rate_with_discount = 0.1875) :
  let selling_price_with_discount := cost_price * (1 - discount_rate)
  let profit_amount := cost_price * profit_rate_with_discount
  let selling_price_without_discount := cost_price + profit_amount
  let profit_rate_without_discount := (selling_price_without_discount - cost_price) / cost_price
  profit_rate_without_discount = profit_rate_with_discount :=
by sorry

end profit_percentage_without_discount_l1472_147263


namespace negation_existence_statement_l1472_147296

theorem negation_existence_statement :
  (¬ ∃ x : ℝ, x < -1 ∧ x^2 ≥ 1) ↔ (∀ x : ℝ, x < -1 → x^2 < 1) :=
by sorry

end negation_existence_statement_l1472_147296


namespace half_sufficient_not_necessary_l1472_147276

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ → ℝ → Prop := λ x y => x + 2 * a * y - 1 = 0
  line2 : ℝ → ℝ → Prop := λ x y => (a - 1) * x - a * y - 1 = 0

/-- The lines are parallel -/
def are_parallel (l : TwoLines) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l.line1 x y ↔ l.line2 (k * x) (k * y)

/-- The statement that a = 1/2 is sufficient but not necessary for the lines to be parallel -/
theorem half_sufficient_not_necessary :
  (∃ l : TwoLines, l.a = 1/2 ∧ ¬are_parallel l) ∧
  (∃ l : TwoLines, l.a ≠ 1/2 ∧ are_parallel l) ∧
  (∀ l : TwoLines, l.a = 1/2 → are_parallel l) :=
sorry

end half_sufficient_not_necessary_l1472_147276


namespace albert_more_than_joshua_l1472_147256

/-- The number of rocks collected by Joshua, Jose, and Albert -/
def rock_collection (joshua jose albert : ℕ) : Prop :=
  (jose = joshua - 14) ∧ 
  (albert = jose + 20) ∧ 
  (joshua = 80)

/-- Theorem stating that Albert collected 6 more rocks than Joshua -/
theorem albert_more_than_joshua {joshua jose albert : ℕ} 
  (h : rock_collection joshua jose albert) : albert - joshua = 6 := by
  sorry

end albert_more_than_joshua_l1472_147256


namespace power_function_range_l1472_147210

-- Define the power function f
def f (x : ℝ) : ℝ := x^2

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 1

-- State the theorem
theorem power_function_range (m : ℝ) : 
  (f (Real.sqrt 3) = 3) → 
  (∀ x ∈ Set.Icc m 2, g x ∈ Set.Icc 1 5) → 
  (m ∈ Set.Icc (-2) 0) :=
by sorry

end power_function_range_l1472_147210
