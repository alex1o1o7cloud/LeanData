import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_ratio_theorem_l528_52833

noncomputable section

open Real

/-- The ratio of the shaded area to the area of the circle with diameter CD -/
def shaded_area_ratio (r : ℝ) : ℝ :=
  let ab := 2 * r
  let ac := 1.25 * r
  let cb := 0.75 * r
  let cd := (sqrt 15 * r) / 4
  let shaded_area := π * r^2 / 2 - (π * (1.25 * r / 2)^2 / 2 + π * (0.75 * r / 2)^2 / 2) + π * (1.25 * r / 4)^2 / 2
  let circle_area := π * cd^2
  shaded_area / circle_area

/-- The main theorem stating the ratio of shaded area to circle area -/
theorem shaded_area_ratio_theorem (r : ℝ) (h : r > 0) :
  ∃ (k : ℝ), shaded_area_ratio r = k := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_ratio_theorem_l528_52833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_valid_board_configuration_l528_52808

/-- Represents a board configuration -/
def Board := Fin 8 → Fin 8 → ℕ

/-- Represents a domino placement on the board -/
def Domino := { p : (Fin 8 × Fin 8) × (Fin 8 × Fin 8) // 
  (p.1.1 = p.2.1 ∧ p.1.2.succ = p.2.2) ∨ 
  (p.1.2 = p.2.2 ∧ p.1.1.succ = p.2.1) }

/-- The set of all possible domino tilings of the board -/
def AllTilings : Set (Set Domino) := sorry

/-- The sum of numbers on a domino for a given board configuration -/
def dominoSum (b : Board) (d : Domino) : ℕ := 
  b d.val.1.1 d.val.1.2 + b d.val.2.1 d.val.2.2

theorem existence_of_valid_board_configuration : 
  ∃ (b : Board), 
    (∀ i j, b i j ≤ 32) ∧ 
    (∀ t, t ∈ AllTilings → ∀ d1 d2, d1 ∈ t → d2 ∈ t → d1 ≠ d2 → dominoSum b d1 ≠ dominoSum b d2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_valid_board_configuration_l528_52808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_value_monotonicity_of_f_l528_52883

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

theorem tangent_line_parallel_implies_a_value (a : ℝ) :
  (∀ x > 0, ∃ y, f a x = y) →
  (∃ k b : ℝ, ∀ x, (deriv (f a)) 1 * (x - 1) + f a 1 = 4 * x + 1) →
  a = 3 :=
sorry

theorem monotonicity_of_f (a : ℝ) :
  (a > 0 → ∀ x > 0, (deriv (f a)) x > 0) ∧
  (a < 0 → (∀ x, 0 < x → x < -1/a → (deriv (f a)) x > 0) ∧
           (∀ x > -1/a, (deriv (f a)) x < 0)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_value_monotonicity_of_f_l528_52883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_tuple_l528_52849

def is_valid_tuple (a : List Nat) : Prop :=
  (List.foldl Nat.lcm 1 a = 1985) ∧
  (∀ i j, i < a.length ∧ j < a.length → Nat.gcd (a.get! i) (a.get! j) ≠ 1) ∧
  (∃ k : Nat, a.prod = k * k) ∧
  (243 ∣ a.prod)

theorem smallest_valid_tuple :
  (∀ n < 7, ¬ ∃ a : List Nat, a.length = n ∧ is_valid_tuple a) ∧
  (∃! a : List Nat, a.length = 7 ∧ is_valid_tuple a ∧ a = [15, 21, 57, 105, 285, 399, 665]) :=
by sorry

#check smallest_valid_tuple

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_tuple_l528_52849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_item_sum_cost_l528_52895

-- Define the cost of each item in cents
def pencil_cost : ℕ := sorry
def eraser_cost : ℕ := sorry
def notebook_cost : ℕ := sorry

-- Define the conditions
axiom total_cost : 9 * pencil_cost + 7 * eraser_cost + 4 * notebook_cost = 220
axiom whole_numbers : pencil_cost > 0 ∧ eraser_cost > 0 ∧ notebook_cost > 0
axiom cost_order : pencil_cost > notebook_cost ∧ notebook_cost > eraser_cost

-- Theorem to prove
theorem item_sum_cost : pencil_cost + eraser_cost + notebook_cost = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_item_sum_cost_l528_52895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l528_52815

/-- The area of a sector given its radius and central angle -/
noncomputable def sectorArea (r : ℝ) (α : ℝ) : ℝ := (1/2) * r^2 * α

/-- The perimeter of a sector given its radius and central angle -/
noncomputable def sectorPerimeter (r : ℝ) (α : ℝ) : ℝ := 2*r + r*α

theorem sector_max_area :
  ∃ (r α : ℝ), 
    sectorPerimeter r α = 20 ∧ 
    r = 5 ∧ 
    α = 2 ∧
    sectorArea r α = 50 ∧
    ∀ (r' α' : ℝ), sectorPerimeter r' α' = 20 → sectorArea r' α' ≤ sectorArea r α :=
by
  sorry

#check sector_max_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l528_52815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_most_accurate_l528_52897

/-- Represents a categorical feature (factor variable) -/
structure CategoricalFeature where
  name : String

/-- Represents a two-way classified contingency table -/
structure ContingencyTable where
  feature1 : CategoricalFeature
  feature2 : CategoricalFeature
  data : Matrix (Fin 2) (Fin 2) ℕ

/-- Represents a method for testing correlation or independence of categorical features -/
structure TestMethod where
  name : String

/-- The independence test method -/
def independenceTest : TestMethod :=
  { name := "Independence Test" }

/-- A set of commonly used test methods -/
def commonlyUsedMethods : Set TestMethod :=
  {independenceTest}

/-- Measures the accuracy of a test method for a given contingency table -/
noncomputable def accuracy (method : TestMethod) (table : ContingencyTable) : ℝ :=
  sorry

/-- Theorem stating that the independence test is the most accurate method -/
theorem independence_test_most_accurate (table : ContingencyTable) :
  ∀ m ∈ commonlyUsedMethods, 
    m ≠ independenceTest → accuracy independenceTest table ≥ accuracy m table :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_most_accurate_l528_52897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_mixture_intensity_l528_52884

theorem paint_mixture_intensity 
  (original_intensity : ℝ) 
  (replacement_intensity : ℝ) 
  (replacement_fraction : ℝ) : 
  original_intensity = 0.15 →
  replacement_intensity = 0.25 →
  replacement_fraction = 1.5 →
  let original_amount := 100
  let replacement_amount := replacement_fraction * original_amount
  let total_amount := original_amount + replacement_amount
  let original_pigment := original_intensity * original_amount
  let replacement_pigment := replacement_intensity * replacement_amount
  let total_pigment := original_pigment + replacement_pigment
  let new_intensity := total_pigment / total_amount
  new_intensity = 0.21 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_mixture_intensity_l528_52884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l528_52881

/-- The lateral surface area of a frustum of a regular pyramid -/
noncomputable def lateralSurfaceArea (topBase bottomBase slantHeight : ℝ) : ℝ :=
  4 * (topBase + bottomBase) / 2 * Real.sqrt (slantHeight^2 - ((bottomBase - topBase) / 2)^2)

/-- Theorem stating the lateral surface area of a specific frustum -/
theorem frustum_lateral_surface_area :
  lateralSurfaceArea 1 3 2 = 8 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l528_52881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_n_1992_l528_52834

-- Define g(x) as the largest odd divisor of x
def g (x : ℕ) : ℕ :=
  sorry

-- Define f(x) as given in the problem
def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then
    x / 2 + 2 / (g x)
  else
    2 ^ ((x + 1) / 2)

-- Define the sequence x_n
def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => f (x n)

-- Theorem statement
theorem exists_unique_n_1992 : ∃! n : ℕ, x n = 1992 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_n_1992_l528_52834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_book_width_is_443_l528_52844

/-- The widths of the books on Mary's top shelf, in centimeters. -/
noncomputable def book_widths : List ℝ := [8, 1, 3.5, 4, 12, 0.5, 2]

/-- The number of books on Mary's top shelf. -/
def num_books : ℕ := 7

/-- The average width of the books on Mary's top shelf, in centimeters. -/
noncomputable def average_width : ℝ := (List.sum book_widths) / num_books

/-- The approximate equality relation for real numbers. -/
def approx_equal (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

theorem average_book_width_is_443 :
  approx_equal average_width 4.43 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_book_width_is_443_l528_52844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_full_capacity_l528_52871

-- Define the tank's capacity
def tank_capacity : ℚ → Prop := λ c => c > 0

-- Define the initial state of the tank (3/4 full)
def initial_state (c : ℚ) : Prop := tank_capacity c ∧ c * (3/4) = c - (c / 4)

-- Define the state after adding 7 gallons (7/8 full)
def final_state (c : ℚ) : Prop := tank_capacity c ∧ c * (7/8) = c * (3/4) + 7

-- Theorem statement
theorem tank_full_capacity :
  ∀ c : ℚ, tank_capacity c → initial_state c → final_state c → c = 56 := by
  intro c hc hi hf
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_full_capacity_l528_52871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_is_one_l528_52806

-- Define the complex number z
noncomputable def z : ℂ := (2 * Complex.I^3) / (Complex.I - 1)

-- Theorem statement
theorem imaginary_part_of_z_is_one :
  z.im = 1 := by
  -- Simplify z
  have h1 : z = -1 + Complex.I := by
    -- Proof steps here
    sorry
  
  -- The imaginary part of a + bi is b
  have h2 : (-1 + Complex.I).im = 1 := by
    -- Proof steps here
    sorry
  
  -- Combine the results
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_is_one_l528_52806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stripe_area_on_cylinder_specific_stripe_area_l528_52820

/-- The area of a stripe wrapped around a cylindrical tower -/
theorem stripe_area_on_cylinder (diameter : ℝ) (stripe_width : ℝ) (revolutions : ℝ) :
  diameter > 0 → stripe_width > 0 → revolutions > 0 →
  let circumference := π * diameter
  let stripe_length := revolutions * circumference
  let stripe_area := stripe_width * stripe_length
  stripe_area = stripe_width * revolutions * π * diameter := by
  sorry

/-- The specific problem instance -/
theorem specific_stripe_area :
  let diameter := (40 : ℝ)
  let stripe_width := (4 : ℝ)
  let revolutions := (1.5 : ℝ)
  let stripe_area := stripe_width * revolutions * π * diameter
  stripe_area = 240 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stripe_area_on_cylinder_specific_stripe_area_l528_52820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l528_52816

noncomputable def m : ℝ × ℝ := (Real.sqrt 2 / 2, -Real.sqrt 2 / 2)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

noncomputable def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem vector_problem (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) :
  (m.1 * (n x).1 + m.2 * (n x).2 = 0 → Real.tan x = 1) ∧
  (angle_between m (n x) = Real.pi / 3 → x = 5 * Real.pi / 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l528_52816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_spherical_specific_l528_52854

noncomputable def cylindrical_to_spherical (ρ θ z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (ρ^2 + z^2)
  let φ := θ
  let θ' := Real.arccos (z / r)
  (r, φ, θ')

theorem cylindrical_to_spherical_specific :
  let (r, φ, θ) := cylindrical_to_spherical 3 (π/3) 3
  r = 3 * Real.sqrt 2 ∧ φ = π/3 ∧ θ = π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_spherical_specific_l528_52854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_is_sqrt_7_l528_52819

/-- The line y = x + 1 -/
def line (x : ℝ) : ℝ := x + 1

/-- The circle (x-3)^2 + y^2 = 1 -/
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

/-- The minimum length of the tangent line -/
noncomputable def min_tangent_length : ℝ := Real.sqrt 7

/-- Theorem: The minimum length of a tangent line from a point on the line y = x + 1
    to the circle (x-3)^2 + y^2 = 1 is √7 -/
theorem min_tangent_length_is_sqrt_7 :
  ∃ (x₀ y₀ : ℝ), y₀ = line x₀ ∧
  (∀ (x y : ℝ), y = line x →
    circle_eq x y →
    (x - x₀)^2 + (y - y₀)^2 ≥ min_tangent_length^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_is_sqrt_7_l528_52819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_inequality_holds_l528_52869

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + m * Real.log x

-- Part 1: Theorem for f(x) to be increasing
theorem f_increasing (m : ℝ) : 
  (∀ x > 0, HasDerivAt (f m) (2*x - 4 + m/x) x) → m ≥ 2 :=
sorry

-- Part 2: Theorem for the inequality when m = 3
theorem inequality_holds (x : ℝ) (h : x > 0) : 
  (1/9) * x^3 - f 3 x > 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_inequality_holds_l528_52869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_cases_calculation_l528_52856

def boxes_sold_trefoils : ℕ := 10
def boxes_sold_samoas : ℕ := 15
def boxes_sold_thin_mints : ℕ := 20

def boxes_per_case_trefoils : ℕ := 6
def boxes_per_case_samoas : ℕ := 5
def boxes_per_case_thin_mints : ℕ := 10

noncomputable def cases_needed (boxes_sold : ℕ) (boxes_per_case : ℕ) : ℕ :=
  Nat.ceil (boxes_sold / boxes_per_case : ℚ)

theorem cookie_cases_calculation :
  cases_needed boxes_sold_trefoils boxes_per_case_trefoils = 2 ∧
  cases_needed boxes_sold_samoas boxes_per_case_samoas = 3 ∧
  cases_needed boxes_sold_thin_mints boxes_per_case_thin_mints = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_cases_calculation_l528_52856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_length_ratio_lighting_time_correct_l528_52859

/-- Represents a candle with a given burn time in hours -/
structure Candle where
  burnTime : ℚ

/-- The time in hours from 1 PM when the candles are lit -/
noncomputable def lightingTime : ℚ := 36 / 60

/-- The time in hours from 1 PM to 4 PM -/
def totalTime : ℚ := 3

theorem candle_length_ratio (candle1 candle2 : Candle)
  (h1 : candle1.burnTime = 3)
  (h2 : candle2.burnTime = 4) :
  1 - totalTime / candle2.burnTime = 2 * (1 - totalTime / candle1.burnTime) :=
by sorry

theorem lighting_time_correct :
  lightingTime = 36 / 60 :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_length_ratio_lighting_time_correct_l528_52859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johnson_farm_corn_cost_l528_52811

/-- The cost of cultivating corn per acre at Johnson Farm -/
noncomputable def corn_cost_per_acre : ℝ :=
  let total_land : ℝ := 500
  let wheat_land : ℝ := 200
  let corn_land : ℝ := total_land - wheat_land
  let total_budget : ℝ := 18600
  let wheat_cost_per_acre : ℝ := 30
  let wheat_total_cost : ℝ := wheat_land * wheat_cost_per_acre
  let corn_total_cost : ℝ := total_budget - wheat_total_cost
  corn_total_cost / corn_land

/-- Theorem stating that the cost of cultivating corn per acre at Johnson Farm is $42 -/
theorem johnson_farm_corn_cost : corn_cost_per_acre = 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johnson_farm_corn_cost_l528_52811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_high_flyers_loss_percentage_l528_52887

theorem high_flyers_loss_percentage :
  ∀ y : ℕ+,
  let games_won := 17 * y
  let games_lost := 8 * y
  let games_cancelled := 6
  let total_games := games_won + games_lost + games_cancelled
  let loss_percentage := (games_lost : ℝ) / total_games * 100
  ∃ ε : ℝ, ε ≥ 0 ∧ ε < 0.5 ∧ (30 - ε ≤ loss_percentage ∧ loss_percentage < 30 + ε) :=
by
  sorry

#check high_flyers_loss_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_high_flyers_loss_percentage_l528_52887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_values_l528_52836

def is_multiple_of_three (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem count_values (x : ℤ) : 
  (∃ n : ℕ, n = 13 ∧ 
    (∀ y : ℤ, (⌈Real.sqrt (↑y : ℝ)⌉ = 20 ∧ is_multiple_of_three y) ↔ 
      ∃ i : ℕ, i ≤ n ∧ y = 363 + 3 * (i - 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_values_l528_52836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rational_state_l528_52896

/-- Represents the state of the numbers on the board -/
structure BoardState where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The initial state of the board -/
noncomputable def initial_state : BoardState :=
  { x := 1 - Real.sqrt 2
  , y := Real.sqrt 2
  , z := 1 + Real.sqrt 2 }

/-- The transition function for the board state -/
def next_state (s : BoardState) : BoardState :=
  { x := s.x^2 + s.x*s.y + s.y^2
  , y := s.y^2 + s.y*s.z + s.z^2
  , z := s.z^2 + s.x*s.z + s.x^2 }

/-- Predicate to check if all numbers in a board state are rational -/
def all_rational (s : BoardState) : Prop :=
  (∃ q : ℚ, ↑q = s.x) ∧ (∃ q : ℚ, ↑q = s.y) ∧ (∃ q : ℚ, ↑q = s.z)

/-- The main theorem stating that it's impossible to reach a state where all numbers are rational -/
theorem no_rational_state :
  ∀ n : ℕ, ¬(all_rational ((next_state^[n]) initial_state)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rational_state_l528_52896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_coins_percentage_is_45_5_l528_52876

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  beads_rings_percent : ℚ
  beads_percent : ℚ
  silver_coins_percent : ℚ

/-- Calculates the percentage of gold coins in the urn -/
def gold_coins_percent (uc : UrnComposition) : ℚ :=
  let coins_percent := 100 - uc.beads_rings_percent
  let gold_coins_percent_of_coins := 100 - uc.silver_coins_percent
  coins_percent * gold_coins_percent_of_coins / 100

/-- Theorem stating that the percentage of gold coins in the urn is 45.5% -/
theorem gold_coins_percentage_is_45_5 (uc : UrnComposition) 
  (h1 : uc.beads_rings_percent = 30)
  (h2 : uc.beads_percent = uc.beads_rings_percent / 2)
  (h3 : uc.silver_coins_percent = 35) :
  gold_coins_percent uc = 455/10 := by
  sorry

#eval (455 : ℚ) / 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_coins_percentage_is_45_5_l528_52876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_area_sum_l528_52845

/-- Represents a point in 2D space with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : Real :=
  Real.sqrt (((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2) : Real)

/-- Represents a trapezoid with four vertices -/
structure Trapezoid where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the perimeter of a trapezoid -/
noncomputable def perimeter (t : Trapezoid) : Real :=
  distance t.v1 t.v2 + distance t.v2 t.v3 + distance t.v3 t.v4 + distance t.v4 t.v1

/-- Calculates the area of a trapezoid -/
noncomputable def area (t : Trapezoid) : Real :=
  let base1 := (t.v2.x - t.v1.x).natAbs
  let base2 := (t.v3.x - t.v4.x).natAbs
  let height := (t.v3.y - t.v2.y).natAbs
  (base1 + base2 : Real) * height / 2

/-- Theorem: The sum of perimeter and area for the given trapezoid is 42 + 4√5 -/
theorem trapezoid_perimeter_area_sum : 
  let t := Trapezoid.mk 
    (Point.mk 2 3) 
    (Point.mk 7 3) 
    (Point.mk 9 7) 
    (Point.mk 0 7)
  perimeter t + area t = 42 + 4 * Real.sqrt 5 := by
  sorry

#eval "Trapezoid perimeter and area sum theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_area_sum_l528_52845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l528_52837

theorem problem_solution : 
  (Real.sqrt 3 * Real.sqrt 6 - (Real.sqrt (1/2) - Real.sqrt 8) = (9 * Real.sqrt 2) / 2) ∧
  (let x := Real.sqrt 5; (1 + 1/x) / ((x^2 + x) / x) = Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l528_52837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_a_range_l528_52880

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (2*a + 1)/a - 1/(a^2 * x)

-- Statement 1
theorem f_monotone_increasing (a m n : ℝ) (ha : a > 0) (hmn : m * n > 0) :
  StrictMono (fun x => f a x) := by
  sorry

-- Statement 2
theorem a_range (a m n : ℝ) (hmn : 0 < m ∧ m < n) 
  (hf : Set.range (fun x => f a x) = Set.Icc m n) :
  a > (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_a_range_l528_52880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_price_is_correct_l528_52812

/-- The original price of the dress -/
def original_price : ℚ := 250

/-- The initial discount percentage -/
def initial_discount : ℚ := 40

/-- The additional event day discount percentage -/
def event_discount : ℚ := 25

/-- The price after the initial discount -/
noncomputable def price_after_initial_discount : ℚ := original_price * (1 - initial_discount / 100)

/-- The final price after both discounts -/
noncomputable def final_price : ℚ := price_after_initial_discount * (1 - event_discount / 100)

/-- Theorem stating that the final price is $112.50 -/
theorem final_price_is_correct : final_price = 112.50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_price_is_correct_l528_52812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_food_for_children_l528_52824

/-- Represents the amount of food required for one adult relative to one child -/
def adult_food_ratio : ℚ := 9 / 7

/-- Represents the total number of children that can be fed with the full meal -/
def total_children_fed : ℕ := 90

/-- Represents the number of adults who have already eaten -/
def adults_eaten : ℕ := 14

/-- Calculates the number of children that can be fed with the remaining food -/
def remaining_children_fed : ℕ := total_children_fed - Int.toNat ((adults_eaten : ℚ) * adult_food_ratio).ceil

theorem remaining_food_for_children :
  remaining_children_fed = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_food_for_children_l528_52824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_draw_equal_chance_ellipse_condition_no_always_negative_quadratic_l528_52865

-- Define a random draw method
def random_draw : Type → Type := sorry

-- Define the property of equal chance of selection
def equal_chance (method : Type → Type) : Prop :=
  ∀ T : Type, ∀ x y : T, (∃ s : Set T, x ∈ s ∧ y ∈ s) → 
    (∃ p : ℝ, p > 0 ∧ p < 1)

-- Define the equation of an ellipse
def is_ellipse (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (5 - m) + y^2 / (m + 3) = 1

-- Define the necessary condition for the ellipse
def necessary_condition (m : ℝ) : Prop := -3 < m ∧ m < 5

theorem random_draw_equal_chance :
  equal_chance random_draw :=
sorry

theorem ellipse_condition :
  (∀ m : ℝ, is_ellipse m → necessary_condition m) ∧
  (∃ m : ℝ, necessary_condition m ∧ ¬is_ellipse m) :=
sorry

theorem no_always_negative_quadratic :
  ¬(∃ a : ℝ, ∀ x : ℝ, x^2 + 2*x + a < 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_draw_equal_chance_ellipse_condition_no_always_negative_quadratic_l528_52865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_length_is_three_l528_52855

/-- Calculates the length of a boat given its properties and the effect of added mass. -/
noncomputable def boatLength (breadth : ℝ) (sinkDepth : ℝ) (addedMass : ℝ) (waterDensity : ℝ) (gravity : ℝ) : ℝ :=
  (addedMass * gravity) / (waterDensity * gravity * breadth * sinkDepth)

/-- Theorem stating that under given conditions, the boat length is 3 meters. -/
theorem boat_length_is_three :
  let breadth : ℝ := 2
  let sinkDepth : ℝ := 0.02
  let addedMass : ℝ := 120
  let waterDensity : ℝ := 1000
  let gravity : ℝ := 9.81
  boatLength breadth sinkDepth addedMass waterDensity gravity = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_length_is_three_l528_52855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l528_52874

theorem problem_solution :
  (∀ x : ℝ, x ≠ 0 → x - 1/x = 2 → x^2 + 1/x^2 = 6) ∧
  (∀ a : ℝ, a ≠ 0 → a^2 + 1/a^2 = 4 → a - 1/a = Real.sqrt 2 ∨ a - 1/a = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l528_52874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_difference_zero_l528_52825

theorem complex_power_difference_zero : 
  (1 + Complex.I)^20 - (1 - Complex.I)^20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_difference_zero_l528_52825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_conditions_l528_52891

/-- The cubic function f(x) = 2x³ - 6x + k -/
def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x + k

/-- Theorem stating the conditions for the number of zero points of f(x) -/
theorem zero_points_conditions (k : ℝ) : 
  ((∃! x, f k x = 0) ↔ (k < -4 ∨ k > 4)) ∧
  ((∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f k x = 0 ∧ f k y = 0 ∧ f k z = 0) ↔ (-4 < k ∧ k < 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_conditions_l528_52891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pavilion_pillar_E_height_l528_52870

/-- Regular octagon with pillars -/
structure Pavilion where
  /-- Side length of the regular octagon -/
  side_length : ℝ
  /-- Height of pillar at vertex A -/
  height_A : ℝ
  /-- Height of pillar at vertex B -/
  height_B : ℝ
  /-- Height of pillar at vertex C -/
  height_C : ℝ

/-- Calculate the height of pillar E given a Pavilion -/
noncomputable def height_E (p : Pavilion) : ℝ :=
  40 + 24 * (Real.sqrt (2 - Real.sqrt 2)) / Real.sqrt 7

/-- Theorem stating the height of pillar E for a specific Pavilion -/
theorem pavilion_pillar_E_height :
  let p : Pavilion := {
    side_length := 10,
    height_A := 16,
    height_B := 11,
    height_C := 13
  }
  height_E p = 40 + 24 * (Real.sqrt (2 - Real.sqrt 2)) / Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pavilion_pillar_E_height_l528_52870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_theorem_l528_52804

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * ((x - 1) / 2) + 2

-- State the theorem
theorem function_value_theorem (a : ℝ) :
  (∀ x, f (2 * x + 1) = 3 * x + 2) → f a = 2 → a = 1 := by
  intro h1 h2
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_theorem_l528_52804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l528_52810

open Real

theorem trigonometric_equation_solution (k : ℤ) :
  let x : ℝ := (2 * π / 3) + 2 * π * k
  8.444 * (tan x / (2 - 1 / (cos x)^2)) * (sin (3 * x) - sin x) = 2 / ((1 / tan x)^2 - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l528_52810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sequence_equals_two_l528_52899

/-- The limit of the sequence (2n^2 + 5)/(n^2 - 3n) as n approaches infinity is 2 -/
theorem limit_sequence_equals_two :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((2 * n^2 + 5 : ℝ) / (n^2 - 3*n)) - 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sequence_equals_two_l528_52899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_monotonic_increasing_l528_52828

open Real

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * log x

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := exp x - a / x

-- Theorem statement
theorem max_a_for_monotonic_increasing :
  (∀ a : ℝ, a ≤ (exp 1) → ∀ x ∈ Set.Ioo 1 2, 0 ≤ f_derivative a x) ∧
  (∀ ε > 0, ∃ x ∈ Set.Ioo 1 2, f_derivative ((exp 1) + ε) x < 0) :=
by
  sorry

#check max_a_for_monotonic_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_monotonic_increasing_l528_52828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_f_beta_value_l528_52857

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (5 * Real.pi / 4 - x) - Real.cos (Real.pi / 4 + x)

-- Theorem for monotonic increase
theorem f_monotonic_increase (k : ℤ) : 
  StrictMonoOn f (Set.Icc ((2 * k : ℝ) * Real.pi - Real.pi / 4) ((2 * k : ℝ) * Real.pi + 3 * Real.pi / 4)) := by
  sorry

-- Theorem for f(β) value
theorem f_beta_value (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β ≤ Real.pi / 2)
  (h4 : Real.cos (α - β) = 3 / 5) (h5 : Real.cos (α + β) = -3 / 5) : 
  f β = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_f_beta_value_l528_52857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jawbreakers_per_package_l528_52867

def jawbreakers_eaten : ℕ := 20
def jawbreakers_left : ℕ := 4

theorem jawbreakers_per_package : 
  jawbreakers_eaten + jawbreakers_left = 24 := by
  rfl

#eval jawbreakers_eaten + jawbreakers_left

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jawbreakers_per_package_l528_52867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l528_52882

-- Define the triangle ABC
variable (A B C D E : ℝ × ℝ)

-- Define vectors
def AB (A B : ℝ × ℝ) : ℝ × ℝ := B - A
def AC (A C : ℝ × ℝ) : ℝ × ℝ := C - A
def AD (A D : ℝ × ℝ) : ℝ × ℝ := D - A
def AE (A E : ℝ × ℝ) : ℝ × ℝ := E - A
def BC (B C : ℝ × ℝ) : ℝ × ℝ := C - B
def BD (B D : ℝ × ℝ) : ℝ × ℝ := D - B
def DE (D E : ℝ × ℝ) : ℝ × ℝ := E - D
def EC (E C : ℝ × ℝ) : ℝ × ℝ := C - E

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the conditions
variable (h1 : dot_product (BC B C) (BC B C) = 36) -- BC = 6
variable (h2 : BD B D = DE D E) -- BD = DE
variable (h3 : DE D E = EC E C) -- DE = EC
variable (h4 : dot_product (AD A D) (AE A E) = 8) -- AD · AE = 8

-- Theorem to prove
theorem triangle_properties :
  (AD A D = (1/2 : ℝ) • (AB A B) + (1/2 : ℝ) • (AE A E)) ∧
  (dot_product (AB A B) (AB A B) + dot_product (AC A C) (AC A C) = 36) ∧
  (dot_product (AB A B) (AC A C) = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l528_52882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_process_result_l528_52846

noncomputable def truncate_to_two_decimals (x : ℝ) : ℝ :=
  ⌊x * 100⌋ / 100

noncomputable def process (α : ℝ) : ℝ :=
  truncate_to_two_decimals ((truncate_to_two_decimals α) / α)

theorem process_result (α : ℝ) (h : α > 0) :
  ∃ n : ℕ, n ≤ 100 ∧ process α = n / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_process_result_l528_52846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l528_52840

/-- Calculates the length of a train given the parameters of two trains passing each other. -/
noncomputable def calculate_train_length (length_A : ℝ) (speed_A : ℝ) (speed_B : ℝ) (time : ℝ) : ℝ :=
  let relative_speed := (speed_A + speed_B) * 1000 / 3600
  relative_speed * time - length_A

/-- Theorem stating that given the specified conditions, the length of Train B is approximately 279.95 meters. -/
theorem train_length_calculation (length_A : ℝ) (speed_A : ℝ) (speed_B : ℝ) (time : ℝ) 
  (h1 : length_A = 220)
  (h2 : speed_A = 120)
  (h3 : speed_B = 80)
  (h4 : time = 9) :
  |calculate_train_length length_A speed_A speed_B time - 279.95| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l528_52840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_IV_in_rectangle_C_l528_52850

-- Define the tiles and their properties
structure Tile :=
  (top : ℕ) (right : ℕ) (bottom : ℕ) (left : ℕ)

def tile_I : Tile := ⟨5, 3, 6, 2⟩
def tile_II : Tile := ⟨3, 5, 2, 6⟩
def tile_III : Tile := ⟨0, 7, 1, 3⟩
def tile_IV : Tile := ⟨3, 5, 6, 4⟩

-- Define the rectangles
inductive Rectangle
| A | B | C | D

-- Define the placement of tiles
def Placement := Rectangle → Tile

-- Sum of numbers on a tile
def tile_sum (t : Tile) : ℕ := t.top + t.right + t.bottom + t.left

-- Define adjacency for rectangles
def are_adjacent (r1 r2 : Rectangle) : Prop :=
  (r1 = Rectangle.A ∧ r2 = Rectangle.B) ∨
  (r1 = Rectangle.B ∧ r2 = Rectangle.C) ∨
  (r1 = Rectangle.C ∧ r2 = Rectangle.D) ∨
  (r1 = Rectangle.B ∧ r2 = Rectangle.A) ∨
  (r1 = Rectangle.C ∧ r2 = Rectangle.B) ∨
  (r1 = Rectangle.D ∧ r2 = Rectangle.C)

-- Define the side that is adjacent between two rectangles
inductive Side
| Top | Right | Bottom | Left

def adjacent_side (r1 r2 : Rectangle) : Side :=
  match r1, r2 with
  | Rectangle.A, Rectangle.B => Side.Right
  | Rectangle.B, Rectangle.C => Side.Bottom
  | Rectangle.C, Rectangle.D => Side.Left
  | Rectangle.B, Rectangle.A => Side.Left
  | Rectangle.C, Rectangle.B => Side.Top
  | Rectangle.D, Rectangle.C => Side.Right
  | _, _ => Side.Top  -- Default case, shouldn't occur in valid adjacencies

-- Adjacent tiles have equal sums
def adjacent_sums_equal (p : Placement) : Prop :=
  ∀ r1 r2 : Rectangle, are_adjacent r1 r2 → tile_sum (p r1) = tile_sum (p r2)

-- Common sides of adjacent tiles have the same numbers
def common_sides_match (p : Placement) : Prop :=
  ∀ r1 r2 : Rectangle, are_adjacent r1 r2 →
    match adjacent_side r1 r2 with
    | Side.Top    => (p r1).top = (p r2).bottom
    | Side.Right  => (p r1).right = (p r2).left
    | Side.Bottom => (p r1).bottom = (p r2).top
    | Side.Left   => (p r1).left = (p r2).right

-- The main theorem
theorem tile_IV_in_rectangle_C (p : Placement) 
  (h1 : p Rectangle.A = tile_I ∨ p Rectangle.A = tile_II ∨ p Rectangle.A = tile_III ∨ p Rectangle.A = tile_IV)
  (h2 : p Rectangle.B = tile_I ∨ p Rectangle.B = tile_II ∨ p Rectangle.B = tile_III ∨ p Rectangle.B = tile_IV)
  (h3 : p Rectangle.C = tile_I ∨ p Rectangle.C = tile_II ∨ p Rectangle.C = tile_III ∨ p Rectangle.C = tile_IV)
  (h4 : p Rectangle.D = tile_I ∨ p Rectangle.D = tile_II ∨ p Rectangle.D = tile_III ∨ p Rectangle.D = tile_IV)
  (h5 : adjacent_sums_equal p)
  (h6 : common_sides_match p)
  : p Rectangle.C = tile_IV := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_IV_in_rectangle_C_l528_52850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_probability_l528_52889

-- Define the set of possible values for m and n
def M : Finset Int := {1, 2, 3, 4}
def N : Finset Int := {-12, -8, -4, -2}

-- Define the function f(x)
def f (m n x : ℝ) : ℝ := x^3 + m*x + n

-- Define the condition for f(x) to have a root in [1, 2]
def has_root_in_interval (m n : ℝ) : Prop :=
  f m n 1 ≤ 0 ∧ f m n 2 ≥ 0

-- Count the number of (m, n) pairs satisfying the condition
def count_satisfying_pairs : ℕ := 11

-- Total number of possible (m, n) pairs
def total_pairs : ℕ := M.card * N.card

-- The main theorem
theorem root_probability :
  (count_satisfying_pairs : ℚ) / total_pairs = 11 / 16 := by
  sorry

#eval total_pairs -- To verify the total number of pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_probability_l528_52889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l528_52847

/-- Definition of the ellipse -/
def is_ellipse (a b : ℝ) (h : a > b ∧ b > 0) : (ℝ × ℝ) → Prop :=
  λ p ↦ p.1^2 / a^2 + p.2^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- Definition of the length of AB when l is perpendicular to x-axis -/
noncomputable def AB_length (a b : ℝ) : ℝ := 2 * b^2 / a

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties (a b : ℝ) (h : a > b ∧ b > 0) 
  (h_ecc : eccentricity a b = 1/2) (h_AB : AB_length a b = 3) :
  (a = 2 ∧ b = Real.sqrt 3) ∧
  (∃ P : ℝ × ℝ, P.1 = 4 ∧ P.2 = 0 ∧
    ∀ X : ℝ × ℝ, X.2 = 0 →
      ∃ A B : ℝ × ℝ, is_ellipse a b h A ∧ is_ellipse a b h B ∧
        (A.1 - P.1) * (B.2 - P.2) = (B.1 - P.1) * (A.2 - P.2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l528_52847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_sum_powers_of_two_l528_52830

theorem unit_digit_sum_powers_of_two : ∃ (k : ℕ), k > 0 ∧ (((2^2023 : ℕ) + (2^2022 : ℕ) + (2^2021 : ℕ) + (2^2020 : ℕ) + 2^2 + 2 + 1) % 10 = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_sum_powers_of_two_l528_52830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_confidence_interval_lower_bound_gt_50_784_population_mean_51_in_interval_l528_52875

/-- Represents a normal distribution with given parameters -/
structure NormalDistribution where
  stdDev : ℝ
  sampleSize : ℝ
  lowerBound : ℝ

/-- Calculates the lower bound of the confidence interval -/
noncomputable def confidenceIntervalLowerBound (d : NormalDistribution) (z : ℝ) : ℝ :=
  d.lowerBound + 3 * d.stdDev + z * (d.stdDev / Real.sqrt d.sampleSize)

theorem confidence_interval_lower_bound_gt_50_784
  (d : NormalDistribution)
  (h1 : d.stdDev = 2)
  (h2 : d.sampleSize = 25)
  (h3 : d.lowerBound > 44)
  (z : ℝ)
  (h4 : z = 1.96) :
  confidenceIntervalLowerBound d z > 50.784 :=
by sorry

theorem population_mean_51_in_interval
  (d : NormalDistribution)
  (h1 : d.stdDev = 2)
  (h2 : d.sampleSize = 25)
  (h3 : d.lowerBound > 44)
  (z : ℝ)
  (h4 : z = 1.96) :
  51 > confidenceIntervalLowerBound d z ∧ 51 < confidenceIntervalLowerBound d z + 2 * z * (d.stdDev / Real.sqrt d.sampleSize) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_confidence_interval_lower_bound_gt_50_784_population_mean_51_in_interval_l528_52875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_rate_l528_52831

noncomputable def h (t : ℝ) : ℝ := (2/3) * t^3 + 3 * t^2

noncomputable def h' (t : ℝ) : ℝ := 2 * t^2 + 6 * t

theorem liquid_rise_rate (t₀ : ℝ) :
  h' t₀ = 8 → h' (t₀ + 1) = 20 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check liquid_rise_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_rate_l528_52831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_balls_unchanged_l528_52803

/-- Given a box with blue and red balls, adding red balls does not change the number of blue balls. -/
theorem blue_balls_unchanged (initial_blue initial_red added_red : ℕ) :
  initial_blue = 3 → initial_red = 5 → added_red = 2 →
  initial_blue = initial_blue := by
    intros h1 h2 h3
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_balls_unchanged_l528_52803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_surface_area_l528_52864

/-- The total surface area of a right pyramid with an equilateral triangular base -/
noncomputable def pyramidSurfaceArea (baseSide : ℝ) (height : ℝ) : ℝ :=
  let baseArea := (Real.sqrt 3 / 4) * baseSide^2
  let slantHeight := Real.sqrt (height^2 + (baseSide * Real.sqrt 3 / 2)^2)
  let lateralArea := 3 * (1/2 * baseSide * slantHeight)
  lateralArea + baseArea

/-- Theorem stating the surface area of the specific pyramid -/
theorem specific_pyramid_surface_area :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |pyramidSurfaceArea 15 20 - 634.055| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_surface_area_l528_52864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_values_l528_52866

noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x + 5

theorem f_max_min_values :
  ∀ x : ℝ, 0 ≤ x → x ≤ 2 →
    (∀ y : ℝ, 0 ≤ y → y ≤ 2 → f y ≤ 5/2) ∧
    (∃ z : ℝ, 0 ≤ z ∧ z ≤ 2 ∧ f z = 5/2) ∧
    (∀ y : ℝ, 0 ≤ y → y ≤ 2 → 1/2 ≤ f y) ∧
    (∃ w : ℝ, 0 ≤ w ∧ w ≤ 2 ∧ f w = 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_values_l528_52866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l528_52879

/-- The speed of a train given its length and time to cross a fixed point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: A train with length 2500 meters that crosses an electric pole in 50 seconds
    has a speed of 50 meters per second -/
theorem train_speed_theorem :
  train_speed 2500 50 = 50 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l528_52879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l528_52872

theorem sin_beta_value (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α = 4/5) (h4 : Real.cos (α + β) = 3/5) : Real.sin β = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l528_52872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heart_nested_calc_l528_52821

-- Define the heart operation for positive real numbers
noncomputable def heart (a b : ℝ) : ℝ := a + 1 / b

-- Theorem statement
theorem heart_nested_calc :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  heart 3 (heart 3 3) = 33 / 10 := by
  intros a b ha hb
  -- Unfold the definition of heart
  unfold heart
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heart_nested_calc_l528_52821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalent_to_no_solution_l528_52848

/-- The negation of "There is at least one solution" is equivalent to "There is no solution" -/
theorem negation_equivalent_to_no_solution {α : Type*} (P : α → Prop) :
  ¬(∃ x, P x) ↔ ∀ x, ¬(P x) :=
by
  apply Iff.intro
  · intro h x px
    apply h
    exact ⟨x, px⟩
  · intro h ⟨x, px⟩
    exact h x px

#check negation_equivalent_to_no_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalent_to_no_solution_l528_52848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sphere_radius_l528_52862

-- Define the volume of a sphere as noncomputable
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- State the theorem
theorem larger_sphere_radius (r : ℝ) (h : r = 2) :
  ∃ R : ℝ, sphereVolume R = 6 * sphereVolume r ∧ R = (48 : ℝ)^(1/3) := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#check larger_sphere_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sphere_radius_l528_52862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_A_l528_52851

inductive AnswerChoice
  | A
  | B
  | C
  | D

def correctAnswer : AnswerChoice := AnswerChoice.A

theorem correct_answer_is_A : correctAnswer = AnswerChoice.A := by
  rfl

#check correct_answer_is_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_A_l528_52851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_income_calculation_l528_52853

/-- Given a person's savings rate and saved amount, calculate their monthly income. -/
def calculate_monthly_income (savings_rate : ℚ) (saved_amount : ℕ) : ℚ :=
  (saved_amount : ℚ) / savings_rate

/-- Theorem stating that if a person saves 10% of their income and saves 9000, their income is 90000. -/
theorem monthly_income_calculation (savings_rate : ℚ) (saved_amount : ℕ) :
  savings_rate = 1/10 → saved_amount = 9000 →
  calculate_monthly_income savings_rate saved_amount = 90000 := by
  intros h1 h2
  simp [calculate_monthly_income, h1, h2]
  norm_num

#eval (calculate_monthly_income (1/10) 9000).num.toNat

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_income_calculation_l528_52853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_l528_52852

/-- Represents the merchant's transaction -/
structure MerchantTransaction where
  initialGold : ℚ
  initialPigs : ℚ
  firstHerdRate : ℚ  -- pigs per gold coin
  secondHerdRate : ℚ  -- pigs per gold coin
  pigsSold : ℚ
  goldEarned : ℚ

/-- The specific transaction described in the problem -/
def problemTransaction : MerchantTransaction :=
  { initialGold := 200
  , initialPigs := 500
  , firstHerdRate := 3
  , secondHerdRate := 2
  , pigsSold := 480
  , goldEarned := 200 }

/-- Calculate the initial price per pig -/
def initialPricePerPig (t : MerchantTransaction) : ℚ :=
  t.initialGold / t.initialPigs

/-- Calculate the average price per pig after selling -/
def averagePricePerPig (t : MerchantTransaction) : ℚ :=
  t.goldEarned / t.pigsSold

/-- Theorem stating that the average price per pig after selling
    is higher than the initial price per pig -/
theorem price_increase (t : MerchantTransaction) :
  averagePricePerPig t > initialPricePerPig t := by
  sorry

#eval initialPricePerPig problemTransaction
#eval averagePricePerPig problemTransaction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_l528_52852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_even_f_right_shift_f_correct_answer_is_C_l528_52832

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin ((x / 2) + φ)

-- Statement 1: There exists a φ such that f is even
theorem exists_even_f : ∃ φ : ℝ, ∀ x : ℝ, f φ x = f φ (-x) := by sorry

-- Statement 2: For φ < 0, f is a right shift of sin(x/2) by |2φ|
theorem right_shift_f (φ : ℝ) (h : φ < 0) :
  ∀ x : ℝ, f φ x = Real.sin ((x + 2*φ) / 2) := by sorry

-- Additional theorem to state that the correct answer is C
theorem correct_answer_is_C : True := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_even_f_right_shift_f_correct_answer_is_C_l528_52832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l528_52817

/-- The rational function under consideration -/
noncomputable def f (x : ℝ) : ℝ := (3*x^2 - 4*x + 2) / (x + 2)

/-- The slope of the slant asymptote -/
def m : ℝ := 3

/-- The y-intercept of the slant asymptote -/
def b : ℝ := -10

/-- Theorem stating that the sum of the slope and y-intercept of the slant asymptote is -7 -/
theorem slant_asymptote_sum : m + b = -7 := by
  -- Unfold the definitions of m and b
  unfold m b
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l528_52817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l528_52805

theorem no_valid_n : ¬∃ n : ℕ, 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l528_52805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_l528_52873

-- Define the function f(x) = x / (x + 2)
noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

-- Define the interval [2, 4]
def interval : Set ℝ := { x | 2 ≤ x ∧ x ≤ 4 }

-- Theorem statement
theorem range_of_f_on_interval :
  { y | ∃ x ∈ interval, f x = y } = { y | 1/2 ≤ y ∧ y ≤ 2/3 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_l528_52873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_without_paddles_l528_52863

/-- Represents the velocity of the river current -/
noncomputable def river_velocity : ℝ := sorry

/-- Represents the average velocity of the canoe in still water -/
noncomputable def canoe_velocity : ℝ := sorry

/-- Represents the distance from village A to village B along the river -/
noncomputable def distance : ℝ := sorry

/-- Time to travel from A to B -/
noncomputable def time_A_to_B : ℝ := distance / (canoe_velocity + river_velocity)

/-- Time to travel from B to A -/
noncomputable def time_B_to_A : ℝ := distance / (canoe_velocity - river_velocity)

/-- Time to travel from B to A without paddles -/
noncomputable def time_B_to_A_no_paddles : ℝ := distance / river_velocity

/-- The condition that travel time from A to B is 3 times longer than from B to A -/
axiom travel_time_condition : time_A_to_B = 3 * time_B_to_A

/-- Theorem: The time to travel from B to A without paddles is 3 times longer than the usual time -/
theorem travel_time_without_paddles :
  time_B_to_A_no_paddles = 3 * time_B_to_A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_without_paddles_l528_52863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l528_52841

-- Problem 1
theorem problem_1 : 
  (0.027 : ℝ)^(-(1/3 : ℝ)) - (-1/7 : ℝ)^(-(2 : ℝ)) + 256^(3/4 : ℝ) - 3^(-(1 : ℝ)) + (Real.sqrt 2 - 1)^(0 : ℝ) = 19 := by sorry

-- Problem 2
theorem problem_2 : 
  (Real.log 8 + Real.log 125 - Real.log 2 - Real.log 5) / (Real.log (Real.sqrt 10) * Real.log 0.1) = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l528_52841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_squared_minus_one_div_thirty_l528_52839

/-- The largest prime number with 2023 digits -/
noncomputable def p : ℕ := sorry

/-- p is prime -/
axiom p_prime : Nat.Prime p

/-- p has 2023 digits -/
axiom p_digits : (Nat.digits 10 p).length = 2023

/-- p is the largest prime with 2023 digits -/
axiom p_largest : ∀ q : ℕ, Nat.Prime q → (Nat.digits 10 q).length = 2023 → q ≤ p

theorem p_squared_minus_one_div_thirty : 30 ∣ (p^2 - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_squared_minus_one_div_thirty_l528_52839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_cos_2alpha_plus_pi_4_l528_52838

open Real

-- Define the function f
noncomputable def f (α : ℝ) : ℝ := (tan (π - α) * cos (2*π - α) * sin (π/2 + α)) / cos (-α - π)

-- Theorem 1: Simplification of f(α)
theorem f_simplification (α : ℝ) : f α = sin α := by sorry

-- Theorem 2: Value of cos(2α + π/4) given conditions
theorem cos_2alpha_plus_pi_4 (α : ℝ) (h1 : f α = 4/5) (h2 : π/2 < α ∧ α < π) :
  cos (2*α + π/4) = 17 * sqrt 2 / 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_cos_2alpha_plus_pi_4_l528_52838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l528_52829

-- Define the points and vectors
def A : ℝ × ℝ := (1, 0)
def R (a : ℝ) : ℝ × ℝ := (a, 2 * a - 4)
def P : ℝ × ℝ → Prop := λ _ ↦ True  -- Define P as a predicate that always holds

-- Define the line l
def on_line_l (p : ℝ × ℝ) : Prop :=
  p.2 = 2 * p.1 - 4

-- Define vector equality
def vector_eq (v w : ℝ × ℝ) : Prop :=
  v.1 = w.1 ∧ v.2 = w.2

-- Main theorem
theorem trajectory_of_P :
  ∀ a x y, on_line_l (R a) →
    vector_eq (A.1 - (R a).1, A.2 - (R a).2) (x - A.1, y - A.2) →
    P (x, y) →
    y = 2 * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l528_52829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shared_vertex_angle_l528_52801

-- Define the angle of a regular polygon
noncomputable def regular_polygon_angle (n : ℕ) : ℝ := (n - 2 : ℝ) * 180 / n

-- Define the setup
structure CircleInscription where
  pentagon_angle : ℝ
  triangle_angle : ℝ
  shared_vertex : ℝ

-- State the theorem
theorem shared_vertex_angle (c : CircleInscription) 
  (h1 : c.pentagon_angle = regular_polygon_angle 5)
  (h2 : c.triangle_angle = regular_polygon_angle 3)
  (h3 : c.shared_vertex = 360 - (2 * c.pentagon_angle + 2 * c.triangle_angle) / 2) :
  c.shared_vertex = 192 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shared_vertex_angle_l528_52801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_midpoint_locus_four_points_distance_l528_52802

-- Define the circle C and line l
def circle_eq (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5
def line_eq (m x y : ℝ) : Prop := m * x - y + 1 + 2 * m = 0

-- Theorem 1: Line l always intersects circle C at two distinct points
theorem line_intersects_circle (m : ℝ) : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ line_eq m x₁ y₁ ∧ line_eq m x₂ y₂ :=
sorry

-- Theorem 2: The locus of the midpoint M
theorem midpoint_locus (x y : ℝ) : 
  (∃ m : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ line_eq m x₁ y₁ ∧ line_eq m x₂ y₂ ∧ 
    x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2) ↔ 
  (x + 2)^2 + (y - 1/2)^2 = 1/4 :=
sorry

-- Theorem 3: Existence of m for four points at a specific distance
theorem four_points_distance (m : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ, 
    (∀ i j, (i, j) ∈ [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)] → circle_eq i j) ∧
    (∀ i j, (i, j) ∈ [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)] → 
      (abs (m * i - j + 1 + 2 * m) / Real.sqrt (1 + m^2) = 4 * Real.sqrt 5 / 5))) ↔ 
  (m > 2 ∨ m < -2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_midpoint_locus_four_points_distance_l528_52802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_winning_config_l528_52814

-- Define the nim-value function for a single wall
def nim_value (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 0
  | 5 => 3
  | 6 => 1
  | 7 => 2
  | 8 => 1
  | _ => 0  -- Default case, though not needed for our specific problem

-- Define the nim-sum function for a configuration
def nim_sum (config : List ℕ) : ℕ :=
  config.map nim_value |> List.foldl Nat.xor 0

-- Theorem statement
theorem beth_winning_config :
  nim_sum [8, 5, 2] = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_winning_config_l528_52814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_formula_l528_52842

/-- The function f(x) = x^2 - x -/
noncomputable def f (x : ℝ) : ℝ := x^2 - x

/-- The average rate of change of f from 2 to 2 + Δx -/
noncomputable def averageRateOfChange (Δx : ℝ) : ℝ := (f (2 + Δx) - f 2) / Δx

theorem average_rate_of_change_formula (Δx : ℝ) (h : Δx ≠ 0) :
  averageRateOfChange Δx = Δx + 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_formula_l528_52842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_for_C_to_finish_is_two_and_half_l528_52818

/-- The time it takes for C to finish the remaining work after A and B have worked -/
noncomputable def time_for_C_to_finish (a_days b_days c_days : ℝ) (a_worked b_worked : ℝ) : ℝ :=
  let a_rate := 1 / a_days
  let b_rate := 1 / b_days
  let c_rate := 1 / c_days
  let work_done := a_worked * a_rate + b_worked * b_rate
  let remaining_work := 1 - work_done
  remaining_work / c_rate

theorem time_for_C_to_finish_is_two_and_half :
  time_for_C_to_finish 15 20 30 10 5 = 2.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_for_C_to_finish 15 20 30 10 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_for_C_to_finish_is_two_and_half_l528_52818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_max_area_l528_52835

noncomputable section

-- Define the points A, B, and F
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def F : ℝ × ℝ := (0, 1)

-- Define the condition for point M
def is_valid_M (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  (y / (x + 1)) * (y / (x - 1)) = -2 ∧ x ≠ 1 ∧ x ≠ -1

-- Define the curve C
def on_curve_C (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2/2 = 1 ∧ x ≠ 1 ∧ x ≠ -1

-- Define a line through F
def line_through_F (k : ℝ) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  y = k * x + 1

-- Define the area of triangle OPQ
noncomputable def area_OPQ (P Q : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := P
  let (x₂, y₂) := Q
  (1/2) * Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem curve_C_and_max_area :
  (∀ M : ℝ × ℝ, is_valid_M M → on_curve_C M) ∧
  (∃ max_area : ℝ, max_area = Real.sqrt 2 / 2 ∧
    ∀ k : ℝ, ∀ P Q : ℝ × ℝ,
      on_curve_C P ∧ on_curve_C Q ∧
      line_through_F k P ∧ line_through_F k Q →
      area_OPQ P Q ≤ max_area) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_max_area_l528_52835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l528_52858

-- Define the hyperbola C: px^2 - qy^2 = r
def hyperbola (p q r : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | p * x^2 - q * y^2 = r}

-- Define the geometric sequence condition
def geometric_sequence (p q r : ℝ) : Prop :=
  q / p = 2 ∧ r / q = 2

-- Theorem statement
theorem hyperbola_properties
  (p q r : ℝ) (h_pos : p > 0) (h_geom : geometric_sequence p q r) :
  ∃ (a b c : ℝ),
    -- The length of the real axis is 4
    2 * a = 4 ∧
    -- The distance from the focus to the asymptote is √2
    (c^2 - a^2) / c = Real.sqrt 2 ∧
    -- Additional properties to establish the hyperbola
    a^2 = 4 ∧ b^2 = 2 ∧ c^2 = a^2 + b^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l528_52858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_lateral_faces_equal_areas_l528_52890

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Triangle where
  a : Point
  b : Point
  c : Point

structure Parallelogram where
  vertices : Fin 4 → Point

structure Sphere where
  center : Point
  radius : ℝ

structure Pyramid where
  base : Parallelogram
  vertex : Point
  inscribedSphere : Sphere

-- Define the lateral face
def lateralFace (p : Pyramid) (v1 v2 : Point) : Triangle :=
  { a := p.vertex, b := v1, c := v2 }

-- Define the lateral face area (noncomputable as it involves real numbers)
noncomputable def lateralFaceArea (p : Pyramid) (v1 v2 : Point) : ℝ :=
  sorry -- Placeholder for actual area calculation

-- Main theorem
theorem opposite_lateral_faces_equal_areas (p : Pyramid) :
  let A := p.base.vertices 0
  let B := p.base.vertices 1
  let C := p.base.vertices 2
  let D := p.base.vertices 3
  lateralFaceArea p A B + lateralFaceArea p C D =
  lateralFaceArea p A D + lateralFaceArea p B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_lateral_faces_equal_areas_l528_52890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l528_52827

noncomputable def f (x a : ℝ) : ℝ := Real.exp x * (Real.sin x + Real.cos x) + a

noncomputable def g (x a : ℝ) : ℝ := (a^2 - a + 10) * Real.exp x

noncomputable def φ (x a b : ℝ) : ℝ := 
  (b * (1 + Real.exp 2) * g x a) / ((a^2 - a + 10) * Real.exp (2*x)) - 1/x + 1 + Real.log x

theorem problem_solution (a b : ℝ) (hb : b > 1) :
  (∃ x : ℝ, HasDerivAt (f · a) ((f 0 a - 2) / -1) 0 ∧ HasDerivAt (f · a) (2 * Real.exp x * Real.cos x) x) →
  a = -1 ∧
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 Real.pi ∧ x₂ ∈ Set.Icc 0 Real.pi ∧ 
    g x₂ a < f x₁ a + 13 - Real.exp (Real.pi / 2)) →
  a ∈ Set.Ioo (-1) 3 ∧
  ∀ x : ℝ, x > 0 → φ x a b ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l528_52827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_correct_l528_52823

/-- A square on the complex plane with three known vertices -/
structure ComplexSquare where
  a : ℂ
  b : ℂ
  c : ℂ
  is_square : a = 1 + 2*Complex.I ∧ b = -2 + Complex.I ∧ c = -1 - 2*Complex.I

/-- The fourth vertex of the square -/
def fourth_vertex (s : ComplexSquare) : ℂ := 2 - Complex.I

/-- Theorem stating that the fourth vertex is correct -/
theorem fourth_vertex_correct (s : ComplexSquare) : 
  let d := fourth_vertex s
  (d - s.a) = (s.c - s.b) ∧ 
  (d - s.c) = (s.a - s.b) ∧
  (d - s.a).re * (s.b - s.a).re + (d - s.a).im * (s.b - s.a).im = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_correct_l528_52823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_calculation_l528_52892

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: For a trapezium with parallel sides of 20 cm and 18 cm, and an area of 95 cm², 
    the distance between the parallel sides is 5 cm. -/
theorem trapezium_height_calculation :
  let a : ℝ := 20
  let b : ℝ := 18
  let area : ℝ := 95
  ∃ h : ℝ, trapezium_area a b h = area ∧ h = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_calculation_l528_52892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_phenomena_are_translational_l528_52868

/-- Enum representing different types of motion --/
inductive MotionType
  | Translational
  | Rotational
  | Other

/-- Enum representing the phenomena in the question --/
inductive Phenomenon
  | Thermometer
  | Pump
  | Pendulum
  | ConveyorBelt

/-- Function to classify the motion type of each phenomenon --/
def classifyMotion (p : Phenomenon) : MotionType :=
  match p with
  | Phenomenon.Thermometer => MotionType.Translational
  | Phenomenon.Pump => MotionType.Translational
  | Phenomenon.Pendulum => MotionType.Rotational
  | Phenomenon.ConveyorBelt => MotionType.Translational

/-- Theorem stating that the correct phenomena involve translational motion --/
theorem correct_phenomena_are_translational :
  (classifyMotion Phenomenon.Thermometer = MotionType.Translational) ∧
  (classifyMotion Phenomenon.Pump = MotionType.Translational) ∧
  (classifyMotion Phenomenon.ConveyorBelt = MotionType.Translational) :=
by
  apply And.intro
  · rfl
  · apply And.intro
    · rfl
    · rfl

#check correct_phenomena_are_translational

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_phenomena_are_translational_l528_52868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l528_52886

/-- The function f(x) defined as x/(x+1) + (x+1)/(x+2) + (x+2)/(x+3) -/
noncomputable def f (x : ℝ) : ℝ := x / (x + 1) + (x + 1) / (x + 2) + (x + 2) / (x + 3)

/-- Theorem stating that f(x) has a symmetry center at (-2, 3) -/
theorem symmetry_center_of_f :
  ∀ x : ℝ, f x = 6 - f (-4 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l528_52886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_intersecting_labeling_l528_52893

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of n points in the plane -/
def PointSet (n : ℕ) := Fin n → Point

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Check if two line segments intersect -/
def intersect (p1 q1 p2 q2 : Point) : Prop := sorry

theorem non_intersecting_labeling 
  (n : ℕ) 
  (A B : PointSet n) 
  (h_disjoint : ∀ i j : Fin n, A i ≠ B j) 
  (h_non_collinear : ∀ p q r : Point, (∃ i, p = A i ∨ p = B i) → 
                     (∃ j, q = A j ∨ q = B j) → 
                     (∃ k, r = A k ∨ r = B k) → 
                     p ≠ q → q ≠ r → p ≠ r → ¬ collinear p q r) :
  ∃ σ : Fin n ≃ Fin n, ∀ i j : Fin n, i ≠ j → ¬ intersect (A i) (B (σ i)) (A j) (B (σ j)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_intersecting_labeling_l528_52893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_l528_52885

/-- The number of values in the set --/
def n : ℕ := 12

/-- The incorrect average of the values --/
def incorrect_avg : ℚ := 402/10

/-- The difference between the first incorrect and correct number --/
def diff1 : ℤ := 19

/-- The incorrect second number --/
def incorrect2 : ℕ := 13

/-- The correct second number --/
def correct2 : ℕ := 31

/-- The incorrect third number --/
def incorrect3 : ℕ := 45

/-- The correct third number --/
def correct3 : ℕ := 25

/-- The difference between the fourth incorrect and correct number --/
def diff4 : ℤ := -11

/-- Theorem stating that the correct average is 39.4 (rounded to one decimal place) --/
theorem correct_average : 
  let incorrect_sum := n * incorrect_avg
  let correction := -diff1 + (correct2 - incorrect2) + (correct3 - incorrect3) + -diff4
  let correct_sum := incorrect_sum + correction
  let correct_avg := correct_sum / n
  (⌊correct_avg * 10 + 1/2⌋ : ℚ) / 10 = 394/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_l528_52885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_younger_students_count_l528_52898

theorem younger_students_count (total : ℕ) (younger_percent : ℚ) (older_percent : ℚ) :
  total = 35 →
  younger_percent = 2/5 →
  older_percent = 1/4 →
  ∃ (younger older : ℕ),
    younger + older = total ∧
    younger_percent * younger = older_percent * older ∧
    younger = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_younger_students_count_l528_52898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l528_52826

/-- Represents a hyperbola with equation x²/4 - y²/3 = 1 -/
structure Hyperbola where
  eq : ℝ → ℝ → Prop
  eq_def : ∀ x y, eq x y ↔ x^2/4 - y^2/3 = 1

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For the given hyperbola, if P is a point on the hyperbola with PF₁ = 3, then PF₂ = 7 -/
theorem hyperbola_focal_distance 
  (h : Hyperbola) 
  (f1 f2 : Point) -- Left and right foci
  (p : Point) -- Point on the hyperbola
  (on_hyperbola : h.eq p.x p.y)
  (dist_pf1 : distance p f1 = 3) :
  distance p f2 = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l528_52826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_volume_l528_52843

/-- The volume of a regular quadrilateral pyramid with an inscribed sphere -/
noncomputable def pyramidVolume (r : ℝ) (α : ℝ) : ℝ :=
  (2 * r^3 * (Real.sqrt (2 * Real.tan α^2 + 1) + 1)^3) / (3 * Real.tan α^2)

/-- Theorem: Volume of a regular quadrilateral pyramid with inscribed sphere -/
theorem regular_quadrilateral_pyramid_volume 
  (r : ℝ) (α : ℝ) (h_r : r > 0) (h_α : 0 < α ∧ α < Real.pi / 2) :
  pyramidVolume r α = (2 * r^3 * (Real.sqrt (2 * Real.tan α^2 + 1) + 1)^3) / (3 * Real.tan α^2) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_volume_l528_52843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l528_52800

-- Define the polynomials
noncomputable def dividend : Polynomial ℚ := Polynomial.X^3 - 3*Polynomial.X + 1
noncomputable def divisor : Polynomial ℚ := Polynomial.X^2 - Polynomial.X - 2
noncomputable def remainder : Polynomial ℚ := -Polynomial.X^2 - Polynomial.X + 1

-- State the theorem
theorem polynomial_division_theorem :
  ∃ q : Polynomial ℚ, dividend = divisor * q + remainder := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l528_52800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l528_52894

noncomputable def F₁ : ℝ × ℝ := (-2, 2 - Real.sqrt 3 / 2)
noncomputable def F₂ : ℝ × ℝ := (-2, 2 + Real.sqrt 3 / 2)

def is_on_hyperbola (P : ℝ × ℝ) : Prop :=
  Real.sqrt 2 = Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) -
                Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)

def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

theorem hyperbola_sum (h k a b : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (heq : ∀ x y : ℝ, is_on_hyperbola (x, y) ↔ hyperbola_equation x y h k a b) :
  h + k + a + b = (Real.sqrt 2 + 1) / 2 := by
  sorry

#check hyperbola_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l528_52894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_free_time_proof_l528_52888

/-- Calculates the number of hours Madeline has left over in a week. -/
def madeline_free_time (total_hours_per_week : ℕ) 
                       (class_hours_per_week : ℕ) 
                       (homework_hours_per_day : ℕ) 
                       (sleep_hours_per_day : ℕ) 
                       (work_hours_per_week : ℕ) : ℕ :=
  let total_homework_hours := homework_hours_per_day * 7
  let total_sleep_hours := sleep_hours_per_day * 7
  let total_occupied_hours := class_hours_per_week + total_homework_hours + total_sleep_hours + work_hours_per_week
  total_hours_per_week - total_occupied_hours

/-- Proves that Madeline has 46 hours left over in a week. -/
theorem madeline_free_time_proof :
  madeline_free_time 168 18 4 8 20 = 46 := by
  rfl

#eval madeline_free_time 168 18 4 8 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_free_time_proof_l528_52888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_equation_l528_52813

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define point P
def P : ℝ × ℝ := (4, 2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the property of tangent lines
def forms_tangent_lines (P : ℝ × ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ, circle A.1 A.2 ∧ circle B.1 B.2 ∧ 
         (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0

-- Theorem statement
theorem circumcircle_equation :
  forms_tangent_lines P circle_eq →
  ∃ A B : ℝ × ℝ, circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
         ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 5 ↔ 
                (x - O.1) * (A.2 - O.2) = (y - O.2) * (A.1 - O.1) ∧
                (x - O.1) * (B.2 - O.2) = (y - O.2) * (B.1 - O.1) ∧
                (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_equation_l528_52813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_chips_price_is_correct_l528_52822

noncomputable def chips_price : ℚ := 2
def chips_quantity : ℕ := 15
noncomputable def total_budget : ℚ := 45
def corn_chips_quantity : ℕ := 10

noncomputable def corn_chips_price : ℚ :=
  (total_budget - chips_price * chips_quantity) / corn_chips_quantity

theorem corn_chips_price_is_correct : corn_chips_price = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_chips_price_is_correct_l528_52822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_four_lt_b_seven_l528_52861

def b (α : ℕ → ℕ+) : ℕ → ℚ
  | 0 => 1 + 1 / α 1  -- Added case for 0
  | 1 => 1 + 1 / α 1
  | n + 1 => 1 + 1 / (α (n + 1) + 1 / b α n)

theorem b_four_lt_b_seven (α : ℕ → ℕ+) : b α 4 < b α 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_four_lt_b_seven_l528_52861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_line_for_transformed_points_l528_52878

/-- Two points P(x, y) and P'(x', y') that satisfy the given transformation -/
structure TransformedPoints (x y x' y' : ℝ) : Prop where
  eq1 : x' = 3 * x + 2 * y + 1
  eq2 : y' = x + 4 * y - 3

/-- A line in the 2D plane represented by the equation Ax + By + C = 0 -/
structure Line (A B C : ℝ) : Prop where
  non_zero : A ≠ 0 ∨ B ≠ 0

/-- A point (x, y) lies on a line Ax + By + C = 0 -/
def PointOnLine (x y : ℝ) (A B C : ℝ) : Prop :=
  A * x + B * y + C = 0

/-- Theorem stating that there exists a line on which both P and P' lie -/
theorem exists_line_for_transformed_points :
  ∀ (x y x' y' : ℝ), TransformedPoints x y x' y' →
  ∃ (A B C : ℝ), Line A B C ∧ PointOnLine x y A B C ∧ PointOnLine x' y' A B C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_line_for_transformed_points_l528_52878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_25_l528_52809

-- Define the cost function
noncomputable def C (x : ℝ) : ℝ :=
  if x < 20 then x^2 + 20*x else 54*x + 2500/x - 500

-- Define the profit function
noncomputable def f (x : ℝ) : ℝ :=
  if x < 20 then -x^2 + 30*x else 500 - 4*(x + 625/x)

-- Theorem statement
theorem max_profit_at_25 :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x ∧ x = 25 ∧ f x = 300 := by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_25_l528_52809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_formula_l528_52877

/-- Represents a geometric sequence with common ratio q > 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_geometric : ∀ n, a (n + 1) = q * a n
  h_q_gt_one : q > 1

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.a 1 * (g.q^n - 1) / (g.q - 1)

theorem geometric_sum_formula (g : GeometricSequence) 
  (h1 : g.a 3 + g.a 5 = 20)
  (h2 : g.a 4 = 8) :
  ∀ n, geometricSum g n = 2^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_formula_l528_52877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_distance_sum_l528_52860

-- Define the circle C
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*y

-- Define the line l
def line_equation (t x y : ℝ) : Prop := x = -Real.sqrt 3/2 * t ∧ y = 2 + t/2

-- Define the intersection points A and B
def intersection_points (t : ℝ) : Prop := t^2 = 4

-- Define point P
def point_p (t : ℝ) : Prop := 2 + t/2 = 0

theorem circle_center_and_distance_sum :
  -- Part I: The center of the circle is at (0, 2) in Cartesian coordinates
  (∃ (x y : ℝ), x = 0 ∧ y = 2 ∧ ∀ (x' y' : ℝ), circle_equation x' y' → (x' - x)^2 + (y' - y)^2 ≤ (x' - 0)^2 + (y' - 2)^2) ∧
  -- Which corresponds to (2, π/2) in polar coordinates
  (∃ (ρ θ : ℝ), ρ = 2 ∧ θ = Real.pi/2 ∧ 0 = ρ * Real.cos θ ∧ 2 = ρ * Real.sin θ) ∧
  -- Part II: The sum of distances |PA| + |PB| = 8
  (∃ (t_a t_b t_p : ℝ), 
    intersection_points t_a ∧ 
    intersection_points t_b ∧ 
    point_p t_p ∧
    |t_a - t_p| + |t_b - t_p| = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_distance_sum_l528_52860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l528_52807

/-- Represents the number of days A takes to complete the work alone -/
noncomputable def days_A : ℝ := 15

/-- Represents the number of days B takes to complete the work alone -/
noncomputable def days_B : ℝ := 20

/-- Represents the fraction of work left after A and B work together -/
noncomputable def work_left : ℝ := 0.3

/-- Calculates the number of days A and B worked together -/
noncomputable def days_worked_together : ℝ := 
  (1 - work_left) / ((1 / days_A) + (1 / days_B))

theorem work_completion_time : days_worked_together = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l528_52807
