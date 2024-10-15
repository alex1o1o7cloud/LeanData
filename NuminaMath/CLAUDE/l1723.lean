import Mathlib

namespace NUMINAMATH_CALUDE_custom_operation_calculation_l1723_172310

-- Define the custom operation *
def star (a b : ℕ) : ℕ := a + 2 * b

-- Theorem statement
theorem custom_operation_calculation :
  star (star (star 2 3) 4) 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_calculation_l1723_172310


namespace NUMINAMATH_CALUDE_octagon_area_in_circle_l1723_172304

theorem octagon_area_in_circle (R : ℝ) : 
  R > 0 → 
  (4 * (1/2 * R^2 * Real.sin (π/4)) + 4 * (1/2 * R^2 * Real.sin (π/2))) = R^2 * (Real.sqrt 2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_in_circle_l1723_172304


namespace NUMINAMATH_CALUDE_equation_solutions_l1723_172397

theorem equation_solutions :
  (∃ x₁ x₂, (3 * x₁ + 2)^2 = 16 ∧ (3 * x₂ + 2)^2 = 16 ∧ x₁ = 2/3 ∧ x₂ = -2) ∧
  (∃ x, (1/2) * (2 * x - 1)^3 = -4 ∧ x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1723_172397


namespace NUMINAMATH_CALUDE_inequality_proof_l1723_172302

theorem inequality_proof (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a^2 + b^2 = 4) :
  (a * b) / (a + b + 2) ≤ Real.sqrt 2 - 1 ∧
  ((a * b) / (a + b + 2) = Real.sqrt 2 - 1 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2) :=
by sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l1723_172302


namespace NUMINAMATH_CALUDE_log_exponent_sum_l1723_172376

theorem log_exponent_sum (a : ℝ) (h : a = Real.log 5 / Real.log 4) :
  2^a + 2^(-a) = 6 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_log_exponent_sum_l1723_172376


namespace NUMINAMATH_CALUDE_systematic_sampling_largest_number_l1723_172360

/-- Systematic sampling theorem for class selection -/
theorem systematic_sampling_largest_number
  (total_classes : ℕ)
  (selected_classes : ℕ)
  (smallest_number : ℕ)
  (h1 : total_classes = 24)
  (h2 : selected_classes = 4)
  (h3 : smallest_number = 3)
  (h4 : smallest_number > 0)
  (h5 : smallest_number ≤ total_classes)
  (h6 : selected_classes > 0)
  (h7 : selected_classes ≤ total_classes) :
  ∃ (largest_number : ℕ),
    largest_number = 21 ∧
    largest_number ≤ total_classes ∧
    (largest_number - smallest_number) = (selected_classes - 1) * (total_classes / selected_classes) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_largest_number_l1723_172360


namespace NUMINAMATH_CALUDE_min_value_inequality_l1723_172389

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  x / y + 1 / x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1723_172389


namespace NUMINAMATH_CALUDE_max_abs_sum_on_ellipse_l1723_172381

theorem max_abs_sum_on_ellipse :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ |x| + |y|
  let S : Set (ℝ × ℝ) := {(x, y) | 4 * x^2 + y^2 = 4}
  ∃ (x y : ℝ), (x, y) ∈ S ∧ f (x, y) = (3 * Real.sqrt 2) / Real.sqrt 5 ∧
  ∀ (a b : ℝ), (a, b) ∈ S → f (a, b) ≤ (3 * Real.sqrt 2) / Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_ellipse_l1723_172381


namespace NUMINAMATH_CALUDE_y_coordinate_abs_value_l1723_172326

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance from a point to the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- The distance from a point to the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

theorem y_coordinate_abs_value (p : Point) 
  (h1 : distToXAxis p = (1/2) * distToYAxis p) 
  (h2 : distToYAxis p = 12) : 
  |p.y| = 6 := by sorry

end NUMINAMATH_CALUDE_y_coordinate_abs_value_l1723_172326


namespace NUMINAMATH_CALUDE_problem_solution_l1723_172399

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t, Real.sqrt 2 / 2 + Real.sqrt 3 * t)

-- Define the curve C in polar form
def curve_C (θ : ℝ) : ℝ := 2 * Real.cos (θ - Real.pi / 4)

-- Define point P
def point_P : ℝ × ℝ := (0, Real.sqrt 2 / 2)

-- Theorem statement
theorem problem_solution :
  -- 1. The slope angle of line l is π/3
  (let slope := (Real.sqrt 3);
   Real.arctan slope = Real.pi / 3) ∧
  -- 2. The rectangular equation of curve C
  (∀ x y : ℝ, (x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 1 ↔
    ∃ θ : ℝ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) ∧
  -- 3. If line l intersects curve C at points A and B, then |PA| + |PB| = √10/2
  (∃ A B : ℝ × ℝ,
    (∃ t : ℝ, line_l t = A) ∧
    (∃ t : ℝ, line_l t = B) ∧
    (∃ θ : ℝ, A.1 = curve_C θ * Real.cos θ ∧ A.2 = curve_C θ * Real.sin θ) ∧
    (∃ θ : ℝ, B.1 = curve_C θ * Real.cos θ ∧ B.2 = curve_C θ * Real.sin θ) ∧
    Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
    Real.sqrt 10 / 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1723_172399


namespace NUMINAMATH_CALUDE_clients_using_all_three_l1723_172307

def total_clients : ℕ := 180
def tv_clients : ℕ := 115
def radio_clients : ℕ := 110
def magazine_clients : ℕ := 130
def tv_and_magazine : ℕ := 85
def tv_and_radio : ℕ := 75
def radio_and_magazine : ℕ := 95

theorem clients_using_all_three :
  tv_clients + radio_clients + magazine_clients -
  tv_and_magazine - tv_and_radio - radio_and_magazine +
  (total_clients - (tv_clients + radio_clients + magazine_clients -
  tv_and_magazine - tv_and_radio - radio_and_magazine)) = 80 :=
by sorry

end NUMINAMATH_CALUDE_clients_using_all_three_l1723_172307


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1723_172383

/-- The quadratic equation (m-3)x^2 - 2x + 1 = 0 has real roots if and only if m ≤ 4 and m ≠ 3. -/
theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1723_172383


namespace NUMINAMATH_CALUDE_factorial_simplification_l1723_172380

theorem factorial_simplification (N : ℕ) :
  (Nat.factorial (N + 2)) / (Nat.factorial N * (N + 3)) = ((N + 2) * (N + 1)) / (N + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l1723_172380


namespace NUMINAMATH_CALUDE_dice_surface_area_l1723_172342

/-- The surface area of a cube with edge length 11 cm is 726 cm^2. -/
theorem dice_surface_area (edge_length : ℝ) (h : edge_length = 11) :
  6 * edge_length^2 = 726 := by
  sorry

end NUMINAMATH_CALUDE_dice_surface_area_l1723_172342


namespace NUMINAMATH_CALUDE_qrs_company_profit_change_l1723_172393

theorem qrs_company_profit_change (march_profit : ℝ) : 
  let april_profit := 1.10 * march_profit
  let may_profit := april_profit * (1 - x / 100)
  let june_profit := may_profit * 1.50
  june_profit = 1.3200000000000003 * march_profit →
  x = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_qrs_company_profit_change_l1723_172393


namespace NUMINAMATH_CALUDE_weaving_problem_l1723_172333

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

theorem weaving_problem (a₁ d : ℚ) (n : ℕ) 
  (h₁ : a₁ = 5)
  (h₂ : n = 30)
  (h₃ : sum_arithmetic_sequence a₁ d n = 390) :
  d = 16/29 := by
  sorry

end NUMINAMATH_CALUDE_weaving_problem_l1723_172333


namespace NUMINAMATH_CALUDE_equation_solution_exists_l1723_172386

def f (x : ℝ) := x^3 - x - 1

theorem equation_solution_exists :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 1.5 ∧ |f x| ≤ 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l1723_172386


namespace NUMINAMATH_CALUDE_g_function_equality_l1723_172359

/-- Given that 4x^4 + 5x^2 - 2x + 7 + g(x) = 6x^3 - 4x^2 + 8x - 1,
    prove that g(x) = -4x^4 + 6x^3 - 9x^2 + 10x - 8 -/
theorem g_function_equality (x : ℝ) (g : ℝ → ℝ)
    (h : ∀ x, 4 * x^4 + 5 * x^2 - 2 * x + 7 + g x = 6 * x^3 - 4 * x^2 + 8 * x - 1) :
  g x = -4 * x^4 + 6 * x^3 - 9 * x^2 + 10 * x - 8 := by
  sorry

end NUMINAMATH_CALUDE_g_function_equality_l1723_172359


namespace NUMINAMATH_CALUDE_opposite_pairs_l1723_172335

theorem opposite_pairs (a b : ℝ) : 
  (∀ x, (a + b) + (-a - b) = x ↔ x = 0) ∧ 
  (∀ x, (-a + b) + (a - b) = x ↔ x = 0) ∧ 
  ¬(∀ x, (a - b) + (-a - b) = x ↔ x = 0) ∧ 
  ¬(∀ x, (a + 1) + (1 - a) = x ↔ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_opposite_pairs_l1723_172335


namespace NUMINAMATH_CALUDE_transportation_time_savings_l1723_172367

def walking_time : ℕ := 98
def bicycle_saved_time : ℕ := 64
def car_saved_time : ℕ := 85
def bus_saved_time : ℕ := 55

theorem transportation_time_savings :
  (walking_time - (walking_time - bicycle_saved_time) = bicycle_saved_time) ∧
  (walking_time - (walking_time - car_saved_time) = car_saved_time) ∧
  (walking_time - (walking_time - bus_saved_time) = bus_saved_time) := by
  sorry

end NUMINAMATH_CALUDE_transportation_time_savings_l1723_172367


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l1723_172341

theorem stratified_sampling_sample_size 
  (total_employees : ℕ) 
  (young_employees : ℕ) 
  (young_in_sample : ℕ) 
  (h1 : total_employees = 750) 
  (h2 : young_employees = 350) 
  (h3 : young_in_sample = 7) : 
  (young_in_sample * total_employees) / young_employees = 15 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l1723_172341


namespace NUMINAMATH_CALUDE_sallys_remaining_cards_l1723_172313

/-- Given Sally's initial number of baseball cards, the number of torn cards, 
    and the number of cards Sara bought, prove the number of cards Sally has now. -/
theorem sallys_remaining_cards (initial_cards torn_cards cards_bought : ℕ) :
  initial_cards = 39 →
  torn_cards = 9 →
  cards_bought = 24 →
  initial_cards - torn_cards - cards_bought = 6 := by
  sorry

end NUMINAMATH_CALUDE_sallys_remaining_cards_l1723_172313


namespace NUMINAMATH_CALUDE_largest_circle_radius_is_b_l1723_172338

/-- An ellipsoid with semi-axes a > b > c -/
structure Ellipsoid where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > b
  h2 : b > c

/-- The radius of the largest circle on an ellipsoid -/
def largest_circle_radius (e : Ellipsoid) : ℝ := e.b

/-- Theorem: The radius of the largest circle on an ellipsoid with semi-axes a > b > c is b -/
theorem largest_circle_radius_is_b (e : Ellipsoid) :
  largest_circle_radius e = e.b :=
by sorry

end NUMINAMATH_CALUDE_largest_circle_radius_is_b_l1723_172338


namespace NUMINAMATH_CALUDE_decaf_percentage_l1723_172369

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
theorem decaf_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (additional_stock : ℝ)
  (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 30)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 60) :
  let total_stock := initial_stock + additional_stock
  let total_decaf := initial_stock * (initial_decaf_percent / 100) +
                     additional_stock * (additional_decaf_percent / 100)
  total_decaf / total_stock * 100 = 36 := by
sorry


end NUMINAMATH_CALUDE_decaf_percentage_l1723_172369


namespace NUMINAMATH_CALUDE_percentage_of_defective_meters_l1723_172361

theorem percentage_of_defective_meters 
  (total_meters : ℕ) 
  (rejected_meters : ℕ) 
  (h1 : total_meters = 200) 
  (h2 : rejected_meters = 20) : 
  (rejected_meters : ℝ) / (total_meters : ℝ) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_defective_meters_l1723_172361


namespace NUMINAMATH_CALUDE_joan_seashells_left_l1723_172362

/-- The number of seashells Joan has left after giving some to Sam -/
def seashells_left (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

/-- Theorem stating that Joan has 27 seashells left -/
theorem joan_seashells_left : seashells_left 70 43 = 27 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_left_l1723_172362


namespace NUMINAMATH_CALUDE_alex_calculation_l1723_172350

theorem alex_calculation (x : ℝ) : 
  (x / 9 - 21 = 24) → (x * 9 + 21 = 3666) := by
  sorry

end NUMINAMATH_CALUDE_alex_calculation_l1723_172350


namespace NUMINAMATH_CALUDE_share_multiple_is_four_l1723_172394

/-- Represents the shares of three people in a division problem. -/
structure Shares where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Proves that the multiple of a's share is 4 under given conditions. -/
theorem share_multiple_is_four 
  (total : ℝ) 
  (shares : Shares) 
  (h_total : total = 880)
  (h_c_share : shares.c = 160)
  (h_sum : shares.a + shares.b + shares.c = total)
  (h_equal : ∃ x : ℝ, x * shares.a = 5 * shares.b ∧ x * shares.a = 10 * shares.c) :
  ∃ x : ℝ, x = 4 ∧ x * shares.a = 5 * shares.b ∧ x * shares.a = 10 * shares.c := by
  sorry

end NUMINAMATH_CALUDE_share_multiple_is_four_l1723_172394


namespace NUMINAMATH_CALUDE_parity_equality_of_extrema_l1723_172331

/-- A set of elements related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The maximum element of A_P -/
def max_element (A : Set ℤ) : ℤ := sorry

/-- The minimum element of A_P -/
def min_element (A : Set ℤ) : ℤ := sorry

/-- Parity of an integer -/
def parity (n : ℤ) : Bool := n % 2 = 0

/-- Theorem: The parity of the smallest and largest elements of A_P is the same -/
theorem parity_equality_of_extrema :
  parity (min_element A_P) = parity (max_element A_P) := by
  sorry

end NUMINAMATH_CALUDE_parity_equality_of_extrema_l1723_172331


namespace NUMINAMATH_CALUDE_alternating_coloring_uniform_rows_l1723_172384

/-- Represents a color in the pattern -/
inductive Color
| A
| B

/-- Represents the grid of the bracelet -/
def BraceletGrid := Fin 10 → Fin 2 → Color

/-- A coloring function that alternates colors in each column -/
def alternatingColoring : BraceletGrid :=
  fun i j => if j = 0 then Color.A else Color.B

/-- Theorem stating that the alternating coloring results in uniform rows -/
theorem alternating_coloring_uniform_rows :
  (∀ i : Fin 10, alternatingColoring i 0 = Color.A) ∧
  (∀ i : Fin 10, alternatingColoring i 1 = Color.B) := by
  sorry


end NUMINAMATH_CALUDE_alternating_coloring_uniform_rows_l1723_172384


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4851_l1723_172347

theorem largest_prime_factor_of_4851 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 4851 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 4851 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4851_l1723_172347


namespace NUMINAMATH_CALUDE_total_golf_balls_l1723_172390

/-- Represents the number of golf balls in one dozen -/
def dozen : ℕ := 12

/-- Represents the number of dozens Dan buys -/
def dan_dozens : ℕ := 5

/-- Represents the number of dozens Gus buys -/
def gus_dozens : ℕ := 3

/-- Represents the number of dozens Chris buys -/
def chris_dozens : ℕ := 4

/-- Represents the additional golf balls Chris buys -/
def chris_extra : ℕ := 6

/-- Represents the number of dozens Emily buys -/
def emily_dozens : ℕ := 2

/-- Represents the number of dozens Fred buys -/
def fred_dozens : ℕ := 1

/-- Theorem stating the total number of golf balls bought by the friends -/
theorem total_golf_balls :
  (dan_dozens + gus_dozens + chris_dozens + emily_dozens + fred_dozens) * dozen + chris_extra = 186 := by
  sorry

end NUMINAMATH_CALUDE_total_golf_balls_l1723_172390


namespace NUMINAMATH_CALUDE_environmental_law_support_l1723_172363

theorem environmental_law_support (men : ℕ) (women : ℕ) 
  (men_support_percent : ℚ) (women_support_percent : ℚ) 
  (h1 : men = 200) 
  (h2 : women = 800) 
  (h3 : men_support_percent = 75 / 100) 
  (h4 : women_support_percent = 65 / 100) : 
  (men_support_percent * men + women_support_percent * women) / (men + women) = 67 / 100 := by
  sorry

end NUMINAMATH_CALUDE_environmental_law_support_l1723_172363


namespace NUMINAMATH_CALUDE_absolute_value_greater_than_two_l1723_172352

theorem absolute_value_greater_than_two (x : ℝ) : |x| > 2 ↔ x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_greater_than_two_l1723_172352


namespace NUMINAMATH_CALUDE_staircase_perimeter_l1723_172366

/-- A staircase-shaped region with right angles -/
structure StaircaseRegion where
  /-- The number of 1-foot sides in the staircase -/
  num_sides : ℕ
  /-- The area of the region in square feet -/
  area : ℝ
  /-- Assumption that the number of sides is 10 -/
  sides_eq_ten : num_sides = 10
  /-- Assumption that the area is 85 square feet -/
  area_eq_85 : area = 85

/-- Calculate the perimeter of a staircase region -/
def perimeter (r : StaircaseRegion) : ℝ := sorry

/-- Theorem stating that the perimeter of the given staircase region is 30.5 feet -/
theorem staircase_perimeter (r : StaircaseRegion) : perimeter r = 30.5 := by sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l1723_172366


namespace NUMINAMATH_CALUDE_rogers_coin_piles_l1723_172375

theorem rogers_coin_piles (num_quarter_piles num_dime_piles coins_per_pile total_coins : ℕ) :
  num_quarter_piles = num_dime_piles →
  coins_per_pile = 7 →
  total_coins = 42 →
  num_quarter_piles * coins_per_pile + num_dime_piles * coins_per_pile = total_coins →
  num_quarter_piles = 3 := by
  sorry

end NUMINAMATH_CALUDE_rogers_coin_piles_l1723_172375


namespace NUMINAMATH_CALUDE_scientific_notation_of_120_l1723_172329

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_120 :
  toScientificNotation 120 = ScientificNotation.mk 1.2 2 (by norm_num) (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_120_l1723_172329


namespace NUMINAMATH_CALUDE_product_sum_inequality_l1723_172373

theorem product_sum_inequality (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : a₁ < a₂) (h2 : b₁ < b₂) : 
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l1723_172373


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1723_172357

/-- 
Given an equation (x^2)/(15-k) + (y^2)/(k-9) = 1 that represents an ellipse with foci on the y-axis,
prove that k is in the open interval (12, 15).
-/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 / (15 - k) + y^2 / (k - 9) = 1) →  -- equation represents an ellipse
  (∃ c : ℝ, ∀ x y : ℝ, x^2 / (15 - k) + y^2 / (k - 9) = 1 ↔ 
    y^2 / (k - 9) + x^2 / (15 - k) = 1 ∧ 
    y^2 / c^2 - x^2 / (k - 9 - c^2) = 1) →  -- foci are on y-axis
  k > 12 ∧ k < 15 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1723_172357


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1723_172349

theorem functional_equation_solution (a : ℝ) (ha : a ≠ 0) :
  ∀ f : ℝ → ℝ, (∀ x : ℝ, f (a + x) = f x - x) →
  ∃ C : ℝ, ∀ x : ℝ, f x = C + x^2 / (2 * a) - x / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1723_172349


namespace NUMINAMATH_CALUDE_steven_peach_apple_difference_l1723_172321

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 17

/-- The number of apples Steven has -/
def steven_apples : ℕ := 16

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - 6

/-- The number of apples Jake has -/
def jake_apples : ℕ := steven_apples + 8

/-- Theorem stating that Steven has 1 more peach than apples -/
theorem steven_peach_apple_difference :
  steven_peaches - steven_apples = 1 := by sorry

end NUMINAMATH_CALUDE_steven_peach_apple_difference_l1723_172321


namespace NUMINAMATH_CALUDE_find_y_l1723_172315

theorem find_y (a b : ℝ) (y : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let s := (3 * a) ^ (2 * b)
  s = 5 * a^b * y^b →
  y = 9 * a / 5 := by
sorry

end NUMINAMATH_CALUDE_find_y_l1723_172315


namespace NUMINAMATH_CALUDE_waitress_average_orders_per_hour_l1723_172387

theorem waitress_average_orders_per_hour
  (hourly_wage : ℝ)
  (tip_rate : ℝ)
  (num_shifts : ℕ)
  (hours_per_shift : ℕ)
  (total_earnings : ℝ)
  (h1 : hourly_wage = 4)
  (h2 : tip_rate = 0.15)
  (h3 : num_shifts = 3)
  (h4 : hours_per_shift = 8)
  (h5 : total_earnings = 240) :
  let total_hours : ℕ := num_shifts * hours_per_shift
  let wage_earnings : ℝ := hourly_wage * total_hours
  let tip_earnings : ℝ := total_earnings - wage_earnings
  let total_orders : ℝ := tip_earnings / tip_rate
  let avg_orders_per_hour : ℝ := total_orders / total_hours
  avg_orders_per_hour = 40 := by
sorry

end NUMINAMATH_CALUDE_waitress_average_orders_per_hour_l1723_172387


namespace NUMINAMATH_CALUDE_polynomial_equality_l1723_172343

theorem polynomial_equality (x : ℝ) :
  (∃ t c : ℝ, (6*x^2 - 8*x + 9)*(3*x^2 + t*x + 8) = 18*x^4 - 54*x^3 + c*x^2 - 56*x + 72) ↔
  (∃ t c : ℝ, t = -5 ∧ c = 115) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1723_172343


namespace NUMINAMATH_CALUDE_percent_less_u_than_y_l1723_172305

theorem percent_less_u_than_y 
  (w u y z : ℝ) 
  (hw : w = 0.60 * u) 
  (hz1 : z = 0.54 * y) 
  (hz2 : z = 1.50 * w) : 
  u = 0.60 * y := by sorry

end NUMINAMATH_CALUDE_percent_less_u_than_y_l1723_172305


namespace NUMINAMATH_CALUDE_rectangle_horizontal_length_l1723_172318

/-- Given a square with side length 80 cm and a rectangle with vertical length 100 cm,
    if their perimeters are equal, then the horizontal length of the rectangle is 60 cm. -/
theorem rectangle_horizontal_length (square_side : ℝ) (rect_vertical : ℝ) (rect_horizontal : ℝ) :
  square_side = 80 ∧ rect_vertical = 100 ∧ 4 * square_side = 2 * (rect_vertical + rect_horizontal) →
  rect_horizontal = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_horizontal_length_l1723_172318


namespace NUMINAMATH_CALUDE_total_gum_pieces_l1723_172353

theorem total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) (h1 : packages = 9) (h2 : pieces_per_package = 15) :
  packages * pieces_per_package = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_gum_pieces_l1723_172353


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_with_difference_l1723_172377

theorem smallest_sum_of_squares_with_difference (x y : ℕ) : 
  x^2 - y^2 = 221 → 
  x^2 + y^2 ≥ 229 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_with_difference_l1723_172377


namespace NUMINAMATH_CALUDE_unique_solutions_l1723_172311

-- Define the coprime relation
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the equation
def satisfies_equation (x y : ℕ) : Prop := x^2 - x + 1 = y^3

-- Main theorem
theorem unique_solutions :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  coprime x (y-1) →
  satisfies_equation x y →
  (x = 1 ∧ y = 1) ∨ (x = 19 ∧ y = 7) :=
sorry

end NUMINAMATH_CALUDE_unique_solutions_l1723_172311


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1723_172340

theorem modulus_of_complex_number (z : ℂ) : z = 3 - 2*I → Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1723_172340


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1723_172354

theorem sin_cos_identity : 
  Real.sin (15 * π / 180) * Real.sin (105 * π / 180) - 
  Real.cos (15 * π / 180) * Real.cos (105 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1723_172354


namespace NUMINAMATH_CALUDE_perpendicular_tangents_point_l1723_172337

/-- The point on the line y = x from which two perpendicular tangents 
    can be drawn to the parabola y = x^2 -/
theorem perpendicular_tangents_point :
  ∃! P : ℝ × ℝ, 
    (P.1 = P.2) ∧ 
    (∃ m₁ m₂ : ℝ, 
      (m₁ * m₂ = -1) ∧
      (∀ x y : ℝ, y = m₁ * (x - P.1) + P.2 → y = x^2 → x = P.1) ∧
      (∀ x y : ℝ, y = m₂ * (x - P.1) + P.2 → y = x^2 → x = P.1)) ∧
    P = (-1/4, -1/4) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_point_l1723_172337


namespace NUMINAMATH_CALUDE_pythagorean_field_planting_l1723_172382

theorem pythagorean_field_planting (a b : ℝ) (h1 : a = 5) (h2 : b = 12) : 
  let c := Real.sqrt (a^2 + b^2)
  let x := (a * b) / c
  let triangle_area := (a * b) / 2
  let square_area := x^2
  let planted_area := triangle_area - square_area
  let shortest_distance := (2 * square_area) / c
  shortest_distance = 3 → planted_area / triangle_area = 792 / 845 := by
sorry


end NUMINAMATH_CALUDE_pythagorean_field_planting_l1723_172382


namespace NUMINAMATH_CALUDE_expression_value_l1723_172300

theorem expression_value (a b : ℝ) (h : a + 3*b = 4) : 2*a + 6*b - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1723_172300


namespace NUMINAMATH_CALUDE_nursery_school_fraction_l1723_172378

theorem nursery_school_fraction (total_students : ℕ) 
  (under_three : ℕ) (not_between_three_and_four : ℕ) :
  total_students = 50 →
  under_three = 20 →
  not_between_three_and_four = 25 →
  (total_students - not_between_three_and_four : ℚ) / total_students = 9 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_nursery_school_fraction_l1723_172378


namespace NUMINAMATH_CALUDE_cistern_filling_fraction_l1723_172328

theorem cistern_filling_fraction (fill_time : ℝ) (fraction : ℝ) : 
  (fill_time = 25) → 
  (fraction * fill_time = 25) → 
  (fraction = 1 / 25) :=
by sorry

end NUMINAMATH_CALUDE_cistern_filling_fraction_l1723_172328


namespace NUMINAMATH_CALUDE_area_of_graph_l1723_172308

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop := abs (2 * x) + abs (3 * y) = 6

/-- The set of points satisfying the equation -/
def graph_set : Set (ℝ × ℝ) := {p | graph_equation p.1 p.2}

/-- The area enclosed by the graph -/
noncomputable def enclosed_area : ℝ := sorry

theorem area_of_graph : enclosed_area = 12 := by sorry

end NUMINAMATH_CALUDE_area_of_graph_l1723_172308


namespace NUMINAMATH_CALUDE_place_two_after_three_digit_number_l1723_172312

/-- Given a three-digit number with hundreds digit a, tens digit b, and units digit c,
    prove that placing the digit 2 after this number results in 1000a + 100b + 10c + 2 -/
theorem place_two_after_three_digit_number (a b c : ℕ) :
  let original := 100 * a + 10 * b + c
  10 * original + 2 = 1000 * a + 100 * b + 10 * c + 2 := by
  sorry

end NUMINAMATH_CALUDE_place_two_after_three_digit_number_l1723_172312


namespace NUMINAMATH_CALUDE_lydia_apple_eating_age_l1723_172322

/-- The age at which Lydia will eat an apple from her tree for the first time -/
def apple_eating_age (planting_age : ℕ) (years_to_bear_fruit : ℕ) : ℕ :=
  planting_age + years_to_bear_fruit

/-- Theorem stating Lydia's age when she first eats an apple from her tree -/
theorem lydia_apple_eating_age :
  apple_eating_age 4 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_lydia_apple_eating_age_l1723_172322


namespace NUMINAMATH_CALUDE_students_playing_neither_sport_l1723_172346

theorem students_playing_neither_sport (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ)
  (h_total : total = 40)
  (h_football : football = 26)
  (h_tennis : tennis = 20)
  (h_both : both = 17) :
  total - (football + tennis - both) = 11 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_sport_l1723_172346


namespace NUMINAMATH_CALUDE_sine_function_translation_l1723_172368

theorem sine_function_translation (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x)
  let g : ℝ → ℝ := λ x ↦ f (x + π / (4 * ω))
  (∀ x : ℝ, g (2 * ω - x) = g x) →
  (∀ x y : ℝ, -ω < x ∧ x < y ∧ y < ω → g x < g y) →
  ω = Real.sqrt (π / 2) := by
sorry

end NUMINAMATH_CALUDE_sine_function_translation_l1723_172368


namespace NUMINAMATH_CALUDE_min_points_for_proximity_l1723_172301

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define the distance function between two points on the circle
def circleDistance (p q : Circle) : ℝ := sorry

-- Define the sequence of points
def circlePoints : ℕ → Circle := sorry

-- Theorem statement
theorem min_points_for_proximity :
  ∀ n : ℕ, n < 20 →
  ∃ i j : ℕ, i < j ∧ j < n ∧ circleDistance (circlePoints i) (circlePoints j) ≥ 1/5 :=
sorry

end NUMINAMATH_CALUDE_min_points_for_proximity_l1723_172301


namespace NUMINAMATH_CALUDE_midpoint_of_intersection_l1723_172336

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

-- Define the line that intersects the ellipse
def intersecting_line (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  intersecting_line A.1 A.2 ∧ intersecting_line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem midpoint_of_intersection :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  (A.1 + B.1) / 2 = -9/5 ∧ (A.2 + B.2) / 2 = 1/5 :=
sorry

end NUMINAMATH_CALUDE_midpoint_of_intersection_l1723_172336


namespace NUMINAMATH_CALUDE_product_of_decimals_l1723_172317

theorem product_of_decimals : (0.7 : ℝ) * 0.3 = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1723_172317


namespace NUMINAMATH_CALUDE_extremum_at_one_lower_bound_ln_two_l1723_172351

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

-- Theorem for the first part of the problem
theorem extremum_at_one (a : ℝ) (h : a > 0) :
  (∃ ε > 0, ∀ x ∈ Set.Icc (1 - ε) (1 + ε), f a x ≥ f a 1) ↔ a = 1 :=
sorry

-- Theorem for the second part of the problem
theorem lower_bound_ln_two (a : ℝ) (h : a > 0) :
  (∀ x ≥ 0, f a x ≥ Real.log 2) ↔ a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_extremum_at_one_lower_bound_ln_two_l1723_172351


namespace NUMINAMATH_CALUDE_value_of_x_l1723_172371

theorem value_of_x (n : ℝ) (x : ℝ) 
  (h1 : x = 3 * n) 
  (h2 : 2 * n + 3 = 0.20 * 25) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l1723_172371


namespace NUMINAMATH_CALUDE_solution_in_interval_l1723_172344

open Real

theorem solution_in_interval (x₀ : ℝ) (k : ℤ) : 
  (8 - x₀ = log x₀) → 
  (x₀ ∈ Set.Ioo (k : ℝ) (k + 1)) → 
  k = 7 := by
sorry

end NUMINAMATH_CALUDE_solution_in_interval_l1723_172344


namespace NUMINAMATH_CALUDE_extra_day_percentage_increase_l1723_172345

/-- Calculates the percentage increase in daily rate for an extra workday --/
theorem extra_day_percentage_increase
  (regular_daily_rate : ℚ)
  (regular_work_days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_monthly_earnings_with_extra_day : ℚ)
  (h1 : regular_daily_rate = 8)
  (h2 : regular_work_days_per_week = 5)
  (h3 : weeks_per_month = 4)
  (h4 : total_monthly_earnings_with_extra_day = 208) :
  let regular_monthly_earnings := regular_daily_rate * regular_work_days_per_week * weeks_per_month
  let extra_day_earnings := total_monthly_earnings_with_extra_day - regular_monthly_earnings
  let extra_day_rate := extra_day_earnings / weeks_per_month
  let percentage_increase := (extra_day_rate - regular_daily_rate) / regular_daily_rate * 100
  percentage_increase = 50 := by
sorry

end NUMINAMATH_CALUDE_extra_day_percentage_increase_l1723_172345


namespace NUMINAMATH_CALUDE_jane_started_with_87_crayons_l1723_172396

/-- The number of crayons Jane started with -/
def initial_crayons : ℕ := sorry

/-- The number of crayons eaten by the hippopotamus -/
def eaten_crayons : ℕ := 7

/-- The number of crayons Jane ended up with -/
def remaining_crayons : ℕ := 80

/-- Theorem stating that Jane started with 87 crayons -/
theorem jane_started_with_87_crayons :
  initial_crayons = eaten_crayons + remaining_crayons :=
by sorry

end NUMINAMATH_CALUDE_jane_started_with_87_crayons_l1723_172396


namespace NUMINAMATH_CALUDE_angle_C_measure_l1723_172372

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.b^2 + t.c^2 - Real.sqrt 3 * t.b * t.c = t.a^2 ∧
  t.b * t.c = Real.sqrt 3 * t.a^2

-- Theorem statement
theorem angle_C_measure (t : Triangle) 
  (h : satisfiesConditions t) : t.angleC = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l1723_172372


namespace NUMINAMATH_CALUDE_a4_range_l1723_172356

theorem a4_range (a₁ a₂ a₃ a₄ : ℝ) 
  (sum_zero : a₁ + a₂ + a₃ = 0)
  (quad_eq : a₁ * a₄^2 + a₂ * a₄ - a₂ = 0)
  (order : a₁ > a₂ ∧ a₂ > a₃) :
  -1/2 - Real.sqrt 5/2 < a₄ ∧ a₄ < -1/2 + Real.sqrt 5/2 := by
sorry

end NUMINAMATH_CALUDE_a4_range_l1723_172356


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1723_172316

theorem simplify_trig_expression (α : Real) (h : π / 2 < α ∧ α < π) :
  Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) - Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) = -2 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1723_172316


namespace NUMINAMATH_CALUDE_geometric_progression_existence_l1723_172374

/-- A geometric progression containing 27, 8, and 12 exists, and their positions satisfy m = 3p - 2n -/
theorem geometric_progression_existence :
  ∃ (a q : ℝ) (m n p : ℕ), 
    (a * q^(m-1) = 27) ∧ 
    (a * q^(n-1) = 8) ∧ 
    (a * q^(p-1) = 12) ∧ 
    (m = 3*p - 2*n) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_existence_l1723_172374


namespace NUMINAMATH_CALUDE_channel_system_properties_l1723_172323

/-- Represents a water channel system with nodes A to H -/
structure ChannelSystem where
  /-- Flow rate in channel BC -/
  q₀ : ℝ
  /-- Flow rate in channel AB -/
  q_AB : ℝ
  /-- Flow rate in channel AH -/
  q_AH : ℝ

/-- The flow rates in the channel system satisfy the given conditions -/
def is_valid_system (sys : ChannelSystem) : Prop :=
  sys.q_AB = (1/2) * sys.q₀ ∧
  sys.q_AH = (3/4) * sys.q₀

/-- The total flow rate entering at node A -/
def total_flow_A (sys : ChannelSystem) : ℝ :=
  sys.q_AB + sys.q_AH

/-- Theorem stating the properties of the channel system -/
theorem channel_system_properties (sys : ChannelSystem) 
  (h : is_valid_system sys) : 
  sys.q_AB = (1/2) * sys.q₀ ∧ 
  sys.q_AH = (3/4) * sys.q₀ ∧ 
  total_flow_A sys = (7/4) * sys.q₀ := by
  sorry

end NUMINAMATH_CALUDE_channel_system_properties_l1723_172323


namespace NUMINAMATH_CALUDE_positive_sum_and_product_equivalence_l1723_172370

theorem positive_sum_and_product_equivalence (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by sorry

end NUMINAMATH_CALUDE_positive_sum_and_product_equivalence_l1723_172370


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l1723_172388

/-- Given a quadratic expression 3x^2 + 9x + 20, prove that when expressed in the form a(x - h)^2 + k, the value of h is -3/2. -/
theorem quadratic_completion_of_square (x : ℝ) :
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l1723_172388


namespace NUMINAMATH_CALUDE_no_intersection_l1723_172385

theorem no_intersection :
  ¬∃ (x y : ℝ), (y = |3*x + 4| ∧ y = -|2*x + 1|) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_l1723_172385


namespace NUMINAMATH_CALUDE_wood_square_weight_relation_second_wood_square_weight_l1723_172391

/-- Represents the properties of a square piece of wood -/
structure WoodSquare where
  side_length : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two square pieces of wood with uniform density and thickness -/
theorem wood_square_weight_relation (w1 w2 : WoodSquare) 
  (h1 : w1.side_length = 3)
  (h2 : w1.weight = 12)
  (h3 : w2.side_length = 6) :
  w2.weight = 48 := by
  sorry

/-- Main theorem proving the weight of the second piece of wood -/
theorem second_wood_square_weight :
  ∃ (w1 w2 : WoodSquare), 
    w1.side_length = 3 ∧ 
    w1.weight = 12 ∧ 
    w2.side_length = 6 ∧ 
    w2.weight = 48 := by
  sorry

end NUMINAMATH_CALUDE_wood_square_weight_relation_second_wood_square_weight_l1723_172391


namespace NUMINAMATH_CALUDE_smallest_y_value_l1723_172395

theorem smallest_y_value (y : ℝ) : 
  (3 * y^2 + 33 * y - 90 = y * (y + 18)) → y ≥ -9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_value_l1723_172395


namespace NUMINAMATH_CALUDE_root_equation_l1723_172332

theorem root_equation (k : ℝ) : 
  ((-2 : ℝ)^2 + k*(-2) - 2 = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_l1723_172332


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1723_172334

-- Define the line equation
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x - 3

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 25

-- Theorem stating that any real slope m results in an intersection
theorem line_intersects_circle (m : ℝ) :
  ∃ x : ℝ, circle_equation x (line_equation m x) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1723_172334


namespace NUMINAMATH_CALUDE_claire_earnings_l1723_172325

def total_flowers : ℕ := 400
def tulips : ℕ := 120
def white_roses : ℕ := 80
def red_rose_price : ℚ := 3/4

def roses : ℕ := total_flowers - tulips
def red_roses : ℕ := roses - white_roses
def red_roses_to_sell : ℕ := red_roses / 2

theorem claire_earnings : 
  (red_roses_to_sell : ℚ) * red_rose_price = 75 := by sorry

end NUMINAMATH_CALUDE_claire_earnings_l1723_172325


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l1723_172358

theorem triangle_third_side_length 
  (a b : ℝ) 
  (θ : ℝ) 
  (ha : a = 10) 
  (hb : b = 15) 
  (hθ : θ = Real.pi / 3) : 
  Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos θ)) = 5 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l1723_172358


namespace NUMINAMATH_CALUDE_second_term_of_specific_sequence_l1723_172320

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem second_term_of_specific_sequence :
  ∀ (d : ℝ),
  arithmetic_sequence 2020 d 1 = 2020 ∧
  arithmetic_sequence 2020 d 5 = 4040 →
  arithmetic_sequence 2020 d 2 = 2525 :=
by
  sorry

end NUMINAMATH_CALUDE_second_term_of_specific_sequence_l1723_172320


namespace NUMINAMATH_CALUDE_sum_at_one_and_neg_one_l1723_172364

/-- A cubic polynomial Q satisfying specific conditions -/
structure CubicPolynomial (l : ℝ) where
  Q : ℝ → ℝ
  is_cubic : ∃ a b c : ℝ, ∀ x, Q x = a * x^3 + b * x^2 + c * x + l
  cond_0 : Q 0 = l
  cond_2 : Q 2 = 3 * l
  cond_neg_2 : Q (-2) = 5 * l

/-- Theorem stating the sum of Q(1) and Q(-1) -/
theorem sum_at_one_and_neg_one (l : ℝ) (poly : CubicPolynomial l) : 
  poly.Q 1 + poly.Q (-1) = (7/2) * l := by
  sorry

end NUMINAMATH_CALUDE_sum_at_one_and_neg_one_l1723_172364


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_b_value_l1723_172392

theorem polynomial_factor_implies_b_value (a b : ℤ) :
  (∃ (c : ℤ), (X^2 - X - 1) * (a*X - c) = a*X^3 + b*X^2 - X + 1) →
  b = -1 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_b_value_l1723_172392


namespace NUMINAMATH_CALUDE_f_negative_two_lt_f_one_l1723_172330

/-- A function f : ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The derivative of f for positive x -/
def DerivativePositive (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, deriv f x = (x - 1) * (x - 2)

theorem f_negative_two_lt_f_one
  (f : ℝ → ℝ)
  (heven : IsEven f)
  (hderiv : DerivativePositive f) :
  f (-2) < f 1 :=
sorry

end NUMINAMATH_CALUDE_f_negative_two_lt_f_one_l1723_172330


namespace NUMINAMATH_CALUDE_amount_after_two_years_l1723_172327

theorem amount_after_two_years 
  (present_value : ℝ) 
  (yearly_increase_rate : ℝ) 
  (h1 : present_value = 57600) 
  (h2 : yearly_increase_rate = 1/8) 
  (h3 : (present_value * (1 + yearly_increase_rate)^2 : ℝ) = 72900) : 
  (present_value * (1 + yearly_increase_rate)^2 : ℝ) = 72900 := by
  sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l1723_172327


namespace NUMINAMATH_CALUDE_sixty_second_pair_l1723_172314

/-- Definition of our sequence of pairs -/
def pair_sequence : ℕ → ℕ × ℕ
| 0 => (1, 1)
| n + 1 =>
  let (a, b) := pair_sequence n
  if a = 1 then (b + 1, 1)
  else (a - 1, b + 1)

/-- The 62nd pair in the sequence is (7,5) -/
theorem sixty_second_pair :
  pair_sequence 61 = (7, 5) :=
sorry

end NUMINAMATH_CALUDE_sixty_second_pair_l1723_172314


namespace NUMINAMATH_CALUDE_optimal_price_l1723_172309

/-- Represents the daily sales profit function for an agricultural product. -/
def W (x : ℝ) : ℝ := -2 * x^2 + 120 * x - 1600

/-- Represents the daily sales quantity function for an agricultural product. -/
def y (x : ℝ) : ℝ := -2 * x + 80

/-- The cost price per kilogram of the agricultural product. -/
def cost_price : ℝ := 20

/-- The maximum allowed selling price per kilogram. -/
def max_price : ℝ := 30

/-- The desired daily sales profit. -/
def target_profit : ℝ := 150

/-- Theorem stating that a selling price of 25 achieves the target profit
    while satisfying the given conditions. -/
theorem optimal_price :
  W 25 = target_profit ∧
  25 ≤ max_price ∧
  y 25 > 0 :=
sorry

end NUMINAMATH_CALUDE_optimal_price_l1723_172309


namespace NUMINAMATH_CALUDE_jenna_costume_cost_l1723_172319

/-- Represents the cost of material for Jenna's costume --/
def costume_cost (skirt_length : ℝ) (skirt_width : ℝ) (num_skirts : ℕ) 
                 (bodice_area : ℝ) (sleeve_area : ℝ) (num_sleeves : ℕ) 
                 (cost_per_sqft : ℝ) : ℝ :=
  let skirt_area := skirt_length * skirt_width
  let total_skirt_area := skirt_area * num_skirts
  let total_sleeve_area := sleeve_area * num_sleeves
  let total_area := total_skirt_area + total_sleeve_area + bodice_area
  total_area * cost_per_sqft

/-- Theorem: The total cost of material for Jenna's costume is $468 --/
theorem jenna_costume_cost : 
  costume_cost 12 4 3 2 5 2 3 = 468 := by
  sorry

end NUMINAMATH_CALUDE_jenna_costume_cost_l1723_172319


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l1723_172324

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x, 2 * a * x^2 - 2 * x + 3 < 0 ↔ 2 < x ∧ x < b) →
  (∀ x, 3 * x^2 + 2 * x + 2 * a < 0 ↔ -1/2 < x ∧ x < -1/6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l1723_172324


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1723_172379

theorem absolute_value_inequality (x : ℝ) :
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ ((1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1723_172379


namespace NUMINAMATH_CALUDE_m_range_l1723_172398

/-- The range of m given the specified conditions -/
theorem m_range (h1 : ∀ x : ℝ, 2 * x > m * (x^2 + 1)) 
                (h2 : ∃ x₀ : ℝ, x₀^2 + 2*x₀ - m - 1 = 0) : 
  -2 ≤ m ∧ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l1723_172398


namespace NUMINAMATH_CALUDE_quadratic_solution_l1723_172348

theorem quadratic_solution :
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ x = 2 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1723_172348


namespace NUMINAMATH_CALUDE_nell_initial_cards_l1723_172355

/-- The number of baseball cards Nell initially had -/
def initial_cards : ℕ := sorry

/-- The number of cards Nell has at the end -/
def final_cards : ℕ := 154

/-- The number of cards Nell gave to Jeff -/
def cards_given_to_jeff : ℕ := 301

/-- The number of new cards Nell bought -/
def new_cards_bought : ℕ := 60

/-- The number of cards Nell traded away to Sam -/
def cards_traded_away : ℕ := 45

/-- The number of cards Nell received from Sam -/
def cards_received : ℕ := 30

/-- Theorem stating that Nell's initial number of baseball cards was 410 -/
theorem nell_initial_cards :
  initial_cards = 410 :=
by sorry

end NUMINAMATH_CALUDE_nell_initial_cards_l1723_172355


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l1723_172303

def a (n : ℕ) : ℚ := (1 - 2 * n^2) / (2 + 4 * n^2)

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-1/2)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l1723_172303


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1723_172339

theorem cube_sum_reciprocal (x : ℝ) (h : x ≠ 0) :
  x + 1 / x = 3 → x^3 + 1 / x^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1723_172339


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_plus_one_l1723_172306

theorem root_sum_reciprocal_plus_one (a b c : ℂ) : 
  (a^3 - a - 2 = 0) → (b^3 - b - 2 = 0) → (c^3 - c - 2 = 0) →
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 2) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_plus_one_l1723_172306


namespace NUMINAMATH_CALUDE_find_x_l1723_172365

theorem find_x : ∃ x : ℕ, 
  (∃ k : ℕ, x = 8 * k) ∧ 
  x^2 > 100 ∧ 
  x < 20 ∧ 
  x = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1723_172365
