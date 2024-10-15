import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_minimum_zero_l2540_254068

/-- Given a quadratic function y = (1+a)x^2 + px + q with a minimum value of zero,
    where a is a positive constant, prove that q = p^2 / (4(1+a)). -/
theorem quadratic_minimum_zero (a p q : ℝ) (ha : a > 0) :
  (∃ (k : ℝ), ∀ (x : ℝ), (1 + a) * x^2 + p * x + q ≥ k) ∧ 
  (∃ (x : ℝ), (1 + a) * x^2 + p * x + q = 0) →
  q = p^2 / (4 * (1 + a)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_zero_l2540_254068


namespace NUMINAMATH_CALUDE_park_visitors_l2540_254062

/-- Given a park with visitors on Saturday and Sunday, calculate the total number of visitors over two days. -/
theorem park_visitors (saturday_visitors : ℕ) (sunday_extra : ℕ) : 
  saturday_visitors = 200 → sunday_extra = 40 → 
  saturday_visitors + (saturday_visitors + sunday_extra) = 440 := by
  sorry

#check park_visitors

end NUMINAMATH_CALUDE_park_visitors_l2540_254062


namespace NUMINAMATH_CALUDE_james_total_money_l2540_254047

-- Define the currency types
inductive Currency
| USD
| EUR

-- Define the money type
structure Money where
  amount : ℚ
  currency : Currency

-- Define the exchange rate
def exchange_rate : ℚ := 1.20

-- Define James's wallet contents
def wallet_contents : List Money := [
  ⟨50, Currency.USD⟩,
  ⟨20, Currency.USD⟩,
  ⟨5, Currency.USD⟩
]

-- Define James's pocket contents
def pocket_contents : List Money := [
  ⟨20, Currency.USD⟩,
  ⟨10, Currency.USD⟩,
  ⟨5, Currency.EUR⟩
]

-- Define James's coin contents
def coin_contents : List Money := [
  ⟨0.25, Currency.USD⟩,
  ⟨0.25, Currency.USD⟩,
  ⟨0.10, Currency.USD⟩,
  ⟨0.10, Currency.USD⟩,
  ⟨0.10, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩
]

-- Function to convert EUR to USD
def convert_to_usd (m : Money) : Money :=
  match m.currency with
  | Currency.USD => m
  | Currency.EUR => ⟨m.amount * exchange_rate, Currency.USD⟩

-- Function to sum up all money in USD
def total_usd (money_list : List Money) : ℚ :=
  (money_list.map convert_to_usd).foldl (fun acc m => acc + m.amount) 0

-- Theorem statement
theorem james_total_money :
  total_usd (wallet_contents ++ pocket_contents ++ coin_contents) = 111.85 := by
  sorry

end NUMINAMATH_CALUDE_james_total_money_l2540_254047


namespace NUMINAMATH_CALUDE_min_value_on_transformed_curve_l2540_254046

-- Define the original curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x y t : ℝ) : Prop := x = 1 + t/2 ∧ y = 2 + (Real.sqrt 3)/2 * t

-- Define the transformation
def transform (x y x' y' : ℝ) : Prop := x' = 3*x ∧ y' = y

-- Define the transformed curve C'
def curve_C' (x y : ℝ) : Prop := x^2/9 + y^2 = 1

-- State the theorem
theorem min_value_on_transformed_curve :
  ∀ (x y : ℝ), curve_C' x y → (x + 2 * Real.sqrt 3 * y ≥ -Real.sqrt 21) :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_transformed_curve_l2540_254046


namespace NUMINAMATH_CALUDE_shenny_vacation_shirts_l2540_254010

/-- The number of shirts Shenny needs to pack for her vacation -/
def shirts_to_pack (vacation_days : ℕ) (same_shirt_days : ℕ) (shirts_per_day : ℕ) : ℕ :=
  (vacation_days - same_shirt_days) * shirts_per_day + 1

/-- Proof that Shenny needs to pack 11 shirts for her vacation -/
theorem shenny_vacation_shirts :
  shirts_to_pack 7 2 2 = 11 :=
by sorry

end NUMINAMATH_CALUDE_shenny_vacation_shirts_l2540_254010


namespace NUMINAMATH_CALUDE_increase_amount_l2540_254016

theorem increase_amount (x : ℝ) (amount : ℝ) (h : 15 * x + amount = 14) :
  amount = 14 - 14/15 := by
  sorry

end NUMINAMATH_CALUDE_increase_amount_l2540_254016


namespace NUMINAMATH_CALUDE_rectangular_prism_face_fits_in_rectangle_l2540_254015

/-- Represents a rectangular prism with dimensions a ≤ b ≤ c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : 0 < a
  h2 : a ≤ b
  h3 : b ≤ c

/-- Represents a rectangle with dimensions d₁ ≤ d₂ -/
structure Rectangle where
  d1 : ℝ
  d2 : ℝ
  h : d1 ≤ d2

/-- Theorem: Given a rectangular prism and a rectangle that can contain
    the prism's hexagonal cross-section, prove that the rectangle can
    contain one face of the prism -/
theorem rectangular_prism_face_fits_in_rectangle
  (prism : RectangularPrism) (rect : Rectangle)
  (hex_fits : ∃ (h : ℝ), h > 0 ∧ h^2 + rect.d1^2 ≥ prism.b^2 + prism.c^2) :
  min rect.d1 rect.d2 ≥ prism.a ∧ max rect.d1 rect.d2 ≥ prism.b :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_face_fits_in_rectangle_l2540_254015


namespace NUMINAMATH_CALUDE_family_reunion_soda_cost_l2540_254097

-- Define the given conditions
def people_attending : ℕ := 5 * 12
def cans_per_box : ℕ := 10
def cost_per_box : ℚ := 2
def cans_per_person : ℕ := 2
def family_members : ℕ := 6

-- Define the theorem
theorem family_reunion_soda_cost :
  (people_attending * cans_per_person / cans_per_box * cost_per_box) / family_members = 4 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_soda_cost_l2540_254097


namespace NUMINAMATH_CALUDE_triangle_inequality_l2540_254070

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2540_254070


namespace NUMINAMATH_CALUDE_todd_initial_gum_l2540_254091

/-- 
Given:
- Todd receives 16 pieces of gum from Steve.
- After receiving gum from Steve, Todd has 54 pieces of gum.

Prove that Todd initially had 38 pieces of gum.
-/
theorem todd_initial_gum (initial_gum : ℕ) : initial_gum + 16 = 54 ↔ initial_gum = 38 := by
  sorry

end NUMINAMATH_CALUDE_todd_initial_gum_l2540_254091


namespace NUMINAMATH_CALUDE_divisible_by_27_l2540_254067

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, (10 : ℤ)^n + 18*n - 1 = 27*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_27_l2540_254067


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2540_254049

theorem complex_number_in_first_quadrant : 
  let z : ℂ := 1 - (1 / Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2540_254049


namespace NUMINAMATH_CALUDE_alex_money_left_l2540_254075

/-- Calculates the amount of money Alex has left after deductions --/
theorem alex_money_left (weekly_income : ℕ) (tax_rate : ℚ) (water_bill : ℕ) (tithe_rate : ℚ) : 
  weekly_income = 500 →
  tax_rate = 1/10 →
  water_bill = 55 →
  tithe_rate = 1/10 →
  ↑weekly_income - (↑weekly_income * tax_rate + ↑water_bill + ↑weekly_income * tithe_rate) = 345 := by
sorry

end NUMINAMATH_CALUDE_alex_money_left_l2540_254075


namespace NUMINAMATH_CALUDE_polynomial_property_l2540_254057

def P (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem polynomial_property (a b c : ℝ) :
  P a b c 0 = 5 →
  (-a/3 : ℝ) = -c →
  (-a/3 : ℝ) = 1 + a + b + c →
  b = -26 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l2540_254057


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2540_254086

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-19, -7; 10, 3]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![85/14, -109/14; -3, 4]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2540_254086


namespace NUMINAMATH_CALUDE_distinct_values_theorem_l2540_254033

/-- The number of distinct values expressible as ip + jq -/
def distinct_values (n p q : ℕ) : ℕ :=
  if p = q ∧ p = 1 then
    n + 1
  else if p > q ∧ n < p then
    (n + 1) * (n + 2) / 2
  else if p > q ∧ n ≥ p then
    p * (2 * n - p + 3) / 2
  else
    0  -- This case is not specified in the problem, but needed for completeness

/-- Theorem stating the number of distinct values expressible as ip + jq -/
theorem distinct_values_theorem (n p q : ℕ) (h_coprime : Nat.Coprime p q) :
  distinct_values n p q =
    if p = q ∧ p = 1 then
      n + 1
    else if p > q ∧ n < p then
      (n + 1) * (n + 2) / 2
    else if p > q ∧ n ≥ p then
      p * (2 * n - p + 3) / 2
    else
      0 := by sorry

end NUMINAMATH_CALUDE_distinct_values_theorem_l2540_254033


namespace NUMINAMATH_CALUDE_total_distance_walked_l2540_254083

/-- Represents the hiking trail with flat and uphill sections -/
structure HikingTrail where
  flat_distance : ℝ  -- Distance of flat section (P to Q)
  uphill_distance : ℝ  -- Distance of uphill section (Q to R)

/-- Represents the hiker's journey -/
structure HikerJourney where
  trail : HikingTrail
  flat_speed : ℝ  -- Speed on flat sections
  uphill_speed : ℝ  -- Speed going uphill
  downhill_speed : ℝ  -- Speed going downhill
  total_time : ℝ  -- Total time of the journey in hours
  rest_time : ℝ  -- Time spent resting at point R

/-- Theorem stating the total distance walked by the hiker -/
theorem total_distance_walked (journey : HikerJourney) 
  (h1 : journey.flat_speed = 4)
  (h2 : journey.uphill_speed = 3)
  (h3 : journey.downhill_speed = 6)
  (h4 : journey.total_time = 7)
  (h5 : journey.rest_time = 1)
  (h6 : journey.flat_speed * (journey.total_time - journey.rest_time) / 2 + 
        journey.trail.uphill_distance * (1 / journey.uphill_speed + 1 / journey.downhill_speed) = 
        journey.total_time - journey.rest_time) :
  2 * (journey.trail.flat_distance + journey.trail.uphill_distance) = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_l2540_254083


namespace NUMINAMATH_CALUDE_square_difference_division_l2540_254052

theorem square_difference_division (a b c : ℕ) (h : (a^2 - b^2) / c = 488) :
  (144^2 - 100^2) / 22 = 488 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_division_l2540_254052


namespace NUMINAMATH_CALUDE_license_plate_count_l2540_254018

/-- Represents the number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- Represents the number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- Represents the number of digits in the license plate -/
def digit_count : ℕ := 5

/-- Represents the number of letters in the license plate -/
def letter_count : ℕ := 3

/-- Represents the total number of characters in the license plate -/
def total_chars : ℕ := digit_count + letter_count

/-- Calculates the number of ways to arrange the letter block within the license plate -/
def letter_block_positions : ℕ := total_chars - letter_count + 1

/-- Calculates the number of valid letter combinations (at least one 'A') -/
def valid_letter_combinations : ℕ := 3 * num_letters^2

/-- The main theorem stating the total number of distinct license plates -/
theorem license_plate_count : 
  letter_block_positions * num_digits^digit_count * valid_letter_combinations = 1216800000 := by
  sorry


end NUMINAMATH_CALUDE_license_plate_count_l2540_254018


namespace NUMINAMATH_CALUDE_one_and_two_thirds_of_number_is_45_l2540_254077

theorem one_and_two_thirds_of_number_is_45 : ∃ x : ℚ, (5 / 3) * x = 45 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_one_and_two_thirds_of_number_is_45_l2540_254077


namespace NUMINAMATH_CALUDE_price_reduction_proof_l2540_254048

/-- Given the initial price of a box of cereal, the number of boxes bought, and the total amount paid,
    prove that the price reduction per box is correct. -/
theorem price_reduction_proof (initial_price : ℕ) (boxes_bought : ℕ) (total_paid : ℕ) :
  initial_price = 104 →
  boxes_bought = 20 →
  total_paid = 1600 →
  initial_price - (total_paid / boxes_bought) = 24 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_proof_l2540_254048


namespace NUMINAMATH_CALUDE_peanut_mixture_solution_l2540_254080

/-- Represents the peanut mixture problem -/
def peanut_mixture (virginia_weight : ℝ) (virginia_cost : ℝ) (spanish_cost : ℝ) (mixture_cost : ℝ) : ℝ → Prop :=
  λ spanish_weight : ℝ =>
    (virginia_weight * virginia_cost + spanish_weight * spanish_cost) / (virginia_weight + spanish_weight) = mixture_cost

/-- Proves that 2.5 pounds of Spanish peanuts is the correct amount for the desired mixture -/
theorem peanut_mixture_solution :
  peanut_mixture 10 3.5 3 3.4 2.5 := by
  sorry

end NUMINAMATH_CALUDE_peanut_mixture_solution_l2540_254080


namespace NUMINAMATH_CALUDE_butterfly_ratio_l2540_254063

/-- Prove that the ratio of blue butterflies to yellow butterflies is 2:1 -/
theorem butterfly_ratio (total : ℕ) (black : ℕ) (blue : ℕ) 
  (h1 : total = 11)
  (h2 : black = 5)
  (h3 : blue = 4)
  : (blue : ℚ) / (total - black - blue) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_ratio_l2540_254063


namespace NUMINAMATH_CALUDE_sum_interior_angles_polygon_l2540_254037

theorem sum_interior_angles_polygon (n : ℕ) (h : n ≥ 3) :
  (360 / 30 : ℕ) = n → (n - 2) * 180 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_polygon_l2540_254037


namespace NUMINAMATH_CALUDE_jerrys_shelf_l2540_254099

theorem jerrys_shelf (books : ℕ) (added_figures : ℕ) (difference : ℕ) : 
  books = 7 → added_figures = 2 → difference = 2 →
  ∃ initial_figures : ℕ, 
    initial_figures = 3 ∧ 
    books = (initial_figures + added_figures) + difference :=
by sorry

end NUMINAMATH_CALUDE_jerrys_shelf_l2540_254099


namespace NUMINAMATH_CALUDE_pens_left_in_jar_l2540_254089

/-- The number of pens left in a jar after removing some pens -/
theorem pens_left_in_jar
  (initial_blue : ℕ)
  (initial_black : ℕ)
  (initial_red : ℕ)
  (blue_removed : ℕ)
  (black_removed : ℕ)
  (h1 : initial_blue = 9)
  (h2 : initial_black = 21)
  (h3 : initial_red = 6)
  (h4 : blue_removed = 4)
  (h5 : black_removed = 7)
  : initial_blue + initial_black + initial_red - blue_removed - black_removed = 25 := by
  sorry


end NUMINAMATH_CALUDE_pens_left_in_jar_l2540_254089


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l2540_254071

theorem digit_sum_puzzle :
  ∀ (A B C D E F : ℕ),
  (A ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) →
  (B ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) →
  (C ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) →
  (D ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) →
  (E ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) →
  (F ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F →
  (A + B) % 2 = 0 →
  (C + D) % 3 = 0 →
  (E + F) % 5 = 0 →
  min C D = 1 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l2540_254071


namespace NUMINAMATH_CALUDE_ribbon_calculation_l2540_254028

/-- Represents the types of ribbons available --/
inductive RibbonType
  | A
  | B

/-- Represents the wrapping pattern for a gift --/
structure WrappingPattern where
  typeA : Nat
  typeB : Nat

/-- Calculates the number of ribbons needed for a given number of gifts and wrapping pattern --/
def ribbonsNeeded (numGifts : Nat) (pattern : WrappingPattern) : Nat × Nat :=
  (numGifts * pattern.typeA, numGifts * pattern.typeB)

theorem ribbon_calculation (tomSupplyA tomSupplyB : Nat) :
  let oddPattern : WrappingPattern := { typeA := 1, typeB := 2 }
  let evenPattern : WrappingPattern := { typeA := 2, typeB := 1 }
  let (oddA, oddB) := ribbonsNeeded 4 oddPattern
  let (evenA, evenB) := ribbonsNeeded 4 evenPattern
  let totalA := oddA + evenA
  let totalB := oddB + evenB
  tomSupplyA = 10 ∧ tomSupplyB = 12 →
  totalA - tomSupplyA = 2 ∧ totalB - tomSupplyB = 0 := by
  sorry


end NUMINAMATH_CALUDE_ribbon_calculation_l2540_254028


namespace NUMINAMATH_CALUDE_thirteen_gumballs_needed_l2540_254085

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)
  (green : ℕ)

/-- Calculates the least number of gumballs needed to ensure four of the same color -/
def leastGumballs (machine : GumballMachine) : ℕ :=
  sorry

/-- Theorem stating that for the given gumball machine, 13 is the least number of gumballs needed -/
theorem thirteen_gumballs_needed (machine : GumballMachine) 
  (h1 : machine.red = 10)
  (h2 : machine.white = 9)
  (h3 : machine.blue = 8)
  (h4 : machine.green = 6) :
  leastGumballs machine = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_thirteen_gumballs_needed_l2540_254085


namespace NUMINAMATH_CALUDE_largest_b_no_real_roots_l2540_254076

theorem largest_b_no_real_roots : 
  ∀ b : ℤ, (∀ x : ℝ, x^2 + b*x + 15 ≠ 0) → b ≤ 7 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_b_no_real_roots_l2540_254076


namespace NUMINAMATH_CALUDE_inequality_proof_l2540_254038

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + y * z + z * x = 1) : 
  (27 / 4) * (x + y) * (y + z) * (z + x) ≥ (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x))^2 ∧ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x))^2 ≥ 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2540_254038


namespace NUMINAMATH_CALUDE_x_equals_four_l2540_254029

theorem x_equals_four (a : ℝ) (x y : ℝ) 
  (h1 : a^(x - y) = 343) 
  (h2 : a^(x + y) = 16807) : 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_x_equals_four_l2540_254029


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2540_254093

/-- A right circular cylinder inscribed in a right circular cone --/
structure InscribedCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ
  h_diameter_height : cylinder_radius * 2 = cylinder_radius * 2
  h_cone_cylinder_axes : True  -- This condition is implicit and cannot be directly expressed

/-- The radius of the inscribed cylinder is 90/19 --/
theorem inscribed_cylinder_radius 
  (c : InscribedCylinder) 
  (h_cone_diameter : c.cone_diameter = 18) 
  (h_cone_altitude : c.cone_altitude = 20) : 
  c.cylinder_radius = 90 / 19 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2540_254093


namespace NUMINAMATH_CALUDE_daily_evaporation_rate_l2540_254092

/-- Calculates the daily evaporation rate given initial water amount, time period, and evaporation percentage. -/
theorem daily_evaporation_rate 
  (initial_water : ℝ) 
  (days : ℕ) 
  (evaporation_percentage : ℝ) : 
  initial_water * evaporation_percentage / 100 / days = 0.1 :=
by
  -- Assuming initial_water = 10, days = 20, and evaporation_percentage = 2
  sorry

#check daily_evaporation_rate

end NUMINAMATH_CALUDE_daily_evaporation_rate_l2540_254092


namespace NUMINAMATH_CALUDE_square_eq_nine_solutions_l2540_254012

theorem square_eq_nine_solutions (x : ℝ) : (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_eq_nine_solutions_l2540_254012


namespace NUMINAMATH_CALUDE_water_dumped_calculation_l2540_254082

/-- Calculates the amount of water dumped out of a bathtub given specific conditions --/
theorem water_dumped_calculation (faucet_rate : ℝ) (evaporation_rate : ℝ) (time : ℝ) (water_left : ℝ) : 
  faucet_rate = 40 ∧ 
  evaporation_rate = 200 / 60 ∧ 
  time = 9 * 60 ∧ 
  water_left = 7800 → 
  (faucet_rate * time - evaporation_rate * time - water_left) / 1000 = 12 := by
  sorry


end NUMINAMATH_CALUDE_water_dumped_calculation_l2540_254082


namespace NUMINAMATH_CALUDE_solve_equation_l2540_254042

theorem solve_equation : ∃ r : ℝ, 5 * (r - 9) = 6 * (3 - 3 * r) + 6 ∧ r = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2540_254042


namespace NUMINAMATH_CALUDE_function_growth_l2540_254055

open Real

theorem function_growth (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_growth : ∀ x, (deriv f) x > f x ∧ f x > 0) : 
  f 8 > 2022 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_growth_l2540_254055


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2540_254072

theorem rectangular_prism_diagonal (l w h : ℝ) (hl : l = 10) (hw : w = 20) (hh : h = 10) :
  Real.sqrt (l^2 + w^2 + h^2) = 10 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2540_254072


namespace NUMINAMATH_CALUDE_complex_multiplication_l2540_254053

theorem complex_multiplication (i : ℂ) (h : i * i = -1) : i * (1 + i) = -1 + i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2540_254053


namespace NUMINAMATH_CALUDE_condition_satisfied_pairs_l2540_254027

/-- Checks if a pair of positive integers (m, n) satisfies the given condition -/
def satisfies_condition (m n : ℕ+) : Prop :=
  ∀ x y : ℝ, m ≤ x ∧ x ≤ n ∧ m ≤ y ∧ y ≤ n → m ≤ (5/x + 7/y) ∧ (5/x + 7/y) ≤ n

/-- The only positive integer pairs (m, n) satisfying the condition are (1,12), (2,6), and (3,4) -/
theorem condition_satisfied_pairs :
  ∀ m n : ℕ+, satisfies_condition m n ↔ (m = 1 ∧ n = 12) ∨ (m = 2 ∧ n = 6) ∨ (m = 3 ∧ n = 4) :=
sorry

end NUMINAMATH_CALUDE_condition_satisfied_pairs_l2540_254027


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2540_254061

def f (x : ℝ) : ℝ := x^3

theorem derivative_f_at_zero : 
  deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2540_254061


namespace NUMINAMATH_CALUDE_correlation_coefficient_formula_correlation_coefficient_problem_l2540_254079

/-- Given a linear regression equation ŷ = bx + a, where b is the slope,
    Sy^2 is the variance of y, and Sx^2 is the variance of x,
    prove that the correlation coefficient r = b * (√(Sx^2) / √(Sy^2)) -/
theorem correlation_coefficient_formula 
  (b : ℝ) (Sy_squared : ℝ) (Sx_squared : ℝ) (h1 : Sy_squared > 0) (h2 : Sx_squared > 0) :
  let r := b * (Real.sqrt Sx_squared / Real.sqrt Sy_squared)
  ∀ ε > 0, |r - 0.94| < ε := by
sorry

/-- Given the specific values from the problem, prove that the correlation coefficient is 0.94 -/
theorem correlation_coefficient_problem :
  let b := 4.7
  let Sy_squared := 50
  let Sx_squared := 2
  let r := b * (Real.sqrt Sx_squared / Real.sqrt Sy_squared)
  ∀ ε > 0, |r - 0.94| < ε := by
sorry

end NUMINAMATH_CALUDE_correlation_coefficient_formula_correlation_coefficient_problem_l2540_254079


namespace NUMINAMATH_CALUDE_smallest_winning_number_sum_of_digits_56_l2540_254059

def bernardo_win (N : ℕ) : Prop :=
  N ≤ 999 ∧
  3 * N < 1000 ∧
  3 * N + 100 < 1000 ∧
  9 * N + 300 < 1000 ∧
  9 * N + 400 < 1000 ∧
  27 * N + 1200 ≥ 1000

theorem smallest_winning_number :
  ∀ n : ℕ, n < 56 → ¬(bernardo_win n) ∧ bernardo_win 56 :=
sorry

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_56 :
  sum_of_digits 56 = 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_number_sum_of_digits_56_l2540_254059


namespace NUMINAMATH_CALUDE_fifth_number_ninth_row_is_61_l2540_254056

/-- Represents a lattice pattern with a given number of columns per row -/
structure LatticePattern where
  columnsPerRow : ℕ

/-- Calculates the last number in a given row of the lattice pattern -/
def lastNumberInRow (pattern : LatticePattern) (row : ℕ) : ℕ :=
  pattern.columnsPerRow * row

/-- Calculates the nth number from the end in a given row -/
def nthNumberFromEnd (pattern : LatticePattern) (row : ℕ) (n : ℕ) : ℕ :=
  lastNumberInRow pattern row - (n - 1)

/-- The theorem to be proved -/
theorem fifth_number_ninth_row_is_61 :
  let pattern : LatticePattern := ⟨7⟩
  nthNumberFromEnd pattern 9 5 = 61 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_ninth_row_is_61_l2540_254056


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2540_254058

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 5*x + 6 = -4) → (∃ y : ℝ, y^2 - 5*y + 6 = -4 ∧ x + y = 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2540_254058


namespace NUMINAMATH_CALUDE_profit_function_max_profit_profit_2400_l2540_254078

-- Define the cost price
def cost_price : ℝ := 80

-- Define the sales quantity function
def sales_quantity (x : ℝ) : ℝ := -2 * x + 320

-- Define the valid price range
def valid_price (x : ℝ) : Prop := 80 ≤ x ∧ x ≤ 160

-- Define the daily profit function
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * sales_quantity x

-- Theorem statements
theorem profit_function (x : ℝ) (h : valid_price x) :
  daily_profit x = -2 * x^2 + 480 * x - 25600 := by sorry

theorem max_profit (x : ℝ) (h : valid_price x) :
  daily_profit x ≤ 3200 ∧ daily_profit 120 = 3200 := by sorry

theorem profit_2400 :
  ∃ x, valid_price x ∧ daily_profit x = 2400 ∧
  ∀ y, valid_price y → daily_profit y = 2400 → x ≤ y := by sorry

end NUMINAMATH_CALUDE_profit_function_max_profit_profit_2400_l2540_254078


namespace NUMINAMATH_CALUDE_geometric_sequence_q_value_l2540_254034

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_q_value
  (a : ℕ → ℝ)
  (h_monotone : ∀ n : ℕ, a n ≤ a (n + 1))
  (h_geometric : geometric_sequence a)
  (h_sum : a 3 + a 7 = 5)
  (h_product : a 6 * a 4 = 6) :
  ∃ q : ℝ, q > 1 ∧ q^4 = 3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_q_value_l2540_254034


namespace NUMINAMATH_CALUDE_train_length_calculation_l2540_254065

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed_kmh : ℝ) (cross_time_s : ℝ) :
  speed_kmh = 56 →
  cross_time_s = 9 →
  ∃ (length_m : ℝ), 139 < length_m ∧ length_m < 141 :=
by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l2540_254065


namespace NUMINAMATH_CALUDE_exists_center_nail_pierces_one_cardboard_l2540_254024

/-- A cardboard figure -/
structure Cardboard where
  shape : Set (ℝ × ℝ)

/-- A rectangular box bottom -/
structure Box where
  width : ℝ
  height : ℝ

/-- A configuration of two cardboard pieces on a box bottom -/
structure Configuration where
  box : Box
  piece1 : Cardboard
  piece2 : Cardboard
  position1 : ℝ × ℝ
  position2 : ℝ × ℝ

/-- Predicate to check if a point is covered by a cardboard piece at a given position -/
def covers (c : Cardboard) (pos : ℝ × ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - pos.1, point.2 - pos.2) ∈ c.shape

/-- Predicate to check if a configuration completely covers the box bottom -/
def completelyCovers (config : Configuration) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ config.box.width → 0 ≤ y ∧ y ≤ config.box.height →
    covers config.piece1 config.position1 (x, y) ∨ covers config.piece2 config.position2 (x, y)

/-- Theorem stating that there exists a configuration where the center nail pierces only one cardboard -/
theorem exists_center_nail_pierces_one_cardboard :
  ∃ (config : Configuration), completelyCovers config ∧
    (covers config.piece1 config.position1 (config.box.width / 2, config.box.height / 2) ≠
     covers config.piece2 config.position2 (config.box.width / 2, config.box.height / 2)) :=
by sorry

end NUMINAMATH_CALUDE_exists_center_nail_pierces_one_cardboard_l2540_254024


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l2540_254081

theorem sphere_radius_ratio (V_large V_small : ℝ) (r_large r_small : ℝ) : 
  V_large = 576 * Real.pi ∧ 
  V_small = 0.0625 * V_large ∧
  V_large = (4/3) * Real.pi * r_large^3 ∧
  V_small = (4/3) * Real.pi * r_small^3 →
  r_small / r_large = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l2540_254081


namespace NUMINAMATH_CALUDE_paths_from_A_to_D_l2540_254040

/-- Represents a point in the graph -/
inductive Point
| A
| B
| C
| D

/-- Represents the number of paths between two points -/
def num_paths (start finish : Point) : ℕ :=
  match start, finish with
  | Point.A, Point.B => 2
  | Point.B, Point.C => 2
  | Point.C, Point.D => 2
  | Point.A, Point.C => 1
  | _, _ => 0

/-- The total number of paths from A to D -/
def total_paths : ℕ :=
  (num_paths Point.A Point.B) * (num_paths Point.B Point.C) * (num_paths Point.C Point.D) +
  (num_paths Point.A Point.C) * (num_paths Point.C Point.D)

theorem paths_from_A_to_D : total_paths = 10 := by
  sorry

end NUMINAMATH_CALUDE_paths_from_A_to_D_l2540_254040


namespace NUMINAMATH_CALUDE_equal_sums_exist_l2540_254039

/-- A 3x3 table with entries of 1, 0, or -1 -/
def Table := Fin 3 → Fin 3 → Int

/-- Predicate to check if a table is valid (contains only 1, 0, or -1) -/
def isValidTable (t : Table) : Prop :=
  ∀ i j, t i j = 1 ∨ t i j = 0 ∨ t i j = -1

/-- Sum of a row in the table -/
def rowSum (t : Table) (i : Fin 3) : Int :=
  (t i 0) + (t i 1) + (t i 2)

/-- Sum of a column in the table -/
def colSum (t : Table) (j : Fin 3) : Int :=
  (t 0 j) + (t 1 j) + (t 2 j)

/-- List of all row and column sums -/
def allSums (t : Table) : List Int :=
  (List.range 3).map (rowSum t) ++ (List.range 3).map (colSum t)

/-- Theorem: In a valid 3x3 table, there exist at least two equal sums among row and column sums -/
theorem equal_sums_exist (t : Table) (h : isValidTable t) :
  ∃ (x y : Fin 6), x ≠ y ∧ (allSums t).get x = (allSums t).get y :=
sorry

end NUMINAMATH_CALUDE_equal_sums_exist_l2540_254039


namespace NUMINAMATH_CALUDE_money_distribution_l2540_254007

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (AC_sum : A + C = 200)
  (BC_sum : B + C = 340) :
  C = 40 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l2540_254007


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2540_254054

theorem complex_equation_solution :
  ∀ z : ℂ, (Complex.I / (z + Complex.I) = 2 - Complex.I) → z = (-1/5 : ℂ) - (3/5 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2540_254054


namespace NUMINAMATH_CALUDE_derivative_sin_cos_product_l2540_254090

theorem derivative_sin_cos_product (x : ℝ) :
  deriv (fun x => 2 * Real.sin x * Real.cos x) x = 2 * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_product_l2540_254090


namespace NUMINAMATH_CALUDE_min_value_of_w_l2540_254098

theorem min_value_of_w (x y : ℝ) :
  let w := 3 * x^2 + 5 * y^2 + 8 * x - 10 * y + 34
  w ≥ 71 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_w_l2540_254098


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2005_l2540_254066

/-- 
Given an arithmetic sequence with first term a₁ = -1 and common difference d = 2,
prove that the 1004th term is equal to 2005.
-/
theorem arithmetic_sequence_2005 :
  let a : ℕ → ℤ := λ n => -1 + (n - 1) * 2
  a 1004 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2005_l2540_254066


namespace NUMINAMATH_CALUDE_square_gt_iff_abs_gt_l2540_254032

theorem square_gt_iff_abs_gt (a b : ℝ) : a^2 > b^2 ↔ |a| > |b| := by sorry

end NUMINAMATH_CALUDE_square_gt_iff_abs_gt_l2540_254032


namespace NUMINAMATH_CALUDE_power_of_two_equality_l2540_254013

theorem power_of_two_equality (x : ℕ) : (1 / 8 : ℝ) * 2^33 = 2^x → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l2540_254013


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2540_254025

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 4 = 12 → a 1 + a 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2540_254025


namespace NUMINAMATH_CALUDE_x_days_to_complete_work_l2540_254023

/-- The number of days required for x and y to complete the work together -/
def days_xy : ℚ := 12

/-- The number of days required for y to complete the work alone -/
def days_y : ℚ := 24

/-- The fraction of work completed by a worker in one day -/
def work_per_day (days : ℚ) : ℚ := 1 / days

theorem x_days_to_complete_work : 
  1 / (work_per_day days_xy - work_per_day days_y) = 24 := by sorry

end NUMINAMATH_CALUDE_x_days_to_complete_work_l2540_254023


namespace NUMINAMATH_CALUDE_bruce_total_payment_l2540_254096

/-- Calculates the total amount Bruce paid for fruits -/
def total_amount_paid (grape_quantity grape_price mango_quantity mango_price
                       orange_quantity orange_price apple_quantity apple_price : ℕ) : ℕ :=
  grape_quantity * grape_price + mango_quantity * mango_price +
  orange_quantity * orange_price + apple_quantity * apple_price

/-- Theorem: Bruce paid $1480 for the fruits -/
theorem bruce_total_payment :
  total_amount_paid 9 70 7 55 5 45 3 80 = 1480 := by
  sorry

end NUMINAMATH_CALUDE_bruce_total_payment_l2540_254096


namespace NUMINAMATH_CALUDE_inequality_system_no_solution_l2540_254045

theorem inequality_system_no_solution (a : ℝ) :
  (∀ x : ℝ, ¬(x > a + 2 ∧ x < 3*a - 2)) ↔ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_no_solution_l2540_254045


namespace NUMINAMATH_CALUDE_vitamin_c_in_two_juices_l2540_254043

/-- Amount of vitamin C (in mg) in one 8-oz glass of apple juice -/
def apple_juice_vc : ℝ := 103

/-- Amount of vitamin C (in mg) in one 8-oz glass of orange juice -/
def orange_juice_vc : ℝ := 82

/-- Total amount of vitamin C (in mg) in two glasses of apple juice and three glasses of orange juice -/
def total_vc_five_glasses : ℝ := 452

/-- Theorem stating that one glass each of apple and orange juice contain 185 mg of vitamin C -/
theorem vitamin_c_in_two_juices :
  apple_juice_vc + orange_juice_vc = 185 ∧
  2 * apple_juice_vc + 3 * orange_juice_vc = total_vc_five_glasses :=
sorry

end NUMINAMATH_CALUDE_vitamin_c_in_two_juices_l2540_254043


namespace NUMINAMATH_CALUDE_same_color_probability_l2540_254035

/-- The probability of drawing two balls of the same color from a box containing
    4 white balls and 2 black balls when drawing two balls at once. -/
theorem same_color_probability (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ)
    (h_total : total_balls = white_balls + black_balls)
    (h_white : white_balls = 4)
    (h_black : black_balls = 2) :
    (Nat.choose white_balls 2 + Nat.choose black_balls 2) / Nat.choose total_balls 2 = 7 / 15 :=
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2540_254035


namespace NUMINAMATH_CALUDE_largest_dividend_l2540_254087

theorem largest_dividend (dividend quotient divisor remainder : ℕ) : 
  dividend = quotient * divisor + remainder →
  remainder < divisor →
  quotient = 32 →
  divisor = 18 →
  dividend ≤ 593 := by
sorry

end NUMINAMATH_CALUDE_largest_dividend_l2540_254087


namespace NUMINAMATH_CALUDE_soccer_team_starters_l2540_254094

theorem soccer_team_starters (n : ℕ) (q : ℕ) (s : ℕ) (qc : ℕ) :
  n = 15 →  -- Total number of players
  q = 4 →   -- Number of quadruplets
  s = 7 →   -- Number of starters
  qc = 2 →  -- Number of quadruplets in starting lineup
  (Nat.choose q qc) * (Nat.choose (n - q) (s - qc)) = 2772 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_starters_l2540_254094


namespace NUMINAMATH_CALUDE_three_divides_difference_l2540_254088

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Reverses a three-digit number -/
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber :=
  { hundreds := n.ones
  , tens := n.tens
  , ones := n.hundreds
  , is_valid := by sorry }

/-- Converts a ThreeDigitNumber to a natural number -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The difference between a number and its reverse -/
def difference (n : ThreeDigitNumber) : Int :=
  Int.natAbs (to_nat n - to_nat (reverse n))

theorem three_divides_difference (n : ThreeDigitNumber) (h : n.hundreds ≠ n.ones) :
  3 ∣ difference n := by
  sorry

end NUMINAMATH_CALUDE_three_divides_difference_l2540_254088


namespace NUMINAMATH_CALUDE_distance_to_office_l2540_254074

theorem distance_to_office : 
  ∀ (v : ℝ) (d : ℝ),
  (d = v * (1/2)) →  -- Distance in heavy traffic
  (d = (v + 20) * (1/5)) →  -- Distance without traffic
  d = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_office_l2540_254074


namespace NUMINAMATH_CALUDE_smoothie_combinations_l2540_254026

theorem smoothie_combinations (n_flavors : ℕ) (n_supplements : ℕ) : 
  n_flavors = 5 → n_supplements = 8 → n_flavors * (n_supplements.choose 3) = 280 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_combinations_l2540_254026


namespace NUMINAMATH_CALUDE_fraction_sum_mixed_number_equality_main_theorem_l2540_254003

theorem fraction_sum : (3 : ℚ) / 4 + (5 : ℚ) / 6 + (4 : ℚ) / 3 = (35 : ℚ) / 12 := by
  sorry

theorem mixed_number_equality : (35 : ℚ) / 12 = 2 + (11 : ℚ) / 12 := by
  sorry

theorem main_theorem : (3 : ℚ) / 4 + (5 : ℚ) / 6 + (1 + (1 : ℚ) / 3) = 2 + (11 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_mixed_number_equality_main_theorem_l2540_254003


namespace NUMINAMATH_CALUDE_probability_rain_all_three_days_l2540_254050

theorem probability_rain_all_three_days 
  (prob_friday : ℝ) 
  (prob_saturday : ℝ) 
  (prob_sunday : ℝ) 
  (h1 : prob_friday = 0.4)
  (h2 : prob_saturday = 0.5)
  (h3 : prob_sunday = 0.3)
  (h4 : 0 ≤ prob_friday ∧ prob_friday ≤ 1)
  (h5 : 0 ≤ prob_saturday ∧ prob_saturday ≤ 1)
  (h6 : 0 ≤ prob_sunday ∧ prob_sunday ≤ 1) :
  prob_friday * prob_saturday * prob_sunday = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_probability_rain_all_three_days_l2540_254050


namespace NUMINAMATH_CALUDE_average_ducks_is_35_l2540_254036

/-- The average number of ducks bought by three students -/
def averageDucks (adelaide ephraim kolton : ℕ) : ℚ :=
  (adelaide + ephraim + kolton : ℚ) / 3

/-- Theorem: The average number of ducks bought is 35 -/
theorem average_ducks_is_35 :
  let adelaide := 30
  let ephraim := adelaide / 2
  let kolton := ephraim + 45
  averageDucks adelaide ephraim kolton = 35 := by
sorry

end NUMINAMATH_CALUDE_average_ducks_is_35_l2540_254036


namespace NUMINAMATH_CALUDE_cone_max_cross_section_area_l2540_254031

/-- Given a cone with lateral surface formed by a sector of radius 1 and central angle 3/2 π,
    the maximum area of a cross-section passing through the vertex is 1/2. -/
theorem cone_max_cross_section_area (r : ℝ) (θ : ℝ) (h : ℝ) : 
  r = 1 → θ = 3/2 * Real.pi → h = Real.sqrt (r^2 - (r * θ / (2 * Real.pi))^2) → 
  (1/2 : ℝ) * (r * θ / (2 * Real.pi)) * h ≤ 1/2 := by
  sorry

#check cone_max_cross_section_area

end NUMINAMATH_CALUDE_cone_max_cross_section_area_l2540_254031


namespace NUMINAMATH_CALUDE_f_equals_F_l2540_254019

/-- The function f(x) = 3x^4 - x^3 -/
def f (x : ℝ) : ℝ := 3 * x^4 - x^3

/-- The function F(x) = x(3x^3 - 1) -/
def F (x : ℝ) : ℝ := x * (3 * x^3 - 1)

/-- Theorem stating that f and F are the same function -/
theorem f_equals_F : f = F := by sorry

end NUMINAMATH_CALUDE_f_equals_F_l2540_254019


namespace NUMINAMATH_CALUDE_g_four_equals_thirteen_l2540_254011

-- Define the function g
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^6 + b * x^4 + c * x^2 + 7

-- State the theorem
theorem g_four_equals_thirteen 
  (a b c : ℝ) 
  (h : g a b c (-4) = 13) : 
  g a b c 4 = 13 := by
sorry

end NUMINAMATH_CALUDE_g_four_equals_thirteen_l2540_254011


namespace NUMINAMATH_CALUDE_bug_probability_after_8_meters_l2540_254064

/-- Probability of the bug being at vertex A after n meters -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/3 * (1 - P n)

/-- The probability of the bug being at vertex A after 8 meters is 547/2187 -/
theorem bug_probability_after_8_meters : P 8 = 547/2187 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_after_8_meters_l2540_254064


namespace NUMINAMATH_CALUDE_five_segments_max_regions_l2540_254095

/-- The maximum number of regions formed by n line segments in a plane -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of regions formed by 5 line segments in a plane is 16 -/
theorem five_segments_max_regions : max_regions 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_segments_max_regions_l2540_254095


namespace NUMINAMATH_CALUDE_fish_value_in_rice_l2540_254084

-- Define the trade ratios
def fish_to_bread : ℚ := 4 / 5
def bread_to_rice : ℚ := 6
def fish_to_rice : ℚ := 8 / 3

-- Theorem to prove
theorem fish_value_in_rice : fish_to_rice = 8 / 3 := by
  sorry

#eval fish_to_rice

end NUMINAMATH_CALUDE_fish_value_in_rice_l2540_254084


namespace NUMINAMATH_CALUDE_film_casting_theorem_l2540_254005

theorem film_casting_theorem 
  (n : ℕ) 
  (a : Fin n → ℕ) 
  (p : ℕ) 
  (k : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_p_ge : ∀ i : Fin n, p ≥ max (a i) n) 
  (h_k_le_n : k ≤ n) : 
  ∃ (castings : Fin (p^k) → (Fin n → Fin p)),
    ∀ (roles : Fin k → Fin n) (people : Fin k → ℕ),
      (∀ i : Fin k, people i < a (roles i)) →
      (∀ i j : Fin k, i ≠ j → roles i ≠ roles j) →
      ∃ day : Fin (p^k), ∀ i : Fin k, castings day (roles i) = people i :=
sorry

end NUMINAMATH_CALUDE_film_casting_theorem_l2540_254005


namespace NUMINAMATH_CALUDE_red_paint_amount_l2540_254044

/-- Given a paint mixture with a ratio of red:green:white as 4:3:5,
    and using 15 quarts of white paint, prove that the amount of
    red paint required is 12 quarts. -/
theorem red_paint_amount (red green white : ℚ) : 
  red / white = 4 / 5 →
  green / white = 3 / 5 →
  white = 15 →
  red = 12 := by
  sorry

end NUMINAMATH_CALUDE_red_paint_amount_l2540_254044


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2540_254017

theorem linear_equation_solution (m : ℝ) : 
  (∃ x : ℝ, 2 * x + m = 5 ∧ x = 1) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2540_254017


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l2540_254002

theorem isosceles_triangle_largest_angle (a b c : ℝ) : 
  -- The triangle is isosceles
  a = b →
  -- One of the angles opposite an equal side is 50°
  c = 50 →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 80°
  max a (max b c) = 80 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l2540_254002


namespace NUMINAMATH_CALUDE_childrens_ticket_cost_l2540_254073

/-- Given ticket information, prove the cost of a children's ticket -/
theorem childrens_ticket_cost 
  (adult_ticket_cost : ℝ) 
  (total_tickets : ℕ) 
  (total_cost : ℝ) 
  (childrens_tickets : ℕ) 
  (h1 : adult_ticket_cost = 5.50)
  (h2 : total_tickets = 21)
  (h3 : total_cost = 83.50)
  (h4 : childrens_tickets = 16) :
  ∃ (childrens_ticket_cost : ℝ),
    childrens_ticket_cost * childrens_tickets + 
    adult_ticket_cost * (total_tickets - childrens_tickets) = total_cost ∧ 
    childrens_ticket_cost = 3.50 :=
by sorry

end NUMINAMATH_CALUDE_childrens_ticket_cost_l2540_254073


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2540_254020

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2540_254020


namespace NUMINAMATH_CALUDE_nellys_friends_l2540_254030

def pizza_cost : ℕ := 12
def people_per_pizza : ℕ := 3
def babysitting_pay : ℕ := 4
def nights_babysitting : ℕ := 15

def total_earned : ℕ := babysitting_pay * nights_babysitting
def pizzas_bought : ℕ := total_earned / pizza_cost
def total_people_fed : ℕ := pizzas_bought * people_per_pizza

theorem nellys_friends (nelly : ℕ := 1) : 
  total_people_fed - nelly = 14 := by sorry

end NUMINAMATH_CALUDE_nellys_friends_l2540_254030


namespace NUMINAMATH_CALUDE_shopping_remaining_amount_l2540_254008

theorem shopping_remaining_amount (initial_amount : ℝ) (spent_percentage : ℝ) 
  (h1 : initial_amount = 5000)
  (h2 : spent_percentage = 0.30) : 
  initial_amount - (spent_percentage * initial_amount) = 3500 := by
  sorry

end NUMINAMATH_CALUDE_shopping_remaining_amount_l2540_254008


namespace NUMINAMATH_CALUDE_two_digit_three_digit_sum_l2540_254069

theorem two_digit_three_digit_sum : ∃! (x y : ℕ), 
  10 ≤ x ∧ x < 100 ∧ 100 ≤ y ∧ y < 1000 ∧ 
  1000 * x + y = 11 * x * y ∧ 
  x + y = 919 := by sorry

end NUMINAMATH_CALUDE_two_digit_three_digit_sum_l2540_254069


namespace NUMINAMATH_CALUDE_range_of_a_l2540_254060

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∪ B a = B a → -1 < a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2540_254060


namespace NUMINAMATH_CALUDE_parallel_perpendicular_to_plane_l2540_254004

/-- Two lines are parallel -/
def parallel (a b : Line3) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line3) (p : Plane3) : Prop := sorry

/-- The theorem statement -/
theorem parallel_perpendicular_to_plane 
  (a b : Line3) (α : Plane3) 
  (h1 : parallel a b) 
  (h2 : perpendicular_to_plane a α) : 
  perpendicular_to_plane b α := by sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_to_plane_l2540_254004


namespace NUMINAMATH_CALUDE_function_value_at_minus_ten_l2540_254006

/-- Given a function f(x) = (x-6)/(x+2), prove that f(-10) = 2 -/
theorem function_value_at_minus_ten :
  let f : ℝ → ℝ := λ x ↦ (x - 6) / (x + 2)
  f (-10) = 2 := by sorry

end NUMINAMATH_CALUDE_function_value_at_minus_ten_l2540_254006


namespace NUMINAMATH_CALUDE_frontal_view_correct_l2540_254021

/-- Represents a column of stacked cubes -/
def Column := List Nat

/-- Calculates the maximum height of a column -/
def maxHeight (col : Column) : Nat :=
  col.foldl max 0

/-- Represents the arrangement of cube stacks -/
structure CubeArrangement where
  col1 : Column
  col2 : Column
  col3 : Column

/-- Calculates the frontal view heights of a cube arrangement -/
def frontalView (arr : CubeArrangement) : List Nat :=
  [maxHeight arr.col1, maxHeight arr.col2, maxHeight arr.col3]

/-- The specific cube arrangement described in the problem -/
def problemArrangement : CubeArrangement :=
  { col1 := [4, 2]
    col2 := [3, 0, 3]
    col3 := [1, 5] }

theorem frontal_view_correct :
  frontalView problemArrangement = [4, 3, 5] := by sorry

end NUMINAMATH_CALUDE_frontal_view_correct_l2540_254021


namespace NUMINAMATH_CALUDE_m_range_l2540_254022

-- Define propositions p and q
def p (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0
def q (x : ℝ) : Prop := |x - 3| ≤ 1

-- Define the condition that q is sufficient but not necessary for p
def q_sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, q x → p x m) ∧ (∃ x, p x m ∧ ¬q x)

-- Main theorem
theorem m_range (m : ℝ) (h1 : m > 0) (h2 : q_sufficient_not_necessary m) :
  m > 4/3 ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2540_254022


namespace NUMINAMATH_CALUDE_unique_b_values_l2540_254000

theorem unique_b_values : ∃! (b₂ b₃ b₄ b₅ b₆ : ℕ),
  (11 : ℚ) / 15 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_b_values_l2540_254000


namespace NUMINAMATH_CALUDE_scale_E_accurate_l2540_254014

/-- Represents the weight measured by a scale -/
structure Scale where
  weight : ℝ

/-- Represents a set of five scales used in a health check center -/
structure HealthCheckScales where
  A : Scale
  B : Scale
  C : Scale
  D : Scale
  E : Scale

/-- The conditions of the health check scales problem -/
def ScaleConditions (s : HealthCheckScales) : Prop :=
  s.C.weight = s.B.weight - 0.3 ∧
  s.D.weight = s.C.weight - 0.1 ∧
  s.E.weight = s.A.weight - 0.1 ∧
  s.C.weight = s.E.weight - 0.1

/-- The average weight of all scales is accurate -/
def AverageWeightAccurate (s : HealthCheckScales) (actualWeight : ℝ) : Prop :=
  (s.A.weight + s.B.weight + s.C.weight + s.D.weight + s.E.weight) / 5 = actualWeight

/-- Theorem stating that scale E is accurate given the conditions -/
theorem scale_E_accurate (s : HealthCheckScales) (actualWeight : ℝ)
  (h1 : ScaleConditions s)
  (h2 : AverageWeightAccurate s actualWeight) :
  s.E.weight = actualWeight :=
sorry

end NUMINAMATH_CALUDE_scale_E_accurate_l2540_254014


namespace NUMINAMATH_CALUDE_star_value_l2540_254051

-- Define the sequence type
def Sequence := Fin 12 → ℕ

-- Define the property that the sum of any four adjacent numbers is 11
def SumProperty (s : Sequence) : Prop :=
  ∀ i : Fin 9, s i + s (i + 1) + s (i + 2) + s (i + 3) = 11

-- Define the repeating pattern property
def PatternProperty (s : Sequence) : Prop :=
  ∀ i : Fin 3, 
    s (4 * i) = 2 ∧ 
    s (4 * i + 1) = 0 ∧ 
    s (4 * i + 2) = 1

-- Main theorem
theorem star_value (s : Sequence) 
  (h1 : SumProperty s) 
  (h2 : PatternProperty s) : 
  ∀ i : Fin 3, s (4 * i + 3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l2540_254051


namespace NUMINAMATH_CALUDE_venus_speed_mph_l2540_254041

/-- The speed of Venus in miles per second -/
def venus_speed_mps : ℝ := 21.9

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: Venus's speed in miles per hour -/
theorem venus_speed_mph : ⌊venus_speed_mps * seconds_per_hour⌋ = 78840 := by
  sorry

end NUMINAMATH_CALUDE_venus_speed_mph_l2540_254041


namespace NUMINAMATH_CALUDE_max_distance_to_complex_point_l2540_254001

open Complex

theorem max_distance_to_complex_point (z : ℂ) :
  let z₁ : ℂ := 2 - 2*I
  (abs z = 1) →
  (∀ w : ℂ, abs w = 1 → abs (w - z₁) ≤ 2*Real.sqrt 2 + 1) ∧
  (∃ w : ℂ, abs w = 1 ∧ abs (w - z₁) = 2*Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_complex_point_l2540_254001


namespace NUMINAMATH_CALUDE_largest_number_hcf_lcm_l2540_254009

theorem largest_number_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 42) 
  (h2 : Nat.lcm a b = 42 * 10 * 20) : max a b = 840 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_hcf_lcm_l2540_254009
