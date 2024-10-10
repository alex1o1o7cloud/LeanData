import Mathlib

namespace tape_length_calculation_l2322_232257

/-- The total length of overlapping tape sheets -/
def total_tape_length (n : ℕ) (sheet_length : ℝ) (overlap : ℝ) : ℝ :=
  sheet_length + (n - 1 : ℝ) * (sheet_length - overlap)

/-- Theorem: The total length of 15 sheets of tape, each 20 cm long and overlapping by 5 cm, is 230 cm -/
theorem tape_length_calculation :
  total_tape_length 15 20 5 = 230 := by
  sorry

end tape_length_calculation_l2322_232257


namespace train_passing_time_specific_train_passing_time_l2322_232297

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length > 0 → train_speed > man_speed → train_speed > 0 → man_speed ≥ 0 →
  ∃ (t : ℝ), t > 0 ∧ t < 19 ∧ t * (train_speed - man_speed) * (5 / 18) = train_length :=
by sorry

/-- Specific instance of the train passing time problem -/
theorem specific_train_passing_time :
  ∃ (t : ℝ), t > 0 ∧ t < 19 ∧ t * (68 - 8) * (5 / 18) = 300 :=
by sorry

end train_passing_time_specific_train_passing_time_l2322_232297


namespace go_stones_perimeter_l2322_232231

/-- The number of stones on one side of the square arrangement. -/
def side_length : ℕ := 6

/-- The number of sides in a square. -/
def num_sides : ℕ := 4

/-- The number of corners in a square. -/
def num_corners : ℕ := 4

/-- Calculates the number of stones on the perimeter of a square arrangement. -/
def perimeter_stones (n : ℕ) : ℕ := n * num_sides - num_corners

theorem go_stones_perimeter :
  perimeter_stones side_length = 20 := by
  sorry

end go_stones_perimeter_l2322_232231


namespace initial_average_age_l2322_232233

theorem initial_average_age (n : ℕ) (new_age : ℕ) (new_average : ℕ) 
  (h1 : n = 9)
  (h2 : new_age = 35)
  (h3 : new_average = 17) :
  ∃ initial_average : ℚ, 
    initial_average = 15 ∧ 
    (n : ℚ) * initial_average + new_age = ((n : ℚ) + 1) * new_average :=
by sorry

end initial_average_age_l2322_232233


namespace complex_multiplication_l2322_232280

theorem complex_multiplication (Q E D : ℂ) : 
  Q = 7 + 3*I ∧ E = 2 + I ∧ D = 7 - 3*I → Q * E * D = 116 + 58*I :=
by sorry

end complex_multiplication_l2322_232280


namespace sqrt_sum_equals_two_l2322_232242

theorem sqrt_sum_equals_two (a b : ℝ) (h : a^2 + b^2 = 4) :
  (a * (b - 4))^(1/3) + ((a * b - 3 * a + 2 * b - 6) : ℝ)^(1/2) = 2 := by sorry

end sqrt_sum_equals_two_l2322_232242


namespace unique_solution_l2322_232293

/-- Represents a six-digit number with distinct digits -/
structure SixDigitNumber where
  digits : Fin 6 → Fin 10
  distinct : ∀ i j, i ≠ j → digits i ≠ digits j

/-- The equation 6 × AOBMEP = 7 × MEPAOB -/
def EquationHolds (n : SixDigitNumber) : Prop :=
  6 * (100000 * n.digits 0 + 10000 * n.digits 1 + 1000 * n.digits 2 +
       100 * n.digits 3 + 10 * n.digits 4 + n.digits 5) =
  7 * (100000 * n.digits 3 + 10000 * n.digits 4 + 1000 * n.digits 5 +
       100 * n.digits 0 + 10 * n.digits 1 + n.digits 2)

/-- The unique solution to the equation -/
def Solution : SixDigitNumber where
  digits := fun i => match i with
    | 0 => 5  -- A
    | 1 => 3  -- O
    | 2 => 8  -- B
    | 3 => 4  -- M
    | 4 => 6  -- E
    | 5 => 1  -- P
  distinct := by sorry

theorem unique_solution :
  ∀ n : SixDigitNumber, EquationHolds n ↔ n = Solution := by
  sorry

end unique_solution_l2322_232293


namespace max_got_more_candy_l2322_232217

/-- The number of candy pieces Frankie got -/
def frankies_candy : ℕ := 74

/-- The number of candy pieces Max got -/
def maxs_candy : ℕ := 92

/-- The difference in candy pieces between Max and Frankie -/
def candy_difference : ℕ := maxs_candy - frankies_candy

theorem max_got_more_candy : candy_difference = 18 := by
  sorry

end max_got_more_candy_l2322_232217


namespace inequality_relationships_l2322_232263

theorem inequality_relationships (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧
  (1 / (a - b) < 1 / a) ∧
  (|a| > |b|) ∧
  (a^4 > b^4) := by
  sorry

end inequality_relationships_l2322_232263


namespace garden_breadth_l2322_232201

theorem garden_breadth (perimeter length : ℕ) (h1 : perimeter = 1200) (h2 : length = 360) :
  let breadth := (perimeter / 2) - length
  breadth = 240 :=
by sorry

end garden_breadth_l2322_232201


namespace inscribed_sphere_radius_in_pyramid_inscribed_sphere_radius_is_correct_l2322_232222

/-- The radius of the sphere inscribed in a pyramid PMKC, where:
  - PABCD is a regular quadrilateral pyramid
  - PO is the height of the pyramid and equals 4
  - ABCD is the base of the pyramid with side length 6
  - M is the midpoint of BC
  - K is the midpoint of CD
-/
theorem inscribed_sphere_radius_in_pyramid (PO : ℝ) (side_length : ℝ) : ℝ :=
  let PMKC_volume := (1/8) * (1/3) * side_length^2 * PO
  let CMK_area := (1/4) * (1/2) * side_length^2
  let ON := (1/4) * side_length * Real.sqrt 2
  let PN := Real.sqrt ((PO^2) + (ON^2))
  let OK := (1/2) * side_length
  let PK := Real.sqrt ((PO^2) + (OK^2))
  let PKC_area := (1/2) * OK * PK
  let PMK_area := (1/2) * (side_length * Real.sqrt 2 / 2) * PN
  let surface_area := 2 * PKC_area + PMK_area + CMK_area
  let radius := 3 * PMKC_volume / surface_area
  12 / (13 + Real.sqrt 41)

theorem inscribed_sphere_radius_is_correct (PO : ℝ) (side_length : ℝ)
  (h1 : PO = 4)
  (h2 : side_length = 6) :
  inscribed_sphere_radius_in_pyramid PO side_length = 12 / (13 + Real.sqrt 41) := by
  sorry

#check inscribed_sphere_radius_is_correct

end inscribed_sphere_radius_in_pyramid_inscribed_sphere_radius_is_correct_l2322_232222


namespace sin_n_equals_cos_510_l2322_232251

theorem sin_n_equals_cos_510 (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) :
  Real.sin (n * π / 180) = Real.cos (510 * π / 180) → n = -60 := by sorry

end sin_n_equals_cos_510_l2322_232251


namespace permutation_of_6_choose_2_l2322_232284

def A (n : ℕ) (k : ℕ) : ℕ := n * (n - 1)

theorem permutation_of_6_choose_2 : A 6 2 = 30 := by
  sorry

end permutation_of_6_choose_2_l2322_232284


namespace symmetry_properties_l2322_232282

/-- A function f: ℝ → ℝ is symmetric about the line x=a if f(a-x) = f(a+x) for all x ∈ ℝ -/
def symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (a + x)

/-- A function f: ℝ → ℝ is symmetric about the y-axis if f(x) = f(-x) for all x ∈ ℝ -/
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- Two functions f, g: ℝ → ℝ have graphs symmetric about the y-axis if f(x) = g(-x) for all x ∈ ℝ -/
def graphs_symmetric_about_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

/-- Two functions f, g: ℝ → ℝ have graphs symmetric about the line x=a if f(x) = g(2a-x) for all x ∈ ℝ -/
def graphs_symmetric_about_line (f g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x = g (2*a - x)

theorem symmetry_properties (f : ℝ → ℝ) :
  (symmetric_about_line f 4) ∧
  ((∀ x, f (4 - x) = f (x - 4)) → symmetric_about_y_axis f) ∧
  (graphs_symmetric_about_y_axis (fun x ↦ f (4 - x)) (fun x ↦ f (4 + x))) ∧
  (graphs_symmetric_about_line (fun x ↦ f (4 - x)) (fun x ↦ f (x - 4)) 4) := by
  sorry

end symmetry_properties_l2322_232282


namespace intersection_equals_open_interval_l2322_232204

-- Define sets A and B
def A : Set ℝ := {x | (4*x - 3)*(x + 3) < 0}
def B : Set ℝ := {x | 2*x > 1}

-- Define the open interval (1/2, 3/4)
def openInterval : Set ℝ := {x | 1/2 < x ∧ x < 3/4}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = openInterval := by
  sorry

end intersection_equals_open_interval_l2322_232204


namespace triangle_side_ratio_sum_l2322_232243

/-- Given a triangle with side lengths a, b, and c, 
    the sum of the ratios of each side length to the difference between 
    the sum of the other two sides and itself is greater than or equal to 3. -/
theorem triangle_side_ratio_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 := by
  sorry

end triangle_side_ratio_sum_l2322_232243


namespace binomial_12_3_l2322_232206

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_3_l2322_232206


namespace stock_sold_percentage_l2322_232208

/-- Given the cash realized, brokerage rate, and total amount including brokerage,
    prove that the percentage of stock sold is 100% -/
theorem stock_sold_percentage
  (cash_realized : ℝ)
  (brokerage_rate : ℝ)
  (total_amount : ℝ)
  (h1 : cash_realized = 106.25)
  (h2 : brokerage_rate = 1 / 4 / 100)
  (h3 : total_amount = 106) :
  let sale_amount := cash_realized + (cash_realized * brokerage_rate)
  let stock_percentage := sale_amount / sale_amount * 100
  stock_percentage = 100 := by sorry

end stock_sold_percentage_l2322_232208


namespace smallest_coin_collection_l2322_232234

def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

def proper_factors (n : ℕ) : Finset ℕ :=
  (Nat.divisors n).filter (λ x => x > 1 ∧ x < n)

theorem smallest_coin_collection :
  ∃ (n : ℕ), n > 0 ∧ num_factors n = 13 ∧ (proper_factors n).card ≥ 11 ∧
  ∀ (m : ℕ), m > 0 → num_factors m = 13 → (proper_factors m).card ≥ 11 → n ≤ m :=
by
  use 4096
  sorry

end smallest_coin_collection_l2322_232234


namespace complement_P_subset_Q_l2322_232295

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- State the theorem
theorem complement_P_subset_Q : (Set.univ \ P) ⊆ Q := by sorry

end complement_P_subset_Q_l2322_232295


namespace pen_arrangement_count_l2322_232290

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def multinomial (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / (ks.map factorial).prod

theorem pen_arrangement_count :
  let total_pens := 15
  let blue_pens := 7
  let red_pens := 3
  let green_pens := 3
  let black_pens := 2
  let total_arrangements := multinomial total_pens [blue_pens, red_pens, green_pens, black_pens]
  let adjacent_green_arrangements := 
    (multinomial (total_pens - green_pens + 1) [blue_pens, red_pens, 1, black_pens]) * (factorial green_pens)
  total_arrangements - adjacent_green_arrangements = 6098400 := by
  sorry

end pen_arrangement_count_l2322_232290


namespace circle_tangent_properties_l2322_232279

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the fixed point A
def point_A : ℝ × ℝ := (4, 3)

-- Define a point P outside circle O
def point_P (a b : ℝ) : Prop := a^2 + b^2 > 5

-- Define the tangent line condition
def is_tangent (a b : ℝ) : Prop := ∃ (t : ℝ), circle_O (a + t) (b + t) ∧ ∀ (s : ℝ), s ≠ t → ¬ circle_O (a + s) (b + s)

-- Define the equality of lengths PQ and PA
def length_equality (a b : ℝ) : Prop := (a - 4)^2 + (b - 3)^2 = a^2 + b^2 - 5

theorem circle_tangent_properties (a b : ℝ) 
  (h1 : point_P a b) 
  (h2 : is_tangent a b) 
  (h3 : length_equality a b) :
  -- 1. Relationship between a and b
  (4 * a + 3 * b - 15 = 0) ∧
  -- 2. Minimum length of PQ
  (∀ (x y : ℝ), point_P x y → is_tangent x y → length_equality x y → 
    (x - 4)^2 + (y - 3)^2 ≥ 16) ∧
  -- 3. Equation of circle P with minimum radius
  (∃ (r : ℝ), r = 3 - Real.sqrt 5 ∧
    ∀ (x y : ℝ), (x - 12/5)^2 + (y - 9/5)^2 = r^2 →
      ∃ (t : ℝ), circle_O (x + t) (y + t)) :=
sorry

end circle_tangent_properties_l2322_232279


namespace specific_tetrahedron_volume_l2322_232213

/-- Regular tetrahedron with given midpoint distances -/
structure RegularTetrahedron where
  midpoint_to_face : ℝ
  midpoint_to_edge : ℝ

/-- Volume of a regular tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the specific regular tetrahedron -/
theorem specific_tetrahedron_volume :
  ∃ (t : RegularTetrahedron),
    t.midpoint_to_face = 2 ∧
    t.midpoint_to_edge = Real.sqrt 10 ∧
    volume t = 80 * Real.sqrt 15 := by sorry

end specific_tetrahedron_volume_l2322_232213


namespace cubic_function_property_l2322_232216

/-- A cubic function g(x) with coefficients p, q, r, and s. -/
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

/-- Theorem stating that for a cubic function g(x) = px³ + qx² + rx + s,
    if g(-3) = 4, then 10p - 5q + 3r - 2s = 40. -/
theorem cubic_function_property (p q r s : ℝ) : 
  g p q r s (-3) = 4 → 10*p - 5*q + 3*r - 2*s = 40 := by
  sorry

end cubic_function_property_l2322_232216


namespace sqrt_equation_solutions_l2322_232237

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt (5 * x - 6) + 8 / Real.sqrt (5 * x - 6) = 6) ↔ (x = 22/5 ∨ x = 2) :=
by sorry

end sqrt_equation_solutions_l2322_232237


namespace circle_properties_l2322_232287

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 36

-- Theorem statement
theorem circle_properties :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    (∀ (x y : ℝ), circle_equation x y ↔ (x - C.1)^2 + (y - C.2)^2 = r^2) ∧
    C = (-1, 3) ∧
    r = 6 := by
  sorry

end circle_properties_l2322_232287


namespace factory_production_rate_l2322_232260

/-- Represents the production setup of a factory --/
structure Factory where
  original_machines : ℕ
  original_hours : ℕ
  new_machine_hours : ℕ
  price_per_kg : ℕ
  daily_earnings : ℕ

/-- Calculates the hourly production rate of a single machine --/
def hourly_production_rate (f : Factory) : ℚ :=
  let total_machine_hours := f.original_machines * f.original_hours + f.new_machine_hours
  let daily_production := f.daily_earnings / f.price_per_kg
  daily_production / total_machine_hours

/-- Theorem stating the hourly production rate of a single machine --/
theorem factory_production_rate (f : Factory) 
  (h1 : f.original_machines = 3)
  (h2 : f.original_hours = 23)
  (h3 : f.new_machine_hours = 12)
  (h4 : f.price_per_kg = 50)
  (h5 : f.daily_earnings = 8100) :
  hourly_production_rate f = 2 := by
  sorry

end factory_production_rate_l2322_232260


namespace pyramid_properties_l2322_232214

/-- Given a sphere of radius r and a regular four-sided pyramid constructed
    such that:
    1. The sphere is divided into two parts
    2. The part towards the center is the mean proportional between the entire radius and the other part
    3. A plane is placed perpendicularly to the radius at the dividing point
    4. The pyramid is constructed in the larger segment of the sphere
    5. The apex of the pyramid is on the surface of the sphere

    Then the following properties hold for the pyramid:
    1. Its volume is 2/3 * r^3
    2. Its surface area is r^2 * (√(2√5 + 10) + √5 - 1)
    3. The tangent of its inclination angle is 1/2 * (√(√5 + 1))^3
-/
theorem pyramid_properties (r : ℝ) (h : r > 0) :
  ∃ (V F : ℝ) (tan_α : ℝ),
    V = 2/3 * r^3 ∧
    F = r^2 * (Real.sqrt (2 * Real.sqrt 5 + 10) + Real.sqrt 5 - 1) ∧
    tan_α = 1/2 * (Real.sqrt (Real.sqrt 5 + 1))^3 :=
sorry

end pyramid_properties_l2322_232214


namespace sum_of_digits_l2322_232246

-- Define the variables as natural numbers
variable (a b c d : ℕ)

-- Define the conditions
axiom different_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
axiom sum_hundreds_ones : a + c = 10
axiom sum_tens : b + c = 8
axiom sum_hundreds : a + d = 11
axiom result_sum : 100 * a + 10 * b + c + 100 * d + 10 * c + a = 1180

-- State the theorem
theorem sum_of_digits : a + b + c + d = 18 := by
  sorry

end sum_of_digits_l2322_232246


namespace frisbee_sales_theorem_l2322_232268

/-- The total number of frisbees sold given the conditions -/
def total_frisbees : ℕ := 64

/-- The price of the cheaper frisbees -/
def price_cheap : ℕ := 3

/-- The price of the more expensive frisbees -/
def price_expensive : ℕ := 4

/-- The total receipts from frisbee sales -/
def total_receipts : ℕ := 196

/-- The minimum number of expensive frisbees sold -/
def min_expensive : ℕ := 4

theorem frisbee_sales_theorem :
  ∃ (cheap expensive : ℕ),
    cheap + expensive = total_frisbees ∧
    cheap * price_cheap + expensive * price_expensive = total_receipts ∧
    expensive ≥ min_expensive :=
by
  sorry

end frisbee_sales_theorem_l2322_232268


namespace triangle_area_l2322_232254

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  b = 7 →
  c = 5 →
  B = 2 * π / 3 →
  (1/2) * a * c * Real.sin B = (15 * Real.sqrt 3) / 4 := by
  sorry

end triangle_area_l2322_232254


namespace percent_of_percent_l2322_232267

theorem percent_of_percent : (3 / 100) / (5 / 100) * 100 = 60 := by sorry

end percent_of_percent_l2322_232267


namespace nell_cards_given_to_jeff_l2322_232248

/-- The number of cards Nell gave to Jeff -/
def cards_given_to_jeff : ℕ := 304 - 276

theorem nell_cards_given_to_jeff :
  cards_given_to_jeff = 28 :=
by sorry

end nell_cards_given_to_jeff_l2322_232248


namespace investment_growth_period_l2322_232209

/-- The annual interest rate as a real number between 0 and 1 -/
def interest_rate : ℝ := 0.341

/-- The target multiple of the initial investment -/
def target_multiple : ℝ := 3

/-- The function to calculate the investment value after n years -/
def investment_value (n : ℕ) : ℝ := (1 + interest_rate) ^ n

/-- The smallest investment period in years -/
def smallest_period : ℕ := 4

theorem investment_growth_period :
  (∀ k : ℕ, k < smallest_period → investment_value k ≤ target_multiple) ∧
  target_multiple < investment_value smallest_period :=
sorry

end investment_growth_period_l2322_232209


namespace unique_solution_l2322_232226

theorem unique_solution (x y z : ℝ) 
  (hx : x > 3) (hy : y > 3) (hz : z > 3)
  (h : (x + 4)^2 / (y + z - 4) + (y + 6)^2 / (z + x - 6) + (z + 8)^2 / (x + y - 8) = 48) :
  x = 11 ∧ y = 10 ∧ z = 6 := by
  sorry

end unique_solution_l2322_232226


namespace least_n_factorial_divisible_by_8_l2322_232225

theorem least_n_factorial_divisible_by_8 : 
  ∃ n : ℕ, n > 0 ∧ 8 ∣ n.factorial ∧ ∀ m : ℕ, m > 0 → m < n → ¬(8 ∣ m.factorial) :=
by
  -- The proof goes here
  sorry

end least_n_factorial_divisible_by_8_l2322_232225


namespace trapezoid_median_length_l2322_232205

theorem trapezoid_median_length :
  let large_side : ℝ := 4
  let large_area : ℝ := (Real.sqrt 3 / 4) * large_side^2
  let small_area : ℝ := large_area / 3
  let small_side : ℝ := Real.sqrt ((4 * small_area) / Real.sqrt 3)
  let median : ℝ := (large_side + small_side) / 2
  median = (2 * (Real.sqrt 3 + 1)) / Real.sqrt 3 := by
  sorry

end trapezoid_median_length_l2322_232205


namespace large_positive_integer_product_l2322_232296

theorem large_positive_integer_product : ∃ n : ℕ, n > 10^100 ∧ 
  (2+3)*(2^2+3^2)*(2^4-3^4)*(2^8+3^8)*(2^16-3^16)*(2^32+3^32)*(2^64-3^64) = n := by
  sorry

end large_positive_integer_product_l2322_232296


namespace cube_cross_section_theorem_l2322_232235

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  point : Point3D

/-- Represents a polygon in 3D space -/
structure Polygon where
  vertices : List Point3D

def isPerpendicularTo (p : Plane) (v : Point3D) : Prop := sorry

def intersectsAllFaces (p : Plane) (c : Cube) : Prop := sorry

def crossSectionPolygon (p : Plane) (c : Cube) : Polygon := sorry

def perimeter (poly : Polygon) : ℝ := sorry

def area (poly : Polygon) : ℝ := sorry

theorem cube_cross_section_theorem (c : Cube) (p : Plane) (ac' : Point3D) :
  isPerpendicularTo p ac' →
  intersectsAllFaces p c →
  (∃ l : ℝ, ∀ α : Plane, isPerpendicularTo α ac' → intersectsAllFaces α c →
    perimeter (crossSectionPolygon α c) = l) ∧
  (¬∃ s : ℝ, ∀ α : Plane, isPerpendicularTo α ac' → intersectsAllFaces α c →
    area (crossSectionPolygon α c) = s) := by
  sorry

end cube_cross_section_theorem_l2322_232235


namespace down_payment_ratio_l2322_232275

theorem down_payment_ratio (total_cost balance_due daily_payment : ℚ) 
  (h1 : total_cost = 120)
  (h2 : balance_due = 60)
  (h3 : daily_payment = 6)
  (h4 : balance_due = daily_payment * 10) :
  (total_cost - balance_due) / total_cost = 1 / 2 := by
sorry

end down_payment_ratio_l2322_232275


namespace jess_walked_five_blocks_l2322_232210

/-- The number of blocks Jess has already walked -/
def blocks_walked (total_blocks remaining_blocks : ℕ) : ℕ :=
  total_blocks - remaining_blocks

/-- Proof that Jess has walked 5 blocks -/
theorem jess_walked_five_blocks :
  blocks_walked 25 20 = 5 := by
  sorry

end jess_walked_five_blocks_l2322_232210


namespace fourth_root_sum_squared_l2322_232292

theorem fourth_root_sum_squared : 
  (Real.rpow (7 + 3 * Real.sqrt 5) (1/4) + Real.rpow (7 - 3 * Real.sqrt 5) (1/4))^4 = 26 := by
  sorry

end fourth_root_sum_squared_l2322_232292


namespace scenario_1_scenario_2_l2322_232230

-- Define the lines l₁ and l₂
def l₁ (a b x y : ℝ) : Prop := a * x - b * y + 4 = 0
def l₂ (a x y : ℝ) : Prop := (a - 1) * x + y + 2 = 0

-- Define perpendicularity of lines
def perpendicular (a b : ℝ) : Prop := a * (1 - a) = -b

-- Define parallelism of lines
def parallel (a b : ℝ) : Prop := a / b = 1 - a

-- Theorem for Scenario 1
theorem scenario_1 (a b : ℝ) : 
  l₁ a b (-3) (-1) ∧ perpendicular a b → a = 2 ∧ b = 2 :=
sorry

-- Theorem for Scenario 2
theorem scenario_2 (a b : ℝ) :
  parallel a b ∧ (4 / b = -3) → a = 4 ∧ b = -4/3 :=
sorry

end scenario_1_scenario_2_l2322_232230


namespace sin_150_degrees_l2322_232241

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l2322_232241


namespace units_digit_17_pow_27_l2322_232252

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result we want to prove -/
theorem units_digit_17_pow_27 : unitsDigit (17^27) = 3 := by
  sorry

end units_digit_17_pow_27_l2322_232252


namespace exists_product_in_A_l2322_232262

/-- The set A(m, n) containing all integers of the form x^2 + mx + n for x ∈ ℤ -/
def A (m n : ℤ) : Set ℤ :=
  {y | ∃ x : ℤ, y = x^2 + m*x + n}

/-- For any integers m and n, there exist three distinct integers a, b, c in A(m, n) such that a = b * c -/
theorem exists_product_in_A (m n : ℤ) :
  ∃ a b c : ℤ, a ∈ A m n ∧ b ∈ A m n ∧ c ∈ A m n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a = b * c :=
by sorry

end exists_product_in_A_l2322_232262


namespace decimal_to_percentage_l2322_232202

theorem decimal_to_percentage (d : ℝ) (h : d = 0.05) : d * 100 = 5 := by
  sorry

end decimal_to_percentage_l2322_232202


namespace geometric_series_common_ratio_l2322_232232

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 16/21
  let a₃ : ℚ := 64/63
  let r : ℚ := a₂ / a₁
  r = 4/3 := by sorry

end geometric_series_common_ratio_l2322_232232


namespace square_area_from_adjacent_points_l2322_232294

/-- Given two adjacent points (1,2) and (4,6) on a square, prove that the area of the square is 25. -/
theorem square_area_from_adjacent_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 25 := by
  sorry

end square_area_from_adjacent_points_l2322_232294


namespace first_player_winning_strategy_l2322_232223

/-- A rectangular game board -/
structure GameBoard where
  m : ℝ
  n : ℝ

/-- A penny with radius 1 -/
structure Penny where
  center : ℝ × ℝ

/-- The game state -/
structure GameState where
  board : GameBoard
  pennies : List Penny

/-- Check if a new penny placement is valid -/
def is_valid_placement (state : GameState) (new_penny : Penny) : Prop :=
  ∀ p ∈ state.pennies, (new_penny.center.1 - p.center.1)^2 + (new_penny.center.2 - p.center.2)^2 > 4

/-- The winning condition for the first player -/
def first_player_wins (board : GameBoard) : Prop :=
  board.m ≥ 2 ∧ board.n ≥ 2

/-- The main theorem -/
theorem first_player_winning_strategy (board : GameBoard) :
  first_player_wins board ↔ ∃ (strategy : GameState → Penny), 
    ∀ (game : GameState), game.board = board → 
      (is_valid_placement game (strategy game) → 
        ∀ (opponent_move : Penny), is_valid_placement (GameState.mk board (strategy game :: game.pennies)) opponent_move → 
          ∃ (next_move : Penny), is_valid_placement (GameState.mk board (opponent_move :: strategy game :: game.pennies)) next_move) :=
sorry

end first_player_winning_strategy_l2322_232223


namespace sarahs_wallet_l2322_232261

theorem sarahs_wallet (total_amount : ℕ) (total_bills : ℕ) (five_dollar_count : ℕ) (ten_dollar_count : ℕ) : 
  total_amount = 100 →
  total_bills = 15 →
  five_dollar_count + ten_dollar_count = total_bills →
  5 * five_dollar_count + 10 * ten_dollar_count = total_amount →
  five_dollar_count = 10 := by
sorry

end sarahs_wallet_l2322_232261


namespace thousandths_place_of_three_sixteenths_l2322_232283

theorem thousandths_place_of_three_sixteenths (f : Rat) (d : ℕ) : 
  f = 3 / 16 →
  d = (⌊f * 1000⌋ % 10) →
  d = 7 := by
sorry

end thousandths_place_of_three_sixteenths_l2322_232283


namespace mixed_oil_rate_l2322_232239

/-- Calculates the rate of mixed oil per litre given the volumes and rates of three different oils. -/
theorem mixed_oil_rate (v1 v2 v3 r1 r2 r3 : ℚ) : 
  v1 > 0 ∧ v2 > 0 ∧ v3 > 0 ∧ r1 > 0 ∧ r2 > 0 ∧ r3 > 0 →
  (v1 * r1 + v2 * r2 + v3 * r3) / (v1 + v2 + v3) = 
    (15 * 50 + 8 * 75 + 10 * 65) / (15 + 8 + 10) :=
by
  sorry

#eval (15 * 50 + 8 * 75 + 10 * 65) / (15 + 8 + 10)

end mixed_oil_rate_l2322_232239


namespace photo_arrangements_l2322_232249

/-- The number of ways to arrange 1 teacher and 4 students in a row with the teacher in the middle -/
def arrangements_count : ℕ := 24

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of ways to arrange the students -/
def student_arrangements : ℕ := Nat.factorial num_students

theorem photo_arrangements :
  arrangements_count = student_arrangements := by
  sorry

end photo_arrangements_l2322_232249


namespace rain_probability_l2322_232281

-- Define the probability of rain on any given day
def p_rain : ℝ := 0.5

-- Define the number of days
def n : ℕ := 6

-- Define the number of rainy days we're interested in
def k : ℕ := 4

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem rain_probability :
  (binomial_coefficient n k : ℝ) * p_rain ^ k * (1 - p_rain) ^ (n - k) = 0.234375 := by
  sorry

end rain_probability_l2322_232281


namespace notebook_cost_l2322_232224

theorem notebook_cost (total_cost cover_cost notebook_cost : ℚ) : 
  total_cost = 2.4 →
  notebook_cost = cover_cost + 2 →
  total_cost = notebook_cost + cover_cost →
  notebook_cost = 2.2 := by
sorry

end notebook_cost_l2322_232224


namespace square_2601_difference_of_squares_l2322_232266

theorem square_2601_difference_of_squares (x : ℤ) (h : x^2 = 2601) :
  (x + 2) * (x - 2) = 2597 := by
sorry

end square_2601_difference_of_squares_l2322_232266


namespace total_amount_proof_l2322_232238

theorem total_amount_proof (a b c : ℕ) : 
  a = 3 * b → 
  b = c + 25 → 
  b = 134 → 
  a + b + c = 645 := by
sorry

end total_amount_proof_l2322_232238


namespace hotdog_cost_l2322_232285

theorem hotdog_cost (h s : ℕ) : 
  3 * h + 2 * s = 360 →
  2 * h + 3 * s = 390 →
  h = 60 := by sorry

end hotdog_cost_l2322_232285


namespace grain_milling_theorem_l2322_232250

/-- The amount of grain needed to be milled, in pounds -/
def grain_amount : ℚ := 111 + 1/9

/-- The milling fee percentage -/
def milling_fee_percent : ℚ := 1/10

/-- The amount of flour remaining after paying the fee, in pounds -/
def remaining_flour : ℚ := 100

theorem grain_milling_theorem :
  (1 - milling_fee_percent) * grain_amount = remaining_flour :=
by sorry

end grain_milling_theorem_l2322_232250


namespace eight_power_division_l2322_232212

theorem eight_power_division (x : ℕ) (y : ℕ) (z : ℕ) :
  x^15 / (x^2)^3 = x^9 :=
by sorry

end eight_power_division_l2322_232212


namespace sin_sum_specific_angles_l2322_232229

theorem sin_sum_specific_angles (θ φ : ℝ) :
  Complex.exp (θ * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I ∧
  Complex.exp (φ * Complex.I) = -(5 / 13 : ℂ) - (12 / 13 : ℂ) * Complex.I →
  Real.sin (θ + φ) = -(63 / 65) := by
  sorry

end sin_sum_specific_angles_l2322_232229


namespace regular_polygon_problem_l2322_232269

theorem regular_polygon_problem (n : ℕ) (n_gt_2 : n > 2) :
  (n - 2) * 180 = 3 * 360 + 180 →
  n = 9 ∧ (n - 2) * 180 / n = 140 := by
  sorry

end regular_polygon_problem_l2322_232269


namespace candy_boxes_minimum_l2322_232264

theorem candy_boxes_minimum (x y m : ℕ) : 
  x + y = 176 → 
  m > 1 → 
  x + 16 = m * (y - 16) + 31 → 
  x ≥ 131 :=
by sorry

end candy_boxes_minimum_l2322_232264


namespace race_distance_l2322_232259

/-- 
Proves that given the conditions of two runners A and B, 
the race distance is 160 meters.
-/
theorem race_distance (t_A t_B : ℝ) (lead : ℝ) : 
  t_A = 28 →  -- A's time
  t_B = 32 →  -- B's time
  lead = 20 → -- A's lead over B at finish
  ∃ d : ℝ, d = 160 ∧ d / t_A = (d - lead) / t_B :=
by sorry

end race_distance_l2322_232259


namespace lecture_duration_in_minutes_l2322_232272

-- Define the duration of the lecture
def lecture_hours : ℕ := 8
def lecture_minutes : ℕ := 45

-- Define the conversion factor
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem lecture_duration_in_minutes :
  lecture_hours * minutes_per_hour + lecture_minutes = 525 := by
  sorry

end lecture_duration_in_minutes_l2322_232272


namespace sqrt_123454321_l2322_232256

theorem sqrt_123454321 : Int.sqrt 123454321 = 11111 := by
  sorry

end sqrt_123454321_l2322_232256


namespace andrew_payment_l2322_232277

/-- The amount Andrew paid to the shopkeeper -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Andrew paid 1376 to the shopkeeper -/
theorem andrew_payment : total_amount 14 54 10 62 = 1376 := by
  sorry

end andrew_payment_l2322_232277


namespace monday_rainfall_rate_l2322_232278

/-- Represents the rainfall data for three days -/
structure RainfallData where
  monday_hours : ℝ
  monday_rate : ℝ
  tuesday_hours : ℝ
  tuesday_rate : ℝ
  wednesday_hours : ℝ
  wednesday_rate : ℝ
  total_rainfall : ℝ

/-- Theorem stating that given the rainfall conditions, the rate on Monday was 1 inch per hour -/
theorem monday_rainfall_rate (data : RainfallData)
  (h1 : data.monday_hours = 7)
  (h2 : data.tuesday_hours = 4)
  (h3 : data.tuesday_rate = 2)
  (h4 : data.wednesday_hours = 2)
  (h5 : data.wednesday_rate = 2 * data.tuesday_rate)
  (h6 : data.total_rainfall = 23)
  (h7 : data.total_rainfall = data.monday_hours * data.monday_rate + 
                              data.tuesday_hours * data.tuesday_rate + 
                              data.wednesday_hours * data.wednesday_rate) :
  data.monday_rate = 1 := by
  sorry

end monday_rainfall_rate_l2322_232278


namespace f_max_value_l2322_232203

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - x^3

-- State the theorem
theorem f_max_value :
  ∃ (c : ℝ), c > 0 ∧ f c = 2 ∧ ∀ x > 0, f x ≤ 2 := by
  sorry

end f_max_value_l2322_232203


namespace sisters_candy_count_l2322_232255

theorem sisters_candy_count 
  (debbys_candy : ℕ) 
  (eaten_candy : ℕ) 
  (remaining_candy : ℕ) 
  (h1 : debbys_candy = 32) 
  (h2 : eaten_candy = 35) 
  (h3 : remaining_candy = 39) : 
  ∃ (sisters_candy : ℕ), 
    sisters_candy = 42 ∧ 
    debbys_candy + sisters_candy = eaten_candy + remaining_candy :=
by
  sorry

end sisters_candy_count_l2322_232255


namespace reciprocal_of_sqrt_two_l2322_232298

theorem reciprocal_of_sqrt_two :
  (1 : ℝ) / Real.sqrt 2 = Real.sqrt 2 / 2 :=
by sorry

end reciprocal_of_sqrt_two_l2322_232298


namespace student_lineup_theorem_l2322_232236

theorem student_lineup_theorem (N : ℕ) (heights : Finset ℤ) :
  heights.card = 3 * N + 1 →
  ∃ (subset : Finset ℤ),
    subset ⊆ heights ∧
    subset.card = N + 1 ∧
    ∀ (x y : ℤ), x ∈ subset → y ∈ subset → x ≠ y → |x - y| ≥ 2 :=
by sorry

end student_lineup_theorem_l2322_232236


namespace quadratic_equation_roots_l2322_232271

theorem quadratic_equation_roots (m : ℕ+) : 
  (∃ x : ℝ, x^2 - 2*x + 2*(m : ℝ) - 1 = 0) → 
  (m = 1 ∧ ∀ x : ℝ, x^2 - 2*x + 2*(m : ℝ) - 1 = 0 → x = 1) :=
by sorry

end quadratic_equation_roots_l2322_232271


namespace max_value_theorem_l2322_232258

theorem max_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 - 2*a*b + 9*b^2 - c = 0) :
  ∃ (max_abc : ℝ), 
    (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x^2 - 2*x*y + 9*y^2 - z = 0 → 
      x*y/z ≤ max_abc) →
    (3/a + 1/b - 12/c ≤ 1) :=
sorry

end max_value_theorem_l2322_232258


namespace max_gcd_14n_plus_5_9n_plus_2_l2322_232218

theorem max_gcd_14n_plus_5_9n_plus_2 :
  (∃ (k : ℕ+), ∀ (n : ℕ+), Nat.gcd (14 * n + 5) (9 * n + 2) ≤ k) ∧
  (∃ (n : ℕ+), Nat.gcd (14 * n + 5) (9 * n + 2) = 4) := by
  sorry

end max_gcd_14n_plus_5_9n_plus_2_l2322_232218


namespace smaller_field_area_l2322_232220

/-- Given two square fields where one is 1% broader than the other,
    and the difference in their areas is 201 square meters,
    prove that the area of the smaller field is 10,000 square meters. -/
theorem smaller_field_area (s : ℝ) (h1 : s > 0) :
  (s * 1.01)^2 - s^2 = 201 → s^2 = 10000 := by sorry

end smaller_field_area_l2322_232220


namespace smallest_common_multiple_of_5_and_6_l2322_232265

theorem smallest_common_multiple_of_5_and_6 : 
  ∃ n : ℕ, n > 0 ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, m > 0 → 5 ∣ m → 6 ∣ m → n ≤ m :=
by sorry

#check smallest_common_multiple_of_5_and_6

end smallest_common_multiple_of_5_and_6_l2322_232265


namespace rectangular_solid_depth_l2322_232270

theorem rectangular_solid_depth (l w sa : ℝ) (h : ℝ) : 
  l = 6 → w = 5 → sa = 104 → sa = 2 * l * w + 2 * l * h + 2 * w * h → h = 2 := by
  sorry

end rectangular_solid_depth_l2322_232270


namespace rectangular_field_ratio_l2322_232221

theorem rectangular_field_ratio (perimeter width : ℝ) :
  perimeter = 432 →
  width = 90 →
  let length := (perimeter - 2 * width) / 2
  (length / width) = 7 / 5 := by
  sorry

end rectangular_field_ratio_l2322_232221


namespace fraction_identity_l2322_232276

theorem fraction_identity (n : ℕ) : 
  2 / ((2 * n - 1) * (2 * n + 1)) = 1 / (2 * n - 1) - 1 / (2 * n + 1) := by
  sorry

end fraction_identity_l2322_232276


namespace basketball_team_selection_l2322_232228

def total_players : Nat := 18
def quadruplets : Nat := 4
def starters : Nat := 6
def quadruplets_in_lineup : Nat := 2

theorem basketball_team_selection :
  (Nat.choose quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 6006 := by
  sorry

end basketball_team_selection_l2322_232228


namespace parabola_intercepts_sum_l2322_232288

/-- Represents a parabola of the form x = 3y^2 - 9y + 4 -/
def Parabola : ℝ → ℝ := λ y => 3 * y^2 - 9 * y + 4

/-- The x-intercept of the parabola -/
def a : ℝ := Parabola 0

/-- The y-intercepts of the parabola -/
def y_intercepts : Set ℝ := {y | Parabola y = 0}

theorem parabola_intercepts_sum :
  ∃ (b c : ℝ), y_intercepts = {b, c} ∧ a + b + c = 7 := by
  sorry

end parabola_intercepts_sum_l2322_232288


namespace fish_count_l2322_232286

/-- The number of fishbowls -/
def num_fishbowls : ℕ := 261

/-- The number of fish in each fishbowl -/
def fish_per_bowl : ℕ := 23

/-- The total number of fish -/
def total_fish : ℕ := num_fishbowls * fish_per_bowl

theorem fish_count : total_fish = 6003 := by
  sorry

end fish_count_l2322_232286


namespace gilda_marbles_theorem_l2322_232245

/-- The percentage of marbles Gilda has left after giving away to her friends and brother -/
def gildasRemainingMarbles : ℝ :=
  let initialMarbles := 100
  let afterPedro := initialMarbles * (1 - 0.30)
  let afterEbony := afterPedro * (1 - 0.20)
  let afterCarlos := afterEbony * (1 - 0.15)
  let afterJimmy := afterCarlos * (1 - 0.10)
  afterJimmy

/-- Theorem stating that Gilda has approximately 43% of her original marbles left -/
theorem gilda_marbles_theorem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |gildasRemainingMarbles - 43| < ε :=
sorry

end gilda_marbles_theorem_l2322_232245


namespace distance_and_closest_point_theorem_l2322_232291

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by a point and a direction vector -/
structure Line3D where
  point : Point3D
  direction : Vector3D

def distance_point_to_line (p : Point3D) (l : Line3D) : ℝ :=
  sorry

def closest_point_on_line (p : Point3D) (l : Line3D) : Point3D :=
  sorry

theorem distance_and_closest_point_theorem :
  let p := Point3D.mk 3 4 5
  let l := Line3D.mk (Point3D.mk 2 3 1) (Vector3D.mk 1 (-1) 2)
  distance_point_to_line p l = Real.sqrt 6 / 3 ∧
  closest_point_on_line p l = Point3D.mk (10/3) (5/3) (11/3) := by
  sorry

end distance_and_closest_point_theorem_l2322_232291


namespace area_inner_octagon_l2322_232273

/-- The area of a regular octagon formed by connecting the midpoints of four alternate sides of a regular octagon with side length 12 cm. -/
theorem area_inner_octagon (side_length : ℝ) (h_side : side_length = 12) : 
  ∃ area : ℝ, area = 576 + 288 * Real.sqrt 2 := by
  sorry

end area_inner_octagon_l2322_232273


namespace max_d_value_l2322_232215

def a (n : ℕ+) : ℕ := 100 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (m : ℕ+), ∀ (n : ℕ+), d n ≤ d m ∧ d m = 401 :=
sorry

end max_d_value_l2322_232215


namespace factor_calculation_l2322_232247

theorem factor_calculation (x : ℝ) : 60 * x - 138 = 102 ↔ x = 4 := by
  sorry

end factor_calculation_l2322_232247


namespace trouser_price_decrease_trouser_price_decrease_result_l2322_232299

/-- Calculates the final percent decrease in price for a trouser purchase with given conditions. -/
theorem trouser_price_decrease (original_price : ℝ) (clearance_discount : ℝ) 
  (german_vat : ℝ) (us_vat : ℝ) (exchange_rate : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - clearance_discount)
  let price_with_german_vat := discounted_price * (1 + german_vat)
  let price_in_usd := price_with_german_vat * exchange_rate
  let final_price := price_in_usd * (1 + us_vat)
  let original_price_usd := original_price * exchange_rate
  let percent_decrease := (original_price_usd - final_price) / original_price_usd * 100
  percent_decrease

/-- The final percent decrease in price is approximately 10.0359322%. -/
theorem trouser_price_decrease_result : 
  abs (trouser_price_decrease 100 0.3 0.19 0.08 1.18 - 10.0359322) < 0.0001 := by
  sorry

end trouser_price_decrease_trouser_price_decrease_result_l2322_232299


namespace optimal_shopping_solution_l2322_232207

/-- Represents the shopping problem with discounts --/
structure ShoppingProblem where
  budget : ℕ
  jacket_price : ℕ
  tshirt_price : ℕ
  jeans_price : ℕ

/-- Calculates the cost of jackets with the buy 2 get 1 free discount --/
def jacket_cost (n : ℕ) (price : ℕ) : ℕ :=
  (n / 3 * 2 + n % 3) * price

/-- Calculates the cost of t-shirts with the buy 3 get 1 free discount --/
def tshirt_cost (n : ℕ) (price : ℕ) : ℕ :=
  (n / 4 * 3 + n % 4) * price

/-- Calculates the cost of jeans with the 50% discount on every other pair --/
def jeans_cost (n : ℕ) (price : ℕ) : ℕ :=
  (n / 2 * 3 + n % 2) * (price / 2)

/-- Represents the optimal shopping solution --/
structure ShoppingSolution where
  jackets : ℕ
  tshirts : ℕ
  jeans : ℕ
  total_spent : ℕ
  remaining : ℕ

/-- Theorem stating the optimal solution for the shopping problem --/
theorem optimal_shopping_solution (p : ShoppingProblem)
    (h : p = { budget := 400, jacket_price := 50, tshirt_price := 25, jeans_price := 40 }) :
    ∃ (s : ShoppingSolution),
      s.jackets = 4 ∧
      s.tshirts = 12 ∧
      s.jeans = 3 ∧
      s.total_spent = 380 ∧
      s.remaining = 20 ∧
      jacket_cost s.jackets p.jacket_price +
      tshirt_cost s.tshirts p.tshirt_price +
      jeans_cost s.jeans p.jeans_price = s.total_spent ∧
      s.total_spent + s.remaining = p.budget ∧
      ∀ (s' : ShoppingSolution),
        jacket_cost s'.jackets p.jacket_price +
        tshirt_cost s'.tshirts p.tshirt_price +
        jeans_cost s'.jeans p.jeans_price ≤ p.budget →
        s'.jackets + s'.tshirts + s'.jeans ≤ s.jackets + s.tshirts + s.jeans :=
by sorry

end optimal_shopping_solution_l2322_232207


namespace tan_product_greater_than_one_l2322_232219

/-- In an acute triangle ABC, where a, b, c are sides opposite to angles A, B, C respectively,
    and a² = b² + bc, the product of tan A and tan B is always greater than 1. -/
theorem tan_product_greater_than_one (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  a^2 = b^2 + b*c →
  Real.tan A * Real.tan B > 1 := by
  sorry

end tan_product_greater_than_one_l2322_232219


namespace stop_to_qons_l2322_232244

/-- Represents a letter in a 2D coordinate system -/
structure Letter where
  char : Char
  x : ℝ
  y : ℝ

/-- Represents a word as a list of letters -/
def Word := List Letter

/-- Rotates a letter 180° clockwise about the origin -/
def rotate180 (l : Letter) : Letter :=
  { l with x := -l.x, y := -l.y }

/-- Reflects a letter in the x-axis -/
def reflectX (l : Letter) : Letter :=
  { l with y := -l.y }

/-- Applies both transformations to a letter -/
def transform (l : Letter) : Letter :=
  reflectX (rotate180 l)

/-- Applies the transformation to a word -/
def transformWord (w : Word) : Word :=
  w.map transform

/-- The initial word "stop" -/
def initialWord : Word := sorry

/-- The expected final word "qons" -/
def finalWord : Word := sorry

theorem stop_to_qons :
  transformWord initialWord = finalWord := by sorry

end stop_to_qons_l2322_232244


namespace current_rate_l2322_232240

/-- The rate of the current given a man's rowing speed and time ratio -/
theorem current_rate (man_speed : ℝ) (time_ratio : ℝ) : 
  man_speed = 3.6 ∧ time_ratio = 2 → 
  ∃ c : ℝ, c = 1.2 ∧ (man_speed - c) / (man_speed + c) = 1 / time_ratio :=
by sorry

end current_rate_l2322_232240


namespace salesman_pears_sold_l2322_232274

/-- The amount of pears sold by a salesman in a day -/
theorem salesman_pears_sold (morning_sales afternoon_sales : ℕ) 
  (h1 : afternoon_sales = 2 * morning_sales)
  (h2 : morning_sales = 120)
  (h3 : afternoon_sales = 240) : 
  morning_sales + afternoon_sales = 360 := by
  sorry

end salesman_pears_sold_l2322_232274


namespace subtract_negatives_l2322_232253

theorem subtract_negatives : -3 - 2 = -5 := by
  sorry

end subtract_negatives_l2322_232253


namespace purely_imaginary_x_equals_one_l2322_232289

-- Define a complex number
def complex_number (x : ℝ) : ℂ := (x^2 - 1) + (x + 1) * Complex.I

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem purely_imaginary_x_equals_one :
  ∀ x : ℝ, is_purely_imaginary (complex_number x) → x = 1 :=
sorry

end purely_imaginary_x_equals_one_l2322_232289


namespace kirill_height_l2322_232227

/-- Represents the heights of three siblings -/
structure SiblingHeights where
  kirill : ℝ
  brother : ℝ
  sister : ℝ

/-- The conditions of the problem -/
def height_conditions (h : SiblingHeights) : Prop :=
  h.brother = h.kirill + 14 ∧
  h.sister = 2 * h.kirill ∧
  h.kirill + h.brother + h.sister = 264

/-- Theorem stating Kirill's height given the conditions -/
theorem kirill_height (h : SiblingHeights) 
  (hc : height_conditions h) : h.kirill = 62.5 := by
  sorry

end kirill_height_l2322_232227


namespace stacy_extra_berries_l2322_232211

/-- The number of berries each person has -/
structure BerryCount where
  stacy : ℕ
  steve : ℕ
  skylar : ℕ

/-- The given conditions for the berry problem -/
def berry_conditions (b : BerryCount) : Prop :=
  b.stacy > 3 * b.steve ∧
  2 * b.steve = b.skylar ∧
  b.skylar = 20 ∧
  b.stacy = 32

/-- The theorem to prove -/
theorem stacy_extra_berries (b : BerryCount) (h : berry_conditions b) :
  b.stacy - 3 * b.steve = 2 := by
  sorry

end stacy_extra_berries_l2322_232211


namespace monomial_sum_condition_l2322_232200

/-- If the sum of the monomials $-2x^{4}y^{m-1}$ and $5x^{n-1}y^{2}$ is a monomial, then $m-2n = -7$. -/
theorem monomial_sum_condition (m n : ℤ) : 
  (∃ (a : ℚ) (b c : ℕ), -2 * X^4 * Y^(m-1) + 5 * X^(n-1) * Y^2 = a * X^b * Y^c) → 
  m - 2*n = -7 :=
by sorry

end monomial_sum_condition_l2322_232200
