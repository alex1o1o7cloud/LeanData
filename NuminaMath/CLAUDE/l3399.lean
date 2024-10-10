import Mathlib

namespace sequence_sum_equals_33_l3399_339949

def arithmetic_sequence (n : ℕ) : ℕ := 3 * n - 2

def geometric_sequence (n : ℕ) : ℕ := 3^(n - 1)

theorem sequence_sum_equals_33 :
  arithmetic_sequence (geometric_sequence 1) +
  arithmetic_sequence (geometric_sequence 2) +
  arithmetic_sequence (geometric_sequence 3) = 33 := by
  sorry

end sequence_sum_equals_33_l3399_339949


namespace segment_length_l3399_339918

/-- Given a line segment AB with points P and Q on it, prove that AB has length 120 -/
theorem segment_length (A B P Q : Real) : 
  (∃ x y u v : Real,
    -- P divides AB in ratio 3:5
    5 * x = 3 * y ∧ 
    -- Q divides AB in ratio 2:3
    3 * u = 2 * v ∧ 
    -- P is closer to A than Q
    u = x + 3 ∧ 
    v = y - 3 ∧ 
    -- AB is the sum of its parts
    A + B = x + y) → 
  A + B = 120 := by
sorry

end segment_length_l3399_339918


namespace total_ninja_stars_l3399_339904

-- Define the number of ninja throwing stars for each person
def eric_stars : ℕ := 4
def chad_stars_initial : ℕ := 2 * eric_stars
def jeff_bought : ℕ := 2
def jeff_stars_final : ℕ := 6

-- Define Chad's final number of stars
def chad_stars_final : ℕ := chad_stars_initial - jeff_bought

-- Theorem to prove
theorem total_ninja_stars :
  eric_stars + chad_stars_final + jeff_stars_final = 16 :=
by sorry

end total_ninja_stars_l3399_339904


namespace room_population_l3399_339948

theorem room_population (total : ℕ) (women : ℕ) (married : ℕ) (unmarried_women : ℕ) :
  women = total / 4 →
  married = 3 * total / 4 →
  unmarried_women ≤ 20 →
  unmarried_women = total - married →
  total = 80 :=
by
  sorry

end room_population_l3399_339948


namespace centrifuge_force_scientific_notation_l3399_339972

theorem centrifuge_force_scientific_notation :
  17000 = 1.7 * (10 ^ 4) := by sorry

end centrifuge_force_scientific_notation_l3399_339972


namespace jacob_number_problem_l3399_339958

theorem jacob_number_problem : 
  ∃! x : ℕ, 10 ≤ x ∧ x < 100 ∧ 
  (∃ a b : ℕ, 4 * x - 8 = 10 * a + b ∧ 
              25 ≤ 10 * b + a ∧ 10 * b + a ≤ 30) ∧
  x = 15 := by
  sorry

end jacob_number_problem_l3399_339958


namespace open_spots_difference_is_five_l3399_339980

/-- Represents a parking garage with given characteristics -/
structure ParkingGarage where
  totalLevels : Nat
  spotsPerLevel : Nat
  openSpotsFirstLevel : Nat
  openSpotsSecondLevel : Nat
  openSpotsFourthLevel : Nat
  totalFullSpots : Nat

/-- Calculates the difference between open spots on third and second levels -/
def openSpotsDifference (garage : ParkingGarage) : Int :=
  let totalSpots := garage.totalLevels * garage.spotsPerLevel
  let totalOpenSpots := totalSpots - garage.totalFullSpots
  let openSpotsThirdLevel := totalOpenSpots - garage.openSpotsFirstLevel - garage.openSpotsSecondLevel - garage.openSpotsFourthLevel
  openSpotsThirdLevel - garage.openSpotsSecondLevel

/-- Theorem stating the difference between open spots on third and second levels is 5 -/
theorem open_spots_difference_is_five (garage : ParkingGarage)
  (h1 : garage.totalLevels = 4)
  (h2 : garage.spotsPerLevel = 100)
  (h3 : garage.openSpotsFirstLevel = 58)
  (h4 : garage.openSpotsSecondLevel = garage.openSpotsFirstLevel + 2)
  (h5 : garage.openSpotsFourthLevel = 31)
  (h6 : garage.totalFullSpots = 186) :
  openSpotsDifference garage = 5 := by
  sorry

end open_spots_difference_is_five_l3399_339980


namespace average_hamburgers_per_day_l3399_339944

theorem average_hamburgers_per_day :
  let total_hamburgers : ℕ := 49
  let days_in_week : ℕ := 7
  let average := total_hamburgers / days_in_week
  average = 7 := by sorry

end average_hamburgers_per_day_l3399_339944


namespace box_plates_cups_weight_l3399_339990

/-- Given the weights of various combinations of a box, plates, and cups, 
    prove that the weight of the box with 10 plates and 20 cups is 3 kg. -/
theorem box_plates_cups_weight :
  ∀ (b p c : ℝ),
  (b + 20 * p + 30 * c = 4.8) →
  (b + 40 * p + 50 * c = 8.4) →
  (b + 10 * p + 20 * c = 3) :=
by sorry

end box_plates_cups_weight_l3399_339990


namespace half_marathon_total_yards_l3399_339931

/-- Represents the length of a race in miles and yards -/
structure RaceLength where
  miles : ℕ
  yards : ℚ

def half_marathon : RaceLength := { miles := 13, yards := 192.5 }

def yards_per_mile : ℕ := 1760

def num_races : ℕ := 6

theorem half_marathon_total_yards (m : ℕ) (y : ℚ) 
  (h1 : 0 ≤ y) (h2 : y < yards_per_mile) :
  m * yards_per_mile + y = 
    num_races * (half_marathon.miles * yards_per_mile + half_marathon.yards) → 
  y = 1155 := by
  sorry

end half_marathon_total_yards_l3399_339931


namespace kittens_sold_l3399_339935

theorem kittens_sold (initial_puppies initial_kittens puppies_sold remaining_pets : ℕ) : 
  initial_puppies = 7 →
  initial_kittens = 6 →
  puppies_sold = 2 →
  remaining_pets = 8 →
  initial_puppies + initial_kittens - puppies_sold - remaining_pets = 3 :=
by
  sorry

end kittens_sold_l3399_339935


namespace sum_of_number_and_square_l3399_339916

theorem sum_of_number_and_square : 
  let n : ℕ := 15
  n + n^2 = 240 := by sorry

end sum_of_number_and_square_l3399_339916


namespace equation_solutions_l3399_339995

def is_solution (x y : ℤ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 1987

def solution_count : ℕ := 5

theorem equation_solutions :
  (∃! (s : Finset (ℤ × ℤ)), s.card = solution_count ∧
    ∀ (p : ℤ × ℤ), p ∈ s ↔ is_solution p.1 p.2) :=
sorry

end equation_solutions_l3399_339995


namespace solution_ratio_l3399_339957

/-- Given a system of equations with solution (2, 5), prove that a/c = 3 -/
theorem solution_ratio (a c : ℝ) : 
  (∃ x y : ℝ, x = 2 ∧ y = 5 ∧ a * x + 2 * y = 16 ∧ 3 * x - y = c) →
  a / c = 3 := by
sorry

end solution_ratio_l3399_339957


namespace remaining_distance_condition_l3399_339921

/-- The total distance between points A and B in kilometers -/
def total_distance : ℕ := 500

/-- Alpha's daily cycling distance in kilometers -/
def alpha_daily_distance : ℕ := 30

/-- Beta's cycling distance on active days in kilometers -/
def beta_active_day_distance : ℕ := 50

/-- The number of days after which the condition is met -/
def condition_day : ℕ := 15

/-- The remaining distance for Alpha after n days -/
def alpha_remaining (n : ℕ) : ℕ := total_distance - n * alpha_daily_distance

/-- The remaining distance for Beta after n days -/
def beta_remaining (n : ℕ) : ℕ := total_distance - n * (beta_active_day_distance / 2)

/-- Theorem stating that on the 15th day, Beta's remaining distance is twice Alpha's -/
theorem remaining_distance_condition :
  beta_remaining condition_day = 2 * alpha_remaining condition_day :=
sorry

end remaining_distance_condition_l3399_339921


namespace M_intersect_N_eq_singleton_one_l3399_339917

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a : ℕ+, x = 2 * a - 1}

theorem M_intersect_N_eq_singleton_one : M ∩ N = {1} := by
  sorry

end M_intersect_N_eq_singleton_one_l3399_339917


namespace max_squared_ratio_is_one_l3399_339933

/-- The maximum squared ratio of a to b satisfying the given conditions -/
def max_squared_ratio (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≥ b ∧
  ∃ ρ : ℝ, ρ > 0 ∧
    ∀ x y : ℝ,
      0 ≤ x ∧ x < a ∧
      0 ≤ y ∧ y < b ∧
      a^2 + y^2 = b^2 + x^2 ∧
      b^2 + x^2 = (a - x)^2 + (b - y)^2 ∧
      (a - x) * (b - y) = 0 →
      (a / b)^2 ≤ ρ^2 ∧
      ρ^2 = 1

theorem max_squared_ratio_is_one (a b : ℝ) (h : max_squared_ratio a b) :
  ∃ ρ : ℝ, ρ > 0 ∧ ρ^2 = 1 :=
sorry

end max_squared_ratio_is_one_l3399_339933


namespace bisecting_line_theorem_l3399_339910

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculates the area of a quadrilateral given its vertices -/
def quadrilateralArea (a b c d : Point) : ℝ := sorry

/-- Checks if two lines are parallel -/
def isParallel (l1 l2 : Line) : Prop := sorry

/-- Calculates the area of the part of the quadrilateral below a given line -/
def areaBelow (a b c d : Point) (l : Line) : ℝ := sorry

/-- The main theorem to be proved -/
theorem bisecting_line_theorem (a b c d : Point) (l : Line) : 
  a = Point.mk 0 0 →
  b = Point.mk 16 0 →
  c = Point.mk 8 8 →
  d = Point.mk 0 8 →
  l = Line.mk 1 (-4) →
  isParallel l (Line.mk 1 0) ∧ 
  areaBelow a b c d l = (quadrilateralArea a b c d) / 2 := by
  sorry

end bisecting_line_theorem_l3399_339910


namespace sqrt_x_minus_two_real_l3399_339959

theorem sqrt_x_minus_two_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end sqrt_x_minus_two_real_l3399_339959


namespace mystery_book_shelves_l3399_339961

theorem mystery_book_shelves :
  ∀ (books_per_shelf : ℕ) 
    (picture_book_shelves : ℕ) 
    (total_books : ℕ),
  books_per_shelf = 8 →
  picture_book_shelves = 4 →
  total_books = 72 →
  ∃ (mystery_book_shelves : ℕ),
    mystery_book_shelves * books_per_shelf + 
    picture_book_shelves * books_per_shelf = total_books ∧
    mystery_book_shelves = 5 :=
by sorry

end mystery_book_shelves_l3399_339961


namespace sum_of_coefficients_l3399_339912

def polynomial (x : ℝ) : ℝ :=
  3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (x^4 + 3*x^2 - 2*x) - 4 * (2*x^6 - 5)

theorem sum_of_coefficients : 
  (polynomial 1) = -31 :=
sorry

end sum_of_coefficients_l3399_339912


namespace wednesday_sales_l3399_339940

def initial_stock : ℕ := 700
def monday_sales : ℕ := 50
def tuesday_sales : ℕ := 82
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40
def unsold_percentage : ℚ := 60 / 100

theorem wednesday_sales :
  let total_sales := initial_stock - (initial_stock * unsold_percentage).floor
  let other_days_sales := monday_sales + tuesday_sales + thursday_sales + friday_sales
  total_sales - other_days_sales = 60 := by
sorry

end wednesday_sales_l3399_339940


namespace parallel_vectors_x_value_l3399_339929

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 8)
  let b : ℝ × ℝ := (x, 4)
  parallel a b → x = 2 := by
  sorry

end parallel_vectors_x_value_l3399_339929


namespace repeating_decimal_sum_l3399_339954

theorem repeating_decimal_sum : 
  (0.12121212 : ℚ) + (0.003003003 : ℚ) + (0.0000500005 : ℚ) = 124215 / 999999 :=
by sorry

end repeating_decimal_sum_l3399_339954


namespace lyon_marseille_distance_l3399_339964

/-- Given a map distance and scale, calculates the real distance between two points. -/
def real_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Proves that the real distance between Lyon and Marseille is 1200 km. -/
theorem lyon_marseille_distance :
  let map_distance : ℝ := 120
  let scale : ℝ := 10
  real_distance map_distance scale = 1200 := by
  sorry

end lyon_marseille_distance_l3399_339964


namespace constant_term_expansion_l3399_339941

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the function for the general term of the expansion
def generalTerm (r : ℕ) : ℚ := (-1/2)^r * binomial 6 r

-- Define the constant term as the term where the power of x is zero
def constantTerm : ℚ := generalTerm 4

-- Theorem statement
theorem constant_term_expansion :
  constantTerm = 15/16 := by sorry

end constant_term_expansion_l3399_339941


namespace shaded_fraction_is_37_72_l3399_339919

/-- Represents a digit drawn on the grid -/
inductive Digit
  | one
  | nine
  | eight

/-- Represents the grid with drawn digits -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)
  (digits : List Digit)

/-- Calculates the number of small squares occupied by a digit -/
def squaresOccupied (d : Digit) : Nat :=
  match d with
  | Digit.one => 8
  | Digit.nine => 15
  | Digit.eight => 16

/-- Calculates the total number of squares in the grid -/
def totalSquares (g : Grid) : Nat :=
  g.rows * g.cols

/-- Calculates the number of squares occupied by all digits -/
def occupiedSquares (g : Grid) : Nat :=
  g.digits.foldl (fun acc d => acc + squaresOccupied d) 0

/-- Represents the fraction of shaded area -/
def shadedFraction (g : Grid) : Rat :=
  occupiedSquares g / totalSquares g

theorem shaded_fraction_is_37_72 (g : Grid) 
  (h1 : g.rows = 18)
  (h2 : g.cols = 8)
  (h3 : g.digits = [Digit.one, Digit.nine, Digit.nine, Digit.eight]) :
  shadedFraction g = 37 / 72 := by
  sorry

#eval shadedFraction { rows := 18, cols := 8, digits := [Digit.one, Digit.nine, Digit.nine, Digit.eight] }

end shaded_fraction_is_37_72_l3399_339919


namespace polynomial_root_product_l3399_339901

theorem polynomial_root_product (d e f : ℝ) : 
  let Q : ℝ → ℝ := λ x ↦ x^3 + d*x^2 + e*x + f
  (Q (Real.cos (π/5)) = 0) ∧ 
  (Q (Real.cos (3*π/5)) = 0) ∧ 
  (Q (Real.cos (4*π/5)) = 0) →
  d * e * f = 0 := by
sorry

end polynomial_root_product_l3399_339901


namespace isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3399_339996

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 50,
    where one side of the equilateral triangle is a side of the isosceles triangle,
    the base of the isosceles triangle is 10 units long. -/
theorem isosceles_triangle_base_length : ℝ → ℝ → ℝ → Prop :=
  fun equilateral_perimeter isosceles_perimeter isosceles_base =>
    equilateral_perimeter = 60 →
    isosceles_perimeter = 50 →
    let equilateral_side := equilateral_perimeter / 3
    let isosceles_side := equilateral_side
    isosceles_perimeter = 2 * isosceles_side + isosceles_base →
    isosceles_base = 10

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof :
  isosceles_triangle_base_length 60 50 10 :=
by
  sorry

end isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3399_339996


namespace race_speed_ratio_l3399_339984

/-- Proves the speed ratio in a race with given conditions -/
theorem race_speed_ratio (L : ℝ) (h : L > 0) : 
  ∃ R : ℝ, 
    (R > 0) ∧ 
    (0.26 * L = (1 - 0.74) * L) ∧
    (R * L = (1 - 0.60) * L) →
    R = 0.26 := by
  sorry

end race_speed_ratio_l3399_339984


namespace B_squared_equals_451_l3399_339969

/-- The function g defined as g(x) = √31 + 105/x -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 31 + 105 / x

/-- The equation from the problem -/
def problem_equation (x : ℝ) : Prop :=
  x = g (g (g (g (g x))))

/-- The sum of absolute values of roots of the equation -/
noncomputable def B : ℝ :=
  abs ((Real.sqrt 31 + Real.sqrt 451) / 2) +
  abs ((Real.sqrt 31 - Real.sqrt 451) / 2)

/-- Theorem stating that B^2 equals 451 -/
theorem B_squared_equals_451 : B^2 = 451 := by
  sorry

end B_squared_equals_451_l3399_339969


namespace investment_calculation_l3399_339936

/-- Represents the total investment amount in dollars -/
def total_investment : ℝ := 22000

/-- Represents the amount invested at 8% interest rate in dollars -/
def investment_at_8_percent : ℝ := 17000

/-- Represents the total interest earned in dollars -/
def total_interest : ℝ := 1710

/-- Represents the interest rate for the 8% investment -/
def rate_8_percent : ℝ := 0.08

/-- Represents the interest rate for the 7% investment -/
def rate_7_percent : ℝ := 0.07

theorem investment_calculation :
  rate_8_percent * investment_at_8_percent +
  rate_7_percent * (total_investment - investment_at_8_percent) =
  total_interest :=
sorry

end investment_calculation_l3399_339936


namespace victoria_worked_five_weeks_l3399_339911

/-- Calculates the number of weeks worked given the total hours and daily hours. -/
def weeksWorked (totalHours : ℕ) (dailyHours : ℕ) : ℚ :=
  (totalHours : ℚ) / (dailyHours * 7 : ℚ)

/-- Theorem: Victoria worked for 5 weeks -/
theorem victoria_worked_five_weeks :
  weeksWorked 315 9 = 5 := by sorry

end victoria_worked_five_weeks_l3399_339911


namespace child_ticket_cost_l3399_339915

theorem child_ticket_cost (num_children num_adults : ℕ) (adult_ticket_cost total_cost : ℚ) :
  num_children = 6 →
  num_adults = 10 →
  adult_ticket_cost = 16 →
  total_cost = 220 →
  (total_cost - num_adults * adult_ticket_cost) / num_children = 10 :=
by
  sorry

end child_ticket_cost_l3399_339915


namespace natural_numbers_less_than_two_l3399_339900

theorem natural_numbers_less_than_two :
  {n : ℕ | n < 2} = {0, 1} := by sorry

end natural_numbers_less_than_two_l3399_339900


namespace partner_investment_time_l3399_339945

/-- Given two partners p and q with investment ratio 7:5 and profit ratio 7:10,
    where p invested for 20 months, prove that q invested for 40 months. -/
theorem partner_investment_time (x : ℝ) (t : ℝ) : 
  (7 : ℝ) / 5 = 7 * x / (5 * x) →  -- investment ratio
  (7 : ℝ) / 10 = (7 * x * 20) / (5 * x * t) →  -- profit ratio
  t = 40 := by
sorry

end partner_investment_time_l3399_339945


namespace fraction_equality_l3399_339986

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4*x + y) / (x - 4*y) = 3) : 
  (x + 4*y) / (4*x - y) = 9/53 := by
  sorry

end fraction_equality_l3399_339986


namespace max_product_difference_l3399_339906

theorem max_product_difference (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : 0 ≤ a₁ ∧ a₁ ≤ 1) (h₂ : 0 ≤ a₂ ∧ a₂ ≤ 1) (h₃ : 0 ≤ a₃ ∧ a₃ ≤ 1) 
  (h₄ : 0 ≤ a₄ ∧ a₄ ≤ 1) (h₅ : 0 ≤ a₅ ∧ a₅ ≤ 1) : 
  |a₁ - a₂| * |a₁ - a₃| * |a₁ - a₄| * |a₁ - a₅| * 
  |a₂ - a₃| * |a₂ - a₄| * |a₂ - a₅| * 
  |a₃ - a₄| * |a₃ - a₅| * 
  |a₄ - a₅| ≤ 3 * Real.sqrt 21 / 38416 := by
  sorry

end max_product_difference_l3399_339906


namespace inequality_one_l3399_339939

theorem inequality_one (x : ℝ) : 
  (x + 2) / (x - 4) ≤ 0 ↔ -2 ≤ x ∧ x < 4 :=
sorry

end inequality_one_l3399_339939


namespace sufficient_not_necessary_condition_l3399_339975

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ ¬(1/a < 1 → a > 1) := by sorry

end sufficient_not_necessary_condition_l3399_339975


namespace certain_value_problem_l3399_339994

theorem certain_value_problem (n : ℤ) (v : ℤ) (h1 : n = -7) (h2 : 3 * n = 2 * n - v) : v = 7 := by
  sorry

end certain_value_problem_l3399_339994


namespace modulus_of_z_plus_one_equals_two_l3399_339993

def i : ℂ := Complex.I

theorem modulus_of_z_plus_one_equals_two :
  Complex.abs ((1 - 3 * i) / (1 + i) + 1) = 2 := by
  sorry

end modulus_of_z_plus_one_equals_two_l3399_339993


namespace sphere_radius_from_shadows_l3399_339966

/-- The radius of a sphere given its shadow and a reference post's shadow. -/
theorem sphere_radius_from_shadows
  (sphere_shadow : ℝ)
  (post_height : ℝ)
  (post_shadow : ℝ)
  (h1 : sphere_shadow = 15)
  (h2 : post_height = 1.5)
  (h3 : post_shadow = 3)
  (h4 : post_shadow > 0) -- Ensure division is valid
  : ∃ (r : ℝ), r = sphere_shadow * (post_height / post_shadow) ∧ r = 7.5 :=
by
  sorry


end sphere_radius_from_shadows_l3399_339966


namespace monkeys_eating_bananas_l3399_339988

/-- Given the rate at which monkeys eat bananas, prove that 6 monkeys are needed to eat 18 bananas in 18 minutes -/
theorem monkeys_eating_bananas 
  (initial_monkeys : ℕ) 
  (initial_time : ℕ) 
  (initial_bananas : ℕ) 
  (target_time : ℕ) 
  (target_bananas : ℕ) 
  (h1 : initial_monkeys = 6) 
  (h2 : initial_time = 6) 
  (h3 : initial_bananas = 6) 
  (h4 : target_time = 18) 
  (h5 : target_bananas = 18) : 
  (target_bananas * initial_time * initial_monkeys) / (initial_bananas * target_time) = 6 := by
  sorry

end monkeys_eating_bananas_l3399_339988


namespace max_power_under_500_l3399_339987

theorem max_power_under_500 :
  ∃ (a b : ℕ), 
    b > 1 ∧
    a^b < 500 ∧
    (∀ (c d : ℕ), d > 1 → c^d < 500 → c^d ≤ a^b) ∧
    a = 22 ∧
    b = 2 ∧
    a + b = 24 :=
by sorry

end max_power_under_500_l3399_339987


namespace mikes_land_profit_l3399_339934

/-- Calculates the profit from a land development project -/
def calculate_profit (total_acres : ℕ) (purchase_price_per_acre : ℕ) (sell_price_per_acre : ℕ) : ℕ :=
  let total_cost := total_acres * purchase_price_per_acre
  let acres_sold := total_acres / 2
  let total_revenue := acres_sold * sell_price_per_acre
  total_revenue - total_cost

/-- Proves that the profit from Mike's land development project is $6,000 -/
theorem mikes_land_profit :
  calculate_profit 200 70 200 = 6000 := by
  sorry

#eval calculate_profit 200 70 200

end mikes_land_profit_l3399_339934


namespace stating_pyramid_base_is_isosceles_l3399_339920

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  /-- The length of each lateral edge -/
  edge_length : ℝ
  /-- The area of each lateral face -/
  face_area : ℝ
  /-- Assumption that all lateral edges have the same length -/
  equal_edges : edge_length > 0
  /-- Assumption that all lateral faces have the same area -/
  equal_faces : face_area > 0

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  /-- The length of the two equal sides -/
  equal_side : ℝ
  /-- The length of the base -/
  base : ℝ
  /-- Assumption that the equal sides are positive -/
  positive_equal_side : equal_side > 0
  /-- Assumption that the base is positive -/
  positive_base : base > 0

/-- 
Theorem stating that the base of a triangular pyramid with equal lateral edges 
and equal lateral face areas is an isosceles triangle 
-/
theorem pyramid_base_is_isosceles (p : TriangularPyramid) : 
  ∃ (t : IsoscelesTriangle), True :=
sorry

end stating_pyramid_base_is_isosceles_l3399_339920


namespace at_least_one_inequality_holds_l3399_339937

theorem at_least_one_inequality_holds (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end at_least_one_inequality_holds_l3399_339937


namespace function_composition_property_l3399_339985

theorem function_composition_property (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + g y) = 2 * x + y) :
  ∀ x y : ℝ, g (x + f y) = x / 2 + y := by
  sorry

end function_composition_property_l3399_339985


namespace willie_cream_total_l3399_339908

/-- The amount of whipped cream Willie needs in total -/
def total_cream (farm_cream : ℕ) (bought_cream : ℕ) : ℕ :=
  farm_cream + bought_cream

/-- Theorem stating that Willie needs 300 lbs. of whipped cream in total -/
theorem willie_cream_total :
  total_cream 149 151 = 300 := by
  sorry

end willie_cream_total_l3399_339908


namespace base_k_subtraction_l3399_339927

/-- Represents a digit in base k -/
def Digit (k : ℕ) := {d : ℕ // d < k}

/-- Converts a two-digit number in base k to its decimal representation -/
def toDecimal (k : ℕ) (x y : Digit k) : ℕ := k * x.val + y.val

theorem base_k_subtraction (k : ℕ) (X Y : Digit k) 
  (h_k : k > 8)
  (h_eq : toDecimal k X Y + toDecimal k X X = 2 * k + 1) :
  X.val - Y.val = k - 4 := by sorry

end base_k_subtraction_l3399_339927


namespace quadratic_inequality_all_reals_l3399_339950

/-- The quadratic inequality ax² + bx + c < 0 has a solution set of all real numbers
    if and only if a < 0 and b² - 4ac < 0 -/
theorem quadratic_inequality_all_reals
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end quadratic_inequality_all_reals_l3399_339950


namespace isosceles_triangle_perimeter_l3399_339965

theorem isosceles_triangle_perimeter (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Sides are positive
  (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2) →  -- Two sides measure 2 and 5
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  (a = b ∨ b = c ∨ c = a) →  -- Isosceles condition
  a + b + c = 12 :=  -- Perimeter is 12
by sorry

end isosceles_triangle_perimeter_l3399_339965


namespace athletes_leaving_rate_l3399_339956

/-- The rate at which athletes left the camp per hour -/
def leaving_rate : ℝ := 24.5

/-- The initial number of athletes at the camp -/
def initial_athletes : ℕ := 300

/-- The number of hours athletes left the camp -/
def leaving_hours : ℕ := 4

/-- The rate at which new athletes entered the camp per hour -/
def entering_rate : ℕ := 15

/-- The number of hours new athletes entered the camp -/
def entering_hours : ℕ := 7

/-- The difference in total number of athletes over the two nights -/
def athlete_difference : ℕ := 7

theorem athletes_leaving_rate : 
  initial_athletes - leaving_rate * leaving_hours + entering_rate * entering_hours 
  = initial_athletes + athlete_difference :=
sorry

end athletes_leaving_rate_l3399_339956


namespace area_of_specific_triangle_l3399_339914

/-- The line equation ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A triangle bounded by the x-axis, y-axis, and a line --/
structure AxisAlignedTriangle where
  boundingLine : Line

/-- The area of an axis-aligned triangle --/
def areaOfAxisAlignedTriangle (t : AxisAlignedTriangle) : ℝ :=
  sorry

theorem area_of_specific_triangle : 
  let t : AxisAlignedTriangle := { boundingLine := { a := 3, b := 4, c := 12 } }
  areaOfAxisAlignedTriangle t = 6 := by sorry

end area_of_specific_triangle_l3399_339914


namespace min_value_theorem_l3399_339979

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∀ z : ℝ, z = 1 / x + 4 / y → z ≥ 9 :=
by
  sorry

end min_value_theorem_l3399_339979


namespace arithmetic_geometric_sequence_problem_l3399_339960

/-- Three positive numbers form an arithmetic sequence -/
def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

/-- Three numbers form a geometric sequence -/
def is_geometric_sequence (a b c : ℝ) : Prop := b / a = c / b

theorem arithmetic_geometric_sequence_problem :
  ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  is_arithmetic_sequence a b c →
  a + b + c = 15 →
  is_geometric_sequence (a + 1) (b + 3) (c + 9) →
  a = 3 ∧ b = 5 ∧ c = 7 :=
sorry

end arithmetic_geometric_sequence_problem_l3399_339960


namespace max_power_of_two_product_l3399_339970

open BigOperators

def is_permutation (a : Fin 17 → ℕ) : Prop :=
  ∀ i : Fin 17, ∃ j : Fin 17, a j = i.val + 1

theorem max_power_of_two_product (a : Fin 17 → ℕ) (n : ℕ) 
  (h_perm : is_permutation a) 
  (h_prod : ∏ i : Fin 17, (a i - a (i + 1)) = 2^n) : 
  n ≤ 40 ∧ ∃ a₀ : Fin 17 → ℕ, is_permutation a₀ ∧ ∏ i : Fin 17, (a₀ i - a₀ (i + 1)) = 2^40 :=
sorry

end max_power_of_two_product_l3399_339970


namespace factorization_equality_l3399_339930

theorem factorization_equality (x y : ℝ) : 
  (x + y)^2 - 14*(x + y) + 49 = (x + y - 7)^2 := by sorry

end factorization_equality_l3399_339930


namespace hyogeun_weight_l3399_339971

/-- Given the weights of three people satisfying certain conditions, 
    prove that one person's weight is as specified. -/
theorem hyogeun_weight (H S G : ℝ) : 
  H + S + G = 106.6 →
  G = S - 7.7 →
  S = H - 4.8 →
  H = 41.3 := by
sorry

end hyogeun_weight_l3399_339971


namespace inequality_solution_set_l3399_339998

/-- A function satisfying the given conditions -/
def f_satisfies (f : ℝ → ℝ) : Prop :=
  f 0 = 2 ∧ 
  ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 1

/-- The solution set of the inequality -/
def solution_set (x : ℝ) : Prop :=
  Real.log 2 < x ∧ x < Real.log 3

/-- The main theorem -/
theorem inequality_solution_set (f : ℝ → ℝ) (hf : f_satisfies f) :
  ∀ x, f (Real.log (Real.exp x - 2)) < 2 + Real.log (Real.exp x - 2) ↔ solution_set x := by
  sorry

end inequality_solution_set_l3399_339998


namespace total_skittles_l3399_339963

theorem total_skittles (num_students : ℕ) (skittles_per_student : ℕ) 
  (h1 : num_students = 9)
  (h2 : skittles_per_student = 3) :
  num_students * skittles_per_student = 27 := by
  sorry

end total_skittles_l3399_339963


namespace downstream_distance_l3399_339955

/-- Calculates the distance traveled downstream by a boat -/
theorem downstream_distance
  (boat_speed : ℝ)  -- Speed of the boat in still water
  (stream_speed : ℝ) -- Speed of the stream
  (time : ℝ)  -- Time taken to travel downstream
  (h1 : boat_speed = 40)  -- Condition: Boat speed is 40 km/hr
  (h2 : stream_speed = 5)  -- Condition: Stream speed is 5 km/hr
  (h3 : time = 1)  -- Condition: Time taken is 1 hour
  : boat_speed + stream_speed * time = 45 := by
  sorry

#check downstream_distance

end downstream_distance_l3399_339955


namespace Q_four_roots_implies_d_value_l3399_339967

/-- The polynomial Q(x) -/
def Q (d x : ℂ) : ℂ := (x^2 - 3*x + 3) * (x^2 - d*x + 5) * (x^2 - 5*x + 15)

/-- The theorem stating that if Q(x) has exactly 4 distinct roots, then |d| = 13/2 -/
theorem Q_four_roots_implies_d_value (d : ℂ) :
  (∃ (s : Finset ℂ), s.card = 4 ∧ (∀ x ∈ s, Q d x = 0) ∧ (∀ x, Q d x = 0 → x ∈ s)) →
  Complex.abs d = 13/2 := by
  sorry

end Q_four_roots_implies_d_value_l3399_339967


namespace counterexample_exists_l3399_339947

theorem counterexample_exists : ∃ (a b : ℝ), a < b ∧ a^2 ≥ b^2 := by
  sorry

end counterexample_exists_l3399_339947


namespace sum_of_21st_set_l3399_339903

/-- The sum of elements in the n-th set of a sequence where:
    1. Each set contains consecutive integers
    2. Each set contains one more element than the previous set
    3. The first element of each set is one greater than the last element of the previous set
-/
def S (n : ℕ) : ℚ :=
  n * (n^2 - n + 2) / 2

theorem sum_of_21st_set :
  S 21 = 4641 :=
sorry

end sum_of_21st_set_l3399_339903


namespace rectangle_area_l3399_339982

-- Define a rectangle type
structure Rectangle where
  width : ℝ
  length : ℝ
  diagonal : ℝ

-- Theorem statement
theorem rectangle_area (r : Rectangle) 
  (h1 : r.length = 2 * r.width) 
  (h2 : r.diagonal = 15 * Real.sqrt 2) : 
  r.width * r.length = 180 := by
  sorry

end rectangle_area_l3399_339982


namespace chair_capacity_l3399_339909

theorem chair_capacity (total_chairs : ℕ) (attended : ℕ) : 
  total_chairs = 40 →
  (2 : ℚ) / 5 * total_chairs = total_chairs - (3 : ℚ) / 5 * total_chairs →
  2 * ((3 : ℚ) / 5 * total_chairs) = attended →
  attended = 48 →
  ∃ (capacity : ℕ), capacity = 48 ∧ capacity * total_chairs = capacity * attended :=
by
  sorry

end chair_capacity_l3399_339909


namespace sin_15_cos_15_equals_quarter_l3399_339983

theorem sin_15_cos_15_equals_quarter :
  Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by sorry

end sin_15_cos_15_equals_quarter_l3399_339983


namespace complex_equation_solution_l3399_339926

theorem complex_equation_solution (z : ℂ) (h : Complex.I * (z + 2 * Complex.I) = 1) : z = -3 * Complex.I := by
  sorry

end complex_equation_solution_l3399_339926


namespace projection_a_on_b_l3399_339997

def a : ℝ × ℝ := (-1, 3)
def b : ℝ × ℝ := (3, -4)

theorem projection_a_on_b :
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -3 := by sorry

end projection_a_on_b_l3399_339997


namespace fraction_change_l3399_339973

theorem fraction_change (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (0.6 * x) / (0.4 * y) = 1.5 * (x / y) := by
sorry

end fraction_change_l3399_339973


namespace geometric_sum_first_five_terms_l3399_339976

/-- Sum of first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric sequence -/
def a : ℚ := 1/4

/-- The common ratio of the geometric sequence -/
def r : ℚ := 1/4

/-- The number of terms to sum -/
def n : ℕ := 5

theorem geometric_sum_first_five_terms :
  geometric_sum a r n = 341/1024 := by
  sorry

end geometric_sum_first_five_terms_l3399_339976


namespace can_identify_80_weights_l3399_339946

/-- Represents a comparison between two sets of weights -/
def Comparison := List ℕ → List ℕ → Bool

/-- Given a list of weights and a number of comparisons, 
    determines if it's possible to uniquely identify all weights -/
def can_identify (weights : List ℕ) (num_comparisons : ℕ) : Prop :=
  ∃ (comparisons : List Comparison), 
    comparisons.length = num_comparisons ∧ 
    ∀ (w1 w2 : List ℕ), w1 ≠ w2 → w1.length = weights.length → w2.length = weights.length →
      ∃ (c : Comparison), c ∈ comparisons ∧ c w1 ≠ c w2

theorem can_identify_80_weights :
  ∀ (weights : List ℕ), 
    weights.length = 80 → 
    weights.Nodup → 
    (can_identify weights 4 ∧ ¬can_identify weights 3) := by
  sorry

#check can_identify_80_weights

end can_identify_80_weights_l3399_339946


namespace fraction_reduction_l3399_339925

theorem fraction_reduction (a b : ℕ) (h : a = 4128 ∧ b = 4386) :
  ∃ (c d : ℕ), c = 295 ∧ d = 313 ∧ a / b = c / d ∧ Nat.gcd c d = 1 := by
  sorry

end fraction_reduction_l3399_339925


namespace cube_volume_in_pyramid_l3399_339902

/-- Represents a pyramid with a regular hexagonal base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side_length : ℝ)

/-- Represents a cube placed inside the pyramid -/
structure Cube :=
  (side_length : ℝ)

/-- Calculate the volume of a cube -/
def cube_volume (c : Cube) : ℝ := c.side_length ^ 3

/-- The configuration of the pyramid and cube as described in the problem -/
def pyramid_cube_configuration (p : Pyramid) (c : Cube) : Prop :=
  p.base_side_length = 2 ∧
  c.side_length = 2 * Real.sqrt 3 / 9

/-- The theorem stating the volume of the cube in the given configuration -/
theorem cube_volume_in_pyramid (p : Pyramid) (c : Cube) :
  pyramid_cube_configuration p c →
  cube_volume c = 8 * Real.sqrt 3 / 243 := by
  sorry

end cube_volume_in_pyramid_l3399_339902


namespace diamond_value_l3399_339974

def diamond (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

theorem diamond_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : a + b = 10) (h2 : a * b = 24) : 
  diamond a b = 5 / 12 := by
  sorry

end diamond_value_l3399_339974


namespace greatest_integer_radius_l3399_339907

theorem greatest_integer_radius (A : ℝ) (h : A < 200 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi ≤ A ∧ ∀ (s : ℕ), s * s * Real.pi ≤ A → s ≤ r :=
by
  sorry

end greatest_integer_radius_l3399_339907


namespace marble_probability_l3399_339942

/-- The probability of drawing 1 blue marble and 2 black marbles from a basket -/
theorem marble_probability (blue yellow black : ℕ) 
  (h_blue : blue = 4)
  (h_yellow : yellow = 6)
  (h_black : black = 7) :
  let total := blue + yellow + black
  (blue : ℚ) / total * (black * (black - 1) : ℚ) / ((total - 1) * (total - 2)) = 7 / 170 := by
  sorry

end marble_probability_l3399_339942


namespace unique_prime_with_prime_sums_l3399_339977

theorem unique_prime_with_prime_sums : ∃! p : ℕ, 
  Nat.Prime p ∧ Nat.Prime (p + 10) ∧ Nat.Prime (p + 14) := by sorry

end unique_prime_with_prime_sums_l3399_339977


namespace f_max_value_l3399_339932

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (x + Real.pi / 2) + Real.cos (Real.pi / 6 - x)

theorem f_max_value : ∀ x : ℝ, f x ≤ Real.sqrt 13 / 2 ∧ ∃ y : ℝ, f y = Real.sqrt 13 / 2 := by sorry

end f_max_value_l3399_339932


namespace catch_game_end_state_l3399_339923

/-- Represents the state of the game at each throw -/
structure GameState where
  throw_number : ℕ
  distance : ℕ

/-- Calculates the game state for a given throw number -/
def game_state (n : ℕ) : GameState :=
  { throw_number := n,
    distance := (n + 1) / 2 }

/-- Determines if Pat is throwing based on the throw number -/
def is_pat_throwing (n : ℕ) : Prop :=
  n % 2 = 1

theorem catch_game_end_state :
  let final_throw := 29
  let final_state := game_state final_throw
  final_state.distance = 15 ∧ is_pat_throwing final_throw := by
sorry

end catch_game_end_state_l3399_339923


namespace smallest_cut_length_five_is_smallest_smallest_integral_cut_l3399_339978

theorem smallest_cut_length (x : ℕ) : x ≥ 5 ↔ ¬(9 - x + 14 - x > 18 - x) :=
  sorry

theorem five_is_smallest : ∀ y : ℕ, y < 5 → (9 - y + 14 - y > 18 - y) :=
  sorry

theorem smallest_integral_cut : 
  ∃ x : ℕ, (x ≥ 5) ∧ (∀ y : ℕ, y < x → (9 - y + 14 - y > 18 - y)) :=
  sorry

end smallest_cut_length_five_is_smallest_smallest_integral_cut_l3399_339978


namespace abs_sum_zero_implies_sum_l3399_339951

theorem abs_sum_zero_implies_sum (x y : ℝ) :
  |x - 1| + |y + 3| = 0 → x + y = -2 := by
sorry

end abs_sum_zero_implies_sum_l3399_339951


namespace giraffe_difference_l3399_339913

/-- In a zoo with giraffes and other animals, where the number of giraffes
    is 3 times the number of all other animals, prove that there are 200
    more giraffes than other animals. -/
theorem giraffe_difference (total_giraffes : ℕ) (other_animals : ℕ) : 
  total_giraffes = 300 →
  total_giraffes = 3 * other_animals →
  total_giraffes - other_animals = 200 :=
by sorry

end giraffe_difference_l3399_339913


namespace monotone_increasing_iff_a_in_range_l3399_339924

/-- A quadratic function f(x) = ax^2 + 2x - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 3

/-- The statement that f is monotonically increasing on (-∞, 4) iff a ∈ [-1/4, 0] -/
theorem monotone_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y, x < y → x < 4 → f a x < f a y) ↔ -1/4 ≤ a ∧ a ≤ 0 :=
sorry

end monotone_increasing_iff_a_in_range_l3399_339924


namespace number_puzzle_l3399_339999

theorem number_puzzle (A B : ℝ) (h1 : A + B = 14.85) (h2 : B = 10 * A) : A = 1.35 := by
  sorry

end number_puzzle_l3399_339999


namespace sequence_divisibility_l3399_339962

theorem sequence_divisibility (m n k : ℕ) (a : ℕ → ℕ) (hm : m > 1) (hn : n ≥ 0) :
  m^n ∣ a k → m^(n+1) ∣ (a (k+1))^m - (a (k-1))^m :=
by sorry

end sequence_divisibility_l3399_339962


namespace quadratic_root_sum_l3399_339928

/-- Given that 3i - 2 is a root of the quadratic equation 2x^2 + px + q = 0,
    prove that p + q = 34. -/
theorem quadratic_root_sum (p q : ℝ) : 
  (2 * (Complex.I * 3 - 2)^2 + p * (Complex.I * 3 - 2) + q = 0) →
  p + q = 34 := by
sorry

end quadratic_root_sum_l3399_339928


namespace square_area_from_circles_l3399_339981

/-- Given two circles where one passes through the center of and is tangent to the other,
    and the other is inscribed in a square, this theorem proves the area of the square
    given the area of the first circle. -/
theorem square_area_from_circles (circle_I circle_II : Real → Prop) (square : Real → Prop) : 
  (∃ r R s : Real,
    -- Circle I has area 9π
    circle_I r ∧ π * r^2 = 9 * π ∧
    -- Circle I passes through center of and is tangent to Circle II
    circle_II R ∧ R = 2 * r ∧
    -- Circle II is inscribed in the square
    square s ∧ s = 2 * R) →
  (∃ area : Real, square area ∧ area = 36) :=
by sorry

end square_area_from_circles_l3399_339981


namespace percentage_of_B_grades_l3399_339905

def grading_scale : List (String × (Int × Int)) :=
  [("A", (94, 100)), ("B", (87, 93)), ("C", (78, 86)), ("D", (70, 77)), ("F", (0, 69))]

def scores : List Int := [93, 65, 88, 100, 72, 95, 82, 68, 79, 56, 87, 81, 74, 85, 91]

def is_grade (score : Int) (grade : String × (Int × Int)) : Bool :=
  let (_, (low, high)) := grade
  low ≤ score ∧ score ≤ high

def count_grade (scores : List Int) (grade : String × (Int × Int)) : Nat :=
  (scores.filter (fun score => is_grade score grade)).length

theorem percentage_of_B_grades :
  let b_grade := ("B", (87, 93))
  let total_students := scores.length
  let b_students := count_grade scores b_grade
  (b_students : Rat) / total_students * 100 = 20 := by
  sorry

end percentage_of_B_grades_l3399_339905


namespace round_trip_distance_prove_round_trip_distance_l3399_339991

def boat_speed : ℝ := 9
def stream_speed : ℝ := 6
def total_time : ℝ := 68

theorem round_trip_distance : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let distance := (total_time * downstream_speed * upstream_speed) / (downstream_speed + upstream_speed)
  170

theorem prove_round_trip_distance : round_trip_distance = 170 := by
  sorry

end round_trip_distance_prove_round_trip_distance_l3399_339991


namespace f_value_at_one_l3399_339938

/-- The polynomial g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 10

/-- The polynomial f(x) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 100*x + c

theorem f_value_at_one (a b c : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0) →
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →
  f b c 1 = -7007 := by
  sorry

end f_value_at_one_l3399_339938


namespace sum_even_integers_100_to_200_l3399_339968

-- Define the sum of the first n positive even integers
def sumFirstNEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

-- Define the sum of even integers from a to b inclusive
def sumEvenIntegersFromTo (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  n * (a + b) / 2

-- Theorem statement
theorem sum_even_integers_100_to_200 :
  sumFirstNEvenIntegers 50 = 2550 →
  sumEvenIntegersFromTo 100 200 = 7650 := by
  sorry

end sum_even_integers_100_to_200_l3399_339968


namespace line_m_equation_l3399_339953

/-- Two distinct lines in the xy-plane -/
structure TwoLines where
  ℓ : Set (ℝ × ℝ)
  m : Set (ℝ × ℝ)
  distinct : ℓ ≠ m
  intersect_origin : (0, 0) ∈ ℓ ∩ m

/-- Equation of a line in the form ax + by = 0 -/
def LineEquation (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | a * x + b * y = 0}

/-- Reflection of a point about a line -/
def reflect (p : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem line_m_equation (lines : TwoLines) 
  (h_ℓ : lines.ℓ = LineEquation 2 1)
  (h_Q : reflect (3, -2) lines.ℓ = reflect (reflect (3, -2) lines.ℓ) lines.m)
  (h_Q'' : reflect (reflect (3, -2) lines.ℓ) lines.m = (-1, 5)) :
  lines.m = LineEquation 3 1 := by sorry

end line_m_equation_l3399_339953


namespace fitness_center_member_ratio_l3399_339989

theorem fitness_center_member_ratio 
  (f m : ℕ) -- number of female and male members
  (avg_female : ℕ := 45) -- average age of female members
  (avg_male : ℕ := 30) -- average age of male members
  (avg_all : ℕ := 35) -- average age of all members
  (h : (f * avg_female + m * avg_male) / (f + m) = avg_all) : 
  f / m = 1 / 2 := by
sorry

end fitness_center_member_ratio_l3399_339989


namespace isosceles_right_triangle_probability_l3399_339992

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Calculate the probability of a point being in a specific region of a triangle -/
def probabilityInRegion (t : Triangle) (condition : Point → Bool) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem isosceles_right_triangle_probability :
  let A : Point := ⟨0, 8⟩
  let B : Point := ⟨0, 0⟩
  let C : Point := ⟨8, 0⟩
  let ABC : Triangle := ⟨A, B, C⟩
  probabilityInRegion ABC (fun P => 
    triangleArea ⟨P, B, C⟩ < (1/3) * triangleArea ABC) = 7/32 := by
  sorry

end isosceles_right_triangle_probability_l3399_339992


namespace broker_commission_rate_l3399_339943

theorem broker_commission_rate 
  (initial_rate : ℝ) 
  (slump_percentage : ℝ) 
  (new_rate : ℝ) :
  initial_rate = 0.04 →
  slump_percentage = 0.20000000000000007 →
  new_rate = initial_rate / (1 - slump_percentage) →
  new_rate = 0.05 := by
sorry

end broker_commission_rate_l3399_339943


namespace sum_a_b_value_l3399_339952

theorem sum_a_b_value (a b : ℚ) (h1 : 2 * a + 5 * b = 43) (h2 : 8 * a + 2 * b = 50) :
  a + b = 34 / 3 := by
sorry

end sum_a_b_value_l3399_339952


namespace inverse_proportion_ordering_l3399_339922

/-- Proves the ordering of y-coordinates for three points on an inverse proportion function -/
theorem inverse_proportion_ordering (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h_pos : k > 0)
  (h_A : y₁ = k / (-3))
  (h_B : y₂ = k / (-2))
  (h_C : y₃ = k / 2) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end inverse_proportion_ordering_l3399_339922
