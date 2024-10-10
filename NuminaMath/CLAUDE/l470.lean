import Mathlib

namespace diver_min_trips_l470_47038

/-- Calculates the minimum number of trips required to transport objects --/
def min_trips (objects_per_trip : ℕ) (total_objects : ℕ) : ℕ :=
  (total_objects + objects_per_trip - 1) / objects_per_trip

/-- Theorem: Given a diver who can carry 3 objects at a time and has found 17 objects,
    the minimum number of trips required to transport all objects is 6 --/
theorem diver_min_trips :
  min_trips 3 17 = 6 := by
  sorry

end diver_min_trips_l470_47038


namespace car_rental_rates_equal_l470_47034

/-- The daily rate of Sunshine Car Rentals -/
def sunshine_daily_rate : ℝ := 17.99

/-- The per-mile rate of Sunshine Car Rentals -/
def sunshine_mile_rate : ℝ := 0.18

/-- The per-mile rate of the second car rental company -/
def second_company_mile_rate : ℝ := 0.16

/-- The number of miles driven -/
def miles_driven : ℝ := 48

/-- The daily rate of the second car rental company -/
def second_company_daily_rate : ℝ := 18.95

theorem car_rental_rates_equal :
  sunshine_daily_rate + sunshine_mile_rate * miles_driven =
  second_company_daily_rate + second_company_mile_rate * miles_driven :=
by sorry

end car_rental_rates_equal_l470_47034


namespace equation_solutions_l470_47063

theorem equation_solutions :
  (∀ x : ℝ, 4 * (x - 1)^2 = 100 ↔ x = 6 ∨ x = -4) ∧
  (∀ x : ℝ, (2*x - 1)^3 = -8 ↔ x = -1/2) := by
  sorry

end equation_solutions_l470_47063


namespace matchstick_20th_stage_l470_47004

/-- Arithmetic sequence with first term 3 and common difference 3 -/
def matchstick_sequence (n : ℕ) : ℕ := 3 + (n - 1) * 3

/-- The 20th term of the matchstick sequence is 60 -/
theorem matchstick_20th_stage : matchstick_sequence 20 = 60 := by
  sorry

end matchstick_20th_stage_l470_47004


namespace vegetables_in_soup_serving_l470_47001

/-- Proves that the number of cups of vegetables in one serving of soup is 1 -/
theorem vegetables_in_soup_serving (V : ℝ) : V = 1 :=
  by
  -- One serving contains V cups of vegetables and 2.5 cups of broth
  have h1 : V + 2.5 = (14 * 2) / 8 := by sorry
  -- 8 servings require 14 pints of vegetables and broth combined
  -- 1 pint = 2 cups
  -- So, 14 pints = 14 * 2 cups = 28 cups
  -- Solve the equation: 8 * (V + 2.5) = 28
  sorry

end vegetables_in_soup_serving_l470_47001


namespace expansion_contains_2017_l470_47011

/-- The first term in the expansion of n^3 -/
def first_term (n : ℕ) : ℕ := n^2 - n + 1

/-- The last term in the expansion of n^3 -/
def last_term (n : ℕ) : ℕ := n^2 + n - 1

/-- The sum of n consecutive odd numbers starting from the first term -/
def sum_expansion (n : ℕ) : ℕ := n * (first_term n + last_term n) / 2

theorem expansion_contains_2017 :
  ∃ (n : ℕ), n = 45 ∧ 
  first_term n ≤ 2017 ∧ 
  2017 ≤ last_term n ∧ 
  sum_expansion n = n^3 :=
sorry

end expansion_contains_2017_l470_47011


namespace negation_of_forall_even_square_plus_self_l470_47079

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

theorem negation_of_forall_even_square_plus_self :
  (¬ ∀ n : ℕ, is_even (n^2 + n)) ↔ (∃ x : ℕ, ¬ is_even (x^2 + x)) :=
sorry

end negation_of_forall_even_square_plus_self_l470_47079


namespace club_size_l470_47059

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 3

/-- The cost of a jersey in dollars -/
def jersey_cost : ℕ := sock_cost + 7

/-- The cost of a warm-up jacket in dollars -/
def jacket_cost : ℕ := 2 * jersey_cost

/-- The total cost for one player's equipment in dollars -/
def player_cost : ℕ := 2 * (sock_cost + jersey_cost) + jacket_cost

/-- The total expenditure for the club in dollars -/
def total_expenditure : ℕ := 3276

/-- The number of players in the club -/
def num_players : ℕ := total_expenditure / player_cost

theorem club_size :
  num_players = 71 :=
sorry

end club_size_l470_47059


namespace fence_length_for_specific_yard_l470_47085

/-- A rectangular yard with given dimensions and area -/
structure RectangularYard where
  length : ℝ
  width : ℝ
  area : ℝ
  length_positive : 0 < length
  width_positive : 0 < width
  area_eq : area = length * width

/-- The fence length for a rectangular yard -/
def fence_length (yard : RectangularYard) : ℝ :=
  2 * yard.width + yard.length

/-- Theorem: For a rectangular yard with one side of 40 feet and an area of 240 square feet,
    the fence length (perimeter minus one side) is 52 feet -/
theorem fence_length_for_specific_yard :
  ∃ (yard : RectangularYard),
    yard.length = 40 ∧
    yard.area = 240 ∧
    fence_length yard = 52 := by
  sorry


end fence_length_for_specific_yard_l470_47085


namespace anna_transportation_tax_l470_47009

/-- Calculates the transportation tax for a vehicle -/
def calculate_tax (engine_power : ℕ) (tax_rate : ℕ) (months_owned : ℕ) (months_in_year : ℕ) : ℕ :=
  (engine_power * tax_rate * months_owned) / months_in_year

/-- Represents the transportation tax problem for Anna Ivanovna -/
theorem anna_transportation_tax :
  let engine_power : ℕ := 250
  let tax_rate : ℕ := 75
  let months_owned : ℕ := 2
  let months_in_year : ℕ := 12
  calculate_tax engine_power tax_rate months_owned months_in_year = 3125 := by
  sorry


end anna_transportation_tax_l470_47009


namespace vector_problem_l470_47058

theorem vector_problem (α : Real) 
  (h1 : α ∈ Set.Ioo (3*π/2) (2*π))
  (h2 : (3*Real.sin α)*(2*Real.sin α) + (Real.cos α)*(5*Real.sin α - 4*Real.cos α) = 0) :
  Real.tan α = -4/3 ∧ Real.cos (α/2 + π/3) = -(2*Real.sqrt 5 + Real.sqrt 15)/10 := by
  sorry

end vector_problem_l470_47058


namespace stamp_collection_problem_l470_47094

theorem stamp_collection_problem (current_stamps : ℕ) : 
  (current_stamps : ℚ) * (1 + 20 / 100) = 48 → current_stamps = 40 := by
  sorry

end stamp_collection_problem_l470_47094


namespace birds_left_l470_47025

theorem birds_left (initial_chickens ducks turkeys chickens_sold : ℕ) :
  initial_chickens ≥ chickens_sold →
  (initial_chickens - chickens_sold + ducks + turkeys : ℕ) =
    initial_chickens + ducks + turkeys - chickens_sold :=
by sorry

end birds_left_l470_47025


namespace unique_prime_solution_l470_47051

theorem unique_prime_solution :
  ∀ p m : ℕ,
    p.Prime →
    m > 0 →
    p * (p + m) + p = (m + 1)^3 →
    p = 2 ∧ m = 1 := by
  sorry

end unique_prime_solution_l470_47051


namespace min_operations_for_jugs_l470_47091

/-- Represents the state of the two jugs -/
structure JugState :=
  (jug7 : ℕ)
  (jug5 : ℕ)

/-- Represents an operation on the jugs -/
inductive Operation
  | Fill7
  | Fill5
  | Empty7
  | Empty5
  | Pour7to5
  | Pour5to7

/-- Applies an operation to a JugState -/
def applyOperation (state : JugState) (op : Operation) : JugState :=
  match op with
  | Operation.Fill7 => ⟨7, state.jug5⟩
  | Operation.Fill5 => ⟨state.jug7, 5⟩
  | Operation.Empty7 => ⟨0, state.jug5⟩
  | Operation.Empty5 => ⟨state.jug7, 0⟩
  | Operation.Pour7to5 => 
      let amount := min state.jug7 (5 - state.jug5)
      ⟨state.jug7 - amount, state.jug5 + amount⟩
  | Operation.Pour5to7 => 
      let amount := min state.jug5 (7 - state.jug7)
      ⟨state.jug7 + amount, state.jug5 - amount⟩

/-- Checks if a sequence of operations results in the desired state -/
def isValidSolution (ops : List Operation) : Prop :=
  let finalState := ops.foldl applyOperation ⟨0, 0⟩
  finalState.jug7 = 1 ∧ finalState.jug5 = 1

/-- The main theorem to be proved -/
theorem min_operations_for_jugs : 
  ∃ (ops : List Operation), isValidSolution ops ∧ ops.length = 42 ∧
  (∀ (other_ops : List Operation), isValidSolution other_ops → other_ops.length ≥ 42) :=
sorry

end min_operations_for_jugs_l470_47091


namespace proposition_1_proposition_4_proposition_5_l470_47023

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations and operations
variable (contains : Plane → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (point_not_on_line : Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Proposition 1
theorem proposition_1 (l m : Line) (α : Plane) :
  contains α m → contains α l → point_not_on_line m → skew l m :=
sorry

-- Proposition 4
theorem proposition_4 (l m : Line) (α : Plane) :
  line_perpendicular_plane m α → line_parallel_plane l α → perpendicular l m :=
sorry

-- Proposition 5
theorem proposition_5 (m n : Line) (α β : Plane) :
  skew m n → contains α m → line_parallel_plane m β → 
  contains β n → line_parallel_plane n α → parallel_plane α β :=
sorry

end proposition_1_proposition_4_proposition_5_l470_47023


namespace parabola_shift_l470_47024

def original_parabola (x : ℝ) : ℝ := -2 * x^2 + 4

def shifted_parabola (x : ℝ) : ℝ := -2 * (x + 2)^2 + 7

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 2) + 3 :=
by
  sorry

end parabola_shift_l470_47024


namespace largest_four_digit_divisible_by_12_l470_47096

theorem largest_four_digit_divisible_by_12 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 12 = 0 → n ≤ 9996 :=
by sorry

end largest_four_digit_divisible_by_12_l470_47096


namespace perimeter_is_96_l470_47000

/-- A figure composed of perpendicular line segments -/
structure PerpendicularFigure where
  x : ℝ
  y : ℝ
  area : ℝ
  x_eq_2y : x = 2 * y
  area_eq_252 : area = 252

/-- The perimeter of the perpendicular figure -/
def perimeter (f : PerpendicularFigure) : ℝ :=
  16 * f.y

theorem perimeter_is_96 (f : PerpendicularFigure) : perimeter f = 96 := by
  sorry

end perimeter_is_96_l470_47000


namespace room_length_calculation_l470_47068

/-- Given a room with width 4 meters and a paving cost of 750 per square meter,
    if the total cost of paving is 16500, then the length of the room is 5.5 meters. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (length : ℝ) :
  width = 4 →
  cost_per_sqm = 750 →
  total_cost = 16500 →
  length * width * cost_per_sqm = total_cost →
  length = 5.5 := by
  sorry

#check room_length_calculation

end room_length_calculation_l470_47068


namespace unique_solution_l470_47043

/-- A discrete random variable with three possible values -/
structure DiscreteRV where
  p₁ : ℝ
  p₂ : ℝ
  p₃ : ℝ
  sum_to_one : p₁ + p₂ + p₃ = 1
  nonnegative : 0 ≤ p₁ ∧ 0 ≤ p₂ ∧ 0 ≤ p₃

/-- The expected value of X -/
def expectation (X : DiscreteRV) : ℝ := -X.p₁ + X.p₃

/-- The expected value of X² -/
def expectation_squared (X : DiscreteRV) : ℝ := X.p₁ + X.p₃

/-- Theorem stating the unique solution for the given conditions -/
theorem unique_solution (X : DiscreteRV) 
  (h₁ : expectation X = 0.1) 
  (h₂ : expectation_squared X = 0.9) : 
  X.p₁ = 0.4 ∧ X.p₂ = 0.1 ∧ X.p₃ = 0.5 := by
  sorry

end unique_solution_l470_47043


namespace seat_difference_l470_47070

/-- Represents the seating configuration of a bus --/
structure BusSeating where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeat : ℕ
  seatCapacity : ℕ
  totalCapacity : ℕ

/-- Theorem stating the difference in seats between left and right sides --/
theorem seat_difference (bus : BusSeating) : 
  bus.leftSeats = 15 →
  bus.backSeat = 12 →
  bus.seatCapacity = 3 →
  bus.totalCapacity = 93 →
  bus.leftSeats - bus.rightSeats = 3 := by
  sorry

#check seat_difference

end seat_difference_l470_47070


namespace at_least_one_outstanding_equiv_l470_47099

/-- Represents whether a person is an outstanding student -/
def IsOutstandingStudent (person : Prop) : Prop := person

/-- The statement "At least one of person A and person B is an outstanding student" -/
def AtLeastOneOutstanding (A B : Prop) : Prop :=
  IsOutstandingStudent A ∨ IsOutstandingStudent B

theorem at_least_one_outstanding_equiv (A B : Prop) :
  AtLeastOneOutstanding A B ↔ (IsOutstandingStudent A ∨ IsOutstandingStudent B) :=
sorry

end at_least_one_outstanding_equiv_l470_47099


namespace parallel_vectors_x_value_l470_47005

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two vectors a and b, where a = (1, 2) and b = (2x, -3),
    if a is parallel to b, then x = 3 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2 * x, -3)
  parallel a b → x = 3 := by
sorry

end parallel_vectors_x_value_l470_47005


namespace negation_p_iff_valid_range_l470_47054

/-- The proposition p: There exists x₀ ∈ ℝ such that x₀² + ax₀ + a < 0 -/
def p (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + a*x₀ + a < 0

/-- The range of a for which ¬p holds -/
def valid_range (a : ℝ) : Prop := a ≤ 0 ∨ a ≥ 4

theorem negation_p_iff_valid_range (a : ℝ) :
  ¬(p a) ↔ valid_range a := by sorry

end negation_p_iff_valid_range_l470_47054


namespace ellipse_eccentricity_l470_47095

/-- The eccentricity of an ellipse with equation x²/3 + y²/9 = 1 is √6/3 -/
theorem ellipse_eccentricity : 
  let a : ℝ := 3
  let b : ℝ := Real.sqrt 3
  let e : ℝ := Real.sqrt (a^2 - b^2) / a
  e = Real.sqrt 6 / 3 := by sorry

end ellipse_eccentricity_l470_47095


namespace food_company_inspection_l470_47092

theorem food_company_inspection (large_companies medium_companies total_inspected medium_inspected : ℕ) 
  (h1 : large_companies = 4)
  (h2 : medium_companies = 20)
  (h3 : total_inspected = 40)
  (h4 : medium_inspected = 5) :
  ∃ (small_companies : ℕ), 
    small_companies = 136 ∧ 
    total_inspected = large_companies + medium_inspected + (total_inspected - large_companies - medium_inspected) :=
by sorry

end food_company_inspection_l470_47092


namespace triangle_rectangle_perimeter_equality_l470_47072

/-- The perimeter of an isosceles triangle with two sides of 12 cm and one side of 14 cm 
    is equal to the perimeter of a rectangle with width 8 cm and length x cm. -/
theorem triangle_rectangle_perimeter_equality (x : ℝ) : 
  (12 : ℝ) + 12 + 14 = 2 * (x + 8) → x = 11 := by
  sorry

end triangle_rectangle_perimeter_equality_l470_47072


namespace problem_1_problem_2_l470_47040

-- Define the logarithm function
noncomputable def lg (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem 1
theorem problem_1 : (lg 2 2)^2 + (lg 2 2) * (lg 2 5) + (lg 2 5) = 1 := by sorry

-- Theorem 2
theorem problem_2 : (2^(1/3) * 3^(1/2))^6 - 8 * (16/49)^(-1/2) - 2^(1/4) * 8^0.25 - (-2016)^0 = 91 := by sorry

end problem_1_problem_2_l470_47040


namespace A_div_B_equals_37_l470_47039

-- Define the series A
def A : ℝ := sorry

-- Define the series B
def B : ℝ := sorry

-- Theorem statement
theorem A_div_B_equals_37 : A / B = 37 := by sorry

end A_div_B_equals_37_l470_47039


namespace quadratic_inequality_range_l470_47032

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 2 > 0) ↔ (0 ≤ a ∧ a < 8) :=
sorry

end quadratic_inequality_range_l470_47032


namespace functional_equation_solution_l470_47030

def MonotonousFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ ∀ x y, x ≤ y → f x ≥ f y

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_mono : MonotonousFunction f)
  (h_eq : ∀ x, f (f x) = f (-f x) ∧ f (f x) = (f x)^2) :
  (∀ x, f x = 0) ∨ (∀ x, f x = 1) :=
sorry

end functional_equation_solution_l470_47030


namespace carton_height_proof_l470_47086

/-- Given a carton and soap boxes with specific dimensions, prove the height of the carton. -/
theorem carton_height_proof (carton_length carton_width carton_height : ℝ)
  (box_length box_width box_height : ℝ) (max_boxes : ℕ) :
  carton_length = 25 ∧ 
  carton_width = 48 ∧
  box_length = 8 ∧
  box_width = 6 ∧
  box_height = 5 ∧
  max_boxes = 300 ∧
  (carton_length * carton_width * carton_height) = 
    (↑max_boxes * box_length * box_width * box_height) →
  carton_height = 60 := by
sorry

end carton_height_proof_l470_47086


namespace melissa_initial_oranges_l470_47022

/-- The number of oranges Melissa has initially -/
def initial_oranges : ℕ := sorry

/-- The number of oranges John takes away -/
def oranges_taken : ℕ := 19

/-- The number of oranges Melissa has left -/
def oranges_left : ℕ := 51

/-- Theorem stating that Melissa's initial number of oranges is 70 -/
theorem melissa_initial_oranges : 
  initial_oranges = oranges_taken + oranges_left :=
sorry

end melissa_initial_oranges_l470_47022


namespace tiles_along_width_l470_47053

theorem tiles_along_width (area : ℝ) (tile_size : ℝ) : 
  area = 360 → tile_size = 9 → (8 : ℝ) * Real.sqrt 5 = (12 * Real.sqrt (area / 2)) / tile_size := by
  sorry

end tiles_along_width_l470_47053


namespace ellipse_iff_m_range_l470_47010

-- Define the equation
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (2 + m) - y^2 / (m + 1) = 1

-- Define the condition for the equation to represent an ellipse
def is_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), 
    ellipse_equation x y m ↔ x^2 / a^2 + y^2 / b^2 = 1

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  (m > -2 ∧ m < -3/2) ∨ (m > -3/2 ∧ m < -1)

-- State the theorem
theorem ellipse_iff_m_range :
  ∀ m : ℝ, is_ellipse m ↔ m_range m :=
sorry

end ellipse_iff_m_range_l470_47010


namespace sum_smallest_largest_prime_1_to_25_l470_47067

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def smallestPrimeBetween1And25 : ℕ := 2

def largestPrimeBetween1And25 : ℕ := 23

theorem sum_smallest_largest_prime_1_to_25 :
  isPrime smallestPrimeBetween1And25 ∧
  isPrime largestPrimeBetween1And25 ∧
  (∀ n : ℕ, 1 < n → n < 25 → isPrime n → smallestPrimeBetween1And25 ≤ n) ∧
  (∀ n : ℕ, 1 < n → n < 25 → isPrime n → n ≤ largestPrimeBetween1And25) →
  smallestPrimeBetween1And25 + largestPrimeBetween1And25 = 25 :=
by sorry

end sum_smallest_largest_prime_1_to_25_l470_47067


namespace derivative_f_at_zero_l470_47036

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then
    Real.sqrt (1 + Real.log (1 + 3 * x^2 * Real.cos (2 / x))) - 1
  else
    0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end derivative_f_at_zero_l470_47036


namespace percentage_of_democrat_voters_l470_47028

theorem percentage_of_democrat_voters (d r : ℝ) : 
  d + r = 100 →
  0.65 * d + 0.2 * r = 47 →
  d = 60 :=
by sorry

end percentage_of_democrat_voters_l470_47028


namespace equation_solution_l470_47061

open Real

theorem equation_solution (x : ℝ) :
  (sin x ≠ 0) →
  (cos x ≠ 0) →
  (sin x + cos x ≥ 0) →
  (Real.sqrt (1 + tan x) = sin x + cos x) ↔
  (∃ n : ℤ, (x = π/4 + 2*π*↑n) ∨ (x = -π/4 + 2*π*↑n) ∨ (x = 3*π/4 + 2*π*↑n)) :=
by sorry

end equation_solution_l470_47061


namespace symmetry_axis_property_l470_47048

/-- Given a function f(x) = 3sin(x) + 4cos(x), if x = θ is an axis of symmetry
    for the curve y = f(x), then cos(2θ) + sin(θ)cos(θ) = 19/25 -/
theorem symmetry_axis_property (θ : ℝ) :
  (∀ x, 3 * Real.sin x + 4 * Real.cos x = 3 * Real.sin (2 * θ - x) + 4 * Real.cos (2 * θ - x)) →
  Real.cos (2 * θ) + Real.sin θ * Real.cos θ = 19 / 25 := by
  sorry

end symmetry_axis_property_l470_47048


namespace ellipse_proof_hyperbola_proof_l470_47078

-- Ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 20 = 1

theorem ellipse_proof (major_axis_length : ℝ) (eccentricity : ℝ) 
  (h1 : major_axis_length = 12) 
  (h2 : eccentricity = 2/3) : 
  ∀ x y : ℝ, ellipse_equation x y ↔ 
    ∃ a b : ℝ, a^2 * y^2 + b^2 * x^2 = a^2 * b^2 ∧ 
    2 * a = major_axis_length ∧ 
    (a^2 - b^2) / a^2 = eccentricity^2 :=
sorry

-- Hyperbola
def original_hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

def new_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 24 = 1

theorem hyperbola_proof :
  ∀ x y : ℝ, new_hyperbola x y ↔ 
    (∃ c : ℝ, (∀ x₀ y₀ : ℝ, original_hyperbola x₀ y₀ → 
      (x₀ - c)^2 - y₀^2 = c^2 ∧ (x₀ + c)^2 - y₀^2 = c^2) ∧
    new_hyperbola (-Real.sqrt 5 / 2) (-Real.sqrt 6)) :=
sorry

end ellipse_proof_hyperbola_proof_l470_47078


namespace combined_cost_increase_percentage_l470_47003

def bicycle_initial_cost : ℝ := 200
def skates_initial_cost : ℝ := 50
def bicycle_increase_rate : ℝ := 0.06
def skates_increase_rate : ℝ := 0.15

theorem combined_cost_increase_percentage :
  let bicycle_new_cost := bicycle_initial_cost * (1 + bicycle_increase_rate)
  let skates_new_cost := skates_initial_cost * (1 + skates_increase_rate)
  let initial_total_cost := bicycle_initial_cost + skates_initial_cost
  let new_total_cost := bicycle_new_cost + skates_new_cost
  (new_total_cost - initial_total_cost) / initial_total_cost = 0.078 := by
  sorry

end combined_cost_increase_percentage_l470_47003


namespace geometric_sequence_property_l470_47081

theorem geometric_sequence_property (a b c q : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ 
    b + c - a = (a + b + c) * q ∧
    c + a - b = (a + b + c) * q^2 ∧
    a + b - c = (a + b + c) * q^3) →
  q^3 + q^2 + q = 1 := by
sorry

end geometric_sequence_property_l470_47081


namespace omega_value_l470_47037

theorem omega_value (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.sin (ω * x + π / 6)) →
  ω > 0 →
  (∀ x y, 0 < x ∧ x < y ∧ y < π / 3 → f x < f y) →
  f (π / 4) = f (π / 2) →
  ω = 8 / 9 := by
sorry

end omega_value_l470_47037


namespace A_l470_47098

def A' : ℕ → ℕ → ℕ → ℕ
  | 0, n, k => n + k
  | m+1, 0, k => A' m k 1
  | m+1, n+1, k => A' m (A' (m+1) n k) k

theorem A'_3_2_2 : A' 3 2 2 = 17 := by sorry

end A_l470_47098


namespace min_value_theorem_l470_47080

theorem min_value_theorem (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (z w : ℝ), z^2 + w^2 = 2 → |z| ≠ |w| →
    1 / (z + w)^2 + 1 / (z - w)^2 ≥ m :=
by sorry

end min_value_theorem_l470_47080


namespace soccer_goals_proof_l470_47083

def goals_first_6 : List Nat := [5, 2, 4, 3, 6, 2]

def total_goals_6 : Nat := goals_first_6.sum

theorem soccer_goals_proof (goals_7 goals_8 : Nat) : 
  goals_7 < 7 →
  goals_8 < 7 →
  (total_goals_6 + goals_7) % 7 = 0 →
  (total_goals_6 + goals_7 + goals_8) % 8 = 0 →
  goals_7 * goals_8 = 24 := by
  sorry

#eval total_goals_6

end soccer_goals_proof_l470_47083


namespace fraction_equality_l470_47075

theorem fraction_equality : (1998 - 998) / 1000 = 1 := by
  sorry

end fraction_equality_l470_47075


namespace sues_shoe_probability_l470_47060

/-- Represents the number of pairs of shoes for each color --/
structure ShoePairs where
  black : ℕ
  brown : ℕ
  gray : ℕ

/-- Calculates the probability of selecting two shoes of the same color,
    one left and one right, given the number of pairs for each color --/
def samePairColorProbability (pairs : ShoePairs) : ℚ :=
  let totalShoes := 2 * (pairs.black + pairs.brown + pairs.gray)
  let blackProb := (2 * pairs.black) * pairs.black / (totalShoes * (totalShoes - 1))
  let brownProb := (2 * pairs.brown) * pairs.brown / (totalShoes * (totalShoes - 1))
  let grayProb := (2 * pairs.gray) * pairs.gray / (totalShoes * (totalShoes - 1))
  blackProb + brownProb + grayProb

/-- Theorem stating that for Sue's shoe collection, the probability of
    selecting two shoes of the same color, one left and one right, is 7/33 --/
theorem sues_shoe_probability :
  samePairColorProbability ⟨6, 3, 2⟩ = 7 / 33 := by
  sorry

end sues_shoe_probability_l470_47060


namespace f_properties_l470_47020

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem f_properties :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₁ < -1 ∧ x₂ < -1 → f x₁ > f x₂) ∧
  (∀ x₁ x₂, -1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (f (-1) = -(1 / Real.exp 1)) ∧
  (∀ x, f x ≥ -(1 / Real.exp 1)) ∧
  (∀ y : ℝ, ∃ x, f x > y) ∧
  (∃ a : ℝ, a ≥ -2 ∧
    ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ →
      (f x₂ - f a) / (x₂ - a) > (f x₁ - f a) / (x₁ - a)) :=
sorry

end f_properties_l470_47020


namespace circle_tangent_relation_l470_47044

/-- Two circles with radii R₁ and R₂ are externally tangent. A line of length d is perpendicular to their common tangent. -/
structure CircleConfiguration where
  R₁ : ℝ
  R₂ : ℝ
  d : ℝ
  R₁_pos : 0 < R₁
  R₂_pos : 0 < R₂
  d_pos : 0 < d
  externally_tangent : R₁ + R₂ > 0

/-- The relation between the radii of two externally tangent circles and the length of a line perpendicular to their common tangent. -/
theorem circle_tangent_relation (c : CircleConfiguration) :
  1 / c.R₁ + 1 / c.R₂ = 2 / c.d := by
  sorry

end circle_tangent_relation_l470_47044


namespace silk_order_total_l470_47056

/-- Calculates the total yards of silk dyed given the yards of each color and the percentage of red silk -/
def total_silk_dyed (green pink blue yellow : ℝ) (red_percent : ℝ) : ℝ :=
  let non_red := green + pink + blue + yellow
  let red := red_percent * non_red
  non_red + red

/-- Theorem stating the total yards of silk dyed for the given order -/
theorem silk_order_total :
  total_silk_dyed 61921 49500 75678 34874.5 0.1 = 245270.85 := by
  sorry

end silk_order_total_l470_47056


namespace route_ratio_is_three_l470_47074

-- Define the grid structure
structure Grid where
  -- Add necessary fields to represent the grid

-- Define a function to count routes
def countRoutes (g : Grid) (start : Nat × Nat) (steps : Nat) : Nat :=
  sorry

-- Define points A and B
def pointA : Nat × Nat := sorry
def pointB : Nat × Nat := sorry

-- Define the specific grid
def specificGrid : Grid := sorry

-- Theorem statement
theorem route_ratio_is_three :
  let m := countRoutes specificGrid pointA 4
  let n := countRoutes specificGrid pointB 4
  n / m = 3 := by sorry

end route_ratio_is_three_l470_47074


namespace division_problem_l470_47017

theorem division_problem (dividend quotient remainder : ℕ) (h1 : dividend = 1375) (h2 : quotient = 20) (h3 : remainder = 55) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 66 := by
sorry

end division_problem_l470_47017


namespace no_integer_a_with_one_integer_solution_l470_47073

theorem no_integer_a_with_one_integer_solution :
  ¬ ∃ (a : ℤ), ∃! (x : ℤ), x^3 - a*x^2 - 6*a*x + a^2 - 3 = 0 :=
by sorry

end no_integer_a_with_one_integer_solution_l470_47073


namespace distinct_colorings_tetrahedron_l470_47033

-- Define the number of colors
def num_colors : ℕ := 4

-- Define the symmetry group size of a tetrahedron
def symmetry_group_size : ℕ := 12

-- Define the number of vertices in a tetrahedron
def num_vertices : ℕ := 4

-- Define the number of colorings fixed by identity rotation
def fixed_by_identity : ℕ := num_colors ^ num_vertices

-- Define the number of colorings fixed by 180° rotations
def fixed_by_180_rotation : ℕ := num_colors ^ 2

-- Define the number of 180° rotations
def num_180_rotations : ℕ := 3

-- Theorem statement
theorem distinct_colorings_tetrahedron :
  (fixed_by_identity + num_180_rotations * fixed_by_180_rotation) / symmetry_group_size = 36 :=
by sorry

end distinct_colorings_tetrahedron_l470_47033


namespace jerry_pool_time_l470_47014

/-- Represents the time spent in the pool by each person --/
structure PoolTime where
  jerry : ℝ
  elaine : ℝ
  george : ℝ
  kramer : ℝ

/-- The conditions of the problem --/
def poolConditions (t : PoolTime) : Prop :=
  t.elaine = 2 * t.jerry ∧
  t.george = (1/3) * t.elaine ∧
  t.kramer = 0 ∧
  t.jerry + t.elaine + t.george + t.kramer = 11

/-- The theorem to be proved --/
theorem jerry_pool_time (t : PoolTime) :
  poolConditions t → t.jerry = 3 :=
by
  sorry

end jerry_pool_time_l470_47014


namespace intersection_properties_l470_47064

/-- Two lines intersecting at point P -/
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x - y - 3 * m + 1 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := x + m * y - 3 * m - 1 = 0

/-- Circle C -/
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4

/-- Point P satisfies both lines -/
def point_P (m : ℝ) (x y : ℝ) : Prop := line1 m x y ∧ line2 m x y

/-- AB is a chord of circle C with length 2√3 -/
def chord_AB (xa ya xb yb : ℝ) : Prop :=
  circle_C xa ya ∧ circle_C xb yb ∧ (xa - xb)^2 + (ya - yb)^2 = 12

/-- Q is the midpoint of AB -/
def midpoint_Q (xa ya xb yb xq yq : ℝ) : Prop :=
  xq = (xa + xb) / 2 ∧ yq = (ya + yb) / 2

theorem intersection_properties (m : ℝ) :
  ∃ x y xa ya xb yb xq yq,
    point_P m x y ∧
    chord_AB xa ya xb yb ∧
    midpoint_Q xa ya xb yb xq yq →
    (¬ circle_C x y) ∧  -- P lies outside circle C
    (∃ pq_max, pq_max = 6 + Real.sqrt 2 ∧
      ∀ x' y', point_P m x' y' →
        ∀ xa' ya' xb' yb' xq' yq',
          chord_AB xa' ya' xb' yb' ∧
          midpoint_Q xa' ya' xb' yb' xq' yq' →
            ((x' - xq')^2 + (y' - yq')^2)^(1/2) ≤ pq_max) ∧  -- Max length of PQ
    (∃ pa_pb_min, pa_pb_min = 15 - 8 * Real.sqrt 2 ∧
      ∀ x' y', point_P m x' y' →
        ∀ xa' ya' xb' yb',
          chord_AB xa' ya' xb' yb' →
            (x' - xa') * (x' - xb') + (y' - ya') * (y' - yb') ≥ pa_pb_min)  -- Min value of PA · PB
  := by sorry

end intersection_properties_l470_47064


namespace joan_seashells_l470_47041

/-- The number of seashells Joan has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Joan has 16 seashells after giving away 63 from her initial 79 -/
theorem joan_seashells : remaining_seashells 79 63 = 16 := by
  sorry

end joan_seashells_l470_47041


namespace cubic_expression_evaluation_l470_47052

theorem cubic_expression_evaluation : 101^3 + 3*(101^2) - 3*101 + 9 = 1060610 := by
  sorry

end cubic_expression_evaluation_l470_47052


namespace integral_x_plus_exp_x_l470_47049

theorem integral_x_plus_exp_x : ∫ x in (0:ℝ)..2, (x + Real.exp x) = Real.exp 2 + 1 := by sorry

end integral_x_plus_exp_x_l470_47049


namespace tangent_circle_center_and_radius_l470_47082

/-- A circle tangent to y=x, y=-x, and y=10 with center above (10,10) -/
structure TangentCircle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_gt_ten : h > 10
  k_gt_ten : k > 10
  tangent_y_eq_x : r = |h - k| / Real.sqrt 2
  tangent_y_eq_neg_x : r = |h + k| / Real.sqrt 2
  tangent_y_eq_ten : r = k - 10

/-- The center and radius of a circle tangent to y=x, y=-x, and y=10 -/
theorem tangent_circle_center_and_radius (c : TangentCircle) :
  c.h = 10 + (1 + Real.sqrt 2) * c.r ∧ c.k = 10 + c.r :=
by sorry

end tangent_circle_center_and_radius_l470_47082


namespace point_below_line_range_l470_47047

/-- Given a point (-2,t) located below the line 2x-3y+6=0, prove that the range of t is (-∞, 2/3) -/
theorem point_below_line_range (t : ℝ) : 
  (2 * (-2) - 3 * t + 6 > 0) → (t < 2/3) :=
by sorry

end point_below_line_range_l470_47047


namespace decimal_to_binary_and_remainder_l470_47097

def decimal_to_binary (n : ℕ) : List Bool :=
  sorry

def binary_to_decimal (b : List Bool) : ℕ :=
  sorry

def binary_division_remainder (dividend : List Bool) (divisor : List Bool) : List Bool :=
  sorry

theorem decimal_to_binary_and_remainder : 
  let binary_126 := decimal_to_binary 126
  let remainder := binary_division_remainder binary_126 [true, false, true, true]
  binary_126 = [true, true, true, true, true, true, false] ∧ 
  remainder = [true, false, false, true] :=
by sorry

end decimal_to_binary_and_remainder_l470_47097


namespace no_intersection_l470_47016

theorem no_intersection : ¬∃ x : ℝ, |3 * x + 6| = -2 * |2 * x - 1| := by
  sorry

end no_intersection_l470_47016


namespace smallest_common_factor_l470_47093

theorem smallest_common_factor (n : ℕ) : 
  (∀ k < 60, ¬ ∃ m > 1, m ∣ (11 * k - 6) ∧ m ∣ (8 * k + 5)) ∧
  (∃ m > 1, m ∣ (11 * 60 - 6) ∧ m ∣ (8 * 60 + 5)) := by
  sorry

end smallest_common_factor_l470_47093


namespace cut_cube_height_l470_47007

/-- The height of a cube with a corner cut off -/
theorem cut_cube_height : 
  let s : ℝ := 2  -- side length of the original cube
  let triangle_side : ℝ := s * Real.sqrt 2  -- side length of the cut triangle
  let base_area : ℝ := (Real.sqrt 3 / 4) * triangle_side ^ 2  -- area of the cut face
  let pyramid_volume : ℝ := s ^ 3 / 6  -- volume of the cut-off pyramid
  let h : ℝ := pyramid_volume / (base_area / 6)  -- height of the cut-off pyramid
  2 - h = 2 - (2 * Real.sqrt 3) / 3 :=
by sorry

end cut_cube_height_l470_47007


namespace remainder_theorem_polynomial_remainder_l470_47012

def f (x : ℝ) : ℝ := x^5 - 6*x^4 + 11*x^3 + 21*x^2 - 17*x + 10

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := by sorry

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + 84 := by sorry

end remainder_theorem_polynomial_remainder_l470_47012


namespace multiplicative_inverse_484_mod_1123_l470_47084

theorem multiplicative_inverse_484_mod_1123 :
  ∃ (n : ℤ), 0 ≤ n ∧ n < 1123 ∧ (484 * n) % 1123 = 1 :=
by
  use 535
  sorry

end multiplicative_inverse_484_mod_1123_l470_47084


namespace fibSeriesSum_l470_47062

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of the infinite series of Fibonacci numbers divided by powers of 5 -/
noncomputable def fibSeries : ℝ := ∑' n, (fib n : ℝ) / 5^n

/-- The sum of the infinite series of Fibonacci numbers divided by powers of 5 is 5/19 -/
theorem fibSeriesSum : fibSeries = 5 / 19 := by sorry

end fibSeriesSum_l470_47062


namespace exists_number_with_removable_digit_l470_47076

-- Define a function to check if a number has a non-zero digit
def has_nonzero_digit (n : ℕ) : Prop :=
  ∃ (k : ℕ), (n / 10^k) % 10 ≠ 0

-- Define a function to check if a number can be obtained by removing a non-zero digit from another number
def can_remove_nonzero_digit (n n' : ℕ) : Prop :=
  ∃ (k : ℕ), 
    let d := (n / 10^k) % 10
    d ≠ 0 ∧ n' = (n / 10^(k+1)) * 10^k + n % 10^k

theorem exists_number_with_removable_digit (d : ℕ) (hd : d > 0) : 
  ∃ (n : ℕ), 
    n % d = 0 ∧ 
    has_nonzero_digit n ∧ 
    ∃ (n' : ℕ), can_remove_nonzero_digit n n' ∧ n' % d = 0 :=
sorry

end exists_number_with_removable_digit_l470_47076


namespace marble_box_count_l470_47015

theorem marble_box_count (blue : ℕ) (red : ℕ) : 
  red = blue + 12 →
  (blue : ℚ) / (blue + red : ℚ) = 1 / 4 →
  blue + red = 24 :=
by sorry

end marble_box_count_l470_47015


namespace yellow_jelly_bean_probability_l470_47026

theorem yellow_jelly_bean_probability 
  (red_prob : ℝ) 
  (orange_prob : ℝ) 
  (blue_prob : ℝ) 
  (yellow_prob : ℝ)
  (h1 : red_prob = 0.1)
  (h2 : orange_prob = 0.4)
  (h3 : blue_prob = 0.2)
  (h4 : red_prob + orange_prob + blue_prob + yellow_prob = 1) :
  yellow_prob = 0.3 := by
sorry

end yellow_jelly_bean_probability_l470_47026


namespace intersection_of_D_sets_nonempty_l470_47069

def D (n : ℕ) : Set ℕ :=
  {x | ∃ a b : ℕ, a * b = n ∧ a > b ∧ b > 0 ∧ x = a - b}

theorem intersection_of_D_sets_nonempty (k : ℕ) (hk : k > 1) :
  ∃ (n : Fin k → ℕ), (∀ i, n i > 1) ∧ 
  (∀ i j, i ≠ j → n i ≠ n j) ∧
  (∃ x y : ℕ, x ≠ y ∧ ∀ i, x ∈ D (n i) ∧ y ∈ D (n i)) :=
sorry

end intersection_of_D_sets_nonempty_l470_47069


namespace baseball_gear_expense_l470_47077

theorem baseball_gear_expense (initial_amount : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 79)
  (h2 : remaining_amount = 32) :
  initial_amount - remaining_amount = 47 := by
  sorry

end baseball_gear_expense_l470_47077


namespace monotonic_cubic_function_implies_m_bound_l470_47087

/-- A function f: ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing. -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∨ (∀ x y : ℝ, x ≤ y → f y ≤ f x)

/-- The main theorem: If f(x) = x³ + x² + mx + 1 is monotonic on ℝ, then m ≥ 1/3. -/
theorem monotonic_cubic_function_implies_m_bound (m : ℝ) :
  Monotonic (fun x : ℝ => x^3 + x^2 + m*x + 1) → m ≥ 1/3 := by
  sorry

end monotonic_cubic_function_implies_m_bound_l470_47087


namespace line_tangent_to_circle_l470_47031

theorem line_tangent_to_circle (a : ℝ) (h1 : a > 0) :
  (∀ y : ℝ, (a - 1)^2 + y^2 = 4) →
  (∀ x y : ℝ, x = a → (x - 1)^2 + y^2 ≥ 4) →
  a = 3 :=
by sorry

end line_tangent_to_circle_l470_47031


namespace weight_of_A_l470_47021

-- Define the weights and ages of persons A, B, C, D, and E
variable (W_A W_B W_C W_D W_E : ℝ)
variable (Age_A Age_B Age_C Age_D Age_E : ℝ)

-- State the conditions from the problem
axiom avg_weight_ABC : (W_A + W_B + W_C) / 3 = 84
axiom avg_age_ABC : (Age_A + Age_B + Age_C) / 3 = 30
axiom avg_weight_ABCD : (W_A + W_B + W_C + W_D) / 4 = 80
axiom avg_age_ABCD : (Age_A + Age_B + Age_C + Age_D) / 4 = 28
axiom avg_weight_BCDE : (W_B + W_C + W_D + W_E) / 4 = 79
axiom avg_age_BCDE : (Age_B + Age_C + Age_D + Age_E) / 4 = 27
axiom weight_E : W_E = W_D + 7
axiom age_E : Age_E = Age_A - 3

-- State the theorem to be proved
theorem weight_of_A : W_A = 79 := by
  sorry

end weight_of_A_l470_47021


namespace pyramid_height_equals_cube_volume_l470_47088

/-- Given a cube with edge length 5 and a square-based pyramid with base edge length 10,
    prove that the height of the pyramid is 3.75 when their volumes are equal. -/
theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) : 
  cube_edge = 5 →
  pyramid_base = 10 →
  (cube_edge ^ 3) = (1 / 3) * (pyramid_base ^ 2) * pyramid_height →
  pyramid_height = 3.75 := by
sorry

end pyramid_height_equals_cube_volume_l470_47088


namespace approximate_0_9915_l470_47066

theorem approximate_0_9915 : 
  ∃ (x : ℚ), (x = 0.956) ∧ 
  (∀ (y : ℚ), abs (y - 0.9915) < abs (x - 0.9915) → abs (y - 0.9915) ≥ 0.0005) :=
by sorry

end approximate_0_9915_l470_47066


namespace sara_cannot_have_two_l470_47050

-- Define the set of cards
def Cards : Finset ℕ := {1, 2, 3, 4}

-- Define the players
inductive Player
| Ben
| Wendy
| Riley
| Sara

-- Define the distribution of cards
def Distribution := Player → ℕ

-- Define the conditions
def ValidDistribution (d : Distribution) : Prop :=
  (∀ p : Player, d p ∈ Cards) ∧
  (∀ p q : Player, p ≠ q → d p ≠ d q) ∧
  (d Player.Ben ≠ 1) ∧
  (d Player.Wendy = d Player.Riley + 1)

-- Theorem statement
theorem sara_cannot_have_two (d : Distribution) :
  ValidDistribution d → d Player.Sara ≠ 2 :=
by sorry

end sara_cannot_have_two_l470_47050


namespace palindrome_count_ratio_l470_47046

/-- A palindrome is a natural number whose decimal representation reads the same from left to right and right to left. -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The sum of digits of a natural number. -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Count of palindromes with even sum of digits between 10,000 and 999,999. -/
def evenSumPalindromeCount : ℕ := sorry

/-- Count of palindromes with odd sum of digits between 10,000 and 999,999. -/
def oddSumPalindromeCount : ℕ := sorry

theorem palindrome_count_ratio :
  evenSumPalindromeCount = 3 * oddSumPalindromeCount := by sorry

end palindrome_count_ratio_l470_47046


namespace hyperbola_eccentricity_l470_47065

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity : ∀ (a b : ℝ), a > 0 → b > 0 →
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ y = Real.sqrt x ∧
  (∃ (m : ℝ), m * (x + 1) = y ∧ m = 1 / (2 * Real.sqrt x))) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c = 1) →
  (a^2 + b^2) / a = (Real.sqrt 5 + 1) / 2 := by
  sorry

end hyperbola_eccentricity_l470_47065


namespace circle_radius_proof_l470_47055

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Theorem: Given the conditions, prove that the radius of circle k is 17 -/
theorem circle_radius_proof (k k1 k2 : Circle)
  (h1 : k1.radius = 8)
  (h2 : k2.radius = 15)
  (h3 : k1.radius < k.radius)
  (h4 : (k.radius ^ 2 - k1.radius ^ 2) * Real.pi = (k2.radius ^ 2 - k.radius ^ 2) * Real.pi) :
  k.radius = 17 := by
  sorry

end circle_radius_proof_l470_47055


namespace alcohol_mixture_proof_l470_47013

/-- Proves that adding 6 liters of 75% alcohol solution to 6 liters of 25% alcohol solution results in a 50% alcohol solution -/
theorem alcohol_mixture_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.25
  let added_volume : ℝ := 6
  let added_concentration : ℝ := 0.75
  let target_concentration : ℝ := 0.50
  let final_volume : ℝ := initial_volume + added_volume
  let final_alcohol_amount : ℝ := initial_volume * initial_concentration + added_volume * added_concentration
  final_alcohol_amount / final_volume = target_concentration := by
  sorry


end alcohol_mixture_proof_l470_47013


namespace total_students_is_600_l470_47057

/-- Represents a school with boys and girls -/
structure School where
  numBoys : ℕ
  numGirls : ℕ
  avgAgeBoys : ℝ
  avgAgeGirls : ℝ
  avgAgeSchool : ℝ

/-- The conditions of the problem -/
def problemSchool : School :=
  { numBoys := 0,  -- We don't know this yet, so we set it to 0
    numGirls := 150,
    avgAgeBoys := 12,
    avgAgeGirls := 11,
    avgAgeSchool := 11.75 }

/-- The theorem stating that the total number of students is 600 -/
theorem total_students_is_600 (s : School) 
  (h1 : s.numGirls = problemSchool.numGirls)
  (h2 : s.avgAgeBoys = problemSchool.avgAgeBoys)
  (h3 : s.avgAgeGirls = problemSchool.avgAgeGirls)
  (h4 : s.avgAgeSchool = problemSchool.avgAgeSchool)
  (h5 : s.avgAgeSchool * (s.numBoys + s.numGirls) = 
        s.avgAgeBoys * s.numBoys + s.avgAgeGirls * s.numGirls) :
  s.numBoys + s.numGirls = 600 := by
  sorry

#check total_students_is_600

end total_students_is_600_l470_47057


namespace basic_algorithm_statements_correct_l470_47089

/-- Represents a type of algorithm statement -/
inductive AlgorithmStatement
  | INPUT
  | PRINT
  | IF_THEN
  | DO
  | END
  | WHILE
  | END_IF

/-- Defines the set of basic algorithm statements -/
def BasicAlgorithmStatements : Set AlgorithmStatement :=
  {AlgorithmStatement.INPUT, AlgorithmStatement.PRINT, AlgorithmStatement.IF_THEN,
   AlgorithmStatement.DO, AlgorithmStatement.WHILE}

/-- Theorem stating that the set of basic algorithm statements is correct -/
theorem basic_algorithm_statements_correct :
  BasicAlgorithmStatements = {AlgorithmStatement.INPUT, AlgorithmStatement.PRINT,
    AlgorithmStatement.IF_THEN, AlgorithmStatement.DO, AlgorithmStatement.WHILE} := by
  sorry

end basic_algorithm_statements_correct_l470_47089


namespace certain_number_proof_l470_47042

theorem certain_number_proof : ∃ x : ℝ, x * 2 + (12 + 4) * (1 / 8) = 602 ∧ x = 300 := by
  sorry

end certain_number_proof_l470_47042


namespace increasing_function_condition_l470_47035

def f (a b x : ℝ) : ℝ := (a + 1) * x + b

theorem increasing_function_condition (a b : ℝ) :
  (∀ x y : ℝ, x < y → f a b x < f a b y) → a > -1 := by
  sorry

end increasing_function_condition_l470_47035


namespace travel_ways_eq_nine_l470_47045

/-- The number of different ways to travel from location A to location B in one day -/
def travel_ways (car_departures train_departures ship_departures : ℕ) : ℕ :=
  car_departures + train_departures + ship_departures

/-- Theorem: The number of different ways to travel is 9 given the specified departures -/
theorem travel_ways_eq_nine :
  travel_ways 3 4 2 = 9 := by
  sorry

end travel_ways_eq_nine_l470_47045


namespace two_greater_than_sqrt_three_l470_47002

theorem two_greater_than_sqrt_three : 2 > Real.sqrt 3 := by
  sorry

end two_greater_than_sqrt_three_l470_47002


namespace inequality_solution_set_l470_47008

theorem inequality_solution_set (x : ℝ) : 
  (abs (2*x - 1) + abs (2*x + 3) < 5) ↔ (-3/2 ≤ x ∧ x < 3/4) :=
sorry

end inequality_solution_set_l470_47008


namespace ice_cream_sundaes_l470_47029

theorem ice_cream_sundaes (total_flavors : ℕ) (h : total_flavors = 8) :
  let vanilla_sundaes := total_flavors - 1
  vanilla_sundaes = 7 :=
by
  sorry

end ice_cream_sundaes_l470_47029


namespace two_people_walking_problem_l470_47018

/-- Two people walking problem -/
theorem two_people_walking_problem (x y : ℝ) : 
  (∃ (distance : ℝ), distance = 18) →
  (∃ (time_meeting : ℝ), time_meeting = 2) →
  (∃ (time_catchup : ℝ), time_catchup = 4) →
  (∃ (time_headstart : ℝ), time_headstart = 1) →
  (2 * x + 2 * y = 18 ∧ 5 * x - 4 * y = 18) := by
sorry

end two_people_walking_problem_l470_47018


namespace oatmeal_cookie_baggies_l470_47019

def total_cookies : ℝ := 41.0
def chocolate_chip_cookies : ℝ := 13.0
def cookies_per_bag : ℝ := 9.0

theorem oatmeal_cookie_baggies :
  ⌊(total_cookies - chocolate_chip_cookies) / cookies_per_bag⌋ = 3 :=
by sorry

end oatmeal_cookie_baggies_l470_47019


namespace jeremy_songs_l470_47027

theorem jeremy_songs (x : ℕ) (h1 : x + (x + 5) = 23) : x + 5 = 14 := by
  sorry

end jeremy_songs_l470_47027


namespace male_rabbits_count_l470_47006

theorem male_rabbits_count (white : ℕ) (black : ℕ) (female : ℕ) 
  (h1 : white = 12) 
  (h2 : black = 9) 
  (h3 : female = 8) : 
  white + black - female = 13 := by
  sorry

end male_rabbits_count_l470_47006


namespace hyperbola_asymptotes_l470_47071

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and the length of the real axis is 4 and the length of the imaginary axis is 6,
    prove that the equation of its asymptotes is y = ±(3/2)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (real_axis : 2 * a = 4) (imag_axis : 2 * b = 6) :
  ∃ (k : ℝ), k = 3/2 ∧ (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → (y = k*x ∨ y = -k*x)) :=
by sorry

end hyperbola_asymptotes_l470_47071


namespace line_passes_through_point_l470_47090

/-- A line in the form y = k(x-1) + 2 always passes through the point (1, 2) -/
theorem line_passes_through_point (k : ℝ) : 
  let f : ℝ → ℝ := λ x => k * (x - 1) + 2
  f 1 = 2 := by
  sorry

end line_passes_through_point_l470_47090
