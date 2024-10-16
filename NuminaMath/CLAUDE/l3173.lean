import Mathlib

namespace NUMINAMATH_CALUDE_first_column_is_seven_l3173_317307

/-- Represents a 5x2 grid with one empty cell -/
def Grid := Fin 9 → Fin 9

/-- The sum of a column in the grid -/
def column_sum (g : Grid) (col : Fin 5) : ℕ :=
  if col = 0 then g 0
  else if col = 1 then g 1 + g 2
  else if col = 2 then g 3 + g 4
  else if col = 3 then g 5 + g 6
  else g 7 + g 8

/-- Predicate for a valid grid arrangement -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j : Fin 9, i ≠ j → g i ≠ g j) ∧
  (∀ col : Fin 4, column_sum g (col + 1) = column_sum g col + 1)

theorem first_column_is_seven (g : Grid) (h : is_valid_grid g) : g 0 = 7 := by
  sorry

end NUMINAMATH_CALUDE_first_column_is_seven_l3173_317307


namespace NUMINAMATH_CALUDE_simplify_expression_l3173_317348

theorem simplify_expression (n : ℕ) : 
  (3^(n+5) - 3*(3^n)) / (3*(3^(n+4))) = 80 / 27 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3173_317348


namespace NUMINAMATH_CALUDE_rational_squares_problem_l3173_317371

theorem rational_squares_problem (m n : ℚ) 
  (h1 : (m + n)^2 = 9) 
  (h2 : (m - n)^2 = 1) : 
  m * n = 2 ∧ m^2 + n^2 - m * n = 3 := by
  sorry

end NUMINAMATH_CALUDE_rational_squares_problem_l3173_317371


namespace NUMINAMATH_CALUDE_point_inside_circle_l3173_317368

theorem point_inside_circle (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) := by
sorry

end NUMINAMATH_CALUDE_point_inside_circle_l3173_317368


namespace NUMINAMATH_CALUDE_pages_copied_for_35_dollars_l3173_317333

/-- Given the cost of copying 5 pages is 7 cents, this theorem proves that
    the number of pages that can be copied for $35 is 2500. -/
theorem pages_copied_for_35_dollars : 
  let cost_per_5_pages : ℚ := 7 / 100  -- 7 cents in dollars
  let dollars : ℚ := 35
  let pages_per_dollar : ℚ := 5 / cost_per_5_pages
  ⌊dollars * pages_per_dollar⌋ = 2500 :=
by sorry

end NUMINAMATH_CALUDE_pages_copied_for_35_dollars_l3173_317333


namespace NUMINAMATH_CALUDE_problem_statement_l3173_317396

theorem problem_statement (x y : ℝ) (h : |x - 3| + Real.sqrt (y - 2) = 0) : 
  (y - x)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3173_317396


namespace NUMINAMATH_CALUDE_trapezoid_properties_l3173_317385

structure Trapezoid where
  EF : ℝ
  GH : ℝ
  EG : ℝ
  FH : ℝ
  height : ℝ

def perimeter (t : Trapezoid) : ℝ := t.EF + t.GH + t.EG + t.FH

theorem trapezoid_properties (t : Trapezoid) 
  (h1 : t.EF = 60)
  (h2 : t.GH = 30)
  (h3 : t.EG = 40)
  (h4 : t.FH = 50)
  (h5 : t.height = 24) :
  perimeter t = 191 ∧ t.EG = 51 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_properties_l3173_317385


namespace NUMINAMATH_CALUDE_vector_addition_l3173_317350

def vector_AB : ℝ × ℝ := (1, 2)
def vector_BC : ℝ × ℝ := (3, 4)

theorem vector_addition :
  let vector_AC := (vector_AB.1 + vector_BC.1, vector_AB.2 + vector_BC.2)
  vector_AC = (4, 6) := by sorry

end NUMINAMATH_CALUDE_vector_addition_l3173_317350


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l3173_317344

theorem inverse_proportion_ordering (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : y₃ = 2 / x₃) 
  (h4 : x₁ < x₂) 
  (h5 : x₂ < 0) 
  (h6 : 0 < x₃) : 
  y₂ < y₁ ∧ y₁ < y₃ := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l3173_317344


namespace NUMINAMATH_CALUDE_vacation_tents_l3173_317305

/-- Calculates the number of tents needed given the total number of people,
    the number of people the house can sleep, and the number of people per tent. -/
def tents_needed (total_people : ℕ) (house_capacity : ℕ) (people_per_tent : ℕ) : ℕ :=
  ((total_people - house_capacity) + (people_per_tent - 1)) / people_per_tent

theorem vacation_tents :
  tents_needed 14 4 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vacation_tents_l3173_317305


namespace NUMINAMATH_CALUDE_toys_in_box_time_l3173_317393

/-- Represents the time taken to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (toys_in_per_cycle : ℕ) (toys_out_per_cycle : ℕ) (cycle_time_seconds : ℕ) : ℚ :=
  let net_toys_per_cycle := toys_in_per_cycle - toys_out_per_cycle
  let full_cycles := (total_toys - toys_in_per_cycle) / net_toys_per_cycle
  let total_seconds := full_cycles * cycle_time_seconds + cycle_time_seconds
  total_seconds / 60

/-- The problem statement -/
theorem toys_in_box_time :
  time_to_put_toys_in_box 50 4 3 45 = 36 := by
  sorry

#eval time_to_put_toys_in_box 50 4 3 45

end NUMINAMATH_CALUDE_toys_in_box_time_l3173_317393


namespace NUMINAMATH_CALUDE_skittles_transfer_l3173_317365

theorem skittles_transfer (bridget_initial : ℕ) (henry_initial : ℕ) : 
  bridget_initial = 4 → henry_initial = 4 → bridget_initial + henry_initial = 8 := by
sorry

end NUMINAMATH_CALUDE_skittles_transfer_l3173_317365


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3173_317357

theorem sum_of_numbers (x y : ℝ) : 
  y = 2 * x - 3 →
  y = 37 →
  x + y = 57 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3173_317357


namespace NUMINAMATH_CALUDE_stratified_sampling_appropriate_l3173_317388

/-- Represents a group of teachers -/
structure TeacherGroup where
  size : ℕ

/-- Represents the entire population of teachers -/
structure TeacherPopulation where
  senior : TeacherGroup
  intermediate : TeacherGroup
  junior : TeacherGroup

/-- Represents a sampling method -/
inductive SamplingMethod
  | Stratified
  | Simple
  | Cluster

/-- States that stratified sampling is appropriate for a population with distinct groups -/
theorem stratified_sampling_appropriate (population : TeacherPopulation) (sample_size : ℕ) :
  SamplingMethod.Stratified = 
    (fun (pop : TeacherPopulation) (s : ℕ) => 
      if pop.senior.size ≠ pop.intermediate.size ∧ 
         pop.senior.size ≠ pop.junior.size ∧ 
         pop.intermediate.size ≠ pop.junior.size
      then SamplingMethod.Stratified
      else SamplingMethod.Simple) population sample_size :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_appropriate_l3173_317388


namespace NUMINAMATH_CALUDE_negative_sqrt_of_squared_negative_five_l3173_317330

theorem negative_sqrt_of_squared_negative_five :
  -Real.sqrt ((-5)^2) = -5 := by sorry

end NUMINAMATH_CALUDE_negative_sqrt_of_squared_negative_five_l3173_317330


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_20_l3173_317335

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_20 :
  units_digit (factorial_sum 20) = units_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4) :=
by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_20_l3173_317335


namespace NUMINAMATH_CALUDE_point_coordinates_l3173_317347

/-- The coordinates of a point A(a,b) satisfying given conditions -/
theorem point_coordinates :
  ∀ (a b : ℝ),
    (|b| = 3) →  -- Distance from A to x-axis is 3
    (|a| = 4) →  -- Distance from A to y-axis is 4
    (a > b) →    -- Given condition a > b
    ((a = 4 ∧ b = -3) ∨ (a = 4 ∧ b = 3)) := by
  sorry


end NUMINAMATH_CALUDE_point_coordinates_l3173_317347


namespace NUMINAMATH_CALUDE_largest_intersection_is_one_l3173_317310

/-- The polynomial function f(x) = x^5 - 5x^4 + 10x^3 - 10x^2 + 5x - b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - b

/-- The linear function g(x) = cx - d -/
def g (c d : ℝ) (x : ℝ) : ℝ := c*x - d

/-- The difference between f and g -/
def h (b c d : ℝ) (x : ℝ) : ℝ := f b x - g c d x

theorem largest_intersection_is_one (b c d : ℝ) :
  (∃ p q r : ℝ, p < q ∧ q < r ∧ 
    (∀ x : ℝ, h b c d x = 0 ↔ x = p ∨ x = q ∨ x = r)) →
  r = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_intersection_is_one_l3173_317310


namespace NUMINAMATH_CALUDE_largest_non_representable_integer_l3173_317382

theorem largest_non_representable_integer : 
  (∀ n > 97, ∃ a b : ℕ, n = 8 * a + 15 * b) ∧ 
  (¬ ∃ a b : ℕ, 97 = 8 * a + 15 * b) := by
sorry

end NUMINAMATH_CALUDE_largest_non_representable_integer_l3173_317382


namespace NUMINAMATH_CALUDE_sausage_pieces_l3173_317319

/-- Given a sausage with red, yellow, and green rings, prove that cutting along all rings results in 21 pieces. -/
theorem sausage_pieces (red_pieces yellow_pieces green_pieces : ℕ) 
  (h_red : red_pieces = 5)
  (h_yellow : yellow_pieces = 7)
  (h_green : green_pieces = 11) :
  red_pieces + yellow_pieces + green_pieces - 2 = 21 :=
by sorry

end NUMINAMATH_CALUDE_sausage_pieces_l3173_317319


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3173_317372

/-- The quadratic inequality mx^2 - mx - 1 < 0 has all real numbers as its solution set -/
def all_reals_solution (m : ℝ) : Prop :=
  ∀ x : ℝ, m * x^2 - m * x - 1 < 0

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := -4 < m ∧ m < 0

/-- Theorem stating that if the quadratic inequality has all real numbers as its solution set,
    then m is in the range (-4, 0) -/
theorem quadratic_inequality_range :
  ∀ m : ℝ, all_reals_solution m → m_range m :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3173_317372


namespace NUMINAMATH_CALUDE_scalene_not_unique_from_two_angles_l3173_317374

-- Define a triangle
structure Triangle :=
  (a b c : ℝ) -- side lengths
  (α β γ : ℝ) -- angles
  (h1 : a > 0 ∧ b > 0 ∧ c > 0) -- positive side lengths
  (h2 : α > 0 ∧ β > 0 ∧ γ > 0) -- positive angles
  (h3 : α + β + γ = π) -- sum of angles is π

-- Define a scalene triangle
def isScalene (t : Triangle) : Prop :=
  t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c

-- Theorem statement
theorem scalene_not_unique_from_two_angles :
  ∃ (t1 t2 : Triangle) (α β : ℝ),
    isScalene t1 ∧ isScalene t2 ∧
    t1.α = α ∧ t1.β = β ∧
    t2.α = α ∧ t2.β = β ∧
    t1 ≠ t2 :=
sorry

end NUMINAMATH_CALUDE_scalene_not_unique_from_two_angles_l3173_317374


namespace NUMINAMATH_CALUDE_possible_values_of_x_l3173_317363

def S (x : ℝ) : Set ℝ := {1, 2, x^2}

theorem possible_values_of_x : {x : ℝ | x ∈ S x} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_possible_values_of_x_l3173_317363


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3173_317304

/-- Proves that the cost of an adult ticket is $5.50 given the specified conditions -/
theorem adult_ticket_cost : 
  let child_ticket_cost : ℝ := 3.50
  let total_tickets : ℕ := 21
  let total_cost : ℝ := 83.50
  let child_tickets : ℕ := 16
  let adult_tickets : ℕ := total_tickets - child_tickets
  let adult_ticket_cost : ℝ := (total_cost - child_ticket_cost * child_tickets) / adult_tickets
  adult_ticket_cost = 5.50 := by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3173_317304


namespace NUMINAMATH_CALUDE_f_4_1981_equals_tower_exp_l3173_317342

/-- A function f : ℕ → ℕ → ℕ satisfying the given recursive conditions -/
noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| x + 1, 0 => f x 1
| x + 1, y + 1 => f x (f (x + 1) y)

/-- Helper function to represent towering exponentiation -/
def tower_exp : ℕ → ℕ → ℕ
| 0, n => n
| m + 1, n => 2^(tower_exp m n)

/-- The main theorem stating that f(4, 1981) is equal to a specific towering exponentiation -/
theorem f_4_1981_equals_tower_exp : 
  f 4 1981 = tower_exp 12 (2^2) :=
sorry

end NUMINAMATH_CALUDE_f_4_1981_equals_tower_exp_l3173_317342


namespace NUMINAMATH_CALUDE_f_properties_l3173_317318

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ 
    ∀ T' : ℝ, T' > 0 ∧ (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  (∀ k : ℤ, ∀ x : ℝ, -π/3 + k * π ≤ x ∧ x ≤ π/6 + k * π → 
    ∀ y : ℝ, -π/3 + k * π ≤ y ∧ y ≤ x → f y ≤ f x) ∧
  (∀ A B C a b c : ℝ, 
    0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
    A + B + C = π ∧
    a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
    (a + 2*c) * Real.cos B = -b * Real.cos A →
    2 < f A ∧ f A ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3173_317318


namespace NUMINAMATH_CALUDE_cubic_inequality_and_fraction_inequality_l3173_317349

theorem cubic_inequality_and_fraction_inequality 
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x^3 + y^3 ≥ x^2*y + x*y^2) ∧ 
  ((x/(y*z) + y/(z*x) + z/(x*y)) ≥ (1/x + 1/y + 1/z)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_and_fraction_inequality_l3173_317349


namespace NUMINAMATH_CALUDE_z_power_2000_eq_one_l3173_317366

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The complex number (3 + 4i) / (4 - 3i) -/
noncomputable def z : ℂ := (3 + 4 * i) / (4 - 3 * i)

/-- Theorem stating that z^2000 = 1 -/
theorem z_power_2000_eq_one : z ^ 2000 = 1 := by sorry

end NUMINAMATH_CALUDE_z_power_2000_eq_one_l3173_317366


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_approx_l3173_317392

theorem sqrt_fraction_sum_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |(Real.sqrt 1.1 / Real.sqrt 0.81) + (Real.sqrt 1.44 / Real.sqrt 0.49) - 2.879| < ε :=
sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_approx_l3173_317392


namespace NUMINAMATH_CALUDE_rita_backstroke_hours_l3173_317301

/-- Calculates the number of backstroke hours completed by Rita --/
def backstroke_hours (total_required : ℕ) (breaststroke : ℕ) (butterfly : ℕ) 
  (freestyle_sidestroke_per_month : ℕ) (months : ℕ) : ℕ :=
  total_required - (breaststroke + butterfly + freestyle_sidestroke_per_month * months)

/-- Theorem stating that Rita completed 50 hours of backstroke --/
theorem rita_backstroke_hours : 
  backstroke_hours 1500 9 121 220 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_rita_backstroke_hours_l3173_317301


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l3173_317317

def U : Set ℕ := {1, 2, 3, 4, 5}

def A : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

def B : Set ℕ := {x ∈ U | ∃ α ∈ A, x = 2*α}

theorem complement_of_A_union_B (h : Set ℕ) : 
  h = U \ (A ∪ B) → h = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l3173_317317


namespace NUMINAMATH_CALUDE_inequality_range_l3173_317376

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m^2 + 4*m - 5)*x^2 - 4*(m - 1)*x + 3 > 0) ↔ 
  (1 ≤ m ∧ m < 19) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3173_317376


namespace NUMINAMATH_CALUDE_four_valid_configurations_l3173_317378

/-- Represents a square piece -/
inductive Square
| A | B | C | D | E | F | G | H

/-- Represents the F-shaped figure -/
structure FShape :=
  (squares : Fin 6 → Unit)

/-- Represents a topless rectangular box -/
structure ToplessBox :=
  (base : Unit)
  (sides : Fin 4 → Unit)

/-- Function to check if a square can be combined with the F-shape to form a topless box -/
def canFormBox (s : Square) (f : FShape) : Prop :=
  ∃ (box : ToplessBox), true  -- Placeholder, actual implementation would be more complex

/-- The main theorem stating that exactly 4 squares can form a topless box with the F-shape -/
theorem four_valid_configurations (squares : Fin 8 → Square) (f : FShape) :
  (∃! (validSquares : Finset Square), 
    validSquares.card = 4 ∧ 
    ∀ s, s ∈ validSquares ↔ canFormBox s f) :=
sorry

end NUMINAMATH_CALUDE_four_valid_configurations_l3173_317378


namespace NUMINAMATH_CALUDE_sum_three_digit_even_numbers_l3173_317397

/-- The sum of all even natural numbers between 100 and 998 (inclusive) is 247050. -/
theorem sum_three_digit_even_numbers : 
  (Finset.range 450).sum (fun i => 100 + 2 * i) = 247050 := by
  sorry

end NUMINAMATH_CALUDE_sum_three_digit_even_numbers_l3173_317397


namespace NUMINAMATH_CALUDE_optimal_selling_price_l3173_317358

-- Define the initial conditions
def initial_purchase_price : ℝ := 10
def initial_selling_price : ℝ := 18
def initial_daily_sales : ℝ := 60

-- Define the price-sales relationships
def price_increase_effect (price_change : ℝ) : ℝ := -5 * price_change
def price_decrease_effect (price_change : ℝ) : ℝ := 10 * price_change

-- Define the profit functions
def profit_function_high (x : ℝ) : ℝ := -5 * (x - 20)^2 + 500
def profit_function_low (x : ℝ) : ℝ := -10 * (x - 17)^2 + 490

-- Theorem statement
theorem optimal_selling_price :
  ∃ (x : ℝ), x = 20 ∧
  ∀ (y : ℝ), y ≥ initial_selling_price →
    profit_function_high y ≤ profit_function_high x ∧
  ∀ (z : ℝ), z < initial_selling_price →
    profit_function_low z ≤ profit_function_high x :=
by sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l3173_317358


namespace NUMINAMATH_CALUDE_telescope_visual_range_l3173_317354

/-- Given a telescope that increases the visual range by 150 percent from an original range of 60 kilometers, 
    the new visual range is 150 kilometers. -/
theorem telescope_visual_range : 
  let original_range : ℝ := 60
  let increase_percent : ℝ := 150
  let new_range : ℝ := original_range * (1 + increase_percent / 100)
  new_range = 150 := by sorry

end NUMINAMATH_CALUDE_telescope_visual_range_l3173_317354


namespace NUMINAMATH_CALUDE_planted_fraction_of_field_l3173_317375

theorem planted_fraction_of_field (a b c x : ℝ) (h1 : a = 5) (h2 : b = 12) 
  (h3 : c^2 = a^2 + b^2) (h4 : x^2 * (a - x) * (b - x) = 3 * c * x^2) : 
  (a * b / 2 - x^2) / (a * b / 2) = 1461 / 1470 := by
  sorry

end NUMINAMATH_CALUDE_planted_fraction_of_field_l3173_317375


namespace NUMINAMATH_CALUDE_friend_brought_30_chocolates_l3173_317329

/-- The number of chocolates Nida's friend brought -/
def friend_chocolates (
  initial_chocolates : ℕ)  -- Nida's initial number of chocolates
  (loose_chocolates : ℕ)   -- Number of chocolates not in a box
  (filled_boxes : ℕ)       -- Number of filled boxes initially
  (extra_boxes_needed : ℕ) -- Number of extra boxes needed after friend brings chocolates
  : ℕ :=
  30

/-- Theorem stating that the number of chocolates Nida's friend brought is 30 -/
theorem friend_brought_30_chocolates :
  friend_chocolates 50 5 3 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_friend_brought_30_chocolates_l3173_317329


namespace NUMINAMATH_CALUDE_greater_solution_quadratic_l3173_317320

theorem greater_solution_quadratic (x : ℝ) : 
  (x^2 + 15*x - 54 = 0) → (∃ y : ℝ, y^2 + 15*y - 54 = 0 ∧ y ≠ x) → 
  (x ≥ y ↔ x = 3) :=
sorry

end NUMINAMATH_CALUDE_greater_solution_quadratic_l3173_317320


namespace NUMINAMATH_CALUDE_double_inequality_solution_l3173_317340

theorem double_inequality_solution (x : ℝ) : 
  (4 * x - 3 < (x - 2)^2 ∧ (x - 2)^2 < 6 * x - 5) ↔ (7 < x ∧ x < 9) :=
sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l3173_317340


namespace NUMINAMATH_CALUDE_one_digit_sum_problem_l3173_317377

theorem one_digit_sum_problem :
  ∀ (x y : ℕ),
  x ∈ Finset.range 10 →
  y ∈ Finset.range 10 →
  y ≠ 0 →
  7 * x + y = 68 →
  y = 5 :=
by sorry

end NUMINAMATH_CALUDE_one_digit_sum_problem_l3173_317377


namespace NUMINAMATH_CALUDE_august_eighth_is_saturday_l3173_317336

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific properties -/
structure Month where
  days : Nat
  mondays : Nat
  tuesdays : Nat

/-- Function to determine the day of the week for a given date -/
def dayOfWeek (m : Month) (date : Nat) : DayOfWeek := sorry

theorem august_eighth_is_saturday 
  (m : Month) 
  (h1 : m.days = 31) 
  (h2 : m.mondays = 5) 
  (h3 : m.tuesdays = 4) : 
  dayOfWeek m 8 = DayOfWeek.Saturday := by sorry

end NUMINAMATH_CALUDE_august_eighth_is_saturday_l3173_317336


namespace NUMINAMATH_CALUDE_ballet_class_size_l3173_317391

/-- The number of large groups formed in the ballet class -/
def large_groups : ℕ := 12

/-- The number of members in each large group -/
def members_per_large_group : ℕ := 7

/-- The total number of members in the ballet class -/
def total_members : ℕ := large_groups * members_per_large_group

theorem ballet_class_size : total_members = 84 := by
  sorry

end NUMINAMATH_CALUDE_ballet_class_size_l3173_317391


namespace NUMINAMATH_CALUDE_mrs_hilt_dogs_l3173_317362

/-- The number of dogs Mrs. Hilt saw -/
def num_dogs : ℕ := 2

/-- The number of chickens Mrs. Hilt saw -/
def num_chickens : ℕ := 2

/-- The total number of legs Mrs. Hilt saw -/
def total_legs : ℕ := 12

/-- The number of legs each dog has -/
def dog_legs : ℕ := 4

/-- The number of legs each chicken has -/
def chicken_legs : ℕ := 2

theorem mrs_hilt_dogs :
  num_dogs * dog_legs + num_chickens * chicken_legs = total_legs :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_dogs_l3173_317362


namespace NUMINAMATH_CALUDE_initial_men_count_l3173_317308

/-- Represents a road construction project --/
structure RoadProject where
  length : ℝ  -- Length of the road in km
  duration : ℝ  -- Total duration of the project in days
  initialProgress : ℝ  -- Length of road completed after 10 days
  initialDays : ℝ  -- Number of days for initial progress
  extraMen : ℕ  -- Number of extra men needed to finish on time

/-- Calculates the initial number of men employed in the project --/
def initialMen (project : RoadProject) : ℕ :=
  sorry

/-- Theorem stating the initial number of men for the given project --/
theorem initial_men_count (project : RoadProject) 
  (h1 : project.length = 10)
  (h2 : project.duration = 30)
  (h3 : project.initialProgress = 2)
  (h4 : project.initialDays = 10)
  (h5 : project.extraMen = 30) :
  initialMen project = 75 :=
sorry

end NUMINAMATH_CALUDE_initial_men_count_l3173_317308


namespace NUMINAMATH_CALUDE_evaluate_expression_l3173_317316

theorem evaluate_expression : 5 - (-3)^(2 - (1 - 3)) = -76 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3173_317316


namespace NUMINAMATH_CALUDE_quadrilateral_and_triangle_theorem_l3173_317341

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Intersection point of two lines -/
def intersect (l₁ l₂ : Line) : Point := sorry

/-- Line passing through two points -/
def line_through (p q : Point) : Line := sorry

/-- Point where a line parallel to a given direction through a point intersects another line -/
def parallel_intersect (p : Point) (l : Line) (dir : ℝ) : Point := sorry

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Check if two triangles are perspective -/
def perspective (t₁ t₂ : Point × Point × Point) : Prop := sorry

/-- The main theorem -/
theorem quadrilateral_and_triangle_theorem 
  (A B C D E F : Point) 
  (dir₁ dir₂ : ℝ) :
  let EF := line_through E F
  let A₁ := parallel_intersect A EF dir₁
  let B₁ := parallel_intersect B EF dir₁
  let C₁ := parallel_intersect C EF dir₁
  let D₁ := parallel_intersect D EF dir₁
  let desargues_line := sorry -- Definition of Desargues line
  let A' := parallel_intersect A desargues_line dir₂
  let B' := parallel_intersect B desargues_line dir₂
  let C' := parallel_intersect C desargues_line dir₂
  let A₁' := parallel_intersect A₁ desargues_line dir₂
  let B₁' := parallel_intersect B₁ desargues_line dir₂
  let C₁' := parallel_intersect C₁ desargues_line dir₂
  collinear E F (intersect (line_through A C) (line_through B D)) ∧
  perspective (A, B, C) (A₁, B₁, C₁) →
  (1 / distance A A₁ + 1 / distance C C₁ = 1 / distance B B₁ + 1 / distance D D₁) ∧
  (1 / distance A A' + 1 / distance B B' + 1 / distance C C' = 
   1 / distance A₁ A₁' + 1 / distance B₁ B₁' + 1 / distance C₁ C₁') := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_and_triangle_theorem_l3173_317341


namespace NUMINAMATH_CALUDE_weight_difference_theorem_l3173_317338

def weight_difference (robbie_weight patty_multiplier jim_multiplier mary_multiplier patty_loss jim_loss mary_gain : ℝ) : ℝ :=
  let patty_weight := patty_multiplier * robbie_weight - patty_loss
  let jim_weight := jim_multiplier * robbie_weight - jim_loss
  let mary_weight := mary_multiplier * robbie_weight + mary_gain
  patty_weight + jim_weight + mary_weight - robbie_weight

theorem weight_difference_theorem :
  weight_difference 100 4.5 3 2 235 180 45 = 480 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_theorem_l3173_317338


namespace NUMINAMATH_CALUDE_rearranged_cube_surface_area_l3173_317326

def slice_heights : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

def last_slice_height (heights : List ℚ) : ℚ :=
  1 - (heights.sum)

def surface_area (heights : List ℚ) : ℚ :=
  2 + 2 + 2  -- top/bottom + sides + front/back

theorem rearranged_cube_surface_area :
  surface_area slice_heights = 6 := by
  sorry

end NUMINAMATH_CALUDE_rearranged_cube_surface_area_l3173_317326


namespace NUMINAMATH_CALUDE_part_one_part_two_l3173_317321

-- Define the sets M and N
def M : Set ℝ := {x | (2*x - 2)/(x + 3) > 1}
def N (a : ℝ) : Set ℝ := {x | x^2 + (a - 8)*x - 8*a ≤ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ M
def q (a x : ℝ) : Prop := x ∈ N a

-- Part I: Relationship when a = -6
theorem part_one : 
  (∀ x, q (-6) x → p x) ∧ 
  (∃ x, p x ∧ ¬(q (-6) x)) := by sorry

-- Part II: Range of a where p is necessary but not sufficient for q
theorem part_two : 
  (∀ a, (∀ x, q a x → p x) ∧ (∃ x, p x ∧ ¬(q a x))) ↔ 
  a < -5 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3173_317321


namespace NUMINAMATH_CALUDE_problem_solution_l3173_317323

theorem problem_solution (x : ℝ) (h1 : Real.sqrt ((3 * x) / 7) = x) (h2 : x ≠ 0) : x = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3173_317323


namespace NUMINAMATH_CALUDE_Ca_concentration_after_mixing_l3173_317300

-- Define the constants
def K_sp : ℝ := 4.96e-9
def c_Na2CO3 : ℝ := 0.40
def c_CaCl2 : ℝ := 0.20

-- Define the theorem
theorem Ca_concentration_after_mixing :
  let c_CO3_remaining : ℝ := (c_Na2CO3 - c_CaCl2) / 2
  let c_Ca : ℝ := K_sp / c_CO3_remaining
  c_Ca = 4.96e-8 := by sorry

end NUMINAMATH_CALUDE_Ca_concentration_after_mixing_l3173_317300


namespace NUMINAMATH_CALUDE_number_of_students_in_section_B_l3173_317314

def section_A_students : ℕ := 40
def section_A_avg_weight : ℚ := 50
def section_B_avg_weight : ℚ := 60
def total_avg_weight : ℚ := 380 / 7  -- Approximation of 54.285714285714285

theorem number_of_students_in_section_B :
  ∃ (section_B_students : ℕ),
    (section_A_students * section_A_avg_weight + section_B_students * section_B_avg_weight) / 
    (section_A_students + section_B_students) = total_avg_weight ∧
    section_B_students = 30 := by
  sorry


end NUMINAMATH_CALUDE_number_of_students_in_section_B_l3173_317314


namespace NUMINAMATH_CALUDE_miles_trombones_l3173_317389

/-- Represents the number of musical instruments Miles owns -/
structure MilesInstruments where
  trumpets : ℕ
  guitars : ℕ
  french_horns : ℕ
  trombones : ℕ

/-- The total number of Miles' instruments -/
def total_instruments (m : MilesInstruments) : ℕ :=
  m.trumpets + m.guitars + m.french_horns + m.trombones

theorem miles_trombones :
  ∃ (m : MilesInstruments),
    m.trumpets = 10 - 3 ∧
    m.guitars = 2 + 2 ∧
    m.french_horns = m.guitars - 1 ∧
    m.trombones = 1 + 2 ∧
    total_instruments m = 17 →
    m.trombones = 3 := by
  sorry

end NUMINAMATH_CALUDE_miles_trombones_l3173_317389


namespace NUMINAMATH_CALUDE_sequence_property_l3173_317353

/-- Given an arithmetic sequence {aₙ} and a geometric sequence {bₙ} where
    a₃ = b₃ = a, a₆ = b₆ = b, and a > b, prove that if (a₄-b₄)(a₅-b₅) < 0, then ab < 0. -/
theorem sequence_property (a b : ℝ) (aₙ : ℕ → ℝ) (bₙ : ℕ → ℝ) 
    (h_arithmetic : ∀ n : ℕ, aₙ (n + 1) - aₙ n = aₙ 2 - aₙ 1)
    (h_geometric : ∀ n : ℕ, bₙ (n + 1) / bₙ n = bₙ 2 / bₙ 1)
    (h_a3 : aₙ 3 = a) (h_b3 : bₙ 3 = a)
    (h_a6 : aₙ 6 = b) (h_b6 : bₙ 6 = b)
    (h_a_gt_b : a > b) :
  (aₙ 4 - bₙ 4) * (aₙ 5 - bₙ 5) < 0 → a * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l3173_317353


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3173_317327

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {y | y > 0}

-- Define set B
def B : Set ℝ := {-2, -1, 1, 2}

-- Theorem statement
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3173_317327


namespace NUMINAMATH_CALUDE_polynomial_factorization_exists_l3173_317395

theorem polynomial_factorization_exists :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (p q r s : ℤ),
    ∀ (x : ℤ), x * (x - a) * (x - b) * (x - c) + 1 = (x^2 + p*x + q) * (x^2 + r*x + s) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_exists_l3173_317395


namespace NUMINAMATH_CALUDE_monster_count_theorem_l3173_317322

/-- Calculates the total number of monsters after 5 days given the initial count and daily growth factors -/
def total_monsters (initial : ℕ) (factor2 factor3 factor4 factor5 : ℕ) : ℕ :=
  initial + 
  initial * factor2 + 
  initial * factor2 * factor3 + 
  initial * factor2 * factor3 * factor4 + 
  initial * factor2 * factor3 * factor4 * factor5

/-- Theorem stating that given the specific initial count and growth factors, the total number of monsters after 5 days is 872 -/
theorem monster_count_theorem : total_monsters 2 3 4 5 6 = 872 := by
  sorry

end NUMINAMATH_CALUDE_monster_count_theorem_l3173_317322


namespace NUMINAMATH_CALUDE_sawyer_cut_difference_l3173_317399

/-- Represents a sawyer with their stick length and number of sections sawed -/
structure Sawyer where
  stickLength : Nat
  sectionsSawed : Nat

/-- Calculates the number of cuts made by a sawyer -/
def calculateCuts (s : Sawyer) : Nat :=
  (s.stickLength / 2 - 1) * (s.sectionsSawed / (s.stickLength / 2))

theorem sawyer_cut_difference (a b c : Sawyer)
  (h1 : a.stickLength = 8 ∧ b.stickLength = 10 ∧ c.stickLength = 6)
  (h2 : a.sectionsSawed = 24 ∧ b.sectionsSawed = 25 ∧ c.sectionsSawed = 27) :
  (max (max (calculateCuts a) (calculateCuts b)) (calculateCuts c) -
   min (min (calculateCuts a) (calculateCuts b)) (calculateCuts c)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sawyer_cut_difference_l3173_317399


namespace NUMINAMATH_CALUDE_range_of_a_l3173_317351

theorem range_of_a (a : ℝ) : 
  (∀ x₀ : ℝ, ∀ x : ℝ, x + a * x₀ + 1 ≥ 0) → a ∈ Set.Icc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3173_317351


namespace NUMINAMATH_CALUDE_k_range_k_values_circle_origin_l3173_317381

-- Define the line and hyperbola equations
def line (k x : ℝ) : ℝ := k * x + 1
def hyperbola (x y : ℝ) : Prop := 3 * x^2 - y^2 = 1

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ < 0 ∧ x₂ > 0 ∧
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
    y₁ = line k x₁ ∧ y₂ = line k x₂

-- Theorem for the range of k
theorem k_range :
  ∀ k : ℝ, intersection_points k ↔ -Real.sqrt 3 < k ∧ k < Real.sqrt 3 :=
sorry

-- Define the condition for the circle passing through the origin
def circle_through_origin (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    intersection_points k ∧
    x₁ * x₂ + y₁ * y₂ = 0

-- Theorem for the values of k when the circle passes through the origin
theorem k_values_circle_origin :
  ∀ k : ℝ, circle_through_origin k ↔ k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_CALUDE_k_range_k_values_circle_origin_l3173_317381


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3173_317343

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence with S_2 = 4 and S_4 = 20, the common difference is 3 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h2 : seq.S 2 = 4)
  (h4 : seq.S 4 = 20) :
  seq.d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3173_317343


namespace NUMINAMATH_CALUDE_min_box_value_l3173_317334

theorem min_box_value (a b Box : ℤ) : 
  (∀ x, (a*x + b) * (b*x + a) = 45*x^2 + Box*x + 45) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  ∀ Box', (∀ x, (∃ a' b', a' ≠ b' ∧ b' ≠ Box' ∧ a' ≠ Box' ∧
                 (a'*x + b') * (b'*x + a') = 45*x^2 + Box'*x + 45)) →
  Box' ≥ Box →
  Box ≥ 106 :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l3173_317334


namespace NUMINAMATH_CALUDE_food_distribution_proof_l3173_317398

/-- The initial number of men in the group -/
def initial_men : ℕ := 760

/-- The number of additional men who join after 2 days -/
def additional_men : ℕ := 190

/-- The initial number of days the food would last -/
def initial_days : ℕ := 22

/-- The number of days that pass before additional men join -/
def days_before_addition : ℕ := 2

/-- The number of days the food lasts after additional men join -/
def remaining_days : ℕ := 16

theorem food_distribution_proof :
  initial_men * initial_days = 
  (initial_men * days_before_addition) + 
  ((initial_men + additional_men) * remaining_days) := by
  sorry

end NUMINAMATH_CALUDE_food_distribution_proof_l3173_317398


namespace NUMINAMATH_CALUDE_min_value_of_g_l3173_317352

/-- The function f as defined in the problem -/
def f (x₁ x₂ x₃ : ℝ) : ℝ :=
  -2 * (x₁^3 + x₂^3 + x₃^3) + 3 * (x₁^2*(x₂ + x₃) + x₂^2*(x₁ + x₃) + x₃^2*(x₁ + x₂)) - 12*x₁*x₂*x₃

/-- The function g as defined in the problem -/
noncomputable def g (r s t : ℝ) : ℝ :=
  ⨆ (x₃ : ℝ) (h : t ≤ x₃ ∧ x₃ ≤ t + 2), |f r (r + 2) x₃ + s|

/-- The main theorem stating the minimum value of g -/
theorem min_value_of_g :
  (∀ r s t : ℝ, g r s t ≥ 12 * Real.sqrt 3) ∧
  (∃ r₀ s₀ t₀ : ℝ, g r₀ s₀ t₀ = 12 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_g_l3173_317352


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3173_317387

/-- Given a hyperbola with equation (x-2)^2/144 - (y+3)^2/81 = 1, 
    the slope of its asymptotes is 3/4 -/
theorem hyperbola_asymptote_slope :
  ∀ (x y : ℝ), 
  ((x - 2)^2 / 144 - (y + 3)^2 / 81 = 1) →
  (∃ m : ℝ, m = 3/4 ∧ 
   (∀ ε > 0, ∃ x₀ y₀ : ℝ, 
    ((x₀ - 2)^2 / 144 - (y₀ + 3)^2 / 81 = 1) ∧
    abs (y₀ - (m * x₀ - 9/2)) < ε)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3173_317387


namespace NUMINAMATH_CALUDE_john_vowel_learning_days_l3173_317315

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of days John takes to learn one alphabet -/
def days_per_alphabet : ℕ := 3

/-- The total number of days John needs to finish learning all vowels -/
def total_days : ℕ := num_vowels * days_per_alphabet

/-- Theorem: John needs 15 days to finish learning all vowels -/
theorem john_vowel_learning_days : total_days = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_vowel_learning_days_l3173_317315


namespace NUMINAMATH_CALUDE_not_divisible_by_2020_l3173_317386

theorem not_divisible_by_2020 (k : ℕ) : ¬(2020 ∣ (k^3 - 3*k^2 + 2*k + 2)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_2020_l3173_317386


namespace NUMINAMATH_CALUDE_variable_conditions_l3173_317373

theorem variable_conditions (a b c d e : ℝ) 
  (h : (a + b + e) / (b + c) = (c + d + e) / (d + a)) : 
  a = c ∨ a + b + c + d + e = 0 := by sorry

end NUMINAMATH_CALUDE_variable_conditions_l3173_317373


namespace NUMINAMATH_CALUDE_chessboard_cut_parts_l3173_317331

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)
  (white_squares : ℕ)
  (black_squares : ℕ)

/-- Represents the possible number of parts a chessboard can be cut into --/
def PossibleParts : Set ℕ := {2, 4, 8, 16, 32}

/-- Main theorem: The number of parts a chessboard can be cut into is a subset of PossibleParts --/
theorem chessboard_cut_parts (board : Chessboard) 
  (h1 : board.size = 8) 
  (h2 : board.white_squares = 32) 
  (h3 : board.black_squares = 32) : 
  ∃ (n : ℕ), n ∈ PossibleParts ∧ 
  (board.white_squares % n = 0) ∧ 
  (n > 1) ∧ 
  (n ≤ board.black_squares) :=
sorry

end NUMINAMATH_CALUDE_chessboard_cut_parts_l3173_317331


namespace NUMINAMATH_CALUDE_snow_probability_l3173_317312

theorem snow_probability (p : ℝ) (n : ℕ) 
  (h_p : p = 3/4) 
  (h_n : n = 4) : 
  1 - (1 - p)^n = 255/256 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l3173_317312


namespace NUMINAMATH_CALUDE_meghan_money_l3173_317390

/-- The total amount of money Meghan had -/
def total_money (hundred_bills : ℕ) (fifty_bills : ℕ) (ten_bills : ℕ) : ℕ :=
  100 * hundred_bills + 50 * fifty_bills + 10 * ten_bills

/-- Proof that Meghan had $550 -/
theorem meghan_money : total_money 2 5 10 = 550 := by
  sorry

end NUMINAMATH_CALUDE_meghan_money_l3173_317390


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3173_317369

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  bridge_length = 205 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l3173_317369


namespace NUMINAMATH_CALUDE_arthur_summer_reading_l3173_317380

theorem arthur_summer_reading (book1_pages book2_pages : ℕ) 
  (book1_read_percent : ℚ) (book2_read_fraction : ℚ) (pages_left : ℕ) : 
  book1_pages = 500 → 
  book2_pages = 1000 → 
  book1_read_percent = 80 / 100 → 
  book2_read_fraction = 1 / 5 → 
  pages_left = 200 → 
  (book1_pages * book1_read_percent).floor + 
  (book2_pages * book2_read_fraction).floor + 
  pages_left = 800 := by
  sorry

end NUMINAMATH_CALUDE_arthur_summer_reading_l3173_317380


namespace NUMINAMATH_CALUDE_digits_of_2_pow_15_times_5_pow_10_l3173_317361

/-- The number of digits in 2^15 * 5^10 is 12 -/
theorem digits_of_2_pow_15_times_5_pow_10 : 
  (Nat.digits 10 (2^15 * 5^10)).length = 12 := by sorry

end NUMINAMATH_CALUDE_digits_of_2_pow_15_times_5_pow_10_l3173_317361


namespace NUMINAMATH_CALUDE_median_is_82_l3173_317339

/-- Represents the list where each integer n (1 ≤ n ≤ 100) appears 2n times -/
def special_list : List ℕ := sorry

/-- The total number of elements in the special list -/
def total_elements : ℕ := sorry

/-- The median of the special list -/
def median_of_special_list : ℚ := sorry

/-- Theorem stating that the median of the special list is 82 -/
theorem median_is_82 : median_of_special_list = 82 := by sorry

end NUMINAMATH_CALUDE_median_is_82_l3173_317339


namespace NUMINAMATH_CALUDE_original_house_price_l3173_317370

/-- Given a house that increases in value by 25% and is then sold to cover 25% of a $500,000 new house,
    prove that the original purchase price of the first house was $100,000. -/
theorem original_house_price (original_price : ℝ) : 
  (original_price * 1.25 = 500000 * 0.25) → original_price = 100000 := by
  sorry

end NUMINAMATH_CALUDE_original_house_price_l3173_317370


namespace NUMINAMATH_CALUDE_triangle_properties_l3173_317379

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.cos x - Real.sqrt 3 * Real.sin (2 * x)

theorem triangle_properties :
  ∀ (A B C : ℝ) (a b c : ℝ),
  f A = -1 →
  a = Real.sqrt 7 →
  ∃ (m n : ℝ × ℝ), m = (3, Real.sin B) ∧ n = (2, Real.sin C) ∧ ∃ (k : ℝ), m = k • n →
  A = π / 3 ∧
  b = 3 ∧
  c = 2 ∧
  (1 / 2 : ℝ) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3173_317379


namespace NUMINAMATH_CALUDE_project_hours_difference_l3173_317302

theorem project_hours_difference (total_hours : ℕ) 
  (h_total : total_hours = 189) 
  (h_pat_kate : ∃ k : ℕ, pat = 2 * k ∧ kate = k) 
  (h_pat_mark : ∃ m : ℕ, mark = 3 * pat ∧ pat = m) 
  (h_sum : pat + kate + mark = total_hours) :
  mark - kate = 105 :=
by sorry

end NUMINAMATH_CALUDE_project_hours_difference_l3173_317302


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l3173_317394

theorem smallest_solution_quadratic (y : ℝ) :
  (3 * y^2 + 33 * y - 105 = y * (y + 16)) →
  y ≥ -21/2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l3173_317394


namespace NUMINAMATH_CALUDE_cone_min_lateral_area_l3173_317311

/-- For a cone with volume π/6, when its lateral area is minimum, 
    the tangent of the angle between the slant height and the base is √2 -/
theorem cone_min_lateral_area (r h : ℝ) : 
  r > 0 → h > 0 → 
  (1/3) * π * r^2 * h = π/6 →
  (∀ r' h', r' > 0 → h' > 0 → (1/3) * π * r'^2 * h' = π/6 → 
    π * r * (r^2 + h^2).sqrt ≤ π * r' * (r'^2 + h'^2).sqrt) →
  h / r = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cone_min_lateral_area_l3173_317311


namespace NUMINAMATH_CALUDE_exists_min_value_l3173_317360

/-- The function we want to minimize -/
def f (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y + y^3

/-- Theorem stating that there exists a minimum value for the function -/
theorem exists_min_value :
  ∃ (y : ℝ), ∃ (min_val : ℝ), ∀ (x : ℝ), f x y ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_exists_min_value_l3173_317360


namespace NUMINAMATH_CALUDE_mutually_inscribed_tetrahedra_exist_l3173_317306

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a tetrahedron as a set of four points
structure Tetrahedron where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D

-- Define the property of being coplanar
def coplanar (p q r s : Point3D) : Prop := sorry

-- Define the property of a tetrahedron being inscribed in another
def inscribed (t1 t2 : Tetrahedron) : Prop :=
  coplanar t2.a t1.b t1.c t1.d ∧
  coplanar t2.b t1.a t1.c t1.d ∧
  coplanar t2.c t1.a t1.b t1.d ∧
  coplanar t2.d t1.a t1.b t1.c

-- Define the property of two tetrahedra not sharing vertices
def no_shared_vertices (t1 t2 : Tetrahedron) : Prop :=
  t1.a ≠ t2.a ∧ t1.a ≠ t2.b ∧ t1.a ≠ t2.c ∧ t1.a ≠ t2.d ∧
  t1.b ≠ t2.a ∧ t1.b ≠ t2.b ∧ t1.b ≠ t2.c ∧ t1.b ≠ t2.d ∧
  t1.c ≠ t2.a ∧ t1.c ≠ t2.b ∧ t1.c ≠ t2.c ∧ t1.c ≠ t2.d ∧
  t1.d ≠ t2.a ∧ t1.d ≠ t2.b ∧ t1.d ≠ t2.c ∧ t1.d ≠ t2.d

-- The theorem to be proved
theorem mutually_inscribed_tetrahedra_exist : 
  ∃ (t1 t2 : Tetrahedron), inscribed t1 t2 ∧ inscribed t2 t1 ∧ no_shared_vertices t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_mutually_inscribed_tetrahedra_exist_l3173_317306


namespace NUMINAMATH_CALUDE_correct_calculation_l3173_317356

theorem correct_calculation (a : ℝ) : 3 * (a^2)^3 - 6 * a^6 = -3 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3173_317356


namespace NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l3173_317345

theorem sum_of_four_primes_divisible_by_60 (p q r s : ℕ) :
  Prime p → Prime q → Prime r → Prime s →
  5 < p → p < q → q < r → r < s → s < p + 10 →
  60 ∣ (p + q + r + s) := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l3173_317345


namespace NUMINAMATH_CALUDE_sum_three_squares_not_7_mod_8_l3173_317324

theorem sum_three_squares_not_7_mod_8 (a b c : ℤ) : (a^2 + b^2 + c^2) % 8 ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_three_squares_not_7_mod_8_l3173_317324


namespace NUMINAMATH_CALUDE_daily_rental_cost_l3173_317303

/-- Represents the daily car rental cost problem -/
theorem daily_rental_cost (total_cost : ℝ) (miles_driven : ℝ) (per_mile_rate : ℝ) :
  total_cost = 46.12 ∧
  miles_driven = 214.0 ∧
  per_mile_rate = 0.08 →
  ∃ (daily_rate : ℝ), daily_rate = 29.00 ∧ total_cost = daily_rate + miles_driven * per_mile_rate :=
by sorry

end NUMINAMATH_CALUDE_daily_rental_cost_l3173_317303


namespace NUMINAMATH_CALUDE_softball_team_composition_l3173_317337

theorem softball_team_composition (total : ℕ) (ratio : ℚ) : 
  total = 14 → ratio = 5/9 → ∃ (men women : ℕ), 
    men + women = total ∧ 
    (men : ℚ) / (women : ℚ) = ratio ∧ 
    women - men = 4 := by
  sorry

end NUMINAMATH_CALUDE_softball_team_composition_l3173_317337


namespace NUMINAMATH_CALUDE_park_outer_boundary_diameter_l3173_317383

/-- Represents a circular park with concentric features -/
structure CircularPark where
  pond_diameter : ℝ
  garden_width : ℝ
  grassy_area_width : ℝ
  walking_path_width : ℝ

/-- Calculates the diameter of the outer boundary of a circular park -/
def outer_boundary_diameter (park : CircularPark) : ℝ :=
  park.pond_diameter + 2 * (park.garden_width + park.grassy_area_width + park.walking_path_width)

/-- Theorem stating that for a park with given measurements, the outer boundary diameter is 52 feet -/
theorem park_outer_boundary_diameter :
  let park : CircularPark := {
    pond_diameter := 12,
    garden_width := 10,
    grassy_area_width := 4,
    walking_path_width := 6
  }
  outer_boundary_diameter park = 52 := by
  sorry

end NUMINAMATH_CALUDE_park_outer_boundary_diameter_l3173_317383


namespace NUMINAMATH_CALUDE_johns_annual_epipen_cost_l3173_317325

/-- Calculates the annual cost of EpiPens for John given the replacement frequency,
    cost per EpiPen, and insurance coverage percentage. -/
def annual_epipen_cost (replacement_months : ℕ) (cost_per_epipen : ℕ) (insurance_coverage_percent : ℕ) : ℕ :=
  let epipens_per_year : ℕ := 12 / replacement_months
  let insurance_coverage : ℕ := cost_per_epipen * insurance_coverage_percent / 100
  let cost_after_insurance : ℕ := cost_per_epipen - insurance_coverage
  epipens_per_year * cost_after_insurance

/-- Theorem stating that John's annual cost for EpiPens is $250 -/
theorem johns_annual_epipen_cost :
  annual_epipen_cost 6 500 75 = 250 := by
  sorry

end NUMINAMATH_CALUDE_johns_annual_epipen_cost_l3173_317325


namespace NUMINAMATH_CALUDE_number_plus_four_equals_six_l3173_317367

theorem number_plus_four_equals_six (x : ℤ) : x + 4 = 6 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_four_equals_six_l3173_317367


namespace NUMINAMATH_CALUDE_simplify_expression_l3173_317328

theorem simplify_expression :
  ∃ (a b c : ℕ+),
    ((Real.sqrt 3 - 1) ^ (2 - Real.sqrt 2)) / ((Real.sqrt 3 + 1) ^ (2 + Real.sqrt 2)) = a - b * Real.sqrt c ∧
    ¬ ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ k > 1 ∧ p ^ k ∣ c.val ∧
    a = 21 ∧ b = 12 ∧ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3173_317328


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3173_317309

theorem pure_imaginary_complex_number (m : ℝ) : 
  (((m^2 - 5*m + 6) : ℂ) + (m^2 - 3*m)*I = (0 : ℂ) + (m^2 - 3*m)*I) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3173_317309


namespace NUMINAMATH_CALUDE_min_value_expression_l3173_317364

theorem min_value_expression (a b c : ℝ) 
  (sum_condition : a + b + c = -1) 
  (product_condition : a * b * c ≤ -3) :
  (a * b + 1) / (a + b) + (b * c + 1) / (b + c) + (c * a + 1) / (c + a) ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3173_317364


namespace NUMINAMATH_CALUDE_league_games_count_l3173_317332

/-- The number of teams in the league -/
def num_teams : ℕ := 25

/-- The total number of games played in the league -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- Theorem stating that the total number of games in the league is 300 -/
theorem league_games_count : total_games = 300 := by
  sorry

end NUMINAMATH_CALUDE_league_games_count_l3173_317332


namespace NUMINAMATH_CALUDE_mindy_mork_earnings_ratio_l3173_317313

theorem mindy_mork_earnings_ratio :
  ∀ (mork_earnings mindy_earnings : ℝ),
    mork_earnings > 0 →
    mindy_earnings > 0 →
    0.45 * mork_earnings + 0.15 * mindy_earnings = 0.21 * (mork_earnings + mindy_earnings) →
    mindy_earnings / mork_earnings = 4 := by
  sorry

end NUMINAMATH_CALUDE_mindy_mork_earnings_ratio_l3173_317313


namespace NUMINAMATH_CALUDE_ship_length_in_emily_steps_l3173_317346

theorem ship_length_in_emily_steps :
  ∀ (emily_speed ship_speed : ℝ) (emily_steps_forward emily_steps_backward : ℕ),
    emily_speed > ship_speed →
    emily_steps_forward = 300 →
    emily_steps_backward = 60 →
    ship_speed > 0 →
    ∃ (ship_length : ℝ),
      ship_length = emily_steps_forward * emily_speed / (emily_speed + ship_speed) +
                    emily_steps_backward * emily_speed / (emily_speed - ship_speed) ∧
      ship_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_ship_length_in_emily_steps_l3173_317346


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3173_317384

theorem trig_identity_proof :
  Real.sin (50 * π / 180) * Real.cos (20 * π / 180) -
  Real.cos (50 * π / 180) * Real.sin (20 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3173_317384


namespace NUMINAMATH_CALUDE_inverse_sum_inverse_l3173_317355

theorem inverse_sum_inverse (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (x * z + y * z + x * y) :=
by sorry

end NUMINAMATH_CALUDE_inverse_sum_inverse_l3173_317355


namespace NUMINAMATH_CALUDE_xiaotong_grade_l3173_317359

/-- Represents the grading system for a physical education course -/
structure GradingSystem where
  maxScore : ℝ
  classroomWeight : ℝ
  midtermWeight : ℝ
  finalWeight : ℝ

/-- Represents a student's scores in the physical education course -/
structure StudentScores where
  classroom : ℝ
  midterm : ℝ
  final : ℝ

/-- Calculates the final grade based on the grading system and student scores -/
def calculateGrade (sys : GradingSystem) (scores : StudentScores) : ℝ :=
  sys.classroomWeight * scores.classroom +
  sys.midtermWeight * scores.midterm +
  sys.finalWeight * scores.final

/-- The theorem stating that Xiaotong's grade is 55 given the specified grading system and scores -/
theorem xiaotong_grade :
  let sys : GradingSystem := {
    maxScore := 60,
    classroomWeight := 0.2,
    midtermWeight := 0.3,
    finalWeight := 0.5
  }
  let scores : StudentScores := {
    classroom := 60,
    midterm := 50,
    final := 56
  }
  calculateGrade sys scores = 55 := by
  sorry

end NUMINAMATH_CALUDE_xiaotong_grade_l3173_317359
