import Mathlib

namespace average_of_xyz_l2778_277891

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) :
  (x + y + z) / 3 = 4 := by sorry

end average_of_xyz_l2778_277891


namespace most_cost_effective_plan_verify_bus_capacities_l2778_277865

/-- Represents the capacity and cost of buses -/
structure BusInfo where
  small_capacity : ℕ
  large_capacity : ℕ
  small_cost : ℕ
  large_cost : ℕ

/-- Represents a rental plan -/
structure RentalPlan where
  small_buses : ℕ
  large_buses : ℕ

/-- Checks if a rental plan is valid for the given number of students -/
def is_valid_plan (info : BusInfo) (students : ℕ) (plan : RentalPlan) : Prop :=
  plan.small_buses * info.small_capacity + plan.large_buses * info.large_capacity = students

/-- Calculates the cost of a rental plan -/
def plan_cost (info : BusInfo) (plan : RentalPlan) : ℕ :=
  plan.small_buses * info.small_cost + plan.large_buses * info.large_cost

/-- Theorem stating the most cost-effective plan -/
theorem most_cost_effective_plan (info : BusInfo) (students : ℕ) : 
  info.small_capacity = 20 →
  info.large_capacity = 45 →
  info.small_cost = 200 →
  info.large_cost = 400 →
  students = 400 →
  ∃ (optimal_plan : RentalPlan),
    is_valid_plan info students optimal_plan ∧
    optimal_plan.small_buses = 2 ∧
    optimal_plan.large_buses = 8 ∧
    plan_cost info optimal_plan = 3600 ∧
    ∀ (plan : RentalPlan), 
      is_valid_plan info students plan → 
      plan_cost info optimal_plan ≤ plan_cost info plan :=
by
  sorry

/-- Verifies the given bus capacities -/
theorem verify_bus_capacities (info : BusInfo) :
  info.small_capacity = 20 →
  info.large_capacity = 45 →
  3 * info.small_capacity + info.large_capacity = 105 ∧
  info.small_capacity + 2 * info.large_capacity = 110 :=
by
  sorry

end most_cost_effective_plan_verify_bus_capacities_l2778_277865


namespace exactly_nine_heads_probability_l2778_277818

/-- The probability of getting heads when flipping the biased coin -/
def p : ℚ := 3/4

/-- The number of coin flips -/
def n : ℕ := 12

/-- The number of heads we want to get -/
def k : ℕ := 9

/-- The probability of getting exactly k heads in n flips of a coin with probability p of heads -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem exactly_nine_heads_probability :
  binomial_probability n k p = 4330260/16777216 := by
  sorry

end exactly_nine_heads_probability_l2778_277818


namespace decimal_to_fraction_l2778_277838

theorem decimal_to_fraction : (2.375 : ℚ) = 19 / 8 := by
  sorry

end decimal_to_fraction_l2778_277838


namespace abs_sum_minimum_l2778_277864

theorem abs_sum_minimum (x y : ℝ) : |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 := by
  sorry

end abs_sum_minimum_l2778_277864


namespace root_product_l2778_277877

theorem root_product (f g : ℝ → ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (∀ x, f x = x^5 - x^3 + 1) →
  (∀ x, g x = x^2 - 2) →
  f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0 →
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = -7 := by
  sorry

end root_product_l2778_277877


namespace assignment_conditions_l2778_277810

/-- The number of ways to assign four students to three classes -/
def assignStudents : ℕ :=
  Nat.choose 4 2 * (3 * 2 * 1) - (3 * 2 * 1)

/-- Conditions of the problem -/
theorem assignment_conditions :
  (∀ (assignment : Fin 4 → Fin 3), 
    (∀ c : Fin 3, ∃ s : Fin 4, assignment s = c) ∧ 
    (assignment 0 ≠ assignment 1)) →
  (assignStudents = 30) :=
sorry

end assignment_conditions_l2778_277810


namespace fraction_equality_l2778_277879

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - 4*b ≠ 0) (h4 : 4*a - b ≠ 0)
  (h5 : (4*a + 2*b) / (a - 4*b) = 3) : (a + 4*b) / (4*a - b) = 10/57 := by
  sorry

end fraction_equality_l2778_277879


namespace triangle_sine_inequality_l2778_277882

theorem triangle_sine_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.sin A * Real.sin (A/2) + Real.sin B * Real.sin (B/2) + Real.sin C * Real.sin (C/2) ≤ 4 * Real.sqrt 3 / 3 := by
  sorry

end triangle_sine_inequality_l2778_277882


namespace tan_equality_implies_45_l2778_277814

theorem tan_equality_implies_45 (n : ℤ) :
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (225 * π / 180) → n = 45 := by
  sorry

end tan_equality_implies_45_l2778_277814


namespace quadratic_inequality_range_l2778_277885

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a-1)*x + 1 ≤ 0) → a ∈ Set.Ioo (-1 : ℝ) 3 :=
by
  sorry

end quadratic_inequality_range_l2778_277885


namespace banana_arrangement_count_l2778_277895

/-- The number of ways to arrange the letters of BANANA with indistinguishable A's and N's -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in BANANA -/
def total_letters : ℕ := 6

/-- The number of A's in BANANA -/
def num_a : ℕ := 3

/-- The number of N's in BANANA -/
def num_n : ℕ := 2

/-- The number of B's in BANANA -/
def num_b : ℕ := 1

theorem banana_arrangement_count : 
  banana_arrangements = (Nat.factorial total_letters) / 
    ((Nat.factorial num_a) * (Nat.factorial num_n) * (Nat.factorial num_b)) :=
by sorry

end banana_arrangement_count_l2778_277895


namespace box_office_scientific_notation_equality_l2778_277867

-- Define the box office revenue
def box_office_revenue : ℝ := 1824000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.824 * (10 ^ 9)

-- Theorem stating that the box office revenue is equal to its scientific notation representation
theorem box_office_scientific_notation_equality : 
  box_office_revenue = scientific_notation := by
  sorry

end box_office_scientific_notation_equality_l2778_277867


namespace expression_equals_six_l2778_277807

theorem expression_equals_six : 
  Real.sqrt 16 - 2 * Real.tan (45 * π / 180) + |(-3)| + (π - 2022) ^ (0 : ℕ) = 6 := by
  sorry

end expression_equals_six_l2778_277807


namespace perpendicular_implies_parallel_perpendicular_to_plane_implies_parallel_l2778_277808

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: If m ⟂ α and m ⟂ β, then α ∥ β
theorem perpendicular_implies_parallel 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular m β) : 
  parallel α β :=
sorry

-- Theorem 2: If m ⟂ β and n ⟂ β, then m ∥ n
theorem perpendicular_to_plane_implies_parallel 
  (m n : Line) (β : Plane) 
  (h1 : perpendicular m β) (h2 : perpendicular n β) : 
  parallel_lines m n :=
sorry

end perpendicular_implies_parallel_perpendicular_to_plane_implies_parallel_l2778_277808


namespace enclosed_area_is_four_l2778_277833

-- Define the functions
def f (x : ℝ) : ℝ := 4 * x
def g (x : ℝ) : ℝ := x^3

-- Define the region
def region := {x : ℝ | 0 ≤ x ∧ x ≤ 2 ∧ g x ≤ f x}

-- State the theorem
theorem enclosed_area_is_four : 
  ∫ x in region, (f x - g x) = 4 := by sorry

end enclosed_area_is_four_l2778_277833


namespace group_division_ways_l2778_277888

theorem group_division_ways (n : ℕ) (g₁ g₂ g₃ : ℕ) (h₁ : n = 8) (h₂ : g₁ = 2) (h₃ : g₂ = 3) (h₄ : g₃ = 3) :
  (Nat.choose n g₂ * Nat.choose (n - g₂) g₃) / 2 = 280 :=
by sorry

end group_division_ways_l2778_277888


namespace marble_158_is_gray_l2778_277859

/-- Represents the color of a marble -/
inductive Color
  | Gray
  | White
  | Black

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : Color :=
  match n % 12 with
  | 0 | 1 | 2 | 3 | 4 => Color.Gray
  | 5 | 6 | 7 | 8 => Color.White
  | _ => Color.Black

theorem marble_158_is_gray : marbleColor 158 = Color.Gray := by
  sorry

end marble_158_is_gray_l2778_277859


namespace log_equality_implies_base_l2778_277831

theorem log_equality_implies_base (y : ℝ) (h : y > 0) :
  (Real.log 8 / Real.log y = Real.log 5 / Real.log 125) → y = 512 := by
  sorry

end log_equality_implies_base_l2778_277831


namespace parabola_properties_l2778_277819

-- Define the parabola
def parabola (a b c x : ℝ) := a * x^2 + b * x + c

-- State the theorem
theorem parabola_properties (a b c : ℝ) 
  (h1 : parabola a b c 2 = 0) :
  (b = -2*a → parabola a b c 0 = 0) ∧ 
  (c ≠ 4*a → (b^2 - 4*a*c > 0)) ∧
  (∀ x1 x2, x1 > x2 ∧ x2 > -1 ∧ parabola a b c x1 > parabola a b c x2 → 8*a + c ≤ 0) :=
by sorry

end parabola_properties_l2778_277819


namespace abs_inequality_equivalence_l2778_277828

theorem abs_inequality_equivalence (x : ℝ) : 
  |2*x - 1| - |x - 2| < 0 ↔ -1 < x ∧ x < 1 := by
sorry

end abs_inequality_equivalence_l2778_277828


namespace pentagon_diagonals_from_vertex_l2778_277884

/-- The number of diagonals that can be drawn from a vertex of an n-sided polygon. -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- A pentagon has 5 sides. -/
def pentagon_sides : ℕ := 5

theorem pentagon_diagonals_from_vertex :
  diagonals_from_vertex pentagon_sides = 2 := by
  sorry

end pentagon_diagonals_from_vertex_l2778_277884


namespace average_monthly_balance_l2778_277823

def monthly_balances : List ℝ := [120, 240, 180, 180, 160, 200]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℝ) = 180 := by
  sorry

end average_monthly_balance_l2778_277823


namespace difference_of_squares_special_case_l2778_277839

theorem difference_of_squares_special_case : (4 + Real.sqrt 6) * (4 - Real.sqrt 6) = 10 := by
  sorry

end difference_of_squares_special_case_l2778_277839


namespace first_number_a10_l2778_277889

def first_number (n : ℕ) : ℕ :=
  1 + 2 * (n * (n - 1) / 2)

theorem first_number_a10 : first_number 10 = 91 := by
  sorry

end first_number_a10_l2778_277889


namespace factorial_difference_l2778_277849

theorem factorial_difference : Nat.factorial 9 - Nat.factorial 8 = 322560 := by
  sorry

end factorial_difference_l2778_277849


namespace circle_line_problem_l2778_277822

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line l
def l (x y k : ℝ) : Prop := y = k*x - 2

-- Define tangency condition
def is_tangent (k : ℝ) : Prop := ∃ x y : ℝ, C x y ∧ l x y k

-- Define the condition for a point on l to be within distance 2 from the center of C
def point_within_distance (k : ℝ) : Prop := 
  ∃ x y : ℝ, l x y k ∧ (x - 4)^2 + y^2 ≤ 4

theorem circle_line_problem (k : ℝ) :
  (is_tangent k → k = (8 + Real.sqrt 19) / 15 ∨ k = (8 - Real.sqrt 19) / 15) ∧
  (point_within_distance k → 0 ≤ k ∧ k ≤ 4/3) :=
sorry

end circle_line_problem_l2778_277822


namespace pizza_slice_volume_l2778_277837

/-- The volume of a pizza slice -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_slices : ℕ) :
  thickness = 1/2 →
  diameter = 16 →
  num_slices = 8 →
  (π * (diameter/2)^2 * thickness) / num_slices = 4 * π :=
by sorry

end pizza_slice_volume_l2778_277837


namespace power_equation_solution_l2778_277886

theorem power_equation_solution (n : ℕ) : 5^n = 5 * 25^3 * 125^2 → n = 13 := by
  sorry

end power_equation_solution_l2778_277886


namespace calculator_key_presses_l2778_277829

def f (x : ℕ) : ℕ := x^2 - 3

def iterate_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => f (iterate_f n x)

theorem calculator_key_presses :
  iterate_f 2 4 ≤ 2000 ∧ iterate_f 3 4 > 2000 := by
  sorry

end calculator_key_presses_l2778_277829


namespace circle_equation_l2778_277856

theorem circle_equation (x y : ℝ) : 
  (∃ (R : ℝ), (x - 3)^2 + (y - 1)^2 = R^2) ∧ 
  (0 - 3)^2 + (0 - 1)^2 = 10 →
  (x - 3)^2 + (y - 1)^2 = 10 :=
by sorry

end circle_equation_l2778_277856


namespace triple_composition_even_l2778_277855

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem triple_composition_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
by
  sorry

end triple_composition_even_l2778_277855


namespace min_a_value_l2778_277860

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x - 1 - a * Real.log x
def g (x : ℝ) : ℝ := x / Real.exp (x - 1)

-- State the theorem
theorem min_a_value (a : ℝ) :
  (a < 0) →
  (∀ x₁ x₂ : ℝ, 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 4 →
    (f a x₁ - f a x₂) / (g x₁ - g x₂) > -1 / (g x₁ * g x₂)) →
  a ≥ 3 - 2 / 3 * Real.exp 2 :=
sorry

end

end min_a_value_l2778_277860


namespace shortest_chord_through_A_equals_4_l2778_277806

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 1)

-- Define the function to calculate the shortest chord length
noncomputable def shortest_chord_length (c : (ℝ → ℝ → Prop)) (p : ℝ × ℝ) : ℝ :=
  sorry -- Implementation details are omitted

-- Theorem statement
theorem shortest_chord_through_A_equals_4 :
  shortest_chord_length circle_M point_A = 4 := by sorry

end shortest_chord_through_A_equals_4_l2778_277806


namespace tangent_line_equation_l2778_277863

/-- Given a real number a and a function f(x) = x³ + ax² + (a-3)x with derivative f'(x),
    where f'(x) is an even function, prove that the equation of the tangent line to
    the curve y = f(x) at the point (2, f(2)) is 9x - y - 16 = 0. -/
theorem tangent_line_equation (a : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + (a-3)*x
  let f' : ℝ → ℝ := λ x => 3*x^2 + 2*a*x + (a-3)
  (∀ x, f' x = f' (-x)) → 
  (λ x y => 9*x - y - 16 = 0) = (λ x y => y - f 2 = f' 2 * (x - 2)) := by
  sorry

end tangent_line_equation_l2778_277863


namespace special_rectangle_area_l2778_277881

/-- A rectangle ABCD with specific properties -/
structure SpecialRectangle where
  -- AB, BC, CD are sides of the rectangle
  AB : ℝ
  BC : ℝ
  CD : ℝ
  -- E is the midpoint of BC
  BE : ℝ
  -- Conditions
  rectangle_condition : AB = CD
  perimeter_condition : AB + BC + CD = 20
  midpoint_condition : BE = BC / 2
  diagonal_condition : AB^2 + BE^2 = 9^2

/-- The area of a SpecialRectangle is 19 -/
theorem special_rectangle_area (r : SpecialRectangle) : r.AB * r.BC = 19 := by
  sorry

end special_rectangle_area_l2778_277881


namespace factorial_ratio_l2778_277802

theorem factorial_ratio : (Nat.factorial 9) / (Nat.factorial 8) = 9 := by
  sorry

end factorial_ratio_l2778_277802


namespace no_solution_sqrt_equation_l2778_277846

theorem no_solution_sqrt_equation : ¬ ∃ x : ℝ, Real.sqrt (3 * x - 2) + Real.sqrt (2 * x - 3) = 1 := by
  sorry

end no_solution_sqrt_equation_l2778_277846


namespace train_crossing_time_l2778_277801

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 48 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end train_crossing_time_l2778_277801


namespace tank_capacity_is_640_verify_capacity_l2778_277874

/-- Represents the capacity of a tank in litres. -/
def tank_capacity : ℝ := 640

/-- Represents the time in hours it takes to empty the tank with only the outlet pipe open. -/
def outlet_time : ℝ := 10

/-- Represents the rate at which the inlet pipe adds water, in litres per minute. -/
def inlet_rate : ℝ := 4

/-- Represents the time in hours it takes to empty the tank with both inlet and outlet pipes open. -/
def both_pipes_time : ℝ := 16

/-- Theorem stating that the tank capacity is 640 litres given the conditions. -/
theorem tank_capacity_is_640 :
  tank_capacity = outlet_time * (inlet_rate * 60) * both_pipes_time / (both_pipes_time - outlet_time) :=
by
  sorry

/-- Verifies that the calculated capacity matches the given value of 640 litres. -/
theorem verify_capacity :
  tank_capacity = 640 :=
by
  sorry

end tank_capacity_is_640_verify_capacity_l2778_277874


namespace cost_of_48_doughnuts_l2778_277824

/-- The cost of buying a specified number of doughnuts -/
def doughnutCost (n : ℕ) : ℚ :=
  1 + 6 * ((n - 1) / 12)

/-- Theorem stating the cost of 48 doughnuts -/
theorem cost_of_48_doughnuts : doughnutCost 48 = 25 := by
  sorry

end cost_of_48_doughnuts_l2778_277824


namespace alice_average_speed_l2778_277816

/-- Alice's cycling journey --/
def alice_journey : Prop :=
  let first_distance : ℝ := 240
  let first_time : ℝ := 4.5
  let second_distance : ℝ := 300
  let second_time : ℝ := 5.25
  let total_distance : ℝ := first_distance + second_distance
  let total_time : ℝ := first_time + second_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 540 / 9.75

theorem alice_average_speed : alice_journey := by
  sorry

end alice_average_speed_l2778_277816


namespace sin_135_degrees_l2778_277892

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_135_degrees_l2778_277892


namespace unique_solution_condition_l2778_277827

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, ∃ y : ℝ, 4 * x - 7 + c = d * x + 2 * y + 4) ↔ d ≠ 4 :=
by sorry

end unique_solution_condition_l2778_277827


namespace square_perimeter_l2778_277840

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (∃ (w h : ℝ), w > 0 ∧ h > 0 ∧ w * h = s^2 / 4 ∧ 2*(w + h) = 40) → 
  4 * s = 64 := by
sorry

end square_perimeter_l2778_277840


namespace cone_cylinder_volume_relation_l2778_277813

/-- The volume of a cone with the same radius and height as a cylinder with volume 150π cm³ is 50π cm³ -/
theorem cone_cylinder_volume_relation (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  π * r^2 * h = 150 * π → (1/3) * π * r^2 * h = 50 * π := by
  sorry

end cone_cylinder_volume_relation_l2778_277813


namespace f_2007_equals_neg_two_l2778_277896

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def symmetric_around_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

theorem f_2007_equals_neg_two
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_sym : symmetric_around_two f)
  (h_neg_three : f (-3) = -2) :
  f 2007 = -2 := by
  sorry

end f_2007_equals_neg_two_l2778_277896


namespace equilateral_triangle_side_length_squared_l2778_277809

theorem equilateral_triangle_side_length_squared 
  (α β γ : ℂ) (s t : ℂ) :
  (∀ z, z^3 + s*z + t = 0 ↔ z = α ∨ z = β ∨ z = γ) →
  Complex.abs α ^ 2 + Complex.abs β ^ 2 + Complex.abs γ ^ 2 = 360 →
  ∃ l : ℝ, l > 0 ∧ 
    Complex.abs (α - β) = l ∧
    Complex.abs (β - γ) = l ∧
    Complex.abs (γ - α) = l →
  Complex.abs (α - β) ^ 2 = 360 :=
sorry

end equilateral_triangle_side_length_squared_l2778_277809


namespace perpendicular_lines_slope_product_l2778_277844

/-- Given two lines in the plane, if they are perpendicular, then the product of their slopes is -1 -/
theorem perpendicular_lines_slope_product (a : ℝ) : 
  (∀ x y : ℝ, 2*x + y + 1 = 0 → x + a*y + 3 = 0 → (2 : ℝ) * (1/a) = -1) → 
  a = -2 :=
sorry

end perpendicular_lines_slope_product_l2778_277844


namespace burger_cost_l2778_277887

/-- Given Alice's and Charlie's purchases, prove the cost of a burger -/
theorem burger_cost :
  ∀ (burger_cost soda_cost : ℕ),
  5 * burger_cost + 3 * soda_cost = 500 →
  3 * burger_cost + 2 * soda_cost = 310 →
  burger_cost = 70 := by
sorry

end burger_cost_l2778_277887


namespace rectangle_area_perimeter_inequality_l2778_277876

theorem rectangle_area_perimeter_inequality (a b : ℕ+) : (a + 2) * (b + 2) - 8 ≠ 100 := by
  sorry

end rectangle_area_perimeter_inequality_l2778_277876


namespace triangle_side_length_l2778_277842

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if a = 3, C = 120°, and the area S = 15√3/4, then c = 7. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  a = 3 →
  C = 2 * π / 3 →  -- 120° in radians
  S = 15 * Real.sqrt 3 / 4 →
  S = 1/2 * a * b * Real.sin C →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 7 := by
sorry

end triangle_side_length_l2778_277842


namespace percentage_of_returned_books_l2778_277871

def initial_books : ℕ := 75
def final_books : ℕ := 57
def loaned_books : ℕ := 60

theorem percentage_of_returned_books :
  (initial_books - final_books) * 100 / loaned_books = 70 := by
  sorry

end percentage_of_returned_books_l2778_277871


namespace inequality_proof_l2778_277880

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * (a + b)) + Real.sqrt (b * c * (b + c)) + Real.sqrt (c * a * (c + a)) >
  Real.sqrt ((a + b) * (b + c) * (c + a)) := by
  sorry

end inequality_proof_l2778_277880


namespace peters_parrots_l2778_277832

/-- Calculates the number of parrots Peter has based on the given conditions -/
theorem peters_parrots :
  let parakeet_consumption : ℕ := 2 -- grams per day
  let parrot_consumption : ℕ := 14 -- grams per day
  let finch_consumption : ℕ := parakeet_consumption / 2 -- grams per day
  let num_parakeets : ℕ := 3
  let num_finches : ℕ := 4
  let total_birdseed : ℕ := 266 -- grams for a week
  let days_in_week : ℕ := 7

  let parakeet_weekly_consumption : ℕ := num_parakeets * parakeet_consumption * days_in_week
  let finch_weekly_consumption : ℕ := num_finches * finch_consumption * days_in_week
  let remaining_birdseed : ℕ := total_birdseed - parakeet_weekly_consumption - finch_weekly_consumption
  let parrot_weekly_consumption : ℕ := parrot_consumption * days_in_week

  remaining_birdseed / parrot_weekly_consumption = 2 :=
by sorry

end peters_parrots_l2778_277832


namespace slower_train_speed_l2778_277852

/-- Proves that the speed of the slower train is 37 km/hr given the conditions of the problem -/
theorem slower_train_speed 
  (train_length : ℝ) 
  (faster_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 62.5)
  (h2 : faster_speed = 46)
  (h3 : passing_time = 45)
  : ∃ (slower_speed : ℝ), 
    slower_speed = 37 ∧ 
    2 * train_length = (faster_speed - slower_speed) * (5 / 18) * passing_time :=
by
  sorry

#check slower_train_speed

end slower_train_speed_l2778_277852


namespace complex_power_modulus_l2778_277805

theorem complex_power_modulus : Complex.abs ((1/3 : ℂ) + (2/3 : ℂ) * Complex.I) ^ 8 = 625/6561 := by
  sorry

end complex_power_modulus_l2778_277805


namespace existence_of_sequence_l2778_277878

theorem existence_of_sequence : ∃ (s : List ℕ), 
  (s.length > 10) ∧ 
  (s.sum = 20) ∧ 
  (∀ (i j : ℕ), i ≤ j → j < s.length → (s.take (j + 1)).drop i ≠ [3]) :=
sorry

end existence_of_sequence_l2778_277878


namespace isosceles_trapezoid_bases_l2778_277803

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  lateral_side : ℝ
  height : ℝ
  median : ℝ
  base_small : ℝ
  base_large : ℝ

/-- The theorem stating the bases of the isosceles trapezoid with given properties -/
theorem isosceles_trapezoid_bases
  (t : IsoscelesTrapezoid)
  (h1 : t.lateral_side = 41)
  (h2 : t.height = 40)
  (h3 : t.median = 45) :
  t.base_small = 36 ∧ t.base_large = 54 := by
  sorry

#check isosceles_trapezoid_bases

end isosceles_trapezoid_bases_l2778_277803


namespace max_intersections_circle_quadrilateral_l2778_277875

/-- A circle in a 2D plane. -/
structure Circle where
  -- We don't need to define the specifics of a circle for this problem

/-- A quadrilateral in a 2D plane. -/
structure Quadrilateral where
  -- We don't need to define the specifics of a quadrilateral for this problem

/-- The maximum number of intersection points between a line segment and a circle. -/
def max_intersections_line_circle : ℕ := 2

/-- The number of sides in a quadrilateral. -/
def quadrilateral_sides : ℕ := 4

/-- Theorem: The maximum number of intersection points between a circle and a quadrilateral is 8. -/
theorem max_intersections_circle_quadrilateral (c : Circle) (q : Quadrilateral) :
  (max_intersections_line_circle * quadrilateral_sides : ℕ) = 8 := by
  sorry

end max_intersections_circle_quadrilateral_l2778_277875


namespace square_sum_zero_implies_both_zero_l2778_277894

theorem square_sum_zero_implies_both_zero (a b : ℝ) : 
  a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by sorry

end square_sum_zero_implies_both_zero_l2778_277894


namespace cakes_sold_l2778_277811

theorem cakes_sold (made bought left : ℕ) 
  (h1 : made = 173)
  (h2 : bought = 103)
  (h3 : left = 190) :
  made + bought - left = 86 := by
sorry

end cakes_sold_l2778_277811


namespace tax_savings_proof_l2778_277812

/-- Represents the tax brackets and rates -/
structure TaxBracket :=
  (lower : ℕ) (upper : ℕ) (rate : ℚ)

/-- Calculates the tax for a given income and tax brackets -/
def calculateTax (income : ℕ) (brackets : List TaxBracket) : ℚ :=
  sorry

/-- Represents the tax system -/
structure TaxSystem :=
  (brackets : List TaxBracket)
  (standardDeduction : ℕ)
  (childCredit : ℕ)

/-- Calculates the total tax liability for a given income and tax system -/
def calculateTaxLiability (income : ℕ) (children : ℕ) (system : TaxSystem) : ℚ :=
  sorry

/-- The current tax system -/
def currentSystem : TaxSystem :=
  { brackets := [
      ⟨0, 15000, 15/100⟩,
      ⟨15001, 45000, 42/100⟩,
      ⟨45001, 1000000, 50/100⟩  -- Using a large number for the upper bound of the highest bracket
    ],
    standardDeduction := 3000,
    childCredit := 1000
  }

/-- The proposed tax system -/
def proposedSystem : TaxSystem :=
  { brackets := [
      ⟨0, 15000, 12/100⟩,
      ⟨15001, 45000, 28/100⟩,
      ⟨45001, 1000000, 50/100⟩  -- Using a large number for the upper bound of the highest bracket
    ],
    standardDeduction := 3000,
    childCredit := 1000
  }

theorem tax_savings_proof (income : ℕ) (h : income = 34500) :
  calculateTaxLiability income 2 currentSystem - calculateTaxLiability income 2 proposedSystem = 2760 := by
  sorry

end tax_savings_proof_l2778_277812


namespace current_trees_proof_current_trees_is_25_l2778_277841

/-- The number of popular trees currently in the park -/
def current_trees : ℕ := sorry

/-- The number of popular trees to be planted today -/
def trees_to_plant : ℕ := 73

/-- The total number of popular trees after planting -/
def total_trees : ℕ := 98

/-- Theorem stating that the current number of trees plus the trees to be planted equals the total trees after planting -/
theorem current_trees_proof : current_trees + trees_to_plant = total_trees := by sorry

/-- Theorem proving that the number of current trees is 25 -/
theorem current_trees_is_25 : current_trees = 25 := by sorry

end current_trees_proof_current_trees_is_25_l2778_277841


namespace rolls_combination_count_l2778_277858

theorem rolls_combination_count :
  let total_rolls : ℕ := 8
  let min_per_kind : ℕ := 2
  let num_kinds : ℕ := 3
  let remaining_rolls : ℕ := total_rolls - (min_per_kind * num_kinds)
  Nat.choose (remaining_rolls + num_kinds - 1) (num_kinds - 1) = 6 := by
  sorry

end rolls_combination_count_l2778_277858


namespace correct_calculation_l2778_277830

theorem correct_calculation (x : ℤ) (h : x + 65 = 125) : x + 95 = 155 := by
  sorry

end correct_calculation_l2778_277830


namespace smallest_a_value_l2778_277890

/-- Given a polynomial x^3 - ax^2 + bx - 3003 with three positive integer roots,
    the smallest possible value of a is 45 -/
theorem smallest_a_value (a b : ℤ) (r₁ r₂ r₃ : ℕ+) : 
  (∀ x, x^3 - a*x^2 + b*x - 3003 = (x - r₁)*(x - r₂)*(x - r₃)) →
  a ≥ 45 ∧ ∃ a₀ b₀ r₁₀ r₂₀ r₃₀, 
    a₀ = 45 ∧ 
    (∀ x, x^3 - a₀*x^2 + b₀*x - 3003 = (x - r₁₀)*(x - r₂₀)*(x - r₃₀)) :=
by sorry


end smallest_a_value_l2778_277890


namespace each_angle_less_than_sum_implies_acute_l2778_277898

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the property that each angle is less than the sum of the other two
def each_angle_less_than_sum (t : Triangle) : Prop :=
  t.A < t.B + t.C ∧ t.B < t.A + t.C ∧ t.C < t.A + t.B

-- Define an acute triangle
def is_acute_triangle (t : Triangle) : Prop :=
  t.A < 90 ∧ t.B < 90 ∧ t.C < 90

-- Theorem statement
theorem each_angle_less_than_sum_implies_acute (t : Triangle) :
  each_angle_less_than_sum t → is_acute_triangle t :=
by sorry

end each_angle_less_than_sum_implies_acute_l2778_277898


namespace reading_time_difference_is_360_l2778_277835

/-- Calculates the difference in reading time between two people in minutes -/
def reading_time_difference (xanthia_speed molly_speed book_pages : ℕ) : ℕ :=
  let xanthia_time := book_pages / xanthia_speed
  let molly_time := book_pages / molly_speed
  (molly_time - xanthia_time) * 60

/-- The difference in reading time between Molly and Xanthia is 360 minutes -/
theorem reading_time_difference_is_360 :
  reading_time_difference 120 40 360 = 360 :=
by sorry

end reading_time_difference_is_360_l2778_277835


namespace hen_price_l2778_277862

theorem hen_price (total_cost : ℕ) (pig_price : ℕ) (num_pigs : ℕ) (num_hens : ℕ) :
  total_cost = 1200 →
  pig_price = 300 →
  num_pigs = 3 →
  num_hens = 10 →
  (total_cost - num_pigs * pig_price) / num_hens = 30 :=
by sorry

end hen_price_l2778_277862


namespace percentage_square_divide_l2778_277845

theorem percentage_square_divide (x : ℝ) :
  ((208 / 100 * 1265) ^ 2) / 12 = 576857.87 := by
  sorry

end percentage_square_divide_l2778_277845


namespace a_to_m_eq_2023_l2778_277869

theorem a_to_m_eq_2023 (a m : ℝ) (h : m = Real.sqrt (a - 2023) - Real.sqrt (2023 - a) + 1) : 
  a ^ m = 2023 := by
sorry

end a_to_m_eq_2023_l2778_277869


namespace angle4_measure_l2778_277857

-- Define the angles
def angle1 : ℝ := 85
def angle2 : ℝ := 34
def angle3 : ℝ := 20

-- Define the theorem
theorem angle4_measure : 
  ∀ (angle4 angle5 angle6 : ℝ),
  -- Conditions
  (angle1 + angle2 + angle3 + angle5 + angle6 = 180) →
  (angle4 + angle5 + angle6 = 180) →
  -- Conclusion
  angle4 = 139 := by
sorry

end angle4_measure_l2778_277857


namespace det_scalar_mult_l2778_277883

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -2; 4, 3]
def k : ℝ := 3

theorem det_scalar_mult :
  Matrix.det (k • A) = 207 := by sorry

end det_scalar_mult_l2778_277883


namespace geometric_sequence_common_ratio_l2778_277893

theorem geometric_sequence_common_ratio (q : ℝ) : 
  (1 + q + q^2 = 13) ↔ (q = 3 ∨ q = -4) := by
  sorry

end geometric_sequence_common_ratio_l2778_277893


namespace expression_equals_one_l2778_277817

theorem expression_equals_one : 
  (120^2 - 13^2) / (90^2 - 19^2) * ((90 - 19) * (90 + 19)) / ((120 - 13) * (120 + 13)) = 1 := by
sorry

end expression_equals_one_l2778_277817


namespace inverse_sum_reciprocal_l2778_277815

theorem inverse_sum_reciprocal (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (x * z + y * z + x * y) := by
  sorry

end inverse_sum_reciprocal_l2778_277815


namespace lowest_divisible_by_even_14_to_21_l2778_277853

theorem lowest_divisible_by_even_14_to_21 : ∃! n : ℕ+, 
  (∀ k : ℕ, 14 ≤ k ∧ k ≤ 21 ∧ Even k → (n : ℕ) % k = 0) ∧ 
  (∀ m : ℕ+, (∀ k : ℕ, 14 ≤ k ∧ k ≤ 21 ∧ Even k → (m : ℕ) % k = 0) → n ≤ m) ∧
  n = 5040 := by
sorry

end lowest_divisible_by_even_14_to_21_l2778_277853


namespace cube_surface_area_equal_volume_l2778_277861

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_edge : ℝ) :
  l = 10 ∧ w = 5 ∧ h = 20 →
  cube_edge^3 = l * w * h →
  6 * cube_edge^2 = 600 := by
  sorry

end cube_surface_area_equal_volume_l2778_277861


namespace print_shop_charges_l2778_277872

/-- The charge per color copy at print shop X -/
def charge_x : ℚ := 1.20

/-- The charge per color copy at print shop Y -/
def charge_y : ℚ := 1.70

/-- The number of copies -/
def num_copies : ℕ := 70

/-- The additional charge at print shop Y compared to print shop X -/
def additional_charge : ℚ := 35

theorem print_shop_charges :
  charge_y * num_copies = charge_x * num_copies + additional_charge := by
  sorry

end print_shop_charges_l2778_277872


namespace test_questions_l2778_277870

theorem test_questions (Q : ℝ) : 
  (0.9 * (Q / 2) + 0.95 * (Q / 2) = 74) → Q = 80 := by
  sorry

end test_questions_l2778_277870


namespace games_next_month_l2778_277826

/-- Calculates the number of games Jason plans to attend next month -/
theorem games_next_month 
  (games_this_month : ℕ) 
  (games_last_month : ℕ) 
  (total_games : ℕ) 
  (h1 : games_this_month = 11)
  (h2 : games_last_month = 17)
  (h3 : total_games = 44) :
  total_games - (games_this_month + games_last_month) = 16 := by
sorry

end games_next_month_l2778_277826


namespace wood_cost_is_1_50_l2778_277821

/-- The cost of producing birdhouses and selling them to Danny -/
structure BirdhouseProduction where
  wood_per_birdhouse : ℕ
  profit_per_birdhouse : ℚ
  price_for_two : ℚ

/-- Calculate the cost of each piece of wood -/
def wood_cost (p : BirdhouseProduction) : ℚ :=
  (p.price_for_two - 2 * p.profit_per_birdhouse) / (2 * p.wood_per_birdhouse)

/-- Theorem: Given the conditions, the cost of each piece of wood is $1.50 -/
theorem wood_cost_is_1_50 (p : BirdhouseProduction) 
  (h1 : p.wood_per_birdhouse = 7)
  (h2 : p.profit_per_birdhouse = 11/2)
  (h3 : p.price_for_two = 32) : 
  wood_cost p = 3/2 := by
  sorry

#eval wood_cost ⟨7, 11/2, 32⟩

end wood_cost_is_1_50_l2778_277821


namespace smallest_multiple_l2778_277850

theorem smallest_multiple : ∃ (a : ℕ), 
  (∀ (n : ℕ), n > 0 ∧ 6 ∣ n ∧ 15 ∣ n ∧ n > 40 → a ≤ n) ∧
  6 ∣ a ∧ 15 ∣ a ∧ a > 40 :=
by
  -- The proof goes here
  sorry

end smallest_multiple_l2778_277850


namespace intersection_of_M_and_N_l2778_277836

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (3 - x^2)}

-- Define the closed interval [-1, √3]
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ Real.sqrt 3}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = interval := by sorry

end intersection_of_M_and_N_l2778_277836


namespace jerry_george_sticker_ratio_l2778_277804

/-- The ratio of Jerry's stickers to George's stickers -/
def stickerRatio (jerryStickers georgeStickers : ℕ) : ℚ :=
  jerryStickers / georgeStickers

/-- Proof that the ratio of Jerry's stickers to George's stickers is 3 -/
theorem jerry_george_sticker_ratio :
  let fredStickers : ℕ := 18
  let georgeStickers : ℕ := fredStickers - 6
  let jerryStickers : ℕ := 36
  stickerRatio jerryStickers georgeStickers = 3 := by
sorry

end jerry_george_sticker_ratio_l2778_277804


namespace solve_for_b_l2778_277897

theorem solve_for_b (a b : ℝ) (eq1 : 2 * a + 1 = 1) (eq2 : b + a = 3) : b = 3 := by
  sorry

end solve_for_b_l2778_277897


namespace expected_value_is_thirteen_eighths_l2778_277866

/-- Represents the outcome of a die roll -/
inductive DieOutcome
| Prime (n : Nat)
| NonPrimeSquare (n : Nat)
| Other (n : Nat)

/-- The set of possible outcomes for an 8-sided die -/
def dieOutcomes : Finset DieOutcome := sorry

/-- The probability of each outcome, assuming a fair die -/
def prob (outcome : DieOutcome) : ℚ := sorry

/-- The winnings for each outcome -/
def winnings (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Prime n => n
  | DieOutcome.NonPrimeSquare _ => 2
  | DieOutcome.Other _ => -4

/-- The expected value of winnings for one die toss -/
def expectedValue : ℚ := sorry

/-- Theorem stating that the expected value of winnings is 13/8 -/
theorem expected_value_is_thirteen_eighths :
  expectedValue = 13 / 8 := by sorry

end expected_value_is_thirteen_eighths_l2778_277866


namespace inequality_solution_set_l2778_277843

theorem inequality_solution_set (x : ℝ) :
  (((x + 5) / 2) - 2 < (3 * x + 2) / 2) ↔ (x > -1/2) := by
  sorry

end inequality_solution_set_l2778_277843


namespace min_value_cyclic_sum_l2778_277854

theorem min_value_cyclic_sum (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  (k * a / b) + (k * b / c) + (k * c / a) ≥ 3 * k ∧
  ((k * a / b) + (k * b / c) + (k * c / a) = 3 * k ↔ a = b ∧ b = c) :=
sorry

end min_value_cyclic_sum_l2778_277854


namespace symmetric_point_wrt_y_axis_l2778_277899

/-- Given a point A with coordinates (-2, 4), this theorem proves that the point
    symmetric to A with respect to the y-axis has coordinates (2, 4). -/
theorem symmetric_point_wrt_y_axis :
  let A : ℝ × ℝ := (-2, 4)
  let symmetric_point := (-(A.1), A.2)
  symmetric_point = (2, 4) := by sorry

end symmetric_point_wrt_y_axis_l2778_277899


namespace brad_balloons_l2778_277825

/-- Given that Brad has 8 red balloons and 9 green balloons, prove that he has 17 balloons in total. -/
theorem brad_balloons (red_balloons green_balloons : ℕ) 
  (h1 : red_balloons = 8) 
  (h2 : green_balloons = 9) : 
  red_balloons + green_balloons = 17 := by
  sorry

end brad_balloons_l2778_277825


namespace probability_theorem_l2778_277851

def total_cups : ℕ := 8
def white_cups : ℕ := 3
def red_cups : ℕ := 3
def black_cups : ℕ := 2
def selected_cups : ℕ := 5

def probability_specific_order : ℚ := (white_cups * (white_cups - 1) * red_cups * (red_cups - 1) * black_cups) / 
  (total_cups * (total_cups - 1) * (total_cups - 2) * (total_cups - 3) * (total_cups - 4))

def number_of_arrangements : ℕ := Nat.factorial selected_cups / 
  (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

theorem probability_theorem : 
  (↑number_of_arrangements * probability_specific_order : ℚ) = 9 / 14 := by
  sorry

end probability_theorem_l2778_277851


namespace snow_probability_l2778_277873

-- Define the probability of snow on Friday
def prob_snow_friday : ℝ := 0.4

-- Define the probability of snow on Saturday
def prob_snow_saturday : ℝ := 0.3

-- Define the probability of snow on both days
def prob_snow_both_days : ℝ := prob_snow_friday * prob_snow_saturday

-- Theorem to prove
theorem snow_probability : prob_snow_both_days = 0.12 := by
  sorry

end snow_probability_l2778_277873


namespace arrangement_sum_l2778_277848

theorem arrangement_sum (n : ℕ+) 
  (h1 : n + 3 ≤ 2 * n) 
  (h2 : n + 1 ≤ 4) : 
  Nat.descFactorial (2 * n) (n + 3) + Nat.descFactorial 4 (n + 1) = 744 :=
sorry

end arrangement_sum_l2778_277848


namespace trip_cost_per_person_l2778_277820

/-- Given a group of 11 people and a total cost of $12,100 for a trip,
    the cost per person is $1,100. -/
theorem trip_cost_per_person :
  let total_people : ℕ := 11
  let total_cost : ℕ := 12100
  total_cost / total_people = 1100 := by
  sorry

end trip_cost_per_person_l2778_277820


namespace logical_conclusion_l2778_277868

/-- Represents whether a student submitted all required essays -/
def submitted_all_essays (student : ℕ) : Prop := sorry

/-- Represents whether a student failed the course -/
def failed_course (student : ℕ) : Prop := sorry

/-- Ms. Thompson's statement -/
axiom thompson_statement : ∀ (student : ℕ), ¬(submitted_all_essays student) → failed_course student

/-- The statement to be proved -/
theorem logical_conclusion : ∀ (student : ℕ), ¬(failed_course student) → submitted_all_essays student :=
sorry

end logical_conclusion_l2778_277868


namespace fifteen_percent_of_600_is_90_l2778_277834

theorem fifteen_percent_of_600_is_90 : (15 / 100) * 600 = 90 := by
  sorry

end fifteen_percent_of_600_is_90_l2778_277834


namespace expression_evaluation_l2778_277800

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = -1) :
  2*x*y - 1/2*(4*x*y - 8*x^2*y^2) + 2*(3*x*y - 5*x^2*y^2) = -36 := by
  sorry

end expression_evaluation_l2778_277800


namespace segment_length_bound_polygon_perimeter_bound_l2778_277847

-- Define a segment in 2D plane
structure Segment where
  length : ℝ
  projection1 : ℝ
  projection2 : ℝ

-- Define a polygon in 2D plane
structure Polygon where
  perimeter : ℝ
  totalProjection1 : ℝ
  totalProjection2 : ℝ

-- Theorem for segment
theorem segment_length_bound (s : Segment) : 
  s.length ≥ (s.projection1 + s.projection2) / Real.sqrt 2 := by sorry

-- Theorem for polygon
theorem polygon_perimeter_bound (p : Polygon) :
  p.perimeter ≥ Real.sqrt 2 * (p.totalProjection1 + p.totalProjection2) := by sorry

end segment_length_bound_polygon_perimeter_bound_l2778_277847
