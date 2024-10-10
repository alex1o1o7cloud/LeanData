import Mathlib

namespace arccos_equation_solution_l1263_126314

theorem arccos_equation_solution :
  ∀ x : ℝ, (Real.arccos (3 * x) - Real.arccos x = π / 6) ↔ (x = 1/12 ∨ x = -1/12) :=
by sorry

end arccos_equation_solution_l1263_126314


namespace bus_boarding_problem_l1263_126399

theorem bus_boarding_problem (total_rows : Nat) (seats_per_row : Nat) 
  (initial_boarding : Nat) (first_stop_exit : Nat) (second_stop_boarding : Nat) 
  (second_stop_exit : Nat) (final_empty_seats : Nat) :
  let total_seats := total_rows * seats_per_row
  let empty_seats_after_start := total_seats - initial_boarding
  let first_stop_boarding := total_seats - empty_seats_after_start + first_stop_exit - 
    (total_seats - (empty_seats_after_start - (second_stop_boarding - second_stop_exit) - final_empty_seats))
  total_rows = 23 →
  seats_per_row = 4 →
  initial_boarding = 16 →
  first_stop_exit = 3 →
  second_stop_boarding = 17 →
  second_stop_exit = 10 →
  final_empty_seats = 57 →
  first_stop_boarding = 15 := by
    sorry

#check bus_boarding_problem

end bus_boarding_problem_l1263_126399


namespace water_volume_in_cone_l1263_126379

/-- The volume of water remaining in a conical container after pouring from a cylindrical container -/
theorem water_volume_in_cone (base_radius : ℝ) (height : ℝ) (overflow_volume : ℝ) :
  base_radius > 0 ∧ height > 0 ∧ overflow_volume = 36.2 →
  let cone_volume := (1 / 3) * Real.pi * base_radius^2 * height
  let cylinder_volume := Real.pi * base_radius^2 * height
  overflow_volume = 2 / 3 * cylinder_volume →
  cone_volume = 18.1 := by
sorry

end water_volume_in_cone_l1263_126379


namespace sum_of_largest_and_smallest_l1263_126361

/-- A function that generates all three-digit numbers using the digits 5, 6, and 7 only once -/
def threeDigitNumbers : List Nat := sorry

/-- The smallest three-digit number formed using 5, 6, and 7 only once -/
def smallestNumber : Nat := sorry

/-- The largest three-digit number formed using 5, 6, and 7 only once -/
def largestNumber : Nat := sorry

/-- Theorem stating that the sum of the largest and smallest three-digit numbers
    formed using 5, 6, and 7 only once is 1332 -/
theorem sum_of_largest_and_smallest : smallestNumber + largestNumber = 1332 := by
  sorry

end sum_of_largest_and_smallest_l1263_126361


namespace percentage_problem_l1263_126358

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 100) : 
  (600 / x) * 100 = 120 := by
sorry

end percentage_problem_l1263_126358


namespace exists_quadratic_sequence_l1263_126311

/-- A quadratic sequence is a finite sequence of integers where the absolute difference
    between consecutive terms is equal to the square of their position. -/
def IsQuadraticSequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i ≤ n → |a i - a (i - 1)| = i^2

/-- For any two integers, there exists a quadratic sequence connecting them. -/
theorem exists_quadratic_sequence (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ IsQuadraticSequence a n :=
sorry

end exists_quadratic_sequence_l1263_126311


namespace milkman_profit_is_90_l1263_126396

/-- Calculates the profit of a milkman given the following conditions:
  * The milkman has 30 liters of milk
  * 5 liters of water is mixed with 20 liters of pure milk
  * Water is freely available
  * Cost of pure milk is Rs. 18 per liter
  * Milkman sells all the mixture at cost price
-/
def milkman_profit (total_milk : ℕ) (mixed_milk : ℕ) (water : ℕ) (cost_per_liter : ℕ) : ℕ :=
  let mixture_volume := mixed_milk + water
  let mixture_revenue := mixture_volume * cost_per_liter
  let mixed_milk_cost := mixed_milk * cost_per_liter
  mixture_revenue - mixed_milk_cost

/-- The profit of the milkman is Rs. 90 given the specified conditions. -/
theorem milkman_profit_is_90 :
  milkman_profit 30 20 5 18 = 90 := by
  sorry

end milkman_profit_is_90_l1263_126396


namespace cody_grandmother_age_l1263_126364

/-- Given that Cody is 14 years old and his grandmother is 6 times as old as he is,
    prove that Cody's grandmother is 84 years old. -/
theorem cody_grandmother_age (cody_age : ℕ) (grandmother_age_ratio : ℕ) 
  (h1 : cody_age = 14)
  (h2 : grandmother_age_ratio = 6) :
  cody_age * grandmother_age_ratio = 84 := by
  sorry

end cody_grandmother_age_l1263_126364


namespace quadratic_point_range_l1263_126301

/-- Given a quadratic function y = ax² + 4ax + c with a ≠ 0, and points A, B, C on its graph,
    prove that m < -3 under certain conditions. -/
theorem quadratic_point_range (a c m y₁ y₂ x₀ y₀ : ℝ) : 
  a ≠ 0 →
  y₁ = a * m^2 + 4 * a * m + c →
  y₂ = a * (m + 2)^2 + 4 * a * (m + 2) + c →
  y₀ = a * x₀^2 + 4 * a * x₀ + c →
  x₀ = -2 →
  y₀ ≥ y₂ →
  y₂ > y₁ →
  m < -3 := by
sorry

end quadratic_point_range_l1263_126301


namespace not_divisible_by_five_l1263_126353

theorem not_divisible_by_five (a : ℤ) (h : ¬(5 ∣ a)) : ¬(5 ∣ (3 * a^4 + 1)) := by
  sorry

end not_divisible_by_five_l1263_126353


namespace anayet_speed_l1263_126303

/-- Calculates Anayet's speed given the total distance, Amoli's speed and driving time,
    Anayet's driving time, and the remaining distance. -/
theorem anayet_speed
  (total_distance : ℝ)
  (amoli_speed : ℝ)
  (amoli_time : ℝ)
  (anayet_time : ℝ)
  (remaining_distance : ℝ)
  (h1 : total_distance = 369)
  (h2 : amoli_speed = 42)
  (h3 : amoli_time = 3)
  (h4 : anayet_time = 2)
  (h5 : remaining_distance = 121) :
  (total_distance - (amoli_speed * amoli_time) - remaining_distance) / anayet_time = 61 :=
by sorry

end anayet_speed_l1263_126303


namespace bulb_longevity_probability_l1263_126342

/-- Probability that a bulb from Factory X works for over 4000 hours -/
def prob_x : ℝ := 0.59

/-- Probability that a bulb from Factory Y works for over 4000 hours -/
def prob_y : ℝ := 0.65

/-- Proportion of bulbs supplied by Factory X -/
def supply_x : ℝ := 0.60

/-- Proportion of bulbs supplied by Factory Y -/
def supply_y : ℝ := 1 - supply_x

/-- Theorem stating the probability that a purchased bulb will work for longer than 4000 hours -/
theorem bulb_longevity_probability :
  prob_x * supply_x + prob_y * supply_y = 0.614 := by
  sorry

end bulb_longevity_probability_l1263_126342


namespace solution_value_l1263_126338

theorem solution_value (a : ℝ) (h : 3 * a^2 + 2 * a - 1 = 0) : 
  3 * a^2 + 2 * a - 2019 = -2018 := by
sorry

end solution_value_l1263_126338


namespace gcd_72_120_168_l1263_126389

theorem gcd_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by sorry

end gcd_72_120_168_l1263_126389


namespace smallest_input_129_l1263_126339

def f (n : ℕ+) : ℕ := 9 * n.val + 120

theorem smallest_input_129 :
  ∀ m : ℕ+, f m ≥ f 129 → m ≥ 129 :=
sorry

end smallest_input_129_l1263_126339


namespace parallel_vectors_magnitude_l1263_126370

/-- Given two vectors a and b in R², prove that if they are parallel,
    then the magnitude of a + 2b is 3√5. -/
theorem parallel_vectors_magnitude (t : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, t]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  ‖(a + 2 • b)‖ = 3 * Real.sqrt 5 := by
  sorry

end parallel_vectors_magnitude_l1263_126370


namespace water_bottles_problem_l1263_126348

theorem water_bottles_problem (initial_bottles : ℕ) : 
  (initial_bottles : ℚ) * (2/3) * (1/2) = 8 → initial_bottles = 24 := by
  sorry

end water_bottles_problem_l1263_126348


namespace sin_increasing_omega_range_l1263_126390

theorem sin_increasing_omega_range (ω : ℝ) (f : ℝ → ℝ) :
  ω > 0 →
  (∀ x ∈ Set.Icc 0 (π / 3), f x = Real.sin (ω * x)) →
  StrictMonoOn f (Set.Icc 0 (π / 3)) →
  ω ∈ Set.Ioo 0 (3 / 2) :=
sorry

end sin_increasing_omega_range_l1263_126390


namespace f_is_quadratic_l1263_126371

-- Define a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function
def f (x : ℝ) : ℝ := -4 * x^2 + 5

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end f_is_quadratic_l1263_126371


namespace barry_sotter_magic_l1263_126310

/-- The increase factor for day k --/
def increase_factor (k : ℕ) : ℚ := (k + 3) / (k + 2)

/-- The overall increase factor after n days --/
def overall_increase (n : ℕ) : ℚ := (n + 3) / 3

theorem barry_sotter_magic (n : ℕ) : overall_increase n = 50 → n = 147 := by
  sorry

end barry_sotter_magic_l1263_126310


namespace fixed_point_on_linear_function_l1263_126393

/-- Given a linear function y = kx + b where 3k - b = 2, 
    prove that the point (-3, -2) lies on the graph of the function. -/
theorem fixed_point_on_linear_function (k b : ℝ) 
  (h : 3 * k - b = 2) : 
  k * (-3) + b = -2 := by
  sorry

end fixed_point_on_linear_function_l1263_126393


namespace cubic_sum_value_l1263_126392

theorem cubic_sum_value (w z : ℂ) (h1 : Complex.abs (w + z) = 2) (h2 : Complex.abs (w^2 + z^2) = 10) :
  Complex.abs (w^3 + z^3) = 26 := by
  sorry

end cubic_sum_value_l1263_126392


namespace wall_bricks_count_l1263_126335

theorem wall_bricks_count :
  ∀ (x : ℕ),
  (∃ (rate1 rate2 : ℚ),
    rate1 = x / 9 ∧
    rate2 = x / 10 ∧
    5 * (rate1 + rate2 - 10) = x) →
  x = 900 := by
sorry

end wall_bricks_count_l1263_126335


namespace exists_valid_sign_assignment_l1263_126385

/-- Represents a vertex in the triangular grid --/
structure Vertex :=
  (x : ℤ)
  (y : ℤ)

/-- Represents a triangle in the grid --/
structure Triangle :=
  (a : Vertex)
  (b : Vertex)
  (c : Vertex)

/-- The type of sign assignment functions --/
def SignAssignment := Vertex → Int

/-- Predicate to check if a triangle satisfies the sign rule --/
def satisfiesRule (f : SignAssignment) (t : Triangle) : Prop :=
  (f t.a = f t.b → f t.c = 1) ∧
  (f t.a ≠ f t.b → f t.c = -1)

/-- The set of all triangles in the grid --/
def allTriangles : Set Triangle := sorry

/-- Statement of the theorem --/
theorem exists_valid_sign_assignment :
  ∃ (f : SignAssignment),
    (∀ t ∈ allTriangles, satisfiesRule f t) ∧
    (∃ v w : Vertex, f v ≠ f w) :=
sorry

end exists_valid_sign_assignment_l1263_126385


namespace sequence_general_term_l1263_126318

/-- Given a sequence {a_n} where S_n represents the sum of the first n terms,
    prove that the general term a_n can be expressed as a_1 + (n-1)d,
    where d is the common difference (a_2 - a_1). -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, S k = k / 2 * (a 1 + a k)) →
  ∃ d : ℝ, d = a 2 - a 1 ∧ ∀ m, a m = a 1 + (m - 1) * d :=
by sorry

end sequence_general_term_l1263_126318


namespace quadratic_function_theorem_l1263_126346

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  min_value : ℝ
  min_x : ℝ
  point_1 : ℝ × ℝ
  point_2 : ℝ × ℝ

/-- The theorem statement -/
theorem quadratic_function_theorem (f : QuadraticFunction) 
  (h1 : f.min_value = -3)
  (h2 : f.min_x = -2)
  (h3 : f.point_1 = (1, 10))
  (h4 : f.point_2.1 = 3) :
  f.point_2.2 = 298 / 9 := by
  sorry

end quadratic_function_theorem_l1263_126346


namespace fractional_method_optimization_l1263_126384

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The theorem for the fractional method optimization -/
theorem fractional_method_optimization (range : ℕ) (division_points : ℕ) (n : ℕ) :
  range = 21 →
  division_points = 20 →
  fib (n + 1) - 1 = division_points →
  n = 6 :=
by sorry

end fractional_method_optimization_l1263_126384


namespace intersection_right_isosceles_l1263_126307

-- Define the universe set of all triangles
def Triangle : Type := sorry

-- Define the property of being a right triangle
def IsRight (t : Triangle) : Prop := sorry

-- Define the property of being an isosceles triangle
def IsIsosceles (t : Triangle) : Prop := sorry

-- Define the set of right triangles
def RightTriangles : Set Triangle := {t : Triangle | IsRight t}

-- Define the set of isosceles triangles
def IsoscelesTriangles : Set Triangle := {t : Triangle | IsIsosceles t}

-- Define the property of being both right and isosceles
def IsRightAndIsosceles (t : Triangle) : Prop := IsRight t ∧ IsIsosceles t

-- Define the set of isosceles right triangles
def IsoscelesRightTriangles : Set Triangle := {t : Triangle | IsRightAndIsosceles t}

-- Theorem statement
theorem intersection_right_isosceles :
  RightTriangles ∩ IsoscelesTriangles = IsoscelesRightTriangles := by sorry

end intersection_right_isosceles_l1263_126307


namespace sequence_formulas_l1263_126332

/-- Given a sequence {a_n} with sum of first n terms S_n satisfying S_n = 2 - a_n,
    and sequence {b_n} satisfying b_1 = 1 and b_{n+1} = b_n + a_n,
    prove the general term formulas for both sequences. -/
theorem sequence_formulas (a b : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → S n = 2 - a n) →
  b 1 = 1 →
  (∀ n : ℕ, n ≥ 1 → b (n + 1) = b n + a n) →
  (∀ n : ℕ, n ≥ 1 → a n = (1/2)^(n-1)) ∧
  (∀ n : ℕ, n ≥ 2 → b n = 3 - 1/(2^(n-2))) :=
by sorry

end sequence_formulas_l1263_126332


namespace math_club_team_selection_l1263_126376

theorem math_club_team_selection (boys girls team_size : ℕ) 
  (h1 : boys = 7) 
  (h2 : girls = 9) 
  (h3 : team_size = 5) : 
  Nat.choose (boys + girls) team_size = 4368 := by
  sorry

end math_club_team_selection_l1263_126376


namespace parabola_through_point_l1263_126360

/-- A parabola passing through the point (4, -2) has either the equation y² = x or x² = -8y -/
theorem parabola_through_point (P : ℝ × ℝ) (h : P = (4, -2)) :
  (∃ (x y : ℝ), y^2 = x ∧ P = (x, y)) ∨ (∃ (x y : ℝ), x^2 = -8*y ∧ P = (x, y)) :=
sorry

end parabola_through_point_l1263_126360


namespace area_at_stage_4_l1263_126345

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℕ := r.width * r.length

/-- Represents the growth of the rectangle at each stage --/
def grow (r : Rectangle) : Rectangle :=
  { width := r.width + 2, length := r.length + 3 }

/-- Calculates the rectangle at a given stage --/
def rectangleAtStage (n : ℕ) : Rectangle :=
  match n with
  | 0 => { width := 2, length := 3 }
  | n + 1 => grow (rectangleAtStage n)

theorem area_at_stage_4 : area (rectangleAtStage 4) = 150 := by
  sorry

end area_at_stage_4_l1263_126345


namespace imaginary_unit_power_l1263_126350

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2015 = -i := by sorry

end imaginary_unit_power_l1263_126350


namespace rectangular_plot_width_l1263_126347

theorem rectangular_plot_width
  (length : ℝ)
  (num_poles : ℕ)
  (pole_distance : ℝ)
  (width : ℝ)
  (h1 : length = 60)
  (h2 : num_poles = 44)
  (h3 : pole_distance = 5)
  (h4 : 2 * (length + width) = pole_distance * num_poles) :
  width = 50 := by
sorry

end rectangular_plot_width_l1263_126347


namespace problem_solution_l1263_126302

theorem problem_solution (n : ℝ) : 32 - 16 = n * 4 → (n / 4) + 16 = 17 := by
  sorry

end problem_solution_l1263_126302


namespace pau_total_cost_l1263_126362

/-- Represents the cost of fried chicken orders for three people -/
def fried_chicken_cost 
  (kobe_pieces : ℕ) 
  (kobe_price : ℚ) 
  (pau_multiplier : ℕ) 
  (pau_extra : ℚ) 
  (pau_price : ℚ) 
  (shaq_multiplier : ℚ) 
  (discount : ℚ) : ℚ :=
  let pau_pieces := pau_multiplier * kobe_pieces + pau_extra
  let pau_initial := pau_pieces * pau_price
  pau_initial + pau_initial * (1 - discount)

/-- Theorem stating the total cost of Pau's fried chicken orders -/
theorem pau_total_cost : 
  fried_chicken_cost 5 (175/100) 2 (5/2) (3/2) (3/2) (15/100) = 346875/10000 := by
  sorry

end pau_total_cost_l1263_126362


namespace initial_students_count_l1263_126322

/-- The number of students initially on the bus -/
def initial_students : ℕ := sorry

/-- The number of students who got on at the first stop -/
def students_who_got_on : ℕ := 3

/-- The total number of students on the bus after the first stop -/
def total_students : ℕ := 13

/-- Theorem stating that the initial number of students was 10 -/
theorem initial_students_count : initial_students = 10 := by
  sorry

end initial_students_count_l1263_126322


namespace boat_current_rate_l1263_126331

/-- Proves that the rate of current is 5 km/hr given the conditions of the boat problem -/
theorem boat_current_rate (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 20 →
  downstream_distance = 5 →
  downstream_time = 1/5 →
  ∃ (current_rate : ℝ), 
    downstream_distance = (boat_speed + current_rate) * downstream_time ∧
    current_rate = 5 :=
by sorry

end boat_current_rate_l1263_126331


namespace random_placement_probability_l1263_126356

-- Define the number of bins and items
def num_bins : ℕ := 4
def num_items : ℕ := 4

-- Define the probability of correct placement
def correct_placement_probability : ℚ := 1 / (num_bins.factorial)

-- Theorem statement
theorem random_placement_probability :
  correct_placement_probability = 1 / 24 := by
  sorry

end random_placement_probability_l1263_126356


namespace cross_sectional_area_of_cone_l1263_126368

-- Define the cone
structure Cone :=
  (baseRadius : ℝ)
  (height : ℝ)

-- Define the cutting plane
structure CuttingPlane :=
  (distanceFromBase : ℝ)
  (isParallelToBase : Bool)

-- Theorem statement
theorem cross_sectional_area_of_cone (c : Cone) (p : CuttingPlane) :
  c.baseRadius = 2 →
  p.distanceFromBase = c.height / 2 →
  p.isParallelToBase = true →
  (π : ℝ) = π := by sorry

end cross_sectional_area_of_cone_l1263_126368


namespace train_bridge_crossing_time_l1263_126315

theorem train_bridge_crossing_time (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) :
  train_length = 130 →
  bridge_length = 150 →
  train_speed_kmph = 36 →
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 28 := by
  sorry

end train_bridge_crossing_time_l1263_126315


namespace john_ate_three_slices_l1263_126349

/-- Represents the number of slices in a pizza -/
def total_slices : ℕ := 12

/-- Represents the number of slices left -/
def slices_left : ℕ := 3

/-- Represents the number of slices John ate -/
def john_slices : ℕ := 3

/-- Represents the number of slices Sam ate -/
def sam_slices : ℕ := 2 * john_slices

theorem john_ate_three_slices :
  john_slices = 3 ∧
  sam_slices = 2 * john_slices ∧
  total_slices = john_slices + sam_slices + slices_left :=
by sorry

end john_ate_three_slices_l1263_126349


namespace zoo_lion_cubs_l1263_126375

theorem zoo_lion_cubs (initial_count final_count : ℕ) 
  (gorillas_sent : ℕ) (hippo_adopted : ℕ) (rhinos_taken : ℕ) : 
  initial_count = 68 →
  gorillas_sent = 6 →
  hippo_adopted = 1 →
  rhinos_taken = 3 →
  final_count = 90 →
  ∃ (lion_cubs : ℕ), 
    final_count = initial_count - gorillas_sent + hippo_adopted + rhinos_taken + lion_cubs + 2 * lion_cubs ∧
    lion_cubs = 8 :=
by
  sorry

end zoo_lion_cubs_l1263_126375


namespace max_sum_of_squares_l1263_126377

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 83 →
  a * d + b * c = 174 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 702 :=
by
  sorry


end max_sum_of_squares_l1263_126377


namespace cubic_roots_arithmetic_progression_l1263_126391

/-- A cubic polynomial with coefficients a, b, and c -/
def cubic_polynomial (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The condition for the roots of a cubic polynomial to form an arithmetic progression -/
def arithmetic_progression_condition (a b c : ℝ) : Prop :=
  2 * a^3 / 27 - a * b / 3 + c = 0

/-- Theorem stating that the roots of a cubic polynomial form an arithmetic progression
    if and only if the coefficients satisfy the arithmetic progression condition -/
theorem cubic_roots_arithmetic_progression (a b c : ℝ) :
  (∃ x y z : ℝ, x - y = y - z ∧ 
    (∀ t : ℝ, cubic_polynomial a b c t = 0 ↔ t = x ∨ t = y ∨ t = z)) ↔ 
  arithmetic_progression_condition a b c :=
sorry

end cubic_roots_arithmetic_progression_l1263_126391


namespace substitution_remainder_l1263_126369

/-- Represents the number of players on the roster. -/
def totalPlayers : ℕ := 15

/-- Represents the number of players in the starting lineup. -/
def startingLineup : ℕ := 10

/-- Represents the number of substitute players. -/
def substitutes : ℕ := 5

/-- Represents the maximum number of substitutions allowed. -/
def maxSubstitutions : ℕ := 2

/-- Calculates the number of ways to make substitutions given the number of substitutions. -/
def substitutionWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => startingLineup * substitutes
  | 2 => startingLineup * substitutes * (startingLineup - 1) * (substitutes - 1)
  | _ => 0

/-- Calculates the total number of possible substitution scenarios. -/
def totalScenarios : ℕ :=
  (List.range (maxSubstitutions + 1)).map substitutionWays |>.sum

/-- The main theorem stating that the remainder of totalScenarios divided by 500 is 351. -/
theorem substitution_remainder :
  totalScenarios % 500 = 351 := by
  sorry

end substitution_remainder_l1263_126369


namespace min_value_expression_l1263_126343

theorem min_value_expression (a b : ℝ) (h1 : a * b - 2 * a - b + 1 = 0) (h2 : a > 1) :
  ∀ x y : ℝ, x * y - 2 * x - y + 1 = 0 → x > 1 → (a + 3) * (b + 2) ≤ (x + 3) * (y + 2) ∧
  ∃ a₀ b₀ : ℝ, a₀ * b₀ - 2 * a₀ - b₀ + 1 = 0 ∧ a₀ > 1 ∧ (a₀ + 3) * (b₀ + 2) = 25 :=
sorry

end min_value_expression_l1263_126343


namespace paulas_travel_time_fraction_l1263_126327

theorem paulas_travel_time_fraction (luke_bus_time : ℕ) (total_travel_time : ℕ) 
  (h1 : luke_bus_time = 70)
  (h2 : total_travel_time = 504) :
  ∃ f : ℚ, 
    f = 3/5 ∧ 
    (luke_bus_time + 5 * luke_bus_time + 2 * (f * luke_bus_time) : ℚ) = total_travel_time :=
by sorry

end paulas_travel_time_fraction_l1263_126327


namespace largest_d_for_two_in_range_l1263_126363

/-- The function g(x) defined as x^2 - 6x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + d

/-- Theorem stating that the largest value of d for which 2 is in the range of g(x) is 11 -/
theorem largest_d_for_two_in_range :
  (∃ (d : ℝ), ∀ (e : ℝ), (∃ (x : ℝ), g d x = 2) → (e ≤ d)) ∧
  (∃ (x : ℝ), g 11 x = 2) :=
sorry

end largest_d_for_two_in_range_l1263_126363


namespace geometric_sequence_property_l1263_126372

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 3 * a 3 - 6 * a 3 + 8 = 0) →
  (a 15 * a 15 - 6 * a 15 + 8 = 0) →
  (a 1 * a 17) / a 9 = 2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_property_l1263_126372


namespace complex_modulus_problem_l1263_126300

theorem complex_modulus_problem (z : ℂ) : z = (2 * Complex.I) / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l1263_126300


namespace negative_one_power_difference_l1263_126305

theorem negative_one_power_difference : (-1 : ℤ)^5 - (-1 : ℤ)^4 = -2 := by
  sorry

end negative_one_power_difference_l1263_126305


namespace percent_relation_l1263_126380

theorem percent_relation (x y z : ℝ) 
  (h1 : x = 1.20 * y) 
  (h2 : y = 0.30 * z) : 
  x = 0.36 * z := by
sorry

end percent_relation_l1263_126380


namespace min_value_of_expression_l1263_126333

/-- The set of available numbers -/
def S : Finset Int := {-9, -6, -4, -1, 3, 5, 7, 12}

/-- The expression to be minimized -/
def f (p q r s t u v w : Int) : ℚ :=
  ((p + q + r + s : ℚ) ^ 2 + (t + u + v + w : ℚ) ^ 2 : ℚ)

/-- The theorem stating the minimum value of the expression -/
theorem min_value_of_expression :
  ∀ p q r s t u v w : Int,
    p ∈ S → q ∈ S → r ∈ S → s ∈ S → t ∈ S → u ∈ S → v ∈ S → w ∈ S →
    p ≠ q → p ≠ r → p ≠ s → p ≠ t → p ≠ u → p ≠ v → p ≠ w →
    q ≠ r → q ≠ s → q ≠ t → q ≠ u → q ≠ v → q ≠ w →
    r ≠ s → r ≠ t → r ≠ u → r ≠ v → r ≠ w →
    s ≠ t → s ≠ u → s ≠ v → s ≠ w →
    t ≠ u → t ≠ v → t ≠ w →
    u ≠ v → u ≠ w →
    v ≠ w →
    f p q r s t u v w ≥ 26.5 :=
by sorry

end min_value_of_expression_l1263_126333


namespace age_difference_proof_l1263_126313

theorem age_difference_proof (people : Fin 5 → ℕ) 
  (h1 : people 0 = people 1 + 1)
  (h2 : people 2 = people 3 + 2)
  (h3 : people 4 = people 5 + 3)
  (h4 : people 6 = people 7 + 4) :
  people 9 = people 8 + 10 := by
  sorry

end age_difference_proof_l1263_126313


namespace min_dot_product_in_triangle_l1263_126365

theorem min_dot_product_in_triangle (A B C : ℝ × ℝ) : 
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let angle_A := Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / 
                  (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))
  BC = 2 → angle_A = 2 * Real.pi / 3 → 
  (∀ A' B' C' : ℝ × ℝ, 
    let BC' := Real.sqrt ((B'.1 - C'.1)^2 + (B'.2 - C'.2)^2)
    let angle_A' := Real.arccos ((B'.1 - A'.1) * (C'.1 - A'.1) + (B'.2 - A'.2) * (C'.2 - A'.2)) / 
                    (Real.sqrt ((B'.1 - A'.1)^2 + (B'.2 - A'.2)^2) * Real.sqrt ((C'.1 - A'.1)^2 + (C'.2 - A'.2)^2))
    BC' = 2 → angle_A' = 2 * Real.pi / 3 → 
    ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) ≤ 
    ((B'.1 - A'.1) * (C'.1 - A'.1) + (B'.2 - A'.2) * (C'.2 - A'.2))) →
  ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) = -2/3 := by
sorry

end min_dot_product_in_triangle_l1263_126365


namespace gcf_of_45_and_75_l1263_126381

theorem gcf_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcf_of_45_and_75_l1263_126381


namespace april_earnings_l1263_126359

def rose_price : ℕ := 7
def initial_roses : ℕ := 9
def remaining_roses : ℕ := 4

theorem april_earnings : (initial_roses - remaining_roses) * rose_price = 35 := by
  sorry

end april_earnings_l1263_126359


namespace events_mutually_exclusive_events_not_opposite_mutually_exclusive_not_opposite_l1263_126344

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 2

/-- Represents the total number of students in the group -/
def total_students : ℕ := num_boys + num_girls

/-- Represents the number of students selected -/
def selected_students : ℕ := 2

/-- Represents the event of exactly one boy being selected -/
def one_boy_selected (k : ℕ) : Prop := k = 1

/-- Represents the event of exactly two boys being selected -/
def two_boys_selected (k : ℕ) : Prop := k = 2

/-- States that the events are mutually exclusive -/
theorem events_mutually_exclusive : 
  ∀ k : ℕ, ¬(one_boy_selected k ∧ two_boys_selected k) :=
sorry

/-- States that the events are not opposite -/
theorem events_not_opposite : 
  ∃ k : ℕ, ¬(one_boy_selected k ∨ two_boys_selected k) :=
sorry

/-- Main theorem stating that the events are mutually exclusive but not opposite -/
theorem mutually_exclusive_not_opposite : 
  (∀ k : ℕ, ¬(one_boy_selected k ∧ two_boys_selected k)) ∧
  (∃ k : ℕ, ¬(one_boy_selected k ∨ two_boys_selected k)) :=
sorry

end events_mutually_exclusive_events_not_opposite_mutually_exclusive_not_opposite_l1263_126344


namespace max_k_value_l1263_126323

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 13) / 2 :=
by sorry

end max_k_value_l1263_126323


namespace arithmetic_calculations_l1263_126304

theorem arithmetic_calculations :
  (24 - (-16) + (-25) - 15 = 0) ∧
  ((-81) + 2.25 * (4/9) / (-16) = -81 - 1/16) := by
  sorry

end arithmetic_calculations_l1263_126304


namespace triangle_angle_proof_l1263_126312

theorem triangle_angle_proof (A B C : Real) (a b c : Real) :
  -- Conditions
  (a = 2) →
  (b = Real.sqrt 3) →
  (B = π / 3) →
  -- Triangle definition (implicitly assuming it's a valid triangle)
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  -- Sine law
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion
  A = π / 2 := by
  sorry

end triangle_angle_proof_l1263_126312


namespace max_games_purchasable_l1263_126394

def initial_amount : ℕ := 35
def spent_amount : ℕ := 7
def game_cost : ℕ := 4

theorem max_games_purchasable :
  (initial_amount - spent_amount) / game_cost = 7 := by
  sorry

end max_games_purchasable_l1263_126394


namespace empty_solution_set_implies_k_range_l1263_126378

theorem empty_solution_set_implies_k_range (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 ≥ 0) → k ∈ Set.Icc (-2) 2 := by
  sorry

end empty_solution_set_implies_k_range_l1263_126378


namespace friends_attended_reception_l1263_126340

/-- The number of friends attending a wedding reception --/
def friends_at_reception (total_guests : ℕ) (family_couples : ℕ) (coworkers : ℕ) (distant_relatives : ℕ) : ℕ :=
  total_guests - (2 * (2 * family_couples + coworkers + distant_relatives))

/-- Theorem: Given the conditions of the wedding reception, 180 friends attended --/
theorem friends_attended_reception :
  friends_at_reception 400 40 10 20 = 180 := by
  sorry

end friends_attended_reception_l1263_126340


namespace imaginary_part_of_z_l1263_126321

theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) / Complex.I = 1 - z) : 
  z.im = -1 := by sorry

end imaginary_part_of_z_l1263_126321


namespace certain_number_proof_l1263_126308

theorem certain_number_proof : ∃ x : ℝ, 45 * x = 0.35 * 900 ∧ x = 7 := by
  sorry

end certain_number_proof_l1263_126308


namespace smallest_root_property_l1263_126354

theorem smallest_root_property : ∃ a : ℝ, 
  (∀ x : ℝ, x^2 - 9*x - 10 = 0 → a ≤ x) ∧ 
  (a^2 - 9*a - 10 = 0) ∧
  (a^4 - 909*a = 910) := by
  sorry

end smallest_root_property_l1263_126354


namespace arithmetic_sequence_sum_l1263_126383

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 3 + a 10 + a 11 = 40 →
  a 6 + a 7 = 20 := by
  sorry

end arithmetic_sequence_sum_l1263_126383


namespace correlation_coefficient_properties_l1263_126395

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

-- Define the concept of increasing
def increasing (f : ℝ → ℝ) : Prop := 
  ∀ a b, a < b → f a < f b

-- Define the concept of linear correlation strength
def linear_correlation_strength (r : ℝ) : ℝ := sorry

-- Define the concept of functional relationship
def functional_relationship (x y : ℝ → ℝ) : Prop := sorry

theorem correlation_coefficient_properties 
  (x y : ℝ → ℝ) (r : ℝ) (h : r = correlation_coefficient x y) :
  (r > 0 → increasing y) ∧ 
  (∀ s : ℝ, abs s < abs r → linear_correlation_strength s < linear_correlation_strength r) ∧
  ((r = 1 ∨ r = -1) → functional_relationship x y) := by
  sorry

end correlation_coefficient_properties_l1263_126395


namespace pentagon_from_reflections_l1263_126355

/-- Given a set of reflection points, there exists a unique pentagon satisfying the reflection properties. -/
theorem pentagon_from_reflections (B : Fin 5 → ℝ × ℝ) :
  ∃! (A : Fin 5 → ℝ × ℝ), ∀ i : Fin 5, B i = 2 * A (i.succ) - A i :=
by sorry

end pentagon_from_reflections_l1263_126355


namespace infinitely_many_perfect_squares_l1263_126352

/-- An arithmetic sequence of natural numbers -/
def arithmeticSequence (a d : ℕ) (n : ℕ) : ℕ := a + n * d

/-- Predicate for perfect squares -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem infinitely_many_perfect_squares
  (a d : ℕ) -- First term and common difference of the sequence
  (h : ∃ n₀ : ℕ, isPerfectSquare (arithmeticSequence a d n₀)) :
  ∀ m : ℕ, ∃ n > m, isPerfectSquare (arithmeticSequence a d n) :=
sorry

end infinitely_many_perfect_squares_l1263_126352


namespace four_bb_two_divisible_by_9_l1263_126329

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def digit_sum (B : ℕ) : ℕ :=
  4 + B + B + 2

theorem four_bb_two_divisible_by_9 (B : ℕ) (h1 : B < 10) :
  is_divisible_by_9 (4000 + 100 * B + 10 * B + 2) ↔ B = 6 :=
by
  sorry

end four_bb_two_divisible_by_9_l1263_126329


namespace new_year_markup_l1263_126316

theorem new_year_markup (initial_markup : ℝ) (february_discount : ℝ) (final_profit : ℝ) :
  initial_markup = 0.20 →
  february_discount = 0.12 →
  final_profit = 0.32 →
  ∃ (new_year_markup : ℝ),
    (1 + initial_markup) * (1 + new_year_markup) * (1 - february_discount) = 1 + final_profit ∧
    new_year_markup = 0.25 :=
by sorry

end new_year_markup_l1263_126316


namespace james_weekly_beats_l1263_126341

/-- The number of beats James hears per week -/
def beats_per_week : ℕ :=
  let beats_per_minute : ℕ := 200
  let hours_per_day : ℕ := 2
  let minutes_per_hour : ℕ := 60
  let days_per_week : ℕ := 7
  beats_per_minute * hours_per_day * minutes_per_hour * days_per_week

/-- Theorem stating that James hears 168,000 beats per week -/
theorem james_weekly_beats : beats_per_week = 168000 := by
  sorry

end james_weekly_beats_l1263_126341


namespace parabola_vertex_y_coordinate_l1263_126388

theorem parabola_vertex_y_coordinate (x y : ℝ) :
  y = -6 * x^2 + 24 * x - 7 →
  ∃ h k : ℝ, y = -6 * (x - h)^2 + k ∧ k = 17 := by
  sorry

end parabola_vertex_y_coordinate_l1263_126388


namespace inequality_proof_l1263_126367

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) : a^3 * b^2 < a^2 * b^3 := by
  sorry

end inequality_proof_l1263_126367


namespace inequality_and_system_solution_l1263_126309

theorem inequality_and_system_solution :
  (∀ x : ℝ, (2*x - 3)/3 > (3*x + 1)/6 - 1 ↔ x > 1) ∧
  (∀ x : ℝ, x ≤ 3*x - 6 ∧ 3*x + 1 > 2*(x - 1) ↔ x ≥ 3) := by
  sorry

end inequality_and_system_solution_l1263_126309


namespace initial_distance_between_trains_l1263_126387

/-- Proves that the initial distance between two trains is 200 meters. -/
theorem initial_distance_between_trains (length1 length2 : ℝ) (speed1 speed2 : ℝ) (time : ℝ) :
  length1 = 90 →
  length2 = 100 →
  speed1 = 71 * 1000 / 3600 →
  speed2 = 89 * 1000 / 3600 →
  time = 4.499640028797696 →
  speed1 * time + speed2 * time = 200 := by
  sorry

end initial_distance_between_trains_l1263_126387


namespace coin_problem_l1263_126328

/-- Represents the number of different values that can be produced with given coins -/
def different_values (five_cent_coins ten_cent_coins : ℕ) : ℕ :=
  29 - five_cent_coins

theorem coin_problem (total_coins : ℕ) (distinct_values : ℕ) 
  (h1 : total_coins = 15)
  (h2 : distinct_values = 26) :
  ∃ (five_cent_coins ten_cent_coins : ℕ),
    five_cent_coins + ten_cent_coins = total_coins ∧
    different_values five_cent_coins ten_cent_coins = distinct_values ∧
    ten_cent_coins = 12 := by
  sorry

end coin_problem_l1263_126328


namespace total_oranges_l1263_126397

theorem total_oranges (oranges_per_child : ℕ) (num_children : ℕ) 
  (h1 : oranges_per_child = 3) 
  (h2 : num_children = 4) : 
  oranges_per_child * num_children = 12 :=
by sorry

end total_oranges_l1263_126397


namespace no_real_roots_iff_k_gt_9_l1263_126320

theorem no_real_roots_iff_k_gt_9 (k : ℝ) : 
  (∀ x : ℝ, x^2 + k ≠ 6*x) ↔ k > 9 := by
sorry

end no_real_roots_iff_k_gt_9_l1263_126320


namespace madeline_rent_correct_l1263_126357

/-- Calculate the amount Madeline needs for rent given her expenses, savings, hourly wage, and hours worked -/
def rent_amount (hourly_wage : ℝ) (hours_worked : ℝ) (groceries : ℝ) (medical : ℝ) (utilities : ℝ) (savings : ℝ) : ℝ :=
  hourly_wage * hours_worked - (groceries + medical + utilities + savings)

/-- Theorem stating that Madeline's rent amount is correct -/
theorem madeline_rent_correct (hourly_wage : ℝ) (hours_worked : ℝ) (groceries : ℝ) (medical : ℝ) (utilities : ℝ) (savings : ℝ) :
  rent_amount hourly_wage hours_worked groceries medical utilities savings = 1210 :=
by
  sorry

#eval rent_amount 15 138 400 200 60 200

end madeline_rent_correct_l1263_126357


namespace wig_cost_calculation_l1263_126398

-- Define the given conditions
def total_plays : ℕ := 3
def acts_per_play : ℕ := 5
def wigs_per_act : ℕ := 2
def dropped_play_sale : ℚ := 4
def total_spent : ℚ := 110

-- Define the theorem
theorem wig_cost_calculation :
  let wigs_per_play := acts_per_play * wigs_per_act
  let total_wigs := total_plays * wigs_per_play
  let remaining_wigs := total_wigs - wigs_per_play
  let cost_per_wig := total_spent / remaining_wigs
  cost_per_wig = 5.5 := by sorry

end wig_cost_calculation_l1263_126398


namespace line_through_point_parallel_to_line_line_through_point_parallel_to_line_proof_l1263_126337

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem line_through_point_parallel_to_line 
  (given_line : Line) 
  (given_point : Point) 
  (result_line : Line) : Prop :=
  given_line.a = 1 ∧ 
  given_line.b = -2 ∧ 
  given_line.c = 3 ∧
  given_point.x = -1 ∧
  given_point.y = 3 ∧
  result_line.a = 1 ∧
  result_line.b = -2 ∧
  result_line.c = 7 →
  given_point.liesOn result_line ∧
  result_line.isParallelTo given_line

-- The proof of the theorem
theorem line_through_point_parallel_to_line_proof 
  (given_line : Line) 
  (given_point : Point) 
  (result_line : Line) : 
  line_through_point_parallel_to_line given_line given_point result_line :=
by
  sorry -- Proof is omitted as per instructions

end line_through_point_parallel_to_line_line_through_point_parallel_to_line_proof_l1263_126337


namespace yuna_has_most_points_l1263_126326

-- Define the point totals for each person
def yoongi_points : ℕ := 7
def jungkook_points : ℕ := 6
def yuna_points : ℕ := 9
def yoojung_points : ℕ := 8

-- Theorem stating that Yuna has the largest number of points
theorem yuna_has_most_points :
  yuna_points ≥ yoongi_points ∧
  yuna_points ≥ jungkook_points ∧
  yuna_points ≥ yoojung_points :=
by sorry

end yuna_has_most_points_l1263_126326


namespace min_sum_of_square_areas_l1263_126317

theorem min_sum_of_square_areas (wire_length : ℝ) (h : wire_length = 16) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ wire_length ∧
  (x^2 + (wire_length - x)^2 ≥ 8 ∧
   ∀ (y : ℝ), 0 ≤ y ∧ y ≤ wire_length →
     y^2 + (wire_length - y)^2 ≥ x^2 + (wire_length - x)^2) :=
by sorry

end min_sum_of_square_areas_l1263_126317


namespace eagles_falcons_games_l1263_126306

theorem eagles_falcons_games (N : ℕ) : 
  (∀ n : ℕ, n < N → (3 + n : ℚ) / (7 + n) < 9/10) ∧ 
  (3 + N : ℚ) / (7 + N) ≥ 9/10 → 
  N = 33 :=
sorry

end eagles_falcons_games_l1263_126306


namespace water_in_pool_l1263_126373

-- Define the parameters
def initial_bucket : ℝ := 1
def additional_buckets : ℝ := 8.8
def liters_per_bucket : ℝ := 10
def evaporation_rate : ℝ := 0.2
def splashing_rate : ℝ := 0.5
def time_taken : ℝ := 20

-- Define the theorem
theorem water_in_pool : 
  let total_buckets := initial_bucket + additional_buckets
  let total_water := total_buckets * liters_per_bucket
  let evaporation_loss := evaporation_rate * time_taken
  let splashing_loss := splashing_rate * time_taken
  let total_loss := evaporation_loss + splashing_loss
  let net_water := total_water - total_loss
  net_water = 84 := by
  sorry


end water_in_pool_l1263_126373


namespace imaginary_part_of_z_l1263_126330

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (-1 + Complex.I) → z.im = -1 := by
  sorry

end imaginary_part_of_z_l1263_126330


namespace repeating_decimal_division_l1263_126334

/-- Represents a repeating decimal with a whole number part and a repeating part -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def repeating_decimal_to_rational (d : RepeatingDecimal) : ℚ :=
  d.whole + d.repeating / (99 : ℚ)

/-- The theorem stating that the division of two specific repeating decimals equals 3/10 -/
theorem repeating_decimal_division :
  let d1 := RepeatingDecimal.mk 0 81
  let d2 := RepeatingDecimal.mk 2 72
  (repeating_decimal_to_rational d1) / (repeating_decimal_to_rational d2) = 3 / 10 := by
  sorry

end repeating_decimal_division_l1263_126334


namespace smallest_m_has_n_14_l1263_126324

def is_valid_m (m : ℕ) : Prop :=
  ∃ (n : ℕ) (r : ℝ), 
    n > 0 ∧ 
    r > 0 ∧ 
    r < 1/10000 ∧ 
    m^(1/4 : ℝ) = n + r

theorem smallest_m_has_n_14 : 
  ∃ (m : ℕ), is_valid_m m ∧ 
  (∀ (k : ℕ), k < m → ¬is_valid_m k) ∧
  (∃ (r : ℝ), m^(1/4 : ℝ) = 14 + r ∧ r > 0 ∧ r < 1/10000) :=
sorry

end smallest_m_has_n_14_l1263_126324


namespace sufficient_not_necessary_l1263_126382

theorem sufficient_not_necessary (a b : ℝ) :
  (a < b ∧ b < 0 → a^2 > b^2) ∧
  ¬(a^2 > b^2 → a < b ∧ b < 0) := by
  sorry

end sufficient_not_necessary_l1263_126382


namespace train_crossing_time_l1263_126374

/-- Time taken for a faster train to cross a slower train moving in the same direction -/
theorem train_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 300) 
  (h2 : length2 = 500)
  (h3 : speed1 = 72)
  (h4 : speed2 = 36)
  (h5 : speed1 > speed2) : 
  (length1 + length2) / ((speed1 - speed2) * (1000 / 3600)) = 80 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1263_126374


namespace min_value_of_squared_differences_l1263_126386

theorem min_value_of_squared_differences (a α β : ℝ) : 
  (α^2 - 2*a*α + a + 6 = 0) →
  (β^2 - 2*a*β + a + 6 = 0) →
  α ≠ β →
  ∃ (m : ℝ), m = 8 ∧ ∀ (x y : ℝ), 
    (x^2 - 2*a*x + a + 6 = 0) → 
    (y^2 - 2*a*y + a + 6 = 0) → 
    x ≠ y → 
    (x - 1)^2 + (y - 1)^2 ≥ m :=
by sorry

end min_value_of_squared_differences_l1263_126386


namespace nine_digit_repeat_gcd_l1263_126319

theorem nine_digit_repeat_gcd : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), 100 ≤ m ∧ m < 1000 → 
    (∃ (k : ℕ), k = m * 1001001 ∧ 
      Nat.gcd n k = n)) ∧ 
  (∀ (d : ℕ), d > n → 
    ∃ (m₁ m₂ : ℕ), 100 ≤ m₁ ∧ m₁ < 1000 ∧ 100 ≤ m₂ ∧ m₂ < 1000 ∧ 
      Nat.gcd (m₁ * 1001001) (m₂ * 1001001) < d) :=
by sorry

end nine_digit_repeat_gcd_l1263_126319


namespace cubic_line_bounded_area_l1263_126325

/-- The area bounded by a cubic function and a line -/
noncomputable def boundedArea (a b c d p q α β γ : ℝ) : ℝ :=
  |a| / 12 * (γ - α)^3 * |2*β - γ - α|

/-- Theorem stating the area bounded by a cubic function and a line -/
theorem cubic_line_bounded_area
  (a b c d p q α β γ : ℝ)
  (h_a : a ≠ 0)
  (h_order : α < β ∧ β < γ)
  (h_cubic : ∀ x, a*x^3 + b*x^2 + c*x + d = p*x + q → x = α ∨ x = β ∨ x = γ) :
  ∃ A, A = boundedArea a b c d p q α β γ ∧
    A = |∫ (x : ℝ) in α..γ, (a*x^3 + b*x^2 + c*x + d) - (p*x + q)| :=
by
  sorry

end cubic_line_bounded_area_l1263_126325


namespace sector_radius_l1263_126351

/-- Given a sector with a central angle of 90° and an arc length of 3π, its radius is 6. -/
theorem sector_radius (θ : Real) (l : Real) (r : Real) : 
  θ = 90 → l = 3 * Real.pi → l = (θ * Real.pi * r) / 180 → r = 6 := by
  sorry

end sector_radius_l1263_126351


namespace gold_alloy_composition_l1263_126366

theorem gold_alloy_composition
  (initial_weight : ℝ)
  (initial_gold_percentage : ℝ)
  (target_gold_percentage : ℝ)
  (added_gold : ℝ)
  (h1 : initial_weight = 48)
  (h2 : initial_gold_percentage = 0.25)
  (h3 : target_gold_percentage = 0.40)
  (h4 : added_gold = 12) :
  let initial_gold := initial_weight * initial_gold_percentage
  let final_weight := initial_weight + added_gold
  let final_gold := initial_gold + added_gold
  (final_gold / final_weight) = target_gold_percentage :=
by
  sorry

#check gold_alloy_composition

end gold_alloy_composition_l1263_126366


namespace stratified_sampling_sample_size_l1263_126336

theorem stratified_sampling_sample_size 
  (total_employees : ℕ) 
  (young_workers : ℕ) 
  (sample_young : ℕ) 
  (h1 : total_employees = 750) 
  (h2 : young_workers = 350) 
  (h3 : sample_young = 7) : 
  ∃ (sample_size : ℕ), 
    sample_size * young_workers = sample_young * total_employees ∧ 
    sample_size = 15 := by
  sorry

end stratified_sampling_sample_size_l1263_126336
