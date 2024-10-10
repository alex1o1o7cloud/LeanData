import Mathlib

namespace university_box_cost_l3312_331259

theorem university_box_cost (box_length box_width box_height : ℝ)
  (box_cost : ℝ) (total_volume : ℝ) :
  box_length = 20 ∧ box_width = 20 ∧ box_height = 12 ∧
  box_cost = 0.5 ∧ total_volume = 2160000 →
  (⌈total_volume / (box_length * box_width * box_height)⌉ : ℝ) * box_cost = 225 := by
  sorry

end university_box_cost_l3312_331259


namespace team_a_score_l3312_331279

theorem team_a_score (total_points team_b_points team_c_points : ℕ) 
  (h1 : team_b_points = 9)
  (h2 : team_c_points = 4)
  (h3 : total_points = 15)
  : total_points - (team_b_points + team_c_points) = 2 := by
  sorry

end team_a_score_l3312_331279


namespace final_dog_count_l3312_331251

def initial_dogs : ℕ := 80
def adoption_rate : ℚ := 40 / 100
def returned_dogs : ℕ := 5

theorem final_dog_count : 
  initial_dogs - (initial_dogs * adoption_rate).floor + returned_dogs = 53 := by
  sorry

end final_dog_count_l3312_331251


namespace gcd_factorial_8_and_factorial_11_times_9_squared_l3312_331248

theorem gcd_factorial_8_and_factorial_11_times_9_squared :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 11 * 9^2) = Nat.factorial 8 := by
  sorry

end gcd_factorial_8_and_factorial_11_times_9_squared_l3312_331248


namespace solve_refrigerator_problem_l3312_331221

def refrigerator_problem (refrigerator_price : ℝ) : Prop :=
  let mobile_price : ℝ := 8000
  let refrigerator_sold : ℝ := refrigerator_price * 0.96
  let mobile_sold : ℝ := mobile_price * 1.1
  let total_bought : ℝ := refrigerator_price + mobile_price
  let total_sold : ℝ := refrigerator_sold + mobile_sold
  let profit : ℝ := 200
  total_sold = total_bought + profit

theorem solve_refrigerator_problem :
  ∃ (price : ℝ), refrigerator_problem price ∧ price = 15000 := by
  sorry

end solve_refrigerator_problem_l3312_331221


namespace a_range_l3312_331203

-- Define the linear equation
def linear_equation (a x : ℝ) : ℝ := a * x + x + 4

-- Define the condition that the root is within [-2, 1]
def root_in_interval (a : ℝ) : Prop :=
  ∃ x, x ∈ Set.Icc (-2) 1 ∧ linear_equation a x = 0

-- State the theorem
theorem a_range (a : ℝ) : 
  root_in_interval a ↔ a ∈ Set.Ioi 1 ∪ Set.Iio (-5) :=
sorry

end a_range_l3312_331203


namespace loot_box_cost_l3312_331289

/-- Proves that the cost of each loot box is $5 given the specified conditions -/
theorem loot_box_cost (avg_value : ℝ) (total_spent : ℝ) (total_loss : ℝ) :
  avg_value = 3.5 →
  total_spent = 40 →
  total_loss = 12 →
  ∃ (cost : ℝ), cost = 5 ∧ cost * (total_spent - total_loss) / total_spent = avg_value :=
by
  sorry


end loot_box_cost_l3312_331289


namespace complex_equation_system_l3312_331226

theorem complex_equation_system (p q r s t u : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (eq1 : p = (q + r) / (s - 3))
  (eq2 : q = (p + r) / (t - 3))
  (eq3 : r = (p + q) / (u - 3))
  (eq4 : s * t + s * u + t * u = 7)
  (eq5 : s + t + u = 4) :
  s * t * u = 6 := by
sorry

end complex_equation_system_l3312_331226


namespace group_size_calculation_l3312_331202

theorem group_size_calculation (initial_avg : ℝ) (new_person_age : ℝ) (new_avg : ℝ) : 
  initial_avg = 15 → new_person_age = 37 → new_avg = 17 → 
  ∃ n : ℕ, (n : ℝ) * initial_avg + new_person_age = (n + 1) * new_avg ∧ n = 10 :=
by sorry

end group_size_calculation_l3312_331202


namespace sine_equality_implies_equal_coefficients_l3312_331241

theorem sine_equality_implies_equal_coefficients 
  (α β γ δ : ℝ) 
  (h_positive : α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0) 
  (h_equality : ∀ x : ℝ, Real.sin (α * x) + Real.sin (β * x) = Real.sin (γ * x) + Real.sin (δ * x)) : 
  α = γ ∨ α = δ := by
sorry

end sine_equality_implies_equal_coefficients_l3312_331241


namespace implicit_derivative_l3312_331299

noncomputable section

open Real

-- Define the implicit function
def F (x y : ℝ) : ℝ := log (sqrt (x^2 + y^2)) - arctan (y / x)

-- State the theorem
theorem implicit_derivative (x y : ℝ) (h1 : x ≠ 0) (h2 : x ≠ y) :
  let y' := (x + y) / (x - y)
  (∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ → 
    |F (x + h) (y + y' * h) - F x y| ≤ ε * |h|) :=
sorry

end

end implicit_derivative_l3312_331299


namespace garden_perimeter_l3312_331297

/-- A rectangular garden with equal length and breadth of 150 meters has a perimeter of 600 meters. -/
theorem garden_perimeter :
  ∀ (length breadth : ℝ),
  length > 0 →
  breadth > 0 →
  (length = 150 ∧ breadth = 150) →
  4 * length = 600 := by
sorry

end garden_perimeter_l3312_331297


namespace unique_pair_with_single_solution_l3312_331201

theorem unique_pair_with_single_solution :
  ∃! p : ℕ × ℕ, 
    let b := p.1
    let c := p.2
    b > 0 ∧ c > 0 ∧
    (∃! x : ℝ, x^2 + b*x + c = 0) ∧
    (∃! x : ℝ, x^2 + c*x + b = 0) :=
by sorry

end unique_pair_with_single_solution_l3312_331201


namespace number_problem_l3312_331273

theorem number_problem (x : ℝ) : (0.6 * x = 0.3 * 10 + 27) → x = 50 := by
  sorry

end number_problem_l3312_331273


namespace ellipse_slope_at_pi_third_l3312_331218

/-- The slope of the line connecting the origin to a point on an ellipse --/
theorem ellipse_slope_at_pi_third :
  let x (t : Real) := 2 * Real.cos t
  let y (t : Real) := 4 * Real.sin t
  let t₀ : Real := Real.pi / 3
  let x₀ : Real := x t₀
  let y₀ : Real := y t₀
  (y₀ - 0) / (x₀ - 0) = 2 * Real.sqrt 3 := by
  sorry

end ellipse_slope_at_pi_third_l3312_331218


namespace quadratic_composition_no_roots_l3312_331252

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The statement that f(x) = x has no real roots -/
def NoRealRoots (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x ≠ x

theorem quadratic_composition_no_roots (a b c : ℝ) (ha : a ≠ 0) :
  let f := QuadraticFunction a b c
  NoRealRoots f → NoRealRoots (f ∘ f) := by
  sorry

#check quadratic_composition_no_roots

end quadratic_composition_no_roots_l3312_331252


namespace amoeba_population_l3312_331237

/-- The number of amoebas in the puddle after n days -/
def amoebas (n : ℕ) : ℕ :=
  3^n

/-- The number of days the amoeba population grows -/
def days : ℕ := 10

theorem amoeba_population : amoebas days = 59049 := by
  sorry

end amoeba_population_l3312_331237


namespace poverty_alleviation_rate_l3312_331213

theorem poverty_alleviation_rate (initial_population final_population : ℕ) 
  (years : ℕ) (decrease_rate : ℝ) : 
  initial_population = 90000 →
  final_population = 10000 →
  years = 2 →
  final_population = initial_population * (1 - decrease_rate) ^ years →
  9 * (1 - decrease_rate) ^ 2 = 1 := by
  sorry

end poverty_alleviation_rate_l3312_331213


namespace quadratic_radical_combination_l3312_331298

theorem quadratic_radical_combination (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 1 ∧ y = Real.sqrt 3) → x = 2 := by
  sorry

end quadratic_radical_combination_l3312_331298


namespace school_trip_photos_l3312_331224

theorem school_trip_photos (claire lisa robert : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 28) :
  claire = 14 := by
sorry

end school_trip_photos_l3312_331224


namespace certain_number_equation_l3312_331295

theorem certain_number_equation : ∃ x : ℝ, 
  (5 * x - (2 * 1.4) / 1.3 = 4) ∧ 
  (abs (x - 1.23076923077) < 0.00000000001) := by
  sorry

end certain_number_equation_l3312_331295


namespace smallest_cookie_packages_l3312_331227

theorem smallest_cookie_packages (cookie_per_package : Nat) (milk_per_package : Nat) 
  (h1 : cookie_per_package = 5) (h2 : milk_per_package = 7) :
  ∃ n : Nat, n > 0 ∧ (cookie_per_package * n) % milk_per_package = 0 ∧
  ∀ m : Nat, m > 0 ∧ (cookie_per_package * m) % milk_per_package = 0 → n ≤ m :=
by
  sorry

end smallest_cookie_packages_l3312_331227


namespace arithmetic_sequence_sum_l3312_331278

theorem arithmetic_sequence_sum : ∀ (a₁ aₙ d n : ℕ),
  a₁ = 1 →
  aₙ = 21 →
  d = 2 →
  n * (a₁ + aₙ) = (aₙ - a₁ + d) * (aₙ - a₁ + d) →
  n * (a₁ + aₙ) / 2 = 121 :=
by
  sorry

#check arithmetic_sequence_sum

end arithmetic_sequence_sum_l3312_331278


namespace square_sides_theorem_l3312_331216

theorem square_sides_theorem (total_length : ℝ) (area_difference : ℝ) 
  (h1 : total_length = 20)
  (h2 : area_difference = 120) :
  ∃ (x y : ℝ), x + y = total_length ∧ x^2 - y^2 = area_difference ∧ x = 13 ∧ y = 7 := by
  sorry

end square_sides_theorem_l3312_331216


namespace rory_tank_water_l3312_331204

/-- Calculates the final amount of water in Rory's tank after a rainstorm --/
def final_water_amount (initial_water : ℝ) (inflow_rate_1 inflow_rate_2 : ℝ) 
  (leak_rate : ℝ) (evap_rate_1 evap_rate_2 : ℝ) (evap_reduction : ℝ) 
  (duration_1 duration_2 : ℝ) : ℝ :=
  let total_inflow := inflow_rate_1 * duration_1 + inflow_rate_2 * duration_2
  let total_leak := leak_rate * (duration_1 + duration_2)
  let total_evap := (evap_rate_1 * duration_1 + evap_rate_2 * duration_2) * (1 - evap_reduction)
  initial_water + total_inflow - total_leak - total_evap

/-- Theorem stating the final amount of water in Rory's tank --/
theorem rory_tank_water : 
  final_water_amount 100 2 3 0.5 0.2 0.1 0.75 45 45 = 276.625 := by
  sorry

end rory_tank_water_l3312_331204


namespace return_probability_after_2012_moves_chessboard_return_probability_l3312_331264

/-- Represents the size of the chessboard -/
def boardSize : ℕ := 8

/-- Represents the total number of moves -/
def totalMoves : ℕ := 2012

/-- Represents the probability of returning to the original position after a given number of moves -/
noncomputable def returnProbability (n : ℕ) : ℚ :=
  ((1 + 2^(n / 2 - 1)) / 2^(n / 2 + 1))^2

/-- Theorem stating the probability of returning to the original position after 2012 moves -/
theorem return_probability_after_2012_moves :
  returnProbability totalMoves = ((1 + 2^1005) / 2^1007)^2 := by
  sorry

/-- Theorem stating that the calculated probability is correct for the given chessboard and moves -/
theorem chessboard_return_probability :
  boardSize = 8 →
  totalMoves = 2012 →
  returnProbability totalMoves = ((1 + 2^1005) / 2^1007)^2 := by
  sorry

end return_probability_after_2012_moves_chessboard_return_probability_l3312_331264


namespace gift_cost_l3312_331290

theorem gift_cost (half_cost : ℝ) (h : half_cost = 14) : 
  2 * half_cost = 28 := by
  sorry

end gift_cost_l3312_331290


namespace unique_integer_between_sqrt5_and_sqrt15_l3312_331277

theorem unique_integer_between_sqrt5_and_sqrt15 : 
  ∃! n : ℤ, (↑n : ℝ) > Real.sqrt 5 ∧ (↑n : ℝ) < Real.sqrt 15 :=
by
  -- The proof goes here
  sorry

end unique_integer_between_sqrt5_and_sqrt15_l3312_331277


namespace min_value_of_f_l3312_331261

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 5) / (2*x - 4)

theorem min_value_of_f (x : ℝ) (h : x ≥ 5/2) : f x ≥ 1 := by
  sorry

end min_value_of_f_l3312_331261


namespace number_of_boys_l3312_331257

theorem number_of_boys (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  girls = (60 : ℕ) * total / 100 →
  girls = 450 →
  boys = total - girls →
  boys = 300 := by
sorry

end number_of_boys_l3312_331257


namespace minimum_value_theorem_l3312_331206

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n

theorem minimum_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (4 * a 1 = a m) →
  (a m)^2 = a 1 * a n →
  (m + n = 6) →
  (1 / m + 4 / n ≥ 3 / 2) ∧
  (∃ m₀ n₀ : ℕ, m₀ + n₀ = 6 ∧ 1 / m₀ + 4 / n₀ = 3 / 2) :=
by sorry

end minimum_value_theorem_l3312_331206


namespace fraction_equality_l3312_331294

theorem fraction_equality (x : ℚ) : (4 + x) / (6 + x) = (2 + x) / (3 + x) ↔ x = 0 := by
  sorry

end fraction_equality_l3312_331294


namespace consecutive_numbers_sum_l3312_331287

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) := by
  sorry

end consecutive_numbers_sum_l3312_331287


namespace seating_arrangements_l3312_331274

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def totalArrangements : ℕ := factorial 10

def restrictedArrangements : ℕ := factorial 7 * factorial 4

theorem seating_arrangements :
  totalArrangements - restrictedArrangements = 3507840 := by sorry

end seating_arrangements_l3312_331274


namespace correct_quadratic_equation_l3312_331207

theorem correct_quadratic_equation :
  ∃ (r₁ r₂ : ℝ), 
    (r₁ + r₂ = 9) ∧ 
    (r₁ * r₂ = 18) ∧ 
    (∃ (s₁ s₂ : ℝ), s₁ + s₂ = 5 - 1 ∧ s₁ + s₂ = 9) ∧
    (r₁ * r₂ = r₁ * r₂ - 9 * (r₁ + r₂) + 18) := by
  sorry

end correct_quadratic_equation_l3312_331207


namespace unit_square_fits_in_parallelogram_l3312_331229

/-- A parallelogram with heights greater than 1 -/
structure Parallelogram where
  heights : ℝ → ℝ
  height_gt_one : ∀ h, heights h > 1

/-- A unit square -/
structure UnitSquare where
  side_length : ℝ
  is_unit : side_length = 1

/-- A placement of a shape inside a parallelogram -/
structure Placement (P : Parallelogram) (S : Type) where
  is_inside : S → Bool

/-- Theorem: For any parallelogram with heights greater than 1, 
    there exists a placement of a unit square inside it -/
theorem unit_square_fits_in_parallelogram (P : Parallelogram) :
  ∃ (U : UnitSquare) (place : Placement P UnitSquare), place.is_inside U = true := by
  sorry

end unit_square_fits_in_parallelogram_l3312_331229


namespace complementary_angles_imply_right_triangle_l3312_331283

-- Define a triangle
structure Triangle where
  a : ℝ  -- angle a
  b : ℝ  -- angle b
  c : ℝ  -- angle c
  sum_180 : a + b + c = 180  -- sum of angles in a triangle is 180 degrees

-- Define what it means for two angles to be complementary
def complementary (x y : ℝ) : Prop := x + y = 90

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop := t.a = 90 ∨ t.b = 90 ∨ t.c = 90

-- Theorem statement
theorem complementary_angles_imply_right_triangle (t : Triangle) :
  (complementary t.a t.b ∨ complementary t.b t.c ∨ complementary t.a t.c) →
  is_right_triangle t :=
sorry

end complementary_angles_imply_right_triangle_l3312_331283


namespace secondary_spermatocyte_may_have_two_y_l3312_331293

-- Define the different stages of cell division
inductive CellDivisionStage
  | PrimarySpermatocyte
  | SecondarySpermatocyte
  | SpermatogoniumMitosis
  | SpermatogoniumMeiosis

-- Define the possible Y chromosome counts
inductive YChromosomeCount
  | Zero
  | One
  | Two

-- Define a function that returns the possible Y chromosome counts for each stage
def possibleYChromosomeCounts (stage : CellDivisionStage) : Set YChromosomeCount :=
  match stage with
  | CellDivisionStage.PrimarySpermatocyte => {YChromosomeCount.One}
  | CellDivisionStage.SecondarySpermatocyte => {YChromosomeCount.Zero, YChromosomeCount.One, YChromosomeCount.Two}
  | CellDivisionStage.SpermatogoniumMitosis => {YChromosomeCount.One}
  | CellDivisionStage.SpermatogoniumMeiosis => {YChromosomeCount.One}

-- Theorem stating that secondary spermatocytes may contain two Y chromosomes
theorem secondary_spermatocyte_may_have_two_y :
  YChromosomeCount.Two ∈ possibleYChromosomeCounts CellDivisionStage.SecondarySpermatocyte :=
by sorry

end secondary_spermatocyte_may_have_two_y_l3312_331293


namespace min_jam_prob_route_l3312_331281

structure Route where
  segments : List (Char × Char)

def no_jam_prob (r : Route) (probs : List ℚ) : ℚ :=
  probs.prod

def jam_prob (r : Route) (probs : List ℚ) : ℚ :=
  1 - no_jam_prob r probs

theorem min_jam_prob_route (route1 route2 route3 : Route)
  (probs1 probs2 probs3 : List ℚ) :
  route1.segments = [('A', 'C'), ('C', 'D'), ('D', 'B')] →
  route2.segments = [('A', 'C'), ('C', 'F'), ('F', 'B')] →
  route3.segments = [('A', 'E'), ('E', 'F'), ('F', 'B')] →
  probs1 = [9/10, 14/15, 5/6] →
  probs2 = [9/10, 9/10, 15/16] →
  probs3 = [9/10, 9/10, 19/20] →
  jam_prob route1 probs1 < jam_prob route2 probs2 ∧
  jam_prob route1 probs1 < jam_prob route3 probs3 :=
by sorry

end min_jam_prob_route_l3312_331281


namespace fathers_age_ratio_l3312_331247

theorem fathers_age_ratio (father_age ronit_age : ℕ) : 
  (father_age + 8 = (ronit_age + 8) * 5 / 2) →
  (father_age + 16 = (ronit_age + 16) * 2) →
  father_age = ronit_age * 4 := by
sorry

end fathers_age_ratio_l3312_331247


namespace smallest_solution_of_equation_l3312_331272

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 2 * x / (x - 2) + (2 * x^2 - 24) / x - 11
  ∃ (y : ℝ), y = (1 - Real.sqrt 65) / 4 ∧ f y = 0 ∧ ∀ (z : ℝ), f z = 0 → y ≤ z :=
by sorry

end smallest_solution_of_equation_l3312_331272


namespace square_of_difference_three_minus_sqrt_two_l3312_331239

theorem square_of_difference_three_minus_sqrt_two : (3 - Real.sqrt 2)^2 = 11 - 6 * Real.sqrt 2 := by
  sorry

end square_of_difference_three_minus_sqrt_two_l3312_331239


namespace square_of_1307_squared_l3312_331223

theorem square_of_1307_squared : (1307 * 1307)^2 = 2918129502401 := by
  sorry

end square_of_1307_squared_l3312_331223


namespace min_perimeter_triangle_l3312_331209

theorem min_perimeter_triangle (a b c : ℕ) (A B C : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  Real.cos A = 3/5 →
  Real.cos B = 5/13 →
  Real.cos C = -1/3 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  (∀ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 →
    Real.cos A = 3/5 →
    Real.cos B = 5/13 →
    Real.cos C = -1/3 →
    x + y > z ∧ y + z > x ∧ z + x > y →
    a + b + c ≤ x + y + z) →
  a + b + c = 192 :=
sorry

end min_perimeter_triangle_l3312_331209


namespace max_value_on_circle_l3312_331230

theorem max_value_on_circle :
  let circle := {p : ℝ × ℝ | (p.1^2 + p.2^2 + 4*p.1 - 6*p.2 + 4) = 0}
  ∃ (max : ℝ), max = -13 ∧ 
    (∀ p ∈ circle, 3*p.1 - 4*p.2 ≤ max) ∧
    (∃ p ∈ circle, 3*p.1 - 4*p.2 = max) :=
sorry

end max_value_on_circle_l3312_331230


namespace base6_addition_l3312_331263

-- Define a function to convert from base 6 to base 10
def base6ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 6
def base10ToBase6 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem base6_addition : 
  base10ToBase6 (base6ToBase10 25 + base6ToBase10 35) = 104 := by sorry

end base6_addition_l3312_331263


namespace x_minus_y_value_l3312_331265

theorem x_minus_y_value (x y : ℝ) (h : |x + 2| + (y - 3)^2 = 0) : x - y = -5 := by
  sorry

end x_minus_y_value_l3312_331265


namespace expression_simplification_l3312_331217

theorem expression_simplification :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := by
  sorry

end expression_simplification_l3312_331217


namespace books_on_shelf_l3312_331271

theorem books_on_shelf (initial_books : ℕ) (added_books : ℕ) (removed_books : ℕ) 
  (h1 : initial_books = 38) 
  (h2 : added_books = 10) 
  (h3 : removed_books = 5) : 
  initial_books + added_books - removed_books = 43 := by
  sorry

end books_on_shelf_l3312_331271


namespace income_mean_difference_l3312_331270

def num_families : ℕ := 1500

def correct_largest_income_1 : ℝ := 150000
def correct_largest_income_2 : ℝ := 148000
def incorrect_largest_income : ℝ := 1500000

def sum_other_incomes : ℝ := sorry  -- This represents the sum S in the solution

theorem income_mean_difference :
  let actual_mean := (sum_other_incomes + correct_largest_income_1 + correct_largest_income_2) / num_families
  let incorrect_mean := (sum_other_incomes + 2 * incorrect_largest_income) / num_families
  incorrect_mean - actual_mean = 1801.33 := by sorry

end income_mean_difference_l3312_331270


namespace unique_pair_existence_l3312_331205

theorem unique_pair_existence (n : ℕ+) :
  ∃! (k l : ℕ), 0 ≤ l ∧ l < k ∧ n = (k * (k - 1)) / 2 + l := by
  sorry

end unique_pair_existence_l3312_331205


namespace system_equations_properties_l3312_331208

theorem system_equations_properties (a b : ℝ) 
  (eq1 : 2 * a + b = 7) 
  (eq2 : a - b = 2) : 
  (b = 7 - 2 * a) ∧ 
  (a = b + 2) ∧ 
  (3 * a = 9) ∧ 
  (3 * b = 3) := by
  sorry

end system_equations_properties_l3312_331208


namespace current_speed_l3312_331276

/-- Given a boat's downstream and upstream speeds, calculate the speed of the current. -/
theorem current_speed (downstream_time upstream_time : ℚ) : 
  downstream_time = 6 / 60 → 
  upstream_time = 10 / 60 → 
  (1 / downstream_time - 1 / upstream_time) / 2 = 2 := by
  sorry

#check current_speed

end current_speed_l3312_331276


namespace factor_expression_l3312_331282

theorem factor_expression (y : ℝ) : 16 * y^3 + 8 * y^2 = 8 * y^2 * (2 * y + 1) := by
  sorry

end factor_expression_l3312_331282


namespace beth_crayons_l3312_331267

theorem beth_crayons (initial_packs : ℚ) : 
  (initial_packs / 10 + 6 = 6.4) → initial_packs = 4 := by
  sorry

end beth_crayons_l3312_331267


namespace inscribed_dodecagon_radius_inscribed_dodecagon_radius_proof_l3312_331246

/-- The radius of a circle circumscribing a convex dodecagon with alternating side lengths of √2 and √24 is √38. -/
theorem inscribed_dodecagon_radius : ℝ → ℝ → ℝ → Prop :=
  fun (r : ℝ) (side1 : ℝ) (side2 : ℝ) =>
    side1 = Real.sqrt 2 ∧
    side2 = Real.sqrt 24 ∧
    r = Real.sqrt 38

/-- Proof of the theorem -/
theorem inscribed_dodecagon_radius_proof :
  ∃ (r : ℝ), inscribed_dodecagon_radius r (Real.sqrt 2) (Real.sqrt 24) :=
by
  sorry

#check inscribed_dodecagon_radius
#check inscribed_dodecagon_radius_proof

end inscribed_dodecagon_radius_inscribed_dodecagon_radius_proof_l3312_331246


namespace monkey_banana_theorem_l3312_331235

/-- Represents the monkey's banana transportation problem -/
structure BananaProblem where
  total_bananas : ℕ
  distance : ℕ
  max_carry : ℕ
  eat_rate : ℕ

/-- Calculates the maximum number of bananas the monkey can bring home -/
def max_bananas_home (problem : BananaProblem) : ℕ :=
  sorry

/-- Theorem stating that for the given problem, the maximum number of bananas brought home is 25 -/
theorem monkey_banana_theorem (problem : BananaProblem) 
  (h1 : problem.total_bananas = 100)
  (h2 : problem.distance = 50)
  (h3 : problem.max_carry = 50)
  (h4 : problem.eat_rate = 1) :
  max_bananas_home problem = 25 := by
  sorry

end monkey_banana_theorem_l3312_331235


namespace complex_to_exponential_l3312_331212

theorem complex_to_exponential (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → z = 2 * Complex.exp (Complex.I * (Real.pi / 3)) := by
  sorry

end complex_to_exponential_l3312_331212


namespace quadratic_root_implies_coefficients_l3312_331269

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def is_root (x a b : ℂ) : Prop := x^2 + a*x + b = 0

-- State the theorem
theorem quadratic_root_implies_coefficients :
  ∀ (a b : ℝ), is_root (1 - i) a b → a = -2 ∧ b = 2 := by
  sorry

end quadratic_root_implies_coefficients_l3312_331269


namespace opposite_of_two_l3312_331249

-- Define the concept of opposite
def opposite (x : ℝ) : ℝ := -x

-- Theorem statement
theorem opposite_of_two : opposite 2 = -2 := by sorry

end opposite_of_two_l3312_331249


namespace rectangle_area_proof_l3312_331291

/-- Rectangle with known side length and area -/
structure Rectangle1 where
  side : ℝ
  area : ℝ

/-- Rectangle similar to Rectangle1 with known diagonal -/
structure Rectangle2 where
  diagonal : ℝ

/-- The area of Rectangle2 given the properties of Rectangle1 and Rectangle2 -/
def area_rectangle2 (r1 : Rectangle1) (r2 : Rectangle2) : ℝ :=
  160

theorem rectangle_area_proof (r1 : Rectangle1) (r2 : Rectangle2) 
  (h1 : r1.side = 4)
  (h2 : r1.area = 32)
  (h3 : r2.diagonal = 20) :
  area_rectangle2 r1 r2 = 160 := by
  sorry

end rectangle_area_proof_l3312_331291


namespace power_tower_mod_1000_l3312_331244

theorem power_tower_mod_1000 : 5^(5^(5^5)) % 1000 = 125 := by sorry

end power_tower_mod_1000_l3312_331244


namespace least_integer_greater_than_negative_eighteen_fifths_l3312_331219

theorem least_integer_greater_than_negative_eighteen_fifths :
  ∃ n : ℤ, n > -18/5 ∧ ∀ m : ℤ, m > -18/5 → m ≥ n :=
by sorry

end least_integer_greater_than_negative_eighteen_fifths_l3312_331219


namespace intersection_of_A_and_B_l3312_331275

def A : Set ℝ := {x | (x + 2) * (x - 5) < 0}
def B : Set ℝ := {x | -3 < x ∧ x < 4}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 < x ∧ x < 4} := by sorry

end intersection_of_A_and_B_l3312_331275


namespace f_2_eq_137_60_l3312_331238

def f (n : ℕ+) : ℚ :=
  Finset.sum (Finset.range (2 * n + 2)) (fun i => 1 / (i + 1 : ℚ))

theorem f_2_eq_137_60 : f 2 = 137 / 60 := by
  sorry

end f_2_eq_137_60_l3312_331238


namespace range_of_a_l3312_331296

-- Define the propositions p and q
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, q x a → p x) ∧ (∃ x, p x ∧ ¬q x a)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary a → a ∈ Set.Ici 1 :=
sorry

end range_of_a_l3312_331296


namespace expression_simplification_l3312_331292

theorem expression_simplification (a : ℝ) (h1 : a ≠ 3) (h2 : a ≠ -3) :
  (1 / (a + 3) + 1 / (a^2 - 9)) / ((a - 2) / (2 * a + 6)) = 2 / (a - 3) := by
  sorry

end expression_simplification_l3312_331292


namespace paul_bought_six_chocolate_boxes_l3312_331222

/-- Represents the number of boxes of chocolate candy Paul bought. -/
def chocolate_boxes : ℕ := sorry

/-- Represents the number of boxes of caramel candy Paul bought. -/
def caramel_boxes : ℕ := 4

/-- Represents the number of pieces of candy in each box. -/
def pieces_per_box : ℕ := 9

/-- Represents the total number of candies Paul had. -/
def total_candies : ℕ := 90

/-- Theorem stating that Paul bought 6 boxes of chocolate candy. -/
theorem paul_bought_six_chocolate_boxes :
  chocolate_boxes = 6 :=
by
  sorry

end paul_bought_six_chocolate_boxes_l3312_331222


namespace paperclips_exceed_500_l3312_331286

def paperclips (n : ℕ) : ℕ := 5 * 4^n

theorem paperclips_exceed_500 : 
  (∃ k, paperclips k > 500) ∧ 
  (∀ j, j < 3 → paperclips j ≤ 500) ∧
  (paperclips 3 > 500) := by
  sorry

end paperclips_exceed_500_l3312_331286


namespace ben_egg_count_l3312_331214

/-- Given that Ben has 7 trays of eggs and each tray contains 10 eggs,
    prove that the total number of eggs Ben examined is 70. -/
theorem ben_egg_count (num_trays : ℕ) (eggs_per_tray : ℕ) :
  num_trays = 7 → eggs_per_tray = 10 → num_trays * eggs_per_tray = 70 := by
  sorry

end ben_egg_count_l3312_331214


namespace investment_satisfies_profit_ratio_q_investment_is_correct_l3312_331245

/-- Represents the investment amounts and profit ratio of two business partners -/
structure BusinessInvestment where
  p_investment : ℝ
  q_investment : ℝ
  p_profit_ratio : ℝ
  q_profit_ratio : ℝ

/-- The business investment scenario described in the problem -/
def problem_investment : BusinessInvestment where
  p_investment := 50000
  q_investment := 66666.67
  p_profit_ratio := 3
  q_profit_ratio := 4

/-- Theorem stating that the given investment amounts satisfy the profit ratio condition -/
theorem investment_satisfies_profit_ratio (bi : BusinessInvestment) :
  bi.p_investment / bi.q_investment = bi.p_profit_ratio / bi.q_profit_ratio →
  bi = problem_investment :=
by
  sorry

/-- Main theorem proving that q's investment is correct given the conditions -/
theorem q_investment_is_correct :
  ∃ (bi : BusinessInvestment),
    bi.p_investment = 50000 ∧
    bi.p_profit_ratio = 3 ∧
    bi.q_profit_ratio = 4 ∧
    bi.p_investment / bi.q_investment = bi.p_profit_ratio / bi.q_profit_ratio ∧
    bi.q_investment = 66666.67 :=
by
  sorry

end investment_satisfies_profit_ratio_q_investment_is_correct_l3312_331245


namespace smallest_a_for_parabola_l3312_331243

/-- The smallest possible value of a for a parabola with given conditions -/
theorem smallest_a_for_parabola (a b c : ℝ) : 
  a > 0 → 
  (∃ n : ℤ, a + b + c = n) →
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ y + 2 = a * (x - 3/4)^2) →
  (∀ a' : ℝ, a' > 0 ∧ 
    (∃ b' c' : ℝ, (∃ n : ℤ, a' + b' + c' = n) ∧
    (∀ x y : ℝ, y = a' * x^2 + b' * x + c' ↔ y + 2 = a' * (x - 3/4)^2)) →
    a ≤ a') →
  a = 16 :=
by sorry

end smallest_a_for_parabola_l3312_331243


namespace custard_pie_problem_l3312_331240

theorem custard_pie_problem (price_per_slice : ℚ) (slices_per_pie : ℕ) (total_revenue : ℚ) :
  price_per_slice = 3 →
  slices_per_pie = 10 →
  total_revenue = 180 →
  (total_revenue / (price_per_slice * slices_per_pie : ℚ)) = 6 := by
  sorry

end custard_pie_problem_l3312_331240


namespace calculation_proof_l3312_331228

theorem calculation_proof :
  (4.8 * (3.5 - 2.1) / 7 = 0.96) ∧
  (18.75 - 0.23 * 2 - 4.54 = 13.75) ∧
  (0.9 + 99 * 0.9 = 90) ∧
  (4 / 0.8 - 0.8 / 4 = 4.8) := by
sorry

end calculation_proof_l3312_331228


namespace simplest_quadratic_radical_l3312_331220

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (a : ℚ), x = Real.sqrt a ∧ ∀ (b : ℚ), b ≠ a → Real.sqrt b ≠ x

theorem simplest_quadratic_radical :
  let options : List ℝ := [1 / Real.sqrt 3, Real.sqrt (5 / 6), Real.sqrt 24, Real.sqrt 21]
  ∀ y ∈ options, is_simplest_quadratic_radical (Real.sqrt 21) ∧ 
    (is_simplest_quadratic_radical y → y = Real.sqrt 21) :=
by sorry

end simplest_quadratic_radical_l3312_331220


namespace average_age_decrease_l3312_331231

theorem average_age_decrease (initial_size : ℕ) (replaced_age new_age : ℕ) : 
  initial_size = 10 → replaced_age = 42 → new_age = 12 → 
  (replaced_age - new_age) / initial_size = 3 := by
  sorry

end average_age_decrease_l3312_331231


namespace power_of_power_three_cubed_squared_l3312_331250

theorem power_of_power_three_cubed_squared : (3^3)^2 = 729 := by
  sorry

end power_of_power_three_cubed_squared_l3312_331250


namespace five_people_arrangement_with_restriction_l3312_331280

/-- The number of ways to arrange n people in a line -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line where one specific person cannot be first or last -/
def arrangementsWithRestriction (n : ℕ) : ℕ :=
  (n - 2) * Nat.factorial (n - 1)

/-- Theorem: There are 72 ways to arrange 5 people in a line where one specific person cannot be first or last -/
theorem five_people_arrangement_with_restriction :
  arrangementsWithRestriction 5 = 72 := by
  sorry

end five_people_arrangement_with_restriction_l3312_331280


namespace jessicas_class_farm_trip_cost_l3312_331266

/-- Calculate the total cost for a field trip to a farm -/
def farm_trip_cost (num_students : ℕ) (num_adults : ℕ) (student_fee : ℕ) (adult_fee : ℕ) : ℕ :=
  num_students * student_fee + num_adults * adult_fee

/-- Theorem: The total cost for Jessica's class field trip to the farm is $199 -/
theorem jessicas_class_farm_trip_cost : farm_trip_cost 35 4 5 6 = 199 := by
  sorry

end jessicas_class_farm_trip_cost_l3312_331266


namespace gcd_lcm_sum_l3312_331254

theorem gcd_lcm_sum : Nat.gcd 40 72 + Nat.lcm 48 18 = 152 := by
  sorry

end gcd_lcm_sum_l3312_331254


namespace consecutive_pages_sum_l3312_331258

theorem consecutive_pages_sum (n : ℕ) : 
  n * (n + 1) * (n + 2) = 136080 → n + (n + 1) + (n + 2) = 144 := by
sorry

end consecutive_pages_sum_l3312_331258


namespace modulo_graph_intercepts_l3312_331285

theorem modulo_graph_intercepts (x₀ y₀ : ℕ) : 
  x₀ < 29 → y₀ < 29 → 
  (5 * x₀ ≡ 3 [ZMOD 29]) → 
  (2 * y₀ ≡ 26 [ZMOD 29]) → 
  x₀ + y₀ = 31 := by sorry

end modulo_graph_intercepts_l3312_331285


namespace inequality_proof_l3312_331242

theorem inequality_proof (a b c m n p : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 1) 
  (h2 : m^2 + n^2 + p^2 = 1) : 
  (|a*m + b*n + c*p| ≤ 1) ∧ 
  (a*b*c ≠ 0 → m^4/a^2 + n^4/b^2 + p^4/c^2 ≥ 1) := by
sorry

end inequality_proof_l3312_331242


namespace line_parameterization_l3312_331233

/-- Given a line y = 2x - 3 parameterized as (x, y) = (-8, s) + t(l, -7),
    prove that s = -19 and l = -7/2 -/
theorem line_parameterization (s l : ℝ) : 
  (∀ (x y t : ℝ), y = 2*x - 3 ↔ (x, y) = (-8, s) + t • (l, -7)) →
  s = -19 ∧ l = -7/2 := by
sorry

end line_parameterization_l3312_331233


namespace angle_terminal_side_set_l3312_331232

/-- 
Given an angle α whose terminal side, when rotated counterclockwise by 30°, 
coincides with the terminal side of 120°, the set of all angles β that have 
the same terminal side as α is {β | β = k × 360° + 90°, k ∈ ℤ}.
-/
theorem angle_terminal_side_set (α : Real) 
  (h : α + 30 = 120 + 360 * (⌊(α + 30 - 120) / 360⌋ : ℤ)) :
  {β : Real | ∃ k : ℤ, β = k * 360 + 90} = 
  {β : Real | ∃ k : ℤ, β = k * 360 + α} :=
by sorry


end angle_terminal_side_set_l3312_331232


namespace bruce_savings_l3312_331288

-- Define the given amounts and rates
def aunt_money : ℝ := 87.32
def grandfather_money : ℝ := 152.68
def savings_rate : ℝ := 0.35
def interest_rate : ℝ := 0.025

-- Define the function to calculate the amount after one year
def amount_after_one_year (aunt_money grandfather_money savings_rate interest_rate : ℝ) : ℝ :=
  let total_money := aunt_money + grandfather_money
  let saved_amount := total_money * savings_rate
  let interest := saved_amount * interest_rate
  saved_amount + interest

-- Theorem statement
theorem bruce_savings : 
  amount_after_one_year aunt_money grandfather_money savings_rate interest_rate = 86.10 := by
  sorry

end bruce_savings_l3312_331288


namespace pentagon_area_sum_l3312_331210

/-- Represents a pentagon with given side lengths and angle -/
structure Pentagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EA : ℝ
  angleCDE : ℝ
  ABparallelDE : Prop

/-- Represents the area of a pentagon in the form √a + b·√c -/
structure PentagonArea where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Function to calculate the area of a pentagon -/
noncomputable def calculatePentagonArea (p : Pentagon) : ℝ := sorry

/-- Function to express the pentagon area in the form √a + b·√c -/
noncomputable def expressAreaAsSum (area : ℝ) : PentagonArea := sorry

theorem pentagon_area_sum (p : Pentagon) 
  (h1 : p.AB = 8)
  (h2 : p.BC = 4)
  (h3 : p.CD = 10)
  (h4 : p.DE = 7)
  (h5 : p.EA = 10)
  (h6 : p.angleCDE = π / 3)  -- 60° in radians
  (h7 : p.ABparallelDE) :
  let area := calculatePentagonArea p
  let expression := expressAreaAsSum area
  expression.a + expression.b + expression.c = 39 := by sorry

end pentagon_area_sum_l3312_331210


namespace smallest_prime_is_prime_q_value_l3312_331253

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def smallest_prime : ℕ := 2

theorem smallest_prime_is_prime : is_prime smallest_prime := by sorry

theorem q_value (p q : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (h_relation : q = 13 * p + 1) 
  (h_smallest : p = smallest_prime) : 
  q = 29 := by sorry

end smallest_prime_is_prime_q_value_l3312_331253


namespace max_intersections_circle_square_l3312_331200

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A square in a plane -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- The number of intersection points between a circle and a square -/
def intersection_points (c : Circle) (s : Square) : ℕ :=
  sorry

/-- The maximum number of intersection points between any circle and any square -/
def max_intersection_points : ℕ := sorry

/-- Theorem: The maximum number of intersection points between a circle and a square is 8 -/
theorem max_intersections_circle_square : max_intersection_points = 8 := by
  sorry

end max_intersections_circle_square_l3312_331200


namespace solve_linear_equation_l3312_331234

theorem solve_linear_equation (x : ℝ) : 2*x + 3*x + 4*x = 12 + 9 + 6 → x = 3 := by
  sorry

end solve_linear_equation_l3312_331234


namespace inscribed_circle_diameter_l3312_331215

/-- The diameter of the inscribed circle in a triangle with sides 11, 6, and 7 is √10 -/
theorem inscribed_circle_diameter (a b c : ℝ) (h1 : a = 11) (h2 : b = 6) (h3 : c = 7) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  2 * area / s = Real.sqrt 10 := by sorry

end inscribed_circle_diameter_l3312_331215


namespace expensive_candy_price_l3312_331260

/-- Proves that the price of the more expensive candy is $3 per pound -/
theorem expensive_candy_price
  (total_mixture : ℝ)
  (selling_price : ℝ)
  (expensive_amount : ℝ)
  (cheap_price : ℝ)
  (h1 : total_mixture = 80)
  (h2 : selling_price = 2.20)
  (h3 : expensive_amount = 16)
  (h4 : cheap_price = 2)
  : ∃ (expensive_price : ℝ),
    expensive_price * expensive_amount + cheap_price * (total_mixture - expensive_amount) =
    selling_price * total_mixture ∧ expensive_price = 3 := by
  sorry

end expensive_candy_price_l3312_331260


namespace solve_equation_l3312_331268

theorem solve_equation (x : ℝ) : 3 * x = (26 - x) + 14 → x = 10 := by
  sorry

end solve_equation_l3312_331268


namespace existence_of_monochromatic_triangle_l3312_331256

/-- A point in the six-pointed star --/
structure Point :=
  (index : Fin 13)

/-- The color of a point --/
inductive Color
| Red
| Green

/-- A coloring of the points in the star --/
def Coloring := Point → Color

/-- Predicate to check if three points form an equilateral triangle --/
def IsEquilateralTriangle (p q r : Point) : Prop := sorry

/-- The main theorem --/
theorem existence_of_monochromatic_triangle (coloring : Coloring) :
  ∃ (p q r : Point), coloring p = coloring q ∧ coloring q = coloring r ∧ IsEquilateralTriangle p q r :=
sorry

end existence_of_monochromatic_triangle_l3312_331256


namespace train_length_train_length_is_120_l3312_331255

/-- The length of a train given specific conditions -/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  relative_speed * passing_time - initial_distance

/-- Proof that the train length is 120 meters under given conditions -/
theorem train_length_is_120 :
  train_length 9 45 120 24 = 120 := by
  sorry

end train_length_train_length_is_120_l3312_331255


namespace sum_of_parts_l3312_331225

theorem sum_of_parts (x y : ℝ) : x + y = 24 → y = 13 → y > x → 7 * x + 5 * y = 142 := by
  sorry

end sum_of_parts_l3312_331225


namespace height_of_specific_prism_l3312_331262

/-- A right triangular prism with base PQR -/
structure RightTriangularPrism where
  /-- Length of side PQ of the base triangle -/
  pq : ℝ
  /-- Length of side PR of the base triangle -/
  pr : ℝ
  /-- Volume of the prism -/
  volume : ℝ

/-- Theorem: The height of a specific right triangular prism is 10 -/
theorem height_of_specific_prism (prism : RightTriangularPrism)
  (h_pq : prism.pq = Real.sqrt 5)
  (h_pr : prism.pr = Real.sqrt 5)
  (h_vol : prism.volume = 25.000000000000004) :
  (2 * prism.volume) / (prism.pq * prism.pr) = 10 := by
  sorry


end height_of_specific_prism_l3312_331262


namespace oneRedOneWhite_bothWhite_mutually_exclusive_not_contradictory_l3312_331284

/-- Represents the color of a ball -/
inductive BallColor
| White
| Red

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The bag containing balls -/
def bag : Multiset BallColor := 2 • {BallColor.White} + 3 • {BallColor.Red}

/-- Event: One red ball and one white ball -/
def oneRedOneWhite (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.Red ∧ outcome.second = BallColor.White) ∨
  (outcome.first = BallColor.White ∧ outcome.second = BallColor.Red)

/-- Event: Both balls are white -/
def bothWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∧ outcome.second = BallColor.White

/-- Two events are mutually exclusive -/
def mutuallyExclusive (e1 e2 : DrawOutcome → Prop) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

/-- Two events are contradictory -/
def contradictory (e1 e2 : DrawOutcome → Prop) : Prop :=
  mutuallyExclusive e1 e2 ∧ ∀ outcome, e1 outcome ∨ e2 outcome

theorem oneRedOneWhite_bothWhite_mutually_exclusive_not_contradictory :
  mutuallyExclusive oneRedOneWhite bothWhite ∧
  ¬contradictory oneRedOneWhite bothWhite :=
sorry

end oneRedOneWhite_bothWhite_mutually_exclusive_not_contradictory_l3312_331284


namespace quadrilateral_area_l3312_331236

theorem quadrilateral_area (d h₁ h₂ : ℝ) (hd : d = 26) (hh₁ : h₁ = 9) (hh₂ : h₂ = 6) :
  (1/2) * d * (h₁ + h₂) = 195 :=
sorry

end quadrilateral_area_l3312_331236


namespace escalator_travel_time_l3312_331211

/-- Proves that a person walking on a moving escalator takes 10 seconds to cover the entire length -/
theorem escalator_travel_time
  (escalator_speed : ℝ)
  (escalator_length : ℝ)
  (person_speed : ℝ)
  (h1 : escalator_speed = 11)
  (h2 : escalator_length = 140)
  (h3 : person_speed = 3) :
  escalator_length / (escalator_speed + person_speed) = 10 := by
  sorry


end escalator_travel_time_l3312_331211
