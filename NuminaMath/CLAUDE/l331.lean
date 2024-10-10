import Mathlib

namespace square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three_l331_33166

theorem square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three :
  ∀ x : ℝ, x = Real.sqrt 2 + 1 → x^2 - 2*x + 2 = 3 := by
  sorry

end square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three_l331_33166


namespace max_identical_bathrooms_l331_33146

theorem max_identical_bathrooms (toilet_paper soap towels shower_gel shampoo toothpaste : ℕ) 
  (h1 : toilet_paper = 45)
  (h2 : soap = 30)
  (h3 : towels = 36)
  (h4 : shower_gel = 18)
  (h5 : shampoo = 27)
  (h6 : toothpaste = 24) :
  ∃ (max_bathrooms : ℕ), 
    max_bathrooms = 3 ∧ 
    (toilet_paper % max_bathrooms = 0) ∧
    (soap % max_bathrooms = 0) ∧
    (towels % max_bathrooms = 0) ∧
    (shower_gel % max_bathrooms = 0) ∧
    (shampoo % max_bathrooms = 0) ∧
    (toothpaste % max_bathrooms = 0) ∧
    ∀ (n : ℕ), n > max_bathrooms → 
      (toilet_paper % n ≠ 0) ∨
      (soap % n ≠ 0) ∨
      (towels % n ≠ 0) ∨
      (shower_gel % n ≠ 0) ∨
      (shampoo % n ≠ 0) ∨
      (toothpaste % n ≠ 0) :=
by
  sorry

end max_identical_bathrooms_l331_33146


namespace chase_travel_time_l331_33145

/-- Represents the journey from Granville to Salisbury with intermediate stops -/
structure Journey where
  chase_speed : ℝ
  cameron_speed : ℝ
  danielle_speed : ℝ
  chase_scooter_speed : ℝ
  cameron_bike_speed : ℝ
  danielle_time : ℝ

/-- The conditions of the journey -/
def journey_conditions (j : Journey) : Prop :=
  j.cameron_speed = 2 * j.chase_speed ∧
  j.danielle_speed = 3 * j.cameron_speed ∧
  j.cameron_bike_speed = 0.75 * j.cameron_speed ∧
  j.chase_scooter_speed = 1.25 * j.chase_speed ∧
  j.danielle_time = 30

/-- The theorem stating that Chase's travel time is 180 minutes -/
theorem chase_travel_time (j : Journey) 
  (h : journey_conditions j) : 
  (180 : ℝ) * j.chase_speed = j.danielle_speed * j.danielle_time :=
sorry

end chase_travel_time_l331_33145


namespace intersection_k_range_l331_33119

-- Define the line equation
def line_eq (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 = 6

-- Define the condition for intersection at two distinct points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧
  hyperbola_eq x₁ (line_eq k x₁) ∧
  hyperbola_eq x₂ (line_eq k x₂)

-- State the theorem
theorem intersection_k_range :
  ∀ k : ℝ, intersects_at_two_points k ↔ 1 < k ∧ k < Real.sqrt 15 / 3 :=
sorry

end intersection_k_range_l331_33119


namespace estimate_theorem_l331_33113

/-- Represents a company with employees and their distance from workplace -/
structure Company where
  total_employees : ℕ
  sample_size : ℕ
  within_1000m : ℕ
  within_2000m : ℕ

/-- Calculates the estimated number of employees living between 1000 and 2000 meters -/
def estimate_between_1000_2000 (c : Company) : ℕ :=
  let sample_between := c.within_2000m - c.within_1000m
  (sample_between * c.total_employees) / c.sample_size

/-- Theorem stating the estimated number of employees living between 1000 and 2000 meters -/
theorem estimate_theorem (c : Company) 
  (h1 : c.total_employees = 2000)
  (h2 : c.sample_size = 200)
  (h3 : c.within_1000m = 10)
  (h4 : c.within_2000m = 30) :
  estimate_between_1000_2000 c = 200 := by
  sorry

#eval estimate_between_1000_2000 { total_employees := 2000, sample_size := 200, within_1000m := 10, within_2000m := 30 }

end estimate_theorem_l331_33113


namespace xy_equals_five_l331_33124

theorem xy_equals_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdistinct : x ≠ y)
  (h : x + 5 / x = y + 5 / y) : x * y = 5 := by
  sorry

end xy_equals_five_l331_33124


namespace instrument_players_fraction_l331_33179

theorem instrument_players_fraction 
  (total_people : ℕ) 
  (two_or_more : ℕ) 
  (prob_exactly_one : ℚ) 
  (h1 : total_people = 800) 
  (h2 : two_or_more = 128) 
  (h3 : prob_exactly_one = 1/25) : 
  (↑two_or_more + ↑total_people * prob_exactly_one) / ↑total_people = 1/5 := by
sorry

end instrument_players_fraction_l331_33179


namespace stratified_sampling_equal_allocation_l331_33148

/-- Represents a group of workers -/
structure WorkerGroup where
  total : Nat
  female : Nat

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  groupA : WorkerGroup
  groupB : WorkerGroup
  totalSamples : Nat

/-- Theorem: In a stratified sampling scenario with two equal-sized strata,
    the number of samples drawn from each stratum is equal to half of the total sample size -/
theorem stratified_sampling_equal_allocation 
  (sample : StratifiedSample) 
  (h1 : sample.groupA.total = sample.groupB.total)
  (h2 : sample.totalSamples % 2 = 0) :
  ∃ (n : Nat), n = sample.totalSamples / 2 ∧ 
               n = sample.totalSamples - n :=
sorry

#check stratified_sampling_equal_allocation

end stratified_sampling_equal_allocation_l331_33148


namespace unique_solution_l331_33132

-- Define the two equations
def equation1 (x y : ℝ) : Prop := (x + y - 5) * (2 * x - 3 * y + 5) = 0
def equation2 (x y : ℝ) : Prop := (x - y + 1) * (3 * x + 2 * y - 12) = 0

-- Define a solution as a point satisfying both equations
def is_solution (p : ℝ × ℝ) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

-- Theorem stating that there is exactly one solution
theorem unique_solution : ∃! p : ℝ × ℝ, is_solution p :=
sorry

end unique_solution_l331_33132


namespace gain_percent_calculation_l331_33126

theorem gain_percent_calculation (MP : ℝ) (MP_pos : MP > 0) : 
  let CP := 0.64 * MP
  let SP := 0.82 * MP
  let gain_percent := ((SP - CP) / CP) * 100
  gain_percent = 28.125 := by sorry

end gain_percent_calculation_l331_33126


namespace min_value_theorem_l331_33123

theorem min_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 2) :
  (1/3 : ℝ) * x^3 + y^2 + z ≥ 13/12 := by
sorry

end min_value_theorem_l331_33123


namespace tori_trash_total_l331_33125

/-- The number of pieces of trash Tori picked up in the classrooms -/
def classroom_trash : ℕ := 344

/-- The number of pieces of trash Tori picked up outside the classrooms -/
def outside_trash : ℕ := 1232

/-- The total number of pieces of trash Tori picked up last week -/
def total_trash : ℕ := classroom_trash + outside_trash

/-- Theorem stating that the total number of pieces of trash Tori picked up is 1576 -/
theorem tori_trash_total : total_trash = 1576 := by
  sorry

end tori_trash_total_l331_33125


namespace min_value_theorem_l331_33122

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x + 2*y + 3*z = 12) : 
  9/x + 4/y + 1/z ≥ 49/12 := by
sorry

end min_value_theorem_l331_33122


namespace largest_t_value_l331_33173

theorem largest_t_value (t : ℚ) : 
  (13 * t^2 - 34 * t + 12) / (3 * t - 2) + 5 * t = 6 * t - 1 →
  t ≤ 5/2 :=
by sorry

end largest_t_value_l331_33173


namespace cattle_purchase_cost_l331_33175

theorem cattle_purchase_cost 
  (num_cattle : ℕ) 
  (feeding_cost_ratio : ℝ) 
  (weight_per_cattle : ℝ) 
  (selling_price_per_pound : ℝ) 
  (profit : ℝ) 
  (h1 : num_cattle = 100)
  (h2 : feeding_cost_ratio = 1.2)
  (h3 : weight_per_cattle = 1000)
  (h4 : selling_price_per_pound = 2)
  (h5 : profit = 112000) : 
  ∃ (purchase_cost : ℝ), purchase_cost = 40000 ∧ 
    num_cattle * weight_per_cattle * selling_price_per_pound - 
    (purchase_cost * (1 + (feeding_cost_ratio - 1))) = profit :=
by sorry

end cattle_purchase_cost_l331_33175


namespace circles_cover_quadrilateral_l331_33135

-- Define a convex quadrilateral
def ConvexQuadrilateral (A B C D : Real × Real) : Prop :=
  -- Add conditions for convexity
  sorry

-- Define a circle with diameter as a side of the quadrilateral
def CircleOnSide (A B : Real × Real) : Set (Real × Real) :=
  {P | ∃ (t : Real), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B ∧ 
    (P.1 - A.1)^2 + (P.2 - A.2)^2 ≤ ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 4}

-- Define the union of four circles on the sides of the quadrilateral
def UnionOfCircles (A B C D : Real × Real) : Set (Real × Real) :=
  CircleOnSide A B ∪ CircleOnSide B C ∪ CircleOnSide C D ∪ CircleOnSide D A

-- Define the interior of the quadrilateral
def QuadrilateralInterior (A B C D : Real × Real) : Set (Real × Real) :=
  -- Add definition for the interior of the quadrilateral
  sorry

-- Theorem statement
theorem circles_cover_quadrilateral (A B C D : Real × Real) :
  ConvexQuadrilateral A B C D →
  QuadrilateralInterior A B C D ⊆ UnionOfCircles A B C D :=
sorry

end circles_cover_quadrilateral_l331_33135


namespace viewers_scientific_notation_equality_l331_33109

-- Define the number of viewers
def viewers : ℕ := 16300000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.63 * (10 ^ 10)

-- Theorem to prove the equality
theorem viewers_scientific_notation_equality :
  (viewers : ℝ) = scientific_notation := by sorry

end viewers_scientific_notation_equality_l331_33109


namespace quadratic_roots_and_m_value_l331_33100

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - (m-3)*x - m

-- Theorem statement
theorem quadratic_roots_and_m_value (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0) ∧
  (∀ x₁ x₂ : ℝ, quadratic m x₁ = 0 → quadratic m x₂ = 0 → x₁^2 + x₂^2 - x₁*x₂ = 7 → m = 1 ∨ m = 2) :=
by sorry

end quadratic_roots_and_m_value_l331_33100


namespace min_value_theorem_l331_33103

theorem min_value_theorem (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  (1 / (2 * x)) + (x / (y + 1)) ≥ 5/4 := by
  sorry

end min_value_theorem_l331_33103


namespace largest_integer_with_remainder_l331_33139

theorem largest_integer_with_remainder : ∃ n : ℕ, 
  n < 200 ∧ 
  n % 7 = 4 ∧ 
  ∀ m : ℕ, m < 200 → m % 7 = 4 → m ≤ n :=
by
  -- The proof would go here
  sorry

end largest_integer_with_remainder_l331_33139


namespace residue_of_neg1237_mod37_l331_33131

theorem residue_of_neg1237_mod37 : ∃ (k : ℤ), -1237 = 37 * k + 21 ∧ (0 ≤ 21 ∧ 21 < 37) := by
  sorry

end residue_of_neg1237_mod37_l331_33131


namespace first_player_wins_l331_33136

/-- Represents the state of a Kayles game -/
structure KaylesGame where
  pins : List Bool
  turn : Nat

/-- Knocks over one pin or two adjacent pins -/
def makeMove (game : KaylesGame) (start : Nat) (count : Nat) : KaylesGame :=
  sorry

/-- Checks if the game is over (no pins left standing) -/
def isGameOver (game : KaylesGame) : Bool :=
  sorry

/-- Represents a strategy for playing Kayles -/
def Strategy := KaylesGame → Nat × Nat

/-- Checks if a strategy is winning for the current player -/
def isWinningStrategy (strat : Strategy) (game : KaylesGame) : Bool :=
  sorry

/-- The main theorem: there exists a winning strategy for the first player -/
theorem first_player_wins :
  ∀ n : Nat, ∃ strat : Strategy, isWinningStrategy strat (KaylesGame.mk (List.replicate n true) 0) :=
  sorry

end first_player_wins_l331_33136


namespace earl_floor_problem_l331_33108

theorem earl_floor_problem (total_floors : ℕ) (initial_floor : ℕ) (first_up : ℕ) (second_up : ℕ) (floors_from_top : ℕ) (floors_down : ℕ) :
  total_floors = 20 →
  initial_floor = 1 →
  first_up = 5 →
  second_up = 7 →
  floors_from_top = 9 →
  initial_floor + first_up - floors_down + second_up = total_floors - floors_from_top →
  floors_down = 2 := by
sorry

end earl_floor_problem_l331_33108


namespace range_of_f_l331_33187

def f (x : ℕ) : ℤ := 2 * x - 3

def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 5}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 1, 3, 5, 7} := by
  sorry

end range_of_f_l331_33187


namespace milk_pouring_l331_33102

theorem milk_pouring (initial_amount : ℚ) (pour_fraction : ℚ) : 
  initial_amount = 3/7 → pour_fraction = 5/8 → pour_fraction * initial_amount = 15/56 := by
  sorry

end milk_pouring_l331_33102


namespace sum_of_coefficients_l331_33199

theorem sum_of_coefficients (d : ℝ) (a b c : ℤ) (h : d ≠ 0) :
  (8 : ℝ) * d + 9 + 10 * d^2 + 4 * d + 3 = (a : ℝ) * d + b + (c : ℝ) * d^2 →
  a + b + c = 34 := by
  sorry

end sum_of_coefficients_l331_33199


namespace channels_taken_away_proof_l331_33176

/-- Calculates the number of channels initially taken away --/
def channels_taken_away (initial_channels : ℕ) 
  (replaced_channels : ℕ) (reduced_channels : ℕ) 
  (sports_package : ℕ) (supreme_sports : ℕ) (final_channels : ℕ) : ℕ :=
  initial_channels + replaced_channels - reduced_channels + sports_package + supreme_sports - final_channels

/-- Proves that 20 channels were initially taken away --/
theorem channels_taken_away_proof : 
  channels_taken_away 150 12 10 8 7 147 = 20 := by sorry

end channels_taken_away_proof_l331_33176


namespace donation_ratio_l331_33107

theorem donation_ratio : 
  ∀ (total parents teachers students : ℝ),
  parents = 0.25 * total →
  teachers + students = 0.75 * total →
  teachers = (2/5) * (teachers + students) →
  students = (3/5) * (teachers + students) →
  parents / students = 5 / 9 := by
sorry

end donation_ratio_l331_33107


namespace dress_savings_l331_33155

/-- Given a dress with an original cost of $180, if someone buys it for 10 dollars less than half the price, they save $100. -/
theorem dress_savings (original_cost : ℕ) (purchase_price : ℕ) : 
  original_cost = 180 → 
  purchase_price = original_cost / 2 - 10 → 
  original_cost - purchase_price = 100 := by
sorry

end dress_savings_l331_33155


namespace task_completion_probability_l331_33151

theorem task_completion_probability 
  (task1_prob : ℝ) 
  (task1_not_task2_prob : ℝ) 
  (h1 : task1_prob = 2/3) 
  (h2 : task1_not_task2_prob = 4/15) 
  (h_independent : task1_not_task2_prob = task1_prob * (1 - task2_prob)) : 
  task2_prob = 3/5 :=
by
  sorry

#check task_completion_probability

end task_completion_probability_l331_33151


namespace five_Y_three_equals_two_l331_33104

-- Define the Y operation
def Y (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2 - x + y

-- Theorem statement
theorem five_Y_three_equals_two : Y 5 3 = 2 := by
  sorry

end five_Y_three_equals_two_l331_33104


namespace ellipse_point_inside_circle_l331_33112

theorem ellipse_point_inside_circle 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (he : c / a = 1 / 2) 
  (hf : c > 0) 
  (x₁ x₂ : ℝ) 
  (hroots : x₁ * x₂ = -c / a ∧ x₁ + x₂ = -b / a) :
  x₁^2 + x₂^2 < 2 := by
sorry

end ellipse_point_inside_circle_l331_33112


namespace books_calculation_initial_books_count_l331_33196

/-- The number of books initially on the shelf -/
def initial_books : ℕ := sorry

/-- The number of books Marta added to the shelf -/
def books_added : ℕ := 10

/-- The final number of books on the shelf -/
def final_books : ℕ := 48

/-- Theorem stating that the initial number of books plus the added books equals the final number of books -/
theorem books_calculation : initial_books + books_added = final_books := by sorry

/-- Theorem proving that the initial number of books is 38 -/
theorem initial_books_count : initial_books = 38 := by sorry

end books_calculation_initial_books_count_l331_33196


namespace functional_equation_solution_l331_33134

theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * Real.cos y) :
  ∀ t : ℝ, f t = f 0 * Real.cos t + f (Real.pi / 2) * Real.sin t :=
by sorry

end functional_equation_solution_l331_33134


namespace max_inscribed_cylinder_volume_l331_33114

/-- 
Given a right circular cone with base radius R and height M, 
prove that the maximum volume of an inscribed right circular cylinder 
is 4πMR²/27, and this volume is 4/9 of the cone's volume.
-/
theorem max_inscribed_cylinder_volume (R M : ℝ) (hR : R > 0) (hM : M > 0) :
  let cone_volume := (1/3) * π * R^2 * M
  let max_cylinder_volume := (4/27) * π * M * R^2
  max_cylinder_volume = (4/9) * cone_volume := by
  sorry


end max_inscribed_cylinder_volume_l331_33114


namespace solve_equations_l331_33164

theorem solve_equations :
  (∃ x : ℝ, 5 * x - 2.9 = 12) ∧
  (∃ x : ℝ, 10.5 * x + 0.6 * x = 44) ∧
  (∃ x : ℝ, 8 * x / 2 = 1.5) :=
by
  constructor
  · use 1.82
    sorry
  constructor
  · use 3
    sorry
  · use 0.375
    sorry

end solve_equations_l331_33164


namespace f_range_of_a_l331_33191

/-- The function f(x) defined as |x-1| + |x-a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

/-- The theorem stating that if f(x) ≥ 2 for all real x, then a is in (-∞, -1] ∪ [3, +∞) -/
theorem f_range_of_a (a : ℝ) : (∀ x : ℝ, f a x ≥ 2) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end f_range_of_a_l331_33191


namespace comprehensive_score_example_l331_33140

/-- Calculates the comprehensive score given regular assessment and final exam scores and their weightings -/
def comprehensive_score (regular_score : ℝ) (final_score : ℝ) (regular_weight : ℝ) (final_weight : ℝ) : ℝ :=
  regular_score * regular_weight + final_score * final_weight

/-- Proves that the comprehensive score is 91 given the specified scores and weightings -/
theorem comprehensive_score_example : 
  comprehensive_score 95 90 0.2 0.8 = 91 := by
  sorry

end comprehensive_score_example_l331_33140


namespace woman_work_days_l331_33190

/-- A woman's work and pay scenario -/
theorem woman_work_days (total_days : ℕ) (pay_per_day : ℕ) (forfeit_per_day : ℕ) (net_earnings : ℕ) 
    (h1 : total_days = 25)
    (h2 : pay_per_day = 20)
    (h3 : forfeit_per_day = 5)
    (h4 : net_earnings = 450) :
  ∃ (work_days : ℕ), 
    work_days ≤ total_days ∧ 
    (pay_per_day * work_days - forfeit_per_day * (total_days - work_days) = net_earnings) ∧
    work_days = 23 := by
  sorry

end woman_work_days_l331_33190


namespace triangle_formation_l331_33111

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def can_form_triangle (sides : List ℝ) : Prop :=
  sides.length = 3 ∧ 
  let a := sides[0]!
  let b := sides[1]!
  let c := sides[2]!
  triangle_inequality a b c

theorem triangle_formation :
  ¬(can_form_triangle [1, 2, 4]) ∧
  ¬(can_form_triangle [2, 3, 6]) ∧
  ¬(can_form_triangle [12, 5, 6]) ∧
  can_form_triangle [8, 6, 4] :=
by sorry

end triangle_formation_l331_33111


namespace work_completion_time_l331_33116

/-- The time it takes for A to finish the remaining work after B has worked for 10 days -/
def remaining_time_for_A (a_time b_time b_work_days : ℚ) : ℚ :=
  (1 - b_work_days / b_time) / (1 / a_time)

theorem work_completion_time :
  remaining_time_for_A 9 15 10 = 3 := by sorry

end work_completion_time_l331_33116


namespace inverse_as_linear_combination_l331_33118

def M : Matrix (Fin 2) (Fin 2) ℚ :=
  !![3, 1; 0, -2]

theorem inverse_as_linear_combination :
  ∃ (a b : ℚ), a = 1/6 ∧ b = -1/6 ∧ M⁻¹ = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end inverse_as_linear_combination_l331_33118


namespace differentiable_implies_continuous_l331_33192

theorem differentiable_implies_continuous (f : ℝ → ℝ) (x₀ : ℝ) :
  DifferentiableAt ℝ f x₀ → ContinuousAt f x₀ := by
  sorry

end differentiable_implies_continuous_l331_33192


namespace no_opposite_divisibility_l331_33120

theorem no_opposite_divisibility (k n a : ℕ) : 
  k ≥ 3 → n ≥ 3 → Odd k → Odd n → a ≥ 1 → 
  k ∣ (2^a + 1) → n ∣ (2^a - 1) → 
  ¬∃ b : ℕ, b ≥ 1 ∧ k ∣ (2^b - 1) ∧ n ∣ (2^b + 1) :=
by sorry

end no_opposite_divisibility_l331_33120


namespace average_fish_caught_l331_33185

def fish_caught (person : String) : ℕ :=
  match person with
  | "Aang" => 7
  | "Sokka" => 5
  | "Toph" => 12
  | _ => 0

def people : List String := ["Aang", "Sokka", "Toph"]

theorem average_fish_caught :
  (people.map fish_caught).sum / people.length = 8 := by
  sorry

end average_fish_caught_l331_33185


namespace original_price_after_discounts_l331_33174

/-- 
Given an article sold at $126 after two successive discounts of 10% and 20%,
prove that its original price was $175.
-/
theorem original_price_after_discounts (final_price : ℝ) 
  (h1 : final_price = 126) 
  (discount1 : ℝ) (h2 : discount1 = 0.1)
  (discount2 : ℝ) (h3 : discount2 = 0.2) : 
  ∃ (original_price : ℝ), 
    original_price = 175 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) := by
  sorry

end original_price_after_discounts_l331_33174


namespace ellipse_chord_properties_l331_33137

/-- Given an ellipse with equation x²/2 + y² = 1, this theorem proves various properties
    related to chords and their midpoints. -/
theorem ellipse_chord_properties :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/2 + y^2 = 1}
  let P := (1/2, 1/2)
  let A := (2, 1)
  ∀ (x y : ℝ), (x, y) ∈ ellipse →
    (∃ (m b : ℝ), 2*x + 4*y - 3 = 0 ∧ 
      ∀ (x' y' : ℝ), (x', y') ∈ ellipse → 
        (y' - P.2 = m*(x' - P.1) + b ↔ y - P.2 = m*(x - P.1) + b)) ∧
    (∃ (x₀ y₀ : ℝ), x₀ + 4*y₀ = 0 ∧ -Real.sqrt 2 < x₀ ∧ x₀ < Real.sqrt 2 ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ ellipse ∧ (x₂, y₂) ∈ ellipse ∧
        (y₂ - y₁)/(x₂ - x₁) = 2 ∧ x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2) ∧
    (∃ (x₀ y₀ : ℝ), x₀^2 - 2*x₀ + 2*y₀^2 - 2*y₀ = 0 ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ ellipse ∧ (x₂, y₂) ∈ ellipse ∧
        (y₁ - A.2)/(x₁ - A.1) = (y₂ - A.2)/(x₂ - A.1) ∧
        x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2) ∧
    (∃ (x₀ y₀ : ℝ), x₀^2 + 2*y₀^2 = 1 ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ ellipse ∧ (x₂, y₂) ∈ ellipse ∧
        (y₁/x₁) * (y₂/x₂) = -1/2 ∧ x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2) :=
by sorry

end ellipse_chord_properties_l331_33137


namespace distance_between_intersection_points_l331_33171

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 9

-- Define the line that intersects the circle
def intersecting_line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ circle_C x y ∧ intersecting_line x y}

-- State the theorem
theorem distance_between_intersection_points :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 2 := by
  sorry

end distance_between_intersection_points_l331_33171


namespace volleyball_contributions_l331_33193

theorem volleyball_contributions :
  ∀ (x y z : ℝ),
  -- Condition 1: Third boy contributed 6.4 rubles more than the first boy
  z = x + 6.4 →
  -- Condition 2: Half of first boy's contribution equals one-third of second boy's
  (1/2) * x = (1/3) * y →
  -- Condition 3: Half of first boy's contribution equals one-fourth of third boy's
  (1/2) * x = (1/4) * z →
  -- Conclusion: The contributions are 6.4, 9.6, and 12.8 rubles
  x = 6.4 ∧ y = 9.6 ∧ z = 12.8 :=
by
  sorry


end volleyball_contributions_l331_33193


namespace trig_expression_equality_l331_33144

theorem trig_expression_equality : 
  (Real.sin (20 * π / 180) * Real.sqrt (1 + Real.cos (40 * π / 180))) / Real.cos (50 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end trig_expression_equality_l331_33144


namespace sqrt_72_plus_sqrt_32_l331_33117

theorem sqrt_72_plus_sqrt_32 : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end sqrt_72_plus_sqrt_32_l331_33117


namespace line_not_in_third_quadrant_l331_33156

-- Define the line
def line (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0

-- Define the third quadrant
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem: The line does not pass through the third quadrant
theorem line_not_in_third_quadrant : 
  ¬ ∃ (x y : ℝ), line x y ∧ third_quadrant x y := by
  sorry

end line_not_in_third_quadrant_l331_33156


namespace ball_in_cylinder_l331_33150

/-- Given a horizontal cylindrical measuring cup with base radius √3 cm and a solid ball
    of radius R cm that is submerged and causes the water level to rise exactly R cm,
    prove that R = 3/2 cm. -/
theorem ball_in_cylinder (R : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * R^3 = Real.pi * 3 * R → R = 3 / 2 := by sorry

end ball_in_cylinder_l331_33150


namespace geometric_sequence_property_l331_33121

/-- Represents a geometric sequence -/
structure GeometricSequence (α : Type*) [Ring α] where
  a : ℕ → α
  r : α
  h : ∀ n, a (n + 1) = r * a n

/-- Sum of the first n terms of a geometric sequence -/
def sum_n {α : Type*} [Ring α] (seq : GeometricSequence α) (n : ℕ) : α :=
  sorry

/-- The main theorem stating that for any geometric sequence, 
    a_{2016}(S_{2016}-S_{2015}) ≠ 0 -/
theorem geometric_sequence_property {α : Type*} [Field α] (seq : GeometricSequence α) :
  seq.a 2016 * (sum_n seq 2016 - sum_n seq 2015) ≠ 0 :=
sorry

end geometric_sequence_property_l331_33121


namespace negation_equivalence_l331_33189

-- Define the universe of switches and lights
variable (Switch Light : Type)

-- Define the state of switches and lights
variable (is_off : Switch → Prop)
variable (is_on : Light → Prop)

-- Define the main switch
variable (main_switch : Switch)

-- Define the conditions
variable (h1 : ∀ s : Switch, is_off s → ∀ l : Light, ¬(is_on l))
variable (h2 : is_off main_switch → ∀ s : Switch, is_off s)

-- The theorem to prove
theorem negation_equivalence :
  ¬(is_off main_switch → ∀ l : Light, ¬(is_on l)) ↔
  (is_off main_switch ∧ ∃ l : Light, is_on l) :=
by sorry

end negation_equivalence_l331_33189


namespace smallest_a_for_polynomial_l331_33138

theorem smallest_a_for_polynomial (a b : ℤ) (r₁ r₂ r₃ : ℕ+) : 
  r₁ * r₂ * r₃ = 1806 →
  r₁ + r₂ + r₃ = a →
  ∀ a' : ℤ, (∃ b' r₁' r₂' r₃' : ℕ+, 
    r₁' * r₂' * r₃' = 1806 ∧ 
    r₁' + r₂' + r₃' = a') → 
  a ≤ a' →
  a = 76 := by
sorry

end smallest_a_for_polynomial_l331_33138


namespace fixed_point_of_f_l331_33184

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 2) + 2

theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = 2 ∧ f a 2 = 3 :=
by sorry

end fixed_point_of_f_l331_33184


namespace min_value_of_f_l331_33133

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

theorem min_value_of_f (a : ℝ) :
  (∃ (h : ℝ), ∀ x, f a x ≥ f a (-2)) →
  (∃ (m : ℝ), ∀ x, f a x ≥ m ∧ ∃ y, f a y = m) →
  (∃ (m : ℝ), ∀ x, f a x ≥ m ∧ ∃ y, f a y = m ∧ m = -1) :=
by sorry

end min_value_of_f_l331_33133


namespace total_blocks_l331_33170

theorem total_blocks (initial_blocks additional_blocks : ℕ) :
  initial_blocks = 86 →
  additional_blocks = 9 →
  initial_blocks + additional_blocks = 95 :=
by sorry

end total_blocks_l331_33170


namespace sin_675_degrees_l331_33152

theorem sin_675_degrees : Real.sin (675 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_675_degrees_l331_33152


namespace solve_equation_and_evaluate_l331_33106

theorem solve_equation_and_evaluate (x : ℚ) : 
  (4 * x - 8 = 12 * x + 4) → (5 * (x - 3) = -45 / 2) := by
  sorry

end solve_equation_and_evaluate_l331_33106


namespace hyperbola_from_ellipse_l331_33181

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define the hyperbola equations
def hyperbola1 (x y : ℝ) : Prop := x^2/16 - y^2/48 = 1
def hyperbola2 (x y : ℝ) : Prop := y^2/9 - x^2/27 = 1

-- Define the eccentricity
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_from_ellipse :
  ∀ x y : ℝ, ellipse x y →
  (∃ a b : ℝ, (hyperbola1 a b ∧ (a = x ∨ a = -x) ∧ (b = y ∨ b = -y)) ∨
              (hyperbola2 a b ∧ (a = x ∨ a = -x) ∧ (b = y ∨ b = -y))) :=
by sorry

end hyperbola_from_ellipse_l331_33181


namespace allans_balloons_prove_allans_balloons_l331_33129

theorem allans_balloons (jake_balloons : ℕ) (difference : ℕ) : ℕ :=
  jake_balloons + difference

theorem prove_allans_balloons :
  allans_balloons 3 2 = 5 := by
  sorry

end allans_balloons_prove_allans_balloons_l331_33129


namespace solve_system_of_equations_l331_33115

theorem solve_system_of_equations (x y : ℤ) 
  (h1 : x + y = 250) 
  (h2 : x - y = 200) : 
  y = 25 := by
sorry

end solve_system_of_equations_l331_33115


namespace seeds_per_pack_l331_33141

def desired_flowers : ℕ := 20
def survival_rate : ℚ := 1/2
def pack_cost : ℕ := 5
def total_spent : ℕ := 10

theorem seeds_per_pack : 
  ∃ (seeds_per_pack : ℕ), 
    (total_spent / pack_cost) * seeds_per_pack = desired_flowers / survival_rate :=
by sorry

end seeds_per_pack_l331_33141


namespace intersection_implies_a_value_l331_33160

theorem intersection_implies_a_value (a : ℝ) : 
  let M : Set ℝ := {1, 2, a^2 - 3*a - 1}
  let N : Set ℝ := {-1, a, 3}
  (M ∩ N = {3}) → a = 4 := by
sorry

end intersection_implies_a_value_l331_33160


namespace quadratic_function_m_range_l331_33165

theorem quadratic_function_m_range (a c m : ℝ) :
  let f := fun x : ℝ => a * x^2 - 2 * a * x + c
  (∀ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, x < y → f x > f y) →
  f m ≤ f 0 →
  m ∈ Set.Icc 0 2 :=
by sorry

end quadratic_function_m_range_l331_33165


namespace line_intersection_canonical_form_l331_33180

/-- Given two planes in 3D space, this theorem proves that their line of intersection
    can be represented by specific canonical equations. -/
theorem line_intersection_canonical_form :
  ∀ (x y z : ℝ),
  (x + y - 2*z - 2 = 0 ∧ x - y + z + 2 = 0) →
  ∃ (t : ℝ), x = -t ∧ y = -3*t + 2 ∧ z = -2*t := by sorry

end line_intersection_canonical_form_l331_33180


namespace solution_value_l331_33168

theorem solution_value (t : ℝ) : 
  (let y := -(t - 1)
   2 * y - 4 = 3 * (y - 2)) → 
  t = -1 := by
  sorry

end solution_value_l331_33168


namespace proportional_function_decreases_l331_33167

/-- Proves that for a proportional function y = kx passing through the point (4, -1),
    where k is a non-zero constant, y decreases as x increases. -/
theorem proportional_function_decreases (k : ℝ) (h1 : k ≠ 0) (h2 : k * 4 = -1) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ > k * x₂ := by
  sorry

end proportional_function_decreases_l331_33167


namespace tourist_attraction_arrangements_l331_33147

def total_attractions : ℕ := 10
def daytime_attractions : ℕ := 8
def nighttime_attractions : ℕ := 2
def selected_attractions : ℕ := 5
def day1_slots : ℕ := 3
def day2_slots : ℕ := 2

theorem tourist_attraction_arrangements :
  (∃ (arrangements_with_A_or_B : ℕ) 
      (arrangements_A_and_B_same_day : ℕ) 
      (arrangements_without_A_and_B_together : ℕ),
    arrangements_with_A_or_B = 2352 ∧
    arrangements_A_and_B_same_day = 28560 ∧
    arrangements_without_A_and_B_together = 2352) := by
  sorry

end tourist_attraction_arrangements_l331_33147


namespace min_value_expression_l331_33142

theorem min_value_expression (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + 3*y = 2) :
  1/x + 3/y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 2 ∧ 1/x₀ + 3/y₀ = 8 :=
by sorry

end min_value_expression_l331_33142


namespace chris_parents_gift_l331_33183

/-- The amount of money Chris had before his birthday -/
def before_birthday : ℕ := 159

/-- The amount Chris received from his grandmother -/
def from_grandmother : ℕ := 25

/-- The amount Chris received from his aunt and uncle -/
def from_aunt_uncle : ℕ := 20

/-- The total amount Chris had after his birthday -/
def total_after_birthday : ℕ := 279

/-- The amount Chris's parents gave him -/
def from_parents : ℕ := total_after_birthday - before_birthday - from_grandmother - from_aunt_uncle

theorem chris_parents_gift : from_parents = 75 := by
  sorry

end chris_parents_gift_l331_33183


namespace sum_of_three_odd_squares_l331_33149

theorem sum_of_three_odd_squares (a b c : ℕ) : 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →  -- pairwise different
  (∃ k l m : ℕ, a = 2*k + 1 ∧ b = 2*l + 1 ∧ c = 2*m + 1) →  -- odd integers
  (∃ x₁ x₂ x₃ x₄ x₅ x₆ : ℕ, a^2 + b^2 + c^2 = x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 + x₆^2) :=
by sorry

end sum_of_three_odd_squares_l331_33149


namespace car_speed_problem_l331_33186

theorem car_speed_problem (V : ℝ) (x : ℝ) : 
  let V1 := V * (1 - x / 100)
  let V2 := V1 * (1 + 0.5 * x / 100)
  V2 = V * (1 - 0.6 * x / 100) →
  x = 20 := by
sorry

end car_speed_problem_l331_33186


namespace gcf_lcm_problem_l331_33159

-- Define GCF (Greatest Common Factor)
def GCF (a b : ℕ) : ℕ := sorry

-- Define LCM (Least Common Multiple)
def LCM (c d : ℕ) : ℕ := sorry

-- Theorem statement
theorem gcf_lcm_problem : GCF (LCM 9 15) (LCM 10 21) = 15 := by sorry

end gcf_lcm_problem_l331_33159


namespace initial_seashell_count_l331_33130

theorem initial_seashell_count (henry paul leo : ℕ) : 
  henry = 11 →
  paul = 24 →
  henry + paul + (3/4 * leo) = 53 →
  henry + paul + leo = 59 :=
by sorry

end initial_seashell_count_l331_33130


namespace intersection_of_M_and_N_l331_33162

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x < 2}
def N : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end intersection_of_M_and_N_l331_33162


namespace probability_two_black_balls_l331_33178

def total_balls : ℕ := 15
def black_balls : ℕ := 10
def white_balls : ℕ := 5

theorem probability_two_black_balls :
  let p_first_black : ℚ := black_balls / total_balls
  let p_second_black : ℚ := (black_balls - 1) / (total_balls - 1)
  p_first_black * p_second_black = 3 / 7 := by sorry

end probability_two_black_balls_l331_33178


namespace overall_percentage_calculation_l331_33169

theorem overall_percentage_calculation (grade1 grade2 grade3 : ℚ) :
  grade1 = 50 / 100 →
  grade2 = 60 / 100 →
  grade3 = 70 / 100 →
  (grade1 + grade2 + grade3) / 3 = 60 / 100 := by
  sorry

end overall_percentage_calculation_l331_33169


namespace increasing_sequences_count_l331_33172

theorem increasing_sequences_count :
  let n := 2013
  let k := 12
  let count := Nat.choose (((n - 1) / 2) + k - 1) k
  (count = Nat.choose 1017 12) ∧
  (1017 % 1000 = 17) := by sorry

end increasing_sequences_count_l331_33172


namespace same_terminal_side_diff_multiple_360_l331_33161

/-- Two angles have the same terminal side -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

/-- Theorem: If two angles have the same terminal side, their difference is a multiple of 360° -/
theorem same_terminal_side_diff_multiple_360 (α β : ℝ) :
  same_terminal_side α β → ∃ k : ℤ, α - β = k * 360 := by
  sorry

end same_terminal_side_diff_multiple_360_l331_33161


namespace rectangular_solid_surface_area_l331_33128

/-- A rectangular solid with prime edge lengths and volume 399 has surface area 422. -/
theorem rectangular_solid_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 399 →
  2 * (a * b + b * c + c * a) = 422 := by
  sorry

end rectangular_solid_surface_area_l331_33128


namespace third_derivative_y_l331_33101

noncomputable def y (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative_y (x : ℝ) :
  (deriv^[3] y) x = 4 / (1 + x^2)^2 := by sorry

end third_derivative_y_l331_33101


namespace new_ratio_is_one_to_two_l331_33143

/-- Represents the ratio of boarders to day students -/
structure Ratio where
  boarders : ℕ
  day_students : ℕ

/-- Represents the school's student composition -/
structure School where
  initial_boarders : ℕ
  initial_ratio : Ratio
  new_boarders : ℕ

/-- Calculates the new ratio of boarders to day students after new boarders join -/
def new_ratio (school : School) : Ratio :=
  sorry

/-- Theorem stating that the new ratio is 1:2 given the initial conditions -/
theorem new_ratio_is_one_to_two (school : School) 
  (h1 : school.initial_boarders = 120)
  (h2 : school.initial_ratio = Ratio.mk 2 5)
  (h3 : school.new_boarders = 30) :
  new_ratio school = Ratio.mk 1 2 :=
sorry

end new_ratio_is_one_to_two_l331_33143


namespace different_smallest_angles_l331_33105

/-- A type representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A type representing a set of 6 points in a plane -/
structure SixPoints :=
  (points : Fin 6 → Point)

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- Predicate to check if no three points in a set of six points are collinear -/
def no_three_collinear (s : SixPoints) : Prop :=
  ∀ (i j k : Fin 6), i ≠ j → j ≠ k → i ≠ k →
    ¬collinear (s.points i) (s.points j) (s.points k)

/-- Function to calculate the angle between three points -/
noncomputable def angle (p q r : Point) : ℝ := sorry

/-- Function to find the smallest angle in a triangle -/
noncomputable def smallest_angle (p q r : Point) : ℝ :=
  min (angle p q r) (min (angle q r p) (angle r p q))

/-- The main theorem -/
theorem different_smallest_angles (s : SixPoints) (h : no_three_collinear s) :
  ∃ (i₁ j₁ k₁ i₂ j₂ k₂ : Fin 6),
    smallest_angle (s.points i₁) (s.points j₁) (s.points k₁) ≠
    smallest_angle (s.points i₂) (s.points j₂) (s.points k₂) :=
  sorry

end different_smallest_angles_l331_33105


namespace viggo_age_ratio_l331_33127

theorem viggo_age_ratio :
  ∀ (viggo_current_age brother_current_age M Y : ℕ),
    viggo_current_age + brother_current_age = 32 →
    brother_current_age = 10 →
    viggo_current_age - brother_current_age = M * 2 + Y - 2 →
    (M * 2 + Y) / 2 = 7 := by
  sorry

end viggo_age_ratio_l331_33127


namespace texas_passengers_on_l331_33110

/-- Represents the number of passengers at different stages of the flight --/
structure PassengerCount where
  initial : ℕ
  texas_off : ℕ
  texas_on : ℕ
  nc_off : ℕ
  nc_on : ℕ
  crew : ℕ
  final : ℕ

/-- Theorem stating that given the flight conditions, 24 passengers got on in Texas --/
theorem texas_passengers_on (p : PassengerCount) 
  (h1 : p.initial = 124)
  (h2 : p.texas_off = 58)
  (h3 : p.nc_off = 47)
  (h4 : p.nc_on = 14)
  (h5 : p.crew = 10)
  (h6 : p.final = 67)
  (h7 : p.final = p.initial - p.texas_off + p.texas_on - p.nc_off + p.nc_on + p.crew) :
  p.texas_on = 24 := by
  sorry

end texas_passengers_on_l331_33110


namespace complex_equation_solution_l331_33157

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I := by
  sorry

end complex_equation_solution_l331_33157


namespace task_completion_time_relation_l331_33182

/-- 
Theorem: Given three individuals A, B, and C working on a task, where:
- A's time = m * (B and C's time together)
- B's time = n * (A and C's time together)
- C's time = k * (A and B's time together)
Then k can be expressed in terms of m and n as: k = (m + n + 2) / (mn - 1)
-/
theorem task_completion_time_relation (m n k : ℝ) (hm : m > 0) (hn : n > 0) (hk : k > 0) :
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    (1 / x = m / (y + z)) ∧
    (1 / y = n / (x + z)) ∧
    (1 / z = k / (x + y))) →
  k = (m + n + 2) / (m * n - 1) :=
by sorry

end task_completion_time_relation_l331_33182


namespace water_saving_calculation_l331_33154

/-- The amount of water Hyunwoo's family uses daily in liters -/
def daily_water_usage : ℝ := 215

/-- The fraction of water saved when adjusting the water pressure valve weakly -/
def water_saving_fraction : ℝ := 0.32

/-- The amount of water saved when adjusting the water pressure valve weakly -/
def water_saved : ℝ := daily_water_usage * water_saving_fraction

theorem water_saving_calculation :
  water_saved = 68.8 := by sorry

end water_saving_calculation_l331_33154


namespace dice_probability_l331_33177

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice being rolled -/
def num_dice : ℕ := 6

/-- The probability of rolling all the same numbers -/
def prob_all_same : ℚ := 1 / (sides ^ (num_dice - 1))

/-- The probability of not rolling all the same numbers -/
def prob_not_all_same : ℚ := 1 - prob_all_same

theorem dice_probability :
  prob_not_all_same = 7775 / 7776 :=
sorry

end dice_probability_l331_33177


namespace haley_spent_32_l331_33198

/-- The amount Haley spent on concert tickets -/
def haley_spent (ticket_price : ℕ) (self_and_friends : ℕ) (extra : ℕ) : ℕ :=
  (self_and_friends + extra) * ticket_price

/-- Proof that Haley spent $32 on concert tickets -/
theorem haley_spent_32 :
  haley_spent 4 3 5 = 32 := by
  sorry

end haley_spent_32_l331_33198


namespace rod_weight_10m_l331_33153

/-- Represents the weight of a rod given its length -/
def rod_weight (length : ℝ) : ℝ := sorry

/-- The constant of proportionality between rod length and weight -/
def weight_per_meter : ℝ := sorry

theorem rod_weight_10m (h1 : rod_weight 6 = 14.04) 
  (h2 : ∀ l : ℝ, rod_weight l = weight_per_meter * l) : 
  rod_weight 10 = 23.4 := by sorry

end rod_weight_10m_l331_33153


namespace speed_conversion_l331_33188

/-- Conversion factor from km/h to m/s -/
def km_h_to_m_s : ℝ := 0.27777777777778

/-- Given speed in km/h -/
def speed_km_h : ℝ := 0.8666666666666666

/-- Calculated speed in m/s -/
def speed_m_s : ℝ := 0.24074074074074

theorem speed_conversion : speed_km_h * km_h_to_m_s = speed_m_s := by sorry

end speed_conversion_l331_33188


namespace perpendicular_dot_product_zero_l331_33163

/-- Given a point P on the curve y = x + 2/x for x > 0, prove that the dot product
    of PA and PB is zero, where A is the foot of the perpendicular from P to y = x,
    and B is the foot of the perpendicular from P to x = 0. -/
theorem perpendicular_dot_product_zero (x : ℝ) (hx : x > 0) :
  let P : ℝ × ℝ := (x, x + 2/x)
  let A : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
  let B : ℝ × ℝ := (0, x + 2/x)
  let PA : ℝ × ℝ := (A.1 - P.1, A.2 - P.2)
  let PB : ℝ × ℝ := (B.1 - P.1, B.2 - P.2)
  PA.1 * PB.1 + PA.2 * PB.2 = 0 :=
by sorry

end perpendicular_dot_product_zero_l331_33163


namespace trash_outside_classrooms_l331_33197

-- Define the number of classrooms
def num_classrooms : Nat := 8

-- Define the total number of trash pieces picked up
def total_trash : Nat := 1576

-- Define the number of trash pieces picked up in each classroom
def classroom_trash : Fin num_classrooms → Nat
  | ⟨0, _⟩ => 124  -- Classroom 1
  | ⟨1, _⟩ => 98   -- Classroom 2
  | ⟨2, _⟩ => 176  -- Classroom 3
  | ⟨3, _⟩ => 212  -- Classroom 4
  | ⟨4, _⟩ => 89   -- Classroom 5
  | ⟨5, _⟩ => 241  -- Classroom 6
  | ⟨6, _⟩ => 121  -- Classroom 7
  | ⟨7, _⟩ => 102  -- Classroom 8
  | ⟨n+8, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 8 n))

-- Theorem to prove
theorem trash_outside_classrooms :
  total_trash - (Finset.sum Finset.univ classroom_trash) = 413 := by
  sorry

end trash_outside_classrooms_l331_33197


namespace transistors_in_2010_l331_33195

/-- The number of transistors in a typical CPU doubles every three years -/
def doubling_period : ℕ := 3

/-- The number of transistors in a typical CPU in 1992 -/
def initial_transistors : ℕ := 2000000

/-- The year from which we start counting -/
def initial_year : ℕ := 1992

/-- The year for which we want to calculate the number of transistors -/
def target_year : ℕ := 2010

/-- Calculates the number of transistors in a given year -/
def transistors_in_year (year : ℕ) : ℕ :=
  initial_transistors * 2^((year - initial_year) / doubling_period)

theorem transistors_in_2010 :
  transistors_in_year target_year = 128000000 := by
  sorry

end transistors_in_2010_l331_33195


namespace shortest_median_le_longest_angle_bisector_l331_33158

/-- Represents a triangle with side lengths a ≤ b ≤ c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hab : a ≤ b
  hbc : b ≤ c

/-- The length of the shortest median in a triangle -/
def shortestMedian (t : Triangle) : ℝ := sorry

/-- The length of the longest angle bisector in a triangle -/
def longestAngleBisector (t : Triangle) : ℝ := sorry

/-- Theorem: The shortest median is always shorter than or equal to the longest angle bisector -/
theorem shortest_median_le_longest_angle_bisector (t : Triangle) :
  shortestMedian t ≤ longestAngleBisector t := by sorry

end shortest_median_le_longest_angle_bisector_l331_33158


namespace ratio_change_proof_l331_33194

/-- Represents the ratio of bleach, detergent, and water in a solution -/
structure SolutionRatio where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- The original solution ratio -/
def original_ratio : SolutionRatio := ⟨2, 40, 100⟩

/-- The altered solution ratio -/
def altered_ratio : SolutionRatio := ⟨6, 60, 300⟩

/-- The factor by which the ratio of detergent to water changes -/
def ratio_change_factor : ℚ := 2

theorem ratio_change_proof : 
  (original_ratio.detergent / original_ratio.water) / 
  (altered_ratio.detergent / altered_ratio.water) = ratio_change_factor := by
  sorry

end ratio_change_proof_l331_33194
