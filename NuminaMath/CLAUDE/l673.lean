import Mathlib

namespace john_rachel_toy_difference_l673_67381

theorem john_rachel_toy_difference (jason_toys : ℕ) (rachel_toys : ℕ) :
  jason_toys = 21 →
  rachel_toys = 1 →
  ∃ (john_toys : ℕ),
    jason_toys = 3 * john_toys ∧
    john_toys > rachel_toys ∧
    john_toys - rachel_toys = 6 :=
by sorry

end john_rachel_toy_difference_l673_67381


namespace xyz_inequality_and_sum_l673_67367

theorem xyz_inequality_and_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 8) :
  ((x + y < 7) → (x / (1 + x) + y / (1 + y) > 2 * Real.sqrt ((x * y) / (x * y + 8)))) ∧
  (⌈(1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) + 1 / Real.sqrt (1 + z))⌉ = 2) := by
sorry

end xyz_inequality_and_sum_l673_67367


namespace lucas_pet_capacity_l673_67350

/-- The number of pets Lucas can accommodate given his pet bed situation -/
def pets_accommodated (initial_beds : ℕ) (additional_beds : ℕ) (beds_per_pet : ℕ) : ℕ :=
  (initial_beds + additional_beds) / beds_per_pet

theorem lucas_pet_capacity : pets_accommodated 12 8 2 = 10 := by
  sorry

end lucas_pet_capacity_l673_67350


namespace less_expensive_coat_cost_l673_67391

/-- Represents the cost of a coat and its lifespan in years -/
structure Coat where
  cost : ℕ
  lifespan : ℕ

/-- Calculates the total cost of a coat over a given period -/
def totalCost (coat : Coat) (period : ℕ) : ℕ :=
  (period / coat.lifespan) * coat.cost

theorem less_expensive_coat_cost (expensive_coat less_expensive_coat : Coat) : 
  expensive_coat.cost = 300 →
  expensive_coat.lifespan = 15 →
  less_expensive_coat.lifespan = 5 →
  totalCost expensive_coat 30 + 120 = totalCost less_expensive_coat 30 →
  less_expensive_coat.cost = 120 := by
sorry

end less_expensive_coat_cost_l673_67391


namespace ocean_area_scientific_notation_l673_67303

-- Define the original number
def original_number : ℝ := 2997000

-- Define the scientific notation components
def scientific_base : ℝ := 2.997
def scientific_exponent : ℤ := 6

-- Theorem statement
theorem ocean_area_scientific_notation :
  original_number = scientific_base * (10 : ℝ) ^ scientific_exponent :=
by sorry

end ocean_area_scientific_notation_l673_67303


namespace gina_netflix_minutes_l673_67370

/-- Represents the number of times Gina chooses what to watch compared to her sister -/
def gina_choice_ratio : ℕ := 3

/-- Represents the number of times Gina's sister chooses what to watch -/
def sister_choice_ratio : ℕ := 1

/-- The number of shows Gina's sister watches per week -/
def sister_shows_per_week : ℕ := 24

/-- The length of each show in minutes -/
def show_length : ℕ := 50

/-- Theorem stating that Gina chooses 3600 minutes of Netflix per week -/
theorem gina_netflix_minutes :
  (sister_shows_per_week * gina_choice_ratio * show_length) / (gina_choice_ratio + sister_choice_ratio) = 3600 :=
sorry

end gina_netflix_minutes_l673_67370


namespace last_remaining_is_125_l673_67343

/-- Represents the marking process on a list of numbers -/
def markingProcess (n : ℕ) : ℕ → Prop :=
  sorry

/-- The last remaining number after the marking process -/
def lastRemaining (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the last remaining number is 125 when starting with 150 numbers -/
theorem last_remaining_is_125 : lastRemaining 150 = 125 :=
  sorry

end last_remaining_is_125_l673_67343


namespace polynomial_remainder_theorem_l673_67362

/-- Given a polynomial p(x) = ax³ + bx² + cx + d -/
def p (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem polynomial_remainder_theorem (a b c d : ℝ) :
  (∃ q₁ : ℝ → ℝ, ∀ x, p a b c d x = (x - 1) * q₁ x + 1) →
  (∃ q₂ : ℝ → ℝ, ∀ x, p a b c d x = (x - 2) * q₂ x + 3) →
  ∃ q : ℝ → ℝ, ∀ x, p a b c d x = (x - 1) * (x - 2) * q x + (2 * x - 1) :=
by
  sorry

end polynomial_remainder_theorem_l673_67362


namespace expression_equality_l673_67384

theorem expression_equality : 2 * (2^7 + 2^7 + 2^8)^(1/4) = 8 * 2^(1/4) := by
  sorry

end expression_equality_l673_67384


namespace b_age_is_27_l673_67390

/-- The ages of four people A, B, C, and D. -/
structure Ages where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The conditions of the problem. -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.a + ages.b + ages.c + ages.d) / 4 = 28 ∧
  (ages.a + ages.c) / 2 = 29 ∧
  (2 * ages.b + 3 * ages.d) / 5 = 27 ∧
  ages.a = 1.1 * (ages.a / 1.1) ∧
  ages.c = 1.1 * (ages.c / 1.1) ∧
  ages.b = 1.15 * (ages.b / 1.15) ∧
  ages.d = 1.15 * (ages.d / 1.15)

/-- The theorem stating that given the problem conditions, B's age is 27. -/
theorem b_age_is_27 (ages : Ages) (h : problem_conditions ages) : ages.b = 27 := by
  sorry

end b_age_is_27_l673_67390


namespace consecutive_root_count_l673_67363

/-- A function that checks if a number is divisible by 5 -/
def divisible_by_five (m : ℤ) : Prop := ∃ k : ℤ, m = 5 * k

/-- A function that checks if two integers are consecutive -/
def consecutive (a b : ℤ) : Prop := b = a + 1

/-- A function that checks if a number is a positive integer -/
def is_positive_integer (x : ℤ) : Prop := x > 0

/-- The main theorem -/
theorem consecutive_root_count :
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, n < 50 ∧ is_positive_integer n) ∧ 
    (∀ n ∈ S, ∃ m : ℤ, 
      divisible_by_five m ∧
      ∃ a b : ℤ, is_positive_integer a ∧ is_positive_integer b ∧ consecutive a b ∧
      a * b = m ∧ a + b = n) ∧
    Finset.card S = 5 := by sorry

end consecutive_root_count_l673_67363


namespace f_has_two_zeros_l673_67317

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -1 + Real.log x else 3 * x + 4

theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end f_has_two_zeros_l673_67317


namespace jasons_tip_is_two_dollars_l673_67357

/-- Calculates the tip amount given the check amount, tax rate, and customer payment. -/
def calculate_tip (check_amount : ℝ) (tax_rate : ℝ) (customer_payment : ℝ) : ℝ :=
  let total_with_tax := check_amount * (1 + tax_rate)
  customer_payment - total_with_tax

/-- Proves that given the specific conditions, Jason's tip is $2.00 -/
theorem jasons_tip_is_two_dollars :
  calculate_tip 15 0.2 20 = 2 := by
  sorry

end jasons_tip_is_two_dollars_l673_67357


namespace gcd_459_357_l673_67307

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by sorry

end gcd_459_357_l673_67307


namespace magnitude_ratio_not_sufficient_for_parallel_l673_67360

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : V) : Prop := ∃ k : ℝ, a = k • b

theorem magnitude_ratio_not_sufficient_for_parallel (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ (a b : V), ‖a‖ = 2 * ‖b‖ → parallel a b) := by
  sorry


end magnitude_ratio_not_sufficient_for_parallel_l673_67360


namespace complex_magnitude_squared_l673_67321

theorem complex_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 4 + 6*I) : 
  Complex.abs z^2 = 13/2 := by
  sorry

end complex_magnitude_squared_l673_67321


namespace distance_washington_to_idaho_l673_67376

/-- The distance from Washington to Idaho in miles -/
def distance_WI : ℝ := 640

/-- The distance from Idaho to Nevada in miles -/
def distance_IN : ℝ := 550

/-- The speed from Washington to Idaho in miles per hour -/
def speed_WI : ℝ := 80

/-- The speed from Idaho to Nevada in miles per hour -/
def speed_IN : ℝ := 50

/-- The total travel time in hours -/
def total_time : ℝ := 19

/-- Theorem stating that the distance from Washington to Idaho is 640 miles -/
theorem distance_washington_to_idaho : 
  distance_WI = 640 ∧ 
  distance_WI / speed_WI + distance_IN / speed_IN = total_time := by
  sorry


end distance_washington_to_idaho_l673_67376


namespace smallest_undefined_inverse_seven_undefined_inverse_smallest_is_seven_l673_67300

theorem smallest_undefined_inverse (b : ℕ) : b > 0 ∧ 
  ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 77]) → 
  b ≥ 7 :=
by sorry

theorem seven_undefined_inverse : 
  ¬ (∃ x : ℕ, x * 7 ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ x : ℕ, x * 7 ≡ 1 [MOD 77]) :=
by sorry

theorem smallest_is_seven : 
  ∃ (b : ℕ), b > 0 ∧ 
  ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 77]) ∧
  ∀ (c : ℕ), c > 0 ∧ 
  ¬ (∃ x : ℕ, x * c ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ x : ℕ, x * c ≡ 1 [MOD 77]) →
  c ≥ b :=
by sorry

end smallest_undefined_inverse_seven_undefined_inverse_smallest_is_seven_l673_67300


namespace coordinate_transform_sum_l673_67396

/-- Definition of the original coordinate system -/
structure OriginalCoord where
  x : ℝ
  y : ℝ

/-- Definition of the new coordinate system -/
structure NewCoord where
  x : ℝ
  y : ℝ

/-- Definition of a line -/
structure Line where
  slope : ℝ
  point : OriginalCoord

/-- Function to transform coordinates from original to new system -/
def transform (p : OriginalCoord) (L M : Line) : NewCoord :=
  sorry

/-- Theorem statement -/
theorem coordinate_transform_sum :
  let A : OriginalCoord := ⟨24, -1⟩
  let B : OriginalCoord := ⟨5, 6⟩
  let P : OriginalCoord := ⟨-14, 27⟩
  let L : Line := ⟨5/12, A⟩
  let M : Line := ⟨-12/5, B⟩  -- Perpendicular slope
  let new_P : NewCoord := transform P L M
  new_P.x + new_P.y = 31 := by
  sorry

end coordinate_transform_sum_l673_67396


namespace score_difference_theorem_l673_67322

def score_distribution : List (Float × Float) := [
  (75, 0.15),
  (85, 0.30),
  (90, 0.25),
  (95, 0.10),
  (100, 0.20)
]

def mean (dist : List (Float × Float)) : Float :=
  (dist.map (fun (score, freq) => score * freq)).sum

def median (dist : List (Float × Float)) : Float :=
  90  -- The median is 90 based on the given distribution

theorem score_difference_theorem :
  mean score_distribution - median score_distribution = -1.25 := by
  sorry

end score_difference_theorem_l673_67322


namespace parking_lot_vehicles_l673_67334

-- Define the initial number of cars and trucks
def initial_cars : ℝ := 14
def initial_trucks : ℝ := 49

-- Define the changes in the parking lot
def cars_left : ℕ := 3
def trucks_arrived : ℕ := 6

-- Define the ratios
def initial_ratio : ℝ := 3.5
def final_ratio : ℝ := 2.3

-- Theorem statement
theorem parking_lot_vehicles :
  -- Initial condition
  initial_cars = initial_ratio * initial_trucks ∧
  -- Final condition after changes
  (initial_cars - cars_left) = final_ratio * (initial_trucks + trucks_arrived) →
  -- Conclusion: Total number of vehicles originally parked
  initial_cars + initial_trucks = 63 :=
by
  sorry -- Proof is omitted as per instructions

end parking_lot_vehicles_l673_67334


namespace cricket_overs_calculation_l673_67311

theorem cricket_overs_calculation (total_target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) 
  (remaining_overs : ℝ) (h1 : total_target = 262) (h2 : initial_rate = 3.2) 
  (h3 : required_rate = 5.75) (h4 : remaining_overs = 40) : 
  ∃ (initial_overs : ℝ), initial_overs = 10 ∧ 
  total_target = initial_rate * initial_overs + required_rate * remaining_overs :=
by
  sorry

end cricket_overs_calculation_l673_67311


namespace max_value_sin_squared_minus_two_sin_minus_two_l673_67372

theorem max_value_sin_squared_minus_two_sin_minus_two :
  ∀ x : ℝ, 
    -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 →
    ∀ y : ℝ, 
      y = Real.sin x ^ 2 - 2 * Real.sin x - 2 →
      y ≤ 1 ∧ ∃ x₀ : ℝ, Real.sin x₀ ^ 2 - 2 * Real.sin x₀ - 2 = 1 :=
sorry

end max_value_sin_squared_minus_two_sin_minus_two_l673_67372


namespace range_of_p_exists_point_C_l673_67359

-- Define the parabola L: x^2 = 2py
def L (p : ℝ) := {(x, y) : ℝ × ℝ | x^2 = 2*p*y ∧ p > 0}

-- Define point M
def M : ℝ × ℝ := (2, 2)

-- Define the condition for points A and B
def satisfies_condition (A B : ℝ × ℝ) (p : ℝ) :=
  A ∈ L p ∧ B ∈ L p ∧ A ≠ B ∧ 
  (A.1 - M.1, A.2 - M.2) = (-B.1 + M.1, -B.2 + M.2)

-- Theorem 1: Range of p
theorem range_of_p (p : ℝ) :
  (∃ A B, satisfies_condition A B p) → p > 1 :=
sorry

-- Define the circle through three points
def circle_through (A B C : ℝ × ℝ) := 
  {(x, y) : ℝ × ℝ | (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
                    (x - A.1)^2 + (y - A.2)^2 = (x - C.1)^2 + (y - C.2)^2}

-- Define the tangent line to the parabola at a point
def tangent_line (p : ℝ) (C : ℝ × ℝ) :=
  {(x, y) : ℝ × ℝ | y - C.2 = (C.1 / (2*p)) * (x - C.1)}

-- Theorem 2: Existence of point C when p = 2
theorem exists_point_C :
  ∃ C, C ∈ L 2 ∧ C ≠ (0, 0) ∧ C ≠ (4, 4) ∧
       C.1 = -2 ∧ C.2 = 1 ∧
       (∀ x y, (x, y) ∈ circle_through (0, 0) (4, 4) C →
               (x, y) ∈ tangent_line 2 C) :=
sorry

end range_of_p_exists_point_C_l673_67359


namespace count_valid_plans_l673_67388

/-- Represents a teacher --/
inductive Teacher : Type
  | A | B | C | D | E

/-- Represents a remote area --/
inductive Area : Type
  | One | Two | Three

/-- A dispatch plan assigns teachers to areas --/
def DispatchPlan := Teacher → Area

/-- Checks if a dispatch plan is valid according to the given conditions --/
def isValidPlan (plan : DispatchPlan) : Prop :=
  (∀ a : Area, ∃ t : Teacher, plan t = a) ∧  -- Each area has at least 1 person
  (plan Teacher.A ≠ plan Teacher.B) ∧        -- A and B are not in the same area
  (plan Teacher.A = plan Teacher.C)          -- A and C are in the same area

/-- The number of valid dispatch plans --/
def numValidPlans : ℕ := sorry

/-- Theorem stating that the number of valid dispatch plans is 30 --/
theorem count_valid_plans : numValidPlans = 30 := by sorry

end count_valid_plans_l673_67388


namespace points_four_units_from_negative_two_l673_67316

def distance (x y : ℝ) : ℝ := |x - y|

theorem points_four_units_from_negative_two : 
  {x : ℝ | distance x (-2) = 4} = {2, -6} := by
  sorry

end points_four_units_from_negative_two_l673_67316


namespace song_count_difference_l673_67399

/- Define the problem parameters -/
def total_days_in_june : ℕ := 30
def weekend_days : ℕ := 8
def vivian_daily_songs : ℕ := 10
def total_monthly_songs : ℕ := 396

/- Calculate the number of days they played songs -/
def playing_days : ℕ := total_days_in_june - weekend_days

/- Calculate Vivian's total songs for the month -/
def vivian_monthly_songs : ℕ := vivian_daily_songs * playing_days

/- Calculate Clara's total songs for the month -/
def clara_monthly_songs : ℕ := total_monthly_songs - vivian_monthly_songs

/- Calculate Clara's daily song count -/
def clara_daily_songs : ℕ := clara_monthly_songs / playing_days

/- Theorem to prove -/
theorem song_count_difference : vivian_daily_songs - clara_daily_songs = 2 := by
  sorry

end song_count_difference_l673_67399


namespace floor_abs_sum_l673_67318

theorem floor_abs_sum : ⌊|(-3.7 : ℝ)|⌋ + |⌊(-3.7 : ℝ)⌋| = 7 := by
  sorry

end floor_abs_sum_l673_67318


namespace floor_sqrt_23_squared_l673_67352

theorem floor_sqrt_23_squared : ⌊Real.sqrt 23⌋^2 = 16 := by
  sorry

end floor_sqrt_23_squared_l673_67352


namespace circle_differences_l673_67304

theorem circle_differences (n : ℕ) (a : ℕ → ℝ) 
  (h : ∀ i, |a i - a ((i + 1) % n)| ≥ 2 * |a i - a ((i + 2) % n)|) :
  ∀ i, |a i - a ((i + 3) % n)| ≥ |a i - a ((i + 2) % n)| :=
by sorry

end circle_differences_l673_67304


namespace point_coordinates_sum_l673_67312

/-- Given points A, B, C in a plane rectangular coordinate system,
    where AB is parallel to the x-axis and AC is parallel to the y-axis,
    prove that a + b = -1 -/
theorem point_coordinates_sum (a b : ℝ) : 
  (∃ (A B C : ℝ × ℝ),
    A = (a, -1) ∧
    B = (2, 3 - b) ∧
    C = (-5, 4) ∧
    A.2 = B.2 ∧  -- AB is parallel to x-axis
    A.1 = C.1    -- AC is parallel to y-axis
  ) →
  a + b = -1 := by
  sorry

end point_coordinates_sum_l673_67312


namespace company_workers_count_l673_67361

/-- Represents the hierarchical structure of a company -/
structure CompanyHierarchy where
  supervisors : ℕ
  teamLeadsPerSupervisor : ℕ
  workersPerTeamLead : ℕ

/-- Calculates the total number of workers in a company given its hierarchy -/
def totalWorkers (c : CompanyHierarchy) : ℕ :=
  c.supervisors * c.teamLeadsPerSupervisor * c.workersPerTeamLead

/-- Theorem stating that a company with 13 supervisors, 3 team leads per supervisor,
    and 10 workers per team lead has 390 workers in total -/
theorem company_workers_count :
  let c : CompanyHierarchy := {
    supervisors := 13,
    teamLeadsPerSupervisor := 3,
    workersPerTeamLead := 10
  }
  totalWorkers c = 390 := by
  sorry


end company_workers_count_l673_67361


namespace inequality_system_solution_set_l673_67342

theorem inequality_system_solution_set :
  {x : ℝ | (6 - 2*x ≥ 0) ∧ (2*x + 4 > 0)} = {x : ℝ | -2 < x ∧ x ≤ 3} := by
  sorry

end inequality_system_solution_set_l673_67342


namespace no_natural_solution_l673_67369

theorem no_natural_solution : ¬ ∃ (m n : ℕ), (1 : ℚ) / m + (1 : ℚ) / n = (7 : ℚ) / 100 := by
  sorry

end no_natural_solution_l673_67369


namespace pasta_bins_l673_67320

theorem pasta_bins (total_bins soup_bins vegetable_bins : ℝ) 
  (h_total : total_bins = 0.75)
  (h_soup : soup_bins = 0.12)
  (h_vegetable : vegetable_bins = 0.12) :
  total_bins - soup_bins - vegetable_bins = 0.51 := by
sorry

end pasta_bins_l673_67320


namespace monotonicity_and_minimum_l673_67333

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + a + 1)

def f_derivative (a : ℝ) (x : ℝ) : ℝ := 
  Real.exp (-x) * (-a * x^2 + 2 * a * x - a - 1)

theorem monotonicity_and_minimum :
  (∀ x, a ≥ 0 → f_derivative a x < 0) ∧
  (a < 0 → ∃ r₁ r₂, r₁ < r₂ ∧ r₂ < 0 ∧
    (∀ x, x < r₁ → f_derivative a x > 0) ∧
    (∀ x, r₁ < x ∧ x < r₂ → f_derivative a x < 0) ∧
    (∀ x, x > r₂ → f_derivative a x > 0)) ∧
  (-1 < a ∧ a < 0 → ∀ x, 1 ≤ x ∧ x ≤ 2 → f a x ≥ f a 2) :=
by sorry

end

end monotonicity_and_minimum_l673_67333


namespace carson_gardening_time_l673_67398

/-- The total time Carson spends gardening is 108 minutes -/
theorem carson_gardening_time :
  let lines_to_mow : ℕ := 40
  let time_per_line : ℕ := 2
  let flower_rows : ℕ := 8
  let flowers_per_row : ℕ := 7
  let time_per_flower : ℚ := 1/2
  lines_to_mow * time_per_line + flower_rows * flowers_per_row * time_per_flower = 108 := by
  sorry

end carson_gardening_time_l673_67398


namespace roses_before_and_after_cutting_l673_67371

/-- Given the initial conditions of Mary's rose garden, prove the number of roses before and after cutting. -/
theorem roses_before_and_after_cutting 
  (R : ℕ) -- Initial number of roses in the garden
  (B : ℕ) -- Number of roses left in the garden after cutting
  (h1 : R = B + 10) -- Relation between R and B
  (h2 : ∃ C : ℕ, C = 10 ∧ R - C = B) -- Existence of C satisfying the conditions
  : R = B + 10 ∧ R - 10 = B := by
  sorry

end roses_before_and_after_cutting_l673_67371


namespace bridge_length_l673_67380

/-- The length of a bridge given specific train conditions -/
theorem bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  train_speed = 45 * 1000 / 3600 →
  crossing_time = 30 →
  train_speed * crossing_time - train_length = 205 :=
by
  sorry

end bridge_length_l673_67380


namespace cookies_sold_l673_67335

/-- Proves the number of cookies sold given the problem conditions -/
theorem cookies_sold (original_cupcake_price original_cookie_price : ℚ)
  (price_reduction : ℚ) (cupcakes_sold : ℕ) (total_revenue : ℚ)
  (h1 : original_cupcake_price = 3)
  (h2 : original_cookie_price = 2)
  (h3 : price_reduction = 1/2)
  (h4 : cupcakes_sold = 16)
  (h5 : total_revenue = 32) :
  (total_revenue - cupcakes_sold * (original_cupcake_price * price_reduction)) / (original_cookie_price * price_reduction) = 8 := by
  sorry

end cookies_sold_l673_67335


namespace box_packing_l673_67394

theorem box_packing (total_items : Nat) (items_per_small_box : Nat) (small_boxes_per_big_box : Nat)
  (h1 : total_items = 8640)
  (h2 : items_per_small_box = 12)
  (h3 : small_boxes_per_big_box = 6)
  (h4 : items_per_small_box > 0)
  (h5 : small_boxes_per_big_box > 0) :
  total_items / (items_per_small_box * small_boxes_per_big_box) = 120 := by
sorry

end box_packing_l673_67394


namespace sufficient_not_necessary_condition_l673_67348

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x : ℝ, (abs x > a → x^2 - x - 2 > 0) ∧ 
  (∃ y : ℝ, y^2 - y - 2 > 0 ∧ abs y ≤ a)) ↔ 
  a ≥ 2 := by
sorry

end sufficient_not_necessary_condition_l673_67348


namespace domain_subset_iff_a_range_l673_67374

theorem domain_subset_iff_a_range (a : ℝ) (h : a < 1) :
  (∀ x, (x - a - 1) * (2 * a - x) > 0 → x ∈ (Set.Iic (-1) ∪ Set.Ici 1)) ↔
  a ∈ (Set.Iic (-2) ∪ Set.Icc (1/2) 1) :=
sorry

end domain_subset_iff_a_range_l673_67374


namespace correlation_relationships_l673_67302

/-- A relationship between two variables -/
inductive Relationship
| AgeWealth
| CurveCoordinates
| AppleProductionClimate
| TreeDiameterHeight
| StudentSchool

/-- Predicate to determine if a relationship involves correlation -/
def involves_correlation (r : Relationship) : Prop :=
  match r with
  | Relationship.AgeWealth => true
  | Relationship.CurveCoordinates => false
  | Relationship.AppleProductionClimate => true
  | Relationship.TreeDiameterHeight => true
  | Relationship.StudentSchool => false

/-- The set of all relationships -/
def all_relationships : Set Relationship :=
  {Relationship.AgeWealth, Relationship.CurveCoordinates, Relationship.AppleProductionClimate,
   Relationship.TreeDiameterHeight, Relationship.StudentSchool}

/-- The theorem stating which relationships involve correlation -/
theorem correlation_relationships :
  {r ∈ all_relationships | involves_correlation r} =
  {Relationship.AgeWealth, Relationship.AppleProductionClimate, Relationship.TreeDiameterHeight} :=
by sorry

end correlation_relationships_l673_67302


namespace difference_of_odd_squares_divisible_by_eight_l673_67336

theorem difference_of_odd_squares_divisible_by_eight (n p : ℤ) :
  ∃ k : ℤ, (2 * n + 1)^2 - (2 * p + 1)^2 = 8 * k := by
  sorry

end difference_of_odd_squares_divisible_by_eight_l673_67336


namespace tan_pi_four_minus_theta_l673_67358

theorem tan_pi_four_minus_theta (θ : Real) (h : (Real.tan θ) = -2) :
  Real.tan (π / 4 - θ) = -3 := by
  sorry

end tan_pi_four_minus_theta_l673_67358


namespace simple_interest_principal_calculation_l673_67364

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation
  (rate : ℚ) (interest : ℚ) (time : ℕ) :
  rate = 4 / 100 →
  interest = 128 →
  time = 4 →
  ∃ (principal : ℚ), principal * rate * time = interest ∧ principal = 800 :=
by sorry

end simple_interest_principal_calculation_l673_67364


namespace first_meeting_cd_l673_67353

-- Define the cars and their properties
structure Car where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

-- Define the race scenario
def race_scenario (a b c d : Car) : Prop :=
  a.direction ∧ b.direction ∧ ¬c.direction ∧ ¬d.direction ∧
  a.speed ≠ b.speed ∧ a.speed ≠ c.speed ∧ a.speed ≠ d.speed ∧
  b.speed ≠ c.speed ∧ b.speed ≠ d.speed ∧ c.speed ≠ d.speed ∧
  a.speed + c.speed = b.speed + d.speed ∧
  a.speed - b.speed = d.speed - c.speed

-- Define the meeting times
def first_meeting_ac_bd : ℝ := 7
def first_meeting_ab : ℝ := 53

-- Theorem statement
theorem first_meeting_cd 
  (a b c d : Car) 
  (h : race_scenario a b c d) :
  ∃ t : ℝ, t = first_meeting_ab :=
sorry

end first_meeting_cd_l673_67353


namespace intersection_and_lines_l673_67378

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - 2*y + 4 = 0
def l₂ (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0

-- Define point A
def A : ℝ × ℝ := (-1, -2)

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 1)

-- Define the condition for a
def a_condition (a : ℝ) : Prop := a ≠ -2 ∧ a ≠ -1 ∧ a ≠ 8/3

-- Define the equations of line l
def l_eq₁ (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0
def l_eq₂ (x y : ℝ) : Prop := x + 2 = 0

theorem intersection_and_lines :
  -- 1. P is the intersection point of l₁ and l₂
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧
  -- 2. Condition for a to form a triangle
  (∀ a : ℝ, (∃ x y : ℝ, l₁ x y ∧ l₂ x y ∧ (a*x + 2*y - 6 = 0)) → a_condition a) ∧
  -- 3. Equations of line l passing through P with distance 1 from A
  (∀ x y : ℝ, (l_eq₁ x y ∨ l_eq₂ x y) ↔
    ((x - P.1)^2 + (y - P.2)^2 = 0 ∧
     ((x - A.1)^2 + (y - A.2)^2 - 1)^2 = 
     ((x - P.1)*(A.2 - P.2) - (y - P.2)*(A.1 - P.1))^2 / ((x - P.1)^2 + (y - P.2)^2)))
  := by sorry

end intersection_and_lines_l673_67378


namespace e_pi_plus_pi_e_approx_l673_67346

/-- Approximate value of e -/
def e_approx : ℝ := 2.718

/-- Approximate value of π -/
def π_approx : ℝ := 3.14159

/-- Theorem stating that e^π + π^e is approximately equal to 45.5999 -/
theorem e_pi_plus_pi_e_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |Real.exp π_approx + Real.exp e_approx - 45.5999| < ε :=
sorry

end e_pi_plus_pi_e_approx_l673_67346


namespace max_visible_cubes_9x9x9_l673_67314

/-- Represents a cube formed by unit cubes -/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point -/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  let face_area := cube.size ^ 2
  let total_faces := 3 * face_area
  let shared_edges := 3 * (cube.size - 1)
  let corner_cube := 1
  total_faces - shared_edges + corner_cube

/-- Theorem stating that for a 9x9x9 cube, the maximum number of visible unit cubes is 220 -/
theorem max_visible_cubes_9x9x9 :
  max_visible_cubes ⟨9⟩ = 220 := by
  sorry

#eval max_visible_cubes ⟨9⟩

end max_visible_cubes_9x9x9_l673_67314


namespace nathalie_cake_fraction_l673_67323

theorem nathalie_cake_fraction (cake_weight : ℝ) (num_parts : ℕ) 
  (pierre_amount : ℝ) (nathalie_fraction : ℝ) : 
  cake_weight = 400 →
  num_parts = 8 →
  pierre_amount = 100 →
  pierre_amount = 2 * (nathalie_fraction * cake_weight) →
  nathalie_fraction = 1 / 8 := by
  sorry

end nathalie_cake_fraction_l673_67323


namespace complex_equation_unit_circle_l673_67366

theorem complex_equation_unit_circle (z : ℂ) :
  11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0 →
  Complex.abs z = 1 := by
sorry

end complex_equation_unit_circle_l673_67366


namespace final_price_is_correct_l673_67347

def electronic_discount_rate : ℚ := 0.20
def clothing_discount_rate : ℚ := 0.15
def voucher_threshold : ℚ := 200
def voucher_value : ℚ := 20
def electronic_item_price : ℚ := 150
def clothing_item_price : ℚ := 80
def clothing_item_count : ℕ := 2

def calculate_final_price : ℚ := by
  -- Define the calculation here
  sorry

theorem final_price_is_correct :
  calculate_final_price = 236 := by
  sorry

end final_price_is_correct_l673_67347


namespace butcher_purchase_cost_l673_67330

/-- Calculates the total cost of a butcher's purchase given the weights and prices of various items. -/
theorem butcher_purchase_cost (steak_weight : ℚ) (steak_price : ℚ)
                               (chicken_weight : ℚ) (chicken_price : ℚ)
                               (sausage_weight : ℚ) (sausage_price : ℚ)
                               (pork_weight : ℚ) (pork_price : ℚ)
                               (bacon_weight : ℚ) (bacon_price : ℚ)
                               (salmon_weight : ℚ) (salmon_price : ℚ) :
  steak_weight = 3/2 ∧ steak_price = 15 ∧
  chicken_weight = 3/2 ∧ chicken_price = 8 ∧
  sausage_weight = 2 ∧ sausage_price = 13/2 ∧
  pork_weight = 7/2 ∧ pork_price = 10 ∧
  bacon_weight = 1/2 ∧ bacon_price = 9 ∧
  salmon_weight = 1/4 ∧ salmon_price = 30 →
  steak_weight * steak_price +
  chicken_weight * chicken_price +
  sausage_weight * sausage_price +
  pork_weight * pork_price +
  bacon_weight * bacon_price +
  salmon_weight * salmon_price = 189/2 := by
sorry


end butcher_purchase_cost_l673_67330


namespace max_area_difference_l673_67392

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the left vertex A and left focus F
def A : ℝ × ℝ := (-2, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define a line passing through F
def line_through_F (k : ℝ) (x y : ℝ) : Prop := x = k*y - 1

-- Define the intersection points C and D
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | E p.1 p.2 ∧ line_through_F k p.1 p.2}

-- Define the area difference function
def area_difference (C D : ℝ × ℝ) : ℝ :=
  |C.2 + D.2|

-- Theorem statement
theorem max_area_difference :
  ∃ (max_diff : ℝ), max_diff = Real.sqrt 3 / 2 ∧
  ∀ (k : ℝ) (C D : ℝ × ℝ),
    C ∈ intersection_points k → D ∈ intersection_points k →
    area_difference C D ≤ max_diff :=
sorry

end max_area_difference_l673_67392


namespace equation_has_real_root_l673_67319

theorem equation_has_real_root :
  ∃ x : ℝ, (Real.sqrt (x + 16) + 4 / Real.sqrt (x + 16) = 7) := by
  sorry

end equation_has_real_root_l673_67319


namespace jerica_louis_age_ratio_l673_67313

theorem jerica_louis_age_ratio :
  ∀ (jerica_age louis_age matilda_age : ℕ),
    louis_age = 14 →
    matilda_age = 35 →
    matilda_age = jerica_age + 7 →
    ∃ k : ℕ, jerica_age = k * louis_age →
    jerica_age / louis_age = 2 := by
  sorry

end jerica_louis_age_ratio_l673_67313


namespace sequence_a_property_l673_67324

def sequence_a (n : ℕ+) : ℚ := 1 / ((n + 1) * (n + 2))

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := (n * (n + 1) : ℚ) / 2 * a n

theorem sequence_a_property (a : ℕ+ → ℚ) : 
  a 1 = 1/6 → 
  (∀ n : ℕ+, S n a = (n * (n + 1) : ℚ) / 2 * a n) → 
  ∀ n : ℕ+, a n = sequence_a n :=
sorry

end sequence_a_property_l673_67324


namespace gray_sections_total_seeds_l673_67310

theorem gray_sections_total_seeds (circle1_total : ℕ) (circle2_total : ℕ) (white_section : ℕ)
  (h1 : circle1_total = 87)
  (h2 : circle2_total = 110)
  (h3 : white_section = 68) :
  (circle1_total - white_section) + (circle2_total - white_section) = 61 := by
  sorry

end gray_sections_total_seeds_l673_67310


namespace one_km_equals_500_chains_l673_67339

-- Define the units
def kilometer : ℕ → ℕ := id
def hectometer : ℕ → ℕ := id
def chain : ℕ → ℕ := id

-- Define the conversion factors
axiom km_to_hm : ∀ x : ℕ, kilometer x = hectometer (10 * x)
axiom hm_to_chain : ∀ x : ℕ, hectometer x = chain (50 * x)

-- Theorem to prove
theorem one_km_equals_500_chains : kilometer 1 = chain 500 := by
  sorry

end one_km_equals_500_chains_l673_67339


namespace balloon_arrangement_count_l673_67395

def balloon_permutations : ℕ := 1260

theorem balloon_arrangement_count :
  let total_letters : ℕ := 7
  let repeated_l : ℕ := 2
  let repeated_o : ℕ := 2
  balloon_permutations = Nat.factorial total_letters / (Nat.factorial repeated_l * Nat.factorial repeated_o) := by
  sorry

end balloon_arrangement_count_l673_67395


namespace arithmetic_calculation_l673_67385

theorem arithmetic_calculation : 4 * (8 - 3) / 2 - 7 = 3 := by
  sorry

end arithmetic_calculation_l673_67385


namespace equality_sum_l673_67331

theorem equality_sum (M N : ℚ) : 
  (3 : ℚ) / 5 = M / 30 ∧ (3 : ℚ) / 5 = 90 / N → M + N = 168 := by
  sorry

end equality_sum_l673_67331


namespace percentage_difference_l673_67397

theorem percentage_difference (A C : ℝ) (h1 : C > A) (h2 : A > 0) (h3 : C = 1.2 * A) :
  (C - A) / C * 100 = 50 / 3 := by
  sorry

end percentage_difference_l673_67397


namespace probability_three_blue_pens_l673_67356

def total_pens : ℕ := 15
def blue_pens : ℕ := 8
def red_pens : ℕ := 7
def num_trials : ℕ := 7
def num_blue_picks : ℕ := 3

def prob_blue : ℚ := blue_pens / total_pens
def prob_red : ℚ := red_pens / total_pens

def binomial_coefficient (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem probability_three_blue_pens :
  (binomial_coefficient num_trials num_blue_picks : ℚ) *
  (prob_blue ^ num_blue_picks) *
  (prob_red ^ (num_trials - num_blue_picks)) =
  43025920 / 170859375 := by sorry

end probability_three_blue_pens_l673_67356


namespace journey_time_difference_l673_67393

/-- Proves that the difference in arrival times is 15 minutes for a 70 km journey 
    when comparing speeds of 40 km/hr (on time) and 35 km/hr (late). -/
theorem journey_time_difference (distance : ℝ) (speed_on_time speed_late : ℝ) : 
  distance = 70 ∧ speed_on_time = 40 ∧ speed_late = 35 →
  (distance / speed_late - distance / speed_on_time) * 60 = 15 := by
sorry

end journey_time_difference_l673_67393


namespace painting_time_ratio_l673_67341

-- Define the painting times for each person
def matt_time : ℝ := 12
def rachel_time : ℝ := 13

-- Define Patty's time in terms of a variable
def patty_time : ℝ → ℝ := λ p => p

-- Define Rachel's time in terms of Patty's time
def rachel_time_calc : ℝ → ℝ := λ p => 2 * p + 5

-- Theorem statement
theorem painting_time_ratio :
  ∃ p : ℝ, 
    rachel_time_calc p = rachel_time ∧ 
    (patty_time p) / matt_time = 1 / 3 := by
  sorry

end painting_time_ratio_l673_67341


namespace sin_18_cos_12_plus_cos_18_sin_12_l673_67337

theorem sin_18_cos_12_plus_cos_18_sin_12 :
  Real.sin (18 * π / 180) * Real.cos (12 * π / 180) +
  Real.cos (18 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end sin_18_cos_12_plus_cos_18_sin_12_l673_67337


namespace three_sum_exists_l673_67355

theorem three_sum_exists (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h_pos : ∀ i, 0 < a i) 
  (h_increasing : ∀ i j, i < j → a i < a j) 
  (h_bound : a (Fin.last n) < 2 * n) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i + a j = a k :=
sorry

end three_sum_exists_l673_67355


namespace solve_for_t_l673_67305

theorem solve_for_t (Q m h t : ℝ) (hQ : Q > 0) (hm : m ≠ 0) (hh : h > -2) :
  Q = m^2 / (2 + h)^t ↔ t = Real.log (m^2 / Q) / Real.log (2 + h) :=
sorry

end solve_for_t_l673_67305


namespace arithmetic_sequence_ratio_property_l673_67389

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_correct : ∀ n, S n = (n : ℚ) * (a 1 + a n) / 2

/-- If S_2 / S_4 = 1/3 for an arithmetic sequence, then S_4 / S_8 = 3/10 -/
theorem arithmetic_sequence_ratio_property (seq : ArithmeticSequence) 
    (h : seq.S 2 / seq.S 4 = 1/3) : 
    seq.S 4 / seq.S 8 = 3/10 := by
  sorry

end arithmetic_sequence_ratio_property_l673_67389


namespace area_formula_l673_67382

/-- Triangle with sides a, b, c and angle A -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ

/-- Area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Theorem: Area formula for triangles with angle A = 60° or 120° -/
theorem area_formula (t : Triangle) :
  (t.angleA = 60 → area t = (Real.sqrt 3 / 4) * (t.a^2 - (t.b - t.c)^2)) ∧
  (t.angleA = 120 → area t = (Real.sqrt 3 / 12) * (t.a^2 - (t.b - t.c)^2)) := by
  sorry

end area_formula_l673_67382


namespace expression_evaluation_l673_67349

theorem expression_evaluation :
  let x : ℚ := 1/2
  let y : ℚ := -1/4
  ((3*x + 2*y) * (3*x - 2*y) - (3*x - 2*y)^2) / (4*y) = 2 := by sorry

end expression_evaluation_l673_67349


namespace farm_ratio_change_l673_67379

/-- Represents the farm's livestock inventory --/
structure Farm where
  horses : ℕ
  cows : ℕ

/-- Calculates the ratio of horses to cows as a pair of natural numbers --/
def ratio (f : Farm) : ℕ × ℕ :=
  let gcd := Nat.gcd f.horses f.cows
  (f.horses / gcd, f.cows / gcd)

theorem farm_ratio_change (initial : Farm) (final : Farm) : 
  (ratio initial = (3, 1)) →
  (final.horses = initial.horses - 15) →
  (final.cows = initial.cows + 15) →
  (final.horses = final.cows + 30) →
  (ratio final = (5, 3)) := by
  sorry


end farm_ratio_change_l673_67379


namespace distance_from_origin_l673_67328

theorem distance_from_origin (x y : ℝ) (h1 : y = 20) 
  (h2 : Real.sqrt ((x - 2)^2 + (y - 15)^2) = 15) (h3 : x > 2) : 
  Real.sqrt (x^2 + y^2) = Real.sqrt (604 + 40 * Real.sqrt 2) := by
sorry

end distance_from_origin_l673_67328


namespace polygon_d_largest_area_l673_67327

-- Define the structure of a polygon
structure Polygon where
  unitSquares : ℕ
  rightTriangles : ℕ

-- Define the area calculation function
def area (p : Polygon) : ℚ :=
  p.unitSquares + p.rightTriangles / 2

-- Define the five polygons
def polygonA : Polygon := ⟨6, 0⟩
def polygonB : Polygon := ⟨3, 4⟩
def polygonC : Polygon := ⟨4, 5⟩
def polygonD : Polygon := ⟨7, 0⟩
def polygonE : Polygon := ⟨2, 6⟩

-- Define the list of all polygons
def allPolygons : List Polygon := [polygonA, polygonB, polygonC, polygonD, polygonE]

-- Theorem: Polygon D has the largest area
theorem polygon_d_largest_area :
  ∀ p ∈ allPolygons, area polygonD ≥ area p :=
sorry

end polygon_d_largest_area_l673_67327


namespace inequality_solution_and_sum_of_roots_l673_67387

-- Define the inequality
def inequality (m n x : ℝ) : Prop :=
  |x^2 + m*x + n| ≤ |3*x^2 - 6*x - 9|

-- Main theorem
theorem inequality_solution_and_sum_of_roots (m n : ℝ) 
  (h : ∀ x, inequality m n x) : 
  m = -2 ∧ n = -3 ∧ 
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = m - n → 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
sorry


end inequality_solution_and_sum_of_roots_l673_67387


namespace reciprocal_negative_four_l673_67326

theorem reciprocal_negative_four (x : ℚ) : x⁻¹ = -4 → x = -1/4 := by
  sorry

end reciprocal_negative_four_l673_67326


namespace unreachable_141_l673_67340

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  (n % 10) * digit_product (n / 10)

def next_number (n : ℕ) : Set ℕ :=
  {n + digit_product n, n - digit_product n}

def reachable (start : ℕ) : Set ℕ :=
  sorry

theorem unreachable_141 :
  141 ∉ reachable 141 \ {141} :=
sorry

end unreachable_141_l673_67340


namespace total_cds_l673_67365

/-- The number of CDs each person has -/
structure CDCounts where
  dawn : ℕ
  kristine : ℕ
  mark : ℕ
  alice : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (c : CDCounts) : Prop :=
  c.dawn = 10 ∧
  c.kristine = c.dawn + 7 ∧
  c.mark = 2 * c.kristine ∧
  c.alice = c.kristine + c.mark - 5

/-- The theorem to prove -/
theorem total_cds (c : CDCounts) (h : satisfiesConditions c) :
  c.dawn + c.kristine + c.mark + c.alice = 107 := by
  sorry

#check total_cds

end total_cds_l673_67365


namespace bridge_length_calculation_l673_67315

/-- Given a train crossing a bridge, calculate the length of the bridge -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 ∧ train_speed_kmh = 45 ∧ crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 235 := by
  sorry

end bridge_length_calculation_l673_67315


namespace thursday_spending_l673_67345

def monday_savings : ℕ := 15
def tuesday_savings : ℕ := 28
def wednesday_savings : ℕ := 13

def total_savings : ℕ := monday_savings + tuesday_savings + wednesday_savings

theorem thursday_spending :
  (total_savings : ℚ) / 2 = 28 := by sorry

end thursday_spending_l673_67345


namespace carries_strawberry_harvest_l673_67306

/-- Represents the dimensions of Carrie's garden -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Represents the planting and yield information -/
structure PlantingInfo where
  plantsPerSquareFoot : ℕ
  strawberriesPerPlant : ℕ

/-- Calculates the expected strawberry harvest given garden dimensions and planting information -/
def expectedHarvest (garden : GardenDimensions) (info : PlantingInfo) : ℕ :=
  garden.length * garden.width * info.plantsPerSquareFoot * info.strawberriesPerPlant

/-- Theorem stating that Carrie's expected strawberry harvest is 3150 -/
theorem carries_strawberry_harvest :
  let garden := GardenDimensions.mk 7 9
  let info := PlantingInfo.mk 5 10
  expectedHarvest garden info = 3150 := by
  sorry

end carries_strawberry_harvest_l673_67306


namespace expression_always_zero_l673_67373

theorem expression_always_zero (x y : ℝ) : 
  5 * (x^3 - 3*x^2*y - 2*x*y^2) - 3 * (x^3 - 5*x^2*y + 2*y^3) + 2 * (-x^3 + 5*x*y^2 + 3*y^3) = 0 := by
  sorry

end expression_always_zero_l673_67373


namespace prob_one_common_correct_l673_67368

/-- The number of numbers in the lottery -/
def total_numbers : ℕ := 45

/-- The number of numbers each participant chooses -/
def chosen_numbers : ℕ := 6

/-- Calculates the probability of exactly one common number between two independently chosen combinations -/
def prob_one_common : ℚ :=
  (chosen_numbers : ℚ) * (Nat.choose (total_numbers - chosen_numbers) (chosen_numbers - 1) : ℚ) /
  (Nat.choose total_numbers chosen_numbers : ℚ)

/-- Theorem stating that the probability of exactly one common number is correct -/
theorem prob_one_common_correct :
  prob_one_common = (6 : ℚ) * (Nat.choose 39 5 : ℚ) / (Nat.choose 45 6 : ℚ) :=
by sorry

end prob_one_common_correct_l673_67368


namespace new_quad_inscribable_l673_67301

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define the points on the circle
variable (A₁ A₂ B₁ B₂ C₁ C₂ D₁ D₂ : ℝ × ℝ)

-- Define the convex quadrilateral
variable (quad : Set (ℝ × ℝ))

-- Define the condition that the quadrilateral is inscribed in the circle
variable (quad_inscribed : quad ⊆ circle)

-- Define the condition that the extended sides intersect the circle at the given points
variable (extended_sides : 
  A₁ ∈ circle ∧ A₂ ∈ circle ∧ 
  B₁ ∈ circle ∧ B₂ ∈ circle ∧ 
  C₁ ∈ circle ∧ C₂ ∈ circle ∧ 
  D₁ ∈ circle ∧ D₂ ∈ circle)

-- Define the equality condition
variable (equality_condition : 
  dist A₁ B₂ = dist B₁ C₂ ∧ 
  dist B₁ C₂ = dist C₁ D₂ ∧ 
  dist C₁ D₂ = dist D₁ A₂)

-- Define the quadrilateral formed by the lines A₁A₂, B₁B₂, C₁C₂, D₁D₂
def new_quad : Set (ℝ × ℝ) := sorry

-- The theorem to be proved
theorem new_quad_inscribable :
  ∃ (new_circle : Set (ℝ × ℝ)), new_quad ⊆ new_circle :=
sorry

end new_quad_inscribable_l673_67301


namespace identity_mapping_implies_sum_l673_67377

theorem identity_mapping_implies_sum (a b : ℝ) : 
  let M : Set ℝ := {-1, b/a, 1}
  let N : Set ℝ := {a, b, b-a}
  (∀ x ∈ M, x ∈ N) → (a + b = 1 ∨ a + b = -1) := by
  sorry

end identity_mapping_implies_sum_l673_67377


namespace exists_m_f_even_l673_67332

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = x^2 + mx -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x

/-- There exists an m ∈ ℝ such that f(x) = x^2 + mx is an even function -/
theorem exists_m_f_even : ∃ m : ℝ, IsEven (f m) := by
  sorry

end exists_m_f_even_l673_67332


namespace variance_of_binomial_distribution_l673_67325

/-- The number of trials -/
def n : ℕ := 100

/-- The probability of success (drawing a second) -/
def p : ℝ := 0.02

/-- The variance of a binomial distribution -/
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- Theorem: The variance of the given binomial distribution is 1.96 -/
theorem variance_of_binomial_distribution :
  binomial_variance n p = 1.96 := by
  sorry

end variance_of_binomial_distribution_l673_67325


namespace xy_sum_problem_l673_67354

theorem xy_sum_problem (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (hx_bound : x < 15) (hy_bound : y < 15) (h_eq : x + y + x * y = 119) : 
  x + y = 21 ∨ x + y = 20 := by
sorry

end xy_sum_problem_l673_67354


namespace age_problem_l673_67308

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = a - 3 →
  a + b + c + d = 44 →
  b = 12 := by
sorry

end age_problem_l673_67308


namespace intersection_complement_equality_l673_67383

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : S ∩ (U \ T) = {1, 5} := by
  sorry

end intersection_complement_equality_l673_67383


namespace reflection_over_y_eq_neg_x_l673_67386

/-- Reflects a point (x, y) over the line y = -x -/
def reflect_over_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), -(p.1))

/-- The original point -/
def original_point : ℝ × ℝ := (7, -3)

/-- The expected reflected point -/
def expected_reflected_point : ℝ × ℝ := (3, -7)

theorem reflection_over_y_eq_neg_x :
  reflect_over_y_eq_neg_x original_point = expected_reflected_point := by
  sorry

end reflection_over_y_eq_neg_x_l673_67386


namespace solve_for_c_l673_67344

theorem solve_for_c (x y c : ℝ) (h1 : x / (2 * y) = 5 / 2) (h2 : (7 * x + 4 * y) / c = 13) :
  c = 3 * y := by
sorry

end solve_for_c_l673_67344


namespace uniform_pickup_ways_l673_67338

def number_of_students : ℕ := 5
def correct_picks : ℕ := 2

theorem uniform_pickup_ways :
  (number_of_students.choose correct_picks) * 2 = 20 := by
  sorry

end uniform_pickup_ways_l673_67338


namespace train_crossing_time_l673_67375

/-- Proves that a train with given length and speed takes the calculated time to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 360 →
  train_speed_kmh = 216 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 6 := by
  sorry

#check train_crossing_time

end train_crossing_time_l673_67375


namespace work_completion_equality_first_group_size_l673_67309

/-- The number of days it takes the first group to complete the work -/
def days_first_group : ℕ := 20

/-- The number of men in the second group -/
def men_second_group : ℕ := 12

/-- The number of days it takes the second group to complete the work -/
def days_second_group : ℕ := 30

/-- The number of men in the first group -/
def men_first_group : ℕ := 18

theorem work_completion_equality :
  men_first_group * days_first_group = men_second_group * days_second_group :=
by sorry

theorem first_group_size :
  men_first_group = (men_second_group * days_second_group) / days_first_group :=
by sorry

end work_completion_equality_first_group_size_l673_67309


namespace base5_1234_equals_194_l673_67329

def base5_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem base5_1234_equals_194 :
  base5_to_decimal [4, 3, 2, 1] = 194 := by
  sorry

end base5_1234_equals_194_l673_67329


namespace rectangle_perimeter_l673_67351

/-- Given three rectangles with the following properties:
    Rectangle 1: length = 16 cm, width = 8 cm
    Rectangle 2: length = 1/2 of Rectangle 1's length, width = 1/2 of Rectangle 1's width
    Rectangle 3: length = 1/2 of Rectangle 2's length, width = 1/2 of Rectangle 2's width
    The perimeter of the figure formed by these rectangles is 60 cm. -/
theorem rectangle_perimeter (rect1_length rect1_width : ℝ) 
  (rect2_length rect2_width : ℝ) (rect3_length rect3_width : ℝ) :
  rect1_length = 16 ∧ 
  rect1_width = 8 ∧
  rect2_length = rect1_length / 2 ∧
  rect2_width = rect1_width / 2 ∧
  rect3_length = rect2_length / 2 ∧
  rect3_width = rect2_width / 2 →
  2 * (rect1_length + rect1_width + rect2_width + rect3_width) = 60 := by
  sorry

end rectangle_perimeter_l673_67351
