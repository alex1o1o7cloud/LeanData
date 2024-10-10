import Mathlib

namespace inverse_proportion_l538_53872

/-- Given that the product of x and y is constant, and x = 30 when y = 10,
    prove that x = 60 when y = 5 and the relationship doesn't hold for x = 48 and y = 15 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 30 * 10 = k) :
  (5 * 60 = k) ∧ ¬(48 * 15 = k) := by
  sorry

end inverse_proportion_l538_53872


namespace range_of_a_l538_53842

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : ∀ x, -1 < x ∧ x < 1 → ∃ y, f x = y)  -- f is defined on (-1, 1)
  (h2 : ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y)  -- f is decreasing on (-1, 1)
  (h3 : f (a - 1) > f (2 * a))  -- f(a-1) > f(2a)
  (h4 : -1 < a - 1 ∧ a - 1 < 1)  -- -1 < a-1 < 1
  (h5 : -1 < 2 * a ∧ 2 * a < 1)  -- -1 < 2a < 1
  : 0 < a ∧ a < 1/2 := by
  sorry

end range_of_a_l538_53842


namespace function_decreasing_implies_a_range_a_in_range_l538_53885

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

-- State the theorem
theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  0 < a ∧ a ≤ 1/4 := by
  sorry

-- Define the set of possible values for a
def a_range : Set ℝ := { a | 0 < a ∧ a ≤ 1/4 }

-- State the final theorem
theorem a_in_range :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ a_range := by
  sorry

end function_decreasing_implies_a_range_a_in_range_l538_53885


namespace sum_reciprocal_equals_two_max_weighted_sum_reciprocal_l538_53860

-- Define the variables and conditions
variable (a b x y : ℝ)
variable (ha : a > 1)
variable (hb : b > 1)
variable (hx : a^x = 2)
variable (hy : b^y = 2)

-- Theorem 1
theorem sum_reciprocal_equals_two (hab : a * b = 4) :
  1 / x + 1 / y = 2 := by sorry

-- Theorem 2
theorem max_weighted_sum_reciprocal (hab : a^2 + b = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b x y : ℝ), a > 1 → b > 1 → a^x = 2 → b^y = 2 → a^2 + b = 8 →
    2 / x + 1 / y ≤ m := by sorry

end sum_reciprocal_equals_two_max_weighted_sum_reciprocal_l538_53860


namespace pattern_equality_l538_53830

theorem pattern_equality (n : ℕ) : n * (n + 2) + 1 = (n + 1)^2 := by
  sorry

end pattern_equality_l538_53830


namespace semicircle_function_max_point_max_value_max_point_trig_l538_53892

noncomputable section

variables (R : ℝ) (x : ℝ)

def semicircle_point (R x : ℝ) : ℝ × ℝ :=
  (x, Real.sqrt (4 * R^2 - x^2))

def y (R x : ℝ) : ℝ :=
  2 * x + 3 * (2 * R - x^2 / (2 * R))

theorem semicircle_function (R : ℝ) (h : R > 0) :
  ∀ x, 0 ≤ x ∧ x ≤ 2 * R →
  y R x = -3 / (2 * R) * x^2 + 2 * x + 6 * R :=
sorry

theorem max_point (R : ℝ) (h : R > 0) :
  ∃ x_max, x_max = 2 * R / 3 ∧
  ∀ x, 0 ≤ x ∧ x ≤ 2 * R → y R x ≤ y R x_max :=
sorry

theorem max_value (R : ℝ) (h : R > 0) :
  y R (2 * R / 3) = 20 * R / 3 :=
sorry

theorem max_point_trig (R : ℝ) (h : R > 0) :
  let x_max := 2 * R / 3
  let α := Real.arccos (1 - x_max^2 / (2 * R^2))
  Real.cos α = 7 / 9 ∧ Real.sin α = 4 * Real.sqrt 2 / 9 :=
sorry

end semicircle_function_max_point_max_value_max_point_trig_l538_53892


namespace difference_61st_terms_arithmetic_sequences_l538_53802

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem difference_61st_terms_arithmetic_sequences :
  let C := arithmetic_sequence 45 15
  let D := arithmetic_sequence 45 (-15)
  |C 61 - D 61| = 1800 := by
sorry

end difference_61st_terms_arithmetic_sequences_l538_53802


namespace M_equals_divisors_of_151_l538_53887

def M : Set Nat :=
  {d | ∃ m n : Nat, d = Nat.gcd (2*n + 3*m + 13) (Nat.gcd (3*n + 5*m + 1) (6*n + 6*m - 1))}

theorem M_equals_divisors_of_151 : M = {d : Nat | d > 0 ∧ d ∣ 151} := by
  sorry

end M_equals_divisors_of_151_l538_53887


namespace partial_fraction_decomposition_l538_53808

theorem partial_fraction_decomposition (x : ℝ) (h2 : x ≠ 2) (h3 : x ≠ 3) (h4 : x ≠ 4) :
  (x^2 - 10*x + 16) / ((x - 2) * (x - 3) * (x - 4)) =
  2 / (x - 2) + 5 / (x - 3) + 0 / (x - 4) := by
  sorry

end partial_fraction_decomposition_l538_53808


namespace shortest_side_length_l538_53822

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, the shortest side has length 1. -/
theorem shortest_side_length (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  a * c + c^2 = b^2 - a^2 →  -- given condition
  b = Real.sqrt 7 →  -- longest side is √7
  Real.sin C = 2 * Real.sin A →  -- given condition
  b ≥ a ∧ b ≥ c →  -- b is the longest side
  min a c = 1 :=  -- the shortest side has length 1
by sorry

end shortest_side_length_l538_53822


namespace arithmetic_sequence_general_term_l538_53848

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1)))
  (h2 : a 4 - a 2 = 4)
  (h3 : S 3 = 9) :
  ∀ n, a n = 2 * n - 1 := by
sorry

end arithmetic_sequence_general_term_l538_53848


namespace reciprocal_sum_theorem_l538_53894

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 5 * x * y) : 1 / x + 1 / y = 5 := by
  sorry

end reciprocal_sum_theorem_l538_53894


namespace solution_correctness_l538_53832

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + Real.sqrt (x + 2*y) - 2*y = 7/2
def equation2 (x y : ℝ) : Prop := x^2 + x + 2*y - 4*y^2 = 27/2

-- State the theorem
theorem solution_correctness : 
  equation1 (19/4) (17/8) ∧ equation2 (19/4) (17/8) := by sorry

end solution_correctness_l538_53832


namespace lcm_count_l538_53838

theorem lcm_count : 
  ∃! (n : ℕ), n > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ k ∈ S, k > 0 ∧ Nat.lcm (9^9) (Nat.lcm (12^12) k) = 18^18) ∧
    (∀ k ∉ S, k > 0 → Nat.lcm (9^9) (Nat.lcm (12^12) k) ≠ 18^18)) :=
sorry

end lcm_count_l538_53838


namespace ariella_meetings_percentage_l538_53824

theorem ariella_meetings_percentage : 
  let work_day_hours : ℝ := 8
  let first_meeting_minutes : ℝ := 60
  let second_meeting_factor : ℝ := 1.5
  let work_day_minutes : ℝ := work_day_hours * 60
  let second_meeting_minutes : ℝ := second_meeting_factor * first_meeting_minutes
  let total_meeting_minutes : ℝ := first_meeting_minutes + second_meeting_minutes
  let meeting_percentage : ℝ := (total_meeting_minutes / work_day_minutes) * 100
  meeting_percentage = 31.25 := by sorry

end ariella_meetings_percentage_l538_53824


namespace min_value_theorem_min_value_attained_l538_53899

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 6) : 
  (9 / x + 16 / y + 25 / z) ≥ 24 := by
  sorry

theorem min_value_attained (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 6) : 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 6 ∧ 
  (9 / x₀ + 16 / y₀ + 25 / z₀) = 24 := by
  sorry

end min_value_theorem_min_value_attained_l538_53899


namespace probability_x_greater_than_3y_l538_53882

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  y_min : ℝ
  x_max : ℝ
  y_max : ℝ

/-- A point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is inside a rectangle --/
def Point.insideRectangle (p : Point) (r : Rectangle) : Prop :=
  r.x_min ≤ p.x ∧ p.x ≤ r.x_max ∧ r.y_min ≤ p.y ∧ p.y ≤ r.y_max

/-- The probability of an event occurring for a point randomly picked from a rectangle --/
def probability (r : Rectangle) (event : Point → Prop) : ℝ :=
  sorry

/-- The specific rectangle in the problem --/
def problemRectangle : Rectangle :=
  { x_min := 0, y_min := 0, x_max := 3000, y_max := 3000 }

/-- The event x > 3y --/
def xGreaterThan3y (p : Point) : Prop :=
  p.x > 3 * p.y

theorem probability_x_greater_than_3y :
  probability problemRectangle xGreaterThan3y = 1/6 :=
sorry

end probability_x_greater_than_3y_l538_53882


namespace unique_solution_exponential_equation_l538_53854

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ) ^ (x^2 - 6*x + 8) = (1/2 : ℝ) := by
  sorry

end unique_solution_exponential_equation_l538_53854


namespace min_value_a_l538_53871

theorem min_value_a (h : ∀ x y : ℝ, x > 0 → y > 0 → x + y ≥ 9) :
  ∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, x > 0 → x + a ≥ 9) ∧
  (∀ b : ℝ, b > 0 → (∀ x : ℝ, x > 0 → x + b ≥ 9) → b ≥ a) ∧
  a = 4 :=
sorry

end min_value_a_l538_53871


namespace asian_math_competition_l538_53835

theorem asian_math_competition (total_countries : ℕ) 
  (solved_1 solved_1_2 solved_1_3 solved_1_4 solved_all : ℕ) :
  total_countries = 846 →
  solved_1 = 235 →
  solved_1_2 = 59 →
  solved_1_3 = 29 →
  solved_1_4 = 15 →
  solved_all = 3 →
  ∃ (country : ℕ), country ≤ total_countries ∧ 
    ∃ (students : ℕ), students ≥ 4 ∧
      students ≤ (solved_1 - solved_1_2 - solved_1_3 - solved_1_4 + solved_all) :=
by sorry

end asian_math_competition_l538_53835


namespace crucian_carp_cultivation_optimal_l538_53850

/-- Represents the seafood wholesaler's crucian carp cultivation problem -/
structure CrucianCarpProblem where
  initialWeight : ℝ  -- Initial weight of crucian carp in kg
  initialPrice : ℝ   -- Initial price per kg in yuan
  priceIncrease : ℝ  -- Daily price increase per kg in yuan
  maxDays : ℕ        -- Maximum culture period in days
  dailyLoss : ℝ      -- Daily weight loss due to oxygen deficiency in kg
  lossPrice : ℝ      -- Price of oxygen-deficient carp per kg in yuan
  dailyExpense : ℝ   -- Daily expenses during culture in yuan

/-- Calculates the profit for a given number of culture days -/
def profit (p : CrucianCarpProblem) (days : ℝ) : ℝ :=
  p.dailyLoss * days * (p.lossPrice - p.initialPrice) +
  (p.initialWeight - p.dailyLoss * days) * (p.initialPrice + p.priceIncrease * days) -
  p.initialWeight * p.initialPrice -
  p.dailyExpense * days

/-- The main theorem to be proved -/
theorem crucian_carp_cultivation_optimal (p : CrucianCarpProblem)
  (h1 : p.initialWeight = 1000)
  (h2 : p.initialPrice = 10)
  (h3 : p.priceIncrease = 1)
  (h4 : p.maxDays = 20)
  (h5 : p.dailyLoss = 10)
  (h6 : p.lossPrice = 5)
  (h7 : p.dailyExpense = 450) :
  (∃ x : ℝ, x ≤ p.maxDays ∧ profit p x = 8500 ∧ x = 10) ∧
  (∀ x : ℝ, x ≤ p.maxDays → profit p x ≤ 6000) ∧
  (∃ x : ℝ, x ≤ p.maxDays ∧ profit p x = 6000) := by
  sorry


end crucian_carp_cultivation_optimal_l538_53850


namespace multiples_of_four_between_100_and_350_l538_53819

theorem multiples_of_four_between_100_and_350 : 
  (Finset.filter (fun n => n % 4 = 0) (Finset.range 350 \ Finset.range 100)).card = 62 := by
  sorry

end multiples_of_four_between_100_and_350_l538_53819


namespace equation_solution_l538_53851

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (2*x + 1)*(3*x + 1)*(5*x + 1)*(30*x + 1)
  ∀ x : ℝ, f x = 10 ↔ x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15 := by
  sorry

end equation_solution_l538_53851


namespace first_nonzero_digit_after_decimal_1_149_l538_53879

theorem first_nonzero_digit_after_decimal_1_149 : ∃ (n : ℕ) (d : ℕ),
  (1 : ℚ) / 149 = (n : ℚ) / 10^(d + 1) + (7 : ℚ) / 10^(d + 2) + (r : ℚ)
  ∧ 0 ≤ r
  ∧ r < 1 / 10^(d + 2)
  ∧ n < 10^(d + 1) :=
sorry

end first_nonzero_digit_after_decimal_1_149_l538_53879


namespace equation_solution_l538_53844

theorem equation_solution (x : ℝ) : x > 0 → (5 * x^(1/4) - 3 * (x / x^(3/4)) = 10 + x^(1/4)) ↔ x = 10000 := by
  sorry

end equation_solution_l538_53844


namespace hilt_pies_theorem_l538_53864

/-- The total number of pies Mrs. Hilt needs to bake -/
def total_pies (pecan_pies apple_pies : ℝ) (factor : ℝ) : ℝ :=
  (pecan_pies + apple_pies) * factor

/-- Theorem: Given the initial number of pecan pies (16.0) and apple pies (14.0),
    and a multiplication factor (5.0), the total number of pies Mrs. Hilt
    needs to bake is 150.0. -/
theorem hilt_pies_theorem :
  total_pies 16.0 14.0 5.0 = 150.0 := by
  sorry

end hilt_pies_theorem_l538_53864


namespace quadratic_inequality_l538_53855

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots -1 and 4, and a < 0,
    prove that ax^2 + bx + c < 0 when x < -1 or x > 4 -/
theorem quadratic_inequality (a b c : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = -1 ∨ x = 4) :
  ∀ x, a * x^2 + b * x + c < 0 ↔ x < -1 ∨ x > 4 := by
  sorry

end quadratic_inequality_l538_53855


namespace cyclist_pedestrian_speed_ratio_l538_53847

/-- Represents the speed of a person -/
structure Speed :=
  (value : ℝ)

/-- Represents a point in time -/
structure Time :=
  (hours : ℝ)

/-- Represents a distance between two points -/
structure Distance :=
  (value : ℝ)

/-- The problem setup -/
structure ProblemSetup :=
  (pedestrian_start : Time)
  (cyclist_start : Time)
  (meetup_time : Time)
  (cyclist_return : Time)
  (final_meetup : Time)
  (distance_AB : Distance)

/-- The theorem to be proved -/
theorem cyclist_pedestrian_speed_ratio 
  (setup : ProblemSetup)
  (pedestrian_speed : Speed)
  (cyclist_speed : Speed)
  (h1 : setup.pedestrian_start.hours = 12)
  (h2 : setup.meetup_time.hours = 13)
  (h3 : setup.final_meetup.hours = 16)
  (h4 : setup.pedestrian_start.hours < setup.cyclist_start.hours)
  (h5 : setup.cyclist_start.hours < setup.meetup_time.hours) :
  cyclist_speed.value / pedestrian_speed.value = 5 / 3 := by
  sorry

end cyclist_pedestrian_speed_ratio_l538_53847


namespace triangle_side_lengths_l538_53868

theorem triangle_side_lengths 
  (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = c + 2) 
  (h3 : Real.sin (Real.arcsin (Real.sqrt 3 / 2)) = Real.sqrt 3 / 2) : 
  a = 7 ∧ b = 5 ∧ c = 3 := by
  sorry

#check triangle_side_lengths

end triangle_side_lengths_l538_53868


namespace water_bottles_per_day_l538_53809

theorem water_bottles_per_day 
  (total_bottles : ℕ) 
  (total_days : ℕ) 
  (h1 : total_bottles = 28) 
  (h2 : total_days = 4) 
  (h3 : total_days ≠ 0) : 
  total_bottles / total_days = 7 := by
sorry

end water_bottles_per_day_l538_53809


namespace reflection_over_x_axis_l538_53874

/-- Reflects a point over the x-axis -/
def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The reflection of (-4, 3) over the x-axis is (-4, -3) -/
theorem reflection_over_x_axis :
  reflect_over_x_axis (-4, 3) = (-4, -3) := by
  sorry

end reflection_over_x_axis_l538_53874


namespace xyz_sum_l538_53898

theorem xyz_sum (x y z : ℝ) (eq1 : 2*x + 3*y + 4*z = 10) (eq2 : y + 2*z = 2) : 
  x + y + z = 4 := by
sorry

end xyz_sum_l538_53898


namespace expression_evaluation_l538_53858

theorem expression_evaluation :
  let x : ℤ := -2
  (x - 2)^2 - 4*x*(x - 1) + (2*x + 1)*(2*x - 1) = 7 := by
  sorry

end expression_evaluation_l538_53858


namespace product_divisible_by_5_probability_l538_53823

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The probability that the product of the numbers rolled is divisible by 5 -/
def prob_divisible_by_5 : ℚ := 144495 / 262144

/-- Theorem stating the probability of the product being divisible by 5 -/
theorem product_divisible_by_5_probability :
  (1 : ℚ) - (1 - 1 / num_sides) ^ num_dice = prob_divisible_by_5 := by
  sorry

end product_divisible_by_5_probability_l538_53823


namespace xiao_ming_brother_age_l538_53859

def is_multiple_of_19 (year : ℕ) : Prop := ∃ k : ℕ, year = 19 * k

def has_repeated_digits (year : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ (∃ i j : ℕ, i ≠ j ∧ (year / 10^i) % 10 = d ∧ (year / 10^j) % 10 = d)

def first_non_repeating_year (birth_year : ℕ) (target_year : ℕ) : Prop :=
  ¬(has_repeated_digits target_year) ∧
  ∀ y : ℕ, birth_year ≤ y ∧ y < target_year → has_repeated_digits y

theorem xiao_ming_brother_age :
  ∀ birth_year : ℕ,
    is_multiple_of_19 birth_year →
    first_non_repeating_year birth_year 2013 →
    2013 - birth_year = 18 :=
by sorry

end xiao_ming_brother_age_l538_53859


namespace william_land_percentage_l538_53891

def total_tax : ℝ := 3840
def tax_percentage : ℝ := 0.75
def william_tax : ℝ := 480

theorem william_land_percentage :
  let total_taxable_income := total_tax / tax_percentage
  let william_percentage := (william_tax / total_taxable_income) * 100
  william_percentage = 9.375 := by sorry

end william_land_percentage_l538_53891


namespace spaceship_reach_boundary_l538_53800

/-- A path in 3D space --/
structure Path3D where
  points : List (ℝ × ℝ × ℝ)

/-- Distance of a point from a plane --/
def distanceFromPlane (point : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ → ℝ) : ℝ :=
  sorry

/-- Length of a path --/
def pathLength (path : Path3D) : ℝ :=
  sorry

/-- Check if a path reaches the boundary plane --/
def reachesBoundary (path : Path3D) (boundaryPlane : ℝ × ℝ × ℝ → ℝ) : Prop :=
  sorry

/-- The main theorem --/
theorem spaceship_reach_boundary (a : ℝ) (startPoint : ℝ × ℝ × ℝ) (boundaryPlane : ℝ × ℝ × ℝ → ℝ) 
    (h : distanceFromPlane startPoint boundaryPlane = a) :
    ∃ (path : Path3D), pathLength path ≤ 14 * a ∧ reachesBoundary path boundaryPlane :=
  sorry

end spaceship_reach_boundary_l538_53800


namespace cube_surface_area_l538_53893

/-- Given a cube with the sum of edge lengths equal to 36 and space diagonal length equal to 3√3,
    the total surface area is 54. -/
theorem cube_surface_area (s : ℝ) 
  (h1 : 12 * s = 36) 
  (h2 : s * Real.sqrt 3 = 3 * Real.sqrt 3) : 
  6 * s^2 = 54 := by
  sorry

end cube_surface_area_l538_53893


namespace exam_students_count_l538_53865

theorem exam_students_count (total_average : ℝ) (excluded_average : ℝ) (new_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 40) 
  (h3 : new_average = 90) (h4 : excluded_count = 5) : 
  ∃ (n : ℕ), n = 25 ∧ 
    (n : ℝ) * total_average = 
      ((n - excluded_count) : ℝ) * new_average + (excluded_count : ℝ) * excluded_average :=
by
  sorry

end exam_students_count_l538_53865


namespace trig_sum_equality_l538_53863

theorem trig_sum_equality : 
  3.423 * Real.sin (10 * π / 180) + Real.sin (20 * π / 180) + Real.sin (30 * π / 180) + 
  Real.sin (40 * π / 180) + Real.sin (50 * π / 180) = 
  Real.sin (25 * π / 180) / (2 * Real.sin (5 * π / 180)) := by
  sorry

end trig_sum_equality_l538_53863


namespace wenzhou_population_scientific_notation_l538_53849

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) (h : x > 0) : ScientificNotation :=
  sorry

theorem wenzhou_population_scientific_notation :
  toScientificNotation 9570000 (by norm_num) =
    ScientificNotation.mk 9.57 6 (by norm_num) :=
  sorry

end wenzhou_population_scientific_notation_l538_53849


namespace goose_eggs_count_l538_53813

/-- The number of goose eggs laid at a pond -/
def total_eggs : ℕ := 400

/-- The fraction of eggs that hatched -/
def hatch_rate : ℚ := 1/2

/-- The fraction of hatched geese that survived the first month -/
def first_month_survival_rate : ℚ := 3/4

/-- The fraction of geese that survived the first month but did not survive the first year -/
def first_year_death_rate : ℚ := 3/5

/-- The number of geese that survived the first year -/
def survived_first_year : ℕ := 120

theorem goose_eggs_count :
  (total_eggs : ℚ) * hatch_rate * first_month_survival_rate * (1 - first_year_death_rate) = survived_first_year :=
sorry

end goose_eggs_count_l538_53813


namespace quadratic_function_properties_l538_53869

noncomputable section

variables (a b c : ℝ) (f : ℝ → ℝ)

def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_properties
  (h1 : quadratic_function f)
  (h2 : f 0 = 2)
  (h3 : ∀ x, f (x + 1) - f x = 2 * x - 1) :
  (∀ x, f x = x^2 - 2*x + 2) ∧
  (∀ x, x > 1 → (deriv f) x > 0) ∧
  (∀ x, x < 1 → (deriv f) x < 0) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 5) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 1) :=
by
  sorry

end

end quadratic_function_properties_l538_53869


namespace final_statue_weight_approx_l538_53861

/-- The weight of the final statue given the initial weights and removal percentages --/
def final_statue_weight (initial_marble : ℝ) (initial_granite : ℝ) 
  (marble_removal1 : ℝ) (marble_removal2 : ℝ) (marble_removal3 : ℝ) 
  (granite_removal1 : ℝ) (granite_removal2 : ℝ) 
  (marble_removal_final : ℝ) (granite_removal_final : ℝ) : ℝ :=
  let remaining_marble1 := initial_marble * (1 - marble_removal1)
  let remaining_marble2 := remaining_marble1 * (1 - marble_removal2)
  let remaining_marble3 := remaining_marble2 * (1 - marble_removal3)
  let final_marble := remaining_marble3 * (1 - marble_removal_final)
  
  let remaining_granite1 := initial_granite * (1 - granite_removal1)
  let remaining_granite2 := remaining_granite1 * (1 - granite_removal2)
  let final_granite := remaining_granite2 * (1 - granite_removal_final)
  
  final_marble + final_granite

/-- The final weight of the statue is approximately 119.0826 kg --/
theorem final_statue_weight_approx :
  ∃ ε > 0, ε < 0.0001 ∧ 
  |final_statue_weight 225 65 0.32 0.22 0.15 0.40 0.25 0.10 0.05 - 119.0826| < ε :=
sorry

end final_statue_weight_approx_l538_53861


namespace condition_relationship_l538_53837

theorem condition_relationship : 
  ∀ x : ℝ, (x > 3 → x > 2) ∧ ¬(x > 2 → x > 3) :=
by sorry

end condition_relationship_l538_53837


namespace f_of_5_equals_15_l538_53883

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem f_of_5_equals_15 : f 5 = 15 := by sorry

end f_of_5_equals_15_l538_53883


namespace largest_n_binomial_sum_l538_53846

theorem largest_n_binomial_sum : 
  (∃ n : ℕ, (Nat.choose 9 4 + Nat.choose 9 5 = Nat.choose 10 n) ∧ 
   (∀ m : ℕ, m > n → Nat.choose 9 4 + Nat.choose 9 5 ≠ Nat.choose 10 m)) → 
  (∃ n : ℕ, n = 5 ∧ (Nat.choose 9 4 + Nat.choose 9 5 = Nat.choose 10 n) ∧ 
   (∀ m : ℕ, m > n → Nat.choose 9 4 + Nat.choose 9 5 ≠ Nat.choose 10 m)) :=
by sorry

end largest_n_binomial_sum_l538_53846


namespace parabola_point_relationship_l538_53897

/-- Parabola function -/
def f (x : ℝ) : ℝ := -(x + 1)^2 + 5

/-- Point A on the parabola -/
def A : ℝ × ℝ := (-2, f (-2))

/-- Point B on the parabola -/
def B : ℝ × ℝ := (1, f 1)

/-- Point C on the parabola -/
def C : ℝ × ℝ := (2, f 2)

/-- Theorem stating the relationship between y-coordinates of A, B, and C -/
theorem parabola_point_relationship : A.2 > B.2 ∧ B.2 > C.2 := by
  sorry

end parabola_point_relationship_l538_53897


namespace a_zero_sufficient_not_necessary_l538_53888

def M (a : ℝ) : Set ℝ := {1, a}
def N : Set ℝ := {-1, 0, 1}

theorem a_zero_sufficient_not_necessary :
  (∀ a : ℝ, a = 0 → M a ⊆ N) ∧
  (∃ a : ℝ, a ≠ 0 ∧ M a ⊆ N) :=
by sorry

end a_zero_sufficient_not_necessary_l538_53888


namespace convex_polygon_30_sides_diagonals_l538_53896

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 30 sides has 202 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 202 := by
  sorry

end convex_polygon_30_sides_diagonals_l538_53896


namespace kelly_gave_away_64_games_l538_53803

/-- The number of games Kelly gave away -/
def games_given_away (initial_games final_games : ℕ) : ℕ :=
  initial_games - final_games

/-- Theorem: Kelly gave away 64 games -/
theorem kelly_gave_away_64_games :
  games_given_away 106 42 = 64 := by
  sorry

end kelly_gave_away_64_games_l538_53803


namespace paper_clip_distribution_l538_53812

theorem paper_clip_distribution (total_clips : ℕ) (num_boxes : ℕ) (clips_per_box : ℕ) : 
  total_clips = 81 → num_boxes = 9 → clips_per_box = total_clips / num_boxes → clips_per_box = 9 := by
  sorry

end paper_clip_distribution_l538_53812


namespace linked_rings_length_l538_53839

/-- Represents a sequence of linked rings with specific properties. -/
structure LinkedRings where
  ringThickness : ℝ
  topRingDiameter : ℝ
  bottomRingDiameter : ℝ
  diameterDecrease : ℝ

/-- Calculates the total length of the linked rings. -/
def totalLength (rings : LinkedRings) : ℝ :=
  sorry

/-- Theorem stating that the total length of the linked rings with given properties is 342 cm. -/
theorem linked_rings_length :
  let rings : LinkedRings := {
    ringThickness := 2,
    topRingDiameter := 40,
    bottomRingDiameter := 4,
    diameterDecrease := 2
  }
  totalLength rings = 342 := by sorry

end linked_rings_length_l538_53839


namespace profit_percentage_calculation_l538_53806

/-- 
Given an article with a selling price of 600 and a cost price of 375,
prove that the profit percentage is 60%.
-/
theorem profit_percentage_calculation (selling_price cost_price : ℝ) 
  (h1 : selling_price = 600)
  (h2 : cost_price = 375) : 
  (selling_price - cost_price) / cost_price * 100 = 60 := by
  sorry

end profit_percentage_calculation_l538_53806


namespace gcd_of_powers_minus_one_l538_53884

theorem gcd_of_powers_minus_one : Nat.gcd (4^8 - 1) (8^12 - 1) = 15 := by
  sorry

end gcd_of_powers_minus_one_l538_53884


namespace paul_initial_pens_l538_53825

/-- The number of pens Paul sold in the garage sale. -/
def pens_sold : ℕ := 92

/-- The number of pens Paul had left after the garage sale. -/
def pens_left : ℕ := 14

/-- The initial number of pens Paul had. -/
def initial_pens : ℕ := pens_sold + pens_left

theorem paul_initial_pens : initial_pens = 106 := by
  sorry

end paul_initial_pens_l538_53825


namespace smallest_factor_of_32_not_8_l538_53878

theorem smallest_factor_of_32_not_8 : ∃ n : ℕ, n = 16 ∧ 
  (32 % n = 0) ∧ (8 % n ≠ 0) ∧ 
  (∀ m : ℕ, m < n → (32 % m = 0 → 8 % m = 0)) :=
by sorry

end smallest_factor_of_32_not_8_l538_53878


namespace min_value_of_f_l538_53841

open Real

noncomputable def f (x : ℝ) : ℝ := (log x)^2 / x

theorem min_value_of_f :
  ∀ x > 0, f x ≥ 0 ∧ ∃ x₀ > 0, f x₀ = 0 :=
by sorry

end min_value_of_f_l538_53841


namespace quadratic_root_implies_m_value_l538_53828

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - m * x + 3 = 0 ∧ x = 3) → m = 7 := by
  sorry

end quadratic_root_implies_m_value_l538_53828


namespace chocolates_in_cost_price_l538_53804

/-- The number of chocolates in the cost price -/
def n : ℕ := sorry

/-- The cost price of one chocolate -/
def C : ℝ := sorry

/-- The selling price of one chocolate -/
def S : ℝ := sorry

/-- The cost price of n chocolates equals the selling price of 16 chocolates -/
axiom cost_price_eq_selling_price : n * C = 16 * S

/-- The gain percent is 50% -/
axiom gain_percent : S = 1.5 * C

theorem chocolates_in_cost_price : n = 24 := by sorry

end chocolates_in_cost_price_l538_53804


namespace sector_area_l538_53836

theorem sector_area (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 2) :
  (1 / 2) * θ * r^2 = (2 * π) / 3 := by
  sorry

end sector_area_l538_53836


namespace expand_and_simplify_expression_l538_53889

theorem expand_and_simplify_expression (x : ℝ) : 
  6 * (x - 7) * (2 * x + 15) + (3 * x - 4) * (x + 5) = 15 * x^2 + 17 * x - 650 := by
  sorry

end expand_and_simplify_expression_l538_53889


namespace intersection_complement_equality_l538_53886

open Set

def A : Set ℝ := {x | x^2 - 1 ≤ 0}
def B : Set ℝ := {x | x < 1}

theorem intersection_complement_equality : A ∩ (𝒰 \ B) = {x | x = 1} := by sorry

end intersection_complement_equality_l538_53886


namespace f_even_iff_a_eq_zero_l538_53827

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^2 + ax for some a ∈ ℝ -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x

/-- Theorem: f is an even function if and only if a = 0 -/
theorem f_even_iff_a_eq_zero (a : ℝ) :
  IsEven (f a) ↔ a = 0 := by sorry

end f_even_iff_a_eq_zero_l538_53827


namespace cash_percentage_is_twenty_percent_l538_53840

def raw_materials : ℝ := 35000
def machinery : ℝ := 40000
def total_amount : ℝ := 93750

theorem cash_percentage_is_twenty_percent :
  (total_amount - (raw_materials + machinery)) / total_amount * 100 = 20 := by
  sorry

end cash_percentage_is_twenty_percent_l538_53840


namespace BEE_has_largest_value_l538_53834

def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | _   => 0

def word_value (w : String) : ℕ :=
  w.toList.map letter_value |>.sum

theorem BEE_has_largest_value :
  let BAD := "BAD"
  let CAB := "CAB"
  let DAD := "DAD"
  let BEE := "BEE"
  let BED := "BED"
  (word_value BEE > word_value BAD) ∧
  (word_value BEE > word_value CAB) ∧
  (word_value BEE > word_value DAD) ∧
  (word_value BEE > word_value BED) := by
  sorry

end BEE_has_largest_value_l538_53834


namespace root_sum_theorem_l538_53875

theorem root_sum_theorem (a b c : ℝ) : 
  (a^3 - 6*a^2 + 8*a - 3 = 0) → 
  (b^3 - 6*b^2 + 8*b - 3 = 0) → 
  (c^3 - 6*c^2 + 8*c - 3 = 0) → 
  (a/(b*c + 2) + b/(a*c + 2) + c/(a*b + 2) = 0) := by
sorry

end root_sum_theorem_l538_53875


namespace k_range_theorem_l538_53820

/-- Proposition p: The equation represents an ellipse with foci on the y-axis -/
def p (k : ℝ) : Prop := 3 < k ∧ k < 9/2

/-- Proposition q: The equation represents a hyperbola with eccentricity e in (√3, 2) -/
def q (k : ℝ) : Prop := 4 < k ∧ k < 6

/-- The range of real values for k -/
def k_range (k : ℝ) : Prop := (3 < k ∧ k ≤ 4) ∨ (9/2 ≤ k ∧ k < 6)

theorem k_range_theorem (k : ℝ) : 
  (¬(p k ∧ q k) ∧ (p k ∨ q k)) → k_range k := by
  sorry

end k_range_theorem_l538_53820


namespace inverse_false_implies_negation_false_l538_53877

theorem inverse_false_implies_negation_false (p : Prop) :
  (¬p → False) → (¬p = False) := by
  sorry

end inverse_false_implies_negation_false_l538_53877


namespace no_valid_triples_l538_53895

theorem no_valid_triples : ¬∃ (a b c : ℤ), 
  (|a + b| + c = 23) ∧ 
  (a * b + |c| = 85) ∧ 
  (∃ k : ℤ, b = 3 * k) := by
sorry

end no_valid_triples_l538_53895


namespace A_minus_B_equals_1790_l538_53821

/-- Calculates the value of A based on the given groups -/
def calculate_A : ℕ := 1 * 1000 + 16 * 100 + 28 * 10

/-- Calculates the value of B based on the given jumps and interval -/
def calculate_B : ℕ := 355 + 3 * 245

/-- Proves that A - B equals 1790 -/
theorem A_minus_B_equals_1790 : calculate_A - calculate_B = 1790 := by
  sorry

end A_minus_B_equals_1790_l538_53821


namespace min_trees_triangular_plot_l538_53845

/-- Given a triangular plot with 5 trees planted on each side, 
    the minimum number of trees that can be planted is 12. -/
theorem min_trees_triangular_plot : 
  ∀ (trees_per_side : ℕ), 
  trees_per_side = 5 → 
  (∃ (min_trees : ℕ), 
    min_trees = 12 ∧ 
    ∀ (total_trees : ℕ), 
      (total_trees ≥ min_trees ∧ 
       ∃ (trees_on_edges : ℕ), 
         trees_on_edges = total_trees - 3 ∧ 
         trees_on_edges % 3 = 0 ∧ 
         trees_on_edges / 3 + 1 = trees_per_side)) :=
by sorry

end min_trees_triangular_plot_l538_53845


namespace bridge_length_calculation_l538_53831

/-- Calculate the length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 200 →
  train_speed_kmh = 60 →
  crossing_time = 45 →
  ∃ bridge_length : ℝ,
    (bridge_length ≥ 550) ∧ 
    (bridge_length ≤ 551) ∧
    (train_speed_kmh * 1000 / 3600 * crossing_time = train_length + bridge_length) :=
by sorry

end bridge_length_calculation_l538_53831


namespace calculate_expression_l538_53867

theorem calculate_expression : 8 * (2 / 16) * 32 - 10 = 22 := by
  sorry

end calculate_expression_l538_53867


namespace product_of_sequence_a_l538_53818

def sequence_a : ℕ → ℚ
  | 0 => 3/2
  | n + 1 => 3 + (sequence_a n - 2)^2

def infinite_product (f : ℕ → ℚ) : ℚ := sorry

theorem product_of_sequence_a :
  infinite_product sequence_a = 4/3 := by sorry

end product_of_sequence_a_l538_53818


namespace standing_arrangements_l538_53856

def number_of_people : ℕ := 5

-- Function to calculate the number of ways person A and B can stand next to each other
def ways_next_to_each_other (n : ℕ) : ℕ := sorry

-- Function to calculate the total number of ways n people can stand
def total_ways (n : ℕ) : ℕ := sorry

-- Function to calculate the number of ways person A and B can stand not next to each other
def ways_not_next_to_each_other (n : ℕ) : ℕ := sorry

theorem standing_arrangements :
  (ways_next_to_each_other number_of_people = 48) ∧
  (ways_not_next_to_each_other number_of_people = 72) := by sorry

end standing_arrangements_l538_53856


namespace least_integer_proof_l538_53880

/-- The least positive integer divisible by all numbers from 1 to 22 and 25 to 30 -/
def least_integer : ℕ := 1237834741500

/-- The set of divisors from 1 to 30, excluding 23 and 24 -/
def divisors : Set ℕ := {n : ℕ | n ∈ Finset.range 31 ∧ n ≠ 23 ∧ n ≠ 24}

theorem least_integer_proof :
  (∀ n ∈ divisors, least_integer % n = 0) ∧
  (∀ m : ℕ, m < least_integer →
    ∃ k ∈ divisors, m % k ≠ 0) :=
sorry

end least_integer_proof_l538_53880


namespace probabilities_ascending_order_order_matches_sequence_l538_53815

-- Define the probabilities of each event
def prob_event1 : ℚ := 2/3
def prob_event2 : ℚ := 1
def prob_event3 : ℚ := 1/3
def prob_event4 : ℚ := 1/2
def prob_event5 : ℚ := 0

-- Define a function to represent the correct order
def correct_order : Fin 5 → ℚ
  | 0 => prob_event5
  | 1 => prob_event3
  | 2 => prob_event4
  | 3 => prob_event1
  | 4 => prob_event2

-- Theorem stating that the probabilities are in ascending order
theorem probabilities_ascending_order :
  ∀ i j : Fin 5, i < j → correct_order i ≤ correct_order j :=
by sorry

-- Theorem stating that this order matches the given sequence (5) (3) (4) (1) (2)
theorem order_matches_sequence :
  correct_order 0 = prob_event5 ∧
  correct_order 1 = prob_event3 ∧
  correct_order 2 = prob_event4 ∧
  correct_order 3 = prob_event1 ∧
  correct_order 4 = prob_event2 :=
by sorry

end probabilities_ascending_order_order_matches_sequence_l538_53815


namespace insurance_cost_calculation_l538_53810

def apartment_cost : ℝ := 7000000
def loan_amount : ℝ := 4000000
def interest_rate : ℝ := 0.101
def property_insurance_rate : ℝ := 0.0009
def life_health_insurance_female : ℝ := 0.0017
def life_health_insurance_male : ℝ := 0.0019
def title_insurance_rate : ℝ := 0.0027
def svetlana_ratio : ℝ := 0.2
def dmitry_ratio : ℝ := 0.8

def total_insurance_cost : ℝ :=
  let total_loan := loan_amount * (1 + interest_rate)
  let property_insurance := total_loan * property_insurance_rate
  let title_insurance := total_loan * title_insurance_rate
  let svetlana_insurance := total_loan * svetlana_ratio * life_health_insurance_female
  let dmitry_insurance := total_loan * dmitry_ratio * life_health_insurance_male
  property_insurance + title_insurance + svetlana_insurance + dmitry_insurance

theorem insurance_cost_calculation :
  total_insurance_cost = 24045.84 := by sorry

end insurance_cost_calculation_l538_53810


namespace other_solution_of_quadratic_equation_l538_53890

theorem other_solution_of_quadratic_equation :
  let f (x : ℚ) := 77 * x^2 + 35 - (125 * x - 14)
  ∃ (x : ℚ), x ≠ 8/11 ∧ f x = 0 ∧ x = 1 := by
  sorry

end other_solution_of_quadratic_equation_l538_53890


namespace area_ratio_of_rectangles_l538_53817

/-- A structure representing a rectangle with width and length --/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- A structure representing a square composed of five rectangles --/
structure SquareOfRectangles where
  shaded : Rectangle
  unshaded : Rectangle
  total_width : ℝ
  total_height : ℝ

/-- The theorem stating the ratio of areas of shaded to unshaded rectangles --/
theorem area_ratio_of_rectangles (s : SquareOfRectangles) 
  (h1 : s.shaded.width + s.shaded.width + s.unshaded.width = s.total_width)
  (h2 : s.shaded.length = s.total_height)
  (h3 : 2 * (s.shaded.width + s.shaded.length) = 2 * (s.unshaded.width + s.unshaded.length))
  (h4 : s.shaded.width > 0)
  (h5 : s.shaded.length > 0)
  (h6 : s.unshaded.width > 0)
  (h7 : s.unshaded.length > 0) :
  (s.shaded.width * s.shaded.length) / (s.unshaded.width * s.unshaded.length) = 3 / 7 := by
  sorry

end area_ratio_of_rectangles_l538_53817


namespace factorization_equality_l538_53876

theorem factorization_equality (a b m : ℝ) : 
  a^2 * (m - 1) + b^2 * (1 - m) = (m - 1) * (a + b) * (a - b) := by
  sorry

end factorization_equality_l538_53876


namespace square_area_from_rectangle_perimeter_l538_53873

/-- If a square is cut into two identical rectangles, each with a perimeter of 24 cm,
    then the area of the original square is 64 cm². -/
theorem square_area_from_rectangle_perimeter :
  ∀ (side : ℝ), side > 0 →
  (2 * (side + side / 2) = 24) →
  side * side = 64 := by
sorry

end square_area_from_rectangle_perimeter_l538_53873


namespace abs_neg_2023_eq_2023_l538_53857

theorem abs_neg_2023_eq_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end abs_neg_2023_eq_2023_l538_53857


namespace gcd_product_equivalence_l538_53862

theorem gcd_product_equivalence (a m n : ℤ) : 
  Int.gcd a (m * n) = 1 ↔ Int.gcd a m = 1 ∧ Int.gcd a n = 1 := by
  sorry

end gcd_product_equivalence_l538_53862


namespace prove_vector_sum_with_scalar_multiple_l538_53843

def vector_sum_with_scalar_multiple : Prop :=
  let v1 : Fin 3 → ℝ := ![3, -2, 5]
  let v2 : Fin 3 → ℝ := ![-1, 4, -3]
  let result : Fin 3 → ℝ := ![1, 6, -1]
  v1 + 2 • v2 = result

theorem prove_vector_sum_with_scalar_multiple : vector_sum_with_scalar_multiple := by
  sorry

end prove_vector_sum_with_scalar_multiple_l538_53843


namespace expression_evaluation_l538_53866

theorem expression_evaluation (x : ℝ) (h : x = 6) :
  (1 + 2 / (x + 1)) * ((x^2 + x) / (x^2 - 9)) = 2 := by
  sorry

end expression_evaluation_l538_53866


namespace cara_arrangements_l538_53852

def num_friends : ℕ := 7

def arrangements (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

theorem cara_arrangements :
  arrangements num_friends = 6 :=
sorry

end cara_arrangements_l538_53852


namespace trajectory_equation_l538_53829

/-- The equation of the trajectory of point P in the xOy plane, given point A at (0,0,4) and |PA| = 5 -/
theorem trajectory_equation :
  ∀ (x y : ℝ),
  let P : ℝ × ℝ × ℝ := (x, y, 0)
  let A : ℝ × ℝ × ℝ := (0, 0, 4)
  (x^2 + y^2 + (0 - 4)^2 = 5^2) →
  (x^2 + y^2 = 9) :=
by
  sorry

end trajectory_equation_l538_53829


namespace geometric_series_sum_l538_53814

/-- The limiting sum of a geometric series with first term 6 and common ratio -2/5 is 30/7 -/
theorem geometric_series_sum : 
  let a : ℚ := 6
  let r : ℚ := -2/5
  let s : ℚ := a / (1 - r)
  s = 30/7 := by sorry

end geometric_series_sum_l538_53814


namespace infinite_primes_with_property_l538_53881

theorem infinite_primes_with_property : 
  ∃ (S : Set Nat), 
    (∀ p ∈ S, Nat.Prime p) ∧ 
    (Set.Infinite S) ∧ 
    (∀ p ∈ S, ∃ n : Nat, ¬(n ∣ (p - 1)) ∧ (p ∣ (Nat.factorial n + 1))) := by
  sorry

end infinite_primes_with_property_l538_53881


namespace quadratic_equation_solution_l538_53853

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x - 3
  ∃ x1 x2 : ℝ, x1 = 3 + 2*Real.sqrt 3 ∧ 
             x2 = 3 - 2*Real.sqrt 3 ∧ 
             f x1 = 0 ∧ f x2 = 0 ∧
             ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2 := by
  sorry

end quadratic_equation_solution_l538_53853


namespace quadratic_root_l538_53816

/-- Given a quadratic equation ax^2 + bx + c = 0 with coefficients defined in terms of p and q,
    if 1 is a root, then -2p / (p - 2) is the other root. -/
theorem quadratic_root (p q : ℝ) : 
  let a := p + q
  let b := p - q
  let c := p * q
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -2 * p / (p - 2)) :=
by sorry

end quadratic_root_l538_53816


namespace vertex_ordinate_zero_l538_53870

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The number of solutions to the equation (f x)^3 - f x = 0 -/
def numSolutions (f : ℝ → ℝ) : ℕ := sorry

/-- The ordinate (y-coordinate) of the vertex of a quadratic polynomial -/
def vertexOrdinate (f : ℝ → ℝ) : ℝ := sorry

/-- 
If f is a quadratic polynomial and (f x)^3 - f x = 0 has exactly three solutions,
then the ordinate of the vertex of f is 0
-/
theorem vertex_ordinate_zero 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (hf : f = QuadraticPolynomial a b c) 
  (h_solutions : numSolutions f = 3) : 
  vertexOrdinate f = 0 := by sorry

end vertex_ordinate_zero_l538_53870


namespace one_point_three_six_billion_scientific_notation_l538_53807

/-- Proves that 1.36 billion is equal to 1.36 × 10^9 -/
theorem one_point_three_six_billion_scientific_notation :
  (1.36 : ℝ) * (10 ^ 9 : ℝ) = 1.36e9 := by sorry

end one_point_three_six_billion_scientific_notation_l538_53807


namespace probability_white_ball_l538_53805

/-- The probability of drawing a white ball from a box with red and white balls -/
theorem probability_white_ball (red_balls white_balls : ℕ) :
  red_balls = 5 →
  white_balls = 4 →
  (white_balls : ℚ) / (red_balls + white_balls : ℚ) = 4 / 9 :=
by sorry

end probability_white_ball_l538_53805


namespace total_watching_time_l538_53801

-- Define the TV series data
def pride_and_prejudice_episodes : ℕ := 6
def pride_and_prejudice_duration : ℕ := 50
def breaking_bad_episodes : ℕ := 62
def breaking_bad_duration : ℕ := 47
def stranger_things_episodes : ℕ := 33
def stranger_things_duration : ℕ := 51

-- Calculate total watching time in minutes
def total_minutes : ℕ := 
  pride_and_prejudice_episodes * pride_and_prejudice_duration +
  breaking_bad_episodes * breaking_bad_duration +
  stranger_things_episodes * stranger_things_duration

-- Convert minutes to hours and round to nearest whole number
def total_hours : ℕ := (total_minutes + 30) / 60

-- Theorem to prove
theorem total_watching_time : total_hours = 82 := by
  sorry

end total_watching_time_l538_53801


namespace cost_price_is_47_5_l538_53811

/-- Given an article with a marked price and discount rate, calculates the cost price -/
def calculate_cost_price (marked_price : ℚ) (discount_rate : ℚ) (profit_rate : ℚ) : ℚ :=
  let selling_price := marked_price * (1 - discount_rate)
  selling_price / (1 + profit_rate)

/-- Theorem stating that the cost price of the article is 47.5 given the conditions -/
theorem cost_price_is_47_5 :
  let marked_price : ℚ := 74.21875
  let discount_rate : ℚ := 0.20
  let profit_rate : ℚ := 0.25
  calculate_cost_price marked_price discount_rate profit_rate = 47.5 := by
  sorry

#eval calculate_cost_price 74.21875 0.20 0.25

end cost_price_is_47_5_l538_53811


namespace log_sum_sqrt_equals_sqrt_thirteen_sixths_l538_53833

theorem log_sum_sqrt_equals_sqrt_thirteen_sixths :
  Real.sqrt (Real.log 8 / Real.log 4 + Real.log 4 / Real.log 8) = Real.sqrt (13 / 6) := by
  sorry

end log_sum_sqrt_equals_sqrt_thirteen_sixths_l538_53833


namespace solution_to_system_of_equations_l538_53826

theorem solution_to_system_of_equations :
  ∃ (x y : ℚ), 3 * x - 18 * y = 5 ∧ 4 * y - x = 6 ∧ x = -64/3 ∧ y = -23/6 := by
  sorry

end solution_to_system_of_equations_l538_53826
