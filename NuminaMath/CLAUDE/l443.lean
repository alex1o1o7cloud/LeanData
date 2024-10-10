import Mathlib

namespace first_month_sale_l443_44376

/-- Proves that the sale in the first month was 6235, given the sales for months 2-6
    and the desired average sale for 6 months. -/
theorem first_month_sale
  (sale_2 : ℕ) (sale_3 : ℕ) (sale_4 : ℕ) (sale_5 : ℕ) (sale_6 : ℕ) (average : ℕ)
  (h1 : sale_2 = 6927)
  (h2 : sale_3 = 6855)
  (h3 : sale_4 = 7230)
  (h4 : sale_5 = 6562)
  (h5 : sale_6 = 5191)
  (h6 : average = 6500) :
  6235 = 6 * average - (sale_2 + sale_3 + sale_4 + sale_5 + sale_6) :=
by sorry

end first_month_sale_l443_44376


namespace cricket_players_count_l443_44324

theorem cricket_players_count (total players : ℕ) (hockey football softball : ℕ) :
  total = 55 →
  hockey = 12 →
  football = 13 →
  softball = 15 →
  players = total - (hockey + football + softball) →
  players = 15 :=
by
  sorry

end cricket_players_count_l443_44324


namespace regular_hourly_wage_l443_44396

theorem regular_hourly_wage (
  working_days_per_week : ℕ)
  (working_hours_per_day : ℕ)
  (overtime_rate : ℚ)
  (total_earnings : ℚ)
  (total_hours_worked : ℕ)
  (weeks : ℕ)
  (h1 : working_days_per_week = 6)
  (h2 : working_hours_per_day = 10)
  (h3 : overtime_rate = 21/5)
  (h4 : total_earnings = 525)
  (h5 : total_hours_worked = 245)
  (h6 : weeks = 4) :
  let regular_hours := working_days_per_week * working_hours_per_day * weeks
  let overtime_hours := total_hours_worked - regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  let regular_pay := total_earnings - overtime_pay
  let regular_hourly_wage := regular_pay / regular_hours
  regular_hourly_wage = 21/10 := by
sorry

end regular_hourly_wage_l443_44396


namespace composite_4p_plus_1_l443_44385

theorem composite_4p_plus_1 (p : ℕ) (h1 : p ≥ 5) (h2 : Nat.Prime p) (h3 : Nat.Prime (2 * p + 1)) :
  ¬(Nat.Prime (4 * p + 1)) :=
sorry

end composite_4p_plus_1_l443_44385


namespace union_of_M_and_N_l443_44399

def M : Set ℝ := {x | x^2 + 2*x = 0}
def N : Set ℝ := {x | x^2 - 2*x = 0}

theorem union_of_M_and_N : M ∪ N = {-2, 0, 2} := by sorry

end union_of_M_and_N_l443_44399


namespace fraction_division_l443_44335

theorem fraction_division (a b : ℚ) (ha : a = 3) (hb : b = 4) :
  (1 / b) / (1 / a) = 3 / 4 := by
  sorry

end fraction_division_l443_44335


namespace at_least_three_prime_factors_l443_44308

theorem at_least_three_prime_factors
  (p : Nat)
  (h_prime : Nat.Prime p)
  (h_div : p^2 ∣ 2^(p-1) - 1)
  (n : Nat) :
  ∃ (q₁ q₂ q₃ : Nat),
    Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₂ ≠ q₃ ∧
    (q₁ ∣ (p-1) * (Nat.factorial p + 2^n)) ∧
    (q₂ ∣ (p-1) * (Nat.factorial p + 2^n)) ∧
    (q₃ ∣ (p-1) * (Nat.factorial p + 2^n)) :=
by sorry

end at_least_three_prime_factors_l443_44308


namespace percentage_of_muslim_boys_l443_44374

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 850 →
  hindu_percentage = 28 / 100 →
  sikh_percentage = 10 / 100 →
  other_boys = 187 →
  (total_boys - (hindu_percentage * total_boys + sikh_percentage * total_boys + other_boys)) / total_boys = 40 / 100 := by
  sorry

end percentage_of_muslim_boys_l443_44374


namespace probability_joined_1890_to_1969_l443_44370

def total_provinces : ℕ := 13
def joined_1890_to_1969 : ℕ := 4

theorem probability_joined_1890_to_1969 :
  (joined_1890_to_1969 : ℚ) / total_provinces = 4 / 13 := by
  sorry

end probability_joined_1890_to_1969_l443_44370


namespace jacket_cost_price_l443_44387

theorem jacket_cost_price (original_price discount profit : ℝ) 
  (h1 : original_price = 500)
  (h2 : discount = 0.3)
  (h3 : profit = 50) :
  original_price * (1 - discount) = 300 + profit := by
  sorry

end jacket_cost_price_l443_44387


namespace chocolate_bar_cost_l443_44344

theorem chocolate_bar_cost (total_bars : ℕ) (unsold_bars : ℕ) (total_revenue : ℕ) : 
  total_bars = 11 → 
  unsold_bars = 7 → 
  total_revenue = 16 → 
  (total_revenue : ℚ) / ((total_bars - unsold_bars) : ℚ) = 4 := by
  sorry

end chocolate_bar_cost_l443_44344


namespace gary_egg_collection_l443_44304

/-- Represents the egg-laying rates of the initial chickens -/
def initial_rates : List Nat := [6, 5, 7, 4]

/-- Calculates the number of surviving chickens after two years -/
def surviving_chickens (initial : Nat) (growth_factor : Nat) (mortality_rate : Rat) : Nat :=
  Nat.floor ((initial * growth_factor : Rat) * (1 - mortality_rate))

/-- Calculates the average egg-laying rate -/
def average_rate (rates : List Nat) : Rat :=
  (rates.sum : Rat) / rates.length

/-- Calculates the total eggs per week -/
def total_eggs_per_week (chickens : Nat) (avg_rate : Rat) : Nat :=
  Nat.floor (7 * (chickens : Rat) * avg_rate)

/-- Theorem stating the number of eggs Gary collects per week -/
theorem gary_egg_collection :
  total_eggs_per_week
    (surviving_chickens 4 8 (1/5))
    (average_rate initial_rates) = 959 := by
  sorry

end gary_egg_collection_l443_44304


namespace line_hyperbola_intersection_l443_44327

theorem line_hyperbola_intersection :
  ∃ (k : ℝ), k > 0 ∧
  ∃ (x y : ℝ), y = Real.sqrt 3 * x ∧ y = k / x :=
by sorry

end line_hyperbola_intersection_l443_44327


namespace sphere_radius_ratio_bound_l443_44360

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  -- We don't need to specify the exact structure, just that it exists
  dummy : Unit

/-- The radius of the inscribed sphere of a regular quadrilateral pyramid -/
def inscribed_sphere_radius (p : RegularQuadrilateralPyramid) : ℝ :=
  sorry

/-- The radius of the circumscribed sphere of a regular quadrilateral pyramid -/
def circumscribed_sphere_radius (p : RegularQuadrilateralPyramid) : ℝ :=
  sorry

/-- The theorem stating that the ratio of the circumscribed sphere radius to the inscribed sphere radius
    is greater than or equal to 1 + √2 for any regular quadrilateral pyramid -/
theorem sphere_radius_ratio_bound (p : RegularQuadrilateralPyramid) :
  circumscribed_sphere_radius p / inscribed_sphere_radius p ≥ 1 + Real.sqrt 2 :=
sorry

end sphere_radius_ratio_bound_l443_44360


namespace planes_perpendicular_l443_44334

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line_line : Line → Line → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Define the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : perp_line_plane m α) 
  (h4 : perp_line_plane n β) 
  (h5 : perp_line_line m n) : 
  perp_plane_plane α β :=
sorry

end planes_perpendicular_l443_44334


namespace problem_1_problem_2_problem_3_l443_44306

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Theorem 1
theorem problem_1 (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, g a x ∈ Set.Icc 0 4) ∧
  (∃ x ∈ Set.Icc 1 3, g a x = 0) ∧
  (∃ x ∈ Set.Icc 1 3, g a x = 4) →
  a = 1 := by sorry

-- Theorem 2
theorem problem_2 (k : ℝ) :
  (∀ x ≥ 1, g 1 (2^x) - k * 4^x ≥ 0) →
  k ≤ 1/4 := by sorry

-- Theorem 3
theorem problem_3 (k : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (g 1 (|2^x₁ - 1|) / |2^x₁ - 1| + k * (2 / |2^x₁ - 1|) - 3*k = 0) ∧
    (g 1 (|2^x₂ - 1|) / |2^x₂ - 1| + k * (2 / |2^x₂ - 1|) - 3*k = 0) ∧
    (g 1 (|2^x₃ - 1|) / |2^x₃ - 1| + k * (2 / |2^x₃ - 1|) - 3*k = 0)) →
  k > 0 := by sorry

end problem_1_problem_2_problem_3_l443_44306


namespace fraction_order_l443_44359

theorem fraction_order : 
  let f1 : ℚ := -16/12
  let f2 : ℚ := -18/14
  let f3 : ℚ := -20/15
  f3 = f1 ∧ f1 < f2 := by sorry

end fraction_order_l443_44359


namespace A_intersect_B_eq_singleton_two_l443_44357

def A : Set ℝ := {-1, 2, 3}
def B : Set ℝ := {x : ℝ | x * (x - 3) < 0}

theorem A_intersect_B_eq_singleton_two : A ∩ B = {2} := by sorry

end A_intersect_B_eq_singleton_two_l443_44357


namespace base_k_fraction_l443_44337

/-- If k is a positive integer and 7/51 equals 0.23̅ₖ in base k, then k equals 16. -/
theorem base_k_fraction (k : ℕ) (h1 : k > 0) 
  (h2 : (7 : ℚ) / 51 = (2 * k + 3 : ℚ) / (k^2 - 1)) : k = 16 := by
  sorry

end base_k_fraction_l443_44337


namespace same_remainder_divisor_l443_44350

theorem same_remainder_divisor : ∃ (N : ℕ), N > 1 ∧ 
  N = 23 ∧ 
  (1743 % N = 2019 % N) ∧ 
  (2019 % N = 3008 % N) ∧ 
  ∀ (M : ℕ), M > N → (1743 % M ≠ 2019 % M ∨ 2019 % M ≠ 3008 % M) := by
  sorry

end same_remainder_divisor_l443_44350


namespace interest_difference_theorem_l443_44375

theorem interest_difference_theorem (P : ℝ) : 
  P * ((1 + 0.1)^2 - 1) - P * 0.1 * 2 = 36 → P = 3600 := by
  sorry

end interest_difference_theorem_l443_44375


namespace increasing_function_a_range_l443_44309

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (a - 1) * x + 3 * a - 4 else a^x

-- State the theorem
theorem increasing_function_a_range (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x < f a y) : 
  a > 1 ∧ a ≤ 5/3 := by
  sorry

end increasing_function_a_range_l443_44309


namespace ratio_x_to_y_l443_44328

theorem ratio_x_to_y (x y : ℝ) (h : 0.1 * x = 0.2 * y) : x / y = 2 := by
  sorry

end ratio_x_to_y_l443_44328


namespace complex_fraction_theorem_l443_44395

theorem complex_fraction_theorem (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ = Real.sqrt 5)
  (h2 : Complex.abs z₂ = Real.sqrt 5)
  (h3 : z₁ + z₃ = z₂) : 
  z₁ * z₂ / z₃^2 = -5 := by
  sorry

end complex_fraction_theorem_l443_44395


namespace tangent_x_intercept_difference_l443_44326

/-- 
Given a point (x₀, y₀) on the curve y = e^x, if the tangent line at this point 
intersects the x-axis at (x₁, 0), then x₁ - x₀ = -1.
-/
theorem tangent_x_intercept_difference (x₀ : ℝ) : 
  let y₀ : ℝ := Real.exp x₀
  let f : ℝ → ℝ := λ x => Real.exp x
  let f' : ℝ → ℝ := λ x => Real.exp x
  let tangent_line : ℝ → ℝ := λ x => f' x₀ * (x - x₀) + y₀
  let x₁ : ℝ := x₀ - 1 / f' x₀
  (tangent_line x₁ = 0) → (x₁ - x₀ = -1) := by
  sorry

#check tangent_x_intercept_difference

end tangent_x_intercept_difference_l443_44326


namespace convention_handshakes_l443_44302

/-- The number of handshakes in a convention with multiple companies -/
def number_of_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a convention with 5 companies, each having 4 representatives,
    where every person shakes hands once with every person except those
    from their own company, the total number of handshakes is 160. -/
theorem convention_handshakes :
  number_of_handshakes 5 4 = 160 := by
  sorry

end convention_handshakes_l443_44302


namespace new_member_age_l443_44392

theorem new_member_age (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) : 
  n = 10 → initial_avg = 15 → new_avg = 17 → 
  ∃ (new_member_age : ℝ), 
    (n * initial_avg + new_member_age) / (n + 1) = new_avg ∧ 
    new_member_age = 37 := by
  sorry

end new_member_age_l443_44392


namespace sasha_salt_adjustment_l443_44351

theorem sasha_salt_adjustment (x y : ℝ) 
  (h1 : y > 0)  -- Yesterday's extra salt was positive
  (h2 : x > 0)  -- Initial salt amount is positive
  (h3 : x + y = 2*x + y/2)  -- Total salt needed is the same for both days
  : (3*x) / (2*x) = 3/2 := by
  sorry

end sasha_salt_adjustment_l443_44351


namespace sin_cos_pi_12_l443_44389

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end sin_cos_pi_12_l443_44389


namespace britney_lemon_tea_l443_44356

/-- The number of people sharing the lemon tea -/
def number_of_people : ℕ := 5

/-- The number of cups each person gets -/
def cups_per_person : ℕ := 2

/-- The total number of cups of lemon tea Britney brewed -/
def total_cups : ℕ := number_of_people * cups_per_person

theorem britney_lemon_tea : total_cups = 10 := by
  sorry

end britney_lemon_tea_l443_44356


namespace multiplication_equation_solution_l443_44390

theorem multiplication_equation_solution : ∃ x : ℚ, 9 * x = 36 ∧ x = 4 := by
  sorry

end multiplication_equation_solution_l443_44390


namespace two_digit_addition_problem_l443_44322

theorem two_digit_addition_problem (A B : ℕ) : 
  A ≠ B →
  A * 10 + 7 + 30 + B = 73 →
  A = 3 := by
sorry

end two_digit_addition_problem_l443_44322


namespace integer_between_sqrt2_and_sqrt8_l443_44339

theorem integer_between_sqrt2_and_sqrt8 (a : ℤ) : Real.sqrt 2 < a ∧ a < Real.sqrt 8 → a = 2 := by
  sorry

end integer_between_sqrt2_and_sqrt8_l443_44339


namespace fescue_percentage_in_y_l443_44323

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
def final_mixture (x y : SeedMixture) (x_proportion : ℝ) : SeedMixture :=
  { ryegrass := x_proportion * x.ryegrass + (1 - x_proportion) * y.ryegrass,
    bluegrass := x_proportion * x.bluegrass + (1 - x_proportion) * y.bluegrass,
    fescue := x_proportion * x.fescue + (1 - x_proportion) * y.fescue }

theorem fescue_percentage_in_y
  (x : SeedMixture)
  (y : SeedMixture)
  (h_x_ryegrass : x.ryegrass = 0.4)
  (h_x_bluegrass : x.bluegrass = 0.6)
  (h_x_fescue : x.fescue = 0)
  (h_y_ryegrass : y.ryegrass = 0.25)
  (h_final_ryegrass : (final_mixture x y 0.6667).ryegrass = 0.35)
  : y.fescue = 0.75 := by
  sorry

end fescue_percentage_in_y_l443_44323


namespace range_of_a_circles_intersect_l443_44319

noncomputable section

-- Define the circles and line
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def circle_D (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x = 0
def line (x y a : ℝ) : Prop := x + y - a = 0

-- Define the inequality condition
def inequality_condition (x y m : ℝ) : Prop :=
  x^2 + y^2 - (m + Real.sqrt 2 / 2) * x - (m + Real.sqrt 2 / 2) * y ≤ 0

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x y, circle_C x y ∧ line x y a) →
  2 - Real.sqrt 2 ≤ a ∧ a ≤ 2 + Real.sqrt 2 :=
sorry

-- Theorem for the intersection of circles
theorem circles_intersect (m : ℝ) :
  (∀ x y, circle_C x y → inequality_condition x y m) →
  ∃ x y, circle_C x y ∧ circle_D x y m :=
sorry

end range_of_a_circles_intersect_l443_44319


namespace set_forms_triangle_l443_44358

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three given lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set (6, 8, 13) can form a triangle -/
theorem set_forms_triangle : can_form_triangle 6 8 13 := by
  sorry


end set_forms_triangle_l443_44358


namespace ellipse_properties_l443_44386

/-- Ellipse structure -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Focal distance of an ellipse -/
def focal_distance (e : Ellipse) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_properties (e : Ellipse) (l : Line) :
  l.point.1 = focal_distance e / 2 →  -- line passes through right focus
  l.slope = Real.tan (60 * π / 180) →  -- slope angle is 60°
  distance_point_to_line (-focal_distance e / 2, 0) l = 2 →  -- distance from left focus to line is 2
  focal_distance e = 4 ∧  -- focal distance is 4
  (e.b = 2 → e.a = 3 ∧ focal_distance e = 2 * Real.sqrt 5) :=  -- when b = 2, a = 3 and c = 2√5
by sorry

end ellipse_properties_l443_44386


namespace max_towns_is_four_l443_44305

/-- Represents the type of connection between two towns -/
inductive Connection
  | Air
  | Bus
  | Train

/-- Represents a town -/
structure Town where
  id : Nat

/-- Represents the network of towns and their connections -/
structure TownNetwork where
  towns : Finset Town
  connections : Town → Town → Option Connection

/-- Checks if the given network satisfies all conditions -/
def satisfiesConditions (network : TownNetwork) : Prop :=
  -- Condition 1: Each pair of towns is directly linked by just one of air, bus, or train
  (∀ t1 t2 : Town, t1 ∈ network.towns → t2 ∈ network.towns → t1 ≠ t2 →
    ∃! c : Connection, network.connections t1 t2 = some c) ∧
  -- Condition 2: At least one pair is linked by each type of connection
  (∃ t1 t2 : Town, t1 ∈ network.towns ∧ t2 ∈ network.towns ∧ network.connections t1 t2 = some Connection.Air) ∧
  (∃ t1 t2 : Town, t1 ∈ network.towns ∧ t2 ∈ network.towns ∧ network.connections t1 t2 = some Connection.Bus) ∧
  (∃ t1 t2 : Town, t1 ∈ network.towns ∧ t2 ∈ network.towns ∧ network.connections t1 t2 = some Connection.Train) ∧
  -- Condition 3: No town has all three types of connections
  (∀ t : Town, t ∈ network.towns →
    ¬(∃ t1 t2 t3 : Town, t1 ∈ network.towns ∧ t2 ∈ network.towns ∧ t3 ∈ network.towns ∧
      network.connections t t1 = some Connection.Air ∧
      network.connections t t2 = some Connection.Bus ∧
      network.connections t t3 = some Connection.Train)) ∧
  -- Condition 4: No three towns have all connections of the same type
  (∀ t1 t2 t3 : Town, t1 ∈ network.towns → t2 ∈ network.towns → t3 ∈ network.towns →
    t1 ≠ t2 → t2 ≠ t3 → t1 ≠ t3 →
    ¬(network.connections t1 t2 = network.connections t2 t3 ∧
      network.connections t2 t3 = network.connections t1 t3))

/-- The main theorem stating that the maximum number of towns satisfying the conditions is 4 -/
theorem max_towns_is_four :
  (∃ (network : TownNetwork), satisfiesConditions network ∧ network.towns.card = 4) ∧
  (∀ (network : TownNetwork), satisfiesConditions network → network.towns.card ≤ 4) :=
sorry

end max_towns_is_four_l443_44305


namespace johns_paintball_expenditure_l443_44330

/-- John's monthly paintball expenditure --/
theorem johns_paintball_expenditure :
  ∀ (plays_per_month boxes_per_play box_cost : ℕ),
  plays_per_month = 3 →
  boxes_per_play = 3 →
  box_cost = 25 →
  plays_per_month * boxes_per_play * box_cost = 225 :=
by
  sorry

end johns_paintball_expenditure_l443_44330


namespace regular_polygon_sides_l443_44381

theorem regular_polygon_sides (exterior_angle : ℝ) :
  exterior_angle = 18 →
  (360 / exterior_angle : ℝ) = 20 :=
by sorry

end regular_polygon_sides_l443_44381


namespace largest_two_digit_multiple_of_17_l443_44332

theorem largest_two_digit_multiple_of_17 : ∃ n : ℕ, n = 85 ∧ 
  (∀ m : ℕ, m ≤ 99 → m ≥ 10 → m % 17 = 0 → m ≤ n) :=
by sorry

end largest_two_digit_multiple_of_17_l443_44332


namespace fraction_simplification_and_evaluation_l443_44372

theorem fraction_simplification_and_evaluation (x : ℝ) (h : x ≠ 2) :
  (x^6 - 16*x^3 + 64) / (x^3 - 8) = x^3 - 8 ∧ 
  (6^6 - 16*6^3 + 64) / (6^3 - 8) = 208 :=
sorry

end fraction_simplification_and_evaluation_l443_44372


namespace geometric_sequence_theorem_l443_44355

def geometric_sequence (a₁ : ℚ) (q : ℚ) : ℕ → ℚ
  | 0 => a₁
  | n + 1 => a₁ * q^n

def sum_sequence (a : ℕ → ℚ) : ℕ → ℚ
  | 0 => 0
  | n + 1 => sum_sequence a n + a (n + 1)

def b (S : ℕ → ℚ) (n : ℕ) : ℚ := S n + 1 / (S n)

def is_arithmetic_sequence (a b c : ℚ) : Prop := b - a = c - b

theorem geometric_sequence_theorem (a : ℕ → ℚ) (S : ℕ → ℚ) 
  (h1 : a 1 = 3/2)
  (h2 : ∀ n, S n = sum_sequence a n)
  (h3 : is_arithmetic_sequence (-2 * S 2) (S 3) (4 * S 4)) :
  (∃ q : ℚ, ∀ n, a n = geometric_sequence (3/2) q n) ∧
  (∃ l m : ℚ, ∀ n, l ≤ b S n ∧ b S n ≤ m ∧ m - l = 1/6) :=
sorry

end geometric_sequence_theorem_l443_44355


namespace intersection_of_A_and_B_l443_44368

def A : Set Int := {-2, -1, 0, 1, 2}
def B : Set Int := {x | 2 * x - 1 > 0}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l443_44368


namespace combined_contingency_funds_l443_44366

/-- Calculates the combined contingency funds from two donations given specific conditions. -/
theorem combined_contingency_funds 
  (donation1 : ℝ) 
  (donation2 : ℝ) 
  (community_pantry_rate : ℝ) 
  (crisis_fund_rate : ℝ) 
  (livelihood_rate : ℝ) 
  (disaster_relief_rate : ℝ) 
  (international_aid_rate : ℝ) 
  (education_rate : ℝ) 
  (healthcare_rate : ℝ) 
  (conversion_rate : ℝ) :
  donation1 = 360 →
  donation2 = 180 →
  community_pantry_rate = 0.35 →
  crisis_fund_rate = 0.40 →
  livelihood_rate = 0.10 →
  disaster_relief_rate = 0.05 →
  international_aid_rate = 0.30 →
  education_rate = 0.25 →
  healthcare_rate = 0.25 →
  conversion_rate = 1.20 →
  (donation1 - (community_pantry_rate + crisis_fund_rate + livelihood_rate + disaster_relief_rate) * donation1) +
  (conversion_rate * donation2 - (international_aid_rate + education_rate + healthcare_rate) * conversion_rate * donation2) = 79.20 := by
  sorry


end combined_contingency_funds_l443_44366


namespace intersection_complement_theorem_l443_44369

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

theorem intersection_complement_theorem : A ∩ (U \ B) = {x | 1 ≤ x ∧ x < 3} := by sorry

end intersection_complement_theorem_l443_44369


namespace hula_hoop_difference_is_three_l443_44329

/-- The difference in hula hooping times between Nancy and Casey -/
def hula_hoop_time_difference (nancy_time : ℕ) (morgan_time : ℕ) : ℕ :=
  let casey_time := morgan_time / 3
  nancy_time - casey_time

/-- Theorem stating that the difference in hula hooping times between Nancy and Casey is 3 minutes -/
theorem hula_hoop_difference_is_three :
  hula_hoop_time_difference 10 21 = 3 := by
  sorry

end hula_hoop_difference_is_three_l443_44329


namespace base_number_problem_l443_44349

theorem base_number_problem (a : ℝ) : 
  (a > 0) → (a^14 - a^12 = 3 * a^12) → a = 2 := by sorry

end base_number_problem_l443_44349


namespace train_length_l443_44367

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 72 → time_s = 9 → speed_kmh * (1000 / 3600) * time_s = 180 := by
  sorry

end train_length_l443_44367


namespace geometric_sequence_problem_l443_44391

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  (q > 0) →                     -- positive sequence
  (a 3 = 2) →                   -- given condition
  (a 4 = 8 * a 7) →             -- given condition
  (a 9 = 1 / 32) :=              -- conclusion to prove
by sorry

end geometric_sequence_problem_l443_44391


namespace dragon_poker_partitions_l443_44314

/-- The number of suits in the deck -/
def num_suits : ℕ := 4

/-- The target score to achieve -/
def target_score : ℕ := 2018

/-- The number of ways to partition the target score into exactly num_suits non-negative integers -/
def num_partitions : ℕ := (target_score + num_suits - 1).choose (num_suits - 1)

theorem dragon_poker_partitions :
  num_partitions = 1373734330 := by sorry

end dragon_poker_partitions_l443_44314


namespace inequality_proof_l443_44316

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * Real.sqrt (b * c + c * a + a * b) ≤ Real.sqrt 3 * (((b + c) * (c + a) * (a + b)) ^ (1/3)) := by
  sorry

end inequality_proof_l443_44316


namespace ferry_tourists_sum_l443_44300

/-- The number of trips made by the ferry -/
def num_trips : ℕ := 7

/-- The initial number of tourists -/
def initial_tourists : ℕ := 100

/-- The decrease in tourists per trip -/
def tourist_decrease : ℕ := 2

/-- The sum of tourists over all trips -/
def total_tourists : ℕ := 658

/-- Theorem stating that the sum of the arithmetic sequence
    representing the number of tourists per trip equals the total number of tourists -/
theorem ferry_tourists_sum :
  (num_trips / 2 : ℚ) * (2 * initial_tourists - (num_trips - 1) * tourist_decrease) = total_tourists := by
  sorry

end ferry_tourists_sum_l443_44300


namespace candy_count_l443_44313

/-- The number of bags of candy -/
def num_bags : ℕ := 26

/-- The number of candy pieces in each bag -/
def pieces_per_bag : ℕ := 33

/-- The total number of candy pieces -/
def total_pieces : ℕ := num_bags * pieces_per_bag

theorem candy_count : total_pieces = 858 := by
  sorry

end candy_count_l443_44313


namespace inequality_solution_l443_44352

theorem inequality_solution (x : ℝ) : 
  (Real.sqrt (x^3 - 18*x - 5) + 2) * abs (x^3 - 4*x^2 - 5*x + 18) ≤ 0 ↔ x = 1 - Real.sqrt 10 :=
by sorry

end inequality_solution_l443_44352


namespace two_pump_filling_time_l443_44384

/-- Given two pumps with different filling rates, calculate the time taken to fill a tank when both pumps work together. -/
theorem two_pump_filling_time 
  (small_pump_rate : ℝ) 
  (large_pump_rate : ℝ) 
  (h1 : small_pump_rate = 1 / 4) 
  (h2 : large_pump_rate = 2) : 
  1 / (small_pump_rate + large_pump_rate) = 4 / 9 := by
  sorry

#check two_pump_filling_time

end two_pump_filling_time_l443_44384


namespace solve_system_for_x_l443_44348

theorem solve_system_for_x :
  ∀ (x y : ℚ),
  (3 * x - 2 * y = 8) →
  (x + 3 * y = 7) →
  x = 38 / 11 := by
sorry

end solve_system_for_x_l443_44348


namespace valid_sequence_only_for_3_and_4_l443_44342

/-- A sequence of positive integers satisfying the given recurrence relation -/
def ValidSequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ k, 2 ≤ k → k < n → a (k+1) = (a k ^ 2 + 1) / (a (k-1) + 1) - 1

/-- The theorem stating that only n = 3 and n = 4 satisfy the condition -/
theorem valid_sequence_only_for_3_and_4 :
  ∀ n : ℕ, n > 0 → (∃ a : ℕ → ℕ, ValidSequence a n) ↔ (n = 3 ∨ n = 4) :=
sorry

end valid_sequence_only_for_3_and_4_l443_44342


namespace prob_aces_or_kings_correct_l443_44363

/-- The number of cards in the deck -/
def deck_size : ℕ := 52

/-- The number of aces in the deck -/
def num_aces : ℕ := 5

/-- The number of kings in the deck -/
def num_kings : ℕ := 4

/-- The probability of drawing either two aces or at least one king -/
def prob_aces_or_kings : ℚ := 104 / 663

theorem prob_aces_or_kings_correct :
  let prob_two_aces := (num_aces * (num_aces - 1)) / (deck_size * (deck_size - 1))
  let prob_one_king := 2 * (num_kings * (deck_size - num_kings)) / (deck_size * (deck_size - 1))
  let prob_two_kings := (num_kings * (num_kings - 1)) / (deck_size * (deck_size - 1))
  prob_two_aces + prob_one_king + prob_two_kings = prob_aces_or_kings := by
  sorry

end prob_aces_or_kings_correct_l443_44363


namespace curve_in_second_quadrant_l443_44373

-- Define the curve C
def C (a x y : ℝ) : Prop := x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem curve_in_second_quadrant :
  ∀ a : ℝ, (∀ x y : ℝ, C a x y → second_quadrant x y) ↔ a > 2 :=
sorry

end curve_in_second_quadrant_l443_44373


namespace power_calculation_l443_44303

theorem power_calculation : (9^4 * 3^10) / 27^7 = 1 / 27 := by
  sorry

end power_calculation_l443_44303


namespace tourist_distribution_theorem_l443_44346

/-- The number of ways to distribute tourists among guides --/
def distribute_tourists (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  num_guides ^ num_tourists

/-- The number of ways all tourists can choose the same guide --/
def all_same_guide (num_guides : ℕ) : ℕ := num_guides

/-- The number of valid distributions of tourists among guides --/
def valid_distributions (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  distribute_tourists num_tourists num_guides - all_same_guide num_guides

theorem tourist_distribution_theorem :
  valid_distributions 8 3 = 6558 := by
  sorry

end tourist_distribution_theorem_l443_44346


namespace shooting_events_l443_44345

-- Define the sample space
variable (Ω : Type)
variable [MeasurableSpace Ω]

-- Define the events
variable (both_hit : Set Ω)
variable (exactly_one_hit : Set Ω)
variable (both_miss : Set Ω)
variable (at_least_one_hit : Set Ω)

-- Define the probability measure
variable (P : Measure Ω)

-- Theorem statement
theorem shooting_events :
  (Disjoint exactly_one_hit both_hit) ∧
  (both_miss = at_least_one_hit.compl) := by
sorry

end shooting_events_l443_44345


namespace workshop_percentage_l443_44362

-- Define the workday duration in minutes
def workday_minutes : ℕ := 8 * 60

-- Define the duration of the first workshop in minutes
def first_workshop_minutes : ℕ := 60

-- Define the duration of the second workshop in minutes
def second_workshop_minutes : ℕ := 2 * first_workshop_minutes

-- Define the total time spent in workshops
def total_workshop_minutes : ℕ := first_workshop_minutes + second_workshop_minutes

-- Theorem statement
theorem workshop_percentage :
  (total_workshop_minutes : ℚ) / (workday_minutes : ℚ) * 100 = 37.5 := by
  sorry

end workshop_percentage_l443_44362


namespace binomial_distribution_parameters_l443_44307

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The mean of a binomial distribution -/
def mean (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_parameters :
  ∃ X : BinomialDistribution, mean X = 15 ∧ variance X = 12 ∧ X.n = 60 ∧ X.p = 0.25 := by sorry

end binomial_distribution_parameters_l443_44307


namespace basketball_free_throws_l443_44340

/-- Proves that DeShawn made 12 free-throws given the conditions of the basketball practice problem. -/
theorem basketball_free_throws 
  (deshawn : ℕ) -- DeShawn's free-throws
  (kayla : ℕ) -- Kayla's free-throws
  (annieka : ℕ) -- Annieka's free-throws
  (h1 : kayla = deshawn + deshawn / 2) -- Kayla made 50% more than DeShawn
  (h2 : annieka = kayla - 4) -- Annieka made 4 fewer than Kayla
  (h3 : annieka = 14) -- Annieka made 14 free-throws
  : deshawn = 12 := by
  sorry

end basketball_free_throws_l443_44340


namespace square_area_with_side_30_l443_44343

theorem square_area_with_side_30 :
  let side : ℝ := 30
  let square_area := side * side
  square_area = 900 := by sorry

end square_area_with_side_30_l443_44343


namespace solve_linear_equation_l443_44311

theorem solve_linear_equation :
  ∀ x : ℚ, -3 * x - 8 = 5 * x + 4 → x = -3/2 := by
  sorry

end solve_linear_equation_l443_44311


namespace cylinder_height_l443_44371

/-- Proves that a cylinder with a circular base perimeter of 6 feet and a side surface
    formed by a rectangular plate with a diagonal of 10 feet has a height of 8 feet. -/
theorem cylinder_height (base_perimeter : ℝ) (diagonal : ℝ) (height : ℝ) : 
  base_perimeter = 6 → diagonal = 10 → height = 8 := by
  sorry

end cylinder_height_l443_44371


namespace unequal_gender_probability_l443_44325

/-- The number of grandchildren Mrs. Lee has -/
def n : ℕ := 12

/-- The probability of having a grandson (or granddaughter) -/
def p : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters -/
def unequal_probability : ℚ := 793/1024

theorem unequal_gender_probability :
  (1 - (n.choose (n/2)) * p^n) = unequal_probability :=
sorry

end unequal_gender_probability_l443_44325


namespace trigonometric_propositions_l443_44388

theorem trigonometric_propositions :
  (∃ α : ℝ, Real.sin α + Real.cos α = Real.sqrt 2) ∧
  (∀ x : ℝ, Real.sin (3 * Real.pi / 2 + x) = Real.sin (3 * Real.pi / 2 + (-x))) :=
by sorry

end trigonometric_propositions_l443_44388


namespace monic_quartic_specific_values_l443_44364

-- Define a monic quartic polynomial
def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem monic_quartic_specific_values (f : ℝ → ℝ) 
  (h_monic : is_monic_quartic f)
  (h1 : f (-2) = -4)
  (h2 : f 1 = -1)
  (h3 : f (-3) = -9)
  (h4 : f 5 = -25) :
  f 2 = -64 := by
  sorry

end monic_quartic_specific_values_l443_44364


namespace total_limbs_is_108_l443_44347

/-- The total number of legs, arms, and tentacles of Daniel's animals -/
def total_limbs : ℕ :=
  let horses := 2
  let dogs := 5
  let cats := 7
  let turtles := 3
  let goats := 1
  let snakes := 4
  let spiders := 2
  let birds := 3
  let starfish_arms := 5
  let octopus_tentacles := 6
  let three_legged_dog := 1

  horses * 4 + 
  dogs * 4 + 
  cats * 4 + 
  turtles * 4 + 
  goats * 4 + 
  snakes * 0 + 
  spiders * 8 + 
  birds * 2 + 
  starfish_arms + 
  octopus_tentacles + 
  three_legged_dog * 3

theorem total_limbs_is_108 : total_limbs = 108 := by
  sorry

end total_limbs_is_108_l443_44347


namespace triangle_properties_l443_44312

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  let m : ℝ × ℝ := (Real.cos B, Real.cos C)
  let n : ℝ × ℝ := (2*a + c, b)
  (m.1 * n.1 + m.2 * n.2 = 0) →  -- m ⟂ n
  (b = Real.sqrt 13) →
  (a + c = 4) →
  (B = 2 * Real.pi / 3) ∧
  (Real.sqrt 3 / 2 < Real.sin (2*A) + Real.sin (2*C)) ∧
  (Real.sin (2*A) + Real.sin (2*C) ≤ Real.sqrt 3) ∧
  (1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) :=
by sorry

end triangle_properties_l443_44312


namespace negative_half_power_times_two_power_l443_44361

theorem negative_half_power_times_two_power : (-0.5)^2016 * 2^2017 = 2 := by
  sorry

end negative_half_power_times_two_power_l443_44361


namespace extreme_points_imply_a_l443_44379

/-- Given a function f(x) = a * ln(x) + b * x^2 + x, if x = 1 and x = 2 are extreme points,
    then a = -2/3 --/
theorem extreme_points_imply_a (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * Real.log x + b * x^2 + x
  (∃ (c : ℝ), c ≠ 1 ∧ c ≠ 2 ∧ 
    (deriv f 1 = 0 ∧ deriv f 2 = 0) ∧
    (∀ x ∈ Set.Ioo 1 2, deriv f x ≠ 0)) →
  a = -2/3 := by
sorry

end extreme_points_imply_a_l443_44379


namespace smallest_three_digit_multiple_of_17_l443_44320

theorem smallest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) → (n % 17 = 0) → n ≥ 102 :=
by sorry

end smallest_three_digit_multiple_of_17_l443_44320


namespace negative_result_operations_l443_44365

theorem negative_result_operations : 
  (-(-4) > 0) ∧ 
  (abs (-4) > 0) ∧ 
  (-4^2 < 0) ∧ 
  ((-4)^2 > 0) := by
  sorry

end negative_result_operations_l443_44365


namespace x_squared_plus_nine_y_squared_l443_44318

theorem x_squared_plus_nine_y_squared (x y : ℝ) 
  (h1 : x + 3*y = 6) (h2 : x*y = -9) : x^2 + 9*y^2 = 90 := by
  sorry

end x_squared_plus_nine_y_squared_l443_44318


namespace transformation_of_curve_l443_44394

-- Define the transformation φ
def φ (p : ℝ × ℝ) : ℝ × ℝ := (3 * p.1, 4 * p.2)

-- Define the initial curve
def initial_curve (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 1

-- Define the final curve
def final_curve (p : ℝ × ℝ) : Prop := p.1^2 / 9 + p.2^2 / 16 = 1

-- Theorem statement
theorem transformation_of_curve :
  ∀ p : ℝ × ℝ, initial_curve p ↔ final_curve (φ p) := by sorry

end transformation_of_curve_l443_44394


namespace vectors_perpendicular_l443_44317

theorem vectors_perpendicular : ∀ (a b : ℝ × ℝ), 
  a = (2, -3) → b = (3, 2) → a.1 * b.1 + a.2 * b.2 = 0 := by
  sorry

end vectors_perpendicular_l443_44317


namespace sugar_substitute_usage_l443_44397

/-- Proves that Christopher uses 1 packet of sugar substitute per coffee --/
theorem sugar_substitute_usage
  (coffees_per_day : ℕ)
  (packets_per_box : ℕ)
  (cost_per_box : ℚ)
  (total_cost : ℚ)
  (total_days : ℕ)
  (h1 : coffees_per_day = 2)
  (h2 : packets_per_box = 30)
  (h3 : cost_per_box = 4)
  (h4 : total_cost = 24)
  (h5 : total_days = 90) :
  (total_cost / cost_per_box * packets_per_box) / (total_days * coffees_per_day) = 1 := by
  sorry


end sugar_substitute_usage_l443_44397


namespace complete_graph_inequality_l443_44354

theorem complete_graph_inequality (n k : ℕ) (N_k N_k_plus_1 : ℕ) 
  (h1 : 2 ≤ k) (h2 : k < n) (h3 : N_k > 0) (h4 : N_k_plus_1 > 0) :
  (N_k_plus_1 : ℚ) / N_k ≥ (1 : ℚ) / (k^2 - 1) * (k^2 * N_k / N_k_plus_1 - n) := by
  sorry

end complete_graph_inequality_l443_44354


namespace division_problem_l443_44315

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 159 → quotient = 9 → remainder = 6 → 
  dividend = divisor * quotient + remainder → 
  divisor = 17 := by
sorry

end division_problem_l443_44315


namespace units_digit_sum_base_8_l443_44301

/-- The units digit of a number in a given base -/
def unitsDigit (n : ℕ) (base : ℕ) : ℕ :=
  n % base

/-- Addition in a given base -/
def baseAddition (a b base : ℕ) : ℕ :=
  (a + b) % base^2

theorem units_digit_sum_base_8 :
  unitsDigit (baseAddition 35 47 8) 8 = 4 := by sorry

end units_digit_sum_base_8_l443_44301


namespace inverse_power_inequality_l443_44321

theorem inverse_power_inequality (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  (1 / a) ^ (1 / b) ≤ 1 := by sorry

end inverse_power_inequality_l443_44321


namespace f_min_implies_a_range_l443_44331

/-- A function f with a real parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := |3*x - 1| + a*x + 2

/-- The theorem stating that if f has a minimum value, then a is in [-3, 3] -/
theorem f_min_implies_a_range (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) → a ∈ Set.Icc (-3) 3 := by
  sorry

end f_min_implies_a_range_l443_44331


namespace rent_equation_l443_44380

/-- The monthly rent of Janet's apartment -/
def monthly_rent : ℝ := 1250

/-- Janet's savings -/
def savings : ℝ := 2225

/-- Additional amount Janet needs -/
def additional_amount : ℝ := 775

/-- Deposit required by the landlord -/
def deposit : ℝ := 500

/-- Number of months' rent required in advance -/
def months_in_advance : ℕ := 2

theorem rent_equation :
  2 * monthly_rent + deposit = savings + additional_amount :=
by sorry

end rent_equation_l443_44380


namespace energy_drink_consumption_l443_44336

/-- Represents the relationship between relaxation hours and energy drink consumption --/
def inverse_proportional (h g : ℝ) (k : ℝ) : Prop := h * g = k

theorem energy_drink_consumption 
  (h₁ h₂ g₁ g₂ : ℝ) 
  (h₁_pos : h₁ > 0) 
  (h₂_pos : h₂ > 0) 
  (g₁_pos : g₁ > 0) 
  (h₁_val : h₁ = 4) 
  (h₂_val : h₂ = 2) 
  (g₁_val : g₁ = 5) 
  (prop_const : ℝ) 
  (inv_prop₁ : inverse_proportional h₁ g₁ prop_const) 
  (inv_prop₂ : inverse_proportional h₂ g₂ prop_const) : 
  g₂ = 10 := by
sorry

end energy_drink_consumption_l443_44336


namespace complex_equation_sum_l443_44310

theorem complex_equation_sum (a b : ℝ) (h : (a : ℂ) + b * Complex.I = (1 - Complex.I) * (2 + Complex.I)) : 
  a + b = 2 := by sorry

end complex_equation_sum_l443_44310


namespace proposition_counterexample_l443_44341

theorem proposition_counterexample : 
  ∃ (α β : Real), 
    α > β ∧ 
    0 < α ∧ α < Real.pi / 2 ∧
    0 < β ∧ β < Real.pi / 2 ∧
    Real.tan α ≤ Real.tan β :=
by sorry

end proposition_counterexample_l443_44341


namespace modular_inverse_of_7_mod_10000_l443_44382

-- Define the modulus
def m : ℕ := 10000

-- Define the number we're finding the inverse for
def a : ℕ := 7

-- Define the claimed inverse
def claimed_inverse : ℕ := 8571

-- Theorem statement
theorem modular_inverse_of_7_mod_10000 :
  (a * claimed_inverse) % m = 1 ∧ 0 ≤ claimed_inverse ∧ claimed_inverse < m :=
by sorry

end modular_inverse_of_7_mod_10000_l443_44382


namespace half_angle_quadrant_l443_44377

-- Define a function to determine if an angle is in the first quadrant
def is_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + Real.pi / 2

-- Define a function to determine if an angle is in the first or third quadrant
def is_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi < α ∧ α < k * Real.pi + Real.pi / 2

-- Theorem statement
theorem half_angle_quadrant (α : ℝ) :
  is_first_quadrant α → is_first_or_third_quadrant (α / 2) :=
by sorry

end half_angle_quadrant_l443_44377


namespace sum_digits_greatest_prime_divisor_of_16385_l443_44338

def n : ℕ := 16385

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (m : ℕ) : ℕ := 
  sorry

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := 
  sorry

-- Theorem statement
theorem sum_digits_greatest_prime_divisor_of_16385 : 
  sum_of_digits (greatest_prime_divisor n) = 13 := by
  sorry

end sum_digits_greatest_prime_divisor_of_16385_l443_44338


namespace quadratic_inequality_l443_44393

theorem quadratic_inequality (x : ℝ) : x^2 - 36*x + 323 ≤ 3 ↔ 16 ≤ x ∧ x ≤ 20 := by
  sorry

end quadratic_inequality_l443_44393


namespace monotonic_increasing_condition_l443_44378

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x + 3

-- State the theorem
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, Monotone (f a)) → a ≥ 1 := by sorry

end monotonic_increasing_condition_l443_44378


namespace rectangle_area_l443_44333

/-- Given a rectangle with perimeter 176 inches and length 8 inches more than its width,
    prove that its area is 1920 square inches. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) (h2 : 4 * w + 16 = 176) : w * (w + 8) = 1920 := by
  sorry

end rectangle_area_l443_44333


namespace gcd_of_abcd_plus_dcba_l443_44353

def abcd_plus_dcba (a : ℕ) : ℕ := 4201 * a + 12606

def number_set : Set ℕ := {n | ∃ a : ℕ, 1 ≤ a ∧ a ≤ 4 ∧ n = abcd_plus_dcba a}

theorem gcd_of_abcd_plus_dcba :
  ∃ g : ℕ, g > 0 ∧ (∀ n ∈ number_set, g ∣ n) ∧
  (∀ d : ℕ, d > 0 → (∀ n ∈ number_set, d ∣ n) → d ≤ g) ∧
  g = 4201 := by
  sorry

end gcd_of_abcd_plus_dcba_l443_44353


namespace carltons_outfits_l443_44398

/-- Represents a person's wardrobe and outfit combinations -/
structure Wardrobe where
  buttonUpShirts : ℕ
  sweaterVests : ℕ
  outfits : ℕ

/-- Calculates the number of outfits for a given wardrobe -/
def calculateOutfits (w : Wardrobe) : Prop :=
  w.sweaterVests = 2 * w.buttonUpShirts ∧
  w.outfits = w.sweaterVests * w.buttonUpShirts

/-- Theorem: Carlton's wardrobe has 18 outfits -/
theorem carltons_outfits :
  ∃ (w : Wardrobe), w.buttonUpShirts = 3 ∧ calculateOutfits w ∧ w.outfits = 18 := by
  sorry


end carltons_outfits_l443_44398


namespace expired_yogurt_percentage_l443_44383

theorem expired_yogurt_percentage (total_packs : ℕ) (cost_per_pack : ℚ) (refund_amount : ℚ) :
  total_packs = 80 →
  cost_per_pack = 12 →
  refund_amount = 384 →
  (refund_amount / cost_per_pack) / total_packs * 100 = 40 := by
  sorry

end expired_yogurt_percentage_l443_44383
