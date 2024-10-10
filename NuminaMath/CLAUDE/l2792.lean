import Mathlib

namespace train_length_l2792_279288

theorem train_length (platform_time : ℝ) (pole_time : ℝ) (platform_length : ℝ)
  (h1 : platform_time = 39)
  (h2 : pole_time = 18)
  (h3 : platform_length = 350) :
  ∃ (train_length : ℝ) (train_speed : ℝ),
    train_length = train_speed * pole_time ∧
    train_length + platform_length = train_speed * platform_time ∧
    train_length = 300 := by
  sorry

end train_length_l2792_279288


namespace julia_tuesday_kids_l2792_279233

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 16

/-- The difference between the number of kids Julia played with on Monday and Tuesday -/
def difference : ℕ := 12

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := monday_kids - difference

theorem julia_tuesday_kids : tuesday_kids = 4 := by
  sorry

end julia_tuesday_kids_l2792_279233


namespace special_sequence_1000th_term_l2792_279225

def special_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1010 ∧ 
  a 2 = 1015 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n + 1

theorem special_sequence_1000th_term (a : ℕ → ℕ) 
  (h : special_sequence a) : a 1000 = 1676 := by
  sorry

end special_sequence_1000th_term_l2792_279225


namespace english_physical_novels_count_l2792_279250

/-- Represents Iesha's book collection -/
structure BookCollection where
  total : ℕ
  english : ℕ
  school : ℕ
  sports : ℕ
  novels : ℕ
  english_sports : ℕ
  english_school : ℕ
  english_novels : ℕ
  digital_novels : ℕ
  physical_novels : ℕ

/-- Theorem stating the number of English physical format novels in Iesha's collection -/
theorem english_physical_novels_count (c : BookCollection) : c.physical_novels = 135 :=
  by
  have h1 : c.total = 2000 := by sorry
  have h2 : c.english = c.total / 2 := by sorry
  have h3 : c.school = c.total * 30 / 100 := by sorry
  have h4 : c.sports = c.total * 25 / 100 := by sorry
  have h5 : c.novels = c.total - c.school - c.sports := by sorry
  have h6 : c.english_sports = c.english * 10 / 100 := by sorry
  have h7 : c.english_school = c.english * 45 / 100 := by sorry
  have h8 : c.english_novels = c.english - c.english_sports - c.english_school := by sorry
  have h9 : c.digital_novels = c.english_novels * 70 / 100 := by sorry
  have h10 : c.physical_novels = c.english_novels - c.digital_novels := by sorry
  sorry

end english_physical_novels_count_l2792_279250


namespace simplify_and_evaluate_expression_l2792_279228

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = -4) :
  (1 - (x + 1) / (x^2 - 2*x + 1)) / ((x - 3) / (x - 1)) = 4/5 := by
  sorry

end simplify_and_evaluate_expression_l2792_279228


namespace trains_meet_at_360km_l2792_279246

/-- Represents a train with its departure time and speed -/
structure Train where
  departureTime : ℕ  -- Departure time in hours after midnight
  speed : ℕ         -- Speed in km/h
  deriving Repr

/-- Calculates the meeting point of three trains -/
def meetingPoint (trainA trainB trainC : Train) : ℕ :=
  let t : ℕ := 18  -- 6 p.m. in 24-hour format
  let distanceA : ℕ := trainA.speed * (t - trainA.departureTime) 
  let distanceB : ℕ := trainB.speed * (t - trainB.departureTime)
  let timeAfterC : ℕ := (distanceB - distanceA) / (trainA.speed - trainB.speed)
  trainC.speed * timeAfterC

theorem trains_meet_at_360km :
  let trainA : Train := { departureTime := 9, speed := 30 }
  let trainB : Train := { departureTime := 15, speed := 40 }
  let trainC : Train := { departureTime := 18, speed := 60 }
  meetingPoint trainA trainB trainC = 360 := by
  sorry

#eval meetingPoint { departureTime := 9, speed := 30 } { departureTime := 15, speed := 40 } { departureTime := 18, speed := 60 }

end trains_meet_at_360km_l2792_279246


namespace exists_composite_invariant_under_triplet_replacement_l2792_279281

/-- A function that replaces a triplet of digits at a given position in a natural number --/
def replaceTriplet (n : ℕ) (pos : ℕ) (newTriplet : ℕ) : ℕ :=
  sorry

/-- Predicate to check if a number is composite --/
def isComposite (n : ℕ) : Prop :=
  ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b

/-- The main theorem statement --/
theorem exists_composite_invariant_under_triplet_replacement :
  ∃ (N : ℕ), ∀ (pos : ℕ) (newTriplet : ℕ),
    isComposite (replaceTriplet N pos newTriplet) :=
  sorry

end exists_composite_invariant_under_triplet_replacement_l2792_279281


namespace quadratic_roots_relation_l2792_279275

theorem quadratic_roots_relation (p A B : ℤ) : 
  (∃ α β : ℝ, α + 1 ≠ β + 1 ∧ 
    (∀ x : ℝ, x^2 + p*x + 19 = 0 ↔ x = α + 1 ∨ x = β + 1) ∧
    (∀ x : ℝ, x^2 - A*x + B = 0 ↔ x = α ∨ x = β)) →
  A + B = 18 := by
sorry

end quadratic_roots_relation_l2792_279275


namespace inequality_hold_l2792_279201

theorem inequality_hold (x : ℝ) : 
  x ≥ -1/2 → x ≠ 0 → 
  (4 * x^2 / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9 ↔ 
   (-1/2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x < 24)) :=
by sorry

end inequality_hold_l2792_279201


namespace linear_inequality_solution_set_l2792_279268

theorem linear_inequality_solution_set 
  (m n : ℝ) 
  (h1 : m = -1) 
  (h2 : n = -1) : 
  {x : ℝ | m * x - n ≤ 2} = {x : ℝ | x ≥ -1} := by
sorry

end linear_inequality_solution_set_l2792_279268


namespace discriminant_zero_iff_unique_solution_unique_solution_iff_m_eq_three_l2792_279248

/-- A quadratic equation ax^2 + bx + c = 0 has exactly one solution if and only if its discriminant is zero -/
theorem discriminant_zero_iff_unique_solution (a b c : ℝ) (ha : a ≠ 0) :
  (b^2 - 4*a*c = 0) ↔ (∃! x, a*x^2 + b*x + c = 0) :=
sorry

/-- The quadratic equation 3x^2 - 6x + m = 0 has exactly one solution if and only if m = 3 -/
theorem unique_solution_iff_m_eq_three :
  (∃! x, 3*x^2 - 6*x + m = 0) ↔ m = 3 :=
sorry

end discriminant_zero_iff_unique_solution_unique_solution_iff_m_eq_three_l2792_279248


namespace symmetry_and_line_equation_l2792_279235

/-- The curve on which points P and Q lie -/
def curve (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

/-- The line of symmetry for points P and Q -/
def symmetry_line (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

/-- The condition satisfied by the coordinates of P and Q -/
def coordinate_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

/-- The theorem stating the value of m and the equation of line PQ -/
theorem symmetry_and_line_equation 
  (x₁ y₁ x₂ y₂ m : ℝ) 
  (h_curve_P : curve x₁ y₁)
  (h_curve_Q : curve x₂ y₂)
  (h_symmetry : symmetry_line m x₁ y₁ ∧ symmetry_line m x₂ y₂)
  (h_condition : coordinate_condition x₁ y₁ x₂ y₂) :
  m = -1 ∧ ∀ (x y : ℝ), y = -x + 1 ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) :=
sorry

end symmetry_and_line_equation_l2792_279235


namespace inequality_proof_l2792_279269

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * (a + 1)) / (b + 1) + (b * (b + 1)) / (a + 1) ≥ a + b :=
by sorry

end inequality_proof_l2792_279269


namespace complex_fraction_simplification_l2792_279237

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 6 * I
  let z₂ : ℂ := 4 - 6 * I
  z₁ / z₂ - z₂ / z₁ = 24 * I / 13 := by
  sorry

end complex_fraction_simplification_l2792_279237


namespace largest_m_is_nine_l2792_279295

/-- A quadratic function f(x) = ax² + bx + c satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  symmetry : ∀ x : ℝ, a * (x - 4)^2 + b * (x - 4) + c = a * (2 - x)^2 + b * (2 - x) + c
  lower_bound : ∀ x : ℝ, a * x^2 + b * x + c ≥ x
  upper_bound : ∀ x ∈ Set.Ioo 0 2, a * x^2 + b * x + c ≤ ((x + 1) / 2)^2
  min_value : ∃ x : ℝ, ∀ y : ℝ, a * x^2 + b * x + c ≤ a * y^2 + b * y + c ∧ a * x^2 + b * x + c = 0

/-- The theorem stating that the largest m > 1 satisfying the given conditions is 9 -/
theorem largest_m_is_nine (f : QuadraticFunction) :
  ∃ m : ℝ, m = 9 ∧ 
  (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f.a * (x + t)^2 + f.b * (x + t) + f.c ≤ x) ∧
  ∀ m' > m, ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f.a * (x + t)^2 + f.b * (x + t) + f.c ≤ x :=
sorry

end largest_m_is_nine_l2792_279295


namespace cylinder_lateral_surface_area_l2792_279265

theorem cylinder_lateral_surface_area 
  (r h : ℝ) 
  (hr : r = 2) 
  (hh : h = 2) : 
  2 * Real.pi * r * h = 8 * Real.pi :=
by sorry

end cylinder_lateral_surface_area_l2792_279265


namespace martinez_family_height_l2792_279204

def chiquitaHeight : ℝ := 5

def mrMartinezHeight : ℝ := chiquitaHeight + 2

def mrsMartinezHeight : ℝ := chiquitaHeight - 1

def sonHeight : ℝ := chiquitaHeight + 3

def combinedFamilyHeight : ℝ := chiquitaHeight + mrMartinezHeight + mrsMartinezHeight + sonHeight

theorem martinez_family_height : combinedFamilyHeight = 24 := by
  sorry

end martinez_family_height_l2792_279204


namespace min_a_for_nonnegative_f_l2792_279263

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (-3 * x^2 + a * x) - a / x

theorem min_a_for_nonnegative_f :
  ∀ a : ℝ, a > 0 →
  (∃ x₀ : ℝ, f a x₀ ≥ 0) →
  a ≥ 12 * Real.sqrt 3 :=
by sorry

end min_a_for_nonnegative_f_l2792_279263


namespace B_is_midpoint_of_AC_l2792_279291

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A, B, C, and O
variable (O A B C : V)

-- Define the collinearity of points A, B, and C
def collinear (A B C : V) : Prop :=
  ∃ t : ℝ, B - A = t • (C - A)

-- Define the vector equation
def vector_equation (m : ℝ) : Prop :=
  m • (A - O) - 2 • (B - O) + (C - O) = 0

-- Theorem statement
theorem B_is_midpoint_of_AC 
  (h_collinear : collinear A B C)
  (h_equation : ∃ m : ℝ, vector_equation O A B C m) :
  B - O = (1/2) • ((A - O) + (C - O)) :=
sorry

end B_is_midpoint_of_AC_l2792_279291


namespace bikers_meeting_time_l2792_279271

/-- The time (in minutes) it takes for two bikers to meet again at the starting point of a circular path -/
def meetingTime (t1 t2 : ℕ) : ℕ :=
  Nat.lcm t1 t2

/-- Theorem stating that two bikers with given round completion times will meet at the starting point after a specific time -/
theorem bikers_meeting_time :
  let t1 : ℕ := 12  -- Time for first biker to complete a round
  let t2 : ℕ := 18  -- Time for second biker to complete a round
  meetingTime t1 t2 = 36 := by
  sorry

end bikers_meeting_time_l2792_279271


namespace pyramid_volume_in_cube_l2792_279297

theorem pyramid_volume_in_cube (s : ℝ) (h : s > 0) :
  let cube_volume := s^3
  let pyramid_volume := (1/3) * (s^2/2) * s
  pyramid_volume = (1/6) * cube_volume := by
sorry

end pyramid_volume_in_cube_l2792_279297


namespace min_groups_for_children_l2792_279234

/-- Given a total of 30 children and a maximum of 7 children per group,
    prove that the minimum number of equal-sized groups needed is 5. -/
theorem min_groups_for_children (total_children : Nat) (max_per_group : Nat) 
    (h1 : total_children = 30) (h2 : max_per_group = 7) : 
    (∃ (group_size : Nat), group_size ≤ max_per_group ∧ 
    total_children % group_size = 0 ∧ 
    total_children / group_size = 5 ∧
    ∀ (other_size : Nat), other_size ≤ max_per_group ∧ 
    total_children % other_size = 0 → 
    total_children / other_size ≥ 5) := by
  sorry

end min_groups_for_children_l2792_279234


namespace train_speed_problem_l2792_279283

theorem train_speed_problem (length1 length2 : Real) (crossing_time : Real) (speed1 : Real) (speed2 : Real) :
  length1 = 150 ∧ 
  length2 = 160 ∧ 
  crossing_time = 11.159107271418288 ∧
  speed1 = 60 ∧
  (length1 + length2) / crossing_time = (speed1 * 1000 / 3600) + (speed2 * 1000 / 3600) →
  speed2 = 40 := by
sorry

end train_speed_problem_l2792_279283


namespace distance_between_cities_l2792_279249

/-- The distance between two cities given the speeds of two cars and their time difference --/
theorem distance_between_cities (v1 v2 : ℝ) (t_diff : ℝ) (h1 : v1 = 60) (h2 : v2 = 70) (h3 : t_diff = 0.25) :
  ∃ d : ℝ, d = 105 ∧ d = v1 * (d / v1) ∧ d = v2 * (d / v2 - t_diff) := by
  sorry

end distance_between_cities_l2792_279249


namespace combine_squares_simplify_expression_linear_combination_l2792_279272

-- Part 1
theorem combine_squares (a b : ℝ) :
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 := by sorry

-- Part 2
theorem simplify_expression (x y : ℝ) (h : x^2 - 2*y = 4) :
  3*x^2 - 6*y - 21 = -9 := by sorry

-- Part 3
theorem linear_combination (a b c d : ℝ) 
  (h1 : a - 5*b = 3) (h2 : 5*b - 3*c = -5) (h3 : 3*c - d = 10) :
  (a - 3*c) + (5*b - d) - (5*b - 3*c) = 8 := by sorry

end combine_squares_simplify_expression_linear_combination_l2792_279272


namespace square_field_area_l2792_279280

/-- Given a square field where a horse takes 7 hours to run around it at a speed of 20 km/h,
    the area of the field is 1225 km². -/
theorem square_field_area (s : ℝ) (h : s > 0) : 
  (4 * s = 20 * 7) → s^2 = 1225 := by sorry

end square_field_area_l2792_279280


namespace oil_depth_relationship_l2792_279210

/-- Represents a right cylindrical tank -/
structure CylindricalTank where
  height : ℝ
  baseDiameter : ℝ

/-- Represents the oil level in the tank -/
structure OilLevel where
  depthWhenFlat : ℝ
  depthWhenUpright : ℝ

/-- The theorem stating the relationship between oil depths -/
theorem oil_depth_relationship (tank : CylindricalTank) (oil : OilLevel) :
  tank.height = 15 ∧ 
  tank.baseDiameter = 6 ∧ 
  oil.depthWhenFlat = 4 →
  oil.depthWhenUpright = 15 := by
  sorry


end oil_depth_relationship_l2792_279210


namespace cost_of_450_candies_l2792_279226

/-- The cost of buying a given number of chocolate candies -/
def cost_of_candies (candies_per_box : ℕ) (cost_per_box : ℚ) (total_candies : ℕ) : ℚ :=
  (total_candies / candies_per_box : ℚ) * cost_per_box

/-- Theorem: The cost of 450 chocolate candies is $112.50, given that a box of 30 costs $7.50 -/
theorem cost_of_450_candies :
  cost_of_candies 30 (7.5 : ℚ) 450 = (112.5 : ℚ) := by
  sorry

end cost_of_450_candies_l2792_279226


namespace dealership_sales_theorem_l2792_279289

/-- Represents the ratio of trucks to minivans sold -/
def truck_to_minivan_ratio : ℚ := 5 / 3

/-- Number of trucks expected to be sold -/
def expected_trucks : ℕ := 45

/-- Price of each truck in dollars -/
def truck_price : ℕ := 25000

/-- Price of each minivan in dollars -/
def minivan_price : ℕ := 20000

/-- Calculates the expected number of minivans to be sold -/
def expected_minivans : ℕ := (expected_trucks * 3) / 5

/-- Calculates the total revenue from truck and minivan sales -/
def total_revenue : ℕ := expected_trucks * truck_price + expected_minivans * minivan_price

theorem dealership_sales_theorem :
  expected_minivans = 27 ∧ total_revenue = 1665000 := by
  sorry

end dealership_sales_theorem_l2792_279289


namespace smallest_n_for_prob_less_than_half_l2792_279217

def probability_red (n : ℕ) : ℚ :=
  9 / (11 - n)

theorem smallest_n_for_prob_less_than_half :
  ∀ n : ℕ, n > 0 → (∀ k : ℕ, 0 < k → k ≤ n → probability_red k < (1/2)) →
    n ≥ 8 :=
sorry

end smallest_n_for_prob_less_than_half_l2792_279217


namespace evaluate_expression_l2792_279207

theorem evaluate_expression : 
  3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^7 = 6^5 + 3^7 := by
  sorry

end evaluate_expression_l2792_279207


namespace prob_A_level_l2792_279276

/-- The probability of producing a B-level product -/
def prob_B : ℝ := 0.03

/-- The probability of producing a C-level product -/
def prob_C : ℝ := 0.01

/-- Theorem: The probability of selecting an A-level product is 0.96 -/
theorem prob_A_level (h1 : prob_B = 0.03) (h2 : prob_C = 0.01) :
  1 - (prob_B + prob_C) = 0.96 := by
  sorry

end prob_A_level_l2792_279276


namespace triangle_angle_proof_l2792_279242

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where C = π/6 and 2acosB = c, prove that A = 5π/12. -/
theorem triangle_angle_proof (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  A + B + C = π →  -- Sum of angles in a triangle
  C = π / 6 →  -- Given condition
  2 * a * Real.cos B = c →  -- Given condition
  A = 5 * π / 12 := by sorry

end triangle_angle_proof_l2792_279242


namespace quadratic_minimum_l2792_279254

/-- The quadratic function we're minimizing -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- The point where the function is minimized -/
def min_point : ℝ := 3

theorem quadratic_minimum :
  ∀ x : ℝ, f x ≥ f min_point :=
sorry

end quadratic_minimum_l2792_279254


namespace u_limit_and_bound_l2792_279292

def u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 3 * u n - 3 * (u n)^2

theorem u_limit_and_bound : 
  (∀ k : ℕ, u k = 1/3) ∧ |u 0 - 1/3| ≤ 1/(2^1000) := by sorry

end u_limit_and_bound_l2792_279292


namespace carries_hourly_rate_l2792_279218

/-- Represents Carrie's cake-making scenario -/
structure CakeScenario where
  hoursPerDay : ℕ
  daysWorked : ℕ
  suppliesCost : ℕ
  profit : ℕ

/-- Calculates Carrie's hourly rate given the scenario -/
def hourlyRate (scenario : CakeScenario) : ℚ :=
  (scenario.profit + scenario.suppliesCost) / (scenario.hoursPerDay * scenario.daysWorked)

/-- Theorem stating that Carrie's hourly rate was $22 -/
theorem carries_hourly_rate :
  let scenario : CakeScenario := {
    hoursPerDay := 2,
    daysWorked := 4,
    suppliesCost := 54,
    profit := 122
  }
  hourlyRate scenario = 22 := by sorry

end carries_hourly_rate_l2792_279218


namespace distribution_ways_for_problem_l2792_279208

/-- Represents a hotel with a fixed number of rooms -/
structure Hotel where
  numRooms : Nat
  maxPerRoom : Nat

/-- Represents a group of friends -/
structure FriendGroup where
  numFriends : Nat

/-- Calculates the number of ways to distribute friends in rooms -/
def distributionWays (h : Hotel) (f : FriendGroup) : Nat :=
  sorry

/-- The specific hotel in the problem -/
def problemHotel : Hotel :=
  { numRooms := 5, maxPerRoom := 2 }

/-- The specific friend group in the problem -/
def problemFriendGroup : FriendGroup :=
  { numFriends := 5 }

theorem distribution_ways_for_problem :
  distributionWays problemHotel problemFriendGroup = 2220 :=
sorry

end distribution_ways_for_problem_l2792_279208


namespace lloyd_decks_required_l2792_279284

/-- Represents the number of cards in a standard deck --/
def cards_per_deck : ℕ := 52

/-- Represents the number of layers in Lloyd's house of cards --/
def num_layers : ℕ := 32

/-- Represents the number of cards per layer in Lloyd's house of cards --/
def cards_per_layer : ℕ := 26

/-- Calculates the total number of cards in the house of cards --/
def total_cards : ℕ := num_layers * cards_per_layer

/-- Theorem: The number of complete decks required for Lloyd's house of cards is 16 --/
theorem lloyd_decks_required : (total_cards / cards_per_deck) = 16 := by
  sorry

end lloyd_decks_required_l2792_279284


namespace compensation_problem_l2792_279285

/-- Represents the compensation amounts for cow, horse, and sheep respectively -/
structure Compensation where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The problem statement -/
theorem compensation_problem (comp : Compensation) : 
  -- Total compensation is 5 measures (50 liters)
  comp.a + comp.b + comp.c = 50 →
  -- Sheep ate half as much as horse
  comp.c = (1/2) * comp.b →
  -- Horse ate half as much as cow
  comp.b = (1/2) * comp.a →
  -- Compensation is proportional to what each animal ate
  (∃ (k : ℚ), k > 0 ∧ comp.a = k * 4 ∧ comp.b = k * 2 ∧ comp.c = k * 1) →
  -- Prove that a, b, c form a geometric sequence with ratio 1/2 and c = 50/7
  (comp.b = (1/2) * comp.a ∧ comp.c = (1/2) * comp.b) ∧ comp.c = 50/7 := by
  sorry

end compensation_problem_l2792_279285


namespace range_of_a_l2792_279262

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

-- Define the range of a
def range_a (a : ℝ) : Prop := (1 ≤ a ∧ a < 2) ∨ a ≤ -2

-- State the theorem
theorem range_of_a : ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_a a := by
  sorry

end range_of_a_l2792_279262


namespace power_calculation_l2792_279209

theorem power_calculation : 3^18 / 27^3 * 9 = 177147 := by
  sorry

end power_calculation_l2792_279209


namespace library_shelves_problem_l2792_279205

/-- Calculates the number of shelves needed to store books -/
def shelves_needed (large_books small_books shelf_capacity : ℕ) : ℕ :=
  let total_units := 2 * large_books + small_books
  (total_units + shelf_capacity - 1) / shelf_capacity

theorem library_shelves_problem :
  let initial_large_books := 18
  let initial_small_books := 18
  let removed_large_books := 4
  let removed_small_books := 2
  let shelf_capacity := 6
  let remaining_large_books := initial_large_books - removed_large_books
  let remaining_small_books := initial_small_books - removed_small_books
  shelves_needed remaining_large_books remaining_small_books shelf_capacity = 8 := by
  sorry

end library_shelves_problem_l2792_279205


namespace arithmetic_geometric_sequence_l2792_279232

/-- Given an arithmetic sequence with common difference 3 where a₁, a₃, a₄ form a geometric sequence, a₂ = -6 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = 3) →  -- arithmetic sequence with common difference 3
  (a 3)^2 = a 1 * a 4 →         -- a₁, a₃, a₄ form a geometric sequence
  a 2 = -6 := by
sorry

end arithmetic_geometric_sequence_l2792_279232


namespace min_remainders_consecutive_numbers_l2792_279259

theorem min_remainders_consecutive_numbers : ∃ (x a r : ℕ), 
  (100 ≤ x) ∧ (x < 1000) ∧
  (10 ≤ a) ∧ (a < 100) ∧
  (r < a) ∧ (a + r ≥ 100) ∧
  (∀ i : Fin 4, (x + i) % (a + i) = r) :=
by sorry

end min_remainders_consecutive_numbers_l2792_279259


namespace ratio_problem_l2792_279294

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 1)
  (h3 : d / b = 2 / 5) :
  a / c = 25 / 32 := by
sorry

end ratio_problem_l2792_279294


namespace brothers_age_sum_l2792_279257

theorem brothers_age_sum : ∃ x : ℤ, 5 * x - 6 = 89 :=
  sorry

end brothers_age_sum_l2792_279257


namespace theta_range_theorem_l2792_279206

-- Define the set of valid θ values
def ValidTheta : Set ℝ := { θ | -Real.pi ≤ θ ∧ θ ≤ Real.pi }

-- Define the inequality condition
def InequalityCondition (θ : ℝ) : Prop :=
  Real.cos (θ + Real.pi / 4) < 3 * (Real.sin θ ^ 5 - Real.cos θ ^ 5)

-- Define the solution set
def SolutionSet : Set ℝ := 
  { θ | (-Real.pi ≤ θ ∧ θ < -3 * Real.pi / 4) ∨ (Real.pi / 4 < θ ∧ θ ≤ Real.pi) }

-- Theorem statement
theorem theta_range_theorem :
  ∀ θ ∈ ValidTheta, InequalityCondition θ ↔ θ ∈ SolutionSet :=
sorry

end theta_range_theorem_l2792_279206


namespace room_length_calculation_l2792_279247

theorem room_length_calculation (area : Real) (width : Real) (length : Real) :
  area = 10 ∧ width = 2 ∧ area = length * width → length = 5 := by
  sorry

end room_length_calculation_l2792_279247


namespace cake_mass_proof_l2792_279267

/-- The original mass of the cake in grams -/
def original_mass : ℝ := 750

/-- The mass of cake eaten by Carlson as a fraction -/
def carlson_fraction : ℝ := 0.4

/-- The mass of cake eaten by Little Man in grams -/
def little_man_mass : ℝ := 150

/-- The fraction of remaining cake eaten by Freken Bok -/
def freken_bok_fraction : ℝ := 0.3

/-- The additional mass of cake eaten by Freken Bok in grams -/
def freken_bok_additional : ℝ := 120

/-- The mass of cake crumbs eaten by Matilda in grams -/
def matilda_crumbs : ℝ := 90

theorem cake_mass_proof :
  let remaining_after_carlson := original_mass * (1 - carlson_fraction)
  let remaining_after_little_man := remaining_after_carlson - little_man_mass
  let remaining_after_freken_bok := remaining_after_little_man * (1 - freken_bok_fraction) - freken_bok_additional
  remaining_after_freken_bok = matilda_crumbs :=
by sorry

end cake_mass_proof_l2792_279267


namespace courtyard_ratio_l2792_279264

/-- Given a courtyard with trees, stones, and birds, prove the ratio of trees to stones -/
theorem courtyard_ratio (stones birds : ℕ) (h1 : stones = 40) (h2 : birds = 400)
  (h3 : birds = 2 * (trees + stones)) : (trees : ℚ) / stones = 4 / 1 :=
by
  sorry

end courtyard_ratio_l2792_279264


namespace inequality_proof_l2792_279266

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 0) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := by
  sorry

end inequality_proof_l2792_279266


namespace fraction_simplification_l2792_279236

theorem fraction_simplification (b y : ℝ) (h : b^2 ≠ y^2) :
  (Real.sqrt (b^2 + y^2) + (y^2 - b^2) / Real.sqrt (b^2 + y^2)) / (b^2 - y^2) = (b^2 + y^2) / (b^2 - y^2) := by
  sorry

end fraction_simplification_l2792_279236


namespace number_of_children_l2792_279200

theorem number_of_children (B C : ℕ) : 
  B = 2 * C →
  B = 4 * (C - 160) →
  C = 320 := by
sorry

end number_of_children_l2792_279200


namespace dale_max_nuts_l2792_279239

/-- The maximum number of nuts Dale can guarantee to get -/
def max_nuts_dale : ℕ := 71

/-- The total number of nuts -/
def total_nuts : ℕ := 1001

/-- The number of initial piles -/
def initial_piles : ℕ := 3

/-- The number of possible pile configurations -/
def pile_configs : ℕ := 8

theorem dale_max_nuts :
  ∀ (a b c : ℕ) (N : ℕ),
  a + b + c = total_nuts →
  1 ≤ N ∧ N ≤ total_nuts →
  (∃ (moved : ℕ), moved ≤ max_nuts_dale ∧
    (N = 0 ∨ N = a ∨ N = b ∨ N = c ∨ N = a + b ∨ N = b + c ∨ N = c + a ∨ N = total_nuts ∨
     (N < total_nuts ∧ moved = N - min N (min a (min b (min c (min (a + b) (min (b + c) (c + a))))))) ∨
     (N > 0 ∧ moved = min (a - N) (min (b - N) (min (c - N) (min (a + b - N) (min (b + c - N) (c + a - N)))))))) :=
by sorry

end dale_max_nuts_l2792_279239


namespace phone_price_reduction_l2792_279243

theorem phone_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 2000)
  (h2 : final_price = 1280)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : x > 0 ∧ x < 1) :
  x = 0.18 := by sorry

end phone_price_reduction_l2792_279243


namespace game_winnable_iff_game_not_winnable_equal_game_winnable_greater_l2792_279244

/-- Represents a winning strategy for the card game -/
structure WinningStrategy (n k : ℕ) :=
  (moves : ℕ)
  (strategy : Unit)  -- Placeholder for the actual strategy

/-- The existence of a winning strategy for the card game -/
def winnable (n k : ℕ) : Prop :=
  ∃ (s : WinningStrategy n k), true

/-- Main theorem: The game is winnable if and only if n > k, given n ≥ k ≥ 2 -/
theorem game_winnable_iff (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 2) :
  winnable n k ↔ n > k :=
sorry

/-- The game is not winnable when n = k -/
theorem game_not_winnable_equal (n : ℕ) (h : n ≥ 2) :
  ¬ winnable n n :=
sorry

/-- The game is winnable when n > k -/
theorem game_winnable_greater (n k : ℕ) (h1 : n > k) (h2 : k ≥ 2) :
  winnable n k :=
sorry

end game_winnable_iff_game_not_winnable_equal_game_winnable_greater_l2792_279244


namespace infinite_numbers_with_equal_digit_sum_l2792_279245

/-- Given a natural number, returns the sum of its digits in decimal representation -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a natural number contains the digit 0 in its decimal representation -/
def contains_zero (n : ℕ) : Prop := sorry

theorem infinite_numbers_with_equal_digit_sum (k : ℕ) :
  ∃ (T : Set ℕ), Set.Infinite T ∧ ∀ t ∈ T,
    ¬contains_zero t ∧ sum_of_digits t = sum_of_digits (k * t) := by
  sorry

end infinite_numbers_with_equal_digit_sum_l2792_279245


namespace max_value_sum_cubes_fourth_powers_l2792_279298

theorem max_value_sum_cubes_fourth_powers (a b c : ℕ+) 
  (h : a + b + c = 2) : 
  (∀ x y z : ℕ+, x + y + z = 2 → a + b^3 + c^4 ≥ x + y^3 + z^4) ∧ 
  (∃ x y z : ℕ+, x + y + z = 2 ∧ a + b^3 + c^4 = x + y^3 + z^4) :=
sorry

end max_value_sum_cubes_fourth_powers_l2792_279298


namespace range_of_m_l2792_279252

theorem range_of_m (x y m : ℝ) (h1 : x^2 + 4*y^2*(m^2 + 3*m)*x*y = 0) (h2 : x*y ≠ 0) : -4 < m ∧ m < 1 := by
  sorry

end range_of_m_l2792_279252


namespace solve_system_equations_solve_system_inequalities_l2792_279270

-- Part 1: System of Equations
theorem solve_system_equations :
  ∃! (x y : ℝ), x - 2*y = 1 ∧ 3*x + 4*y = 9 ∧ x = 2.2 ∧ y = 0.6 :=
by sorry

-- Part 2: System of Inequalities
theorem solve_system_inequalities :
  ∀ x : ℝ, ((x - 3) / 2 + 3 ≥ x + 1 ∧ 1 - 3*(x - 1) < 8 - x) ↔ (-2 < x ∧ x ≤ 1) :=
by sorry

end solve_system_equations_solve_system_inequalities_l2792_279270


namespace cookie_jar_final_amount_l2792_279287

theorem cookie_jar_final_amount : 
  let initial_amount : ℚ := 21
  let doris_spent : ℚ := 6
  let martha_spent : ℚ := doris_spent / 2
  let john_added : ℚ := 10
  let john_spent_percentage : ℚ := 1 / 4
  let final_amount : ℚ := 
    (initial_amount - doris_spent - martha_spent + john_added) * 
    (1 - john_spent_percentage)
  final_amount = 33 / 2 := by sorry

end cookie_jar_final_amount_l2792_279287


namespace marble_jar_problem_l2792_279223

theorem marble_jar_problem (M : ℕ) : 
  (∀ (x : ℕ), x = M / 16 → x - 1 = M / 18) → M = 144 := by
  sorry

end marble_jar_problem_l2792_279223


namespace proposition_1_proposition_2_l2792_279212

-- Proposition 1
theorem proposition_1 (x y : ℝ) :
  (xy = 0 → x = 0 ∨ y = 0) ↔
  ((x = 0 ∨ y = 0) → xy = 0) ∧
  (xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧
  ((x ≠ 0 ∧ y ≠ 0) → xy ≠ 0) :=
sorry

-- Proposition 2
theorem proposition_2 (x y : ℝ) :
  ((x > 0 ∧ y > 0) → xy > 0) ↔
  (xy > 0 → x > 0 ∧ y > 0) ∧
  ((x ≤ 0 ∨ y ≤ 0) → xy ≤ 0) ∧
  (xy ≤ 0 → x ≤ 0 ∨ y ≤ 0) :=
sorry

end proposition_1_proposition_2_l2792_279212


namespace min_overlap_coffee_tea_l2792_279221

theorem min_overlap_coffee_tea (coffee_drinkers tea_drinkers : ℝ) 
  (h1 : coffee_drinkers = 0.85)
  (h2 : tea_drinkers = 0.80) :
  0.65 ≤ coffee_drinkers + tea_drinkers - 1 :=
sorry

end min_overlap_coffee_tea_l2792_279221


namespace tree_planting_problem_l2792_279216

-- Define the types for our numbers
def ThreeDigitNumber := { n : ℕ // n ≥ 100 ∧ n < 1000 }
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n < 100 }

-- Function to reverse digits of a number
def reverseDigits (n : ℕ) : ℕ :=
  let rec aux (n acc : ℕ) : ℕ :=
    if n = 0 then acc
    else aux (n / 10) (acc * 10 + n % 10)
  aux n 0

-- Define our theorem
theorem tree_planting_problem 
  (poplars : ThreeDigitNumber) 
  (lindens : TwoDigitNumber) 
  (h1 : poplars.val + lindens.val = 144)
  (h2 : reverseDigits poplars.val + reverseDigits lindens.val = 603) :
  poplars.val = 105 ∧ lindens.val = 39 := by
  sorry


end tree_planting_problem_l2792_279216


namespace min_value_reciprocal_sum_min_value_reciprocal_sum_achieved_l2792_279253

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_value_reciprocal_sum_achieved (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 1 ∧ (1 / m₀ + 1 / n₀ = 3 + 2 * Real.sqrt 2) := by
  sorry

end min_value_reciprocal_sum_min_value_reciprocal_sum_achieved_l2792_279253


namespace complement_of_A_l2792_279258

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 1}

theorem complement_of_A : Set.compl A = {x : ℝ | x ≥ 1} := by sorry

end complement_of_A_l2792_279258


namespace happy_boys_count_l2792_279255

theorem happy_boys_count (total_children happy_children sad_children neutral_children
                          total_boys total_girls sad_girls neutral_boys : ℕ)
                         (h1 : total_children = 60)
                         (h2 : happy_children = 30)
                         (h3 : sad_children = 10)
                         (h4 : neutral_children = 20)
                         (h5 : total_boys = 17)
                         (h6 : total_girls = 43)
                         (h7 : sad_girls = 4)
                         (h8 : neutral_boys = 5)
                         (h9 : total_children = happy_children + sad_children + neutral_children)
                         (h10 : total_children = total_boys + total_girls) :
  total_boys - (sad_children - sad_girls) - neutral_boys = 6 := by
  sorry

end happy_boys_count_l2792_279255


namespace range_of_m_l2792_279256

def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m (m : ℝ) :
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 3) :=
sorry

end range_of_m_l2792_279256


namespace triangle_angle_C_l2792_279282

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if A = π/6, a = 1, and b = √3, then C = π/2 -/
theorem triangle_angle_C (A B C a b c : Real) : 
  A = π/6 → a = 1 → b = Real.sqrt 3 → 
  a / Real.sin A = b / Real.sin B →
  A + B + C = π →
  C = π/2 := by sorry

end triangle_angle_C_l2792_279282


namespace wilsons_theorem_l2792_279260

theorem wilsons_theorem (p : ℕ) (hp : p > 1) :
  Nat.Prime p ↔ (Nat.factorial (p - 1) % p = p - 1) := by
  sorry

end wilsons_theorem_l2792_279260


namespace smallest_n_for_terminating_decimal_l2792_279299

/-- A fraction a/b is a terminating decimal if b can be written as 2^m * 5^n for some non-negative integers m and n. -/
def IsTerminatingDecimal (a b : ℕ) : Prop :=
  ∃ (m n : ℕ), b = 2^m * 5^n

/-- The smallest positive integer n such that n/(n+107) is a terminating decimal is 143. -/
theorem smallest_n_for_terminating_decimal : 
  (∀ k : ℕ, 0 < k → k < 143 → ¬ IsTerminatingDecimal k (k + 107)) ∧ 
  IsTerminatingDecimal 143 (143 + 107) := by
  sorry

#check smallest_n_for_terminating_decimal

end smallest_n_for_terminating_decimal_l2792_279299


namespace inequality_proof_l2792_279213

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 3) :
  (1 / (5 * a^2 - 4 * a + 1)) + (1 / (5 * b^2 - 4 * b + 1)) + (1 / (5 * c^2 - 4 * c + 1)) ≤ 1/4 :=
by sorry

end inequality_proof_l2792_279213


namespace last_digit_322_369_l2792_279240

theorem last_digit_322_369 : (322^369) % 10 = 2 := by sorry

end last_digit_322_369_l2792_279240


namespace complex_number_in_first_quadrant_l2792_279278

theorem complex_number_in_first_quadrant : 
  let z : ℂ := 1 / (1 - Complex.I)
  0 < z.re ∧ 0 < z.im :=
by sorry

end complex_number_in_first_quadrant_l2792_279278


namespace archer_fish_catch_l2792_279279

def fish_problem (first_round : ℕ) (second_round_increase : ℕ) (third_round_percentage : ℕ) : Prop :=
  let second_round := first_round + second_round_increase
  let third_round := second_round + (second_round * third_round_percentage) / 100
  let total_fish := first_round + second_round + third_round
  total_fish = 60

theorem archer_fish_catch :
  fish_problem 8 12 60 :=
sorry

end archer_fish_catch_l2792_279279


namespace smallest_four_digit_geometric_even_l2792_279286

def is_geometric_sequence (a b c d : ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

def digits_are_distinct (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

theorem smallest_four_digit_geometric_even :
  ∀ n : ℕ,
    1000 ≤ n ∧ n < 10000 ∧
    digits_are_distinct n ∧
    is_geometric_sequence (n / 1000) ((n / 100) % 10) ((n / 10) % 10) (n % 10) ∧
    Even n →
    n ≥ 1248 :=
sorry

end smallest_four_digit_geometric_even_l2792_279286


namespace stating_locus_of_vertex_c_l2792_279290

/-- Represents a triangle ABC with specific properties -/
structure SpecialTriangle where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of median from A to BC -/
  median_a : ℝ
  /-- Length of altitude from A to BC -/
  altitude_a : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  /-- Center of the circle -/
  center : ℝ × ℝ
  /-- Radius of the circle -/
  radius : ℝ

/-- 
Theorem stating that the locus of vertex C in the special triangle 
is a circle with specific properties
-/
theorem locus_of_vertex_c (t : SpecialTriangle) 
  (h1 : t.ab = 6)
  (h2 : t.median_a = 4)
  (h3 : t.altitude_a = 3) :
  ∃ (c : Circle), 
    c.radius = 3 ∧ 
    c.center.1 = 4 ∧ 
    c.center.2 = 3 ∧
    (∀ (x y : ℝ), (x, y) ∈ {p | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2} ↔ 
      ∃ (triangle : SpecialTriangle), 
        triangle.ab = t.ab ∧ 
        triangle.median_a = t.median_a ∧ 
        triangle.altitude_a = t.altitude_a) :=
sorry

end stating_locus_of_vertex_c_l2792_279290


namespace email_subscription_day_l2792_279211

theorem email_subscription_day :
  ∀ (x : ℕ),
  (x ≤ 30) →
  (20 * x + 25 * (30 - x) = 675) →
  x = 15 :=
by
  sorry

end email_subscription_day_l2792_279211


namespace rhombus_other_diagonal_l2792_279241

/-- Given a rhombus with one diagonal of length 70 meters and an area of 5600 square meters,
    the other diagonal has a length of 160 meters. -/
theorem rhombus_other_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) : 
  d1 = 70 → area = 5600 → area = (d1 * d2) / 2 → d2 = 160 := by
  sorry

end rhombus_other_diagonal_l2792_279241


namespace worker_b_time_l2792_279251

/-- Given workers a, b, and c, and their work rates, prove that b alone takes 6 hours to complete the work. -/
theorem worker_b_time (a b c : ℝ) : 
  a = 1/3 →                -- a can do the work in 3 hours
  b + c = 1/3 →            -- b and c together can do the work in 3 hours
  a + c = 1/2 →            -- a and c together can do the work in 2 hours
  1/b = 6                  -- b alone takes 6 hours to do the work
:= by sorry

end worker_b_time_l2792_279251


namespace cube_equation_solution_l2792_279219

theorem cube_equation_solution : ∃ x : ℝ, (x - 1)^3 = 64 ∧ x = 5 := by
  sorry

end cube_equation_solution_l2792_279219


namespace det_A_plus_three_eq_two_l2792_279214

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 7; 3, 4]

theorem det_A_plus_three_eq_two :
  Matrix.det A + 3 = 2 := by
  sorry

end det_A_plus_three_eq_two_l2792_279214


namespace average_of_three_liquids_l2792_279277

/-- Given the average of water and milk is 94 liters and there are 100 liters of coffee,
    prove that the average of water, milk, and coffee is 96 liters. -/
theorem average_of_three_liquids (water_milk_avg : ℝ) (coffee : ℝ) :
  water_milk_avg = 94 →
  coffee = 100 →
  (2 * water_milk_avg + coffee) / 3 = 96 := by
sorry

end average_of_three_liquids_l2792_279277


namespace distance_to_big_rock_is_4_l2792_279202

/-- Represents the distance to Big Rock in kilometers -/
def distance_to_big_rock : ℝ := sorry

/-- Rower's speed in still water in km/h -/
def rower_speed : ℝ := 6

/-- Current speed to Big Rock in km/h -/
def current_speed_to : ℝ := 2

/-- Current speed from Big Rock in km/h -/
def current_speed_from : ℝ := 3

/-- Rower's speed from Big Rock in km/h -/
def rower_speed_back : ℝ := 7

/-- Total round trip time in hours -/
def total_time : ℝ := 2

theorem distance_to_big_rock_is_4 :
  distance_to_big_rock = 4 ∧
  (distance_to_big_rock / (rower_speed - current_speed_to) +
   distance_to_big_rock / (rower_speed_back - current_speed_from) = total_time) :=
sorry

end distance_to_big_rock_is_4_l2792_279202


namespace product_terminal_zeros_l2792_279273

/-- The number of terminal zeros in a natural number -/
def terminalZeros (n : ℕ) : ℕ := sorry

/-- The product of 50, 480, and 7 -/
def product : ℕ := 50 * 480 * 7

theorem product_terminal_zeros : terminalZeros product = 3 := by sorry

end product_terminal_zeros_l2792_279273


namespace system_solution_l2792_279296

/-- Given a system of linear equations and an additional equation,
    prove that k must equal 2 for the equations to have a common solution. -/
theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, x + y = 5*k ∧ x - y = k ∧ 2*x + 3*y = 24) → k = 2 := by
  sorry

end system_solution_l2792_279296


namespace max_value_sum_of_fractions_l2792_279230

theorem max_value_sum_of_fractions (a b c : ℝ) 
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
  (h_sum : a + b + c = 3) : 
  (a * b) / (a + b + 1) + (a * c) / (a + c + 1) + (b * c) / (b + c + 1) ≤ 3/2 := by
sorry

end max_value_sum_of_fractions_l2792_279230


namespace vector_operation_l2792_279224

def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-3, -4)

theorem vector_operation : 
  (2 : ℝ) • a - b = (-1, 6) := by sorry

end vector_operation_l2792_279224


namespace sqrt_expression_equals_two_l2792_279220

theorem sqrt_expression_equals_two :
  Real.sqrt 4 + Real.sqrt 2 * Real.sqrt 6 - 6 * Real.sqrt (1/3) = 2 := by
  sorry

end sqrt_expression_equals_two_l2792_279220


namespace three_number_average_l2792_279261

theorem three_number_average : 
  ∀ (x y z : ℝ),
  y = 2 * x →
  z = 4 * y →
  x = 45 →
  (x + y + z) / 3 = 165 := by
sorry

end three_number_average_l2792_279261


namespace sqrt_12_bounds_l2792_279274

theorem sqrt_12_bounds : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end sqrt_12_bounds_l2792_279274


namespace rotated_angle_measure_l2792_279238

/-- Given an initial angle of 50 degrees that is rotated 540 degrees clockwise,
    the resulting new acute angle is also 50 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) (h1 : initial_angle = 50)
    (h2 : rotation = 540) : 
    (initial_angle + rotation) % 360 = 50 ∨ 
    (360 - (initial_angle + rotation) % 360) = 50 := by
  sorry

end rotated_angle_measure_l2792_279238


namespace childrens_book_balances_weights_l2792_279215

/-- Represents a two-arm scale with items on both sides -/
structure TwoArmScale where
  left_side : ℝ
  right_side : ℝ

/-- Checks if the scale is balanced -/
def is_balanced (scale : TwoArmScale) : Prop :=
  scale.left_side = scale.right_side

/-- The weight of the children's book -/
def childrens_book_weight : ℝ := 1.1

/-- The combined weight of the weights on the right side of the scale -/
def right_side_weight : ℝ := 0.5 + 0.3 + 0.3

/-- Theorem stating that the children's book weight balances the given weights -/
theorem childrens_book_balances_weights :
  is_balanced { left_side := childrens_book_weight, right_side := right_side_weight } :=
by sorry

end childrens_book_balances_weights_l2792_279215


namespace homework_difference_l2792_279203

def math_pages : ℕ := 5
def reading_pages : ℕ := 2

theorem homework_difference : math_pages - reading_pages = 3 := by
  sorry

end homework_difference_l2792_279203


namespace square_area_from_oblique_projection_l2792_279293

/-- Represents a square in 2D space -/
structure Square where
  side_length : ℝ
  area : ℝ := side_length ^ 2

/-- Represents a parallelogram in 2D space -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ

/-- Represents an oblique projection transformation -/
def obliqueProjection (s : Square) : Parallelogram :=
  sorry

theorem square_area_from_oblique_projection 
  (s : Square) 
  (p : Parallelogram) 
  (h1 : p = obliqueProjection s) 
  (h2 : p.side1 = 4 ∨ p.side2 = 4) : 
  s.area = 16 ∨ s.area = 64 := by
  sorry

end square_area_from_oblique_projection_l2792_279293


namespace pencil_price_l2792_279229

theorem pencil_price (joy_pencils colleen_pencils : ℕ) (price_difference : ℚ) :
  joy_pencils = 30 →
  colleen_pencils = 50 →
  price_difference = 80 →
  ∃ (price : ℚ), 
    colleen_pencils * price = joy_pencils * price + price_difference ∧
    price = 4 := by
  sorry

end pencil_price_l2792_279229


namespace expression_equals_one_eighth_l2792_279227

theorem expression_equals_one_eighth :
  let a := 404445
  let b := 202222
  let c := 202223
  let d := 202224
  let e := 12639
  (a^2 / (b * c * d) - c / (b * d) - b / (c * d)) * e = 1/8 := by sorry

end expression_equals_one_eighth_l2792_279227


namespace volleyball_club_girls_count_l2792_279231

theorem volleyball_club_girls_count :
  ∀ (total_members : ℕ) (meeting_attendees : ℕ) (girls : ℕ) (boys : ℕ),
    total_members = 32 →
    meeting_attendees = 20 →
    total_members = girls + boys →
    meeting_attendees = boys + girls / 3 →
    girls = 18 := by
  sorry

end volleyball_club_girls_count_l2792_279231


namespace larger_number_proof_l2792_279222

theorem larger_number_proof (x y : ℕ) 
  (h1 : y - x = 1360) 
  (h2 : y = 6 * x + 15) : 
  y = 1629 := by
sorry

end larger_number_proof_l2792_279222
