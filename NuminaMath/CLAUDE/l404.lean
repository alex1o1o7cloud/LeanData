import Mathlib

namespace NUMINAMATH_CALUDE_well_depth_proof_well_depth_l404_40411

theorem well_depth_proof (total_time : ℝ) (stone_fall_law : ℝ → ℝ) (sound_velocity : ℝ) : ℝ :=
  let depth := 2000
  let stone_fall_time := Real.sqrt (depth / 20)
  let sound_travel_time := depth / sound_velocity
  
  by
    have h1 : total_time = 10 := by sorry
    have h2 : ∀ t, stone_fall_law t = 20 * t^2 := by sorry
    have h3 : sound_velocity = 1120 := by sorry
    have h4 : stone_fall_time + sound_travel_time = total_time := by sorry
    
    -- The proof would go here
    sorry

-- The theorem states that given the conditions, the depth of the well is 2000 feet
theorem well_depth : well_depth_proof 10 (λ t => 20 * t^2) 1120 = 2000 := by sorry

end NUMINAMATH_CALUDE_well_depth_proof_well_depth_l404_40411


namespace NUMINAMATH_CALUDE_trapezoid_AD_length_l404_40407

/-- Represents a trapezoid ABCD with point M on CD and perpendicular AH from A to BM -/
structure Trapezoid :=
  (A B C D M H : ℝ × ℝ)
  (is_parallel : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1))
  (M_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • C + t • D)
  (AH_perpendicular : (H.2 - A.2) * (M.1 - B.1) + (H.1 - A.1) * (M.2 - B.2) = 0)
  (AD_eq_HD : dist A D = dist H D)
  (BC_length : dist B C = 16)
  (CM_length : dist C M = 8)
  (MD_length : dist M D = 9)

/-- The length of AD in the given trapezoid is 18 -/
theorem trapezoid_AD_length (t : Trapezoid) : dist t.A t.D = 18 :=
  sorry

end NUMINAMATH_CALUDE_trapezoid_AD_length_l404_40407


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l404_40478

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 3 * y = 11 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l404_40478


namespace NUMINAMATH_CALUDE_marble_count_l404_40438

/-- The number of marbles in a bag satisfying certain conditions -/
theorem marble_count : ∀ (total yellow green red blue : ℕ),
  yellow = 20 →
  green = yellow / 2 →
  red = blue →
  blue = total / 4 →
  total = yellow + green + red + blue →
  total = 60 := by
  sorry

#check marble_count

end NUMINAMATH_CALUDE_marble_count_l404_40438


namespace NUMINAMATH_CALUDE_expansion_terms_count_l404_40405

/-- The number of terms in the simplified expansion of (x+y+z)^2010 + (x-y-z)^2010 -/
def num_terms : ℕ := 1012036

/-- The exponent in the expression (x+y+z)^n + (x-y-z)^n -/
def exponent : ℕ := 2010

theorem expansion_terms_count :
  num_terms = (exponent / 2 + 1)^2 := by sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l404_40405


namespace NUMINAMATH_CALUDE_multiplication_equality_l404_40429

theorem multiplication_equality : 62519 * 9999 = 625127481 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equality_l404_40429


namespace NUMINAMATH_CALUDE_fish_sales_profit_maximization_l404_40494

-- Define the linear relationship between y and x
def linear_relationship (k b x : ℝ) : ℝ := k * x + b

-- Define the daily sales profit function
def daily_sales_profit (x : ℝ) : ℝ := (x - 30) * (linear_relationship (-10) 600 x)

theorem fish_sales_profit_maximization :
  -- Given conditions
  let y₁ := linear_relationship (-10) 600 50
  let y₂ := linear_relationship (-10) 600 40
  -- Theorem statements
  y₁ = 100 ∧
  y₂ = 200 ∧
  (∀ x : ℝ, 30 ≤ x → x < 60 → daily_sales_profit x ≤ daily_sales_profit 45) ∧
  daily_sales_profit 45 = 2250 :=
by sorry


end NUMINAMATH_CALUDE_fish_sales_profit_maximization_l404_40494


namespace NUMINAMATH_CALUDE_kaleb_video_games_l404_40484

def video_game_problem (non_working_games : ℕ) (total_earnings : ℕ) (price_per_game : ℕ) : Prop :=
  let working_games := total_earnings / price_per_game
  working_games + non_working_games = 10

theorem kaleb_video_games :
  video_game_problem 8 12 6 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_video_games_l404_40484


namespace NUMINAMATH_CALUDE_circle_tangent_to_hyperbola_radius_l404_40403

/-- The radius of a circle tangent to a hyperbola --/
theorem circle_tangent_to_hyperbola_radius 
  (a b : ℝ) 
  (h_a : a = 6) 
  (h_b : b = 5) : 
  let c := Real.sqrt (a^2 + b^2)
  let r := |a - c|
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (x - c)^2 + y^2 ≥ r^2) ∧ 
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ (x - c)^2 + y^2 = r^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_hyperbola_radius_l404_40403


namespace NUMINAMATH_CALUDE_gcd_problem_l404_40445

theorem gcd_problem (h1 : Nat.Prime 361) 
                    (h2 : 172 = 2 * 2 * 43) 
                    (h3 : 473 = 43 * 11) 
                    (h4 : 360 = 4 * 90) : 
  Nat.gcd (360 * 473) (172 * 361) = 172 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l404_40445


namespace NUMINAMATH_CALUDE_angle_function_value_l404_40427

open Real

/-- Given an angle α in the third quadrant and f(α) = (cos(π/2 + α) * cos(π - α)) / sin(π + α),
    if cos(α - 3π/2) = 1/5, then f(α) = 2√6/5 -/
theorem angle_function_value (α : ℝ) :
  π < α ∧ α < 3*π/2 →
  cos (α - 3*π/2) = 1/5 →
  (cos (π/2 + α) * cos (π - α)) / sin (π + α) = 2*Real.sqrt 6/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_function_value_l404_40427


namespace NUMINAMATH_CALUDE_combine_terms_power_l404_40440

/-- Given that two terms can be combined, prove that m^n = 8 -/
theorem combine_terms_power (a b c m n : ℕ) : 
  (∃ k : ℚ, k * a^m * b^3 * c^4 = -3 * a^2 * b^n * c^4) → m^n = 8 := by
sorry

end NUMINAMATH_CALUDE_combine_terms_power_l404_40440


namespace NUMINAMATH_CALUDE_salmon_migration_multiple_l404_40420

/-- 
Given an initial number of salmons and the current number of salmons in a river,
calculate the multiple of the initial number that migrated to the river.
-/
theorem salmon_migration_multiple (initial : ℕ) (current : ℕ) : 
  initial = 500 → current = 5500 → (current - initial) / initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_salmon_migration_multiple_l404_40420


namespace NUMINAMATH_CALUDE_divisibility_by_37_l404_40424

theorem divisibility_by_37 (a b c d e f : ℕ) :
  (a < 10) → (b < 10) → (c < 10) → (d < 10) → (e < 10) → (f < 10) →
  (37 ∣ (100 * a + 10 * b + c + 100 * d + 10 * e + f)) →
  (37 ∣ (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f)) :=
by sorry

#check divisibility_by_37

end NUMINAMATH_CALUDE_divisibility_by_37_l404_40424


namespace NUMINAMATH_CALUDE_yellow_packs_count_l404_40459

/-- The number of packs of yellow bouncy balls Jill bought -/
def yellow_packs : ℕ := sorry

/-- The number of packs of red bouncy balls Jill bought -/
def red_packs : ℕ := 5

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 18

theorem yellow_packs_count : yellow_packs = 4 := by
  have h1 : red_packs * balls_per_pack = yellow_packs * balls_per_pack + 18 := by sorry
  sorry

end NUMINAMATH_CALUDE_yellow_packs_count_l404_40459


namespace NUMINAMATH_CALUDE_wire_bending_l404_40486

/-- Given a wire that can be bent into either a circle or a square, 
    if the area of the square is 7737.769850454057 cm², 
    then the radius of the circle is approximately 56 cm. -/
theorem wire_bending (square_area : ℝ) (circle_radius : ℝ) : 
  square_area = 7737.769850454057 → 
  (circle_radius ≥ 55.99 ∧ circle_radius ≤ 56.01) := by
  sorry

#check wire_bending

end NUMINAMATH_CALUDE_wire_bending_l404_40486


namespace NUMINAMATH_CALUDE_same_terminal_side_l404_40408

/-- Proves that -2π/3 has the same terminal side as 240° --/
theorem same_terminal_side : ∃ (k : ℤ), -2 * π / 3 = 240 * π / 180 + 2 * k * π := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l404_40408


namespace NUMINAMATH_CALUDE_line_slope_l404_40462

/-- Given a line l and two points on it, prove that the slope of the line is -3 -/
theorem line_slope (l : Set (ℝ × ℝ)) (a b : ℝ) 
  (h1 : (a, b) ∈ l) 
  (h2 : (a + 1, b - 3) ∈ l) : 
  ∃ (m : ℝ), m = -3 ∧ ∀ (x y : ℝ), (x, y) ∈ l → y = m * (x - a) + b :=
sorry

end NUMINAMATH_CALUDE_line_slope_l404_40462


namespace NUMINAMATH_CALUDE_jim_tire_repairs_l404_40433

/-- Represents the financial details of Jim's bike shop for a month. -/
structure BikeShopFinances where
  tireFee : ℕ           -- Fee charged for fixing a bike tire
  tireCost : ℕ          -- Cost of parts for fixing a bike tire
  complexRepairs : ℕ    -- Number of complex repairs
  complexFee : ℕ        -- Fee charged for a complex repair
  complexCost : ℕ       -- Cost of parts for a complex repair
  retailProfit : ℕ      -- Profit from retail sales
  fixedExpenses : ℕ     -- Monthly fixed expenses
  totalProfit : ℕ       -- Total profit for the month

/-- Calculates the number of bike tire repairs given the shop's finances. -/
def calculateTireRepairs (finances : BikeShopFinances) : ℕ :=
  let tireProfit := finances.tireFee - finances.tireCost
  let complexProfit := finances.complexRepairs * (finances.complexFee - finances.complexCost)
  (finances.totalProfit + finances.fixedExpenses - finances.retailProfit - complexProfit) / tireProfit

/-- Theorem stating that Jim does 300 bike tire repairs in a month. -/
theorem jim_tire_repairs : 
  let finances : BikeShopFinances := {
    tireFee := 20
    tireCost := 5
    complexRepairs := 2
    complexFee := 300
    complexCost := 50
    retailProfit := 2000
    fixedExpenses := 4000
    totalProfit := 3000
  }
  calculateTireRepairs finances = 300 := by
  sorry


end NUMINAMATH_CALUDE_jim_tire_repairs_l404_40433


namespace NUMINAMATH_CALUDE_power_2m_equals_half_l404_40483

theorem power_2m_equals_half (a m n : ℝ) 
  (h1 : a^(m+n) = 1/4)
  (h2 : a^(m-n) = 2)
  (h3 : a > 0)
  (h4 : a ≠ 1) :
  a^(2*m) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_power_2m_equals_half_l404_40483


namespace NUMINAMATH_CALUDE_karinas_brother_birth_year_l404_40463

/-- Proves the birth year of Karina's brother given the conditions of the problem -/
theorem karinas_brother_birth_year 
  (karina_birth_year : ℕ) 
  (karina_current_age : ℕ) 
  (h1 : karina_birth_year = 1970)
  (h2 : karina_current_age = 40)
  (h3 : karina_current_age = 2 * (karina_current_age - (karina_birth_year - brother_birth_year)))
  : brother_birth_year = 1990 := by
  sorry


end NUMINAMATH_CALUDE_karinas_brother_birth_year_l404_40463


namespace NUMINAMATH_CALUDE_a_greater_than_b_squared_l404_40456

theorem a_greater_than_b_squared (a b : ℝ) (ha : a > 1) (hb1 : b > -1) (hb2 : 1 > b) : a > b^2 := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_squared_l404_40456


namespace NUMINAMATH_CALUDE_second_number_is_068_l404_40469

/-- Represents a random number table as a list of natural numbers -/
def RandomNumberTable : List ℕ := [84, 42, 17, 53, 31, 57, 24, 55, 6, 88, 77, 4, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 6, 76]

/-- The total number of restaurants -/
def TotalRestaurants : ℕ := 200

/-- The number of restaurants to be selected -/
def SelectedRestaurants : ℕ := 5

/-- The starting column in the random number table -/
def StartColumn : ℕ := 5

/-- Function to select numbers from the random number table -/
def selectNumbers (table : List ℕ) (start : ℕ) (count : ℕ) : List ℕ :=
  (table.drop start).take count

/-- Theorem stating that the second selected number is 068 -/
theorem second_number_is_068 : 
  (selectNumbers RandomNumberTable StartColumn SelectedRestaurants)[1] = 68 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_068_l404_40469


namespace NUMINAMATH_CALUDE_nth_equation_proof_l404_40418

theorem nth_equation_proof (n : ℕ) : (((n + 3)^2 - n^2 - 9) / 2 : ℚ) = 3 * n := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_proof_l404_40418


namespace NUMINAMATH_CALUDE_max_value_linear_program_l404_40475

/-- Given a set of linear constraints, prove that the maximum value of the objective function is 2 -/
theorem max_value_linear_program :
  ∀ x y : ℝ,
  x + y ≥ 1 →
  2 * x - y ≤ 0 →
  3 * x - 2 * y + 2 ≥ 0 →
  (∀ x' y' : ℝ,
    x' + y' ≥ 1 →
    2 * x' - y' ≤ 0 →
    3 * x' - 2 * y' + 2 ≥ 0 →
    3 * x - y ≥ 3 * x' - y') →
  3 * x - y = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_linear_program_l404_40475


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l404_40400

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 36) (h2 : x = 28) : x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l404_40400


namespace NUMINAMATH_CALUDE_tuesday_most_available_l404_40485

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

-- Define the people
inductive Person
  | Anna
  | Bill
  | Carl
  | Dave
  | Eve

-- Define the availability function
def isAvailable (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Tuesday => true
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Thursday => true
  | Person.Anna, Day.Friday => true
  | Person.Bill, Day.Monday => true
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Wednesday => true
  | Person.Bill, Day.Thursday => false
  | Person.Bill, Day.Friday => false
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Wednesday => true
  | Person.Carl, Day.Thursday => false
  | Person.Carl, Day.Friday => false
  | Person.Dave, Day.Monday => true
  | Person.Dave, Day.Tuesday => true
  | Person.Dave, Day.Wednesday => false
  | Person.Dave, Day.Thursday => true
  | Person.Dave, Day.Friday => false
  | Person.Eve, Day.Monday => false
  | Person.Eve, Day.Tuesday => true
  | Person.Eve, Day.Wednesday => true
  | Person.Eve, Day.Thursday => false
  | Person.Eve, Day.Friday => true

-- Count available people for a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (fun p => isAvailable p d) [Person.Anna, Person.Bill, Person.Carl, Person.Dave, Person.Eve]).length

-- Define the theorem
theorem tuesday_most_available :
  ∀ d : Day, d ≠ Day.Tuesday → countAvailable Day.Tuesday ≥ countAvailable d :=
sorry

end NUMINAMATH_CALUDE_tuesday_most_available_l404_40485


namespace NUMINAMATH_CALUDE_a_equals_two_sufficient_not_necessary_l404_40432

theorem a_equals_two_sufficient_not_necessary (a : ℝ) :
  (a = 2 → |a| = 2) ∧ (∃ b : ℝ, b ≠ 2 ∧ |b| = 2) := by
  sorry

end NUMINAMATH_CALUDE_a_equals_two_sufficient_not_necessary_l404_40432


namespace NUMINAMATH_CALUDE_ball_distribution_l404_40474

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The number of ways to choose m items from n items -/
def choose (n : ℕ) (m : ℕ) : ℕ :=
  sorry

theorem ball_distribution :
  let total_balls : ℕ := 6
  let num_boxes : ℕ := 4
  distribute_balls total_balls num_boxes = choose num_boxes 2 + num_boxes := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_l404_40474


namespace NUMINAMATH_CALUDE_product_divide_theorem_l404_40417

theorem product_divide_theorem : (3.6 * 0.3) / 0.6 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_product_divide_theorem_l404_40417


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2050_l404_40497

theorem units_digit_of_7_power_2050 : (7^2050 : ℕ) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2050_l404_40497


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_a_value_l404_40471

-- Define the vectors m and n
def m : ℝ × ℝ := (-2, 3)
def n (a : ℝ) : ℝ × ℝ := (a + 1, 3)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2 = k * w.2

-- Theorem statement
theorem parallel_vectors_imply_a_value :
  parallel m (n a) → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_a_value_l404_40471


namespace NUMINAMATH_CALUDE_charlie_and_diana_qualify_l404_40401

structure Person :=
  (name : String)
  (qualifies : Prop)

def Alice : Person := ⟨"Alice", sorry⟩
def Bob : Person := ⟨"Bob", sorry⟩
def Charlie : Person := ⟨"Charlie", sorry⟩
def Diana : Person := ⟨"Diana", sorry⟩

def Statements : Prop :=
  (Alice.qualifies → Bob.qualifies) ∧
  (Bob.qualifies → Charlie.qualifies) ∧
  (Charlie.qualifies → (Diana.qualifies ∧ ¬Alice.qualifies))

def ExactlyTwoQualify : Prop :=
  ∃! (p1 p2 : Person), p1 ≠ p2 ∧ p1.qualifies ∧ p2.qualifies ∧
    ∀ (p : Person), p.qualifies → (p = p1 ∨ p = p2)

theorem charlie_and_diana_qualify :
  Statements ∧ ExactlyTwoQualify →
  Charlie.qualifies ∧ Diana.qualifies ∧ ¬Alice.qualifies ∧ ¬Bob.qualifies :=
by sorry

end NUMINAMATH_CALUDE_charlie_and_diana_qualify_l404_40401


namespace NUMINAMATH_CALUDE_solution_of_equation_l404_40480

theorem solution_of_equation (x : ℝ) : (2 / x = 1 / (x + 1)) ↔ (x = -2) := by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l404_40480


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_S_l404_40465

/-- S_n is the sum of the reciprocals of the non-zero digits of the integers from 1 to 10^n inclusive -/
def S (n : ℕ+) : ℚ :=
  sorry

/-- 63 is the smallest positive integer n for which S_n is an integer -/
theorem smallest_n_for_integer_S :
  ∀ k : ℕ+, k < 63 → ¬ (S k).isInt ∧ (S 63).isInt := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_S_l404_40465


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l404_40453

theorem sqrt_fraction_equality (x : ℝ) (h : x > 0) :
  Real.sqrt (x / (1 - (3 * x - 2) / (2 * x))) = Real.sqrt ((2 * x^2) / (2 - x)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l404_40453


namespace NUMINAMATH_CALUDE_reflection_theorem_l404_40416

/-- Represents a 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a reflection plane -/
inductive ReflectionPlane
  | XY
  | YZ
  | ZX

/-- Reflects a vector across a given plane -/
def reflect (v : Vector3D) (plane : ReflectionPlane) : Vector3D :=
  match plane with
  | ReflectionPlane.XY => ⟨v.x, v.y, -v.z⟩
  | ReflectionPlane.YZ => ⟨-v.x, v.y, v.z⟩
  | ReflectionPlane.ZX => ⟨v.x, -v.y, v.z⟩

/-- Reflects a vector across all three coordinate planes -/
def reflectAll (v : Vector3D) : Vector3D :=
  reflect (reflect (reflect v ReflectionPlane.XY) ReflectionPlane.YZ) ReflectionPlane.ZX

theorem reflection_theorem (v : Vector3D) :
  reflectAll v = Vector3D.mk (-v.x) (-v.y) (-v.z) := by
  sorry

#check reflection_theorem

end NUMINAMATH_CALUDE_reflection_theorem_l404_40416


namespace NUMINAMATH_CALUDE_equation_solution_l404_40492

theorem equation_solution (n : ℝ) : 
  1 / (n + 1) + 2 / (n + 2) + n / (n + 3) = 1 ↔ n = 2 + Real.sqrt 10 ∨ n = 2 - Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l404_40492


namespace NUMINAMATH_CALUDE_singer_arrangements_eq_18_l404_40452

/-- The number of different arrangements for 5 singers with restrictions -/
def singer_arrangements : ℕ :=
  let total_singers : ℕ := 5
  let singers_to_arrange : ℕ := total_singers - 1  -- excluding the last singer
  let first_position_choices : ℕ := singers_to_arrange - 1  -- excluding the singer who can't be first
  first_position_choices * Nat.factorial (singers_to_arrange - 1)

theorem singer_arrangements_eq_18 : singer_arrangements = 18 := by
  sorry

end NUMINAMATH_CALUDE_singer_arrangements_eq_18_l404_40452


namespace NUMINAMATH_CALUDE_product_from_gcd_lcm_l404_40410

theorem product_from_gcd_lcm (a b : ℕ+) :
  Nat.gcd a b = 5 → Nat.lcm a b = 60 → a * b = 300 := by
  sorry

end NUMINAMATH_CALUDE_product_from_gcd_lcm_l404_40410


namespace NUMINAMATH_CALUDE_alan_market_expenditure_l404_40422

/-- The total amount Alan spent at the market --/
def total_spent (egg_count : ℕ) (egg_price : ℕ) (chicken_count : ℕ) (chicken_price : ℕ) : ℕ :=
  egg_count * egg_price + chicken_count * chicken_price

/-- Theorem stating that Alan spent $88 at the market --/
theorem alan_market_expenditure :
  total_spent 20 2 6 8 = 88 := by
  sorry

end NUMINAMATH_CALUDE_alan_market_expenditure_l404_40422


namespace NUMINAMATH_CALUDE_similar_triangles_segment_length_l404_40402

/-- Two triangles are similar -/
structure SimilarTriangles (P Q R X Y Z : ℝ × ℝ) : Prop where
  similar : True  -- We don't need to define the full similarity conditions for this problem

/-- The length of a line segment between two points -/
def segmentLength (A B : ℝ × ℝ) : ℝ := sorry

theorem similar_triangles_segment_length 
  (P Q R X Y Z : ℝ × ℝ) 
  (h_similar : SimilarTriangles P Q R X Y Z) 
  (h_PQ : segmentLength P Q = 8)
  (h_QR : segmentLength Q R = 16)
  (h_YZ : segmentLength Y Z = 24) :
  segmentLength X Y = 12 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_segment_length_l404_40402


namespace NUMINAMATH_CALUDE_sqrt_68_minus_sqrt_64_approx_l404_40413

theorem sqrt_68_minus_sqrt_64_approx : |Real.sqrt 68 - Real.sqrt 64 - 0.24| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_68_minus_sqrt_64_approx_l404_40413


namespace NUMINAMATH_CALUDE_common_root_existence_l404_40435

/-- The common rational root of two polynomials -/
def k : ℚ := -3/5

/-- First polynomial -/
def P (x : ℚ) (a b c : ℚ) : ℚ := 90 * x^4 + a * x^3 + b * x^2 + c * x + 15

/-- Second polynomial -/
def Q (x : ℚ) (d e f g : ℚ) : ℚ := 15 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 90

theorem common_root_existence (a b c d e f g : ℚ) :
  (k ≠ 0) ∧ 
  (k < 0) ∧ 
  (∃ (m n : ℤ), k = m / n ∧ m ≠ 0 ∧ n ≠ 0 ∧ Int.gcd m n = 1) ∧
  (P k a b c = 0) ∧ 
  (Q k d e f g = 0) :=
sorry

end NUMINAMATH_CALUDE_common_root_existence_l404_40435


namespace NUMINAMATH_CALUDE_tenisha_dogs_l404_40449

/-- The initial number of dogs Tenisha had -/
def initial_dogs : ℕ := 40

/-- The proportion of female dogs -/
def female_ratio : ℚ := 3/5

/-- The proportion of female dogs that give birth -/
def birth_ratio : ℚ := 3/4

/-- The number of puppies each female dog gives birth to -/
def puppies_per_dog : ℕ := 10

/-- The number of puppies donated -/
def donated_puppies : ℕ := 130

/-- The number of puppies remaining after donation -/
def remaining_puppies : ℕ := 50

theorem tenisha_dogs :
  (initial_dogs : ℚ) * female_ratio * birth_ratio * puppies_per_dog =
  (donated_puppies + remaining_puppies : ℚ) :=
sorry

end NUMINAMATH_CALUDE_tenisha_dogs_l404_40449


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l404_40498

-- Define the sequence b_n
def b (n : ℕ) : ℕ := n.factorial + 2 * n

-- Theorem statement
theorem max_gcd_consecutive_terms :
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (b n) (b (n + 1)) ≤ k ∧
  ∃ (m : ℕ), Nat.gcd (b m) (b (m + 1)) = k :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l404_40498


namespace NUMINAMATH_CALUDE_second_square_area_l404_40499

/-- Represents an isosceles right triangle with inscribed squares -/
structure IsoscelesRightTriangleWithSquares where
  /-- Side length of the first inscribed square -/
  s : ℝ
  /-- Area of the first inscribed square is 484 cm² -/
  first_square_area : s^2 = 484
  /-- Side length of the second inscribed square -/
  x : ℝ
  /-- Relationship between side lengths of the triangle and the second square -/
  triangle_side_relation : 3 * x = 2 * s

/-- The area of the second inscribed square is 1936/9 cm² -/
theorem second_square_area (triangle : IsoscelesRightTriangleWithSquares) :
  triangle.x^2 = 1936 / 9 := by
  sorry

end NUMINAMATH_CALUDE_second_square_area_l404_40499


namespace NUMINAMATH_CALUDE_exp_inequality_l404_40457

theorem exp_inequality (a b : ℝ) (h : a > b) : Real.exp (-a) - Real.exp (-b) < 0 := by
  sorry

end NUMINAMATH_CALUDE_exp_inequality_l404_40457


namespace NUMINAMATH_CALUDE_triangle_cosine_inequality_l404_40482

theorem triangle_cosine_inequality (α β γ : Real) 
  (h_acute : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_sum : α + β + γ = π) : 
  (Real.cos α / Real.cos (β - γ)) + 
  (Real.cos β / Real.cos (γ - α)) + 
  (Real.cos γ / Real.cos (α - β)) ≥ 3/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_inequality_l404_40482


namespace NUMINAMATH_CALUDE_prime_equation_solution_l404_40409

theorem prime_equation_solution :
  ∀ p : ℕ, Prime p →
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔
    (p = 2 ∨ p = 3 ∨ p = 7) :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l404_40409


namespace NUMINAMATH_CALUDE_first_group_size_is_four_l404_40406

/-- The number of men in the first group -/
def first_group_size : ℕ := 4

/-- The length of cloth colored by the first group -/
def first_group_cloth_length : ℝ := 48

/-- The time taken by the first group to color their cloth -/
def first_group_time : ℝ := 2

/-- The number of men in the second group -/
def second_group_size : ℕ := 5

/-- The length of cloth colored by the second group -/
def second_group_cloth_length : ℝ := 36

/-- The time taken by the second group to color their cloth -/
def second_group_time : ℝ := 1.2

theorem first_group_size_is_four :
  first_group_size = 4 :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_is_four_l404_40406


namespace NUMINAMATH_CALUDE_taps_teps_equivalence_l404_40458

/-- Given the equivalences between taps, tops, and teps, prove that 15 taps are equivalent to 48 teps -/
theorem taps_teps_equivalence (tap top tep : ℕ → ℚ) 
  (h1 : 5 * tap 1 = 4 * top 1)  -- 5 taps are equivalent to 4 tops
  (h2 : 3 * top 1 = 12 * tep 1) -- 3 tops are equivalent to 12 teps
  : 15 * tap 1 = 48 * tep 1 := by
  sorry

end NUMINAMATH_CALUDE_taps_teps_equivalence_l404_40458


namespace NUMINAMATH_CALUDE_inequality_proof_l404_40451

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ∧
  (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l404_40451


namespace NUMINAMATH_CALUDE_cycle_price_proof_l404_40441

/-- Proves that given a cycle sold for 1350 with a 50% gain, the original price was 900 -/
theorem cycle_price_proof (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 1350)
  (h2 : gain_percentage = 50) : 
  selling_price / (1 + gain_percentage / 100) = 900 := by
  sorry

#check cycle_price_proof

end NUMINAMATH_CALUDE_cycle_price_proof_l404_40441


namespace NUMINAMATH_CALUDE_max_square_plots_exists_valid_partition_8_largest_num_square_plots_l404_40472

def field_length : ℕ := 30
def field_width : ℕ := 60
def available_fencing : ℕ := 2500

def is_valid_partition (s : ℕ) : Prop :=
  s ∣ field_length ∧ s ∣ field_width ∧
  (field_length / s - 1) * field_width + (field_width / s - 1) * field_length ≤ available_fencing

def num_plots (s : ℕ) : ℕ :=
  (field_length / s) * (field_width / s)

theorem max_square_plots :
  ∀ s : ℕ, is_valid_partition s → num_plots s ≤ 8 :=
by sorry

theorem exists_valid_partition_8 :
  ∃ s : ℕ, is_valid_partition s ∧ num_plots s = 8 :=
by sorry

theorem largest_num_square_plots : 
  (∃ s : ℕ, is_valid_partition s ∧ num_plots s = 8) ∧
  (∀ s : ℕ, is_valid_partition s → num_plots s ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_max_square_plots_exists_valid_partition_8_largest_num_square_plots_l404_40472


namespace NUMINAMATH_CALUDE_student_calculation_greater_than_true_average_l404_40479

theorem student_calculation_greater_than_true_average : 
  ((2 + 4 + 6) / 2 + (8 + 10) / 2) / 2 > (2 + 4 + 6 + 8 + 10) / 5 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_greater_than_true_average_l404_40479


namespace NUMINAMATH_CALUDE_factorization_x_cubed_minus_x_l404_40437

/-- Factorization of x^3 - x --/
theorem factorization_x_cubed_minus_x :
  ∀ x : ℝ, x^3 - x = x * (x + 1) * (x - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_x_cubed_minus_x_l404_40437


namespace NUMINAMATH_CALUDE_no_periodic_sum_l404_40434

/-- A function is periodic if it takes at least two different values and there exists a positive real number p such that f(x + p) = f(x) for all x. -/
def Periodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ ∃ p > 0, ∀ x, f (x + p) = f x

/-- g is a periodic function with period 1 -/
def g : ℝ → ℝ := sorry

/-- h is a periodic function with period π -/
def h : ℝ → ℝ := sorry

/-- g has period 1 -/
axiom g_periodic : Periodic g ∧ ∀ x, g (x + 1) = g x

/-- h has period π -/
axiom h_periodic : Periodic h ∧ ∀ x, h (x + Real.pi) = h x

/-- Theorem: It's not possible to construct non-trivial periodic functions g and h
    with periods 1 and π respectively, such that g + h is also a periodic function -/
theorem no_periodic_sum : ¬(Periodic (g + h)) := by sorry

end NUMINAMATH_CALUDE_no_periodic_sum_l404_40434


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l404_40425

/-- Prove that for a hyperbola with the given properties, its eccentricity is 5/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (P : ℝ × ℝ),
    (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
    (∃ (F₁ F₂ : ℝ × ℝ),
      (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
       Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 3 * b) ∧
      (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) *
       Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 9 * a * b / 4)) →
  Real.sqrt (a^2 + b^2) / a = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l404_40425


namespace NUMINAMATH_CALUDE_f_symmetry_g_zero_l404_40446

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define g as the derivative of f
def g : ℝ → ℝ := f'

-- State the conditions
axiom f_even : ∀ x, f (3/2 - 2*x) = f (3/2 + 2*x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- State the theorems to be proved
theorem f_symmetry : f (-1) = f 4 := by sorry

theorem g_zero : g (-1/2) = 0 := by sorry

end NUMINAMATH_CALUDE_f_symmetry_g_zero_l404_40446


namespace NUMINAMATH_CALUDE_right_triangle_area_arithmetic_sides_l404_40488

/-- A right-angled triangle with area 37.5 m² and sides forming an arithmetic sequence has side lengths 7.5 m, 10 m, and 12.5 m. -/
theorem right_triangle_area_arithmetic_sides (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a < b ∧ b < c →  -- Ordered side lengths
  b - a = c - b →  -- Arithmetic sequence condition
  a^2 + b^2 = c^2 →  -- Pythagorean theorem (right-angled triangle)
  (1/2) * a * b = 37.5 →  -- Area condition
  (a, b, c) = (7.5, 10, 12.5) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_arithmetic_sides_l404_40488


namespace NUMINAMATH_CALUDE_cole_drive_to_work_time_l404_40442

/-- The time taken for Cole to drive to work, given his speeds and total round trip time -/
theorem cole_drive_to_work_time 
  (speed_to_work : ℝ) 
  (speed_from_work : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_to_work = 75) 
  (h2 : speed_from_work = 105) 
  (h3 : total_time = 1) : 
  ∃ (distance : ℝ), 
    distance / speed_to_work + distance / speed_from_work = total_time ∧ 
    (distance / speed_to_work) * 60 = 35 :=
by sorry

end NUMINAMATH_CALUDE_cole_drive_to_work_time_l404_40442


namespace NUMINAMATH_CALUDE_fraction_square_simplification_l404_40450

theorem fraction_square_simplification (a b c : ℝ) (ha : a ≠ 0) :
  (3 * b * c / (-2 * a^2))^2 = 9 * b^2 * c^2 / (4 * a^4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_square_simplification_l404_40450


namespace NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_9_pow_2050_l404_40447

theorem sum_of_tens_and_units_digits_of_9_pow_2050 :
  ∃ (n : ℕ), 9^2050 = 10000 * n + 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_9_pow_2050_l404_40447


namespace NUMINAMATH_CALUDE_building_units_l404_40481

-- Define the cost of 1-bedroom and 2-bedroom units
def cost_1bed : ℕ := 360
def cost_2bed : ℕ := 450

-- Define the total cost when all units are full
def total_cost : ℕ := 4950

-- Define the number of 2-bedroom units
def num_2bed : ℕ := 7

-- Define the function to calculate the total number of units
def total_units (num_1bed : ℕ) : ℕ := num_1bed + num_2bed

-- Theorem statement
theorem building_units : 
  ∃ (num_1bed : ℕ), 
    num_1bed * cost_1bed + num_2bed * cost_2bed = total_cost ∧ 
    total_units num_1bed = 12 :=
sorry

end NUMINAMATH_CALUDE_building_units_l404_40481


namespace NUMINAMATH_CALUDE_work_completion_l404_40470

/-- The number of days A and B worked together before A left the job -/
def days_worked_together : ℕ := 5

/-- A's work rate in terms of the total work per day -/
def rate_A : ℚ := 1 / 20

/-- B's work rate in terms of the total work per day -/
def rate_B : ℚ := 1 / 12

/-- The number of days B worked alone after A left -/
def days_B_alone : ℕ := 3

theorem work_completion :
  (days_worked_together : ℚ) * (rate_A + rate_B) + (days_B_alone : ℚ) * rate_B = 1 := by
  sorry

#check work_completion

end NUMINAMATH_CALUDE_work_completion_l404_40470


namespace NUMINAMATH_CALUDE_street_house_numbers_l404_40412

/-- Calculates the sum of digits for all numbers in an arithmetic sequence -/
def sumOfDigits (start : Nat) (diff : Nat) (count : Nat) : Nat :=
  sorry

theorem street_house_numbers (eastStart : Nat) (eastDiff : Nat) (westStart : Nat) (westDiff : Nat) 
  (houseCount : Nat) (costPerDigit : Nat) :
  eastStart = 5 → eastDiff = 7 → westStart = 7 → westDiff = 8 → houseCount = 30 → costPerDigit = 1 →
  sumOfDigits eastStart eastDiff houseCount + sumOfDigits westStart westDiff houseCount = 149 :=
by sorry

end NUMINAMATH_CALUDE_street_house_numbers_l404_40412


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l404_40415

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m : Line) (α β : Plane) :
  perpendicular α β → 
  perpendicularLP m β → 
  ¬subset m α → 
  parallel m α :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l404_40415


namespace NUMINAMATH_CALUDE_total_interest_calculation_l404_40490

/-- Calculate simple interest -/
def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem total_interest_calculation :
  let principal_B : ℕ := 5000
  let principal_C : ℕ := 3000
  let rate : ℕ := 10
  let time_B : ℕ := 2
  let time_C : ℕ := 4
  let interest_B := simple_interest principal_B rate time_B
  let interest_C := simple_interest principal_C rate time_C
  interest_B + interest_C = 2200 := by
sorry

end NUMINAMATH_CALUDE_total_interest_calculation_l404_40490


namespace NUMINAMATH_CALUDE_cylindrical_to_cartesian_l404_40468

/-- Given a point M with cylindrical coordinates (√2, 5π/4, √2), 
    its Cartesian coordinates are (-1, -1, √2) -/
theorem cylindrical_to_cartesian :
  let ρ : ℝ := Real.sqrt 2
  let θ : ℝ := 5 * Real.pi / 4
  let z : ℝ := Real.sqrt 2
  let x : ℝ := ρ * Real.cos θ
  let y : ℝ := ρ * Real.sin θ
  x = -1 ∧ y = -1 ∧ z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cylindrical_to_cartesian_l404_40468


namespace NUMINAMATH_CALUDE_power_function_m_value_l404_40423

/-- A function of the form y = ax^n where a and n are constants and a ≠ 0 -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- The main theorem: if y = (2m-1)x^(m^2) is a power function, then m = 1 -/
theorem power_function_m_value (m : ℝ) :
  isPowerFunction (fun x => (2*m - 1) * x^(m^2)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_m_value_l404_40423


namespace NUMINAMATH_CALUDE_admin_staff_selected_is_six_l404_40436

/-- Represents the total number of staff members -/
def total_staff : ℕ := 200

/-- Represents the number of administrative staff members -/
def admin_staff : ℕ := 24

/-- Represents the total number of samples to be taken -/
def total_samples : ℕ := 50

/-- Calculates the number of administrative staff to be selected in a stratified sampling -/
def admin_staff_selected : ℕ := (admin_staff * total_samples) / total_staff

/-- Theorem stating that the number of administrative staff to be selected is 6 -/
theorem admin_staff_selected_is_six : admin_staff_selected = 6 := by
  sorry

end NUMINAMATH_CALUDE_admin_staff_selected_is_six_l404_40436


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l404_40487

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ + 2 = 0) ∧ (x₂^2 - 3*x₂ + 2 = 0) → x₁ + x₂ = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l404_40487


namespace NUMINAMATH_CALUDE_worker_earnings_l404_40414

/-- Calculates the total earnings for a worker based on survey completions and bonus structure -/
def calculate_earnings (regular_rate : ℚ) 
                       (simple_surveys : ℕ) 
                       (moderate_surveys : ℕ) 
                       (complex_surveys : ℕ) 
                       (non_cellphone_surveys : ℕ) : ℚ :=
  let simple_rate := regular_rate * (1 + 30 / 100)
  let moderate_rate := regular_rate * (1 + 50 / 100)
  let complex_rate := regular_rate * (1 + 75 / 100)
  let total_surveys := simple_surveys + moderate_surveys + complex_surveys + non_cellphone_surveys
  let survey_earnings := 
    regular_rate * non_cellphone_surveys +
    simple_rate * simple_surveys +
    moderate_rate * moderate_surveys +
    complex_rate * complex_surveys
  let tiered_bonus := 
    if total_surveys ≥ 100 then 250
    else if total_surveys ≥ 75 then 150
    else if total_surveys ≥ 50 then 100
    else 0
  let milestone_bonus := 
    (if simple_surveys ≥ 25 then 50 else 0) +
    (if moderate_surveys ≥ 15 then 75 else 0) +
    (if complex_surveys ≥ 5 then 125 else 0)
  survey_earnings + tiered_bonus + milestone_bonus

/-- The total earnings for the worker is 1765 -/
theorem worker_earnings : 
  calculate_earnings 10 30 20 10 40 = 1765 := by sorry

end NUMINAMATH_CALUDE_worker_earnings_l404_40414


namespace NUMINAMATH_CALUDE_carSalesmanFebruarySales_l404_40421

/-- Represents the earnings and sales of a car salesman -/
structure CarSalesman where
  baseSalary : ℕ
  commission : ℕ
  januaryEarnings : ℕ

/-- Calculates the number of cars needed to be sold to reach a target earnings -/
def carsNeededForEarnings (s : CarSalesman) (targetEarnings : ℕ) : ℕ :=
  ((targetEarnings - s.baseSalary) + s.commission - 1) / s.commission

/-- Theorem: The car salesman needs to sell 13 cars in February to double January earnings -/
theorem carSalesmanFebruarySales (s : CarSalesman)
    (h1 : s.baseSalary = 1000)
    (h2 : s.commission = 200)
    (h3 : s.januaryEarnings = 1800) :
    carsNeededForEarnings s (2 * s.januaryEarnings) = 13 := by
  sorry


end NUMINAMATH_CALUDE_carSalesmanFebruarySales_l404_40421


namespace NUMINAMATH_CALUDE_locus_is_ellipse_and_angle_bisector_l404_40404

-- Define the circle A
def circle_A (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 15 = 0

-- Define point B
def point_B : ℝ × ℝ := (1, 0)

-- Define the locus of points N
def locus_N (x y : ℝ) : Prop :=
  ∃ (mx my : ℝ),
    circle_A mx my ∧
    (x - mx)^2 + (y - my)^2 = (x - point_B.1)^2 + (y - point_B.2)^2 ∧
    (x - mx) * (1 - mx) + (y - my) * (0 - my) = 0

-- Theorem statement
theorem locus_is_ellipse_and_angle_bisector :
  (∀ x y, locus_N x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∀ k x₁ y₁ x₂ y₂,
    x₁^2/4 + y₁^2/3 = 1 ∧
    x₂^2/4 + y₂^2/3 = 1 ∧
    y₁ = k*(x₁ - 1) ∧
    y₂ = k*(x₂ - 1) →
    (y₁ / (x₁ - 4) + y₂ / (x₂ - 4) = 0)) :=
sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_and_angle_bisector_l404_40404


namespace NUMINAMATH_CALUDE_perfect_square_mod_four_l404_40444

theorem perfect_square_mod_four (n : ℕ) : (n^2) % 4 = 0 ∨ (n^2) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_mod_four_l404_40444


namespace NUMINAMATH_CALUDE_rationalize_denominator_l404_40467

theorem rationalize_denominator :
  ∃ (A B C D E F : ℤ),
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) =
    (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = -1 ∧ B = -3 ∧ C = 1 ∧ D = 2 ∧ E = 33 ∧ F = 17 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l404_40467


namespace NUMINAMATH_CALUDE_mans_downstream_rate_l404_40426

/-- A man rowing in a river with a current -/
theorem mans_downstream_rate 
  (rate_still_water : ℝ) 
  (rate_current : ℝ) 
  (rate_upstream : ℝ) 
  (h1 : rate_still_water = 6) 
  (h2 : rate_current = 6) 
  (h3 : rate_upstream = 6) :
  rate_still_water + rate_current = 12 := by sorry

end NUMINAMATH_CALUDE_mans_downstream_rate_l404_40426


namespace NUMINAMATH_CALUDE_sherries_banana_bread_l404_40430

theorem sherries_banana_bread (recipe_loaves : ℕ) (recipe_bananas : ℕ) (total_bananas : ℕ) 
  (h1 : recipe_loaves = 3)
  (h2 : recipe_bananas = 1)
  (h3 : total_bananas = 33) :
  (total_bananas * recipe_loaves) / recipe_bananas = 99 :=
by sorry

end NUMINAMATH_CALUDE_sherries_banana_bread_l404_40430


namespace NUMINAMATH_CALUDE_problem_solution_l404_40439

theorem problem_solution (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) (h3 : a + b > 0) :
  (a / b > -1) ∧ (abs a < abs b) ∧ ((a - 1) * (b - 1) < 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l404_40439


namespace NUMINAMATH_CALUDE_polygon_diagonals_l404_40460

theorem polygon_diagonals (n : ℕ) (h : n > 0) : 
  3 * n = n * (n - 3) / 2 ↔ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l404_40460


namespace NUMINAMATH_CALUDE_impossibleTiling_l404_40461

/-- Represents a chessboard with two opposite corners removed -/
structure ModifiedChessboard :=
  (size : Nat)
  (cornersRemoved : Nat)

/-- Represents a domino tile -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Function to check if a modified chessboard can be tiled with dominoes -/
def canTileChessboard (board : ModifiedChessboard) (tile : Domino) : Prop :=
  board.size = 8 ∧
  board.cornersRemoved = 2 ∧
  tile.length = 2 ∧
  tile.width = 1 ∧
  ∃ (tiling : Nat), False  -- This represents the impossibility of tiling

/-- Theorem stating that it's impossible to tile the modified chessboard with 2x1 dominoes -/
theorem impossibleTiling (board : ModifiedChessboard) (tile : Domino) :
  ¬(canTileChessboard board tile) :=
by
  sorry

end NUMINAMATH_CALUDE_impossibleTiling_l404_40461


namespace NUMINAMATH_CALUDE_area_formula_correct_perimeter_formula_correct_l404_40428

/-- Represents a figure composed of a square and a rectangle -/
structure CompositeFigure where
  a : ℝ
  h : a > 0

namespace CompositeFigure

/-- The area of the composite figure -/
def area (f : CompositeFigure) : ℝ := f.a^2 + 1.5 * f.a

/-- The perimeter of the composite figure -/
def perimeter (f : CompositeFigure) : ℝ := 4 * f.a + 3

/-- Theorem stating that the area formula is correct -/
theorem area_formula_correct (f : CompositeFigure) : 
  area f = f.a^2 + 1.5 * f.a := by sorry

/-- Theorem stating that the perimeter formula is correct -/
theorem perimeter_formula_correct (f : CompositeFigure) : 
  perimeter f = 4 * f.a + 3 := by sorry

end CompositeFigure

end NUMINAMATH_CALUDE_area_formula_correct_perimeter_formula_correct_l404_40428


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l404_40493

/-- Given a geometric sequence {a_n} with common ratio q and S_n as the sum of the first n terms,
    if a_3 = S_3 + 1, then q = 3 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum formula for geometric sequence
  a 3 = S 3 + 1 →
  q = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l404_40493


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l404_40455

theorem bowling_team_average_weight 
  (original_players : ℕ) 
  (original_average : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) :
  original_players = 7 →
  original_average = 76 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  let total_weight := original_players * original_average + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  let new_average := total_weight / new_total_players
  new_average = 78 := by
sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l404_40455


namespace NUMINAMATH_CALUDE_fair_attendance_proof_l404_40473

def fair_attendance (last_year this_year next_year : ℕ) : Prop :=
  (this_year = 600) ∧
  (next_year = 2 * this_year) ∧
  (last_year = next_year - 200)

theorem fair_attendance_proof :
  ∃ (last_year this_year next_year : ℕ),
    fair_attendance last_year this_year next_year ∧
    last_year = 1000 ∧ this_year = 600 ∧ next_year = 1200 := by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_proof_l404_40473


namespace NUMINAMATH_CALUDE_floor_equation_solution_l404_40476

theorem floor_equation_solution :
  ∃! x : ℝ, ⌊x⌋ + x + (1/2 : ℝ) = 20.7 :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l404_40476


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_6_l404_40477

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def last_digit (n : ℕ) : ℕ := n % 10

def first_four_digits (n : ℕ) : ℕ := n / 10

theorem five_digit_multiple_of_6 (n : ℕ) :
  n ≥ 52280 ∧ n < 52290 ∧ 
  first_four_digits n = 5228 ∧
  is_multiple_of_6 n →
  last_digit n = 4 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_6_l404_40477


namespace NUMINAMATH_CALUDE_inequality_proof_l404_40448

theorem inequality_proof (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  1 / Real.sqrt (1 + x^2) + 1 / Real.sqrt (1 + y^2) ≤ 2 / Real.sqrt (1 + x*y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l404_40448


namespace NUMINAMATH_CALUDE_water_drinkers_l404_40464

theorem water_drinkers (total : ℕ) (juice_percent : ℚ) (water_percent : ℚ) (juice_count : ℕ) :
  juice_percent = 2/5 →
  water_percent = 3/10 →
  juice_count = 100 →
  ∃ water_count : ℕ, water_count = 75 ∧ (water_count : ℚ) / total = water_percent :=
by sorry

end NUMINAMATH_CALUDE_water_drinkers_l404_40464


namespace NUMINAMATH_CALUDE_income_comparison_l404_40431

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = 0.8 * juan) 
  (h2 : mary = 1.28 * juan) : 
  (mary - tim) / tim = 0.6 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l404_40431


namespace NUMINAMATH_CALUDE_correct_num_boys_l404_40491

/-- The number of trees -/
def total_trees : ℕ := 29

/-- The number of trees left unwatered -/
def unwatered_trees : ℕ := 2

/-- The number of boys who went to water the trees -/
def num_boys : ℕ := 3

/-- Theorem stating that the number of boys is correct -/
theorem correct_num_boys :
  ∃ (trees_per_boy : ℕ), 
    num_boys * trees_per_boy = total_trees - unwatered_trees ∧ 
    trees_per_boy > 0 :=
sorry

end NUMINAMATH_CALUDE_correct_num_boys_l404_40491


namespace NUMINAMATH_CALUDE_integer_squared_less_than_triple_l404_40496

theorem integer_squared_less_than_triple :
  ∀ x : ℤ, x^2 < 3*x ↔ x = 1 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_integer_squared_less_than_triple_l404_40496


namespace NUMINAMATH_CALUDE_min_value_of_quartic_l404_40466

theorem min_value_of_quartic (x : ℝ) : 
  let y := (x - 16) * (x - 14) * (x + 14) * (x + 16)
  ∀ z : ℝ, y ≥ -900 ∧ ∃ x : ℝ, y = -900 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_quartic_l404_40466


namespace NUMINAMATH_CALUDE_max_divisibility_of_product_l404_40443

theorem max_divisibility_of_product (w x y z : ℕ) :
  w % 2 = 1 → x % 2 = 1 → y % 2 = 1 → z % 2 = 1 →
  w ≠ x → w ≠ y → w ≠ z → x ≠ y → x ≠ z → y ≠ z →
  w > 0 → x > 0 → y > 0 → z > 0 →
  ∃ (k : ℕ), (w^2 + x^2) * (y^2 + z^2) = 4 * k ∧
  ∀ (m : ℕ), m > 4 → (w^2 + x^2) * (y^2 + z^2) % m ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_divisibility_of_product_l404_40443


namespace NUMINAMATH_CALUDE_polynomial_degree_l404_40419

/-- The degree of the polynomial (x^5-2x^4+3x^3+x-14)(4x^11-8x^8+7x^5+40)-(2x^3+3)^7 is 21 -/
theorem polynomial_degree : ∃ p : Polynomial ℝ, 
  p = (X^5 - 2*X^4 + 3*X^3 + X - 14) * (4*X^11 - 8*X^8 + 7*X^5 + 40) - (2*X^3 + 3)^7 ∧ 
  p.degree = some 21 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_l404_40419


namespace NUMINAMATH_CALUDE_clothing_store_pricing_strategy_l404_40489

/-- Clothing store pricing and sales model -/
structure ClothingStore where
  cost : ℕ             -- Cost price per piece in yuan
  price : ℕ            -- Original selling price per piece in yuan
  baseSales : ℕ        -- Original daily sales in pieces
  salesIncrease : ℕ    -- Additional sales per yuan of price reduction

/-- Calculate daily sales after price reduction -/
def dailySales (store : ClothingStore) (reduction : ℕ) : ℕ :=
  store.baseSales + store.salesIncrease * reduction

/-- Calculate profit per piece after price reduction -/
def profitPerPiece (store : ClothingStore) (reduction : ℕ) : ℤ :=
  (store.price - store.cost - reduction : ℤ)

/-- Calculate total daily profit after price reduction -/
def dailyProfit (store : ClothingStore) (reduction : ℕ) : ℤ :=
  (dailySales store reduction) * (profitPerPiece store reduction)

/-- The main theorem about the clothing store's pricing strategy -/
theorem clothing_store_pricing_strategy 
  (store : ClothingStore) 
  (h_cost : store.cost = 45) 
  (h_price : store.price = 65) 
  (h_baseSales : store.baseSales = 30) 
  (h_salesIncrease : store.salesIncrease = 5) : 
  (dailySales store 3 = 45 ∧ profitPerPiece store 3 = 17) ∧
  (∃ x : ℕ, x = 10 ∧ dailyProfit store x = 800) := by
  sorry

end NUMINAMATH_CALUDE_clothing_store_pricing_strategy_l404_40489


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l404_40454

/-- The number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 8

/-- The number of colors of ribbon -/
def ribbon_colors : ℕ := 3

/-- The number of types of gift cards -/
def gift_card_types : ℕ := 4

/-- The total number of possible gift wrapping combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_card_types

/-- Theorem stating that the total number of combinations is 96 -/
theorem gift_wrapping_combinations :
  total_combinations = 96 := by sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l404_40454


namespace NUMINAMATH_CALUDE_converse_not_hold_for_naturals_l404_40495

theorem converse_not_hold_for_naturals : 
  ∃ (a b c d : ℕ), a + d = b + c ∧ (a < c ∨ b < d) :=
sorry

end NUMINAMATH_CALUDE_converse_not_hold_for_naturals_l404_40495
