import Mathlib

namespace goldfish_cost_graph_l94_9492

/-- Represents the cost function for buying goldfish -/
def cost (n : ℕ) : ℚ :=
  18 * n + 3

/-- Represents the set of points on the graph -/
def graph : Set (ℕ × ℚ) :=
  {p | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 20 ∧ p = (n, cost n)}

theorem goldfish_cost_graph :
  (∃ (S : Set (ℕ × ℚ)), S.Finite ∧ (∀ p ∈ S, ∃ q ∈ S, p ≠ q) ∧ S = graph) :=
sorry

end goldfish_cost_graph_l94_9492


namespace problem_solution_l94_9412

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 + Real.log x₀ = 0

-- Theorem to prove
theorem problem_solution : (¬p) ∧ q := by sorry

end problem_solution_l94_9412


namespace min_value_fourth_power_l94_9430

theorem min_value_fourth_power (x : ℝ) : 
  x ∈ Set.Icc 0 1 → (x^4 + (1-x)^4 : ℝ) ≥ 1/8 ∧ ∃ y ∈ Set.Icc 0 1, y^4 + (1-y)^4 = 1/8 := by
  sorry

end min_value_fourth_power_l94_9430


namespace negation_inequality_statement_l94_9451

theorem negation_inequality_statement :
  ¬(∀ (x : ℝ), x^2 + 1 > 0) ≠ (∃ (x : ℝ), x^2 + 1 < 0) :=
by sorry

end negation_inequality_statement_l94_9451


namespace dogsled_race_time_difference_l94_9463

/-- Proves that the difference in time taken to complete a 300-mile course between two teams,
    where one team's average speed is 5 miles per hour greater than the other team's speed
    of 20 miles per hour, is 3 hours. -/
theorem dogsled_race_time_difference :
  let course_length : ℝ := 300
  let team_b_speed : ℝ := 20
  let team_a_speed : ℝ := team_b_speed + 5
  let team_b_time : ℝ := course_length / team_b_speed
  let team_a_time : ℝ := course_length / team_a_speed
  team_b_time - team_a_time = 3 := by
  sorry

end dogsled_race_time_difference_l94_9463


namespace abigail_typing_speed_l94_9436

/-- The number of words Abigail can type in half an hour -/
def words_per_half_hour : ℕ := sorry

/-- The total length of the report in words -/
def total_report_length : ℕ := 1000

/-- The number of words Abigail has already written -/
def words_already_written : ℕ := 200

/-- The number of minutes Abigail needs to finish the report -/
def minutes_to_finish : ℕ := 80

theorem abigail_typing_speed :
  words_per_half_hour = 300 := by sorry

end abigail_typing_speed_l94_9436


namespace square_area_equals_perimeter_implies_perimeter_16_l94_9454

theorem square_area_equals_perimeter_implies_perimeter_16 :
  ∀ s : ℝ, s > 0 → s^2 = 4*s → 4*s = 16 := by sorry

end square_area_equals_perimeter_implies_perimeter_16_l94_9454


namespace female_employees_with_advanced_degrees_l94_9466

theorem female_employees_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (total_advanced_degrees : ℕ) 
  (males_college_only : ℕ) 
  (h1 : total_employees = 160)
  (h2 : total_females = 90)
  (h3 : total_advanced_degrees = 80)
  (h4 : males_college_only = 40) :
  total_advanced_degrees - (total_employees - total_females - males_college_only) = 50 := by
  sorry

end female_employees_with_advanced_degrees_l94_9466


namespace road_travel_cost_l94_9448

/-- The cost of traveling two intersecting roads on a rectangular lawn -/
theorem road_travel_cost (lawn_length lawn_width road_width cost_per_sqm : ℕ) : 
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  cost_per_sqm = 3 → 
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * cost_per_sqm = 3900 :=
by sorry

end road_travel_cost_l94_9448


namespace no_solution_exists_l94_9446

theorem no_solution_exists : ¬ ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a + b ≠ 0 ∧ 1 / a + 2 / b = 3 / (a + b) := by
  sorry

end no_solution_exists_l94_9446


namespace slipper_cost_l94_9458

theorem slipper_cost (total_items : ℕ) (slipper_count : ℕ) (lipstick_count : ℕ) (lipstick_price : ℚ) 
  (hair_color_count : ℕ) (hair_color_price : ℚ) (total_paid : ℚ) :
  total_items = slipper_count + lipstick_count + hair_color_count →
  total_items = 18 →
  slipper_count = 6 →
  lipstick_count = 4 →
  lipstick_price = 5/4 →
  hair_color_count = 8 →
  hair_color_price = 3 →
  total_paid = 44 →
  (total_paid - (lipstick_count * lipstick_price + hair_color_count * hair_color_price)) / slipper_count = 5/2 :=
by
  sorry

#check slipper_cost

end slipper_cost_l94_9458


namespace rihlelo_symmetry_l94_9414

/-- Represents a design pattern -/
structure Design where
  /-- The type of object the design is for -/
  objectType : String
  /-- The country of origin for the design -/
  origin : String
  /-- The number of lines of symmetry in the design -/
  symmetryLines : ℕ

/-- The rihlèlò design from Mozambique -/
def rihlelo : Design where
  objectType := "winnowing tray"
  origin := "Mozambique"
  symmetryLines := 4

/-- Theorem stating that the rihlèlò design has 4 lines of symmetry -/
theorem rihlelo_symmetry : rihlelo.symmetryLines = 4 := by
  sorry

end rihlelo_symmetry_l94_9414


namespace fraction_product_simplification_l94_9477

theorem fraction_product_simplification : 
  (36 : ℚ) / 34 * 26 / 48 * 136 / 78 * 9 / 4 = 1 := by sorry

end fraction_product_simplification_l94_9477


namespace tenth_place_is_unnamed_l94_9440

/-- Represents a racer in the race --/
inductive Racer
| Eda
| Simon
| Jacob
| Naomi
| Cal
| Iris
| Unnamed

/-- Represents the finishing position of a racer --/
def Position := Fin 15

/-- The race results, mapping each racer to their position --/
def RaceResult := Racer → Position

def valid_race_result (result : RaceResult) : Prop :=
  (result Racer.Jacob).val + 4 = (result Racer.Eda).val
  ∧ (result Racer.Naomi).val = (result Racer.Simon).val + 1
  ∧ (result Racer.Jacob).val = (result Racer.Cal).val + 3
  ∧ (result Racer.Simon).val = (result Racer.Iris).val + 2
  ∧ (result Racer.Cal).val + 2 = (result Racer.Iris).val
  ∧ (result Racer.Naomi).val = 7

theorem tenth_place_is_unnamed (result : RaceResult) 
  (h : valid_race_result result) : 
  ∀ r : Racer, r ≠ Racer.Unnamed → (result r).val ≠ 10 := by
  sorry

end tenth_place_is_unnamed_l94_9440


namespace weight_problem_l94_9467

/-- The weight problem -/
theorem weight_problem (student_weight sister_weight brother_weight : ℝ) : 
  (student_weight - 8 = sister_weight + brother_weight) →
  (brother_weight = sister_weight + 5) →
  (sister_weight + brother_weight = 180) →
  (student_weight = 188) :=
by
  sorry

end weight_problem_l94_9467


namespace factorial_of_factorial_divided_by_factorial_l94_9462

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end factorial_of_factorial_divided_by_factorial_l94_9462


namespace gcd_2021_2048_l94_9406

theorem gcd_2021_2048 : Nat.gcd 2021 2048 = 1 := by
  sorry

end gcd_2021_2048_l94_9406


namespace cedarwood_earnings_theorem_l94_9482

/-- Represents the data for each school's participation in the community project -/
structure SchoolData where
  name : String
  students : ℕ
  days : ℕ

/-- Calculates the total earnings for Cedarwood school given the project data -/
def cedarwoodEarnings (ashwood briarwood cedarwood : SchoolData) (totalPaid : ℚ) : ℚ :=
  let totalStudentDays := ashwood.students * ashwood.days + briarwood.students * briarwood.days + cedarwood.students * cedarwood.days
  let dailyWage := totalPaid / totalStudentDays
  dailyWage * (cedarwood.students * cedarwood.days)

/-- Theorem stating that Cedarwood school's earnings are 454.74 given the project conditions -/
theorem cedarwood_earnings_theorem (ashwood briarwood cedarwood : SchoolData) (totalPaid : ℚ) :
  ashwood.name = "Ashwood" ∧ ashwood.students = 9 ∧ ashwood.days = 4 ∧
  briarwood.name = "Briarwood" ∧ briarwood.students = 5 ∧ briarwood.days = 6 ∧
  cedarwood.name = "Cedarwood" ∧ cedarwood.students = 6 ∧ cedarwood.days = 8 ∧
  totalPaid = 1080 →
  cedarwoodEarnings ashwood briarwood cedarwood totalPaid = 454.74 := by
  sorry

#eval cedarwoodEarnings
  { name := "Ashwood", students := 9, days := 4 }
  { name := "Briarwood", students := 5, days := 6 }
  { name := "Cedarwood", students := 6, days := 8 }
  1080

end cedarwood_earnings_theorem_l94_9482


namespace binomial_square_constant_l94_9438

theorem binomial_square_constant (a : ℝ) : 
  (∃ b c : ℝ, ∀ x, 9*x^2 - 27*x + a = (b*x + c)^2) → a = 20.25 := by
  sorry

end binomial_square_constant_l94_9438


namespace quadratic_coefficient_l94_9485

/-- A quadratic function with vertex (2, 5) passing through (0, 0) has a = -5/4 --/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- quadratic function definition
  (2, 5) = (2, a * 2^2 + b * 2 + c) →     -- vertex condition
  (0, 0) = (0, a * 0^2 + b * 0 + c) →     -- point condition
  a = -5/4 := by
sorry

end quadratic_coefficient_l94_9485


namespace angle_A_measure_l94_9483

theorem angle_A_measure :
  ∀ (A B C : ℝ) (small_angle : ℝ),
  B = 120 →
  B + C = 180 →
  small_angle = 50 →
  small_angle + C + 70 = 180 →
  A + B = 180 →
  A = 60 :=
by sorry

end angle_A_measure_l94_9483


namespace distributor_profit_percentage_l94_9461

/-- The commission rate of the online store -/
def commission_rate : ℚ := 1/5

/-- The price at which the distributor obtains the product from the producer -/
def producer_price : ℚ := 18

/-- The price observed by the buyer on the online store -/
def buyer_price : ℚ := 27

/-- The selling price of the distributor to the online store -/
def selling_price : ℚ := buyer_price / (1 + commission_rate)

/-- The profit made by the distributor per item -/
def profit : ℚ := selling_price - producer_price

/-- The profit percentage of the distributor -/
def profit_percentage : ℚ := profit / producer_price * 100

theorem distributor_profit_percentage :
  profit_percentage = 25 := by sorry

end distributor_profit_percentage_l94_9461


namespace cycle_iteration_equivalence_l94_9499

/-- A function that represents the k-th iteration of f -/
def iterate (f : α → α) : ℕ → α → α
  | 0, x => x
  | n + 1, x => f (iterate f n x)

/-- The main theorem -/
theorem cycle_iteration_equivalence
  {α : Type*} (f : α → α) (x₀ : α) (s k : ℕ) :
  (∃ (n : ℕ), iterate f s x₀ = x₀) →  -- x₀ belongs to a cycle of length s
  (k % s = 0 ↔ iterate f k x₀ = x₀) :=
sorry

end cycle_iteration_equivalence_l94_9499


namespace four_digit_number_property_l94_9475

theorem four_digit_number_property (m : ℕ) : 
  1000 ≤ m ∧ m ≤ 2025 →
  ∃ (n : ℕ), n > 0 ∧ Nat.Prime (m - n) ∧ ∃ (k : ℕ), m * n = k ^ 2 := by
  sorry

end four_digit_number_property_l94_9475


namespace school_fundraising_admin_fee_percentage_l94_9445

/-- Proves that the percentage deducted for administration fees is 2% --/
theorem school_fundraising_admin_fee_percentage 
  (johnson_amount : ℝ)
  (sutton_amount : ℝ)
  (rollin_amount : ℝ)
  (total_amount : ℝ)
  (remaining_amount : ℝ)
  (h1 : johnson_amount = 2300)
  (h2 : johnson_amount = 2 * sutton_amount)
  (h3 : rollin_amount = 8 * sutton_amount)
  (h4 : rollin_amount = total_amount / 3)
  (h5 : remaining_amount = 27048) :
  (total_amount - remaining_amount) / total_amount * 100 = 2 := by
  sorry

end school_fundraising_admin_fee_percentage_l94_9445


namespace students_per_class_l94_9410

theorem students_per_class 
  (cards_per_student : ℕ) 
  (periods_per_day : ℕ) 
  (cards_per_pack : ℕ) 
  (cost_per_pack : ℕ) 
  (total_spent : ℕ) 
  (h1 : cards_per_student = 10)
  (h2 : periods_per_day = 6)
  (h3 : cards_per_pack = 50)
  (h4 : cost_per_pack = 3)
  (h5 : total_spent = 108) :
  total_spent / cost_per_pack * cards_per_pack / cards_per_student / periods_per_day = 30 := by
sorry

end students_per_class_l94_9410


namespace arithmetic_proof_l94_9421

theorem arithmetic_proof : (3 + 2) - (2 + 1) = 2 := by
  sorry

end arithmetic_proof_l94_9421


namespace amusement_park_tickets_l94_9405

theorem amusement_park_tickets (adult_price child_price total_paid total_tickets : ℕ)
  (h1 : adult_price = 8)
  (h2 : child_price = 5)
  (h3 : total_paid = 201)
  (h4 : total_tickets = 33)
  : ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_paid ∧
    child_tickets = 21 := by
  sorry

end amusement_park_tickets_l94_9405


namespace no_integer_roots_for_odd_coefficients_l94_9465

theorem no_integer_roots_for_odd_coefficients (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) : 
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
  sorry

end no_integer_roots_for_odd_coefficients_l94_9465


namespace choose_two_from_three_l94_9487

theorem choose_two_from_three (n : ℕ) (k : ℕ) : n = 3 ∧ k = 2 → Nat.choose n k = 3 := by
  sorry

end choose_two_from_three_l94_9487


namespace geometric_sequence_sum_l94_9435

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_sum : a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 16) :
  a 2 + a 4 = 4 := by
sorry

end geometric_sequence_sum_l94_9435


namespace one_positive_real_solution_l94_9471

def f (x : ℝ) : ℝ := x^8 + 5*x^7 + 10*x^6 + 2023*x^5 - 2021*x^4

theorem one_positive_real_solution :
  ∃! (x : ℝ), x > 0 ∧ f x = 0 :=
sorry

end one_positive_real_solution_l94_9471


namespace starting_team_combinations_count_l94_9428

def team_size : ℕ := 18
def starting_team_size : ℕ := 8
def other_players_size : ℕ := starting_team_size - 2  -- 6 players excluding goalie and captain

def number_of_starting_team_combinations : ℕ :=
  team_size *  -- ways to choose goalie
  (team_size - 1) *  -- ways to choose captain (excluding goalie)
  (Nat.choose (team_size - 2) other_players_size)  -- ways to choose remaining 6 players

theorem starting_team_combinations_count :
  number_of_starting_team_combinations = 2455344 :=
by sorry

end starting_team_combinations_count_l94_9428


namespace bd_squared_equals_nine_l94_9423

theorem bd_squared_equals_nine 
  (h1 : a - b - c + d = 12) 
  (h2 : a + b - c - d = 6) : 
  (b - d)^2 = 9 := by
sorry

end bd_squared_equals_nine_l94_9423


namespace g_neg_three_eq_four_l94_9491

/-- The function g is defined as g(x) = x^2 + 2x + 1 for all real x. -/
def g (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- Theorem: The value of g(-3) is equal to 4. -/
theorem g_neg_three_eq_four : g (-3) = 4 := by sorry

end g_neg_three_eq_four_l94_9491


namespace quadratic_vertex_on_line_quadratic_intersects_line_l94_9441

/-- The quadratic function parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + m - 1

/-- The line y = x - 1 -/
def g (x : ℝ) : ℝ := x - 1

/-- The line y = x + b parameterized by b -/
def h (b : ℝ) (x : ℝ) : ℝ := x + b

/-- The vertex of a quadratic function ax^2 + bx + c is at (-b/(2a), f(-b/(2a))) -/
def vertex (m : ℝ) : ℝ × ℝ := (m, f m m)

theorem quadratic_vertex_on_line (m : ℝ) : 
  g (vertex m).1 = (vertex m).2 :=
sorry

theorem quadratic_intersects_line (m b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = h b x₁ ∧ f m x₂ = h b x₂) ↔ b > -5/4 :=
sorry

end quadratic_vertex_on_line_quadratic_intersects_line_l94_9441


namespace heaviest_lightest_difference_l94_9476

def pumpkin_contest (brad_weight jessica_weight betty_weight : ℝ) : Prop :=
  jessica_weight = brad_weight / 2 ∧
  betty_weight = 4 * jessica_weight ∧
  brad_weight = 54

theorem heaviest_lightest_difference (brad_weight jessica_weight betty_weight : ℝ) 
  (h : pumpkin_contest brad_weight jessica_weight betty_weight) :
  max betty_weight (max brad_weight jessica_weight) - 
  min betty_weight (min brad_weight jessica_weight) = 81 := by
  sorry

end heaviest_lightest_difference_l94_9476


namespace right_triangle_sets_l94_9495

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that among the given sets, only (6, 8, 10) forms a right triangle --/
theorem right_triangle_sets :
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle (Real.sqrt 7) 3 5 ∧
  is_right_triangle 6 8 10 ∧
  ¬ is_right_triangle 5 12 12 :=
sorry

end right_triangle_sets_l94_9495


namespace surface_polygon_angle_sum_sum_all_defects_l94_9479

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  
/-- An m-gon on the surface of a polyhedron -/
structure SurfacePolygon (P : ConvexPolyhedron) where
  m : ℕ  -- number of sides
  -- Add other necessary fields here

/-- The defect of a polyhedral angle -/
def defect (P : ConvexPolyhedron) (v : ℝ × ℝ × ℝ) : ℝ := sorry

/-- The sum of angles of a surface polygon -/
def sumAngles (P : ConvexPolyhedron) (S : SurfacePolygon P) : ℝ := sorry

/-- The sum of defects of vertices inside a surface polygon -/
def sumDefectsInside (P : ConvexPolyhedron) (S : SurfacePolygon P) : ℝ := sorry

/-- The sum of defects of all vertices of a polyhedron -/
def sumAllDefects (P : ConvexPolyhedron) : ℝ := sorry

theorem surface_polygon_angle_sum (P : ConvexPolyhedron) (S : SurfacePolygon P) :
  sumAngles P S = 2 * Real.pi * (S.m - 2 : ℝ) + sumDefectsInside P S := by sorry

theorem sum_all_defects (P : ConvexPolyhedron) :
  sumAllDefects P = 4 * Real.pi := by sorry

end surface_polygon_angle_sum_sum_all_defects_l94_9479


namespace inequality_proof_l94_9411

theorem inequality_proof (a b c : ℝ) : 
  a = 0.1 * Real.exp 0.1 → 
  b = 1 / 9 → 
  c = -Real.log 0.9 → 
  c < a ∧ a < b := by
sorry

end inequality_proof_l94_9411


namespace hall_seats_l94_9456

theorem hall_seats (total_seats : ℕ) : 
  (total_seats : ℝ) / 2 = 300 → total_seats = 600 := by
  sorry

end hall_seats_l94_9456


namespace smallest_result_l94_9429

def S : Finset Nat := {2, 4, 6, 8, 10, 12}

def process (a b c : Nat) : Nat := (a + b) * c

def valid_triple (a b c : Nat) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_result :
  ∃ (a b c : Nat), valid_triple a b c ∧
    (∀ (x y z : Nat), valid_triple x y z →
      min (process a b c) (min (process a c b) (process b c a)) ≤
      min (process x y z) (min (process x z y) (process y z x))) ∧
    min (process a b c) (min (process a c b) (process b c a)) = 20 :=
by sorry

end smallest_result_l94_9429


namespace pipe_b_rate_is_30_l94_9420

/-- Represents the capacity of the tank in liters -/
def tank_capacity : ℕ := 900

/-- Represents the rate at which pipe A fills the tank in liters per minute -/
def pipe_a_rate : ℕ := 40

/-- Represents the rate at which pipe C drains the tank in liters per minute -/
def pipe_c_rate : ℕ := 20

/-- Represents the time taken to fill the tank in minutes -/
def fill_time : ℕ := 54

/-- Represents the duration of each pipe's operation in a cycle in minutes -/
def cycle_duration : ℕ := 3

/-- Theorem: Given the tank capacity, fill rates of pipes A and C, fill time, and cycle duration,
    the fill rate of pipe B is 30 liters per minute -/
theorem pipe_b_rate_is_30 :
  ∃ (pipe_b_rate : ℕ),
    pipe_b_rate = 30 ∧
    (fill_time / cycle_duration) * (pipe_a_rate + pipe_b_rate - pipe_c_rate) = tank_capacity :=
  sorry

end pipe_b_rate_is_30_l94_9420


namespace only_C_is_pythagorean_triple_l94_9407

-- Define the sets of numbers
def set_A : Vector ℕ 3 := ⟨[7, 8, 9], by rfl⟩
def set_B : Vector ℕ 3 := ⟨[5, 6, 7], by rfl⟩
def set_C : Vector ℕ 3 := ⟨[5, 12, 13], by rfl⟩
def set_D : Vector ℕ 3 := ⟨[21, 25, 28], by rfl⟩

-- Define a function to check if a set of three numbers is a Pythagorean triple
def is_pythagorean_triple (v : Vector ℕ 3) : Prop :=
  v[0] * v[0] + v[1] * v[1] = v[2] * v[2]

-- Theorem statement
theorem only_C_is_pythagorean_triple :
  ¬(is_pythagorean_triple set_A) ∧
  ¬(is_pythagorean_triple set_B) ∧
  (is_pythagorean_triple set_C) ∧
  ¬(is_pythagorean_triple set_D) :=
by sorry

end only_C_is_pythagorean_triple_l94_9407


namespace geometric_sequence_sum_l94_9490

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_prod : a 2 * a 5 = -3/4)
  (h_sum : a 2 + a 3 + a 4 + a 5 = 5/4) :
  1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 = -5/3 :=
sorry

end geometric_sequence_sum_l94_9490


namespace division_remainder_problem_l94_9439

theorem division_remainder_problem (dividend quotient divisor remainder : ℕ) : 
  dividend = 95 →
  quotient = 6 →
  divisor = 15 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
sorry

end division_remainder_problem_l94_9439


namespace flight_time_theorem_l94_9498

/-- Represents the flight scenario between two towns -/
structure FlightScenario where
  d : ℝ  -- distance between towns
  p : ℝ  -- speed of plane in still air
  w : ℝ  -- speed of wind
  time_against : ℝ  -- time for flight against wind
  time_diff : ℝ  -- difference in time between with-wind and still air flights

/-- The theorem stating the conditions and the result to be proved -/
theorem flight_time_theorem (scenario : FlightScenario) 
  (h1 : scenario.time_against = 84)
  (h2 : scenario.time_diff = 9)
  (h3 : scenario.d = scenario.time_against * (scenario.p - scenario.w))
  (h4 : scenario.d / (scenario.p + scenario.w) = scenario.d / scenario.p - scenario.time_diff) :
  scenario.d / (scenario.p + scenario.w) = 63 ∨ scenario.d / (scenario.p + scenario.w) = 12 := by
  sorry

end flight_time_theorem_l94_9498


namespace fraction_problem_l94_9432

theorem fraction_problem (x : ℚ) : 
  x < 0.4 ∧ x * 180 = 48 → x = 4 / 15 := by
  sorry

end fraction_problem_l94_9432


namespace part1_part2_l94_9426

-- Define the quadratic function f(x)
def f (q : ℝ) (x : ℝ) : ℝ := x^2 - 16*x + q + 3

-- Part 1: Prove that if f has a root in [-1, 1], then q ∈ [-20, 12]
theorem part1 (q : ℝ) : 
  (∃ x ∈ Set.Icc (-1) 1, f q x = 0) → q ∈ Set.Icc (-20) 12 :=
by sorry

-- Part 2: Prove that if f(x) + 51 ≥ 0 for all x ∈ [q, 10], then q ∈ [9, 10)
theorem part2 (q : ℝ) :
  (∀ x ∈ Set.Icc q 10, f q x + 51 ≥ 0) → q ∈ Set.Ici 9 ∩ Set.Iio 10 :=
by sorry

end part1_part2_l94_9426


namespace go_out_is_better_l94_9484

/-- Represents the decision of the fishing boat -/
inductive Decision
| GoOut
| StayIn

/-- Represents the weather conditions -/
inductive Weather
| Good
| Bad

/-- The profit or loss for each scenario -/
def profit (d : Decision) (w : Weather) : ℝ :=
  match d, w with
  | Decision.GoOut, Weather.Good => 6000
  | Decision.GoOut, Weather.Bad => -8000
  | Decision.StayIn, _ => -1000

/-- The probability of each weather condition -/
def weather_prob (w : Weather) : ℝ :=
  match w with
  | Weather.Good => 0.6
  | Weather.Bad => 0.4

/-- The expected value of a decision -/
def expected_value (d : Decision) : ℝ :=
  (profit d Weather.Good * weather_prob Weather.Good) +
  (profit d Weather.Bad * weather_prob Weather.Bad)

/-- Theorem stating that going out to sea has a higher expected value -/
theorem go_out_is_better :
  expected_value Decision.GoOut > expected_value Decision.StayIn :=
by sorry

end go_out_is_better_l94_9484


namespace john_mission_duration_l94_9402

theorem john_mission_duration :
  let initial_duration : ℝ := 5
  let first_mission_duration : ℝ := initial_duration * (1 + 0.6)
  let second_mission_duration : ℝ := first_mission_duration * 0.5
  let third_mission_duration : ℝ := min (2 * second_mission_duration) (first_mission_duration * 0.8)
  let fourth_mission_duration : ℝ := 3 + (third_mission_duration * 0.5)
  first_mission_duration + second_mission_duration + third_mission_duration + fourth_mission_duration = 24.6 :=
by sorry

end john_mission_duration_l94_9402


namespace sqrt_7200_minus_61_cube_l94_9416

theorem sqrt_7200_minus_61_cube (a b : ℕ+) :
  (Real.sqrt 7200 - 61 : ℝ) = (Real.sqrt a.val - b.val)^3 →
  a.val + b.val = 21 := by
sorry

end sqrt_7200_minus_61_cube_l94_9416


namespace polygon_interior_angle_sum_l94_9457

theorem polygon_interior_angle_sum (n : ℕ) (h : n * 36 = 360) :
  (n - 2) * 180 = 1440 :=
sorry

end polygon_interior_angle_sum_l94_9457


namespace cyclists_problem_l94_9404

/-- Two cyclists problem -/
theorem cyclists_problem (x : ℝ) 
  (h1 : x > 0) -- Distance between A and B is positive
  (h2 : 70 + x + 90 = 3 * 70) -- Equation derived from the problem conditions
  : x = 120 := by
  sorry

end cyclists_problem_l94_9404


namespace same_color_probability_l94_9459

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def total_plates : ℕ := red_plates + blue_plates
def selected_plates : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates selected_plates + Nat.choose blue_plates selected_plates) / 
  Nat.choose total_plates selected_plates = 2 / 11 := by
  sorry

end same_color_probability_l94_9459


namespace other_ticket_cost_l94_9437

/-- Given a total of 29 tickets, with 11 tickets costing $9 each,
    and a total cost of $225 for all tickets,
    prove that the remaining tickets cost $7 each. -/
theorem other_ticket_cost (total_tickets : ℕ) (nine_dollar_tickets : ℕ) 
  (total_cost : ℕ) (h1 : total_tickets = 29) (h2 : nine_dollar_tickets = 11) 
  (h3 : total_cost = 225) : 
  (total_cost - nine_dollar_tickets * 9) / (total_tickets - nine_dollar_tickets) = 7 :=
by sorry

end other_ticket_cost_l94_9437


namespace apples_left_after_pies_l94_9450

theorem apples_left_after_pies (initial_apples : ℕ) (difference : ℕ) (apples_left : ℕ) : 
  initial_apples = 46 → difference = 32 → apples_left = initial_apples - difference → apples_left = 14 := by
  sorry

end apples_left_after_pies_l94_9450


namespace hunter_has_ten_rats_l94_9474

/-- The number of rats Hunter has -/
def hunter_rats : ℕ := sorry

/-- The number of rats Elodie has -/
def elodie_rats : ℕ := hunter_rats + 30

/-- The number of rats Kenia has -/
def kenia_rats : ℕ := 3 * (hunter_rats + elodie_rats)

/-- The total number of pets -/
def total_pets : ℕ := 200

theorem hunter_has_ten_rats :
  hunter_rats + elodie_rats + kenia_rats = total_pets →
  hunter_rats = 10 := by sorry

end hunter_has_ten_rats_l94_9474


namespace smallest_integer_y_l94_9470

theorem smallest_integer_y : ∀ y : ℤ, (7 - 3*y < 22) → y ≥ -4 :=
  sorry

end smallest_integer_y_l94_9470


namespace triangle_determination_l94_9444

/-- Represents the different combinations of triangle information --/
inductive TriangleInfo
  | SSS  -- Three sides
  | SAS  -- Two sides and included angle
  | ASA  -- Two angles and included side
  | SSA  -- Two sides and angle opposite one of them

/-- Predicate to determine if a given combination of triangle information can uniquely determine a triangle --/
def uniquely_determines_triangle (info : TriangleInfo) : Prop :=
  match info with
  | TriangleInfo.SSS => true
  | TriangleInfo.SAS => true
  | TriangleInfo.ASA => true
  | TriangleInfo.SSA => false

/-- Theorem stating which combinations of triangle information can uniquely determine a triangle --/
theorem triangle_determination :
  (uniquely_determines_triangle TriangleInfo.SSS) ∧
  (uniquely_determines_triangle TriangleInfo.SAS) ∧
  (uniquely_determines_triangle TriangleInfo.ASA) ∧
  ¬(uniquely_determines_triangle TriangleInfo.SSA) :=
by sorry

end triangle_determination_l94_9444


namespace milk_container_problem_l94_9419

-- Define the capacity of container A
def A : ℝ := 1232

-- Define the quantity of milk in container B after initial pouring
def B : ℝ := 0.375 * A

-- Define the quantity of milk in container C after initial pouring
def C : ℝ := 0.625 * A

-- Define the amount transferred from C to B
def transfer : ℝ := 154

-- Theorem statement
theorem milk_container_problem :
  -- All milk from A was poured into B and C
  (B + C = A) ∧
  -- B had 62.5% less milk than A's capacity
  (B = 0.375 * A) ∧
  -- After transfer, B and C have equal quantities
  (B + transfer = C - transfer) →
  -- The initial quantity of milk in A was 1232 liters
  A = 1232 := by
  sorry


end milk_container_problem_l94_9419


namespace sams_letters_l94_9403

theorem sams_letters (letters_tuesday : ℕ) (average_per_day : ℕ) (total_days : ℕ) :
  letters_tuesday = 7 →
  average_per_day = 5 →
  total_days = 2 →
  (average_per_day * total_days - letters_tuesday : ℕ) = 3 := by
  sorry

end sams_letters_l94_9403


namespace wrapping_paper_area_is_8lh_l94_9488

/-- Represents a rectangular box -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the area of wrapping paper needed for a given box -/
def wrappingPaperArea (box : Box) : ℝ :=
  8 * box.length * box.height

/-- Theorem stating that the area of wrapping paper needed is 8lh -/
theorem wrapping_paper_area_is_8lh (box : Box) :
  wrappingPaperArea box = 8 * box.length * box.height :=
by sorry

end wrapping_paper_area_is_8lh_l94_9488


namespace brendas_age_l94_9443

theorem brendas_age (addison brenda janet : ℚ) 
  (h1 : addison = 4 * brenda)
  (h2 : janet = brenda + 8)
  (h3 : addison = janet + 2) :
  brenda = 10 / 3 := by
sorry

end brendas_age_l94_9443


namespace cubic_minus_linear_factorization_l94_9478

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end cubic_minus_linear_factorization_l94_9478


namespace magnitude_2a_minus_b_l94_9418

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-1, 1)

theorem magnitude_2a_minus_b :
  Real.sqrt ((2 * vector_a.1 - vector_b.1)^2 + (2 * vector_a.2 - vector_b.2)^2) = 3 * Real.sqrt 2 := by
  sorry

end magnitude_2a_minus_b_l94_9418


namespace min_value_problem_l94_9433

theorem min_value_problem (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) (hmn : m + n = 1) :
  m^2 / (m + 2) + n^2 / (n + 1) ≥ 1/4 := by
  sorry

end min_value_problem_l94_9433


namespace locus_of_rectangle_vertex_l94_9425

/-- Given a circle centered at the origin with radius r and a point M(a,b) inside the circle,
    prove that the locus of point T for all rectangles MKTP where K and P lie on the circle
    is a circle centered at the origin with radius √(2r² - (a² + b²)). -/
theorem locus_of_rectangle_vertex (r a b : ℝ) (hr : r > 0) (hab : a^2 + b^2 < r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = 2 * r^2 - (a^2 + b^2) := by
  sorry

end locus_of_rectangle_vertex_l94_9425


namespace solve_bracket_equation_l94_9489

-- Define the bracket function
def bracket (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 + 1 else 2 * x + 1

-- State the theorem
theorem solve_bracket_equation :
  ∃ x : ℤ, (bracket 6) * (bracket x) = 28 ∧ x = 12 := by
  sorry

end solve_bracket_equation_l94_9489


namespace julia_payment_l94_9481

def snickers_price : ℝ := 1.5
def snickers_quantity : ℕ := 2
def mm_quantity : ℕ := 3
def change : ℝ := 8

def mm_price : ℝ := 2 * snickers_price

def total_cost : ℝ := snickers_price * snickers_quantity + mm_price * mm_quantity

theorem julia_payment : total_cost + change = 20 := by
  sorry

end julia_payment_l94_9481


namespace min_value_3m_plus_n_l94_9453

/-- The minimum value of 3m + n given the conditions -/
theorem min_value_3m_plus_n (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) : 
  let f := fun x => a^(x + 3) - 2
  let A := (-3, -1)
  (A.1 / m + A.2 / n + 1 = 0) →
  ∀ m' n', m' > 0 → n' > 0 → 
    (m' / m' + n' / n' + 1 = 0) → 
    (3 * m + n ≤ 3 * m' + n') :=
by sorry

end min_value_3m_plus_n_l94_9453


namespace new_room_size_l94_9413

/-- Given a bedroom and bathroom size, calculate the size of a new room that is twice as large as both combined -/
theorem new_room_size (bedroom : ℝ) (bathroom : ℝ) (new_room : ℝ) : 
  bedroom = 309 → bathroom = 150 → new_room = 2 * (bedroom + bathroom) → new_room = 918 := by
  sorry

end new_room_size_l94_9413


namespace knights_in_gamma_quarter_l94_9480

/-- Represents a resident of the town -/
inductive Resident
| Knight
| Liar

/-- The total number of residents in the town -/
def total_residents : ℕ := 200

/-- The total number of affirmative answers received -/
def total_affirmative_answers : ℕ := 430

/-- The number of affirmative answers received in quarter Γ -/
def gamma_quarter_affirmative_answers : ℕ := 119

/-- The number of affirmative answers a knight gives to every four questions -/
def knight_affirmative_rate : ℚ := 1/4

/-- The number of affirmative answers a liar gives to every four questions -/
def liar_affirmative_rate : ℚ := 3/4

/-- The total number of liars in the town -/
def total_liars : ℕ := (total_affirmative_answers - total_residents) / 2

theorem knights_in_gamma_quarter : 
  ∃ (k : ℕ), k = 4 ∧ 
  k ≤ gamma_quarter_affirmative_answers ∧
  (gamma_quarter_affirmative_answers - k : ℤ) = total_liars - (k - 4 : ℤ) ∧
  ∀ (other_quarter : ℕ), other_quarter ≠ gamma_quarter_affirmative_answers →
    (other_quarter : ℤ) - (total_residents - total_liars) > (total_residents - total_liars : ℤ) :=
sorry

end knights_in_gamma_quarter_l94_9480


namespace lunch_ratio_is_one_half_l94_9422

/-- The number of school days in the academic year -/
def total_school_days : ℕ := 180

/-- The number of days Becky packs her lunch -/
def becky_lunch_days : ℕ := 45

/-- The number of days Aliyah packs her lunch -/
def aliyah_lunch_days : ℕ := 2 * becky_lunch_days

/-- The ratio of Aliyah's lunch-packing days to total school days -/
def lunch_ratio : ℚ := aliyah_lunch_days / total_school_days

theorem lunch_ratio_is_one_half : lunch_ratio = 1 / 2 := by
  sorry

end lunch_ratio_is_one_half_l94_9422


namespace money_distribution_l94_9494

theorem money_distribution (a b c : ℕ) : 
  a = 3 * b →
  b > c →
  a + b + c = 645 →
  b = 134 →
  b - c = 25 :=
by sorry

end money_distribution_l94_9494


namespace max_constant_C_l94_9497

theorem max_constant_C : ∃ (C : ℝ), C = Real.sqrt 2 ∧
  (∀ (x y : ℝ), x^2 + y^2 + 1 ≥ C*(x + y)) ∧
  (∀ (x y : ℝ), x^2 + y^2 + x*y + 1 ≥ C*(x + y)) ∧
  (∀ (C' : ℝ), C' > C →
    (∃ (x y : ℝ), x^2 + y^2 + 1 < C'*(x + y) ∨ x^2 + y^2 + x*y + 1 < C'*(x + y))) := by
  sorry

end max_constant_C_l94_9497


namespace max_intersections_square_decagon_l94_9449

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  vertices : Finset (ℝ × ℝ)
  edges : Finset ((ℝ × ℝ) × (ℝ × ℝ))
  convex : Bool
  planar : Bool

/-- Represents the number of sides in a polygon -/
def numSides (p : ConvexPolygon) : ℕ := p.edges.card

/-- Determines if one polygon is inscribed in another -/
def isInscribed (p₁ p₂ : ConvexPolygon) : Prop := sorry

/-- Counts the number of shared vertices between two polygons -/
def sharedVertices (p₁ p₂ : ConvexPolygon) : ℕ := sorry

/-- Counts the number of intersections between edges of two polygons -/
def countIntersections (p₁ p₂ : ConvexPolygon) : ℕ := sorry

theorem max_intersections_square_decagon (p₁ p₂ : ConvexPolygon) : 
  numSides p₁ = 4 →
  numSides p₂ = 10 →
  p₁.convex →
  p₂.convex →
  p₁.planar →
  p₂.planar →
  isInscribed p₁ p₂ →
  sharedVertices p₁ p₂ = 4 →
  countIntersections p₁ p₂ ≤ 8 ∧ 
  ∃ (q₁ q₂ : ConvexPolygon), 
    numSides q₁ = 4 ∧
    numSides q₂ = 10 ∧
    q₁.convex ∧
    q₂.convex ∧
    q₁.planar ∧
    q₂.planar ∧
    isInscribed q₁ q₂ ∧
    sharedVertices q₁ q₂ = 4 ∧
    countIntersections q₁ q₂ = 8 := by
  sorry

end max_intersections_square_decagon_l94_9449


namespace gis_not_just_computer_system_l94_9408

/-- Represents a Geographic Information System (GIS) -/
structure GIS where
  provides_decision_info : Bool
  used_in_urban_management : Bool
  has_data_functions : Bool
  is_computer_system : Bool

/-- The properties of a valid GIS based on the given conditions -/
def is_valid_gis (g : GIS) : Prop :=
  g.provides_decision_info ∧
  g.used_in_urban_management ∧
  g.has_data_functions ∧
  ¬g.is_computer_system

/-- The statement to be proven false -/
def incorrect_statement (g : GIS) : Prop :=
  g.is_computer_system

theorem gis_not_just_computer_system :
  ∃ (g : GIS), is_valid_gis g ∧ ¬incorrect_statement g :=
sorry

end gis_not_just_computer_system_l94_9408


namespace f_negative_five_l94_9472

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.tan x + 1

theorem f_negative_five (a b : ℝ) 
  (h : f a b 5 = 7) : 
  f a b (-5) = -5 := by
sorry

end f_negative_five_l94_9472


namespace isosceles_triangle_perimeter_l94_9409

theorem isosceles_triangle_perimeter : ∀ a b : ℝ,
  (a^2 - 6*a + 8 = 0) →
  (b^2 - 6*b + 8 = 0) →
  (a ≠ b) →
  (∃ c : ℝ, c = max a b ∧ c = min a b + (max a b - min a b) ∧ a + b + c = 10) :=
by sorry

end isosceles_triangle_perimeter_l94_9409


namespace triangle_trig_max_value_l94_9452

theorem triangle_trig_max_value (A C : ℝ) (h1 : 0 ≤ A ∧ A ≤ π) (h2 : 0 ≤ C ∧ C ≤ π) 
  (h3 : Real.sin A + Real.sin C = 3/2) :
  let t := 2 * Real.sin A * Real.sin C
  (∃ (x : ℝ), x = t * Real.sqrt ((9/4 - t) * (t - 1/4))) ∧
  (∀ (y : ℝ), y = t * Real.sqrt ((9/4 - t) * (t - 1/4)) → y ≤ 27 * Real.sqrt 7 / 64) ∧
  (∃ (z : ℝ), z = t * Real.sqrt ((9/4 - t) * (t - 1/4)) ∧ z = 27 * Real.sqrt 7 / 64) :=
by sorry

end triangle_trig_max_value_l94_9452


namespace greatest_value_l94_9431

theorem greatest_value (a b : ℝ) (ha : a = 2) (hb : b = 5) :
  let expr1 := a / b
  let expr2 := b / a
  let expr3 := a - b
  let expr4 := b - a
  let expr5 := (1 / 2) * a
  (expr4 ≥ expr1) ∧ (expr4 ≥ expr2) ∧ (expr4 ≥ expr3) ∧ (expr4 ≥ expr5) :=
by sorry

end greatest_value_l94_9431


namespace somus_age_l94_9427

theorem somus_age (s f : ℕ) (h1 : s = f / 3) (h2 : s - 9 = (f - 9) / 5) : s = 18 := by
  sorry

end somus_age_l94_9427


namespace island_length_calculation_l94_9455

/-- Represents the dimensions of a rectangular island -/
structure IslandDimensions where
  width : ℝ
  length : ℝ
  perimeter : ℝ

/-- Theorem: An island with width 4 miles and perimeter 22 miles has a length of 7 miles -/
theorem island_length_calculation (island : IslandDimensions) 
    (h1 : island.width = 4)
    (h2 : island.perimeter = 22)
    (h3 : island.perimeter = 2 * (island.length + island.width)) : 
  island.length = 7 := by
  sorry


end island_length_calculation_l94_9455


namespace f_of_2_equals_5_l94_9468

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x - 1

-- State the theorem
theorem f_of_2_equals_5 : f 2 = 5 := by
  sorry

end f_of_2_equals_5_l94_9468


namespace hypotenuse_squared_of_complex_zeros_l94_9473

-- Define the polynomial P(z)
def P (z : ℂ) : ℂ := z^3 - 2*z^2 + 2*z + 4

-- State the theorem
theorem hypotenuse_squared_of_complex_zeros (a b c : ℂ) :
  P a = 0 → P b = 0 → P c = 0 →
  Complex.abs a ^ 2 + Complex.abs b ^ 2 + Complex.abs c ^ 2 = 300 →
  (a - b).re * (c - b).re + (a - b).im * (c - b).im = 0 →
  (Complex.abs (b - c)) ^ 2 = 450 := by
  sorry

end hypotenuse_squared_of_complex_zeros_l94_9473


namespace rectangle_area_l94_9401

theorem rectangle_area (ratio_long : ℕ) (ratio_short : ℕ) (perimeter : ℕ) :
  ratio_long = 4 →
  ratio_short = 3 →
  perimeter = 126 →
  ∃ (length width : ℕ),
    length * ratio_short = width * ratio_long ∧
    2 * (length + width) = perimeter ∧
    length * width = 972 := by
  sorry

end rectangle_area_l94_9401


namespace victor_trips_l94_9442

/-- Calculate the number of trips needed to carry a given number of trays -/
def tripsNeeded (trays : ℕ) (capacity : ℕ) : ℕ :=
  (trays + capacity - 1) / capacity

/-- The problem setup -/
def victorProblem : Prop :=
  let capacity := 6
  let table1 := 23
  let table2 := 5
  let table3 := 12
  let table4 := 18
  let table5 := 27
  let totalTrips := tripsNeeded table1 capacity + tripsNeeded table2 capacity +
                    tripsNeeded table3 capacity + tripsNeeded table4 capacity +
                    tripsNeeded table5 capacity
  totalTrips = 15

theorem victor_trips : victorProblem := by
  sorry

end victor_trips_l94_9442


namespace evaluate_expression_l94_9460

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 2/3) (hz : z = -3) :
  x^3 * y^2 * z^2 = 1/16 := by
  sorry

end evaluate_expression_l94_9460


namespace negation_equivalence_l94_9415

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) := by
  sorry

end negation_equivalence_l94_9415


namespace lcm_gcd_relation_l94_9424

theorem lcm_gcd_relation (m n : ℕ) (h1 : m > n) (h2 : m > 0) (h3 : n > 0) 
  (h4 : Nat.lcm m n = 30 * Nat.gcd m n) 
  (h5 : (m - n) ∣ Nat.lcm m n) : 
  (m + n) / Nat.gcd m n = 11 := by
  sorry

end lcm_gcd_relation_l94_9424


namespace quadratic_inequality_solution_implies_function_ordering_l94_9417

/-- The quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution_implies_function_ordering
  (a b c : ℝ)
  (h : ∀ x, a * x^2 + b * x + c > 0 ↔ -2 < x ∧ x < 4) :
  f a b c 5 < f a b c (-1) ∧ f a b c (-1) < f a b c 2 :=
by sorry

end quadratic_inequality_solution_implies_function_ordering_l94_9417


namespace exam_mean_score_l94_9496

theorem exam_mean_score (SD : ℝ) :
  (∃ M : ℝ, (58 = M - 2 * SD) ∧ (98 = M + 3 * SD)) → 
  (∃ M : ℝ, (58 = M - 2 * SD) ∧ (98 = M + 3 * SD) ∧ M = 74) :=
by sorry

end exam_mean_score_l94_9496


namespace relic_age_conversion_l94_9464

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

/-- The octal representation of the relic's age --/
def relic_age_octal : List Nat := [4, 6, 5, 7]

theorem relic_age_conversion :
  octal_to_decimal relic_age_octal = 3956 := by
  sorry

end relic_age_conversion_l94_9464


namespace a_property_l94_9486

def gcd_notation (a b : ℕ) : ℕ := Nat.gcd a b

theorem a_property (a : ℕ) : 
  gcd_notation (gcd_notation a 16) (gcd_notation 18 24) = 2 → 
  Even a ∧ ¬(4 ∣ a) := by
  sorry

end a_property_l94_9486


namespace fixed_point_on_line_l94_9493

theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end fixed_point_on_line_l94_9493


namespace exists_coprime_sequence_l94_9434

theorem exists_coprime_sequence : ∃ (a : ℕ → ℕ), 
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ i j p q r, i ≠ j ∧ i ≠ p ∧ i ≠ q ∧ i ≠ r ∧ j ≠ p ∧ j ≠ q ∧ j ≠ r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r → 
    Nat.gcd (a i + a j) (a p + a q + a r) = 1) :=
by sorry

end exists_coprime_sequence_l94_9434


namespace range_of_m_l94_9447

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - m ≥ 0

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m*x + 1 > 0

-- Theorem statement
theorem range_of_m (m : ℝ) :
  p m ∧ q m → -2 < m ∧ m ≤ 1 :=
by
  sorry


end range_of_m_l94_9447


namespace distance_after_pie_is_18_l94_9469

/-- Calculates the distance driven after buying pie and before stopping for gas -/
def distance_after_pie (total_distance : ℕ) (distance_before_pie : ℕ) (remaining_distance : ℕ) : ℕ :=
  total_distance - distance_before_pie - remaining_distance

/-- Proves that the distance driven after buying pie and before stopping for gas is 18 miles -/
theorem distance_after_pie_is_18 :
  distance_after_pie 78 35 25 = 18 := by
  sorry

end distance_after_pie_is_18_l94_9469


namespace train_platform_passing_time_l94_9400

/-- Given a train of length 2400 meters that takes 60 seconds to pass a point,
    calculate the time required for the same train to pass a platform of length 800 meters. -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (time_to_pass_point : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 2400)
  (h2 : time_to_pass_point = 60)
  (h3 : platform_length = 800) :
  (train_length + platform_length) / (train_length / time_to_pass_point) = 80 := by
  sorry

#check train_platform_passing_time

end train_platform_passing_time_l94_9400
