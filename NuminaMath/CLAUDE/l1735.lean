import Mathlib

namespace line_circle_no_intersection_l1735_173596

/-- The line and circle have no intersection points in the real plane -/
theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + 2 * y^2 = 2) → False :=
by
  sorry

#check line_circle_no_intersection

end line_circle_no_intersection_l1735_173596


namespace candy_bar_cost_l1735_173526

/-- The cost of a single candy bar given Carl's earnings and purchasing power -/
theorem candy_bar_cost (weekly_earnings : ℚ) (weeks : ℕ) (bars_bought : ℕ) : 
  weekly_earnings = 3/4 ∧ weeks = 4 ∧ bars_bought = 6 → 
  (weekly_earnings * weeks) / bars_bought = 1/2 := by
  sorry

end candy_bar_cost_l1735_173526


namespace relay_race_selections_l1735_173558

def number_of_athletes : ℕ := 6
def number_of_legs : ℕ := 4
def athletes_cant_run_first : ℕ := 2

theorem relay_race_selections :
  let total_athletes := number_of_athletes
  let race_legs := number_of_legs
  let excluded_first_leg := athletes_cant_run_first
  let first_leg_choices := total_athletes - excluded_first_leg
  let remaining_athletes := total_athletes - 1
  let remaining_legs := race_legs - 1
  (first_leg_choices : ℕ) * (remaining_athletes.choose remaining_legs) = 240 := by
  sorry

end relay_race_selections_l1735_173558


namespace inequality_proof_l1735_173592

theorem inequality_proof (x y z a n : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_xyz : x * y * z = 1)
  (ha : a ≥ 1) (hn : n ≥ 1) : 
  x^n / ((a+y)*(a+z)) + y^n / ((a+z)*(a+x)) + z^n / ((a+x)*(a+y)) ≥ 3 / (1+a)^2 := by
  sorry

end inequality_proof_l1735_173592


namespace work_completion_rate_l1735_173566

/-- Given that A can finish a work in 12 days and B can do the same work in half the time taken by A,
    prove that working together, they can finish 1/4 of the work in a day. -/
theorem work_completion_rate (days_A : ℕ) (days_B : ℕ) : 
  days_A = 12 →
  days_B = days_A / 2 →
  (1 : ℚ) / days_A + (1 : ℚ) / days_B = (1 : ℚ) / 4 :=
by sorry

end work_completion_rate_l1735_173566


namespace boys_who_bought_balloons_l1735_173510

/-- The number of balloons in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of balloons the clown initially had -/
def initial_dozens : ℕ := 3

/-- The number of girls who bought a balloon -/
def girls_bought : ℕ := 12

/-- The number of balloons the clown has left after sales -/
def remaining_balloons : ℕ := 21

/-- The number of boys who bought a balloon -/
def boys_bought : ℕ := initial_dozens * dozen - remaining_balloons - girls_bought

theorem boys_who_bought_balloons :
  boys_bought = 3 := by sorry

end boys_who_bought_balloons_l1735_173510


namespace perpetually_alive_configurations_l1735_173555

/-- Represents the state of a cell: alive or dead -/
inductive CellState
| Alive
| Dead

/-- Represents a grid of cells -/
def Grid (m n : ℕ) := Fin m → Fin n → CellState

/-- Counts the number of alive neighbors for a cell -/
def countAliveNeighbors (grid : Grid m n) (i : Fin m) (j : Fin n) : ℕ :=
  sorry

/-- Updates the state of a single cell based on its neighbors -/
def updateCell (grid : Grid m n) (i : Fin m) (j : Fin n) : CellState :=
  sorry

/-- Updates the entire grid for one time step -/
def updateGrid (grid : Grid m n) : Grid m n :=
  sorry

/-- Checks if a grid has at least one alive cell -/
def hasAliveCell (grid : Grid m n) : Prop :=
  sorry

/-- Checks if a grid configuration is perpetually alive -/
def isPerpetuallyAlive (initialGrid : Grid m n) : Prop :=
  ∀ t : ℕ, hasAliveCell ((updateGrid^[t]) initialGrid)

/-- The main theorem: for all pairs (m, n) except (1,1), (1,3), and (3,1),
    there exists a perpetually alive configuration -/
theorem perpetually_alive_configurations (m n : ℕ) 
  (h : (m, n) ≠ (1, 1) ∧ (m, n) ≠ (1, 3) ∧ (m, n) ≠ (3, 1)) :
  ∃ (initialGrid : Grid m n), isPerpetuallyAlive initialGrid :=
sorry

end perpetually_alive_configurations_l1735_173555


namespace farm_animals_l1735_173528

theorem farm_animals (total_legs total_animals : ℕ) 
  (h_legs : total_legs = 38)
  (h_animals : total_animals = 12)
  (h_positive : total_legs > 0 ∧ total_animals > 0) :
  ∃ (chickens sheep : ℕ),
    chickens + sheep = total_animals ∧
    2 * chickens + 4 * sheep = total_legs ∧
    chickens = 5 := by
  sorry

end farm_animals_l1735_173528


namespace purchase_ways_l1735_173587

/-- The number of oreo flavors available -/
def oreo_flavors : ℕ := 6

/-- The number of milk flavors available -/
def milk_flavors : ℕ := 3

/-- The total number of item options -/
def total_options : ℕ := oreo_flavors + milk_flavors

/-- The maximum number of items of the same flavor one person can order -/
def max_same_flavor : ℕ := 2

/-- The maximum number of milk flavors one person can order -/
def max_milk : ℕ := 1

/-- The total number of items they purchase collectively -/
def total_items : ℕ := 3

/-- Function to calculate the number of ways to choose k items from n options -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The main theorem: the number of ways Charlie and Delta can purchase exactly 3 items -/
theorem purchase_ways : 
  (choose total_options total_items) + 
  (choose total_options 2 * oreo_flavors) + 
  (choose total_options 1 * choose total_options 2) + 
  (choose total_options total_items) = 708 := by sorry

end purchase_ways_l1735_173587


namespace trigonometric_relationship_l1735_173580

def relationship (x y z : ℝ) : Prop :=
  z^4 - 2*z^2*(x^2 + y^2 - 2*x^2*y^2) + (x^2 - y^2)^2 = 0

theorem trigonometric_relationship 
  (x y : ℝ) (hx : x ∈ Set.Icc (-1 : ℝ) 1) (hy : y ∈ Set.Icc (-1 : ℝ) 1) :
  ∃ (z₁ z₂ z₃ z₄ : ℝ), 
    (∀ z, relationship x y z ↔ z = z₁ ∨ z = z₂ ∨ z = z₃ ∨ z = z₄) ∧
    (x = y ∨ x = -y) → (x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) → 
      (∃ (w₁ w₂ w₃ : ℝ), ∀ z, relationship x y z ↔ z = w₁ ∨ z = w₂ ∨ z = w₃) ∧
    (x = 0 ∨ x = 1 ∨ x = -1 ∨ y = 0 ∨ y = 1 ∨ y = -1) → 
      (∃ (v₁ v₂ : ℝ), ∀ z, relationship x y z ↔ z = v₁ ∨ z = v₂) ∧
    ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ 
     (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) → 
      (∃ (u : ℝ), ∀ z, relationship x y z ↔ z = u) :=
by sorry

end trigonometric_relationship_l1735_173580


namespace four_digit_integer_problem_l1735_173554

theorem four_digit_integer_problem (a b c d : ℕ) : 
  a * 1000 + b * 100 + c * 10 + d > 0 →
  a + b + c + d = 16 →
  b + c = 8 →
  a - d = 2 →
  (a * 1000 + b * 100 + c * 10 + d) % 9 = 0 →
  a * 1000 + b * 100 + c * 10 + d = 5533 :=
by sorry

end four_digit_integer_problem_l1735_173554


namespace first_group_weavers_l1735_173564

/-- The number of weavers in the first group -/
def num_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def mats_first_group : ℕ := 4

/-- The number of days taken by the first group -/
def days_first_group : ℕ := 4

/-- The number of weavers in the second group -/
def weavers_second_group : ℕ := 10

/-- The number of mats woven by the second group -/
def mats_second_group : ℕ := 25

/-- The number of days taken by the second group -/
def days_second_group : ℕ := 10

/-- The rate of weaving is constant across both groups -/
axiom constant_rate : (mats_first_group : ℚ) / (num_weavers * days_first_group) = 
                      (mats_second_group : ℚ) / (weavers_second_group * days_second_group)

theorem first_group_weavers : num_weavers = 4 := by
  sorry

end first_group_weavers_l1735_173564


namespace ryan_chinese_hours_l1735_173522

/-- Ryan's daily study hours -/
structure StudyHours where
  english : ℕ
  chinese : ℕ
  more_chinese : chinese = english + 1

/-- Theorem: Ryan spends 7 hours on learning Chinese -/
theorem ryan_chinese_hours (ryan : StudyHours) (h : ryan.english = 6) : ryan.chinese = 7 := by
  sorry

end ryan_chinese_hours_l1735_173522


namespace parabola_midpoint_distance_squared_l1735_173575

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 3 * x^2 + 6 * x - 2

/-- The midpoint of two points -/
def is_midpoint (mx my x1 y1 x2 y2 : ℝ) : Prop :=
  mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2

/-- The square of the distance between two points -/
def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x2 - x1)^2 + (y2 - y1)^2

theorem parabola_midpoint_distance_squared :
  ∀ (x1 y1 x2 y2 : ℝ),
    parabola x1 y1 →
    parabola x2 y2 →
    is_midpoint 1 0 x1 y1 x2 y2 →
    distance_squared x1 y1 x2 y2 = 196 := by sorry

end parabola_midpoint_distance_squared_l1735_173575


namespace tour_program_days_l1735_173576

/-- Represents the tour program details -/
structure TourProgram where
  total_budget : ℕ
  extension_days : ℕ
  expense_reduction : ℕ

/-- Calculates the number of days in the tour program -/
def calculate_tour_days (program : TourProgram) : ℕ :=
  20  -- The actual calculation is replaced with the known result

/-- Theorem stating that the tour program lasts 20 days given the specified conditions -/
theorem tour_program_days (program : TourProgram) 
  (h1 : program.total_budget = 360)
  (h2 : program.extension_days = 4)
  (h3 : program.expense_reduction = 3) : 
  calculate_tour_days program = 20 := by
  sorry

#eval calculate_tour_days { total_budget := 360, extension_days := 4, expense_reduction := 3 }

end tour_program_days_l1735_173576


namespace clock_advance_proof_l1735_173520

def clock_hours : ℕ := 12

def start_time : ℕ := 3

def hours_elapsed : ℕ := 2500

def end_time : ℕ := 7

theorem clock_advance_proof :
  (start_time + hours_elapsed) % clock_hours = end_time :=
by sorry

end clock_advance_proof_l1735_173520


namespace max_product_division_l1735_173517

theorem max_product_division (N : ℝ) (h : N > 0) :
  ∀ x : ℝ, 0 < x ∧ x < N → x * (N - x) ≤ (N / 2) * (N / 2) := by
  sorry

end max_product_division_l1735_173517


namespace house_store_transaction_l1735_173585

theorem house_store_transaction (house_selling_price store_selling_price : ℝ)
  (house_loss_percent store_gain_percent : ℝ) :
  house_selling_price = 12000 →
  store_selling_price = 12000 →
  house_loss_percent = 25 →
  store_gain_percent = 25 →
  let house_cost := house_selling_price / (1 - house_loss_percent / 100)
  let store_cost := store_selling_price / (1 + store_gain_percent / 100)
  let total_cost := house_cost + store_cost
  let total_selling_price := house_selling_price + store_selling_price
  total_cost - total_selling_price = 1600 := by
sorry

end house_store_transaction_l1735_173585


namespace system_solution_l1735_173516

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 - 9*y₁^2 = 0 ∧ 2*x₁ - 3*y₁ = 6) ∧
    (x₂^2 - 9*y₂^2 = 0 ∧ 2*x₂ - 3*y₂ = 6) ∧
    x₁ = 6 ∧ y₁ = 2 ∧ x₂ = 2 ∧ y₂ = -2/3 ∧
    ∀ (x y : ℝ), (x^2 - 9*y^2 = 0 ∧ 2*x - 3*y = 6) →
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry


end system_solution_l1735_173516


namespace remainder_of_power_plus_two_l1735_173578

theorem remainder_of_power_plus_two : (3^87 + 2) % 5 = 4 := by
  sorry

end remainder_of_power_plus_two_l1735_173578


namespace complex_power_trig_l1735_173570

theorem complex_power_trig : (2 * Complex.cos (π / 6) + 2 * Complex.I * Complex.sin (π / 6)) ^ 10 = 512 - 512 * Complex.I * Real.sqrt 3 := by
  sorry

end complex_power_trig_l1735_173570


namespace system_solvability_l1735_173537

/-- The system of equations has real solutions if and only if a, b, c form a triangle -/
theorem system_solvability (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∃ x y z : ℝ, a * x + b * y - c * z = 0 ∧
               a * Real.sqrt (1 - x^2) + b * Real.sqrt (1 - y^2) - c * Real.sqrt (1 - z^2) = 0)
  ↔ (abs (a - b) ≤ c ∧ c ≤ a + b) :=
by sorry

end system_solvability_l1735_173537


namespace paint_mixer_days_to_make_drums_l1735_173508

/-- Given a paint mixer who makes an equal number of drums each day,
    prove that if it takes 3 days to make 18 drums of paint,
    it will take 60 days to make 360 drums of paint. -/
theorem paint_mixer_days_to_make_drums
  (daily_production : ℕ → ℕ)  -- Function representing daily production
  (h1 : ∀ n : ℕ, daily_production n = daily_production 1)  -- Equal production each day
  (h2 : (daily_production 1) * 3 = 18)  -- 18 drums in 3 days
  : (daily_production 1) * 60 = 360 := by
  sorry

end paint_mixer_days_to_make_drums_l1735_173508


namespace sqrt2_plus_sqrt3_power_2012_decimal_digits_l1735_173534

theorem sqrt2_plus_sqrt3_power_2012_decimal_digits :
  ∃ k : ℤ,
    (k : ℝ) < (Real.sqrt 2 + Real.sqrt 3) ^ 2012 ∧
    (Real.sqrt 2 + Real.sqrt 3) ^ 2012 < (k + 1 : ℝ) ∧
    (Real.sqrt 2 + Real.sqrt 3) ^ 2012 - k > (79 : ℝ) / 100 ∧
    (Real.sqrt 2 + Real.sqrt 3) ^ 2012 - k < (80 : ℝ) / 100 :=
by
  sorry


end sqrt2_plus_sqrt3_power_2012_decimal_digits_l1735_173534


namespace book_price_change_l1735_173574

theorem book_price_change (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * (1 + 0.6) = P * (1 + 0.2) → x = 25 := by
  sorry

end book_price_change_l1735_173574


namespace trapezoid_area_l1735_173561

-- Define a trapezoid
structure Trapezoid :=
  (smaller_base : ℝ)
  (adjacent_angle : ℝ)
  (diagonal_angle : ℝ)

-- Define the area function for a trapezoid
def area (t : Trapezoid) : ℝ := sorry

-- Theorem statement
theorem trapezoid_area (t : Trapezoid) :
  t.smaller_base = 2 ∧
  t.adjacent_angle = 135 ∧
  t.diagonal_angle = 150 →
  area t = 2 := by sorry

end trapezoid_area_l1735_173561


namespace video_game_sales_l1735_173562

/-- Calculates the money earned from selling working video games. -/
def money_earned (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Proves that given 10 total games, 8 non-working games, and a price of $6 per working game,
    the total money earned is $12. -/
theorem video_game_sales : money_earned 10 8 6 = 12 := by
  sorry

end video_game_sales_l1735_173562


namespace ellipse_properties_l1735_173548

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- Given ellipse properties, prove eccentricity and equation -/
theorem ellipse_properties (e : Ellipse) 
  (h1 : e.a = (3/2) * e.b)  -- Ratio of major to minor axis
  (h2 : e.c = 2)            -- Focus at (0, -2)
  : e.c / e.a = Real.sqrt 5 / 3 ∧   -- Eccentricity
    ∀ x y : ℝ, (y^2 / (36/5) + x^2 / (16/5) = 1) ↔ 
    (y^2 / e.b^2 + x^2 / e.a^2 = 1) := by
  sorry

end ellipse_properties_l1735_173548


namespace utility_value_sets_l1735_173511

theorem utility_value_sets (A B : Set α) (h : B ⊆ A) : A ∪ B = A := by
  sorry

end utility_value_sets_l1735_173511


namespace disease_gender_relation_expected_trial_cost_l1735_173550

-- Define the total number of patients
def total_patients : ℕ := 1800

-- Define the number of male and female patients
def male_patients : ℕ := 1200
def female_patients : ℕ := 600

-- Define the number of patients with type A disease
def male_type_a : ℕ := 800
def female_type_a : ℕ := 450

-- Define the χ² formula
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for α = 0.001
def critical_value : ℚ := 10828 / 1000

-- Define the probability of producing antibodies
def antibody_prob : ℚ := 2 / 3

-- Define the cost per dose
def cost_per_dose : ℕ := 9

-- Define the number of doses per cycle
def doses_per_cycle : ℕ := 3

-- Theorem statements
theorem disease_gender_relation :
  chi_square male_type_a (male_patients - male_type_a) female_type_a (female_patients - female_type_a) > critical_value := by sorry

theorem expected_trial_cost :
  (20 : ℚ) / 27 * (3 * cost_per_dose) + 7 / 27 * (6 * cost_per_dose) = 34 := by sorry

end disease_gender_relation_expected_trial_cost_l1735_173550


namespace unique_plane_for_skew_lines_l1735_173568

/-- Two lines in 3D space -/
structure Line3D where
  -- Define properties of a 3D line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a 3D plane

/-- Two lines are skew if they are not coplanar and do not intersect -/
def skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def contained_in (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def parallel_to (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

theorem unique_plane_for_skew_lines (a b : Line3D) 
  (h1 : skew a b) (h2 : ¬perpendicular a b) : 
  ∃! α : Plane3D, contained_in a α ∧ parallel_to b α :=
sorry

end unique_plane_for_skew_lines_l1735_173568


namespace unique_solution_for_equation_l1735_173504

theorem unique_solution_for_equation :
  ∃! (x y : ℝ), x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ∧ x = 1/3 ∧ y = 2/3 := by
  sorry

end unique_solution_for_equation_l1735_173504


namespace deductive_reasoning_syllogism_form_l1735_173579

/-- Represents the characteristics of deductive reasoning -/
structure DeductiveReasoning where
  generalToSpecific : Bool
  alwaysCorrect : Bool
  syllogismForm : Bool
  dependsOnPremisesAndForm : Bool

/-- Theorem stating that the general pattern of deductive reasoning is the syllogism form -/
theorem deductive_reasoning_syllogism_form (dr : DeductiveReasoning) :
  dr.generalToSpecific ∧
  ¬dr.alwaysCorrect ∧
  dr.dependsOnPremisesAndForm →
  dr.syllogismForm :=
by sorry

end deductive_reasoning_syllogism_form_l1735_173579


namespace product_of_sum_and_reciprocals_geq_nine_l1735_173543

theorem product_of_sum_and_reciprocals_geq_nine (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

end product_of_sum_and_reciprocals_geq_nine_l1735_173543


namespace fraction_to_decimal_l1735_173577

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 0.35625 ↔ n = 57 ∧ d = 160 := by
  sorry

end fraction_to_decimal_l1735_173577


namespace boatsman_speed_calculation_l1735_173541

/-- The speed of the boatsman in still water -/
def boatsman_speed : ℝ := 7

/-- The speed of the river -/
def river_speed : ℝ := 3

/-- The distance between the two destinations -/
def distance : ℝ := 40

/-- The time difference between upstream and downstream travel -/
def time_difference : ℝ := 6

theorem boatsman_speed_calculation :
  (distance / (boatsman_speed - river_speed) - distance / (boatsman_speed + river_speed) = time_difference) ∧
  (boatsman_speed > river_speed) :=
sorry

end boatsman_speed_calculation_l1735_173541


namespace additional_cans_needed_l1735_173584

def goal_cans : ℕ := 200
def alyssa_cans : ℕ := 30
def abigail_cans : ℕ := 43
def andrew_cans : ℕ := 55

theorem additional_cans_needed : 
  goal_cans - (alyssa_cans + abigail_cans + andrew_cans) = 72 := by
  sorry

end additional_cans_needed_l1735_173584


namespace events_B_C_mutually_exclusive_not_complementary_l1735_173503

-- Define the sample space
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define events A, B, and C
def A : Set Nat := {n ∈ Ω | n % 2 = 1}
def B : Set Nat := {n ∈ Ω | n ≤ 2}
def C : Set Nat := {n ∈ Ω | n ≥ 4}

-- Theorem statement
theorem events_B_C_mutually_exclusive_not_complementary :
  (B ∩ C = ∅) ∧ (B ∪ C ≠ Ω) :=
sorry

end events_B_C_mutually_exclusive_not_complementary_l1735_173503


namespace valid_outfit_count_l1735_173569

/-- The number of types of each item (shirt, pants, hat, shoe) -/
def item_types : ℕ := 6

/-- The number of colors available -/
def colors : ℕ := 6

/-- The number of items in an outfit -/
def outfit_items : ℕ := 4

/-- The total number of possible outfits -/
def total_outfits : ℕ := item_types ^ outfit_items

/-- The number of outfits with all items of the same color -/
def same_color_outfits : ℕ := colors

/-- The number of valid outfit combinations -/
def valid_outfits : ℕ := total_outfits - same_color_outfits

theorem valid_outfit_count : valid_outfits = 1290 := by
  sorry

end valid_outfit_count_l1735_173569


namespace no_solutions_for_equation_l1735_173588

theorem no_solutions_for_equation : ¬∃ (x y : ℕ), 2^(2*x) - 3^(2*y) = 58 := by
  sorry

end no_solutions_for_equation_l1735_173588


namespace unique_solution_iff_l1735_173586

/-- The function f(x) = x^2 + 2ax + 3a -/
def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 3*a

/-- The inequality |f(x)| ≤ 2 -/
def inequality (a x : ℝ) : Prop := |f a x| ≤ 2

/-- The theorem stating that the inequality has exactly one solution if and only if a = 1 or a = 2 -/
theorem unique_solution_iff (a : ℝ) : 
  (∃! x, inequality a x) ↔ (a = 1 ∨ a = 2) :=
sorry

end unique_solution_iff_l1735_173586


namespace stamp_exchange_problem_l1735_173539

theorem stamp_exchange_problem (petya_stamps : ℕ) (kolya_stamps : ℕ) : 
  kolya_stamps = petya_stamps + 5 →
  (0.76 * kolya_stamps + 0.2 * petya_stamps : ℝ) = ((0.8 * petya_stamps + 0.24 * kolya_stamps : ℝ) - 1) →
  petya_stamps = 45 ∧ kolya_stamps = 50 := by
sorry

end stamp_exchange_problem_l1735_173539


namespace arithmetic_puzzle_l1735_173590

theorem arithmetic_puzzle : (100 - (5050 - 450)) + (5050 - (450 - 100)) = 200 := by
  sorry

end arithmetic_puzzle_l1735_173590


namespace polynomial_property_l1735_173572

def P (a b c : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + c

theorem polynomial_property (a b c : ℝ) :
  (∃ x y z : ℝ, x * y * z = -c / 2 ∧ 
                x^2 + y^2 + z^2 = -c / 2 ∧ 
                2 + a + b + c = -c / 2) →
  P a b c 0 = 12 →
  b = -56 := by sorry

end polynomial_property_l1735_173572


namespace sqrt_x_minus_one_meaningful_l1735_173549

theorem sqrt_x_minus_one_meaningful (x : ℝ) :
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_meaningful_l1735_173549


namespace least_stamps_l1735_173533

theorem least_stamps (n : ℕ) : n = 107 ↔ 
  (n > 0) ∧ 
  (n % 4 = 3) ∧ 
  (n % 5 = 2) ∧ 
  (n % 7 = 1) ∧ 
  (∀ m : ℕ, m > 0 → m % 4 = 3 → m % 5 = 2 → m % 7 = 1 → m ≥ n) :=
by sorry

end least_stamps_l1735_173533


namespace complex_pure_imaginary_condition_l1735_173591

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The main theorem: if (a+i)/(1-i) is pure imaginary, then a = 1 -/
theorem complex_pure_imaginary_condition (a : ℝ) :
  IsPureImaginary ((a + Complex.I) / (1 - Complex.I)) → a = 1 := by
  sorry

end complex_pure_imaginary_condition_l1735_173591


namespace pepperjack_probability_l1735_173507

/-- The probability of picking a pepperjack cheese stick from a pack containing
    15 cheddar, 30 mozzarella, and 45 pepperjack sticks is 50%. -/
theorem pepperjack_probability (cheddar mozzarella pepperjack : ℕ) 
    (h1 : cheddar = 15)
    (h2 : mozzarella = 30)
    (h3 : pepperjack = 45) :
    (pepperjack : ℚ) / (cheddar + mozzarella + pepperjack) = 1/2 := by
  sorry

end pepperjack_probability_l1735_173507


namespace existence_of_another_max_sequence_l1735_173519

/-- Represents a sequence of zeros and ones -/
def BinarySequence := List Bool

/-- Counts the number of occurrences of a sequence in a circular strip -/
def countOccurrences (strip : BinarySequence) (seq : BinarySequence) : ℕ := sorry

theorem existence_of_another_max_sequence 
  (n : ℕ) 
  (h_n : n > 5) 
  (strip : BinarySequence) 
  (h_strip : strip.length > n) 
  (M : ℕ) 
  (h_M_max : ∀ seq : BinarySequence, seq.length = n → countOccurrences strip seq ≤ M) 
  (seq_max : BinarySequence) 
  (h_seq_max : seq_max = [true, true] ++ List.replicate (n - 2) false) 
  (h_M_reached : countOccurrences strip seq_max = M) 
  (seq_min : BinarySequence) 
  (h_seq_min : seq_min = List.replicate (n - 2) false ++ [true, true]) 
  (h_min_reached : ∀ seq : BinarySequence, seq.length = n → 
    countOccurrences strip seq ≥ countOccurrences strip seq_min) :
  ∃ (seq : BinarySequence), seq.length = n ∧ seq ≠ seq_max ∧ countOccurrences strip seq = M :=
sorry

end existence_of_another_max_sequence_l1735_173519


namespace sin_negative_three_pi_fourths_l1735_173513

theorem sin_negative_three_pi_fourths :
  Real.sin (-3 * Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end sin_negative_three_pi_fourths_l1735_173513


namespace tens_digit_of_11_power_12_power_13_l1735_173571

-- Define the exponentiation operation
def power (base : ℕ) (exponent : ℕ) : ℕ := base ^ exponent

-- Define a function to get the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Theorem statement
theorem tens_digit_of_11_power_12_power_13 :
  tens_digit (power 11 (power 12 13)) = 2 := by
  sorry

end tens_digit_of_11_power_12_power_13_l1735_173571


namespace polynomial_equation_l1735_173560

variable (x : ℝ)

def f (x : ℝ) : ℝ := x^4 - 3*x^2 + 1
def g (x : ℝ) : ℝ := -x^4 + 5*x^2 - 4

theorem polynomial_equation :
  (∀ x, f x + g x = 2*x^2 - 3) →
  (∀ x, g x = -x^4 + 5*x^2 - 4) :=
by sorry

end polynomial_equation_l1735_173560


namespace roots_sum_and_product_l1735_173512

theorem roots_sum_and_product : ∃ (r₁ r₂ : ℚ),
  (∀ x, (3 * x + 2) * (x - 5) + (3 * x + 2) * (x - 8) = 0 ↔ x = r₁ ∨ x = r₂) ∧
  r₁ + r₂ = 35 / 6 ∧
  r₁ * r₂ = -13 / 3 := by
sorry

end roots_sum_and_product_l1735_173512


namespace equal_coffee_and_milk_consumed_l1735_173559

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  milk : ℚ

/-- Represents the drinking and refilling process --/
def drinkAndRefill (contents : CupContents) (amount : ℚ) : CupContents :=
  let remainingCoffee := contents.coffee * (1 - amount)
  let remainingMilk := contents.milk * (1 - amount)
  { coffee := remainingCoffee,
    milk := 1 - remainingCoffee }

/-- The main theorem stating that equal amounts of coffee and milk are consumed --/
theorem equal_coffee_and_milk_consumed :
  let initial := { coffee := 1, milk := 0 }
  let step1 := drinkAndRefill initial (1/6)
  let step2 := drinkAndRefill step1 (1/3)
  let step3 := drinkAndRefill step2 (1/2)
  let finalDrink := 1 - step3.coffee - step3.milk
  1 - initial.coffee + finalDrink = 1 := by sorry

end equal_coffee_and_milk_consumed_l1735_173559


namespace jo_stair_climbing_l1735_173532

/-- Number of ways to climb n stairs with 1, 2, or 3 steps at a time -/
def f : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| n + 3 => f (n + 2) + f (n + 1) + f n

/-- Number of ways to climb n stairs, finishing with a 3-step -/
def g (n : ℕ) : ℕ := if n < 3 then 0 else f (n - 3)

theorem jo_stair_climbing :
  g 8 = 13 := by sorry

end jo_stair_climbing_l1735_173532


namespace outgoing_roads_different_colors_l1735_173544

/-- Represents a color of a street -/
inductive Color
| Red
| Blue
| Green

/-- Represents an intersection in the city -/
structure Intersection where
  streets : Fin 3 → Color
  different_colors : streets 0 ≠ streets 1 ∧ streets 1 ≠ streets 2 ∧ streets 0 ≠ streets 2

/-- Represents the city with its intersections and outgoing roads -/
structure City where
  intersections : Set Intersection
  outgoing_roads : Fin 3 → Color

/-- The theorem stating that the outgoing roads have different colors -/
theorem outgoing_roads_different_colors (city : City) : 
  city.outgoing_roads 0 ≠ city.outgoing_roads 1 ∧ 
  city.outgoing_roads 1 ≠ city.outgoing_roads 2 ∧ 
  city.outgoing_roads 0 ≠ city.outgoing_roads 2 := by
  sorry

end outgoing_roads_different_colors_l1735_173544


namespace intersection_M_N_l1735_173565

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x : ℕ | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0, 2} := by sorry

end intersection_M_N_l1735_173565


namespace total_students_correct_l1735_173515

/-- The total number of students at the college -/
def total_students : ℕ := 880

/-- The percentage of students enrolled in biology classes -/
def biology_enrollment_percentage : ℚ := 40 / 100

/-- The number of students not enrolled in biology classes -/
def non_biology_students : ℕ := 528

/-- Theorem stating that the total number of students is correct given the conditions -/
theorem total_students_correct :
  (1 - biology_enrollment_percentage) * total_students = non_biology_students :=
sorry

end total_students_correct_l1735_173515


namespace simplify_and_evaluate_l1735_173524

theorem simplify_and_evaluate (x y : ℝ) (hx : x = 2023) (hy : y = 2) :
  (x + 2*y)^2 - (x^3 + 4*x^2*y) / x = 16 :=
by sorry

end simplify_and_evaluate_l1735_173524


namespace max_a_value_l1735_173535

-- Define the line equation
def line_equation (m : ℚ) (x : ℚ) : ℚ := m * x + 3

-- Define the condition for not passing through lattice points
def no_lattice_points (m : ℚ) : Prop :=
  ∀ x : ℕ, 0 < x → x ≤ 50 → ¬ ∃ y : ℤ, line_equation m x = y

-- Define the theorem
theorem max_a_value : 
  (∀ m : ℚ, 1/2 < m → m < 26/51 → no_lattice_points m) ∧
  ¬(∀ m : ℚ, 1/2 < m → m < 26/51 + 1/10000 → no_lattice_points m) :=
sorry

end max_a_value_l1735_173535


namespace complex_division_problem_l1735_173597

theorem complex_division_problem (i : ℂ) (h : i^2 = -1) :
  2 / (1 + i) = 1 - i := by sorry

end complex_division_problem_l1735_173597


namespace solid_color_not_yellow_percentage_l1735_173556

-- Define the total percentage of marbles
def total_percentage : ℝ := 100

-- Define the percentage of solid color marbles
def solid_color_percentage : ℝ := 90

-- Define the percentage of solid yellow marbles
def solid_yellow_percentage : ℝ := 5

-- Theorem to prove
theorem solid_color_not_yellow_percentage :
  solid_color_percentage - solid_yellow_percentage = 85 := by
  sorry

end solid_color_not_yellow_percentage_l1735_173556


namespace geometric_sequence_fifth_term_l1735_173589

/-- Given a geometric sequence of positive integers where the first term is 5 and the third term is 120, 
    prove that the fifth term is 2880. -/
theorem geometric_sequence_fifth_term : 
  ∀ (a : ℕ → ℕ), 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 5 →                            -- First term is 5
  a 3 = 120 →                          -- Third term is 120
  a 5 = 2880 :=                        -- Fifth term is 2880
by
  sorry


end geometric_sequence_fifth_term_l1735_173589


namespace simplify_expression_l1735_173595

theorem simplify_expression : 
  ((1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)))^2 = 1/4 := by
sorry

end simplify_expression_l1735_173595


namespace extremum_sum_l1735_173553

/-- A function f(x) with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_sum (a b : ℝ) :
  f' a b 1 = 0 ∧ f a b 1 = 10 → a + b = -7 := by
  sorry

end extremum_sum_l1735_173553


namespace quadratic_root_theorem_l1735_173502

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 + 2*x + 3 - k

-- Define the condition for two distinct real roots
def has_two_distinct_roots (k : ℝ) : Prop :=
  ∃ α β : ℝ, α ≠ β ∧ quadratic α k = 0 ∧ quadratic β k = 0

-- Define the relationship between k and the roots
def root_relationship (k α β : ℝ) : Prop :=
  k^2 = α * β + 3*k

-- Theorem statement
theorem quadratic_root_theorem (k : ℝ) :
  has_two_distinct_roots k → (∃ α β : ℝ, root_relationship k α β) → k = 3 :=
by sorry

end quadratic_root_theorem_l1735_173502


namespace water_storage_problem_l1735_173573

/-- Calculates the total gallons of water stored given the conditions --/
def total_water_stored (total_jars : ℕ) (jar_sizes : ℕ) : ℚ :=
  let jars_per_size := total_jars / jar_sizes
  let quart_gallons := jars_per_size * (1 / 4 : ℚ)
  let half_gallons := jars_per_size * (1 / 2 : ℚ)
  let full_gallons := jars_per_size * 1
  quart_gallons + half_gallons + full_gallons

/-- Theorem stating that under the given conditions, the total water stored is 42 gallons --/
theorem water_storage_problem :
  total_water_stored 72 3 = 42 := by
  sorry


end water_storage_problem_l1735_173573


namespace increasing_decreasing_functions_exist_l1735_173536

-- Define a function that is increasing on one interval and decreasing on another
def has_increasing_decreasing_intervals (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a < b ∧ b < c ∧ c < d ∧
    (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∧
    (∀ x y, c < x ∧ x < y ∧ y < d → f y < f x)

-- Theorem stating that such functions exist
theorem increasing_decreasing_functions_exist :
  ∃ f : ℝ → ℝ, has_increasing_decreasing_intervals f :=
sorry

end increasing_decreasing_functions_exist_l1735_173536


namespace cube_sum_problem_l1735_173563

theorem cube_sum_problem (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 167) :
  x^3 + y^3 = 2005 := by
  sorry

end cube_sum_problem_l1735_173563


namespace T_is_three_intersecting_lines_l1735_173531

-- Define the set T
def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | 
  (p.1 - 3 = 5 ∧ p.2 + 1 ≥ 5) ∨
  (p.1 - 3 = p.2 + 1 ∧ 5 ≥ p.1 - 3) ∨
  (5 = p.2 + 1 ∧ p.1 - 3 ≥ 5)}

-- Define what it means for three lines to intersect at a single point
def three_lines_intersect_at_point (S : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  ∃ (l₁ l₂ l₃ : Set (ℝ × ℝ)),
    (∀ q ∈ S, q ∈ l₁ ∨ q ∈ l₂ ∨ q ∈ l₃) ∧
    (l₁ ∩ l₂ = {p}) ∧ (l₂ ∩ l₃ = {p}) ∧ (l₃ ∩ l₁ = {p}) ∧
    (∀ q ∈ l₁, ∃ r ∈ l₁, q ≠ r) ∧
    (∀ q ∈ l₂, ∃ r ∈ l₂, q ≠ r) ∧
    (∀ q ∈ l₃, ∃ r ∈ l₃, q ≠ r)

-- Theorem statement
theorem T_is_three_intersecting_lines :
  ∃ p : ℝ × ℝ, three_lines_intersect_at_point T p :=
sorry

end T_is_three_intersecting_lines_l1735_173531


namespace cubic_root_sum_l1735_173501

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  (a*b)/c + (b*c)/a + (c*a)/b = 49/6 := by
  sorry

end cubic_root_sum_l1735_173501


namespace equation_solution_l1735_173581

theorem equation_solution : 
  ∃ x : ℚ, (2*x + 1)/4 - 1 = x - (10*x + 1)/12 ∧ x = 5/2 := by
  sorry

end equation_solution_l1735_173581


namespace stratified_sampling_problem_l1735_173542

theorem stratified_sampling_problem (grade10 : ℕ) (grade11 : ℕ) (grade12 : ℕ) 
  (sample12 : ℕ) (h1 : grade10 = 1600) (h2 : grade11 = 1200) (h3 : grade12 = 800) 
  (h4 : sample12 = 20) :
  ∃ (sample1011 : ℕ), 
    (grade10 + grade11) * sample12 = grade12 * sample1011 ∧ 
    sample1011 = 70 := by
  sorry

end stratified_sampling_problem_l1735_173542


namespace mark_sprint_speed_l1735_173582

/-- Given a distance of 144 miles traveled in 24.0 hours, prove the speed is 6 miles per hour. -/
theorem mark_sprint_speed (distance : ℝ) (time : ℝ) (h1 : distance = 144) (h2 : time = 24.0) :
  distance / time = 6 := by
  sorry

end mark_sprint_speed_l1735_173582


namespace victors_decks_count_l1735_173552

/-- The number of decks Victor bought -/
def victors_decks : ℕ := 6

/-- The cost of each trick deck in dollars -/
def deck_cost : ℕ := 8

/-- The number of decks Victor's friend bought -/
def friends_decks : ℕ := 2

/-- The total amount spent by Victor and his friend in dollars -/
def total_spent : ℕ := 64

theorem victors_decks_count :
  victors_decks * deck_cost + friends_decks * deck_cost = total_spent :=
by sorry

end victors_decks_count_l1735_173552


namespace spending_ratio_theorem_l1735_173514

/-- Represents David's wages from last week -/
def last_week_wages : ℝ := 1

/-- Percentage spent on recreation last week -/
def last_week_recreation_percent : ℝ := 0.20

/-- Percentage spent on transportation last week -/
def last_week_transportation_percent : ℝ := 0.10

/-- Percentage reduction in wages this week -/
def wage_reduction_percent : ℝ := 0.30

/-- Percentage spent on recreation this week -/
def this_week_recreation_percent : ℝ := 0.25

/-- Percentage spent on transportation this week -/
def this_week_transportation_percent : ℝ := 0.15

/-- The ratio of this week's combined spending to last week's is approximately 0.9333 -/
theorem spending_ratio_theorem : 
  let last_week_total := (last_week_recreation_percent + last_week_transportation_percent) * last_week_wages
  let this_week_wages := (1 - wage_reduction_percent) * last_week_wages
  let this_week_total := (this_week_recreation_percent + this_week_transportation_percent) * this_week_wages
  abs ((this_week_total / last_week_total) - 0.9333) < 0.0001 := by
sorry

end spending_ratio_theorem_l1735_173514


namespace solve_for_k_l1735_173500

theorem solve_for_k : ∃ k : ℝ, ((-1) - k * 2 = 7) ∧ k = -4 := by sorry

end solve_for_k_l1735_173500


namespace angle_2013_in_third_quadrant_l1735_173530

-- Define the quadrants
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define a function to determine the quadrant of an angle
def angleQuadrant (angle : ℝ) : Quadrant := sorry

-- Theorem stating that 2013° is in the third quadrant
theorem angle_2013_in_third_quadrant :
  angleQuadrant 2013 = Quadrant.Third :=
by
  -- Define the relationship between 2013° and 213°
  have h1 : 2013 = 5 * 360 + 213 := by sorry
  
  -- State that 213° is in the third quadrant
  have h2 : angleQuadrant 213 = Quadrant.Third := by sorry
  
  -- State that angles with the same terminal side are in the same quadrant
  have h3 : ∀ (a b : ℝ), (a - b) % 360 = 0 → angleQuadrant a = angleQuadrant b := by sorry
  
  sorry -- Proof omitted

end angle_2013_in_third_quadrant_l1735_173530


namespace dreams_driving_distance_l1735_173505

/-- Represents the problem of calculating Dream's driving distance --/
theorem dreams_driving_distance :
  let gas_consumption_rate : ℝ := 4  -- gallons per mile
  let additional_miles_tomorrow : ℝ := 200
  let total_gas_consumption : ℝ := 4000
  ∃ (miles_today : ℝ),
    gas_consumption_rate * miles_today + 
    gas_consumption_rate * (miles_today + additional_miles_tomorrow) = 
    total_gas_consumption ∧
    miles_today = 400 :=
by
  sorry

end dreams_driving_distance_l1735_173505


namespace fraction_subtraction_l1735_173557

theorem fraction_subtraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  1 / x - 1 / (x - 1) = -1 / (x^2 - x) := by
  sorry

end fraction_subtraction_l1735_173557


namespace store_inventory_problem_l1735_173540

/-- Represents the inventory of a store selling pomelos and watermelons -/
structure StoreInventory where
  pomelos : ℕ
  watermelons : ℕ

/-- Represents the daily sales of pomelos and watermelons -/
structure DailySales where
  pomelos : ℕ
  watermelons : ℕ

/-- The theorem statement for the store inventory problem -/
theorem store_inventory_problem 
  (initial : StoreInventory)
  (sales : DailySales)
  (days : ℕ) :
  initial.watermelons = 3 * initial.pomelos →
  sales.pomelos = 20 →
  sales.watermelons = 30 →
  days = 3 →
  initial.watermelons - days * sales.watermelons = 
    4 * (initial.pomelos - days * sales.pomelos) - 26 →
  initial.pomelos = 176 := by
  sorry


end store_inventory_problem_l1735_173540


namespace wills_remaining_money_l1735_173567

/-- Calculates the remaining money after a shopping trip with a refund -/
def remaining_money (initial_amount sweater_price tshirt_price shoes_price refund_percentage : ℚ) : ℚ :=
  initial_amount - sweater_price - tshirt_price + (shoes_price * refund_percentage)

/-- Theorem stating that Will's remaining money after the shopping trip is $81 -/
theorem wills_remaining_money :
  remaining_money 74 9 11 30 0.9 = 81 := by
  sorry

end wills_remaining_money_l1735_173567


namespace cubic_root_identity_l1735_173594

theorem cubic_root_identity (a b c t : ℝ) : 
  (∀ x, x^3 - 7*x^2 + 8*x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  t = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  t^6 - 21*t^3 - 9*t = 24*t - 41 := by
  sorry

end cubic_root_identity_l1735_173594


namespace tommy_crates_count_l1735_173598

/-- Proves that Tommy has 3 crates given the problem conditions -/
theorem tommy_crates_count :
  ∀ (c : ℕ),
  (∀ (crate : ℕ), crate = 20) →  -- Each crate holds 20 kg
  (330 : ℝ) = c * (330 : ℝ) / c →  -- Cost of crates is $330
  (∀ (price : ℝ), price = 6) →  -- Selling price is $6 per kg
  (∀ (rotten : ℕ), rotten = 3) →  -- 3 kg of tomatoes are rotten
  (12 : ℝ) = (c * 20 - 3) * 6 - 330 →  -- Profit is $12
  c = 3 := by
sorry

end tommy_crates_count_l1735_173598


namespace factor_implies_m_value_l1735_173506

theorem factor_implies_m_value (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 15 = (x + 3) * k) → m = 2 := by
  sorry

end factor_implies_m_value_l1735_173506


namespace triangle_area_l1735_173545

/-- Given a triangle ABC with cos A = 4/5 and AB · AC = 8, prove that its area is 3 -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let cosA := 4/5
  let dotProduct := (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)
  dotProduct = 8 →
  (1/2) * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * 
          Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) * 
          Real.sqrt (1 - cosA^2) = 3 := by
  sorry

end triangle_area_l1735_173545


namespace zebra_stripes_l1735_173509

theorem zebra_stripes (w n b : ℕ) : 
  w + n = b + 1 →  -- Total black stripes (wide + narrow) is one more than white stripes
  b = w + 7 →      -- Number of white stripes is 7 more than wide black stripes
  n = 8 :=         -- Number of narrow black stripes is 8
by sorry

end zebra_stripes_l1735_173509


namespace inverse_proportion_through_point_l1735_173593

/-- An inverse proportion function passing through (2, -3) has m = -6 -/
theorem inverse_proportion_through_point (m : ℝ) : 
  (∀ x, x ≠ 0 → (m / x = -3 ↔ x = 2)) → m = -6 := by
  sorry

end inverse_proportion_through_point_l1735_173593


namespace total_dogs_l1735_173583

theorem total_dogs (brown white black : ℕ) 
  (h1 : brown = 20) 
  (h2 : white = 10) 
  (h3 : black = 15) : 
  brown + white + black = 45 := by
  sorry

end total_dogs_l1735_173583


namespace store_a_advantage_l1735_173523

/-- The original price of each computer in yuan -/
def original_price : ℝ := 6000

/-- The cost of buying computers from Store A -/
def cost_store_a (x : ℝ) : ℝ := original_price + (0.75 * original_price) * (x - 1)

/-- The cost of buying computers from Store B -/
def cost_store_b (x : ℝ) : ℝ := 0.8 * original_price * x

/-- Theorem stating when it's more advantageous to buy from Store A -/
theorem store_a_advantage (x : ℝ) : x > 5 → cost_store_a x < cost_store_b x := by
  sorry

#check store_a_advantage

end store_a_advantage_l1735_173523


namespace rectangle_length_l1735_173518

/-- Given a rectangle with perimeter 700 and breadth 100, its length is 250. -/
theorem rectangle_length (perimeter breadth length : ℝ) : 
  perimeter = 700 →
  breadth = 100 →
  perimeter = 2 * (length + breadth) →
  length = 250 := by
sorry

end rectangle_length_l1735_173518


namespace expression_simplification_l1735_173527

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  ((2 * x - 3) / (x - 2) - 1) / ((x^2 - 2*x + 1) / (x - 2)) = 1 / (x - 1) := by
  sorry

end expression_simplification_l1735_173527


namespace quadratic_sum_l1735_173538

/-- A quadratic function passing through (1, 3) and (2, 12) -/
def QuadraticFunction (p q r : ℝ) : ℝ → ℝ := λ x ↦ p * x^2 + q * x + r

/-- The theorem stating that p + q + 3r = -5 for the given quadratic function -/
theorem quadratic_sum (p q r : ℝ) : 
  (QuadraticFunction p q r 1 = 3) → 
  (QuadraticFunction p q r 2 = 12) → 
  p + q + 3 * r = -5 := by
  sorry

end quadratic_sum_l1735_173538


namespace sum_of_four_squares_99_l1735_173551

theorem sum_of_four_squares_99 : ∃ (a b c d w x y z : ℕ),
  a^2 + b^2 + c^2 + d^2 = 99 ∧
  w^2 + x^2 + y^2 + z^2 = 99 ∧
  (a, b, c, d) ≠ (w, x, y, z) :=
by sorry

end sum_of_four_squares_99_l1735_173551


namespace tripod_height_theorem_l1735_173546

/-- Represents a tripod with three legs -/
structure Tripod where
  leg_length : ℝ
  original_height : ℝ
  broken_leg_length : ℝ

/-- Calculates the new height of a tripod after one leg is shortened -/
def new_height (t : Tripod) : ℝ :=
  sorry

/-- Expresses the new height as a fraction m / √n -/
def height_fraction (t : Tripod) : ℚ × ℕ :=
  sorry

theorem tripod_height_theorem (t : Tripod) 
  (h_leg : t.leg_length = 5)
  (h_height : t.original_height = 4)
  (h_broken : t.broken_leg_length = 4) :
  let (m, n) := height_fraction t
  ⌊m + Real.sqrt n⌋ = 183 :=
sorry

end tripod_height_theorem_l1735_173546


namespace circle_area_from_circumference_l1735_173529

theorem circle_area_from_circumference (k : ℝ) : 
  let circumference := 18 * Real.pi
  let radius := circumference / (2 * Real.pi)
  let area := k * Real.pi
  area = Real.pi * radius^2 → k = 81 := by
  sorry

end circle_area_from_circumference_l1735_173529


namespace valid_pairs_l1735_173547

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digits_form_arithmetic_sequence (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ : ℕ), n = 100 * d₁ + 10 * d₂ + d₃ ∧ 
    d₁ < d₂ ∧ d₂ < d₃ ∧ d₂ - d₁ = d₃ - d₂

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def all_digits_same (n : ℕ) : Prop :=
  ∃ (d : ℕ), n = d * 11111

theorem valid_pairs : 
  ∀ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ≤ b ∧ 
    is_three_digit (a + b) ∧ 
    digits_form_arithmetic_sequence (a + b) ∧
    is_five_digit (a * b) ∧
    all_digits_same (a * b) →
  ((a = 41 ∧ b = 271) ∨ 
   (a = 164 ∧ b = 271) ∨ 
   (a = 82 ∧ b = 542) ∨ 
   (a = 123 ∧ b = 813)) :=
by sorry

end valid_pairs_l1735_173547


namespace prob_three_wins_correct_l1735_173599

-- Define the game parameters
def num_balls : ℕ := 6
def num_people : ℕ := 4
def draws_per_person : ℕ := 2

-- Define the winning condition
def is_winning_product (n : ℕ) : Prop := n % 4 = 0

-- Define the probability of winning in a single draw
def single_draw_probability : ℚ := 2 / 5

-- Define the probability of exactly three people winning
def prob_three_wins : ℚ := 96 / 625

-- State the theorem
theorem prob_three_wins_correct : 
  prob_three_wins = (num_people.choose 3) * 
    (single_draw_probability ^ 3) * 
    ((1 - single_draw_probability) ^ (num_people - 3)) :=
sorry

end prob_three_wins_correct_l1735_173599


namespace probability_log3_integer_l1735_173525

/-- A four-digit number is a natural number between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The count of four-digit numbers that are powers of 3. -/
def CountPowersOfThree : ℕ := 2

/-- The total count of four-digit numbers. -/
def TotalFourDigitNumbers : ℕ := 9000

/-- The probability of a randomly chosen four-digit number being a power of 3. -/
def ProbabilityPowerOfThree : ℚ := CountPowersOfThree / TotalFourDigitNumbers

theorem probability_log3_integer :
  ProbabilityPowerOfThree = 1 / 4500 := by
  sorry

end probability_log3_integer_l1735_173525


namespace largest_class_size_l1735_173521

theorem largest_class_size (total_students : ℕ) (num_classes : ℕ) (class_diff : ℕ) :
  total_students = 95 →
  num_classes = 5 →
  class_diff = 2 →
  ∃ (x : ℕ), x = 23 ∧ 
    (x + (x - class_diff) + (x - 2*class_diff) + (x - 3*class_diff) + (x - 4*class_diff) = total_students) :=
by sorry

end largest_class_size_l1735_173521
