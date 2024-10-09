import Mathlib

namespace real_solution_of_equation_l1269_126904

theorem real_solution_of_equation :
  ∀ x : ℝ, (x ≠ 5) → (x ≠ 3) →
  ((x - 2) * (x - 5) * (x - 3) * (x - 2) * (x - 4) * (x - 5) * (x - 3)) 
  / ((x - 5) * (x - 3) * (x - 5)) = 1 ↔ x = 1 :=
by sorry

end real_solution_of_equation_l1269_126904


namespace angle_terminal_side_equiv_l1269_126949

def angle_equiv_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem angle_terminal_side_equiv : angle_equiv_terminal_side (-Real.pi / 3) (5 * Real.pi / 3) :=
by
  sorry

end angle_terminal_side_equiv_l1269_126949


namespace teairras_pants_count_l1269_126942

-- Definitions according to the given conditions
def total_shirts := 5
def plaid_shirts := 3
def purple_pants := 5
def neither_plaid_nor_purple := 21

-- The theorem we need to prove
theorem teairras_pants_count :
  ∃ (pants : ℕ), pants = (neither_plaid_nor_purple - (total_shirts - plaid_shirts)) + purple_pants ∧ pants = 24 :=
by
  sorry

end teairras_pants_count_l1269_126942


namespace determine_x_l1269_126901

variable (a b c d x : ℝ)
variable (h1 : (a^2 + x)/(b^2 + x) = c/d)
variable (h2 : a ≠ b)
variable (h3 : b ≠ 0)
variable (h4 : d ≠ c) -- added condition from solution step

theorem determine_x : x = (a^2 * d - b^2 * c) / (c - d) := by
  sorry

end determine_x_l1269_126901


namespace problem_statement_l1269_126986

def scientific_notation (n: ℝ) (mantissa: ℝ) (exponent: ℤ) : Prop :=
  n = mantissa * 10 ^ exponent

theorem problem_statement : scientific_notation 320000 3.2 5 :=
by {
  sorry
}

end problem_statement_l1269_126986


namespace smallest_white_marbles_l1269_126907

/-
Let n be the total number of Peter's marbles.
Half of the marbles are orange.
One fifth of the marbles are purple.
Peter has 8 silver marbles.
-/
def total_marbles (n : ℕ) : ℕ :=
  n

def orange_marbles (n : ℕ) : ℕ :=
  n / 2

def purple_marbles (n : ℕ) : ℕ :=
  n / 5

def silver_marbles : ℕ :=
  8

def white_marbles (n : ℕ) : ℕ :=
  n - (orange_marbles n + purple_marbles n + silver_marbles)

-- Prove that the smallest number of white marbles Peter could have is 1.
theorem smallest_white_marbles : ∃ n : ℕ, n % 10 = 0 ∧ white_marbles n = 1 :=
sorry

end smallest_white_marbles_l1269_126907


namespace determinant_expression_l1269_126964

theorem determinant_expression (a b c p q : ℝ) 
  (h_root : ∀ x : ℝ, (x = a ∨ x = b ∨ x = c) → (Polynomial.eval x (Polynomial.X ^ 3 - 3 * Polynomial.C p * Polynomial.X + 2 * Polynomial.C q) = 0)) :
  Matrix.det ![![2 + a, 1, 1], ![1, 2 + b, 1], ![1, 1, 2 + c]] = -3 * p - 2 * q + 4 :=
by {
  sorry
}

end determinant_expression_l1269_126964


namespace students_passing_in_sixth_year_l1269_126947

def numStudentsPassed (year : ℕ) : ℕ :=
 if year = 1 then 200 else 
 if year = 2 then 300 else 
 if year = 3 then 390 else 
 if year = 4 then 565 else 
 if year = 5 then 643 else 
 if year = 6 then 780 else 0

theorem students_passing_in_sixth_year : numStudentsPassed 6 = 780 := by
  sorry

end students_passing_in_sixth_year_l1269_126947


namespace oil_remaining_in_tank_l1269_126939

/- Definitions for the problem conditions -/
def tankCapacity : Nat := 32
def totalOilPurchased : Nat := 728

/- Theorem statement -/
theorem oil_remaining_in_tank : totalOilPurchased % tankCapacity = 24 := by
  sorry

end oil_remaining_in_tank_l1269_126939


namespace fraction_received_A_correct_l1269_126961

def fraction_of_students_received_A := 0.7
def fraction_of_students_received_B := 0.2
def fraction_of_students_received_A_or_B := 0.9

theorem fraction_received_A_correct :
  fraction_of_students_received_A_or_B - fraction_of_students_received_B = fraction_of_students_received_A :=
by
  sorry

end fraction_received_A_correct_l1269_126961


namespace twentieth_triangular_number_l1269_126927

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem twentieth_triangular_number : triangular_number 20 = 210 :=
by
  sorry

end twentieth_triangular_number_l1269_126927


namespace smaller_angle_at_7_30_is_45_degrees_l1269_126990

noncomputable def calculateAngle (hour minute : Nat) : Real :=
  let minuteAngle := (minute * 6 : Real)
  let hourAngle := (hour % 12 * 30 : Real) + (minute / 60 * 30 : Real)
  let diff := abs (hourAngle - minuteAngle)
  if diff > 180 then 360 - diff else diff

theorem smaller_angle_at_7_30_is_45_degrees :
  calculateAngle 7 30 = 45 := 
sorry

end smaller_angle_at_7_30_is_45_degrees_l1269_126990


namespace trip_correct_graph_l1269_126932

-- Define a structure representing the trip
structure Trip :=
  (initial_city_traffic_duration : ℕ)
  (highway_duration_to_mall : ℕ)
  (shopping_duration : ℕ)
  (highway_duration_from_mall : ℕ)
  (return_city_traffic_duration : ℕ)

-- Define the conditions about the trip
def conditions (t : Trip) : Prop :=
  t.shopping_duration = 1 ∧ -- Shopping for one hour
  t.initial_city_traffic_duration < t.highway_duration_to_mall ∧ -- Travel more rapidly on the highway
  t.return_city_traffic_duration < t.highway_duration_from_mall -- Return more rapidly on the highway

-- Define the graph representation of the trip
inductive Graph
| A | B | C | D | E

-- Define the property that graph B correctly represents the trip
def correct_graph (t : Trip) (g : Graph) : Prop :=
  g = Graph.B

-- The theorem stating that given the conditions, the correct graph is B
theorem trip_correct_graph (t : Trip) (h : conditions t) : correct_graph t Graph.B :=
by
  sorry

end trip_correct_graph_l1269_126932


namespace cost_of_fencing_per_meter_l1269_126941

theorem cost_of_fencing_per_meter (length breadth : ℕ) (total_cost : ℚ) 
    (h_length : length = 61) 
    (h_rule : length = breadth + 22) 
    (h_total_cost : total_cost = 5300) :
    total_cost / (2 * length + 2 * breadth) = 26.5 := 
by 
  sorry

end cost_of_fencing_per_meter_l1269_126941


namespace temperature_difference_l1269_126944

-- Definitions based on the conditions
def refrigeration_compartment_temperature : ℤ := 5
def freezer_compartment_temperature : ℤ := -2

-- Mathematically equivalent proof problem statement
theorem temperature_difference : refrigeration_compartment_temperature - freezer_compartment_temperature = 7 := by
  sorry

end temperature_difference_l1269_126944


namespace collective_land_area_l1269_126912

theorem collective_land_area 
  (C W : ℕ) 
  (h1 : 42 * C + 35 * W = 165200)
  (h2 : W = 3400)
  : C + W = 4500 :=
sorry

end collective_land_area_l1269_126912


namespace pie_shop_revenue_l1269_126957

def costPerSlice : Int := 5
def slicesPerPie : Int := 4
def piesSold : Int := 9

theorem pie_shop_revenue : (costPerSlice * slicesPerPie * piesSold) = 180 := 
by
  sorry

end pie_shop_revenue_l1269_126957


namespace probability_all_co_captains_l1269_126914

-- Define the number of students in each team
def students_team1 : ℕ := 4
def students_team2 : ℕ := 6
def students_team3 : ℕ := 7
def students_team4 : ℕ := 9

-- Define the probability of selecting each team
def prob_selecting_team : ℚ := 1 / 4

-- Define the probability of selecting three co-captains from each team
def prob_team1 : ℚ := 1 / Nat.choose students_team1 3
def prob_team2 : ℚ := 1 / Nat.choose students_team2 3
def prob_team3 : ℚ := 1 / Nat.choose students_team3 3
def prob_team4 : ℚ := 1 / Nat.choose students_team4 3

-- Define the total probability
def total_prob : ℚ :=
  prob_selecting_team * (prob_team1 + prob_team2 + prob_team3 + prob_team4)

theorem probability_all_co_captains :
  total_prob = 59 / 1680 := by
  sorry

end probability_all_co_captains_l1269_126914


namespace tom_is_15_years_younger_l1269_126945

/-- 
Alice is now 30 years old.
Ten years ago, Alice was 4 times as old as Tom was then.
Prove that Tom is 15 years younger than Alice.
-/
theorem tom_is_15_years_younger (A T : ℕ) (h1 : A = 30) (h2 : A - 10 = 4 * (T - 10)) : A - T = 15 :=
by
  sorry

end tom_is_15_years_younger_l1269_126945


namespace geom_sequence_sum_l1269_126930

theorem geom_sequence_sum (n : ℕ) (a : ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 4 ^ n + a) : 
  a = -1 := 
by
  sorry

end geom_sequence_sum_l1269_126930


namespace mask_usage_duration_l1269_126918

-- Define given conditions
def TotalMasks : ℕ := 75
def FamilyMembers : ℕ := 7
def MaskChangeInterval : ℕ := 2

-- Define the goal statement, which is to prove that the family will take 21 days to use all masks
theorem mask_usage_duration 
  (M : ℕ := 75)  -- total masks
  (N : ℕ := 7)   -- family members
  (d : ℕ := 2)   -- mask change interval
  : (M / N) * d + 1 = 21 :=
sorry

end mask_usage_duration_l1269_126918


namespace expression_value_l1269_126905

theorem expression_value (x : ℝ) (h : x = Real.sqrt (19 - 8 * Real.sqrt 3)) :
  (x ^ 4 - 6 * x ^ 3 - 2 * x ^ 2 + 18 * x + 23) / (x ^ 2 - 8 * x + 15) = 5 :=
by
  sorry

end expression_value_l1269_126905


namespace train_length_l1269_126977

theorem train_length :
  let speed_kmph := 63
  let time_seconds := 16
  let speed_mps := (speed_kmph * 1000) / 3600
  let length_meters := speed_mps * time_seconds
  length_meters = 280 := 
by
  sorry

end train_length_l1269_126977


namespace number_of_lines_l1269_126983

-- Define the point (0, 1)
def point : ℝ × ℝ := (0, 1)

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the condition that a line intersects a parabola at only one point
def line_intersects_parabola_at_one_point (m b x y : ℝ) : Prop :=
  y - (m * x + b) = 0 ∧ parabola x y

-- The proof problem: Prove there are 3 such lines
theorem number_of_lines : ∃ (n : ℕ), n = 3 ∧ (
  ∃ (m b : ℝ), line_intersects_parabola_at_one_point m b 0 1) :=
sorry

end number_of_lines_l1269_126983


namespace kilometers_driven_equal_l1269_126924

theorem kilometers_driven_equal (x : ℝ) :
  (20 + 0.25 * x = 24 + 0.16 * x) → x = 44 := by
  sorry

end kilometers_driven_equal_l1269_126924


namespace number_of_n_l1269_126985

theorem number_of_n (h1: n > 0) (h2: n ≤ 2000) (h3: ∃ m, 10 * n = m^2) : n = 14 :=
by sorry

end number_of_n_l1269_126985


namespace solve_for_y_l1269_126936

noncomputable def roots := [(-126 + Real.sqrt 13540) / 8, (-126 - Real.sqrt 13540) / 8]

theorem solve_for_y (y : ℝ) :
  (8*y^2 + 176*y + 2) / (3*y + 74) = 4*y + 2 →
  y = roots[0] ∨ y = roots[1] :=
by
  intros
  sorry

end solve_for_y_l1269_126936


namespace sector_area_is_80pi_l1269_126953

noncomputable def sectorArea (θ r : ℝ) : ℝ := 
  1 / 2 * θ * r^2

theorem sector_area_is_80pi :
  sectorArea (2 * Real.pi / 5) 20 = 80 * Real.pi :=
by
  sorry

end sector_area_is_80pi_l1269_126953


namespace total_parking_spaces_l1269_126974

-- Definitions of conditions
def caravan_space : ℕ := 2
def number_of_caravans : ℕ := 3
def spaces_left : ℕ := 24

-- Proof statement
theorem total_parking_spaces :
  (number_of_caravans * caravan_space + spaces_left) = 30 :=
by
  sorry

end total_parking_spaces_l1269_126974


namespace a3_plus_a4_l1269_126943

def sum_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 3^(n + 1)

theorem a3_plus_a4 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : sum_of_sequence S a) :
  a 3 + a 4 = 216 :=
sorry

end a3_plus_a4_l1269_126943


namespace convex_quad_no_triangle_l1269_126919

/-- Given four angles of a convex quadrilateral, it is not always possible to choose any 
three of these angles so that they represent the lengths of the sides of some triangle. -/
theorem convex_quad_no_triangle (α β γ δ : ℝ) 
  (h_sum : α + β + γ + δ = 360) :
  ¬(∀ a b c : ℝ, a + b + c = 360 → (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by
  sorry

end convex_quad_no_triangle_l1269_126919


namespace an_values_and_formula_is_geometric_sequence_l1269_126933

-- Definitions based on the conditions
def Sn (n : ℕ) : ℝ := sorry  -- S_n to be defined in the context or problem details
def a (n : ℕ) : ℝ := 2 - Sn n

-- Prove the specific values and general formula given the condition a_n = 2 - S_n
theorem an_values_and_formula (Sn : ℕ → ℝ) :
  a 1 = 1 ∧ a 2 = 1 / 2 ∧ a 3 = 1 / 4 ∧ a 4 = 1 / 8 ∧ (∀ n, a n = (1 / 2)^(n-1)) :=
sorry

-- Prove the sequence is geometric
theorem is_geometric_sequence (Sn : ℕ → ℝ) :
  (∀ n, a n = (1 / 2)^(n-1)) → ∀ n, a (n + 1) / a n = 1 / 2 :=
sorry

end an_values_and_formula_is_geometric_sequence_l1269_126933


namespace house_A_cost_l1269_126963

theorem house_A_cost (base_salary earnings commission_rate total_houses cost_A cost_B cost_C : ℝ)
  (H_base_salary : base_salary = 3000)
  (H_earnings : earnings = 8000)
  (H_commission_rate : commission_rate = 0.02)
  (H_cost_B : cost_B = 3 * cost_A)
  (H_cost_C : cost_C = 2 * cost_A - 110000)
  (H_total_commission : earnings - base_salary = 5000)
  (H_total_cost : 5000 / commission_rate = 250000)
  (H_total_houses : base_salary + commission_rate * (cost_A + cost_B + cost_C) = earnings) :
  cost_A = 60000 := sorry

end house_A_cost_l1269_126963


namespace ambiguous_dates_in_year_l1269_126999

def is_ambiguous_date (m d : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 12 ∧ 1 ≤ d ∧ d ≤ 12 ∧ m ≠ d

theorem ambiguous_dates_in_year :
  ∃ n : ℕ, n = 132 ∧ (∀ m d : ℕ, is_ambiguous_date m d → n = 132) :=
sorry

end ambiguous_dates_in_year_l1269_126999


namespace pair_exists_l1269_126911

theorem pair_exists (x : Fin 670 → ℝ) (h_distinct : Function.Injective x) (h_bounds : ∀ i, 0 < x i ∧ x i < 1) :
  ∃ (i j : Fin 670), 0 < x i * x j * (x j - x i) ∧ x i * x j * (x j - x i) < 1 / 2007 := 
by
  sorry

end pair_exists_l1269_126911


namespace compression_strength_value_l1269_126989

def compression_strength (T H : ℕ) : ℚ :=
  (15 * T^5) / (H^3)

theorem compression_strength_value : 
  compression_strength 3 6 = 55 / 13 := by
  sorry

end compression_strength_value_l1269_126989


namespace find_x_l1269_126998

noncomputable def satisfy_equation (x : ℝ) : Prop :=
  8 / (Real.sqrt (x - 10) - 10) +
  2 / (Real.sqrt (x - 10) - 5) +
  10 / (Real.sqrt (x - 10) + 5) +
  16 / (Real.sqrt (x - 10) + 10) = 0

theorem find_x : ∃ x : ℝ, satisfy_equation x ∧ x = 60 := sorry

end find_x_l1269_126998


namespace comparison_of_a_b_c_l1269_126991

noncomputable def a : ℝ := 2018 ^ (1 / 2018)
noncomputable def b : ℝ := Real.logb 2017 (Real.sqrt 2018)
noncomputable def c : ℝ := Real.logb 2018 (Real.sqrt 2017)

theorem comparison_of_a_b_c :
  a > b ∧ b > c :=
by
  -- Definitions
  have def_a : a = 2018 ^ (1 / 2018) := rfl
  have def_b : b = Real.logb 2017 (Real.sqrt 2018) := rfl
  have def_c : c = Real.logb 2018 (Real.sqrt 2017) := rfl

  -- Sorry is added to skip the proof
  sorry

end comparison_of_a_b_c_l1269_126991


namespace solve_firm_problem_l1269_126909

def firm_problem : Prop :=
  ∃ (P A : ℕ), 
    (P / A = 2 / 63) ∧ 
    (P / (A + 50) = 1 / 34) ∧ 
    (P = 20)

theorem solve_firm_problem : firm_problem :=
  sorry

end solve_firm_problem_l1269_126909


namespace find_x_l1269_126948

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 210) (h2 : ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) (h3 : 0 < x) : x = 14 :=
sorry

end find_x_l1269_126948


namespace A_visits_all_seats_iff_even_l1269_126979

def move_distance_unique (n : ℕ) : Prop := 
  ∀ k l : ℕ, (1 ≤ k ∧ k < n) → (1 ≤ l ∧ l < n) → k ≠ l → (k ≠ l % n)

def visits_all_seats (n : ℕ) : Prop := 
  ∃ A : ℕ → ℕ, 
  (∀ (k : ℕ), 0 ≤ k ∧ k < n → (0 ≤ A k ∧ A k < n)) ∧ 
  (∀ (k : ℕ), 0 ≤ k ∧ k < n → ∃ (m : ℕ), m ≠ n ∧ A k ≠ (A m % n))

theorem A_visits_all_seats_iff_even (n : ℕ) :
  (move_distance_unique n ∧ visits_all_seats n) ↔ (n % 2 = 0) := 
sorry

end A_visits_all_seats_iff_even_l1269_126979


namespace HA_appears_at_least_once_l1269_126965

-- Define the set of letters to be arranged
def letters : List Char := ['A', 'A', 'A', 'H', 'H']

-- Define a function to count the number of ways to arrange letters such that "HA" appears at least once
def countHA(A : List Char) : Nat := sorry

-- The proof problem to establish that there are 9 such arrangements
theorem HA_appears_at_least_once : countHA letters = 9 :=
sorry

end HA_appears_at_least_once_l1269_126965


namespace problem_solution_l1269_126946

theorem problem_solution 
  (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) :
  4 * x^4 + 17 * x^2 * y + 4 * y^2 < (m / 4) * (x^4 + 2 * x^2 * y + y^2) ↔ 25 < m :=
sorry

end problem_solution_l1269_126946


namespace grid_problem_l1269_126995

theorem grid_problem 
  (A B : ℕ) 
  (grid : (Fin 3) → (Fin 3) → ℕ)
  (h1 : ∀ i, grid 0 i ≠ grid 1 i)
  (h2 : ∀ i, grid 0 i ≠ grid 2 i)
  (h3 : ∀ i, grid 1 i ≠ grid 2 i)
  (h4 : ∀ i, (∃! x, grid x i = 1))
  (h5 : ∀ i, (∃! x, grid x i = 2))
  (h6 : ∀ i, (∃! x, grid x i = 3))
  (h7 : grid 1 2 = A)
  (h8 : grid 2 2 = B) : 
  A + B + 4 = 8 :=
by sorry

end grid_problem_l1269_126995


namespace sarah_average_speed_l1269_126994

theorem sarah_average_speed :
  ∀ (total_distance race_time : ℕ) 
    (sadie_speed sadie_time ariana_speed ariana_time : ℕ)
    (distance_sarah speed_sarah time_sarah : ℚ),
  sadie_speed = 3 → 
  sadie_time = 2 → 
  ariana_speed = 6 → 
  ariana_time = 1 / 2 → 
  race_time = 9 / 2 → 
  total_distance = 17 →
  distance_sarah = total_distance - (sadie_speed * sadie_time + ariana_speed * ariana_time) →
  time_sarah = race_time - (sadie_time + ariana_time) →
  speed_sarah = distance_sarah / time_sarah →
  speed_sarah = 4 :=
by
  intros total_distance race_time sadie_speed sadie_time ariana_speed ariana_time distance_sarah speed_sarah time_sarah
  intros sadie_speed_eq sadie_time_eq ariana_speed_eq ariana_time_eq race_time_eq total_distance_eq distance_sarah_eq time_sarah_eq speed_sarah_eq
  sorry

end sarah_average_speed_l1269_126994


namespace rice_yield_l1269_126958

theorem rice_yield (X : ℝ) (h1 : 0 ≤ X ∧ X ≤ 40) :
    0.75 * 400 * X + 0.25 * 800 * X + 500 * (40 - X) = 20000 := by
  sorry

end rice_yield_l1269_126958


namespace fraction_meaningfulness_l1269_126900

def fraction_is_meaningful (x : ℝ) : Prop :=
  x ≠ 3 / 2

theorem fraction_meaningfulness (x : ℝ) : 
  (2 * x - 3) ≠ 0 ↔ fraction_is_meaningful x :=
by
  sorry

end fraction_meaningfulness_l1269_126900


namespace volume_of_cube_l1269_126937

theorem volume_of_cube (SA : ℝ) (h : SA = 486) : ∃ V : ℝ, V = 729 :=
by
  sorry

end volume_of_cube_l1269_126937


namespace age_ratio_l1269_126984

theorem age_ratio (A B : ℕ) 
  (h1 : A = 39) 
  (h2 : B = 16) 
  (h3 : (A - 5) + (B - 5) = 45) 
  (h4 : A + 5 = 44) : A / B = 39 / 16 := 
by 
  sorry

end age_ratio_l1269_126984


namespace project_completion_time_l1269_126987

theorem project_completion_time 
    (w₁ w₂ : ℕ) 
    (d₁ d₂ : ℕ) 
    (fraction₁ fraction₂ : ℝ)
    (h_work_fraction : fraction₁ = 1/2)
    (h_work_time : d₁ = 6)
    (h_first_workforce : w₁ = 90)
    (h_second_workforce : w₂ = 60)
    (h_fraction_done_by_first_team : w₁ * d₁ * (1 / 1080) = fraction₁)
    (h_fraction_done_by_second_team : w₂ * d₂ * (1 / 1080) = fraction₂)
    (h_total_fraction : fraction₂ = 1 - fraction₁) :
    d₂ = 9 :=
by 
  sorry

end project_completion_time_l1269_126987


namespace hypotenuse_length_l1269_126950

theorem hypotenuse_length (x y : ℝ) (V1 V2 : ℝ) 
  (h1 : V1 = 1350 * Real.pi) 
  (h2 : V2 = 2430 * Real.pi) 
  (h3 : (1/3) * Real.pi * y^2 * x = V1) 
  (h4 : (1/3) * Real.pi * x^2 * y = V2) 
  : Real.sqrt (x^2 + y^2) = Real.sqrt 954 :=
sorry

end hypotenuse_length_l1269_126950


namespace yi_catches_jia_on_DA_l1269_126938

def square_side_length : ℝ := 90
def jia_speed : ℝ := 65
def yi_speed : ℝ := 72
def jia_start : ℝ := 0
def yi_start : ℝ := 90

theorem yi_catches_jia_on_DA :
  let square_perimeter := 4 * square_side_length
  let initial_gap := 3 * square_side_length
  let relative_speed := yi_speed - jia_speed
  let time_to_catch := initial_gap / relative_speed
  let distance_travelled_by_yi := yi_speed * time_to_catch
  let number_of_laps := distance_travelled_by_yi / square_perimeter
  let additional_distance := distance_travelled_by_yi % square_perimeter
  additional_distance = 0 →
  square_side_length * (number_of_laps % 4) = 0 ∨ number_of_laps % 4 = 3 :=
by
  -- We only provide the statement, the proof is omitted.
  sorry

end yi_catches_jia_on_DA_l1269_126938


namespace problem1_problem2_l1269_126934

-- Definition for Problem 1
def cube_root_8 : ℝ := 2
def abs_neg5 : ℝ := 5
def pow_neg1_2023 : ℝ := -1

theorem problem1 : cube_root_8 + abs_neg5 + pow_neg1_2023 = 6 := by
  sorry

-- Definitions for Problem 2
structure Point where
  x : ℝ
  y : ℝ

def point1 : Point := { x := 0, y := 1 }
def point2 : Point := { x := 2, y := 5 }

def linear_function (k b x : ℝ) : ℝ := k * x + b

def is_linear_function (p1 p2 : Point) (k b : ℝ) : Prop :=
  p1.y = linear_function k b p1.x ∧ p2.y = linear_function k b p2.x

theorem problem2 : ∃ k b : ℝ, is_linear_function point1 point2 k b ∧ (linear_function k b x = 2 * x + 1) := by
  sorry

end problem1_problem2_l1269_126934


namespace diamond_eq_l1269_126935

noncomputable def diamond_op (a b : ℝ) (k : ℝ) : ℝ := sorry

theorem diamond_eq (x : ℝ) :
  let k := 2
  let a := 2023
  let b := 7
  let c := x
  (diamond_op a (diamond_op b c k) k = 150) ∧ 
  (∀ a b c, diamond_op a (diamond_op b c k) k = k * (diamond_op a b k) * c) ∧
  (∀ a, diamond_op a a k = k) →
  x = 150 / 2023 :=
sorry

end diamond_eq_l1269_126935


namespace hyperbola_aux_lines_l1269_126978

theorem hyperbola_aux_lines (a : ℝ) (h_a_positive : a > 0)
  (h_hyperbola_eqn : ∀ x y, (x^2 / a^2) - (y^2 / 16) = 1)
  (h_asymptotes : ∀ x y, y = 4/3 * x ∨ y = -4/3 * x) : 
  ∀ x, (x = 9/5 ∨ x = -9/5) := sorry

end hyperbola_aux_lines_l1269_126978


namespace tangent_line_at_1_l1269_126993

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

noncomputable def f' (x : ℝ) : ℝ := 1 / x - 3

theorem tangent_line_at_1 :
  let y := f 1
  let k := f' 1
  y = -3 ∧ k = -2 →
  ∀ (x y : ℝ), y = k * (x - 1) + f 1 ↔ 2 * x + y + 1 = 0 :=
by
  sorry

end tangent_line_at_1_l1269_126993


namespace prime_of_form_4k_plus_1_as_sum_of_two_squares_prime_of_form_8k_plus_3_as_sum_of_three_squares_l1269_126967

theorem prime_of_form_4k_plus_1_as_sum_of_two_squares (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hk : p = 4 * k + 1) :
  ∃ a b : ℤ, p = a^2 + b^2 :=
sorry

theorem prime_of_form_8k_plus_3_as_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hk : p = 8 * k + 3) :
  ∃ a b c : ℤ, p = a^2 + b^2 + c^2 :=
sorry

end prime_of_form_4k_plus_1_as_sum_of_two_squares_prime_of_form_8k_plus_3_as_sum_of_three_squares_l1269_126967


namespace Janice_time_left_l1269_126980

-- Define the conditions as variables and parameters
def homework_time := 30
def cleaning_time := homework_time / 2
def dog_walking_time := homework_time + 5
def trash_time := homework_time / 6
def total_time_before_movie := 2 * 60

-- Calculation of total time required for all tasks
def total_time_required_for_tasks : Nat :=
  homework_time + cleaning_time + dog_walking_time + trash_time

-- Time left before the movie starts after completing all tasks
def time_left_before_movie : Nat :=
  total_time_before_movie - total_time_required_for_tasks

-- The final statement to prove
theorem Janice_time_left : time_left_before_movie = 35 :=
  by
    -- This will execute automatically to verify the theorem
    sorry

end Janice_time_left_l1269_126980


namespace derivative_of_f_eq_f_deriv_l1269_126928

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.cos a) ^ x - (Real.sin a) ^ x

noncomputable def f_deriv (a x : ℝ) : ℝ :=
  (Real.cos a) ^ x * Real.log (Real.cos a) - (Real.sin a) ^ x * Real.log (Real.sin a)

theorem derivative_of_f_eq_f_deriv (a : ℝ) (h : 0 < a ∧ a < Real.pi / 2) :
  (deriv (f a)) = f_deriv a :=
by
  sorry

end derivative_of_f_eq_f_deriv_l1269_126928


namespace angle_bisector_ratio_l1269_126908

theorem angle_bisector_ratio (A B C Q : Type) (AC CB AQ QB : ℝ) (k : ℝ) 
  (hAC : AC = 4 * k) (hCB : CB = 5 * k) (angle_bisector_theorem : AQ / QB = AC / CB) :
  AQ / QB = 4 / 5 := 
by sorry

end angle_bisector_ratio_l1269_126908


namespace sum_of_1984_consecutive_integers_not_square_l1269_126969

theorem sum_of_1984_consecutive_integers_not_square :
  ∀ n : ℕ, ¬ ∃ k : ℕ, 992 * (2 * n + 1985) = k * k := by
  sorry

end sum_of_1984_consecutive_integers_not_square_l1269_126969


namespace students_in_fifth_and_sixth_classes_l1269_126997

theorem students_in_fifth_and_sixth_classes :
  let c1 := 20
  let c2 := 25
  let c3 := 25
  let c4 := c1 / 2
  let total_students := 136
  let total_first_four_classes := c1 + c2 + c3 + c4
  let c5_and_c6 := total_students - total_first_four_classes
  c5_and_c6 = 56 :=
by
  sorry

end students_in_fifth_and_sixth_classes_l1269_126997


namespace Brad_pumpkin_weight_l1269_126926

theorem Brad_pumpkin_weight (B : ℝ)
  (h1 : ∃ J : ℝ, J = B / 2)
  (h2 : ∃ Be : ℝ, Be = 4 * (B / 2))
  (h3 : ∃ Be J : ℝ, Be - J = 81) : B = 54 := by
  obtain ⟨J, hJ⟩ := h1
  obtain ⟨Be, hBe⟩ := h2
  obtain ⟨_, hBeJ⟩ := h3
  sorry

end Brad_pumpkin_weight_l1269_126926


namespace find_c_l1269_126981

theorem find_c
  (m b d c : ℝ)
  (h : m = b * d * c / (d + c)) :
  c = m * d / (b * d - m) :=
sorry

end find_c_l1269_126981


namespace ratio_of_numbers_l1269_126955

theorem ratio_of_numbers (a b : ℕ) (ha : a = 45) (hb : b = 60) (lcm_ab : Nat.lcm a b = 180) : (a : ℚ) / b = 3 / 4 :=
by
  sorry

end ratio_of_numbers_l1269_126955


namespace change_in_expression_l1269_126922

theorem change_in_expression (x b : ℝ) (hb : 0 < b) :
  let original_expr := x^2 - 5 * x + 2
  let new_x := x + b
  let new_expr := (new_x)^2 - 5 * (new_x) + 2
  new_expr - original_expr = 2 * b * x + b^2 - 5 * b :=
by
  sorry

end change_in_expression_l1269_126922


namespace max_sum_abc_divisible_by_13_l1269_126951

theorem max_sum_abc_divisible_by_13 :
  ∃ (A B C : ℕ), A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ 13 ∣ (2000 + 100 * A + 10 * B + C) ∧ (A + B + C = 26) :=
by
  sorry

end max_sum_abc_divisible_by_13_l1269_126951


namespace range_of_alpha_minus_beta_l1269_126915

theorem range_of_alpha_minus_beta (α β : ℝ) (h1 : -180 < α) (h2 : α < β) (h3 : β < 180) :
  -360 < α - β ∧ α - β < 0 :=
by
  sorry

end range_of_alpha_minus_beta_l1269_126915


namespace triangle_type_l1269_126910

-- Let's define what it means for a triangle to be acute, obtuse, and right in terms of angle
def is_acute_triangle (a b c : ℝ) : Prop := (a < 90) ∧ (b < 90) ∧ (c < 90)
def is_obtuse_triangle (a b c : ℝ) : Prop := (a > 90) ∨ (b > 90) ∨ (c > 90)
def is_right_triangle (a b c : ℝ) : Prop := (a = 90) ∨ (b = 90) ∨ (c = 90)

-- The problem statement
theorem triangle_type (A B C : ℝ) (h : A = 100) : is_obtuse_triangle A B C :=
by {
  -- Sorry is used to indicate a placeholder for the proof
  sorry
}

end triangle_type_l1269_126910


namespace algebra_expression_eq_l1269_126968

theorem algebra_expression_eq (x : ℝ) (h : x = Real.sqrt 2 + 1) : x^2 - 2 * x + 2 = 3 := by
  sorry

end algebra_expression_eq_l1269_126968


namespace sole_mart_meals_l1269_126906

theorem sole_mart_meals (c_c_meals : ℕ) (meals_given_away : ℕ) (meals_left : ℕ)
  (h1 : c_c_meals = 113) (h2 : meals_givenAway = 85) (h3 : meals_left = 78)  :
  ∃ m : ℕ, m + c_c_meals = meals_givenAway + meals_left ∧ m = 50 := 
by
  sorry

end sole_mart_meals_l1269_126906


namespace equal_sharing_l1269_126920

theorem equal_sharing (total_cards friends : ℕ) (h1 : total_cards = 455) (h2 : friends = 5) : total_cards / friends = 91 := by
  sorry

end equal_sharing_l1269_126920


namespace seq_ratio_l1269_126973

noncomputable def arith_seq (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem seq_ratio (a d : ℝ) (h₁ : d ≠ 0) (h₂ : (arith_seq a d 2)^2 = (arith_seq a d 0) * (arith_seq a d 8)) :
  (arith_seq a d 0 + arith_seq a d 2 + arith_seq a d 4) / (arith_seq a d 1 + arith_seq a d 3 + arith_seq a d 5) = 3 / 4 :=
by
  sorry

end seq_ratio_l1269_126973


namespace snakes_in_pond_l1269_126970

theorem snakes_in_pond (S : ℕ) (alligators : ℕ := 10) (total_eyes : ℕ := 56) (alligator_eyes : ℕ := 2) (snake_eyes : ℕ := 2) :
  (alligators * alligator_eyes) + (S * snake_eyes) = total_eyes → S = 18 :=
by
  intro h
  sorry

end snakes_in_pond_l1269_126970


namespace profit_percentage_l1269_126956

theorem profit_percentage (SP : ℝ) (h1 : SP > 0) (h2 : CP = 0.99 * SP) : (SP - CP) / CP * 100 = 1.01 :=
by
  sorry

end profit_percentage_l1269_126956


namespace min_games_to_predict_l1269_126929

theorem min_games_to_predict (W B : ℕ) (total_games : ℕ) (n : ℕ) : 
  W = 15 → B = 20 → total_games = W * B → n = 280 → 
  (∃ x, x ∈ {i | ∃ j, i < W ∧ j < B}) :=
by
  intros hW hB htotal hn
  sorry

end min_games_to_predict_l1269_126929


namespace slope_to_y_intercept_ratio_l1269_126954

theorem slope_to_y_intercept_ratio (m b : ℝ) (c : ℝ) (h1 : m = c * b) (h2 : 2 * m + b = 0) : c = -1 / 2 :=
by sorry

end slope_to_y_intercept_ratio_l1269_126954


namespace max_sum_pyramid_l1269_126982

theorem max_sum_pyramid (F_pentagonal : ℕ) (F_rectangular : ℕ) (E_pentagonal : ℕ) (E_rectangular : ℕ) (V_pentagonal : ℕ) (V_rectangular : ℕ)
  (original_faces : ℕ) (original_edges : ℕ) (original_vertices : ℕ)
  (H1 : original_faces = 7)
  (H2 : original_edges = 15)
  (H3 : original_vertices = 10)
  (H4 : F_pentagonal = 11)
  (H5 : E_pentagonal = 20)
  (H6 : V_pentagonal = 11)
  (H7 : F_rectangular = 10)
  (H8 : E_rectangular = 19)
  (H9 : V_rectangular = 11) :
  max (F_pentagonal + E_pentagonal + V_pentagonal) (F_rectangular + E_rectangular + V_rectangular) = 42 :=
by
  sorry

end max_sum_pyramid_l1269_126982


namespace fewest_coach_handshakes_l1269_126925

noncomputable def binom (n : ℕ) := n * (n - 1) / 2

theorem fewest_coach_handshakes : 
  ∃ (k1 k2 k3 : ℕ), binom 43 + k1 + k2 + k3 = 903 ∧ k1 + k2 + k3 = 0 := 
by
  use 0, 0, 0
  sorry

end fewest_coach_handshakes_l1269_126925


namespace probability_of_U_l1269_126992

def pinyin : List Char := ['S', 'H', 'U', 'X', 'U', 'E']
def total_letters : Nat := 6
def u_count : Nat := 2

theorem probability_of_U :
  ((u_count : ℚ) / (total_letters : ℚ)) = (1 / 3) :=
by
  sorry

end probability_of_U_l1269_126992


namespace intersection_points_of_line_l1269_126975

theorem intersection_points_of_line (x y : ℝ) :
  ((y = 2 * x - 1) → (y = 0 → x = 0.5)) ∧
  ((y = 2 * x - 1) → (x = 0 → y = -1)) :=
by sorry

end intersection_points_of_line_l1269_126975


namespace permutations_BANANA_l1269_126917

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l1269_126917


namespace product_of_rational_solutions_eq_twelve_l1269_126903

theorem product_of_rational_solutions_eq_twelve :
  ∃ c1 c2 : ℕ, (c1 > 0) ∧ (c2 > 0) ∧ 
               (∀ x : ℚ, ∃ (a b : ℤ), 5 * x^2 + 11 * x + c1 = 0 → 
                             ∃ (d : ℤ), b^2 - 4 * a * c1 = d^2) ∧
               (∀ x : ℚ, ∃ (a b : ℤ), 5 * x^2 + 11 * x + c2 = 0 → 
                             ∃ (d : ℤ), b^2 - 4 * a * c2 = d^2) ∧
               c1 * c2 = 12 := sorry

end product_of_rational_solutions_eq_twelve_l1269_126903


namespace maximum_notebooks_maria_can_buy_l1269_126966

def price_single : ℕ := 1
def price_pack_4 : ℕ := 3
def price_pack_7 : ℕ := 5
def total_budget : ℕ := 10

def max_notebooks (budget : ℕ) : ℕ :=
  if budget < price_single then 0
  else if budget < price_pack_4 then budget / price_single
  else if budget < price_pack_7 then max (budget / price_single) (4 * (budget / price_pack_4))
  else max (budget / price_single) (7 * (budget / price_pack_7))

theorem maximum_notebooks_maria_can_buy :
  max_notebooks total_budget = 14 := by
  sorry

end maximum_notebooks_maria_can_buy_l1269_126966


namespace hotel_towels_l1269_126931

def num_rooms : Nat := 10
def people_per_room : Nat := 3
def towels_per_person : Nat := 2

theorem hotel_towels : num_rooms * people_per_room * towels_per_person = 60 :=
by
  sorry

end hotel_towels_l1269_126931


namespace range_of_a_l1269_126902

variable (a : ℝ)
def p : Prop := a > 1/4
def q : Prop := a ≤ -1 ∨ a ≥ 1

theorem range_of_a :
  ((p a ∧ ¬ (q a)) ∨ (q a ∧ ¬ (p a))) ↔ (a > 1/4 ∧ a < 1) ∨ (a ≤ -1) :=
by
  sorry

end range_of_a_l1269_126902


namespace arithmetic_sequence_problem_l1269_126923

theorem arithmetic_sequence_problem (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : S 6 = 36)
  (h2 : S n = 324)
  (h3 : S (n - 6) = 144) :
  n = 18 := by
  sorry

end arithmetic_sequence_problem_l1269_126923


namespace mass_of_fourth_metal_l1269_126972

theorem mass_of_fourth_metal 
  (m1 m2 m3 m4 : ℝ)
  (total_mass : m1 + m2 + m3 + m4 = 20)
  (h1 : m1 = 1.5 * m2)
  (h2 : m2 = 3/4 * m3)
  (h3 : m3 = 5/6 * m4) :
  m4 = 20 * (48 / 163) :=
sorry

end mass_of_fourth_metal_l1269_126972


namespace inverse_proportional_x_y_l1269_126913

theorem inverse_proportional_x_y (x y k : ℝ) (h_inverse : x * y = k) (h_given : 40 * 5 = k) : x = 20 :=
by 
  sorry

end inverse_proportional_x_y_l1269_126913


namespace inequality_C_l1269_126962

variable (a b : ℝ)
variable (h : a > b)
variable (h' : b > 0)

theorem inequality_C : a + b > 2 * b := by
  sorry

end inequality_C_l1269_126962


namespace parabola_directrix_eq_l1269_126916

theorem parabola_directrix_eq (x : ℝ) : 
  (∀ y : ℝ, y = 3 * x^2 - 6 * x + 2 → True) →
  y = -13/12 := 
  sorry

end parabola_directrix_eq_l1269_126916


namespace farey_sequence_problem_l1269_126952

theorem farey_sequence_problem (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 5 / 8) 
  (h_minimal_q : ∀ p' q', 0 < p' ∧ 0 < q' → 3 / 5 < p' / q' → p' / q' < 5 / 8 → q' ≥ q) : 
  q - p = 5 := 
sorry

end farey_sequence_problem_l1269_126952


namespace martin_class_number_l1269_126971

theorem martin_class_number (b : ℕ) (h1 : 100 < b) (h2 : b < 200) 
  (h3 : b % 3 = 2) (h4 : b % 4 = 1) (h5 : b % 5 = 1) : 
  b = 101 ∨ b = 161 := 
by
  sorry

end martin_class_number_l1269_126971


namespace sallys_woodworking_llc_reimbursement_l1269_126976

/-
Conditions:
1. Remy paid $20,700 for 150 pieces of furniture.
2. The cost of a piece of furniture is $134.
-/
def reimbursement_amount (pieces_paid : ℕ) (total_paid : ℕ) (price_per_piece : ℕ) : ℕ :=
  total_paid - (pieces_paid * price_per_piece)

theorem sallys_woodworking_llc_reimbursement :
  reimbursement_amount 150 20700 134 = 600 :=
by 
  sorry

end sallys_woodworking_llc_reimbursement_l1269_126976


namespace total_books_l1269_126959

-- Defining the conditions
def darla_books := 6
def katie_books := darla_books / 2
def combined_books := darla_books + katie_books
def gary_books := 5 * combined_books

-- Statement to prove
theorem total_books : darla_books + katie_books + gary_books = 54 := by
  sorry

end total_books_l1269_126959


namespace sec_240_eq_neg2_l1269_126988

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_240_eq_neg2 : sec 240 = -2 := by
  -- Proof goes here
  sorry

end sec_240_eq_neg2_l1269_126988


namespace union_of_A_and_B_l1269_126921

open Set

def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {3, 5, 7, 8}

theorem union_of_A_and_B : A ∪ B = {3, 4, 5, 6, 7, 8} := by
  sorry

end union_of_A_and_B_l1269_126921


namespace value_of_m_l1269_126940

-- Defining the quadratic equation condition
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 3 * x + m^2 - 4

-- Defining the condition where the constant term in the quadratic equation is 0
def constant_term_zero (m : ℝ) : Prop := m^2 - 4 = 0

-- Stating the proof problem: given the conditions, prove that m = -2
theorem value_of_m (m : ℝ) (h1 : constant_term_zero m) (h2 : m ≠ 2) : m = -2 :=
by {
  sorry -- Proof to be developed
}

end value_of_m_l1269_126940


namespace solution_set_of_inequality_l1269_126960

theorem solution_set_of_inequality (x : ℝ) : (1 / x ≤ x) ↔ (-1 ≤ x ∧ x < 0) ∨ (x ≥ 1) := sorry

end solution_set_of_inequality_l1269_126960


namespace tangent_line_curve_l1269_126996

theorem tangent_line_curve (a b : ℝ)
  (h1 : ∀ (x : ℝ), (x - (x^2 + a*x + b) + 1 = 0) ↔ (a = 1 ∧ b = 1))
  (h2 : ∀ (y : ℝ), (0, y) ∈ { p : ℝ × ℝ | p.2 = 0 ^ 2 + a * 0 + b }) :
  a = 1 ∧ b = 1 :=
by
  sorry

end tangent_line_curve_l1269_126996
