import Mathlib

namespace partition_with_equal_product_l3009_300909

def numbers : List Nat := [2, 3, 12, 14, 15, 20, 21]

theorem partition_with_equal_product :
  ∃ (s₁ s₂ : List Nat),
    s₁ ∪ s₂ = numbers ∧
    s₁ ∩ s₂ = [] ∧
    s₁ ≠ [] ∧
    s₂ ≠ [] ∧
    (s₁.prod = 2520 ∧ s₂.prod = 2520) :=
  sorry

end partition_with_equal_product_l3009_300909


namespace jovana_shells_total_l3009_300968

/-- The total amount of shells in Jovana's bucket -/
def total_shells (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that given the initial and additional amounts of shells,
    the total amount in Jovana's bucket is 17 pounds -/
theorem jovana_shells_total :
  total_shells 5 12 = 17 := by
  sorry

end jovana_shells_total_l3009_300968


namespace probability_is_three_fifths_l3009_300943

-- Define the set S
def S : Finset ℤ := {-3, 0, 0, 4, 7, 8}

-- Define the function to check if a pair of integers has a product of 0
def productIsZero (x y : ℤ) : Bool :=
  x * y = 0

-- Define the probability calculation function
def probabilityOfZeroProduct (s : Finset ℤ) : ℚ :=
  let totalPairs := (s.card.choose 2 : ℚ)
  let zeroPairs := (s.filter (· = 0)).card * (s.filter (· ≠ 0)).card +
                   (if (s.filter (· = 0)).card ≥ 2 then 1 else 0)
  zeroPairs / totalPairs

-- State the theorem
theorem probability_is_three_fifths :
  probabilityOfZeroProduct S = 3/5 := by sorry

end probability_is_three_fifths_l3009_300943


namespace supplementary_angle_measures_l3009_300904

theorem supplementary_angle_measures :
  ∃ (possible_measures : Finset ℕ),
    (∀ A ∈ possible_measures,
      ∃ (B : ℕ) (k : ℕ),
        A > 0 ∧ B > 0 ∧ k > 0 ∧
        A + B = 180 ∧
        A = k * B) ∧
    (∀ A : ℕ,
      (∃ (B : ℕ) (k : ℕ),
        A > 0 ∧ B > 0 ∧ k > 0 ∧
        A + B = 180 ∧
        A = k * B) →
      A ∈ possible_measures) ∧
    Finset.card possible_measures = 17 :=
by sorry

end supplementary_angle_measures_l3009_300904


namespace cos_120_degrees_l3009_300931

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1/2 := by
  sorry

end cos_120_degrees_l3009_300931


namespace larger_tart_flour_usage_l3009_300988

theorem larger_tart_flour_usage
  (small_tarts : ℕ)
  (large_tarts : ℕ)
  (small_flour : ℚ)
  (h1 : small_tarts = 50)
  (h2 : large_tarts = 25)
  (h3 : small_flour = 1 / 8)
  (h4 : small_tarts * small_flour = large_tarts * large_flour) :
  large_flour = 1 / 4 :=
by
  sorry

#check larger_tart_flour_usage

end larger_tart_flour_usage_l3009_300988


namespace course_selection_combinations_l3009_300901

/-- The number of available courses -/
def num_courses : ℕ := 4

/-- The number of courses student A chooses -/
def courses_A : ℕ := 2

/-- The number of courses students B and C each choose -/
def courses_BC : ℕ := 3

/-- The total number of different possible combinations -/
def total_combinations : ℕ := Nat.choose num_courses courses_A * (Nat.choose num_courses courses_BC)^2

theorem course_selection_combinations :
  total_combinations = 96 :=
by sorry

end course_selection_combinations_l3009_300901


namespace prove_callys_colored_shirts_l3009_300969

/-- The number of colored shirts Cally washed -/
def callys_colored_shirts : ℕ := 5

theorem prove_callys_colored_shirts :
  let callys_other_clothes : ℕ := 10 + 7 + 6 -- white shirts + shorts + pants
  let dannys_clothes : ℕ := 6 + 8 + 10 + 6 -- white shirts + colored shirts + shorts + pants
  let total_clothes : ℕ := 58
  callys_colored_shirts = total_clothes - (callys_other_clothes + dannys_clothes) :=
by sorry

end prove_callys_colored_shirts_l3009_300969


namespace no_points_above_diagonal_l3009_300916

-- Define the triangle
def triangle : Set (ℝ × ℝ) :=
  {p | ∃ (t₁ t₂ : ℝ), 0 ≤ t₁ ∧ 0 ≤ t₂ ∧ t₁ + t₂ ≤ 1 ∧
    p = (4 * t₁ + 4 * t₂, 10 * t₂)}

-- Theorem statement
theorem no_points_above_diagonal (a b : ℝ) :
  (a, b) ∈ triangle → a - b ≤ 0 := by sorry

end no_points_above_diagonal_l3009_300916


namespace square_side_length_l3009_300940

theorem square_side_length (area : ℚ) (side : ℚ) 
  (h1 : area = 9/16) (h2 : side^2 = area) : side = 3/4 := by
  sorry

end square_side_length_l3009_300940


namespace largest_decimal_l3009_300995

theorem largest_decimal : 
  let a := 0.9123
  let b := 0.9912
  let c := 0.9191
  let d := 0.9301
  let e := 0.9091
  b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end largest_decimal_l3009_300995


namespace pet_store_dogs_l3009_300932

theorem pet_store_dogs (initial_dogs : ℕ) (sunday_dogs : ℕ) (monday_dogs : ℕ) (final_dogs : ℕ)
  (h1 : initial_dogs = 2)
  (h2 : monday_dogs = 3)
  (h3 : final_dogs = 10)
  (h4 : initial_dogs + sunday_dogs + monday_dogs = final_dogs) :
  sunday_dogs = 5 := by
  sorry

end pet_store_dogs_l3009_300932


namespace filling_pipe_time_calculation_l3009_300947

/-- The time it takes to fill the tank when both pipes are open -/
def both_pipes_time : ℝ := 180

/-- The time it takes for the emptying pipe to empty the tank -/
def emptying_pipe_time : ℝ := 45

/-- The time it takes for the filling pipe to fill the tank -/
def filling_pipe_time : ℝ := 36

theorem filling_pipe_time_calculation :
  (1 / filling_pipe_time) - (1 / emptying_pipe_time) = (1 / both_pipes_time) :=
sorry

end filling_pipe_time_calculation_l3009_300947


namespace specific_tetrahedron_volume_l3009_300954

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 20/3 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 5,
    PS := 5,
    QR := 5,
    QS := 4,
    RS := (10/3) * Real.sqrt 3
  }
  tetrahedronVolume t = 20/3 := by sorry

end specific_tetrahedron_volume_l3009_300954


namespace potato_cooking_time_l3009_300946

def cooking_problem (total_potatoes cooked_potatoes time_per_potato : ℕ) : Prop :=
  let remaining_potatoes := total_potatoes - cooked_potatoes
  remaining_potatoes * time_per_potato = 45

theorem potato_cooking_time :
  cooking_problem 16 7 5 := by
  sorry

end potato_cooking_time_l3009_300946


namespace quadratic_inequality_solution_l3009_300970

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 3 * x + a

-- Define the solution set condition
def is_solution_set (a m : ℝ) : Prop :=
  ∀ x, f a x < 0 ↔ m < x ∧ x < 1

-- Theorem statement
theorem quadratic_inequality_solution (a m : ℝ) 
  (h : is_solution_set a m) : m = 1/2 := by
  sorry

end quadratic_inequality_solution_l3009_300970


namespace common_root_quadratic_equations_l3009_300933

theorem common_root_quadratic_equations (b : ℤ) :
  (∃ x : ℝ, 2 * x^2 + (3 * b - 1) * x - 3 = 0 ∧ 6 * x^2 - (2 * b - 3) * x - 1 = 0) ↔ b = 2 :=
by sorry

end common_root_quadratic_equations_l3009_300933


namespace johns_age_l3009_300918

theorem johns_age (john_age dad_age : ℕ) : 
  dad_age = john_age + 30 →
  john_age + dad_age = 80 →
  john_age = 25 := by
sorry

end johns_age_l3009_300918


namespace page_number_added_twice_l3009_300977

theorem page_number_added_twice (n : ℕ) : 
  (n * (n + 1) / 2 ≤ 1986) ∧ 
  ((n + 1) * (n + 2) / 2 > 1986) →
  1986 - (n * (n + 1) / 2) = 33 := by
sorry

end page_number_added_twice_l3009_300977


namespace distance_to_origin_of_complex_fraction_l3009_300985

theorem distance_to_origin_of_complex_fraction : 
  let z : ℂ := 2 / (1 + Complex.I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end distance_to_origin_of_complex_fraction_l3009_300985


namespace base8_to_base10_3206_l3009_300915

/-- Converts a base 8 number to base 10 --/
def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base 8 representation of the number --/
def base8_num : List Nat := [6, 0, 2, 3]

/-- Theorem stating that the base 10 representation of 3206₈ is 1670 --/
theorem base8_to_base10_3206 : base8_to_base10 base8_num = 1670 := by
  sorry

end base8_to_base10_3206_l3009_300915


namespace final_load_is_30600_l3009_300907

def initial_load : ℝ := 50000

def first_unload_percent : ℝ := 0.1
def second_unload_percent : ℝ := 0.2
def third_unload_percent : ℝ := 0.15

def remaining_after_first (load : ℝ) : ℝ :=
  load * (1 - first_unload_percent)

def remaining_after_second (load : ℝ) : ℝ :=
  load * (1 - second_unload_percent)

def remaining_after_third (load : ℝ) : ℝ :=
  load * (1 - third_unload_percent)

theorem final_load_is_30600 :
  remaining_after_third (remaining_after_second (remaining_after_first initial_load)) = 30600 := by
  sorry

end final_load_is_30600_l3009_300907


namespace three_fifths_of_twelve_times_ten_minus_twenty_l3009_300944

theorem three_fifths_of_twelve_times_ten_minus_twenty : 
  (3 : ℚ) / 5 * ((12 * 10) - 20) = 60 := by
  sorry

end three_fifths_of_twelve_times_ten_minus_twenty_l3009_300944


namespace second_wing_floors_is_seven_l3009_300964

/-- A hotel with two wings -/
structure Hotel where
  total_rooms : ℕ
  wing1_floors : ℕ
  wing1_halls_per_floor : ℕ
  wing1_rooms_per_hall : ℕ
  wing2_halls_per_floor : ℕ
  wing2_rooms_per_hall : ℕ

/-- Calculate the number of floors in the second wing -/
def second_wing_floors (h : Hotel) : ℕ :=
  let wing1_rooms := h.wing1_floors * h.wing1_halls_per_floor * h.wing1_rooms_per_hall
  let wing2_rooms := h.total_rooms - wing1_rooms
  let rooms_per_floor_wing2 := h.wing2_halls_per_floor * h.wing2_rooms_per_hall
  wing2_rooms / rooms_per_floor_wing2

/-- The theorem stating that the number of floors in the second wing is 7 -/
theorem second_wing_floors_is_seven (h : Hotel) 
    (h_total : h.total_rooms = 4248)
    (h_wing1_floors : h.wing1_floors = 9)
    (h_wing1_halls : h.wing1_halls_per_floor = 6)
    (h_wing1_rooms : h.wing1_rooms_per_hall = 32)
    (h_wing2_halls : h.wing2_halls_per_floor = 9)
    (h_wing2_rooms : h.wing2_rooms_per_hall = 40) : 
  second_wing_floors h = 7 := by
  sorry

#eval second_wing_floors {
  total_rooms := 4248,
  wing1_floors := 9,
  wing1_halls_per_floor := 6,
  wing1_rooms_per_hall := 32,
  wing2_halls_per_floor := 9,
  wing2_rooms_per_hall := 40
}

end second_wing_floors_is_seven_l3009_300964


namespace license_plate_combinations_l3009_300961

def letter_count : ℕ := 26
def digit_count : ℕ := 10
def letter_positions : ℕ := 4
def digit_positions : ℕ := 2

theorem license_plate_combinations : 
  (Nat.choose letter_count 2 * 2 * Nat.choose letter_positions 2 * digit_count ^ digit_positions) = 390000 := by
  sorry

end license_plate_combinations_l3009_300961


namespace final_cell_population_l3009_300980

/-- Represents the cell population growth over time -/
def cell_population (initial_cells : ℕ) (split_factor : ℕ) (days : ℕ) : ℕ :=
  initial_cells * split_factor ^ (days / 3)

/-- Theorem: Given the conditions, the final cell population after 9 days is 18 -/
theorem final_cell_population :
  cell_population 2 3 9 = 18 := by
  sorry

#eval cell_population 2 3 9

end final_cell_population_l3009_300980


namespace distinct_prime_factors_of_30_factorial_l3009_300917

theorem distinct_prime_factors_of_30_factorial :
  (∀ p : ℕ, p.Prime → p ≤ 30 → p ∣ Nat.factorial 30) ∧
  (∃ S : Finset ℕ, (∀ p ∈ S, p.Prime ∧ p ≤ 30) ∧ 
                   (∀ p : ℕ, p.Prime → p ≤ 30 → p ∈ S) ∧ 
                   S.card = 10) :=
by sorry

end distinct_prime_factors_of_30_factorial_l3009_300917


namespace max_value_of_a_l3009_300913

-- Define the condition function
def condition (x : ℝ) : Prop := x^2 - 2*x - 3 > 0

-- Define the theorem
theorem max_value_of_a :
  (∃ a : ℝ, ∀ x : ℝ, x < a → condition x) ∧
  (∀ a : ℝ, ∃ x : ℝ, condition x ∧ x ≥ a) →
  (∀ a : ℝ, (∀ x : ℝ, x < a → condition x) → a ≤ -1) ∧
  (∀ x : ℝ, x < -1 → condition x) :=
sorry

end max_value_of_a_l3009_300913


namespace nonagon_prism_edges_l3009_300963

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  base : Nat  -- number of sides in the base shape

/-- The number of edges in a prism. -/
def Prism.edges (p : Prism) : Nat :=
  3 * p.base

theorem nonagon_prism_edges :
  ∀ p : Prism, p.base = 9 → p.edges = 27 :=
by
  sorry

end nonagon_prism_edges_l3009_300963


namespace arithmetic_progression_sum_l3009_300926

/-- An arithmetic progression with sum of first n terms S_n -/
structure ArithmeticProgression where
  S : ℕ → ℚ
  sum_formula : ∀ n : ℕ, S n = n / 2 * (2 * a + (n - 1) * d)
  a : ℚ
  d : ℚ

/-- Theorem stating that if S_3 = 2 and S_6 = 6, then S_24 = 510 for an arithmetic progression -/
theorem arithmetic_progression_sum (ap : ArithmeticProgression) 
  (h1 : ap.S 3 = 2) (h2 : ap.S 6 = 6) : ap.S 24 = 510 := by
  sorry

end arithmetic_progression_sum_l3009_300926


namespace dividend_calculation_l3009_300974

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 19)
  (h_quotient : quotient = 7)
  (h_remainder : remainder = 6) :
  divisor * quotient + remainder = 139 := by
sorry

end dividend_calculation_l3009_300974


namespace contrapositive_equivalence_l3009_300920

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(- Real.sqrt b < a ∧ a < Real.sqrt b) → ¬(a^2 < b)) ↔
  ((a ≥ Real.sqrt b ∨ a ≤ - Real.sqrt b) → a^2 ≥ b) :=
sorry

end contrapositive_equivalence_l3009_300920


namespace overall_average_speed_l3009_300999

theorem overall_average_speed
  (car_time : Real) (car_speed : Real) (horse_time : Real) (horse_speed : Real)
  (h1 : car_time = 45 / 60)
  (h2 : car_speed = 20)
  (h3 : horse_time = 30 / 60)
  (h4 : horse_speed = 6)
  : (car_speed * car_time + horse_speed * horse_time) / (car_time + horse_time) = 14.4 := by
  sorry

end overall_average_speed_l3009_300999


namespace power_difference_squared_l3009_300927

theorem power_difference_squared (n : ℕ) :
  (5^(1001 : ℕ) + 6^(1002 : ℕ))^2 - (5^(1001 : ℕ) - 6^(1002 : ℕ))^2 = 24 * 30^(1001 : ℕ) := by
  sorry

end power_difference_squared_l3009_300927


namespace otimes_inequality_implies_a_range_l3009_300908

-- Define the ⊗ operation
def otimes (x y : ℝ) := x * (1 - y)

-- State the theorem
theorem otimes_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) →
  -1/2 < a ∧ a < 3/2 :=
by sorry

end otimes_inequality_implies_a_range_l3009_300908


namespace product_of_roots_l3009_300906

theorem product_of_roots (x : ℝ) : 
  (∃ α β : ℝ, α * β = -21 ∧ -α^2 + 4*α = -21 ∧ -β^2 + 4*β = -21) := by
  sorry

end product_of_roots_l3009_300906


namespace number_division_property_l3009_300923

theorem number_division_property : ∃ x : ℝ, x / 5 = 80 + x / 6 := by
  sorry

end number_division_property_l3009_300923


namespace polyhedron_parity_l3009_300900

-- Define a polyhedron structure
structure Polyhedron where
  vertices : Set (ℕ × ℕ × ℕ)
  edges : Set (Set (ℕ × ℕ × ℕ))
  faces : Set (Set (ℕ × ℕ × ℕ))
  -- Add necessary conditions for a valid polyhedron

-- Function to count faces with odd number of sides
def count_odd_faces (p : Polyhedron) : ℕ := sorry

-- Function to count vertices with odd degree
def count_odd_degree_vertices (p : Polyhedron) : ℕ := sorry

-- Theorem statement
theorem polyhedron_parity (p : Polyhedron) : 
  Even (count_odd_faces p) ∧ Even (count_odd_degree_vertices p) := by sorry

end polyhedron_parity_l3009_300900


namespace point_in_second_quadrant_implies_m_range_l3009_300962

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The theorem states that if the point P(m-3, m-2) is in the second quadrant,
    then m is strictly between 2 and 3. -/
theorem point_in_second_quadrant_implies_m_range
  (m : ℝ)
  (h : is_in_second_quadrant (m - 3) (m - 2)) :
  2 < m ∧ m < 3 :=
sorry

end point_in_second_quadrant_implies_m_range_l3009_300962


namespace min_value_quadratic_l3009_300924

theorem min_value_quadratic :
  ∃ (min_y : ℝ), min_y = -44 ∧ ∀ (x y : ℝ), y = x^2 + 16*x + 20 → y ≥ min_y :=
by sorry

end min_value_quadratic_l3009_300924


namespace sufficient_condition_transitivity_l3009_300984

theorem sufficient_condition_transitivity (p q r : Prop) :
  (p → q) → (q → r) → (p → r) := by
  sorry

end sufficient_condition_transitivity_l3009_300984


namespace download_speed_calculation_l3009_300993

theorem download_speed_calculation (total_size : ℝ) (downloaded : ℝ) (remaining_time : ℝ)
  (h1 : total_size = 880)
  (h2 : downloaded = 310)
  (h3 : remaining_time = 190) :
  (total_size - downloaded) / remaining_time = 3 := by
  sorry

end download_speed_calculation_l3009_300993


namespace min_workers_for_profit_l3009_300967

/-- Represents the company's financial model -/
structure CompanyModel where
  maintenance_fee : ℕ  -- Daily maintenance fee in dollars
  hourly_wage : ℕ      -- Hourly wage per worker in dollars
  widgets_per_hour : ℕ -- Widgets produced per worker per hour
  widget_price : ℚ     -- Selling price per widget in dollars
  work_hours : ℕ       -- Work hours per day

/-- Calculates the daily cost for a given number of workers -/
def daily_cost (model : CompanyModel) (workers : ℕ) : ℕ :=
  model.maintenance_fee + model.hourly_wage * workers * model.work_hours

/-- Calculates the daily revenue for a given number of workers -/
def daily_revenue (model : CompanyModel) (workers : ℕ) : ℚ :=
  (model.widgets_per_hour : ℚ) * model.widget_price * (workers : ℚ) * (model.work_hours : ℚ)

/-- Theorem stating the minimum number of workers needed for profit -/
theorem min_workers_for_profit (model : CompanyModel) 
  (h_maintenance : model.maintenance_fee = 600)
  (h_wage : model.hourly_wage = 20)
  (h_widgets : model.widgets_per_hour = 6)
  (h_price : model.widget_price = 7/2)
  (h_hours : model.work_hours = 7) :
  ∃ n : ℕ, (∀ m : ℕ, m ≥ n → daily_revenue model m > daily_cost model m) ∧
           (∀ m : ℕ, m < n → daily_revenue model m ≤ daily_cost model m) ∧
           n = 86 :=
sorry

end min_workers_for_profit_l3009_300967


namespace fraction_comparison_l3009_300938

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  b / (a - c) < a / (b - d) := by
  sorry

end fraction_comparison_l3009_300938


namespace length_of_AB_l3009_300928

-- Define the line l: kx + y - 2 = 0
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x + y - 2 = 0

-- Define the circle C: x^2 + y^2 - 6x + 2y + 9 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 2 * y + 9 = 0

-- Define the point A
def point_A (k : ℝ) : ℝ × ℝ := (0, k)

-- Define the condition that line l is the axis of symmetry for circle C
def is_axis_of_symmetry (k : ℝ) : Prop :=
  ∃ (center_x center_y : ℝ), line_l k center_x center_y ∧
    ∀ (x y : ℝ), circle_C x y ↔ circle_C (2 * center_x - x) (2 * center_y - y)

-- Define the tangency condition
def is_tangent (k : ℝ) (B : ℝ × ℝ) : Prop :=
  circle_C B.1 B.2 ∧
  ∃ (t : ℝ), B = (t, k * t + 2) ∧
    ∀ (x y : ℝ), line_l k x y → (circle_C x y → x = B.1 ∧ y = B.2)

-- State the theorem
theorem length_of_AB (k : ℝ) (B : ℝ × ℝ) :
  is_axis_of_symmetry k →
  is_tangent k B →
  Real.sqrt ((B.1 - 0)^2 + (B.2 - k)^2) = 2 * Real.sqrt 3 :=
sorry

end length_of_AB_l3009_300928


namespace prob_shortest_diagonal_icosagon_l3009_300921

/-- A regular polygon with n sides -/
structure RegularPolygon where
  n : ℕ
  h : n ≥ 3

/-- The number of diagonals in a regular polygon -/
def num_diagonals (p : RegularPolygon) : ℕ := p.n * (p.n - 3) / 2

/-- The number of shortest diagonals in a regular polygon -/
def num_shortest_diagonals (p : RegularPolygon) : ℕ := p.n / 2

/-- An icosagon is a regular polygon with 20 sides -/
def icosagon : RegularPolygon where
  n := 20
  h := by norm_num

/-- The probability of selecting a shortest diagonal in an icosagon -/
def prob_shortest_diagonal (p : RegularPolygon) : ℚ :=
  (num_shortest_diagonals p : ℚ) / (num_diagonals p : ℚ)

theorem prob_shortest_diagonal_icosagon :
  prob_shortest_diagonal icosagon = 1 / 17 := by
  sorry

end prob_shortest_diagonal_icosagon_l3009_300921


namespace captain_selection_count_l3009_300982

/-- The number of ways to choose k items from n items without regard to order -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of people in the team -/
def team_size : ℕ := 15

/-- The number of captains to be chosen -/
def captain_count : ℕ := 4

/-- Theorem: The number of ways to choose 4 captains from a team of 15 people is 1365 -/
theorem captain_selection_count : choose team_size captain_count = 1365 := by
  sorry

end captain_selection_count_l3009_300982


namespace sum_interior_angles_regular_polygon_l3009_300976

/-- For a regular polygon with exterior angles of 40 degrees, 
    the sum of interior angles is 1260 degrees. -/
theorem sum_interior_angles_regular_polygon : 
  ∀ n : ℕ, 
  n > 2 → 
  360 / n = 40 → 
  (n - 2) * 180 = 1260 :=
by
  sorry

end sum_interior_angles_regular_polygon_l3009_300976


namespace logarithmic_inequality_l3009_300991

theorem logarithmic_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (Real.log a)/((a-b)*(a-c)) + (Real.log b)/((b-c)*(b-a)) + (Real.log c)/((c-a)*(c-b)) < 0 := by
  sorry

end logarithmic_inequality_l3009_300991


namespace original_number_proof_l3009_300998

theorem original_number_proof (x : ℝ) 
  (h1 : x * 74 = 19832) 
  (h2 : x / 100 * 0.74 = 1.9832) : 
  x = 268 := by
sorry

end original_number_proof_l3009_300998


namespace stratified_sample_over_45_l3009_300922

/-- Represents the number of employees in a stratified sample from a given population -/
def stratified_sample (total_population : ℕ) (group_size : ℕ) (sample_size : ℕ) : ℕ :=
  (group_size * sample_size) / total_population

/-- Proves that the number of employees over 45 in the stratified sample is 10 -/
theorem stratified_sample_over_45 :
  stratified_sample 200 80 25 = 10 := by
  sorry

end stratified_sample_over_45_l3009_300922


namespace simplify_radicals_l3009_300929

theorem simplify_radicals : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end simplify_radicals_l3009_300929


namespace simplify_and_evaluate_l3009_300911

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (1 - a / (a + 1)) / ((a^2 - 2*a + 1) / (a^2 - 1)) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l3009_300911


namespace third_number_from_lcm_hcf_l3009_300951

/-- Given three positive integers with known LCM and HCF, prove the third number -/
theorem third_number_from_lcm_hcf (A B C : ℕ+) : 
  A = 36 → B = 44 → Nat.lcm A (Nat.lcm B C) = 792 → Nat.gcd A (Nat.gcd B C) = 12 → C = 6 := by
  sorry

end third_number_from_lcm_hcf_l3009_300951


namespace inequality_proof_l3009_300960

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 * a * b) / (a + b) < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 := by
  sorry

end inequality_proof_l3009_300960


namespace variance_invariant_under_translation_l3009_300965

def variance (data : List ℝ) : ℝ := sorry

theorem variance_invariant_under_translation (data : List ℝ) (c : ℝ) :
  variance data = variance (data.map (λ x => x - c)) := by sorry

end variance_invariant_under_translation_l3009_300965


namespace office_distance_l3009_300950

/-- The distance to the office in kilometers -/
def distance : ℝ := sorry

/-- The time it takes to reach the office on time in hours -/
def on_time : ℝ := sorry

/-- Condition 1: At 10 kmph, the person arrives 10 minutes late -/
axiom condition_1 : distance = 10 * (on_time + 1/6)

/-- Condition 2: At 15 kmph, the person arrives 10 minutes early -/
axiom condition_2 : distance = 15 * (on_time - 1/6)

/-- Theorem: The distance to the office is 10 kilometers -/
theorem office_distance : distance = 10 := by sorry

end office_distance_l3009_300950


namespace stream_speed_l3009_300981

/-- Proves that given a boat with a speed of 22 km/hr in still water,
    traveling 216 km downstream in 8 hours, the speed of the stream is 5 km/hr. -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) :
  boat_speed = 22 →
  distance = 216 →
  time = 8 →
  distance = (boat_speed + stream_speed) * time →
  stream_speed = 5 := by
sorry

end stream_speed_l3009_300981


namespace alices_preferred_number_l3009_300990

def is_between (n : ℕ) (a b : ℕ) : Prop := a ≤ n ∧ n ≤ b

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem alices_preferred_number :
  ∃! n : ℕ,
    is_between n 100 200 ∧
    11 ∣ n ∧
    ¬(2 ∣ n) ∧
    3 ∣ sum_of_digits n ∧
    n = 165 :=
by sorry

end alices_preferred_number_l3009_300990


namespace total_pears_picked_l3009_300983

theorem total_pears_picked (sara_pears tim_pears : ℕ) 
  (h1 : sara_pears = 6) 
  (h2 : tim_pears = 5) : 
  sara_pears + tim_pears = 11 := by
  sorry

end total_pears_picked_l3009_300983


namespace rose_price_l3009_300930

/-- The price of roses given Hanna's budget and distribution to friends -/
theorem rose_price (total_budget : ℚ) (jenna_fraction : ℚ) (imma_fraction : ℚ) (friends_roses : ℕ) : 
  total_budget = 300 →
  jenna_fraction = 1/3 →
  imma_fraction = 1/2 →
  friends_roses = 125 →
  total_budget / ((friends_roses : ℚ) / (jenna_fraction + imma_fraction)) = 2 :=
by sorry

end rose_price_l3009_300930


namespace negation_of_existential_inequality_l3009_300994

theorem negation_of_existential_inequality :
  ¬(∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) := by sorry

end negation_of_existential_inequality_l3009_300994


namespace number_of_boys_l3009_300952

/-- Given conditions about men, women, and boys with their earnings, prove the number of boys --/
theorem number_of_boys (men women boys : ℕ) (total_earnings men_wage : ℕ) : 
  men = 5 →
  men = women →
  women = boys →
  total_earnings = 150 →
  men_wage = 10 →
  boys = 10 := by
  sorry

end number_of_boys_l3009_300952


namespace problem_statement_l3009_300925

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end problem_statement_l3009_300925


namespace percent_problem_l3009_300919

theorem percent_problem (x : ℝ) : (0.0001 * x = 1.2356) → x = 12356 := by
  sorry

end percent_problem_l3009_300919


namespace student_marks_proof_l3009_300936

/-- Given a student's marks in mathematics, physics, and chemistry,
    prove that the total marks in mathematics and physics is 50. -/
theorem student_marks_proof (M P C : ℕ) : 
  C = P + 20 →
  (M + C) / 2 = 35 →
  M + P = 50 := by
sorry

end student_marks_proof_l3009_300936


namespace toms_weekly_income_l3009_300902

/-- Tom's crab fishing business --/
def crab_business (num_buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ) : ℕ :=
  num_buckets * crabs_per_bucket * price_per_crab * days_per_week

/-- Tom's weekly income from selling crabs --/
theorem toms_weekly_income :
  crab_business 8 12 5 7 = 3360 := by
  sorry

#eval crab_business 8 12 5 7

end toms_weekly_income_l3009_300902


namespace equal_intercept_line_equation_l3009_300957

-- Define a line passing through (2,5) with equal intercepts
structure EqualInterceptLine where
  -- The slope of the line
  slope : ℝ
  -- The y-intercept of the line
  y_intercept : ℝ
  -- Condition: Line passes through (2,5)
  point_condition : 5 = slope * 2 + y_intercept
  -- Condition: Equal intercepts on both axes
  equal_intercepts : y_intercept = slope * y_intercept

-- Theorem statement
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = 5/2 ∧ l.y_intercept = 0) ∨ (l.slope = -1 ∧ l.y_intercept = 7) :=
sorry

end equal_intercept_line_equation_l3009_300957


namespace remove_six_maximizes_probability_l3009_300956

def original_list : List Int := [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def remove_number (list : List Int) (n : Int) : List Int :=
  list.filter (· ≠ n)

def count_pairs_sum_12 (list : List Int) : Nat :=
  (list.filterMap (λ x => 
    if x < 12 ∧ list.contains (12 - x) ∧ x ≠ 12 - x
    then some (min x (12 - x))
    else none
  )).dedup.length

theorem remove_six_maximizes_probability : 
  ∀ n ∈ original_list, n ≠ 6 → 
    count_pairs_sum_12 (remove_number original_list 6) ≥ 
    count_pairs_sum_12 (remove_number original_list n) :=
by sorry

end remove_six_maximizes_probability_l3009_300956


namespace cubic_sum_problem_l3009_300903

theorem cubic_sum_problem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_problem_l3009_300903


namespace fraction_equality_l3009_300941

theorem fraction_equality : (4 + 5) / (7 + 5) = 3 / 4 := by
  sorry

end fraction_equality_l3009_300941


namespace charlie_share_l3009_300934

def distribute_money (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) (deduct1 deduct2 deduct3 : ℕ) : ℕ × ℕ × ℕ :=
  sorry

theorem charlie_share :
  let (alice, bond, charlie) := distribute_money 1105 11 18 24 10 20 15
  charlie = 495 := by sorry

end charlie_share_l3009_300934


namespace inequality_solution_l3009_300989

theorem inequality_solution (x : ℝ) :
  x ≠ -3 ∧ x ≠ 4 →
  ((x - 3) / (x + 3) > (2 * x - 1) / (x - 4) ↔
   (x > -6 - 3 * Real.sqrt 17 ∧ x < -6 + 3 * Real.sqrt 17) ∨
   (x > -3 ∧ x < 4)) :=
by sorry

end inequality_solution_l3009_300989


namespace angle_A_is_pi_over_three_max_area_when_a_is_four_l3009_300978

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  (t.a + t.b) * (Real.sin t.A - Real.sin t.B) = t.c * (Real.sin t.C - Real.sin t.B)

-- Theorem for part 1
theorem angle_A_is_pi_over_three (t : Triangle) (h : condition t) : t.A = π / 3 := by
  sorry

-- Theorem for part 2
theorem max_area_when_a_is_four (t : Triangle) (h1 : condition t) (h2 : t.a = 4) :
  ∃ (S : ℝ), S = 4 * Real.sqrt 3 ∧ ∀ (S' : ℝ), S' = 1/2 * t.b * t.c * Real.sin t.A → S' ≤ S := by
  sorry

end angle_A_is_pi_over_three_max_area_when_a_is_four_l3009_300978


namespace simplify_expression_l3009_300992

theorem simplify_expression : (((3 + 4 + 5 + 6) / 3) + ((3 * 4 + 9) / 4)) = 45 / 4 := by
  sorry

end simplify_expression_l3009_300992


namespace consecutive_odd_integers_sum_l3009_300942

theorem consecutive_odd_integers_sum (x : ℤ) : 
  (∃ y : ℤ, y = x + 2 ∧ Odd x ∧ Odd y ∧ y = 5 * x - 2) → x + (x + 2) = 4 := by
  sorry

end consecutive_odd_integers_sum_l3009_300942


namespace x_over_y_equals_four_l3009_300971

theorem x_over_y_equals_four (x y : ℝ) (h1 : y ≠ 0) (h2 : 2 * x - y = 1.75 * x) : x / y = 4 := by
  sorry

end x_over_y_equals_four_l3009_300971


namespace total_profit_calculation_l3009_300948

/-- Represents a partner's investment information -/
structure PartnerInvestment where
  initial : ℝ
  monthly : ℝ

/-- Calculates the total annual investment for a partner -/
def annualInvestment (p : PartnerInvestment) : ℝ :=
  p.initial + 12 * p.monthly

/-- Represents the investment information for all partners -/
structure Investments where
  a : PartnerInvestment
  b : PartnerInvestment
  c : PartnerInvestment

/-- Calculates the total investment for all partners -/
def totalInvestment (inv : Investments) : ℝ :=
  annualInvestment inv.a + annualInvestment inv.b + annualInvestment inv.c

/-- The main theorem stating the total profit given the conditions -/
theorem total_profit_calculation (inv : Investments) 
    (h1 : inv.a = { initial := 45000, monthly := 1500 })
    (h2 : inv.b = { initial := 63000, monthly := 2100 })
    (h3 : inv.c = { initial := 72000, monthly := 2400 })
    (h4 : (annualInvestment inv.c / totalInvestment inv) * 60000 = 24000) :
    60000 = (totalInvestment inv * 24000) / (annualInvestment inv.c) := by
  sorry

end total_profit_calculation_l3009_300948


namespace sports_suits_cost_price_l3009_300939

/-- The cost price of one set of type A sports suits -/
def cost_A : ℝ := 180

/-- The cost price of one set of type B sports suits -/
def cost_B : ℝ := 200

/-- The total cost of purchasing one set of each type -/
def total_cost : ℝ := 380

/-- The amount spent on type A sports suits -/
def amount_A : ℝ := 8100

/-- The amount spent on type B sports suits -/
def amount_B : ℝ := 9000

theorem sports_suits_cost_price :
  cost_A + cost_B = total_cost ∧
  amount_A / cost_A = amount_B / cost_B :=
by sorry

end sports_suits_cost_price_l3009_300939


namespace circle_condition_l3009_300910

-- Define the equation
def circle_equation (a x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = 0

-- Define what it means for an equation to represent a circle
def represents_circle (a : ℝ) : Prop :=
  a^2 = a + 2 ∧ a ≠ 0

-- Theorem statement
theorem circle_condition (a : ℝ) :
  represents_circle a ↔ a = -1 :=
sorry

end circle_condition_l3009_300910


namespace monica_money_exchange_l3009_300935

def exchange_rate : ℚ := 8 / 5

theorem monica_money_exchange (x : ℚ) : 
  (exchange_rate * x - 40 = x) → x = 200 := by
  sorry

end monica_money_exchange_l3009_300935


namespace tree_height_when_boy_is_36_inches_l3009_300987

/-- Calculates the final height of a tree given initial heights and growth rates -/
def final_tree_height (initial_tree_height : ℝ) (initial_boy_height : ℝ) (final_boy_height : ℝ) : ℝ :=
  initial_tree_height + 2 * (final_boy_height - initial_boy_height)

/-- Proves that the tree will be 40 inches tall when the boy is 36 inches tall -/
theorem tree_height_when_boy_is_36_inches :
  final_tree_height 16 24 36 = 40 := by
  sorry

end tree_height_when_boy_is_36_inches_l3009_300987


namespace orange_cost_proof_l3009_300973

/-- Given that 5 dozen oranges cost $39.00, prove that 8 dozen oranges at the same rate cost $62.40 -/
theorem orange_cost_proof (cost_five_dozen : ℝ) (h1 : cost_five_dozen = 39) :
  let cost_per_dozen : ℝ := cost_five_dozen / 5
  let cost_eight_dozen : ℝ := 8 * cost_per_dozen
  cost_eight_dozen = 62.4 := by
  sorry

end orange_cost_proof_l3009_300973


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3009_300959

/-- The distance from point (2,1) to the line x=a is 3 -/
def distance_condition (a : ℝ) : Prop := |a - 2| = 3

/-- a=5 is a sufficient condition -/
theorem sufficient_condition : distance_condition 5 := by sorry

/-- a=5 is not a necessary condition -/
theorem not_necessary_condition : ∃ x, x ≠ 5 ∧ distance_condition x := by sorry

/-- a=5 is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary : 
  (distance_condition 5) ∧ (∃ x, x ≠ 5 ∧ distance_condition x) := by sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3009_300959


namespace f_minimum_f_has_root_l3009_300912

noncomputable section

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (x - m) - x

-- Statement for the extremum of f(x)
theorem f_minimum (m : ℝ) : 
  (∀ x : ℝ, f m x ≥ f m m) ∧ f m m = 1 - m :=
sorry

-- Statement for the existence of a root in (m, 2m) when m > 1
theorem f_has_root (m : ℝ) (h : m > 1) : 
  ∃ x : ℝ, m < x ∧ x < 2*m ∧ f m x = 0 :=
sorry

end

end f_minimum_f_has_root_l3009_300912


namespace problem_solution_l3009_300972

theorem problem_solution : (((3^1 : ℝ) + 2 + 6^2 + 3)⁻¹ * 6) = 3/22 := by sorry

end problem_solution_l3009_300972


namespace factoring_expression_l3009_300955

theorem factoring_expression (x y : ℝ) : 3*x*(x+3) + y*(x+3) = (x+3)*(3*x+y) := by
  sorry

end factoring_expression_l3009_300955


namespace probability_james_and_david_chosen_l3009_300945

def total_workers : ℕ := 22
def workers_to_choose : ℕ := 4

theorem probability_james_and_david_chosen :
  (Nat.choose (total_workers - 2) (workers_to_choose - 2)) / 
  (Nat.choose total_workers workers_to_choose) = 2 / 231 := by
  sorry

end probability_james_and_david_chosen_l3009_300945


namespace attached_pyramids_volume_l3009_300914

/-- A solid formed by two attached pyramids -/
structure AttachedPyramids where
  /-- Length of each edge in the square-based pyramid -/
  base_edge_length : ℝ
  /-- Total length of all edges in the resulting solid -/
  total_edge_length : ℝ

/-- The volume of the attached pyramids solid -/
noncomputable def volume (ap : AttachedPyramids) : ℝ :=
  2 * Real.sqrt 2

/-- Theorem stating the volume of the attached pyramids solid -/
theorem attached_pyramids_volume (ap : AttachedPyramids) 
  (h1 : ap.base_edge_length = 2)
  (h2 : ap.total_edge_length = 18) :
  volume ap = 2 * Real.sqrt 2 := by
  sorry

end attached_pyramids_volume_l3009_300914


namespace triangle_angle_A_l3009_300979

theorem triangle_angle_A (A B C : ℝ) (a b : ℝ) (angleB : ℝ) : 
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  angleB = π / 4 →
  (∃ (angleA : ℝ), (angleA = π / 3 ∨ angleA = 2 * π / 3) ∧ 
    a / Real.sin angleA = b / Real.sin angleB) :=
by sorry

end triangle_angle_A_l3009_300979


namespace ratio_chain_l3009_300937

theorem ratio_chain (a b c d e : ℚ) 
  (h1 : a / b = 3 / 4)
  (h2 : b / c = 7 / 9)
  (h3 : c / d = 5 / 7)
  (h4 : d / e = 11 / 13) :
  a / e = 165 / 468 := by
  sorry

end ratio_chain_l3009_300937


namespace distribute_four_balls_four_boxes_l3009_300966

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 35 ways to distribute 4 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_four_balls_four_boxes : distribute_balls 4 4 = 35 := by
  sorry

end distribute_four_balls_four_boxes_l3009_300966


namespace deck_size_proof_l3009_300958

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b : ℚ) = 1/4 →
  ((r + 6 : ℚ) / (r + b + 6 : ℚ) = 1/3) →
  r + b = 48 := by
  sorry

end deck_size_proof_l3009_300958


namespace sum_of_roots_l3009_300949

-- Define the quadratic equation
def quadratic (x p q : ℝ) : Prop := x^2 - 2*p*x + q = 0

-- Define the theorem
theorem sum_of_roots (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∃ x y : ℝ, quadratic x p q ∧ quadratic y p q ∧ x ≠ y) →
  x + y = 2 :=
by sorry

end sum_of_roots_l3009_300949


namespace sugar_water_concentration_l3009_300986

theorem sugar_water_concentration (a : ℝ) : 
  (100 * 0.4 + a * 0.2) / (100 + a) = 0.25 → a = 300 := by
  sorry

end sugar_water_concentration_l3009_300986


namespace jones_elementary_population_l3009_300905

theorem jones_elementary_population :
  ∀ (total_students : ℕ) (boys_percentage : ℚ) (sample_size : ℕ),
    boys_percentage = 60 / 100 →
    sample_size = 90 →
    (sample_size : ℚ) / boys_percentage = total_students →
    total_students = 150 := by
  sorry

end jones_elementary_population_l3009_300905


namespace enjoy_both_activities_l3009_300997

theorem enjoy_both_activities (total : ℕ) (reading : ℕ) (movies : ℕ) (neither : ℕ)
  (h1 : total = 50)
  (h2 : reading = 22)
  (h3 : movies = 20)
  (h4 : neither = 15) :
  total - neither - (reading + movies - (total - neither)) = 7 := by
  sorry

end enjoy_both_activities_l3009_300997


namespace series_sum_l3009_300975

/-- The positive real solution to x^3 + (2/5)x - 1 = 0 -/
noncomputable def r : ℝ :=
  Real.sqrt (Real.sqrt (1 + 2/5))

/-- The sum of the series r^2 + 2r^5 + 3r^8 + 4r^11 + ... -/
noncomputable def S : ℝ :=
  ∑' n, (n + 1) * r^(3*n + 2)

theorem series_sum : 
  r > 0 ∧ r^3 + 2/5 * r - 1 = 0 → S = 25/4 :=
by
  sorry

#check series_sum

end series_sum_l3009_300975


namespace max_value_sqrt_sum_l3009_300996

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 15) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) ≤ Real.sqrt 48 ∧
  ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 15 ∧
    Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) = Real.sqrt 48 :=
by sorry

end max_value_sqrt_sum_l3009_300996


namespace ant_problem_l3009_300953

/-- Represents the position of an ant on a square path -/
structure AntPosition where
  side : ℕ  -- 0: bottom, 1: right, 2: top, 3: left
  distance : ℝ  -- distance from the start of the side

/-- Represents a square path -/
structure SquarePath where
  sideLength : ℝ

/-- Represents the state of the three ants -/
structure AntState where
  mu : AntPosition
  ra : AntPosition
  vey : AntPosition

/-- Checks if the ants are aligned on a straight line -/
def areAntsAligned (state : AntState) (paths : List SquarePath) : Prop :=
  sorry

/-- Updates the positions of the ants based on the distance they've traveled -/
def updateAntPositions (initialState : AntState) (paths : List SquarePath) (distance : ℝ) : AntState :=
  sorry

theorem ant_problem (a : ℝ) :
  let paths := [⟨a⟩, ⟨a + 2⟩, ⟨a + 4⟩]
  let initialState : AntState := {
    mu := { side := 0, distance := 0 },
    ra := { side := 0, distance := 0 },
    vey := { side := 0, distance := 0 }
  }
  let finalState := updateAntPositions initialState paths ((a + 4) / 2)
  finalState.mu.side = 1 ∧
  finalState.mu.distance = 0 ∧
  finalState.ra.side = 1 ∧
  finalState.vey.side = 1 ∧
  areAntsAligned finalState paths →
  a = 4 :=
sorry

end ant_problem_l3009_300953
