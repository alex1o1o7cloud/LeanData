import Mathlib

namespace sodium_chloride_formation_l54_5433

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String

-- Define the molar quantities
def moles_NaHSO3 : ℚ := 2
def moles_HCl : ℚ := 2

-- Define the reaction
def sodium_bisulfite_reaction : Reaction :=
  { reactant1 := "NaHSO3"
  , reactant2 := "HCl"
  , product1 := "NaCl"
  , product2 := "H2O"
  , product3 := "SO2" }

-- Theorem statement
theorem sodium_chloride_formation 
  (r : Reaction) 
  (h1 : r = sodium_bisulfite_reaction) 
  (h2 : moles_NaHSO3 = moles_HCl) :
  moles_NaHSO3 = 2 → 2 = (let moles_NaCl := moles_NaHSO3; moles_NaCl) :=
by
  sorry

end sodium_chloride_formation_l54_5433


namespace point_on_graph_and_sum_l54_5495

/-- Given a function g such that g(3) = 10, prove that (1, 7.6) is on the graph of 5y = 4g(3x) - 2
    and the sum of its coordinates is 8.6 -/
theorem point_on_graph_and_sum (g : ℝ → ℝ) (h : g 3 = 10) :
  let f := fun x y => 5 * y = 4 * g (3 * x) - 2
  f 1 7.6 ∧ 1 + 7.6 = 8.6 := by
  sorry

end point_on_graph_and_sum_l54_5495


namespace inequality_proof_l54_5438

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b^2 + b / c^2 + c / a^2 ≥ 1 / a + 1 / b + 1 / c := by
  sorry

end inequality_proof_l54_5438


namespace max_cars_divided_by_ten_limit_l54_5496

/-- Represents the safety distance between cars in car lengths per 20 km/h -/
def safety_distance : ℝ := 1

/-- Represents the length of a car in meters -/
def car_length : ℝ := 4

/-- Calculates the maximum number of cars that can pass a counting device in one hour -/
noncomputable def max_cars_per_hour (m : ℝ) : ℝ :=
  (20000 * m) / (car_length * (m + 1))

/-- Theorem stating that the maximum number of cars passing in one hour divided by 10 approaches 500 -/
theorem max_cars_divided_by_ten_limit :
  ∀ ε > 0, ∃ M, ∀ m > M, |max_cars_per_hour m / 10 - 500| < ε :=
sorry

end max_cars_divided_by_ten_limit_l54_5496


namespace complex_fraction_magnitude_l54_5452

/-- Given that i is the imaginary unit, prove that |((5+3i)/(4-i))| = √2 -/
theorem complex_fraction_magnitude : 
  Complex.abs ((5 + 3 * Complex.I) / (4 - Complex.I)) = Real.sqrt 2 := by
  sorry

end complex_fraction_magnitude_l54_5452


namespace journey_equations_l54_5440

theorem journey_equations (total_time bike_speed walk_speed total_distance : ℝ)
  (h_total_time : total_time = 20)
  (h_bike_speed : bike_speed = 200)
  (h_walk_speed : walk_speed = 70)
  (h_total_distance : total_distance = 3350) :
  ∃ x y : ℝ,
    x + y = total_time ∧
    bike_speed * x + walk_speed * y = total_distance :=
by sorry

end journey_equations_l54_5440


namespace sum_a_b_equals_five_l54_5410

theorem sum_a_b_equals_five (a b : ℝ) (h1 : a + 2*b = 8) (h2 : 3*a + 4*b = 18) : a + b = 5 := by
  sorry

end sum_a_b_equals_five_l54_5410


namespace max_s_value_l54_5428

/-- Definition of the lucky number t for a given s -/
def lucky_number (s : ℕ) : ℕ :=
  let x := s / 100 - 1
  let y := (s / 10) % 10
  if y ≤ 6 then
    1000 * (x + 1) + 100 * y + 30 + y + 3
  else
    1000 * (x + 1) + 100 * y + 30 + y - 7

/-- Definition of the function F for a given lucky number N -/
def F (N : ℕ) : ℚ :=
  let ab := N / 100
  let dc := N % 100
  (ab - dc) / 3

/-- Theorem stating the maximum value of s satisfying all conditions -/
theorem max_s_value : 
  ∃ (s : ℕ), s = 913 ∧ 
  (∀ (x y : ℕ), s = 100 * x + 10 * y + 103 → 
    x ≥ y ∧ 
    y ≤ 8 ∧ 
    x ≤ 8 ∧ 
    (lucky_number s) % 17 = 5 ∧ 
    (F (lucky_number s)).den = 1) ∧
  (∀ (s' : ℕ), s' > s → 
    ¬(∃ (x' y' : ℕ), s' = 100 * x' + 10 * y' + 103 ∧ 
      x' ≥ y' ∧ 
      y' ≤ 8 ∧ 
      x' ≤ 8 ∧ 
      (lucky_number s') % 17 = 5 ∧ 
      (F (lucky_number s')).den = 1)) :=
sorry

end max_s_value_l54_5428


namespace shortest_distance_from_start_l54_5416

-- Define the walker's movements
def north_distance : ℝ := 15
def west_distance : ℝ := 8
def south_distance : ℝ := 10
def east_distance : ℝ := 1

-- Calculate net distances
def net_north : ℝ := north_distance - south_distance
def net_west : ℝ := west_distance - east_distance

-- Theorem statement
theorem shortest_distance_from_start :
  Real.sqrt (net_north ^ 2 + net_west ^ 2) = Real.sqrt 74 := by
  sorry

#check shortest_distance_from_start

end shortest_distance_from_start_l54_5416


namespace postage_fee_420g_l54_5432

/-- Calculates the postage fee for a given weight in grams -/
def postage_fee (weight : ℕ) : ℚ :=
  0.7 + 0.4 * ((weight - 1) / 100 : ℕ)

/-- The postage fee for a 420g book is 2.3 yuan -/
theorem postage_fee_420g : postage_fee 420 = 2.3 := by sorry

end postage_fee_420g_l54_5432


namespace multiply_58_62_l54_5427

theorem multiply_58_62 : 58 * 62 = 3596 := by
  sorry

end multiply_58_62_l54_5427


namespace rectangular_plot_length_difference_l54_5436

theorem rectangular_plot_length_difference (b x : ℝ) : 
  b + x = 64 →                         -- length is 64 meters
  26.5 * (2 * (b + x) + 2 * b) = 5300 →  -- cost of fencing
  x = 28 := by sorry

end rectangular_plot_length_difference_l54_5436


namespace least_nonprime_sum_l54_5424

theorem least_nonprime_sum (p : Nat) (h : Nat.Prime p) : ∃ (n : Nat), 
  (∀ (q : Nat), Nat.Prime q → ¬Nat.Prime (q^2 + n)) ∧ 
  (∀ (m : Nat), m < n → ∃ (r : Nat), Nat.Prime r ∧ Nat.Prime (r^2 + m)) :=
by
  sorry

#check least_nonprime_sum

end least_nonprime_sum_l54_5424


namespace complex_magnitude_problem_l54_5467

theorem complex_magnitude_problem (Z : ℂ) (h : (2 + Complex.I) * Z = 3 - Complex.I) :
  Complex.abs Z = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l54_5467


namespace product_of_squares_l54_5492

theorem product_of_squares (a b : ℝ) (h1 : a + b = 21) (h2 : a^2 - b^2 = 45) :
  a^2 * b^2 = 28606956 / 2401 := by
  sorry

end product_of_squares_l54_5492


namespace exactly_two_ultra_squarish_l54_5444

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A function that extracts the first three digits of a seven-digit base 9 number -/
def first_three_digits (n : ℕ) : ℕ := n / (9^4)

/-- A function that extracts the middle three digits of a seven-digit base 9 number -/
def middle_three_digits (n : ℕ) : ℕ := (n / 9^2) % (9^3)

/-- A function that extracts the last three digits of a seven-digit base 9 number -/
def last_three_digits (n : ℕ) : ℕ := n % (9^3)

/-- A function that checks if a number is ultra-squarish -/
def is_ultra_squarish (n : ℕ) : Prop :=
  n ≥ 9^6 ∧ n < 9^7 ∧  -- seven-digit number in base 9
  (∀ d, d ∈ (List.range 7).map (fun i => (n / (9^i)) % 9) → d ≠ 0) ∧  -- no digit is zero
  is_perfect_square n ∧
  is_perfect_square (first_three_digits n) ∧
  is_perfect_square (middle_three_digits n) ∧
  is_perfect_square (last_three_digits n)

/-- The theorem stating that there are exactly 2 ultra-squarish numbers -/
theorem exactly_two_ultra_squarish : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_ultra_squarish n) ∧ s.card = 2 :=
sorry

end exactly_two_ultra_squarish_l54_5444


namespace work_completion_time_l54_5445

/-- Represents the work rate of a group of workers -/
def WorkRate (num_workers : ℕ) (days : ℕ) : ℚ :=
  1 / (num_workers * days)

/-- The theorem statement -/
theorem work_completion_time 
  (men_rate : WorkRate 8 20 = WorkRate 12 20) 
  (total_work : ℚ := 1) :
  (6 : ℚ) * WorkRate 8 20 + (11 : ℚ) * WorkRate 12 20 = 1 / 12 := by
  sorry

end work_completion_time_l54_5445


namespace tan_105_degrees_l54_5434

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l54_5434


namespace perpendicular_planes_from_lines_l54_5481

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  parallel m n →
  subset m α →
  perpendicular n β →
  perpendicular_planes α β :=
sorry

end perpendicular_planes_from_lines_l54_5481


namespace albany_syracuse_distance_l54_5446

/-- The distance between Albany and Syracuse satisfies the equation relating to travel times. -/
theorem albany_syracuse_distance (D : ℝ) : D > 0 → D / 50 + D / 38.71 = 5.5 := by
  sorry

end albany_syracuse_distance_l54_5446


namespace apple_production_theorem_l54_5476

/-- Apple production over three years -/
def AppleProduction : Prop :=
  let first_year : ℕ := 40
  let second_year : ℕ := 8 + 2 * first_year
  let third_year : ℕ := second_year - (second_year / 4)
  let total : ℕ := first_year + second_year + third_year
  total = 194

theorem apple_production_theorem : AppleProduction := by
  sorry

end apple_production_theorem_l54_5476


namespace percent_of_percent_l54_5454

theorem percent_of_percent (x : ℝ) : (0.3 * (0.6 * x)) = (0.18 * x) := by
  sorry

end percent_of_percent_l54_5454


namespace kwik_e_tax_center_problem_l54_5494

/-- The Kwik-e-Tax Center problem -/
theorem kwik_e_tax_center_problem (federal_price state_price quarterly_price : ℕ) 
  (state_count quarterly_count total_revenue : ℕ) 
  (h1 : federal_price = 50)
  (h2 : state_price = 30)
  (h3 : quarterly_price = 80)
  (h4 : state_count = 20)
  (h5 : quarterly_count = 10)
  (h6 : total_revenue = 4400) :
  ∃ (federal_count : ℕ), 
    federal_count = 60 ∧ 
    federal_price * federal_count + state_price * state_count + quarterly_price * quarterly_count = total_revenue :=
by
  sorry


end kwik_e_tax_center_problem_l54_5494


namespace planes_distance_l54_5402

/-- Represents a plane in 3D space defined by the equation ax + by + cz + d = 0 -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the distance between two parallel planes -/
def distance_between_planes (p1 p2 : Plane) : ℝ :=
  sorry

/-- The two planes in the problem -/
def plane1 : Plane := ⟨3, -1, 2, -4⟩
def plane2 : Plane := ⟨6, -2, 4, 3⟩

theorem planes_distance :
  distance_between_planes plane1 plane2 = 11 * Real.sqrt 14 / 28 := by
  sorry

end planes_distance_l54_5402


namespace min_value_sum_reciprocals_l54_5412

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_9 : a + b + c + d + e + f = 9) :
  1/a + 2/b + 9/c + 8/d + 18/e + 32/f ≥ 24 ∧ 
  ∃ (a' b' c' d' e' f' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 0 < f' ∧
    a' + b' + c' + d' + e' + f' = 9 ∧
    1/a' + 2/b' + 9/c' + 8/d' + 18/e' + 32/f' = 24 :=
by sorry

end min_value_sum_reciprocals_l54_5412


namespace paul_work_days_l54_5450

/-- The number of days it takes Rose to complete the work -/
def rose_days : ℝ := 120

/-- The number of days it takes Paul and Rose together to complete the work -/
def combined_days : ℝ := 48

/-- The number of days it takes Paul to complete the work alone -/
def paul_days : ℝ := 80

/-- Theorem stating that given Rose's and combined work rates, Paul's individual work rate can be determined -/
theorem paul_work_days (rose : ℝ) (combined : ℝ) (paul : ℝ) 
  (h_rose : rose = rose_days) 
  (h_combined : combined = combined_days) :
  paul = paul_days :=
by sorry

end paul_work_days_l54_5450


namespace four_boxes_volume_l54_5423

/-- The volume of a cube with edge length s -/
def cube_volume (s : ℝ) : ℝ := s ^ 3

/-- The total volume of n identical cubes with edge length s -/
def total_volume (n : ℕ) (s : ℝ) : ℝ := n * cube_volume s

/-- Theorem: The total volume of four cubic boxes, each with an edge length of 5 meters, is 500 cubic meters -/
theorem four_boxes_volume : total_volume 4 5 = 500 := by
  sorry

end four_boxes_volume_l54_5423


namespace intersection_A_complement_B_l54_5473

open Set

-- Define the universal set
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 1}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = Ioo 0 1 := by sorry

end intersection_A_complement_B_l54_5473


namespace vector_operation_l54_5474

/-- Given two plane vectors a and b, prove that -2a - b equals the specified result. -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (1, -1)) :
  (-2 : ℝ) • a - b = (-3, -1) := by
  sorry

end vector_operation_l54_5474


namespace fraction_equality_l54_5443

theorem fraction_equality (b : ℕ+) : 
  (b : ℚ) / (b + 15 : ℚ) = 3/4 → b = 45 := by
  sorry

end fraction_equality_l54_5443


namespace smallest_n_for_integer_S_l54_5403

def K : ℚ := (1 : ℚ) + (1/2 : ℚ) + (1/3 : ℚ) + (1/4 : ℚ)

def S (n : ℕ) : ℚ := n * (5^(n-1) : ℚ) * K

def is_integer (q : ℚ) : Prop := ∃ (z : ℤ), q = z

theorem smallest_n_for_integer_S :
  ∀ n : ℕ, (n > 0 ∧ is_integer (S n)) → n ≥ 24 :=
sorry

end smallest_n_for_integer_S_l54_5403


namespace square_side_length_relation_l54_5482

theorem square_side_length_relation (s L : ℝ) (h1 : s > 0) (h2 : L > 0) : 
  (4 * L) / (4 * s) = 2.5 → (L * Real.sqrt 2) / (s * Real.sqrt 2) = 2.5 → L = 2.5 * s := by
  sorry

end square_side_length_relation_l54_5482


namespace moving_circle_trajectory_l54_5488

/-- Two fixed circles in a 2D plane -/
structure FixedCircles where
  C₁ : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x + 4)^2 + y^2 = 2
  C₂ : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - 4)^2 + y^2 = 2

/-- A moving circle tangent to both fixed circles -/
structure MovingCircle (fc : FixedCircles) where
  center : ℝ × ℝ
  isTangentToC₁ : Prop
  isTangentToC₂ : Prop

/-- The trajectory of the center of the moving circle -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 14 = 1 ∨ x = 0

/-- Theorem stating that the trajectory of the moving circle's center
    is described by the given equation -/
theorem moving_circle_trajectory (fc : FixedCircles) :
  ∀ (mc : MovingCircle fc), trajectory mc.center.1 mc.center.2 :=
sorry

end moving_circle_trajectory_l54_5488


namespace income_calculation_l54_5437

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 7 = expenditure * 8 →
  income = expenditure + savings →
  savings = 5000 →
  income = 40000 := by
sorry

end income_calculation_l54_5437


namespace tower_levels_l54_5466

theorem tower_levels (steps_per_level : ℕ) (blocks_per_step : ℕ) (total_blocks : ℕ) :
  steps_per_level = 8 →
  blocks_per_step = 3 →
  total_blocks = 96 →
  total_blocks / (steps_per_level * blocks_per_step) = 4 :=
by sorry

end tower_levels_l54_5466


namespace range_of_abc_l54_5472

theorem range_of_abc (a b c : ℝ) 
  (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) 
  (h4 : 2 < c) (h5 : c < 3) :
  ∀ x, (∃ (a' b' c' : ℝ), 
    -1 < a' ∧ a' < b' ∧ b' < 1 ∧ 
    2 < c' ∧ c' < 3 ∧ 
    x = (a' - b') * c') ↔ 
  -6 < x ∧ x < 0 :=
by sorry

end range_of_abc_l54_5472


namespace notched_circle_distance_l54_5493

theorem notched_circle_distance (r AB BC : ℝ) (h_r : r = Real.sqrt 75) 
  (h_AB : AB = 8) (h_BC : BC = 3) : ∃ (x y : ℝ), x^2 + y^2 = 65 ∧ 
  (x + AB)^2 + y^2 = r^2 ∧ x^2 + (y + BC)^2 = r^2 := by
  sorry

end notched_circle_distance_l54_5493


namespace right_triangle_third_side_l54_5470

theorem right_triangle_third_side 
  (x y z : ℝ) 
  (h_right_triangle : x^2 + y^2 = z^2) 
  (h_equation : |x - 3| + Real.sqrt (2 * y - 8) = 0) : 
  z = Real.sqrt 7 ∨ z = 5 := by
sorry

end right_triangle_third_side_l54_5470


namespace volume_of_five_adjacent_cubes_l54_5414

/-- The volume of a solid formed by placing n equal cubes with side length s adjacent to each other -/
def volume_of_adjacent_cubes (n : ℕ) (s : ℝ) : ℝ := n * s^3

/-- Theorem: The volume of a solid formed by placing five equal cubes with side length 5 cm adjacent to each other is 625 cm³ -/
theorem volume_of_five_adjacent_cubes :
  volume_of_adjacent_cubes 5 5 = 625 := by
  sorry

end volume_of_five_adjacent_cubes_l54_5414


namespace helmet_sales_theorem_l54_5455

/-- Represents the monthly sales data and pricing information for helmets --/
structure HelmetSalesData where
  april_sales : ℕ
  june_sales : ℕ
  cost_price : ℕ
  reference_price : ℕ
  reference_volume : ℕ
  volume_change_rate : ℕ
  target_profit : ℕ

/-- Calculates the monthly growth rate given the sales data --/
def calculate_growth_rate (data : HelmetSalesData) : ℚ :=
  sorry

/-- Calculates the optimal selling price given the sales data --/
def calculate_optimal_price (data : HelmetSalesData) : ℕ :=
  sorry

/-- Theorem stating the correct growth rate and optimal price --/
theorem helmet_sales_theorem (data : HelmetSalesData) 
  (h1 : data.april_sales = 150)
  (h2 : data.june_sales = 216)
  (h3 : data.cost_price = 30)
  (h4 : data.reference_price = 40)
  (h5 : data.reference_volume = 600)
  (h6 : data.volume_change_rate = 10)
  (h7 : data.target_profit = 10000) :
  calculate_growth_rate data = 1/5 ∧ calculate_optimal_price data = 50 := by
  sorry

end helmet_sales_theorem_l54_5455


namespace isosceles_triangle_perimeter_l54_5429

/-- An isosceles triangle with side lengths 3 and 4 has a perimeter of either 10 or 11 -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  (a = 3 ∨ a = 4) → 
  (b = 3 ∨ b = 4) → 
  (c = 3 ∨ c = 4) →
  (a = b ∨ b = c ∨ a = c) → 
  (a + b > c ∧ b + c > a ∧ a + c > b) →
  (a + b + c = 10 ∨ a + b + c = 11) :=
by sorry

end isosceles_triangle_perimeter_l54_5429


namespace max_coincident_area_folded_triangle_l54_5479

theorem max_coincident_area_folded_triangle :
  let a := 3 / 2
  let b := Real.sqrt 5 / 2
  let c := Real.sqrt 2
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let height := 2 * area / a
  let max_coincident_area := area + (1 / (2 * height)) - (1 / (4 * height^2)) - (3 / (4 * height^2))
  max_coincident_area = 9 / 28 := by sorry

end max_coincident_area_folded_triangle_l54_5479


namespace work_completion_time_l54_5499

/-- Given that two workers A and B can complete a work together in 16 days,
    and A alone can complete the work in 24 days, prove that B alone will
    complete the work in 48 days. -/
theorem work_completion_time
  (joint_time : ℝ) (a_time : ℝ) (b_time : ℝ)
  (h1 : joint_time = 16)
  (h2 : a_time = 24)
  (h3 : (1 / joint_time) = (1 / a_time) + (1 / b_time)) :
  b_time = 48 := by
  sorry

end work_completion_time_l54_5499


namespace negation_of_universal_proposition_l54_5465

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) :=
by sorry

end negation_of_universal_proposition_l54_5465


namespace evaluate_expression_l54_5407

theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = 0) : z * (2 * z - 5 * x) = 0 := by
  sorry

end evaluate_expression_l54_5407


namespace washer_dryer_total_cost_l54_5453

/-- The cost of a washer-dryer combination -/
def washer_dryer_cost (dryer_cost washer_cost_difference : ℕ) : ℕ :=
  dryer_cost + (dryer_cost + washer_cost_difference)

/-- Theorem: The washer-dryer combination costs $1200 -/
theorem washer_dryer_total_cost : 
  washer_dryer_cost 490 220 = 1200 := by
  sorry

end washer_dryer_total_cost_l54_5453


namespace workshop_average_salary_l54_5457

theorem workshop_average_salary 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (technician_salary : ℕ) 
  (other_salary : ℕ) 
  (h1 : total_workers = 18) 
  (h2 : num_technicians = 6) 
  (h3 : technician_salary = 12000) 
  (h4 : other_salary = 6000) :
  (num_technicians * technician_salary + (total_workers - num_technicians) * other_salary) / total_workers = 8000 :=
by sorry

end workshop_average_salary_l54_5457


namespace fifth_group_students_l54_5490

theorem fifth_group_students (total : Nat) (group1 group2 group3 group4 : Nat)
  (h1 : total = 40)
  (h2 : group1 = 6)
  (h3 : group2 = 9)
  (h4 : group3 = 8)
  (h5 : group4 = 7) :
  total - (group1 + group2 + group3 + group4) = 10 := by
  sorry

end fifth_group_students_l54_5490


namespace cube_preserves_order_l54_5462

theorem cube_preserves_order (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_preserves_order_l54_5462


namespace davids_initial_money_l54_5426

/-- 
Given that David has $800 less than he spent after spending money on a trip,
and he now has $500 left, prove that he had $1800 at the beginning of his trip.
-/
theorem davids_initial_money :
  ∀ (initial_money spent_money remaining_money : ℕ),
  remaining_money = spent_money - 800 →
  remaining_money = 500 →
  initial_money = spent_money + remaining_money →
  initial_money = 1800 :=
by
  sorry

end davids_initial_money_l54_5426


namespace sequence_general_term_l54_5483

theorem sequence_general_term (a : ℕ+ → ℚ) :
  a 1 = 1 ∧
  (∀ n : ℕ+, a (n + 1) = (2 * a n) / (2 + a n)) →
  ∀ n : ℕ+, a n = 2 / (n + 1) := by
sorry

end sequence_general_term_l54_5483


namespace parallel_vectors_x_value_l54_5406

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x + 1, -2)
  let b : ℝ × ℝ := (-2*x, 3)
  parallel a b → x = 3 := by
sorry

end parallel_vectors_x_value_l54_5406


namespace wendy_cupcakes_l54_5425

/-- Represents the number of pastries in Wendy's bake sale scenario -/
structure BakeSale where
  cupcakes : ℕ
  cookies : ℕ
  pastries_left : ℕ
  pastries_sold : ℕ

/-- The theorem stating the number of cupcakes Wendy baked -/
theorem wendy_cupcakes (b : BakeSale) 
  (h1 : b.cupcakes + b.cookies = b.pastries_left + b.pastries_sold)
  (h2 : b.cookies = 29)
  (h3 : b.pastries_left = 24)
  (h4 : b.pastries_sold = 9) :
  b.cupcakes = 4 := by
  sorry

end wendy_cupcakes_l54_5425


namespace tenth_power_sum_of_roots_l54_5411

theorem tenth_power_sum_of_roots (r s : ℂ) : 
  (r^2 - 2*r + 4 = 0) → (s^2 - 2*s + 4 = 0) → r^10 + s^10 = 1024 := by
  sorry

end tenth_power_sum_of_roots_l54_5411


namespace square_area_is_9_l54_5405

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The square ABCD -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The square is entirely below the x-axis -/
def below_x_axis (s : Square) : Prop :=
  s.A.2 ≤ 0 ∧ s.B.2 ≤ 0 ∧ s.C.2 ≤ 0 ∧ s.D.2 ≤ 0

/-- The square is inscribed within the region bounded by the parabola and the x-axis -/
def inscribed_in_parabola (s : Square) : Prop :=
  s.A.2 = 0 ∧ s.B.2 = 0 ∧ s.C.2 = f s.C.1 ∧ s.D.2 = f s.D.1

/-- The top vertex A lies at (2, 0) -/
def top_vertex_at_2_0 (s : Square) : Prop :=
  s.A = (2, 0)

/-- The theorem stating that the area of the square is 9 -/
theorem square_area_is_9 (s : Square) 
    (h1 : below_x_axis s)
    (h2 : inscribed_in_parabola s)
    (h3 : top_vertex_at_2_0 s) : 
  (s.B.1 - s.A.1)^2 = 9 := by
  sorry

end square_area_is_9_l54_5405


namespace negation_equivalence_l54_5439

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (planet : U → Prop)
variable (orbits_sun : U → Prop)

-- Define the original statement
def every_planet_orbits_sun : Prop := ∀ x, planet x → orbits_sun x

-- Define the negation we want to prove
def some_planets_dont_orbit_sun : Prop := ∃ x, planet x ∧ ¬(orbits_sun x)

-- Theorem statement
theorem negation_equivalence : 
  ¬(every_planet_orbits_sun U planet orbits_sun) ↔ some_planets_dont_orbit_sun U planet orbits_sun :=
by sorry

end negation_equivalence_l54_5439


namespace percentage_less_relation_l54_5497

/-- Given three real numbers A, B, and C, where A is 35% less than C,
    and B is 10.76923076923077% less than A, prove that B is
    approximately 42% less than C. -/
theorem percentage_less_relation (A B C : ℝ) 
  (h1 : A = 0.65 * C)  -- A is 35% less than C
  (h2 : B = 0.8923076923076923 * A)  -- B is 10.76923076923077% less than A
  : ∃ (ε : ℝ), abs (B - 0.58 * C) < ε ∧ ε < 0.0001 := by
  sorry

end percentage_less_relation_l54_5497


namespace trig_inequality_l54_5464

theorem trig_inequality : 
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c := by sorry

end trig_inequality_l54_5464


namespace equal_connections_implies_square_l54_5441

/-- Represents the coloring of vertices in a regular n-gon --/
structure VertexColoring (n : ℕ) where
  red : ℕ
  blue : ℕ
  sum_eq_n : red + blue = n

/-- Condition for equal number of same-colored and different-colored connections --/
def equal_connections (n : ℕ) (c : VertexColoring n) : Prop :=
  (c.red.choose 2) + (c.blue.choose 2) = c.red * c.blue

/-- Theorem stating that if equal_connections holds, then n is a perfect square --/
theorem equal_connections_implies_square (n : ℕ) (c : VertexColoring n) 
  (h : equal_connections n c) : ∃ k : ℕ, n = k^2 := by
  sorry

end equal_connections_implies_square_l54_5441


namespace geometric_sequence_property_l54_5430

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem stating the relationship between a₄, a₇, and a₁₀ in a geometric sequence -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 4 * a 10 = 9 → a 7 = 3 ∨ a 7 = -3 := by
  sorry

end geometric_sequence_property_l54_5430


namespace exists_solution_infinitely_many_solutions_divisibility_property_no_extended_solution_l54_5456

-- Define the condition (*) as a predicate
def condition_star (k a b c : ℕ) : Prop :=
  a^2 + k^2 = b^2 + (k+1)^2 ∧ b^2 + (k+1)^2 = c^2 + (k+2)^2

-- Part (a): Existence of a solution
theorem exists_solution : ∃ k a b c : ℕ, condition_star k a b c :=
sorry

-- Part (b): Infinitely many solutions
theorem infinitely_many_solutions : ∀ n : ℕ, ∃ k a b c : ℕ, k > n ∧ condition_star k a b c :=
sorry

-- Part (c): Divisibility property
theorem divisibility_property : ∀ k a b c : ℕ, condition_star k a b c → 144 ∣ (a * b * c) :=
sorry

-- Part (d): Non-existence of extended solution
def extended_condition (k a b c d : ℕ) : Prop :=
  a^2 + k^2 = b^2 + (k+1)^2 ∧ b^2 + (k+1)^2 = c^2 + (k+2)^2 ∧ c^2 + (k+2)^2 = d^2 + (k+3)^2

theorem no_extended_solution : ¬∃ k a b c d : ℕ, extended_condition k a b c d :=
sorry

end exists_solution_infinitely_many_solutions_divisibility_property_no_extended_solution_l54_5456


namespace common_solution_for_all_a_l54_5413

/-- The linear equation (a-3)x + (2a-5)y + 6-a = 0 has a common solution (7, -3) for all values of a. -/
theorem common_solution_for_all_a :
  ∀ (a : ℝ), (a - 3) * 7 + (2 * a - 5) * (-3) + 6 - a = 0 := by
  sorry

end common_solution_for_all_a_l54_5413


namespace luck_represents_6789_l54_5451

/-- Represents a mapping from characters to digits -/
def DigitMapping := Char → Nat

/-- The 12-letter code -/
def code : String := "AMAZING LUCK"

/-- The condition that the code represents digits 0-9 and repeats for two more digits -/
def valid_mapping (m : DigitMapping) : Prop :=
  ∀ i : Fin 12, 
    m (code.get ⟨i⟩) = if i < 10 then i else i - 10

/-- The substring we're interested in -/
def substring : String := "LUCK"

/-- The theorem to prove -/
theorem luck_represents_6789 (m : DigitMapping) (h : valid_mapping m) : 
  (m 'L', m 'U', m 'C', m 'K') = (6, 7, 8, 9) := by
  sorry

end luck_represents_6789_l54_5451


namespace craig_apples_l54_5469

/-- The number of apples Craig has after receiving more from Eugene -/
def total_apples (initial : Real) (received : Real) : Real :=
  initial + received

/-- Proof that Craig will have 27.0 apples -/
theorem craig_apples : total_apples 20.0 7.0 = 27.0 := by
  sorry

end craig_apples_l54_5469


namespace tangent_line_minimum_sum_l54_5422

/-- Given a circle and a line that are tangent, prove the minimum value of a + b -/
theorem tangent_line_minimum_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → (a - 1) * x + (b - 1) * y + a + b = 0) →
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (a - 1) * x + (b - 1) * y + a + b = 0) →
  a + b ≥ 2 * Real.sqrt 2 - 2 :=
by sorry

end tangent_line_minimum_sum_l54_5422


namespace max_abs_z3_l54_5448

theorem max_abs_z3 (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ ≤ 1)
  (h2 : Complex.abs z₂ ≤ 1)
  (h3 : Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂)) :
  Complex.abs z₃ ≤ Real.sqrt 2 ∧ 
  ∃ w₁ w₂ w₃ : ℂ, Complex.abs w₁ ≤ 1 ∧ 
               Complex.abs w₂ ≤ 1 ∧ 
               Complex.abs (2 * w₃ - (w₁ + w₂)) ≤ Complex.abs (w₁ - w₂) ∧
               Complex.abs w₃ = Real.sqrt 2 :=
by sorry

end max_abs_z3_l54_5448


namespace shoe_box_problem_l54_5475

theorem shoe_box_problem (num_pairs : ℕ) (prob_match : ℚ) :
  num_pairs = 6 →
  prob_match = 1 / 11 →
  (num_pairs * 2 : ℕ) = 12 :=
by sorry

end shoe_box_problem_l54_5475


namespace f_is_odd_and_increasing_l54_5491

-- Define the function
def f (x : ℝ) : ℝ := 3 * x

-- State the theorem
theorem f_is_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ a b, a < b → f a < f b) :=
sorry

end f_is_odd_and_increasing_l54_5491


namespace last_digit_of_nine_power_l54_5468

theorem last_digit_of_nine_power (n : ℕ) : 9^(9^8) % 10 = 1 := by
  sorry

end last_digit_of_nine_power_l54_5468


namespace script_year_proof_l54_5480

theorem script_year_proof : ∃! (year : ℕ), 
  year < 200 ∧ year^13 = 258145266804692077858261512663 :=
by
  -- The proof goes here
  sorry

end script_year_proof_l54_5480


namespace gain_percent_for_equal_cost_and_selling_l54_5447

/-- Given that the cost price of 50 articles equals the selling price of 30 articles,
    prove that the gain percent is 200/3. -/
theorem gain_percent_for_equal_cost_and_selling (C S : ℝ) 
  (h : 50 * C = 30 * S) : 
  (S - C) / C * 100 = 200 / 3 := by
  sorry

end gain_percent_for_equal_cost_and_selling_l54_5447


namespace hours_to_weeks_l54_5435

/-- Proves that 2016 hours is equivalent to 12 weeks -/
theorem hours_to_weeks : 
  (∀ (week : ℕ) (day : ℕ) (hour : ℕ), 
    (1 : ℕ) * week = 7 * day ∧ 
    (1 : ℕ) * day = 24 * hour) → 
  2016 = 12 * (7 * 24) :=
by sorry

end hours_to_weeks_l54_5435


namespace smallest_prime_longest_sequence_l54_5431

def A₁₁ : ℕ := 30

def is_prime_sequence (p : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → Nat.Prime (p + k * A₁₁)

theorem smallest_prime_longest_sequence :
  ∃ n : ℕ, 
    Nat.Prime 7 ∧ 
    is_prime_sequence 7 n ∧
    ∀ p < 7, Nat.Prime p → ∀ m : ℕ, is_prime_sequence p m → m ≤ n :=
by sorry

end smallest_prime_longest_sequence_l54_5431


namespace rational_function_simplification_l54_5498

theorem rational_function_simplification (x : ℝ) (h : x ≠ -1) :
  (x^3 + 4*x^2 + 5*x + 2) / (x + 1) = x^2 + 3*x + 2 := by
  sorry

end rational_function_simplification_l54_5498


namespace product_max_min_two_digit_l54_5409

def max_two_digit : ℕ := 99
def min_two_digit : ℕ := 10

theorem product_max_min_two_digit : max_two_digit * min_two_digit = 990 := by
  sorry

end product_max_min_two_digit_l54_5409


namespace min_value_of_f_in_interval_l54_5449

-- Define the function f(x) = x^3 - 12x
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the interval [-4, 4]
def interval : Set ℝ := Set.Icc (-4) 4

-- Theorem statement
theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), x ∈ interval ∧ f x = -16 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
sorry

end min_value_of_f_in_interval_l54_5449


namespace equation_solution_range_l54_5487

theorem equation_solution_range : 
  {k : ℝ | ∃ x : ℝ, 2*k*(Real.sin x) = 1 + k^2} = {-1, 1} := by sorry

end equation_solution_range_l54_5487


namespace long_jump_distance_difference_long_jump_distance_difference_holds_l54_5486

/-- Proves that Margarita ran and jumped 1 foot farther than Ricciana -/
theorem long_jump_distance_difference : ℕ → Prop :=
  fun margarita_total =>
    let ricciana_run := 20
    let ricciana_jump := 4
    let ricciana_total := ricciana_run + ricciana_jump
    let margarita_run := 18
    let margarita_jump := 2 * ricciana_jump - 1
    margarita_total = margarita_run + margarita_jump ∧
    margarita_total - ricciana_total = 1

/-- The theorem holds for Margarita's total distance of 25 feet -/
theorem long_jump_distance_difference_holds : long_jump_distance_difference 25 := by
  sorry

end long_jump_distance_difference_long_jump_distance_difference_holds_l54_5486


namespace scientific_notation_of_43000000_l54_5404

theorem scientific_notation_of_43000000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 43000000 = a * (10 : ℝ) ^ n :=
by
  -- The proof would go here
  sorry

end scientific_notation_of_43000000_l54_5404


namespace mixture_replacement_l54_5401

/-- Given a mixture of liquids A and B, this theorem proves that
    replacing a certain amount of the mixture with liquid B
    results in the specified final ratio. -/
theorem mixture_replacement (initial_a initial_b replacement : ℚ) :
  initial_a = 16 →
  initial_b = 4 →
  (initial_a - 4/5 * replacement) / (initial_b + 4/5 * replacement) = 2/3 →
  replacement = 10 := by
  sorry

end mixture_replacement_l54_5401


namespace min_box_value_l54_5418

theorem min_box_value (a b Box : ℤ) : 
  (∀ x, (a*x + b)*(b*x + a) = 36*x^2 + Box*x + 36) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  Box = a^2 + b^2 →
  ∃ (min_Box : ℤ), (∀ Box', (∃ a' b' : ℤ, 
    (∀ x, (a'*x + b')*(b'*x + a') = 36*x^2 + Box'*x + 36) ∧
    a' ≠ b' ∧ b' ≠ Box' ∧ a' ≠ Box' ∧
    Box' = a'^2 + b'^2) → 
    min_Box ≤ Box') ∧
  min_Box = 72 :=
sorry

end min_box_value_l54_5418


namespace sequence_third_term_l54_5421

theorem sequence_third_term (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n : ℕ, S n = 2 - 2^(n+1)) → 
  a 3 = -8 := by
sorry

end sequence_third_term_l54_5421


namespace total_chickens_on_farm_l54_5489

/-- Proves that the total number of chickens on a farm is 120, given the number of hens and their relation to roosters. -/
theorem total_chickens_on_farm (num_hens : ℕ) (num_roosters : ℕ) : 
  num_hens = 52 → 
  num_hens + 16 = num_roosters → 
  num_hens + num_roosters = 120 := by
  sorry

end total_chickens_on_farm_l54_5489


namespace road_trip_duration_l54_5461

theorem road_trip_duration 
  (initial_duration : ℕ) 
  (stretch_interval : ℕ) 
  (food_stops : ℕ) 
  (gas_stops : ℕ) 
  (stop_duration : ℕ) 
  (h1 : initial_duration = 14)
  (h2 : stretch_interval = 2)
  (h3 : food_stops = 2)
  (h4 : gas_stops = 3)
  (h5 : stop_duration = 20) :
  initial_duration + 
  (initial_duration / stretch_interval + food_stops + gas_stops) * stop_duration / 60 = 18 := by
  sorry

end road_trip_duration_l54_5461


namespace minas_numbers_l54_5442

theorem minas_numbers (x y : ℤ) (h1 : 3 * x + 4 * y = 135) (h2 : x = 15 ∨ y = 15) : x = 25 ∨ y = 25 :=
sorry

end minas_numbers_l54_5442


namespace parabola_line_intersection_l54_5485

/-- Proves that a parabola and a line intersect at two specific points -/
theorem parabola_line_intersection :
  let parabola (x : ℝ) := 2 * x^2 - 8 * x + 10
  let line (x : ℝ) := x + 1
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    parabola x₁ = line x₁ ∧
    parabola x₂ = line x₂ ∧
    ((x₁ = 3 ∧ parabola x₁ = 4) ∨ (x₁ = 3/2 ∧ parabola x₁ = 5/2)) ∧
    ((x₂ = 3 ∧ parabola x₂ = 4) ∨ (x₂ = 3/2 ∧ parabola x₂ = 5/2)) :=
by sorry

end parabola_line_intersection_l54_5485


namespace traditionalist_fraction_l54_5463

theorem traditionalist_fraction (num_provinces : ℕ) (num_traditionalists_per_province : ℚ) (num_progressives : ℚ) :
  num_provinces = 5 →
  num_traditionalists_per_province = num_progressives / 15 →
  (num_provinces * num_traditionalists_per_province) / (num_progressives + num_provinces * num_traditionalists_per_province) = 1/4 := by
  sorry

end traditionalist_fraction_l54_5463


namespace quadratic_roots_condition_l54_5460

def is_necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem quadratic_roots_condition :
  ∀ m n : ℝ,
  let roots := {x : ℝ | x^2 - m*x + n = 0}
  is_necessary_not_sufficient
    (m > 2 ∧ n > 1)
    (∀ x ∈ roots, x > 1) :=
by sorry

end quadratic_roots_condition_l54_5460


namespace triangle_properties_l54_5420

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the triangle properties -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 2 * t.b) 
  (h2 : 2 * Real.sin t.A = 3 * Real.sin (2 * t.C)) 
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 7) / 2) : 
  t.a = (3 * Real.sqrt 2 / 2) * t.b ∧ 
  (t.c * ((3 * Real.sqrt 7) / 4)) / (2 * ((3 * Real.sqrt 7) / 2)) = (3 * Real.sqrt 7) / 4 := by
  sorry

#check triangle_properties

end triangle_properties_l54_5420


namespace julia_tag_tuesday_l54_5417

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := 7

/-- The total number of kids Julia played tag with -/
def total_kids : ℕ := 20

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := total_kids - monday_kids

theorem julia_tag_tuesday : tuesday_kids = 13 := by
  sorry

end julia_tag_tuesday_l54_5417


namespace final_share_is_132_75_l54_5458

/-- Calculates the final amount each person receives after combining and sharing their updated amounts equally. -/
def final_share (emani_initial : ℚ) (howard_difference : ℚ) (jamal_initial : ℚ) 
                (emani_increase : ℚ) (howard_increase : ℚ) (jamal_increase : ℚ) : ℚ :=
  let howard_initial := emani_initial - howard_difference
  let emani_updated := emani_initial * (1 + emani_increase)
  let howard_updated := howard_initial * (1 + howard_increase)
  let jamal_updated := jamal_initial * (1 + jamal_increase)
  let total_updated := emani_updated + howard_updated + jamal_updated
  total_updated / 3

/-- Theorem stating that each person receives $132.75 after combining and sharing their updated amounts equally. -/
theorem final_share_is_132_75 :
  final_share 150 30 75 (20/100) (10/100) (15/100) = 132.75 := by
  sorry

end final_share_is_132_75_l54_5458


namespace class_artworks_count_l54_5408

/-- Represents the number of artworks created by a group of students -/
structure Artworks :=
  (paintings : ℕ)
  (drawings : ℕ)
  (sculptures : ℕ)

/-- Calculates the total number of artworks -/
def total_artworks (a : Artworks) : ℕ :=
  a.paintings + a.drawings + a.sculptures

theorem class_artworks_count :
  let total_students : ℕ := 36
  let group1_students : ℕ := 24
  let group2_students : ℕ := 12
  let total_kits : ℕ := 48
  let group1_sharing_ratio : ℕ := 3  -- 1 kit per 3 students
  let group2_sharing_ratio : ℕ := 2  -- 1 kit per 2 students
  
  let group1_first_half : Artworks := ⟨2, 4, 1⟩
  let group1_second_half : Artworks := ⟨1, 5, 3⟩
  let group2_first_third : Artworks := ⟨3, 6, 3⟩
  let group2_second_third : Artworks := ⟨4, 7, 1⟩
  
  let group1_artworks : Artworks := ⟨
    12 * group1_first_half.paintings + 12 * group1_second_half.paintings,
    12 * group1_first_half.drawings + 12 * group1_second_half.drawings,
    12 * group1_first_half.sculptures + 12 * group1_second_half.sculptures
  ⟩
  
  let group2_artworks : Artworks := ⟨
    4 * group2_first_third.paintings + 8 * group2_second_third.paintings,
    4 * group2_first_third.drawings + 8 * group2_second_third.drawings,
    4 * group2_first_third.sculptures + 8 * group2_second_third.sculptures
  ⟩
  
  let total_class_artworks : Artworks := ⟨
    group1_artworks.paintings + group2_artworks.paintings,
    group1_artworks.drawings + group2_artworks.drawings,
    group1_artworks.sculptures + group2_artworks.sculptures
  ⟩
  
  total_artworks total_class_artworks = 336 := by sorry

end class_artworks_count_l54_5408


namespace regular_polygon_perimeter_l54_5459

/-- A regular polygon with side length 5 units and exterior angle 120 degrees has a perimeter of 15 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n ≥ 3 →
  side_length = 5 →
  exterior_angle = 120 →
  (360 : ℝ) / n = exterior_angle →
  n * side_length = 15 :=
by sorry

end regular_polygon_perimeter_l54_5459


namespace floor_equation_solution_l54_5471

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋) ↔ (5/3 ≤ x ∧ x < 7/3) := by
sorry

end floor_equation_solution_l54_5471


namespace sqrt_five_squared_times_seven_fourth_l54_5415

theorem sqrt_five_squared_times_seven_fourth (x : ℝ) : 
  x = Real.sqrt (5^2 * 7^4) → x = 245 := by
  sorry

end sqrt_five_squared_times_seven_fourth_l54_5415


namespace roots_sum_properties_l54_5478

theorem roots_sum_properties (a : ℤ) (x₁ x₂ : ℝ) (h_odd : Odd a) (h_roots : x₁^2 + a*x₁ - 1 = 0 ∧ x₂^2 + a*x₂ - 1 = 0) :
  ∀ n : ℕ, 
    (∃ k : ℤ, x₁^n + x₂^n = k) ∧ 
    (∃ m : ℤ, x₁^(n+1) + x₂^(n+1) = m) ∧ 
    (Int.gcd (↑⌊x₁^n + x₂^n⌋) (↑⌊x₁^(n+1) + x₂^(n+1)⌋) = 1) :=
by sorry

end roots_sum_properties_l54_5478


namespace geometric_arithmetic_progression_l54_5477

theorem geometric_arithmetic_progression (b : ℝ) (q : ℝ) :
  b > 0 ∧ q > 1 →
  (∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (b * q ^ i.val + b * q ^ k.val) / 2 = b * q ^ j.val) →
  q = (1 + Real.sqrt 5) / 2 :=
sorry

end geometric_arithmetic_progression_l54_5477


namespace sample_size_determination_l54_5484

theorem sample_size_determination (total_population : Nat) (n : Nat) : 
  total_population = 36 →
  n > 0 →
  total_population % n = 0 →
  (total_population / n) % 6 = 0 →
  35 % (n + 1) = 0 →
  n = 6 := by
  sorry

end sample_size_determination_l54_5484


namespace hyperbola_t_squared_l54_5400

/-- A hyperbola is defined by its center, orientation, and three points it passes through. -/
structure Hyperbola where
  center : ℝ × ℝ
  horizontalOpening : Bool
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- Given a hyperbola with specific properties, calculate t². -/
def calculateTSquared (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem stating that for a hyperbola with given properties, t² = 45/4. -/
theorem hyperbola_t_squared 
  (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_opening : h.horizontalOpening = true)
  (h_point1 : h.point1 = (-3, 4))
  (h_point2 : h.point2 = (-3, 0))
  (h_point3 : ∃ t : ℝ, h.point3 = (t, 3)) :
  calculateTSquared h = 45/4 := by
  sorry

end hyperbola_t_squared_l54_5400


namespace function_identity_l54_5419

def is_strictly_increasing (f : ℕ+ → ℤ) : Prop :=
  ∀ m n : ℕ+, m > n → f m > f n

theorem function_identity (f : ℕ+ → ℤ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ+, f (m * n) = f m * f n)
  (h3 : is_strictly_increasing f) :
  ∀ n : ℕ+, f n = n := by
  sorry

end function_identity_l54_5419
