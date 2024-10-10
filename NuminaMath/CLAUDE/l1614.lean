import Mathlib

namespace round_trip_average_speed_l1614_161487

/-- Calculates the average speed for a round trip given uphill and downhill times and distances -/
theorem round_trip_average_speed
  (uphill_distance : ℝ)
  (uphill_time : ℝ)
  (downhill_distance : ℝ)
  (downhill_time : ℝ)
  (h1 : uphill_distance = 2)
  (h2 : uphill_time = 45 / 60)
  (h3 : downhill_distance = 2)
  (h4 : downhill_time = 15 / 60)
  : (uphill_distance + downhill_distance) / (uphill_time + downhill_time) = 4 := by
  sorry

end round_trip_average_speed_l1614_161487


namespace sequence_properties_l1614_161406

-- Define the sum of the first n terms
def S (n : ℕ) : ℤ := 2 * n^2 - 30 * n

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 4 * n - 32

-- Theorem statement
theorem sequence_properties :
  (a 1 = -28) ∧
  (∀ n : ℕ, a n = S n - S (n-1)) ∧
  (∀ n : ℕ, n ≥ 2 → a n - a (n-1) = 4) :=
by sorry

-- The fact that the sequence is arithmetic follows from the third conjunct
-- of the theorem above, as the difference between consecutive terms is constant.

end sequence_properties_l1614_161406


namespace midpoint_coordinate_product_l1614_161405

/-- Given a line segment CD where C(5,4) is one endpoint and M(4,8) is the midpoint,
    the product of the coordinates of the other endpoint D is 36. -/
theorem midpoint_coordinate_product (C D M : ℝ × ℝ) : 
  C = (5, 4) →
  M = (4, 8) →
  M.1 = (C.1 + D.1) / 2 →
  M.2 = (C.2 + D.2) / 2 →
  D.1 * D.2 = 36 := by
sorry

end midpoint_coordinate_product_l1614_161405


namespace inequality_solution_set_l1614_161450

theorem inequality_solution_set (x : ℝ) :
  (-6 * x^2 - x + 2 < 0) ↔ (x < -2/3 ∨ x > 1/2) := by
  sorry

end inequality_solution_set_l1614_161450


namespace ellipse_triangle_problem_l1614_161470

-- Define the ellipse
def ellipse (x y : ℝ) (b : ℝ) : Prop := x^2/4 + y^2/b^2 = 1

-- Define the line L
def line_L (x y : ℝ) : Prop := y = x + 2

-- Define parallel lines
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the problem statement
theorem ellipse_triangle_problem 
  (b : ℝ) 
  (ABC : Triangle) 
  (h_ellipse : ellipse ABC.A.1 ABC.A.2 b ∧ ellipse ABC.B.1 ABC.B.2 b)
  (h_C_on_L : line_L ABC.C.1 ABC.C.2)
  (h_AB_parallel_L : parallel ((ABC.B.2 - ABC.A.2) / (ABC.B.1 - ABC.A.1)) 1)
  (h_eccentricity : b^2 = 4/3) :
  (∀ (O : ℝ × ℝ), O = (0, 0) → (ABC.A.1 - O.1) * (ABC.B.2 - O.2) = (ABC.A.2 - O.2) * (ABC.B.1 - O.1) →
    (ABC.B.1 - ABC.A.1)^2 + (ABC.B.2 - ABC.A.2)^2 = 8 ∧ 
    (ABC.B.1 - ABC.A.1) * (ABC.C.2 - ABC.A.2) - (ABC.B.2 - ABC.A.2) * (ABC.C.1 - ABC.A.1) = 4) ∧
  (∀ (m : ℝ), (ABC.B.1 - ABC.A.1)^2 + (ABC.B.2 - ABC.A.2)^2 = (ABC.C.1 - ABC.A.1)^2 + (ABC.C.2 - ABC.A.2)^2 →
    (ABC.C.1 - ABC.A.1)^2 + (ABC.C.2 - ABC.A.2)^2 ≥ (ABC.C.1 - ABC.B.1)^2 + (ABC.C.2 - ABC.B.2)^2 →
    ABC.B.2 - ABC.A.2 = ABC.B.1 - ABC.A.1 - (ABC.B.1 - ABC.A.1)) :=
sorry

end ellipse_triangle_problem_l1614_161470


namespace fifteenth_triangular_number_l1614_161435

/-- The nth triangular number -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 15th triangular number is 120 -/
theorem fifteenth_triangular_number : triangularNumber 15 = 120 := by
  sorry

end fifteenth_triangular_number_l1614_161435


namespace last_two_digits_product_l1614_161410

theorem last_two_digits_product (n : ℕ) : 
  (n % 100 ≥ 10) →  -- Ensure it's a two-digit number
  (n % 5 = 0) →     -- Divisible by 5
  ((n / 10) % 10 + n % 10 = 12) →  -- Sum of last two digits is 12
  ((n / 10) % 10 * (n % 10) = 35) :=
by sorry

end last_two_digits_product_l1614_161410


namespace evaluate_g_l1614_161414

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_g : 3 * g 2 - 4 * g (-2) = -89 := by
  sorry

end evaluate_g_l1614_161414


namespace probability_problem_l1614_161493

theorem probability_problem (P : Set α → ℝ) (A B : Set α)
  (h1 : P (A ∩ B) = 1/6)
  (h2 : P (Aᶜ) = 2/3)
  (h3 : P B = 1/2) :
  (P (A ∩ B) ≠ 0 ∧ P A * P B = P (A ∩ B)) :=
by sorry

end probability_problem_l1614_161493


namespace carson_seed_fertilizer_problem_l1614_161454

/-- The problem of calculating the total amount of seed and fertilizer used by Carson. -/
theorem carson_seed_fertilizer_problem :
  ∀ (seed fertilizer : ℝ),
  seed = 45 →
  seed = 3 * fertilizer →
  seed + fertilizer = 60 :=
by
  sorry

end carson_seed_fertilizer_problem_l1614_161454


namespace percentage_less_than_50k_l1614_161499

/-- Represents the percentage of counties in each population category -/
structure PopulationDistribution :=
  (less_than_50k : ℝ)
  (between_50k_and_150k : ℝ)
  (more_than_150k : ℝ)

/-- The given population distribution from the pie chart -/
def given_distribution : PopulationDistribution :=
  { less_than_50k := 35,
    between_50k_and_150k := 40,
    more_than_150k := 25 }

/-- Theorem stating that the percentage of counties with fewer than 50,000 residents is 35% -/
theorem percentage_less_than_50k (dist : PopulationDistribution) 
  (h1 : dist = given_distribution) : 
  dist.less_than_50k = 35 := by
  sorry

end percentage_less_than_50k_l1614_161499


namespace ice_cream_volume_l1614_161475

/-- The volume of ice cream in a cone and sphere -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1 / 3) * π * r^2 * h
  let sphere_volume := (4 / 3) * π * r^3
  h = 12 ∧ r = 3 → cone_volume + sphere_volume = 72 * π := by sorry

end ice_cream_volume_l1614_161475


namespace rhinestone_problem_l1614_161469

theorem rhinestone_problem (total : ℕ) (bought_fraction : ℚ) (found_fraction : ℚ) : 
  total = 45 → 
  bought_fraction = 1/3 → 
  found_fraction = 1/5 → 
  total - (total * bought_fraction).floor - (total * found_fraction).floor = 21 := by
  sorry

end rhinestone_problem_l1614_161469


namespace power_difference_equals_one_third_l1614_161432

def is_greatest_power_of_2_factor (x : ℕ) : Prop :=
  2^x ∣ 200 ∧ ∀ k > x, ¬(2^k ∣ 200)

def is_greatest_power_of_5_factor (y : ℕ) : Prop :=
  5^y ∣ 200 ∧ ∀ k > y, ¬(5^k ∣ 200)

theorem power_difference_equals_one_third
  (x y : ℕ)
  (h2 : is_greatest_power_of_2_factor x)
  (h5 : is_greatest_power_of_5_factor y) :
  (1/3 : ℚ)^(x - y) = 1/3 := by sorry

end power_difference_equals_one_third_l1614_161432


namespace mode_estimate_is_tallest_rectangle_midpoint_l1614_161453

/-- Represents a rectangle in a frequency distribution histogram --/
structure HistogramRectangle where
  height : ℝ
  base_midpoint : ℝ

/-- Represents a sample frequency distribution histogram --/
structure FrequencyHistogram where
  rectangles : List HistogramRectangle

/-- Finds the tallest rectangle in a frequency histogram --/
def tallestRectangle (h : FrequencyHistogram) : HistogramRectangle :=
  sorry

/-- Estimates the mode of a dataset from a frequency histogram --/
def estimateMode (h : FrequencyHistogram) : ℝ :=
  (tallestRectangle h).base_midpoint

theorem mode_estimate_is_tallest_rectangle_midpoint (h : FrequencyHistogram) :
  estimateMode h = (tallestRectangle h).base_midpoint :=
sorry

end mode_estimate_is_tallest_rectangle_midpoint_l1614_161453


namespace A_equals_nine_l1614_161421

/-- Represents the positions in the diagram --/
inductive Position
| A | B | C | D | E | F | G

/-- Represents the assignment of numbers to positions --/
def Assignment := Position → Fin 10

/-- Checks if all numbers from 1 to 10 are used exactly once --/
def is_valid_assignment (a : Assignment) : Prop :=
  ∀ n : Fin 10, ∃! p : Position, a p = n

/-- Checks if the square condition is satisfied --/
def square_condition (a : Assignment) : Prop :=
  a Position.F = |a Position.A - a Position.B|

/-- Checks if the circle condition is satisfied --/
def circle_condition (a : Assignment) : Prop :=
  a Position.G = a Position.D + a Position.E

/-- Main theorem: A equals 9 --/
theorem A_equals_nine :
  ∃ (a : Assignment),
    is_valid_assignment a ∧
    square_condition a ∧
    circle_condition a ∧
    a Position.A = 9 := by
  sorry

end A_equals_nine_l1614_161421


namespace initialMenCountIs8_l1614_161443

/-- The initial number of men in a group where:
  - The average age increases by 2 years when two women replace two men
  - The two men being replaced are aged 20 and 24 years
  - The average age of the women is 30 years
-/
def initialMenCount : ℕ := by
  -- Define the increase in average age
  let averageAgeIncrease : ℕ := 2
  -- Define the ages of the men being replaced
  let replacedManAge1 : ℕ := 20
  let replacedManAge2 : ℕ := 24
  -- Define the average age of the women
  let womenAverageAge : ℕ := 30
  
  -- The proof goes here
  sorry

/-- Theorem stating that the initial number of men is 8 -/
theorem initialMenCountIs8 : initialMenCount = 8 := by sorry

end initialMenCountIs8_l1614_161443


namespace probability_two_white_balls_l1614_161409

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7

theorem probability_two_white_balls :
  (white_balls : ℚ) / total_balls * (white_balls - 1) / (total_balls - 1) = 4 / 15 := by
  sorry

end probability_two_white_balls_l1614_161409


namespace two_students_not_invited_l1614_161434

/-- Represents the social network of students in Mia's class -/
structure ClassNetwork where
  total_students : ℕ
  mia_friends : ℕ
  friends_of_friends : ℕ

/-- Calculates the number of students not invited to Mia's study session -/
def students_not_invited (network : ClassNetwork) : ℕ :=
  network.total_students - (1 + network.mia_friends + network.friends_of_friends)

/-- Theorem stating that 2 students will not be invited to Mia's study session -/
theorem two_students_not_invited (network : ClassNetwork) 
  (h1 : network.total_students = 15)
  (h2 : network.mia_friends = 4)
  (h3 : network.friends_of_friends = 8) : 
  students_not_invited network = 2 := by
  sorry

#eval students_not_invited ⟨15, 4, 8⟩

end two_students_not_invited_l1614_161434


namespace no_solution_condition_l1614_161495

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (x - a) / (x - 1) - 3 / x ≠ 1) ↔ (a = 1 ∨ a = -2) :=
by sorry

end no_solution_condition_l1614_161495


namespace tim_one_dollar_bills_l1614_161430

/-- Represents the number of bills of a certain denomination -/
structure BillCount where
  count : ℕ
  denomination : ℕ

/-- Represents Tim's wallet -/
structure Wallet where
  tenDollarBills : BillCount
  fiveDollarBills : BillCount
  oneDollarBills : BillCount

def Wallet.totalValue (w : Wallet) : ℕ :=
  w.tenDollarBills.count * w.tenDollarBills.denomination +
  w.fiveDollarBills.count * w.fiveDollarBills.denomination +
  w.oneDollarBills.count * w.oneDollarBills.denomination

def Wallet.totalBills (w : Wallet) : ℕ :=
  w.tenDollarBills.count + w.fiveDollarBills.count + w.oneDollarBills.count

theorem tim_one_dollar_bills 
  (w : Wallet)
  (h1 : w.tenDollarBills = ⟨13, 10⟩)
  (h2 : w.fiveDollarBills = ⟨11, 5⟩)
  (h3 : w.totalValue = 128)
  (h4 : w.totalBills ≥ 16) :
  w.oneDollarBills.count = 57 := by
  sorry

end tim_one_dollar_bills_l1614_161430


namespace graph_regions_count_l1614_161401

/-- A line in the coordinate plane defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The set of lines defining the graph -/
def graph_lines : Set Line := {⟨3, 0⟩, ⟨1/3, 0⟩}

/-- The number of regions created by the graph lines -/
def num_regions : ℕ := 4

/-- Theorem stating that the number of regions created by the graph lines is 4 -/
theorem graph_regions_count :
  num_regions = 4 :=
sorry

end graph_regions_count_l1614_161401


namespace domain_of_f_l1614_161480

theorem domain_of_f (m : ℝ) : 
  (∀ x : ℝ, (m^2 - 3*m + 2)*x^2 + (m - 1)*x + 1 > 0) ↔ (m > 7/3 ∨ m ≤ 1) :=
by sorry

end domain_of_f_l1614_161480


namespace log_2_base_10_bounds_l1614_161437

theorem log_2_base_10_bounds :
  (10^2 = 100) →
  (10^3 = 1000) →
  (2^7 = 128) →
  (2^10 = 1024) →
  (2 / 7 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (3 / 10 : ℝ) :=
by sorry

end log_2_base_10_bounds_l1614_161437


namespace completing_square_equivalence_l1614_161459

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 6*x + 4 = 0 ↔ (x - 3)^2 = 5 :=
by sorry

end completing_square_equivalence_l1614_161459


namespace base5_to_base7_conversion_l1614_161419

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- Represents the number 412 in base 5 -/
def num_base5 : ℕ := 412

/-- Represents the number 212 in base 7 -/
def num_base7 : ℕ := 212

theorem base5_to_base7_conversion :
  base10ToBase7 (base5ToBase10 num_base5) = num_base7 := by
  sorry

end base5_to_base7_conversion_l1614_161419


namespace time_difference_walk_bike_l1614_161439

/-- The number of blocks between Youseff's home and office -/
def B : ℕ := 21

/-- The time in minutes it takes to walk one block -/
def walk_time_per_block : ℚ := 1

/-- The time in minutes it takes to bike one block -/
def bike_time_per_block : ℚ := 20 / 60

/-- The total time in minutes it takes to walk to work -/
def total_walk_time : ℚ := B * walk_time_per_block

/-- The total time in minutes it takes to bike to work -/
def total_bike_time : ℚ := B * bike_time_per_block

theorem time_difference_walk_bike : 
  total_walk_time - total_bike_time = 14 := by
  sorry

end time_difference_walk_bike_l1614_161439


namespace new_person_weight_l1614_161411

theorem new_person_weight (n : ℕ) (initial_avg : ℝ) (weight_decrease : ℝ) :
  n = 20 →
  initial_avg = 58 →
  weight_decrease = 5 →
  let total_weight := n * initial_avg
  let new_avg := initial_avg - weight_decrease
  let new_person_weight := total_weight - (n + 1) * new_avg
  new_person_weight = 47 := by
sorry

end new_person_weight_l1614_161411


namespace largest_minus_smallest_difference_l1614_161476

def digits : List Nat := [3, 9, 6, 0, 5, 1, 7]

def largest_number (ds : List Nat) : Nat :=
  sorry

def smallest_number (ds : List Nat) : Nat :=
  sorry

theorem largest_minus_smallest_difference :
  largest_number digits - smallest_number digits = 8729631 := by
  sorry

end largest_minus_smallest_difference_l1614_161476


namespace student_travel_distance_l1614_161451

/-- Proves that given a total distance of 105.00000000000003 km, where 1/5 is traveled by foot
    and 2/3 is traveled by bus, the remaining distance traveled by car is 14.000000000000002 km. -/
theorem student_travel_distance (total_distance : ℝ) 
    (h1 : total_distance = 105.00000000000003) 
    (foot_fraction : ℝ) (h2 : foot_fraction = 1/5)
    (bus_fraction : ℝ) (h3 : bus_fraction = 2/3) : 
    total_distance - (foot_fraction * total_distance + bus_fraction * total_distance) = 14.000000000000002 := by
  sorry

end student_travel_distance_l1614_161451


namespace coffee_needed_l1614_161460

/-- The amount of coffee needed for Taylor's house guests -/
theorem coffee_needed (cups_weak cups_strong : ℕ) 
  (h1 : cups_weak = cups_strong)
  (h2 : cups_weak + cups_strong = 24) : ℕ :=
by
  sorry

#check coffee_needed

end coffee_needed_l1614_161460


namespace complex_equation_solution_l1614_161426

theorem complex_equation_solution (a : ℝ) : 
  (2 + a * Complex.I) / (1 + Complex.I) = (3 : ℂ) + Complex.I → a = 4 := by
  sorry

end complex_equation_solution_l1614_161426


namespace find_other_number_l1614_161467

theorem find_other_number (a b : ℤ) (h1 : 3 * a + 4 * b = 161) (h2 : a = 17 ∨ b = 17) : 
  (a = 31 ∨ b = 31) :=
sorry

end find_other_number_l1614_161467


namespace rectangle_circle_area_ratio_l1614_161408

theorem rectangle_circle_area_ratio :
  ∀ (w r : ℝ),
  w > 0 → r > 0 →
  6 * w = 2 * Real.pi * r →
  (2 * w * w) / (Real.pi * r * r) = 2 * Real.pi / 9 := by
  sorry

end rectangle_circle_area_ratio_l1614_161408


namespace triangle_angle_c_l1614_161448

theorem triangle_angle_c (A B C : Real) (h1 : 2 * Real.sin A + 5 * Real.cos B = 5) 
  (h2 : 5 * Real.sin B + 2 * Real.cos A = 2) 
  (h3 : A + B + C = Real.pi) : 
  C = Real.arcsin (1/5) ∨ C = Real.pi - Real.arcsin (1/5) := by
sorry

end triangle_angle_c_l1614_161448


namespace rectangular_box_volume_l1614_161433

/-- 
Given a rectangular box with face areas of 36, 18, and 8 square inches,
prove that its volume is 72 cubic inches.
-/
theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 36)
  (area2 : w * h = 18)
  (area3 : l * h = 8) :
  l * w * h = 72 := by
  sorry

end rectangular_box_volume_l1614_161433


namespace cubic_function_coefficients_l1614_161413

/-- Given a cubic function f(x) = ax³ - bx + 4, prove that if f(2) = -4/3 and f'(2) = 0,
    then a = 1/3 and b = 4 -/
theorem cubic_function_coefficients (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - b * x + 4
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 - b
  f 2 = -4/3 ∧ f' 2 = 0 → a = 1/3 ∧ b = 4 := by
  sorry

end cubic_function_coefficients_l1614_161413


namespace securities_stamp_duty_difference_l1614_161489

/-- The securities transaction stamp duty problem -/
theorem securities_stamp_duty_difference :
  let old_rate : ℚ := 3 / 1000
  let new_rate : ℚ := 1 / 1000
  let purchase_value : ℚ := 100000
  (purchase_value * old_rate - purchase_value * new_rate) = 200 := by
  sorry

end securities_stamp_duty_difference_l1614_161489


namespace odd_square_minus_one_divisible_by_24_l1614_161456

theorem odd_square_minus_one_divisible_by_24 (n : ℤ) : 
  Odd (n^2) → (n^2 % 9 ≠ 0) → (n^2 - 1) % 24 = 0 := by
  sorry

end odd_square_minus_one_divisible_by_24_l1614_161456


namespace binary_101101_equals_45_l1614_161423

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end binary_101101_equals_45_l1614_161423


namespace x_equals_six_l1614_161482

theorem x_equals_six (a b x : ℝ) 
  (h1 : 2^a = x) 
  (h2 : 3^b = x) 
  (h3 : 1/a + 1/b = 1) : 
  x = 6 := by sorry

end x_equals_six_l1614_161482


namespace vector_AB_l1614_161425

/-- Given points A(1, -1) and B(1, 2), prove that the vector AB is (0, 3) -/
theorem vector_AB (A B : ℝ × ℝ) (hA : A = (1, -1)) (hB : B = (1, 2)) :
  B.1 - A.1 = 0 ∧ B.2 - A.2 = 3 := by
  sorry

end vector_AB_l1614_161425


namespace john_work_hours_l1614_161490

def hours_per_day : ℕ := 8
def start_day : ℕ := 3
def end_day : ℕ := 8

def total_days : ℕ := end_day - start_day

theorem john_work_hours : hours_per_day * total_days = 40 := by
  sorry

end john_work_hours_l1614_161490


namespace sports_competition_theorem_l1614_161416

-- Part a
def highest_average_rank (num_athletes : ℕ) (num_judges : ℕ) (max_rank_diff : ℕ) : ℚ :=
  8/3

-- Part b
def highest_winner_rank (num_players : ℕ) (max_rank_diff : ℕ) : ℕ :=
  21

theorem sports_competition_theorem :
  (highest_average_rank 20 9 3 = 8/3) ∧
  (highest_winner_rank 1024 2 = 21) :=
by sorry

end sports_competition_theorem_l1614_161416


namespace lower_limit_x_l1614_161441

/-- The function f(x) = x - 5 -/
def f (x : ℝ) : ℝ := x - 5

/-- The lower limit of x for which f(x) ≤ 8 is 13 -/
theorem lower_limit_x (x : ℝ) : f x ≤ 8 ↔ x ≤ 13 := by
  sorry

end lower_limit_x_l1614_161441


namespace tree_planting_equation_correct_l1614_161440

/-- Represents the tree planting scenario -/
structure TreePlanting where
  totalTrees : ℕ
  originalRate : ℝ
  actualRateFactor : ℝ
  daysAhead : ℕ

/-- The equation representing the tree planting scenario is correct -/
theorem tree_planting_equation_correct (tp : TreePlanting)
  (h1 : tp.totalTrees = 960)
  (h2 : tp.originalRate > 0)
  (h3 : tp.actualRateFactor = 4/3)
  (h4 : tp.daysAhead = 4) :
  (tp.totalTrees : ℝ) / tp.originalRate - (tp.totalTrees : ℝ) / (tp.actualRateFactor * tp.originalRate) = tp.daysAhead :=
sorry

end tree_planting_equation_correct_l1614_161440


namespace arithmetic_sequence_common_difference_l1614_161497

/-- An arithmetic sequence with given terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 1 - seq.a 0

theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h2 : seq.a 2 = 12)
  (h6 : seq.a 6 = 4) :
  common_difference seq = -2 := by
sorry

end arithmetic_sequence_common_difference_l1614_161497


namespace car_speed_first_hour_l1614_161455

/-- Proves that given a car's speed in the second hour and its average speed over two hours, we can determine its speed in the first hour. -/
theorem car_speed_first_hour 
  (speed_second_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_second_hour = 80) 
  (h2 : average_speed = 90) : 
  ∃ (speed_first_hour : ℝ), 
    speed_first_hour = 100 ∧ 
    average_speed = (speed_first_hour + speed_second_hour) / 2 := by
  sorry

#check car_speed_first_hour

end car_speed_first_hour_l1614_161455


namespace imaginary_part_of_complex_l1614_161496

theorem imaginary_part_of_complex (z : ℂ) (h : z = 1 - 2 * Complex.I) : 
  z.im = -2 := by sorry

end imaginary_part_of_complex_l1614_161496


namespace opposite_of_five_l1614_161424

theorem opposite_of_five : 
  -(5 : ℤ) = -5 := by sorry

end opposite_of_five_l1614_161424


namespace john_finishes_ahead_l1614_161400

/-- The distance John finishes ahead of Steve in a race --/
def distance_ahead (john_speed steve_speed initial_distance push_time : ℝ) : ℝ :=
  (john_speed * push_time) - (steve_speed * push_time + initial_distance)

/-- Theorem stating that John finishes 2 meters ahead of Steve --/
theorem john_finishes_ahead :
  let john_speed : ℝ := 4.2
  let steve_speed : ℝ := 3.7
  let initial_distance : ℝ := 15
  let push_time : ℝ := 34
  distance_ahead john_speed steve_speed initial_distance push_time = 2 := by
sorry


end john_finishes_ahead_l1614_161400


namespace updated_mean_calculation_l1614_161442

theorem updated_mean_calculation (n : ℕ) (original_mean : ℝ) (decrement : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : decrement = 34) :
  (n : ℝ) * original_mean - n * decrement = n * 166 := by
  sorry

end updated_mean_calculation_l1614_161442


namespace seahawks_score_is_37_l1614_161438

/-- The final score of the Seattle Seahawks in the football game -/
def seahawks_final_score : ℕ :=
  let touchdowns : ℕ := 4
  let field_goals : ℕ := 3
  let touchdown_points : ℕ := 7
  let field_goal_points : ℕ := 3
  touchdowns * touchdown_points + field_goals * field_goal_points

/-- Theorem stating that the Seattle Seahawks' final score is 37 points -/
theorem seahawks_score_is_37 : seahawks_final_score = 37 := by
  sorry

end seahawks_score_is_37_l1614_161438


namespace marble_difference_l1614_161463

/-- The number of marbles Amon and Rhonda have combined -/
def total_marbles : ℕ := 215

/-- The number of marbles Rhonda has -/
def rhonda_marbles : ℕ := 80

/-- Amon has more marbles than Rhonda -/
axiom amon_has_more : ∃ (amon_marbles : ℕ), amon_marbles > rhonda_marbles ∧ amon_marbles + rhonda_marbles = total_marbles

/-- The difference between Amon's and Rhonda's marbles is 55 -/
theorem marble_difference : ∃ (amon_marbles : ℕ), amon_marbles - rhonda_marbles = 55 := by sorry

end marble_difference_l1614_161463


namespace red_minus_white_equals_three_l1614_161494

-- Define the flower counts for each category
def total_flowers : ℕ := 100
def yellow_white : ℕ := 13
def red_yellow : ℕ := 17
def red_white : ℕ := 14
def blue_yellow : ℕ := 16
def blue_white : ℕ := 9
def red_blue_yellow : ℕ := 8
def red_white_blue : ℕ := 6

-- Define the number of flowers containing red
def red_flowers : ℕ := red_yellow + red_white + red_blue_yellow + red_white_blue

-- Define the number of flowers containing white
def white_flowers : ℕ := yellow_white + red_white + blue_white + red_white_blue

-- Theorem statement
theorem red_minus_white_equals_three :
  red_flowers - white_flowers = 3 :=
by sorry

end red_minus_white_equals_three_l1614_161494


namespace new_person_weight_l1614_161468

theorem new_person_weight (n : ℕ) (initial_weight replaced_weight avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : replaced_weight = 70)
  (h3 : avg_increase = 2.5) :
  let new_weight := replaced_weight + n * avg_increase
  new_weight = 90 := by
sorry

end new_person_weight_l1614_161468


namespace train_length_l1614_161462

/-- The length of a train given its speed and time to pass a stationary object -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 36 → speed * time * (1000 / 3600) = 630 :=
by
  sorry

end train_length_l1614_161462


namespace lauren_reaches_andrea_in_30_minutes_l1614_161458

/-- Represents the scenario of Andrea and Lauren biking towards each other --/
structure BikingScenario where
  initial_distance : ℝ
  andrea_speed : ℝ
  lauren_speed : ℝ
  decrease_rate : ℝ
  flat_tire_time : ℝ
  lauren_delay : ℝ

/-- Calculates the total time for Lauren to reach Andrea --/
def totalTime (scenario : BikingScenario) : ℝ :=
  sorry

/-- The theorem stating that Lauren reaches Andrea after 30 minutes --/
theorem lauren_reaches_andrea_in_30_minutes (scenario : BikingScenario)
  (h1 : scenario.initial_distance = 30)
  (h2 : scenario.andrea_speed = 2 * scenario.lauren_speed)
  (h3 : scenario.decrease_rate = 2)
  (h4 : scenario.flat_tire_time = 10)
  (h5 : scenario.lauren_delay = 5) :
  totalTime scenario = 30 :=
sorry

end lauren_reaches_andrea_in_30_minutes_l1614_161458


namespace cards_arrangement_unique_l1614_161407

-- Define the suits and ranks
inductive Suit : Type
| Hearts | Diamonds | Clubs

inductive Rank : Type
| Four | Five | Eight

-- Define a card as a pair of rank and suit
def Card : Type := Rank × Suit

-- Define the arrangement of cards
def Arrangement : Type := List Card

-- Define the conditions
def club_right_of_heart_and_diamond (arr : Arrangement) : Prop :=
  ∃ i j k, i < j ∧ j < k ∧ 
    (arr.get i).2 = Suit.Hearts ∧ 
    (arr.get j).2 = Suit.Diamonds ∧ 
    (arr.get k).2 = Suit.Clubs

def five_left_of_heart (arr : Arrangement) : Prop :=
  ∃ i j, i < j ∧ 
    (arr.get i).1 = Rank.Five ∧ 
    (arr.get j).2 = Suit.Hearts

def eight_right_of_four (arr : Arrangement) : Prop :=
  ∃ i j, i < j ∧ 
    (arr.get i).1 = Rank.Four ∧ 
    (arr.get j).1 = Rank.Eight

-- Define the correct arrangement
def correct_arrangement : Arrangement :=
  [(Rank.Five, Suit.Diamonds), (Rank.Four, Suit.Hearts), (Rank.Eight, Suit.Clubs)]

-- Theorem statement
theorem cards_arrangement_unique :
  ∀ (arr : Arrangement),
    arr.length = 3 ∧
    club_right_of_heart_and_diamond arr ∧
    five_left_of_heart arr ∧
    eight_right_of_four arr →
    arr = correct_arrangement :=
sorry

end cards_arrangement_unique_l1614_161407


namespace floor_sqrt_20_squared_l1614_161404

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by
  sorry

end floor_sqrt_20_squared_l1614_161404


namespace die_roll_sequences_l1614_161420

/-- The number of sides on the die -/
def num_sides : ℕ := 6

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 6

/-- The number of distinct sequences when rolling a die -/
def num_sequences : ℕ := num_sides ^ num_rolls

theorem die_roll_sequences :
  num_sequences = 46656 := by
  sorry

end die_roll_sequences_l1614_161420


namespace root_implies_a_values_l1614_161444

theorem root_implies_a_values (a : ℝ) :
  ((-1)^2 * a^2 + 2011 * (-1) * a - 2012 = 0) →
  (a = 2012 ∨ a = -1) :=
by
  sorry

end root_implies_a_values_l1614_161444


namespace specific_triangle_BD_length_l1614_161479

/-- A right triangle with a perpendicular from the right angle to the hypotenuse -/
structure RightTriangleWithAltitude where
  -- The lengths of the sides
  AB : ℝ
  AC : ℝ
  BC : ℝ
  -- The length of the altitude
  AD : ℝ
  -- The length of the segment from B to D
  BD : ℝ
  -- Conditions
  right_angle : AB^2 + AC^2 = BC^2
  altitude_perpendicular : AD * BC = AB * AC
  pythagoras_BD : BD^2 + AD^2 = BC^2

/-- The main theorem about the specific triangle in the problem -/
theorem specific_triangle_BD_length 
  (triangle : RightTriangleWithAltitude)
  (h_AB : triangle.AB = 45)
  (h_AC : triangle.AC = 60) :
  triangle.BD = 63 := by
  sorry

#check specific_triangle_BD_length

end specific_triangle_BD_length_l1614_161479


namespace final_value_is_four_l1614_161431

def increment_sequence (initial : ℕ) : ℕ :=
  let step1 := initial + 1
  let step2 := step1 + 2
  step2

theorem final_value_is_four :
  increment_sequence 1 = 4 := by
  sorry

end final_value_is_four_l1614_161431


namespace frank_weekly_spending_l1614_161422

theorem frank_weekly_spending (lawn_money weed_money weeks : ℕ) 
  (h1 : lawn_money = 5)
  (h2 : weed_money = 58)
  (h3 : weeks = 9) :
  (lawn_money + weed_money) / weeks = 7 := by
  sorry

end frank_weekly_spending_l1614_161422


namespace geometric_sequence_fifth_term_l1614_161428

/-- Given a geometric sequence {aₙ} where a₂a₃a₄ = 1 and a₆a₇a₈ = 64, prove that a₅ = 2 -/
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)  -- a is the sequence
  (h1 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)  -- a is a geometric sequence
  (h2 : a 2 * a 3 * a 4 = 1)  -- Condition 1
  (h3 : a 6 * a 7 * a 8 = 64)  -- Condition 2
  : a 5 = 2 := by
  sorry

end geometric_sequence_fifth_term_l1614_161428


namespace arithmetic_sequence_product_l1614_161412

/-- Given an arithmetic sequence where the fifth term is 15 and the common difference is 2,
    prove that the product of the first two terms is 63. -/
theorem arithmetic_sequence_product (a : ℕ → ℕ) :
  (∀ n, a (n + 1) = a n + 2) →  -- Common difference is 2
  a 5 = 15 →                    -- Fifth term is 15
  a 1 * a 2 = 63 :=              -- Product of first two terms is 63
by sorry

end arithmetic_sequence_product_l1614_161412


namespace ball_game_proof_l1614_161491

theorem ball_game_proof (total_balls : ℕ) (red_prob_1 : ℚ) (black_prob_2 red_prob_2 : ℚ) 
  (green_balls : ℕ) (red_prob_3 : ℚ) :
  total_balls = 10 →
  red_prob_1 = 1 →
  black_prob_2 = 1/2 →
  red_prob_2 = 1/2 →
  green_balls = 2 →
  red_prob_3 = 7/10 →
  ∃ (black_balls : ℕ), black_balls = 1 := by
  sorry

#check ball_game_proof

end ball_game_proof_l1614_161491


namespace lucille_remaining_cents_l1614_161465

-- Define the problem parameters
def cents_per_weed : ℕ := 6
def weeds_flower_bed : ℕ := 11
def weeds_vegetable_patch : ℕ := 14
def weeds_grass : ℕ := 32
def soda_cost : ℕ := 99

-- Calculate the total weeds pulled
def total_weeds_pulled : ℕ := weeds_flower_bed + weeds_vegetable_patch + weeds_grass / 2

-- Calculate the earnings
def earnings : ℕ := total_weeds_pulled * cents_per_weed

-- Calculate the remaining cents
def remaining_cents : ℕ := earnings - soda_cost

-- Theorem to prove
theorem lucille_remaining_cents : remaining_cents = 147 := by
  sorry

end lucille_remaining_cents_l1614_161465


namespace range_of_a_given_quadratic_inequality_l1614_161478

theorem range_of_a_given_quadratic_inequality (a : ℝ) : 
  (∀ x : ℝ, 4 * x^2 + (a - 2) * x + (1/4 : ℝ) > 0) → 
  0 < a ∧ a < 4 :=
by sorry

end range_of_a_given_quadratic_inequality_l1614_161478


namespace unique_y_value_l1614_161488

theorem unique_y_value : ∃! y : ℝ, y > 0 ∧ (y / 100) * y = 9 := by sorry

end unique_y_value_l1614_161488


namespace tan_alpha_plus_pi_fourth_l1614_161472

theorem tan_alpha_plus_pi_fourth (α : ℝ) (M : ℝ × ℝ) :
  M.1 = 1 ∧ M.2 = Real.sqrt 3 →
  (∃ t : ℝ, t > 0 ∧ t * M.1 = 1 ∧ t * M.2 = Real.tan α) →
  Real.tan (α + π / 4) = -2 - Real.sqrt 3 := by
sorry

end tan_alpha_plus_pi_fourth_l1614_161472


namespace inequality_solution_set_l1614_161471

theorem inequality_solution_set (x : ℝ) : 
  (2 * x / 5 ≤ 3 + x ∧ 3 + x < 4 - x / 3) ↔ -5 ≤ x ∧ x < 3/4 := by
  sorry

end inequality_solution_set_l1614_161471


namespace det_trig_matrix_zero_l1614_161492

/-- The determinant of a specific 3x3 matrix involving trigonometric functions is zero. -/
theorem det_trig_matrix_zero (θ φ : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![0, 2 * Real.sin θ, - Real.cos θ;
                                       -2 * Real.sin θ, 0, Real.sin φ;
                                       Real.cos θ, - Real.sin φ, 0]
  Matrix.det M = 0 := by sorry

end det_trig_matrix_zero_l1614_161492


namespace strip_overlap_area_l1614_161445

theorem strip_overlap_area (β : Real) : 
  let strip1_width : Real := 1
  let strip2_width : Real := 2
  let circle_radius : Real := 1
  let rhombus_area : Real := (1/2) * strip1_width * strip2_width * Real.sin β
  let circle_area : Real := Real.pi * circle_radius^2
  rhombus_area - circle_area = Real.sin β - Real.pi := by sorry

end strip_overlap_area_l1614_161445


namespace phase_shift_of_sine_l1614_161417

theorem phase_shift_of_sine (φ : Real) : 
  (0 ≤ φ ∧ φ ≤ 2 * Real.pi) →
  (∀ x, Real.sin (x + φ) = Real.sin (x - Real.pi / 6)) →
  φ = 11 * Real.pi / 6 := by
sorry

end phase_shift_of_sine_l1614_161417


namespace coin_problem_l1614_161474

/-- Represents the value of a coin in paise -/
inductive CoinValue
  | paise20 : CoinValue
  | paise25 : CoinValue

/-- Calculates the total value in rupees given the number of coins of each type -/
def totalValueInRupees (coins20 : ℕ) (coins25 : ℕ) : ℚ :=
  (coins20 * 20 + coins25 * 25) / 100

theorem coin_problem :
  let totalCoins : ℕ := 344
  let coins20 : ℕ := 300
  let coins25 : ℕ := totalCoins - coins20
  totalValueInRupees coins20 coins25 = 71 := by
  sorry

end coin_problem_l1614_161474


namespace function_range_l1614_161461

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 4

-- Define the domain
def domain : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem function_range :
  Set.range (fun x => f x) = Set.Icc (-5) 4 := by sorry

end function_range_l1614_161461


namespace glitched_clock_correct_time_l1614_161429

/-- Represents a 12-hour digital clock with a glitch that displays 7 instead of 5 -/
structure GlitchedClock where
  hours : Fin 12
  minutes : Fin 60

/-- Checks if a given hour is displayed correctly -/
def correctHour (h : Fin 12) : Bool :=
  h ≠ 5

/-- Checks if a given minute is displayed correctly -/
def correctMinute (m : Fin 60) : Bool :=
  m % 10 ≠ 5 ∧ m / 10 ≠ 5

/-- Calculates the fraction of the day the clock shows the correct time -/
def fractionCorrect : ℚ :=
  (11 : ℚ) / 12 * (54 : ℚ) / 60

theorem glitched_clock_correct_time :
  fractionCorrect = 33 / 40 := by
  sorry

#eval fractionCorrect

end glitched_clock_correct_time_l1614_161429


namespace switch_connections_l1614_161466

theorem switch_connections (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 := by
  sorry

end switch_connections_l1614_161466


namespace fifteenth_student_age_l1614_161464

theorem fifteenth_student_age
  (total_students : Nat)
  (avg_age_all : ℝ)
  (num_group1 : Nat)
  (avg_age_group1 : ℝ)
  (num_group2 : Nat)
  (avg_age_group2 : ℝ)
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 8)
  (h4 : avg_age_group1 = 14)
  (h5 : num_group2 = 6)
  (h6 : avg_age_group2 = 16)
  : ℝ :=
by
  sorry

#check fifteenth_student_age

end fifteenth_student_age_l1614_161464


namespace sum_of_cubes_l1614_161457

theorem sum_of_cubes (x y : ℝ) (h1 : x * y = 15) (h2 : x + y = 11) :
  x^3 + y^3 = 836 := by
sorry

end sum_of_cubes_l1614_161457


namespace triangle_third_angle_l1614_161418

theorem triangle_third_angle (a b : ℝ) (ha : a = 37) (hb : b = 75) :
  180 - a - b = 68 := by
  sorry

end triangle_third_angle_l1614_161418


namespace max_xy_over_x2_plus_y2_l1614_161498

theorem max_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 1/3 ≤ x ∧ x ≤ 3/5) (hy : 1/4 ≤ y ∧ y ≤ 1/2) :
  (x * y) / (x^2 + y^2) ≤ 6/13 :=
sorry

end max_xy_over_x2_plus_y2_l1614_161498


namespace area_of_triangle_MOI_l1614_161446

/-- Given a triangle PQR with side lengths, prove that the area of triangle MOI is 11/4 -/
theorem area_of_triangle_MOI (P Q R O I M : ℝ × ℝ) : 
  let pq : ℝ := 10
  let pr : ℝ := 8
  let qr : ℝ := 6
  -- O is the circumcenter
  (O.1 - P.1)^2 + (O.2 - P.2)^2 = (O.1 - Q.1)^2 + (O.2 - Q.2)^2 ∧
  (O.1 - Q.1)^2 + (O.2 - Q.2)^2 = (O.1 - R.1)^2 + (O.2 - R.2)^2 →
  -- I is the incenter
  (I.1 - P.1) / pq + (I.1 - Q.1) / qr + (I.1 - R.1) / pr = 0 ∧
  (I.2 - P.2) / pq + (I.2 - Q.2) / qr + (I.2 - R.2) / pr = 0 →
  -- M is the center of a circle tangent to PR, QR, and the circumcircle
  ∃ (r : ℝ), 
    r = (M.1 - P.1)^2 + (M.2 - P.2)^2 ∧
    r = (M.1 - R.1)^2 + (M.2 - R.2)^2 ∧
    r + ((O.1 - M.1)^2 + (O.2 - M.2)^2).sqrt = (O.1 - P.1)^2 + (O.2 - P.2)^2 →
  -- Area of triangle MOI is 11/4
  abs ((O.1 * (I.2 - M.2) + I.1 * (M.2 - O.2) + M.1 * (O.2 - I.2)) / 2) = 11/4 := by
sorry

end area_of_triangle_MOI_l1614_161446


namespace product_increase_by_2022_l1614_161402

theorem product_increase_by_2022 : ∃ (a b c : ℕ),
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2022 := by
  sorry

end product_increase_by_2022_l1614_161402


namespace cupcake_packages_l1614_161483

/-- Given the initial number of cupcakes, the number of cupcakes eaten, and the number of cupcakes per package,
    calculate the number of complete packages that can be made. -/
def packages_made (initial : ℕ) (eaten : ℕ) (per_package : ℕ) : ℕ :=
  (initial - eaten) / per_package

/-- Theorem stating that with 18 initial cupcakes, 8 eaten, and 2 cupcakes per package,
    the number of packages that can be made is 5. -/
theorem cupcake_packages : packages_made 18 8 2 = 5 := by
  sorry

end cupcake_packages_l1614_161483


namespace smallest_dual_base_representation_l1614_161436

theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (a b : ℕ),
    a > 2 ∧ b > 2 ∧
    n = 2 * a + 1 ∧
    n = 1 * b + 2 ∧
    (∀ (m : ℕ) (c d : ℕ),
      c > 2 → d > 2 →
      m = 2 * c + 1 →
      m = 1 * d + 2 →
      n ≤ m) ∧
    n = 7 :=
by sorry

end smallest_dual_base_representation_l1614_161436


namespace complex_trajectory_l1614_161452

theorem complex_trajectory (z : ℂ) (x y : ℝ) :
  z = x + y * Complex.I →
  Complex.abs z ^ 2 - 2 * Complex.abs z - 3 = 0 →
  x ^ 2 + y ^ 2 = 9 := by
sorry

end complex_trajectory_l1614_161452


namespace negative_two_times_negative_three_l1614_161486

theorem negative_two_times_negative_three : (-2) * (-3) = 6 := by
  sorry

end negative_two_times_negative_three_l1614_161486


namespace equation_rewrite_l1614_161415

theorem equation_rewrite :
  ∃ (m n : ℝ), (∀ x, x^2 - 12*x + 33 = 0 ↔ (x + m)^2 = n) ∧ m = -6 ∧ n = 3 := by
  sorry

end equation_rewrite_l1614_161415


namespace gcd_xyz_square_l1614_161403

theorem gcd_xyz_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, (Nat.gcd x (Nat.gcd y z) * x * y * z) = k ^ 2 :=
sorry

end gcd_xyz_square_l1614_161403


namespace geometric_mean_inequality_l1614_161449

theorem geometric_mean_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let g := Real.sqrt (x * y)
  (g ≥ 3 → 1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≥ 2 / Real.sqrt (1 + g)) ∧
  (g ≤ 2 → 1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≤ 2 / Real.sqrt (1 + g)) := by
  sorry

end geometric_mean_inequality_l1614_161449


namespace min_cost_at_zero_min_cost_value_l1614_161447

/-- Represents a transportation plan for machines between two locations --/
structure TransportPlan where
  x : ℕ  -- Number of machines transported from B to A
  h : x ≤ 6  -- Constraint on x

/-- Calculates the total cost of a transport plan --/
def totalCost (plan : TransportPlan) : ℕ :=
  200 * plan.x + 8600

/-- Theorem: The minimum cost occurs when no machines are moved from B to A --/
theorem min_cost_at_zero :
  ∀ plan : TransportPlan, totalCost plan ≥ 8600 := by
  sorry

/-- Theorem: The minimum cost is 8600 yuan --/
theorem min_cost_value :
  (∃ plan : TransportPlan, totalCost plan = 8600) ∧
  (∀ plan : TransportPlan, totalCost plan ≥ 8600) := by
  sorry

end min_cost_at_zero_min_cost_value_l1614_161447


namespace reciprocal_of_negative_one_sixth_l1614_161481

theorem reciprocal_of_negative_one_sixth :
  ∃ x : ℚ, x * (-1/6 : ℚ) = 1 ∧ x = -6 := by
  sorry

end reciprocal_of_negative_one_sixth_l1614_161481


namespace sum_of_two_valid_numbers_l1614_161485

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ),
    n = d1 * 100000000 + d2 * 10000000 + d3 * 1000000 + d4 * 100000 + 
        d5 * 10000 + d6 * 1000 + d7 * 100 + d8 * 10 + d9 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧ d1 ≠ d9 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧ d2 ≠ d9 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧ d3 ≠ d9 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧ d4 ≠ d9 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧ d5 ≠ d9 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧ d6 ≠ d9 ∧
    d7 ≠ d8 ∧ d7 ≠ d9 ∧
    d8 ≠ d9 ∧
    0 < d1 ∧ d1 ≤ 9 ∧ 0 < d2 ∧ d2 ≤ 9 ∧ 0 < d3 ∧ d3 ≤ 9 ∧ 0 < d4 ∧ d4 ≤ 9 ∧
    0 < d5 ∧ d5 ≤ 9 ∧ 0 < d6 ∧ d6 ≤ 9 ∧ 0 < d7 ∧ d7 ≤ 9 ∧ 0 < d8 ∧ d8 ≤ 9 ∧
    0 < d9 ∧ d9 ≤ 9

theorem sum_of_two_valid_numbers :
  ∃ (a b : ℕ), is_valid_number a ∧ is_valid_number b ∧ a + b = 987654321 :=
by sorry

end sum_of_two_valid_numbers_l1614_161485


namespace sum_of_ages_l1614_161477

/-- Given the ages of siblings and cousins, calculate the sum of their ages. -/
theorem sum_of_ages (juliet ralph maggie nicky lucy lily alex : ℕ) : 
  juliet = 10 ∧ 
  juliet = maggie + 3 ∧ 
  ralph = juliet + 2 ∧ 
  nicky * 2 = ralph ∧ 
  lucy = ralph + 1 ∧ 
  lily = ralph + 1 ∧ 
  alex + 5 = lucy → 
  maggie + ralph + nicky + lucy + lily + alex = 59 := by
sorry

end sum_of_ages_l1614_161477


namespace pentagon_quadrilateral_angle_sum_l1614_161484

theorem pentagon_quadrilateral_angle_sum :
  ∀ (pentagon_interior_angle quadrilateral_reflex_angle : ℝ),
  pentagon_interior_angle = 108 →
  quadrilateral_reflex_angle = 360 - pentagon_interior_angle →
  360 - quadrilateral_reflex_angle = 108 := by
  sorry

end pentagon_quadrilateral_angle_sum_l1614_161484


namespace sets_intersection_and_union_l1614_161427

def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

theorem sets_intersection_and_union :
  (A ∩ B = {x : ℝ | -5 < x ∧ x ≤ -1}) ∧
  (A ∪ (Bᶜ) = {x : ℝ | -5 < x ∧ x < 3}) := by sorry

end sets_intersection_and_union_l1614_161427


namespace allowance_increase_l1614_161473

theorem allowance_increase (base_amount : ℝ) (middle_school_extra : ℝ) (percentage_increase : ℝ) : 
  let middle_school_allowance := base_amount + middle_school_extra
  let senior_year_allowance := middle_school_allowance * (1 + percentage_increase / 100)
  base_amount = 8 ∧ middle_school_extra = 2 ∧ percentage_increase = 150 →
  senior_year_allowance - 2 * middle_school_allowance = 5 := by sorry

end allowance_increase_l1614_161473
