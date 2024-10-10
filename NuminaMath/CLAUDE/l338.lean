import Mathlib

namespace cost_of_planting_flowers_l338_33861

/-- The cost of planting flowers given the prices of flowers, clay pot, and soil bag. -/
theorem cost_of_planting_flowers
  (flower_cost : ℕ)
  (clay_pot_cost : ℕ)
  (soil_bag_cost : ℕ)
  (h1 : flower_cost = 9)
  (h2 : clay_pot_cost = flower_cost + 20)
  (h3 : soil_bag_cost = flower_cost - 2) :
  flower_cost + clay_pot_cost + soil_bag_cost = 45 := by
  sorry

#check cost_of_planting_flowers

end cost_of_planting_flowers_l338_33861


namespace gcf_of_lcms_main_result_l338_33820

theorem gcf_of_lcms (a b c d : ℕ) : 
  Nat.gcd (Nat.lcm a b) (Nat.lcm c d) = Nat.gcd (Nat.lcm 16 21) (Nat.lcm 14 18) := by
  sorry

theorem main_result : Nat.gcd (Nat.lcm 16 21) (Nat.lcm 14 18) = 14 := by
  sorry

end gcf_of_lcms_main_result_l338_33820


namespace cos_sin_transformation_l338_33843

theorem cos_sin_transformation (x : ℝ) : 
  3 * Real.cos x = 3 * Real.sin (2 * (x + 2 * Real.pi / 3) - Real.pi / 6) := by
  sorry

end cos_sin_transformation_l338_33843


namespace no_nonnegative_solutions_l338_33839

theorem no_nonnegative_solutions : ¬∃ x : ℝ, x ≥ 0 ∧ x^2 + 6*x + 9 = 0 := by
  sorry

end no_nonnegative_solutions_l338_33839


namespace larger_number_is_eight_l338_33872

theorem larger_number_is_eight (x y : ℕ) (h1 : x * y = 24) (h2 : x + y = 11) : 
  max x y = 8 := by
sorry

end larger_number_is_eight_l338_33872


namespace inscribed_sphere_volume_l338_33884

/-- The volume of a sphere inscribed in a cube with edge length 8 inches -/
theorem inscribed_sphere_volume :
  let cube_edge : ℝ := 8
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = (256 / 3) * Real.pi :=
by sorry

end inscribed_sphere_volume_l338_33884


namespace kyoko_balls_correct_l338_33882

/-- The number of balls Kyoko bought -/
def num_balls : ℕ := 3

/-- The cost of each ball in dollars -/
def cost_per_ball : ℚ := 154/100

/-- The total amount Kyoko paid in dollars -/
def total_paid : ℚ := 462/100

/-- Theorem stating that the number of balls Kyoko bought is correct -/
theorem kyoko_balls_correct : 
  (cost_per_ball * num_balls : ℚ) = total_paid :=
by sorry

end kyoko_balls_correct_l338_33882


namespace mrs_hilt_hotdog_cost_l338_33835

/-- The total cost in cents for a given number of hot dogs at a given price per hot dog -/
def total_cost (num_hotdogs : ℕ) (price_per_hotdog : ℕ) : ℕ :=
  num_hotdogs * price_per_hotdog

/-- Proof that Mrs. Hilt paid 300 cents for 6 hot dogs at 50 cents each -/
theorem mrs_hilt_hotdog_cost : total_cost 6 50 = 300 := by
  sorry

end mrs_hilt_hotdog_cost_l338_33835


namespace min_cuts_3x3x3_cube_l338_33819

/-- Represents a 3D cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Represents a cut along a plane -/
inductive Cut
  | X : ℕ → Cut
  | Y : ℕ → Cut
  | Z : ℕ → Cut

/-- The minimum number of cuts required to divide a cube into unit cubes -/
def min_cuts (c : Cube 3) : ℕ := 6

/-- Theorem stating that the minimum number of cuts to divide a 3x3x3 cube into 27 unit cubes is 6 -/
theorem min_cuts_3x3x3_cube (c : Cube 3) :
  min_cuts c = 6 :=
by sorry

end min_cuts_3x3x3_cube_l338_33819


namespace warehouse_worker_wage_l338_33818

/-- Represents the problem of calculating warehouse workers' hourly wage --/
theorem warehouse_worker_wage :
  let num_warehouse_workers : ℕ := 4
  let num_managers : ℕ := 2
  let manager_hourly_wage : ℚ := 20
  let fica_tax_rate : ℚ := 1/10
  let days_per_month : ℕ := 25
  let hours_per_day : ℕ := 8
  let total_monthly_cost : ℚ := 22000

  let total_hours : ℕ := days_per_month * hours_per_day
  let manager_monthly_wage : ℚ := num_managers * manager_hourly_wage * total_hours
  
  ∃ (warehouse_hourly_wage : ℚ),
    warehouse_hourly_wage = 15 ∧
    total_monthly_cost = (1 + fica_tax_rate) * (num_warehouse_workers * warehouse_hourly_wage * total_hours + manager_monthly_wage) :=
by sorry

end warehouse_worker_wage_l338_33818


namespace walter_bus_time_l338_33812

def wake_up_time : Nat := 6 * 60 + 30
def leave_time : Nat := 7 * 60 + 30
def return_time : Nat := 16 * 60 + 30
def num_classes : Nat := 7
def class_duration : Nat := 45
def lunch_duration : Nat := 40
def additional_time : Nat := 150

def total_away_time : Nat := return_time - leave_time
def school_time : Nat := num_classes * class_duration + lunch_duration + additional_time

theorem walter_bus_time :
  total_away_time - school_time = 35 := by
  sorry

end walter_bus_time_l338_33812


namespace sqrt_eight_times_half_minus_sqrt_three_power_zero_l338_33828

theorem sqrt_eight_times_half_minus_sqrt_three_power_zero :
  Real.sqrt 8 * (1 / 2) - (Real.sqrt 3) ^ 0 = Real.sqrt 2 - 1 := by
  sorry

end sqrt_eight_times_half_minus_sqrt_three_power_zero_l338_33828


namespace count_integers_satisfying_inequality_l338_33800

theorem count_integers_satisfying_inequality : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, (169 * n : ℝ)^25 > (n : ℝ)^75 ∧ (n : ℝ)^75 > 3^150) ∧ 
    (∀ n : ℕ, (169 * n : ℝ)^25 > (n : ℝ)^75 ∧ (n : ℝ)^75 > 3^150 → n ∈ S) ∧
    S.card = 3 :=
by sorry

end count_integers_satisfying_inequality_l338_33800


namespace cab_speed_fraction_l338_33857

/-- Proves that for a cab with a usual journey time of 40 minutes, if it's 8 minutes late at a reduced speed, then the reduced speed is 5/6 of its usual speed. -/
theorem cab_speed_fraction (usual_time : ℕ) (delay : ℕ) : 
  usual_time = 40 → delay = 8 → (usual_time : ℚ) / (usual_time + delay) = 5 / 6 := by
  sorry

end cab_speed_fraction_l338_33857


namespace volumes_equal_l338_33858

/-- The volume of a solid obtained by rotating a region around the y-axis -/
noncomputable def rotationVolume (f : ℝ → ℝ → Prop) : ℝ := sorry

/-- The region enclosed by the curves x² = 4y, x² = -4y, x = 4, and x = -4 -/
def region1 (x y : ℝ) : Prop :=
  (x^2 = 4*y ∨ x^2 = -4*y) ∧ -4 ≤ x ∧ x ≤ 4

/-- The region represented by points (x, y) that satisfy x² + y² ≤ 16, x² + (y - 2)² ≥ 4, and x² + (y + 2)² ≥ 4 -/
def region2 (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 16 ∧ x^2 + (y - 2)^2 ≥ 4 ∧ x^2 + (y + 2)^2 ≥ 4

/-- The theorem stating that the volumes of the two solids are equal -/
theorem volumes_equal : rotationVolume region1 = rotationVolume region2 := by
  sorry

end volumes_equal_l338_33858


namespace three_positions_from_eight_l338_33822

/-- The number of ways to choose 3 distinct positions from a group of n people. -/
def choose_three_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- The theorem stating that choosing 3 distinct positions from 8 people results in 336 ways. -/
theorem three_positions_from_eight : choose_three_positions 8 = 336 := by
  sorry

end three_positions_from_eight_l338_33822


namespace f_value_at_3_l338_33813

theorem f_value_at_3 (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^7 + a*x^5 + b*x - 5) 
  (h2 : f (-3) = 5) : f 3 = -15 := by
  sorry

end f_value_at_3_l338_33813


namespace wood_length_proof_l338_33838

/-- The original length of a piece of wood, given the length sawed off and the remaining length. -/
def original_length (sawed_off : ℝ) (remaining : ℝ) : ℝ := sawed_off + remaining

/-- Theorem stating that the original length of the wood is 0.41 meters. -/
theorem wood_length_proof :
  let sawed_off : ℝ := 0.33
  let remaining : ℝ := 0.08
  original_length sawed_off remaining = 0.41 := by
  sorry

end wood_length_proof_l338_33838


namespace semicircle_triangle_area_ratio_l338_33887

/-- Given a triangle ABC with sides in ratio 2:3:4 and an inscribed semicircle
    with diameter on the longest side, the ratio of the area of the semicircle
    to the area of the triangle is π√15 / 12 -/
theorem semicircle_triangle_area_ratio (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ratio : a / b = 2 / 3 ∧ b / c = 3 / 4) (h_triangle : (a + b > c) ∧ (b + c > a) ∧ (c + a > b)) :
  let s := (a + b + c) / 2
  let triangle_area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let semicircle_area := π * c^2 / 8
  semicircle_area / triangle_area = π * Real.sqrt 15 / 12 := by
sorry

end semicircle_triangle_area_ratio_l338_33887


namespace boisjoli_farm_egg_production_l338_33867

/-- The number of eggs each hen lays per day at Boisjoli farm -/
theorem boisjoli_farm_egg_production 
  (num_hens : ℕ) 
  (num_days : ℕ) 
  (num_boxes : ℕ) 
  (eggs_per_box : ℕ) 
  (h_hens : num_hens = 270) 
  (h_days : num_days = 7) 
  (h_boxes : num_boxes = 315) 
  (h_eggs_per_box : eggs_per_box = 6) : 
  (num_boxes * eggs_per_box) / (num_hens * num_days) = 1 := by
  sorry

#check boisjoli_farm_egg_production

end boisjoli_farm_egg_production_l338_33867


namespace arithmetic_sequence_of_squares_l338_33845

theorem arithmetic_sequence_of_squares (a b c x y : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ -- a, b, c are positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ -- a, b, c are distinct
  b - a = c - b ∧ -- a, b, c form an arithmetic sequence
  x^2 = a * b ∧ -- x is the geometric mean of a and b
  y^2 = b * c -- y is the geometric mean of b and c
  → 
  (y^2 - b^2 = b^2 - x^2) ∧ -- x^2, b^2, y^2 form an arithmetic sequence
  ¬(y^2 / b^2 = b^2 / x^2) -- x^2, b^2, y^2 do not form a geometric sequence
  := by sorry

end arithmetic_sequence_of_squares_l338_33845


namespace max_distance_difference_l338_33824

-- Define the curve C₂
def C₂ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 1)

-- Define the distance function
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem max_distance_difference :
  ∃ (max : ℝ), max = 2 + 2 * Real.sqrt 39 ∧
  ∀ (P : ℝ × ℝ), C₂ P.1 P.2 →
    dist_squared P A - dist_squared P B ≤ max :=
sorry

end max_distance_difference_l338_33824


namespace statement_d_is_incorrect_l338_33863

theorem statement_d_is_incorrect : ∃ (a b : ℝ), a^2 > b^2 ∧ a * b > 0 ∧ 1 / a ≥ 1 / b := by
  sorry

end statement_d_is_incorrect_l338_33863


namespace first_square_perimeter_l338_33811

/-- Given two squares and a third square with specific properties, 
    prove that the perimeter of the first square is 24 meters. -/
theorem first_square_perimeter : 
  ∀ (s₁ s₂ s₃ : ℝ),
  (4 * s₂ = 32) →  -- Perimeter of second square is 32 m
  (4 * s₃ = 40) →  -- Perimeter of third square is 40 m
  (s₃^2 = s₁^2 + s₂^2) →  -- Area of third square equals sum of areas of first two squares
  (4 * s₁ = 24) :=  -- Perimeter of first square is 24 m
by
  sorry

#check first_square_perimeter

end first_square_perimeter_l338_33811


namespace cubic_function_through_point_l338_33810

/-- Given a function f(x) = ax³ - 3x that passes through the point (-1, 4), prove that a = -1 --/
theorem cubic_function_through_point (a : ℝ) : 
  (fun x : ℝ => a * x^3 - 3*x) (-1) = 4 → a = -1 := by
  sorry

end cubic_function_through_point_l338_33810


namespace right_triangle_set_l338_33869

/-- Checks if a set of three numbers can form a right-angled triangle --/
def isRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The sets of sticks given in the problem --/
def stickSets : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7)]

theorem right_triangle_set :
  ∃! (a b c : ℕ), (a, b, c) ∈ stickSets ∧ isRightTriangle a b c :=
by
  sorry

#check right_triangle_set

end right_triangle_set_l338_33869


namespace marco_card_ratio_l338_33897

/-- Represents the number of cards in Marco's collection -/
def total_cards : ℕ := 500

/-- Represents the number of new cards Marco received in the trade -/
def new_cards : ℕ := 25

/-- Calculates the number of duplicate cards before the trade -/
def duplicate_cards : ℕ := 5 * new_cards

/-- Represents the ratio of duplicate cards to total cards -/
def duplicate_ratio : ℚ := duplicate_cards / total_cards

theorem marco_card_ratio : duplicate_ratio = 1 / 4 := by
  sorry


end marco_card_ratio_l338_33897


namespace max_z_value_l338_33891

theorem max_z_value (x y z : ℝ) 
  (sum_eq : x + y + z = 7)
  (prod_sum_eq : x * y + x * z + y * z = 12)
  (x_pos : x > 0)
  (y_pos : y > 0)
  (z_pos : z > 0) :
  z ≤ 1 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ + 1 = 7 ∧ x₀ * y₀ + x₀ * 1 + y₀ * 1 = 12 :=
sorry

end max_z_value_l338_33891


namespace odd_function_extension_l338_33886

-- Define the function f on the positive real numbers
def f_pos (x : ℝ) : ℝ := x * (x - 1)

-- State the theorem
theorem odd_function_extension {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x) 
  (h_pos : ∀ x > 0, f x = f_pos x) : 
  ∀ x < 0, f x = x * (x + 1) := by
sorry

end odd_function_extension_l338_33886


namespace sum_of_products_l338_33871

theorem sum_of_products : 
  12345 * 5 + 23451 * 4 + 34512 * 3 + 45123 * 2 + 51234 * 1 = 400545 := by
  sorry

end sum_of_products_l338_33871


namespace rod_length_theorem_l338_33875

/-- The length of a rod in meters, given the number of pieces it can be cut into and the length of each piece in centimeters. -/
def rod_length_meters (num_pieces : ℕ) (piece_length_cm : ℕ) : ℚ :=
  (num_pieces * piece_length_cm : ℚ) / 100

/-- Theorem stating that a rod that can be cut into 50 pieces of 85 cm each is 42.5 meters long. -/
theorem rod_length_theorem : rod_length_meters 50 85 = 42.5 := by
  sorry

end rod_length_theorem_l338_33875


namespace arithmetic_sequence_sum_l338_33860

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  (a 3)^2 - 3 * (a 3) - 5 = 0 →
  (a 11)^2 - 3 * (a 11) - 5 = 0 →
  a 5 + a 6 + a 10 = 9/2 := by
  sorry

end arithmetic_sequence_sum_l338_33860


namespace reading_time_calculation_l338_33836

/-- Calculates the time needed to read a book given the reading speed and book properties -/
theorem reading_time_calculation (reading_speed : ℕ) (paragraphs_per_page : ℕ) 
  (sentences_per_paragraph : ℕ) (total_pages : ℕ) : 
  reading_speed = 200 →
  paragraphs_per_page = 20 →
  sentences_per_paragraph = 10 →
  total_pages = 50 →
  (total_pages * paragraphs_per_page * sentences_per_paragraph) / reading_speed = 50 := by
  sorry

#check reading_time_calculation

end reading_time_calculation_l338_33836


namespace range_of_a_l338_33868

-- Define the inequalities p and q
def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the theorem
theorem range_of_a :
  (∃ x : ℝ, (¬(p x) ∧ q x a) ∨ (p x ∧ ¬(q x a))) →
  (a ∈ Set.Icc (0 : ℝ) (1/2)) :=
sorry

end range_of_a_l338_33868


namespace registration_methods_l338_33870

/-- The number of ways to distribute n distinct objects into k non-empty distinct groups -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 students and 3 courses -/
def num_students : ℕ := 5
def num_courses : ℕ := 3

/-- Each student signs up for exactly one course -/
axiom one_course_per_student : distribute num_students num_courses > 0

/-- Each course must have at least one student enrolled -/
axiom non_empty_courses : ∀ (i : Fin num_courses), ∃ (student : Fin num_students), sorry

/-- The number of different registration methods is 150 -/
theorem registration_methods : distribute num_students num_courses = 150 := by sorry

end registration_methods_l338_33870


namespace radiator_water_fraction_l338_33814

/-- The fraction of water remaining after n replacements in a radiator -/
def water_fraction (radiator_capacity : ℚ) (replacement_volume : ℚ) (n : ℕ) : ℚ :=
  (1 - replacement_volume / radiator_capacity) ^ n

theorem radiator_water_fraction :
  let radiator_capacity : ℚ := 20
  let replacement_volume : ℚ := 5
  let num_replacements : ℕ := 5
  water_fraction radiator_capacity replacement_volume num_replacements = 243 / 1024 := by
  sorry

end radiator_water_fraction_l338_33814


namespace pencil_cost_l338_33833

theorem pencil_cost (total_students : ℕ) (total_cost : ℚ) : ∃ (buyers pencils_per_student pencil_cost : ℕ),
  total_students = 30 ∧
  total_cost = 1771 / 100 ∧
  buyers > total_students / 2 ∧
  buyers ≤ total_students ∧
  pencils_per_student > 1 ∧
  pencil_cost > pencils_per_student ∧
  buyers * pencils_per_student * pencil_cost = 1771 ∧
  pencil_cost = 11 :=
by sorry

end pencil_cost_l338_33833


namespace no_real_solutions_system_l338_33804

theorem no_real_solutions_system :
  ¬∃ (x y z : ℝ), (x + y = 3) ∧ (3*x*y - z^2 = 9) := by
  sorry

end no_real_solutions_system_l338_33804


namespace center_is_four_l338_33842

-- Define the grid as a 3x3 matrix of natural numbers
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define a predicate for consecutive numbers being adjacent
def consecutive_adjacent (g : Grid) : Prop := sorry

-- Define a function to get the edge numbers (excluding corners)
def edge_numbers (g : Grid) : List Nat := sorry

-- Define the sum of edge numbers
def edge_sum (g : Grid) : Nat := (edge_numbers g).sum

-- Define a predicate for the grid containing all numbers from 1 to 9
def contains_one_to_nine (g : Grid) : Prop := sorry

-- Define a function to get the center number
def center_number (g : Grid) : Nat := g 1 1

-- Main theorem
theorem center_is_four (g : Grid) 
  (h1 : consecutive_adjacent g)
  (h2 : edge_sum g = 28)
  (h3 : contains_one_to_nine g)
  (h4 : Even (center_number g)) :
  center_number g = 4 := by sorry

end center_is_four_l338_33842


namespace evening_temp_calculation_l338_33865

/-- Given a noon temperature and a temperature drop, calculate the evening temperature. -/
def evening_temperature (noon_temp : ℤ) (temp_drop : ℕ) : ℤ :=
  noon_temp - temp_drop

/-- Theorem: If the noon temperature is 2°C and it drops by 3°C, the evening temperature is -1°C. -/
theorem evening_temp_calculation :
  evening_temperature 2 3 = -1 := by
  sorry

end evening_temp_calculation_l338_33865


namespace smallest_number_divisibility_l338_33847

theorem smallest_number_divisibility (h : ℕ) : 
  (∀ k < 259, ¬(∃ n : ℕ, k + 5 = 8 * n ∧ k + 5 = 11 * n ∧ k + 5 = 3 * n)) ∧
  (∃ n : ℕ, 259 + 5 = 8 * n) ∧
  (∃ n : ℕ, 259 + 5 = 11 * n) ∧
  (∃ n : ℕ, 259 + 5 = 3 * n) :=
by sorry

end smallest_number_divisibility_l338_33847


namespace probability_theorem_l338_33849

/-- Parallelogram with given vertices -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The specific parallelogram ABCD from the problem -/
def ABCD : Parallelogram :=
  { A := (9, 4)
    B := (3, -2)
    C := (-3, -2)
    D := (3, 4) }

/-- Probability of a point in the parallelogram being not above the x-axis -/
def probability_not_above_x_axis (p : Parallelogram) : ℚ :=
  1/2

/-- Theorem stating the probability for the given parallelogram -/
theorem probability_theorem :
  probability_not_above_x_axis ABCD = 1/2 := by sorry

end probability_theorem_l338_33849


namespace order_independent_divisibility_criterion_only_for_3_and_9_l338_33888

/-- A divisibility criterion for a positive integer that depends only on its digits. -/
def DigitDivisibilityCriterion (n : ℕ+) : Type :=
  (digits : List ℕ) → Bool

/-- The property that a divisibility criterion is independent of digit order. -/
def OrderIndependent (n : ℕ+) (criterion : DigitDivisibilityCriterion n) : Prop :=
  ∀ (digits₁ digits₂ : List ℕ), Multiset.ofList digits₁ = Multiset.ofList digits₂ →
    criterion digits₁ = criterion digits₂

/-- Theorem stating that order-independent digit divisibility criteria exist only for 3 and 9. -/
theorem order_independent_divisibility_criterion_only_for_3_and_9 (n : ℕ+) :
    (∃ (criterion : DigitDivisibilityCriterion n), OrderIndependent n criterion) →
    n = 3 ∨ n = 9 := by
  sorry

end order_independent_divisibility_criterion_only_for_3_and_9_l338_33888


namespace no_real_roots_m_range_l338_33827

/-- A quadratic function with parameter m -/
def f (m x : ℝ) : ℝ := x^2 + m*x + (m+3)

/-- The discriminant of the quadratic function -/
def discriminant (m : ℝ) : ℝ := m^2 - 4*(m+3)

theorem no_real_roots_m_range (m : ℝ) :
  (∀ x, f m x ≠ 0) → m ∈ Set.Ioo (-2 : ℝ) 6 := by
  sorry

end no_real_roots_m_range_l338_33827


namespace complement_of_union_l338_33823

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union : (U \ (M ∪ N)) = {4} := by
  sorry

end complement_of_union_l338_33823


namespace greatest_ACCBA_divisible_by_11_and_3_l338_33806

/-- Represents a five-digit number in the form AC,CBA -/
def ACCBA (A B C : Nat) : Nat := A * 10000 + C * 1000 + C * 100 + B * 10 + A

/-- Checks if the digits A, B, and C are distinct -/
def distinct_digits (A B C : Nat) : Prop := A ≠ B ∧ B ≠ C ∧ A ≠ C

/-- Checks if a number is divisible by both 11 and 3 -/
def divisible_by_11_and_3 (n : Nat) : Prop := n % 11 = 0 ∧ n % 3 = 0

/-- The main theorem statement -/
theorem greatest_ACCBA_divisible_by_11_and_3 :
  ∀ A B C : Nat,
  A < 10 ∧ B < 10 ∧ C < 10 →
  distinct_digits A B C →
  divisible_by_11_and_3 (ACCBA A B C) →
  ACCBA A B C ≤ 95695 :=
sorry

end greatest_ACCBA_divisible_by_11_and_3_l338_33806


namespace five_not_in_A_and_B_l338_33844

universe u

def U : Set Nat := {1, 2, 3, 4, 5}

theorem five_not_in_A_and_B
  (A B : Set Nat)
  (h_subset : A ⊆ U ∧ B ⊆ U)
  (h_inter : A ∩ B = {2, 4})
  (h_union : A ∪ B = {1, 2, 3, 4}) :
  5 ∉ A ∧ 5 ∉ B := by
  sorry


end five_not_in_A_and_B_l338_33844


namespace second_question_percentage_l338_33815

theorem second_question_percentage 
  (first_correct : Real) 
  (neither_correct : Real) 
  (both_correct : Real)
  (h1 : first_correct = 0.75)
  (h2 : neither_correct = 0.2)
  (h3 : both_correct = 0.6) :
  ∃ second_correct : Real, 
    second_correct = 0.65 ∧ 
    first_correct + second_correct - both_correct = 1 - neither_correct :=
by sorry

end second_question_percentage_l338_33815


namespace base_conversion_correct_l338_33853

-- Define the base 10 number
def base_10_num : ℕ := 3527

-- Define the base 7 representation
def base_7_representation : List ℕ := [1, 3, 1, 6, 6]

-- Theorem statement
theorem base_conversion_correct :
  base_10_num = (List.foldr (λ (digit : ℕ) (acc : ℕ) => digit + 7 * acc) 0 base_7_representation) :=
by sorry

end base_conversion_correct_l338_33853


namespace inserted_digit_divisible_by_seven_l338_33885

theorem inserted_digit_divisible_by_seven :
  ∀ x : ℕ, x < 10 →
    (20000 + x * 100 + 6) % 7 = 0 ↔ x = 0 ∨ x = 7 := by
  sorry

end inserted_digit_divisible_by_seven_l338_33885


namespace book_cost_range_l338_33859

theorem book_cost_range (p : ℝ) 
  (h1 : 11 * p < 15)
  (h2 : 12 * p > 16) : 
  4 / 3 < p ∧ p < 15 / 11 :=
by sorry

end book_cost_range_l338_33859


namespace star_one_one_eq_neg_eleven_l338_33881

/-- Definition of the * operation for rational numbers -/
def star (a b c : ℚ) (x y : ℚ) : ℚ := a * x + b * y + c

/-- Theorem: Given the conditions, 1 * 1 = -11 -/
theorem star_one_one_eq_neg_eleven 
  (a b c : ℚ) 
  (h1 : star a b c 3 5 = 15) 
  (h2 : star a b c 4 7 = 28) : 
  star a b c 1 1 = -11 := by
  sorry

end star_one_one_eq_neg_eleven_l338_33881


namespace left_handed_jazz_lovers_l338_33854

/-- Represents a club with members of different characteristics -/
structure Club where
  total_members : ℕ
  left_handed : ℕ
  jazz_lovers : ℕ
  right_handed_non_jazz : ℕ

/-- Theorem stating the number of left-handed jazz lovers in the club -/
theorem left_handed_jazz_lovers (c : Club)
  (h1 : c.total_members = 30)
  (h2 : c.left_handed = 11)
  (h3 : c.jazz_lovers = 20)
  (h4 : c.right_handed_non_jazz = 4)
  (h5 : c.left_handed + (c.total_members - c.left_handed) = c.total_members) :
  c.left_handed + c.jazz_lovers - c.total_members + c.right_handed_non_jazz = 5 := by
  sorry

#check left_handed_jazz_lovers

end left_handed_jazz_lovers_l338_33854


namespace cube_sum_geq_triple_product_l338_33889

theorem cube_sum_geq_triple_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 ≥ 3 * a * b * c := by
  sorry

end cube_sum_geq_triple_product_l338_33889


namespace no_double_apply_1987_function_l338_33830

theorem no_double_apply_1987_function :
  ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1987 := by
  sorry

end no_double_apply_1987_function_l338_33830


namespace inequality_equivalence_l338_33808

theorem inequality_equivalence (x : ℝ) :
  (2 * (5 ^ (2 * x)) * Real.sin (2 * x) - 3 ^ x ≥ 5 ^ (2 * x) - 2 * (3 ^ x) * Real.sin (2 * x)) ↔
  (∃ k : ℤ, π / 12 + k * π ≤ x ∧ x ≤ 5 * π / 12 + k * π) :=
by sorry

end inequality_equivalence_l338_33808


namespace cube_surface_area_l338_33851

theorem cube_surface_area (d : ℝ) (h : d = 8 * Real.sqrt 2) : 
  6 * (d / Real.sqrt 2) ^ 2 = 384 := by
  sorry

end cube_surface_area_l338_33851


namespace machines_needed_for_faster_job_additional_machines_needed_l338_33832

theorem machines_needed_for_faster_job (initial_machines : ℕ) (initial_days : ℕ) : ℕ :=
  let total_machine_days := initial_machines * initial_days
  let new_days := initial_days * 3 / 4
  let new_machines := total_machine_days / new_days
  new_machines - initial_machines

theorem additional_machines_needed :
  machines_needed_for_faster_job 12 40 = 4 := by
  sorry

end machines_needed_for_faster_job_additional_machines_needed_l338_33832


namespace two_rats_through_wall_l338_33829

/-- The sum of lengths burrowed by two rats in n days -/
def S (n : ℕ) : ℚ :=
  (2^n - 1) + (2 - 1/(2^(n-1)))

/-- The problem statement -/
theorem two_rats_through_wall : S 5 = 32 + 15/16 := by
  sorry

end two_rats_through_wall_l338_33829


namespace equal_probability_for_all_positions_l338_33880

/-- Represents a lottery draw with n tickets, where one is winning. -/
structure LotteryDraw (n : ℕ) where
  tickets : Fin n → Bool
  winning_exists : ∃ t, tickets t = true
  only_one_winning : ∀ t₁ t₂, tickets t₁ = true → tickets t₂ = true → t₁ = t₂

/-- The probability of drawing the winning ticket in any position of a sequence of n draws. -/
def winning_probability (n : ℕ) (pos : Fin n) (draw : LotteryDraw n) : ℚ :=
  1 / n

/-- Theorem stating that the probability of drawing the winning ticket is equal for all positions in a sequence of 5 draws. -/
theorem equal_probability_for_all_positions (draw : LotteryDraw 5) :
    ∀ pos₁ pos₂ : Fin 5, winning_probability 5 pos₁ draw = winning_probability 5 pos₂ draw :=
  sorry

end equal_probability_for_all_positions_l338_33880


namespace expression_value_l338_33840

theorem expression_value (x y z : ℝ) 
  (h1 : (1 / (y + z)) + (1 / (x + z)) + (1 / (x + y)) = 5)
  (h2 : x + y + z = 2) :
  (x / (y + z)) + (y / (x + z)) + (z / (x + y)) = 7 := by
  sorry

end expression_value_l338_33840


namespace angle_of_inclination_x_eq_neg_one_l338_33877

-- Define a vertical line
def vertical_line (a : ℝ) : Set (ℝ × ℝ) := {p | p.1 = a}

-- Define the angle of inclination for a vertical line
def angle_of_inclination_vertical (l : Set (ℝ × ℝ)) : ℝ := 90

-- Theorem statement
theorem angle_of_inclination_x_eq_neg_one :
  angle_of_inclination_vertical (vertical_line (-1)) = 90 := by
  sorry

end angle_of_inclination_x_eq_neg_one_l338_33877


namespace unique_magnitude_of_quadratic_roots_l338_33896

theorem unique_magnitude_of_quadratic_roots (w : ℂ) : 
  w^2 - 6*w + 40 = 0 → ∃! x : ℝ, ∃ w : ℂ, w^2 - 6*w + 40 = 0 ∧ Complex.abs w = x :=
by
  sorry

end unique_magnitude_of_quadratic_roots_l338_33896


namespace surface_sum_bounds_l338_33803

/-- Represents a standard die with 6 faces -/
structure Die :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7)

/-- Represents the large cube assembled from smaller dice -/
structure LargeCube :=
  (dice : Fin 125 → Die)

/-- The sum of visible numbers on the large cube's surface -/
def surface_sum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the bounds of the surface sum -/
theorem surface_sum_bounds (cube : LargeCube) :
  210 ≤ surface_sum cube ∧ surface_sum cube ≤ 840 := by
  sorry

end surface_sum_bounds_l338_33803


namespace difference_in_amounts_l338_33841

/-- Represents the three products A, B, and C --/
inductive Product
| A
| B
| C

/-- The initial price of a product --/
def initialPrice (p : Product) : ℝ :=
  match p with
  | Product.A => 100
  | Product.B => 150
  | Product.C => 200

/-- The price increase percentage for a product --/
def priceIncrease (p : Product) : ℝ :=
  match p with
  | Product.A => 0.10
  | Product.B => 0.15
  | Product.C => 0.20

/-- The quantity bought after price increase as a fraction of initial quantity --/
def quantityAfterIncrease (p : Product) : ℝ :=
  match p with
  | Product.A => 0.90
  | Product.B => 0.85
  | Product.C => 0.80

/-- The discount percentage --/
def discount : ℝ := 0.05

/-- The additional quantity bought on discount day as a fraction of initial quantity --/
def additionalQuantity (p : Product) : ℝ :=
  match p with
  | Product.A => 0.10
  | Product.B => 0.15
  | Product.C => 0.20

/-- The total amount paid on the price increase day --/
def amountOnIncreaseDay : ℝ :=
  (initialPrice Product.A * (1 + priceIncrease Product.A) * quantityAfterIncrease Product.A) +
  (initialPrice Product.B * (1 + priceIncrease Product.B) * quantityAfterIncrease Product.B) +
  (initialPrice Product.C * (1 + priceIncrease Product.C) * quantityAfterIncrease Product.C)

/-- The total amount paid on the discount day --/
def amountOnDiscountDay : ℝ :=
  (initialPrice Product.A * (1 - discount) * (1 + additionalQuantity Product.A)) +
  (initialPrice Product.B * (1 - discount) * (1 + additionalQuantity Product.B)) +
  (initialPrice Product.C * (1 - discount) * (1 + additionalQuantity Product.C))

/-- The theorem stating the difference in amounts paid --/
theorem difference_in_amounts : amountOnIncreaseDay - amountOnDiscountDay = 58.75 := by
  sorry

end difference_in_amounts_l338_33841


namespace decimal_digits_sum_l338_33825

theorem decimal_digits_sum (a : ℕ) : ∃ (n m : ℕ),
  (10^(n-1) ≤ a ∧ a < 10^n) ∧
  (10^(3*(n-1)) ≤ a^3 ∧ a^3 < 10^(3*n)) ∧
  (3*n - 2 ≤ m ∧ m ≤ 3*n) →
  n + m ≠ 2001 := by
sorry

end decimal_digits_sum_l338_33825


namespace range_of_f_l338_33876

noncomputable def f (x : ℝ) : ℝ := (Real.arccos x)^4 + (Real.arcsin x)^4

theorem range_of_f :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1,
  ∃ y ∈ Set.Icc 0 (π^4/8),
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc 0 (π^4/8) := by
sorry

end range_of_f_l338_33876


namespace opposite_to_83_is_84_l338_33890

/-- Represents a circle divided into 100 equal arcs with numbers assigned to each arc. -/
structure NumberedCircle where
  /-- The assignment of numbers to arcs, represented as a function from arc index to number. -/
  number_assignment : Fin 100 → Fin 100
  /-- The assignment is a bijection (each number is used exactly once). -/
  bijective : Function.Bijective number_assignment

/-- Checks if numbers less than k are evenly distributed on both sides of the diameter through k. -/
def evenlyDistributed (c : NumberedCircle) (k : Fin 100) : Prop :=
  ∀ (i : Fin 100), c.number_assignment i < k →
    (∃ (j : Fin 100), c.number_assignment j < k ∧ (i + 50) % 100 = j)

/-- The main theorem stating that if numbers are evenly distributed for all k,
    then the number opposite to 83 is 84. -/
theorem opposite_to_83_is_84 (c : NumberedCircle) 
    (h : ∀ (k : Fin 100), evenlyDistributed c k) :
    ∃ (i : Fin 100), c.number_assignment i = 83 ∧ c.number_assignment ((i + 50) % 100) = 84 := by
  sorry

end opposite_to_83_is_84_l338_33890


namespace smallest_n_divisible_by_50_and_288_l338_33805

theorem smallest_n_divisible_by_50_and_288 :
  ∃ (n : ℕ), n > 0 ∧ 
    50 ∣ n^2 ∧ 
    288 ∣ n^3 ∧ 
    ∀ (m : ℕ), m > 0 → 50 ∣ m^2 → 288 ∣ m^3 → n ≤ m :=
by
  use 60
  sorry

end smallest_n_divisible_by_50_and_288_l338_33805


namespace real_roots_iff_k_nonzero_l338_33864

theorem real_roots_iff_k_nonzero (K : ℝ) :
  (∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3)) ↔ K ≠ 0 := by
  sorry

end real_roots_iff_k_nonzero_l338_33864


namespace equation_solution_l338_33879

theorem equation_solution (x : ℝ) : 
  Real.sqrt (5 * x - 4) + 15 / Real.sqrt (5 * x - 4) = 8 → x = 29/5 ∨ x = 13/5 := by
  sorry

end equation_solution_l338_33879


namespace coefficient_x_squared_proof_l338_33856

/-- The coefficient of x^2 in the expansion of (1-3x)^7 -/
def coefficient_x_squared : ℕ := 7

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_x_squared_proof :
  coefficient_x_squared = binomial 7 6 := by
  sorry

end coefficient_x_squared_proof_l338_33856


namespace largest_kappa_l338_33831

theorem largest_kappa : ∃ κ : ℝ, κ = 2 ∧ 
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + d^2 = b^2 + c^2 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*c + κ*b*d + a*d) ∧ 
  (∀ κ' : ℝ, κ' > κ → 
    ∃ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
      a^2 + d^2 = b^2 + c^2 ∧ 
      a^2 + b^2 + c^2 + d^2 < a*c + κ'*b*d + a*d) :=
by sorry

end largest_kappa_l338_33831


namespace cubic_arithmetic_progression_complex_root_l338_33898

/-- A cubic polynomial with coefficients in ℝ -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The roots of a cubic polynomial form an arithmetic progression -/
def roots_form_arithmetic_progression (p : CubicPolynomial) : Prop :=
  ∃ (r d : ℂ), p.a * (r - d)^3 + p.b * (r - d)^2 + p.c * (r - d) + p.d = 0 ∧
                p.a * r^3 + p.b * r^2 + p.c * r + p.d = 0 ∧
                p.a * (r + d)^3 + p.b * (r + d)^2 + p.c * (r + d) + p.d = 0

/-- One of the roots of a cubic polynomial is complex -/
def has_complex_root (p : CubicPolynomial) : Prop :=
  ∃ (z : ℂ), z.im ≠ 0 ∧ p.a * z^3 + p.b * z^2 + p.c * z + p.d = 0

/-- The main theorem -/
theorem cubic_arithmetic_progression_complex_root :
  ∃! (a : ℝ), roots_form_arithmetic_progression { a := 1, b := -9, c := 30, d := a } ∧
               has_complex_root { a := 1, b := -9, c := 30, d := a } ∧
               a = -12 := by sorry

end cubic_arithmetic_progression_complex_root_l338_33898


namespace music_store_purchase_total_l338_33852

def trumpet_price : ℝ := 149.16
def music_tool_price : ℝ := 9.98
def song_book_price : ℝ := 4.14
def accessories_price : ℝ := 21.47
def valve_oil_original_price : ℝ := 8.20
def tshirt_price : ℝ := 14.95
def valve_oil_discount_rate : ℝ := 0.20
def sales_tax_rate : ℝ := 0.065

def total_spent : ℝ := 219.67

theorem music_store_purchase_total :
  let valve_oil_price := valve_oil_original_price * (1 - valve_oil_discount_rate)
  let subtotal := trumpet_price + music_tool_price + song_book_price + 
                  accessories_price + valve_oil_price + tshirt_price
  let sales_tax := subtotal * sales_tax_rate
  subtotal + sales_tax = total_spent := by sorry

end music_store_purchase_total_l338_33852


namespace line_slope_is_four_l338_33809

/-- Given a line passing through points (0, 100) and (50, 300), prove that its slope is 4. -/
theorem line_slope_is_four :
  let x₁ : ℝ := 0
  let y₁ : ℝ := 100
  let x₂ : ℝ := 50
  let y₂ : ℝ := 300
  let slope : ℝ := (y₂ - y₁) / (x₂ - x₁)
  slope = 4 := by
  sorry

end line_slope_is_four_l338_33809


namespace house_height_from_shadows_l338_33807

/-- Given a tree and a house casting shadows, calculate the height of the house -/
theorem house_height_from_shadows 
  (tree_height : ℝ) 
  (tree_shadow : ℝ) 
  (house_shadow : ℝ) 
  (h_tree_height : tree_height = 15)
  (h_tree_shadow : tree_shadow = 18)
  (h_house_shadow : house_shadow = 72)
  (h_similar_triangles : tree_height / tree_shadow = house_height / house_shadow) :
  house_height = 60 :=
by
  sorry


end house_height_from_shadows_l338_33807


namespace perimeter_of_square_region_l338_33893

theorem perimeter_of_square_region (total_area : ℝ) (num_squares : ℕ) (perimeter : ℝ) :
  total_area = 588 →
  num_squares = 14 →
  perimeter = 15 * Real.sqrt 42 :=
by
  sorry

end perimeter_of_square_region_l338_33893


namespace game_cost_l338_33802

theorem game_cost (initial_money : ℕ) (toys_count : ℕ) (toy_price : ℕ) (remaining_money : ℕ) :
  initial_money = 68 →
  toys_count = 3 →
  toy_price = 7 →
  remaining_money = toys_count * toy_price →
  initial_money - remaining_money = 47 :=
by
  sorry

end game_cost_l338_33802


namespace no_natural_number_with_digit_product_6552_l338_33878

theorem no_natural_number_with_digit_product_6552 :
  ¬ ∃ n : ℕ, (n.digits 10).prod = 6552 := by
  sorry

end no_natural_number_with_digit_product_6552_l338_33878


namespace coefficient_of_x_fifth_power_l338_33899

theorem coefficient_of_x_fifth_power (x : ℝ) : 
  ∃ (a₆ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ), 
    (x - 2) * (x + 2)^5 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀ ∧ 
    a₅ = 8 := by
  sorry

end coefficient_of_x_fifth_power_l338_33899


namespace tensor_product_of_A_and_B_l338_33817

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {y : ℝ | y ≥ 0}

-- Define the ⊗ operation
def tensorProduct (X Y : Set ℝ) : Set ℝ := (X ∪ Y) \ (X ∩ Y)

-- Theorem statement
theorem tensor_product_of_A_and_B :
  tensorProduct A B = {x : ℝ | x = 0 ∨ x ≥ 2} := by
  sorry

end tensor_product_of_A_and_B_l338_33817


namespace cylinder_volume_problem_l338_33816

theorem cylinder_volume_problem (h₁ : ℝ) (h₂ : ℝ) (r₁ r₂ : ℝ) :
  r₁ = 7 →
  r₂ = 1.2 * r₁ →
  h₂ = 0.85 * h₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  π * r₁^2 * h₁ = 49 * π * h₁ :=
by sorry

end cylinder_volume_problem_l338_33816


namespace complement_union_theorem_l338_33862

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 2}

-- Define set B
def B : Finset Nat := {2, 3, 4}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4, 5} := by
  sorry

end complement_union_theorem_l338_33862


namespace temperature_difference_problem_l338_33837

theorem temperature_difference_problem (M L N : ℝ) : 
  M = L + N →                           -- Minneapolis is N degrees warmer at noon
  (M - 8) - (L + 6) = 4 ∨ (M - 8) - (L + 6) = -4 →  -- Temperature difference at 6:00 PM
  (N = 18 ∨ N = 10) ∧ N * N = 180 := by
sorry

end temperature_difference_problem_l338_33837


namespace subtraction_problem_l338_33821

theorem subtraction_problem (x : ℤ) : x - 46 = 15 → x - 29 = 32 := by
  sorry

end subtraction_problem_l338_33821


namespace f_zero_eq_zero_l338_33883

-- Define the function f
def f : ℝ → ℝ := fun x => sorry

-- State the theorem
theorem f_zero_eq_zero :
  (∀ x : ℝ, f (x + 1) = x^2 + 2*x + 1) →
  f 0 = 0 := by sorry

end f_zero_eq_zero_l338_33883


namespace missing_integers_count_l338_33846

theorem missing_integers_count (n : ℕ) (h : n = 2017) : 
  n - (n - n / 3 + n / 6 - n / 54) = 373 :=
by sorry

end missing_integers_count_l338_33846


namespace compare_large_exponents_l338_33834

theorem compare_large_exponents :
  20^(19^20) > 19^(20^19) := by
  sorry

end compare_large_exponents_l338_33834


namespace bridget_profit_is_fifty_l338_33894

/-- Calculates Bridget's profit from baking and selling bread --/
def bridget_profit (total_loaves : ℕ) (cost_per_loaf : ℚ) 
  (morning_price : ℚ) (late_afternoon_price : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3
  let morning_revenue := morning_sales * morning_price
  let afternoon_remaining := total_loaves - morning_sales
  let afternoon_sales := afternoon_remaining / 2
  let afternoon_revenue := afternoon_sales * (morning_price / 2)
  let late_afternoon_remaining := afternoon_remaining - afternoon_sales
  let late_afternoon_sales := (late_afternoon_remaining * 2) / 3
  let late_afternoon_revenue := late_afternoon_sales * late_afternoon_price
  let evening_sales := late_afternoon_remaining - late_afternoon_sales
  let evening_revenue := evening_sales * cost_per_loaf
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue + evening_revenue
  let total_cost := total_loaves * cost_per_loaf
  total_revenue - total_cost

/-- Bridget's profit is $50 --/
theorem bridget_profit_is_fifty :
  bridget_profit 60 1 3 1 = 50 :=
by sorry

end bridget_profit_is_fifty_l338_33894


namespace right_triangle_third_side_product_l338_33895

theorem right_triangle_third_side_product (a b c d : ℝ) : 
  a = 6 → b = 8 → 
  (a^2 + b^2 = c^2 ∨ a^2 + d^2 = b^2) → 
  c * d = 20 * Real.sqrt 7 := by
  sorry

end right_triangle_third_side_product_l338_33895


namespace hyperbola_eccentricity_l338_33874

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h_arithmetic : (a + b) / 2 = 5/2) (h_geometric : Real.sqrt (a * b) = Real.sqrt 6) :
  let e := Real.sqrt (1 + b^2 / a^2)
  e = Real.sqrt 13 / 3 := by
sorry

end hyperbola_eccentricity_l338_33874


namespace train_platform_crossing_time_l338_33855

/-- The time required for a train to cross a platform -/
theorem train_platform_crossing_time
  (train_speed : Real)
  (man_crossing_time : Real)
  (platform_length : Real)
  (h1 : train_speed = 72 / 3.6) -- 72 kmph converted to m/s
  (h2 : man_crossing_time = 18)
  (h3 : platform_length = 340) :
  let train_length := train_speed * man_crossing_time
  let total_distance := train_length + platform_length
  total_distance / train_speed = 35 := by sorry

end train_platform_crossing_time_l338_33855


namespace cone_slant_height_l338_33866

/-- Given a cone with lateral area 10π cm² and base radius 2 cm, 
    the slant height of the cone is 5 cm. -/
theorem cone_slant_height (lateral_area base_radius : ℝ) : 
  lateral_area = 10 * Real.pi ∧ base_radius = 2 → 
  lateral_area = (1 / 2) * (2 * Real.pi * base_radius) * 5 := by
sorry

end cone_slant_height_l338_33866


namespace distance_between_A_and_B_l338_33801

/-- Represents a person traveling from point A to B -/
structure Traveler where
  departureTime : ℕ  -- departure time in minutes after 8:00
  speed : ℝ          -- speed in meters per minute

/-- The problem setup -/
def travelProblem (v : ℝ) : Prop :=
  let personA : Traveler := ⟨0, v⟩
  let personB : Traveler := ⟨20, v⟩
  let personC : Traveler := ⟨30, v⟩
  let totalDistance : ℝ := 60 * v
  
  -- At 8:40 (40 minutes after 8:00), A's remaining distance is half of B's
  (totalDistance - 40 * v) = (1/2) * (totalDistance - 20 * v) ∧
  -- At 8:40, C is 2015 meters away from B
  (totalDistance - 10 * v) = 2015

theorem distance_between_A_and_B :
  ∃ v : ℝ, travelProblem v → 60 * v = 2418 :=
sorry

end distance_between_A_and_B_l338_33801


namespace eds_walking_speed_l338_33850

/-- Proves that Ed's walking speed is 4 km/h given the specified conditions -/
theorem eds_walking_speed (total_distance : ℝ) (sandys_speed : ℝ) (sandys_distance : ℝ) (time_difference : ℝ) :
  total_distance = 52 →
  sandys_speed = 6 →
  sandys_distance = 36 →
  time_difference = 2 →
  ∃ (eds_speed : ℝ), eds_speed = 4 :=
by sorry

end eds_walking_speed_l338_33850


namespace f_properties_l338_33892

def f (x m : ℝ) : ℝ := |x + 1| + |x + m + 1|

theorem f_properties (m : ℝ) :
  (∀ x, f x m ≥ |m - 2|) ↔ m ≥ 1 ∧
  (m ≤ 0 → ∀ x, ¬(f (-x) m < 2*m)) ∧
  (m > 0 → ∀ x, f (-x) m < 2*m ↔ 1 - m/2 < x ∧ x < 3*m/2 + 1) :=
sorry

end f_properties_l338_33892


namespace inequality_proof_l338_33848

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (2 * a^2 / (1 + a + a * b)^2 + 
   2 * b^2 / (1 + b + b * c)^2 + 
   2 * c^2 / (1 + c + c * a)^2 + 
   9 / ((1 + a + a * b) * (1 + b + b * c) * (1 + c + c * a))) ≥ 1 := by
  sorry

end inequality_proof_l338_33848


namespace bivalent_metal_relative_atomic_mass_l338_33826

-- Define the bivalent metal
structure BivalentMetal where
  relative_atomic_mass : ℝ

-- Define the reaction conditions
def hcl_moles : ℝ := 0.25

-- Define the reaction properties
def incomplete_reaction (m : BivalentMetal) : Prop :=
  3.5 / m.relative_atomic_mass > hcl_moles / 2

def complete_reaction (m : BivalentMetal) : Prop :=
  2.5 / m.relative_atomic_mass < hcl_moles / 2

-- Theorem to prove
theorem bivalent_metal_relative_atomic_mass :
  ∃ (m : BivalentMetal), 
    m.relative_atomic_mass = 24 ∧ 
    incomplete_reaction m ∧ 
    complete_reaction m :=
by
  sorry

end bivalent_metal_relative_atomic_mass_l338_33826


namespace smallest_dual_base_representation_l338_33873

def is_valid_representation (n : ℕ) (a b : ℕ) : Prop :=
  a > 2 ∧ b > 2 ∧ 2 * a + 1 = n ∧ b + 2 = n

theorem smallest_dual_base_representation : 
  (∃ (a b : ℕ), is_valid_representation 7 a b) ∧ 
  (∀ (n : ℕ), n < 7 → ¬∃ (a b : ℕ), is_valid_representation n a b) :=
sorry

end smallest_dual_base_representation_l338_33873
