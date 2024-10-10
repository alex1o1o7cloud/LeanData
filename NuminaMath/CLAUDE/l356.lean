import Mathlib

namespace simplify_expression_1_simplify_expression_2_l356_35695

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  2*x - (3*y - 5*x) + 7*y = 7*x + 4*y :=
by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) :
  (-x^2 + 4*x) - 3*x^2 + 2*(2*x^2 - 3*x) = -2*x :=
by sorry

end simplify_expression_1_simplify_expression_2_l356_35695


namespace black_faces_alignment_l356_35622

/-- Represents a cube with one black face and five white faces -/
structure Cube where
  blackFace : Fin 6

/-- Represents the 8x8 grid of cubes -/
def Grid := Fin 8 → Fin 8 → Cube

/-- Rotates all cubes in a given row -/
def rotateRow (g : Grid) (row : Fin 8) : Grid :=
  sorry

/-- Rotates all cubes in a given column -/
def rotateColumn (g : Grid) (col : Fin 8) : Grid :=
  sorry

/-- Checks if all cubes have their black faces pointing in the same direction -/
def allFacingSameDirection (g : Grid) : Prop :=
  sorry

/-- The main theorem stating that it's always possible to make all black faces point in the same direction -/
theorem black_faces_alignment (g : Grid) :
  ∃ (ops : List (Sum (Fin 8) (Fin 8))), 
    let finalGrid := ops.foldl (λ acc op => match op with
      | Sum.inl row => rotateRow acc row
      | Sum.inr col => rotateColumn acc col) g
    allFacingSameDirection finalGrid :=
  sorry

end black_faces_alignment_l356_35622


namespace special_function_characterization_l356_35606

/-- A function satisfying the given properties -/
def IsSpecialFunction (f : ℝ → ℝ) : Prop :=
  f 1 = 0 ∧ ∀ x y : ℝ, |f x - f y| = |x - y|

/-- The theorem stating that any function satisfying the given properties
    must be either x - 1 or 1 - x -/
theorem special_function_characterization (f : ℝ → ℝ) (hf : IsSpecialFunction f) :
  (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) := by
  sorry

end special_function_characterization_l356_35606


namespace marble_problem_solution_l356_35655

/-- Represents the number of marbles of each color in a box -/
structure MarbleBox where
  red : ℕ
  green : ℕ
  yellow : ℕ
  other : ℕ

/-- The conditions of the marble problem -/
def marble_problem (box : MarbleBox) : Prop :=
  box.red = 20 ∧
  box.green = 3 * box.red ∧
  box.yellow = box.green / 5 ∧
  box.red + box.green + box.yellow + box.other = 4 * box.green

/-- The theorem stating the solution to the marble problem -/
theorem marble_problem_solution (box : MarbleBox) :
  marble_problem box → box.other = 148 := by
  sorry


end marble_problem_solution_l356_35655


namespace stone_collision_distance_l356_35604

/-- The initial distance between two colliding stones -/
theorem stone_collision_distance (v₀ H g : ℝ) (h_v₀_pos : 0 < v₀) (h_H_pos : 0 < H) (h_g_pos : 0 < g) :
  let t := H / v₀
  let y₁ := H - (1/2) * g * t^2
  let y₂ := v₀ * t - (1/2) * g * t^2
  let x₁ := v₀ * t
  y₁ = y₂ →
  Real.sqrt (H^2 + x₁^2) = H * Real.sqrt 2 :=
by sorry

end stone_collision_distance_l356_35604


namespace full_price_revenue_is_3000_total_tickets_correct_total_revenue_correct_l356_35661

/-- Represents the revenue from full-price tickets at a concert venue. -/
def revenue_full_price (total_tickets : ℕ) (total_revenue : ℕ) : ℕ :=
  let p : ℕ := 20  -- Price of a full-price ticket
  let f : ℕ := 150 -- Number of full-price tickets
  f * p

/-- Theorem stating that the revenue from full-price tickets is $3000. -/
theorem full_price_revenue_is_3000 :
  revenue_full_price 250 3500 = 3000 := by
  sorry

/-- Verifies that the total number of tickets is correct. -/
theorem total_tickets_correct (f h q : ℕ) :
  f + h + q = 250 := by
  sorry

/-- Verifies that the total revenue is correct. -/
theorem total_revenue_correct (f h q p : ℕ) :
  f * p + h * (p / 2) + q * (p / 4) = 3500 := by
  sorry

end full_price_revenue_is_3000_total_tickets_correct_total_revenue_correct_l356_35661


namespace existence_of_m_l356_35607

/-- The number of factors of 2 in m! -/
def n (m : ℕ) : ℕ := sorry

/-- Theorem stating the existence of m satisfying the given conditions -/
theorem existence_of_m : ∃ m : ℕ, m > 1990^1990 ∧ m = 3^1990 + n m := by sorry

end existence_of_m_l356_35607


namespace fiftieth_term_is_198_l356_35652

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
theorem fiftieth_term_is_198 :
  arithmeticSequenceTerm 2 4 50 = 198 := by
  sorry

end fiftieth_term_is_198_l356_35652


namespace probability_of_all_even_sums_l356_35672

/-- Represents a tile with a number from 1 to 9 -/
def Tile := Fin 9

/-- Represents a player's selection of three tiles -/
def Selection := Fin 3 → Tile

/-- The set of all possible selections -/
def AllSelections := Fin 3 → Selection

/-- Checks if a selection results in an even sum -/
def isEvenSum (s : Selection) : Prop :=
  ∃ (a b c : Tile), s 0 = a ∧ s 1 = b ∧ s 2 = c ∧ (a.val + b.val + c.val) % 2 = 0

/-- Checks if all three players have even sums -/
def allEvenSums (selections : AllSelections) : Prop :=
  ∀ i : Fin 3, isEvenSum (selections i)

/-- The total number of possible ways to distribute the tiles -/
def totalOutcomes : ℕ := 1680

/-- The number of favorable outcomes where all players have even sums -/
def favorableOutcomes : ℕ := 400

theorem probability_of_all_even_sums :
  (favorableOutcomes : ℚ) / totalOutcomes = 5 / 21 := by sorry

end probability_of_all_even_sums_l356_35672


namespace danielles_rooms_l356_35667

theorem danielles_rooms (heidi grant danielle : ℕ) : 
  heidi = 3 * danielle →
  grant = heidi / 9 →
  grant = 2 →
  danielle = 6 := by
sorry

end danielles_rooms_l356_35667


namespace right_rectangular_prism_volume_l356_35605

theorem right_rectangular_prism_volume 
  (face_area1 face_area2 face_area3 : ℝ) 
  (h1 : face_area1 = 30)
  (h2 : face_area2 = 45)
  (h3 : face_area3 = 75) :
  ∃ (x y z : ℝ), 
    x * y = face_area1 ∧ 
    x * z = face_area2 ∧ 
    y * z = face_area3 ∧ 
    x * y * z = 150 := by
  sorry

end right_rectangular_prism_volume_l356_35605


namespace function_monotonic_decreasing_l356_35624

/-- The function f(x) = 3x^2 - 2ln(x) is monotonically decreasing on the interval (0, √3/3) -/
theorem function_monotonic_decreasing (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 2 * Real.log x
  0 < x → x < Real.sqrt 3 / 3 → StrictMonoOn f (Set.Ioo 0 (Real.sqrt 3 / 3)) := by
  sorry

end function_monotonic_decreasing_l356_35624


namespace complement_of_intersection_l356_35656

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 2, 3}

-- Define set B
def B : Set ℕ := {2, 3, 4}

-- Theorem statement
theorem complement_of_intersection :
  (Aᶜ : Set ℕ) ∩ B = {1, 4, 5} := by sorry

end complement_of_intersection_l356_35656


namespace percentage_problem_l356_35636

theorem percentage_problem (N : ℝ) (P : ℝ) 
  (h1 : 0.3 * N = 120) 
  (h2 : (P / 100) * N = 160) : 
  P = 40 := by
  sorry

end percentage_problem_l356_35636


namespace inequality_solution_l356_35693

theorem inequality_solution (x : ℝ) : 
  x ≠ 0 → (x > (9 : ℝ) / x ↔ (x > -3 ∧ x < 0) ∨ x > 3) := by
  sorry

end inequality_solution_l356_35693


namespace households_with_only_bike_l356_35642

/-- Proves the number of households with only a bike given the provided information -/
theorem households_with_only_bike
  (total_households : ℕ)
  (households_without_car_or_bike : ℕ)
  (households_with_both_car_and_bike : ℕ)
  (households_with_car : ℕ)
  (h1 : total_households = 90)
  (h2 : households_without_car_or_bike = 11)
  (h3 : households_with_both_car_and_bike = 22)
  (h4 : households_with_car = 44) :
  total_households - households_without_car_or_bike - households_with_car - households_with_both_car_and_bike = 35 :=
by sorry

end households_with_only_bike_l356_35642


namespace gcd_lcm_sum_8_12_l356_35612

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by sorry

end gcd_lcm_sum_8_12_l356_35612


namespace geometric_sequence_property_l356_35678

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 16 = -6 ∧ a 2 * a 16 = 2) →
  (a 2 * a 16) / a 9 = -Real.sqrt 2 := by
  sorry

end geometric_sequence_property_l356_35678


namespace max_elevation_l356_35692

/-- The elevation function of a particle projected vertically upward -/
def s (t : ℝ) : ℝ := 200 * t - 20 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ (t : ℝ), ∀ (t' : ℝ), s t' ≤ s t ∧ s t = 500 := by
  sorry

end max_elevation_l356_35692


namespace odd_sum_representation_l356_35608

theorem odd_sum_representation (a b : ℤ) (h : Odd (a + b)) :
  ∀ n : ℤ, ∃ x y : ℤ, n = x^2 - y^2 + a*x + b*y :=
by sorry

end odd_sum_representation_l356_35608


namespace selena_leftover_money_l356_35668

def meal_cost (tip steak_price steak_count burger_price burger_count
               ice_cream_price ice_cream_count steak_tax burger_tax ice_cream_tax : ℝ) : ℝ :=
  let steak_total := steak_price * steak_count * (1 + steak_tax)
  let burger_total := burger_price * burger_count * (1 + burger_tax)
  let ice_cream_total := ice_cream_price * ice_cream_count * (1 + ice_cream_tax)
  steak_total + burger_total + ice_cream_total

theorem selena_leftover_money :
  let tip := 99
  let steak_price := 24
  let steak_count := 2
  let burger_price := 3.5
  let burger_count := 2
  let ice_cream_price := 2
  let ice_cream_count := 3
  let steak_tax := 0.07
  let burger_tax := 0.06
  let ice_cream_tax := 0.08
  tip - meal_cost tip steak_price steak_count burger_price burger_count
                  ice_cream_price ice_cream_count steak_tax burger_tax ice_cream_tax = 33.74 := by
  sorry

end selena_leftover_money_l356_35668


namespace minutes_on_eleventh_day_l356_35670

/-- The number of minutes Gage skated each day for the first 6 days -/
def minutes_per_day_first_6 : ℕ := 80

/-- The number of minutes Gage skated each day for the next 4 days -/
def minutes_per_day_next_4 : ℕ := 95

/-- The total number of days Gage has skated -/
def total_days : ℕ := 10

/-- The desired average number of minutes per day -/
def desired_average : ℕ := 90

/-- The total number of days including the day in question -/
def total_days_with_extra : ℕ := total_days + 1

/-- Theorem stating the number of minutes Gage must skate on the eleventh day -/
theorem minutes_on_eleventh_day :
  (total_days_with_extra * desired_average) - (6 * minutes_per_day_first_6 + 4 * minutes_per_day_next_4) = 130 := by
  sorry

end minutes_on_eleventh_day_l356_35670


namespace mascot_costs_l356_35623

/-- The cost of an Auspicious Mascot Plush Toy -/
def plush_toy_cost : ℝ := 80

/-- The cost of an Auspicious Mascot Metal Ornament -/
def metal_ornament_cost : ℝ := 100

/-- The total cost of Plush Toys purchased -/
def plush_toy_total : ℝ := 6400

/-- The total cost of Metal Ornaments purchased -/
def metal_ornament_total : ℝ := 4000

theorem mascot_costs :
  (metal_ornament_cost = plush_toy_cost + 20) ∧
  (plush_toy_total / plush_toy_cost = 2 * (metal_ornament_total / metal_ornament_cost)) ∧
  (plush_toy_cost = 80) ∧
  (metal_ornament_cost = 100) := by
  sorry

end mascot_costs_l356_35623


namespace geometric_sequence_ratio_l356_35645

/-- Given a geometric sequence {a_n} with common ratio q > 1,
    if 2a₁, (3/2)a₂, and a₃ form an arithmetic sequence,
    then S₄/a₄ = 15/8, where S₄ is the sum of the first 4 terms. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence condition
  q > 1 →  -- Common ratio > 1
  (2 * a 1 - (3/2 * a 2) = (3/2 * a 2) - a 3) →  -- Arithmetic sequence condition
  (a 1 * (1 - q^4) / (1 - q)) / (a 1 * q^3) = 15/8 := by
  sorry

end geometric_sequence_ratio_l356_35645


namespace point_on_line_iff_vector_sum_l356_35684

/-- 
Given distinct points A and B, and a point O not on line AB,
a point C is on line AB if and only if there exist real numbers x and y 
such that OC = x * OA + y * OB and x + y = 1.
-/
theorem point_on_line_iff_vector_sum (O A B C : EuclideanSpace ℝ (Fin 3)) 
  (hAB : A ≠ B) (hO : O ∉ affineSpan ℝ {A, B}) : 
  C ∈ affineSpan ℝ {A, B} ↔ 
  ∃ (x y : ℝ), C - O = x • (A - O) + y • (B - O) ∧ x + y = 1 := by
  sorry

end point_on_line_iff_vector_sum_l356_35684


namespace kata_friends_and_money_l356_35677

/-- The number of friends Káta has -/
def n : ℕ := sorry

/-- The amount of money Káta has for gifts (in Kč) -/
def x : ℕ := sorry

/-- The cost of a hair clip (in Kč) -/
def hair_clip_cost : ℕ := 28

/-- The cost of a teddy bear (in Kč) -/
def teddy_bear_cost : ℕ := 42

/-- The amount left after buying hair clips (in Kč) -/
def hair_clip_remainder : ℕ := 29

/-- The amount short after buying teddy bears (in Kč) -/
def teddy_bear_shortage : ℕ := 13

theorem kata_friends_and_money :
  (n * hair_clip_cost + hair_clip_remainder = x) ∧
  (n * teddy_bear_cost - teddy_bear_shortage = x) →
  n = 3 ∧ x = 113 := by
  sorry

end kata_friends_and_money_l356_35677


namespace distance_AB_is_abs_x_minus_one_l356_35660

/-- Represents a point on a number line with a rational coordinate -/
structure Point where
  coord : ℚ

/-- Calculates the distance between two points on a number line -/
def distance (p q : Point) : ℚ :=
  |p.coord - q.coord|

theorem distance_AB_is_abs_x_minus_one (x : ℚ) :
  let A : Point := ⟨x⟩
  let B : Point := ⟨1⟩
  let C : Point := ⟨-1⟩
  distance A B = |x - 1| := by
  sorry

end distance_AB_is_abs_x_minus_one_l356_35660


namespace ferris_wheel_cost_l356_35627

/-- The cost of attractions at an amusement park. -/
structure AttractionCosts where
  roller_coaster : ℕ
  log_ride : ℕ
  ferris_wheel : ℕ

/-- The number of tickets Antonieta has and needs. -/
structure AntonietaTickets where
  current : ℕ
  needed : ℕ

/-- Theorem stating the cost of the Ferris wheel given the other costs and ticket information. -/
theorem ferris_wheel_cost (costs : AttractionCosts) (antonieta : AntonietaTickets) : 
  costs.roller_coaster = 5 →
  costs.log_ride = 7 →
  antonieta.current = 2 →
  antonieta.needed = 16 →
  costs.ferris_wheel = 6 := by
  sorry

end ferris_wheel_cost_l356_35627


namespace area_of_region_l356_35639

/-- The area enclosed by the region defined by x^2 + y^2 - 4x + 2y = -2 is 3π -/
theorem area_of_region (x y : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 4*x + 2*y = -2) → 
  (∃ (center : ℝ × ℝ) (r : ℝ), 
    center = (2, -1) ∧ 
    r = Real.sqrt 3 ∧ 
    (∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2) ∧
    π * r^2 = 3 * π) :=
by sorry

end area_of_region_l356_35639


namespace notebook_pen_cost_l356_35628

theorem notebook_pen_cost :
  ∀ (n p : ℕ),
  15 * n + 4 * p = 160 →
  n > p →
  n + p = 18 :=
by
  sorry

end notebook_pen_cost_l356_35628


namespace penny_money_left_l356_35663

/-- Calculates the amount of money Penny has left after purchasing socks and a hat. -/
def money_left (initial_amount : ℝ) (num_sock_pairs : ℕ) (sock_pair_cost : ℝ) (hat_cost : ℝ) : ℝ :=
  initial_amount - (num_sock_pairs * sock_pair_cost + hat_cost)

/-- Proves that Penny has $5 left after her purchases. -/
theorem penny_money_left :
  money_left 20 4 2 7 = 5 := by
  sorry

end penny_money_left_l356_35663


namespace four_digit_no_repeat_count_five_digit_no_repeat_div_by_5_count_l356_35625

/-- The count of four-digit numbers with no repeated digits -/
def fourDigitNoRepeat : Nat :=
  5 * 4 * 3 * 2

/-- The count of five-digit numbers with no repeated digits and divisible by 5 -/
def fiveDigitNoRepeatDivBy5 : Nat :=
  2 * (4 * 4 * 3 * 2)

theorem four_digit_no_repeat_count :
  fourDigitNoRepeat = 120 := by sorry

theorem five_digit_no_repeat_div_by_5_count :
  fiveDigitNoRepeatDivBy5 = 216 := by sorry

end four_digit_no_repeat_count_five_digit_no_repeat_div_by_5_count_l356_35625


namespace imaginary_part_of_z_l356_35601

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 3 + 7 * Complex.I) : 
  z.im = 5 := by
  sorry

end imaginary_part_of_z_l356_35601


namespace three_color_theorem_l356_35683

/-- A complete graph with n vertices -/
def CompleteGraph (n : ℕ) := { v : Fin n // True }

/-- An edge in a complete graph connects any two distinct vertices -/
def Edge (n : ℕ) := { e : CompleteGraph n × CompleteGraph n // e.1 ≠ e.2 }

/-- A coloring assignment for vertices -/
def Coloring (n : ℕ) := CompleteGraph n → Fin 3

/-- A valid coloring ensures no two adjacent vertices have the same color -/
def ValidColoring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (e : Edge n), c e.1.1 ≠ c e.1.2

theorem three_color_theorem (n : ℕ) (h : n ≥ 2) :
  ∃ (c : Coloring n), ValidColoring n c :=
sorry

end three_color_theorem_l356_35683


namespace quadratic_roots_problem_l356_35633

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - k*x₁ - 4 = 0 →
  x₂^2 - k*x₂ - 4 = 0 →
  x₁^2 + x₂^2 + x₁*x₂ = 6 →
  k^2 = 2 := by
sorry

end quadratic_roots_problem_l356_35633


namespace rectangle_area_l356_35658

-- Define the points
variable (P Q R S T U : Point)

-- Define the rectangle PQRS
def is_rectangle (P Q R S : Point) : Prop := sorry

-- Define the trisection of angle S
def trisects_angle (S T U : Point) : Prop := sorry

-- Define that T is on PQ
def point_on_line (T P Q : Point) : Prop := sorry

-- Define that U is on PS
def point_on_line_2 (U P S : Point) : Prop := sorry

-- Define the length of QT
def length_QT (Q T : Point) : ℝ := sorry

-- Define the length of PU
def length_PU (P U : Point) : ℝ := sorry

-- Define the area of a rectangle
def area_rectangle (P Q R S : Point) : ℝ := sorry

-- Theorem statement
theorem rectangle_area (P Q R S T U : Point) 
  (h1 : is_rectangle P Q R S)
  (h2 : trisects_angle S T U)
  (h3 : point_on_line T P Q)
  (h4 : point_on_line_2 U P S)
  (h5 : length_QT Q T = 8)
  (h6 : length_PU P U = 4) :
  area_rectangle P Q R S = 64 * Real.sqrt 3 := by
  sorry

end rectangle_area_l356_35658


namespace orange_harvest_days_l356_35694

/-- The number of days required to harvest a given number of sacks of ripe oranges -/
def harvest_days (total_sacks : ℕ) (sacks_per_day : ℕ) : ℕ :=
  total_sacks / sacks_per_day

/-- Theorem stating that it takes 25 days to harvest 2050 sacks of ripe oranges when harvesting 82 sacks per day -/
theorem orange_harvest_days : harvest_days 2050 82 = 25 := by
  sorry

end orange_harvest_days_l356_35694


namespace part_one_part_two_l356_35640

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 4|

-- Part I
theorem part_one : 
  ∀ x : ℝ, (f 1 x ≤ 2 * |x - 4|) ↔ (x < 1.5) := by sorry

-- Part II
theorem part_two : 
  (∀ x : ℝ, f a x ≥ 3) ↔ (a ≤ -7 ∨ a ≥ -1) := by sorry

end part_one_part_two_l356_35640


namespace placemat_length_l356_35621

theorem placemat_length (R : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) :
  R = 5 →
  n = 8 →
  w = 2 →
  y = 2 * R * Real.sin (π / (2 * n)) →
  y = 5 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end placemat_length_l356_35621


namespace unique_prime_seven_digit_number_l356_35629

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def seven_digit_number (B : ℕ) : ℕ := 2024050 + B

theorem unique_prime_seven_digit_number :
  ∃! B : ℕ, B < 10 ∧ is_prime (seven_digit_number B) ∧ seven_digit_number B = 2024051 :=
sorry

end unique_prime_seven_digit_number_l356_35629


namespace sum_and_fraction_difference_l356_35600

theorem sum_and_fraction_difference (x y : ℝ) 
  (h1 : x + y = 480) 
  (h2 : x / y = 0.8) : 
  y - x = 53.34 := by sorry

end sum_and_fraction_difference_l356_35600


namespace base_conversion_645_to_base_5_l356_35664

theorem base_conversion_645_to_base_5 :
  (1 * 5^4 + 0 * 5^3 + 4 * 5^1 + 0 * 5^0 : ℕ) = 645 := by
  sorry

end base_conversion_645_to_base_5_l356_35664


namespace problem_statement_l356_35650

theorem problem_statement (s x y : ℝ) 
  (h1 : s > 0) 
  (h2 : s ≠ 1) 
  (h3 : x * y ≠ 0) 
  (h4 : s * x > y) : 
  s < y / x := by
  sorry

end problem_statement_l356_35650


namespace triangle_is_equilateral_l356_35676

/-- Triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- A triangle is equilateral if all its sides are equal -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- The given conditions for the triangle -/
def satisfiesConditions (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.b + t.c - t.a) = 3 * t.b * t.c ∧
  Real.sin t.A = 2 * Real.sin t.B * Real.cos t.C

/-- Theorem: If a triangle satisfies the given conditions, then it is equilateral -/
theorem triangle_is_equilateral (t : Triangle) 
  (h : satisfiesConditions t) : t.isEquilateral := by
  sorry

end triangle_is_equilateral_l356_35676


namespace paper_folding_thickness_l356_35687

/-- The thickness of a folded paper after a given number of folds. -/
def thickness (initial_thickness : ℝ) (num_folds : ℕ) : ℝ :=
  initial_thickness * (2 ^ num_folds)

/-- Theorem stating that folding a 0.1 mm thick paper 5 times results in 3.2 mm thickness. -/
theorem paper_folding_thickness :
  thickness 0.1 5 = 3.2 := by
  sorry

end paper_folding_thickness_l356_35687


namespace willies_stickers_l356_35618

theorem willies_stickers (starting_stickers : Real) (received_stickers : Real) :
  starting_stickers = 36.0 →
  received_stickers = 7.0 →
  starting_stickers + received_stickers = 43.0 := by
  sorry

end willies_stickers_l356_35618


namespace divisibility_by_square_of_n_minus_one_l356_35690

theorem divisibility_by_square_of_n_minus_one (n : ℕ) (h : n > 2) :
  ∃ k : ℤ, (n : ℤ)^(n - 1) - 1 = k * (n - 1)^2 := by
  sorry

end divisibility_by_square_of_n_minus_one_l356_35690


namespace tan_1500_deg_l356_35626

theorem tan_1500_deg (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1500 * π / 180) → n = 60 := by
  sorry

end tan_1500_deg_l356_35626


namespace cone_lateral_area_l356_35634

/-- The lateral area of a cone with a central angle of 60° and base radius of 8 is 384π. -/
theorem cone_lateral_area : 
  ∀ (r : ℝ) (central_angle : ℝ) (lateral_area : ℝ),
  r = 8 →
  central_angle = 60 * π / 180 →
  lateral_area = π * r * (2 * π * r) / central_angle →
  lateral_area = 384 * π :=
by
  sorry

end cone_lateral_area_l356_35634


namespace sector_perimeter_example_l356_35651

/-- The perimeter of a sector with given radius and central angle -/
def sector_perimeter (r : ℝ) (θ : ℝ) : ℝ := r * θ + 2 * r

/-- Theorem: The perimeter of a sector with radius 1.5 and central angle 2 radians is 6 -/
theorem sector_perimeter_example : sector_perimeter 1.5 2 = 6 := by
  sorry

end sector_perimeter_example_l356_35651


namespace gcd_888_1147_l356_35682

theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end gcd_888_1147_l356_35682


namespace largest_prime_divisor_test_l356_35616

theorem largest_prime_divisor_test (n : ℕ) : 
  1000 ≤ n → n ≤ 1050 → 
  (∀ p : ℕ, p ≤ 31 → Nat.Prime p → ¬(p ∣ n)) → 
  Nat.Prime n :=
sorry

end largest_prime_divisor_test_l356_35616


namespace polynomial_evaluation_l356_35671

theorem polynomial_evaluation (x : ℤ) (h : x = -2) : x^3 - x^2 + x - 1 = -15 := by
  sorry

end polynomial_evaluation_l356_35671


namespace fraction_equality_implies_values_l356_35644

theorem fraction_equality_implies_values (A B : ℚ) :
  (∀ x : ℚ, x ≠ 5 ∧ x ≠ -2 → A / (x - 5) + B / (x + 2) = (5 * x - 4) / (x^2 - 3*x - 10)) →
  A = 3 ∧ B = 2 := by
  sorry

end fraction_equality_implies_values_l356_35644


namespace cafe_chair_distribution_l356_35691

/-- Given a cafe with indoor and outdoor tables, prove the number of chairs at each indoor table. -/
theorem cafe_chair_distribution (indoor_tables outdoor_tables : ℕ) 
  (chairs_per_outdoor_table : ℕ) (total_chairs : ℕ) :
  indoor_tables = 9 →
  outdoor_tables = 11 →
  chairs_per_outdoor_table = 3 →
  total_chairs = 123 →
  ∃ (chairs_per_indoor_table : ℕ),
    chairs_per_indoor_table = 10 ∧
    total_chairs = indoor_tables * chairs_per_indoor_table + outdoor_tables * chairs_per_outdoor_table :=
by
  sorry

end cafe_chair_distribution_l356_35691


namespace jan_skips_proof_l356_35615

/-- Calculates the total number of skips in a given time period after doubling the initial speed -/
def total_skips (initial_speed : ℕ) (time : ℕ) : ℕ :=
  2 * initial_speed * time

/-- Proves that given an initial speed of 70 skips per minute, which doubles after training,
    the total number of skips in 5 minutes is equal to 700 -/
theorem jan_skips_proof :
  total_skips 70 5 = 700 := by
  sorry

#eval total_skips 70 5

end jan_skips_proof_l356_35615


namespace nested_root_simplification_l356_35680

theorem nested_root_simplification (b : ℝ) :
  (((b^16)^(1/8))^(1/4))^6 * (((b^16)^(1/4))^(1/8))^6 = b^6 := by
  sorry

end nested_root_simplification_l356_35680


namespace product_expansion_l356_35647

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end product_expansion_l356_35647


namespace optimal_departure_time_l356_35665

/-- The travel scenario for two people A and B -/
structure TravelScenario where
  distance : ℝ
  walking_speed : ℝ
  travel_speed : ℝ
  walking_distance : ℝ

/-- The theorem stating the optimal departure time -/
theorem optimal_departure_time (scenario : TravelScenario) 
  (h1 : scenario.distance = 15)
  (h2 : scenario.walking_speed = 1)
  (h3 : scenario.travel_speed > scenario.walking_speed)
  (h4 : scenario.walking_distance = 60 / 11)
  (h5 : scenario.travel_speed * (3 / 11) = scenario.distance - scenario.walking_distance) :
  3 / 11 = scenario.walking_distance / scenario.walking_speed := by
  sorry

#check optimal_departure_time

end optimal_departure_time_l356_35665


namespace license_plate_equality_l356_35673

def florida_plates : ℕ := 26^2 * 10^3 * 26^1
def north_dakota_plates : ℕ := 26^3 * 10^3

theorem license_plate_equality :
  florida_plates = north_dakota_plates :=
by sorry

end license_plate_equality_l356_35673


namespace second_number_is_30_l356_35675

theorem second_number_is_30 (x y : ℤ) : 
  y = x + 4 →  -- The second number is 4 more than the first
  x + y = 56 → -- The sum of the two numbers is 56
  y = 30       -- The second number is 30
:= by sorry

end second_number_is_30_l356_35675


namespace negation_of_implication_l356_35649

theorem negation_of_implication (p q : Prop) : ¬(p → q) ↔ (p ∧ ¬q) := by sorry

end negation_of_implication_l356_35649


namespace valid_pairs_characterization_l356_35696

def is_single_digit (n : ℕ) : Prop := 1 < n ∧ n < 10

def product_contains_factor (a b : ℕ) : Prop :=
  let product := a * b
  (product % 10 = a ∨ product % 10 = b ∨ product / 10 = a ∨ product / 10 = b)

def valid_pairs : Set (ℕ × ℕ) :=
  {p | is_single_digit p.1 ∧ is_single_digit p.2 ∧ product_contains_factor p.1 p.2}

theorem valid_pairs_characterization :
  valid_pairs = {(5, 3), (5, 5), (5, 7), (5, 9), (6, 2), (6, 4), (6, 6), (6, 8)} := by sorry

end valid_pairs_characterization_l356_35696


namespace pens_distribution_l356_35654

def number_of_friends (kendra_packs tony_packs pens_per_pack pens_kept_per_person : ℕ) : ℕ :=
  (kendra_packs + tony_packs) * pens_per_pack - 2 * pens_kept_per_person

theorem pens_distribution (kendra_packs tony_packs pens_per_pack pens_kept_per_person : ℕ) 
  (h1 : kendra_packs = 4)
  (h2 : tony_packs = 2)
  (h3 : pens_per_pack = 3)
  (h4 : pens_kept_per_person = 2) :
  number_of_friends kendra_packs tony_packs pens_per_pack pens_kept_per_person = 14 := by
  sorry

end pens_distribution_l356_35654


namespace john_percentage_increase_l356_35632

/-- Represents the data for John's work at two hospitals --/
structure HospitalData where
  patients_first_hospital : ℕ  -- patients per day at first hospital
  total_patients_per_year : ℕ  -- total patients per year
  days_per_week : ℕ           -- working days per week
  weeks_per_year : ℕ          -- working weeks per year

/-- Calculates the percentage increase in patients at the second hospital compared to the first --/
def percentage_increase (data : HospitalData) : ℚ :=
  let total_working_days := data.days_per_week * data.weeks_per_year
  let patients_second_hospital := (data.total_patients_per_year - data.patients_first_hospital * total_working_days) / total_working_days
  ((patients_second_hospital - data.patients_first_hospital) / data.patients_first_hospital) * 100

/-- Theorem stating that given John's work conditions, the percentage increase is 20% --/
theorem john_percentage_increase :
  let john_data : HospitalData := {
    patients_first_hospital := 20,
    total_patients_per_year := 11000,
    days_per_week := 5,
    weeks_per_year := 50
  }
  percentage_increase john_data = 20 := by
  sorry


end john_percentage_increase_l356_35632


namespace trent_tadpoles_l356_35666

theorem trent_tadpoles (caught : ℕ) (kept : ℕ) (released_percentage : ℚ) : 
  released_percentage = 75 / 100 →
  kept = 45 →
  kept = (1 - released_percentage) * caught →
  caught = 180 :=
by sorry

end trent_tadpoles_l356_35666


namespace sqrt_product_plus_one_l356_35611

theorem sqrt_product_plus_one : 
  Real.sqrt (31 * 30 * 29 * 28 + 1) = 869 := by
sorry

end sqrt_product_plus_one_l356_35611


namespace solution_set_f_positive_solution_set_f_leq_g_l356_35688

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + (2 - m) * x - m

/-- The function g(x) defined in the problem -/
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - x + 2 * m

/-- Theorem for part (1) of the problem -/
theorem solution_set_f_positive :
  {x : ℝ | f 1 x > 0} = {x : ℝ | x > 1/2 ∨ x < -1} := by sorry

/-- Theorem for part (2) of the problem -/
theorem solution_set_f_leq_g (m : ℝ) (h : m > 0) :
  {x : ℝ | f m x ≤ g m x} = {x : ℝ | -3 ≤ x ∧ x ≤ m} := by sorry

end solution_set_f_positive_solution_set_f_leq_g_l356_35688


namespace odd_function_extension_l356_35685

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : ∀ x < 0, f x = 1 - Real.exp (-x + 1)) :
  ∀ x > 0, f x = Real.exp (x + 1) - 1 := by
sorry

end odd_function_extension_l356_35685


namespace function_property_l356_35681

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y)
  (h2 : f 1000 = 6) : 
  f 800 = 7.5 := by
sorry

end function_property_l356_35681


namespace two_by_one_prism_net_removable_squares_l356_35657

/-- Represents a rectangular prism --/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a net of a rectangular prism --/
structure PrismNet where
  squares : ℕ

/-- Function to create a net from a rectangular prism --/
def createNet (prism : RectangularPrism) : PrismNet :=
  { squares := 2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height) }

/-- Function to count removable squares in a net --/
def countRemovableSquares (net : PrismNet) : ℕ := sorry

/-- Theorem stating that a 2×1×1 prism net has exactly 5 removable squares --/
theorem two_by_one_prism_net_removable_squares :
  let prism : RectangularPrism := { length := 2, width := 1, height := 1 }
  let net := createNet prism
  countRemovableSquares net = 5 ∧ net.squares - 1 = 9 := by sorry

end two_by_one_prism_net_removable_squares_l356_35657


namespace sqrt_23_squared_minus_one_l356_35619

theorem sqrt_23_squared_minus_one : (Real.sqrt 23 - 1) * (Real.sqrt 23 + 1) = 22 := by
  sorry

end sqrt_23_squared_minus_one_l356_35619


namespace correct_num_students_l356_35641

/-- The number of students in the class -/
def num_students : ℕ := 20

/-- The cost of one pack of instant noodles in yuan -/
def noodle_cost : ℚ := 3.5

/-- The cost of one sausage in yuan -/
def sausage_cost : ℚ := 7.5

/-- The total amount spent in yuan -/
def total_spent : ℚ := 290

/-- Theorem stating that the number of students is correct given the problem conditions -/
theorem correct_num_students :
  (num_students : ℚ) * (2 * noodle_cost + sausage_cost) = total_spent :=
by sorry

end correct_num_students_l356_35641


namespace divisible_by_five_l356_35630

theorem divisible_by_five (a b : ℕ) : 
  (∃ k : ℕ, a * b = 5 * k) → (∃ m : ℕ, a = 5 * m) ∨ (∃ n : ℕ, b = 5 * n) :=
by sorry

end divisible_by_five_l356_35630


namespace parabola_equation_l356_35638

/-- A parabola with focus F and a point M satisfying given conditions -/
structure Parabola where
  p : ℝ
  F : ℝ × ℝ
  M : ℝ × ℝ
  h_p_pos : p > 0
  h_F_focus : F = (p/2, 0)
  h_M_on_C : M.2^2 = 2*p*M.1
  h_M_x : M.1 = 4
  h_MF_dist : Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = 5

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation (C : Parabola) : C.M.2^2 = 18 * C.M.1 := by
  sorry

end parabola_equation_l356_35638


namespace jeff_payment_correct_l356_35614

/-- The amount Jeff paid when picking up the Halloween costumes -/
def jeff_payment (num_costumes : ℕ) 
                 (deposit_rate : ℝ) 
                 (price_increase : ℝ) 
                 (last_year_price : ℝ) 
                 (jeff_discount : ℝ) 
                 (friend_discount : ℝ) : ℝ :=
  let this_year_price := last_year_price * (1 + price_increase)
  let total_cost := num_costumes * this_year_price
  let discounts := jeff_discount * this_year_price + friend_discount * this_year_price
  let adjusted_cost := total_cost - discounts
  let deposit := deposit_rate * adjusted_cost
  adjusted_cost - deposit

/-- Theorem stating that Jeff's payment matches the calculated amount -/
theorem jeff_payment_correct : 
  jeff_payment 3 0.1 0.4 250 0.15 0.1 = 866.25 := by
  sorry


end jeff_payment_correct_l356_35614


namespace jakes_peaches_l356_35659

theorem jakes_peaches (steven jill jake : ℕ) : 
  steven = 16 →
  jake < steven →
  jake = jill + 9 →
  ∃ (l : ℕ), jake = l + 9 :=
by sorry

end jakes_peaches_l356_35659


namespace factorial_product_less_than_factorial_sum_l356_35689

theorem factorial_product_less_than_factorial_sum {n : ℕ} (k : ℕ) (a : Fin n → ℕ) 
  (h_pos : ∀ i, a i > 0) (h_sum : (Finset.univ.sum a) < k) : 
  (Finset.univ.prod (λ i => Nat.factorial (a i))) < Nat.factorial k := by
sorry

end factorial_product_less_than_factorial_sum_l356_35689


namespace ladder_problem_l356_35698

theorem ladder_problem (ladder_length height : Real) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : Real), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end ladder_problem_l356_35698


namespace f_f_neg_two_equals_one_l356_35653

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -2 then x + 2
  else if x < 3 then 2^x
  else Real.log x

-- State the theorem
theorem f_f_neg_two_equals_one : f (f (-2)) = 1 := by
  sorry

end f_f_neg_two_equals_one_l356_35653


namespace soft_drink_bottles_sold_l356_35602

theorem soft_drink_bottles_sold (small_bottles : ℕ) (big_bottles : ℕ) 
  (small_sold_percent : ℚ) (total_remaining : ℕ) : 
  small_bottles = 6000 →
  big_bottles = 14000 →
  small_sold_percent = 1/5 →
  total_remaining = 15580 →
  (big_bottles - (total_remaining - (small_bottles - small_bottles * small_sold_percent))) / big_bottles = 23/100 := by
  sorry

end soft_drink_bottles_sold_l356_35602


namespace private_schools_in_district_a_l356_35669

/-- Represents the number of schools in Veenapaniville -/
def total_schools : Nat := 50

/-- Represents the number of public schools -/
def public_schools : Nat := 25

/-- Represents the number of parochial schools -/
def parochial_schools : Nat := 16

/-- Represents the number of private independent schools -/
def private_schools : Nat := 9

/-- Represents the number of schools in District A -/
def district_a_schools : Nat := 18

/-- Represents the number of schools in District B -/
def district_b_schools : Nat := 17

/-- Represents the number of private independent schools in District B -/
def district_b_private_schools : Nat := 2

/-- Theorem stating that the number of private independent schools in District A is 2 -/
theorem private_schools_in_district_a :
  total_schools = public_schools + parochial_schools + private_schools →
  district_a_schools + district_b_schools < total_schools →
  (total_schools - district_a_schools - district_b_schools) % 3 = 0 →
  private_schools - district_b_private_schools - (total_schools - district_a_schools - district_b_schools) / 3 = 2 := by
  sorry

end private_schools_in_district_a_l356_35669


namespace equation_solutions_l356_35686

def f (x : ℝ) : ℝ := (x - 4)^6 + (x - 6)^6

theorem equation_solutions :
  {x : ℝ | f x = 64} = {4, 6} := by sorry

end equation_solutions_l356_35686


namespace smallest_n_congruence_l356_35648

theorem smallest_n_congruence (n : ℕ) : n = 7 ↔ 
  (n > 0 ∧ 5 * n % 26 = 789 % 26 ∧ ∀ m : ℕ, m > 0 → m < n → 5 * m % 26 ≠ 789 % 26) :=
by sorry

end smallest_n_congruence_l356_35648


namespace regular_polygon_properties_l356_35631

theorem regular_polygon_properties (exterior_angle : ℝ) 
  (h1 : exterior_angle = 15)
  (h2 : exterior_angle > 0) :
  ∃ (n : ℕ) (sum_interior : ℝ),
    n = 24 ∧ 
    sum_interior = 3960 ∧
    n * exterior_angle = 360 ∧
    sum_interior = 180 * (n - 2) :=
by sorry

end regular_polygon_properties_l356_35631


namespace perpendicular_planes_l356_35699

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (not_parallel : Line → Line → Prop)
variable (line_not_parallel_plane : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (m n : Line) (α β γ : Plane)
  (different_lines : m ≠ n)
  (different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h1 : perpendicular m α)
  (h2 : not_parallel m n)
  (h3 : line_not_parallel_plane n β) :
  planes_perpendicular α β :=
sorry

end perpendicular_planes_l356_35699


namespace fixed_points_for_specific_values_range_of_a_for_two_fixed_points_l356_35643

/-- Definition of the function f(x) -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + (b - 1)

/-- A point x₀ is a fixed point of f if f(x₀) = x₀ -/
def is_fixed_point (a b x₀ : ℝ) : Prop := f a b x₀ = x₀

theorem fixed_points_for_specific_values :
  is_fixed_point 1 (-2) 3 ∧ is_fixed_point 1 (-2) (-1) :=
sorry

theorem range_of_a_for_two_fixed_points :
  (∀ b : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_fixed_point a b x₁ ∧ is_fixed_point a b x₂) →
  (0 < a ∧ a < 1) :=
sorry

end fixed_points_for_specific_values_range_of_a_for_two_fixed_points_l356_35643


namespace geometric_propositions_l356_35635

/-- Two lines in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne : point1 ≠ point2

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ
  nonzero : normal ≠ (0, 0, 0)

/-- Perpendicularity between lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Perpendicularity between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between planes -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Parallelism between lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallelism between planes -/
def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem geometric_propositions :
  ∃ (l1 l2 l3 : Line3D) (p1 p2 p3 : Plane3D),
    (¬∀ l1 l2 l3, perpendicular_lines l1 l3 → perpendicular_lines l2 l3 → parallel_lines l1 l2) ∧
    (∀ l1 l2 p, perpendicular_line_plane l1 p → perpendicular_line_plane l2 p → parallel_lines l1 l2) ∧
    (∀ p1 p2 l, perpendicular_line_plane l p1 → perpendicular_line_plane l p2 → parallel_planes p1 p2) ∧
    (¬∀ p1 p2 p3, perpendicular_planes p1 p3 → perpendicular_planes p2 p3 → perpendicular_planes p1 p2) :=
by sorry

end geometric_propositions_l356_35635


namespace quadratic_equations_solutions_l356_35610

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 6 ∧ x₂ = 2 ∧ x₁^2 - 8*x₁ + 12 = 0 ∧ x₂^2 - 8*x₂ + 12 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 3 ∧ y₂ = -3 ∧ (y₁ - 3)^2 = 2*y₁*(y₁ - 3) ∧ (y₂ - 3)^2 = 2*y₂*(y₂ - 3)) :=
by sorry

end quadratic_equations_solutions_l356_35610


namespace intersection_point_l356_35617

def circle_C (ρ θ : ℝ) : Prop := ρ = Real.cos θ + Real.sin θ

def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

def valid_polar_coord (ρ θ : ℝ) : Prop := ρ ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

theorem intersection_point :
  ∃ (ρ θ : ℝ), 
    circle_C ρ θ ∧ 
    line_l ρ θ ∧ 
    valid_polar_coord ρ θ ∧ 
    ρ = 1 ∧ 
    θ = Real.pi / 2 :=
sorry

end intersection_point_l356_35617


namespace cube_volume_proof_l356_35620

theorem cube_volume_proof (a b c : ℝ) 
  (h1 : a^2 + b^2 = 81)
  (h2 : a^2 + c^2 = 169)
  (h3 : c^2 + b^2 = 196) :
  a * b * c = 18 * Real.sqrt 71 := by
sorry

end cube_volume_proof_l356_35620


namespace average_of_remaining_digits_l356_35603

theorem average_of_remaining_digits 
  (total_count : Nat) 
  (subset_count : Nat) 
  (total_average : ℝ) 
  (subset_average : ℝ) 
  (h1 : total_count = 20) 
  (h2 : subset_count = 14) 
  (h3 : total_average = 500) 
  (h4 : subset_average = 390) :
  let remaining_count := total_count - subset_count
  let remaining_sum := total_count * total_average - subset_count * subset_average
  remaining_sum / remaining_count = 756.67 := by
sorry

#eval (20 * 500 - 14 * 390) / 6

end average_of_remaining_digits_l356_35603


namespace bus_trip_speed_l356_35674

/-- The average speed of a bus trip, given the conditions of the problem -/
def average_speed : ℝ → Prop :=
  fun v =>
    v > 0 ∧
    560 / v - 560 / (v + 10) = 2

theorem bus_trip_speed : ∃ v : ℝ, average_speed v ∧ v = 50 :=
  sorry

end bus_trip_speed_l356_35674


namespace simplify_expression_factorize_expression_l356_35646

-- Problem 1
theorem simplify_expression (x : ℝ) : (-2*x)^2 + 3*x*x = 7*x^2 := by
  sorry

-- Problem 2
theorem factorize_expression (m a b : ℝ) : m*a^2 - m*b^2 = m*(a - b)*(a + b) := by
  sorry

end simplify_expression_factorize_expression_l356_35646


namespace tourist_contact_probability_l356_35609

/-- The probability that two groups of tourists can contact each other -/
def contact_probability (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 40

/-- Theorem: Given two groups of tourists with 5 and 8 members respectively,
    and the probability p that a tourist from the first group has the phone number
    of a tourist from the second group, the probability that the two groups
    will be able to contact each other is 1 - (1-p)^40. -/
theorem tourist_contact_probability (p : ℝ) 
  (h1 : 0 ≤ p) (h2 : p ≤ 1) : 
  contact_probability p = 1 - (1 - p) ^ 40 := by
  sorry

end tourist_contact_probability_l356_35609


namespace max_d_value_l356_35637

def is_valid_number (d e : ℕ) : Prop :=
  d < 10 ∧ e < 10 ∧ 
  (5000000 + d * 100000 + 500000 + 2200 + 20 + e) % 44 = 0

theorem max_d_value :
  ∃ (d : ℕ), is_valid_number d 6 ∧
  ∀ (d' e : ℕ), is_valid_number d' e → d' ≤ d :=
by sorry

end max_d_value_l356_35637


namespace last_fish_is_sudak_l356_35697

-- Define the types of fish
inductive Fish : Type
| Perch : Fish
| Pike : Fish
| Sudak : Fish

-- Define the initial fish counts
def initial_perches : Nat := 6
def initial_pikes : Nat := 7
def initial_sudaks : Nat := 8

-- Define the eating rules
def can_eat (eater prey : Fish) : Prop :=
  match eater, prey with
  | Fish.Perch, Fish.Pike => True
  | Fish.Pike, Fish.Pike => True
  | Fish.Pike, Fish.Perch => True
  | _, _ => False

-- Define the restriction on eating fish that have eaten an odd number
def odd_eater_restriction (f : Fish → Nat) : Prop :=
  ∀ fish, f fish % 2 = 1 → ∀ other, ¬(can_eat other fish)

-- Define the theorem
theorem last_fish_is_sudak :
  ∃ (final_state : Fish → Nat),
    (final_state Fish.Perch + final_state Fish.Pike + final_state Fish.Sudak = 1) ∧
    (final_state Fish.Sudak = 1) ∧
    (∃ (intermediate_state : Fish → Nat),
      odd_eater_restriction intermediate_state ∧
      (∀ fish, final_state fish ≤ intermediate_state fish) ∧
      (intermediate_state Fish.Perch ≤ initial_perches) ∧
      (intermediate_state Fish.Pike ≤ initial_pikes) ∧
      (intermediate_state Fish.Sudak ≤ initial_sudaks)) :=
sorry

end last_fish_is_sudak_l356_35697


namespace valid_hexagonal_star_exists_l356_35613

/-- A configuration of numbers in the hexagonal star format -/
structure HexagonalStar where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  g : Nat

/-- Check if a number is 2, 3, or 5 -/
def isValidNumber (n : Nat) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5

/-- Check if all numbers in the configuration are valid -/
def allValidNumbers (hs : HexagonalStar) : Prop :=
  isValidNumber hs.a ∧ isValidNumber hs.b ∧ isValidNumber hs.c ∧
  isValidNumber hs.d ∧ isValidNumber hs.e ∧ isValidNumber hs.f ∧
  isValidNumber hs.g

/-- Check if all triangles have the same sum -/
def allTrianglesSameSum (hs : HexagonalStar) : Prop :=
  let sum1 := hs.a + hs.b + hs.g
  let sum2 := hs.b + hs.c + hs.g
  let sum3 := hs.c + hs.d + hs.g
  let sum4 := hs.d + hs.e + hs.g
  let sum5 := hs.e + hs.f + hs.g
  let sum6 := hs.f + hs.a + hs.g
  sum1 = sum2 ∧ sum2 = sum3 ∧ sum3 = sum4 ∧ sum4 = sum5 ∧ sum5 = sum6

/-- Theorem: There exists a valid hexagonal star configuration -/
theorem valid_hexagonal_star_exists : ∃ (hs : HexagonalStar), 
  allValidNumbers hs ∧ allTrianglesSameSum hs := by
  sorry

end valid_hexagonal_star_exists_l356_35613


namespace unique_solution_system_l356_35679

theorem unique_solution_system :
  ∃! (x y z : ℝ),
    2 * x - y + z = 14 ∧
    y = 2 ∧
    x + z = 3 * y + 5 :=
by
  -- The proof would go here
  sorry

end unique_solution_system_l356_35679


namespace number_thought_of_l356_35662

theorem number_thought_of (x : ℝ) : (x / 5 + 6 = 65) → x = 295 := by
  sorry

end number_thought_of_l356_35662
