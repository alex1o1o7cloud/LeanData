import Mathlib

namespace regression_analysis_properties_l2918_291834

-- Define the basic concepts
def FunctionRelationship : Type := Unit
def CorrelationRelationship : Type := Unit
def RegressionAnalysis : Type := Unit

-- Define properties
def isDeterministic (r : Type) : Prop := sorry
def isNonDeterministic (r : Type) : Prop := sorry
def usedFor (a : Type) (r : Type) : Prop := sorry

-- Theorem statement
theorem regression_analysis_properties :
  isDeterministic FunctionRelationship ∧
  isNonDeterministic CorrelationRelationship ∧
  usedFor RegressionAnalysis CorrelationRelationship :=
by sorry

end regression_analysis_properties_l2918_291834


namespace delta_calculation_l2918_291820

-- Define the operation Δ
def delta (a b : ℝ) : ℝ := a^3 - b^2

-- State the theorem
theorem delta_calculation :
  delta (3^(delta 5 14)) (4^(delta 4 6)) = -4^56 := by
  sorry

end delta_calculation_l2918_291820


namespace cos_2alpha_value_l2918_291836

theorem cos_2alpha_value (α : Real) (h : Real.tan (π/4 - α) = -1/3) : 
  Real.cos (2*α) = -3/5 := by
  sorry

end cos_2alpha_value_l2918_291836


namespace necessary_but_not_sufficient_l2918_291809

theorem necessary_but_not_sufficient :
  (∃ a : ℝ, (a < 1 → a ≤ 1) ∧ ¬(a ≤ 1 → a < 1)) ∧
  (∃ x y : ℝ, (x = 1 ∧ y = 0 → x^2 + y^2 = 1) ∧ ¬(x^2 + y^2 = 1 → x = 1 ∧ y = 0)) :=
by sorry

end necessary_but_not_sufficient_l2918_291809


namespace point_five_units_from_origin_l2918_291849

theorem point_five_units_from_origin (a : ℝ) : |a| = 5 ↔ a = 5 ∨ a = -5 := by
  sorry

end point_five_units_from_origin_l2918_291849


namespace at_least_one_hit_l2918_291833

theorem at_least_one_hit (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) 
  (h_B : p_B = 1/3) 
  (h_C : p_C = 1/4) : 
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end at_least_one_hit_l2918_291833


namespace sharas_age_l2918_291894

theorem sharas_age (jaymee_age shara_age : ℕ) : 
  jaymee_age = 22 →
  jaymee_age = 2 * shara_age + 2 →
  shara_age = 10 := by
  sorry

end sharas_age_l2918_291894


namespace inequality_addition_l2918_291811

theorem inequality_addition {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end inequality_addition_l2918_291811


namespace problem_solution_l2918_291837

def U : Set ℕ := {1, 2, 3, 4, 5}

def A (q : ℤ) : Set ℕ := {x ∈ U | x^2 - 5*x + q = 0}

def B (p : ℤ) : Set ℕ := {x ∈ U | x^2 + p*x + 12 = 0}

theorem problem_solution :
  ∃ (p q : ℤ),
    (U \ A q) ∪ B p = {1, 3, 4, 5} ∧
    p = -7 ∧
    q = 6 ∧
    A q = {2, 3} ∧
    B p = {3, 4} := by
  sorry

end problem_solution_l2918_291837


namespace sine_shift_left_l2918_291869

/-- Shifting a sine function to the left --/
theorem sine_shift_left (x : ℝ) :
  let f (t : ℝ) := Real.sin t
  let g (t : ℝ) := Real.sin (t + π / 6)
  ∀ y : ℝ, f (x + π / 6) = g x :=
by sorry

end sine_shift_left_l2918_291869


namespace gift_wrap_sales_l2918_291860

/-- Proves that the total number of gift wrap rolls sold is 480 given the specified conditions -/
theorem gift_wrap_sales (solid_price print_price total_amount print_rolls : ℚ)
  (h1 : solid_price = 4)
  (h2 : print_price = 6)
  (h3 : total_amount = 2340)
  (h4 : print_rolls = 210)
  (h5 : ∃ solid_rolls : ℚ, solid_price * solid_rolls + print_price * print_rolls = total_amount) :
  ∃ total_rolls : ℚ, total_rolls = 480 ∧ 
    ∃ solid_rolls : ℚ, total_rolls = solid_rolls + print_rolls ∧
    solid_price * solid_rolls + print_price * print_rolls = total_amount := by
  sorry


end gift_wrap_sales_l2918_291860


namespace gavins_dreams_l2918_291828

/-- The number of dreams Gavin has every day this year -/
def dreams_per_day : ℕ := sorry

/-- The number of days in a year -/
def days_in_year : ℕ := 365

/-- The total number of dreams in two years -/
def total_dreams : ℕ := 4380

theorem gavins_dreams : 
  dreams_per_day * days_in_year + 2 * (dreams_per_day * days_in_year) = total_dreams ∧ 
  dreams_per_day = 4 := by sorry

end gavins_dreams_l2918_291828


namespace ellipse_equation_l2918_291808

theorem ellipse_equation (A B : ℝ × ℝ) (h1 : A = (0, 5/3)) (h2 : B = (1, 1)) :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧
  (∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ 16 * x^2 + 9 * y^2 = 225) :=
sorry


end ellipse_equation_l2918_291808


namespace intersection_points_on_parabola_l2918_291805

theorem intersection_points_on_parabola :
  ∀ (x y : ℝ),
  (x = y^2 ∧ (x - 11)^2 + (y - 1)^2 = 25) →
  y = (1/2) * x^2 - (21/2) * x + 97/2 :=
by sorry

end intersection_points_on_parabola_l2918_291805


namespace radical_product_equals_27_l2918_291810

theorem radical_product_equals_27 :
  let a := 81
  let b := 27
  let c := 9
  (a = 3^4) → (b = 3^3) → (c = 3^2) →
  (a^(1/4) * b^(1/3) * c^(1/2) : ℝ) = 27 := by
  sorry

end radical_product_equals_27_l2918_291810


namespace simplify_product_of_square_roots_l2918_291839

theorem simplify_product_of_square_roots (x : ℝ) :
  Real.sqrt (x^2 - 4*x + 4) * Real.sqrt (x^2 + 4*x + 4) = |x - 2| * |x + 2| := by
  sorry

end simplify_product_of_square_roots_l2918_291839


namespace square_area_from_diagonal_l2918_291848

/-- The area of a square with a diagonal of 28 meters is 392 square meters. -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 28) : 
  (d ^ 2 / 2) = 392 := by
  sorry

end square_area_from_diagonal_l2918_291848


namespace distance_after_three_minutes_l2918_291874

/-- The distance between two vehicles after a given time, given their speeds and initial positions. -/
def distanceBetweenVehicles (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed2 - speed1) * time

theorem distance_after_three_minutes :
  let truckSpeed : ℝ := 65
  let carSpeed : ℝ := 85
  let time : ℝ := 3 / 60
  distanceBetweenVehicles truckSpeed carSpeed time = 1 := by
  sorry

#check distance_after_three_minutes

end distance_after_three_minutes_l2918_291874


namespace unknown_cube_edge_length_l2918_291802

theorem unknown_cube_edge_length 
  (edge1 : ℝ) (edge2 : ℝ) (edge_unknown : ℝ) (edge_new : ℝ)
  (h1 : edge1 = 6)
  (h2 : edge2 = 8)
  (h3 : edge_new = 12)
  (h4 : edge1^3 + edge2^3 + edge_unknown^3 = edge_new^3) :
  edge_unknown = 10 := by sorry

end unknown_cube_edge_length_l2918_291802


namespace ten_mile_taxi_cost_l2918_291870

/-- Calculates the cost of a taxi ride given the fixed cost, per-mile cost, and distance traveled. -/
def taxi_cost (fixed_cost : ℝ) (per_mile_cost : ℝ) (distance : ℝ) : ℝ :=
  fixed_cost + per_mile_cost * distance

/-- Theorem: The cost of a 10-mile taxi ride with a $2.00 fixed cost and $0.30 per-mile cost is $5.00. -/
theorem ten_mile_taxi_cost : 
  taxi_cost 2 0.3 10 = 5 := by
  sorry

end ten_mile_taxi_cost_l2918_291870


namespace blending_markers_count_l2918_291889

/-- Proof that the number of drawings made with blending markers is 7 -/
theorem blending_markers_count (total : ℕ) (colored_pencils : ℕ) (charcoal : ℕ) 
  (h1 : total = 25)
  (h2 : colored_pencils = 14)
  (h3 : charcoal = 4) :
  total - (colored_pencils + charcoal) = 7 := by
  sorry

end blending_markers_count_l2918_291889


namespace soy_sauce_bottles_l2918_291886

/-- Represents the amount of soy sauce in ounces -/
def OuncesPerBottle : ℕ := 16

/-- Represents the number of ounces in a cup -/
def OuncesPerCup : ℕ := 8

/-- Represents the amount of soy sauce needed for each recipe in cups -/
def RecipeCups : List ℕ := [2, 1, 3]

/-- Calculates the total number of cups needed for all recipes -/
def TotalCups : ℕ := RecipeCups.sum

/-- Calculates the total number of ounces needed for all recipes -/
def TotalOunces : ℕ := TotalCups * OuncesPerCup

/-- Calculates the number of bottles needed, rounding up to the nearest whole number -/
def BottlesNeeded : ℕ := (TotalOunces + OuncesPerBottle - 1) / OuncesPerBottle

theorem soy_sauce_bottles : BottlesNeeded = 3 := by sorry

end soy_sauce_bottles_l2918_291886


namespace three_people_eight_seats_l2918_291882

/-- The number of ways 3 people can sit in 8 seats with empty seats between them -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  let available_positions := total_seats - 2 * people + 1
  (available_positions.choose people) * (Nat.factorial people)

/-- Theorem stating that there are 24 ways for 3 people to sit in 8 seats with empty seats between them -/
theorem three_people_eight_seats : seating_arrangements 8 3 = 24 := by
  sorry

end three_people_eight_seats_l2918_291882


namespace time_per_trip_is_three_l2918_291844

/-- Represents the number of trips Melissa makes to town in a year -/
def trips_per_year : ℕ := 24

/-- Represents the total hours Melissa spends driving in a year -/
def total_driving_hours : ℕ := 72

/-- Calculates the time for one round trip to town and back -/
def time_per_trip : ℚ := total_driving_hours / trips_per_year

theorem time_per_trip_is_three : time_per_trip = 3 := by
  sorry

end time_per_trip_is_three_l2918_291844


namespace combination_permutation_equation_l2918_291812

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Falling factorial -/
def fallingFactorial (n k : ℕ) : ℕ := (n - k + 1).factorial / (n - k).factorial

theorem combination_permutation_equation : 
  ∃ x : ℕ, binomial (x + 5) x = binomial (x + 3) (x - 1) + binomial (x + 3) (x - 2) + 
    (3 * fallingFactorial (x + 3) 3) / 4 ∧ x = 14 := by
  sorry

end combination_permutation_equation_l2918_291812


namespace parabola_distance_theorem_l2918_291826

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -2

-- Define the condition for Q being on the parabola
def Q_on_parabola (Q : ℝ × ℝ) : Prop := parabola Q.1 Q.2

-- Define the vector relation
def vector_relation (P Q : ℝ × ℝ) : Prop :=
  (Q.1 - focus.1, Q.2 - focus.2) = (-4 * (focus.1 - P.1), -4 * (focus.2 - P.2))

-- The theorem to prove
theorem parabola_distance_theorem (P Q : ℝ × ℝ) :
  directrix P.1 →
  Q_on_parabola Q →
  vector_relation P Q →
  Real.sqrt ((Q.1 - focus.1)^2 + (Q.2 - focus.2)^2) = 20 :=
by sorry

end parabola_distance_theorem_l2918_291826


namespace quadratic_solution_l2918_291851

theorem quadratic_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : d^2 + c*d + d = 0) : 
  c = 1 ∧ d = -2 := by
sorry

end quadratic_solution_l2918_291851


namespace fault_line_movement_l2918_291815

/-- The movement of a fault line over two years -/
theorem fault_line_movement 
  (movement_past_year : ℝ) 
  (movement_year_before : ℝ) 
  (h1 : movement_past_year = 1.25)
  (h2 : movement_year_before = 5.25) : 
  movement_past_year + movement_year_before = 6.50 := by
sorry

end fault_line_movement_l2918_291815


namespace complement_A_intersect_B_l2918_291804

-- Define the universal set U
def U : Set ℤ := {x : ℤ | x^2 - 5*x - 6 < 0}

-- Define set A
def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 2}

-- Define set B
def B : Set ℤ := {2, 3, 5}

-- Theorem statement
theorem complement_A_intersect_B : (U \ A) ∩ B = {3, 5} := by sorry

end complement_A_intersect_B_l2918_291804


namespace parallel_equal_sides_is_parallelogram_l2918_291824

/-- A quadrilateral in 2D space --/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Definition of parallel sides in a quadrilateral --/
def has_parallel_sides (q : Quadrilateral) : Prop :=
  (q.A.1 - q.B.1) / (q.A.2 - q.B.2) = (q.D.1 - q.C.1) / (q.D.2 - q.C.2) ∧
  (q.A.1 - q.D.1) / (q.A.2 - q.D.2) = (q.B.1 - q.C.1) / (q.B.2 - q.C.2)

/-- Definition of equal sides in a quadrilateral --/
def has_equal_sides (q : Quadrilateral) : Prop :=
  (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2 = (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2 ∧
  (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2 = (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 ∧
  (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 = (q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2

/-- Definition of a parallelogram --/
def is_parallelogram (q : Quadrilateral) : Prop :=
  has_parallel_sides q

/-- Theorem: A quadrilateral with parallel and equal sides is a parallelogram --/
theorem parallel_equal_sides_is_parallelogram (q : Quadrilateral) :
  has_parallel_sides q → has_equal_sides q → is_parallelogram q :=
by
  sorry

end parallel_equal_sides_is_parallelogram_l2918_291824


namespace simplify_expression_l2918_291821

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^8 = 0 := by
  sorry

end simplify_expression_l2918_291821


namespace smallest_x_for_cube_l2918_291858

theorem smallest_x_for_cube (x M : ℕ+) : 
  (∀ y : ℕ+, y < x → ¬∃ N : ℕ+, 720 * y = N^3) → 
  (∃ N : ℕ+, 720 * x = N^3) → 
  x = 300 := by
sorry

end smallest_x_for_cube_l2918_291858


namespace circle_tangent_to_line_and_center_at_parabola_focus_l2918_291892

theorem circle_tangent_to_line_and_center_at_parabola_focus :
  ∀ (x y : ℝ),
  (∃ (h : ℝ), y^2 = 8*x → (2, 0) = (h, 0)) →
  (∃ (r : ℝ), r = Real.sqrt 2) →
  (x - 2)^2 + y^2 = 2 →
  (∃ (t : ℝ), t = x ∧ t = y) →
  (∃ (d : ℝ), d = |x - y| / Real.sqrt 2 ∧ d = Real.sqrt 2) :=
by sorry

end circle_tangent_to_line_and_center_at_parabola_focus_l2918_291892


namespace cookies_milk_ratio_l2918_291888

/-- Proof that 5 cookies require 20/3 pints of milk given the established ratio and conversion rates -/
theorem cookies_milk_ratio (cookies_base : ℕ) (milk_base : ℕ) (cookies_target : ℕ) :
  cookies_base = 18 →
  milk_base = 3 →
  cookies_target = 5 →
  (milk_base * 4 * 2 : ℚ) / cookies_base * cookies_target = 20 / 3 := by
  sorry

end cookies_milk_ratio_l2918_291888


namespace quadratic_roots_l2918_291856

theorem quadratic_roots (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end quadratic_roots_l2918_291856


namespace factor_z4_minus_81_l2918_291867

theorem factor_z4_minus_81 (z : ℂ) : 
  z^4 - 81 = (z - 3) * (z + 3) * (z^2 + 9) := by sorry

end factor_z4_minus_81_l2918_291867


namespace employee_count_l2918_291865

theorem employee_count (total_profit : ℝ) (owner_percentage : ℝ) (employee_share : ℝ) : 
  total_profit = 50 →
  owner_percentage = 0.1 →
  employee_share = 5 →
  (1 - owner_percentage) * total_profit / employee_share = 9 := by
  sorry

end employee_count_l2918_291865


namespace zero_score_probability_l2918_291829

def num_balls : ℕ := 6
def num_red : ℕ := 1
def num_yellow : ℕ := 2
def num_blue : ℕ := 3
def num_draws : ℕ := 3

def score_red : ℤ := 1
def score_yellow : ℤ := 0
def score_blue : ℤ := -1

def prob_zero_score : ℚ := 11 / 54

theorem zero_score_probability :
  (num_balls = num_red + num_yellow + num_blue) →
  (prob_zero_score = (num_yellow^num_draws + num_red * num_yellow * num_blue * 6) / num_balls^num_draws) :=
by sorry

end zero_score_probability_l2918_291829


namespace percentage_of_female_guests_l2918_291871

theorem percentage_of_female_guests 
  (total_guests : ℕ) 
  (jays_family_females : ℕ) 
  (h1 : total_guests = 240)
  (h2 : jays_family_females = 72)
  (h3 : jays_family_females * 2 = total_guests * (percentage_female_guests / 100)) :
  percentage_female_guests = 60 := by
  sorry

end percentage_of_female_guests_l2918_291871


namespace second_next_perfect_square_l2918_291895

theorem second_next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k ^ 2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, n = m ^ 2) ∧
  (∀ y : ℕ, y > x ∧ (∃ l : ℕ, y = l ^ 2) → y ≥ n) ∧
  n = x + 4 * Int.sqrt x + 4 :=
sorry

end second_next_perfect_square_l2918_291895


namespace triangle_angle_measure_l2918_291861

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * c = b^2 - a^2 →
  A = π / 6 →
  B = π / 3 := by
sorry

end triangle_angle_measure_l2918_291861


namespace seating_theorem_l2918_291843

/-- The number of ways to arrange 5 boys and 4 girls in a row of 9 chairs such that at least 2 boys are next to each other -/
def seating_arrangements (num_boys num_girls : ℕ) : ℕ :=
  Nat.factorial (num_boys + num_girls) - (Nat.factorial num_boys * Nat.factorial num_girls)

/-- Theorem stating that the number of seating arrangements for 5 boys and 4 girls with at least 2 boys next to each other is 359000 -/
theorem seating_theorem :
  seating_arrangements 5 4 = 359000 := by
  sorry

end seating_theorem_l2918_291843


namespace complex_cube_root_l2918_291841

theorem complex_cube_root (a b : ℕ+) :
  (↑a + ↑b * Complex.I) ^ 3 = 2 + 11 * Complex.I →
  ↑a + ↑b * Complex.I = 2 + Complex.I := by
  sorry

end complex_cube_root_l2918_291841


namespace probability_is_two_thirds_l2918_291817

structure Diagram where
  total_triangles : ℕ
  triangles_with_G : ℕ
  equal_probability : Bool

def probability_including_G (d : Diagram) : ℚ :=
  d.triangles_with_G / d.total_triangles

theorem probability_is_two_thirds (d : Diagram) 
  (h1 : d.total_triangles = 6)
  (h2 : d.triangles_with_G = 4)
  (h3 : d.equal_probability = true) :
  probability_including_G d = 2/3 := by
  sorry

end probability_is_two_thirds_l2918_291817


namespace phd_basics_time_l2918_291881

/-- Represents the time John spent on his PhD journey -/
structure PhDTime where
  total : ℝ
  acclimation : ℝ
  basics : ℝ
  research : ℝ
  dissertation : ℝ

/-- The conditions of John's PhD journey -/
def phd_conditions (t : PhDTime) : Prop :=
  t.total = 7 ∧
  t.acclimation = 1 ∧
  t.research = t.basics + 0.75 * t.basics ∧
  t.dissertation = 0.5 * t.acclimation ∧
  t.total = t.acclimation + t.basics + t.research + t.dissertation

/-- Theorem stating that given the PhD conditions, the time spent learning basics is 2 years -/
theorem phd_basics_time (t : PhDTime) (h : phd_conditions t) : t.basics = 2 := by
  sorry

end phd_basics_time_l2918_291881


namespace windows_already_installed_l2918_291832

/-- Proves that the number of windows already installed is 6 -/
theorem windows_already_installed
  (total_windows : ℕ)
  (install_time_per_window : ℕ)
  (time_left : ℕ)
  (h1 : total_windows = 10)
  (h2 : install_time_per_window = 5)
  (h3 : time_left = 20) :
  total_windows - (time_left / install_time_per_window) = 6 := by
  sorry

#check windows_already_installed

end windows_already_installed_l2918_291832


namespace happy_children_count_l2918_291822

theorem happy_children_count (total : ℕ) (sad : ℕ) (neither : ℕ) (boys : ℕ) (girls : ℕ) 
  (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ) : ℕ :=
  by
  -- Define the given conditions
  have h1 : total = 60 := by sorry
  have h2 : sad = 10 := by sorry
  have h3 : neither = 20 := by sorry
  have h4 : boys = 16 := by sorry
  have h5 : girls = 44 := by sorry
  have h6 : happy_boys = 6 := by sorry
  have h7 : sad_girls = 4 := by sorry
  have h8 : neither_boys = 4 := by sorry

  -- Prove that the number of happy children is 30
  have happy_children : ℕ := total - (sad + neither)
  exact happy_children

end happy_children_count_l2918_291822


namespace birds_berries_consumption_l2918_291876

theorem birds_berries_consumption (num_birds : ℕ) (total_berries : ℕ) (num_days : ℕ) 
  (h1 : num_birds = 5)
  (h2 : total_berries = 140)
  (h3 : num_days = 4) :
  total_berries / num_days / num_birds = 7 := by
  sorry

end birds_berries_consumption_l2918_291876


namespace bakers_remaining_cakes_l2918_291852

/-- Calculates the number of remaining cakes for a baker --/
def remaining_cakes (initial : ℕ) (additional : ℕ) (sold : ℕ) : ℕ :=
  initial + additional - sold

/-- Theorem: The baker's remaining cakes is 67 --/
theorem bakers_remaining_cakes :
  remaining_cakes 62 149 144 = 67 := by
  sorry

end bakers_remaining_cakes_l2918_291852


namespace x_value_l2918_291846

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem x_value (x : ℕ) (h1 : x ∉ A) (h2 : x ∈ B) : x = 3 := by
  sorry

end x_value_l2918_291846


namespace tennis_tournament_l2918_291850

theorem tennis_tournament (n : ℕ) : n > 0 → (
  ∃ (women_wins men_wins : ℕ),
    women_wins + men_wins = (4 * n).choose 2 ∧
    women_wins * 11 = men_wins * 4 ∧
    ∀ m : ℕ, m > 0 ∧ m < n → ¬(
      ∃ (w_wins m_wins : ℕ),
        w_wins + m_wins = (4 * m).choose 2 ∧
        w_wins * 11 = m_wins * 4
    )
) → n = 4 := by
  sorry

end tennis_tournament_l2918_291850


namespace divisible_by_seven_last_digit_l2918_291859

theorem divisible_by_seven_last_digit :
  ∀ d : ℕ, d < 10 → ∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d :=
by sorry

end divisible_by_seven_last_digit_l2918_291859


namespace solution_range_l2918_291879

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ (x + m) / (x - 2) + (2 * m) / (2 - x) = 3) → 
  m < 6 ∧ m ≠ 2 := by
sorry

end solution_range_l2918_291879


namespace solve_equation_l2918_291875

theorem solve_equation (b : ℚ) (h : b + b/4 = 5/2) : b = 2 := by
  sorry

end solve_equation_l2918_291875


namespace a_less_equal_two_l2918_291814

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | 2 * a - x > 1}

-- State the theorem
theorem a_less_equal_two (a : ℝ) : A ∩ (Set.univ \ B a) = A → a ≤ 2 := by
  sorry

end a_less_equal_two_l2918_291814


namespace inequality_solution_l2918_291891

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, x > 0 → (Real.log x / Real.log a) + |a + (Real.log x / Real.log a)| * (Real.log a / Real.log (Real.sqrt x)) ≥ a * (Real.log a / Real.log x)) ↔
  -1/3 ≤ a ∧ a ≤ 1 :=
by sorry

end inequality_solution_l2918_291891


namespace area_of_ABHFGD_l2918_291806

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (p1 : Point) (p2 : Point) : Prop := sorry

/-- Checks if a point divides a line segment in a given ratio -/
def divideSegment (p : Point) (a : Point) (b : Point) (ratio : ℝ) : Prop := sorry

/-- Calculates the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ := sorry

theorem area_of_ABHFGD (a b c d e f g h : Point) :
  let abcd := Square.mk a b c d
  let efgd := Square.mk e f g d
  squareArea abcd = 25 ∧
  squareArea efgd = 25 ∧
  isMidpoint h e f ∧
  divideSegment h b c (1/3) →
  abs (polygonArea [a, b, h, f, g, d] - 27.09) < 0.01 := by sorry

end area_of_ABHFGD_l2918_291806


namespace least_addition_for_divisibility_problem_solution_l2918_291899

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 19 ∧ (1156 + x) % 25 = 0 ∧ ∀ (y : ℕ), y < x → (1156 + y) % 25 ≠ 0 :=
by
  sorry

end least_addition_for_divisibility_problem_solution_l2918_291899


namespace perfect_square_identification_l2918_291893

theorem perfect_square_identification :
  let a := 3^6 * 7^7 * 8^8
  let b := 3^8 * 7^6 * 8^7
  let c := 3^7 * 7^8 * 8^6
  let d := 3^7 * 7^7 * 8^8
  let e := 3^8 * 7^8 * 8^8
  ∃ n : ℕ, e = n^2 ∧ 
  (∀ m : ℕ, a ≠ m^2) ∧ 
  (∀ m : ℕ, b ≠ m^2) ∧ 
  (∀ m : ℕ, c ≠ m^2) ∧ 
  (∀ m : ℕ, d ≠ m^2) :=
by sorry

end perfect_square_identification_l2918_291893


namespace arithmetic_sequence_proof_l2918_291835

-- Define an arithmetic sequence
def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_proof :
  (arithmeticSequence 8 (-3) 20 = -49) ∧
  (arithmeticSequence (-5) (-4) 100 = -401) := by
  sorry

end arithmetic_sequence_proof_l2918_291835


namespace john_age_proof_l2918_291800

/-- John's current age -/
def john_age : ℕ := 18

/-- Proposition: John's current age satisfies the given condition -/
theorem john_age_proof : 
  (john_age - 5 : ℤ) = (john_age + 8 : ℤ) / 2 := by
  sorry

end john_age_proof_l2918_291800


namespace f_4_1981_l2918_291877

def f : ℕ → ℕ → ℕ 
  | 0, y => y + 1
  | x + 1, 0 => f x 1
  | x + 1, y + 1 => f x (f (x + 1) y)

theorem f_4_1981 : f 4 1981 = 2^(2^(2^(1981 + 1) + 1)) - 3 := by
  sorry

end f_4_1981_l2918_291877


namespace triangle_ABC_properties_l2918_291823

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- Define the altitude from C to AB
def altitude_equation (x y : ℝ) : Prop :=
  2 * x + 5 * y - 30 = 0

-- Define the midline parallel to AC
def midline_equation (x : ℝ) : Prop :=
  x = 4

-- Theorem statement
theorem triangle_ABC_properties :
  -- The altitude from C to AB satisfies the equation
  (∀ x y : ℝ, altitude_equation x y ↔ 
    (x - C.1) * (B.2 - A.2) = (y - C.2) * (B.1 - A.1)) ∧
  -- The midline parallel to AC satisfies the equation
  (∀ x : ℝ, midline_equation x ↔ 
    x = (B.1 + C.1) / 2) :=
by sorry

end triangle_ABC_properties_l2918_291823


namespace port_vessel_count_port_vessel_count_proof_l2918_291807

theorem port_vessel_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun cruise_ships cargo_ships sailboats fishing_boats =>
    cruise_ships = 4 ∧
    cargo_ships = 2 * cruise_ships ∧
    sailboats = cargo_ships + 6 ∧
    sailboats = 7 * fishing_boats →
    cruise_ships + cargo_ships + sailboats + fishing_boats = 28

/-- Proof of the theorem -/
theorem port_vessel_count_proof : port_vessel_count 4 8 14 2 := by
  sorry

end port_vessel_count_port_vessel_count_proof_l2918_291807


namespace abs_neg_five_eq_five_l2918_291816

theorem abs_neg_five_eq_five : |(-5 : ℤ)| = 5 := by
  sorry

end abs_neg_five_eq_five_l2918_291816


namespace litter_patrol_problem_l2918_291825

/-- The Litter Patrol Problem -/
theorem litter_patrol_problem (total_litter : ℕ) (aluminum_cans : ℕ) (glass_bottles : ℕ) :
  total_litter = 18 →
  aluminum_cans = 8 →
  total_litter = aluminum_cans + glass_bottles →
  glass_bottles = 10 := by
sorry

end litter_patrol_problem_l2918_291825


namespace lcm_of_48_and_14_l2918_291868

theorem lcm_of_48_and_14 (n : ℕ) (h1 : n = 48) (h2 : Nat.gcd n 14 = 12) :
  Nat.lcm n 14 = 56 := by
  sorry

end lcm_of_48_and_14_l2918_291868


namespace angle_between_lines_l2918_291896

def line1 (x : ℝ) : ℝ := -2 * x
def line2 (x : ℝ) : ℝ := 3 * x + 5

theorem angle_between_lines :
  let k1 := -2
  let k2 := 3
  let tan_phi := abs ((k2 - k1) / (1 + k1 * k2))
  Real.arctan tan_phi * (180 / Real.pi) = 45 :=
sorry

end angle_between_lines_l2918_291896


namespace percent_commutation_l2918_291864

theorem percent_commutation (x : ℝ) (h : (25 / 100) * ((10 / 100) * x) = 15) :
  (10 / 100) * ((25 / 100) * x) = 15 := by
sorry

end percent_commutation_l2918_291864


namespace binary_sum_theorem_l2918_291887

def binary_to_nat : List Bool → Nat
  | [] => 0
  | b::bs => (if b then 1 else 0) + 2 * binary_to_nat bs

def num1 : List Bool := [true, false, true, true]  -- 1101₂
def num2 : List Bool := [true, false, true]        -- 101₂
def num3 : List Bool := [false, true, true, true]  -- 1110₂
def num4 : List Bool := [true, true, true]         -- 111₂
def num5 : List Bool := [false, true, false, true] -- 1010₂
def result : List Bool := [true, false, true, false, true] -- 10101₂

theorem binary_sum_theorem :
  binary_to_nat num1 + binary_to_nat num2 + binary_to_nat num3 +
  binary_to_nat num4 + binary_to_nat num5 = binary_to_nat result := by
  sorry

end binary_sum_theorem_l2918_291887


namespace driver_net_pay_rate_driver_net_pay_is_24_l2918_291801

/-- Calculates the net rate of pay per hour for a driver given specific conditions --/
theorem driver_net_pay_rate (travel_time : ℝ) (travel_speed : ℝ) (car_efficiency : ℝ) 
  (pay_rate : ℝ) (gasoline_cost : ℝ) : ℝ :=
  let total_distance := travel_time * travel_speed
  let gasoline_used := total_distance / car_efficiency
  let earnings := pay_rate * total_distance
  let gasoline_expense := gasoline_cost * gasoline_used
  let net_earnings := earnings - gasoline_expense
  let net_pay_rate := net_earnings / travel_time
  net_pay_rate

/-- Proves that the driver's net rate of pay is $24 per hour under given conditions --/
theorem driver_net_pay_is_24 :
  driver_net_pay_rate 3 50 25 0.60 3.00 = 24 := by
  sorry

end driver_net_pay_rate_driver_net_pay_is_24_l2918_291801


namespace solve_inequality_find_range_of_a_l2918_291884

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem solve_inequality (x : ℝ) :
  f x (-1) ≥ 3 ↔ x ≤ -3/2 ∨ x ≥ 3/2 :=
sorry

-- Part 2
theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 2) ↔ a ≤ -1 ∨ a ≥ 3 :=
sorry

end solve_inequality_find_range_of_a_l2918_291884


namespace parabola_points_range_l2918_291883

-- Define the parabola
def parabola (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + b

-- Define the theorem
theorem parabola_points_range (a b y₁ y₂ n : ℝ) :
  a > 0 →
  y₁ < y₂ →
  parabola a b (2 * n + 3) = y₁ →
  parabola a b (n - 1) = y₂ →
  (2 * n + 3 - 1) * (n - 1 - 1) < 0 →  -- Opposite sides of axis of symmetry
  -1 < n ∧ n < 0 := by
sorry

end parabola_points_range_l2918_291883


namespace arithmetic_proof_l2918_291878

theorem arithmetic_proof : 4 * (8 - 3) - 2 * 6 = 8 := by
  sorry

end arithmetic_proof_l2918_291878


namespace solve_system_l2918_291818

theorem solve_system (x y : ℚ) : 
  (1 / 3 - 1 / 4 = 1 / x) → (x + y = 10) → (x = 12 ∧ y = -2) := by
  sorry

end solve_system_l2918_291818


namespace berry_fraction_proof_l2918_291857

theorem berry_fraction_proof (steve skylar stacy : ℕ) : 
  skylar = 20 →
  stacy = 32 →
  stacy = 3 * steve + 2 →
  steve * 2 = skylar :=
by
  sorry

end berry_fraction_proof_l2918_291857


namespace range_of_a_l2918_291830

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 5}

def B (a : ℝ) : Set ℝ := {x : ℝ | (x - a + 1) * (x - a - 1) ≤ 0}

def p (x : ℝ) : Prop := x ∈ A

def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

theorem range_of_a : 
  {a : ℝ | (∀ x, q a x → p x) ∧ (∃ x, p x ∧ ¬q a x)} = {a : ℝ | 2 ≤ a ∧ a ≤ 4} := by
  sorry

end range_of_a_l2918_291830


namespace equivalent_inequalities_l2918_291847

theorem equivalent_inequalities :
  ∀ x : ℝ, (0 < x ∧ x < 1) ↔ ((1 / x > 1) ∧ (Real.log x < 0)) :=
sorry

end equivalent_inequalities_l2918_291847


namespace rectangular_field_width_l2918_291845

theorem rectangular_field_width (width length perimeter : ℝ) : 
  length = (7/5) * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 384 →
  width = 80 := by
sorry

end rectangular_field_width_l2918_291845


namespace equation_transformation_l2918_291840

theorem equation_transformation (x : ℝ) (y : ℝ) (h : y = (x^2 + 2) / (x + 1)) :
  ((x^2 + 2) / (x + 1) + (5*x + 5) / (x^2 + 2) = 6) ↔ (y^2 - 6*y + 5 = 0) :=
sorry

end equation_transformation_l2918_291840


namespace orange_apple_difference_l2918_291803

def apples : ℕ := 14
def dozen : ℕ := 12
def oranges : ℕ := 2 * dozen

theorem orange_apple_difference : oranges - apples = 10 := by
  sorry

end orange_apple_difference_l2918_291803


namespace square_difference_equality_l2918_291842

theorem square_difference_equality : 1012^2 - 1008^2 - 1006^2 + 1002^2 = 48 := by
  sorry

end square_difference_equality_l2918_291842


namespace cost_per_square_foot_calculation_l2918_291872

/-- Calculates the cost per square foot of a rented house. -/
theorem cost_per_square_foot_calculation 
  (master_bedroom_bath_area : ℝ)
  (guest_bedroom_area : ℝ)
  (num_guest_bedrooms : ℕ)
  (kitchen_bath_living_area : ℝ)
  (monthly_rent : ℝ)
  (h1 : master_bedroom_bath_area = 500)
  (h2 : guest_bedroom_area = 200)
  (h3 : num_guest_bedrooms = 2)
  (h4 : kitchen_bath_living_area = 600)
  (h5 : monthly_rent = 3000) :
  monthly_rent / (master_bedroom_bath_area + num_guest_bedrooms * guest_bedroom_area + kitchen_bath_living_area) = 2 :=
by
  sorry

end cost_per_square_foot_calculation_l2918_291872


namespace office_age_problem_l2918_291885

theorem office_age_problem (total_persons : ℕ) (avg_age_all : ℝ) 
  (group1_size : ℕ) (group2_size : ℕ) (avg_age_group2 : ℝ) 
  (age_person15 : ℕ) :
  total_persons = 18 →
  avg_age_all = 15 →
  group1_size = 5 →
  group2_size = 9 →
  avg_age_group2 = 16 →
  age_person15 = 56 →
  (total_persons * avg_age_all - group2_size * avg_age_group2 - age_person15) / group1_size = 14 := by
sorry

end office_age_problem_l2918_291885


namespace sin_cos_product_trig_expression_value_l2918_291898

-- Part I
theorem sin_cos_product (α : ℝ) 
  (h : (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7) :
  Real.sin α * Real.cos α = 3 / 10 := by
  sorry

-- Part II
theorem trig_expression_value :
  (Real.sqrt (1 - 2 * Real.sin (10 * π / 180) * Real.cos (10 * π / 180))) / 
  (Real.cos (10 * π / 180) - Real.sqrt (1 - Real.cos (170 * π / 180)^2)) = 1 := by
  sorry

end sin_cos_product_trig_expression_value_l2918_291898


namespace james_tin_collection_l2918_291880

/-- The number of tins James collected on the first day -/
def first_day_tins : ℕ := sorry

/-- The total number of tins James collected in a week -/
def total_tins : ℕ := 500

/-- The number of tins James collected on each of the last four days -/
def last_four_days_tins : ℕ := 50

theorem james_tin_collection :
  first_day_tins = 50 ∧
  first_day_tins +
  (3 * first_day_tins) +
  (3 * first_day_tins - 50) +
  (4 * last_four_days_tins) = total_tins :=
sorry

end james_tin_collection_l2918_291880


namespace systematic_sampling_interval_example_l2918_291854

/-- The interval between segments in systematic sampling --/
def systematic_sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: Given a population of 2000 and a sample size of 40, 
    the interval between segments in systematic sampling is 50 --/
theorem systematic_sampling_interval_example :
  systematic_sampling_interval 2000 40 = 50 := by
  sorry

end systematic_sampling_interval_example_l2918_291854


namespace angle4_value_l2918_291827

-- Define the angles
variable (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)

-- Define the given conditions
axiom sum_angles_1_2 : angle1 + angle2 = 180
axiom equal_angles_3_4 : angle3 = angle4
axiom new_angle1 : angle1 = 85
axiom new_angle5 : angle5 = 45
axiom triangle_sum : angle1 + angle5 + angle6 = 180

-- Define the theorem to prove
theorem angle4_value : angle4 = 22.5 := by
  sorry

end angle4_value_l2918_291827


namespace point_in_fourth_quadrant_l2918_291838

theorem point_in_fourth_quadrant (x : ℝ) : 
  let P : ℝ × ℝ := (x^2 + 1, -2)
  P.1 > 0 ∧ P.2 < 0 :=
by sorry

end point_in_fourth_quadrant_l2918_291838


namespace pie_eating_contest_l2918_291863

theorem pie_eating_contest (a b c : ℚ) 
  (ha : a = 4/5) (hb : b = 5/6) (hc : c = 3/4) : 
  (max a (max b c) - min a (min b c) : ℚ) = 1/12 := by
  sorry

end pie_eating_contest_l2918_291863


namespace final_price_percentage_l2918_291862

-- Define the discounts and tax rate
def discount1 : ℝ := 0.5
def discount2 : ℝ := 0.1
def discount3 : ℝ := 0.2
def taxRate : ℝ := 0.08

-- Define the function to calculate the final price
def finalPrice (originalPrice : ℝ) : ℝ :=
  let price1 := originalPrice * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  let price3 := price2 * (1 - discount3)
  price3 * (1 + taxRate)

-- Theorem statement
theorem final_price_percentage (originalPrice : ℝ) (originalPrice_pos : originalPrice > 0) :
  finalPrice originalPrice / originalPrice = 0.3888 := by
  sorry

end final_price_percentage_l2918_291862


namespace equation_solutions_l2918_291873

theorem equation_solutions :
  (∀ x, x * (x - 6) = 2 * (x - 8) ↔ x = 4) ∧
  (∀ x, (2 * x - 1)^2 + 3 * (2 * x - 1) + 2 = 0 ↔ x = 0 ∨ x = -1/2) := by
  sorry

end equation_solutions_l2918_291873


namespace watermelon_juice_percentage_l2918_291813

def total_volume : ℝ := 120
def orange_juice_percentage : ℝ := 15
def grape_juice_volume : ℝ := 30

theorem watermelon_juice_percentage :
  let orange_juice_volume := total_volume * (orange_juice_percentage / 100)
  let watermelon_juice_volume := total_volume - orange_juice_volume - grape_juice_volume
  (watermelon_juice_volume / total_volume) * 100 = 60 := by
sorry

end watermelon_juice_percentage_l2918_291813


namespace businessmen_beverage_theorem_l2918_291819

theorem businessmen_beverage_theorem (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ) 
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 13)
  (h4 : both = 7) :
  total - (coffee + tea - both) = 9 := by
  sorry

end businessmen_beverage_theorem_l2918_291819


namespace age_difference_l2918_291866

theorem age_difference (D M : ℕ) : 
  (M = 11 * D) →
  (M + 13 = 2 * (D + 13)) →
  (M - D = 40) := by
sorry

end age_difference_l2918_291866


namespace equation_solution_l2918_291831

theorem equation_solution : ∃! x : ℝ, (3 / (x - 2) = 6 / (x - 3)) ∧ x = 1 := by
  sorry

end equation_solution_l2918_291831


namespace chocolate_packaging_cost_l2918_291855

theorem chocolate_packaging_cost 
  (num_bars : ℕ) 
  (cost_per_bar : ℚ) 
  (total_selling_price : ℚ) 
  (total_profit : ℚ) 
  (h1 : num_bars = 5)
  (h2 : cost_per_bar = 5)
  (h3 : total_selling_price = 90)
  (h4 : total_profit = 55) :
  (total_selling_price - total_profit - (↑num_bars * cost_per_bar)) / ↑num_bars = 2 :=
by sorry

end chocolate_packaging_cost_l2918_291855


namespace banana_theorem_l2918_291897

def banana_problem (initial_bananas final_bananas : ℕ) : Prop :=
  final_bananas - initial_bananas = 7

theorem banana_theorem : banana_problem 2 9 := by
  sorry

end banana_theorem_l2918_291897


namespace max_acute_angles_2000_sided_polygon_l2918_291890

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : ℕ
  convex : Bool
  sides_eq : sides = n

/-- The maximum number of acute angles in a convex polygon -/
def max_acute_angles (p : ConvexPolygon n) : ℕ :=
  sorry

/-- Theorem: The maximum number of acute angles in a convex 2000-sided polygon is 3 -/
theorem max_acute_angles_2000_sided_polygon :
  ∀ (p : ConvexPolygon 2000), max_acute_angles p = 3 :=
by sorry

end max_acute_angles_2000_sided_polygon_l2918_291890


namespace inequality_theorem_l2918_291853

theorem inequality_theorem (a b : ℝ) (h : a < b) : -a - 1 > -b - 1 := by
  sorry

end inequality_theorem_l2918_291853
