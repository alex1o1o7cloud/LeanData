import Mathlib

namespace NUMINAMATH_CALUDE_video_views_proof_l935_93595

/-- Calculates the total views of a video given initial views and subsequent increases -/
def total_views (initial_views : ℕ) (increase_factor : ℕ) (additional_views : ℕ) : ℕ :=
  initial_views + increase_factor * initial_views + additional_views

/-- Theorem stating that given the specific conditions, the total views equal 94000 -/
theorem video_views_proof :
  let initial_views : ℕ := 4000
  let increase_factor : ℕ := 10
  let additional_views : ℕ := 50000
  total_views initial_views increase_factor additional_views = 94000 := by
  sorry

#eval total_views 4000 10 50000

end NUMINAMATH_CALUDE_video_views_proof_l935_93595


namespace NUMINAMATH_CALUDE_x_value_when_y_is_3_l935_93517

/-- The inverse square relationship between x and y -/
def inverse_square_relation (x y : ℝ) (k : ℝ) : Prop :=
  x = k / (y ^ 2)

/-- Theorem: Given the inverse square relationship between x and y,
    and the condition that x ≈ 0.1111111111111111 when y = 9,
    prove that x = 1 when y = 3 -/
theorem x_value_when_y_is_3
  (h1 : ∃ k, ∀ x y, inverse_square_relation x y k)
  (h2 : ∃ x, inverse_square_relation x 9 (9 * 0.1111111111111111)) :
  ∃ x, inverse_square_relation x 3 1 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_3_l935_93517


namespace NUMINAMATH_CALUDE_bike_fundraising_days_l935_93536

/-- The number of days required to raise money for a bike by selling bracelets -/
def days_to_raise_money (bike_cost : ℕ) (bracelet_price : ℕ) (bracelets_per_day : ℕ) : ℕ :=
  bike_cost / (bracelet_price * bracelets_per_day)

/-- Theorem: Given the specific costs and sales plan, it takes 14 days to raise money for the bike -/
theorem bike_fundraising_days :
  days_to_raise_money 112 1 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_bike_fundraising_days_l935_93536


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l935_93500

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length : ℝ) (cost_per_meter : ℝ) : ℝ :=
  let breadth := length - 10
  let perimeter := 2 * (length + breadth)
  cost_per_meter * perimeter

/-- Theorem: The total cost of fencing the given rectangular plot is 5300 currency units -/
theorem fencing_cost_theorem :
  total_fencing_cost 55 26.50 = 5300 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_theorem_l935_93500


namespace NUMINAMATH_CALUDE_income_percentage_l935_93576

theorem income_percentage (juan tim mart : ℝ) 
  (h1 : mart = tim + 0.6 * tim) 
  (h2 : tim = juan - 0.6 * juan) : 
  mart = 0.64 * juan := by
sorry

end NUMINAMATH_CALUDE_income_percentage_l935_93576


namespace NUMINAMATH_CALUDE_square_perimeter_greater_than_circle_circumference_l935_93560

theorem square_perimeter_greater_than_circle_circumference :
  ∀ (a r : ℝ), a > 0 → r > 0 →
  a^2 = π * r^2 →
  4 * a > 2 * π * r :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_greater_than_circle_circumference_l935_93560


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l935_93514

theorem sum_of_squares_of_roots (q r s : ℝ) : 
  (3 * q^3 - 4 * q^2 + 6 * q + 15 = 0) →
  (3 * r^3 - 4 * r^2 + 6 * r + 15 = 0) →
  (3 * s^3 - 4 * s^2 + 6 * s + 15 = 0) →
  q^2 + r^2 + s^2 = -20/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l935_93514


namespace NUMINAMATH_CALUDE_percentage_problem_l935_93593

theorem percentage_problem (P : ℝ) (x : ℝ) (h1 : x = 412.5) 
  (h2 : (P / 100) * x = (1 / 3) * x + 110) : P = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l935_93593


namespace NUMINAMATH_CALUDE_f_parity_l935_93510

-- Define the function f(x) = x|x| + px^2
def f (p : ℝ) (x : ℝ) : ℝ := x * abs x + p * x^2

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem stating the parity of the function depends on p
theorem f_parity (p : ℝ) :
  (p = 0 → is_odd (f p)) ∧
  (p ≠ 0 → ¬(is_even (f p)) ∧ ¬(is_odd (f p))) :=
sorry

end NUMINAMATH_CALUDE_f_parity_l935_93510


namespace NUMINAMATH_CALUDE_john_average_speed_l935_93504

-- Define the start time, break time, end time, and total distance
def start_time : ℕ := 8 * 60 + 15  -- 8:15 AM in minutes
def break_start : ℕ := 12 * 60  -- 12:00 PM in minutes
def break_duration : ℕ := 30  -- 30 minutes
def end_time : ℕ := 14 * 60 + 45  -- 2:45 PM in minutes
def total_distance : ℕ := 240  -- miles

-- Calculate the total driving time in hours
def total_driving_time : ℚ :=
  (break_start - start_time + (end_time - (break_start + break_duration))) / 60

-- Define the average speed
def average_speed : ℚ := total_distance / total_driving_time

-- Theorem to prove
theorem john_average_speed :
  average_speed = 40 :=
sorry

end NUMINAMATH_CALUDE_john_average_speed_l935_93504


namespace NUMINAMATH_CALUDE_first_week_sales_l935_93567

/-- Represents the sales of chips in a convenience store over a month -/
structure ChipSales where
  total : ℕ
  first_week : ℕ
  second_week : ℕ
  third_week : ℕ
  fourth_week : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem first_week_sales (s : ChipSales) :
  s.total = 100 ∧
  s.second_week = 3 * s.first_week ∧
  s.third_week = 20 ∧
  s.fourth_week = 20 ∧
  s.total = s.first_week + s.second_week + s.third_week + s.fourth_week →
  s.first_week = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_week_sales_l935_93567


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l935_93552

/-- Number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute 6 3 = 92 := by sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l935_93552


namespace NUMINAMATH_CALUDE_solve_equation_l935_93596

theorem solve_equation : ∃ x : ℝ, (3 * x) / 4 = 24 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l935_93596


namespace NUMINAMATH_CALUDE_number_reciprocal_problem_l935_93522

theorem number_reciprocal_problem (x : ℝ) (h : 8 * x - 6 = 10) :
  50 * (1 / x) + 150 = 175 := by
  sorry

end NUMINAMATH_CALUDE_number_reciprocal_problem_l935_93522


namespace NUMINAMATH_CALUDE_wheel_configuration_theorem_l935_93540

/-- Represents a configuration of wheels with spokes -/
structure WheelConfiguration where
  total_spokes : Nat
  max_spokes_per_wheel : Nat

/-- Checks if a given number of wheels is possible for the configuration -/
def isPossible (config : WheelConfiguration) (num_wheels : Nat) : Prop :=
  num_wheels * config.max_spokes_per_wheel ≥ config.total_spokes

theorem wheel_configuration_theorem (config : WheelConfiguration) 
  (h1 : config.total_spokes = 7)
  (h2 : config.max_spokes_per_wheel = 3) :
  isPossible config 3 ∧ ¬isPossible config 2 := by
  sorry

#check wheel_configuration_theorem

end NUMINAMATH_CALUDE_wheel_configuration_theorem_l935_93540


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l935_93511

/-- An arithmetic sequence with a common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence -/
def arithmetic_term (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_nonconstant : d ≠ 0)
  (h_geom : ∃ r, 
    arithmetic_term (a 1) d 10 = r * arithmetic_term (a 1) d 5 ∧
    arithmetic_term (a 1) d 20 = r * arithmetic_term (a 1) d 10) :
  ∃ r, r = 2 ∧
    arithmetic_term (a 1) d 10 = r * arithmetic_term (a 1) d 5 ∧
    arithmetic_term (a 1) d 20 = r * arithmetic_term (a 1) d 10 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l935_93511


namespace NUMINAMATH_CALUDE_total_washing_time_l935_93518

/-- The time William spends washing a normal car -/
def normal_car_time : ℕ := 4 + 7 + 4 + 9

/-- The number of normal cars William washed -/
def normal_cars : ℕ := 2

/-- The number of SUVs William washed -/
def suvs : ℕ := 1

/-- The time multiplier for washing an SUV compared to a normal car -/
def suv_time_multiplier : ℕ := 2

/-- Theorem: William spent 96 minutes washing all vehicles -/
theorem total_washing_time : 
  normal_car_time * normal_cars + normal_car_time * suv_time_multiplier * suvs = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_washing_time_l935_93518


namespace NUMINAMATH_CALUDE_ant_return_probability_2006_l935_93509

/-- A regular octahedron --/
structure Octahedron :=
  (vertices : Finset (Fin 6))
  (edges : Finset (Fin 6 × Fin 6))
  (is_regular : True)  -- This is a simplification; we're assuming it's regular

/-- An ant's position on the octahedron --/
structure AntPosition (O : Octahedron) :=
  (vertex : Fin 6)

/-- The probability distribution of the ant's position after n moves --/
def probability_distribution (O : Octahedron) (n : ℕ) : AntPosition O → ℝ := sorry

/-- The probability of the ant returning to the starting vertex after n moves --/
def return_probability (O : Octahedron) (n : ℕ) : ℝ := sorry

/-- The main theorem --/
theorem ant_return_probability_2006 (O : Octahedron) :
  return_probability O 2006 = (2^2005 + 1) / (3 * 2^2006) := by sorry

end NUMINAMATH_CALUDE_ant_return_probability_2006_l935_93509


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l935_93542

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 + i) / (2 - i)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l935_93542


namespace NUMINAMATH_CALUDE_complex_division_real_l935_93505

def complex (a b : ℝ) : ℂ := a + b * Complex.I

theorem complex_division_real (b : ℝ) :
  let z₁ : ℂ := complex 3 (-b)
  let z₂ : ℂ := complex 1 (-2)
  (∃ (r : ℝ), z₁ / z₂ = r) → b = 6 := by sorry

end NUMINAMATH_CALUDE_complex_division_real_l935_93505


namespace NUMINAMATH_CALUDE_polynomial_at_most_one_zero_l935_93502

theorem polynomial_at_most_one_zero (n : ℤ) :
  ∃! (r : ℝ), r^4 - 1994*r^3 + (1993 + n : ℝ)*r^2 - 11*r + (n : ℝ) = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_at_most_one_zero_l935_93502


namespace NUMINAMATH_CALUDE_sibling_pizza_order_l935_93586

-- Define the siblings
inductive Sibling
| Alex
| Beth
| Cyril
| Dan
| Emma

-- Define the function that returns the fraction of pizza eaten by each sibling
def pizza_fraction (s : Sibling) : ℚ :=
  match s with
  | Sibling.Alex => 1/6
  | Sibling.Beth => 1/4
  | Sibling.Cyril => 1/5
  | Sibling.Dan => 1/3
  | Sibling.Emma => 1 - (1/6 + 1/4 + 1/5 + 1/3)

-- Define the order of siblings
def sibling_order : List Sibling :=
  [Sibling.Dan, Sibling.Beth, Sibling.Cyril, Sibling.Alex, Sibling.Emma]

-- Theorem statement
theorem sibling_pizza_order : 
  List.Pairwise (λ a b => pizza_fraction a > pizza_fraction b) sibling_order :=
sorry

end NUMINAMATH_CALUDE_sibling_pizza_order_l935_93586


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l935_93551

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop :=
  3 * x^2 - 36 * y^2 - 18 * x + 27 = 0

/-- The two lines represented by the equation -/
def line1 (x y : ℝ) : Prop :=
  x = 3 + 2 * Real.sqrt 3 * y

def line2 (x y : ℝ) : Prop :=
  x = 3 - 2 * Real.sqrt 3 * y

/-- Theorem stating that the equation represents two lines -/
theorem equation_represents_two_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l935_93551


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l935_93523

theorem cubic_equation_roots :
  let f : ℝ → ℝ := λ x ↦ x^3 - 2*x
  f 0 = 0 ∧ f (Real.sqrt 2) = 0 ∧ f (-Real.sqrt 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l935_93523


namespace NUMINAMATH_CALUDE_planks_per_table_is_15_l935_93562

def planks_per_table (trees : ℕ) (planks_per_tree : ℕ) (table_price : ℕ) (labor_cost : ℕ) (profit : ℕ) : ℕ :=
  let total_planks := trees * planks_per_tree
  let total_revenue := profit + labor_cost
  let num_tables := total_revenue / table_price
  total_planks / num_tables

theorem planks_per_table_is_15 :
  planks_per_table 30 25 300 3000 12000 = 15 := by
  sorry

end NUMINAMATH_CALUDE_planks_per_table_is_15_l935_93562


namespace NUMINAMATH_CALUDE_log_sum_equality_l935_93578

theorem log_sum_equality : 
  Real.sqrt (Real.log 8 / Real.log 4 + Real.log 10 / Real.log 5) = 
  Real.sqrt (5 / 2 + Real.log 2 / Real.log 5) := by
sorry

end NUMINAMATH_CALUDE_log_sum_equality_l935_93578


namespace NUMINAMATH_CALUDE_at_least_one_is_diff_of_squares_l935_93531

theorem at_least_one_is_diff_of_squares (a b : ℕ) : 
  ∃ (x y z w : ℤ), (a = x^2 - y^2) ∨ (b = z^2 - w^2) ∨ (a + b = x^2 - y^2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_is_diff_of_squares_l935_93531


namespace NUMINAMATH_CALUDE_line_through_P_with_equal_intercepts_l935_93526

/-- A line in the 2D plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The point P(2,3) -/
def P : ℝ × ℝ := (2, 3)

/-- A line passes through a point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- A line has equal intercepts on both coordinate axes -/
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ -l.c/l.a = -l.c/l.b

/-- The equation of the line is y = 3/2 * x -/
def is_y_eq_3_2x (l : Line) : Prop :=
  l.a = 2 ∧ l.b = -3 ∧ l.c = 0

/-- The equation of the line is x + y - 5 = 0 -/
def is_x_plus_y_eq_5 (l : Line) : Prop :=
  l.a = 1 ∧ l.b = 1 ∧ l.c = -5

theorem line_through_P_with_equal_intercepts (l : Line) :
  passes_through l P ∧ has_equal_intercepts l →
  is_y_eq_3_2x l ∨ is_x_plus_y_eq_5 l := by
  sorry

end NUMINAMATH_CALUDE_line_through_P_with_equal_intercepts_l935_93526


namespace NUMINAMATH_CALUDE_team_games_count_l935_93568

/-- The number of games the team plays -/
def total_games : ℕ := 14

/-- The number of shots John gets per foul -/
def shots_per_foul : ℕ := 2

/-- The number of times John gets fouled per game -/
def fouls_per_game : ℕ := 5

/-- The percentage of games John plays, expressed as a rational number -/
def games_played_percentage : ℚ := 4/5

/-- The total number of free throws John gets -/
def total_free_throws : ℕ := 112

/-- Theorem stating that the number of games the team plays is 14 -/
theorem team_games_count : 
  (shots_per_foul * fouls_per_game * (games_played_percentage * total_games) : ℚ) = total_free_throws := by
  sorry

end NUMINAMATH_CALUDE_team_games_count_l935_93568


namespace NUMINAMATH_CALUDE_f_equals_g_l935_93561

-- Define the two functions
def f (x : ℝ) : ℝ := x - 1
def g (t : ℝ) : ℝ := t - 1

-- Theorem statement
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l935_93561


namespace NUMINAMATH_CALUDE_count_valid_bouquets_l935_93570

/-- The number of valid bouquet combinations -/
def num_bouquets : ℕ := 11

/-- Represents a bouquet with roses and carnations -/
structure Bouquet where
  roses : ℕ
  carnations : ℕ

/-- The cost of a single rose -/
def rose_cost : ℕ := 4

/-- The cost of a single carnation -/
def carnation_cost : ℕ := 2

/-- The total budget for the bouquet -/
def total_budget : ℕ := 60

/-- Checks if a bouquet is valid according to the problem constraints -/
def is_valid_bouquet (b : Bouquet) : Prop :=
  b.roses ≥ 5 ∧
  b.roses * rose_cost + b.carnations * carnation_cost = total_budget

/-- The main theorem stating that there are exactly 11 valid bouquet combinations -/
theorem count_valid_bouquets :
  (∃ (bouquets : Finset Bouquet),
    bouquets.card = num_bouquets ∧
    (∀ b ∈ bouquets, is_valid_bouquet b) ∧
    (∀ b : Bouquet, is_valid_bouquet b → b ∈ bouquets)) :=
sorry

end NUMINAMATH_CALUDE_count_valid_bouquets_l935_93570


namespace NUMINAMATH_CALUDE_condition_relationship_l935_93530

theorem condition_relationship (a : ℝ) : 
  (a = 1 → a^2 - 3*a + 2 = 0) ∧ 
  (∃ b : ℝ, b ≠ 1 ∧ b^2 - 3*b + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l935_93530


namespace NUMINAMATH_CALUDE_cosine_product_square_root_l935_93583

theorem cosine_product_square_root : 
  Real.sqrt ((2 - Real.cos (π / 9) ^ 2) * (2 - Real.cos (2 * π / 9) ^ 2) * (2 - Real.cos (3 * π / 9) ^ 2)) = Real.sqrt 377 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_square_root_l935_93583


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_s_max_min_l935_93525

theorem sum_of_reciprocals_of_s_max_min (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) : 
  let s := x^2 + y^2
  ∃ (s_max s_min : ℝ), (∀ (x' y' : ℝ), 4 * x'^2 - 5 * x' * y' + 4 * y'^2 = 5 → x'^2 + y'^2 ≤ s_max) ∧
                       (∀ (x' y' : ℝ), 4 * x'^2 - 5 * x' * y' + 4 * y'^2 = 5 → s_min ≤ x'^2 + y'^2) ∧
                       (1 / s_max + 1 / s_min = 8 / 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_s_max_min_l935_93525


namespace NUMINAMATH_CALUDE_pgcd_and_divisibility_properties_l935_93543

/-- For a ≥ 2 and m ≥ n ≥ 1, prove properties of PGCD and divisibility -/
theorem pgcd_and_divisibility_properties 
  (a m n : ℕ) 
  (ha : a ≥ 2) 
  (hmn : m ≥ n) 
  (hn : n ≥ 1) :
  (Nat.gcd (a^m - 1) (a^n - 1) = Nat.gcd (a^(m-n) - 1) (a^n - 1)) ∧
  (Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1) ∧
  ((a^m - 1) ∣ (a^n - 1) ↔ m ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_pgcd_and_divisibility_properties_l935_93543


namespace NUMINAMATH_CALUDE_colin_average_time_l935_93559

/-- Represents Colin's running times for each mile -/
def colinTimes : List ℕ := [6, 5, 5, 4]

/-- The number of miles Colin ran -/
def totalMiles : ℕ := colinTimes.length

/-- Calculates the average time per mile -/
def averageTime : ℚ := (colinTimes.sum : ℚ) / totalMiles

theorem colin_average_time :
  averageTime = 5 := by sorry

end NUMINAMATH_CALUDE_colin_average_time_l935_93559


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l935_93565

-- Define the parabola equation
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

-- Define the x-intercept
def a : ℝ := parabola 0

-- Define the y-intercepts
def b_and_c : Set ℝ := {y | parabola y = 0}

-- Theorem statement
theorem parabola_intercepts_sum :
  ∃ (b c : ℝ), b ∈ b_and_c ∧ c ∈ b_and_c ∧ b ≠ c ∧ a + b + c = 7 :=
sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l935_93565


namespace NUMINAMATH_CALUDE_puppies_percentage_proof_l935_93553

/-- The percentage of students who have puppies in Professor Plum's biology class -/
def percentage_with_puppies : ℝ := 80

theorem puppies_percentage_proof (total_students : ℕ) (both_puppies_parrots : ℕ) 
  (h1 : total_students = 40)
  (h2 : both_puppies_parrots = 8)
  (h3 : (25 : ℝ) / 100 * (percentage_with_puppies / 100 * total_students) = both_puppies_parrots) :
  percentage_with_puppies = 80 := by
  sorry

#check puppies_percentage_proof

end NUMINAMATH_CALUDE_puppies_percentage_proof_l935_93553


namespace NUMINAMATH_CALUDE_milk_packet_price_problem_l935_93566

/-- Given 5 packets of milk with an average price of 20 cents, if 2 packets are returned
    and the average price of the remaining 3 packets is 12 cents, then the average price
    of the 2 returned packets is 32 cents. -/
theorem milk_packet_price_problem (total_packets : Nat) (remaining_packets : Nat) 
    (initial_avg_price : ℚ) (remaining_avg_price : ℚ) :
  total_packets = 5 →
  remaining_packets = 3 →
  initial_avg_price = 20 →
  remaining_avg_price = 12 →
  let returned_packets := total_packets - remaining_packets
  let total_cost := total_packets * initial_avg_price
  let remaining_cost := remaining_packets * remaining_avg_price
  let returned_cost := total_cost - remaining_cost
  (returned_cost / returned_packets : ℚ) = 32 := by
sorry

end NUMINAMATH_CALUDE_milk_packet_price_problem_l935_93566


namespace NUMINAMATH_CALUDE_special_triangle_exists_l935_93521

-- Define the color type
inductive Color
| Red
| Green
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
def colorFunction : Point → Color := sorry

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the circumradius of a triangle
def circumradius (t : Triangle) : ℝ := sorry

-- Define a predicate for monochromatic triangle
def isMonochromatic (t : Triangle) : Prop :=
  colorFunction t.A = colorFunction t.B ∧ colorFunction t.B = colorFunction t.C

-- Define a predicate for angle ratio condition
def satisfiesAngleRatio (t : Triangle) : Prop := sorry

-- The main theorem
theorem special_triangle_exists :
  ∃ (t : Triangle), isMonochromatic t ∧ circumradius t = 2008 ∧ satisfiesAngleRatio t := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_exists_l935_93521


namespace NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l935_93557

/-- 
Given an arithmetic sequence with:
- First term a = 2
- Last term l = 2008
- Common difference d = 3

Prove that the number of terms in the sequence is 669.
-/
theorem arithmetic_sequence_term_count : 
  ∀ (a l d n : ℕ), 
    a = 2 → 
    l = 2008 → 
    d = 3 → 
    l = a + (n - 1) * d → 
    n = 669 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l935_93557


namespace NUMINAMATH_CALUDE_exists_tangent_circle_l935_93591

-- Define the basic geometric objects
structure Point :=
  (x y : ℝ)

structure Line :=
  (a b c : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the given objects
variable (M : Point)
variable (l : Line)
variable (S : Circle)

-- Define the tangency and passing through relations
def isTangentToLine (c : Circle) (l : Line) : Prop := sorry
def isTangentToCircle (c1 c2 : Circle) : Prop := sorry
def passesThrough (c : Circle) (p : Point) : Prop := sorry

-- Theorem statement
theorem exists_tangent_circle :
  ∃ (Ω : Circle),
    passesThrough Ω M ∧
    isTangentToLine Ω l ∧
    isTangentToCircle Ω S :=
sorry

end NUMINAMATH_CALUDE_exists_tangent_circle_l935_93591


namespace NUMINAMATH_CALUDE_optimal_allocation_l935_93563

/-- Represents the production capacity of workers in a workshop --/
structure Workshop where
  total_workers : ℕ
  bolts_per_worker : ℕ
  nuts_per_worker : ℕ
  nuts_per_bolt : ℕ

/-- Represents the allocation of workers to bolt and nut production --/
structure WorkerAllocation where
  bolt_workers : ℕ
  nut_workers : ℕ

/-- Checks if a given allocation is valid for the workshop --/
def is_valid_allocation (w : Workshop) (a : WorkerAllocation) : Prop :=
  a.bolt_workers + a.nut_workers = w.total_workers ∧
  a.bolt_workers * w.bolts_per_worker * w.nuts_per_bolt = a.nut_workers * w.nuts_per_worker

/-- The theorem stating the optimal allocation for the given workshop conditions --/
theorem optimal_allocation (w : Workshop) 
    (h1 : w.total_workers = 28)
    (h2 : w.bolts_per_worker = 12)
    (h3 : w.nuts_per_worker = 18)
    (h4 : w.nuts_per_bolt = 2) :
  ∃ (a : WorkerAllocation), 
    is_valid_allocation w a ∧ 
    a.bolt_workers = 12 ∧ 
    a.nut_workers = 16 := by
  sorry

end NUMINAMATH_CALUDE_optimal_allocation_l935_93563


namespace NUMINAMATH_CALUDE_modified_tic_tac_toe_tie_probability_l935_93528

theorem modified_tic_tac_toe_tie_probability 
  (amy_win_prob : ℚ) 
  (lily_win_prob : ℚ) 
  (h1 : amy_win_prob = 2/5) 
  (h2 : lily_win_prob = 1/4) 
  (h3 : amy_win_prob ≥ 2 * lily_win_prob ∨ lily_win_prob ≥ 2 * amy_win_prob) : 
  1 - (amy_win_prob + lily_win_prob) = 7/20 :=
by sorry

end NUMINAMATH_CALUDE_modified_tic_tac_toe_tie_probability_l935_93528


namespace NUMINAMATH_CALUDE_triangle_properties_l935_93554

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the main results -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.A = 2 * t.B) : 
  (t.b = 2 ∧ t.c = 1 → t.a = Real.sqrt 6) ∧
  (t.b + t.c = Real.sqrt 3 * t.a → t.B = π / 6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l935_93554


namespace NUMINAMATH_CALUDE_train_length_l935_93541

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * 1000 / 3600 → 
  crossing_time = 30 → 
  bridge_length = 255 → 
  train_speed * crossing_time - bridge_length = 120 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l935_93541


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l935_93550

theorem modular_arithmetic_problem :
  ∃ (a b : ℕ), 
    (7 * a) % 60 = 1 ∧ 
    (13 * b) % 60 = 1 ∧ 
    ((3 * a + 6 * b) % 60) = 51 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l935_93550


namespace NUMINAMATH_CALUDE_train_passing_pole_time_l935_93524

/-- Proves that a train with given speed and crossing time will take 10 seconds to pass a pole -/
theorem train_passing_pole_time 
  (train_speed_kmh : ℝ) 
  (stationary_train_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_speed_kmh = 72) 
  (h2 : stationary_train_length = 500) 
  (h3 : crossing_time = 35) :
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let train_length := train_speed_ms * crossing_time - stationary_train_length
  train_length / train_speed_ms = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_pole_time_l935_93524


namespace NUMINAMATH_CALUDE_matching_shoe_probability_l935_93558

/-- The probability of selecting a matching pair of shoes from a box containing 6 pairs -/
theorem matching_shoe_probability (total_shoes : ℕ) (total_pairs : ℕ) (h1 : total_shoes = 12) (h2 : total_pairs = 6) :
  (total_pairs : ℚ) / ((total_shoes.choose 2) : ℚ) = 1 / 11 := by
  sorry

#check matching_shoe_probability

end NUMINAMATH_CALUDE_matching_shoe_probability_l935_93558


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l935_93572

theorem diophantine_equation_solutions (x y z : ℤ) :
  (x * y / z : ℚ) + (y * z / x : ℚ) + (z * x / y : ℚ) = 3 ↔
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 1 ∧ y = -1 ∧ z = -1) ∨
  (x = -1 ∧ y = 1 ∧ z = -1) ∨
  (x = -1 ∧ y = -1 ∧ z = 1) := by
  sorry

#check diophantine_equation_solutions

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l935_93572


namespace NUMINAMATH_CALUDE_square_less_than_power_of_three_l935_93555

theorem square_less_than_power_of_three (n : ℕ) (h : n ≥ 3) : (n + 1)^2 < 3^n := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_power_of_three_l935_93555


namespace NUMINAMATH_CALUDE_last_three_digits_is_419_l935_93599

/-- A function that generates the nth digit in the list of increasing positive integers starting with 2 -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- The three-digit number formed by the 1998th, 1999th, and 2000th digits -/
def lastThreeDigits : ℕ := 
  100 * (nthDigit 1998) + 10 * (nthDigit 1999) + (nthDigit 2000)

/-- Theorem stating that the last three digits form the number 419 -/
theorem last_three_digits_is_419 : lastThreeDigits = 419 := by sorry

end NUMINAMATH_CALUDE_last_three_digits_is_419_l935_93599


namespace NUMINAMATH_CALUDE_find_d_l935_93515

theorem find_d (a b c d : ℕ+) 
  (eq1 : a^2 = c * (d + 20))
  (eq2 : b^2 = c * (d - 18)) : 
  d = 180 := by
sorry

end NUMINAMATH_CALUDE_find_d_l935_93515


namespace NUMINAMATH_CALUDE_number_categorization_l935_93584

def given_numbers : List ℚ := [8, -1, -2/5, 3/5, 0, 1/3, -10/7, 5, -20/7]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative (x : ℚ) : Prop := x < 0
def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n
def is_fraction (x : ℚ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b
def is_non_negative_rational (x : ℚ) : Prop := x ≥ 0

def positive_set : Set ℚ := {x | is_positive x}
def negative_set : Set ℚ := {x | is_negative x}
def integer_set : Set ℚ := {x | is_integer x}
def fraction_set : Set ℚ := {x | is_fraction x}
def non_negative_rational_set : Set ℚ := {x | is_non_negative_rational x}

theorem number_categorization :
  positive_set = {8, 3/5, 1/3, 5} ∧
  negative_set = {-1, -2/5, -10/7, -20/7} ∧
  integer_set = {8, -1, 0, 5} ∧
  fraction_set = {-2/5, 3/5, 1/3, -10/7, -20/7} ∧
  non_negative_rational_set = {8, 3/5, 0, 1/3, 5} := by
  sorry

end NUMINAMATH_CALUDE_number_categorization_l935_93584


namespace NUMINAMATH_CALUDE_connor_hourly_wage_l935_93503

def sarah_daily_wage : ℝ := 288
def sarah_hours_worked : ℝ := 8
def sarah_connor_wage_ratio : ℝ := 6

theorem connor_hourly_wage :
  let sarah_hourly_wage := sarah_daily_wage / sarah_hours_worked
  sarah_hourly_wage / sarah_connor_wage_ratio = 6 := by
  sorry

end NUMINAMATH_CALUDE_connor_hourly_wage_l935_93503


namespace NUMINAMATH_CALUDE_painting_time_with_break_l935_93513

/-- The time it takes Doug and Dave to paint a room together, including a break -/
theorem painting_time_with_break (doug_time dave_time break_time : ℝ) 
  (h_doug : doug_time = 4)
  (h_dave : dave_time = 6)
  (h_break : break_time = 2) : 
  ∃ s : ℝ, s = 22 / 5 ∧ 
  (1 / doug_time + 1 / dave_time) * (s - break_time) = 1 := by
  sorry

end NUMINAMATH_CALUDE_painting_time_with_break_l935_93513


namespace NUMINAMATH_CALUDE_distribution_methods_eq_72_l935_93573

/-- Number of teachers -/
def num_teachers : ℕ := 3

/-- Number of students -/
def num_students : ℕ := 3

/-- Total number of tickets -/
def total_tickets : ℕ := 6

/-- Function to calculate the number of distribution methods -/
def distribution_methods : ℕ := sorry

/-- Theorem stating that the number of distribution methods is 72 -/
theorem distribution_methods_eq_72 : distribution_methods = 72 := by sorry

end NUMINAMATH_CALUDE_distribution_methods_eq_72_l935_93573


namespace NUMINAMATH_CALUDE_ticket_queue_arrangements_l935_93549

/-- Represents the number of valid arrangements for a ticket queue --/
def validArrangements (n : ℕ) : ℕ :=
  Nat.factorial (2 * n) / (Nat.factorial n * Nat.factorial (n + 1))

/-- Theorem stating the number of valid arrangements for a ticket queue --/
theorem ticket_queue_arrangements (n : ℕ) :
  validArrangements n = 
    let total_people := 2 * n
    let people_with_five_yuan := n
    let people_with_ten_yuan := n
    let ticket_price := 5
    -- The actual number of valid arrangements
    Nat.factorial total_people / (Nat.factorial people_with_five_yuan * Nat.factorial (people_with_ten_yuan + 1)) :=
by sorry

#check ticket_queue_arrangements

end NUMINAMATH_CALUDE_ticket_queue_arrangements_l935_93549


namespace NUMINAMATH_CALUDE_intersection_of_lines_l935_93534

/-- Given four points in 3D space, this theorem states that the intersection of the lines
    passing through the first two points and the last two points is at a specific coordinate. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) : 
  A = (5, -6, 8) →
  B = (15, -16, 13) →
  C = (1, 4, -5) →
  D = (3, -4, 11) →
  ∃ t s : ℝ, 
    (5 + 10*t, -6 - 10*t, 8 + 5*t) = (1 + 2*s, 4 - 8*s, -5 + 16*s) ∧
    (5 + 10*t, -6 - 10*t, 8 + 5*t) = (3, -4, 7) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l935_93534


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l935_93579

/-- A parabola in the cartesian coordinate plane with equation y^2 = -16x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  h : equation = fun x y ↦ y^2 = -16*x

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Theorem: The coordinates of the focus of the parabola y^2 = -16x are (-4, 0) -/
theorem parabola_focus_coordinates (p : Parabola) : focus p = (-4, 0) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l935_93579


namespace NUMINAMATH_CALUDE_race_distance_l935_93592

/-- Represents the race scenario where p runs x% faster than q, q has a head start, and the race ends in a tie -/
def race_scenario (x y : ℝ) : Prop :=
  ∀ (vq : ℝ), vq > 0 →
    let vp := vq * (1 + x / 100)
    let head_start := (x / 10) * y
    let dq := 1000 * y / x
    let dp := dq + head_start
    dq / vq = dp / vp

/-- The theorem stating that both runners cover the same distance in the given race scenario -/
theorem race_distance (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  race_scenario x y →
  ∃ (d : ℝ), d = 1000 * y / x ∧ 
    (∀ (vq : ℝ), vq > 0 →
      let vp := vq * (1 + x / 100)
      let head_start := (x / 10) * y
      d = 1000 * y / x ∧ d + head_start = (10000 * y + x * y^2) / (10 * x)) :=
sorry

end NUMINAMATH_CALUDE_race_distance_l935_93592


namespace NUMINAMATH_CALUDE_flea_collar_count_l935_93581

/-- Represents the number of dogs with flea collars in a kennel -/
def dogs_with_flea_collars (total : ℕ) (with_tags : ℕ) (with_both : ℕ) (with_neither : ℕ) : ℕ :=
  total - with_tags + with_both - with_neither

/-- Theorem stating that in a kennel of 80 dogs, where 45 dogs wear tags, 
    6 dogs wear both tags and flea collars, and 1 dog wears neither, 
    the number of dogs wearing flea collars is 40. -/
theorem flea_collar_count : 
  dogs_with_flea_collars 80 45 6 1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_flea_collar_count_l935_93581


namespace NUMINAMATH_CALUDE_managers_salary_l935_93532

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 ∧ 
  avg_salary = 1200 ∧ 
  salary_increase = 100 → 
  (num_employees * avg_salary + (avg_salary + salary_increase) * (num_employees + 1) - num_employees * avg_salary) = 3300 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_l935_93532


namespace NUMINAMATH_CALUDE_power_function_continuous_l935_93590

theorem power_function_continuous (n : ℕ+) :
  Continuous (fun x : ℝ => x ^ (n : ℕ)) :=
sorry

end NUMINAMATH_CALUDE_power_function_continuous_l935_93590


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l935_93556

theorem simplify_algebraic_expression (a b : ℝ) (h : b ≠ 0) :
  (14 * a^3 * b^2 - 7 * a * b^2) / (7 * a * b^2) = 2 * a^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l935_93556


namespace NUMINAMATH_CALUDE_bears_per_shelf_l935_93519

theorem bears_per_shelf (initial_stock : ℕ) (new_shipment : ℕ) (num_shelves : ℕ) :
  initial_stock = 4 →
  new_shipment = 10 →
  num_shelves = 2 →
  (initial_stock + new_shipment) / num_shelves = 7 :=
by sorry

end NUMINAMATH_CALUDE_bears_per_shelf_l935_93519


namespace NUMINAMATH_CALUDE_probability_at_least_one_of_three_l935_93538

theorem probability_at_least_one_of_three (p : ℝ) (h : p = 1 / 3) :
  1 - (1 - p)^3 = 19 / 27 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_of_three_l935_93538


namespace NUMINAMATH_CALUDE_expression_value_l935_93580

theorem expression_value : 5^3 - 3 * 5^2 + 3 * 5 - 1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l935_93580


namespace NUMINAMATH_CALUDE_circle_diameter_property_l935_93539

theorem circle_diameter_property (BC BD DA : ℝ) (h1 : BC = Real.sqrt 901) (h2 : BD = 1) (h3 : DA = 16) : ∃ EC : ℝ, EC = 1 ∧ BC * EC = BD * (BC - BD) := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_property_l935_93539


namespace NUMINAMATH_CALUDE_regression_lines_intersect_l935_93507

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a regression line -/
def lies_on (line : RegressionLine) (x y : ℝ) : Prop :=
  y = line.slope * x + line.intercept

/-- Theorem: Two regression lines with the same average observed values intersect at those average values -/
theorem regression_lines_intersect (t1 t2 : RegressionLine) (s t : ℝ)
  (h1 : lies_on t1 s t)
  (h2 : lies_on t2 s t) :
  ∃ (x y : ℝ), lies_on t1 x y ∧ lies_on t2 x y ∧ x = s ∧ y = t :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersect_l935_93507


namespace NUMINAMATH_CALUDE_a_is_geometric_sequence_l935_93585

/-- A linear function f(x) = bx + 1 where b is a constant not equal to 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := b * x + 1

/-- A recursive function g(n) defined as:
    g(0) = 1
    g(n) = f(g(n-1)) for n ≥ 1 -/
def g (b : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => f b (g b n)

/-- The sequence a_n defined as a_n = g(n) - g(n-1) for n ∈ ℕ* -/
def a (b : ℝ) (n : ℕ) : ℝ := g b (n + 1) - g b n

/-- Theorem: The sequence {a_n} is a geometric sequence -/
theorem a_is_geometric_sequence (b : ℝ) (h : b ≠ 1) :
  ∃ r : ℝ, ∀ n : ℕ, a b (n + 1) = r * a b n :=
sorry

end NUMINAMATH_CALUDE_a_is_geometric_sequence_l935_93585


namespace NUMINAMATH_CALUDE_estimate_larger_than_original_l935_93516

theorem estimate_larger_than_original 
  (x y ε δ : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : x > y) 
  (hε : ε > 0) 
  (hδ : δ > 0) 
  (hεδ : ε ≠ δ) : 
  (x + ε) - (y - δ) > x - y := by
sorry

end NUMINAMATH_CALUDE_estimate_larger_than_original_l935_93516


namespace NUMINAMATH_CALUDE_fraction_equality_l935_93506

theorem fraction_equality (m n p q r : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5)
  (h4 : m / r = 10) :
  r / q = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l935_93506


namespace NUMINAMATH_CALUDE_album_distribution_l935_93512

/-- The number of ways to distribute n identical items among k people --/
def distribute (n k : ℕ) : ℕ := Nat.choose k n

theorem album_distribution :
  let photo_albums := 2
  let stamp_albums := 3
  let total_albums := photo_albums + stamp_albums
  let friends := 5
  distribute photo_albums friends * distribute stamp_albums (friends - photo_albums) = 10 := by
  sorry

end NUMINAMATH_CALUDE_album_distribution_l935_93512


namespace NUMINAMATH_CALUDE_arun_weight_average_l935_93574

theorem arun_weight_average :
  let min_weight := 61
  let max_weight := 64
  let average := (min_weight + max_weight) / 2
  (∀ w, min_weight < w ∧ w ≤ max_weight → 
    w > 60 ∧ w < 70 ∧ w > 61 ∧ w < 72 ∧ w ≤ 64) →
  average = 62.5 := by
sorry

end NUMINAMATH_CALUDE_arun_weight_average_l935_93574


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l935_93547

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 7) :
  let perimeter := a + 2 * b
  perimeter = 17 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l935_93547


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l935_93533

def A : Set ℤ := {-2, -1, 3, 4}
def B : Set ℤ := {-1, 2, 3}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l935_93533


namespace NUMINAMATH_CALUDE_unique_triple_solution_l935_93589

theorem unique_triple_solution : 
  ∃! (x y z : ℝ), x + y = 2 ∧ x * y - z^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l935_93589


namespace NUMINAMATH_CALUDE_parallel_transitivity_l935_93527

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop :=
  ∀ (x y : ℝ), l1 x y ↔ l2 x y

-- State the theorem
theorem parallel_transitivity (l1 l2 l3 : Line) :
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l935_93527


namespace NUMINAMATH_CALUDE_base_8_6_equality_l935_93544

/-- Checks if a number is a valid digit in a given base -/
def isValidDigit (digit : ℕ) (base : ℕ) : Prop :=
  digit < base

/-- Converts a two-digit number from a given base to base 10 -/
def toBase10 (c d : ℕ) (base : ℕ) : ℕ :=
  base * c + d

/-- The main theorem stating that 0 is the only number satisfying the conditions -/
theorem base_8_6_equality (n : ℕ) : n > 0 → 
  (∃ (c d : ℕ), isValidDigit c 8 ∧ isValidDigit d 8 ∧ 
   isValidDigit c 6 ∧ isValidDigit d 6 ∧
   n = toBase10 c d 8 ∧ n = toBase10 d c 6) → n = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_8_6_equality_l935_93544


namespace NUMINAMATH_CALUDE_new_apartment_rent_is_1400_l935_93535

/-- Calculates the monthly rent of John's new apartment -/
def new_apartment_rent (former_rent_per_sqft : ℚ) (former_sqft : ℕ) (annual_savings : ℚ) : ℚ :=
  let former_monthly_rent := former_rent_per_sqft * former_sqft
  let former_annual_rent := former_monthly_rent * 12
  let new_annual_rent := former_annual_rent - annual_savings
  new_annual_rent / 12

/-- Proves that the monthly rent of John's new apartment is $1400 -/
theorem new_apartment_rent_is_1400 :
  new_apartment_rent 2 750 1200 = 1400 := by sorry

end NUMINAMATH_CALUDE_new_apartment_rent_is_1400_l935_93535


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_24_l935_93529

theorem largest_four_digit_divisible_by_24 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 24 ∣ n → n ≤ 9984 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_24_l935_93529


namespace NUMINAMATH_CALUDE_cylinder_not_unique_l935_93571

theorem cylinder_not_unique (S V : ℝ) (h_pos_S : S > 0) (h_pos_V : V > 0)
  (h_inequality : S > 3 * (2 * π * V^2)^(1/3)) :
  ∃ (r₁ r₂ h₁ h₂ : ℝ),
    r₁ ≠ r₂ ∧
    2 * π * r₁ * h₁ + 2 * π * r₁^2 = S ∧
    2 * π * r₂ * h₂ + 2 * π * r₂^2 = S ∧
    π * r₁^2 * h₁ = V ∧
    π * r₂^2 * h₂ = V :=
by sorry

end NUMINAMATH_CALUDE_cylinder_not_unique_l935_93571


namespace NUMINAMATH_CALUDE_jeff_total_hours_l935_93588

/-- Represents Jeff's weekly schedule --/
structure JeffSchedule where
  facebook_hours_per_day : ℕ
  weekend_work_ratio : ℕ
  twitter_hours_per_weekend_day : ℕ
  instagram_hours_per_weekday : ℕ
  weekday_work_ratio : ℕ

/-- Calculates Jeff's total hours spent on work, Twitter, and Instagram in a week --/
def total_hours (schedule : JeffSchedule) : ℕ :=
  let weekend_work_hours := 2 * (schedule.facebook_hours_per_day / schedule.weekend_work_ratio)
  let weekday_work_hours := 5 * (4 * (schedule.facebook_hours_per_day + schedule.instagram_hours_per_weekday))
  let twitter_hours := 2 * schedule.twitter_hours_per_weekend_day
  let instagram_hours := 5 * schedule.instagram_hours_per_weekday
  weekend_work_hours + weekday_work_hours + twitter_hours + instagram_hours

/-- Theorem stating Jeff's total hours in a week --/
theorem jeff_total_hours : 
  ∀ (schedule : JeffSchedule),
    schedule.facebook_hours_per_day = 3 ∧
    schedule.weekend_work_ratio = 3 ∧
    schedule.twitter_hours_per_weekend_day = 2 ∧
    schedule.instagram_hours_per_weekday = 1 ∧
    schedule.weekday_work_ratio = 4 →
    total_hours schedule = 91 := by
  sorry

end NUMINAMATH_CALUDE_jeff_total_hours_l935_93588


namespace NUMINAMATH_CALUDE_sand_cone_weight_l935_93545

/-- The weight of a sand cone given its dimensions and sand density -/
theorem sand_cone_weight (diameter : ℝ) (height_ratio : ℝ) (sand_density : ℝ) :
  diameter = 12 →
  height_ratio = 0.8 →
  sand_density = 100 →
  let radius := diameter / 2
  let height := height_ratio * diameter
  let volume := (1/3) * π * radius^2 * height
  volume * sand_density = 11520 * π :=
by sorry

end NUMINAMATH_CALUDE_sand_cone_weight_l935_93545


namespace NUMINAMATH_CALUDE_even_monotone_increasing_inequality_l935_93508

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_monotone_increasing_inequality
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  f (-2) > f (-1) ∧ f (-1) > f 0 := by
  sorry

end NUMINAMATH_CALUDE_even_monotone_increasing_inequality_l935_93508


namespace NUMINAMATH_CALUDE_caitlin_age_l935_93537

theorem caitlin_age (anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ) : 
  anna_age = 48 →
  brianna_age = anna_age / 2 →
  caitlin_age = brianna_age - 7 →
  caitlin_age = 17 := by
sorry

end NUMINAMATH_CALUDE_caitlin_age_l935_93537


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l935_93520

theorem smallest_number_divisible (n : ℕ) : n = 257 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ : ℕ, 
    m + 7 = 8 * k₁ ∧ 
    m + 7 = 11 * k₂ ∧ 
    m + 7 = 24 * k₃)) ∧ 
  (∃ k₁ k₂ k₃ : ℕ, 
    n + 7 = 8 * k₁ ∧ 
    n + 7 = 11 * k₂ ∧ 
    n + 7 = 24 * k₃) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l935_93520


namespace NUMINAMATH_CALUDE_salt_production_january_l935_93569

/-- The salt production problem -/
theorem salt_production_january (
  monthly_increase : ℕ → ℝ)
  (average_daily_production : ℝ)
  (h1 : ∀ n : ℕ, n ≥ 1 ∧ n ≤ 11 → monthly_increase n = 100)
  (h2 : average_daily_production = 100.27397260273973)
  (h3 : ∃ january_production : ℝ,
    (january_production +
      (january_production + monthly_increase 1) +
      (january_production + monthly_increase 1 + monthly_increase 2) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7 + monthly_increase 8) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7 + monthly_increase 8 + monthly_increase 9) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7 + monthly_increase 8 + monthly_increase 9 + monthly_increase 10) +
      (january_production + monthly_increase 1 + monthly_increase 2 + monthly_increase 3 + monthly_increase 4 + monthly_increase 5 + monthly_increase 6 + monthly_increase 7 + monthly_increase 8 + monthly_increase 9 + monthly_increase 10 + monthly_increase 11)) / 365 = average_daily_production) :
  ∃ january_production : ℝ, january_production = 2500 :=
by sorry

end NUMINAMATH_CALUDE_salt_production_january_l935_93569


namespace NUMINAMATH_CALUDE_factorization_sum_l935_93546

theorem factorization_sum (a b : ℤ) : 
  (∀ x, 25 * x^2 - 160 * x - 144 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -68 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l935_93546


namespace NUMINAMATH_CALUDE_toothpick_problem_l935_93594

theorem toothpick_problem (n : ℕ) : 
  n > 5000 ∧
  n % 10 = 9 ∧
  n % 9 = 8 ∧
  n % 8 = 7 ∧
  n % 7 = 6 ∧
  n % 6 = 5 ∧
  n % 5 = 4 →
  n = 5039 :=
by sorry

end NUMINAMATH_CALUDE_toothpick_problem_l935_93594


namespace NUMINAMATH_CALUDE_tangent_line_slope_relation_l935_93598

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem tangent_line_slope_relation (a b : ℝ) :
  a^2 + b = 0 →
  ∃ (m n : ℝ),
    let k1 := 3*m^2 + 2*a*m + b
    let k2 := 3*n^2 + 2*a*n + b
    k2 = 4*k1 →
    a^2 = 3*b :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_relation_l935_93598


namespace NUMINAMATH_CALUDE_ellipse_b_value_l935_93577

/-- An ellipse with foci at (1, 1) and (1, -1) passing through (7, 0) -/
structure Ellipse where
  foci1 : ℝ × ℝ := (1, 1)
  foci2 : ℝ × ℝ := (1, -1)
  point : ℝ × ℝ := (7, 0)

/-- The standard form of an ellipse equation -/
def standard_equation (h k a b : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- The theorem stating that b = 6 for the given ellipse -/
theorem ellipse_b_value (e : Ellipse) :
  ∃ (h k a b : ℝ), a > 0 ∧ b > 0 ∧
  standard_equation h k a b (e.point.1) (e.point.2) ∧
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_b_value_l935_93577


namespace NUMINAMATH_CALUDE_boutique_packaging_combinations_l935_93548

theorem boutique_packaging_combinations :
  let wrapping_paper_designs : ℕ := 10
  let ribbon_colors : ℕ := 5
  let gift_card_varieties : ℕ := 6
  let decorative_tag_types : ℕ := 2
  wrapping_paper_designs * ribbon_colors * gift_card_varieties * decorative_tag_types = 600 :=
by sorry

end NUMINAMATH_CALUDE_boutique_packaging_combinations_l935_93548


namespace NUMINAMATH_CALUDE_herd_size_l935_93575

theorem herd_size (herd : ℕ) : 
  (1 / 3 : ℚ) * herd + (1 / 6 : ℚ) * herd + (1 / 7 : ℚ) * herd + 15 = herd →
  herd = 42 := by
sorry

end NUMINAMATH_CALUDE_herd_size_l935_93575


namespace NUMINAMATH_CALUDE_no_twelve_consecutive_primes_in_ap_l935_93587

theorem no_twelve_consecutive_primes_in_ap (a d : ℕ) (h_d : d < 2000) :
  ¬ ∀ k : Fin 12, Nat.Prime (a + k.val * d) := by
  sorry

end NUMINAMATH_CALUDE_no_twelve_consecutive_primes_in_ap_l935_93587


namespace NUMINAMATH_CALUDE_unique_triple_prime_l935_93582

theorem unique_triple_prime (p : ℕ) : 
  (p > 0 ∧ Nat.Prime p ∧ Nat.Prime (p + 4) ∧ Nat.Prime (p + 8)) ↔ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_prime_l935_93582


namespace NUMINAMATH_CALUDE_max_d_is_one_l935_93597

def a (n : ℕ+) : ℕ := 100 + n^3

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_is_one : ∀ n : ℕ+, d n = 1 := by sorry

end NUMINAMATH_CALUDE_max_d_is_one_l935_93597


namespace NUMINAMATH_CALUDE_unique_prime_in_range_l935_93564

def f (n : ℕ) : ℤ := n^3 - 7*n^2 + 15*n - 12

def is_prime (z : ℤ) : Prop := z > 1 ∧ ∀ m : ℕ, 1 < m → m < |z| → ¬(z % m = 0)

theorem unique_prime_in_range :
  ∃! (n : ℕ), 0 < n ∧ n ≤ 6 ∧ is_prime (f n) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_in_range_l935_93564


namespace NUMINAMATH_CALUDE_abs_sum_lower_bound_l935_93501

theorem abs_sum_lower_bound :
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ 3) ∧
  (∀ ε > 0, ∃ x : ℝ, |x - 1| + |x + 2| < 3 + ε) :=
by sorry

end NUMINAMATH_CALUDE_abs_sum_lower_bound_l935_93501
