import Mathlib

namespace m_less_than_n_l1796_179628

/-- Represents a quadratic function f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines N based on the coefficients of a quadratic function -/
def N (f : QuadraticFunction) : ℝ :=
  |f.a + f.b + f.c| + |2*f.a - f.b|

/-- Defines M based on the coefficients of a quadratic function -/
def M (f : QuadraticFunction) : ℝ :=
  |f.a - f.b + f.c| + |2*f.a + f.b|

/-- Theorem stating that M < N for a quadratic function satisfying certain conditions -/
theorem m_less_than_n (f : QuadraticFunction)
  (h1 : f.a + f.b + f.c < 0)
  (h2 : f.a - f.b + f.c > 0)
  (h3 : f.a > 0)
  (h4 : -f.b / (2 * f.a) > 1) :
  M f < N f := by
  sorry

end m_less_than_n_l1796_179628


namespace parabola_y_axis_intersection_l1796_179642

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -(x + 2)^2 + 6

/-- The y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- Theorem: The intersection of the parabola and y-axis is at (0, 2) -/
theorem parabola_y_axis_intersection :
  ∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ y_axis p.1 ∧ p = (0, 2) := by
sorry

end parabola_y_axis_intersection_l1796_179642


namespace intersection_M_N_l1796_179676

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | |x| > 2}

theorem intersection_M_N : M ∩ N = {x | 2 < x ∧ x < 3} := by sorry

end intersection_M_N_l1796_179676


namespace committee_probability_l1796_179675

/-- The probability of selecting a 5-person committee with at least one boy and one girl
    from a group of 25 members (10 boys and 15 girls) is equal to 475/506. -/
theorem committee_probability (total_members : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) :
  total_members = 25 →
  boys = 10 →
  girls = 15 →
  committee_size = 5 →
  (Nat.choose total_members committee_size - Nat.choose boys committee_size - Nat.choose girls committee_size) /
  Nat.choose total_members committee_size = 475 / 506 := by
  sorry

#eval Nat.choose 25 5
#eval Nat.choose 10 5
#eval Nat.choose 15 5

end committee_probability_l1796_179675


namespace log_equation_solution_l1796_179619

theorem log_equation_solution (x : ℝ) :
  Real.log x / Real.log 2 = 5/2 → x = 4 * Real.sqrt 2 := by
  sorry

end log_equation_solution_l1796_179619


namespace original_number_proof_l1796_179631

theorem original_number_proof (x : ℝ) : 
  (x * 1.125 - x * 0.75 = 30) → x = 80 := by
  sorry

end original_number_proof_l1796_179631


namespace sum_of_valid_a_l1796_179646

theorem sum_of_valid_a : ∃ (S : Finset Int), 
  (∀ a ∈ S, (∃ x : Int, x ≤ 2 ∧ x > a + 2) ∧ 
             (∃ x y : Nat, a * x + 2 * y = -4 ∧ x + y = 4)) ∧
  (∀ a : Int, (∃ x : Int, x ≤ 2 ∧ x > a + 2) ∧ 
              (∃ x y : Nat, a * x + 2 * y = -4 ∧ x + y = 4) → a ∈ S) ∧
  (S.sum id = -16) := by
sorry

end sum_of_valid_a_l1796_179646


namespace count_x_values_l1796_179639

theorem count_x_values (x y z w : ℕ+) 
  (h1 : x > y ∧ y > z ∧ z > w)
  (h2 : x + y + z + w = 4020)
  (h3 : x^2 - y^2 + z^2 - w^2 = 4020) :
  ∃ (S : Finset ℕ+), (∀ a ∈ S, ∃ y z w : ℕ+, 
    x = a ∧ 
    a > y ∧ y > z ∧ z > w ∧
    a + y + z + w = 4020 ∧
    a^2 - y^2 + z^2 - w^2 = 4020) ∧ 
  S.card = 1003 :=
sorry

end count_x_values_l1796_179639


namespace decimal_100_to_binary_l1796_179645

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_100_to_binary :
  decimal_to_binary 100 = [1, 1, 0, 0, 1, 0, 0] := by
  sorry

end decimal_100_to_binary_l1796_179645


namespace circumscribed_circle_area_l1796_179685

/-- An isosceles triangle with two sides of length 4 and a base of length 3 -/
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  is_isosceles : side = 4 ∧ base = 3

/-- A circle passing through the vertices of a triangle -/
structure CircumscribedCircle (t : IsoscelesTriangle) where
  radius : ℝ
  passes_through_vertices : True  -- This is a simplification, as we can't easily express this condition in Lean

/-- The theorem stating that the area of the circumscribed circle is 16π -/
theorem circumscribed_circle_area (t : IsoscelesTriangle) 
  (c : CircumscribedCircle t) : Real.pi * c.radius ^ 2 = 16 * Real.pi := by
  sorry

#check circumscribed_circle_area

end circumscribed_circle_area_l1796_179685


namespace tank_truck_ratio_l1796_179660

theorem tank_truck_ratio (trucks : ℕ) (total : ℕ) : 
  trucks = 20 → total = 140 → (total - trucks) / trucks = 6 := by
  sorry

end tank_truck_ratio_l1796_179660


namespace no_four_naturals_exist_l1796_179697

theorem no_four_naturals_exist : ¬∃ (a b c d : ℕ), 
  a + b + c + d = 2^100 ∧ a * b * c * d = 17^100 := by
  sorry

end no_four_naturals_exist_l1796_179697


namespace vertical_angles_equal_l1796_179604

/-- Two lines in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- An angle formed by two lines -/
structure Angle :=
  (line1 : Line)
  (line2 : Line)

/-- Two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  l1.slope ≠ l2.slope

/-- Vertical angles are formed when two lines intersect -/
def vertical_angles (a1 a2 : Angle) : Prop :=
  ∃ (l1 l2 : Line), intersect l1 l2 ∧ 
    ((a1.line1 = l1 ∧ a1.line2 = l2 ∧ a2.line1 = l2 ∧ a2.line2 = l1) ∨
     (a1.line1 = l2 ∧ a1.line2 = l1 ∧ a2.line1 = l1 ∧ a2.line2 = l2))

/-- Angles are equal -/
def angle_equal (a1 a2 : Angle) : Prop :=
  sorry  -- Definition of angle equality

/-- Theorem: Vertical angles are equal -/
theorem vertical_angles_equal (a1 a2 : Angle) : 
  vertical_angles a1 a2 → angle_equal a1 a2 := by
  sorry

end vertical_angles_equal_l1796_179604


namespace M_intersect_N_l1796_179653

def M : Set ℝ := {x | -2 ≤ x - 1 ∧ x - 1 ≤ 2}

def N : Set ℝ := {x | ∃ k : ℕ+, x = 2 * k - 1}

theorem M_intersect_N : M ∩ N = {1, 3} := by sorry

end M_intersect_N_l1796_179653


namespace full_face_time_l1796_179664

/-- Represents the time taken for Wendy's skincare routine and makeup application -/
def skincare_routine : List ℕ := [2, 3, 3, 4, 1, 3, 2, 5, 2, 2]

/-- The time taken for makeup application -/
def makeup_time : ℕ := 30

/-- Theorem stating that the total time for Wendy's "full face" routine is 57 minutes -/
theorem full_face_time : (skincare_routine.sum + makeup_time) = 57 := by
  sorry

end full_face_time_l1796_179664


namespace circus_performers_time_ratio_l1796_179644

theorem circus_performers_time_ratio :
  ∀ (polly_time pulsar_time petra_time : ℕ),
    pulsar_time = 10 →
    ∃ k : ℕ, polly_time = k * pulsar_time →
    petra_time = polly_time / 6 →
    pulsar_time + polly_time + petra_time = 45 →
    polly_time / pulsar_time = 3 := by
  sorry

end circus_performers_time_ratio_l1796_179644


namespace triangle_side_length_l1796_179643

theorem triangle_side_length (y : ℝ) :
  y > 0 →  -- y is positive
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧  -- a and b are positive
    a = 10 ∧  -- shorter leg is 10
    a^2 + b^2 = y^2 ∧  -- Pythagorean theorem
    b = a * Real.sqrt 3 →  -- ratio of sides in a 30-60-90 triangle
  y = 10 * Real.sqrt 3 := by
sorry

end triangle_side_length_l1796_179643


namespace alex_speed_l1796_179652

/-- Given the running speeds of Rick, Jen, Mark, and Alex, prove Alex's speed -/
theorem alex_speed (rick_speed : ℚ) (jen_ratio : ℚ) (mark_ratio : ℚ) (alex_ratio : ℚ)
  (h1 : rick_speed = 5)
  (h2 : jen_ratio = 3 / 4)
  (h3 : mark_ratio = 4 / 3)
  (h4 : alex_ratio = 5 / 6) :
  alex_ratio * mark_ratio * jen_ratio * rick_speed = 25 / 6 := by
  sorry

end alex_speed_l1796_179652


namespace problem_statement_l1796_179621

theorem problem_statement (a b : ℝ) (h : |a + 2| + (b - 1)^2 = 0) : (a + b)^2012 = 1 := by
  sorry

end problem_statement_l1796_179621


namespace fish_population_estimate_l1796_179654

theorem fish_population_estimate 
  (tagged_june : ℕ) 
  (caught_october : ℕ) 
  (tagged_october : ℕ) 
  (death_migration_rate : ℚ) 
  (new_fish_rate : ℚ) 
  (h1 : tagged_june = 50) 
  (h2 : caught_october = 80) 
  (h3 : tagged_october = 4) 
  (h4 : death_migration_rate = 30/100) 
  (h5 : new_fish_rate = 35/100) : 
  ℕ := by
  sorry

#check fish_population_estimate

end fish_population_estimate_l1796_179654


namespace A_intersect_B_eq_closed_interval_l1796_179615

/-- Set A defined by the quadratic inequality -/
def A : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

/-- Set B defined in terms of x ∈ A -/
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 + 4*x + 3}

/-- The intersection of A and B is equal to the closed interval [-1, 1] -/
theorem A_intersect_B_eq_closed_interval :
  A ∩ B = Set.Icc (-1 : ℝ) 1 := by sorry

end A_intersect_B_eq_closed_interval_l1796_179615


namespace james_vegetable_consumption_l1796_179663

/-- Represents James' vegetable consumption --/
structure VegetableConsumption where
  asparagus : Real
  broccoli : Real
  kale : Real

/-- Calculates the total weekly consumption given daily consumption of asparagus and broccoli --/
def weekly_consumption (daily : VegetableConsumption) : Real :=
  (daily.asparagus + daily.broccoli) * 7

/-- James' initial daily consumption --/
def initial_daily : VegetableConsumption :=
  { asparagus := 0.25, broccoli := 0.25, kale := 0 }

/-- James' consumption after doubling asparagus and broccoli and adding kale --/
def final_weekly : VegetableConsumption :=
  { asparagus := initial_daily.asparagus * 2 * 7,
    broccoli := initial_daily.broccoli * 2 * 7,
    kale := 3 }

/-- Theorem stating James' final weekly vegetable consumption --/
theorem james_vegetable_consumption :
  final_weekly.asparagus + final_weekly.broccoli + final_weekly.kale = 10 := by
  sorry


end james_vegetable_consumption_l1796_179663


namespace fish_count_proof_l1796_179638

/-- The number of fish Jerk Tuna has -/
def jerk_tuna_fish : ℕ := 144

/-- The number of fish Tall Tuna has -/
def tall_tuna_fish : ℕ := 2 * jerk_tuna_fish

/-- The total number of fish Jerk Tuna and Tall Tuna have together -/
def total_fish : ℕ := jerk_tuna_fish + tall_tuna_fish

theorem fish_count_proof : total_fish = 432 := by
  sorry

end fish_count_proof_l1796_179638


namespace negation_equivalence_l1796_179626

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end negation_equivalence_l1796_179626


namespace first_term_arithmetic_progression_l1796_179601

/-- 
For a decreasing arithmetic progression with first term a, sum S, 
number of terms n, and common difference d, the following equation holds:
a = S/n + (n-1)d/2
-/
theorem first_term_arithmetic_progression 
  (a : ℝ) (S : ℝ) (n : ℕ) (d : ℝ) 
  (h1 : n > 0) 
  (h2 : d < 0) -- Ensures it's a decreasing progression
  (h3 : S = n/2 * (2*a + (n-1)*d)) -- Sum formula for arithmetic progression
  : a = S/n + (n-1)*d/2 := by
  sorry

end first_term_arithmetic_progression_l1796_179601


namespace not_prime_cubic_polynomial_l1796_179686

theorem not_prime_cubic_polynomial (n : ℕ+) : ¬ Prime (n.val^3 - 9*n.val^2 + 19*n.val - 13) := by
  sorry

end not_prime_cubic_polynomial_l1796_179686


namespace circle_radius_is_five_l1796_179699

/-- Triangle ABC with vertices A(2,0), B(8,0), and C(5,5) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨2, 0⟩, ⟨8, 0⟩, ⟨5, 5⟩}

/-- The circle circumscribing triangle ABC -/
def circumcircle : Set (ℝ × ℝ) :=
  sorry

/-- A square with side length 5 -/
def square_PQRS : Set (ℝ × ℝ) :=
  sorry

/-- The radius of the circumcircle -/
def radius (circle : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Two vertices of the square lie on the sides of the triangle -/
axiom square_vertices_on_triangle :
  ∃ (P Q : ℝ × ℝ), P ∈ square_PQRS ∧ Q ∈ square_PQRS ∧
    (P.1 - 2) / 3 = P.2 / 5 ∧ (Q.1 - 5) / 3 = -Q.2 / 5

/-- The other two vertices of the square lie on the circumcircle -/
axiom square_vertices_on_circle :
  ∃ (R S : ℝ × ℝ), R ∈ square_PQRS ∧ S ∈ square_PQRS ∧
    R ∈ circumcircle ∧ S ∈ circumcircle

/-- The side length of the square is 5 -/
axiom square_side_length :
  ∀ (X Y : ℝ × ℝ), X ∈ square_PQRS → Y ∈ square_PQRS →
    X ≠ Y → (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 25

theorem circle_radius_is_five :
  radius circumcircle = 5 := by
  sorry

end circle_radius_is_five_l1796_179699


namespace grandpa_jungmin_age_ratio_l1796_179616

/-- The ratio of grandpa's age to Jung-min's age this year, given their ages last year -/
def age_ratio (grandpa_last_year : ℕ) (jungmin_last_year : ℕ) : ℚ :=
  (grandpa_last_year + 1) / (jungmin_last_year + 1)

/-- Theorem stating that the ratio of grandpa's age to Jung-min's age this year is 8 -/
theorem grandpa_jungmin_age_ratio : age_ratio 71 8 = 8 := by
  sorry

end grandpa_jungmin_age_ratio_l1796_179616


namespace train_journey_speed_l1796_179635

/-- Given a train journey with the following conditions:
  - The total distance is 5x km
  - The first part of the journey is x km at 40 kmph
  - The second part of the journey is 2x km at speed v
  - The average speed for the entire journey is 40 kmph
  Prove that the speed v during the second part of the journey is 20 kmph -/
theorem train_journey_speed (x : ℝ) (v : ℝ) 
  (h1 : x > 0) 
  (h2 : x / 40 + 2 * x / v = 5 * x / 40) : v = 20 := by
  sorry

end train_journey_speed_l1796_179635


namespace ryan_chinese_time_l1796_179695

/-- The time Ryan spends on learning English and Chinese daily -/
def total_time : ℝ := 3

/-- The time Ryan spends on learning English daily -/
def english_time : ℝ := 2

/-- The time Ryan spends on learning Chinese daily -/
def chinese_time : ℝ := total_time - english_time

theorem ryan_chinese_time : chinese_time = 1 := by sorry

end ryan_chinese_time_l1796_179695


namespace pet_store_snake_distribution_l1796_179634

/-- Given a total number of snakes and cages, calculate the number of snakes per cage -/
def snakesPerCage (totalSnakes : ℕ) (totalCages : ℕ) : ℕ :=
  totalSnakes / totalCages

theorem pet_store_snake_distribution :
  let totalSnakes : ℕ := 4
  let totalCages : ℕ := 2
  snakesPerCage totalSnakes totalCages = 2 := by
  sorry

end pet_store_snake_distribution_l1796_179634


namespace ceiling_sqrt_225_l1796_179606

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by sorry

end ceiling_sqrt_225_l1796_179606


namespace exactville_running_difference_l1796_179670

/-- Represents the town layout with square blocks and streets --/
structure TownLayout where
  block_side : ℝ  -- Side length of a square block
  street_width : ℝ  -- Width of the streets

/-- Calculates the difference in running distance between outer and inner paths --/
def running_distance_difference (town : TownLayout) : ℝ :=
  4 * (town.block_side + 2 * town.street_width) - 4 * town.block_side

/-- Theorem stating the difference in running distance for Exactville --/
theorem exactville_running_difference :
  let town : TownLayout := { block_side := 500, street_width := 25 }
  running_distance_difference town = 200 := by
  sorry

end exactville_running_difference_l1796_179670


namespace prime_square_in_A_implies_prime_in_A_l1796_179690

-- Define the set A
def A : Set ℕ := {x | ∃ (a b : ℤ), x = a^2 + 2*b^2 ∧ a ≠ 0 ∧ b ≠ 0}

-- State the theorem
theorem prime_square_in_A_implies_prime_in_A (p : ℕ) (h_prime : Nat.Prime p) :
  p^2 ∈ A → p ∈ A := by
  sorry

end prime_square_in_A_implies_prime_in_A_l1796_179690


namespace mayoral_election_votes_l1796_179609

theorem mayoral_election_votes (x y z : ℕ) : 
  x = y + y / 2 →
  y = z - 2 * z / 5 →
  x = 22500 →
  z = 25000 :=
by sorry

end mayoral_election_votes_l1796_179609


namespace thirty_day_month_equal_tuesdays_thursdays_l1796_179693

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- A function that checks if a given starting day results in equal Tuesdays and Thursdays in a 30-day month -/
def equalTuesdaysThursdays (startDay : DayOfWeek) : Bool :=
  sorry

/-- The number of days that can be the first day of a 30-day month with equal Tuesdays and Thursdays -/
def validStartDays : Nat :=
  sorry

theorem thirty_day_month_equal_tuesdays_thursdays :
  validStartDays = 4 :=
sorry

end thirty_day_month_equal_tuesdays_thursdays_l1796_179693


namespace animal_shelter_multiple_l1796_179625

theorem animal_shelter_multiple (puppies kittens : ℕ) (h1 : puppies = 32) (h2 : kittens = 78)
  (h3 : ∃ x : ℕ, kittens = x * puppies + 14) : 
  ∃ x : ℕ, x = 2 ∧ kittens = x * puppies + 14 := by
  sorry

end animal_shelter_multiple_l1796_179625


namespace parallel_segments_l1796_179624

/-- Given four points on a Cartesian plane, if AB is parallel to XY, then k = -6 -/
theorem parallel_segments (k : ℝ) : 
  let A : ℝ × ℝ := (-6, 0)
  let B : ℝ × ℝ := (0, -6)
  let X : ℝ × ℝ := (0, 10)
  let Y : ℝ × ℝ := (16, k)
  let slope_AB := (B.2 - A.2) / (B.1 - A.1)
  let slope_XY := (Y.2 - X.2) / (Y.1 - X.1)
  slope_AB = slope_XY → k = -6 := by
sorry

end parallel_segments_l1796_179624


namespace soccer_team_selection_l1796_179617

theorem soccer_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quad_starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 6 →
  quad_starters = 2 →
  (quadruplets.choose quad_starters) * ((total_players - quadruplets).choose (starters - quad_starters)) = 2970 :=
by sorry

end soccer_team_selection_l1796_179617


namespace smallest_z_value_l1796_179649

theorem smallest_z_value (w x y z : ℤ) : 
  (∀ n : ℤ, n ≥ 0 → (w + 2*n)^3 + (x + 2*n)^3 + (y + 2*n)^3 = (z + 2*n)^3) →
  (x = w + 2) →
  (y = x + 2) →
  (z = y + 2) →
  (w > 0) →
  (2 : ℤ) ≤ z :=
sorry

end smallest_z_value_l1796_179649


namespace bug_position_after_1995_jumps_l1796_179680

/-- Represents the five points on the circle -/
inductive Point
| one
| two
| three
| four
| five

/-- Determines if a point is odd -/
def is_odd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true

/-- Moves the bug according to the rules -/
def move (p : Point) : Point :=
  match p with
  | Point.one => Point.two
  | Point.two => Point.five
  | Point.three => Point.four
  | Point.four => Point.two
  | Point.five => Point.one

/-- Performs n jumps starting from a given point -/
def jump (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => jump (move start) n

theorem bug_position_after_1995_jumps :
  jump Point.three 1995 = Point.one := by sorry

end bug_position_after_1995_jumps_l1796_179680


namespace base_8_4513_equals_2379_l1796_179656

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4513_equals_2379 :
  base_8_to_10 [3, 1, 5, 4] = 2379 := by
  sorry

end base_8_4513_equals_2379_l1796_179656


namespace gabby_fruit_count_l1796_179657

def watermelons : ℕ := 1

def peaches (w : ℕ) : ℕ := w + 12

def plums (p : ℕ) : ℕ := 3 * p

def total_fruits (w p l : ℕ) : ℕ := w + p + l

theorem gabby_fruit_count :
  total_fruits watermelons (peaches watermelons) (plums (peaches watermelons)) = 53 := by
  sorry

end gabby_fruit_count_l1796_179657


namespace sum_of_five_decimals_theorem_l1796_179691

/-- Represents a two-digit number with a decimal point between the digits -/
structure TwoDigitDecimal where
  firstDigit : ℕ
  secondDigit : ℕ
  first_digit_valid : firstDigit < 10
  second_digit_valid : secondDigit < 10

/-- The sum of five TwoDigitDecimal numbers -/
def sumFiveDecimals (a b c d e : TwoDigitDecimal) : ℚ :=
  (a.firstDigit + a.secondDigit / 10 : ℚ) +
  (b.firstDigit + b.secondDigit / 10 : ℚ) +
  (c.firstDigit + c.secondDigit / 10 : ℚ) +
  (d.firstDigit + d.secondDigit / 10 : ℚ) +
  (e.firstDigit + e.secondDigit / 10 : ℚ)

/-- All digits are different -/
def allDifferent (a b c d e : TwoDigitDecimal) : Prop :=
  a.firstDigit ≠ b.firstDigit ∧ a.firstDigit ≠ c.firstDigit ∧ a.firstDigit ≠ d.firstDigit ∧ a.firstDigit ≠ e.firstDigit ∧
  a.firstDigit ≠ a.secondDigit ∧ a.firstDigit ≠ b.secondDigit ∧ a.firstDigit ≠ c.secondDigit ∧ a.firstDigit ≠ d.secondDigit ∧ a.firstDigit ≠ e.secondDigit ∧
  b.firstDigit ≠ c.firstDigit ∧ b.firstDigit ≠ d.firstDigit ∧ b.firstDigit ≠ e.firstDigit ∧
  b.firstDigit ≠ b.secondDigit ∧ b.firstDigit ≠ c.secondDigit ∧ b.firstDigit ≠ d.secondDigit ∧ b.firstDigit ≠ e.secondDigit ∧
  c.firstDigit ≠ d.firstDigit ∧ c.firstDigit ≠ e.firstDigit ∧
  c.firstDigit ≠ c.secondDigit ∧ c.firstDigit ≠ d.secondDigit ∧ c.firstDigit ≠ e.secondDigit ∧
  d.firstDigit ≠ e.firstDigit ∧
  d.firstDigit ≠ d.secondDigit ∧ d.firstDigit ≠ e.secondDigit ∧
  e.firstDigit ≠ e.secondDigit ∧
  a.secondDigit ≠ b.secondDigit ∧ a.secondDigit ≠ c.secondDigit ∧ a.secondDigit ≠ d.secondDigit ∧ a.secondDigit ≠ e.secondDigit ∧
  b.secondDigit ≠ c.secondDigit ∧ b.secondDigit ≠ d.secondDigit ∧ b.secondDigit ≠ e.secondDigit ∧
  c.secondDigit ≠ d.secondDigit ∧ c.secondDigit ≠ e.secondDigit ∧
  d.secondDigit ≠ e.secondDigit

theorem sum_of_five_decimals_theorem (a b c d e : TwoDigitDecimal) 
  (h1 : allDifferent a b c d e)
  (h2 : ∀ x ∈ [a, b, c, d, e], x.secondDigit ≠ 0) :
  sumFiveDecimals a b c d e = 27 ∨ sumFiveDecimals a b c d e = 18 :=
sorry

end sum_of_five_decimals_theorem_l1796_179691


namespace solve_inequality_find_a_l1796_179673

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + 2 * |x - a|

-- Theorem for part I
theorem solve_inequality (x : ℝ) :
  f x 1 < 5 ↔ -2/3 < x ∧ x < 8/3 :=
sorry

-- Theorem for part II
theorem find_a :
  ∃ (a : ℝ), (∀ x, f x a ≥ 5) ∧ (∃ x, f x a = 5) → a = -4 :=
sorry

end solve_inequality_find_a_l1796_179673


namespace anna_candy_distribution_l1796_179666

/-- Given a number of candies and friends, returns the minimum number of candies
    to remove for equal distribution -/
def min_candies_to_remove (candies : ℕ) (friends : ℕ) : ℕ :=
  candies % friends

theorem anna_candy_distribution :
  let total_candies : ℕ := 30
  let num_friends : ℕ := 4
  min_candies_to_remove total_candies num_friends = 2 := by
sorry

end anna_candy_distribution_l1796_179666


namespace quadratic_expression_value_l1796_179648

theorem quadratic_expression_value (x : ℝ) (h : 3 * x^2 + 5 * x + 1 = 0) :
  (x + 2)^2 + x * (2 * x + 1) = 3 := by
  sorry

end quadratic_expression_value_l1796_179648


namespace independence_test_suitable_for_categorical_variables_l1796_179611

/-- Independence test is a statistical method used to determine the relationship between two variables -/
structure IndependenceTest where
  is_statistical_method : Bool
  determines_relationship : Bool
  between_two_variables : Bool

/-- Categorical variables are a type of variable -/
structure CategoricalVariable where
  is_variable : Bool

/-- The statement that the independence test is suitable for examining the relationship between categorical variables -/
theorem independence_test_suitable_for_categorical_variables 
  (test : IndependenceTest) 
  (cat_var : CategoricalVariable) : 
  test.is_statistical_method ∧ 
  test.determines_relationship ∧ 
  test.between_two_variables → 
  (∃ (relationship : CategoricalVariable → CategoricalVariable → Prop), 
    test.determines_relationship ∧ 
    ∀ (x y : CategoricalVariable), relationship x y) := by
  sorry

end independence_test_suitable_for_categorical_variables_l1796_179611


namespace janet_waterpark_cost_l1796_179667

/-- Calculates the total cost for a group visiting a waterpark with a discount -/
def waterpark_cost (adult_price : ℚ) (num_adults num_children : ℕ) (discount_percent : ℚ) (soda_price : ℚ) : ℚ :=
  let child_price := adult_price / 2
  let total_admission := adult_price * num_adults + child_price * num_children
  let discounted_admission := total_admission * (1 - discount_percent / 100)
  discounted_admission + soda_price

/-- The total cost for Janet's group visit to the waterpark -/
theorem janet_waterpark_cost :
  waterpark_cost 30 6 4 20 5 = 197 := by
  sorry

end janet_waterpark_cost_l1796_179667


namespace total_earnings_1200_l1796_179669

/-- Represents the prices for services of a car model -/
structure ModelPrices where
  oil_change : ℕ
  repair : ℕ
  car_wash : ℕ

/-- Represents the number of services performed for a car model -/
structure ModelServices where
  oil_changes : ℕ
  repairs : ℕ
  car_washes : ℕ

/-- Calculates the total earnings for a single car model -/
def modelEarnings (prices : ModelPrices) (services : ModelServices) : ℕ :=
  prices.oil_change * services.oil_changes +
  prices.repair * services.repairs +
  prices.car_wash * services.car_washes

/-- Theorem stating that the total earnings for the day is $1200 -/
theorem total_earnings_1200 
  (prices_A : ModelPrices)
  (prices_B : ModelPrices)
  (prices_C : ModelPrices)
  (services_A : ModelServices)
  (services_B : ModelServices)
  (services_C : ModelServices)
  (h1 : prices_A = ⟨20, 30, 5⟩)
  (h2 : prices_B = ⟨25, 40, 8⟩)
  (h3 : prices_C = ⟨30, 50, 10⟩)
  (h4 : services_A = ⟨5, 10, 15⟩)
  (h5 : services_B = ⟨3, 4, 10⟩)
  (h6 : services_C = ⟨2, 6, 5⟩) :
  modelEarnings prices_A services_A + 
  modelEarnings prices_B services_B + 
  modelEarnings prices_C services_C = 1200 := by
  sorry

end total_earnings_1200_l1796_179669


namespace skyler_song_count_l1796_179620

/-- The number of songs Skyler wrote in total -/
def total_songs (hit_songs top_100_songs unreleased_songs : ℕ) : ℕ :=
  hit_songs + top_100_songs + unreleased_songs

/-- Theorem stating the total number of songs Skyler wrote -/
theorem skyler_song_count :
  ∀ (hit_songs : ℕ),
    hit_songs = 25 →
    ∀ (top_100_songs : ℕ),
      top_100_songs = hit_songs + 10 →
      ∀ (unreleased_songs : ℕ),
        unreleased_songs = hit_songs - 5 →
        total_songs hit_songs top_100_songs unreleased_songs = 80 := by
  sorry

end skyler_song_count_l1796_179620


namespace initial_green_papayas_l1796_179668

/-- The number of green papayas that turned yellow on Friday -/
def friday_yellow : ℕ := 2

/-- The number of green papayas that turned yellow on Sunday -/
def sunday_yellow : ℕ := 2 * friday_yellow

/-- The number of green papayas left on the tree -/
def remaining_green : ℕ := 8

/-- The initial number of green papayas on the tree -/
def initial_green : ℕ := remaining_green + friday_yellow + sunday_yellow

theorem initial_green_papayas : initial_green = 14 := by
  sorry

end initial_green_papayas_l1796_179668


namespace largest_fraction_l1796_179647

theorem largest_fraction : 
  (101 : ℚ) / 199 > 5 / 11 ∧
  (101 : ℚ) / 199 > 6 / 13 ∧
  (101 : ℚ) / 199 > 19 / 39 ∧
  (101 : ℚ) / 199 > 159 / 319 :=
by sorry

end largest_fraction_l1796_179647


namespace arithmetic_calculation_l1796_179629

theorem arithmetic_calculation : (10 - 9 + 8) * 7 + 6 - 5 * (4 - 3 + 2) - 1 = 53 := by
  sorry

end arithmetic_calculation_l1796_179629


namespace find_number_l1796_179665

theorem find_number : ∃ x : ℤ, (305 + x) / 16 = 31 ∧ x = 191 := by
  sorry

end find_number_l1796_179665


namespace final_brownies_count_l1796_179600

/-- The number of brownies in a dozen -/
def dozen : ℕ := 12

/-- The initial number of brownies made by Mother -/
def initial_brownies : ℕ := 2 * dozen

/-- The number of brownies Father ate -/
def father_ate : ℕ := 8

/-- The number of brownies Mooney ate -/
def mooney_ate : ℕ := 4

/-- The number of new brownies Mother made the next day -/
def new_brownies : ℕ := 2 * dozen

/-- Theorem stating the final number of brownies on the counter -/
theorem final_brownies_count :
  initial_brownies - father_ate - mooney_ate + new_brownies = 36 := by
  sorry

end final_brownies_count_l1796_179600


namespace unique_modular_equivalence_l1796_179684

theorem unique_modular_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -2050 [ZMOD 13] ∧ n = 4 := by
  sorry

end unique_modular_equivalence_l1796_179684


namespace fifth_term_is_negative_one_l1796_179651

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum : ℕ → ℝ
  sum_def : ∀ n, sum n = (n * (a 1 + a n)) / 2
  sum_condition : sum 2 = sum 6
  a4_condition : a 4 = 1

/-- The fifth term of the special arithmetic sequence is -1 -/
theorem fifth_term_is_negative_one (seq : ArithmeticSequence) : seq.a 5 = -1 := by
  sorry

end fifth_term_is_negative_one_l1796_179651


namespace ott_final_fraction_l1796_179636

-- Define the friends
inductive Friend
| Moe
| Loki
| Nick
| Ott
| Pat

-- Define the function for initial money
def initialMoney (f : Friend) : ℚ :=
  match f with
  | Friend.Moe => 14
  | Friend.Loki => 10
  | Friend.Nick => 8
  | Friend.Pat => 12
  | Friend.Ott => 0

-- Define the function for the fraction given by each friend
def fractionGiven (f : Friend) : ℚ :=
  match f with
  | Friend.Moe => 1/7
  | Friend.Loki => 1/5
  | Friend.Nick => 1/4
  | Friend.Pat => 1/6
  | Friend.Ott => 0

-- Define the amount given by each friend
def amountGiven : ℚ := 2

-- Theorem statement
theorem ott_final_fraction :
  let totalInitial := (initialMoney Friend.Moe) + (initialMoney Friend.Loki) + 
                      (initialMoney Friend.Nick) + (initialMoney Friend.Pat)
  let totalGiven := 4 * amountGiven
  (totalGiven / (totalInitial + totalGiven)) = 2/11 := by sorry

end ott_final_fraction_l1796_179636


namespace division_and_addition_l1796_179696

theorem division_and_addition : -4 + 6 / (-2) = -7 := by
  sorry

end division_and_addition_l1796_179696


namespace sum_of_digits_c_plus_d_l1796_179678

/-- The sum of digits of c + d, where c and d are defined as follows:
    c = 10^1986 - 1
    d = 6(10^1986 - 1)/9 -/
theorem sum_of_digits_c_plus_d : ℕ :=
  let c : ℕ := 10^1986 - 1
  let d : ℕ := 6 * (10^1986 - 1) / 9
  9931

#check sum_of_digits_c_plus_d

end sum_of_digits_c_plus_d_l1796_179678


namespace no_geometric_progression_l1796_179622

/-- The sequence a_n defined as 3^n - 2^n -/
def a (n : ℕ) : ℤ := 3^n - 2^n

/-- Theorem stating that no three consecutive terms of the sequence form a geometric progression -/
theorem no_geometric_progression (m n : ℕ) (h : m < n) :
  a m * a (2*n - m) < a n ^ 2 ∧ a n ^ 2 < a m * a (2*n - m + 1) :=
by sorry

end no_geometric_progression_l1796_179622


namespace completing_square_equivalence_l1796_179630

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
  sorry

end completing_square_equivalence_l1796_179630


namespace positive_A_value_l1796_179605

-- Define the relation #
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 218) : A = 13 := by
  sorry

end positive_A_value_l1796_179605


namespace beads_per_necklace_is_20_l1796_179661

/-- The number of beads needed to make one necklace -/
def beads_per_necklace : ℕ := sorry

/-- The number of necklaces made on Monday -/
def monday_necklaces : ℕ := 10

/-- The number of necklaces made on Tuesday -/
def tuesday_necklaces : ℕ := 2

/-- The number of bracelets made on Wednesday -/
def wednesday_bracelets : ℕ := 5

/-- The number of earrings made on Wednesday -/
def wednesday_earrings : ℕ := 7

/-- The number of beads needed for one bracelet -/
def beads_per_bracelet : ℕ := 10

/-- The number of beads needed for one earring -/
def beads_per_earring : ℕ := 5

/-- The total number of beads used -/
def total_beads : ℕ := 325

theorem beads_per_necklace_is_20 : 
  beads_per_necklace = 20 :=
by
  have h1 : monday_necklaces * beads_per_necklace + 
            tuesday_necklaces * beads_per_necklace + 
            wednesday_bracelets * beads_per_bracelet + 
            wednesday_earrings * beads_per_earring = total_beads := by sorry
  sorry

end beads_per_necklace_is_20_l1796_179661


namespace james_barrels_l1796_179681

/-- The number of barrels James has -/
def number_of_barrels : ℕ := 3

/-- The capacity of a cask in gallons -/
def cask_capacity : ℕ := 20

/-- The capacity of a barrel in gallons -/
def barrel_capacity : ℕ := 2 * cask_capacity + 3

/-- The total storage capacity in gallons -/
def total_capacity : ℕ := 172

/-- Proof that James has 3 barrels -/
theorem james_barrels :
  number_of_barrels * barrel_capacity + cask_capacity = total_capacity :=
by sorry

end james_barrels_l1796_179681


namespace sequence_2018th_term_l1796_179610

theorem sequence_2018th_term (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : ∀ n, 3 * S n = 2 * a n - 3 * n) : 
  a 2018 = 2^2018 - 1 := by
sorry

end sequence_2018th_term_l1796_179610


namespace coefficient_is_40_l1796_179688

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x³y³ in the expansion of (x+y)(2x-y)⁵
def coefficient_x3y3 : ℤ :=
  2^2 * (-1)^3 * binomial 5 3 + 2^3 * binomial 5 2

-- Theorem statement
theorem coefficient_is_40 : coefficient_x3y3 = 40 := by sorry

end coefficient_is_40_l1796_179688


namespace problem_solution_l1796_179658

def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0

def q (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0

theorem problem_solution (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, m = 4 → (p x ∧ q x m) → 4 < x ∧ x < 5) ∧
  ((∀ x : ℝ, ¬(q x m) → ¬(p x)) ∧ (∃ x : ℝ, ¬(p x) ∧ q x m) → 5/3 ≤ m ∧ m ≤ 2) :=
sorry

end problem_solution_l1796_179658


namespace integer_divisibility_problem_l1796_179683

theorem integer_divisibility_problem (a b : ℤ) :
  (a^6 + 1) ∣ (b^11 - 2023*b^3 + 40*b) →
  (a^4 - 1) ∣ (b^10 - 2023*b^2 - 41) →
  a = 0 := by
sorry

end integer_divisibility_problem_l1796_179683


namespace parallel_lines_a_value_l1796_179671

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 2 * a * y - 1 = 0 ↔ (3 * a - 1) * x + y - 1 = 0) → 
  a = 1 / 5 := by
sorry

end parallel_lines_a_value_l1796_179671


namespace quadratic_form_sum_l1796_179633

theorem quadratic_form_sum (a b c : ℝ) : 
  (∀ x, 8 * x^2 - 48 * x - 320 = a * (x + b)^2 + c) → 
  a + b + c = -387 := by
  sorry

end quadratic_form_sum_l1796_179633


namespace robin_gum_packages_l1796_179698

theorem robin_gum_packages :
  ∀ (packages : ℕ),
  (7 * packages + 6 = 41) →
  packages = 5 := by
sorry

end robin_gum_packages_l1796_179698


namespace composite_shape_perimeter_l1796_179679

/-- A figure composed of two unit squares and one unit equilateral triangle. -/
structure CompositeShape where
  /-- The side length of each square -/
  square_side : ℝ
  /-- The side length of the equilateral triangle -/
  triangle_side : ℝ
  /-- Assertion that both squares and the triangle have unit side length -/
  h_unit_sides : square_side = 1 ∧ triangle_side = 1

/-- The perimeter of the composite shape -/
def perimeter (shape : CompositeShape) : ℝ :=
  3 * shape.square_side + 2 * shape.triangle_side

/-- Theorem stating that the perimeter of the composite shape is 5 units -/
theorem composite_shape_perimeter (shape : CompositeShape) :
  perimeter shape = 5 :=
sorry

end composite_shape_perimeter_l1796_179679


namespace broken_flagpole_l1796_179682

theorem broken_flagpole (h : ℝ) (d : ℝ) (x : ℝ) : 
  h = 6 → d = 2 → x * x + d * d = (h - x) * (h - x) → x = Real.sqrt 10 := by
  sorry

end broken_flagpole_l1796_179682


namespace total_cost_is_75_l1796_179637

/-- Calculates the total cost for two siblings attending a music school with a sibling discount -/
def total_cost_for_siblings (regular_tuition : ℕ) (sibling_discount : ℕ) : ℕ :=
  regular_tuition + (regular_tuition - sibling_discount)

/-- Theorem stating that the total cost for two siblings is $75 given the specific tuition and discount -/
theorem total_cost_is_75 :
  total_cost_for_siblings 45 15 = 75 := by
  sorry

#eval total_cost_for_siblings 45 15

end total_cost_is_75_l1796_179637


namespace exists_non_negative_sums_l1796_179614

/-- Represents a sign change operation on a matrix -/
inductive SignChange
| Row (i : Nat)
| Col (j : Nat)

/-- Applies a sequence of sign changes to a matrix -/
def applySignChanges (A : Matrix (Fin m) (Fin n) ℝ) (changes : List SignChange) : Matrix (Fin m) (Fin n) ℝ :=
  sorry

/-- Checks if all row sums and column sums are non-negative -/
def allSumsNonNegative (A : Matrix (Fin m) (Fin n) ℝ) : Prop :=
  sorry

/-- Main theorem: For any matrix, there exists a sequence of sign changes that makes all sums non-negative -/
theorem exists_non_negative_sums (m n : Nat) (A : Matrix (Fin m) (Fin n) ℝ) :
  ∃ (changes : List SignChange), allSumsNonNegative (applySignChanges A changes) :=
by
  sorry

end exists_non_negative_sums_l1796_179614


namespace mrs_randall_teaching_years_l1796_179659

theorem mrs_randall_teaching_years (third_grade_years second_grade_years : ℕ) 
  (h1 : third_grade_years = 18) 
  (h2 : second_grade_years = 8) : 
  third_grade_years + second_grade_years = 26 := by
  sorry

end mrs_randall_teaching_years_l1796_179659


namespace fishmonger_sales_l1796_179623

/-- The total amount of fish sold by a fishmonger in two weeks, given the first week's sales and a multiplier for the second week. -/
def total_fish_sales (first_week : ℕ) (multiplier : ℕ) : ℕ :=
  first_week + multiplier * first_week

/-- Theorem stating that if a fishmonger sold 50 kg of salmon in the first week and three times that amount in the second week, the total amount of fish sold in two weeks is 200 kg. -/
theorem fishmonger_sales : total_fish_sales 50 3 = 200 := by
  sorry

#eval total_fish_sales 50 3

end fishmonger_sales_l1796_179623


namespace rectangular_plot_area_l1796_179677

theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 18 →
  length = 3 * breadth →
  area = length * breadth →
  area = 972 := by
sorry

end rectangular_plot_area_l1796_179677


namespace min_production_volume_for_break_even_l1796_179632

/-- The total cost function -/
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

/-- The revenue function -/
def revenue (x : ℝ) : ℝ := 25 * x

/-- The break-even condition -/
def break_even (x : ℝ) : Prop := revenue x ≥ total_cost x

theorem min_production_volume_for_break_even :
  ∃ (x : ℝ), x = 150 ∧ 0 < x ∧ x < 240 ∧ break_even x ∧
  ∀ (y : ℝ), 0 < y ∧ y < x → ¬(break_even y) := by
  sorry

end min_production_volume_for_break_even_l1796_179632


namespace kims_class_hours_l1796_179627

/-- Calculates the total class hours after dropping a class -/
def total_class_hours_after_drop (initial_classes : ℕ) (hours_per_class : ℕ) (dropped_classes : ℕ) : ℕ :=
  (initial_classes - dropped_classes) * hours_per_class

/-- Proves that Kim's total class hours after dropping a class is 6 -/
theorem kims_class_hours : total_class_hours_after_drop 4 2 1 = 6 := by
  sorry

end kims_class_hours_l1796_179627


namespace polyhedron_face_edges_divisible_by_three_l1796_179618

-- Define a polyhedron
structure Polyhedron where
  faces : Set Face
  edges : Set Edge
  vertices : Set Vertex

-- Define a face
structure Face where
  edges : Set Edge

-- Define an edge
structure Edge where
  vertices : Fin 2 → Vertex

-- Define a vertex
structure Vertex where

-- Define a color
inductive Color
  | White
  | Black

-- Define a coloring function
def coloring (p : Polyhedron) : Face → Color := sorry

-- Define the number of edges for a face
def numEdges (f : Face) : Nat := sorry

-- Define adjacency for faces
def adjacent (f1 f2 : Face) : Prop := sorry

-- Theorem statement
theorem polyhedron_face_edges_divisible_by_three 
  (p : Polyhedron) 
  (h1 : ∀ f1 f2 : Face, f1 ∈ p.faces → f2 ∈ p.faces → adjacent f1 f2 → coloring p f1 ≠ coloring p f2)
  (h2 : ∃ f : Face, f ∈ p.faces ∧ ∀ f' : Face, f' ∈ p.faces → f' ≠ f → (numEdges f') % 3 = 0) :
  ∀ f : Face, f ∈ p.faces → (numEdges f) % 3 = 0 := by
  sorry

end polyhedron_face_edges_divisible_by_three_l1796_179618


namespace sin_750_degrees_l1796_179640

theorem sin_750_degrees (h : ∀ x, Real.sin (x + 2 * Real.pi) = Real.sin x) : 
  Real.sin (750 * Real.pi / 180) = 1/2 := by
  sorry

end sin_750_degrees_l1796_179640


namespace inequality_proof_l1796_179655

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (x / (x + y*z)) + (y / (y + z*x)) + (z / (z + x*y)) ≤ 2 / (1 - 3*x*y*z) := by
  sorry

end inequality_proof_l1796_179655


namespace valentines_distribution_l1796_179650

theorem valentines_distribution (initial_valentines : Real) (additional_valentines : Real) (num_students : Nat) :
  initial_valentines = 58.0 →
  additional_valentines = 16.0 →
  num_students = 74 →
  (initial_valentines + additional_valentines) / num_students = 1 := by
  sorry

end valentines_distribution_l1796_179650


namespace triangle_ABC_properties_l1796_179689

/-- Triangle ABC with vertices A(-3,0), B(2,1), and C(-2,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def ABC : Triangle := ⟨(-3, 0), (2, 1), (-2, 3)⟩

/-- The equation of line BC -/
def line_BC : LineEquation := ⟨1, 2, -4⟩

/-- The equation of the perpendicular bisector of BC -/
def perp_bisector_BC : LineEquation := ⟨2, -1, 2⟩

theorem triangle_ABC_properties :
  let t := ABC
  (line_BC.a * t.B.1 + line_BC.b * t.B.2 + line_BC.c = 0 ∧
   line_BC.a * t.C.1 + line_BC.b * t.C.2 + line_BC.c = 0) ∧
  (perp_bisector_BC.a * ((t.B.1 + t.C.1) / 2) + 
   perp_bisector_BC.b * ((t.B.2 + t.C.2) / 2) + 
   perp_bisector_BC.c = 0 ∧
   perp_bisector_BC.a * line_BC.b = -perp_bisector_BC.b * line_BC.a) := by
  sorry

end triangle_ABC_properties_l1796_179689


namespace no_base_for_131_perfect_square_l1796_179672

theorem no_base_for_131_perfect_square :
  ¬ ∃ (b : ℕ), b ≥ 2 ∧ ∃ (n : ℕ), b^2 + 3*b + 1 = n^2 := by
  sorry

end no_base_for_131_perfect_square_l1796_179672


namespace transformed_variance_l1796_179687

-- Define a type for our dataset
def Dataset := Fin 10 → ℝ

-- Define the variance of a dataset
noncomputable def variance (X : Dataset) : ℝ := sorry

-- State the theorem
theorem transformed_variance (X : Dataset) 
  (h : variance X = 3) : 
  variance (fun i => 2 * (X i) + 3) = 12 := by
  sorry

end transformed_variance_l1796_179687


namespace steven_jill_difference_l1796_179613

/-- The number of peaches each person has -/
structure PeachCounts where
  jake : ℕ
  steven : ℕ
  jill : ℕ

/-- The conditions given in the problem -/
def problem_conditions (p : PeachCounts) : Prop :=
  p.jake + 6 = p.steven ∧
  p.steven > p.jill ∧
  p.jill = 5 ∧
  p.jake = 17

/-- The theorem to be proved -/
theorem steven_jill_difference (p : PeachCounts) 
  (h : problem_conditions p) : p.steven - p.jill = 18 := by
  sorry

end steven_jill_difference_l1796_179613


namespace female_students_count_l1796_179603

theorem female_students_count (total_students sample_size male_sample : ℕ) 
  (h1 : total_students = 2000)
  (h2 : sample_size = 200)
  (h3 : male_sample = 103) : 
  (total_students : ℚ) * ((sample_size - male_sample) : ℚ) / (sample_size : ℚ) = 970 := by
  sorry

end female_students_count_l1796_179603


namespace min_filtration_layers_l1796_179694

theorem min_filtration_layers (a : ℝ) (ha : a > 0) : 
  (∃ n : ℕ, n ≥ 5 ∧ a * (4/5)^n ≤ (1/3) * a ∧ ∀ m : ℕ, m < 5 → a * (4/5)^m > (1/3) * a) :=
sorry

end min_filtration_layers_l1796_179694


namespace no_indefinite_cutting_l1796_179612

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ
  length_pos : length > 0
  width_pos : width > 0

/-- Defines the cutting procedure for rectangles -/
def cut_rectangle (T : Rectangle) : Option (Rectangle × Rectangle) :=
  sorry

/-- Checks if two rectangles are similar -/
def are_similar (T1 T2 : Rectangle) : Prop :=
  T1.length / T1.width = T2.length / T2.width

/-- Checks if two rectangles are congruent -/
def are_congruent (T1 T2 : Rectangle) : Prop :=
  T1.length = T2.length ∧ T1.width = T2.width

/-- Defines the property of indefinite cutting -/
def can_cut_indefinitely (T : Rectangle) : Prop :=
  ∀ n : ℕ, ∃ (T_seq : ℕ → Rectangle), 
    T_seq 0 = T ∧
    (∀ i < n, 
      ∃ T1 T2 : Rectangle, 
        cut_rectangle (T_seq i) = some (T1, T2) ∧
        are_similar T1 T2 ∧
        ¬are_congruent T1 T2 ∧
        T_seq (i + 1) = T1)

theorem no_indefinite_cutting : ¬∃ T : Rectangle, can_cut_indefinitely T := by
  sorry

end no_indefinite_cutting_l1796_179612


namespace gear_speed_proportion_l1796_179608

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four sequentially meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  meshed : A.teeth * A.speed = B.teeth * B.speed ∧
           B.teeth * B.speed = C.teeth * C.speed ∧
           C.teeth * C.speed = D.teeth * D.speed

/-- The theorem stating the proportion of angular speeds for the gear system -/
theorem gear_speed_proportion (sys : GearSystem) :
  ∃ (k : ℝ), k > 0 ∧ 
    sys.A.speed = k * (sys.B.teeth * sys.C.teeth * sys.D.teeth) ∧
    sys.B.speed = k * (sys.A.teeth * sys.C.teeth * sys.D.teeth) ∧
    sys.C.speed = k * (sys.A.teeth * sys.B.teeth * sys.D.teeth) ∧
    sys.D.speed = k * (sys.A.teeth * sys.B.teeth * sys.C.teeth) :=
  sorry

end gear_speed_proportion_l1796_179608


namespace count_valid_paths_l1796_179692

/-- The number of paths from (0,1) to (n-1,n) that stay strictly above y=x -/
def validPaths (n : ℕ) : ℚ :=
  (1 : ℚ) / n * (Nat.choose (2*n - 2) (n - 1))

/-- Theorem stating the number of valid paths -/
theorem count_valid_paths (n : ℕ) (h : n > 0) :
  validPaths n = (1 : ℚ) / n * (Nat.choose (2*n - 2) (n - 1)) :=
by sorry

end count_valid_paths_l1796_179692


namespace harry_travel_time_l1796_179662

theorem harry_travel_time (initial_bus_time remaining_bus_time : ℕ) 
  (h1 : initial_bus_time = 15)
  (h2 : remaining_bus_time = 25) : 
  let total_bus_time := initial_bus_time + remaining_bus_time
  let walking_time := total_bus_time / 2
  initial_bus_time + remaining_bus_time + walking_time = 60 := by
sorry

end harry_travel_time_l1796_179662


namespace problem_statement_l1796_179607

/-- The problem statement as a theorem -/
theorem problem_statement 
  (ω : ℝ) 
  (hω : ω > 0)
  (a : ℝ → ℝ × ℝ)
  (b : ℝ → ℝ × ℝ)
  (ha : ∀ x, a x = (Real.sin (ω * x) + Real.cos (ω * x), Real.sqrt 3 * Real.cos (ω * x)))
  (hb : ∀ x, b x = (Real.cos (ω * x) - Real.sin (ω * x), 2 * Real.sin (ω * x)))
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = (a x).1 * (b x).1 + (a x).2 * (b x).2)
  (hsymmetry : ∀ x, f (x + π / (2 * ω)) = f x)
  (A B C : ℝ)
  (hC : f C = 1)
  (c : ℝ)
  (hc : c = 2)
  (hsin : Real.sin C + Real.sin (B - A) = 3 * Real.sin (2 * A))
  : ω = 1 ∧ 
    (Real.sqrt 3 / 3 * c ^ 2 = 2 * Real.sqrt 3 / 3 ∨ 
     Real.sqrt 3 / 3 * c ^ 2 = 3 * Real.sqrt 3 / 7) :=
by sorry

end problem_statement_l1796_179607


namespace correct_propositions_l1796_179674

theorem correct_propositions (a b : ℝ) : 
  ((a > |b| → a^2 > b^2) ∧ (a > b → a^3 > b^3)) := by
  sorry

end correct_propositions_l1796_179674


namespace dot_product_MN_MO_l1796_179602

-- Define the circle O
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}

-- Define a line l (we don't need to specify its equation, just that it exists)
def line_l : Set (ℝ × ℝ) := sorry

-- Define points M and N as the intersection of line l and circle O
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

-- Define point O as the center of the circle
def O : ℝ × ℝ := (0, 0)

-- State that M and N are on the circle
axiom M_on_circle : M ∈ circle_O
axiom N_on_circle : N ∈ circle_O

-- State that M and N are on the line l
axiom M_on_line : M ∈ line_l
axiom N_on_line : N ∈ line_l

-- Define the distance between M and N
axiom MN_distance : Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4

-- Define vectors MN and MO
def vec_MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)
def vec_MO : ℝ × ℝ := (O.1 - M.1, O.2 - M.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem to prove
theorem dot_product_MN_MO : dot_product vec_MN vec_MO = 8 := by sorry

end dot_product_MN_MO_l1796_179602


namespace hexagon_angle_measure_l1796_179641

theorem hexagon_angle_measure (N U M B E R S : ℝ) : 
  -- Hexagon condition
  N + U + M + B + E + R + S = 720 →
  -- Congruent angles
  N = M →
  B = R →
  -- Supplementary angles
  U + S = 180 →
  -- Conclusion
  E = 180 := by
  sorry

end hexagon_angle_measure_l1796_179641
