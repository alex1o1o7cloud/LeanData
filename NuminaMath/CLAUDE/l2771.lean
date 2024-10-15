import Mathlib

namespace NUMINAMATH_CALUDE_problem_1_l2771_277158

theorem problem_1 : 
  4 + 1/4 - 19/5 + 4/5 + 11/4 = 4 := by sorry

end NUMINAMATH_CALUDE_problem_1_l2771_277158


namespace NUMINAMATH_CALUDE_tan_order_l2771_277115

theorem tan_order : 
  (1 < Real.pi / 2) → 
  (Real.pi / 2 < 2) → 
  (2 < 3) → 
  (3 < Real.pi) → 
  (∀ x y, Real.pi / 2 < x → x < y → y < Real.pi → Real.tan x < Real.tan y) →
  Real.tan 1 > Real.tan 3 ∧ Real.tan 3 > Real.tan 2 :=
by sorry

end NUMINAMATH_CALUDE_tan_order_l2771_277115


namespace NUMINAMATH_CALUDE_equation_solution_l2771_277104

theorem equation_solution :
  ∀ x : ℝ, x ≠ 0 ∧ 8*x + 3 ≠ 0 ∧ 7*x - 3 ≠ 0 →
    (2 + 5/(4*x) - 15/(4*x*(8*x+3)) = 2*(7*x+1)/(7*x-3)) ↔ x = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2771_277104


namespace NUMINAMATH_CALUDE_xiaogong_speed_l2771_277106

/-- The speed of Xiaogong in meters per minute -/
def v_x : ℝ := 28

/-- The speed of Dachen in meters per minute -/
def v_d : ℝ := v_x + 20

/-- The total distance between points A and B in meters -/
def total_distance : ℝ := 1200

/-- The time Dachen walks before meeting Xiaogong, in minutes -/
def t_d : ℝ := 18

/-- The time Xiaogong walks before meeting Dachen, in minutes -/
def t_x : ℝ := 12

theorem xiaogong_speed :
  v_x * t_x + v_d * t_d = total_distance ∧
  v_d = v_x + 20 →
  v_x = 28 := by sorry

end NUMINAMATH_CALUDE_xiaogong_speed_l2771_277106


namespace NUMINAMATH_CALUDE_tank_filling_time_l2771_277143

/-- Represents the time (in minutes) it takes for pipe A to fill the tank alone -/
def A : ℝ := 24

/-- Represents the time (in minutes) it takes for pipe B to fill the tank alone -/
def B : ℝ := 32

/-- Represents the time (in minutes) both pipes are open before pipe B is closed -/
def t_both : ℝ := 8

/-- Represents the total time (in minutes) to fill the tank using both pipes as described -/
def t_total : ℝ := 18

theorem tank_filling_time : 
  (t_both * (1 / A + 1 / B)) + ((t_total - t_both) * (1 / A)) = 1 ∧ 
  A = 24 := by
  sorry

#check tank_filling_time

end NUMINAMATH_CALUDE_tank_filling_time_l2771_277143


namespace NUMINAMATH_CALUDE_basketball_not_table_tennis_count_l2771_277183

/-- Represents the class of students and their sports preferences -/
structure ClassSports where
  total : ℕ
  basketball : ℕ
  tableTennis : ℕ
  neither : ℕ

/-- The number of students who like basketball but not table tennis -/
def basketballNotTableTennis (c : ClassSports) : ℕ :=
  c.basketball - (c.total - c.tableTennis - c.neither)

/-- Theorem stating the number of students who like basketball but not table tennis -/
theorem basketball_not_table_tennis_count (c : ClassSports) 
  (h1 : c.total = 30)
  (h2 : c.basketball = 15)
  (h3 : c.tableTennis = 10)
  (h4 : c.neither = 8) :
  basketballNotTableTennis c = 12 := by
  sorry

end NUMINAMATH_CALUDE_basketball_not_table_tennis_count_l2771_277183


namespace NUMINAMATH_CALUDE_lcm_852_1491_l2771_277121

theorem lcm_852_1491 : Nat.lcm 852 1491 = 5961 := by
  sorry

end NUMINAMATH_CALUDE_lcm_852_1491_l2771_277121


namespace NUMINAMATH_CALUDE_min_area_is_three_l2771_277129

/-- Triangle ABC with A at origin, B at (30, 18), and C with integer coordinates -/
structure Triangle :=
  (c_x : ℤ)
  (c_y : ℤ)

/-- Area of triangle ABC given coordinates of C -/
def area (t : Triangle) : ℚ :=
  (1 / 2 : ℚ) * |30 * t.c_y - 18 * t.c_x|

/-- The minimum area of triangle ABC is 3 -/
theorem min_area_is_three :
  ∃ (t : Triangle), area t = 3 ∧ ∀ (t' : Triangle), area t' ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_area_is_three_l2771_277129


namespace NUMINAMATH_CALUDE_stick_length_average_l2771_277187

theorem stick_length_average (total_sticks : ℕ) (all_avg : ℝ) (two_avg : ℝ) :
  total_sticks = 11 →
  all_avg = 145.7 →
  two_avg = 142.1 →
  let remaining_sticks := total_sticks - 2
  let total_length := all_avg * total_sticks
  let two_length := two_avg * 2
  let remaining_length := total_length - two_length
  remaining_length / remaining_sticks = 146.5 := by
sorry

end NUMINAMATH_CALUDE_stick_length_average_l2771_277187


namespace NUMINAMATH_CALUDE_cos_150_plus_cos_neg_150_l2771_277151

theorem cos_150_plus_cos_neg_150 : Real.cos (150 * π / 180) + Real.cos (-150 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_plus_cos_neg_150_l2771_277151


namespace NUMINAMATH_CALUDE_cassandra_pies_l2771_277193

/-- Calculates the number of apple pies Cassandra made -/
def number_of_pies (apples_bought : ℕ) (slices_per_pie : ℕ) (apples_per_slice : ℕ) : ℕ :=
  (apples_bought / apples_per_slice) / slices_per_pie

theorem cassandra_pies :
  let apples_bought := 4 * 12 -- four dozen
  let slices_per_pie := 6
  let apples_per_slice := 2
  number_of_pies apples_bought slices_per_pie apples_per_slice = 4 := by
  sorry

#eval number_of_pies (4 * 12) 6 2

end NUMINAMATH_CALUDE_cassandra_pies_l2771_277193


namespace NUMINAMATH_CALUDE_skateboard_travel_distance_l2771_277191

/-- Represents the distance traveled by a skateboard in a given number of seconds -/
def skateboardDistance (initialDistance : ℕ) (firstAcceleration : ℕ) (secondAcceleration : ℕ) (totalSeconds : ℕ) : ℕ :=
  let firstPeriodDistance := (5 : ℕ) * (2 * initialDistance + 4 * firstAcceleration) / 2
  let secondPeriodInitialDistance := initialDistance + 5 * firstAcceleration
  let secondPeriodDistance := (5 : ℕ) * (2 * secondPeriodInitialDistance + 4 * secondAcceleration) / 2
  firstPeriodDistance + secondPeriodDistance

theorem skateboard_travel_distance :
  skateboardDistance 8 6 9 10 = 380 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_travel_distance_l2771_277191


namespace NUMINAMATH_CALUDE_mountain_elevation_difference_l2771_277127

/-- The elevation difference between two mountains -/
def elevation_difference (h b : ℕ) : ℕ := h - b

/-- Proves that the elevation difference between two mountains is 2500 feet -/
theorem mountain_elevation_difference :
  ∃ (h b : ℕ),
    h = 10000 ∧
    3 * h = 4 * b ∧
    elevation_difference h b = 2500 := by
  sorry

end NUMINAMATH_CALUDE_mountain_elevation_difference_l2771_277127


namespace NUMINAMATH_CALUDE_loan_duration_l2771_277172

theorem loan_duration (principal_B principal_C interest_rate total_interest : ℚ) 
  (duration_C : ℕ) : 
  principal_B = 5000 →
  principal_C = 3000 →
  duration_C = 4 →
  interest_rate = 15 / 100 →
  total_interest = 3300 →
  principal_B * interest_rate * (duration_B : ℚ) + principal_C * interest_rate * (duration_C : ℚ) = total_interest →
  duration_B = 2 := by
  sorry

#check loan_duration

end NUMINAMATH_CALUDE_loan_duration_l2771_277172


namespace NUMINAMATH_CALUDE_greatest_five_digit_with_product_90_l2771_277157

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).prod

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem greatest_five_digit_with_product_90 :
  ∃ M : ℕ, is_five_digit M ∧
           digit_product M = 90 ∧
           (∀ n : ℕ, is_five_digit n ∧ digit_product n = 90 → n ≤ M) ∧
           digit_sum M = 18 :=
sorry

end NUMINAMATH_CALUDE_greatest_five_digit_with_product_90_l2771_277157


namespace NUMINAMATH_CALUDE_two_days_satisfy_l2771_277130

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- The number of days in the month -/
def monthLength : Nat := 30

/-- Function to check if a given day results in equal Tuesdays and Thursdays -/
def equalTuesdaysThursdays (startDay : DayOfWeek) : Bool :=
  sorry -- Implementation details omitted

/-- Count the number of days that satisfy the condition -/
def countSatisfyingDays : Nat :=
  sorry -- Implementation details omitted

/-- Theorem stating that exactly two days satisfy the condition -/
theorem two_days_satisfy :
  countSatisfyingDays = 2 :=
sorry

end NUMINAMATH_CALUDE_two_days_satisfy_l2771_277130


namespace NUMINAMATH_CALUDE_quadratic_completion_l2771_277138

theorem quadratic_completion (y : ℝ) : ∃ (k : ℤ) (a : ℝ), y^2 + 10*y + 47 = (y + a)^2 + k ∧ k = 22 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l2771_277138


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2771_277128

/-- Represents a cylindrical water tank. -/
structure WaterTank where
  capacity : ℝ
  initialFill : ℝ

/-- Condition that the tank is 1/6 full initially. -/
def isInitiallySixthFull (tank : WaterTank) : Prop :=
  tank.initialFill / tank.capacity = 1 / 6

/-- Condition that the tank becomes 1/3 full after adding 5 liters. -/
def isThirdFullAfterAddingFive (tank : WaterTank) : Prop :=
  (tank.initialFill + 5) / tank.capacity = 1 / 3

/-- Theorem stating that if a water tank satisfies the given conditions, its capacity is 30 liters. -/
theorem water_tank_capacity
    (tank : WaterTank)
    (h1 : isInitiallySixthFull tank)
    (h2 : isThirdFullAfterAddingFive tank) :
    tank.capacity = 30 := by
  sorry


end NUMINAMATH_CALUDE_water_tank_capacity_l2771_277128


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l2771_277155

theorem square_sum_given_difference_and_product (a b : ℝ) 
  (h1 : a - b = 10) 
  (h2 : a * b = 55) : 
  a^2 + b^2 = 210 :=
by sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l2771_277155


namespace NUMINAMATH_CALUDE_sibling_age_difference_l2771_277152

/-- Given the ages of three siblings, prove the age difference between two of them. -/
theorem sibling_age_difference (juliet maggie ralph : ℕ) : 
  juliet > maggie ∧ 
  juliet = ralph - 2 ∧ 
  juliet = 10 ∧ 
  maggie + ralph = 19 → 
  juliet - maggie = 3 := by
sorry

end NUMINAMATH_CALUDE_sibling_age_difference_l2771_277152


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2771_277177

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2771_277177


namespace NUMINAMATH_CALUDE_least_x_squared_divisible_by_240_l2771_277119

theorem least_x_squared_divisible_by_240 :
  ∀ x : ℕ, x > 0 → x^2 % 240 = 0 → x ≥ 60 :=
by
  sorry

end NUMINAMATH_CALUDE_least_x_squared_divisible_by_240_l2771_277119


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2771_277178

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℕ), x > 0 → (x - 1)^2 > 0) ↔ (∃ (x : ℕ), x > 0 ∧ (x - 1)^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2771_277178


namespace NUMINAMATH_CALUDE_probability_two_hearts_l2771_277125

theorem probability_two_hearts (total_cards : Nat) (heart_cards : Nat) (drawn_cards : Nat) :
  total_cards = 52 →
  heart_cards = 13 →
  drawn_cards = 2 →
  (Nat.choose heart_cards drawn_cards : ℚ) / (Nat.choose total_cards drawn_cards : ℚ) = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_hearts_l2771_277125


namespace NUMINAMATH_CALUDE_fraction_value_l2771_277175

theorem fraction_value : (3000 - 2883)^2 / 121 = 106.36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2771_277175


namespace NUMINAMATH_CALUDE_tuesday_steps_l2771_277134

/-- The number of steps Toby walked on each day of the week --/
structure WeekSteps where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Theorem stating that given the conditions, Toby walked 8300 steps on Tuesday --/
theorem tuesday_steps (w : WeekSteps) : 
  w.sunday = 9400 ∧ 
  w.monday = 9100 ∧ 
  (w.wednesday = 9200 ∨ w.thursday = 9200) ∧
  (w.wednesday = 8900 ∨ w.thursday = 8900) ∧
  w.friday + w.saturday = 18100 ∧
  w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday = 63000 
  → w.tuesday = 8300 := by
  sorry

#check tuesday_steps

end NUMINAMATH_CALUDE_tuesday_steps_l2771_277134


namespace NUMINAMATH_CALUDE_egyptian_triangle_bisecting_line_exists_l2771_277117

/-- Represents a right triangle with sides 3, 4, and 5 -/
structure EgyptianTriangle where
  a : Real
  b : Real
  c : Real
  ha : a = 3
  hb : b = 4
  hc : c = 5
  right_angle : a^2 + b^2 = c^2

/-- Represents a line that intersects the triangle -/
structure BisectingLine where
  x : Real -- intersection point on shorter leg
  y : Real -- intersection point on hypotenuse
  hx : x = 3 - Real.sqrt 6 / 2
  hy : y = 3 + Real.sqrt 6 / 2

/-- Theorem stating the existence of a bisecting line for an Egyptian triangle -/
theorem egyptian_triangle_bisecting_line_exists (t : EgyptianTriangle) :
  ∃ (l : BisectingLine),
    (l.x + l.y = t.a + t.b) ∧                          -- Bisects perimeter
    (l.x * l.y * (t.b / t.c) / 2 = t.a * t.b / 4) :=   -- Bisects area
by sorry

end NUMINAMATH_CALUDE_egyptian_triangle_bisecting_line_exists_l2771_277117


namespace NUMINAMATH_CALUDE_jace_initial_earnings_l2771_277110

theorem jace_initial_earnings (debt : ℕ) (remaining : ℕ) (h1 : debt = 358) (h2 : remaining = 642) :
  debt + remaining = 1000 := by
  sorry

end NUMINAMATH_CALUDE_jace_initial_earnings_l2771_277110


namespace NUMINAMATH_CALUDE_football_team_size_l2771_277196

/-- Represents the composition of a football team -/
structure FootballTeam where
  total : ℕ
  throwers : ℕ
  rightHanded : ℕ
  leftHanded : ℕ

/-- The properties of our specific football team -/
def ourTeam : FootballTeam where
  total := 70
  throwers := 40
  rightHanded := 60
  leftHanded := 70 - 40 - (60 - 40)

theorem football_team_size : 
  ∀ (team : FootballTeam), 
  team.throwers = 40 ∧ 
  team.rightHanded = 60 ∧ 
  team.leftHanded = (team.total - team.throwers) / 3 ∧
  team.rightHanded = team.throwers + 2 * (team.total - team.throwers) / 3 →
  team.total = 70 := by
  sorry

#check football_team_size

end NUMINAMATH_CALUDE_football_team_size_l2771_277196


namespace NUMINAMATH_CALUDE_largest_guaranteed_divisor_l2771_277112

def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def Q (s : Finset ℕ) : ℕ := s.prod id

theorem largest_guaranteed_divisor :
  ∀ s : Finset ℕ, s ⊆ die_faces → s.card = 7 → 960 ∣ Q s ∧
  ∀ n : ℕ, n > 960 → ∃ t : Finset ℕ, t ⊆ die_faces ∧ t.card = 7 ∧ ¬(n ∣ Q t) :=
by sorry

end NUMINAMATH_CALUDE_largest_guaranteed_divisor_l2771_277112


namespace NUMINAMATH_CALUDE_total_points_earned_l2771_277189

def enemies_defeated : ℕ := 6
def points_per_enemy : ℕ := 9
def level_completion_points : ℕ := 8

theorem total_points_earned :
  enemies_defeated * points_per_enemy + level_completion_points = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_points_earned_l2771_277189


namespace NUMINAMATH_CALUDE_geometric_mean_a2_a8_l2771_277182

theorem geometric_mean_a2_a8 (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                     -- first term
  q = 2 →                       -- common ratio
  (a 2 * a 8).sqrt = 48 ∨ (a 2 * a 8).sqrt = -48 :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_a2_a8_l2771_277182


namespace NUMINAMATH_CALUDE_min_distance_scaled_circle_to_line_l2771_277124

/-- The minimum distance from a point on the scaled circle to a line -/
theorem min_distance_scaled_circle_to_line :
  let C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let l : Set (ℝ × ℝ) := {p | p.1 + Real.sqrt 3 * p.2 - 6 = 0}
  let C' : Set (ℝ × ℝ) := {p | (p.1^2 / 9) + p.2^2 = 1}
  ∃ (d : ℝ), d = 3 - Real.sqrt 3 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ C' → 
      ∀ (q : ℝ × ℝ), q ∈ l → 
        d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_scaled_circle_to_line_l2771_277124


namespace NUMINAMATH_CALUDE_special_square_side_length_l2771_277142

/-- Square with special points -/
structure SpecialSquare where
  /-- Side length of the square -/
  side : ℝ
  /-- Point M on side CD -/
  m : ℝ × ℝ
  /-- Point E where AM intersects the circumscribed circle -/
  e : ℝ × ℝ

/-- The theorem statement -/
theorem special_square_side_length (s : SpecialSquare) :
  /- Point M is on side CD -/
  s.m.1 = s.side ∧ 0 ≤ s.m.2 ∧ s.m.2 ≤ s.side ∧
  /- CM:MD = 1:3 -/
  s.m.2 = s.side / 4 ∧
  /- E is on the circumscribed circle -/
  (s.e.1 - s.side / 2)^2 + (s.e.2 - s.side / 2)^2 = 2 * (s.side / 2)^2 ∧
  /- Area of triangle ACE is 14 -/
  1/2 * s.e.1 * s.e.2 = 14 →
  /- The side length of the square is 10 -/
  s.side = 10 := by
  sorry

end NUMINAMATH_CALUDE_special_square_side_length_l2771_277142


namespace NUMINAMATH_CALUDE_prime_sum_divides_cube_diff_l2771_277190

theorem prime_sum_divides_cube_diff (p q : ℕ) : 
  Prime p → Prime q → (p + q) ∣ (p^3 - q^3) → p = q := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_divides_cube_diff_l2771_277190


namespace NUMINAMATH_CALUDE_nested_subtraction_1999_always_true_l2771_277194

/-- The nested subtraction function with n levels of nesting -/
def nestedSubtraction (x : ℝ) : ℕ → ℝ
  | 0 => x - 1
  | n + 1 => x - nestedSubtraction x n

/-- Theorem stating that for 1999 levels of nesting, the equation is always true for any real x -/
theorem nested_subtraction_1999_always_true (x : ℝ) :
  nestedSubtraction x 1999 = 1 := by
  sorry

#check nested_subtraction_1999_always_true

end NUMINAMATH_CALUDE_nested_subtraction_1999_always_true_l2771_277194


namespace NUMINAMATH_CALUDE_circle_angle_problem_l2771_277118

theorem circle_angle_problem (x y : ℝ) : 
  y = 2 * x → 7 * x + 6 * x + 3 * x + (2 * x + y) = 360 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_angle_problem_l2771_277118


namespace NUMINAMATH_CALUDE_triangle_formation_proof_l2771_277137

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Given sticks of lengths 4 and 10, proves which of the given lengths can form a triangle -/
theorem triangle_formation_proof :
  let a : ℝ := 4
  let b : ℝ := 10
  (¬ can_form_triangle a b 3) ∧
  (¬ can_form_triangle a b 5) ∧
  (can_form_triangle a b 8) ∧
  (¬ can_form_triangle a b 15) := by
  sorry

#check triangle_formation_proof

end NUMINAMATH_CALUDE_triangle_formation_proof_l2771_277137


namespace NUMINAMATH_CALUDE_tan_squared_sum_lower_bound_l2771_277166

theorem tan_squared_sum_lower_bound 
  (α β γ : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < γ) (h4 : γ < π / 2)
  (h5 : Real.sin α ^ 3 + Real.sin β ^ 3 + Real.sin γ ^ 3 = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 / (9 ^ (1/3) - 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_squared_sum_lower_bound_l2771_277166


namespace NUMINAMATH_CALUDE_balls_after_2010_steps_l2771_277113

/-- Converts a natural number to its base-6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  if n < 6 then [n]
  else (n % 6) :: toBase6 (n / 6)

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.sum

theorem balls_after_2010_steps :
  sumDigits (toBase6 2010) = 10 := by
  sorry

end NUMINAMATH_CALUDE_balls_after_2010_steps_l2771_277113


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_min_value_sqrt_sum_equals_l2771_277126

theorem min_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 4) :
  ∀ a b c : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → a + b + c = 4 →
    Real.sqrt (2 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (4 * z + 1) ≤
    Real.sqrt (2 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (4 * c + 1) :=
by
  sorry

theorem min_value_sqrt_sum_equals (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 4) :
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 4 ∧
    Real.sqrt (2 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (4 * z + 1) =
    Real.sqrt (61 / 27) + Real.sqrt (183 / 36) + Real.sqrt (976 / 108) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_min_value_sqrt_sum_equals_l2771_277126


namespace NUMINAMATH_CALUDE_intersection_points_with_ellipse_l2771_277136

/-- The line equation mx - ny = 4 and circle x^2 + y^2 = 4 have no intersection points -/
def no_intersection (m n : ℝ) : Prop :=
  ∀ x y : ℝ, (m * x - n * y = 4) → (x^2 + y^2 ≠ 4)

/-- The ellipse equation x^2/9 + y^2/4 = 1 -/
def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

/-- A point (x, y) is on the line passing through (m, n) -/
def on_line_through_P (m n x y : ℝ) : Prop :=
  ∃ t : ℝ, x = m * t ∧ y = n * t

/-- The theorem statement -/
theorem intersection_points_with_ellipse (m n : ℝ) :
  no_intersection m n →
  (∃! (x1 y1 x2 y2 : ℝ), 
    x1 ≠ x2 ∧ 
    on_ellipse x1 y1 ∧ 
    on_ellipse x2 y2 ∧ 
    on_line_through_P m n x1 y1 ∧ 
    on_line_through_P m n x2 y2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_with_ellipse_l2771_277136


namespace NUMINAMATH_CALUDE_multiply_decimals_l2771_277111

theorem multiply_decimals : (2.4 : ℝ) * 0.2 = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimals_l2771_277111


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l2771_277141

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℚ) :
  team_size = 11 →
  captain_age = 27 →
  team_avg_age = 24 →
  ∃ (wicket_keeper_age : ℕ),
    (team_avg_age * team_size - captain_age - wicket_keeper_age) / (team_size - 2) = team_avg_age - 1 →
    wicket_keeper_age = captain_age + 3 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l2771_277141


namespace NUMINAMATH_CALUDE_pen_notebook_cost_l2771_277131

theorem pen_notebook_cost :
  ∀ (p n : ℕ), 
    p > n ∧ 
    p > 0 ∧ 
    n > 0 ∧ 
    17 * p + 5 * n = 200 →
    p + n = 16 := by
  sorry

end NUMINAMATH_CALUDE_pen_notebook_cost_l2771_277131


namespace NUMINAMATH_CALUDE_phoenix_flight_l2771_277149

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem phoenix_flight :
  let a₁ := 3
  let r := 3
  ∀ n : ℕ, n < 8 → geometric_sequence a₁ r n ≤ 6560 ∧
  geometric_sequence a₁ r 8 > 6560 :=
by sorry

end NUMINAMATH_CALUDE_phoenix_flight_l2771_277149


namespace NUMINAMATH_CALUDE_cos_210_degrees_l2771_277198

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l2771_277198


namespace NUMINAMATH_CALUDE_arccos_one_eq_zero_l2771_277170

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_eq_zero_l2771_277170


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2771_277186

/-- Given a line segment with one endpoint at (10, 4) and midpoint at (7, -5),
    the sum of coordinates of the other endpoint is -10. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (10 + x) / 2 = 7 → 
  (4 + y) / 2 = -5 → 
  x + y = -10 := by
sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2771_277186


namespace NUMINAMATH_CALUDE_sqrt_seven_expressions_l2771_277101

theorem sqrt_seven_expressions (a b : ℝ) 
  (ha : a = Real.sqrt 7 + 2) 
  (hb : b = Real.sqrt 7 - 2) : 
  a^2 * b + b^2 * a = 6 * Real.sqrt 7 ∧ 
  a^2 + a * b + b^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_expressions_l2771_277101


namespace NUMINAMATH_CALUDE_lamp_sales_problem_l2771_277197

/-- Shopping mall lamp sales problem -/
theorem lamp_sales_problem
  (initial_price : ℝ)
  (cost_price : ℝ)
  (initial_sales : ℝ)
  (price_increase : ℝ)
  (sales_decrease_rate : ℝ)
  (h1 : initial_price = 40)
  (h2 : cost_price = 30)
  (h3 : initial_sales = 600)
  (h4 : 0 < price_increase ∧ price_increase < 20)
  (h5 : sales_decrease_rate = 10) :
  let new_sales := initial_sales - sales_decrease_rate * price_increase
  let new_price := initial_price + price_increase
  let profit := (new_price - cost_price) * new_sales
  ∃ (optimal_increase : ℝ) (max_profit_price : ℝ),
    (new_sales = 600 - 10 * price_increase) ∧
    (profit = 10000 → new_price = 50 ∧ new_sales = 500) ∧
    (max_profit_price = 59 ∧ ∀ x, 0 < x ∧ x < 20 → profit ≤ (59 - cost_price) * (initial_sales - sales_decrease_rate * (59 - initial_price))) :=
by sorry

end NUMINAMATH_CALUDE_lamp_sales_problem_l2771_277197


namespace NUMINAMATH_CALUDE_fourth_grade_students_l2771_277184

/-- The total number of students at the end of the year in fourth grade -/
def total_students (initial : ℝ) (added : ℝ) (new : ℝ) : ℝ :=
  initial + added + new

/-- Theorem: The total number of students at the end of the year is 56.0 -/
theorem fourth_grade_students :
  total_students 10.0 4.0 42.0 = 56.0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l2771_277184


namespace NUMINAMATH_CALUDE_prob_HTTH_is_one_sixteenth_l2771_277159

/-- The probability of obtaining the sequence HTTH in four consecutive fair coin tosses -/
def prob_HTTH : ℚ := 1 / 16

/-- A fair coin toss is modeled as a probability space with two outcomes -/
structure FairCoin where
  sample_space : Type
  prob : sample_space → ℚ
  head : sample_space
  tail : sample_space
  fair_head : prob head = 1 / 2
  fair_tail : prob tail = 1 / 2
  total_prob : prob head + prob tail = 1

/-- Four consecutive fair coin tosses -/
def four_tosses (c : FairCoin) : Type := c.sample_space × c.sample_space × c.sample_space × c.sample_space

/-- The probability of a specific sequence of four tosses -/
def sequence_prob (c : FairCoin) (s : four_tosses c) : ℚ :=
  c.prob s.1 * c.prob s.2.1 * c.prob s.2.2.1 * c.prob s.2.2.2

/-- Theorem: The probability of obtaining HTTH in four consecutive fair coin tosses is 1/16 -/
theorem prob_HTTH_is_one_sixteenth (c : FairCoin) :
  sequence_prob c (c.head, c.tail, c.tail, c.head) = prob_HTTH := by
  sorry

end NUMINAMATH_CALUDE_prob_HTTH_is_one_sixteenth_l2771_277159


namespace NUMINAMATH_CALUDE_no_linear_term_implies_a_equals_negative_four_l2771_277123

theorem no_linear_term_implies_a_equals_negative_four (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, (x + 4) * (x + a) = x^2 + b*x + c) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_a_equals_negative_four_l2771_277123


namespace NUMINAMATH_CALUDE_calculation_difference_l2771_277105

def harry_calculation : ℤ := 12 - (3 + 4 * 2)

def terry_calculation : ℤ :=
  let step1 := 12 - 3
  let step2 := step1 + 4
  step2 * 2

theorem calculation_difference :
  harry_calculation - terry_calculation = -25 := by sorry

end NUMINAMATH_CALUDE_calculation_difference_l2771_277105


namespace NUMINAMATH_CALUDE_shortcut_rectangle_ratio_l2771_277180

/-- A rectangle where the diagonal shortcut saves 1/3 of the longer side -/
structure ShortcutRectangle where
  x : ℝ  -- shorter side
  y : ℝ  -- longer side
  x_pos : 0 < x
  y_pos : 0 < y
  x_lt_y : x < y
  shortcut_saves : x + y - Real.sqrt (x^2 + y^2) = (1/3) * y

theorem shortcut_rectangle_ratio (r : ShortcutRectangle) : r.x / r.y = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_shortcut_rectangle_ratio_l2771_277180


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_1992_l2771_277107

theorem last_three_digits_of_7_to_1992 : ∃ n : ℕ, 7^1992 ≡ 201 + 1000 * n [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_1992_l2771_277107


namespace NUMINAMATH_CALUDE_factor_implies_k_value_l2771_277144

/-- Given a quadratic trinomial 2x^2 + 3x - k with a factor (2x - 5), k equals 20 -/
theorem factor_implies_k_value (k : ℝ) : 
  (∃ (q : ℝ → ℝ), ∀ x, 2*x^2 + 3*x - k = (2*x - 5) * q x) → 
  k = 20 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_k_value_l2771_277144


namespace NUMINAMATH_CALUDE_existence_of_divisible_power_sum_l2771_277185

theorem existence_of_divisible_power_sum (a b : ℕ) (h : b > 1) :
  ∃ n : ℕ, n < b^2 ∧ b ∣ (a^n + n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_divisible_power_sum_l2771_277185


namespace NUMINAMATH_CALUDE_rational_inequality_solution_set_l2771_277156

theorem rational_inequality_solution_set :
  {x : ℝ | (x + 1) / (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_set_l2771_277156


namespace NUMINAMATH_CALUDE_intersection_M_N_l2771_277140

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {-1, 0, 2, 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2771_277140


namespace NUMINAMATH_CALUDE_factorization_example_l2771_277169

-- Define factorization
def is_factorization (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x ∧ ∃ (p q : ℝ → ℝ), g x = p x * q x

-- Define the left-hand side of the equation
def lhs (m : ℝ) : ℝ := m^2 - 4

-- Define the right-hand side of the equation
def rhs (m : ℝ) : ℝ := (m + 2) * (m - 2)

-- Theorem statement
theorem factorization_example : is_factorization lhs rhs := by sorry

end NUMINAMATH_CALUDE_factorization_example_l2771_277169


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2771_277179

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (2*k + 3)*x₁ + k^2 = 0 ∧ 
    x₂^2 + (2*k + 3)*x₂ + k^2 = 0 ∧
    1/x₁ + 1/x₂ = -1) → 
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2771_277179


namespace NUMINAMATH_CALUDE_tyler_purchase_theorem_l2771_277108

def remaining_money (initial_amount scissors_cost eraser_cost scissors_quantity eraser_quantity : ℕ) : ℕ :=
  initial_amount - (scissors_cost * scissors_quantity + eraser_cost * eraser_quantity)

theorem tyler_purchase_theorem (initial_amount scissors_cost eraser_cost scissors_quantity eraser_quantity : ℕ) :
  initial_amount = 100 ∧ 
  scissors_cost = 5 ∧ 
  eraser_cost = 4 ∧ 
  scissors_quantity = 8 ∧ 
  eraser_quantity = 10 → 
  remaining_money initial_amount scissors_cost eraser_cost scissors_quantity eraser_quantity = 20 := by
  sorry

end NUMINAMATH_CALUDE_tyler_purchase_theorem_l2771_277108


namespace NUMINAMATH_CALUDE_smallest_an_correct_l2771_277132

def smallest_an (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 3 then 11
  else 4 * n + 1

theorem smallest_an_correct (n : ℕ) (h : n ≥ 1) :
  ∀ (a : ℕ → ℕ),
  (∀ i j, 0 ≤ i → i < j → j ≤ n → a i < a j) →
  (∀ i j, 0 ≤ i → i < j → j ≤ n → ¬ Nat.Prime (a j - a i)) →
  a n ≥ smallest_an n :=
sorry

end NUMINAMATH_CALUDE_smallest_an_correct_l2771_277132


namespace NUMINAMATH_CALUDE_min_value_expression_l2771_277162

theorem min_value_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  4 * x^2 + 9 * y^2 + 16 / x^2 + 6 * y / x ≥ 2 * Real.sqrt 564 / 3 ∧
  ∃ x₀ y₀ : ℝ, x₀ ≠ 0 ∧ y₀ ≠ 0 ∧
    4 * x₀^2 + 9 * y₀^2 + 16 / x₀^2 + 6 * y₀ / x₀ = 2 * Real.sqrt 564 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2771_277162


namespace NUMINAMATH_CALUDE_farmer_ploughing_problem_l2771_277154

theorem farmer_ploughing_problem (planned_rate : ℕ) (actual_rate : ℕ) (extra_days : ℕ) (area_left : ℕ) (total_area : ℕ) :
  planned_rate = 120 →
  actual_rate = 85 →
  extra_days = 2 →
  area_left = 40 →
  total_area = 720 →
  ∃ (planned_days : ℕ), 
    planned_days * planned_rate = total_area ∧
    (planned_days + extra_days) * actual_rate + area_left = total_area ∧
    planned_days = 6 :=
by sorry

end NUMINAMATH_CALUDE_farmer_ploughing_problem_l2771_277154


namespace NUMINAMATH_CALUDE_linear_function_proof_l2771_277167

/-- A linear function passing through (0,5) and parallel to y=x -/
def f (x : ℝ) : ℝ := x + 5

theorem linear_function_proof :
  (f 0 = 5) ∧ 
  (∀ x y : ℝ, f (x + y) - f x = y) ∧
  (∀ x : ℝ, f x = x + 5) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_proof_l2771_277167


namespace NUMINAMATH_CALUDE_abs_x_minus_one_l2771_277188

theorem abs_x_minus_one (x : ℚ) (h : |1 - x| = 1 + |x|) : |x - 1| = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_l2771_277188


namespace NUMINAMATH_CALUDE_min_beacons_required_l2771_277174

/-- Represents a room in the maze --/
structure Room where
  x : Nat
  y : Nat

/-- Represents the maze structure --/
def Maze := List Room

/-- Calculates the distance between two rooms in the maze --/
def distance (maze : Maze) (r1 r2 : Room) : Nat :=
  sorry

/-- Checks if a set of beacons can uniquely identify all rooms --/
def can_identify_all_rooms (maze : Maze) (beacons : List Room) : Prop :=
  sorry

/-- The main theorem stating that at least 3 beacons are required --/
theorem min_beacons_required (maze : Maze) :
  ∀ (beacons : List Room),
    can_identify_all_rooms maze beacons →
    beacons.length ≥ 3 :=
  sorry

end NUMINAMATH_CALUDE_min_beacons_required_l2771_277174


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2771_277114

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  x + y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 4/y = 1 ∧ x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2771_277114


namespace NUMINAMATH_CALUDE_gcd_9011_2147_l2771_277148

theorem gcd_9011_2147 : Nat.gcd 9011 2147 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9011_2147_l2771_277148


namespace NUMINAMATH_CALUDE_subset_implies_m_range_l2771_277199

theorem subset_implies_m_range (m : ℝ) : 
  let A : Set ℝ := {x | 4 ≤ x ∧ x ≤ 8}
  let B : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 2}
  B ⊆ A → m ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_range_l2771_277199


namespace NUMINAMATH_CALUDE_yoongi_number_division_l2771_277102

theorem yoongi_number_division (x : ℤ) : 
  x - 17 = 55 → x / 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_number_division_l2771_277102


namespace NUMINAMATH_CALUDE_quadratic_vertex_l2771_277163

/-- A quadratic function passing through specific points has its vertex at x = 5 -/
theorem quadratic_vertex (a b c : ℝ) : 
  (4 = a * 2^2 + b * 2 + c) →
  (4 = a * 8^2 + b * 8 + c) →
  (13 = a * 10^2 + b * 10 + c) →
  (-b / (2 * a) = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l2771_277163


namespace NUMINAMATH_CALUDE_volume_is_12pi_l2771_277147

/-- Represents a solid object with three views and dimensions -/
structure Solid where
  frontView : Real × Real
  sideView : Real × Real
  topView : Real × Real

/-- Calculates the volume of a solid based on its views and dimensions -/
def volumeOfSolid (s : Solid) : Real := sorry

/-- Theorem stating that the volume of the given solid is 12π cm³ -/
theorem volume_is_12pi (s : Solid) : volumeOfSolid s = 12 * Real.pi := by sorry

end NUMINAMATH_CALUDE_volume_is_12pi_l2771_277147


namespace NUMINAMATH_CALUDE_travel_agency_comparison_l2771_277109

/-- Represents the fee calculation for a travel agency. -/
structure TravelAgency where
  parentDiscount : ℝ  -- Discount for parents (1 means no discount)
  studentDiscount : ℝ  -- Discount for students
  basePrice : ℝ        -- Base price per person

/-- Calculate the total fee for a travel agency given the number of students. -/
def calculateFee (agency : TravelAgency) (numStudents : ℝ) : ℝ :=
  agency.basePrice * (2 * agency.parentDiscount + numStudents * agency.studentDiscount)

/-- Travel Agency A with full price for parents and 70% for students. -/
def agencyA : TravelAgency :=
  { parentDiscount := 1
  , studentDiscount := 0.7
  , basePrice := 500 }

/-- Travel Agency B with 80% price for both parents and students. -/
def agencyB : TravelAgency :=
  { parentDiscount := 0.8
  , studentDiscount := 0.8
  , basePrice := 500 }

theorem travel_agency_comparison :
  ∀ x : ℝ,
    (calculateFee agencyA x = 350 * x + 1000) ∧
    (calculateFee agencyB x = 400 * x + 800) ∧
    (0 < x ∧ x < 4 → calculateFee agencyB x < calculateFee agencyA x) ∧
    (x = 4 → calculateFee agencyA x = calculateFee agencyB x) ∧
    (x > 4 → calculateFee agencyA x < calculateFee agencyB x) :=
by sorry

end NUMINAMATH_CALUDE_travel_agency_comparison_l2771_277109


namespace NUMINAMATH_CALUDE_thousand_to_hundred_power_l2771_277168

theorem thousand_to_hundred_power (h : 1000 = 10^3) : 1000^100 = 10^300 := by
  sorry

end NUMINAMATH_CALUDE_thousand_to_hundred_power_l2771_277168


namespace NUMINAMATH_CALUDE_train_speed_problem_l2771_277192

/-- Proves that given a train journey where the distance is covered in 276 minutes
    at speed S1, and the same distance can be covered in 69 minutes at 16 kmph,
    then S1 = 4 kmph -/
theorem train_speed_problem (S1 : ℝ) : 
  (276 : ℝ) * S1 = 69 * 16 → S1 = 4 := by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2771_277192


namespace NUMINAMATH_CALUDE_count_divisible_sum_l2771_277153

theorem count_divisible_sum : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ (n * (n + 1) / 2) ∣ (8 * n)) ∧ 
  (∀ n : ℕ, n > 0 ∧ (n * (n + 1) / 2) ∣ (8 * n) → n ∈ S) ∧ 
  Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_sum_l2771_277153


namespace NUMINAMATH_CALUDE_least_common_multiple_15_36_l2771_277150

theorem least_common_multiple_15_36 : Nat.lcm 15 36 = 180 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_15_36_l2771_277150


namespace NUMINAMATH_CALUDE_x_eq_y_sufficient_not_necessary_l2771_277173

theorem x_eq_y_sufficient_not_necessary :
  (∀ x y : ℝ, x = y → |x| = |y|) ∧
  (∃ x y : ℝ, |x| = |y| ∧ x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_y_sufficient_not_necessary_l2771_277173


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2771_277122

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2771_277122


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l2771_277181

/-- Given a line in the form (3, -4) · ((x, y) - (2, 8)) = 0,
    prove that it's equivalent to y = (3/4)x + 6.5 -/
theorem line_equation_equivalence (x y : ℝ) :
  (3 * (x - 2) + (-4) * (y - 8) = 0) ↔ (y = (3/4) * x + 6.5) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l2771_277181


namespace NUMINAMATH_CALUDE_fraction_floor_value_l2771_277139

theorem fraction_floor_value : ⌊(1500^2 : ℝ) / ((500^2 : ℝ) - (496^2 : ℝ))⌋ = 564 := by
  sorry

end NUMINAMATH_CALUDE_fraction_floor_value_l2771_277139


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2771_277116

/-- Given a circle centered at (0,b) with radius a, and a hyperbola C: y²/a² - x²/b² = 1 (a > 0, b > 0),
    if the circle and the asymptotes of the hyperbola C are disjoint, 
    then the eccentricity e of C satisfies 1 < e < (√5 + 1)/2. -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y - b)^2 = a^2}
  let hyperbola := {(x, y) : ℝ × ℝ | y^2 / a^2 - x^2 / b^2 = 1}
  let asymptotes := {(x, y) : ℝ × ℝ | b * y = a * x ∨ b * y = -a * x}
  let e := Real.sqrt (1 + b^2 / a^2)  -- eccentricity of the hyperbola
  (circle ∩ asymptotes = ∅) → 1 < e ∧ e < (Real.sqrt 5 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2771_277116


namespace NUMINAMATH_CALUDE_solve_for_y_l2771_277165

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 7 = y + 3) (h2 : x = -5) : y = 44 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2771_277165


namespace NUMINAMATH_CALUDE_systematic_sampling_l2771_277146

/-- Systematic sampling problem -/
theorem systematic_sampling 
  (total_items : ℕ) 
  (selected_items : ℕ) 
  (first_selected : ℕ) 
  (group_number : ℕ) :
  total_items = 3000 →
  selected_items = 150 →
  first_selected = 11 →
  group_number = 61 →
  (group_number - 1) * (total_items / selected_items) + first_selected = 1211 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l2771_277146


namespace NUMINAMATH_CALUDE_average_and_difference_l2771_277103

theorem average_and_difference (y : ℝ) : 
  (47 + y) / 2 = 53 → |y - 47| = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l2771_277103


namespace NUMINAMATH_CALUDE_oddSumProbability_l2771_277133

/-- Represents an unfair die where even numbers are 5 times as likely as odd numbers -/
structure UnfairDie where
  /-- Probability of rolling an odd number -/
  oddProb : ℝ
  /-- Probability of rolling an even number -/
  evenProb : ℝ
  /-- Even probability is 5 times odd probability -/
  evenOddRatio : evenProb = 5 * oddProb
  /-- Total probability is 1 -/
  totalProb : oddProb + evenProb = 1

/-- The probability of rolling an odd sum with two rolls of the unfair die -/
def oddSumProb (d : UnfairDie) : ℝ :=
  2 * d.oddProb * d.evenProb

theorem oddSumProbability (d : UnfairDie) : oddSumProb d = 5 / 18 := by
  sorry


end NUMINAMATH_CALUDE_oddSumProbability_l2771_277133


namespace NUMINAMATH_CALUDE_trig_ratios_on_line_l2771_277161

/-- Given an angle α whose terminal side lies on the line y = 2x, 
    prove its trigonometric ratios. -/
theorem trig_ratios_on_line (α : Real) : 
  (∃ k : Real, k ≠ 0 ∧ Real.cos α = k ∧ Real.sin α = 2 * k) → 
  (Real.sin α)^2 = 4/5 ∧ (Real.cos α)^2 = 1/5 ∧ Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_ratios_on_line_l2771_277161


namespace NUMINAMATH_CALUDE_final_antifreeze_ratio_l2771_277176

/-- Calculates the fraction of antifreeze in a tank after multiple replacements --/
def antifreezeRatio (tankCapacity : ℚ) (initialRatio : ℚ) (replacementAmount : ℚ) (replacements : ℕ) : ℚ :=
  let initialAntifreeze := tankCapacity * initialRatio
  let remainingRatio := (tankCapacity - replacementAmount) / tankCapacity
  initialAntifreeze * remainingRatio ^ replacements / tankCapacity

/-- Theorem stating the final antifreeze ratio after 4 replacements --/
theorem final_antifreeze_ratio :
  antifreezeRatio 20 (1/4) 4 4 = 1024/5000 := by
  sorry

#eval antifreezeRatio 20 (1/4) 4 4

end NUMINAMATH_CALUDE_final_antifreeze_ratio_l2771_277176


namespace NUMINAMATH_CALUDE_average_age_of_ten_students_l2771_277164

theorem average_age_of_ten_students
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_group1 : Nat)
  (average_age_group1 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : num_group1 = 4)
  (h4 : average_age_group1 = 14)
  (h5 : age_last_student = 9)
  : (total_students * average_age_all - num_group1 * average_age_group1 - age_last_student) / (total_students - num_group1 - 1) = 16 := by
  sorry

#check average_age_of_ten_students

end NUMINAMATH_CALUDE_average_age_of_ten_students_l2771_277164


namespace NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l2771_277160

def total_products : ℕ := 10

-- Define the event A
def event_A (defective : ℕ) : Prop := defective ≥ 2 ∧ defective ≤ total_products

-- Define the complementary event of A
def complement_A (defective : ℕ) : Prop := defective ≤ 1

-- Theorem statement
theorem complement_of_at_least_two_defective :
  ∀ (defective : ℕ), defective ≤ total_products →
  (¬ event_A defective ↔ complement_A defective) :=
sorry

end NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l2771_277160


namespace NUMINAMATH_CALUDE_pages_left_to_read_total_annotated_pages_l2771_277100

-- Define the book and reading parameters
def total_pages : ℕ := 567
def pages_read_week1 : ℕ := 279
def pages_read_week2 : ℕ := 124
def pages_annotated_week1 : ℕ := 35
def pages_annotated_week2 : ℕ := 15

-- Theorem for pages left to read
theorem pages_left_to_read : 
  total_pages - (pages_read_week1 + pages_read_week2) = 164 := by sorry

-- Theorem for total annotated pages
theorem total_annotated_pages :
  pages_annotated_week1 + pages_annotated_week2 = 50 := by sorry

end NUMINAMATH_CALUDE_pages_left_to_read_total_annotated_pages_l2771_277100


namespace NUMINAMATH_CALUDE_log_13_3x_bounds_l2771_277135

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_13_3x_bounds (x : ℝ) (h : log 7 (x + 6) = 2) : 
  1 < log 13 (3 * x) ∧ log 13 (3 * x) < 2 := by
  sorry

end NUMINAMATH_CALUDE_log_13_3x_bounds_l2771_277135


namespace NUMINAMATH_CALUDE_exhibition_survey_l2771_277171

/-- The percentage of visitors who liked the first part of the exhibition -/
def first_part_percentage : ℝ := 25

/-- The percentage of visitors who liked the second part of the exhibition -/
def second_part_percentage : ℝ := 40

theorem exhibition_survey (total_visitors : ℝ) (h_total_positive : total_visitors > 0) :
  let visitors_first_part := (first_part_percentage / 100) * total_visitors
  let visitors_second_part := (second_part_percentage / 100) * total_visitors
  (96 / 100 * visitors_first_part = 60 / 100 * visitors_second_part) ∧
  (59 / 100 * total_visitors = total_visitors - (visitors_first_part + visitors_second_part - 96 / 100 * visitors_first_part)) →
  first_part_percentage = 25 := by
sorry


end NUMINAMATH_CALUDE_exhibition_survey_l2771_277171


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2771_277145

theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 36 → inradius = 2.5 → area = 45 → 
  area = inradius * (perimeter / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2771_277145


namespace NUMINAMATH_CALUDE_citadel_school_earnings_l2771_277120

/-- Represents the total earnings for a school in the summer project. -/
def schoolEarnings (totalPayment : ℚ) (totalStudentDays : ℕ) (schoolStudentDays : ℕ) : ℚ :=
  (totalPayment / totalStudentDays) * schoolStudentDays

/-- Theorem: The earnings for Citadel school in the summer project. -/
theorem citadel_school_earnings :
  let apexDays : ℕ := 9 * 5
  let beaconDays : ℕ := 3 * 4
  let citadelDays : ℕ := 6 * 7
  let totalDays : ℕ := apexDays + beaconDays + citadelDays
  let totalPayment : ℚ := 864
  schoolEarnings totalPayment totalDays citadelDays = 864 / 99 * 42 :=
by sorry

#eval schoolEarnings 864 99 42

end NUMINAMATH_CALUDE_citadel_school_earnings_l2771_277120


namespace NUMINAMATH_CALUDE_sum_of_possible_distances_l2771_277195

theorem sum_of_possible_distances (a b c d : ℝ) 
  (hab : |a - b| = 2)
  (hbc : |b - c| = 3)
  (hcd : |c - d| = 4) :
  ∃ S : Finset ℝ, (∀ x ∈ S, ∃ a' b' c' d' : ℝ, 
    |a' - b'| = 2 ∧ |b' - c'| = 3 ∧ |c' - d'| = 4 ∧ |a' - d'| = x) ∧
  (∀ y : ℝ, (∃ a' b' c' d' : ℝ, 
    |a' - b'| = 2 ∧ |b' - c'| = 3 ∧ |c' - d'| = 4 ∧ |a' - d'| = y) → y ∈ S) ∧
  S.sum id = 18 :=
sorry

end NUMINAMATH_CALUDE_sum_of_possible_distances_l2771_277195
