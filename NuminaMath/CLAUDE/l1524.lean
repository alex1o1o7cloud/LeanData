import Mathlib

namespace square_sum_formula_l1524_152476

theorem square_sum_formula (x y c a : ℝ) 
  (h1 : x * y = 2 * c) 
  (h2 : 1 / x^2 + 1 / y^2 = 3 * a) : 
  (x + y)^2 = 12 * a * c^2 + 4 * c := by
  sorry

end square_sum_formula_l1524_152476


namespace sin_two_x_value_l1524_152442

theorem sin_two_x_value (x : ℝ) (h : Real.sin (π / 4 - x) = 1 / 6) : 
  Real.sin (2 * x) = 17 / 18 := by
  sorry

end sin_two_x_value_l1524_152442


namespace businessman_travel_l1524_152498

theorem businessman_travel (morning_bike : ℕ) (evening_bike : ℕ) (car_trips : ℕ) :
  morning_bike = 10 →
  evening_bike = 12 →
  car_trips = 8 →
  morning_bike + evening_bike + car_trips - 15 = 0 :=
by
  sorry

end businessman_travel_l1524_152498


namespace geese_ducks_difference_l1524_152428

def geese : ℝ := 58.0
def ducks : ℝ := 37.0

theorem geese_ducks_difference : geese - ducks = 21.0 := by
  sorry

end geese_ducks_difference_l1524_152428


namespace m_range_l1524_152436

-- Define the condition on x
def X := { x : ℝ | x ≤ -1 }

-- Define the inequality condition
def inequality (m : ℝ) : Prop :=
  ∀ x ∈ X, (m^2 - m) * 4^x - 2^x < 0

-- Theorem statement
theorem m_range :
  ∀ m : ℝ, inequality m ↔ -1 < m ∧ m < 2 := by sorry

end m_range_l1524_152436


namespace staff_distribution_ways_l1524_152443

/-- The number of ways to distribute n indistinguishable objects among k distinct containers,
    with each container receiving at least min_per_container and at most max_per_container objects. -/
def distribute_objects (n : ℕ) (k : ℕ) (min_per_container : ℕ) (max_per_container : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 90 ways to distribute 5 staff members among 3 schools
    with each school receiving at least 1 and at most 2 staff members. -/
theorem staff_distribution_ways : distribute_objects 5 3 1 2 = 90 := by
  sorry

end staff_distribution_ways_l1524_152443


namespace minimal_distance_point_l1524_152432

/-- Given points A, B, and C in the xy-plane, prove that the value of m that minimizes 
    the sum of distances AC + CB is -7/5 when C is constrained to the y-axis. -/
theorem minimal_distance_point (A B C : ℝ × ℝ) : 
  A = (-2, -3) → 
  B = (3, 1) → 
  C.1 = 0 →
  (∀ m' : ℝ, dist A C + dist C B ≤ dist A (0, m') + dist (0, m') B) →
  C.2 = -7/5 := by
sorry

end minimal_distance_point_l1524_152432


namespace fourth_root_equation_l1524_152466

theorem fourth_root_equation (X : ℝ) : 
  (X^5)^(1/4) = 32 * (32^(1/16)) → X = 16 * (2^(1/4)) := by
  sorry

end fourth_root_equation_l1524_152466


namespace susan_money_left_l1524_152463

/-- The amount of money Susan has left after spending at the fair -/
def money_left (initial_amount food_cost ride_cost game_cost : ℕ) : ℕ :=
  initial_amount - (food_cost + ride_cost + game_cost)

/-- Theorem stating that Susan has 10 dollars left to spend -/
theorem susan_money_left :
  let initial_amount := 80
  let food_cost := 15
  let ride_cost := 3 * food_cost
  let game_cost := 10
  money_left initial_amount food_cost ride_cost game_cost = 10 := by
  sorry

end susan_money_left_l1524_152463


namespace kwi_wins_l1524_152409

/-- Represents a frog in the race -/
structure Frog where
  name : String
  jump_length : ℚ
  jumps_per_time_unit : ℚ

/-- Calculates the time taken to complete the race for a given frog -/
def race_time (f : Frog) (race_distance : ℚ) : ℚ :=
  (race_distance / f.jump_length) / f.jumps_per_time_unit

/-- The race distance in decimeters -/
def total_race_distance : ℚ := 400

/-- Kwa, the first frog -/
def kwa : Frog := ⟨"Kwa", 6, 2⟩

/-- Kwi, the second frog -/
def kwi : Frog := ⟨"Kwi", 4, 3⟩

theorem kwi_wins : race_time kwi total_race_distance < race_time kwa total_race_distance := by
  sorry

end kwi_wins_l1524_152409


namespace arithmetic_sequence_nth_term_l1524_152471

/-- 
Given an arithmetic sequence where:
- The first term is 3x - 4
- The second term is 6x - 15
- The third term is 4x + 3
- The nth term is 4021

Prove that n = 627
-/
theorem arithmetic_sequence_nth_term (x : ℚ) (n : ℕ) :
  (3 * x - 4 : ℚ) = (6 * x - 15 : ℚ) - (3 * x - 4 : ℚ) ∧
  (4 * x + 3 : ℚ) = (6 * x - 15 : ℚ) + ((6 * x - 15 : ℚ) - (3 * x - 4 : ℚ)) ∧
  (3 * x - 4 : ℚ) + (n - 1 : ℕ) * ((6 * x - 15 : ℚ) - (3 * x - 4 : ℚ)) = 4021 →
  n = 627 := by
  sorry

end arithmetic_sequence_nth_term_l1524_152471


namespace square_difference_equals_720_l1524_152488

theorem square_difference_equals_720 : (30 + 12)^2 - (12^2 + 30^2) = 720 := by
  sorry

end square_difference_equals_720_l1524_152488


namespace employee_salary_calculation_l1524_152474

/-- Proves that given two employees m and n with a total weekly pay of $572, 
    where m's salary is 120% of n's salary, n's weekly pay is $260. -/
theorem employee_salary_calculation (total_pay m_salary n_salary : ℝ) : 
  total_pay = 572 →
  m_salary = 1.2 * n_salary →
  total_pay = m_salary + n_salary →
  n_salary = 260 := by
  sorry

end employee_salary_calculation_l1524_152474


namespace rectangular_to_polar_conversion_l1524_152483

theorem rectangular_to_polar_conversion :
  ∀ (x y r θ : ℝ),
  x = 8 ∧ y = -8 * Real.sqrt 3 ∧
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = Real.sqrt (x^2 + y^2) ∧
  θ = 2 * Real.pi - Real.pi / 3 →
  r = 16 ∧ θ = 5 * Real.pi / 3 := by
sorry

end rectangular_to_polar_conversion_l1524_152483


namespace arithmetic_calculation_l1524_152480

theorem arithmetic_calculation : 2 + 3 * 4^2 - 5 + 6 = 51 := by
  sorry

end arithmetic_calculation_l1524_152480


namespace gcd_of_840_and_1764_l1524_152429

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_of_840_and_1764_l1524_152429


namespace farm_animals_l1524_152407

/-- Given a farm with cows and horses, prove the number of horses -/
theorem farm_animals (cow_count : ℕ) (horse_count : ℕ) : 
  (cow_count : ℚ) / horse_count = 7 / 2 → cow_count = 21 → horse_count = 6 := by
  sorry

end farm_animals_l1524_152407


namespace probability_sum_23_l1524_152451

/-- Represents a 20-faced die with specific numbered faces and one blank face -/
structure SpecialDie :=
  (numbered_faces : List Nat)
  (blank_face : Unit)

/-- Defines Die A with faces 1-18, 20, and one blank -/
def dieA : SpecialDie :=
  { numbered_faces := List.range 18 ++ [20],
    blank_face := () }

/-- Defines Die B with faces 1-7, 9-20, and one blank -/
def dieB : SpecialDie :=
  { numbered_faces := List.range 7 ++ List.range' 9 20,
    blank_face := () }

/-- Calculates the probability of rolling a sum of 23 with two specific dice -/
def probabilitySum23 (d1 d2 : SpecialDie) : Rat :=
  sorry

theorem probability_sum_23 :
  probabilitySum23 dieA dieB = 7 / 200 := by
  sorry

end probability_sum_23_l1524_152451


namespace max_value_of_a_l1524_152426

theorem max_value_of_a (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  a ≤ Real.sqrt 6 / 3 ∧ ∃ (a₀ : ℝ), a₀ = Real.sqrt 6 / 3 ∧ 
  ∃ (b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 0 ∧ a₀^2 + b₀^2 + c₀^2 = 1 :=
sorry

end max_value_of_a_l1524_152426


namespace simplify_expression_l1524_152489

theorem simplify_expression (a b : ℝ) : 
  3*b*(3*b^2 + 2*b) - b^2 + 2*a*(2*a^2 - 3*a) - 4*a*b = 
  9*b^3 + 5*b^2 + 4*a^3 - 6*a^2 - 4*a*b := by sorry

end simplify_expression_l1524_152489


namespace tank_capacity_l1524_152404

/-- The capacity of a tank with specific leak and inlet properties -/
theorem tank_capacity 
  (leak_empty_time : ℝ) 
  (inlet_rate : ℝ) 
  (combined_empty_time : ℝ) 
  (h1 : leak_empty_time = 6) 
  (h2 : inlet_rate = 2.5) 
  (h3 : combined_empty_time = 8) : 
  ∃ C : ℝ, C = 3600 / 7 ∧ 
    C / leak_empty_time - inlet_rate * 60 = C / combined_empty_time :=
by
  sorry

#check tank_capacity

end tank_capacity_l1524_152404


namespace greater_number_problem_l1524_152449

theorem greater_number_problem (x y : ℝ) : 
  x + y = 40 → x - y = 10 → x > y → x = 25 := by
sorry

end greater_number_problem_l1524_152449


namespace ball_drawing_theorem_l1524_152462

def num_red_balls : ℕ := 4
def num_white_balls : ℕ := 6

def score_red : ℕ := 2
def score_white : ℕ := 1

def ways_to_draw (n r w : ℕ) : ℕ := Nat.choose num_red_balls r * Nat.choose num_white_balls w

theorem ball_drawing_theorem :
  (ways_to_draw 4 4 0 + ways_to_draw 4 3 1 + ways_to_draw 4 2 2 = 115) ∧
  (ways_to_draw 5 2 3 + ways_to_draw 5 3 2 + ways_to_draw 5 4 1 = 186) := by
  sorry

end ball_drawing_theorem_l1524_152462


namespace solution_set_is_open_interval_l1524_152401

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_decreasing : ∀ x y, 0 ≤ x → x < y → f y < f x)

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (2*x - 1) > f (1/3)}

-- Theorem statement
theorem solution_set_is_open_interval :
  solution_set f = Set.Ioo (1/3) (2/3) := by
  sorry

end solution_set_is_open_interval_l1524_152401


namespace game_points_total_l1524_152491

/-- Game points calculation -/
theorem game_points_total (eric mark samanta daisy jake : ℕ) : 
  eric = 6 ∧ 
  mark = eric + eric / 2 ∧ 
  samanta = mark + 8 ∧ 
  daisy = (samanta + mark + eric) - (samanta + mark + eric) / 4 ∧
  jake = max samanta (max mark (max eric daisy)) - min samanta (min mark (min eric daisy)) →
  samanta + mark + eric + daisy + jake = 67 := by
  sorry


end game_points_total_l1524_152491


namespace hyperbola_triangle_perimeter_l1524_152414

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 8

-- Define the points and foci
variable (P Q F₁ F₂ : ℝ × ℝ)

-- Define the chord passing through left focus
def chord_through_left_focus : Prop := 
  (∃ t : ℝ, P = F₁ + t • (Q - F₁)) ∨ (∃ t : ℝ, Q = F₁ + t • (P - F₁))

-- Define the length of PQ
def PQ_length : Prop := 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 7

-- Define F₂ as the right focus
def right_focus (F₂ : ℝ × ℝ) : Prop :=
  F₂.1 > 0 ∧ F₂.1^2 - F₂.2^2 = 8

-- Theorem statement
theorem hyperbola_triangle_perimeter 
  (h_hyperbola_P : hyperbola P.1 P.2)
  (h_hyperbola_Q : hyperbola Q.1 Q.2)
  (h_chord : chord_through_left_focus P Q F₁)
  (h_PQ_length : PQ_length P Q)
  (h_right_focus : right_focus F₂) :
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) +
  Real.sqrt ((Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2) +
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) =
  14 + 8 * Real.sqrt 2 :=
sorry

end hyperbola_triangle_perimeter_l1524_152414


namespace violet_hiking_time_l1524_152421

/-- Proves that Violet and her dog can spend 4 hours hiking given the conditions --/
theorem violet_hiking_time :
  let violet_water_per_hour : ℚ := 800 / 1000  -- Convert ml to L
  let dog_water_per_hour : ℚ := 400 / 1000     -- Convert ml to L
  let total_water_capacity : ℚ := 4.8          -- In L
  
  (total_water_capacity / (violet_water_per_hour + dog_water_per_hour) : ℚ) = 4 := by
  sorry

end violet_hiking_time_l1524_152421


namespace z_in_second_quadrant_l1524_152465

def i : ℂ := Complex.I

def z : ℂ := 2 * i * (1 + i)

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end z_in_second_quadrant_l1524_152465


namespace no_primes_for_d_10_l1524_152452

theorem no_primes_for_d_10 : ¬∃ (p q r : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  (q * r) ∣ (p^2 + 10) ∧
  (r * p) ∣ (q^2 + 10) ∧
  (p * q) ∣ (r^2 + 10) :=
by
  sorry

-- Note: The case for d = 11 is not included as the solution was inconclusive

end no_primes_for_d_10_l1524_152452


namespace triangle_altitude_on_rectangle_diagonal_l1524_152400

/-- Given a rectangle with side lengths a and b, and a triangle constructed on its diagonal
    as base with area equal to the rectangle's area, the altitude of the triangle is
    (2 * a * b) / sqrt(a^2 + b^2). -/
theorem triangle_altitude_on_rectangle_diagonal 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ (h : ℝ), h = (2 * a * b) / Real.sqrt (a^2 + b^2) ∧ 
  h * Real.sqrt (a^2 + b^2) / 2 = a * b := by
  sorry

end triangle_altitude_on_rectangle_diagonal_l1524_152400


namespace easter_egg_hunt_l1524_152461

/-- The Easter egg hunt problem -/
theorem easter_egg_hunt (kevin_eggs bonnie_eggs george_eggs cheryl_eggs : ℕ) 
  (hk : kevin_eggs = 5)
  (hb : bonnie_eggs = 13)
  (hg : george_eggs = 9)
  (hc : cheryl_eggs = 56) :
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
sorry

end easter_egg_hunt_l1524_152461


namespace coefficient_of_x_l1524_152446

/-- The coefficient of x in the simplified form of 2(x - 5) + 5(8 - 3x^2 + 6x) - 9(3x - 2) is 5 -/
theorem coefficient_of_x (x : ℝ) : 
  let expression := 2*(x - 5) + 5*(8 - 3*x^2 + 6*x) - 9*(3*x - 2)
  ∃ a b c : ℝ, expression = a*x^2 + 5*x + c := by
  sorry

end coefficient_of_x_l1524_152446


namespace circumscribing_circle_diameter_l1524_152475

/-- The diameter of a circle circumscribing 8 tangent circles -/
theorem circumscribing_circle_diameter (r : ℝ) (h : r = 5) : 
  let n : ℕ := 8
  let small_circle_radius := r
  let large_circle_diameter := 2 * r * (3 + Real.sqrt 3)
  large_circle_diameter = 10 * (3 + Real.sqrt 3) :=
by sorry

end circumscribing_circle_diameter_l1524_152475


namespace unique_intersection_l1524_152431

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|4 * x - 3|

-- State the theorem
theorem unique_intersection :
  ∃! p : ℝ × ℝ, f p.1 = p.2 ∧ g p.1 = p.2 :=
sorry

end unique_intersection_l1524_152431


namespace invalid_triangle_1_invalid_triangle_2_l1524_152438

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the property that triangle angles sum to 180 degrees
def valid_triangle (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

-- Theorem: A triangle with angles 90°, 60°, and 60° cannot exist
theorem invalid_triangle_1 : 
  ¬ ∃ (t : Triangle), t.angle1 = 90 ∧ t.angle2 = 60 ∧ t.angle3 = 60 ∧ valid_triangle t :=
sorry

-- Theorem: A triangle with angles 90°, 50°, and 50° cannot exist
theorem invalid_triangle_2 :
  ¬ ∃ (t : Triangle), t.angle1 = 90 ∧ t.angle2 = 50 ∧ t.angle3 = 50 ∧ valid_triangle t :=
sorry

end invalid_triangle_1_invalid_triangle_2_l1524_152438


namespace range_of_x_l1524_152415

def p (x : ℝ) := 1 / (x - 2) < 0
def q (x : ℝ) := x^2 - 4*x - 5 < 0

theorem range_of_x (x : ℝ) :
  (p x ∨ q x) ∧ ¬(p x ∧ q x) →
  x ∈ Set.Iic (-1) ∪ Set.Ico 3 5 :=
by sorry

end range_of_x_l1524_152415


namespace total_weekly_sleep_time_l1524_152478

/-- Represents the sleep patterns of animals -/
structure SleepPattern where
  evenDaySleep : ℕ
  oddDaySleep : ℕ

/-- Calculates the total weekly sleep for an animal given its sleep pattern -/
def weeklyTotalSleep (pattern : SleepPattern) : ℕ :=
  3 * pattern.evenDaySleep + 4 * pattern.oddDaySleep

/-- The sleep pattern of a cougar -/
def cougarSleep : SleepPattern :=
  { evenDaySleep := 4, oddDaySleep := 6 }

/-- The sleep pattern of a zebra -/
def zebraSleep : SleepPattern :=
  { evenDaySleep := cougarSleep.evenDaySleep + 2,
    oddDaySleep := cougarSleep.oddDaySleep + 2 }

/-- Theorem stating the total weekly sleep time for both animals -/
theorem total_weekly_sleep_time :
  weeklyTotalSleep cougarSleep + weeklyTotalSleep zebraSleep = 86 := by
  sorry


end total_weekly_sleep_time_l1524_152478


namespace smallest_abs_value_rational_l1524_152482

theorem smallest_abs_value_rational (q : ℚ) : |0| ≤ |q| := by
  sorry

end smallest_abs_value_rational_l1524_152482


namespace lcm_fraction_evenness_l1524_152477

theorem lcm_fraction_evenness (x y z : ℕ+) :
  ∃ (k : ℕ), k > 0 ∧ k % 2 = 0 ∧
  (Nat.lcm x.val y.val + Nat.lcm y.val z.val) / Nat.lcm x.val z.val = k ∧
  ∀ (n : ℕ), n > 0 → n % 2 = 0 →
    ∃ (a b c : ℕ+), (Nat.lcm a.val b.val + Nat.lcm b.val c.val) / Nat.lcm a.val c.val = n :=
by sorry

end lcm_fraction_evenness_l1524_152477


namespace percent_increase_l1524_152424

theorem percent_increase (N M P : ℝ) (h : M = N * (1 + P / 100)) :
  (M - N) / N * 100 = P := by
  sorry

end percent_increase_l1524_152424


namespace minesweeper_configurations_l1524_152412

def valid_configuration (A B C D E : ℕ) : Prop :=
  A + B = 2 ∧ B + C + D = 1 ∧ D + E = 2

def count_configurations : ℕ := sorry

theorem minesweeper_configurations :
  count_configurations = 4545 := by sorry

end minesweeper_configurations_l1524_152412


namespace teachers_made_28_materials_l1524_152456

/-- Given the number of recycled materials made by a group and the total number of recycled products to be sold, 
    calculate the number of recycled materials made by teachers. -/
def teachers_recycled_materials (group_materials : ℕ) (total_products : ℕ) : ℕ :=
  total_products - group_materials

/-- Theorem: Given that the group made 65 recycled materials and the total number of recycled products
    to be sold is 93, prove that the teachers made 28 recycled materials. -/
theorem teachers_made_28_materials : teachers_recycled_materials 65 93 = 28 := by
  sorry

end teachers_made_28_materials_l1524_152456


namespace no_odd_tens_digit_squares_l1524_152457

/-- The set of numbers from 1 to 50 -/
def S : Set Nat := {n | 1 ≤ n ∧ n ≤ 50}

/-- A number ends in 3 or 7 -/
def ends_in_3_or_7 (n : Nat) : Prop := n % 10 = 3 ∨ n % 10 = 7

/-- The tens digit of a number -/
def tens_digit (n : Nat) : Nat := (n / 10) % 10

/-- A number is even -/
def is_even (n : Nat) : Prop := n % 2 = 0

theorem no_odd_tens_digit_squares :
  ∀ n ∈ S, ends_in_3_or_7 n → is_even (tens_digit (n^2)) := by sorry

end no_odd_tens_digit_squares_l1524_152457


namespace money_left_after_distributions_l1524_152408

/-- Calculates the amount of money left after distributions --/
theorem money_left_after_distributions (income : ℝ) : 
  income = 1000 → 
  income * (1 - 0.2 - 0.2) * (1 - 0.1) = 540 := by
  sorry

#check money_left_after_distributions

end money_left_after_distributions_l1524_152408


namespace simplify_expression_l1524_152402

theorem simplify_expression : (81 ^ (1/4) - Real.sqrt 12.25) ^ 2 = 1/4 := by sorry

end simplify_expression_l1524_152402


namespace part_to_whole_ratio_l1524_152420

theorem part_to_whole_ratio (N : ℝ) (h1 : (1/3) * (2/5) * N = 17) (h2 : 0.4 * N = 204) : 
  17 / N = 1 / 30 := by
sorry

end part_to_whole_ratio_l1524_152420


namespace max_monthly_profit_l1524_152425

/-- Represents the monthly profit function for a product with given cost and pricing conditions. -/
def monthly_profit (x : ℕ) : ℚ :=
  -10 * x^2 + 110 * x + 2100

/-- Theorem stating the maximum monthly profit and the optimal selling prices. -/
theorem max_monthly_profit :
  (∀ x : ℕ, 0 < x → x ≤ 15 → monthly_profit x ≤ 2400) ∧
  monthly_profit 5 = 2400 ∧
  monthly_profit 6 = 2400 :=
sorry

end max_monthly_profit_l1524_152425


namespace marvins_substitution_l1524_152460

theorem marvins_substitution (a b c d f : ℤ) : 
  a = 3 → b = 4 → c = 7 → d = 5 →
  (a + b - c + d - f = a + (b - (c + (d - f)))) →
  f = 5 := by sorry

end marvins_substitution_l1524_152460


namespace reggies_money_l1524_152490

/-- The amount of money Reggie's father gave him -/
def money_given : ℕ := sorry

/-- The number of books Reggie bought -/
def books_bought : ℕ := 5

/-- The cost of each book in dollars -/
def book_cost : ℕ := 2

/-- The amount of money Reggie has left after buying the books -/
def money_left : ℕ := 38

/-- Theorem stating that the money given by Reggie's father is $48 -/
theorem reggies_money : money_given = books_bought * book_cost + money_left := by sorry

end reggies_money_l1524_152490


namespace parabola_properties_l1524_152492

/-- Given a parabola y = ax² - 5x - 3 passing through (-1, 4), prove its properties -/
theorem parabola_properties (a : ℝ) : 
  (a * (-1)^2 - 5 * (-1) - 3 = 4) → -- The parabola passes through (-1, 4)
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * x₁^2 - 5 * x₁ - 3 = 0 ∧ a * x₂^2 - 5 * x₂ - 3 = 0) ∧ -- Intersects x-axis at two points
  (- (-5) / (2 * a) = 5/4) -- Axis of symmetry is x = 5/4
  := by sorry

end parabola_properties_l1524_152492


namespace sara_initial_quarters_l1524_152416

/-- The number of quarters Sara had initially -/
def initial_quarters : ℕ := sorry

/-- The number of quarters Sara's dad gave her -/
def quarters_from_dad : ℕ := 49

/-- The total number of quarters Sara has after receiving quarters from her dad -/
def total_quarters : ℕ := 70

/-- Theorem stating that Sara initially had 21 quarters -/
theorem sara_initial_quarters : initial_quarters = 21 := by
  sorry

end sara_initial_quarters_l1524_152416


namespace fraction_equality_l1524_152437

theorem fraction_equality : (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end fraction_equality_l1524_152437


namespace pomelo_sales_theorem_l1524_152453

/-- Represents the sales data for a week -/
structure WeeklySales where
  planned_daily : ℕ
  deviations : List ℤ
  selling_price : ℕ
  shipping_cost : ℕ

/-- Calculates the difference between highest and lowest sales days -/
def sales_difference (sales : WeeklySales) : ℕ :=
  let max_dev := sales.deviations.maximum?
  let min_dev := sales.deviations.minimum?
  match max_dev, min_dev with
  | some max, some min => (max - min).natAbs
  | _, _ => 0

/-- Calculates the total sales for the week -/
def total_sales (sales : WeeklySales) : ℕ :=
  sales.planned_daily * 7 + sales.deviations.sum.natAbs

/-- Calculates the total profit for the week -/
def total_profit (sales : WeeklySales) : ℕ :=
  (sales.selling_price - sales.shipping_cost) * (total_sales sales)

/-- Main theorem to prove -/
theorem pomelo_sales_theorem (sales : WeeklySales)
  (h1 : sales.planned_daily = 100)
  (h2 : sales.deviations = [3, -5, -2, 11, -7, 13, 5])
  (h3 : sales.selling_price = 8)
  (h4 : sales.shipping_cost = 3) :
  sales_difference sales = 20 ∧
  total_sales sales = 718 ∧
  total_profit sales = 3590 := by
  sorry


end pomelo_sales_theorem_l1524_152453


namespace gcd_lcm_392_count_l1524_152423

theorem gcd_lcm_392_count : 
  ∃! n : ℕ, n > 0 ∧ 
  (∃ S : Finset ℕ, S.card = n ∧
    ∀ d ∈ S, d > 0 ∧
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧
      Nat.gcd a b * Nat.lcm a b = 392 ∧
      Nat.gcd a b = d) ∧
    (∀ a b : ℕ, a > 0 → b > 0 →
      Nat.gcd a b * Nat.lcm a b = 392 →
      Nat.gcd a b ∈ S)) :=
sorry

end gcd_lcm_392_count_l1524_152423


namespace number_problem_l1524_152439

theorem number_problem : ∃ (x : ℝ), x = 40 ∧ 0.8 * x > (4/5 * 15 + 20) := by
  sorry

end number_problem_l1524_152439


namespace first_question_percentage_l1524_152473

theorem first_question_percentage (second_correct : Real) 
                                  (neither_correct : Real)
                                  (both_correct : Real) :
  second_correct = 25 →
  neither_correct = 20 →
  both_correct = 20 →
  ∃ (first_correct : Real),
    first_correct = 75 ∧
    first_correct + second_correct - both_correct = 100 - neither_correct :=
by sorry

end first_question_percentage_l1524_152473


namespace tic_tac_toe_tie_probability_l1524_152435

theorem tic_tac_toe_tie_probability 
  (amy_win_prob : ℚ) 
  (lily_win_prob : ℚ) 
  (h1 : amy_win_prob = 3/8) 
  (h2 : lily_win_prob = 3/10) 
  (h3 : amy_win_prob + lily_win_prob ≤ 1) :
  1 - (amy_win_prob + lily_win_prob) = 13/40 :=
by sorry

end tic_tac_toe_tie_probability_l1524_152435


namespace v_2008_equals_3703_l1524_152441

/-- Defines the sequence v_n as described in the problem -/
def v : ℕ → ℕ := sorry

/-- The 2008th term of the sequence v_n is 3703 -/
theorem v_2008_equals_3703 : v 2008 = 3703 := by sorry

end v_2008_equals_3703_l1524_152441


namespace quadratic_range_on_interval_l1524_152410

/-- A quadratic function defined on a closed interval -/
def QuadraticFunction (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The range of a quadratic function on a closed interval -/
def QuadraticRange (a b c : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc (-1 : ℝ) 2, y = QuadraticFunction a b c x}

theorem quadratic_range_on_interval
  (a b c : ℝ) (h : a > 0) :
  QuadraticRange a b c =
    Set.Icc (min (a - b + c) (c - b^2 / (4 * a))) (4 * a + 2 * b + c) := by
  sorry

end quadratic_range_on_interval_l1524_152410


namespace regular_polygon_sides_l1524_152418

/-- A regular polygon with perimeter 160 and side length 10 has 16 sides -/
theorem regular_polygon_sides (p : ℕ) (perimeter side_length : ℝ) 
  (h_perimeter : perimeter = 160)
  (h_side_length : side_length = 10)
  (h_regular : p * side_length = perimeter) : 
  p = 16 := by sorry

end regular_polygon_sides_l1524_152418


namespace mixture_carbonated_water_percentage_l1524_152468

/-- Calculates the percentage of carbonated water in a mixture of two solutions -/
def carbonated_water_percentage (solution1_percent : ℝ) (solution1_carbonated : ℝ) 
  (solution2_carbonated : ℝ) : ℝ :=
  solution1_percent * solution1_carbonated + (1 - solution1_percent) * solution2_carbonated

theorem mixture_carbonated_water_percentage :
  carbonated_water_percentage 0.1999999999999997 0.80 0.55 = 0.5999999999999999 := by
  sorry

#eval carbonated_water_percentage 0.1999999999999997 0.80 0.55

end mixture_carbonated_water_percentage_l1524_152468


namespace sum_a_equals_1649_l1524_152484

def a (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 20 = 0 then 15
  else if n % 20 = 0 ∧ n % 18 = 0 then 20
  else if n % 18 = 0 ∧ n % 15 = 0 then 18
  else 0

theorem sum_a_equals_1649 :
  (Finset.range 2999).sum a = 1649 := by
  sorry

end sum_a_equals_1649_l1524_152484


namespace inequality_proof_l1524_152459

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c := by
  sorry

end inequality_proof_l1524_152459


namespace runners_meet_time_l1524_152440

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.minutes + m
  { hours := t.hours + totalMinutes / 60
    minutes := totalMinutes % 60 }

theorem runners_meet_time (startTime : Time) (lapTime1 lapTime2 lapTime3 : Nat) : 
  startTime.hours = 7 ∧ startTime.minutes = 45 ∧
  lapTime1 = 5 ∧ lapTime2 = 8 ∧ lapTime3 = 10 →
  let meetTime := addMinutes startTime (Nat.lcm lapTime1 (Nat.lcm lapTime2 lapTime3))
  meetTime.hours = 8 ∧ meetTime.minutes = 25 := by
  sorry

end runners_meet_time_l1524_152440


namespace red_marbles_in_bag_l1524_152467

theorem red_marbles_in_bag (total : ℕ) (prob : ℚ) (h_total : total = 84) (h_prob : prob = 36/49) :
  ∃ red : ℕ, red = 12 ∧ (1 - (red : ℚ) / total) * (1 - (red : ℚ) / total) = prob :=
sorry

end red_marbles_in_bag_l1524_152467


namespace sibling_product_specific_household_l1524_152455

/-- In a household with girls and boys, one boy counts all other children as siblings. -/
structure Household where
  girls : ℕ
  boys : ℕ
  counter : ℕ
  counter_is_boy : counter < boys

/-- The number of sisters the counter sees -/
def sisters (h : Household) : ℕ := h.girls

/-- The number of brothers the counter sees -/
def brothers (h : Household) : ℕ := h.boys - 1

/-- The product of sisters and brothers the counter sees -/
def sibling_product (h : Household) : ℕ := sisters h * brothers h

theorem sibling_product_specific_household :
  ∀ h : Household, h.girls = 5 → h.boys = 7 → sibling_product h = 24 := by
  sorry

end sibling_product_specific_household_l1524_152455


namespace second_company_base_rate_l1524_152454

/-- Represents the base rate and per-minute charge for a telephone company -/
structure TelephoneRate where
  baseRate : ℝ
  perMinuteCharge : ℝ

/-- Calculates the total charge for a given number of minutes -/
def totalCharge (rate : TelephoneRate) (minutes : ℝ) : ℝ :=
  rate.baseRate + rate.perMinuteCharge * minutes

theorem second_company_base_rate :
  let unitedRate : TelephoneRate := { baseRate := 11, perMinuteCharge := 0.25 }
  let otherRate : TelephoneRate := { baseRate := x, perMinuteCharge := 0.20 }
  let minutes : ℝ := 20
  totalCharge unitedRate minutes = totalCharge otherRate minutes →
  x = 12 := by
sorry

end second_company_base_rate_l1524_152454


namespace parabola_equation_l1524_152419

/-- Represents a parabola -/
structure Parabola where
  -- The equation of the parabola in the form y² = 2px or x² = 2py
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

/-- Checks if the parabola has a given focus -/
def Parabola.hasFocus (p : Parabola) (fx fy : ℝ) : Prop :=
  ∃ (a : ℝ), (p.equation = fun x y ↦ (y - fy)^2 = 4*a*(x - fx)) ∨
             (p.equation = fun x y ↦ (x - fx)^2 = 4*a*(y - fy))

/-- Checks if the parabola has a given directrix -/
def Parabola.hasDirectrix (p : Parabola) (d : ℝ) : Prop :=
  ∃ (a : ℝ), (p.equation = fun x y ↦ y^2 = 4*a*(x + a)) ∧ d = -a ∨
             (p.equation = fun x y ↦ x^2 = 4*a*(y + a)) ∧ d = -a

theorem parabola_equation (p : Parabola) :
  p.hasFocus (-2) 0 →
  p.hasDirectrix (-1) →
  p.contains 1 2 →
  (p.equation = fun x y ↦ y^2 = 4*x) ∨
  (p.equation = fun x y ↦ x^2 = 1/2*y) :=
sorry

end parabola_equation_l1524_152419


namespace min_value_problem_l1524_152430

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2 * a + 3 * b = 1) :
  (2 / a) + (3 / b) ≥ 25 := by
  sorry

end min_value_problem_l1524_152430


namespace value_of_expression_l1524_152495

-- Define the polynomial
def p (x h k : ℝ) : ℝ := 5 * x^4 - h * x^2 + k

-- State the theorem
theorem value_of_expression (h k : ℝ) :
  (p 3 h k = 0) → (p (-1) h k = 0) → (p 2 h k = 0) → |5 * h - 4 * k| = 70 := by
  sorry

end value_of_expression_l1524_152495


namespace expression_factorization_l1524_152481

theorem expression_factorization (x : ℝ) : 
  (12 * x^3 + 95 * x - 6) - (-3 * x^3 + 5 * x - 6) = 15 * x * (x^2 + 6) := by
  sorry

end expression_factorization_l1524_152481


namespace student_rank_l1524_152411

theorem student_rank (total : Nat) (left_rank : Nat) (right_rank : Nat) : 
  total = 20 → left_rank = 8 → right_rank = total - left_rank + 1 → right_rank = 13 := by
  sorry

end student_rank_l1524_152411


namespace right_triangle_area_l1524_152464

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 8 * Real.sqrt 2 →
  angle = 45 * (π / 180) →
  let area := (hypotenuse^2 / 4)
  area = 32 := by sorry

end right_triangle_area_l1524_152464


namespace non_technicians_percentage_l1524_152406

/-- Represents the composition of workers in a factory -/
structure Factory where
  total : ℕ
  technicians : ℕ
  permanent_technicians : ℕ
  permanent_non_technicians : ℕ
  temporary : ℕ

/-- The conditions of the factory as described in the problem -/
def factory_conditions (f : Factory) : Prop :=
  f.technicians = f.total / 2 ∧
  f.permanent_technicians = f.technicians / 2 ∧
  f.permanent_non_technicians = (f.total - f.technicians) / 2 ∧
  f.temporary = f.total / 2

/-- The theorem stating that under the given conditions, 
    non-technicians make up 50% of the workforce -/
theorem non_technicians_percentage (f : Factory) 
  (h : factory_conditions f) : 
  (f.total - f.technicians) * 100 / f.total = 50 := by
  sorry


end non_technicians_percentage_l1524_152406


namespace shannon_stones_l1524_152403

/-- The number of heart-shaped stones Shannon wants in each bracelet -/
def stones_per_bracelet : ℕ := 8

/-- The number of bracelets Shannon can make -/
def number_of_bracelets : ℕ := 6

/-- The total number of heart-shaped stones Shannon brought -/
def total_stones : ℕ := stones_per_bracelet * number_of_bracelets

theorem shannon_stones : total_stones = 48 := by
  sorry

end shannon_stones_l1524_152403


namespace complex_product_equals_2401_l1524_152444

theorem complex_product_equals_2401 :
  let x : ℂ := Complex.exp (2 * Real.pi * I / 9)
  (3 * x + x^2) * (3 * x^2 + x^4) * (3 * x^3 + x^6) * (3 * x^4 + x^8) *
  (3 * x^5 + x^10) * (3 * x^6 + x^12) * (3 * x^7 + x^14) = 2401 := by
  sorry

end complex_product_equals_2401_l1524_152444


namespace edward_escape_problem_l1524_152427

/-- The problem of Edward escaping from prison and being hit by an arrow. -/
theorem edward_escape_problem (initial_distance : ℝ) (arrow_initial_velocity : ℝ) 
  (edward_acceleration : ℝ) (arrow_deceleration : ℝ) :
  initial_distance = 1875 →
  arrow_initial_velocity = 100 →
  edward_acceleration = 1 →
  arrow_deceleration = 1 →
  ∃ t : ℝ, t > 0 ∧ 
    (-1/2 * arrow_deceleration * t^2 + arrow_initial_velocity * t) = 
    (1/2 * edward_acceleration * t^2 + initial_distance) ∧
    (arrow_initial_velocity - arrow_deceleration * t) = 75 :=
by sorry

end edward_escape_problem_l1524_152427


namespace wilsons_theorem_l1524_152496

theorem wilsons_theorem (p : ℕ) (h : p > 1) :
  Nat.Prime p ↔ p ∣ (Nat.factorial (p - 1) + 1) := by
  sorry

end wilsons_theorem_l1524_152496


namespace bennys_seashells_l1524_152422

theorem bennys_seashells (initial_seashells given_away_seashells : ℚ) 
  (h1 : initial_seashells = 66.5)
  (h2 : given_away_seashells = 52.5) :
  initial_seashells - given_away_seashells = 14 :=
by sorry

end bennys_seashells_l1524_152422


namespace interest_rate_proof_l1524_152485

/-- Represents the rate of interest per annum as a percentage -/
def rate : ℝ := 9

/-- The amount lent to B -/
def principal_B : ℝ := 5000

/-- The amount lent to C -/
def principal_C : ℝ := 3000

/-- The time period for B's loan in years -/
def time_B : ℝ := 2

/-- The time period for C's loan in years -/
def time_C : ℝ := 4

/-- The total interest received from both B and C -/
def total_interest : ℝ := 1980

/-- Theorem stating that the given rate satisfies the problem conditions -/
theorem interest_rate_proof :
  (principal_B * rate * time_B / 100 + principal_C * rate * time_C / 100) = total_interest :=
by sorry

end interest_rate_proof_l1524_152485


namespace correct_cobs_per_row_l1524_152499

/-- Represents the number of corn cobs in each row -/
def cobs_per_row : ℕ := 4

/-- Represents the number of rows in the first field -/
def rows_field1 : ℕ := 13

/-- Represents the number of rows in the second field -/
def rows_field2 : ℕ := 16

/-- Represents the total number of corn cobs -/
def total_cobs : ℕ := 116

/-- Theorem stating that the number of corn cobs per row is correct -/
theorem correct_cobs_per_row : 
  cobs_per_row * rows_field1 + cobs_per_row * rows_field2 = total_cobs := by
  sorry

end correct_cobs_per_row_l1524_152499


namespace smallest_sum_of_perfect_squares_l1524_152472

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 221 → ∃ (a b : ℕ), a^2 - b^2 = 221 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 229 :=
by sorry

end smallest_sum_of_perfect_squares_l1524_152472


namespace volume_ratio_octahedron_cube_l1524_152487

/-- A regular octahedron -/
structure RegularOctahedron where
  edge_length : ℝ
  volume : ℝ

/-- A cube whose vertices are the centers of the faces of a regular octahedron -/
structure RelatedCube where
  diagonal : ℝ
  volume : ℝ

/-- The relationship between a regular octahedron and its related cube -/
def octahedron_cube_relation (o : RegularOctahedron) (c : RelatedCube) : Prop :=
  c.diagonal = 2 * o.edge_length

theorem volume_ratio_octahedron_cube (o : RegularOctahedron) (c : RelatedCube) 
  (h : octahedron_cube_relation o c) : 
  o.volume / c.volume = 3 / 8 := by
  sorry

end volume_ratio_octahedron_cube_l1524_152487


namespace f_has_three_distinct_roots_l1524_152417

/-- The polynomial function whose roots we're counting -/
def f (x : ℝ) : ℝ := (x - 8) * (x^2 + 4*x + 3)

/-- The theorem stating that f has exactly 3 distinct real roots -/
theorem f_has_three_distinct_roots : 
  ∃ (r₁ r₂ r₃ : ℝ), (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) ∧ 
  (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃) ∧
  (∀ x : ℝ, f x = 0 → x = r₁ ∨ x = r₂ ∨ x = r₃) :=
sorry

end f_has_three_distinct_roots_l1524_152417


namespace sequence_a_4_equals_zero_l1524_152413

theorem sequence_a_4_equals_zero :
  let a : ℕ+ → ℤ := fun n => n.val^2 - 3*n.val - 4
  a 4 = 0 := by
  sorry

end sequence_a_4_equals_zero_l1524_152413


namespace sqrt_sum_inequality_l1524_152445

theorem sqrt_sum_inequality {a b c d : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt (a / (b + c + d)) + Real.sqrt (b / (a + c + d)) +
  Real.sqrt (c / (a + b + d)) + Real.sqrt (d / (a + b + c)) > 2 ∧
  ∀ m : ℝ, (∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 →
    Real.sqrt (a / (b + c + d)) + Real.sqrt (b / (a + c + d)) +
    Real.sqrt (c / (a + b + d)) + Real.sqrt (d / (a + b + c)) > m) →
  m ≤ 2 :=
by sorry

end sqrt_sum_inequality_l1524_152445


namespace tan_alpha_value_l1524_152448

theorem tan_alpha_value (α : Real) (h : Real.tan (α - Real.pi/4) = 1/6) : 
  Real.tan α = 7/5 := by
  sorry

end tan_alpha_value_l1524_152448


namespace sum_of_photo_areas_l1524_152486

-- Define the side lengths of the three square photos
def photo1_side : ℝ := 2
def photo2_side : ℝ := 3
def photo3_side : ℝ := 1

-- Define the function to calculate the area of a square
def square_area (side : ℝ) : ℝ := side * side

-- Theorem: The sum of the areas of the three square photos is 14 square inches
theorem sum_of_photo_areas :
  square_area photo1_side + square_area photo2_side + square_area photo3_side = 14 := by
  sorry

end sum_of_photo_areas_l1524_152486


namespace stuffed_dogs_count_l1524_152493

/-- The number of boxes of stuffed toy dogs -/
def num_boxes : ℕ := 7

/-- The number of dogs in each box -/
def dogs_per_box : ℕ := 4

/-- The total number of dogs -/
def total_dogs : ℕ := num_boxes * dogs_per_box

theorem stuffed_dogs_count : total_dogs = 28 := by
  sorry

end stuffed_dogs_count_l1524_152493


namespace physics_marks_l1524_152479

def marks_english : ℕ := 81
def marks_mathematics : ℕ := 65
def marks_chemistry : ℕ := 67
def marks_biology : ℕ := 85
def average_marks : ℕ := 76
def total_subjects : ℕ := 5

theorem physics_marks :
  let total_marks := average_marks * total_subjects
  let known_marks := marks_english + marks_mathematics + marks_chemistry + marks_biology
  total_marks - known_marks = 82 := by sorry

end physics_marks_l1524_152479


namespace stratified_sampling_sample_size_l1524_152469

theorem stratified_sampling_sample_size (total_population : ℕ) (elderly_population : ℕ) (elderly_sample : ℕ) (sample_size : ℕ) :
  total_population = 162 →
  elderly_population = 27 →
  elderly_sample = 6 →
  (elderly_sample : ℚ) / sample_size = (elderly_population : ℚ) / total_population →
  sample_size = 36 := by
sorry

end stratified_sampling_sample_size_l1524_152469


namespace quadratic_roots_product_l1524_152433

theorem quadratic_roots_product (c d : ℝ) : 
  (3 * c^2 + 9 * c - 21 = 0) → 
  (3 * d^2 + 9 * d - 21 = 0) → 
  (3 * c - 4) * (6 * d - 8) = 122 := by
sorry

end quadratic_roots_product_l1524_152433


namespace vector_magnitude_problem_l1524_152405

theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (ha : ‖a‖ = 3)
  (hb : ‖b‖ = 4)
  (hab : ‖a + b‖ = 2) :
  ‖a - b‖ = Real.sqrt 46 := by
  sorry

end vector_magnitude_problem_l1524_152405


namespace stratified_sampling_size_l1524_152494

theorem stratified_sampling_size (total_employees : ℕ) (male_employees : ℕ) (female_sample : ℕ) (sample_size : ℕ) : 
  total_employees = 120 →
  male_employees = 90 →
  female_sample = 9 →
  (total_employees - male_employees) / total_employees = female_sample / sample_size →
  sample_size = 36 := by
sorry

end stratified_sampling_size_l1524_152494


namespace complex_power_48_l1524_152470

theorem complex_power_48 :
  (Complex.exp (Complex.I * Real.pi * (125 / 180)))^48 = Complex.ofReal (-1/2) + Complex.I * Complex.ofReal (Real.sqrt 3 / 2) := by
  sorry

end complex_power_48_l1524_152470


namespace molly_gift_cost_l1524_152458

/-- Represents the cost structure and family composition for Molly's gift-sending scenario -/
structure GiftSendingScenario where
  cost_per_package : ℕ
  num_parents : ℕ
  num_brothers : ℕ
  num_children_per_brother : ℕ

/-- Calculates the total number of relatives Molly needs to send gifts to -/
def total_relatives (scenario : GiftSendingScenario) : ℕ :=
  scenario.num_parents + 
  scenario.num_brothers + 
  scenario.num_brothers + -- for sisters-in-law
  scenario.num_brothers * scenario.num_children_per_brother

/-- Calculates the total cost of sending gifts to all relatives -/
def total_cost (scenario : GiftSendingScenario) : ℕ :=
  scenario.cost_per_package * total_relatives scenario

/-- Theorem stating that Molly's total cost for sending gifts is $70 -/
theorem molly_gift_cost : 
  ∀ (scenario : GiftSendingScenario), 
  scenario.cost_per_package = 5 ∧ 
  scenario.num_parents = 2 ∧ 
  scenario.num_brothers = 3 ∧ 
  scenario.num_children_per_brother = 2 → 
  total_cost scenario = 70 := by
  sorry

end molly_gift_cost_l1524_152458


namespace probability_white_then_red_l1524_152447

/-- The probability of drawing a white marble first and then a red marble second, without replacement, from a bag containing 4 red marbles and 6 white marbles. -/
theorem probability_white_then_red (red_marbles white_marbles : ℕ) 
  (h_red : red_marbles = 4) 
  (h_white : white_marbles = 6) : 
  (white_marbles : ℚ) / (red_marbles + white_marbles) * 
  (red_marbles : ℚ) / (red_marbles + white_marbles - 1) = 4 / 15 := by
  sorry

end probability_white_then_red_l1524_152447


namespace largest_integer_less_than_93_remainder_4_mod_7_l1524_152434

theorem largest_integer_less_than_93_remainder_4_mod_7 :
  ∃ n : ℕ, n < 93 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 93 ∧ m % 7 = 4 → m ≤ n :=
by
  use 88
  sorry

end largest_integer_less_than_93_remainder_4_mod_7_l1524_152434


namespace completing_square_quadratic_l1524_152497

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) :=
sorry

end completing_square_quadratic_l1524_152497


namespace intersection_M_N_l1524_152450

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4*x + 3}
def N : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {y | y ≥ -1} := by sorry

end intersection_M_N_l1524_152450
