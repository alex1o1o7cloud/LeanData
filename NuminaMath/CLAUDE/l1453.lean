import Mathlib

namespace seeds_per_flowerbed_is_four_l1453_145359

/-- The number of flowerbeds -/
def num_flowerbeds : ℕ := 8

/-- The total number of seeds planted -/
def total_seeds : ℕ := 32

/-- The number of seeds in each flowerbed -/
def seeds_per_flowerbed : ℕ := total_seeds / num_flowerbeds

/-- Theorem: The number of seeds per flowerbed is 4 -/
theorem seeds_per_flowerbed_is_four :
  seeds_per_flowerbed = 4 :=
by sorry

end seeds_per_flowerbed_is_four_l1453_145359


namespace max_cubes_fit_l1453_145380

theorem max_cubes_fit (large_side : ℕ) (small_edge : ℕ) : large_side = 10 ∧ small_edge = 2 →
  (large_side ^ 3) / (small_edge ^ 3) = 125 := by
  sorry

end max_cubes_fit_l1453_145380


namespace sum_of_solutions_quadratic_l1453_145304

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 7*x - 20) → (∃ y : ℝ, y^2 = 7*y - 20 ∧ x + y = 7) := by
  sorry

end sum_of_solutions_quadratic_l1453_145304


namespace sin_two_alpha_zero_l1453_145310

theorem sin_two_alpha_zero (α : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = Real.sin x - Real.cos x) 
  (h2 : f α = 1) : 
  Real.sin (2 * α) = 0 := by
  sorry

end sin_two_alpha_zero_l1453_145310


namespace smallest_angle_in_triangle_l1453_145309

theorem smallest_angle_in_triangle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = 40 →           -- One angle is 40°
  c = 3 * b →        -- The other two angles are in the ratio 1:3
  min a (min b c) = 35 :=  -- The smallest angle is 35°
by sorry

end smallest_angle_in_triangle_l1453_145309


namespace problem_statement_l1453_145325

def p (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, 2 * x - 2 ≥ m^2 - 3 * m

def q (m a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1) 1, m ≤ a * x

theorem problem_statement (m : ℝ) :
  (p m ↔ m ∈ Set.Icc 1 2) ∧
  ((¬(p m) ∧ ¬(q m 1)) ∧ (p m ∨ q m 1) ↔ m ∈ Set.Ioi 1 ∪ Set.Iic 2 \ {1}) :=
sorry

end problem_statement_l1453_145325


namespace monotonicity_condition_l1453_145385

/-- The function f(x) = √(x² + 1) - ax is monotonic on [0,+∞) if and only if a ≥ 1, given that a > 0 -/
theorem monotonicity_condition (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → (Real.sqrt (x^2 + 1) - a * x < Real.sqrt (y^2 + 1) - a * y ∨
                               Real.sqrt (x^2 + 1) - a * x > Real.sqrt (y^2 + 1) - a * y)) ↔
  a ≥ 1 := by
sorry

end monotonicity_condition_l1453_145385


namespace triangle_theorem_l1453_145306

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def given_conditions (t : Triangle) : Prop :=
  t.b = 3 ∧
  t.b * t.c * Real.cos t.A = -6 ∧
  1/2 * t.b * t.c * Real.sin t.A = 3

-- State the theorem
theorem triangle_theorem (t : Triangle) :
  given_conditions t →
  t.A = Real.pi * 3/4 ∧ t.a = Real.sqrt 29 := by
  sorry

end triangle_theorem_l1453_145306


namespace infinite_divisibility_sequence_l1453_145399

theorem infinite_divisibility_sequence : 
  ∃ (a : ℕ → ℕ), ∀ n, (a n)^2 ∣ (2^(a n) + 3^(a n)) :=
sorry

end infinite_divisibility_sequence_l1453_145399


namespace square_perimeter_relation_l1453_145351

/-- Given a square C with perimeter 40 cm and a square D with area equal to one-third the area of square C, 
    the perimeter of square D is (40√3)/3 cm. -/
theorem square_perimeter_relation (C D : Real) : 
  (C * 4 = 40) →  -- Perimeter of square C is 40 cm
  (D^2 = (C^2) / 3) →  -- Area of square D is one-third the area of square C
  (D * 4 = 40 * Real.sqrt 3 / 3) :=  -- Perimeter of square D is (40√3)/3 cm
by sorry

end square_perimeter_relation_l1453_145351


namespace car_fuel_tank_cost_l1453_145330

/-- Proves that the cost to fill a car fuel tank is $45 given specific conditions -/
theorem car_fuel_tank_cost : ∃ (F : ℚ),
  (2000 / 500 : ℚ) * F + (3/5) * ((2000 / 500 : ℚ) * F) = 288 ∧ F = 45 := by
  sorry

end car_fuel_tank_cost_l1453_145330


namespace power_comparison_l1453_145334

theorem power_comparison : 2^444 = 4^222 ∧ 2^444 < 3^333 := by
  sorry

end power_comparison_l1453_145334


namespace zero_product_implies_zero_factor_l1453_145312

/-- p-arithmetic, where p is prime -/
structure PArithmetic (p : ℕ) :=
  (carrier : Type)
  (add : carrier → carrier → carrier)
  (mul : carrier → carrier → carrier)
  (zero : carrier)
  (isPrime : Nat.Prime p)

/-- Statement: In p-arithmetic, if the product of two numbers is zero, then at least one of the numbers must be zero -/
theorem zero_product_implies_zero_factor {p : ℕ} (parith : PArithmetic p) :
  ∀ (a b : parith.carrier), parith.mul a b = parith.zero → a = parith.zero ∨ b = parith.zero :=
sorry

end zero_product_implies_zero_factor_l1453_145312


namespace area_bounded_region_l1453_145374

/-- The area of a region bounded by horizontal lines, a vertical line, and a semicircle -/
theorem area_bounded_region (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let rectangular_area := (a + b) * d
  let semicircle_area := (1 / 2) * Real.pi * c^2
  rectangular_area + semicircle_area = (a + b) * d + (1 / 2) * Real.pi * c^2 := by
  sorry

end area_bounded_region_l1453_145374


namespace principal_amount_l1453_145314

/-- Proves that given the specified conditions, the principal amount is 2600 --/
theorem principal_amount (rate : ℚ) (time : ℕ) (interest_difference : ℚ) : 
  rate = 4/100 → 
  time = 5 → 
  interest_difference = 2080 → 
  (∃ (principal : ℚ), 
    principal * rate * time = principal - interest_difference ∧ 
    principal = 2600) := by
  sorry

end principal_amount_l1453_145314


namespace magic_8_ball_probability_l1453_145341

/-- The probability of getting exactly k positive answers out of n questions,
    where each question has a p probability of a positive answer. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 3 positive answers out of 7 questions
    from a Magic 8 Ball, where each question has a 3/7 chance of a positive answer. -/
theorem magic_8_ball_probability :
  binomial_probability 7 3 (3/7) = 242520/823543 := by
  sorry

end magic_8_ball_probability_l1453_145341


namespace computer_table_cost_price_l1453_145340

-- Define the markup percentage
def markup : ℚ := 24 / 100

-- Define the selling price
def selling_price : ℚ := 8215

-- Define the cost price calculation function
def cost_price (sp : ℚ) (m : ℚ) : ℚ := sp / (1 + m)

-- Theorem statement
theorem computer_table_cost_price : 
  cost_price selling_price markup = 6625 := by
  sorry

end computer_table_cost_price_l1453_145340


namespace same_color_probability_l1453_145357

/-- The probability of drawing 2 balls of the same color from a bag -/
theorem same_color_probability (total : ℕ) (red : ℕ) (yellow : ℕ) (blue : ℕ) :
  total = red + yellow + blue →
  red = 3 →
  yellow = 2 →
  blue = 1 →
  (Nat.choose red 2 + Nat.choose yellow 2) / Nat.choose total 2 = 4 / 15 := by
  sorry

end same_color_probability_l1453_145357


namespace angle_sine_relation_l1453_145348

open Real

/-- For angles α and β in the first quadrant, "α > β" is neither a sufficient nor a necessary condition for "sin α > sin β". -/
theorem angle_sine_relation (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) :
  (∃ α' β' : ℝ, α' > β' ∧ sin α' = sin β') ∧
  (∃ α'' β'' : ℝ, sin α'' > sin β'' ∧ ¬(α'' > β'')) := by
  sorry


end angle_sine_relation_l1453_145348


namespace min_value_expression_l1453_145398

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 18) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 22 ∧
  ∃ x₀ > 4, (x₀ + 18) / Real.sqrt (x₀ - 4) = 2 * Real.sqrt 22 := by
sorry

end min_value_expression_l1453_145398


namespace average_age_combined_l1453_145300

theorem average_age_combined (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℝ) (avg_age_parents : ℝ) :
  num_students = 33 →
  num_parents = 55 →
  avg_age_students = 11 →
  avg_age_parents = 33 →
  ((num_students : ℝ) * avg_age_students + (num_parents : ℝ) * avg_age_parents) / 
   ((num_students : ℝ) + (num_parents : ℝ)) = 24.75 := by
  sorry

end average_age_combined_l1453_145300


namespace boys_in_class_l1453_145311

theorem boys_in_class (total_students : ℕ) (girls_ratio : ℚ) : 
  total_students = 160 → girls_ratio = 5/8 → total_students - (girls_ratio * total_students).num = 60 := by
  sorry

end boys_in_class_l1453_145311


namespace inscribed_prism_surface_area_l1453_145358

/-- The surface area of a right square prism inscribed in a sphere -/
theorem inscribed_prism_surface_area (r h : ℝ) (a : ℝ) :
  r = Real.sqrt 6 →
  h = 4 →
  2 * a^2 + h^2 = 4 * r^2 →
  2 * a^2 + 4 * a * h = 40 := by
  sorry

end inscribed_prism_surface_area_l1453_145358


namespace min_faces_to_paint_correct_faces_to_paint_less_than_total_l1453_145393

/-- The minimum number of cube faces Vasya needs to paint to prevent Petya from assembling
    an nxnxn cube that is completely white on the outside, given n^3 white 1x1x1 cubes. -/
def min_faces_to_paint (n : ℕ) : ℕ :=
  match n with
  | 2 => 2
  | 3 => 12
  | _ => 0  -- undefined for other values of n

/-- Theorem stating the correct minimum number of faces to paint for n=2 and n=3 -/
theorem min_faces_to_paint_correct :
  (min_faces_to_paint 2 = 2) ∧ (min_faces_to_paint 3 = 12) :=
by sorry

/-- Helper function to calculate the total number of small cubes -/
def total_small_cubes (n : ℕ) : ℕ := n^3

/-- Theorem stating that the number of faces to paint is less than the total number of cube faces -/
theorem faces_to_paint_less_than_total (n : ℕ) :
  n = 2 ∨ n = 3 → min_faces_to_paint n < 6 * total_small_cubes n :=
by sorry

end min_faces_to_paint_correct_faces_to_paint_less_than_total_l1453_145393


namespace equation_solution_l1453_145313

theorem equation_solution (x : ℝ) (h : 9 - 4/x = 7 + 8/x) : x = 6 := by
  sorry

end equation_solution_l1453_145313


namespace tan_value_for_given_point_l1453_145354

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_value_for_given_point (θ : Real) :
  (∃ (t : Real), t > 0 ∧ t * (-Real.sqrt 3 / 2) = Real.cos θ ∧ t * (1 / 2) = Real.sin θ) →
  Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end tan_value_for_given_point_l1453_145354


namespace intersection_A_B_l1453_145365

def A : Set ℝ := {x | (x + 3) * (2 - x) > 0}
def B : Set ℝ := {-5, -4, 0, 1, 4}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end intersection_A_B_l1453_145365


namespace sufficient_not_necessary_condition_l1453_145355

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x < 1 → x < 2) ∧ 
  (∃ x : ℝ, x < 2 ∧ ¬(x < 1)) :=
by sorry

end sufficient_not_necessary_condition_l1453_145355


namespace probability_two_same_pair_l1453_145343

/-- The number of students participating in the events -/
def num_students : ℕ := 3

/-- The number of events available -/
def num_events : ℕ := 3

/-- The number of events each student chooses -/
def events_per_student : ℕ := 2

/-- The total number of possible combinations for all students' choices -/
def total_combinations : ℕ := num_students ^ num_events

/-- The number of ways to choose 2 students out of 3 -/
def ways_to_choose_2_students : ℕ := 3

/-- The number of ways to choose 1 pair of events out of 3 possible pairs -/
def ways_to_choose_event_pair : ℕ := 3

/-- The number of choices for the remaining student -/
def choices_for_remaining_student : ℕ := 2

/-- The number of favorable outcomes (where exactly two students choose the same pair) -/
def favorable_outcomes : ℕ := ways_to_choose_2_students * ways_to_choose_event_pair * choices_for_remaining_student

/-- The probability of exactly two students choosing the same pair of events -/
theorem probability_two_same_pair : 
  (favorable_outcomes : ℚ) / total_combinations = 2 / 3 := by sorry

end probability_two_same_pair_l1453_145343


namespace fraction_problem_l1453_145322

theorem fraction_problem (A B x : ℝ) : 
  A + B = 27 → 
  B = 15 → 
  0.5 * A + x * B = 11 → 
  x = 1/3 := by
sorry

end fraction_problem_l1453_145322


namespace ball_count_after_500_steps_l1453_145339

/-- Converts a natural number to its base 3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 3) :: aux (m / 3)
  aux n

/-- Sums the digits in a list -/
def sumDigits (l : List ℕ) : ℕ :=
  l.sum

theorem ball_count_after_500_steps : sumDigits (toBase3 500) = 6 := by
  sorry

end ball_count_after_500_steps_l1453_145339


namespace f_is_algebraic_fraction_l1453_145368

/-- An algebraic fraction is a ratio of algebraic expressions. -/
def is_algebraic_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (n d : ℚ → ℚ), ∀ x, d x ≠ 0 → f x = (n x) / (d x)

/-- The function f(x) = 2/(x+3) for x ≠ -3 -/
def f (x : ℚ) : ℚ := 2 / (x + 3)

/-- Theorem: f(x) = 2/(x+3) is an algebraic fraction -/
theorem f_is_algebraic_fraction : is_algebraic_fraction f :=
sorry

end f_is_algebraic_fraction_l1453_145368


namespace apartment_occupancy_l1453_145329

/-- Calculates the number of people in an apartment building given specific conditions. -/
def people_in_building (total_floors : ℕ) (apartments_per_floor : ℕ) (people_per_apartment : ℕ) : ℕ :=
  let full_floors := total_floors / 2
  let half_full_floors := total_floors - full_floors
  let full_apartments := full_floors * apartments_per_floor
  let half_full_apartments := half_full_floors * (apartments_per_floor / 2)
  let total_apartments := full_apartments + half_full_apartments
  total_apartments * people_per_apartment

/-- Theorem stating that under given conditions, the number of people in the building is 360. -/
theorem apartment_occupancy : 
  people_in_building 12 10 4 = 360 := by
  sorry


end apartment_occupancy_l1453_145329


namespace complementary_fraction_irreducible_l1453_145372

theorem complementary_fraction_irreducible (a b : ℤ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : Nat.gcd a.natAbs b.natAbs = 1) : 
  Nat.gcd (b - a).natAbs b.natAbs = 1 := by
sorry

end complementary_fraction_irreducible_l1453_145372


namespace rational_function_zero_l1453_145369

-- Define the numerator and denominator of the rational function
def numerator (x : ℝ) : ℝ := x^2 - x - 6
def denominator (x : ℝ) : ℝ := 5*x - 15

-- Define the domain of the function (all real numbers except 3)
def domain (x : ℝ) : Prop := x ≠ 3

-- State the theorem
theorem rational_function_zero (x : ℝ) (h : domain x) : 
  (numerator x) / (denominator x) = 0 ↔ x = -2 :=
sorry

end rational_function_zero_l1453_145369


namespace fraction_calculation_l1453_145301

theorem fraction_calculation : 
  (1 / 4 - 1 / 5) / (1 / 3 - 1 / 6) = 3 / 10 := by sorry

end fraction_calculation_l1453_145301


namespace two_digit_number_difference_l1453_145347

theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 81 → x - y = 9 := by
  sorry

end two_digit_number_difference_l1453_145347


namespace cubic_equation_solution_l1453_145324

theorem cubic_equation_solution :
  ∀ x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 :=
by sorry

end cubic_equation_solution_l1453_145324


namespace exactly_one_line_through_6_5_l1453_145384

/-- Represents a line in the xy-plane with given x and y intercepts -/
structure Line where
  x_intercept : ℝ
  y_intercept : ℝ

/-- Checks if a real number is a positive even integer -/
def is_positive_even (n : ℝ) : Prop :=
  n > 0 ∧ ∃ k : ℤ, n = 2 * k

/-- Checks if a real number is a positive odd integer -/
def is_positive_odd (n : ℝ) : Prop :=
  n > 0 ∧ ∃ k : ℤ, n = 2 * k + 1

/-- Checks if a line passes through the point (6,5) -/
def passes_through_6_5 (l : Line) : Prop :=
  6 / l.x_intercept + 5 / l.y_intercept = 1

/-- The main theorem to be proved -/
theorem exactly_one_line_through_6_5 :
  ∃! l : Line,
    is_positive_even l.x_intercept ∧
    is_positive_odd l.y_intercept ∧
    passes_through_6_5 l :=
  sorry

end exactly_one_line_through_6_5_l1453_145384


namespace soccer_ball_price_is_40_l1453_145349

def soccer_ball_price (total_balls : ℕ) (amount_given : ℕ) (change_received : ℕ) : ℕ :=
  (amount_given - change_received) / total_balls

theorem soccer_ball_price_is_40 :
  soccer_ball_price 2 100 20 = 40 := by
  sorry

end soccer_ball_price_is_40_l1453_145349


namespace diamond_area_is_50_l1453_145331

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Represents the diamond-shaped region in the square -/
structure DiamondRegion where
  square : Square
  pointA : Point
  pointB : Point

/-- The area of the diamond-shaped region in a 10x10 square -/
def diamondArea (d : DiamondRegion) : ℝ :=
  sorry

theorem diamond_area_is_50 (d : DiamondRegion) : 
  d.square.side = 10 →
  d.pointA.x = 5 ∧ d.pointA.y = 10 →
  d.pointB.x = 5 ∧ d.pointB.y = 0 →
  diamondArea d = 50 := by
  sorry

end diamond_area_is_50_l1453_145331


namespace no_solution_arccos_equation_l1453_145319

theorem no_solution_arccos_equation : ¬∃ x : ℝ, Real.arccos (4/5) - Real.arccos (-4/5) = Real.arcsin x := by
  sorry

end no_solution_arccos_equation_l1453_145319


namespace tournament_games_per_pair_l1453_145390

/-- Represents a chess tournament --/
structure ChessTournament where
  num_players : ℕ
  total_games : ℕ
  games_per_pair : ℕ
  h_players : num_players = 19
  h_total_games : total_games = 342
  h_games_formula : total_games = (num_players * (num_players - 1) * games_per_pair) / 2

/-- Theorem stating that in the given tournament, each player plays against each opponent twice --/
theorem tournament_games_per_pair (t : ChessTournament) : t.games_per_pair = 2 := by
  sorry

#check tournament_games_per_pair

end tournament_games_per_pair_l1453_145390


namespace y_percent_of_x_l1453_145350

theorem y_percent_of_x (y x : ℕ+) (h1 : y = (125 : ℕ+)) (h2 : (y : ℝ) = 0.125 * (x : ℝ)) :
  (y : ℝ) / 100 * (x : ℝ) = 1250 := by
  sorry

end y_percent_of_x_l1453_145350


namespace meal_cost_45_dollars_l1453_145335

/-- The cost of a meal consisting of one pizza and three burgers -/
def meal_cost (burger_price : ℝ) : ℝ :=
  let pizza_price := 2 * burger_price
  pizza_price + 3 * burger_price

/-- Theorem: The cost of one pizza and three burgers is $45 when a burger costs $9 -/
theorem meal_cost_45_dollars :
  meal_cost 9 = 45 := by
  sorry

end meal_cost_45_dollars_l1453_145335


namespace fraction_simplification_l1453_145382

theorem fraction_simplification : (144 : ℚ) / 1296 = 1 / 9 := by
  sorry

end fraction_simplification_l1453_145382


namespace consecutive_sets_sum_150_l1453_145352

/-- A structure representing a set of consecutive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  sum_is_150 : (length * (2 * start + length - 1)) / 2 = 150
  length_ge_2 : length ≥ 2

/-- The theorem stating that there are exactly 5 sets of consecutive integers summing to 150 -/
theorem consecutive_sets_sum_150 :
  (∃ (sets : Finset ConsecutiveSet), sets.card = 5 ∧
    (∀ s : ConsecutiveSet, s ∈ sets ↔ 
      (s.length * (2 * s.start + s.length - 1)) / 2 = 150 ∧ 
      s.length ≥ 2)) :=
sorry

end consecutive_sets_sum_150_l1453_145352


namespace pentagon_rectangle_ratio_l1453_145342

/-- The ratio of the side length of a regular pentagon to the width of a rectangle is 6/5, 
    given that both shapes have a perimeter of 60 inches and the rectangle's length is twice its width. -/
theorem pentagon_rectangle_ratio : 
  ∀ (pentagon_side rectangle_width rectangle_length : ℝ),
  pentagon_side * 5 = 60 →
  rectangle_width * 2 + rectangle_length * 2 = 60 →
  rectangle_length = 2 * rectangle_width →
  pentagon_side / rectangle_width = 6 / 5 := by
sorry

end pentagon_rectangle_ratio_l1453_145342


namespace solution_set_inequality_l1453_145317

theorem solution_set_inequality (x : ℝ) : 
  (x + 2) * (x - 1) > 0 ↔ x < -2 ∨ x > 1 := by
sorry

end solution_set_inequality_l1453_145317


namespace ratio_a_to_b_l1453_145321

theorem ratio_a_to_b (a b c : ℝ) (h1 : c = 0.2 * a) (h2 : c = 0.1 * b) : a / b = 1 / 2 := by
  sorry

end ratio_a_to_b_l1453_145321


namespace inverse_of_3_mod_37_l1453_145323

theorem inverse_of_3_mod_37 : ∃ x : ℕ, x ≤ 36 ∧ (3 * x) % 37 = 1 :=
by
  use 25
  sorry

end inverse_of_3_mod_37_l1453_145323


namespace time_difference_l1453_145332

def brian_time : ℕ := 96
def todd_time : ℕ := 88

theorem time_difference : brian_time - todd_time = 8 := by
  sorry

end time_difference_l1453_145332


namespace sandys_pumpkins_l1453_145367

/-- Sandy and Mike grew pumpkins. This theorem proves how many pumpkins Sandy grew. -/
theorem sandys_pumpkins (mike_pumpkins total_pumpkins : ℕ) 
  (h1 : mike_pumpkins = 23)
  (h2 : mike_pumpkins + sandy_pumpkins = total_pumpkins)
  (h3 : total_pumpkins = 74) :
  sandy_pumpkins = 51 :=
by
  sorry

end sandys_pumpkins_l1453_145367


namespace princes_wish_fulfilled_l1453_145320

/-- Represents a knight at the round table -/
structure Knight where
  city : Nat
  hasGoldGoblet : Bool

/-- The state of the round table at any given moment -/
def RoundTable := Vector Knight 13

/-- Checks if two knights from the same city both have gold goblets -/
def sameCity2GoldGoblets (table : RoundTable) : Bool :=
  sorry

/-- Passes goblets to the right -/
def passGoblets (table : RoundTable) : RoundTable :=
  sorry

/-- The main theorem to be proved -/
theorem princes_wish_fulfilled (k : Nat) (h1 : 1 < k) (h2 : k < 13)
  (initial_table : RoundTable)
  (h3 : (initial_table.toList.filter Knight.hasGoldGoblet).length = k)
  (h4 : (initial_table.toList.map Knight.city).toFinset.card = k) :
  ∃ n : Nat, sameCity2GoldGoblets (n.iterate passGoblets initial_table) := by
  sorry

end princes_wish_fulfilled_l1453_145320


namespace crate_height_difference_l1453_145366

/-- The height difference between two crate packing methods for cylindrical pipes -/
theorem crate_height_difference (n : ℕ) (d : ℝ) :
  let h_direct := n * d
  let h_staggered := (n / 2) * (d + d * Real.sqrt 3 / 2)
  n = 200 ∧ d = 12 →
  h_direct - h_staggered = 120 - 60 * Real.sqrt 3 := by
sorry

end crate_height_difference_l1453_145366


namespace david_total_cost_l1453_145388

/-- Calculates the total cost of a cell phone plan given usage and plan details -/
def calculateTotalCost (baseCost monthlyTexts monthlyHours monthlyData : ℕ)
                       (extraTextCost extraMinuteCost extraGBCost : ℚ)
                       (usedTexts usedHours usedData : ℕ) : ℚ :=
  let extraTexts := max (usedTexts - monthlyTexts) 0
  let extraMinutes := max (usedHours * 60 - monthlyHours * 60) 0
  let extraData := max (usedData - monthlyData) 0
  baseCost + extraTextCost * extraTexts + extraMinuteCost * extraMinutes + extraGBCost * extraData

/-- Theorem stating that David's total cost is $54.50 -/
theorem david_total_cost :
  calculateTotalCost 25 200 40 3 (3/100) (15/100) 10 250 42 4 = 54.5 := by
  sorry

end david_total_cost_l1453_145388


namespace storage_unit_blocks_l1453_145307

/-- Represents the dimensions of the storage unit --/
def storage_unit_side : ℕ := 8

/-- Represents the thickness of the walls, floor, and ceiling --/
def wall_thickness : ℕ := 1

/-- Calculates the number of blocks required for the storage unit construction --/
def blocks_required : ℕ :=
  storage_unit_side ^ 3 - (storage_unit_side - 2 * wall_thickness) ^ 3

/-- Theorem stating that 296 blocks are required for the storage unit construction --/
theorem storage_unit_blocks : blocks_required = 296 := by
  sorry

end storage_unit_blocks_l1453_145307


namespace power_set_of_A_l1453_145376

-- Define set A
def A : Set ℕ := {1, 2}

-- Define set B
def B : Set (Set ℕ) := {x | x ⊆ A}

-- Theorem statement
theorem power_set_of_A : B = {∅, {1}, {2}, {1, 2}} := by
  sorry

end power_set_of_A_l1453_145376


namespace c_nonzero_l1453_145375

def Q (a b c d e : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

theorem c_nonzero (a b c d e : ℝ) :
  (∀ x : ℝ, x = 0 ∨ x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2 → Q a b c d e x = 0) →
  c ≠ 0 := by sorry

end c_nonzero_l1453_145375


namespace line_through_point_l1453_145346

/-- Given a line represented by the equation 3kx - k = -4y - 2 that contains the point (2, 1),
    prove that k = -6/5 -/
theorem line_through_point (k : ℚ) :
  (3 * k * 2 - k = -4 * 1 - 2) → k = -6/5 := by
  sorry

end line_through_point_l1453_145346


namespace log_inequality_l1453_145364

theorem log_inequality (x y z : ℝ) 
  (hx : x = 6 * (Real.log 3 / Real.log 64))
  (hy : y = (1/3) * (Real.log 64 / Real.log 3))
  (hz : z = (3/2) * (Real.log 3 / Real.log 8)) :
  x > y ∧ y > z := by sorry

end log_inequality_l1453_145364


namespace parallel_vectors_x_value_l1453_145345

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, if they are parallel, then x = 2 or x = -1 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2, x)
  let b : ℝ × ℝ := (x - 1, 1)
  parallel a b → x = 2 ∨ x = -1 := by
  sorry

end parallel_vectors_x_value_l1453_145345


namespace pages_per_notepad_l1453_145391

/-- Proves that given the total cost, cost per notepad, and total pages,
    the number of pages per notepad can be determined. -/
theorem pages_per_notepad
  (total_cost : ℝ)
  (cost_per_notepad : ℝ)
  (total_pages : ℕ)
  (h1 : total_cost = 10)
  (h2 : cost_per_notepad = 1.25)
  (h3 : total_pages = 480) :
  (total_pages : ℝ) / (total_cost / cost_per_notepad) = 60 :=
by
  sorry


end pages_per_notepad_l1453_145391


namespace polar_to_cartesian_and_intersection_l1453_145394

-- Define the polar equations
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - 2 * Real.pi / 3) = -Real.sqrt 3

def circle_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ + 2 * Real.sin θ

-- Define the standard equations
def standard_line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 2 * Real.sqrt 3

def standard_circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Define the theorem
theorem polar_to_cartesian_and_intersection :
  (∀ ρ θ : ℝ, line_l ρ θ ↔ ∃ x y : ℝ, standard_line_l x y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∀ ρ θ : ℝ, circle_C ρ θ ↔ ∃ x y : ℝ, standard_circle_C x y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∃ A B : ℝ × ℝ, 
    standard_line_l A.1 A.2 ∧ standard_circle_C A.1 A.2 ∧
    standard_line_l B.1 B.2 ∧ standard_circle_C B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 19) :=
sorry

end polar_to_cartesian_and_intersection_l1453_145394


namespace absolute_value_five_l1453_145387

theorem absolute_value_five (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by sorry

end absolute_value_five_l1453_145387


namespace ladder_problem_l1453_145303

theorem ladder_problem (ladder_length height base : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12)
  (h3 : ladder_length ^ 2 = height ^ 2 + base ^ 2) : 
  base = 5 := by
sorry

end ladder_problem_l1453_145303


namespace jayden_coins_l1453_145356

theorem jayden_coins (jason_coins jayden_coins total_coins : ℕ) 
  (h1 : jason_coins = jayden_coins + 60)
  (h2 : jason_coins + jayden_coins = total_coins)
  (h3 : total_coins = 660) : 
  jayden_coins = 300 := by
sorry

end jayden_coins_l1453_145356


namespace five_divided_triangle_has_48_triangles_l1453_145373

/-- Represents an equilateral triangle with sides divided into n equal parts -/
structure DividedEquilateralTriangle where
  n : ℕ
  n_pos : 0 < n

/-- Counts the number of distinct equilateral triangles in a divided equilateral triangle -/
def count_distinct_triangles (t : DividedEquilateralTriangle) : ℕ :=
  sorry

/-- Theorem stating that a 5-divided equilateral triangle contains 48 distinct equilateral triangles -/
theorem five_divided_triangle_has_48_triangles :
  ∀ (t : DividedEquilateralTriangle), t.n = 5 → count_distinct_triangles t = 48 :=
by sorry

end five_divided_triangle_has_48_triangles_l1453_145373


namespace stating_sandy_comic_books_l1453_145392

/-- 
Given a person with an initial number of comic books, who sells half of them and then buys more,
this function calculates the final number of comic books.
-/
def final_comic_books (initial : ℕ) (bought : ℕ) : ℕ :=
  initial / 2 + bought

/-- 
Theorem stating that if Sandy starts with 14 comic books, sells half, and buys 6 more,
she will end up with 13 comic books.
-/
theorem sandy_comic_books : final_comic_books 14 6 = 13 := by
  sorry

end stating_sandy_comic_books_l1453_145392


namespace simplify_expression_1_l1453_145397

theorem simplify_expression_1 (a b : ℝ) : a * (a - b) - (a + b) * (a - 2 * b) = 2 * b^2 := by
  sorry

end simplify_expression_1_l1453_145397


namespace parallel_vectors_k_value_l1453_145302

theorem parallel_vectors_k_value (k : ℝ) : 
  let a : Fin 2 → ℝ := ![3*k + 1, 2]
  let b : Fin 2 → ℝ := ![k, 1]
  (∃ (c : ℝ), a = c • b) → k = -1 := by
sorry

end parallel_vectors_k_value_l1453_145302


namespace y_not_between_l1453_145360

theorem y_not_between (a b x y : ℝ) (ha : a > 0) (hb : b > 0) 
  (hy : y = (a * Real.sin x + b) / (a * Real.sin x - b)) :
  y ≥ (a - b) / (a + b) ∨ y ≤ (a + b) / (a - b) := by
  sorry

end y_not_between_l1453_145360


namespace expression_evaluation_l1453_145362

theorem expression_evaluation : 1 * 2 + 3 * 4 + 5 * 6 + 7 * 8 + 9 * 10 = 190 := by
  sorry

end expression_evaluation_l1453_145362


namespace probability_4_vertices_in_same_plane_l1453_145326

-- Define a cube type
def Cube := Unit

-- Define a function to represent the number of vertices in a cube
def num_vertices (c : Cube) : ℕ := 8

-- Define a function to represent the number of ways to select 4 vertices from 8
def ways_to_select_4_from_8 (c : Cube) : ℕ := 70

-- Define a function to represent the number of ways 4 vertices can lie in the same plane
def ways_4_vertices_in_same_plane (c : Cube) : ℕ := 12

-- Theorem statement
theorem probability_4_vertices_in_same_plane (c : Cube) :
  (ways_4_vertices_in_same_plane c : ℚ) / (ways_to_select_4_from_8 c : ℚ) = 6 / 35 := by
  sorry

end probability_4_vertices_in_same_plane_l1453_145326


namespace complex_subtraction_l1453_145337

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + 4*I) :
  a - 3*b = -1 - 15*I :=
by sorry

end complex_subtraction_l1453_145337


namespace solution_to_system_of_equations_l1453_145338

theorem solution_to_system_of_equations :
  ∃ (x y : ℝ), 3 * x - 4 * y = -7 ∧ 6 * x - 5 * y = 5 :=
by
  use 7, 7
  sorry

end solution_to_system_of_equations_l1453_145338


namespace inverse_proportion_point_order_l1453_145383

/-- Prove that for points on an inverse proportion function, their y-coordinates follow a specific order -/
theorem inverse_proportion_point_order (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h_pos : k > 0)
  (h_A : y₁ = k / (-1))
  (h_B : y₂ = k / 2)
  (h_C : y₃ = k / 3) :
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end inverse_proportion_point_order_l1453_145383


namespace donna_dog_walking_rate_l1453_145333

def dog_walking_hours : ℕ := 2 * 7
def card_shop_earnings : ℚ := 2 * 5 * 12.5
def babysitting_earnings : ℚ := 4 * 10
def total_earnings : ℚ := 305

theorem donna_dog_walking_rate : 
  ∃ (rate : ℚ), rate * dog_walking_hours + card_shop_earnings + babysitting_earnings = total_earnings ∧ rate = 10 := by
sorry

end donna_dog_walking_rate_l1453_145333


namespace special_power_function_unique_m_l1453_145379

/-- A power function with exponent (m^2 - 2m - 3) that has no intersection with axes and is symmetric about the origin -/
def special_power_function (m : ℕ+) : ℝ → ℝ := fun x ↦ x ^ (m.val ^ 2 - 2 * m.val - 3)

/-- The function has no intersection with x-axis and y-axis -/
def no_axis_intersection (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≠ 0) ∧ (f 0 ≠ 0)

/-- The function is symmetric about the origin -/
def origin_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Main theorem: If the special power function satisfies the conditions, then m = 2 -/
theorem special_power_function_unique_m (m : ℕ+) :
  no_axis_intersection (special_power_function m) ∧
  origin_symmetry (special_power_function m) →
  m = 2 := by
  sorry

end special_power_function_unique_m_l1453_145379


namespace least_prime_factor_of_11_pow_5_minus_11_pow_2_l1453_145371

theorem least_prime_factor_of_11_pow_5_minus_11_pow_2 :
  Nat.minFac (11^5 - 11^2) = 2 := by
sorry

end least_prime_factor_of_11_pow_5_minus_11_pow_2_l1453_145371


namespace certain_multiple_remainder_l1453_145305

theorem certain_multiple_remainder (m : ℤ) (h : m % 5 = 2) :
  (∃ k : ℕ+, k * m % 5 = 1) ∧ (∀ k : ℕ+, k * m % 5 = 1 → k ≥ 3) :=
sorry

end certain_multiple_remainder_l1453_145305


namespace group_trip_cost_l1453_145328

/-- The total cost for a group trip given the number of people and cost per person -/
def total_cost (num_people : ℕ) (cost_per_person : ℕ) : ℕ :=
  num_people * cost_per_person

/-- Proof that the total cost for 11 people at $1100 each is $12100 -/
theorem group_trip_cost : total_cost 11 1100 = 12100 := by
  sorry

end group_trip_cost_l1453_145328


namespace arrangement_count_l1453_145318

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of teachers per group -/
def teachers_per_group : ℕ := 1

/-- The number of students per group -/
def students_per_group : ℕ := 2

/-- The total number of arrangements -/
def total_arrangements : ℕ := choose num_teachers teachers_per_group * choose num_students students_per_group

theorem arrangement_count :
  total_arrangements = 12 :=
sorry

end arrangement_count_l1453_145318


namespace fibonacci_like_sequence_l1453_145336

def is_fibonacci_like (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) = b (n + 1) + b n

theorem fibonacci_like_sequence (b : ℕ → ℕ) :
  is_fibonacci_like b →
  (∀ n m : ℕ, n < m → b n < b m) →
  b 6 = 96 →
  b 7 = 184 := by
sorry

end fibonacci_like_sequence_l1453_145336


namespace cylinder_properties_l1453_145316

/-- Properties of a cylinder with height 15 and radius 5 -/
theorem cylinder_properties :
  ∀ (h r : ℝ),
  h = 15 →
  r = 5 →
  (2 * π * r^2 + 2 * π * r * h = 200 * π) ∧
  (π * r^2 * h = 375 * π) := by
  sorry

end cylinder_properties_l1453_145316


namespace parabola_b_value_l1453_145370

/-- A parabola passing through two points -/
def Parabola (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

theorem parabola_b_value :
  ∀ b c : ℝ,
  Parabola b c 1 = 2 →
  Parabola b c 5 = 2 →
  b = -6 := by
sorry

end parabola_b_value_l1453_145370


namespace inequality_proof_l1453_145389

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2*(a-1)*(b-1) ≥ 1 := by
  sorry

end inequality_proof_l1453_145389


namespace gummy_bear_production_l1453_145308

/-- The number of gummy bears in each packet -/
def bears_per_packet : ℕ := 50

/-- The number of packets filled in 40 minutes -/
def packets_filled : ℕ := 240

/-- The time taken to fill the packets (in minutes) -/
def time_taken : ℕ := 40

/-- The number of gummy bears manufactured per minute -/
def bears_per_minute : ℕ := packets_filled * bears_per_packet / time_taken

theorem gummy_bear_production :
  bears_per_minute = 300 := by
  sorry

end gummy_bear_production_l1453_145308


namespace monicas_savings_l1453_145381

theorem monicas_savings (weekly_savings : ℕ) (weeks_to_fill : ℕ) (repetitions : ℕ) : 
  weekly_savings = 15 → weeks_to_fill = 60 → repetitions = 5 →
  weekly_savings * weeks_to_fill * repetitions = 4500 := by
  sorry

end monicas_savings_l1453_145381


namespace total_cost_l1453_145386

/-- The cost of a bottle of soda -/
def soda_cost : ℚ := sorry

/-- The cost of a bottle of mineral water -/
def mineral_cost : ℚ := sorry

/-- First condition: 2 bottles of soda and 1 bottle of mineral water cost 7 yuan -/
axiom condition1 : 2 * soda_cost + mineral_cost = 7

/-- Second condition: 4 bottles of soda and 3 bottles of mineral water cost 16 yuan -/
axiom condition2 : 4 * soda_cost + 3 * mineral_cost = 16

/-- Theorem: The cost of 10 bottles of soda and 10 bottles of mineral water is 45 yuan -/
theorem total_cost : 10 * soda_cost + 10 * mineral_cost = 45 := by
  sorry

end total_cost_l1453_145386


namespace hexagon_with_90_degree_angle_l1453_145327

/-- A hexagon with angles in geometric progression has an angle of 90 degrees. -/
theorem hexagon_with_90_degree_angle :
  ∃ (a r : ℝ), 
    a > 0 ∧ r > 0 ∧
    a + a*r + a*r^2 + a*r^3 + a*r^4 + a*r^5 = 720 ∧
    (a = 90 ∨ a*r = 90 ∨ a*r^2 = 90 ∨ a*r^3 = 90 ∨ a*r^4 = 90 ∨ a*r^5 = 90) :=
by sorry

end hexagon_with_90_degree_angle_l1453_145327


namespace product_mod_seventeen_l1453_145344

theorem product_mod_seventeen : (2022 * 2023 * 2024 * 2025) % 17 = 0 := by
  sorry

end product_mod_seventeen_l1453_145344


namespace angle_AMB_largest_l1453_145361

/-- Given a right angle XOY with OA = a and OB = b (a < b) on side OY, 
    and a point M on OX such that OM = x, 
    prove that the angle AMB is largest when x = √(ab) -/
theorem angle_AMB_largest (a b x : ℝ) (h_ab : 0 < a ∧ a < b) :
  let φ := Real.arctan ((b - a) * x / (x^2 + a * b))
  ∀ y : ℝ, y > 0 → φ ≤ Real.arctan ((b - a) * y / (y^2 + a * b)) →
  x = Real.sqrt (a * b) := by
  sorry

end angle_AMB_largest_l1453_145361


namespace chef_cakes_problem_l1453_145378

def chef_cakes (total_eggs : ℕ) (fridge_eggs : ℕ) (eggs_per_cake : ℕ) : ℕ :=
  (total_eggs - fridge_eggs) / eggs_per_cake

theorem chef_cakes_problem :
  chef_cakes 60 10 5 = 10 := by
  sorry

end chef_cakes_problem_l1453_145378


namespace problem_solution_l1453_145353

theorem problem_solution : (12346 * 24689 * 37033 + 12347 * 37034) / 12345^2 = 74072 := by
  sorry

end problem_solution_l1453_145353


namespace rotate_angle_result_l1453_145377

/-- Given an initial angle of 30 degrees and a 450-degree counterclockwise rotation,
    the resulting acute angle measures 60 degrees. -/
theorem rotate_angle_result (initial_angle rotation : ℝ) (h1 : initial_angle = 30)
    (h2 : rotation = 450) : 
    (initial_angle + rotation) % 360 = 60 ∨ 360 - (initial_angle + rotation) % 360 = 60 :=
by sorry

end rotate_angle_result_l1453_145377


namespace correct_num_recipes_l1453_145395

/-- The number of recipes to be made for a chocolate chip cookie bake sale. -/
def num_recipes : ℕ := 23

/-- The number of cups of chocolate chips required for one recipe. -/
def cups_per_recipe : ℕ := 2

/-- The total number of cups of chocolate chips needed for all recipes. -/
def total_cups_needed : ℕ := 46

/-- Theorem stating that the number of recipes is correct given the conditions. -/
theorem correct_num_recipes : 
  num_recipes * cups_per_recipe = total_cups_needed :=
by sorry

end correct_num_recipes_l1453_145395


namespace champagne_glasses_per_guest_l1453_145315

/-- Calculates the number of champagne glasses per guest at Ashley's wedding. -/
theorem champagne_glasses_per_guest :
  let num_guests : ℕ := 120
  let servings_per_bottle : ℕ := 6
  let num_bottles : ℕ := 40
  let total_servings : ℕ := num_bottles * servings_per_bottle
  let glasses_per_guest : ℕ := total_servings / num_guests
  glasses_per_guest = 2 := by
  sorry

end champagne_glasses_per_guest_l1453_145315


namespace town_population_problem_l1453_145396

theorem town_population_problem (original_population : ℕ) : 
  (original_population + 1200 : ℕ) * 89 / 100 = original_population - 32 →
  original_population = 10000 :=
by
  sorry

end town_population_problem_l1453_145396


namespace work_completion_time_l1453_145363

/-- Given that:
  - A can do a work in 4 days
  - A and B together can finish the work in 3 days
  Prove that B can do the work alone in 12 days -/
theorem work_completion_time (a_time b_time combined_time : ℝ) 
  (ha : a_time = 4)
  (hc : combined_time = 3)
  (h_combined : 1 / a_time + 1 / b_time = 1 / combined_time) :
  b_time = 12 := by sorry

end work_completion_time_l1453_145363
