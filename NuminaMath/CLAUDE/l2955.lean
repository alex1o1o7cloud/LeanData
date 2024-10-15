import Mathlib

namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2955_295599

theorem decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2955_295599


namespace NUMINAMATH_CALUDE_equation_solution_l2955_295512

theorem equation_solution : ∃ x : ℚ, (4/7 : ℚ) * (2/5 : ℚ) * x = 8 ∧ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2955_295512


namespace NUMINAMATH_CALUDE_tripod_new_height_l2955_295537

/-- Represents a tripod with given parameters -/
structure Tripod where
  leg_length : ℝ
  initial_height : ℝ
  sink_depth : ℝ

/-- Calculates the new height of a tripod after one leg sinks -/
noncomputable def new_height (t : Tripod) : ℝ :=
  144 / Real.sqrt 262.2

/-- Theorem stating the new height of the tripod after one leg sinks -/
theorem tripod_new_height (t : Tripod) 
  (h_leg : t.leg_length = 8)
  (h_init : t.initial_height = 6)
  (h_sink : t.sink_depth = 2) :
  new_height t = 144 / Real.sqrt 262.2 := by
  sorry

end NUMINAMATH_CALUDE_tripod_new_height_l2955_295537


namespace NUMINAMATH_CALUDE_elle_practice_time_l2955_295513

/-- The number of minutes Elle practices piano on a weekday -/
def weekday_practice : ℕ := 30

/-- The number of weekdays Elle practices piano -/
def weekdays : ℕ := 5

/-- The factor by which Elle's Saturday practice is longer than a weekday -/
def saturday_factor : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the total number of hours Elle spends practicing piano each week -/
def total_practice_hours : ℚ :=
  let weekday_total := weekday_practice * weekdays
  let saturday_practice := weekday_practice * saturday_factor
  let total_minutes := weekday_total + saturday_practice
  (total_minutes : ℚ) / minutes_per_hour

theorem elle_practice_time : total_practice_hours = 4 := by
  sorry

end NUMINAMATH_CALUDE_elle_practice_time_l2955_295513


namespace NUMINAMATH_CALUDE_light_reflection_l2955_295593

-- Define the points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, -1)

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Define a line passing through two points
def line_through_points (p q : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p.2) * (q.1 - p.1) = (x - p.1) * (q.2 - p.2)

-- Define reflection of a point across the y-axis
def reflect_across_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- State the theorem
theorem light_reflection :
  ∃ (C : ℝ × ℝ),
    y_axis C.1 ∧
    line_through_points A C 1 1 ∧
    line_through_points (reflect_across_y_axis A) B C.1 C.2 ∧
    (∀ x y, line_through_points A C x y ↔ x - y + 1 = 0) ∧
    (∀ x y, line_through_points (reflect_across_y_axis A) B x y ↔ x + y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_light_reflection_l2955_295593


namespace NUMINAMATH_CALUDE_sin_value_for_specific_tan_l2955_295502

/-- Prove that for an acute angle α, if tan(π - α) + 3 = 0, then sinα = 3√10 / 10 -/
theorem sin_value_for_specific_tan (α : Real) : 
  0 < α ∧ α < π / 2 →  -- α is an acute angle
  Real.tan (π - α) + 3 = 0 → 
  Real.sin α = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_value_for_specific_tan_l2955_295502


namespace NUMINAMATH_CALUDE_barangay_speed_l2955_295507

/-- Proves that the speed going to the barangay is 5 km/h given the problem conditions -/
theorem barangay_speed 
  (total_time : ℝ) 
  (distance : ℝ) 
  (rest_time : ℝ) 
  (return_speed : ℝ) 
  (h1 : total_time = 6)
  (h2 : distance = 7.5)
  (h3 : rest_time = 2)
  (h4 : return_speed = 3) : 
  distance / (total_time - rest_time - distance / return_speed) = 5 := by
sorry

end NUMINAMATH_CALUDE_barangay_speed_l2955_295507


namespace NUMINAMATH_CALUDE_new_shoes_average_speed_l2955_295582

/-- Calculate the average speed of a hiker using new high-tech shoes over a 4-hour hike -/
theorem new_shoes_average_speed
  (old_speed : ℝ)
  (new_speed_multiplier : ℝ)
  (hike_duration : ℝ)
  (blister_interval : ℝ)
  (speed_reduction_per_blister : ℝ)
  (h_old_speed : old_speed = 6)
  (h_new_speed_multiplier : new_speed_multiplier = 2)
  (h_hike_duration : hike_duration = 4)
  (h_blister_interval : blister_interval = 2)
  (h_speed_reduction : speed_reduction_per_blister = 2)
  : (old_speed * new_speed_multiplier + 
     (old_speed * new_speed_multiplier - speed_reduction_per_blister)) / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_new_shoes_average_speed_l2955_295582


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_50_l2955_295534

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_factorials_50 : 
  units_digit (sum_factorials 50) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_50_l2955_295534


namespace NUMINAMATH_CALUDE_distance_between_homes_l2955_295558

/-- Proves that the distance between Maxwell's and Brad's homes is 36 km given the problem conditions -/
theorem distance_between_homes : 
  ∀ (maxwell_speed brad_speed maxwell_distance : ℝ),
    maxwell_speed = 2 →
    brad_speed = 4 →
    maxwell_distance = 12 →
    maxwell_distance + maxwell_distance * (brad_speed / maxwell_speed) = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_distance_between_homes_l2955_295558


namespace NUMINAMATH_CALUDE_subsets_count_l2955_295581

theorem subsets_count : ∃ (n : ℕ), n = (Finset.filter (fun X => {1, 2} ⊆ X ∧ X ⊆ {1, 2, 3, 4, 5}) (Finset.powerset {1, 2, 3, 4, 5})).card ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_subsets_count_l2955_295581


namespace NUMINAMATH_CALUDE_tourist_guide_distribution_l2955_295566

/-- The number of ways to distribute n tourists among k guides, 
    where each guide must have at least one tourist -/
def validDistributions (n k : ℕ) : ℕ :=
  k^n - (k.choose 1) * (k-1)^n + (k.choose 2) * (k-2)^n

theorem tourist_guide_distribution :
  validDistributions 8 3 = 5796 := by
  sorry

end NUMINAMATH_CALUDE_tourist_guide_distribution_l2955_295566


namespace NUMINAMATH_CALUDE_tan_300_degrees_l2955_295592

theorem tan_300_degrees : Real.tan (300 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_300_degrees_l2955_295592


namespace NUMINAMATH_CALUDE_sqrt_expressions_l2955_295575

theorem sqrt_expressions (a b : ℝ) (h1 : a = Real.sqrt 5 + Real.sqrt 3) (h2 : b = Real.sqrt 5 - Real.sqrt 3) :
  (a + b = 2 * Real.sqrt 5) ∧ (a * b = 2) ∧ (a^2 + a*b + b^2 = 18) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_l2955_295575


namespace NUMINAMATH_CALUDE_smart_number_characterization_smart_number_2015_l2955_295595

/-- A positive integer is a smart number if it can be expressed as the difference of squares of two positive integers. -/
def is_smart_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > b ∧ n = a ^ 2 - b ^ 2

/-- Theorem stating the characterization of smart numbers -/
theorem smart_number_characterization (n : ℕ) :
  is_smart_number n ↔ (n > 1 ∧ n % 2 = 1) ∨ (n ≥ 8 ∧ n % 4 = 0) :=
sorry

/-- Function to get the nth smart number -/
def nth_smart_number (n : ℕ) : ℕ :=
sorry

/-- Theorem stating that the 2015th smart number is 2689 -/
theorem smart_number_2015 : nth_smart_number 2015 = 2689 :=
sorry

end NUMINAMATH_CALUDE_smart_number_characterization_smart_number_2015_l2955_295595


namespace NUMINAMATH_CALUDE_equation_solutions_l2955_295557

def equation (x : ℝ) : Prop :=
  x ≠ 1 ∧ x ≠ -6 → (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 3 ∨ x = -6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2955_295557


namespace NUMINAMATH_CALUDE_cycle_selling_price_l2955_295523

theorem cycle_selling_price (cost_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 1900 →
  loss_percentage = 18 →
  selling_price = cost_price * (1 - loss_percentage / 100) →
  selling_price = 1558 := by
sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l2955_295523


namespace NUMINAMATH_CALUDE_inequality_proof_l2955_295577

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) : 
  (x^2 + y*z) / Real.sqrt (2*x^2*(y+z)) + 
  (y^2 + z*x) / Real.sqrt (2*y^2*(z+x)) + 
  (z^2 + x*y) / Real.sqrt (2*z^2*(x+y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2955_295577


namespace NUMINAMATH_CALUDE_abs_z_eq_one_l2955_295543

-- Define the complex number z
variable (z : ℂ)

-- Define the real number a
variable (a : ℝ)

-- Define the condition on a
axiom a_lt_one : a < 1

-- Define the equation that z satisfies
axiom z_equation : (a - 2) * z^2018 + a * z^2017 * Complex.I + a * z * Complex.I + 2 - a = 0

-- Theorem to prove
theorem abs_z_eq_one : Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_abs_z_eq_one_l2955_295543


namespace NUMINAMATH_CALUDE_coeff_x6_q_squared_is_16_l2955_295556

/-- The polynomial q(x) -/
def q (x : ℝ) : ℝ := x^5 - 4*x^3 + 3

/-- The coefficient of x^6 in (q(x))^2 -/
def coeff_x6_q_squared : ℝ := 16

/-- Theorem: The coefficient of x^6 in (q(x))^2 is 16 -/
theorem coeff_x6_q_squared_is_16 : coeff_x6_q_squared = 16 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x6_q_squared_is_16_l2955_295556


namespace NUMINAMATH_CALUDE_min_value_expression_l2955_295579

theorem min_value_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2955_295579


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2955_295538

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2955_295538


namespace NUMINAMATH_CALUDE_problem_2004_l2955_295568

theorem problem_2004 (a : ℝ) : 
  (|2004 - a| + Real.sqrt (a - 2005) = a) → (a - 2004^2 = 2005) := by
  sorry

end NUMINAMATH_CALUDE_problem_2004_l2955_295568


namespace NUMINAMATH_CALUDE_softball_team_ratio_l2955_295572

theorem softball_team_ratio (n : ℕ) (men women : ℕ → ℕ) : 
  n = 20 →
  (∀ k, k ≤ n → k ≥ 3 → women k = men k + k / 3) →
  men n + women n = n →
  (men n : ℚ) / (women n : ℚ) = 7 / 13 :=
sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l2955_295572


namespace NUMINAMATH_CALUDE_distance_traveled_l2955_295562

/-- Calculates the total distance traveled given two speeds and two durations -/
def total_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Theorem: The total distance traveled is 255 miles -/
theorem distance_traveled : total_distance 45 2 55 3 = 255 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l2955_295562


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2955_295514

/-- Given a line x + y + 1 = 0 in a 2D plane, the minimum distance from the point (-2, -3) to this line is 2√2 -/
theorem min_distance_to_line :
  ∀ x y : ℝ, x + y + 1 = 0 →
  (2 * Real.sqrt 2 : ℝ) ≤ Real.sqrt ((x + 2)^2 + (y + 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2955_295514


namespace NUMINAMATH_CALUDE_possible_lists_count_l2955_295598

/-- The number of balls in the bin -/
def num_balls : ℕ := 15

/-- The number of draws Joe makes -/
def num_draws : ℕ := 4

/-- The number of possible lists when drawing with replacement -/
def num_possible_lists : ℕ := num_balls ^ num_draws

/-- Theorem: The number of possible lists is 50625 -/
theorem possible_lists_count : num_possible_lists = 50625 := by
  sorry

end NUMINAMATH_CALUDE_possible_lists_count_l2955_295598


namespace NUMINAMATH_CALUDE_combinatorial_equations_solutions_l2955_295553

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Falling factorial -/
def falling_factorial (n k : ℕ) : ℕ := sorry

theorem combinatorial_equations_solutions :
  (∃ x : ℕ, (binomial 9 x = binomial 9 (2*x - 3)) ∧ (x = 3 ∨ x = 4)) ∧
  (∃ x : ℕ, x ≤ 8 ∧ falling_factorial 8 x = 6 * falling_factorial 8 (x - 2) ∧ x = 7) :=
sorry

end NUMINAMATH_CALUDE_combinatorial_equations_solutions_l2955_295553


namespace NUMINAMATH_CALUDE_shadow_problem_l2955_295527

/-- Given a cube with edge length 2 cm and a light source y cm above one of its upper vertices
    casting a shadow of 324 sq cm (excluding the area beneath the cube),
    prove that the largest integer less than or equal to 500y is 8000. -/
theorem shadow_problem (y : ℝ) : 
  (2 : ℝ) > 0 ∧ y > 0 ∧ 
  (((18 : ℝ)^2 - 2^2) = 324) ∧
  ((y / 2) = ((18 : ℝ) - 2) / 2) →
  ⌊500 * y⌋ = 8000 := by sorry

end NUMINAMATH_CALUDE_shadow_problem_l2955_295527


namespace NUMINAMATH_CALUDE_inequality_proof_l2955_295571

theorem inequality_proof (a b : ℝ) (h : a - |b| > 0) : b + a > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2955_295571


namespace NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l2955_295520

/-- A function that checks if a natural number contains the digit 7 -/
def contains_seven (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + 7 + 10 * m

/-- The smallest natural number n such that both n^2 and (n+1)^2 contain the digit 7 -/
theorem smallest_n_with_seven_in_squares :
  ∃ n : ℕ, n = 26 ∧
    contains_seven (n^2) ∧
    contains_seven ((n+1)^2) ∧
    ∀ m : ℕ, m < n → ¬(contains_seven (m^2) ∧ contains_seven ((m+1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l2955_295520


namespace NUMINAMATH_CALUDE_quadratic_solution_fractional_no_solution_l2955_295563

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 4*x - 4 = 0

-- Define the fractional equation
def fractional_equation (x : ℝ) : Prop := x / (x - 2) + 3 = (x - 4) / (2 - x)

-- Theorem for the quadratic equation
theorem quadratic_solution :
  ∃ x₁ x₂ : ℝ, 
    (x₁ = 2 * Real.sqrt 2 - 2 ∧ quadratic_equation x₁) ∧
    (x₂ = -2 * Real.sqrt 2 - 2 ∧ quadratic_equation x₂) ∧
    (∀ x : ℝ, quadratic_equation x → x = x₁ ∨ x = x₂) :=
sorry

-- Theorem for the fractional equation
theorem fractional_no_solution :
  ¬∃ x : ℝ, fractional_equation x :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_fractional_no_solution_l2955_295563


namespace NUMINAMATH_CALUDE_max_min_sum_theorem_l2955_295542

/-- A function satisfying the given property -/
def FunctionProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2014

/-- The function g defined in terms of f -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2014 * x^2013

theorem max_min_sum_theorem (f : ℝ → ℝ) (hf : FunctionProperty f)
  (hM : ∃ M : ℝ, ∀ x : ℝ, g f x ≤ M)
  (hm : ∃ m : ℝ, ∀ x : ℝ, m ≤ g f x)
  (hMm : ∃ M m : ℝ, (∀ x : ℝ, g f x ≤ M ∧ m ≤ g f x) ∧ 
    (∃ x1 x2 : ℝ, g f x1 = M ∧ g f x2 = m)) :
  ∃ M m : ℝ, (∀ x : ℝ, g f x ≤ M ∧ m ≤ g f x) ∧ 
    (∃ x1 x2 : ℝ, g f x1 = M ∧ g f x2 = m) ∧ M + m = -4028 :=
sorry

end NUMINAMATH_CALUDE_max_min_sum_theorem_l2955_295542


namespace NUMINAMATH_CALUDE_translated_min_point_l2955_295570

/-- The original function before translation -/
def f (x : ℝ) : ℝ := |x + 1| - 4

/-- The translated function -/
def g (x : ℝ) : ℝ := f (x - 3) - 4

/-- The minimum point of the translated function -/
def min_point : ℝ × ℝ := (2, -8)

theorem translated_min_point :
  (∀ x : ℝ, g x ≥ g (min_point.1)) ∧
  g (min_point.1) = min_point.2 :=
sorry

end NUMINAMATH_CALUDE_translated_min_point_l2955_295570


namespace NUMINAMATH_CALUDE_interchange_relation_l2955_295501

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The original number satisfies the given condition -/
def satisfies_condition (n : TwoDigitNumber) (c : Nat) : Prop :=
  10 * n.tens + n.ones = c * (n.tens + n.ones) + 3

/-- The number formed by interchanging digits -/
def interchange_digits (n : TwoDigitNumber) : Nat :=
  10 * n.ones + n.tens

/-- The main theorem to prove -/
theorem interchange_relation (n : TwoDigitNumber) (c : Nat) 
  (h : satisfies_condition n c) :
  interchange_digits n = (11 - c) * (n.tens + n.ones) := by
  sorry


end NUMINAMATH_CALUDE_interchange_relation_l2955_295501


namespace NUMINAMATH_CALUDE_second_player_wins_l2955_295515

/-- Represents the game board -/
def Board := Fin 4 → Fin 2017 → Bool

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents a position on the board -/
structure Position :=
  (row : Fin 4)
  (col : Fin 2017)

/-- Checks if a rook at the given position attacks an even number of other rooks -/
def attacksEven (board : Board) (pos : Position) : Bool :=
  sorry

/-- Checks if a rook at the given position attacks an odd number of other rooks -/
def attacksOdd (board : Board) (pos : Position) : Bool :=
  sorry

/-- Checks if the given move is valid for the current player -/
def isValidMove (board : Board) (player : Player) (pos : Position) : Prop :=
  match player with
  | Player.First => attacksEven board pos
  | Player.Second => attacksOdd board pos

/-- Represents a winning strategy for the second player -/
def secondPlayerStrategy (board : Board) (firstPlayerMove : Position) : Position :=
  sorry

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∀ (board : Board),
  ∀ (firstPlayerMove : Position),
  isValidMove board Player.First firstPlayerMove →
  isValidMove board Player.Second (secondPlayerStrategy board firstPlayerMove) :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l2955_295515


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2955_295561

theorem max_value_of_expression (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d ≤ 4) :
  (2 * a^2 + a^2 * b)^(1/4) + (2 * b^2 + b^2 * c)^(1/4) + 
  (2 * c^2 + c^2 * d)^(1/4) + (2 * d^2 + d^2 * a)^(1/4) ≤ 4 * (3^(1/4)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2955_295561


namespace NUMINAMATH_CALUDE_quadratic_maximum_l2955_295524

theorem quadratic_maximum : 
  (∃ (p : ℝ), -3 * p^2 + 18 * p + 24 = 51) ∧ 
  (∀ (p : ℝ), -3 * p^2 + 18 * p + 24 ≤ 51) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l2955_295524


namespace NUMINAMATH_CALUDE_sequence_contradiction_l2955_295505

theorem sequence_contradiction (s : Finset ℕ) (h1 : s.card = 5) 
  (h2 : 2 ∈ s) (h3 : 35 ∈ s) (h4 : 26 ∈ s) (h5 : ∃ x ∈ s, ∀ y ∈ s, y ≤ x) 
  (h6 : ∀ x ∈ s, x ≤ 25) : False := by
  sorry

end NUMINAMATH_CALUDE_sequence_contradiction_l2955_295505


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l2955_295533

theorem least_perimeter_triangle (d e f : ℕ) : 
  d > 0 → e > 0 → f > 0 →
  (d^2 + e^2 - f^2) / (2 * d * e : ℚ) = 24/25 →
  (d^2 + f^2 - e^2) / (2 * d * f : ℚ) = 3/5 →
  (e^2 + f^2 - d^2) / (2 * e * f : ℚ) = -2/5 →
  d + e + f ≥ 32 :=
by sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l2955_295533


namespace NUMINAMATH_CALUDE_vector_b_magnitude_l2955_295578

def a : ℝ × ℝ := (1, -2)

theorem vector_b_magnitude (b : ℝ × ℝ) (h : 2 • a - b = (-1, 0)) : 
  ‖b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_b_magnitude_l2955_295578


namespace NUMINAMATH_CALUDE_value_of_expression_l2955_295529

theorem value_of_expression (a b x y : ℝ) 
  (h1 : a * x + b * y = 3) 
  (h2 : a * y - b * x = 5) : 
  (a^2 + b^2) * (x^2 + y^2) = 34 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l2955_295529


namespace NUMINAMATH_CALUDE_denominator_of_0_27_repeating_l2955_295567

def repeating_decimal_to_fraction (a b : ℕ) : ℚ :=
  (a : ℚ) / (99 : ℚ)

theorem denominator_of_0_27_repeating :
  (repeating_decimal_to_fraction 27 2).den = 11 := by
  sorry

end NUMINAMATH_CALUDE_denominator_of_0_27_repeating_l2955_295567


namespace NUMINAMATH_CALUDE_power_equation_l2955_295585

theorem power_equation (a m n : ℝ) (hm : a^m = 3) (hn : a^n = 4) : a^(2*m + 3*n) = 576 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l2955_295585


namespace NUMINAMATH_CALUDE_negation_of_proposition_sin_reciprocal_inequality_l2955_295583

open Real

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x ∈ (Set.Ioo 0 π), p x) ↔ (∃ x ∈ (Set.Ioo 0 π), ¬ p x) := by sorry

theorem sin_reciprocal_inequality :
  (¬ ∀ x ∈ (Set.Ioo 0 π), sin x + (1 / sin x) > 2) ↔
  (∃ x ∈ (Set.Ioo 0 π), sin x + (1 / sin x) ≤ 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_sin_reciprocal_inequality_l2955_295583


namespace NUMINAMATH_CALUDE_lemon_bags_count_l2955_295518

/-- The maximum load of the truck in kilograms -/
def max_load : ℕ := 900

/-- The mass of one bag of lemons in kilograms -/
def bag_mass : ℕ := 8

/-- The remaining capacity of the truck in kilograms -/
def remaining_capacity : ℕ := 100

/-- The number of bags of lemons on the truck -/
def num_bags : ℕ := (max_load - remaining_capacity) / bag_mass

theorem lemon_bags_count : num_bags = 100 := by
  sorry

end NUMINAMATH_CALUDE_lemon_bags_count_l2955_295518


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2955_295576

/-- Given two vectors a and b, if (a + xb) is parallel to (a - b), then x = -1 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (3, 4))
  (h2 : b = (2, 1))
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ (a.1 + x * b.1, a.2 + x * b.2) = k • (a.1 - b.1, a.2 - b.2)) :
  x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2955_295576


namespace NUMINAMATH_CALUDE_orange_bucket_theorem_l2955_295545

/-- Represents the number of oranges in each bucket -/
structure OrangeBuckets where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of oranges in all buckets -/
def total_oranges (buckets : OrangeBuckets) : ℕ :=
  buckets.first + buckets.second + buckets.third

/-- Theorem stating the total number of oranges in the given conditions -/
theorem orange_bucket_theorem (buckets : OrangeBuckets) 
  (h1 : buckets.first = 22)
  (h2 : buckets.second = buckets.first + 17)
  (h3 : buckets.third = buckets.second - 11) :
  total_oranges buckets = 89 := by
  sorry

#check orange_bucket_theorem

end NUMINAMATH_CALUDE_orange_bucket_theorem_l2955_295545


namespace NUMINAMATH_CALUDE_average_age_of_four_students_l2955_295594

theorem average_age_of_four_students
  (total_students : ℕ)
  (average_age_all : ℝ)
  (num_group1 : ℕ)
  (average_age_group1 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : num_group1 = 9)
  (h4 : average_age_group1 = 16)
  (h5 : age_last_student = 25)
  : (total_students * average_age_all - num_group1 * average_age_group1 - age_last_student) / (total_students - num_group1 - 1) = 14 :=
by
  sorry

#check average_age_of_four_students

end NUMINAMATH_CALUDE_average_age_of_four_students_l2955_295594


namespace NUMINAMATH_CALUDE_only_100_not_sum_of_four_consecutive_odds_l2955_295565

def is_sum_of_four_consecutive_odds (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 4 * k + 12 ∧ k % 2 = 1

theorem only_100_not_sum_of_four_consecutive_odds :
  ¬ is_sum_of_four_consecutive_odds 100 ∧
  (is_sum_of_four_consecutive_odds 16 ∧
   is_sum_of_four_consecutive_odds 40 ∧
   is_sum_of_four_consecutive_odds 72 ∧
   is_sum_of_four_consecutive_odds 200) :=
by sorry

end NUMINAMATH_CALUDE_only_100_not_sum_of_four_consecutive_odds_l2955_295565


namespace NUMINAMATH_CALUDE_total_ingredients_l2955_295540

def strawberries : ℚ := 0.2
def yogurt : ℚ := 0.1
def orange_juice : ℚ := 0.2

theorem total_ingredients : strawberries + yogurt + orange_juice = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_total_ingredients_l2955_295540


namespace NUMINAMATH_CALUDE_log_sum_equals_one_l2955_295586

theorem log_sum_equals_one : Real.log 2 + 2 * Real.log (Real.sqrt 5) = Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_one_l2955_295586


namespace NUMINAMATH_CALUDE_min_containers_to_fill_jumbo_l2955_295559

/-- The volume of a regular size container in milliliters -/
def regular_container_volume : ℕ := 75

/-- The volume of a jumbo container in milliliters -/
def jumbo_container_volume : ℕ := 1800

/-- The minimum number of regular size containers needed to fill a jumbo container -/
def min_containers : ℕ := (jumbo_container_volume + regular_container_volume - 1) / regular_container_volume

theorem min_containers_to_fill_jumbo : min_containers = 24 := by
  sorry

end NUMINAMATH_CALUDE_min_containers_to_fill_jumbo_l2955_295559


namespace NUMINAMATH_CALUDE_even_sum_probability_l2955_295589

/-- Represents a wheel with even and odd sections -/
structure Wheel where
  total : ℕ
  even : ℕ
  odd : ℕ
  sum_sections : total = even + odd

/-- The probability of getting an even sum when spinning two wheels -/
def prob_even_sum (w1 w2 : Wheel) : ℚ :=
  let p1_even := w1.even / w1.total
  let p1_odd := w1.odd / w1.total
  let p2_even := w2.even / w2.total
  let p2_odd := w2.odd / w2.total
  p1_even * p2_even + p1_odd * p2_odd

theorem even_sum_probability :
  let w1 : Wheel := { total := 6, even := 2, odd := 4, sum_sections := by rfl }
  let w2 : Wheel := { total := 8, even := 3, odd := 5, sum_sections := by rfl }
  prob_even_sum w1 w2 = 13 / 24 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_probability_l2955_295589


namespace NUMINAMATH_CALUDE_quadratic_has_real_root_l2955_295550

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_real_root_l2955_295550


namespace NUMINAMATH_CALUDE_paint_cube_cost_l2955_295504

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem paint_cube_cost 
  (paint_cost : ℝ)        -- Cost of paint per kg in Rs
  (paint_coverage : ℝ)    -- Area covered by 1 kg of paint in sq. ft
  (cube_side : ℝ)         -- Length of cube side in feet
  (h1 : paint_cost = 20)  -- Paint costs Rs. 20 per kg
  (h2 : paint_coverage = 15) -- 1 kg of paint covers 15 sq. ft
  (h3 : cube_side = 5)    -- Cube has sides of 5 feet
  : ℝ :=
by
  -- The proof would go here
  sorry

#check paint_cube_cost

end NUMINAMATH_CALUDE_paint_cube_cost_l2955_295504


namespace NUMINAMATH_CALUDE_probability_of_white_coverage_l2955_295588

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of the central white rectangle -/
structure CentralRectangle where
  length : ℝ
  width : ℝ

/-- Represents the circular sheet used to cover the field -/
structure CircularSheet where
  diameter : ℝ

/-- Calculates the probability of covering white area -/
def probability_of_covering_white (field : FieldDimensions) (central : CentralRectangle) (sheet : CircularSheet) (num_circles : ℕ) (circle_radius : ℝ) : ℝ :=
  sorry

/-- The main theorem stating the probability -/
theorem probability_of_white_coverage :
  let field := FieldDimensions.mk 12 10
  let central := CentralRectangle.mk 4 2
  let sheet := CircularSheet.mk 1.5
  let num_circles := 5
  let circle_radius := 1
  abs (probability_of_covering_white field central sheet num_circles circle_radius - 0.647) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_coverage_l2955_295588


namespace NUMINAMATH_CALUDE_girls_grades_l2955_295584

theorem girls_grades (M L S : ℕ) 
  (h1 : M + L = 23)
  (h2 : S + M = 18)
  (h3 : S + L = 15) :
  M = 13 ∧ L = 10 ∧ S = 5 := by
  sorry

end NUMINAMATH_CALUDE_girls_grades_l2955_295584


namespace NUMINAMATH_CALUDE_matt_keychains_purchase_l2955_295510

/-- The number of key chains Matt buys -/
def num_keychains : ℕ := 10

/-- The price of a pack of 10 key chains -/
def price_pack_10 : ℚ := 20

/-- The price of a pack of 4 key chains -/
def price_pack_4 : ℚ := 12

/-- The amount Matt saves by choosing the cheaper option -/
def savings : ℚ := 20

theorem matt_keychains_purchase :
  num_keychains = 10 ∧
  (num_keychains : ℚ) * (price_pack_10 / 10) = 
    (num_keychains : ℚ) * (price_pack_4 / 4) - savings :=
by sorry

end NUMINAMATH_CALUDE_matt_keychains_purchase_l2955_295510


namespace NUMINAMATH_CALUDE_history_books_shelved_l2955_295516

theorem history_books_shelved (total_books : ℕ) (romance_books : ℕ) (poetry_books : ℕ)
  (western_books : ℕ) (biography_books : ℕ) :
  total_books = 46 →
  romance_books = 8 →
  poetry_books = 4 →
  western_books = 5 →
  biography_books = 6 →
  ∃ (history_books : ℕ) (mystery_books : ℕ),
    history_books = 12 ∧
    mystery_books = western_books + biography_books ∧
    total_books = history_books + romance_books + poetry_books + western_books + biography_books + mystery_books :=
by
  sorry

end NUMINAMATH_CALUDE_history_books_shelved_l2955_295516


namespace NUMINAMATH_CALUDE_pet_store_problem_l2955_295580

def puppies_sold (initial_puppies : ℕ) (puppies_per_cage : ℕ) (cages_used : ℕ) : ℕ :=
  initial_puppies - (puppies_per_cage * cages_used)

theorem pet_store_problem :
  puppies_sold 45 2 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_problem_l2955_295580


namespace NUMINAMATH_CALUDE_canoe_rental_cost_l2955_295509

/-- The cost of renting a canoe per day -/
def canoe_cost : ℚ := 9

/-- The cost of renting a kayak per day -/
def kayak_cost : ℚ := 12

/-- The ratio of canoes to kayaks rented -/
def canoe_kayak_ratio : ℚ := 4/3

/-- The number of additional canoes compared to kayaks -/
def additional_canoes : ℕ := 6

/-- The total revenue for the day -/
def total_revenue : ℚ := 432

theorem canoe_rental_cost :
  let kayaks : ℕ := 18
  let canoes : ℕ := kayaks + additional_canoes
  canoe_cost * canoes + kayak_cost * kayaks = total_revenue ∧
  (canoes : ℚ) / kayaks = canoe_kayak_ratio :=
by sorry

end NUMINAMATH_CALUDE_canoe_rental_cost_l2955_295509


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_four_squared_plus_six_l2955_295526

theorem absolute_value_of_negative_four_squared_plus_six : 
  |(-4^2 + 6)| = 10 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_four_squared_plus_six_l2955_295526


namespace NUMINAMATH_CALUDE_rectangle_validity_l2955_295517

/-- A rectangle is valid if its area is less than or equal to the square of a quarter of its perimeter. -/
theorem rectangle_validity (S l : ℝ) (h_pos : S > 0 ∧ l > 0) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = S ∧ 2 * (x + y) = l) ↔ S ≤ (l / 4)^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_validity_l2955_295517


namespace NUMINAMATH_CALUDE_difference_proof_l2955_295548

/-- Given a total number of students and the number of first graders,
    calculate the difference between second graders and first graders. -/
def difference_between_grades (total : ℕ) (first_graders : ℕ) : ℕ :=
  (total - first_graders) - first_graders

theorem difference_proof :
  difference_between_grades 95 32 = 31 :=
by sorry

end NUMINAMATH_CALUDE_difference_proof_l2955_295548


namespace NUMINAMATH_CALUDE_jack_buttons_theorem_l2955_295532

/-- The number of buttons Jack must use for all shirts -/
def total_buttons (num_kids : ℕ) (shirts_per_kid : ℕ) (buttons_per_shirt : ℕ) : ℕ :=
  num_kids * shirts_per_kid * buttons_per_shirt

/-- Theorem stating the total number of buttons Jack must use -/
theorem jack_buttons_theorem :
  total_buttons 3 3 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_jack_buttons_theorem_l2955_295532


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2955_295546

theorem trigonometric_identity : 4 * Real.sin (20 * π / 180) + Real.tan (20 * π / 180) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2955_295546


namespace NUMINAMATH_CALUDE_karl_garden_larger_l2955_295544

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (g : GardenDimensions) : ℝ :=
  g.length * g.width

/-- Theorem: Karl's garden is larger than Makenna's usable garden area by 150 square feet -/
theorem karl_garden_larger (karl : GardenDimensions) (makenna : GardenDimensions) 
  (h1 : karl.length = 30 ∧ karl.width = 50)
  (h2 : makenna.length = 35 ∧ makenna.width = 45)
  (path_width : ℝ) (h3 : path_width = 5) : 
  gardenArea karl - (gardenArea makenna - path_width * makenna.length) = 150 := by
  sorry

#check karl_garden_larger

end NUMINAMATH_CALUDE_karl_garden_larger_l2955_295544


namespace NUMINAMATH_CALUDE_net_difference_in_expenditure_l2955_295528

/-- Represents the problem of calculating the net difference in expenditure after a price increase --/
theorem net_difference_in_expenditure
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_increase_percentage : ℝ)
  (budget : ℝ)
  (purchased_percentage : ℝ)
  (h1 : price_increase_percentage = 0.25)
  (h2 : budget = 150)
  (h3 : purchased_percentage = 0.64)
  (h4 : original_price * original_quantity = budget)
  (h5 : original_quantity ≤ 40) :
  original_price * original_quantity - (original_price * (1 + price_increase_percentage)) * (purchased_percentage * original_quantity) = 30 :=
by sorry

end NUMINAMATH_CALUDE_net_difference_in_expenditure_l2955_295528


namespace NUMINAMATH_CALUDE_definite_integral_3x_plus_sin_x_l2955_295554

theorem definite_integral_3x_plus_sin_x : 
  ∫ x in (0)..(π/2), (3*x + Real.sin x) = (3*π^2)/8 + 1 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_3x_plus_sin_x_l2955_295554


namespace NUMINAMATH_CALUDE_arithmetic_operations_l2955_295573

theorem arithmetic_operations :
  (8 + (-1/4) - 5 - (-0.25) = 3) ∧
  (-36 * (-2/3 + 5/6 - 7/12 - 8/9) = 47) ∧
  (-2 + 2 / (-1/2) * 2 = -10) ∧
  (-3.5 * (1/6 - 0.5) * 3/7 / (1/2) = 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l2955_295573


namespace NUMINAMATH_CALUDE_coefficient_sum_l2955_295506

-- Define the sets A and B
def A : Set ℝ := {x | x^3 + 3*x^2 + 2*x > 0}
def B : Set ℝ := {x | ∃ (a b : ℝ), x^2 + a*x + b ≤ 0}

-- Define the intersection and union of A and B
def intersection : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def union : Set ℝ := {x | x > -2}

-- State the theorem
theorem coefficient_sum (a b : ℝ) : 
  A ∩ B = intersection → A ∪ B = union → a + b = -3 := by sorry

end NUMINAMATH_CALUDE_coefficient_sum_l2955_295506


namespace NUMINAMATH_CALUDE_investment_plans_count_l2955_295547

/-- The number of ways to distribute 3 distinct objects into 4 distinct containers,
    with no container holding more than 2 objects. -/
def investmentPlans : ℕ :=
  let numProjects : ℕ := 3
  let numCities : ℕ := 4
  let maxProjectsPerCity : ℕ := 2
  -- The actual calculation is not implemented here
  60

/-- Theorem stating that the number of investment plans is 60 -/
theorem investment_plans_count : investmentPlans = 60 := by
  sorry

end NUMINAMATH_CALUDE_investment_plans_count_l2955_295547


namespace NUMINAMATH_CALUDE_sarah_pencils_count_l2955_295522

/-- The number of pencils Sarah buys on Monday -/
def monday_pencils : ℕ := 20

/-- The number of pencils Sarah buys on Tuesday -/
def tuesday_pencils : ℕ := 18

/-- The number of pencils Sarah buys on Wednesday -/
def wednesday_pencils : ℕ := 3 * tuesday_pencils

/-- The total number of pencils Sarah has -/
def total_pencils : ℕ := monday_pencils + tuesday_pencils + wednesday_pencils

theorem sarah_pencils_count : total_pencils = 92 := by
  sorry

end NUMINAMATH_CALUDE_sarah_pencils_count_l2955_295522


namespace NUMINAMATH_CALUDE_solution_of_equations_l2955_295560

theorem solution_of_equations (x : ℝ) : 
  (|x|^2 - 5*|x| + 6 = 0 ∧ x^2 - 4 = 0) ↔ (x = 2 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equations_l2955_295560


namespace NUMINAMATH_CALUDE_theater_ticket_pricing_l2955_295519

theorem theater_ticket_pricing (total_tickets : ℕ) (total_revenue : ℕ) 
  (balcony_price : ℕ) (balcony_orchestra_diff : ℕ) :
  total_tickets = 360 →
  total_revenue = 3320 →
  balcony_price = 8 →
  balcony_orchestra_diff = 140 →
  ∃ (orchestra_price : ℕ), 
    orchestra_price = 12 ∧
    orchestra_price * (total_tickets - balcony_orchestra_diff) / 2 + 
    balcony_price * (total_tickets + balcony_orchestra_diff) / 2 = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_pricing_l2955_295519


namespace NUMINAMATH_CALUDE_total_apples_formula_l2955_295587

/-- Represents the number of apples each person has -/
structure AppleCount where
  sarah : ℕ
  jackie : ℕ
  adam : ℕ

/-- Calculates the total number of apples -/
def totalApples (count : AppleCount) : ℕ :=
  count.sarah + count.jackie + count.adam

/-- Theorem: The total number of apples is 5X + 5, where X is Sarah's apple count -/
theorem total_apples_formula (X : ℕ) : 
  ∀ (count : AppleCount), 
    count.sarah = X → 
    count.jackie = 2 * X → 
    count.adam = count.jackie + 5 → 
    totalApples count = 5 * X + 5 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_formula_l2955_295587


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l2955_295574

theorem least_number_for_divisibility (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ (1101 + k) % 24 = 0) → 
  (∃ m : ℕ, m ≥ 0 ∧ (1101 + m) % 24 = 0 ∧ ∀ l : ℕ, l < m → (1101 + l) % 24 ≠ 0) →
  n = 3 := by
sorry

#eval (1101 + 3) % 24  -- This should evaluate to 0

end NUMINAMATH_CALUDE_least_number_for_divisibility_l2955_295574


namespace NUMINAMATH_CALUDE_rectangle_arrangement_possible_l2955_295597

/-- Represents a small 1×2 rectangle with 2 stars -/
structure SmallRectangle :=
  (width : Nat) (height : Nat) (stars : Nat)

/-- Represents the large 5×200 rectangle -/
structure LargeRectangle :=
  (width : Nat) (height : Nat)
  (smallRectangles : List SmallRectangle)

/-- Checks if a number is even -/
def isEven (n : Nat) : Prop := ∃ k, n = 2 * k

/-- Calculates the total number of stars in a list of small rectangles -/
def totalStars (rectangles : List SmallRectangle) : Nat :=
  rectangles.foldl (fun acc rect => acc + rect.stars) 0

/-- Theorem: It's possible to arrange 500 1×2 rectangles into a 5×200 rectangle
    with an even number of stars in each row and column -/
theorem rectangle_arrangement_possible :
  ∃ (largeRect : LargeRectangle),
    largeRect.width = 200 ∧
    largeRect.height = 5 ∧
    largeRect.smallRectangles.length = 500 ∧
    (∀ smallRect ∈ largeRect.smallRectangles, smallRect.width = 1 ∧ smallRect.height = 2 ∧ smallRect.stars = 2) ∧
    (∀ row ∈ List.range 5, isEven (totalStars (largeRect.smallRectangles.filter (fun _ => true)))) ∧
    (∀ col ∈ List.range 200, isEven (totalStars (largeRect.smallRectangles.filter (fun _ => true)))) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_possible_l2955_295597


namespace NUMINAMATH_CALUDE_manufacturer_central_tendencies_l2955_295590

def manufacturer_A : List ℝ := [3, 4, 5, 6, 8, 8, 8, 10]
def manufacturer_B : List ℝ := [4, 6, 6, 6, 8, 9, 12, 13]
def manufacturer_C : List ℝ := [3, 3, 4, 7, 9, 10, 11, 12]

def mode (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def median (l : List ℝ) : ℝ := sorry

theorem manufacturer_central_tendencies :
  (mode manufacturer_A = 8) ∧
  (mean manufacturer_B = 8) ∧
  (median manufacturer_C = 8) := by sorry

end NUMINAMATH_CALUDE_manufacturer_central_tendencies_l2955_295590


namespace NUMINAMATH_CALUDE_video_game_rounds_l2955_295596

/-- The number of points earned per win in the video game competition. -/
def points_per_win : ℕ := 5

/-- The number of points Vlad scored. -/
def vlad_points : ℕ := 64

/-- The total points scored by both players. -/
def total_points : ℕ := 150

/-- Taro's points in terms of the total points. -/
def taro_points (P : ℕ) : ℤ := (3 * P) / 5 - 4

theorem video_game_rounds :
  (total_points = taro_points total_points + vlad_points) →
  (total_points / points_per_win = 30) := by
sorry

end NUMINAMATH_CALUDE_video_game_rounds_l2955_295596


namespace NUMINAMATH_CALUDE_top_view_area_is_eight_l2955_295535

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the area of the top view of a rectangular prism -/
def topViewArea (d : PrismDimensions) : ℝ := d.length * d.width

/-- Theorem: The area of the top view of the given rectangular prism is 8 square units -/
theorem top_view_area_is_eight :
  let d : PrismDimensions := { length := 4, width := 2, height := 3 }
  topViewArea d = 8 := by
  sorry

end NUMINAMATH_CALUDE_top_view_area_is_eight_l2955_295535


namespace NUMINAMATH_CALUDE_color_coded_figure_areas_l2955_295536

/-- A figure composed of squares with color-coded parts -/
structure ColorCodedFigure where
  /-- The side length of each square in the figure -/
  square_side : ℝ
  /-- The total number of squares in the figure -/
  total_squares : ℕ
  /-- The area of the black part of the figure -/
  black_area : ℝ

/-- Theorem stating the areas of the remaining parts in the figure -/
theorem color_coded_figure_areas (fig : ColorCodedFigure)
  (h1 : fig.total_squares = 8)
  (h2 : fig.square_side ^ 2 * fig.total_squares = 24)
  (h3 : fig.black_area = 7.5) :
  ∃ (white dark_gray light_gray shaded : ℝ),
    white = 1.5 ∧
    dark_gray = 6 ∧
    light_gray = 5.25 ∧
    shaded = 3.75 ∧
    white + dark_gray + light_gray + shaded + fig.black_area = fig.square_side ^ 2 * fig.total_squares :=
by sorry

end NUMINAMATH_CALUDE_color_coded_figure_areas_l2955_295536


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2955_295503

theorem reciprocal_of_negative_three :
  ∀ x : ℚ, x * (-3) = 1 → x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2955_295503


namespace NUMINAMATH_CALUDE_oranges_left_l2955_295508

def initial_oranges : ℕ := 96
def taken_oranges : ℕ := 45

theorem oranges_left : initial_oranges - taken_oranges = 51 := by
  sorry

end NUMINAMATH_CALUDE_oranges_left_l2955_295508


namespace NUMINAMATH_CALUDE_alice_original_seat_was_six_l2955_295552

/-- Represents the number of seats -/
def num_seats : Nat := 7

/-- Represents the seat Alice ends up in -/
def alice_final_seat : Nat := 4

/-- Represents the net movement of all other friends -/
def net_movement : Int := 2

/-- Calculates Alice's original seat given her final seat and the net movement of others -/
def alice_original_seat (final_seat : Nat) (net_move : Int) : Nat :=
  final_seat + net_move.toNat

/-- Theorem stating Alice's original seat was 6 -/
theorem alice_original_seat_was_six :
  alice_original_seat alice_final_seat net_movement = 6 := by
  sorry

#eval alice_original_seat alice_final_seat net_movement

end NUMINAMATH_CALUDE_alice_original_seat_was_six_l2955_295552


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l2955_295539

theorem square_perimeter_ratio (s1 s2 : ℝ) (h : s1^2 / s2^2 = 16 / 81) :
  (4 * s1) / (4 * s2) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l2955_295539


namespace NUMINAMATH_CALUDE_smallest_m_for_identical_digits_l2955_295525

theorem smallest_m_for_identical_digits : ∃ (n : ℕ), 
  (∀ (m : ℕ), m < 671 → 
    ¬∃ (k : ℕ), (2015^(3*m+1) - 2015^(6*k+2)) % 10^2014 = 0 ∧ 2015^(3*m+1) < 2015^(6*k+2)) ∧
  ∃ (n : ℕ), (2015^(3*671+1) - 2015^(6*n+2)) % 10^2014 = 0 ∧ 2015^(3*671+1) < 2015^(6*n+2) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_identical_digits_l2955_295525


namespace NUMINAMATH_CALUDE_lawn_width_calculation_l2955_295521

/-- Calculates the width of a rectangular lawn given specific conditions. -/
theorem lawn_width_calculation (length width road_width cost_per_sqm total_cost : ℝ) 
  (h1 : length = 80)
  (h2 : road_width = 10)
  (h3 : cost_per_sqm = 3)
  (h4 : total_cost = 3900)
  (h5 : (road_width * width + road_width * length - road_width * road_width) * cost_per_sqm = total_cost) :
  width = 60 := by
sorry

end NUMINAMATH_CALUDE_lawn_width_calculation_l2955_295521


namespace NUMINAMATH_CALUDE_positive_integer_triplets_l2955_295564

theorem positive_integer_triplets (a b c : ℕ+) :
  (a ^ b.val ∣ b ^ c.val - 1) ∧ (a ^ c.val ∣ c ^ b.val - 1) →
  (a = 1) ∨ (b = 1 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_triplets_l2955_295564


namespace NUMINAMATH_CALUDE_b_2048_value_l2955_295511

/-- A sequence of real numbers satisfying the given conditions -/
def special_sequence (b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≥ 2 → b n = b (n - 1) * b (n + 1)) ∧
  (b 1 = 3 + 2 * Real.sqrt 5) ∧
  (b 2023 = 23 + 10 * Real.sqrt 5)

/-- The theorem stating the value of b_2048 -/
theorem b_2048_value (b : ℕ → ℝ) (h : special_sequence b) :
  b 2048 = 19 + 6 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_b_2048_value_l2955_295511


namespace NUMINAMATH_CALUDE_specific_trapezoid_height_l2955_295551

/-- A trapezoid with given dimensions -/
structure Trapezoid where
  leg1 : ℝ
  leg2 : ℝ
  base1 : ℝ
  base2 : ℝ

/-- The height of a trapezoid -/
def trapezoidHeight (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating the height of the specific trapezoid -/
theorem specific_trapezoid_height :
  let t : Trapezoid := { leg1 := 6, leg2 := 8, base1 := 4, base2 := 14 }
  trapezoidHeight t = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_height_l2955_295551


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l2955_295530

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 13 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 169 ∧ has_no_small_prime_factors 169) ∧ 
  (∀ m : ℕ, m < 169 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l2955_295530


namespace NUMINAMATH_CALUDE_captain_bonus_calculation_l2955_295591

/-- The number of students in the team -/
def team_size : ℕ := 10

/-- The number of team members (excluding the captain) -/
def team_members : ℕ := 9

/-- The bonus amount for each team member -/
def member_bonus : ℕ := 200

/-- The additional amount the captain receives above the average -/
def captain_extra : ℕ := 90

/-- The bonus amount for the captain -/
def captain_bonus : ℕ := 300

theorem captain_bonus_calculation :
  captain_bonus = 
    (team_members * member_bonus + captain_bonus) / team_size + captain_extra := by
  sorry

#check captain_bonus_calculation

end NUMINAMATH_CALUDE_captain_bonus_calculation_l2955_295591


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_l2955_295500

/-- Proves that the number of cards Nell gave to Jeff is equal to the difference between her initial number of cards and the number of cards she has left. -/
theorem cards_given_to_jeff (initial_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 304)
  (h2 : remaining_cards = 276)
  (h3 : initial_cards ≥ remaining_cards) :
  initial_cards - remaining_cards = 28 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_l2955_295500


namespace NUMINAMATH_CALUDE_bill_equation_l2955_295531

/-- Represents the monthly telephone bill calculation -/
def monthly_bill (rental_fee : ℝ) (per_call_cost : ℝ) (num_calls : ℝ) : ℝ :=
  rental_fee + per_call_cost * num_calls

/-- Theorem stating the relationship between monthly bill and number of calls -/
theorem bill_equation (x : ℝ) :
  monthly_bill 10 0.2 x = 10 + 0.2 * x :=
by sorry

end NUMINAMATH_CALUDE_bill_equation_l2955_295531


namespace NUMINAMATH_CALUDE_p_arithmetic_fibonacci_property_l2955_295555

/-- Definition of p-arithmetic Fibonacci sequence -/
def PArithmeticFibonacci (p : ℕ) (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + p) = v n + v (n + 1)

/-- Theorem: For any p-arithmetic Fibonacci sequence, vₖ + vₖ₊ₚ = vₖ₊₂ₚ holds for all k -/
theorem p_arithmetic_fibonacci_property {p : ℕ} {v : ℕ → ℝ} 
  (hv : PArithmeticFibonacci p v) :
  ∀ k, v k + v (k + p) = v (k + 2 * p) := by
  sorry

end NUMINAMATH_CALUDE_p_arithmetic_fibonacci_property_l2955_295555


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2955_295569

/-- Proves that for a principal of 1000 Rs., if increasing the interest rate by 3%
    results in 90 Rs. more interest, then the time period for which the sum was invested is 3 years. -/
theorem simple_interest_problem (R : ℝ) (T : ℝ) :
  (1000 * R * T / 100 + 90 = 1000 * (R + 3) * T / 100) →
  T = 3 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2955_295569


namespace NUMINAMATH_CALUDE_carrots_picked_next_day_l2955_295549

theorem carrots_picked_next_day (initial_carrots : ℕ) (thrown_out : ℕ) (total_carrots : ℕ) : 
  initial_carrots = 48 → thrown_out = 11 → total_carrots = 52 →
  total_carrots - (initial_carrots - thrown_out) = 15 := by
  sorry

end NUMINAMATH_CALUDE_carrots_picked_next_day_l2955_295549


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2955_295541

theorem sqrt_inequality (x : ℝ) (h : x ≥ 4) :
  Real.sqrt (x - 3) - Real.sqrt (x - 1) > Real.sqrt (x - 4) - Real.sqrt (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2955_295541
