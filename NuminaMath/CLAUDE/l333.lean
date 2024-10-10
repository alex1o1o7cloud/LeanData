import Mathlib

namespace girls_walking_time_l333_33302

/-- The time taken for two girls walking in opposite directions to be 120 km apart -/
theorem girls_walking_time (speed1 speed2 distance : ℝ) (h1 : speed1 = 7)
  (h2 : speed2 = 3) (h3 : distance = 120) : 
  distance / (speed1 + speed2) = 12 := by
  sorry

end girls_walking_time_l333_33302


namespace initial_puppies_count_l333_33318

/-- The number of puppies Alyssa had initially --/
def initial_puppies : ℕ := 12

/-- The number of puppies Alyssa gave away --/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has left --/
def puppies_left : ℕ := 5

/-- Theorem stating that the initial number of puppies is equal to
    the sum of puppies given away and puppies left --/
theorem initial_puppies_count :
  initial_puppies = puppies_given_away + puppies_left := by
  sorry

end initial_puppies_count_l333_33318


namespace min_total_distance_l333_33374

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 13 ∧ dist B C = 14 ∧ dist C A = 15

-- Define the total distance function
def TotalDistance (A B C P : ℝ × ℝ) : ℝ :=
  dist A P + 5 * dist B P + 4 * dist C P

-- State the theorem
theorem min_total_distance (A B C : ℝ × ℝ) (h : Triangle A B C) :
  ∀ P : ℝ × ℝ, TotalDistance A B C P ≥ 69 ∧
  (TotalDistance A B C B = 69) :=
by sorry

end min_total_distance_l333_33374


namespace camp_kids_count_l333_33371

theorem camp_kids_count (total : ℕ) (soccer : ℕ) (morning : ℕ) (afternoon : ℕ) :
  soccer = total / 2 →
  morning = soccer / 4 →
  afternoon = 750 →
  afternoon = soccer * 3 / 4 →
  total = 2000 := by
sorry

end camp_kids_count_l333_33371


namespace parallelogram_exclusive_properties_l333_33380

structure Parallelogram where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  diagonals : Fin 2 → ℝ
  vertex_midpoint_segments : Fin 4 → ℝ
  has_symmetry_axes : Bool
  is_circumscribable : Bool
  is_inscribable : Bool

def all_sides_equal (p : Parallelogram) : Prop :=
  ∀ i j : Fin 4, p.sides i = p.sides j

def all_angles_equal (p : Parallelogram) : Prop :=
  ∀ i j : Fin 4, p.angles i = p.angles j

def all_diagonals_equal (p : Parallelogram) : Prop :=
  p.diagonals 0 = p.diagonals 1

def all_vertex_midpoint_segments_equal (p : Parallelogram) : Prop :=
  ∀ i j : Fin 4, p.vertex_midpoint_segments i = p.vertex_midpoint_segments j

def vertex_midpoint_segments_perpendicular (p : Parallelogram) : Prop :=
  sorry -- This would require more complex geometry definitions

def vertex_midpoint_segments_intersect (p : Parallelogram) : Prop :=
  sorry -- This would require more complex geometry definitions

theorem parallelogram_exclusive_properties (p : Parallelogram) : 
  ¬(all_sides_equal p ∧ all_angles_equal p) ∧
  ¬(all_sides_equal p ∧ all_diagonals_equal p) ∧
  ¬(all_sides_equal p ∧ all_vertex_midpoint_segments_equal p) ∧
  ¬(all_sides_equal p ∧ vertex_midpoint_segments_perpendicular p) ∧
  ¬(all_sides_equal p ∧ p.has_symmetry_axes) ∧
  ¬(all_sides_equal p ∧ p.is_circumscribable) ∧
  ¬(all_angles_equal p ∧ all_diagonals_equal p) ∧
  ¬(all_angles_equal p ∧ all_vertex_midpoint_segments_equal p) ∧
  ¬(all_angles_equal p ∧ vertex_midpoint_segments_perpendicular p) ∧
  ¬(all_angles_equal p ∧ p.has_symmetry_axes) ∧
  ¬(all_angles_equal p ∧ p.is_inscribable) ∧
  ¬(all_diagonals_equal p ∧ vertex_midpoint_segments_perpendicular p) ∧
  ¬(all_diagonals_equal p ∧ p.is_inscribable) ∧
  ¬(all_vertex_midpoint_segments_equal p ∧ vertex_midpoint_segments_perpendicular p) ∧
  ¬(all_vertex_midpoint_segments_equal p ∧ p.is_inscribable) ∧
  ¬(vertex_midpoint_segments_perpendicular p ∧ p.is_circumscribable) := by
  sorry

#check parallelogram_exclusive_properties

end parallelogram_exclusive_properties_l333_33380


namespace perimeter_of_arranged_rectangles_l333_33366

theorem perimeter_of_arranged_rectangles :
  let small_length : ℕ := 9
  let small_width : ℕ := 3
  let horizontal_count : ℕ := 8
  let vertical_count : ℕ := 4
  let additional_edges : ℕ := 2 * 3
  let large_length : ℕ := small_length * horizontal_count
  let large_width : ℕ := small_width * vertical_count
  let perimeter : ℕ := 2 * (large_length + large_width) + additional_edges
  perimeter = 180 := by sorry

end perimeter_of_arranged_rectangles_l333_33366


namespace perpendicular_to_same_plane_implies_parallel_perpendicular_to_two_planes_implies_parallel_l333_33378

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, then they are parallel
theorem perpendicular_to_same_plane_implies_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel_lines m n :=
sorry

-- Theorem 2: If a line is perpendicular to two planes, then those planes are parallel
theorem perpendicular_to_two_planes_implies_parallel 
  (n : Line) (α β : Plane) :
  perpendicular n α → perpendicular n β → parallel_planes α β :=
sorry

end perpendicular_to_same_plane_implies_parallel_perpendicular_to_two_planes_implies_parallel_l333_33378


namespace root_parity_l333_33342

theorem root_parity (n : ℤ) (x₁ x₂ : ℤ) : 
  x₁^2 + (4*n + 1)*x₁ + 2*n = 0 ∧ 
  x₂^2 + (4*n + 1)*x₂ + 2*n = 0 → 
  (Odd x₁ ∧ Even x₂) ∨ (Even x₁ ∧ Odd x₂) := by
sorry

end root_parity_l333_33342


namespace tan_negative_seven_pi_fourths_l333_33379

theorem tan_negative_seven_pi_fourths : Real.tan (-7 * π / 4) = 1 := by
  sorry

end tan_negative_seven_pi_fourths_l333_33379


namespace range_of_k_l333_33376

theorem range_of_k (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   2*n - 1 < x₁ ∧ x₁ < 2*n + 1 ∧
   2*n - 1 < x₂ ∧ x₂ < 2*n + 1 ∧
   |x₁ - 2*n| = k * Real.sqrt x₁ ∧
   |x₂ - 2*n| = k * Real.sqrt x₂) →
  (0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1)) :=
by sorry

end range_of_k_l333_33376


namespace evaluate_expression_l333_33316

theorem evaluate_expression (x y z : ℚ) :
  x = 1/3 → y = 2/3 → z = -9 → x^2 * y^3 * z = -8/27 := by
  sorry

end evaluate_expression_l333_33316


namespace some_number_value_l333_33339

theorem some_number_value (x y : ℝ) (hx : x = 12) 
  (heq : ((17.28 / x) / (3.6 * y)) = 2) : y = 0.2 := by
  sorry

end some_number_value_l333_33339


namespace sinusoidal_vertical_shift_l333_33381

theorem sinusoidal_vertical_shift 
  (A B C D : ℝ) 
  (h_max : ∀ x, A * Real.sin (B * x + C) + D ≤ 5)
  (h_min : ∀ x, A * Real.sin (B * x + C) + D ≥ -3)
  (h_max_achieved : ∃ x, A * Real.sin (B * x + C) + D = 5)
  (h_min_achieved : ∃ x, A * Real.sin (B * x + C) + D = -3) :
  D = 1 := by
sorry

end sinusoidal_vertical_shift_l333_33381


namespace lukes_weekly_spending_l333_33319

/-- Luke's weekly spending given his earnings and duration --/
theorem lukes_weekly_spending (mowing_earnings weed_eating_earnings : ℕ) (weeks : ℕ) :
  mowing_earnings = 9 →
  weed_eating_earnings = 18 →
  weeks = 9 →
  (mowing_earnings + weed_eating_earnings) / weeks = 3 := by
  sorry

end lukes_weekly_spending_l333_33319


namespace min_draws_for_even_product_l333_33321

theorem min_draws_for_even_product (n : ℕ) (h : n = 14) :
  let S := Finset.range n
  let even_count := (S.filter (λ x => x % 2 = 0)).card
  let odd_count := (S.filter (λ x => x % 2 ≠ 0)).card
  odd_count + 1 = 8 ∧ odd_count = even_count :=
by sorry

end min_draws_for_even_product_l333_33321


namespace polynomial_division_remainder_l333_33395

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^4 - 3*X^2 + 2 : Polynomial ℝ) = (X^2 - 3) * q + 2 := by
  sorry

end polynomial_division_remainder_l333_33395


namespace intersection_theorem_l333_33388

/-- The number of intersection points between two curves -/
def intersection_count (a : ℝ) : ℕ := sorry

/-- First curve equation: x^2 + y^2 = 4a^2 -/
def curve1 (a x y : ℝ) : Prop := x^2 + y^2 = 4 * a^2

/-- Second curve equation: y = x^2 - 4a + 1 -/
def curve2 (a x y : ℝ) : Prop := y = x^2 - 4 * a + 1

theorem intersection_theorem (a : ℝ) :
  intersection_count a = 3 ↔ a > 1/8 := by sorry

end intersection_theorem_l333_33388


namespace combined_sum_equals_3751_l333_33399

/-- The first element of the nth set in the pattern -/
def first_element (n : ℕ) : ℕ := 1 + n * (n - 1) / 2

/-- The last element of the nth set in the pattern -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set -/
def set_sum (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- The combined sum of elements in the 15th and 16th sets -/
def combined_sum : ℕ := set_sum 15 + set_sum 16

theorem combined_sum_equals_3751 : combined_sum = 3751 := by
  sorry

end combined_sum_equals_3751_l333_33399


namespace largest_number_with_sum_17_l333_33396

/-- The largest number with all different digits whose sum is 17 -/
def largest_number : ℕ := 763210

/-- Function to get the digits of a natural number -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- Theorem stating that 763210 is the largest number with all different digits whose sum is 17 -/
theorem largest_number_with_sum_17 :
  (∀ n : ℕ, n ≤ largest_number ∨
    (digits n).sum ≠ 17 ∨
    (digits n).length ≠ (digits n).toFinset.card) ∧
  (digits largest_number).sum = 17 ∧
  (digits largest_number).length = (digits largest_number).toFinset.card :=
sorry

end largest_number_with_sum_17_l333_33396


namespace monitor_width_l333_33310

theorem monitor_width (width height diagonal : ℝ) : 
  width / height = 16 / 9 →
  width ^ 2 + height ^ 2 = diagonal ^ 2 →
  diagonal = 24 →
  width = 384 / Real.sqrt 337 :=
by sorry

end monitor_width_l333_33310


namespace tournament_result_l333_33352

-- Define the type for teams
inductive Team : Type
| A | B | C | D

-- Define the type for match results
inductive MatchResult : Type
| Win | Loss

-- Define a function to represent the number of wins for each team
def wins : Team → Nat
| Team.A => 2
| Team.B => 0
| Team.C => 1
| Team.D => 3

-- Define a function to represent the number of losses for each team
def losses : Team → Nat
| Team.A => 1
| Team.B => 3
| Team.C => 2
| Team.D => 0

-- Theorem statement
theorem tournament_result :
  (∀ t : Team, wins t + losses t = 3) ∧
  (wins Team.A + wins Team.B + wins Team.C + wins Team.D = 6) ∧
  (losses Team.A + losses Team.B + losses Team.C + losses Team.D = 6) :=
by sorry

end tournament_result_l333_33352


namespace system_solution_l333_33305

theorem system_solution (x y m : ℝ) : 
  (2 * x + y = 5) → 
  (x - 2 * y = m) → 
  (2 * x - 3 * y = 1) → 
  (m = 0) := by
sorry

end system_solution_l333_33305


namespace range_of_cosine_function_l333_33325

theorem range_of_cosine_function (f : ℝ → ℝ) (x : ℝ) :
  (f = λ x => 3 * Real.cos (2 * x + π / 3)) →
  (x ∈ Set.Icc 0 (π / 3)) →
  ∃ y ∈ Set.Icc (-3) (3 / 2), f x = y :=
by sorry

end range_of_cosine_function_l333_33325


namespace triangle_ratio_l333_33322

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 :=
sorry


end triangle_ratio_l333_33322


namespace max_travel_distance_is_3_4_l333_33338

/-- Represents the taxi fare structure and travel constraints -/
structure TaxiRide where
  initialFare : ℝ
  initialDistance : ℝ
  additionalFarePerUnit : ℝ
  additionalDistanceUnit : ℝ
  tip : ℝ
  totalBudget : ℝ
  timeLimit : ℝ
  averageSpeed : ℝ

/-- Calculates the maximum distance that can be traveled given the taxi fare structure and constraints -/
def maxTravelDistance (ride : TaxiRide) : ℝ :=
  sorry

/-- Theorem stating that the maximum travel distance is approximately 3.4 miles -/
theorem max_travel_distance_is_3_4 (ride : TaxiRide) 
  (h1 : ride.initialFare = 4)
  (h2 : ride.initialDistance = 3/4)
  (h3 : ride.additionalFarePerUnit = 0.3)
  (h4 : ride.additionalDistanceUnit = 0.1)
  (h5 : ride.tip = 3)
  (h6 : ride.totalBudget = 15)
  (h7 : ride.timeLimit = 45/60)
  (h8 : ride.averageSpeed = 30) :
  ∃ ε > 0, abs (maxTravelDistance ride - 3.4) < ε :=
sorry

end max_travel_distance_is_3_4_l333_33338


namespace tangent_line_equation_l333_33354

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + x + 1

-- Define the point through which the tangent line passes
def point : ℝ × ℝ := (-1, 1)

-- Theorem statement
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := (2 * x₀ + 1)  -- Slope of the tangent line
  (∀ x y, y - y₀ = m * (x - x₀)) ↔ (∀ x y, x + y = 0) :=
sorry

end tangent_line_equation_l333_33354


namespace class_size_calculation_l333_33385

theorem class_size_calculation (tables : Nat) (students_per_table : Nat)
  (bathroom_girls : Nat) (canteen_multiplier : Nat)
  (group1 : Nat) (group2 : Nat) (group3 : Nat) (group4 : Nat)
  (germany : Nat) (france : Nat) (norway : Nat) (italy : Nat) (spain : Nat) (australia : Nat) :
  tables = 6 →
  students_per_table = 3 →
  bathroom_girls = 5 →
  canteen_multiplier = 5 →
  group1 = 4 →
  group2 = 5 →
  group3 = 6 →
  group4 = 3 →
  germany = 3 →
  france = 4 →
  norway = 3 →
  italy = 2 →
  spain = 2 →
  australia = 1 →
  (tables * students_per_table + bathroom_girls + bathroom_girls * canteen_multiplier +
   group1 + group2 + group3 + group4 +
   germany + france + norway + italy + spain + australia) = 81 :=
by sorry

end class_size_calculation_l333_33385


namespace additional_track_length_l333_33358

/-- Calculates the additional track length required when changing the grade of a railroad track. -/
theorem additional_track_length
  (elevation : ℝ)
  (initial_grade : ℝ)
  (final_grade : ℝ)
  (h1 : elevation = 1200)
  (h2 : initial_grade = 0.04)
  (h3 : final_grade = 0.03) :
  (elevation / final_grade) - (elevation / initial_grade) = 10000 :=
by sorry

end additional_track_length_l333_33358


namespace quadratic_factorization_l333_33350

theorem quadratic_factorization (y : ℝ) : y^2 + 14*y + 40 = (y + 4) * (y + 10) := by
  sorry

end quadratic_factorization_l333_33350


namespace special_complex_sum_l333_33357

-- Define the complex function f
def f (z : ℂ) : ℂ := z^2 - 19*z

-- Define the condition for a right triangle
def is_right_triangle (z : ℂ) : Prop :=
  (f z - z) • (f (f z) - f z) = 0

-- Define the structure of z
structure SpecialComplex where
  m : ℕ+
  n : ℕ+
  z : ℂ
  h : z = m + Real.sqrt n + 11*Complex.I

-- State the theorem
theorem special_complex_sum (sc : SpecialComplex) (h : is_right_triangle sc.z) :
  sc.m + sc.n = 230 :=
sorry

end special_complex_sum_l333_33357


namespace max_log_sum_l333_33393

theorem max_log_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 4 * x + y = 40) :
  (Real.log x + Real.log y) ≤ 2 * Real.log 10 :=
sorry

end max_log_sum_l333_33393


namespace mark_initial_punch_l333_33312

/-- The amount of punch in gallons that Mark initially added to the bowl -/
def initial_punch : ℝ := 4

/-- The capacity of the punch bowl in gallons -/
def bowl_capacity : ℝ := 16

/-- The amount of punch Mark adds after his cousin drinks -/
def second_addition : ℝ := 4

/-- The amount of punch Sally drinks -/
def sally_drinks : ℝ := 2

/-- The amount of punch Mark adds to fill the bowl completely -/
def final_addition : ℝ := 12

theorem mark_initial_punch :
  initial_punch / 2 + second_addition - sally_drinks + final_addition = bowl_capacity :=
by sorry

end mark_initial_punch_l333_33312


namespace sum_of_binary_digits_312_l333_33340

def sum_of_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

theorem sum_of_binary_digits_312 : sum_of_binary_digits 312 = 3 := by
  sorry

end sum_of_binary_digits_312_l333_33340


namespace stream_speed_l333_33317

/-- Proves that the speed of a stream is 3 kmph given the conditions of boat travel -/
theorem stream_speed (boat_speed : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : downstream_time = 1)
  (h3 : upstream_time = 1.5) :
  ∃ stream_speed : ℝ, 
    stream_speed = 3 ∧ 
    (boat_speed + stream_speed) * downstream_time = (boat_speed - stream_speed) * upstream_time :=
by sorry

end stream_speed_l333_33317


namespace birthday_gift_savings_l333_33353

/-- Calculates the total amount saved for a mother's birthday gift based on orange sales --/
def total_savings (liam_oranges : ℕ) (liam_price : ℚ) (claire_oranges : ℕ) (claire_price : ℚ)
  (jake_oranges : ℕ) (jake_bundles : ℕ) (jake_price1 : ℚ) (jake_price2 : ℚ) (jake_discount : ℚ) : ℚ :=
  let liam_earnings := (liam_oranges / 2 : ℚ) * liam_price
  let claire_earnings := (claire_oranges : ℚ) * claire_price
  let jake_earnings1 := (jake_bundles / 2 : ℚ) * jake_price1
  let jake_earnings2 := (jake_bundles / 2 : ℚ) * jake_price2
  let jake_total := jake_earnings1 + jake_earnings2
  let jake_discount_amount := jake_total * jake_discount
  let jake_earnings := jake_total - jake_discount_amount
  liam_earnings + claire_earnings + jake_earnings

/-- Theorem stating that the total savings for the mother's birthday gift is $117.88 --/
theorem birthday_gift_savings :
  total_savings 40 (5/2) 30 (6/5) 50 10 3 (9/2) (3/20) = 11788/100 := by
  sorry

end birthday_gift_savings_l333_33353


namespace gcd_problem_l333_33398

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 17 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 65 * b + 143) (5 * b + 22) = 33 := by
  sorry

end gcd_problem_l333_33398


namespace max_value_problem_l333_33360

theorem max_value_problem (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 6 * y ≤ 12) : 
  ∀ a b : ℝ, 4 * a + 3 * b ≤ 10 → 3 * a + 6 * b ≤ 12 → 2 * x + y ≥ 2 * a + b :=
by sorry

end max_value_problem_l333_33360


namespace remaining_payment_l333_33303

theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (h1 : deposit = 150) (h2 : deposit_percentage = 0.1) : 
  (deposit / deposit_percentage) - deposit = 1350 := by
  sorry

end remaining_payment_l333_33303


namespace rectangle_division_l333_33314

theorem rectangle_division (n : ℕ+) 
  (h1 : ∃ a : ℕ+, n = a * a)
  (h2 : ∃ b : ℕ+, n = (n + 98) * b * b) :
  (∃ x y : ℕ+, n = x * y ∧ ((x = 3 ∧ y = 42) ∨ (x = 6 ∧ y = 21) ∨ (x = 24 ∧ y = 48))) :=
by sorry

end rectangle_division_l333_33314


namespace union_equals_N_l333_33327

def M : Set ℝ := {x | x^2 - x < 0}
def N : Set ℝ := {x | -3 < x ∧ x < 3}

theorem union_equals_N : M ∪ N = N := by sorry

end union_equals_N_l333_33327


namespace integral_x_cubed_plus_one_l333_33335

theorem integral_x_cubed_plus_one : ∫ x in (-2)..2, (x^3 + 1) = 4 := by sorry

end integral_x_cubed_plus_one_l333_33335


namespace special_numbers_characterization_l333_33309

/-- Definition of partial numbers for a natural number -/
def partialNumbers (n : ℕ) : Set ℕ :=
  sorry

/-- Predicate to check if all partial numbers of a natural number are prime -/
def allPartialNumbersPrime (n : ℕ) : Prop :=
  ∀ m ∈ partialNumbers n, Nat.Prime m

/-- The set of natural numbers whose partial numbers are all prime -/
def specialNumbers : Set ℕ :=
  {n : ℕ | allPartialNumbersPrime n}

/-- Theorem stating that the set of natural numbers whose partial numbers
    are all prime is exactly {2, 3, 5, 7, 23, 37, 53, 73} -/
theorem special_numbers_characterization :
  specialNumbers = {2, 3, 5, 7, 23, 37, 53, 73} :=
sorry

end special_numbers_characterization_l333_33309


namespace mairead_running_distance_l333_33326

theorem mairead_running_distance (run walk jog : ℝ) : 
  walk = (3/5) * run → 
  jog = 5 * walk → 
  run + walk + jog = 184 → 
  run = 40 := by
  sorry

end mairead_running_distance_l333_33326


namespace function_properties_l333_33349

/-- Given f(x) = a(x+b)(x+c) and g(x) = xf(x) where a ≠ 0 and a, b, c ∈ ℝ,
    prove the following statements -/
theorem function_properties :
  ∃ (a b c : ℝ), a ≠ 0 ∧
    (∀ x, (a * (1 + x) * (x + b) * (x + c) = 0) ↔ (a * (1 - x) * (x + b) * (x + c) = 0)) ∧
    (∀ x, (2 * a * x = -(2 * a * (-x))) ∧ (a * (3 * x^2 + 2 * (b + c) * x + b * c) = a * (3 * (-x)^2 + 2 * (b + c) * (-x) + b * c))) :=
by sorry

end function_properties_l333_33349


namespace no_real_solutions_quadratic_l333_33345

theorem no_real_solutions_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x - k ≠ 0) ↔ k < -9/4 := by
  sorry

end no_real_solutions_quadratic_l333_33345


namespace perfect_square_trinomial_l333_33362

theorem perfect_square_trinomial (a k : ℝ) : 
  (∃ b : ℝ, a^2 + 2*k*a + 9 = (a + b)^2) → (k = 3 ∨ k = -3) :=
by sorry

end perfect_square_trinomial_l333_33362


namespace fraction_reducible_by_11_l333_33337

theorem fraction_reducible_by_11 (k : ℕ) 
  (h : (k^2 - 5*k + 8) % 11 = 0 ∨ (k^2 + 6*k + 19) % 11 = 0) : 
  (k^2 - 5*k + 8) % 11 = 0 ∧ (k^2 + 6*k + 19) % 11 = 0 := by
  sorry

end fraction_reducible_by_11_l333_33337


namespace polynomial_divisibility_l333_33372

theorem polynomial_divisibility (F : Int → Int) (A : Finset Int) :
  (∀ (x : Int), ∃ (a : Int), a ∈ A ∧ (∃ (k : Int), F x = a * k)) →
  (∀ (n : Int), ∃ (coeff : Int), F n = F (n + coeff) - F n) →
  ∃ (B : Finset Int), B ⊆ A ∧ B.card = 2 ∧
    ∀ (n : Int), ∃ (b : Int), b ∈ B ∧ (∃ (k : Int), F n = b * k) :=
by sorry

end polynomial_divisibility_l333_33372


namespace initial_amount_calculation_l333_33369

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Final amount after simple interest -/
def final_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + simple_interest principal rate time

/-- Theorem: Initial amount calculation for given simple interest scenario -/
theorem initial_amount_calculation (rate : ℝ) (time : ℝ) (final : ℝ) 
  (h_rate : rate = 0.04)
  (h_time : time = 5)
  (h_final : final = 900) :
  ∃ (principal : ℝ), final_amount principal rate time = final ∧ principal = 750 := by
  sorry

end initial_amount_calculation_l333_33369


namespace speed_increase_reduces_time_l333_33377

/-- Given a 600-mile trip at 50 mph, prove that increasing speed by 25 mph reduces travel time by 4 hours -/
theorem speed_increase_reduces_time : ∀ (distance : ℝ) (initial_speed : ℝ) (speed_increase : ℝ),
  distance = 600 →
  initial_speed = 50 →
  speed_increase = 25 →
  distance / initial_speed - distance / (initial_speed + speed_increase) = 4 :=
by
  sorry

#check speed_increase_reduces_time

end speed_increase_reduces_time_l333_33377


namespace E_is_integer_l333_33313

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The expression E as defined in the problem -/
def E (n k : ℕ) : ℚ :=
  ((n - 2*k - 2) : ℚ) / ((k + 2) : ℚ) * binomial n k

theorem E_is_integer (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  ∃ (m : ℤ), E n k = m :=
sorry

end E_is_integer_l333_33313


namespace negation_of_existence_proposition_l333_33315

open Real

theorem negation_of_existence_proposition :
  (¬ ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔
  (∀ a : ℝ, ∀ x : ℝ, a * x^2 + 1 ≠ 0) :=
by sorry

end negation_of_existence_proposition_l333_33315


namespace max_y_over_x_on_circle_l333_33383

theorem max_y_over_x_on_circle (z : ℂ) (x y : ℝ) :
  z = x + y * I →
  x ≠ 0 →
  Complex.abs (z - 2) = Real.sqrt 3 →
  ∃ (k : ℝ), ∀ (w : ℂ) (u v : ℝ),
    w = u + v * I →
    u ≠ 0 →
    Complex.abs (w - 2) = Real.sqrt 3 →
    |v / u| ≤ k ∧
    k = Real.sqrt 3 :=
sorry

end max_y_over_x_on_circle_l333_33383


namespace arrangement_count_l333_33386

/-- Represents the number of boys -/
def num_boys : Nat := 2

/-- Represents the number of girls -/
def num_girls : Nat := 3

/-- Represents the total number of students -/
def total_students : Nat := num_boys + num_girls

/-- Represents that the girls are adjacent -/
def girls_adjacent : Prop := True

/-- Represents that boy A is to the left of boy B -/
def boy_A_left_of_B : Prop := True

/-- The number of different arrangements -/
def num_arrangements : Nat := 18

theorem arrangement_count :
  girls_adjacent →
  boy_A_left_of_B →
  num_arrangements = 18 :=
by
  sorry

end arrangement_count_l333_33386


namespace solve_equation_l333_33391

theorem solve_equation : ∃ x : ℚ, 25 - (3 * 5) = (2 * x) + 1 ∧ x = 9/2 := by
  sorry

end solve_equation_l333_33391


namespace marble_fraction_after_tripling_l333_33348

theorem marble_fraction_after_tripling (total : ℚ) (h1 : total > 0) : 
  let initial_blue := (4/7) * total
  let initial_red := total - initial_blue
  let new_blue := 3 * initial_blue
  let new_total := new_blue + initial_red
  initial_red / new_total = 1/5 := by
sorry

end marble_fraction_after_tripling_l333_33348


namespace area_of_EFGH_l333_33389

-- Define the rectangle and squares
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

structure Square :=
  (side : ℝ)

-- Define the problem setup
def smallest_square : Square :=
  { side := 1 }

def rectangle_EFGH : Rectangle :=
  { width := 4, height := 3 }

-- Define the theorem
theorem area_of_EFGH :
  (rectangle_EFGH.width * rectangle_EFGH.height : ℝ) = 6 := by
  sorry

end area_of_EFGH_l333_33389


namespace bruce_initial_eggs_l333_33392

theorem bruce_initial_eggs (bruce_final : ℕ) (eggs_lost : ℕ) : 
  bruce_final = 5 → eggs_lost = 70 → bruce_final + eggs_lost = 75 := by
  sorry

end bruce_initial_eggs_l333_33392


namespace ashley_pies_eaten_l333_33382

theorem ashley_pies_eaten (pies_per_day : ℕ) (days : ℕ) (remaining_pies : ℕ) :
  pies_per_day = 7 → days = 12 → remaining_pies = 34 →
  pies_per_day * days - remaining_pies = 50 := by
  sorry

end ashley_pies_eaten_l333_33382


namespace drummer_stick_sets_l333_33368

/-- Calculates the total number of drum stick sets used by a drummer over multiple nights. -/
theorem drummer_stick_sets (sets_per_show : ℕ) (sets_tossed : ℕ) (nights : ℕ) : 
  sets_per_show = 5 → sets_tossed = 6 → nights = 30 → 
  (sets_per_show + sets_tossed) * nights = 330 := by
  sorry

#check drummer_stick_sets

end drummer_stick_sets_l333_33368


namespace power_sum_l333_33334

theorem power_sum (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m+n) = 6 := by
  sorry

end power_sum_l333_33334


namespace profit_share_ratio_l333_33363

theorem profit_share_ratio (total_profit : ℝ) (difference : ℝ) 
  (h_total : total_profit = 1000)
  (h_diff : difference = 200) :
  ∃ (x y : ℝ), 
    x + y = total_profit ∧ 
    x - y = difference ∧ 
    x / total_profit = 3 / 5 := by
  sorry

end profit_share_ratio_l333_33363


namespace largest_square_tile_size_l333_33328

theorem largest_square_tile_size (length width : ℕ) (h1 : length = 378) (h2 : width = 595) :
  ∃ (tile_size : ℕ), tile_size = Nat.gcd length width ∧ tile_size = 7 := by
  sorry

end largest_square_tile_size_l333_33328


namespace polynomial_factorization_l333_33365

theorem polynomial_factorization (x : ℝ) : 
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3)*(8*x^2 + x + 3) := by
  sorry

end polynomial_factorization_l333_33365


namespace quadratic_non_real_roots_l333_33364

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end quadratic_non_real_roots_l333_33364


namespace curve_intersection_distance_l333_33329

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - x + y = 0

def C₂ (t x y : ℝ) : Prop := x = 1/2 - (Real.sqrt 2 / 2) * t ∧ y = (Real.sqrt 2 / 2) * t

-- Theorem statement
theorem curve_intersection_distance : 
  -- The polar curve ρ = cos θ - sin θ is equivalent to C₁
  (∀ (ρ θ : ℝ), ρ = Real.cos θ - Real.sin θ ↔ C₁ (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  -- The distance between intersection points is √6/2
  (∃ (t₁ t₂ : ℝ), 
    (C₁ (1/2 - (Real.sqrt 2 / 2) * t₁) ((Real.sqrt 2 / 2) * t₁)) ∧
    (C₁ (1/2 - (Real.sqrt 2 / 2) * t₂) ((Real.sqrt 2 / 2) * t₂)) ∧
    (C₂ t₁ (1/2 - (Real.sqrt 2 / 2) * t₁) ((Real.sqrt 2 / 2) * t₁)) ∧
    (C₂ t₂ (1/2 - (Real.sqrt 2 / 2) * t₂) ((Real.sqrt 2 / 2) * t₂)) ∧
    (t₁ ≠ t₂) ∧
    ((1/2 - (Real.sqrt 2 / 2) * t₁ - (1/2 - (Real.sqrt 2 / 2) * t₂))^2 + 
     ((Real.sqrt 2 / 2) * t₁ - (Real.sqrt 2 / 2) * t₂)^2 = 3/2)) := by
  sorry

end curve_intersection_distance_l333_33329


namespace symmetric_points_sum_l333_33323

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal. -/
def symmetric_wrt_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = y₂

/-- Given point A(2, -3) is symmetric to point A'(a, b) with respect to the y-axis, prove that a + b = -5. -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_y_axis 2 (-3) a b) : a + b = -5 := by
  sorry

end symmetric_points_sum_l333_33323


namespace square_pyramid_sum_l333_33343

/-- A square pyramid is a three-dimensional geometric shape with a square base and four triangular faces -/
structure SquarePyramid where
  base : Square
  apex : Point

/-- The number of faces in a square pyramid -/
def num_faces (sp : SquarePyramid) : ℕ := 5

/-- The number of edges in a square pyramid -/
def num_edges (sp : SquarePyramid) : ℕ := 8

/-- The number of vertices in a square pyramid -/
def num_vertices (sp : SquarePyramid) : ℕ := 5

/-- The sum of faces, edges, and vertices of a square pyramid is 18 -/
theorem square_pyramid_sum (sp : SquarePyramid) : 
  num_faces sp + num_edges sp + num_vertices sp = 18 := by
  sorry

end square_pyramid_sum_l333_33343


namespace train_bridge_crossing_time_l333_33307

/-- Proves that a train with given length and speed takes a specific time to cross a bridge -/
theorem train_bridge_crossing_time
  (train_length : Real)
  (train_speed_kmh : Real)
  (total_length : Real)
  (h1 : train_length = 100)
  (h2 : train_speed_kmh = 45)
  (h3 : total_length = 275) :
  let train_speed_ms : Real := train_speed_kmh * 1000 / 3600
  let bridge_length : Real := total_length - train_length
  let distance_to_cross : Real := train_length + bridge_length
  let time_to_cross : Real := distance_to_cross / train_speed_ms
  time_to_cross = 30 := by
  sorry

end train_bridge_crossing_time_l333_33307


namespace qt_squared_eq_three_l333_33306

-- Define the points
variable (X Y Z W P Q R S T U : ℝ × ℝ)

-- Define the square XYZW
def is_square (X Y Z W : ℝ × ℝ) : Prop := sorry

-- Define that P and S lie on XZ and XW respectively
def on_line (P X Z : ℝ × ℝ) : Prop := sorry
def on_line' (S X W : ℝ × ℝ) : Prop := sorry

-- Define XP = XS = √3
def distance_eq_sqrt3 (X P S : ℝ × ℝ) : Prop := sorry

-- Define Q and R lie on YZ and YW respectively
def on_line'' (Q Y Z : ℝ × ℝ) : Prop := sorry
def on_line''' (R Y W : ℝ × ℝ) : Prop := sorry

-- Define T and U lie on PS
def on_line'''' (T P S : ℝ × ℝ) : Prop := sorry
def on_line''''' (U P S : ℝ × ℝ) : Prop := sorry

-- Define QT ⊥ PS and RU ⊥ PS
def perpendicular (Q T P S : ℝ × ℝ) : Prop := sorry
def perpendicular' (R U P S : ℝ × ℝ) : Prop := sorry

-- Define areas of the shapes
def area_eq_1_5 (X P S : ℝ × ℝ) : Prop := sorry
def area_eq_1_5' (Y Q T P : ℝ × ℝ) : Prop := sorry
def area_eq_1_5'' (W S U R : ℝ × ℝ) : Prop := sorry
def area_eq_1_5''' (Y R U T Q : ℝ × ℝ) : Prop := sorry

-- The theorem to prove
theorem qt_squared_eq_three 
  (h1 : is_square X Y Z W)
  (h2 : on_line P X Z)
  (h3 : on_line' S X W)
  (h4 : distance_eq_sqrt3 X P S)
  (h5 : on_line'' Q Y Z)
  (h6 : on_line''' R Y W)
  (h7 : on_line'''' T P S)
  (h8 : on_line''''' U P S)
  (h9 : perpendicular Q T P S)
  (h10 : perpendicular' R U P S)
  (h11 : area_eq_1_5 X P S)
  (h12 : area_eq_1_5' Y Q T P)
  (h13 : area_eq_1_5'' W S U R)
  (h14 : area_eq_1_5''' Y R U T Q) :
  (Q.1 - T.1)^2 + (Q.2 - T.2)^2 = 3 := by sorry

end qt_squared_eq_three_l333_33306


namespace quadratic_form_minimum_l333_33346

theorem quadratic_form_minimum : ∀ x y : ℝ, 
  2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x - 6 * y + 1 ≥ -3 ∧ 
  (2 * (3/2)^2 + 4 * (3/2) * (1/2) + 5 * (1/2)^2 - 4 * (3/2) - 6 * (1/2) + 1 = -3) := by
  sorry

end quadratic_form_minimum_l333_33346


namespace josh_bracelets_l333_33397

-- Define the parameters
def cost_per_bracelet : ℚ := 1
def selling_price : ℚ := 1.5
def cookie_cost : ℚ := 3
def money_left : ℚ := 3

-- Define the function to calculate the number of bracelets
def num_bracelets : ℚ := (cookie_cost + money_left) / (selling_price - cost_per_bracelet)

-- Theorem statement
theorem josh_bracelets : num_bracelets = 12 := by
  sorry

end josh_bracelets_l333_33397


namespace feifei_leilei_age_sum_feifei_leilei_age_sum_proof_l333_33387

theorem feifei_leilei_age_sum : ℕ → ℕ → Prop :=
  fun feifei_age leilei_age =>
    (feifei_age = leilei_age / 2 + 12) →
    (feifei_age + 1 = 2 * (leilei_age + 1) - 34) →
    (feifei_age + leilei_age = 57)

theorem feifei_leilei_age_sum_proof : ∃ (f l : ℕ), feifei_leilei_age_sum f l :=
  sorry

end feifei_leilei_age_sum_feifei_leilei_age_sum_proof_l333_33387


namespace equation_describes_spiral_l333_33347

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The equation r * θ = c -/
def spiralEquation (p : CylindricalPoint) (c : ℝ) : Prop :=
  p.r * p.θ = c

/-- A spiral in cylindrical coordinates -/
def isSpiral (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, ∀ p ∈ S, spiralEquation p c

/-- The shape described by r * θ = c is a spiral -/
theorem equation_describes_spiral (c : ℝ) :
  isSpiral {p : CylindricalPoint | spiralEquation p c} :=
sorry

end equation_describes_spiral_l333_33347


namespace complement_of_beta_l333_33331

/-- Given two angles α and β that are complementary and α > β, 
    the complement of β is (α - β)/2 -/
theorem complement_of_beta (α β : ℝ) 
  (h1 : α + β = 90) -- α and β are complementary
  (h2 : α > β) : 
  90 - β = (α - β) / 2 := by
  sorry

end complement_of_beta_l333_33331


namespace range_of_a_l333_33332

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 3*a < x ∧ x < a ∧ a < 0}
def B : Set ℝ := {x | x < -4 ∨ x ≥ -2}

-- Define the conditions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- State the theorem
theorem range_of_a :
  (∀ x, p x a → q x) ∧ 
  (∃ x, ¬p x a ∧ q x) →
  (a ≤ -4 ∨ (-2/3 ≤ a ∧ a < 0)) :=
sorry

end range_of_a_l333_33332


namespace f_4_eq_7_solutions_l333_33324

/-- The function f(x) = x^2 - 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The fourth composition of f -/
def f_4 (x : ℝ) : ℝ := f (f (f (f x)))

/-- The theorem stating that there are exactly 5 distinct real solutions to f⁴(c) = 7 -/
theorem f_4_eq_7_solutions :
  ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ c : ℝ, c ∈ s ↔ f_4 c = 7 := by sorry

end f_4_eq_7_solutions_l333_33324


namespace min_value_theorem_l333_33375

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ 4 :=
sorry

end min_value_theorem_l333_33375


namespace count_multiples_of_30_l333_33384

def smallest_square_multiple_of_30 : ℕ := 900
def smallest_fourth_power_multiple_of_30 : ℕ := 810000

theorem count_multiples_of_30 : 
  (smallest_fourth_power_multiple_of_30 / 30) - (smallest_square_multiple_of_30 / 30) + 1 = 26971 := by
  sorry

end count_multiples_of_30_l333_33384


namespace gcd_count_for_product_252_l333_33304

theorem gcd_count_for_product_252 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 252) :
  ∃! (s : Finset ℕ+), s.card = 8 ∧ ∀ d, d ∈ s ↔ ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 252 :=
sorry

end gcd_count_for_product_252_l333_33304


namespace inequality_solution_l333_33356

-- Define the function f(x) = 1/√(x+1)
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x + 1)

-- Define the solution set
def solution_set : Set ℝ := Set.Icc 0 (1/2)

-- Define the inequality
def inequality (l k x : ℝ) : Prop := 
  1 - l * x ≤ f x ∧ f x ≤ 1 - k * x

-- Theorem statement
theorem inequality_solution (l k : ℝ) : 
  (∀ x ∈ solution_set, inequality l k x) ↔ (l = 1/2 ∧ k = 2 - 2 * Real.sqrt 6 / 3) :=
sorry

end inequality_solution_l333_33356


namespace prime_power_constraints_l333_33311

theorem prime_power_constraints (a b m n : ℕ) : 
  a > 1 → b > 1 → m > 1 → n > 1 → 
  Nat.Prime (a^n - 1) → Nat.Prime (b^m + 1) → 
  (∃ k : ℕ, m = 2^k) ∧ n = 2 := by
sorry

end prime_power_constraints_l333_33311


namespace negative_half_power_twenty_times_negative_two_power_twentysix_l333_33330

theorem negative_half_power_twenty_times_negative_two_power_twentysix :
  -0.5^20 * (-2)^26 = -64 := by
  sorry

end negative_half_power_twenty_times_negative_two_power_twentysix_l333_33330


namespace quadratic_function_inequality_l333_33361

theorem quadratic_function_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a < 0) 
  (hx : x₁ < x₂) 
  (hsum : x₁ + x₂ = 0) : 
  2 * a * x₁^2 - a * x₁ + 1 < 2 * a * x₂^2 - a * x₂ + 1 := by
  sorry

end quadratic_function_inequality_l333_33361


namespace fourth_root_sqrt_five_squared_l333_33300

theorem fourth_root_sqrt_five_squared : 
  ((5 ^ (1 / 2)) ^ 5) ^ (1 / 4) ^ 2 = 5 * (5 ^ (1 / 4)) := by
  sorry

end fourth_root_sqrt_five_squared_l333_33300


namespace paving_stones_required_l333_33373

/-- The minimum number of paving stones required to cover a rectangular courtyard -/
theorem paving_stones_required (courtyard_length courtyard_width stone_length stone_width : ℝ) 
  (courtyard_length_pos : 0 < courtyard_length)
  (courtyard_width_pos : 0 < courtyard_width)
  (stone_length_pos : 0 < stone_length)
  (stone_width_pos : 0 < stone_width)
  (h_courtyard_length : courtyard_length = 120)
  (h_courtyard_width : courtyard_width = 25.5)
  (h_stone_length : stone_length = 3.5)
  (h_stone_width : stone_width = 3) : 
  ⌈(courtyard_length * courtyard_width) / (stone_length * stone_width)⌉ = 292 := by
  sorry

end paving_stones_required_l333_33373


namespace fraction_equalities_l333_33351

theorem fraction_equalities (a b : ℚ) (h : a / b = 5 / 6) : 
  ((a + 2 * b) / b = 17 / 6) ∧
  (b / (2 * a - b) = 3 / 2) ∧
  ((a + 3 * b) / (2 * a) = 23 / 10) ∧
  (a / (3 * b) = 5 / 18) ∧
  ((a - 2 * b) / b = -7 / 6) := by
sorry

end fraction_equalities_l333_33351


namespace line_angle_and_parallel_distance_l333_33390

/-- Line in 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Angle between two lines -/
def angle_between_lines (l1 l2 : Line) : ℝ := sorry

/-- Distance between two parallel lines -/
def distance_between_parallel_lines (l1 l2 : Line) : ℝ := sorry

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

theorem line_angle_and_parallel_distance 
  (l : Line) 
  (l1 : Line) 
  (l2 : Line) 
  (h1 : l.a = 1 ∧ l.b = -2 ∧ l.c = 1) 
  (h2 : l1.a = 2 ∧ l1.b = 1 ∧ l1.c = 1) 
  (h3 : are_parallel l l2) 
  (h4 : distance_between_parallel_lines l l2 = 1) : 
  (angle_between_lines l l1 = π / 2) ∧ 
  ((l2.a = l.a ∧ l2.b = l.b ∧ (l2.c = l.c - Real.sqrt 5 ∨ l2.c = l.c + Real.sqrt 5))) := 
by sorry

end line_angle_and_parallel_distance_l333_33390


namespace lucky_larry_problem_l333_33301

theorem lucky_larry_problem (a b c d e : ℤ) : 
  a = 3 → b = 6 → c = 2 → d = 5 →
  (a - b + c - d + e = a - (b + (c - (d - e)))) → e = 3 := by
  sorry

end lucky_larry_problem_l333_33301


namespace inverse_32_mod_97_l333_33394

theorem inverse_32_mod_97 (h : (2⁻¹ : ZMod 97) = 49) : (32⁻¹ : ZMod 97) = 49 := by
  sorry

end inverse_32_mod_97_l333_33394


namespace sarah_mia_games_together_l333_33336

def total_players : ℕ := 12
def players_per_game : ℕ := 6

theorem sarah_mia_games_together :
  let total_combinations := Nat.choose total_players players_per_game
  let games_per_player := total_combinations / 2
  let other_players := total_players - 2
  games_per_player * (players_per_game - 1) / other_players = 210 := by
  sorry

end sarah_mia_games_together_l333_33336


namespace normal_distribution_probability_l333_33341

-- Define a random variable following a normal distribution
def normal_distribution (μ : ℝ) (σ : ℝ) (x : ℝ) : ℝ := sorry

-- Define the cumulative distribution function (CDF) for the normal distribution
def normal_cdf (μ : ℝ) (σ : ℝ) (x : ℝ) : ℝ := sorry

theorem normal_distribution_probability 
  (ξ : ℝ → ℝ) -- ξ is a function representing the random variable
  (σ : ℝ) -- standard deviation
  (h1 : σ > 0) -- condition that σ is positive
  (h2 : ∀ x, ξ x = normal_distribution 1 σ x) -- ξ follows N(1, σ²)
  (h3 : normal_cdf 1 σ 1 - normal_cdf 1 σ 0 = 0.4) -- P(0 < ξ < 1) = 0.4
  : normal_cdf 1 σ 2 - normal_cdf 1 σ 0 = 0.8 := by sorry

end normal_distribution_probability_l333_33341


namespace no_four_consecutive_integers_product_perfect_square_l333_33308

theorem no_four_consecutive_integers_product_perfect_square :
  ∀ x : ℕ+, ¬∃ y : ℕ, x * (x + 1) * (x + 2) * (x + 3) = y^2 := by
sorry

end no_four_consecutive_integers_product_perfect_square_l333_33308


namespace quadratic_root_value_l333_33355

theorem quadratic_root_value (a : ℝ) : (1 : ℝ)^2 + a * 1 + 4 = 0 → a = -5 := by
  sorry

end quadratic_root_value_l333_33355


namespace intersection_A_complement_B_find_m_value_l333_33320

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x - 5 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x - m < 0}

-- Part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 3) = {x : ℝ | x = -1 ∨ (3 ≤ x ∧ x ≤ 5)} := by sorry

-- Part 2
theorem find_m_value (h : A ∩ B m = {x : ℝ | -1 ≤ x ∧ x < 4}) : m = 8 := by sorry

end intersection_A_complement_B_find_m_value_l333_33320


namespace undefined_rational_function_l333_33359

theorem undefined_rational_function (x : ℝ) :
  (x^2 - 24*x + 144 = 0) ↔ (x = 12) :=
by sorry

end undefined_rational_function_l333_33359


namespace unique_positive_integer_solution_l333_33344

theorem unique_positive_integer_solution :
  ∃! (x : ℕ+), (4 * (x - 1) : ℝ) < 3 * x - 2 := by
  sorry

end unique_positive_integer_solution_l333_33344


namespace fraction_inequality_l333_33333

theorem fraction_inequality (x : ℝ) :
  0 ≤ x ∧ x ≤ 3 →
  (3 * x + 2 < 2 * (5 * x - 4) ↔ 10 / 7 < x ∧ x ≤ 3) :=
by sorry

end fraction_inequality_l333_33333


namespace difference_of_x_and_y_l333_33367

theorem difference_of_x_and_y (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (diff_squares : x^2 - y^2 = 24) : 
  x - y = 3 := by
  sorry

end difference_of_x_and_y_l333_33367


namespace second_number_is_40_l333_33370

theorem second_number_is_40 (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (ratio_ab : a / b = 3 / 4)
  (ratio_bc : b / c = 4 / 5)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0) :
  b = 40 := by
sorry

end second_number_is_40_l333_33370
