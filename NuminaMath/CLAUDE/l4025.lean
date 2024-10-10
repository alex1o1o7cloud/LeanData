import Mathlib

namespace bus_driver_hours_l4025_402569

theorem bus_driver_hours (regular_rate overtime_rate_factor total_compensation : ℚ) : 
  regular_rate = 14 →
  overtime_rate_factor = 1.75 →
  total_compensation = 982 →
  ∃ (regular_hours overtime_hours : ℕ),
    regular_hours = 40 ∧
    overtime_hours = 17 ∧
    regular_hours + overtime_hours = 57 ∧
    regular_rate * regular_hours + (regular_rate * overtime_rate_factor) * overtime_hours = total_compensation :=
by sorry

end bus_driver_hours_l4025_402569


namespace chicken_count_after_purchase_l4025_402502

theorem chicken_count_after_purchase (initial_count purchase_count : ℕ) 
  (h1 : initial_count = 26) 
  (h2 : purchase_count = 28) : 
  initial_count + purchase_count = 54 := by
  sorry

end chicken_count_after_purchase_l4025_402502


namespace prob_at_least_half_girls_l4025_402576

/-- The number of children in the family -/
def num_children : ℕ := 5

/-- The probability of having a girl for each child -/
def prob_girl : ℚ := 1/2

/-- The number of possible combinations of boys and girls -/
def total_combinations : ℕ := 2^num_children

/-- The number of combinations with at least half girls -/
def favorable_combinations : ℕ := (num_children.choose 3) + (num_children.choose 4) + (num_children.choose 5)

/-- The probability of having at least half girls in a family of five children -/
theorem prob_at_least_half_girls : 
  (favorable_combinations : ℚ) / total_combinations = 1/2 := by sorry

end prob_at_least_half_girls_l4025_402576


namespace employee_pay_l4025_402543

theorem employee_pay (total_pay : ℚ) (a_pay : ℚ) (b_pay : ℚ) :
  total_pay = 570 →
  a_pay = 1.5 * b_pay →
  total_pay = a_pay + b_pay →
  b_pay = 228 := by
sorry

end employee_pay_l4025_402543


namespace square_vector_problem_l4025_402567

theorem square_vector_problem (a b c : ℝ × ℝ) : 
  (∀ x : ℝ × ℝ, ‖x‖ = 1 → ‖x + x‖ = ‖a‖) →  -- side length is 1
  ‖a‖ = 1 →                                -- |a| = 1 (side length)
  ‖c‖ = Real.sqrt 2 →                      -- |c| = √2 (diagonal)
  a + b = c →                              -- vector addition
  ‖b - a - c‖ = 2 := by sorry

end square_vector_problem_l4025_402567


namespace max_distance_point_to_line_l4025_402524

/-- The maximum distance from point A(1,1) to the line x*cos(θ) + y*sin(θ) - 2 = 0 -/
theorem max_distance_point_to_line :
  let A : ℝ × ℝ := (1, 1)
  let line (θ : ℝ) (x y : ℝ) := x * Real.cos θ + y * Real.sin θ - 2 = 0
  let distance (θ : ℝ) := |Real.cos θ + Real.sin θ - 2| / Real.sqrt (Real.cos θ ^ 2 + Real.sin θ ^ 2)
  (∀ θ : ℝ, distance θ ≤ 2 + Real.sqrt 2) ∧ (∃ θ : ℝ, distance θ = 2 + Real.sqrt 2) :=
by sorry


end max_distance_point_to_line_l4025_402524


namespace saras_quarters_l4025_402521

/-- Sara's quarters problem -/
theorem saras_quarters (initial_quarters final_quarters dad_quarters : ℕ) 
  (h1 : initial_quarters = 21)
  (h2 : dad_quarters = 49)
  (h3 : final_quarters = initial_quarters + dad_quarters) :
  final_quarters = 70 := by
  sorry

end saras_quarters_l4025_402521


namespace total_time_calculation_l4025_402565

-- Define the time spent sharpening the knife
def sharpening_time : ℕ := 10

-- Define the multiplier for peeling time
def peeling_multiplier : ℕ := 3

-- Theorem to prove
theorem total_time_calculation :
  sharpening_time + peeling_multiplier * sharpening_time = 40 := by
  sorry

end total_time_calculation_l4025_402565


namespace arithmetic_mean_geq_geometric_mean_l4025_402540

theorem arithmetic_mean_geq_geometric_mean {x y : ℝ} (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y) / 2 ≥ Real.sqrt (x * y) := by
  sorry

end arithmetic_mean_geq_geometric_mean_l4025_402540


namespace solution_set_equivalence_l4025_402507

theorem solution_set_equivalence : 
  {x : ℝ | (x + 3)^2 < 1} = {x : ℝ | -4 < x ∧ x < -2} := by
sorry

end solution_set_equivalence_l4025_402507


namespace trapezoid_segment_length_l4025_402588

/-- Represents a trapezoid ABCD with sides AB and CD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ

/-- The theorem statement -/
theorem trapezoid_segment_length (t : Trapezoid) :
  (t.AB / t.CD = 3) →  -- Area ratio implies base ratio
  (t.AB + t.CD = 320) →
  t.AB = 240 := by
  sorry

end trapezoid_segment_length_l4025_402588


namespace cube_division_impossibility_l4025_402562

/-- Represents a rectangular parallelepiped with dimensions (n, n+1, n+2) --/
structure Parallelepiped where
  n : ℕ

/-- The volume of a parallelepiped --/
def volume (p : Parallelepiped) : ℕ := p.n * (p.n + 1) * (p.n + 2)

/-- Theorem: It's impossible to divide a cube of volume 8000 into parallelepipeds
    with consecutive natural number dimensions --/
theorem cube_division_impossibility :
  ¬ ∃ (parallelepipeds : List Parallelepiped),
    (parallelepipeds.map volume).sum = 8000 :=
sorry

end cube_division_impossibility_l4025_402562


namespace perpendicular_vectors_l4025_402586

def vector_a : ℝ × ℝ := (1, -2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem perpendicular_vectors (x : ℝ) : 
  (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) → x = 2 := by
  sorry

end perpendicular_vectors_l4025_402586


namespace min_value_fraction_l4025_402530

theorem min_value_fraction (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (∀ y, 0 < y ∧ y < 1 → (1 / (4 * x) + 4 / (1 - x)) ≤ (1 / (4 * y) + 4 / (1 - y))) →
  1 / (4 * x) + 4 / (1 - x) = 25 / 4 :=
by sorry

end min_value_fraction_l4025_402530


namespace no_self_inverse_plus_one_function_l4025_402551

theorem no_self_inverse_plus_one_function : ¬∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 := by
  sorry

end no_self_inverse_plus_one_function_l4025_402551


namespace system_solutions_l4025_402564

theorem system_solutions (x₁ x₂ x₃ : ℝ) : 
  (2 * x₁^2 / (1 + x₁^2) = x₂ ∧ 
   2 * x₂^2 / (1 + x₂^2) = x₃ ∧ 
   2 * x₃^2 / (1 + x₃^2) = x₁) ↔ 
  ((x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0) ∨ 
   (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1)) :=
by sorry

end system_solutions_l4025_402564


namespace stockholm_to_malmo_via_gothenburg_l4025_402566

/-- Represents a distance on a map --/
structure MapDistance :=
  (cm : ℝ)

/-- Represents a real-world distance --/
structure RealDistance :=
  (km : ℝ)

/-- Represents a map scale --/
structure MapScale :=
  (km_per_cm : ℝ)

/-- Converts a map distance to a real distance given a scale --/
def convert_distance (md : MapDistance) (scale : MapScale) : RealDistance :=
  ⟨md.cm * scale.km_per_cm⟩

/-- Adds two real distances --/
def add_distances (d1 d2 : RealDistance) : RealDistance :=
  ⟨d1.km + d2.km⟩

theorem stockholm_to_malmo_via_gothenburg 
  (stockholm_gothenburg : MapDistance)
  (gothenburg_malmo : MapDistance)
  (scale : MapScale)
  (h1 : stockholm_gothenburg.cm = 120)
  (h2 : gothenburg_malmo.cm = 150)
  (h3 : scale.km_per_cm = 20) :
  (add_distances 
    (convert_distance stockholm_gothenburg scale)
    (convert_distance gothenburg_malmo scale)).km = 5400 :=
by
  sorry

end stockholm_to_malmo_via_gothenburg_l4025_402566


namespace insect_jumps_l4025_402599

theorem insect_jumps (s : ℝ) (h_s : 1/2 < s ∧ s < 1) (c : ℝ) (h_c : 0 ≤ c ∧ c ≤ 1) :
  ∀ ε > 0, ∃ (n : ℕ) (x : ℕ → ℝ),
    (x 0 = 0 ∨ x 0 = 1) ∧
    (∀ i, i < n → (x (i + 1) = x i * s ∨ x (i + 1) = (x i - 1) * s + 1)) ∧
    |x n - c| < ε :=
by sorry

end insect_jumps_l4025_402599


namespace range_of_a_l4025_402545

open Set Real

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Ioo 2 3, x^2 + 5 > a*x) = false → 
  a ∈ Ici (2 * sqrt 5) :=
sorry

end range_of_a_l4025_402545


namespace points_on_line_l4025_402508

-- Define the points
def p1 : ℝ × ℝ := (4, 8)
def p2 : ℝ × ℝ := (2, 2)
def p3 : ℝ × ℝ := (3, 5)
def p4 : ℝ × ℝ := (0, -2)
def p5 : ℝ × ℝ := (1, 1)
def p6 : ℝ × ℝ := (5, 11)
def p7 : ℝ × ℝ := (6, 14)

-- Function to check if a point lies on the line
def lies_on_line (p : ℝ × ℝ) : Prop :=
  let m := (p1.2 - p2.2) / (p1.1 - p2.1)
  let b := p1.2 - m * p1.1
  p.2 = m * p.1 + b

-- Theorem stating which points lie on the line
theorem points_on_line :
  lies_on_line p3 ∧ lies_on_line p6 ∧ lies_on_line p7 ∧
  ¬lies_on_line p4 ∧ ¬lies_on_line p5 :=
sorry

end points_on_line_l4025_402508


namespace range_of_m_l4025_402546

-- Define a decreasing function on (-∞, 0)
def DecreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f x > f y

-- Define the theorem
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : DecreasingOnNegative f) 
  (h2 : f (1 - m) < f (m - 3)) : 
  1 < m ∧ m < 2 := by
  sorry

end range_of_m_l4025_402546


namespace max_product_constraint_l4025_402582

theorem max_product_constraint (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a + b = 4) :
  (a - 1) * (b - 1) ≤ 1 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 1 ∧ b₀ > 1 ∧ a₀ + b₀ = 4 ∧ (a₀ - 1) * (b₀ - 1) = 1 :=
by sorry

end max_product_constraint_l4025_402582


namespace judes_current_age_jude_is_two_years_old_l4025_402574

/-- Proves Jude's current age given Heath's current age and their future age relationship -/
theorem judes_current_age (heath_current_age : ℕ) (future_years : ℕ) (future_age_ratio : ℕ) : ℕ :=
  let heath_future_age := heath_current_age + future_years
  let jude_future_age := heath_future_age / future_age_ratio
  let age_difference := heath_future_age - jude_future_age
  heath_current_age - age_difference

/-- The main theorem that proves Jude's current age is 2 years old -/
theorem jude_is_two_years_old : judes_current_age 16 5 3 = 2 := by
  sorry

end judes_current_age_jude_is_two_years_old_l4025_402574


namespace hyperbola_a_plus_h_value_l4025_402585

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  pos_a : a > 0
  pos_b : b > 0

/-- Theorem: For a hyperbola with given asymptotes and passing through a specific point, a + h = 16/3 -/
theorem hyperbola_a_plus_h_value (H : Hyperbola) 
  (asymptote1 : ∀ x y : ℝ, y = 3*x + 3 → (∀ t : ℝ, (y - H.k)^2/(H.a^2) - (x - H.h)^2/(H.b^2) = t))
  (asymptote2 : ∀ x y : ℝ, y = -3*x - 1 → (∀ t : ℝ, (y - H.k)^2/(H.a^2) - (x - H.h)^2/(H.b^2) = t))
  (point_on_hyperbola : (11 - H.k)^2/(H.a^2) - (2 - H.h)^2/(H.b^2) = 1) :
  H.a + H.h = 16/3 := by
  sorry

end hyperbola_a_plus_h_value_l4025_402585


namespace square_roots_equation_l4025_402519

theorem square_roots_equation (a b : ℝ) :
  let f (x : ℝ) := a * b * x^2 - (a + b) * x + 1
  let g (x : ℝ) := a^2 * b^2 * x^2 - (a^2 + b^2) * x + 1
  ∀ (r : ℝ), f r = 0 → g (r^2) = 0 :=
by sorry

end square_roots_equation_l4025_402519


namespace elliot_reading_rate_l4025_402517

/-- Given a book with a certain number of pages, the number of pages read before a week,
    and the number of pages left after a week of reading, calculate the number of pages read per day. -/
def pages_per_day (total_pages : ℕ) (pages_read_before : ℕ) (pages_left : ℕ) : ℕ :=
  ((total_pages - pages_left) - pages_read_before) / 7

/-- Theorem stating that for Elliot's specific reading scenario, he reads 20 pages per day. -/
theorem elliot_reading_rate : pages_per_day 381 149 92 = 20 := by
  sorry

end elliot_reading_rate_l4025_402517


namespace arithmetic_sequence_sum_l4025_402561

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first, third, and fifth terms is 9. -/
def SumOdd (a : ℕ → ℝ) : Prop :=
  a 1 + a 3 + a 5 = 9

/-- The sum of the second, fourth, and sixth terms is 15. -/
def SumEven (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 = 15

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) 
  (h2 : SumOdd a) 
  (h3 : SumEven a) : 
  a 3 + a 4 = 8 := by
  sorry

end arithmetic_sequence_sum_l4025_402561


namespace expression_equals_zero_l4025_402552

theorem expression_equals_zero :
  (-1 : ℝ) ^ 2022 + |-2| - (1/2 : ℝ) ^ 0 - 2 * Real.tan (π/4) = 0 := by
  sorry

end expression_equals_zero_l4025_402552


namespace part_one_solution_part_two_solution_l4025_402532

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + a - 4| + x + 1

-- Part I
theorem part_one_solution :
  let a : ℝ := 2
  ∀ x : ℝ, f a x < 9 ↔ -6 < x ∧ x < 10/3 :=
sorry

-- Part II
theorem part_two_solution :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 2 → f a x ≤ (x + 2)^2) ↔ -3 ≤ a ∧ a ≤ 17/3 :=
sorry

end part_one_solution_part_two_solution_l4025_402532


namespace monkey_reaches_top_l4025_402571

/-- A monkey climbing a tree -/
def monkey_climb (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) : ℕ → ℕ
| 0 => 0
| (n + 1) => min tree_height (monkey_climb tree_height hop_distance slip_distance n + hop_distance - slip_distance)

theorem monkey_reaches_top (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) 
  (h1 : tree_height = 50)
  (h2 : hop_distance = 4)
  (h3 : slip_distance = 3)
  (h4 : hop_distance > slip_distance) :
  ∃ t : ℕ, monkey_climb tree_height hop_distance slip_distance t = tree_height ∧ t = 50 := by
  sorry

end monkey_reaches_top_l4025_402571


namespace elevator_stop_time_is_three_l4025_402550

/-- Represents the race to the top of a building --/
structure BuildingRace where
  stories : ℕ
  lola_time_per_story : ℕ
  elevator_time_per_story : ℕ
  total_time : ℕ

/-- Calculates the time the elevator stops on each floor --/
def elevator_stop_time (race : BuildingRace) : ℕ :=
  let lola_total_time := race.stories * race.lola_time_per_story
  let elevator_move_time := race.stories * race.elevator_time_per_story
  let total_stop_time := race.total_time - elevator_move_time
  total_stop_time / (race.stories - 1)

/-- The theorem stating that the elevator stops for 3 seconds on each floor --/
theorem elevator_stop_time_is_three (race : BuildingRace) 
    (h1 : race.stories = 20)
    (h2 : race.lola_time_per_story = 10)
    (h3 : race.elevator_time_per_story = 8)
    (h4 : race.total_time = 220) :
  elevator_stop_time race = 3 := by
  sorry

#eval elevator_stop_time { stories := 20, lola_time_per_story := 10, elevator_time_per_story := 8, total_time := 220 }

end elevator_stop_time_is_three_l4025_402550


namespace chord_length_l4025_402518

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

-- Define the line l
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 11 = 0}

-- Define the intersection points A and B
def intersection_points : Set (ℝ × ℝ) :=
  circle_C ∩ line_l

-- Theorem statement
theorem chord_length :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 := by
  sorry


end chord_length_l4025_402518


namespace cans_per_bag_l4025_402522

theorem cans_per_bag (total_bags : ℕ) (total_cans : ℕ) (h1 : total_bags = 8) (h2 : total_cans = 40) :
  total_cans / total_bags = 5 := by
  sorry

end cans_per_bag_l4025_402522


namespace basil_planter_problem_l4025_402509

theorem basil_planter_problem (total_seeds : Nat) (large_planters : Nat) (seeds_per_large : Nat) (seeds_per_small : Nat) :
  total_seeds = 200 →
  large_planters = 4 →
  seeds_per_large = 20 →
  seeds_per_small = 4 →
  (total_seeds - large_planters * seeds_per_large) / seeds_per_small = 30 := by
  sorry

end basil_planter_problem_l4025_402509


namespace vector_b_coordinates_l4025_402593

def vector_a : ℝ × ℝ := (3, -4)

def opposite_direction (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ w = (k * v.1, k * v.2)

theorem vector_b_coordinates :
  ∀ (b : ℝ × ℝ),
    opposite_direction vector_a b →
    Real.sqrt (b.1^2 + b.2^2) = 10 →
    b = (-6, 8) := by sorry

end vector_b_coordinates_l4025_402593


namespace skating_time_for_average_l4025_402537

def minutes_per_day_1 : ℕ := 80
def days_1 : ℕ := 4
def minutes_per_day_2 : ℕ := 105
def days_2 : ℕ := 3
def total_days : ℕ := 8
def target_average : ℕ := 95

theorem skating_time_for_average :
  (minutes_per_day_1 * days_1 + minutes_per_day_2 * days_2 + 125) / total_days = target_average :=
by sorry

end skating_time_for_average_l4025_402537


namespace unique_sequence_l4025_402538

theorem unique_sequence (n : ℕ) (h : n > 1) :
  ∃! (x : ℕ → ℕ), 
    (∀ k, k ∈ Finset.range (n - 1) → x k > 0) ∧ 
    (∀ i j, i < j ∧ j < n - 1 → x i < x j) ∧
    (∀ i, i ∈ Finset.range (n - 1) → x i + x (n - 1 - i) = 2 * n) ∧
    (∀ i j, i ∈ Finset.range (n - 1) ∧ j ∈ Finset.range (n - 1) ∧ x i + x j < 2 * n → 
      ∃ k, k ∈ Finset.range (n - 1) ∧ x i + x j = x k) ∧
    (∀ k, k ∈ Finset.range (n - 1) → x k = 2 * (k + 1)) :=
by
  sorry

end unique_sequence_l4025_402538


namespace train_crossing_time_l4025_402541

/-- Given a train and a platform with specific dimensions and time to pass,
    calculate the time it takes for the train to cross a stationary point. -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_pass_platform : ℝ) 
  (h1 : train_length = 600)
  (h2 : platform_length = 450)
  (h3 : time_to_pass_platform = 105) :
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 60 :=
by sorry

end train_crossing_time_l4025_402541


namespace digit_sum_properties_l4025_402527

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if two natural numbers have the same digits in a different order -/
def same_digits (m k : ℕ) : Prop := sorry

theorem digit_sum_properties (M K : ℕ) (h : same_digits M K) :
  (sum_of_digits (2 * M) = sum_of_digits (2 * K)) ∧
  (M % 2 = 0 → K % 2 = 0 → sum_of_digits (M / 2) = sum_of_digits (K / 2)) ∧
  (sum_of_digits (5 * M) = sum_of_digits (5 * K)) := by
  sorry

end digit_sum_properties_l4025_402527


namespace sinusoidal_period_l4025_402516

/-- 
Given a sinusoidal function y = a * sin(b * x + c) + d where a, b, c, and d are positive constants,
if the function completes five periods over an interval of 2π, then b = 5.
-/
theorem sinusoidal_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_periods : (2 * Real.pi) / b = (2 * Real.pi) / 5) : b = 5 := by
  sorry

end sinusoidal_period_l4025_402516


namespace infinitely_many_non_representable_l4025_402559

theorem infinitely_many_non_representable : 
  ∃ f : ℕ → ℤ, Function.Injective f ∧ 
    ∀ (k : ℕ) (a b c : ℕ), f k ≠ 2^a + 3^b - 5^c := by
  sorry

end infinitely_many_non_representable_l4025_402559


namespace f_two_roots_range_l4025_402534

/-- The cubic function f(x) = x^3 - 3x + 5 -/
def f (x : ℝ) : ℝ := x^3 - 3*x + 5

/-- Theorem stating the range of a for which f(x) = a has at least two distinct real roots -/
theorem f_two_roots_range :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f x = a ∧ f y = a) ↔ 3 ≤ a ∧ a ≤ 7 := by
  sorry

end f_two_roots_range_l4025_402534


namespace replaced_person_weight_l4025_402557

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (num_persons : ℕ) (avg_weight_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  new_person_weight - (num_persons * avg_weight_increase)

/-- Theorem stating the weight of the replaced person under the given conditions -/
theorem replaced_person_weight :
  weight_of_replaced_person 5 10.0 90 = 40.0 := by
  sorry

end replaced_person_weight_l4025_402557


namespace min_value_equality_l4025_402525

theorem min_value_equality (x y a : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) :
  (∀ x y, x + 2*y = 1 → (3/x + a/y ≥ 6*Real.sqrt 3)) ∧
  (∃ x y, x + 2*y = 1 ∧ 3/x + a/y = 6*Real.sqrt 3) →
  (∀ x y, 1/x + 2/y = 1 → (3*x + a*y ≥ 6*Real.sqrt 3)) ∧
  (∃ x y, 1/x + 2/y = 1 ∧ 3*x + a*y = 6*Real.sqrt 3) :=
by sorry

end min_value_equality_l4025_402525


namespace custom_op_result_l4025_402500

-- Define the custom operation ⊗
def customOp (A B : Set ℝ) : Set ℝ := (A ∪ B) \ (A ∩ B)

-- Define sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem statement
theorem custom_op_result :
  customOp M N = {x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (2 ≤ x ∧ x < 3)} := by sorry

end custom_op_result_l4025_402500


namespace fractional_equation_solution_l4025_402580

theorem fractional_equation_solution :
  ∃ (x : ℝ), (2 / (x - 2) - (2 * x) / (2 - x) = 1) ∧ (x - 2 ≠ 0) ∧ (2 - x ≠ 0) ∧ (x = -4) := by
  sorry

end fractional_equation_solution_l4025_402580


namespace sum_of_numbers_l4025_402558

def total_numbers (joyce xavier coraline jayden mickey yvonne : ℕ) : Prop :=
  xavier = 4 * joyce ∧
  coraline = xavier + 50 ∧
  jayden = coraline - 40 ∧
  mickey = jayden + 20 ∧
  yvonne = xavier + joyce ∧
  joyce = 30 ∧
  joyce + xavier + coraline + jayden + mickey + yvonne = 750

theorem sum_of_numbers :
  ∃ (joyce xavier coraline jayden mickey yvonne : ℕ),
    total_numbers joyce xavier coraline jayden mickey yvonne :=
by
  sorry

end sum_of_numbers_l4025_402558


namespace comparison_and_inequality_l4025_402594

theorem comparison_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^2 + b^2 ≥ 2*(2*a - b) - 5 ∧ 
  a^a * b^b ≥ (a*b)^((a+b)/2) ∧ 
  (a^a * b^b = (a*b)^((a+b)/2) ↔ a = b) := by
  sorry

end comparison_and_inequality_l4025_402594


namespace store_uniforms_l4025_402589

theorem store_uniforms (total_uniforms : ℕ) (additional_uniform : ℕ) : 
  total_uniforms = 927 → 
  additional_uniform = 1 → 
  ∃ (employees : ℕ), 
    employees > 1 ∧ 
    (total_uniforms + additional_uniform) % employees = 0 ∧ 
    total_uniforms % employees ≠ 0 ∧
    ∀ (n : ℕ), n > employees → (total_uniforms + additional_uniform) % n ≠ 0 ∨ total_uniforms % n = 0 →
    employees = 29 := by
sorry

end store_uniforms_l4025_402589


namespace kekai_garage_sale_earnings_l4025_402531

/-- Calculates the amount of money Kekai has left after a garage sale --/
def kekais_money (num_shirts : ℕ) (num_pants : ℕ) (shirt_price : ℕ) (pants_price : ℕ) (share_fraction : ℚ) : ℚ :=
  let total_earned := num_shirts * shirt_price + num_pants * pants_price
  (total_earned : ℚ) * (1 - share_fraction)

/-- Proves that Kekai has $10 left after the garage sale --/
theorem kekai_garage_sale_earnings : kekais_money 5 5 1 3 (1/2) = 10 := by
  sorry

end kekai_garage_sale_earnings_l4025_402531


namespace root_implies_inequality_l4025_402544

theorem root_implies_inequality (a b : ℝ) 
  (h : ∃ x, (x + a) * (x + b) = 9 ∧ x = a + b) : a * b ≤ 1 := by
  sorry

end root_implies_inequality_l4025_402544


namespace three_digit_cubes_divisible_by_eight_l4025_402504

theorem three_digit_cubes_divisible_by_eight :
  (∃! (s : Finset Nat), 
    (∀ n ∈ s, 100 ≤ n ∧ n ≤ 999 ∧ ∃ k, n = k^3 ∧ 8 ∣ n) ∧ 
    s.card = 2) :=
sorry

end three_digit_cubes_divisible_by_eight_l4025_402504


namespace lcm_of_12_18_24_l4025_402575

theorem lcm_of_12_18_24 : Nat.lcm 12 (Nat.lcm 18 24) = 72 := by
  sorry

end lcm_of_12_18_24_l4025_402575


namespace house_painting_time_l4025_402514

theorem house_painting_time (total_time joint_time john_time : ℝ) 
  (h1 : joint_time = 2.4)
  (h2 : john_time = 6)
  (h3 : 1 / total_time + 1 / john_time = 1 / joint_time) :
  total_time = 4 := by sorry

end house_painting_time_l4025_402514


namespace water_added_to_bowl_l4025_402556

theorem water_added_to_bowl (C : ℝ) (h1 : C > 0) : 
  (C / 2 + (14 - C / 2) = 0.7 * C) → (14 - C / 2 = 4) := by
  sorry

end water_added_to_bowl_l4025_402556


namespace odd_power_sum_divisibility_l4025_402510

theorem odd_power_sum_divisibility (k : ℕ) (x y : ℤ) :
  (∃ q : ℤ, x^(2*k-1) + y^(2*k-1) = (x+y) * q) →
  (∃ r : ℤ, x^(2*k+1) + y^(2*k+1) = (x+y) * r) :=
by sorry

end odd_power_sum_divisibility_l4025_402510


namespace currency_denominations_l4025_402528

/-- The number of different denominations that can be formed with a given number of coins/bills of three types -/
def total_denominations (fifty_cent : ℕ) (five_yuan : ℕ) (hundred_yuan : ℕ) : ℕ :=
  let single_denom := fifty_cent + five_yuan + hundred_yuan
  let double_denom := fifty_cent * five_yuan + five_yuan * hundred_yuan + hundred_yuan * fifty_cent
  let triple_denom := fifty_cent * five_yuan * hundred_yuan
  single_denom + double_denom + triple_denom

/-- Theorem stating that the total number of denominations with 3 fifty-cent coins, 
    6 five-yuan bills, and 4 one-hundred-yuan bills is 139 -/
theorem currency_denominations : 
  total_denominations 3 6 4 = 139 := by
  sorry

end currency_denominations_l4025_402528


namespace arithmetic_calculation_l4025_402515

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 - 10 = 188 := by
  sorry

end arithmetic_calculation_l4025_402515


namespace phone_plan_comparison_l4025_402581

/-- Represents a mobile phone plan with a monthly fee and a per-minute call charge. -/
structure PhonePlan where
  monthly_fee : ℝ
  per_minute_charge : ℝ

/-- Calculates the monthly bill for a given phone plan and call duration. -/
def monthly_bill (plan : PhonePlan) (duration : ℝ) : ℝ :=
  plan.monthly_fee + plan.per_minute_charge * duration

/-- Plan A with a monthly fee of 15 yuan and a call charge of 0.1 yuan per minute. -/
def plan_a : PhonePlan := ⟨15, 0.1⟩

/-- Plan B with no monthly fee and a call charge of 0.15 yuan per minute. -/
def plan_b : PhonePlan := ⟨0, 0.15⟩

theorem phone_plan_comparison :
  /- 1. Functional relationships are correct -/
  (∀ x, monthly_bill plan_a x = 15 + 0.1 * x) ∧
  (∀ x, monthly_bill plan_b x = 0.15 * x) ∧
  /- 2. For Plan A, a monthly bill of 50 yuan corresponds to 350 minutes -/
  (monthly_bill plan_a 350 = 50) ∧
  /- 3. For 280 minutes, Plan B is more cost-effective -/
  (monthly_bill plan_b 280 < monthly_bill plan_a 280) := by
  sorry

#eval monthly_bill plan_a 350  -- Should output 50
#eval monthly_bill plan_b 280  -- Should output 42
#eval monthly_bill plan_a 280  -- Should output 43

end phone_plan_comparison_l4025_402581


namespace area_diagonal_constant_for_specific_rectangle_l4025_402597

/-- Represents a rectangle with given ratio and perimeter -/
structure Rectangle where
  ratio : Rat
  perimeter : ℝ

/-- The constant k for which the area of the rectangle equals k * d^2, where d is the diagonal length -/
def area_diagonal_constant (rect : Rectangle) : ℝ :=
  sorry

theorem area_diagonal_constant_for_specific_rectangle :
  let rect : Rectangle := { ratio := 5/2, perimeter := 28 }
  area_diagonal_constant rect = 10/29 := by
  sorry

end area_diagonal_constant_for_specific_rectangle_l4025_402597


namespace sequence_property_l4025_402511

theorem sequence_property (a : ℕ → ℤ) (h1 : a 2 = 4)
  (h2 : ∀ n : ℕ, n ≥ 1 → (a (n + 1) - a n : ℚ) < 2^n + 1/2)
  (h3 : ∀ n : ℕ, n ≥ 1 → (a (n + 2) - a n : ℤ) > 3 * 2^n - 1) :
  a 2018 = 2^2018 :=
by sorry

end sequence_property_l4025_402511


namespace months_C_is_three_l4025_402539

/-- Represents the number of months C put his oxen for grazing -/
def months_C : ℕ := sorry

/-- Total rent of the pasture in rupees -/
def total_rent : ℕ := 280

/-- Number of oxen A put for grazing -/
def oxen_A : ℕ := 10

/-- Number of months A put his oxen for grazing -/
def months_A : ℕ := 7

/-- Number of oxen B put for grazing -/
def oxen_B : ℕ := 12

/-- Number of months B put his oxen for grazing -/
def months_B : ℕ := 5

/-- Number of oxen C put for grazing -/
def oxen_C : ℕ := 15

/-- C's share of rent in rupees -/
def rent_C : ℕ := 72

/-- Theorem stating that C put his oxen for grazing for 3 months -/
theorem months_C_is_three : months_C = 3 := by sorry

end months_C_is_three_l4025_402539


namespace brads_money_l4025_402583

theorem brads_money (total : ℚ) (josh_brad_ratio : ℚ) (josh_doug_ratio : ℚ) :
  total = 68 →
  josh_brad_ratio = 2 →
  josh_doug_ratio = 3/4 →
  ∃ (brad josh doug : ℚ),
    brad + josh + doug = total ∧
    josh = josh_brad_ratio * brad ∧
    josh = josh_doug_ratio * doug ∧
    brad = 12 :=
by sorry

end brads_money_l4025_402583


namespace debby_candy_eaten_l4025_402529

/-- Given that Debby initially had 12 pieces of candy and ended up with 3 pieces,
    prove that she ate 9 pieces. -/
theorem debby_candy_eaten (initial : ℕ) (final : ℕ) (eaten : ℕ) 
    (h1 : initial = 12) 
    (h2 : final = 3) 
    (h3 : initial = final + eaten) : eaten = 9 := by
  sorry

end debby_candy_eaten_l4025_402529


namespace maggis_cupcakes_l4025_402501

theorem maggis_cupcakes (cupcakes_per_package : ℕ) (cupcakes_eaten : ℕ) (cupcakes_left : ℕ) :
  cupcakes_per_package = 4 →
  cupcakes_eaten = 5 →
  cupcakes_left = 12 →
  ∃ (initial_packages : ℕ), 
    initial_packages * cupcakes_per_package = cupcakes_left + cupcakes_eaten ∧
    initial_packages = 4 :=
by sorry

end maggis_cupcakes_l4025_402501


namespace solutions_to_quartic_equation_l4025_402596

theorem solutions_to_quartic_equation :
  let S : Set ℂ := {x : ℂ | x^4 - 81 = 0}
  S = {3, -3, 3*I, -3*I} := by
  sorry

end solutions_to_quartic_equation_l4025_402596


namespace front_view_length_l4025_402584

theorem front_view_length
  (body_diagonal : ℝ)
  (side_view : ℝ)
  (top_view : ℝ)
  (h1 : body_diagonal = 5 * Real.sqrt 2)
  (h2 : side_view = 5)
  (h3 : top_view = Real.sqrt 34) :
  ∃ front_view : ℝ,
    front_view = Real.sqrt 41 ∧
    side_view ^ 2 + top_view ^ 2 + front_view ^ 2 = body_diagonal ^ 2 :=
by sorry

end front_view_length_l4025_402584


namespace min_moves_at_least_50_l4025_402512

/-- A 4x4 grid representing the puzzle state -/
def PuzzleState := Fin 4 → Fin 4 → Option (Fin 16)

/-- A move in the puzzle -/
inductive Move
| slide : Fin 4 → Fin 4 → Fin 4 → Fin 4 → Move
| jump  : Fin 4 → Fin 4 → Fin 4 → Fin 4 → Move

/-- Check if a PuzzleState is a valid magic square with sum 30 -/
def isMagicSquare (state : PuzzleState) : Prop := sorry

/-- Check if a move is valid for a given state -/
def isValidMove (state : PuzzleState) (move : Move) : Prop := sorry

/-- Apply a move to a state -/
def applyMove (state : PuzzleState) (move : Move) : PuzzleState := sorry

/-- The minimum number of moves required to solve the puzzle -/
def minMoves (initial : PuzzleState) : ℕ := sorry

/-- The theorem stating that the minimum number of moves is at least 50 -/
theorem min_moves_at_least_50 (initial : PuzzleState) : 
  minMoves initial ≥ 50 := by sorry

end min_moves_at_least_50_l4025_402512


namespace shark_sightings_multiple_l4025_402549

/-- The number of shark sightings in Daytona Beach -/
def daytona_sightings : ℕ := 26

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := 7

/-- The additional number of sightings in Daytona Beach beyond the multiple -/
def additional_sightings : ℕ := 5

/-- The theorem stating the multiple of shark sightings in Cape May compared to Daytona Beach -/
theorem shark_sightings_multiple :
  ∃ (x : ℚ), x * cape_may_sightings + additional_sightings = daytona_sightings ∧ x = 3 := by
  sorry

end shark_sightings_multiple_l4025_402549


namespace handball_final_score_l4025_402572

/-- Represents the score of a handball match -/
structure Score where
  home : ℕ
  visitors : ℕ

/-- Calculates the final score given the initial score and goals scored in the second half -/
def finalScore (initial : Score) (visitorGoals : ℕ) : Score :=
  { home := initial.home + 2 * visitorGoals,
    visitors := initial.visitors + visitorGoals }

/-- Theorem stating the final score of the handball match -/
theorem handball_final_score :
  ∀ (initial : Score) (visitorGoals : ℕ),
    initial.home = 9 →
    initial.visitors = 14 →
    let final := finalScore initial visitorGoals
    (final.home = final.visitors + 1) →
    final.home = 21 ∧ final.visitors = 20 := by
  sorry

#check handball_final_score

end handball_final_score_l4025_402572


namespace platform_length_platform_length_approx_l4025_402555

/-- The length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) : ℝ :=
  let train_speed := train_length / pole_time
  let platform_length := train_speed * platform_time - train_length
  platform_length

/-- The platform length is approximately 300 meters -/
theorem platform_length_approx :
  let result := platform_length 300 36 18
  ∃ ε > 0, abs (result - 300) < ε :=
sorry

end platform_length_platform_length_approx_l4025_402555


namespace inscribed_cube_volume_l4025_402591

/-- Represents a pyramid with an equilateral triangular base and isosceles lateral faces -/
structure Pyramid where
  base_side_length : ℝ
  lateral_side_length : ℝ
  (base_is_equilateral : base_side_length = 2)
  (lateral_is_isosceles : lateral_side_length = 3)

/-- Represents a cube inscribed in the pyramid -/
structure InscribedCube (p : Pyramid) where
  side_length : ℝ
  (base_on_pyramid_base : True)
  (top_vertices_touch_midpoints : True)

/-- The volume of the inscribed cube -/
def cube_volume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.side_length ^ 3

theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p) :
  cube_volume p c = (4 * Real.sqrt 2 - 3) ^ 3 := by
  sorry

end inscribed_cube_volume_l4025_402591


namespace quadratic_solution_set_l4025_402505

theorem quadratic_solution_set (b c : ℝ) : 
  (∀ x, x^2 + 2*b*x + c ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1) → b + c = -1 := by
  sorry

end quadratic_solution_set_l4025_402505


namespace mod_congruence_l4025_402523

theorem mod_congruence (m : ℕ) : 
  198 * 963 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 24 := by
  sorry

end mod_congruence_l4025_402523


namespace even_digits_base7_512_l4025_402547

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem: The number of even digits in the base-7 representation of 512₁₀ is 0 -/
theorem even_digits_base7_512 : countEvenDigits (toBase7 512) = 0 := by
  sorry

end even_digits_base7_512_l4025_402547


namespace imaginary_part_of_z_is_zero_l4025_402595

theorem imaginary_part_of_z_is_zero (z : ℂ) (h : z * (Complex.I + 1) = 2 / (Complex.I - 1)) : 
  z.im = 0 := by
  sorry

end imaginary_part_of_z_is_zero_l4025_402595


namespace simplify_expression_l4025_402520

theorem simplify_expression (a : ℝ) : 2*a*(3*a^2 - 4*a + 3) - 3*a^2*(2*a - 4) = 4*a^2 + 6*a := by
  sorry

end simplify_expression_l4025_402520


namespace smallest_n_divisors_not_multiple_of_ten_l4025_402563

def is_perfect_cube (m : ℕ) : Prop := ∃ k : ℕ, m = k^3

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k^2

def is_perfect_seventh (m : ℕ) : Prop := ∃ k : ℕ, m = k^7

def count_non_ten_divisors (n : ℕ) : ℕ := 
  (Finset.filter (fun d => ¬(10 ∣ d)) (Nat.divisors n)).card

theorem smallest_n_divisors_not_multiple_of_ten :
  ∃ n : ℕ, 
    (∀ m < n, ¬(is_perfect_cube (m / 2) ∧ is_perfect_square (m / 3) ∧ is_perfect_seventh (m / 5))) ∧
    is_perfect_cube (n / 2) ∧
    is_perfect_square (n / 3) ∧
    is_perfect_seventh (n / 5) ∧
    count_non_ten_divisors n = 52 := by
  sorry

end smallest_n_divisors_not_multiple_of_ten_l4025_402563


namespace matrix_inverse_l4025_402548

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

theorem matrix_inverse :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 2/46, 4/46]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end matrix_inverse_l4025_402548


namespace solution_to_linear_equation_l4025_402536

theorem solution_to_linear_equation :
  let x : ℝ := 4
  let y : ℝ := 2
  2 * x - y = 6 :=
by sorry

end solution_to_linear_equation_l4025_402536


namespace min_value_implies_m_l4025_402587

/-- Given a function f(x) = x + m / (x - 2) where x > 2 and m > 0,
    if the minimum value of f(x) is 6, then m = 4. -/
theorem min_value_implies_m (m : ℝ) (h_m_pos : m > 0) :
  (∀ x > 2, x + m / (x - 2) ≥ 6) ∧
  (∃ x > 2, x + m / (x - 2) = 6) →
  m = 4 := by
sorry

end min_value_implies_m_l4025_402587


namespace platform_length_calculation_l4025_402535

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is
    approximately 350.13 meters. -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 300 →
  time_platform = 39 →
  time_pole = 18 →
  ∃ (platform_length : ℝ), abs (platform_length - 350.13) < 0.01 :=
by
  sorry

end platform_length_calculation_l4025_402535


namespace question_always_truthful_l4025_402592

-- Define the types of residents
inductive ResidentType
| Knight
| Liar

-- Define the possible answers
inductive Answer
| Yes
| No

-- Define a function to represent the truth about having a crocodile
def hasCrocodile : ResidentType → Bool → Answer
| ResidentType.Knight, true => Answer.Yes
| ResidentType.Knight, false => Answer.No
| ResidentType.Liar, true => Answer.No
| ResidentType.Liar, false => Answer.Yes

-- Define the function that represents the response to the question
def responseToQuestion (resident : ResidentType) (hasCroc : Bool) : Answer :=
  hasCrocodile resident hasCroc

-- Theorem: The response to the question always gives the truthful answer
theorem question_always_truthful (resident : ResidentType) (hasCroc : Bool) :
  responseToQuestion resident hasCroc = hasCrocodile ResidentType.Knight hasCroc :=
by sorry

end question_always_truthful_l4025_402592


namespace correct_calculation_result_proof_correct_calculation_l4025_402579

theorem correct_calculation_result : ℤ → Prop :=
  fun x => (x + 9 = 30) → (x - 7 = 14)

-- The proof is omitted
theorem proof_correct_calculation : correct_calculation_result 21 := by
  sorry

end correct_calculation_result_proof_correct_calculation_l4025_402579


namespace carpet_square_cost_l4025_402503

/-- The cost of each carpet square given floor and carpet dimensions and total cost -/
theorem carpet_square_cost
  (floor_length : ℝ)
  (floor_width : ℝ)
  (square_side : ℝ)
  (total_cost : ℝ)
  (h1 : floor_length = 6)
  (h2 : floor_width = 10)
  (h3 : square_side = 2)
  (h4 : total_cost = 225) :
  (total_cost / ((floor_length * floor_width) / (square_side * square_side))) = 15 := by
  sorry

#check carpet_square_cost

end carpet_square_cost_l4025_402503


namespace field_ratio_is_two_to_one_l4025_402526

/-- Proves that the ratio of length to width of a rectangular field is 2:1 given specific conditions --/
theorem field_ratio_is_two_to_one (field_length field_width pond_side : ℝ) : 
  field_length = 80 →
  pond_side = 8 →
  field_length * field_width = 50 * (pond_side * pond_side) →
  field_length / field_width = 2 := by
sorry

end field_ratio_is_two_to_one_l4025_402526


namespace min_value_a_plus_5b_l4025_402533

theorem min_value_a_plus_5b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b + b^2 = b + 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x * y + y^2 = y + 1 → a + 5 * b ≤ x + 5 * y ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x * y + y^2 = y + 1 ∧ x + 5 * y = 7/2) :=
sorry

end min_value_a_plus_5b_l4025_402533


namespace isosceles_triangle_angle_bisector_length_l4025_402577

theorem isosceles_triangle_angle_bisector_length 
  (AB BC AC : ℝ) (h_isosceles : AC = BC) (h_base : AB = 5) (h_lateral : AC = 20) :
  let AD := Real.sqrt (AB * AC * (1 - BC / (AB + AC)))
  AD = 6 := by
  sorry

end isosceles_triangle_angle_bisector_length_l4025_402577


namespace concert_ticket_cost_l4025_402542

/-- The price of a child ticket -/
def child_ticket_price : ℝ := sorry

/-- The price of an adult ticket -/
def adult_ticket_price : ℝ := 2 * child_ticket_price

/-- The condition that 6 adult tickets and 5 child tickets cost $37.50 -/
axiom ticket_condition : 6 * adult_ticket_price + 5 * child_ticket_price = 37.50

/-- The theorem to prove -/
theorem concert_ticket_cost : 
  10 * adult_ticket_price + 8 * child_ticket_price = 61.78 := by sorry

end concert_ticket_cost_l4025_402542


namespace brownie_triangles_l4025_402573

theorem brownie_triangles (pan_length : ℝ) (pan_width : ℝ) 
                          (triangle_base : ℝ) (triangle_height : ℝ) :
  pan_length = 15 →
  pan_width = 24 →
  triangle_base = 3 →
  triangle_height = 4 →
  (pan_length * pan_width) / ((1/2) * triangle_base * triangle_height) = 60 := by
  sorry

end brownie_triangles_l4025_402573


namespace dog_ate_cost_l4025_402568

-- Define the given conditions
def total_slices : ℕ := 6
def total_cost : ℚ := 9
def mother_slices : ℕ := 2

-- Define the theorem
theorem dog_ate_cost : 
  (total_cost / total_slices) * (total_slices - mother_slices) = 6 := by
  sorry

end dog_ate_cost_l4025_402568


namespace sequence_formula_l4025_402590

theorem sequence_formula (n : ℕ+) (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h1 : ∀ k, S k = a k / 2 + 1 / a k - 1)
  (h2 : ∀ k, a k > 0) :
  a n = Real.sqrt (2 * n + 1) - Real.sqrt (2 * n - 1) :=
by sorry

end sequence_formula_l4025_402590


namespace lcm_gcf_relation_l4025_402513

theorem lcm_gcf_relation (n : ℕ) (h1 : Nat.lcm n 14 = 56) (h2 : Nat.gcd n 14 = 10) : n = 40 := by
  sorry

end lcm_gcf_relation_l4025_402513


namespace equation_solutions_may_days_l4025_402554

-- Define the interval [0°, 360°]
def angle_interval : Set ℝ := {x | 0 ≤ x ∧ x ≤ 360}

-- Define the equation cos³α - cosα = 0
def equation (α : ℝ) : Prop := Real.cos α ^ 3 - Real.cos α = 0

-- Define the day of the week as an enumeration
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define a function to get the day of the week for a given day in May
def day_in_may (day : ℕ) : DayOfWeek := sorry

-- Theorem 1: There are exactly 5 values of α in [0°, 360°] that satisfy cos³α - cosα = 0
theorem equation_solutions :
  ∃ (S : Finset ℝ), S.card = 5 ∧ (∀ α ∈ S, α ∈ angle_interval ∧ equation α) ∧
    (∀ α, α ∈ angle_interval → equation α → α ∈ S) :=
  sorry

-- Theorem 2: If the 5th day of May is Thursday, then the 16th day of May is Monday
theorem may_days :
  day_in_may 5 = DayOfWeek.Thursday → day_in_may 16 = DayOfWeek.Monday :=
  sorry

end equation_solutions_may_days_l4025_402554


namespace carnation_percentage_l4025_402553

/-- Represents a floral arrangement with different types of flowers -/
structure FloralArrangement where
  total : ℕ
  pink_roses : ℕ
  red_roses : ℕ
  white_roses : ℕ
  pink_carnations : ℕ
  red_carnations : ℕ
  white_carnations : ℕ

/-- Conditions for the floral arrangement -/
def valid_arrangement (f : FloralArrangement) : Prop :=
  -- Half of the pink flowers are roses
  f.pink_roses = f.pink_carnations
  -- One-third of the red flowers are carnations
  ∧ 3 * f.red_carnations = f.red_roses + f.red_carnations
  -- Three-fifths of the flowers are pink
  ∧ 5 * (f.pink_roses + f.pink_carnations) = 3 * f.total
  -- Total flowers equals sum of all flower types
  ∧ f.total = f.pink_roses + f.red_roses + f.white_roses + 
              f.pink_carnations + f.red_carnations + f.white_carnations

/-- Theorem: The percentage of carnations in a valid floral arrangement is 50% -/
theorem carnation_percentage (f : FloralArrangement) 
  (h : valid_arrangement f) : 
  (f.pink_carnations + f.red_carnations + f.white_carnations) * 2 = f.total := by
  sorry

#check carnation_percentage

end carnation_percentage_l4025_402553


namespace lars_bakery_production_l4025_402506

-- Define the baking rates and working hours
def bread_per_hour : ℕ := 10
def baguettes_per_two_hours : ℕ := 30
def working_hours : ℕ := 6

-- Define the function to calculate total breads per day
def total_breads_per_day : ℕ :=
  (bread_per_hour * working_hours) + (baguettes_per_two_hours * (working_hours / 2))

-- Theorem statement
theorem lars_bakery_production :
  total_breads_per_day = 150 := by sorry

end lars_bakery_production_l4025_402506


namespace set_intersection_equals_interval_l4025_402578

-- Define the sets M and N
def M : Set ℝ := {x | 2 * x^2 - 3 * x - 2 ≤ 0}
def N : Set ℝ := {x | x > 0 ∧ x ≠ 1}

-- Define the interval (0,1) ∪ (1,2]
def interval : Set ℝ := {x | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2)}

-- State the theorem
theorem set_intersection_equals_interval : M ∩ N = interval := by
  sorry

end set_intersection_equals_interval_l4025_402578


namespace f_f_two_equals_two_l4025_402570

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2/x

theorem f_f_two_equals_two : f (f 2) = 2 := by
  sorry

end f_f_two_equals_two_l4025_402570


namespace max_gcd_sum_1998_l4025_402598

theorem max_gcd_sum_1998 : ∃ (a b c : ℕ+), 
  (a + b + c : ℕ) = 1998 ∧ 
  Nat.gcd (Nat.gcd a.val b.val) c.val = 74 ∧ 
  0 < a.val ∧ a.val < b.val ∧ b.val ≤ c.val ∧ c.val < 2 * a.val := by
  sorry

end max_gcd_sum_1998_l4025_402598


namespace apple_pies_count_l4025_402560

theorem apple_pies_count (pecan_pies : ℕ) (total_rows : ℕ) (pies_per_row : ℕ) : 
  pecan_pies = 16 →
  total_rows = 6 →
  pies_per_row = 5 →
  ∃ (apple_pies : ℕ), apple_pies = total_rows * pies_per_row - pecan_pies ∧ apple_pies = 14 := by
  sorry

end apple_pies_count_l4025_402560
