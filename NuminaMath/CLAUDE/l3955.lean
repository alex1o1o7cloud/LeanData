import Mathlib

namespace congruence_solution_l3955_395591

theorem congruence_solution : ∃! n : ℕ, n < 47 ∧ (13 * n) % 47 = 15 % 47 := by
  sorry

end congruence_solution_l3955_395591


namespace jerrys_age_l3955_395585

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 20 → 
  mickey_age = 2 * jerry_age - 8 → 
  jerry_age = 14 := by
sorry

end jerrys_age_l3955_395585


namespace sum_of_digits_0_to_999_l3955_395547

/-- The sum of all digits of integers from 0 to 999 inclusive -/
def sumOfDigits : ℕ := sorry

/-- The range of integers we're considering -/
def integerRange : Set ℕ := { n | 0 ≤ n ∧ n ≤ 999 }

theorem sum_of_digits_0_to_999 : sumOfDigits = 13500 := by sorry

end sum_of_digits_0_to_999_l3955_395547


namespace sqrt_two_div_sqrt_eighteen_equals_one_third_l3955_395508

theorem sqrt_two_div_sqrt_eighteen_equals_one_third :
  Real.sqrt 2 / Real.sqrt 18 = 1 / 3 := by
  sorry

end sqrt_two_div_sqrt_eighteen_equals_one_third_l3955_395508


namespace x_range_for_quadratic_inequality_l3955_395501

theorem x_range_for_quadratic_inequality :
  ∀ x : ℝ,
  (∀ m : ℝ, |m| ≤ 2 → m * x^2 - 2*x - m + 1 < 0) →
  (-1 + Real.sqrt 7) / 2 < x ∧ x < (1 + Real.sqrt 3) / 2 :=
by sorry

end x_range_for_quadratic_inequality_l3955_395501


namespace reflection_theorem_l3955_395598

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 1)
  let p'' := (-p'.2, -p'.1)
  (p''.1, p''.2 + 1)

def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, 8)
def C : ℝ × ℝ := (7, 4)

theorem reflection_theorem :
  reflect_line (reflect_x C) = (-5, 8) := by sorry

end reflection_theorem_l3955_395598


namespace test_score_problem_l3955_395565

theorem test_score_problem (total_questions : ℕ) (correct_points : ℚ) (incorrect_penalty : ℚ) 
  (final_score : ℚ) (h1 : total_questions = 120) (h2 : correct_points = 1) 
  (h3 : incorrect_penalty = 1/4) (h4 : final_score = 100) : 
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_points - (total_questions - correct_answers) * incorrect_penalty = final_score ∧
    correct_answers = 104 := by
  sorry

end test_score_problem_l3955_395565


namespace isosceles_triangle_smallest_angle_measure_l3955_395575

theorem isosceles_triangle_smallest_angle_measure :
  ∀ (a b c : ℝ),
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Isosceles triangle condition
  c = 90 + 0.4 * 90  -- One angle is 40% larger than a right angle
  →
  a = 27 :=          -- One of the two smallest angles is 27°
by
  sorry

end isosceles_triangle_smallest_angle_measure_l3955_395575


namespace joan_change_l3955_395567

/-- The change Joan received after buying a cat toy and a cage -/
theorem joan_change (cat_toy_cost cage_cost bill_amount : ℚ) : 
  cat_toy_cost = 8.77 →
  cage_cost = 10.97 →
  bill_amount = 20 →
  bill_amount - (cat_toy_cost + cage_cost) = 0.26 := by
sorry

end joan_change_l3955_395567


namespace sum_of_hundredth_powers_divisibility_l3955_395512

/-- QuadraticTrinomial represents a quadratic trinomial with integer coefficients -/
structure QuadraticTrinomial where
  p : ℤ
  q : ℤ

/-- Condition for the discriminant to be positive -/
def has_positive_discriminant (t : QuadraticTrinomial) : Prop :=
  t.p^2 - 4*t.q > 0

/-- Condition for coefficients to be divisible by 5 -/
def coeffs_divisible_by_5 (t : QuadraticTrinomial) : Prop :=
  5 ∣ t.p ∧ 5 ∣ t.q

/-- The sum of the hundredth powers of the roots -/
noncomputable def sum_of_hundredth_powers (t : QuadraticTrinomial) : ℝ :=
  let α := (-t.p + Real.sqrt (t.p^2 - 4*t.q)) / 2
  let β := (-t.p - Real.sqrt (t.p^2 - 4*t.q)) / 2
  α^100 + β^100

/-- The main theorem -/
theorem sum_of_hundredth_powers_divisibility
  (t : QuadraticTrinomial)
  (h_pos : has_positive_discriminant t)
  (h_div : coeffs_divisible_by_5 t) :
  ∃ (k : ℤ), sum_of_hundredth_powers t = k * (5^50 : ℝ) ∧
  ∀ (n : ℕ), n > 50 → ¬∃ (k : ℤ), sum_of_hundredth_powers t = k * (5^n : ℝ) :=
sorry

end sum_of_hundredth_powers_divisibility_l3955_395512


namespace max_area_rectangle_in_345_triangle_l3955_395560

/-- The maximum area of a rectangle inscribed in a 3-4-5 right triangle -/
theorem max_area_rectangle_in_345_triangle : 
  ∃ (A : ℝ), A = 3 ∧ 
  ∀ (x y : ℝ), 
    0 ≤ x ∧ 0 ≤ y ∧ 
    (x ≤ 4 ∧ y ≤ 3 - (3/4) * x) ∨ (y ≤ 3 ∧ x ≤ 4 - (4/3) * y) →
    x * y ≤ A :=
by sorry

end max_area_rectangle_in_345_triangle_l3955_395560


namespace target_hit_probability_l3955_395514

theorem target_hit_probability (prob_A prob_B : ℝ) 
  (h_A : prob_A = 0.6) (h_B : prob_B = 0.5) : 
  1 - (1 - prob_A) * (1 - prob_B) = 0.8 := by
  sorry

end target_hit_probability_l3955_395514


namespace surrounding_circle_area_l3955_395526

/-- Given a circle of radius R surrounded by four equal circles, each touching 
    the given circle and each other, the area of one surrounding circle is πR²(3 + 2√2) -/
theorem surrounding_circle_area (R : ℝ) (R_pos : R > 0) : 
  ∃ (r : ℝ), 
    r > 0 ∧ 
    (R + r)^2 + (R + r)^2 = (2*r)^2 ∧ 
    r = R * (1 + Real.sqrt 2) ∧
    π * r^2 = π * R^2 * (3 + 2 * Real.sqrt 2) :=
by sorry

end surrounding_circle_area_l3955_395526


namespace rhombus_longer_diagonal_l3955_395594

/-- A rhombus with side length 65 and shorter diagonal 56 has a longer diagonal of length 118 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side = 65 → shorter_diagonal = 56 → longer_diagonal = 118 → 
  side^2 = (shorter_diagonal / 2)^2 + (longer_diagonal / 2)^2 :=
by sorry

end rhombus_longer_diagonal_l3955_395594


namespace quadratic_rewrite_product_l3955_395517

theorem quadratic_rewrite_product (a b c : ℤ) : 
  (∀ x : ℝ, 16 * x^2 - 40 * x - 72 = (a * x + b)^2 + c) → a * b = -20 := by
  sorry

end quadratic_rewrite_product_l3955_395517


namespace mutually_exclusive_events_l3955_395503

/-- A batch of products -/
structure Batch where
  good : ℕ
  defective : ℕ
  h_good : good > 2
  h_defective : defective > 2

/-- A sample of two items from a batch -/
structure Sample (b : Batch) where
  good : Fin 3
  defective : Fin 3
  h_sum : good.val + defective.val = 2

/-- Event: At least one defective product in the sample -/
def at_least_one_defective (s : Sample b) : Prop :=
  s.defective.val ≥ 1

/-- Event: All products in the sample are good -/
def all_good (s : Sample b) : Prop :=
  s.good.val = 2

/-- The main theorem: "At least one defective" and "All good" are mutually exclusive -/
theorem mutually_exclusive_events (b : Batch) :
  ∀ (s : Sample b), ¬(at_least_one_defective s ∧ all_good s) :=
by sorry

end mutually_exclusive_events_l3955_395503


namespace factorial_fraction_simplification_l3955_395532

theorem factorial_fraction_simplification (N : ℕ) (h : N ≥ 2) :
  (Nat.factorial (N - 2) * N * (N - 1)) / Nat.factorial (N + 2) = 1 / ((N + 1) * (N + 2)) := by
  sorry

end factorial_fraction_simplification_l3955_395532


namespace painted_area_is_33_l3955_395545

/-- Represents the arrangement of cubes -/
structure CubeArrangement where
  width : Nat
  length : Nat
  height : Nat
  total_cubes : Nat

/-- Calculates the total painted area for a given cube arrangement -/
def painted_area (arr : CubeArrangement) : Nat :=
  let top_area := arr.width * arr.length
  let side_area := 2 * (arr.width * arr.height + arr.length * arr.height)
  top_area + side_area

/-- The specific arrangement described in the problem -/
def problem_arrangement : CubeArrangement :=
  { width := 3
  , length := 3
  , height := 1
  , total_cubes := 14 }

/-- Theorem stating that the painted area for the given arrangement is 33 square meters -/
theorem painted_area_is_33 : painted_area problem_arrangement = 33 := by
  sorry

end painted_area_is_33_l3955_395545


namespace closest_to_99_times_9_l3955_395504

def options : List ℤ := [10000, 100, 100000, 1000, 10]

theorem closest_to_99_times_9 :
  ∀ x ∈ options, |99 * 9 - 1000| ≤ |99 * 9 - x| :=
sorry

end closest_to_99_times_9_l3955_395504


namespace absent_men_count_solve_work_scenario_l3955_395586

/-- Represents the work scenario with absences -/
structure WorkScenario where
  total_men : ℕ
  original_days : ℕ
  actual_days : ℕ
  absent_men : ℕ

/-- Calculates the total work in man-days -/
def total_work (s : WorkScenario) : ℕ := s.total_men * s.original_days

/-- Calculates the work done by remaining men -/
def remaining_work (s : WorkScenario) : ℕ := (s.total_men - s.absent_men) * s.actual_days

/-- Theorem stating that 8 men became absent -/
theorem absent_men_count (s : WorkScenario) 
  (h1 : s.total_men = 48)
  (h2 : s.original_days = 15)
  (h3 : s.actual_days = 18)
  (h4 : total_work s = remaining_work s) :
  s.absent_men = 8 := by
  sorry

/-- Main theorem proving the solution -/
theorem solve_work_scenario : 
  ∃ (s : WorkScenario), s.total_men = 48 ∧ s.original_days = 15 ∧ s.actual_days = 18 ∧ 
  total_work s = remaining_work s ∧ s.absent_men = 8 := by
  sorry

end absent_men_count_solve_work_scenario_l3955_395586


namespace square_area_from_vertices_l3955_395553

/-- The area of a square with adjacent vertices at (-1, 4) and (2, -3) is 58 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (-1, 4)
  let p2 : ℝ × ℝ := (2, -3)
  let distance_squared := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
  distance_squared = 58 := by sorry

end square_area_from_vertices_l3955_395553


namespace ship_power_at_6_knots_l3955_395578

-- Define the quadratic function
def H (a b c : ℝ) (v : ℝ) : ℝ := a * v^2 + b * v + c

-- State the theorem
theorem ship_power_at_6_knots 
  (a b c : ℝ) 
  (h1 : H a b c 5 = 300)
  (h2 : H a b c 7 = 780)
  (h3 : H a b c 9 = 1420) :
  H a b c 6 = 520 := by
  sorry

end ship_power_at_6_knots_l3955_395578


namespace smallest_number_game_l3955_395510

theorem smallest_number_game (alice_number : ℕ) (bob_number : ℕ) : 
  alice_number = 45 →
  (∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ bob_number) →
  5 ∣ bob_number →
  bob_number > 0 →
  (∀ n : ℕ, n > 0 → (∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ n) → 5 ∣ n → n ≥ bob_number) →
  bob_number = 15 := by
sorry

end smallest_number_game_l3955_395510


namespace mans_speed_with_stream_l3955_395589

/-- Given a man's rowing speed against the stream and in still water, 
    calculate his speed with the stream. -/
theorem mans_speed_with_stream 
  (speed_against_stream : ℝ) 
  (speed_still_water : ℝ) 
  (h1 : speed_against_stream = 4) 
  (h2 : speed_still_water = 11) : 
  speed_still_water + (speed_still_water - speed_against_stream) = 18 := by
  sorry

#check mans_speed_with_stream

end mans_speed_with_stream_l3955_395589


namespace hyperbola_parabola_shared_focus_l3955_395577

/-- The focus of a parabola y² = 8x is at (2, 0) -/
def parabola_focus : ℝ × ℝ := (2, 0)

/-- The equation of the hyperbola (x²/a² - y²/3 = 1) with a > 0 -/
def is_hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ x^2 / a^2 - y^2 / 3 = 1

/-- The focus of the hyperbola coincides with the focus of the parabola -/
def hyperbola_focus (a : ℝ) : ℝ × ℝ := parabola_focus

/-- Theorem: The value of 'a' for the hyperbola sharing a focus with the parabola is 1 -/
theorem hyperbola_parabola_shared_focus :
  ∃ (a : ℝ), is_hyperbola a (hyperbola_focus a).1 (hyperbola_focus a).2 ∧ a = 1 :=
sorry

end hyperbola_parabola_shared_focus_l3955_395577


namespace triangle_transformation_l3955_395506

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def transform (p : ℝ × ℝ) : ℝ × ℝ := 
  reflect_y (rotate_180 (reflect_x p))

theorem triangle_transformation :
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (4, 1)
  let C : ℝ × ℝ := (2, 3)
  (transform A = A) ∧ (transform B = B) ∧ (transform C = C) :=
sorry

end triangle_transformation_l3955_395506


namespace max_distance_to_origin_l3955_395543

theorem max_distance_to_origin : 
  let curve := {(x, y) : ℝ × ℝ | ∃ θ : ℝ, x = Real.sqrt 3 + Real.cos θ ∧ y = 1 + Real.sin θ}
  ∀ p ∈ curve, ∃ q ∈ curve, ∀ r ∈ curve, Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) ≥ Real.sqrt ((r.1 - 0)^2 + (r.2 - 0)^2) ∧
  Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) = 3 := by
sorry


end max_distance_to_origin_l3955_395543


namespace factor_expression_l3955_395588

theorem factor_expression (x : ℝ) : 100 * x^23 + 225 * x^46 = 25 * x^23 * (4 + 9 * x^23) := by
  sorry

end factor_expression_l3955_395588


namespace bill_apples_left_l3955_395505

/-- The number of apples Bill has left after distributing and baking -/
def apples_left (initial_apples : ℕ) (children : ℕ) (apples_per_teacher : ℕ) 
  (teachers_per_child : ℕ) (pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - (children * apples_per_teacher * teachers_per_child) - (pies * apples_per_pie)

/-- Theorem stating that Bill has 18 apples left -/
theorem bill_apples_left : 
  apples_left 50 2 3 2 2 10 = 18 := by sorry

end bill_apples_left_l3955_395505


namespace inequality_solution_set_l3955_395587

theorem inequality_solution_set (x : ℝ) : 
  (3/16 : ℝ) + |x - 5/32| < 7/32 ↔ x ∈ Set.Ioo (1/8 : ℝ) (3/16 : ℝ) := by
  sorry

end inequality_solution_set_l3955_395587


namespace phone_problem_solution_l3955_395599

/-- Represents a phone model with purchase and selling prices -/
structure PhoneModel where
  purchase_price : ℝ
  selling_price : ℝ

/-- The problem setup -/
def phone_problem : Prop :=
  let a : PhoneModel := ⟨3000, 3400⟩
  let b : PhoneModel := ⟨3500, 4000⟩
  ∃ (x y : ℕ),
    (x * a.purchase_price + y * b.purchase_price = 32000) ∧
    (x * (a.selling_price - a.purchase_price) + y * (b.selling_price - b.purchase_price) = 4400) ∧
    x = 6 ∧ y = 4

/-- The profit maximization problem -/
def profit_maximization : Prop :=
  let a : PhoneModel := ⟨3000, 3400⟩
  let b : PhoneModel := ⟨3500, 4000⟩
  ∃ (x : ℕ),
    x ≥ 10 ∧
    (30 - x) ≤ 2 * x ∧
    400 * x + 500 * (30 - x) = 14000 ∧
    ∀ (y : ℕ), y ≥ 10 → (30 - y) ≤ 2 * y → 400 * y + 500 * (30 - y) ≤ 14000

theorem phone_problem_solution : 
  phone_problem ∧ profit_maximization :=
sorry

end phone_problem_solution_l3955_395599


namespace largest_number_l3955_395596

theorem largest_number (S : Set ℝ) (hS : S = {-1, 0, 1, 1/3}) : 
  ∃ m ∈ S, ∀ x ∈ S, x ≤ m ∧ m = 1 := by
sorry

end largest_number_l3955_395596


namespace min_omega_for_translated_sine_l3955_395580

theorem min_omega_for_translated_sine (ω : ℝ) (h1 : ω > 0) :
  (∃ k : ℤ, ω * (3 * π / 4 - π / 4) = k * π) →
  (∀ ω' : ℝ, ω' > 0 → (∃ k : ℤ, ω' * (3 * π / 4 - π / 4) = k * π) → ω' ≥ ω) →
  ω = 2 := by
sorry

end min_omega_for_translated_sine_l3955_395580


namespace intersection_set_equality_l3955_395558

theorem intersection_set_equality : 
  let S := {α : ℝ | ∃ k : ℤ, α = k * π / 2 - π / 5 ∧ 0 < α ∧ α < π}
  S = {3 * π / 10, 4 * π / 5} := by
  sorry

end intersection_set_equality_l3955_395558


namespace greatest_common_divisor_of_98_and_n_l3955_395524

theorem greatest_common_divisor_of_98_and_n (n : ℕ) : 
  (∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ 
   {d : ℕ | d ∣ 98 ∧ d ∣ n} = {d1, d2, d3}) → 
  Nat.gcd 98 n = 49 := by
sorry

end greatest_common_divisor_of_98_and_n_l3955_395524


namespace perfect_square_trinomial_condition_l3955_395509

/-- A quadratic expression x^2 + bx + c is a perfect square trinomial if there exists a real number k such that x^2 + bx + c = (x + k)^2 for all x. -/
def IsPerfectSquareTrinomial (b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x^2 + b*x + c = (x + k)^2

/-- If x^2 - 8x + a is a perfect square trinomial, then a = 16. -/
theorem perfect_square_trinomial_condition (a : ℝ) :
  IsPerfectSquareTrinomial (-8) a → a = 16 := by
  sorry

end perfect_square_trinomial_condition_l3955_395509


namespace hyperbola_condition_l3955_395537

/-- A curve is a hyperbola if it can be represented by an equation of the form
    (x²/a²) - (y²/b²) = 1 or (y²/a²) - (x²/b²) = 1, where a and b are non-zero real numbers. -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧
    (∀ x y, f x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1) ∨
    (∀ x y, f x y ↔ (y^2 / a^2) - (x^2 / b^2) = 1)

/-- The curve represented by the equation x²/(k-3) - y²/(k+3) = 1 -/
def curve (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k - 3) - y^2 / (k + 3) = 1

theorem hyperbola_condition (k : ℝ) :
  (k > 3 → is_hyperbola (curve k)) ∧
  ¬(is_hyperbola (curve k) → k > 3) :=
sorry

end hyperbola_condition_l3955_395537


namespace pirate_treasure_sum_l3955_395573

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- The value of silver medallions in base 7 --/
def silverValue : List Nat := [6, 2, 3, 5]

/-- The value of precious gemstones in base 7 --/
def gemstonesValue : List Nat := [1, 6, 4, 3]

/-- The value of spices in base 7 --/
def spicesValue : List Nat := [6, 5, 6]

theorem pirate_treasure_sum :
  base7ToBase10 silverValue + base7ToBase10 gemstonesValue + base7ToBase10 spicesValue = 3485 := by
  sorry


end pirate_treasure_sum_l3955_395573


namespace area_of_triangle_hyperbola_triangle_area_l3955_395520

/-- A hyperbola with center at the origin, foci on the x-axis, and eccentricity √2 -/
structure Hyperbola where
  eccentricity : ℝ
  passes_through : ℝ × ℝ
  eccentricity_eq : eccentricity = Real.sqrt 2
  point_on_hyperbola : passes_through = (4, Real.sqrt 10)

/-- A point M on the hyperbola where MF₁ ⟂ MF₂ -/
structure PointM (h : Hyperbola) where
  point : ℝ × ℝ
  on_hyperbola : point ∈ {p : ℝ × ℝ | p.1^2 - p.2^2 = 6}
  perpendicular : ∃ (f₁ f₂ : ℝ × ℝ), f₁.2 = 0 ∧ f₂.2 = 0 ∧ 
    (point.1 - f₁.1) * (point.1 - f₂.1) + point.2 * point.2 = 0

/-- The theorem stating that the area of triangle F₁MF₂ is 6 -/
theorem area_of_triangle (h : Hyperbola) (m : PointM h) : ℝ :=
  6

/-- The main theorem to be proved -/
theorem hyperbola_triangle_area (h : Hyperbola) (m : PointM h) :
  area_of_triangle h m = 6 := by sorry

end area_of_triangle_hyperbola_triangle_area_l3955_395520


namespace range_of_a_for_inequality_l3955_395519

open Real

theorem range_of_a_for_inequality (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ a * x₀ - log x₀ < 0) → a < 1 / exp 1 :=
sorry

end range_of_a_for_inequality_l3955_395519


namespace total_cost_of_sarees_l3955_395521

/-- Calculates the final price of a saree after applying discounts -/
def apply_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (fun p d => p * (1 - d)) price

/-- Converts a price from one currency to INR -/
def convert_to_inr (price : ℝ) (rate : ℝ) : ℝ :=
  price * rate

/-- Applies sales tax to a price -/
def apply_sales_tax (price : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 + tax_rate)

/-- Theorem: The total cost of purchasing three sarees is 39421.08 INR -/
theorem total_cost_of_sarees : 
  let saree1_price : ℝ := 200
  let saree1_discounts : List ℝ := [0.20, 0.15, 0.05]
  let saree1_rate : ℝ := 75

  let saree2_price : ℝ := 150
  let saree2_discounts : List ℝ := [0.10, 0.07]
  let saree2_rate : ℝ := 100

  let saree3_price : ℝ := 180
  let saree3_discounts : List ℝ := [0.12]
  let saree3_rate : ℝ := 90

  let sales_tax : ℝ := 0.08

  let saree1_final := apply_sales_tax (convert_to_inr (apply_discounts saree1_price saree1_discounts) saree1_rate) sales_tax
  let saree2_final := apply_sales_tax (convert_to_inr (apply_discounts saree2_price saree2_discounts) saree2_rate) sales_tax
  let saree3_final := apply_sales_tax (convert_to_inr (apply_discounts saree3_price saree3_discounts) saree3_rate) sales_tax

  saree1_final + saree2_final + saree3_final = 39421.08 :=
by sorry


end total_cost_of_sarees_l3955_395521


namespace new_tax_rate_calculation_l3955_395542

theorem new_tax_rate_calculation (original_rate : ℝ) (income : ℝ) (savings : ℝ) : 
  original_rate = 0.46 → 
  income = 36000 → 
  savings = 5040 → 
  (income * original_rate - savings) / income = 0.32 := by
  sorry

end new_tax_rate_calculation_l3955_395542


namespace weight_comparison_l3955_395569

/-- Given the weights of Mildred, Carol, and Tom, prove statements about their combined weights -/
theorem weight_comparison (mildred_weight carol_weight tom_weight : ℕ) 
  (h1 : mildred_weight = 59)
  (h2 : carol_weight = 9)
  (h3 : tom_weight = 20) :
  let combined_weight := carol_weight + tom_weight
  (combined_weight = 29) ∧ 
  (mildred_weight = combined_weight + 30) := by
  sorry

end weight_comparison_l3955_395569


namespace flyers_left_to_hand_out_l3955_395574

theorem flyers_left_to_hand_out 
  (total_flyers : ℕ) 
  (jack_handed : ℕ) 
  (rose_handed : ℕ) 
  (h1 : total_flyers = 1236)
  (h2 : jack_handed = 120)
  (h3 : rose_handed = 320) :
  total_flyers - (jack_handed + rose_handed) = 796 :=
by sorry

end flyers_left_to_hand_out_l3955_395574


namespace triangle_angle_less_than_right_angle_l3955_395590

theorem triangle_angle_less_than_right_angle 
  (A B C : ℝ) (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 2/b = 1/a + 1/c) : B < π/2 :=
sorry

end triangle_angle_less_than_right_angle_l3955_395590


namespace min_sum_of_product_2310_l3955_395522

theorem min_sum_of_product_2310 (a b c : ℕ+) (h : a * b * c = 2310) :
  ∃ (x y z : ℕ+), x * y * z = 2310 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 42 :=
sorry

end min_sum_of_product_2310_l3955_395522


namespace sum_to_k_perfect_square_l3955_395584

def sum_to_k (k : ℕ) : ℕ := k * (k + 1) / 2

theorem sum_to_k_perfect_square (k : ℕ) :
  k ≤ 49 →
  (∃ n : ℕ, n < 100 ∧ sum_to_k k = n^2) ↔
  k = 1 ∨ k = 8 ∨ k = 49 := by
  sorry

end sum_to_k_perfect_square_l3955_395584


namespace max_discount_rate_l3955_395561

/-- Represents the maximum discount rate problem -/
theorem max_discount_rate
  (cost_price : ℝ)
  (original_price : ℝ)
  (min_profit_margin : ℝ)
  (h1 : cost_price = 4)
  (h2 : original_price = 5)
  (h3 : min_profit_margin = 0.1)
  : ∃ (max_discount : ℝ),
    max_discount = 0.12 ∧
    ∀ (discount : ℝ),
      discount ≤ max_discount →
      (original_price * (1 - discount) - cost_price) / cost_price ≥ min_profit_margin :=
sorry

end max_discount_rate_l3955_395561


namespace complex_exponential_sum_l3955_395581

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (-2/3 : ℂ) + (1/9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (-2/3 : ℂ) - (1/9 : ℂ) * Complex.I :=
by
  sorry

end complex_exponential_sum_l3955_395581


namespace tetrahedron_regularity_l3955_395536

-- Define a tetrahedron
structure Tetrahedron :=
  (A B C D : Point)

-- Define properties of the tetrahedron
def has_inscribed_sphere (t : Tetrahedron) : Prop := sorry

def sphere_touches_incenter (t : Tetrahedron) : Prop := sorry

def sphere_touches_orthocenter (t : Tetrahedron) : Prop := sorry

def sphere_touches_centroid (t : Tetrahedron) : Prop := sorry

def is_regular (t : Tetrahedron) : Prop := sorry

-- Theorem statement
theorem tetrahedron_regularity (t : Tetrahedron) :
  has_inscribed_sphere t ∧
  sphere_touches_incenter t ∧
  sphere_touches_orthocenter t ∧
  sphere_touches_centroid t →
  is_regular t :=
by sorry

end tetrahedron_regularity_l3955_395536


namespace complement_of_union_is_four_l3955_395502

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_is_four :
  (U \ (A ∪ B)) = {4} := by
  sorry

end complement_of_union_is_four_l3955_395502


namespace cost_increase_l3955_395559

theorem cost_increase (t b : ℝ) : 
  let original_cost := t * b^5
  let new_cost := (3*t) * (2*b)^5
  (new_cost / original_cost) * 100 = 9600 := by
sorry

end cost_increase_l3955_395559


namespace sum_20_is_850_l3955_395516

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums of the sequence
  sum_5 : S 5 = 10
  sum_10 : S 10 = 50

/-- The sum of the first 20 terms of the geometric sequence is 850 -/
theorem sum_20_is_850 (seq : GeometricSequence) : seq.S 20 = 850 := by
  sorry

end sum_20_is_850_l3955_395516


namespace trig_fraction_equality_l3955_395556

theorem trig_fraction_equality (α : ℝ) (h : (1 + Real.sin α) / Real.cos α = -1/2) :
  Real.cos α / (Real.sin α - 1) = 1/2 := by
  sorry

end trig_fraction_equality_l3955_395556


namespace cars_triangle_right_angle_l3955_395541

/-- Represents a car traveling on a triangular path -/
structure Car where
  speedAB : ℝ
  speedBC : ℝ
  speedCA : ℝ

/-- Represents a triangle with three cars traveling on its sides -/
structure TriangleWithCars where
  -- Lengths of the sides of the triangle
  ab : ℝ
  bc : ℝ
  ca : ℝ
  -- The three cars
  car1 : Car
  car2 : Car
  car3 : Car

/-- The theorem stating that if three cars travel on a triangle and return at the same time, 
    the angle ABC is 90 degrees -/
theorem cars_triangle_right_angle (t : TriangleWithCars) : 
  (t.ab / t.car1.speedAB + t.bc / t.car1.speedBC + t.ca / t.car1.speedCA = 
   t.ab / t.car2.speedAB + t.bc / t.car2.speedBC + t.ca / t.car2.speedCA) ∧
  (t.ab / t.car1.speedAB + t.bc / t.car1.speedBC + t.ca / t.car1.speedCA = 
   t.ab / t.car3.speedAB + t.bc / t.car3.speedBC + t.ca / t.car3.speedCA) ∧
  (t.car1.speedAB = 12) ∧ (t.car1.speedBC = 10) ∧ (t.car1.speedCA = 15) ∧
  (t.car2.speedAB = 15) ∧ (t.car2.speedBC = 15) ∧ (t.car2.speedCA = 10) ∧
  (t.car3.speedAB = 10) ∧ (t.car3.speedBC = 20) ∧ (t.car3.speedCA = 12) →
  ∃ (A B C : ℝ × ℝ), 
    let angleABC := Real.arccos ((t.ab^2 + t.bc^2 - t.ca^2) / (2 * t.ab * t.bc))
    angleABC = Real.pi / 2 := by
  sorry

end cars_triangle_right_angle_l3955_395541


namespace flour_to_add_l3955_395546

/-- Given the total amount of flour required for a recipe and the amount already added,
    this theorem proves that the amount of flour needed to be added is the difference
    between the total required and the amount already added. -/
theorem flour_to_add (total_flour : ℕ) (flour_added : ℕ) :
  total_flour ≥ flour_added →
  total_flour - flour_added = total_flour - flour_added := by
  sorry

#check flour_to_add

end flour_to_add_l3955_395546


namespace solution_set_f_geq_1_range_of_a_l3955_395518

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x ≥ a^2 - a - 2} = {a : ℝ | -1 ≤ a ∧ a ≤ 2} := by sorry

end solution_set_f_geq_1_range_of_a_l3955_395518


namespace no_solution_equations_l3955_395572

theorem no_solution_equations :
  (∀ x : ℝ, (x - 5)^2 ≠ -1) ∧
  (∀ x : ℝ, |2*x| + 3 ≠ 0) ∧
  (∃ x : ℝ, Real.sqrt (x + 3) - 1 = 0) ∧
  (∃ x : ℝ, Real.sqrt (4 - x) - 3 = 0) ∧
  (∃ x : ℝ, |2*x| - 4 = 0) :=
by sorry

end no_solution_equations_l3955_395572


namespace diamond_seven_three_l3955_395549

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ :=
  sorry

-- Axioms for the diamond operation
axiom diamond_zero (x : ℝ) : diamond x 0 = x
axiom diamond_comm (x y : ℝ) : diamond x y = diamond y x
axiom diamond_rec (x y : ℝ) : diamond (x + 2) y = diamond x y + y + 2

-- Theorem to prove
theorem diamond_seven_three : diamond 7 3 = 21 := by
  sorry

end diamond_seven_three_l3955_395549


namespace simplest_quadratic_radical_l3955_395563

/-- A quadratic radical is considered simpler if it cannot be further simplified by factoring out perfect squares or simplifying fractions. -/
def is_simplest_quadratic_radical (x : ℝ) (options : List ℝ) : Prop :=
  x ∈ options ∧ 
  (∀ y ∈ options, x ≠ y → ∃ (n : ℕ) (m : ℚ), n > 1 ∧ y = n • (Real.sqrt m) ∨ ∃ (a b : ℚ), b ≠ 1 ∧ y = (Real.sqrt a) / b)

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (Real.sqrt 7) [Real.sqrt 12, Real.sqrt 7, Real.sqrt (2/3), Real.sqrt 0.2] :=
sorry

end simplest_quadratic_radical_l3955_395563


namespace probability_one_suit_each_probability_calculation_correct_l3955_395557

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards drawn -/
def NumberOfDraws : ℕ := 4

/-- Represents the probability of drawing one card from each suit in four draws with replacement -/
def ProbabilityOneSuitEach : ℚ := 3 / 32

/-- Theorem stating that the probability of drawing one card from each suit
    in four draws with replacement from a standard 52-card deck is 3/32 -/
theorem probability_one_suit_each :
  (3 / 4 : ℚ) * (1 / 2 : ℚ) * (1 / 4 : ℚ) = ProbabilityOneSuitEach :=
by sorry

/-- Theorem stating that the calculated probability is correct -/
theorem probability_calculation_correct :
  ProbabilityOneSuitEach = (3 : ℚ) / 32 :=
by sorry

end probability_one_suit_each_probability_calculation_correct_l3955_395557


namespace inequality_solution_set_l3955_395535

theorem inequality_solution_set (x : ℝ) :
  (2 / (x^2 + 2*x + 1) + 4 / (x^2 + 8*x + 7) > 3/2) ↔ 
  (x < -7 ∨ (-7 < x ∧ x < -1) ∨ x > -1) :=
by sorry

end inequality_solution_set_l3955_395535


namespace alice_average_speed_l3955_395530

/-- Alice's cycling trip -/
theorem alice_average_speed :
  let distance1 : ℝ := 40  -- First segment distance in miles
  let speed1 : ℝ := 8      -- First segment speed in miles per hour
  let distance2 : ℝ := 20  -- Second segment distance in miles
  let speed2 : ℝ := 40     -- Second segment speed in miles per hour
  let total_distance : ℝ := distance1 + distance2
  let total_time : ℝ := distance1 / speed1 + distance2 / speed2
  let average_speed : ℝ := total_distance / total_time
  average_speed = 120 / 11
  := by sorry

end alice_average_speed_l3955_395530


namespace set_star_A_B_l3955_395513

-- Define the sets A and B
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define the set difference operation
def set_difference (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

-- Define the * operation
def set_star (X Y : Set ℝ) : Set ℝ := (set_difference X Y) ∪ (set_difference Y X)

-- State the theorem
theorem set_star_A_B :
  set_star A B = {x | -3 < x ∧ x < 0} ∪ {x | x > 3} := by
  sorry

end set_star_A_B_l3955_395513


namespace inequality_proof_l3955_395528

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end inequality_proof_l3955_395528


namespace mango_tree_count_l3955_395595

theorem mango_tree_count (mango_count coconut_count : ℕ) : 
  coconut_count = mango_count / 2 - 5 →
  mango_count + coconut_count = 85 →
  mango_count = 60 := by
sorry

end mango_tree_count_l3955_395595


namespace owl_money_problem_l3955_395533

theorem owl_money_problem (x : ℚ) : 
  (((3 * ((3 * ((3 * ((3 * x) - 50)) - 50)) - 50)) - 50) = 0) → 
  (x = 2000 / 81) := by
sorry

end owl_money_problem_l3955_395533


namespace least_possible_FG_l3955_395582

-- Define the triangle EFG
structure TriangleEFG where
  EF : ℝ
  EG : ℝ
  FG : ℝ

-- Define the triangle HFG
structure TriangleHFG where
  HF : ℝ
  HG : ℝ
  FG : ℝ

-- Define the shared triangle configuration
def SharedTriangles (t1 : TriangleEFG) (t2 : TriangleHFG) : Prop :=
  t1.FG = t2.FG ∧
  t1.EF = 7 ∧
  t1.EG = 15 ∧
  t2.HG = 10 ∧
  t2.HF = 25

-- Theorem statement
theorem least_possible_FG (t1 : TriangleEFG) (t2 : TriangleHFG) 
  (h : SharedTriangles t1 t2) : 
  ∃ (n : ℕ), n = 15 ∧ t1.FG = n ∧ 
  (∀ (m : ℕ), m < n → ¬(∃ (t1' : TriangleEFG) (t2' : TriangleHFG), 
    SharedTriangles t1' t2' ∧ t1'.FG = m)) :=
  sorry

end least_possible_FG_l3955_395582


namespace hyperbola_parameters_l3955_395527

/-- Hyperbola properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  eccentricity : ℝ
  vertex_to_asymptote : ℝ

/-- Theorem: Given a hyperbola with specific eccentricity and vertex-to-asymptote distance, prove its parameters -/
theorem hyperbola_parameters (h : Hyperbola) 
  (h_eccentricity : h.eccentricity = Real.sqrt 6 / 2)
  (h_vertex_to_asymptote : h.vertex_to_asymptote = 2 * Real.sqrt 6 / 3) :
  h.a = 2 * Real.sqrt 2 ∧ h.b = 2 := by
  sorry

#check hyperbola_parameters

end hyperbola_parameters_l3955_395527


namespace store_revenue_l3955_395579

def shirt_price : ℚ := 10
def jean_price : ℚ := 2 * shirt_price
def jacket_price : ℚ := 3 * jean_price
def sock_price : ℚ := 2

def shirt_quantity : ℕ := 20
def jean_quantity : ℕ := 10
def jacket_quantity : ℕ := 15
def sock_quantity : ℕ := 30

def jacket_discount : ℚ := 0.1
def sock_bulk_discount : ℚ := 0.2

def shirt_revenue : ℚ := (shirt_quantity / 2 : ℚ) * shirt_price
def jean_revenue : ℚ := (jean_quantity : ℚ) * jean_price
def jacket_revenue : ℚ := (jacket_quantity : ℚ) * jacket_price * (1 - jacket_discount)
def sock_revenue : ℚ := (sock_quantity : ℚ) * sock_price * (1 - sock_bulk_discount)

def total_revenue : ℚ := shirt_revenue + jean_revenue + jacket_revenue + sock_revenue

theorem store_revenue : total_revenue = 1158 := by sorry

end store_revenue_l3955_395579


namespace father_daughter_ages_l3955_395551

theorem father_daughter_ages (father daughter : ℕ) : 
  father = 4 * daughter ∧ 
  father + 20 = 2 * (daughter + 20) → 
  father = 40 ∧ daughter = 10 := by
sorry

end father_daughter_ages_l3955_395551


namespace total_weight_of_good_fruits_l3955_395500

/-- Calculates the total weight in kilograms of fruits in good condition --/
def totalWeightOfGoodFruits (
  oranges bananas apples avocados grapes pineapples : ℕ
) (
  rottenOrangesPercent rottenBananasPercent rottenApplesPercent
  rottenAvocadosPercent rottenGrapesPercent rottenPineapplesPercent : ℚ
) (
  orangeWeight bananaWeight appleWeight avocadoWeight grapeWeight pineappleWeight : ℚ
) : ℚ :=
  let goodOranges := oranges - (oranges * rottenOrangesPercent).floor
  let goodBananas := bananas - (bananas * rottenBananasPercent).floor
  let goodApples := apples - (apples * rottenApplesPercent).floor
  let goodAvocados := avocados - (avocados * rottenAvocadosPercent).floor
  let goodGrapes := grapes - (grapes * rottenGrapesPercent).floor
  let goodPineapples := pineapples - (pineapples * rottenPineapplesPercent).floor

  (goodOranges * orangeWeight + goodBananas * bananaWeight +
   goodApples * appleWeight + goodAvocados * avocadoWeight +
   goodGrapes * grapeWeight + goodPineapples * pineappleWeight) / 1000

/-- The total weight of fruits in good condition is 204.585kg --/
theorem total_weight_of_good_fruits :
  totalWeightOfGoodFruits
    600 400 300 200 100 50
    (15/100) (5/100) (8/100) (10/100) (3/100) (20/100)
    150 120 100 80 5 1000 = 204585/1000 := by
  sorry

end total_weight_of_good_fruits_l3955_395500


namespace largest_prime_divisor_l3955_395538

/-- Converts a base 7 number to decimal --/
def toDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- The number in base 7 --/
def base7Number : List Nat := [1, 2, 0, 2, 1, 0, 1, 2]

/-- The decimal representation of the base 7 number --/
def decimalNumber : Nat := toDecimal base7Number

/-- Predicate to check if a number is prime --/
def isPrime (n : Nat) : Prop := sorry

theorem largest_prime_divisor :
  ∃ (p : Nat), isPrime p ∧ p ∣ decimalNumber ∧ 
  ∀ (q : Nat), isPrime q → q ∣ decimalNumber → q ≤ p ∧ p = 397 := by
  sorry

end largest_prime_divisor_l3955_395538


namespace binomial_200_200_l3955_395548

theorem binomial_200_200 : Nat.choose 200 200 = 1 := by
  sorry

end binomial_200_200_l3955_395548


namespace five_lapping_points_l3955_395552

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  initialPosition : ℝ

/-- The circular track model -/
def CircularTrack := Unit

/-- Calculates the number of lapping points on a circular track -/
def numberOfLappingPoints (track : CircularTrack) (a b : Runner) : ℕ :=
  sorry

theorem five_lapping_points (track : CircularTrack) (a b : Runner) :
  a.speed > 0 ∧ b.speed > 0 ∧
  a.initialPosition = b.initialPosition + 10 ∧
  b.speed * 22 = a.speed * 32 →
  numberOfLappingPoints track a b = 5 :=
sorry

end five_lapping_points_l3955_395552


namespace horizontal_figure_area_l3955_395570

/-- Represents a horizontally placed figure with specific properties -/
structure HorizontalFigure where
  /-- The oblique section diagram is an isosceles trapezoid -/
  is_isosceles_trapezoid : Bool
  /-- The base angle of the trapezoid is 45° -/
  base_angle : ℝ
  /-- The length of the legs of the trapezoid -/
  leg_length : ℝ
  /-- The length of the upper base of the trapezoid -/
  upper_base_length : ℝ

/-- Calculates the area of the original plane figure -/
def area (fig : HorizontalFigure) : ℝ :=
  sorry

/-- Theorem stating the area of the original plane figure -/
theorem horizontal_figure_area (fig : HorizontalFigure) 
  (h1 : fig.is_isosceles_trapezoid = true)
  (h2 : fig.base_angle = π / 4)
  (h3 : fig.leg_length = 1)
  (h4 : fig.upper_base_length = 1) :
  area fig = 2 + Real.sqrt 2 :=
sorry

end horizontal_figure_area_l3955_395570


namespace adam_figurines_count_l3955_395592

/-- Adam's wood carving shop problem -/
theorem adam_figurines_count :
  -- Define the number of figurines per block for each wood type
  let basswood_figurines : ℕ := 3
  let butternut_figurines : ℕ := 4
  let aspen_figurines : ℕ := 2 * basswood_figurines
  let oak_figurines : ℕ := 5
  let cherry_figurines : ℕ := 7

  -- Define the number of blocks for each wood type
  let basswood_blocks : ℕ := 25
  let butternut_blocks : ℕ := 30
  let aspen_blocks : ℕ := 35
  let oak_blocks : ℕ := 40
  let cherry_blocks : ℕ := 45

  -- Calculate total figurines
  let total_figurines : ℕ := 
    basswood_blocks * basswood_figurines +
    butternut_blocks * butternut_figurines +
    aspen_blocks * aspen_figurines +
    oak_blocks * oak_figurines +
    cherry_blocks * cherry_figurines

  -- Prove that the total number of figurines is 920
  total_figurines = 920 := by
  sorry


end adam_figurines_count_l3955_395592


namespace max_value_d_l3955_395583

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10) 
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) : 
  d ≤ (5 + Real.sqrt 105) / 2 := by
  sorry

end max_value_d_l3955_395583


namespace officer_selection_count_l3955_395550

/-- The number of members in the club -/
def clubSize : ℕ := 12

/-- The number of officers to be elected -/
def officerCount : ℕ := 5

/-- Calculates the number of ways to select distinct officers from club members -/
def officerSelections (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else (List.range k).foldl (fun acc i => acc * (n - i)) 1

/-- Theorem stating the number of ways to select 5 distinct officers from 12 members -/
theorem officer_selection_count :
  officerSelections clubSize officerCount = 95040 := by
  sorry

end officer_selection_count_l3955_395550


namespace tracy_art_fair_sales_l3955_395555

theorem tracy_art_fair_sales : 
  let total_customers : ℕ := 20
  let group1_customers : ℕ := 4
  let group1_paintings_per_customer : ℕ := 2
  let group2_customers : ℕ := 12
  let group2_paintings_per_customer : ℕ := 1
  let group3_customers : ℕ := 4
  let group3_paintings_per_customer : ℕ := 4
  let total_paintings_sold := 
    group1_customers * group1_paintings_per_customer +
    group2_customers * group2_paintings_per_customer +
    group3_customers * group3_paintings_per_customer
  total_customers = group1_customers + group2_customers + group3_customers →
  total_paintings_sold = 36 := by
sorry


end tracy_art_fair_sales_l3955_395555


namespace remainder_of_power_minus_seven_l3955_395571

theorem remainder_of_power_minus_seven (n : Nat) : (10^23 - 7) % 6 = 3 := by
  sorry

end remainder_of_power_minus_seven_l3955_395571


namespace range_of_a_l3955_395523

/-- Proposition p: The equation a²x² + ax - 2 = 0 has a solution in the interval [-1, 1] -/
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0

/-- Proposition q: There is only one real number x that satisfies x² + 2ax + 2a ≤ 0 -/
def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

/-- If both p and q are false, then -1 < a < 0 or 0 < a < 1 -/
theorem range_of_a (a : ℝ) : ¬(p a) ∧ ¬(q a) → (-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1) := by
  sorry

end range_of_a_l3955_395523


namespace f_positive_at_one_f_solution_set_l3955_395507

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

-- Theorem 1
theorem f_positive_at_one (a : ℝ) :
  f a 1 > 0 ↔ a ∈ Set.Ioo (3 - 2 * Real.sqrt 3) (3 + 2 * Real.sqrt 3) :=
sorry

-- Theorem 2
theorem f_solution_set (a b : ℝ) :
  (∀ x, f a x > b ↔ x ∈ Set.Ioo (-1) 3) ↔
  ((a = 3 - Real.sqrt 3 ∨ a = 3 + Real.sqrt 3) ∧ b = -3) :=
sorry

end f_positive_at_one_f_solution_set_l3955_395507


namespace cylinder_surface_area_l3955_395564

/-- Given a cylinder whose lateral surface unfolds into a rectangle with sides of length 6π and 4π,
    its total surface area is either 24π² + 18π or 24π² + 8π. -/
theorem cylinder_surface_area (h : ℝ) (r : ℝ) :
  (h = 6 * Real.pi ∧ 2 * Real.pi * r = 4 * Real.pi) ∨ 
  (h = 4 * Real.pi ∧ 2 * Real.pi * r = 6 * Real.pi) →
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 18 * Real.pi ∨
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 8 * Real.pi :=
by sorry

end cylinder_surface_area_l3955_395564


namespace apartment_utilities_cost_l3955_395539

/-- Proves that the utilities cost for an apartment is $114 given specific conditions -/
theorem apartment_utilities_cost 
  (rent : ℕ) 
  (groceries : ℕ) 
  (one_roommate_payment : ℕ) 
  (h_rent : rent = 1100)
  (h_groceries : groceries = 300)
  (h_one_roommate : one_roommate_payment = 757)
  (h_equal_split : ∀ total_cost, one_roommate_payment * 2 = total_cost) :
  ∃ utilities : ℕ, utilities = 114 ∧ rent + utilities + groceries = one_roommate_payment * 2 :=
by sorry

end apartment_utilities_cost_l3955_395539


namespace natural_number_representation_l3955_395593

theorem natural_number_representation (n : ℕ) : 
  ∃ (x y : ℕ), n = x^3 / y^4 := by sorry

end natural_number_representation_l3955_395593


namespace square_property_fourth_power_property_smallest_square_smallest_fourth_power_sum_is_1130_l3955_395544

/-- The smallest positive integer x such that 720x is a perfect square -/
def smallest_square_factor : ℕ := 5

/-- The smallest positive integer y such that 720y is a perfect fourth power -/
def smallest_fourth_power_factor : ℕ := 1125

/-- 720 * smallest_square_factor is a perfect square -/
theorem square_property : ∃ (n : ℕ), 720 * smallest_square_factor = n^2 := by sorry

/-- 720 * smallest_fourth_power_factor is a perfect fourth power -/
theorem fourth_power_property : ∃ (n : ℕ), 720 * smallest_fourth_power_factor = n^4 := by sorry

/-- smallest_square_factor is the smallest positive integer with the square property -/
theorem smallest_square :
  ∀ (k : ℕ), k > 0 ∧ k < smallest_square_factor → ¬∃ (n : ℕ), 720 * k = n^2 := by sorry

/-- smallest_fourth_power_factor is the smallest positive integer with the fourth power property -/
theorem smallest_fourth_power :
  ∀ (k : ℕ), k > 0 ∧ k < smallest_fourth_power_factor → ¬∃ (n : ℕ), 720 * k = n^4 := by sorry

/-- The sum of smallest_square_factor and smallest_fourth_power_factor -/
def sum_of_factors : ℕ := smallest_square_factor + smallest_fourth_power_factor

/-- The sum of the factors is 1130 -/
theorem sum_is_1130 : sum_of_factors = 1130 := by sorry

end square_property_fourth_power_property_smallest_square_smallest_fourth_power_sum_is_1130_l3955_395544


namespace percentage_problem_l3955_395554

theorem percentage_problem (P : ℝ) : 
  (0.1 * 0.3 * (P / 100) * 6000 = 90) → P = 50 := by
sorry

end percentage_problem_l3955_395554


namespace binomial_8_choose_5_l3955_395597

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end binomial_8_choose_5_l3955_395597


namespace tunnel_length_l3955_395566

/-- Calculates the length of a tunnel given train parameters and transit time -/
theorem tunnel_length
  (train_length : Real)
  (train_speed_kmh : Real)
  (transit_time_min : Real)
  (h1 : train_length = 100)
  (h2 : train_speed_kmh = 72)
  (h3 : transit_time_min = 2.5) :
  let train_speed_ms : Real := train_speed_kmh * 1000 / 3600
  let transit_time_s : Real := transit_time_min * 60
  let total_distance : Real := train_speed_ms * transit_time_s
  let tunnel_length_m : Real := total_distance - train_length
  let tunnel_length_km : Real := tunnel_length_m / 1000
  tunnel_length_km = 2.9 := by
sorry

end tunnel_length_l3955_395566


namespace right_triangle_and_modular_inverse_l3955_395576

theorem right_triangle_and_modular_inverse :
  -- Define the sides of the triangle
  let a : ℕ := 15
  let b : ℕ := 112
  let c : ℕ := 113
  -- Define the modulus
  let m : ℕ := 2799
  -- Define the number we're finding the inverse for
  let x : ℕ := 225
  -- Condition: a, b, c form a right triangle
  (a^2 + b^2 = c^2) →
  -- Conclusion: 1 is the multiplicative inverse of x modulo m
  (1 * x) % m = 1 := by
sorry

end right_triangle_and_modular_inverse_l3955_395576


namespace work_completion_time_l3955_395515

/-- Given that:
  * p can complete the work in 20 days
  * p and q work together for 2 days
  * After 2 days of working together, 0.7 of the work is left
  Prove that q can complete the work alone in 10 days -/
theorem work_completion_time (p_time q_time : ℝ) (h1 : p_time = 20) 
  (h2 : 2 * (1 / p_time + 1 / q_time) = 0.3) : q_time = 10 := by
  sorry


end work_completion_time_l3955_395515


namespace right_triangle_leg_sum_l3955_395534

theorem right_triangle_leg_sum : ∃ (a b : ℕ), 
  (a + 1 = b) ∧                -- legs are consecutive whole numbers
  (a^2 + b^2 = 41^2) ∧         -- Pythagorean theorem with hypotenuse 41
  (a + b = 57) :=              -- sum of legs is 57
sorry

end right_triangle_leg_sum_l3955_395534


namespace triangle_relation_l3955_395562

-- Define the triangles and their properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- State the theorem
theorem triangle_relation (abc : Triangle) (a'b'c' : Triangle) 
  (h1 : abc.angleB = a'b'c'.angleB) 
  (h2 : abc.angleA + a'b'c'.angleA = π) : 
  abc.a * a'b'c'.a = abc.b * a'b'c'.b + abc.c * a'b'c'.c := by
  sorry


end triangle_relation_l3955_395562


namespace imaginary_part_of_complex_fraction_l3955_395568

theorem imaginary_part_of_complex_fraction : Complex.im (5 * Complex.I / (1 + 2 * Complex.I)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l3955_395568


namespace quadratic_equation_linear_term_l3955_395529

theorem quadratic_equation_linear_term 
  (m : ℝ) 
  (h : 2 * m = 6) : 
  ∃ (a b c : ℝ), 
    a * x^2 + b * x + c = 0 ∧ 
    c = 6 ∧ 
    b = -1 := by
  sorry

end quadratic_equation_linear_term_l3955_395529


namespace cubic_roots_sum_l3955_395511

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 15*a^2 + 22*a - 8 = 0) →
  (b^3 - 15*b^2 + 22*b - 8 = 0) →
  (c^3 - 15*c^2 + 22*c - 8 = 0) →
  (a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 181/9) :=
by sorry

end cubic_roots_sum_l3955_395511


namespace existence_of_a_values_l3955_395531

theorem existence_of_a_values (n : ℕ) (x : Fin n → Fin n → ℝ) 
  (h : ∀ (i j k : Fin n), x i j + x j k + x k i = 0) :
  ∃ (a : Fin n → ℝ), ∀ (i j : Fin n), x i j = a i - a j := by
  sorry

end existence_of_a_values_l3955_395531


namespace nineteen_to_binary_l3955_395540

-- Define a function to convert decimal to binary
def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec toBinary (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else toBinary (m / 2) ((m % 2) :: acc)
    toBinary n []

-- State the theorem
theorem nineteen_to_binary :
  decimalToBinary 19 = [1, 0, 0, 1, 1] := by
  sorry

end nineteen_to_binary_l3955_395540


namespace heath_age_l3955_395525

theorem heath_age (heath_age jude_age : ℕ) : 
  jude_age = 2 →
  heath_age + 5 = 3 * (jude_age + 5) →
  heath_age = 16 := by
sorry

end heath_age_l3955_395525
