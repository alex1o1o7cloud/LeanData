import Mathlib

namespace shooting_competition_score_l3270_327015

theorem shooting_competition_score 
  (team_size : ℕ) 
  (best_score : ℕ) 
  (hypothetical_best_score : ℕ) 
  (hypothetical_average : ℕ) 
  (h1 : team_size = 8)
  (h2 : best_score = 85)
  (h3 : hypothetical_best_score = 92)
  (h4 : hypothetical_average = 84)
  (h5 : hypothetical_average * team_size = 
        (hypothetical_best_score - best_score) + total_score) :
  total_score = 665 :=
by
  sorry

#check shooting_competition_score

end shooting_competition_score_l3270_327015


namespace circle_tangents_l3270_327091

-- Define the circles
def circle_C (m : ℝ) (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8 - m
def circle_D (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 1

-- Define the property of having three common tangents
def has_three_common_tangents (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (circle_C m x1 y1 ∧ circle_D x1 y1) ∧
    (circle_C m x2 y2 ∧ circle_D x2 y2) ∧
    (circle_C m x3 y3 ∧ circle_D x3 y3) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3)

-- Theorem statement
theorem circle_tangents (m : ℝ) :
  has_three_common_tangents m → m = -8 :=
by sorry

end circle_tangents_l3270_327091


namespace dress_making_hours_l3270_327040

def total_fabric : ℕ := 56
def fabric_per_dress : ℕ := 4
def hours_per_dress : ℕ := 3

theorem dress_making_hours : 
  (total_fabric / fabric_per_dress) * hours_per_dress = 42 := by
  sorry

end dress_making_hours_l3270_327040


namespace term_free_of_x_l3270_327079

theorem term_free_of_x (m n k : ℕ) : 
  (∃ r : ℕ, r ≤ k ∧ m * k - (m + n) * r = 0) ↔ (m * k) % (m + n) = 0 := by
  sorry

end term_free_of_x_l3270_327079


namespace square_field_area_l3270_327083

/-- Given a square field where a horse takes 4 hours to run around it at 20 km/h, 
    prove that the area of the field is 400 km². -/
theorem square_field_area (s : ℝ) (h : s > 0) : 
  (4 * s = 20 * 4) → s^2 = 400 := by sorry

end square_field_area_l3270_327083


namespace sqrt_meaningful_range_l3270_327077

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) → x ≥ 1 := by sorry

end sqrt_meaningful_range_l3270_327077


namespace chessboard_tiling_l3270_327024

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Represents a polyomino -/
structure Polyomino :=
  (width : ℕ)
  (height : ℕ)

/-- Represents an L-shaped polyomino -/
def LPolyomino : Polyomino :=
  ⟨2, 2⟩

/-- Checks if a chessboard can be tiled with a given polyomino -/
def can_tile (board : Chessboard) (tile : Polyomino) : Prop :=
  ∃ n : ℕ, board.size * board.size = n * (tile.width * tile.height)

theorem chessboard_tiling (board : Chessboard) :
  board.size = 9 →
  ¬(can_tile board ⟨2, 1⟩) ∧
  (can_tile board ⟨3, 1⟩) ∧
  (can_tile board LPolyomino) :=
sorry

end chessboard_tiling_l3270_327024


namespace arithmetic_sequence_middle_term_l3270_327062

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 9 = 10) :
  a 5 = 5 :=
sorry

end arithmetic_sequence_middle_term_l3270_327062


namespace correct_calculation_l3270_327099

theorem correct_calculation (a b : ℝ) : 7 * a * b - 6 * a * b = a * b := by
  sorry

end correct_calculation_l3270_327099


namespace polynomial_factorization_l3270_327039

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 12) * (x^2 + 6*x + 3) := by
  sorry

end polynomial_factorization_l3270_327039


namespace problem_statement_l3270_327059

theorem problem_statement (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end problem_statement_l3270_327059


namespace coefficient_of_x_squared_l3270_327087

theorem coefficient_of_x_squared (k : ℝ) : 
  k = 1.7777777777777777 → 2 * k = 3.5555555555555554 := by
  sorry

end coefficient_of_x_squared_l3270_327087


namespace alcohol_fraction_in_mixture_l3270_327032

theorem alcohol_fraction_in_mixture (alcohol_water_ratio : ℚ) :
  alcohol_water_ratio = 2 / 3 →
  (alcohol_water_ratio / (1 + alcohol_water_ratio)) = 2 / 5 := by
  sorry

end alcohol_fraction_in_mixture_l3270_327032


namespace sheila_weekly_earnings_l3270_327031

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  hourly_rate : ℕ

/-- Calculates the total weekly hours worked --/
def total_weekly_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu

/-- Calculates the weekly earnings --/
def weekly_earnings (schedule : WorkSchedule) : ℕ :=
  (total_weekly_hours schedule) * schedule.hourly_rate

/-- Theorem stating Sheila's weekly earnings --/
theorem sheila_weekly_earnings :
  ∀ (schedule : WorkSchedule),
  schedule.hours_mon_wed_fri = 8 →
  schedule.hours_tue_thu = 6 →
  schedule.hourly_rate = 10 →
  weekly_earnings schedule = 360 := by
  sorry


end sheila_weekly_earnings_l3270_327031


namespace saturday_to_monday_ratio_is_two_to_one_l3270_327073

/-- Represents Mona's weekly biking schedule -/
structure BikeSchedule where
  total_distance : ℕ
  monday_distance : ℕ
  wednesday_distance : ℕ
  saturday_distance : ℕ
  total_eq : total_distance = monday_distance + wednesday_distance + saturday_distance

/-- Calculates the ratio of Saturday's distance to Monday's distance -/
def saturday_to_monday_ratio (schedule : BikeSchedule) : ℚ :=
  schedule.saturday_distance / schedule.monday_distance

/-- Theorem stating that the ratio of Saturday's distance to Monday's distance is 2:1 -/
theorem saturday_to_monday_ratio_is_two_to_one (schedule : BikeSchedule)
  (h1 : schedule.total_distance = 30)
  (h2 : schedule.monday_distance = 6)
  (h3 : schedule.wednesday_distance = 12) :
  saturday_to_monday_ratio schedule = 2 := by
  sorry

#eval saturday_to_monday_ratio {
  total_distance := 30,
  monday_distance := 6,
  wednesday_distance := 12,
  saturday_distance := 12,
  total_eq := by rfl
}

end saturday_to_monday_ratio_is_two_to_one_l3270_327073


namespace cheaper_to_buy_more_count_l3270_327055

-- Define the cost function C(n)
def C (n : ℕ) : ℕ :=
  if n ≤ 30 then 15 * n
  else if n ≤ 60 then 13 * n
  else 12 * n

-- Define a function that checks if buying n+1 books is cheaper than n books
def isCheaperToBuyMore (n : ℕ) : Prop :=
  C (n + 1) < C n

-- Theorem statement
theorem cheaper_to_buy_more_count :
  (∃ (S : Finset ℕ), S.card = 5 ∧ (∀ n, n ∈ S ↔ isCheaperToBuyMore n)) :=
by sorry

end cheaper_to_buy_more_count_l3270_327055


namespace product_abcd_l3270_327029

theorem product_abcd : 
  ∀ (a b c d : ℚ),
  (3 * a + 2 * b + 4 * c + 6 * d = 36) →
  (4 * (d + c) = b) →
  (4 * b + 2 * c = a) →
  (c - 2 = d) →
  (a * b * c * d = -315/32) :=
by
  sorry

end product_abcd_l3270_327029


namespace solution_set_of_inequality_l3270_327064

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a - 1) * x^2

-- Define the property of f being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Main theorem
theorem solution_set_of_inequality (a : ℝ) :
  (is_odd_function (f a)) →
  {x : ℝ | f a (a * x) > f a (a - x)} = {x : ℝ | x > 1/2} :=
by sorry

end solution_set_of_inequality_l3270_327064


namespace train_speed_calculation_l3270_327003

/-- Given a train and bridge scenario, calculate the train's speed in km/h -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (time : ℝ) 
  (h1 : train_length = 360) 
  (h2 : bridge_length = 140) 
  (h3 : time = 40) : 
  (train_length + bridge_length) / time * 3.6 = 45 := by
  sorry

end train_speed_calculation_l3270_327003


namespace problem_solution_l3270_327051

def A (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2*m + 3}

def B : Set ℝ := {x | -x^2 + 2*x + 8 > 0}

theorem problem_solution :
  (∀ m : ℝ, 
    (m = 2 → A m ∪ B = {x | -2 < x ∧ x ≤ 7}) ∧
    (m = 2 → (Set.univ \ A m) ∩ B = {x | -2 < x ∧ x < 1})) ∧
  (∀ m : ℝ, A m ∩ B = A m ↔ m < -4 ∨ (-1 < m ∧ m < 1/2)) :=
by sorry

end problem_solution_l3270_327051


namespace smallest_cube_divisor_l3270_327065

theorem smallest_cube_divisor (a b c : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  let m := a^3 * b^5 * c^7
  ∀ k : ℕ, k^3 ∣ m → (a * b * c^3)^3 ≤ k^3 := by
  sorry

end smallest_cube_divisor_l3270_327065


namespace timothy_speed_l3270_327006

theorem timothy_speed (mother_speed : ℝ) (distance : ℝ) (head_start : ℝ) :
  mother_speed = 36 →
  distance = 1.8 →
  head_start = 0.25 →
  let mother_time : ℝ := distance / mother_speed
  let total_time : ℝ := mother_time + head_start
  let timothy_speed : ℝ := distance / total_time
  timothy_speed = 6 := by sorry

end timothy_speed_l3270_327006


namespace alice_most_dogs_l3270_327086

-- Define the number of cats and dogs for each person
variable (Kc Ac Bc Kd Ad Bd : ℕ)

-- Define the conditions
variable (h1 : Kc > Ac)  -- Kathy owns more cats than Alice
variable (h2 : Kd > Bd)  -- Kathy owns more dogs than Bruce
variable (h3 : Ad > Kd)  -- Alice owns more dogs than Kathy
variable (h4 : Bc > Ac)  -- Bruce owns more cats than Alice

-- Theorem statement
theorem alice_most_dogs : Ad > Kd ∧ Ad > Bd := by
  sorry

end alice_most_dogs_l3270_327086


namespace range_of_a_l3270_327046

def p (x a : ℝ) : Prop := |x - a| < 4
def q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

theorem range_of_a : 
  (∀ x a : ℝ, (¬(p x a) → ¬(q x)) ∧ ∃ x, q x ∧ p x a) → 
  ∃ a : ℝ, -1 ≤ a ∧ a ≤ 6 :=
sorry

end range_of_a_l3270_327046


namespace customs_duration_l3270_327014

theorem customs_duration (navigation_time transport_time total_time : ℕ) 
  (h1 : navigation_time = 21)
  (h2 : transport_time = 7)
  (h3 : total_time = 30) :
  total_time - navigation_time - transport_time = 2 := by
  sorry

end customs_duration_l3270_327014


namespace exists_square_composition_function_l3270_327017

theorem exists_square_composition_function :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n ^ 2 := by
  sorry

end exists_square_composition_function_l3270_327017


namespace sum_interior_angles_180_l3270_327028

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define interior angles of a triangle
def interior_angles (t : Triangle) : Fin 3 → ℝ := sorry

-- Theorem: The sum of interior angles of any triangle is 180°
theorem sum_interior_angles_180 (t : Triangle) : 
  (interior_angles t 0) + (interior_angles t 1) + (interior_angles t 2) = 180 := by
  sorry

end sum_interior_angles_180_l3270_327028


namespace original_number_proof_l3270_327081

theorem original_number_proof : ∃! N : ℕ, N > 0 ∧ (N - 5) % 13 = 0 ∧ ∀ M : ℕ, M > 0 → (M - 5) % 13 = 0 → M ≥ N :=
by sorry

end original_number_proof_l3270_327081


namespace least_subtraction_for_divisibility_problem_solution_l3270_327085

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem problem_solution :
  ∃ (k : ℕ), k < 17 ∧ (9857621 - k) % 17 = 0 ∧ ∀ (m : ℕ), m < k → (9857621 - m) % 17 ≠ 0 ∧ k = 8 :=
by sorry

end least_subtraction_for_divisibility_problem_solution_l3270_327085


namespace vertex_on_parabola_and_line_intersection_l3270_327027

/-- The quadratic function -/
def f (m x : ℝ) : ℝ := x^2 + 2*(m + 1)*x - m + 1

/-- The vertex of the quadratic function -/
def vertex (m : ℝ) : ℝ × ℝ := (-m - 1, -m^2 - 3*m)

/-- The parabola on which the vertex lies -/
def parabola (x : ℝ) : ℝ := -x^2 + x + 2

/-- The line that may pass through the vertex -/
def line (x : ℝ) : ℝ := x + 1

theorem vertex_on_parabola_and_line_intersection (m : ℝ) :
  (∀ m, parabola (vertex m).1 = (vertex m).2) ∧
  (line (vertex m).1 = (vertex m).2 ↔ m = -2 ∨ m = 0) :=
by sorry

end vertex_on_parabola_and_line_intersection_l3270_327027


namespace intersection_A_notB_C_subset_A_implies_a_range_l3270_327018

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- Define the complement of B in ℝ
def notB : Set ℝ := {x | ¬(x ∈ B)}

-- Theorem for part (I)
theorem intersection_A_notB : A ∩ notB = {x : ℝ | -3 < x ∧ x ≤ 2} := by sorry

-- Theorem for part (II)
theorem C_subset_A_implies_a_range (a : ℝ) (h : a ≠ 0) :
  C a ⊆ A → (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a ≤ 4/3) := by sorry

end intersection_A_notB_C_subset_A_implies_a_range_l3270_327018


namespace subtraction_multiplication_equality_l3270_327001

theorem subtraction_multiplication_equality : 
  ((2000000000000 - 1111111111111) * 2) = 1777777777778 := by
  sorry

end subtraction_multiplication_equality_l3270_327001


namespace equation_equivalence_and_product_l3270_327089

theorem equation_equivalence_and_product (a c x y : ℝ) :
  ∃ (r s t u : ℤ),
    ((a^r * x - a^s) * (a^t * y - a^u) = a^6 * c^3) ↔
    (a^10 * x * y - a^8 * y - a^7 * x = a^6 * (c^3 - 1)) ∧
    r * s * t * u = 0 :=
by sorry

end equation_equivalence_and_product_l3270_327089


namespace sum_of_new_observations_l3270_327033

/-- Given 10 observations with an average of 21, prove that adding two new observations
    that increase the average by 2 results in the sum of the two new observations being 66. -/
theorem sum_of_new_observations (initial_count : Nat) (initial_avg : ℝ) (new_count : Nat) (avg_increase : ℝ) :
  initial_count = 10 →
  initial_avg = 21 →
  new_count = initial_count + 2 →
  avg_increase = 2 →
  (new_count : ℝ) * (initial_avg + avg_increase) - (initial_count : ℝ) * initial_avg = 66 := by
  sorry

#check sum_of_new_observations

end sum_of_new_observations_l3270_327033


namespace arithmetic_mean_reciprocals_first_four_primes_l3270_327009

theorem arithmetic_mean_reciprocals_first_four_primes : 
  let first_four_primes := [2, 3, 5, 7]
  ((first_four_primes.map (λ x => 1 / x)).sum) / first_four_primes.length = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l3270_327009


namespace division_sum_theorem_l3270_327030

theorem division_sum_theorem (quotient divisor remainder : ℕ) : 
  quotient = 120 → divisor = 456 → remainder = 333 → 
  (divisor * quotient + remainder = 55053) := by sorry

end division_sum_theorem_l3270_327030


namespace isosceles_triangle_largest_angle_l3270_327053

theorem isosceles_triangle_largest_angle (α β γ : ℝ) :
  -- The triangle is isosceles with two equal angles
  α = β ∧
  -- One of the equal angles is 30°
  α = 30 ∧
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 120°
  max α (max β γ) = 120 := by
sorry

end isosceles_triangle_largest_angle_l3270_327053


namespace problem_statement_l3270_327019

theorem problem_statement (a b : ℝ) (h : |a + 5| + (b - 2)^2 = 0) :
  (a + b)^2010 = 3^2010 := by
  sorry

end problem_statement_l3270_327019


namespace pony_price_is_20_l3270_327011

/-- The regular price of fox jeans in dollars -/
def fox_price : ℝ := 15

/-- The regular price of pony jeans in dollars -/
def pony_price : ℝ := 20

/-- The number of fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings in dollars -/
def total_savings : ℝ := 9

/-- The sum of the two discount rates as a percentage -/
def total_discount_rate : ℝ := 22

/-- The discount rate on pony jeans as a percentage -/
def pony_discount_rate : ℝ := 18

/-- Theorem stating that the regular price of pony jeans is $20 given the conditions -/
theorem pony_price_is_20 : 
  fox_price * fox_quantity * (total_discount_rate - pony_discount_rate) / 100 +
  pony_price * pony_quantity * pony_discount_rate / 100 = total_savings :=
by sorry

end pony_price_is_20_l3270_327011


namespace diagonals_30_sided_polygon_l3270_327022

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by
  sorry

end diagonals_30_sided_polygon_l3270_327022


namespace first_month_sales_l3270_327098

def sales_second_month : ℤ := 8550
def sales_third_month : ℤ := 6855
def sales_fourth_month : ℤ := 3850
def sales_fifth_month : ℤ := 14045
def average_sale : ℤ := 7800
def num_months : ℤ := 5

theorem first_month_sales :
  (average_sale * num_months) - (sales_second_month + sales_third_month + sales_fourth_month + sales_fifth_month) = 8700 := by
  sorry

end first_month_sales_l3270_327098


namespace distance_point_to_line_polar_l3270_327035

/-- The distance between a point in polar coordinates and a line given by a polar equation -/
theorem distance_point_to_line_polar (ρ_A θ_A : ℝ) :
  let r := 2 * ρ_A * Real.sin (θ_A - π / 4) - Real.sqrt 2
  let x := ρ_A * Real.cos θ_A
  let y := ρ_A * Real.sin θ_A
  ρ_A = 2 * Real.sqrt 2 ∧ θ_A = 7 * π / 4 →
  (r^2 / (1 + 1)) = (5 * Real.sqrt 2 / 2)^2 := by
sorry

end distance_point_to_line_polar_l3270_327035


namespace min_surface_area_angle_l3270_327082

/-- The angle that minimizes the surface area of a rotated right triangle -/
theorem min_surface_area_angle (AC BC CD : ℝ) (h1 : AC = 3) (h2 : BC = 4) (h3 : CD = 10) :
  let α := Real.arctan (2 / 3)
  let surface_area (θ : ℝ) := π * (240 - 12 * (2 * Real.sin θ + 3 * Real.cos θ))
  ∀ θ, surface_area α ≤ surface_area θ := by
sorry


end min_surface_area_angle_l3270_327082


namespace quadratic_function_proof_l3270_327045

def f (x : ℝ) : ℝ := -3 * (x - 2)^2 + 12

theorem quadratic_function_proof :
  (∀ x, f x > 0 ↔ 0 < x ∧ x < 4) ∧
  (∀ x ∈ Set.Icc (-1) 5, f x ≤ 12) ∧
  (∃ x ∈ Set.Icc (-1) 5, f x = 12) →
  ∀ x, f x = -3 * (x - 2)^2 + 12 :=
by
  sorry

end quadratic_function_proof_l3270_327045


namespace even_function_shift_l3270_327023

/-- Given a function f and a real number a, proves that if f(x) = 3sin(2x - π/3) 
    and y = f(x + a) is an even function where 0 < a < π/2, then a = 5π/12 -/
theorem even_function_shift (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = 3 * Real.sin (2 * x - π / 3)) →
  (∀ x, f (x + a) = f (-x + a)) →
  (0 < a) →
  (a < π / 2) →
  a = 5 * π / 12 := by
sorry

end even_function_shift_l3270_327023


namespace sum_of_four_consecutive_integers_divisible_by_two_l3270_327049

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) :
  ∃ k : ℤ, (n - 1) + n + (n + 1) + (n + 2) = 2 * k :=
sorry

end sum_of_four_consecutive_integers_divisible_by_two_l3270_327049


namespace cyclic_inequality_l3270_327041

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) ≥ (a*b + b*c + c*a)^2 := by
  sorry

end cyclic_inequality_l3270_327041


namespace bananas_to_kiwis_ratio_l3270_327002

/-- Represents the cost of a dozen apples in dollars -/
def dozen_apples_cost : ℚ := 14

/-- Represents the amount Brian spent on kiwis in dollars -/
def kiwis_cost : ℚ := 10

/-- Represents the maximum number of apples Brian can buy -/
def max_apples : ℕ := 24

/-- Represents the amount Brian left his house with in dollars -/
def initial_amount : ℚ := 50

/-- Represents the subway fare in dollars -/
def subway_fare : ℚ := 3.5

/-- Calculates the amount spent on bananas -/
def bananas_cost : ℚ := initial_amount - 2 * subway_fare - kiwis_cost - (max_apples / 12) * dozen_apples_cost

/-- Theorem stating that the ratio of bananas cost to kiwis cost is 1:2 -/
theorem bananas_to_kiwis_ratio : bananas_cost / kiwis_cost = 1 / 2 := by
  sorry

end bananas_to_kiwis_ratio_l3270_327002


namespace golf_score_difference_l3270_327080

/-- Given Richard's and Bruno's golf scores, prove the difference between their scores. -/
theorem golf_score_difference (richard_score bruno_score : ℕ) 
  (h1 : richard_score = 62) 
  (h2 : bruno_score = 48) : 
  richard_score - bruno_score = 14 := by
  sorry

end golf_score_difference_l3270_327080


namespace divisible_by_three_exists_l3270_327092

/-- A type representing the arrangement of natural numbers in a circle. -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- Predicate to check if two numbers differ by 1, 2, or by a factor of two. -/
def ValidDifference (a b : ℕ) : Prop :=
  (a = b + 1) ∨ (b = a + 1) ∨ (a = b + 2) ∨ (b = a + 2) ∨ (a = 2 * b) ∨ (b = 2 * a)

/-- Theorem stating that in any arrangement of 99 natural numbers in a circle
    where any two neighboring numbers differ either by 1, or by 2, or by a factor of two,
    at least one of these numbers is divisible by 3. -/
theorem divisible_by_three_exists (arr : CircularArrangement 99)
  (h : ∀ i : Fin 99, ValidDifference (arr i) (arr (i + 1))) :
  ∃ i : Fin 99, 3 ∣ arr i := by
  sorry

end divisible_by_three_exists_l3270_327092


namespace valid_seating_arrangements_l3270_327020

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to seat 9 people in a row with the given constraint. -/
def seating_arrangements : ℕ :=
  factorial 9 - factorial 7 * factorial 3

/-- Theorem stating the number of valid seating arrangements. -/
theorem valid_seating_arrangements :
  seating_arrangements = 332640 := by sorry

end valid_seating_arrangements_l3270_327020


namespace count_perfect_square_factors_of_4500_l3270_327010

/-- The number of perfect square factors of 4500 -/
def perfectSquareFactorsOf4500 : ℕ :=
  -- Define the number of perfect square factors of 4500
  -- We don't implement the calculation here, just define it
  -- The actual value will be proven to be 8
  sorry

/-- Theorem: The number of perfect square factors of 4500 is 8 -/
theorem count_perfect_square_factors_of_4500 :
  perfectSquareFactorsOf4500 = 8 := by
  sorry

end count_perfect_square_factors_of_4500_l3270_327010


namespace solve_pencil_problem_l3270_327013

def pencil_problem (anna_pencils : ℕ) (harry_multiplier : ℕ) (harry_lost : ℕ) : Prop :=
  let harry_initial := anna_pencils * harry_multiplier
  harry_initial - harry_lost = 81

theorem solve_pencil_problem :
  pencil_problem 50 2 19 := by sorry

end solve_pencil_problem_l3270_327013


namespace square_root_problem_l3270_327071

theorem square_root_problem (a b c : ℝ) : 
  (a - 4)^(1/3) = 1 →
  (3 * a - b - 2)^(1/2) = 3 →
  c = ⌊Real.sqrt 13⌋ →
  (2 * a - 3 * b + c)^(1/2) = 1 ∨ (2 * a - 3 * b + c)^(1/2) = -1 := by
  sorry

end square_root_problem_l3270_327071


namespace octahedron_colorings_l3270_327096

/-- The symmetry group of a regular octahedron -/
structure OctahedronSymmetryGroup where
  order : ℕ
  h_order : order = 24

/-- The number of distinct vertex colorings of a regular octahedron -/
def vertex_colorings (G : OctahedronSymmetryGroup) (m : ℕ) : ℚ :=
  (m^6 + 3*m^4 + 12*m^3 + 8*m^2) / G.order

/-- The number of distinct face colorings of a regular octahedron -/
def face_colorings (G : OctahedronSymmetryGroup) (m : ℕ) : ℚ :=
  (m^8 + 17*m^6 + 6*m^2) / G.order

/-- Theorem stating the number of distinct colorings for vertices and faces -/
theorem octahedron_colorings (G : OctahedronSymmetryGroup) (m : ℕ) :
  (vertex_colorings G m = (m^6 + 3*m^4 + 12*m^3 + 8*m^2) / 24) ∧
  (face_colorings G m = (m^8 + 17*m^6 + 6*m^2) / 24) := by
  sorry

end octahedron_colorings_l3270_327096


namespace janes_change_calculation_l3270_327093

-- Define the prices and quantities
def skirt_price : ℝ := 65
def skirt_quantity : ℕ := 2
def blouse_price : ℝ := 30
def blouse_quantity : ℕ := 3
def shoes_price : ℝ := 125
def handbag_price : ℝ := 175

-- Define the discounts and taxes
def handbag_discount : ℝ := 0.10
def total_discount : ℝ := 0.05
def coupon_discount : ℝ := 20
def sales_tax : ℝ := 0.08

-- Define the exchange rate and amount paid
def exchange_rate : ℝ := 0.8
def amount_paid : ℝ := 600

-- Theorem to prove
theorem janes_change_calculation :
  let initial_total := skirt_price * skirt_quantity + blouse_price * blouse_quantity + shoes_price + handbag_price
  let handbag_discounted := initial_total - handbag_discount * handbag_price
  let total_discounted := handbag_discounted * (1 - total_discount)
  let coupon_applied := total_discounted - coupon_discount
  let taxed_total := coupon_applied * (1 + sales_tax)
  let home_currency_total := taxed_total * exchange_rate
  amount_paid - home_currency_total = 204.828 := by sorry

end janes_change_calculation_l3270_327093


namespace intersection_points_l3270_327058

-- Define the quadratic and linear functions
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def linear (s t x : ℝ) : ℝ := s * x + t

-- Define the discriminant
def discriminant (a b c s t : ℝ) : ℝ := (b - s)^2 - 4 * a * (c - t)

-- Theorem statement
theorem intersection_points (a b c s t : ℝ) (ha : a ≠ 0) (hs : s ≠ 0) :
  let Δ := discriminant a b c s t
  -- Two intersection points when Δ > 0
  (Δ > 0 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ quadratic a b c x₁ = linear s t x₁ ∧ quadratic a b c x₂ = linear s t x₂) ∧
  -- One intersection point when Δ = 0
  (Δ = 0 → ∃! x, quadratic a b c x = linear s t x) ∧
  -- No intersection points when Δ < 0
  (Δ < 0 → ∀ x, quadratic a b c x ≠ linear s t x) :=
by sorry

end intersection_points_l3270_327058


namespace regular_polygon_sides_l3270_327047

theorem regular_polygon_sides (D : ℕ) : D = 12 → ∃ n : ℕ, n = 6 ∧ D = n * (n - 3) / 2 := by
  sorry

end regular_polygon_sides_l3270_327047


namespace parallelogram_area_l3270_327052

def v1 : Fin 3 → ℝ := ![4, -1, 3]
def v2 : Fin 3 → ℝ := ![-2, 5, -1]

theorem parallelogram_area : 
  Real.sqrt ((v1 1 * v2 2 - v1 2 * v2 1)^2 + 
             (v1 2 * v2 0 - v1 0 * v2 2)^2 + 
             (v1 0 * v2 1 - v1 1 * v2 0)^2) = Real.sqrt 684 := by
  sorry

end parallelogram_area_l3270_327052


namespace simplify_expression_l3270_327005

theorem simplify_expression (y : ℝ) :
  4 * y + 8 * y^2 + 6 - (3 - 4 * y - 8 * y^2) = 16 * y^2 + 8 * y + 3 := by
  sorry

end simplify_expression_l3270_327005


namespace max_triangle_area_is_85_l3270_327038

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D plane -/
structure Line where
  slope : ℝ
  point : Point

/-- Triangle formed by three lines -/
structure Triangle where
  l1 : Line
  l2 : Line
  l3 : Line

/-- Rotation of a line around its point -/
def rotate (l : Line) (angle : ℝ) : Line :=
  sorry

/-- Area of a triangle formed by three lines -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Maximum area of triangle formed by rotating lines -/
def maxTriangleArea (l1 l2 l3 : Line) : ℝ :=
  sorry

theorem max_triangle_area_is_85 :
  let a := Point.mk 0 0
  let b := Point.mk 11 0
  let c := Point.mk 18 0
  let la := Line.mk 1 a
  let lb := Line.mk 0 b  -- Vertical line represented with slope 0
  let lc := Line.mk (-1) c
  maxTriangleArea la lb lc = 85 := by
  sorry

end max_triangle_area_is_85_l3270_327038


namespace divisor_sum_of_2_3_power_l3270_327088

/-- Sum of positive divisors of n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Sum of geometric series -/
def geometric_sum (a r : ℕ) (n : ℕ) : ℕ := sorry

theorem divisor_sum_of_2_3_power (i j : ℕ) :
  sum_of_divisors (2^i * 3^j) = 540 → i + j = 5 := by sorry

end divisor_sum_of_2_3_power_l3270_327088


namespace combined_output_fraction_l3270_327037

/-- Represents the production rate of a machine relative to a base rate -/
structure ProductionRate :=
  (rate : ℚ)

/-- Represents a machine with its production rate -/
structure Machine :=
  (name : String)
  (rate : ProductionRate)

/-- The problem setup with four machines and their relative production rates -/
def production_problem (t n o p : Machine) : Prop :=
  t.rate.rate = 4 / 3 * n.rate.rate ∧
  n.rate.rate = 3 / 2 * o.rate.rate ∧
  o.rate = p.rate

/-- The theorem stating that machines N and P produce 6/13 of the total output -/
theorem combined_output_fraction 
  (t n o p : Machine) 
  (h : production_problem t n o p) : 
  (n.rate.rate + p.rate.rate) / (t.rate.rate + n.rate.rate + o.rate.rate + p.rate.rate) = 6 / 13 :=
sorry

end combined_output_fraction_l3270_327037


namespace hammer_order_sequence_l3270_327025

theorem hammer_order_sequence (sequence : ℕ → ℕ) : 
  sequence 1 = 3 →  -- June (1st month)
  sequence 3 = 6 →  -- August (3rd month)
  sequence 4 = 9 →  -- September (4th month)
  sequence 5 = 13 → -- October (5th month)
  sequence 2 = 6    -- July (2nd month)
:= by sorry

end hammer_order_sequence_l3270_327025


namespace expression_equality_l3270_327050

theorem expression_equality : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end expression_equality_l3270_327050


namespace unique_solution_to_exponential_equation_l3270_327078

theorem unique_solution_to_exponential_equation :
  ∀ x y z : ℕ, 3^x + 4^y = 5^z ↔ x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end unique_solution_to_exponential_equation_l3270_327078


namespace restaurant_bill_proof_l3270_327066

theorem restaurant_bill_proof (n : ℕ) (extra : ℚ) (total_bill : ℚ) : 
  n = 10 →
  extra = 3 →
  (n - 1) * ((total_bill / n) + extra) = total_bill →
  total_bill = 270 := by
  sorry

end restaurant_bill_proof_l3270_327066


namespace classroom_count_l3270_327008

/-- Given a classroom with a 1:2 ratio of girls to boys and 20 boys, prove the total number of students is 30. -/
theorem classroom_count (num_boys : ℕ) (ratio_girls_to_boys : ℚ) : 
  num_boys = 20 → ratio_girls_to_boys = 1/2 → num_boys + (ratio_girls_to_boys * num_boys) = 30 := by
  sorry

end classroom_count_l3270_327008


namespace p_false_and_q_true_l3270_327072

-- Define proposition p
def p : Prop := ∀ x > 0, 3^x > 1

-- Define proposition q
def q : Prop := ∀ a, (a < -2 → ∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) ∧
                    (∃ b, b ≥ -2 ∧ ∃ x ∈ Set.Icc (-1) 2, b * x + 3 = 0)

-- Theorem stating that p is false and q is true
theorem p_false_and_q_true : ¬p ∧ q := by sorry

end p_false_and_q_true_l3270_327072


namespace isosceles_triangle_perimeter_l3270_327063

/-- An isosceles triangle with two sides of length 9 and one side of length 4 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 9 → b = 9 → c = 4 →
  (a + b > c ∧ b + c > a ∧ a + c > b) →  -- Triangle inequality
  a + b + c = 22 := by
  sorry


end isosceles_triangle_perimeter_l3270_327063


namespace factor_expression_l3270_327061

theorem factor_expression :
  ∀ x : ℝ, 63 * x^2 + 42 = 21 * (3 * x^2 + 2) := by
  sorry

end factor_expression_l3270_327061


namespace cubic_inequality_l3270_327057

theorem cubic_inequality (x : ℝ) : (x^3 - 125) / (x + 3) < 0 ↔ -3 < x ∧ x < 5 :=
by sorry

end cubic_inequality_l3270_327057


namespace alchemerion_age_ratio_l3270_327034

/-- Represents the ages of Alchemerion, his son, and his father -/
structure WizardFamily where
  alchemerion : ℕ
  son : ℕ
  father : ℕ

/-- Defines the properties of the Wizard family's ages -/
def is_valid_wizard_family (f : WizardFamily) : Prop :=
  f.alchemerion = 360 ∧
  f.father = 2 * f.alchemerion + 40 ∧
  f.alchemerion + f.son + f.father = 1240 ∧
  ∃ k : ℕ, f.alchemerion = k * f.son

/-- Theorem stating that Alchemerion is 3 times older than his son -/
theorem alchemerion_age_ratio (f : WizardFamily) 
  (h : is_valid_wizard_family f) : 
  f.alchemerion = 3 * f.son :=
sorry

end alchemerion_age_ratio_l3270_327034


namespace binomial_coefficient_third_term_l3270_327042

theorem binomial_coefficient_third_term (x : ℝ) : 
  Nat.choose 4 2 = 6 := by sorry

end binomial_coefficient_third_term_l3270_327042


namespace kitty_cleanup_time_l3270_327026

/-- Represents the weekly cleaning routine for a living room -/
structure CleaningRoutine where
  pickup_time : ℕ  -- Time spent picking up toys and straightening
  vacuum_time : ℕ  -- Time spent vacuuming
  window_time : ℕ  -- Time spent cleaning windows
  dusting_time : ℕ  -- Time spent dusting furniture

/-- Calculates the total cleaning time for a given number of weeks -/
def total_cleaning_time (routine : CleaningRoutine) (weeks : ℕ) : ℕ :=
  weeks * (routine.pickup_time + routine.vacuum_time + routine.window_time + routine.dusting_time)

theorem kitty_cleanup_time :
  ∃ (routine : CleaningRoutine),
    routine.vacuum_time = 20 ∧
    routine.window_time = 15 ∧
    routine.dusting_time = 10 ∧
    total_cleaning_time routine 4 = 200 ∧
    routine.pickup_time = 5 := by
  sorry

end kitty_cleanup_time_l3270_327026


namespace complement_A_intersect_B_B_union_C_eq_B_iff_m_lt_4_l3270_327074

-- Define the sets A, B, and C
def A : Set ℝ := {x | (x - 7) / (x + 2) > 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 3*x + 28)}
def C (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem 1: Prove that (complement of A) ∩ B = [-2, 7)
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = Set.Icc (-2) 7 := by sorry

-- Theorem 2: Prove that B ∪ C = B if and only if m < 4
theorem B_union_C_eq_B_iff_m_lt_4 (m : ℝ) :
  B ∪ C m = B ↔ m < 4 := by sorry

end complement_A_intersect_B_B_union_C_eq_B_iff_m_lt_4_l3270_327074


namespace multiplication_factor_l3270_327090

theorem multiplication_factor (N : ℝ) (h : N ≠ 0) : 
  let X : ℝ := 5
  let incorrect_value := N / 10
  let correct_value := N * X
  let percentage_error := |correct_value - incorrect_value| / correct_value * 100
  percentage_error = 98 := by sorry

end multiplication_factor_l3270_327090


namespace unique_congruence_l3270_327056

theorem unique_congruence (n : ℤ) : 3 ≤ n ∧ n ≤ 11 ∧ n ≡ 2023 [ZMOD 7] → n = 7 := by
  sorry

end unique_congruence_l3270_327056


namespace arun_weight_average_l3270_327084

def weight_range (w : ℝ) : Prop :=
  66 < w ∧ w ≤ 69 ∧ 60 < w ∧ w < 70

theorem arun_weight_average : 
  ∃ (w₁ w₂ w₃ : ℝ), 
    weight_range w₁ ∧ 
    weight_range w₂ ∧ 
    weight_range w₃ ∧ 
    w₁ ≠ w₂ ∧ w₁ ≠ w₃ ∧ w₂ ≠ w₃ ∧
    (w₁ + w₂ + w₃) / 3 = 68 := by
  sorry

end arun_weight_average_l3270_327084


namespace union_of_A_and_B_l3270_327094

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | -1 < x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > -1} := by sorry

end union_of_A_and_B_l3270_327094


namespace simplify_expression_l3270_327044

theorem simplify_expression (m n : ℝ) : (8*m - 7*n) - 2*(m - 3*n) = 6*m - n := by
  sorry

end simplify_expression_l3270_327044


namespace third_number_in_sum_l3270_327060

theorem third_number_in_sum (a b c : ℝ) (h1 : a = 3.15) (h2 : b = 0.014) (h3 : a + b + c = 3.622) : c = 0.458 := by
  sorry

end third_number_in_sum_l3270_327060


namespace sin_product_equals_one_eighth_l3270_327016

theorem sin_product_equals_one_eighth : 
  Real.sin (12 * Real.pi / 180) * Real.sin (36 * Real.pi / 180) * 
  Real.sin (54 * Real.pi / 180) * Real.sin (72 * Real.pi / 180) = 1/8 := by
  sorry

end sin_product_equals_one_eighth_l3270_327016


namespace harry_bid_difference_l3270_327070

/-- Represents the auction process and calculates the difference between Harry's final bid and the third bidder's bid. -/
def auctionBidDifference (startingBid : ℕ) (harryFirstIncrement : ℕ) (harryFinalBid : ℕ) : ℕ :=
  let harryFirstBid := startingBid + harryFirstIncrement
  let secondBid := harryFirstBid * 2
  let thirdBid := secondBid + harryFirstIncrement * 3
  harryFinalBid - thirdBid

/-- Theorem stating that given the specific auction conditions, Harry's final bid exceeds the third bidder's bid by $2400. -/
theorem harry_bid_difference :
  auctionBidDifference 300 200 4000 = 2400 := by
  sorry

end harry_bid_difference_l3270_327070


namespace savings_in_cents_l3270_327004

/-- The in-store price of the appliance in dollars -/
def in_store_price : ℚ := 99.99

/-- The price of one payment in the TV commercial in dollars -/
def tv_payment : ℚ := 29.98

/-- The number of payments in the TV commercial -/
def num_payments : ℕ := 3

/-- The shipping and handling charge in dollars -/
def shipping_charge : ℚ := 9.98

/-- The total cost from the TV advertiser in dollars -/
def tv_total_cost : ℚ := tv_payment * num_payments + shipping_charge

/-- The savings in dollars -/
def savings : ℚ := in_store_price - tv_total_cost

/-- Convert dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℕ := (dollars * 100).ceil.toNat

theorem savings_in_cents : dollars_to_cents savings = 7 := by sorry

end savings_in_cents_l3270_327004


namespace consecutive_integers_product_l3270_327007

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  b = a + 1 →
  c = b + 1 →
  d = c + 1 →
  e = d + 1 →
  a * b * c * d * e = 15120 →
  e = 9 := by
sorry

end consecutive_integers_product_l3270_327007


namespace fruits_left_l3270_327097

theorem fruits_left (oranges apples : ℕ) 
  (h1 : oranges = 40)
  (h2 : apples = 70)
  (h3 : oranges / 4 + apples / 2 = oranges + apples - 65) : 
  oranges + apples - (oranges / 4 + apples / 2) = 65 := by
  sorry

#check fruits_left

end fruits_left_l3270_327097


namespace max_non_multiples_of_three_l3270_327043

/-- Given a list of 6 positive integers whose product is a multiple of 3,
    the maximum number of integers in the list that are not multiples of 3 is 5. -/
theorem max_non_multiples_of_three (integers : List ℕ+) : 
  integers.length = 6 → 
  integers.prod.val % 3 = 0 → 
  (integers.filter (fun x => x.val % 3 ≠ 0)).length ≤ 5 :=
sorry

end max_non_multiples_of_three_l3270_327043


namespace pascal_triangle_row20_elements_l3270_327095

theorem pascal_triangle_row20_elements : 
  (Nat.choose 20 4 = 4845) ∧ (Nat.choose 20 5 = 15504) := by
  sorry

end pascal_triangle_row20_elements_l3270_327095


namespace clubsuit_difference_l3270_327069

/-- The clubsuit operation -/
def clubsuit (x y : ℝ) : ℝ := 4*x + 6*y

/-- Theorem stating that (5 ♣ 3) - (1 ♣ 4) = 10 -/
theorem clubsuit_difference : (clubsuit 5 3) - (clubsuit 1 4) = 10 := by
  sorry

end clubsuit_difference_l3270_327069


namespace lady_bird_flour_theorem_l3270_327012

/-- The amount of flour needed for a given number of guests at Lady Bird's Junior League club meeting -/
def flour_needed (guests : ℕ) : ℚ :=
  let biscuits_per_guest : ℕ := 2
  let biscuits_per_batch : ℕ := 9
  let flour_per_batch : ℚ := 5 / 4
  let total_biscuits : ℕ := guests * biscuits_per_guest
  let batches : ℕ := (total_biscuits + biscuits_per_batch - 1) / biscuits_per_batch
  (batches : ℚ) * flour_per_batch

/-- Theorem stating that Lady Bird needs 5 cups of flour for 18 guests -/
theorem lady_bird_flour_theorem :
  flour_needed 18 = 5 := by
  sorry

end lady_bird_flour_theorem_l3270_327012


namespace linear_equation_solutions_l3270_327054

theorem linear_equation_solutions : 
  {(x, y) : ℕ × ℕ | 5 * x + 2 * y = 25 ∧ x > 0 ∧ y > 0} = {(1, 10), (3, 5)} := by
  sorry

end linear_equation_solutions_l3270_327054


namespace complex_ratio_l3270_327000

theorem complex_ratio (z₁ z₂ : ℂ) (h1 : Complex.abs z₁ = 1) (h2 : Complex.abs z₂ = 5/2) 
  (h3 : Complex.abs (3 * z₁ - 2 * z₂) = 7) :
  z₁ / z₂ = -1/5 * (1 - Complex.I * Real.sqrt 3) ∨ z₁ / z₂ = -1/5 * (1 + Complex.I * Real.sqrt 3) := by
sorry

end complex_ratio_l3270_327000


namespace wrong_number_correction_l3270_327021

theorem wrong_number_correction (n : ℕ) (initial_avg correct_avg : ℚ) 
  (first_error second_correct : ℚ) :
  n = 10 ∧ 
  initial_avg = 40.2 ∧ 
  correct_avg = 40.1 ∧ 
  first_error = 17 ∧ 
  second_correct = 31 →
  ∃ second_error : ℚ,
    n * initial_avg - first_error - second_error + second_correct = n * correct_avg ∧
    second_error = 15 :=
by sorry

end wrong_number_correction_l3270_327021


namespace boys_transferred_l3270_327076

theorem boys_transferred (initial_boys : ℕ) (initial_ratio_boys : ℕ) (initial_ratio_girls : ℕ)
  (final_ratio_boys : ℕ) (final_ratio_girls : ℕ) :
  initial_boys = 120 →
  initial_ratio_boys = 3 →
  initial_ratio_girls = 4 →
  final_ratio_boys = 4 →
  final_ratio_girls = 5 →
  ∃ (transferred_boys : ℕ),
    transferred_boys = 13 ∧
    ∃ (initial_girls : ℕ),
      initial_girls * initial_ratio_boys = initial_boys * initial_ratio_girls ∧
      (initial_boys - transferred_boys) * final_ratio_girls = 
      (initial_girls - 2 * transferred_boys) * final_ratio_boys :=
by sorry

end boys_transferred_l3270_327076


namespace dogs_adopted_twenty_dogs_adopted_l3270_327048

/-- The number of dogs adopted from a pet center --/
theorem dogs_adopted (initial_dogs : ℕ) (initial_cats : ℕ) (additional_cats : ℕ) (final_total : ℕ) : ℕ :=
  let remaining_dogs := initial_dogs - (initial_dogs + initial_cats + additional_cats - final_total)
  initial_dogs - remaining_dogs

/-- Proof that 20 dogs were adopted given the problem conditions --/
theorem twenty_dogs_adopted : dogs_adopted 36 29 12 57 = 20 := by
  sorry

end dogs_adopted_twenty_dogs_adopted_l3270_327048


namespace smallest_value_l3270_327036

theorem smallest_value : 
  let a := -((-3 - 2)^2)
  let b := (-3) * (-2)
  let c := (-3)^2 / (-2)^2
  let d := (-3)^2 / (-2)
  (a ≤ b) ∧ (a ≤ c) ∧ (a ≤ d) := by sorry

end smallest_value_l3270_327036


namespace magic_square_base_is_three_l3270_327068

/-- Represents a 3x3 magic square with elements in base b -/
def MagicSquare (b : ℕ) : Type :=
  Fin 3 → Fin 3 → ℕ

/-- The sum of a row, column, or diagonal in the magic square -/
def MagicSum (b : ℕ) (square : MagicSquare b) : ℕ :=
  square 0 0 + square 0 1 + square 0 2

/-- Predicate to check if a given square is magic -/
def IsMagicSquare (b : ℕ) (square : MagicSquare b) : Prop :=
  (∀ i : Fin 3, square i 0 + square i 1 + square i 2 = MagicSum b square) ∧
  (∀ j : Fin 3, square 0 j + square 1 j + square 2 j = MagicSum b square) ∧
  (square 0 0 + square 1 1 + square 2 2 = MagicSum b square) ∧
  (square 0 2 + square 1 1 + square 2 0 = MagicSum b square)

/-- The specific magic square given in the problem -/
def GivenSquare (b : ℕ) : MagicSquare b :=
  fun i j => match i, j with
  | 0, 0 => 5
  | 0, 1 => 11
  | 0, 2 => 15
  | 1, 0 => 4
  | 1, 1 => 11
  | 1, 2 => 12
  | 2, 0 => 14
  | 2, 1 => 2
  | 2, 2 => 3

theorem magic_square_base_is_three :
  ∃ (b : ℕ), b > 1 ∧ IsMagicSquare b (GivenSquare b) ∧ b = 3 :=
sorry

end magic_square_base_is_three_l3270_327068


namespace solution_difference_l3270_327075

theorem solution_difference (a b : ℝ) : 
  (∀ x, (x - 5) * (x + 5) = 26 * x - 130 ↔ x = a ∨ x = b) →
  a ≠ b →
  a > b →
  a - b = 16 := by
sorry

end solution_difference_l3270_327075


namespace isosceles_triangle_perimeter_l3270_327067

-- Define the equation
def equation (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 5

-- Define the roots of the equation
def roots (m : ℝ) : Set ℝ := {x | equation m x = 0}

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  base_positive : base > 0
  side_positive : side > 0
  isosceles : side ≥ base

-- State the theorem
theorem isosceles_triangle_perimeter : 
  ∃ (m : ℝ) (t : IsoscelesTriangle), 
    1 ∈ roots m ∧ 
    (∃ (x : ℝ), x ∈ roots m ∧ x ≠ 1) ∧
    {t.base, t.side} = roots m ∧
    t.base + 2 * t.side = 11 := by sorry

end isosceles_triangle_perimeter_l3270_327067
