import Mathlib

namespace smallest_integer_with_given_remainders_l3303_330377

theorem smallest_integer_with_given_remainders :
  ∃ (a : ℕ), a > 0 ∧ a % 8 = 6 ∧ a % 9 = 5 ∧
  ∀ (b : ℕ), b > 0 → b % 8 = 6 → b % 9 = 5 → a ≤ b :=
by
  use 14
  sorry

end smallest_integer_with_given_remainders_l3303_330377


namespace possible_values_of_a_l3303_330366

theorem possible_values_of_a (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^3 + y^3 = a) 
  (h3 : x^5 + y^5 = a) : 
  a ∈ ({-2, -1, 0, 1, 2} : Set ℝ) := by
  sorry

end possible_values_of_a_l3303_330366


namespace ammonia_molecular_weight_l3303_330306

/-- The atomic weight of nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The number of nitrogen atoms in an ammonia molecule -/
def nitrogen_count : ℕ := 1

/-- The number of hydrogen atoms in an ammonia molecule -/
def hydrogen_count : ℕ := 3

/-- The molecular weight of ammonia in atomic mass units (amu) -/
def ammonia_weight : ℝ := nitrogen_weight * nitrogen_count + hydrogen_weight * hydrogen_count

theorem ammonia_molecular_weight :
  ammonia_weight = 17.034 := by sorry

end ammonia_molecular_weight_l3303_330306


namespace odd_product_probability_l3303_330312

theorem odd_product_probability : 
  let n : ℕ := 25
  let odd_count : ℕ := (n + 1) / 2
  let total_combinations : ℕ := n * (n - 1) / 2
  let odd_combinations : ℕ := odd_count * (odd_count - 1) / 2
  (odd_combinations : ℚ) / total_combinations = 13 / 50 := by sorry

end odd_product_probability_l3303_330312


namespace rudolph_stop_signs_l3303_330347

/-- Calculates the number of stop signs encountered on a car trip -/
def stop_signs_encountered (base_distance : ℕ) (additional_distance : ℕ) (signs_per_mile : ℕ) : ℕ :=
  (base_distance + additional_distance) * signs_per_mile

/-- Theorem: Rudolph encountered 14 stop signs on his trip -/
theorem rudolph_stop_signs :
  stop_signs_encountered 5 2 2 = 14 := by
  sorry

end rudolph_stop_signs_l3303_330347


namespace derivative_of_y_l3303_330322

-- Define the function y
def y (x a b c : ℝ) : ℝ := (x - a) * (x - b) * (x - c)

-- State the theorem
theorem derivative_of_y (x a b c : ℝ) :
  deriv (fun x => y x a b c) x = 3 * x^2 - 2 * (a + b + c) * x + (a * b + a * c + b * c) :=
by sorry

end derivative_of_y_l3303_330322


namespace square_root_equation_l3303_330334

theorem square_root_equation (n : ℝ) : 
  Real.sqrt (10 + n) = 9 → n = 71 := by sorry

end square_root_equation_l3303_330334


namespace quadratic_roots_l3303_330361

/-- A quadratic function passing through specific points has roots -4 and 1 -/
theorem quadratic_roots (a b c : ℝ) (h_a : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (f (-5) = 6) ∧ (f (-4) = 0) ∧ (f (-2) = -6) ∧ (f 0 = -4) ∧ (f 2 = 6) →
  (∀ x, f x = 0 ↔ x = -4 ∨ x = 1) :=
by sorry

end quadratic_roots_l3303_330361


namespace system_solution_ratio_l3303_330362

theorem system_solution_ratio (x y a b : ℝ) (h1 : 4 * x - 2 * y = a) 
  (h2 : 6 * y - 12 * x = b) (h3 : b ≠ 0) : a / b = 1 / 3 := by
  sorry

end system_solution_ratio_l3303_330362


namespace horner_v2_value_l3303_330326

/-- Horner's method for polynomial evaluation -/
def horner_step (x : ℝ) (a : ℝ) (v : ℝ) : ℝ := v * x + a

/-- The polynomial f(x) = x^4 + 2x^3 - 3x^2 + x + 5 -/
def f (x : ℝ) : ℝ := x^4 + 2*x^3 - 3*x^2 + x + 5

theorem horner_v2_value :
  let x : ℝ := 2
  let v₁ : ℝ := horner_step x 2 1  -- Corresponds to x + 2
  let v₂ : ℝ := horner_step x (-3) v₁  -- Corresponds to v₁ * x - 3
  v₂ = 5 := by sorry

end horner_v2_value_l3303_330326


namespace mangoes_rate_per_kg_l3303_330304

/-- Given Tom's fruit purchase details, prove the rate per kg for mangoes -/
theorem mangoes_rate_per_kg 
  (apple_quantity : ℕ) 
  (apple_rate : ℕ) 
  (mango_quantity : ℕ) 
  (total_paid : ℕ) 
  (h1 : apple_quantity = 8)
  (h2 : apple_rate = 70)
  (h3 : mango_quantity = 9)
  (h4 : total_paid = 965) :
  (total_paid - apple_quantity * apple_rate) / mango_quantity = 45 :=
by sorry

end mangoes_rate_per_kg_l3303_330304


namespace soy_milk_calculation_l3303_330336

/-- The amount of soy milk drunk by Mitch's family in a week -/
def soy_milk : ℝ := 0.1

/-- The total amount of milk drunk by Mitch's family in a week -/
def total_milk : ℝ := 0.6

/-- The amount of regular milk drunk by Mitch's family in a week -/
def regular_milk : ℝ := 0.5

/-- Theorem stating that the amount of soy milk is the difference between total milk and regular milk -/
theorem soy_milk_calculation : soy_milk = total_milk - regular_milk := by
  sorry

end soy_milk_calculation_l3303_330336


namespace grassland_area_ratio_l3303_330381

/-- Represents a grassland with two parts -/
structure Grassland where
  areaA : ℝ
  areaB : ℝ
  growthRate : ℝ
  cowEatingRate : ℝ

/-- The conditions of the problem -/
def problem_conditions (g : Grassland) : Prop :=
  g.areaA > 0 ∧ g.areaB > 0 ∧ g.areaA ≠ g.areaB ∧
  g.growthRate > 0 ∧ g.cowEatingRate > 0 ∧
  g.areaA * g.growthRate = 7 * g.cowEatingRate ∧
  g.areaB * g.growthRate = 4 * g.cowEatingRate ∧
  7 * g.growthRate = g.areaA * g.growthRate

/-- The theorem stating the ratio of areas -/
theorem grassland_area_ratio (g : Grassland) :
  problem_conditions g → g.areaA / g.areaB = 105 / 44 :=
by sorry

end grassland_area_ratio_l3303_330381


namespace emily_egg_collection_l3303_330344

theorem emily_egg_collection (num_baskets : ℕ) (eggs_per_basket : ℕ) 
  (h1 : num_baskets = 303) 
  (h2 : eggs_per_basket = 28) : 
  num_baskets * eggs_per_basket = 8484 := by
  sorry

end emily_egg_collection_l3303_330344


namespace number_ratio_l3303_330349

theorem number_ratio (A B C : ℚ) (k : ℤ) (h1 : A = 2 * B) (h2 : A = k * C)
  (h3 : (A + B + C) / 3 = 88) (h4 : A - C = 96) : A / C = 15 / 7 := by
  sorry

end number_ratio_l3303_330349


namespace train_length_proof_l3303_330301

def train_problem (distance_apart : ℝ) (train2_length : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time_to_meet : ℝ) : Prop :=
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  let distance_covered := relative_speed * time_to_meet
  let train1_length := distance_covered - train2_length
  train1_length = 430

theorem train_length_proof :
  train_problem 630 200 90 72 13.998880089592832 :=
sorry

end train_length_proof_l3303_330301


namespace intersection_centroids_exist_l3303_330358

/-- Represents a point on the grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a line on the grid -/
inductive GridLine
  | Horizontal (y : Int)
  | Vertical (x : Int)

/-- The grid size -/
def gridSize : Nat := 4030

/-- The number of selected lines in each direction -/
def selectedLines : Nat := 2017

/-- Checks if a point is within the grid bounds -/
def isWithinGrid (p : GridPoint) : Prop :=
  -gridSize / 2 ≤ p.x ∧ p.x ≤ gridSize / 2 ∧
  -gridSize / 2 ≤ p.y ∧ p.y ≤ gridSize / 2

/-- Checks if a point is an intersection of selected lines -/
def isIntersection (p : GridPoint) (horizontalLines : List Int) (verticalLines : List Int) : Prop :=
  p.y ∈ horizontalLines ∧ p.x ∈ verticalLines

/-- Calculates the centroid of a triangle -/
def centroid (a b c : GridPoint) : GridPoint :=
  { x := (a.x + b.x + c.x) / 3
  , y := (a.y + b.y + c.y) / 3 }

/-- The main theorem -/
theorem intersection_centroids_exist 
  (horizontalLines : List Int) 
  (verticalLines : List Int) 
  (h1 : horizontalLines.length = selectedLines)
  (h2 : verticalLines.length = selectedLines)
  (h3 : ∀ y ∈ horizontalLines, -gridSize / 2 ≤ y ∧ y ≤ gridSize / 2)
  (h4 : ∀ x ∈ verticalLines, -gridSize / 2 ≤ x ∧ x ≤ gridSize / 2) :
  ∃ (a b c d e f : GridPoint),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    isWithinGrid a ∧ isWithinGrid b ∧ isWithinGrid c ∧
    isWithinGrid d ∧ isWithinGrid e ∧ isWithinGrid f ∧
    isIntersection a horizontalLines verticalLines ∧
    isIntersection b horizontalLines verticalLines ∧
    isIntersection c horizontalLines verticalLines ∧
    isIntersection d horizontalLines verticalLines ∧
    isIntersection e horizontalLines verticalLines ∧
    isIntersection f horizontalLines verticalLines ∧
    centroid a b c = { x := 0, y := 0 } ∧
    centroid d e f = { x := 0, y := 0 } :=
  by sorry


end intersection_centroids_exist_l3303_330358


namespace triangle_inequality_l3303_330333

theorem triangle_inequality (a b c : ℝ) (S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0) 
  (h_S : S = Real.sqrt (((a + b + c) / 2) * (((a + b + c) / 2) - a) * 
    (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c))) :
  a^2 + b^2 + c^2 - (a-b)^2 - (b-c)^2 - (c-a)^2 ≥ 4 * Real.sqrt 3 * S := by
  sorry

end triangle_inequality_l3303_330333


namespace rachel_bought_seven_chairs_l3303_330342

/-- Calculates the number of chairs Rachel bought given the number of tables,
    time spent per furniture piece, and total time spent. -/
def chairs_bought (num_tables : ℕ) (time_per_piece : ℕ) (total_time : ℕ) : ℕ :=
  (total_time - num_tables * time_per_piece) / time_per_piece

/-- Theorem stating that Rachel bought 7 chairs given the problem conditions. -/
theorem rachel_bought_seven_chairs :
  chairs_bought 3 4 40 = 7 := by
  sorry

end rachel_bought_seven_chairs_l3303_330342


namespace sin_240_degrees_l3303_330365

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l3303_330365


namespace reading_time_difference_l3303_330354

/-- The difference in reading time between two people reading the same book -/
theorem reading_time_difference 
  (jonathan_speed : ℝ) 
  (alice_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : jonathan_speed = 150) 
  (h2 : alice_speed = 75) 
  (h3 : book_pages = 450) : 
  (book_pages / alice_speed - book_pages / jonathan_speed) * 60 = 180 := by
  sorry

end reading_time_difference_l3303_330354


namespace imaginary_part_of_z_l3303_330380

theorem imaginary_part_of_z (z : ℂ) (h : z + Complex.abs z = 1 + 2*I) : 
  Complex.im z = 2 := by
  sorry

end imaginary_part_of_z_l3303_330380


namespace inequality_solution_set_l3303_330308

theorem inequality_solution_set (x : ℝ) :
  (|2*x + 1| - 2*|x - 1| > 0) ↔ (x > 0) := by sorry

end inequality_solution_set_l3303_330308


namespace units_digit_of_7_to_2010_l3303_330348

theorem units_digit_of_7_to_2010 : (7^2010) % 10 = 9 := by sorry

end units_digit_of_7_to_2010_l3303_330348


namespace curve_is_hyperbola_with_foci_on_y_axis_l3303_330341

theorem curve_is_hyperbola_with_foci_on_y_axis (θ : Real) 
  (h1 : π < θ ∧ θ < 3*π/2) -- θ is in the third quadrant
  (h2 : ∀ x y : Real, x^2 + y^2 * Real.sin θ = Real.cos θ) -- curve equation
  : ∃ (a b : Real), 
    a > 0 ∧ b > 0 ∧ 
    (∀ x y : Real, y^2 / b^2 - x^2 / a^2 = 1) ∧ -- standard form of hyperbola with foci on y-axis
    (∃ c : Real, c > 0 ∧ c^2 = a^2 + b^2) -- condition for foci on y-axis
  := by sorry

end curve_is_hyperbola_with_foci_on_y_axis_l3303_330341


namespace probability_all_sides_of_decagon_l3303_330328

/-- A regular decagon --/
structure RegularDecagon where

/-- A triangle formed from three vertices of a regular decagon --/
structure DecagonTriangle where
  decagon : RegularDecagon
  vertex1 : Nat
  vertex2 : Nat
  vertex3 : Nat

/-- Predicate to check if three vertices are sequentially adjacent in a decagon --/
def are_sequential_adjacent (v1 v2 v3 : Nat) : Prop :=
  (v2 = (v1 + 1) % 10) ∧ (v3 = (v2 + 1) % 10)

/-- Predicate to check if a triangle's sides are all sides of the decagon --/
def all_sides_of_decagon (t : DecagonTriangle) : Prop :=
  are_sequential_adjacent t.vertex1 t.vertex2 t.vertex3

/-- The total number of possible triangles in a decagon --/
def total_triangles : Nat := 120

/-- The number of triangles with all sides being sides of the decagon --/
def favorable_triangles : Nat := 10

/-- The main theorem --/
theorem probability_all_sides_of_decagon :
  (favorable_triangles : ℚ) / total_triangles = 1 / 12 := by
  sorry

end probability_all_sides_of_decagon_l3303_330328


namespace school_fire_problem_l3303_330399

/-- Represents the initial state and changes in a school after a fire incident -/
structure SchoolState where
  initialClassCount : ℕ
  initialStudentsPerClass : ℕ
  firstUnusableClasses : ℕ
  firstAddedStudents : ℕ
  secondUnusableClasses : ℕ
  secondAddedStudents : ℕ

/-- Calculates the total number of students after the changes -/
def totalStudentsAfterChanges (s : SchoolState) : ℕ :=
  let remainingClasses := s.initialClassCount - s.firstUnusableClasses - s.secondUnusableClasses
  remainingClasses * (s.initialStudentsPerClass + s.firstAddedStudents + s.secondAddedStudents)

/-- Theorem stating that the initial number of students in the school was 900 -/
theorem school_fire_problem (s : SchoolState) 
  (h1 : s.firstUnusableClasses = 6)
  (h2 : s.firstAddedStudents = 5)
  (h3 : s.secondUnusableClasses = 10)
  (h4 : s.secondAddedStudents = 15)
  (h5 : totalStudentsAfterChanges s = s.initialClassCount * s.initialStudentsPerClass) :
  s.initialClassCount * s.initialStudentsPerClass = 900 := by
  sorry

end school_fire_problem_l3303_330399


namespace arithmetic_sequence_middle_term_positive_l3303_330320

/-- Given an arithmetic sequence {a_n} where S_n denotes the sum of its first n terms,
    if S_(2k+1) > 0, then a_(k+1) > 0. -/
theorem arithmetic_sequence_middle_term_positive
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (k : ℕ)      -- An arbitrary natural number
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0)  -- Arithmetic sequence condition
  (h_sum : ∀ n, S n = (n * (a 0 + a (n - 1))) / 2)  -- Sum formula for arithmetic sequence
  (h_positive : S (2 * k + 1) > 0)  -- Given condition
  : a (k + 1) > 0 :=
sorry

end arithmetic_sequence_middle_term_positive_l3303_330320


namespace provisions_duration_l3303_330379

/-- Given provisions for a certain number of boys and days, calculate how long the provisions will last with additional boys. -/
theorem provisions_duration (initial_boys : ℕ) (initial_days : ℕ) (additional_boys : ℕ) :
  let total_boys := initial_boys + additional_boys
  let new_days := (initial_boys * initial_days) / total_boys
  initial_boys = 1500 → initial_days = 25 → additional_boys = 350 →
  ⌊(new_days : ℚ)⌋ = 20 := by
  sorry

end provisions_duration_l3303_330379


namespace other_root_of_quadratic_l3303_330360

theorem other_root_of_quadratic (a : ℝ) : 
  ((-1)^2 + a*(-1) - 2 = 0) → (2^2 + a*2 - 2 = 0) := by
  sorry

end other_root_of_quadratic_l3303_330360


namespace vector_sum_equals_result_l3303_330395

def vector_a : ℝ × ℝ := (0, -1)
def vector_b : ℝ × ℝ := (3, 2)

theorem vector_sum_equals_result : 2 • vector_a + vector_b = (3, 0) := by sorry

end vector_sum_equals_result_l3303_330395


namespace x_coordinate_of_Q_l3303_330398

theorem x_coordinate_of_Q (P Q : ℝ × ℝ) (α : ℝ) : 
  P = (3/5, 4/5) →
  (Q.1 < 0 ∧ Q.2 < 0) →
  Real.sqrt (Q.1^2 + Q.2^2) = 1 →
  α = Real.arccos (3/5) →
  α + 3 * Real.pi / 4 = Real.arccos Q.1 →
  Q.1 = -7 * Real.sqrt 2 / 10 :=
by sorry

end x_coordinate_of_Q_l3303_330398


namespace line_point_order_l3303_330303

theorem line_point_order (b : ℝ) (y₁ y₂ y₃ : ℝ) : 
  (y₁ = 3 * (-2.3) + b) → 
  (y₂ = 3 * (-1.3) + b) → 
  (y₃ = 3 * 2.7 + b) → 
  y₁ < y₂ ∧ y₂ < y₃ :=
by sorry

end line_point_order_l3303_330303


namespace labor_costs_l3303_330335

/-- Calculate the overall labor costs for one day given the salaries of different workers -/
theorem labor_costs (worker_salary : ℕ) : 
  worker_salary = 100 →
  (2 * worker_salary) + (2 * worker_salary) + (5/2 * worker_salary) = 650 := by
  sorry

#check labor_costs

end labor_costs_l3303_330335


namespace two_white_balls_probability_l3303_330387

/-- The probability of drawing two white balls successively without replacement
    from a box containing 8 white balls and 9 black balls is 7/34. -/
theorem two_white_balls_probability :
  let total_balls : ℕ := 8 + 9
  let white_balls : ℕ := 8
  let black_balls : ℕ := 9
  let prob_first_white : ℚ := white_balls / total_balls
  let prob_second_white : ℚ := (white_balls - 1) / (total_balls - 1)
  prob_first_white * prob_second_white = 7 / 34 := by
  sorry

end two_white_balls_probability_l3303_330387


namespace sum_of_repeated_digit_numbers_theorem_l3303_330372

theorem sum_of_repeated_digit_numbers_theorem :
  ∃ (a b c : ℕ),
    (∃ (d : ℕ), a = d * 11111 ∧ d < 10) ∧
    (∃ (e : ℕ), b = e * 1111 ∧ e < 10) ∧
    (∃ (f : ℕ), c = f * 111 ∧ f < 10) ∧
    (10000 ≤ a + b + c ∧ a + b + c < 100000) ∧
    (∃ (v w x y z : ℕ),
      v < 10 ∧ w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧
      v ≠ w ∧ v ≠ x ∧ v ≠ y ∧ v ≠ z ∧
      w ≠ x ∧ w ≠ y ∧ w ≠ z ∧
      x ≠ y ∧ x ≠ z ∧
      y ≠ z ∧
      a + b + c = v * 10000 + w * 1000 + x * 100 + y * 10 + z) :=
by
  sorry

end sum_of_repeated_digit_numbers_theorem_l3303_330372


namespace solution_x_l3303_330300

theorem solution_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 104) : x = 34 := by
  sorry

end solution_x_l3303_330300


namespace solve_equation_and_expression_l3303_330356

theorem solve_equation_and_expression (x : ℝ) (h : 5 * x - 7 = 15 * x + 13) : 
  3 * (x - 4) + 2 = -16 := by
sorry

end solve_equation_and_expression_l3303_330356


namespace candy_necklace_problem_l3303_330373

/-- Candy necklace problem -/
theorem candy_necklace_problem (blocks : ℕ) (pieces_per_block : ℕ) (people : ℕ) 
  (h1 : blocks = 3) 
  (h2 : pieces_per_block = 30) 
  (h3 : people = 9) :
  (blocks * pieces_per_block) / people = 10 := by
  sorry

end candy_necklace_problem_l3303_330373


namespace max_volume_right_triangle_rotation_l3303_330392

theorem max_volume_right_triangle_rotation (a b c : ℝ) : 
  a = 3 → b = 4 → c = 5 → a^2 + b^2 = c^2 →
  (max (1/3 * Real.pi * a^2 * b) (max (1/3 * Real.pi * b^2 * a) (1/3 * Real.pi * (2 * (1/2 * a * b) / c)^2 * c))) = 16 * Real.pi := by
  sorry

end max_volume_right_triangle_rotation_l3303_330392


namespace second_highest_coefficient_of_g_l3303_330316

/-- Given a polynomial g(x) satisfying g(x + 1) - g(x) = 6x^2 + 4x + 2 for all x,
    prove that the second highest coefficient of g(x) is 2/3 -/
theorem second_highest_coefficient_of_g (g : ℝ → ℝ) 
  (h : ∀ x, g (x + 1) - g x = 6 * x^2 + 4 * x + 2) :
  ∃ a b c d : ℝ, (∀ x, g x = a * x^3 + b * x^2 + c * x + d) ∧ b = 2/3 := by
  sorry

end second_highest_coefficient_of_g_l3303_330316


namespace sum_of_integers_l3303_330343

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val^2 + y.val^2 = 130) 
  (h2 : x.val * y.val = 36) : 
  x.val + y.val = Real.sqrt 202 := by
  sorry

end sum_of_integers_l3303_330343


namespace inequality_proof_l3303_330382

theorem inequality_proof (x₁ x₂ x₃ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) :
  (x₁^2 + x₂^2 + x₃^2)^3 ≤ 3 * (x₁^3 + x₂^3 + x₃^3)^2 := by
  sorry

end inequality_proof_l3303_330382


namespace closed_under_subtraction_l3303_330388

/-- A set of integers with special properties -/
structure SpecialIntegerSet where
  M : Set Int
  has_pos : ∃ x ∈ M, x > 0
  has_neg : ∃ x ∈ M, x < 0
  closed_double : ∀ a ∈ M, (2 * a) ∈ M
  closed_sum : ∀ a b, a ∈ M → b ∈ M → (a + b) ∈ M

/-- The main theorem: M is closed under subtraction -/
theorem closed_under_subtraction (S : SpecialIntegerSet) :
  ∀ a b, a ∈ S.M → b ∈ S.M → (a - b) ∈ S.M := by
  sorry

end closed_under_subtraction_l3303_330388


namespace seven_couples_handshakes_l3303_330313

/-- The number of handshakes in a gathering of couples -/
def handshakes (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 7 couples, where each person shakes hands with everyone
    except their spouse and one other person, the total number of handshakes is 77. -/
theorem seven_couples_handshakes :
  handshakes 7 = 77 := by
  sorry

end seven_couples_handshakes_l3303_330313


namespace smallest_class_size_l3303_330310

theorem smallest_class_size : 
  (∃ n : ℕ, n > 30 ∧ 
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 
      n = 3 * x + y ∧ 
      y = x + 1) ∧
    (∀ m : ℕ, m > 30 → 
      (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 
        m = 3 * a + b ∧ 
        b = a + 1) → 
      m ≥ n)) →
  (∃ n : ℕ, n = 33 ∧ n > 30 ∧ 
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 
      n = 3 * x + y ∧ 
      y = x + 1) ∧
    (∀ m : ℕ, m > 30 → 
      (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 
        m = 3 * a + b ∧ 
        b = a + 1) → 
      m ≥ n)) :=
by
  sorry

end smallest_class_size_l3303_330310


namespace no_real_solutions_l3303_330367

theorem no_real_solutions :
  ∀ x : ℝ, x ≠ 2 → (3 * x^2) / (x - 2) - (x + 4) / 4 + (5 - 3 * x) / (x - 2) + 2 ≠ 0 := by
  sorry

end no_real_solutions_l3303_330367


namespace point_on_line_l3303_330350

/-- A point (x, 3) lies on the straight line joining (1, 5) and (5, -3) if and only if x = 2 -/
theorem point_on_line (x : ℝ) : 
  (3 - 5) / (x - 1) = (-3 - 5) / (5 - 1) ↔ x = 2 := by
  sorry

end point_on_line_l3303_330350


namespace simplest_quadratic_radical_l3303_330307

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define what it means for a quadratic radical to be in its simplest form
def isSimplestQuadraticRadical (n : ℚ) : Prop :=
  ∃ (a : ℕ), isPrime a ∧ n = (a : ℚ).sqrt

-- Theorem statement
theorem simplest_quadratic_radical :
  let options : List ℚ := [9, 7, 20, (1/3 : ℚ)]
  ∃ (x : ℚ), x ∈ options ∧ isSimplestQuadraticRadical x ∧
    ∀ (y : ℚ), y ∈ options → y ≠ x → ¬(isSimplestQuadraticRadical y) :=
by sorry

end simplest_quadratic_radical_l3303_330307


namespace logarithm_equations_solutions_l3303_330325

theorem logarithm_equations_solutions :
  (∀ x : ℝ, x^2 - 1 > 0 ∧ x^2 - 1 ≠ 1 ∧ x^3 + 6 > 0 ∧ x^3 + 6 = 4*x^2 - x →
    x = 2 ∨ x = 3) ∧
  (∀ x : ℝ, x^2 - 4 > 0 ∧ x^3 + x > 0 ∧ x^3 + x ≠ 1 ∧ x^3 + x = 4*x^2 - 6 →
    x = 3) :=
by sorry

end logarithm_equations_solutions_l3303_330325


namespace correct_total_paid_l3303_330384

/-- Calculates the total amount paid after discount for a bulk purchase -/
def total_amount_paid (item_count : ℕ) (price_per_item : ℚ) (discount_amount : ℚ) (discount_threshold : ℚ) : ℚ :=
  let total_cost := item_count * price_per_item
  let discount_count := ⌊total_cost / discount_threshold⌋
  let total_discount := discount_count * discount_amount
  total_cost - total_discount

/-- Theorem stating the correct total amount paid for the given scenario -/
theorem correct_total_paid :
  total_amount_paid 400 (40/100) 2 10 = 128 := by
  sorry

end correct_total_paid_l3303_330384


namespace trees_cut_l3303_330359

theorem trees_cut (original : ℕ) (died : ℕ) (left : ℕ) (cut : ℕ) : 
  original = 86 → died = 15 → left = 48 → cut = original - died - left → cut = 23 := by
  sorry

end trees_cut_l3303_330359


namespace remaining_distance_is_546_point_5_l3303_330314

-- Define the total distance
def total_distance : ℝ := 1045

-- Define Amoli's driving
def amoli_speed : ℝ := 42
def amoli_time : ℝ := 3

-- Define Anayet's driving
def anayet_speed : ℝ := 61
def anayet_time : ℝ := 2.5

-- Define Bimal's driving
def bimal_speed : ℝ := 55
def bimal_time : ℝ := 4

-- Theorem statement
theorem remaining_distance_is_546_point_5 :
  total_distance - (amoli_speed * amoli_time + anayet_speed * anayet_time + bimal_speed * bimal_time) = 546.5 := by
  sorry

end remaining_distance_is_546_point_5_l3303_330314


namespace crazy_silly_school_books_read_l3303_330397

/-- The number of books read in the 'Crazy Silly School' series -/
def books_read (total_books unread_books : ℕ) : ℕ :=
  total_books - unread_books

/-- Theorem stating that the number of books read is 33 -/
theorem crazy_silly_school_books_read :
  books_read 50 17 = 33 := by
  sorry

end crazy_silly_school_books_read_l3303_330397


namespace m_value_theorem_l3303_330376

theorem m_value_theorem (m : ℕ) : 
  2^2000 - 3 * 2^1999 + 5 * 2^1998 - 2^1997 = m * 2^1997 → m = 5 := by
  sorry

end m_value_theorem_l3303_330376


namespace car_speed_problem_l3303_330352

/-- 
Given a car that travels for two hours, with speed x km/h in the first hour
and 60 km/h in the second hour, if the average speed is 79 km/h, 
then the speed x in the first hour must be 98 km/h.
-/
theorem car_speed_problem (x : ℝ) : 
  (x + 60) / 2 = 79 → x = 98 := by
  sorry

end car_speed_problem_l3303_330352


namespace candied_apple_price_l3303_330302

/-- Given the conditions of candy production and sales, prove the price of each candied apple. -/
theorem candied_apple_price :
  ∀ (num_apples num_grapes : ℕ) (grape_price total_earnings : ℚ),
    num_apples = 15 →
    num_grapes = 12 →
    grape_price = 3/2 →
    total_earnings = 48 →
    ∃ (apple_price : ℚ),
      apple_price * num_apples + grape_price * num_grapes = total_earnings ∧
      apple_price = 2 := by
sorry

end candied_apple_price_l3303_330302


namespace product_of_numbers_l3303_330383

theorem product_of_numbers (x y : ℝ) : 
  |x - y| = 12 → x^2 + y^2 = 245 → x * y = 50.30 := by
  sorry

end product_of_numbers_l3303_330383


namespace grassy_width_is_55_l3303_330315

/-- Represents the dimensions and cost of a rectangular plot with a gravel path -/
structure Plot where
  length : ℝ
  path_width : ℝ
  gravel_cost_per_sqm : ℝ
  total_gravel_cost : ℝ

/-- Calculates the width of the grassy area given the plot dimensions and gravel cost -/
def calculate_grassy_width (p : Plot) : ℝ :=
  sorry

/-- Theorem stating that for the given dimensions and cost, the grassy width is 55 meters -/
theorem grassy_width_is_55 (p : Plot) 
  (h1 : p.length = 110)
  (h2 : p.path_width = 2.5)
  (h3 : p.gravel_cost_per_sqm = 0.6)
  (h4 : p.total_gravel_cost = 510) :
  calculate_grassy_width p = 55 :=
sorry

end grassy_width_is_55_l3303_330315


namespace cos_240_degrees_l3303_330374

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end cos_240_degrees_l3303_330374


namespace phone_storage_theorem_l3303_330331

/-- Calculates the maximum number of songs that can be stored on a phone given the total storage, used storage, and size of each song. -/
def max_songs (total_storage : ℕ) (used_storage : ℕ) (song_size : ℕ) : ℕ :=
  ((total_storage - used_storage) * 1000) / song_size

/-- Theorem stating that given a phone with 16 GB total storage, 4 GB already used, and songs of 30 MB each, the maximum number of additional songs that can be stored is 400. -/
theorem phone_storage_theorem :
  max_songs 16 4 30 = 400 := by
  sorry

end phone_storage_theorem_l3303_330331


namespace square_carpet_side_length_l3303_330396

theorem square_carpet_side_length 
  (floor_length : ℝ) 
  (floor_width : ℝ) 
  (uncovered_area : ℝ) 
  (h1 : floor_length = 10)
  (h2 : floor_width = 8)
  (h3 : uncovered_area = 64)
  : ∃ (side_length : ℝ), 
    side_length^2 = floor_length * floor_width - uncovered_area ∧ 
    side_length = 4 := by
  sorry

end square_carpet_side_length_l3303_330396


namespace food_drive_problem_l3303_330345

/-- Food drive problem -/
theorem food_drive_problem (rachel_cans jaydon_cans mark_cans : ℕ) : 
  jaydon_cans = 2 * rachel_cans + 5 →
  mark_cans = 4 * jaydon_cans →
  rachel_cans + jaydon_cans + mark_cans = 135 →
  mark_cans = 100 := by
  sorry

#check food_drive_problem

end food_drive_problem_l3303_330345


namespace average_weight_of_three_l3303_330368

/-- Given the weights of three people with specific relationships, prove their average weight. -/
theorem average_weight_of_three (ishmael ponce jalen : ℝ) : 
  ishmael = ponce + 20 →
  ponce = jalen - 10 →
  jalen = 160 →
  (ishmael + ponce + jalen) / 3 = 160 := by
sorry

end average_weight_of_three_l3303_330368


namespace set_game_combinations_l3303_330390

theorem set_game_combinations (n : ℕ) (k : ℕ) (h1 : n = 81) (h2 : k = 3) :
  Nat.choose n k = 85320 := by
  sorry

end set_game_combinations_l3303_330390


namespace isosceles_right_triangle_l3303_330389

theorem isosceles_right_triangle (a b c : ℝ) :
  a = 2 * Real.sqrt 6 ∧ 
  b = 2 * Real.sqrt 3 ∧ 
  c = 2 * Real.sqrt 3 →
  (a^2 = b^2 + c^2) ∧ (b = c) :=
by sorry

end isosceles_right_triangle_l3303_330389


namespace smallest_four_digit_mod_seven_l3303_330327

theorem smallest_four_digit_mod_seven : 
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 7 = 6 → n ≥ 1000 :=
by sorry

end smallest_four_digit_mod_seven_l3303_330327


namespace diameter_is_65_l3303_330324

/-- Represents a circle with a diameter and a perpendicular chord --/
structure Circle where
  diameter : ℕ
  chord : ℕ
  is_two_digit : 10 ≤ diameter ∧ diameter < 100
  is_reversed : chord = (diameter % 10) * 10 + (diameter / 10)

/-- The distance from the center to the intersection of the chord and diameter --/
def center_to_intersection (c : Circle) : ℚ :=
  let r := c.diameter / 2
  let h := c.chord / 2
  ((r * r - h * h : ℚ) / (r * r)).sqrt

theorem diameter_is_65 (c : Circle) 
  (h_rational : ∃ (q : ℚ), center_to_intersection c = q) :
  c.diameter = 65 := by
  sorry

#check diameter_is_65

end diameter_is_65_l3303_330324


namespace nelly_winning_bid_l3303_330311

-- Define Joe's bid
def joes_bid : ℕ := 160000

-- Define Nelly's bid calculation
def nellys_bid : ℕ := 3 * joes_bid + 2000

-- Theorem to prove
theorem nelly_winning_bid : nellys_bid = 482000 := by
  sorry

end nelly_winning_bid_l3303_330311


namespace crushing_load_calculation_l3303_330321

theorem crushing_load_calculation (T H L : ℝ) : 
  T = 3 → H = 9 → L = (36 * T^3) / H^3 → L = 4/3 := by
  sorry

end crushing_load_calculation_l3303_330321


namespace complex_magnitude_equation_l3303_330332

theorem complex_magnitude_equation (t : ℝ) : 
  (t > 0 ∧ Complex.abs (8 + t * Complex.I) = 15) ↔ t = Real.sqrt 161 := by
  sorry

end complex_magnitude_equation_l3303_330332


namespace perpendicular_lines_minimum_product_l3303_330353

theorem perpendicular_lines_minimum_product (b a : ℝ) : 
  b > 0 → 
  ((b^2 + 1) * (-b^2) = -1) →
  ab ≥ 2 :=
by sorry

end perpendicular_lines_minimum_product_l3303_330353


namespace quadratic_equation_solution_l3303_330346

theorem quadratic_equation_solution :
  let x₁ : ℝ := (-1 + Real.sqrt 5) / 2
  let x₂ : ℝ := (-1 - Real.sqrt 5) / 2
  (x₁^2 + x₁ - 1 = 0) ∧ (x₂^2 + x₂ - 1 = 0) := by
  sorry

end quadratic_equation_solution_l3303_330346


namespace inner_quadrilateral_area_l3303_330317

/-- A square with side length 10 cm, partitioned by lines from corners to opposite midpoints -/
structure PartitionedSquare where
  side_length : ℝ
  is_ten_cm : side_length = 10

/-- The inner quadrilateral formed by the intersecting lines -/
def inner_quadrilateral (s : PartitionedSquare) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The area of the inner quadrilateral is 25 cm² -/
theorem inner_quadrilateral_area (s : PartitionedSquare) :
  area (inner_quadrilateral s) = 25 := by
  sorry

end inner_quadrilateral_area_l3303_330317


namespace smallest_four_digit_multiple_of_18_l3303_330393

theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 18 = 0 → n ≥ 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l3303_330393


namespace sandwich_combinations_l3303_330305

/-- The number of toppings available for sandwiches -/
def num_toppings : ℕ := 9

/-- The number of patty choices available for sandwiches -/
def num_patties : ℕ := 2

/-- The total number of different sandwich combinations -/
def total_combinations : ℕ := 2^num_toppings * num_patties

theorem sandwich_combinations :
  total_combinations = 1024 :=
sorry

end sandwich_combinations_l3303_330305


namespace tangent_points_coordinates_fixed_points_on_circle_l3303_330319

/-- Circle M with equation x^2 + (y-2)^2 = 1 -/
def circle_M (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

/-- Line l with equation x - 2y = 0 -/
def line_l (x y : ℝ) : Prop := x - 2*y = 0

/-- Point P lies on line l -/
def P_on_line_l (x y : ℝ) : Prop := line_l x y

/-- PA and PB are tangents to circle M -/
def tangents_to_M (xp yp xa ya xb yb : ℝ) : Prop :=
  circle_M xa ya ∧ circle_M xb yb ∧
  ((xp - xa) * xa + (yp - ya) * (ya - 2) = 0) ∧
  ((xp - xb) * xb + (yp - yb) * (yb - 2) = 0)

/-- Angle APB is 60 degrees -/
def angle_APB_60 (xp yp xa ya xb yb : ℝ) : Prop :=
  let v1x := xa - xp
  let v1y := ya - yp
  let v2x := xb - xp
  let v2y := yb - yp
  (v1x * v2x + v1y * v2y)^2 = 3 * ((v1x^2 + v1y^2) * (v2x^2 + v2y^2)) / 4

theorem tangent_points_coordinates :
  ∀ (xp yp xa ya xb yb : ℝ),
  P_on_line_l xp yp →
  tangents_to_M xp yp xa ya xb yb →
  angle_APB_60 xp yp xa ya xb yb →
  (xp = 0 ∧ yp = 0) ∨ (xp = 8/5 ∧ yp = 4/5) :=
sorry

theorem fixed_points_on_circle :
  ∀ (xp yp xa ya : ℝ),
  P_on_line_l xp yp →
  tangents_to_M xp yp xa ya xp yp →
  ∃ (t : ℝ),
  (1 - t) * xp + t * xa = 0 ∧ (1 - t) * yp + t * ya = 2 ∨
  (1 - t) * xp + t * xa = 4/5 ∧ (1 - t) * yp + t * ya = 2/5 :=
sorry

end tangent_points_coordinates_fixed_points_on_circle_l3303_330319


namespace max_pairs_after_loss_l3303_330323

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_remaining_pairs (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - (shoes_lost + 1) / 2

/-- Theorem stating that with 25 initial pairs and 9 shoes lost,
    the maximum number of complete pairs remaining is 20. -/
theorem max_pairs_after_loss : max_remaining_pairs 25 9 = 20 := by
  sorry

#eval max_remaining_pairs 25 9

end max_pairs_after_loss_l3303_330323


namespace complex_product_real_condition_l3303_330378

theorem complex_product_real_condition (a b c d : ℝ) :
  (Complex.I * b + a) * (Complex.I * d + c) ∈ Set.range Complex.ofReal ↔ a * d + b * c = 0 := by
  sorry

end complex_product_real_condition_l3303_330378


namespace king_middle_school_teachers_l3303_330391

/-- Calculates the number of teachers at King Middle School given the specified conditions -/
theorem king_middle_school_teachers :
  let num_students : ℕ := 1500
  let classes_per_student : ℕ := 5
  let classes_per_teacher : ℕ := 5
  let students_per_class : ℕ := 25
  let total_class_instances : ℕ := num_students * classes_per_student
  let unique_classes : ℕ := total_class_instances / students_per_class
  let num_teachers : ℕ := unique_classes / classes_per_teacher
  num_teachers = 60 := by sorry

end king_middle_school_teachers_l3303_330391


namespace least_positive_integer_multiple_of_53_l3303_330357

theorem least_positive_integer_multiple_of_53 :
  ∃ (x : ℕ+), (2 * x.val)^2 + 2 * 41 * (2 * x.val) + 41^2 ≡ 0 [MOD 53] ∧
  ∀ (y : ℕ+), ((2 * y.val)^2 + 2 * 41 * (2 * y.val) + 41^2 ≡ 0 [MOD 53]) → x ≤ y :=
by sorry

end least_positive_integer_multiple_of_53_l3303_330357


namespace birds_on_fence_l3303_330329

theorem birds_on_fence (initial_birds : ℕ) (new_birds : ℕ) : 
  initial_birds = 12 → new_birds = 8 → initial_birds + new_birds = 20 := by
  sorry

end birds_on_fence_l3303_330329


namespace man_work_days_l3303_330385

/-- Proves that if a woman can do a piece of work in 40 days and a man is 25% more efficient,
    then the man can do the same piece of work in 32 days. -/
theorem man_work_days (woman_days : ℕ) (man_efficiency : ℚ) :
  woman_days = 40 →
  man_efficiency = 1.25 →
  (woman_days : ℚ) / man_efficiency = 32 :=
by sorry

end man_work_days_l3303_330385


namespace m_range_l3303_330337

theorem m_range (m : ℝ) (h1 : m < 0) (h2 : ∀ x : ℝ, x^2 + m*x + 1 > 0) : m ∈ Set.Ioo (-2 : ℝ) 0 := by
  sorry

end m_range_l3303_330337


namespace units_digit_of_smallest_n_with_2016_digits_l3303_330339

theorem units_digit_of_smallest_n_with_2016_digits : ∃ n : ℕ,
  (∀ m : ℕ, 7 * m < 10^2015 → m < n) ∧
  7 * n ≥ 10^2015 ∧
  7 * n < 10^2016 ∧
  n % 10 = 6 :=
sorry

end units_digit_of_smallest_n_with_2016_digits_l3303_330339


namespace five_twelve_thirteen_pythagorean_triple_l3303_330369

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- Theorem stating that (5, 12, 13) is a Pythagorean triple -/
theorem five_twelve_thirteen_pythagorean_triple :
  is_pythagorean_triple 5 12 13 := by
  sorry

end five_twelve_thirteen_pythagorean_triple_l3303_330369


namespace rectangular_solid_surface_area_l3303_330309

/-- The total surface area of a rectangular solid -/
def total_surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 5 meters, width 4 meters, and depth 1 meter is 58 square meters -/
theorem rectangular_solid_surface_area :
  total_surface_area 5 4 1 = 58 := by
  sorry

end rectangular_solid_surface_area_l3303_330309


namespace inequality_proofs_l3303_330363

theorem inequality_proofs :
  (∀ (a b : ℝ), a > 0 → b > 0 → a^3 + b^3 ≥ a*b^2 + a^2*b) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y > 2 → (1 + y) / x < 2 ∨ (1 + x) / y < 2) :=
by sorry

end inequality_proofs_l3303_330363


namespace paco_marble_purchase_l3303_330330

theorem paco_marble_purchase : 
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 = 0.75 := by
  sorry

end paco_marble_purchase_l3303_330330


namespace three_primes_sum_l3303_330364

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def smallest_square_greater_than_15 : ℕ := 16

theorem three_primes_sum (p q r : ℕ) : 
  is_prime p → is_prime q → is_prime r →
  p + q + r = smallest_square_greater_than_15 →
  1 < p → p < q → q < r →
  p = 2 := by sorry

end three_primes_sum_l3303_330364


namespace asymptote_sum_l3303_330371

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x^3 + A*x^2 + B*x + C = 0 ↔ x = -3 ∨ x = 0 ∨ x = 2) → 
  A + B + C = -5 := by
  sorry

end asymptote_sum_l3303_330371


namespace field_division_l3303_330340

theorem field_division (total_area smaller_area larger_area : ℝ) : 
  total_area = 700 ∧ 
  smaller_area + larger_area = total_area ∧ 
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area = 315 :=
by sorry

end field_division_l3303_330340


namespace natural_number_operations_l3303_330355

theorem natural_number_operations (x y : ℕ) (h1 : x > y) (h2 : x + y + (x - y) + x * y + x / y = 243) :
  (x = 54 ∧ y = 2) ∨ (x = 24 ∧ y = 8) := by
  sorry

end natural_number_operations_l3303_330355


namespace trail_mix_portions_l3303_330370

theorem trail_mix_portions (nuts dried_fruit chocolate coconut : ℕ) 
  (h1 : nuts = 16) (h2 : dried_fruit = 6) (h3 : chocolate = 8) (h4 : coconut = 4) :
  Nat.gcd nuts (Nat.gcd dried_fruit (Nat.gcd chocolate coconut)) = 2 := by
  sorry

end trail_mix_portions_l3303_330370


namespace women_count_correct_l3303_330318

/-- The number of women working with men to complete a job -/
def num_women : ℕ := 15

/-- The number of men working on the job -/
def num_men : ℕ := 10

/-- The number of days it takes for the group to complete the job -/
def group_days : ℕ := 6

/-- The number of days it takes for one man to complete the job -/
def man_days : ℕ := 100

/-- The number of days it takes for one woman to complete the job -/
def woman_days : ℕ := 225

/-- Theorem stating that the number of women working with the men is correct -/
theorem women_count_correct :
  (num_men : ℚ) / man_days + (num_women : ℚ) / woman_days = 1 / group_days :=
sorry


end women_count_correct_l3303_330318


namespace stake_B_maximizes_grazing_area_l3303_330375

/-- Represents a stake on the edge of the pond -/
inductive Stake
| A
| B
| C
| D

/-- The side length of the square pond in meters -/
def pondSideLength : ℝ := 12

/-- The distance between adjacent stakes in meters -/
def stakesDistance : ℝ := 3

/-- The length of the rope in meters -/
def ropeLength : ℝ := 4

/-- Calculates the grazing area for a given stake -/
noncomputable def grazingArea (s : Stake) : ℝ :=
  match s with
  | Stake.A => 4.25 * Real.pi
  | Stake.B => 8 * Real.pi
  | Stake.C => 4.25 * Real.pi
  | Stake.D => 4.25 * Real.pi

/-- Theorem stating that stake B maximizes the grazing area -/
theorem stake_B_maximizes_grazing_area :
  ∀ s : Stake, grazingArea Stake.B ≥ grazingArea s :=
sorry


end stake_B_maximizes_grazing_area_l3303_330375


namespace bridge_length_calculation_bridge_length_result_l3303_330394

/-- Calculates the length of a bridge given train specifications --/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length

/-- The length of the bridge is approximately 299.95 meters --/
theorem bridge_length_result : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |bridge_length_calculation 200 45 40 - 299.95| < ε :=
sorry

end bridge_length_calculation_bridge_length_result_l3303_330394


namespace alan_age_is_29_l3303_330338

/-- Represents the ages of Alan and Chris --/
structure Ages where
  alan : ℕ
  chris : ℕ

/-- The condition that the sum of their ages is 52 --/
def sum_of_ages (ages : Ages) : Prop :=
  ages.alan + ages.chris = 52

/-- The complex age relationship between Alan and Chris --/
def age_relationship (ages : Ages) : Prop :=
  ages.chris = ages.alan - (ages.alan - (ages.alan - (ages.alan / 3)))

/-- The theorem stating Alan's age is 29 given the conditions --/
theorem alan_age_is_29 (ages : Ages) 
  (h1 : sum_of_ages ages) 
  (h2 : age_relationship ages) : 
  ages.alan = 29 := by
  sorry


end alan_age_is_29_l3303_330338


namespace all_cars_meet_time_prove_all_cars_meet_time_l3303_330386

/-- Represents a car on a circular track -/
structure Car where
  speed : ℝ
  direction : Bool -- true for clockwise, false for counterclockwise

/-- Represents the race scenario -/
structure RaceScenario where
  track_length : ℝ
  car_a : Car
  car_b : Car
  car_c : Car
  first_ac_meet : ℝ
  first_ab_meet : ℝ

/-- Theorem stating when all three cars meet for the first time -/
theorem all_cars_meet_time (race : RaceScenario) : ℝ :=
  let first_ac_meet := race.first_ac_meet
  let first_ab_meet := race.first_ab_meet
  371

#check all_cars_meet_time

/-- Main theorem proving the time when all three cars meet -/
theorem prove_all_cars_meet_time (race : RaceScenario) 
  (h1 : race.car_a.direction = true)
  (h2 : race.car_b.direction = true)
  (h3 : race.car_c.direction = false)
  (h4 : race.car_a.speed ≠ race.car_b.speed)
  (h5 : race.car_a.speed ≠ race.car_c.speed)
  (h6 : race.car_b.speed ≠ race.car_c.speed)
  (h7 : race.first_ac_meet = 7)
  (h8 : race.first_ab_meet = 53)
  : all_cars_meet_time race = 371 := by
  sorry

#check prove_all_cars_meet_time

end all_cars_meet_time_prove_all_cars_meet_time_l3303_330386


namespace ascending_order_of_rationals_l3303_330351

theorem ascending_order_of_rationals (a b : ℚ) 
  (ha : a > 0) (hb : b < 0) (hab : a + b < 0) :
  b < -a ∧ -a < a ∧ a < -b :=
by sorry

end ascending_order_of_rationals_l3303_330351
