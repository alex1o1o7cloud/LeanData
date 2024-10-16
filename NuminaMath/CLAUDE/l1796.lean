import Mathlib

namespace NUMINAMATH_CALUDE_smallest_number_l1796_179608

theorem smallest_number (a b c d : ℚ) (ha : a = 1) (hb : b = -2) (hc : c = 0) (hd : d = -5/2) :
  d < b ∧ b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1796_179608


namespace NUMINAMATH_CALUDE_integral_3_minus_7x_squared_cos_2x_l1796_179656

theorem integral_3_minus_7x_squared_cos_2x (π : ℝ) :
  (∫ x in (0 : ℝ)..(2 * π), (3 - 7 * x^2) * Real.cos (2 * x)) = -7 * π := by
  sorry

end NUMINAMATH_CALUDE_integral_3_minus_7x_squared_cos_2x_l1796_179656


namespace NUMINAMATH_CALUDE_next_shared_meeting_proof_l1796_179665

/-- The number of days between drama club meetings -/
def drama_interval : ℕ := 3

/-- The number of days between choir meetings -/
def choir_interval : ℕ := 5

/-- The number of days until both groups meet again -/
def next_shared_meeting : ℕ := 30

theorem next_shared_meeting_proof :
  ∃ (n : ℕ), n > 0 ∧ n * drama_interval = next_shared_meeting ∧ n * choir_interval = next_shared_meeting :=
sorry

end NUMINAMATH_CALUDE_next_shared_meeting_proof_l1796_179665


namespace NUMINAMATH_CALUDE_rope_segments_pattern_l1796_179694

/-- The number of segments produced by folding a rope n times and cutting it in the middle -/
def num_segments (n : ℕ) : ℕ := 2^n + 1

/-- Theorem stating that the number of segments follows the pattern 2^n + 1 -/
theorem rope_segments_pattern (n : ℕ) :
  (num_segments 1 = 3) ∧
  (num_segments 2 = 5) ∧
  (num_segments 3 = 9) →
  num_segments n = 2^n + 1 :=
by sorry

end NUMINAMATH_CALUDE_rope_segments_pattern_l1796_179694


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l1796_179627

/-- Calculates the total number of wheels in a parking lot --/
def total_wheels (num_cars num_motorcycles num_trucks num_vans : ℕ) : ℕ :=
  let car_wheels := 4
  let motorcycle_wheels := 2
  let truck_wheels := 6
  let van_wheels := 4
  num_cars * car_wheels + 
  num_motorcycles * motorcycle_wheels + 
  num_trucks * truck_wheels + 
  num_vans * van_wheels

/-- The number of wheels in Dylan's parents' vehicles --/
def parents_wheels : ℕ := 8

theorem parking_lot_wheels : 
  total_wheels 7 4 3 2 + parents_wheels = 62 := by sorry

end NUMINAMATH_CALUDE_parking_lot_wheels_l1796_179627


namespace NUMINAMATH_CALUDE_cosine_sum_simplification_l1796_179639

theorem cosine_sum_simplification :
  let x := Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17)
  x = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_simplification_l1796_179639


namespace NUMINAMATH_CALUDE_product_of_fractions_l1796_179662

theorem product_of_fractions :
  (3 : ℚ) / 5 * (4 : ℚ) / 7 * (5 : ℚ) / 9 = (4 : ℚ) / 21 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1796_179662


namespace NUMINAMATH_CALUDE_camp_kids_count_l1796_179600

theorem camp_kids_count (total : ℕ) (soccer : ℕ) (morning : ℕ) (afternoon : ℕ) :
  soccer = total / 2 →
  morning = soccer / 4 →
  afternoon = 750 →
  afternoon = soccer * 3 / 4 →
  total = 2000 := by
sorry

end NUMINAMATH_CALUDE_camp_kids_count_l1796_179600


namespace NUMINAMATH_CALUDE_book_sale_gain_percentage_l1796_179636

/-- Represents the problem of calculating the gain percentage on a book sale --/
theorem book_sale_gain_percentage 
  (total_cost : ℝ) 
  (cost_book1 : ℝ) 
  (loss_percentage : ℝ) 
  (total_cost_eq : total_cost = 360) 
  (cost_book1_eq : cost_book1 = 210) 
  (loss_percentage_eq : loss_percentage = 15) 
  (cost_book2_eq : total_cost = cost_book1 + cost_book2) 
  (same_selling_price : 
    cost_book1 * (1 - loss_percentage / 100) = 
    cost_book2 * (1 + gain_percentage / 100)) : 
  gain_percentage = 19 := by sorry

end NUMINAMATH_CALUDE_book_sale_gain_percentage_l1796_179636


namespace NUMINAMATH_CALUDE_exists_distinct_subsequences_l1796_179638

/-- A binary sequence is a function from ℕ to Bool -/
def BinarySequence := ℕ → Bool

/-- Cyclic index function to wrap around the sequence -/
def cyclicIndex (len : ℕ) (i : ℕ) : ℕ :=
  i % len

/-- Check if all n-length subsequences in a sequence of length 2^n are distinct -/
def allSubsequencesDistinct (n : ℕ) (seq : BinarySequence) : Prop :=
  ∀ i j, i < 2^n → j < 2^n → i ≠ j →
    (∃ k, k < n ∧ seq (cyclicIndex (2^n) (i + k)) ≠ seq (cyclicIndex (2^n) (j + k)))

/-- Main theorem: For any positive n, there exists a binary sequence of length 2^n
    where all n-length subsequences are distinct when considered cyclically -/
theorem exists_distinct_subsequences (n : ℕ) (hn : n > 0) :
  ∃ seq : BinarySequence, allSubsequencesDistinct n seq :=
sorry

end NUMINAMATH_CALUDE_exists_distinct_subsequences_l1796_179638


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1796_179663

theorem imaginary_part_of_z (z : ℂ) : 
  z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I →
  z.im = (Real.sqrt 2 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1796_179663


namespace NUMINAMATH_CALUDE_triangle_shortest_side_l1796_179611

theorem triangle_shortest_side (a b c : ℕ) (h : ℕ) : 
  a = 24 →                                  -- One side is 24
  a + b + c = 66 →                          -- Perimeter is 66
  b ≤ c →                                   -- b is the shortest side
  ∃ (A : ℕ), A * A = 297 * (33 - b) * (b - 9) →  -- Area is an integer (using Heron's formula)
  24 * h = 2 * A →                          -- Integer altitude condition
  b = 15 := by sorry

end NUMINAMATH_CALUDE_triangle_shortest_side_l1796_179611


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1796_179643

theorem sufficient_not_necessary : 
  let A := {x : ℝ | 1 < x ∧ x < 2}
  let B := {x : ℝ | x < 2}
  (A ⊂ B) ∧ (B \ A).Nonempty := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1796_179643


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_side_c_equation_l1796_179605

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The cosine law for triangle ABC --/
axiom cosine_law (t : Triangle) : t.c^2 = t.a^2 + t.b^2 - 2*t.a*t.b*Real.cos t.C

/-- The given condition c/2 = b - a cos(C) --/
def condition (t : Triangle) : Prop := t.c/2 = t.b - t.a * Real.cos t.C

theorem angle_A_is_60_degrees (t : Triangle) (h : condition t) : Real.cos t.A = 1/2 := by sorry

theorem side_c_equation (t : Triangle) (h : condition t) (ha : t.a = Real.sqrt 15) (hb : t.b = 4) :
  t.c^2 - 4*t.c + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_side_c_equation_l1796_179605


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1796_179697

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := circle_radius / 4
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof (h1 : square_area = 784) (h2 : rectangle_breadth = 5) :
  rectangle_area square_area rectangle_breadth = 35 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1796_179697


namespace NUMINAMATH_CALUDE_suv_max_distance_l1796_179681

/-- Represents the fuel efficiency of an SUV in miles per gallon -/
structure FuelEfficiency where
  highway : ℝ
  city : ℝ

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and available fuel -/
def max_distance (efficiency : FuelEfficiency) (fuel : ℝ) : ℝ :=
  efficiency.highway * fuel

/-- Theorem: The maximum distance an SUV with 12.2 mpg highway efficiency can travel on 23 gallons of fuel is 280.6 miles -/
theorem suv_max_distance :
  let suv_efficiency : FuelEfficiency := { highway := 12.2, city := 7.6 }
  let available_fuel : ℝ := 23
  max_distance suv_efficiency available_fuel = 280.6 := by
  sorry


end NUMINAMATH_CALUDE_suv_max_distance_l1796_179681


namespace NUMINAMATH_CALUDE_four_people_permutations_l1796_179602

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- There are 4 distinct individuals -/
def num_people : ℕ := 4

/-- Theorem: The number of ways 4 people can stand in a line is 24 -/
theorem four_people_permutations : permutations num_people = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_people_permutations_l1796_179602


namespace NUMINAMATH_CALUDE_sarah_mia_games_together_l1796_179632

def total_players : ℕ := 12
def players_per_game : ℕ := 6

theorem sarah_mia_games_together :
  let total_combinations := Nat.choose total_players players_per_game
  let games_per_player := total_combinations / 2
  let other_players := total_players - 2
  games_per_player * (players_per_game - 1) / other_players = 210 := by
  sorry

end NUMINAMATH_CALUDE_sarah_mia_games_together_l1796_179632


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l1796_179612

theorem stratified_sampling_medium_stores
  (total_stores : ℕ)
  (medium_stores : ℕ)
  (sample_size : ℕ)
  (h1 : total_stores = 300)
  (h2 : medium_stores = 75)
  (h3 : sample_size = 20) :
  (medium_stores : ℚ) / total_stores * sample_size = 5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l1796_179612


namespace NUMINAMATH_CALUDE_parabola_vertex_l1796_179685

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = -5 * (x + 2)^2 - 6

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-2, -6)

/-- Theorem: The vertex of the parabola y = -5(x+2)^2 - 6 is at the point (-2, -6) -/
theorem parabola_vertex :
  ∀ (x y : ℝ), parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1796_179685


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l1796_179691

/-- A line passing through (1, 0) with slope 3 has the equation 3x - y - 3 = 0 -/
theorem line_equation_through_point_with_slope (x y : ℝ) :
  (3 : ℝ) * x - y - 3 = 0 ↔ (y - 0 = 3 * (x - 1) ∧ (1, 0) ∈ {p : ℝ × ℝ | (3 : ℝ) * p.1 - p.2 - 3 = 0}) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l1796_179691


namespace NUMINAMATH_CALUDE_inverse_A_times_B_l1796_179626

open Matrix

theorem inverse_A_times_B :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 1]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 2, 5]
  A⁻¹ * B = !![1/2, -1/2; 2, 5] := by
sorry

end NUMINAMATH_CALUDE_inverse_A_times_B_l1796_179626


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1796_179672

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n, sum_arithmetic a n / sum_arithmetic b n = (7 * n) / (n + 3)) →
  a.a 5 / b.a 5 = 21 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1796_179672


namespace NUMINAMATH_CALUDE_charity_fundraising_l1796_179658

theorem charity_fundraising (total_amount : ℕ) (num_people : ℕ) (amount_per_person : ℕ) :
  total_amount = 1500 →
  num_people = 6 →
  amount_per_person * num_people = total_amount →
  amount_per_person = 250 := by
  sorry

end NUMINAMATH_CALUDE_charity_fundraising_l1796_179658


namespace NUMINAMATH_CALUDE_sqrt_225_equals_15_l1796_179666

theorem sqrt_225_equals_15 : Real.sqrt 225 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_225_equals_15_l1796_179666


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l1796_179674

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  area_ratio : ℝ
  sum_of_parallel_sides : ℝ
  area_ratio_condition : area_ratio = 5 / 3
  sum_condition : AB + CD = sum_of_parallel_sides

/-- Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC to the area of triangle ADC
    is 5:3, and AB + CD = 160 cm, then AB = 100 cm -/
theorem trapezoid_segment_length (t : Trapezoid) (h : t.sum_of_parallel_sides = 160) : t.AB = 100 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_segment_length_l1796_179674


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inverse_l1796_179668

theorem sqrt_x_plus_inverse (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 49) : 
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 51 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inverse_l1796_179668


namespace NUMINAMATH_CALUDE_quadratic_equation_d_has_two_distinct_roots_l1796_179664

/-- Discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Predicate for a quadratic equation having two distinct real roots -/
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ discriminant a b c > 0

theorem quadratic_equation_d_has_two_distinct_roots :
  has_two_distinct_real_roots 1 2 (-1) ∧
  ¬has_two_distinct_real_roots 1 0 4 ∧
  ¬has_two_distinct_real_roots 4 (-4) 1 ∧
  ¬has_two_distinct_real_roots 1 (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_d_has_two_distinct_roots_l1796_179664


namespace NUMINAMATH_CALUDE_absolute_value_sum_l1796_179653

theorem absolute_value_sum (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : |a - b| = 3) (h5 : |b - c| = 4) (h6 : |c - d| = 5) :
  |a - d| = 12 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l1796_179653


namespace NUMINAMATH_CALUDE_prism_volume_l1796_179675

-- Define a right rectangular prism
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Define the volume of a rectangular prism
def volume (p : RectangularPrism) : ℝ := p.length * p.width * p.height

-- Define the areas of the faces
def faceArea1 (p : RectangularPrism) : ℝ := p.length * p.width
def faceArea2 (p : RectangularPrism) : ℝ := p.width * p.height
def faceArea3 (p : RectangularPrism) : ℝ := p.length * p.height

-- State the theorem
theorem prism_volume (p : RectangularPrism)
  (h1 : faceArea1 p = 60)
  (h2 : faceArea2 p = 72)
  (h3 : faceArea3 p = 90) :
  volume p = 4320 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1796_179675


namespace NUMINAMATH_CALUDE_man_swimming_speed_l1796_179607

/-- The swimming speed of a man in still water, given that it takes him twice as long to swim upstream
    than downstream in a stream with a speed of 1 km/h. -/
def swimming_speed : ℝ := 2

/-- The speed of the stream in km/h. -/
def stream_speed : ℝ := 1

/-- The time ratio of swimming upstream to downstream. -/
def upstream_downstream_ratio : ℝ := 2

theorem man_swimming_speed :
  swimming_speed = 2 ∧
  stream_speed = 1 ∧
  upstream_downstream_ratio = 2 →
  swimming_speed + stream_speed = upstream_downstream_ratio * (swimming_speed - stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_man_swimming_speed_l1796_179607


namespace NUMINAMATH_CALUDE_min_value_expression_l1796_179693

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (heq : a * b = 1 / 2) :
  (4 * a^2 + b^2 + 3) / (2 * a - b) ≥ 2 * Real.sqrt 5 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ > b₀ ∧ a₀ * b₀ = 1 / 2 ∧
    (4 * a₀^2 + b₀^2 + 3) / (2 * a₀ - b₀) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1796_179693


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1796_179628

theorem chess_tournament_games (n : ℕ) (h : n = 18) : 
  (n * (n - 1)) / 2 = 153 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1796_179628


namespace NUMINAMATH_CALUDE_triangle_properties_l1796_179624

open Real

structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)
  (h5 : a > 0 ∧ b > 0 ∧ c > 0)
  (h6 : a * cos C + Real.sqrt 3 * a * sin C - b - c = 0)
  (h7 : b^2 + c^2 = 2 * a^2)

def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem triangle_properties (t : Triangle) :
  t.A = π / 3 ∧
  isEquilateral t ∧
  ∃ (D : ℝ × ℝ), 
    let B := (0, 0)
    let C := (t.c, 0)
    let A := (t.b * cos t.C, t.b * sin t.C)
    let AC := (A.1 - C.1, A.2 - C.2)
    let AD := (D.1 - A.1, D.2 - A.2)
    2 * (D.1 - B.1, D.2 - B.2) = (C.1 - D.1, C.2 - D.2) ∧
    (AD.1 * AC.1 + AD.2 * AC.2) / Real.sqrt (AC.1^2 + AC.2^2) = 2/3 * Real.sqrt (AC.1^2 + AC.2^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1796_179624


namespace NUMINAMATH_CALUDE_sqrt_2_4_3_6_5_2_l1796_179678

theorem sqrt_2_4_3_6_5_2 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_4_3_6_5_2_l1796_179678


namespace NUMINAMATH_CALUDE_quadratic_no_solution_l1796_179601

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_no_solution (a b c : ℝ) (h_a : a ≠ 0) :
  f a b c 0 = 2 →
  f a b c 1 = 1 →
  f a b c 2 = 2 →
  f a b c 3 = 5 →
  f a b c 4 = 10 →
  ∀ x, f a b c x ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_solution_l1796_179601


namespace NUMINAMATH_CALUDE_project_completion_time_l1796_179671

/-- Represents the time to complete a project -/
structure ProjectTime where
  days : ℝ
  inv_days : ℝ
  inv_days_eq : inv_days = 1 / days

/-- Represents the work done on a project -/
def work_done (time : ProjectTime) (days_worked : ℝ) : ℝ :=
  days_worked * time.inv_days

/-- The theorem states that given the conditions of the problem, 
    the project will be completed in 15 days -/
theorem project_completion_time 
  (a_time b_time : ProjectTime)
  (a_quits_before : ℝ)
  (h1 : a_time.days = 20)
  (h2 : b_time.days = 30)
  (h3 : a_quits_before = 5) :
  ∃ (total_days : ℝ), 
    total_days = 15 ∧ 
    work_done a_time (total_days - a_quits_before) + 
    work_done b_time total_days = 1 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l1796_179671


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1796_179673

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 2 + a 8 = 12) : 
  a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1796_179673


namespace NUMINAMATH_CALUDE_ben_pea_picking_time_l1796_179642

/-- Given Ben's rate of picking sugar snap peas, calculate the time needed to pick a different amount -/
theorem ben_pea_picking_time (initial_peas initial_time target_peas : ℕ) : 
  initial_peas > 0 → initial_time > 0 → target_peas > 0 →
  (target_peas * initial_time) / initial_peas = 9 :=
by
  sorry

#check ben_pea_picking_time 56 7 72

end NUMINAMATH_CALUDE_ben_pea_picking_time_l1796_179642


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l1796_179613

/-- The area of a stripe wrapping around a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h1 : diameter = 20) 
  (h2 : stripe_width = 4) 
  (h3 : revolutions = 3) : 
  stripe_width * revolutions * (π * diameter) = 240 * π := by
sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l1796_179613


namespace NUMINAMATH_CALUDE_total_turnips_l1796_179676

theorem total_turnips (melanie_turnips benny_turnips : ℕ) 
  (h1 : melanie_turnips = 139) 
  (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l1796_179676


namespace NUMINAMATH_CALUDE_exists_max_volume_l1796_179609

/-- A rectangular prism with specific diagonal lengths --/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  h_space_diagonal : a^2 + b^2 + c^2 = 1
  h_face_diagonal : b^2 + c^2 = 2
  h_positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The volume of a rectangular prism --/
def volume (prism : RectangularPrism) : ℝ :=
  prism.a * prism.b * prism.c

/-- There exists a value p that maximizes the volume of the rectangular prism --/
theorem exists_max_volume : 
  ∃ p : ℝ, p > 0 ∧ 
  ∃ prism : RectangularPrism, 
    prism.a = p ∧
    ∀ other : RectangularPrism, volume prism ≥ volume other := by
  sorry


end NUMINAMATH_CALUDE_exists_max_volume_l1796_179609


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l1796_179686

/-- The line passing through points (2, 6) and (4, 10) intersects the x-axis at (-1, 0) -/
theorem line_intersection_x_axis :
  let p1 : ℝ × ℝ := (2, 6)
  let p2 : ℝ × ℝ := (4, 10)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let line (x : ℝ) : ℝ := m * x + b
  ∃ x : ℝ, line x = 0 ∧ x = -1 :=
sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l1796_179686


namespace NUMINAMATH_CALUDE_sequence_is_cubic_polynomial_l1796_179630

def fourth_difference (u : ℕ → ℝ) : ℕ → ℝ :=
  λ n => u (n + 4) - 4 * u (n + 3) + 6 * u (n + 2) - 4 * u (n + 1) + u n

theorem sequence_is_cubic_polynomial 
  (u : ℕ → ℝ) 
  (h : ∀ n, fourth_difference u n = 0) : 
  ∃ a b c d : ℝ, ∀ n, u n = a * n^3 + b * n^2 + c * n + d :=
sorry

end NUMINAMATH_CALUDE_sequence_is_cubic_polynomial_l1796_179630


namespace NUMINAMATH_CALUDE_smallest_ratio_of_equation_l1796_179660

theorem smallest_ratio_of_equation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : 18 * x - 4 * x^2 + 2 * x^3 - 9 * y - 10 * x * y - x^2 * y + 6 * y^2 + 2 * x * y^2 - y^3 = 0) :
  ∃ (k : ℝ), k = y / x ∧ k ≥ 4/3 ∧ (∀ (k' : ℝ), k' = y / x → k' ≥ k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_ratio_of_equation_l1796_179660


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l1796_179645

/-- Given two congruent squares ABCD and EFGH with side length 20 units that overlap
    to form a 20 by 35 rectangle AEGD, prove that 14% of AEGD's area is shaded. -/
theorem shaded_area_percentage (side_length : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  side_length = 20 →
  rectangle_width = 20 →
  rectangle_length = 35 →
  (((2 * side_length - rectangle_length) * side_length) / (rectangle_width * rectangle_length)) * 100 = 14 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l1796_179645


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1796_179625

/-- Given a line L1 with equation x - 2y + m = 0 and a point P (-1, 3),
    this theorem states that the line L2 with equation 2x + y - 1 = 0
    passes through P and is perpendicular to L1. -/
theorem perpendicular_line_through_point (m : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ x - 2*y + m = 0
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 2*x + y - 1 = 0
  let P : ℝ × ℝ := (-1, 3)
  (L2 P.1 P.2) ∧                   -- L2 passes through P
  (∀ x1 y1 x2 y2, L1 x1 y1 → L2 x2 y2 →
    (x2 - x1) * (x1 - 2*y1) + (y2 - y1) * (-2*x1 - y1) = 0) -- L2 is perpendicular to L1
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1796_179625


namespace NUMINAMATH_CALUDE_roof_area_theorem_l1796_179654

/-- Represents the dimensions and area of a rectangular roof. -/
structure RectangularRoof where
  width : ℚ
  length : ℚ
  area : ℚ

/-- Calculates the area of a rectangular roof given its width and length. -/
def calculateArea (w : ℚ) (l : ℚ) : ℚ := w * l

/-- Theorem: The area of a rectangular roof with specific proportions is 455 1/9 square feet. -/
theorem roof_area_theorem (roof : RectangularRoof) : 
  roof.length = 3 * roof.width → 
  roof.length - roof.width = 32 → 
  roof.area = calculateArea roof.width roof.length → 
  roof.area = 455 + 1/9 := by
  sorry

end NUMINAMATH_CALUDE_roof_area_theorem_l1796_179654


namespace NUMINAMATH_CALUDE_birds_theorem_l1796_179616

def birds_problem (grey_birds : ℕ) (white_birds : ℕ) : Prop :=
  white_birds = grey_birds + 6 ∧
  grey_birds = 40 ∧
  (grey_birds / 2 + white_birds = 66)

theorem birds_theorem :
  ∃ (grey_birds white_birds : ℕ), birds_problem grey_birds white_birds :=
sorry

end NUMINAMATH_CALUDE_birds_theorem_l1796_179616


namespace NUMINAMATH_CALUDE_milton_more_accelerated_l1796_179677

/-- Represents the percentage of at-home workforce for a city at a given year --/
structure WorkforceData :=
  (year2000 : ℝ)
  (year2010 : ℝ)
  (year2020 : ℝ)
  (year2030 : ℝ)

/-- Determines if a city's workforce growth is accelerating --/
def isAccelerating (data : WorkforceData) : Prop :=
  let diff2010 := data.year2010 - data.year2000
  let diff2020 := data.year2020 - data.year2010
  let diff2030 := data.year2030 - data.year2020
  diff2030 > diff2020 ∧ diff2020 > diff2010

/-- Milton City's workforce data --/
def miltonCity : WorkforceData :=
  { year2000 := 3
  , year2010 := 9
  , year2020 := 18
  , year2030 := 35 }

/-- Rivertown's workforce data --/
def rivertown : WorkforceData :=
  { year2000 := 4
  , year2010 := 7
  , year2020 := 13
  , year2030 := 20 }

/-- Theorem stating that Milton City's growth is more accelerated than Rivertown's --/
theorem milton_more_accelerated :
  isAccelerating miltonCity ∧ ¬isAccelerating rivertown :=
sorry

end NUMINAMATH_CALUDE_milton_more_accelerated_l1796_179677


namespace NUMINAMATH_CALUDE_solution_implies_sum_l1796_179621

/-- The function f(x) = |x+1| + |x-3| -/
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

/-- The function g(x) = a - |x-2| -/
def g (a x : ℝ) : ℝ := a - |x - 2|

/-- The theorem stating that if the solution set of f(x) < g(x) is (b, 7/2), then a + b = 6 -/
theorem solution_implies_sum (a b : ℝ) :
  (∀ x, f x < g a x ↔ b < x ∧ x < 7/2) →
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_sum_l1796_179621


namespace NUMINAMATH_CALUDE_complex_z_and_magnitude_l1796_179614

def complex_number (a b : ℝ) : ℂ := Complex.mk a b

theorem complex_z_and_magnitude : 
  let i : ℂ := complex_number 0 1
  let z : ℂ := (1 - i) / (1 + i) + 2*i
  (z = i) ∧ (Complex.abs z = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_z_and_magnitude_l1796_179614


namespace NUMINAMATH_CALUDE_integral_x_cubed_plus_one_l1796_179631

theorem integral_x_cubed_plus_one : ∫ x in (-2)..2, (x^3 + 1) = 4 := by sorry

end NUMINAMATH_CALUDE_integral_x_cubed_plus_one_l1796_179631


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1796_179635

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- State the theorem
theorem tangent_line_equation :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = f x → (a * x + b * y + c = 0 ↔ (x, y) = point ∨ 
      ∃ h > 0, ∀ t : ℝ, 0 < |t| → |t| < h → 
        (a * (point.1 + t) + b * f (point.1 + t) + c) * (a * point.1 + b * point.2 + c) > 0)) ∧
    a = 2 ∧ b = -1 ∧ c = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1796_179635


namespace NUMINAMATH_CALUDE_jane_change_l1796_179684

-- Define the cost of the apple
def apple_cost : ℚ := 75/100

-- Define the amount Jane pays
def amount_paid : ℚ := 5

-- Define the change function
def change (cost paid : ℚ) : ℚ := paid - cost

-- Theorem statement
theorem jane_change : change apple_cost amount_paid = 425/100 := by
  sorry

end NUMINAMATH_CALUDE_jane_change_l1796_179684


namespace NUMINAMATH_CALUDE_new_students_average_age_l1796_179634

theorem new_students_average_age
  (original_strength : ℕ)
  (original_average_age : ℝ)
  (new_students : ℕ)
  (new_average_age : ℝ) :
  original_strength = 10 →
  original_average_age = 40 →
  new_students = 10 →
  new_average_age = 36 →
  let total_original_age := original_strength * original_average_age
  let total_new_age := (original_strength + new_students) * new_average_age
  let new_students_total_age := total_new_age - total_original_age
  new_students_total_age / new_students = 32 := by
sorry

end NUMINAMATH_CALUDE_new_students_average_age_l1796_179634


namespace NUMINAMATH_CALUDE_arc_length_of_inscribed_pentagon_l1796_179615

-- Define the circle radius
def circle_radius : ℝ := 5

-- Define the number of sides in a regular pentagon
def pentagon_sides : ℕ := 5

-- Theorem statement
theorem arc_length_of_inscribed_pentagon (π : ℝ) :
  let circumference := 2 * π * circle_radius
  let arc_length := circumference / pentagon_sides
  arc_length = 2 * π := by sorry

end NUMINAMATH_CALUDE_arc_length_of_inscribed_pentagon_l1796_179615


namespace NUMINAMATH_CALUDE_smallest_n_divisible_l1796_179603

theorem smallest_n_divisible : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m^2 % 24 = 0 ∧ m^3 % 450 = 0 → n ≤ m) ∧
  n^2 % 24 = 0 ∧ n^3 % 450 = 0 := by
  use 60
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_l1796_179603


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1796_179670

theorem average_of_remaining_numbers
  (total : ℝ)
  (avg_all : ℝ)
  (avg_first_two : ℝ)
  (avg_second_two : ℝ)
  (h1 : total = 6)
  (h2 : avg_all = 2.80)
  (h3 : avg_first_two = 2.4)
  (h4 : avg_second_two = 2.3) :
  (total * avg_all - 2 * avg_first_two - 2 * avg_second_two) / 2 = 3.7 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1796_179670


namespace NUMINAMATH_CALUDE_conference_games_count_l1796_179695

/-- The number of teams in the conference -/
def total_teams : ℕ := 16

/-- The number of divisions in the conference -/
def num_divisions : ℕ := 2

/-- The number of teams in each division -/
def teams_per_division : ℕ := 8

/-- The number of times a team plays each team in its own division -/
def games_within_division : ℕ := 2

/-- The number of times a team plays each team in the other division -/
def games_between_divisions : ℕ := 1

/-- The total number of games in a complete season for the conference -/
def total_games : ℕ := 176

theorem conference_games_count :
  total_games = (total_teams * (
    (teams_per_division - 1) * games_within_division +
    teams_per_division * games_between_divisions
  )) / 2 :=
sorry

end NUMINAMATH_CALUDE_conference_games_count_l1796_179695


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l1796_179649

/-- If 16x^2 + 32x + a is the square of a binomial, then a = 16 -/
theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b c : ℝ, ∀ x, 16 * x^2 + 32 * x + a = (b * x + c)^2) → a = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l1796_179649


namespace NUMINAMATH_CALUDE_frank_candy_total_l1796_179692

/-- Given that Frank put candy equally into 2 bags and there are 8 pieces of candy in each bag,
    prove that the total number of pieces of candy is 16. -/
theorem frank_candy_total (num_bags : ℕ) (pieces_per_bag : ℕ) 
    (h1 : num_bags = 2) 
    (h2 : pieces_per_bag = 8) : 
  num_bags * pieces_per_bag = 16 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_total_l1796_179692


namespace NUMINAMATH_CALUDE_tim_balloons_l1796_179696

theorem tim_balloons (dan_balloons : ℝ) (ratio : ℝ) : 
  dan_balloons = 29.0 → 
  ratio = 7.0 → 
  ⌊dan_balloons / ratio⌋ = 4 := by
sorry

end NUMINAMATH_CALUDE_tim_balloons_l1796_179696


namespace NUMINAMATH_CALUDE_total_balloons_l1796_179604

theorem total_balloons (gold : ℕ) (silver : ℕ) (black : ℕ) : 
  gold = 141 → 
  silver = 2 * gold → 
  black = 150 → 
  gold + silver + black = 573 := by
sorry

end NUMINAMATH_CALUDE_total_balloons_l1796_179604


namespace NUMINAMATH_CALUDE_solve_equation_l1796_179610

theorem solve_equation (x y : ℝ) : y = 2 / (5 * x + 2) → y = 2 → x = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1796_179610


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1796_179689

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) :
  (2 * x₁^2 + 5 * x₁ - 12 = 0) →
  (2 * x₂^2 + 5 * x₂ - 12 = 0) →
  x₁ ≠ x₂ →
  x₁^2 + x₂^2 = 73/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1796_179689


namespace NUMINAMATH_CALUDE_linear_system_solution_l1796_179648

theorem linear_system_solution (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47)
  (eq2 : 4 * a + 3 * b = 39) :
  2 * a + 2 * b = 164 / 7 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1796_179648


namespace NUMINAMATH_CALUDE_total_floor_area_square_slabs_l1796_179644

/-- Calculates the total floor area covered by square stone slabs. -/
theorem total_floor_area_square_slabs 
  (num_slabs : ℕ) 
  (slab_length : ℝ) 
  (h1 : num_slabs = 30)
  (h2 : slab_length = 200)
  : (num_slabs * (slab_length / 100)^2 : ℝ) = 120 := by
  sorry

#check total_floor_area_square_slabs

end NUMINAMATH_CALUDE_total_floor_area_square_slabs_l1796_179644


namespace NUMINAMATH_CALUDE_kates_retirement_fund_l1796_179633

/-- The initial value of Kate's retirement fund, given the current value and the decrease amount. -/
def initial_value (current_value decrease : ℕ) : ℕ := current_value + decrease

/-- Theorem stating that Kate's initial retirement fund value was $1472. -/
theorem kates_retirement_fund : initial_value 1460 12 = 1472 := by
  sorry

end NUMINAMATH_CALUDE_kates_retirement_fund_l1796_179633


namespace NUMINAMATH_CALUDE_time_to_see_again_is_48_l1796_179688

-- Define the parameters of the problem
def path_distance : ℝ := 300
def building_diameter : ℝ := 150
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def initial_distance : ℝ := 300

-- Define the function to calculate the time until they can see each other again
def time_to_see_again (pd : ℝ) (bd : ℝ) (ks : ℝ) (js : ℝ) (id : ℝ) : ℝ :=
  -- The actual calculation would go here, but we'll use sorry to skip the proof
  sorry

-- State the theorem
theorem time_to_see_again_is_48 :
  time_to_see_again path_distance building_diameter kenny_speed jenny_speed initial_distance = 48 := by
  sorry

end NUMINAMATH_CALUDE_time_to_see_again_is_48_l1796_179688


namespace NUMINAMATH_CALUDE_min_x2_coeff_and_x7_coeff_l1796_179669

-- Define the function f(x)
def f (m n : ℕ) (x : ℝ) : ℝ := (1 + x)^m + (1 + x)^n

-- Define the coefficient of x^k in the expansion of (1 + x)^n
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem min_x2_coeff_and_x7_coeff 
  (m n : ℕ) 
  (h1 : binomial_coefficient m 1 + binomial_coefficient n 1 = 19) :
  (∃ (min_coeff : ℕ), 
    (∀ (m' n' : ℕ), 
      binomial_coefficient m' 1 + binomial_coefficient n' 1 = 19 →
      binomial_coefficient m' 2 + binomial_coefficient n' 2 ≥ min_coeff) ∧
    min_coeff = 81) ∧
  (∃ (m' n' : ℕ), 
    binomial_coefficient m' 1 + binomial_coefficient n' 1 = 19 ∧
    binomial_coefficient m' 2 + binomial_coefficient n' 2 = 81 ∧
    binomial_coefficient m' 7 + binomial_coefficient n' 7 = 156) :=
by sorry

end NUMINAMATH_CALUDE_min_x2_coeff_and_x7_coeff_l1796_179669


namespace NUMINAMATH_CALUDE_alyssa_cherries_cost_l1796_179641

/-- The amount Alyssa paid for cherries -/
def cherries_cost (total_spent grapes_cost : ℚ) : ℚ :=
  total_spent - grapes_cost

/-- Proof that Alyssa paid $9.85 for cherries -/
theorem alyssa_cherries_cost :
  let total_spent : ℚ := 21.93
  let grapes_cost : ℚ := 12.08
  cherries_cost total_spent grapes_cost = 9.85 := by
  sorry

#eval cherries_cost 21.93 12.08

end NUMINAMATH_CALUDE_alyssa_cherries_cost_l1796_179641


namespace NUMINAMATH_CALUDE_least_number_divisible_by_11_with_remainder_2_l1796_179647

def is_divisible_by_11 (n : ℕ) : Prop := ∃ k : ℕ, n = 11 * k

def leaves_remainder_2 (n : ℕ) (d : ℕ) : Prop := ∃ k : ℕ, n = d * k + 2

theorem least_number_divisible_by_11_with_remainder_2 : 
  (is_divisible_by_11 1262) ∧ 
  (∀ d : ℕ, 3 ≤ d → d ≤ 7 → leaves_remainder_2 1262 d) ∧
  (∀ m : ℕ, m < 1262 → 
    ¬(is_divisible_by_11 m ∧ 
      (∀ d : ℕ, 3 ≤ d → d ≤ 7 → leaves_remainder_2 m d))) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_11_with_remainder_2_l1796_179647


namespace NUMINAMATH_CALUDE_greatest_area_difference_l1796_179617

/-- A rectangle with integer dimensions and perimeter 200 cm -/
structure Rectangle where
  width : ℕ
  height : ℕ
  perimeter_eq : width * 2 + height * 2 = 200

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- A rectangle with one side of length 80 cm -/
structure DoorRectangle extends Rectangle where
  door_side : width = 80 ∨ height = 80

theorem greatest_area_difference :
  ∃ (r : Rectangle) (d : DoorRectangle),
    ∀ (r' : Rectangle) (d' : DoorRectangle),
      d.area - r.area ≥ d'.area - r'.area ∧
      d.area - r.area = 2300 := by
  sorry

end NUMINAMATH_CALUDE_greatest_area_difference_l1796_179617


namespace NUMINAMATH_CALUDE_fraction_of_larger_part_l1796_179661

theorem fraction_of_larger_part (total : ℝ) (larger : ℝ) (f : ℝ) : 
  total = 66 →
  larger = 50 →
  f * larger = 0.625 * (total - larger) + 10 →
  f = 0.4 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_larger_part_l1796_179661


namespace NUMINAMATH_CALUDE_red_box_position_l1796_179652

/-- Given a collection of boxes with a red box among them, this function
    calculates the position of the red box from the right when arranged
    from largest to smallest, given its position from the right when
    arranged from smallest to largest. -/
def position_from_right_largest_to_smallest (total_boxes : ℕ) (position_smallest_to_largest : ℕ) : ℕ :=
  total_boxes - (position_smallest_to_largest - 1)

/-- Theorem stating that for 45 boxes with the red box 29th from the right
    when arranged smallest to largest, it will be 17th from the right
    when arranged largest to smallest. -/
theorem red_box_position (total_boxes : ℕ) (position_smallest_to_largest : ℕ) 
    (h1 : total_boxes = 45)
    (h2 : position_smallest_to_largest = 29) :
    position_from_right_largest_to_smallest total_boxes position_smallest_to_largest = 17 := by
  sorry

#eval position_from_right_largest_to_smallest 45 29

end NUMINAMATH_CALUDE_red_box_position_l1796_179652


namespace NUMINAMATH_CALUDE_three_fourths_of_four_fifths_of_two_thirds_l1796_179629

theorem three_fourths_of_four_fifths_of_two_thirds : (3 : ℚ) / 4 * (4 : ℚ) / 5 * (2 : ℚ) / 3 = (2 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_of_four_fifths_of_two_thirds_l1796_179629


namespace NUMINAMATH_CALUDE_square_gt_of_abs_lt_l1796_179650

theorem square_gt_of_abs_lt (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_gt_of_abs_lt_l1796_179650


namespace NUMINAMATH_CALUDE_carolyn_embroiders_50_flowers_l1796_179687

/-- Represents the embroidery problem with given conditions -/
structure EmbroideryProblem where
  stitches_per_minute : ℕ
  stitches_per_flower : ℕ
  stitches_per_unicorn : ℕ
  stitches_for_godzilla : ℕ
  num_unicorns : ℕ
  total_minutes : ℕ

/-- Calculates the number of flowers Carolyn wants to embroider -/
def flowers_to_embroider (p : EmbroideryProblem) : ℕ :=
  let total_stitches := p.stitches_per_minute * p.total_minutes
  let stitches_for_creatures := p.stitches_for_godzilla + p.num_unicorns * p.stitches_per_unicorn
  let remaining_stitches := total_stitches - stitches_for_creatures
  remaining_stitches / p.stitches_per_flower

/-- Theorem stating that given the problem conditions, Carolyn wants to embroider 50 flowers -/
theorem carolyn_embroiders_50_flowers :
  let p := EmbroideryProblem.mk 4 60 180 800 3 1085
  flowers_to_embroider p = 50 := by
  sorry


end NUMINAMATH_CALUDE_carolyn_embroiders_50_flowers_l1796_179687


namespace NUMINAMATH_CALUDE_solution_set_properties_l1796_179698

def M (k : ℝ) : Set ℝ :=
  {x : ℝ | (k^2 + 2*k - 3)*x^2 + (k + 3)*x - 1 > 0}

theorem solution_set_properties (k : ℝ) :
  (M k = ∅ → k ∈ Set.Icc (-3 : ℝ) (1/5)) ∧
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ M k = Set.Ioo a b → k ∈ Set.Ioo (1/5 : ℝ) 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_properties_l1796_179698


namespace NUMINAMATH_CALUDE_perfume_fundraising_l1796_179622

/-- The amount of additional money needed to buy a perfume --/
def additional_money_needed (perfume_cost initial_christian initial_sue yards_mowed yard_price dogs_walked dog_price : ℚ) : ℚ :=
  perfume_cost - (initial_christian + initial_sue + yards_mowed * yard_price + dogs_walked * dog_price)

/-- Theorem stating the additional money needed is $6.00 --/
theorem perfume_fundraising :
  additional_money_needed 50 5 7 4 5 6 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_perfume_fundraising_l1796_179622


namespace NUMINAMATH_CALUDE_exists_special_function_l1796_179682

theorem exists_special_function : ∃ (f : ℝ → ℝ),
  (∀ (b : ℝ), ∃! (x : ℝ), f x = b) ∧
  (∀ (a b : ℝ), a > 0 → ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a * x₁ + b ∧ f x₂ = a * x₂ + b) :=
by sorry

end NUMINAMATH_CALUDE_exists_special_function_l1796_179682


namespace NUMINAMATH_CALUDE_a_range_l1796_179680

def A : Set ℝ := {x | x > 1}
def B (a : ℝ) : Set ℝ := {a + 2}

theorem a_range (a : ℝ) : A ∩ B a = ∅ → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l1796_179680


namespace NUMINAMATH_CALUDE_radius_scientific_notation_l1796_179623

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- The given radius in centimeters -/
def radius : ℝ := 0.000012

/-- The scientific notation representation of the radius -/
def radiusScientific : ScientificNotation :=
  { coefficient := 1.2
    exponent := -5
    h1 := by sorry
    h2 := by sorry }

theorem radius_scientific_notation :
  radius = radiusScientific.coefficient * (10 : ℝ) ^ radiusScientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_radius_scientific_notation_l1796_179623


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1796_179640

theorem quadratic_equation_solution (a : ℝ) : 
  (1 : ℝ)^2 + a*(1 : ℝ) + 1 = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1796_179640


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1796_179690

/-- Given a geometric sequence of positive terms {a_n}, prove that if the sum of logarithms of certain terms equals 6, then the product of the first and fifteenth terms is 10000. -/
theorem geometric_sequence_property (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a n > 0) →  -- Sequence of positive terms
  (∀ n, a (n + 1) = a n * r) →  -- Geometric sequence property
  Real.log (a 3) + Real.log (a 8) + Real.log (a 13) = 6 →
  a 1 * a 15 = 10000 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1796_179690


namespace NUMINAMATH_CALUDE_admission_ways_l1796_179651

theorem admission_ways (n : Nat) (k : Nat) (s : Nat) : 
  n = 23 → k = 2 → s = 3 → 
  (Nat.choose s 1) * (Nat.choose k k) * (Nat.choose n k) = 1518 := by
  sorry

end NUMINAMATH_CALUDE_admission_ways_l1796_179651


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1796_179606

theorem exponent_multiplication (x : ℝ) : x^8 * x^2 = x^10 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1796_179606


namespace NUMINAMATH_CALUDE_oscars_voting_problem_l1796_179667

/-- Represents a film critic's vote --/
structure Vote where
  actor : Nat
  actress : Nat

/-- The problem statement --/
theorem oscars_voting_problem 
  (critics : Finset Vote) 
  (h_count : critics.card = 3366)
  (h_unique : ∀ n : Nat, 1 ≤ n ∧ n ≤ 100 → ∃ v : Vote, (critics.filter (λ x => x.actor = v.actor ∨ x.actress = v.actress)).card = n) :
  ∃ v1 v2 : Vote, v1 ∈ critics ∧ v2 ∈ critics ∧ v1 ≠ v2 ∧ v1.actor = v2.actor ∧ v1.actress = v2.actress :=
sorry

end NUMINAMATH_CALUDE_oscars_voting_problem_l1796_179667


namespace NUMINAMATH_CALUDE_stream_speed_l1796_179637

/-- Proves that the speed of a stream is 3 kmph given the conditions of boat travel -/
theorem stream_speed (boat_speed : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : downstream_time = 1)
  (h3 : upstream_time = 1.5) :
  ∃ stream_speed : ℝ, 
    stream_speed = 3 ∧ 
    (boat_speed + stream_speed) * downstream_time = (boat_speed - stream_speed) * upstream_time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l1796_179637


namespace NUMINAMATH_CALUDE_production_theorem_l1796_179618

-- Define production lines
structure ProductionLine where
  process_rate : ℝ → ℝ
  inv_process_rate : ℝ → ℝ

-- Define the company
structure Company where
  line_A : ProductionLine
  line_B : ProductionLine

-- Define the problem
def production_problem (c : Company) : Prop :=
  -- Line A processes a tons in (4a+1) hours
  (c.line_A.process_rate = fun a => 4 * a + 1) ∧
  (c.line_A.inv_process_rate = fun t => (t - 1) / 4) ∧
  -- Line B processes b tons in (2b+3) hours
  (c.line_B.process_rate = fun b => 2 * b + 3) ∧
  (c.line_B.inv_process_rate = fun t => (t - 3) / 2) ∧
  -- Day 1: 5 tons allocated with equal processing time
  ∃ (x : ℝ), 0 < x ∧ x < 5 ∧ c.line_A.process_rate x = c.line_B.process_rate (5 - x) ∧
  -- Day 2: 5 tons allocated based on day 1 results, plus m to A and n to B
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧
    c.line_A.process_rate (x + m) = c.line_B.process_rate (5 - x + n) ∧
    c.line_A.process_rate (x + m) ≤ 24 ∧ c.line_B.process_rate (5 - x + n) ≤ 24

-- Theorem to prove
theorem production_theorem (c : Company) :
  production_problem c →
  (∃ (x : ℝ), x = 2 ∧ 5 - x = 3) ∧
  (∃ (m n : ℝ), m / n = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_production_theorem_l1796_179618


namespace NUMINAMATH_CALUDE_impossible_61_cents_l1796_179657

/-- Represents the types of coins available in the piggy bank -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents -/
def coin_value (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a combination of coins -/
def CoinCombination := List Coin

/-- Calculates the total value of a coin combination in cents -/
def total_value (comb : CoinCombination) : Nat :=
  comb.map coin_value |>.sum

/-- Theorem: It's impossible to make 61 cents with exactly 6 coins -/
theorem impossible_61_cents :
  ¬∃ (comb : CoinCombination), comb.length = 6 ∧ total_value comb = 61 := by
  sorry


end NUMINAMATH_CALUDE_impossible_61_cents_l1796_179657


namespace NUMINAMATH_CALUDE_sqrt_ratio_equation_l1796_179679

theorem sqrt_ratio_equation (x : ℝ) :
  (Real.sqrt (2 * x + 7) / Real.sqrt (4 * x + 7) = Real.sqrt 7 / 2) →
  x = -21 / 20 := by
sorry

end NUMINAMATH_CALUDE_sqrt_ratio_equation_l1796_179679


namespace NUMINAMATH_CALUDE_mitch_earnings_l1796_179655

/-- Represents Mitch's work schedule and earnings --/
structure MitchSchedule where
  weekday_hours : ℕ
  weekend_hours : ℕ
  weekday_rate : ℕ
  weekend_rate : ℕ

/-- Calculates Mitch's weekly earnings --/
def weekly_earnings (schedule : MitchSchedule) : ℕ :=
  (schedule.weekday_hours * 5 * schedule.weekday_rate) +
  (schedule.weekend_hours * 2 * schedule.weekend_rate)

/-- Theorem stating Mitch's weekly earnings --/
theorem mitch_earnings : 
  let schedule := MitchSchedule.mk 5 3 3 6
  weekly_earnings schedule = 111 := by
  sorry


end NUMINAMATH_CALUDE_mitch_earnings_l1796_179655


namespace NUMINAMATH_CALUDE_triangle_area_l1796_179683

theorem triangle_area (a b c : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 5) (h3 : c = 2) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = Real.sqrt 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1796_179683


namespace NUMINAMATH_CALUDE_lunch_group_probability_l1796_179620

theorem lunch_group_probability (total_students : ℕ) (num_groups : ℕ) (friends : ℕ) 
  (h1 : total_students = 800)
  (h2 : num_groups = 4)
  (h3 : friends = 4)
  (h4 : total_students % num_groups = 0) :
  (1 : ℚ) / (num_groups ^ (friends - 1)) = 1 / 64 :=
sorry

end NUMINAMATH_CALUDE_lunch_group_probability_l1796_179620


namespace NUMINAMATH_CALUDE_complex_arithmetic_equation_l1796_179646

theorem complex_arithmetic_equation : 
  (8 * 2.25 - 5 * 0.85) / 2.5 + (3/5 * 1.5 - 7/8 * 0.35) / 1.25 = 5.975 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equation_l1796_179646


namespace NUMINAMATH_CALUDE_jane_initial_pick_is_one_fourth_l1796_179659

/-- The fraction of tomatoes Jane initially picked from a tomato plant -/
def jane_initial_pick : ℚ :=
  let initial_tomatoes : ℕ := 100
  let second_pick : ℕ := 20
  let third_pick : ℕ := 2 * second_pick
  let remaining_tomatoes : ℕ := 15
  (initial_tomatoes - second_pick - third_pick - remaining_tomatoes) / initial_tomatoes

theorem jane_initial_pick_is_one_fourth :
  jane_initial_pick = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_jane_initial_pick_is_one_fourth_l1796_179659


namespace NUMINAMATH_CALUDE_room_width_l1796_179699

/-- Given a rectangular room with length 21 m, surrounded by a 2 m wide veranda on all sides,
    and the veranda area is 148 m², prove that the width of the room is 12 m. -/
theorem room_width (room_length : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_length = 21 →
  veranda_width = 2 →
  veranda_area = 148 →
  ∃ (room_width : ℝ),
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) -
    room_length * room_width = veranda_area ∧
    room_width = 12 := by
  sorry

end NUMINAMATH_CALUDE_room_width_l1796_179699


namespace NUMINAMATH_CALUDE_hyperbola_center_is_correct_l1796_179619

/-- The center of a hyperbola given by the equation 9x^2 - 54x - 36y^2 + 288y - 576 = 0 -/
def hyperbola_center : ℝ × ℝ := (3, 4)

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0

/-- Theorem stating that the center of the hyperbola is (3, 4) -/
theorem hyperbola_center_is_correct :
  let (h₁, h₂) := hyperbola_center
  ∀ (ε : ℝ), ε ≠ 0 →
    ∃ (δ : ℝ), δ > 0 ∧
      ∀ (x y : ℝ),
        hyperbola_equation x y →
        (x - h₁)^2 + (y - h₂)^2 < δ^2 →
        (x - h₁)^2 + (y - h₂)^2 < ε^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_correct_l1796_179619
