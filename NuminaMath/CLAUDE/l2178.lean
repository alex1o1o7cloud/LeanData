import Mathlib

namespace absolute_value_inequality_l2178_217872

theorem absolute_value_inequality (x : ℝ) : 
  |x - 2| + |x + 1| < 4 ↔ x ∈ Set.Ioo (-7/2) (-1) ∪ Set.Ico (-1) (5/2) := by
  sorry

end absolute_value_inequality_l2178_217872


namespace phone_number_probability_l2178_217836

/-- The set of possible area codes -/
def areaCodes : Finset ℕ := {407, 410, 415}

/-- The set of digits for the remaining part of the phone number -/
def remainingDigits : Finset ℕ := {0, 1, 2, 3, 4}

/-- The total number of digits in the phone number -/
def totalDigits : ℕ := 8

/-- The number of digits after the area code -/
def remainingDigitsCount : ℕ := 5

theorem phone_number_probability :
  (1 : ℚ) / (areaCodes.card * Nat.factorial remainingDigitsCount) =
  (1 : ℚ) / 360 := by sorry

end phone_number_probability_l2178_217836


namespace space_shuttle_speed_conversion_l2178_217851

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- Proves that 12 kilometers per second is equal to 43,200 kilometers per hour -/
theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 12 = 43200 := by
  sorry

end space_shuttle_speed_conversion_l2178_217851


namespace derivative_at_three_l2178_217803

/-- Given a function f(x) = -x^2 + 10, prove that its derivative at x = 3 is -3. -/
theorem derivative_at_three (f : ℝ → ℝ) (h : ∀ x, f x = -x^2 + 10) :
  deriv f 3 = -3 := by
  sorry

end derivative_at_three_l2178_217803


namespace total_blood_cells_l2178_217891

/-- The total number of blood cells in two samples is 7,341, given that the first sample contains 4,221 blood cells and the second sample contains 3,120 blood cells. -/
theorem total_blood_cells (sample1 : Nat) (sample2 : Nat)
  (h1 : sample1 = 4221)
  (h2 : sample2 = 3120) :
  sample1 + sample2 = 7341 := by
  sorry

end total_blood_cells_l2178_217891


namespace cubic_inequality_and_fraction_inequality_l2178_217897

theorem cubic_inequality_and_fraction_inequality 
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x^3 + y^3 ≥ x^2*y + x*y^2) ∧ 
  ((x/(y*z) + y/(z*x) + z/(x*y)) ≥ (1/x + 1/y + 1/z)) := by
  sorry

end cubic_inequality_and_fraction_inequality_l2178_217897


namespace tuna_salmon_ratio_l2178_217838

/-- Proves that the ratio of tuna weight to salmon weight is 2:1 given specific conditions --/
theorem tuna_salmon_ratio (trout_weight salmon_weight tuna_weight : ℝ) : 
  trout_weight = 200 →
  salmon_weight = trout_weight * 1.5 →
  trout_weight + salmon_weight + tuna_weight = 1100 →
  tuna_weight / salmon_weight = 2 := by
  sorry

end tuna_salmon_ratio_l2178_217838


namespace ferris_wheel_capacity_l2178_217816

/-- The number of large seats on the Ferris wheel -/
def num_large_seats : ℕ := 7

/-- The total number of people that can be accommodated on large seats -/
def total_people_large_seats : ℕ := 84

/-- The number of people each large seat can hold -/
def people_per_large_seat : ℕ := total_people_large_seats / num_large_seats

theorem ferris_wheel_capacity : people_per_large_seat = 12 := by
  sorry

end ferris_wheel_capacity_l2178_217816


namespace min_value_quadratic_l2178_217847

theorem min_value_quadratic :
  ∃ (m : ℝ), (∀ x : ℝ, 4 * x^2 + 8 * x + 12 ≥ m) ∧
  (∃ x : ℝ, 4 * x^2 + 8 * x + 12 = m) ∧
  m = 8 := by
sorry

end min_value_quadratic_l2178_217847


namespace number_equality_l2178_217828

theorem number_equality (x : ℝ) : 9^6 = x^12 → x = 3 := by
  sorry

end number_equality_l2178_217828


namespace perfect_square_condition_l2178_217817

/-- A quadratic form ax^2 + bxy + cy^2 is a perfect square if and only if its discriminant b^2 - 4ac is zero. -/
def is_perfect_square (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- If 4x^2 + mxy + 25y^2 is a perfect square, then m = ±20. -/
theorem perfect_square_condition (m : ℝ) :
  is_perfect_square 4 m 25 → m = 20 ∨ m = -20 := by
  sorry

end perfect_square_condition_l2178_217817


namespace balloons_in_park_l2178_217863

/-- The number of balloons Allan and Jake have in the park -/
def total_balloons (allan_initial : ℕ) (jake : ℕ) (allan_bought : ℕ) : ℕ :=
  (allan_initial + allan_bought) + jake

/-- Theorem: Allan and Jake have 10 balloons in total -/
theorem balloons_in_park :
  total_balloons 3 5 2 = 10 := by
  sorry

end balloons_in_park_l2178_217863


namespace arithmetic_seq_property_l2178_217805

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a where a₂ + a₆ = 2, prove that a₄ = 1 -/
theorem arithmetic_seq_property (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_seq a) (h_sum : a 2 + a 6 = 2) : 
  a 4 = 1 := by
  sorry

end arithmetic_seq_property_l2178_217805


namespace square_root_equals_arithmetic_square_root_l2178_217822

theorem square_root_equals_arithmetic_square_root (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x ∧ y = Real.sqrt x) ↔ x = 0 := by sorry

end square_root_equals_arithmetic_square_root_l2178_217822


namespace least_upper_bound_inequality_inequality_holds_l2178_217892

theorem least_upper_bound_inequality (a b c : ℝ) : 
  let M : ℝ := (9 * Real.sqrt 2) / 32
  ∀ N : ℝ, (∀ x y z : ℝ, 
    |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N*(x^2 + y^2 + z^2)^2) →
  N ≥ M :=
by sorry

theorem inequality_holds (a b c : ℝ) : 
  let M : ℝ := (9 * Real.sqrt 2) / 32
  |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M*(a^2 + b^2 + c^2)^2 :=
by sorry

end least_upper_bound_inequality_inequality_holds_l2178_217892


namespace complement_of_union_l2178_217885

def U : Set Nat := {1,2,3,4,5}
def M : Set Nat := {1,3}
def N : Set Nat := {1,2}

theorem complement_of_union (U M N : Set Nat) : 
  U \ (M ∪ N) = {4,5} := by
  sorry

end complement_of_union_l2178_217885


namespace probability_of_pair_l2178_217837

/-- Represents a deck of cards with their counts -/
def Deck := List (Nat × Nat)

/-- The initial deck configuration -/
def initial_deck : Deck := List.replicate 10 (5, 5)

/-- Remove a matching pair from the deck -/
def remove_pair (d : Deck) : Deck :=
  match d with
  | (n, count) :: rest => if count ≥ 2 then (n, count - 2) :: rest else d
  | [] => []

/-- Calculate the total number of cards in the deck -/
def total_cards (d : Deck) : Nat :=
  d.foldr (fun (_, count) acc => acc + count) 0

/-- Calculate the number of ways to choose 2 cards from n cards -/
def choose_2 (n : Nat) : Nat := n * (n - 1) / 2

/-- Calculate the number of possible pairs in the deck -/
def count_pairs (d : Deck) : Nat :=
  d.foldr (fun (_, count) acc => acc + choose_2 count) 0

theorem probability_of_pair (d : Deck) :
  let remaining_deck := remove_pair initial_deck
  let total := total_cards remaining_deck
  let pairs := count_pairs remaining_deck
  (pairs : Rat) / (choose_2 total) = 31 / 376 := by sorry

end probability_of_pair_l2178_217837


namespace expected_boy_girl_adjacencies_l2178_217827

/-- The expected number of boy-girl adjacencies in a circular arrangement -/
theorem expected_boy_girl_adjacencies (n_boys n_girls : ℕ) (h : n_boys = 10 ∧ n_girls = 8) :
  let total := n_boys + n_girls
  let prob_boy_girl := (n_boys : ℚ) * n_girls / (total * (total - 1))
  total * (2 * prob_boy_girl) = 480 / 51 := by
  sorry

#check expected_boy_girl_adjacencies

end expected_boy_girl_adjacencies_l2178_217827


namespace sticker_count_l2178_217802

theorem sticker_count (stickers_per_page : ℕ) (number_of_pages : ℕ) : 
  stickers_per_page = 10 → number_of_pages = 22 → stickers_per_page * number_of_pages = 220 := by
  sorry

end sticker_count_l2178_217802


namespace parallelogram_grid_non_congruent_triangles_l2178_217875

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the parallelogram grid array -/
def ParallelogramGrid : List Point := [
  ⟨0, 0⟩,   -- Point 1
  ⟨1, 0⟩,   -- Point 2
  ⟨1.5, 0.5⟩, -- Point 3
  ⟨2.5, 0.5⟩, -- Point 4
  ⟨0.5, 0.25⟩, -- Point 5 (midpoint)
  ⟨1.75, 0.25⟩, -- Point 6 (midpoint)
  ⟨1.75, 0⟩, -- Point 7 (midpoint)
  ⟨1.25, 0.25⟩  -- Point 8 (center)
]

/-- Determines if two triangles are congruent -/
def areTrianglesCongruent (t1 t2 : List Point) : Bool :=
  sorry -- Implementation details omitted

/-- Counts the number of non-congruent triangles in the grid -/
def countNonCongruentTriangles (grid : List Point) : Nat :=
  sorry -- Implementation details omitted

/-- Theorem: The number of non-congruent triangles in the parallelogram grid is 9 -/
theorem parallelogram_grid_non_congruent_triangles :
  countNonCongruentTriangles ParallelogramGrid = 9 := by
  sorry

end parallelogram_grid_non_congruent_triangles_l2178_217875


namespace train_distance_difference_l2178_217880

/-- Represents the distance traveled by a train given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating the difference in distance traveled by two trains -/
theorem train_distance_difference 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (total_distance : ℝ) 
  (h1 : speed1 = 20) 
  (h2 : speed2 = 25) 
  (h3 : total_distance = 585) :
  ∃ (time : ℝ), 
    distance speed1 time + distance speed2 time = total_distance ∧ 
    distance speed2 time - distance speed1 time = 65 := by
  sorry

#check train_distance_difference

end train_distance_difference_l2178_217880


namespace same_angle_from_P_l2178_217849

-- Define the basic geometric objects
structure Point : Type :=
  (x : ℝ) (y : ℝ)

structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define the given conditions
def P : Point := sorry
def Q : Point := sorry
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry

def circle1 : Circle := sorry
def circle2 : Circle := sorry

-- Define the properties of the configuration
def circles_intersect (c1 c2 : Circle) (P Q : Point) : Prop := sorry
def line_perpendicular_to_PQ (A Q B : Point) : Prop := sorry
def Q_between_A_and_B (A Q B : Point) : Prop := sorry
def tangent_intersection (A B C : Point) (c1 c2 : Circle) : Prop := sorry

-- Define the angle between three points
def angle (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem same_angle_from_P :
  circles_intersect circle1 circle2 P Q →
  line_perpendicular_to_PQ A Q B →
  Q_between_A_and_B A Q B →
  tangent_intersection A B C circle1 circle2 →
  angle B P Q = angle Q P A := by sorry

end same_angle_from_P_l2178_217849


namespace quadratic_inequality_l2178_217808

theorem quadratic_inequality (x : ℝ) : 
  -10 * x^2 + 4 * x + 2 > 0 ↔ (1 - Real.sqrt 6) / 5 < x ∧ x < (1 + Real.sqrt 6) / 5 := by
  sorry

end quadratic_inequality_l2178_217808


namespace student_number_problem_l2178_217877

theorem student_number_problem (x y : ℝ) : 
  3 * x - y = 110 → x = 110 → y = 220 := by
  sorry

end student_number_problem_l2178_217877


namespace bracket_equation_solution_l2178_217884

theorem bracket_equation_solution (x : ℝ) : 
  45 - (28 - (37 - (15 - x))) = 59 → x = 20 := by
sorry

end bracket_equation_solution_l2178_217884


namespace sum_difference_is_4750_l2178_217889

/-- Rounds a number to the nearest multiple of 5, rounding 2.5s up -/
def roundToNearestFive (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

/-- Sums all integers from 1 to n -/
def sumToN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Sums all integers from 1 to n after rounding each to the nearest multiple of 5 -/
def sumRoundedToN (n : ℕ) : ℕ :=
  (n / 5) * (2 * 0 + 3 * 5)

theorem sum_difference_is_4750 :
  sumToN 100 - sumRoundedToN 100 = 4750 := by
  sorry

end sum_difference_is_4750_l2178_217889


namespace quadratic_roots_imply_right_triangle_l2178_217873

theorem quadratic_roots_imply_right_triangle 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (hroots : ∃ x : ℝ, x^2 - (a + b + c)*x + (a*b + b*c + c*a) = 0 ∧ 
    ∀ y : ℝ, y^2 - (a + b + c)*y + (a*b + b*c + c*a) = 0 → y = x) :
  ∃ p q r : ℝ, p^4 = a ∧ q^4 = b ∧ r^4 = c ∧ p^2 = q^2 + r^2 := by
  sorry

end quadratic_roots_imply_right_triangle_l2178_217873


namespace complex_fraction_evaluation_l2178_217846

theorem complex_fraction_evaluation :
  (⌈(19 : ℚ) / 7 - ⌈(35 : ℚ) / 19⌉⌉) / (⌈(35 : ℚ) / 7 + ⌈(7 * 19 : ℚ) / 35⌉⌉) = 1 / 9 := by
  sorry

end complex_fraction_evaluation_l2178_217846


namespace complex_expression_evaluation_l2178_217840

theorem complex_expression_evaluation :
  ∀ (c d : ℂ), c = 7 - 3*I → d = 2 + 5*I → 3*c - 4*d = 13 - 29*I :=
by
  sorry

end complex_expression_evaluation_l2178_217840


namespace inequality_proof_l2178_217883

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2*b^2 + b^2*c^2 + c^2*a^2 ≥ 15/16 ∧
  (a^2 + b^2 + c^2 + a^2*b^2 + b^2*c^2 + c^2*a^2 = 15/16 ↔ a = 1/2 ∧ b = 1/2 ∧ c = 1/2) := by
  sorry

end inequality_proof_l2178_217883


namespace painting_distance_l2178_217824

theorem painting_distance (wall_width painting_width : ℝ) 
  (hw : wall_width = 26) 
  (hp : painting_width = 4) : 
  (wall_width - painting_width) / 2 = 11 := by
  sorry

end painting_distance_l2178_217824


namespace g_minus_two_equals_eleven_l2178_217845

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 3*x + 1

-- State the theorem
theorem g_minus_two_equals_eleven : g (-2) = 11 := by
  sorry

end g_minus_two_equals_eleven_l2178_217845


namespace parabola_has_one_x_intercept_l2178_217899

-- Define the parabola equation
def parabola (y : ℝ) : ℝ := -3 * y^2 + 4 * y + 2

-- State the theorem
theorem parabola_has_one_x_intercept :
  ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
sorry

end parabola_has_one_x_intercept_l2178_217899


namespace gcd_of_polynomial_and_linear_l2178_217888

theorem gcd_of_polynomial_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 1573 * k) :
  Int.gcd (b^2 + 11*b + 28) (b + 6) = 2 := by sorry

end gcd_of_polynomial_and_linear_l2178_217888


namespace absolute_value_calculation_l2178_217858

theorem absolute_value_calculation : |-3| - (Real.sqrt 7 + 1)^0 - 2^2 = -2 := by
  sorry

end absolute_value_calculation_l2178_217858


namespace division_problem_l2178_217857

theorem division_problem (divisor : ℕ) : 
  (109 / divisor = 9) ∧ (109 % divisor = 1) → divisor = 12 :=
by sorry

end division_problem_l2178_217857


namespace trig_identity_l2178_217814

theorem trig_identity (α : ℝ) : 
  4.62 * (Real.cos (2 * α))^4 - 6 * (Real.cos (2 * α))^2 * (Real.sin (2 * α))^2 + (Real.sin (2 * α))^4 = Real.cos (8 * α) := by
  sorry

end trig_identity_l2178_217814


namespace remainder_512_power_512_mod_13_l2178_217854

theorem remainder_512_power_512_mod_13 : 512^512 ≡ 1 [ZMOD 13] := by sorry

end remainder_512_power_512_mod_13_l2178_217854


namespace polynomial_equality_l2178_217898

theorem polynomial_equality (m n : ℝ) : 
  (∀ x : ℝ, (x + 1) * (2 * x - 3) = 2 * x^2 + m * x + n) → 
  m = -1 ∧ n = -3 := by
sorry

end polynomial_equality_l2178_217898


namespace log_sum_equals_two_l2178_217812

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end log_sum_equals_two_l2178_217812


namespace lottery_probability_l2178_217859

/-- A lottery with probabilities for certain number ranges -/
structure Lottery where
  prob_1_to_45 : ℚ
  prob_1_or_larger : ℚ
  prob_1_to_45_is_valid : prob_1_to_45 = 7/15
  prob_1_or_larger_is_valid : prob_1_or_larger = 14/15

/-- The probability of drawing a number less than or equal to 45 in the lottery -/
def prob_le_45 (l : Lottery) : ℚ := l.prob_1_to_45

theorem lottery_probability (l : Lottery) :
  prob_le_45 l = l.prob_1_to_45 := by sorry

end lottery_probability_l2178_217859


namespace number_of_incorrect_statements_l2178_217844

-- Define the triangles
def triangle1 (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 9^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 12^2 ∧
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2)

def triangle2 (a b c : ℝ) : Prop :=
  a = 7 ∧ b = 24 ∧ c = 25

def triangle3 (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 6^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 8^2 ∧
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2)

def triangle4 (a b c : ℝ) : Prop :=
  (a = 3 ∧ b = 3 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 3)

-- Define the statements
def statement1 (A B C : ℝ × ℝ) : Prop :=
  triangle1 A B C → abs ((B.2 - A.2) * C.1 + (A.1 - B.1) * C.2 + (B.1 * A.2 - A.1 * B.2)) / 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 9

def statement2 (a b c : ℝ) : Prop :=
  triangle2 a b c → a^2 + b^2 = c^2

def statement3 (A B C : ℝ × ℝ) : Prop :=
  triangle3 A B C → ∃ (M : ℝ × ℝ), 
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    Real.sqrt ((M.1 - C.1)^2 + (M.2 - C.2)^2) = 5

def statement4 (a b c : ℝ) : Prop :=
  triangle4 a b c → a + b + c = 13

-- Theorem to prove
theorem number_of_incorrect_statements :
  ∃ (A1 B1 C1 A3 B3 C3 : ℝ × ℝ) (a2 b2 c2 a4 b4 c4 : ℝ),
    (¬ statement1 A1 B1 C1) ∧
    statement2 a2 b2 c2 ∧
    (¬ statement3 A3 B3 C3) ∧
    (¬ statement4 a4 b4 c4) := by
  sorry

end number_of_incorrect_statements_l2178_217844


namespace right_angle_point_location_l2178_217800

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the plane
def Point := ℝ × ℝ

-- Define the property of being on the circle
def OnCircle (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the angle between three points
def Angle (p1 p2 p3 : Point) : ℝ := sorry

-- Define a right angle
def IsRightAngle (angle : ℝ) : Prop :=
  angle = Real.pi / 2

-- Define the property of being diametrically opposite
def DiametricallyOpposite (p1 p2 : Point) (c : Circle) : Prop :=
  (p1.1 + p2.1) / 2 = c.center.1 ∧ (p1.2 + p2.2) / 2 = c.center.2

-- The main theorem
theorem right_angle_point_location
  (c : Circle) (C : Point) (ho : OnCircle C c) :
  ∃! X, OnCircle X c ∧ IsRightAngle (Angle C X c.center) ∧ DiametricallyOpposite C X c :=
sorry

end right_angle_point_location_l2178_217800


namespace sine_cosine_equality_l2178_217826

theorem sine_cosine_equality (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (612 * π / 180) → n = -18 := by
  sorry

end sine_cosine_equality_l2178_217826


namespace kendras_earnings_theorem_l2178_217831

/-- Kendra's total earnings in 2014 and 2015 -/
def kendras_total_earnings (laurel_2014 : ℝ) : ℝ :=
  let kendra_2014 := laurel_2014 - 8000
  let kendra_2015 := laurel_2014 * 1.2
  kendra_2014 + kendra_2015

/-- Theorem stating Kendra's total earnings given Laurel's 2014 earnings -/
theorem kendras_earnings_theorem (laurel_2014 : ℝ) 
  (h : laurel_2014 = 30000) : 
  kendras_total_earnings laurel_2014 = 58000 := by
  sorry

end kendras_earnings_theorem_l2178_217831


namespace largest_share_proof_l2178_217866

def profit_split (partners : ℕ) (ratios : List ℕ) (total_profit : ℕ) : ℕ :=
  let total_parts := ratios.sum
  let part_value := total_profit / total_parts
  (ratios.maximum? |>.getD 0) * part_value

theorem largest_share_proof (partners : ℕ) (ratios : List ℕ) (total_profit : ℕ) 
  (h_partners : partners = 5)
  (h_ratios : ratios = [1, 2, 3, 3, 6])
  (h_profit : total_profit = 36000) :
  profit_split partners ratios total_profit = 14400 :=
by
  sorry

#eval profit_split 5 [1, 2, 3, 3, 6] 36000

end largest_share_proof_l2178_217866


namespace bulbs_chosen_l2178_217829

theorem bulbs_chosen (total : ℕ) (defective : ℕ) (prob : ℝ) :
  total = 20 →
  defective = 4 →
  prob = 0.368421052631579 →
  (∃ n : ℕ, n = 2 ∧ 1 - ((total - defective : ℝ) / total) ^ n = prob) :=
by sorry

end bulbs_chosen_l2178_217829


namespace sufficient_condition_implies_a_range_l2178_217853

theorem sufficient_condition_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by
  sorry

end sufficient_condition_implies_a_range_l2178_217853


namespace sum_of_ratios_l2178_217819

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) = f a * f b

theorem sum_of_ratios (f : ℝ → ℝ) 
  (h1 : functional_equation f) 
  (h2 : f 1 = 2) : 
  f 2 / f 1 + f 4 / f 3 + f 6 / f 5 = 6 := by
  sorry

end sum_of_ratios_l2178_217819


namespace fraction_difference_prime_l2178_217878

theorem fraction_difference_prime (p : ℕ) (hp : Prime p) :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / p ↔ x = p - 1 ∧ y = p * (p - 1) :=
by sorry

end fraction_difference_prime_l2178_217878


namespace house_sale_loss_percentage_l2178_217841

def initial_value : ℝ := 100000
def profit_percentage : ℝ := 0.10
def final_selling_price : ℝ := 99000

theorem house_sale_loss_percentage :
  let first_sale_price := initial_value * (1 + profit_percentage)
  let loss_amount := first_sale_price - final_selling_price
  let loss_percentage := loss_amount / first_sale_price * 100
  loss_percentage = 10 := by sorry

end house_sale_loss_percentage_l2178_217841


namespace angle_range_theorem_l2178_217856

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is monotonically increasing on an interval (a, b) if
    for all x, y in (a, b), x < y implies f(x) < f(y) -/
def MonoIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem angle_range_theorem (f : ℝ → ℝ) (A : ℝ) :
  IsOdd f →
  MonoIncreasing f 0 Real.pi →
  f (1/2) = 0 →
  f (Real.cos A) < 0 →
  (π/3 < A ∧ A < π/2) ∨ (2*π/3 < A ∧ A < π) :=
sorry

end angle_range_theorem_l2178_217856


namespace return_journey_time_l2178_217894

/-- Proves that given a round trip with specified conditions, the return journey takes 7 hours -/
theorem return_journey_time 
  (total_distance : ℝ) 
  (outbound_time : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_distance = 2000) 
  (h2 : outbound_time = 10) 
  (h3 : average_speed = 142.85714285714286) : 
  (total_distance / average_speed) - outbound_time = 7 := by
  sorry

end return_journey_time_l2178_217894


namespace power_of_two_expression_l2178_217813

theorem power_of_two_expression : 2^3 * 2^3 + 2^3 = 72 := by
  sorry

end power_of_two_expression_l2178_217813


namespace simplify_expression_l2178_217895

theorem simplify_expression (a b : ℝ) : 6*a - 8*b - 2*(3*a + b) = -10*b := by
  sorry

end simplify_expression_l2178_217895


namespace sale_price_determination_l2178_217834

/-- Proves that the sale price of each machine is $10,000 given the commission structure and total commission --/
theorem sale_price_determination (commission_rate_first_100 : ℝ) (commission_rate_after_100 : ℝ) 
  (total_machines : ℕ) (machines_at_first_rate : ℕ) (total_commission : ℝ) :
  commission_rate_first_100 = 0.03 →
  commission_rate_after_100 = 0.04 →
  total_machines = 130 →
  machines_at_first_rate = 100 →
  total_commission = 42000 →
  ∃ (sale_price : ℝ), 
    sale_price = 10000 ∧
    (machines_at_first_rate : ℝ) * commission_rate_first_100 * sale_price + 
    ((total_machines - machines_at_first_rate) : ℝ) * commission_rate_after_100 * sale_price = 
    total_commission :=
by
  sorry

#check sale_price_determination

end sale_price_determination_l2178_217834


namespace sin_2alpha_value_l2178_217821

theorem sin_2alpha_value (α : ℝ) (h : Real.sin (α - π/4) = -Real.cos (2*α)) :
  Real.sin (2*α) = -1/2 := by
  sorry

end sin_2alpha_value_l2178_217821


namespace perfect_square_analysis_l2178_217848

theorem perfect_square_analysis :
  (∃ (x : ℕ), 8^2050 = x^2) ∧
  (∃ (x : ℕ), 9^2048 = x^2) ∧
  (∀ (x : ℕ), 10^2051 ≠ x^2) ∧
  (∃ (x : ℕ), 11^2052 = x^2) ∧
  (∃ (x : ℕ), 12^2050 = x^2) :=
by sorry

end perfect_square_analysis_l2178_217848


namespace paperboy_delivery_patterns_l2178_217806

def deliverySequences (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 4 => 15
  | m + 5 => deliverySequences (m + 4) + deliverySequences (m + 3) + 
             deliverySequences (m + 2) + deliverySequences (m + 1)

theorem paperboy_delivery_patterns : deliverySequences 12 = 2873 := by
  sorry

end paperboy_delivery_patterns_l2178_217806


namespace democrat_ratio_l2178_217890

/-- Represents the number of participants in a meeting with democrats -/
structure Meeting where
  total : ℕ
  female : ℕ
  male : ℕ
  femaleDemocrats : ℕ
  maleDemocrats : ℕ

/-- The properties of the meeting as described in the problem -/
def meetingProperties (m : Meeting) : Prop :=
  m.total = 750 ∧
  m.female + m.male = m.total ∧
  m.femaleDemocrats = m.female / 2 ∧
  m.maleDemocrats = m.male / 4 ∧
  m.femaleDemocrats = 125

/-- The theorem stating that the ratio of democrats to total participants is 1:3 -/
theorem democrat_ratio (m : Meeting) (h : meetingProperties m) :
  (m.femaleDemocrats + m.maleDemocrats) * 3 = m.total := by
  sorry


end democrat_ratio_l2178_217890


namespace barts_earnings_l2178_217879

/-- Represents the earnings for a single day --/
structure DayEarnings where
  rate : Rat
  questionsPerSurvey : Nat
  surveysCompleted : Nat

/-- Calculates the total earnings for a given day --/
def calculateDayEarnings (day : DayEarnings) : Rat :=
  day.rate * day.questionsPerSurvey * day.surveysCompleted

/-- Calculates the total earnings for three days --/
def calculateTotalEarnings (day1 day2 day3 : DayEarnings) : Rat :=
  calculateDayEarnings day1 + calculateDayEarnings day2 + calculateDayEarnings day3

/-- Theorem statement for Bart's earnings over three days --/
theorem barts_earnings :
  let monday : DayEarnings := { rate := 1/5, questionsPerSurvey := 10, surveysCompleted := 3 }
  let tuesday : DayEarnings := { rate := 1/4, questionsPerSurvey := 12, surveysCompleted := 4 }
  let wednesday : DayEarnings := { rate := 1/10, questionsPerSurvey := 15, surveysCompleted := 5 }
  calculateTotalEarnings monday tuesday wednesday = 51/2 := by
  sorry

end barts_earnings_l2178_217879


namespace vector_dot_product_problem_l2178_217893

def a : ℝ × ℝ := (-1, 1)
def b (m : ℝ) : ℝ × ℝ := (1, m)

theorem vector_dot_product_problem (m : ℝ) : 
  (2 * a - b m) • a = 4 → m = 1 := by sorry

end vector_dot_product_problem_l2178_217893


namespace fraction_defined_iff_not_five_l2178_217865

theorem fraction_defined_iff_not_five (x : ℝ) : IsRegular (x - 5)⁻¹ ↔ x ≠ 5 := by sorry

end fraction_defined_iff_not_five_l2178_217865


namespace repeating_decimal_ratio_l2178_217811

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := a * 10 + b / 99

/-- The main theorem stating that the ratio of two specific repeating decimals equals 9/4 -/
theorem repeating_decimal_ratio : 
  (RepeatingDecimal 8 1) / (RepeatingDecimal 3 6) = 9 / 4 := by
  sorry

end repeating_decimal_ratio_l2178_217811


namespace symmetry_coincidence_l2178_217807

-- Define the type for points in the plane
def Point : Type := ℝ × ℝ

-- Define the symmetry operation
def symmetric (A B O : Point) : Prop := 
  ∃ (x y : ℝ), A = (x, y) ∧ B = (2 * O.1 - x, 2 * O.2 - y)

-- Define the given points
variable (A A₁ A₂ A₃ A₄ A₅ A₆ O₁ O₂ O₃ : Point)

-- State the theorem
theorem symmetry_coincidence 
  (h1 : symmetric A A₁ O₁)
  (h2 : symmetric A₁ A₂ O₂)
  (h3 : symmetric A₂ A₃ O₃)
  (h4 : symmetric A₃ A₄ O₁)
  (h5 : symmetric A₄ A₅ O₂)
  (h6 : symmetric A₅ A₆ O₃) :
  A = A₆ := by sorry

end symmetry_coincidence_l2178_217807


namespace gcd_18_30_42_l2178_217804

theorem gcd_18_30_42 : Nat.gcd 18 (Nat.gcd 30 42) = 6 := by
  sorry

end gcd_18_30_42_l2178_217804


namespace odd_function_negative_values_l2178_217815

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_values
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = x + 1) :
  ∀ x < 0, f x = x - 1 :=
by sorry

end odd_function_negative_values_l2178_217815


namespace range_of_a_for_equation_solution_l2178_217871

theorem range_of_a_for_equation_solution (a : ℝ) : 
  (∃ x : ℝ, (a + Real.cos x) * (a - Real.sin x) = 1) ↔ 
  (a ∈ Set.Icc (-1 - Real.sqrt 2 / 2) (-1 + Real.sqrt 2 / 2) ∪ 
   Set.Icc (1 - Real.sqrt 2 / 2) (1 + Real.sqrt 2 / 2)) := by
  sorry

end range_of_a_for_equation_solution_l2178_217871


namespace study_time_calculation_l2178_217801

theorem study_time_calculation (total_hours : ℝ) (tv_fraction : ℝ) (study_fraction : ℝ) : 
  total_hours = 24 →
  tv_fraction = 1/5 →
  study_fraction = 1/4 →
  (total_hours * (1 - tv_fraction) * study_fraction) * 60 = 288 := by
sorry

end study_time_calculation_l2178_217801


namespace modulus_z_is_sqrt_2_l2178_217810

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number z -/
noncomputable def z : ℂ := (2 * i) / (1 + i)

/-- Theorem: The modulus of z is √2 -/
theorem modulus_z_is_sqrt_2 : Complex.abs z = Real.sqrt 2 := by sorry

end modulus_z_is_sqrt_2_l2178_217810


namespace coral_reef_age_conversion_l2178_217860

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : Nat) : Nat :=
  let d0 := octal % 10
  let d1 := (octal / 10) % 10
  let d2 := (octal / 100) % 10
  let d3 := (octal / 1000) % 10
  d0 * 8^0 + d1 * 8^1 + d2 * 8^2 + d3 * 8^3

theorem coral_reef_age_conversion :
  octal_to_decimal 3456 = 1838 := by
  sorry

end coral_reef_age_conversion_l2178_217860


namespace set_theory_propositions_l2178_217864

theorem set_theory_propositions (A B : Set α) : 
  (∀ a, a ∈ A → a ∈ A ∪ B) ∧
  (A ⊆ B → A ∪ B = B) ∧
  (A ∪ B = B → A ∩ B = A) ∧
  ¬(∀ a, a ∈ B → a ∈ A ∩ B) ∧
  ¬(∀ C, A ∪ B = B ∪ C → A = C) :=
by sorry

end set_theory_propositions_l2178_217864


namespace seed_mixture_percentage_l2178_217861

/-- Given two seed mixtures X and Y, and their combination, 
    this theorem proves that the percentage of mixture X in the final mixture is 1/3. -/
theorem seed_mixture_percentage (x y : ℝ) :
  x ≥ 0 → y ≥ 0 →  -- Ensure non-negative weights
  0.40 * x + 0.25 * y = 0.30 * (x + y) →  -- Ryegrass balance equation
  x / (x + y) = 1 / 3 := by sorry

end seed_mixture_percentage_l2178_217861


namespace original_class_size_l2178_217832

theorem original_class_size (x : ℕ) : 
  (40 * x + 15 * 32 = (x + 15) * 36) → x = 15 := by
  sorry

end original_class_size_l2178_217832


namespace find_n_l2178_217839

def A (i : ℕ) : ℕ := 2 * i - 1

def B (n i : ℕ) : ℕ := n - 2 * (i - 1)

theorem find_n : ∃ n : ℕ, 
  (∃ k : ℕ, A k = 19 ∧ B n k = 89) → n = 107 := by
  sorry

end find_n_l2178_217839


namespace quadrilateral_area_l2178_217830

-- Define the quadrilateral ABCD
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the point of intersection of diagonals
def P (q : Quadrilateral) : Point := sorry

-- Define the distances from A, B, and P to line CD
def a (q : Quadrilateral) : ℝ := sorry
def b (q : Quadrilateral) : ℝ := sorry
def p (q : Quadrilateral) : ℝ := sorry

-- Define the length of side CD
def CD (q : Quadrilateral) : ℝ := sorry

-- Define the area of the quadrilateral
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area (q : Quadrilateral) : 
  area q = (a q * b q * CD q) / (2 * p q) := by sorry

end quadrilateral_area_l2178_217830


namespace number_with_inserted_zero_l2178_217809

def insert_zero (n : ℕ) : ℕ :=
  10000 * (n / 1000) + 1000 * ((n / 100) % 10) + (n % 100)

theorem number_with_inserted_zero (N : ℕ) :
  (insert_zero N = 9 * N) → (N = 225 ∨ N = 450 ∨ N = 675) := by
sorry

end number_with_inserted_zero_l2178_217809


namespace largest_power_dividing_factorial_squared_l2178_217825

theorem largest_power_dividing_factorial_squared (p : ℕ) (hp : Nat.Prime p) :
  (∀ n : ℕ, (Nat.factorial p)^n ∣ Nat.factorial (p^2)) ↔ n ≤ p + 1 :=
sorry

end largest_power_dividing_factorial_squared_l2178_217825


namespace exponent_product_equals_twentyfive_l2178_217882

theorem exponent_product_equals_twentyfive :
  (5 ^ 0.4) * (5 ^ 0.6) * (5 ^ 0.2) * (5 ^ 0.3) * (5 ^ 0.5) = 25 := by
  sorry

end exponent_product_equals_twentyfive_l2178_217882


namespace max_value_of_function_max_value_achievable_l2178_217870

theorem max_value_of_function (x : ℝ) (hx : x > 0) : 
  (-2 * x^2 + x - 3) / x ≤ 1 - 2 * Real.sqrt 6 := by sorry

theorem max_value_achievable : 
  ∃ x : ℝ, x > 0 ∧ (-2 * x^2 + x - 3) / x = 1 - 2 * Real.sqrt 6 := by sorry

end max_value_of_function_max_value_achievable_l2178_217870


namespace two_digit_special_property_l2178_217855

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

theorem two_digit_special_property : 
  {n : ℕ | is_two_digit n ∧ n = 6 * sum_of_digits (n + 7)} = {24, 78} := by
  sorry

end two_digit_special_property_l2178_217855


namespace min_reciprocal_distances_min_value_achieved_l2178_217820

/-- Given a right triangle ABC with AC = 4 and BC = 1, and a point P on the hypotenuse AB
    (excluding endpoints) with distances d1 and d2 to the legs, the minimum value of (1/d1 + 1/d2) is 9/4 -/
theorem min_reciprocal_distances (d1 d2 : ℝ) (h1 : d1 > 0) (h2 : d2 > 0) (h3 : d1 + 4 * d2 = 4) :
  (1 / d1 + 1 / d2) ≥ 9 / 4 := by
  sorry

/-- The minimum value 9/4 is achieved when d1 = 4/3 and d2 = 2/3 -/
theorem min_value_achieved (d1 d2 : ℝ) (h1 : d1 > 0) (h2 : d2 > 0) (h3 : d1 + 4 * d2 = 4) :
  (1 / d1 + 1 / d2 = 9 / 4) ↔ (d1 = 4 / 3 ∧ d2 = 2 / 3) := by
  sorry

end min_reciprocal_distances_min_value_achieved_l2178_217820


namespace symmetric_point_6_1_l2178_217833

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) -/
def origin : Point2D := ⟨0, 0⟩

/-- Function to find the symmetric point with respect to the origin -/
def symmetricPoint (p : Point2D) : Point2D :=
  ⟨-p.x, -p.y⟩

/-- Theorem: The point symmetric to (6, 1) with respect to the origin is (-6, -1) -/
theorem symmetric_point_6_1 :
  symmetricPoint ⟨6, 1⟩ = ⟨-6, -1⟩ := by
  sorry

end symmetric_point_6_1_l2178_217833


namespace clayton_first_game_score_l2178_217823

def clayton_basketball_score (game1 : ℝ) : Prop :=
  let game2 : ℝ := 14
  let game3 : ℝ := 6
  let game4 : ℝ := (game1 + game2 + game3) / 3
  let total : ℝ := 40
  (game1 + game2 + game3 + game4 = total) ∧ (game1 = 10)

theorem clayton_first_game_score :
  ∃ (game1 : ℝ), clayton_basketball_score game1 :=
sorry

end clayton_first_game_score_l2178_217823


namespace range_of_m_l2178_217881

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x > m

def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*m*x + 2 - m ≤ 0

theorem range_of_m (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ Set.Ioo (-2) (-1) ∪ Set.Ici 1 :=
sorry

end range_of_m_l2178_217881


namespace reciprocal_of_negative_sqrt_three_l2178_217850

theorem reciprocal_of_negative_sqrt_three :
  ((-Real.sqrt 3)⁻¹ : ℝ) = -(Real.sqrt 3 / 3) := by sorry

end reciprocal_of_negative_sqrt_three_l2178_217850


namespace inequality_solution_set_l2178_217868

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 6) / (x - 4) ≥ 3 ↔ x ∈ Set.Iio 4 ∪ Set.Ioi 4 :=
by sorry

end inequality_solution_set_l2178_217868


namespace only_third_set_forms_triangle_l2178_217867

/-- Checks if three lengths can form a triangle according to the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The sets of line segments given in the problem -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(1, 2, 3), (2, 2, 4), (3, 4, 5), (3, 5, 9)]

theorem only_third_set_forms_triangle :
  ∃! set : ℝ × ℝ × ℝ, set ∈ segment_sets ∧ can_form_triangle set.1 set.2.1 set.2.2 :=
by
  sorry

end only_third_set_forms_triangle_l2178_217867


namespace unique_number_with_gcd_l2178_217876

theorem unique_number_with_gcd : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 85 ∧ Nat.gcd 36 n = 9 :=
by
  -- The proof goes here
  sorry

end unique_number_with_gcd_l2178_217876


namespace largest_number_with_property_l2178_217842

def sum_of_digits (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.sum

def property (n : Nat) : Prop :=
  n % sum_of_digits n = 0

theorem largest_number_with_property :
  ∃ (n : Nat), n < 900 ∧ property n ∧ ∀ (m : Nat), m < 900 → property m → m ≤ n :=
by
  use 888
  sorry

#eval sum_of_digits 888  -- Should output 24
#eval 888 % 24           -- Should output 0

end largest_number_with_property_l2178_217842


namespace alcohol_mixture_ratio_l2178_217869

/-- Proves that mixing equal volumes of two alcohol solutions results in a specific alcohol-to-water ratio -/
theorem alcohol_mixture_ratio (volume : ℝ) (p_concentration q_concentration : ℝ)
  (h_volume_pos : volume > 0)
  (h_p_conc : p_concentration = 0.625)
  (h_q_conc : q_concentration = 0.875) :
  let total_volume := 2 * volume
  let total_alcohol := volume * (p_concentration + q_concentration)
  let total_water := total_volume - total_alcohol
  (total_alcohol / total_water) = 3 := by
  sorry

end alcohol_mixture_ratio_l2178_217869


namespace consecutive_odd_product_sum_l2178_217852

theorem consecutive_odd_product_sum (a b c : ℤ) : 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧  -- a, b, c are odd
  (b = a + 2) ∧ (c = b + 2) ∧                -- a, b, c are consecutive
  (a * b * c = 9177) →                       -- their product is 9177
  (a + b + c = 63) :=                        -- their sum is 63
by sorry

end consecutive_odd_product_sum_l2178_217852


namespace nth_equation_l2178_217862

theorem nth_equation (n : ℕ) (h : n > 0) :
  (n + 2 : ℚ) / n - 2 / (n + 2) = ((n + 2)^2 + n^2 : ℚ) / (n * (n + 2)) - 1 := by
  sorry

end nth_equation_l2178_217862


namespace mahesh_estimate_less_than_true_value_l2178_217874

theorem mahesh_estimate_less_than_true_value 
  (a b d : ℕ) 
  (h1 : a > b) 
  (h2 : d > 0) : 
  (a - d)^2 - (b + d)^2 < a^2 - b^2 := by
  sorry

end mahesh_estimate_less_than_true_value_l2178_217874


namespace simplify_expression_l2178_217896

theorem simplify_expression (n : ℕ) : 
  (3^(n+5) - 3*(3^n)) / (3*(3^(n+4))) = 80 / 27 := by
sorry

end simplify_expression_l2178_217896


namespace polynomial_division_l2178_217818

theorem polynomial_division (x : ℝ) :
  x^5 + 3*x^4 - 28*x^3 + 15*x^2 - 21*x + 8 =
  (x - 3) * (x^4 + 6*x^3 - 10*x^2 - 15*x - 66) + (-100) := by
  sorry

end polynomial_division_l2178_217818


namespace max_true_statements_l2178_217835

theorem max_true_statements (x : ℝ) : ∃ (n : ℕ), n ≤ 3 ∧
  n = (Bool.toNat (0 < x^2 ∧ x^2 < 4) +
       Bool.toNat (x^2 > 4) +
       Bool.toNat (-2 < x ∧ x < 0) +
       Bool.toNat (0 < x ∧ x < 2) +
       Bool.toNat (0 < x - x^2 ∧ x - x^2 < 4)) :=
by sorry

end max_true_statements_l2178_217835


namespace triangle_area_l2178_217886

-- Define the point P
def P : ℝ × ℝ := (2, 5)

-- Define the slopes of the two lines
def slope1 : ℝ := -1
def slope2 : ℝ := 1.5

-- Define Q and R as the x-intercepts of the lines
def Q : ℝ × ℝ := (-3, 0)
def R : ℝ × ℝ := (5.33, 0)

-- Theorem statement
theorem triangle_area : 
  let triangle_area := (1/2) * (R.1 - Q.1) * P.2
  triangle_area = 20.825 := by sorry

end triangle_area_l2178_217886


namespace min_pages_for_baseball_cards_l2178_217843

/-- Represents the number of cards that can be held by each type of page -/
structure PageCapacity where
  x : Nat
  y : Nat

/-- Calculates the minimum number of pages needed to hold all cards -/
def minPages (totalCards : Nat) (capacity : PageCapacity) : Nat :=
  let fullXPages := totalCards / capacity.x
  let remainingCards := totalCards % capacity.x
  if remainingCards = 0 then
    fullXPages
  else if remainingCards ≤ capacity.y then
    fullXPages + 1
  else
    fullXPages + 2

/-- Theorem stating the minimum number of pages needed for the given problem -/
theorem min_pages_for_baseball_cards :
  let totalCards := 1040
  let capacity : PageCapacity := { x := 12, y := 10 }
  minPages totalCards capacity = 87 := by
  sorry

#eval minPages 1040 { x := 12, y := 10 }

end min_pages_for_baseball_cards_l2178_217843


namespace min_value_7x_5y_l2178_217887

theorem min_value_7x_5y (x y : ℕ) 
  (h1 : ∃ k : ℤ, x + 2*y = 5*k)
  (h2 : ∃ m : ℤ, x + y = 3*m)
  (h3 : 2*x + y ≥ 99) :
  7*x + 5*y ≥ 366 := by
sorry

end min_value_7x_5y_l2178_217887
