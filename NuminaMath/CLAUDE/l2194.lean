import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2194_219405

/-- Given two lines in a 2D plane:
    1. y = 3x + 4
    2. y = x
    This theorem states that the line symmetric to y = 3x + 4
    with respect to y = x has the equation y = (1/3)x - (4/3) -/
theorem symmetric_line_equation :
  let line1 : ℝ → ℝ := λ x => 3 * x + 4
  let line2 : ℝ → ℝ := λ x => x
  let symmetric_line : ℝ → ℝ := λ x => (1/3) * x - (4/3)
  ∀ x y : ℝ,
    (y = line1 x ∧ 
     ∃ x' y', x' = y ∧ y' = x ∧ y' = line2 x') →
    y = symmetric_line x :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l2194_219405


namespace NUMINAMATH_CALUDE_zero_in_A_l2194_219404

def A : Set ℝ := {x | x * (x + 1) = 0}

theorem zero_in_A : (0 : ℝ) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_zero_in_A_l2194_219404


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_l2194_219481

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) + 1

theorem tangent_line_at_zero : 
  let p : ℝ × ℝ := (0, f 0)
  let m : ℝ := -((deriv f) 0)
  ∀ x y : ℝ, (y - p.2 = m * (x - p.1)) ↔ (x + y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_l2194_219481


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_of_14400_l2194_219448

/-- The number of perfect square factors of 14400 -/
def perfect_square_factors_of_14400 : ℕ :=
  let n := 14400
  let prime_factorization := (2, 4) :: (3, 2) :: (5, 2) :: []
  sorry

/-- Theorem stating that the number of perfect square factors of 14400 is 12 -/
theorem count_perfect_square_factors_of_14400 :
  perfect_square_factors_of_14400 = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_of_14400_l2194_219448


namespace NUMINAMATH_CALUDE_shooting_probability_l2194_219483

theorem shooting_probability (accuracy : ℝ) (two_shots : ℝ) :
  accuracy = 9/10 →
  two_shots = 1/2 →
  (two_shots / accuracy) = 5/9 :=
by sorry

end NUMINAMATH_CALUDE_shooting_probability_l2194_219483


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2194_219477

theorem absolute_value_inequality (y : ℝ) : 
  (2 ≤ |y - 5| ∧ |y - 5| ≤ 8) ↔ ((-3 ≤ y ∧ y ≤ 3) ∨ (7 ≤ y ∧ y ≤ 13)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2194_219477


namespace NUMINAMATH_CALUDE_geometric_properties_l2194_219417

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of a line passing through a point
def passes_through (l : Line) (p : Point) : Prop := sorry

-- Define the concept of corresponding angles
def corresponding_angles (a1 a2 : Angle) : Prop := sorry

-- Define vertical angles
def vertical_angles (a1 a2 : Angle) : Prop := sorry

theorem geometric_properties :
  -- Statement 1
  (∀ a b c : Line, parallel a b → parallel b c → parallel a c) ∧
  -- Statement 2
  (∀ a1 a2 : Angle, corresponding_angles a1 a2 → a1 = a2) ∧
  -- Statement 3
  (∀ p : Point, ∀ l : Line, ∃! m : Line, passes_through m p ∧ parallel m l) ∧
  -- Statement 4
  (∀ a1 a2 : Angle, vertical_angles a1 a2 → a1 = a2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_properties_l2194_219417


namespace NUMINAMATH_CALUDE_triangle_side_sum_l2194_219413

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  b * Real.cos C + c * Real.cos B = 3 * a * Real.cos B →
  b = 2 →
  (1 / 2) * a * c * Real.sin B = 3 * Real.sqrt 2 / 2 →
  a + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l2194_219413


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2194_219490

theorem polynomial_factorization (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3)*(8*x^2 + x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2194_219490


namespace NUMINAMATH_CALUDE_max_value_x4y2z_l2194_219470

theorem max_value_x4y2z (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x^2 + y^2 + z^2 = 1) :
  x^4 * y^2 * z ≤ 32 / (16807 * Real.sqrt 7) ∧ 
  ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 + z^2 = 1 ∧ x^4 * y^2 * z = 32 / (16807 * Real.sqrt 7) := by
  sorry

#check max_value_x4y2z

end NUMINAMATH_CALUDE_max_value_x4y2z_l2194_219470


namespace NUMINAMATH_CALUDE_sum_of_digits_l2194_219476

theorem sum_of_digits (P Q R : ℕ) : 
  P < 10 → Q < 10 → R < 10 →
  P ≠ Q → P ≠ R → Q ≠ R →
  P * 100 + 70 + R + 300 + 90 + R = R * 100 + Q * 10 →
  P + Q + R = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l2194_219476


namespace NUMINAMATH_CALUDE_remaining_credit_l2194_219415

/-- Calculates the remaining credit to be paid given a credit limit and two payments -/
theorem remaining_credit (credit_limit : ℕ) (payment1 : ℕ) (payment2 : ℕ) :
  credit_limit = 100 →
  payment1 = 15 →
  payment2 = 23 →
  credit_limit - (payment1 + payment2) = 62 := by
  sorry

#check remaining_credit

end NUMINAMATH_CALUDE_remaining_credit_l2194_219415


namespace NUMINAMATH_CALUDE_phone_plan_comparison_l2194_219427

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

end NUMINAMATH_CALUDE_phone_plan_comparison_l2194_219427


namespace NUMINAMATH_CALUDE_line_maximizing_midpoint_distance_l2194_219431

/-- The equation of a line that intercepts a circle, maximizing the distance from the origin to the chord's midpoint -/
theorem line_maximizing_midpoint_distance 
  (x y a b c : ℝ) 
  (circle_eq : x^2 + y^2 = 16)
  (line_eq : a*x + b*y + c = 0)
  (condition : a + 2*b - c = 0)
  (is_max : ∀ (x' y' : ℝ), x'^2 + y'^2 ≤ (x^2 + y^2) / 4) :
  x + 2*y + 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_line_maximizing_midpoint_distance_l2194_219431


namespace NUMINAMATH_CALUDE_prob_no_red_3x3_is_170_171_l2194_219457

/-- Represents a 4x4 grid where each cell can be either red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a 3x3 subgrid starting at (i, j) is all red -/
def is_red_3x3 (g : Grid) (i j : Fin 2) : Prop :=
  ∀ x y, g (i + x) (j + y) = true

/-- The probability of a random 4x4 grid not containing a 3x3 red square -/
def prob_no_red_3x3 : ℚ :=
  170 / 171

/-- The main theorem stating the probability of a 4x4 grid not containing a 3x3 red square -/
theorem prob_no_red_3x3_is_170_171 :
  prob_no_red_3x3 = 170 / 171 := by sorry

end NUMINAMATH_CALUDE_prob_no_red_3x3_is_170_171_l2194_219457


namespace NUMINAMATH_CALUDE_zeroth_power_of_nonzero_rational_l2194_219479

theorem zeroth_power_of_nonzero_rational (r : ℚ) (h : r ≠ 0) : r^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zeroth_power_of_nonzero_rational_l2194_219479


namespace NUMINAMATH_CALUDE_rational_as_cube_sum_ratio_l2194_219469

theorem rational_as_cube_sum_ratio (q : ℚ) (hq : 0 < q) : 
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
    q = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_rational_as_cube_sum_ratio_l2194_219469


namespace NUMINAMATH_CALUDE_polynomial_sum_l2194_219452

theorem polynomial_sum (p : ℝ → ℝ) : 
  (∀ x, p x + (2 * x^2 + 5 * x - 2) = 2 * x^2 + 5 * x + 4) → 
  (∀ x, p x = 6) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2194_219452


namespace NUMINAMATH_CALUDE_equation_infinite_solutions_l2194_219446

theorem equation_infinite_solutions (c : ℝ) : 
  (∀ y : ℝ, 3 * (5 + 2 * c * y) = 18 * y + 15) ↔ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_infinite_solutions_l2194_219446


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l2194_219454

theorem max_value_of_expression (x : ℝ) :
  (x^4) / (x^8 + 4*x^6 + x^4 + 4*x^2 + 16) ≤ 1/17 :=
by sorry

theorem max_value_achievable :
  ∃ x : ℝ, (x^4) / (x^8 + 4*x^6 + x^4 + 4*x^2 + 16) = 1/17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l2194_219454


namespace NUMINAMATH_CALUDE_circular_track_length_l2194_219437

/-- Represents a circular track with two runners --/
structure CircularTrack where
  length : ℝ
  runner1_speed : ℝ
  runner2_speed : ℝ

/-- Represents a meeting point of the runners --/
structure MeetingPoint where
  distance1 : ℝ  -- Distance run by runner 1
  distance2 : ℝ  -- Distance run by runner 2

/-- The theorem to be proved --/
theorem circular_track_length
  (track : CircularTrack)
  (first_meeting : MeetingPoint)
  (second_meeting : MeetingPoint) :
  (first_meeting.distance1 = 100) →
  (second_meeting.distance2 - first_meeting.distance2 = 150) →
  (track.runner1_speed > 0) →
  (track.runner2_speed > 0) →
  (track.length = 500) :=
by sorry

end NUMINAMATH_CALUDE_circular_track_length_l2194_219437


namespace NUMINAMATH_CALUDE_euler_formula_imaginary_part_l2194_219494

open Complex

theorem euler_formula_imaginary_part :
  let z : ℂ := Complex.exp (I * Real.pi / 4)
  let w : ℂ := z / (1 - I)
  Complex.im w = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_euler_formula_imaginary_part_l2194_219494


namespace NUMINAMATH_CALUDE_solution_set_part_i_solution_range_part_ii_l2194_219435

-- Define the functions f and g
def f (x : ℝ) := |x - 1|
def g (a x : ℝ) := 2 * |x + a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f x - g 1 x > 1} = {x : ℝ | -1 < x ∧ x < -1/3} :=
sorry

-- Part II
theorem solution_range_part_ii :
  ∀ a : ℝ, (∃ x : ℝ, 2 * f x + g a x ≤ (a + 1)^2) ↔ (a ≤ -3 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_i_solution_range_part_ii_l2194_219435


namespace NUMINAMATH_CALUDE_sum_of_unique_areas_l2194_219450

-- Define a structure for right triangles with integer leg lengths
structure SuperCoolTriangle where
  a : ℕ
  b : ℕ
  h : (a * b) / 2 = 3 * (a + b)

-- Define a function to calculate the area of a triangle
def triangleArea (t : SuperCoolTriangle) : ℕ := (t.a * t.b) / 2

-- Define a function to get all unique areas of super cool triangles
def uniqueAreas : List ℕ := sorry

-- Theorem statement
theorem sum_of_unique_areas : (uniqueAreas.sum) = 471 := by sorry

end NUMINAMATH_CALUDE_sum_of_unique_areas_l2194_219450


namespace NUMINAMATH_CALUDE_root_problem_l2194_219488

-- Define the polynomials
def p (c d : ℝ) (x : ℝ) : ℝ := (x + c) * (x + d) * (x + 15)
def q (c d : ℝ) (x : ℝ) : ℝ := (x + 3 * c) * (x + 5) * (x + 9)

-- State the theorem
theorem root_problem (c d : ℝ) :
  (∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
    p c d r1 = 0 ∧ p c d r2 = 0 ∧ p c d r3 = 0) ∧
  (∃! (r : ℝ), q c d r = 0) ∧
  c ≠ d ∧ c ≠ 4 ∧ c ≠ 15 ∧ d ≠ 4 ∧ d ≠ 15 ∧ d ≠ 5 →
  100 * c + d = 157 := by
sorry

end NUMINAMATH_CALUDE_root_problem_l2194_219488


namespace NUMINAMATH_CALUDE_sequences_and_sum_theorem_l2194_219412

/-- Definition of sequence a_n -/
def a (n : ℕ+) : ℕ :=
  if n = 1 then 1 else 2 * n.val - 1

/-- Definition of sequence b_n -/
def b (n : ℕ+) : ℚ :=
  if n = 1 then 1 else 2^(2 - n.val)

/-- Definition of S_n (sum of first n terms of a_n) -/
def S (n : ℕ+) : ℕ := (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- Definition of T_n (sum of first n terms of a_n * b_n) -/
def T (n : ℕ+) : ℚ := 11 - (2 * n.val + 3) * 2^(2 - n.val)

theorem sequences_and_sum_theorem (n : ℕ+) :
  (∀ (k : ℕ+), k ≥ 2 → S (k + 1) + S (k - 1) = 2 * (S k + 1)) ∧
  (∀ (k : ℕ+), (Finset.range k.val).sum (λ i => 2^i * b ⟨i + 1, Nat.succ_pos i⟩) = a k) →
  (∀ (k : ℕ+), a k = 2 * k.val - 1) ∧
  (∀ (k : ℕ+), b k = if k = 1 then 1 else 2^(2 - k.val)) ∧
  (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩ * b ⟨i + 1, Nat.succ_pos i⟩) = T n :=
by sorry


end NUMINAMATH_CALUDE_sequences_and_sum_theorem_l2194_219412


namespace NUMINAMATH_CALUDE_lottery_problem_l2194_219414

/-- Represents a lottery with prizes and blanks. -/
structure Lottery where
  prizes : ℕ
  blanks : ℕ
  prob_win : ℝ
  h_prob : prob_win = prizes / (prizes + blanks : ℝ)

/-- The lottery problem statement. -/
theorem lottery_problem (L : Lottery)
  (h_prizes : L.prizes = 10)
  (h_prob : L.prob_win = 0.2857142857142857) :
  L.blanks = 25 := by
  sorry

#check lottery_problem

end NUMINAMATH_CALUDE_lottery_problem_l2194_219414


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l2194_219496

def systematic_sample_count (total_population : ℕ) (sample_size : ℕ) (range_start : ℕ) (range_end : ℕ) : ℕ :=
  let group_size := total_population / sample_size
  ((range_end - range_start + 1) / group_size)

theorem systematic_sample_theorem :
  systematic_sample_count 800 20 121 400 = 7 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l2194_219496


namespace NUMINAMATH_CALUDE_smaller_cube_volume_l2194_219433

theorem smaller_cube_volume 
  (large_cube_volume : ℝ) 
  (num_small_cubes : ℕ) 
  (surface_area_diff : ℝ) : ℝ :=
by
  have h1 : large_cube_volume = 343 := by sorry
  have h2 : num_small_cubes = 343 := by sorry
  have h3 : surface_area_diff = 1764 := by sorry
  
  -- Define the side length of the large cube
  let large_side : ℝ := large_cube_volume ^ (1/3)
  
  -- Define the surface area of the large cube
  let large_surface_area : ℝ := 6 * large_side^2
  
  -- Define the volume of each small cube
  let small_cube_volume : ℝ := large_cube_volume / num_small_cubes
  
  -- Define the side length of each small cube
  let small_side : ℝ := small_cube_volume ^ (1/3)
  
  -- Define the total surface area of all small cubes
  let total_small_surface_area : ℝ := 6 * small_side^2 * num_small_cubes
  
  -- The main theorem
  have : small_cube_volume = 1 := by sorry

  exact small_cube_volume

end NUMINAMATH_CALUDE_smaller_cube_volume_l2194_219433


namespace NUMINAMATH_CALUDE_volume_ratio_equals_edge_product_ratio_l2194_219400

/-- Represent a tetrahedron with vertex O and edges OA, OB, OC -/
structure Tetrahedron where
  a : ℝ  -- length of OA
  b : ℝ  -- length of OB
  c : ℝ  -- length of OC
  volume : ℝ  -- volume of the tetrahedron

/-- Two tetrahedrons with congruent trihedral angles at O and O' -/
def CongruentTrihedralTetrahedrons (t1 t2 : Tetrahedron) : Prop :=
  -- We don't explicitly define the congruence, as it's given in the problem statement
  True

theorem volume_ratio_equals_edge_product_ratio
  (t1 t2 : Tetrahedron)
  (h : CongruentTrihedralTetrahedrons t1 t2) :
  t2.volume / t1.volume = (t2.a * t2.b * t2.c) / (t1.a * t1.b * t1.c) := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_equals_edge_product_ratio_l2194_219400


namespace NUMINAMATH_CALUDE_smallest_multiple_of_9_and_21_l2194_219461

theorem smallest_multiple_of_9_and_21 :
  ∃ (b : ℕ), b > 0 ∧ 9 ∣ b ∧ 21 ∣ b ∧ ∀ (x : ℕ), x > 0 ∧ 9 ∣ x ∧ 21 ∣ x → b ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_9_and_21_l2194_219461


namespace NUMINAMATH_CALUDE_factorization_proof_l2194_219441

theorem factorization_proof (x y : ℝ) : 9*x^2*y - y = y*(3*x + 1)*(3*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2194_219441


namespace NUMINAMATH_CALUDE_complement_of_intersection_l2194_219472

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {1, 2, 3}

theorem complement_of_intersection :
  (U \ (A ∩ B)) = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l2194_219472


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_perpendicular_plane_and_contained_implies_perpendicular_l2194_219462

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (contained : Line → Plane → Prop)

-- Non-coincident lines and planes
variable (m n : Line)
variable (α β : Plane)
variable (h_diff_lines : m ≠ n)
variable (h_diff_planes : α ≠ β)

-- Theorem 1
theorem perpendicular_implies_parallel 
  (h1 : perpendicular_plane m α) 
  (h2 : perpendicular_plane m β) : 
  parallel_plane α β :=
sorry

-- Theorem 2
theorem perpendicular_plane_and_contained_implies_perpendicular 
  (h1 : perpendicular_plane m α) 
  (h2 : contained n α) : 
  perpendicular m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_perpendicular_plane_and_contained_implies_perpendicular_l2194_219462


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_equals_zero_one_l2194_219487

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Theorem statement
theorem intersection_A_complement_B_equals_zero_one :
  A ∩ (U \ B) = {x | 0 ≤ x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_equals_zero_one_l2194_219487


namespace NUMINAMATH_CALUDE_kekai_garage_sale_earnings_l2194_219459

/-- Calculates the amount of money Kekai has left after a garage sale --/
def kekais_money (num_shirts : ℕ) (num_pants : ℕ) (shirt_price : ℕ) (pants_price : ℕ) (share_fraction : ℚ) : ℚ :=
  let total_earned := num_shirts * shirt_price + num_pants * pants_price
  (total_earned : ℚ) * (1 - share_fraction)

/-- Proves that Kekai has $10 left after the garage sale --/
theorem kekai_garage_sale_earnings : kekais_money 5 5 1 3 (1/2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_kekai_garage_sale_earnings_l2194_219459


namespace NUMINAMATH_CALUDE_remainder_of_product_l2194_219445

theorem remainder_of_product (n : ℕ) (h : n = 67545) : (n * 11) % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_l2194_219445


namespace NUMINAMATH_CALUDE_complement_of_A_l2194_219478

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 1| > 1}

-- State the theorem
theorem complement_of_A : 
  Set.compl A = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2194_219478


namespace NUMINAMATH_CALUDE_license_plate_increase_l2194_219456

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  new_plates / old_plates = 26^2 / 10 := by
sorry

end NUMINAMATH_CALUDE_license_plate_increase_l2194_219456


namespace NUMINAMATH_CALUDE_factorial_sum_theorem_l2194_219409

def is_solution (x y : ℕ) (z : ℤ) : Prop :=
  (Nat.factorial x + Nat.factorial y = 16 * z + 2017) ∧
  z % 2 ≠ 0

theorem factorial_sum_theorem :
  ∀ x y : ℕ, ∀ z : ℤ,
    is_solution x y z →
    ((x = 1 ∧ y = 6 ∧ z = -81) ∨
     (x = 6 ∧ y = 1 ∧ z = -81) ∨
     (x = 1 ∧ y = 7 ∧ z = 189) ∨
     (x = 7 ∧ y = 1 ∧ z = 189)) :=
by
  sorry

#check factorial_sum_theorem

end NUMINAMATH_CALUDE_factorial_sum_theorem_l2194_219409


namespace NUMINAMATH_CALUDE_max_points_2079_l2194_219498

def points (x : ℕ) : ℕ :=
  (if x % 3 = 0 then 3 else 0) +
  (if x % 5 = 0 then 5 else 0) +
  (if x % 7 = 0 then 7 else 0) +
  (if x % 9 = 0 then 9 else 0) +
  (if x % 11 = 0 then 11 else 0)

theorem max_points_2079 :
  ∀ x : ℕ, 2017 ≤ x → x ≤ 2117 → points x ≤ points 2079 :=
by
  sorry

end NUMINAMATH_CALUDE_max_points_2079_l2194_219498


namespace NUMINAMATH_CALUDE_james_older_brother_age_l2194_219411

/-- Given information about John and James' ages, prove James' older brother's age -/
theorem james_older_brother_age :
  ∀ (john_age james_age : ℕ),
  john_age = 39 →
  john_age - 3 = 2 * (james_age + 6) →
  ∃ (james_brother_age : ℕ),
  james_brother_age = james_age + 4 ∧
  james_brother_age = 16 := by
sorry

end NUMINAMATH_CALUDE_james_older_brother_age_l2194_219411


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l2194_219497

/-- Two parallel lines in the plane -/
structure ParallelLines :=
  (a : ℝ)
  (l₁ : ℝ → ℝ → Prop)
  (l₂ : ℝ → ℝ → Prop)
  (h_l₁ : ∀ x y, l₁ x y ↔ x + (a - 1) * y + 2 = 0)
  (h_l₂ : ∀ x y, l₂ x y ↔ a * x + 2 * y + 1 = 0)
  (h_parallel : ∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₂ x₂ y₂ → (x₁ - x₂) * 2 = (y₁ - y₂) * (1 - a))

/-- Distance between two lines -/
def distance (l₁ l₂ : ℝ → ℝ → Prop) : ℝ := sorry

/-- Theorem: If the distance between two parallel lines is 3√5/5, then a = -1 -/
theorem parallel_lines_distance (lines : ParallelLines) :
  distance lines.l₁ lines.l₂ = 3 * Real.sqrt 5 / 5 → lines.a = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l2194_219497


namespace NUMINAMATH_CALUDE_gasoline_tank_capacity_l2194_219468

theorem gasoline_tank_capacity : ∃ (x : ℝ),
  x > 0 ∧
  (7/8 * x - 15 = 2/3 * x) ∧
  x = 72 := by
sorry

end NUMINAMATH_CALUDE_gasoline_tank_capacity_l2194_219468


namespace NUMINAMATH_CALUDE_expected_value_eight_sided_die_l2194_219453

def winnings (n : Nat) : ℝ := 8 - n

theorem expected_value_eight_sided_die :
  let outcomes := Finset.range 8
  (1 : ℝ) / 8 * (outcomes.sum (fun i => winnings (i + 1))) = (7 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_expected_value_eight_sided_die_l2194_219453


namespace NUMINAMATH_CALUDE_race_head_start_l2194_219440

theorem race_head_start (course_length : ℝ) (speed_ratio : ℝ) (head_start : ℝ) : 
  course_length = 84 →
  speed_ratio = 2 →
  course_length / speed_ratio = (course_length - head_start) / 1 →
  head_start = 42 := by
sorry

end NUMINAMATH_CALUDE_race_head_start_l2194_219440


namespace NUMINAMATH_CALUDE_ice_cream_sundae_combinations_l2194_219491

/-- The number of different kinds of ice cream. -/
def n : ℕ := 8

/-- The number of scoops in a sundae. -/
def k : ℕ := 2

/-- The number of unique two-scoop sundaes with different ice cream flavors. -/
def different_flavors : ℕ := n.choose k

/-- The number of unique two-scoop sundaes with identical ice cream flavors. -/
def identical_flavors : ℕ := n

/-- The total number of unique two-scoop sundaes. -/
def total_sundaes : ℕ := different_flavors + identical_flavors

theorem ice_cream_sundae_combinations :
  total_sundaes = 36 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundae_combinations_l2194_219491


namespace NUMINAMATH_CALUDE_method_of_continued_proportion_is_correct_l2194_219465

-- Define the possible methods
inductive AncientChineseMathMethod
| CircleCutting
| ContinuedProportion
| SuJiushaoAlgorithm
| SunTzuRemainder

-- Define a property for methods that can find GCD
def canFindGCD (method : AncientChineseMathMethod) : Prop := sorry

-- Define a property for methods from Song and Yuan dynasties
def fromSongYuanDynasties (method : AncientChineseMathMethod) : Prop := sorry

-- Define a property for methods comparable to Euclidean algorithm
def comparableToEuclidean (method : AncientChineseMathMethod) : Prop := sorry

-- Theorem stating that the Method of Continued Proportion is the correct answer
theorem method_of_continued_proportion_is_correct :
  ∃ (method : AncientChineseMathMethod),
    method = AncientChineseMathMethod.ContinuedProportion ∧
    canFindGCD method ∧
    fromSongYuanDynasties method ∧
    comparableToEuclidean method ∧
    (∀ (other : AncientChineseMathMethod),
      other ≠ AncientChineseMathMethod.ContinuedProportion →
      ¬(canFindGCD other ∧ fromSongYuanDynasties other ∧ comparableToEuclidean other)) :=
sorry

end NUMINAMATH_CALUDE_method_of_continued_proportion_is_correct_l2194_219465


namespace NUMINAMATH_CALUDE_triangle_cosA_value_l2194_219442

theorem triangle_cosA_value (a b c : ℝ) (A B C : ℝ) (S : ℝ) 
  (h1 : (a^2 + b^2) * Real.tan C = 8 * S)
  (h2 : Real.sin A * Real.cos B = 2 * Real.cos A * Real.sin B)
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π)
  (h8 : S = (1/2) * a * b * Real.sin C) :
  Real.cos A = Real.sqrt 30 / 15 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosA_value_l2194_219442


namespace NUMINAMATH_CALUDE_min_value_fraction_l2194_219458

theorem min_value_fraction (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (∀ y, 0 < y ∧ y < 1 → (1 / (4 * x) + 4 / (1 - x)) ≤ (1 / (4 * y) + 4 / (1 - y))) →
  1 / (4 * x) + 4 / (1 - x) = 25 / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2194_219458


namespace NUMINAMATH_CALUDE_cubic_value_l2194_219406

theorem cubic_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2010 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_cubic_value_l2194_219406


namespace NUMINAMATH_CALUDE_inequality_proof_l2194_219444

theorem inequality_proof (a x : ℝ) (h1 : 0 < x) (h2 : x < a) :
  (a - x)^6 - 3*a*(a - x)^5 + 5/2 * a^2 * (a - x)^4 - 1/2 * a^4 * (a - x)^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2194_219444


namespace NUMINAMATH_CALUDE_right_triangle_area_l2194_219475

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) (h4 : a ≤ b) : (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2194_219475


namespace NUMINAMATH_CALUDE_borya_segments_imply_isosceles_l2194_219407

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A segment represented by its length. -/
structure Segment where
  length : ℝ

/-- The set of nine segments drawn by Borya. -/
def BoryaSegments : Set Segment := sorry

/-- The three altitudes of the triangle. -/
def altitudes (t : Triangle) : Set Segment := sorry

/-- The three angle bisectors of the triangle. -/
def angleBisectors (t : Triangle) : Set Segment := sorry

/-- The three medians of the triangle. -/
def medians (t : Triangle) : Set Segment := sorry

/-- Predicate to check if a triangle is isosceles. -/
def isIsosceles (t : Triangle) : Prop := sorry

theorem borya_segments_imply_isosceles (t : Triangle) 
  (h1 : ∃ s1 s2 s3 : Segment, s1 ∈ BoryaSegments ∧ s2 ∈ BoryaSegments ∧ s3 ∈ BoryaSegments ∧ 
                              {s1, s2, s3} = altitudes t)
  (h2 : ∃ s1 s2 s3 : Segment, s1 ∈ BoryaSegments ∧ s2 ∈ BoryaSegments ∧ s3 ∈ BoryaSegments ∧ 
                              {s1, s2, s3} = angleBisectors t)
  (h3 : ∃ s1 s2 s3 : Segment, s1 ∈ BoryaSegments ∧ s2 ∈ BoryaSegments ∧ s3 ∈ BoryaSegments ∧ 
                              {s1, s2, s3} = medians t)
  (h4 : ∀ s ∈ BoryaSegments, ∃ s' ∈ BoryaSegments, s ≠ s' ∧ s.length = s'.length) :
  isIsosceles t :=
sorry

end NUMINAMATH_CALUDE_borya_segments_imply_isosceles_l2194_219407


namespace NUMINAMATH_CALUDE_max_revenue_l2194_219484

def price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def revenue (t : ℕ) : ℝ :=
  price t * sales_volume t

theorem max_revenue :
  ∃ (t : ℕ), t = 25 ∧ revenue t = 1125 ∧
  ∀ (s : ℕ), 0 < s ∧ s ≤ 30 → revenue s ≤ revenue t := by
  sorry

end NUMINAMATH_CALUDE_max_revenue_l2194_219484


namespace NUMINAMATH_CALUDE_jills_total_earnings_l2194_219410

/-- Calculates Jill's earnings over three months based on specific working conditions. -/
def jills_earnings (days_per_month : ℕ) (first_month_rate : ℕ) : ℕ :=
  let second_month_rate := 2 * first_month_rate
  let first_month := days_per_month * first_month_rate
  let second_month := days_per_month * second_month_rate
  let third_month := (days_per_month / 2) * second_month_rate
  first_month + second_month + third_month

/-- Theorem stating that Jill's earnings over three months equal $1,200 -/
theorem jills_total_earnings : 
  jills_earnings 30 10 = 1200 := by
  sorry

#eval jills_earnings 30 10

end NUMINAMATH_CALUDE_jills_total_earnings_l2194_219410


namespace NUMINAMATH_CALUDE_fourth_pillar_height_17_l2194_219460

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space using the general form Ax + By + Cz = D -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the height of the fourth pillar in a square arrangement -/
def calculateFourthPillarHeight (a b c : ℝ) : ℝ :=
  sorry

theorem fourth_pillar_height_17 :
  calculateFourthPillarHeight 15 10 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_fourth_pillar_height_17_l2194_219460


namespace NUMINAMATH_CALUDE_georges_earnings_l2194_219471

-- Define the daily wages and hours worked
def monday_wage : ℝ := 5
def monday_hours : ℝ := 7
def tuesday_wage : ℝ := 6
def tuesday_hours : ℝ := 2
def wednesday_wage : ℝ := 4
def wednesday_hours : ℝ := 5
def saturday_wage : ℝ := 7
def saturday_hours : ℝ := 3

-- Define the tax rate and uniform fee
def tax_rate : ℝ := 0.1
def uniform_fee : ℝ := 15

-- Calculate total earnings before deductions
def total_earnings : ℝ := 
  monday_wage * monday_hours + 
  tuesday_wage * tuesday_hours + 
  wednesday_wage * wednesday_hours + 
  saturday_wage * saturday_hours

-- Calculate earnings after tax deduction
def earnings_after_tax : ℝ := total_earnings * (1 - tax_rate)

-- Calculate final earnings after uniform fee deduction
def final_earnings : ℝ := earnings_after_tax - uniform_fee

-- Theorem statement
theorem georges_earnings : final_earnings = 64.2 := by
  sorry

end NUMINAMATH_CALUDE_georges_earnings_l2194_219471


namespace NUMINAMATH_CALUDE_second_friend_is_nina_l2194_219421

structure Friend where
  hasChild : Bool
  name : String
  childName : String

def isNinotchka (name : String) : Bool :=
  name = "Nina" || name = "Ninotchka"

theorem second_friend_is_nina (friend1 friend2 : Friend) :
  friend2.hasChild = true →
  friend2.childName = friend2.name →
  isNinotchka friend2.childName →
  friend2.name = "Nina" :=
by
  sorry

end NUMINAMATH_CALUDE_second_friend_is_nina_l2194_219421


namespace NUMINAMATH_CALUDE_lcm_gcd_difference_nineteen_l2194_219420

theorem lcm_gcd_difference_nineteen (a b : ℕ+) :
  Nat.lcm a b - Nat.gcd a b = 19 →
  ((a = 1 ∧ b = 20) ∨ (a = 20 ∧ b = 1)) ∨
  ((a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4)) ∨
  ((a = 19 ∧ b = 38) ∨ (a = 38 ∧ b = 19)) :=
by sorry

end NUMINAMATH_CALUDE_lcm_gcd_difference_nineteen_l2194_219420


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2194_219486

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_negative_2023 :
  reciprocal (-2023) = -1/2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2194_219486


namespace NUMINAMATH_CALUDE_monkey_reaches_top_l2194_219426

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

end NUMINAMATH_CALUDE_monkey_reaches_top_l2194_219426


namespace NUMINAMATH_CALUDE_greatest_negative_root_of_sine_cosine_equation_l2194_219432

theorem greatest_negative_root_of_sine_cosine_equation :
  let α : ℝ := Real.arctan (1 / 8)
  let β : ℝ := Real.arctan (4 / 7)
  let root : ℝ := (α + β - 2 * Real.pi) / 9
  (∀ x : ℝ, x < 0 → Real.sin x + 8 * Real.cos x = 4 * Real.sin (8 * x) + 7 * Real.cos (8 * x) → x ≤ root) ∧
  Real.sin root + 8 * Real.cos root = 4 * Real.sin (8 * root) + 7 * Real.cos (8 * root) ∧
  root < 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_negative_root_of_sine_cosine_equation_l2194_219432


namespace NUMINAMATH_CALUDE_sally_coin_problem_l2194_219418

/-- Represents Sally's coin collection --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ

/-- Represents the changes in Sally's coin collection --/
def update_collection (initial : CoinCollection) (dad_nickels mom_nickels : ℕ) : CoinCollection :=
  { pennies := initial.pennies,
    nickels := initial.nickels + dad_nickels + mom_nickels }

theorem sally_coin_problem (initial : CoinCollection) (dad_nickels mom_nickels : ℕ) 
  (h1 : initial.nickels = 7)
  (h2 : dad_nickels = 9)
  (h3 : mom_nickels = 2)
  (h4 : (update_collection initial dad_nickels mom_nickels).nickels = 18) :
  initial.nickels = 7 ∧ ∀ (p : ℕ), ∃ (initial' : CoinCollection), 
    initial'.pennies = p ∧ 
    initial'.nickels = initial.nickels ∧
    (update_collection initial' dad_nickels mom_nickels).nickels = 18 :=
sorry

end NUMINAMATH_CALUDE_sally_coin_problem_l2194_219418


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2194_219489

/-- Proves that a rectangle with width w and length 3w, whose perimeter is twice its area, has width 4/3 and length 4 -/
theorem rectangle_dimensions (w : ℝ) (h1 : w > 0) : 
  (2 * (w + 3*w) = 2 * (w * 3*w)) → w = 4/3 ∧ 3*w = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2194_219489


namespace NUMINAMATH_CALUDE_egg_supply_solution_l2194_219423

/-- Represents the egg supply problem for Mark's farm --/
def egg_supply_problem (daily_supply_store1 : ℕ) (weekly_total : ℕ) : Prop :=
  ∃ (daily_supply_store2 : ℕ),
    daily_supply_store1 = 5 * 12 ∧
    weekly_total = 7 * (daily_supply_store1 + daily_supply_store2) ∧
    daily_supply_store2 = 30

/-- Theorem stating the solution to the egg supply problem --/
theorem egg_supply_solution : 
  egg_supply_problem 60 630 := by
  sorry

end NUMINAMATH_CALUDE_egg_supply_solution_l2194_219423


namespace NUMINAMATH_CALUDE_triangle_extension_similarity_l2194_219419

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the similarity of triangles
def SimilarTriangles (t1 t2 : Triangle) : Prop := sorry

-- Define the extension of a line segment
def ExtendSegment (p1 p2 : ℝ × ℝ) (t : ℝ) : ℝ × ℝ := sorry

-- Define the length of a line segment
def SegmentLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem triangle_extension_similarity (ABC : Triangle) (P : ℝ × ℝ) :
  SegmentLength ABC.A ABC.B = 10 →
  SegmentLength ABC.B ABC.C = 9 →
  SegmentLength ABC.C ABC.A = 7 →
  P = ExtendSegment ABC.B ABC.C 1 →
  SimilarTriangles ⟨P, ABC.A, ABC.B⟩ ⟨P, ABC.C, ABC.A⟩ →
  SegmentLength P ABC.C = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_extension_similarity_l2194_219419


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2194_219466

theorem imaginary_part_of_z (z : ℂ) : z = Complex.I * (1 + Complex.I) → Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2194_219466


namespace NUMINAMATH_CALUDE_part_one_solution_part_two_solution_l2194_219429

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

end NUMINAMATH_CALUDE_part_one_solution_part_two_solution_l2194_219429


namespace NUMINAMATH_CALUDE_sock_pair_count_l2194_219447

/-- Given 8 pairs of socks, calculates the number of different pairs that can be formed
    by selecting 2 socks that are not from the same original pair -/
def sockPairs (totalPairs : Nat) : Nat :=
  let totalSocks := 2 * totalPairs
  let firstChoice := totalSocks
  let secondChoice := totalSocks - 2
  (firstChoice * secondChoice) / 2

/-- Theorem stating that with 8 pairs of socks, the number of different pairs
    that can be formed by selecting 2 socks not from the same original pair is 112 -/
theorem sock_pair_count : sockPairs 8 = 112 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l2194_219447


namespace NUMINAMATH_CALUDE_f_f_two_equals_two_l2194_219425

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2/x

theorem f_f_two_equals_two : f (f 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_f_two_equals_two_l2194_219425


namespace NUMINAMATH_CALUDE_find_unknown_number_l2194_219438

theorem find_unknown_number (known_numbers : List ℕ) (average : ℕ) : 
  known_numbers = [55, 48, 507, 2, 42] → 
  average = 223 → 
  ∃ x : ℕ, (List.sum known_numbers + x) / 6 = average ∧ x = 684 :=
by sorry

end NUMINAMATH_CALUDE_find_unknown_number_l2194_219438


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l2194_219402

/-- The area of a square with perimeter 48 cm is 144 cm² -/
theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 48 → area = (perimeter / 4) ^ 2 → area = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l2194_219402


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l2194_219434

/-- Represents the stratified sampling problem -/
structure StratifiedSample where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  sample_size : ℕ
  interview_size : ℕ

/-- Calculates the number of male students in the sample -/
def male_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.male_students) / s.total_students

/-- Calculates the number of female students in the sample -/
def female_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.female_students) / s.total_students

/-- Calculates the probability of selecting exactly one female student for interview -/
def prob_one_female (s : StratifiedSample) : ℚ :=
  let male_count := male_in_sample s
  let female_count := female_in_sample s
  (male_count * female_count : ℚ) / ((s.sample_size * (s.sample_size - 1)) / 2 : ℚ)

/-- The main theorem to be proved -/
theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.total_students = 50)
  (h2 : s.male_students = 30)
  (h3 : s.female_students = 20)
  (h4 : s.sample_size = 5)
  (h5 : s.interview_size = 2) :
  male_in_sample s = 3 ∧ 
  female_in_sample s = 2 ∧ 
  prob_one_female s = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l2194_219434


namespace NUMINAMATH_CALUDE_f_of_f_of_3_l2194_219408

def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 4

theorem f_of_f_of_3 : f (f 3) = 692 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_of_3_l2194_219408


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2194_219416

theorem gcd_of_specific_numbers : Nat.gcd 333333 7777777 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2194_219416


namespace NUMINAMATH_CALUDE_octagon_triangle_side_ratio_l2194_219449

theorem octagon_triangle_side_ratio : 
  ∀ (s_o s_t : ℝ), s_o > 0 → s_t > 0 →
  (2 * Real.sqrt 2) * s_o^2 = (Real.sqrt 3 / 4) * s_t^2 →
  s_t / s_o = 2 * (2 : ℝ)^(1/4) :=
by
  sorry

end NUMINAMATH_CALUDE_octagon_triangle_side_ratio_l2194_219449


namespace NUMINAMATH_CALUDE_sin_double_angle_for_point_l2194_219436

theorem sin_double_angle_for_point (a : ℝ) (θ : ℝ) (h : a > 0) :
  let P : ℝ × ℝ := (-4 * a, 3 * a)
  (∃ r : ℝ, r > 0 ∧ P.1 = r * Real.cos θ ∧ P.2 = r * Real.sin θ) →
  Real.sin (2 * θ) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_for_point_l2194_219436


namespace NUMINAMATH_CALUDE_system_solution_l2194_219495

theorem system_solution (a b x y : ℝ) 
  (h1 : (x - y) / (1 - x * y) = 2 * a / (1 + a^2))
  (h2 : (x + y) / (1 + x * y) = 2 * b / (1 + b^2))
  (ha : a^2 ≠ 1)
  (hb : b^2 ≠ 1)
  (hab : a ≠ b)
  (hnr : a * b ≠ 1) :
  ((x = (a * b + 1) / (a + b) ∧ y = (a * b - 1) / (a - b)) ∨
   (x = (a + b) / (a * b + 1) ∧ y = (a - b) / (a * b - 1))) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2194_219495


namespace NUMINAMATH_CALUDE_special_function_a_range_l2194_219480

/-- A function satisfying the given properties -/
structure SpecialFunction where
  f : ℝ → ℝ
  even : ∀ x, f (-x) = f x
  increasing_nonneg : ∀ x₁ x₂, 0 ≤ x₁ ∧ 0 ≤ x₂ ∧ x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

/-- The theorem statement -/
theorem special_function_a_range (f : SpecialFunction) :
  {a : ℝ | ∀ x ∈ Set.Icc (1/2 : ℝ) 1, f.f (a * x + 1) ≤ f.f (x - 2)} = Set.Icc (-2 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_a_range_l2194_219480


namespace NUMINAMATH_CALUDE_divisibility_condition_l2194_219424

theorem divisibility_condition (p : Nat) (α : Nat) (x : Int) :
  Prime p → p > 2 → α > 0 →
  (∃ k : Int, x^2 - 1 = k * p^α) ↔
  (∃ t : Int, x = t * p^α + 1 ∨ x = t * p^α - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2194_219424


namespace NUMINAMATH_CALUDE_inequality_proof_l2194_219474

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha₁ : 0 < a₁) (ha₂ : 0 < a₂) (ha₃ : 0 < a₃) 
  (hb₁ : 0 < b₁) (hb₂ : 0 < b₂) (hb₃ : 0 < b₃) : 
  (a₁*b₂ + a₂*b₁ + a₂*b₃ + a₃*b₂ + a₃*b₁ + a₁*b₃)^2 ≥ 
  4*(a₁*a₂ + a₂*a₃ + a₃*a₁)*(b₁*b₂ + b₂*b₃ + b₃*b₁) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2194_219474


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2194_219499

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : M ∩ (U \ N) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2194_219499


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2194_219467

theorem rationalize_denominator :
  ∃ (A B C D E : ℚ),
    (3 : ℚ) / (4 * Real.sqrt 7 + 5 * Real.sqrt 3) = (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = 12 ∧
    B = 7 ∧
    C = -15 ∧
    D = 3 ∧
    E = 37 ∧
    (∀ k : ℚ, k ≠ 0 → (k * A * Real.sqrt B + k * C * Real.sqrt D) / (k * E) = (A * Real.sqrt B + C * Real.sqrt D) / E) ∧
    (∀ n : ℕ, n > 1 → ¬(∃ m : ℕ, B = m^2 * n)) ∧
    (∀ n : ℕ, n > 1 → ¬(∃ m : ℕ, D = m^2 * n)) :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2194_219467


namespace NUMINAMATH_CALUDE_roxy_initial_flowering_plants_l2194_219422

/-- The initial number of flowering plants in Roxy's garden -/
def initial_flowering_plants : ℕ := 7

/-- The initial number of fruiting plants in Roxy's garden -/
def initial_fruiting_plants : ℕ := 2 * initial_flowering_plants

/-- The number of flowering plants bought on Saturday -/
def flowering_plants_bought : ℕ := 3

/-- The number of fruiting plants bought on Saturday -/
def fruiting_plants_bought : ℕ := 2

/-- The number of flowering plants given away on Sunday -/
def flowering_plants_given : ℕ := 1

/-- The number of fruiting plants given away on Sunday -/
def fruiting_plants_given : ℕ := 4

/-- The total number of plants remaining after all transactions -/
def total_plants_remaining : ℕ := 21

theorem roxy_initial_flowering_plants :
  (initial_flowering_plants + flowering_plants_bought - flowering_plants_given) +
  (initial_fruiting_plants + fruiting_plants_bought - fruiting_plants_given) =
  total_plants_remaining :=
by sorry

end NUMINAMATH_CALUDE_roxy_initial_flowering_plants_l2194_219422


namespace NUMINAMATH_CALUDE_max_product_constraint_l2194_219428

theorem max_product_constraint (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a + b = 4) :
  (a - 1) * (b - 1) ≤ 1 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 1 ∧ b₀ > 1 ∧ a₀ + b₀ = 4 ∧ (a₀ - 1) * (b₀ - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_product_constraint_l2194_219428


namespace NUMINAMATH_CALUDE_range_of_a_l2194_219492

/-- Given a system of equations and an inequality, prove the range of values for a. -/
theorem range_of_a (a x y : ℝ) 
  (eq1 : x + y = 3 * a + 4)
  (eq2 : x - y = 7 * a - 4)
  (ineq : 3 * x - 2 * y < 11) :
  a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2194_219492


namespace NUMINAMATH_CALUDE_donut_sharing_l2194_219439

def total_donuts (delta_donuts : ℕ) (gamma_donuts : ℕ) (beta_multiplier : ℕ) : ℕ :=
  delta_donuts + gamma_donuts + (beta_multiplier * gamma_donuts)

theorem donut_sharing :
  let delta_donuts : ℕ := 8
  let gamma_donuts : ℕ := 8
  let beta_multiplier : ℕ := 3
  total_donuts delta_donuts gamma_donuts beta_multiplier = 40 := by
  sorry

end NUMINAMATH_CALUDE_donut_sharing_l2194_219439


namespace NUMINAMATH_CALUDE_only_negative_number_l2194_219464

theorem only_negative_number (a b c d : ℝ) : 
  a = 2023 ∧ b = -2023 ∧ c = 1/2023 ∧ d = 0 →
  (b < 0 ∧ a ≥ 0 ∧ c > 0 ∧ d = 0) := by
  sorry

end NUMINAMATH_CALUDE_only_negative_number_l2194_219464


namespace NUMINAMATH_CALUDE_complex_fraction_sum_complex_equation_solution_l2194_219403

-- Define the complex number i
def i : ℂ := Complex.I

-- Problem 1
theorem complex_fraction_sum : 
  (1 + i)^2 / (1 + 2*i) + (1 - i)^2 / (2 - i) = 6/5 - 2/5 * i := by sorry

-- Problem 2
theorem complex_equation_solution (x y : ℝ) :
  x / (1 + i) + y / (1 + 2*i) = 10 / (1 + 3*i) → x = -2 ∧ y = 10 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_complex_equation_solution_l2194_219403


namespace NUMINAMATH_CALUDE_distance_between_points_l2194_219401

theorem distance_between_points : 
  let pointA : ℝ × ℝ := (1, -3)
  let pointB : ℝ × ℝ := (4, 6)
  Real.sqrt ((pointB.1 - pointA.1)^2 + (pointB.2 - pointA.2)^2) = Real.sqrt 90 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2194_219401


namespace NUMINAMATH_CALUDE_workshop_workers_count_l2194_219473

/-- Proves that the total number of workers in a workshop is 14 given specific salary conditions -/
theorem workshop_workers_count : ∀ (W : ℕ) (N : ℕ),
  -- Average salary of all workers is 9000
  W * 9000 = 7 * 12000 + N * 6000 →
  -- Total workers is sum of technicians and non-technicians
  W = 7 + N →
  -- Conclusion: Total number of workers is 14
  W = 14 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l2194_219473


namespace NUMINAMATH_CALUDE_min_value_a_plus_5b_l2194_219430

theorem min_value_a_plus_5b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b + b^2 = b + 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x * y + y^2 = y + 1 → a + 5 * b ≤ x + 5 * y ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x * y + y^2 = y + 1 ∧ x + 5 * y = 7/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_5b_l2194_219430


namespace NUMINAMATH_CALUDE_johann_delivery_correct_l2194_219463

/-- The number of pieces Johann needs to deliver -/
def johann_delivery (total friend1 friend2 friend3 friend4 : ℕ) : ℕ :=
  total - (friend1 + friend2 + friend3 + friend4)

/-- Theorem stating that Johann's delivery is correct -/
theorem johann_delivery_correct (total friend1 friend2 friend3 friend4 : ℕ) 
  (h_total : total = 250)
  (h_friend1 : friend1 = 35)
  (h_friend2 : friend2 = 42)
  (h_friend3 : friend3 = 38)
  (h_friend4 : friend4 = 45) :
  johann_delivery total friend1 friend2 friend3 friend4 = 90 := by
  sorry

#eval johann_delivery 250 35 42 38 45

end NUMINAMATH_CALUDE_johann_delivery_correct_l2194_219463


namespace NUMINAMATH_CALUDE_penny_sock_cost_l2194_219493

/-- Given Penny's shopping scenario, prove the cost of each pair of socks. -/
theorem penny_sock_cost (initial_amount : ℚ) (num_sock_pairs : ℕ) (hat_cost remaining_amount : ℚ) :
  initial_amount = 20 →
  num_sock_pairs = 4 →
  hat_cost = 7 →
  remaining_amount = 5 →
  ∃ (sock_cost : ℚ), 
    initial_amount - hat_cost - (num_sock_pairs : ℚ) * sock_cost = remaining_amount ∧
    sock_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_penny_sock_cost_l2194_219493


namespace NUMINAMATH_CALUDE_inequality_proof_l2194_219443

theorem inequality_proof (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) : 
  1 / (a + b) < 1 / (a * b) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2194_219443


namespace NUMINAMATH_CALUDE_original_earnings_before_raise_l2194_219485

theorem original_earnings_before_raise (new_earnings : ℝ) (percent_increase : ℝ) 
  (h1 : new_earnings = 80)
  (h2 : percent_increase = 60) :
  let original_earnings := new_earnings / (1 + percent_increase / 100)
  original_earnings = 50 := by
sorry

end NUMINAMATH_CALUDE_original_earnings_before_raise_l2194_219485


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2194_219455

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, x^2 - 4*x + (k - 1) = 0) ↔ k ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2194_219455


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_determination_l2194_219451

/-- A geometric sequence is defined by its first term and common ratio -/
structure GeometricSequence where
  first_term : ℝ
  common_ratio : ℝ

/-- The nth term of a geometric sequence -/
def nth_term (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem geometric_sequence_first_term_determination 
  (seq : GeometricSequence) 
  (h5 : nth_term seq 5 = 72)
  (h8 : nth_term seq 8 = 576) : 
  seq.first_term = 4.5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_determination_l2194_219451


namespace NUMINAMATH_CALUDE_father_picked_22_8_pounds_l2194_219482

/-- Represents the amount of strawberries picked by each person in pounds -/
structure StrawberryPicking where
  marco : ℝ
  sister : ℝ
  father : ℝ

/-- Converts kilograms to pounds -/
def kg_to_pounds (kg : ℝ) : ℝ := kg * 2.2

/-- Calculates the amount of strawberries picked by each person -/
def strawberry_picking : StrawberryPicking :=
  let marco_pounds := 1 + kg_to_pounds 3
  let sister_pounds := 1.5 * marco_pounds
  let father_pounds := 2 * sister_pounds
  { marco := marco_pounds,
    sister := sister_pounds,
    father := father_pounds }

/-- Theorem stating that the father picked 22.8 pounds of strawberries -/
theorem father_picked_22_8_pounds :
  strawberry_picking.father = 22.8 := by
  sorry

end NUMINAMATH_CALUDE_father_picked_22_8_pounds_l2194_219482
