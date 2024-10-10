import Mathlib

namespace no_base_ends_with_one_l1848_184877

theorem no_base_ends_with_one : 
  ∀ b : ℕ, 3 ≤ b ∧ b ≤ 10 → ¬(842 % b = 1) := by
  sorry

end no_base_ends_with_one_l1848_184877


namespace quadratic_through_origin_l1848_184865

theorem quadratic_through_origin (a : ℝ) : 
  (∀ x y : ℝ, y = (a - 1) * x^2 - x + a^2 - 1 → (x = 0 → y = 0)) → 
  a = -1 :=
sorry

end quadratic_through_origin_l1848_184865


namespace one_root_condition_l1848_184864

theorem one_root_condition (k : ℝ) : 
  (∃! x : ℝ, Real.log (k * x) = 2 * Real.log (x + 1)) → (k < 0 ∨ k = 4) :=
sorry

end one_root_condition_l1848_184864


namespace shooting_probabilities_l1848_184863

/-- Probability of hitting a specific ring in one shot -/
def ring_probability : Fin 3 → ℝ
| 0 => 0.13  -- 10-ring
| 1 => 0.28  -- 9-ring
| 2 => 0.31  -- 8-ring

/-- The sum of probabilities for 10-ring and 9-ring -/
def prob_10_or_9 : ℝ := ring_probability 0 + ring_probability 1

/-- The probability of hitting less than 9 rings -/
def prob_less_than_9 : ℝ := 1 - prob_10_or_9

theorem shooting_probabilities :
  prob_10_or_9 = 0.41 ∧ prob_less_than_9 = 0.59 := by sorry

end shooting_probabilities_l1848_184863


namespace proposition_false_implies_a_equals_one_l1848_184855

theorem proposition_false_implies_a_equals_one (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (1 - a) * x < 0) → a = 1 := by
  sorry

end proposition_false_implies_a_equals_one_l1848_184855


namespace equilateral_triangle_perimeter_l1848_184825

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_perimeter_l1848_184825


namespace third_to_first_l1848_184841

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Definition of a point being in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Theorem: If P is in the third quadrant, then Q(-a, -b) is in the first quadrant -/
theorem third_to_first (P : Point) (hP : isInThirdQuadrant P) :
  let Q : Point := ⟨-P.x, -P.y⟩
  isInFirstQuadrant Q := by
  sorry

end third_to_first_l1848_184841


namespace semicircle_radius_l1848_184836

theorem semicircle_radius (x y z : ℝ) (h_right_angle : x^2 + y^2 = z^2)
  (h_xy_area : π * x^2 / 2 = 12 * π) (h_xz_arc : π * y = 10 * π) :
  z / 2 = 2 * Real.sqrt 31 := by
  sorry

end semicircle_radius_l1848_184836


namespace next_integer_divisibility_l1848_184848

theorem next_integer_divisibility (n : ℕ) :
  ∃ k : ℤ, (k : ℝ) = ⌊(Real.sqrt 3 + 1)^(2*n)⌋ + 1 ∧ (2^(n+1) : ℤ) ∣ k :=
sorry

end next_integer_divisibility_l1848_184848


namespace last_locker_theorem_l1848_184884

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the direction the student is walking -/
inductive Direction
| Forward
| Backward

/-- 
Simulates the student's locker-opening process and returns the number of the last locker opened.
n: The total number of lockers
-/
def lastLockerOpened (n : Nat) : Nat :=
  sorry

/-- The main theorem stating that for 1024 lockers, the last one opened is number 854 -/
theorem last_locker_theorem : lastLockerOpened 1024 = 854 := by
  sorry

end last_locker_theorem_l1848_184884


namespace only_23_is_prime_l1848_184869

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 0 → d ∣ n → d = 1 ∨ d = n

theorem only_23_is_prime :
  isPrime 23 ∧
  ¬isPrime 20 ∧
  ¬isPrime 21 ∧
  ¬isPrime 25 ∧
  ¬isPrime 27 :=
by
  sorry

end only_23_is_prime_l1848_184869


namespace supplements_calculation_l1848_184876

/-- The number of boxes of supplements delivered by Mr. Anderson -/
def boxes_of_supplements : ℕ := 760 - 472

/-- The total number of boxes of medicine delivered -/
def total_boxes : ℕ := 760

/-- The number of boxes of vitamins delivered -/
def vitamin_boxes : ℕ := 472

theorem supplements_calculation :
  boxes_of_supplements = 288 :=
by sorry

end supplements_calculation_l1848_184876


namespace sum_of_f_evaluations_l1848_184833

/-- The operation f defined for rational numbers -/
def f (a b c : ℚ) : ℚ := a^2 + 2*b*c

/-- Theorem stating the sum of specific f evaluations -/
theorem sum_of_f_evaluations : 
  f 1 23 76 + f 23 76 1 + f 76 1 23 = 10000 := by
  sorry

end sum_of_f_evaluations_l1848_184833


namespace sequence_problem_l1848_184819

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n ≠ 0) →
  a 3 - (a 7)^2 / 2 + a 11 = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 1 * b 13 = 16 := by
sorry

end sequence_problem_l1848_184819


namespace min_value_trig_expression_min_value_trig_expression_achievable_l1848_184837

theorem min_value_trig_expression (α β : ℝ) :
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 ≥ 100 :=
by sorry

theorem min_value_trig_expression_achievable :
  ∃ α β : ℝ, (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2 = 100 :=
by sorry

end min_value_trig_expression_min_value_trig_expression_achievable_l1848_184837


namespace dani_pants_after_five_years_l1848_184899

/-- Calculates the total number of pants after a given number of years -/
def totalPantsAfterYears (initialPants : ℕ) (pairsPerYear : ℕ) (pantsPerPair : ℕ) (years : ℕ) : ℕ :=
  initialPants + years * pairsPerYear * pantsPerPair

/-- Theorem: Given the initial conditions, Dani will have 90 pants after 5 years -/
theorem dani_pants_after_five_years :
  totalPantsAfterYears 50 4 2 5 = 90 := by
  sorry

#eval totalPantsAfterYears 50 4 2 5

end dani_pants_after_five_years_l1848_184899


namespace burger_cost_is_100_l1848_184814

/-- The cost of a burger in cents -/
def burger_cost : ℕ := 100

/-- The cost of a soda in cents -/
def soda_cost : ℕ := 50

/-- Charles' purchase -/
def charles_purchase (b s : ℕ) : Prop := 4 * b + 3 * s = 550

/-- Alice's purchase -/
def alice_purchase (b s : ℕ) : Prop := 3 * b + 2 * s = 400

/-- Bill's purchase -/
def bill_purchase (b s : ℕ) : Prop := 2 * b + s = 250

theorem burger_cost_is_100 :
  charles_purchase burger_cost soda_cost ∧
  alice_purchase burger_cost soda_cost ∧
  bill_purchase burger_cost soda_cost ∧
  burger_cost = 100 :=
sorry

end burger_cost_is_100_l1848_184814


namespace cleanup_solution_l1848_184835

/-- The time spent cleaning up eggs and toilet paper -/
def cleanup_problem (time_per_roll : ℕ) (total_time : ℕ) (num_eggs : ℕ) (num_rolls : ℕ) : Prop :=
  ∃ (time_per_egg : ℕ),
    time_per_egg * num_eggs + time_per_roll * num_rolls * 60 = total_time * 60 ∧
    time_per_egg = 15

/-- Theorem stating the solution to the cleanup problem -/
theorem cleanup_solution :
  cleanup_problem 30 225 60 7 := by
  sorry

end cleanup_solution_l1848_184835


namespace triangle_inequality_l1848_184853

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l1848_184853


namespace shirts_cost_after_discount_l1848_184849

def first_shirt_cost : ℝ := 15
def price_difference : ℝ := 6
def discount_rate : ℝ := 0.1

def second_shirt_cost : ℝ := first_shirt_cost - price_difference
def total_cost : ℝ := first_shirt_cost + second_shirt_cost
def discounted_cost : ℝ := total_cost * (1 - discount_rate)

theorem shirts_cost_after_discount :
  discounted_cost = 21.60 := by sorry

end shirts_cost_after_discount_l1848_184849


namespace binomial_coefficient_divisibility_l1848_184880

def infinitely_many_n (k : ℤ) : Prop :=
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ ¬((n : ℤ) + k ∣ Nat.choose (2*n) n)

theorem binomial_coefficient_divisibility :
  infinitely_many_n (-1) :=
sorry

end binomial_coefficient_divisibility_l1848_184880


namespace function_properties_l1848_184879

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x ∈ Set.Ioo (-3 : ℝ) 2, f a b x > 0) ∧
    (∀ x ∈ Set.Iic (-3 : ℝ) ∪ Set.Ici 2, f a b x < 0) ∧
    (f a b (-3) = 0 ∧ f a b 2 = 0) →
    (∀ x, f a b x = -3 * x^2 - 3 * x + 18) ∧
    (∀ c : ℝ, (∀ x : ℝ, a * x^2 + b * x + c ≤ 0) ↔ c ≤ -25/12) ∧
    (∃ y_max : ℝ, y_max = -3 ∧
      ∀ x > -1, (f a b x - 21) / (x + 1) ≤ y_max) :=
by sorry

end function_properties_l1848_184879


namespace intersection_of_A_and_B_l1848_184803

def A : Set (ℝ × ℝ) := {p | p.2 = p.1^2}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1}

theorem intersection_of_A_and_B : A ∩ B = {(0, 0), (1, 1)} := by
  sorry

end intersection_of_A_and_B_l1848_184803


namespace total_people_count_l1848_184858

theorem total_people_count (num_students : ℕ) (ratio : ℕ) : 
  num_students = 37500 →
  ratio = 15 →
  num_students + (num_students / ratio) = 40000 := by
sorry

end total_people_count_l1848_184858


namespace largest_invertible_interval_for_f_l1848_184845

/-- The quadratic function f(x) = 3x^2 - 6x - 9 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 9

/-- The theorem stating that [1, ∞) is the largest interval containing x=2 where f is invertible -/
theorem largest_invertible_interval_for_f :
  ∃ (a : ℝ), a = 1 ∧ 
  (∀ x ∈ Set.Ici a, Function.Injective (f ∘ (λ t => t + a))) ∧
  (∀ b < a, ¬ Function.Injective (f ∘ (λ t => t + b))) ∧
  (2 ∈ Set.Ici a) :=
sorry

end largest_invertible_interval_for_f_l1848_184845


namespace skittles_left_l1848_184881

def initial_skittles : ℕ := 250
def reduction_percentage : ℚ := 175 / 1000

theorem skittles_left :
  ⌊(initial_skittles : ℚ) - (initial_skittles : ℚ) * reduction_percentage⌋ = 206 :=
by
  sorry

end skittles_left_l1848_184881


namespace complement_of_M_in_U_l1848_184801

def U : Finset Nat := {1,2,3,4,5,6}
def M : Finset Nat := {1,3,4}

theorem complement_of_M_in_U : 
  (U \ M) = {2,5,6} := by sorry

end complement_of_M_in_U_l1848_184801


namespace calculation_proof_l1848_184895

theorem calculation_proof : 2⁻¹ + Real.sin (30 * π / 180) - (π - 3.14)^0 + abs (-3) - Real.sqrt 9 = 0 := by
  sorry

end calculation_proof_l1848_184895


namespace tan_70_cos_10_expression_l1848_184800

theorem tan_70_cos_10_expression : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end tan_70_cos_10_expression_l1848_184800


namespace prob_four_ones_l1848_184851

/-- The number of sides on a standard die -/
def die_sides : ℕ := 6

/-- The probability of rolling a specific number on a standard die -/
def prob_single_roll : ℚ := 1 / die_sides

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- Theorem: The probability of rolling four 1s on four standard dice is 1/1296 -/
theorem prob_four_ones (die_sides : ℕ) (prob_single_roll : ℚ) (num_dice : ℕ) :
  die_sides = 6 →
  prob_single_roll = 1 / die_sides →
  num_dice = 4 →
  prob_single_roll ^ num_dice = 1 / 1296 := by
  sorry

#check prob_four_ones

end prob_four_ones_l1848_184851


namespace similar_triangle_perimeter_l1848_184873

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def Triangle.isSimilar (t1 t2 : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem similar_triangle_perimeter (t1 t2 : Triangle) :
  t1.isIsosceles ∧
  t1.a = 18 ∧ t1.b = 18 ∧ t1.c = 12 ∧
  t2.isSimilar t1 ∧
  min t2.a (min t2.b t2.c) = 30 →
  t2.perimeter = 120 := by
sorry

end similar_triangle_perimeter_l1848_184873


namespace xy_value_l1848_184861

theorem xy_value (x y : ℝ) (h : |x^3 - 1/8| + Real.sqrt (y - 4) = 0) : x * y = 2 := by
  sorry

end xy_value_l1848_184861


namespace correct_division_result_l1848_184846

theorem correct_division_result (incorrect_divisor incorrect_quotient correct_divisor : ℕ) 
  (h1 : incorrect_divisor = 48)
  (h2 : incorrect_quotient = 24)
  (h3 : correct_divisor = 36) :
  (incorrect_divisor * incorrect_quotient) / correct_divisor = 32 :=
by
  sorry

end correct_division_result_l1848_184846


namespace replaced_man_age_l1848_184829

theorem replaced_man_age
  (total_men : Nat)
  (age_increase : Nat)
  (known_man_age : Nat)
  (women_avg_age : Nat)
  (h1 : total_men = 7)
  (h2 : age_increase = 4)
  (h3 : known_man_age = 26)
  (h4 : women_avg_age = 42) :
  ∃ (replaced_man_age : Nat),
    replaced_man_age = 30 ∧
    (∃ (initial_avg : ℚ),
      (total_men : ℚ) * initial_avg =
        (total_men - 2 : ℚ) * (initial_avg + age_increase) +
        2 * women_avg_age -
        (known_man_age + replaced_man_age : ℚ)) :=
by sorry

end replaced_man_age_l1848_184829


namespace repeating_decimal_equals_expansion_l1848_184894

/-- The repeating decimal 0.73246̅ expressed as a fraction with denominator 999900 -/
def repeating_decimal : ℚ :=
  731514 / 999900

/-- The repeating decimal 0.73246̅ as a real number -/
noncomputable def decimal_expansion : ℝ :=
  0.73 + (246 : ℝ) / 1000 * (1 / (1 - 1/1000))

theorem repeating_decimal_equals_expansion :
  (repeating_decimal : ℝ) = decimal_expansion :=
sorry

end repeating_decimal_equals_expansion_l1848_184894


namespace cube_side_length_l1848_184874

theorem cube_side_length (volume_submerged_min : ℝ) (volume_submerged_max : ℝ)
  (density_ratio : ℝ) (volume_above_min : ℝ) (volume_above_max : ℝ) :
  volume_submerged_min = 0.58 →
  volume_submerged_max = 0.87 →
  density_ratio = 0.95 →
  volume_above_min = 10 →
  volume_above_max = 29 →
  ∃ (s : ℕ), s = 4 ∧
    (volume_submerged_min * s^3 ≤ density_ratio * s^3) ∧
    (density_ratio * s^3 ≤ volume_submerged_max * s^3) ∧
    (volume_above_min ≤ s^3 - volume_submerged_max * s^3) ∧
    (s^3 - volume_submerged_min * s^3 ≤ volume_above_max) :=
by sorry

end cube_side_length_l1848_184874


namespace girls_fraction_l1848_184806

theorem girls_fraction (T G B : ℚ) 
  (h1 : G > 0) 
  (h2 : T > 0) 
  (h3 : ∃ X : ℚ, X * G = (1/5) * T) 
  (h4 : B / G = 7/3) 
  (h5 : T = B + G) : 
  ∃ X : ℚ, X * G = (1/5) * T ∧ X = 2/3 := by
sorry

end girls_fraction_l1848_184806


namespace rotation_result_l1848_184892

-- Define the shapes
inductive Shape
| Triangle
| Circle
| Square

-- Define the position of shapes in the figure
structure Figure :=
(pos1 : Shape)
(pos2 : Shape)
(pos3 : Shape)

-- Define the rotation operation
def rotate120 (f : Figure) : Figure :=
{ pos1 := f.pos3,
  pos2 := f.pos1,
  pos3 := f.pos2 }

-- Theorem statement
theorem rotation_result (f : Figure) 
  (h1 : f.pos1 ≠ f.pos2) 
  (h2 : f.pos2 ≠ f.pos3) 
  (h3 : f.pos3 ≠ f.pos1) : 
  rotate120 f = 
  { pos1 := f.pos3,
    pos2 := f.pos1,
    pos3 := f.pos2 } := by
  sorry

#check rotation_result

end rotation_result_l1848_184892


namespace part_one_part_two_l1848_184886

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| - |x + m|

-- Part 1
theorem part_one :
  ∀ x : ℝ, (f x 2 + 2 < 0) ↔ (x > 1/2) :=
sorry

-- Part 2
theorem part_two :
  (∀ x ∈ Set.Icc 0 2, f x m + |x - 4| > 0) ↔ m ∈ Set.Ioo (-4) 1 :=
sorry

end part_one_part_two_l1848_184886


namespace calculate_expression_l1848_184832

theorem calculate_expression : (3 - Real.pi) ^ 0 + (1/2) ^ (-1 : ℤ) = 3 := by
  sorry

end calculate_expression_l1848_184832


namespace smallest_visible_sum_l1848_184804

/-- Represents a die with 6 faces -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7)

/-- Represents a 4x4x4 cube made of dice -/
def GiantCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- The sum of visible values on the 6 faces of the giant cube -/
def visible_sum (cube : GiantCube) : ℕ :=
  sorry

/-- The theorem stating the smallest possible sum of visible values -/
theorem smallest_visible_sum (cube : GiantCube) :
  visible_sum cube ≥ 144 :=
sorry

end smallest_visible_sum_l1848_184804


namespace ratio_x_to_y_l1848_184807

theorem ratio_x_to_y (x y : ℝ) (h : (8*x - 5*y) / (11*x - 3*y) = 2/7) : 
  x/y = 29/34 := by sorry

end ratio_x_to_y_l1848_184807


namespace triangle_angle_and_max_area_l1848_184850

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and vectors m and n, prove the measure of angle C and the maximum area. -/
theorem triangle_angle_and_max_area 
  (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Triangle angle sum
  m = (Real.sin A, Real.sin B) →           -- Definition of m
  n = (Real.cos B, Real.cos A) →           -- Definition of n
  m.1 * n.1 + m.2 * n.2 = -Real.sin (2 * C) →  -- Dot product condition
  c = 2 * Real.sqrt 3 →                    -- Given value of c
  C = 2 * π / 3 ∧                          -- Angle C
  (∃ (S : ℝ), S ≤ Real.sqrt 3 ∧            -- Maximum area
    ∀ (S' : ℝ), S' = 1/2 * a * b * Real.sin C → S' ≤ S) :=
by sorry

end triangle_angle_and_max_area_l1848_184850


namespace sum_of_quadratic_roots_l1848_184888

theorem sum_of_quadratic_roots (x : ℝ) : 
  x^2 - 17*x + 54 = 0 → ∃ r s : ℝ, r + s = 17 ∧ r * s = 54 := by
  sorry

end sum_of_quadratic_roots_l1848_184888


namespace equation_solution_l1848_184824

theorem equation_solution :
  ∃ x : ℚ, x ≠ 1 ∧ (x^2 - x + 2) / (x - 1) = x + 3 ∧ x = 5/3 := by
  sorry

end equation_solution_l1848_184824


namespace sum_squares_regression_example_l1848_184842

/-- Given a total sum of squared deviations and a correlation coefficient,
    calculate the sum of squares due to regression -/
def sum_squares_regression (total_sum_squared_dev : ℝ) (correlation_coeff : ℝ) : ℝ :=
  total_sum_squared_dev * correlation_coeff^2

/-- Theorem stating that given the specified conditions, 
    the sum of squares due to regression is 72 -/
theorem sum_squares_regression_example :
  sum_squares_regression 120 0.6 = 72 := by
  sorry

end sum_squares_regression_example_l1848_184842


namespace table_area_is_128_l1848_184815

/-- A rectangular table with one side against a wall -/
structure RectangularTable where
  -- Length of the side opposite the wall
  opposite_side : ℝ
  -- Length of each of the other two free sides
  other_side : ℝ
  -- The side opposite the wall is twice the length of each of the other two free sides
  opposite_twice_other : opposite_side = 2 * other_side
  -- The total length of the table's free sides is 32 feet
  total_free_sides : opposite_side + 2 * other_side = 32

/-- The area of the rectangular table is 128 square feet -/
theorem table_area_is_128 (table : RectangularTable) : table.opposite_side * table.other_side = 128 := by
  sorry

end table_area_is_128_l1848_184815


namespace square_divisibility_l1848_184857

theorem square_divisibility (n : ℤ) : ∃ k : ℤ, n^2 = 4*k ∨ n^2 = 4*k + 1 := by
  sorry

end square_divisibility_l1848_184857


namespace candle_height_ratio_time_l1848_184823

/-- Represents a candle with its initial height and burning time. -/
structure Candle where
  initial_height : ℝ
  burning_time : ℝ

/-- The problem setup -/
def candle_problem : Prop :=
  let candle_a : Candle := { initial_height := 12, burning_time := 6 }
  let candle_b : Candle := { initial_height := 15, burning_time := 5 }
  let burn_rate (c : Candle) : ℝ := c.initial_height / c.burning_time
  let height_at_time (c : Candle) (t : ℝ) : ℝ := c.initial_height - (burn_rate c) * t
  ∃ t : ℝ, t > 0 ∧ height_at_time candle_a t = (1/3) * height_at_time candle_b t ∧ t = 7

/-- The theorem to be proved -/
theorem candle_height_ratio_time : candle_problem := by
  sorry

end candle_height_ratio_time_l1848_184823


namespace inequality_theorem_l1848_184889

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  sorry

end inequality_theorem_l1848_184889


namespace total_price_of_hats_l1848_184866

/-- Calculates the total price of hats given the conditions --/
theorem total_price_of_hats :
  let total_hats : ℕ := 85
  let green_hats : ℕ := 30
  let blue_hats : ℕ := total_hats - green_hats
  let price_green : ℕ := 7
  let price_blue : ℕ := 6
  (green_hats * price_green + blue_hats * price_blue) = 540 := by
  sorry

end total_price_of_hats_l1848_184866


namespace least_k_divisible_by_480_l1848_184809

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem least_k_divisible_by_480 :
  ∃ k : ℕ+, (k : ℕ) = 101250 ∧
    is_divisible (k^4) 480 ∧
    ∀ m : ℕ+, m < k → ¬is_divisible (m^4) 480 := by
  sorry

end least_k_divisible_by_480_l1848_184809


namespace conference_attendees_payment_registration_l1848_184839

theorem conference_attendees_payment_registration (
  early_registration : Real) 
  (mid_registration : Real)
  (late_registration : Real)
  (credit_card_percent : Real)
  (debit_card_percent : Real)
  (other_payment_percent : Real) :
  early_registration = 80 →
  mid_registration = 12 →
  late_registration = 100 - early_registration - mid_registration →
  credit_card_percent + debit_card_percent + other_payment_percent = 100 →
  credit_card_percent = 20 →
  debit_card_percent = 60 →
  other_payment_percent = 20 →
  early_registration + mid_registration = 
    (credit_card_percent + debit_card_percent + other_payment_percent) * 
    (early_registration + mid_registration) / 100 :=
by sorry

end conference_attendees_payment_registration_l1848_184839


namespace no_solution_exists_l1848_184854

theorem no_solution_exists : ¬∃ x : ℝ, (16 : ℝ)^(3*x - 1) = (64 : ℝ)^(2*x + 3) := by
  sorry

end no_solution_exists_l1848_184854


namespace room_length_calculation_l1848_184828

/-- Given a rectangular room with width 12 m, surrounded by a 2 m wide veranda on all sides,
    and the area of the veranda being 148 m², the length of the room is 21 m. -/
theorem room_length_calculation (room_width : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_width = 12 →
  veranda_width = 2 →
  veranda_area = 148 →
  ∃ (room_length : ℝ),
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - 
    room_length * room_width = veranda_area ∧
    room_length = 21 :=
by sorry

end room_length_calculation_l1848_184828


namespace permutation_equation_solution_l1848_184812

def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem permutation_equation_solution (x : ℕ) : 
  (3 * A 8 x = 4 * A 9 (x - 1)) → x ≤ 8 → x = 6 := by
  sorry

end permutation_equation_solution_l1848_184812


namespace rectangle_area_21_implies_y_7_l1848_184813

/-- Represents a rectangle EFGH with vertices E(0, 0), F(0, 3), G(y, 3), and H(y, 0) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := 3 * r.y

theorem rectangle_area_21_implies_y_7 (r : Rectangle) (h : area r = 21) : r.y = 7 := by
  sorry

end rectangle_area_21_implies_y_7_l1848_184813


namespace smallest_number_l1848_184860

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def is_smallest (n : Nat) (lst : List Nat) : Prop :=
  ∀ m ∈ lst, n ≤ m

theorem smallest_number :
  let n1 := base_to_decimal [8, 5] 9
  let n2 := base_to_decimal [2, 1, 0] 6
  let n3 := base_to_decimal [1, 0, 0, 0] 4
  let n4 := base_to_decimal [1, 1, 1, 1, 1, 1] 2
  is_smallest n4 [n1, n2, n3, n4] := by
sorry

end smallest_number_l1848_184860


namespace range_of_m_solution_when_m_minimum_l1848_184840

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |5 - 2*x| - |2*x - 1|

-- Theorem for the range of m
theorem range_of_m :
  (∃ x, f m x = 0) → m ∈ Set.Ici 4 :=
sorry

-- Theorem for the solution of the inequality when m is minimum
theorem solution_when_m_minimum :
  let m : ℝ := 4
  ∀ x, |x - 3| + |x + m| ≤ 2*m ↔ x ∈ Set.Icc (-9/2) (7/2) :=
sorry

end range_of_m_solution_when_m_minimum_l1848_184840


namespace rachel_songs_theorem_l1848_184872

/-- The number of songs in each of Rachel's albums -/
def album_songs : List Nat := [5, 6, 8, 10, 12, 14, 16, 7, 9, 11, 13, 15, 17, 4, 6, 8, 10, 12, 14, 3]

/-- The total number of songs Rachel bought -/
def total_songs : Nat := album_songs.sum

theorem rachel_songs_theorem : total_songs = 200 := by
  sorry

end rachel_songs_theorem_l1848_184872


namespace factorization_of_quadratic_l1848_184856

theorem factorization_of_quadratic (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by
  sorry

end factorization_of_quadratic_l1848_184856


namespace quadratic_properties_l1848_184805

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 4

-- Theorem stating the properties of the function
theorem quadratic_properties :
  (∃ (max_value : ℝ), ∀ (x : ℝ), f x ≤ max_value ∧ max_value = 10.75) ∧
  (∃ (max_point : ℝ), max_point > 0 ∧ max_point = 1.5 ∧ ∀ (x : ℝ), f x ≤ f max_point) ∧
  (∀ (x y : ℝ), x > 1.5 → y > x → f y < f x) :=
by sorry

end quadratic_properties_l1848_184805


namespace correct_average_weight_l1848_184871

/-- Given a class of boys with an incorrect average weight due to a misread measurement,
    calculate the correct average weight. -/
theorem correct_average_weight
  (n : ℕ) -- number of boys
  (initial_avg : ℝ) -- initial (incorrect) average weight
  (misread_weight : ℝ) -- weight that was misread
  (correct_weight : ℝ) -- correct weight for the misread value
  (h1 : n = 20) -- there are 20 boys
  (h2 : initial_avg = 58.4) -- initial average was 58.4 kg
  (h3 : misread_weight = 56) -- misread weight was 56 kg
  (h4 : correct_weight = 60) -- correct weight is 60 kg
  : (n * initial_avg + correct_weight - misread_weight) / n = 58.6 := by
  sorry

end correct_average_weight_l1848_184871


namespace rectangle_area_diagonal_l1848_184882

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 4) (h2 : l^2 + w^2 = d^2) :
  l * w = (20 / 41) * d^2 := by sorry

end rectangle_area_diagonal_l1848_184882


namespace base10_to_base7_5423_l1848_184890

/-- Converts a base 10 number to base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Converts a list of digits in base 7 to a natural number --/
def fromBase7 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 7 * acc) 0

theorem base10_to_base7_5423 :
  toBase7 5423 = [5, 4, 5, 1, 2] ∧ fromBase7 [5, 4, 5, 1, 2] = 5423 := by sorry

end base10_to_base7_5423_l1848_184890


namespace water_added_proof_l1848_184843

/-- The amount of water added to a pool given initial and final amounts -/
def water_added (initial : Real) (final : Real) : Real :=
  final - initial

/-- Theorem: Given an initial amount of 1 bucket and a final amount of 9.8 buckets,
    the amount of water added later is 8.8 buckets -/
theorem water_added_proof :
  water_added 1 9.8 = 8.8 := by
  sorry

end water_added_proof_l1848_184843


namespace cafeteria_red_apples_l1848_184862

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := sorry

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 23

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 21

/-- The number of extra apples -/
def extra_apples : ℕ := 35

/-- Theorem stating that the number of red apples ordered is 33 -/
theorem cafeteria_red_apples : 
  red_apples = 33 :=
by sorry

end cafeteria_red_apples_l1848_184862


namespace arctan_sum_special_case_l1848_184811

theorem arctan_sum_special_case : 
  ∀ (a b : ℝ), 
    a = -1/3 → 
    (2*a + 1)*(2*b + 1) = 1 → 
    Real.arctan a + Real.arctan b = π/4 := by
  sorry

end arctan_sum_special_case_l1848_184811


namespace correct_evaluation_l1848_184893

-- Define the expression
def expression : ℤ → ℤ → ℤ → ℤ := λ a b c => a - b * c

-- Define the order of operations
def evaluate_expression (a b c : ℤ) : ℤ :=
  a - (b * c)

-- Theorem statement
theorem correct_evaluation :
  evaluate_expression 65 13 2 = 39 :=
by
  sorry

#eval evaluate_expression 65 13 2

end correct_evaluation_l1848_184893


namespace cosine_value_in_triangle_l1848_184896

theorem cosine_value_in_triangle (a b c : ℝ) (h : 3 * a^2 + 3 * b^2 - 3 * c^2 = 2 * a * b) :
  let cosC := (a^2 + b^2 - c^2) / (2 * a * b)
  cosC = 1/3 := by sorry

end cosine_value_in_triangle_l1848_184896


namespace min_b_over_a_l1848_184878

open Real

/-- The minimum value of b/a given the conditions on f(x) -/
theorem min_b_over_a (f : ℝ → ℝ) (a b : ℝ) (h : ∀ x > 0, f x ≤ 0) :
  (∀ x > 0, f x = log x + (2 * exp 2 - a) * x - b / 2) →
  ∃ m : ℝ, m = -2 / exp 2 ∧ ∀ k : ℝ, (∃ a' b' : ℝ, (∀ x > 0, f x = log x + (2 * exp 2 - a') * x - b' / 2) ∧ b' / a' = k) → k ≥ m :=
by sorry

end min_b_over_a_l1848_184878


namespace hockey_handshakes_l1848_184816

theorem hockey_handshakes (team_size : Nat) (num_teams : Nat) (num_referees : Nat) : 
  team_size = 6 → num_teams = 2 → num_referees = 3 → 
  (team_size * team_size) + (team_size * num_teams * num_referees) = 72 := by
  sorry

end hockey_handshakes_l1848_184816


namespace shortest_distance_l1848_184887

theorem shortest_distance (a b : ℝ) (ha : a = 8) (hb : b = 6) :
  Real.sqrt (a ^ 2 + b ^ 2) = 10 := by
sorry

end shortest_distance_l1848_184887


namespace simple_interest_time_period_l1848_184822

/-- Calculates the time period for a simple interest problem -/
theorem simple_interest_time_period 
  (P : ℝ) (R : ℝ) (A : ℝ) 
  (h_P : P = 1300)
  (h_R : R = 5)
  (h_A : A = 1456) :
  ∃ T : ℝ, T = 2.4 ∧ A = P + (P * R * T / 100) := by
  sorry

end simple_interest_time_period_l1848_184822


namespace green_height_l1848_184827

/-- The heights of the dwarves -/
structure DwarfHeights where
  blue : ℝ
  black : ℝ
  yellow : ℝ
  red : ℝ
  green : ℝ

/-- The conditions of the problem -/
def dwarfProblem (h : DwarfHeights) : Prop :=
  h.blue = 88 ∧
  h.black = 84 ∧
  h.yellow = 76 ∧
  (h.blue + h.black + h.yellow + h.red + h.green) / 5 = 81.6 ∧
  ((h.blue + h.black + h.yellow + h.green) / 4) = ((h.blue + h.black + h.yellow + h.red) / 4 - 6)

theorem green_height (h : DwarfHeights) (hc : dwarfProblem h) : h.green = 68 := by
  sorry

end green_height_l1848_184827


namespace simplify_and_evaluate_l1848_184883

theorem simplify_and_evaluate (a b : ℤ) (ha : a = 2) (hb : b = -1) :
  (a + 3*b)^2 + (a + 3*b)*(a - 3*b) = -4 := by
  sorry

end simplify_and_evaluate_l1848_184883


namespace arithmetic_computation_l1848_184820

theorem arithmetic_computation : 7^2 - 4*5 + 4^3 = 93 := by
  sorry

end arithmetic_computation_l1848_184820


namespace ellipse_eccentricity_l1848_184831

/-- The eccentricity of an ellipse with given properties is 1/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (d₁ d₂ : ℝ → ℝ → ℝ) : 
  a > b ∧ b > 0 →
  (∀ x y, x^2/a^2 + y^2/b^2 = 1 → d₁ x y + d₂ x y = 2*a) →
  (∀ x y, x^2/a^2 + y^2/b^2 = 1 → d₁ x y + d₂ x y = 4*c) →
  c/a = 1/2 := by
sorry

end ellipse_eccentricity_l1848_184831


namespace polynomial_division_remainder_l1848_184808

def f (x : ℝ) : ℝ := x^4 - 3*x^3 + 10*x^2 - 16*x + 5

def g (x k : ℝ) : ℝ := x^2 - x + k

theorem polynomial_division_remainder (k a : ℝ) :
  (∃ q : ℝ → ℝ, ∀ x, f x = g x k * q x + (2*x + a)) ↔ k = 8.5 ∧ a = 9.25 := by
  sorry

end polynomial_division_remainder_l1848_184808


namespace rectangle_longer_side_l1848_184847

/-- Given a rectangle with perimeter 60 feet and area 221 square feet, 
    the length of the longer side is 17 feet. -/
theorem rectangle_longer_side (x y : ℝ) 
  (h_perimeter : 2 * x + 2 * y = 60) 
  (h_area : x * y = 221) 
  (h_longer : x ≥ y) : x = 17 := by
  sorry

end rectangle_longer_side_l1848_184847


namespace box_volume_theorem_l1848_184859

theorem box_volume_theorem : ∃ (x y z : ℕ+), 
  (x : ℚ) / 2 = (y : ℚ) / 5 ∧ (y : ℚ) / 5 = (z : ℚ) / 7 ∧ 
  (x : ℕ) * y * z = 70 := by
  sorry

end box_volume_theorem_l1848_184859


namespace oligarch_wealth_comparison_l1848_184802

/-- Represents the wealth of an oligarch at a given time -/
structure OligarchWealth where
  amount : ℝ
  year : ℕ
  name : String

/-- Represents the national wealth of the country -/
def NationalWealth : Type := ℝ

/-- The problem statement -/
theorem oligarch_wealth_comparison 
  (maximilian_2011 maximilian_2012 alejandro_2011 alejandro_2012 : OligarchWealth)
  (national_wealth : NationalWealth) :
  (alejandro_2012.amount = 2 * maximilian_2011.amount) →
  (maximilian_2012.amount < alejandro_2011.amount) →
  (national_wealth = alejandro_2012.amount + maximilian_2012.amount - alejandro_2011.amount - maximilian_2011.amount) →
  (maximilian_2011.amount > national_wealth) := by
  sorry

end oligarch_wealth_comparison_l1848_184802


namespace recipe_total_cups_l1848_184844

/-- Given a recipe with a butter:flour:sugar ratio of 1:6:4, prove that when 8 cups of sugar are used, the total cups of ingredients is 22. -/
theorem recipe_total_cups (butter flour sugar total : ℚ) : 
  butter / sugar = 1 / 4 →
  flour / sugar = 6 / 4 →
  sugar = 8 →
  total = butter + flour + sugar →
  total = 22 := by
sorry

end recipe_total_cups_l1848_184844


namespace number_puzzle_l1848_184821

theorem number_puzzle (x : ℝ) : (72 / 6 + x = 17) ↔ (x = 5) := by sorry

end number_puzzle_l1848_184821


namespace points_form_circle_l1848_184817

theorem points_form_circle :
  ∀ (x y : ℝ), (∃ t : ℝ, x = Real.cos t ∧ y = Real.sin t) → x^2 + y^2 = 1 :=
by
  sorry

end points_form_circle_l1848_184817


namespace natural_number_pairs_satisfying_equation_and_condition_l1848_184830

theorem natural_number_pairs_satisfying_equation_and_condition :
  ∀ x y : ℕ,
    (2^(10*x + 24*y - 493) + 1 = 9 * 2^(5*x + 12*y - 248) ∧ x + y > 40) ↔
    ((x = 4 ∧ y = 36) ∨ (x = 49 ∧ y = 0) ∨ (x = 37 ∧ y = 7)) :=
by sorry

end natural_number_pairs_satisfying_equation_and_condition_l1848_184830


namespace joey_exam_in_six_weeks_l1848_184852

/-- Joey's SAT exam preparation schedule --/
structure SATPrep where
  weekday_hours : ℕ  -- Hours studied per weekday night
  weekday_nights : ℕ  -- Number of weekday nights studied per week
  weekend_hours : ℕ  -- Hours studied per weekend day
  total_hours : ℕ  -- Total hours to be studied

/-- Calculate the number of weeks until Joey's SAT exam --/
def weeks_until_exam (prep : SATPrep) : ℚ :=
  prep.total_hours / (prep.weekday_hours * prep.weekday_nights + prep.weekend_hours * 2)

/-- Theorem: Joey's SAT exam is 6 weeks away --/
theorem joey_exam_in_six_weeks (prep : SATPrep) 
  (h1 : prep.weekday_hours = 2)
  (h2 : prep.weekday_nights = 5)
  (h3 : prep.weekend_hours = 3)
  (h4 : prep.total_hours = 96) : 
  weeks_until_exam prep = 6 := by
  sorry

end joey_exam_in_six_weeks_l1848_184852


namespace inequality_system_solution_l1848_184867

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x > a ∧ x ≥ 3) ↔ x ≥ 3) → a < 3 := by
  sorry

end inequality_system_solution_l1848_184867


namespace bus_students_problem_l1848_184870

theorem bus_students_problem (initial_students final_students : ℕ) 
  (h1 : initial_students = 28)
  (h2 : final_students = 58) :
  (0.4 : ℝ) * (final_students - initial_students) = 12 := by
  sorry

end bus_students_problem_l1848_184870


namespace axis_of_symmetry_translated_trig_l1848_184875

/-- The axis of symmetry of a translated trigonometric function -/
theorem axis_of_symmetry_translated_trig (k : ℤ) :
  let f (x : ℝ) := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)
  let g (x : ℝ) := f (x + π / 6)
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), 
    g x = A * Real.sin (B * x + C) ∧
    (x = k * π / 2 - π / 12) → (B * x + C = n * π + π / 2) :=
by sorry

end axis_of_symmetry_translated_trig_l1848_184875


namespace regular_pentagon_ratio_sum_l1848_184891

/-- For a regular pentagon with side length a and diagonal length b, (a/b + b/a) = √5 -/
theorem regular_pentagon_ratio_sum (a b : ℝ) (h : a / b = (Real.sqrt 5 - 1) / 2) :
  a / b + b / a = Real.sqrt 5 := by
  sorry

end regular_pentagon_ratio_sum_l1848_184891


namespace tangent_perpendicular_condition_l1848_184885

/-- The function f(x) = x³ - x² + ax + b -/
def f (a b x : ℝ) : ℝ := x^3 - x^2 + a*x + b

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_perpendicular_condition (a b : ℝ) : 
  (f_derivative a 1) * 2 = -1 ↔ a = -3/2 := by sorry

end tangent_perpendicular_condition_l1848_184885


namespace smallest_divisor_divisor_is_four_l1848_184898

theorem smallest_divisor (d : ℕ) : d > 0 ∧ d > 3 ∧
  (∃ n : ℤ, n % d = 1 ∧ (3 * n) % d = 3) →
  d ≥ 4 :=
sorry

theorem divisor_is_four : ∃ d : ℕ, d > 0 ∧ d > 3 ∧
  (∃ n : ℤ, n % d = 1 ∧ (3 * n) % d = 3) ∧
  ∀ k : ℕ, k > 0 ∧ k > 3 ∧ (∃ m : ℤ, m % k = 1 ∧ (3 * m) % k = 3) →
  k ≥ d :=
sorry

end smallest_divisor_divisor_is_four_l1848_184898


namespace julia_tag_game_l1848_184868

/-- 
Given that Julia played tag with a total of 18 kids over two days,
and she played with 14 kids on Tuesday, prove that she played with 4 kids on Monday.
-/
theorem julia_tag_game (total : ℕ) (tuesday : ℕ) (monday : ℕ) 
    (h1 : total = 18) 
    (h2 : tuesday = 14) 
    (h3 : total = monday + tuesday) : 
  monday = 4 := by
  sorry

end julia_tag_game_l1848_184868


namespace total_balloons_l1848_184838

/-- Given an initial number of balloons and an additional number of balloons,
    the total number of balloons is equal to their sum. -/
theorem total_balloons (initial additional : ℕ) :
  initial + additional = (initial + additional) := by sorry

end total_balloons_l1848_184838


namespace square_overlap_percentage_l1848_184826

/-- The percentage of overlap between two squares forming a rectangle -/
theorem square_overlap_percentage (s1 s2 l w : ℝ) (h1 : s1 = 10) (h2 : s2 = 15) 
  (h3 : l = 25) (h4 : w = 20) : 
  (min s1 s2)^2 / (l * w) = 1/5 := by
  sorry

end square_overlap_percentage_l1848_184826


namespace problem_1_problem_2_problem_3_l1848_184810

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define permutation
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem problem_1 : (binomial 100 2 + binomial 100 97) / (permutation 101 3) = 1 / 6 := by
  sorry

theorem problem_2 : (Finset.sum (Finset.range 8) (λ i => binomial (i + 3) 3)) = 330 := by
  sorry

theorem problem_3 (n m : ℕ) (h : m ≤ n) : 
  (binomial (n + 1) m / binomial n m) - (binomial n (n - m + 1) / binomial n (n - m)) = 1 := by
  sorry

end problem_1_problem_2_problem_3_l1848_184810


namespace johns_total_hours_l1848_184818

/-- Represents John's volunteering schedule for the year -/
structure VolunteerSchedule where
  jan_to_mar : Nat  -- Hours for January to March
  apr_to_jun : Nat  -- Hours for April to June
  jul_to_aug : Nat  -- Hours for July and August
  sep_to_oct : Nat  -- Hours for September and October
  november : Nat    -- Hours for November
  december : Nat    -- Hours for December
  bonus_days : Nat  -- Hours for bonus days (third Saturday of every month except May and June)
  charity_run : Nat -- Hours for annual charity run in June

/-- Calculates the total volunteering hours for the year -/
def total_hours (schedule : VolunteerSchedule) : Nat :=
  schedule.jan_to_mar +
  schedule.apr_to_jun +
  schedule.jul_to_aug +
  schedule.sep_to_oct +
  schedule.november +
  schedule.december +
  schedule.bonus_days +
  schedule.charity_run

/-- John's actual volunteering schedule for the year -/
def johns_schedule : VolunteerSchedule :=
  { jan_to_mar := 18
  , apr_to_jun := 24
  , jul_to_aug := 64
  , sep_to_oct := 24
  , november := 6
  , december := 6
  , bonus_days := 40
  , charity_run := 8
  }

/-- Theorem stating that John's total volunteering hours for the year is 190 -/
theorem johns_total_hours : total_hours johns_schedule = 190 := by
  sorry

end johns_total_hours_l1848_184818


namespace waitress_hourly_wage_l1848_184834

/-- Calculates the hourly wage of a waitress given her work hours, tips, and total earnings -/
theorem waitress_hourly_wage 
  (monday_hours tuesday_hours wednesday_hours : ℕ)
  (monday_tips tuesday_tips wednesday_tips : ℚ)
  (total_earnings : ℚ)
  (h1 : monday_hours = 7)
  (h2 : tuesday_hours = 5)
  (h3 : wednesday_hours = 7)
  (h4 : monday_tips = 18)
  (h5 : tuesday_tips = 12)
  (h6 : wednesday_tips = 20)
  (h7 : total_earnings = 240) :
  let total_hours := monday_hours + tuesday_hours + wednesday_hours
  let total_tips := monday_tips + tuesday_tips + wednesday_tips
  let hourly_wage := (total_earnings - total_tips) / total_hours
  hourly_wage = 10 := by
sorry

end waitress_hourly_wage_l1848_184834


namespace kenneth_remaining_money_l1848_184897

def remaining_money (initial_amount baguette_cost water_cost baguette_count water_count : ℕ) : ℕ :=
  initial_amount - (baguette_cost * baguette_count + water_cost * water_count)

theorem kenneth_remaining_money :
  remaining_money 50 2 1 2 2 = 44 := by
  sorry

end kenneth_remaining_money_l1848_184897
