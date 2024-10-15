import Mathlib

namespace NUMINAMATH_CALUDE_multiply_and_add_l389_38923

theorem multiply_and_add : 45 * 21 + 45 * 79 = 4500 := by sorry

end NUMINAMATH_CALUDE_multiply_and_add_l389_38923


namespace NUMINAMATH_CALUDE_prob_green_ball_l389_38957

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def probGreen (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The four containers described in the problem -/
def containerA : Container := ⟨5, 7⟩
def containerB : Container := ⟨7, 3⟩
def containerC : Container := ⟨8, 2⟩
def containerD : Container := ⟨4, 6⟩

/-- The probability of selecting each container -/
def probContainer : ℚ := 1 / 4

/-- Theorem stating the probability of selecting a green ball -/
theorem prob_green_ball : 
  probContainer * probGreen containerA +
  probContainer * probGreen containerB +
  probContainer * probGreen containerC +
  probContainer * probGreen containerD = 101 / 240 := by
  sorry


end NUMINAMATH_CALUDE_prob_green_ball_l389_38957


namespace NUMINAMATH_CALUDE_toy_purchase_cost_l389_38913

theorem toy_purchase_cost (num_toys : ℕ) (cost_per_toy : ℝ) (discount_percent : ℝ) : 
  num_toys = 5 → 
  cost_per_toy = 3 → 
  discount_percent = 20 →
  (num_toys * cost_per_toy) * (1 - discount_percent / 100) = 12 := by
  sorry

end NUMINAMATH_CALUDE_toy_purchase_cost_l389_38913


namespace NUMINAMATH_CALUDE_pup_difference_l389_38981

/-- Represents the number of pups each type of dog has -/
structure PupCounts where
  husky : ℕ
  pitbull : ℕ
  golden : ℕ

/-- Represents the counts of each type of dog -/
structure DogCounts where
  husky : ℕ
  pitbull : ℕ
  golden : ℕ

/-- Calculates the total number of pups -/
def totalPups (counts : DogCounts) (pupCounts : PupCounts) : ℕ :=
  counts.husky * pupCounts.husky + counts.pitbull * pupCounts.pitbull + counts.golden * pupCounts.golden

/-- Calculates the total number of adult dogs -/
def totalAdultDogs (counts : DogCounts) : ℕ :=
  counts.husky + counts.pitbull + counts.golden

theorem pup_difference (counts : DogCounts) (pupCounts : PupCounts) :
  counts.husky = 5 →
  counts.pitbull = 2 →
  counts.golden = 4 →
  pupCounts.husky = 3 →
  pupCounts.pitbull = 3 →
  pupCounts.golden = pupCounts.husky + 2 →
  totalPups counts pupCounts - totalAdultDogs counts = 30 := by
  sorry

#check pup_difference

end NUMINAMATH_CALUDE_pup_difference_l389_38981


namespace NUMINAMATH_CALUDE_sum_of_A_and_C_is_seven_l389_38926

theorem sum_of_A_and_C_is_seven (A B C D : ℕ) : 
  A ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  B ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  C ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  D ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (A : ℚ) / B + (C : ℚ) / D = 3 →
  A + C = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_A_and_C_is_seven_l389_38926


namespace NUMINAMATH_CALUDE_two_digit_multiplication_error_l389_38964

theorem two_digit_multiplication_error (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →
  (10 ≤ b ∧ b < 100) →
  a * b = 936 →
  ((a + 40) * b = 2496 ∨ a * (b + 40) = 2496) →
  a + b = 63 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_multiplication_error_l389_38964


namespace NUMINAMATH_CALUDE_gathering_handshakes_l389_38916

def num_dwarves : ℕ := 25
def num_elves : ℕ := 18

def handshakes_among_dwarves (n : ℕ) : ℕ := n * (n - 1) / 2
def handshakes_between_dwarves_and_elves (d e : ℕ) : ℕ := d * e

def total_handshakes (d e : ℕ) : ℕ :=
  handshakes_among_dwarves d + handshakes_between_dwarves_and_elves d e

theorem gathering_handshakes :
  total_handshakes num_dwarves num_elves = 750 := by
  sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l389_38916


namespace NUMINAMATH_CALUDE_prime_sequence_l389_38988

theorem prime_sequence (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
by sorry

end NUMINAMATH_CALUDE_prime_sequence_l389_38988


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l389_38983

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 104 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 104 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l389_38983


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l389_38977

variable (a b : ℝ)

theorem problem_1 : 2 * a * (a^2 - 3*a - 1) = 2*a^3 - 6*a^2 - 2*a := by
  sorry

theorem problem_2 : (a^2*b - 2*a*b^2 + b^3) / b - (a + b)^2 = -4*a*b := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l389_38977


namespace NUMINAMATH_CALUDE_workshop_workers_l389_38993

/-- The total number of workers in a workshop given specific salary conditions -/
theorem workshop_workers (average_salary : ℚ) (technician_count : ℕ) (technician_salary : ℚ) (rest_salary : ℚ) : 
  average_salary = 850 ∧ 
  technician_count = 7 ∧ 
  technician_salary = 1000 ∧ 
  rest_salary = 780 →
  ∃ (total_workers : ℕ), total_workers = 22 ∧
    (technician_count : ℚ) * technician_salary + 
    (total_workers - technician_count : ℚ) * rest_salary = 
    (total_workers : ℚ) * average_salary :=
by
  sorry


end NUMINAMATH_CALUDE_workshop_workers_l389_38993


namespace NUMINAMATH_CALUDE_lindas_coins_l389_38940

theorem lindas_coins (total_coins : ℕ) (nickel_value dime_value : ℚ) 
  (swap_increase : ℚ) (h1 : total_coins = 30) 
  (h2 : nickel_value = 5/100) (h3 : dime_value = 10/100)
  (h4 : swap_increase = 90/100) : ∃ (nickels : ℕ), 
  nickels * nickel_value + (total_coins - nickels) * dime_value = 180/100 := by
  sorry

end NUMINAMATH_CALUDE_lindas_coins_l389_38940


namespace NUMINAMATH_CALUDE_unit_vector_parallel_to_3_4_l389_38968

def is_unit_vector (v : ℝ × ℝ) : Prop :=
  v.1^2 + v.2^2 = 1

def is_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem unit_vector_parallel_to_3_4 :
  ∃ (v : ℝ × ℝ), is_unit_vector v ∧ is_parallel v (3, 4) ∧
  (v = (3/5, 4/5) ∨ v = (-3/5, -4/5)) :=
sorry

end NUMINAMATH_CALUDE_unit_vector_parallel_to_3_4_l389_38968


namespace NUMINAMATH_CALUDE_trig_identity_l389_38925

theorem trig_identity (α : ℝ) : 
  (1 - 2 * Real.sin (2 * α) ^ 2) / (1 - Real.sin (4 * α)) = 
  (1 + Real.tan (2 * α)) / (1 - Real.tan (2 * α)) := by sorry

end NUMINAMATH_CALUDE_trig_identity_l389_38925


namespace NUMINAMATH_CALUDE_hcf_is_three_l389_38936

-- Define the properties of our two numbers
def number_properties (a b : ℕ) : Prop :=
  ∃ (k : ℕ), a = 3 * k ∧ b = 4 * k ∧ Nat.lcm a b = 36

-- Theorem statement
theorem hcf_is_three {a b : ℕ} (h : number_properties a b) : Nat.gcd a b = 3 := by
  sorry

end NUMINAMATH_CALUDE_hcf_is_three_l389_38936


namespace NUMINAMATH_CALUDE_remainder_3_pow_2040_mod_11_l389_38946

theorem remainder_3_pow_2040_mod_11 : 3^2040 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2040_mod_11_l389_38946


namespace NUMINAMATH_CALUDE_probability_face_card_is_three_thirteenths_l389_38960

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of jacks, queens, and kings in a standard deck -/
def face_cards : ℕ := 12

/-- The probability of drawing a jack, queen, or king from a standard deck -/
def probability_face_card : ℚ := face_cards / deck_size

theorem probability_face_card_is_three_thirteenths :
  probability_face_card = 3 / 13 := by sorry

end NUMINAMATH_CALUDE_probability_face_card_is_three_thirteenths_l389_38960


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l389_38905

theorem arithmetic_calculation : 4 * 6 * 8 - 24 / 3 = 184 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l389_38905


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l389_38956

-- Define the arithmetic sequence
def arithmetic_sequence (x y z : ℤ) : Prop :=
  ∃ d : ℤ, y = x + d ∧ z = y + d

-- Theorem statement
theorem arithmetic_sequence_problem (x y z w u : ℤ) 
  (h1 : arithmetic_sequence x y z)
  (h2 : x = 1370)
  (h3 : z = 1070)
  (h4 : w = -180)
  (h5 : u = -6430) :
  w^3 - u^2 + y^2 = -44200100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l389_38956


namespace NUMINAMATH_CALUDE_train_crossing_time_l389_38963

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 450 ∧ train_speed_kmh = 180 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l389_38963


namespace NUMINAMATH_CALUDE_solution_set_part_i_min_pq_part_ii_l389_38987

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x - 3|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x ≥ 4} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := by sorry

-- Part II
theorem min_pq_part_ii (m p q : ℝ) (hm : m > 0) (hp : p > 0) (hq : q > 0) :
  (∀ x, f m x ≥ 3) ∧ (∃ x, f m x = 3) ∧ (1/p + 1/(2*q) = m) →
  ∀ r s, r > 0 ∧ s > 0 ∧ 1/r + 1/(2*s) = m → p*q ≤ r*s ∧ p*q = 1/18 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_min_pq_part_ii_l389_38987


namespace NUMINAMATH_CALUDE_triangle_covering_theorem_l389_38969

/-- A triangle represented by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A convex polygon represented by its vertices -/
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)

/-- Predicate to check if a triangle covers a convex polygon -/
def covers (t : Triangle) (p : ConvexPolygon) : Prop := sorry

/-- Predicate to check if two triangles are congruent -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- Predicate to check if a line is parallel to or coincident with a side of a polygon -/
def parallel_or_coincident_with_side (line : ℝ × ℝ → ℝ × ℝ → Prop) (p : ConvexPolygon) : Prop := sorry

theorem triangle_covering_theorem (ABC : Triangle) (M : ConvexPolygon) :
  covers ABC M →
  ∃ T : Triangle, congruent T ABC ∧ covers T M ∧
    ∃ side : ℝ × ℝ → ℝ × ℝ → Prop, parallel_or_coincident_with_side side M :=
by sorry

end NUMINAMATH_CALUDE_triangle_covering_theorem_l389_38969


namespace NUMINAMATH_CALUDE_infinite_sum_equals_one_tenth_l389_38935

/-- The infinite sum of n^2 / (n^6 + 5) from n = 0 to infinity equals 1/10 -/
theorem infinite_sum_equals_one_tenth :
  (∑' n : ℕ, (n^2 : ℝ) / (n^6 + 5)) = 1/10 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_one_tenth_l389_38935


namespace NUMINAMATH_CALUDE_abs_sum_equals_sum_abs_necessary_not_sufficient_l389_38933

theorem abs_sum_equals_sum_abs_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a * b > 0 → |a + b| = |a| + |b|) ∧
  (∃ a b : ℝ, |a + b| = |a| + |b| ∧ a * b ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_abs_sum_equals_sum_abs_necessary_not_sufficient_l389_38933


namespace NUMINAMATH_CALUDE_black_circle_area_black_circle_area_proof_l389_38978

theorem black_circle_area (cube_edge : Real) (yellow_paint_area : Real) : Real :=
  let cube_face_area := cube_edge ^ 2
  let total_surface_area := 6 * cube_face_area
  let yellow_area_per_face := yellow_paint_area / 6
  let black_circle_area := cube_face_area - yellow_area_per_face
  
  black_circle_area

theorem black_circle_area_proof :
  black_circle_area 12 432 = 72 := by
  sorry

end NUMINAMATH_CALUDE_black_circle_area_black_circle_area_proof_l389_38978


namespace NUMINAMATH_CALUDE_bens_daily_start_amount_l389_38973

/-- Proves that given the conditions of Ben's savings scenario, he must start with $50 each day -/
theorem bens_daily_start_amount :
  ∀ (X : ℚ),
  (∃ (D : ℕ),
    (2 * (D * (X - 15)) + 10 = 500) ∧
    (D = 7)) →
  X = 50 := by
  sorry

end NUMINAMATH_CALUDE_bens_daily_start_amount_l389_38973


namespace NUMINAMATH_CALUDE_cloth_sale_problem_l389_38938

/-- Given a shopkeeper's cloth sale scenario, calculate the number of metres sold. -/
theorem cloth_sale_problem (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ) : 
  total_selling_price = 9000 →
  loss_per_metre = 6 →
  cost_price_per_metre = 36 →
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_problem_l389_38938


namespace NUMINAMATH_CALUDE_factorization_equality_l389_38980

theorem factorization_equality (a : ℝ) : 
  (a^2 + a)^2 + 4*(a^2 + a) - 12 = (a - 1)*(a + 2)*(a^2 + a + 6) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l389_38980


namespace NUMINAMATH_CALUDE_inequality_proof_l389_38999

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1) / (y + 1) + (y + 1) / (z + 1) + (z + 1) / (x + 1) ≤ x / y + y / z + z / x :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l389_38999


namespace NUMINAMATH_CALUDE_arithmetic_progressions_in_S_l389_38931

def S : Set ℤ := {n : ℤ | ∃ k : ℕ, n = ⌊k * Real.pi⌋}

theorem arithmetic_progressions_in_S :
  (∀ k : ℕ, ∃ (a d : ℤ) (f : Fin k → ℤ), (∀ i : Fin k, f i ∈ S) ∧ 
    (∀ i : Fin k, f i = a + i.val * d)) ∧
  ¬(∃ (a d : ℤ) (f : ℕ → ℤ), (∀ n : ℕ, f n ∈ S) ∧ 
    (∀ n : ℕ, f (n + 1) - f n = d)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progressions_in_S_l389_38931


namespace NUMINAMATH_CALUDE_profit_calculation_l389_38904

def number_of_bags : ℕ := 100
def selling_price : ℚ := 10
def buying_price : ℚ := 7

theorem profit_calculation :
  (number_of_bags : ℚ) * (selling_price - buying_price) = 300 := by sorry

end NUMINAMATH_CALUDE_profit_calculation_l389_38904


namespace NUMINAMATH_CALUDE_sqrt_difference_approximation_l389_38959

/-- Approximation of square root of 11 -/
def sqrt11_approx : ℝ := 3.31662

/-- Approximation of square root of 6 -/
def sqrt6_approx : ℝ := 2.44948

/-- The result we want to prove is close to the actual difference -/
def result : ℝ := 0.87

/-- Theorem stating that the difference between sqrt(11) and sqrt(6) is close to 0.87 -/
theorem sqrt_difference_approximation : |Real.sqrt 11 - Real.sqrt 6 - result| < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_approximation_l389_38959


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l389_38954

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 6

-- Define the solution set condition
def solution_set (a b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

-- Theorem statement
theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, f a x > 4 ↔ x ∈ solution_set a b) →
  (a = 1 ∧ b = 2) ∧
  (∀ c, ∀ x, a * x^2 - (a * c + b) * x + b * c < 0 ↔ 1 < x ∧ x < 2 * c) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l389_38954


namespace NUMINAMATH_CALUDE_noodles_given_to_william_l389_38986

/-- Given that Daniel initially had 54.0 noodles and was left with 42 noodles after giving some to William,
    prove that the number of noodles Daniel gave to William is 12. -/
theorem noodles_given_to_william (initial_noodles : ℝ) (remaining_noodles : ℝ) 
    (h1 : initial_noodles = 54.0) 
    (h2 : remaining_noodles = 42) : 
  initial_noodles - remaining_noodles = 12 := by
  sorry

end NUMINAMATH_CALUDE_noodles_given_to_william_l389_38986


namespace NUMINAMATH_CALUDE_tom_vegetable_ratio_l389_38914

/-- The ratio of broccoli to carrots eaten by Tom -/
def broccoli_to_carrots_ratio : ℚ := by sorry

theorem tom_vegetable_ratio :
  let carrot_calories_per_pound : ℚ := 51
  let carrot_amount : ℚ := 1
  let broccoli_calories_per_pound : ℚ := carrot_calories_per_pound / 3
  let total_calories : ℚ := 85
  let broccoli_amount : ℚ := (total_calories - carrot_calories_per_pound * carrot_amount) / broccoli_calories_per_pound
  broccoli_to_carrots_ratio = broccoli_amount / carrot_amount := by sorry

end NUMINAMATH_CALUDE_tom_vegetable_ratio_l389_38914


namespace NUMINAMATH_CALUDE_modified_counting_game_l389_38962

theorem modified_counting_game (n : ℕ) (a₁ : ℕ) (d : ℕ) (aₙ : ℕ → ℕ) :
  a₁ = 1 →
  d = 2 →
  (∀ k, aₙ k = a₁ + (k - 1) * d) →
  aₙ 53 = 105 :=
by sorry

end NUMINAMATH_CALUDE_modified_counting_game_l389_38962


namespace NUMINAMATH_CALUDE_perpendicular_lines_condition_l389_38902

-- Define the slopes of the two lines
def slope1 (a : ℝ) := a
def slope2 (a : ℝ) := -4 * a

-- Define perpendicularity condition
def isPerpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_condition (a : ℝ) :
  (isPerpendicular (slope1 a) (slope2 a) → (a = 1/2 ∨ a = -1/2)) ∧
  ¬(a = 1/2 → isPerpendicular (slope1 a) (slope2 a)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_condition_l389_38902


namespace NUMINAMATH_CALUDE_factorization_equality_l389_38961

theorem factorization_equality (a b : ℝ) : 12 * a^3 * b - 12 * a^2 * b + 3 * a * b = 3 * a * b * (2*a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l389_38961


namespace NUMINAMATH_CALUDE_solve_for_y_l389_38998

theorem solve_for_y (x y : ℤ) (h1 : x^2 - x + 6 = y + 2) (h2 : x = -5) : y = 34 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l389_38998


namespace NUMINAMATH_CALUDE_cube_volume_fourth_power_l389_38974

/-- The volume of a cube with surface area 864 square units, expressed as the fourth power of its side length -/
theorem cube_volume_fourth_power (s : ℝ) (h : 6 * s^2 = 864) : s^4 = 20736 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_fourth_power_l389_38974


namespace NUMINAMATH_CALUDE_relationship_abc_l389_38915

theorem relationship_abc : ∀ (a b c : ℝ), 
  a = 1/3 → b = Real.sin (1/3) → c = 1/Real.pi → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l389_38915


namespace NUMINAMATH_CALUDE_complex_number_properties_l389_38976

open Complex

theorem complex_number_properties (z₁ z₂ z : ℂ) (b : ℝ) : 
  z₁ = 1 - I ∧ 
  z₂ = 4 + 6*I ∧ 
  z = 1 + b*I ∧ 
  (z + z₁).im = 0 →
  (abs z₁ + z₂ = Complex.mk (Real.sqrt 2 + 4) 6) ∧ 
  abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_properties_l389_38976


namespace NUMINAMATH_CALUDE_bug_shortest_distance_l389_38994

/-- The shortest distance between two bugs moving on an equilateral triangle --/
theorem bug_shortest_distance (side_length : ℝ) (speed1 speed2 : ℝ) :
  side_length = 60 ∧ speed1 = 4 ∧ speed2 = 3 →
  ∃ (t d : ℝ),
    t = 300 / 37 ∧
    d = Real.sqrt (43200 / 37) ∧
    ∀ (t' : ℝ), t' ≥ 0 →
      (speed1 * t')^2 + (side_length - speed2 * t')^2 -
      2 * (speed1 * t') * (side_length - speed2 * t') * (1/2) ≥ d^2 :=
by sorry

end NUMINAMATH_CALUDE_bug_shortest_distance_l389_38994


namespace NUMINAMATH_CALUDE_max_unpainted_cubes_l389_38948

/-- Represents a 3D coordinate in a 3x3x3 cube arrangement -/
structure Coordinate where
  x : Fin 3
  y : Fin 3
  z : Fin 3

/-- Represents a cube in the 3x3x3 arrangement -/
structure Cube where
  coord : Coordinate
  painted : Bool

/-- Represents the 3x3x3 cube arrangement -/
def CubeArrangement : Type := Array Cube

/-- Checks if a cube is on the surface of the 3x3x3 arrangement -/
def isOnSurface (c : Coordinate) : Bool :=
  c.x = 0 || c.x = 2 || c.y = 0 || c.y = 2 || c.z = 0 || c.z = 2

/-- Counts the number of unpainted cubes in the arrangement -/
def countUnpaintedCubes (arr : CubeArrangement) : Nat :=
  arr.foldl (fun count cube => if !cube.painted then count + 1 else count) 0

/-- The main theorem stating the maximum number of unpainted cubes -/
theorem max_unpainted_cubes (arr : CubeArrangement) :
  arr.size = 27 → countUnpaintedCubes arr ≤ 15 := by sorry

end NUMINAMATH_CALUDE_max_unpainted_cubes_l389_38948


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l389_38928

theorem geometric_progression_fourth_term : 
  ∀ (a : ℝ) (r : ℝ),
  a > 0 → r > 0 →
  a = 2^(1/3 : ℝ) →
  a * r = 2^(1/4 : ℝ) →
  a * r^2 = 2^(1/5 : ℝ) →
  a * r^3 = 2^(1/9 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l389_38928


namespace NUMINAMATH_CALUDE_factorization_x4_minus_1_l389_38932

theorem factorization_x4_minus_1 (x : ℂ) : x^4 - 1 = (x + Complex.I) * (x - Complex.I) * (x - 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_1_l389_38932


namespace NUMINAMATH_CALUDE_base_7_representation_of_500_l389_38997

/-- Converts a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 7 to a natural number -/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_7_representation_of_500 :
  toBase7 500 = [1, 3, 1, 3] ∧ fromBase7 [1, 3, 1, 3] = 500 := by
  sorry

end NUMINAMATH_CALUDE_base_7_representation_of_500_l389_38997


namespace NUMINAMATH_CALUDE_machines_needed_for_multiple_production_l389_38966

/-- Given that 4 machines produce x units in 6 days, prove that 4m machines are needed to produce m*x units in 6 days, where all machines work at the same constant rate. -/
theorem machines_needed_for_multiple_production 
  (x : ℝ) (m : ℝ) (rate : ℝ) (h1 : x > 0) (h2 : m > 0) (h3 : rate > 0) :
  4 * rate * 6 = x → (4 * m) * rate * 6 = m * x :=
by
  sorry

#check machines_needed_for_multiple_production

end NUMINAMATH_CALUDE_machines_needed_for_multiple_production_l389_38966


namespace NUMINAMATH_CALUDE_paper_towel_case_rolls_l389_38911

/-- The number of rolls in a case of paper towels -/
def number_of_rolls : ℕ := 12

/-- The price of the case in dollars -/
def case_price : ℚ := 9

/-- The price of an individual roll in dollars -/
def individual_roll_price : ℚ := 1

/-- The savings percentage per roll when buying the case -/
def savings_percentage : ℚ := 25 / 100

theorem paper_towel_case_rolls :
  case_price = number_of_rolls * (individual_roll_price * (1 - savings_percentage)) :=
sorry

end NUMINAMATH_CALUDE_paper_towel_case_rolls_l389_38911


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l389_38991

/-- Given a coat with an original price and a price reduction, 
    calculate the percentage reduction in price. -/
theorem price_reduction_percentage 
  (original_price : ℝ) 
  (price_reduction : ℝ) 
  (h1 : original_price = 500)
  (h2 : price_reduction = 350) : 
  (price_reduction / original_price) * 100 = 70 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l389_38991


namespace NUMINAMATH_CALUDE_plane_through_points_l389_38912

def point1 : ℝ × ℝ × ℝ := (2, -3, 5)
def point2 : ℝ × ℝ × ℝ := (4, -3, 6)
def point3 : ℝ × ℝ × ℝ := (6, -4, 8)

def plane_equation (x y z : ℝ) : ℝ := x - 2*y + 2*z - 18

theorem plane_through_points :
  (plane_equation point1.1 point1.2.1 point1.2.2 = 0) ∧
  (plane_equation point2.1 point2.2.1 point2.2.2 = 0) ∧
  (plane_equation point3.1 point3.2.1 point3.2.2 = 0) ∧
  (1 > 0) ∧
  (Nat.gcd (Nat.gcd 1 2) (Nat.gcd 2 18) = 1) := by
  sorry

end NUMINAMATH_CALUDE_plane_through_points_l389_38912


namespace NUMINAMATH_CALUDE_triangle_point_distance_l389_38919

/-- Given a triangle ABC with AB = 8, BC = 20, CA = 16, and points D and E on BC
    such that CD = 8 and ∠BAE = ∠CAD, prove that BE = 2 -/
theorem triangle_point_distance (A B C D E : ℝ × ℝ) : 
  let dist := (fun (P Q : ℝ × ℝ) => Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))
  let angle := (fun (P Q R : ℝ × ℝ) => Real.arccos (((Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2)) / 
                (dist P Q * dist P R)))
  dist A B = 8 →
  dist B C = 20 →
  dist C A = 16 →
  D.1 = B.1 + 12 / 20 * (C.1 - B.1) ∧ D.2 = B.2 + 12 / 20 * (C.2 - B.2) →
  angle B A E = angle C A D →
  dist B E = 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_point_distance_l389_38919


namespace NUMINAMATH_CALUDE_problem_statement_l389_38924

/-- Given a function f(x) = -ax^5 - x^3 + bx - 7 where f(2) = -9, prove that f(-2) = -5 -/
theorem problem_statement (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = -a * x^5 - x^3 + b * x - 7)
  (h2 : f 2 = -9) : 
  f (-2) = -5 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l389_38924


namespace NUMINAMATH_CALUDE_carmen_daniel_difference_l389_38934

/-- Calculates the difference in miles biked between two cyclists after a given time -/
def miles_difference (carmen_rate daniel_rate time : ℝ) : ℝ :=
  carmen_rate * time - daniel_rate * time

theorem carmen_daniel_difference :
  miles_difference 15 10 3 = 15 := by sorry

end NUMINAMATH_CALUDE_carmen_daniel_difference_l389_38934


namespace NUMINAMATH_CALUDE_binomial_coefficient_times_n_l389_38906

theorem binomial_coefficient_times_n (n : ℕ+) : n * Nat.choose 4 3 = 4 * n := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_times_n_l389_38906


namespace NUMINAMATH_CALUDE_rod_friction_coefficient_l389_38975

noncomputable def coefficient_of_friction (initial_normal_force_ratio : ℝ) (tilt_angle : ℝ) : ℝ :=
  (1 - initial_normal_force_ratio * Real.cos tilt_angle) / (initial_normal_force_ratio * Real.sin tilt_angle)

theorem rod_friction_coefficient (initial_normal_force_ratio : ℝ) (tilt_angle : ℝ) 
  (h1 : initial_normal_force_ratio = 11)
  (h2 : tilt_angle = 80 * π / 180) :
  ∃ ε > 0, |coefficient_of_friction initial_normal_force_ratio tilt_angle - 0.17| < ε :=
sorry

end NUMINAMATH_CALUDE_rod_friction_coefficient_l389_38975


namespace NUMINAMATH_CALUDE_range_of_m_l389_38901

/-- Given a quadratic inequality and a function with specific domain,
    prove that the range of m is [-1, 0] -/
theorem range_of_m (a : ℝ) (m : ℝ) : 
  (a > 0 ∧ a ≠ 1) →
  (∀ x : ℝ, a * x^2 - a * x - 2 * a^2 > 1 ↔ -a < x ∧ x < 2*a) →
  (∀ x : ℝ, (1/a)^(x^2 + 2*m*x - m) - 1 ≥ 0) →
  m ∈ Set.Icc (-1 : ℝ) 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l389_38901


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l389_38917

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * (x + 2 * y) - 5 * y = -1) ∧ (3 * (x - y) + y = 2) ∧ (x = -4) ∧ (y = -7) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l389_38917


namespace NUMINAMATH_CALUDE_intersection_M_N_l389_38929

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

theorem intersection_M_N : N ∩ M = {x | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l389_38929


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_hundred_l389_38950

theorem twenty_five_percent_less_than_hundred (x : ℝ) : x + (1/4) * x = 75 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_hundred_l389_38950


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l389_38955

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a 2 → a 1 + a 3 = 5 → a 2 + a 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l389_38955


namespace NUMINAMATH_CALUDE_square_with_ascending_digits_l389_38920

theorem square_with_ascending_digits : ∃ n : ℕ, 
  (n^2).repr.takeRight 5 = "23456" ∧ 
  n^2 = 54563456 := by
  sorry

end NUMINAMATH_CALUDE_square_with_ascending_digits_l389_38920


namespace NUMINAMATH_CALUDE_monomial_count_l389_38942

def is_monomial (expr : String) : Bool :=
  match expr with
  | "a" => true
  | "-2ab" => true
  | "x+y" => false
  | "x^2+y^2" => false
  | "-1" => true
  | "1/2ab^2c^3" => true
  | _ => false

def expressions : List String := ["a", "-2ab", "x+y", "x^2+y^2", "-1", "1/2ab^2c^3"]

theorem monomial_count :
  (expressions.filter is_monomial).length = 4 := by sorry

end NUMINAMATH_CALUDE_monomial_count_l389_38942


namespace NUMINAMATH_CALUDE_circle_not_in_second_quadrant_l389_38996

/-- A circle in the xy-plane with center (a, 0) and radius 2 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = 4}

/-- The second quadrant of the xy-plane -/
def SecondQuadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

/-- The circle does not pass through the second quadrant -/
def NotInSecondQuadrant (a : ℝ) : Prop :=
  Circle a ∩ SecondQuadrant = ∅

theorem circle_not_in_second_quadrant (a : ℝ) :
  NotInSecondQuadrant a → a ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_circle_not_in_second_quadrant_l389_38996


namespace NUMINAMATH_CALUDE_man_work_time_l389_38947

theorem man_work_time (total_work : ℝ) (man_rate : ℝ) (son_rate : ℝ) 
  (h1 : man_rate + son_rate = total_work / 3)
  (h2 : son_rate = total_work / 5.25) :
  man_rate = total_work / 7 := by
sorry

end NUMINAMATH_CALUDE_man_work_time_l389_38947


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l389_38967

theorem quadratic_solution_property (a : ℝ) : 
  a^2 - 2*a - 1 = 0 → 2*a^2 - 4*a + 2023 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l389_38967


namespace NUMINAMATH_CALUDE_base_b_difference_divisibility_l389_38982

def base_conversion (b : ℕ) : ℤ := 2 * b^3 - 2 * b^2 + b - 1

theorem base_b_difference_divisibility (b : ℕ) (h : 4 ≤ b ∧ b ≤ 8) :
  ¬(5 ∣ base_conversion b) ↔ b = 6 :=
by sorry

end NUMINAMATH_CALUDE_base_b_difference_divisibility_l389_38982


namespace NUMINAMATH_CALUDE_perfect_square_sums_l389_38930

theorem perfect_square_sums : ∃ (x y : ℕ+), 
  ∃ (a b c : ℕ+),
  (x + y : ℕ) = a^2 ∧
  (x^2 + y^2 : ℕ) = b^2 ∧
  (x^3 + y^3 : ℕ) = c^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sums_l389_38930


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l389_38995

/-- Represents a square quilt block -/
structure QuiltBlock where
  size : Nat
  total_squares : Nat
  shaded_area : Rat

/-- Creates a quilt block with the given specifications -/
def create_quilt_block : QuiltBlock :=
  { size := 4,
    total_squares := 16,
    shaded_area := 2 }

/-- Calculates the fraction of the quilt that is shaded -/
def shaded_fraction (quilt : QuiltBlock) : Rat :=
  quilt.shaded_area / quilt.total_squares

theorem quilt_shaded_fraction :
  let quilt := create_quilt_block
  shaded_fraction quilt = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l389_38995


namespace NUMINAMATH_CALUDE_triangle_inradius_l389_38979

/-- Given a triangle with perimeter 32 cm and area 40 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : p = 32) 
  (h_area : A = 40) 
  (h_inradius : A = r * p / 2) : 
  r = 2.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_inradius_l389_38979


namespace NUMINAMATH_CALUDE_ladder_problem_l389_38921

theorem ladder_problem (ladder_length height_on_wall : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height_on_wall = 12) :
  ∃ (base_distance : ℝ), 
    base_distance^2 + height_on_wall^2 = ladder_length^2 ∧ 
    base_distance = 5 :=
by sorry

end NUMINAMATH_CALUDE_ladder_problem_l389_38921


namespace NUMINAMATH_CALUDE_gear_teeth_problem_l389_38949

theorem gear_teeth_problem :
  ∀ (initial_teeth_1 initial_teeth_2 final_teeth_1 final_teeth_2 : ℕ),
    (initial_teeth_1 : ℚ) / initial_teeth_2 = 7 / 9 →
    final_teeth_1 = initial_teeth_1 + 3 →
    final_teeth_2 = initial_teeth_2 - 3 →
    (final_teeth_1 : ℚ) / final_teeth_2 = 3 / 1 →
    initial_teeth_1 = 9 ∧ initial_teeth_2 = 7 ∧ final_teeth_1 = 12 ∧ final_teeth_2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_gear_teeth_problem_l389_38949


namespace NUMINAMATH_CALUDE_john_candy_count_l389_38953

/-- Represents the number of candies each friend has -/
structure CandyCounts where
  bob : ℕ
  mary : ℕ
  john : ℕ
  sue : ℕ
  sam : ℕ

/-- The total number of candies all friends have together -/
def totalCandies : ℕ := 50

/-- The given candy counts for Bob, Mary, Sue, and Sam -/
def givenCounts : CandyCounts where
  bob := 10
  mary := 5
  john := 0  -- We don't know John's count yet
  sue := 20
  sam := 10

/-- Theorem stating that John's candy count is equal to the total minus the sum of others -/
theorem john_candy_count (c : CandyCounts) (h : c = givenCounts) :
  c.john = totalCandies - (c.bob + c.mary + c.sue + c.sam) :=
by sorry

end NUMINAMATH_CALUDE_john_candy_count_l389_38953


namespace NUMINAMATH_CALUDE_largest_reciprocal_l389_38971

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/3 → b = 2/5 → c = 1 → d = 5 → e = 1986 →
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l389_38971


namespace NUMINAMATH_CALUDE_tangent_line_equation_l389_38900

/-- The curve C defined by y = x^3 -/
def C : ℝ → ℝ := fun x ↦ x^3

/-- The point P through which the tangent line passes -/
def P : ℝ × ℝ := (1, 1)

/-- Predicate to check if a line passes through the fourth quadrant -/
def passes_through_fourth_quadrant (a b c : ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y < 0 ∧ a * x + b * y + c = 0

/-- The tangent line to curve C at point (x₀, C x₀) -/
def tangent_line (x₀ : ℝ) : ℝ → ℝ := fun x ↦ C x₀ + (3 * x₀^2) * (x - x₀)

theorem tangent_line_equation :
  ∃ x₀ : ℝ, 
    tangent_line x₀ P.1 = P.2 ∧ 
    ¬passes_through_fourth_quadrant 3 (-4) 1 ∧
    ∀ x, tangent_line x₀ x = 3*x - 4*(C x) + 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l389_38900


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l389_38989

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let x := total / (a + b + c)
  let part1 := a * x
  let part2 := b * x
  let part3 := c * x
  (total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7) →
  min part1 (min part2 part3) = 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l389_38989


namespace NUMINAMATH_CALUDE_max_incorrect_answers_is_correct_l389_38907

/-- The passing threshold for the exam as a percentage -/
def pass_threshold : ℝ := 85

/-- The total number of questions in the exam -/
def total_questions : ℕ := 50

/-- The maximum number of questions that can be answered incorrectly while still passing -/
def max_incorrect_answers : ℕ := 7

/-- Theorem stating that max_incorrect_answers is the maximum number of questions
    that can be answered incorrectly while still passing the exam -/
theorem max_incorrect_answers_is_correct :
  ∀ n : ℕ, 
    (n ≤ max_incorrect_answers ↔ 
      (total_questions - n : ℝ) / total_questions * 100 ≥ pass_threshold) :=
by sorry

end NUMINAMATH_CALUDE_max_incorrect_answers_is_correct_l389_38907


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l389_38909

theorem intersection_point_of_lines (x y : ℚ) : 
  (5 * x - 2 * y = 8) ∧ (3 * x + 4 * y = 12) ↔ (x = 28/13 ∧ y = 18/13) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l389_38909


namespace NUMINAMATH_CALUDE_least_common_time_for_seven_horses_l389_38984

def horse_times : Finset ℕ := Finset.range 12

theorem least_common_time_for_seven_horses :
  ∃ (S : Finset ℕ), S ⊆ horse_times ∧ S.card = 7 ∧
  (∀ n ∈ S, n > 0) ∧
  (∀ (T : ℕ), (∀ n ∈ S, T % n = 0) → T ≥ 420) ∧
  (∀ n ∈ S, 420 % n = 0) :=
sorry

end NUMINAMATH_CALUDE_least_common_time_for_seven_horses_l389_38984


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_a_range_l389_38927

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The theorem states that if f(x) is decreasing on (-∞, 4], then a < -5 -/
theorem quadratic_decreasing_implies_a_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) →
  a < -5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_a_range_l389_38927


namespace NUMINAMATH_CALUDE_savings_account_calculation_final_amount_is_690_l389_38965

/-- Calculates the final amount in a savings account after two years with given conditions --/
theorem savings_account_calculation (initial_deposit : ℝ) (first_year_rate : ℝ) 
  (withdrawal_percentage : ℝ) (second_year_rate : ℝ) : ℝ :=
  let first_year_balance := initial_deposit * (1 + first_year_rate)
  let remaining_after_withdrawal := first_year_balance * (1 - withdrawal_percentage)
  let final_balance := remaining_after_withdrawal * (1 + second_year_rate)
  final_balance

/-- Proves that the final amount in the account is $690 given the specified conditions --/
theorem final_amount_is_690 : 
  savings_account_calculation 1000 0.20 0.50 0.15 = 690 := by
sorry

end NUMINAMATH_CALUDE_savings_account_calculation_final_amount_is_690_l389_38965


namespace NUMINAMATH_CALUDE_circle_equation_l389_38990

/-- A circle with center (a, 1) that is tangent to both lines x-y+1=0 and x-y-3=0 -/
structure TangentCircle where
  a : ℝ
  center : ℝ × ℝ
  tangent_line1 : ℝ → ℝ → ℝ
  tangent_line2 : ℝ → ℝ → ℝ
  center_def : center = (a, 1)
  tangent_line1_def : tangent_line1 = fun x y => x - y + 1
  tangent_line2_def : tangent_line2 = fun x y => x - y - 3
  is_tangent1 : ∃ (x y : ℝ), tangent_line1 x y = 0 ∧ (x - a)^2 + (y - 1)^2 = (x - center.1)^2 + (y - center.2)^2
  is_tangent2 : ∃ (x y : ℝ), tangent_line2 x y = 0 ∧ (x - a)^2 + (y - 1)^2 = (x - center.1)^2 + (y - center.2)^2

/-- The standard equation of the circle is (x-2)^2+(y-1)^2=2 -/
theorem circle_equation (c : TangentCircle) : 
  ∃ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 2 ∧ 
  (x - c.center.1)^2 + (y - c.center.2)^2 = (x - 2)^2 + (y - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l389_38990


namespace NUMINAMATH_CALUDE_function_property_l389_38945

theorem function_property (f : ℝ → ℝ) :
  (∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y) →
  f 400 = 4 →
  f 800 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l389_38945


namespace NUMINAMATH_CALUDE_sum_and_square_difference_implies_difference_l389_38937

theorem sum_and_square_difference_implies_difference
  (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 190) :
  x - y = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_square_difference_implies_difference_l389_38937


namespace NUMINAMATH_CALUDE_problem_solution_l389_38943

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : a^b = b^a) (h2 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l389_38943


namespace NUMINAMATH_CALUDE_min_value_of_function_l389_38972

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x + 1 / (x + 1) ≥ 1 ∧ ∃ y > -1, y + 1 / (y + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l389_38972


namespace NUMINAMATH_CALUDE_upstream_distance_l389_38970

/-- Proves that a man swimming downstream 16 km in 2 hours and upstream for 2 hours,
    with a speed of 6.5 km/h in still water, swims 10 km upstream. -/
theorem upstream_distance
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (still_water_speed : ℝ)
  (h_downstream_distance : downstream_distance = 16)
  (h_downstream_time : downstream_time = 2)
  (h_upstream_time : upstream_time = 2)
  (h_still_water_speed : still_water_speed = 6.5)
  : ∃ upstream_distance : ℝ,
    upstream_distance = 10 ∧
    upstream_distance = still_water_speed * upstream_time - 
      (downstream_distance / downstream_time - still_water_speed) * upstream_time :=
by
  sorry

end NUMINAMATH_CALUDE_upstream_distance_l389_38970


namespace NUMINAMATH_CALUDE_f_properties_l389_38941

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x + k * x^2 + (2 * k + 1) * x

theorem f_properties (k : ℝ) :
  (k ≥ 0 → ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f k x₁ < f k x₂) ∧
  (k < 0 → ∀ x : ℝ, 0 < x → f k x ≤ -3 / (4 * k) - 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l389_38941


namespace NUMINAMATH_CALUDE_set_equality_l389_38922

def S : Set ℕ := {x | ∃ k : ℤ, 12 = k * (6 - x)}

theorem set_equality : S = {0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 18} := by sorry

end NUMINAMATH_CALUDE_set_equality_l389_38922


namespace NUMINAMATH_CALUDE_max_gcd_of_eight_numbers_sum_595_l389_38939

/-- The maximum possible GCD of eight natural numbers summing to 595 -/
theorem max_gcd_of_eight_numbers_sum_595 :
  ∃ (a b c d e f g h : ℕ),
    a + b + c + d + e + f + g + h = 595 ∧
    ∀ (k : ℕ),
      k ∣ a ∧ k ∣ b ∧ k ∣ c ∧ k ∣ d ∧ k ∣ e ∧ k ∣ f ∧ k ∣ g ∧ k ∣ h →
      k ≤ 35 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_eight_numbers_sum_595_l389_38939


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l389_38992

theorem triangle_side_calculation (AB : ℝ) : 
  AB = 10 →
  ∃ (AC AD CD : ℝ),
    -- ABD is a 45-45-90 triangle
    AD = AB ∧
    -- ACD is a 30-60-90 triangle
    CD = 2 * AC ∧
    AD^2 = AC^2 + CD^2 ∧
    -- The result we want to prove
    CD = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l389_38992


namespace NUMINAMATH_CALUDE_ace_king_queen_probability_l389_38903

def standard_deck : ℕ := 52
def num_aces : ℕ := 4
def num_kings : ℕ := 4
def num_queens : ℕ := 4

theorem ace_king_queen_probability :
  (num_aces / standard_deck) * (num_kings / (standard_deck - 1)) * (num_queens / (standard_deck - 2)) = 16 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_ace_king_queen_probability_l389_38903


namespace NUMINAMATH_CALUDE_largest_difference_l389_38944

def A : ℕ := 3 * 1003^1004
def B : ℕ := 1003^1004
def C : ℕ := 1002 * 1003^1003
def D : ℕ := 3 * 1003^1003
def E : ℕ := 1003^1003
def F : ℕ := 1003^1002

theorem largest_difference :
  A - B > max (B - C) (max (C - D) (max (D - E) (E - F))) :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_l389_38944


namespace NUMINAMATH_CALUDE_smallest_debt_resolution_l389_38910

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 280

/-- A debt resolution is valid if it can be expressed as a combination of pigs and goats -/
def is_valid_resolution (debt : ℕ) : Prop :=
  ∃ (p g : ℤ), debt = pig_value * p + goat_value * g

/-- The smallest positive debt that can be resolved -/
def smallest_resolvable_debt : ℕ := 800

theorem smallest_debt_resolution :
  (is_valid_resolution smallest_resolvable_debt) ∧
  (∀ d : ℕ, d > 0 ∧ d < smallest_resolvable_debt → ¬(is_valid_resolution d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_debt_resolution_l389_38910


namespace NUMINAMATH_CALUDE_w_squared_value_l389_38985

theorem w_squared_value (w : ℝ) (h : (w + 10)^2 = (4*w + 6)*(w + 5)) : w^2 = 70/3 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l389_38985


namespace NUMINAMATH_CALUDE_sabrina_pencils_l389_38952

theorem sabrina_pencils (total : ℕ) (justin_extra : ℕ) : 
  total = 50 → justin_extra = 8 →
  ∃ (sabrina : ℕ), 
    sabrina + (2 * sabrina + justin_extra) = total ∧ 
    sabrina = 14 := by
  sorry

end NUMINAMATH_CALUDE_sabrina_pencils_l389_38952


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_on_terminal_side_l389_38908

/-- 
If the terminal side of angle α passes through point P(m, 2m) where m > 0, 
then sin(α) = 2√5/5.
-/
theorem sin_alpha_for_point_on_terminal_side (m : ℝ) (α : ℝ) 
  (h1 : m > 0) 
  (h2 : ∃ (x y : ℝ), x = m ∧ y = 2*m ∧ 
       x = Real.cos α * Real.sqrt (m^2 + (2*m)^2) ∧ 
       y = Real.sin α * Real.sqrt (m^2 + (2*m)^2)) : 
  Real.sin α = 2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_on_terminal_side_l389_38908


namespace NUMINAMATH_CALUDE_percentage_relation_l389_38951

theorem percentage_relation (A B x : ℝ) (hA : A > 0) (hB : B > 0) (h : A = (x / 100) * B) : 
  x = 100 * (A / B) := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l389_38951


namespace NUMINAMATH_CALUDE_snake_owners_count_l389_38918

/-- Represents the number of pet owners for different combinations of pets --/
structure PetOwners where
  total : Nat
  onlyDogs : Nat
  onlyCats : Nat
  onlyBirds : Nat
  onlySnakes : Nat
  catsAndDogs : Nat
  dogsAndBirds : Nat
  catsAndBirds : Nat
  catsAndSnakes : Nat
  dogsAndSnakes : Nat
  allCategories : Nat

/-- Calculates the total number of snake owners --/
def totalSnakeOwners (po : PetOwners) : Nat :=
  po.onlySnakes + po.catsAndSnakes + po.dogsAndSnakes + po.allCategories

/-- Theorem stating that the total number of snake owners is 25 --/
theorem snake_owners_count (po : PetOwners) 
  (h1 : po.total = 75)
  (h2 : po.onlyDogs = 20)
  (h3 : po.onlyCats = 15)
  (h4 : po.onlyBirds = 8)
  (h5 : po.onlySnakes = 10)
  (h6 : po.catsAndDogs = 5)
  (h7 : po.dogsAndBirds = 4)
  (h8 : po.catsAndBirds = 3)
  (h9 : po.catsAndSnakes = 7)
  (h10 : po.dogsAndSnakes = 6)
  (h11 : po.allCategories = 2) :
  totalSnakeOwners po = 25 := by
  sorry

end NUMINAMATH_CALUDE_snake_owners_count_l389_38918


namespace NUMINAMATH_CALUDE_correct_article_usage_l389_38958

/-- Represents the possible articles that can be used. -/
inductive Article
  | The
  | A
  | Blank

/-- Represents a pair of articles used in the sentence. -/
structure ArticlePair where
  first : Article
  second : Article

/-- Defines the correct article usage for the given sentence. -/
def correct_usage : ArticlePair :=
  { first := Article.The, second := Article.The }

/-- Determines if a noun is specific and known. -/
def is_specific_known (noun : String) : Bool :=
  match noun with
  | "bed" => true
  | _ => false

/-- Determines if a noun is made specific by additional information. -/
def is_specific_by_info (noun : String) (info : String) : Bool :=
  match noun, info with
  | "book", "I lost last week" => true
  | _, _ => false

/-- Theorem stating that the correct article usage is "the; the" given the conditions. -/
theorem correct_article_usage
  (bed : String)
  (book : String)
  (info : String)
  (h1 : is_specific_known bed = true)
  (h2 : is_specific_by_info book info = true) :
  correct_usage = { first := Article.The, second := Article.The } :=
by sorry

end NUMINAMATH_CALUDE_correct_article_usage_l389_38958
