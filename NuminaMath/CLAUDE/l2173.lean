import Mathlib

namespace NUMINAMATH_CALUDE_five_workers_completion_time_l2173_217399

/-- The productivity rates of five workers -/
structure WorkerRates where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ
  x₅ : ℝ

/-- The total amount of work to be done -/
def total_work : ℝ → ℝ := id

theorem five_workers_completion_time 
  (rates : WorkerRates) 
  (y : ℝ) 
  (h₁ : rates.x₁ + rates.x₂ + rates.x₃ = y / 327.5)
  (h₂ : rates.x₁ + rates.x₃ + rates.x₅ = y / 5)
  (h₃ : rates.x₁ + rates.x₃ + rates.x₄ = y / 6)
  (h₄ : rates.x₂ + rates.x₄ + rates.x₅ = y / 4) :
  y / (rates.x₁ + rates.x₂ + rates.x₃ + rates.x₄ + rates.x₅) = 3 := by
  sorry

end NUMINAMATH_CALUDE_five_workers_completion_time_l2173_217399


namespace NUMINAMATH_CALUDE_wheel_distance_covered_l2173_217328

/-- The distance covered by a wheel given its diameter and number of revolutions -/
theorem wheel_distance_covered (diameter : ℝ) (revolutions : ℝ) : 
  diameter = 14 → revolutions = 15.013648771610555 → 
  ∃ distance : ℝ, abs (distance - (π * diameter * revolutions)) < 0.001 ∧ abs (distance - 660.477) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_wheel_distance_covered_l2173_217328


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l2173_217394

/-- The probability that no two of three independently chosen real numbers 
    from [0, n] are within 2 units of each other is greater than 1/2 -/
def probability_condition (n : ℕ) : Prop :=
  (n - 4)^3 / n^3 > 1/2

/-- 12 is the smallest positive integer satisfying the probability condition -/
theorem smallest_n_satisfying_condition : 
  (∀ k < 12, ¬ probability_condition k) ∧ probability_condition 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l2173_217394


namespace NUMINAMATH_CALUDE_factorial_fraction_l2173_217391

theorem factorial_fraction (N : ℕ) :
  (Nat.factorial (N + 1)) / ((Nat.factorial (N + 2)) + (Nat.factorial N)) = 
  (N + 1) / (N^2 + 3*N + 3) := by
sorry

end NUMINAMATH_CALUDE_factorial_fraction_l2173_217391


namespace NUMINAMATH_CALUDE_cards_given_away_l2173_217366

theorem cards_given_away (brother_sets sister_sets friend_sets : ℕ) 
  (cards_per_set : ℕ) (h1 : brother_sets = 15) (h2 : sister_sets = 8) 
  (h3 : friend_sets = 4) (h4 : cards_per_set = 25) : 
  (brother_sets + sister_sets + friend_sets) * cards_per_set = 675 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_away_l2173_217366


namespace NUMINAMATH_CALUDE_S_singleton_I_singleton_l2173_217307

-- Define the set X
inductive X
| zero : X
| a : X
| b : X
| c : X

-- Define the addition operation on X
def add : X → X → X
| X.zero, y => y
| X.a, X.zero => X.a
| X.a, X.a => X.zero
| X.a, X.b => X.c
| X.a, X.c => X.b
| X.b, X.zero => X.b
| X.b, X.a => X.c
| X.b, X.b => X.zero
| X.b, X.c => X.a
| X.c, X.zero => X.c
| X.c, X.a => X.b
| X.c, X.b => X.a
| X.c, X.c => X.zero

-- Define the set of all functions from X to X
def M : Type := X → X

-- Define the set S
def S : Set M := {f : M | ∀ x y : X, f (add (add x y) x) = add (add (f x) (f y)) (f x)}

-- Define the set I
def I : Set M := {f : M | ∀ x : X, f (add x x) = add (f x) (f x)}

-- Theorem: S contains only one function (the zero function)
theorem S_singleton : ∃! f : M, f ∈ S := sorry

-- Theorem: I contains only one function (the zero function)
theorem I_singleton : ∃! f : M, f ∈ I := sorry

end NUMINAMATH_CALUDE_S_singleton_I_singleton_l2173_217307


namespace NUMINAMATH_CALUDE_inequality_proof_l2173_217398

theorem inequality_proof (a b c d e f : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h5 : 0 ≤ e) (h6 : 0 ≤ f)
  (h7 : a + b ≤ e) (h8 : c + d ≤ f) : 
  Real.sqrt (a * c) + Real.sqrt (b * d) ≤ Real.sqrt (e * f) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2173_217398


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2173_217318

-- Define the quadratic function
def f (x : ℝ) := x^2 - 5*x + 6

-- Define the solution set
def solution_set := { x : ℝ | 2 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem quadratic_inequality_solution :
  { x : ℝ | f x ≤ 0 } = solution_set := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2173_217318


namespace NUMINAMATH_CALUDE_flu_infection_rate_l2173_217320

/-- The average number of people infected by one person in each round -/
def average_infections : ℝ := 4

/-- The number of people initially infected -/
def initial_infected : ℕ := 2

/-- The total number of people infected after two rounds -/
def total_infected : ℕ := 50

theorem flu_infection_rate :
  initial_infected +
  initial_infected * average_infections +
  (initial_infected + initial_infected * average_infections) * average_infections =
  total_infected :=
sorry

end NUMINAMATH_CALUDE_flu_infection_rate_l2173_217320


namespace NUMINAMATH_CALUDE_almond_butter_servings_l2173_217325

-- Define the total amount of almond butter in cups
def total_almond_butter : ℚ := 17 + 1/3

-- Define the serving size in cups
def serving_size : ℚ := 1 + 1/2

-- Theorem: The number of servings in the container is 11 5/9
theorem almond_butter_servings :
  total_almond_butter / serving_size = 11 + 5/9 := by
  sorry

end NUMINAMATH_CALUDE_almond_butter_servings_l2173_217325


namespace NUMINAMATH_CALUDE_choose_two_from_three_l2173_217371

theorem choose_two_from_three (n : ℕ) (k : ℕ) : n = 3 ∧ k = 2 → Nat.choose n k = 3 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_three_l2173_217371


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l2173_217322

/-- Calculates the principal amount given the interest rate, time, and total interest. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℕ) : ℚ :=
  (interest : ℚ) * 100 / (rate * time)

/-- Theorem: Given a loan with 12% p.a. simple interest rate, if the interest
    amount after 10 years is Rs. 1500, then the principal amount borrowed was Rs. 1250. -/
theorem loan_principal_calculation (rate : ℚ) (time : ℕ) (interest : ℕ) :
  rate = 12 → time = 10 → interest = 1500 →
  calculate_principal rate time interest = 1250 := by
  sorry

#eval calculate_principal 12 10 1500

end NUMINAMATH_CALUDE_loan_principal_calculation_l2173_217322


namespace NUMINAMATH_CALUDE_triangle_problem_l2173_217355

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : (2 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C)
  (h2 : t.a = Real.sqrt 13)
  (h3 : t.b + t.c = 5) :
  t.A = π / 3 ∧ 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2173_217355


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2173_217341

def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem arithmetic_sequence_ratio 
  (a x b : ℝ) 
  (h1 : ∃ d, arithmetic_sequence a d 1 = a ∧ 
             arithmetic_sequence a d 2 = x ∧ 
             arithmetic_sequence a d 3 = b ∧ 
             arithmetic_sequence a d 4 = 2*x) :
  a / b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2173_217341


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2173_217378

theorem quadratic_factorization (a : ℝ) : a^2 - a + 1/4 = (a - 1/2)^2 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2173_217378


namespace NUMINAMATH_CALUDE_inequality_proof_l2173_217333

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 9*b*c) + b / Real.sqrt (b^2 + 9*c*a) + c / Real.sqrt (c^2 + 9*a*b) ≥ 3 / Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2173_217333


namespace NUMINAMATH_CALUDE_angle_DAE_in_special_triangle_l2173_217382

-- Define the triangle ABC
def Triangle (A B C : Point) : Prop := sorry

-- Define the angle measure in degrees
def AngleMeasure (A B C : Point) : ℝ := sorry

-- Define the foot of the perpendicular
def PerpendicularFoot (A D : Point) (B C : Point) : Prop := sorry

-- Define the center of the circumscribed circle
def CircumcenterOfTriangle (O A B C : Point) : Prop := sorry

-- Define the diameter of a circle
def DiameterOfCircle (A E O : Point) : Prop := sorry

theorem angle_DAE_in_special_triangle 
  (A B C D E O : Point) 
  (triangle_ABC : Triangle A B C)
  (angle_ACB : AngleMeasure A C B = 40)
  (angle_CBA : AngleMeasure C B A = 60)
  (D_perpendicular : PerpendicularFoot A D B C)
  (O_circumcenter : CircumcenterOfTriangle O A B C)
  (AE_diameter : DiameterOfCircle A E O) :
  AngleMeasure D A E = 20 := by
sorry

end NUMINAMATH_CALUDE_angle_DAE_in_special_triangle_l2173_217382


namespace NUMINAMATH_CALUDE_function_increasing_condition_l2173_217332

theorem function_increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 2 ∧ x₂ > 2 →
    let f := fun x => x^2 - 2*a*x + 3
    (f x₁ - f x₂) / (x₁ - x₂) > 0) →
  a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_function_increasing_condition_l2173_217332


namespace NUMINAMATH_CALUDE_dog_shampoo_count_l2173_217338

def clean_time : ℕ := 55
def hose_time : ℕ := 10
def shampoo_time : ℕ := 15

theorem dog_shampoo_count : 
  ∃ n : ℕ, n * shampoo_time + hose_time = clean_time ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_dog_shampoo_count_l2173_217338


namespace NUMINAMATH_CALUDE_time_to_cut_kids_hair_l2173_217388

/-- Proves that the time to cut a kid's hair is 25 minutes given the specified conditions --/
theorem time_to_cut_kids_hair (
  time_woman : ℕ)
  (time_man : ℕ)
  (num_women : ℕ)
  (num_men : ℕ)
  (num_children : ℕ)
  (total_time : ℕ)
  (h1 : time_woman = 50)
  (h2 : time_man = 15)
  (h3 : num_women = 3)
  (h4 : num_men = 2)
  (h5 : num_children = 3)
  (h6 : total_time = 255)
  (h7 : total_time = time_woman * num_women + time_man * num_men + num_children * (total_time - time_woman * num_women - time_man * num_men) / num_children) :
  (total_time - time_woman * num_women - time_man * num_men) / num_children = 25 := by
  sorry

end NUMINAMATH_CALUDE_time_to_cut_kids_hair_l2173_217388


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l2173_217370

theorem cube_sum_theorem (p q r : ℝ) 
  (h1 : p + q + r = 4)
  (h2 : p * q + q * r + r * p = 6)
  (h3 : p * q * r = -8) :
  p^3 + q^3 + r^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l2173_217370


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2173_217379

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottomRadius : ℝ
  topRadius : ℝ
  sphereRadius : ℝ
  isTangent : Bool

/-- The theorem stating the radius of the sphere in the given problem -/
theorem sphere_radius_in_truncated_cone 
  (cone : TruncatedConeWithSphere)
  (h1 : cone.bottomRadius = 24)
  (h2 : cone.topRadius = 6)
  (h3 : cone.isTangent = true) :
  cone.sphereRadius = 12 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2173_217379


namespace NUMINAMATH_CALUDE_means_of_reciprocals_of_first_four_primes_l2173_217300

def first_four_primes : List Nat := [2, 3, 5, 7]

def reciprocals (lst : List Nat) : List Rat :=
  lst.map (λ x => 1 / x)

def arithmetic_mean (lst : List Rat) : Rat :=
  lst.sum / lst.length

def harmonic_mean (lst : List Rat) : Rat :=
  lst.length / (lst.map (λ x => 1 / x)).sum

theorem means_of_reciprocals_of_first_four_primes :
  let recip := reciprocals first_four_primes
  arithmetic_mean recip = 247 / 840 ∧
  harmonic_mean recip = 4 / 17 := by
  sorry

#eval arithmetic_mean (reciprocals first_four_primes)
#eval harmonic_mean (reciprocals first_four_primes)

end NUMINAMATH_CALUDE_means_of_reciprocals_of_first_four_primes_l2173_217300


namespace NUMINAMATH_CALUDE_stratified_sample_grade12_l2173_217343

/-- Represents the number of students in each grade and in the sample -/
structure SchoolSample where
  total : ℕ
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ
  sample10 : ℕ
  sample12 : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem stratified_sample_grade12 (s : SchoolSample) 
  (h_total : s.total = 1290)
  (h_grade10 : s.grade10 = 480)
  (h_grade_diff : s.grade11 = s.grade12 + 30)
  (h_sum : s.grade10 + s.grade11 + s.grade12 = s.total)
  (h_sample10 : s.sample10 = 96)
  (h_prop : s.sample10 / s.grade10 = s.sample12 / s.grade12) :
  s.sample12 = 78 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_grade12_l2173_217343


namespace NUMINAMATH_CALUDE_platform_length_calculation_l2173_217361

/-- Calculates the length of a platform given train parameters -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 300 →
  time_platform = 39 →
  time_pole = 18 →
  ∃ platform_length : ℝ,
    (platform_length > 350.12 ∧ platform_length < 350.14) ∧
    platform_length = train_length * (time_platform / time_pole - 1) :=
by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_platform_length_calculation_l2173_217361


namespace NUMINAMATH_CALUDE_distribute_five_balls_two_boxes_l2173_217311

/-- The number of ways to distribute n indistinguishable balls into 2 indistinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ :=
  (n + 1) / 2 + 1

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_five_balls_two_boxes : distribute_balls 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_two_boxes_l2173_217311


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2173_217353

theorem triangle_angle_sum (X Y Z : ℝ) (h1 : X + Y = 80) (h2 : X + Y + Z = 180) : Z = 100 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2173_217353


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2173_217312

-- Define the sets M and N
def M : Set ℝ := {x | 2/x < 1}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2173_217312


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2173_217313

/-- Proves that a train 165 meters long, running at 54 kmph, takes 59 seconds to cross a bridge 720 meters in length. -/
theorem train_bridge_crossing_time :
  let train_length : ℝ := 165
  let bridge_length : ℝ := 720
  let train_speed_kmph : ℝ := 54
  let train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
  let total_distance : ℝ := train_length + bridge_length
  let crossing_time : ℝ := total_distance / train_speed_mps
  crossing_time = 59 := by sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2173_217313


namespace NUMINAMATH_CALUDE_rational_power_difference_integer_implies_integer_l2173_217329

/-- Given distinct positive rational numbers a and b, if a^n - b^n is a positive integer
    for infinitely many positive integers n, then a and b are positive integers. -/
theorem rational_power_difference_integer_implies_integer
  (a b : ℚ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_distinct : a ≠ b)
  (h_infinite : ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, ∃ k : ℤ, k > 0 ∧ a^n - b^n = k) :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ a = m ∧ b = n :=
sorry

end NUMINAMATH_CALUDE_rational_power_difference_integer_implies_integer_l2173_217329


namespace NUMINAMATH_CALUDE_hexagonal_prism_square_pyramid_edge_lengths_l2173_217308

/-- Represents a regular hexagonal prism -/
structure HexagonalPrism where
  edge_length : ℝ
  total_edge_length : ℝ
  edge_count : ℕ := 18
  h_total_edge : total_edge_length = edge_length * edge_count

/-- Represents a square pyramid -/
structure SquarePyramid where
  edge_length : ℝ
  total_edge_length : ℝ
  edge_count : ℕ := 8
  h_total_edge : total_edge_length = edge_length * edge_count

/-- Theorem stating the relationship between the total edge lengths of a hexagonal prism and a square pyramid with the same edge length -/
theorem hexagonal_prism_square_pyramid_edge_lengths 
  (h : HexagonalPrism) (p : SquarePyramid) 
  (h_same_edge : h.edge_length = p.edge_length) 
  (h_total_81 : h.total_edge_length = 81) : 
  p.total_edge_length = 36 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_prism_square_pyramid_edge_lengths_l2173_217308


namespace NUMINAMATH_CALUDE_wood_per_sack_l2173_217385

/-- Given that 4 sacks were filled with a total of 80 pieces of wood,
    prove that each sack contains 20 pieces of wood. -/
theorem wood_per_sack (total_wood : ℕ) (num_sacks : ℕ) 
  (h1 : total_wood = 80) (h2 : num_sacks = 4) :
  total_wood / num_sacks = 20 := by
  sorry

end NUMINAMATH_CALUDE_wood_per_sack_l2173_217385


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l2173_217369

-- Define the conditions p and q
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l2173_217369


namespace NUMINAMATH_CALUDE_base_5_divisible_by_7_l2173_217342

def base_5_to_10 (d : ℕ) : ℕ := 3 * 5^3 + d * 5^2 + d * 5 + 4

theorem base_5_divisible_by_7 :
  ∃! d : ℕ, d < 5 ∧ (base_5_to_10 d) % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_5_divisible_by_7_l2173_217342


namespace NUMINAMATH_CALUDE_find_y_l2173_217365

theorem find_y : ∃ y : ℕ, (12^3 * 6^4) / y = 5184 ∧ y = 432 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2173_217365


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l2173_217336

/-- Given a polynomial P(x) = P(0) + P(1)x + P(2)x^2 where P(-2) = 4,
    prove that P(x) = (4x^2 - 6x) / 7 -/
theorem polynomial_uniqueness (P : ℝ → ℝ) (h1 : ∀ x, P x = P 0 + P 1 * x + P 2 * x^2) 
    (h2 : P (-2) = 4) : 
  ∀ x, P x = (4 * x^2 - 6 * x) / 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l2173_217336


namespace NUMINAMATH_CALUDE_remainder_sum_reverse_order_l2173_217302

theorem remainder_sum_reverse_order (n : ℕ) : 
  n % 12 = 56 → n % 34 = 78 → (n % 34) % 12 + n % 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_reverse_order_l2173_217302


namespace NUMINAMATH_CALUDE_dishwasher_manager_wage_ratio_l2173_217390

/-- Proves that the ratio of dishwasher's wage to manager's wage is 0.5 -/
theorem dishwasher_manager_wage_ratio :
  ∀ (dishwasher_wage chef_wage manager_wage : ℝ),
  manager_wage = 7.5 →
  chef_wage = manager_wage - 3 →
  chef_wage = dishwasher_wage * 1.2 →
  dishwasher_wage / manager_wage = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_dishwasher_manager_wage_ratio_l2173_217390


namespace NUMINAMATH_CALUDE_quadrilateral_properties_l2173_217330

/-- Definition of a quadrilateral with given vertices -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Function to calculate the intersection point of diagonals -/
def diagonalIntersection (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Function to calculate the area of a quadrilateral -/
def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating the properties of the given quadrilateral -/
theorem quadrilateral_properties :
  let q := Quadrilateral.mk (5, 6) (-1, 2) (-2, -1) (4, -5)
  diagonalIntersection q = (-1/6, 5/6) ∧
  quadrilateralArea q = 42 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_properties_l2173_217330


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2173_217350

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + z) * Complex.I = 1 - Complex.I → z = -2 - Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2173_217350


namespace NUMINAMATH_CALUDE_distinct_tower_heights_l2173_217377

/-- Represents the number of bricks in the tower. -/
def num_bricks : ℕ := 50

/-- Represents the minimum possible height of the tower in inches. -/
def min_height : ℕ := 250

/-- Represents the maximum possible height of the tower in inches. -/
def max_height : ℕ := 900

/-- The theorem stating the number of distinct tower heights achievable. -/
theorem distinct_tower_heights :
  ∃ (heights : Finset ℕ),
    (∀ h ∈ heights, min_height ≤ h ∧ h ≤ max_height) ∧
    (∀ h, min_height ≤ h → h ≤ max_height →
      (∃ (a b c : ℕ), a + b + c = num_bricks ∧ 5*a + 12*b + 18*c = h) ↔ h ∈ heights) ∧
    heights.card = 651 := by
  sorry

end NUMINAMATH_CALUDE_distinct_tower_heights_l2173_217377


namespace NUMINAMATH_CALUDE_triangle_properties_l2173_217352

/-- Given a triangle ABC with the specified properties, prove the cosine of angle B and the perimeter. -/
theorem triangle_properties (A B C : ℝ) (AB BC AC : ℝ) : 
  C = 2 * A →
  Real.cos A = 3 / 4 →
  2 * (AB * BC * Real.cos B) = -27 →
  AB = 6 →
  BC = 4 →
  AC = 5 →
  Real.cos B = 9 / 16 ∧ AB + BC + AC = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2173_217352


namespace NUMINAMATH_CALUDE_probability_two_red_apples_l2173_217305

def total_apples : ℕ := 10
def red_apples : ℕ := 6
def green_apples : ℕ := 4
def chosen_apples : ℕ := 3

theorem probability_two_red_apples :
  (Nat.choose red_apples 2 * Nat.choose green_apples 1) / Nat.choose total_apples chosen_apples = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_apples_l2173_217305


namespace NUMINAMATH_CALUDE_batsman_average_l2173_217340

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) : 
  total_innings = 17 → 
  last_innings_score = 85 → 
  average_increase = 3 → 
  (((total_innings - 1) * ((total_innings * (37 - average_increase)) / total_innings) + last_innings_score) / total_innings : ℚ) = 37 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l2173_217340


namespace NUMINAMATH_CALUDE_window_probability_l2173_217395

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

def probability_BIRD : ℚ := 1 / (choose 4 2)
def probability_WINDS : ℚ := 3 / (choose 5 3)
def probability_FLOW : ℚ := 1 / (choose 4 2)

theorem window_probability : 
  probability_BIRD * probability_WINDS * probability_FLOW = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_window_probability_l2173_217395


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l2173_217364

theorem absolute_value_inequality_solution :
  {y : ℝ | 3 ≤ |y - 4| ∧ |y - 4| ≤ 7} = {y : ℝ | (7 ≤ y ∧ y ≤ 11) ∨ (-3 ≤ y ∧ y ≤ 1)} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l2173_217364


namespace NUMINAMATH_CALUDE_check_amount_proof_l2173_217375

theorem check_amount_proof :
  ∃! (x y : ℕ), 
    y ≤ 99 ∧
    (y : ℚ) + (x : ℚ) / 100 - 5 / 100 = 2 * ((x : ℚ) + (y : ℚ) / 100) ∧
    x = 31 ∧ y = 63 := by
  sorry

end NUMINAMATH_CALUDE_check_amount_proof_l2173_217375


namespace NUMINAMATH_CALUDE_third_year_afforestation_l2173_217337

/-- Represents the yearly afforestation area -/
def afforestation (n : ℕ) : ℝ :=
  match n with
  | 0 => 10000  -- Initial afforestation
  | m + 1 => afforestation m * 1.2  -- 20% increase each year

/-- Theorem stating the area afforested in the third year -/
theorem third_year_afforestation :
  afforestation 2 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_third_year_afforestation_l2173_217337


namespace NUMINAMATH_CALUDE_not_both_perfect_squares_l2173_217381

/-- For any natural numbers x and y, at least one of x^2 + y + 1 or y^2 + 4x + 3 is not a perfect square. -/
theorem not_both_perfect_squares (x y : ℕ) : 
  ¬(∃ a b : ℕ, (x^2 + y + 1 = a^2) ∧ (y^2 + 4*x + 3 = b^2)) := by
  sorry

end NUMINAMATH_CALUDE_not_both_perfect_squares_l2173_217381


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2173_217326

theorem inequality_solution_set (a : ℝ) :
  (∀ x, (a - x) * (x - 1) < 0 ↔ 
    (a > 1 ∧ (x > a ∨ x < 1)) ∨
    (a < 1 ∧ (x > 1 ∨ x < a)) ∨
    (a = 1 ∧ x ≠ 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2173_217326


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l2173_217317

theorem president_vice_president_selection (n : ℕ) (h : n = 8) : 
  (n * (n - 1) : ℕ) = 56 := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l2173_217317


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l2173_217347

theorem triangle_angle_ratio (A B C : ℝ) : 
  A = 60 → B = 80 → A + B + C = 180 → B / C = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l2173_217347


namespace NUMINAMATH_CALUDE_scientific_notation_properties_l2173_217349

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : Float
  exponent : Int

/-- Counts the number of significant figures in a scientific notation number -/
def count_significant_figures (n : ScientificNotation) : Nat :=
  sorry

/-- Determines the place value of the last significant digit -/
def last_significant_place (n : ScientificNotation) : String :=
  sorry

/-- The main theorem -/
theorem scientific_notation_properties :
  let n : ScientificNotation := { coefficient := 6.30, exponent := 5 }
  count_significant_figures n = 3 ∧
  last_significant_place n = "ten thousand's place" :=
  sorry

end NUMINAMATH_CALUDE_scientific_notation_properties_l2173_217349


namespace NUMINAMATH_CALUDE_car_distance_calculation_l2173_217319

/-- The distance a car needs to cover given initial time and new speed requirements -/
theorem car_distance_calculation (initial_time : ℝ) (new_speed : ℝ) : 
  initial_time = 6 →
  new_speed = 36 →
  (new_speed * (3/2 * initial_time)) = 324 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_calculation_l2173_217319


namespace NUMINAMATH_CALUDE_composite_expressions_l2173_217368

theorem composite_expressions (p : ℕ) (hp : Nat.Prime p) : 
  (¬ Nat.Prime (p^2 + 35)) ∧ (¬ Nat.Prime (p^2 + 55)) := by
  sorry

end NUMINAMATH_CALUDE_composite_expressions_l2173_217368


namespace NUMINAMATH_CALUDE_equation_solution_l2173_217301

theorem equation_solution : ∃! y : ℚ, 4 * (5 * y + 3) - 3 = -3 * (2 - 8 * y) ∧ y = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2173_217301


namespace NUMINAMATH_CALUDE_bicycle_separation_l2173_217344

/-- Adam's speed in miles per hour -/
def adam_speed : ℝ := 12

/-- Simon's speed in miles per hour -/
def simon_speed : ℝ := 16

/-- Time in hours after which Adam and Simon are 100 miles apart -/
def separation_time : ℝ := 5

/-- Distance between Adam and Simon after separation_time hours -/
def separation_distance : ℝ := 100

theorem bicycle_separation :
  let adam_distance := adam_speed * separation_time
  let simon_distance := simon_speed * separation_time
  (adam_distance ^ 2 + simon_distance ^ 2 : ℝ) = separation_distance ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_separation_l2173_217344


namespace NUMINAMATH_CALUDE_middle_term_binomial_coefficient_l2173_217383

theorem middle_term_binomial_coefficient 
  (n : ℕ) 
  (x : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : 2^(n-1) = 1024) : 
  Nat.choose n ((n-1)/2) = 462 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_binomial_coefficient_l2173_217383


namespace NUMINAMATH_CALUDE_slope_of_line_l2173_217335

/-- The slope of a line given by the equation √3x - y + 1 = 0 is √3. -/
theorem slope_of_line (x y : ℝ) : 
  (Real.sqrt 3) * x - y + 1 = 0 → 
  ∃ m : ℝ, m = Real.sqrt 3 ∧ y = m * x + 1 := by
sorry

end NUMINAMATH_CALUDE_slope_of_line_l2173_217335


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l2173_217315

/-- Given a line and a circle with common points, prove the range of the circle's center x-coordinate. -/
theorem circle_line_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) → 
  -3 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l2173_217315


namespace NUMINAMATH_CALUDE_total_money_proof_l2173_217363

/-- Given the money distribution among Cecil, Catherine, and Carmela, 
    prove that their total money is $2800 -/
theorem total_money_proof (cecil_money : ℕ) 
  (h1 : cecil_money = 600)
  (catherine_money : ℕ) 
  (h2 : catherine_money = 2 * cecil_money - 250)
  (carmela_money : ℕ) 
  (h3 : carmela_money = 2 * cecil_money + 50) : 
  cecil_money + catherine_money + carmela_money = 2800 := by
  sorry

end NUMINAMATH_CALUDE_total_money_proof_l2173_217363


namespace NUMINAMATH_CALUDE_fish_left_in_tank_l2173_217304

/-- The number of fish left in Lucy's first tank after moving some to another tank -/
theorem fish_left_in_tank (initial_fish : ℝ) (moved_fish : ℝ) 
  (h1 : initial_fish = 212.0)
  (h2 : moved_fish = 68.0) : 
  initial_fish - moved_fish = 144.0 := by
  sorry

end NUMINAMATH_CALUDE_fish_left_in_tank_l2173_217304


namespace NUMINAMATH_CALUDE_yogurt_combinations_count_l2173_217372

/-- The number of combinations of one item from a set of 4 and two different items from a set of 6 -/
def yogurt_combinations (flavors : Nat) (toppings : Nat) : Nat :=
  flavors * (toppings.choose 2)

/-- Theorem stating that the number of combinations is 60 -/
theorem yogurt_combinations_count :
  yogurt_combinations 4 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_count_l2173_217372


namespace NUMINAMATH_CALUDE_paper_towel_cost_l2173_217387

theorem paper_towel_cost (case_price : ℝ) (num_rolls : ℕ) (savings_percent : ℝ) :
  case_price = 9 →
  num_rolls = 12 →
  savings_percent = 25 →
  ∃ (individual_price : ℝ),
    case_price = (1 - savings_percent / 100) * (num_rolls * individual_price) ∧
    individual_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_paper_towel_cost_l2173_217387


namespace NUMINAMATH_CALUDE_fair_rides_l2173_217345

theorem fair_rides (initial_tickets : ℕ) (spent_tickets : ℕ) (tickets_per_ride : ℕ) 
  (h1 : initial_tickets = 79)
  (h2 : spent_tickets = 23)
  (h3 : tickets_per_ride = 7) :
  (initial_tickets - spent_tickets) / tickets_per_ride = 8 := by
  sorry

end NUMINAMATH_CALUDE_fair_rides_l2173_217345


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2173_217380

theorem purely_imaginary_complex_number (x : ℝ) : 
  (Complex.ofReal (x^2 - 1) + Complex.I * Complex.ofReal (x + 1)).im ≠ 0 ∧
  (Complex.ofReal (x^2 - 1) + Complex.I * Complex.ofReal (x + 1)).re = 0 →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2173_217380


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l2173_217357

/-- The weight of a single kayak in pounds -/
def kayak_weight : ℚ := 32

/-- The number of kayaks -/
def num_kayaks : ℕ := 4

/-- The number of bowling balls -/
def num_bowling_balls : ℕ := 9

/-- The weight of a single bowling ball in pounds -/
def bowling_ball_weight : ℚ := 128 / 9

theorem bowling_ball_weight_proof :
  num_bowling_balls * bowling_ball_weight = num_kayaks * kayak_weight :=
by sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l2173_217357


namespace NUMINAMATH_CALUDE_valid_numbers_count_l2173_217397

/-- Counts the number of valid eight-digit numbers where each digit appears exactly as many times as its value. -/
def count_valid_numbers : ℕ :=
  let single_eight := 1
  let seven_sevens_one_one := 8
  let six_sixes_two_twos := 28
  let five_fives_two_twos_one_one := 168
  let five_fives_three_threes := 56
  let four_fours_three_threes_one_one := 280
  single_eight + seven_sevens_one_one + six_sixes_two_twos + 
  five_fives_two_twos_one_one + five_fives_three_threes + 
  four_fours_three_threes_one_one

theorem valid_numbers_count : count_valid_numbers = 541 := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_count_l2173_217397


namespace NUMINAMATH_CALUDE_shortest_hexpath_distribution_l2173_217316

/-- Represents a direction in the hexagonal grid -/
inductive Direction
| Horizontal
| Diagonal1
| Diagonal2

/-- Represents a path in the hexagonal grid -/
structure HexPath where
  length : ℕ
  horizontal : ℕ
  diagonal1 : ℕ
  diagonal2 : ℕ
  sum_constraint : length = horizontal + diagonal1 + diagonal2

/-- A shortest path in a hexagonal grid -/
def is_shortest_path (path : HexPath) : Prop :=
  path.horizontal = path.diagonal1 + path.diagonal2

theorem shortest_hexpath_distribution (path : HexPath) 
  (h_shortest : is_shortest_path path) (h_length : path.length = 100) :
  path.horizontal = 50 ∧ path.diagonal1 + path.diagonal2 = 50 := by
  sorry

#check shortest_hexpath_distribution

end NUMINAMATH_CALUDE_shortest_hexpath_distribution_l2173_217316


namespace NUMINAMATH_CALUDE_special_ap_ratio_l2173_217367

/-- An arithmetic progression with the property that the sum of its first ten terms
    is four times the sum of its first five terms. -/
structure SpecialAP where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_condition : (10 * a + 45 * d) = 4 * (5 * a + 10 * d)

/-- The ratio of the first term to the common difference in a SpecialAP is 1:2. -/
theorem special_ap_ratio (ap : SpecialAP) : ap.a / ap.d = 1 / 2 := by
  sorry

#check special_ap_ratio

end NUMINAMATH_CALUDE_special_ap_ratio_l2173_217367


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l2173_217331

theorem parallelogram_side_length 
  (s : ℝ) 
  (h_positive : s > 0) 
  (h_angle : Real.cos (π / 3) = 1 / 2) 
  (h_area : (3 * s) * (s * Real.sin (π / 3)) = 27 * Real.sqrt 3) : 
  s = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l2173_217331


namespace NUMINAMATH_CALUDE_flour_measurement_l2173_217321

theorem flour_measurement (required : ℚ) (container : ℚ) (excess : ℚ) : 
  required = 15/4 ∧ container = 4/3 ∧ excess = 2/3 → 
  ∃ (n : ℕ), n * container = required - excess ∧ n = 3 := by
sorry

end NUMINAMATH_CALUDE_flour_measurement_l2173_217321


namespace NUMINAMATH_CALUDE_mango_rate_per_kg_l2173_217356

/-- The rate per kg for mangoes given the purchase details --/
theorem mango_rate_per_kg (grape_quantity grape_rate mango_quantity total_payment : ℕ) : 
  grape_quantity = 9 →
  grape_rate = 70 →
  mango_quantity = 9 →
  total_payment = 1125 →
  (total_payment - grape_quantity * grape_rate) / mango_quantity = 55 := by
sorry

end NUMINAMATH_CALUDE_mango_rate_per_kg_l2173_217356


namespace NUMINAMATH_CALUDE_expression_evaluation_l2173_217360

theorem expression_evaluation (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x > y) (hyz : y > z) :
  (x^z * y^x * z^y) / (z^z * y^y * x^x) = x^(z-x) * y^(x-y) * z^(y-z) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2173_217360


namespace NUMINAMATH_CALUDE_four_number_average_l2173_217354

theorem four_number_average (a b c d : ℝ) 
  (h1 : b + c + d = 24)
  (h2 : a + c + d = 36)
  (h3 : a + b + d = 28)
  (h4 : a + b + c = 32) :
  (a + b + c + d) / 4 = 10 := by
sorry

end NUMINAMATH_CALUDE_four_number_average_l2173_217354


namespace NUMINAMATH_CALUDE_function_range_and_inequality_l2173_217376

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp x * Real.sin x

theorem function_range_and_inequality (e : ℝ) (π : ℝ) :
  (∀ x ∈ Set.Icc 0 (π / 2), f x ∈ Set.Icc 0 1) ∧
  (∃ k : ℝ, k = Real.exp (π / 2) / (π / 2 - 1) ∧
    ∀ x ∈ Set.Icc 0 (π / 2), f x ≥ k * (x - 1) * (1 - Real.sin x) ∧
    ∀ k' > k, ∃ x ∈ Set.Icc 0 (π / 2), f x < k' * (x - 1) * (1 - Real.sin x)) :=
by sorry

end NUMINAMATH_CALUDE_function_range_and_inequality_l2173_217376


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2173_217348

theorem min_value_of_expression (x y : ℝ) :
  Real.sqrt (2 * x^2 - 6 * x + 5) + Real.sqrt (y^2 - 4 * y + 5) + Real.sqrt (2 * x^2 - 2 * x * y + y^2) ≥ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2173_217348


namespace NUMINAMATH_CALUDE_max_regular_hours_is_40_l2173_217303

/-- Calculates the maximum number of regular hours worked given total pay, overtime hours, and pay rates. -/
def max_regular_hours (total_pay : ℚ) (overtime_hours : ℚ) (regular_rate : ℚ) : ℚ :=
  let overtime_rate := 2 * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let regular_pay := total_pay - overtime_pay
  regular_pay / regular_rate

/-- Proves that given the specified conditions, the maximum number of regular hours is 40. -/
theorem max_regular_hours_is_40 :
  max_regular_hours 168 8 3 = 40 := by
  sorry

#eval max_regular_hours 168 8 3

end NUMINAMATH_CALUDE_max_regular_hours_is_40_l2173_217303


namespace NUMINAMATH_CALUDE_west_representation_l2173_217310

-- Define a type for direction
inductive Direction
  | East
  | West

-- Define a function to represent distance with direction
def representDistance (dir : Direction) (distance : ℝ) : ℝ :=
  match dir with
  | Direction.East => distance
  | Direction.West => -distance

-- State the theorem
theorem west_representation :
  representDistance Direction.East 80 = 80 →
  representDistance Direction.West 200 = -200 :=
by sorry

end NUMINAMATH_CALUDE_west_representation_l2173_217310


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2173_217334

/-- Represents the total number of students in a school -/
def total_students : ℕ := 3500 + 1500

/-- Represents the number of middle school students -/
def middle_school_students : ℕ := 1500

/-- Represents the number of students sampled from the middle school stratum -/
def middle_school_sample : ℕ := 30

/-- Calculates the total sample size in a stratified sampling -/
def total_sample_size : ℕ := (middle_school_sample * total_students) / middle_school_students

theorem stratified_sampling_theorem :
  total_sample_size = 100 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2173_217334


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l2173_217323

theorem alpha_beta_sum (α β : ℝ) 
  (h1 : α^3 - 3*α^2 + 5*α - 4 = 0)
  (h2 : β^3 - 3*β^2 + 5*β - 2 = 0) : 
  α + β = 2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l2173_217323


namespace NUMINAMATH_CALUDE_card_probability_l2173_217359

theorem card_probability (diamonds hearts : ℕ) (a : ℕ) :
  diamonds = 3 →
  hearts = 2 →
  (a : ℚ) / (a + diamonds + hearts) = 1 / 2 →
  a = 5 := by
  sorry

end NUMINAMATH_CALUDE_card_probability_l2173_217359


namespace NUMINAMATH_CALUDE_problem_solution_l2173_217327

theorem problem_solution (x : ℝ) : ((12 * x - 20) + (x / 2)) / 7 = 15 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2173_217327


namespace NUMINAMATH_CALUDE_event_probability_l2173_217384

theorem event_probability (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →
  (1 - p)^3 = 1 - 63/64 →
  3 * p * (1 - p)^2 = 9/64 :=
by sorry

end NUMINAMATH_CALUDE_event_probability_l2173_217384


namespace NUMINAMATH_CALUDE_zero_in_interval_l2173_217346

theorem zero_in_interval :
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ Real.log c - 6 + 2 * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2173_217346


namespace NUMINAMATH_CALUDE_max_d_value_l2173_217374

def a (n : ℕ+) : ℕ := 100 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ k : ℕ+, d k = 401) ∧ (∀ n : ℕ+, d n ≤ 401) := by
  sorry

end NUMINAMATH_CALUDE_max_d_value_l2173_217374


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2173_217339

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2173_217339


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l2173_217306

/-- Given a quadratic equation (2√3 + √2)x² + 2(√3 + √2)x + (√2 - 2√3) = 0,
    prove that the sum of its roots is -(4 + √6)/5 and the product of its roots is (2√6 - 7)/5 -/
theorem quadratic_roots_sum_and_product :
  let a : ℝ := 2 * Real.sqrt 3 + Real.sqrt 2
  let b : ℝ := 2 * (Real.sqrt 3 + Real.sqrt 2)
  let c : ℝ := Real.sqrt 2 - 2 * Real.sqrt 3
  (-(b / a) = -(4 + Real.sqrt 6) / 5) ∧ 
  (c / a = (2 * Real.sqrt 6 - 7) / 5) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l2173_217306


namespace NUMINAMATH_CALUDE_railway_distances_l2173_217389

theorem railway_distances (total_distance : ℝ) 
  (moscow_mozhaysk_ratio : ℝ) (mozhaysk_vyazma_ratio : ℝ) 
  (vyazma_smolensk_ratio : ℝ) :
  total_distance = 415 ∧ 
  moscow_mozhaysk_ratio = 7/9 ∧ 
  mozhaysk_vyazma_ratio = 27/35 →
  ∃ (moscow_mozhaysk vyazma_smolensk mozhaysk_vyazma : ℝ),
    moscow_mozhaysk = 105 ∧
    mozhaysk_vyazma = 135 ∧
    vyazma_smolensk = 175 ∧
    moscow_mozhaysk + mozhaysk_vyazma + vyazma_smolensk = total_distance ∧
    moscow_mozhaysk = moscow_mozhaysk_ratio * mozhaysk_vyazma ∧
    mozhaysk_vyazma = mozhaysk_vyazma_ratio * vyazma_smolensk :=
by sorry

end NUMINAMATH_CALUDE_railway_distances_l2173_217389


namespace NUMINAMATH_CALUDE_arbelos_external_tangent_l2173_217314

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an arbelos configuration -/
structure Arbelos where
  A : Point
  B : Point
  C : Point
  D : Point
  M : Point
  N : Point
  O₁ : Point
  O₂ : Point
  smallCircle1 : Circle
  smallCircle2 : Circle
  largeCircle : Circle

/-- Checks if a line is tangent to a circle -/
def isTangent (p1 p2 : Point) (c : Circle) : Prop :=
  sorry

/-- Angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Main theorem: MN is a common external tangent to the small circles of the arbelos -/
theorem arbelos_external_tangent (arb : Arbelos) (α : ℝ) 
    (h1 : angle arb.B arb.A arb.D = α)
    (h2 : arb.smallCircle1.center = arb.O₁)
    (h3 : arb.smallCircle2.center = arb.O₂) :
  isTangent arb.M arb.N arb.smallCircle1 ∧ isTangent arb.M arb.N arb.smallCircle2 :=
by sorry

end NUMINAMATH_CALUDE_arbelos_external_tangent_l2173_217314


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l2173_217358

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : 
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l2173_217358


namespace NUMINAMATH_CALUDE_count_rearranged_even_numbers_l2173_217393

/-- The number of different even numbers that can be formed by rearranging the digits of 124669 -/
def rearrangedEvenNumbers : ℕ := 240

/-- The original number -/
def originalNumber : ℕ := 124669

/-- Theorem stating that the number of different even numbers formed by rearranging the digits of 124669 is 240 -/
theorem count_rearranged_even_numbers :
  rearrangedEvenNumbers = 240 ∧ originalNumber ≠ rearrangedEvenNumbers :=
by sorry

end NUMINAMATH_CALUDE_count_rearranged_even_numbers_l2173_217393


namespace NUMINAMATH_CALUDE_necessary_condition_range_l2173_217392

theorem necessary_condition_range (a : ℝ) : 
  (∀ x : ℝ, x < a + 2 → x ≤ 2) → a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_necessary_condition_range_l2173_217392


namespace NUMINAMATH_CALUDE_min_distance_between_ellipses_l2173_217396

/-- The minimum distance between two ellipses -/
theorem min_distance_between_ellipses :
  let ellipse1 := {(x, y) : ℝ × ℝ | x^2 / 4 + y^2 = 1}
  let ellipse2 := {(x, y) : ℝ × ℝ | (x - 1)^2 / 9 + y^2 / 9 = 1}
  (∃ (A B : ℝ × ℝ), A ∈ ellipse1 ∧ B ∈ ellipse2 ∧
    ∀ (C D : ℝ × ℝ), C ∈ ellipse1 → D ∈ ellipse2 →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)) ∧
  (∀ (A B : ℝ × ℝ), A ∈ ellipse1 → B ∈ ellipse2 →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ 2) ∧
  (∃ (A B : ℝ × ℝ), A ∈ ellipse1 ∧ B ∈ ellipse2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_ellipses_l2173_217396


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2173_217362

theorem quadratic_inequality_solution_set (m : ℝ) (h : m * (m - 1) < 0) :
  {x : ℝ | x^2 - (m + 1/m) * x + 1 < 0} = {x : ℝ | m < x ∧ x < 1/m} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2173_217362


namespace NUMINAMATH_CALUDE_prob_at_least_one_female_l2173_217309

/-- The probability of selecting at least one female student when randomly choosing 2 students
    from a group of 3 males and 1 female is equal to 1/2. -/
theorem prob_at_least_one_female (total_students : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (team_size : ℕ) (h1 : total_students = male_students + female_students) 
  (h2 : total_students = 4) (h3 : male_students = 3) (h4 : female_students = 1) (h5 : team_size = 2) :
  1 - (Nat.choose male_students team_size : ℚ) / (Nat.choose total_students team_size : ℚ) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_female_l2173_217309


namespace NUMINAMATH_CALUDE_circus_tent_capacity_l2173_217386

/-- The number of sections in the circus tent -/
def num_sections : ℕ := 4

/-- The capacity of each section in the circus tent -/
def section_capacity : ℕ := 246

/-- The total capacity of the circus tent -/
def total_capacity : ℕ := num_sections * section_capacity

theorem circus_tent_capacity : total_capacity = 984 := by
  sorry

end NUMINAMATH_CALUDE_circus_tent_capacity_l2173_217386


namespace NUMINAMATH_CALUDE_line_moved_down_l2173_217324

/-- Given a line with equation y = -3x + 5, prove that moving it down 3 units
    results in the line with equation y = -3x + 2 -/
theorem line_moved_down (x y : ℝ) :
  (y = -3 * x + 5) → (y - 3 = -3 * x + 2) := by sorry

end NUMINAMATH_CALUDE_line_moved_down_l2173_217324


namespace NUMINAMATH_CALUDE_polynomial_real_root_l2173_217351

/-- The polynomial p(x) = x^6 + bx^4 - x^3 + bx^2 + 1 -/
def p (b : ℝ) (x : ℝ) : ℝ := x^6 + b*x^4 - x^3 + b*x^2 + 1

/-- The theorem stating the condition for the polynomial to have at least one real root -/
theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, p b x = 0) ↔ b ≤ -3/2 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l2173_217351


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2173_217373

theorem sufficient_but_not_necessary (p q : Prop) 
  (h : (¬p → ¬q) ∧ ¬(¬q → ¬p)) : 
  (q → p) ∧ ¬(p → q) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2173_217373
