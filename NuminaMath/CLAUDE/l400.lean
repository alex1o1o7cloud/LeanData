import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l400_40093

theorem problem_solution (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 2) (h3 : c^2 / a = 3) : 
  a = 12^(1/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l400_40093


namespace NUMINAMATH_CALUDE_cheaper_call_rate_l400_40030

/-- China Mobile's promotion factor -/
def china_mobile_promotion : ℚ := 130 / 100

/-- China Telecom's promotion factor -/
def china_telecom_promotion : ℚ := 100 / 40

/-- China Mobile's standard call rate (yuan per minute) -/
def china_mobile_standard_rate : ℚ := 26 / 100

/-- China Telecom's standard call rate (yuan per minute) -/
def china_telecom_standard_rate : ℚ := 30 / 100

/-- China Mobile's actual call rate (yuan per minute) -/
def china_mobile_actual_rate : ℚ := china_mobile_standard_rate / china_mobile_promotion

/-- China Telecom's actual call rate (yuan per minute) -/
def china_telecom_actual_rate : ℚ := china_telecom_standard_rate / china_telecom_promotion

theorem cheaper_call_rate :
  china_telecom_actual_rate < china_mobile_actual_rate ∧
  china_mobile_actual_rate - china_telecom_actual_rate = 8 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cheaper_call_rate_l400_40030


namespace NUMINAMATH_CALUDE_problem_solution_l400_40038

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x) + 2, Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem problem_solution (A : ℝ) (b c : ℝ) (h1 : 0 ≤ A ∧ A ≤ π/4) 
  (h2 : f A = 4) (h3 : b = 1) (h4 : 1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) :
  (∀ x ∈ Set.Icc 0 (π/4), f x ≤ 5 ∧ 4 ≤ f x) ∧ 
  c^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l400_40038


namespace NUMINAMATH_CALUDE_unique_a_value_l400_40014

-- Define the sets M and N as functions of a
def M (a : ℝ) : Set ℝ := {1, 2, a^2 - 3*a - 1}
def N (a : ℝ) : Set ℝ := {-1, a, 3}

-- State the theorem
theorem unique_a_value : ∃! a : ℝ, (M a ∩ N a = {3} ∧ a ≠ -1) := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l400_40014


namespace NUMINAMATH_CALUDE_worker_speed_reduction_l400_40045

theorem worker_speed_reduction (usual_time : ℕ) (delay : ℕ) : 
  usual_time = 60 → delay = 12 → 
  (usual_time : ℚ) / (usual_time + delay) = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_worker_speed_reduction_l400_40045


namespace NUMINAMATH_CALUDE_roosters_count_l400_40066

theorem roosters_count (total_chickens egg_laying_hens non_egg_laying_hens : ℕ) 
  (h1 : total_chickens = 325)
  (h2 : egg_laying_hens = 277)
  (h3 : non_egg_laying_hens = 20) :
  total_chickens - (egg_laying_hens + non_egg_laying_hens) = 28 :=
by sorry

end NUMINAMATH_CALUDE_roosters_count_l400_40066


namespace NUMINAMATH_CALUDE_problem_statement_l400_40042

theorem problem_statement (x n f : ℝ) : 
  x = (3 + Real.sqrt 8)^500 →
  n = ⌊x⌋ →
  f = x - n →
  x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l400_40042


namespace NUMINAMATH_CALUDE_library_average_MB_per_hour_l400_40072

/-- Calculates the average megabytes per hour of music in a digital library -/
def averageMBPerHour (days : ℕ) (totalMB : ℕ) : ℕ :=
  let hoursPerDay : ℕ := 24
  let totalHours : ℕ := days * hoursPerDay
  let exactAverage : ℚ := totalMB / totalHours
  (exactAverage + 1/2).floor.toNat

/-- Proves that the average megabytes per hour for the given library is 67 MB -/
theorem library_average_MB_per_hour :
  averageMBPerHour 15 24000 = 67 := by
  sorry

end NUMINAMATH_CALUDE_library_average_MB_per_hour_l400_40072


namespace NUMINAMATH_CALUDE_sin_50_sin_70_minus_cos_50_sin_20_l400_40055

open Real

theorem sin_50_sin_70_minus_cos_50_sin_20 :
  sin (50 * π / 180) * sin (70 * π / 180) - cos (50 * π / 180) * sin (20 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_sin_70_minus_cos_50_sin_20_l400_40055


namespace NUMINAMATH_CALUDE_movie_theater_revenue_is_6600_l400_40071

/-- Calculates the total revenue of a movie theater given ticket prices and quantities sold --/
def movie_theater_revenue (matinee_price evening_price three_d_price : ℕ) 
                          (matinee_sold evening_sold three_d_sold : ℕ) : ℕ :=
  matinee_price * matinee_sold + evening_price * evening_sold + three_d_price * three_d_sold

/-- Theorem stating that the movie theater's revenue is $6600 given the specified prices and quantities --/
theorem movie_theater_revenue_is_6600 :
  movie_theater_revenue 5 12 20 200 300 100 = 6600 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_revenue_is_6600_l400_40071


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_capacity_l400_40081

/-- The number of people that can ride a Ferris wheel simultaneously -/
def ferris_wheel_capacity (num_seats : ℕ) (people_per_seat : ℕ) : ℕ :=
  num_seats * people_per_seat

/-- Theorem: A Ferris wheel with 14 seats, each holding 6 people, can accommodate 84 people -/
theorem paradise_park_ferris_wheel_capacity :
  ferris_wheel_capacity 14 6 = 84 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_capacity_l400_40081


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l400_40087

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (∀ x, x < -1 → x^2 - 1 > 0) ∧ 
  (∃ x, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l400_40087


namespace NUMINAMATH_CALUDE_circle_center_transformation_l400_40084

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (-3, 4)
  let reflected := reflect_x initial_center
  let final_center := translate_up reflected 5
  final_center = (-3, 1) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l400_40084


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l400_40040

theorem divisibility_equivalence (n m : ℤ) : 
  (∃ k : ℤ, 2*n + 5*m = 9*k) ↔ (∃ l : ℤ, 5*n + 8*m = 9*l) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l400_40040


namespace NUMINAMATH_CALUDE_smoothie_mix_amount_l400_40024

/-- The amount of smoothie mix in ounces per packet -/
def smoothie_mix_per_packet (total_smoothies : ℕ) (smoothie_size : ℕ) (total_packets : ℕ) : ℚ :=
  (total_smoothies * smoothie_size : ℚ) / total_packets

theorem smoothie_mix_amount : 
  smoothie_mix_per_packet 150 12 180 = 10 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_mix_amount_l400_40024


namespace NUMINAMATH_CALUDE_distance_between_foci_l400_40041

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 45 + y^2 / 5 = 9

-- Theorem statement
theorem distance_between_foci :
  ∃ (a b c : ℝ), 
    (∀ x y, ellipse_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    2 * c = 12 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_distance_between_foci_l400_40041


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l400_40023

theorem consecutive_integers_sum (n : ℕ) (h : n > 0) :
  (6 * n + 15 = 2013) → (n + 5 = 338) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l400_40023


namespace NUMINAMATH_CALUDE_second_discount_percentage_l400_40021

theorem second_discount_percentage
  (initial_price : ℝ)
  (first_discount : ℝ)
  (final_price : ℝ)
  (h1 : initial_price = 400)
  (h2 : first_discount = 25)
  (h3 : final_price = 240)
  : ∃ (second_discount : ℝ),
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l400_40021


namespace NUMINAMATH_CALUDE_unique_sequence_coefficients_l400_40058

/-- Given two distinct roots of a characteristic equation and two initial terms of a sequence,
    there exists a unique pair of coefficients that generates the entire sequence. -/
theorem unique_sequence_coefficients
  (x₁ x₂ : ℝ) (a₀ a₁ : ℝ) (h : x₁ ≠ x₂) :
  ∃! (c₁ c₂ : ℝ), ∀ (n : ℕ), c₁ * x₁^n + c₂ * x₂^n = 
    if n = 0 then a₀ else if n = 1 then a₁ else c₁ * x₁^n + c₂ * x₂^n :=
sorry

end NUMINAMATH_CALUDE_unique_sequence_coefficients_l400_40058


namespace NUMINAMATH_CALUDE_remaining_cheese_calories_l400_40006

/-- Calculates the remaining calories in a block of cheese after a portion is removed -/
theorem remaining_cheese_calories (length width height : ℝ) 
  (calorie_density : ℝ) (eaten_side_length : ℝ) : 
  length = 4 → width = 8 → height = 2 → calorie_density = 110 → eaten_side_length = 2 →
  (length * width * height - eaten_side_length ^ 3) * calorie_density = 6160 := by
  sorry

#check remaining_cheese_calories

end NUMINAMATH_CALUDE_remaining_cheese_calories_l400_40006


namespace NUMINAMATH_CALUDE_angie_tax_payment_l400_40068

/-- Represents Angie's monthly finances -/
structure AngieFinances where
  salary : ℕ
  necessities : ℕ
  leftOver : ℕ

/-- Calculates Angie's tax payment based on her finances -/
def taxPayment (finances : AngieFinances) : ℕ :=
  finances.salary - finances.necessities - finances.leftOver

/-- Theorem stating that Angie's tax payment is $20 given her financial situation -/
theorem angie_tax_payment :
  let finances : AngieFinances := { salary := 80, necessities := 42, leftOver := 18 }
  taxPayment finances = 20 := by
  sorry


end NUMINAMATH_CALUDE_angie_tax_payment_l400_40068


namespace NUMINAMATH_CALUDE_minimum_cost_theorem_l400_40092

/-- Represents the number and cost of diesel generators --/
structure DieselGenerators where
  totalCount : Nat
  typeACount : Nat
  typeBCount : Nat
  typeCCount : Nat
  typeACost : Nat
  typeBCost : Nat
  typeCCost : Nat

/-- Represents the irrigation capacity of the generators --/
def irrigationCapacity (g : DieselGenerators) : Nat :=
  4 * g.typeACount + 3 * g.typeBCount + 2 * g.typeCCount

/-- Represents the total cost of operating the generators --/
def operatingCost (g : DieselGenerators) : Nat :=
  g.typeACost * g.typeACount + g.typeBCost * g.typeBCount + g.typeCCost * g.typeCCount

/-- Theorem stating the minimum cost of operation --/
theorem minimum_cost_theorem (g : DieselGenerators) :
  g.totalCount = 10 ∧
  g.typeACount > 0 ∧ g.typeBCount > 0 ∧ g.typeCCount > 0 ∧
  g.typeACount + g.typeBCount + g.typeCCount = g.totalCount ∧
  irrigationCapacity g = 32 ∧
  g.typeACost = 130 ∧ g.typeBCost = 120 ∧ g.typeCCost = 100 →
  ∃ (minCost : Nat), minCost = 1190 ∧
    ∀ (h : DieselGenerators), 
      h.totalCount = 10 ∧
      h.typeACount > 0 ∧ h.typeBCount > 0 ∧ h.typeCCount > 0 ∧
      h.typeACount + h.typeBCount + h.typeCCount = h.totalCount ∧
      irrigationCapacity h = 32 ∧
      h.typeACost = 130 ∧ h.typeBCost = 120 ∧ h.typeCCost = 100 →
      operatingCost h ≥ minCost := by
  sorry

end NUMINAMATH_CALUDE_minimum_cost_theorem_l400_40092


namespace NUMINAMATH_CALUDE_cosine_inequality_l400_40091

theorem cosine_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) : 
  Real.cos (1 + a) < Real.cos (1 - a) := by
sorry

end NUMINAMATH_CALUDE_cosine_inequality_l400_40091


namespace NUMINAMATH_CALUDE_service_center_location_l400_40017

/-- Represents a highway with exits and a service center -/
structure Highway where
  third_exit : ℝ
  seventh_exit : ℝ
  twelfth_exit : ℝ
  service_center : ℝ

/-- Theorem stating the location of the service center on the highway -/
theorem service_center_location (h : Highway) 
  (h_third : h.third_exit = 30)
  (h_seventh : h.seventh_exit = 90)
  (h_twelfth : h.twelfth_exit = 195)
  (h_service : h.service_center = h.third_exit + 2/3 * (h.seventh_exit - h.third_exit)) :
  h.service_center = 70 := by
  sorry

#check service_center_location

end NUMINAMATH_CALUDE_service_center_location_l400_40017


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l400_40082

-- Problem 1
theorem problem_1 : (-36 : ℚ) * (5/4 - 5/6 - 11/12) = 18 := by sorry

-- Problem 2
theorem problem_2 : (-2)^2 - 3 * (-1)^3 + 0 * (-2)^3 = 7 := by sorry

-- Problem 3
theorem problem_3 (x y : ℚ) (hx : x = -2) (hy : y = 1/2) :
  3 * x^2 * y - 2 * x * y^2 - 3/2 * (x^2 * y - 2 * x * y^2) = 5/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l400_40082


namespace NUMINAMATH_CALUDE_linear_inequality_solution_l400_40039

theorem linear_inequality_solution (x : ℝ) : (2 * x - 1 ≥ 3) ↔ (x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_linear_inequality_solution_l400_40039


namespace NUMINAMATH_CALUDE_positive_real_inequality_l400_40051

theorem positive_real_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l400_40051


namespace NUMINAMATH_CALUDE_special_number_exists_l400_40096

theorem special_number_exists : ∃ n : ℕ+, 
  (Nat.digits 10 n.val).length = 1000 ∧ 
  0 ∉ Nat.digits 10 n.val ∧
  ∃ pairs : List (ℕ × ℕ), 
    pairs.length = 500 ∧
    (pairs.map (λ p => p.1 * p.2)).sum ∣ n.val ∧
    ∀ d ∈ Nat.digits 10 n.val, ∃ p ∈ pairs, d = p.1 ∨ d = p.2 :=
by sorry

end NUMINAMATH_CALUDE_special_number_exists_l400_40096


namespace NUMINAMATH_CALUDE_frog_jump_distance_l400_40085

/-- The jumping contest problem -/
theorem frog_jump_distance 
  (grasshopper_jump : ℕ) 
  (grasshopper_frog_diff : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : grasshopper_jump = grasshopper_frog_diff + frog_jump) :
  frog_jump = 15 :=
by
  sorry

#check frog_jump_distance

end NUMINAMATH_CALUDE_frog_jump_distance_l400_40085


namespace NUMINAMATH_CALUDE_namjoon_candies_l400_40004

/-- The number of candies Namjoon gave to Yoongi -/
def candies_given : ℕ := 18

/-- The number of candies left over -/
def candies_left : ℕ := 16

/-- The total number of candies Namjoon had in the beginning -/
def total_candies : ℕ := candies_given + candies_left

theorem namjoon_candies : total_candies = 34 := by
  sorry

end NUMINAMATH_CALUDE_namjoon_candies_l400_40004


namespace NUMINAMATH_CALUDE_simplify_expression_l400_40013

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^2 = 327680 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l400_40013


namespace NUMINAMATH_CALUDE_final_eraser_count_l400_40020

def initial_erasers : Float := 95.0
def bought_erasers : Float := 42.0

theorem final_eraser_count :
  initial_erasers + bought_erasers = 137.0 := by
  sorry

end NUMINAMATH_CALUDE_final_eraser_count_l400_40020


namespace NUMINAMATH_CALUDE_total_groups_is_1026_l400_40002

/-- The number of boys in the class -/
def num_boys : ℕ := 9

/-- The number of girls in the class -/
def num_girls : ℕ := 12

/-- The size of each group -/
def group_size : ℕ := 3

/-- Calculate the number of combinations of n items taken k at a time -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.choose n k

/-- Calculate the number of groups with 2 boys and 1 girl -/
def groups_2boys1girl : ℕ :=
  combinations num_boys 2 * combinations num_girls 1

/-- Calculate the number of groups with 2 girls and 1 boy -/
def groups_2girls1boy : ℕ :=
  combinations num_girls 2 * combinations num_boys 1

/-- The total number of possible groups -/
def total_groups : ℕ :=
  groups_2boys1girl + groups_2girls1boy

/-- Theorem stating that the total number of possible groups is 1026 -/
theorem total_groups_is_1026 : total_groups = 1026 := by
  sorry

end NUMINAMATH_CALUDE_total_groups_is_1026_l400_40002


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l400_40053

/-- Represents a student with an ID number -/
structure Student where
  id : Nat
  deriving Repr

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Lottery
  deriving Repr

/-- Checks if a number is divisible by 5 -/
def isDivisibleByFive (n : Nat) : Bool :=
  n % 5 == 0

/-- Selects students whose IDs are divisible by 5 -/
def selectStudents (students : List Student) : List Student :=
  students.filter (fun s => isDivisibleByFive s.id)

/-- Theorem: Selecting students with IDs divisible by 5 from a group of 60 students
    numbered 1 to 60 is an example of systematic sampling -/
theorem systematic_sampling_proof (students : List Student) 
    (h1 : students.length = 60)
    (h2 : ∀ i, 1 ≤ i ∧ i ≤ 60 → ∃ s ∈ students, s.id = i)
    (h3 : ∀ s ∈ students, 1 ≤ s.id ∧ s.id ≤ 60) :
    (selectStudents students).length = 12 ∧ 
    SamplingMethod.Systematic = 
      (match (selectStudents students) with
       | [] => SamplingMethod.SimpleRandom  -- Default case, should not occur
       | _ => SamplingMethod.Systematic) := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l400_40053


namespace NUMINAMATH_CALUDE_money_ratio_problem_l400_40016

/-- Given the ratio of money between Ravi and Giri, and the amounts of money
    Ravi and Kiran have, prove the ratio of money between Giri and Kiran. -/
theorem money_ratio_problem (ravi_giri_ratio : ℚ) (ravi_money kiran_money : ℕ) :
  ravi_giri_ratio = 6 / 7 →
  ravi_money = 36 →
  kiran_money = 105 →
  ∃ (giri_money : ℕ), 
    (ravi_money : ℚ) / giri_money = ravi_giri_ratio ∧
    (giri_money : ℚ) / kiran_money = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_money_ratio_problem_l400_40016


namespace NUMINAMATH_CALUDE_roots_difference_squared_l400_40052

theorem roots_difference_squared (p q r s : ℝ) : 
  (r^2 - p*r + q = 0) → (s^2 - p*s + q = 0) → (r - s)^2 = p^2 - 4*q := by
  sorry

end NUMINAMATH_CALUDE_roots_difference_squared_l400_40052


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l400_40064

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem parallel_vectors_m_value :
  ∀ m : ℝ, (∃ k : ℝ, vector_a = k • (vector_b m)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l400_40064


namespace NUMINAMATH_CALUDE_min_value_problem_l400_40062

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z * (x + y + z) = 1) : 
  (x + y) * (y + z) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l400_40062


namespace NUMINAMATH_CALUDE_two_plus_three_equals_eight_is_proposition_l400_40070

/-- A statement is a proposition if it can be judged as either true or false. -/
def is_proposition (statement : Prop) : Prop :=
  (statement ∨ ¬statement) ∧ ¬(statement ∧ ¬statement)

/-- The statement "2 + 3 = 8" is a proposition. -/
theorem two_plus_three_equals_eight_is_proposition :
  is_proposition (2 + 3 = 8) := by
  sorry

end NUMINAMATH_CALUDE_two_plus_three_equals_eight_is_proposition_l400_40070


namespace NUMINAMATH_CALUDE_pen_difference_after_four_weeks_l400_40078

/-- The difference in pens between Alex and Jane after 4 weeks -/
def pen_difference (A B : ℕ) (X Y : ℝ) (M N : ℕ) : ℕ :=
  M - N

/-- Theorem stating the difference in pens after 4 weeks -/
theorem pen_difference_after_four_weeks 
  (A B : ℕ) (X Y : ℝ) (M N : ℕ) 
  (hM : M = A * X^4) 
  (hN : N = B * Y^4) :
  pen_difference A B X Y M N = M - N :=
by
  sorry

end NUMINAMATH_CALUDE_pen_difference_after_four_weeks_l400_40078


namespace NUMINAMATH_CALUDE_min_value_F_l400_40009

/-- The function F(x, y) -/
def F (x y : ℝ) : ℝ := 6*y + 8*x - 9

/-- The constraint equation -/
def constraint (x y : ℝ) : Prop := x^2 + y^2 + 25 = 10*(x + y)

/-- Theorem stating that the minimum value of F(x, y) is 11 given the constraint -/
theorem min_value_F :
  ∃ (min : ℝ), min = 11 ∧
  (∀ x y : ℝ, constraint x y → F x y ≥ min) ∧
  (∃ x y : ℝ, constraint x y ∧ F x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_F_l400_40009


namespace NUMINAMATH_CALUDE_dog_groupings_count_l400_40067

/-- The number of ways to divide 12 dogs into three groups -/
def dog_groupings : ℕ :=
  let total_dogs : ℕ := 12
  let group1_size : ℕ := 4  -- Fluffy's group
  let group2_size : ℕ := 5  -- Nipper's group
  let group3_size : ℕ := 3
  let remaining_dogs : ℕ := total_dogs - 2  -- Excluding Fluffy and Nipper
  (Nat.choose remaining_dogs (group1_size - 1)) * (Nat.choose (remaining_dogs - (group1_size - 1)) (group2_size - 1))

/-- Theorem stating the number of ways to divide the dogs is 4200 -/
theorem dog_groupings_count : dog_groupings = 4200 := by
  sorry

end NUMINAMATH_CALUDE_dog_groupings_count_l400_40067


namespace NUMINAMATH_CALUDE_expression_simplification_l400_40010

theorem expression_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ((x + 1) / (x^2 - 4) - 1 / (x + 2)) / (3 / (x - 2)) = 1 / (x + 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l400_40010


namespace NUMINAMATH_CALUDE_min_area_ratio_l400_40069

-- Define the triangles
structure EquilateralTriangle :=
  (A B C : ℝ × ℝ)

structure RightTriangle :=
  (D E F : ℝ × ℝ)

-- Define the conditions
def inscribed (rt : RightTriangle) (et : EquilateralTriangle) : Prop :=
  sorry

def right_angle (rt : RightTriangle) : Prop :=
  sorry

def angle_edf_30 (rt : RightTriangle) : Prop :=
  sorry

-- Define the area ratio
def area_ratio (rt : RightTriangle) (et : EquilateralTriangle) : ℝ :=
  sorry

-- Theorem statement
theorem min_area_ratio 
  (et : EquilateralTriangle) 
  (rt : RightTriangle) 
  (h1 : inscribed rt et) 
  (h2 : right_angle rt) 
  (h3 : angle_edf_30 rt) :
  ∃ (min_ratio : ℝ), 
    (∀ (rt' : RightTriangle), inscribed rt' et → right_angle rt' → angle_edf_30 rt' → 
      area_ratio rt' et ≥ min_ratio) ∧ 
    min_ratio = 3/14 :=
  sorry

end NUMINAMATH_CALUDE_min_area_ratio_l400_40069


namespace NUMINAMATH_CALUDE_same_solution_implies_b_value_l400_40022

theorem same_solution_implies_b_value :
  ∀ (x b : ℚ),
  (3 * x + 9 = 0) ∧ (b * x + 15 = 5) →
  b = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_b_value_l400_40022


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l400_40074

/-- The minimum number of additional coins needed -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (friends * (friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for Alex -/
theorem alex_coin_distribution (friends : ℕ) (initial_coins : ℕ) 
  (h1 : friends = 15) (h2 : initial_coins = 100) : 
  min_additional_coins friends initial_coins = 20 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l400_40074


namespace NUMINAMATH_CALUDE_pentagon_reassembly_l400_40003

/-- Given a 10x15 rectangle cut into two congruent pentagons and reassembled into a larger rectangle,
    prove that one-third of the longer side of the new rectangle is 5√2. -/
theorem pentagon_reassembly (original_length original_width : ℝ) 
                            (new_length new_width : ℝ) (y : ℝ) : 
  original_length = 10 →
  original_width = 15 →
  new_length * new_width = original_length * original_width →
  y = new_length / 3 →
  y = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_reassembly_l400_40003


namespace NUMINAMATH_CALUDE_ellipse_m_value_l400_40018

/-- Given an ellipse with equation x²/4 + y²/m = 1, foci on the x-axis, 
    and eccentricity 1/2, prove that m = 3 -/
theorem ellipse_m_value (m : ℝ) 
  (h1 : ∀ (x y : ℝ), x^2/4 + y^2/m = 1 → (∃ (a b c : ℝ), a^2 = 4 ∧ b^2 = m ∧ c^2 = a^2 - b^2))
  (h2 : ∃ (e : ℝ), e = 1/2 ∧ e^2 = (4 - m)/4) : 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l400_40018


namespace NUMINAMATH_CALUDE_pond_eyes_count_l400_40061

/-- The number of eyes for each frog -/
def frog_eyes : ℕ := 2

/-- The number of eyes for each crocodile -/
def crocodile_eyes : ℕ := 2

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 20

/-- The number of crocodiles in the pond -/
def num_crocodiles : ℕ := 10

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := num_frogs * frog_eyes + num_crocodiles * crocodile_eyes

theorem pond_eyes_count : total_eyes = 60 := by
  sorry

end NUMINAMATH_CALUDE_pond_eyes_count_l400_40061


namespace NUMINAMATH_CALUDE_ten_coin_flips_sequences_l400_40001

/-- The number of distinct sequences when flipping a coin n times -/
def coin_flip_sequences (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of distinct sequences when flipping a coin 10 times is 1024 -/
theorem ten_coin_flips_sequences : coin_flip_sequences 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_ten_coin_flips_sequences_l400_40001


namespace NUMINAMATH_CALUDE_exactly_fifteen_numbers_l400_40015

/-- Represents a three-digit positive integer in base 10 -/
def ThreeDigitInteger (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : ℕ :=
  sorry

/-- Converts a natural number to its base-8 representation -/
def toBase8 (n : ℕ) : ℕ :=
  sorry

/-- Checks if the two rightmost digits of two numbers are the same -/
def sameLastTwoDigits (a b : ℕ) : Prop :=
  a % 100 = b % 100

/-- The main theorem stating that there are exactly 15 numbers satisfying the condition -/
theorem exactly_fifteen_numbers :
  ∃! (s : Finset ℕ),
    Finset.card s = 15 ∧
    ∀ n, n ∈ s ↔ 
      ThreeDigitInteger n ∧
      sameLastTwoDigits (toBase7 n * toBase8 n) (3 * n) :=
  sorry


end NUMINAMATH_CALUDE_exactly_fifteen_numbers_l400_40015


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l400_40090

theorem geometric_sequence_sixth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^7 = 2) : 
  a * r^5 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l400_40090


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l400_40098

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ((2 * x - 1) / (x - 2) - 1) / ((x + 1) / (x^2 - 4)) = x + 2 :=
by sorry

theorem expression_evaluation_at_3 :
  let x : ℝ := 3
  ((2 * x - 1) / (x - 2) - 1) / ((x + 1) / (x^2 - 4)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l400_40098


namespace NUMINAMATH_CALUDE_sum_of_numbers_l400_40073

theorem sum_of_numbers : 4321 + 3214 + 2143 - 1432 = 8246 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l400_40073


namespace NUMINAMATH_CALUDE_barium_oxide_moles_l400_40065

-- Define the chemical reaction
structure Reaction where
  bao : ℝ    -- moles of Barium oxide
  h2o : ℝ    -- moles of Water
  baoh2 : ℝ  -- moles of Barium hydroxide

-- Define the reaction conditions
def reaction_conditions (r : Reaction) : Prop :=
  r.h2o = 1 ∧ r.baoh2 = r.bao

-- Theorem statement
theorem barium_oxide_moles (e : ℝ) :
  ∀ r : Reaction, reaction_conditions r → r.baoh2 = e → r.bao = e :=
by
  sorry

end NUMINAMATH_CALUDE_barium_oxide_moles_l400_40065


namespace NUMINAMATH_CALUDE_smallest_integer_cubic_inequality_l400_40031

theorem smallest_integer_cubic_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^3 - 12*m^2 + 44*m - 48 ≤ 0 → n ≤ m) ∧ 
  (n^3 - 12*n^2 + 44*n - 48 ≤ 0) ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_cubic_inequality_l400_40031


namespace NUMINAMATH_CALUDE_thousand_chime_date_l400_40057

/-- Represents a date --/
structure Date :=
  (year : Nat)
  (month : Nat)
  (day : Nat)

/-- Represents a time --/
structure Time :=
  (hour : Nat)
  (minute : Nat)

/-- Represents the chiming pattern of the clock --/
def clockChime (hour : Nat) (minute : Nat) : Nat :=
  if minute == 30 then 1
  else if minute == 0 then (if hour == 0 || hour == 12 then 12 else hour)
  else 0

/-- Calculates the number of chimes from a given start date and time to a given end date and time --/
def countChimes (startDate : Date) (startTime : Time) (endDate : Date) (endTime : Time) : Nat :=
  sorry -- Implementation details omitted

/-- The theorem to be proved --/
theorem thousand_chime_date :
  let startDate := Date.mk 2003 2 26
  let startTime := Time.mk 10 15
  let endDate := Date.mk 2003 3 7
  countChimes startDate startTime endDate (Time.mk 23 59) ≥ 1000 ∧
  ∀ (d : Date), d.year == 2003 ∧ d.month == 3 ∧ d.day < 7 →
    countChimes startDate startTime d (Time.mk 23 59) < 1000 :=
by sorry

end NUMINAMATH_CALUDE_thousand_chime_date_l400_40057


namespace NUMINAMATH_CALUDE_figure_area_l400_40049

/-- The area of a rectangle given its width and height -/
def rectangle_area (width : ℕ) (height : ℕ) : ℕ := width * height

/-- The total area of three rectangles -/
def total_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ) : ℕ :=
  rectangle_area rect1_width rect1_height +
  rectangle_area rect2_width rect2_height +
  rectangle_area rect3_width rect3_height

/-- Theorem: The total area of the figure is 71 square units -/
theorem figure_area :
  total_area 7 7 3 2 4 4 = 71 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_l400_40049


namespace NUMINAMATH_CALUDE_inequality_proof_l400_40099

theorem inequality_proof (e : ℝ) (h : e > 0) : 
  (1 : ℝ) / e > Real.log ((1 + e^2) / e^2) ∧ 
  Real.log ((1 + e^2) / e^2) > 1 / (1 + e^2) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l400_40099


namespace NUMINAMATH_CALUDE_min_rental_cost_is_2860_l400_40088

/-- Represents a rental plan for cars --/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a rental plan is valid for transporting the given amount of goods --/
def isValidPlan (plan : RentalPlan) (totalGoods : ℕ) : Prop :=
  3 * plan.typeA + 4 * plan.typeB = totalGoods

/-- Calculates the rental cost for a given plan --/
def rentalCost (plan : RentalPlan) : ℕ :=
  300 * plan.typeA + 320 * plan.typeB

/-- Theorem stating that the minimum rental cost to transport 35 tons of goods is 2860 yuan --/
theorem min_rental_cost_is_2860 :
  ∃ (plan : RentalPlan),
    isValidPlan plan 35 ∧
    rentalCost plan = 2860 ∧
    ∀ (otherPlan : RentalPlan), isValidPlan otherPlan 35 → rentalCost plan ≤ rentalCost otherPlan :=
sorry

end NUMINAMATH_CALUDE_min_rental_cost_is_2860_l400_40088


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l400_40044

theorem contrapositive_equivalence (a b : ℝ) :
  (((a + b = 1) → (a^2 + b^2 ≥ 1/2)) ↔ ((a^2 + b^2 < 1/2) → (a + b ≠ 1))) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l400_40044


namespace NUMINAMATH_CALUDE_annies_ride_distance_l400_40089

/-- Represents the taxi fare structure -/
structure TaxiFare where
  startFee : ℝ
  perMileFee : ℝ
  tollFee : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (fare : TaxiFare) (miles : ℝ) : ℝ :=
  fare.startFee + fare.tollFee + fare.perMileFee * miles

theorem annies_ride_distance (mikeFare annieFare : TaxiFare) 
  (h1 : mikeFare.startFee = 2.5)
  (h2 : mikeFare.perMileFee = 0.25)
  (h3 : mikeFare.tollFee = 0)
  (h4 : annieFare.startFee = 2.5)
  (h5 : annieFare.perMileFee = 0.25)
  (h6 : annieFare.tollFee = 5)
  (h7 : totalFare mikeFare 36 = totalFare annieFare (annies_miles : ℝ)) :
  annies_miles = 16 := by
  sorry

end NUMINAMATH_CALUDE_annies_ride_distance_l400_40089


namespace NUMINAMATH_CALUDE_jordan_scoring_breakdown_l400_40080

/-- Represents the scoring statistics of a basketball player in a game. -/
structure ScoringStats where
  total_points : ℕ
  total_shots : ℕ
  total_hits : ℕ
  three_pointers_made : ℕ
  three_pointer_attempts : ℕ

/-- Calculates the number of 2-point shots and free throws made given scoring statistics. -/
def calculate_shots (stats : ScoringStats) : ℕ × ℕ := sorry

/-- Theorem stating that given Jordan's scoring statistics, he made 8 2-point shots and 3 free throws. -/
theorem jordan_scoring_breakdown (stats : ScoringStats) 
  (h1 : stats.total_points = 28)
  (h2 : stats.total_shots = 24)
  (h3 : stats.total_hits = 14)
  (h4 : stats.three_pointers_made = 3)
  (h5 : stats.three_pointer_attempts = 3) :
  calculate_shots stats = (8, 3) := by sorry

end NUMINAMATH_CALUDE_jordan_scoring_breakdown_l400_40080


namespace NUMINAMATH_CALUDE_fraction_division_addition_l400_40005

theorem fraction_division_addition : (3 / 7 : ℚ) / 4 + 1 / 2 = 17 / 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_addition_l400_40005


namespace NUMINAMATH_CALUDE_interval_representation_l400_40000

def open_closed_interval (a b : ℝ) : Set ℝ := {x | a < x ∧ x ≤ b}

theorem interval_representation :
  open_closed_interval (-3) 2 = {x : ℝ | -3 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_interval_representation_l400_40000


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l400_40036

/-- Given a line segment CD with midpoint M(5,4) and one endpoint C(7,-2),
    the sum of the coordinates of the other endpoint D is 13. -/
theorem midpoint_coordinate_sum :
  ∀ (D : ℝ × ℝ),
  (5, 4) = ((7, -2) + D) / 2 →
  D.1 + D.2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l400_40036


namespace NUMINAMATH_CALUDE_sequence_general_term_l400_40035

theorem sequence_general_term (a : ℕ → ℝ) :
  (a 1 = 1) →
  (∀ n : ℕ, n > 1 → a n = 2 * a (n - 1) + 1) →
  (∀ n : ℕ, n > 0 → a n = 2^n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l400_40035


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_condition_l400_40012

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_plane_condition
  (m n : Line) (α β : Plane)
  (h1 : parallel m n)
  (h2 : perpendicular_line_plane n β)
  (h3 : parallel_plane α β) :
  perpendicular_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_condition_l400_40012


namespace NUMINAMATH_CALUDE_chess_group_players_l400_40026

theorem chess_group_players (n : ℕ) : 
  (∀ i j : Fin n, i ≠ j → ∃! game : ℕ, game ≤ 36) →  -- Each player plays each other once
  (∀ game : ℕ, game ≤ 36 → ∃! i j : Fin n, i ≠ j) →  -- Each game is played by two distinct players
  (Nat.choose n 2 = 36) →                            -- Total number of games is 36
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_chess_group_players_l400_40026


namespace NUMINAMATH_CALUDE_hydras_always_live_l400_40059

/-- Represents the number of new heads a hydra can grow in a week -/
inductive NewHeads
  | five : NewHeads
  | seven : NewHeads

/-- The state of the hydras after a certain number of weeks -/
structure HydraState where
  weeks : ℕ
  totalHeads : ℕ

/-- The initial state of the hydras -/
def initialState : HydraState :=
  { weeks := 0, totalHeads := 2016 + 2017 }

/-- The change in total heads after one week -/
def weeklyChange (a b : NewHeads) : ℕ :=
  match a, b with
  | NewHeads.five, NewHeads.five => 6
  | NewHeads.five, NewHeads.seven => 8
  | NewHeads.seven, NewHeads.five => 8
  | NewHeads.seven, NewHeads.seven => 10

/-- The state transition function -/
def nextState (state : HydraState) (a b : NewHeads) : HydraState :=
  { weeks := state.weeks + 1
  , totalHeads := state.totalHeads + weeklyChange a b }

theorem hydras_always_live :
  ∀ (state : HydraState), state.totalHeads % 2 = 1 →
    ∀ (a b : NewHeads), (nextState state a b).totalHeads % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hydras_always_live_l400_40059


namespace NUMINAMATH_CALUDE_simplify_expression_l400_40034

theorem simplify_expression (x : ℝ) : (3 * x + 20) + (97 * x + 30) = 100 * x + 50 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l400_40034


namespace NUMINAMATH_CALUDE_frequency_calculation_l400_40028

theorem frequency_calculation (sample_size : ℕ) (area_percentage : ℚ) (h1 : sample_size = 50) (h2 : area_percentage = 16/100) :
  (sample_size : ℚ) * area_percentage = 8 := by
  sorry

end NUMINAMATH_CALUDE_frequency_calculation_l400_40028


namespace NUMINAMATH_CALUDE_p_or_q_necessary_not_sufficient_l400_40060

theorem p_or_q_necessary_not_sufficient (p q : Prop) :
  (¬¬p → (p ∨ q)) ∧ ¬((p ∨ q) → ¬¬p) := by
  sorry

end NUMINAMATH_CALUDE_p_or_q_necessary_not_sufficient_l400_40060


namespace NUMINAMATH_CALUDE_no_fixed_point_for_h_h_condition_l400_40050

-- Define the function h
def h (x : ℝ) : ℝ := x - 6

-- Theorem statement
theorem no_fixed_point_for_h : ¬ ∃ x : ℝ, h x = x := by
  sorry

-- Condition from the original problem
theorem h_condition (x : ℝ) : h (3 * x + 2) = 3 * x - 4 := by
  sorry

end NUMINAMATH_CALUDE_no_fixed_point_for_h_h_condition_l400_40050


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_perimeter_is_22_l400_40008

/-- An isosceles triangle with two sides of length 9 and one side of length 4 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∀ a b c : ℝ,
      a = 9 ∧ b = 9 ∧ c = 4 →
      a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
      a = b →  -- Isosceles condition
      perimeter = a + b + c

/-- The perimeter of the isosceles triangle is 22 -/
theorem perimeter_is_22 : isosceles_triangle_perimeter 22 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_perimeter_is_22_l400_40008


namespace NUMINAMATH_CALUDE_pipe_length_problem_l400_40097

theorem pipe_length_problem (total_length : ℝ) (short_length : ℝ) (long_length : ℝ) : 
  total_length = 177 →
  long_length = 2 * short_length →
  total_length = short_length + long_length →
  long_length = 118 := by
sorry

end NUMINAMATH_CALUDE_pipe_length_problem_l400_40097


namespace NUMINAMATH_CALUDE_sum_of_constants_l400_40095

/-- Given a function y(x) = a + b/x, where a and b are constants,
    prove that a + b = -34 if y(-2) = 2 and y(-4) = 8 -/
theorem sum_of_constants (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 2 ↔ x = -2) ∧ (a + b / x = 8 ↔ x = -4)) →
  a + b = -34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_constants_l400_40095


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_line_equation_l400_40079

/-- A line that passes through a point and forms an isosceles right triangle with coordinate axes -/
structure IsoscelesRightTriangleLine where
  -- The slope of the line
  slope : ℝ
  -- The y-intercept of the line
  y_intercept : ℝ
  -- The point that the line passes through
  point : ℝ × ℝ
  -- The line passes through the given point
  point_on_line : y_intercept = point.2 - slope * point.1
  -- The line forms an isosceles right triangle with coordinate axes
  isosceles_right_triangle : slope = 1 ∨ slope = -1

/-- The equation of a line that passes through (2,3) and forms an isosceles right triangle with coordinate axes -/
theorem isosceles_right_triangle_line_equation (l : IsoscelesRightTriangleLine) 
  (h : l.point = (2, 3)) :
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x + y - 5 = 0) ∨
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x - y + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_line_equation_l400_40079


namespace NUMINAMATH_CALUDE_one_third_of_one_fourth_implies_three_tenths_l400_40033

theorem one_third_of_one_fourth_implies_three_tenths (x : ℝ) : 
  (1 / 3) * (1 / 4) * x = 18 → (3 / 10) * x = 64.8 := by
sorry

end NUMINAMATH_CALUDE_one_third_of_one_fourth_implies_three_tenths_l400_40033


namespace NUMINAMATH_CALUDE_distribute_six_balls_four_boxes_l400_40037

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 65 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_six_balls_four_boxes : distribute_balls 6 4 = 65 := by sorry

end NUMINAMATH_CALUDE_distribute_six_balls_four_boxes_l400_40037


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l400_40086

theorem modulus_of_complex_number : Complex.abs (2 / (1 + Complex.I)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l400_40086


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l400_40025

/-- The surface area of a cylinder given its unfolded lateral surface dimensions -/
theorem cylinder_surface_area (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0)
  (h_is_6pi : h = 6 * Real.pi) (w_is_4pi : w = 4 * Real.pi) :
  let r := min (h / (2 * Real.pi)) (w / (2 * Real.pi))
  let surface_area := h * w + 2 * Real.pi * r^2
  surface_area = 24 * Real.pi^2 + 18 * Real.pi ∨
  surface_area = 24 * Real.pi^2 + 8 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l400_40025


namespace NUMINAMATH_CALUDE_parabola_focus_hyperbola_vertex_asymptote_distance_l400_40027

-- Define the parabola
def parabola (a : ℝ) (x y : ℝ) : Prop := x = a * y^2 ∧ a ≠ 0

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 12 - y^2 / 4 = 1

-- Theorem for the focus of the parabola
theorem parabola_focus (a : ℝ) :
  ∃ (x y : ℝ), parabola a x y → (x = 1 / (4 * a) ∧ y = 0) :=
sorry

-- Theorem for the distance from vertex to asymptote of the hyperbola
theorem hyperbola_vertex_asymptote_distance :
  ∃ (d : ℝ), (∀ x y : ℝ, hyperbola x y → d = Real.sqrt 30 / 5) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_hyperbola_vertex_asymptote_distance_l400_40027


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l400_40043

theorem ferris_wheel_capacity (total_capacity : ℕ) (num_seats : ℕ) (people_per_seat : ℕ) 
  (h1 : total_capacity = 4)
  (h2 : num_seats = 2)
  (h3 : people_per_seat * num_seats = total_capacity) :
  people_per_seat = 2 := by
sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l400_40043


namespace NUMINAMATH_CALUDE_dividend_calculation_l400_40048

theorem dividend_calculation (dividend quotient remainder divisor : ℕ) 
  (h1 : divisor = 28)
  (h2 : quotient = 7)
  (h3 : remainder = 11)
  (h4 : dividend = divisor * quotient + remainder) :
  dividend = 207 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l400_40048


namespace NUMINAMATH_CALUDE_distance_foci_to_asymptotes_l400_40007

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci
def foci : Set (ℝ × ℝ) := {(5, 0), (-5, 0)}

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := (3 * x - 4 * y = 0) ∨ (3 * x + 4 * y = 0)

-- Theorem statement
theorem distance_foci_to_asymptotes :
  ∀ (f : ℝ × ℝ) (x y : ℝ),
  f ∈ foci → asymptotes x y →
  ∃ (d : ℝ), d = 3 ∧ d = |3 * f.1 + 4 * f.2| / Real.sqrt 25 :=
sorry

end NUMINAMATH_CALUDE_distance_foci_to_asymptotes_l400_40007


namespace NUMINAMATH_CALUDE_supermarket_spending_l400_40063

theorem supermarket_spending (total : ℚ) : 
  (1/2 : ℚ) * total + (1/3 : ℚ) * total + (1/10 : ℚ) * total + 5 = total →
  total = 75 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l400_40063


namespace NUMINAMATH_CALUDE_sock_probability_theorem_l400_40075

/-- Represents the number of pairs of socks for each color -/
structure SockPairs :=
  (blue : ℕ)
  (red : ℕ)
  (green : ℕ)

/-- Calculates the probability of picking two socks of the same color -/
def probabilitySameColor (pairs : SockPairs) : ℚ :=
  let totalSocks := 2 * (pairs.blue + pairs.red + pairs.green)
  let blueProbability := (2 * pairs.blue * (2 * pairs.blue - 1)) / (totalSocks * (totalSocks - 1))
  let redProbability := (2 * pairs.red * (2 * pairs.red - 1)) / (totalSocks * (totalSocks - 1))
  let greenProbability := (2 * pairs.green * (2 * pairs.green - 1)) / (totalSocks * (totalSocks - 1))
  blueProbability + redProbability + greenProbability

/-- Theorem: The probability of picking two socks of the same color is 77/189 -/
theorem sock_probability_theorem (pairs : SockPairs) 
  (h1 : pairs.blue = 8) 
  (h2 : pairs.red = 4) 
  (h3 : pairs.green = 2) : 
  probabilitySameColor pairs = 77 / 189 := by
  sorry

#eval probabilitySameColor { blue := 8, red := 4, green := 2 }

end NUMINAMATH_CALUDE_sock_probability_theorem_l400_40075


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l400_40094

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticProgression (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A number is a perfect square if it's the square of some natural number. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- The main theorem stating that an arithmetic progression of positive integers
    with at least one perfect square contains infinitely many perfect squares. -/
theorem infinitely_many_perfect_squares
  (a : ℕ → ℕ)
  (h_arith : ArithmeticProgression a)
  (h_positive : ∀ n, a n > 0)
  (h_one_square : ∃ n, IsPerfectSquare (a n)) :
  ∀ k : ℕ, ∃ n > k, IsPerfectSquare (a n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l400_40094


namespace NUMINAMATH_CALUDE_sum_first_last_33_l400_40032

/-- A sequence of ten terms -/
def Sequence := Fin 10 → ℕ

/-- The property that C (the third term) is 7 -/
def third_is_seven (s : Sequence) : Prop := s 2 = 7

/-- The property that the sum of any three consecutive terms is 40 -/
def consecutive_sum_40 (s : Sequence) : Prop :=
  ∀ i, i < 8 → s i + s (i + 1) + s (i + 2) = 40

/-- The main theorem: If C is 7 and the sum of any three consecutive terms is 40,
    then A + J = 33 -/
theorem sum_first_last_33 (s : Sequence) 
  (h1 : third_is_seven s) (h2 : consecutive_sum_40 s) : s 0 + s 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_last_33_l400_40032


namespace NUMINAMATH_CALUDE_prob_three_even_out_of_five_l400_40077

-- Define a fair 20-sided die
def fair_20_sided_die : Finset ℕ := Finset.range 20

-- Define the probability of rolling an even number on a fair 20-sided die
def prob_even (d : Finset ℕ) : ℚ :=
  (d.filter (λ x => x % 2 = 0)).card / d.card

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the number of dice we want to show even
def num_even : ℕ := 3

-- Theorem statement
theorem prob_three_even_out_of_five :
  prob_even fair_20_sided_die = 1/2 →
  (num_dice.choose num_even : ℚ) * (1/2)^num_dice = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_out_of_five_l400_40077


namespace NUMINAMATH_CALUDE_frustum_center_height_for_specific_pyramid_l400_40047

/-- Represents a rectangular pyramid with a parallel cut -/
structure CutPyramid where
  base_length : ℝ
  base_width : ℝ
  height : ℝ
  volume_ratio : ℝ  -- ratio of smaller pyramid to whole pyramid

/-- Calculate the distance from the center of the frustum's circumsphere to the base -/
def frustum_center_height (p : CutPyramid) : ℝ :=
  sorry

/-- The main theorem -/
theorem frustum_center_height_for_specific_pyramid :
  let p : CutPyramid := {
    base_length := 15,
    base_width := 20,
    height := 30,
    volume_ratio := 1/9
  }
  abs (frustum_center_height p - 25.73) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_frustum_center_height_for_specific_pyramid_l400_40047


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l400_40029

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, 2 * x^2 + 1 > 0)) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l400_40029


namespace NUMINAMATH_CALUDE_circle_equation_l400_40046

/-- The standard equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def standard_circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The center of the circle -/
def center : ℝ × ℝ := (-3, 4)

/-- The radius of the circle -/
def radius : ℝ := 2

/-- Theorem: The equation of the circle with center (-3, 4) and radius 2 is (x+3)^2 + (y-4)^2 = 4 -/
theorem circle_equation (x y : ℝ) :
  standard_circle_equation x y center.1 center.2 radius ↔ (x + 3)^2 + (y - 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l400_40046


namespace NUMINAMATH_CALUDE_grocery_store_diet_soda_l400_40083

/-- The number of bottles of diet soda in a grocery store -/
def diet_soda_bottles (regular_soda_bottles : ℕ) (difference : ℕ) : ℕ :=
  regular_soda_bottles - difference

/-- Theorem: The grocery store has 4 bottles of diet soda -/
theorem grocery_store_diet_soda :
  diet_soda_bottles 83 79 = 4 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_diet_soda_l400_40083


namespace NUMINAMATH_CALUDE_elijah_masking_tape_l400_40011

/-- The amount of masking tape needed for Elijah's room -/
def masking_tape_needed (narrow_wall_width : ℕ) (wide_wall_width : ℕ) : ℕ :=
  2 * narrow_wall_width + 2 * wide_wall_width

/-- Theorem: The amount of masking tape needed for Elijah's room is 20 meters -/
theorem elijah_masking_tape : masking_tape_needed 4 6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_elijah_masking_tape_l400_40011


namespace NUMINAMATH_CALUDE_min_value_theorem_l400_40019

theorem min_value_theorem (m n : ℝ) (h1 : 2 * n + m = 4) (h2 : m > 0) (h3 : n > 0) :
  2 / m + 1 / n ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l400_40019


namespace NUMINAMATH_CALUDE_cos_three_pi_fourth_plus_two_alpha_l400_40054

theorem cos_three_pi_fourth_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 8 - α) = 1 / 6) : 
  Real.cos (3 * π / 4 + 2 * α) = 17 / 18 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_fourth_plus_two_alpha_l400_40054


namespace NUMINAMATH_CALUDE_polar_bear_club_time_l400_40056

/-- Represents the time spent in the pool by each person -/
structure PoolTime where
  jerry : ℕ
  elaine : ℕ
  george : ℕ
  kramer : ℕ

/-- Calculates the total time spent in the pool -/
def total_time (pt : PoolTime) : ℕ :=
  pt.jerry + pt.elaine + pt.george + pt.kramer

/-- Theorem stating the total time spent in the pool is 11 minutes -/
theorem polar_bear_club_time : ∃ (pt : PoolTime),
  pt.jerry = 3 ∧
  pt.elaine = 2 * pt.jerry ∧
  pt.george = pt.elaine / 3 ∧
  pt.kramer = 0 ∧
  total_time pt = 11 := by
  sorry

end NUMINAMATH_CALUDE_polar_bear_club_time_l400_40056


namespace NUMINAMATH_CALUDE_inverse_g_at_negative_seven_sixty_four_l400_40076

open Real

noncomputable def g (x : ℝ) : ℝ := (x^5 - 1) / 4

theorem inverse_g_at_negative_seven_sixty_four :
  g⁻¹ (-7/64) = (9/16)^(1/5) :=
by sorry

end NUMINAMATH_CALUDE_inverse_g_at_negative_seven_sixty_four_l400_40076
