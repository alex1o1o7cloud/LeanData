import Mathlib

namespace NUMINAMATH_CALUDE_twentieth_term_of_arithmetic_sequence_l3737_373706

/-- Given an arithmetic sequence with first term 2 and common difference 3,
    prove that the 20th term is 59. -/
theorem twentieth_term_of_arithmetic_sequence :
  let a : ℕ → ℤ := λ n => 2 + 3 * (n - 1)
  a 20 = 59 := by sorry

end NUMINAMATH_CALUDE_twentieth_term_of_arithmetic_sequence_l3737_373706


namespace NUMINAMATH_CALUDE_equation_equivalence_l3737_373756

theorem equation_equivalence (x y : ℝ) : 
  (2 * x - y = 3) ↔ (y = 2 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3737_373756


namespace NUMINAMATH_CALUDE_min_sum_squares_l3737_373720

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + 3*z = 6) :
  x^2 + y^2 + z^2 ≥ 18/7 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3737_373720


namespace NUMINAMATH_CALUDE_fraction_sum_l3737_373709

theorem fraction_sum (a b : ℕ) (h1 : a.Coprime b) (h2 : a > 0) (h3 : b > 0)
  (h4 : (5 : ℚ) / 6 * (a^2 : ℚ) / (b^2 : ℚ) = 2 * (a : ℚ) / (b : ℚ)) :
  a + b = 17 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_l3737_373709


namespace NUMINAMATH_CALUDE_tomato_production_l3737_373721

theorem tomato_production (plant1 plant2 plant3 total : ℕ) : 
  plant1 = 24 →
  plant2 = (plant1 / 2) + 5 →
  plant3 = plant2 + 2 →
  total = plant1 + plant2 + plant3 →
  total = 60 :=
by sorry

end NUMINAMATH_CALUDE_tomato_production_l3737_373721


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l3737_373788

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a + 1}

-- State the theorem
theorem set_operations_and_subset :
  (∃ a : ℝ, C a ⊆ B) →
  (Set.compl (A ∩ B) = {x : ℝ | x < 3 ∨ 6 ≤ x}) ∧
  (Set.compl B ∪ A = {x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ 9 ≤ x}) ∧
  (Set.Icc 2 8 = {a : ℝ | C a ⊆ B}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l3737_373788


namespace NUMINAMATH_CALUDE_log_10_50_between_consecutive_integers_sum_l3737_373711

theorem log_10_50_between_consecutive_integers_sum :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < b ∧ a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_log_10_50_between_consecutive_integers_sum_l3737_373711


namespace NUMINAMATH_CALUDE_brick_weight_l3737_373783

theorem brick_weight : ∃ x : ℝ, x = 2 + x / 3 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_brick_weight_l3737_373783


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3737_373754

/-- Proves that given the conditions of two trains passing each other, the speed of the first train is 72 kmph -/
theorem train_speed_calculation (length_train1 length_train2 speed_train2 time_to_cross : ℝ) 
  (h1 : length_train1 = 380)
  (h2 : length_train2 = 540)
  (h3 : speed_train2 = 36)
  (h4 : time_to_cross = 91.9926405887529)
  : (length_train1 + length_train2) / time_to_cross * 3.6 + speed_train2 = 72 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l3737_373754


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_value_l3737_373795

/-- Given a curve y = ax - ln(x + 1), prove that if its tangent line at (0, 0) is y = 2x, then a = 3 -/
theorem tangent_line_implies_a_value (a : ℝ) : 
  (∀ x, ∃ y, y = a * x - Real.log (x + 1)) →  -- Curve equation
  (∃ m, ∀ x, 2 * x = m * x) →                 -- Tangent line at (0, 0) is y = 2x
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_value_l3737_373795


namespace NUMINAMATH_CALUDE_calculation_proof_l3737_373779

theorem calculation_proof : (-3)^3 + 5^2 - (-2)^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3737_373779


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3737_373731

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) → 
  (3 * q^2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = -22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3737_373731


namespace NUMINAMATH_CALUDE_range_of_a_l3737_373747

def M (a : ℝ) : Set ℝ := { x | -1 < x - a ∧ x - a < 2 }
def N : Set ℝ := { x | x^2 ≥ x }

theorem range_of_a (a : ℝ) : M a ∪ N = Set.univ → a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3737_373747


namespace NUMINAMATH_CALUDE_gcd_g_y_l3737_373748

def g (y : ℤ) : ℤ := (3*y + 5)*(8*y + 1)*(11*y + 3)*(y + 15)

theorem gcd_g_y (y : ℤ) (h : ∃ k : ℤ, y = 4060 * k) : 
  Int.gcd (g y) y = 5 := by sorry

end NUMINAMATH_CALUDE_gcd_g_y_l3737_373748


namespace NUMINAMATH_CALUDE_valid_triples_l3737_373794

def is_valid_triple (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧
  Nat.gcd a (Nat.gcd b c) = 1 ∧
  (a^2 * b) ∣ (a^3 + b^3 + c^3) ∧
  (b^2 * c) ∣ (a^3 + b^3 + c^3) ∧
  (c^2 * a) ∣ (a^3 + b^3 + c^3)

theorem valid_triples :
  ∀ a b c : ℕ, is_valid_triple a b c ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 2 ∧ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_valid_triples_l3737_373794


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l3737_373746

theorem greatest_integer_difference (x y : ℝ) (hx : 5 < x ∧ x < 8) (hy : 8 < y ∧ y < 13) :
  ∃ (n : ℕ), n = 2 ∧ ∀ (m : ℕ), (∃ (a b : ℝ), 5 < a ∧ a < 8 ∧ 8 < b ∧ b < 13 ∧ m = ⌊b - a⌋) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l3737_373746


namespace NUMINAMATH_CALUDE_euclid_schools_count_l3737_373799

theorem euclid_schools_count :
  ∀ (n : ℕ) (andrea_rank beth_rank carla_rank : ℕ),
    -- Each school sends 3 students
    -- Total number of students is 3n
    -- Andrea's score is the median
    andrea_rank = (3 * n + 1) / 2 →
    -- Andrea's score is highest on her team
    andrea_rank < beth_rank →
    andrea_rank < carla_rank →
    -- Beth and Carla's ranks
    beth_rank = 37 →
    carla_rank = 64 →
    -- Each participant received a different score
    andrea_rank ≠ beth_rank ∧ andrea_rank ≠ carla_rank ∧ beth_rank ≠ carla_rank →
    -- Prove that the number of schools is 23
    n = 23 := by
  sorry


end NUMINAMATH_CALUDE_euclid_schools_count_l3737_373799


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l3737_373780

def f (a x : ℝ) : ℝ := |x - a| - 2

theorem solution_set_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, |f a x| < 1 ↔ x ∈ Set.union (Set.Ioo (-2) 0) (Set.Ioo 2 4)) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l3737_373780


namespace NUMINAMATH_CALUDE_signup_theorem_l3737_373761

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of competitions --/
def num_competitions : ℕ := 3

/-- The total number of ways to sign up --/
def total_ways : ℕ := num_competitions ^ num_students

/-- The number of ways to sign up if each event has participants --/
def ways_with_all_events : ℕ := 
  (Nat.choose num_students (num_students - num_competitions)) * (Nat.factorial num_competitions)

theorem signup_theorem : 
  total_ways = 81 ∧ ways_with_all_events = 36 := by
  sorry

end NUMINAMATH_CALUDE_signup_theorem_l3737_373761


namespace NUMINAMATH_CALUDE_modulus_of_z_l3737_373750

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3737_373750


namespace NUMINAMATH_CALUDE_sean_total_spend_l3737_373793

def almond_croissant_price : ℚ := 4.5
def salami_cheese_croissant_price : ℚ := 4.5
def plain_croissant_price : ℚ := 3
def focaccia_price : ℚ := 4
def latte_price : ℚ := 2.5

def almond_croissant_quantity : ℕ := 1
def salami_cheese_croissant_quantity : ℕ := 1
def plain_croissant_quantity : ℕ := 1
def focaccia_quantity : ℕ := 1
def latte_quantity : ℕ := 2

theorem sean_total_spend :
  almond_croissant_price * almond_croissant_quantity +
  salami_cheese_croissant_price * salami_cheese_croissant_quantity +
  plain_croissant_price * plain_croissant_quantity +
  focaccia_price * focaccia_quantity +
  latte_price * latte_quantity = 21 := by
sorry

end NUMINAMATH_CALUDE_sean_total_spend_l3737_373793


namespace NUMINAMATH_CALUDE_welders_problem_l3737_373775

/-- The number of welders initially working on the order -/
def initial_welders : ℕ := 16

/-- The number of days needed to complete the order with the initial number of welders -/
def total_days : ℕ := 8

/-- The number of welders that leave after the first day -/
def leaving_welders : ℕ := 9

/-- The number of additional days needed by the remaining welders to complete the order -/
def additional_days : ℕ := 16

/-- The work completed in one day by all initial welders -/
def daily_work : ℚ := 1 / initial_welders

theorem welders_problem :
  (1 : ℚ) + additional_days * ((initial_welders - leaving_welders : ℚ) * daily_work) = total_days := by
  sorry

#eval initial_welders

end NUMINAMATH_CALUDE_welders_problem_l3737_373775


namespace NUMINAMATH_CALUDE_range_of_a_l3737_373749

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -|p.1| - 2}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = a^2}

theorem range_of_a (a : ℝ) :
  (set_A ∩ set_B a = ∅) ↔ (-2*Real.sqrt 2 - 2 < a ∧ a < 2*Real.sqrt 2 + 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3737_373749


namespace NUMINAMATH_CALUDE_serena_mother_triple_age_l3737_373774

/-- The number of years it will take for the mother to be three times as old as the daughter. -/
def years_until_triple_age (daughter_age : ℕ) (mother_age : ℕ) : ℕ :=
  (mother_age - 3 * daughter_age) / 2

/-- Theorem stating that it will take 6 years for Serena's mother to be three times as old as Serena. -/
theorem serena_mother_triple_age :
  years_until_triple_age 9 39 = 6 := by
  sorry

end NUMINAMATH_CALUDE_serena_mother_triple_age_l3737_373774


namespace NUMINAMATH_CALUDE_sqrt_10_power_identity_l3737_373769

theorem sqrt_10_power_identity : (Real.sqrt 10 + 3)^2023 * (Real.sqrt 10 - 3)^2022 = Real.sqrt 10 + 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_10_power_identity_l3737_373769


namespace NUMINAMATH_CALUDE_little_krish_money_distribution_l3737_373715

-- Define the problem parameters
def initial_amount : ℚ := 200.50
def spent_on_sweets : ℚ := 35.25
def amount_left : ℚ := 114.85
def num_friends : ℕ := 2

-- Define the theorem
theorem little_krish_money_distribution :
  ∃ (amount_per_friend : ℚ),
    amount_per_friend = 25.20 ∧
    initial_amount - spent_on_sweets - (num_friends : ℚ) * amount_per_friend = amount_left :=
by
  sorry


end NUMINAMATH_CALUDE_little_krish_money_distribution_l3737_373715


namespace NUMINAMATH_CALUDE_integral_sin_cos_power_l3737_373791

theorem integral_sin_cos_power : ∫ x in (-π/2)..0, (2^8 * Real.sin x^4 * Real.cos x^4) = 3*π := by sorry

end NUMINAMATH_CALUDE_integral_sin_cos_power_l3737_373791


namespace NUMINAMATH_CALUDE_b_investment_is_correct_l3737_373726

-- Define the partnership
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  duration : ℕ
  a_share : ℕ
  b_share : ℕ

-- Define the problem
def partnership_problem : Partnership :=
  { a_investment := 7000
  , b_investment := 11000  -- This is what we want to prove
  , c_investment := 18000
  , duration := 8
  , a_share := 1400
  , b_share := 2200 }

-- Theorem statement
theorem b_investment_is_correct (p : Partnership) 
  (h1 : p.a_investment = 7000)
  (h2 : p.c_investment = 18000)
  (h3 : p.duration = 8)
  (h4 : p.a_share = 1400)
  (h5 : p.b_share = 2200) :
  p.b_investment = 11000 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_is_correct_l3737_373726


namespace NUMINAMATH_CALUDE_train_length_train_length_specific_l3737_373713

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : Real) (man_speed : Real) (passing_time : Real) : Real :=
  let relative_speed := train_speed + man_speed
  let relative_speed_mps := relative_speed * 1000 / 3600
  relative_speed_mps * passing_time

/-- Proof that a train with speed 114.99 kmph passing a man walking at 5 kmph in the opposite direction in 6 seconds has a length of approximately 199.98 meters. -/
theorem train_length_specific : 
  ∃ (ε : Real), ε > 0 ∧ abs (train_length 114.99 5 6 - 199.98) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_specific_l3737_373713


namespace NUMINAMATH_CALUDE_reaping_capacity_theorem_l3737_373752

/-- Represents the reaping capacity of a group of men -/
structure ReapingCapacity where
  men : ℕ
  hectares : ℝ
  days : ℕ

/-- Given the reaping capacity of one group, calculate the reaping capacity of another group -/
def calculate_reaping_capacity (base : ReapingCapacity) (target : ReapingCapacity) : Prop :=
  (target.men : ℝ) / base.men * (base.hectares / base.days) * target.days = target.hectares

/-- Theorem stating the relationship between the reaping capacities of two groups -/
theorem reaping_capacity_theorem (base target : ReapingCapacity) :
  base.men = 10 ∧ base.hectares = 80 ∧ base.days = 24 ∧
  target.men = 36 ∧ target.hectares = 360 ∧ target.days = 30 →
  calculate_reaping_capacity base target := by
  sorry

end NUMINAMATH_CALUDE_reaping_capacity_theorem_l3737_373752


namespace NUMINAMATH_CALUDE_h_is_even_l3737_373708

-- Define k as an even function
def k_even (k : ℝ → ℝ) : Prop :=
  ∀ x, k (-x) = k x

-- Define h using k
def h (k : ℝ → ℝ) (x : ℝ) : ℝ :=
  |k (x^5)|

-- Theorem statement
theorem h_is_even (k : ℝ → ℝ) (h_even : k_even k) :
  ∀ x, h k (-x) = h k x :=
by sorry

end NUMINAMATH_CALUDE_h_is_even_l3737_373708


namespace NUMINAMATH_CALUDE_smallest_value_l3737_373777

theorem smallest_value (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^3 ≤ x ∧ x^3 ≤ 3*x ∧ x^3 ≤ x^(1/3) ∧ x^3 ≤ 1/x^2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_l3737_373777


namespace NUMINAMATH_CALUDE_unique_a_value_l3737_373797

def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem unique_a_value : ∃! a : ℝ, A a ∪ B a = {0, 1, 2, 3, 9} ∧ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l3737_373797


namespace NUMINAMATH_CALUDE_complement_union_eq_inter_complements_l3737_373724

variable {Ω : Type*} [MeasurableSpace Ω]
variable (A B : Set Ω)

theorem complement_union_eq_inter_complements :
  (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ := by sorry

end NUMINAMATH_CALUDE_complement_union_eq_inter_complements_l3737_373724


namespace NUMINAMATH_CALUDE_eliza_ironed_17_clothes_l3737_373767

/-- Calculates the total number of clothes Eliza ironed given the time spent and ironing rates. -/
def total_clothes_ironed (blouse_time : ℕ) (dress_time : ℕ) (blouse_hours : ℕ) (dress_hours : ℕ) : ℕ :=
  let blouses := (blouse_hours * 60) / blouse_time
  let dresses := (dress_hours * 60) / dress_time
  blouses + dresses

/-- Proves that Eliza ironed 17 pieces of clothes given the conditions. -/
theorem eliza_ironed_17_clothes :
  total_clothes_ironed 15 20 2 3 = 17 := by
  sorry

#eval total_clothes_ironed 15 20 2 3

end NUMINAMATH_CALUDE_eliza_ironed_17_clothes_l3737_373767


namespace NUMINAMATH_CALUDE_quadratic_b_value_l3737_373716

/-- Given a quadratic function y = ax² + bx + c, prove that b = 3 when
    (2, y₁) and (-2, y₂) are points on the graph and y₁ - y₂ = 12 -/
theorem quadratic_b_value (a c y₁ y₂ : ℝ) :
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = 12 →
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_b_value_l3737_373716


namespace NUMINAMATH_CALUDE_interior_angles_sum_l3737_373773

theorem interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 3240) → (180 * ((n + 3) - 2) = 3780) := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l3737_373773


namespace NUMINAMATH_CALUDE_don_bottles_from_shop_c_l3737_373744

/-- The number of bottles Don can buy in total -/
def total_bottles : ℕ := 550

/-- The number of bottles Don buys from Shop A -/
def shop_a_bottles : ℕ := 150

/-- The number of bottles Don buys from Shop B -/
def shop_b_bottles : ℕ := 180

/-- The number of bottles Don buys from Shop C -/
def shop_c_bottles : ℕ := total_bottles - (shop_a_bottles + shop_b_bottles)

theorem don_bottles_from_shop_c : 
  shop_c_bottles = 550 - (150 + 180) := by sorry

end NUMINAMATH_CALUDE_don_bottles_from_shop_c_l3737_373744


namespace NUMINAMATH_CALUDE_factorial_10_trailing_zeros_base_15_l3737_373719

/-- The number of trailing zeros in the base 15 representation of a natural number -/
def trailingZerosBase15 (n : ℕ) : ℕ := sorry

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeros in the base 15 representation of 10! is 2 -/
theorem factorial_10_trailing_zeros_base_15 : 
  trailingZerosBase15 (factorial 10) = 2 := by sorry

end NUMINAMATH_CALUDE_factorial_10_trailing_zeros_base_15_l3737_373719


namespace NUMINAMATH_CALUDE_vector_basis_range_l3737_373751

/-- Two vectors form a basis of a 2D plane if they are linearly independent -/
def is_basis (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 ≠ a.2 * b.1

/-- The range of m for which (1,2) and (m,3m-2) form a basis -/
theorem vector_basis_range :
  ∀ m : ℝ, is_basis (1, 2) (m, 3*m-2) ↔ m ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_vector_basis_range_l3737_373751


namespace NUMINAMATH_CALUDE_no_integer_solution_l3737_373733

theorem no_integer_solution :
  ¬ ∃ (x y z : ℤ), x * (x - y) + y * (y - z) + z * (z - x) = 3 ∧ x > y ∧ y > z :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3737_373733


namespace NUMINAMATH_CALUDE_square_root_equation_l3737_373703

theorem square_root_equation (n : ℕ+) :
  Real.sqrt (1 + 1 / (n : ℝ)^2 + 1 / ((n + 1) : ℝ)^2) = 1 + 1 / ((n : ℝ) * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l3737_373703


namespace NUMINAMATH_CALUDE_kimberly_peanuts_l3737_373790

/-- The number of times Kimberly went to the store last month -/
def store_visits : ℕ := 3

/-- The number of peanuts Kimberly buys each time she goes to the store -/
def peanuts_per_visit : ℕ := 7

/-- The total number of peanuts Kimberly bought last month -/
def total_peanuts : ℕ := store_visits * peanuts_per_visit

theorem kimberly_peanuts : total_peanuts = 21 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_peanuts_l3737_373790


namespace NUMINAMATH_CALUDE_max_gcd_of_consecutive_cubic_sequence_l3737_373792

theorem max_gcd_of_consecutive_cubic_sequence :
  let b : ℕ → ℕ := fun n => 150 + n^3
  let d : ℕ → ℕ := fun n => Nat.gcd (b n) (b (n + 1))
  ∀ n : ℕ, n ≥ 1 → d n ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_consecutive_cubic_sequence_l3737_373792


namespace NUMINAMATH_CALUDE_solve_equation_and_calculate_l3737_373735

theorem solve_equation_and_calculate (x : ℝ) :
  Real.sqrt ((3 / x) + 1) = 4 / 3 →
  x = 27 / 7 ∧ x + 6 = 69 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_and_calculate_l3737_373735


namespace NUMINAMATH_CALUDE_determinant_transformation_l3737_373755

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 3 →
  Matrix.det !![p, 5*p + 4*q; r, 5*r + 4*s] = 12 := by
  sorry

end NUMINAMATH_CALUDE_determinant_transformation_l3737_373755


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3737_373705

/-- The intersection of {x | x ≥ -1} and {x | -2 < x < 2} is [-1, 2) -/
theorem intersection_of_sets : 
  let M : Set ℝ := {x | x ≥ -1}
  let N : Set ℝ := {x | -2 < x ∧ x < 2}
  M ∩ N = Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3737_373705


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_11_after_change_all_replacements_divisible_by_11_l3737_373712

/-- A function that replaces a digit at a given position in a number with a new digit. -/
def replaceDigit (n : ℕ) (pos : ℕ) (newDigit : ℕ) : ℕ :=
  sorry

/-- A function that checks if a number is divisible by 11. -/
def isDivisibleBy11 (n : ℕ) : Prop :=
  n % 11 = 0

/-- A function that generates all possible numbers after replacing one digit. -/
def allPossibleReplacements (n : ℕ) : List ℕ :=
  sorry

theorem smallest_number_divisible_by_11_after_change : 
  ∀ n : ℕ, n < 909090909 → 
  ∃ m ∈ allPossibleReplacements n, ¬isDivisibleBy11 m :=
by sorry

theorem all_replacements_divisible_by_11 : 
  ∀ m ∈ allPossibleReplacements 909090909, isDivisibleBy11 m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_11_after_change_all_replacements_divisible_by_11_l3737_373712


namespace NUMINAMATH_CALUDE_tangerine_sum_l3737_373796

theorem tangerine_sum (initial_count : ℕ) (final_counts : List ℕ) : 
  initial_count = 20 →
  final_counts = [10, 18, 17, 13, 16] →
  (final_counts.filter (· ≤ 13)).sum = 23 := by
  sorry

end NUMINAMATH_CALUDE_tangerine_sum_l3737_373796


namespace NUMINAMATH_CALUDE_symmetric_point_about_origin_l3737_373701

/-- Given a point P (-2, -3), prove that (2, 3) is its symmetric point about the origin -/
theorem symmetric_point_about_origin :
  let P : ℝ × ℝ := (-2, -3)
  let Q : ℝ × ℝ := (2, 3)
  (∀ (x y : ℝ), (x, y) = P → (-x, -y) = Q) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_about_origin_l3737_373701


namespace NUMINAMATH_CALUDE_place_digit_two_equals_formula_l3737_373787

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_bound : hundreds < 10
  tens_bound : tens < 10
  units_bound : units < 10

/-- Converts a ThreeDigitNumber to its integer value -/
def ThreeDigitNumber.toInt (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Places the digit 2 before a three-digit number -/
def placeDigitTwo (n : ThreeDigitNumber) : ℕ :=
  2000 + 10 * n.toInt

theorem place_digit_two_equals_formula (n : ThreeDigitNumber) :
  placeDigitTwo n = 1000 * (n.hundreds + 2) + 100 * n.tens + 10 * n.units := by
  sorry

end NUMINAMATH_CALUDE_place_digit_two_equals_formula_l3737_373787


namespace NUMINAMATH_CALUDE_fraction_males_first_class_l3737_373714

/-- Given a flight with passengers, prove the fraction of males in first class -/
theorem fraction_males_first_class
  (total_passengers : ℕ)
  (female_percentage : ℚ)
  (first_class_percentage : ℚ)
  (females_in_coach : ℕ)
  (h_total : total_passengers = 120)
  (h_female : female_percentage = 30 / 100)
  (h_first_class : first_class_percentage = 10 / 100)
  (h_females_coach : females_in_coach = 28) :
  (↑(total_passengers * first_class_percentage.num - (total_passengers * female_percentage.num - females_in_coach)) /
   ↑(total_passengers * first_class_percentage.num) : ℚ) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_males_first_class_l3737_373714


namespace NUMINAMATH_CALUDE_solution_set_characterization_l3737_373700

open Real

theorem solution_set_characterization 
  (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h_derivative : ∀ x, HasDerivAt f (f' x) x)
  (h_initial : f 0 = 2)
  (h_bound : ∀ x, f x + f' x > 1) :
  ∀ x, (exp x * f x > exp x + 1) ↔ x > 0 := by
sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l3737_373700


namespace NUMINAMATH_CALUDE_john_star_wars_toys_cost_l3737_373758

/-- The total cost of John's Star Wars toys, including the lightsaber -/
def total_cost (other_toys_cost lightsaber_cost : ℕ) : ℕ :=
  other_toys_cost + lightsaber_cost

/-- The cost of the lightsaber -/
def lightsaber_cost (other_toys_cost : ℕ) : ℕ :=
  2 * other_toys_cost

theorem john_star_wars_toys_cost (other_toys_cost : ℕ) 
  (h : other_toys_cost = 1000) : 
  total_cost other_toys_cost (lightsaber_cost other_toys_cost) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_john_star_wars_toys_cost_l3737_373758


namespace NUMINAMATH_CALUDE_ratio_of_a_over_3_to_b_over_2_l3737_373786

theorem ratio_of_a_over_3_to_b_over_2 (a b c : ℝ) 
  (h1 : 2 * a = 3 * b) 
  (h2 : c ≠ 0) 
  (h3 : 3 * a + 2 * b = c) : 
  (a / 3) / (b / 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_a_over_3_to_b_over_2_l3737_373786


namespace NUMINAMATH_CALUDE_arccos_negative_half_equals_two_pi_thirds_l3737_373770

theorem arccos_negative_half_equals_two_pi_thirds : 
  Real.arccos (-1/2) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_half_equals_two_pi_thirds_l3737_373770


namespace NUMINAMATH_CALUDE_sum_53_to_100_l3737_373778

def sum_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_53_to_100 (h : sum_range 51 100 = 3775) : sum_range 53 100 = 3672 := by
  sorry

end NUMINAMATH_CALUDE_sum_53_to_100_l3737_373778


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l3737_373743

theorem elevator_weight_problem (adults_avg_weight : ℝ) (elevator_max_weight : ℝ) (next_person_max_weight : ℝ) :
  adults_avg_weight = 140 →
  elevator_max_weight = 600 →
  next_person_max_weight = 52 →
  (elevator_max_weight - 3 * adults_avg_weight - next_person_max_weight) = 128 :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l3737_373743


namespace NUMINAMATH_CALUDE_valentine_cards_theorem_l3737_373739

theorem valentine_cards_theorem (x y : ℕ) : 
  x * y = x + y + 30 → x * y = 64 := by
  sorry

end NUMINAMATH_CALUDE_valentine_cards_theorem_l3737_373739


namespace NUMINAMATH_CALUDE_bob_total_earnings_l3737_373764

-- Define constants
def regular_rate : ℚ := 5
def overtime_rate : ℚ := 6
def regular_hours : ℕ := 40
def first_week_hours : ℕ := 44
def second_week_hours : ℕ := 48

-- Define function to calculate weekly earnings
def weekly_earnings (hours_worked : ℕ) : ℚ :=
  let regular := min hours_worked regular_hours
  let overtime := max (hours_worked - regular_hours) 0
  regular * regular_rate + overtime * overtime_rate

-- Theorem statement
theorem bob_total_earnings :
  weekly_earnings first_week_hours + weekly_earnings second_week_hours = 472 :=
by sorry

end NUMINAMATH_CALUDE_bob_total_earnings_l3737_373764


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3737_373763

theorem interest_rate_calculation (P : ℝ) (t : ℝ) (diff : ℝ) (r : ℝ) : 
  P = 3600 → 
  t = 2 → 
  P * ((1 + r)^t - 1) - P * r * t = diff → 
  diff = 36 → 
  r = 0.1 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3737_373763


namespace NUMINAMATH_CALUDE_largest_circle_equation_l3737_373765

/-- The standard equation of the circle with the largest area, centered at (2, -3) and tangent to the line 2mx-y-2m-1=0 (m ∈ ℝ) -/
theorem largest_circle_equation (m : ℝ) : 
  ∃ (x y : ℝ), (x - 2)^2 + (y + 3)^2 = 5 ∧ 
  ∀ (x' y' r : ℝ), 
    ((x' - 2)^2 + (y' + 3)^2 = r^2) → 
    (2*m*x' - y' - 2*m - 1 = 0) → 
    r^2 ≤ 5 := by
  sorry

#check largest_circle_equation

end NUMINAMATH_CALUDE_largest_circle_equation_l3737_373765


namespace NUMINAMATH_CALUDE_show_episodes_l3737_373742

/-- Calculates the number of episodes in a show given the watching conditions -/
def num_episodes (days : ℕ) (episode_length : ℕ) (hours_per_day : ℕ) : ℕ :=
  (days * hours_per_day * 60) / episode_length

/-- Proves that the number of episodes in the show is 20 -/
theorem show_episodes : num_episodes 5 30 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_show_episodes_l3737_373742


namespace NUMINAMATH_CALUDE_square_root_three_expansion_square_root_three_specific_case_simplify_square_root_expression_l3737_373762

-- Part 1
theorem square_root_three_expansion (a b m n : ℕ+) :
  a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 →
  a = m^2 + 3*n^2 ∧ b = 2*m*n :=
sorry

-- Part 2
theorem square_root_three_specific_case (a m n : ℕ+) :
  a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 →
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem simplify_square_root_expression :
  Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_square_root_three_expansion_square_root_three_specific_case_simplify_square_root_expression_l3737_373762


namespace NUMINAMATH_CALUDE_log_relation_l3737_373729

theorem log_relation (c b : ℝ) (hc : c = Real.log 625 / Real.log 16) (hb : b = Real.log 25 / Real.log 2) : 
  c = b / 2 := by
sorry

end NUMINAMATH_CALUDE_log_relation_l3737_373729


namespace NUMINAMATH_CALUDE_cube_side_area_l3737_373760

theorem cube_side_area (V : ℝ) (s : ℝ) (h : V = 125) :
  V = s^3 → s^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_area_l3737_373760


namespace NUMINAMATH_CALUDE_travel_time_calculation_l3737_373784

/-- Given a person traveling at a constant speed for a certain distance,
    calculate the time taken for the journey. -/
theorem travel_time_calculation (speed : ℝ) (distance : ℝ) (h1 : speed = 75) (h2 : distance = 300) :
  distance / speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l3737_373784


namespace NUMINAMATH_CALUDE_cone_height_from_lateral_surface_l3737_373768

/-- 
Given a cone whose lateral surface is a semicircle with radius a,
prove that the height of the cone is (√3/2)a
-/
theorem cone_height_from_lateral_surface (a : ℝ) (h : a > 0) :
  let slant_height := a
  let base_circumference := π * a
  let base_radius := a / 2
  let height := Real.sqrt ((3 * a^2) / 4)
  height = (Real.sqrt 3 / 2) * a :=
by sorry

end NUMINAMATH_CALUDE_cone_height_from_lateral_surface_l3737_373768


namespace NUMINAMATH_CALUDE_middle_digit_is_zero_l3737_373722

/-- Represents a three-digit number in base 8 -/
structure Base8Number where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds < 8 ∧ tens < 8 ∧ ones < 8

/-- Converts a Base8Number to its decimal (base 10) representation -/
def toDecimal (n : Base8Number) : Nat :=
  64 * n.hundreds + 8 * n.tens + n.ones

/-- Represents a three-digit number in base 10 -/
structure Base10Number where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- Checks if a Base8Number has its digits reversed in base 10 representation -/
def hasReversedDigits (n : Base8Number) : Prop :=
  ∃ (m : Base10Number), 
    toDecimal n = 100 * m.hundreds + 10 * m.tens + m.ones ∧
    n.hundreds = m.ones ∧
    n.tens = m.tens ∧
    n.ones = m.hundreds

theorem middle_digit_is_zero (n : Base8Number) 
  (h : hasReversedDigits n) : n.tens = 0 := by
  sorry

end NUMINAMATH_CALUDE_middle_digit_is_zero_l3737_373722


namespace NUMINAMATH_CALUDE_yang_hui_problem_l3737_373759

theorem yang_hui_problem : ∃ (x : ℕ), 
  (x % 2 = 1) ∧ 
  (x % 5 = 2) ∧ 
  (x % 7 = 3) ∧ 
  (x % 9 = 4) ∧ 
  (∀ y : ℕ, y < x → ¬((y % 2 = 1) ∧ (y % 5 = 2) ∧ (y % 7 = 3) ∧ (y % 9 = 4))) ∧
  x = 157 :=
by sorry

end NUMINAMATH_CALUDE_yang_hui_problem_l3737_373759


namespace NUMINAMATH_CALUDE_set_a_range_l3737_373702

theorem set_a_range (a : ℝ) : 
  let A : Set ℝ := {x | 6 * x + a > 0}
  1 ∉ A → a ∈ Set.Iic (-6) :=
by sorry

end NUMINAMATH_CALUDE_set_a_range_l3737_373702


namespace NUMINAMATH_CALUDE_probability_of_favorable_outcome_l3737_373727

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 60 ∧ 1 ≤ b ∧ b ≤ 60 ∧ a ≠ b

def is_favorable_pair (a b : ℕ) : Prop :=
  is_valid_pair a b ∧ ∃ k : ℕ, a * b + a + b = 7 * k - 1

def total_pairs : ℕ := Nat.choose 60 2

def favorable_pairs : ℕ := 444

theorem probability_of_favorable_outcome :
  (favorable_pairs : ℚ) / total_pairs = 74 / 295 := by sorry

end NUMINAMATH_CALUDE_probability_of_favorable_outcome_l3737_373727


namespace NUMINAMATH_CALUDE_second_number_in_expression_l3737_373798

theorem second_number_in_expression : 
  ∃ x : ℝ, (26.3 * x * 20) / 3 + 125 = 2229 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_number_in_expression_l3737_373798


namespace NUMINAMATH_CALUDE_number_of_hens_l3737_373736

theorem number_of_hens (total_animals : ℕ) (total_feet : ℕ) (hen_feet cow_feet : ℕ) :
  total_animals = 48 →
  total_feet = 140 →
  hen_feet = 2 →
  cow_feet = 4 →
  ∃ (num_hens num_cows : ℕ),
    num_hens + num_cows = total_animals ∧
    num_hens * hen_feet + num_cows * cow_feet = total_feet ∧
    num_hens = 26 :=
by sorry

end NUMINAMATH_CALUDE_number_of_hens_l3737_373736


namespace NUMINAMATH_CALUDE_inequality_implies_a_bound_l3737_373718

theorem inequality_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, x > 0 → 2 * x * Real.log x ≥ -x^2 + a*x - 3) → 
  a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_bound_l3737_373718


namespace NUMINAMATH_CALUDE_track_length_proof_l3737_373785

/-- The length of the circular track -/
def track_length : ℝ := 520

/-- The distance Brenda runs to the first meeting point -/
def brenda_first_distance : ℝ := 80

/-- The distance Sue runs past the first meeting point to the second meeting point -/
def sue_second_distance : ℝ := 180

/-- Theorem stating the track length given the conditions -/
theorem track_length_proof :
  ∀ (x : ℝ),
  x > 0 →
  (x / 2 - brenda_first_distance) / brenda_first_distance = 
  (x / 2 - (sue_second_distance + brenda_first_distance)) / (x / 2 + sue_second_distance) →
  x = track_length := by
sorry

end NUMINAMATH_CALUDE_track_length_proof_l3737_373785


namespace NUMINAMATH_CALUDE_stratified_sampling_proof_l3737_373782

theorem stratified_sampling_proof (total_population : ℕ) (female_students : ℕ) 
  (sampled_female : ℕ) (sample_size : ℕ) 
  (h1 : total_population = 1200)
  (h2 : female_students = 500)
  (h3 : sampled_female = 40)
  (h4 : (sample_size : ℚ) / total_population = (sampled_female : ℚ) / female_students) :
  sample_size = 96 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_proof_l3737_373782


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l3737_373717

theorem sqrt_difference_equality : Real.sqrt 27 - Real.sqrt (1/3) = (8/3) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l3737_373717


namespace NUMINAMATH_CALUDE_sine_function_period_l3737_373723

/-- Given a sinusoidal function with angular frequency ω > 0 and smallest positive period 2π/3, prove that ω = 3. -/
theorem sine_function_period (ω : ℝ) : ω > 0 → (∀ x, 2 * Real.sin (ω * x + π / 6) = 2 * Real.sin (ω * (x + 2 * π / 3) + π / 6)) → ω = 3 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_period_l3737_373723


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l3737_373781

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    if a_2 : a_3 = 5 : 2, then S_3 : S_5 = 3 : 2 -/
theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →
  (∀ n, a (n + 1) = a n + (a 2 - a 1)) →
  (a 2 : ℝ) / (a 3) = 5 / 2 →
  (S 3 : ℝ) / (S 5) = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l3737_373781


namespace NUMINAMATH_CALUDE_tangent_circles_count_l3737_373789

-- Define a type for lines in a plane
structure Line where
  -- Add necessary properties for a line

-- Define a type for circles in a plane
structure Circle where
  -- Add necessary properties for a circle

-- Define a function to check if a circle is tangent to a line
def is_tangent (c : Circle) (l : Line) : Prop :=
  sorry

-- Define a function to count the number of circles tangent to three lines
def count_tangent_circles (l1 l2 l3 : Line) : ℕ :=
  sorry

-- Define predicates for different line configurations
def general_position (l1 l2 l3 : Line) : Prop :=
  sorry

def intersect_at_point (l1 l2 l3 : Line) : Prop :=
  sorry

def all_parallel (l1 l2 l3 : Line) : Prop :=
  sorry

def two_parallel_one_intersecting (l1 l2 l3 : Line) : Prop :=
  sorry

theorem tangent_circles_count 
  (l1 l2 l3 : Line) : 
  (general_position l1 l2 l3 → count_tangent_circles l1 l2 l3 = 4) ∧
  (intersect_at_point l1 l2 l3 → count_tangent_circles l1 l2 l3 = 0) ∧
  (all_parallel l1 l2 l3 → count_tangent_circles l1 l2 l3 = 0) ∧
  (two_parallel_one_intersecting l1 l2 l3 → count_tangent_circles l1 l2 l3 = 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l3737_373789


namespace NUMINAMATH_CALUDE_fortiethSelectedNumber_l3737_373730

/-- Calculates the nth selected number in a sequence -/
def nthSelectedNumber (totalParticipants : ℕ) (numSelected : ℕ) (firstNumber : ℕ) (n : ℕ) : ℕ :=
  let spacing := totalParticipants / numSelected
  (n - 1) * spacing + firstNumber

theorem fortiethSelectedNumber :
  nthSelectedNumber 1000 50 15 40 = 795 := by
  sorry

end NUMINAMATH_CALUDE_fortiethSelectedNumber_l3737_373730


namespace NUMINAMATH_CALUDE_floor_tiles_theorem_l3737_373704

/-- A rectangular floor covered with congruent square tiles. -/
structure TiledFloor where
  width : ℕ
  length : ℕ
  perimeterTiles : ℕ
  lengthTwiceWidth : length = 2 * width
  tilesAlongPerimeter : perimeterTiles = 2 * (width + length)

/-- The total number of tiles covering the floor. -/
def totalTiles (floor : TiledFloor) : ℕ :=
  floor.width * floor.length

/-- Theorem stating that a rectangular floor with 88 tiles along the perimeter
    and length twice the width has 430 tiles in total. -/
theorem floor_tiles_theorem (floor : TiledFloor) 
    (h : floor.perimeterTiles = 88) : totalTiles floor = 430 := by
  sorry

end NUMINAMATH_CALUDE_floor_tiles_theorem_l3737_373704


namespace NUMINAMATH_CALUDE_circle_symmetry_l3737_373745

def circle_equation (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem circle_symmetry (h : Set (ℝ × ℝ)) :
  h = circle_equation (-2, 1) 1 →
  (λ (x, y) => circle_equation (2, -1) 1 (x, y)) =
  (λ (x, y) => (x - 2)^2 + (y + 1)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3737_373745


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3737_373738

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 9) (h2 : x ≠ -7) :
  (4 * x - 6) / (x^2 - 2*x - 63) = (15 / 8) / (x - 9) + (17 / 8) / (x + 7) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3737_373738


namespace NUMINAMATH_CALUDE_acute_triangle_side_range_l3737_373737

theorem acute_triangle_side_range (x : ℝ) : 
  x > 0 → 
  (∀ α β γ : ℝ, 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2 ∧ 
   α + β + γ = π ∧
   x^2 = 2^2 + 3^2 - 2*2*3*Real.cos γ ∧
   2^2 = 3^2 + x^2 - 2*3*x*Real.cos α ∧
   3^2 = 2^2 + x^2 - 2*2*x*Real.cos β) →
  Real.sqrt 5 < x ∧ x < Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_side_range_l3737_373737


namespace NUMINAMATH_CALUDE_sphere_radius_from_segment_l3737_373710

/-- A spherical segment is a portion of a sphere cut off by a plane. -/
structure SphericalSegment where
  base_diameter : ℝ
  height : ℝ

/-- The radius of a sphere given a spherical segment. -/
def sphere_radius (segment : SphericalSegment) : ℝ :=
  sorry

theorem sphere_radius_from_segment (segment : SphericalSegment) 
  (h1 : segment.base_diameter = 24)
  (h2 : segment.height = 8) :
  sphere_radius segment = 13 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_segment_l3737_373710


namespace NUMINAMATH_CALUDE_tan_pi_third_plus_cos_nineteen_sixths_pi_l3737_373728

theorem tan_pi_third_plus_cos_nineteen_sixths_pi :
  Real.tan (π / 3) + Real.cos (19 * π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_third_plus_cos_nineteen_sixths_pi_l3737_373728


namespace NUMINAMATH_CALUDE_expression_result_l3737_373740

theorem expression_result : (7.5 * 7.5 + 37.5 + 2.5 * 2.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l3737_373740


namespace NUMINAMATH_CALUDE_cookie_batch_size_l3737_373732

theorem cookie_batch_size (batch_count : ℕ) (oatmeal_count : ℕ) (total_count : ℕ) : 
  batch_count = 2 → oatmeal_count = 4 → total_count = 10 → 
  ∃ (cookies_per_batch : ℕ), cookies_per_batch = 3 ∧ batch_count * cookies_per_batch + oatmeal_count = total_count :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_batch_size_l3737_373732


namespace NUMINAMATH_CALUDE_nested_sqrt_problem_l3737_373766

theorem nested_sqrt_problem (n : ℕ) :
  (∃ k : ℕ, (n * (n * n^(1/2))^(1/2))^(1/2) = k) ∧ 
  (n * (n * n^(1/2))^(1/2))^(1/2) < 2217 →
  n = 256 := by
sorry

end NUMINAMATH_CALUDE_nested_sqrt_problem_l3737_373766


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l3737_373741

/-- Given a curve in polar coordinates defined by r = 2 / (2sin θ - cos θ),
    prove that it represents a line in Cartesian coordinates. -/
theorem polar_to_cartesian_line :
  ∀ θ r : ℝ,
  r = 2 / (2 * Real.sin θ - Real.cos θ) →
  ∃ m c : ℝ, ∀ x y : ℝ,
  x = r * Real.cos θ ∧ y = r * Real.sin θ →
  y = m * x + c :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l3737_373741


namespace NUMINAMATH_CALUDE_zac_strawberries_l3737_373725

def strawberry_problem (total : ℕ) (jonathan_matthew : ℕ) (matthew_zac : ℕ) : Prop :=
  ∃ (jonathan matthew zac : ℕ),
    jonathan + matthew + zac = total ∧
    jonathan + matthew = jonathan_matthew ∧
    matthew + zac = matthew_zac ∧
    zac = 200

theorem zac_strawberries :
  strawberry_problem 550 350 250 :=
sorry

end NUMINAMATH_CALUDE_zac_strawberries_l3737_373725


namespace NUMINAMATH_CALUDE_baxter_peanut_purchase_l3737_373776

/-- Represents the peanut purchase scenario at the Peanut Emporium -/
structure PeanutPurchase where
  pricePerPound : ℝ
  minimumPurchase : ℝ
  bulkDiscountThreshold : ℝ
  bulkDiscountRate : ℝ
  earlyBirdDiscountRate : ℝ
  salesTaxRate : ℝ
  totalSpent : ℝ

/-- Calculates the pounds of peanuts purchased given the purchase scenario -/
def calculatePoundsPurchased (p : PeanutPurchase) : ℝ :=
  sorry

/-- Theorem: Given the purchase conditions, Baxter bought 28 pounds over the minimum -/
theorem baxter_peanut_purchase :
  let p := PeanutPurchase.mk 3 15 25 0.1 0.05 0.08 119.88
  calculatePoundsPurchased p - p.minimumPurchase = 28 := by
  sorry

end NUMINAMATH_CALUDE_baxter_peanut_purchase_l3737_373776


namespace NUMINAMATH_CALUDE_product_of_absolute_sum_l3737_373734

theorem product_of_absolute_sum (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^2 / (b*c) + b^2 / (c*a) + c^2 / (a*b) = 1) : 
  (Complex.abs (a + b + c) - 1) * 
  (Complex.abs (a + b + c) - (1 + Real.sqrt 3)) * 
  (Complex.abs (a + b + c) - (1 - Real.sqrt 3)) = 2 := by
sorry

end NUMINAMATH_CALUDE_product_of_absolute_sum_l3737_373734


namespace NUMINAMATH_CALUDE_only_villages_comprehensive_villages_only_comprehensive_option_l3737_373772

/-- Represents a survey option -/
inductive SurveyOption
  | VillagesPollution
  | DrugQuality
  | PublicOpinion
  | RiverWaterQuality

/-- Defines what makes a survey comprehensive -/
def is_comprehensive (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.VillagesPollution => true
  | _ => false

/-- Theorem stating that only the villages pollution survey is comprehensive -/
theorem only_villages_comprehensive :
  ∀ (option : SurveyOption),
    is_comprehensive option ↔ option = SurveyOption.VillagesPollution :=
by sorry

/-- Main theorem proving that investigating five villages is the only suitable option -/
theorem villages_only_comprehensive_option :
  ∃! (option : SurveyOption), is_comprehensive option :=
by sorry

end NUMINAMATH_CALUDE_only_villages_comprehensive_villages_only_comprehensive_option_l3737_373772


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3737_373757

theorem solve_linear_equation (x y : ℝ) :
  4 * x - y = 3 → y = 4 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3737_373757


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3737_373753

theorem max_value_quadratic :
  (∀ x : ℝ, -3 * x^2 + 15 * x + 9 ≤ 111/4) ∧
  (∃ x : ℝ, -3 * x^2 + 15 * x + 9 = 111/4) := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3737_373753


namespace NUMINAMATH_CALUDE_inequality_proof_l3737_373707

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a * b / (c - 1) + b * c / (a - 1) + c * a / (b - 1) ≥ 12) ∧
  (a * b / (c - 1) + b * c / (a - 1) + c * a / (b - 1) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3737_373707


namespace NUMINAMATH_CALUDE_tom_catch_l3737_373771

/-- The number of trout Melanie caught -/
def melanie_catch : ℕ := 8

/-- The factor by which Tom's catch exceeds Melanie's -/
def tom_factor : ℕ := 2

/-- Tom's catch is equal to the product of Melanie's catch and Tom's factor -/
theorem tom_catch : melanie_catch * tom_factor = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_catch_l3737_373771
