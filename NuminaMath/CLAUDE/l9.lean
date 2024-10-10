import Mathlib

namespace quadratic_vertex_form_h_l9_956

/-- Given a quadratic expression 3x^2 + 9x + 20, when expressed in the form a(x - h)^2 + k, h = -3/2 -/
theorem quadratic_vertex_form_h (x : ℝ) : 
  ∃ (a k : ℝ), 3*x^2 + 9*x + 20 = a*(x - (-3/2))^2 + k :=
by sorry

end quadratic_vertex_form_h_l9_956


namespace second_number_value_l9_953

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 330 ∧ 
  a = 2 * b ∧ 
  c = (1/3) * a → 
  b = 90 := by
sorry

end second_number_value_l9_953


namespace multiplications_in_thirty_minutes_l9_974

/-- Represents the number of multiplications a computer can perform per second -/
def multiplications_per_second : ℕ := 20000

/-- Represents the number of minutes we want to calculate for -/
def minutes : ℕ := 30

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Theorem stating that the computer can perform 36,000,000 multiplications in 30 minutes -/
theorem multiplications_in_thirty_minutes :
  multiplications_per_second * minutes * seconds_per_minute = 36000000 := by
  sorry

end multiplications_in_thirty_minutes_l9_974


namespace bottles_from_625_l9_914

/-- The number of new bottles that can be made from a given number of initial bottles -/
def new_bottles (initial : ℕ) : ℕ :=
  if initial < 5 then 0
  else (initial / 5) + new_bottles (initial / 5)

/-- The theorem stating that 625 initial bottles will result in 195 new bottles -/
theorem bottles_from_625 :
  new_bottles 625 = 195 := by
sorry

end bottles_from_625_l9_914


namespace isosceles_triangle_condition_l9_928

theorem isosceles_triangle_condition (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- Ensure positive angles
  A + B + C = π →  -- Triangle angle sum
  2 * Real.cos B * Real.sin A = Real.sin C →  -- Given condition
  A = B  -- Conclusion: isosceles triangle
:= by sorry

end isosceles_triangle_condition_l9_928


namespace inverse_of_proposition_l9_901

-- Define the original proposition
def original_prop (a c : ℝ) : Prop := a > 0 → a * c^2 ≥ 0

-- Define the inverse proposition
def inverse_prop (a c : ℝ) : Prop := a * c^2 ≥ 0 → a > 0

-- Theorem stating that inverse_prop is the inverse of original_prop
theorem inverse_of_proposition :
  ∀ a c : ℝ, inverse_prop a c ↔ ¬(∃ a c : ℝ, original_prop a c ∧ ¬(inverse_prop a c)) :=
by sorry

end inverse_of_proposition_l9_901


namespace subtract_preserves_inequality_l9_910

theorem subtract_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end subtract_preserves_inequality_l9_910


namespace consecutive_integers_cube_sum_l9_935

theorem consecutive_integers_cube_sum (n : ℕ) :
  (n > 0) →
  ((n - 1)^2 + n^2 + (n + 1)^2 = 7805) →
  ((n - 1)^3 + n^3 + (n + 1)^3 = 398259) :=
by sorry

end consecutive_integers_cube_sum_l9_935


namespace sequence_inequality_l9_991

theorem sequence_inequality (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  ∀ n : ℕ, (a^n / (n : ℝ)^b) < (a^(n+1) / ((n+1) : ℝ)^b) :=
sorry

end sequence_inequality_l9_991


namespace min_perimeter_triangle_APF_l9_913

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the right focus F
def F : ℝ × ℝ := (2, 0)

-- Define point A
def A : ℝ × ℝ := (-1, 1)

-- Define a point P on the left branch of the hyperbola
def P : ℝ × ℝ := sorry

-- Define the perimeter of triangle APF
def perimeter (P : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem min_perimeter_triangle_APF :
  ∀ P, hyperbola P.1 P.2 → P.1 < 0 →
  perimeter P ≥ 3 * Real.sqrt 2 + Real.sqrt 10 :=
sorry

end min_perimeter_triangle_APF_l9_913


namespace greatest_integer_with_gcf_three_one_forty_one_satisfies_conditions_main_result_l9_959

theorem greatest_integer_with_gcf_three (n : ℕ) : n < 150 ∧ Nat.gcd n 24 = 3 → n ≤ 141 :=
by
  sorry

theorem one_forty_one_satisfies_conditions : 141 < 150 ∧ Nat.gcd 141 24 = 3 :=
by
  sorry

theorem main_result : ∃ (n : ℕ), n < 150 ∧ Nat.gcd n 24 = 3 ∧ 
  ∀ (m : ℕ), m < 150 ∧ Nat.gcd m 24 = 3 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_three_one_forty_one_satisfies_conditions_main_result_l9_959


namespace digit_sum_characterization_l9_904

/-- Digit sum in base 4038 -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Sequence of distinct positive integers -/
def validSequence (s : ℕ → ℕ) : Prop :=
  (∀ i j, i ≠ j → s i ≠ s j) ∧ (∀ n, s n > 0)

/-- Condition on the sequence growth -/
def boundedByA (s : ℕ → ℕ) (a : ℝ) : Prop :=
  ∀ n, (s n : ℝ) ≤ a * n

/-- Infinitely many terms with digit sum not divisible by 2019 -/
def infinitelyManyNotDivisible (s : ℕ → ℕ) : Prop :=
  ∀ N, ∃ n > N, ¬ 2019 ∣ digitSum (s n)

theorem digit_sum_characterization (a : ℝ) (h : a ≥ 1) :
  (∀ s : ℕ → ℕ, validSequence s → boundedByA s a → infinitelyManyNotDivisible s) ↔
  a < 2019 := by sorry

end digit_sum_characterization_l9_904


namespace S_is_three_rays_with_common_point_l9_990

/-- The set S of points (x, y) in the coordinate plane satisfying the given conditions -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 6 ≤ 5) ∨
               (5 = y - 6 ∧ x + 3 ≤ 5) ∨
               (x + 3 = y - 6 ∧ 5 ≤ x + 3)}

/-- The three rays that make up set S -/
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≤ 11}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 11 ∧ p.1 ≤ 2}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 9 ∧ p.1 ≥ 2}

/-- The common point of the three rays -/
def commonPoint : ℝ × ℝ := (2, 11)

theorem S_is_three_rays_with_common_point :
  S = ray1 ∪ ray2 ∪ ray3 ∧
  ray1 ∩ ray2 ∩ ray3 = {commonPoint} :=
by sorry

end S_is_three_rays_with_common_point_l9_990


namespace largest_guaranteed_divisor_l9_988

def die_numbers : Finset Nat := {1, 2, 3, 4, 5, 6}

def is_valid_product (P : Nat) : Prop :=
  ∃ (S : Finset Nat), S ⊆ die_numbers ∧ S.card = 5 ∧ P = S.prod id

theorem largest_guaranteed_divisor :
  ∀ P, is_valid_product P → (12 ∣ P) ∧ ∀ n, n > 12 → ¬∀ Q, is_valid_product Q → (n ∣ Q) :=
by sorry

end largest_guaranteed_divisor_l9_988


namespace vector_parallelism_l9_958

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, 3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 = t * w.1 ∧ v.2 = t * w.2

theorem vector_parallelism :
  ∃! k : ℝ, parallel ((k * a.1 + b.1, k * a.2 + b.2) : ℝ × ℝ) ((a.1 - 3 * b.1, a.2 - 3 * b.2) : ℝ × ℝ) ∧
  k = -1/3 := by sorry

end vector_parallelism_l9_958


namespace corey_sunday_vs_saturday_l9_924

/-- Corey's goal for finding golf balls -/
def goal : ℕ := 48

/-- Number of golf balls Corey found on Saturday -/
def saturdayBalls : ℕ := 16

/-- Number of golf balls Corey still needs to reach his goal -/
def stillNeeded : ℕ := 14

/-- Number of golf balls Corey found on Sunday -/
def sundayBalls : ℕ := goal - saturdayBalls - stillNeeded

theorem corey_sunday_vs_saturday : sundayBalls - saturdayBalls = 2 := by
  sorry

end corey_sunday_vs_saturday_l9_924


namespace train_crossing_time_l9_933

/-- Proves that a train crossing a platform of its own length in 60 seconds
    will take 30 seconds to cross a signal pole. -/
theorem train_crossing_time (train_length : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ)
  (h1 : train_length = 420)
  (h2 : platform_length = train_length)
  (h3 : platform_crossing_time = 60) :
  train_length / ((train_length + platform_length) / platform_crossing_time) = 30 := by
  sorry

end train_crossing_time_l9_933


namespace cos_2017pi_over_6_l9_992

theorem cos_2017pi_over_6 : Real.cos (2017 * Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end cos_2017pi_over_6_l9_992


namespace zero_in_interval_implies_m_leq_neg_one_l9_903

/-- A function f(x) = x² + (m-1)x + 1 has a zero point in the interval [0, 2] -/
def has_zero_in_interval (m : ℝ) : Prop :=
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^2 + (m-1)*x + 1 = 0

/-- If f(x) = x² + (m-1)x + 1 has a zero point in the interval [0, 2], then m ≤ -1 -/
theorem zero_in_interval_implies_m_leq_neg_one (m : ℝ) :
  has_zero_in_interval m → m ≤ -1 := by
  sorry

end zero_in_interval_implies_m_leq_neg_one_l9_903


namespace inequality_solution_set_l9_921

theorem inequality_solution_set (a b : ℝ) (h : a ≠ b) :
  {x : ℝ | a^2 * x + b^2 * (1 - x) ≥ (a * x + b * (1 - x))^2} = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end inequality_solution_set_l9_921


namespace sum_of_fractions_l9_950

theorem sum_of_fractions : (1 : ℚ) / 3 + 2 / 9 + 1 / 6 = 13 / 18 := by
  sorry

end sum_of_fractions_l9_950


namespace percentage_not_sold_is_66_l9_999

def initial_stock : ℕ := 800
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

theorem percentage_not_sold_is_66 : 
  (books_not_sold : ℚ) / (initial_stock : ℚ) * 100 = 66 := by
  sorry

end percentage_not_sold_is_66_l9_999


namespace f_derivative_zero_l9_923

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom f_differentiable : Differentiable ℝ f
axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y

-- State the theorem
theorem f_derivative_zero : deriv f 0 = 0 := by sorry

end f_derivative_zero_l9_923


namespace sum_and_reciprocal_sum_zero_implies_extremes_sum_zero_l9_965

theorem sum_and_reciprocal_sum_zero_implies_extremes_sum_zero
  (a b c d : ℝ)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (h_sum : a + b + c + d = 0)
  (h_reciprocal_sum : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 := by
  sorry

end sum_and_reciprocal_sum_zero_implies_extremes_sum_zero_l9_965


namespace cubic_identity_l9_996

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end cubic_identity_l9_996


namespace train_speed_problem_l9_993

theorem train_speed_problem (distance : ℝ) (time_diff : ℝ) (speed_increase : ℝ) 
  (h1 : distance = 600)
  (h2 : time_diff = 4)
  (h3 : speed_increase = 12) :
  ∃ (normal_speed : ℝ),
    normal_speed > 0 ∧
    (distance / normal_speed) - (distance / (normal_speed + speed_increase)) = time_diff ∧
    normal_speed = 37 := by
  sorry

end train_speed_problem_l9_993


namespace special_function_value_l9_929

/-- A binary function on positive integers satisfying certain properties -/
def special_function (f : ℕ+ → ℕ+ → ℕ+) : Prop :=
  (∀ x, f x x = x) ∧
  (∀ x y, f x y = f y x) ∧
  (∀ x y, (x + y) * (f x y) = y * (f x (x + y)))

/-- Theorem stating that f(12, 16) = 48 for any function satisfying the special properties -/
theorem special_function_value (f : ℕ+ → ℕ+ → ℕ+) (h : special_function f) : 
  f 12 16 = 48 := by
  sorry

end special_function_value_l9_929


namespace multiplicative_inverse_123_mod_400_l9_930

theorem multiplicative_inverse_123_mod_400 : ∃ a : ℕ, a < 400 ∧ (123 * a) % 400 = 1 :=
by
  use 387
  sorry

end multiplicative_inverse_123_mod_400_l9_930


namespace two_numbers_problem_l9_931

theorem two_numbers_problem (x y : ℕ) (h1 : x + y = 60) (h2 : Nat.gcd x y + Nat.lcm x y = 84) :
  (x = 24 ∧ y = 36) ∨ (x = 36 ∧ y = 24) := by
  sorry

end two_numbers_problem_l9_931


namespace egg_count_l9_922

theorem egg_count (initial_eggs used_eggs chickens eggs_per_chicken : ℕ) :
  initial_eggs ≥ used_eggs →
  (initial_eggs - used_eggs) + chickens * eggs_per_chicken =
  initial_eggs - used_eggs + chickens * eggs_per_chicken :=
by sorry

end egg_count_l9_922


namespace problem_one_l9_987

theorem problem_one : 
  64.83 - 5 * (18/19 : ℚ) + 35.17 - 44 * (1/19 : ℚ) = 50 := by sorry

end problem_one_l9_987


namespace first_five_terms_of_sequence_l9_975

def a (n : ℕ) : ℤ := (-1: ℤ)^n + n

theorem first_five_terms_of_sequence :
  (a 1 = 0) ∧ (a 2 = 3) ∧ (a 3 = 2) ∧ (a 4 = 5) ∧ (a 5 = 4) :=
by sorry

end first_five_terms_of_sequence_l9_975


namespace password_length_l9_952

/-- Represents the structure of Pat's password --/
structure PasswordStructure where
  lowercase_letters : Nat
  alternating_chars : Nat
  digits : Nat
  symbols : Nat

/-- Theorem stating that Pat's password contains 22 characters --/
theorem password_length (pw : PasswordStructure) 
  (h1 : pw.lowercase_letters = 10)
  (h2 : pw.alternating_chars = 6)
  (h3 : pw.digits = 4)
  (h4 : pw.symbols = 2) : 
  pw.lowercase_letters + pw.alternating_chars + pw.digits + pw.symbols = 22 := by
  sorry

#check password_length

end password_length_l9_952


namespace hyperbola_eccentricity_l9_954

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) is 2,
    given that one of its asymptotes is tangent to the circle (x - √3)² + (y - 1)² = 1. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 ∧
    ((x - Real.sqrt 3)^2 + (y - 1)^2 = 1 ∨
     (x + Real.sqrt 3)^2 + (y - 1)^2 = 1)) →
  Real.sqrt ((a^2 + b^2) / a^2) = 2 :=
sorry

end hyperbola_eccentricity_l9_954


namespace range_of_a_for_increasing_f_l9_949

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  (a ≥ 3/2 ∧ a < 3) :=
by sorry

end range_of_a_for_increasing_f_l9_949


namespace green_and_yellow_peaches_count_l9_984

/-- Given a basket of peaches, prove that the total number of green and yellow peaches is 20. -/
theorem green_and_yellow_peaches_count (yellow_peaches green_peaches : ℕ) 
  (h1 : yellow_peaches = 14)
  (h2 : green_peaches = 6) : 
  yellow_peaches + green_peaches = 20 := by
  sorry

end green_and_yellow_peaches_count_l9_984


namespace unique_alpha_l9_940

def A (α : ℝ) : Set ℕ := {n : ℕ | ∃ k : ℕ, n = ⌊k * α⌋}

theorem unique_alpha : ∃! α : ℝ, 
  α ≥ 1 ∧ 
  (∃ r : ℕ, r < 2021 ∧ 
    (∀ n : ℕ, n > 0 → (n ∉ A α ↔ n % 2021 = r))) ∧
  α = 2021 / 2020 := by
sorry

end unique_alpha_l9_940


namespace total_distance_meters_l9_973

def distance_feet : ℝ := 30
def feet_to_meters : ℝ := 0.3048
def num_trips : ℕ := 4

theorem total_distance_meters : 
  distance_feet * feet_to_meters * (num_trips : ℝ) = 36.576 := by
  sorry

end total_distance_meters_l9_973


namespace prob_same_heads_value_l9_919

-- Define the probability of heads for a fair coin
def fair_coin_prob : ℚ := 1/2

-- Define the probability of heads for the biased coin
def biased_coin_prob : ℚ := 5/8

-- Define the number of fair coins
def num_fair_coins : ℕ := 3

-- Define the number of biased coins
def num_biased_coins : ℕ := 1

-- Define the total number of coins
def total_coins : ℕ := num_fair_coins + num_biased_coins

-- Define the function to calculate the probability of getting the same number of heads
def prob_same_heads : ℚ := sorry

-- Theorem statement
theorem prob_same_heads_value : prob_same_heads = 77/225 := by sorry

end prob_same_heads_value_l9_919


namespace factor_expression_l9_925

theorem factor_expression (x : ℝ) : 3*x*(x-5) + 7*(x-5) - 2*(x-5) = (3*x+5)*(x-5) := by
  sorry

end factor_expression_l9_925


namespace binomial_coefficient_ratio_l9_927

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 1 / 3 ∧
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 3 / 5 →
  n + k = 8 := by
  sorry

end binomial_coefficient_ratio_l9_927


namespace symmetric_line_equation_l9_948

/-- Given a line L1 with equation x - 2y + 1 = 0, 
    and a line of symmetry y = x,
    the line L2 symmetric to L1 with respect to y = x
    has the equation x + 2y - 1 = 0 -/
theorem symmetric_line_equation :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ x - 2*y + 1 = 0
  let symmetry_line : ℝ → ℝ → Prop := λ x y ↦ y = x
  let L2 : ℝ → ℝ → Prop := λ x y ↦ x + 2*y - 1 = 0
  ∀ x y : ℝ, L2 x y ↔ (∃ x' y' : ℝ, L1 x' y' ∧ 
    (x = (x' + y')/2 ∧ y = (x' + y')/2))
:= by sorry

end symmetric_line_equation_l9_948


namespace break_even_point_l9_912

/-- The break-even point for a company producing exam preparation manuals -/
theorem break_even_point (Q : ℝ) :
  (Q > 0) →  -- Ensure Q is positive for division
  (300 : ℝ) = 100 + 100000 / Q →  -- Price equals average cost
  Q = 500 := by
sorry

end break_even_point_l9_912


namespace binomial_18_12_l9_977

theorem binomial_18_12 (h1 : Nat.choose 17 10 = 19448)
                        (h2 : Nat.choose 17 11 = 12376)
                        (h3 : Nat.choose 19 12 = 50388) :
  Nat.choose 18 12 = 18564 := by
  sorry

end binomial_18_12_l9_977


namespace equation_positive_root_m_value_l9_934

theorem equation_positive_root_m_value (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ m / (x - 3) - 1 / (3 - x) = 2) →
  m = -1 := by
sorry

end equation_positive_root_m_value_l9_934


namespace quadratic_factorization_l9_972

/-- A quadratic expression of the form 15x^2 + ax + 15 can be factored into two linear binomial 
    factors with integer coefficients if and only if a = 34 -/
theorem quadratic_factorization (a : ℤ) : 
  (∃ (m n p q : ℤ), 15 * X^2 + a * X + 15 = (m * X + n) * (p * X + q)) ↔ a = 34 :=
by sorry

end quadratic_factorization_l9_972


namespace book_selling_price_l9_946

-- Define the cost price and profit rate
def cost_price : ℝ := 50
def profit_rate : ℝ := 0.20

-- Define the selling price function
def selling_price (cost : ℝ) (rate : ℝ) : ℝ :=
  cost * (1 + rate)

-- Theorem statement
theorem book_selling_price :
  selling_price cost_price profit_rate = 60 := by
  sorry

end book_selling_price_l9_946


namespace susie_large_rooms_l9_969

/-- Represents the number of rooms of each size in Susie's house. -/
structure RoomCounts where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the time needed to vacuum each type of room. -/
structure VacuumTimes where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total vacuuming time for all rooms. -/
def totalVacuumTime (counts : RoomCounts) (times : VacuumTimes) : Nat :=
  counts.small * times.small + counts.medium * times.medium + counts.large * times.large

/-- The theorem stating that given the conditions, Susie has 2 large rooms. -/
theorem susie_large_rooms : 
  ∀ (counts : RoomCounts) (times : VacuumTimes),
    counts.small = 4 →
    counts.medium = 3 →
    times.small = 15 →
    times.medium = 25 →
    times.large = 35 →
    totalVacuumTime counts times = 225 →
    counts.large = 2 := by
  sorry

end susie_large_rooms_l9_969


namespace intersection_problem_l9_998

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_problem (a : ℝ) :
  (a = 1/2 → A a ∩ B = {x | 0 < x ∧ x < 1}) ∧
  (A a ∩ B = ∅ ↔ a ≤ -1/2 ∨ a ≥ 2) :=
sorry

end intersection_problem_l9_998


namespace rod_string_equilibrium_theorem_l9_995

/-- Represents the equilibrium conditions for a rod and string system --/
def rod_string_equilibrium (a b : ℝ) (θ : ℝ) : Prop :=
  (θ = 0 ∨ (θ.cos = (b^2 + 2*a^2) / (3*a*b) ∧ 1/2 * b < a ∧ a ≤ b)) ∧ a > 0 ∧ b > 0

/-- Theorem stating the equilibrium conditions for the rod and string system --/
theorem rod_string_equilibrium_theorem (a b : ℝ) (θ : ℝ) :
  a > 0 → b > 0 → rod_string_equilibrium a b θ ↔
    (θ = 0 ∨ (θ.cos = (b^2 + 2*a^2) / (3*a*b) ∧ 1/2 * b < a ∧ a ≤ b)) :=
by sorry

end rod_string_equilibrium_theorem_l9_995


namespace animal_count_l9_918

theorem animal_count (frogs : ℕ) (h1 : frogs = 160) : ∃ (dogs cats : ℕ),
  frogs = 2 * dogs ∧
  cats = dogs - dogs / 5 ∧
  frogs + dogs + cats = 304 := by
sorry

end animal_count_l9_918


namespace f_eval_approx_l9_902

/-- The polynomial function f(x) -/
def f (x : ℝ) : ℝ := 1 + x + 0.5*x^2 + 0.16667*x^3 + 0.04167*x^4 + 0.00833*x^5

/-- The evaluation point -/
def x₀ : ℝ := -0.2

/-- The theorem stating that f(x₀) is approximately equal to 0.81873 -/
theorem f_eval_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ |f x₀ - 0.81873| < ε := by
  sorry

end f_eval_approx_l9_902


namespace alligator_growth_rate_l9_916

def alligator_population (initial_population : ℕ) (rate : ℕ) (periods : ℕ) : ℕ :=
  initial_population + rate * periods

theorem alligator_growth_rate :
  ∀ (rate : ℕ),
    alligator_population 4 rate 2 = 16 →
    rate = 6 := by
  sorry

end alligator_growth_rate_l9_916


namespace cake_fraction_eaten_l9_907

theorem cake_fraction_eaten (total_slices : ℕ) (kept_slices : ℕ) : 
  total_slices = 12 → kept_slices = 9 → (total_slices - kept_slices : ℚ) / total_slices = 1/4 := by
  sorry

end cake_fraction_eaten_l9_907


namespace polygon_with_120_degree_interior_angles_has_6_sides_l9_963

theorem polygon_with_120_degree_interior_angles_has_6_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 120 →
    (n - 2) * 180 = n * interior_angle →
    n = 6 :=
by
  sorry

end polygon_with_120_degree_interior_angles_has_6_sides_l9_963


namespace f_upper_bound_l9_920

def f_property (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 ≤ x ∧ 0 ≤ y → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ 2016)

theorem f_upper_bound (f : ℝ → ℝ) (h : f_property f) :
  ∀ x, 0 ≤ x → f x ≤ x^2 := by sorry

end f_upper_bound_l9_920


namespace taxi_ride_cost_l9_983

def base_fee : ℚ := 1.5
def cost_per_mile : ℚ := 0.25
def ride1_distance : ℕ := 5
def ride2_distance : ℕ := 8
def ride3_distance : ℕ := 3

theorem taxi_ride_cost : 
  (base_fee + cost_per_mile * ride1_distance) + 
  (base_fee + cost_per_mile * ride2_distance) + 
  (base_fee + cost_per_mile * ride3_distance) = 8.5 := by
sorry

end taxi_ride_cost_l9_983


namespace total_amount_shared_l9_941

/-- The total amount shared by A, B, and C given specific conditions -/
theorem total_amount_shared (a b c : ℝ) : 
  a = (1/3) * (b + c) →  -- A gets one-third of what B and C together get
  b = (2/7) * (a + c) →  -- B gets two-sevenths of what A and C together get
  a = b + 35 →           -- A's amount is $35 more than B's amount
  a + b + c = 1260 :=    -- The total amount shared
by sorry

end total_amount_shared_l9_941


namespace four_digit_multiples_of_five_l9_985

theorem four_digit_multiples_of_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 9000)).card = 1800 := by
  sorry

end four_digit_multiples_of_five_l9_985


namespace max_sum_abs_on_unit_sphere_l9_909

theorem max_sum_abs_on_unit_sphere :
  ∃ (M : ℝ), M = Real.sqrt 3 ∧
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |x| + |y| + |z| ≤ M) ∧
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 1 ∧ |x| + |y| + |z| = M) :=
by sorry

end max_sum_abs_on_unit_sphere_l9_909


namespace pen_sale_profit_percentage_l9_915

/-- Calculate the profit percentage for a store owner's pen sale --/
theorem pen_sale_profit_percentage 
  (purchase_quantity : ℕ) 
  (marked_price_quantity : ℕ) 
  (discount_percentage : ℝ) : ℝ :=
by
  -- Assume purchase_quantity = 200
  -- Assume marked_price_quantity = 180
  -- Assume discount_percentage = 2
  
  -- Define cost price
  let cost_price := marked_price_quantity

  -- Define selling price per item
  let selling_price_per_item := 1 - (1 * discount_percentage / 100)

  -- Calculate total revenue
  let total_revenue := purchase_quantity * selling_price_per_item

  -- Calculate profit
  let profit := total_revenue - cost_price

  -- Calculate profit percentage
  let profit_percentage := (profit / cost_price) * 100

  -- Prove that profit_percentage ≈ 8.89
  sorry

-- The statement of the theorem
#check pen_sale_profit_percentage

end pen_sale_profit_percentage_l9_915


namespace boris_candy_problem_l9_947

/-- Given the initial number of candy pieces, the number eaten by the daughter,
    the number of bowls, and the number of pieces taken from each bowl,
    calculate the final number of pieces in one bowl. -/
def candyInBowl (initial : ℕ) (eaten : ℕ) (bowls : ℕ) (taken : ℕ) : ℕ :=
  (initial - eaten) / bowls - taken

theorem boris_candy_problem :
  candyInBowl 100 8 4 3 = 20 := by
  sorry

end boris_candy_problem_l9_947


namespace sqrt_six_times_sqrt_three_l9_962

theorem sqrt_six_times_sqrt_three : Real.sqrt 6 * Real.sqrt 3 = 3 * Real.sqrt 2 := by
  sorry

end sqrt_six_times_sqrt_three_l9_962


namespace batsman_average_increase_l9_976

def batsman_average (total_runs : ℕ) (innings : ℕ) : ℚ :=
  (total_runs : ℚ) / (innings : ℚ)

theorem batsman_average_increase
  (prev_total : ℕ)
  (new_score : ℕ)
  (innings : ℕ)
  (avg_increase : ℚ)
  (h1 : innings = 17)
  (h2 : new_score = 88)
  (h3 : avg_increase = 3)
  (h4 : batsman_average (prev_total + new_score) innings - batsman_average prev_total (innings - 1) = avg_increase) :
  batsman_average (prev_total + new_score) innings = 40 :=
by
  sorry

#check batsman_average_increase

end batsman_average_increase_l9_976


namespace range_of_m_l9_944

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.log x + x^2 - a*x + 2

theorem range_of_m (a : ℝ) (x₀ : ℝ) :
  (∀ a ∈ Set.Icc (-2) 0, ∃ x₀ ∈ Set.Ioc 0 1, 
    f x₀ a > a^2 + 3*a + 2 - 2*m*(Real.exp a)*(a+1)) →
  m ∈ Set.Icc (-1/2) (5*(Real.exp 2)/2) :=
sorry

end range_of_m_l9_944


namespace vector_addition_l9_951

theorem vector_addition : 
  let v1 : Fin 2 → ℝ := ![- 5, 3]
  let v2 : Fin 2 → ℝ := ![7, -6]
  v1 + v2 = ![2, -3] :=
by sorry

end vector_addition_l9_951


namespace same_solution_for_k_17_l9_917

theorem same_solution_for_k_17 :
  ∃ x : ℝ, (2 * x + 4 = 4 * (x - 2)) ∧ (17 * x - 91 = 2 * x - 1) := by
  sorry

#check same_solution_for_k_17

end same_solution_for_k_17_l9_917


namespace sum_of_common_ratios_is_five_l9_942

/-- Given two nonconstant geometric sequences with different common ratios,
    if a specific condition is met, prove that the sum of their common ratios is 5. -/
theorem sum_of_common_ratios_is_five
  (k : ℝ) (p r : ℝ) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r)
  (h : k * p^2 - k * r^2 = 5 * (k * p - k * r)) (hk : k ≠ 0) :
  p + r = 5 := by
  sorry

end sum_of_common_ratios_is_five_l9_942


namespace polygon_with_72_degree_exterior_angles_has_5_sides_l9_981

/-- A polygon with exterior angles each measuring 72° has 5 sides -/
theorem polygon_with_72_degree_exterior_angles_has_5_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    exterior_angle = 72 →
    n * exterior_angle = 360 →
    n = 5 := by
  sorry

end polygon_with_72_degree_exterior_angles_has_5_sides_l9_981


namespace inequality_proof_l9_932

theorem inequality_proof (x : ℝ) : 3 * (2 * x - 1) - 2 * (x + 1) ≤ 1 → x ≤ 3 / 2 := by
  sorry

end inequality_proof_l9_932


namespace log_sum_equality_l9_980

-- Define the theorem
theorem log_sum_equality (p q : ℝ) (h : q ≠ 1) :
  Real.log p + Real.log q = Real.log (p + 2*q) → p = 2*q / (q - 1) := by
  sorry

end log_sum_equality_l9_980


namespace A_B_symmetrical_wrt_origin_l9_943

/-- Two points are symmetrical with respect to the origin if their coordinates are negatives of each other -/
def symmetrical_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

/-- Given points A and B in the Cartesian coordinate system -/
def A : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (-2, 1)

/-- Theorem: Points A and B are symmetrical with respect to the origin -/
theorem A_B_symmetrical_wrt_origin : symmetrical_wrt_origin A B := by
  sorry

end A_B_symmetrical_wrt_origin_l9_943


namespace ellipse_eccentricity_ratio_l9_906

/-- For an ellipse with equation mx^2 + ny^2 = 1, foci on the x-axis, and eccentricity 1/2,
    the ratio m/n is equal to 3/4. -/
theorem ellipse_eccentricity_ratio (m n : ℝ) (m_pos : 0 < m) (n_pos : 0 < n) :
  (∀ x y : ℝ, m * x^2 + n * y^2 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c^2 = 1/m - 1/n) →  -- Foci on x-axis condition
  ((1 - 1/n) / (1/m))^(1/2) = 1/2 →  -- Eccentricity condition
  m / n = 3/4 := by sorry

end ellipse_eccentricity_ratio_l9_906


namespace ceiling_minus_half_integer_l9_970

theorem ceiling_minus_half_integer (n : ℤ) : 
  let x : ℝ := n + 1/2
  ⌈x⌉ - x = 1/2 := by sorry

end ceiling_minus_half_integer_l9_970


namespace difference_of_squares_equals_cube_l9_964

theorem difference_of_squares_equals_cube (r : ℕ+) :
  ∃ m n : ℤ, m^2 - n^2 = (r : ℤ)^3 := by
  sorry

end difference_of_squares_equals_cube_l9_964


namespace quadratic_roots_property_l9_967

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 - 2*m - 2025 = 0) → 
  (n^2 - 2*n - 2025 = 0) → 
  (m^2 - 3*m - n = 2023) := by
sorry

end quadratic_roots_property_l9_967


namespace square_1011_position_l9_986

-- Define the possible positions of the square
inductive SquarePosition
| ABCD
| BCDA
| DCBA

-- Define the transformation function
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.BCDA
  | SquarePosition.BCDA => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

-- Define the function to get the nth position
def nthPosition (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.BCDA
  | 2 => SquarePosition.DCBA
  | _ => SquarePosition.ABCD

-- Theorem stating that the 1011th square is in position DCBA
theorem square_1011_position :
  nthPosition 1011 = SquarePosition.DCBA := by
  sorry

end square_1011_position_l9_986


namespace geometric_sequence_minimum_l9_939

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a (k + 1) = 3 * a k) →  -- Common ratio is 3
  (∀ k, a k > 0) →  -- Positive terms
  a m * a n = 9 * a 2 ^ 2 →  -- Given condition
  (∀ p q : ℕ, a p * a q = 9 * a 2 ^ 2 → 2 / m + 1 / (2 * n) ≤ 2 / p + 1 / (2 * q)) →
  2 / m + 1 / (2 * n) = 3 / 4 :=
by sorry

end geometric_sequence_minimum_l9_939


namespace coolant_replacement_l9_978

/-- Calculates the amount of original coolant left in a car's cooling system after partial replacement. -/
theorem coolant_replacement (initial_volume : ℝ) (initial_concentration : ℝ) 
  (replacement_concentration : ℝ) (final_concentration : ℝ) 
  (h1 : initial_volume = 19) 
  (h2 : initial_concentration = 0.3)
  (h3 : replacement_concentration = 0.8)
  (h4 : final_concentration = 0.5) : 
  initial_volume - (final_concentration * initial_volume - initial_concentration * initial_volume) / 
  (replacement_concentration - initial_concentration) = 11.4 := by
sorry

end coolant_replacement_l9_978


namespace odd_number_proposition_l9_905

theorem odd_number_proposition (P : ℕ → Prop) 
  (base : P 1) 
  (step : ∀ k : ℕ, P k → P (k + 2)) : 
  ∀ n : ℕ, Odd n → P n :=
sorry

end odd_number_proposition_l9_905


namespace twelfth_day_is_monday_l9_979

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  numDays : Nat
  numFridays : Nat
  firstNotFriday : firstDay ≠ DayOfWeek.Friday
  lastNotFriday : lastDay ≠ DayOfWeek.Friday
  exactlyFiveFridays : numFridays = 5

/-- Function to determine the day of the week for a given day number -/
def dayOfWeekForDay (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Theorem stating that the 12th day is a Monday -/
theorem twelfth_day_is_monday (m : Month) :
  dayOfWeekForDay m 12 = DayOfWeek.Monday :=
sorry

end twelfth_day_is_monday_l9_979


namespace sisters_sandcastle_height_l9_955

theorem sisters_sandcastle_height 
  (janet_height : Float) 
  (height_difference : Float) 
  (h1 : janet_height = 3.6666666666666665) 
  (h2 : height_difference = 1.3333333333333333) : 
  janet_height - height_difference = 2.333333333333333 := by
  sorry

end sisters_sandcastle_height_l9_955


namespace safety_cost_per_mile_approx_l9_997

/-- Safety Rent-a-Car's daily rate -/
def safety_daily_rate : ℝ := 21.95

/-- City Rentals' daily rate -/
def city_daily_rate : ℝ := 18.95

/-- City Rentals' cost per mile -/
def city_cost_per_mile : ℝ := 0.21

/-- Number of miles for equal cost -/
def equal_cost_miles : ℝ := 150.0

/-- Safety Rent-a-Car's cost per mile -/
noncomputable def safety_cost_per_mile : ℝ := 
  (city_daily_rate + city_cost_per_mile * equal_cost_miles - safety_daily_rate) / equal_cost_miles

theorem safety_cost_per_mile_approx :
  ∃ ε > 0, abs (safety_cost_per_mile - 0.177) < ε ∧ ε < 0.001 :=
sorry

end safety_cost_per_mile_approx_l9_997


namespace mango_rate_is_65_l9_957

/-- The rate per kg for mangoes given the following conditions:
    - Tom purchased 8 kg of apples at 70 per kg
    - Tom purchased 9 kg of mangoes
    - Tom paid a total of 1145 to the shopkeeper -/
def mango_rate (apple_weight : ℕ) (apple_rate : ℕ) (mango_weight : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid - apple_weight * apple_rate) / mango_weight

/-- Theorem stating that the rate per kg for mangoes is 65 -/
theorem mango_rate_is_65 : mango_rate 8 70 9 1145 = 65 := by
  sorry

end mango_rate_is_65_l9_957


namespace exponent_properties_l9_911

theorem exponent_properties (a m n : ℝ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) (h4 : n > 1) :
  (a^(m/n) = (a^m)^(1/n)) ∧ 
  (a^0 = 1) ∧ 
  (a^(-m/n) = 1 / (a^m)^(1/n)) := by
sorry

end exponent_properties_l9_911


namespace max_students_with_different_options_l9_938

/-- Represents an answer sheet for a test with 6 questions, each with 3 options -/
def AnswerSheet := Fin 6 → Fin 3

/-- Checks if three answer sheets have at least one question where all options are different -/
def hasDifferentOptions (s1 s2 s3 : AnswerSheet) : Prop :=
  ∃ q : Fin 6, s1 q ≠ s2 q ∧ s1 q ≠ s3 q ∧ s2 q ≠ s3 q

/-- The main theorem stating the maximum number of students -/
theorem max_students_with_different_options :
  ∀ n : ℕ,
  (∀ sheets : Fin n → AnswerSheet,
    ∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
      hasDifferentOptions (sheets i) (sheets j) (sheets k)) →
  n ≤ 13 :=
sorry

end max_students_with_different_options_l9_938


namespace evaluate_g_l9_945

/-- The function g(x) = 3x^2 - 5x + 7 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

/-- Theorem: 3g(5) + 4g(-2) = 287 -/
theorem evaluate_g : 3 * g 5 + 4 * g (-2) = 287 := by
  sorry

end evaluate_g_l9_945


namespace tan_alpha_value_l9_960

theorem tan_alpha_value (α β : Real) 
  (h1 : Real.tan (α - β) = 2) 
  (h2 : Real.tan β = 3) : 
  Real.tan α = -1 := by
  sorry

end tan_alpha_value_l9_960


namespace quadratic_inequality_l9_936

theorem quadratic_inequality (x : ℝ) : (x - 2) * (x + 2) > 0 ↔ x > 2 ∨ x < -2 := by
  sorry

end quadratic_inequality_l9_936


namespace bus_capacity_fraction_l9_926

/-- The capacity of the train in number of people -/
def train_capacity : ℕ := 120

/-- The combined capacity of the two buses in number of people -/
def combined_bus_capacity : ℕ := 40

/-- The fraction of the train's capacity that each bus can hold -/
def bus_fraction : ℚ := 1 / 6

theorem bus_capacity_fraction :
  bus_fraction = combined_bus_capacity / (2 * train_capacity) :=
sorry

end bus_capacity_fraction_l9_926


namespace incenter_centroid_parallel_implies_arithmetic_sequence_l9_908

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle. -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The centroid of a triangle. -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Checks if two points form a line parallel to a side of the triangle. -/
def is_parallel_to_side (p q : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- The lengths of the sides of a triangle. -/
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- Checks if three numbers form an arithmetic sequence. -/
def is_arithmetic_sequence (a b c : ℝ) : Prop := sorry

theorem incenter_centroid_parallel_implies_arithmetic_sequence (t : Triangle) :
  is_parallel_to_side (incenter t) (centroid t) t →
  let (a, b, c) := side_lengths t
  is_arithmetic_sequence a b c := by sorry

end incenter_centroid_parallel_implies_arithmetic_sequence_l9_908


namespace escalator_length_is_210_l9_989

/-- The length of an escalator given its speed, a person's walking speed, and the time taken to cover the entire length. -/
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ :=
  (escalator_speed + person_speed) * time

/-- Theorem stating that under the given conditions, the escalator length is 210 feet. -/
theorem escalator_length_is_210 :
  escalator_length 12 2 15 = 210 := by
  sorry

#eval escalator_length 12 2 15

end escalator_length_is_210_l9_989


namespace polynomial_simplification_l9_971

-- Define the polynomials A, B, and C
def A (x : ℝ) : ℝ := 5 * x^2 + 4 * x - 1
def B (x : ℝ) : ℝ := -x^2 - 3 * x + 3
def C (x : ℝ) : ℝ := 8 - 7 * x - 6 * x^2

-- Theorem statement
theorem polynomial_simplification (x : ℝ) : A x - B x + C x = 4 := by
  sorry

end polynomial_simplification_l9_971


namespace fourth_side_is_six_l9_966

/-- Represents a quadrilateral pyramid with a base ABCD and apex S -/
structure QuadrilateralPyramid where
  /-- Length of side AB of the base -/
  ab : ℝ
  /-- Length of side BC of the base -/
  bc : ℝ
  /-- Length of side CD of the base -/
  cd : ℝ
  /-- Length of side DA of the base -/
  da : ℝ
  /-- Predicate indicating that all dihedral angles at the base are equal -/
  equal_dihedral_angles : Prop

/-- Theorem stating that for a quadrilateral pyramid with given side lengths and equal dihedral angles,
    the fourth side of the base is 6 -/
theorem fourth_side_is_six (p : QuadrilateralPyramid)
  (h1 : p.ab = 5)
  (h2 : p.bc = 7)
  (h3 : p.cd = 8)
  (h4 : p.equal_dihedral_angles) :
  p.da = 6 := by
  sorry

end fourth_side_is_six_l9_966


namespace trig_ratio_sum_l9_994

theorem trig_ratio_sum (a b : ℝ) 
  (h1 : (Real.sin a / Real.cos b) + (Real.sin b / Real.cos a) = 3)
  (h2 : (Real.cos a / Real.sin b) + (Real.cos b / Real.sin a) = 4) :
  (Real.tan a / Real.tan b) + (Real.tan b / Real.tan a) = 40/3 := by
  sorry

end trig_ratio_sum_l9_994


namespace existence_uniqueness_midpoint_l9_900

/-- Polygonal distance between two points -/
def polygonal_distance (A B : ℝ × ℝ) : ℝ :=
  |A.1 - B.1| + |A.2 - B.2|

/-- Theorem: Existence and uniqueness of point C satisfying given conditions -/
theorem existence_uniqueness_midpoint (A B : ℝ × ℝ) (h : A ≠ B) :
  ∃! C : ℝ × ℝ, 
    polygonal_distance A C + polygonal_distance C B = polygonal_distance A B ∧
    polygonal_distance A C = polygonal_distance C B ∧
    C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) := by
  sorry

#check existence_uniqueness_midpoint

end existence_uniqueness_midpoint_l9_900


namespace max_triples_1955_l9_968

/-- The maximum number of triples that can be chosen from a set of points,
    such that each pair of triples has one point in common. -/
def max_triples (n : ℕ) : ℕ :=
  ((n - 1) * (n - 2)) / 4

/-- Theorem stating that for 1955 points, the maximum number of triples
    that can be chosen such that each pair of triples has one point in
    common is 977. -/
theorem max_triples_1955 :
  max_triples 1955 = 977 := by
  sorry


end max_triples_1955_l9_968


namespace symmetry_about_59_l9_961

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the two functions y₁ and y₂
def y₁ (x : ℝ) : ℝ := f (x - 19)
def y₂ (x : ℝ) : ℝ := f (99 - x)

-- Theorem stating that y₁ and y₂ are symmetric about x = 59
theorem symmetry_about_59 :
  ∀ (x : ℝ), y₁ f (118 - x) = y₂ f x :=
sorry

end symmetry_about_59_l9_961


namespace yearly_pet_feeding_cost_l9_937

/-- Calculate the yearly cost to feed Harry's pets -/
theorem yearly_pet_feeding_cost :
  let num_geckos : ℕ := 3
  let num_iguanas : ℕ := 2
  let num_snakes : ℕ := 4
  let gecko_cost_per_month : ℕ := 15
  let iguana_cost_per_month : ℕ := 5
  let snake_cost_per_month : ℕ := 10
  let months_per_year : ℕ := 12
  
  (num_geckos * gecko_cost_per_month + 
   num_iguanas * iguana_cost_per_month + 
   num_snakes * snake_cost_per_month) * months_per_year = 1140 := by
  sorry

end yearly_pet_feeding_cost_l9_937


namespace compare_fractions_l9_982

theorem compare_fractions (a b : ℝ) (h : a + b > 0) :
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b := by
  sorry

end compare_fractions_l9_982
