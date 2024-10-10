import Mathlib

namespace complex_fraction_simplification_l1716_171649

/-- The complex number i -/
noncomputable def i : ℂ := Complex.I

/-- Proof that (1+i)/(1-i) = i -/
theorem complex_fraction_simplification : (1 + i) / (1 - i) = i := by
  sorry

end complex_fraction_simplification_l1716_171649


namespace correct_average_l1716_171674

theorem correct_average (n : ℕ) (initial_avg : ℚ) (error1 : ℚ) (error2 : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  error1 = 16 →
  error2 = 18 →
  (n : ℚ) * initial_avg - error1 + error2 = n * 40.4 := by
  sorry

end correct_average_l1716_171674


namespace group_average_difference_l1716_171676

/-- Represents the first element of the n-th group -/
def a (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n % 2 = 1 then a (n - 1) + (n - 1)
  else a (n - 1) + (n - 2)

/-- Sum of elements in the n-th group -/
def S (n : ℕ) : ℕ :=
  n * (2 * a n + (n - 1) * 2) / 2

/-- Average of elements in the n-th group -/
def avg (n : ℕ) : ℚ :=
  (S n : ℚ) / n

theorem group_average_difference (n : ℕ) :
  avg (2 * n + 1) - avg (2 * n) = 2 * n := by
  sorry

end group_average_difference_l1716_171676


namespace split_meal_cost_l1716_171661

def meal_contribution (total_price : ℚ) (coupon_value : ℚ) (num_people : ℕ) : ℚ :=
  (total_price - coupon_value) / num_people

theorem split_meal_cost :
  meal_contribution 67 4 3 = 21 := by
  sorry

end split_meal_cost_l1716_171661


namespace king_then_ace_probability_l1716_171687

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of Kings in a standard deck -/
def numKings : ℕ := 4

/-- The number of Aces in a standard deck -/
def numAces : ℕ := 4

/-- The probability of drawing a King first and an Ace second from a standard deck -/
def probKingThenAce : ℚ := (numKings : ℚ) / standardDeckSize * numAces / (standardDeckSize - 1)

theorem king_then_ace_probability :
  probKingThenAce = 4 / 663 := by
  sorry

end king_then_ace_probability_l1716_171687


namespace gcd_of_three_numbers_l1716_171630

theorem gcd_of_three_numbers : Nat.gcd 779 (Nat.gcd 209 589) = 19 := by
  sorry

end gcd_of_three_numbers_l1716_171630


namespace weight_loss_program_result_l1716_171672

/-- Calculates the final weight after a weight loss program -/
def finalWeight (initialWeight : ℕ) (weeklyLoss1 weeklyLoss2 : ℕ) (weeks1 weeks2 : ℕ) : ℕ :=
  initialWeight - (weeklyLoss1 * weeks1 + weeklyLoss2 * weeks2)

/-- Proves that the weight loss program results in the correct final weight -/
theorem weight_loss_program_result :
  finalWeight 250 3 2 4 8 = 222 := by
  sorry

end weight_loss_program_result_l1716_171672


namespace range_of_a_l1716_171683

def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 ≥ a

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) :
  prop_p a ∧ prop_q a ↔ a ∈ Set.Iic (-2) ∪ {1} :=
by sorry

end range_of_a_l1716_171683


namespace complex_number_properties_l1716_171680

theorem complex_number_properties (z : ℂ) (h : z = 1 + I) : 
  (Complex.abs z = Real.sqrt 2) ∧ 
  (z ≠ 1 - I) ∧
  (z.im ≠ 1) ∧
  (0 < z.re ∧ 0 < z.im) :=
by sorry

end complex_number_properties_l1716_171680


namespace pure_imaginary_product_l1716_171612

theorem pure_imaginary_product (x : ℝ) : 
  (x^4 + 6*x^3 + 7*x^2 - 14*x - 12 = 0) ↔ (x = -4 ∨ x = -3 ∨ x = -1 ∨ x = 2) :=
by sorry

end pure_imaginary_product_l1716_171612


namespace quadratic_function_property_l1716_171660

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_property (f : ℝ → ℝ) (h_quad : is_quadratic f)
  (h_cond : ∀ a b : ℝ, a ≠ b → f a = f b → f (a^2 - 6*b - 1) = f (b^2 + 8)) :
  f 2 = f 1 := by
  sorry

end quadratic_function_property_l1716_171660


namespace heidi_painting_rate_l1716_171690

theorem heidi_painting_rate (total_time minutes : ℕ) (fraction : ℚ) : 
  (total_time = 30) → (minutes = 10) → (fraction = 1 / 3) →
  (fraction = (minutes : ℚ) / total_time) :=
by sorry

end heidi_painting_rate_l1716_171690


namespace g_neg_one_value_l1716_171682

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of y = f(x) + x^2 being an odd function
def is_odd_composite (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) + (-x)^2 = -(f x + x^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem g_neg_one_value (f : ℝ → ℝ) 
  (h1 : is_odd_composite f) 
  (h2 : f 1 = 1) : 
  g f (-1) = -1 := by
  sorry

end g_neg_one_value_l1716_171682


namespace tan_alpha_value_l1716_171678

theorem tan_alpha_value (α : ℝ) :
  (3 * Real.sin (Real.pi + α) + Real.cos (-α)) / (4 * Real.sin (-α) - Real.cos (9 * Real.pi + α)) = 2 →
  Real.tan α = 1 / 5 := by
  sorry

end tan_alpha_value_l1716_171678


namespace perfect_score_correct_l1716_171638

/-- The perfect score for a single game, given that 3 perfect games result in 63 points. -/
def perfect_score : ℕ := 21

/-- The total score for three perfect games. -/
def three_game_score : ℕ := 63

/-- Theorem stating that the perfect score for a single game is correct. -/
theorem perfect_score_correct : perfect_score * 3 = three_game_score := by
  sorry

end perfect_score_correct_l1716_171638


namespace circular_pool_volume_l1716_171651

/-- The volume of a circular cylinder with diameter 80 feet and height 10 feet is approximately 50265.6 cubic feet. -/
theorem circular_pool_volume :
  let diameter : ℝ := 80
  let height : ℝ := 10
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * height
  ∃ ε > 0, |volume - 50265.6| < ε :=
by sorry

end circular_pool_volume_l1716_171651


namespace quartic_roots_equivalence_l1716_171608

theorem quartic_roots_equivalence (x : ℂ) :
  (3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0) ↔
  (∃ y : ℂ, (y = x + 1/x) ∧
    ((y = (-1 + Real.sqrt 43) / 3) ∨ (y = (-1 - Real.sqrt 43) / 3))) := by
  sorry

end quartic_roots_equivalence_l1716_171608


namespace total_seedlings_sold_l1716_171627

/-- Represents the number of seedlings sold for each type -/
structure Seedlings where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Theorem stating the total number of seedlings sold given the conditions -/
theorem total_seedlings_sold (s : Seedlings) : 
  (s.A : ℚ) / s.B = 1 / 2 →
  (s.B : ℚ) / s.C = 3 / 4 →
  3 * s.A + 2 * s.B + s.C = 29000 →
  s.A + s.B + s.C = 17000 := by
  sorry

#check total_seedlings_sold

end total_seedlings_sold_l1716_171627


namespace percent_to_decimal_three_percent_to_decimal_l1716_171621

theorem percent_to_decimal (p : ℚ) : p / 100 = p * (1 / 100) := by sorry

theorem three_percent_to_decimal : (3 : ℚ) / 100 = 0.03 := by sorry

end percent_to_decimal_three_percent_to_decimal_l1716_171621


namespace division_theorem_l1716_171625

/-- The dividend polynomial -/
def p (x : ℝ) : ℝ := 3 * x^5 - 2 * x^3 + 5 * x - 8

/-- The divisor polynomial -/
def d (x : ℝ) : ℝ := x^2 - 2 * x + 1

/-- The proposed remainder -/
def r (x : ℝ) : ℝ := 14 * x - 14

/-- Theorem stating that r is the remainder when p is divided by d -/
theorem division_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = d x * q x + r x :=
sorry

end division_theorem_l1716_171625


namespace job_completion_time_l1716_171617

/-- The time taken for two workers to complete a job together, given their relative efficiencies and the time taken by one worker. -/
theorem job_completion_time 
  (p_efficiency : ℝ) 
  (q_efficiency : ℝ) 
  (p_time : ℝ) 
  (h1 : p_efficiency = q_efficiency + 0.6 * q_efficiency) 
  (h2 : p_time = 26) :
  (p_efficiency * q_efficiency * p_time) / (p_efficiency * q_efficiency + p_efficiency * p_efficiency) = 1690 / 91 := by
  sorry

end job_completion_time_l1716_171617


namespace fill_time_is_100_l1716_171673

/-- Represents the water filling system with three pipes and a tank -/
structure WaterSystem where
  tankCapacity : ℕ
  pipeARate : ℕ
  pipeBRate : ℕ
  pipeCRate : ℕ
  pipeATime : ℕ
  pipeBTime : ℕ
  pipeCTime : ℕ

/-- Calculates the time required to fill the tank -/
def fillTime (sys : WaterSystem) : ℕ :=
  let cycleAmount := sys.pipeARate * sys.pipeATime + sys.pipeBRate * sys.pipeBTime - sys.pipeCRate * sys.pipeCTime
  let cycles := (sys.tankCapacity + cycleAmount - 1) / cycleAmount
  cycles * (sys.pipeATime + sys.pipeBTime + sys.pipeCTime)

/-- Theorem stating that the fill time for the given system is 100 minutes -/
theorem fill_time_is_100 (sys : WaterSystem) 
  (h1 : sys.tankCapacity = 5000)
  (h2 : sys.pipeARate = 200)
  (h3 : sys.pipeBRate = 50)
  (h4 : sys.pipeCRate = 25)
  (h5 : sys.pipeATime = 1)
  (h6 : sys.pipeBTime = 2)
  (h7 : sys.pipeCTime = 2) :
  fillTime sys = 100 := by
  sorry

#eval fillTime { tankCapacity := 5000, pipeARate := 200, pipeBRate := 50, pipeCRate := 25, 
                 pipeATime := 1, pipeBTime := 2, pipeCTime := 2 }

end fill_time_is_100_l1716_171673


namespace age_difference_proof_l1716_171644

theorem age_difference_proof (patrick_age michael_age monica_age : ℕ) :
  patrick_age * 5 = michael_age * 3 →
  michael_age * 4 = monica_age * 3 →
  patrick_age + michael_age + monica_age = 88 →
  monica_age - patrick_age = 22 := by
sorry


end age_difference_proof_l1716_171644


namespace unique_solution_quadratic_l1716_171631

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + m * x + 16 = 0) → m = 8 * Real.sqrt 3 :=
by sorry

end unique_solution_quadratic_l1716_171631


namespace aluminum_atomic_weight_l1716_171656

/-- The atomic weight of chlorine in atomic mass units (amu) -/
def chlorine_weight : ℝ := 35.45

/-- The molecular weight of the compound in atomic mass units (amu) -/
def compound_weight : ℝ := 132

/-- The number of chlorine atoms in the compound -/
def chlorine_count : ℕ := 3

/-- The atomic weight of aluminum in atomic mass units (amu) -/
def aluminum_weight : ℝ := compound_weight - chlorine_count * chlorine_weight

theorem aluminum_atomic_weight :
  aluminum_weight = 25.65 := by sorry

end aluminum_atomic_weight_l1716_171656


namespace no_solution_l1716_171641

/-- ab is a two-digit number -/
def ab : ℕ := sorry

/-- ba is a two-digit number, which is the reverse of ab -/
def ba : ℕ := sorry

/-- ab and ba are distinct -/
axiom ab_ne_ba : ab ≠ ba

/-- There is no real number x that satisfies the equation (ab)^x - 2 = (ba)^x - 7 -/
theorem no_solution : ¬∃ x : ℝ, (ab : ℝ) ^ x - 2 = (ba : ℝ) ^ x - 7 := by sorry

end no_solution_l1716_171641


namespace corn_acreage_l1716_171610

theorem corn_acreage (total_land : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : bean_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) :
  (total_land * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end corn_acreage_l1716_171610


namespace arithmetic_sequence_sum_product_l1716_171664

theorem arithmetic_sequence_sum_product (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n / 2) * (a 1 + a n)) →  -- Definition of S_n
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- Arithmetic sequence property
  a 7 < 0 →
  a 8 > 0 →
  a 8 > |a 7| →
  S 13 * S 14 < 0 := by
sorry

end arithmetic_sequence_sum_product_l1716_171664


namespace panda_equation_l1716_171655

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Condition that all digits are distinct -/
def all_distinct (a b c d e : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

/-- Convert a two-digit number to a natural number -/
def to_nat (tens units : Digit) : ℕ :=
  10 * tens.val + units.val

/-- Convert a three-digit number to a natural number -/
def to_nat_3 (hundreds tens units : Digit) : ℕ :=
  100 * hundreds.val + 10 * tens.val + units.val

theorem panda_equation (tuan yuan da xiong mao : Digit)
  (h_distinct : all_distinct tuan yuan da xiong mao)
  (h_eq : to_nat tuan tuan * to_nat yuan yuan = to_nat_3 da xiong mao) :
  da.val + xiong.val + mao.val = 23 := by
  sorry

end panda_equation_l1716_171655


namespace time_to_write_michaels_name_l1716_171600

/-- The number of letters in Michael's name -/
def name_length : ℕ := 7

/-- The number of rearrangements Michael can write per minute -/
def rearrangements_per_minute : ℕ := 10

/-- Calculate the total number of rearrangements for a name with distinct letters -/
def total_rearrangements (n : ℕ) : ℕ := Nat.factorial n

/-- Calculate the time in hours to write all rearrangements -/
def time_to_write_all (name_len : ℕ) (rearr_per_min : ℕ) : ℚ :=
  (total_rearrangements name_len : ℚ) / (rearr_per_min : ℚ) / 60

/-- Theorem: It takes 8.4 hours to write all rearrangements of Michael's name -/
theorem time_to_write_michaels_name :
  time_to_write_all name_length rearrangements_per_minute = 84 / 10 := by
  sorry

end time_to_write_michaels_name_l1716_171600


namespace triangle_area_l1716_171611

/-- A triangle with integral sides and perimeter 12 has area 6 -/
theorem triangle_area (a b c : ℕ) : 
  a + b + c = 12 → 
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  (a * b : ℝ) / 2 = 6 := by
sorry

end triangle_area_l1716_171611


namespace certain_number_problem_l1716_171659

theorem certain_number_problem : ∃ x : ℝ, (0.60 * 50 = 0.40 * x + 18) ∧ (x = 30) := by
  sorry

end certain_number_problem_l1716_171659


namespace trig_identity_l1716_171636

theorem trig_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin (5*π/6 - x) + (Real.cos (π/3 - x))^2 = 5/16 := by
  sorry

end trig_identity_l1716_171636


namespace unique_solution_square_equation_l1716_171637

theorem unique_solution_square_equation :
  ∀ x y : ℕ+, x^2 = y^2 + 7*y + 6 ↔ x = 6 ∧ y = 3 := by
sorry

end unique_solution_square_equation_l1716_171637


namespace math_problem_l1716_171629

theorem math_problem :
  (8 * 40 = 320) ∧
  (5 * (1 / 6) = 5 / 6) ∧
  (6 * 500 = 3000) ∧
  (∃ n : ℕ, 3000 = n * 1000) := by
sorry

end math_problem_l1716_171629


namespace train_speed_l1716_171619

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 250) (h2 : crossing_time = 4) :
  train_length / crossing_time = 62.5 := by
  sorry

end train_speed_l1716_171619


namespace simple_interest_problem_l1716_171605

/-- Given a sum put at simple interest for 10 years, if increasing the interest
    rate by 5% results in Rs. 600 more interest, then the sum is Rs. 1200. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 600 → P = 1200 := by
  sorry

end simple_interest_problem_l1716_171605


namespace square_has_most_symmetry_axes_l1716_171607

/-- The number of symmetry axes for a line segment -/
def line_segment_symmetry_axes : ℕ := 2

/-- The number of symmetry axes for an angle -/
def angle_symmetry_axes : ℕ := 1

/-- The minimum number of symmetry axes for an isosceles triangle -/
def isosceles_triangle_min_symmetry_axes : ℕ := 1

/-- The maximum number of symmetry axes for an isosceles triangle -/
def isosceles_triangle_max_symmetry_axes : ℕ := 3

/-- The number of symmetry axes for a square -/
def square_symmetry_axes : ℕ := 4

/-- Theorem stating that a square has the most symmetry axes among the given shapes -/
theorem square_has_most_symmetry_axes :
  square_symmetry_axes > line_segment_symmetry_axes ∧
  square_symmetry_axes > angle_symmetry_axes ∧
  square_symmetry_axes > isosceles_triangle_min_symmetry_axes ∧
  square_symmetry_axes > isosceles_triangle_max_symmetry_axes :=
sorry

end square_has_most_symmetry_axes_l1716_171607


namespace one_basket_of_peaches_l1716_171663

def basket_count (red_peaches green_peaches total_peaches : ℕ) : ℕ :=
  if red_peaches + green_peaches = total_peaches then 1 else 0

theorem one_basket_of_peaches (red_peaches green_peaches total_peaches : ℕ) 
  (h1 : red_peaches = 7)
  (h2 : green_peaches = 3)
  (h3 : total_peaches = 10) :
  basket_count red_peaches green_peaches total_peaches = 1 := by
sorry

end one_basket_of_peaches_l1716_171663


namespace andy_solves_two_problems_l1716_171616

/-- Returns true if n is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- Returns the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Returns true if n has an odd digit sum, false otherwise -/
def hasOddDigitSum (n : ℕ) : Prop := sorry

/-- The set of numbers we're considering -/
def problemSet : Set ℕ := {n : ℕ | 78 ≤ n ∧ n ≤ 125}

/-- The count of prime numbers with odd digit sums in our problem set -/
def countPrimesWithOddDigitSum : ℕ := sorry

theorem andy_solves_two_problems : countPrimesWithOddDigitSum = 2 := by sorry

end andy_solves_two_problems_l1716_171616


namespace pi_is_irrational_one_third_is_rational_sqrt_16_is_rational_finite_decimal_is_rational_pi_only_irrational_option_l1716_171633

theorem pi_is_irrational :
  ∀ (a b : ℚ), (a : ℝ) ≠ π ∧ (b : ℝ) ≠ π → Irrational π := by
  sorry

theorem one_third_is_rational : ∃ (a b : ℤ), b ≠ 0 ∧ (1 : ℝ) / 3 = (a : ℝ) / (b : ℝ) := by
  sorry

theorem sqrt_16_is_rational : ∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 16 = (a : ℝ) / (b : ℝ) := by
  sorry

theorem finite_decimal_is_rational : ∃ (a b : ℤ), b ≠ 0 ∧ 3.1415926 = (a : ℝ) / (b : ℝ) := by
  sorry

theorem pi_only_irrational_option : Irrational π := by
  sorry

end pi_is_irrational_one_third_is_rational_sqrt_16_is_rational_finite_decimal_is_rational_pi_only_irrational_option_l1716_171633


namespace like_terms_imply_x_power_y_equals_nine_l1716_171658

theorem like_terms_imply_x_power_y_equals_nine (a b x y : ℝ) 
  (h : ∃ (k : ℝ), 3 * a^(x+7) * b^4 = k * (-a^4 * b^(2*y))) : 
  x^y = 9 := by sorry

end like_terms_imply_x_power_y_equals_nine_l1716_171658


namespace problem_solution_l1716_171684

theorem problem_solution (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 1) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 := by
  sorry

end problem_solution_l1716_171684


namespace andras_bela_numbers_l1716_171628

theorem andras_bela_numbers :
  ∀ (a b : ℕ+),
  (a = b + 1992 ∨ b = a + 1992) →
  a > 1992 →
  b > 3984 →
  a ≤ 5976 →
  (a + 1 > 5976) →
  (a = 5976 ∧ b = 7968) :=
by sorry

end andras_bela_numbers_l1716_171628


namespace mutually_expressible_implies_symmetric_zero_l1716_171668

/-- A function f is symmetric if f(x, y) = f(y, x) for all x and y. -/
def IsSymmetric (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = f y x

/-- Two variables x and y are mutually expressible if there exists a symmetric function f
    such that f(x, y) = 0 implies both y = g(x) and x = g(y) for some function g. -/
def MutuallyExpressible (x y : ℝ) : Prop :=
  ∃ (f : ℝ → ℝ → ℝ) (g : ℝ → ℝ), IsSymmetric f ∧ f x y = 0 ∧ y = g x ∧ x = g y

/-- Theorem: If two variables are mutually expressible, then there exists a symmetric function
    that equals zero for those variables. -/
theorem mutually_expressible_implies_symmetric_zero (x y : ℝ) :
  MutuallyExpressible x y → ∃ (f : ℝ → ℝ → ℝ), IsSymmetric f ∧ f x y = 0 := by
  sorry

end mutually_expressible_implies_symmetric_zero_l1716_171668


namespace triangle_property_l1716_171602

theorem triangle_property (A B C a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) ∧
  -- Sides are positive
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →
  -- Part 1
  (a * Real.cos B + Real.sqrt 3 * b * Real.sin A = c → A = π / 6) ∧
  -- Part 2
  (a = 1 ∧ b * c = 2 - Real.sqrt 3 → b + c = Real.sqrt 2) :=
by sorry

end triangle_property_l1716_171602


namespace pink_roses_count_is_300_l1716_171657

/-- Calculates the number of pink roses in Mrs. Dawson's garden -/
def pink_roses_count : ℕ :=
  let total_rows : ℕ := 30
  let roses_per_row : ℕ := 50
  let red_roses : ℕ := (2 * roses_per_row) / 5
  let blue_roses : ℕ := 1
  let remaining_after_blue : ℕ := roses_per_row - red_roses - blue_roses
  let white_roses : ℕ := remaining_after_blue / 4
  let yellow_roses : ℕ := 2
  let remaining_after_yellow : ℕ := remaining_after_blue - white_roses - yellow_roses
  let purple_roses : ℕ := (3 * remaining_after_yellow) / 8
  let orange_roses : ℕ := 3
  let pink_roses_per_row : ℕ := remaining_after_yellow - purple_roses - orange_roses
  total_rows * pink_roses_per_row

theorem pink_roses_count_is_300 : pink_roses_count = 300 := by
  sorry

end pink_roses_count_is_300_l1716_171657


namespace no_real_roots_for_geometric_sequence_quadratic_l1716_171603

/-- Given a, b, c form a geometric sequence, prove that ax^2 + bx + c has no real roots -/
theorem no_real_roots_for_geometric_sequence_quadratic 
  (a b c : ℝ) 
  (h_geometric : b^2 = a*c) 
  (h_positive : a*c > 0) : 
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 := by
sorry

end no_real_roots_for_geometric_sequence_quadratic_l1716_171603


namespace binary_101_to_decimal_l1716_171671

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 101 in base 2. -/
def binary_101 : List Bool := [true, false, true]

theorem binary_101_to_decimal :
  binary_to_decimal binary_101 = 5 := by
  sorry

end binary_101_to_decimal_l1716_171671


namespace correct_smaller_type_pages_l1716_171601

/-- Represents the number of pages in smaller type -/
def smaller_type_pages : ℕ := 17

/-- Represents the number of pages in larger type -/
def larger_type_pages : ℕ := 21 - smaller_type_pages

/-- The total number of words in the article -/
def total_words : ℕ := 48000

/-- The number of words per page in larger type -/
def words_per_page_large : ℕ := 1800

/-- The number of words per page in smaller type -/
def words_per_page_small : ℕ := 2400

/-- The total number of pages -/
def total_pages : ℕ := 21

theorem correct_smaller_type_pages : 
  smaller_type_pages = 17 ∧ 
  larger_type_pages + smaller_type_pages = total_pages ∧
  words_per_page_large * larger_type_pages + words_per_page_small * smaller_type_pages = total_words :=
by sorry

end correct_smaller_type_pages_l1716_171601


namespace soda_distribution_l1716_171618

theorem soda_distribution (total_sodas : ℕ) (sisters : ℕ) : 
  total_sodas = 12 →
  sisters = 2 →
  let brothers := 2 * sisters
  let total_siblings := sisters + brothers
  total_sodas / total_siblings = 2 := by
  sorry

end soda_distribution_l1716_171618


namespace intersection_line_circle_l1716_171662

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define a line with slope 1
def line_with_slope_1 (x y b : ℝ) : Prop := y = x + b

-- Define a point on both the circle and the line
def point_on_circle_and_line (x y b : ℝ) : Prop :=
  circle_C x y ∧ line_with_slope_1 x y b

-- Define that a circle passes through the origin
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂)^2 + (y₁ + y₂)^2 = x₁^2 + y₁^2 + x₂^2 + y₂^2

theorem intersection_line_circle :
  ∃ b : ℝ, b = 1 ∨ b = -4 ∧
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    point_on_circle_and_line x₁ y₁ b ∧
    point_on_circle_and_line x₂ y₂ b ∧
    x₁ ≠ x₂ ∧
    circle_through_origin x₁ y₁ x₂ y₂ :=
sorry

end intersection_line_circle_l1716_171662


namespace highest_throw_l1716_171623

def christine_first : ℕ := 20

def janice_first (christine_first : ℕ) : ℕ := christine_first - 4

def christine_second (christine_first : ℕ) : ℕ := christine_first + 10

def janice_second (janice_first : ℕ) : ℕ := janice_first * 2

def christine_third (christine_second : ℕ) : ℕ := christine_second + 4

def janice_third (christine_first : ℕ) : ℕ := christine_first + 17

theorem highest_throw :
  let c1 := christine_first
  let j1 := janice_first c1
  let c2 := christine_second c1
  let j2 := janice_second j1
  let c3 := christine_third c2
  let j3 := janice_third c1
  max c1 (max j1 (max c2 (max j2 (max c3 j3)))) = 37 := by sorry

end highest_throw_l1716_171623


namespace recipe_people_l1716_171609

/-- The number of people the original recipe is intended for -/
def P : ℕ := sorry

/-- The number of eggs required for the original recipe -/
def original_eggs : ℕ := 2

/-- The number of people Tyler wants to make the cake for -/
def tyler_people : ℕ := 8

/-- The number of eggs Tyler needs for his cake -/
def tyler_eggs : ℕ := 4

theorem recipe_people : P = 4 := by
  sorry

end recipe_people_l1716_171609


namespace light_bulb_cost_exceeds_budget_l1716_171639

/-- Represents the cost of light bulbs for Valerie's lamps --/
def light_bulb_cost : ℝ :=
  let small_cost : ℝ := 3 * 8.50
  let large_cost : ℝ := 1 * 14.25
  let medium_cost : ℝ := 2 * 10.75
  let extra_small_cost : ℝ := 4 * 6.25
  small_cost + large_cost + medium_cost + extra_small_cost

/-- Valerie's budget for light bulbs --/
def budget : ℝ := 80

/-- Theorem stating that the total cost of light bulbs exceeds Valerie's budget --/
theorem light_bulb_cost_exceeds_budget : light_bulb_cost > budget := by
  sorry

end light_bulb_cost_exceeds_budget_l1716_171639


namespace ratio_of_amounts_l1716_171693

theorem ratio_of_amounts (total_amount : ℕ) (r_amount : ℕ) 
  (h1 : total_amount = 8000)
  (h2 : r_amount = 3200) :
  r_amount / (total_amount - r_amount) = 2 / 3 := by
  sorry

end ratio_of_amounts_l1716_171693


namespace intersection_value_l1716_171654

theorem intersection_value (a : ℝ) : 
  let A := {x : ℝ | x^2 - 4 ≤ 0}
  let B := {x : ℝ | 2*x + a ≤ 0}
  (A ∩ B = {x : ℝ | -2 ≤ x ∧ x ≤ 1}) → a = -4 := by
sorry

end intersection_value_l1716_171654


namespace solution_difference_l1716_171624

theorem solution_difference (m : ℚ) : 
  (∃ x₁ x₂ : ℚ, 5*m + 3*x₁ = 1 + x₁ ∧ 2*x₂ + m = 3*m ∧ x₁ = x₂ + 2) ↔ m = -3/7 := by
  sorry

end solution_difference_l1716_171624


namespace cookie_theorem_l1716_171615

def cookie_problem (initial_total initial_chocolate initial_sugar initial_oatmeal : ℕ)
  (morning_chocolate morning_sugar : ℕ)
  (lunch_chocolate lunch_sugar lunch_oatmeal : ℕ)
  (afternoon_chocolate afternoon_sugar afternoon_oatmeal : ℕ)
  (damage_percent : ℚ) : Prop :=
  let total_chocolate_sold := morning_chocolate + lunch_chocolate + afternoon_chocolate
  let total_sugar_sold := morning_sugar + lunch_sugar + afternoon_sugar
  let total_oatmeal_sold := lunch_oatmeal + afternoon_oatmeal
  let remaining_chocolate := max (initial_chocolate - total_chocolate_sold) 0
  let remaining_sugar := initial_sugar - total_sugar_sold
  let remaining_oatmeal := initial_oatmeal - total_oatmeal_sold
  let total_remaining := remaining_chocolate + remaining_sugar + remaining_oatmeal
  let damaged := ⌊(damage_percent * total_remaining : ℚ)⌋
  total_remaining - damaged = 18

theorem cookie_theorem :
  cookie_problem 120 60 40 20 24 12 33 20 4 10 4 2 (1/20) := by sorry

end cookie_theorem_l1716_171615


namespace crossword_solvable_l1716_171647

-- Define the structure of a crossword puzzle
structure Crossword :=
  (grid : List (List Char))
  (vertical_clues : List String)
  (horizontal_clues : List String)

-- Define the words for the crossword
def words : List String := ["счет", "евро", "доллар", "вклад", "золото", "ломбард", "обмен", "система"]

-- Define the clues for the crossword
def vertical_clues : List String := [
  "What a bank opens for a person who wants to become its client",
  "This currency is used in Italy and other places",
  "One of the most well-known international currencies, accepted for payment in many countries",
  "The way to store and gradually increase family money in the bank",
  "A precious metal, whose reserves are accounted for by the Bank of Russia",
  "An organization from which you can borrow money and pay a small interest"
]

def horizontal_clues : List String := [
  "To pay abroad, you need to carry out ... of currency",
  "In Russia, there is a multi-level banking ...: the Central Bank of the Russian Federation, banks with a universal license, and with a basic one",
  "The place where you can take jewelry and get a loan for it"
]

-- Define the function to check if the crossword is valid
def is_valid_crossword (c : Crossword) : Prop :=
  c.vertical_clues.length = 6 ∧
  c.horizontal_clues.length = 3 ∧
  c.grid.all (λ row => row.length = 6) ∧
  c.grid.length = 7

-- Define the theorem to prove
theorem crossword_solvable :
  ∃ (c : Crossword), is_valid_crossword c ∧
    (∀ w ∈ words, w.length ≤ 7) ∧
    (∀ clue ∈ c.vertical_clues ++ c.horizontal_clues, ∃ w ∈ words, clue.length > 0 ∧ w.length > 0) :=
sorry

end crossword_solvable_l1716_171647


namespace infinite_sums_equalities_l1716_171645

/-- Given a right triangle ΔAB₀C₀ with right angle at B₀ and angle α at A,
    with perpendiculars drawn as described in the problem -/
structure RightTriangleWithPerpendiculars (α : ℝ) :=
  (A B₀ C₀ : ℝ × ℝ)
  (is_right_angle : (B₀.1 - A.1) * (C₀.1 - A.1) + (B₀.2 - A.2) * (C₀.2 - A.2) = 0)
  (angle_α : Real.cos α = (B₀.1 - A.1) / Real.sqrt ((B₀.1 - A.1)^2 + (B₀.2 - A.2)^2))
  (B : ℕ → ℝ × ℝ)
  (C : ℕ → ℝ × ℝ)

/-- The theorem stating the equalities for the infinite sums -/
theorem infinite_sums_equalities {α : ℝ} (t : RightTriangleWithPerpendiculars α) :
  (∑' i, Real.sqrt ((t.B i).1 - (t.C i).1)^2 + ((t.B i).2 - (t.C i).2)^2) = 
    Real.sqrt ((t.A.1 - t.C₀.1)^2 + (t.A.2 - t.C₀.2)^2) / Real.sin α ∧
  (∑' i, Real.sqrt ((t.A.1 - (t.B i).1)^2 + (t.A.2 - (t.B i).2)^2)) = 
    Real.sqrt ((t.A.1 - t.B₀.1)^2 + (t.A.2 - t.B₀.2)^2) / Real.sin α^2 ∧
  (∑' i, Real.sqrt ((t.A.1 - (t.C i).1)^2 + (t.A.2 - (t.C i).2)^2)) = 
    Real.sqrt ((t.A.1 - t.C₀.1)^2 + (t.A.2 - t.C₀.2)^2) / Real.sin α^2 ∧
  (∑' i, Real.sqrt (((t.C (i+1)).1 - (t.B i).1)^2 + ((t.C (i+1)).2 - (t.B i).2)^2)) = 
    Real.sqrt ((t.A.1 - t.B₀.1)^2 + (t.A.2 - t.B₀.2)^2) / Real.sin α :=
sorry

end infinite_sums_equalities_l1716_171645


namespace taxi_charge_proof_l1716_171620

/-- Calculates the total charge for a taxi trip -/
def calculate_taxi_charge (initial_fee : ℚ) (charge_per_increment : ℚ) (increment_distance : ℚ) (trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / increment_distance).floor * charge_per_increment

/-- Proves that the total charge for a 3.6-mile trip is $4.95 -/
theorem taxi_charge_proof :
  let initial_fee : ℚ := 225/100
  let charge_per_increment : ℚ := 3/10
  let increment_distance : ℚ := 2/5
  let trip_distance : ℚ := 36/10
  calculate_taxi_charge initial_fee charge_per_increment increment_distance trip_distance = 495/100 := by
  sorry

#eval calculate_taxi_charge (225/100) (3/10) (2/5) (36/10)

end taxi_charge_proof_l1716_171620


namespace perpendicular_equivalence_l1716_171640

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line : Line → Line → Prop)

-- Define the intersection of planes
variable (intersect : Plane → Plane → Line)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_equivalence 
  (α β : Plane) (m n l : Line) 
  (h1 : perp_plane α β) 
  (h2 : intersect α β = l) 
  (h3 : subset m α) 
  (h4 : subset n β) : 
  perp_line m n ↔ (perp_line m l ∨ perp_line n l) := by
  sorry

end perpendicular_equivalence_l1716_171640


namespace function_coefficient_sum_l1716_171606

theorem function_coefficient_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x - 2) = 2 * x^2 - 5 * x + 3) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 6 := by
  sorry

end function_coefficient_sum_l1716_171606


namespace girls_in_first_grade_l1716_171686

/-- Represents the first grade class configuration -/
structure FirstGrade where
  classrooms : ℕ
  boys : ℕ
  students_per_classroom : ℕ

/-- Calculates the number of girls in the first grade -/
def girls_count (fg : FirstGrade) : ℕ :=
  fg.classrooms * fg.students_per_classroom - fg.boys

/-- Theorem stating the number of girls in the first grade -/
theorem girls_in_first_grade (fg : FirstGrade) 
  (h1 : fg.classrooms = 4)
  (h2 : fg.boys = 56)
  (h3 : fg.students_per_classroom = 25)
  (h4 : ∀ c, c ≤ fg.classrooms → fg.boys / fg.classrooms = (girls_count fg) / fg.classrooms) :
  girls_count fg = 44 := by
  sorry

#eval girls_count ⟨4, 56, 25⟩

end girls_in_first_grade_l1716_171686


namespace probability_of_no_growth_pie_l1716_171681

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def pies_given : ℕ := 3

def probability_no_growth_pie : ℚ := 7/10

theorem probability_of_no_growth_pie :
  (1 - (Nat.choose (total_pies - growth_pies) (pies_given - growth_pies) : ℚ) / 
   (Nat.choose total_pies pies_given : ℚ)) = probability_no_growth_pie :=
sorry

end probability_of_no_growth_pie_l1716_171681


namespace board_intersection_area_l1716_171695

/-- The area of intersection of two rectangular boards crossing at a 45-degree angle -/
theorem board_intersection_area (width1 width2 : ℝ) (angle : ℝ) :
  width1 = 5 →
  width2 = 7 →
  angle = 45 →
  (width1 * width2 : ℝ) = 35 :=
by sorry

end board_intersection_area_l1716_171695


namespace total_songs_bought_l1716_171653

theorem total_songs_bought (country_albums : ℕ) (pop_albums : ℕ) 
  (songs_per_country_album : ℕ) (songs_per_pop_album : ℕ) : 
  country_albums = 4 → pop_albums = 7 → 
  songs_per_country_album = 5 → songs_per_pop_album = 6 → 
  country_albums * songs_per_country_album + pop_albums * songs_per_pop_album = 62 := by
  sorry

#check total_songs_bought

end total_songs_bought_l1716_171653


namespace function_value_alternation_l1716_171646

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) where a, b, α, β are non-zero real numbers,
    if f(2013) = -1, then f(2014) = 1 -/
theorem function_value_alternation (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 2013 = -1 → f 2014 = 1 := by
  sorry

end function_value_alternation_l1716_171646


namespace correct_num_technicians_l1716_171613

/-- Represents the number of technicians in the workshop -/
def num_technicians : ℕ := 7

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 21

/-- Represents the average salary of all workers in Rupees -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians in Rupees -/
def avg_salary_technicians : ℕ := 12000

/-- Represents the average salary of non-technician workers in Rupees -/
def avg_salary_rest : ℕ := 6000

/-- Theorem stating that the number of technicians is correct given the conditions -/
theorem correct_num_technicians :
  num_technicians = 7 ∧
  num_technicians ≤ total_workers ∧
  num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_rest =
    total_workers * avg_salary_all :=
by sorry

end correct_num_technicians_l1716_171613


namespace lindas_furniture_fraction_l1716_171669

/-- Given Linda's original savings and the cost of a TV, prove the fraction spent on furniture. -/
theorem lindas_furniture_fraction (original_savings : ℚ) (tv_cost : ℚ) 
  (h1 : original_savings = 600)
  (h2 : tv_cost = 300) :
  (original_savings - tv_cost) / original_savings = 1/2 := by
  sorry

end lindas_furniture_fraction_l1716_171669


namespace f_derivative_neg_one_l1716_171650

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Theorem statement
theorem f_derivative_neg_one (a b c : ℝ) : 
  f' a b 1 = 2 → f' a b (-1) = -2 := by
sorry

end f_derivative_neg_one_l1716_171650


namespace candy_seller_problem_l1716_171685

/-- The number of candies the seller had initially, given the number of clowns,
    children, candies per person, and candies left after selling. -/
def initial_candies (clowns children candies_per_person candies_left : ℕ) : ℕ :=
  (clowns + children) * candies_per_person + candies_left

/-- Theorem stating that given the specific conditions in the problem,
    the initial number of candies is 700. -/
theorem candy_seller_problem :
  initial_candies 4 30 20 20 = 700 := by
  sorry

end candy_seller_problem_l1716_171685


namespace expression_evaluation_l1716_171666

theorem expression_evaluation :
  let a : ℚ := -1/2
  (a + 3)^2 + (a + 3)*(a - 3) - 2*a*(3 - a) = 1 := by
sorry

end expression_evaluation_l1716_171666


namespace expression_value_l1716_171698

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (3 * x^3 + 4 * y^3) / 9 = 37 + 4/9 := by
  sorry

end expression_value_l1716_171698


namespace money_ratio_problem_l1716_171634

theorem money_ratio_problem (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  ram = 588 →
  krishan = 3468 →
  (gopal : ℚ) / krishan = 100 / 243 := by
sorry

end money_ratio_problem_l1716_171634


namespace exist_same_color_perfect_square_diff_l1716_171670

/-- A coloring of integers using three colors. -/
def Coloring := ℤ → Fin 3

/-- Predicate to check if a number is a perfect square. -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

/-- Main theorem: For any coloring of integers using three colors,
    there exist two different integers of the same color
    whose difference is a perfect square. -/
theorem exist_same_color_perfect_square_diff (c : Coloring) :
  ∃ a b : ℤ, a ≠ b ∧ c a = c b ∧ is_perfect_square (a - b) := by
  sorry


end exist_same_color_perfect_square_diff_l1716_171670


namespace greg_earnings_l1716_171691

/-- Calculates the earnings for dog walking based on the given parameters -/
def dog_walking_earnings (base_charge : ℕ) (per_minute_charge : ℕ) 
  (dogs : List (ℕ × ℕ)) : ℕ :=
  dogs.foldl (fun acc (num_dogs, minutes) => 
    acc + num_dogs * base_charge + num_dogs * minutes * per_minute_charge) 0

/-- Theorem stating Greg's earnings from his dog walking business -/
theorem greg_earnings : 
  dog_walking_earnings 20 1 [(1, 10), (2, 7), (3, 9)] = 171 := by
  sorry

#eval dog_walking_earnings 20 1 [(1, 10), (2, 7), (3, 9)]

end greg_earnings_l1716_171691


namespace square_of_85_l1716_171622

theorem square_of_85 : (85 : ℕ) ^ 2 = 7225 := by
  sorry

end square_of_85_l1716_171622


namespace unique_solution_for_g_l1716_171688

/-- Given functions f and g where g(x) = 4f⁻¹(x) and f(x) = 30 / (x + 4),
    prove that the unique value of x satisfying g(x) = 20 is 10/3 -/
theorem unique_solution_for_g (f g : ℝ → ℝ) 
    (h1 : ∀ x, g x = 4 * (f⁻¹ x)) 
    (h2 : ∀ x, f x = 30 / (x + 4)) : 
    ∃! x, g x = 20 ∧ x = 10/3 := by
  sorry

end unique_solution_for_g_l1716_171688


namespace distribute_six_balls_three_boxes_l1716_171652

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 67 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute 6 3 = 67 := by sorry

end distribute_six_balls_three_boxes_l1716_171652


namespace sqrt_non_square_irrational_l1716_171643

theorem sqrt_non_square_irrational (a : ℤ) 
  (h : ∀ n : ℤ, n^2 ≠ a) : 
  Irrational (Real.sqrt (a : ℝ)) := by
  sorry

end sqrt_non_square_irrational_l1716_171643


namespace solve_candy_problem_l1716_171626

def candy_problem (megan_candy : ℕ) (mary_multiplier : ℕ) (mary_additional : ℕ) : Prop :=
  let mary_initial := mary_multiplier * megan_candy
  let mary_total := mary_initial + mary_additional
  megan_candy = 5 ∧ mary_multiplier = 3 ∧ mary_additional = 10 → mary_total = 25

theorem solve_candy_problem :
  candy_problem 5 3 10 := by
  sorry

end solve_candy_problem_l1716_171626


namespace system_solution_unique_l1716_171648

theorem system_solution_unique : 
  ∃! (x y : ℝ), (x + 3 * y = 1) ∧ (3 * x + y = -5) :=
by
  -- The proof would go here
  sorry

end system_solution_unique_l1716_171648


namespace intersecting_circles_radius_l1716_171642

/-- Two circles on a plane where each passes through the center of the other -/
structure IntersectingCircles where
  O₁ : ℝ × ℝ  -- Center of first circle
  O₂ : ℝ × ℝ  -- Center of second circle
  A : ℝ × ℝ   -- First intersection point
  B : ℝ × ℝ   -- Second intersection point
  radius : ℝ  -- Common radius of both circles
  passes_through_center : dist O₁ O₂ = radius
  on_circle : dist O₁ A = radius ∧ dist O₂ A = radius ∧ dist O₁ B = radius ∧ dist O₂ B = radius

/-- The area of a quadrilateral given its four vertices -/
def quadrilateralArea (p₁ p₂ p₃ p₄ : ℝ × ℝ) : ℝ := sorry

theorem intersecting_circles_radius 
  (circles : IntersectingCircles) 
  (area_condition : quadrilateralArea circles.O₁ circles.A circles.O₂ circles.B = 2 * Real.sqrt 3) :
  circles.radius = 2 := by sorry

end intersecting_circles_radius_l1716_171642


namespace iron_cotton_mass_equality_l1716_171675

-- Define the conversion factor from kilograms to grams
def kgToGrams : ℝ → ℝ := (· * 1000)

-- Define the masses in their given units
def ironMassKg : ℝ := 5
def cottonMassG : ℝ := 5000

-- Theorem stating that the masses are equal
theorem iron_cotton_mass_equality :
  kgToGrams ironMassKg = cottonMassG := by sorry

end iron_cotton_mass_equality_l1716_171675


namespace binary_multiplication_theorem_l1716_171677

/-- Convert a list of binary digits to a natural number -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Convert a natural number to a list of binary digits -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Multiply two binary numbers represented as lists of booleans -/
def binary_multiply (a b : List Bool) : List Bool :=
  nat_to_binary ((binary_to_nat a) * (binary_to_nat b))

theorem binary_multiplication_theorem :
  let a := [true, false, true, true, false, true, true]  -- 1101101₂
  let b := [true, true, true]  -- 111₂
  let product := binary_multiply a b
  binary_to_nat product = 1267 ∧ 
  product = [true, true, false, false, true, true, true, true, false, false, true] :=
by sorry

end binary_multiplication_theorem_l1716_171677


namespace seventh_root_of_unity_sum_l1716_171679

theorem seventh_root_of_unity_sum (z : ℂ) :
  z ^ 7 = 1 ∧ z ≠ 1 →
  z + z^2 + z^4 = (-1 + Complex.I * Real.sqrt 11) / 2 ∨
  z + z^2 + z^4 = (-1 - Complex.I * Real.sqrt 11) / 2 := by
  sorry

end seventh_root_of_unity_sum_l1716_171679


namespace complement_of_A_in_U_l1716_171697

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {2, 4} := by
  sorry

end complement_of_A_in_U_l1716_171697


namespace cubic_sum_over_product_square_l1716_171667

theorem cubic_sum_over_product_square (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (sum_zero : x + y + z = 0) (sum_prod_nonzero : x*y + x*z + y*z ≠ 0) :
  (x^3 + y^3 + z^3) / (x*y*z * (x*y + x*z + y*z)^2) = 3 / (x^2 + x*y + y^2)^2 := by
  sorry

end cubic_sum_over_product_square_l1716_171667


namespace negation_of_x_squared_plus_two_gt_zero_is_false_l1716_171699

theorem negation_of_x_squared_plus_two_gt_zero_is_false :
  ¬(∃ x : ℝ, x^2 + 2 ≤ 0) :=
by sorry

end negation_of_x_squared_plus_two_gt_zero_is_false_l1716_171699


namespace geometric_sequence_third_term_l1716_171632

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (p : ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_roots : a 1 + a 5 = p ∧ a 1 * a 5 = 4)
  (h_p_neg : p < 0) :
  a 3 = -2 :=
sorry

end geometric_sequence_third_term_l1716_171632


namespace population_scientific_notation_l1716_171696

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem population_scientific_notation :
  let population : ℝ := 1412.60 * 1000000
  let scientific_form := toScientificNotation population
  scientific_form.coefficient = 1.4126 ∧ scientific_form.exponent = 5 := by
  sorry

end population_scientific_notation_l1716_171696


namespace cos_2alpha_value_l1716_171604

theorem cos_2alpha_value (α : Real) (h : Real.tan (α - π/4) = -1/3) :
  Real.cos (2 * α) = 3/5 := by
  sorry

end cos_2alpha_value_l1716_171604


namespace rectangle_area_l1716_171694

theorem rectangle_area (perimeter : ℝ) (length width : ℝ) (h1 : perimeter = 280) 
  (h2 : length / width = 5 / 2) (h3 : perimeter = 2 * (length + width)) 
  (h4 : width * Real.sqrt 2 = length / 2) : length * width = 4000 := by
  sorry

end rectangle_area_l1716_171694


namespace circle_chord_intersection_l1716_171614

theorem circle_chord_intersection (r : ℝ) (chord_length : ℝ) : 
  r = 5 →
  chord_length = 8 →
  ∃ (ak kb : ℝ),
    ak + kb = 2 * r ∧
    ak * kb = r^2 - (chord_length / 2)^2 ∧
    ak = 1.25 ∧
    kb = 8.75 := by
sorry

end circle_chord_intersection_l1716_171614


namespace quadratic_transformation_l1716_171635

theorem quadratic_transformation (x : ℝ) : x^2 - 8*x - 9 = (x - 4)^2 - 25 := by
  sorry

end quadratic_transformation_l1716_171635


namespace six_digit_divisible_by_1001_l1716_171692

-- Define a three-digit number
def three_digit_number (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

-- Define the six-digit number formed by repeating the three-digit number
def six_digit_number (a b c : Nat) : Nat :=
  1000 * (three_digit_number a b c) + (three_digit_number a b c)

-- Theorem statement
theorem six_digit_divisible_by_1001 (a b c : Nat) :
  (a < 10) → (b < 10) → (c < 10) →
  (six_digit_number a b c) % 1001 = 0 := by
  sorry

end six_digit_divisible_by_1001_l1716_171692


namespace system_solution_l1716_171665

theorem system_solution : 
  ∃ (x y : ℝ), (6 / (x^2 + y^2) + x^2 * y^2 = 10) ∧ 
               (x^4 + y^4 + 7 * x^2 * y^2 = 81) ∧
               ((x = Real.sqrt 3 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨
                (x = -Real.sqrt 3 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3))) :=
by sorry

end system_solution_l1716_171665


namespace rectangle_ratio_l1716_171689

/-- Proves that a rectangle with area 100 m² and length 20 m has a length-to-width ratio of 4:1 -/
theorem rectangle_ratio (area : ℝ) (length : ℝ) (width : ℝ) 
  (h_area : area = 100) 
  (h_length : length = 20) 
  (h_rect : area = length * width) : 
  length / width = 4 := by
sorry

end rectangle_ratio_l1716_171689
