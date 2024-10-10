import Mathlib

namespace hexagon_side_length_l220_22022

theorem hexagon_side_length (perimeter : ℝ) (h : perimeter = 30) : 
  perimeter / 6 = 5 := by sorry

end hexagon_side_length_l220_22022


namespace midpoint_coordinate_sum_l220_22038

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (1, 4) and (7, 10) is 11. -/
theorem midpoint_coordinate_sum : 
  let x1 : ℝ := 1
  let y1 : ℝ := 4
  let x2 : ℝ := 7
  let y2 : ℝ := 10
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 11 := by
  sorry

end midpoint_coordinate_sum_l220_22038


namespace shopkeeper_loss_percent_l220_22095

theorem shopkeeper_loss_percent 
  (initial_value : ℝ) 
  (profit_rate : ℝ) 
  (theft_rate : ℝ) 
  (profit_rate_is_10_percent : profit_rate = 0.1)
  (theft_rate_is_30_percent : theft_rate = 0.3)
  (initial_value_positive : initial_value > 0) :
  let remaining_value := initial_value * (1 - theft_rate)
  let selling_price := remaining_value * (1 + profit_rate)
  let loss := initial_value - selling_price
  let loss_percent := (loss / initial_value) * 100
  loss_percent = 23 := by
sorry

end shopkeeper_loss_percent_l220_22095


namespace some_number_value_l220_22069

theorem some_number_value (x : ℝ) : 60 + 5 * 12 / (x / 3) = 61 → x = 180 := by
  sorry

end some_number_value_l220_22069


namespace binomial_60_3_l220_22083

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l220_22083


namespace tangent_and_decreasing_interval_l220_22075

-- Define the function f
def f (m n : ℝ) (x : ℝ) : ℝ := m * x^3 + n * x^2

-- Define the derivative of f
def f_derivative (m n : ℝ) (x : ℝ) : ℝ := 3 * m * x^2 + 2 * n * x

-- Theorem statement
theorem tangent_and_decreasing_interval 
  (m n : ℝ) 
  (h1 : f m n (-1) = 2)
  (h2 : f_derivative m n (-1) = -3)
  (h3 : ∀ t : ℝ, ∀ x ∈ Set.Icc t (t + 1), 
        f_derivative m n x ≤ 0 → 
        -2 ≤ t ∧ t ≤ -1) :
  ∀ t : ℝ, (∀ x ∈ Set.Icc t (t + 1), f_derivative m n x ≤ 0) → 
    t ∈ Set.Icc (-2) (-1) :=
sorry

end tangent_and_decreasing_interval_l220_22075


namespace certain_number_proof_l220_22023

theorem certain_number_proof (x : ℤ) : x + 34 - 53 = 28 ↔ x = 47 := by
  sorry

end certain_number_proof_l220_22023


namespace imaginary_part_of_z_l220_22008

theorem imaginary_part_of_z (z : ℂ) : (1 + z) * (1 - Complex.I) = 2 → Complex.im z = 1 := by
  sorry

end imaginary_part_of_z_l220_22008


namespace cistern_fill_time_l220_22020

/-- Time to fill cistern with all pipes open simultaneously -/
theorem cistern_fill_time (fill_time_A fill_time_B empty_time_C : ℝ) 
  (h_A : fill_time_A = 45)
  (h_B : fill_time_B = 60)
  (h_C : empty_time_C = 72) : 
  (1 / ((1 / fill_time_A) + (1 / fill_time_B) - (1 / empty_time_C))) = 40 := by
  sorry

end cistern_fill_time_l220_22020


namespace sets_A_B_characterization_l220_22034

theorem sets_A_B_characterization (A B : Set ℤ) :
  (A ∪ B = Set.univ) ∧
  (∀ x, x ∈ A → x - 1 ∈ B) ∧
  (∀ x y, x ∈ B ∧ y ∈ B → x + y ∈ A) →
  ((A = {x | ∃ k, x = 2 * k} ∧ B = {x | ∃ k, x = 2 * k + 1}) ∨
   (A = Set.univ ∧ B = Set.univ)) :=
by sorry

end sets_A_B_characterization_l220_22034


namespace intersection_of_A_and_B_l220_22014

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l220_22014


namespace j_range_l220_22035

def h (x : ℝ) : ℝ := 2 * x + 1

def j (x : ℝ) : ℝ := h (h (h (h (h x))))

theorem j_range :
  ∀ y ∈ Set.range j,
  -1 ≤ y ∧ y ≤ 127 ∧
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ j x = y :=
by sorry

end j_range_l220_22035


namespace sum_of_squares_over_factorial_l220_22074

theorem sum_of_squares_over_factorial : (1^2 + 2^2 + 3^2 + 4^2) / (1 * 2 * 3) = 5 := by
  sorry

end sum_of_squares_over_factorial_l220_22074


namespace derivative_implies_function_l220_22011

open Real

theorem derivative_implies_function (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, deriv f x = 1 + cos x) →
  ∃ C, ∀ x, f x = x + sin x + C :=
by
  sorry

end derivative_implies_function_l220_22011


namespace leak_drops_per_minute_l220_22045

/-- Proves that the leak drips 3 drops per minute given the conditions -/
theorem leak_drops_per_minute 
  (drop_volume : ℝ) 
  (pot_capacity : ℝ) 
  (fill_time : ℝ) 
  (h1 : drop_volume = 20) 
  (h2 : pot_capacity = 3000) 
  (h3 : fill_time = 50) : 
  (pot_capacity / drop_volume) / fill_time = 3 := by
  sorry

#check leak_drops_per_minute

end leak_drops_per_minute_l220_22045


namespace factorization_of_4m_squared_minus_64_l220_22018

theorem factorization_of_4m_squared_minus_64 (m : ℝ) : 4 * m^2 - 64 = 4 * (m + 4) * (m - 4) := by
  sorry

end factorization_of_4m_squared_minus_64_l220_22018


namespace magnitude_comparison_l220_22043

theorem magnitude_comparison : 7^(0.3 : ℝ) > (0.3 : ℝ)^7 ∧ (0.3 : ℝ)^7 > Real.log 0.3 := by
  sorry

end magnitude_comparison_l220_22043


namespace ellipse_eccentricity_l220_22077

theorem ellipse_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2/m + y^2/4 = 1) →  -- Equation of the ellipse
  (∃ a b c : ℝ, a > b ∧ b > 0 ∧ c^2 = a^2 - b^2 ∧ 2*c = 2) →  -- Eccentricity is 2
  (m = 3 ∨ m = 5) :=
by sorry

end ellipse_eccentricity_l220_22077


namespace quadratic_monotonicity_l220_22064

theorem quadratic_monotonicity (a b c : ℝ) (h_a : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y) →
  (f 1 < f 5) ∧
  ¬ ((f 1 < f 5) → (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y)) :=
by sorry

end quadratic_monotonicity_l220_22064


namespace mens_wages_l220_22007

/-- Proves that the wage of one man is 24 Rs given the problem conditions -/
theorem mens_wages (men women boys : ℕ) (total_earnings : ℚ) : 
  men = 5 → 
  boys = 8 → 
  total_earnings = 120 → 
  ∃ (w : ℕ), (5 : ℚ) * (total_earnings / (men + w + boys)) = 24 := by
  sorry

end mens_wages_l220_22007


namespace linear_function_value_l220_22039

/-- A linear function in three variables -/
def LinearFunction (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c x y z : ℝ, f (a + x) (b + y) (c + z) = f a b c + f x y z

theorem linear_function_value (f : ℝ → ℝ → ℝ → ℝ) 
  (h_linear : LinearFunction f)
  (h_value_3 : f 3 3 3 = 1 / (3 * 3 * 3))
  (h_value_4 : f 4 4 4 = 1 / (4 * 4 * 4)) :
  f 5 5 5 = 1 / 216 := by
  sorry

end linear_function_value_l220_22039


namespace triangle_property_l220_22054

open Real

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.c / (t.b - t.a) = (sin t.A + sin t.B) / (sin t.A + sin t.C))
  (h2 : t.b = 2 * sqrt 2)
  (h3 : t.a + t.c = 3) :
  t.B = 2 * π / 3 ∧ 
  (1/2) * t.a * t.c * sin t.B = sqrt 3 / 4 := by
  sorry


end triangle_property_l220_22054


namespace smallest_integer_greater_than_sqrt_three_l220_22052

theorem smallest_integer_greater_than_sqrt_three : 
  ∀ n : ℤ, n > Real.sqrt 3 → n ≥ 2 :=
by sorry

end smallest_integer_greater_than_sqrt_three_l220_22052


namespace dart_board_probability_l220_22006

/-- The probability of a dart landing within the center square of a regular hexagon dart board -/
theorem dart_board_probability (x : ℝ) (x_pos : x > 0) : 
  let hexagon_area := 3 * Real.sqrt 3 * x^2 / 2
  let square_area := 3 * x^2 / 4
  square_area / hexagon_area = 1 / (2 * Real.sqrt 3) := by
  sorry

end dart_board_probability_l220_22006


namespace collin_savings_l220_22092

/-- Represents the number of cans Collin collected from various sources --/
structure CanCollection where
  home : ℕ
  grandparents : ℕ
  neighbor : ℕ
  dad : ℕ

/-- Calculates the total number of cans collected --/
def total_cans (c : CanCollection) : ℕ :=
  c.home + c.grandparents + c.neighbor + c.dad

/-- Represents the recycling scenario for Collin --/
structure RecyclingScenario where
  collection : CanCollection
  price_per_can : ℚ
  savings_ratio : ℚ

/-- Calculates the amount Collin will put into savings --/
def savings_amount (s : RecyclingScenario) : ℚ :=
  s.savings_ratio * s.price_per_can * (total_cans s.collection)

/-- Theorem stating that Collin will put $43.00 into savings --/
theorem collin_savings (s : RecyclingScenario) 
  (h1 : s.collection.home = 12)
  (h2 : s.collection.grandparents = 3 * s.collection.home)
  (h3 : s.collection.neighbor = 46)
  (h4 : s.collection.dad = 250)
  (h5 : s.price_per_can = 1/4)
  (h6 : s.savings_ratio = 1/2) :
  savings_amount s = 43 := by
  sorry

end collin_savings_l220_22092


namespace correct_factorization_l220_22028

theorem correct_factorization (x : ℝ) : 10 * x^2 - 5 * x = 5 * x * (2 * x - 1) := by
  sorry

#check correct_factorization

end correct_factorization_l220_22028


namespace triangle_max_perimeter_l220_22009

theorem triangle_max_perimeter (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  A = Real.pi / 3 →
  0 < B →
  B < 2 * Real.pi / 3 →
  0 < C →
  C < 2 * Real.pi / 3 →
  A + B + C = Real.pi →
  b = 2 * Real.sin B →
  c = 2 * Real.sin C →
  a + b + c ≤ 3 * Real.sqrt 3 :=
sorry

end triangle_max_perimeter_l220_22009


namespace triangle_properties_l220_22093

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  b = 2 * Real.sqrt 3 →
  (Real.cos B) / (Real.cos C) = b / (2 * a - c) →
  1 / (Real.tan A) + 1 / (Real.tan B) = (Real.sin C) / (Real.sqrt 3 * Real.sin A * Real.cos B) →
  4 * Real.sqrt 3 * S + 3 * (b^2 - a^2) = 3 * c^2 →
  S = Real.sqrt 3 / 3 ∧
  (0 < A ∧ A < Real.pi / 2 ∧ 0 < B ∧ B < Real.pi / 2 ∧ 0 < C ∧ C < Real.pi / 2 →
    (Real.sqrt 3 + 1) / 2 < (b + c) / a ∧ (b + c) / a < Real.sqrt 3 + 2) :=
by sorry

end triangle_properties_l220_22093


namespace composite_product_properties_l220_22085

def first_five_composites : List Nat := [4, 6, 8, 9, 10]

def product_of_composites : Nat := first_five_composites.prod

theorem composite_product_properties :
  (product_of_composites % 10 = 0) ∧
  (Nat.digits 10 product_of_composites).sum = 18 := by
  sorry

end composite_product_properties_l220_22085


namespace fraction_problem_l220_22055

theorem fraction_problem (x : ℚ) : 
  (3/4 : ℚ) * x * (2/5 : ℚ) * 5100 = 765.0000000000001 → x = 1/2 := by
  sorry

end fraction_problem_l220_22055


namespace least_addition_for_divisibility_l220_22097

theorem least_addition_for_divisibility :
  ∃ (n : ℕ), n = 8 ∧
  (∀ (m : ℕ), m < n → ¬((821562 + m) % 5 = 0 ∧ (821562 + m) % 13 = 0)) ∧
  (821562 + n) % 5 = 0 ∧ (821562 + n) % 13 = 0 := by
  sorry

end least_addition_for_divisibility_l220_22097


namespace work_completion_time_l220_22044

-- Define the efficiency of worker B
def B_efficiency : ℚ := 1 / 24

-- Define the efficiency of worker A (twice as efficient as B)
def A_efficiency : ℚ := 2 * B_efficiency

-- Define the combined efficiency of A and B
def combined_efficiency : ℚ := A_efficiency + B_efficiency

-- Theorem: A and B together can complete the work in 8 days
theorem work_completion_time : (1 : ℚ) / combined_efficiency = 8 := by sorry

end work_completion_time_l220_22044


namespace problem_statement_l220_22082

theorem problem_statement (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b < 0) 
  (hab : abs a < abs b) 
  (hbc : abs b < abs c) : 
  (abs (a * b) < abs (b * c)) ∧ 
  (a * c < abs (b * c)) ∧ 
  (abs (a + b) < abs (b + c)) := by
sorry

end problem_statement_l220_22082


namespace inequality_solution_set_l220_22027

theorem inequality_solution_set (x : ℝ) :
  (x + 5) / (x^2 + 3*x + 9) ≥ 0 ↔ x ≥ -5 := by
  sorry

end inequality_solution_set_l220_22027


namespace complex_magnitude_equality_l220_22013

theorem complex_magnitude_equality : ∃ t : ℝ, t > 0 ∧ Complex.abs (Complex.mk (-3) t) = 5 * Real.sqrt 5 :=
by
  use 2 * Real.sqrt 29
  sorry

end complex_magnitude_equality_l220_22013


namespace cubic_equation_solutions_l220_22021

theorem cubic_equation_solutions :
  ∀ (z : ℂ), z^3 = -27 ↔ z = -3 ∨ z = (3 / 2 : ℂ) + (3 / 2 : ℂ) * Complex.I * Real.sqrt 3 ∨ z = (3 / 2 : ℂ) - (3 / 2 : ℂ) * Complex.I * Real.sqrt 3 :=
by sorry

end cubic_equation_solutions_l220_22021


namespace min_sum_squares_l220_22070

theorem min_sum_squares (a b c : ℕ+) (h : a.val^2 + b.val^2 - c.val = 2022) :
  (∀ a' b' c' : ℕ+, a'.val^2 + b'.val^2 - c'.val = 2022 →
    a.val^2 + b.val^2 + c.val^2 ≤ a'.val^2 + b'.val^2 + c'.val^2) ∧
  a.val^2 + b.val^2 + c.val^2 = 2034 ∧
  a.val = 27 ∧ b.val = 36 ∧ c.val = 3 := by
sorry

end min_sum_squares_l220_22070


namespace factorization_x2_4xy_4y2_l220_22067

/-- Factorization of a polynomial x^2 - 4xy + 4y^2 --/
theorem factorization_x2_4xy_4y2 (x y : ℝ) :
  x^2 - 4*x*y + 4*y^2 = (x - 2*y)^2 := by sorry

end factorization_x2_4xy_4y2_l220_22067


namespace part_one_part_two_l220_22078

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 < 0}

-- Define the specific set A as given in the problem
def A_specific : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- Theorem for part (1)
theorem part_one (a b : ℝ) (h : A a b = A_specific) : a + b = -7 := by
  sorry

-- Theorem for part (2)
theorem part_two (m : ℝ) (h : A (-3) (-4) = A_specific) :
  (∀ x, x ∈ A (-3) (-4) → x ∉ B m) → m ≤ -3 ∨ m ≥ 6 := by
  sorry

end part_one_part_two_l220_22078


namespace b_investment_value_l220_22076

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- Theorem stating that given the conditions, B's investment is 32,000 --/
theorem b_investment_value (p : Partnership) 
  (h1 : p.a_investment = 24000)
  (h2 : p.c_investment = 36000)
  (h3 : p.total_profit = 92000)
  (h4 : p.c_profit_share = 36000)
  (h5 : p.c_investment / p.c_profit_share = (p.a_investment + p.b_investment + p.c_investment) / p.total_profit) : 
  p.b_investment = 32000 := by
  sorry

#check b_investment_value

end b_investment_value_l220_22076


namespace probability_other_note_counterfeit_l220_22063

/-- Represents the total number of banknotes -/
def total_notes : ℕ := 20

/-- Represents the number of counterfeit notes -/
def counterfeit_notes : ℕ := 5

/-- Represents the number of genuine notes -/
def genuine_notes : ℕ := total_notes - counterfeit_notes

/-- Calculates the probability that both drawn notes are counterfeit -/
def prob_both_counterfeit : ℚ :=
  (counterfeit_notes.choose 2 : ℚ) / (total_notes.choose 2 : ℚ)

/-- Calculates the probability that at least one drawn note is counterfeit -/
def prob_at_least_one_counterfeit : ℚ :=
  ((counterfeit_notes.choose 2 + counterfeit_notes * genuine_notes) : ℚ) / (total_notes.choose 2 : ℚ)

/-- The main theorem to be proved -/
theorem probability_other_note_counterfeit :
  prob_both_counterfeit / prob_at_least_one_counterfeit = 2 / 17 := by
  sorry

end probability_other_note_counterfeit_l220_22063


namespace number_equation_solution_l220_22080

theorem number_equation_solution : ∃ x : ℝ, 35 + 3 * x = 50 ∧ x = 5 := by
  sorry

end number_equation_solution_l220_22080


namespace justin_jersey_problem_l220_22002

/-- Represents the problem of determining the number of long-sleeved jerseys Justin bought. -/
theorem justin_jersey_problem (long_sleeve_cost stripe_cost total_cost : ℕ) 
                               (stripe_count : ℕ) (total_spent : ℕ) : 
  long_sleeve_cost = 15 →
  stripe_cost = 10 →
  stripe_count = 2 →
  total_spent = 80 →
  ∃ long_sleeve_count : ℕ, 
    long_sleeve_count * long_sleeve_cost + stripe_count * stripe_cost = total_spent ∧
    long_sleeve_count = 4 :=
by sorry

end justin_jersey_problem_l220_22002


namespace time_per_bone_l220_22062

/-- Proves that analyzing 206 bones in 206 hours with equal time per bone results in 1 hour per bone -/
theorem time_per_bone (total_time : ℕ) (num_bones : ℕ) (time_per_bone : ℚ) :
  total_time = 206 →
  num_bones = 206 →
  time_per_bone = total_time / num_bones →
  time_per_bone = 1 := by
  sorry

#check time_per_bone

end time_per_bone_l220_22062


namespace even_perfect_square_divisible_by_eight_l220_22098

theorem even_perfect_square_divisible_by_eight (b n : ℕ) : 
  b > 0 → 
  Even b → 
  n > 1 → 
  ∃ k : ℕ, (b^n - 1) / (b - 1) = k^2 → 
  8 ∣ b :=
sorry

end even_perfect_square_divisible_by_eight_l220_22098


namespace positive_solution_equation_l220_22030

theorem positive_solution_equation : ∃ x : ℝ, 
  x > 0 ∧ 
  x = 21 + Real.sqrt 449 ∧ 
  (1 / 2) * (4 * x^2 - 2) = (x^2 - 40*x - 8) * (x^2 + 20*x + 4) := by
  sorry

end positive_solution_equation_l220_22030


namespace range_of_m_for_p_or_q_l220_22015

-- Define the propositions p and q as functions of m
def proposition_p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def proposition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m+2)*x + 1 ≠ 0

-- State the theorem
theorem range_of_m_for_p_or_q :
  ∀ m : ℝ, (proposition_p m ∨ proposition_q m) ↔ m < -1 :=
sorry

end range_of_m_for_p_or_q_l220_22015


namespace roots_sum_powers_l220_22001

theorem roots_sum_powers (p q : ℝ) : 
  p^2 - 6*p + 10 = 0 → q^2 - 6*q + 10 = 0 → p^3 + p^4*q^2 + p^2*q^4 + p*q^3 + p^5*q^3 = 38676 := by
  sorry

end roots_sum_powers_l220_22001


namespace factorial_inequality_l220_22042

theorem factorial_inequality (n : ℕ) (h : n ≥ 2) :
  2 * Real.log (Nat.factorial n) > (n^2 - 2*n + 1) / n := by
  sorry

end factorial_inequality_l220_22042


namespace additional_concession_percentage_l220_22048

def original_price : ℝ := 2000
def standard_concession : ℝ := 30
def final_price : ℝ := 1120

theorem additional_concession_percentage :
  ∃ (additional_concession : ℝ),
    (original_price * (1 - standard_concession / 100) * (1 - additional_concession / 100) = final_price) ∧
    additional_concession = 20 := by
  sorry

end additional_concession_percentage_l220_22048


namespace solution_bijection_l220_22088

def equation_x (x : Fin 10 → ℕ+) : Prop :=
  (x 0) + 2^3 * (x 1) + 3^3 * (x 2) + 4^3 * (x 3) + 5^3 * (x 4) + 
  6^3 * (x 5) + 7^3 * (x 6) + 8^3 * (x 7) + 9^3 * (x 8) + 10^3 * (x 9) = 3025

def equation_y (y : Fin 10 → ℕ) : Prop :=
  (y 0) + 2^3 * (y 1) + 3^3 * (y 2) + 4^3 * (y 3) + 5^3 * (y 4) + 
  6^3 * (y 5) + 7^3 * (y 6) + 8^3 * (y 7) + 9^3 * (y 8) + 10^3 * (y 9) = 0

theorem solution_bijection :
  ∃ (f : {x : Fin 10 → ℕ+ // equation_x x} → {y : Fin 10 → ℕ // equation_y y}),
    Function.Bijective f ∧
    f ⟨λ _ => 1, sorry⟩ = ⟨λ _ => 0, sorry⟩ :=
sorry

end solution_bijection_l220_22088


namespace max_rectangle_area_l220_22089

/-- The maximum area of a rectangle with integer dimensions and perimeter 34 cm is 72 square cm. -/
theorem max_rectangle_area : ∀ l w : ℕ, 
  2 * l + 2 * w = 34 → 
  l * w ≤ 72 :=
by
  sorry

end max_rectangle_area_l220_22089


namespace shadow_length_proportion_l220_22041

/-- Represents a pot with its height and shadow length -/
structure Pot where
  height : ℝ
  shadowLength : ℝ

/-- Theorem stating the relationship between pot heights and shadow lengths -/
theorem shadow_length_proportion (pot1 pot2 : Pot)
  (h1 : pot1.height = 20)
  (h2 : pot1.shadowLength = 10)
  (h3 : pot2.height = 40)
  (h4 : pot2.shadowLength = 20)
  (h5 : pot2.height = 2 * pot1.height)
  (h6 : pot2.shadowLength = 2 * pot1.shadowLength) :
  pot1.shadowLength = pot2.shadowLength / 2 := by
  sorry

end shadow_length_proportion_l220_22041


namespace set_equality_implies_sum_of_powers_l220_22000

theorem set_equality_implies_sum_of_powers (a b : ℝ) :
  let A : Set ℝ := {a, a^2, a*b}
  let B : Set ℝ := {1, a, b}
  A = B → a^2004 + b^2004 = 1 := by
sorry

end set_equality_implies_sum_of_powers_l220_22000


namespace f_composition_half_l220_22099

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_half : f (f (1/2)) = 1/2 := by
  sorry

end f_composition_half_l220_22099


namespace estate_value_l220_22068

-- Define the estate and its components
def estate : ℝ := sorry
def older_child_share : ℝ := sorry
def younger_child_share : ℝ := sorry
def wife_share : ℝ := sorry
def charity_share : ℝ := 800

-- Define the conditions
axiom children_share : older_child_share + younger_child_share = 0.6 * estate
axiom children_ratio : older_child_share = (3/2) * younger_child_share
axiom wife_share_relation : wife_share = 4 * older_child_share
axiom total_distribution : estate = older_child_share + younger_child_share + wife_share + charity_share

-- Theorem to prove
theorem estate_value : estate = 1923 := by sorry

end estate_value_l220_22068


namespace range_of_t_t_value_for_diameter_6_l220_22061

-- Define the equation of the circle
def circle_equation (x y t : ℝ) : Prop :=
  x^2 + y^2 + (Real.sqrt 3 * t + 1) * x + t * y + t^2 - 2 = 0

-- Theorem for the range of t
theorem range_of_t (t : ℝ) :
  (∃ x y, circle_equation x y t) → t > -3 * Real.sqrt 3 / 2 :=
sorry

-- Theorem for the value of t when diameter is 6
theorem t_value_for_diameter_6 (t : ℝ) :
  (∃ x y, circle_equation x y t) →
  (∃ x₁ y₁ x₂ y₂, circle_equation x₁ y₁ t ∧ circle_equation x₂ y₂ t ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6) →
  t = 9 * Real.sqrt 3 / 2 :=
sorry

end range_of_t_t_value_for_diameter_6_l220_22061


namespace max_nSn_l220_22053

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  sum : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The problem statement -/
theorem max_nSn (seq : ArithmeticSequence) 
  (h1 : seq.sum 6 = 26)
  (h2 : seq.a 7 = 2) :
  ∃ m : ℚ, m = 338 ∧ ∀ n : ℕ, n * seq.sum n ≤ m :=
sorry

end max_nSn_l220_22053


namespace cubic_function_properties_l220_22087

/-- A cubic function with a maximum at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2

/-- The derivative of f -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem cubic_function_properties (a b : ℝ) :
  f a b 1 = 3 ∧ f_deriv a b 1 = 0 →
  a = -6 ∧ b = 9 ∧ ∀ x, f a b x ≥ 0 ∧ (∃ x₀, f a b x₀ = 0) := by
  sorry

end cubic_function_properties_l220_22087


namespace circle_circumference_l220_22059

/-- Given a circle with area 1800 cm² and ratio of area to circumference 15, 
    prove that its circumference is 120 cm. -/
theorem circle_circumference (A : ℝ) (r : ℝ) :
  A = 1800 →
  A / (2 * Real.pi * r) = 15 →
  2 * Real.pi * r = 120 :=
by sorry

end circle_circumference_l220_22059


namespace impossible_configuration_l220_22003

/-- Represents the sign at a vertex -/
inductive Sign
| Plus
| Minus

/-- Represents the state of the 12-gon -/
def TwelveGonState := Fin 12 → Sign

/-- Initial state of the 12-gon -/
def initialState : TwelveGonState :=
  fun i => if i = 0 then Sign.Minus else Sign.Plus

/-- Applies an operation to change signs at consecutive vertices -/
def applyOperation (state : TwelveGonState) (start : Fin 12) (count : Nat) : TwelveGonState :=
  fun i => if (i - start) % 12 < count then
    match state i with
    | Sign.Plus => Sign.Minus
    | Sign.Minus => Sign.Plus
    else state i

/-- Checks if the state matches the target configuration -/
def isTargetState (state : TwelveGonState) : Prop :=
  state 1 = Sign.Minus ∧ ∀ i : Fin 12, i ≠ 1 → state i = Sign.Plus

/-- The main theorem to be proved -/
theorem impossible_configuration
  (n : Nat)
  (h : n = 6 ∨ n = 4 ∨ n = 3)
  : ¬ ∃ (operations : List (Fin 12)), 
    let finalState := operations.foldl (fun s (start : Fin 12) => applyOperation s start n) initialState
    isTargetState finalState :=
sorry

end impossible_configuration_l220_22003


namespace quarter_equals_point_two_five_l220_22090

theorem quarter_equals_point_two_five : (1 : ℚ) / 4 = 0.250000000 := by
  sorry

end quarter_equals_point_two_five_l220_22090


namespace justin_lost_flowers_l220_22058

/-- Calculates the number of lost flowers given the gathering time, average time per flower,
    number of classmates, and additional time needed. -/
def lostFlowers (gatheringTime minutes : ℕ) (avgTimePerFlower : ℕ) (classmates : ℕ) (additionalTime : ℕ) : ℕ :=
  let flowersFilled := gatheringTime / avgTimePerFlower
  let additionalFlowers := additionalTime / avgTimePerFlower
  flowersFilled + additionalFlowers - classmates

/-- Theorem stating that Justin has lost 3 flowers. -/
theorem justin_lost_flowers : 
  lostFlowers 120 10 30 210 = 3 := by
  sorry

end justin_lost_flowers_l220_22058


namespace quadratic_equation_identity_l220_22036

theorem quadratic_equation_identity 
  (a₀ a₁ a₂ r s x : ℝ) 
  (h₁ : a₂ ≠ 0) 
  (h₂ : a₀ ≠ 0) 
  (h₃ : a₀ + a₁ * r + a₂ * r^2 = 0) 
  (h₄ : a₀ + a₁ * s + a₂ * s^2 = 0) :
  a₀ + a₁ * x + a₂ * x^2 = a₀ * (1 - x / r) * (1 - x / s) := by
  sorry

end quadratic_equation_identity_l220_22036


namespace items_can_fit_in_containers_l220_22096

/-- Represents an item with a weight -/
structure Item where
  weight : ℝ
  weight_bound : weight ≤ 1/2

/-- Represents a set of items -/
def ItemSet := List Item

/-- Calculate the total weight of a set of items -/
def totalWeight (items : ItemSet) : ℝ :=
  items.foldl (fun acc item => acc + item.weight) 0

/-- Theorem: Given a set of items, each weighing at most 1/2 unit, 
    with a total weight W > 1/3, these items can be placed into 
    ⌈(3W - 1)/2⌉ or fewer containers, each with a capacity of 1 unit. -/
theorem items_can_fit_in_containers (items : ItemSet) 
    (h_total_weight : totalWeight items > 1/3) :
    ∃ (num_containers : ℕ), 
      num_containers ≤ Int.ceil ((3 * totalWeight items - 1) / 2) ∧ 
      (∃ (partition : List (List Item)), 
        partition.length = num_containers ∧
        partition.all (fun container => totalWeight container ≤ 1) ∧
        partition.join = items) := by
  sorry

end items_can_fit_in_containers_l220_22096


namespace smallest_integer_in_consecutive_even_set_l220_22086

theorem smallest_integer_in_consecutive_even_set (n : ℤ) : 
  n % 2 = 0 ∧ 
  (n + 8 < 3 * ((n + (n + 2) + (n + 4) + (n + 6) + (n + 8)) / 5)) →
  n = 0 ∧ ∀ m : ℤ, (m % 2 = 0 ∧ 
    m + 8 < 3 * ((m + (m + 2) + (m + 4) + (m + 6) + (m + 8)) / 5)) →
    m ≥ n :=
by sorry

end smallest_integer_in_consecutive_even_set_l220_22086


namespace expression_evaluation_l220_22004

theorem expression_evaluation :
  68 + (156 / 12) + (11 * 19) - 250 - (450 / 9) = -10 := by
  sorry

end expression_evaluation_l220_22004


namespace candy_original_pencils_l220_22047

/-- The number of pencils each person has -/
structure PencilCounts where
  calen : ℕ
  caleb : ℕ
  candy : ℕ
  darlene : ℕ

/-- The conditions of the problem -/
def pencil_problem (p : PencilCounts) : Prop :=
  p.calen = p.caleb + 5 ∧
  p.caleb = 2 * p.candy - 3 ∧
  p.darlene = p.calen + p.caleb + p.candy + 4 ∧
  p.calen - 10 = 10

theorem candy_original_pencils (p : PencilCounts) : 
  pencil_problem p → p.candy = 9 := by
  sorry

end candy_original_pencils_l220_22047


namespace perpendicular_vectors_imply_x_coord_l220_22010

/-- Given vectors a and b in R², if a is perpendicular to (a - b), then the x-coordinate of b is 9. -/
theorem perpendicular_vectors_imply_x_coord (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.2 = -2 →
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) →
  b.1 = 9 := by
  sorry

#check perpendicular_vectors_imply_x_coord

end perpendicular_vectors_imply_x_coord_l220_22010


namespace trig_expression_equality_l220_22012

theorem trig_expression_equality : 
  (Real.sin (π/4) + Real.cos (π/6)) / (3 - 2 * Real.cos (π/3)) - 
  Real.sin (π/3) * (1 - Real.sin (π/6)) = Real.sqrt 2 / 4 := by
  sorry

end trig_expression_equality_l220_22012


namespace nina_running_distance_l220_22019

theorem nina_running_distance : 
  let first_run : ℝ := 0.08
  let second_run_part1 : ℝ := 0.08
  let second_run_part2 : ℝ := 0.67
  first_run + second_run_part1 + second_run_part2 = 0.83 := by
sorry

end nina_running_distance_l220_22019


namespace max_stamps_purchasable_l220_22049

/-- Given a stamp price of 25 cents and a budget of 5000 cents,
    the maximum number of stamps that can be purchased is 200. -/
theorem max_stamps_purchasable (stamp_price : ℕ) (budget : ℕ) (h1 : stamp_price = 25) (h2 : budget = 5000) :
  ∃ (n : ℕ), n = 200 ∧ n * stamp_price ≤ budget ∧ ∀ m : ℕ, m * stamp_price ≤ budget → m ≤ n :=
sorry

end max_stamps_purchasable_l220_22049


namespace fourth_number_in_sequence_l220_22081

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = a (n - 1) + a (n - 2)

theorem fourth_number_in_sequence 
  (a : ℕ → ℕ) 
  (h_seq : fibonacci_like_sequence a) 
  (h_7 : a 7 = 42) 
  (h_9 : a 9 = 110) : 
  a 4 = 10 := by
  sorry

end fourth_number_in_sequence_l220_22081


namespace sequence_properties_l220_22073

/-- Arithmetic sequence with a₈ = 6 and a₁₀ = 0 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  30 - 3 * n

/-- Geometric sequence with a₁ = 1/2 and a₄ = 4 -/
def geometric_sequence (n : ℕ) : ℚ :=
  2^(n - 2)

/-- Sum of the first n terms of the geometric sequence -/
def geometric_sum (n : ℕ) : ℚ :=
  2^(n - 1) - 1/2

theorem sequence_properties :
  (arithmetic_sequence 8 = 6 ∧ arithmetic_sequence 10 = 0) ∧
  (geometric_sequence 1 = 1/2 ∧ geometric_sequence 4 = 4) ∧
  (∀ n : ℕ, geometric_sum n = (geometric_sequence 1) * (1 - (2^n)) / (1 - 2)) := by
  sorry

#check sequence_properties

end sequence_properties_l220_22073


namespace library_books_count_l220_22072

theorem library_books_count (children_percentage : ℝ) (adult_count : ℕ) : 
  children_percentage = 35 →
  adult_count = 104 →
  ∃ (total : ℕ), (total : ℝ) * (1 - children_percentage / 100) = adult_count ∧ total = 160 :=
by sorry

end library_books_count_l220_22072


namespace range_of_f_l220_22032

-- Define the function f
def f (x : ℝ) : ℝ := (x^2 - 2)^2 + 1

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ f x = y) ↔ y ≥ 1 := by
  sorry

end range_of_f_l220_22032


namespace log_3_bounds_l220_22056

theorem log_3_bounds :
  2/5 < Real.log 3 / Real.log 10 ∧ Real.log 3 / Real.log 10 < 1/2 := by
  have h1 : (3 : ℝ)^5 = 243 := by norm_num
  have h2 : (3 : ℝ)^6 = 729 := by norm_num
  have h3 : (2 : ℝ)^8 = 256 := by norm_num
  have h4 : (2 : ℝ)^10 = 1024 := by norm_num
  have h5 : (10 : ℝ)^2 = 100 := by norm_num
  have h6 : (10 : ℝ)^3 = 1000 := by norm_num
  sorry

end log_3_bounds_l220_22056


namespace cubic_equation_root_l220_22005

theorem cubic_equation_root : 
  ∃ (x : ℝ), x = -4/3 ∧ (x + 1)^(1/3) + (2*x + 3)^(1/3) + 3*x + 4 = 0 :=
sorry

end cubic_equation_root_l220_22005


namespace symmetric_point_coordinates_l220_22057

/-- Given a point A with coordinates (3,2), prove that the point symmetric 
    to A' with respect to the y-axis has coordinates (1,2), where A' is obtained 
    by translating A 4 units left along the x-axis. -/
theorem symmetric_point_coordinates : 
  let A : ℝ × ℝ := (3, 2)
  let A' : ℝ × ℝ := (A.1 - 4, A.2)
  let symmetric_point : ℝ × ℝ := (-A'.1, A'.2)
  symmetric_point = (1, 2) := by sorry

end symmetric_point_coordinates_l220_22057


namespace cn_tower_height_is_553_l220_22079

/-- The height of the Space Needle in meters -/
def space_needle_height : ℕ := 184

/-- The difference in height between the CN Tower and the Space Needle in meters -/
def height_difference : ℕ := 369

/-- The height of the CN Tower in meters -/
def cn_tower_height : ℕ := space_needle_height + height_difference

theorem cn_tower_height_is_553 : cn_tower_height = 553 := by
  sorry

end cn_tower_height_is_553_l220_22079


namespace card_game_draw_probability_l220_22060

theorem card_game_draw_probability (ben_win : ℚ) (sara_win : ℚ) (h1 : ben_win = 5 / 12) (h2 : sara_win = 1 / 4) :
  1 - (ben_win + sara_win) = 1 / 3 := by
  sorry

end card_game_draw_probability_l220_22060


namespace fraction_simplification_l220_22071

theorem fraction_simplification : (1 : ℚ) / (2 + 2/3) = 3/8 := by
  sorry

end fraction_simplification_l220_22071


namespace candidate_a_votes_l220_22066

theorem candidate_a_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_a_percent : ℚ) : 
  total_votes = 560000 →
  invalid_percent = 15 / 100 →
  candidate_a_percent = 60 / 100 →
  ⌊(1 - invalid_percent) * candidate_a_percent * total_votes⌋ = 285600 :=
by sorry

end candidate_a_votes_l220_22066


namespace veridux_managers_count_l220_22026

/-- Veridux Corporation employee structure -/
structure VeriduxCorp where
  total_employees : ℕ
  female_employees : ℕ
  male_associates : ℕ
  female_managers : ℕ

/-- Theorem: The total number of managers at Veridux Corporation is 40 -/
theorem veridux_managers_count (v : VeriduxCorp)
  (h1 : v.total_employees = 250)
  (h2 : v.female_employees = 90)
  (h3 : v.male_associates = 160)
  (h4 : v.female_managers = 40)
  (h5 : v.total_employees = v.female_employees + (v.male_associates + v.female_managers)) :
  v.female_managers + (v.total_employees - v.female_employees - v.male_associates) = 40 := by
  sorry

#check veridux_managers_count

end veridux_managers_count_l220_22026


namespace unique_three_digit_divisible_by_11_l220_22065

theorem unique_three_digit_divisible_by_11 : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 10 = 3 ∧          -- units digit is 3
  n / 100 = 6 ∧         -- hundreds digit is 6
  n % 11 = 0 ∧          -- divisible by 11
  n = 693               -- the number is 693
  := by sorry

end unique_three_digit_divisible_by_11_l220_22065


namespace solution_difference_l220_22024

theorem solution_difference (r s : ℝ) : 
  ((r - 5) * (r + 5) = 25 * r - 125) →
  ((s - 5) * (s + 5) = 25 * s - 125) →
  r ≠ s →
  r > s →
  r - s = 15 := by
sorry

end solution_difference_l220_22024


namespace number_multiplication_l220_22050

theorem number_multiplication (x : ℤ) : x - 27 = 46 → x * 46 = 3358 := by
  sorry

end number_multiplication_l220_22050


namespace runner_speed_increase_l220_22033

/-- Represents a runner's speed and time improvement factors -/
structure Runner where
  initialSpeed : ℝ
  speedIncrease1 : ℝ
  speedIncrease2 : ℝ
  timeFactor1 : ℝ

/-- Theorem: If increasing speed by speedIncrease1 makes the runner timeFactor1 times faster,
    then increasing speed by speedIncrease2 will make them speedRatio times faster -/
theorem runner_speed_increase (runner : Runner)
  (h1 : runner.speedIncrease1 = 2)
  (h2 : runner.timeFactor1 = 2.5)
  (h3 : runner.speedIncrease2 = 4)
  (h4 : runner.initialSpeed > 0)
  : (runner.initialSpeed + runner.speedIncrease2) / runner.initialSpeed = 4 := by
  sorry

end runner_speed_increase_l220_22033


namespace water_flow_restrictor_problem_l220_22091

/-- Proves that given a reduced flow rate of 2 gallons per minute, which is 1 gallon per minute less than 0.6 times the original flow rate, the original flow rate is 5 gallons per minute. -/
theorem water_flow_restrictor_problem (original_rate : ℝ) : 
  (2 : ℝ) = 0.6 * original_rate - 1 → original_rate = 5 := by
  sorry

end water_flow_restrictor_problem_l220_22091


namespace lea_purchases_cost_l220_22084

/-- The cost of Léa's purchases -/
def total_cost (book_price : ℕ) (binder_price : ℕ) (notebook_price : ℕ) 
  (num_binders : ℕ) (num_notebooks : ℕ) : ℕ :=
  book_price + (binder_price * num_binders) + (notebook_price * num_notebooks)

/-- Theorem stating that the total cost of Léa's purchases is $28 -/
theorem lea_purchases_cost :
  total_cost 16 2 1 3 6 = 28 := by
  sorry

end lea_purchases_cost_l220_22084


namespace fifteen_members_without_A_l220_22051

/-- Represents the number of club members who did not receive an A in either activity. -/
def members_without_A (total_members art_A science_A both_A : ℕ) : ℕ :=
  total_members - (art_A + science_A - both_A)

/-- Theorem stating that 15 club members did not receive an A in either activity. -/
theorem fifteen_members_without_A :
  members_without_A 50 20 30 15 = 15 := by
  sorry

end fifteen_members_without_A_l220_22051


namespace quadratic_translation_l220_22094

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a horizontal translation to a quadratic function -/
def horizontalTranslation (f : QuadraticFunction) (h : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := 2 * f.a * h + f.b
  , c := f.a * h^2 + f.b * h + f.c }

/-- Applies a vertical translation to a quadratic function -/
def verticalTranslation (f : QuadraticFunction) (v : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := f.b
  , c := f.c + v }

/-- The original quadratic function y = x^2 + 1 -/
def originalFunction : QuadraticFunction :=
  { a := 1, b := 0, c := 1 }

theorem quadratic_translation :
  let f := originalFunction
  let g := verticalTranslation (horizontalTranslation f (-2)) (-3)
  g = { a := 1, b := 4, c := 2 } := by sorry

end quadratic_translation_l220_22094


namespace initial_peanuts_count_l220_22037

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Mary adds to the box -/
def peanuts_added : ℕ := 2

/-- The total number of peanuts after Mary adds some -/
def total_peanuts : ℕ := 6

/-- Theorem stating that the initial number of peanuts is 4 -/
theorem initial_peanuts_count : initial_peanuts = 4 :=
  by sorry

end initial_peanuts_count_l220_22037


namespace age_difference_l220_22016

theorem age_difference (rona_age rachel_age collete_age : ℕ) : 
  rona_age = 8 →
  rachel_age = 2 * rona_age →
  collete_age = rona_age / 2 →
  rachel_age - collete_age = 12 := by
sorry

end age_difference_l220_22016


namespace orphanage_children_count_l220_22017

/-- Represents the number of cupcakes in a package -/
inductive PackageSize
| small : PackageSize
| large : PackageSize

/-- Returns the number of cupcakes in a package -/
def packageCupcakes (size : PackageSize) : ℕ :=
  match size with
  | PackageSize.small => 10
  | PackageSize.large => 15

/-- Calculates the total number of cupcakes from a given number of packages -/
def totalCupcakes (size : PackageSize) (numPackages : ℕ) : ℕ :=
  numPackages * packageCupcakes size

/-- Represents Jean's cupcake purchase and distribution plan -/
structure CupcakePlan where
  largePacks : ℕ
  smallPacks : ℕ
  childrenCount : ℕ

/-- Theorem: The number of children in the orphanage equals the total number of cupcakes -/
theorem orphanage_children_count (plan : CupcakePlan)
  (h1 : plan.largePacks = 4)
  (h2 : plan.smallPacks = 4)
  (h3 : plan.childrenCount = totalCupcakes PackageSize.large plan.largePacks + totalCupcakes PackageSize.small plan.smallPacks) :
  plan.childrenCount = 100 := by
  sorry

end orphanage_children_count_l220_22017


namespace product_of_solutions_l220_22046

theorem product_of_solutions (x : ℝ) : 
  (|18 / x + 4| = 3) → (∃ y : ℝ, (|18 / y + 4| = 3) ∧ x * y = 324 / 7) := by
sorry

end product_of_solutions_l220_22046


namespace projection_of_v_onto_u_l220_22031

def v : Fin 2 → ℚ := ![5, 7]
def u : Fin 2 → ℚ := ![1, -3]

def projection (v u : Fin 2 → ℚ) : Fin 2 → ℚ :=
  let dot_product := (v 0) * (u 0) + (v 1) * (u 1)
  let magnitude_squared := (u 0)^2 + (u 1)^2
  let scalar := dot_product / magnitude_squared
  ![scalar * (u 0), scalar * (u 1)]

theorem projection_of_v_onto_u :
  projection v u = ![-8/5, 24/5] := by sorry

end projection_of_v_onto_u_l220_22031


namespace angle_triple_complement_l220_22029

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end angle_triple_complement_l220_22029


namespace carol_can_invite_198_friends_l220_22025

/-- The number of invitations in each package -/
def invitations_per_pack : ℕ := 18

/-- The number of packs Carol bought -/
def packs_bought : ℕ := 11

/-- The total number of friends Carol can invite -/
def friends_to_invite : ℕ := invitations_per_pack * packs_bought

/-- Theorem stating that Carol can invite 198 friends -/
theorem carol_can_invite_198_friends : friends_to_invite = 198 := by
  sorry

end carol_can_invite_198_friends_l220_22025


namespace soda_discount_percentage_l220_22040

/-- Proves that the discount percentage is 15% given the regular price and discounted price for soda cans. -/
theorem soda_discount_percentage 
  (regular_price : ℝ) 
  (discounted_price : ℝ) 
  (can_count : ℕ) :
  regular_price = 0.30 →
  discounted_price = 18.36 →
  can_count = 72 →
  (1 - discounted_price / (regular_price * can_count)) * 100 = 15 := by
  sorry

end soda_discount_percentage_l220_22040
