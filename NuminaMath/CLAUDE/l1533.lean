import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_equation_l1533_153351

-- Define the hyperbola
def is_hyperbola (a b : ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ t, x t^2 / a^2 - y t^2 / b^2 = 1

-- Define the focal length
def focal_length (c : ℝ) : Prop := c = 4 * Real.sqrt 5

-- Define the asymptotes
def asymptotes (x y : ℝ → ℝ) : Prop :=
  ∀ t, (2 * x t = y t) ∨ (2 * x t = -y t)

theorem hyperbola_equation (a b : ℝ) (x y : ℝ → ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : is_hyperbola a b x y)
  (h4 : focal_length (Real.sqrt (a^2 + b^2)))
  (h5 : asymptotes x y) :
  is_hyperbola 2 4 x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1533_153351


namespace NUMINAMATH_CALUDE_linear_function_passes_through_point_l1533_153374

/-- A linear function f(x) = kx + k - 1 passes through the point (-1, -1) for any real k. -/
theorem linear_function_passes_through_point
  (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ k * x + k - 1
  f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_point_l1533_153374


namespace NUMINAMATH_CALUDE_contest_participants_l1533_153319

theorem contest_participants (P : ℕ) 
  (h1 : (P / 2 : ℚ) = P * (1 / 2 : ℚ)) 
  (h2 : (P / 2 + P / 2 / 7 : ℚ) = P * (57.14285714285714 / 100 : ℚ)) : 
  ∃ k : ℕ, P = 7 * k :=
sorry

end NUMINAMATH_CALUDE_contest_participants_l1533_153319


namespace NUMINAMATH_CALUDE_greatest_consecutive_mixed_number_l1533_153379

/-- 
Given 6 consecutive mixed numbers with a sum of 75.5, 
prove that the greatest number is 15 1/12.
-/
theorem greatest_consecutive_mixed_number :
  ∀ (a b c d e f : ℚ),
    a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f →  -- consecutive
    b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 →  -- mixed numbers
    a + b + c + d + e + f = 75.5 →  -- sum condition
    f = 15 + 1/12 :=  -- greatest number
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_mixed_number_l1533_153379


namespace NUMINAMATH_CALUDE_additive_inverses_imply_x_equals_one_l1533_153369

theorem additive_inverses_imply_x_equals_one :
  ∀ x : ℝ, (4 * x - 1) + (3 * x - 6) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverses_imply_x_equals_one_l1533_153369


namespace NUMINAMATH_CALUDE_congruence_problem_l1533_153349

theorem congruence_problem : ∃ n : ℕ, 0 ≤ n ∧ n < 9 ∧ 
  (3 * (2 + 44 + 666 + 8888 + 111110 + 13131312 + 1515151514)) % 9 = n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1533_153349


namespace NUMINAMATH_CALUDE_equation_solution_l1533_153313

theorem equation_solution : 
  ∃ x : ℝ, (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/3)) ∧ 
  (x = ((((1 + Real.sqrt 17) / 2) ^ 3 - 2) ^ 2)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1533_153313


namespace NUMINAMATH_CALUDE_units_digit_17_pow_27_l1533_153308

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result we want to prove -/
theorem units_digit_17_pow_27 : unitsDigit (17^27) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_27_l1533_153308


namespace NUMINAMATH_CALUDE_base5_product_l1533_153306

/-- Converts a list of digits in base 5 to a natural number -/
def fromBase5 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 5 * acc + d) 0

/-- Converts a natural number to a list of digits in base 5 -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The main theorem stating that the product of 1324₅ and 23₅ in base 5 is 42112₅ -/
theorem base5_product :
  toBase5 (fromBase5 [1, 3, 2, 4] * fromBase5 [2, 3]) = [4, 2, 1, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_base5_product_l1533_153306


namespace NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l1533_153305

theorem power_three_nineteen_mod_ten : 3^19 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l1533_153305


namespace NUMINAMATH_CALUDE_sum_of_values_l1533_153358

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  prob₁ : ℝ
  prob₂ : ℝ
  h₁ : x₁ < x₂
  h₂ : prob₁ = (1 : ℝ) / 2
  h₃ : prob₂ = (1 : ℝ) / 2
  h₄ : prob₁ + prob₂ = 1

/-- Expected value of the discrete random variable -/
def expectation (X : DiscreteRV) : ℝ :=
  X.x₁ * X.prob₁ + X.x₂ * X.prob₂

/-- Variance of the discrete random variable -/
def variance (X : DiscreteRV) : ℝ :=
  (X.x₁ - expectation X)^2 * X.prob₁ + (X.x₂ - expectation X)^2 * X.prob₂

theorem sum_of_values (X : DiscreteRV) 
    (h_exp : expectation X = 2) 
    (h_var : variance X = (1 : ℝ) / 2) : 
  X.x₁ + X.x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_values_l1533_153358


namespace NUMINAMATH_CALUDE_fraction_problem_l1533_153336

theorem fraction_problem (F : ℚ) (m : ℕ) : 
  F = 1/5 ∧ m = 4 → (F^m) * (1/4)^2 = 1/((10:ℚ)^4) :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1533_153336


namespace NUMINAMATH_CALUDE_milo_running_distance_l1533_153377

def cory_speed : ℝ := 12

theorem milo_running_distance
  (h1 : cory_speed = 12)
  (h2 : ∃ milo_skateboard_speed : ℝ, cory_speed = 2 * milo_skateboard_speed)
  (h3 : ∃ milo_running_speed : ℝ, milo_skateboard_speed = 2 * milo_running_speed)
  : ∃ distance : ℝ, distance = 2 * milo_running_speed ∧ distance = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_milo_running_distance_l1533_153377


namespace NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l1533_153345

theorem power_of_eight_sum_equals_power_of_two : 8^18 + 8^18 + 8^18 = 2^56 := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l1533_153345


namespace NUMINAMATH_CALUDE_tip_calculation_correct_l1533_153386

/-- Calculates the tip amount for a family's salon visit -/
def calculate_tip (womens_haircut_price : ℚ) 
                  (childrens_haircut_price : ℚ) 
                  (teens_haircut_price : ℚ) 
                  (num_women : ℕ) 
                  (num_children : ℕ) 
                  (num_teens : ℕ) 
                  (hair_treatment_price : ℚ)
                  (tip_percentage : ℚ) : ℚ :=
  let total_cost := womens_haircut_price * num_women +
                    childrens_haircut_price * num_children +
                    teens_haircut_price * num_teens +
                    hair_treatment_price
  tip_percentage * total_cost

theorem tip_calculation_correct :
  calculate_tip 40 30 35 1 2 1 20 (1/4) = 155/4 :=
by sorry

end NUMINAMATH_CALUDE_tip_calculation_correct_l1533_153386


namespace NUMINAMATH_CALUDE_range_of_f_greater_than_x_l1533_153396

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/2) * x - 1 else 1/x

-- State the theorem
theorem range_of_f_greater_than_x :
  ∀ a : ℝ, f a > a ↔ a ∈ Set.Iio (-1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_greater_than_x_l1533_153396


namespace NUMINAMATH_CALUDE_oil_price_reduction_l1533_153357

/-- Proves that a 25% reduction in oil price allows purchasing 5 kg more oil for Rs. 1100 --/
theorem oil_price_reduction (original_price : ℝ) : 
  (original_price * 0.75 = 55) →  -- Reduced price is 55
  (1100 / 55 - 1100 / original_price = 5) := by
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l1533_153357


namespace NUMINAMATH_CALUDE_jose_bottle_caps_l1533_153327

theorem jose_bottle_caps (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 7 → received = 2 → total = initial + received → total = 9 := by
  sorry

end NUMINAMATH_CALUDE_jose_bottle_caps_l1533_153327


namespace NUMINAMATH_CALUDE_degenerate_ellipse_l1533_153368

/-- An ellipse is degenerate if and only if it consists of a single point -/
theorem degenerate_ellipse (x y c : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * (p.1)^2 + (p.2)^2 + 6 * p.1 - 12 * p.2 + c = 0) ↔ c = -39 := by
  sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_l1533_153368


namespace NUMINAMATH_CALUDE_largest_non_prime_sequence_l1533_153353

theorem largest_non_prime_sequence : ∃ (a : ℕ), 
  (a ≥ 10 ∧ a + 6 ≤ 50) ∧ 
  (∀ i ∈ (Finset.range 7), ¬ Nat.Prime (a + i)) ∧
  (∀ b : ℕ, b > a + 6 → 
    ¬(b ≥ 10 ∧ b + 6 ≤ 50 ∧ 
      (∀ i ∈ (Finset.range 7), ¬ Nat.Prime (b + i)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_non_prime_sequence_l1533_153353


namespace NUMINAMATH_CALUDE_triangle_side_length_l1533_153321

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the function to calculate the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_side_length (t : Triangle) :
  distance t.A t.B = 7 →
  distance t.A t.C = 5 →
  angle t.A t.C t.B = 2 * Real.pi / 3 →
  distance t.B t.C = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1533_153321


namespace NUMINAMATH_CALUDE_raquel_has_40_dollars_l1533_153363

-- Define the amounts of money for each person
def raquel_money : ℝ := sorry
def nataly_money : ℝ := sorry
def tom_money : ℝ := sorry

-- State the theorem
theorem raquel_has_40_dollars :
  -- Conditions
  (tom_money = (1/4) * nataly_money) →
  (nataly_money = 3 * raquel_money) →
  (tom_money + nataly_money + raquel_money = 190) →
  -- Conclusion
  raquel_money = 40 := by
  sorry

end NUMINAMATH_CALUDE_raquel_has_40_dollars_l1533_153363


namespace NUMINAMATH_CALUDE_finite_prime_triples_l1533_153312

theorem finite_prime_triples (k : ℕ) :
  Set.Finite {triple : ℕ × ℕ × ℕ | 
    let (p, q, r) := triple
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (q * r - k) % p = 0 ∧
    (p * r - k) % q = 0 ∧
    (p * q - k) % r = 0} :=
by sorry

end NUMINAMATH_CALUDE_finite_prime_triples_l1533_153312


namespace NUMINAMATH_CALUDE_garden_breadth_l1533_153348

/-- The breadth of a rectangular garden with given perimeter and length -/
theorem garden_breadth (perimeter length : ℝ) (h₁ : perimeter = 950) (h₂ : length = 375) :
  perimeter = 2 * (length + 100) := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_l1533_153348


namespace NUMINAMATH_CALUDE_ratio_sum_max_l1533_153323

theorem ratio_sum_max (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 3 / 4) (h4 : a + b = 21) : 
  max a b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_max_l1533_153323


namespace NUMINAMATH_CALUDE_ackermann_3_2_l1533_153324

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem ackermann_3_2 : A 3 2 = 11 := by sorry

end NUMINAMATH_CALUDE_ackermann_3_2_l1533_153324


namespace NUMINAMATH_CALUDE_double_base_exponent_l1533_153330

theorem double_base_exponent (a b x : ℝ) (hb : b ≠ 0) :
  (2 * a)^(2 * b) = a^b * x^b → x = 4 * a := by
  sorry

end NUMINAMATH_CALUDE_double_base_exponent_l1533_153330


namespace NUMINAMATH_CALUDE_subtract_negatives_l1533_153309

theorem subtract_negatives : -3 - 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negatives_l1533_153309


namespace NUMINAMATH_CALUDE_investment_distribution_l1533_153340

/-- Investment problem with given conditions and amounts -/
theorem investment_distribution (total : ℝ) (bonds stocks mutual_funds : ℝ) : 
  total = 210000 ∧ 
  stocks = 2 * bonds ∧ 
  mutual_funds = 4 * stocks ∧ 
  total = bonds + stocks + mutual_funds →
  bonds = 19090.91 ∧ 
  stocks = 38181.82 ∧ 
  mutual_funds = 152727.27 := by
  sorry

end NUMINAMATH_CALUDE_investment_distribution_l1533_153340


namespace NUMINAMATH_CALUDE_vector_properties_l1533_153343

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_properties :
  (∀ (m : ℝ) (a b : V), m • (a - b) = m • a - m • b) ∧
  (∀ (m n : ℝ) (a : V), (m - n) • a = m • a - n • a) ∧
  (∃ (m : ℝ) (a b : V), m • a = m • b ∧ a ≠ b) ∧
  (∀ (m n : ℝ) (a : V), a ≠ 0 → m • a = n • a → m = n) :=
sorry

end NUMINAMATH_CALUDE_vector_properties_l1533_153343


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1533_153384

/-- The complex number z = i(3+i) corresponds to a point in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant : ∃ (x y : ℝ), Complex.I * (3 + Complex.I) = Complex.mk x y ∧ x < 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1533_153384


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1533_153328

theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ 
                x = a^2 / c ∧ 
                y = -a * b / c ∧
                ∃ (x_f1 y_f1 : ℝ), (x_f1 = -c ∧ y_f1 = 0) ∧
                                   (x + x_f1) / 2 = a^2 / c ∧
                                   (y + y_f1) / 2 = -a * b / c) →
  c / a = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1533_153328


namespace NUMINAMATH_CALUDE_range_of_f_l1533_153300

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = { y | y ≥ 2 } := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1533_153300


namespace NUMINAMATH_CALUDE_third_concert_highest_attendance_l1533_153398

/-- Represents a concert with its attendance and early departure numbers -/
structure Concert where
  attendance : ℕ
  early_departure : ℕ

/-- Calculates the number of people who remained until the end of the concert -/
def remaining_attendance (c : Concert) : ℕ :=
  c.attendance - c.early_departure

/-- The three concerts attended -/
def concert1 : Concert := { attendance := 65899, early_departure := 375 }
def concert2 : Concert := { attendance := 65899 + 119, early_departure := 498 }
def concert3 : Concert := { attendance := 80453, early_departure := 612 }

theorem third_concert_highest_attendance :
  remaining_attendance concert3 > remaining_attendance concert1 ∧
  remaining_attendance concert3 > remaining_attendance concert2 :=
by sorry

end NUMINAMATH_CALUDE_third_concert_highest_attendance_l1533_153398


namespace NUMINAMATH_CALUDE_intended_number_is_five_l1533_153326

theorem intended_number_is_five : ∃! x : ℚ, (((3 * x * 10 + 2) / 19) + 7) = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_intended_number_is_five_l1533_153326


namespace NUMINAMATH_CALUDE_power_sum_difference_l1533_153393

theorem power_sum_difference : 2^(1+2+3) - (2^1 + 2^2 + 2^3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l1533_153393


namespace NUMINAMATH_CALUDE_shortest_assembly_time_is_13_l1533_153333

/-- Represents the time taken for each step in the assembly process -/
structure AssemblyTimes where
  ac : ℕ -- Time from A to C
  cd : ℕ -- Time from C to D
  be : ℕ -- Time from B to E
  ed : ℕ -- Time from E to D
  df : ℕ -- Time from D to F

/-- Calculates the shortest assembly time given the times for each step -/
def shortestAssemblyTime (times : AssemblyTimes) : ℕ :=
  max (times.ac + times.cd) (times.be + times.ed + times.df)

/-- Theorem stating that for the given assembly times, the shortest assembly time is 13 hours -/
theorem shortest_assembly_time_is_13 :
  let times : AssemblyTimes := {
    ac := 3,
    cd := 4,
    be := 3,
    ed := 4,
    df := 2
  }
  shortestAssemblyTime times = 13 := by
  sorry

end NUMINAMATH_CALUDE_shortest_assembly_time_is_13_l1533_153333


namespace NUMINAMATH_CALUDE_dress_shirt_cost_l1533_153395

theorem dress_shirt_cost (num_shirts : ℕ) (tax_rate : ℝ) (total_paid : ℝ) :
  num_shirts = 3 ∧ tax_rate = 0.1 ∧ total_paid = 66 →
  ∃ (shirt_cost : ℝ), 
    shirt_cost * num_shirts * (1 + tax_rate) = total_paid ∧
    shirt_cost = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_dress_shirt_cost_l1533_153395


namespace NUMINAMATH_CALUDE_combined_mixture_indeterminate_l1533_153399

structure TrailMix where
  nuts : ℝ
  dried_fruit : ℝ
  chocolate_chips : ℝ
  pretzels : ℝ
  granola : ℝ
  sum_to_one : nuts + dried_fruit + chocolate_chips + pretzels + granola = 1

def sue_mix : TrailMix := {
  nuts := 0.3,
  dried_fruit := 0.7,
  chocolate_chips := 0,
  pretzels := 0,
  granola := 0,
  sum_to_one := by norm_num
}

def jane_mix : TrailMix := {
  nuts := 0.6,
  dried_fruit := 0,
  chocolate_chips := 0.3,
  pretzels := 0.1,
  granola := 0,
  sum_to_one := by norm_num
}

def tom_mix : TrailMix := {
  nuts := 0.4,
  dried_fruit := 0.5,
  chocolate_chips := 0,
  pretzels := 0,
  granola := 0.1,
  sum_to_one := by norm_num
}

theorem combined_mixture_indeterminate 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c = 1) 
  (h_nuts : a * sue_mix.nuts + b * jane_mix.nuts + c * tom_mix.nuts = 0.45) :
  ∃ (x y : ℝ), 
    x ≠ y ∧ 
    (a * sue_mix.dried_fruit + b * jane_mix.dried_fruit + c * tom_mix.dried_fruit = x) ∧
    (a * sue_mix.dried_fruit + b * jane_mix.dried_fruit + c * tom_mix.dried_fruit = y) :=
sorry

end NUMINAMATH_CALUDE_combined_mixture_indeterminate_l1533_153399


namespace NUMINAMATH_CALUDE_sacks_filled_twice_l1533_153314

/-- Represents the number of times sacks can be filled with wood --/
def times_sacks_filled (father_capacity : ℕ) (ranger_capacity : ℕ) (volunteer_capacity : ℕ) (num_volunteers : ℕ) (total_wood : ℕ) : ℕ :=
  total_wood / (father_capacity + ranger_capacity + num_volunteers * volunteer_capacity)

/-- Theorem stating that under the given conditions, the sacks can be filled 2 times --/
theorem sacks_filled_twice :
  times_sacks_filled 20 30 25 2 200 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sacks_filled_twice_l1533_153314


namespace NUMINAMATH_CALUDE_loan_duration_l1533_153376

/-- Proves that the first part of a loan is lent for 8 years given specific conditions -/
theorem loan_duration (total_sum interest_rate1 interest_rate2 duration2 : ℚ) 
  (second_part : ℚ) : 
  total_sum = 2743 →
  second_part = 1688 →
  interest_rate1 = 3/100 →
  interest_rate2 = 5/100 →
  duration2 = 3 →
  let first_part := total_sum - second_part
  let duration1 := (second_part * interest_rate2 * duration2) / (first_part * interest_rate1)
  duration1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_loan_duration_l1533_153376


namespace NUMINAMATH_CALUDE_total_dumbbell_weight_l1533_153342

/-- Represents the weight of a single dumbbell in a pair --/
def dumbbell_weights : List ℕ := [3, 5, 8, 12, 18, 27]

/-- Theorem: The total weight of the dumbbell system is 146 lb --/
theorem total_dumbbell_weight : 
  (dumbbell_weights.map (·*2)).sum = 146 := by sorry

end NUMINAMATH_CALUDE_total_dumbbell_weight_l1533_153342


namespace NUMINAMATH_CALUDE_parabola_with_vertex_and_focus_parabola_through_point_l1533_153371

-- Define a parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Theorem 1
theorem parabola_with_vertex_and_focus
  (p : Parabola)
  (vertex : Point)
  (focus : Point)
  (h1 : vertex.x = 0 ∧ vertex.y = 0)
  (h2 : focus.x = 6 ∧ focus.y = 0) :
  p.equation = fun x y ↦ y^2 = 24*x :=
sorry

-- Theorem 2
theorem parabola_through_point
  (p : Parabola)
  (point : Point)
  (h : point.x = 1 ∧ point.y = 2) :
  (p.equation = fun x y ↦ x^2 = (1/2)*y) ∨
  (p.equation = fun x y ↦ y^2 = 4*x) :=
sorry

end NUMINAMATH_CALUDE_parabola_with_vertex_and_focus_parabola_through_point_l1533_153371


namespace NUMINAMATH_CALUDE_loan_duration_l1533_153375

/-- Proves that the first part of a loan was lent for 8 years given specific conditions -/
theorem loan_duration (total sum : ℕ) (second_part : ℕ) (first_rate second_rate : ℚ) (second_duration : ℕ) : 
  total = 2730 →
  second_part = 1680 →
  first_rate = 3 / 100 →
  second_rate = 5 / 100 →
  second_duration = 3 →
  ∃ (first_duration : ℕ), 
    (total - second_part) * first_rate * first_duration = second_part * second_rate * second_duration ∧
    first_duration = 8 :=
by sorry

end NUMINAMATH_CALUDE_loan_duration_l1533_153375


namespace NUMINAMATH_CALUDE_aaron_matthews_more_cows_l1533_153335

/-- Represents the number of cows each person has -/
structure CowCounts where
  aaron : ℕ
  matthews : ℕ
  marovich : ℕ

/-- The conditions of the problem -/
def cow_problem (c : CowCounts) : Prop :=
  c.aaron = 4 * c.matthews ∧
  c.matthews = 60 ∧
  c.aaron + c.matthews + c.marovich = 570

/-- The theorem to prove -/
theorem aaron_matthews_more_cows (c : CowCounts) 
  (h : cow_problem c) : c.aaron + c.matthews - c.marovich = 30 := by
  sorry


end NUMINAMATH_CALUDE_aaron_matthews_more_cows_l1533_153335


namespace NUMINAMATH_CALUDE_unique_tiling_l1533_153302

/-- A set is bounded below or above -/
def BoundedBelowOrAbove (A : Set ℝ) : Prop :=
  (∃ l, ∀ a ∈ A, l ≤ a) ∨ (∃ u, ∀ a ∈ A, a ≤ u)

/-- S tiles A -/
def Tiles (S A : Set ℝ) : Prop :=
  ∃ (I : Type) (f : I → Set ℝ), (∀ i, f i ⊆ S) ∧ (∀ i j, i ≠ j → f i ∩ f j = ∅) ∧ (⋃ i, f i) = A

/-- Unique tiling -/
def UniqueTiling (S A : Set ℝ) : Prop :=
  ∀ (I J : Type) (f : I → Set ℝ) (g : J → Set ℝ),
    (∀ i, f i ⊆ S) ∧ (∀ i j, i ≠ j → f i ∩ f j = ∅) ∧ (⋃ i, f i) = A →
    (∀ i, g i ⊆ S) ∧ (∀ i j, i ≠ j → g i ∩ g j = ∅) ∧ (⋃ i, g i) = A →
    ∃ (h : I ≃ J), ∀ i, f i = g (h i)

theorem unique_tiling (A : Set ℝ) (S : Set ℝ) :
  BoundedBelowOrAbove A → Tiles S A → UniqueTiling S A := by
  sorry

end NUMINAMATH_CALUDE_unique_tiling_l1533_153302


namespace NUMINAMATH_CALUDE_expected_value_is_500_l1533_153301

/-- Represents the prize structure for a game activity -/
structure PrizeStructure where
  firstPrize : ℝ
  commonDifference : ℝ

/-- Represents the probability distribution for winning prizes -/
structure ProbabilityDistribution where
  firstTerm : ℝ
  commonRatio : ℝ

/-- Calculates the expected value of the prize -/
def expectedValue (ps : PrizeStructure) (pd : ProbabilityDistribution) : ℝ :=
  let secondPrize := ps.firstPrize + ps.commonDifference
  let thirdPrize := ps.firstPrize + 2 * ps.commonDifference
  let secondProb := pd.firstTerm * pd.commonRatio
  let thirdProb := pd.firstTerm * pd.commonRatio * pd.commonRatio
  ps.firstPrize * pd.firstTerm + secondPrize * secondProb + thirdPrize * thirdProb

/-- The main theorem stating that the expected value is 500 yuan -/
theorem expected_value_is_500 
  (ps : PrizeStructure) 
  (pd : ProbabilityDistribution) 
  (h1 : ps.firstPrize = 700)
  (h2 : ps.commonDifference = -140)
  (h3 : pd.commonRatio = 2)
  (h4 : pd.firstTerm + pd.firstTerm * pd.commonRatio + pd.firstTerm * pd.commonRatio * pd.commonRatio = 1) :
  expectedValue ps pd = 500 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_500_l1533_153301


namespace NUMINAMATH_CALUDE_ellipse_theorem_l1533_153318

/-- Ellipse E with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  right_focus_dist : ℝ → ℝ → ℝ
  h_focus : right_focus_dist 1 (-1) = 3
  h_point : -1^2 / a^2 + (-Real.sqrt 6 / 2)^2 / b^2 = 1

/-- Line l intersecting the ellipse -/
def Line (m t : ℝ) (x y : ℝ) : Prop :=
  x - m * y - t = 0

/-- Statement of the theorem -/
theorem ellipse_theorem (E : Ellipse) :
  E.a^2 = 4 ∧ E.b^2 = 2 ∧
  ∀ m t, ∃ M N : ℝ × ℝ,
    M ≠ N ∧
    M ≠ (-E.a, 0) ∧ N ≠ (-E.a, 0) ∧
    Line m t M.1 M.2 ∧ Line m t N.1 N.2 ∧
    M.1^2 / 4 + M.2^2 / 2 = 1 ∧
    N.1^2 / 4 + N.2^2 / 2 = 1 ∧
    ((M.1 + E.a)^2 + M.2^2) * ((N.1 + E.a)^2 + N.2^2) =
      ((M.1 - N.1)^2 + (M.2 - N.2)^2) * ((M.1 + N.1 + 2*E.a)^2 + (M.2 + N.2)^2) / 4 →
    t = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l1533_153318


namespace NUMINAMATH_CALUDE_CH4_yield_is_zero_l1533_153334

-- Define the molecules and their amounts
structure Molecule :=
  (C : ℕ) (H : ℕ) (O : ℕ)

-- Define the reactions
def reaction_CH4 (m : Molecule) : Molecule :=
  {C := m.C - 1, H := m.H - 4, O := m.O}

def reaction_CO2 (m : Molecule) : Molecule :=
  {C := m.C - 1, H := m.H, O := m.O - 2}

def reaction_H2O (m : Molecule) : Molecule :=
  {C := m.C, H := m.H - 4, O := m.O - 2}

-- Define the initial amounts
def initial_amounts : Molecule :=
  {C := 3, H := 12, O := 8}  -- 3 moles C, 6 moles H2 (12 H atoms), 4 moles O2 (8 O atoms)

-- Define the theoretical yield of CH4
def theoretical_yield_CH4 (m : Molecule) : ℕ :=
  min m.C (m.H / 4)

-- Theorem statement
theorem CH4_yield_is_zero :
  theoretical_yield_CH4 (reaction_H2O (reaction_CO2 initial_amounts)) = 0 :=
sorry

end NUMINAMATH_CALUDE_CH4_yield_is_zero_l1533_153334


namespace NUMINAMATH_CALUDE_gcd_lcm_equality_l1533_153303

theorem gcd_lcm_equality (a b c x y z : ℕ+) :
  ({Nat.gcd a.val b.val, Nat.gcd b.val c.val, Nat.gcd c.val a.val} : Finset ℕ) =
  ({Nat.lcm x.val y.val, Nat.lcm y.val z.val, Nat.lcm z.val x.val} : Finset ℕ) →
  a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_equality_l1533_153303


namespace NUMINAMATH_CALUDE_M_intersect_N_l1533_153364

def M : Set ℕ := {1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a - 1}

theorem M_intersect_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1533_153364


namespace NUMINAMATH_CALUDE_product_inequality_l1533_153390

theorem product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1533_153390


namespace NUMINAMATH_CALUDE_simplified_expression_value_l1533_153310

theorem simplified_expression_value (a b : ℚ) 
  (h1 : a = -1) 
  (h2 : b = 1/4) : 
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_value_l1533_153310


namespace NUMINAMATH_CALUDE_monthly_rent_is_400_l1533_153341

/-- Calculates the monthly rent per resident in a rental building -/
def monthly_rent_per_resident (total_units : ℕ) (occupancy_rate : ℚ) (total_annual_rent : ℕ) : ℚ :=
  let occupied_units : ℚ := total_units * occupancy_rate
  let annual_rent_per_resident : ℚ := total_annual_rent / occupied_units
  annual_rent_per_resident / 12

/-- Proves that the monthly rent per resident is $400 -/
theorem monthly_rent_is_400 :
  monthly_rent_per_resident 100 (3/4) 360000 = 400 := by
  sorry

end NUMINAMATH_CALUDE_monthly_rent_is_400_l1533_153341


namespace NUMINAMATH_CALUDE_mr_green_garden_yield_l1533_153372

/-- Calculates the total expected yield from a rectangular garden -/
def gardenYield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) 
                (potato_yield : ℝ) (carrot_yield : ℝ) : ℝ :=
  let area := (length_steps : ℝ) * step_length * (width_steps : ℝ) * step_length
  area * (potato_yield + carrot_yield)

/-- Theorem stating the expected yield from Mr. Green's garden -/
theorem mr_green_garden_yield :
  gardenYield 20 25 2.5 0.5 0.25 = 2343.75 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_garden_yield_l1533_153372


namespace NUMINAMATH_CALUDE_exists_greater_or_equal_scores_64_exists_greater_or_equal_scores_49_l1533_153352

/-- Represents a student's scores on three problems -/
structure StudentScores where
  problem1 : Nat
  problem2 : Nat
  problem3 : Nat
  h1 : problem1 ≤ 7
  h2 : problem2 ≤ 7
  h3 : problem3 ≤ 7

/-- Checks if one student's scores are greater than or equal to another's -/
def scoresGreaterOrEqual (a b : StudentScores) : Prop :=
  a.problem1 ≥ b.problem1 ∧ a.problem2 ≥ b.problem2 ∧ a.problem3 ≥ b.problem3

/-- Main theorem for part (a) -/
theorem exists_greater_or_equal_scores_64 :
  ∀ (students : Fin 64 → StudentScores),
  ∃ (i j : Fin 64), i ≠ j ∧ scoresGreaterOrEqual (students i) (students j) := by
  sorry

/-- Main theorem for part (b) -/
theorem exists_greater_or_equal_scores_49 :
  ∀ (students : Fin 49 → StudentScores),
  ∃ (i j : Fin 49), i ≠ j ∧ scoresGreaterOrEqual (students i) (students j) := by
  sorry

end NUMINAMATH_CALUDE_exists_greater_or_equal_scores_64_exists_greater_or_equal_scores_49_l1533_153352


namespace NUMINAMATH_CALUDE_external_tangent_chord_length_l1533_153388

theorem external_tangent_chord_length (r₁ r₂ R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : R = 12) 
  (h₄ : r₁ + r₂ = R - r₁) (h₅ : r₁ + r₂ = R - r₂) : 
  ∃ (l : ℝ), l^2 = 518.4 ∧ 
  l^2 = 4 * ((R^2) - (((2 * r₂ + r₁) / 3)^2)) :=
by sorry

end NUMINAMATH_CALUDE_external_tangent_chord_length_l1533_153388


namespace NUMINAMATH_CALUDE_min_value_theorem_l1533_153356

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 2) :
  2/x + 1/y ≥ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 2 ∧ 2/x₀ + 1/y₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1533_153356


namespace NUMINAMATH_CALUDE_equation_solution_l1533_153350

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 5 ∧ x₂ = (4 * Real.sqrt 3) / 3 ∧ 
  (∀ x : ℝ, Real.sqrt 3 * x * (x - 5) + 4 * (5 - x) = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1533_153350


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l1533_153338

theorem consecutive_numbers_sum (a : ℕ) (h : a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 60) :
  a + 4 = 14 := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l1533_153338


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1533_153378

theorem unique_three_digit_number : ∃! abc : ℕ,
  (abc ≥ 100 ∧ abc < 1000) ∧
  (abc % 100 = (abc / 100) ^ 2) ∧
  (abc % 9 = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1533_153378


namespace NUMINAMATH_CALUDE_diesel_consumption_calculation_l1533_153331

/-- Calculates the diesel consumption of a car given its fuel efficiency, travel time, and speed. -/
theorem diesel_consumption_calculation
  (fuel_efficiency : ℝ)  -- Diesel consumption in liters per kilometer
  (travel_time : ℝ)      -- Travel time in hours
  (speed : ℝ)            -- Speed in kilometers per hour
  (h1 : fuel_efficiency = 0.14)
  (h2 : travel_time = 2.5)
  (h3 : speed = 93.6) :
  fuel_efficiency * travel_time * speed = 32.76 := by
    sorry

#check diesel_consumption_calculation

end NUMINAMATH_CALUDE_diesel_consumption_calculation_l1533_153331


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1533_153383

/-- Given a circle C with equation x^2 - 8y - 5 = -y^2 - 6x, 
    prove that a + b + r = 1 + √30, 
    where (a,b) is the center of C and r is its radius. -/
theorem circle_center_radius_sum (x y : ℝ) :
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 - 8*y - 5 = -y^2 - 6*x}
  ∃ (a b r : ℝ), (∀ (x y : ℝ), (x, y) ∈ C ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
                 (a + b + r = 1 + Real.sqrt 30) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1533_153383


namespace NUMINAMATH_CALUDE_tickets_to_sell_l1533_153366

theorem tickets_to_sell (total : ℕ) (jude andrea sandra : ℕ) : 
  total = 200 →
  andrea = 4 * jude →
  sandra = 2 * jude + 8 →
  jude = 16 →
  total - (jude + andrea + sandra) = 80 :=
by sorry

end NUMINAMATH_CALUDE_tickets_to_sell_l1533_153366


namespace NUMINAMATH_CALUDE_theo_cookie_consumption_l1533_153346

def cookies_per_sitting : ℕ := 25
def sittings_per_day : ℕ := 5
def days_per_month : ℕ := 27
def months : ℕ := 9

theorem theo_cookie_consumption :
  cookies_per_sitting * sittings_per_day * days_per_month * months = 30375 :=
by
  sorry

end NUMINAMATH_CALUDE_theo_cookie_consumption_l1533_153346


namespace NUMINAMATH_CALUDE_work_completion_time_l1533_153387

-- Define the work completion time for B
def b_time : ℝ := 8

-- Define the work completion time for A and B together
def ab_time : ℝ := 4.444444444444445

-- Define the work completion time for A
def a_time : ℝ := 10

-- Theorem statement
theorem work_completion_time :
  b_time = 8 ∧ ab_time = 4.444444444444445 →
  a_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1533_153387


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l1533_153339

/-- Given a mixture of zinc and copper in the ratio 9:11, where 27 kg of zinc is used,
    the total weight of the mixture is 60 kg. -/
theorem zinc_copper_mixture_weight : 
  ∀ (zinc copper total : ℝ),
  zinc = 27 →
  zinc / copper = 9 / 11 →
  total = zinc + copper →
  total = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l1533_153339


namespace NUMINAMATH_CALUDE_distance_incenter_circumcenter_squared_l1533_153304

-- Define a 30-60-90 right triangle with hypotenuse 2
structure Triangle30_60_90 where
  hypotenuse : ℝ
  is_30_60_90 : hypotenuse = 2

-- Define the distance between incenter and circumcenter
def distance_incenter_circumcenter (t : Triangle30_60_90) : ℝ := sorry

theorem distance_incenter_circumcenter_squared (t : Triangle30_60_90) :
  (distance_incenter_circumcenter t)^2 = 2 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_distance_incenter_circumcenter_squared_l1533_153304


namespace NUMINAMATH_CALUDE_log_equation_solution_l1533_153397

theorem log_equation_solution (a x : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (h : (Real.log x) / (Real.log (a^3)) + (Real.log a) / (Real.log (x^2)) = 2) :
  x = a^(3 + (5 * Real.sqrt 3) / 2) ∨ x = a^(3 - (5 * Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1533_153397


namespace NUMINAMATH_CALUDE_square_gt_when_abs_lt_l1533_153317

theorem square_gt_when_abs_lt (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_gt_when_abs_lt_l1533_153317


namespace NUMINAMATH_CALUDE_max_intersections_cubic_curve_l1533_153325

/-- Given a cubic curve y = x^3 - x, the maximum number of intersections
    with any tangent line passing through a point (t, 0) on the x-axis is 3 -/
theorem max_intersections_cubic_curve (t : ℝ) :
  let f (x : ℝ) := x^3 - x
  let tangent_line (x₀ : ℝ) (x : ℝ) := (3 * x₀^2 - 1) * (x - x₀) + f x₀
  ∃ (n : ℕ), n ≤ 3 ∧
    ∀ (m : ℕ), (∃ (S : Finset ℝ), S.card = m ∧
      (∀ x ∈ S, f x = tangent_line x x ∧ tangent_line x t = 0)) →
    m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_intersections_cubic_curve_l1533_153325


namespace NUMINAMATH_CALUDE_sin_double_angle_l1533_153360

theorem sin_double_angle (x : Real) (h : Real.sin (x + π/4) = 4/5) : 
  Real.sin (2*x) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_l1533_153360


namespace NUMINAMATH_CALUDE_prob_draw_heart_is_one_fourth_l1533_153370

/-- A deck of cards with a specific number of cards, ranks, and suits. -/
structure Deck where
  total_cards : ℕ
  num_ranks : ℕ
  num_suits : ℕ
  cards_per_suit : ℕ
  h1 : total_cards = num_suits * cards_per_suit
  h2 : cards_per_suit = num_ranks

/-- The probability of drawing a card from a specific suit in a given deck. -/
def prob_draw_suit (d : Deck) : ℚ :=
  d.cards_per_suit / d.total_cards

/-- The special deck described in the problem. -/
def special_deck : Deck where
  total_cards := 60
  num_ranks := 15
  num_suits := 4
  cards_per_suit := 15
  h1 := by rfl
  h2 := by rfl

theorem prob_draw_heart_is_one_fourth :
  prob_draw_suit special_deck = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_draw_heart_is_one_fourth_l1533_153370


namespace NUMINAMATH_CALUDE_negative_two_exponent_sum_l1533_153344

theorem negative_two_exponent_sum : (-2)^2023 + (-2)^2024 = 2^2023 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_exponent_sum_l1533_153344


namespace NUMINAMATH_CALUDE_board_length_problem_l1533_153329

/-- The length of a board before the final cut, given initial length, first cut, and final adjustment cut. -/
def board_length_before_final_cut (initial_length first_cut final_cut : ℕ) : ℕ :=
  initial_length - first_cut + final_cut

/-- Theorem stating that the length of the boards before the final 7 cm cut was 125 cm. -/
theorem board_length_problem :
  board_length_before_final_cut 143 25 7 = 125 := by
  sorry

end NUMINAMATH_CALUDE_board_length_problem_l1533_153329


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1533_153391

-- Define an isosceles triangle with side lengths 3 and 7
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 3 ∧ b = 7 ∧ c = 7) ∨ (a = 7 ∧ b = 3 ∧ c = 7) ∨ (a = 7 ∧ b = 7 ∧ c = 3)

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → Perimeter a b c = 17 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1533_153391


namespace NUMINAMATH_CALUDE_problem_statements_l1533_153354

theorem problem_statements :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) ∧
  (∀ P Q : Set ℝ, ∀ a : ℝ, a ∈ P ∩ Q → a ∈ P) ∧
  (∀ x : ℝ, (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ¬(∀ x : ℝ, x^2 + x + 1 ≥ 0)) ∧
  (∀ a b c : ℝ, (1 : ℝ) = 0 ↔ a + b + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l1533_153354


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l1533_153362

theorem square_perimeter_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 / b^2 = 16 / 25) → ((4 * a) / (4 * b) = 4 / 5) :=
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l1533_153362


namespace NUMINAMATH_CALUDE_second_smallest_perimeter_l1533_153332

/-- A triangle with consecutive integer side lengths -/
structure ConsecutiveIntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : b = a + 1 ∧ c = b + 1
  is_triangle : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of a triangle -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ :=
  t.a + t.b + t.c

/-- The second smallest perimeter of a triangle with consecutive integer side lengths is 12 -/
theorem second_smallest_perimeter :
  ∃ (t : ConsecutiveIntegerTriangle), perimeter t = 12 ∧
  ∀ (s : ConsecutiveIntegerTriangle), perimeter s ≠ 9 → perimeter s ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_perimeter_l1533_153332


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1533_153382

theorem imaginary_part_of_complex_fraction :
  Complex.im ((5 : ℂ) + Complex.I) / ((1 : ℂ) + Complex.I) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1533_153382


namespace NUMINAMATH_CALUDE_oil_price_reduction_l1533_153359

/-- Proves that the percentage reduction in oil price is 40% given the problem conditions -/
theorem oil_price_reduction (original_price reduced_price : ℝ) : 
  reduced_price = 120 →
  2400 / reduced_price - 2400 / original_price = 8 →
  (original_price - reduced_price) / original_price * 100 = 40 := by
  sorry

#check oil_price_reduction

end NUMINAMATH_CALUDE_oil_price_reduction_l1533_153359


namespace NUMINAMATH_CALUDE_nested_root_simplification_l1533_153316

theorem nested_root_simplification (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x * (x^3)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l1533_153316


namespace NUMINAMATH_CALUDE_total_consumption_is_7700_l1533_153337

/-- Fuel consumption rates --/
def highway_rate : ℝ := 3
def city_rate : ℝ := 5

/-- Miles driven each day --/
def day1_highway : ℝ := 200
def day1_city : ℝ := 300
def day2_highway : ℝ := 300
def day2_city : ℝ := 500
def day3_highway : ℝ := 150
def day3_city : ℝ := 350

/-- Total gas consumption calculation --/
def total_consumption : ℝ :=
  (day1_highway * highway_rate + day1_city * city_rate) +
  (day2_highway * highway_rate + day2_city * city_rate) +
  (day3_highway * highway_rate + day3_city * city_rate)

/-- Theorem stating that the total gas consumption is 7700 gallons --/
theorem total_consumption_is_7700 : total_consumption = 7700 := by
  sorry

end NUMINAMATH_CALUDE_total_consumption_is_7700_l1533_153337


namespace NUMINAMATH_CALUDE_no_integer_points_on_circle_l1533_153307

theorem no_integer_points_on_circle : 
  ¬ ∃ (x : ℤ), (x - 3)^2 + (x + 1 + 2)^2 ≤ 8^2 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_points_on_circle_l1533_153307


namespace NUMINAMATH_CALUDE_twenty_paise_coins_count_l1533_153347

/-- Given a total of 324 coins consisting of 20 paise and 25 paise denominations,
    and a total sum of Rs. 70, prove that the number of 20 paise coins is 220. -/
theorem twenty_paise_coins_count (x y : ℕ) : 
  x + y = 324 → 
  20 * x + 25 * y = 7000 → 
  x = 220 :=
by sorry

end NUMINAMATH_CALUDE_twenty_paise_coins_count_l1533_153347


namespace NUMINAMATH_CALUDE_min_value_of_z_l1533_153385

theorem min_value_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) : 
  ∀ z : ℝ, z = x^2 + 4*y^2 → z ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_z_l1533_153385


namespace NUMINAMATH_CALUDE_fraction_sum_denominator_l1533_153322

theorem fraction_sum_denominator (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 1) :
  let f1 := 3 * a / (5 * b)
  let f2 := 2 * a / (9 * b)
  let f3 := 4 * a / (15 * b)
  (f1 + f2 + f3 : ℚ) = 28 / 45 →
  5 * b + 9 * b + 15 * b = 203 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_denominator_l1533_153322


namespace NUMINAMATH_CALUDE_additional_carrots_is_38_l1533_153320

/-- The number of additional carrots picked by Carol and her mother -/
def additional_carrots (carol_carrots mother_carrots total_bad_carrots : ℝ) : ℝ :=
  total_bad_carrots - (carol_carrots + mother_carrots)

/-- Theorem stating that the number of additional carrots picked is 38 -/
theorem additional_carrots_is_38 :
  additional_carrots 29 16 83 = 38 := by
  sorry

end NUMINAMATH_CALUDE_additional_carrots_is_38_l1533_153320


namespace NUMINAMATH_CALUDE_symmetric_line_correct_l1533_153355

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-axis -/
def yAxis : Line := { a := 1, b := 0, c := 0 }

/-- Check if a line is symmetric to another line with respect to the y-axis -/
def isSymmetricToYAxis (l1 l2 : Line) : Prop :=
  l1.a = -l2.a ∧ l1.b = l2.b ∧ l1.c = l2.c

/-- The original line x - y + 1 = 0 -/
def originalLine : Line := { a := 1, b := -1, c := 1 }

/-- The symmetric line we want to prove -/
def symmetricLine : Line := { a := 1, b := 1, c := -1 }

theorem symmetric_line_correct : 
  isSymmetricToYAxis originalLine symmetricLine :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_correct_l1533_153355


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1533_153361

-- Define set A
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Define set B (domain of log(4x - x^2))
def B : Set ℝ := {x | 0 < x ∧ x < 4}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 1 4 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1533_153361


namespace NUMINAMATH_CALUDE_pushup_difference_l1533_153373

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 59

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 44

/-- The number of push-ups David did -/
def david_pushups : ℕ := 78

/-- The difference in crunches between Zachary and David -/
def crunch_difference : ℕ := 27

/-- Theorem stating the difference in push-ups between David and Zachary -/
theorem pushup_difference : david_pushups - zachary_pushups = 19 := by
  sorry

end NUMINAMATH_CALUDE_pushup_difference_l1533_153373


namespace NUMINAMATH_CALUDE_square_on_hypotenuse_side_length_l1533_153311

/-- Given a right triangle PQR with leg PR = 9 and leg PQ = 12, 
    prove that a square with one side along the hypotenuse and 
    one vertex each on legs PR and PQ has a side length of 5 5/7 -/
theorem square_on_hypotenuse_side_length 
  (P Q R : ℝ × ℝ) 
  (right_angle : (P.1 - Q.1) * (P.1 - R.1) + (P.2 - Q.2) * (P.2 - R.2) = 0)
  (leg_PR : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 9)
  (leg_PQ : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 12)
  (S : ℝ × ℝ) 
  (T : ℝ × ℝ) 
  (square_side_on_hypotenuse : ∃ U : ℝ × ℝ, 
    (S.1 - T.1) * (Q.1 - R.1) + (S.2 - T.2) * (Q.2 - R.2) = 0 ∧
    Real.sqrt ((S.1 - T.1)^2 + (S.2 - T.2)^2) = 
    Real.sqrt ((S.1 - U.1)^2 + (S.2 - U.2)^2) ∧
    Real.sqrt ((T.1 - U.1)^2 + (T.2 - U.2)^2) = 
    Real.sqrt ((S.1 - T.1)^2 + (S.2 - T.2)^2))
  (S_on_PR : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (P.1 + t * (R.1 - P.1), P.2 + t * (R.2 - P.2)))
  (T_on_PQ : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ T = (P.1 + s * (Q.1 - P.1), P.2 + s * (Q.2 - P.2))) :
  Real.sqrt ((S.1 - T.1)^2 + (S.2 - T.2)^2) = 5 + 5/7 := by
  sorry


end NUMINAMATH_CALUDE_square_on_hypotenuse_side_length_l1533_153311


namespace NUMINAMATH_CALUDE_half_plus_seven_equals_seventeen_l1533_153315

theorem half_plus_seven_equals_seventeen (x : ℝ) : (1/2) * x + 7 = 17 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_seven_equals_seventeen_l1533_153315


namespace NUMINAMATH_CALUDE_julia_dimes_count_l1533_153380

theorem julia_dimes_count : ∃ d : ℕ, 
  20 < d ∧ d < 200 ∧ 
  d % 6 = 1 ∧ 
  d % 7 = 1 ∧ 
  d % 8 = 1 ∧ 
  d = 169 := by sorry

end NUMINAMATH_CALUDE_julia_dimes_count_l1533_153380


namespace NUMINAMATH_CALUDE_total_teachers_l1533_153389

theorem total_teachers (senior : ℕ) (intermediate : ℕ) (sampled_total : ℕ) (sampled_other : ℕ)
  (h1 : senior = 26)
  (h2 : intermediate = 104)
  (h3 : sampled_total = 56)
  (h4 : sampled_other = 16)
  (h5 : ∀ (category : ℕ) (sampled_category : ℕ) (total : ℕ),
    (category : ℚ) / total = (sampled_category : ℚ) / sampled_total) :
  ∃ (total : ℕ), total = 52 := by
sorry

end NUMINAMATH_CALUDE_total_teachers_l1533_153389


namespace NUMINAMATH_CALUDE_distinct_digit_sum_l1533_153394

theorem distinct_digit_sum (A B C D : Nat) : 
  A < 10 → B < 10 → C < 10 → D < 10 →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  A + B + 1 = D →
  C + D = D + 1 →
  (∃ (count : Nat), count = 6 ∧ 
    (∀ (x : Nat), x < 10 → 
      (∃ (a b c : Nat), a < 10 ∧ b < 10 ∧ c < 10 ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ x ∧ b ≠ c ∧ b ≠ x ∧ c ≠ x ∧
        a + b + 1 = x ∧ c + x = x + 1) ↔ 
      x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9)) :=
by sorry

end NUMINAMATH_CALUDE_distinct_digit_sum_l1533_153394


namespace NUMINAMATH_CALUDE_cube_root_5488000_l1533_153392

theorem cube_root_5488000 :
  let n : ℝ := 5488000
  ∀ (x : ℝ), x^3 = n → x = 140 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_5488000_l1533_153392


namespace NUMINAMATH_CALUDE_sausage_pepperoni_difference_l1533_153365

def pizza_problem (pepperoni ham sausage : ℕ) : Prop :=
  let total_slices : ℕ := 6
  let meat_per_slice : ℕ := 22
  pepperoni = 30 ∧
  ham = 2 * pepperoni ∧
  sausage > pepperoni ∧
  (pepperoni + ham + sausage) / total_slices = meat_per_slice

theorem sausage_pepperoni_difference :
  ∀ (pepperoni ham sausage : ℕ),
    pizza_problem pepperoni ham sausage →
    sausage - pepperoni = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sausage_pepperoni_difference_l1533_153365


namespace NUMINAMATH_CALUDE_intercepts_count_l1533_153381

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 3*x - 2

-- Define x-intercepts
def is_x_intercept (x : ℝ) : Prop := f x = 0

-- Define y-intercepts
def is_y_intercept (y : ℝ) : Prop := ∃ x, f x = y ∧ x = 0

-- Theorem statement
theorem intercepts_count :
  (∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, is_x_intercept x) ∧
  (∃! y, is_y_intercept y) :=
sorry

end NUMINAMATH_CALUDE_intercepts_count_l1533_153381


namespace NUMINAMATH_CALUDE_quadratic_negative_root_condition_l1533_153367

/-- The quadratic equation ax^2 + 2x + 1 = 0 -/
def quadratic_equation (a : ℝ) (x : ℝ) : Prop := a * x^2 + 2 * x + 1 = 0

/-- A root of the quadratic equation is negative -/
def has_negative_root (a : ℝ) : Prop := ∃ x : ℝ, x < 0 ∧ quadratic_equation a x

theorem quadratic_negative_root_condition :
  (∀ a : ℝ, a < 0 → has_negative_root a) ∧
  (∃ a : ℝ, a ≥ 0 ∧ has_negative_root a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_condition_l1533_153367
