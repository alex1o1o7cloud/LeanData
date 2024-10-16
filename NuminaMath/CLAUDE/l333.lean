import Mathlib

namespace NUMINAMATH_CALUDE_adult_ticket_price_l333_33359

/-- Represents the cost of movie tickets for different age groups --/
structure TicketPrices where
  adult : ℕ
  child : ℕ
  senior : ℕ

/-- Represents the composition of Mrs. Lopez's family --/
structure Family where
  adults : ℕ
  children : ℕ
  seniors : ℕ

/-- The theorem states that given the family composition and ticket prices,
    the adult ticket price is 10 when the total cost is 64 --/
theorem adult_ticket_price 
  (prices : TicketPrices) 
  (family : Family) 
  (h1 : prices.child = 8)
  (h2 : prices.senior = 9)
  (h3 : family.adults = 3)
  (h4 : family.children = 2)
  (h5 : family.seniors = 2)
  (h6 : family.adults * prices.adult + family.children * prices.child + family.seniors * prices.senior = 64) :
  prices.adult = 10 := by
  sorry

#check adult_ticket_price

end NUMINAMATH_CALUDE_adult_ticket_price_l333_33359


namespace NUMINAMATH_CALUDE_january_salary_l333_33308

/-- Represents the monthly salary structure --/
structure MonthlySalary where
  january : ℝ
  february : ℝ
  march : ℝ
  april : ℝ
  may : ℝ

/-- Theorem stating the salary for January given the conditions --/
theorem january_salary (s : MonthlySalary) 
  (h1 : (s.january + s.february + s.march + s.april) / 4 = 8000)
  (h2 : (s.february + s.march + s.april + s.may) / 4 = 8450)
  (h3 : s.may = 6500) :
  s.january = 4700 := by
sorry

end NUMINAMATH_CALUDE_january_salary_l333_33308


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_system_l333_33347

def inequality_system (x : ℝ) : Prop :=
  2 * x - 4 ≥ 2 ∧ 3 * x - 7 < 8

theorem solution_set_of_inequality_system :
  {x : ℝ | inequality_system x} = {x : ℝ | 3 ≤ x ∧ x < 5} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_system_l333_33347


namespace NUMINAMATH_CALUDE_square_area_with_diagonal_l333_33351

/-- The area of a square with sides of length 12 meters is 144 square meters, 
    given that the diagonal of the square satisfies the Pythagorean theorem. -/
theorem square_area_with_diagonal (x : ℝ) : 
  (x^2 = 2 * 12^2) →  -- Pythagorean theorem for the diagonal
  (12 * 12 : ℝ) = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_diagonal_l333_33351


namespace NUMINAMATH_CALUDE_bill_vote_change_l333_33397

theorem bill_vote_change (total_voters : ℕ) (first_for first_against : ℕ) 
  (second_for second_against : ℕ) : 
  total_voters = 400 →
  first_for + first_against = total_voters →
  first_against > first_for →
  second_for + second_against = total_voters →
  second_for > second_against →
  (second_for - second_against) = 2 * (first_against - first_for) →
  second_for = (12 * first_against) / 11 →
  second_for - first_for = 60 := by
sorry

end NUMINAMATH_CALUDE_bill_vote_change_l333_33397


namespace NUMINAMATH_CALUDE_spade_calculation_l333_33307

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_calculation : spade 3 (spade 4 5) = -72 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l333_33307


namespace NUMINAMATH_CALUDE_largest_prime_factor_133_l333_33368

def numbers : List Nat := [45, 65, 91, 85, 133]

def largest_prime_factor (n : Nat) : Nat :=
  sorry

theorem largest_prime_factor_133 :
  ∀ m ∈ numbers, m ≠ 133 → largest_prime_factor 133 > largest_prime_factor m :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_133_l333_33368


namespace NUMINAMATH_CALUDE_systematic_sample_third_element_l333_33325

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  first_sample : ℕ
  interval : ℕ

/-- Checks if a seat number is in the systematic sample -/
def in_sample (s : SystematicSample) (seat : ℕ) : Prop :=
  ∃ k : ℕ, seat = s.first_sample + k * s.interval ∧ seat ≤ s.population_size

theorem systematic_sample_third_element 
  (s : SystematicSample)
  (h_pop : s.population_size = 45)
  (h_sample : s.sample_size = 3)
  (h_interval : s.interval = s.population_size / s.sample_size)
  (h_11 : in_sample s 11)
  (h_41 : in_sample s 41) :
  in_sample s 26 := by
  sorry

#check systematic_sample_third_element

end NUMINAMATH_CALUDE_systematic_sample_third_element_l333_33325


namespace NUMINAMATH_CALUDE_worker_c_completion_time_l333_33309

/-- The time it takes for worker c to complete a job alone, given the work rates of combinations of workers. -/
theorem worker_c_completion_time 
  (ab_rate : ℚ)  -- Rate at which workers a and b complete the job together
  (abc_rate : ℚ) -- Rate at which workers a, b, and c complete the job together
  (h1 : ab_rate = 1 / 15)  -- a and b finish the job in 15 days
  (h2 : abc_rate = 1 / 5)  -- a, b, and c finish the job in 5 days
  : (1 : ℚ) / (abc_rate - ab_rate) = 15 / 2 := by
  sorry


end NUMINAMATH_CALUDE_worker_c_completion_time_l333_33309


namespace NUMINAMATH_CALUDE_coffee_preference_expectation_l333_33339

theorem coffee_preference_expectation (total_sample : ℕ) 
  (coffee_ratio : ℚ) (h1 : coffee_ratio = 3 / 7) (h2 : total_sample = 350) : 
  ℕ := by
  sorry

#check coffee_preference_expectation

end NUMINAMATH_CALUDE_coffee_preference_expectation_l333_33339


namespace NUMINAMATH_CALUDE_tan_alpha_value_l333_33384

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = -5) : 
  Real.tan α = -23/16 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l333_33384


namespace NUMINAMATH_CALUDE_james_adoption_payment_l333_33310

/-- The total amount James pays for adopting a puppy and a kitten -/
def jamesPayment (puppyFee kittenFee : ℚ) (multiPetDiscount : ℚ) 
  (friendPuppyContribution friendKittenContribution : ℚ) : ℚ :=
  let totalFee := puppyFee + kittenFee
  let discountedFee := totalFee * (1 - multiPetDiscount)
  let friendContributions := puppyFee * friendPuppyContribution + kittenFee * friendKittenContribution
  discountedFee - friendContributions

/-- Theorem stating that James pays $242.50 for adopting a puppy and a kitten -/
theorem james_adoption_payment :
  jamesPayment 200 150 (1/10) (1/4) (3/20) = 485/2 :=
by sorry

end NUMINAMATH_CALUDE_james_adoption_payment_l333_33310


namespace NUMINAMATH_CALUDE_course_combinations_l333_33367

def type_A_courses : ℕ := 3
def type_B_courses : ℕ := 4
def total_courses_to_choose : ℕ := 3

def combinations_with_both_types (a b k : ℕ) : ℕ :=
  Nat.choose a (k - 1) * Nat.choose b 1 + Nat.choose a 1 * Nat.choose b (k - 1)

theorem course_combinations :
  combinations_with_both_types type_A_courses type_B_courses total_courses_to_choose = 30 := by
  sorry

end NUMINAMATH_CALUDE_course_combinations_l333_33367


namespace NUMINAMATH_CALUDE_smallest_period_of_special_function_l333_33361

/-- A function satisfying the given condition -/
def is_special_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The smallest positive period of a function -/
def is_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_period f p ∧ ∀ q : ℝ, 0 < q ∧ q < p → ¬ is_period f q

/-- The main theorem -/
theorem smallest_period_of_special_function (f : ℝ → ℝ) (h : is_special_function f) :
  is_smallest_positive_period f 30 :=
sorry

end NUMINAMATH_CALUDE_smallest_period_of_special_function_l333_33361


namespace NUMINAMATH_CALUDE_inequality_preservation_l333_33322

theorem inequality_preservation (x y : ℝ) (h : x > y) : x/2 > y/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l333_33322


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l333_33358

theorem interest_rate_calculation (total_investment : ℝ) (total_interest : ℝ) 
  (higher_rate : ℝ) (higher_fraction : ℝ) :
  total_investment = 20000 →
  total_interest = 1440 →
  higher_rate = 0.09 →
  higher_fraction = 0.55 →
  ∃ lower_rate : ℝ,
    lower_rate = 0.05 ∧
    total_interest = (higher_fraction * total_investment * higher_rate) + 
      ((1 - higher_fraction) * total_investment * lower_rate) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l333_33358


namespace NUMINAMATH_CALUDE_bug_meeting_point_l333_33374

/-- Triangle with side lengths 7, 8, and 9 -/
structure Triangle :=
  (PQ : ℝ) (QR : ℝ) (RP : ℝ)
  (h_PQ : PQ = 7)
  (h_QR : QR = 8)
  (h_RP : RP = 9)

/-- The meeting point of two bugs crawling from P in opposite directions -/
def meetingPoint (t : Triangle) : ℝ := sorry

/-- Theorem stating that QS = 5 in the given triangle -/
theorem bug_meeting_point (t : Triangle) : meetingPoint t = 5 := by sorry

end NUMINAMATH_CALUDE_bug_meeting_point_l333_33374


namespace NUMINAMATH_CALUDE_proportion_solution_l333_33364

theorem proportion_solution : 
  ∀ x : ℚ, (2 : ℚ) / 5 = (4 : ℚ) / 3 / x → x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l333_33364


namespace NUMINAMATH_CALUDE_apple_sale_revenue_is_408_l333_33385

/-- Calculates the money brought in from selling apples in bags -/
def apple_sale_revenue (total_harvest : ℕ) (juice_weight : ℕ) (restaurant_weight : ℕ) (bag_weight : ℕ) (price_per_bag : ℕ) : ℕ :=
  let remaining_weight := total_harvest - juice_weight - restaurant_weight
  let num_bags := remaining_weight / bag_weight
  num_bags * price_per_bag

/-- Theorem stating that the apple sale revenue is $408 given the problem conditions -/
theorem apple_sale_revenue_is_408 :
  apple_sale_revenue 405 90 60 5 8 = 408 := by
  sorry

end NUMINAMATH_CALUDE_apple_sale_revenue_is_408_l333_33385


namespace NUMINAMATH_CALUDE_outfit_combinations_l333_33356

theorem outfit_combinations : 
  let blue_shirts : ℕ := 6
  let yellow_shirts : ℕ := 4
  let pants : ℕ := 7
  let blue_hats : ℕ := 9
  let yellow_hats : ℕ := 6
  blue_shirts * pants * yellow_hats + yellow_shirts * pants * blue_hats = 504 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l333_33356


namespace NUMINAMATH_CALUDE_cauchy_schwarz_and_inequality_proof_l333_33394

theorem cauchy_schwarz_and_inequality_proof :
  (∀ a b c x y z : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 →
    (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) ≥ (a*x + b*y + c*z)^2 ∧
    ((a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = (a*x + b*y + c*z)^2 ↔ a/x = b/y ∧ b/y = c/z)) ∧
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 →
    Real.sqrt a + Real.sqrt (2*b) + Real.sqrt (3*c) ≤ Real.sqrt 6 ∧
    (Real.sqrt a + Real.sqrt (2*b) + Real.sqrt (3*c) = Real.sqrt 6 ↔ a = 1/6 ∧ b = 1/3 ∧ c = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_and_inequality_proof_l333_33394


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l333_33323

/-- The line on which the reflection occurs -/
def reflection_line (x y : ℝ) : Prop := 8 * x + 6 * y = 25

/-- The point through which the reflected ray passes -/
def reflection_point : ℝ × ℝ := (-4, 3)

/-- The origin point from which the incident ray starts -/
def origin : ℝ × ℝ := (0, 0)

/-- Theorem stating that the reflected ray has the equation y = 3 -/
theorem reflected_ray_equation :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (∃ (t : ℝ), reflection_line ((1 - t) * origin.1 + t * x) ((1 - t) * origin.2 + t * y)) →
    (∃ (s : ℝ), x = (1 - s) * reflection_point.1 + s * m ∧ 
                y = (1 - s) * reflection_point.2 + s * 3) :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l333_33323


namespace NUMINAMATH_CALUDE_intersection_slope_l333_33383

/-- The slope of the line passing through the intersection points of two circles -/
theorem intersection_slope (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 4*y - 20 = 0) ∧ 
  (x^2 + y^2 - 16*x + 8*y + 40 = 0) → 
  (∃ m : ℝ, m = -5/2 ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 20 = 0) ∧ 
      (x₁^2 + y₁^2 - 16*x₁ + 8*y₁ + 40 = 0) ∧
      (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 20 = 0) ∧ 
      (x₂^2 + y₂^2 - 16*x₂ + 8*y₂ + 40 = 0) ∧
      (x₁ ≠ x₂) →
      m = (y₂ - y₁) / (x₂ - x₁)) :=
sorry

end NUMINAMATH_CALUDE_intersection_slope_l333_33383


namespace NUMINAMATH_CALUDE_equation_solutions_l333_33380

theorem equation_solutions :
  let f (x : ℝ) := (x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)
  let g (x : ℝ) := (x - 5) * (x - 6) * (x - 5)
  ∀ x : ℝ, x ≠ 5 ∧ x ≠ 6 →
    (f x / g x = 1 ↔ x = 2 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l333_33380


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l333_33389

theorem reciprocal_inequality (a b : ℝ) : a < b → b < 0 → (1 : ℝ) / a > (1 : ℝ) / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l333_33389


namespace NUMINAMATH_CALUDE_return_speed_calculation_l333_33373

/-- Calculates the return speed given the distance, outbound speed, and total time for a round trip -/
theorem return_speed_calculation (distance : ℝ) (outbound_speed : ℝ) (total_time : ℝ) :
  distance = 19.999999999999996 →
  outbound_speed = 25 →
  total_time = 5 + 48 / 60 →
  4 = distance / (total_time - distance / outbound_speed) := by
  sorry

end NUMINAMATH_CALUDE_return_speed_calculation_l333_33373


namespace NUMINAMATH_CALUDE_ab_nonpositive_l333_33332

theorem ab_nonpositive (a b : ℚ) (ha : |a| = a) (hb : |b| ≠ b) : a * b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_nonpositive_l333_33332


namespace NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l333_33381

theorem base_b_not_divisible_by_five (b : ℤ) : b ∈ ({4, 5, 7, 8, 10} : Set ℤ) →
  (3 * b^3 - b^2 + b - 1) % 5 ≠ 0 ↔ b = 4 ∨ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l333_33381


namespace NUMINAMATH_CALUDE_cell_growth_proof_l333_33348

def geometric_sequence (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a₁ * r^(n - 1)

theorem cell_growth_proof :
  let initial_cells : ℕ := 3
  let growth_factor : ℕ := 2
  let num_terms : ℕ := 5
  geometric_sequence initial_cells growth_factor num_terms = 48 := by
  sorry

end NUMINAMATH_CALUDE_cell_growth_proof_l333_33348


namespace NUMINAMATH_CALUDE_solution_of_system_l333_33324

theorem solution_of_system (α β : ℝ) : 
  (∃ (n k : ℤ), (α = π/6 ∨ α = -π/6) ∧ α = α + 2*π*n ∧ 
                 (β = π/4 ∨ β = -π/4) ∧ β = β + 2*π*k) ∨
  (∃ (n k : ℤ), (α = π/4 ∨ α = -π/4) ∧ α = α + 2*π*n ∧ 
                 (β = π/6 ∨ β = -π/6) ∧ β = β + 2*π*k) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l333_33324


namespace NUMINAMATH_CALUDE_expression_evaluation_l333_33382

theorem expression_evaluation : (π - 2) ^ 0 - 2 * Real.sqrt 3 * 2⁻¹ - Real.sqrt 16 + |1 - Real.sqrt 3| = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l333_33382


namespace NUMINAMATH_CALUDE_shopkeeper_total_amount_l333_33336

/-- Represents the total amount a shopkeeper receives for selling cloth. -/
def totalAmount (totalMetres : ℕ) (costPrice : ℕ) (lossPerMetre : ℕ) : ℕ :=
  totalMetres * (costPrice - lossPerMetre)

/-- Proves that the shopkeeper's total amount is 18000 for the given conditions. -/
theorem shopkeeper_total_amount :
  totalAmount 600 35 5 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_total_amount_l333_33336


namespace NUMINAMATH_CALUDE_no_finite_arithmetic_partition_l333_33314

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define an arithmetic progression
def ArithmeticProgression (a : ℕ) (r : ℕ) : Set ℕ :=
  {n : ℕ | ∃ k : ℕ, n = a + k * r}

-- Define a finite partition of ℕ* into arithmetic progressions
structure FiniteArithmeticPartition :=
  (k : ℕ)
  (k_gt_one : k > 1)
  (a : Fin k → ℕ)
  (r : Fin k → ℕ)
  (distinct_r : ∀ i j : Fin k, i ≠ j → r i ≠ r j)
  (covers : ∀ n : ℕ, n > 0 → ∃ i : Fin k, n ∈ ArithmeticProgression (a i) (r i))
  (disjoint : ∀ i j : Fin k, i ≠ j → 
    ArithmeticProgression (a i) (r i) ∩ ArithmeticProgression (a j) (r j) = ∅)

-- The main theorem
theorem no_finite_arithmetic_partition :
  ¬ ∃ p : FiniteArithmeticPartition, True :=
sorry

end NUMINAMATH_CALUDE_no_finite_arithmetic_partition_l333_33314


namespace NUMINAMATH_CALUDE_minimum_balls_drawn_minimum_balls_drawn_correct_minimum_balls_drawn_minimal_l333_33318

theorem minimum_balls_drawn (blue_balls red_balls : ℕ) 
  (h_blue : blue_balls = 7) (h_red : red_balls = 5) : ℕ :=
  let total_balls := blue_balls + red_balls
  let min_blue := 2
  let min_red := 1
  8

theorem minimum_balls_drawn_correct (blue_balls red_balls : ℕ) 
  (h_blue : blue_balls = 7) (h_red : red_balls = 5) :
  ∀ n : ℕ, n ≥ minimum_balls_drawn blue_balls red_balls h_blue h_red →
  (∃ b r : ℕ, b ≥ 2 ∧ r ≥ 1 ∧ b + r ≤ n ∧ b ≤ blue_balls ∧ r ≤ red_balls) :=
by
  sorry

theorem minimum_balls_drawn_minimal (blue_balls red_balls : ℕ) 
  (h_blue : blue_balls = 7) (h_red : red_balls = 5) :
  ¬∃ m : ℕ, m < minimum_balls_drawn blue_balls red_balls h_blue h_red ∧
  (∀ n : ℕ, n ≥ m →
  (∃ b r : ℕ, b ≥ 2 ∧ r ≥ 1 ∧ b + r ≤ n ∧ b ≤ blue_balls ∧ r ≤ red_balls)) :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_balls_drawn_minimum_balls_drawn_correct_minimum_balls_drawn_minimal_l333_33318


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l333_33377

/-- An arithmetic sequence with first term a and common difference d -/
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1 : ℝ) * d

theorem arithmetic_sequence_properties (a d : ℝ) :
  let seq := arithmetic_sequence a d
  (∀ n : ℕ, n > 0 → seq (n + 1) - seq n = d) ∧
  (seq 4 = 15 ∧ seq 15 = 59) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l333_33377


namespace NUMINAMATH_CALUDE_sequence_3_9_729_arithmetic_and_geometric_l333_33338

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- A sequence is geometric if the ratio between consecutive terms is constant -/
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) / a n = r

theorem sequence_3_9_729_arithmetic_and_geometric :
  ∃ (a g : ℕ → ℝ),
    is_arithmetic a ∧ is_geometric g ∧
    (∃ i j k : ℕ, a i = 3 ∧ a j = 9 ∧ a k = 729) ∧
    (∃ x y z : ℕ, g x = 3 ∧ g y = 9 ∧ g z = 729) := by
  sorry

end NUMINAMATH_CALUDE_sequence_3_9_729_arithmetic_and_geometric_l333_33338


namespace NUMINAMATH_CALUDE_investment_proof_l333_33326

/-- Represents the total amount invested -/
def total_investment : ℝ := 15280

/-- Represents the amount invested at 6% rate -/
def investment_at_6_percent : ℝ := 8200

/-- Represents the total simple interest yield in one year -/
def total_interest : ℝ := 1023

/-- First investment rate -/
def rate_1 : ℝ := 0.06

/-- Second investment rate -/
def rate_2 : ℝ := 0.075

theorem investment_proof :
  total_investment * rate_1 * (investment_at_6_percent / total_investment) +
  total_investment * rate_2 * (1 - investment_at_6_percent / total_investment) = total_interest :=
by sorry

end NUMINAMATH_CALUDE_investment_proof_l333_33326


namespace NUMINAMATH_CALUDE_products_sum_bounds_l333_33375

def CircularArray (α : Type) := Fin 999 → α

def CircularProduct (arr : CircularArray Int) (start : Fin 999) : Int :=
  (List.range 10).foldl (λ acc i => acc * arr ((start + i) % 999)) 1

def SumOfProducts (arr : CircularArray Int) : Int :=
  (List.range 999).foldl (λ acc i => acc + CircularProduct arr i) 0

theorem products_sum_bounds 
  (arr : CircularArray Int) 
  (h1 : ∀ i, arr i = 1 ∨ arr i = -1) 
  (h2 : ∃ i j, arr i ≠ arr j) : 
  -997 ≤ SumOfProducts arr ∧ SumOfProducts arr ≤ 995 :=
sorry

end NUMINAMATH_CALUDE_products_sum_bounds_l333_33375


namespace NUMINAMATH_CALUDE_chameleon_distance_l333_33376

/-- A chameleon is a sequence of letters a, b, and c. -/
structure Chameleon (n : ℕ) where
  sequence : List Char
  length_eq : sequence.length = 3 * n
  count_a : sequence.count 'a' = n
  count_b : sequence.count 'b' = n
  count_c : sequence.count 'c' = n

/-- A swap is a transposition of two adjacent letters in a chameleon. -/
def swap (c : Chameleon n) (i : ℕ) : Chameleon n :=
  sorry

/-- The minimum number of swaps required to transform one chameleon into another. -/
def min_swaps (x y : Chameleon n) : ℕ :=
  sorry

/-- For any chameleon, there exists another chameleon that requires at least 3n²/2 swaps to reach. -/
theorem chameleon_distance (n : ℕ) (hn : 0 < n) (x : Chameleon n) :
  ∃ y : Chameleon n, 3 * n^2 / 2 ≤ min_swaps x y :=
  sorry

end NUMINAMATH_CALUDE_chameleon_distance_l333_33376


namespace NUMINAMATH_CALUDE_lily_received_35_books_l333_33391

/-- The number of books Lily received -/
def books_lily_received (mike_books_tuesday : ℕ) (corey_books_tuesday : ℕ) (mike_gave : ℕ) (corey_gave_extra : ℕ) : ℕ :=
  mike_gave + (mike_gave + corey_gave_extra)

/-- Theorem stating that Lily received 35 books -/
theorem lily_received_35_books :
  ∀ (mike_books_tuesday corey_books_tuesday mike_gave corey_gave_extra : ℕ),
    mike_books_tuesday = 45 →
    corey_books_tuesday = 2 * mike_books_tuesday →
    mike_gave = 10 →
    corey_gave_extra = 15 →
    books_lily_received mike_books_tuesday corey_books_tuesday mike_gave corey_gave_extra = 35 := by
  sorry

#eval books_lily_received 45 90 10 15

end NUMINAMATH_CALUDE_lily_received_35_books_l333_33391


namespace NUMINAMATH_CALUDE_jumping_contest_l333_33302

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) : 
  grasshopper_jump = 19 →
  frog_jump = grasshopper_jump + 10 →
  mouse_jump = grasshopper_jump + 30 →
  mouse_jump - frog_jump = 20 := by
  sorry

end NUMINAMATH_CALUDE_jumping_contest_l333_33302


namespace NUMINAMATH_CALUDE_two_numbers_sum_product_l333_33305

theorem two_numbers_sum_product (S P : ℝ) :
  ∃ (x y : ℝ), x + y = S ∧ x * y = P →
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_sum_product_l333_33305


namespace NUMINAMATH_CALUDE_storage_wheels_count_l333_33360

def total_wheels (bicycles tricycles unicycles four_wheelers : ℕ) : ℕ :=
  bicycles * 2 + tricycles * 3 + unicycles * 1 + four_wheelers * 4

theorem storage_wheels_count : total_wheels 16 7 10 5 = 83 := by
  sorry

end NUMINAMATH_CALUDE_storage_wheels_count_l333_33360


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l333_33370

/-- Given a quadratic equation x^2 + 2x + k = 0 with two equal real roots, prove that k = 1 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 + 2*y + k = 0 → y = x) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l333_33370


namespace NUMINAMATH_CALUDE_total_raisins_l333_33303

theorem total_raisins (yellow_raisins : ℝ) (black_raisins : ℝ) (red_raisins : ℝ)
  (h1 : yellow_raisins = 0.3)
  (h2 : black_raisins = 0.4)
  (h3 : red_raisins = 0.5) :
  yellow_raisins + black_raisins + red_raisins = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_total_raisins_l333_33303


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l333_33311

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℝ) : ℝ := x^2 + 1

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-2) = 5 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

#eval q (-2)
#eval q 1
#eval q 3

end NUMINAMATH_CALUDE_q_satisfies_conditions_l333_33311


namespace NUMINAMATH_CALUDE_xyz_product_l333_33337

theorem xyz_product (x y z : ℕ+) 
  (h1 : x + 2*y = z) 
  (h2 : x^2 - 4*y^2 + z^2 = 310) : 
  x*y*z = 11935 ∨ x*y*z = 2015 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l333_33337


namespace NUMINAMATH_CALUDE_more_freshmen_than_sophomores_l333_33387

theorem more_freshmen_than_sophomores :
  ∀ (total juniors not_sophomores seniors freshmen sophomores : ℕ),
  total = 800 →
  juniors = (22 * total) / 100 →
  not_sophomores = (75 * total) / 100 →
  seniors = 160 →
  freshmen + sophomores + juniors + seniors = total →
  sophomores = total - not_sophomores →
  freshmen - sophomores = 64 :=
by sorry

end NUMINAMATH_CALUDE_more_freshmen_than_sophomores_l333_33387


namespace NUMINAMATH_CALUDE_positive_real_solution_l333_33353

theorem positive_real_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a ^ b = b ^ a) (h2 : b = 4 * a) : a = (4 : ℝ) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_positive_real_solution_l333_33353


namespace NUMINAMATH_CALUDE_total_tickets_l333_33321

def tate_initial_tickets : ℕ := 32
def additional_tickets : ℕ := 2

def tate_total_tickets : ℕ := tate_initial_tickets + additional_tickets

def peyton_tickets : ℕ := tate_total_tickets / 2

theorem total_tickets : tate_total_tickets + peyton_tickets = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_l333_33321


namespace NUMINAMATH_CALUDE_bicycle_sale_price_l333_33355

def price_store_p : ℝ := 200

def regular_price_store_q : ℝ := price_store_p * 1.15

def sale_price_store_q : ℝ := regular_price_store_q * 0.9

theorem bicycle_sale_price : sale_price_store_q = 207 := by sorry

end NUMINAMATH_CALUDE_bicycle_sale_price_l333_33355


namespace NUMINAMATH_CALUDE_correct_email_sequence_l333_33330

/-- Represents the steps in sending an email -/
inductive EmailStep
  | OpenEmailBox
  | EnterRecipientAddress
  | EnterSubject
  | EnterContent
  | ClickCompose
  | ClickSend

/-- Represents a sequence of email steps -/
def EmailSequence := List EmailStep

/-- The correct sequence of steps for sending an email -/
def correctEmailSequence : EmailSequence :=
  [EmailStep.OpenEmailBox, EmailStep.ClickCompose, EmailStep.EnterRecipientAddress,
   EmailStep.EnterSubject, EmailStep.EnterContent, EmailStep.ClickSend]

/-- Theorem stating that the given sequence is the correct one for sending an email -/
theorem correct_email_sequence :
  correctEmailSequence =
    [EmailStep.OpenEmailBox, EmailStep.ClickCompose, EmailStep.EnterRecipientAddress,
     EmailStep.EnterSubject, EmailStep.EnterContent, EmailStep.ClickSend] :=
by
  sorry

end NUMINAMATH_CALUDE_correct_email_sequence_l333_33330


namespace NUMINAMATH_CALUDE_instrument_probability_l333_33313

theorem instrument_probability (total : ℕ) (at_least_one : ℕ) (two_or_more : ℕ) :
  total = 800 →
  at_least_one = total / 5 →
  two_or_more = 32 →
  (at_least_one - two_or_more : ℚ) / total = 16 / 100 := by
sorry

end NUMINAMATH_CALUDE_instrument_probability_l333_33313


namespace NUMINAMATH_CALUDE_chickens_per_coop_l333_33354

/-- Given a farm with chicken coops, prove that the number of chickens per coop is as stated. -/
theorem chickens_per_coop
  (total_coops : ℕ)
  (total_chickens : ℕ)
  (h_coops : total_coops = 9)
  (h_chickens : total_chickens = 540) :
  total_chickens / total_coops = 60 := by
  sorry

end NUMINAMATH_CALUDE_chickens_per_coop_l333_33354


namespace NUMINAMATH_CALUDE_linear_system_solution_ratio_l333_33343

/-- Given a system of linear equations with parameter k:
    x + ky + 3z = 0
    3x + ky - 2z = 0
    x + 6y - 5z = 0
    which has a nontrivial solution where x, y, z are all non-zero,
    prove that yz/x^2 = 2/3 -/
theorem linear_system_solution_ratio (k : ℝ) (x y z : ℝ) :
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + k*y + 3*z = 0 →
  3*x + k*y - 2*z = 0 →
  x + 6*y - 5*z = 0 →
  y*z / (x^2) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_ratio_l333_33343


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l333_33357

theorem decimal_to_fraction :
  (0.35 : ℚ) = 7 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l333_33357


namespace NUMINAMATH_CALUDE_x_value_l333_33333

theorem x_value : ∃ x : ℝ, 3 * x = (26 - x) + 10 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l333_33333


namespace NUMINAMATH_CALUDE_max_value_quadratic_sum_l333_33345

theorem max_value_quadratic_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - x*y + y^2 = 9) : 
  x^2 + x*y + y^2 ≤ 27 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 - a*b + b^2 = 9 ∧ a^2 + a*b + b^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_sum_l333_33345


namespace NUMINAMATH_CALUDE_inequality_proof_l333_33334

theorem inequality_proof (m n : ℕ) (h : m < Real.sqrt 2 * n) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * n^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l333_33334


namespace NUMINAMATH_CALUDE_exact_blue_marbles_probability_l333_33317

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def trials : ℕ := 6
def blue_selections : ℕ := 3

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_red : ℚ := red_marbles / total_marbles

theorem exact_blue_marbles_probability :
  (Nat.choose trials blue_selections : ℚ) *
  probability_blue ^ blue_selections *
  probability_red ^ (trials - blue_selections) =
  3512320 / 11390625 := by
sorry

end NUMINAMATH_CALUDE_exact_blue_marbles_probability_l333_33317


namespace NUMINAMATH_CALUDE_workshop_salary_calculation_l333_33327

/-- Calculates the average salary of non-technician workers in a workshop --/
theorem workshop_salary_calculation 
  (total_workers : ℕ) 
  (technicians : ℕ) 
  (avg_salary_all : ℚ) 
  (avg_salary_technicians : ℚ) 
  (h1 : total_workers = 22)
  (h2 : technicians = 7)
  (h3 : avg_salary_all = 850)
  (h4 : avg_salary_technicians = 1000) :
  let non_technicians := total_workers - technicians
  let total_salary := avg_salary_all * total_workers
  let technicians_salary := avg_salary_technicians * technicians
  let non_technicians_salary := total_salary - technicians_salary
  non_technicians_salary / non_technicians = 780 := by
  sorry


end NUMINAMATH_CALUDE_workshop_salary_calculation_l333_33327


namespace NUMINAMATH_CALUDE_range_of_a_l333_33320

theorem range_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x + 2*y + 4 = 4*x*y) 
  (h2 : ∀ x y, x > 0 → y > 0 → x + 2*y + 4 = 4*x*y → x*y + (1/2)*a^2*x + a^2*y + a - 17 ≥ 0) : 
  a ≤ -3 ∨ a ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l333_33320


namespace NUMINAMATH_CALUDE_dihedral_angle_range_l333_33319

/-- The dihedral angle between two adjacent faces in a regular n-sided polyhedron -/
def dihedral_angle (n : ℕ) (θ : ℝ) : Prop :=
  n ≥ 3 ∧ ((n - 2) / n : ℝ) * Real.pi < θ ∧ θ < Real.pi

/-- Theorem stating the range of dihedral angles in a regular n-sided polyhedron -/
theorem dihedral_angle_range (n : ℕ) :
  ∃ θ : ℝ, dihedral_angle n θ :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_range_l333_33319


namespace NUMINAMATH_CALUDE_point_transformation_theorem_l333_33398

/-- Rotate a point (x, y) counterclockwise by 90° around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

/-- Reflect a point (x, y) about the line y = x -/
def reflectAboutYeqX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation_theorem (c d : ℝ) :
  let (x₁, y₁) := rotate90 c d 3 2
  let (x₂, y₂) := reflectAboutYeqX x₁ y₁
  (x₂ = 1 ∧ y₂ = -4) → d - c = -9 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_theorem_l333_33398


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l333_33379

theorem smallest_square_containing_circle (r : ℝ) (h : r = 5) :
  (2 * r) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l333_33379


namespace NUMINAMATH_CALUDE_wrapping_paper_division_l333_33306

theorem wrapping_paper_division (total_fraction : ℚ) (num_presents : ℕ) :
  total_fraction = 1/2 →
  num_presents = 5 →
  total_fraction / num_presents = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_division_l333_33306


namespace NUMINAMATH_CALUDE_roots_when_m_zero_m_value_when_product_41_perimeter_of_isosceles_triangle_l333_33369

-- Define the quadratic equation
def quadratic_eq (m : ℝ) (x : ℝ) : Prop :=
  x^2 - 2*(m+2)*x + m^2 = 0

-- Define the roots of the equation
def roots (m : ℝ) (x₁ x₂ : ℝ) : Prop :=
  quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ x₁ ≠ x₂

-- Theorem for part 1
theorem roots_when_m_zero :
  roots 0 0 4 :=
sorry

-- Theorem for part 2
theorem m_value_when_product_41 :
  ∀ x₁ x₂ : ℝ, roots 9 x₁ x₂ → (x₁ - 2) * (x₂ - 2) = 41 :=
sorry

-- Define an isosceles triangle
def isosceles_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a = b ∨ b = c ∨ a = c)

-- Theorem for part 3
theorem perimeter_of_isosceles_triangle :
  ∀ m x₁ x₂ : ℝ, 
    roots m x₁ x₂ → 
    isosceles_triangle 9 x₁ x₂ → 
    x₁ + x₂ + 9 = 19 :=
sorry

end NUMINAMATH_CALUDE_roots_when_m_zero_m_value_when_product_41_perimeter_of_isosceles_triangle_l333_33369


namespace NUMINAMATH_CALUDE_xyz_value_l333_33365

theorem xyz_value (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (sum_sq_eq : x^2 + y^2 + z^2 = 14)
  (sum_cube_eq : x^3 + y^3 + z^3 = 17) :
  x * y * z = -7 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l333_33365


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l333_33378

theorem sin_plus_cos_value (x : ℝ) 
  (h1 : 0 < x ∧ x < π/2) 
  (h2 : Real.sin (2*x - π/4) = -Real.sqrt 2/10) : 
  Real.sin x + Real.cos x = 2*Real.sqrt 10/5 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l333_33378


namespace NUMINAMATH_CALUDE_correct_calculation_l333_33300

theorem correct_calculation (x : ℚ) (h : x + 7/5 = 81/20) : (x - 7/5) * 5 = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l333_33300


namespace NUMINAMATH_CALUDE_quadratic_function_solution_set_l333_33329

/-- Given a quadratic function f(x) = x^2 + bx + 1 where f(-1) = f(3),
    prove that the solution set of f(x) > 0 is {x ∈ ℝ | x ≠ 1} -/
theorem quadratic_function_solution_set
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^2 + b*x + 1)
  (h2 : f (-1) = f 3)
  : {x : ℝ | f x > 0} = {x : ℝ | x ≠ 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_solution_set_l333_33329


namespace NUMINAMATH_CALUDE_line_parameterization_l333_33301

/-- Given a line y = 2x - 30 parameterized by (x,y) = (f(t), 20t - 10),
    prove that f(t) = 10t + 10 -/
theorem line_parameterization (f : ℝ → ℝ) :
  (∀ x y, y = 2 * x - 30 ↔ ∃ t, x = f t ∧ y = 20 * t - 10) →
  f = fun t => 10 * t + 10 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l333_33301


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l333_33316

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define symmetry with respect to x-axis
def symmetricXAxis (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem symmetric_point_x_axis :
  let M : Point := (3, -4)
  let M' : Point := symmetricXAxis M
  M' = (3, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l333_33316


namespace NUMINAMATH_CALUDE_milk_storage_theorem_l333_33363

def initial_milk : ℕ := 30000
def pump_out_rate : ℕ := 2880
def pump_out_hours : ℕ := 4
def add_milk_hours : ℕ := 7
def initial_add_rate : ℕ := 1200
def add_rate_increase : ℕ := 200

def final_milk_amount : ℕ := 31080

theorem milk_storage_theorem :
  let milk_after_pump_out := initial_milk - pump_out_rate * pump_out_hours
  let milk_added := add_milk_hours * (initial_add_rate + (initial_add_rate + (add_milk_hours - 1) * add_rate_increase)) / 2
  milk_after_pump_out + milk_added = final_milk_amount := by sorry

end NUMINAMATH_CALUDE_milk_storage_theorem_l333_33363


namespace NUMINAMATH_CALUDE_shopkeeper_oranges_l333_33346

/-- The number of oranges bought by a shopkeeper -/
def oranges : ℕ := sorry

/-- The number of bananas bought by the shopkeeper -/
def bananas : ℕ := 400

/-- The percentage of oranges that are not rotten -/
def good_orange_percentage : ℚ := 85 / 100

/-- The percentage of bananas that are not rotten -/
def good_banana_percentage : ℚ := 92 / 100

/-- The percentage of all fruits that are in good condition -/
def total_good_percentage : ℚ := 878 / 1000

theorem shopkeeper_oranges :
  (↑oranges * good_orange_percentage + ↑bananas * good_banana_percentage) / (↑oranges + ↑bananas) = total_good_percentage ∧
  oranges = 600 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_oranges_l333_33346


namespace NUMINAMATH_CALUDE_guide_is_native_l333_33331

-- Define the two tribes
inductive Tribe
| Native
| Alien

-- Define a function to represent whether a statement is true or false
def isTruthful (t : Tribe) (s : Prop) : Prop :=
  match t with
  | Tribe.Native => s
  | Tribe.Alien => ¬s

-- Define the guide's statement
def guideStatement (encounteredTribe : Tribe) : Prop :=
  isTruthful encounteredTribe (encounteredTribe = Tribe.Native)

-- Theorem: The guide must be a native
theorem guide_is_native :
  ∀ (guideTribe : Tribe),
    (∀ (encounteredTribe : Tribe),
      isTruthful guideTribe (guideStatement encounteredTribe)) →
    guideTribe = Tribe.Native :=
by sorry


end NUMINAMATH_CALUDE_guide_is_native_l333_33331


namespace NUMINAMATH_CALUDE_gondor_repair_earnings_l333_33315

/-- Gondor's repair earnings problem -/
theorem gondor_repair_earnings (phone_repair_price laptop_repair_price : ℕ)
  (monday_phones wednesday_laptops thursday_laptops : ℕ)
  (total_earnings : ℕ) :
  phone_repair_price = 10 →
  laptop_repair_price = 20 →
  monday_phones = 3 →
  wednesday_laptops = 2 →
  thursday_laptops = 4 →
  total_earnings = 200 →
  ∃ tuesday_phones : ℕ,
    total_earnings = 
      phone_repair_price * (monday_phones + tuesday_phones) +
      laptop_repair_price * (wednesday_laptops + thursday_laptops) ∧
    tuesday_phones = 5 :=
by sorry

end NUMINAMATH_CALUDE_gondor_repair_earnings_l333_33315


namespace NUMINAMATH_CALUDE_julio_lime_cost_l333_33304

/-- Represents the number of days Julio makes mocktails -/
def days : ℕ := 30

/-- Represents the amount of lime juice used per mocktail in tablespoons -/
def juice_per_mocktail : ℚ := 1

/-- Represents the amount of lime juice that can be squeezed from one lime in tablespoons -/
def juice_per_lime : ℚ := 2

/-- Represents the number of limes sold for $1.00 -/
def limes_per_dollar : ℚ := 3

/-- Calculates the total cost of limes for Julio's mocktails over the given number of days -/
def lime_cost (d : ℕ) (j_mocktail j_lime l_dollar : ℚ) : ℚ :=
  (d * j_mocktail / j_lime) / l_dollar

/-- Theorem stating that Julio will spend $5.00 on limes after 30 days -/
theorem julio_lime_cost : 
  lime_cost days juice_per_mocktail juice_per_lime limes_per_dollar = 5 := by
  sorry

end NUMINAMATH_CALUDE_julio_lime_cost_l333_33304


namespace NUMINAMATH_CALUDE_cubes_remaining_after_removal_l333_33349

/-- Represents a cube arrangement --/
structure CubeArrangement where
  width : Nat
  height : Nat
  depth : Nat

/-- Calculates the total number of cubes in an arrangement --/
def totalCubes (arrangement : CubeArrangement) : Nat :=
  arrangement.width * arrangement.height * arrangement.depth

/-- Represents the number of vertical columns removed from the front --/
def removedColumns : Nat := 6

/-- Represents the height of each removed column --/
def removedColumnHeight : Nat := 3

/-- Calculates the number of remaining cubes after removal --/
def remainingCubes (arrangement : CubeArrangement) : Nat :=
  totalCubes arrangement - (removedColumns * removedColumnHeight)

/-- The theorem to be proved --/
theorem cubes_remaining_after_removal :
  let arrangement : CubeArrangement := { width := 4, height := 4, depth := 4 }
  remainingCubes arrangement = 46 := by
  sorry

end NUMINAMATH_CALUDE_cubes_remaining_after_removal_l333_33349


namespace NUMINAMATH_CALUDE_dogsled_race_distance_l333_33328

/-- The distance of a dogsled race course given the speeds and time differences of two teams. -/
theorem dogsled_race_distance
  (team_e_speed : ℝ)
  (team_a_speed_diff : ℝ)
  (team_a_time_diff : ℝ)
  (h1 : team_e_speed = 20)
  (h2 : team_a_speed_diff = 5)
  (h3 : team_a_time_diff = 3) :
  let team_a_speed := team_e_speed + team_a_speed_diff
  let team_e_time := (team_a_speed * team_a_time_diff) / (team_a_speed - team_e_speed)
  let distance := team_e_speed * team_e_time
  distance = 300 := by
  sorry

end NUMINAMATH_CALUDE_dogsled_race_distance_l333_33328


namespace NUMINAMATH_CALUDE_probability_ace_spade_three_correct_l333_33341

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Nat := 52

/-- Number of Aces in a standard deck -/
def NumAces : Nat := 4

/-- Number of Spades in a standard deck -/
def NumSpades : Nat := 13

/-- Number of 3s in a standard deck -/
def NumThrees : Nat := 4

/-- Probability of drawing an Ace as the first card, a Spade as the second card,
    and a 3 as the third card when dealing three cards at random from a standard deck -/
def probability_ace_spade_three : ℚ :=
  17 / 11050

theorem probability_ace_spade_three_correct :
  probability_ace_spade_three = 17 / 11050 := by
  sorry

end NUMINAMATH_CALUDE_probability_ace_spade_three_correct_l333_33341


namespace NUMINAMATH_CALUDE_small_triangle_perimeter_l333_33335

/-- Given a triangle with perimeter 11 and three trapezoids formed by cuts parallel to its sides
    with perimeters 5, 7, and 9, the perimeter of the small triangle formed after the cuts is 10. -/
theorem small_triangle_perimeter (original_perimeter : ℝ) (trapezoid1_perimeter trapezoid2_perimeter trapezoid3_perimeter : ℝ)
    (h1 : original_perimeter = 11)
    (h2 : trapezoid1_perimeter = 5)
    (h3 : trapezoid2_perimeter = 7)
    (h4 : trapezoid3_perimeter = 9) :
    trapezoid1_perimeter + trapezoid2_perimeter + trapezoid3_perimeter = original_perimeter + 10 := by
  sorry

end NUMINAMATH_CALUDE_small_triangle_perimeter_l333_33335


namespace NUMINAMATH_CALUDE_pen_pencil_ratio_l333_33342

theorem pen_pencil_ratio : 
  ∀ (num_pencils num_pens : ℕ),
    num_pencils = 42 →
    num_pencils = num_pens + 7 →
    (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_ratio_l333_33342


namespace NUMINAMATH_CALUDE_sam_total_spending_l333_33372

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1 / 10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- Calculate the total value of coins -/
def coin_value (pennies nickels dimes quarters : ℕ) : ℚ :=
  (pennies : ℚ) * penny_value + (nickels : ℚ) * nickel_value +
  (dimes : ℚ) * dime_value + (quarters : ℚ) * quarter_value

/-- Sam's spending for each day of the week -/
def monday_spending : ℚ := coin_value 5 3 0 0
def tuesday_spending : ℚ := coin_value 0 0 8 4
def wednesday_spending : ℚ := coin_value 0 7 10 2
def thursday_spending : ℚ := coin_value 20 15 12 6
def friday_spending : ℚ := coin_value 45 20 25 10

/-- The total amount Sam spent during the week -/
def total_spending : ℚ :=
  monday_spending + tuesday_spending + wednesday_spending + thursday_spending + friday_spending

/-- Theorem: Sam spent $14.05 in total during the week -/
theorem sam_total_spending : total_spending = 1405 / 100 := by
  sorry


end NUMINAMATH_CALUDE_sam_total_spending_l333_33372


namespace NUMINAMATH_CALUDE_max_value_is_b_l333_33352

theorem max_value_is_b (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b = max (max (max (1/2) b) (2*a*b)) (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_is_b_l333_33352


namespace NUMINAMATH_CALUDE_course_selection_theorem_l333_33344

def category_A_courses : ℕ := 3
def category_B_courses : ℕ := 4
def total_courses_to_choose : ℕ := 3

/-- The number of ways to choose courses from two categories with the given constraints -/
def number_of_ways_to_choose : ℕ :=
  (Nat.choose category_A_courses 1 * Nat.choose category_B_courses 2) +
  (Nat.choose category_A_courses 2 * Nat.choose category_B_courses 1)

theorem course_selection_theorem :
  number_of_ways_to_choose = 30 :=
by sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l333_33344


namespace NUMINAMATH_CALUDE_multiplication_problem_l333_33388

theorem multiplication_problem (x : ℝ) : 4 * x = 60 → 8 * x = 120 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l333_33388


namespace NUMINAMATH_CALUDE_quadratic_condition_necessary_not_sufficient_l333_33390

theorem quadratic_condition_necessary_not_sufficient :
  (∀ b : ℝ, (∀ x : ℝ, x^2 - b*x + 1 > 0) → b ∈ Set.Icc 0 1) ∧
  ¬(∀ b : ℝ, b ∈ Set.Icc 0 1 → (∀ x : ℝ, x^2 - b*x + 1 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_necessary_not_sufficient_l333_33390


namespace NUMINAMATH_CALUDE_inequality_solution_set_l333_33392

theorem inequality_solution_set (x : ℝ) : 
  (5 ≤ x / (3 * x - 8) ∧ x / (3 * x - 8) < 10) ↔ (8 / 3 < x ∧ x ≤ 20 / 7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l333_33392


namespace NUMINAMATH_CALUDE_inequality_range_l333_33393

theorem inequality_range (a : ℝ) : 
  (∀ (x y : ℝ), x > 0 ∧ y > 0 → (y / 4) - Real.cos x ^ 2 ≥ a * Real.sin x - (9 / y)) →
  -3 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l333_33393


namespace NUMINAMATH_CALUDE_festival_attendance_l333_33312

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (h_total : total_students = 1500)
  (h_attendees : festival_attendees = 820) :
  ∃ (girls boys : ℕ),
    girls + boys = total_students ∧
    (3 * girls) / 4 + (2 * boys) / 5 = festival_attendees ∧
    (3 * girls) / 4 = 471 := by
  sorry

end NUMINAMATH_CALUDE_festival_attendance_l333_33312


namespace NUMINAMATH_CALUDE_fabric_difference_total_fabric_l333_33396

/-- The amount of fabric used to make a coat, in meters -/
def coat_fabric : ℝ := 1.55

/-- The amount of fabric used to make a pair of pants, in meters -/
def pants_fabric : ℝ := 1.05

/-- The difference in fabric usage between a coat and pants is 0.5 meters -/
theorem fabric_difference : coat_fabric - pants_fabric = 0.5 := by sorry

/-- The total fabric needed for a coat and pants is 2.6 meters -/
theorem total_fabric : coat_fabric + pants_fabric = 2.6 := by sorry

end NUMINAMATH_CALUDE_fabric_difference_total_fabric_l333_33396


namespace NUMINAMATH_CALUDE_triangle_perimeter_strict_l333_33350

theorem triangle_perimeter_strict (a b x : ℝ) : 
  a = 12 → b = 25 → a > 0 → b > 0 → x > 0 → 
  a + b > x → a + x > b → b + x > a → 
  a + b + x > 50 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_strict_l333_33350


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l333_33399

theorem simplify_and_rationalize (x : ℝ) (hx : x > 0) :
  (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt x / Real.sqrt 12) * (Real.sqrt 6 / Real.sqrt 8) = 
  Real.sqrt (1260 * x) / 168 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l333_33399


namespace NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l333_33386

-- Define the property of being a positive number
def IsPositive (x : ℚ) : Prop := x > 0

-- Define the property of being a negative number
def IsNegative (x : ℚ) : Prop := x < 0

-- Theorem statement
theorem zero_neither_positive_nor_negative : 
  ¬(IsPositive 0) ∧ ¬(IsNegative 0) :=
sorry

end NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l333_33386


namespace NUMINAMATH_CALUDE_infinite_solutions_l333_33371

theorem infinite_solutions (a b : ℝ) :
  (∀ x, 4 * (3 * x - a) = 3 * (4 * x + b)) ↔ b = -4/3 * a := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_l333_33371


namespace NUMINAMATH_CALUDE_vector_operation_l333_33395

/-- Given two vectors AB and AC in R², prove that 2AB - AC equals (5,7) -/
theorem vector_operation (AB AC : ℝ × ℝ) 
  (h1 : AB = (2, 3)) 
  (h2 : AC = (-1, -1)) : 
  (2 : ℝ) • AB - AC = (5, 7) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l333_33395


namespace NUMINAMATH_CALUDE_arithmetic_sequence_range_l333_33362

theorem arithmetic_sequence_range (a : ℝ) :
  (∀ n : ℕ+, (1 + (a + n - 1)) / (a + n - 1) ≤ (1 + (a + 5 - 1)) / (a + 5 - 1)) →
  -4 < a ∧ a < -3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_range_l333_33362


namespace NUMINAMATH_CALUDE_ascending_four_digit_difference_l333_33366

/-- Represents a four-digit number where each subsequent digit is 1 greater than the previous one -/
structure AscendingFourDigitNumber where
  first_digit : ℕ
  constraint : first_digit ≤ 6

/-- Calculates the value of the four-digit number -/
def value (n : AscendingFourDigitNumber) : ℕ :=
  1000 * n.first_digit + 100 * (n.first_digit + 1) + 10 * (n.first_digit + 2) + (n.first_digit + 3)

/-- Calculates the value of the reversed four-digit number -/
def reverse_value (n : AscendingFourDigitNumber) : ℕ :=
  1000 * (n.first_digit + 3) + 100 * (n.first_digit + 2) + 10 * (n.first_digit + 1) + n.first_digit

/-- The main theorem stating that the difference between the reversed number and the original number is always 3087 -/
theorem ascending_four_digit_difference (n : AscendingFourDigitNumber) :
  reverse_value n - value n = 3087 := by
  sorry

end NUMINAMATH_CALUDE_ascending_four_digit_difference_l333_33366


namespace NUMINAMATH_CALUDE_car_travel_distance_l333_33340

/-- Given a car that travels 300 miles on 10 gallons of gas, 
    prove that it will travel 450 miles on 15 gallons of gas. -/
theorem car_travel_distance (miles : ℝ) (gallons : ℝ) 
  (h1 : miles = 300) (h2 : gallons = 10) :
  (miles / gallons) * 15 = 450 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l333_33340
