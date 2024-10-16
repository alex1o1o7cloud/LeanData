import Mathlib

namespace NUMINAMATH_CALUDE_unique_prime_satisfying_conditions_l3378_337851

theorem unique_prime_satisfying_conditions :
  ∃! (n : ℕ), n.Prime ∧ 
    (n^2 + 10).Prime ∧ 
    (n^2 - 2).Prime ∧ 
    (n^3 + 6).Prime ∧ 
    (n^5 + 36).Prime ∧ 
    n = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_satisfying_conditions_l3378_337851


namespace NUMINAMATH_CALUDE_shirt_sale_price_l3378_337857

/-- Given a shirt with a cost price, profit margin, and discount percentage,
    calculate the final sale price. -/
def final_sale_price (cost_price : ℝ) (profit_margin : ℝ) (discount : ℝ) : ℝ :=
  let selling_price := cost_price * (1 + profit_margin)
  selling_price * (1 - discount)

/-- Theorem stating that for a shirt with a cost price of $20, a profit margin of 30%,
    and a discount of 50%, the final sale price is $13. -/
theorem shirt_sale_price :
  final_sale_price 20 0.3 0.5 = 13 := by
sorry

end NUMINAMATH_CALUDE_shirt_sale_price_l3378_337857


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3378_337812

-- Define set A
def A : Set Int := {x | (x + 2) * (x - 1) < 0}

-- Define set B
def B : Set Int := {-2, -1}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3378_337812


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3378_337822

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 3 * a 4 * a 5 = 3 →
  a 6 * a 7 * a 8 = 21 →
  a 9 * a 10 * a 11 = 147 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l3378_337822


namespace NUMINAMATH_CALUDE_tire_pricing_ratio_l3378_337881

/-- Represents the daily tire production capacity --/
def daily_production : ℕ := 1000

/-- Represents the daily tire demand --/
def daily_demand : ℕ := 1200

/-- Represents the production cost of each tire in cents --/
def production_cost : ℕ := 25000

/-- Represents the weekly loss in cents due to limited production capacity --/
def weekly_loss : ℕ := 17500000

/-- Represents the ratio of selling price to production cost --/
def selling_price_ratio : ℚ := 3/2

theorem tire_pricing_ratio :
  daily_production = 1000 →
  daily_demand = 1200 →
  production_cost = 25000 →
  weekly_loss = 17500000 →
  selling_price_ratio = 3/2 := by sorry

end NUMINAMATH_CALUDE_tire_pricing_ratio_l3378_337881


namespace NUMINAMATH_CALUDE_water_level_decrease_l3378_337861

def water_level_change (change : ℝ) : ℝ := change

theorem water_level_decrease (decrease : ℝ) : 
  water_level_change (-decrease) = -decrease :=
by sorry

end NUMINAMATH_CALUDE_water_level_decrease_l3378_337861


namespace NUMINAMATH_CALUDE_max_value_of_a_l3378_337831

/-- An odd function that is increasing on the non-negative reals -/
structure OddIncreasingFunction (f : ℝ → ℝ) : Prop where
  odd : ∀ x, f (-x) = -f x
  increasing_nonneg : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

/-- The condition relating f, a, x, and t -/
def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x t, x ∈ Set.Icc 1 2 → t ∈ Set.Icc 1 2 →
    f (x^2 + a*x + a) ≤ f (-a*t^2 - t + 1)

theorem max_value_of_a (f : ℝ → ℝ) (hf : OddIncreasingFunction f) :
  (∃ a, condition f a) → (∀ a, condition f a → a ≤ -1) ∧ (condition f (-1)) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3378_337831


namespace NUMINAMATH_CALUDE_line_x_intercept_l3378_337825

theorem line_x_intercept (t : ℝ) (h : t ∈ Set.Icc 0 (2 * Real.pi)) :
  let x := 2 * Real.cos t + 3
  let y := -1 + 5 * Real.sin t
  y = 0 → Real.sin t = 1/5 ∧ x = 2 * Real.cos (Real.arcsin (1/5)) + 3 := by
  sorry

end NUMINAMATH_CALUDE_line_x_intercept_l3378_337825


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3378_337865

-- Problem 1
theorem problem_1 : -7 - |(-9)| - (-11) - 3 = -8 := by sorry

-- Problem 2
theorem problem_2 : 5.6 + (-0.9) + 4.4 + (-8.1) = 1 := by sorry

-- Problem 3
theorem problem_3 : (-1/6 : ℚ) + (1/3 : ℚ) + (-1/12 : ℚ) = 1/12 := by sorry

-- Problem 4
theorem problem_4 : (2/5 : ℚ) - |(-1.5 : ℚ)| - (2.25 : ℚ) - (-2.75 : ℚ) = -0.6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3378_337865


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3378_337871

theorem quadratic_root_in_unit_interval (a b c : ℝ) (h : 2*a + 3*b + 6*c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3378_337871


namespace NUMINAMATH_CALUDE_find_number_of_elements_number_of_elements_is_ten_l3378_337878

/-- Given an incorrect average and a correction, find the number of elements -/
theorem find_number_of_elements (incorrect_avg correct_avg : ℚ) 
  (incorrect_value correct_value : ℚ) : ℚ :=
  let n := (correct_value - incorrect_value) / (correct_avg - incorrect_avg)
  n

/-- Proof that the number of elements is 10 given the specific conditions -/
theorem number_of_elements_is_ten : 
  find_number_of_elements 20 26 26 86 = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_number_of_elements_number_of_elements_is_ten_l3378_337878


namespace NUMINAMATH_CALUDE_sqrt_18_greater_than_pi_l3378_337889

theorem sqrt_18_greater_than_pi : Real.sqrt 18 > Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_greater_than_pi_l3378_337889


namespace NUMINAMATH_CALUDE_intersection_and_solution_set_l3378_337828

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the solution set of x^2 + ax - b < 0
def solution_set (a b : ℝ) : Set ℝ := {x | x < -1 ∨ x > 2}

theorem intersection_and_solution_set :
  (A ∩ B = A_intersect_B) ∧
  (∀ a b : ℝ, ({x : ℝ | x^2 + a*x + b < 0} = A_intersect_B) →
              ({x : ℝ | x^2 + a*x - b < 0} = solution_set a b)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_solution_set_l3378_337828


namespace NUMINAMATH_CALUDE_inequality_proof_l3378_337859

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 1) : 
  x^2 + y^2 + z^2 + (Real.sqrt 3 / 2) * Real.sqrt (x * y * z) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3378_337859


namespace NUMINAMATH_CALUDE_largest_unrepresentable_amount_is_correct_l3378_337873

/-- Represents the set of coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Finset ℕ :=
  {6*n + 1, 6*n + 4, 6*n + 7, 6*n + 10}

/-- Predicate to check if an amount can be represented using given coin denominations -/
def is_representable (s : ℕ) (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), s = a*(6*n + 1) + b*(6*n + 4) + c*(6*n + 7) + d*(6*n + 10)

/-- The largest amount that cannot be represented using the given coin denominations -/
def largest_unrepresentable_amount (n : ℕ) : ℕ :=
  12*n^2 + 14*n - 1

/-- Theorem stating that the largest_unrepresentable_amount is correct -/
theorem largest_unrepresentable_amount_is_correct (n : ℕ) :
  (∀ k < largest_unrepresentable_amount n, is_representable k n) ∧
  ¬is_representable (largest_unrepresentable_amount n) n :=
by sorry

end NUMINAMATH_CALUDE_largest_unrepresentable_amount_is_correct_l3378_337873


namespace NUMINAMATH_CALUDE_no_integer_solution_l3378_337885

theorem no_integer_solution (n : ℕ) (hn : n ≥ 11) :
  ¬ ∃ m : ℤ, m^2 + 2 * 3^n = m * (2^(n+1) - 1) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3378_337885


namespace NUMINAMATH_CALUDE_equation_solution_l3378_337858

theorem equation_solution : 
  ∃ (n : ℚ), (2 / (n + 2) + 3 / (n + 2) + n / (n + 2) + 1 / (n + 2) = 4) ∧ (n = -2/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3378_337858


namespace NUMINAMATH_CALUDE_square_of_cube_third_smallest_prime_l3378_337842

def third_smallest_prime : Nat := 5

theorem square_of_cube_third_smallest_prime : 
  (third_smallest_prime ^ 3) ^ 2 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_square_of_cube_third_smallest_prime_l3378_337842


namespace NUMINAMATH_CALUDE_martha_initial_blocks_l3378_337835

/-- Given that Martha finds 80 blocks and ends up with 84 blocks, 
    prove that she initially had 4 blocks. -/
theorem martha_initial_blocks : 
  ∀ (initial_blocks found_blocks final_blocks : ℕ),
    found_blocks = 80 →
    final_blocks = 84 →
    final_blocks = initial_blocks + found_blocks →
    initial_blocks = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_martha_initial_blocks_l3378_337835


namespace NUMINAMATH_CALUDE_least_subtrahend_l3378_337832

def problem (n : ℕ) : Prop :=
  (2590 - n) % 9 = 6 ∧ 
  (2590 - n) % 11 = 6 ∧ 
  (2590 - n) % 13 = 6

theorem least_subtrahend : 
  problem 10 ∧ ∀ m : ℕ, m < 10 → ¬(problem m) :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_l3378_337832


namespace NUMINAMATH_CALUDE_sum_of_squares_l3378_337824

theorem sum_of_squares (a b c : ℝ) : 
  (a * b + b * c + a * c = 50) → 
  (a + b + c = 16) → 
  (a^2 + b^2 + c^2 = 156) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3378_337824


namespace NUMINAMATH_CALUDE_inequality_problem_l3378_337894

theorem inequality_problem (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / (a - 1) ≥ 1 / b) ∧
  (1 / b < 1 / a) ∧
  (|a| > -b) ∧
  (Real.sqrt (-a) > Real.sqrt (-b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l3378_337894


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3378_337846

theorem system_of_equations_solution :
  ∃! (x y z u : ℤ),
    x + y + z = 15 ∧
    x + y + u = 16 ∧
    x + z + u = 18 ∧
    y + z + u = 20 ∧
    x = 3 ∧ y = 5 ∧ z = 7 ∧ u = 8 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3378_337846


namespace NUMINAMATH_CALUDE_bbq_ice_cost_chad_bbq_ice_cost_l3378_337818

/-- The cost of ice for a BBQ given the number of people, ice needed per person, and ice price --/
theorem bbq_ice_cost (people : ℕ) (ice_per_person : ℕ) (pack_size : ℕ) (pack_price : ℚ) : ℚ :=
  let total_ice := people * ice_per_person
  let packs_needed := (total_ice + pack_size - 1) / pack_size  -- Ceiling division
  packs_needed * pack_price

/-- Proof that the cost of ice for Chad's BBQ is $9.00 --/
theorem chad_bbq_ice_cost : bbq_ice_cost 15 2 10 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_bbq_ice_cost_chad_bbq_ice_cost_l3378_337818


namespace NUMINAMATH_CALUDE_pump_time_correct_l3378_337880

/-- The time it takes for the pump to fill the tank without the leak -/
def pump_time : ℝ := 6

/-- The time it takes to fill the tank with both pump and leak -/
def fill_time_with_leak : ℝ := 12

/-- The time it takes for the leak to empty the tank -/
def leak_empty_time : ℝ := 12

/-- Theorem stating that the pump time is correct given the conditions -/
theorem pump_time_correct : 
  (1 / pump_time - 1 / leak_empty_time) = 1 / fill_time_with_leak := by sorry

end NUMINAMATH_CALUDE_pump_time_correct_l3378_337880


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3378_337804

theorem pure_imaginary_condition (z : ℂ) (a b : ℝ) : 
  z = Complex.mk a b → z.re = 0 → a = 0 ∧ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3378_337804


namespace NUMINAMATH_CALUDE_max_profit_is_45_6_l3378_337872

-- Define the profit functions
def profit_A (t : ℕ) : ℚ := 5.06 * t - 0.15 * t^2
def profit_B (t : ℕ) : ℚ := 2 * t

-- Define the total profit function
def total_profit (x : ℕ) : ℚ := profit_A x + profit_B (15 - x)

-- Theorem statement
theorem max_profit_is_45_6 :
  ∃ (x : ℕ), x ≤ 15 ∧ total_profit x = 45.6 ∧
  ∀ (y : ℕ), y ≤ 15 → total_profit y ≤ 45.6 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_is_45_6_l3378_337872


namespace NUMINAMATH_CALUDE_product_polynomials_l3378_337866

theorem product_polynomials (g h : ℚ) :
  (∀ d : ℚ, (7*d^2 - 3*d + g) * (3*d^2 + h*d - 8) = 21*d^4 - 44*d^3 - 35*d^2 + 14*d - 16) →
  g + h = -3 := by
  sorry

end NUMINAMATH_CALUDE_product_polynomials_l3378_337866


namespace NUMINAMATH_CALUDE_pool_depth_is_10_feet_l3378_337892

/-- Represents the dimensions and properties of a pool -/
structure Pool where
  width : ℝ
  length : ℝ
  depth : ℝ
  capacity : ℝ
  drainRate : ℝ
  drainTime : ℝ
  initialFillPercentage : ℝ

/-- Calculates the volume of water drained from the pool -/
def volumeDrained (p : Pool) : ℝ := p.drainRate * p.drainTime

/-- Calculates the total capacity of the pool -/
def totalCapacity (p : Pool) : ℝ := p.width * p.length * p.depth

/-- Theorem stating that the depth of the pool is 10 feet -/
theorem pool_depth_is_10_feet (p : Pool) 
  (h1 : p.width = 40)
  (h2 : p.length = 150)
  (h3 : p.drainRate = 60)
  (h4 : p.drainTime = 800)
  (h5 : p.initialFillPercentage = 0.8)
  (h6 : volumeDrained p = p.initialFillPercentage * totalCapacity p) :
  p.depth = 10 := by
  sorry

#check pool_depth_is_10_feet

end NUMINAMATH_CALUDE_pool_depth_is_10_feet_l3378_337892


namespace NUMINAMATH_CALUDE_armband_cost_is_fifteen_l3378_337855

/-- The cost of an individual ride ticket in dollars -/
def ticket_cost : ℚ := 0.75

/-- The number of rides equivalent to the armband -/
def equivalent_rides : ℕ := 20

/-- The cost of the armband in dollars -/
def armband_cost : ℚ := ticket_cost * equivalent_rides

/-- Theorem stating that the armband costs $15.00 -/
theorem armband_cost_is_fifteen : armband_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_armband_cost_is_fifteen_l3378_337855


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l3378_337802

theorem simplify_sqrt_product : 
  Real.sqrt (3 * 5) * Real.sqrt (5^4 * 3^3) = 45 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l3378_337802


namespace NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l3378_337808

theorem max_value_x_cubed_over_y_fourth (x y : ℝ) 
  (h1 : 3 ≤ x * y^2 ∧ x * y^2 ≤ 8) 
  (h2 : 4 ≤ x^2 / y ∧ x^2 / y ≤ 9) : 
  x^3 / y^4 ≤ 27 := by
sorry

end NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l3378_337808


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l3378_337809

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 1 ∧ p.b = -2 ∧ p.c = -5 →
  let p_shifted := shift_parabola p 2 3
  p_shifted.a = 1 ∧ p_shifted.b = 2 ∧ p_shifted.c = -3 := by
  sorry

#check parabola_shift_theorem

end NUMINAMATH_CALUDE_parabola_shift_theorem_l3378_337809


namespace NUMINAMATH_CALUDE_f_composition_value_l3378_337819

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x else Real.cos x

theorem f_composition_value : f (f (-Real.pi/3)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3378_337819


namespace NUMINAMATH_CALUDE_andrew_sandwiches_l3378_337863

/-- The number of friends coming over to Andrew's game night. -/
def num_friends : ℕ := 4

/-- The number of sandwiches Andrew made for each friend. -/
def sandwiches_per_friend : ℕ := 3

/-- The total number of sandwiches Andrew made. -/
def total_sandwiches : ℕ := num_friends * sandwiches_per_friend

/-- Theorem stating that the total number of sandwiches Andrew made is 12. -/
theorem andrew_sandwiches : total_sandwiches = 12 := by
  sorry

end NUMINAMATH_CALUDE_andrew_sandwiches_l3378_337863


namespace NUMINAMATH_CALUDE_product_evaluation_l3378_337816

theorem product_evaluation : (3 + 1) * (3^3 + 1^3) * (3^9 + 1^9) = 2878848 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3378_337816


namespace NUMINAMATH_CALUDE_afternoon_rowers_calculation_l3378_337875

/-- The number of campers who went rowing in the morning -/
def morning_rowers : ℕ := 13

/-- The number of campers who went hiking in the morning -/
def morning_hikers : ℕ := 59

/-- The total number of campers who went rowing -/
def total_rowers : ℕ := 34

/-- The number of campers who went rowing in the afternoon -/
def afternoon_rowers : ℕ := total_rowers - morning_rowers

theorem afternoon_rowers_calculation :
  afternoon_rowers = 21 := by sorry

end NUMINAMATH_CALUDE_afternoon_rowers_calculation_l3378_337875


namespace NUMINAMATH_CALUDE_ginger_water_usage_l3378_337839

/-- Calculates the total cups of water used by Ginger in her garden --/
def total_water_used (hours_worked : ℕ) (cups_per_bottle : ℕ) (bottles_for_plants : ℕ) : ℕ :=
  (hours_worked * cups_per_bottle) + (bottles_for_plants * cups_per_bottle)

/-- Theorem stating that Ginger used 26 cups of water given the problem conditions --/
theorem ginger_water_usage :
  total_water_used 8 2 5 = 26 := by
  sorry

#eval total_water_used 8 2 5

end NUMINAMATH_CALUDE_ginger_water_usage_l3378_337839


namespace NUMINAMATH_CALUDE_max_missed_questions_correct_l3378_337806

/-- The number of questions in the test -/
def total_questions : ℕ := 50

/-- The minimum passing percentage -/
def passing_percentage : ℚ := 85 / 100

/-- The greatest number of questions a student can miss and still pass -/
def max_missed_questions : ℕ := 7

theorem max_missed_questions_correct :
  max_missed_questions = ⌊(1 - passing_percentage) * total_questions⌋ := by
  sorry

end NUMINAMATH_CALUDE_max_missed_questions_correct_l3378_337806


namespace NUMINAMATH_CALUDE_integer_pair_property_l3378_337850

theorem integer_pair_property (a b : ℤ) :
  (∃ d : ℤ, d ≥ 2 ∧ ∀ n : ℕ, n > 0 → (d ∣ a^n + b^n + 1)) ↔
  ((a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0)) ∨
  ((a % 3 = 1 ∧ b % 3 = 1) ∨ (a % 3 = 2 ∧ b % 3 = 2)) ∨
  ((a % 6 = 1 ∧ b % 6 = 4) ∨ (a % 6 = 4 ∧ b % 6 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_property_l3378_337850


namespace NUMINAMATH_CALUDE_amy_video_files_l3378_337801

/-- Represents the number of video files Amy had initially -/
def initial_video_files : ℕ := 36

theorem amy_video_files :
  let initial_music_files : ℕ := 26
  let deleted_files : ℕ := 48
  let remaining_files : ℕ := 14
  initial_video_files + initial_music_files - deleted_files = remaining_files :=
by sorry

end NUMINAMATH_CALUDE_amy_video_files_l3378_337801


namespace NUMINAMATH_CALUDE_apples_eaten_by_keith_l3378_337895

theorem apples_eaten_by_keith (mike_apples nancy_apples apples_left : ℝ) 
  (h1 : mike_apples = 7.0)
  (h2 : nancy_apples = 3.0)
  (h3 : apples_left = 4.0) :
  mike_apples + nancy_apples - apples_left = 6.0 := by
  sorry

end NUMINAMATH_CALUDE_apples_eaten_by_keith_l3378_337895


namespace NUMINAMATH_CALUDE_cosine_B_one_sixth_area_sqrt_three_halves_l3378_337826

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions of the problem
def triangleConditions (t : Triangle) : Prop :=
  t.b^2 = 3 * t.a * t.c

-- Part I
theorem cosine_B_one_sixth (t : Triangle) 
  (h1 : triangleConditions t) (h2 : t.a = t.b) : 
  Real.cos t.B = 1/6 := sorry

-- Part II
theorem area_sqrt_three_halves (t : Triangle) 
  (h1 : triangleConditions t) (h2 : t.B = 2 * Real.pi / 3) (h3 : t.a = Real.sqrt 2) :
  (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 3 / 2 := sorry

end NUMINAMATH_CALUDE_cosine_B_one_sixth_area_sqrt_three_halves_l3378_337826


namespace NUMINAMATH_CALUDE_problem_statement_l3378_337836

theorem problem_statement (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : a^2 * b^2 / (a^4 - 2 * b^4) = 1) :
  (a^2 - b^2) / (a^2 + b^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3378_337836


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3378_337862

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + m
  (∃ x : ℝ, f x = 0) ↔ m ≤ 4 ∧
  (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁^2 + x₂^2 + (x₁*x₂)^2 = 40 → m = -4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3378_337862


namespace NUMINAMATH_CALUDE_calculate_markup_l3378_337833

/-- Calculate the markup for an article given its purchase price, overhead percentage, and desired net profit. -/
theorem calculate_markup (purchase_price overhead_percent net_profit : ℚ) 
  (h1 : purchase_price = 48)
  (h2 : overhead_percent = 10 / 100)
  (h3 : net_profit = 12) :
  purchase_price * overhead_percent + purchase_price + net_profit - purchase_price = 168 / 10 := by
  sorry

end NUMINAMATH_CALUDE_calculate_markup_l3378_337833


namespace NUMINAMATH_CALUDE_bill_omelet_time_l3378_337853

/-- Represents the time Bill spends on preparing and cooking omelets -/
def total_time (
  pepper_chop_time : ℕ)
  (onion_chop_time : ℕ)
  (cheese_grate_time : ℕ)
  (omelet_cook_time : ℕ)
  (num_peppers : ℕ)
  (num_onions : ℕ)
  (num_omelets : ℕ) : ℕ :=
  pepper_chop_time * num_peppers +
  onion_chop_time * num_onions +
  cheese_grate_time * num_omelets +
  omelet_cook_time * num_omelets

/-- Theorem stating that Bill spends 50 minutes preparing and cooking omelets -/
theorem bill_omelet_time : 
  total_time 3 4 1 5 4 2 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bill_omelet_time_l3378_337853


namespace NUMINAMATH_CALUDE_derivative_of_cosine_at_pi_over_two_l3378_337876

theorem derivative_of_cosine_at_pi_over_two (f : ℝ → ℝ) :
  (∀ x, f x = 5 * Real.cos x) →
  HasDerivAt f (-5) (π / 2) := by
sorry

end NUMINAMATH_CALUDE_derivative_of_cosine_at_pi_over_two_l3378_337876


namespace NUMINAMATH_CALUDE_valentine_treats_l3378_337838

/-- Represents the number of heart biscuits Mrs. Heine buys for each dog -/
def heart_biscuits_per_dog : ℕ := sorry

/-- Represents the total number of items Mrs. Heine buys -/
def total_items : ℕ := 12

/-- Represents the number of dogs -/
def num_dogs : ℕ := 2

/-- Represents the number of sets of puppy boots per dog -/
def puppy_boots_per_dog : ℕ := 1

theorem valentine_treats :
  heart_biscuits_per_dog * num_dogs + puppy_boots_per_dog * num_dogs = total_items ∧
  heart_biscuits_per_dog = 4 := by sorry

end NUMINAMATH_CALUDE_valentine_treats_l3378_337838


namespace NUMINAMATH_CALUDE_person_a_silver_cards_l3378_337843

/-- Represents the number of sheets of each type of card paper -/
structure CardPapers :=
  (red : ℕ)
  (gold : ℕ)
  (silver : ℕ)

/-- Represents the exchange rates between different types of card papers -/
structure ExchangeRates :=
  (red_to_gold : ℕ × ℕ)
  (gold_to_red_and_silver : ℕ × ℕ × ℕ)

/-- Function to perform exchanges and calculate the maximum number of silver cards obtainable -/
def max_silver_obtainable (initial : CardPapers) (rates : ExchangeRates) : ℕ :=
  sorry

/-- Theorem stating that person A can obtain 7 sheets of silver card paper -/
theorem person_a_silver_cards :
  let initial := CardPapers.mk 3 3 0
  let rates := ExchangeRates.mk (5, 2) (1, 1, 1)
  max_silver_obtainable initial rates = 7 :=
sorry

end NUMINAMATH_CALUDE_person_a_silver_cards_l3378_337843


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l3378_337882

theorem fraction_product_theorem : 
  (7 : ℚ) / 4 * 8 / 14 * 16 / 24 * 32 / 48 * 28 / 7 * 15 / 9 * 50 / 25 * 21 / 35 = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l3378_337882


namespace NUMINAMATH_CALUDE_w_over_y_value_l3378_337840

theorem w_over_y_value (w x y : ℝ) 
  (h1 : w / x = 2 / 3)
  (h2 : (x + y) / y = 1.6) :
  w / y = 0.4 := by
sorry

end NUMINAMATH_CALUDE_w_over_y_value_l3378_337840


namespace NUMINAMATH_CALUDE_angle_equality_l3378_337810

-- Define the problem statement
theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (5 * π / 180) = Real.cos θ - Real.sin θ) : 
  θ = 40 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l3378_337810


namespace NUMINAMATH_CALUDE_sum_le_one_plus_product_l3378_337844

theorem sum_le_one_plus_product (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≤ 1 + a * b :=
by sorry

end NUMINAMATH_CALUDE_sum_le_one_plus_product_l3378_337844


namespace NUMINAMATH_CALUDE_adult_ticket_cost_adult_ticket_cost_is_seven_l3378_337854

theorem adult_ticket_cost (child_ticket_cost : ℝ) (total_tickets : ℕ) (total_revenue : ℝ) (child_tickets : ℕ) : ℝ :=
  let adult_tickets := total_tickets - child_tickets
  let adult_ticket_cost := (total_revenue - child_ticket_cost * child_tickets) / adult_tickets
  adult_ticket_cost

#check adult_ticket_cost 4 900 5100 400 = 7

theorem adult_ticket_cost_is_seven :
  adult_ticket_cost 4 900 5100 400 = 7 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_adult_ticket_cost_is_seven_l3378_337854


namespace NUMINAMATH_CALUDE_sector_radius_l3378_337874

/-- Given a circular sector with perimeter 83 cm and central angle 225 degrees,
    prove that the radius of the circle is 332 / (5π + 8) cm. -/
theorem sector_radius (perimeter : ℝ) (central_angle : ℝ) (radius : ℝ) : 
  perimeter = 83 →
  central_angle = 225 →
  radius = 332 / (5 * Real.pi + 8) →
  perimeter = (central_angle / 360) * 2 * Real.pi * radius + 2 * radius :=
by sorry

end NUMINAMATH_CALUDE_sector_radius_l3378_337874


namespace NUMINAMATH_CALUDE_imaginary_part_sum_l3378_337821

theorem imaginary_part_sum (z₁ z₂ : ℂ) : z₁ = (1 : ℂ) / (-2 + Complex.I) ∧ z₂ = (1 : ℂ) / (1 - 2*Complex.I) →
  Complex.im (z₁ + z₂) = (1 : ℝ) / 5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_sum_l3378_337821


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reverse_composite_l3378_337830

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n

def reverse_digits (n : ℕ) : ℕ :=
  let ones := n % 10
  let tens := n / 10
  ones * 10 + tens

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := n / 10

theorem smallest_two_digit_prime_with_reverse_composite :
  ∃ (n : ℕ), 
    is_two_digit n ∧ 
    is_prime n ∧ 
    tens_digit n = 2 ∧ 
    is_composite (reverse_digits n) ∧
    (∀ m : ℕ, is_two_digit m → is_prime m → tens_digit m = 2 → 
      is_composite (reverse_digits m) → n ≤ m) ∧
    n = 23 := by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reverse_composite_l3378_337830


namespace NUMINAMATH_CALUDE_temperature_difference_l3378_337860

theorem temperature_difference (t1 t2 k1 k2 : ℚ) :
  t1 = 5 / 9 * (k1 - 32) →
  t2 = 5 / 9 * (k2 - 32) →
  t1 = 105 →
  t2 = 80 →
  k1 - k2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l3378_337860


namespace NUMINAMATH_CALUDE_teachers_at_queen_high_school_l3378_337884

-- Define the given conditions
def total_students : ℕ := 1500
def classes_per_student : ℕ := 6
def classes_per_teacher : ℕ := 5
def students_per_class : ℕ := 35

-- Define the theorem
theorem teachers_at_queen_high_school :
  (total_students * classes_per_student / students_per_class + 4) / classes_per_teacher = 52 := by
  sorry


end NUMINAMATH_CALUDE_teachers_at_queen_high_school_l3378_337884


namespace NUMINAMATH_CALUDE_surface_area_of_T_l3378_337814

-- Define the cube
structure Cube where
  edge_length : ℝ
  vertex_A : ℝ × ℝ × ℝ

-- Define points on the cube
def L (c : Cube) : ℝ × ℝ × ℝ := (3, 0, 0)
def M (c : Cube) : ℝ × ℝ × ℝ := (0, 3, 0)
def N (c : Cube) : ℝ × ℝ × ℝ := (0, 0, 3)
def P (c : Cube) : ℝ × ℝ × ℝ := (c.edge_length, c.edge_length, c.edge_length)

-- Define the solid T
structure SolidT (c : Cube) where
  tunnel_sides : Set (ℝ × ℝ × ℝ)

-- Define the surface area of T
def surface_area (t : SolidT c) : ℝ := sorry

-- Theorem statement
theorem surface_area_of_T (c : Cube) (t : SolidT c) :
  c.edge_length = 10 →
  surface_area t = 582 + 9 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_T_l3378_337814


namespace NUMINAMATH_CALUDE_prime_arithmetic_sequence_bound_l3378_337856

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem prime_arithmetic_sequence_bound
  (a : ℕ → ℕ)
  (d : ℕ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_prime : ∀ n : ℕ, is_prime (a n))
  (h_d : d < 2000) :
  ∀ n : ℕ, n > 11 → ¬(is_prime (a n)) :=
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_sequence_bound_l3378_337856


namespace NUMINAMATH_CALUDE_division_problem_l3378_337870

theorem division_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 95 →
  divisor = 15 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  quotient = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3378_337870


namespace NUMINAMATH_CALUDE_paco_salty_cookies_l3378_337896

/-- Prove that Paco initially had 56 salty cookies -/
theorem paco_salty_cookies 
  (initial_sweet : ℕ) 
  (eaten_sweet : ℕ) 
  (eaten_salty : ℕ) 
  (remaining_sweet : ℕ) 
  (h1 : initial_sweet = 34)
  (h2 : eaten_sweet = 15)
  (h3 : eaten_salty = 56)
  (h4 : remaining_sweet = 19)
  (h5 : initial_sweet = eaten_sweet + remaining_sweet) :
  eaten_salty = 56 := by
  sorry

end NUMINAMATH_CALUDE_paco_salty_cookies_l3378_337896


namespace NUMINAMATH_CALUDE_game_cost_l3378_337834

/-- 
Given:
- Will made 104 dollars mowing lawns
- He spent 41 dollars on new mower blades
- He bought 7 games with the remaining money
Prove that each game cost 9 dollars
-/
theorem game_cost (total_earned : ℕ) (spent_on_blades : ℕ) (num_games : ℕ) :
  total_earned = 104 →
  spent_on_blades = 41 →
  num_games = 7 →
  (total_earned - spent_on_blades) / num_games = 9 :=
by sorry

end NUMINAMATH_CALUDE_game_cost_l3378_337834


namespace NUMINAMATH_CALUDE_exercise_book_cost_l3378_337805

/-- Proves that the total cost of buying 'a' exercise books at 0.8 yuan each is 0.8a yuan -/
theorem exercise_book_cost (a : ℝ) : 
  let cost_per_book : ℝ := 0.8
  let num_books : ℝ := a
  let total_cost : ℝ := cost_per_book * num_books
  total_cost = 0.8 * a := by sorry

end NUMINAMATH_CALUDE_exercise_book_cost_l3378_337805


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3378_337888

/-- Theorem: For a rectangle with length L and width W, if L/W = 5/2 and L * W = 4000, 
    then the perimeter 2L + 2W = 280. -/
theorem rectangle_perimeter (L W : ℝ) 
    (h1 : L / W = 5 / 2) 
    (h2 : L * W = 4000) : 
  2 * L + 2 * W = 280 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3378_337888


namespace NUMINAMATH_CALUDE_shirts_sold_l3378_337886

def commission_rate : ℚ := 15 / 100
def suit_price : ℚ := 700
def suit_quantity : ℕ := 2
def shirt_price : ℚ := 50
def loafer_price : ℚ := 150
def loafer_quantity : ℕ := 2
def total_commission : ℚ := 300

theorem shirts_sold (shirt_quantity : ℕ) : 
  commission_rate * (suit_price * suit_quantity + shirt_price * shirt_quantity + loafer_price * loafer_quantity) = total_commission →
  shirt_quantity = 6 := by sorry

end NUMINAMATH_CALUDE_shirts_sold_l3378_337886


namespace NUMINAMATH_CALUDE_min_omega_for_even_shifted_sine_l3378_337820

/-- Given a function g and a real number ω, this theorem states that
    if g is defined as g(x) = sin(ω(x - π/3) + π/6),
    ω is positive, and g is an even function,
    then the minimum value of ω is 2. -/
theorem min_omega_for_even_shifted_sine (g : ℝ → ℝ) (ω : ℝ) :
  (∀ x, g x = Real.sin (ω * (x - Real.pi / 3) + Real.pi / 6)) →
  ω > 0 →
  (∀ x, g x = g (-x)) →
  ω ≥ 2 ∧ ∃ ω₀, ω₀ = 2 ∧ 
    (∀ x, Real.sin (ω₀ * (x - Real.pi / 3) + Real.pi / 6) = 
          Real.sin (ω₀ * ((-x) - Real.pi / 3) + Real.pi / 6)) :=
by sorry

end NUMINAMATH_CALUDE_min_omega_for_even_shifted_sine_l3378_337820


namespace NUMINAMATH_CALUDE_gcd_lcm_power_equation_l3378_337852

/-- Given positive integers m and n, if m^(gcd m n) = n^(lcm m n), then m = 1 and n = 1 -/
theorem gcd_lcm_power_equation (m n : ℕ+) :
  m ^ (Nat.gcd m.val n.val) = n ^ (Nat.lcm m.val n.val) → m = 1 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_power_equation_l3378_337852


namespace NUMINAMATH_CALUDE_two_colors_sufficient_l3378_337817

/-- Represents a key on the ring -/
structure Key where
  position : Fin 8
  color : Bool

/-- Represents the ring of keys -/
def KeyRing : Type := Fin 8 → Key

/-- A coloring scheme is valid if it allows each key to be uniquely identified -/
def is_valid_coloring (ring : KeyRing) : Prop :=
  ∀ (i j : Fin 8), i ≠ j → 
    ∃ (k : ℕ), (ring ((i + k) % 8)).color ≠ (ring ((j + k) % 8)).color

/-- There exists a valid coloring scheme using only two colors -/
theorem two_colors_sufficient : 
  ∃ (ring : KeyRing), (∀ k, (ring k).color = true ∨ (ring k).color = false) ∧ is_valid_coloring ring := by
  sorry


end NUMINAMATH_CALUDE_two_colors_sufficient_l3378_337817


namespace NUMINAMATH_CALUDE_building_block_width_l3378_337897

/-- Given a box and building blocks with specified dimensions, prove that the width of the building block is 2 inches. -/
theorem building_block_width (box_height box_width box_length : ℕ)
  (block_height block_length : ℕ) (num_blocks : ℕ) :
  box_height = 8 →
  box_width = 10 →
  box_length = 12 →
  block_height = 3 →
  block_length = 4 →
  num_blocks = 40 →
  (box_height * box_width * box_length) / num_blocks = block_height * 2 * block_length :=
by sorry

end NUMINAMATH_CALUDE_building_block_width_l3378_337897


namespace NUMINAMATH_CALUDE_gear_rotation_problem_l3378_337847

/-- 
Given two gears p and q rotating at constant speeds:
- q makes 40 revolutions per minute
- After 4 seconds, q has made exactly 2 more revolutions than p
Prove that p makes 10 revolutions per minute
-/
theorem gear_rotation_problem (p q : ℝ) 
  (hq : q = 40) -- q makes 40 revolutions per minute
  (h_diff : q * 4 / 60 = p * 4 / 60 + 2) -- After 4 seconds, q has made 2 more revolutions than p
  : p = 10 := by sorry

end NUMINAMATH_CALUDE_gear_rotation_problem_l3378_337847


namespace NUMINAMATH_CALUDE_total_pencils_l3378_337823

/-- Given that each child has 2 pencils and there are 9 children, prove that the total number of pencils is 18. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) (h1 : pencils_per_child = 2) (h2 : num_children = 9) :
  pencils_per_child * num_children = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l3378_337823


namespace NUMINAMATH_CALUDE_unripe_oranges_calculation_l3378_337829

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges_per_day : ℕ := 28

/-- The number of days of harvest -/
def harvest_days : ℕ := 26

/-- The total number of sacks of oranges after the harvest period -/
def total_oranges : ℕ := 2080

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges_per_day : ℕ := (total_oranges - ripe_oranges_per_day * harvest_days) / harvest_days

theorem unripe_oranges_calculation :
  unripe_oranges_per_day = 52 :=
by sorry

end NUMINAMATH_CALUDE_unripe_oranges_calculation_l3378_337829


namespace NUMINAMATH_CALUDE_number_of_children_l3378_337849

def total_cupcakes : ℕ := 96
def cupcakes_per_child : ℕ := 12

theorem number_of_children : 
  total_cupcakes / cupcakes_per_child = 8 := by sorry

end NUMINAMATH_CALUDE_number_of_children_l3378_337849


namespace NUMINAMATH_CALUDE_no_representation_l3378_337893

theorem no_representation (a b c : ℕ+) 
  (h_gcd_ab : Nat.gcd a b = 1)
  (h_gcd_bc : Nat.gcd b c = 1)
  (h_gcd_ca : Nat.gcd c a = 1) :
  ¬ ∃ (x y z : ℕ), 2 * a * b * c - a * b - b * c - c * a = b * c * x + c * a * y + a * b * z :=
by sorry

end NUMINAMATH_CALUDE_no_representation_l3378_337893


namespace NUMINAMATH_CALUDE_circle_tangent_sum_l3378_337879

def circle_radius_sum : ℝ := 14

theorem circle_tangent_sum (C : ℝ × ℝ) (r : ℝ) :
  (C.1 = r ∧ C.2 = r) →  -- Circle center C is at (r, r)
  ((C.1 - 5)^2 + C.2^2 = (r + 2)^2) →  -- External tangency condition
  (∃ (r1 r2 : ℝ), r1 + r2 = circle_radius_sum ∧ 
    ((C.1 = r1 ∧ C.2 = r1) ∨ (C.1 = r2 ∧ C.2 = r2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_sum_l3378_337879


namespace NUMINAMATH_CALUDE_circle_point_inequality_l3378_337891

theorem circle_point_inequality (m n c : ℝ) : 
  (∀ m n, m^2 + (n - 2)^2 = 1 → m + n + c ≥ 1) → c ≥ Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_point_inequality_l3378_337891


namespace NUMINAMATH_CALUDE_function_composition_implies_sum_l3378_337883

/-- Given two functions f and g, where f(x) = ax + b and g(x) = 3x - 6,
    and the condition that g(f(x)) = 4x + 3 for all x,
    prove that a + b = 13/3 -/
theorem function_composition_implies_sum (a b : ℝ) :
  (∀ x, 3 * (a * x + b) - 6 = 4 * x + 3) →
  a + b = 13 / 3 := by
sorry

end NUMINAMATH_CALUDE_function_composition_implies_sum_l3378_337883


namespace NUMINAMATH_CALUDE_probability_red_or_blue_specific_l3378_337867

/-- The probability of drawing either a red or blue marble from a bag -/
def probability_red_or_blue (red blue green yellow : ℕ) : ℚ :=
  (red + blue : ℚ) / (red + blue + green + yellow : ℚ)

/-- Theorem: The probability of drawing either a red or blue marble from a bag
    containing 5 red, 3 blue, 4 green, and 6 yellow marbles is 4/9 -/
theorem probability_red_or_blue_specific : probability_red_or_blue 5 3 4 6 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_blue_specific_l3378_337867


namespace NUMINAMATH_CALUDE_g_of_3_equals_10_l3378_337841

def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem g_of_3_equals_10 : g 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_10_l3378_337841


namespace NUMINAMATH_CALUDE_inequality_preserved_subtraction_l3378_337803

theorem inequality_preserved_subtraction (a b : ℝ) (h : a < b) : a - 1 < b - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preserved_subtraction_l3378_337803


namespace NUMINAMATH_CALUDE_triangle_inequality_l3378_337827

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a < b + c) (hbc : b < a + c) (hca : c < a + b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3378_337827


namespace NUMINAMATH_CALUDE_sum_through_base3_l3378_337848

/-- Converts a natural number from base 10 to base 3 --/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a number from base 3 (represented as a list of digits) to base 10 --/
def fromBase3 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base 3 (represented as lists of digits) --/
def addBase3 (a b : List ℕ) : List ℕ :=
  sorry

/-- Theorem stating that the sum of 10 and 23 in base 10 is equal to 33
    when performed through base 3 conversion and addition --/
theorem sum_through_base3 :
  fromBase3 (addBase3 (toBase3 10) (toBase3 23)) = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_through_base3_l3378_337848


namespace NUMINAMATH_CALUDE_simplify_fraction_l3378_337845

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) :
  (x^2 / (x - 2)) - (4 / (x - 2)) = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3378_337845


namespace NUMINAMATH_CALUDE_fewer_vip_tickets_l3378_337807

/-- Represents the number of tickets sold in a snooker tournament -/
structure TicketSales where
  vip : ℕ
  general : ℕ

/-- The ticket prices and sales data for the snooker tournament -/
def snookerTournament : TicketSales → Prop := fun ts =>
  ts.vip + ts.general = 320 ∧
  40 * ts.vip + 10 * ts.general = 7500

theorem fewer_vip_tickets (ts : TicketSales) 
  (h : snookerTournament ts) : ts.general - ts.vip = 34 := by
  sorry

end NUMINAMATH_CALUDE_fewer_vip_tickets_l3378_337807


namespace NUMINAMATH_CALUDE_least_bananas_l3378_337898

def banana_distribution (total : ℕ) : Prop :=
  ∃ (b₁ b₂ b₃ b₄ : ℕ),
    -- Total number of bananas
    b₁ + b₂ + b₃ + b₄ = total ∧
    -- First monkey's distribution
    ∃ (x₁ y₁ z₁ w₁ : ℕ),
      2 * b₁ = 3 * x₁ ∧
      b₁ - x₁ = 3 * y₁ ∧ y₁ = z₁ ∧ y₁ = w₁ ∧
    -- Second monkey's distribution
    ∃ (x₂ y₂ z₂ w₂ : ℕ),
      b₂ = 3 * y₂ ∧
      2 * b₂ = 3 * (x₂ + z₂ + w₂) ∧ x₂ = z₂ ∧ x₂ = w₂ ∧
    -- Third monkey's distribution
    ∃ (x₃ y₃ z₃ w₃ : ℕ),
      b₃ = 4 * z₃ ∧
      3 * b₃ = 4 * (x₃ + y₃ + w₃) ∧ x₃ = y₃ ∧ x₃ = w₃ ∧
    -- Fourth monkey's distribution
    ∃ (x₄ y₄ z₄ w₄ : ℕ),
      b₄ = 6 * w₄ ∧
      5 * b₄ = 6 * (x₄ + y₄ + z₄) ∧ x₄ = y₄ ∧ x₄ = z₄ ∧
    -- Final distribution ratio
    ∃ (k : ℕ),
      (2 * x₁ + y₂ + z₃ + w₄) = 4 * k ∧
      (y₁ + 2 * y₂ + z₃ + w₄) = 3 * k ∧
      (z₁ + y₂ + 2 * z₃ + w₄) = 2 * k ∧
      (w₁ + y₂ + z₃ + 2 * w₄) = k

theorem least_bananas : 
  ∀ n : ℕ, n < 1128 → ¬(banana_distribution n) ∧ banana_distribution 1128 := by
  sorry

end NUMINAMATH_CALUDE_least_bananas_l3378_337898


namespace NUMINAMATH_CALUDE_right_triangle_set_l3378_337815

theorem right_triangle_set : ∃! (a b c : ℕ), 
  ((a = 7 ∧ b = 24 ∧ c = 25) ∨ 
   (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
   (a = 4 ∧ b = 5 ∧ c = 6) ∨ 
   (a = 8 ∧ b = 15 ∧ c = 18)) ∧ 
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_set_l3378_337815


namespace NUMINAMATH_CALUDE_sums_are_equal_l3378_337811

def S₁ : ℕ := 1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999

def S₂ : ℕ := 9 + 98 + 987 + 9876 + 98765 + 987654 + 9876543 + 98765432 + 987654321

theorem sums_are_equal : S₁ = S₂ := by
  sorry

end NUMINAMATH_CALUDE_sums_are_equal_l3378_337811


namespace NUMINAMATH_CALUDE_min_value_theorem_l3378_337890

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 1) :
  y / x + 4 / y ≥ 8 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ y / x + 4 / y = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3378_337890


namespace NUMINAMATH_CALUDE_no_statements_imply_negation_l3378_337864

theorem no_statements_imply_negation (p q : Prop) : 
  ¬((p ∨ q) → ¬(p ∨ q)) ∧
  ¬((p ∨ ¬q) → ¬(p ∨ q)) ∧
  ¬((¬p ∨ q) → ¬(p ∨ q)) ∧
  ¬((¬p ∧ q) → ¬(p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_no_statements_imply_negation_l3378_337864


namespace NUMINAMATH_CALUDE_variance_of_transformed_data_l3378_337800

def dataset := Fin 10 → ℝ

def variance (X : dataset) : ℝ := sorry

def transform (X : dataset) : dataset := λ i => 2 * X i + 3

theorem variance_of_transformed_data (X : dataset) 
  (h : variance X = 3) : variance (transform X) = 12 := by sorry

end NUMINAMATH_CALUDE_variance_of_transformed_data_l3378_337800


namespace NUMINAMATH_CALUDE_prob_two_dice_shows_two_l3378_337813

def num_sides : ℕ := 8

def prob_at_least_one_two (n : ℕ) : ℚ :=
  1 - ((n - 1) / n)^2

theorem prob_two_dice_shows_two :
  prob_at_least_one_two num_sides = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_dice_shows_two_l3378_337813


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3378_337868

/-- The position function of a particle -/
def s (t : ℝ) : ℝ := 3 * t^2 + t

/-- The velocity function of a particle -/
def v (t : ℝ) : ℝ := 6 * t + 1

theorem instantaneous_velocity_at_3_seconds : v 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3378_337868


namespace NUMINAMATH_CALUDE_average_rate_of_change_l3378_337899

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the theorem
theorem average_rate_of_change (Δx : ℝ) :
  (f (1 + Δx) - f 1) / Δx = 2 + Δx :=
by sorry

end NUMINAMATH_CALUDE_average_rate_of_change_l3378_337899


namespace NUMINAMATH_CALUDE_no_natural_square_diff_2014_l3378_337837

theorem no_natural_square_diff_2014 : ¬∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_square_diff_2014_l3378_337837


namespace NUMINAMATH_CALUDE_corresponding_angles_not_always_equal_l3378_337869

-- Define the concept of corresponding angles
def corresponding_angles (α β : ℝ) : Prop := sorry

-- Theorem stating that the proposition "corresponding angles are equal" is false
theorem corresponding_angles_not_always_equal :
  ¬ ∀ α β : ℝ, corresponding_angles α β → α = β :=
sorry

end NUMINAMATH_CALUDE_corresponding_angles_not_always_equal_l3378_337869


namespace NUMINAMATH_CALUDE_square_area_ratio_l3378_337877

/-- Given three squares with the specified relationships, prove that the ratio of the areas of the first and second squares is 1/2. -/
theorem square_area_ratio (s₃ : ℝ) (h₃ : s₃ > 0) : 
  let s₁ := s₃ * Real.sqrt 2
  let s₂ := s₁ * Real.sqrt 2
  (s₁^2) / (s₂^2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3378_337877


namespace NUMINAMATH_CALUDE_hospital_transfer_l3378_337887

theorem hospital_transfer (x : ℝ) (x_pos : x > 0) : 
  let wing_a := x
  let wing_b := 2 * x
  let wing_c := 3 * x
  let occupied_a := (1/3) * wing_a
  let occupied_b := (1/2) * wing_b
  let occupied_c := (1/4) * wing_c
  let max_capacity_b := (3/4) * wing_b
  let max_capacity_c := (5/6) * wing_c
  occupied_a + occupied_b ≤ max_capacity_b →
  (occupied_a + occupied_b) / wing_b = 2/3 ∧ occupied_c / wing_c = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_hospital_transfer_l3378_337887
