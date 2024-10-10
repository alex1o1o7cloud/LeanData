import Mathlib

namespace coprime_lcm_product_l3431_343189

theorem coprime_lcm_product {a b : ℕ+} (h_coprime : Nat.Coprime a b) 
  (h_lcm_eq_prod : Nat.lcm a b = a * b) : 
  ∃ (k : ℕ+), a * b = k := by sorry

end coprime_lcm_product_l3431_343189


namespace couch_money_calculation_l3431_343171

theorem couch_money_calculation (quarters : ℕ) (pennies : ℕ) 
  (quarter_value : ℚ) (penny_value : ℚ) :
  quarters = 12 →
  pennies = 7 →
  quarter_value = 25 / 100 →
  penny_value = 1 / 100 →
  quarters * quarter_value + pennies * penny_value = 307 / 100 := by
sorry

end couch_money_calculation_l3431_343171


namespace square_area_from_diagonal_l3431_343120

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 144 := by sorry

end square_area_from_diagonal_l3431_343120


namespace puppies_per_cage_l3431_343157

theorem puppies_per_cage 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : initial_puppies = 102) 
  (h2 : sold_puppies = 21) 
  (h3 : num_cages = 9) 
  : (initial_puppies - sold_puppies) / num_cages = 9 := by
  sorry

end puppies_per_cage_l3431_343157


namespace gcd_lcm_product_24_45_l3431_343152

theorem gcd_lcm_product_24_45 : Nat.gcd 24 45 * Nat.lcm 24 45 = 1080 := by
  sorry

end gcd_lcm_product_24_45_l3431_343152


namespace two_economic_reasons_exist_l3431_343141

/-- Represents a European country --/
structure EuropeanCountry where
  name : String

/-- Represents an economic reason for offering free or low-cost education to foreign citizens --/
structure EconomicReason where
  description : String

/-- Represents a government policy --/
structure GovernmentPolicy where
  description : String

/-- Predicate to check if a policy offers free or low-cost education to foreign citizens --/
def is_free_education_policy (policy : GovernmentPolicy) : Prop :=
  policy.description = "Offer free or low-cost education to foreign citizens"

/-- Predicate to check if a reason is valid for a given country and policy --/
def is_valid_reason (country : EuropeanCountry) (policy : GovernmentPolicy) (reason : EconomicReason) : Prop :=
  is_free_education_policy policy ∧ 
  (reason.description = "International Agreements" ∨ reason.description = "Addressing Demographic Changes")

/-- Theorem stating that there exist at least two distinct economic reasons for the policy --/
theorem two_economic_reasons_exist (country : EuropeanCountry) (policy : GovernmentPolicy) :
  is_free_education_policy policy →
  ∃ (reason1 reason2 : EconomicReason), 
    reason1 ≠ reason2 ∧ 
    is_valid_reason country policy reason1 ∧ 
    is_valid_reason country policy reason2 :=
sorry

end two_economic_reasons_exist_l3431_343141


namespace simple_interest_rate_calculation_l3431_343193

/-- Calculates the simple interest rate given principal, final amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  (amount - principal) * 100 / (principal * time)

theorem simple_interest_rate_calculation 
  (principal amount : ℚ) (time : ℕ) 
  (h_principal : principal = 650)
  (h_amount : amount = 950)
  (h_time : time = 5) :
  simple_interest_rate principal amount time = (950 - 650) * 100 / (650 * 5) :=
by
  sorry

#eval simple_interest_rate 650 950 5

end simple_interest_rate_calculation_l3431_343193


namespace apple_distribution_l3431_343168

theorem apple_distribution (total_apples : ℕ) (num_people : ℕ) (apples_per_person : ℕ) :
  total_apples = 15 →
  num_people = 3 →
  apples_per_person * num_people ≤ total_apples →
  apples_per_person = total_apples / num_people →
  apples_per_person = 5 :=
by
  sorry

end apple_distribution_l3431_343168


namespace cos_270_degrees_l3431_343192

-- Define cosine function on the unit circle
noncomputable def cosine (angle : Real) : Real :=
  (Complex.exp (Complex.I * angle)).re

-- State the theorem
theorem cos_270_degrees : cosine (3 * Real.pi / 2) = 0 := by
  sorry

end cos_270_degrees_l3431_343192


namespace unique_number_satisfying_means_l3431_343131

theorem unique_number_satisfying_means : ∃! X : ℝ,
  (28 + X + 70 + 88 + 104) / 5 = 67 ∧
  (50 + 62 + 97 + 124 + X) / 5 = 75.6 := by
  sorry

end unique_number_satisfying_means_l3431_343131


namespace inequality_solution_set_l3431_343119

theorem inequality_solution_set (x : ℝ) : (x - 1) / (x + 2) > 0 ↔ x < -2 ∨ x > 1 := by
  sorry

end inequality_solution_set_l3431_343119


namespace sum_of_coefficients_l3431_343130

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₀ + a₂ + a₄ = -121 := by
sorry

end sum_of_coefficients_l3431_343130


namespace thermostat_adjustment_l3431_343129

theorem thermostat_adjustment (x : ℝ) : 
  let initial_temp := 40
  let jerry_temp := 2 * initial_temp
  let dad_temp := jerry_temp - x
  let mom_temp := dad_temp * 0.7
  let sister_temp := mom_temp + 24
  sister_temp = 59 → x = 10 := by
  sorry

end thermostat_adjustment_l3431_343129


namespace pizzas_ordered_proof_l3431_343165

/-- The number of students in the class -/
def num_students : ℕ := 32

/-- The number of cheese pieces each student gets -/
def cheese_per_student : ℕ := 2

/-- The number of onion pieces each student gets -/
def onion_per_student : ℕ := 1

/-- The number of slices in a large pizza -/
def slices_per_pizza : ℕ := 18

/-- The number of leftover cheese pieces -/
def leftover_cheese : ℕ := 8

/-- The number of leftover onion pieces -/
def leftover_onion : ℕ := 4

/-- The minimum number of pizzas ordered -/
def min_pizzas_ordered : ℕ := 5

theorem pizzas_ordered_proof :
  let total_cheese := num_students * cheese_per_student + leftover_cheese
  let total_onion := num_students * onion_per_student + leftover_onion
  let total_slices := total_cheese + total_onion
  (total_slices + slices_per_pizza - 1) / slices_per_pizza = min_pizzas_ordered :=
by sorry

end pizzas_ordered_proof_l3431_343165


namespace fly_ceiling_distance_l3431_343186

-- Define the room and fly position
def room_fly_distance (wall1_distance wall2_distance point_p_distance : ℝ) : Prop :=
  ∃ (ceiling_distance : ℝ),
    wall1_distance = 2 ∧
    wall2_distance = 7 ∧
    point_p_distance = 10 ∧
    ceiling_distance^2 + wall1_distance^2 + wall2_distance^2 = point_p_distance^2

-- Theorem statement
theorem fly_ceiling_distance :
  ∀ (wall1_distance wall2_distance point_p_distance ceiling_distance : ℝ),
    room_fly_distance wall1_distance wall2_distance point_p_distance →
    ceiling_distance = Real.sqrt 47 := by
  sorry

end fly_ceiling_distance_l3431_343186


namespace sandys_age_l3431_343139

theorem sandys_age (sandy_age molly_age : ℕ) 
  (age_difference : molly_age = sandy_age + 14)
  (age_ratio : sandy_age * 9 = molly_age * 7) :
  sandy_age = 49 := by
sorry

end sandys_age_l3431_343139


namespace systematic_sample_41_l3431_343150

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ
  sample_size : ℕ
  interval : ℕ
  first_selected : ℕ

/-- Checks if a number is in the systematic sample -/
def in_sample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.first_selected + k * s.interval ∧ k < s.sample_size

theorem systematic_sample_41 :
  ∀ s : SystematicSample,
    s.total = 60 →
    s.sample_size = 5 →
    s.interval = s.total / s.sample_size →
    in_sample s 17 →
    in_sample s 41 := by
  sorry

end systematic_sample_41_l3431_343150


namespace probability_product_one_five_dice_l3431_343170

def standard_die := Finset.range 6

def probability_of_one (n : ℕ) : ℚ :=
  (1 : ℚ) / 6

theorem probability_product_one_five_dice :
  (probability_of_one 5)^5 = 1 / 7776 := by
  sorry

end probability_product_one_five_dice_l3431_343170


namespace james_joe_age_ratio_l3431_343159

theorem james_joe_age_ratio : 
  ∀ (joe_age james_age : ℕ),
    joe_age = 22 →
    joe_age = james_age + 10 →
    ∃ (k : ℕ), 2 * (joe_age + 8) = k * (james_age + 8) →
    (james_age + 8) / (joe_age + 8) = 2 / 3 := by
  sorry

end james_joe_age_ratio_l3431_343159


namespace units_digit_sum_of_powers_l3431_343121

theorem units_digit_sum_of_powers : (24^4 + 42^4 + 24^2 + 42^2) % 10 = 2 := by
  sorry

end units_digit_sum_of_powers_l3431_343121


namespace range_of_m_l3431_343110

-- Define the propositions r(x) and s(x)
def r (x m : ℝ) : Prop := Real.sin x + Real.cos x > m
def s (x m : ℝ) : Prop := x^2 + m*x + 1 > 0

-- Define the theorem
theorem range_of_m :
  (∀ x : ℝ, (r x m ∧ ¬(s x m)) ∨ (¬(r x m) ∧ s x m)) →
  (m ≤ -2 ∨ (-Real.sqrt 2 ≤ m ∧ m < 2)) :=
sorry

end range_of_m_l3431_343110


namespace percentage_boys_school_A_l3431_343116

/-- Proves that the percentage of boys from school A in a camp is 20% -/
theorem percentage_boys_school_A (total_boys : ℕ) (boys_A_not_science : ℕ) 
  (percent_A_science : ℚ) :
  total_boys = 250 →
  boys_A_not_science = 35 →
  percent_A_science = 30 / 100 →
  (boys_A_not_science : ℚ) / ((1 - percent_A_science) * total_boys) = 20 / 100 := by
  sorry

#check percentage_boys_school_A

end percentage_boys_school_A_l3431_343116


namespace seed_germination_problem_l3431_343164

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  (0.25 * x + 80) / (x + 200) = 0.31 →
  x = 300 := by
sorry

end seed_germination_problem_l3431_343164


namespace min_value_problem_l3431_343180

theorem min_value_problem (a b c : ℝ) 
  (eq1 : 3 * a + 2 * b + c = 5)
  (eq2 : 2 * a + b - 3 * c = 1)
  (nonneg_a : a ≥ 0)
  (nonneg_b : b ≥ 0)
  (nonneg_c : c ≥ 0) :
  (∀ a' b' c' : ℝ, 
    3 * a' + 2 * b' + c' = 5 → 
    2 * a' + b' - 3 * c' = 1 → 
    a' ≥ 0 → b' ≥ 0 → c' ≥ 0 → 
    3 * a + b - 7 * c ≤ 3 * a' + b' - 7 * c') ∧
  (3 * a + b - 7 * c = -5/7) :=
sorry

end min_value_problem_l3431_343180


namespace shifted_line_y_intercept_l3431_343115

/-- A line in the form y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

/-- Shift a line vertically -/
def shiftLine (l : Line) (shift : ℝ) : Line :=
  { m := l.m, c := l.c + shift }

/-- The y-intercept of a line -/
def yIntercept (l : Line) : ℝ := l.c

theorem shifted_line_y_intercept :
  let original_line : Line := { m := 1, c := -1 }
  let shifted_line := shiftLine original_line 2
  yIntercept shifted_line = 1 := by sorry

end shifted_line_y_intercept_l3431_343115


namespace blocks_per_box_l3431_343145

theorem blocks_per_box (total_blocks : ℕ) (num_boxes : ℕ) (h1 : total_blocks = 12) (h2 : num_boxes = 2) :
  total_blocks / num_boxes = 6 := by
  sorry

end blocks_per_box_l3431_343145


namespace parabola_minimum_value_l3431_343118

theorem parabola_minimum_value (m : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + 2*m*x + m + 2 ≥ -3) ∧ 
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + 2*m*x + m + 2 = -3) →
  m = 3 ∨ m = (1 - Real.sqrt 21) / 2 :=
by sorry

end parabola_minimum_value_l3431_343118


namespace normal_vector_for_line_with_angle_pi_div_3_l3431_343174

/-- A line in 2D space -/
structure Line2D where
  angle : Real

/-- A vector in 2D space -/
structure Vector2D where
  x : Real
  y : Real

/-- Check if a vector is normal to a line -/
def isNormalVector (l : Line2D) (v : Vector2D) : Prop :=
  v.x * Real.cos l.angle + v.y * Real.sin l.angle = 0

/-- Theorem: (1, -√3/3) is a normal vector of a line with inclination angle π/3 -/
theorem normal_vector_for_line_with_angle_pi_div_3 :
  let l : Line2D := { angle := π / 3 }
  let v : Vector2D := { x := 1, y := -Real.sqrt 3 / 3 }
  isNormalVector l v := by
  sorry

end normal_vector_for_line_with_angle_pi_div_3_l3431_343174


namespace gcd_108_450_l3431_343190

theorem gcd_108_450 : Nat.gcd 108 450 = 18 := by
  sorry

end gcd_108_450_l3431_343190


namespace f_has_unique_root_l3431_343124

noncomputable def f (x : ℝ) := Real.exp x + x - 2

theorem f_has_unique_root :
  ∃! x, f x = 0 :=
sorry

end f_has_unique_root_l3431_343124


namespace alyssa_pullups_l3431_343146

/-- Represents the number of exercises done by a person -/
structure ExerciseCount where
  pushups : ℕ
  crunches : ℕ
  pullups : ℕ

/-- Zachary's exercise count -/
def zachary : ExerciseCount := ⟨44, 17, 23⟩

/-- David's exercise count relative to Zachary's -/
def david : ExerciseCount := ⟨zachary.pushups + 29, zachary.crunches - 13, zachary.pullups + 10⟩

/-- Alyssa's exercise count relative to Zachary's -/
def alyssa : ExerciseCount := ⟨zachary.pushups * 2, zachary.crunches / 2, zachary.pullups - 8⟩

theorem alyssa_pullups : alyssa.pullups = 15 := by
  sorry

end alyssa_pullups_l3431_343146


namespace tangent_line_determines_function_l3431_343105

/-- Given a function f(x) = (mx-6)/(x^2+n) with a tangent line at P(-1, f(-1))
    with equation x + 2y + 5 = 0, prove that f(x) = (2x-6)/(x^2+3) -/
theorem tangent_line_determines_function (m n : ℝ) :
  let f : ℝ → ℝ := λ x => (m * x - 6) / (x^2 + n)
  let f' : ℝ → ℝ := λ x => ((m * (x^2 + n) - (2 * x * (m * x - 6))) / (x^2 + n)^2)
  (f' (-1) = -1/2) →
  (f (-1) = -2) →
  (∀ x, f x = (2 * x - 6) / (x^2 + 3)) :=
by
  sorry

end tangent_line_determines_function_l3431_343105


namespace evaluate_expression_l3431_343117

theorem evaluate_expression : (900^2 : ℝ) / (153^2 - 147^2) = 450 := by
  sorry

end evaluate_expression_l3431_343117


namespace cookies_theorem_l3431_343140

/-- The number of cookies each guest had -/
def cookies_per_guest : ℕ := 2

/-- The number of guests -/
def number_of_guests : ℕ := 5

/-- The total number of cookies prepared -/
def total_cookies : ℕ := cookies_per_guest * number_of_guests

theorem cookies_theorem : total_cookies = 10 := by
  sorry

end cookies_theorem_l3431_343140


namespace date_books_ordered_l3431_343179

theorem date_books_ordered (calendar_cost date_book_cost : ℚ) 
  (total_items : ℕ) (total_spent : ℚ) :
  calendar_cost = 3/4 →
  date_book_cost = 1/2 →
  total_items = 500 →
  total_spent = 300 →
  ∃ (calendars date_books : ℕ),
    calendars + date_books = total_items ∧
    calendar_cost * calendars + date_book_cost * date_books = total_spent ∧
    date_books = 300 := by
sorry

end date_books_ordered_l3431_343179


namespace a_range_l3431_343175

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

def inequality_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - a * x + 1 > 0

theorem a_range (a : ℝ) (h_a : a > 0) :
  (¬(is_monotonically_increasing (λ x => a^x)) ∨
   ¬(inequality_holds a)) ∧
  (is_monotonically_increasing (λ x => a^x) ∨
   inequality_holds a) →
  a ∈ Set.Ioc 0 1 ∪ Set.Ici 4 :=
sorry

end a_range_l3431_343175


namespace toothpick_grid_15x12_l3431_343177

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid where
  height : Nat
  width : Nat

/-- Calculates the total number of toothpicks in the grid -/
def totalToothpicks (grid : ToothpickGrid) : Nat :=
  (grid.height + 1) * grid.width + (grid.width + 1) * grid.height

/-- Calculates the number of toothpicks in the boundary of the grid -/
def boundaryToothpicks (grid : ToothpickGrid) : Nat :=
  2 * (grid.height + grid.width)

/-- Theorem stating the properties of a 15x12 toothpick grid -/
theorem toothpick_grid_15x12 :
  let grid : ToothpickGrid := ⟨15, 12⟩
  totalToothpicks grid = 387 ∧ boundaryToothpicks grid = 54 := by
  sorry


end toothpick_grid_15x12_l3431_343177


namespace harmony_sum_l3431_343135

def alphabet_value (n : ℕ) : ℤ :=
  match n % 13 with
  | 0 => -3
  | 1 => -2
  | 2 => -1
  | 3 => 0
  | 4 => 1
  | 5 => 2
  | 6 => 3
  | 7 => 2
  | 8 => 1
  | 9 => 0
  | 10 => -1
  | 11 => -2
  | 12 => -3
  | _ => 0  -- This case should never occur due to the modulo operation

theorem harmony_sum : 
  alphabet_value 8 + alphabet_value 1 + alphabet_value 18 + 
  alphabet_value 13 + alphabet_value 15 + alphabet_value 14 + 
  alphabet_value 25 = -7 := by
sorry

end harmony_sum_l3431_343135


namespace tourist_distribution_count_l3431_343155

/-- The number of tour guides -/
def num_guides : ℕ := 3

/-- The number of tourists -/
def num_tourists : ℕ := 8

/-- The number of ways to distribute tourists among guides -/
def distribute_tourists : ℕ := 3^8

/-- The number of ways where at least one guide has no tourists -/
def at_least_one_empty : ℕ := 3 * 2^8

/-- The number of ways where exactly two guides have no tourists -/
def two_empty : ℕ := 3

/-- The number of valid distributions where each guide has at least one tourist -/
def valid_distributions : ℕ := distribute_tourists - at_least_one_empty + two_empty

theorem tourist_distribution_count :
  valid_distributions = 5796 :=
sorry

end tourist_distribution_count_l3431_343155


namespace intersection_implies_sum_l3431_343166

-- Define the sets A and B
def A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a * p.1 + 1}
def B (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + b}

-- State the theorem
theorem intersection_implies_sum (a b : ℝ) :
  A a ∩ B b = {(2, 5)} → a + b = 5 := by
  sorry

end intersection_implies_sum_l3431_343166


namespace min_rotations_is_twelve_l3431_343176

/-- The number of elements in the letter sequence -/
def letter_sequence_length : ℕ := 6

/-- The number of elements in the digit sequence -/
def digit_sequence_length : ℕ := 4

/-- The minimum number of rotations needed for both sequences to return to their original form -/
def min_rotations : ℕ := lcm letter_sequence_length digit_sequence_length

theorem min_rotations_is_twelve : min_rotations = 12 := by
  sorry

end min_rotations_is_twelve_l3431_343176


namespace inequality_proof_l3431_343195

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a * b + c * d) * (a * d + b * c) / ((a + c) * (b + d)) ≥ Real.sqrt (a * b * c * d) := by
  sorry

end inequality_proof_l3431_343195


namespace parallelogram_projection_sum_l3431_343151

/-- Parallelogram structure -/
structure Parallelogram where
  a : ℝ  -- Length of one side
  b : ℝ  -- Length of adjacent side
  e : ℝ  -- Length of longer diagonal
  pa : ℝ  -- Projection of diagonal on side a
  pb : ℝ  -- Projection of diagonal on side b
  a_pos : 0 < a
  b_pos : 0 < b
  e_pos : 0 < e
  pa_pos : 0 < pa
  pb_pos : 0 < pb

/-- Theorem: In a parallelogram, a * pa + b * pb = e^2 -/
theorem parallelogram_projection_sum (p : Parallelogram) : p.a * p.pa + p.b * p.pb = p.e^2 := by
  sorry

end parallelogram_projection_sum_l3431_343151


namespace bank_account_growth_l3431_343196

/-- Calculates the final amount after compound interest is applied -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that $100 invested at 10% annual interest for 2 years results in $121 -/
theorem bank_account_growth : compound_interest 100 0.1 2 = 121 := by
  sorry

end bank_account_growth_l3431_343196


namespace refrigerator_price_correct_l3431_343161

/-- The purchase price of a refrigerator, given specific conditions on its sale and the sale of a mobile phone --/
def refrigerator_price : ℝ :=
  let mobile_price : ℝ := 8000
  let refrigerator_loss_percent : ℝ := 0.04
  let mobile_profit_percent : ℝ := 0.11
  let total_profit : ℝ := 280
  
  -- Define the equation to solve
  let equation (x : ℝ) : Prop :=
    (mobile_price * (1 + mobile_profit_percent) - mobile_price) - 
    (x - x * (1 - refrigerator_loss_percent)) = total_profit
  
  -- The solution (to be proved)
  15000

theorem refrigerator_price_correct : 
  refrigerator_price = 15000 := by sorry

end refrigerator_price_correct_l3431_343161


namespace triangle_side_sum_max_l3431_343187

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    M is the midpoint of BC, AM = 1, and c*cos(B) + b*cos(C) = 2a*cos(A),
    prove that the maximum value of b + c is 4√3/3 -/
theorem triangle_side_sum_max (a b c : ℝ) (A B C : ℝ) (M : ℝ × ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A →
  M = ((b * Real.cos C) / (b + c), (c * Real.cos B) / (b + c)) →
  Real.sqrt ((M.1 - 1)^2 + M.2^2) = 1 →
  (∀ b' c', b' > 0 ∧ c' > 0 ∧ 
    c' * Real.cos B + b' * Real.cos C = 2 * a * Real.cos A ∧
    Real.sqrt (((b' * Real.cos C) / (b' + c') - 1)^2 + ((c' * Real.cos B) / (b' + c'))^2) = 1 →
    b' + c' ≤ b + c) →
  b + c = 4 * Real.sqrt 3 / 3 :=
sorry

end triangle_side_sum_max_l3431_343187


namespace sqrt5_diamond_sqrt5_equals_20_l3431_343134

-- Define the ¤ operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt5_diamond_sqrt5_equals_20 : diamond (Real.sqrt 5) (Real.sqrt 5) = 20 := by
  sorry

end sqrt5_diamond_sqrt5_equals_20_l3431_343134


namespace inequality_proof_l3431_343144

theorem inequality_proof (x : ℝ) (h : x ≥ 1) : x^5 - 1/x^4 ≥ 9*(x-1) := by
  sorry

end inequality_proof_l3431_343144


namespace madeline_grocery_budget_l3431_343128

/-- Calculates the amount Madeline needs for groceries given her expenses and income. -/
theorem madeline_grocery_budget 
  (rent : ℕ) 
  (medical : ℕ) 
  (utilities : ℕ) 
  (emergency : ℕ) 
  (hourly_rate : ℕ) 
  (hours_worked : ℕ) 
  (h1 : rent = 1200)
  (h2 : medical = 200)
  (h3 : utilities = 60)
  (h4 : emergency = 200)
  (h5 : hourly_rate = 15)
  (h6 : hours_worked = 138) :
  hourly_rate * hours_worked - (rent + medical + utilities + emergency) = 410 := by
  sorry

end madeline_grocery_budget_l3431_343128


namespace no_triangle_with_given_conditions_l3431_343162

theorem no_triangle_with_given_conditions (a b c : ℕ+) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (gcd_one : Nat.gcd a.val (Nat.gcd b.val c.val) = 1)
  (div_a : a.val ∣ (b.val - c.val)^2)
  (div_b : b.val ∣ (c.val - a.val)^2)
  (div_c : c.val ∣ (a.val - b.val)^2) :
  ¬(a.val < b.val + c.val ∧ b.val < a.val + c.val ∧ c.val < a.val + b.val) :=
sorry

end no_triangle_with_given_conditions_l3431_343162


namespace muffin_banana_price_ratio_l3431_343127

/-- The price of a muffin -/
def muffin_price : ℝ := sorry

/-- The price of a banana -/
def banana_price : ℝ := sorry

/-- Elaine's total expenditure -/
def elaine_total : ℝ := 5 * muffin_price + 4 * banana_price

/-- Derek's total expenditure -/
def derek_total : ℝ := 3 * muffin_price + 18 * banana_price

/-- Derek spends three times as much as Elaine -/
axiom derek_spends_triple : derek_total = 3 * elaine_total

theorem muffin_banana_price_ratio : muffin_price = 2 * banana_price := by
  sorry

end muffin_banana_price_ratio_l3431_343127


namespace smallest_batch_size_l3431_343142

theorem smallest_batch_size (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ (21 * N)) : 
  (∀ M : ℕ, M > 70 ∧ 70 ∣ (21 * M) → N ≤ M) → N = 80 :=
by sorry

end smallest_batch_size_l3431_343142


namespace arithmetic_sequence_problem_l3431_343113

/-- Given an arithmetic sequence {a_n} with the following properties:
  1) a_4 = 7
  2) a_3 + a_6 = 16
  3) a_n = 31
  This theorem states that n = 16 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ k m : ℕ, a (k + m) - a k = m * (a 2 - a 1))  -- arithmetic sequence property
  (h2 : a 4 = 7)
  (h3 : a 3 + a 6 = 16)
  (h4 : a n = 31) :
  n = 16 := by
  sorry

end arithmetic_sequence_problem_l3431_343113


namespace original_group_size_l3431_343138

/-- Proves that the original number of men in a group is 36, given the conditions of the work completion times. -/
theorem original_group_size (total_work : ℝ) : 
  (∃ (original_group : ℕ), 
    (original_group : ℝ) / 12 * total_work = total_work ∧
    ((original_group - 6 : ℝ) / 14) * total_work = total_work) →
  ∃ (original_group : ℕ), original_group = 36 := by
sorry

end original_group_size_l3431_343138


namespace unique_root_of_equation_l3431_343178

/-- The equation |x| - 4/x = 3|x|/x has exactly one distinct real root. -/
theorem unique_root_of_equation : ∃! x : ℝ, x ≠ 0 ∧ |x| - 4/x = 3*|x|/x := by
  sorry

end unique_root_of_equation_l3431_343178


namespace smallest_number_l3431_343185

theorem smallest_number (a b c d : ℤ) (ha : a = -4) (hb : b = -3) (hc : c = 0) (hd : d = 1) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d :=
by sorry

end smallest_number_l3431_343185


namespace line_CD_passes_through_fixed_point_l3431_343198

-- Define the Cartesian plane
variable (x y : ℝ)

-- Define points E and F
def E : ℝ × ℝ := (0, 1)
def F : ℝ × ℝ := (0, -1)

-- Define the trajectory of point G
def trajectory (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define line l
def line_l (x : ℝ) : Prop := x = 4

-- Define point P on line l
def P (y₀ : ℝ) : ℝ × ℝ := (4, y₀)

-- Define points A and B (vertices of the trajectory)
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem line_CD_passes_through_fixed_point (x y y₀ : ℝ) 
  (h1 : trajectory x y) 
  (h2 : y₀ ≠ 0) :
  ∃ (xc yc xd yd : ℝ), 
    trajectory xc yc ∧ 
    trajectory xd yd ∧ 
    (yc - yd) / (xc - xd) * (1 - xc) + yc = 0 := 
sorry


end line_CD_passes_through_fixed_point_l3431_343198


namespace problem_1_l3431_343148

theorem problem_1 (x : ℝ) : (x + 2) * (-3 * x + 4) = -3 * x^2 - 2 * x + 8 := by
  sorry

end problem_1_l3431_343148


namespace triangle_property_l3431_343137

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)

/-- The main theorem stating the conditions and conclusion about the triangle. -/
theorem triangle_property (t : Triangle) 
  (h1 : 2*t.a*(Real.sin t.A) = (2*t.b + t.c)*(Real.sin t.B) + (2*t.c + t.b)*(Real.sin t.C))
  (h2 : Real.sin t.B + Real.sin t.C = 1) :
  t.A = 2*π/3 ∧ t.B = π/6 ∧ t.C = π/6 :=
by sorry

end triangle_property_l3431_343137


namespace problem_solution_l3431_343183

theorem problem_solution (x y : ℚ) (hx : x = 5/7) (hy : y = 7/5) : 
  (1/3) * x^8 * y^9 + 1/7 = 64/105 := by
sorry

end problem_solution_l3431_343183


namespace oblique_projection_preserves_parallelogram_l3431_343108

/-- Represents a parallelogram in a 2D plane -/
structure Parallelogram where
  -- Add necessary fields to define a parallelogram
  -- This is a simplified representation
  dummy : Unit

/-- Represents the oblique projection method -/
def obliqueProjection (p : Parallelogram) : Parallelogram :=
  sorry

/-- Theorem stating that the oblique projection of a parallelogram is always a parallelogram -/
theorem oblique_projection_preserves_parallelogram (p : Parallelogram) :
  ∃ (q : Parallelogram), obliqueProjection p = q :=
sorry

end oblique_projection_preserves_parallelogram_l3431_343108


namespace solve_for_a_l3431_343106

theorem solve_for_a (a : ℝ) : (3 * (-1) + 2 * a + 1 = 2) → a = 2 := by
  sorry

end solve_for_a_l3431_343106


namespace candy_box_count_l3431_343101

/-- Given 2 boxes of chocolate candy and 5 boxes of caramel candy,
    with the same number of pieces in each box,
    and a total of 28 candies, prove that there are 4 pieces in each box. -/
theorem candy_box_count (chocolate_boxes : ℕ) (caramel_boxes : ℕ) (total_candies : ℕ) 
    (h1 : chocolate_boxes = 2)
    (h2 : caramel_boxes = 5)
    (h3 : total_candies = 28)
    (h4 : ∃ (pieces_per_box : ℕ), chocolate_boxes * pieces_per_box + caramel_boxes * pieces_per_box = total_candies) :
  ∃ (pieces_per_box : ℕ), pieces_per_box = 4 ∧ 
    chocolate_boxes * pieces_per_box + caramel_boxes * pieces_per_box = total_candies :=
by
  sorry


end candy_box_count_l3431_343101


namespace quadrilateral_sine_equality_l3431_343199

-- Define a structure for a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ)  -- Interior angles
  (AB BC CD DA : ℝ)  -- Side lengths

-- Define the property of being convex
def is_convex (q : Quadrilateral) : Prop :=
  q.A > 0 ∧ q.B > 0 ∧ q.C > 0 ∧ q.D > 0 ∧ q.A + q.B + q.C + q.D = 2 * Real.pi

-- State the theorem
theorem quadrilateral_sine_equality (q : Quadrilateral) (h : is_convex q) :
  (Real.sin q.A) / (q.BC * q.CD) + (Real.sin q.C) / (q.DA * q.AB) =
  (Real.sin q.B) / (q.CD * q.DA) + (Real.sin q.D) / (q.AB * q.BC) :=
sorry

end quadrilateral_sine_equality_l3431_343199


namespace expression_simplification_l3431_343104

theorem expression_simplification (x y : ℝ) (hx : x = 1) (hy : y = -2) :
  2 * (x + y) * (x - y) + (x + y)^2 - (6 * x^3 - 4 * x^2 * y - 2 * x * y^2) / (2 * x) = -8 :=
by sorry

end expression_simplification_l3431_343104


namespace inequality_theorem_l3431_343154

theorem inequality_theorem (p : ℝ) : 
  (∀ q : ℝ, q > 0 → (4 * (p * q^2 + p^3 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 
  (0 ≤ p ∧ p < (2 + 2 * Real.sqrt 13) / 3) :=
sorry

end inequality_theorem_l3431_343154


namespace square_area_from_diagonal_l3431_343194

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let side := d / Real.sqrt 2
  side * side = 144 := by
sorry

end square_area_from_diagonal_l3431_343194


namespace quadratic_expression_value_l3431_343132

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 2*x - 3 = 0) :
  (2*x - 1)^2 - (x - 1)*(x + 3) = 13 := by
  sorry

end quadratic_expression_value_l3431_343132


namespace range_of_a_l3431_343156

theorem range_of_a (x₁ x₂ m a : ℝ) : 
  (∀ m ∈ Set.Icc (-1 : ℝ) 1, x₁^2 - m*x₁ - 2 = 0 ∧ x₂^2 - m*x₂ - 2 = 0) →
  (∀ m ∈ Set.Icc (-1 : ℝ) 1, a^2 - 5*a - 3 ≥ |x₁ - x₂|) →
  (¬∃ x, a*x^2 + 2*x - 1 > 0) →
  (∃ m ∈ Set.Icc (-1 : ℝ) 1, a^2 - 5*a - 3 < |x₁ - x₂|) →
  a ≤ -1 :=
by sorry

end range_of_a_l3431_343156


namespace cos_equality_problem_l3431_343111

theorem cos_equality_problem (n : ℤ) :
  0 ≤ n ∧ n ≤ 360 →
  (Real.cos (n * π / 180) = Real.cos (145 * π / 180) ↔ n = 145 ∨ n = 215) :=
by sorry

end cos_equality_problem_l3431_343111


namespace mirror_reflection_of_16_00_l3431_343158

/-- Represents a clock time with hours and minutes -/
structure ClockTime where
  hours : Nat
  minutes : Nat
  h_valid : hours < 12
  m_valid : minutes < 60

/-- Represents the reflection of a clock time in a mirror -/
def mirror_reflect (t : ClockTime) : ClockTime :=
  { hours := (12 - t.hours) % 12,
    minutes := t.minutes,
    h_valid := by sorry,
    m_valid := t.m_valid }

/-- The theorem stating that 16:00 reflects to approximately 8:00 in a mirror -/
theorem mirror_reflection_of_16_00 :
  let t : ClockTime := ⟨4, 0, by sorry, by sorry⟩
  let reflected : ClockTime := mirror_reflect t
  reflected.hours = 8 ∧ reflected.minutes = 0 :=
by sorry

end mirror_reflection_of_16_00_l3431_343158


namespace no_integer_solution_3x2_plus_2_eq_y2_l3431_343143

theorem no_integer_solution_3x2_plus_2_eq_y2 :
  ∀ (x y : ℤ), 3 * x^2 + 2 ≠ y^2 := by
  sorry

end no_integer_solution_3x2_plus_2_eq_y2_l3431_343143


namespace function_inequality_implies_constant_bound_l3431_343100

open Real MeasureTheory

theorem function_inequality_implies_constant_bound 
  (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x < g x) →
  (∀ x > 0, f x = a * exp (x / 2) - x) →
  (∀ x > 0, g x = x * log x - (1 / 2) * x^2) →
  a < -exp (-2) := by
sorry

end function_inequality_implies_constant_bound_l3431_343100


namespace max_value_implies_k_l3431_343191

def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * k * x + 1

theorem max_value_implies_k (k : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f k x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, f k x = 4) →
  k = 3/8 ∨ k = -3 := by
  sorry

end max_value_implies_k_l3431_343191


namespace single_round_robin_games_planned_games_equation_l3431_343107

/-- Represents the number of games in a single round-robin tournament -/
def num_games (x : ℕ) : ℚ := (x * (x - 1)) / 2

/-- Theorem: In a single round-robin tournament with x teams, 
    the total number of games is given by (x * (x - 1)) / 2 -/
theorem single_round_robin_games (x : ℕ) : 
  num_games x = (x * (x - 1)) / 2 := by
  sorry

/-- Given 15 planned games, prove that the equation (x * (x - 1)) / 2 = 15 
    correctly represents the number of games in terms of x -/
theorem planned_games_equation (x : ℕ) : 
  num_games x = 15 ↔ (x * (x - 1)) / 2 = 15 := by
  sorry

end single_round_robin_games_planned_games_equation_l3431_343107


namespace ship_lock_weight_scientific_notation_l3431_343136

theorem ship_lock_weight_scientific_notation :
  867000 = 8.67 * (10 : ℝ) ^ 5 := by sorry

end ship_lock_weight_scientific_notation_l3431_343136


namespace alices_favorite_number_l3431_343133

def is_multiple (a b : ℕ) : Prop := ∃ k, a = b * k

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem alices_favorite_number :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 200 ∧
  is_multiple n 13 ∧
  ¬is_multiple n 3 ∧
  is_multiple (digit_sum n) 5 ∧
  n = 104 := by
sorry

end alices_favorite_number_l3431_343133


namespace equation_solution_l3431_343103

theorem equation_solution : ∃! x : ℚ, 3 * (x - 2) + 1 = x - (2 * x - 1) := by sorry

end equation_solution_l3431_343103


namespace arithmetic_sequence_common_difference_l3431_343160

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a4 : a 4 = 13)
  (h_a7 : a 7 = 25) :
  ∃ d : ℝ, d = 4 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end arithmetic_sequence_common_difference_l3431_343160


namespace inequality_solutions_l3431_343182

theorem inequality_solutions :
  (∀ x : ℝ, 3 * x > 2 * (1 - x) ↔ x > 2/5) ∧
  (∀ x : ℝ, (3 * x - 7) / 2 ≤ x - 2 ∧ 4 * (x - 1) > 4 ↔ 2 < x ∧ x ≤ 3) := by
  sorry

end inequality_solutions_l3431_343182


namespace range_of_a_l3431_343147

/-- The function f(x) = x^2 - 4x -/
def f (x : ℝ) : ℝ := x^2 - 4*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4) a, f x ∈ Set.Icc (-4) 32) →
  (Set.Icc (-4) a = f ⁻¹' (Set.Icc (-4) 32)) →
  a ∈ Set.Icc 2 8 :=
sorry

end range_of_a_l3431_343147


namespace max_value_of_E_l3431_343169

theorem max_value_of_E (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^5 + b^5 = a^3 + b^3) : 
  ∃ (M : ℝ), M = 1 ∧ ∀ x y, x > 0 → y > 0 → x^5 + y^5 = x^3 + y^3 → 
  x^2 - x*y + y^2 ≤ M :=
by sorry

end max_value_of_E_l3431_343169


namespace f_g_deriv_pos_pos_l3431_343153

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_deriv_pos_neg : ∀ x : ℝ, x < 0 → deriv f x > 0
axiom g_deriv_neg_neg : ∀ x : ℝ, x < 0 → deriv g x < 0

-- State the theorem
theorem f_g_deriv_pos_pos : 
  ∀ x : ℝ, x > 0 → deriv f x > 0 ∧ deriv g x > 0 := by sorry

end f_g_deriv_pos_pos_l3431_343153


namespace probability_multiple_of_three_l3431_343163

/-- The probability of selecting a ticket with a number that is a multiple of 3
    from a set of tickets numbered 1 to 27 is equal to 1/3. -/
theorem probability_multiple_of_three (n : ℕ) (h : n = 27) :
  (Finset.filter (fun x => x % 3 = 0) (Finset.range n)).card / n = 1 / 3 := by
  sorry

end probability_multiple_of_three_l3431_343163


namespace negation_equivalence_l3431_343123

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5*x₀ + 6 > 0) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) :=
by sorry

end negation_equivalence_l3431_343123


namespace money_distribution_l3431_343149

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (ac_sum : A + C = 200)
  (c_value : C = 50) :
  B + C = 350 := by
  sorry

end money_distribution_l3431_343149


namespace archibalds_apples_l3431_343188

/-- Archibald's apple eating problem -/
theorem archibalds_apples :
  let apples_per_day_first_two_weeks : ℕ := 1
  let weeks_first_period : ℕ := 2
  let weeks_second_period : ℕ := 3
  let weeks_third_period : ℕ := 2
  let total_weeks : ℕ := weeks_first_period + weeks_second_period + weeks_third_period
  let average_apples_per_week : ℕ := 10
  let total_apples : ℕ := average_apples_per_week * total_weeks
  let apples_first_two_weeks : ℕ := apples_per_day_first_two_weeks * 7 * weeks_first_period
  let apples_next_three_weeks : ℕ := apples_first_two_weeks * weeks_second_period
  let apples_last_two_weeks : ℕ := total_apples - apples_first_two_weeks - apples_next_three_weeks
  apples_last_two_weeks / (7 * weeks_third_period) = 1 :=
by sorry


end archibalds_apples_l3431_343188


namespace ghost_mansion_paths_8_2_l3431_343126

/-- Calculates the number of ways a ghost can enter and exit a mansion with different windows --/
def ghost_mansion_paths (total_windows : ℕ) (locked_windows : ℕ) : ℕ :=
  let usable_windows := total_windows - locked_windows
  usable_windows * (usable_windows - 1)

/-- Theorem: The number of ways for a ghost to enter and exit a mansion with 8 windows, 2 of which are locked, is 30 --/
theorem ghost_mansion_paths_8_2 :
  ghost_mansion_paths 8 2 = 30 := by
  sorry

#eval ghost_mansion_paths 8 2

end ghost_mansion_paths_8_2_l3431_343126


namespace algebrist_great_probability_l3431_343114

def algebrist : Finset Char := {'A', 'L', 'G', 'E', 'B', 'R', 'I', 'S', 'T'}
def great : Finset Char := {'G', 'R', 'E', 'A', 'T'}

theorem algebrist_great_probability :
  Finset.card (algebrist ∩ great) / Finset.card algebrist = 5 / 9 := by
sorry

end algebrist_great_probability_l3431_343114


namespace litter_collection_weight_l3431_343167

/-- The total weight of litter collected by Gina and her neighbors -/
def total_litter_weight (gina_glass_bags : ℕ) (gina_plastic_bags : ℕ)
  (glass_weight : ℕ) (plastic_weight : ℕ) (neighbor_glass_factor : ℕ)
  (neighbor_plastic_factor : ℕ) : ℕ :=
  let gina_glass := gina_glass_bags * glass_weight
  let gina_plastic := gina_plastic_bags * plastic_weight
  let neighbor_glass := neighbor_glass_factor * gina_glass
  let neighbor_plastic := neighbor_plastic_factor * gina_plastic
  gina_glass + gina_plastic + neighbor_glass + neighbor_plastic

/-- Theorem stating the total weight of litter collected -/
theorem litter_collection_weight :
  total_litter_weight 5 3 7 4 120 80 = 5207 := by
  sorry

end litter_collection_weight_l3431_343167


namespace ellipse_major_axis_length_l3431_343173

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def majorAxisLength (cylinderRadius : ℝ) (majorMinorRatio : ℝ) : ℝ :=
  2 * cylinderRadius * majorMinorRatio

/-- Theorem: The length of the major axis of the ellipse is 12 -/
theorem ellipse_major_axis_length :
  majorAxisLength 3 2 = 12 := by
  sorry

end ellipse_major_axis_length_l3431_343173


namespace consecutive_odd_numbers_problem_l3431_343125

theorem consecutive_odd_numbers_problem (x : ℤ) : 
  (∃ (y z : ℤ), y = x + 2 ∧ z = x + 4 ∧ 
   x % 2 = 1 ∧ y % 2 = 1 ∧ z % 2 = 1 ∧
   8 * x = 3 * z + 2 * y + 5) →
  x = 7 :=
by sorry

end consecutive_odd_numbers_problem_l3431_343125


namespace intersection_line_circle_l3431_343102

theorem intersection_line_circle (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 + A.2 = a) ∧ (B.1 + B.2 = a) ∧
    (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧
    ((A.1 + B.1)^2 + (A.2 + B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2)) →
  (a = 2 ∨ a = -2) :=
by sorry

end intersection_line_circle_l3431_343102


namespace shaded_circle_is_six_l3431_343112

/-- Represents the circle positions in the diagram --/
inductive Position
| Top
| Left
| Right
| Bottom
| Shaded
| Other

/-- Checks if a number is prime --/
def isPrime (n : Nat) : Bool :=
  n > 1 && (Nat.factors n).length == 1

/-- Represents the arrangement of numbers in the circles --/
def Arrangement := Position → Nat

/-- Checks if an arrangement is valid according to the problem conditions --/
def isValidArrangement (arr : Arrangement) : Prop :=
  arr Position.Top = 5 ∧
  ({6, 7, 8, 9, 10} : Set Nat) = {arr Position.Left, arr Position.Right, arr Position.Bottom, arr Position.Shaded, arr Position.Other} ∧
  (∀ p q : Position, p ≠ q → isPrime (arr p + arr q))

theorem shaded_circle_is_six (arr : Arrangement) (h : isValidArrangement arr) : 
  arr Position.Shaded = 6 := by
  sorry

#check shaded_circle_is_six

end shaded_circle_is_six_l3431_343112


namespace nails_on_square_plate_l3431_343197

/-- Represents a square plate with nails along its edges -/
structure SquarePlate where
  side_length : ℕ
  total_nails : ℕ
  nails_per_side : ℕ
  h1 : total_nails > 0
  h2 : nails_per_side > 0
  h3 : total_nails = 4 * nails_per_side

/-- Theorem: For a square plate with 96 nails evenly distributed along its edges,
    there are 24 nails on each side -/
theorem nails_on_square_plate :
  ∀ (plate : SquarePlate), plate.total_nails = 96 → plate.nails_per_side = 24 :=
by
  sorry


end nails_on_square_plate_l3431_343197


namespace rope_division_l3431_343172

theorem rope_division (rope_length : ℚ) (n_parts : ℕ) (h1 : rope_length = 8/15) (h2 : n_parts = 3) :
  let part_fraction : ℚ := 1 / n_parts
  let part_length : ℚ := rope_length / n_parts
  (part_fraction = 1/3) ∧ (part_length = 8/45) := by
  sorry

end rope_division_l3431_343172


namespace sum_geq_sqrt_three_l3431_343109

theorem sum_geq_sqrt_three (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (h : a * b + b * c + c * a = 1) : 
  a + b + c ≥ Real.sqrt 3 ∧ 
  (a + b + c = Real.sqrt 3 ↔ a = 1 / Real.sqrt 3 ∧ b = 1 / Real.sqrt 3 ∧ c = 1 / Real.sqrt 3) :=
sorry

end sum_geq_sqrt_three_l3431_343109


namespace piecewise_function_proof_l3431_343122

theorem piecewise_function_proof (x : ℝ) : 
  let a : ℝ → ℝ := λ x => (3 * x + 3) / 2
  let b : ℝ → ℝ := λ x => 5 * x / 2
  let c : ℝ → ℝ := λ x => -x + 1/2
  (x < -1 → |a x| - |b x| + c x = -1) ∧
  (-1 ≤ x ∧ x ≤ 0 → |a x| - |b x| + c x = 3 * x + 2) ∧
  (0 < x → |a x| - |b x| + c x = -2 * x + 2) :=
by sorry

end piecewise_function_proof_l3431_343122


namespace boat_speed_in_still_water_l3431_343184

/-- The speed of a boat in still water given downstream travel information -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 68)
  (h3 : downstream_time = 4) :
  downstream_distance / downstream_time - stream_speed = 13 := by
sorry

end boat_speed_in_still_water_l3431_343184


namespace frank_miles_proof_l3431_343181

/-- The number of miles Jim ran in 2 hours -/
def jim_miles : ℝ := 16

/-- The number of hours Jim and Frank ran -/
def total_hours : ℝ := 2

/-- The difference in miles per hour between Frank and Jim -/
def frank_jim_diff : ℝ := 2

/-- Frank's total miles run in 2 hours -/
def frank_total_miles : ℝ := 20

theorem frank_miles_proof :
  frank_total_miles = (jim_miles / total_hours + frank_jim_diff) * total_hours :=
by sorry

end frank_miles_proof_l3431_343181
