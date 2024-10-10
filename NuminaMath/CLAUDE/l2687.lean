import Mathlib

namespace cooking_gear_final_cost_l2687_268768

def cookingGearCost (mitts apron utensils recipients discount tax : ℝ) : ℝ :=
  let knife := 2 * utensils
  let setPrice := mitts + apron + utensils + knife
  let discountedPrice := setPrice * (1 - discount)
  let totalBeforeTax := discountedPrice * recipients
  totalBeforeTax * (1 + tax)

theorem cooking_gear_final_cost :
  cookingGearCost 14 16 10 8 0.25 0.08 = 388.80 := by
  sorry

end cooking_gear_final_cost_l2687_268768


namespace module_stock_worth_l2687_268728

/-- Calculates the total worth of a stock of modules -/
theorem module_stock_worth (total_modules : ℕ) (cheap_modules : ℕ) (expensive_cost : ℚ) (cheap_cost : ℚ) 
  (h1 : total_modules = 22)
  (h2 : cheap_modules = 21)
  (h3 : expensive_cost = 10)
  (h4 : cheap_cost = 5/2)
  : (total_modules - cheap_modules) * expensive_cost + cheap_modules * cheap_cost = 125/2 := by
  sorry

#eval (22 - 21) * 10 + 21 * (5/2)  -- This should output 62.5

end module_stock_worth_l2687_268728


namespace triangle_area_solution_l2687_268770

theorem triangle_area_solution (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * (2*x) * x = 50 → x = 5 * Real.sqrt 2 := by
  sorry

end triangle_area_solution_l2687_268770


namespace geometric_sequence_sum_l2687_268744

/-- Given a geometric sequence with common ratio 2 and sum of first 4 terms equal to 1,
    the sum of the first 8 terms is 17 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = 2 * a n) 
    (h2 : a 1 + a 2 + a 3 + a 4 = 1) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 17 := by
  sorry

end geometric_sequence_sum_l2687_268744


namespace triangle_area_with_given_base_height_l2687_268757

/-- The area of a triangle with base 12 cm and height 15 cm is 90 cm². -/
theorem triangle_area_with_given_base_height :
  let base : ℝ := 12
  let height : ℝ := 15
  let area : ℝ := (1 / 2) * base * height
  area = 90 :=
by sorry

end triangle_area_with_given_base_height_l2687_268757


namespace existence_of_counterexample_l2687_268784

theorem existence_of_counterexample : ∃ m n : ℝ, m > n ∧ m^2 ≤ n^2 := by
  sorry

end existence_of_counterexample_l2687_268784


namespace train_bridge_crossing_time_l2687_268775

/-- Proves that a train of given length and speed takes 30 seconds to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : Real)
  (train_speed_kmh : Real)
  (bridge_length : Real)
  (h1 : train_length = 140)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 235) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry


end train_bridge_crossing_time_l2687_268775


namespace cos_sin_difference_l2687_268795

theorem cos_sin_difference (α β : Real) 
  (h : Real.cos (α + β) * Real.cos (α - β) = 1/3) : 
  Real.cos α ^ 2 - Real.sin β ^ 2 = 1/3 := by
sorry

end cos_sin_difference_l2687_268795


namespace sum_from_true_discount_and_simple_interest_l2687_268758

theorem sum_from_true_discount_and_simple_interest 
  (S : ℝ) 
  (D I : ℝ) 
  (h1 : D = 75) 
  (h2 : I = 85) 
  (h3 : D / I = (S - D) / S) : S = 637.5 := by
  sorry

end sum_from_true_discount_and_simple_interest_l2687_268758


namespace expedition_investigation_days_l2687_268727

theorem expedition_investigation_days 
  (upstream_speed : ℕ) 
  (downstream_speed : ℕ) 
  (total_days : ℕ) 
  (final_day_distance : ℕ) 
  (h1 : upstream_speed = 17)
  (h2 : downstream_speed = 25)
  (h3 : total_days = 60)
  (h4 : final_day_distance = 24) :
  ∃ (upstream_days downstream_days investigation_days : ℕ),
    upstream_days + downstream_days + investigation_days = total_days ∧
    upstream_speed * upstream_days - downstream_speed * downstream_days = final_day_distance - downstream_speed ∧
    investigation_days = 23 := by
  sorry

#check expedition_investigation_days

end expedition_investigation_days_l2687_268727


namespace kaleb_first_half_score_l2687_268717

/-- Calculates the first half score in a trivia game given the total score and second half score. -/
def first_half_score (total_score second_half_score : ℕ) : ℕ :=
  total_score - second_half_score

/-- Proves that Kaleb's first half score is 43 points given his total score of 66 and second half score of 23. -/
theorem kaleb_first_half_score :
  first_half_score 66 23 = 43 := by
  sorry

end kaleb_first_half_score_l2687_268717


namespace possible_values_of_a_l2687_268778

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27 * x^3) 
  (h3 : a - b = 2 * x) : 
  a = 3.041 * x ∨ a = -1.041 * x := by
  sorry

end possible_values_of_a_l2687_268778


namespace infinite_power_tower_eq_four_solution_l2687_268741

/-- Define the infinite power tower function --/
noncomputable def infinitePowerTower (x : ℝ) : ℝ :=
  Real.log x / Real.log 2

/-- Theorem: The solution to x^(x^(x^...)) = 4 is √2 --/
theorem infinite_power_tower_eq_four_solution :
  ∀ x : ℝ, x > 0 → infinitePowerTower x = 4 → x = Real.sqrt 2 :=
by
  sorry


end infinite_power_tower_eq_four_solution_l2687_268741


namespace cracker_cost_is_350_l2687_268745

/-- The cost of a box of crackers in dollars -/
def cracker_cost : ℝ := sorry

/-- The total cost before discount in dollars -/
def total_cost_before_discount : ℝ := 5 + 4 * 2 + 3.5 + cracker_cost

/-- The discount rate as a decimal -/
def discount_rate : ℝ := 0.1

/-- The total cost after discount in dollars -/
def total_cost_after_discount : ℝ := total_cost_before_discount * (1 - discount_rate)

theorem cracker_cost_is_350 :
  cracker_cost = 3.5 ∧ total_cost_after_discount = 18 := by sorry

end cracker_cost_is_350_l2687_268745


namespace fixed_point_of_exponential_function_l2687_268769

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f := λ x : ℝ => a^(x - 1) + 4
  f 1 = 5 :=
by
  sorry

end fixed_point_of_exponential_function_l2687_268769


namespace unique_solution_diophantine_equation_l2687_268777

theorem unique_solution_diophantine_equation :
  ∀ x y : ℕ+, 2 * x^2 + 5 * y^2 = 11 * (x * y - 11) ↔ x = 14 ∧ y = 27 := by
  sorry

end unique_solution_diophantine_equation_l2687_268777


namespace quadratic_inequality_solution_l2687_268794

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 2*x > 35) ↔ (x < -5 ∨ x > 7) :=
sorry

end quadratic_inequality_solution_l2687_268794


namespace roots_of_polynomial_l2687_268753

def f (x : ℝ) : ℝ := 4 * x^5 + 13 * x^4 - 30 * x^3 + 8 * x^2

theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = (1 : ℝ) / 2 ∨ x = -2 + 2 * Real.sqrt 2 ∨ x = -2 - 2 * Real.sqrt 2) ∧
  (∃ ε > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < ε → f x / x^2 ≠ 0) :=
by sorry

end roots_of_polynomial_l2687_268753


namespace fine_amount_correct_l2687_268735

/-- Calculates the fine amount given the quantity sold, price per ounce, and amount left after the fine -/
def calculate_fine (quantity_sold : ℕ) (price_per_ounce : ℕ) (amount_left : ℕ) : ℕ :=
  quantity_sold * price_per_ounce - amount_left

/-- Proves that the fine amount is correct given the problem conditions -/
theorem fine_amount_correct : calculate_fine 8 9 22 = 50 := by
  sorry

end fine_amount_correct_l2687_268735


namespace inequality_proofs_l2687_268781

theorem inequality_proofs (a b : ℝ) :
  (a ≥ b ∧ b > 0) →
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b ∧
  (a > 0 ∧ b > 0 ∧ a + b = 10) →
  Real.sqrt (1 + 3 * a) + Real.sqrt (1 + 3 * b) ≤ 8 :=
by sorry

end inequality_proofs_l2687_268781


namespace consecutive_sum_33_l2687_268746

theorem consecutive_sum_33 (m : ℕ) (h1 : m > 1) :
  (∃ a : ℕ, (Finset.range m).sum (λ i => a + i) = 33) ↔ m = 2 ∨ m = 3 ∨ m = 6 := by
  sorry

end consecutive_sum_33_l2687_268746


namespace smallest_reachable_integer_l2687_268779

-- Define the sequence u_n
def u : ℕ → ℕ
  | 0 => 2010^2010
  | (n+1) => if u n % 2 = 1 then u n + 7 else u n / 2

-- Define the property of being reachable by the sequence
def Reachable (m : ℕ) : Prop := ∃ n, u n = m

-- State the theorem
theorem smallest_reachable_integer : 
  (∃ m, Reachable m) ∧ (∀ k, Reachable k → k ≥ 1) := by sorry

end smallest_reachable_integer_l2687_268779


namespace five_kg_to_g_eight_thousand_g_to_kg_l2687_268723

-- Define the conversion factor
def kg_to_g : ℝ := 1000

-- Theorem for converting 5 kg to grams
theorem five_kg_to_g : 5 * kg_to_g = 5000 := by sorry

-- Theorem for converting 8000 g to kg
theorem eight_thousand_g_to_kg : 8000 / kg_to_g = 8 := by sorry

end five_kg_to_g_eight_thousand_g_to_kg_l2687_268723


namespace pure_imaginary_fraction_l2687_268760

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I : ℂ) * Complex.I = -1 →
  (∃ b : ℝ, (a + Complex.I) / (1 - Complex.I) = b * Complex.I) →
  a = 1 := by
sorry

end pure_imaginary_fraction_l2687_268760


namespace trig_identity_proof_l2687_268748

theorem trig_identity_proof : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 
  4 * Real.sin (10 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end trig_identity_proof_l2687_268748


namespace andy_problem_count_l2687_268763

/-- The number of problems Andy solves when he completes problems from 80 to 125 inclusive -/
def problems_solved : ℕ := 125 - 80 + 1

theorem andy_problem_count : problems_solved = 46 := by
  sorry

end andy_problem_count_l2687_268763


namespace jacks_hair_length_l2687_268750

/-- Given the relative lengths of Kate's, Emily's, Logan's, and Jack's hair, prove that Jack's hair is 39 inches long. -/
theorem jacks_hair_length (logan_hair emily_hair kate_hair jack_hair : ℝ) : 
  logan_hair = 20 →
  emily_hair = logan_hair + 6 →
  kate_hair = emily_hair / 2 →
  jack_hair = 3 * kate_hair →
  jack_hair = 39 :=
by
  sorry

#check jacks_hair_length

end jacks_hair_length_l2687_268750


namespace proposition_truth_values_l2687_268751

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def is_solution (x : ℤ) : Prop := x^2 + x - 2 = 0

theorem proposition_truth_values :
  ((is_prime 3 ∨ is_even 3) = true) ∧
  ((is_prime 3 ∧ is_even 3) = false) ∧
  ((¬is_prime 3) = false) ∧
  ((is_solution (-2) ∨ is_solution 1) = true) ∧
  ((is_solution (-2) ∧ is_solution 1) = true) ∧
  ((¬is_solution (-2)) = false) := by
  sorry

end proposition_truth_values_l2687_268751


namespace rank_of_Mn_l2687_268755

/-- Definition of the matrix Mn -/
def Mn (n : ℕ+) : Matrix (Fin (2*n+1)) (Fin (2*n+1)) ℤ :=
  Matrix.of fun i j =>
    if i = j then 0
    else if i > j then
      if i - j ≤ n then 1 else -1
    else
      if j - i ≤ n then -1 else 1

/-- The rank of Mn is 2n for any positive integer n -/
theorem rank_of_Mn (n : ℕ+) : Matrix.rank (Mn n) = 2*n := by sorry

end rank_of_Mn_l2687_268755


namespace victor_sticker_count_l2687_268764

/-- The number of stickers Victor has -/
def total_stickers (flower animal insect space : ℕ) : ℕ :=
  flower + animal + insect + space

theorem victor_sticker_count :
  ∀ (flower animal insect space : ℕ),
    flower = 12 →
    animal = 8 →
    insect = animal - 3 →
    space = flower + 7 →
    total_stickers flower animal insect space = 44 :=
by
  sorry

end victor_sticker_count_l2687_268764


namespace certain_number_problem_l2687_268762

theorem certain_number_problem : ∃! x : ℝ, ((x - 50) / 4) * 3 + 28 = 73 := by
  sorry

end certain_number_problem_l2687_268762


namespace line_through_quadrants_line_through_fixed_point_point_slope_form_line_equation_l2687_268720

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- 1. Line passing through first, second, and fourth quadrants
theorem line_through_quadrants (k b : ℝ) :
  (∀ x y, y = k * x + b → 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))) →
  k < 0 ∧ b > 0 :=
sorry

-- 2. Line passing through fixed point
theorem line_through_fixed_point (a : ℝ) :
  ∀ x y, y = a * x - 3 * a + 2 → (x = 3 → y = 2) :=
sorry

-- 3. Point-slope form equation
theorem point_slope_form (p : Point) (m : ℝ) :
  p.x = 2 ∧ p.y = -1 ∧ m = -Real.sqrt 3 →
  ∀ x y, y + 1 = m * (x - 2) ↔ y = m * (x - p.x) + p.y :=
sorry

-- 4. Line equation with given slope and intercept
theorem line_equation (l : Line) :
  l.slope = -2 ∧ l.intercept = 3 →
  ∀ x y, y = l.slope * x + l.intercept ↔ y = -2 * x + 3 :=
sorry

end line_through_quadrants_line_through_fixed_point_point_slope_form_line_equation_l2687_268720


namespace trigonometric_identities_l2687_268725

theorem trigonometric_identities :
  (((Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) ^ 2) = 1/2) ∧
  ((Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3)) = -1) ∧
  ((Real.cos (24 * π / 180) * Real.cos (36 * π / 180) - Real.cos (66 * π / 180) * Real.cos (54 * π / 180)) = 1/2) :=
by sorry

end trigonometric_identities_l2687_268725


namespace c_invests_after_eight_months_l2687_268737

/-- Represents the investment problem with three partners A, B, and C --/
structure InvestmentProblem where
  initial_investment : ℝ
  annual_gain : ℝ
  a_share : ℝ
  b_invest_time : ℕ
  c_invest_time : ℕ

/-- Calculates the time when C invests given the problem parameters --/
def calculate_c_invest_time (problem : InvestmentProblem) : ℕ :=
  let a_investment := problem.initial_investment * 12
  let b_investment := 2 * problem.initial_investment * (12 - problem.b_invest_time)
  let c_investment := 3 * problem.initial_investment * problem.c_invest_time
  let total_investment := a_investment + b_investment + c_investment
  problem.c_invest_time

/-- Theorem stating that C invests after 8 months --/
theorem c_invests_after_eight_months (problem : InvestmentProblem) 
  (h1 : problem.annual_gain = 21000)
  (h2 : problem.a_share = 7000)
  (h3 : problem.b_invest_time = 6) :
  calculate_c_invest_time problem = 8 := by
  sorry

#eval calculate_c_invest_time {
  initial_investment := 1000,
  annual_gain := 21000,
  a_share := 7000,
  b_invest_time := 6,
  c_invest_time := 8
}

end c_invests_after_eight_months_l2687_268737


namespace croissants_for_breakfast_l2687_268793

theorem croissants_for_breakfast (total_items cakes pizzas : ℕ) 
  (h1 : total_items = 110)
  (h2 : cakes = 18)
  (h3 : pizzas = 30) :
  total_items - cakes - pizzas = 62 := by
  sorry

end croissants_for_breakfast_l2687_268793


namespace power_fraction_equality_l2687_268700

theorem power_fraction_equality : (2^2015 + 2^2011) / (2^2015 - 2^2011) = 17/15 := by
  sorry

end power_fraction_equality_l2687_268700


namespace perpendicular_lines_a_value_l2687_268767

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y + 6 = 0 → x + a*(a+1)*y + (a^2-1) = 0 → 
   (a * 1 + 2 * (a*(a+1)) = 0)) → 
  (a = 0 ∨ a = -3/2) := by
sorry

end perpendicular_lines_a_value_l2687_268767


namespace handshake_problem_l2687_268791

theorem handshake_problem (a b : ℕ) : 
  a + b = 20 →
  (a.choose 2) + (b.choose 2) = 106 →
  a * b = 84 :=
by sorry

end handshake_problem_l2687_268791


namespace unique_prime_with_square_free_remainders_l2687_268704

theorem unique_prime_with_square_free_remainders : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  (∀ q : ℕ, Nat.Prime q → q < p → 
    ∀ k r : ℕ, p = k * q + r → 0 ≤ r → r < q → 
      ∀ a : ℕ, a > 1 → ¬(a * a ∣ r)) ∧
  p = 13 := by
sorry

end unique_prime_with_square_free_remainders_l2687_268704


namespace sum_of_tangent_slopes_l2687_268719

/-- The parabola P defined by y = x^2 -/
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- The point Q -/
def Q : ℝ × ℝ := (10, 36)

/-- The quadratic equation whose roots are the slopes of tangent lines -/
def tangent_slope_equation (m : ℝ) : Prop := m^2 - 40*m + 144 = 0

/-- The theorem stating that the sum of roots of the tangent slope equation is 40 -/
theorem sum_of_tangent_slopes :
  ∃ (r s : ℝ), (∀ m : ℝ, tangent_slope_equation m ↔ m = r ∨ m = s) ∧ r + s = 40 := by sorry

end sum_of_tangent_slopes_l2687_268719


namespace range_of_m_l2687_268790

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  ∀ m : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m) ↔ m < 8 :=
by sorry

end range_of_m_l2687_268790


namespace toms_sleep_deficit_l2687_268780

/-- Calculates the sleep deficit for a week given ideal and actual sleep hours -/
def sleep_deficit (ideal_hours : ℕ) (weeknight_hours : ℕ) (weekend_hours : ℕ) : ℕ :=
  let ideal_total := ideal_hours * 7
  let actual_total := weeknight_hours * 5 + weekend_hours * 2
  ideal_total - actual_total

/-- Proves that Tom's sleep deficit for a week is 19 hours -/
theorem toms_sleep_deficit : sleep_deficit 8 5 6 = 19 := by
  sorry

end toms_sleep_deficit_l2687_268780


namespace coloring_book_problem_l2687_268798

/-- The number of pictures colored given the initial count and remaining count --/
def pictures_colored (initial_count : ℕ) (remaining_count : ℕ) : ℕ :=
  initial_count - remaining_count

/-- Theorem stating that given two coloring books with 44 pictures each and 68 pictures left to color, 
    the number of pictures colored is 20 --/
theorem coloring_book_problem :
  let book1_count : ℕ := 44
  let book2_count : ℕ := 44
  let total_count : ℕ := book1_count + book2_count
  let remaining_count : ℕ := 68
  pictures_colored total_count remaining_count = 20 := by
  sorry

end coloring_book_problem_l2687_268798


namespace value_set_of_x_l2687_268752

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- State the theorem
theorem value_set_of_x (x : ℝ) :
  (∀ a : ℝ, a ≠ 0 → f x ≥ (|a + 1| - |2*a - 1|) / |a|) →
  x ≤ -1 ∨ x ≥ 2 :=
by sorry

end value_set_of_x_l2687_268752


namespace reciprocal_roots_quadratic_l2687_268799

/-- Given a quadratic equation x^2 + mx + (m^2 - 3m + 3) = 0, 
    if its roots are reciprocals of each other, then m = 2 -/
theorem reciprocal_roots_quadratic (m : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 
   x^2 + m*x + (m^2 - 3*m + 3) = 0 ∧
   y^2 + m*y + (m^2 - 3*m + 3) = 0 ∧
   x*y = 1) →
  m = 2 := by
sorry

end reciprocal_roots_quadratic_l2687_268799


namespace paperboy_delivery_ways_l2687_268740

/-- Represents the number of ways to deliver newspapers to n houses without missing four consecutive houses. -/
def delivery_ways : ℕ → ℕ
  | 0 => 1  -- base case: one way to deliver to zero houses
  | 1 => 2  -- base case: two ways to deliver to one house
  | 2 => 4  -- base case: four ways to deliver to two houses
  | 3 => 8  -- base case: eight ways to deliver to three houses
  | n + 4 => delivery_ways (n + 3) + delivery_ways (n + 2) + delivery_ways (n + 1) + delivery_ways n

/-- Theorem stating that there are 2872 ways for a paperboy to deliver newspapers to 12 houses without missing four consecutive houses. -/
theorem paperboy_delivery_ways :
  delivery_ways 12 = 2872 := by
  sorry

end paperboy_delivery_ways_l2687_268740


namespace hyperbola_eccentricity_l2687_268711

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0, 
    and the length of the conjugate axis is twice that of the transverse axis (b = 2a),
    prove that its eccentricity is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_axis : b = 2 * a) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 5 := by sorry

end hyperbola_eccentricity_l2687_268711


namespace algebraic_equation_proof_l2687_268734

theorem algebraic_equation_proof (a b c : ℝ) 
  (h1 : a^2 + b*c = 14) 
  (h2 : b^2 - 2*b*c = -6) : 
  3*a^2 + 4*b^2 - 5*b*c = 18 := by
  sorry

end algebraic_equation_proof_l2687_268734


namespace rectangular_plot_area_l2687_268732

/-- Proves that a rectangular plot with length to breadth ratio 7:5 and perimeter 288 meters has an area of 5040 square meters. -/
theorem rectangular_plot_area (length width : ℝ) : 
  (length / width = 7 / 5) →  -- ratio condition
  (2 * (length + width) = 288) →  -- perimeter condition
  (length * width = 5040) :=  -- area to prove
by
  sorry

end rectangular_plot_area_l2687_268732


namespace frustum_smaller_cone_height_l2687_268759

-- Define the frustum
structure Frustum where
  height : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

-- Define the theorem
theorem frustum_smaller_cone_height (f : Frustum) 
  (h1 : f.height = 18)
  (h2 : f.lower_base_area = 144 * Real.pi)
  (h3 : f.upper_base_area = 16 * Real.pi) :
  ∃ (smaller_cone_height : ℝ), smaller_cone_height = 9 := by
  sorry

end frustum_smaller_cone_height_l2687_268759


namespace least_common_addition_primes_l2687_268729

theorem least_common_addition_primes (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x < y → x + y = 36 → 4 * x + y = 51 := by
  sorry

end least_common_addition_primes_l2687_268729


namespace problem_1_problem_2_l2687_268783

-- Problem 1
theorem problem_1 : 
  (3 * Real.sqrt 18 + (1/6) * Real.sqrt 72 - 4 * Real.sqrt (1/8)) / (4 * Real.sqrt 2) = 9/4 := by
  sorry

-- Problem 2
theorem problem_2 : 
  let x : ℝ := Real.sqrt 2 + 1
  ((x + 2) / (x * (x - 1)) - 1 / (x - 1)) * (x / (x - 1)) = 1 := by
  sorry

end problem_1_problem_2_l2687_268783


namespace sixty_degrees_in_clerts_l2687_268773

/-- The number of clerts in a full circle on Venus -/
def venus_full_circle : ℚ := 800

/-- The number of degrees in a full circle on Earth -/
def earth_full_circle : ℚ := 360

/-- Converts degrees to clerts on Venus -/
def degrees_to_clerts (degrees : ℚ) : ℚ :=
  (degrees / earth_full_circle) * venus_full_circle

/-- Theorem: 60 degrees is equivalent to 133.3 (repeating) clerts on Venus -/
theorem sixty_degrees_in_clerts :
  degrees_to_clerts 60 = 133 + 1/3 := by sorry

end sixty_degrees_in_clerts_l2687_268773


namespace equilateral_triangle_area_l2687_268766

/-- The area of an equilateral triangle with altitude √15 is 5√3 -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = Real.sqrt 15) :
  let base := 2 * Real.sqrt 5
  let area := (1 / 2) * base * h
  area = 5 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_area_l2687_268766


namespace quadratic_inequality_solution_l2687_268706

/-- Given a quadratic inequality ax² + bx + 1 > 0 with solution set {x | -1 < x < 1/3},
    prove that ab = 6 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, (a * x^2 + b * x + 1 > 0) ↔ (-1 < x ∧ x < 1/3)) → 
  a * b = 6 := by
sorry

end quadratic_inequality_solution_l2687_268706


namespace l_shaped_playground_area_l2687_268749

def large_rectangle_length : ℕ := 10
def large_rectangle_width : ℕ := 7
def small_rectangle_length : ℕ := 3
def small_rectangle_width : ℕ := 2
def num_small_rectangles : ℕ := 2

theorem l_shaped_playground_area :
  (large_rectangle_length * large_rectangle_width) -
  (num_small_rectangles * small_rectangle_length * small_rectangle_width) = 58 := by
  sorry

end l_shaped_playground_area_l2687_268749


namespace value_of_x_l2687_268713

theorem value_of_x (x y z a b c : ℝ) 
  (ha : x * y / (x + y) = a)
  (hb : x * z / (x + z) = b)
  (hc : y * z / (y + z) = c)
  (ha_nonzero : a ≠ 0)
  (hb_nonzero : b ≠ 0)
  (hc_nonzero : c ≠ 0) :
  x = 2 * a * b * c / (a * c + b * c - a * b) :=
by sorry

end value_of_x_l2687_268713


namespace abs_neg_three_squared_plus_four_l2687_268730

theorem abs_neg_three_squared_plus_four : |-3^2 + 4| = 5 := by
  sorry

end abs_neg_three_squared_plus_four_l2687_268730


namespace certain_number_problem_l2687_268774

theorem certain_number_problem (x : ℝ) : (((x + 10) * 7) / 5) - 5 = 88 / 2 → x = 25 := by
  sorry

end certain_number_problem_l2687_268774


namespace stone_splitting_game_winner_l2687_268709

/-- The stone-splitting game -/
def StoneSplittingGame (n : ℕ) : Prop :=
  ∃ (winner : Bool),
    (winner = true → n.Prime ∨ ∃ k, n = 2^k) ∧
    (winner = false → ¬(n.Prime ∨ ∃ k, n = 2^k))

/-- Theorem: Characterization of winning conditions in the stone-splitting game -/
theorem stone_splitting_game_winner (n : ℕ) :
  StoneSplittingGame n ↔ (n.Prime ∨ ∃ k, n = 2^k) := by sorry

end stone_splitting_game_winner_l2687_268709


namespace complementary_angles_difference_l2687_268797

theorem complementary_angles_difference (x : ℝ) (h1 : 4 * x + x = 90) (h2 : x > 0) : |4 * x - x| = 54 := by
  sorry

end complementary_angles_difference_l2687_268797


namespace calcium_phosphate_yield_l2687_268731

/-- Represents the coefficients of a chemical reaction --/
structure ReactionCoefficients where
  fe2o3 : ℚ
  caco3 : ℚ
  ca3po42 : ℚ

/-- Represents the available moles of reactants --/
structure AvailableMoles where
  fe2o3 : ℚ
  caco3 : ℚ

/-- Calculates the theoretical yield of Ca3(PO4)2 based on the balanced reaction and available moles --/
def theoreticalYield (coeff : ReactionCoefficients) (available : AvailableMoles) : ℚ :=
  min 
    (available.fe2o3 * coeff.ca3po42 / coeff.fe2o3)
    (available.caco3 * coeff.ca3po42 / coeff.caco3)

/-- Theorem stating the theoretical yield of Ca3(PO4)2 for the given reaction and available moles --/
theorem calcium_phosphate_yield : 
  let coeff : ReactionCoefficients := ⟨2, 6, 3⟩
  let available : AvailableMoles := ⟨4, 10⟩
  theoreticalYield coeff available = 5 := by
  sorry

end calcium_phosphate_yield_l2687_268731


namespace circumradius_right_triangle_l2687_268761

/-- The radius of the circumscribed circle of a right triangle with sides 10, 8, and 6 is 5 -/
theorem circumradius_right_triangle : 
  ∀ (a b c : ℝ), 
    a = 10 → b = 8 → c = 6 →
    a^2 = b^2 + c^2 →
    (a / 2 : ℝ) = 5 :=
by sorry

end circumradius_right_triangle_l2687_268761


namespace power_three_mod_eleven_l2687_268771

theorem power_three_mod_eleven : 3^2048 % 11 = 5 := by
  sorry

end power_three_mod_eleven_l2687_268771


namespace min_value_and_inequality_l2687_268705

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- State the theorem
theorem min_value_and_inequality :
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a ∧ ∃ (x₀ : ℝ), f x₀ = a) ∧
  (∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → p + q + r = 3 → p^2 + q^2 + r^2 ≥ 3) :=
by sorry

end min_value_and_inequality_l2687_268705


namespace marble_count_l2687_268776

theorem marble_count (yellow : ℕ) (blue : ℕ) (red : ℕ) : 
  yellow = 5 →
  blue * 4 = red * 3 →
  red = yellow + 3 →
  yellow + blue + red = 19 := by
sorry

end marble_count_l2687_268776


namespace colonization_theorem_l2687_268756

/-- Represents the number of different combinations of planets that can be colonized --/
def colonization_combinations (total_planets : ℕ) (earth_like : ℕ) (mars_like : ℕ) 
  (earth_effort : ℕ) (mars_effort : ℕ) (total_effort : ℕ) : ℕ :=
  (Finset.range (earth_like + 1)).sum (fun a =>
    if 2 * a ≤ total_effort ∧ (total_effort - 2 * a) % 2 = 0 ∧ (total_effort - 2 * a) / 2 ≤ mars_like
    then Nat.choose earth_like a * Nat.choose mars_like ((total_effort - 2 * a) / 2)
    else 0)

/-- The main theorem stating the number of colonization combinations --/
theorem colonization_theorem : 
  colonization_combinations 15 7 8 2 1 16 = 1141 := by sorry

end colonization_theorem_l2687_268756


namespace last_two_digits_product_l2687_268742

def last_two_digits (n : ℤ) : ℤ × ℤ :=
  let tens := (n / 10) % 10
  let ones := n % 10
  (tens, ones)

def sum_last_two_digits (n : ℤ) : ℤ :=
  let (tens, ones) := last_two_digits n
  tens + ones

def product_last_two_digits (n : ℤ) : ℤ :=
  let (tens, ones) := last_two_digits n
  tens * ones

theorem last_two_digits_product (n : ℤ) :
  n % 5 = 0 → sum_last_two_digits n = 14 → product_last_two_digits n = 45 := by
  sorry

end last_two_digits_product_l2687_268742


namespace intersection_complement_equality_l2687_268718

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {1, 2}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {3, 4} := by sorry

end intersection_complement_equality_l2687_268718


namespace stratified_sample_size_l2687_268792

/-- Given a stratified sample where the ratio of product A to total production
    is 1/5 and 18 products of type A are sampled, prove the total sample size is 90. -/
theorem stratified_sample_size (sample_A : ℕ) (ratio_A : ℚ) (total_sample : ℕ) :
  sample_A = 18 →
  ratio_A = 1 / 5 →
  (sample_A : ℚ) / total_sample = ratio_A →
  total_sample = 90 := by
  sorry

end stratified_sample_size_l2687_268792


namespace smallest_number_with_remainders_l2687_268733

theorem smallest_number_with_remainders : ∃! x : ℕ, 
  x > 0 ∧
  x % 45 = 4 ∧
  x % 454 = 45 ∧
  x % 4545 = 454 ∧
  x % 45454 = 4545 ∧
  ∀ y : ℕ, y > 0 ∧ y % 45 = 4 ∧ y % 454 = 45 ∧ y % 4545 = 454 ∧ y % 45454 = 4545 → x ≤ y :=
by
  -- Proof goes here
  sorry

end smallest_number_with_remainders_l2687_268733


namespace pages_revised_twice_l2687_268787

/-- Represents the manuscript typing scenario -/
structure ManuscriptTyping where
  totalPages : Nat
  revisedOnce : Nat
  revisedTwice : Nat
  firstTypingCost : Nat
  revisionCost : Nat
  totalCost : Nat

/-- Calculates the total cost of typing and revising a manuscript -/
def calculateTotalCost (m : ManuscriptTyping) : Nat :=
  m.firstTypingCost * m.totalPages + 
  m.revisionCost * m.revisedOnce + 
  2 * m.revisionCost * m.revisedTwice

/-- Theorem stating that given the specified conditions, 30 pages were revised twice -/
theorem pages_revised_twice (m : ManuscriptTyping) 
  (h1 : m.totalPages = 100)
  (h2 : m.revisedOnce = 20)
  (h3 : m.firstTypingCost = 10)
  (h4 : m.revisionCost = 5)
  (h5 : m.totalCost = 1400)
  (h6 : calculateTotalCost m = m.totalCost) :
  m.revisedTwice = 30 := by
  sorry

end pages_revised_twice_l2687_268787


namespace sum_of_divisors_900_prime_factors_l2687_268782

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_900_prime_factors :
  ∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  sum_of_divisors 900 = p * q * r ∧
  ∀ (s : ℕ), Nat.Prime s → s ∣ sum_of_divisors 900 → (s = p ∨ s = q ∨ s = r) :=
sorry

end sum_of_divisors_900_prime_factors_l2687_268782


namespace road_travel_cost_l2687_268722

/-- Calculate the cost of traveling two perpendicular roads on a rectangular lawn -/
theorem road_travel_cost (lawn_length lawn_width road_width cost_per_sqm : ℝ) :
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 15 ∧ 
  cost_per_sqm = 3 →
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * cost_per_sqm = 5625 := by
  sorry


end road_travel_cost_l2687_268722


namespace simplify_fraction_l2687_268736

theorem simplify_fraction (a b c : ℕ) (h : b = a * c) :
  (a : ℚ) / b * c = 1 :=
by sorry

end simplify_fraction_l2687_268736


namespace original_class_size_l2687_268789

theorem original_class_size (x : ℕ) : 
  (x * 40 + 12 * 32) / (x + 12) = 36 → x = 12 := by
  sorry

end original_class_size_l2687_268789


namespace inequality_proof_l2687_268703

theorem inequality_proof (a b x : ℝ) (h : 0 ≤ x ∧ x < Real.pi / 2) :
  a^2 * Real.tan x * (Real.cos x)^(1/3) + b^2 * Real.sin x ≥ 2 * x * a * b :=
by sorry

end inequality_proof_l2687_268703


namespace asymptote_angle_is_90_degrees_l2687_268716

/-- A hyperbola with equation x^2 - y^2/b^2 = 1 (b > 0) and eccentricity √2 -/
structure Hyperbola where
  b : ℝ
  h_b_pos : b > 0
  h_ecc : Real.sqrt 2 = Real.sqrt (1 + 1 / b^2)

/-- The angle between the asymptotes of the hyperbola -/
def asymptote_angle (h : Hyperbola) : ℝ := sorry

/-- Theorem stating that the angle between the asymptotes is 90° -/
theorem asymptote_angle_is_90_degrees (h : Hyperbola) :
  asymptote_angle h = 90 * π / 180 := by sorry

end asymptote_angle_is_90_degrees_l2687_268716


namespace quadratic_roots_condition_l2687_268765

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 4*x - 1 = 0 ∧ a * y^2 - 4*y - 1 = 0) ↔ 
  (a > -4 ∧ a ≠ 0) := by sorry

end quadratic_roots_condition_l2687_268765


namespace antonov_candy_count_l2687_268702

/-- The number of candies in a pack -/
def candies_per_pack : ℕ := 20

/-- The number of packs Antonov has left -/
def packs_left : ℕ := 2

/-- The number of packs Antonov gave away -/
def packs_given : ℕ := 1

/-- The total number of candies Antonov bought initially -/
def total_candies : ℕ := (packs_left + packs_given) * candies_per_pack

theorem antonov_candy_count : total_candies = 60 := by sorry

end antonov_candy_count_l2687_268702


namespace tulip_fraction_l2687_268710

/-- Represents the composition of a bouquet of flowers -/
structure Bouquet where
  pink_lilies : ℝ
  red_lilies : ℝ
  pink_tulips : ℝ
  red_tulips : ℝ

/-- The fraction of tulips in a bouquet satisfying given conditions -/
theorem tulip_fraction (b : Bouquet) 
  (half_pink_lilies : b.pink_lilies = b.pink_tulips)
  (third_red_tulips : b.red_tulips = (1/3) * (b.red_lilies + b.red_tulips))
  (three_fifths_pink : b.pink_lilies + b.pink_tulips = (3/5) * (b.pink_lilies + b.red_lilies + b.pink_tulips + b.red_tulips)) :
  (b.pink_tulips + b.red_tulips) / (b.pink_lilies + b.red_lilies + b.pink_tulips + b.red_tulips) = 13/30 := by
  sorry

#check tulip_fraction

end tulip_fraction_l2687_268710


namespace fractional_sides_eq_seven_l2687_268796

/-- A 3-dimensional polyhedron with fractional sides -/
structure Polyhedron where
  F : ℝ  -- number of fractional sides
  D : ℝ  -- number of diagonals
  h1 : D = 2 * F
  h2 : D = F * (F - 3) / 2

/-- The number of fractional sides in the polyhedron is 7 -/
theorem fractional_sides_eq_seven (P : Polyhedron) : P.F = 7 := by
  sorry

end fractional_sides_eq_seven_l2687_268796


namespace custom_op_example_l2687_268708

-- Define the custom operation ※
def custom_op (a b : ℚ) : ℚ := 4 * b - a

-- Theorem statement
theorem custom_op_example : custom_op (custom_op (-1) 3) 2 = -5 := by
  sorry

end custom_op_example_l2687_268708


namespace digit_append_theorem_l2687_268712

theorem digit_append_theorem (n : ℕ) : 
  (∃ d : ℕ, d ≤ 9 ∧ 10 * n + d = 13 * n) ↔ (n = 1 ∨ n = 2 ∨ n = 3) :=
sorry

end digit_append_theorem_l2687_268712


namespace stairs_problem_l2687_268743

/-- Calculates the number of steps climbed given the number of flights, height per flight, and step height. -/
def steps_climbed (flights : ℕ) (flight_height : ℕ) (step_height : ℕ) : ℕ :=
  (flights * flight_height * 12) / step_height

/-- Theorem: Given 9 flights of stairs, with each flight being 10 feet, and each step being 18 inches, 
    the total number of steps climbed is 60. -/
theorem stairs_problem : steps_climbed 9 10 18 = 60 := by
  sorry

end stairs_problem_l2687_268743


namespace drill_bits_total_cost_l2687_268714

/-- Calculates the total amount paid for drill bits including tax -/
theorem drill_bits_total_cost (num_sets : ℕ) (cost_per_set : ℚ) (tax_rate : ℚ) : 
  num_sets = 5 → cost_per_set = 6 → tax_rate = (1/10) → 
  num_sets * cost_per_set * (1 + tax_rate) = 33 := by
  sorry

end drill_bits_total_cost_l2687_268714


namespace twirly_tea_cups_capacity_l2687_268738

/-- The 'Twirly Tea Cups' ride capacity problem -/
theorem twirly_tea_cups_capacity 
  (people_per_teacup : ℕ) 
  (number_of_teacups : ℕ) 
  (h1 : people_per_teacup = 9)
  (h2 : number_of_teacups = 7) : 
  people_per_teacup * number_of_teacups = 63 := by
  sorry

end twirly_tea_cups_capacity_l2687_268738


namespace area_enclosed_by_line_and_curve_l2687_268721

def dataSet : List ℝ := [1, 2, 0, 0, 8, 7, 6, 5]

def median (l : List ℝ) : ℝ := sorry

def areaEnclosed (a : ℝ) : ℝ := sorry

theorem area_enclosed_by_line_and_curve (a : ℝ) :
  a ∈ dataSet →
  median dataSet = 4 →
  areaEnclosed a = 9/2 := by sorry

end area_enclosed_by_line_and_curve_l2687_268721


namespace min_rectangles_cover_l2687_268772

/-- A point in the unit square -/
structure Point where
  x : Real
  y : Real
  x_in_unit : 0 < x ∧ x < 1
  y_in_unit : 0 < y ∧ y < 1

/-- A rectangle with sides parallel to the unit square -/
structure Rectangle where
  left : Real
  right : Real
  bottom : Real
  top : Real
  valid : 0 ≤ left ∧ left < right ∧ right ≤ 1 ∧
          0 ≤ bottom ∧ bottom < top ∧ top ≤ 1

/-- The theorem statement -/
theorem min_rectangles_cover (n : Nat) (S : Finset Point) :
  S.card = n →
  ∃ (k : Nat) (R : Finset Rectangle),
    R.card = k ∧
    (∀ p ∈ S, ∀ r ∈ R, ¬(r.left < p.x ∧ p.x < r.right ∧ r.bottom < p.y ∧ p.y < r.top)) ∧
    (∀ x y : Real, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 →
      (∀ p ∈ S, p.x ≠ x ∨ p.y ≠ y) →
      ∃ r ∈ R, r.left < x ∧ x < r.right ∧ r.bottom < y ∧ y < r.top) ∧
    k = 2 * n + 2 ∧
    (∀ m : Nat, m < k →
      ¬∃ (R' : Finset Rectangle),
        R'.card = m ∧
        (∀ p ∈ S, ∀ r ∈ R', ¬(r.left < p.x ∧ p.x < r.right ∧ r.bottom < p.y ∧ p.y < r.top)) ∧
        (∀ x y : Real, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 →
          (∀ p ∈ S, p.x ≠ x ∨ p.y ≠ y) →
          ∃ r ∈ R', r.left < x ∧ x < r.right ∧ r.bottom < y ∧ y < r.top)) :=
by sorry

end min_rectangles_cover_l2687_268772


namespace remainder_4_100_div_9_l2687_268739

theorem remainder_4_100_div_9 : (4^100) % 9 = 4 := by sorry

end remainder_4_100_div_9_l2687_268739


namespace evaluate_expression_l2687_268701

theorem evaluate_expression : -(18 / 3 * 7^2 - 80 + 4 * 7) = -242 := by sorry

end evaluate_expression_l2687_268701


namespace ratio_odd_even_divisors_M_l2687_268754

def M : ℕ := 36 * 36 * 98 * 150

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors_M :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 62 := by sorry

end ratio_odd_even_divisors_M_l2687_268754


namespace fraction_meaningful_iff_not_neg_one_l2687_268785

theorem fraction_meaningful_iff_not_neg_one (a : ℝ) :
  (∃ (x : ℝ), x = 2 / (a + 1)) ↔ a ≠ -1 := by
  sorry

end fraction_meaningful_iff_not_neg_one_l2687_268785


namespace two_digit_number_theorem_l2687_268747

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Given a two-digit number, insert_zero inserts a 0 between its digits. -/
def insert_zero (n : ℕ) : ℕ := (n / 10) * 100 + (n % 10)

/-- The set of numbers that satisfy the condition in the problem. -/
def solution_set : Set ℕ := {80, 81, 82, 83, 84, 85, 86, 87, 88, 89}

/-- The main theorem that proves the solution to the problem. -/
theorem two_digit_number_theorem (n : ℕ) : 
  TwoDigitNumber n → (insert_zero n = n + 720) → n ∈ solution_set := by
  sorry

end two_digit_number_theorem_l2687_268747


namespace expression_evaluation_l2687_268786

theorem expression_evaluation : 72 + (150 / 25) + (16 * 19) - 250 - (450 / 9) = 82 := by
  sorry

end expression_evaluation_l2687_268786


namespace water_remaining_after_four_replacements_l2687_268724

/-- Represents the fraction of original water remaining after a number of replacements -/
def water_remaining (initial_water : ℚ) (tank_capacity : ℚ) (replacement_volume : ℚ) (n : ℕ) : ℚ :=
  (1 - replacement_volume / tank_capacity) ^ n * initial_water / tank_capacity

/-- Theorem stating the fraction of original water remaining after 4 replacements -/
theorem water_remaining_after_four_replacements : 
  water_remaining 10 20 5 4 = 81 / 256 := by
  sorry

end water_remaining_after_four_replacements_l2687_268724


namespace cannon_hit_probability_l2687_268726

theorem cannon_hit_probability (P1 P2 P3 : ℝ) 
  (h1 : P2 = 0.2)
  (h2 : P3 = 0.3)
  (h3 : (1 - P1) * (1 - P2) * (1 - P3) = 0.28) :
  P1 = 0.5 := by
  sorry

end cannon_hit_probability_l2687_268726


namespace cats_awake_l2687_268788

theorem cats_awake (total : ℕ) (asleep : ℕ) (h1 : total = 98) (h2 : asleep = 92) :
  total - asleep = 6 := by
  sorry

end cats_awake_l2687_268788


namespace car_dealership_shipment_l2687_268707

theorem car_dealership_shipment 
  (initial_cars : ℕ) 
  (initial_silver_percent : ℚ)
  (new_shipment_nonsilver_percent : ℚ)
  (final_silver_percent : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percent = 1/5)
  (h3 : new_shipment_nonsilver_percent = 7/20)
  (h4 : final_silver_percent = 3/10)
  : ∃ (new_shipment : ℕ), 
    (initial_silver_percent * initial_cars + (1 - new_shipment_nonsilver_percent) * new_shipment) / 
    (initial_cars + new_shipment) = final_silver_percent ∧ 
    new_shipment = 11 :=
sorry

end car_dealership_shipment_l2687_268707


namespace polygon_sides_with_45_degree_exterior_angles_l2687_268715

theorem polygon_sides_with_45_degree_exterior_angles :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    exterior_angle = 45 →
    (n : ℝ) * exterior_angle = 360 →
    n = 8 := by
  sorry

end polygon_sides_with_45_degree_exterior_angles_l2687_268715
