import Mathlib

namespace NUMINAMATH_CALUDE_unique_modular_inverse_l1812_181257

theorem unique_modular_inverse (p : Nat) (a : Nat) (h_p : p.Prime) (h_p_odd : p % 2 = 1)
  (h_a_range : 2 ≤ a ∧ a ≤ p - 2) :
  ∃! i : Nat, 2 ≤ i ∧ i ≤ p - 2 ∧ i ≠ a ∧ (i * a) % p = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_inverse_l1812_181257


namespace NUMINAMATH_CALUDE_no_primitive_root_for_multiple_odd_primes_l1812_181237

theorem no_primitive_root_for_multiple_odd_primes (n : ℕ) 
  (h1 : ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ Odd p ∧ Odd q ∧ p ∣ n ∧ q ∣ n) : 
  ¬ ∃ (a : ℕ), IsPrimitiveRoot a n :=
sorry

end NUMINAMATH_CALUDE_no_primitive_root_for_multiple_odd_primes_l1812_181237


namespace NUMINAMATH_CALUDE_soccer_ball_selling_price_l1812_181253

theorem soccer_ball_selling_price 
  (num_balls : ℕ) 
  (cost_per_ball : ℚ) 
  (total_profit : ℚ) 
  (h1 : num_balls = 50)
  (h2 : cost_per_ball = 60)
  (h3 : total_profit = 1950) :
  (total_profit / num_balls + cost_per_ball : ℚ) = 99 := by
sorry

end NUMINAMATH_CALUDE_soccer_ball_selling_price_l1812_181253


namespace NUMINAMATH_CALUDE_truth_table_results_l1812_181277

variable (p q : Prop)

theorem truth_table_results :
  (∀ p, ¬(p ∧ ¬p)) ∧
  (∀ p, p ∨ ¬p) ∧
  (∀ p q, ¬(p ∧ q) ↔ (¬p ∨ ¬q)) ∧
  (∀ p q, (p ∨ q) ∨ ¬p) :=
by sorry

end NUMINAMATH_CALUDE_truth_table_results_l1812_181277


namespace NUMINAMATH_CALUDE_distance_to_origin_l1812_181258

theorem distance_to_origin : let M : ℝ × ℝ := (-3, 4)
                             Real.sqrt ((M.1 - 0)^2 + (M.2 - 0)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1812_181258


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l1812_181245

theorem smallest_constant_inequality (x y z : ℝ) :
  ∃ D : ℝ, (∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + z^2 + 3 ≥ D * (x + y + z)) ∧
  D = -Real.sqrt (72 / 11) ∧
  ∀ E : ℝ, (∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + z^2 + 3 ≥ E * (x + y + z)) → D ≤ E :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l1812_181245


namespace NUMINAMATH_CALUDE_wheel_probability_l1812_181271

theorem wheel_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_C = 1/6 → 
  p_A + p_B + p_C + p_D = 1 →
  p_D = 1/4 := by sorry

end NUMINAMATH_CALUDE_wheel_probability_l1812_181271


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_11_l1812_181279

theorem smallest_digit_divisible_by_11 : 
  ∃ (d : Nat), d < 10 ∧ 
    (∀ (x : Nat), x < d → ¬(489000 + x * 100 + 7).ModEq 0 11) ∧
    (489000 + d * 100 + 7).ModEq 0 11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_11_l1812_181279


namespace NUMINAMATH_CALUDE_sugar_pack_weight_l1812_181213

/-- Given the total sugar, number of packs, and leftover sugar, calculates the weight of each pack. -/
def packWeight (totalSugar : ℕ) (numPacks : ℕ) (leftoverSugar : ℕ) : ℕ :=
  (totalSugar - leftoverSugar) / numPacks

/-- Proves that given 3020 grams of total sugar, 12 packs, and 20 grams of leftover sugar, 
    the weight of each pack is 250 grams. -/
theorem sugar_pack_weight :
  packWeight 3020 12 20 = 250 := by
  sorry

end NUMINAMATH_CALUDE_sugar_pack_weight_l1812_181213


namespace NUMINAMATH_CALUDE_milk_problem_l1812_181255

theorem milk_problem (initial_milk : ℚ) (rachel_fraction : ℚ) (joey_fraction : ℚ) :
  initial_milk = 3/4 →
  rachel_fraction = 1/3 →
  joey_fraction = 1/2 →
  joey_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_milk_problem_l1812_181255


namespace NUMINAMATH_CALUDE_reimbursement_difference_l1812_181283

/-- The problem of reimbursement in a group activity --/
theorem reimbursement_difference (tom emma harry : ℝ) : 
  tom = 95 →
  emma = 140 →
  harry = 165 →
  let total := tom + emma + harry
  let share := total / 3
  let t := share - tom
  let e := share - emma
  e - t = -45 := by
  sorry

end NUMINAMATH_CALUDE_reimbursement_difference_l1812_181283


namespace NUMINAMATH_CALUDE_tucker_tissues_left_l1812_181202

/-- The number of tissues Tucker has left -/
def tissues_left (tissues_per_box : ℕ) (boxes_bought : ℕ) (tissues_used : ℕ) : ℕ :=
  boxes_bought * tissues_per_box - tissues_used

/-- Theorem: Tucker has 270 tissues left -/
theorem tucker_tissues_left :
  tissues_left 160 3 210 = 270 := by
  sorry

end NUMINAMATH_CALUDE_tucker_tissues_left_l1812_181202


namespace NUMINAMATH_CALUDE_actual_tax_expectation_l1812_181246

/-- Represents the fraction of the population that are liars -/
def fraction_liars : ℝ := 0.1

/-- Represents the fraction of the population that are economists -/
def fraction_economists : ℝ := 1 - fraction_liars

/-- Represents the fraction of affirmative answers for raising taxes -/
def affirmative_taxes : ℝ := 0.4

/-- Represents the fraction of affirmative answers for increasing money supply -/
def affirmative_money : ℝ := 0.3

/-- Represents the fraction of affirmative answers for issuing bonds -/
def affirmative_bonds : ℝ := 0.5

/-- Represents the fraction of affirmative answers for spending gold reserves -/
def affirmative_gold : ℝ := 0

/-- The theorem stating that 30% of the population actually expects raising taxes -/
theorem actual_tax_expectation : 
  affirmative_taxes - fraction_liars = 0.3 := by sorry

end NUMINAMATH_CALUDE_actual_tax_expectation_l1812_181246


namespace NUMINAMATH_CALUDE_systematic_sampling_removal_l1812_181217

/-- The number of students that need to be initially removed in a systematic sampling -/
def studentsToRemove (totalStudents sampleSize : ℕ) : ℕ :=
  totalStudents % sampleSize

theorem systematic_sampling_removal (totalStudents sampleSize : ℕ) 
  (h1 : totalStudents = 1387)
  (h2 : sampleSize = 9)
  (h3 : sampleSize > 0) :
  studentsToRemove totalStudents sampleSize = 1 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_removal_l1812_181217


namespace NUMINAMATH_CALUDE_sod_area_calculation_sod_area_is_9474_l1812_181224

/-- Calculates the area of sod needed for Jill's front yard -/
theorem sod_area_calculation (front_yard_width front_yard_length sidewalk_width sidewalk_length
                              flowerbed1_depth flowerbed1_length flowerbed2_width flowerbed2_length
                              flowerbed3_width flowerbed3_length : ℕ) : ℕ :=
  let front_yard_area := front_yard_width * front_yard_length
  let sidewalk_area := sidewalk_width * sidewalk_length
  let flowerbed1_area := 2 * (flowerbed1_depth * flowerbed1_length)
  let flowerbed2_area := flowerbed2_width * flowerbed2_length
  let flowerbed3_area := flowerbed3_width * flowerbed3_length
  let total_subtract_area := sidewalk_area + flowerbed1_area + flowerbed2_area + flowerbed3_area
  front_yard_area - total_subtract_area

/-- Proves that the area of sod needed for Jill's front yard is 9,474 square feet -/
theorem sod_area_is_9474 :
  sod_area_calculation 200 50 3 50 4 25 10 12 7 8 = 9474 := by
  sorry

end NUMINAMATH_CALUDE_sod_area_calculation_sod_area_is_9474_l1812_181224


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l1812_181227

/-- Given a cylinder with a square cross-section of area 4, its lateral area is 4π. -/
theorem cylinder_lateral_area (r h : ℝ) : 
  r * r = 4 → 2 * π * r * h = 4 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l1812_181227


namespace NUMINAMATH_CALUDE_first_year_balance_l1812_181209

/-- Proves that the total balance at the end of the first year is $5500,
    given the initial deposit of $5000 and the interest accrued in the first year of $500. -/
theorem first_year_balance (initial_deposit : ℝ) (interest_first_year : ℝ) 
  (h1 : initial_deposit = 5000)
  (h2 : interest_first_year = 500) :
  initial_deposit + interest_first_year = 5500 := by
  sorry

end NUMINAMATH_CALUDE_first_year_balance_l1812_181209


namespace NUMINAMATH_CALUDE_new_oranges_added_l1812_181216

theorem new_oranges_added (initial : ℕ) (thrown_away : ℕ) (final : ℕ) : 
  initial = 31 → thrown_away = 9 → final = 60 → final - (initial - thrown_away) = 38 := by
  sorry

end NUMINAMATH_CALUDE_new_oranges_added_l1812_181216


namespace NUMINAMATH_CALUDE_range_of_a_l1812_181286

theorem range_of_a (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 1) (ha : ∀ a : ℝ, a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) :
  ∃ S : Set ℝ, S = Set.Ioo 0 (7 / 27) ∪ {7 / 27} ∧ 
  (∀ a : ℝ, a ∈ S ↔ ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧
    a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1812_181286


namespace NUMINAMATH_CALUDE_largest_negative_angle_l1812_181269

-- Define the function for angles with the same terminal side as -2002°
def sameTerminalSide (k : ℤ) : ℝ := k * 360 - 2002

-- Theorem statement
theorem largest_negative_angle :
  ∃ (k : ℤ), sameTerminalSide k = -202 ∧
  ∀ (m : ℤ), sameTerminalSide m < 0 → sameTerminalSide m ≤ -202 :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_angle_l1812_181269


namespace NUMINAMATH_CALUDE_gcd_456_357_l1812_181234

theorem gcd_456_357 : Nat.gcd 456 357 = 3 := by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_gcd_456_357_l1812_181234


namespace NUMINAMATH_CALUDE_gardener_hourly_rate_l1812_181259

/-- Gardening project cost calculation -/
theorem gardener_hourly_rate 
  (num_rose_bushes : ℕ) 
  (cost_per_rose_bush : ℚ) 
  (hours_per_day : ℕ) 
  (num_days : ℕ) 
  (soil_volume : ℕ) 
  (cost_per_cubic_foot : ℚ) 
  (total_project_cost : ℚ) : 
  num_rose_bushes = 20 →
  cost_per_rose_bush = 150 →
  hours_per_day = 5 →
  num_days = 4 →
  soil_volume = 100 →
  cost_per_cubic_foot = 5 →
  total_project_cost = 4100 →
  (total_project_cost - (num_rose_bushes * cost_per_rose_bush + soil_volume * cost_per_cubic_foot)) / (hours_per_day * num_days) = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_gardener_hourly_rate_l1812_181259


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1812_181239

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f (2 * x + y) + 5 * x * y = f (3 * x - y) + 2 * x^2 + 1

/-- The main theorem stating that any function satisfying the functional equation
    must have f(10) = -49 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 10 = -49 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1812_181239


namespace NUMINAMATH_CALUDE_backyard_area_l1812_181256

/-- Represents a rectangular backyard with specific walking properties. -/
structure Backyard where
  length : ℝ
  width : ℝ
  total_distance : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ
  length_covers_total : length * length_walks = total_distance
  perimeter_covers_total : (2 * length + 2 * width) * perimeter_walks = total_distance

/-- The theorem stating the area of the backyard with given properties. -/
theorem backyard_area (b : Backyard) (h1 : b.total_distance = 2000)
    (h2 : b.length_walks = 50) (h3 : b.perimeter_walks = 20) :
    b.length * b.width = 400 := by
  sorry


end NUMINAMATH_CALUDE_backyard_area_l1812_181256


namespace NUMINAMATH_CALUDE_crayons_lost_or_given_away_l1812_181297

/-- Given that Paul gave away 213 crayons and lost 16 crayons,
    prove that the total number of crayons lost or given away is 229. -/
theorem crayons_lost_or_given_away :
  let crayons_given_away : ℕ := 213
  let crayons_lost : ℕ := 16
  crayons_given_away + crayons_lost = 229 := by
  sorry

end NUMINAMATH_CALUDE_crayons_lost_or_given_away_l1812_181297


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1812_181261

theorem negation_of_proposition (x y : ℝ) :
  ¬(((x - 1)^2 + (y - 2)^2 = 0) → (x = 1 ∧ y = 2)) ↔
  (((x - 1)^2 + (y - 2)^2 ≠ 0) → (x ≠ 1 ∨ y ≠ 2)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1812_181261


namespace NUMINAMATH_CALUDE_octagon_circle_circumference_l1812_181265

/-- The circumference of a circle containing an inscribed regular octagon -/
theorem octagon_circle_circumference (side_length : ℝ) (h : side_length = 5) :
  ∃ (circumference : ℝ), circumference = (5 * π) / Real.sin (22.5 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_octagon_circle_circumference_l1812_181265


namespace NUMINAMATH_CALUDE_farm_animals_problem_l1812_181212

theorem farm_animals_problem :
  ∃! (s c : ℕ), s > 0 ∧ c > 0 ∧ 28 * s + 27 * c = 1200 ∧ c > s :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_problem_l1812_181212


namespace NUMINAMATH_CALUDE_alley_width_l1812_181276

/-- The width of an alley given a ladder's two configurations -/
theorem alley_width (L : ℝ) (k h : ℝ) : ∃ w : ℝ,
  (k = L / 2) →
  (h = L * Real.sqrt 3 / 2) →
  (w^2 + (L/2)^2 = L^2) →
  (w^2 + (L * Real.sqrt 3 / 2)^2 = L^2) →
  w = L * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_alley_width_l1812_181276


namespace NUMINAMATH_CALUDE_cookies_per_batch_l1812_181254

/-- Given a bag of chocolate chips with 81 chips, used to make 3 batches of cookies,
    where each cookie contains 9 chips, prove that there are 3 cookies in each batch. -/
theorem cookies_per_batch (total_chips : ℕ) (num_batches : ℕ) (chips_per_cookie : ℕ) 
  (h1 : total_chips = 81)
  (h2 : num_batches = 3)
  (h3 : chips_per_cookie = 9)
  (h4 : total_chips % num_batches = 0)
  (h5 : (total_chips / num_batches) % chips_per_cookie = 0) :
  (total_chips / num_batches) / chips_per_cookie = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_batch_l1812_181254


namespace NUMINAMATH_CALUDE_cube_skew_pairs_l1812_181221

/-- A cube with 8 vertices and 28 lines passing through any two vertices -/
structure Cube :=
  (vertices : Nat)
  (lines : Nat)
  (h_vertices : vertices = 8)
  (h_lines : lines = 28)

/-- The number of sets of 4 points not in the same plane in the cube -/
def sets_of_four_points (c : Cube) : Nat := 58

/-- The number of pairs of skew lines contributed by each set of 4 points -/
def skew_pairs_per_set : Nat := 3

/-- The total number of pairs of skew lines in the cube -/
def total_skew_pairs (c : Cube) : Nat :=
  (sets_of_four_points c) * skew_pairs_per_set

/-- Theorem: The number of pairs of skew lines in the cube is 174 -/
theorem cube_skew_pairs (c : Cube) : total_skew_pairs c = 174 := by
  sorry

end NUMINAMATH_CALUDE_cube_skew_pairs_l1812_181221


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l1812_181215

/-- Given two employees A and B who are paid a total of 580 per week, 
    with B being paid 232 per week, prove that the percentage of A's pay 
    compared to B's pay is 150%. -/
theorem employee_pay_percentage (total_pay b_pay a_pay : ℚ) : 
  total_pay = 580 → 
  b_pay = 232 → 
  a_pay = total_pay - b_pay →
  (a_pay / b_pay) * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l1812_181215


namespace NUMINAMATH_CALUDE_dans_remaining_green_marbles_l1812_181219

def dans_initial_green_marbles : ℕ := 32
def mikes_taken_green_marbles : ℕ := 23

theorem dans_remaining_green_marbles :
  dans_initial_green_marbles - mikes_taken_green_marbles = 9 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_green_marbles_l1812_181219


namespace NUMINAMATH_CALUDE_m_eq_one_sufficient_not_necessary_l1812_181294

/-- A function f(x) = ax^2 is a power function if a = 1 -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a = 1 ∧ ∀ x, f x = a * x^2

/-- The function f(x) = (m^2 - 4m + 4)x^2 -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ (m^2 - 4*m + 4) * x^2

/-- Theorem: m = 1 is sufficient but not necessary for f to be a power function -/
theorem m_eq_one_sufficient_not_necessary :
  (∃ m : ℝ, m = 1 → is_power_function (f m)) ∧
  ¬(∀ m : ℝ, is_power_function (f m) → m = 1) :=
by sorry

end NUMINAMATH_CALUDE_m_eq_one_sufficient_not_necessary_l1812_181294


namespace NUMINAMATH_CALUDE_equation_equivalence_l1812_181274

theorem equation_equivalence :
  ∀ b c : ℝ,
  (∀ x : ℝ, |x - 3| = 4 ↔ x^2 + b*x + c = 0) →
  b = 1 ∧ c = -7 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1812_181274


namespace NUMINAMATH_CALUDE_returning_players_count_correct_returning_players_l1812_181238

theorem returning_players_count (new_players : ℕ) (group_size : ℕ) (num_groups : ℕ) : ℕ :=
  let total_players := group_size * num_groups
  total_players - new_players

theorem correct_returning_players :
  returning_players_count 4 5 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_returning_players_count_correct_returning_players_l1812_181238


namespace NUMINAMATH_CALUDE_train_distance_difference_l1812_181267

/-- Proves that the difference in distance traveled by two trains is 60 km -/
theorem train_distance_difference :
  ∀ (speed1 speed2 total_distance : ℝ),
  speed1 = 20 →
  speed2 = 25 →
  total_distance = 540 →
  ∃ (time : ℝ),
    time > 0 ∧
    speed1 * time + speed2 * time = total_distance ∧
    speed2 * time - speed1 * time = 60 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_difference_l1812_181267


namespace NUMINAMATH_CALUDE_probability_of_selecting_one_each_l1812_181262

/-- The probability of selecting one shirt, one pair of shorts, and one pair of socks
    when randomly choosing three items from a drawer containing 4 shirts, 5 pairs of shorts,
    and 6 pairs of socks. -/
theorem probability_of_selecting_one_each (num_shirts : ℕ) (num_shorts : ℕ) (num_socks : ℕ) :
  num_shirts = 4 →
  num_shorts = 5 →
  num_socks = 6 →
  (num_shirts * num_shorts * num_socks : ℚ) / (Nat.choose (num_shirts + num_shorts + num_socks) 3) = 24 / 91 := by
  sorry

#check probability_of_selecting_one_each

end NUMINAMATH_CALUDE_probability_of_selecting_one_each_l1812_181262


namespace NUMINAMATH_CALUDE_cos_double_angle_from_series_sum_l1812_181242

theorem cos_double_angle_from_series_sum (θ : ℝ) :
  (∑' n, (Real.cos θ) ^ (2 * n) = 8) → Real.cos (2 * θ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_from_series_sum_l1812_181242


namespace NUMINAMATH_CALUDE_system_solution_l1812_181249

theorem system_solution (a₁ a₂ b₁ b₂ : ℝ) :
  (∃ x y : ℝ, a₁ * x + b₁ * y = 21 ∧ a₂ * x + b₂ * y = 12 ∧ x = 3 ∧ y = 6) →
  (∃ m n : ℝ, a₁ * (2 * m + n) + b₁ * (m - n) = 21 ∧ 
              a₂ * (2 * m + n) + b₂ * (m - n) = 12 ∧
              m = 3 ∧ n = -3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1812_181249


namespace NUMINAMATH_CALUDE_simplify_expression_l1812_181280

theorem simplify_expression (x : ℝ) : (5 - 2*x^2) - (7 - 3*x^2) = -2 + x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1812_181280


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1812_181290

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)
def u (x : ℝ) : ℝ × ℝ := a + 2 • (b x)
def v (x : ℝ) : ℝ × ℝ := 2 • a - b x

theorem parallel_vectors_x_value :
  ∃ x : ℝ, (∃ k : ℝ, u x = k • (v x)) ∧ x = 1/2 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1812_181290


namespace NUMINAMATH_CALUDE_regions_in_circle_l1812_181228

/-- The number of regions created by radii and concentric circles in a larger circle -/
def num_regions (num_radii : ℕ) (num_circles : ℕ) : ℕ :=
  (num_circles + 1) * num_radii

/-- Theorem: 16 radii and 10 concentric circles create 176 regions -/
theorem regions_in_circle (num_radii num_circles : ℕ) 
  (h1 : num_radii = 16) 
  (h2 : num_circles = 10) : 
  num_regions num_radii num_circles = 176 := by
  sorry

#eval num_regions 16 10

end NUMINAMATH_CALUDE_regions_in_circle_l1812_181228


namespace NUMINAMATH_CALUDE_remaining_fruits_count_l1812_181266

/-- Represents the number of fruits on each tree type -/
structure FruitTrees :=
  (apples : ℕ)
  (plums : ℕ)
  (pears : ℕ)
  (cherries : ℕ)

/-- Represents the fraction of fruits picked from each tree -/
structure PickedFractions :=
  (apples : ℚ)
  (plums : ℚ)
  (pears : ℚ)
  (cherries : ℚ)

def original_fruits : FruitTrees :=
  { apples := 180
  , plums := 60
  , pears := 120
  , cherries := 720 }

def picked_fractions : PickedFractions :=
  { apples := 3/5
  , plums := 2/3
  , pears := 3/4
  , cherries := 7/10 }

theorem remaining_fruits_count 
  (orig : FruitTrees) 
  (picked : PickedFractions) 
  (h1 : orig.apples = 3 * orig.plums)
  (h2 : orig.pears = 2 * orig.plums)
  (h3 : orig.cherries = 4 * orig.apples)
  (h4 : orig = original_fruits)
  (h5 : picked = picked_fractions) :
  (orig.apples - (picked.apples * orig.apples).num) +
  (orig.plums - (picked.plums * orig.plums).num) +
  (orig.pears - (picked.pears * orig.pears).num) +
  (orig.cherries - (picked.cherries * orig.cherries).num) = 338 :=
by sorry

end NUMINAMATH_CALUDE_remaining_fruits_count_l1812_181266


namespace NUMINAMATH_CALUDE_square_sum_from_means_l1812_181226

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 150) : 
  x^2 + y^2 = 1300 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l1812_181226


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1812_181248

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 12) (h2 : x - y = 20) : y = -4 :=
by sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1812_181248


namespace NUMINAMATH_CALUDE_event_ticket_revenue_l1812_181281

theorem event_ticket_revenue (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_total_tickets : total_tickets = 160)
  (h_total_revenue : total_revenue = 2400) :
  ∃ (full_price : ℕ) (half_price : ℕ) (price : ℕ),
    full_price + half_price = total_tickets ∧
    full_price * price + half_price * (price / 2) = total_revenue ∧
    full_price * price = 960 := by
  sorry

end NUMINAMATH_CALUDE_event_ticket_revenue_l1812_181281


namespace NUMINAMATH_CALUDE_sally_fruit_spending_l1812_181264

/-- The total amount Sally spent on fruit --/
def total_spent (peach_price_after_coupon : ℝ) (peach_coupon : ℝ) (cherry_price : ℝ) (apple_price : ℝ) (apple_discount_percent : ℝ) : ℝ :=
  let peach_price := peach_price_after_coupon + peach_coupon
  let peach_and_cherry := peach_price + cherry_price
  let apple_discount := apple_price * apple_discount_percent
  let apple_price_discounted := apple_price - apple_discount
  peach_and_cherry + apple_price_discounted

/-- Theorem stating the total amount Sally spent on fruit --/
theorem sally_fruit_spending :
  total_spent 12.32 3 11.54 20 0.15 = 43.86 := by
  sorry

#eval total_spent 12.32 3 11.54 20 0.15

end NUMINAMATH_CALUDE_sally_fruit_spending_l1812_181264


namespace NUMINAMATH_CALUDE_max_non_club_members_in_company_l1812_181207

/-- The maximum number of people who did not join any club in a company with 5 clubs -/
def max_non_club_members (total_people : ℕ) (club_a : ℕ) (club_b : ℕ) (club_c : ℕ) (club_d : ℕ) (club_e : ℕ) (c_and_d_overlap : ℕ) (d_and_e_overlap : ℕ) : ℕ :=
  total_people - (club_a + club_b + club_c + (club_d - c_and_d_overlap) + (club_e - d_and_e_overlap))

/-- Theorem stating the maximum number of non-club members in the given scenario -/
theorem max_non_club_members_in_company :
  max_non_club_members 120 25 34 21 16 10 8 4 = 26 :=
by sorry

end NUMINAMATH_CALUDE_max_non_club_members_in_company_l1812_181207


namespace NUMINAMATH_CALUDE_marks_radiator_cost_l1812_181232

/-- Calculates the total cost for Mark's car radiator replacement. -/
def total_cost (labor_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  labor_hours * hourly_rate + part_cost

/-- Proves that the total cost for Mark's car radiator replacement is $300. -/
theorem marks_radiator_cost :
  total_cost 2 75 150 = 300 := by
  sorry

end NUMINAMATH_CALUDE_marks_radiator_cost_l1812_181232


namespace NUMINAMATH_CALUDE_twice_slope_line_equation_l1812_181230

/-- Given a line L1: 2x + 3y + 3 = 0, prove that the line L2 passing through (1,0) 
    with a slope twice that of L1 has the equation 4x + 3y = 4. -/
theorem twice_slope_line_equation : 
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 2 * x + 3 * y + 3 = 0
  let m1 : ℝ := -2 / 3  -- slope of L1
  let m2 : ℝ := 2 * m1  -- slope of L2
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 3 * y = 4
  (∀ x y, L2 x y ↔ y - 0 = m2 * (x - 1)) ∧ L2 1 0 := by
  sorry


end NUMINAMATH_CALUDE_twice_slope_line_equation_l1812_181230


namespace NUMINAMATH_CALUDE_moon_distance_scientific_notation_l1812_181203

theorem moon_distance_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 384000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.84 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_moon_distance_scientific_notation_l1812_181203


namespace NUMINAMATH_CALUDE_a_range_l1812_181260

/-- Proposition p: For all real x, ax^2 + 2ax + 3 > 0 -/
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + 2 * a * x + 3 > 0

/-- Proposition q: There exists a real x such that x^2 + 2ax + a + 2 = 0 -/
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

/-- The theorem stating that if both p and q are true, then a is in the range [2, 3) -/
theorem a_range (a : ℝ) (hp : p a) (hq : q a) : a ∈ Set.Ici 2 ∩ Set.Iio 3 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l1812_181260


namespace NUMINAMATH_CALUDE_similar_triangles_count_l1812_181222

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A B C : Point)

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Represents an altitude of a triangle -/
structure Altitude :=
  (base apex foot : Point)

/-- Checks if a line is an altitude of a triangle -/
def isAltitude (alt : Altitude) (t : Triangle) : Prop := sorry

/-- Represents the intersection of two lines -/
def lineIntersection (p1 p2 q1 q2 : Point) : Point := sorry

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Main theorem -/
theorem similar_triangles_count 
  (ABC : Triangle) 
  (h_acute : isAcute ABC)
  (AL : Altitude)
  (h_AL : isAltitude AL ABC)
  (BM : Altitude)
  (h_BM : isAltitude BM ABC)
  (D : Point)
  (h_D : D = lineIntersection AL.foot BM.foot ABC.A ABC.B) :
  ∃ (pairs : List (Triangle × Triangle)), 
    (∀ (p : Triangle × Triangle), p ∈ pairs → areSimilar p.1 p.2) ∧ 
    pairs.length = 10 ∧
    (∀ (t1 t2 : Triangle), areSimilar t1 t2 → (t1, t2) ∈ pairs ∨ (t2, t1) ∈ pairs) :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_count_l1812_181222


namespace NUMINAMATH_CALUDE_parallel_distinct_iff_a_eq_3_l1812_181275

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ × ℝ → Prop
  line2 : ℝ × ℝ → Prop
  line1_def : ∀ x y, line1 (x, y) ↔ a * x + 2 * y + 3 * a = 0
  line2_def : ∀ x y, line2 (x, y) ↔ 3 * x + (a - 1) * y = a - 7

/-- The lines are parallel -/
def parallel (tl : TwoLines) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, tl.line1 (x, y) ↔ tl.line2 (k * x, k * y)

/-- The lines are distinct -/
def distinct (tl : TwoLines) : Prop :=
  ∃ p, tl.line1 p ∧ ¬tl.line2 p

/-- The main theorem -/
theorem parallel_distinct_iff_a_eq_3 (tl : TwoLines) :
  parallel tl ∧ distinct tl ↔ tl.a = 3 :=
sorry

end NUMINAMATH_CALUDE_parallel_distinct_iff_a_eq_3_l1812_181275


namespace NUMINAMATH_CALUDE_vermont_ads_clicked_l1812_181223

theorem vermont_ads_clicked (page1 page2 page3 page4 page5 page6 : ℕ) : 
  page1 = 18 →
  page2 = 2 * page1 →
  page3 = page2 + 32 →
  page4 = (5 * page2 + 4) / 8 →  -- Rounding up (5/8 * page2)
  page5 = page3 + 15 →
  page6 = page1 + page2 + page3 - 42 →
  ((3 * (page1 + page2 + page3 + page4 + page5 + page6) + 2) / 5 : ℕ) = 185 := by
  sorry

end NUMINAMATH_CALUDE_vermont_ads_clicked_l1812_181223


namespace NUMINAMATH_CALUDE_parabola_vertex_l1812_181282

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 18

/-- The x-coordinate of the vertex -/
def p : ℝ := -2

/-- The y-coordinate of the vertex -/
def q : ℝ := 10

/-- Theorem: The vertex of the parabola y = 2x^2 + 8x + 18 is at (-2, 10) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola p) ∧ parabola p = q := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1812_181282


namespace NUMINAMATH_CALUDE_angle_A_measure_l1812_181270

/-- Given a geometric configuration with connected angles, prove that angle A measures 70°. -/
theorem angle_A_measure (B C D : ℝ) (hB : B = 120) (hC : C = 30) (hD : D = 110) : ∃ A : ℝ,
  A = 70 ∧ 
  A + B + C = 180 ∧  -- Sum of angles at a point
  A + C + (D - C) = 180  -- Sum of angles in the triangle formed by A, C, and the complement of D
  := by sorry

end NUMINAMATH_CALUDE_angle_A_measure_l1812_181270


namespace NUMINAMATH_CALUDE_bacteria_growth_l1812_181201

theorem bacteria_growth (division_time : ℕ) (total_time : ℕ) (initial_count : ℕ) : 
  division_time = 20 → 
  total_time = 180 → 
  initial_count = 1 → 
  2 ^ (total_time / division_time) = 512 :=
by
  sorry

#check bacteria_growth

end NUMINAMATH_CALUDE_bacteria_growth_l1812_181201


namespace NUMINAMATH_CALUDE_fraction_division_addition_l1812_181288

theorem fraction_division_addition : (5 : ℚ) / 6 / ((9 : ℚ) / 10) + (1 : ℚ) / 15 = (402 : ℚ) / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_addition_l1812_181288


namespace NUMINAMATH_CALUDE_circle_radius_l1812_181285

/-- Given a circle with equation x^2 + y^2 + 2ax + 9 = 0 and center coordinates (5, 0), its radius is 4 -/
theorem circle_radius (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + 9 = 0 ↔ (x - 5)^2 + y^2 = 16) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1812_181285


namespace NUMINAMATH_CALUDE_no_roots_of_composite_l1812_181204

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- Theorem: If f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_roots_of_composite (b c : ℝ) :
  (∀ x : ℝ, f b c x ≠ x) →
  (∀ x : ℝ, f b c (f b c x) ≠ x) :=
by
  sorry

end NUMINAMATH_CALUDE_no_roots_of_composite_l1812_181204


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1812_181293

def binomialCoefficient (n k : ℕ) : ℕ := sorry

def constantTermInExpansion (n : ℕ) : ℤ :=
  (binomialCoefficient n (n - 2)) * ((-2) ^ 2)

theorem constant_term_expansion :
  constantTermInExpansion 6 = 60 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1812_181293


namespace NUMINAMATH_CALUDE_k_mod_8_l1812_181244

/-- An integer m covers 1998 if 1, 9, 9, 8 appear in this order as digits of m. -/
def covers_1998 (m : ℕ) : Prop := sorry

/-- k(n) is the number of positive integers that cover 1998 and have exactly n digits, all different from 0. -/
def k (n : ℕ) : ℕ := sorry

/-- The main theorem: k(n) is congruent to 1 modulo 8 for all n ≥ 5. -/
theorem k_mod_8 (n : ℕ) (h : n ≥ 5) : k n ≡ 1 [MOD 8] := by sorry

end NUMINAMATH_CALUDE_k_mod_8_l1812_181244


namespace NUMINAMATH_CALUDE_fourth_tree_height_l1812_181231

/-- Represents a row of trees with specific properties -/
structure TreeRow where
  tallestHeight : ℝ
  shortestHeight : ℝ
  angleTopLine : ℝ
  equalSpacing : Bool

/-- Calculates the height of the nth tree from the left -/
def heightOfNthTree (row : TreeRow) (n : ℕ) : ℝ :=
  sorry

/-- The main theorem stating the height of the 4th tree -/
theorem fourth_tree_height (row : TreeRow) 
  (h1 : row.tallestHeight = 2.8)
  (h2 : row.shortestHeight = 1.4)
  (h3 : row.angleTopLine = 45)
  (h4 : row.equalSpacing = true) :
  heightOfNthTree row 4 = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_tree_height_l1812_181231


namespace NUMINAMATH_CALUDE_parabolas_equal_if_equal_segments_l1812_181272

/-- Two non-parallel lines in the plane -/
structure NonParallelLines where
  l₁ : ℝ → ℝ
  l₂ : ℝ → ℝ
  not_parallel : l₁ ≠ l₂

/-- A parabola of the form f(x) = x² + px + q -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- The length of the segment cut by a parabola on a line -/
def segment_length (para : Parabola) (line : ℝ → ℝ) : ℝ := sorry

/-- Two parabolas cut equal segments on two non-parallel lines -/
def equal_segments (f₁ f₂ : Parabola) (lines : NonParallelLines) : Prop :=
  segment_length f₁ lines.l₁ = segment_length f₂ lines.l₁ ∧
  segment_length f₁ lines.l₂ = segment_length f₂ lines.l₂

/-- Main theorem: If two parabolas cut equal segments on two non-parallel lines, 
    then the parabolas are identical -/
theorem parabolas_equal_if_equal_segments (f₁ f₂ : Parabola) (lines : NonParallelLines) :
  equal_segments f₁ f₂ lines → f₁ = f₂ := by sorry

end NUMINAMATH_CALUDE_parabolas_equal_if_equal_segments_l1812_181272


namespace NUMINAMATH_CALUDE_line_slope_l1812_181289

/-- Given a line y = kx + 1 passing through points (4, b), (a, 5), and (a, b + 1),
    prove that k = 3/4 -/
theorem line_slope (k a b : ℝ) : 
  (b = 4 * k + 1) → 
  (5 = a * k + 1) → 
  (b + 1 = a * k + 1) → 
  k = 3/4 := by sorry

end NUMINAMATH_CALUDE_line_slope_l1812_181289


namespace NUMINAMATH_CALUDE_parabola_directrix_l1812_181299

-- Define a parabola with equation y^2 = 8x
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the directrix of a parabola
def directrix (x : ℝ) : Prop := x = -2

-- Theorem statement
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola x y → directrix x :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1812_181299


namespace NUMINAMATH_CALUDE_age_ratio_sandy_molly_l1812_181233

/-- Given that Sandy is 42 years old and Molly is 12 years older than Sandy,
    prove that the ratio of their ages is 7:9. -/
theorem age_ratio_sandy_molly :
  let sandy_age : ℕ := 42
  let molly_age : ℕ := sandy_age + 12
  (sandy_age : ℚ) / molly_age = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_sandy_molly_l1812_181233


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1812_181225

theorem inequality_equivalence (x : ℝ) :
  (3 ≤ |(x - 3)^2 - 4| ∧ |(x - 3)^2 - 4| ≤ 7) ↔ (3 - Real.sqrt 11 ≤ x ∧ x ≤ 3 + Real.sqrt 11) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1812_181225


namespace NUMINAMATH_CALUDE_coastal_analysis_uses_gis_l1812_181251

-- Define the available technologies
inductive CoastalAnalysisTechnology
  | GPS
  | GIS
  | RemoteSensing
  | GeographicInformationTechnology

-- Define the properties of the analysis
structure CoastalAnalysis where
  involves_sea_level_changes : Bool
  available_technologies : List CoastalAnalysisTechnology

-- Define the main technology used for the analysis
def main_technology_for_coastal_analysis (analysis : CoastalAnalysis) : CoastalAnalysisTechnology :=
  CoastalAnalysisTechnology.GIS

-- Theorem statement
theorem coastal_analysis_uses_gis (analysis : CoastalAnalysis) 
  (h1 : analysis.involves_sea_level_changes = true)
  (h2 : analysis.available_technologies.length ≥ 2) :
  main_technology_for_coastal_analysis analysis = CoastalAnalysisTechnology.GIS := by
  sorry

end NUMINAMATH_CALUDE_coastal_analysis_uses_gis_l1812_181251


namespace NUMINAMATH_CALUDE_function_inequality_l1812_181200

open Real

noncomputable def f (x : ℝ) : ℝ := x / cos x

theorem function_inequality (x₁ x₂ x₃ : ℝ) 
  (h₁ : |x₁| < π/2) (h₂ : |x₂| < π/2) (h₃ : |x₃| < π/2)
  (h₄ : f x₁ + f x₂ ≥ 0) (h₅ : f x₂ + f x₃ ≥ 0) (h₆ : f x₃ + f x₁ ≥ 0) :
  f (x₁ + x₂ + x₃) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1812_181200


namespace NUMINAMATH_CALUDE_problem_statement_l1812_181243

theorem problem_statement :
  (∃ (a b : ℝ), abs (a + b) < 1 ∧ abs a + abs b ≥ 1) ∧
  (∀ x : ℝ, (x ≤ -3 ∨ x ≥ 1) ↔ |x + 1| - 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1812_181243


namespace NUMINAMATH_CALUDE_odd_square_minus_one_divisible_by_eight_l1812_181298

theorem odd_square_minus_one_divisible_by_eight (n : ℤ) (h : Odd n) : 
  ∃ k : ℤ, n^2 - 1 = 8 * k := by
sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_divisible_by_eight_l1812_181298


namespace NUMINAMATH_CALUDE_sqrt_400_div_2_l1812_181250

theorem sqrt_400_div_2 : Real.sqrt 400 / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_400_div_2_l1812_181250


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l1812_181295

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the condition for points on the intersection line
def on_intersection_line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x - 2

-- Define the relation between points O, M, N, and D
def point_relation (xm ym xn yn xd yd t : ℝ) : Prop :=
  xm + xn = t * xd ∧ ym + yn = t * yd

theorem hyperbola_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_real_axis : 2 * a = 4 * Real.sqrt 3)
  (h_focus_asymptote : b * Real.sqrt (b^2 + a^2) / Real.sqrt (b^2 + a^2) = Real.sqrt 3) :
  (∃ (xm ym xn yn xd yd t : ℝ),
    hyperbola a b xm ym ∧
    hyperbola a b xn yn ∧
    hyperbola a b xd yd ∧
    on_intersection_line xm ym ∧
    on_intersection_line xn yn ∧
    point_relation xm ym xn yn xd yd t ∧
    a^2 = 12 ∧
    b^2 = 3 ∧
    t = 4 ∧
    xd = 4 * Real.sqrt 3 ∧
    yd = 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l1812_181295


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l1812_181263

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = π) : -- condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by
    sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l1812_181263


namespace NUMINAMATH_CALUDE_math_books_count_l1812_181236

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℕ) 
  (h1 : total_books = 90)
  (h2 : math_cost = 4)
  (h3 : history_cost = 5)
  (h4 : total_price = 397) :
  ∃ (math_books : ℕ), 
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧ 
    math_books = 53 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l1812_181236


namespace NUMINAMATH_CALUDE_cone_surface_area_l1812_181292

/-- The surface area of a cone formed by rotating a right triangle -/
theorem cone_surface_area (r h l : ℝ) (triangle_condition : r^2 + h^2 = l^2) :
  r = 3 → h = 4 → l = 5 → (π * r * l + π * r^2) = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l1812_181292


namespace NUMINAMATH_CALUDE_jolene_bicycle_purchase_l1812_181240

structure Income where
  babysitting : Nat
  babysittingRate : Nat
  carWashing : Nat
  carWashingRate : Nat
  dogWalking : Nat
  dogWalkingRate : Nat
  cashGift : Nat

structure BicycleOption where
  price : Nat
  discount : Nat

def calculateTotalIncome (income : Income) : Nat :=
  income.babysitting * income.babysittingRate +
  income.carWashing * income.carWashingRate +
  income.dogWalking * income.dogWalkingRate +
  income.cashGift

def calculateDiscountedPrice (option : BicycleOption) : Nat :=
  option.price - (option.price * option.discount / 100)

def canAfford (income : Nat) (price : Nat) : Prop :=
  income ≥ price

theorem jolene_bicycle_purchase (income : Income)
  (optionA optionB optionC : BicycleOption) :
  income.babysitting = 4 ∧
  income.babysittingRate = 30 ∧
  income.carWashing = 5 ∧
  income.carWashingRate = 12 ∧
  income.dogWalking = 3 ∧
  income.dogWalkingRate = 15 ∧
  income.cashGift = 40 ∧
  optionA.price = 250 ∧
  optionA.discount = 0 ∧
  optionB.price = 300 ∧
  optionB.discount = 10 ∧
  optionC.price = 350 ∧
  optionC.discount = 15 →
  canAfford (calculateTotalIncome income) (calculateDiscountedPrice optionA) ∧
  calculateTotalIncome income - calculateDiscountedPrice optionA = 15 :=
by sorry


end NUMINAMATH_CALUDE_jolene_bicycle_purchase_l1812_181240


namespace NUMINAMATH_CALUDE_trapezium_height_l1812_181235

theorem trapezium_height (a b area : ℝ) (ha : a = 20) (hb : b = 16) (harea : area = 270) :
  (2 * area) / (a + b) = 15 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_height_l1812_181235


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1812_181208

theorem polynomial_division_remainder :
  ∃ (q : Polynomial ℝ), x^4 + x^3 - 4*x + 1 = (x^3 - 1) * q + (-3*x + 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1812_181208


namespace NUMINAMATH_CALUDE_problem_solution_l1812_181214

/-- Represents the number of students in different groups -/
structure StudentGroups where
  total : ℕ
  chinese : ℕ
  math : ℕ
  both : ℕ

/-- Calculates the number of students in neither group -/
def studentsInNeither (g : StudentGroups) : ℕ :=
  g.total - (g.chinese + g.math - g.both)

/-- Theorem statement for the given problem -/
theorem problem_solution (g : StudentGroups) 
  (h1 : g.total = 50)
  (h2 : g.chinese = 15)
  (h3 : g.math = 20)
  (h4 : g.both = 8) :
  studentsInNeither g = 23 := by
  sorry

/-- Example usage of the theorem -/
example : studentsInNeither ⟨50, 15, 20, 8⟩ = 23 := by
  apply problem_solution ⟨50, 15, 20, 8⟩
  repeat' rfl


end NUMINAMATH_CALUDE_problem_solution_l1812_181214


namespace NUMINAMATH_CALUDE_count_different_numerators_l1812_181210

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a recurring decimal in the form 0.ẋyż -/
structure RecurringDecimal where
  x : Digit
  y : Digit
  z : Digit

/-- Converts a RecurringDecimal to a rational number -/
def toRational (d : RecurringDecimal) : ℚ :=
  (d.x.val * 100 + d.y.val * 10 + d.z.val : ℕ) / 999

/-- The set of all possible RecurringDecimals -/
def allRecurringDecimals : Finset RecurringDecimal :=
  sorry

/-- The set of all possible numerators when converting RecurringDecimals to lowest terms -/
def allNumerators : Finset ℕ :=
  sorry

theorem count_different_numerators :
  Finset.card allNumerators = 660 :=
sorry

end NUMINAMATH_CALUDE_count_different_numerators_l1812_181210


namespace NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l1812_181252

theorem min_throws_for_repeated_sum (n : ℕ) (d : ℕ) (s : ℕ) : 
  n = 4 →  -- number of dice
  d = 6 →  -- number of sides on each die
  s = (n * d - n * 1 + 1) →  -- number of possible sums
  s + 1 = 22 →  -- minimum number of throws
  ∀ (throws : ℕ), throws ≥ s + 1 → 
    ∃ (sum1 sum2 : ℕ) (i j : ℕ), 
      i ≠ j ∧ i < throws ∧ j < throws ∧ sum1 = sum2 :=
by sorry

end NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l1812_181252


namespace NUMINAMATH_CALUDE_probability_of_two_red_balls_l1812_181296

def total_balls : ℕ := 7 + 5 + 4

def red_balls : ℕ := 7

def balls_picked : ℕ := 2

def probability_both_red : ℚ := 175 / 1000

theorem probability_of_two_red_balls :
  (Nat.choose red_balls balls_picked : ℚ) / (Nat.choose total_balls balls_picked : ℚ) = probability_both_red :=
by sorry

end NUMINAMATH_CALUDE_probability_of_two_red_balls_l1812_181296


namespace NUMINAMATH_CALUDE_total_cost_in_dollars_l1812_181229

/-- The cost of a single pencil in cents -/
def pencil_cost : ℚ := 2

/-- The cost of a single eraser in cents -/
def eraser_cost : ℚ := 5

/-- The number of pencils to be purchased -/
def num_pencils : ℕ := 500

/-- The number of erasers to be purchased -/
def num_erasers : ℕ := 250

/-- The conversion rate from cents to dollars -/
def cents_to_dollars : ℚ := 1 / 100

theorem total_cost_in_dollars : 
  (pencil_cost * num_pencils + eraser_cost * num_erasers) * cents_to_dollars = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_in_dollars_l1812_181229


namespace NUMINAMATH_CALUDE_competition_probabilities_l1812_181287

/-- A student participates in a science and knowledge competition -/
structure Competition where
  /-- Probability of answering the first question correctly -/
  p1 : ℝ
  /-- Probability of answering the second question correctly -/
  p2 : ℝ
  /-- Probability of answering the third question correctly -/
  p3 : ℝ
  /-- All probabilities are between 0 and 1 -/
  h1 : 0 ≤ p1 ∧ p1 ≤ 1
  h2 : 0 ≤ p2 ∧ p2 ≤ 1
  h3 : 0 ≤ p3 ∧ p3 ≤ 1

/-- The probability of scoring 200 points in the competition -/
def prob_200_points (c : Competition) : ℝ :=
  c.p1 * c.p2 * (1 - c.p3) + (1 - c.p1) * (1 - c.p2) * c.p3

/-- The probability of scoring at least 300 points in the competition -/
def prob_at_least_300_points (c : Competition) : ℝ :=
  c.p1 * (1 - c.p2) * c.p3 + (1 - c.p1) * c.p2 * c.p3 + c.p1 * c.p2 * c.p3

/-- The main theorem about the probabilities in the competition -/
theorem competition_probabilities (c : Competition) 
    (h_p1 : c.p1 = 0.8) (h_p2 : c.p2 = 0.7) (h_p3 : c.p3 = 0.6) : 
    prob_200_points c = 0.26 ∧ prob_at_least_300_points c = 0.564 := by
  sorry


end NUMINAMATH_CALUDE_competition_probabilities_l1812_181287


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_l1812_181268

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_sum_remainder : (List.sum (List.map factorial [1, 2, 3, 4, 5, 6])) % 24 = 9 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_remainder_l1812_181268


namespace NUMINAMATH_CALUDE_area_of_AEC_l1812_181218

-- Define the triangle ABC and its area
def triangle_ABC : Real := 40

-- Define the points on the sides of the triangle
def point_D : Real := 3
def point_B : Real := 5

-- Define the equality of areas
def area_equality : Prop := true

-- Theorem to prove
theorem area_of_AEC (triangle_ABC : Real) (point_D point_B : Real) (area_equality : Prop) :
  (3 : Real) / 8 * triangle_ABC = 15 := by
  sorry

end NUMINAMATH_CALUDE_area_of_AEC_l1812_181218


namespace NUMINAMATH_CALUDE_product_of_powers_inequality_l1812_181241

theorem product_of_powers_inequality (a b : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) (hn : n ≥ 2) :
  (a^n + 1) * (b^n + 1) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_inequality_l1812_181241


namespace NUMINAMATH_CALUDE_sequence_properties_l1812_181273

def b (n : ℕ) : ℝ := 2 * n - 1

def c (n : ℕ) : ℝ := 3 * n - 2

def a (n : ℕ) (x y : ℝ) : ℝ := x * b n + y * c n

theorem sequence_properties
  (x y : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : x + y = 1) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) x y - a n x y = d) ∧
  (∃ y' : ℝ, ∀ n : ℕ, a n x y' = (b n + c n) / 2) ∧
  (∀ n : ℕ, n ≥ 2 → b n < a n x y ∧ a n x y < c n) ∧
  (∀ n : ℕ, n ≥ 2 → a n x y + b n > c n ∧ a n x y + c n > b n ∧ b n + c n > a n x y) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1812_181273


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l1812_181205

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![2, 0; 5, -3]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![8, -2; 1, 1]
  A * B = !![16, -4; 37, -13] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l1812_181205


namespace NUMINAMATH_CALUDE_problem_solution_l1812_181211

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ab ≤ 4 ∧ Real.sqrt a + Real.sqrt b ≤ 2 * Real.sqrt 2 ∧ a^2 + b^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1812_181211


namespace NUMINAMATH_CALUDE_geometric_and_arithmetic_properties_l1812_181291

theorem geometric_and_arithmetic_properties :
  ∀ (s r h : ℝ) (a b : ℝ) (x : ℝ),
  s > 0 → r > 0 → h > 0 → b ≠ 0 →
  (2 * s)^2 = 4 * s^2 ∧
  (π * r^2 * (2 * h)) = 2 * (π * r^2 * h) ∧
  (2 * s)^3 = 8 * s^3 ∧
  (2 * a) / (b / 2) = 4 * (a / b) ∧
  x + 0 = x :=
by sorry

end NUMINAMATH_CALUDE_geometric_and_arithmetic_properties_l1812_181291


namespace NUMINAMATH_CALUDE_trapezoid_area_l1812_181247

/-- A trapezoid with the given properties has an area of 260.4 square centimeters. -/
theorem trapezoid_area (h : ℝ) (b₁ b₂ : ℝ) :
  h = 12 →
  b₁ = 15 →
  b₂ = 13 →
  (b₁^2 - h^2).sqrt + (b₂^2 - h^2).sqrt = 14 →
  (1/2) * (((b₁^2 - h^2).sqrt + (b₂^2 - h^2).sqrt + b₁) + 
           ((b₁^2 - h^2).sqrt + (b₂^2 - h^2).sqrt + b₂)) * h = 260.4 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1812_181247


namespace NUMINAMATH_CALUDE_smallest_club_size_club_size_exists_l1812_181220

theorem smallest_club_size (n : ℕ) : 
  (n % 6 = 1) ∧ (n % 8 = 2) ∧ (n % 9 = 3) → n ≥ 343 :=
by sorry

theorem club_size_exists : 
  ∃ n : ℕ, (n % 6 = 1) ∧ (n % 8 = 2) ∧ (n % 9 = 3) ∧ n = 343 :=
by sorry

end NUMINAMATH_CALUDE_smallest_club_size_club_size_exists_l1812_181220


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1812_181206

theorem sum_of_fractions : (3 / 30) + (4 / 40) + (5 / 50) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1812_181206


namespace NUMINAMATH_CALUDE_total_rooms_is_260_l1812_181284

/-- Represents the hotel booking scenario -/
structure HotelBooking where
  singleRooms : ℕ
  doubleRooms : ℕ
  singleRoomCost : ℕ
  doubleRoomCost : ℕ
  totalIncome : ℕ

/-- Calculates the total number of rooms booked -/
def totalRooms (booking : HotelBooking) : ℕ :=
  booking.singleRooms + booking.doubleRooms

/-- Theorem stating that the total number of rooms booked is 260 -/
theorem total_rooms_is_260 (booking : HotelBooking) 
  (h1 : booking.singleRooms = 64)
  (h2 : booking.singleRoomCost = 35)
  (h3 : booking.doubleRoomCost = 60)
  (h4 : booking.totalIncome = 14000) :
  totalRooms booking = 260 := by
  sorry

#eval totalRooms { singleRooms := 64, doubleRooms := 196, singleRoomCost := 35, doubleRoomCost := 60, totalIncome := 14000 }

end NUMINAMATH_CALUDE_total_rooms_is_260_l1812_181284


namespace NUMINAMATH_CALUDE_simplify_expression_l1812_181278

theorem simplify_expression (a b : ℝ) : (-a^2 * b^3)^3 = -a^6 * b^9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1812_181278
