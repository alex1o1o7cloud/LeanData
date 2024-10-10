import Mathlib

namespace sample_size_is_sampled_athletes_l3732_373223

/-- A structure representing a statistical study of athletes' ages -/
structure AthleteStudy where
  total_athletes : ℕ
  sampled_athletes : ℕ
  h_total : total_athletes = 1000
  h_sampled : sampled_athletes = 100

/-- The sample size of an athlete study is equal to the number of sampled athletes -/
theorem sample_size_is_sampled_athletes (study : AthleteStudy) : 
  study.sampled_athletes = 100 := by
  sorry

end sample_size_is_sampled_athletes_l3732_373223


namespace distance_traveled_l3732_373256

/-- Given a car's fuel efficiency and the amount of fuel used, calculate the distance traveled. -/
theorem distance_traveled (efficiency : ℝ) (fuel_used : ℝ) (h1 : efficiency = 20) (h2 : fuel_used = 3) :
  efficiency * fuel_used = 60 := by
  sorry

end distance_traveled_l3732_373256


namespace decreasing_linear_function_negative_slope_l3732_373216

/-- A linear function y = kx - 5 where y decreases as x increases -/
def decreasing_linear_function (k : ℝ) : ℝ → ℝ := λ x ↦ k * x - 5

/-- Theorem: If y decreases as x increases in a linear function y = kx - 5, then k < 0 -/
theorem decreasing_linear_function_negative_slope (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → decreasing_linear_function k x₁ > decreasing_linear_function k x₂) →
  k < 0 :=
by sorry

end decreasing_linear_function_negative_slope_l3732_373216


namespace roots_of_polynomial_l3732_373298

-- Define the polynomial
def f (x : ℝ) := 3 * x^3 - 3 * x^2 - 3 * x - 9

-- Define p, q, r as roots of the polynomial
theorem roots_of_polynomial (p q r : ℝ) : f p = 0 ∧ f q = 0 ∧ f r = 0 → p^2 + q^2 + r^2 = 3 := by
  sorry

end roots_of_polynomial_l3732_373298


namespace strawberries_in_buckets_l3732_373284

theorem strawberries_in_buckets
  (total_strawberries : ℕ)
  (num_buckets : ℕ)
  (removed_per_bucket : ℕ)
  (h1 : total_strawberries = 300)
  (h2 : num_buckets = 5)
  (h3 : removed_per_bucket = 20)
  : (total_strawberries / num_buckets) - removed_per_bucket = 40 := by
  sorry

end strawberries_in_buckets_l3732_373284


namespace M_subset_N_l3732_373239

-- Define the sets M and N
def M : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 4}
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 8 - 1 / 4}

-- Theorem to prove
theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l3732_373239


namespace spinner_probability_l3732_373255

/- Define an isosceles triangle with the given angle property -/
structure IsoscelesTriangle where
  baseAngle : ℝ
  vertexAngle : ℝ
  isIsosceles : baseAngle = 2 * vertexAngle

/- Define the division of the triangle into regions by altitudes -/
def triangleRegions : ℕ := 6

/- Define the number of shaded regions -/
def shadedRegions : ℕ := 4

/- Define the probability of landing in a shaded region -/
def shadedProbability (t : IsoscelesTriangle) : ℚ :=
  shadedRegions / triangleRegions

/- Theorem statement -/
theorem spinner_probability (t : IsoscelesTriangle) :
  shadedProbability t = 2 / 3 := by
  sorry

end spinner_probability_l3732_373255


namespace equation_solution_l3732_373232

theorem equation_solution :
  ∃ x : ℝ, (3639 + 11.95 - x^2 = 3054) ∧ (abs (x - 24.43) < 0.01) := by
  sorry

end equation_solution_l3732_373232


namespace change_calculation_l3732_373218

def initial_amount : ℕ := 20
def num_items : ℕ := 3
def cost_per_item : ℕ := 2

theorem change_calculation :
  initial_amount - (num_items * cost_per_item) = 14 := by
  sorry

end change_calculation_l3732_373218


namespace point_upper_left_of_line_l3732_373271

/-- A point in the plane is represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane is represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determines if a point is on the upper left side of a line -/
def isUpperLeftSide (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c < 0

/-- The main theorem -/
theorem point_upper_left_of_line (t : ℝ) :
  let p : Point := ⟨-2, t⟩
  let l : Line := ⟨1, -1, 4⟩
  isUpperLeftSide p l → t > 2 := by
  sorry


end point_upper_left_of_line_l3732_373271


namespace last_two_digits_product_l3732_373291

theorem last_two_digits_product (A B : ℕ) : 
  (A * 10 + B) % 6 = 0 → A + B = 11 → A * B = 24 := by
  sorry

end last_two_digits_product_l3732_373291


namespace drop_recording_l3732_373240

/-- Represents the change in water level in meters -/
def WaterLevelChange : Type := ℝ

/-- Records a rise in water level -/
def recordRise (meters : ℝ) : WaterLevelChange := meters

/-- Records a drop in water level -/
def recordDrop (meters : ℝ) : WaterLevelChange := -meters

/-- The theorem stating how a drop in water level should be recorded -/
theorem drop_recording (rise : ℝ) (drop : ℝ) :
  recordRise rise = rise → recordDrop drop = -drop :=
by sorry

end drop_recording_l3732_373240


namespace arithmetic_geometric_mean_inequality_l3732_373202

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by sorry

end arithmetic_geometric_mean_inequality_l3732_373202


namespace correct_num_technicians_l3732_373289

/-- Represents the number of technicians in the workshop -/
def num_technicians : ℕ := 7

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 21

/-- Represents the average salary of all workers -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℕ := 12000

/-- Represents the average salary of non-technicians -/
def avg_salary_rest : ℕ := 6000

/-- Theorem stating that the number of technicians is correct given the conditions -/
theorem correct_num_technicians : 
  num_technicians * avg_salary_technicians + 
  (total_workers - num_technicians) * avg_salary_rest = 
  total_workers * avg_salary_all :=
sorry

#check correct_num_technicians

end correct_num_technicians_l3732_373289


namespace xyz_value_l3732_373246

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 27)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11)
  (h3 : x + y + z = 3) : 
  x * y * z = 16 / 3 := by
sorry

end xyz_value_l3732_373246


namespace descending_order_proof_l3732_373248

theorem descending_order_proof :
  (1909 > 1100 ∧ 1100 > 1090 ∧ 1090 > 1009) ∧
  (10000 > 9999 ∧ 9999 > 9990 ∧ 9990 > 8909 ∧ 8909 > 8900) := by
  sorry

end descending_order_proof_l3732_373248


namespace stable_table_configurations_l3732_373263

def stableConfigurations (n : ℕ+) : ℕ :=
  (1/3) * (n+1) * (2*n^2 + 4*n + 3)

theorem stable_table_configurations (n : ℕ+) :
  (stableConfigurations n) =
  (Finset.sum (Finset.range (2*n+1)) (λ k =>
    (if k ≤ n then k + 1 else 2*n - k + 1)^2)) :=
sorry

end stable_table_configurations_l3732_373263


namespace no_integer_solution_l3732_373277

theorem no_integer_solution :
  ∀ (x y z : ℤ), x ≠ 0 → 2 * x^4 + 2 * x^2 * y^2 + y^4 ≠ z^2 := by
  sorry

end no_integer_solution_l3732_373277


namespace max_expected_score_l3732_373253

/-- Xiao Zhang's box configuration -/
structure BoxConfig where
  red : ℕ
  yellow : ℕ
  white : ℕ
  sum_six : red + yellow + white = 6

/-- Expected score for a given box configuration -/
def expectedScore (config : BoxConfig) : ℚ :=
  (3 * config.red + 4 * config.yellow + 3 * config.white) / 36

/-- Theorem stating the maximum expected score and optimal configuration -/
theorem max_expected_score :
  ∃ (config : BoxConfig),
    expectedScore config = 2/3 ∧
    ∀ (other : BoxConfig), expectedScore other ≤ expectedScore config ∧
    config.red = 0 ∧ config.yellow = 6 ∧ config.white = 0 := by
  sorry


end max_expected_score_l3732_373253


namespace andy_remaining_demerits_l3732_373294

/-- The maximum number of demerits allowed in a month before firing -/
def max_demerits : ℕ := 50

/-- The number of demerits per late instance -/
def demerits_per_late : ℕ := 2

/-- The number of times Andy was late -/
def times_late : ℕ := 6

/-- The number of demerits for the inappropriate joke -/
def joke_demerits : ℕ := 15

/-- The number of additional demerits Andy can get before being fired -/
def remaining_demerits : ℕ := max_demerits - (demerits_per_late * times_late + joke_demerits)

theorem andy_remaining_demerits : remaining_demerits = 23 := by
  sorry

end andy_remaining_demerits_l3732_373294


namespace exponent_simplification_l3732_373262

theorem exponent_simplification (x : ℝ) (hx : x ≠ 0) :
  x^5 * x^7 / x^3 = x^9 := by sorry

end exponent_simplification_l3732_373262


namespace hyperbola_b_plus_k_l3732_373258

/-- Given a hyperbola with asymptotes y = 3x + 6 and y = -3x + 2, passing through (2, 12),
    prove that b + k = (16√2 + 36) / 9, where (y-k)²/a² - (x-h)²/b² = 1 is the standard form. -/
theorem hyperbola_b_plus_k (a b h k : ℝ) : a > 0 → b > 0 →
  (∀ x y, y = 3*x + 6 ∨ y = -3*x + 2) →  -- Asymptotes
  ((12 - k)^2 / a^2) - ((2 - h)^2 / b^2) = 1 →  -- Point (2, 12) satisfies the equation
  (∀ x y, (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1) →  -- Standard form
  b + k = (16 * Real.sqrt 2 + 36) / 9 := by
sorry

end hyperbola_b_plus_k_l3732_373258


namespace six_balls_two_boxes_l3732_373209

/-- The number of ways to distribute n distinguishable balls into 2 indistinguishable boxes -/
def distributionWays (n : ℕ) : ℕ :=
  (2^n) / 2 - 1

/-- Theorem: There are 31 ways to distribute 6 distinguishable balls into 2 indistinguishable boxes -/
theorem six_balls_two_boxes : distributionWays 6 = 31 := by
  sorry

end six_balls_two_boxes_l3732_373209


namespace unique_triple_l3732_373201

theorem unique_triple : ∃! (a b c : ℤ), 
  a > 0 ∧ 0 > b ∧ b > c ∧ 
  a + b + c = 0 ∧ 
  ∃ (k : ℤ), 2017 - a^3*b - b^3*c - c^3*a = k^2 ∧
  a = 36 ∧ b = -12 ∧ c = -24 :=
sorry

end unique_triple_l3732_373201


namespace room_width_calculation_l3732_373203

/-- Given a rectangular room with specified length, paving cost per square meter,
    and total paving cost, calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ)
  (h1 : length = 5.5)
  (h2 : cost_per_sqm = 950)
  (h3 : total_cost = 20900) :
  total_cost / cost_per_sqm / length = 4 := by
  sorry

#check room_width_calculation

end room_width_calculation_l3732_373203


namespace mod_9_sum_of_digits_mod_9_sum_mod_9_product_l3732_373281

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Property 1
theorem mod_9_sum_of_digits (n : ℕ) : n % 9 = sumOfDigits n % 9 := by
  sorry

-- Property 2
theorem mod_9_sum (ns : List ℕ) : 
  (ns.sum % 9) = (ns.map (· % 9)).sum % 9 := by
  sorry

-- Property 3
theorem mod_9_product (ns : List ℕ) : 
  (ns.prod % 9) = (ns.map (· % 9)).prod % 9 := by
  sorry

end mod_9_sum_of_digits_mod_9_sum_mod_9_product_l3732_373281


namespace polynomial_divisibility_l3732_373251

/-- A polynomial with integer coefficients -/
def IntPolynomial : Type := ℕ → ℤ

/-- Evaluate a polynomial at a given integer -/
def evaluate (f : IntPolynomial) (x : ℤ) : ℤ :=
  sorry

/-- A number is divisible by another if their remainder is zero -/
def divisible (a b : ℤ) : Prop := a % b = 0

theorem polynomial_divisibility (f : IntPolynomial) :
  divisible (evaluate f 2) 6 →
  divisible (evaluate f 3) 6 →
  divisible (evaluate f 5) 6 :=
sorry

end polynomial_divisibility_l3732_373251


namespace toms_floor_replacement_cost_l3732_373242

/-- The total cost to replace a floor given the room dimensions, removal cost, and new floor cost per square foot. -/
def total_floor_replacement_cost (length width removal_cost cost_per_sqft : ℝ) : ℝ :=
  removal_cost + length * width * cost_per_sqft

/-- Theorem stating that the total cost to replace the floor in Tom's room is $120. -/
theorem toms_floor_replacement_cost :
  total_floor_replacement_cost 8 7 50 1.25 = 120 := by
  sorry

end toms_floor_replacement_cost_l3732_373242


namespace remaining_steps_l3732_373230

/-- Given a total of 96 stair steps and 74 steps already climbed, 
    prove that the remaining steps to climb is 22. -/
theorem remaining_steps (total : Nat) (climbed : Nat) (h1 : total = 96) (h2 : climbed = 74) :
  total - climbed = 22 := by
  sorry

end remaining_steps_l3732_373230


namespace smallest_n_for_divisibility_by_1991_l3732_373217

theorem smallest_n_for_divisibility_by_1991 :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (1991 ∣ (a + b) ∨ 1991 ∣ (a - b))) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℤ), T.card = m ∧
      ∀ (a b : ℤ), a ∈ T → b ∈ T → a ≠ b → ¬(1991 ∣ (a + b)) ∧ ¬(1991 ∣ (a - b))) ∧
  n = 997 :=
by
  sorry

end smallest_n_for_divisibility_by_1991_l3732_373217


namespace sufficient_condition_for_inequality_l3732_373274

theorem sufficient_condition_for_inequality (a : ℝ) :
  a ≥ 5 → ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0 := by sorry

end sufficient_condition_for_inequality_l3732_373274


namespace prob_at_least_two_correct_l3732_373280

def num_questions : ℕ := 30
def num_guessed : ℕ := 5
def num_choices : ℕ := 6

def prob_correct : ℚ := 1 / num_choices
def prob_incorrect : ℚ := 1 - prob_correct

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem prob_at_least_two_correct :
  (1 : ℚ) - (binomial num_guessed 0 : ℚ) * prob_incorrect ^ num_guessed
          - (binomial num_guessed 1 : ℚ) * prob_correct * prob_incorrect ^ (num_guessed - 1)
  = 1526 / 7776 := by sorry

end prob_at_least_two_correct_l3732_373280


namespace basketball_game_ratio_l3732_373290

theorem basketball_game_ratio :
  let girls : ℕ := 30
  let boys : ℕ := girls + 18
  let ratio : ℚ := boys / girls
  ratio = 8 / 5 := by sorry

end basketball_game_ratio_l3732_373290


namespace system_solution_l3732_373267

theorem system_solution :
  ∀ x y z : ℝ,
  (x * y + x * z = 8 - x^2) ∧
  (x * y + y * z = 12 - y^2) ∧
  (y * z + z * x = -4 - z^2) →
  ((x = 2 ∧ y = 3 ∧ z = -1) ∨ (x = -2 ∧ y = -3 ∧ z = 1)) :=
by sorry

end system_solution_l3732_373267


namespace trapezoid_shorter_base_l3732_373250

/-- A trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  midpoints_distance : ℝ
  shorter_base : ℝ

/-- The theorem stating the relationship between the bases and the midpoints distance -/
theorem trapezoid_shorter_base (t : Trapezoid) 
  (h1 : t.longer_base = 24)
  (h2 : t.midpoints_distance = 4) : 
  t.shorter_base = 16 := by
  sorry

#check trapezoid_shorter_base

end trapezoid_shorter_base_l3732_373250


namespace fifteen_star_positive_integer_count_l3732_373269

def star (a b : ℤ) : ℚ := a^3 / b

theorem fifteen_star_positive_integer_count :
  (∃ (S : Finset ℤ), (∀ x ∈ S, x > 0 ∧ (star 15 x).isInt) ∧ S.card = 16) :=
sorry

end fifteen_star_positive_integer_count_l3732_373269


namespace orchard_pure_gala_count_l3732_373206

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  cross_pollinated : ℕ
  pure_gala : ℕ

/-- The number of pure Gala trees in an orchard satisfying specific conditions -/
def pure_gala_count (o : Orchard) : Prop :=
  o.pure_fuji + o.cross_pollinated = 204 ∧
  o.pure_fuji = (3 * o.total) / 4 ∧
  o.cross_pollinated = o.total / 10 ∧
  o.pure_gala = 36

/-- Theorem stating that an orchard satisfying the given conditions has 36 pure Gala trees -/
theorem orchard_pure_gala_count :
  ∃ (o : Orchard), pure_gala_count o :=
sorry

end orchard_pure_gala_count_l3732_373206


namespace triangle_area_from_squares_l3732_373227

theorem triangle_area_from_squares (a b : ℝ) (ha : a^2 = 25) (hb : b^2 = 144) : 
  (1/2) * a * b = 30 := by
sorry

end triangle_area_from_squares_l3732_373227


namespace fruit_distribution_ways_l3732_373225

/-- The number of ways to distribute n indistinguishable items into k distinguishable bins -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of fruits to buy -/
def total_fruits : ℕ := 17

/-- The number of types of fruit -/
def fruit_types : ℕ := 5

/-- The number of fruits remaining after placing one in each type -/
def remaining_fruits : ℕ := total_fruits - fruit_types

theorem fruit_distribution_ways :
  distribute remaining_fruits fruit_types = 1820 :=
sorry

end fruit_distribution_ways_l3732_373225


namespace sqrt_27_div_sqrt_3_equals_3_l3732_373226

theorem sqrt_27_div_sqrt_3_equals_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end sqrt_27_div_sqrt_3_equals_3_l3732_373226


namespace students_per_bus_l3732_373249

theorem students_per_bus (total_students : ℕ) (num_buses : ℕ) 
  (h1 : total_students = 360) (h2 : num_buses = 8) :
  total_students / num_buses = 45 := by
  sorry

end students_per_bus_l3732_373249


namespace R_has_smallest_d_l3732_373212

/-- Represents a square with four labeled sides --/
structure Square where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The four squares given in the problem --/
def P : Square := { a := 2, b := 3, c := 10, d := 8 }
def Q : Square := { a := 8, b := 1, c := 2, d := 6 }
def R : Square := { a := 4, b := 5, c := 7, d := 1 }
def S : Square := { a := 7, b := 6, c := 5, d := 3 }

/-- Theorem stating that R has the smallest d value among the squares --/
theorem R_has_smallest_d : 
  R.d ≤ P.d ∧ R.d ≤ Q.d ∧ R.d ≤ S.d ∧ 
  (R.d < P.d ∨ R.d < Q.d ∨ R.d < S.d) := by
  sorry

end R_has_smallest_d_l3732_373212


namespace triangle_side_length_l3732_373221

theorem triangle_side_length (BC AC : ℝ) (A : ℝ) :
  BC = Real.sqrt 7 →
  AC = 2 * Real.sqrt 3 →
  A = π / 6 →
  ∃ AB : ℝ, (AB = 5 ∨ AB = 1) ∧
    AB^2 + AC^2 - BC^2 = 2 * AB * AC * Real.cos A :=
by sorry

end triangle_side_length_l3732_373221


namespace exhibition_average_l3732_373237

theorem exhibition_average : 
  let works : List ℕ := [58, 52, 58, 60]
  (works.sum / works.length : ℚ) = 57 := by sorry

end exhibition_average_l3732_373237


namespace log_equation_solution_l3732_373228

theorem log_equation_solution (s : ℝ) (h : s > 0) :
  (4 * Real.log s / Real.log 3 = Real.log (4 * s^2) / Real.log 3) → s = 2 := by
  sorry

end log_equation_solution_l3732_373228


namespace billy_lemon_heads_l3732_373235

theorem billy_lemon_heads (total_lemon_heads : ℕ) (lemon_heads_per_friend : ℕ) (h1 : total_lemon_heads = 72) (h2 : lemon_heads_per_friend = 12) :
  total_lemon_heads / lemon_heads_per_friend = 6 := by
sorry

end billy_lemon_heads_l3732_373235


namespace correct_price_reduction_equation_l3732_373244

/-- Represents the price reduction model for a sportswear item -/
def PriceReductionModel (initial_price final_price : ℝ) : Prop :=
  ∃ x : ℝ, 
    0 < x ∧ 
    x < 1 ∧ 
    initial_price * (1 - x)^2 = final_price

/-- Theorem stating that the given equation correctly models the price reduction -/
theorem correct_price_reduction_equation :
  PriceReductionModel 560 315 :=
sorry

end correct_price_reduction_equation_l3732_373244


namespace inner_cube_surface_area_l3732_373224

/-- Given a cube with surface area 54 square meters containing an inscribed sphere,
    which in turn contains an inscribed cube, the surface area of the inner cube
    is 18 square meters. -/
theorem inner_cube_surface_area (outer_cube : Real) (sphere : Real) (inner_cube : Real) :
  outer_cube = 54 →  -- Surface area of outer cube
  sphere ^ 2 = 3 * inner_cube ^ 2 →  -- Relation between sphere and inner cube
  inner_cube ^ 2 = 3 →  -- Side length of inner cube
  6 * inner_cube ^ 2 = 18  -- Surface area of inner cube
  := by sorry

end inner_cube_surface_area_l3732_373224


namespace stock_value_change_l3732_373260

/-- Theorem: Stock Value Change over Two Days
    Given a stock that decreases in value by 25% on the first day and
    increases by 40% on the second day, prove that the overall
    percentage change is a 5% increase. -/
theorem stock_value_change (initial_value : ℝ) (initial_value_pos : initial_value > 0) :
  let day1_value := initial_value * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.40)
  (day2_value - initial_value) / initial_value = 0.05 := by
sorry

end stock_value_change_l3732_373260


namespace rectangular_box_surface_area_l3732_373241

theorem rectangular_box_surface_area 
  (x y z : ℝ) 
  (h1 : x > 0 ∧ y > 0 ∧ z > 0)
  (h2 : 4 * (x + y + z) = 140)
  (h3 : x^2 + y^2 + z^2 = 21^2) :
  2 * (x*y + x*z + y*z) = 784 :=
by sorry

end rectangular_box_surface_area_l3732_373241


namespace max_value_sum_of_roots_l3732_373238

theorem max_value_sum_of_roots (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 5) :
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) ≤ Real.sqrt 39 ∧
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 5 ∧
    Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) = Real.sqrt 39 :=
by sorry

end max_value_sum_of_roots_l3732_373238


namespace arithmetic_sequence_ratio_l3732_373229

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  d : ℚ
  seq_def : ∀ n : ℕ+, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) 
  (h : ∀ n : ℕ+, (sum_n_terms a n) / (sum_n_terms b n) = (7 * n + 1) / (4 * n + 27)) :
  (a.a 7) / (b.a 7) = 92 / 79 := by
  sorry

end arithmetic_sequence_ratio_l3732_373229


namespace system_unique_solution_l3732_373214

/-- The system of equations has a unique solution (1, 2) -/
theorem system_unique_solution :
  ∃! (x y : ℝ), x + 2*y = 5 ∧ 3*x - y = 1 :=
by
  sorry

end system_unique_solution_l3732_373214


namespace sturgeon_books_problem_l3732_373204

theorem sturgeon_books_problem (total_volumes : ℕ) (paperback_price hardcover_price total_cost : ℚ) 
  (h : total_volumes = 12)
  (hp : paperback_price = 15)
  (hh : hardcover_price = 25)
  (ht : total_cost = 240) :
  ∃ (hardcovers : ℕ), 
    hardcovers * hardcover_price + (total_volumes - hardcovers) * paperback_price = total_cost ∧ 
    hardcovers = 6 := by
  sorry

end sturgeon_books_problem_l3732_373204


namespace multiply_and_add_equality_l3732_373276

theorem multiply_and_add_equality : 45 * 56 + 54 * 45 = 4950 := by
  sorry

end multiply_and_add_equality_l3732_373276


namespace arrangements_theorem_l3732_373278

def num_men : ℕ := 5
def num_women : ℕ := 2
def positions_for_man_a : ℕ := 2

def arrangements_count : ℕ :=
  positions_for_man_a * Nat.factorial (num_men - 1 + 1) * Nat.factorial num_women

theorem arrangements_theorem : arrangements_count = 480 := by
  sorry

end arrangements_theorem_l3732_373278


namespace socks_theorem_l3732_373254

/-- The number of pairs of socks Niko bought -/
def total_socks : ℕ := 9

/-- The cost of each pair of socks in dollars -/
def cost_per_pair : ℚ := 2

/-- The number of pairs with 25% profit -/
def pairs_with_25_percent : ℕ := 4

/-- The number of pairs with $0.2 profit -/
def pairs_with_20_cents : ℕ := 5

/-- The total profit in dollars -/
def total_profit : ℚ := 3

/-- The profit percentage for the first group of socks -/
def profit_percentage : ℚ := 25 / 100

/-- The profit amount for the second group of socks in dollars -/
def profit_amount : ℚ := 1 / 5

theorem socks_theorem :
  total_socks = pairs_with_25_percent + pairs_with_20_cents ∧
  total_profit = pairs_with_25_percent * (cost_per_pair * profit_percentage) +
                 pairs_with_20_cents * profit_amount :=
by sorry

end socks_theorem_l3732_373254


namespace number_of_workers_l3732_373210

theorem number_of_workers (total_contribution : ℕ) (extra_contribution : ℕ) (new_total : ℕ) : 
  total_contribution = 300000 →
  extra_contribution = 50 →
  new_total = 320000 →
  ∃ (workers : ℕ), 
    workers * (total_contribution / workers + extra_contribution) = new_total ∧
    workers = 400 := by
  sorry

end number_of_workers_l3732_373210


namespace value_of_2x_minus_y_l3732_373233

theorem value_of_2x_minus_y (x y : ℝ) (hx : |x| = 3) (hy : |y| = 4) (hxy : x > y) :
  2 * x - y = 10 := by
sorry

end value_of_2x_minus_y_l3732_373233


namespace trajectory_of_product_slopes_l3732_373295

/-- The trajectory of a moving point P whose product of slopes to fixed points A(-1,0) and B(1,0) is -1 -/
theorem trajectory_of_product_slopes (x y : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  (y / (x + 1)) * (y / (x - 1)) = -1 → x^2 + y^2 = 1 := by
sorry

end trajectory_of_product_slopes_l3732_373295


namespace union_of_sets_l3732_373268

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x | 3 * x^2 + p * x - 7 = 0}
def B (q : ℝ) : Set ℝ := {x | 3 * x^2 - 7 * x + q = 0}

-- State the theorem
theorem union_of_sets (p q : ℝ) :
  (∃ (p q : ℝ), A p ∩ B q = {-1/3}) →
  (∃ (p q : ℝ), A p ∪ B q = {-1/3, 8/3, 7}) :=
by sorry

end union_of_sets_l3732_373268


namespace smallest_number_divisible_by_multiple_l3732_373287

def isDivisibleBy (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_divisible_by_multiple : 
  ∃! n : ℕ, (∀ m : ℕ, m < n → 
    ¬(isDivisibleBy (m - 6) 12 ∧ 
      isDivisibleBy (m - 6) 16 ∧ 
      isDivisibleBy (m - 6) 18 ∧ 
      isDivisibleBy (m - 6) 21 ∧ 
      isDivisibleBy (m - 6) 28)) ∧ 
    isDivisibleBy (n - 6) 12 ∧ 
    isDivisibleBy (n - 6) 16 ∧ 
    isDivisibleBy (n - 6) 18 ∧ 
    isDivisibleBy (n - 6) 21 ∧ 
    isDivisibleBy (n - 6) 28 ∧
    n = 1014 :=
by sorry

end smallest_number_divisible_by_multiple_l3732_373287


namespace fourth_root_over_seventh_root_of_seven_l3732_373285

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
by sorry

end fourth_root_over_seventh_root_of_seven_l3732_373285


namespace trig_identity_l3732_373279

theorem trig_identity : 
  4 * Real.sin (15 * π / 180) + Real.tan (75 * π / 180) = 
  (4 - 3 * (Real.cos (15 * π / 180))^2 + Real.cos (15 * π / 180)) / Real.sin (15 * π / 180) := by
sorry

end trig_identity_l3732_373279


namespace cos_negative_420_degrees_l3732_373299

theorem cos_negative_420_degrees : Real.cos (-(420 * π / 180)) = 1 / 2 := by
  sorry

end cos_negative_420_degrees_l3732_373299


namespace tonys_fever_degree_l3732_373265

/-- Proves that Tony's temperature is 5 degrees above the fever threshold given the conditions --/
theorem tonys_fever_degree (normal_temp : ℝ) (temp_increase : ℝ) (fever_threshold : ℝ) :
  normal_temp = 95 →
  temp_increase = 10 →
  fever_threshold = 100 →
  normal_temp + temp_increase - fever_threshold = 5 := by
  sorry

end tonys_fever_degree_l3732_373265


namespace courtyard_length_l3732_373283

theorem courtyard_length (width : ℝ) (tiles_per_sqft : ℝ) 
  (green_ratio : ℝ) (red_ratio : ℝ) (green_cost : ℝ) (red_cost : ℝ) 
  (total_cost : ℝ) (L : ℝ) : 
  width = 25 ∧ 
  tiles_per_sqft = 4 ∧ 
  green_ratio = 0.4 ∧ 
  red_ratio = 0.6 ∧ 
  green_cost = 3 ∧ 
  red_cost = 1.5 ∧ 
  total_cost = 2100 ∧ 
  total_cost = (green_ratio * tiles_per_sqft * L * width * green_cost) + 
               (red_ratio * tiles_per_sqft * L * width * red_cost) → 
  L = 10 := by
  sorry

end courtyard_length_l3732_373283


namespace lesser_fraction_l3732_373243

theorem lesser_fraction (x y : ℚ) 
  (sum_eq : x + y = 9/10)
  (prod_eq : x * y = 1/15) :
  min x y = 1/5 := by sorry

end lesser_fraction_l3732_373243


namespace robin_cupcake_ratio_l3732_373211

/-- Given that Robin ate 4 cupcakes with chocolate sauce and 12 cupcakes in total,
    prove that the ratio of cupcakes with buttercream frosting to cupcakes with chocolate sauce is 2:1 -/
theorem robin_cupcake_ratio :
  let chocolate_cupcakes : ℕ := 4
  let total_cupcakes : ℕ := 12
  let buttercream_cupcakes : ℕ := total_cupcakes - chocolate_cupcakes
  (buttercream_cupcakes : ℚ) / chocolate_cupcakes = 2 / 1 :=
by sorry

end robin_cupcake_ratio_l3732_373211


namespace derivative_f_at_zero_l3732_373236

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^2 + 3

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 0 := by sorry

end derivative_f_at_zero_l3732_373236


namespace floor_sqrt_26_squared_l3732_373231

theorem floor_sqrt_26_squared : ⌊Real.sqrt 26⌋^2 = 25 := by
  sorry

end floor_sqrt_26_squared_l3732_373231


namespace barrel_tank_ratio_l3732_373215

theorem barrel_tank_ratio : 
  ∀ (barrel_volume tank_volume : ℝ),
  barrel_volume > 0 → tank_volume > 0 →
  (3/4 : ℝ) * barrel_volume = (5/8 : ℝ) * tank_volume →
  barrel_volume / tank_volume = 5/6 := by
  sorry

end barrel_tank_ratio_l3732_373215


namespace inequality_and_minimum_value_l3732_373205

theorem inequality_and_minimum_value (a b c : ℝ) 
  (ha : 1 < a ∧ a < Real.sqrt 7)
  (hb : 1 < b ∧ b < Real.sqrt 7)
  (hc : 1 < c ∧ c < Real.sqrt 7) :
  (1 / (a^2 - 1) + 1 / (7 - a^2) ≥ 2/3) ∧
  (1 / Real.sqrt ((a^2 - 1) * (7 - b^2)) + 
   1 / Real.sqrt ((b^2 - 1) * (7 - c^2)) + 
   1 / Real.sqrt ((c^2 - 1) * (7 - a^2)) ≥ 1) := by
  sorry

end inequality_and_minimum_value_l3732_373205


namespace cosine_value_l3732_373272

theorem cosine_value (α : Real) 
  (h : Real.cos (α - π/6) - Real.sin α = 2 * Real.sqrt 3 / 5) : 
  Real.cos (α + 7*π/6) = -(2 * Real.sqrt 3 / 5) := by
  sorry

end cosine_value_l3732_373272


namespace red_numbers_structure_l3732_373207

-- Define the color type
inductive Color
| White
| Red

-- Define the coloring function
def coloring : ℕ → Color := sorry

-- Define properties of the coloring
axiom exists_white : ∃ n : ℕ, coloring n = Color.White
axiom exists_red : ∃ n : ℕ, coloring n = Color.Red
axiom sum_white_red_is_white :
  ∀ w r : ℕ, coloring w = Color.White → coloring r = Color.Red →
  coloring (w + r) = Color.White
axiom product_white_red_is_red :
  ∀ w r : ℕ, coloring w = Color.White → coloring r = Color.Red →
  coloring (w * r) = Color.Red

-- Define the set of red numbers
def RedNumbers : Set ℕ := {n : ℕ | coloring n = Color.Red}

-- State the theorem
theorem red_numbers_structure :
  ∃ r₀ : ℕ, r₀ > 0 ∧ r₀ ∈ RedNumbers ∧
  ∀ n : ℕ, n ∈ RedNumbers ↔ ∃ k : ℕ, n = k * r₀ :=
sorry

end red_numbers_structure_l3732_373207


namespace geometric_sequence_308th_term_l3732_373266

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem geometric_sequence_308th_term :
  let a₁ := 12
  let a₂ := -24
  let r := a₂ / a₁
  geometric_sequence a₁ r 308 = -2^307 * 12 :=
by sorry

end geometric_sequence_308th_term_l3732_373266


namespace lottery_expected_wins_l3732_373259

/-- A lottery with a winning probability of 1/4 -/
structure Lottery where
  win_prob : ℝ
  win_prob_eq : win_prob = 1/4

/-- The expected number of winning tickets when drawing n tickets -/
def expected_wins (L : Lottery) (n : ℕ) : ℝ := n * L.win_prob

theorem lottery_expected_wins (L : Lottery) : expected_wins L 4 = 1 := by
  sorry

end lottery_expected_wins_l3732_373259


namespace investment_interest_proof_l3732_373292

/-- Calculate compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Calculate total interest earned --/
def total_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  compound_interest principal rate years - principal

theorem investment_interest_proof :
  let principal := 1500
  let rate := 0.08
  let years := 5
  ∃ ε > 0, abs (total_interest principal rate years - 704) < ε :=
by sorry

end investment_interest_proof_l3732_373292


namespace submarine_hit_guaranteed_l3732_373252

/-- Represents the position of a submarine at time t -/
def submarinePosition (v : ℕ+) (t : ℕ) : ℕ := v.val * t

/-- Represents the position of a missile fired at time n -/
def missilePosition (n : ℕ) : ℕ := n ^ 2

/-- Theorem stating that there exists a firing sequence that will hit the submarine -/
theorem submarine_hit_guaranteed :
  ∀ (v : ℕ+), ∃ (t : ℕ), submarinePosition v t = missilePosition t := by
  sorry


end submarine_hit_guaranteed_l3732_373252


namespace constant_sequence_l3732_373296

def sequence_condition (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ m n : ℕ, m > 0 → n > 0 → |a n - a m| ≤ (2 * m * n : ℝ) / ((m^2 + n^2) : ℝ)

theorem constant_sequence (a : ℕ → ℝ) (h : sequence_condition a) : ∀ n : ℕ, n > 0 → a n = 1 :=
sorry

end constant_sequence_l3732_373296


namespace whale_length_from_relative_speed_l3732_373275

/-- The length of a whale can be determined by the relative speed of two whales
    and the time taken for one to cross the other. -/
theorem whale_length_from_relative_speed (v_fast v_slow t : ℝ) (h1 : v_fast > v_slow) :
  (v_fast - v_slow) * t = (v_fast - v_slow) * 15 → v_fast = 18 → v_slow = 15 → (v_fast - v_slow) * 15 = 45 := by
  sorry

#check whale_length_from_relative_speed

end whale_length_from_relative_speed_l3732_373275


namespace isabels_bouquets_l3732_373245

theorem isabels_bouquets (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) :
  initial_flowers = 66 →
  flowers_per_bouquet = 8 →
  wilted_flowers = 10 →
  (initial_flowers - wilted_flowers) / flowers_per_bouquet = 7 :=
by sorry

end isabels_bouquets_l3732_373245


namespace beads_per_bracelet_l3732_373261

theorem beads_per_bracelet (num_friends : ℕ) (current_beads : ℕ) (additional_beads : ℕ) : 
  num_friends = 6 → 
  current_beads = 36 → 
  additional_beads = 12 → 
  (current_beads + additional_beads) / num_friends = 8 :=
by sorry

end beads_per_bracelet_l3732_373261


namespace min_vertical_distance_is_zero_l3732_373247

-- Define the functions
def f (x : ℝ) := abs x
def g (x : ℝ) := -x^2 - 5*x - 4

-- Define the vertical distance function
def verticalDistance (x : ℝ) := f x - g x

-- Theorem statement
theorem min_vertical_distance_is_zero :
  ∃ x : ℝ, verticalDistance x = 0 ∧ ∀ y : ℝ, verticalDistance y ≥ 0 :=
sorry

end min_vertical_distance_is_zero_l3732_373247


namespace total_share_l3732_373219

theorem total_share (z y x : ℝ) : 
  z = 300 →
  y = 1.2 * z →
  x = 1.25 * y →
  x + y + z = 1110 := by
sorry

end total_share_l3732_373219


namespace population_is_all_scores_l3732_373220

/-- Represents a participant in the math test. -/
structure Participant where
  id : Nat
  score : ℝ

/-- Represents the entire set of participants in the math test. -/
def AllParticipants : Set Participant :=
  { p : Participant | p.id ≤ 1000 }

/-- Represents the sample of participants whose scores are analyzed. -/
def SampleParticipants : Set Participant :=
  { p : Participant | p.id ≤ 100 }

/-- The population in the context of this statistical analysis. -/
def Population : Set ℝ :=
  { score | ∃ p ∈ AllParticipants, p.score = score }

/-- Theorem stating that the population refers to the math scores of all 1000 participants. -/
theorem population_is_all_scores :
  Population = { score | ∃ p ∈ AllParticipants, p.score = score } :=
by sorry

end population_is_all_scores_l3732_373220


namespace number_division_l3732_373257

theorem number_division (x : ℝ) : x + 8 = 88 → x / 10 = 8 := by
  sorry

end number_division_l3732_373257


namespace problem_solution_l3732_373208

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) : 
  (x^3 + 3*y^2) / 7 = 75/7 := by
sorry

end problem_solution_l3732_373208


namespace first_girl_siblings_l3732_373286

-- Define the number of girls in the survey
def num_girls : ℕ := 9

-- Define the mean number of siblings
def mean_siblings : ℚ := 5.7

-- Define the list of known sibling counts
def known_siblings : List ℕ := [6, 10, 4, 3, 3, 11, 3, 10]

-- Define the sum of known sibling counts
def sum_known_siblings : ℕ := known_siblings.sum

-- Theorem to prove
theorem first_girl_siblings :
  ∃ (x : ℕ), x + sum_known_siblings = Int.floor (mean_siblings * num_girls) ∧ x = 1 := by
  sorry


end first_girl_siblings_l3732_373286


namespace inverse_proportionality_l3732_373234

theorem inverse_proportionality (α β : ℚ) (h : α ≠ 0 ∧ β ≠ 0) :
  (∃ k : ℚ, k ≠ 0 ∧ α * β = k) →
  (α = -4 ∧ β = -8) →
  (β = 12 → α = 8/3) :=
by sorry

end inverse_proportionality_l3732_373234


namespace product_of_radicals_l3732_373293

theorem product_of_radicals (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q^3) = 14 * q^2 * Real.sqrt (42 * q) :=
by sorry

end product_of_radicals_l3732_373293


namespace paint_per_statue_l3732_373264

theorem paint_per_statue (total_paint : ℚ) (num_statues : ℕ) 
  (h1 : total_paint = 3/6)
  (h2 : num_statues = 3) :
  total_paint / num_statues = 1/2 := by
  sorry

end paint_per_statue_l3732_373264


namespace unique_solution_equation_l3732_373288

theorem unique_solution_equation (x : ℝ) : 
  x > 12 ∧ (x - 5) / 12 = 5 / (x - 12) ↔ x = 17 := by
  sorry

end unique_solution_equation_l3732_373288


namespace sum_of_roots_quadratic_undefined_expression_sum_l3732_373273

theorem sum_of_roots_quadratic : ∀ (a b c : ℝ), a ≠ 0 →
  let roots := {x : ℝ | a * x^2 + b * x + c = 0}
  (∃ x₁ x₂, roots = {x₁, x₂}) →
  (∃ s, ∀ x ∈ roots, ∃ y ∈ roots, x + y = s) →
  (∃ s, ∀ x ∈ roots, ∃ y ∈ roots, x + y = s) → s = -b / a :=
by sorry

theorem undefined_expression_sum : 
  let roots := {x : ℝ | x^2 - 7*x + 12 = 0}
  (∃ x₁ x₂, roots = {x₁, x₂}) →
  (∃ s, ∀ x ∈ roots, ∃ y ∈ roots, x + y = s) →
  s = 7 :=
by sorry

end sum_of_roots_quadratic_undefined_expression_sum_l3732_373273


namespace smallest_b_in_arithmetic_sequence_l3732_373200

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- All terms are positive
  ∃ (d : ℝ), a = b - d ∧ c = b + d →  -- Terms form an arithmetic sequence
  a * b * c = 125 →  -- Product is 125
  ∀ x : ℝ, (x > 0 ∧ 
    (∃ (y z d : ℝ), y > 0 ∧ z > 0 ∧ 
      y = x - d ∧ z = x + d ∧ 
      y * x * z = 125)) → 
    x ≥ 10 :=
by sorry

end smallest_b_in_arithmetic_sequence_l3732_373200


namespace fractional_equation_solution_range_l3732_373270

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (m + 3) / (x - 1) = 1) → 
  (m > -4 ∧ m ≠ -3) :=
by sorry

end fractional_equation_solution_range_l3732_373270


namespace remainder_theorem_l3732_373297

/-- The polynomial for which we want to find the remainder -/
def f (x : ℝ) : ℝ := 5*x^5 - 8*x^4 + 3*x^3 - x^2 + 4*x - 15

/-- The theorem stating that the remainder of f(x) divided by (x - 2) is 45 -/
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + 45 :=
sorry

end remainder_theorem_l3732_373297


namespace inequality_system_solution_l3732_373213

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 ↔ (3 * x - 1 ≤ a ∧ 2 * x ≥ 6 - b)) →
  a + b = 13 := by
sorry

end inequality_system_solution_l3732_373213


namespace tangent_line_slope_l3732_373282

/-- Given a function f(x) = x^3 + ax^2 + x with a tangent line at (1, f(1)) having slope 6, 
    prove that a = 1. -/
theorem tangent_line_slope (a : ℝ) : 
  let f := λ x : ℝ => x^3 + a*x^2 + x
  let f' := λ x : ℝ => 3*x^2 + 2*a*x + 1
  f' 1 = 6 → a = 1 := by sorry

end tangent_line_slope_l3732_373282


namespace curve_is_ellipse_iff_k_in_range_l3732_373222

/-- The curve equation: x^2 / (4 + k) + y^2 / (1 - k) = 1 -/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 / (4 + k) + y^2 / (1 - k) = 1

/-- The range of k values for which the curve represents an ellipse -/
def ellipse_k_range (k : ℝ) : Prop :=
  (k > -4 ∧ k < -3/2) ∨ (k > -3/2 ∧ k < 1)

/-- Theorem stating that the curve represents an ellipse if and only if k is in the specified range -/
theorem curve_is_ellipse_iff_k_in_range :
  ∀ k : ℝ, (∃ x y : ℝ, curve_equation x y k) ↔ ellipse_k_range k :=
sorry

end curve_is_ellipse_iff_k_in_range_l3732_373222
