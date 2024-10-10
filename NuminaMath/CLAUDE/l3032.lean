import Mathlib

namespace salt_concentration_change_l3032_303232

/-- Proves that adding 1.25 kg of pure salt to 20 kg of 15% saltwater results in 20% saltwater -/
theorem salt_concentration_change (initial_water : ℝ) (initial_concentration : ℝ) 
  (added_salt : ℝ) (final_concentration : ℝ) 
  (h1 : initial_water = 20)
  (h2 : initial_concentration = 0.15)
  (h3 : added_salt = 1.25)
  (h4 : final_concentration = 0.2) :
  initial_water * initial_concentration + added_salt = 
  (initial_water + added_salt) * final_concentration :=
by sorry

end salt_concentration_change_l3032_303232


namespace table_sum_theorem_l3032_303282

/-- A 3x3 table filled with numbers from 1 to 9 -/
def Table := Fin 3 → Fin 3 → Fin 9

/-- The sum of elements along a diagonal -/
def diagonalSum (t : Table) (main : Bool) : Nat :=
  if main then t 0 0 + t 1 1 + t 2 2 else t 0 2 + t 1 1 + t 2 0

/-- The sum of elements in the specified cells -/
def specifiedSum (t : Table) : Nat :=
  t 1 0 + t 1 1 + t 1 2 + t 2 1 + t 2 2

/-- All numbers from 1 to 9 appear exactly once in the table -/
def isValid (t : Table) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), t i j = n

theorem table_sum_theorem (t : Table) (h_valid : isValid t) 
  (h_diag1 : diagonalSum t true = 7) (h_diag2 : diagonalSum t false = 21) :
  specifiedSum t = 25 := by
  sorry

end table_sum_theorem_l3032_303282


namespace bookstore_problem_l3032_303255

theorem bookstore_problem (total_notebooks : ℕ) (cost_A cost_B total_cost : ℚ) 
  (sell_A sell_B : ℚ) (discount_A : ℚ) (profit_threshold : ℚ) :
  total_notebooks = 350 →
  cost_A = 12 →
  cost_B = 15 →
  total_cost = 4800 →
  sell_A = 20 →
  sell_B = 25 →
  discount_A = 0.7 →
  profit_threshold = 2348 →
  ∃ (num_A num_B : ℕ) (m : ℕ),
    num_A + num_B = total_notebooks ∧
    num_A * cost_A + num_B * cost_B = total_cost ∧
    num_A = 150 ∧
    m * sell_A + m * sell_B + (num_A - m) * sell_A * discount_A + (num_B - m) * cost_B - total_cost ≥ profit_threshold ∧
    ∀ k : ℕ, k < m → k * sell_A + k * sell_B + (num_A - k) * sell_A * discount_A + (num_B - k) * cost_B - total_cost < profit_threshold :=
by sorry

end bookstore_problem_l3032_303255


namespace sum_of_binary_digits_300_l3032_303256

/-- The sum of the digits in the binary representation of 300 is 4 -/
theorem sum_of_binary_digits_300 : 
  (Nat.digits 2 300).sum = 4 := by
  sorry

end sum_of_binary_digits_300_l3032_303256


namespace exists_non_intersecting_circle_exists_regular_polygon_M_properties_l3032_303207

/-- Line system M: x cos θ + (y-1) sin θ = 1, where 0 ≤ θ ≤ 2π -/
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ p.1 * Real.cos θ + (p.2 - 1) * Real.sin θ = 1}

/-- There exists a circle that does not intersect any of the lines in M -/
theorem exists_non_intersecting_circle (M : Set (ℝ × ℝ)) : 
  ∃ (c : ℝ × ℝ) (r : ℝ), ∀ p ∈ M, (p.1 - c.1)^2 + (p.2 - c.2)^2 > r^2 := by sorry

/-- For any integer n ≥ 3, there exists a regular n-sided polygon whose edges all lie on lines in M -/
theorem exists_regular_polygon (M : Set (ℝ × ℝ)) (n : ℕ) (hn : n ≥ 3) :
  ∃ (polygon : Fin n → ℝ × ℝ), 
    (∀ i : Fin n, ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ 
      (polygon i).1 * Real.cos θ + ((polygon i).2 - 1) * Real.sin θ = 1) ∧
    (∀ i j : Fin n, (polygon i).1^2 + (polygon i).2^2 = (polygon j).1^2 + (polygon j).2^2) := by sorry

/-- Main theorem combining the two properties -/
theorem M_properties : 
  (∃ (c : ℝ × ℝ) (r : ℝ), ∀ p ∈ M, (p.1 - c.1)^2 + (p.2 - c.2)^2 > r^2) ∧
  (∀ (n : ℕ), n ≥ 3 → 
    ∃ (polygon : Fin n → ℝ × ℝ), 
      (∀ i : Fin n, ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ 
        (polygon i).1 * Real.cos θ + ((polygon i).2 - 1) * Real.sin θ = 1) ∧
      (∀ i j : Fin n, (polygon i).1^2 + (polygon i).2^2 = (polygon j).1^2 + (polygon j).2^2)) := by
  sorry

end exists_non_intersecting_circle_exists_regular_polygon_M_properties_l3032_303207


namespace Jose_age_is_14_l3032_303269

-- Define the ages as natural numbers
def Inez_age : ℕ := 15
def Zack_age : ℕ := Inez_age + 3
def Jose_age : ℕ := Zack_age - 4

-- Theorem statement
theorem Jose_age_is_14 : Jose_age = 14 := by
  sorry

end Jose_age_is_14_l3032_303269


namespace complement_M_in_U_l3032_303211

def U : Set ℕ := {x | x < 5 ∧ x > 0}
def M : Set ℕ := {x | x^2 - 5*x + 6 = 0}

theorem complement_M_in_U : (U \ M) = {1, 4} := by sorry

end complement_M_in_U_l3032_303211


namespace base_8_5624_equals_2964_l3032_303203

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_5624_equals_2964 : 
  base_8_to_10 [4, 2, 6, 5] = 2964 := by
  sorry

end base_8_5624_equals_2964_l3032_303203


namespace riding_ratio_is_half_l3032_303262

/-- Represents the number of horses and men -/
def total_count : ℕ := 14

/-- Represents the number of legs walking on the ground -/
def legs_on_ground : ℕ := 70

/-- Represents the number of legs a horse has -/
def horse_legs : ℕ := 4

/-- Represents the number of legs a man has -/
def man_legs : ℕ := 2

/-- Represents the number of owners riding their horses -/
def riding_owners : ℕ := (total_count * horse_legs - legs_on_ground) / (horse_legs - man_legs)

/-- Represents the ratio of riding owners to total owners -/
def riding_ratio : ℚ := riding_owners / total_count

theorem riding_ratio_is_half : riding_ratio = 1 / 2 := by
  sorry

end riding_ratio_is_half_l3032_303262


namespace cubic_equation_roots_difference_l3032_303257

theorem cubic_equation_roots_difference (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   x^3 + 3*p*x^2 + (4*p - 1)*x + p = 0 ∧ 
   y^3 + 3*p*y^2 + (4*p - 1)*y + p = 0 ∧ 
   y - x = 1) ↔ 
  (p = 0 ∨ p = 6/5 ∨ p = 10/9) :=
sorry

end cubic_equation_roots_difference_l3032_303257


namespace circle_parameter_range_l3032_303245

theorem circle_parameter_range (a : ℝ) : 
  (∃ (h : ℝ) (k : ℝ) (r : ℝ), ∀ (x y : ℝ), 
    x^2 + y^2 + 2*x - 4*y + a + 1 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) → 
  a < 4 := by
sorry

end circle_parameter_range_l3032_303245


namespace factorization_of_cubic_l3032_303223

theorem factorization_of_cubic (x : ℝ) : 4 * x^3 - x = x * (2*x + 1) * (2*x - 1) := by
  sorry

end factorization_of_cubic_l3032_303223


namespace least_reducible_fraction_l3032_303244

def is_reducible (n : ℕ) : Prop :=
  n > 17 ∧ Nat.gcd (n - 17) (7 * n + 4) > 1

theorem least_reducible_fraction :
  (∀ m : ℕ, m < 20 → ¬ is_reducible m) ∧ is_reducible 20 :=
sorry

end least_reducible_fraction_l3032_303244


namespace z_in_fourth_quadrant_l3032_303280

/-- The complex number z -/
def z : ℂ := (2 - Complex.I) ^ 2

/-- Theorem: The point corresponding to z is in the fourth quadrant -/
theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end z_in_fourth_quadrant_l3032_303280


namespace canoe_kayak_rental_difference_l3032_303258

theorem canoe_kayak_rental_difference :
  ∀ (canoe_cost kayak_cost : ℚ) 
    (canoe_count kayak_count : ℕ) 
    (total_revenue : ℚ),
  canoe_cost = 12 →
  kayak_cost = 18 →
  canoe_count = (3 * kayak_count) / 2 →
  total_revenue = canoe_cost * canoe_count + kayak_cost * kayak_count →
  total_revenue = 504 →
  canoe_count - kayak_count = 7 :=
by
  sorry

end canoe_kayak_rental_difference_l3032_303258


namespace no_simultaneous_integer_fractions_l3032_303243

theorem no_simultaneous_integer_fractions :
  ¬ ∃ (n : ℤ), (∃ (a b : ℤ), (n - 6 : ℚ) / 15 = a ∧ (n - 5 : ℚ) / 24 = b) :=
by sorry

end no_simultaneous_integer_fractions_l3032_303243


namespace triangle_problem_l3032_303214

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 2 * t.a * Real.cos t.A * Real.cos t.B - 2 * t.b * Real.sin t.A * Real.sin t.A)
  (h2 : t.a * t.b * Real.sin t.C / 2 = 15 * Real.sqrt 3 / 4)
  (h3 : t.a + t.b + t.c = 15) :
  t.C = 2 * Real.pi / 3 ∧ t.c = 7 := by
  sorry


end triangle_problem_l3032_303214


namespace number_of_elements_in_set_l3032_303246

theorem number_of_elements_in_set (initial_average : ℝ) (incorrect_number : ℝ) (correct_number : ℝ) (correct_average : ℝ) :
  initial_average = 16 ∧ 
  incorrect_number = 26 ∧ 
  correct_number = 46 ∧ 
  correct_average = 18 →
  ∃ n : ℕ, n = 10 ∧ 
    n * initial_average = (n - 1) * initial_average + incorrect_number ∧
    n * correct_average = (n - 1) * initial_average + correct_number :=
by sorry

end number_of_elements_in_set_l3032_303246


namespace boxes_with_neither_l3032_303254

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (crayons : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : markers = 8)
  (h3 : crayons = 5)
  (h4 : both = 3) :
  total - (markers + crayons - both) = 5 :=
by
  sorry

end boxes_with_neither_l3032_303254


namespace angle_complement_supplement_difference_l3032_303252

theorem angle_complement_supplement_difference : 
  ∀ α : ℝ, (90 - α) - (180 - α) = 90 :=
by
  sorry

end angle_complement_supplement_difference_l3032_303252


namespace green_tea_leaves_needed_l3032_303213

/-- The number of sprigs of mint added to each batch of mud -/
def sprigs_of_mint : ℕ := 3

/-- The number of green tea leaves added per sprig of mint -/
def leaves_per_sprig : ℕ := 2

/-- The factor by which the efficacy of ingredients is reduced in the new mud -/
def efficacy_reduction : ℚ := 1/2

/-- The number of green tea leaves needed for the new batch of mud to maintain the same efficacy -/
def new_leaves_needed : ℕ := 12

/-- Theorem stating that the number of green tea leaves needed for the new batch of mud
    to maintain the same efficacy is equal to 12 -/
theorem green_tea_leaves_needed :
  (sprigs_of_mint * leaves_per_sprig : ℚ) / efficacy_reduction = new_leaves_needed := by
  sorry

end green_tea_leaves_needed_l3032_303213


namespace symmetric_quadratic_inequality_l3032_303299

/-- A quadratic function with positive leading coefficient and symmetric about x = 2 -/
def SymmetricQuadratic (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, a > 0 ∧ ∀ x, f x = a * x^2 + b * x + c) ∧
  (∀ x, f (2 + x) = f (2 - x))

theorem symmetric_quadratic_inequality
  (f : ℝ → ℝ) (h : SymmetricQuadratic f) (x : ℝ) :
  f (1 - 2 * x^2) < f (1 + 2 * x - x^2) → -2 < x ∧ x < 0 := by
  sorry

end symmetric_quadratic_inequality_l3032_303299


namespace max_an_over_n_is_half_l3032_303233

/-- The number of trailing zeroes in the base-n representation of n! -/
def a (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the maximum value of a_n/n is 1/2 -/
theorem max_an_over_n_is_half :
  (∀ n > 1, (a n : ℚ) / n ≤ 1/2) ∧ (∃ n > 1, (a n : ℚ) / n = 1/2) :=
sorry

end max_an_over_n_is_half_l3032_303233


namespace number_division_problem_l3032_303209

theorem number_division_problem (x : ℝ) : (x / 5 = 75 + x / 6) ↔ (x = 2250) := by sorry

end number_division_problem_l3032_303209


namespace third_day_distance_is_15_l3032_303260

/-- Represents a three-day hike with given distances --/
structure ThreeDayHike where
  total_distance : ℝ
  first_day_distance : ℝ
  second_day_distance : ℝ

/-- Calculates the distance hiked on the third day --/
def third_day_distance (hike : ThreeDayHike) : ℝ :=
  hike.total_distance - hike.first_day_distance - hike.second_day_distance

/-- Theorem: The distance hiked on the third day is 15 kilometers --/
theorem third_day_distance_is_15 (hike : ThreeDayHike)
    (h1 : hike.total_distance = 50)
    (h2 : hike.first_day_distance = 10)
    (h3 : hike.second_day_distance = hike.total_distance / 2) :
    third_day_distance hike = 15 := by
  sorry

end third_day_distance_is_15_l3032_303260


namespace six_balls_four_boxes_l3032_303241

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def num_distributions (balls : ℕ) (boxes : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 8 ways to distribute 6 indistinguishable balls into 4 indistinguishable boxes -/
theorem six_balls_four_boxes : num_distributions 6 4 = 8 := by
  sorry

end six_balls_four_boxes_l3032_303241


namespace sand_amount_l3032_303296

/-- The total amount of sand in tons -/
def total_sand : ℕ := 180

/-- The originally scheduled daily transport rate in tons -/
def scheduled_rate : ℕ := 15

/-- The actual daily transport rate in tons -/
def actual_rate : ℕ := 20

/-- The number of days the task was completed ahead of schedule -/
def days_ahead : ℕ := 3

/-- Theorem stating that the total amount of sand is 180 tons -/
theorem sand_amount :
  ∃ (scheduled_days : ℕ),
    scheduled_days * scheduled_rate = total_sand ∧
    (scheduled_days - days_ahead) * actual_rate = total_sand :=
by sorry

end sand_amount_l3032_303296


namespace mike_savings_rate_l3032_303221

theorem mike_savings_rate (carol_initial : ℕ) (carol_weekly : ℕ) (mike_initial : ℕ) (weeks : ℕ) :
  carol_initial = 60 →
  carol_weekly = 9 →
  mike_initial = 90 →
  weeks = 5 →
  ∃ (mike_weekly : ℕ),
    carol_initial + carol_weekly * weeks = mike_initial + mike_weekly * weeks ∧
    mike_weekly = 3 :=
by sorry

end mike_savings_rate_l3032_303221


namespace quadratic_inequality_range_l3032_303263

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) ↔ a ∈ Set.Ioi 3 ∪ Set.Iio (-1) :=
sorry

end quadratic_inequality_range_l3032_303263


namespace r_earns_75_l3032_303220

/-- Represents the daily earnings of individuals p, q, r, and s -/
structure DailyEarnings where
  p : ℚ
  q : ℚ
  r : ℚ
  s : ℚ

/-- The conditions of the problem -/
def earnings_conditions (e : DailyEarnings) : Prop :=
  e.p + e.q + e.r + e.s = 2400 / 8 ∧
  e.p + e.r = 600 / 5 ∧
  e.q + e.r = 910 / 7 ∧
  e.s + e.r = 800 / 4 ∧
  e.p + e.s = 700 / 6

/-- Theorem stating that under the given conditions, r earns 75 per day -/
theorem r_earns_75 (e : DailyEarnings) : 
  earnings_conditions e → e.r = 75 := by
  sorry

#check r_earns_75

end r_earns_75_l3032_303220


namespace quadratic_inequality_solution_l3032_303215

theorem quadratic_inequality_solution (x : ℝ) :
  (-5 * x^2 + 10 * x - 3 > 0) ↔ (x > 1 - Real.sqrt 10 / 5 ∧ x < 1 + Real.sqrt 10 / 5) :=
by sorry

end quadratic_inequality_solution_l3032_303215


namespace cube_root_problem_l3032_303224

theorem cube_root_problem (a m : ℝ) (h1 : a > 0) 
  (h2 : (m + 7)^2 = a) (h3 : (2*m - 1)^2 = a) : 
  (a - m)^(1/3 : ℝ) = 3 := by
sorry

end cube_root_problem_l3032_303224


namespace infinite_primes_dividing_power_plus_a_l3032_303230

theorem infinite_primes_dividing_power_plus_a (a : ℕ) (ha : a > 0) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ 2^(2^n) + a} :=
by sorry

end infinite_primes_dividing_power_plus_a_l3032_303230


namespace fruit_profit_equation_l3032_303292

/-- Represents the profit equation for a fruit selling scenario -/
theorem fruit_profit_equation 
  (cost : ℝ) 
  (initial_price : ℝ) 
  (initial_volume : ℝ) 
  (price_increase : ℝ) 
  (volume_decrease : ℝ) 
  (profit : ℝ) :
  cost = 40 →
  initial_price = 50 →
  initial_volume = 500 →
  price_increase > 0 →
  volume_decrease = 10 * price_increase →
  profit = 8000 →
  ∃ x : ℝ, x > 50 ∧ (x - cost) * (initial_volume - volume_decrease) = profit :=
by sorry

end fruit_profit_equation_l3032_303292


namespace opposite_face_of_ten_l3032_303229

/-- Represents a cube with six faces labeled with distinct integers -/
structure Cube where
  faces : Finset ℕ
  distinct : faces.card = 6
  range : ∀ n ∈ faces, 6 ≤ n ∧ n ≤ 11

/-- The sum of all numbers on the cube's faces -/
def Cube.total_sum (c : Cube) : ℕ := c.faces.sum id

/-- Represents a roll of the cube, showing four lateral faces -/
structure Roll (c : Cube) where
  lateral_sum : ℕ
  valid : lateral_sum = c.total_sum - (c.faces.sum id - lateral_sum)

theorem opposite_face_of_ten (c : Cube) 
  (roll1 : Roll c) (roll2 : Roll c)
  (h1 : roll1.lateral_sum = 36)
  (h2 : roll2.lateral_sum = 33)
  : ∃ n ∈ c.faces, n = 8 ∧ (c.faces.sum id - (10 + n) = roll1.lateral_sum ∨ 
                            c.faces.sum id - (10 + n) = roll2.lateral_sum) :=
sorry

end opposite_face_of_ten_l3032_303229


namespace remainder_theorem_l3032_303278

-- Define the polynomial Q
variable (Q : ℝ → ℝ)

-- Define the conditions
axiom rem_20 : ∃ P : ℝ → ℝ, ∀ x, Q x = (x - 20) * (P x) + 120
axiom rem_100 : ∃ P : ℝ → ℝ, ∀ x, Q x = (x - 100) * (P x) + 40

-- Theorem statement
theorem remainder_theorem :
  ∃ R : ℝ → ℝ, ∀ x, Q x = (x - 20) * (x - 100) * (R x) + (-x + 140) :=
sorry

end remainder_theorem_l3032_303278


namespace smallest_possible_a_l3032_303265

theorem smallest_possible_a (a b c : ℝ) : 
  (∃ (x y : ℝ), y = a * (x - 1/3)^2 - 1/4) →  -- parabola with vertex (1/3, -1/4)
  (∃ (x y : ℝ), y = a * x^2 + b * x + c) →    -- equation of parabola
  (a > 0) →                                   -- a is positive
  (∃ (n : ℤ), 2 * a + b + 3 * c = n) →        -- 2a + b + 3c is an integer
  (∀ (a' : ℝ), a' ≥ 9/16 ∨ ¬(
    (∃ (x y : ℝ), y = a' * (x - 1/3)^2 - 1/4) ∧
    (∃ (x y : ℝ), y = a' * x^2 + b * x + c) ∧
    (a' > 0) ∧
    (∃ (n : ℤ), 2 * a' + b + 3 * c = n)
  )) :=
by sorry

end smallest_possible_a_l3032_303265


namespace ratio_closest_to_nine_l3032_303227

theorem ratio_closest_to_nine : 
  ∀ n : ℕ, |((10^3000 + 10^3003) : ℝ) / (10^3001 + 10^3002) - 9| ≤ 
           |((10^3000 + 10^3003) : ℝ) / (10^3001 + 10^3002) - n| :=
by sorry

end ratio_closest_to_nine_l3032_303227


namespace some_number_value_l3032_303261

theorem some_number_value (x : ℝ) : 40 + 5 * 12 / (180 / x) = 41 → x = 3 := by
  sorry

end some_number_value_l3032_303261


namespace sin_pi_half_plus_two_alpha_l3032_303251

theorem sin_pi_half_plus_two_alpha (y₀ : ℝ) (α : ℝ) : 
  (1/2)^2 + y₀^2 = 1 → 
  Real.cos α = 1/2 →
  Real.sin (π/2 + 2*α) = -1/2 := by
sorry

end sin_pi_half_plus_two_alpha_l3032_303251


namespace min_value_sum_l3032_303240

theorem min_value_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 8) :
  x + 3 * y + 6 * z ≥ 18 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 8 ∧ x₀ + 3 * y₀ + 6 * z₀ = 18 :=
sorry

end min_value_sum_l3032_303240


namespace patricia_money_l3032_303276

theorem patricia_money (jethro carmen patricia : ℕ) : 
  carmen = 2 * jethro - 7 →
  patricia = 3 * jethro →
  jethro + carmen + patricia = 113 →
  patricia = 60 := by
sorry

end patricia_money_l3032_303276


namespace intimate_interval_is_two_three_l3032_303259

def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

def intimate_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, |f x - g x| ≤ 1

theorem intimate_interval_is_two_three :
  ∃ (a b : ℝ), a = 2 ∧ b = 3 ∧
  intimate_functions f g a b ∧
  ∀ (c d : ℝ), c < 2 ∨ d > 3 → ¬intimate_functions f g c d :=
sorry

end intimate_interval_is_two_three_l3032_303259


namespace smallest_k_for_divisibility_by_2010_l3032_303297

theorem smallest_k_for_divisibility_by_2010 :
  ∃ (k : ℕ), k > 1 ∧
  (∀ (n : ℕ), n > 0 → (n^k - n) % 2010 = 0) ∧
  (∀ (m : ℕ), m > 1 ∧ m < k → ∃ (n : ℕ), n > 0 ∧ (n^m - n) % 2010 ≠ 0) ∧
  k = 133 := by
  sorry

end smallest_k_for_divisibility_by_2010_l3032_303297


namespace hexagonal_prism_diagonals_truncated_cube_diagonals_l3032_303238

/- Right hexagonal prism -/
theorem hexagonal_prism_diagonals (n : Nat) (v : Nat) (d : Nat) :
  n = 12 → v = 3 → d = n * v / 2 → d = 18 := by sorry

/- Truncated cube -/
theorem truncated_cube_diagonals (n : Nat) (v : Nat) (d : Nat) :
  n = 24 → v = 10 → d = n * v / 2 → d = 120 := by sorry

end hexagonal_prism_diagonals_truncated_cube_diagonals_l3032_303238


namespace circular_arrangement_students_l3032_303266

/-- Given a circular arrangement of students, if the 7th and 27th positions
    are opposite each other, then the total number of students is 40. -/
theorem circular_arrangement_students (n : ℕ) : 
  (7 + n / 2 = 27 ∨ 27 + n / 2 = n + 7) → n = 40 :=
by sorry

end circular_arrangement_students_l3032_303266


namespace prime_sum_product_l3032_303277

theorem prime_sum_product (x y z : ℕ) : 
  Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z ∧
  x ≤ y ∧ y ≤ z ∧
  x + y + z = 12 ∧
  x * y + y * z + x * z = 41 →
  x + 2 * y + 3 * z = 29 := by
sorry

end prime_sum_product_l3032_303277


namespace second_solution_concentration_l3032_303226

/-- 
Given two solutions that are mixed to form a new solution,
this theorem proves that the concentration of the second solution
must be 10% under the specified conditions.
-/
theorem second_solution_concentration
  (volume_first : ℝ)
  (concentration_first : ℝ)
  (volume_second : ℝ)
  (concentration_final : ℝ)
  (h1 : volume_first = 4)
  (h2 : concentration_first = 0.04)
  (h3 : volume_second = 2)
  (h4 : concentration_final = 0.06)
  (h5 : volume_first * concentration_first + volume_second * (concentration_second / 100) = 
        (volume_first + volume_second) * concentration_final) :
  concentration_second = 10 := by
  sorry

#check second_solution_concentration

end second_solution_concentration_l3032_303226


namespace rogers_shelves_l3032_303279

/-- Given the conditions of Roger's book shelving problem, prove that he needs 4 shelves. -/
theorem rogers_shelves (total_books : ℕ) (librarian_books : ℕ) (books_per_shelf : ℕ) 
  (h1 : total_books = 14) 
  (h2 : librarian_books = 2) 
  (h3 : books_per_shelf = 3) : 
  ((total_books - librarian_books) / books_per_shelf : ℕ) = 4 := by
  sorry

end rogers_shelves_l3032_303279


namespace gcd_nine_factorial_six_factorial_squared_l3032_303239

theorem gcd_nine_factorial_six_factorial_squared : Nat.gcd (Nat.factorial 9) ((Nat.factorial 6)^2) = 51840 := by
  sorry

end gcd_nine_factorial_six_factorial_squared_l3032_303239


namespace rest_albums_count_l3032_303236

def total_pictures : ℕ := 25
def first_album_pictures : ℕ := 10
def pictures_per_remaining_album : ℕ := 3

theorem rest_albums_count : 
  (total_pictures - first_album_pictures) / pictures_per_remaining_album = 5 := by
  sorry

end rest_albums_count_l3032_303236


namespace eugene_shoes_count_l3032_303291

/-- The cost of a T-shirt before discount -/
def t_shirt_cost : ℚ := 20

/-- The cost of a pair of pants before discount -/
def pants_cost : ℚ := 80

/-- The cost of a pair of shoes before discount -/
def shoes_cost : ℚ := 150

/-- The discount rate applied to all items -/
def discount_rate : ℚ := 1/10

/-- The number of T-shirts Eugene buys -/
def num_tshirts : ℕ := 4

/-- The number of pairs of pants Eugene buys -/
def num_pants : ℕ := 3

/-- The total amount Eugene pays -/
def total_paid : ℚ := 558

/-- The function to calculate the discounted price -/
def discounted_price (price : ℚ) : ℚ := price * (1 - discount_rate)

/-- The theorem stating the number of pairs of shoes Eugene buys -/
theorem eugene_shoes_count :
  ∃ (n : ℕ), n * discounted_price shoes_cost = 
    total_paid - (num_tshirts * discounted_price t_shirt_cost + num_pants * discounted_price pants_cost) ∧
    n = 2 := by sorry

end eugene_shoes_count_l3032_303291


namespace article_cost_l3032_303283

/-- Proves that the cost of an article is 80, given the specified conditions -/
theorem article_cost (original_profit_percent : Real) (reduced_cost_percent : Real)
  (price_reduction : Real) (new_profit_percent : Real)
  (h1 : original_profit_percent = 25)
  (h2 : reduced_cost_percent = 20)
  (h3 : price_reduction = 16.80)
  (h4 : new_profit_percent = 30) :
  ∃ (cost : Real), cost = 80 ∧
    (cost * (1 + original_profit_percent / 100) - price_reduction =
     (cost * (1 - reduced_cost_percent / 100)) * (1 + new_profit_percent / 100)) :=
by sorry

end article_cost_l3032_303283


namespace vector_magnitude_l3032_303289

def a : ℝ × ℝ := (1, 1)
def b : ℝ → ℝ × ℝ := λ y ↦ (3, y)

theorem vector_magnitude (y : ℝ) : 
  (∃ k : ℝ, b y - a = k • a) → ‖b y - a‖ = 2 * Real.sqrt 2 := by
  sorry

end vector_magnitude_l3032_303289


namespace smoking_chronic_bronchitis_relationship_l3032_303202

-- Define the confidence level
def confidence_level : Real := 0.99

-- Define the relationship between smoking and chronic bronchitis
def smoking_related_to_chronic_bronchitis : Prop := True

-- Define a sample of smokers
def sample_size : Nat := 100

-- Define the possibility of no chronic bronchitis cases in the sample
def possible_no_cases : Prop := True

-- Theorem statement
theorem smoking_chronic_bronchitis_relationship 
  (h1 : confidence_level > 0.99)
  (h2 : smoking_related_to_chronic_bronchitis) :
  possible_no_cases := by
  sorry

end smoking_chronic_bronchitis_relationship_l3032_303202


namespace min_value_expression_l3032_303275

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a^2 / b + b^2 / c + c^2 / a ≥ 3 ∧
  (a^2 / b + b^2 / c + c^2 / a = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end min_value_expression_l3032_303275


namespace last_ball_is_red_l3032_303273

/-- Represents the color of a ball -/
inductive BallColor
  | Blue
  | Red
  | Green

/-- Represents the state of the bottle -/
structure BottleState where
  blue : Nat
  red : Nat
  green : Nat

/-- Represents a single ball removal operation -/
inductive RemovalOperation
  | BlueGreen
  | RedGreen
  | TwoRed
  | Other

/-- Defines the initial state of the bottle -/
def initialState : BottleState :=
  { blue := 1001, red := 1000, green := 1000 }

/-- Applies a single removal operation to the bottle state -/
def applyOperation (state : BottleState) (op : RemovalOperation) : BottleState :=
  match op with
  | RemovalOperation.BlueGreen => { blue := state.blue - 1, red := state.red + 1, green := state.green - 1 }
  | RemovalOperation.RedGreen => { blue := state.blue, red := state.red, green := state.green - 1 }
  | RemovalOperation.TwoRed => { blue := state.blue + 2, red := state.red - 2, green := state.green }
  | RemovalOperation.Other => { blue := state.blue, red := state.red, green := state.green - 1 }

/-- Determines if the game has ended (only one ball left) -/
def isGameOver (state : BottleState) : Bool :=
  state.blue + state.red + state.green = 1

/-- Theorem: The last remaining ball is red -/
theorem last_ball_is_red :
  ∃ (operations : List RemovalOperation),
    let finalState := operations.foldl applyOperation initialState
    isGameOver finalState ∧ finalState.red = 1 :=
  sorry


end last_ball_is_red_l3032_303273


namespace common_ratio_sum_l3032_303281

theorem common_ratio_sum (k p r : ℝ) (h1 : k ≠ 0) (h2 : p ≠ 1) (h3 : r ≠ 1) (h4 : p ≠ r) 
  (h5 : k * p^2 - k * r^2 = 2 * (k * p - k * r)) : 
  p + r = 2 := by
  sorry

end common_ratio_sum_l3032_303281


namespace student_count_correct_l3032_303212

/-- Represents the changes in student numbers for a grade --/
structure GradeChanges where
  initial : Nat
  left : Nat
  joined : Nat
  transferredIn : Nat
  transferredOut : Nat

/-- Calculates the final number of students in a grade --/
def finalStudents (changes : GradeChanges) : Nat :=
  changes.initial - changes.left + changes.joined + changes.transferredIn - changes.transferredOut

/-- Theorem: The calculated final numbers of students in each grade and their total are correct --/
theorem student_count_correct (fourth : GradeChanges) (fifth : GradeChanges) (sixth : GradeChanges) 
    (h4 : fourth = ⟨4, 3, 42, 0, 10⟩)
    (h5 : fifth = ⟨10, 5, 25, 10, 5⟩)
    (h6 : sixth = ⟨15, 7, 30, 5, 0⟩) : 
    finalStudents fourth = 33 ∧ 
    finalStudents fifth = 35 ∧ 
    finalStudents sixth = 43 ∧
    finalStudents fourth + finalStudents fifth + finalStudents sixth = 111 := by
  sorry

end student_count_correct_l3032_303212


namespace mean_diesel_cost_l3032_303298

def diesel_rates : List ℝ := [1.2, 1.3, 1.8, 2.1]

theorem mean_diesel_cost (rates : List ℝ) (h : rates = diesel_rates) :
  (rates.sum / rates.length : ℝ) = 1.6 := by
  sorry

end mean_diesel_cost_l3032_303298


namespace negative_fraction_comparison_l3032_303284

theorem negative_fraction_comparison : -5/6 < -7/9 := by
  sorry

end negative_fraction_comparison_l3032_303284


namespace rectangle_width_on_square_diagonal_l3032_303295

theorem rectangle_width_on_square_diagonal (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let diagonal := s * Real.sqrt 2
  let rectangle_length := diagonal
  let rectangle_width := s / Real.sqrt 2
  square_area = rectangle_length * rectangle_width :=
by sorry

end rectangle_width_on_square_diagonal_l3032_303295


namespace addition_closed_in_P_l3032_303217

-- Define the set P
def P : Set ℝ := {n | ∃ k : ℕ+, n = Real.log k}

-- State the theorem
theorem addition_closed_in_P (a b : ℝ) (ha : a ∈ P) (hb : b ∈ P) : 
  a + b ∈ P := by sorry

end addition_closed_in_P_l3032_303217


namespace s2_side_length_l3032_303231

/-- A composite rectangle structure -/
structure CompositeRectangle where
  width : ℕ
  height : ℕ
  s2_side : ℕ

/-- The composite rectangle satisfies the given conditions -/
def satisfies_conditions (cr : CompositeRectangle) : Prop :=
  cr.width = 3782 ∧ cr.height = 2260 ∧
  ∃ (r : ℕ), 2 * r + cr.s2_side = cr.height ∧ 2 * r + 3 * cr.s2_side = cr.width

/-- Theorem: The side length of S2 in the composite rectangle is 761 units -/
theorem s2_side_length :
  ∀ (cr : CompositeRectangle), satisfies_conditions cr → cr.s2_side = 761 :=
by
  sorry

end s2_side_length_l3032_303231


namespace student_D_most_stable_l3032_303208

-- Define the students
inductive Student : Type
  | A
  | B
  | C
  | D

-- Define the variance function
def variance : Student → Real
  | Student.A => 2.1
  | Student.B => 3.5
  | Student.C => 9
  | Student.D => 0.7

-- Define the concept of stability
def most_stable (s : Student) : Prop :=
  ∀ t : Student, variance s ≤ variance t

-- Theorem statement
theorem student_D_most_stable :
  most_stable Student.D :=
by sorry

end student_D_most_stable_l3032_303208


namespace a_n_property_smallest_n_for_perfect_square_sum_l3032_303218

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

def is_sum_or_diff_of_squares (x : ℕ) : Prop :=
  ∃ a b : ℕ, x = a * a + b * b ∨ x = a * a - b * b ∨ x = b * b - a * a

def largest_n_digit_number (n : ℕ) : ℕ := 10^n - 1

def a_n (n : ℕ) : ℕ := 10^n - 2

def sum_of_squares_of_digits (x : ℕ) : ℕ :=
  (x.digits 10).map (λ d => d * d) |>.sum

theorem a_n_property (n : ℕ) (h : n > 2) :
  ¬(is_sum_or_diff_of_squares (a_n n)) ∧
  ∀ m : ℕ, m > a_n n → m ≤ largest_n_digit_number n → is_sum_or_diff_of_squares m :=
sorry

theorem smallest_n_for_perfect_square_sum :
  ∀ n : ℕ, n < 66 → ¬(is_perfect_square (sum_of_squares_of_digits (a_n n))) ∧
  is_perfect_square (sum_of_squares_of_digits (a_n 66)) :=
sorry

end a_n_property_smallest_n_for_perfect_square_sum_l3032_303218


namespace f_three_point_five_l3032_303206

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def periodic_neg (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

def identity_on_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 < x → x < 1 → f x = x

theorem f_three_point_five 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : periodic_neg f) 
  (h_identity : identity_on_interval f) : 
  f 3.5 = -0.5 := by
sorry

end f_three_point_five_l3032_303206


namespace base8_palindrome_count_l3032_303271

/-- Represents a digit in base 8 -/
def Base8Digit := Fin 8

/-- Represents a six-digit palindrome in base 8 -/
structure Base8Palindrome where
  a : Base8Digit
  b : Base8Digit
  c : Base8Digit
  d : Base8Digit
  h : a.val ≠ 0

/-- The count of six-digit palindromes in base 8 -/
def count_base8_palindromes : Nat :=
  (Finset.range 7).card * (Finset.range 8).card * (Finset.range 8).card * (Finset.range 8).card

theorem base8_palindrome_count :
  count_base8_palindromes = 3584 :=
sorry

end base8_palindrome_count_l3032_303271


namespace walk_bike_time_difference_l3032_303242

def blocks : ℕ := 18
def walk_time_per_block : ℚ := 1
def bike_time_per_block : ℚ := 20 / 60

theorem walk_bike_time_difference :
  (blocks * walk_time_per_block) - (blocks * bike_time_per_block) = 12 := by
  sorry

end walk_bike_time_difference_l3032_303242


namespace parabola_properties_l3032_303290

/-- Definition of the parabola function -/
def f (x : ℝ) : ℝ := 2 * (x + 1)^2 - 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-1, -3)

/-- The x-coordinate of the axis of symmetry -/
def axis_of_symmetry : ℝ := -1

/-- Theorem stating that the given vertex and axis of symmetry are correct for the parabola -/
theorem parabola_properties :
  (∀ x, f x ≥ f (vertex.1)) ∧
  (∀ x, f x = f (2 * axis_of_symmetry - x)) := by
  sorry

end parabola_properties_l3032_303290


namespace smallest_number_with_divisibility_property_l3032_303200

theorem smallest_number_with_divisibility_property : 
  ∀ n : ℕ, n > 0 → (n + 9) % 8 = 0 ∧ (n + 9) % 11 = 0 ∧ (∃ k : ℕ, k > 1 ∧ k ≠ 8 ∧ k ≠ 11 ∧ n % k = 0) → n ≥ 255 :=
by sorry

end smallest_number_with_divisibility_property_l3032_303200


namespace modulus_of_one_minus_i_l3032_303287

theorem modulus_of_one_minus_i :
  let z : ℂ := 1 - I
  Complex.abs z = Real.sqrt 2 := by sorry

end modulus_of_one_minus_i_l3032_303287


namespace complex_product_imaginary_l3032_303264

theorem complex_product_imaginary (a : ℝ) : 
  (Complex.I * (1 + a * Complex.I) + (2 : ℂ) * (1 + a * Complex.I)).re = 0 → a = 2 :=
by
  sorry

end complex_product_imaginary_l3032_303264


namespace book_cost_problem_l3032_303286

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ)
  (h_total : total_cost = 540)
  (h_loss : loss_percent = 15)
  (h_gain : gain_percent = 19)
  (h_equal_sell : (1 - loss_percent / 100) * cost_loss = (1 + gain_percent / 100) * (total_cost - cost_loss)) :
  ∃ (cost_loss : ℝ), cost_loss = 315 := by
  sorry

end book_cost_problem_l3032_303286


namespace gummy_vitamins_cost_l3032_303268

/-- Calculates the total cost of gummy vitamin bottles after discounts and coupons -/
def calculate_total_cost (regular_price : ℚ) (individual_discount : ℚ) (coupon_value : ℚ) (num_bottles : ℕ) (bulk_discount : ℚ) : ℚ :=
  let discounted_price := regular_price * (1 - individual_discount)
  let price_after_coupon := discounted_price - coupon_value
  let total_before_bulk := price_after_coupon * num_bottles
  let bulk_discount_amount := total_before_bulk * bulk_discount
  total_before_bulk - bulk_discount_amount

/-- Theorem stating that the total cost for 3 bottles of gummy vitamins is $29.78 -/
theorem gummy_vitamins_cost :
  calculate_total_cost 15 (17/100) 2 3 (5/100) = 2978/100 :=
by sorry

end gummy_vitamins_cost_l3032_303268


namespace granola_bars_per_box_l3032_303237

theorem granola_bars_per_box 
  (num_kids : ℕ) 
  (bars_per_kid : ℕ) 
  (num_boxes : ℕ) 
  (h1 : num_kids = 30) 
  (h2 : bars_per_kid = 2) 
  (h3 : num_boxes = 5) :
  (num_kids * bars_per_kid) / num_boxes = 12 := by
sorry

end granola_bars_per_box_l3032_303237


namespace box_surface_area_is_288_l3032_303247

/-- Calculates the surface area of the interior of an open box formed by removing square corners from a rectangular sheet and folding the sides. -/
def interior_surface_area (sheet_length sheet_width corner_side : ℕ) : ℕ :=
  let new_length := sheet_length - 2 * corner_side
  let new_width := sheet_width - 2 * corner_side
  new_length * new_width

/-- Theorem: The surface area of the interior of the open box is 288 square units. -/
theorem box_surface_area_is_288 :
  interior_surface_area 36 24 6 = 288 :=
by sorry

end box_surface_area_is_288_l3032_303247


namespace swimming_pool_kids_jose_swimming_pool_l3032_303272

theorem swimming_pool_kids (kids_charge : ℕ) (adults_charge : ℕ) 
  (adults_per_day : ℕ) (weekly_earnings : ℕ) : ℕ :=
  let kids_per_day := 
    (weekly_earnings / 7 - adults_per_day * adults_charge) / kids_charge
  kids_per_day

theorem jose_swimming_pool : swimming_pool_kids 3 6 10 588 = 8 := by
  sorry

end swimming_pool_kids_jose_swimming_pool_l3032_303272


namespace f_monotonicity_and_m_range_l3032_303225

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + 1

theorem f_monotonicity_and_m_range :
  ∀ (a : ℝ),
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ ∧ a ≤ Real.sqrt 2 → f a x₁ < f a x₂) ∧
  (a > Real.sqrt 2 → 
    ∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ x : ℝ, x₁ < x ∧ x < x₂ → f a x > f a x₁ ∧ f a x > f a x₃) ∧
    (∀ x : ℝ, 0 < x ∧ x < x₁ → f a x < f a x₁) ∧
    (∀ x : ℝ, x > x₃ → f a x > f a x₃)) ∧
  (∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ 1 ∧
    (∀ m : ℝ, (∀ a : ℝ, -2 < a ∧ a ≤ 0 → 
      2*m*Real.exp a*(a+1) + f a x₀ > a^2 + 2*a + 4) ↔ 1 < m ∧ m ≤ Real.exp 2)) :=
by sorry

end f_monotonicity_and_m_range_l3032_303225


namespace chromium_percentage_calculation_l3032_303248

/-- The percentage of chromium in the first alloy -/
def chromium_percentage_first : ℝ := 12

/-- The percentage of chromium in the second alloy -/
def chromium_percentage_second : ℝ := 8

/-- The mass of the first alloy in kg -/
def mass_first : ℝ := 15

/-- The mass of the second alloy in kg -/
def mass_second : ℝ := 35

/-- The percentage of chromium in the resulting alloy -/
def chromium_percentage_result : ℝ := 9.2

theorem chromium_percentage_calculation :
  (chromium_percentage_first / 100) * mass_first + 
  (chromium_percentage_second / 100) * mass_second = 
  (chromium_percentage_result / 100) * (mass_first + mass_second) :=
by sorry

#check chromium_percentage_calculation

end chromium_percentage_calculation_l3032_303248


namespace coefficient_x3_is_negative_540_l3032_303274

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (3x^2 - 1/x)^6
def coefficient_x3 : ℤ :=
  -3^3 * binomial 6 3

-- Theorem statement
theorem coefficient_x3_is_negative_540 : coefficient_x3 = -540 := by sorry

end coefficient_x3_is_negative_540_l3032_303274


namespace sophie_total_spent_l3032_303210

def cupcake_quantity : ℕ := 5
def cupcake_price : ℚ := 2

def doughnut_quantity : ℕ := 6
def doughnut_price : ℚ := 1

def apple_pie_slice_quantity : ℕ := 4
def apple_pie_slice_price : ℚ := 2

def cookie_quantity : ℕ := 15
def cookie_price : ℚ := 0.60

def total_spent : ℚ := cupcake_quantity * cupcake_price + 
                        doughnut_quantity * doughnut_price + 
                        apple_pie_slice_quantity * apple_pie_slice_price + 
                        cookie_quantity * cookie_price

theorem sophie_total_spent : total_spent = 33 := by
  sorry

end sophie_total_spent_l3032_303210


namespace ratio_problem_l3032_303201

theorem ratio_problem (a b x m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 4 / 5) 
  (h4 : x = a + 0.75 * a) (h5 : m = b - 0.8 * b) : m / x = 1 / 7 := by
  sorry

end ratio_problem_l3032_303201


namespace problem_solution_l3032_303222

theorem problem_solution (a : ℝ) : 
  (∀ b c : ℝ, a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) →
  a * 15 * 11 = 1 →
  a = 6 := by sorry

end problem_solution_l3032_303222


namespace tims_change_l3032_303253

/-- The amount of change Tim will get after buying a candy bar -/
def change (initial_amount : ℕ) (price : ℕ) : ℕ :=
  initial_amount - price

/-- Theorem: Tim's change is 5 cents -/
theorem tims_change : change 50 45 = 5 := by
  sorry

end tims_change_l3032_303253


namespace problem_statement_l3032_303216

theorem problem_statement (x y : ℝ) 
  (eq1 : x + x*y + y = 2 + 3*Real.sqrt 2) 
  (eq2 : x^2 + y^2 = 6) : 
  |x + y + 1| = 3 + Real.sqrt 2 := by
  sorry

end problem_statement_l3032_303216


namespace three_digit_four_digit_count_l3032_303267

theorem three_digit_four_digit_count : 
  (Finset.filter (fun x : ℕ => 
    100 ≤ 3 * x ∧ 3 * x ≤ 999 ∧ 
    1000 ≤ 4 * x ∧ 4 * x ≤ 9999) (Finset.range 10000)).card = 84 := by
  sorry

end three_digit_four_digit_count_l3032_303267


namespace smallest_cube_root_with_small_fraction_l3032_303270

theorem smallest_cube_root_with_small_fraction (m n : ℕ) (r : ℝ) : 
  0 < n → 0 < r → r < 1 / 500 → 
  (↑m : ℝ) ^ (1/3 : ℝ) = n + r → 
  (∀ k < m, ¬∃ (s : ℝ), 0 < s ∧ s < 1/500 ∧ (↑k : ℝ) ^ (1/3 : ℝ) = ↑(n-1) + s) →
  n = 13 := by
sorry

end smallest_cube_root_with_small_fraction_l3032_303270


namespace inequality_proof_l3032_303235

theorem inequality_proof (a b m n p : ℝ) 
  (h1 : a > b) (h2 : m > n) (h3 : p > 0) : 
  n - a * p < m - b * p := by
  sorry

end inequality_proof_l3032_303235


namespace base_10_to_base_7_l3032_303288

theorem base_10_to_base_7 : 
  ∃ (a b c d : ℕ), 
    784 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a = 2 ∧ b = 2 ∧ c = 0 ∧ d = 0 := by
  sorry

end base_10_to_base_7_l3032_303288


namespace red_balls_count_l3032_303293

theorem red_balls_count (total_balls : ℕ) (prob_red : ℚ) (red_balls : ℕ) : 
  total_balls = 1000 →
  prob_red = 1/5 →
  red_balls = (total_balls : ℚ) * prob_red →
  red_balls = 200 := by
  sorry

end red_balls_count_l3032_303293


namespace cube_sum_power_of_two_l3032_303219

theorem cube_sum_power_of_two (k : ℕ+) :
  (∃ (a b c : ℕ+), |((a:ℤ) - b)^3 + ((b:ℤ) - c)^3 + ((c:ℤ) - a)^3| = 3 * 2^(k:ℕ)) ↔
  (∃ (n : ℕ), k = 3 * n + 1) :=
sorry

end cube_sum_power_of_two_l3032_303219


namespace dividend_calculation_l3032_303294

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 18) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 167 := by
  sorry

end dividend_calculation_l3032_303294


namespace grass_cutting_expenditure_l3032_303205

/-- Represents the four seasons --/
inductive Season
  | Spring
  | Summer
  | Fall
  | Winter

/-- Growth rate of grass per month for each season (in inches) --/
def growth_rate (s : Season) : Real :=
  match s with
  | Season.Spring => 0.6
  | Season.Summer => 0.5
  | Season.Fall => 0.4
  | Season.Winter => 0.2

/-- Number of months in each season --/
def months_per_season : Nat := 3

/-- Initial height of grass after cutting (in inches) --/
def initial_height : Real := 2

/-- Height at which grass needs to be cut (in inches) --/
def cut_height : Real := 4

/-- Initial cost to cut grass --/
def initial_cost : Nat := 100

/-- Cost increase per cut --/
def cost_increase : Nat := 5

/-- Calculate the total growth of grass in a season --/
def season_growth (s : Season) : Real :=
  growth_rate s * months_per_season

/-- Calculate the number of cuts needed in a year --/
def cuts_per_year : Nat := 2

/-- Calculate the total expenditure for cutting grass in a year --/
def total_expenditure : Nat :=
  initial_cost + (initial_cost + cost_increase)

theorem grass_cutting_expenditure :
  total_expenditure = 205 := by
  sorry

end grass_cutting_expenditure_l3032_303205


namespace hot_dog_discount_calculation_l3032_303250

theorem hot_dog_discount_calculation (num_hot_dogs : ℕ) (price_per_hot_dog : ℕ) (discount_rate : ℚ) :
  num_hot_dogs = 6 →
  price_per_hot_dog = 50 →
  discount_rate = 1/10 →
  (num_hot_dogs * price_per_hot_dog) * (1 - discount_rate) = 270 :=
by sorry

end hot_dog_discount_calculation_l3032_303250


namespace binary_representation_of_2_pow_n_minus_1_binary_to_decimal_ten_ones_l3032_303228

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Represents a binary number with n ones -/
def all_ones (n : ℕ) : List Bool :=
  List.replicate n true

theorem binary_representation_of_2_pow_n_minus_1 (n : ℕ) :
  binary_to_decimal (all_ones n) = 2^n - 1 := by
  sorry

/-- The main theorem proving that (1111111111)₂ in decimal form is 2^10 - 1 -/
theorem binary_to_decimal_ten_ones :
  binary_to_decimal (all_ones 10) = 2^10 - 1 := by
  sorry

end binary_representation_of_2_pow_n_minus_1_binary_to_decimal_ten_ones_l3032_303228


namespace smallest_coin_count_l3032_303285

def count_factors (n : ℕ) : ℕ := (Nat.divisors n).card

def count_proper_factors (n : ℕ) : ℕ := (count_factors n) - 2

theorem smallest_coin_count :
  ∀ m : ℕ, m > 0 →
    (count_factors m = 19 ∧ count_proper_factors m = 17) →
    m ≥ 786432 :=
by sorry

end smallest_coin_count_l3032_303285


namespace intersection_of_A_and_B_l3032_303204

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l3032_303204


namespace bike_shop_profit_l3032_303234

/-- The cost of parts for fixing a single bike tire -/
def tire_part_cost : ℝ := 5

theorem bike_shop_profit (tire_repair_price : ℝ) (tire_repairs : ℕ) 
  (complex_repair_price : ℝ) (complex_repair_cost : ℝ) (complex_repairs : ℕ)
  (retail_profit : ℝ) (fixed_expenses : ℝ) (total_profit : ℝ) :
  tire_repair_price = 20 →
  tire_repairs = 300 →
  complex_repair_price = 300 →
  complex_repair_cost = 50 →
  complex_repairs = 2 →
  retail_profit = 2000 →
  fixed_expenses = 4000 →
  total_profit = 3000 →
  tire_part_cost = 5 := by
sorry

end bike_shop_profit_l3032_303234


namespace trapezium_longer_side_length_l3032_303249

/-- Given a trapezium with the following properties:
    - One parallel side is 10 cm long
    - The distance between parallel sides is 15 cm
    - The area is 210 square centimeters
    This theorem proves that the length of the other parallel side is 18 cm. -/
theorem trapezium_longer_side_length (a b h : ℝ) : 
  a = 10 → h = 15 → (a + b) * h / 2 = 210 → b = 18 :=
by sorry

end trapezium_longer_side_length_l3032_303249
