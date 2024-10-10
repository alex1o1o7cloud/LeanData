import Mathlib

namespace partner_c_profit_share_l1821_182128

/-- Given the investment ratios of partners A, B, and C, and a total profit,
    calculate C's share of the profit. -/
theorem partner_c_profit_share 
  (a b c : ℝ) -- Investments of partners A, B, and C
  (total_profit : ℝ) -- Total profit
  (ha : a = 3 * b) -- A invests 3 times as much as B
  (hc : a = 2 / 3 * c) -- A invests 2/3 of what C invests
  : c / (a + b + c) * total_profit = 9 / 17 * total_profit :=
by sorry

end partner_c_profit_share_l1821_182128


namespace geometric_sequence_solution_l1821_182151

theorem geometric_sequence_solution (a : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ (2*a + 2) = a * r ∧ (3*a + 3) = (2*a + 2) * r) → a = -4 :=
by sorry

end geometric_sequence_solution_l1821_182151


namespace inverse_proportion_k_value_l1821_182111

/-- Given an inverse proportion function y = k/x passing through the point (2, -1), 
    prove that k = -2 -/
theorem inverse_proportion_k_value : 
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 0 → -1 = k / 2) → k = -2 := by
  sorry

end inverse_proportion_k_value_l1821_182111


namespace therapy_pricing_theorem_l1821_182155

/-- Represents the pricing structure and total charges for a psychologist's therapy sessions. -/
structure TherapyPricing where
  first_hour : ℕ  -- Price for the first hour
  additional_hour : ℕ  -- Price for each additional hour
  total_5_hours : ℕ  -- Total charge for 5 hours of therapy

/-- Given the pricing structure, calculates the total charge for 2 hours of therapy. -/
def charge_for_2_hours (pricing : TherapyPricing) : ℕ :=
  pricing.first_hour + pricing.additional_hour

/-- Theorem stating the relationship between the pricing structure and the charge for 2 hours. -/
theorem therapy_pricing_theorem (pricing : TherapyPricing) 
  (h1 : pricing.first_hour = pricing.additional_hour + 40)
  (h2 : pricing.total_5_hours = 375)
  (h3 : pricing.first_hour + 4 * pricing.additional_hour = pricing.total_5_hours) :
  charge_for_2_hours pricing = 174 := by
  sorry

#eval charge_for_2_hours { first_hour := 107, additional_hour := 67, total_5_hours := 375 }

end therapy_pricing_theorem_l1821_182155


namespace hundred_with_fewer_threes_l1821_182197

-- Define a datatype for arithmetic expressions
inductive Expr
  | Num : ℕ → Expr
  | Add : Expr → Expr → Expr
  | Sub : Expr → Expr → Expr
  | Mul : Expr → Expr → Expr
  | Div : Expr → Expr → Expr

-- Function to count the number of 3's in an expression
def countThrees : Expr → ℕ
  | Expr.Num n => if n = 3 then 1 else 0
  | Expr.Add e1 e2 => countThrees e1 + countThrees e2
  | Expr.Sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.Mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.Div e1 e2 => countThrees e1 + countThrees e2

-- Function to evaluate an expression
def evaluate : Expr → ℚ
  | Expr.Num n => n
  | Expr.Add e1 e2 => evaluate e1 + evaluate e2
  | Expr.Sub e1 e2 => evaluate e1 - evaluate e2
  | Expr.Mul e1 e2 => evaluate e1 * evaluate e2
  | Expr.Div e1 e2 => evaluate e1 / evaluate e2

-- Theorem statement
theorem hundred_with_fewer_threes : 
  ∃ e : Expr, evaluate e = 100 ∧ countThrees e < 10 :=
sorry

end hundred_with_fewer_threes_l1821_182197


namespace parity_of_S_l1821_182110

theorem parity_of_S (a b c n : ℤ) 
  (h1 : (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) ∨ 
        (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ 
        (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1)) :
  let S := (a + 2*n + 1) * (b + 2*n + 2) * (c + 2*n + 3)
  S % 2 = 0 := by
sorry

end parity_of_S_l1821_182110


namespace sin_cos_transformation_l1821_182192

/-- The transformation between sin and cos functions -/
theorem sin_cos_transformation (f g : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = Real.sin (2 * x - π / 4)) →
  (∀ x, g x = Real.cos (2 * x)) →
  (∀ θ, Real.sin θ = Real.cos (θ - π / 2)) →
  f x = g (x + 3 * π / 8) := by
  sorry

end sin_cos_transformation_l1821_182192


namespace family_money_difference_l1821_182120

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- Calculate the total value of coins for a person -/
def total_value (quarters dimes nickels : ℕ) : ℚ :=
  quarters * quarter_value + dimes * dime_value + nickels * nickel_value

/-- Karen's total value -/
def karen_value : ℚ := total_value 32 0 0

/-- Christopher's total value -/
def christopher_value : ℚ := total_value 64 0 0

/-- Emily's total value -/
def emily_value : ℚ := total_value 20 15 0

/-- Michael's total value -/
def michael_value : ℚ := total_value 12 10 25

/-- Sophia's total value -/
def sophia_value : ℚ := total_value 0 50 40

/-- Alex's total value -/
def alex_value : ℚ := total_value 0 25 100

/-- Total value for Karen and Christopher's family -/
def family1_value : ℚ := karen_value + christopher_value + emily_value + michael_value

/-- Total value for Sophia and Alex's family -/
def family2_value : ℚ := sophia_value + alex_value

theorem family_money_difference :
  family1_value - family2_value = 85/4 := by sorry

end family_money_difference_l1821_182120


namespace min_value_x_plus_3y_l1821_182130

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 1) + 1 / (y + 1) = 1 / 4) :
  x + 3 * y ≥ 5 + 4 * Real.sqrt 3 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧
    1 / (x₀ + 1) + 1 / (y₀ + 1) = 1 / 4 ∧
    x₀ + 3 * y₀ = 5 + 4 * Real.sqrt 3 := by
  sorry

end min_value_x_plus_3y_l1821_182130


namespace parallel_lines_coefficient_product_l1821_182145

/-- Two parallel lines with a specific distance between them -/
structure ParallelLines where
  a : ℝ
  b : ℝ
  l₁ : (x y : ℝ) → a * x + 2 * y + b = 0
  l₂ : (x y : ℝ) → (a - 1) * x + y + b = 0
  parallel : ∀ (x y : ℝ), a * x + 2 * y = (a - 1) * x + y
  distance : ∃ (k : ℝ), k * (b - 0) / Real.sqrt ((a - (a - 1))^2 + (2 - 1)^2) = Real.sqrt 2 / 2 ∧ k = 1 ∨ k = -1

/-- The product of coefficients a and b for parallel lines with specific distance -/
theorem parallel_lines_coefficient_product (pl : ParallelLines) : pl.a * pl.b = 4 ∨ pl.a * pl.b = -4 := by
  sorry

end parallel_lines_coefficient_product_l1821_182145


namespace married_men_fraction_l1821_182181

-- Define the faculty
structure Faculty where
  total : ℕ
  women : ℕ
  married : ℕ
  men : ℕ

-- Define the conditions
def faculty_conditions (f : Faculty) : Prop :=
  f.women = (70 * f.total) / 100 ∧
  f.married = (40 * f.total) / 100 ∧
  f.men = f.total - f.women

-- Define the fraction of single men
def single_men_fraction (f : Faculty) : ℚ :=
  1 / 3

-- Theorem to prove
theorem married_men_fraction (f : Faculty) 
  (h : faculty_conditions f) : 
  (f.married - (f.women - (f.total - f.married))) / f.men = 2 / 3 :=
sorry

end married_men_fraction_l1821_182181


namespace sphere_wall_thickness_l1821_182135

/-- Represents a hollow glass sphere floating in water -/
structure FloatingSphere where
  outer_diameter : ℝ
  specific_gravity : ℝ
  dry_surface_fraction : ℝ

/-- Calculates the wall thickness of a floating sphere -/
noncomputable def wall_thickness (sphere : FloatingSphere) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the wall thickness of the sphere with given properties -/
theorem sphere_wall_thickness :
  let sphere : FloatingSphere := {
    outer_diameter := 16,
    specific_gravity := 2.523,
    dry_surface_fraction := 3/8
  }
  wall_thickness sphere = 0.8 := by sorry

end sphere_wall_thickness_l1821_182135


namespace sons_age_l1821_182103

/-- Proves that the son's age is 7.5 years given the conditions of the problem -/
theorem sons_age (son_age man_age : ℝ) : 
  man_age = son_age + 25 →
  man_age + 5 = 3 * (son_age + 5) →
  son_age = 7.5 := by
sorry

end sons_age_l1821_182103


namespace sinks_per_house_l1821_182175

/-- Given that a carpenter bought 266 sinks to cover 44 houses,
    prove that the number of sinks needed for each house is 6. -/
theorem sinks_per_house (total_sinks : ℕ) (num_houses : ℕ) 
  (h1 : total_sinks = 266) (h2 : num_houses = 44) :
  total_sinks / num_houses = 6 := by
  sorry

#check sinks_per_house

end sinks_per_house_l1821_182175


namespace recycling_project_points_l1821_182137

/-- Calculates points earned for white paper -/
def whitePoints (pounds : ℕ) : ℕ := (pounds / 6) * 2

/-- Calculates points earned for colored paper -/
def colorPoints (pounds : ℕ) : ℕ := (pounds / 8) * 3

/-- Represents a person's recycling contribution -/
structure Recycler where
  whitePaper : ℕ
  coloredPaper : ℕ

/-- Calculates total points for a recycler -/
def totalPoints (r : Recycler) : ℕ :=
  whitePoints r.whitePaper + colorPoints r.coloredPaper

theorem recycling_project_points : 
  let paige : Recycler := { whitePaper := 12, coloredPaper := 18 }
  let alex : Recycler := { whitePaper := 26, coloredPaper := 10 }
  let jordan : Recycler := { whitePaper := 30, coloredPaper := 0 }
  totalPoints paige + totalPoints alex + totalPoints jordan = 31 := by
  sorry

end recycling_project_points_l1821_182137


namespace geometric_series_sum_l1821_182102

theorem geometric_series_sum : 
  let a : ℚ := 2
  let r : ℚ := -2/5
  let series : ℕ → ℚ := λ n => a * r^n
  ∑' n, series n = 10/7 := by
sorry

end geometric_series_sum_l1821_182102


namespace race_head_start_l1821_182114

/-- Given two runners A and B, where A's speed is 20/15 times B's speed,
    the head start A should give B for a dead heat is 1/4 of the race length. -/
theorem race_head_start (speed_a speed_b race_length head_start : ℝ) :
  speed_a = (20 / 15) * speed_b →
  race_length > 0 →
  speed_a > 0 →
  speed_b > 0 →
  (race_length / speed_a = (race_length - head_start) / speed_b ↔ head_start = (1 / 4) * race_length) :=
by sorry

end race_head_start_l1821_182114


namespace monotonically_decreasing_x_ln_x_l1821_182186

/-- The function f(x) = x ln x is monotonically decreasing on the interval (0, 1/e) -/
theorem monotonically_decreasing_x_ln_x :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 / Real.exp 1 →
  x₁ * Real.log x₁ > x₂ * Real.log x₂ := by
sorry

/-- The domain of f(x) = x ln x is (0, +∞) -/
def domain_x_ln_x : Set ℝ := {x : ℝ | x > 0}

end monotonically_decreasing_x_ln_x_l1821_182186


namespace second_half_wants_fifteen_l1821_182150

/-- Represents the BBQ scenario with given conditions -/
structure BBQScenario where
  cooking_time_per_side : ℕ  -- Time to cook one side of a burger
  grill_capacity : ℕ         -- Number of burgers that can fit on the grill
  total_guests : ℕ           -- Total number of guests
  first_half_burgers : ℕ     -- Number of burgers each guest in the first half wants
  total_cooking_time : ℕ     -- Total time taken to cook all burgers

/-- Calculates the number of burgers wanted by the second half of guests -/
def second_half_burgers (scenario : BBQScenario) : ℕ :=
  let total_burgers := scenario.total_cooking_time / (2 * scenario.cooking_time_per_side) * scenario.grill_capacity
  let first_half_total := scenario.total_guests / 2 * scenario.first_half_burgers
  total_burgers - first_half_total

/-- Theorem stating that the second half of guests want 15 burgers -/
theorem second_half_wants_fifteen (scenario : BBQScenario) 
  (h1 : scenario.cooking_time_per_side = 4)
  (h2 : scenario.grill_capacity = 5)
  (h3 : scenario.total_guests = 30)
  (h4 : scenario.first_half_burgers = 2)
  (h5 : scenario.total_cooking_time = 72) : 
  second_half_burgers scenario = 15 := by
  sorry


end second_half_wants_fifteen_l1821_182150


namespace inequality_proof_l1821_182187

theorem inequality_proof (a b c : ℝ) 
  (h : a + b + c + a*b + b*c + a*c + a*b*c ≥ 7) :
  Real.sqrt (a^2 + b^2 + 2) + Real.sqrt (b^2 + c^2 + 2) + Real.sqrt (c^2 + a^2 + 2) ≥ 6 := by
sorry


end inequality_proof_l1821_182187


namespace ratio_of_divisor_sums_l1821_182122

def M : ℕ := 45 * 45 * 98 * 340

def sum_of_even_divisors (n : ℕ) : ℕ := (List.filter (λ x => x % 2 = 0) (List.range (n + 1))).sum

def sum_of_odd_divisors (n : ℕ) : ℕ := (List.filter (λ x => x % 2 ≠ 0) (List.range (n + 1))).sum

theorem ratio_of_divisor_sums :
  (sum_of_even_divisors M) / (sum_of_odd_divisors M) = 14 := by sorry

end ratio_of_divisor_sums_l1821_182122


namespace line_through_M_and_P_line_through_M_perp_to_line_l1821_182182

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3*x + 4*y - 11 = 0
def l₂ (x y : ℝ) : Prop := 2*x + 3*y - 8 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, 2)

-- Define point P
def P : ℝ × ℝ := (3, 1)

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 3*x + 2*y + 5 = 0

-- Part 1: Line equation through M and P
theorem line_through_M_and_P :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0 ↔ (l₁ x y ∧ l₂ x y) ∨ (x = P.1 ∧ y = P.2)) →
    a = 1 ∧ b = 2 ∧ c = -5 :=
sorry

-- Part 2: Line equation through M and perpendicular to perp_line
theorem line_through_M_perp_to_line :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0 ↔ (l₁ x y ∧ l₂ x y) ∨ 
      (∃ (k : ℝ), a*3 + b*2 = 0 ∧ x = M.1 + k*2 ∧ y = M.2 - k*3)) →
    a = 2 ∧ b = -3 ∧ c = 4 :=
sorry

end line_through_M_and_P_line_through_M_perp_to_line_l1821_182182


namespace sum_of_roots_quadratic_sum_of_solutions_specific_equation_l1821_182193

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_solutions_specific_equation :
  let a : ℝ := -48
  let b : ℝ := 66
  let c : ℝ := 195
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = 11/8 :=
by sorry

end sum_of_roots_quadratic_sum_of_solutions_specific_equation_l1821_182193


namespace quadratic_roots_range_l1821_182179

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + a*x + a^2 - 1 = 0 ∧ y^2 + a*y + a^2 - 1 = 0) → 
  -1 < a ∧ a < 1 := by
  sorry

end quadratic_roots_range_l1821_182179


namespace distance_covered_l1821_182101

/-- Proves that the total distance covered is 16 km given the specified conditions -/
theorem distance_covered (walking_speed running_speed : ℝ) (total_time : ℝ) :
  walking_speed = 4 →
  running_speed = 8 →
  total_time = 3 →
  ∃ (distance : ℝ),
    distance / walking_speed / 2 + distance / running_speed / 2 = total_time ∧
    distance = 16 := by
  sorry

end distance_covered_l1821_182101


namespace ring_toss_earnings_l1821_182170

/-- The ring toss game earnings problem -/
theorem ring_toss_earnings 
  (daily_earnings : ℕ) 
  (num_days : ℕ) 
  (h1 : daily_earnings = 33) 
  (h2 : num_days = 5) : 
  daily_earnings * num_days = 165 := by
  sorry

end ring_toss_earnings_l1821_182170


namespace infinite_solutions_l1821_182116

theorem infinite_solutions (a : ℝ) : 
  (a = 5) → (∀ y : ℝ, 3 * (5 + a * y) = 15 * y + 9) :=
by sorry

end infinite_solutions_l1821_182116


namespace puppies_per_dog_l1821_182127

/-- Given information about Chuck's dog breeding operation -/
structure DogBreeding where
  num_pregnant_dogs : ℕ
  shots_per_puppy : ℕ
  cost_per_shot : ℕ
  total_shot_cost : ℕ

/-- Theorem stating the number of puppies per pregnant dog -/
theorem puppies_per_dog (d : DogBreeding)
  (h1 : d.num_pregnant_dogs = 3)
  (h2 : d.shots_per_puppy = 2)
  (h3 : d.cost_per_shot = 5)
  (h4 : d.total_shot_cost = 120) :
  d.total_shot_cost / (d.num_pregnant_dogs * d.shots_per_puppy * d.cost_per_shot) = 4 := by
  sorry

end puppies_per_dog_l1821_182127


namespace thirty_factorial_trailing_zeros_l1821_182100

def trailing_zeros (n : ℕ) : ℕ := 
  (n / 5) + (n / 25)

theorem thirty_factorial_trailing_zeros : 
  trailing_zeros 30 = 7 := by
  sorry

end thirty_factorial_trailing_zeros_l1821_182100


namespace even_function_characterization_l1821_182172

def M (f : ℝ → ℝ) (a : ℝ) : Set ℝ :=
  {t | ∃ x ≥ a, t = f x - f a}

def L (f : ℝ → ℝ) (a : ℝ) : Set ℝ :=
  {t | ∃ x ≤ a, t = f x - f a}

def has_minimum (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, f m ≤ f x

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem even_function_characterization (f : ℝ → ℝ) (h : has_minimum f) :
  is_even_function f ↔ ∀ c > 0, M f (-c) = L f c := by
  sorry

end even_function_characterization_l1821_182172


namespace picnic_cost_is_60_l1821_182129

/-- Calculates the total cost of a picnic basket given the number of people and item prices. -/
def picnic_cost (num_people : ℕ) (sandwich_price fruit_salad_price soda_price snack_price : ℕ) : ℕ :=
  let sandwich_cost := num_people * sandwich_price
  let fruit_salad_cost := num_people * fruit_salad_price
  let soda_cost := num_people * 2 * soda_price
  let snack_cost := 3 * snack_price
  sandwich_cost + fruit_salad_cost + soda_cost + snack_cost

/-- Theorem stating that the total cost of the picnic basket is $60. -/
theorem picnic_cost_is_60 :
  picnic_cost 4 5 3 2 4 = 60 := by
  sorry

#eval picnic_cost 4 5 3 2 4

end picnic_cost_is_60_l1821_182129


namespace ratio_change_problem_l1821_182161

theorem ratio_change_problem (x y z : ℝ) : 
  y / x = 3 / 2 →  -- Initial ratio
  y - x = 8 →  -- Difference between numbers
  (y + z) / (x + z) = 7 / 5 →  -- New ratio after adding z
  z = 4 :=  -- The number added to both
by sorry

end ratio_change_problem_l1821_182161


namespace abs_inequality_solution_set_l1821_182124

theorem abs_inequality_solution_set (x : ℝ) : 
  |x - 3| < 1 ↔ 2 < x ∧ x < 4 :=
sorry

end abs_inequality_solution_set_l1821_182124


namespace quiz_answer_key_l1821_182149

theorem quiz_answer_key (n : ℕ) : 
  (2^5 - 2) * 4^n = 480 → n = 2 := by
  sorry

end quiz_answer_key_l1821_182149


namespace cos_beta_minus_gamma_bounds_l1821_182106

theorem cos_beta_minus_gamma_bounds (k : ℝ) (α β γ : ℝ) 
  (h1 : 0 < k) (h2 : k < 2)
  (eq1 : Real.cos α + k * Real.cos β + (2 - k) * Real.cos γ = 0)
  (eq2 : Real.sin α + k * Real.sin β + (2 - k) * Real.sin γ = 0) :
  (∀ x, Real.cos (β - γ) ≤ x → x ≤ -1/2) ∧ 
  (∃ k₁ k₂, 0 < k₁ ∧ k₁ < 2 ∧ 0 < k₂ ∧ k₂ < 2 ∧ 
    Real.cos (β - γ) = -1/2 ∧ Real.cos (β - γ) = -1) :=
by sorry

end cos_beta_minus_gamma_bounds_l1821_182106


namespace middle_numbers_average_l1821_182199

theorem middle_numbers_average (a b c d : ℕ+) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different positive integers
  (a + b + c + d : ℚ) / 4 = 5 ∧  -- Average is 5
  ∀ w x y z : ℕ+, w < x ∧ x < y ∧ y < z ∧ (w + x + y + z : ℚ) / 4 = 5 → (z - w : ℤ) ≤ (d - a : ℤ) →  -- Maximum possible difference
  (b + c : ℚ) / 2 = 5/2 :=
sorry

end middle_numbers_average_l1821_182199


namespace random_walk_properties_l1821_182104

/-- A random walk on a line with forward probability 3/4 and backward probability 1/4 -/
structure RandomWalk where
  forwardProb : ℝ
  backwardProb : ℝ
  forwardProbEq : forwardProb = 3/4
  backwardProbEq : backwardProb = 1/4
  probSum : forwardProb + backwardProb = 1

/-- The probability of returning to the starting point after n steps -/
def returnProbability (rw : RandomWalk) (n : ℕ) : ℝ :=
  sorry

/-- The probability distribution of the distance from the starting point after n steps -/
def distanceProbability (rw : RandomWalk) (n : ℕ) (d : ℕ) : ℝ :=
  sorry

/-- The expected value of the distance from the starting point after n steps -/
def expectedDistance (rw : RandomWalk) (n : ℕ) : ℝ :=
  sorry

theorem random_walk_properties (rw : RandomWalk) :
  returnProbability rw 4 = 27/128 ∧
  distanceProbability rw 5 1 = 45/128 ∧
  distanceProbability rw 5 3 = 105/256 ∧
  distanceProbability rw 5 5 = 61/256 ∧
  expectedDistance rw 5 = 355/128 := by
  sorry

end random_walk_properties_l1821_182104


namespace distribute_five_items_to_fifteen_recipients_l1821_182143

/-- The number of ways to distribute distinct items to recipients -/
def distribute_items (num_items : ℕ) (num_recipients : ℕ) : ℕ :=
  num_recipients ^ num_items

/-- Theorem: Distributing 5 distinct items to 15 recipients results in 759,375 possible ways -/
theorem distribute_five_items_to_fifteen_recipients :
  distribute_items 5 15 = 759375 := by
  sorry

end distribute_five_items_to_fifteen_recipients_l1821_182143


namespace food_drive_cans_l1821_182112

theorem food_drive_cans (rachel jaydon mark : ℕ) : 
  jaydon = 2 * rachel + 5 →
  mark = 4 * jaydon →
  rachel + jaydon + mark = 135 →
  mark = 100 := by
sorry

end food_drive_cans_l1821_182112


namespace amelias_apples_l1821_182185

theorem amelias_apples (george_oranges : ℕ) (george_apples_diff : ℕ) (amelia_oranges_diff : ℕ) (total_fruits : ℕ) :
  george_oranges = 45 →
  george_apples_diff = 5 →
  amelia_oranges_diff = 18 →
  total_fruits = 107 →
  ∃ (amelia_apples : ℕ),
    total_fruits = george_oranges + (george_oranges - amelia_oranges_diff) + (amelia_apples + george_apples_diff) + amelia_apples ∧
    amelia_apples = 15 :=
by sorry

end amelias_apples_l1821_182185


namespace roses_cut_l1821_182171

theorem roses_cut (initial_roses final_roses : ℕ) (h1 : initial_roses = 6) (h2 : final_roses = 16) :
  final_roses - initial_roses = 10 := by
  sorry

end roses_cut_l1821_182171


namespace total_dolls_l1821_182189

def sister_dolls : ℕ := 8
def hannah_multiplier : ℕ := 5

theorem total_dolls : sister_dolls + hannah_multiplier * sister_dolls = 48 := by
  sorry

end total_dolls_l1821_182189


namespace aarons_brothers_l1821_182107

theorem aarons_brothers (bennett_brothers : ℕ) (aaron_brothers : ℕ) 
  (h1 : bennett_brothers = 6) 
  (h2 : bennett_brothers = 2 * aaron_brothers - 2) : 
  aaron_brothers = 4 := by
sorry

end aarons_brothers_l1821_182107


namespace units_digit_of_N_l1821_182144

def N : ℕ := 3^1001 + 7^1002 + 13^1003

theorem units_digit_of_N (n : ℕ) (h : n = N) : n % 10 = 9 := by
  sorry

end units_digit_of_N_l1821_182144


namespace max_product_constraint_l1821_182126

theorem max_product_constraint (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 3) :
  m * n ≤ 9 / 4 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ m₀ + n₀ = 3 ∧ m₀ * n₀ = 9 / 4 :=
by sorry

end max_product_constraint_l1821_182126


namespace notebook_notepad_pen_cost_l1821_182148

theorem notebook_notepad_pen_cost (x y z : ℤ) : 
  x + 3*y + 2*z = 98 →
  3*x + y = 5*z - 36 →
  Even x →
  x = 4 ∧ y = 22 ∧ z = 14 := by
sorry

end notebook_notepad_pen_cost_l1821_182148


namespace units_digit_of_m_cubed_plus_two_to_m_l1821_182180

theorem units_digit_of_m_cubed_plus_two_to_m (m : ℕ) : 
  m = 2021^2 + 2^2021 → (m^3 + 2^m) % 10 = 5 := by
sorry

end units_digit_of_m_cubed_plus_two_to_m_l1821_182180


namespace bacteria_after_10_hours_l1821_182138

/-- Represents the number of bacteria in the colony after a given number of hours -/
def bacteria_count (hours : ℕ) : ℕ :=
  2^hours

/-- Theorem stating that after 10 hours, the bacteria count is 1024 -/
theorem bacteria_after_10_hours :
  bacteria_count 10 = 1024 := by
  sorry

end bacteria_after_10_hours_l1821_182138


namespace cubic_factorization_l1821_182188

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end cubic_factorization_l1821_182188


namespace project_hours_theorem_l1821_182183

theorem project_hours_theorem (kate_hours mark_hours pat_hours : ℕ) : 
  pat_hours = 2 * kate_hours →
  pat_hours = mark_hours / 3 →
  mark_hours = kate_hours + 100 →
  kate_hours + pat_hours + mark_hours = 180 := by
sorry

end project_hours_theorem_l1821_182183


namespace negation_of_existence_negation_of_proposition_l1821_182140

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by sorry

end negation_of_existence_negation_of_proposition_l1821_182140


namespace fibSum_eq_three_l1821_182152

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of F_n / 2^n from n = 0 to infinity -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / (2 : ℝ) ^ n

/-- Theorem stating that the sum of F_n / 2^n from n = 0 to infinity equals 3 -/
theorem fibSum_eq_three : fibSum = 3 := by sorry

end fibSum_eq_three_l1821_182152


namespace bella_roses_l1821_182113

def dozen : ℕ := 12

def roses_from_parents : ℕ := 2 * dozen

def number_of_friends : ℕ := 10

def roses_per_friend : ℕ := 2

def total_roses : ℕ := roses_from_parents + number_of_friends * roses_per_friend

theorem bella_roses : total_roses = 44 := by sorry

end bella_roses_l1821_182113


namespace earth_inhabitable_fraction_l1821_182117

theorem earth_inhabitable_fraction :
  let water_fraction : ℚ := 3/5
  let inhabitable_land_fraction : ℚ := 2/3
  let total_inhabitable_fraction : ℚ := (1 - water_fraction) * inhabitable_land_fraction
  total_inhabitable_fraction = 4/15 := by sorry

end earth_inhabitable_fraction_l1821_182117


namespace infinitely_many_divisible_by_prime_l1821_182169

theorem infinitely_many_divisible_by_prime (p : ℕ) (hp : Prime p) :
  ∃ (N : Set ℕ), Set.Infinite N ∧ ∀ n ∈ N, p ∣ (2^n - n) :=
sorry

end infinitely_many_divisible_by_prime_l1821_182169


namespace xyz_value_l1821_182165

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 10 := by
  sorry

end xyz_value_l1821_182165


namespace triangle_line_equation_l1821_182125

/-- A line with slope 3/4 that forms a triangle with the coordinate axes -/
structure TriangleLine where
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The perimeter of the triangle formed by the line and the coordinate axes is 12 -/
  perimeter_eq : |b| + |-(4/3)*b| + Real.sqrt (b^2 + (-(4/3)*b)^2) = 12

/-- The equation of a TriangleLine is either 3x-4y+12=0 or 3x-4y-12=0 -/
theorem triangle_line_equation (l : TriangleLine) :
  (3 : ℝ) * l.b = 12 ∨ (3 : ℝ) * l.b = -12 := by sorry

end triangle_line_equation_l1821_182125


namespace hexagon_game_theorem_l1821_182136

/-- Represents a hexagonal grid cell -/
structure HexCell where
  x : ℤ
  y : ℤ

/-- Represents the state of a cell (empty or filled) -/
inductive CellState
  | Empty
  | Filled

/-- Represents the game state -/
structure GameState where
  grid : HexCell → CellState
  turn : ℕ

/-- Represents a player's move -/
inductive Move
  | PlaceCounters (c1 c2 : HexCell)
  | RemoveCounter (c : HexCell)

/-- Checks if two hexagonal cells are adjacent -/
def are_adjacent (c1 c2 : HexCell) : Prop :=
  sorry

/-- Checks if there are k consecutive filled cells in a line -/
def has_k_consecutive_filled (g : GameState) (k : ℕ) : Prop :=
  sorry

/-- Applies a move to the game state -/
def apply_move (g : GameState) (m : Move) : GameState :=
  sorry

/-- Checks if a move is valid according to the game rules -/
def is_valid_move (g : GameState) (m : Move) : Prop :=
  sorry

/-- Represents a winning strategy for player A -/
def winning_strategy (k : ℕ) : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (g : GameState),
      is_valid_move g (strategy g) ∧
      (∃ (n : ℕ), has_k_consecutive_filled (apply_move g (strategy g)) k)

/-- The main theorem stating that 6 is the minimum value of k for which A cannot win -/
theorem hexagon_game_theorem :
  (∀ (k : ℕ), k < 6 → winning_strategy k) ∧
  ¬(winning_strategy 6) :=
sorry

end hexagon_game_theorem_l1821_182136


namespace pentagon_y_coordinate_of_C_l1821_182168

/-- Pentagon with vertices A, B, C, D, E in 2D space -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculate the area of a triangle given three vertices -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the area of a pentagon -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop := sorry

/-- Main theorem -/
theorem pentagon_y_coordinate_of_C (p : Pentagon) 
  (h1 : p.A = (0, 0))
  (h2 : p.B = (0, 4))
  (h3 : p.D = (4, 4))
  (h4 : p.E = (4, 0))
  (h5 : ∃ y, p.C = (2, y))
  (h6 : hasVerticalSymmetry p)
  (h7 : pentagonArea p = 40) :
  ∃ y, p.C = (2, y) ∧ y = 16 := by sorry

end pentagon_y_coordinate_of_C_l1821_182168


namespace arithmetic_sequence_sum_l1821_182156

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  arithmetic_sequence b →
  a 1 + b 1 = 7 →
  a 3 + b 3 = 21 →
  a 5 + b 5 = 35 :=
by
  sorry

end arithmetic_sequence_sum_l1821_182156


namespace two_sqrt_five_less_than_five_l1821_182195

theorem two_sqrt_five_less_than_five : 2 * Real.sqrt 5 < 5 := by
  sorry

end two_sqrt_five_less_than_five_l1821_182195


namespace greater_than_negative_one_by_two_l1821_182108

theorem greater_than_negative_one_by_two : 
  (fun x => x > -1 ∧ x - (-1) = 2) 1 := by sorry

end greater_than_negative_one_by_two_l1821_182108


namespace expression_value_l1821_182158

theorem expression_value : 
  (2020^4 - 3 * 2020^3 * 2021 + 4 * 2020 * 2021^3 - 2021^4 + 1) / (2020 * 2021) = 4096046 := by
  sorry

end expression_value_l1821_182158


namespace optimal_strategy_l1821_182163

/-- Represents the cosmetics store problem -/
structure CosmeticsStore where
  m : ℝ  -- Purchase price of cosmetic A
  n : ℝ  -- Purchase price of cosmetic B
  total_items : ℕ  -- Total number of items to purchase

/-- Conditions for the cosmetics store problem -/
def valid_store (store : CosmeticsStore) : Prop :=
  3 * store.m + 4 * store.n = 620 ∧
  5 * store.m + 3 * store.n = 740 ∧
  store.total_items = 200

/-- Calculate the profit for a given purchase strategy -/
def profit (store : CosmeticsStore) (items_a : ℕ) : ℝ :=
  (250 - store.m) * items_a + (200 - store.n) * (store.total_items - items_a)

/-- Check if a purchase strategy is valid -/
def valid_strategy (store : CosmeticsStore) (items_a : ℕ) : Prop :=
  store.m * items_a + store.n * (store.total_items - items_a) ≤ 18100 ∧
  profit store items_a ≥ 27000

/-- Theorem stating the optimal strategy and maximum profit -/
theorem optimal_strategy (store : CosmeticsStore) :
  valid_store store →
  (∃ (items_a : ℕ), valid_strategy store items_a) →
  (∃ (max_items_a : ℕ), 
    valid_strategy store max_items_a ∧
    ∀ (items_a : ℕ), valid_strategy store items_a → 
      profit store max_items_a ≥ profit store items_a) ∧
  (let max_items_a := 105
   profit store max_items_a = 27150 ∧
   valid_strategy store max_items_a ∧
   ∀ (items_a : ℕ), valid_strategy store items_a → 
     profit store max_items_a ≥ profit store items_a) :=
by sorry


end optimal_strategy_l1821_182163


namespace total_words_eq_443_l1821_182118

def count_words (n : ℕ) : ℕ :=
  if n ≤ 20 ∨ n = 30 ∨ n = 40 ∨ n = 50 ∨ n = 60 ∨ n = 70 ∨ n = 80 ∨ n = 90 ∨ n = 100 ∨ n = 200 then 1
  else if n ≤ 99 then 2
  else if n ≤ 199 then 3
  else 0

def total_words : ℕ := (List.range 200).map (λ i => count_words (i + 1)) |>.sum

theorem total_words_eq_443 : total_words = 443 := by
  sorry

end total_words_eq_443_l1821_182118


namespace calculate_fraction_l1821_182159

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, 2 * b^2 * f a = a^2 * f b

/-- The main theorem -/
theorem calculate_fraction (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 6 ≠ 0) :
  (f 7 - f 3) / f 6 = 5 / 9 := by
  sorry

end calculate_fraction_l1821_182159


namespace haunted_castle_problem_l1821_182141

/-- Represents a castle with windows -/
structure Castle where
  totalWindows : Nat
  forbiddenExitWindows : Nat

/-- Calculates the number of ways to enter and exit the castle -/
def waysToEnterAndExit (castle : Castle) : Nat :=
  castle.totalWindows * (castle.totalWindows - 1 - castle.forbiddenExitWindows)

/-- The haunted castle problem -/
theorem haunted_castle_problem :
  let castle : Castle := { totalWindows := 8, forbiddenExitWindows := 2 }
  waysToEnterAndExit castle = 40 := by
  sorry

end haunted_castle_problem_l1821_182141


namespace right_triangle_ac_length_l1821_182167

/-- 
Given a right triangle ABC in the x-y plane where:
- ∠B = 90°
- The slope of line segment AC is 4/3
- The length of AB is 20

Prove that the length of AC is 25.
-/
theorem right_triangle_ac_length 
  (A B C : ℝ × ℝ) 
  (right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (slope_ac : (C.2 - A.2) / (C.1 - A.1) = 4 / 3)
  (length_ab : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 20) :
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 25 := by
  sorry

end right_triangle_ac_length_l1821_182167


namespace quadratic_always_positive_l1821_182157

theorem quadratic_always_positive : ∀ x : ℝ, x^2 - x + 1 > 0 := by
  sorry

end quadratic_always_positive_l1821_182157


namespace first_cousin_ate_two_l1821_182131

/-- The number of sandwiches Ruth prepared -/
def total_sandwiches : ℕ := 10

/-- The number of sandwiches Ruth ate -/
def ruth_ate : ℕ := 1

/-- The number of sandwiches Ruth gave to her brother -/
def brother_ate : ℕ := 2

/-- The number of sandwiches eaten by the two other cousins -/
def other_cousins_ate : ℕ := 2

/-- The number of sandwiches left -/
def sandwiches_left : ℕ := 3

/-- The number of sandwiches eaten by the first cousin -/
def first_cousin_ate : ℕ := total_sandwiches - (ruth_ate + brother_ate + other_cousins_ate + sandwiches_left)

theorem first_cousin_ate_two : first_cousin_ate = 2 := by
  sorry

end first_cousin_ate_two_l1821_182131


namespace cricket_team_age_difference_l1821_182146

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (keeper_age_diff : ℕ) (team_avg_age : ℚ) : 
  team_size = 11 →
  captain_age = 27 →
  keeper_age_diff = 3 →
  team_avg_age = 24 →
  let keeper_age := captain_age + keeper_age_diff
  let total_age := team_avg_age * team_size
  let remaining_players := team_size - 2
  let remaining_age := total_age - (captain_age + keeper_age)
  let remaining_avg_age := remaining_age / remaining_players
  remaining_avg_age = team_avg_age - 1 := by
  sorry

end cricket_team_age_difference_l1821_182146


namespace chichikov_guarantee_l1821_182184

/-- Represents a distribution of nuts into three boxes -/
def Distribution := (ℕ × ℕ × ℕ)

/-- Checks if a distribution is valid (sum is 1001) -/
def valid_distribution (d : Distribution) : Prop :=
  d.1 + d.2.1 + d.2.2 = 1001

/-- Represents the number of nuts that need to be moved for a given N -/
def nuts_to_move (d : Distribution) (N : ℕ) : ℕ :=
  sorry

/-- The maximum number of nuts that need to be moved for any N -/
def max_nuts_to_move (d : Distribution) : ℕ :=
  sorry

theorem chichikov_guarantee :
  ∀ d : Distribution, valid_distribution d →
  ∃ N : ℕ, 1 ≤ N ∧ N ≤ 1001 ∧ nuts_to_move d N ≥ 71 ∧
  ∀ M : ℕ, M > 71 → ∃ d' : Distribution, valid_distribution d' ∧
  ∀ N' : ℕ, 1 ≤ N' ∧ N' ≤ 1001 → nuts_to_move d' N' < M :=
sorry

end chichikov_guarantee_l1821_182184


namespace jessica_allowance_l1821_182176

def weekly_allowance : ℝ := 26.67

theorem jessica_allowance (allowance : ℝ) 
  (h1 : 0.45 * allowance + 17 = 29) : 
  allowance = weekly_allowance := by
  sorry

#check jessica_allowance

end jessica_allowance_l1821_182176


namespace min_perimeter_isosceles_triangles_l1821_182196

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  base : ℕ
  leg : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := t.base + 2 * t.leg

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * Real.sqrt ((t.leg : ℝ) ^ 2 - ((t.base : ℝ) / 2) ^ 2) / 2

/-- Theorem: The minimum possible value of the common perimeter of two noncongruent
    integer-sided isosceles triangles with the same perimeter, same area, and base
    lengths in the ratio 8:7 is 586 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    8 * t2.base = 7 * t1.base ∧
    perimeter t1 = 586 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      8 * s2.base = 7 * s1.base →
      perimeter s1 ≥ 586) :=
by
  sorry

end min_perimeter_isosceles_triangles_l1821_182196


namespace opposite_signs_sum_zero_l1821_182142

theorem opposite_signs_sum_zero (a b : ℝ) : a * b < 0 → a + b = 0 := by
  sorry

end opposite_signs_sum_zero_l1821_182142


namespace derivative_of_e_squared_l1821_182178

theorem derivative_of_e_squared :
  (deriv (λ _ : ℝ => Real.exp 2)) = (λ _ => 0) := by
  sorry

end derivative_of_e_squared_l1821_182178


namespace rose_cost_l1821_182162

/-- Proves that the cost of each rose is $5 given the conditions of Nadia's flower purchase. -/
theorem rose_cost (num_roses : ℕ) (num_lilies : ℚ) (total_cost : ℚ) : 
  num_roses = 20 →
  num_lilies = 3/4 * num_roses →
  total_cost = 250 →
  ∃ (rose_cost : ℚ), 
    rose_cost * num_roses + (2 * rose_cost) * num_lilies = total_cost ∧
    rose_cost = 5 := by
  sorry

end rose_cost_l1821_182162


namespace basis_properties_l1821_182174

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_basis (S : Set V) : Prop :=
  Submodule.span ℝ S = ⊤ ∧ LinearIndependent ℝ (fun x => x : S → V)

theorem basis_properties {a b c : V} (h : is_basis {a, b, c}) :
  is_basis {a + b, b + c, c + a} ∧
  ∀ p : V, ∃ x y z : ℝ, p = x • a + y • b + z • c :=
sorry

end basis_properties_l1821_182174


namespace seats_per_bus_correct_l1821_182115

/-- Represents a school with classrooms, students, and buses for a field trip. -/
structure School where
  classrooms : ℕ
  students_per_classroom : ℕ
  seats_per_bus : ℕ

/-- Calculates the total number of students in the school. -/
def total_students (s : School) : ℕ :=
  s.classrooms * s.students_per_classroom

/-- Calculates the number of buses needed for the field trip. -/
def buses_needed (s : School) : ℕ :=
  (total_students s + s.seats_per_bus - 1) / s.seats_per_bus

/-- Theorem stating that for a school with 87 classrooms, 58 students per classroom,
    and buses with 29 seats each, the number of seats on each school bus is 29. -/
theorem seats_per_bus_correct (s : School) 
  (h1 : s.classrooms = 87)
  (h2 : s.students_per_classroom = 58)
  (h3 : s.seats_per_bus = 29) :
  s.seats_per_bus = 29 := by
  sorry

#eval buses_needed { classrooms := 87, students_per_classroom := 58, seats_per_bus := 29 }

end seats_per_bus_correct_l1821_182115


namespace inverse_proposition_l1821_182105

theorem inverse_proposition : 
  (∀ a b : ℝ, a > b → b - a < 0) ↔ (∀ a b : ℝ, b - a < 0 → a > b) :=
sorry

end inverse_proposition_l1821_182105


namespace P_union_Q_eq_Q_l1821_182190

-- Define the sets P and Q
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x^2 - x > 0}

-- State the theorem
theorem P_union_Q_eq_Q : P ∪ Q = Q := by
  sorry

end P_union_Q_eq_Q_l1821_182190


namespace tomato_growth_l1821_182194

theorem tomato_growth (initial_tomatoes : ℕ) (increase_factor : ℕ) 
  (h1 : initial_tomatoes = 36) 
  (h2 : increase_factor = 100) : 
  initial_tomatoes * increase_factor = 3600 := by
sorry

end tomato_growth_l1821_182194


namespace jessica_remaining_money_l1821_182109

/-- The remaining money after a purchase --/
def remaining_money (initial : ℚ) (spent : ℚ) : ℚ :=
  initial - spent

/-- Proof that Jessica's remaining money is $1.51 --/
theorem jessica_remaining_money :
  remaining_money 11.73 10.22 = 1.51 := by
  sorry

end jessica_remaining_money_l1821_182109


namespace spherical_coordinate_transformation_l1821_182160

theorem spherical_coordinate_transformation (x y z ρ θ φ : ℝ) :
  x = ρ * Real.sin φ * Real.cos θ →
  y = ρ * Real.sin φ * Real.sin θ →
  z = ρ * Real.cos φ →
  x^2 + y^2 + z^2 = ρ^2 →
  x = 4 →
  y = -3 →
  z = -2 →
  ∃ (x' y' z' : ℝ),
    x' = ρ * Real.sin (-φ) * Real.cos (θ + π) ∧
    y' = ρ * Real.sin (-φ) * Real.sin (θ + π) ∧
    z' = ρ * Real.cos (-φ) ∧
    x' = -4 ∧
    y' = 3 ∧
    z' = -2 :=
by sorry

end spherical_coordinate_transformation_l1821_182160


namespace cone_surface_area_l1821_182153

/-- The surface area of a cone with given slant height and base circumference -/
theorem cone_surface_area (slant_height : ℝ) (base_circumference : ℝ) :
  slant_height = 2 →
  base_circumference = 2 * Real.pi →
  (π * (base_circumference / (2 * π))^2) + (π * (base_circumference / (2 * π)) * slant_height) = 3 * π :=
by sorry

end cone_surface_area_l1821_182153


namespace translation_of_B_l1821_182191

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation function
def translate (p : Point) (v : ℝ × ℝ) : Point :=
  (p.1 + v.1, p.2 + v.2)

-- Define the given points
def A : Point := (-1, 0)
def B : Point := (1, 2)
def A₁ : Point := (2, -1)

-- Define the translation vector
def translation_vector : ℝ × ℝ := (A₁.1 - A.1, A₁.2 - A.2)

-- State the theorem
theorem translation_of_B (h : A₁ = translate A translation_vector) :
  translate B translation_vector = (4, 1) := by
  sorry


end translation_of_B_l1821_182191


namespace distinct_pattern_count_is_17_l1821_182177

/-- Represents a 3x3 grid pattern with exactly 3 shaded squares -/
def Pattern := Fin 9 → Bool

/-- Two patterns are rotationally equivalent if one can be obtained from the other by rotation -/
def RotationallyEquivalent (p1 p2 : Pattern) : Prop := sorry

/-- Count of distinct patterns under rotational equivalence -/
def DistinctPatternCount : ℕ := sorry

theorem distinct_pattern_count_is_17 : DistinctPatternCount = 17 := by sorry

end distinct_pattern_count_is_17_l1821_182177


namespace simplify_expression_l1821_182134

theorem simplify_expression (w : ℝ) : 3*w + 6*w - 9*w + 12*w - 15*w + 21 = -3*w + 21 := by
  sorry

end simplify_expression_l1821_182134


namespace building_floors_l1821_182147

/-- Represents a staircase in the building -/
structure Staircase where
  steps : ℕ

/-- Represents the building with three staircases -/
structure Building where
  staircase_a : Staircase
  staircase_b : Staircase
  staircase_c : Staircase

/-- The number of floors in the building is equal to the GCD of the number of steps in each staircase -/
theorem building_floors (b : Building) 
  (h1 : b.staircase_a.steps = 104)
  (h2 : b.staircase_b.steps = 117)
  (h3 : b.staircase_c.steps = 156) : 
  ∃ (floors : ℕ), floors = Nat.gcd (Nat.gcd b.staircase_a.steps b.staircase_b.steps) b.staircase_c.steps ∧ 
    floors = 13 := by
  sorry

end building_floors_l1821_182147


namespace consecutive_numbers_sum_l1821_182198

theorem consecutive_numbers_sum (n : ℕ) : 
  (∃ a : ℕ, (∀ k : ℕ, k < n → ∃ i j l m : ℕ, 
    i < j ∧ j < l ∧ l < m ∧ m < n ∧ 
    a + i + (a + j) + (a + l) + (a + m) = k + (4 * a + 6))) ∧
  (∀ k : ℕ, k ≥ 385 → ¬∃ i j l m : ℕ, 
    i < j ∧ j < l ∧ l < m ∧ m < n ∧ 
    ∃ a : ℕ, a + i + (a + j) + (a + l) + (a + m) = k + (4 * a + 6)) →
  n = 100 := by
sorry

end consecutive_numbers_sum_l1821_182198


namespace ninth_term_value_l1821_182164

/-- An arithmetic sequence with specified third and sixth terms -/
structure ArithmeticSequence where
  a : ℝ  -- first term
  d : ℝ  -- common difference
  third_term : a + 2 * d = 25
  sixth_term : a + 5 * d = 31

/-- The ninth term of the arithmetic sequence -/
def ninth_term (seq : ArithmeticSequence) : ℝ := seq.a + 8 * seq.d

theorem ninth_term_value (seq : ArithmeticSequence) : ninth_term seq = 37 := by
  sorry

end ninth_term_value_l1821_182164


namespace expand_and_simplify_l1821_182121

theorem expand_and_simplify (x : ℝ) : (2*x - 3)^2 - (x + 3)*(x - 2) = 3*x^2 - 13*x + 15 := by
  sorry

end expand_and_simplify_l1821_182121


namespace solutions_equation1_solutions_equation2_l1821_182173

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x^2 + 2 * x - 1 = 0
def equation2 (x : ℝ) : Prop := (x + 2) * (x - 1) = 2 - 2 * x

-- Theorem for the first equation
theorem solutions_equation1 : 
  ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 1/3 ∧ equation1 x₁ ∧ equation1 x₂ ∧ 
  ∀ x : ℝ, equation1 x → x = x₁ ∨ x = x₂ := by sorry

-- Theorem for the second equation
theorem solutions_equation2 : 
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -4 ∧ equation2 x₁ ∧ equation2 x₂ ∧ 
  ∀ x : ℝ, equation2 x → x = x₁ ∨ x = x₂ := by sorry

end solutions_equation1_solutions_equation2_l1821_182173


namespace three_digit_subtraction_convergence_l1821_182132

-- Define a three-digit number type
def ThreeDigitNumber := { n : ℕ // n ≥ 100 ∧ n ≤ 999 }

-- Function to reverse a three-digit number
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber := sorry

-- Function to perform one step of the operation
def step (n : ThreeDigitNumber) : ThreeDigitNumber := sorry

-- Define the set of possible results
def ResultSet : Set ℕ := {0, 495}

-- Theorem statement
theorem three_digit_subtraction_convergence (start : ThreeDigitNumber) :
  ∃ (k : ℕ), (step^[k] start).val ∈ ResultSet := sorry

end three_digit_subtraction_convergence_l1821_182132


namespace day_305_is_thursday_l1821_182133

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents the number of days after Wednesday -/
def daysAfterWednesday (n : Nat) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Wednesday
  | 1 => DayOfWeek.Thursday
  | 2 => DayOfWeek.Friday
  | 3 => DayOfWeek.Saturday
  | 4 => DayOfWeek.Sunday
  | 5 => DayOfWeek.Monday
  | _ => DayOfWeek.Tuesday

theorem day_305_is_thursday :
  daysAfterWednesday (305 - 17) = DayOfWeek.Thursday := by
  sorry

#check day_305_is_thursday

end day_305_is_thursday_l1821_182133


namespace intersection_A_B_range_of_p_l1821_182119

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}

-- Define the set C
def C (p : ℝ) : Set ℝ := {x | x^2 + 4*x + 4 - p^2 < 0}

-- Statement 1: A ∩ B = {x | -3 ≤ x < -1 or 2 < x ≤ 3}
theorem intersection_A_B : A ∩ B = {x | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3} := by sorry

-- Statement 2: The range of p satisfying the given conditions is 0 < p ≤ 1
theorem range_of_p (p : ℝ) (h_p : p > 0) : 
  (C p ⊆ (A ∩ B)) ↔ (p > 0 ∧ p ≤ 1) := by sorry

end intersection_A_B_range_of_p_l1821_182119


namespace rotation_180_maps_points_l1821_182123

def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem rotation_180_maps_points :
  let C : ℝ × ℝ := (-1, 2)
  let D : ℝ × ℝ := (3, 2)
  let C' : ℝ × ℝ := (1, -2)
  let D' : ℝ × ℝ := (-3, -2)
  rotate180 C = C' ∧ rotate180 D = D' := by sorry

end rotation_180_maps_points_l1821_182123


namespace perpendicular_lines_from_parallel_planes_l1821_182154

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (m n : Line) (α β : Plane) 
  (h1 : perpendicular m α)
  (h2 : parallel_line_plane n β)
  (h3 : parallel_plane α β) :
  perpendicular_lines m n :=
sorry

end perpendicular_lines_from_parallel_planes_l1821_182154


namespace round_robin_tournament_probability_l1821_182139

def num_teams : ℕ := 5

-- Define the type for tournament outcomes
def TournamentOutcome := Fin num_teams → Fin num_teams

-- Function to check if an outcome has unique win counts
def has_unique_win_counts (outcome : TournamentOutcome) : Prop :=
  ∀ i j, i ≠ j → outcome i ≠ outcome j

-- Total number of possible outcomes
def total_outcomes : ℕ := 2^(num_teams * (num_teams - 1) / 2)

-- Number of favorable outcomes (where no two teams have the same number of wins)
def favorable_outcomes : ℕ := Nat.factorial num_teams

-- The probability we want to prove
def target_probability : ℚ := favorable_outcomes / total_outcomes

theorem round_robin_tournament_probability :
  target_probability = 15 / 128 := by sorry

end round_robin_tournament_probability_l1821_182139


namespace office_persons_count_l1821_182166

theorem office_persons_count :
  ∀ (N : ℕ) (avg_age : ℚ) (avg_age_5 : ℚ) (avg_age_9 : ℚ) (age_15th : ℕ),
  avg_age = 15 →
  avg_age_5 = 14 →
  avg_age_9 = 16 →
  age_15th = 26 →
  N * avg_age = 5 * avg_age_5 + 9 * avg_age_9 + age_15th →
  N = 16 := by
sorry

end office_persons_count_l1821_182166
