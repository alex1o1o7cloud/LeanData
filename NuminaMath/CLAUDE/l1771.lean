import Mathlib

namespace zoo_animals_l1771_177158

theorem zoo_animals (M P L : ℕ) : 
  (26 ≤ M + P + L ∧ M + P + L ≤ 32) →
  M + L > P →
  P + L = 2 * M →
  M + P > 3 * L →
  P < 2 * L →
  P = 12 := by
sorry

end zoo_animals_l1771_177158


namespace square_sum_given_conditions_l1771_177134

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : x + 3 * y = 9) 
  (h2 : x * y = -15) : 
  x^2 + 9 * y^2 = 171 := by
sorry

end square_sum_given_conditions_l1771_177134


namespace function_inequality_l1771_177194

theorem function_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, 2 * a * x^2 - a * x > 3 - a) →
  a > 24/7 := by
sorry

end function_inequality_l1771_177194


namespace gift_shop_combinations_l1771_177130

/-- The number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 8

/-- The number of colors of ribbon -/
def ribbon_colors : ℕ := 3

/-- The number of types of gift cards -/
def gift_card_types : ℕ := 5

/-- The number of varieties of stickers -/
def sticker_varieties : ℕ := 5

/-- The total number of possible combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_card_types * sticker_varieties

theorem gift_shop_combinations : total_combinations = 600 := by
  sorry

end gift_shop_combinations_l1771_177130


namespace square_equation_solution_l1771_177195

theorem square_equation_solution (x y : ℝ) 
  (h1 : x^2 = y + 3) 
  (h2 : x = 6) : 
  y = 33 := by sorry

end square_equation_solution_l1771_177195


namespace special_rectangle_area_l1771_177142

/-- Represents a rectangle with specific properties -/
structure SpecialRectangle where
  d : ℝ  -- diagonal length
  w : ℝ  -- width
  h : ℝ  -- height (length)
  h_eq_3w : h = 3 * w  -- length is three times the width
  diagonal_eq : d^2 = w^2 + h^2  -- Pythagorean theorem

/-- The area of a SpecialRectangle is (3/10) * d^2 -/
theorem special_rectangle_area (r : SpecialRectangle) : r.w * r.h = (3/10) * r.d^2 := by
  sorry

#check special_rectangle_area

end special_rectangle_area_l1771_177142


namespace fractional_equation_solution_l1771_177107

theorem fractional_equation_solution :
  ∃ x : ℚ, x ≠ 0 ∧ x ≠ -3 ∧ (1 / x = 6 / (x + 3)) ∧ x = 3 / 5 := by
  sorry

end fractional_equation_solution_l1771_177107


namespace task_selection_ways_l1771_177143

/-- The number of ways to select individuals for tasks with specific requirements -/
def select_for_tasks (total_people : ℕ) (task_a_people : ℕ) (task_b_people : ℕ) (task_c_people : ℕ) : ℕ :=
  Nat.choose total_people task_a_people *
  (Nat.choose (total_people - task_a_people) (task_b_people + task_c_people) * Nat.factorial (task_b_people + task_c_people))

/-- Theorem stating the number of ways to select 4 individuals from 10 for the given tasks -/
theorem task_selection_ways :
  select_for_tasks 10 2 1 1 = 2520 := by
  sorry

end task_selection_ways_l1771_177143


namespace rational_coefficient_sum_for_cube_root_two_plus_x_fifth_power_l1771_177183

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.choose n k)

def rational_coefficient_sum (n : ℕ) : ℕ :=
  2 * (binomial_coefficient n 2) + (binomial_coefficient n n)

theorem rational_coefficient_sum_for_cube_root_two_plus_x_fifth_power :
  rational_coefficient_sum 5 = 21 := by sorry

end rational_coefficient_sum_for_cube_root_two_plus_x_fifth_power_l1771_177183


namespace distance_circle_center_to_point_l1771_177181

-- Define the circle's equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 6*x - 2*y + 9

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, -1)

-- Define the given point
def given_point : ℝ × ℝ := (-3, 4)

-- Theorem statement
theorem distance_circle_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = Real.sqrt 61 :=
by sorry

end distance_circle_center_to_point_l1771_177181


namespace quadratic_root_difference_l1771_177117

def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 11 * x + 5 = 0

def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p^2 ∣ n) → p = 1

theorem quadratic_root_difference (p q : ℕ) : 
  (∃ x₁ x₂ : ℝ, 
    quadratic_equation x₁ ∧ 
    quadratic_equation x₂ ∧ 
    x₁ ≠ x₂ ∧
    |x₁ - x₂| = Real.sqrt p / q) →
  q > 0 →
  is_square_free p →
  p + q = 83 := by
sorry

end quadratic_root_difference_l1771_177117


namespace cubic_roots_sum_l1771_177155

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 4*a^2 + 50*a - 7 = 0) →
  (b^3 - 4*b^2 + 50*b - 7 = 0) →
  (c^3 - 4*c^2 + 50*c - 7 = 0) →
  (a + b + c = 4) →
  (a*b + b*c + c*a = 50) →
  (a*b*c = 7) →
  (a + b + 1)^3 + (b + c + 1)^3 + (c + a + 1)^3 = 991 := by
sorry

end cubic_roots_sum_l1771_177155


namespace tangent_point_and_inequality_condition_l1771_177122

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem tangent_point_and_inequality_condition (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (∀ x : ℝ, f a x₀ + (a - 1 / x₀) * (x - x₀) = 0 → x = 0) ∧ 
    x₀ = Real.exp 1) ∧
  (∀ x : ℝ, x ≥ 1 → f a x ≥ a * (2 * x - x^2) → a ≥ 1) :=
sorry

end tangent_point_and_inequality_condition_l1771_177122


namespace tims_lunch_cost_l1771_177113

/-- The total amount Tim spent on lunch, including taxes, surcharge, and tips -/
def total_lunch_cost (meal_cost : ℝ) (tip_rate state_tax_rate city_tax_rate surcharge_rate : ℝ) : ℝ :=
  let tip := meal_cost * tip_rate
  let state_tax := meal_cost * state_tax_rate
  let city_tax := meal_cost * city_tax_rate
  let subtotal := meal_cost + state_tax + city_tax
  let surcharge := subtotal * surcharge_rate
  meal_cost + tip + state_tax + city_tax + surcharge

/-- Theorem stating that Tim's total lunch cost is $78.43 -/
theorem tims_lunch_cost :
  total_lunch_cost 60.50 0.20 0.05 0.03 0.015 = 78.43 := by
  sorry


end tims_lunch_cost_l1771_177113


namespace smallest_n_both_composite_l1771_177104

def is_composite (n : ℕ) : Prop := ∃ a b, a > 1 ∧ b > 1 ∧ a * b = n

theorem smallest_n_both_composite :
  (∀ n : ℕ, n > 0 ∧ n < 13 → ¬(is_composite (2*n - 1) ∧ is_composite (2*n + 1))) ∧
  (is_composite (2*13 - 1) ∧ is_composite (2*13 + 1)) := by
  sorry

end smallest_n_both_composite_l1771_177104


namespace inequality_proof_l1771_177115

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
sorry

end inequality_proof_l1771_177115


namespace square_divisibility_l1771_177101

theorem square_divisibility (n : ℤ) : (∃ k : ℤ, n^2 = 9*k) ∨ (∃ m : ℤ, n^2 = 3*m + 1) := by
  sorry

end square_divisibility_l1771_177101


namespace s_eight_value_l1771_177129

theorem s_eight_value (x : ℝ) (h : x + 1/x = 4) : 
  let S : ℕ → ℝ := λ m => x^m + 1/(x^m)
  S 8 = 37634 := by
  sorry

end s_eight_value_l1771_177129


namespace factorization_xy_squared_minus_x_l1771_177156

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end factorization_xy_squared_minus_x_l1771_177156


namespace ellipse_angle_ratio_l1771_177161

noncomputable section

variables (a b : ℝ) (x y : ℝ)

-- Define the ellipse
def is_on_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define the angles α and β
def alpha (x y a : ℝ) : ℝ := Real.arctan (y / (x + a))
def beta (x y a : ℝ) : ℝ := Real.arctan (y / (x - a))

theorem ellipse_angle_ratio 
  (h1 : a > b) (h2 : b > 0)
  (h3 : is_on_ellipse x y a b)
  (h4 : eccentricity a b = Real.sqrt 3 / 2)
  (h5 : x ≠ a ∧ x ≠ -a) :
  (Real.cos (alpha x y a - beta x y a)) / 
  (Real.cos (alpha x y a + beta x y a)) = 3/5 :=
sorry

end ellipse_angle_ratio_l1771_177161


namespace fixed_fee_determination_l1771_177120

/-- Represents the billing system for an online service provider -/
structure BillingSystem where
  fixedFee : ℝ
  hourlyCharge : ℝ

/-- Calculates the total bill given the billing system and hours used -/
def calculateBill (bs : BillingSystem) (hours : ℝ) : ℝ :=
  bs.fixedFee + bs.hourlyCharge * hours

theorem fixed_fee_determination (bs : BillingSystem) 
  (h1 : calculateBill bs 1 = 18.70)
  (h2 : calculateBill bs 3 = 34.10) : 
  bs.fixedFee = 11.00 := by
  sorry

end fixed_fee_determination_l1771_177120


namespace cakes_sold_l1771_177159

/-- Given the initial number of cakes, the remaining number of cakes,
    and the fact that some cakes were sold, prove that the number of cakes sold is 10. -/
theorem cakes_sold (initial_cakes remaining_cakes : ℕ) 
  (h1 : initial_cakes = 149)
  (h2 : remaining_cakes = 139)
  (h3 : remaining_cakes < initial_cakes) :
  initial_cakes - remaining_cakes = 10 := by
  sorry

#check cakes_sold

end cakes_sold_l1771_177159


namespace complement_M_intersect_N_l1771_177192

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x - 3 > 0}

def N : Set ℝ := {x | |x| ≤ 3}

theorem complement_M_intersect_N :
  (Set.compl M ∩ N) = Set.Icc (-1 : ℝ) 3 := by sorry

end complement_M_intersect_N_l1771_177192


namespace fraction_sum_l1771_177125

theorem fraction_sum : (3 : ℚ) / 9 + (7 : ℚ) / 12 = (11 : ℚ) / 12 := by
  sorry

end fraction_sum_l1771_177125


namespace trig_identity_l1771_177112

theorem trig_identity (θ a b : ℝ) (h : 0 < a) (h' : 0 < b) :
  (Real.sin θ)^6 / a^2 + (Real.cos θ)^6 / b^2 = 1 / (a + b) →
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 := by
  sorry

end trig_identity_l1771_177112


namespace sheila_hourly_rate_l1771_177146

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  eight_hour_days : Nat
  six_hour_days : Nat
  weekly_earnings : Nat

/-- Calculate Sheila's hourly rate --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (8 * schedule.eight_hour_days + 6 * schedule.six_hour_days)

/-- Theorem: Sheila's hourly rate is $6 --/
theorem sheila_hourly_rate :
  let schedule : WorkSchedule := {
    eight_hour_days := 3,
    six_hour_days := 2,
    weekly_earnings := 216
  }
  hourly_rate schedule = 6 := by sorry

end sheila_hourly_rate_l1771_177146


namespace sheila_hourly_rate_l1771_177149

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the hourly rate given a work schedule --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  let total_hours := schedule.monday_hours + schedule.tuesday_hours + 
                     schedule.wednesday_hours + schedule.thursday_hours + 
                     schedule.friday_hours
  schedule.weekly_earnings / total_hours

/-- Theorem: Sheila's hourly rate is $8 --/
theorem sheila_hourly_rate :
  let sheila_schedule : WorkSchedule := {
    monday_hours := 8,
    tuesday_hours := 6,
    wednesday_hours := 8,
    thursday_hours := 6,
    friday_hours := 8,
    weekly_earnings := 288
  }
  hourly_rate sheila_schedule = 8 := by sorry

end sheila_hourly_rate_l1771_177149


namespace impossible_sum_16_l1771_177170

def standard_die := Finset.range 6

theorem impossible_sum_16 (roll1 roll2 : ℕ) :
  roll1 ∈ standard_die → roll2 ∈ standard_die → roll1 + roll2 ≠ 16 := by
  sorry

end impossible_sum_16_l1771_177170


namespace sum_f_half_integers_l1771_177167

def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem sum_f_half_integers (f : ℝ → ℝ) 
  (h1 : is_even (λ x ↦ f (2*x + 2)))
  (h2 : is_odd (λ x ↦ f (x + 1)))
  (h3 : ∃ a b : ℝ, ∀ x ∈ Set.Icc 0 1, f x = a * x + b)
  (h4 : f 4 = 1) :
  (f (3/2) + f (5/2) + f (7/2)) = -1/2 := by
sorry

end sum_f_half_integers_l1771_177167


namespace product_divisible_by_49_l1771_177118

theorem product_divisible_by_49 (a b : ℕ) (h : 7 ∣ (a^2 + b^2)) : 49 ∣ (a * b) := by
  sorry

end product_divisible_by_49_l1771_177118


namespace quarters_borrowed_l1771_177106

/-- Represents the number of quarters Jessica had initially -/
def initial_quarters : ℕ := 8

/-- Represents the number of quarters Jessica has now -/
def current_quarters : ℕ := 5

/-- Represents the number of quarters Jessica's sister borrowed -/
def borrowed_quarters : ℕ := initial_quarters - current_quarters

theorem quarters_borrowed :
  borrowed_quarters = initial_quarters - current_quarters :=
by sorry

end quarters_borrowed_l1771_177106


namespace intersected_half_of_non_intersected_for_three_l1771_177153

/-- The number of unit cubes intersected by space diagonals in a cube of edge length n -/
def intersected_cubes (n : ℕ) : ℕ :=
  if n % 2 = 0 then 4 * n else 4 * n - 3

/-- The total number of unit cubes in a cube of edge length n -/
def total_cubes (n : ℕ) : ℕ := n^3

/-- The number of unit cubes not intersected by space diagonals in a cube of edge length n -/
def non_intersected_cubes (n : ℕ) : ℕ := total_cubes n - intersected_cubes n

/-- Theorem stating that for a cube with edge length 3, the number of intersected cubes
    is exactly half the number of non-intersected cubes -/
theorem intersected_half_of_non_intersected_for_three :
  2 * intersected_cubes 3 = non_intersected_cubes 3 := by
  sorry

end intersected_half_of_non_intersected_for_three_l1771_177153


namespace quadratic_function_properties_l1771_177133

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 6

theorem quadratic_function_properties :
  (f (-1) = 0) ∧ 
  (f 3 = 0) ∧ 
  (f 1 = -8) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≥ -8) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≤ 0) ∧
  (f 1 = -8) ∧
  (f 3 = 0) ∧
  (∀ x, f x ≥ 0 ↔ x ≤ -1 ∨ x ≥ 3) :=
by sorry

end quadratic_function_properties_l1771_177133


namespace right_triangle_area_and_perimeter_l1771_177198

-- Define the right triangle
def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Theorem statement
theorem right_triangle_area_and_perimeter :
  ∀ (a b c : ℝ),
  right_triangle a b c →
  a = 36 →
  b = 48 →
  (1/2 * a * b = 864) ∧ (a + b + c = 144) := by
  sorry


end right_triangle_area_and_perimeter_l1771_177198


namespace strategy_exists_l1771_177199

/-- Represents a question of the form "Is n smaller than a?" --/
structure Question where
  a : ℕ
  deriving Repr

/-- Represents an answer to a question --/
inductive Answer
  | Yes
  | No
  deriving Repr

/-- Represents a strategy for determining n --/
structure Strategy where
  questions : List Question
  decisionFunction : List Answer → ℕ

/-- Theorem stating that a strategy exists to determine n within the given constraints --/
theorem strategy_exists :
  ∃ (s : Strategy),
    (s.questions.length ≤ 10) ∧
    (∀ n : ℕ,
      n > 0 ∧ n ≤ 144 →
      ∃ (answers : List Answer),
        answers.length = s.questions.length ∧
        s.decisionFunction answers = n) :=
  sorry


end strategy_exists_l1771_177199


namespace fraction_calculation_l1771_177105

theorem fraction_calculation : (1/4 + 1/6 - 1/2) / (-1/24) = 2 := by
  sorry

end fraction_calculation_l1771_177105


namespace sin_cos_difference_36_degrees_l1771_177178

theorem sin_cos_difference_36_degrees : 
  Real.sin (36 * π / 180) * Real.cos (36 * π / 180) - 
  Real.cos (36 * π / 180) * Real.sin (36 * π / 180) = 0 := by
  sorry

end sin_cos_difference_36_degrees_l1771_177178


namespace symmetric_line_fixed_point_l1771_177148

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the symmetry relation
def symmetric_about (l1 l2 : Line) (p : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), (y = l1.slope * (x - 4)) → 
  ∃ (x' y' : ℝ), (y' = l2.slope * x' + l2.intercept) ∧ 
  ((x + x') / 2 = p.1) ∧ ((y + y') / 2 = p.2)

-- Define when a line passes through a point
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

-- Theorem statement
theorem symmetric_line_fixed_point (k : ℝ) :
  ∀ (l2 : Line),
  symmetric_about (Line.mk k (-4*k)) l2 (2, 1) →
  passes_through l2 (0, 2) := by sorry

end symmetric_line_fixed_point_l1771_177148


namespace function_inequality_solution_l1771_177179

theorem function_inequality_solution (f : ℕ → ℝ) 
  (h1 : ∀ n ≥ 2, n * f n - (n - 1) * f (n + 1) ≥ 1)
  (h2 : f 2 = 3) :
  ∃ g : ℕ → ℝ, 
    (∀ n ≥ 2, f n = 1 + (n - 1) * g n) ∧ 
    (∀ n ≥ 2, g n ≥ 1) := by
  sorry

end function_inequality_solution_l1771_177179


namespace factor_81_minus_27x_cubed_l1771_177186

theorem factor_81_minus_27x_cubed (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3*x + x^2) := by
  sorry

end factor_81_minus_27x_cubed_l1771_177186


namespace least_integer_in_ratio_l1771_177140

theorem least_integer_in_ratio (a b c : ℕ+) : 
  a.val + b.val + c.val = 90 →
  2 * a = 3 * a →
  5 * a = 3 * b →
  a ≤ b ∧ a ≤ c →
  a.val = 9 := by
sorry

end least_integer_in_ratio_l1771_177140


namespace prove_theta_value_l1771_177136

-- Define the angles in degrees
def angle_VEK : ℝ := 70
def angle_KEW : ℝ := 40
def angle_EVG : ℝ := 110

-- Define θ as a real number
def θ : ℝ := 40

-- Theorem statement
theorem prove_theta_value :
  angle_VEK = 70 ∧
  angle_KEW = 40 ∧
  angle_EVG = 110 →
  θ = 40 := by
  sorry


end prove_theta_value_l1771_177136


namespace power_calculation_l1771_177165

theorem power_calculation : 16^16 * 8^8 / 4^40 = 256 := by
  sorry

end power_calculation_l1771_177165


namespace third_root_of_polynomial_l1771_177171

theorem third_root_of_polynomial (a b : ℚ) :
  (∀ x : ℚ, a * x^3 + 2*(a + b) * x^2 + (b - 2*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 61/35) :=
by sorry

end third_root_of_polynomial_l1771_177171


namespace grocer_sales_theorem_l1771_177141

def sales : List ℕ := [5420, 5660, 6200, 6350, 6500, 6780, 7000, 7200]
def target_average : ℕ := 6600
def num_months : ℕ := 10

theorem grocer_sales_theorem : 
  let total_target := target_average * num_months
  let current_total := sales.sum
  let remaining_months := num_months - sales.length
  let remaining_sales := total_target - current_total
  remaining_sales / remaining_months = 9445 := by sorry

end grocer_sales_theorem_l1771_177141


namespace d_eq_4_sufficient_not_necessary_l1771_177126

/-- An arithmetic sequence with first term 2 and common difference d -/
def arithmetic_seq (n : ℕ) (d : ℝ) : ℝ := 2 + (n - 1) * d

/-- Condition for a_1, a_2, a_5 to form a geometric sequence -/
def is_geometric (d : ℝ) : Prop :=
  (arithmetic_seq 2 d)^2 = (arithmetic_seq 1 d) * (arithmetic_seq 5 d)

/-- d = 4 is a sufficient but not necessary condition for a_1, a_2, a_5 to form a geometric sequence -/
theorem d_eq_4_sufficient_not_necessary :
  (∀ d : ℝ, d = 4 → is_geometric d) ∧
  ¬(∀ d : ℝ, is_geometric d → d = 4) :=
sorry

end d_eq_4_sufficient_not_necessary_l1771_177126


namespace oil_leak_calculation_l1771_177123

theorem oil_leak_calculation (total_leaked : ℕ) (leaked_before : ℕ) 
  (h1 : total_leaked = 6206)
  (h2 : leaked_before = 2475) :
  total_leaked - leaked_before = 3731 :=
by sorry

end oil_leak_calculation_l1771_177123


namespace trapezoid_bases_l1771_177188

theorem trapezoid_bases (d : ℝ) (l : ℝ) (h : d = 15 ∧ l = 17) :
  ∃ (b₁ b₂ : ℝ),
    b₁ = 9 ∧
    b₂ = 25 ∧
    b₁ + b₂ = 2 * l ∧
    b₂ - b₁ = 2 * Real.sqrt (l^2 - d^2) :=
by sorry

end trapezoid_bases_l1771_177188


namespace lucas_L10_units_digit_l1771_177196

/-- Lucas numbers sequence -/
def lucas : ℕ → ℕ
  | 0 => 3
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem lucas_L10_units_digit :
  unitsDigit (lucas (lucas 10)) = 4 := by sorry

end lucas_L10_units_digit_l1771_177196


namespace two_point_distribution_a_value_l1771_177132

/-- A random variable following a two-point distribution -/
structure TwoPointDistribution where
  a : ℝ
  prob_zero : ℝ := 2 * a^2
  prob_one : ℝ := a

/-- The sum of probabilities in a two-point distribution equals 1 -/
axiom prob_sum_eq_one (X : TwoPointDistribution) : X.prob_zero + X.prob_one = 1

/-- Theorem: The value of 'a' in the two-point distribution is 1/2 -/
theorem two_point_distribution_a_value (X : TwoPointDistribution) : X.a = 1/2 := by
  sorry

end two_point_distribution_a_value_l1771_177132


namespace ratio_of_percentages_l1771_177157

theorem ratio_of_percentages (P Q R M N : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hN : N = 0.4 * R)
  (hR : R = 0.75 * P)
  (hP : P ≠ 0) :
  M / N = 1 / 3 := by
sorry

end ratio_of_percentages_l1771_177157


namespace apartment_occupancy_theorem_l1771_177187

/-- Represents an apartment complex with identical buildings -/
structure ApartmentComplex where
  num_buildings : ℕ
  studio_per_building : ℕ
  two_person_per_building : ℕ
  four_person_per_building : ℕ
  occupancy_rate : ℚ

/-- Calculates the number of people living in the apartment complex at the given occupancy rate -/
def occupancy (complex : ApartmentComplex) : ℕ :=
  let max_per_building := 
    complex.studio_per_building + 
    2 * complex.two_person_per_building + 
    4 * complex.four_person_per_building
  let total_max := complex.num_buildings * max_per_building
  ⌊(total_max : ℚ) * complex.occupancy_rate⌋.toNat

theorem apartment_occupancy_theorem (complex : ApartmentComplex) 
  (h1 : complex.num_buildings = 4)
  (h2 : complex.studio_per_building = 10)
  (h3 : complex.two_person_per_building = 20)
  (h4 : complex.four_person_per_building = 5)
  (h5 : complex.occupancy_rate = 3/4) :
  occupancy complex = 210 := by
  sorry

end apartment_occupancy_theorem_l1771_177187


namespace unknown_number_proof_l1771_177168

theorem unknown_number_proof (a b : ℝ) : 
  (a - 3 = b - a) →  -- arithmetic sequence condition
  ((a - 6) / 3 = b / (a - 6)) →  -- geometric sequence condition
  b = 27 := by
sorry

end unknown_number_proof_l1771_177168


namespace focus_coordinates_l1771_177177

/-- The parabola defined by the equation y = (1/8)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/8) * x^2

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- The focus of the parabola y = (1/8)x^2 -/
def focus_of_parabola : Focus := { x := 0, y := 2 }

/-- Theorem: The focus of the parabola y = (1/8)x^2 is (0, 2) -/
theorem focus_coordinates :
  focus_of_parabola.x = 0 ∧ focus_of_parabola.y = 2 :=
sorry

end focus_coordinates_l1771_177177


namespace min_second_longest_side_unit_area_triangle_l1771_177180

theorem min_second_longest_side_unit_area_triangle (a b c : ℝ) (h_area : (1/2) * a * b * Real.sin γ = 1) (h_order : a ≤ b ∧ b ≤ c) (γ : ℝ) :
  b ≥ Real.sqrt 2 := by
  sorry

end min_second_longest_side_unit_area_triangle_l1771_177180


namespace total_sandwiches_l1771_177160

/-- The number of sandwiches made by each person and the total -/
def sandwiches : ℕ → ℕ
| 0 => 49  -- Billy
| 1 => 49 + (49 * 3 / 10)  -- Katelyn
| 2 => (sandwiches 1 * 3) / 5  -- Chloe
| 3 => 25  -- Emma
| 4 => 25 * 2  -- Stella
| _ => 0

/-- The theorem stating the total number of sandwiches made -/
theorem total_sandwiches : 
  sandwiches 0 + sandwiches 1 + sandwiches 2 + sandwiches 3 + sandwiches 4 = 226 := by
  sorry


end total_sandwiches_l1771_177160


namespace ellipse_tangent_inequality_l1771_177102

/-- Represents an ellipse with foci A and B, and semi-major and semi-minor axes a and b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Checks if a point is outside an ellipse -/
def is_outside (ε : Ellipse) (T : Point) : Prop := sorry

/-- Represents a tangent line from a point to an ellipse -/
def tangent_line (ε : Ellipse) (T : Point) : Type := sorry

/-- The length of a tangent line -/
def tangent_length (l : tangent_line ε T) : ℝ := sorry

theorem ellipse_tangent_inequality (ε : Ellipse) (T : Point) 
  (h_outside : is_outside ε T) 
  (TP TQ : tangent_line ε T) : 
  (tangent_length TP) / (tangent_length TQ) ≥ ε.b / ε.a := by sorry

end ellipse_tangent_inequality_l1771_177102


namespace sin_2theta_value_l1771_177151

theorem sin_2theta_value (θ : Real) (h : Real.tan θ + 1 / Real.tan θ = 4) : 
  Real.sin (2 * θ) = 1 / 2 := by
  sorry

end sin_2theta_value_l1771_177151


namespace relay_race_first_leg_time_l1771_177173

/-- Represents a relay race with two runners -/
structure RelayRace where
  y_time : ℝ  -- Time taken by runner y for the first leg
  z_time : ℝ  -- Time taken by runner z for the second leg

/-- Theorem: In a relay race where the second runner takes 26 seconds and the average time per leg is 42 seconds, the first runner takes 58 seconds. -/
theorem relay_race_first_leg_time (race : RelayRace) 
  (h1 : race.z_time = 26)
  (h2 : (race.y_time + race.z_time) / 2 = 42) : 
  race.y_time = 58 := by
  sorry

end relay_race_first_leg_time_l1771_177173


namespace product_not_always_greater_than_factors_l1771_177116

theorem product_not_always_greater_than_factors : ∃ (a b : ℝ), a * b ≤ a ∨ a * b ≤ b := by
  sorry

end product_not_always_greater_than_factors_l1771_177116


namespace basketball_fall_certain_l1771_177197

-- Define the type for events
inductive Event
  | RainTomorrow
  | RollEvenDice
  | TVAdvertisement
  | BasketballFall

-- Define a predicate for certain events
def IsCertain (e : Event) : Prop :=
  match e with
  | Event.BasketballFall => True
  | _ => False

-- Define the law of gravity (simplified)
axiom law_of_gravity : ∀ (object : Type), object → object → Prop

-- Theorem statement
theorem basketball_fall_certain :
  ∀ (e : Event), IsCertain e ↔ e = Event.BasketballFall :=
sorry

end basketball_fall_certain_l1771_177197


namespace cubic_meter_to_cubic_cm_l1771_177145

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

/-- The number of cubic centimeters in one cubic meter -/
def cubic_cm_in_cubic_meter : ℝ := (meters_to_cm) ^ 3

theorem cubic_meter_to_cubic_cm : 
  cubic_cm_in_cubic_meter = 1000000 :=
sorry

end cubic_meter_to_cubic_cm_l1771_177145


namespace min_pencils_theorem_l1771_177154

def min_pencils_to_take (red blue green : ℕ) (red_goal blue_goal green_goal : ℕ) : ℕ :=
  (red + blue + green) - (red - red_goal).min 0 - (blue - blue_goal).min 0 - (green - green_goal).min 0 + 1

theorem min_pencils_theorem :
  min_pencils_to_take 15 13 8 1 2 3 = 22 :=
by sorry

end min_pencils_theorem_l1771_177154


namespace jasons_quarters_l1771_177137

theorem jasons_quarters (initial final given : ℕ) 
  (h1 : initial = 49)
  (h2 : final = 74)
  (h3 : final = initial + given) :
  given = 25 := by
  sorry

end jasons_quarters_l1771_177137


namespace system_solutions_l1771_177163

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  x^2 + y^2 = z^2 ∧ x*z = y^2 ∧ x*y = 10

-- State the theorem
theorem system_solutions :
  ∃ (x y z : ℝ), system x y z ∧
  ((x = Real.sqrt 10 ∧ y = Real.sqrt 10 ∧ z = Real.sqrt 10) ∨
   (x = -Real.sqrt 10 ∧ y = -Real.sqrt 10 ∧ z = -Real.sqrt 10)) :=
by sorry

end system_solutions_l1771_177163


namespace sqrt_three_irrational_l1771_177110

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_three_irrational_l1771_177110


namespace max_value_sin_cos_l1771_177138

theorem max_value_sin_cos (θ : Real) (h : 0 < θ ∧ θ < π) :
  ∃ (max : Real), max = (4 * Real.sqrt 3) / 9 ∧
  ∀ (x : Real), 0 < x ∧ x < π →
    Real.sin (x / 2) * (1 + Real.cos x) ≤ max :=
by sorry

end max_value_sin_cos_l1771_177138


namespace unique_number_with_specific_divisors_l1771_177111

theorem unique_number_with_specific_divisors : ∃! n : ℕ, 
  (9 ∣ n) ∧ (5 ∣ n) ∧ (Finset.card (Nat.divisors n) = 14) ∧ (n = 3645) := by
  sorry

end unique_number_with_specific_divisors_l1771_177111


namespace necklace_beads_l1771_177103

theorem necklace_beads (total : ℕ) (blue : ℕ) (red : ℕ) (white : ℕ) (silver : ℕ) :
  total = 40 →
  red = 2 * blue →
  white = blue + red →
  silver = 10 →
  blue + red + white + silver = total →
  blue = 5 := by
sorry

end necklace_beads_l1771_177103


namespace investment_doubling_time_l1771_177169

/-- The minimum number of years required for an investment to at least double -/
theorem investment_doubling_time (A r : ℝ) (h1 : A > 0) (h2 : r > 0) :
  let t := Real.log 2 / Real.log (1 + r)
  ∀ s : ℝ, s ≥ t → A * (1 + r) ^ s ≥ 2 * A :=
by sorry

end investment_doubling_time_l1771_177169


namespace largest_remaining_circle_l1771_177166

/-- Represents a circle with a given diameter -/
structure Circle where
  diameter : ℝ
  diameter_pos : diameter > 0

/-- The problem setup -/
def plywood_problem (initial : Circle) (cutout1 : Circle) (cutout2 : Circle) : Prop :=
  initial.diameter = 30 ∧ cutout1.diameter = 20 ∧ cutout2.diameter = 10

/-- The theorem to be proved -/
theorem largest_remaining_circle 
  (initial : Circle) (cutout1 : Circle) (cutout2 : Circle) 
  (h : plywood_problem initial cutout1 cutout2) : 
  ∃ (largest : Circle), largest.diameter = 30 / 7 ∧ 
  ∀ (c : Circle), c.diameter ≤ largest.diameter :=
sorry

end largest_remaining_circle_l1771_177166


namespace fence_length_l1771_177121

/-- For a rectangular yard with one side of 40 feet and an area of 480 square feet,
    the sum of the lengths of the other three sides is 64 feet. -/
theorem fence_length (length width : ℝ) : 
  width = 40 → 
  length * width = 480 → 
  2 * length + width = 64 := by
sorry

end fence_length_l1771_177121


namespace chameleon_theorem_l1771_177147

/-- Represents the resting period before catching the m-th fly -/
def resting_period (m : ℕ) : ℕ :=
  sorry

/-- Represents the total time before catching the m-th fly -/
def total_time (m : ℕ) : ℕ :=
  sorry

/-- Represents the number of flies caught after t minutes -/
def flies_caught (t : ℕ) : ℕ :=
  sorry

/-- The chameleon's resting and catching behavior -/
axiom resting_rule_1 : resting_period 1 = 1
axiom resting_rule_2 : ∀ m : ℕ, resting_period (2 * m) = resting_period m
axiom resting_rule_3 : ∀ m : ℕ, resting_period (2 * m + 1) = resting_period m + 1
axiom catch_instantly : ∀ m : ℕ, total_time (m + 1) = total_time m + resting_period (m + 1) + 1

theorem chameleon_theorem :
  (∃ m : ℕ, m = 510 ∧ resting_period (m + 1) = 9 ∧ ∀ k < m, resting_period (k + 1) < 9) ∧
  (total_time 98 = 312) ∧
  (flies_caught 1999 = 462) :=
sorry

end chameleon_theorem_l1771_177147


namespace barry_larry_reach_l1771_177190

/-- The maximum height Barry and Larry can reach when Barry stands on Larry's shoulders -/
def max_reach (barry_reach : ℝ) (larry_height : ℝ) (larry_shoulder_ratio : ℝ) : ℝ :=
  barry_reach + larry_height * larry_shoulder_ratio

/-- Theorem stating the maximum reach of Barry and Larry -/
theorem barry_larry_reach :
  let barry_reach : ℝ := 5
  let larry_height : ℝ := 5
  let larry_shoulder_ratio : ℝ := 0.8
  max_reach barry_reach larry_height larry_shoulder_ratio = 9 := by
  sorry

end barry_larry_reach_l1771_177190


namespace three_digit_number_from_sum_l1771_177189

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h_a : a < 10
  h_b : b < 10
  h_c : c < 10

/-- Calculates the sum of permutations of a three-digit number -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c +
  100 * n.a + 10 * n.c + n.b +
  100 * n.b + 10 * n.a + n.c +
  100 * n.b + 10 * n.c + n.a +
  100 * n.c + 10 * n.a + n.b +
  100 * n.c + 10 * n.b + n.a

theorem three_digit_number_from_sum (N : Nat) (h_N : N = 3194) :
  ∃ (n : ThreeDigitNumber), sumOfPermutations n = N ∧ n.a = 3 ∧ n.b = 5 ∧ n.c = 8 := by
  sorry

#eval sumOfPermutations { a := 3, b := 5, c := 8, h_a := by norm_num, h_b := by norm_num, h_c := by norm_num }

end three_digit_number_from_sum_l1771_177189


namespace linear_system_ratio_l1771_177135

theorem linear_system_ratio (x y a b : ℝ) 
  (eq1 : 4 * x - 6 * y = a)
  (eq2 : 9 * x - 6 * y = b)
  (x_nonzero : x ≠ 0)
  (y_nonzero : y ≠ 0)
  (b_nonzero : b ≠ 0) :
  a / b = 2 := by
sorry

end linear_system_ratio_l1771_177135


namespace function_passes_through_point_l1771_177119

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 2
  f 1 = 3 := by
  sorry

end function_passes_through_point_l1771_177119


namespace exists_common_language_l1771_177127

/-- Represents a scientist at the conference -/
structure Scientist where
  id : Nat
  languages : Finset String
  lang_count : languages.card ≤ 4

/-- The set of all scientists at the conference -/
def Scientists : Finset Scientist :=
  sorry

/-- The number of scientists at the conference -/
axiom scientist_count : Scientists.card = 200

/-- For any three scientists, at least two share a common language -/
axiom common_language (s1 s2 s3 : Scientist) :
  s1 ∈ Scientists → s2 ∈ Scientists → s3 ∈ Scientists →
  ∃ (l : String), (l ∈ s1.languages ∧ l ∈ s2.languages) ∨
                  (l ∈ s1.languages ∧ l ∈ s3.languages) ∨
                  (l ∈ s2.languages ∧ l ∈ s3.languages)

/-- Main theorem: There exists a language spoken by at least 26 scientists -/
theorem exists_common_language :
  ∃ (l : String), (Scientists.filter (fun s => l ∈ s.languages)).card ≥ 26 :=
sorry

end exists_common_language_l1771_177127


namespace sculpture_surface_area_l1771_177182

/-- Represents a layer in the sculpture -/
structure Layer where
  cubes : Nat
  exposedTopFaces : Nat
  exposedSideFaces : Nat

/-- Represents the sculpture -/
def Sculpture : List Layer := [
  { cubes := 1, exposedTopFaces := 1, exposedSideFaces := 4 },
  { cubes := 4, exposedTopFaces := 4, exposedSideFaces := 12 },
  { cubes := 9, exposedTopFaces := 9, exposedSideFaces := 6 },
  { cubes := 6, exposedTopFaces := 6, exposedSideFaces := 0 }
]

/-- Calculates the exposed surface area of a layer -/
def layerSurfaceArea (layer : Layer) : Nat :=
  layer.exposedTopFaces + layer.exposedSideFaces

/-- Calculates the total exposed surface area of the sculpture -/
def totalSurfaceArea (sculpture : List Layer) : Nat :=
  List.foldl (λ acc layer => acc + layerSurfaceArea layer) 0 sculpture

/-- Theorem: The total exposed surface area of the sculpture is 42 square meters -/
theorem sculpture_surface_area : totalSurfaceArea Sculpture = 42 := by
  sorry

end sculpture_surface_area_l1771_177182


namespace complex_number_opposite_parts_l1771_177124

theorem complex_number_opposite_parts (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
sorry

end complex_number_opposite_parts_l1771_177124


namespace tan_neg_3780_degrees_l1771_177193

theorem tan_neg_3780_degrees : Real.tan ((-3780 : ℝ) * π / 180) = 0 := by
  sorry

end tan_neg_3780_degrees_l1771_177193


namespace vector_dot_product_l1771_177176

theorem vector_dot_product (a b : ℝ × ℝ) (h1 : a = (1, -1)) (h2 : b = (-1, 2)) :
  (2 • a + b) • a = 1 := by
  sorry

end vector_dot_product_l1771_177176


namespace initial_birds_l1771_177100

theorem initial_birds (initial_birds final_birds additional_birds : ℕ) 
  (h1 : additional_birds = 21)
  (h2 : final_birds = 35)
  (h3 : final_birds = initial_birds + additional_birds) : 
  initial_birds = 14 := by
  sorry

end initial_birds_l1771_177100


namespace sum_of_xyz_l1771_177175

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 30) (h2 : x * z = 60) (h3 : y * z = 90) : 
  x + y + z = 11 * Real.sqrt 5 := by
  sorry

end sum_of_xyz_l1771_177175


namespace degree_of_example_monomial_not_six_l1771_177131

/-- The degree of a monomial is the sum of the exponents of its variables -/
def monomial_degree (m : Polynomial ℤ) : ℕ := sorry

/-- A function to represent the monomial -2^2xab^2 -/
def example_monomial : Polynomial ℤ := sorry

theorem degree_of_example_monomial_not_six :
  monomial_degree example_monomial ≠ 6 := by sorry

end degree_of_example_monomial_not_six_l1771_177131


namespace graph_transform_properties_l1771_177139

/-- A graph in a 2D plane -/
structure Graph where
  -- We don't need to define the internal structure of the graph
  -- as we're only concerned with its properties under transformations

/-- Properties of a graph that may or may not change under transformations -/
structure GraphProperties where
  shape : Bool  -- True if shape is preserved
  size : Bool   -- True if size is preserved
  direction : Bool  -- True if direction is preserved

/-- Rotation of a graph -/
def rotate (g : Graph) : Graph :=
  sorry

/-- Translation of a graph -/
def translate (g : Graph) : Graph :=
  sorry

/-- Properties preserved under rotation and translation -/
def properties_after_transform (g : Graph) : GraphProperties :=
  sorry

theorem graph_transform_properties :
  ∀ g : Graph,
    let props := properties_after_transform g
    props.shape = true ∧ props.size = true ∧ props.direction = false :=
by sorry

end graph_transform_properties_l1771_177139


namespace prime_between_n_and_nfactorial_l1771_177144

theorem prime_between_n_and_nfactorial (n : ℕ) (h : n > 2) :
  ∃ p : ℕ, Prime p ∧ n < p ∧ p ≤ n! :=
by sorry

end prime_between_n_and_nfactorial_l1771_177144


namespace f_monotone_increasing_on_negative_l1771_177191

-- Define the function
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem f_monotone_increasing_on_negative : 
  MonotoneOn f (Set.Iic 0) :=
sorry

end f_monotone_increasing_on_negative_l1771_177191


namespace no_solution_double_inequality_l1771_177152

theorem no_solution_double_inequality :
  ¬ ∃ y : ℝ, (3 * y^2 - 4 * y - 5 < (y + 1)^2) ∧ ((y + 1)^2 < 4 * y^2 - y - 1) :=
by sorry

end no_solution_double_inequality_l1771_177152


namespace adjacent_different_country_probability_l1771_177109

/-- Represents a country with delegates -/
structure Country where
  delegates : Nat
  deriving Repr

/-- Represents a seating arrangement -/
structure SeatingArrangement where
  total_seats : Nat
  countries : List Country
  deriving Repr

/-- Calculates the probability of each delegate sitting adjacent to at least one delegate from a different country -/
def probability_adjacent_different_country (arrangement : SeatingArrangement) : Rat :=
  sorry

/-- The specific seating arrangement from the problem -/
def problem_arrangement : SeatingArrangement :=
  { total_seats := 12
  , countries := List.replicate 4 { delegates := 2 }
  }

/-- Theorem stating the probability for the given seating arrangement -/
theorem adjacent_different_country_probability :
  probability_adjacent_different_country problem_arrangement = 4897683 / 9979200 :=
  sorry

end adjacent_different_country_probability_l1771_177109


namespace child_share_calculation_l1771_177174

theorem child_share_calculation (total_amount : ℚ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_amount = 4500 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  (ratio_b : ℚ) / (ratio_a + ratio_b + ratio_c : ℚ) * total_amount = 1500 := by
sorry

end child_share_calculation_l1771_177174


namespace sufficient_not_necessary_condition_l1771_177185

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ a, a > 2 → a * (a - 2) > 0) ∧ 
  (∃ a, a * (a - 2) > 0 ∧ ¬(a > 2)) :=
by sorry

end sufficient_not_necessary_condition_l1771_177185


namespace count_sets_with_seven_l1771_177184

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

theorem count_sets_with_seven :
  (Finset.filter (fun s : Finset ℕ => 
    s.card = 3 ∧ 
    (∀ x ∈ s, x ∈ S) ∧ 
    (s.sum id = 21) ∧ 
    (7 ∈ s))
  (Finset.powerset S)).card = 5 :=
sorry

end count_sets_with_seven_l1771_177184


namespace rectangular_prism_cutout_l1771_177172

theorem rectangular_prism_cutout (x y : ℕ) : 
  (15 * 5 * 4 - y * 5 * x = 120) → (x < 4 ∧ y < 15) → x + y = 15 := by
  sorry

end rectangular_prism_cutout_l1771_177172


namespace min_value_product_l1771_177150

theorem min_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x + 1/x) * (y + 1/y) ≥ 33/4 := by
sorry

end min_value_product_l1771_177150


namespace largest_ball_on_torus_l1771_177162

/-- The radius of the largest spherical ball that can be placed on top of a torus -/
def largest_ball_radius (inner_radius outer_radius : ℝ) (torus_center : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating that the largest ball radius for the given torus is 4 -/
theorem largest_ball_on_torus :
  let inner_radius : ℝ := 3
  let outer_radius : ℝ := 5
  let torus_center : ℝ × ℝ × ℝ := (4, 0, 1)
  largest_ball_radius inner_radius outer_radius torus_center = 4 := by
  sorry

end largest_ball_on_torus_l1771_177162


namespace ratio_problem_l1771_177128

theorem ratio_problem (a b : ℝ) : 
  (a / b = 3 / 8) → 
  ((a - 24) / (b - 24) = 4 / 9) → 
  max a b = 192 := by
sorry

end ratio_problem_l1771_177128


namespace smallest_solution_is_negative_one_l1771_177114

-- Define the equation
def equation (x : ℝ) : Prop :=
  3 * x / (x - 3) + (3 * x^2 - 36) / (x + 3) = 15

-- Theorem statement
theorem smallest_solution_is_negative_one :
  (∃ x : ℝ, equation x) ∧ 
  (∀ y : ℝ, equation y → y ≥ -1) ∧
  equation (-1) :=
sorry

end smallest_solution_is_negative_one_l1771_177114


namespace mustang_length_proof_l1771_177108

theorem mustang_length_proof (smallest_model : ℝ) (mid_size_model : ℝ) (full_size : ℝ)
  (h1 : smallest_model = 12)
  (h2 : smallest_model = mid_size_model / 2)
  (h3 : mid_size_model = full_size / 10) :
  full_size = 240 := by
  sorry

end mustang_length_proof_l1771_177108


namespace first_nonzero_digit_after_decimal_1_198_l1771_177164

theorem first_nonzero_digit_after_decimal_1_198 : ∃ (n : ℕ) (d : ℕ), 
  1 ≤ d ∧ d ≤ 9 ∧ 
  (∃ (m : ℕ), 1/198 = (n : ℚ)/10^m + d/(10^(m+1) : ℚ) + (1/198 - (n : ℚ)/10^m - d/(10^(m+1) : ℚ)) ∧ 
   0 ≤ 1/198 - (n : ℚ)/10^m - d/(10^(m+1) : ℚ) ∧ 
   1/198 - (n : ℚ)/10^m - d/(10^(m+1) : ℚ) < 1/(10^(m+1) : ℚ)) ∧
  d = 5 :=
by sorry

end first_nonzero_digit_after_decimal_1_198_l1771_177164
