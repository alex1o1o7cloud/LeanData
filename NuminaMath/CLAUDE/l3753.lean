import Mathlib

namespace f_increasing_range_of_a_l3753_375305

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (2 - a/2)*x + 2

/-- The theorem stating the range of values for a -/
theorem f_increasing_range_of_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (8/3) 4 :=
sorry

end f_increasing_range_of_a_l3753_375305


namespace factorization_of_2x_squared_minus_2_l3753_375396

theorem factorization_of_2x_squared_minus_2 (x : ℝ) : 2*x^2 - 2 = 2*(x+1)*(x-1) := by
  sorry

end factorization_of_2x_squared_minus_2_l3753_375396


namespace factorization_equality_l3753_375365

theorem factorization_equality (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end factorization_equality_l3753_375365


namespace triangle_area_l3753_375336

theorem triangle_area (a b c : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 5) (h3 : c = 2) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = Real.sqrt 11 / 2 := by
sorry

end triangle_area_l3753_375336


namespace expand_product_l3753_375376

theorem expand_product (x : ℝ) : 3 * (x - 3) * (x + 5) = 3 * x^2 + 6 * x - 45 := by
  sorry

end expand_product_l3753_375376


namespace charity_fundraising_l3753_375358

theorem charity_fundraising (total_amount : ℕ) (num_people : ℕ) (amount_per_person : ℕ) :
  total_amount = 1500 →
  num_people = 6 →
  amount_per_person * num_people = total_amount →
  amount_per_person = 250 := by
  sorry

end charity_fundraising_l3753_375358


namespace inequality_solution_range_l3753_375332

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 2| + |x - 3| ≤ a) → a ≥ 5 := by
  sorry

end inequality_solution_range_l3753_375332


namespace quadratic_root_implies_coefficient_l3753_375375

theorem quadratic_root_implies_coefficient (a : ℝ) : 
  (3 : ℝ)^2 + a * 3 + 9 = 0 → a = -6 := by
  sorry

end quadratic_root_implies_coefficient_l3753_375375


namespace total_supervisors_is_21_l3753_375340

/-- The number of buses used for the field trip. -/
def num_buses : ℕ := 7

/-- The number of adult supervisors per bus. -/
def supervisors_per_bus : ℕ := 3

/-- The total number of supervisors for the field trip. -/
def total_supervisors : ℕ := num_buses * supervisors_per_bus

/-- Theorem stating that the total number of supervisors is 21. -/
theorem total_supervisors_is_21 : total_supervisors = 21 := by
  sorry

end total_supervisors_is_21_l3753_375340


namespace function_value_at_seven_l3753_375383

/-- Given a function f(x) = ax^7 + bx^3 + cx - 5 where a, b, c are constants,
    if f(-7) = 7, then f(7) = -17 -/
theorem function_value_at_seven 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^7 + b * x^3 + c * x - 5) 
  (h2 : f (-7) = 7) : 
  f 7 = -17 := by
sorry

end function_value_at_seven_l3753_375383


namespace remainder_theorem_l3753_375313

theorem remainder_theorem (m : ℤ) (h : m % 9 = 3) : (3 * m + 2436) % 9 = 0 := by
  sorry

end remainder_theorem_l3753_375313


namespace complex_sum_magnitude_l3753_375378

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 1) : 
  Complex.abs (a + b + c) = 1 := by
  sorry

end complex_sum_magnitude_l3753_375378


namespace golf_carts_needed_l3753_375318

theorem golf_carts_needed (patrons_per_cart : ℕ) (car_patrons : ℕ) (bus_patrons : ℕ) : 
  patrons_per_cart = 3 →
  car_patrons = 12 →
  bus_patrons = 27 →
  ((car_patrons + bus_patrons) + patrons_per_cart - 1) / patrons_per_cart = 13 := by
sorry

end golf_carts_needed_l3753_375318


namespace inequality_proof_l3753_375369

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a^2 < 4*b) (h2 : c^2 < 4*d) : 
  ((a + c)/2)^2 < 4*((b + d)/2) := by
  sorry

end inequality_proof_l3753_375369


namespace apple_pie_consumption_l3753_375377

theorem apple_pie_consumption (apples_per_serving : ℝ) (num_guests : ℕ) (num_pies : ℕ) (servings_per_pie : ℕ) :
  apples_per_serving = 1.5 →
  num_guests = 12 →
  num_pies = 3 →
  servings_per_pie = 8 →
  (num_pies * servings_per_pie * apples_per_serving) / num_guests = 3 := by
  sorry

end apple_pie_consumption_l3753_375377


namespace figure_placement_count_l3753_375307

/-- Represents a configuration of figure placements -/
structure FigurePlacement where
  pages : Fin 6 → Fin 3
  order_preserved : ∀ i j : Fin 4, i < j → pages i ≤ pages j

/-- The number of valid figure placements -/
def count_placements : ℕ := sorry

/-- Theorem stating the correct number of placements -/
theorem figure_placement_count : count_placements = 225 := by sorry

end figure_placement_count_l3753_375307


namespace jane_initial_pick_is_one_fourth_l3753_375359

/-- The fraction of tomatoes Jane initially picked from a tomato plant -/
def jane_initial_pick : ℚ :=
  let initial_tomatoes : ℕ := 100
  let second_pick : ℕ := 20
  let third_pick : ℕ := 2 * second_pick
  let remaining_tomatoes : ℕ := 15
  (initial_tomatoes - second_pick - third_pick - remaining_tomatoes) / initial_tomatoes

theorem jane_initial_pick_is_one_fourth :
  jane_initial_pick = 1 / 4 := by
  sorry

end jane_initial_pick_is_one_fourth_l3753_375359


namespace bottle_production_rate_l3753_375386

/-- Given that 5 identical machines can produce 900 bottles in 4 minutes at a constant rate,
    prove that 6 such machines can produce 270 bottles per minute. -/
theorem bottle_production_rate (rate : ℕ → ℕ → ℕ) : 
  (rate 5 4 = 900) → (rate 6 1 = 270) :=
by
  sorry


end bottle_production_rate_l3753_375386


namespace blue_eyed_students_l3753_375357

theorem blue_eyed_students (total : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 40 →
  both = 8 →
  neither = 5 →
  ∃ (blue : ℕ), 
    blue + (3 * blue - both) + both + neither = total ∧
    blue = 10 := by
  sorry

end blue_eyed_students_l3753_375357


namespace max_sphere_radius_squared_for_problem_config_l3753_375338

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- The maximum squared radius of a sphere that can fit within two intersecting cones -/
def maxSphereRadiusSquared (ic : IntersectingCones) : ℝ := sorry

/-- The specific configuration described in the problem -/
def problemConfig : IntersectingCones :=
  { cone1 := { baseRadius := 4, height := 10 }
  , cone2 := { baseRadius := 4, height := 10 }
  , intersectionDistance := 4
  }

theorem max_sphere_radius_squared_for_problem_config :
  maxSphereRadiusSquared problemConfig = 4176 / 841 :=
sorry

end max_sphere_radius_squared_for_problem_config_l3753_375338


namespace solution_satisfies_system_solution_is_unique_l3753_375399

/-- A system of linear equations with three variables -/
structure LinearSystem where
  eq1 : ℝ → ℝ → ℝ → Prop
  eq2 : ℝ → ℝ → ℝ → Prop
  eq3 : ℝ → ℝ → ℝ → Prop

/-- The specific system of equations from the problem -/
def problemSystem : LinearSystem where
  eq1 := fun x y z => x + y + z = 15
  eq2 := fun x y z => x - y + z = 5
  eq3 := fun x y z => x + y - z = 10

/-- The solution to the system of equations -/
def solution : ℝ × ℝ × ℝ := (7.5, 5, 2.5)

/-- Theorem stating that the solution satisfies the system of equations -/
theorem solution_satisfies_system :
  let (x, y, z) := solution
  problemSystem.eq1 x y z ∧
  problemSystem.eq2 x y z ∧
  problemSystem.eq3 x y z :=
by sorry

/-- Theorem stating that the solution is unique -/
theorem solution_is_unique :
  ∀ x y z, 
    problemSystem.eq1 x y z →
    problemSystem.eq2 x y z →
    problemSystem.eq3 x y z →
    (x, y, z) = solution :=
by sorry

end solution_satisfies_system_solution_is_unique_l3753_375399


namespace landscape_length_l3753_375398

theorem landscape_length (breadth : ℝ) 
  (length_eq : length = 8 * breadth)
  (playground_area : ℝ)
  (playground_eq : playground_area = 1200)
  (playground_ratio : playground_area = (1/6) * (length * breadth)) : 
  length = 240 := by
  sorry

end landscape_length_l3753_375398


namespace one_root_quadratic_sum_l3753_375371

theorem one_root_quadratic_sum (a b : ℝ) : 
  (∃! x : ℝ, x^2 + a*x + b = 0) → 
  (a = 2*b - 3) → 
  (∃ b₁ b₂ : ℝ, (b = b₁ ∨ b = b₂) ∧ b₁ + b₂ = 4) := by
sorry

end one_root_quadratic_sum_l3753_375371


namespace C_div_D_eq_17_l3753_375390

noncomputable def C : ℝ := ∑' n, if n % 4 ≠ 0 ∧ n % 2 = 0 then (-1)^((n/2) % 2 + 1) / n^2 else 0

noncomputable def D : ℝ := ∑' n, if n % 4 = 0 then (-1)^(n/4 + 1) / n^2 else 0

theorem C_div_D_eq_17 : C / D = 17 := by sorry

end C_div_D_eq_17_l3753_375390


namespace ramanujan_hardy_complex_game_l3753_375310

theorem ramanujan_hardy_complex_game (product h r : ℂ) : 
  product = 24 - 10*I ∧ h = 3 + 4*I ∧ product = h * r →
  r = 112/25 - 126/25*I := by sorry

end ramanujan_hardy_complex_game_l3753_375310


namespace anthonys_pets_l3753_375361

theorem anthonys_pets (initial_pets : ℕ) : 
  (initial_pets - 6 : ℚ) * (4/5) = 8 → initial_pets = 16 :=
by
  sorry

end anthonys_pets_l3753_375361


namespace curve_is_circle_l3753_375320

/-- The curve represented by the equation |x-1| = √(1-(y+1)²) -/
def curve_equation (x y : ℝ) : Prop := |x - 1| = Real.sqrt (1 - (y + 1)^2)

/-- The equation of a circle with center (1, -1) and radius 1 -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1

/-- Theorem stating that the curve equation represents a circle -/
theorem curve_is_circle :
  ∀ x y : ℝ, curve_equation x y ↔ circle_equation x y :=
by sorry

end curve_is_circle_l3753_375320


namespace positive_sum_one_inequality_l3753_375319

theorem positive_sum_one_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by
  sorry

end positive_sum_one_inequality_l3753_375319


namespace geometry_test_passing_l3753_375331

theorem geometry_test_passing (total_problems : Nat) (passing_percentage : Rat) 
  (hp : total_problems = 50)
  (hq : passing_percentage = 85 / 100) : 
  (max_missed_problems : Nat) → 
  (max_missed_problems = total_problems - Int.ceil (passing_percentage * total_problems)) ∧
  max_missed_problems = 7 := by
  sorry

end geometry_test_passing_l3753_375331


namespace factorization_of_360_l3753_375370

theorem factorization_of_360 : ∃ (p₁ p₂ p₃ : Nat) (e₁ e₂ e₃ : Nat),
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
  360 = p₁^e₁ * p₂^e₂ * p₃^e₃ ∧
  (∀ q : Nat, Prime q → q ∣ 360 → (q = p₁ ∨ q = p₂ ∨ q = p₃)) ∧
  (e₁ ≤ 3 ∧ e₂ ≤ 3 ∧ e₃ ≤ 3) ∧
  (e₁ = 3 ∨ e₂ = 3 ∨ e₃ = 3) :=
by sorry

end factorization_of_360_l3753_375370


namespace sqrt_225_equals_15_l3753_375334

theorem sqrt_225_equals_15 : Real.sqrt 225 = 15 := by
  sorry

end sqrt_225_equals_15_l3753_375334


namespace merchant_discount_l3753_375392

theorem merchant_discount (markup : ℝ) (profit : ℝ) (discount : ℝ) : 
  markup = 0.75 → 
  profit = 0.225 → 
  discount = (markup + 1 - (profit + 1)) / (markup + 1) * 100 →
  discount = 30 := by
  sorry

end merchant_discount_l3753_375392


namespace arithmetic_geq_geometric_l3753_375360

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geq_geometric
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h1 : a 1 = b 1)
  (h1_pos : a 1 > 0)
  (hn : a n = b n)
  (hn_pos : a n > 0)
  (n : ℕ)
  (hn_gt_1 : n > 1) :
  ∀ m : ℕ, 1 < m → m < n → a m ≥ b m :=
sorry

end arithmetic_geq_geometric_l3753_375360


namespace waiter_tables_l3753_375394

/-- Given a waiter's customer and table information, prove the number of tables. -/
theorem waiter_tables
  (initial_customers : ℕ)
  (departed_customers : ℕ)
  (people_per_table : ℕ)
  (h1 : initial_customers = 44)
  (h2 : departed_customers = 12)
  (h3 : people_per_table = 8)
  : (initial_customers - departed_customers) / people_per_table = 4 := by
  sorry

end waiter_tables_l3753_375394


namespace sqrt2_plus_1_power_l3753_375301

theorem sqrt2_plus_1_power (n : ℕ+) :
  ∃ m : ℕ+, (Real.sqrt 2 + 1) ^ n.val = Real.sqrt m.val + Real.sqrt (m.val - 1) := by
  sorry

end sqrt2_plus_1_power_l3753_375301


namespace no_real_solutions_l3753_375330

theorem no_real_solutions :
  ¬∃ (x y z u : ℝ), x^4 - 17 = y^4 - 7 ∧ 
                    x^4 - 17 = z^4 + 19 ∧ 
                    x^4 - 17 = u^4 + 5 ∧ 
                    x^4 - 17 = x * y * z * u :=
by sorry

end no_real_solutions_l3753_375330


namespace original_price_calculation_l3753_375321

-- Define the original cost price as a real number
variable (P : ℝ)

-- Define the selling price
def selling_price : ℝ := 1800

-- Define the sequence of operations on the price
def price_after_operations (original_price : ℝ) : ℝ :=
  original_price * 0.90 * 1.05 * 1.12 * 0.85

-- Define the final selling price with profit
def final_price (original_price : ℝ) : ℝ :=
  price_after_operations original_price * 1.20

-- Theorem stating the relationship between original price and selling price
theorem original_price_calculation :
  final_price P = selling_price :=
sorry

end original_price_calculation_l3753_375321


namespace quadratic_root_proof_l3753_375389

theorem quadratic_root_proof : ∃ x : ℝ, x^2 - 4*x*Real.sqrt 2 + 8 = 0 ∧ x = 2*Real.sqrt 2 := by
  sorry

end quadratic_root_proof_l3753_375389


namespace rainfall_ratio_l3753_375381

/-- Given the total rainfall over two weeks and the rainfall in the second week,
    prove the ratio of rainfall in the second week to the first week. -/
theorem rainfall_ratio (total : ℝ) (second_week : ℝ) 
    (h1 : total = 35)
    (h2 : second_week = 21) :
    second_week / (total - second_week) = 3 / 2 := by
  sorry

end rainfall_ratio_l3753_375381


namespace triangle_angle_difference_l3753_375366

theorem triangle_angle_difference (A B C : ℝ) : 
  A = 24 →
  B = 5 * A →
  A + B + C = 180 →
  C - A = 12 :=
by sorry

end triangle_angle_difference_l3753_375366


namespace locus_of_midpoints_l3753_375364

/-- Given a circle with center O and radius R, and a segment of length a,
    the locus of midpoints of all chords of length a is a circle concentric
    to the original circle with radius √(R² - a²/4). -/
theorem locus_of_midpoints (O : ℝ × ℝ) (R a : ℝ) (h1 : R > 0) (h2 : 0 < a ∧ a < 2*R) :
  ∃ (C : Set (ℝ × ℝ)),
    C = {P | ∃ (A B : ℝ × ℝ),
      (A.1 - O.1)^2 + (A.2 - O.2)^2 = R^2 ∧
      (B.1 - O.1)^2 + (B.2 - O.2)^2 = R^2 ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = a^2 ∧
      P = ((A.1 + B.1)/2, (A.2 + B.2)/2)} ∧
    C = {P | (P.1 - O.1)^2 + (P.2 - O.2)^2 = R^2 - a^2/4} :=
by
  sorry

end locus_of_midpoints_l3753_375364


namespace alcohol_mixture_problem_l3753_375342

theorem alcohol_mixture_problem (A W : ℝ) :
  A / W = 2 / 5 →
  A / (W + 10) = 2 / 7 →
  A = 10 := by
sorry

end alcohol_mixture_problem_l3753_375342


namespace meat_calculation_l3753_375395

/-- Given an initial amount of meat, calculate the remaining amount after using some for meatballs and spring rolls. -/
def remaining_meat (initial : ℝ) (meatball_fraction : ℝ) (spring_roll_amount : ℝ) : ℝ :=
  initial - (initial * meatball_fraction) - spring_roll_amount

/-- Theorem stating that given 20 kg of meat, using 1/4 for meatballs and 3 kg for spring rolls leaves 12 kg. -/
theorem meat_calculation :
  remaining_meat 20 (1/4) 3 = 12 := by
  sorry

end meat_calculation_l3753_375395


namespace birds_theorem_l3753_375348

def birds_problem (grey_birds : ℕ) (white_birds : ℕ) : Prop :=
  white_birds = grey_birds + 6 ∧
  grey_birds = 40 ∧
  (grey_birds / 2 + white_birds = 66)

theorem birds_theorem :
  ∃ (grey_birds white_birds : ℕ), birds_problem grey_birds white_birds :=
sorry

end birds_theorem_l3753_375348


namespace line_tangent_to_curve_l3753_375311

/-- The line y = x + b is tangent to the curve x = √(1 - y²) if and only if b = -√2 -/
theorem line_tangent_to_curve (b : ℝ) : 
  (∀ x y : ℝ, y = x + b ∧ x = Real.sqrt (1 - y^2) → 
    (∃! p : ℝ × ℝ, p.1 = Real.sqrt (1 - p.2^2) ∧ p.2 = p.1 + b)) ↔ 
  b = -Real.sqrt 2 :=
sorry

end line_tangent_to_curve_l3753_375311


namespace whole_number_between_bounds_l3753_375302

theorem whole_number_between_bounds (M : ℤ) : 9 < (M : ℚ) / 4 ∧ (M : ℚ) / 4 < 9.5 → M = 37 := by
  sorry

end whole_number_between_bounds_l3753_375302


namespace abs_neg_reciprocal_of_two_l3753_375303

theorem abs_neg_reciprocal_of_two : |-(1 / 2)| = |1 / 2| := by sorry

end abs_neg_reciprocal_of_two_l3753_375303


namespace salary_increase_percentage_l3753_375380

def old_salary : ℝ := 10000
def new_salary : ℝ := 10200

theorem salary_increase_percentage :
  (new_salary - old_salary) / old_salary * 100 = 2 := by
  sorry

end salary_increase_percentage_l3753_375380


namespace doll_count_difference_l3753_375304

/-- The number of dolls Geraldine has -/
def geraldine_dolls : ℝ := 2186.25

/-- The number of dolls Jazmin has -/
def jazmin_dolls : ℝ := 1209.73

/-- The number of dolls Felicia has -/
def felicia_dolls : ℝ := 1530.48

/-- The difference between Geraldine's dolls and the sum of Jazmin's and Felicia's dolls -/
def doll_difference : ℝ := geraldine_dolls - (jazmin_dolls + felicia_dolls)

theorem doll_count_difference : doll_difference = -553.96 := by
  sorry

end doll_count_difference_l3753_375304


namespace sum_of_square_roots_inequality_l3753_375352

theorem sum_of_square_roots_inequality (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) : 
  Real.sqrt (a / (b + c + d + e)) + 
  Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + 
  Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) > 2 := by
  sorry

end sum_of_square_roots_inequality_l3753_375352


namespace polynomial_coefficient_properties_l3753_375368

theorem polynomial_coefficient_properties (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₁ + a₂ + a₃ + a₄ = 1) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 243) ∧
  (a₁ + a₃ + a₅ = 122) ∧
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243) := by
sorry

end polynomial_coefficient_properties_l3753_375368


namespace tens_digit_of_9_power_2023_l3753_375343

theorem tens_digit_of_9_power_2023 (h1 : 9^10 % 50 = 1) (h2 : 9^3 % 50 = 29) :
  (9^2023 / 10) % 10 = 2 := by
  sorry

end tens_digit_of_9_power_2023_l3753_375343


namespace factorial_divisibility_l3753_375329

theorem factorial_divisibility (m n : ℕ) : 
  (m.factorial * n.factorial * (m + n).factorial) ∣ ((2 * m).factorial * (2 * n).factorial) := by
  sorry

end factorial_divisibility_l3753_375329


namespace arithmetic_sequence_sum_l3753_375324

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_345 : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
sorry

end arithmetic_sequence_sum_l3753_375324


namespace supermarket_spending_l3753_375374

theorem supermarket_spending (total : ℚ) : 
  (1/5 : ℚ) * total + (1/3 : ℚ) * total + (1/10 : ℚ) * total + 11 = total → 
  total = 30 :=
by sorry

end supermarket_spending_l3753_375374


namespace no_natural_solution_l3753_375316

theorem no_natural_solution : ∀ x : ℕ, 19 * x^2 + 97 * x ≠ 1997 := by
  sorry

end no_natural_solution_l3753_375316


namespace presidency_meeting_arrangements_l3753_375323

/-- The number of schools -/
def num_schools : ℕ := 4

/-- The number of members from each school -/
def members_per_school : ℕ := 6

/-- The number of representatives from the host school -/
def host_representatives : ℕ := 3

/-- The number of representatives from each non-host school -/
def other_representatives : ℕ := 1

/-- The total number of members in the club -/
def total_members : ℕ := num_schools * members_per_school

/-- The number of ways to arrange the presidency meeting -/
def meeting_arrangements : ℕ := 
  num_schools * (members_per_school.choose host_representatives) * 
  (members_per_school.choose other_representatives)^(num_schools - 1)

theorem presidency_meeting_arrangements : 
  meeting_arrangements = 17280 :=
sorry

end presidency_meeting_arrangements_l3753_375323


namespace symmetric_difference_of_A_and_B_l3753_375306

-- Define the set difference operation
def setDifference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

-- Define the symmetric difference operation
def symmetricDifference (M N : Set ℝ) : Set ℝ := (setDifference M N) ∪ (setDifference N M)

-- Define sets A and B
def A : Set ℝ := {x | x ≥ -9/4}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {x | x < -9/4 ∨ x ≥ 0} :=
by sorry

end symmetric_difference_of_A_and_B_l3753_375306


namespace tangent_line_at_one_max_value_on_interval_a_range_for_nonnegative_l3753_375387

-- Define the function f(x) = x³ - ax²
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

theorem tangent_line_at_one (a : ℝ) (h : f_derivative a 1 = 3) :
  ∃ m b : ℝ, m = 3 ∧ b = -2 ∧ ∀ x : ℝ, (f a x - (f a 1)) = m * (x - 1) := by sorry

theorem max_value_on_interval :
  ∃ M : ℝ, M = 8 ∧ ∀ x : ℝ, x ∈ Set.Icc 0 2 → f 0 x ≤ M := by sorry

theorem a_range_for_nonnegative (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x + x ≥ 0) ↔ a ≤ 2 := by sorry

end tangent_line_at_one_max_value_on_interval_a_range_for_nonnegative_l3753_375387


namespace smallest_marble_count_thirty_is_smallest_l3753_375326

theorem smallest_marble_count : ℕ → Prop :=
  fun n => n > 0 ∧ 
    (∃ w g r b : ℕ, 
      w + g + r + b = n ∧ 
      w = n / 6 ∧ 
      g = n / 5 ∧ 
      r + b = 19 * n / 30) →
  n ≥ 30

theorem thirty_is_smallest : smallest_marble_count 30 :=
sorry

end smallest_marble_count_thirty_is_smallest_l3753_375326


namespace unfair_coin_probability_l3753_375397

theorem unfair_coin_probability (p : ℝ) : 
  0 < p ∧ p < 1 →
  (6 : ℝ) * p^2 * (1 - p)^2 = (4 : ℝ) * p^3 * (1 - p) →
  p = 3/5 := by
sorry

end unfair_coin_probability_l3753_375397


namespace consecutive_integers_average_l3753_375315

/-- Given six positive consecutive integers starting with c, their average d,
    prove that the average of 7 consecutive integers starting with d is c + 5.5 -/
theorem consecutive_integers_average (c : ℤ) (d : ℚ) : 
  (c > 0) →
  (d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5)) / 6) →
  ((d + (d+1) + (d+2) + (d+3) + (d+4) + (d+5) + (d+6)) / 7 = c + 5.5) :=
by sorry

end consecutive_integers_average_l3753_375315


namespace star_op_value_l3753_375353

-- Define the * operation for non-zero integers
def star_op (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

-- Theorem statement
theorem star_op_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a + b = 15 → a * b = 56 → star_op a b = 15 / 56 := by
  sorry

end star_op_value_l3753_375353


namespace all_error_types_cause_random_errors_at_least_three_random_error_causes_l3753_375393

-- Define the types of errors
inductive ErrorType
  | ApproximationError
  | OmittedVariableError
  | ObservationError

-- Define a predicate for causes of random errors
def is_random_error_cause (error_type : ErrorType) : Prop :=
  match error_type with
  | ErrorType.ApproximationError => true
  | ErrorType.OmittedVariableError => true
  | ErrorType.ObservationError => true

-- Theorem stating that all three error types are causes of random errors
theorem all_error_types_cause_random_errors :
  (∀ (error_type : ErrorType), is_random_error_cause error_type) :=
by
  sorry

-- Theorem stating that there are at least three distinct causes of random errors
theorem at_least_three_random_error_causes :
  ∃ (e1 e2 e3 : ErrorType),
    e1 ≠ e2 ∧ e1 ≠ e3 ∧ e2 ≠ e3 ∧
    is_random_error_cause e1 ∧
    is_random_error_cause e2 ∧
    is_random_error_cause e3 :=
by
  sorry

end all_error_types_cause_random_errors_at_least_three_random_error_causes_l3753_375393


namespace common_ratio_satisfies_cubic_cubic_solution_approx_l3753_375391

/-- A geometric progression with positive terms where each term is the sum of the next three terms -/
structure GeometricProgressionWithSumProperty where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

/-- The common ratio of a geometric progression with the sum property satisfies a cubic equation -/
theorem common_ratio_satisfies_cubic (gp : GeometricProgressionWithSumProperty) :
  gp.r^3 + gp.r^2 + gp.r - 1 = 0 :=
sorry

/-- The positive real solution to the cubic equation x³ + x² + x - 1 = 0 is approximately 0.5437 -/
theorem cubic_solution_approx :
  ∃ x : ℝ, x > 0 ∧ x^3 + x^2 + x - 1 = 0 ∧ abs (x - 0.5437) < 0.0001 :=
sorry

end common_ratio_satisfies_cubic_cubic_solution_approx_l3753_375391


namespace amys_garden_space_l3753_375351

/-- Calculates the total square feet of growing space for Amy's garden beds -/
theorem amys_garden_space : 
  let small_bed_length : ℝ := 3
  let small_bed_width : ℝ := 3
  let large_bed_length : ℝ := 4
  let large_bed_width : ℝ := 3
  let num_small_beds : ℕ := 2
  let num_large_beds : ℕ := 2
  
  let small_bed_area := small_bed_length * small_bed_width
  let large_bed_area := large_bed_length * large_bed_width
  let total_area := (num_small_beds : ℝ) * small_bed_area + (num_large_beds : ℝ) * large_bed_area
  
  total_area = 42 := by sorry

end amys_garden_space_l3753_375351


namespace mitch_earnings_l3753_375363

/-- Represents Mitch's work schedule and earnings --/
structure MitchSchedule where
  weekday_hours : ℕ
  weekend_hours : ℕ
  weekday_rate : ℕ
  weekend_rate : ℕ

/-- Calculates Mitch's weekly earnings --/
def weekly_earnings (schedule : MitchSchedule) : ℕ :=
  (schedule.weekday_hours * 5 * schedule.weekday_rate) +
  (schedule.weekend_hours * 2 * schedule.weekend_rate)

/-- Theorem stating Mitch's weekly earnings --/
theorem mitch_earnings : 
  let schedule := MitchSchedule.mk 5 3 3 6
  weekly_earnings schedule = 111 := by
  sorry


end mitch_earnings_l3753_375363


namespace pages_read_on_fourth_day_l3753_375339

/-- Given a book with 354 pages, if a person reads 63 pages on day one,
    twice that amount on day two, and 10 more pages than day two on day three,
    then the number of pages read on day four is 29. -/
theorem pages_read_on_fourth_day
  (total_pages : ℕ)
  (pages_day_one : ℕ)
  (h1 : total_pages = 354)
  (h2 : pages_day_one = 63)
  : total_pages - pages_day_one - (2 * pages_day_one) - (2 * pages_day_one + 10) = 29 := by
  sorry

#check pages_read_on_fourth_day

end pages_read_on_fourth_day_l3753_375339


namespace shirt_fabric_sum_l3753_375328

theorem shirt_fabric_sum (a : ℝ) (r : ℝ) (h1 : a = 2011) (h2 : r = 4/5) (h3 : r < 1) :
  a / (1 - r) = 10055 := by
  sorry

end shirt_fabric_sum_l3753_375328


namespace unique_solution_power_equation_l3753_375367

theorem unique_solution_power_equation :
  ∃! (n m : ℕ), n > 0 ∧ m > 0 ∧ n^5 + n^4 = 7^m - 1 :=
by
  sorry

end unique_solution_power_equation_l3753_375367


namespace unique_solution_at_two_no_unique_solution_above_two_no_unique_solution_below_two_max_a_for_unique_solution_l3753_375384

/-- The system of equations has a unique solution when a = 2 -/
theorem unique_solution_at_two (x y a : ℝ) : 
  (y = 1 - Real.sqrt x ∧ 
   a - 2 * (a - y)^2 = Real.sqrt x ∧ 
   ∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ a - 2 * (a - y)^2 = Real.sqrt x) 
  → a = 2 := by
  sorry

/-- For any a > 2, the system of equations does not have a unique solution -/
theorem no_unique_solution_above_two (a : ℝ) :
  a > 2 → ¬(∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ a - 2 * (a - y)^2 = Real.sqrt x) := by
  sorry

/-- For any 0 ≤ a < 2, the system of equations does not have a unique solution -/
theorem no_unique_solution_below_two (a : ℝ) :
  0 ≤ a ∧ a < 2 → ¬(∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ a - 2 * (a - y)^2 = Real.sqrt x) := by
  sorry

/-- The maximum value of a for which the system has a unique solution is 2 -/
theorem max_a_for_unique_solution :
  ∃ (a : ℝ), (∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ a - 2 * (a - y)^2 = Real.sqrt x) ∧
  ∀ (b : ℝ), (∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ b - 2 * (b - y)^2 = Real.sqrt x) → b ≤ a := by
  sorry

end unique_solution_at_two_no_unique_solution_above_two_no_unique_solution_below_two_max_a_for_unique_solution_l3753_375384


namespace quadratic_vertex_l3753_375373

/-- The quadratic function f(x) = 2(x-3)^2 + 1 -/
def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 3

/-- The y-coordinate of the vertex -/
def vertex_y : ℝ := 1

/-- Theorem: The vertex of the quadratic function f(x) = 2(x-3)^2 + 1 is at (3,1) -/
theorem quadratic_vertex : 
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y := by
  sorry

end quadratic_vertex_l3753_375373


namespace exists_special_function_l3753_375335

theorem exists_special_function : ∃ (f : ℝ → ℝ),
  (∀ (b : ℝ), ∃! (x : ℝ), f x = b) ∧
  (∀ (a b : ℝ), a > 0 → ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a * x₁ + b ∧ f x₂ = a * x₂ + b) :=
by sorry

end exists_special_function_l3753_375335


namespace parameter_range_l3753_375317

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (a * x / 3) else 3 * Real.log x / x

/-- The maximum value of f(x) on [-3, 3] is 3/e -/
axiom max_value (a : ℝ) : ∀ x ∈ Set.Icc (-3) 3, f a x ≤ 3 / Real.exp 1

/-- The range of parameter a is [1 - ln(3), +∞) -/
theorem parameter_range :
  {a : ℝ | ∀ x ∈ Set.Icc (-3) 3, f a x ≤ 3 / Real.exp 1} = Set.Ici (1 - Real.log 3) := by
  sorry

end parameter_range_l3753_375317


namespace least_N_for_probability_condition_l3753_375388

def P (N : ℕ) : ℚ :=
  (⌊(2 * N : ℚ) / 5⌋ + (N - ⌈(3 * N : ℚ) / 5⌉)) / (N + 1 : ℚ)

theorem least_N_for_probability_condition :
  (∀ k : ℕ, k % 5 = 0 ∧ 0 < k ∧ k < 480 → P k ≥ 321/400) ∧
  P 480 < 321/400 := by
  sorry

end least_N_for_probability_condition_l3753_375388


namespace roof_area_theorem_l3753_375362

/-- Represents the dimensions and area of a rectangular roof. -/
structure RectangularRoof where
  width : ℚ
  length : ℚ
  area : ℚ

/-- Calculates the area of a rectangular roof given its width and length. -/
def calculateArea (w : ℚ) (l : ℚ) : ℚ := w * l

/-- Theorem: The area of a rectangular roof with specific proportions is 455 1/9 square feet. -/
theorem roof_area_theorem (roof : RectangularRoof) : 
  roof.length = 3 * roof.width → 
  roof.length - roof.width = 32 → 
  roof.area = calculateArea roof.width roof.length → 
  roof.area = 455 + 1/9 := by
  sorry

end roof_area_theorem_l3753_375362


namespace replaced_person_weight_l3753_375354

/-- Given a group of 10 persons, if replacing one person with a new person
    weighing 100 kg increases the average weight by 3.5 kg,
    then the weight of the replaced person is 65 kg. -/
theorem replaced_person_weight
  (n : ℕ) (initial_average : ℝ) (new_person_weight : ℝ) (average_increase : ℝ) :
  n = 10 →
  new_person_weight = 100 →
  average_increase = 3.5 →
  initial_average + average_increase = (n * initial_average - replaced_weight + new_person_weight) / n →
  replaced_weight = 65 :=
by sorry

end replaced_person_weight_l3753_375354


namespace largest_integer_below_sqrt_two_l3753_375355

theorem largest_integer_below_sqrt_two :
  ∀ n : ℕ, n > 0 ∧ n < Real.sqrt 2 → n = 1 :=
by sorry

end largest_integer_below_sqrt_two_l3753_375355


namespace equilateral_triangle_side_length_l3753_375308

/-- An equilateral triangle with perimeter 15 meters has sides of length 5 meters. -/
theorem equilateral_triangle_side_length (triangle : Set ℝ) (perimeter : ℝ) : 
  perimeter = 15 → 
  (∃ side : ℝ, side > 0 ∧ 
    (∀ s : ℝ, s ∈ triangle → s = side) ∧ 
    3 * side = perimeter) → 
  (∃ side : ℝ, side = 5 ∧ 
    (∀ s : ℝ, s ∈ triangle → s = side)) :=
by sorry

end equilateral_triangle_side_length_l3753_375308


namespace triangle_altitude_l3753_375309

/-- Given a triangle with area 800 square feet and base 40 feet, its altitude is 40 feet. -/
theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) : 
  area = 800 → base = 40 → area = (1/2) * base * altitude → altitude = 40 := by
  sorry

end triangle_altitude_l3753_375309


namespace alcohol_dilution_l3753_375379

/-- Proves that adding 16 liters of water to 24 liters of a 90% alcohol solution
    results in a new mixture with 54% alcohol. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 24 →
  initial_concentration = 0.90 →
  added_water = 16 →
  final_concentration = 0.54 →
  initial_volume * initial_concentration = 
    (initial_volume + added_water) * final_concentration :=
by
  sorry

#check alcohol_dilution

end alcohol_dilution_l3753_375379


namespace line_intersection_with_y_axis_l3753_375385

/-- Given a line passing through points (3, 10) and (-7, -6), 
    prove that its intersection with the y-axis is the point (0, 5.2) -/
theorem line_intersection_with_y_axis :
  let p₁ : ℝ × ℝ := (3, 10)
  let p₂ : ℝ × ℝ := (-7, -6)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  let b : ℝ := p₁.2 - m * p₁.1
  let line (x : ℝ) : ℝ := m * x + b
  let y_intercept : ℝ := line 0
  (0, y_intercept) = (0, 5.2) := by sorry

end line_intersection_with_y_axis_l3753_375385


namespace x_equation_implies_polynomial_value_l3753_375341

theorem x_equation_implies_polynomial_value (x : ℝ) (h : x + 1/x = Real.sqrt 7) :
  x^12 - 8*x^8 + x^4 = 1365 := by
sorry

end x_equation_implies_polynomial_value_l3753_375341


namespace union_M_N_when_a_9_M_superset_N_iff_a_range_l3753_375322

-- Define the sets M and N
def M : Set ℝ := {x | (x + 5) / (x - 8) ≥ 0}
def N (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}

-- Theorem for part 1
theorem union_M_N_when_a_9 :
  M ∪ N 9 = {x : ℝ | x ≤ -5 ∨ x ≥ 8} := by sorry

-- Theorem for part 2
theorem M_superset_N_iff_a_range (a : ℝ) :
  M ⊇ N a ↔ a ≤ -6 ∨ a > 9 := by sorry

end union_M_N_when_a_9_M_superset_N_iff_a_range_l3753_375322


namespace diamond_operation_l3753_375327

def diamond (a b : ℤ) : ℤ := 12 * a - 10 * b

theorem diamond_operation : diamond (diamond (diamond (diamond 20 22) 22) 22) 22 = 20 := by
  sorry

end diamond_operation_l3753_375327


namespace like_terms_exponent_product_l3753_375325

theorem like_terms_exponent_product (a b : ℝ) (m n : ℤ) : 
  (∃ (k : ℝ), k ≠ 0 ∧ 3 * a^m * b^2 = k * (-a^2 * b^(n+3))) → m * n = -2 :=
by sorry

end like_terms_exponent_product_l3753_375325


namespace jane_change_l3753_375337

-- Define the cost of the apple
def apple_cost : ℚ := 75/100

-- Define the amount Jane pays
def amount_paid : ℚ := 5

-- Define the change function
def change (cost paid : ℚ) : ℚ := paid - cost

-- Theorem statement
theorem jane_change : change apple_cost amount_paid = 425/100 := by
  sorry

end jane_change_l3753_375337


namespace least_partition_size_formula_l3753_375349

/-- A move that can be applied to a permutation -/
inductive Move
  | MedianFirst : Move  -- If a is the median, replace a,b,c with b,c,a
  | MedianLast : Move   -- If c is the median, replace a,b,c with c,a,b

/-- A permutation of 1, 2, 3, ..., n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The number of inversions in a permutation -/
def inversions (n : ℕ) (σ : Permutation n) : ℕ :=
  sorry

/-- Whether two permutations are obtainable from each other by a sequence of moves -/
def obtainable (n : ℕ) (σ τ : Permutation n) : Prop :=
  sorry

/-- The least number of sets in a partition of all n! permutations -/
def least_partition_size (n : ℕ) : ℕ :=
  sorry

/-- The main theorem: the least number of sets in the partition is n^2 - 3n + 4 -/
theorem least_partition_size_formula (n : ℕ) :
  least_partition_size n = n^2 - 3*n + 4 :=
sorry

end least_partition_size_formula_l3753_375349


namespace bruce_payment_l3753_375314

/-- The total amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1165 to the shopkeeper -/
theorem bruce_payment : total_amount 8 70 11 55 = 1165 := by
  sorry

end bruce_payment_l3753_375314


namespace fifth_number_eighth_row_l3753_375300

/-- Represents the end number of the n-th row in the table -/
def end_of_row (n : ℕ) : ℕ := n * n

/-- Represents the first number in the n-th row -/
def start_of_row (n : ℕ) : ℕ := end_of_row (n - 1) + 1

/-- The theorem stating that the 5th number from the left in the 8th row is 54 -/
theorem fifth_number_eighth_row : start_of_row 8 + 4 = 54 := by
  sorry

end fifth_number_eighth_row_l3753_375300


namespace cistern_leak_emptying_time_l3753_375372

/-- Given a cistern that normally fills in 8 hours, but takes 10 hours to fill with a leak,
    prove that it takes 40 hours for a full cistern to empty due to the leak. -/
theorem cistern_leak_emptying_time (normal_fill_time leak_fill_time : ℝ) 
    (h1 : normal_fill_time = 8)
    (h2 : leak_fill_time = 10) : 
  let normal_fill_rate := 1 / normal_fill_time
  let leak_rate := normal_fill_rate - (1 / leak_fill_time)
  (1 / leak_rate) = 40 := by sorry

end cistern_leak_emptying_time_l3753_375372


namespace milton_more_accelerated_l3753_375345

/-- Represents the percentage of at-home workforce for a city at a given year --/
structure WorkforceData :=
  (year2000 : ℝ)
  (year2010 : ℝ)
  (year2020 : ℝ)
  (year2030 : ℝ)

/-- Determines if a city's workforce growth is accelerating --/
def isAccelerating (data : WorkforceData) : Prop :=
  let diff2010 := data.year2010 - data.year2000
  let diff2020 := data.year2020 - data.year2010
  let diff2030 := data.year2030 - data.year2020
  diff2030 > diff2020 ∧ diff2020 > diff2010

/-- Milton City's workforce data --/
def miltonCity : WorkforceData :=
  { year2000 := 3
  , year2010 := 9
  , year2020 := 18
  , year2030 := 35 }

/-- Rivertown's workforce data --/
def rivertown : WorkforceData :=
  { year2000 := 4
  , year2010 := 7
  , year2020 := 13
  , year2030 := 20 }

/-- Theorem stating that Milton City's growth is more accelerated than Rivertown's --/
theorem milton_more_accelerated :
  isAccelerating miltonCity ∧ ¬isAccelerating rivertown :=
sorry

end milton_more_accelerated_l3753_375345


namespace hyeji_total_water_intake_l3753_375312

-- Define the conversion rate from liters to milliliters
def liters_to_ml (liters : ℝ) : ℝ := liters * 1000

-- Define Hyeji's daily water intake in liters
def daily_intake : ℝ := 2

-- Define the additional amount Hyeji drank in milliliters
def additional_intake : ℝ := 460

-- Theorem to prove
theorem hyeji_total_water_intake :
  liters_to_ml daily_intake + additional_intake = 2460 := by
  sorry

end hyeji_total_water_intake_l3753_375312


namespace custom_op_theorem_l3753_375356

/-- Definition of the custom operation ⊕ -/
def custom_op (m n : ℝ) : ℝ := m * n * (m - n)

/-- Theorem stating that (a + b) ⊕ a = a^2 * b + a * b^2 -/
theorem custom_op_theorem (a b : ℝ) : custom_op (a + b) a = a^2 * b + a * b^2 := by
  sorry

end custom_op_theorem_l3753_375356


namespace parallel_vectors_x_value_l3753_375382

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (-1, x)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value :
  parallel vector_a (vector_b x) → x = -2 := by
  sorry

end parallel_vectors_x_value_l3753_375382


namespace arc_length_of_inscribed_pentagon_l3753_375347

-- Define the circle radius
def circle_radius : ℝ := 5

-- Define the number of sides in a regular pentagon
def pentagon_sides : ℕ := 5

-- Theorem statement
theorem arc_length_of_inscribed_pentagon (π : ℝ) :
  let circumference := 2 * π * circle_radius
  let arc_length := circumference / pentagon_sides
  arc_length = 2 * π := by sorry

end arc_length_of_inscribed_pentagon_l3753_375347


namespace complex_number_not_in_fourth_quadrant_l3753_375346

theorem complex_number_not_in_fourth_quadrant :
  ∀ (m : ℝ) (z : ℂ), (1 - Complex.I) * z = m + Complex.I →
    ¬(Complex.re z > 0 ∧ Complex.im z < 0) := by
  sorry

end complex_number_not_in_fourth_quadrant_l3753_375346


namespace work_completion_time_l3753_375344

/-- The time taken to complete a work given the rates of two workers and their working pattern -/
theorem work_completion_time
  (rate_A rate_B : ℝ)  -- Rates at which A and B can complete the work alone
  (days_together : ℝ)  -- Number of days A and B work together
  (h_rate_A : rate_A = 1 / 15)  -- A can complete the work in 15 days
  (h_rate_B : rate_B = 1 / 10)  -- B can complete the work in 10 days
  (h_days_together : days_together = 2)  -- A and B work together for 2 days
  : ∃ (total_days : ℝ), total_days = 12 ∧ 
    rate_A * (total_days - days_together) + (rate_A + rate_B) * days_together = 1 :=
by sorry


end work_completion_time_l3753_375344


namespace ferris_wheel_capacity_l3753_375350

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 14

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := 6

/-- The total number of people who can ride the Ferris wheel at the same time -/
def total_people : ℕ := num_seats * people_per_seat

theorem ferris_wheel_capacity : total_people = 84 := by
  sorry

end ferris_wheel_capacity_l3753_375350


namespace next_shared_meeting_proof_l3753_375333

/-- The number of days between drama club meetings -/
def drama_interval : ℕ := 3

/-- The number of days between choir meetings -/
def choir_interval : ℕ := 5

/-- The number of days until both groups meet again -/
def next_shared_meeting : ℕ := 30

theorem next_shared_meeting_proof :
  ∃ (n : ℕ), n > 0 ∧ n * drama_interval = next_shared_meeting ∧ n * choir_interval = next_shared_meeting :=
sorry

end next_shared_meeting_proof_l3753_375333
