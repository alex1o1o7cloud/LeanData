import Mathlib

namespace NUMINAMATH_CALUDE_factorial_ratio_l529_52926

theorem factorial_ratio : (Nat.factorial 9) / (Nat.factorial 8) = 9 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l529_52926


namespace NUMINAMATH_CALUDE_inequality_solution_set_l529_52925

theorem inequality_solution_set (x : ℝ) : x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l529_52925


namespace NUMINAMATH_CALUDE_sheets_colored_l529_52956

/-- Given 2450 sheets of paper evenly split into 5 binders,
    prove that coloring one-half of the sheets in one binder uses 245 sheets. -/
theorem sheets_colored (total_sheets : ℕ) (num_binders : ℕ) (sheets_per_binder : ℕ) :
  total_sheets = 2450 →
  num_binders = 5 →
  total_sheets = num_binders * sheets_per_binder →
  sheets_per_binder / 2 = 245 := by
  sorry

#check sheets_colored

end NUMINAMATH_CALUDE_sheets_colored_l529_52956


namespace NUMINAMATH_CALUDE_square_sum_difference_specific_square_sum_difference_l529_52904

theorem square_sum_difference (n : ℕ) : 
  (2*n+1)^2 - (2*n-1)^2 + (2*n-3)^2 - (2*n-5)^2 + (2*n-7)^2 - (2*n-9)^2 + 
  (2*n-11)^2 - (2*n-13)^2 + (2*n-15)^2 - (2*n-17)^2 + (2*n-19)^2 - (2*n-21)^2 + (2*n-23)^2 = 
  4*n^2 + 1 :=
by sorry

theorem specific_square_sum_difference : 
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 337 :=
by sorry

end NUMINAMATH_CALUDE_square_sum_difference_specific_square_sum_difference_l529_52904


namespace NUMINAMATH_CALUDE_greatest_common_divisor_problem_l529_52999

theorem greatest_common_divisor_problem :
  Nat.gcd 105 (Nat.gcd 1001 (Nat.gcd 2436 (Nat.gcd 10202 49575))) = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_problem_l529_52999


namespace NUMINAMATH_CALUDE_curve_equation_l529_52989

noncomputable def x (t : ℝ) : ℝ := 3 * Real.cos t + 2 * Real.sin t
noncomputable def y (t : ℝ) : ℝ := 5 * Real.sin t

theorem curve_equation (t : ℝ) :
  let a : ℝ := 1/9
  let b : ℝ := -4/15
  let c : ℝ := 19/375
  a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_equation_l529_52989


namespace NUMINAMATH_CALUDE_janice_office_floor_l529_52955

/-- The floor number of Janice's office -/
def office_floor : ℕ := 3

/-- The number of times Janice goes up the stairs per day -/
def up_times : ℕ := 5

/-- The number of times Janice goes down the stairs per day -/
def down_times : ℕ := 3

/-- The total number of flights of stairs Janice walks in a day -/
def total_flights : ℕ := 24

theorem janice_office_floor :
  office_floor * (up_times + down_times) = total_flights :=
sorry

end NUMINAMATH_CALUDE_janice_office_floor_l529_52955


namespace NUMINAMATH_CALUDE_min_value_of_f_min_value_of_f_equality_inequality_holds_iff_m_in_range_l529_52997

-- Part 1
theorem min_value_of_f (a : ℝ) (ha : a > 0) :
  a^2 + 2/a ≥ 3 :=
sorry

theorem min_value_of_f_equality (a : ℝ) (ha : a > 0) :
  a^2 + 2/a = 3 ↔ a = 1 :=
sorry

-- Part 2
def m_range (m : ℝ) : Prop :=
  m ≤ -3 ∨ m ≥ -1

theorem inequality_holds_iff_m_in_range (m : ℝ) :
  (∀ a : ℝ, a > 0 → a^3 + 2 ≥ 3*a*(|m - 1| - |2*m + 3|)) ↔ m_range m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_min_value_of_f_equality_inequality_holds_iff_m_in_range_l529_52997


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l529_52974

/-- The function f(x) = x³ - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The function g(x) = x² + a -/
def g (a x : ℝ) : ℝ := x^2 + a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The derivative of g(x) -/
def g' (x : ℝ) : ℝ := 2 * x

/-- The tangent line of f at x₁ -/
def tangent_f (x₁ x : ℝ) : ℝ := f' x₁ * (x - x₁) + f x₁

/-- The tangent line of g at x₂ -/
def tangent_g (a x₂ x : ℝ) : ℝ := g' x₂ * (x - x₂) + g a x₂

theorem tangent_line_intersection (a : ℝ) :
  (∃ x₁ x₂ : ℝ, ∀ x : ℝ, tangent_f x₁ x = tangent_g a x₂ x) →
  (x₁ = -1 → a = 3) ∧ (a ≥ -1) := by sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l529_52974


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l529_52978

theorem polynomial_product_sum (g h : ℚ) : 
  (∀ d : ℚ, (5 * d^2 - 2 * d + g) * (4 * d^2 + h * d - 6) = 
             20 * d^4 - 18 * d^3 + 7 * d^2 + 10 * d - 18) →
  g + h = 7/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l529_52978


namespace NUMINAMATH_CALUDE_expression_evaluation_l529_52950

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = b - 4)
  (h2 : b = a + 4)
  (h3 : a = 3)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 10) / (c + 7) = 117 / 40 := by
  sorry

#check expression_evaluation

end NUMINAMATH_CALUDE_expression_evaluation_l529_52950


namespace NUMINAMATH_CALUDE_larger_integer_proof_l529_52967

theorem larger_integer_proof (A B : ℤ) (h1 : A + B = 2010) (h2 : Nat.lcm A.natAbs B.natAbs = 14807) : 
  max A B = 1139 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l529_52967


namespace NUMINAMATH_CALUDE_f_of_2_equals_6_l529_52984

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 - 3

-- Theorem statement
theorem f_of_2_equals_6 : f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_6_l529_52984


namespace NUMINAMATH_CALUDE_project_hours_total_l529_52964

/-- Represents the hours charged by Kate, Pat, and Mark to a project -/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ

/-- Defines the conditions of the project hours -/
def validProjectHours (h : ProjectHours) : Prop :=
  h.pat = 2 * h.kate ∧
  h.pat = h.mark / 3 ∧
  h.mark = h.kate + 110

theorem project_hours_total (h : ProjectHours) (hValid : validProjectHours h) :
  h.kate + h.pat + h.mark = 198 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_total_l529_52964


namespace NUMINAMATH_CALUDE_jump_rope_problem_l529_52944

theorem jump_rope_problem (a : ℕ) : 
  let counts : List ℕ := [180, 182, 173, 175, a, 178, 176]
  (counts.sum / counts.length : ℚ) = 178 →
  a = 182 := by
sorry

end NUMINAMATH_CALUDE_jump_rope_problem_l529_52944


namespace NUMINAMATH_CALUDE_complex_equation_real_part_condition_l529_52934

theorem complex_equation_real_part_condition (z : ℂ) (a b : ℝ) : 
  z * (z + 2*I) * (z + 4*I) = 1001*I → 
  z = a + b*I → 
  a > 0 → 
  b > 0 → 
  a * (a^2 - b^2 - 6*b - 8) = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_real_part_condition_l529_52934


namespace NUMINAMATH_CALUDE_complex_sum_real_l529_52971

theorem complex_sum_real (a : ℝ) : 
  let z₁ : ℂ := (16 / (a + 5)) - (10 - a^2) * I
  let z₂ : ℂ := (2 / (1 - a)) + (2*a - 5) * I
  (z₁ + z₂).im = 0 → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_real_l529_52971


namespace NUMINAMATH_CALUDE_selection_methods_count_l529_52915

/-- Represents the number of workers in each category -/
structure WorkerCounts where
  total : Nat
  pliers_only : Nat
  car_only : Nat
  both : Nat

/-- Represents the number of workers to be selected for each job -/
structure SelectionCounts where
  pliers : Nat
  car : Nat

/-- Calculates the number of ways to select workers for pliers and car jobs -/
def count_selection_methods (w : WorkerCounts) (s : SelectionCounts) : Nat :=
  sorry

/-- The main theorem stating the correct number of selection methods -/
theorem selection_methods_count :
  let w : WorkerCounts := ⟨11, 5, 4, 2⟩
  let s : SelectionCounts := ⟨4, 4⟩
  count_selection_methods w s = 185 := by sorry

end NUMINAMATH_CALUDE_selection_methods_count_l529_52915


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l529_52949

def A : Set ℝ := {x | x^2 - 2*x > 0}

def B : Set ℝ := {x | (x+1)/(x-1) ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l529_52949


namespace NUMINAMATH_CALUDE_jerry_george_sticker_ratio_l529_52928

/-- The ratio of Jerry's stickers to George's stickers -/
def stickerRatio (jerryStickers georgeStickers : ℕ) : ℚ :=
  jerryStickers / georgeStickers

/-- Proof that the ratio of Jerry's stickers to George's stickers is 3 -/
theorem jerry_george_sticker_ratio :
  let fredStickers : ℕ := 18
  let georgeStickers : ℕ := fredStickers - 6
  let jerryStickers : ℕ := 36
  stickerRatio jerryStickers georgeStickers = 3 := by
sorry

end NUMINAMATH_CALUDE_jerry_george_sticker_ratio_l529_52928


namespace NUMINAMATH_CALUDE_triangle_area_l529_52901

/-- Given a triangle ABC with angles A, B, C forming an arithmetic sequence,
    side b = √3, and f(x) = 2√3 sin²x + 2sin x cos x - √3 reaching its maximum at x = A,
    prove that the area of triangle ABC is (3 + √3) / 4 -/
theorem triangle_area (A B C : Real) (b : Real) (f : Real → Real) :
  (∃ d : Real, B = A - d ∧ C = A + d) →  -- Angles form arithmetic sequence
  b = Real.sqrt 3 →  -- Side b equals √3
  (∀ x, f x = 2 * Real.sqrt 3 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3) →  -- Definition of f
  (∀ x, f x ≤ f A) →  -- f reaches maximum at A
  A + B + C = π →  -- Angle sum in triangle
  (∃ a c : Real, a * Real.sin B = b * Real.sin A ∧ c * Real.sin A = b * Real.sin C) →  -- Sine law
  1 / 2 * b * Real.sin A * Real.sin C / Real.sin B = (3 + Real.sqrt 3) / 4 :=  -- Area formula
by sorry

end NUMINAMATH_CALUDE_triangle_area_l529_52901


namespace NUMINAMATH_CALUDE_equation_solutions_l529_52938

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 2*x - 15 = 0 ↔ x = 5 ∨ x = -3) ∧
  (∀ x : ℝ, (x - 1)^2 = 2*(x - 1) ↔ x = 1 ∨ x = 3) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l529_52938


namespace NUMINAMATH_CALUDE_adam_figurines_l529_52902

/-- The number of figurines that can be made from one block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be made from one block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be made from one block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The number of basswood blocks Adam owns -/
def basswood_blocks : ℕ := 15

/-- The number of butternut wood blocks Adam owns -/
def butternut_blocks : ℕ := 20

/-- The number of Aspen wood blocks Adam owns -/
def aspen_blocks : ℕ := 20

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 
  basswood_blocks * basswood_figurines + 
  butternut_blocks * butternut_figurines + 
  aspen_blocks * aspen_figurines

theorem adam_figurines : total_figurines = 245 := by
  sorry

end NUMINAMATH_CALUDE_adam_figurines_l529_52902


namespace NUMINAMATH_CALUDE_second_gym_signup_fee_covers_four_months_l529_52966

-- Define the given constants
def cheap_monthly_fee : ℤ := 10
def cheap_signup_fee : ℤ := 50
def total_paid_first_year : ℤ := 650
def months_in_year : ℕ := 12

-- Define the relationships
def second_monthly_fee : ℤ := 3 * cheap_monthly_fee

-- State the theorem
theorem second_gym_signup_fee_covers_four_months :
  ∃ (second_signup_fee : ℤ),
    (cheap_monthly_fee * months_in_year + cheap_signup_fee +
     second_monthly_fee * months_in_year + second_signup_fee = total_paid_first_year) ∧
    (second_signup_fee / second_monthly_fee = 4) := by
  sorry

end NUMINAMATH_CALUDE_second_gym_signup_fee_covers_four_months_l529_52966


namespace NUMINAMATH_CALUDE_sine_inequality_solution_l529_52951

theorem sine_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, |a * Real.sin x + b * Real.sin (2 * x)| ≤ 1) ∧ 
  (|a| + |b| ≥ 2 / Real.sqrt 3) →
  ((a = 4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = 4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3))) :=
by sorry

end NUMINAMATH_CALUDE_sine_inequality_solution_l529_52951


namespace NUMINAMATH_CALUDE_square_diff_sqrt_l529_52932

theorem square_diff_sqrt : (Real.sqrt 81 - Real.sqrt 144)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_sqrt_l529_52932


namespace NUMINAMATH_CALUDE_candy_mixture_theorem_l529_52957

/-- Represents the price of the cheaper candy per pound -/
def cheaper_candy_price : ℝ := 2

/-- Represents the price of the more expensive candy per pound -/
def expensive_candy_price : ℝ := 3

/-- Total weight of the mixture in pounds -/
def total_mixture_weight : ℝ := 80

/-- Selling price of the mixture per pound -/
def mixture_selling_price : ℝ := 2.2

/-- Weight of the cheaper candy in the mixture -/
def cheaper_candy_weight : ℝ := 64

/-- Weight of the more expensive candy in the mixture -/
def expensive_candy_weight : ℝ := total_mixture_weight - cheaper_candy_weight

theorem candy_mixture_theorem :
  cheaper_candy_price * cheaper_candy_weight +
  expensive_candy_price * expensive_candy_weight =
  mixture_selling_price * total_mixture_weight :=
by sorry

end NUMINAMATH_CALUDE_candy_mixture_theorem_l529_52957


namespace NUMINAMATH_CALUDE_equation_has_four_solutions_l529_52914

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The equation we're solving -/
def equation (x : ℝ) : Prop :=
  9 * x^2 - 27 * (floor x) + 22 = 0

/-- The theorem stating that the equation has exactly 4 real solutions -/
theorem equation_has_four_solutions :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, equation x ∧
  ∀ y : ℝ, equation y → y ∈ s :=
sorry

end NUMINAMATH_CALUDE_equation_has_four_solutions_l529_52914


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l529_52947

theorem alcohol_solution_proof (initial_volume : Real) (initial_concentration : Real) 
  (added_alcohol : Real) (target_concentration : Real) : 
  initial_volume = 6 →
  initial_concentration = 0.35 →
  added_alcohol = 1.8 →
  target_concentration = 0.5 →
  (initial_concentration * initial_volume + added_alcohol) / (initial_volume + added_alcohol) = target_concentration := by
  sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l529_52947


namespace NUMINAMATH_CALUDE_abc_inequality_l529_52987

noncomputable def e : ℝ := Real.exp 1

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1)
  (eqa : a^2 - 2 * Real.log a + 1 = e)
  (eqb : b^2 - 2 * Real.log b + 2 = e^2)
  (eqc : c^2 - 2 * Real.log c + 3 = e^3) : 
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l529_52987


namespace NUMINAMATH_CALUDE_back_wheel_perimeter_l529_52909

/-- Given a front wheel with perimeter 30 that revolves 240 times, and a back wheel that
    revolves 360 times to cover the same distance, the perimeter of the back wheel is 20. -/
theorem back_wheel_perimeter (front_perimeter : ℝ) (front_revolutions : ℝ) 
  (back_revolutions : ℝ) (back_perimeter : ℝ) : 
  front_perimeter = 30 →
  front_revolutions = 240 →
  back_revolutions = 360 →
  front_perimeter * front_revolutions = back_perimeter * back_revolutions →
  back_perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_back_wheel_perimeter_l529_52909


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l529_52942

/-- A hyperbola with the given properties has eccentricity √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (M N Q E P : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  (M.1^2 / a^2 - M.2^2 / b^2 = 1) →
  (N.1^2 / a^2 - N.2^2 / b^2 = 1) →
  (P.1^2 / a^2 - P.2^2 / b^2 = 1) →
  N = (-M.1, -M.2) →
  Q = (M.1, -M.2) →
  E = (M.1, -3 * M.2) →
  (P.2 - M.2) * (P.1 - M.1) = -(N.2 - M.2) * (N.1 - M.1) →
  let e := Real.sqrt (1 + b^2 / a^2)
  e = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l529_52942


namespace NUMINAMATH_CALUDE_cookie_brownie_difference_l529_52946

def initial_cookies : ℕ := 60
def initial_brownies : ℕ := 10
def daily_cookies_eaten : ℕ := 3
def daily_brownies_eaten : ℕ := 1
def days : ℕ := 7

theorem cookie_brownie_difference :
  initial_cookies - (daily_cookies_eaten * days) - (initial_brownies - (daily_brownies_eaten * days)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_cookie_brownie_difference_l529_52946


namespace NUMINAMATH_CALUDE_girls_from_maple_grove_l529_52992

/-- Represents the number of students in different categories -/
structure StudentCounts where
  total : Nat
  girls : Nat
  boys : Nat
  pinecrest : Nat
  mapleGrove : Nat
  boysPinecrest : Nat

/-- The theorem stating that 40 girls are from Maple Grove School -/
theorem girls_from_maple_grove (s : StudentCounts)
  (h_total : s.total = 150)
  (h_girls : s.girls = 90)
  (h_boys : s.boys = 60)
  (h_pinecrest : s.pinecrest = 80)
  (h_mapleGrove : s.mapleGrove = 70)
  (h_boysPinecrest : s.boysPinecrest = 30)
  (h_total_sum : s.total = s.girls + s.boys)
  (h_school_sum : s.total = s.pinecrest + s.mapleGrove)
  : s.girls - (s.pinecrest - s.boysPinecrest) = 40 := by
  sorry


end NUMINAMATH_CALUDE_girls_from_maple_grove_l529_52992


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l529_52903

/-- Proves that mixing 200 mL of 10% alcohol solution with 50 mL of 30% alcohol solution 
    results in a 14% alcohol solution -/
theorem alcohol_mixture_proof (x_vol : ℝ) (y_vol : ℝ) (x_conc : ℝ) (y_conc : ℝ) 
    (mix_conc : ℝ) (h1 : x_vol = 200) (h2 : y_vol = 50) (h3 : x_conc = 0.1) 
    (h4 : y_conc = 0.3) (h5 : mix_conc = 0.14) : 
    (x_vol * x_conc + y_vol * y_conc) / (x_vol + y_vol) = mix_conc := by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l529_52903


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l529_52952

theorem max_sum_on_circle (x y : ℤ) : x^2 + y^2 = 25 → x + y ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l529_52952


namespace NUMINAMATH_CALUDE_pool_drain_time_l529_52996

/-- Represents the pool draining problem -/
structure PoolDraining where
  capacity : ℝ
  fillTime : ℝ
  drainTime : ℝ
  elapsedTime : ℝ
  remainingWater : ℝ

/-- Theorem stating the solution to the pool draining problem -/
theorem pool_drain_time (p : PoolDraining) 
  (h_capacity : p.capacity = 120)
  (h_fillTime : p.fillTime = 6)
  (h_elapsedTime : p.elapsedTime = 3)
  (h_remainingWater : p.remainingWater = 90) :
  p.drainTime = 4 := by
  sorry


end NUMINAMATH_CALUDE_pool_drain_time_l529_52996


namespace NUMINAMATH_CALUDE_child_ticket_cost_l529_52941

/-- Given information about ticket sales for a baseball game, prove the cost of a child ticket. -/
theorem child_ticket_cost
  (adult_ticket_price : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (adult_tickets : ℕ)
  (h1 : adult_ticket_price = 5)
  (h2 : total_tickets = 85)
  (h3 : total_revenue = 275)
  (h4 : adult_tickets = 35) :
  (total_revenue - adult_tickets * adult_ticket_price) / (total_tickets - adult_tickets) = 2 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l529_52941


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l529_52976

/-- A parallelogram with an area of 200 sq m and an altitude that is twice the corresponding base has a base length of 10 meters. -/
theorem parallelogram_base_length (area : ℝ) (base : ℝ) (altitude : ℝ) :
  area = 200 →
  altitude = 2 * base →
  area = base * altitude →
  base = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l529_52976


namespace NUMINAMATH_CALUDE_largest_factorial_as_consecutive_product_l529_52937

theorem largest_factorial_as_consecutive_product : 
  ∀ n : ℕ, n > 0 → 
  (∃ k : ℕ, n! = (k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5)) → 
  n = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_factorial_as_consecutive_product_l529_52937


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l529_52988

theorem absolute_value_inequality (x : ℝ) : 
  |((3 * x - 2) / (2 * x - 3))| > 3 ↔ 11/9 < x ∧ x < 7/3 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l529_52988


namespace NUMINAMATH_CALUDE_inequality_proof_l529_52935

theorem inequality_proof (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a + b + c = 0) : 
  c * b^2 ≤ a * b^2 ∧ a * b > a * c := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l529_52935


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_bases_l529_52927

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  lateral_side : ℝ
  height : ℝ
  median : ℝ
  base_small : ℝ
  base_large : ℝ

/-- The theorem stating the bases of the isosceles trapezoid with given properties -/
theorem isosceles_trapezoid_bases
  (t : IsoscelesTrapezoid)
  (h1 : t.lateral_side = 41)
  (h2 : t.height = 40)
  (h3 : t.median = 45) :
  t.base_small = 36 ∧ t.base_large = 54 := by
  sorry

#check isosceles_trapezoid_bases

end NUMINAMATH_CALUDE_isosceles_trapezoid_bases_l529_52927


namespace NUMINAMATH_CALUDE_zero_exponent_rule_l529_52917

theorem zero_exponent_rule (a : ℝ) (h : a ≠ 0) : a ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_rule_l529_52917


namespace NUMINAMATH_CALUDE_kevin_bought_two_watermelons_l529_52954

-- Define the weights of the watermelons and the total weight
def weight1 : ℝ := 9.91
def weight2 : ℝ := 4.11
def totalWeight : ℝ := 14.02

-- Define the number of watermelons Kevin bought
def numberOfWatermelons : ℕ := 2

-- Theorem to prove
theorem kevin_bought_two_watermelons :
  weight1 + weight2 = totalWeight ∧ numberOfWatermelons = 2 :=
by sorry

end NUMINAMATH_CALUDE_kevin_bought_two_watermelons_l529_52954


namespace NUMINAMATH_CALUDE_kConnectedSubgraph_l529_52993

/-- A graph G is a pair (V, E) where V is a finite set of vertices and E is a set of edges. -/
structure Graph (α : Type*) where
  V : Finset α
  E : Finset (α × α)

/-- The minimum degree of a graph G. -/
def minDegree {α : Type*} (G : Graph α) : ℕ :=
  sorry

/-- A graph G is k-connected if it remains connected after removing any k-1 vertices. -/
def isKConnected {α : Type*} (G : Graph α) (k : ℕ) : Prop :=
  sorry

/-- A subgraph H of G is a graph whose vertices and edges are subsets of G's vertices and edges. -/
def isSubgraph {α : Type*} (H G : Graph α) : Prop :=
  sorry

/-- The main theorem stating that if δ(G) ≥ 8k and |G| ≤ 16k, then G contains a k-connected subgraph. -/
theorem kConnectedSubgraph {α : Type*} (G : Graph α) (k : ℕ) :
  minDegree G ≥ 8 * k →
  G.V.card ≤ 16 * k →
  ∃ H : Graph α, isSubgraph H G ∧ isKConnected H k :=
sorry

end NUMINAMATH_CALUDE_kConnectedSubgraph_l529_52993


namespace NUMINAMATH_CALUDE_count_odd_numbers_between_215_and_500_l529_52973

theorem count_odd_numbers_between_215_and_500 : 
  (Finset.filter (fun n => n % 2 = 1 ∧ n > 215 ∧ n < 500) (Finset.range 500)).card = 142 :=
by sorry

end NUMINAMATH_CALUDE_count_odd_numbers_between_215_and_500_l529_52973


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l529_52922

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : ℕ) : ℕ := sorry

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (d : ℕ) : ℕ := sorry

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal 101001110010) = 221302 := by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l529_52922


namespace NUMINAMATH_CALUDE_valid_numbers_characterization_l529_52939

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (10 * n) % 7 = 0 ∧
  (n / 1000 * 10000 + (n % 1000)) % 7 = 0 ∧
  (n / 100 * 1000 + (n % 100) + (n / 1000 * 10000)) % 7 = 0 ∧
  (n / 10 * 100 + (n % 10) + (n / 100 * 10000)) % 7 = 0 ∧
  (n * 10 + (n / 1000)) % 7 = 0

theorem valid_numbers_characterization :
  {n : ℕ | is_valid_number n} = {7000, 7007, 7070, 7077, 7700, 7707, 7770, 7777} := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_characterization_l529_52939


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l529_52998

theorem contrapositive_equivalence :
  (∀ a b : ℝ, (a + b = 3 → a^2 + b^2 ≥ 4)) ↔
  (∀ a b : ℝ, (a^2 + b^2 < 4 → a + b ≠ 3)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l529_52998


namespace NUMINAMATH_CALUDE_students_playing_soccer_l529_52929

theorem students_playing_soccer 
  (total_students : ℕ) 
  (boy_students : ℕ) 
  (girls_not_playing : ℕ) 
  (soccer_boys_percentage : ℚ) :
  total_students = 420 →
  boy_students = 320 →
  girls_not_playing = 65 →
  soccer_boys_percentage = 86/100 →
  ∃ (students_playing_soccer : ℕ), 
    students_playing_soccer = 250 ∧
    (total_students - boy_students - girls_not_playing : ℚ) = 
      (1 - soccer_boys_percentage) * students_playing_soccer := by
  sorry

end NUMINAMATH_CALUDE_students_playing_soccer_l529_52929


namespace NUMINAMATH_CALUDE_chips_sold_in_month_l529_52912

theorem chips_sold_in_month (week1 : ℕ) (week2 : ℕ) (week3 : ℕ) (week4 : ℕ) 
  (h1 : week1 = 15)
  (h2 : week2 = 3 * week1)
  (h3 : week3 = 20)
  (h4 : week4 = 20) :
  week1 + week2 + week3 + week4 = 100 := by
  sorry

end NUMINAMATH_CALUDE_chips_sold_in_month_l529_52912


namespace NUMINAMATH_CALUDE_sum_even_positive_lt_100_eq_2450_l529_52995

/-- The sum of all even, positive integers less than 100 -/
def sum_even_positive_lt_100 : ℕ :=
  (Finset.range 50).sum (fun i => 2 * i)

/-- Theorem stating that the sum of all even, positive integers less than 100 is 2450 -/
theorem sum_even_positive_lt_100_eq_2450 : sum_even_positive_lt_100 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_positive_lt_100_eq_2450_l529_52995


namespace NUMINAMATH_CALUDE_johns_car_repair_cost_l529_52913

/-- Calculates the total cost of car repairs including sales tax -/
def total_repair_cost (engine_labor_rate : ℝ) (engine_labor_hours : ℝ) (engine_part_cost : ℝ)
                      (brake_labor_rate : ℝ) (brake_labor_hours : ℝ) (brake_part_cost : ℝ)
                      (tire_labor_rate : ℝ) (tire_labor_hours : ℝ) (tire_cost : ℝ)
                      (sales_tax_rate : ℝ) : ℝ :=
  let engine_cost := engine_labor_rate * engine_labor_hours + engine_part_cost
  let brake_cost := brake_labor_rate * brake_labor_hours + brake_part_cost
  let tire_cost := tire_labor_rate * tire_labor_hours + tire_cost
  let total_before_tax := engine_cost + brake_cost + tire_cost
  let tax_amount := sales_tax_rate * total_before_tax
  total_before_tax + tax_amount

/-- Theorem stating that the total repair cost for John's car is $5238 -/
theorem johns_car_repair_cost :
  total_repair_cost 75 16 1200 85 10 800 50 4 600 0.08 = 5238 := by
  sorry

end NUMINAMATH_CALUDE_johns_car_repair_cost_l529_52913


namespace NUMINAMATH_CALUDE_cubic_sum_of_roots_similar_triangles_sqrt_difference_l529_52982

-- Problem 1
theorem cubic_sum_of_roots (p q : ℝ) : 
  p^2 - 3*p - 2 = 0 → q^2 - 3*q - 2 = 0 → p^3 + q^3 = 45 := by sorry

-- Problem 2
theorem similar_triangles (A H B C K : ℝ) :
  A - H = 45 → C - K = 36 → B - K = 12 → 
  (A - H) / (C - K) = (B - H) / (B - K) →
  B - H = 15 := by sorry

-- Problem 3
theorem sqrt_difference (x : ℝ) :
  Real.sqrt (2*x + 23) + Real.sqrt (2*x - 1) = 12 →
  Real.sqrt (2*x + 23) - Real.sqrt (2*x - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_of_roots_similar_triangles_sqrt_difference_l529_52982


namespace NUMINAMATH_CALUDE_abcdef_hex_bit_length_l529_52920

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => 0  -- This case should not occur for valid hex digits

/-- Converts a hexadecimal number (as a string) to its decimal value -/
def hex_to_decimal (s : String) : ℕ :=
  s.foldr (fun c acc => 16 * acc + hex_to_dec c) 0

/-- Calculates the number of bits needed to represent a natural number -/
def bit_length (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

theorem abcdef_hex_bit_length :
  bit_length (hex_to_decimal "ABCDEF") = 24 := by
  sorry

#eval bit_length (hex_to_decimal "ABCDEF")

end NUMINAMATH_CALUDE_abcdef_hex_bit_length_l529_52920


namespace NUMINAMATH_CALUDE_length_of_QR_l529_52916

-- Define the right triangle PQR
def right_triangle_PQR (QR : ℝ) : Prop :=
  ∃ (P Q R : ℝ × ℝ),
    P = (0, 0) ∧  -- P is at the origin
    Q.1 = 12 ∧ Q.2 = 0 ∧  -- Q is on the horizontal axis, 12 units from P
    R.2 ≠ 0 ∧  -- R is not on the horizontal axis (to ensure a right triangle)
    (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = QR^2 ∧  -- Pythagorean theorem
    (R.1 - P.1)^2 + (R.2 - P.2)^2 = QR^2  -- Pythagorean theorem

-- State the theorem
theorem length_of_QR :
  ∀ QR : ℝ, right_triangle_PQR QR → Real.cos (Real.arccos 0.3) = 12 / QR → QR = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_length_of_QR_l529_52916


namespace NUMINAMATH_CALUDE_sum_greater_than_three_l529_52994

theorem sum_greater_than_three (a b c : ℝ) 
  (h1 : a * b + b * c + c * a > a + b + c) 
  (h2 : a + b + c > 0) : 
  a + b + c > 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_three_l529_52994


namespace NUMINAMATH_CALUDE_system_solution_l529_52923

theorem system_solution (x y k : ℝ) : 
  x + y - 5 * k = 0 → 
  x - y - 9 * k = 0 → 
  2 * x + 3 * y = 6 → 
  4 * k - 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l529_52923


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l529_52921

-- Define the quadrilateral ABCD and points E and F
variable (A B C D E F : Point)

-- Define the properties of the quadrilateral
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the angle measure
def angle_measure (A B C : Point) : ℝ := sorry

-- Define the distance between two points
def distance (P Q : Point) : ℝ := sorry

-- Define the intersection of two rays
def ray_intersection (P Q R S : Point) : Point := sorry

-- Define the perimeter of a triangle
def triangle_perimeter (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_perimeter 
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_angle : angle_measure B A D = π / 3)
  (h_side1 : distance B C = 1)
  (h_side2 : distance A D = 1)
  (h_intersect1 : E = ray_intersection A B C D)
  (h_intersect2 : F = ray_intersection B C A D)
  (h_perimeter1 : ∃ n : ℕ, triangle_perimeter B C E = n)
  (h_perimeter2 : ∃ m : ℕ, triangle_perimeter C D F = m) :
  distance A B + distance B C + distance C D + distance D A = 38 / 7 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l529_52921


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l529_52972

-- Define the function g
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then
    ⌈(Real.cos x) / (x + 3)⌉
  else if x < -3 then
    ⌊(Real.cos x) / (x + 3)⌋
  else
    0  -- This value doesn't matter as g is not defined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l529_52972


namespace NUMINAMATH_CALUDE_truncated_cone_base_area_l529_52958

-- Define the radii of the three cones
def r₁ : ℝ := 10
def r₂ : ℝ := 15
def r₃ : ℝ := 15

-- Define the radius of the smaller base of the truncated cone
def r : ℝ := 2

-- Theorem statement
theorem truncated_cone_base_area 
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 15)
  (h₃ : r₃ = 15)
  (h₄ : (r₁ + r)^2 = r₁^2 + (r₂ + r - r₁)^2)
  (h₅ : (r₂ + r)^2 = r₂^2 + (r₁ + r₂ - r)^2)
  (h₆ : (r₃ + r)^2 = r₃^2 + (r₁ + r₃ - r)^2) :
  π * r^2 = 4 * π := by sorry

end NUMINAMATH_CALUDE_truncated_cone_base_area_l529_52958


namespace NUMINAMATH_CALUDE_polynomial_simplification_l529_52990

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^5 + 5 * x^4 - 3 * Real.sqrt 2 * x^3 + 8 * x^2 + 2 * x - 6) + 
  (-5 * x^4 + Real.sqrt 2 * x^3 - 3 * x^2 + x + 10) = 
  2 * x^5 - 2 * Real.sqrt 2 * x^3 + 5 * x^2 + 3 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l529_52990


namespace NUMINAMATH_CALUDE_toy_store_shelves_l529_52965

theorem toy_store_shelves (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : 
  initial_stock = 6 →
  new_shipment = 18 →
  bears_per_shelf = 6 →
  (initial_stock + new_shipment) / bears_per_shelf = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l529_52965


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l529_52991

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x, k * x^2 - 3 * x + 2 = 0) → (k = 0 ∨ k = 9/8) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l529_52991


namespace NUMINAMATH_CALUDE_train_length_proof_l529_52980

/-- Proves that given two trains of equal length running on parallel lines in the same direction,
    with the faster train moving at 52 km/hr and the slower train at 36 km/hr,
    if the faster train passes the slower train in 36 seconds,
    then the length of each train is 80 meters. -/
theorem train_length_proof (faster_speed slower_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) : 
  faster_speed = 52 →
  slower_speed = 36 →
  passing_time = 36 →
  (faster_speed - slower_speed) * passing_time * (5 / 18) = 2 * train_length →
  train_length = 80 := by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l529_52980


namespace NUMINAMATH_CALUDE_min_d_value_l529_52933

theorem min_d_value (t a b d : ℕ) : 
  (3 * t = 2 * a + 2 * b + 2016) →  -- Triangle perimeter exceeds rectangle perimeter by 2016
  (t = a + d) →                     -- Triangle side exceeds one rectangle side by d
  (t = b + 2 * d) →                 -- Triangle side exceeds other rectangle side by 2d
  (a > 0 ∧ b > 0) →                 -- Rectangle has non-zero perimeter
  (∀ d' : ℕ, d' < d → 
    ¬(∃ t' a' b' : ℕ, 
      (3 * t' = 2 * a' + 2 * b' + 2016) ∧ 
      (t' = a' + d') ∧ 
      (t' = b' + 2 * d') ∧ 
      (a' > 0 ∧ b' > 0))) →
  d = 505 :=
by sorry

end NUMINAMATH_CALUDE_min_d_value_l529_52933


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l529_52910

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, S n = (a 1 + a n) * n / 2) →   -- sum formula for arithmetic sequence
  (a 2 - 1)^3 + 2014 * (a 2 - 1) = Real.sin (2011 * Real.pi / 3) →
  (a 2013 - 1)^3 + 2014 * (a 2013 - 1) = Real.cos (2011 * Real.pi / 6) →
  S 2014 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l529_52910


namespace NUMINAMATH_CALUDE_cube_volume_problem_l529_52945

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a - 2) * a * (a + 2) = a^3 - 16 → 
  a^3 = 8 := by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l529_52945


namespace NUMINAMATH_CALUDE_fraction_value_l529_52970

theorem fraction_value : (150 + (150 / 10)) / (15 - 5) = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l529_52970


namespace NUMINAMATH_CALUDE_yard_length_18_trees_l529_52969

/-- The length of a yard with equally spaced trees -/
def yardLength (numTrees : ℕ) (distanceBetweenTrees : ℝ) : ℝ :=
  (numTrees - 1 : ℝ) * distanceBetweenTrees

/-- Theorem: The length of a yard with 18 equally spaced trees,
    where the distance between consecutive trees is 15 meters, is 255 meters -/
theorem yard_length_18_trees : yardLength 18 15 = 255 := by
  sorry

#eval yardLength 18 15

end NUMINAMATH_CALUDE_yard_length_18_trees_l529_52969


namespace NUMINAMATH_CALUDE_complex_product_l529_52962

theorem complex_product (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2) 
  (h2 : Complex.abs z₂ = 3) 
  (h3 : 3 * z₁ - 2 * z₂ = 2 - I) : 
  z₁ * z₂ = -30/13 + 72/13 * I := by
sorry

end NUMINAMATH_CALUDE_complex_product_l529_52962


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l529_52924

/-- The volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume 
  (l : ℝ) -- lateral edge length
  (α : ℝ) -- angle between lateral edge and base plane
  (h1 : l > 0) -- lateral edge length is positive
  (h2 : 0 < α ∧ α < π/2) -- angle is between 0 and π/2
  : ∃ (V : ℝ), V = (Real.sqrt 3 * l^3 * Real.cos α^2 * Real.sin α) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l529_52924


namespace NUMINAMATH_CALUDE_cents_ratio_randi_to_peter_l529_52908

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The total cents Ray has -/
def ray_total_cents : ℕ := 175

/-- The cents Ray gives to Peter -/
def cents_to_peter : ℕ := 30

/-- The number of extra nickels Randi has compared to Peter -/
def extra_nickels_randi : ℕ := 6

/-- Theorem stating the ratio of cents given to Randi vs Peter -/
theorem cents_ratio_randi_to_peter :
  let peter_nickels := cents_to_peter / nickel_value
  let randi_nickels := peter_nickels + extra_nickels_randi
  let cents_to_randi := randi_nickels * nickel_value
  (cents_to_randi : ℚ) / cents_to_peter = 2 := by sorry

end NUMINAMATH_CALUDE_cents_ratio_randi_to_peter_l529_52908


namespace NUMINAMATH_CALUDE_roosters_count_l529_52968

/-- Given a total number of chickens and a proportion of roosters to hens to chicks,
    calculate the number of roosters. -/
def count_roosters (total_chickens : ℕ) (rooster_parts hen_parts chick_parts : ℕ) : ℕ :=
  let total_parts := rooster_parts + hen_parts + chick_parts
  let chickens_per_part := total_chickens / total_parts
  rooster_parts * chickens_per_part

/-- Theorem stating that given 9000 total chickens and a proportion of 2:1:3 for
    roosters:hens:chicks, the number of roosters is 3000. -/
theorem roosters_count :
  count_roosters 9000 2 1 3 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_roosters_count_l529_52968


namespace NUMINAMATH_CALUDE_hyperbola_sum_l529_52940

/-- Given a hyperbola with center (-3, 1), one focus at (2, 1), and one vertex at (-1, 1),
    prove that h + k + a + b = 0 + √21, where (h, k) is the center, a is the distance from
    the center to the vertex, and b^2 = c^2 - a^2 with c being the distance from the center
    to the focus. -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -3 →
  k = 1 →
  (2 : ℝ) - h = c →
  (-1 : ℝ) - h = a →
  b^2 = c^2 - a^2 →
  h + k + a + b = 0 + Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_sum_l529_52940


namespace NUMINAMATH_CALUDE_road_building_equation_l529_52981

theorem road_building_equation (x : ℝ) 
  (h_positive : x > 0) 
  (h_team_a_length : 9 > 0) 
  (h_team_b_length : 12 > 0) 
  (h_team_b_faster : x + 1 > x) : 
  9 / x - 12 / (x + 1) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_road_building_equation_l529_52981


namespace NUMINAMATH_CALUDE_parallelogram_formation_condition_l529_52919

/-- Represents a point in a one-dimensional space -/
structure Point :=
  (x : ℝ)

/-- Represents a line segment between two points -/
def LineSegment (P Q : Point) : ℝ :=
  |Q.x - P.x|

/-- Condition for forming a parallelogram when rotating line segments -/
def ParallelogramCondition (P Q R S T : Point) (a b c : ℝ) : Prop :=
  P.x < Q.x ∧ Q.x < R.x ∧ R.x < S.x ∧ S.x < T.x ∧
  LineSegment P Q = a ∧
  LineSegment P R = b ∧
  LineSegment P T = c ∧
  b = c - a

theorem parallelogram_formation_condition 
  (P Q R S T : Point) (a b c : ℝ) :
  ParallelogramCondition P Q R S T a b c →
  ∃ (P' T' : Point),
    LineSegment Q P' = a ∧
    LineSegment R T' = c - b ∧
    LineSegment P' T' = b - a ∧
    LineSegment S P' = LineSegment S T' :=
sorry

end NUMINAMATH_CALUDE_parallelogram_formation_condition_l529_52919


namespace NUMINAMATH_CALUDE_shortest_path_length_l529_52977

/-- Regular tetrahedron with edge length 2 -/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (is_regular : edge_length = 2)

/-- Point on the surface of a regular tetrahedron -/
structure SurfacePoint (t : RegularTetrahedron) :=
  (coordinates : ℝ × ℝ × ℝ)

/-- Midpoint of an edge on a regular tetrahedron -/
def edge_midpoint (t : RegularTetrahedron) : SurfacePoint t :=
  sorry

/-- Distance between two points on the surface of a regular tetrahedron -/
def surface_distance (t : RegularTetrahedron) (p q : SurfacePoint t) : ℝ :=
  sorry

/-- Sequentially next edge midpoint -/
def next_edge_midpoint (t : RegularTetrahedron) (p : SurfacePoint t) : SurfacePoint t :=
  sorry

/-- Theorem: Shortest path between midpoints of sequentially next edges is √6 -/
theorem shortest_path_length (t : RegularTetrahedron) (p : SurfacePoint t) :
  surface_distance t p (next_edge_midpoint t p) = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_shortest_path_length_l529_52977


namespace NUMINAMATH_CALUDE_parallel_iff_abs_x_eq_two_l529_52960

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b = (k * a.1, k * a.2)

/-- Given two non-zero vectors a = (1, x) and b = (x², 4x), prove that a is parallel to b if and only if |x| = 2 -/
theorem parallel_iff_abs_x_eq_two (x : ℝ) :
  are_parallel (1, x) (x^2, 4*x) ↔ |x| = 2 :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_abs_x_eq_two_l529_52960


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l529_52959

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I)*z = 5) : 
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l529_52959


namespace NUMINAMATH_CALUDE_initial_amount_spent_l529_52911

theorem initial_amount_spent (total_sets : ℕ) (twenty_dollar_sets : ℕ) (price_per_set : ℕ) :
  total_sets = 250 →
  twenty_dollar_sets = 178 →
  price_per_set = 20 →
  (twenty_dollar_sets * price_per_set : ℕ) = 3560 :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_spent_l529_52911


namespace NUMINAMATH_CALUDE_modified_geometric_structure_pieces_l529_52905

/-- Calculates the sum of an arithmetic progression -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Calculates the nth triangular number -/
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The total number of pieces in the modified geometric structure -/
theorem modified_geometric_structure_pieces :
  let num_rows : ℕ := 10
  let first_rod_count : ℕ := 3
  let rod_difference : ℕ := 3
  let connector_rows : ℕ := num_rows + 1
  let rod_count := arithmetic_sum first_rod_count rod_difference num_rows
  let connector_count := triangular_number connector_rows
  rod_count + connector_count = 231 := by
  sorry

end NUMINAMATH_CALUDE_modified_geometric_structure_pieces_l529_52905


namespace NUMINAMATH_CALUDE_cos_alpha_value_l529_52986

theorem cos_alpha_value (α : Real) (h : Real.sin (α / 2) = Real.sqrt 3 / 3) : 
  Real.cos α = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l529_52986


namespace NUMINAMATH_CALUDE_outbound_time_calculation_l529_52963

/-- The time taken for John to drive to the distant city -/
def outbound_time : ℝ := 30

/-- The time taken for John to return from the distant city -/
def return_time : ℝ := 5

/-- The speed increase on the return trip -/
def speed_increase : ℝ := 12

/-- The speed on the outbound trip -/
def outbound_speed : ℝ := 60

/-- The speed on the return trip -/
def return_speed : ℝ := outbound_speed + speed_increase

theorem outbound_time_calculation :
  outbound_time * outbound_speed = return_time * return_speed := by sorry

#check outbound_time_calculation

end NUMINAMATH_CALUDE_outbound_time_calculation_l529_52963


namespace NUMINAMATH_CALUDE_lawn_mowing_problem_l529_52930

theorem lawn_mowing_problem (mary_rate tom_rate : ℚ) (tom_work_time : ℚ) 
  (h1 : mary_rate = 1 / 4)
  (h2 : tom_rate = 1 / 5)
  (h3 : tom_work_time = 2) :
  1 - tom_rate * tom_work_time = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_problem_l529_52930


namespace NUMINAMATH_CALUDE_ice_palace_steps_count_l529_52943

/-- The number of steps in the Ice Palace staircase -/
def ice_palace_steps : ℕ := 30

/-- The time Alice takes to walk 20 steps (in seconds) -/
def time_for_20_steps : ℕ := 120

/-- The time Alice takes to walk all steps (in seconds) -/
def time_for_all_steps : ℕ := 180

/-- Theorem: The number of steps in the Ice Palace staircase is 30 -/
theorem ice_palace_steps_count :
  ice_palace_steps = (time_for_all_steps * 20) / time_for_20_steps :=
sorry

end NUMINAMATH_CALUDE_ice_palace_steps_count_l529_52943


namespace NUMINAMATH_CALUDE_trapezoid_ab_length_l529_52936

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  ab : ℝ
  -- Length of side CD
  cd : ℝ
  -- Ratio of areas of triangles ABC and ADC
  area_ratio : ℝ
  -- Assumption that AB + CD = 150
  sum_sides : ab + cd = 150
  -- Assumption that AB = 3CD
  ab_triple_cd : ab = 3 * cd
  -- Assumption that area ratio of ABC to ADC is 4:1
  area_ratio_def : area_ratio = 4 / 1

/-- Theorem stating that under given conditions, AB = 120 cm -/
theorem trapezoid_ab_length (t : Trapezoid) : t.ab = 120 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_ab_length_l529_52936


namespace NUMINAMATH_CALUDE_days_to_complete_paper_l529_52907

-- Define the paper length and writing rate
def paper_length : ℕ := 63
def pages_per_day : ℕ := 21

-- Theorem statement
theorem days_to_complete_paper : 
  (paper_length / pages_per_day : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_days_to_complete_paper_l529_52907


namespace NUMINAMATH_CALUDE_netGainDifference_l529_52906

/-- Represents a job candidate with their associated costs and revenue --/
structure Candidate where
  salary : ℕ
  revenue : ℕ
  trainingMonths : ℕ
  trainingCostPerMonth : ℕ
  hiringBonusPercent : ℕ

/-- Calculates the net gain for a candidate --/
def netGain (c : Candidate) : ℕ :=
  c.revenue - c.salary - (c.trainingMonths * c.trainingCostPerMonth) - (c.salary * c.hiringBonusPercent / 100)

/-- The two candidates as described in the problem --/
def candidate1 : Candidate :=
  { salary := 42000
    revenue := 93000
    trainingMonths := 3
    trainingCostPerMonth := 1200
    hiringBonusPercent := 0 }

def candidate2 : Candidate :=
  { salary := 45000
    revenue := 92000
    trainingMonths := 0
    trainingCostPerMonth := 0
    hiringBonusPercent := 1 }

/-- Theorem stating the difference in net gain between the two candidates --/
theorem netGainDifference : netGain candidate1 - netGain candidate2 = 850 := by
  sorry

end NUMINAMATH_CALUDE_netGainDifference_l529_52906


namespace NUMINAMATH_CALUDE_catering_pies_l529_52931

theorem catering_pies (total_pies : ℕ) (num_teams : ℕ) (first_team_pies : ℕ) (third_team_pies : ℕ) 
  (h1 : total_pies = 750)
  (h2 : num_teams = 3)
  (h3 : first_team_pies = 235)
  (h4 : third_team_pies = 240) :
  total_pies - first_team_pies - third_team_pies = 275 := by
  sorry

end NUMINAMATH_CALUDE_catering_pies_l529_52931


namespace NUMINAMATH_CALUDE_expand_product_l529_52918

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l529_52918


namespace NUMINAMATH_CALUDE_average_difference_l529_52948

theorem average_difference (x : ℝ) : 
  (10 + 70 + x) / 3 = (20 + 40 + 60) / 3 - 7 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l529_52948


namespace NUMINAMATH_CALUDE_product_sum_base_c_l529_52985

def base_c_to_decimal (n : ℕ) (c : ℕ) : ℕ := c + n

def decimal_to_base_c (n : ℕ) (c : ℕ) : ℕ := n - c

theorem product_sum_base_c (c : ℕ) : 
  (base_c_to_decimal 12 c) * (base_c_to_decimal 14 c) * (base_c_to_decimal 18 c) = 
    5 * c^3 + 3 * c^2 + 2 * c + 0 →
  decimal_to_base_c (base_c_to_decimal 12 c + base_c_to_decimal 14 c + 
                     base_c_to_decimal 18 c + base_c_to_decimal 20 c) c = 40 :=
by sorry

end NUMINAMATH_CALUDE_product_sum_base_c_l529_52985


namespace NUMINAMATH_CALUDE_marc_total_spent_l529_52961

/-- The total amount Marc spent on his purchases -/
def total_spent (model_car_price model_car_quantity paint_price paint_quantity
                 paintbrush_price paintbrush_quantity : ℕ) : ℕ :=
  model_car_price * model_car_quantity +
  paint_price * paint_quantity +
  paintbrush_price * paintbrush_quantity

/-- Theorem stating that Marc spent $160 in total -/
theorem marc_total_spent :
  total_spent 20 5 10 5 2 5 = 160 := by
  sorry

end NUMINAMATH_CALUDE_marc_total_spent_l529_52961


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l529_52983

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares : a^2 + b^2 + c^2 = 4) : 
  a^4 + b^4 + c^4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l529_52983


namespace NUMINAMATH_CALUDE_coefficient_x3y2z5_in_expansion_l529_52953

/-- The coefficient of x³y²z⁵ in the expansion of (2x+y+z)¹⁰ -/
def coefficient : ℕ :=
  2^3 * (Nat.choose 10 3) * (Nat.choose 7 2) * (Nat.choose 5 5)

/-- Theorem stating that the coefficient of x³y²z⁵ in (2x+y+z)¹⁰ is 20160 -/
theorem coefficient_x3y2z5_in_expansion : coefficient = 20160 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y2z5_in_expansion_l529_52953


namespace NUMINAMATH_CALUDE_no_fogh_prime_l529_52975

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem no_fogh_prime :
  ¬∃ (f o g h : ℕ),
    is_digit f ∧ is_digit o ∧ is_digit g ∧ is_digit h ∧
    f ≠ o ∧ f ≠ g ∧ f ≠ h ∧ o ≠ g ∧ o ≠ h ∧ g ≠ h ∧
    (1000 * f + 100 * o + 10 * g + h ≥ 1000) ∧
    (1000 * f + 100 * o + 10 * g + h < 10000) ∧
    is_prime (1000 * f + 100 * o + 10 * g + h) ∧
    (1000 * f + 100 * o + 10 * g + h) * (f * o * g * h) = (1000 * f + 100 * o + 10 * g + h) :=
sorry

end NUMINAMATH_CALUDE_no_fogh_prime_l529_52975


namespace NUMINAMATH_CALUDE_trapezium_second_side_length_l529_52900

/-- Proves that for a trapezium with given measurements, the length of the second parallel side is 18 cm -/
theorem trapezium_second_side_length 
  (side1 : ℝ) 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : side1 = 20) 
  (h2 : height = 12) 
  (h3 : area = 228) :
  ∃ side2 : ℝ, side2 = 18 ∧ area = (1/2) * (side1 + side2) * height :=
by
  sorry

end NUMINAMATH_CALUDE_trapezium_second_side_length_l529_52900


namespace NUMINAMATH_CALUDE_compound_interest_rate_l529_52979

theorem compound_interest_rate : 
  ∀ (P : ℝ) (A : ℝ) (I : ℝ) (t : ℕ) (r : ℝ),
  A = 19828.80 →
  I = 2828.80 →
  t = 2 →
  A = P + I →
  A = P * (1 + r) ^ t →
  r = 0.08 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l529_52979
