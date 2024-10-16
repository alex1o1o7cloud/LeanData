import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_of_specific_integers_l473_47370

theorem absolute_value_of_specific_integers :
  ∃ (a b c : ℤ),
    (∀ x : ℤ, x < 0 → x ≤ a) ∧
    (∀ x : ℤ, |x| ≥ |b|) ∧
    (∀ x : ℤ, x > 0 → c ≤ x) ∧
    |a + b - c| = 2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_of_specific_integers_l473_47370


namespace NUMINAMATH_CALUDE_unique_B_for_divisibility_l473_47301

/-- Represents a four-digit number in the form 4BB2 -/
def fourDigitNumber (B : ℕ) : ℕ := 4000 + 100 * B + 10 * B + 2

/-- Checks if a number is divisible by 11 -/
def divisibleBy11 (n : ℕ) : Prop := n % 11 = 0

/-- B is a single digit -/
def isSingleDigit (B : ℕ) : Prop := B ≥ 0 ∧ B ≤ 9

theorem unique_B_for_divisibility : 
  ∃! B : ℕ, isSingleDigit B ∧ divisibleBy11 (fourDigitNumber B) ∧ B = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_B_for_divisibility_l473_47301


namespace NUMINAMATH_CALUDE_odd_function_derivative_l473_47395

theorem odd_function_derivative (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →
  (∀ x, HasDerivAt f (g x) x) →
  ∀ x, g (-x) = -g x := by
sorry

end NUMINAMATH_CALUDE_odd_function_derivative_l473_47395


namespace NUMINAMATH_CALUDE_sqrt_two_minus_two_cos_four_equals_two_sin_two_l473_47362

theorem sqrt_two_minus_two_cos_four_equals_two_sin_two :
  Real.sqrt (2 - 2 * Real.cos 4) = 2 * Real.sin 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_two_cos_four_equals_two_sin_two_l473_47362


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l473_47340

/-- An isosceles triangle with sides of 3cm and 7cm has a perimeter of 13cm. -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 3 ∧ b = 7 ∧ c = 3 →  -- Two sides are 3cm, one side is 7cm
  (a = b ∨ b = c ∨ a = c) →  -- The triangle is isosceles
  a + b + c = 13 :=  -- The perimeter is 13cm
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l473_47340


namespace NUMINAMATH_CALUDE_max_product_given_sum_l473_47379

theorem max_product_given_sum (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 40 → ∀ x y : ℝ, x > 0 → y > 0 → x + y = 40 → x * y ≤ a * b → a * b ≤ 400 := by
  sorry

end NUMINAMATH_CALUDE_max_product_given_sum_l473_47379


namespace NUMINAMATH_CALUDE_roots_of_equation_l473_47312

theorem roots_of_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁ = -1 ∧ x₂ = 0) ∧ 
  (∀ x : ℝ, (x + 1) * x = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l473_47312


namespace NUMINAMATH_CALUDE_second_derivative_implies_m_l473_47331

/-- Given a function f(x) = 2/x, prove that if its second derivative at m is -1/2, then m = -2 -/
theorem second_derivative_implies_m (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = 2 / x) →
  (deriv^[2] f m = -1/2) →
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_second_derivative_implies_m_l473_47331


namespace NUMINAMATH_CALUDE_n_in_interval_l473_47313

def is_repeating_decimal (d : ℚ) (period : ℕ) : Prop :=
  ∃ (k : ℕ), d * 10^period - d.floor = k / (10^period - 1)

theorem n_in_interval (n : ℕ) (hn : n < 1000) 
  (h1 : is_repeating_decimal (1 / n) 3)
  (h2 : is_repeating_decimal (1 / (n + 4)) 6) :
  n ∈ Set.Icc 1 150 := by
  sorry

end NUMINAMATH_CALUDE_n_in_interval_l473_47313


namespace NUMINAMATH_CALUDE_sum_of_translated_parabolas_is_nonhorizontal_line_l473_47347

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a function obtained from translating a parabola horizontally -/
def TranslatedParabola (p : Parabola) (d : ℝ) : ℝ → ℝ := 
  fun x => p.a * (x - d)^2 + p.b * (x - d) + p.c

/-- The sum of a translated parabola and its reflection about the x-axis -/
def SumOfTranslatedParabolas (p : Parabola) (d : ℝ) : ℝ → ℝ :=
  fun x => TranslatedParabola p d x + TranslatedParabola { a := -p.a, b := -p.b, c := -p.c } (-d) x

/-- Theorem stating that the sum of translated parabolas is a non-horizontal line -/
theorem sum_of_translated_parabolas_is_nonhorizontal_line (p : Parabola) (d : ℝ) :
  ∃ m k, m ≠ 0 ∧ ∀ x, SumOfTranslatedParabolas p d x = m * x + k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_translated_parabolas_is_nonhorizontal_line_l473_47347


namespace NUMINAMATH_CALUDE_line_intersects_parabola_at_one_point_l473_47311

/-- The value of k for which the line x = k intersects the parabola x = -3y² - 4y + 7 at exactly one point -/
def k : ℚ := 25/3

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -3*y^2 - 4*y + 7

theorem line_intersects_parabola_at_one_point :
  ∃! y : ℝ, parabola y = k := by sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_at_one_point_l473_47311


namespace NUMINAMATH_CALUDE_odot_inequality_iff_l473_47310

-- Define the ⊙ operation
def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

-- State the theorem
theorem odot_inequality_iff (x : ℝ) : odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_odot_inequality_iff_l473_47310


namespace NUMINAMATH_CALUDE_point_on_line_trig_identity_l473_47339

theorem point_on_line_trig_identity (θ : Real) :
  2 * Real.cos θ + Real.sin θ = 0 →
  Real.cos (2 * θ) + (1/2) * Real.sin (2 * θ) = -1 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_trig_identity_l473_47339


namespace NUMINAMATH_CALUDE_divisor_problem_l473_47360

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 158 →
  quotient = 9 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 17 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l473_47360


namespace NUMINAMATH_CALUDE_race_finish_order_l473_47322

def race_order : List Nat := [1, 7, 9, 10, 8, 11, 2, 5, 3, 4, 6, 12]

theorem race_finish_order :
  ∀ (finish : Nat → Nat),
  (∀ n, n ∈ race_order → finish n ∈ Finset.range 13) →
  (∀ n, n ∈ race_order → ∃ k, n * (finish n) = 13 * k + 1) →
  (∀ n m, n ≠ m → n ∈ race_order → m ∈ race_order → finish n ≠ finish m) →
  (∀ n, n ∈ race_order → finish n = (List.indexOf n race_order).succ) :=
by sorry

#check race_finish_order

end NUMINAMATH_CALUDE_race_finish_order_l473_47322


namespace NUMINAMATH_CALUDE_min_coach_handshakes_correct_l473_47358

/-- The minimum number of handshakes by coaches in a basketball tournament --/
def min_coach_handshakes : ℕ := 60

/-- Total number of handshakes in the tournament --/
def total_handshakes : ℕ := 495

/-- Number of teams in the tournament --/
def num_teams : ℕ := 2

/-- Function to calculate the number of player-to-player handshakes --/
def player_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the minimum number of handshakes by coaches --/
theorem min_coach_handshakes_correct :
  ∃ (n : ℕ), n % num_teams = 0 ∧
  player_handshakes n + (n / num_teams) * num_teams = total_handshakes ∧
  (n / num_teams) * num_teams = min_coach_handshakes :=
sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_correct_l473_47358


namespace NUMINAMATH_CALUDE_worst_player_is_father_l473_47315

-- Define the family members
inductive FamilyMember
  | Father
  | Sister
  | Daughter
  | Son

-- Define the sex of a family member
def sex : FamilyMember → Bool
  | FamilyMember.Father => true   -- true represents male
  | FamilyMember.Sister => false  -- false represents female
  | FamilyMember.Daughter => false
  | FamilyMember.Son => true

-- Define the twin relationship
def isTwin : FamilyMember → FamilyMember → Bool
  | FamilyMember.Father, FamilyMember.Sister => true
  | FamilyMember.Sister, FamilyMember.Father => true
  | FamilyMember.Daughter, FamilyMember.Son => true
  | FamilyMember.Son, FamilyMember.Daughter => true
  | _, _ => false

-- Define the theorem
theorem worst_player_is_father :
  ∀ (worst best : FamilyMember),
    (∃ twin : FamilyMember, isTwin worst twin ∧ sex twin ≠ sex best) →
    isTwin worst best →
    worst = FamilyMember.Father :=
by sorry

end NUMINAMATH_CALUDE_worst_player_is_father_l473_47315


namespace NUMINAMATH_CALUDE_arrangement_existence_l473_47353

/-- Represents a group of kindergarten children -/
structure ChildrenGroup where
  total : ℕ  -- Total number of children

/-- Represents an arrangement of children in pairs -/
structure Arrangement where
  boy_pairs : ℕ  -- Number of pairs of two boys
  girl_pairs : ℕ  -- Number of pairs of two girls
  mixed_pairs : ℕ  -- Number of pairs with one boy and one girl

/-- Checks if an arrangement is valid for a given group -/
def is_valid_arrangement (group : ChildrenGroup) (arr : Arrangement) : Prop :=
  2 * (arr.boy_pairs + arr.girl_pairs) + arr.mixed_pairs = group.total

/-- Theorem stating the existence of a specific arrangement -/
theorem arrangement_existence (group : ChildrenGroup) 
  (arr1 arr2 : Arrangement) 
  (h1 : is_valid_arrangement group arr1)
  (h2 : is_valid_arrangement group arr2)
  (h3 : arr1.boy_pairs = 3 * arr1.girl_pairs)
  (h4 : arr2.boy_pairs = 4 * arr2.girl_pairs) :
  ∃ (arr3 : Arrangement), 
    is_valid_arrangement group arr3 ∧ 
    arr3.boy_pairs = 7 * arr3.girl_pairs := by
  sorry

end NUMINAMATH_CALUDE_arrangement_existence_l473_47353


namespace NUMINAMATH_CALUDE_polynomial_decrease_l473_47332

theorem polynomial_decrease (b : ℝ) :
  let P : ℝ → ℝ := fun x ↦ -2 * x + b
  ∀ x : ℝ, P (x + 1) = P x - 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_decrease_l473_47332


namespace NUMINAMATH_CALUDE_initial_stock_proof_l473_47352

/-- The number of coloring books sold during the sale -/
def books_sold : ℕ := 6

/-- The number of shelves used for remaining books -/
def shelves_used : ℕ := 3

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 7

/-- The initial number of coloring books in stock -/
def initial_stock : ℕ := shelves_used * books_per_shelf + books_sold

theorem initial_stock_proof : initial_stock = 27 := by
  sorry

end NUMINAMATH_CALUDE_initial_stock_proof_l473_47352


namespace NUMINAMATH_CALUDE_firewood_sacks_filled_l473_47330

theorem firewood_sacks_filled (sack_capacity : ℕ) (father_wood : ℕ) (ranger_wood : ℕ) (worker_wood : ℕ) (num_workers : ℕ) :
  sack_capacity = 20 →
  father_wood = 80 →
  ranger_wood = 80 →
  worker_wood = 120 →
  num_workers = 2 →
  (father_wood + ranger_wood + num_workers * worker_wood) / sack_capacity = 20 :=
by sorry

end NUMINAMATH_CALUDE_firewood_sacks_filled_l473_47330


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l473_47396

theorem sufficient_condition_for_inequality (a : ℝ) :
  (a < 1) → (∀ x : ℝ, a ≤ |x| + |x - 1|) ∧
  ¬(∀ a : ℝ, (∀ x : ℝ, a ≤ |x| + |x - 1|) → (a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l473_47396


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l473_47350

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l473_47350


namespace NUMINAMATH_CALUDE_fourth_task_completion_time_l473_47384

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define the problem parameters
def start_time : Time := { hours := 8, minutes := 45 }
def third_task_completion : Time := { hours := 11, minutes := 25 }
def num_tasks : Nat := 4

-- Calculate the time difference in minutes
def time_diff (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

-- Calculate the duration of a single task
def single_task_duration : Nat :=
  (time_diff start_time third_task_completion) / (num_tasks - 1)

-- Function to add minutes to a given time
def add_minutes (t : Time) (m : Nat) : Time :=
  let total_minutes := t.hours * 60 + t.minutes + m
  { hours := total_minutes / 60, minutes := total_minutes % 60 }

-- Theorem to prove
theorem fourth_task_completion_time :
  add_minutes third_task_completion single_task_duration = { hours := 12, minutes := 18 } := by
  sorry

end NUMINAMATH_CALUDE_fourth_task_completion_time_l473_47384


namespace NUMINAMATH_CALUDE_sphere_surface_area_l473_47387

theorem sphere_surface_area (r : ℝ) (h : r = 4) : 4 * Real.pi * r^2 = 64 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l473_47387


namespace NUMINAMATH_CALUDE_tyler_cake_eggs_l473_47309

/-- Represents the number of eggs needed for a cake --/
def eggs_for_cake (people : ℕ) : ℕ := 2 * (people / 4)

/-- Represents the number of additional eggs needed --/
def additional_eggs_needed (recipe_eggs : ℕ) (available_eggs : ℕ) : ℕ :=
  max (recipe_eggs - available_eggs) 0

theorem tyler_cake_eggs : 
  additional_eggs_needed (eggs_for_cake 8) 3 = 1 := by sorry

end NUMINAMATH_CALUDE_tyler_cake_eggs_l473_47309


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l473_47334

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - 11 * i) / (3 - 4 * i) = 2 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l473_47334


namespace NUMINAMATH_CALUDE_acute_slope_implies_a_is_one_l473_47348

/-- The curve C is defined by y = x³ - 2ax² + 2ax -/
def C (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*a*x^2 + 2*a*x

/-- The derivative of C with respect to x -/
def C_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*a*x + 2*a

/-- The slope is acute if it's greater than 0 -/
def is_slope_acute (slope : ℝ) : Prop := slope > 0

theorem acute_slope_implies_a_is_one :
  ∀ a : ℤ, (∀ x : ℝ, is_slope_acute (C_derivative a x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_acute_slope_implies_a_is_one_l473_47348


namespace NUMINAMATH_CALUDE_equation_solution_l473_47335

theorem equation_solution (a c : ℝ) :
  let x := (c^2 - a^3) / (3*a^2 - 1)
  x^2 + c^2 = (a - x)^3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l473_47335


namespace NUMINAMATH_CALUDE_b_range_for_inequality_l473_47397

/-- Given an inequality ax + b > 2(x + 1) with solution set {x | x < 1}, 
    prove that the range of values for b is (4, +∞) -/
theorem b_range_for_inequality (a b : ℝ) : 
  (∀ x, ax + b > 2*(x + 1) ↔ x < 1) → 
  ∃ y, y > 4 ∧ b > y :=
sorry

end NUMINAMATH_CALUDE_b_range_for_inequality_l473_47397


namespace NUMINAMATH_CALUDE_bread_cost_l473_47369

/-- The cost of the loaf of bread given the conditions of Ted's sandwich-making scenario --/
theorem bread_cost (sandwich_meat_cost meat_packs cheese_cost cheese_packs : ℕ → ℚ)
  (meat_coupon cheese_coupon : ℚ) (sandwich_price : ℚ) (sandwich_count : ℕ) :
  let total_meat_cost := meat_packs 2 * sandwich_meat_cost 1 - meat_coupon
  let total_cheese_cost := cheese_packs 2 * cheese_cost 1 - cheese_coupon
  let total_ingredient_cost := total_meat_cost + total_cheese_cost
  let total_revenue := sandwich_count * sandwich_price
  total_revenue - total_ingredient_cost = 4 :=
by sorry

#check bread_cost (λ _ => 5) (λ _ => 2) (λ _ => 4) (λ _ => 2) 1 1 2 10

end NUMINAMATH_CALUDE_bread_cost_l473_47369


namespace NUMINAMATH_CALUDE_factorization_3mx_minus_9my_l473_47336

theorem factorization_3mx_minus_9my (m x y : ℝ) :
  3 * m * x - 9 * m * y = 3 * m * (x - 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3mx_minus_9my_l473_47336


namespace NUMINAMATH_CALUDE_im_z_squared_gt_two_iff_xy_gt_one_l473_47372

/-- For a complex number z, Im(z^2) > 2 if and only if the product of its real and imaginary parts is greater than 1 -/
theorem im_z_squared_gt_two_iff_xy_gt_one (z : ℂ) :
  Complex.im (z^2) > 2 ↔ Complex.re z * Complex.im z > 1 := by
sorry

end NUMINAMATH_CALUDE_im_z_squared_gt_two_iff_xy_gt_one_l473_47372


namespace NUMINAMATH_CALUDE_sum_of_ages_l473_47368

/-- Given that in 5 years Nacho will be three times older than Divya, 
    and Divya is currently 5 years old, prove that the sum of their 
    current ages is 30 years. -/
theorem sum_of_ages (nacho_age divya_age : ℕ) : 
  divya_age = 5 → 
  nacho_age + 5 = 3 * (divya_age + 5) → 
  nacho_age + divya_age = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l473_47368


namespace NUMINAMATH_CALUDE_m_range_proof_l473_47366

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0

def q (m : ℝ) : Prop := m ∈ Set.Icc (-1 : ℝ) 5

-- Define the range of m
def m_range : Set ℝ := Set.Ioi (-1 : ℝ) ∪ Set.Ioc 1 5

-- Theorem statement
theorem m_range_proof :
  (∀ m : ℝ, (p m ∧ q m → False) ∧ (p m ∨ q m)) →
  (∀ m : ℝ, m ∈ m_range ↔ (p m ∨ q m) ∧ ¬(p m ∧ q m)) :=
by sorry

end NUMINAMATH_CALUDE_m_range_proof_l473_47366


namespace NUMINAMATH_CALUDE_inequality_proof_l473_47341

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  (a / (b + c) + b / (a + c) + c / (a + b)) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l473_47341


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l473_47373

def lowest_price : ℝ := 10
def highest_price : ℝ := 17

theorem percentage_increase_proof :
  (highest_price - lowest_price) / lowest_price * 100 = 70 := by sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l473_47373


namespace NUMINAMATH_CALUDE_square_and_fourth_power_mod_eight_l473_47337

theorem square_and_fourth_power_mod_eight (n : ℤ) :
  (Even n → n ^ 2 % 8 = 0 ∨ n ^ 2 % 8 = 4) ∧
  (Odd n → n ^ 2 % 8 = 1) ∧
  (Odd n → n ^ 4 % 8 = 1) := by
  sorry

end NUMINAMATH_CALUDE_square_and_fourth_power_mod_eight_l473_47337


namespace NUMINAMATH_CALUDE_impossible_to_swap_folds_l473_47319

/-- Represents the number of folds on one side of a rhinoceros -/
structure Folds :=
  (vertical : ℕ)
  (horizontal : ℕ)

/-- Represents the state of folds on both sides of a rhinoceros -/
structure RhinoState :=
  (left : Folds)
  (right : Folds)

/-- A scratch operation that can be performed on the rhinoceros -/
inductive ScratchOp
  | left_vertical
  | left_horizontal
  | right_vertical
  | right_horizontal

/-- Defines a valid initial state for the rhinoceros -/
def valid_initial_state (s : RhinoState) : Prop :=
  s.left.vertical + s.left.horizontal + s.right.vertical + s.right.horizontal = 17

/-- Defines the result of applying a scratch operation to a state -/
def apply_scratch (s : RhinoState) (op : ScratchOp) : RhinoState :=
  sorry

/-- Defines when a state is reachable from the initial state through scratching -/
def reachable (initial : RhinoState) (final : RhinoState) : Prop :=
  sorry

/-- The main theorem stating that it's impossible to swap vertical and horizontal folds -/
theorem impossible_to_swap_folds (initial : RhinoState) :
  valid_initial_state initial →
  ¬∃ (final : RhinoState),
    reachable initial final ∧
    final.left.vertical = initial.left.horizontal ∧
    final.left.horizontal = initial.left.vertical ∧
    final.right.vertical = initial.right.horizontal ∧
    final.right.horizontal = initial.right.vertical :=
  sorry

end NUMINAMATH_CALUDE_impossible_to_swap_folds_l473_47319


namespace NUMINAMATH_CALUDE_cos_sin_identity_l473_47342

theorem cos_sin_identity (α β : Real) :
  (Real.cos (α * π / 180) * Real.cos ((180 - α) * π / 180) + 
   Real.sin (α * π / 180) * Real.sin ((α / 2) * π / 180)) = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l473_47342


namespace NUMINAMATH_CALUDE_smallest_valid_number_l473_47381

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (m : ℕ),
    n = 5 * 10^k + m ∧
    m * 10 + 5 = (5 * 10^k + m) / 4

theorem smallest_valid_number :
  ∃ (n : ℕ),
    is_valid_number n ∧
    ∀ (m : ℕ), is_valid_number m → n ≤ m ∧
    n = 512820
  := by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l473_47381


namespace NUMINAMATH_CALUDE_ellipse_m_value_l473_47359

/-- An ellipse with equation mx^2 + y^2 = 1, foci on the y-axis, and major axis length three times the minor axis length has m = 4/9 --/
theorem ellipse_m_value (m : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), m * x^2 + y^2 = 1 ↔ x^2 / b^2 + y^2 / a^2 = 1) ∧ 
    (∃ (c : ℝ), c > 0 ∧ a^2 = b^2 + c^2) ∧ 
    2 * a = 3 * (2 * b)) →
  m = 4/9 := by
sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l473_47359


namespace NUMINAMATH_CALUDE_remainder_sum_mod_35_l473_47328

theorem remainder_sum_mod_35 (f y z : ℤ) 
  (hf : f % 5 = 3) 
  (hy : y % 5 = 4) 
  (hz : z % 7 = 6) : 
  (f + y + z) % 35 = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_35_l473_47328


namespace NUMINAMATH_CALUDE_remaining_digits_count_l473_47377

theorem remaining_digits_count (total : ℕ) (avg_total : ℚ) (subset : ℕ) (avg_subset : ℚ) (avg_remaining : ℚ)
  (h1 : total = 10)
  (h2 : avg_total = 80)
  (h3 : subset = 6)
  (h4 : avg_subset = 58)
  (h5 : avg_remaining = 113) :
  total - subset = 4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_digits_count_l473_47377


namespace NUMINAMATH_CALUDE_tangent_x_axis_tangent_y_axis_unique_x_intercept_unique_y_intercept_is_parabola_l473_47361

/-- A parabola represented by the equation (x + 1/2y - 1)² = 0 -/
def parabola (x y : ℝ) : Prop := (x + 1/2 * y - 1)^2 = 0

/-- The parabola is tangent to the x-axis at the point (1,0) -/
theorem tangent_x_axis : parabola 1 0 := by sorry

/-- The parabola is tangent to the y-axis at the point (0,2) -/
theorem tangent_y_axis : parabola 0 2 := by sorry

/-- The parabola touches the x-axis only at (1,0) -/
theorem unique_x_intercept (x : ℝ) : 
  parabola x 0 → x = 1 := by sorry

/-- The parabola touches the y-axis only at (0,2) -/
theorem unique_y_intercept (y : ℝ) : 
  parabola 0 y → y = 2 := by sorry

/-- The equation represents a parabola -/
theorem is_parabola : 
  ∃ (a b c : ℝ), ∀ (x y : ℝ), parabola x y ↔ y = a*x^2 + b*x + c := by sorry

end NUMINAMATH_CALUDE_tangent_x_axis_tangent_y_axis_unique_x_intercept_unique_y_intercept_is_parabola_l473_47361


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l473_47323

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) : A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l473_47323


namespace NUMINAMATH_CALUDE_min_apples_in_basket_sixty_two_satisfies_conditions_min_apples_is_sixty_two_l473_47393

theorem min_apples_in_basket (N : ℕ) : 
  (N % 3 = 2) ∧ (N % 4 = 2) ∧ (N % 5 = 2) → N ≥ 62 :=
by sorry

theorem sixty_two_satisfies_conditions : 
  (62 % 3 = 2) ∧ (62 % 4 = 2) ∧ (62 % 5 = 2) :=
by sorry

theorem min_apples_is_sixty_two : 
  ∃ (N : ℕ), (N % 3 = 2) ∧ (N % 4 = 2) ∧ (N % 5 = 2) ∧ N = 62 :=
by sorry

end NUMINAMATH_CALUDE_min_apples_in_basket_sixty_two_satisfies_conditions_min_apples_is_sixty_two_l473_47393


namespace NUMINAMATH_CALUDE_smallest_sequence_sum_l473_47378

theorem smallest_sequence_sum : ∃ (A B C D : ℕ),
  (A > 0 ∧ B > 0 ∧ C > 0) ∧  -- A, B, C are positive integers
  (∃ (r : ℚ), C - B = B - A ∧ C = B * r ∧ D = C * r) ∧  -- arithmetic and geometric sequences
  (C : ℚ) / B = 7 / 4 ∧  -- C/B = 7/4
  (∀ (A' B' C' D' : ℕ),
    (A' > 0 ∧ B' > 0 ∧ C' > 0) →
    (∃ (r' : ℚ), C' - B' = B' - A' ∧ C' = B' * r' ∧ D' = C' * r') →
    (C' : ℚ) / B' = 7 / 4 →
    A + B + C + D ≤ A' + B' + C' + D') ∧
  A + B + C + D = 97 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sequence_sum_l473_47378


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l473_47320

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00001 = 24681 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l473_47320


namespace NUMINAMATH_CALUDE_peaches_picked_up_correct_l473_47356

/-- Represents the fruit stand inventory --/
structure FruitStand where
  initialPeaches : ℕ
  initialOranges : ℕ
  peachesSold : ℕ
  orangesAdded : ℕ
  finalPeaches : ℕ
  finalOranges : ℕ

/-- Calculates the number of peaches picked up from the orchard --/
def peachesPickedUp (stand : FruitStand) : ℕ :=
  stand.finalPeaches - (stand.initialPeaches - stand.peachesSold)

/-- Theorem stating that the number of peaches picked up is correct --/
theorem peaches_picked_up_correct (stand : FruitStand) :
  peachesPickedUp stand = stand.finalPeaches - (stand.initialPeaches - stand.peachesSold) :=
by
  sorry

/-- Sally's fruit stand inventory --/
def sallysStand : FruitStand := {
  initialPeaches := 13
  initialOranges := 5
  peachesSold := 7
  orangesAdded := 22
  finalPeaches := 55
  finalOranges := 27
}

#eval peachesPickedUp sallysStand

end NUMINAMATH_CALUDE_peaches_picked_up_correct_l473_47356


namespace NUMINAMATH_CALUDE_marsupial_protein_consumption_l473_47324

theorem marsupial_protein_consumption (absorption_rate : ℝ) (absorbed_amount : ℝ) (total_consumed : ℝ) : 
  absorption_rate = 0.40 →
  absorbed_amount = 16 →
  absorbed_amount = absorption_rate * total_consumed →
  total_consumed = 40 := by
  sorry

end NUMINAMATH_CALUDE_marsupial_protein_consumption_l473_47324


namespace NUMINAMATH_CALUDE_price_reduction_problem_l473_47333

theorem price_reduction_problem (x : ℝ) : 
  (∀ (P : ℝ), P > 0 → 
    P * (1 - x / 100) * (1 - 20 / 100) = P * (1 - 40 / 100)) → 
  x = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_problem_l473_47333


namespace NUMINAMATH_CALUDE_similar_triangles_l473_47307

/-- Given five complex numbers representing points in a plane, if three triangles formed by these points are directly similar, then a fourth triangle is also directly similar to them. -/
theorem similar_triangles (a b c u v : ℂ) 
  (h : (v - a) / (u - a) = (u - v) / (b - v) ∧ (u - v) / (b - v) = (c - u) / (v - u)) : 
  (v - a) / (u - a) = (c - a) / (b - a) := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_l473_47307


namespace NUMINAMATH_CALUDE_scooter_initial_price_l473_47325

/-- The initial purchase price of a scooter, given the repair cost, selling price, and gain percentage. -/
theorem scooter_initial_price (repair_cost selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : repair_cost = 200)
  (h2 : selling_price = 1400)
  (h3 : gain_percent = 40) :
  ∃ (initial_price : ℝ), 
    selling_price = (1 + gain_percent / 100) * (initial_price + repair_cost) ∧ 
    initial_price = 800 := by
  sorry

end NUMINAMATH_CALUDE_scooter_initial_price_l473_47325


namespace NUMINAMATH_CALUDE_page_number_digit_difference_l473_47345

/-- Counts the occurrences of a digit in a range of numbers -/
def countDigit (d : Nat) (start finish : Nat) : Nat :=
  sorry

/-- The difference between the count of 5's and 3's in page numbers from 1 to 512 -/
theorem page_number_digit_difference :
  let pages := 512
  let start_page := 1
  let end_page := pages
  let digit_five := 5
  let digit_three := 3
  (countDigit digit_five start_page end_page) - (countDigit digit_three start_page end_page) = 22 :=
sorry

end NUMINAMATH_CALUDE_page_number_digit_difference_l473_47345


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l473_47303

theorem completing_square_equivalence (x : ℝ) :
  x^2 - 4*x + 3 = 0 ↔ (x - 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l473_47303


namespace NUMINAMATH_CALUDE_f_zero_is_zero_l473_47302

-- Define the function f
variable (f : ℝ → ℝ)

-- State the condition
axiom functional_equation : ∀ x y : ℝ, f (x + y) = f x + f y

-- Theorem to prove
theorem f_zero_is_zero : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_is_zero_l473_47302


namespace NUMINAMATH_CALUDE_no_prime_sum_10003_l473_47385

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ p + q = 10003 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_10003_l473_47385


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l473_47308

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mwf : ℕ  -- Hours worked on Monday, Wednesday, Friday
  hours_tth : ℕ  -- Hours worked on Tuesday, Thursday
  days_mwf : ℕ   -- Number of days worked with hours_mwf
  days_tth : ℕ   -- Number of days worked with hours_tth
  weekly_earnings : ℕ

/-- Calculate Sheila's hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := schedule.hours_mwf * schedule.days_mwf + schedule.hours_tth * schedule.days_tth
  schedule.weekly_earnings / total_hours

/-- Theorem: Sheila's hourly wage is $11 --/
theorem sheila_hourly_wage :
  let schedule : WorkSchedule := {
    hours_mwf := 8,
    hours_tth := 6,
    days_mwf := 3,
    days_tth := 2,
    weekly_earnings := 396
  }
  hourly_wage schedule = 11 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l473_47308


namespace NUMINAMATH_CALUDE_wood_per_sack_l473_47321

theorem wood_per_sack (total_wood : ℕ) (num_sacks : ℕ) (h1 : total_wood = 80) (h2 : num_sacks = 4) :
  total_wood / num_sacks = 20 := by
  sorry

end NUMINAMATH_CALUDE_wood_per_sack_l473_47321


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_complementary_l473_47399

-- Define the set of cards
inductive Card : Type
  | Hearts | Spades | Diamonds | Clubs

-- Define the set of people
inductive Person : Type
  | A | B | C | D

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "Person A gets a club"
def event_A_club (d : Distribution) : Prop := d Person.A = Card.Clubs

-- Define the event "Person B gets a club"
def event_B_club (d : Distribution) : Prop := d Person.B = Card.Clubs

-- Theorem statement
theorem mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(event_A_club d ∧ event_B_club d)) ∧
  (∃ d : Distribution, ¬event_A_club d ∧ ¬event_B_club d) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_complementary_l473_47399


namespace NUMINAMATH_CALUDE_total_money_division_l473_47363

theorem total_money_division (b c : ℕ) (total : ℕ) : 
  (b : ℚ) / c = 4 / 16 →
  c * 100 = 1600 →
  total = b * 100 + c * 100 →
  total = 2000 := by
sorry

end NUMINAMATH_CALUDE_total_money_division_l473_47363


namespace NUMINAMATH_CALUDE_total_whales_is_178_l473_47365

/-- Represents the number of whales observed during Ishmael's monitoring trips -/
def total_whales (first_trip_male : ℕ) : ℕ :=
  let first_trip_female := 2 * first_trip_male
  let first_trip_total := first_trip_male + first_trip_female
  let second_trip_baby := 8
  let second_trip_parents := 2 * second_trip_baby
  let second_trip_total := second_trip_baby + second_trip_parents
  let third_trip_male := first_trip_male / 2
  let third_trip_female := first_trip_female
  let third_trip_total := third_trip_male + third_trip_female
  first_trip_total + second_trip_total + third_trip_total

/-- Theorem stating that the total number of whales observed is 178 -/
theorem total_whales_is_178 : total_whales 28 = 178 := by
  sorry

end NUMINAMATH_CALUDE_total_whales_is_178_l473_47365


namespace NUMINAMATH_CALUDE_rectangle_segment_sum_l473_47383

theorem rectangle_segment_sum (a b : ℝ) (n : ℕ) (h1 : a = 4) (h2 : b = 3) (h3 : n = 168) :
  let diagonal := Real.sqrt (a^2 + b^2)
  let segment_sum := n * diagonal
  segment_sum = 840 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_segment_sum_l473_47383


namespace NUMINAMATH_CALUDE_f_inequality_and_abs_inequality_l473_47344

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

-- Define the set M
def M : Set ℝ := {x | 1 < x ∧ x < 4}

-- Theorem statement
theorem f_inequality_and_abs_inequality :
  (∀ x, f x < 3 ↔ x ∈ M) ∧
  (∀ a b, a ∈ M → b ∈ M → |a + b| < |1 + a * b|) := by sorry

end NUMINAMATH_CALUDE_f_inequality_and_abs_inequality_l473_47344


namespace NUMINAMATH_CALUDE_range_of_a_l473_47386

def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a = 0}

theorem range_of_a : ∀ a : ℝ, (A ∪ B a = A) ↔ (0 ≤ a ∧ a < 4) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l473_47386


namespace NUMINAMATH_CALUDE_divisible_by_512_l473_47374

theorem divisible_by_512 (n : ℤ) (h : Odd n) :
  ∃ k : ℤ, n^12 - n^8 - n^4 + 1 = 512 * k := by
sorry

end NUMINAMATH_CALUDE_divisible_by_512_l473_47374


namespace NUMINAMATH_CALUDE_cost_calculation_l473_47391

/-- The total cost of buying bread and drinks -/
def total_cost (a b : ℝ) : ℝ := 2 * a + 3 * b

/-- Theorem: The total cost of buying 2 pieces of bread at 'a' yuan each 
    and 3 bottles of drink at 'b' yuan each is equal to 2a+3b yuan -/
theorem cost_calculation (a b : ℝ) : 
  total_cost a b = 2 * a + 3 * b := by sorry

end NUMINAMATH_CALUDE_cost_calculation_l473_47391


namespace NUMINAMATH_CALUDE_container_capacity_proof_l473_47343

/-- The capacity of a container in liters, given the number of portions and volume per portion in milliliters. -/
def container_capacity (portions : ℕ) (ml_per_portion : ℕ) : ℚ :=
  (portions * ml_per_portion : ℚ) / 1000

/-- Proves that a container with 10 portions of 200 ml each has a capacity of 2 liters. -/
theorem container_capacity_proof :
  container_capacity 10 200 = 2 := by
  sorry

#eval container_capacity 10 200

end NUMINAMATH_CALUDE_container_capacity_proof_l473_47343


namespace NUMINAMATH_CALUDE_sqrt_equation_equivalence_l473_47392

theorem sqrt_equation_equivalence (x : ℝ) (h : x > 9) :
  (Real.sqrt (x - 9 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 9 * Real.sqrt (x - 9)) - 3) ↔ 
  x ≥ 40.5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_equivalence_l473_47392


namespace NUMINAMATH_CALUDE_max_four_digit_binary_is_15_l473_47389

/-- The maximum value of a four-digit binary number in decimal -/
def max_four_digit_binary : ℕ := 15

/-- A function to convert a four-digit binary number to decimal -/
def binary_to_decimal (b₃ b₂ b₁ b₀ : Bool) : ℕ :=
  (if b₃ then 8 else 0) + (if b₂ then 4 else 0) + (if b₁ then 2 else 0) + (if b₀ then 1 else 0)

/-- Theorem stating that the maximum value of a four-digit binary number is 15 -/
theorem max_four_digit_binary_is_15 :
  ∀ b₃ b₂ b₁ b₀ : Bool, binary_to_decimal b₃ b₂ b₁ b₀ ≤ max_four_digit_binary :=
by sorry

end NUMINAMATH_CALUDE_max_four_digit_binary_is_15_l473_47389


namespace NUMINAMATH_CALUDE_apartment_rent_calculation_l473_47357

/-- Proves that the rent for a shared apartment is $1100 given specific conditions -/
theorem apartment_rent_calculation (utilities groceries roommate_payment : ℕ) 
  (h1 : utilities = 114)
  (h2 : groceries = 300)
  (h3 : roommate_payment = 757) :
  ∃ (rent : ℕ), rent = 1100 ∧ (rent + utilities + groceries) / 2 = roommate_payment :=
by sorry

end NUMINAMATH_CALUDE_apartment_rent_calculation_l473_47357


namespace NUMINAMATH_CALUDE_neg_half_pow_4_mul_6_three_squared_mul_neg_three_cubed_two_cubed_sum_four_times_find_p_in_equation_l473_47364

-- Operation rule of multiplication of powers with the same base
axiom pow_mul_rule {α : Type*} [Monoid α] (a : α) (m n : ℕ) : a^m * a^n = a^(m+n)

-- Statement 1
theorem neg_half_pow_4_mul_6 : (-1/2 : ℚ)^4 * (-1/2 : ℚ)^6 = (-1/2 : ℚ)^10 := by sorry

-- Statement 2
theorem three_squared_mul_neg_three_cubed : (3 : ℤ)^2 * (-3 : ℤ)^3 = -243 := by sorry

-- Statement 3
theorem two_cubed_sum_four_times : (2 : ℕ)^3 + (2 : ℕ)^3 + (2 : ℕ)^3 + (2 : ℕ)^3 = (2 : ℕ)^5 := by sorry

-- Statement 4
theorem find_p_in_equation (x y : ℝ) :
  ∃ p : ℕ, (x - y)^2 * (x - y)^p * (x - y)^5 = (x - y)^2023 ∧ p = 2016 := by sorry

end NUMINAMATH_CALUDE_neg_half_pow_4_mul_6_three_squared_mul_neg_three_cubed_two_cubed_sum_four_times_find_p_in_equation_l473_47364


namespace NUMINAMATH_CALUDE_buses_passed_count_l473_47329

/-- Represents the frequency of Dallas to Houston buses in minutes -/
def dallas_to_houston_frequency : ℕ := 40

/-- Represents the frequency of Houston to Dallas buses in minutes -/
def houston_to_dallas_frequency : ℕ := 60

/-- Represents the trip duration in hours -/
def trip_duration : ℕ := 6

/-- Represents the minute offset for Houston to Dallas buses -/
def houston_to_dallas_offset : ℕ := 30

/-- Calculates the number of Dallas-bound buses a Houston-bound bus passes on the highway -/
def buses_passed : ℕ := 
  sorry

theorem buses_passed_count : buses_passed = 10 := by
  sorry

end NUMINAMATH_CALUDE_buses_passed_count_l473_47329


namespace NUMINAMATH_CALUDE_inequality_solution_set_l473_47394

theorem inequality_solution_set (x : ℝ) : 
  (6 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x + 4) ↔ (2 + Real.sqrt 2 < x ∧ x < 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l473_47394


namespace NUMINAMATH_CALUDE_alice_age_l473_47398

theorem alice_age (alice_age mother_age : ℕ) 
  (h1 : alice_age = mother_age - 18)
  (h2 : alice_age + mother_age = 50) : 
  alice_age = 16 := by sorry

end NUMINAMATH_CALUDE_alice_age_l473_47398


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l473_47305

/-- Given a two-digit number n, returns the number obtained by switching its digits -/
def switch_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Checks if a number satisfies the condition: switching its digits and multiplying by 3 results in 3n -/
def satisfies_condition (n : ℕ) : Prop :=
  3 * switch_digits n = 3 * n

theorem smallest_satisfying_number :
  ∃ (n : ℕ),
    10 ≤ n ∧ n < 100 ∧
    satisfies_condition n ∧
    (∀ m : ℕ, 10 ≤ m ∧ m < n → ¬satisfies_condition m) ∧
    n = 11 := by
  sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l473_47305


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l473_47300

-- Problem 1
theorem factorization_problem_1 (x : ℝ) : 2*x^2 - 4*x = 2*x*(x - 2) := by sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) : x*y^2 - 2*x*y + x = x*(y - 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l473_47300


namespace NUMINAMATH_CALUDE_xy_sum_zero_l473_47367

theorem xy_sum_zero (x y : ℝ) :
  (x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) = 1 →
  x + y = 0 ∧ ∀ z, ((x + Real.sqrt (x^2 + 1)) * (z + Real.sqrt (z^2 + 1)) = 1 → x + z = 0) :=
by sorry

end NUMINAMATH_CALUDE_xy_sum_zero_l473_47367


namespace NUMINAMATH_CALUDE_four_digit_kabulek_numbers_l473_47314

def is_kabulek (n : ℕ) : Prop :=
  ∃ x y : ℕ,
    n = 100 * x + y ∧
    x < 100 ∧
    y < 100 ∧
    (x + y) ^ 2 = n

theorem four_digit_kabulek_numbers :
  ∀ n : ℕ,
    1000 ≤ n ∧ n < 10000 →
    is_kabulek n ↔ n = 2025 ∨ n = 3025 ∨ n = 9801 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_kabulek_numbers_l473_47314


namespace NUMINAMATH_CALUDE_initial_average_mark_l473_47351

/-- Proves that the initial average mark of a class is 60, given the specified conditions. -/
theorem initial_average_mark (total_students : ℕ) (excluded_students : ℕ) (excluded_avg : ℚ) (remaining_avg : ℚ) :
  total_students = 9 →
  excluded_students = 5 →
  excluded_avg = 44 →
  remaining_avg = 80 →
  (total_students * (total_students * excluded_avg + (total_students - excluded_students) * remaining_avg)) / 
  (excluded_students * total_students + (total_students - excluded_students) * total_students) = 60 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_mark_l473_47351


namespace NUMINAMATH_CALUDE_temperature_difference_l473_47338

/-- The difference between the highest and lowest temperatures of the day -/
theorem temperature_difference (highest lowest : ℤ) (h1 : highest = 1) (h2 : lowest = -9) :
  highest - lowest = 10 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l473_47338


namespace NUMINAMATH_CALUDE_smallest_number_in_sequence_l473_47354

theorem smallest_number_in_sequence (x y z t : ℝ) : 
  y = 2 * x →
  z = 4 * y →
  t = (y + z) / 3 →
  (x + y + z + t) / 4 = 220 →
  x = 2640 / 43 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_in_sequence_l473_47354


namespace NUMINAMATH_CALUDE_chinese_vs_english_spanish_difference_l473_47380

def hours_english : ℕ := 6
def hours_chinese : ℕ := 7
def hours_spanish : ℕ := 4
def hours_french : ℕ := 5

theorem chinese_vs_english_spanish_difference :
  Int.natAbs (hours_chinese - (hours_english + hours_spanish)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chinese_vs_english_spanish_difference_l473_47380


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l473_47306

noncomputable def f (x : ℝ) : ℝ := -2/3 * x^3 + 3/2 * x^2 - x

theorem f_increasing_on_interval :
  ∀ x ∈ Set.Icc (1/2 : ℝ) 1, 
    (∀ y ∈ Set.Icc (1/2 : ℝ) 1, x ≤ y → f x ≤ f y) :=
by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l473_47306


namespace NUMINAMATH_CALUDE_min_sum_of_square_roots_l473_47318

theorem min_sum_of_square_roots (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ x : ℝ, Real.sqrt ((x - a)^2 + b^2) + Real.sqrt ((x - b)^2 + a^2) ≥ Real.sqrt (2 * (a^2 + b^2))) ∧
  (∃ x : ℝ, Real.sqrt ((x - a)^2 + b^2) + Real.sqrt ((x - b)^2 + a^2) = Real.sqrt (2 * (a^2 + b^2))) :=
by
  sorry

#check min_sum_of_square_roots

end NUMINAMATH_CALUDE_min_sum_of_square_roots_l473_47318


namespace NUMINAMATH_CALUDE_cistern_filling_time_l473_47327

theorem cistern_filling_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) : 
  empty_rate = 8 →
  combined_fill_time = 40 / 3 →
  1 / fill_time - 1 / empty_rate = 1 / combined_fill_time →
  fill_time = 5 := by
sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l473_47327


namespace NUMINAMATH_CALUDE_disease_cases_estimation_l473_47355

/-- A function representing the number of disease cases over time -/
def cases (t : ℝ) : ℝ := 800000 - 19995 * (t - 1970)

theorem disease_cases_estimation :
  cases 1995 = 300125 ∧ cases 2005 = 100175 :=
by
  sorry

end NUMINAMATH_CALUDE_disease_cases_estimation_l473_47355


namespace NUMINAMATH_CALUDE_restaurant_bill_problem_l473_47375

/-- Given a group of three people eating at a restaurant where:
    - The first person's dish costs $10
    - The second person's dish costs an unknown amount x
    - The third person's dish costs $17
    - They give a 10% tip
    - The waiter receives $4 in gratuity
    This theorem proves that the second person's dish costs $13. -/
theorem restaurant_bill_problem (x : ℝ) : 
  (10 : ℝ) + x + 17 > 0 →
  0.1 * ((10 : ℝ) + x + 17) = 4 →
  x = 13 := by sorry

end NUMINAMATH_CALUDE_restaurant_bill_problem_l473_47375


namespace NUMINAMATH_CALUDE_exists_k_no_roots_l473_47349

/-- A homogeneous real polynomial of degree 2 -/
def HomogeneousPolynomial2 (a b c : ℝ) (x y : ℝ) : ℝ :=
  a * x^2 + b * x * y + c * y^2

/-- A homogeneous real polynomial of degree 3 -/
noncomputable def HomogeneousPolynomial3 (x y : ℝ) : ℝ :=
  sorry

/-- Main theorem -/
theorem exists_k_no_roots
  (a b c : ℝ)
  (h_pos : b^2 < 4*a*c) :
  ∃ k : ℝ, k > 0 ∧
    ∀ x y : ℝ, x^2 + y^2 < k →
      HomogeneousPolynomial2 a b c x y = HomogeneousPolynomial3 x y →
        x = 0 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_exists_k_no_roots_l473_47349


namespace NUMINAMATH_CALUDE_volume_ratio_l473_47388

theorem volume_ratio (A B C : ℝ) 
  (h1 : A = (B + C) / 4)
  (h2 : B = (C + A) / 6) :
  C / (A + B) = 23 / 12 := by
sorry

end NUMINAMATH_CALUDE_volume_ratio_l473_47388


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l473_47304

theorem same_remainder_divisor : 
  ∃ (d : ℕ), d > 1 ∧ 
  (1059 % d = 1417 % d) ∧ 
  (1059 % d = 2312 % d) ∧ 
  (1417 % d = 2312 % d) ∧
  (∀ (k : ℕ), k > d → 
    (1059 % k ≠ 1417 % k) ∨ 
    (1059 % k ≠ 2312 % k) ∨ 
    (1417 % k ≠ 2312 % k)) →
  d = 179 := by
sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l473_47304


namespace NUMINAMATH_CALUDE_students_who_like_basketball_l473_47376

/-- Given a class of students where some play basketball and/or cricket, 
    this theorem proves the number of students who like basketball. -/
theorem students_who_like_basketball 
  (cricket : ℕ)
  (both : ℕ)
  (basketball_or_cricket : ℕ)
  (h1 : cricket = 8)
  (h2 : both = 4)
  (h3 : basketball_or_cricket = 14) :
  basketball_or_cricket = cricket + (basketball_or_cricket - cricket - both) - both :=
by sorry

end NUMINAMATH_CALUDE_students_who_like_basketball_l473_47376


namespace NUMINAMATH_CALUDE_ariels_female_fish_l473_47316

/-- Given that Ariel has 45 fish in total and 2/3 of the fish are male,
    prove that the number of female fish is 15. -/
theorem ariels_female_fish :
  ∀ (total_fish : ℕ) (male_fraction : ℚ),
    total_fish = 45 →
    male_fraction = 2/3 →
    (total_fish : ℚ) * (1 - male_fraction) = 15 :=
by sorry

end NUMINAMATH_CALUDE_ariels_female_fish_l473_47316


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_8_l473_47317

theorem no_solution_implies_a_leq_8 (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_8_l473_47317


namespace NUMINAMATH_CALUDE_kabadi_players_count_l473_47326

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 15

/-- The number of people who play kho kho only -/
def kho_kho_only : ℕ := 25

/-- The number of people who play both kabadi and kho kho -/
def both_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := 35

theorem kabadi_players_count : 
  kabadi_players = total_players - kho_kho_only + both_players :=
by
  sorry

#check kabadi_players_count

end NUMINAMATH_CALUDE_kabadi_players_count_l473_47326


namespace NUMINAMATH_CALUDE_cubic_equation_has_real_root_l473_47346

theorem cubic_equation_has_real_root (a b : ℝ) : 
  ∃ x : ℝ, a * x^3 + a * x + b = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_has_real_root_l473_47346


namespace NUMINAMATH_CALUDE_sum_of_fifth_terms_l473_47382

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sum_of_fifth_terms (a b : ℕ → ℝ) :
  is_arithmetic_sequence a →
  is_geometric_sequence b →
  a 1 + b 1 = 3 →
  a 2 + b 2 = 7 →
  a 3 + b 3 = 15 →
  a 4 + b 4 = 35 →
  a 5 + b 5 = 91 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fifth_terms_l473_47382


namespace NUMINAMATH_CALUDE_average_battery_lifespan_l473_47390

def battery_lifespans : List ℝ := [30, 35, 25, 25, 30, 34, 26, 25, 29, 21]

theorem average_battery_lifespan :
  (List.sum battery_lifespans) / (List.length battery_lifespans) = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_battery_lifespan_l473_47390


namespace NUMINAMATH_CALUDE_medal_distribution_proof_l473_47371

/-- Represents the number of runners --/
def total_runners : ℕ := 10

/-- Represents the number of British runners --/
def british_runners : ℕ := 4

/-- Represents the number of medals --/
def medals : ℕ := 3

/-- Calculates the number of ways to award medals with at least one British runner winning --/
def ways_to_award_medals : ℕ := sorry

theorem medal_distribution_proof :
  ways_to_award_medals = 492 :=
by sorry

end NUMINAMATH_CALUDE_medal_distribution_proof_l473_47371
