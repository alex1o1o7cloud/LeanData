import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l1527_152702

theorem equation_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 = 2020 * x) ↔
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1527_152702


namespace NUMINAMATH_CALUDE_quadratic_shift_theorem_l1527_152709

/-- Represents a quadratic function of the form y = a(x-h)² + k -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a quadratic function vertically -/
def verticalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h, k := f.k + shift }

/-- Shifts a quadratic function horizontally -/
def horizontalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h - shift, k := f.k }

/-- The theorem stating that shifting y = 5(x-1)² + 1 down by 3 and left by 2 results in y = 5(x+1)² - 2 -/
theorem quadratic_shift_theorem :
  let f : QuadraticFunction := { a := 5, h := 1, k := 1 }
  let g := horizontalShift (verticalShift f (-3)) 2
  g = { a := 5, h := -1, k := -2 } := by sorry

end NUMINAMATH_CALUDE_quadratic_shift_theorem_l1527_152709


namespace NUMINAMATH_CALUDE_even_function_k_value_l1527_152741

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = kx^2 + (k - 1)x + 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 3

theorem even_function_k_value :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_k_value_l1527_152741


namespace NUMINAMATH_CALUDE_four_squared_sum_equals_four_cubed_l1527_152784

theorem four_squared_sum_equals_four_cubed : 4^2 + 4^2 + 4^2 + 4^2 = 4^3 := by
  sorry

end NUMINAMATH_CALUDE_four_squared_sum_equals_four_cubed_l1527_152784


namespace NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l1527_152728

theorem cube_tetrahedron_surface_area_ratio : 
  let cube_side_length : ℝ := 2
  let tetrahedron_side_length : ℝ := cube_side_length * Real.sqrt 2
  let cube_surface_area : ℝ := 6 * cube_side_length^2
  let tetrahedron_surface_area : ℝ := Real.sqrt 3 * tetrahedron_side_length^2
  cube_surface_area / tetrahedron_surface_area = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l1527_152728


namespace NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_parts_l1527_152722

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def first_digit (n : ℕ) : ℕ := n / 100

def second_digit (n : ℕ) : ℕ := (n / 10) % 10

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem smallest_three_digit_divisible_by_parts : 
  ∃ (n : ℕ), is_three_digit n ∧ 
  first_digit n ≠ 0 ∧
  n % (n / 10) = 0 ∧ 
  n % (last_two_digits n) = 0 ∧
  ∀ m, is_three_digit m ∧ 
       first_digit m ≠ 0 ∧ 
       m % (m / 10) = 0 ∧ 
       m % (last_two_digits m) = 0 → 
       n ≤ m ∧
  n = 110 := by
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_divisible_by_parts_l1527_152722


namespace NUMINAMATH_CALUDE_sum_vector_magnitude_l1527_152754

/-- Given planar vectors a and b satisfying specific conditions, 
    prove that the magnitude of their sum is 5. -/
theorem sum_vector_magnitude (a b : ℝ × ℝ) : 
  (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 3) →
  (a = (1/2, Real.sqrt 3/2)) →
  (Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5) →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_vector_magnitude_l1527_152754


namespace NUMINAMATH_CALUDE_max_value_a4a8_l1527_152770

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = r * a n

theorem max_value_a4a8 (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_cond : a 2 * a 6 + a 5 * a 11 = 16) : 
    (∀ x, a 4 * a 8 ≤ x → x = 8) :=
  sorry

end NUMINAMATH_CALUDE_max_value_a4a8_l1527_152770


namespace NUMINAMATH_CALUDE_eighth_number_is_four_l1527_152792

/-- A sequence of 12 numbers satisfying the given conditions -/
def SpecialSequence : Type := 
  {s : Fin 12 → ℕ // s 0 = 5 ∧ s 11 = 10 ∧ ∀ i, i < 10 → s i + s (i + 1) + s (i + 2) = 19}

/-- The theorem stating that the 8th number (index 7) in the sequence is 4 -/
theorem eighth_number_is_four (s : SpecialSequence) : s.val 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_eighth_number_is_four_l1527_152792


namespace NUMINAMATH_CALUDE_school_classes_l1527_152707

theorem school_classes (daily_usage_per_class : ℕ) (weekly_usage_total : ℕ) (school_days_per_week : ℕ) :
  daily_usage_per_class = 200 →
  weekly_usage_total = 9000 →
  school_days_per_week = 5 →
  weekly_usage_total / school_days_per_week / daily_usage_per_class = 9 := by
sorry

end NUMINAMATH_CALUDE_school_classes_l1527_152707


namespace NUMINAMATH_CALUDE_grey_pairs_coincide_l1527_152727

/-- Represents the number of triangles of each color in one half of the shape -/
structure TriangleCounts where
  orange : Nat
  green : Nat
  grey : Nat

/-- Represents the number of pairs of triangles that coincide when folded -/
structure CoincidingPairs where
  orange : Nat
  green : Nat
  orangeGrey : Nat

theorem grey_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) :
  counts.orange = 4 →
  counts.green = 6 →
  counts.grey = 9 →
  pairs.orange = 3 →
  pairs.green = 4 →
  pairs.orangeGrey = 1 →
  ∃ (grey_pairs : Nat), grey_pairs = 6 ∧ 
    grey_pairs = counts.grey - (pairs.orangeGrey + (counts.green - 2 * pairs.green)) :=
by sorry

end NUMINAMATH_CALUDE_grey_pairs_coincide_l1527_152727


namespace NUMINAMATH_CALUDE_betty_oranges_purchase_l1527_152713

/-- Represents the problem of determining how many kg of oranges Betty bought. -/
theorem betty_oranges_purchase :
  ∀ (orange_kg : ℝ) (apple_kg : ℝ) (orange_cost : ℝ) (apple_price_per_kg : ℝ),
    apple_kg = 3 →
    orange_cost = 12 →
    apple_price_per_kg = 2 →
    apple_price_per_kg * 2 = orange_cost / orange_kg →
    orange_kg = 3 := by
  sorry

end NUMINAMATH_CALUDE_betty_oranges_purchase_l1527_152713


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l1527_152745

theorem alcohol_solution_percentage 
  (initial_volume : ℝ) 
  (initial_percentage : ℝ) 
  (added_alcohol : ℝ) 
  (added_water : ℝ) 
  (h1 : initial_volume = 40)
  (h2 : initial_percentage = 5)
  (h3 : added_alcohol = 5.5)
  (h4 : added_water = 4.5) :
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let new_alcohol := initial_alcohol + added_alcohol
  let new_volume := initial_volume + added_alcohol + added_water
  let new_percentage := (new_alcohol / new_volume) * 100
  new_percentage = 15 := by
sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l1527_152745


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_l1527_152761

/-- An isosceles triangle with perimeter 16 and one side 4 has a base of 4 -/
theorem isosceles_triangle_base (a b c : ℝ) : 
  a + b + c = 16 →  -- perimeter is 16
  a = b →           -- isosceles triangle condition
  a = 4 →           -- one side is 4
  c = 4 :=          -- prove that the base is 4
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_l1527_152761


namespace NUMINAMATH_CALUDE_jakes_weight_l1527_152732

theorem jakes_weight (jake kendra : ℕ) 
  (h1 : jake - 8 = 2 * kendra) 
  (h2 : jake + kendra = 287) : 
  jake = 194 := by sorry

end NUMINAMATH_CALUDE_jakes_weight_l1527_152732


namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l1527_152706

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) :
  x = Real.sqrt (8 + 9 / 16) → x = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l1527_152706


namespace NUMINAMATH_CALUDE_h_value_l1527_152733

-- Define polynomials f and h
variable (f h : ℝ → ℝ)

-- Define the conditions
axiom f_def : ∀ x, f x = x^4 - 2*x^3 + x - 1
axiom sum_eq : ∀ x, f x + h x = 3*x^2 + 5*x - 4

-- State the theorem
theorem h_value : ∀ x, h x = -x^4 + 2*x^3 + 3*x^2 + 4*x - 3 :=
sorry

end NUMINAMATH_CALUDE_h_value_l1527_152733


namespace NUMINAMATH_CALUDE_baseball_gear_cost_l1527_152768

def initial_amount : ℕ := 67
def amount_left : ℕ := 33

theorem baseball_gear_cost :
  initial_amount - amount_left = 34 :=
by sorry

end NUMINAMATH_CALUDE_baseball_gear_cost_l1527_152768


namespace NUMINAMATH_CALUDE_geometric_sequence_tangent_l1527_152749

/-- Given a geometric sequence {a_n} where a_2 * a_3 * a_4 = -a_7^2 = -64,
    prove that tan((a_4 * a_6 / 3) * π) = -√3 -/
theorem geometric_sequence_tangent (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n) →  -- geometric sequence condition
  a 2 * a 3 * a 4 = -a 7^2 →                            -- given condition
  a 7^2 = 64 →                                          -- given condition
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_tangent_l1527_152749


namespace NUMINAMATH_CALUDE_cycle_original_price_l1527_152726

/-- Given a cycle sold at a 12% loss for Rs. 1408, prove that the original price was Rs. 1600. -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) : 
  selling_price = 1408 → 
  loss_percentage = 12 → 
  (1 - loss_percentage / 100) * 1600 = selling_price :=
by sorry

end NUMINAMATH_CALUDE_cycle_original_price_l1527_152726


namespace NUMINAMATH_CALUDE_lower_limit_of_g_l1527_152747

-- Define the function f(n)
def f (n : ℕ) : ℕ := Finset.prod (Finset.range (n^2 - 3)) (λ i => i + 4)

-- Define the function g(n) with a parameter m for the lower limit
def g (n m : ℕ) : ℕ := Finset.prod (Finset.range (n - m + 1)) (λ i => (i + m)^2)

-- State the theorem
theorem lower_limit_of_g : ∃ m : ℕ, 
  m = 2 ∧ 
  (∀ n : ℕ, n ≥ m → g n m ≠ 0) ∧
  (∃ k : ℕ, (f 3 / g 3 m).factorization 2 = 4) :=
sorry

end NUMINAMATH_CALUDE_lower_limit_of_g_l1527_152747


namespace NUMINAMATH_CALUDE_tournament_outcomes_l1527_152724

/-- Represents a knockout tournament with 6 players -/
structure Tournament :=
  (num_players : Nat)
  (num_games : Nat)

/-- The number of possible outcomes for each game -/
def outcomes_per_game : Nat := 2

/-- Theorem stating that the number of possible prize orders is 32 -/
theorem tournament_outcomes (t : Tournament) (h1 : t.num_players = 6) (h2 : t.num_games = 5) : 
  outcomes_per_game ^ t.num_games = 32 := by
  sorry

#eval outcomes_per_game ^ 5

end NUMINAMATH_CALUDE_tournament_outcomes_l1527_152724


namespace NUMINAMATH_CALUDE_initial_puppies_count_l1527_152746

/-- The number of puppies initially in the shelter -/
def initial_puppies : ℕ := sorry

/-- The number of additional puppies brought in -/
def additional_puppies : ℕ := 3

/-- The number of puppies adopted per day -/
def adoptions_per_day : ℕ := 3

/-- The number of days it takes for all puppies to be adopted -/
def days_to_adopt_all : ℕ := 2

/-- Theorem stating that the initial number of puppies is 3 -/
theorem initial_puppies_count : initial_puppies = 3 := by sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l1527_152746


namespace NUMINAMATH_CALUDE_green_team_opponent_score_l1527_152704

/-- The final score of a team's opponent given the team's score and lead -/
def opponent_score (team_score : ℕ) (lead : ℕ) : ℕ :=
  team_score - lead

/-- Theorem: Given Green Team's score of 39 and lead of 29, their opponent's score is 10 -/
theorem green_team_opponent_score :
  opponent_score 39 29 = 10 := by
  sorry

end NUMINAMATH_CALUDE_green_team_opponent_score_l1527_152704


namespace NUMINAMATH_CALUDE_average_speed_theorem_l1527_152731

theorem average_speed_theorem (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 ∧ 
  first_half_speed = 80 ∧ 
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 := by
  sorry

#check average_speed_theorem

end NUMINAMATH_CALUDE_average_speed_theorem_l1527_152731


namespace NUMINAMATH_CALUDE_ranas_speed_l1527_152751

/-- Proves that Rana's speed is 5 kmph given the problem conditions -/
theorem ranas_speed (circumference : ℝ) (ajith_speed : ℝ) (meeting_time : ℝ) 
  (h1 : circumference = 115)
  (h2 : ajith_speed = 4)
  (h3 : meeting_time = 115) :
  ∃ v : ℝ, v = 5 ∧ 
    (v * meeting_time - ajith_speed * meeting_time) / circumference = 1 :=
by sorry

end NUMINAMATH_CALUDE_ranas_speed_l1527_152751


namespace NUMINAMATH_CALUDE_sin_product_equals_one_thirty_second_l1527_152718

theorem sin_product_equals_one_thirty_second :
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * Real.sin (72 * π / 180) * Real.sin (84 * π / 180) = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_thirty_second_l1527_152718


namespace NUMINAMATH_CALUDE_unique_zero_implies_a_range_l1527_152774

/-- The cubic function f(x) = ax^3 - 3x^2 + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem unique_zero_implies_a_range 
  (a : ℝ) 
  (h_unique : ∃! x₀ : ℝ, f a x₀ = 0) 
  (h_neg : ∃ x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0) :
  a > 2 :=
sorry

end NUMINAMATH_CALUDE_unique_zero_implies_a_range_l1527_152774


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l1527_152794

theorem quadratic_root_zero (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + x + m^2 - 1 = 0) ∧
  ((m - 1) * 0^2 + 0 + m^2 - 1 = 0) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l1527_152794


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1527_152719

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1527_152719


namespace NUMINAMATH_CALUDE_reading_time_calculation_l1527_152766

theorem reading_time_calculation (total_time math_time spelling_time history_time science_time piano_time break_time : ℕ)
  (h1 : total_time = 180)
  (h2 : math_time = 25)
  (h3 : spelling_time = 30)
  (h4 : history_time = 20)
  (h5 : science_time = 15)
  (h6 : piano_time = 30)
  (h7 : break_time = 20) :
  total_time - (math_time + spelling_time + history_time + science_time + piano_time + break_time) = 40 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l1527_152766


namespace NUMINAMATH_CALUDE_composite_sum_l1527_152712

theorem composite_sum (x y : ℕ) (h1 : x > 1) (h2 : y > 1) 
  (h3 : (x^2 + y^2 - 1) % (x + y - 1) = 0) : 
  ¬ Nat.Prime (x + y - 1) := by
sorry

end NUMINAMATH_CALUDE_composite_sum_l1527_152712


namespace NUMINAMATH_CALUDE_line_perp_plane_criterion_l1527_152748

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line_line : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_criterion 
  (α β γ : Plane) (m n l : Line) :
  perp_line_plane n α → 
  perp_line_plane n β → 
  perp_line_plane m α → 
  perp_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_criterion_l1527_152748


namespace NUMINAMATH_CALUDE_december_sales_fraction_l1527_152760

theorem december_sales_fraction (average_sales : ℝ) (h : average_sales > 0) :
  let january_to_november_sales := 11 * average_sales
  let december_sales := 5 * average_sales
  let total_annual_sales := january_to_november_sales + december_sales
  december_sales / total_annual_sales = 5 / 16 := by
sorry

end NUMINAMATH_CALUDE_december_sales_fraction_l1527_152760


namespace NUMINAMATH_CALUDE_randy_initial_amount_l1527_152701

/-- Represents Randy's piggy bank finances over a year -/
structure PiggyBank where
  initial_amount : ℕ
  monthly_deposit : ℕ
  store_visits : ℕ
  min_cost_per_visit : ℕ
  max_cost_per_visit : ℕ
  final_balance : ℕ

/-- Theorem stating that Randy's initial amount was $104 -/
theorem randy_initial_amount (pb : PiggyBank) 
  (h1 : pb.monthly_deposit = 50)
  (h2 : pb.store_visits = 200)
  (h3 : pb.min_cost_per_visit = 2)
  (h4 : pb.max_cost_per_visit = 3)
  (h5 : pb.final_balance = 104) :
  pb.initial_amount = 104 := by
  sorry

#check randy_initial_amount

end NUMINAMATH_CALUDE_randy_initial_amount_l1527_152701


namespace NUMINAMATH_CALUDE_paige_catfish_l1527_152705

/-- The number of goldfish Paige initially raised -/
def initial_goldfish : ℕ := 7

/-- The number of fish that disappeared -/
def disappeared_fish : ℕ := 4

/-- The number of fish left -/
def remaining_fish : ℕ := 15

/-- The number of catfish Paige initially raised -/
def initial_catfish : ℕ := initial_goldfish + disappeared_fish + remaining_fish - initial_goldfish

theorem paige_catfish : initial_catfish = 12 := by
  sorry

end NUMINAMATH_CALUDE_paige_catfish_l1527_152705


namespace NUMINAMATH_CALUDE_x₁x₂_equals_1008_l1527_152764

noncomputable def x₁ : ℝ := Real.exp (Real.log 2 * 1008 / (Real.log 2 * Real.exp (Real.log 2 * 1008 / (Real.log 2 * Real.exp (Real.log 2 * 1008 / Real.log 2)))))

noncomputable def x₂ : ℝ := Real.log 2 * 1008 / (Real.log 2 * Real.exp (Real.log 2 * 1008 / Real.log 2))

theorem x₁x₂_equals_1008 :
  x₁ * Real.log x₁ / Real.log 2 = 1008 ∧
  x₂ * 2^x₂ = 1008 →
  x₁ * x₂ = 1008 := by
  sorry

end NUMINAMATH_CALUDE_x₁x₂_equals_1008_l1527_152764


namespace NUMINAMATH_CALUDE_expression_factorization_l1527_152782

theorem expression_factorization (y : ℝ) : 
  5 * y * (y - 2) + 10 * (y - 2) - 15 * (y - 2) = 5 * (y - 2) * (y - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1527_152782


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_equation_l1527_152750

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ ∀ x, f x ≠ 0 := by sorry

theorem negation_of_cubic_equation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ ∀ x : ℝ, x^3 - 2*x + 1 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_equation_l1527_152750


namespace NUMINAMATH_CALUDE_child_tickets_sold_l1527_152786

theorem child_tickets_sold (adult_price child_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : total_revenue = 5100) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 400 := by
  sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l1527_152786


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1527_152737

/-- Theorem: In a triangle ABC where angle A is x degrees, angle B is 2x degrees, 
    and angle C is 45°, the value of x is 45°. -/
theorem triangle_angle_calculation (x : ℝ) : 
  x > 0 ∧ x < 180 ∧ 2*x < 180 ∧ 
  x + 2*x + 45 = 180 → 
  x = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1527_152737


namespace NUMINAMATH_CALUDE_relationship_holds_l1527_152736

def x : Fin 5 → ℕ
  | ⟨0, _⟩ => 1
  | ⟨1, _⟩ => 2
  | ⟨2, _⟩ => 3
  | ⟨3, _⟩ => 4
  | ⟨4, _⟩ => 5

def y : Fin 5 → ℕ
  | ⟨0, _⟩ => 4
  | ⟨1, _⟩ => 15
  | ⟨2, _⟩ => 40
  | ⟨3, _⟩ => 85
  | ⟨4, _⟩ => 156

theorem relationship_holds : ∀ i : Fin 5, y i = (x i)^3 + 2*(x i) + 1 := by
  sorry

end NUMINAMATH_CALUDE_relationship_holds_l1527_152736


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l1527_152753

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 17600

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 1.76
    exponent := 4
    is_valid := by sorry }

/-- Theorem stating that the proposed notation correctly represents the original number -/
theorem scientific_notation_correct :
  (proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent) = original_number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l1527_152753


namespace NUMINAMATH_CALUDE_some_number_value_l1527_152777

theorem some_number_value (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 49 * some_number * 25) :
  some_number = 45 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1527_152777


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1527_152767

/-- Given vectors a, b, and c in ℝ², prove that if k*a + 2*b is perpendicular to c,
    then k = -17/3 -/
theorem perpendicular_vectors_k_value (a b c : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (3, 4))
    (h2 : b = (-1, 5))
    (h3 : c = (2, -3))
    (h4 : (k * a.1 + 2 * b.1, k * a.2 + 2 * b.2) • c = 0) :
  k = -17/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1527_152767


namespace NUMINAMATH_CALUDE_translation_exists_l1527_152769

-- Define the set of line segments
def LineSegments : Set (Set ℝ) := sorry

-- Define the property that the total length of line segments is less than 1
def TotalLengthLessThanOne (segments : Set (Set ℝ)) : Prop := sorry

-- Define a set of n points on the line
def Points (n : ℕ) : Set ℝ := sorry

-- Define a translation vector
def TranslationVector : ℝ := sorry

-- Define the property that the translation vector length does not exceed n/2
def TranslationLengthValid (v : ℝ) (n : ℕ) : Prop := 
  abs v ≤ n / 2

-- Define the translated points
def TranslatedPoints (points : Set ℝ) (v : ℝ) : Set ℝ := sorry

-- Define the property that no translated point intersects with any line segment
def NoIntersection (translatedPoints : Set ℝ) (segments : Set (Set ℝ)) : Prop := sorry

-- The main theorem
theorem translation_exists (n : ℕ) (segments : Set (Set ℝ)) (points : Set ℝ) 
  (h1 : TotalLengthLessThanOne segments) 
  (h2 : points = Points n) :
  ∃ v : ℝ, TranslationLengthValid v n ∧ 
    NoIntersection (TranslatedPoints points v) segments := by sorry

end NUMINAMATH_CALUDE_translation_exists_l1527_152769


namespace NUMINAMATH_CALUDE_singer_arrangement_count_l1527_152763

/-- The number of singers -/
def n : ℕ := 6

/-- The number of singers with specific arrangement requirements (A, B, C) -/
def k : ℕ := 3

/-- The number of valid arrangements of A, B, C (A-B-C, A-C-B, B-C-A, C-B-A) -/
def valid_abc_arrangements : ℕ := 4

/-- The total number of arrangements of n singers -/
def total_arrangements : ℕ := n.factorial

/-- The number of arrangements of k singers -/
def k_arrangements : ℕ := k.factorial

theorem singer_arrangement_count :
  (valid_abc_arrangements * total_arrangements / k_arrangements : ℕ) = 480 := by
  sorry

end NUMINAMATH_CALUDE_singer_arrangement_count_l1527_152763


namespace NUMINAMATH_CALUDE_intersection_A_B_when_a_4_A_subset_B_condition_l1527_152723

-- Define the sets A and B
def A : Set ℝ := {x | (1 - x) / (x - 7) > 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x - a^2 - 2*a < 0}

-- Theorem 1: Intersection of A and B when a = 4
theorem intersection_A_B_when_a_4 : A ∩ B 4 = {x | 1 < x ∧ x < 6} := by sorry

-- Theorem 2: Condition for A to be a subset of B
theorem A_subset_B_condition (a : ℝ) : A ⊆ B a ↔ a ≤ -7 ∨ a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_a_4_A_subset_B_condition_l1527_152723


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_interval_l1527_152757

theorem quadratic_inequality_solution_interval (k : ℝ) : 
  (k > 0 ∧ ∃ x : ℝ, x^2 - 8*x + k < 0) ↔ (0 < k ∧ k < 16) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_interval_l1527_152757


namespace NUMINAMATH_CALUDE_wilted_ratio_after_first_night_l1527_152765

/-- Represents the number of roses at different stages --/
structure RoseCount where
  initial : ℕ
  afterFirstNight : ℕ
  afterSecondNight : ℕ

/-- Calculates the ratio of wilted flowers to total flowers after the first night --/
def wiltedRatio (rc : RoseCount) : Rat :=
  (rc.initial - rc.afterFirstNight) / rc.initial

theorem wilted_ratio_after_first_night
  (rc : RoseCount)
  (h1 : rc.initial = 36)
  (h2 : rc.afterSecondNight = 9)
  (h3 : rc.afterFirstNight = 2 * rc.afterSecondNight) :
  wiltedRatio rc = 1/2 := by
  sorry

#eval wiltedRatio { initial := 36, afterFirstNight := 18, afterSecondNight := 9 }

end NUMINAMATH_CALUDE_wilted_ratio_after_first_night_l1527_152765


namespace NUMINAMATH_CALUDE_angelinas_speed_to_gym_l1527_152721

-- Define the constants
def distance_home_to_grocery : ℝ := 200
def distance_grocery_to_gym : ℝ := 300
def time_difference : ℝ := 50

-- Define the variables
variable (v : ℝ) -- Speed from home to grocery

-- Define the theorem
theorem angelinas_speed_to_gym :
  (distance_home_to_grocery / v) - (distance_grocery_to_gym / (2 * v)) = time_difference →
  2 * v = 2 := by
  sorry

end NUMINAMATH_CALUDE_angelinas_speed_to_gym_l1527_152721


namespace NUMINAMATH_CALUDE_unique_solution_two_power_minus_three_power_l1527_152790

theorem unique_solution_two_power_minus_three_power : 
  ∀ m n : ℕ+, 2^(m:ℕ) - 3^(n:ℕ) = 7 → m = 4 ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_two_power_minus_three_power_l1527_152790


namespace NUMINAMATH_CALUDE_farm_chickens_count_l1527_152791

/-- Proves that the total number of chickens on a farm is 69, given the number of ducks, geese, and their relationships to hens and roosters. -/
theorem farm_chickens_count (ducks geese : ℕ) 
  (h1 : ducks = 45)
  (h2 : geese = 28)
  (h3 : ∃ hens : ℕ, hens = ducks - 13)
  (h4 : ∃ roosters : ℕ, roosters = geese + 9) :
  ∃ total_chickens : ℕ, total_chickens = 69 ∧ 
    ∃ (hens roosters : ℕ), 
      hens = ducks - 13 ∧ 
      roosters = geese + 9 ∧ 
      total_chickens = hens + roosters := by
sorry

end NUMINAMATH_CALUDE_farm_chickens_count_l1527_152791


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l1527_152714

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x : ℝ, (x - 1)^2 < 9 → (x + 2) * (x + a) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x + a) < 0 ∧ (x - 1)^2 ≥ 9) →
  a < -4 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l1527_152714


namespace NUMINAMATH_CALUDE_paint_joined_cubes_paint_divided_cube_cube_division_l1527_152796

-- Constants
def paint_coverage : ℝ := 100 -- 1 mL covers 100 cm²

-- Theorem 1
theorem paint_joined_cubes (small_edge large_edge : ℝ) (h1 : small_edge = 10) (h2 : large_edge = 20) :
  (6 * small_edge^2 + 6 * large_edge^2 - 2 * small_edge^2) / paint_coverage = 28 :=
sorry

-- Theorem 2
theorem paint_divided_cube (original_paint : ℝ) (h : original_paint = 54) :
  2 * (original_paint / 6) = 18 :=
sorry

-- Theorem 3
theorem cube_division (original_paint additional_paint : ℝ) (n : ℕ)
  (h1 : original_paint = 54) (h2 : additional_paint = 216) :
  6 * (original_paint / 6) * n = original_paint + additional_paint →
  n = 5 :=
sorry

end NUMINAMATH_CALUDE_paint_joined_cubes_paint_divided_cube_cube_division_l1527_152796


namespace NUMINAMATH_CALUDE_product_of_sums_of_squares_l1527_152776

theorem product_of_sums_of_squares (a b c d : ℤ) :
  ∃ x y : ℤ, (a^2 + b^2) * (c^2 + d^2) = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_of_squares_l1527_152776


namespace NUMINAMATH_CALUDE_expression_equals_one_l1527_152744

theorem expression_equals_one (x z : ℝ) (h1 : x ≠ z) (h2 : x ≠ -z) :
  (x / (x - z) - z / (x + z)) / (z / (x - z) + x / (x + z)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1527_152744


namespace NUMINAMATH_CALUDE_polynomial_product_equality_l1527_152738

theorem polynomial_product_equality (x : ℝ) : 
  (x^4 + 50*x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_equality_l1527_152738


namespace NUMINAMATH_CALUDE_apple_selling_price_l1527_152779

/-- The selling price of an apple given its cost price and loss ratio -/
def selling_price (cost_price : ℚ) (loss_ratio : ℚ) : ℚ :=
  cost_price * (1 - loss_ratio)

/-- Theorem stating the selling price of an apple with given conditions -/
theorem apple_selling_price :
  let cost_price : ℚ := 20
  let loss_ratio : ℚ := 1/6
  selling_price cost_price loss_ratio = 50/3 := by
sorry

end NUMINAMATH_CALUDE_apple_selling_price_l1527_152779


namespace NUMINAMATH_CALUDE_function_growth_l1527_152729

/-- Given a differentiable function f: ℝ → ℝ such that f'(x) > f(x) for all x,
    prove that f(2012) > e^2012 * f(0) -/
theorem function_growth (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : ∀ x, deriv f x > f x) : f 2012 > Real.exp 2012 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_growth_l1527_152729


namespace NUMINAMATH_CALUDE_volleyball_game_employees_l1527_152772

theorem volleyball_game_employees (managers : ℕ) (teams : ℕ) (people_per_team : ℕ) :
  managers = 3 →
  teams = 3 →
  people_per_team = 2 →
  teams * people_per_team - managers = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_game_employees_l1527_152772


namespace NUMINAMATH_CALUDE_additional_time_is_twelve_minutes_l1527_152742

/-- Represents the time (in hours) it takes for a worker to complete the job alone. -/
def completion_time (worker : ℕ) : ℚ :=
  match worker with
  | 1 => 4    -- P's completion time
  | 2 => 15   -- Q's completion time
  | _ => 0    -- Invalid worker

/-- Calculates the portion of the job completed by both workers in 3 hours. -/
def portion_completed : ℚ :=
  3 * ((1 / completion_time 1) + (1 / completion_time 2))

/-- Calculates the remaining portion of the job after 3 hours of joint work. -/
def remaining_portion : ℚ :=
  1 - portion_completed

/-- Calculates the additional time (in hours) needed for P to complete the remaining portion. -/
def additional_time : ℚ :=
  remaining_portion * completion_time 1

/-- The main theorem stating that the additional time for P to finish the job is 12 minutes. -/
theorem additional_time_is_twelve_minutes : additional_time * 60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_time_is_twelve_minutes_l1527_152742


namespace NUMINAMATH_CALUDE_daniels_animals_legs_l1527_152789

/-- The number of legs an animal has -/
def legs (animal : String) : ℕ :=
  match animal with
  | "horse" => 4
  | "dog" => 4
  | "cat" => 4
  | "turtle" => 4
  | "goat" => 4
  | _ => 0

/-- The number of animals Daniel has -/
def animal_count (animal : String) : ℕ :=
  match animal with
  | "horse" => 2
  | "dog" => 5
  | "cat" => 7
  | "turtle" => 3
  | "goat" => 1
  | _ => 0

/-- The total number of legs of all animals -/
def total_legs : ℕ :=
  (animal_count "horse" * legs "horse") +
  (animal_count "dog" * legs "dog") +
  (animal_count "cat" * legs "cat") +
  (animal_count "turtle" * legs "turtle") +
  (animal_count "goat" * legs "goat")

theorem daniels_animals_legs : total_legs = 72 := by
  sorry

end NUMINAMATH_CALUDE_daniels_animals_legs_l1527_152789


namespace NUMINAMATH_CALUDE_valid_arrays_l1527_152700

def is_valid_array (p q r : ℕ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧
  p ≥ q ∧ q ≥ r ∧
  ((Prime p ∧ Prime q) ∨ (Prime p ∧ Prime r) ∨ (Prime q ∧ Prime r)) ∧
  ∃ k : ℕ, k > 0 ∧ (p + q + r)^2 = k * (p * q * r)

theorem valid_arrays :
  ∀ p q r : ℕ, is_valid_array p q r ↔
    (p = 3 ∧ q = 3 ∧ r = 3) ∨
    (p = 2 ∧ q = 2 ∧ r = 4) ∨
    (p = 3 ∧ q = 3 ∧ r = 12) ∨
    (p = 3 ∧ q = 2 ∧ r = 1) ∨
    (p = 3 ∧ q = 2 ∧ r = 25) :=
by sorry

#check valid_arrays

end NUMINAMATH_CALUDE_valid_arrays_l1527_152700


namespace NUMINAMATH_CALUDE_second_friend_shells_l1527_152758

theorem second_friend_shells (jovana_initial : ℕ) (first_friend : ℕ) (total : ℕ) : 
  jovana_initial = 5 → first_friend = 15 → total = 37 → 
  total - (jovana_initial + first_friend) = 17 := by
sorry

end NUMINAMATH_CALUDE_second_friend_shells_l1527_152758


namespace NUMINAMATH_CALUDE_range_of_a_l1527_152783

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 4) → -3 ≤ a ∧ a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1527_152783


namespace NUMINAMATH_CALUDE_function_domain_implies_k_range_l1527_152798

/-- Given a function f(x) = √(kx² + kx + 3) with domain ℝ, k must be in [0, 12] -/
theorem function_domain_implies_k_range (k : ℝ) : 
  (∀ x, ∃ y, y = Real.sqrt (k * x^2 + k * x + 3)) → 0 ≤ k ∧ k ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_function_domain_implies_k_range_l1527_152798


namespace NUMINAMATH_CALUDE_folders_needed_l1527_152734

def initial_files : Real := 93.0
def additional_files : Real := 21.0
def files_per_folder : Real := 8.0

theorem folders_needed : 
  ∃ (n : ℕ), n = Int.ceil ((initial_files + additional_files) / files_per_folder) ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_folders_needed_l1527_152734


namespace NUMINAMATH_CALUDE_investor_purchase_price_l1527_152725

/-- The dividend rate paid by the company -/
def dividend_rate : ℚ := 185 / 1000

/-- The face value of each share -/
def face_value : ℚ := 50

/-- The return on investment received by the investor -/
def roi : ℚ := 1 / 4

/-- The purchase price per share -/
def purchase_price : ℚ := 37

theorem investor_purchase_price : 
  dividend_rate * face_value / purchase_price = roi := by sorry

end NUMINAMATH_CALUDE_investor_purchase_price_l1527_152725


namespace NUMINAMATH_CALUDE_heart_equal_set_is_four_lines_l1527_152795

-- Define the ♥ operation
def heart (a b : ℝ) : ℝ := a^3 * b - a^2 * b^2 + a * b^3

-- Define the set of points satisfying x ♥ y = y ♥ x
def heart_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | heart p.1 p.2 = heart p.2 p.1}

-- Theorem statement
theorem heart_equal_set_is_four_lines :
  heart_equal_set = {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2} :=
by sorry

end NUMINAMATH_CALUDE_heart_equal_set_is_four_lines_l1527_152795


namespace NUMINAMATH_CALUDE_melanie_initial_plums_l1527_152755

/-- The number of plums Melanie initially picked -/
def initial_plums : ℕ := sorry

/-- The number of plums Melanie gave to Sam -/
def plums_given : ℕ := 3

/-- The number of plums Melanie has left -/
def plums_left : ℕ := 4

/-- Theorem: Melanie initially picked 7 plums -/
theorem melanie_initial_plums : initial_plums = 7 := by
  sorry

end NUMINAMATH_CALUDE_melanie_initial_plums_l1527_152755


namespace NUMINAMATH_CALUDE_farmland_area_l1527_152717

theorem farmland_area (lizzie_group_area other_group_area remaining_area : ℕ) 
  (h1 : lizzie_group_area = 250)
  (h2 : other_group_area = 265)
  (h3 : remaining_area = 385) :
  lizzie_group_area + other_group_area + remaining_area = 900 := by
  sorry

end NUMINAMATH_CALUDE_farmland_area_l1527_152717


namespace NUMINAMATH_CALUDE_dragon_lion_equivalence_l1527_152787

theorem dragon_lion_equivalence (P Q : Prop) : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by sorry

end NUMINAMATH_CALUDE_dragon_lion_equivalence_l1527_152787


namespace NUMINAMATH_CALUDE_contest_order_l1527_152710

/-- Represents the scores of contestants in a mathematics competition. -/
structure ContestScores where
  adam : ℝ
  bob : ℝ
  charles : ℝ
  david : ℝ
  nonnegative : adam ≥ 0 ∧ bob ≥ 0 ∧ charles ≥ 0 ∧ david ≥ 0
  sum_equality : adam + bob = charles + david
  interchange_inequality : charles + adam > bob + david
  charles_exceeds_sum : charles > adam + bob

/-- Proves that given the contest conditions, the order of scores from highest to lowest is Charles, Adam, Bob, David. -/
theorem contest_order (scores : ContestScores) : 
  scores.charles > scores.adam ∧ 
  scores.adam > scores.bob ∧ 
  scores.bob > scores.david := by
  sorry


end NUMINAMATH_CALUDE_contest_order_l1527_152710


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1527_152743

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a+1)*x + a*b > 0 ↔ x < -1 ∨ x > 4) → a + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1527_152743


namespace NUMINAMATH_CALUDE_sphere_cube_ratios_l1527_152788

theorem sphere_cube_ratios (R : ℝ) (a : ℝ) (h : a = 2 * R / Real.sqrt 3) :
  let sphere_surface := 4 * Real.pi * R^2
  let cube_surface := 6 * a^2
  let sphere_volume := 4 / 3 * Real.pi * R^3
  let cube_volume := a^3
  (sphere_surface / cube_surface = Real.pi / 2) ∧
  (sphere_volume / cube_volume = Real.pi * Real.sqrt 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_sphere_cube_ratios_l1527_152788


namespace NUMINAMATH_CALUDE_f_properties_l1527_152756

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt ((1 - x^2) / (1 + x^2)) + a * Real.sqrt ((1 + x^2) / (1 - x^2))

theorem f_properties (a : ℝ) (h : a > 0) :
  -- Function domain
  ∀ x : ℝ, -1 < x ∧ x < 1 →
  -- 1. Minimum value when a = 1
  (a = 1 → ∀ x : ℝ, -1 < x ∧ x < 1 → f 1 x ≥ 2) ∧
  -- 2. Monotonicity when a = 1
  (a = 1 → ∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y < 1 → f 1 x < f 1 y) ∧
  -- 3. Range of a for triangle formation
  (∀ r s t : ℝ, -2*Real.sqrt 5/5 ≤ r ∧ r ≤ 2*Real.sqrt 5/5 ∧
                -2*Real.sqrt 5/5 ≤ s ∧ s ≤ 2*Real.sqrt 5/5 ∧
                -2*Real.sqrt 5/5 ≤ t ∧ t ≤ 2*Real.sqrt 5/5 →
    f a r + f a s > f a t ∧ f a s + f a t > f a r ∧ f a t + f a r > f a s) ↔
  (1/15 < a ∧ a < 5/3) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1527_152756


namespace NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l1527_152740

theorem cos_squared_alpha_minus_pi_fourth (α : Real) 
  (h : Real.sin (2 * α) = 1 / 3) : 
  Real.cos (α - π / 4) ^ 2 = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l1527_152740


namespace NUMINAMATH_CALUDE_complement_of_M_l1527_152780

-- Define the universal set U as the set of real numbers
def U := Set ℝ

-- Define the set M
def M : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_of_M (x : ℝ) : 
  x ∈ (Set.univ \ M) ↔ x ≤ -1 ∨ 2 < x := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l1527_152780


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l1527_152715

theorem simplify_trigonometric_expression (α : Real) (h : π < α ∧ α < 2*π) : 
  ((1 + Real.sin α + Real.cos α) * (Real.sin (α/2) - Real.cos (α/2))) / 
  Real.sqrt (2 + 2 * Real.cos α) = Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l1527_152715


namespace NUMINAMATH_CALUDE_red_balls_count_l1527_152708

/-- The number of times 18 balls are taken out after the initial 60 balls -/
def x : ℕ := sorry

/-- The total number of balls in the bag -/
def total_balls : ℕ := 60 + 18 * x

/-- The total number of red balls in the bag -/
def red_balls : ℕ := 56 + 14 * x

/-- The proportion of red balls to total balls is 4/5 -/
axiom proportion_axiom : (red_balls : ℚ) / total_balls = 4 / 5

theorem red_balls_count : red_balls = 336 := by sorry

end NUMINAMATH_CALUDE_red_balls_count_l1527_152708


namespace NUMINAMATH_CALUDE_simplest_common_denominator_l1527_152711

-- Define the fractions
def fraction1 (x y : ℚ) : ℚ := 1 / (2 * x^2 * y)
def fraction2 (x y : ℚ) : ℚ := 1 / (6 * x * y^3)

-- Define the common denominator
def common_denominator (x y : ℚ) : ℚ := 6 * x^2 * y^3

-- Theorem statement
theorem simplest_common_denominator (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (a b : ℚ), 
    fraction1 x y = a / common_denominator x y ∧
    fraction2 x y = b / common_denominator x y ∧
    (∀ (c : ℚ), c > 0 → 
      (∃ (d e : ℚ), fraction1 x y = d / c ∧ fraction2 x y = e / c) →
      c ≥ common_denominator x y) :=
sorry

end NUMINAMATH_CALUDE_simplest_common_denominator_l1527_152711


namespace NUMINAMATH_CALUDE_unique_solution_l1527_152793

-- Define the equation
def equation (x a : ℝ) : Prop :=
  3 * x^2 + 2 * a * x - a^2 = Real.log ((x - a) / (2 * x))

-- Define the domain conditions
def domain_conditions (x a : ℝ) : Prop :=
  x - a > 0 ∧ 2 * x > 0

-- Theorem statement
theorem unique_solution (a : ℝ) (h : a ≠ 0) :
  ∃! x : ℝ, equation x a ∧ domain_conditions x a :=
by
  -- The unique solution is x = -a
  use -a
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_unique_solution_l1527_152793


namespace NUMINAMATH_CALUDE_parcel_cost_correct_l1527_152720

/-- The cost function for sending a parcel post package -/
def parcel_cost (P : ℕ) : ℕ :=
  12 + 5 * P

/-- Theorem stating the correctness of the parcel cost function -/
theorem parcel_cost_correct (P : ℕ) (h : P ≥ 1) :
  parcel_cost P = 15 + 5 * (P - 1) + 2 :=
by sorry

end NUMINAMATH_CALUDE_parcel_cost_correct_l1527_152720


namespace NUMINAMATH_CALUDE_isosceles_triangle_areas_sum_l1527_152773

/-- Represents the areas of right isosceles triangles constructed on the sides of a right triangle -/
structure TriangleAreas where
  A : ℝ  -- Area of the isosceles triangle on side 5
  B : ℝ  -- Area of the isosceles triangle on side 12
  C : ℝ  -- Area of the isosceles triangle on side 13

/-- Theorem: For a right triangle with sides 5, 12, and 13, 
    if right isosceles triangles are constructed on each side, 
    then the sum of the areas of the triangles on the two shorter sides 
    equals the area of the triangle on the hypotenuse -/
theorem isosceles_triangle_areas_sum (areas : TriangleAreas) 
  (h1 : areas.A = (5 * 5) / 2)
  (h2 : areas.B = (12 * 12) / 2)
  (h3 : areas.C = (13 * 13) / 2) : 
  areas.A + areas.B = areas.C := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_areas_sum_l1527_152773


namespace NUMINAMATH_CALUDE_union_determines_k_l1527_152797

def A (k : ℕ) : Set ℕ := {1, 2, k}
def B : Set ℕ := {2, 5}

theorem union_determines_k (k : ℕ) : A k ∪ B = {1, 2, 3, 5} → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_determines_k_l1527_152797


namespace NUMINAMATH_CALUDE_robin_extra_gum_l1527_152785

/-- The number of extra pieces of gum Robin has -/
def extra_gum (packages : ℕ) (pieces_per_package : ℕ) (total_pieces : ℕ) : ℕ :=
  total_pieces - (packages * pieces_per_package)

/-- Theorem: Robin has 8 extra pieces of gum -/
theorem robin_extra_gum :
  extra_gum 43 23 997 = 8 := by
  sorry

end NUMINAMATH_CALUDE_robin_extra_gum_l1527_152785


namespace NUMINAMATH_CALUDE_max_three_match_winners_200_l1527_152781

/-- Represents a single-elimination tournament --/
structure Tournament :=
  (participants : ℕ)

/-- Calculates the total number of matches in a single-elimination tournament --/
def total_matches (t : Tournament) : ℕ :=
  t.participants - 1

/-- Calculates the maximum number of participants who can win at least 3 matches --/
def max_participants_with_three_wins (t : Tournament) : ℕ :=
  (total_matches t) / 3

/-- Theorem stating the maximum number of participants who can win at least 3 matches
    in a tournament with 200 participants --/
theorem max_three_match_winners_200 :
  ∃ (t : Tournament), t.participants = 200 ∧ max_participants_with_three_wins t = 66 :=
by
  sorry


end NUMINAMATH_CALUDE_max_three_match_winners_200_l1527_152781


namespace NUMINAMATH_CALUDE_sqrt5_diamond_sqrt5_equals_20_l1527_152759

-- Define the custom operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt5_diamond_sqrt5_equals_20 : diamond (Real.sqrt 5) (Real.sqrt 5) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sqrt5_diamond_sqrt5_equals_20_l1527_152759


namespace NUMINAMATH_CALUDE_max_min_sum_of_f_l1527_152739

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.log (Real.sqrt (x^2 + 1) + x)) / (x^2 + 1)

theorem max_min_sum_of_f :
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧
                (∀ x, N ≤ f x) ∧ (∃ x, f x = N) ∧
                (M + N = 2) := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_of_f_l1527_152739


namespace NUMINAMATH_CALUDE_order_of_magnitudes_l1527_152762

theorem order_of_magnitudes (x a : ℝ) (hx : x < 0) (ha : a = 2 * x) :
  x^2 < a * x ∧ a * x < a^2 := by
  sorry

end NUMINAMATH_CALUDE_order_of_magnitudes_l1527_152762


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1527_152778

/-- 
Given a hyperbola with equation x²/a² - y²/b² = 1,
if one of its asymptotes is y = (√7/3)x and 
the distance from one of its vertices to the nearer focus is 1,
then a = 3 and b = √7.
-/
theorem hyperbola_equation (a b : ℝ) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  (b/a = Real.sqrt 7 / 3) →               -- Asymptote condition
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c - a = 1) →  -- Vertex-focus distance condition
  (a = 3 ∧ b = Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1527_152778


namespace NUMINAMATH_CALUDE_fifth_element_is_35_l1527_152771

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalElements : ℕ
  sampleSize : ℕ
  firstElement : ℕ

/-- Calculates the nth element in a systematic sample -/
def nthElement (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.firstElement + (n - 1) * (s.totalElements / s.sampleSize)

theorem fifth_element_is_35 (s : SystematicSampling) 
  (h1 : s.totalElements = 160)
  (h2 : s.sampleSize = 20)
  (h3 : s.firstElement = 3) :
  nthElement s 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_is_35_l1527_152771


namespace NUMINAMATH_CALUDE_fred_seashells_l1527_152799

/-- The number of seashells Fred found initially -/
def initial_seashells : ℝ := 47.5

/-- The number of seashells Fred gave to Jessica -/
def given_seashells : ℝ := 25.3

/-- The number of seashells Fred has now -/
def remaining_seashells : ℝ := initial_seashells - given_seashells

theorem fred_seashells : remaining_seashells = 22.2 := by sorry

end NUMINAMATH_CALUDE_fred_seashells_l1527_152799


namespace NUMINAMATH_CALUDE_f_properties_l1527_152775

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a^2 * Real.log x

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x > 0, f a x = x^2 - a*x - a^2 * Real.log x) ∧
  (a = 1 → ∃ x > 0, ∀ y > 0, f a x ≤ f a y) ∧
  (a = 1 → ∃ x > 0, f a x = 0) ∧
  ((-2 ≤ a ∧ a ≤ 1) → ∀ x y, 1 < x ∧ x < y → f a x < f a y) ∧
  (a > 1 → ∀ x y, 1 < x ∧ x < y ∧ y < a → f a x > f a y) ∧
  (a > 1 → ∀ x y, a < x ∧ x < y → f a x < f a y) ∧
  (a < -2 → ∀ x y, 1 < x ∧ x < y ∧ y < -a/2 → f a x > f a y) ∧
  (a < -2 → ∀ x y, -a/2 < x ∧ x < y → f a x < f a y) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l1527_152775


namespace NUMINAMATH_CALUDE_syllogism_arrangement_l1527_152716

-- Define the property of being divisible by 2
def divisible_by_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Define odd numbers
def odd (n : ℕ) : Prop := ¬(divisible_by_2 n)

-- State the theorem
theorem syllogism_arrangement :
  (∀ n : ℕ, odd n → ¬(divisible_by_2 n)) →  -- Statement ②
  (odd 2013) →                              -- Statement ③
  ¬(divisible_by_2 2013)                    -- Statement ①
  := by sorry

end NUMINAMATH_CALUDE_syllogism_arrangement_l1527_152716


namespace NUMINAMATH_CALUDE_either_false_implies_not_p_true_l1527_152730

theorem either_false_implies_not_p_true (p q : Prop) :
  (¬p ∨ ¬q → ¬p) ∧ ¬(¬p → ¬p ∨ ¬q) :=
sorry

end NUMINAMATH_CALUDE_either_false_implies_not_p_true_l1527_152730


namespace NUMINAMATH_CALUDE_certain_number_equation_l1527_152752

theorem certain_number_equation (x : ℝ) : 13 * x + 14 * x + 17 * x + 11 = 143 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l1527_152752


namespace NUMINAMATH_CALUDE_original_number_proof_l1527_152735

theorem original_number_proof (x : ℝ) : x * 1.5 = 120 → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1527_152735


namespace NUMINAMATH_CALUDE_pear_apple_difference_l1527_152703

theorem pear_apple_difference :
  let red_apples : ℕ := 15
  let green_apples : ℕ := 8
  let pears : ℕ := 32
  let total_apples : ℕ := red_apples + green_apples
  pears - total_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_pear_apple_difference_l1527_152703
