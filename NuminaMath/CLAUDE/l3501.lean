import Mathlib

namespace x_minus_y_negative_l3501_350151

theorem x_minus_y_negative (x y : ℝ) (h : x < y) : x - y < 0 := by
  sorry

end x_minus_y_negative_l3501_350151


namespace sqrt_2_plus_sqrt_2_plus_l3501_350191

theorem sqrt_2_plus_sqrt_2_plus : ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (2 + x) := by
  sorry

end sqrt_2_plus_sqrt_2_plus_l3501_350191


namespace friendly_parabola_symmetric_l3501_350103

/-- Represents a parabola of the form y = ax² + bx --/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Defines the "friendly parabola" relationship between two parabolas --/
def is_friendly_parabola (L₁ L₂ : Parabola) : Prop :=
  L₂.a = -L₁.a ∧ 
  L₂.b = -2 * L₁.a * (-L₁.b / (2 * L₁.a)) + L₁.b

/-- Theorem: The "friendly parabola" relationship is symmetric --/
theorem friendly_parabola_symmetric (L₁ L₂ : Parabola) :
  is_friendly_parabola L₁ L₂ → is_friendly_parabola L₂ L₁ := by
  sorry


end friendly_parabola_symmetric_l3501_350103


namespace afternoon_absences_l3501_350180

theorem afternoon_absences (morning_registered : ℕ) (morning_absent : ℕ) 
  (afternoon_registered : ℕ) (total_students : ℕ) 
  (h1 : morning_registered = 25)
  (h2 : morning_absent = 3)
  (h3 : afternoon_registered = 24)
  (h4 : total_students = 42)
  (h5 : total_students = (morning_registered - morning_absent) + (afternoon_registered - afternoon_absent)) :
  afternoon_absent = 4 := by
  sorry

end afternoon_absences_l3501_350180


namespace nelly_painting_bid_l3501_350123

/-- The amount Nelly paid for the painting -/
def nelly_bid (joe_bid sarah_bid : ℕ) : ℕ :=
  max
    (3 * joe_bid + 2000)
    (max (4 * sarah_bid + 1500) (2 * (joe_bid + sarah_bid) + 1000))

/-- Theorem: Given the conditions, Nelly paid $482,000 for the painting -/
theorem nelly_painting_bid :
  nelly_bid 160000 50000 = 482000 := by
  sorry

end nelly_painting_bid_l3501_350123


namespace largest_area_error_l3501_350185

theorem largest_area_error (actual_circumference : ℝ) (error_percent : ℝ) : 
  actual_circumference = 30 →
  error_percent = 10 →
  ∃ (computed_area actual_area : ℝ),
    computed_area = π * ((actual_circumference * (1 + error_percent / 100)) / (2 * π))^2 ∧
    actual_area = π * (actual_circumference / (2 * π))^2 ∧
    (computed_area - actual_area) / actual_area * 100 ≤ 21 ∧
    ∃ (other_computed_area : ℝ),
      other_computed_area = π * ((actual_circumference * (1 - error_percent / 100)) / (2 * π))^2 ∧
      (other_computed_area - actual_area) / actual_area * 100 < 21 :=
by sorry

end largest_area_error_l3501_350185


namespace largest_three_digit_multiple_of_12_with_digit_sum_24_l3501_350131

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if the number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_12_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n % 12 = 0 → digit_sum n = 24 → n ≤ 888 :=
by sorry

end largest_three_digit_multiple_of_12_with_digit_sum_24_l3501_350131


namespace number_equation_proof_l3501_350165

theorem number_equation_proof : 
  ∃ x : ℝ, (3034 - (x / 20.04) = 2984) ∧ (x = 1002) := by
  sorry

end number_equation_proof_l3501_350165


namespace honey_servings_l3501_350115

def total_honey : ℚ := 37 + 1/3
def serving_size : ℚ := 1 + 1/2

theorem honey_servings : (total_honey / serving_size) = 24 + 8/9 := by
  sorry

end honey_servings_l3501_350115


namespace p_sufficient_not_necessary_for_q_l3501_350118

-- Define the propositions p and q
def p (x : ℝ) : Prop := x > 1
def q (x : ℝ) : Prop := 1/x < 1

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end p_sufficient_not_necessary_for_q_l3501_350118


namespace solution_set_inequality_l3501_350138

theorem solution_set_inequality (x : ℝ) : (2 * x + 5) / (x - 2) < 1 ↔ -7 < x ∧ x < 2 := by
  sorry

end solution_set_inequality_l3501_350138


namespace extreme_point_and_extrema_l3501_350150

/-- The function f(x) = ax³ - 3x² -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem extreme_point_and_extrema 
  (a : ℝ) 
  (h1 : f_derivative a 2 = 0) :
  a = 1 ∧ 
  (∀ x ∈ Set.Icc (-1 : ℝ) 5, f 1 x ≥ -4) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 5, f 1 x ≤ 50) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 5, f 1 x = -4) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 5, f 1 x = 50) := by
  sorry

end extreme_point_and_extrema_l3501_350150


namespace one_is_monomial_l3501_350194

/-- A monomial is an algebraic expression with only one term. -/
def is_monomial (expr : ℕ → ℚ) : Prop :=
  ∃ (c : ℚ) (n : ℕ), ∀ m : ℕ, expr m = if m = n then c else 0

/-- The constant function that always returns 1 -/
def const_one : ℕ → ℚ := λ _ => 1

theorem one_is_monomial : is_monomial const_one :=
sorry

end one_is_monomial_l3501_350194


namespace smallest_x_value_l3501_350134

theorem smallest_x_value (x y : ℕ) : 
  (∃ (y : ℕ), (4 : ℚ) / 5 = y / (200 + x)) →
  (∀ (z : ℕ), z < x → ¬(∃ (w : ℕ), (4 : ℚ) / 5 = w / (200 + z))) →
  x = 0 := by
sorry

end smallest_x_value_l3501_350134


namespace fraction_multiplication_equality_l3501_350158

theorem fraction_multiplication_equality : 
  (8 / 9)^2 * (1 / 3)^2 * (2 / 5) = 128 / 3645 := by sorry

end fraction_multiplication_equality_l3501_350158


namespace natalia_cycling_distance_l3501_350125

/-- The total distance ridden by Natalia over four days --/
def total_distance (monday tuesday wednesday thursday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday

/-- The problem statement --/
theorem natalia_cycling_distance :
  ∀ (monday tuesday wednesday thursday : ℕ),
    monday = 40 →
    tuesday = 50 →
    wednesday = tuesday / 2 →
    thursday = monday + wednesday →
    total_distance monday tuesday wednesday thursday = 180 := by
  sorry

end natalia_cycling_distance_l3501_350125


namespace simplify_expression_l3501_350193

theorem simplify_expression (x : ℝ) : (2 + x) * (1 - x) + (x + 2)^2 = 5 * x + 6 := by
  sorry

end simplify_expression_l3501_350193


namespace train_length_l3501_350112

/-- 
Given a train with a speed of 180 km/h that crosses an electric pole in 50 seconds, 
the length of the train is 2500 meters.
-/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 180 → time = 50 → length = speed * (1000 / 3600) * time → length = 2500 := by
  sorry

end train_length_l3501_350112


namespace relay_team_selection_l3501_350142

/-- The number of sprinters available -/
def total_sprinters : ℕ := 6

/-- The number of sprinters needed for the relay race -/
def relay_team_size : ℕ := 4

/-- The number of sprinters who cannot run the first leg -/
def first_leg_restricted : ℕ := 2

/-- The number of possible team compositions for the relay race -/
def team_compositions : ℕ := 240

theorem relay_team_selection :
  (total_sprinters - first_leg_restricted).choose 1 * 
  (total_sprinters - 1).descFactorial (relay_team_size - 1) = 
  team_compositions := by
  sorry

end relay_team_selection_l3501_350142


namespace coin_arrangement_count_l3501_350176

/-- Represents the number of ways to arrange 5 gold coins in 6 gaps between silver coins -/
def gold_arrangements : ℕ := 6

/-- Represents the number of valid orientations satisfying the engraving conditions -/
def valid_orientations : ℕ := 8

/-- The total number of distinguishable arrangements satisfying all conditions -/
def total_arrangements : ℕ := gold_arrangements * valid_orientations

/-- Theorem stating that the number of distinguishable arrangements is 48 -/
theorem coin_arrangement_count :
  total_arrangements = 48 := by sorry

end coin_arrangement_count_l3501_350176


namespace one_way_ticket_cost_l3501_350145

-- Define the cost of a 30-day pass
def pass_cost : ℝ := 50

-- Define the minimum number of rides for the pass to be cheaper
def min_rides : ℕ := 26

-- Define the cost of a one-way ticket
def ticket_cost : ℝ := 2

-- Theorem statement
theorem one_way_ticket_cost :
  (pass_cost / min_rides < ticket_cost) ∧
  (∀ x : ℝ, x > 0 ∧ x < ticket_cost → pass_cost / min_rides ≥ x) := by
  sorry

end one_way_ticket_cost_l3501_350145


namespace f_satisfies_conditions_f_unique_l3501_350168

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 2 then 0 else 2 / (2 - x)

theorem f_satisfies_conditions :
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → f (x * f y) * f y = f (x + y)) ∧
  f 2 = 0 ∧
  (∀ x : ℝ, 0 ≤ x → x < 2 → f x ≠ 0) ∧
  (∀ x : ℝ, x ≥ 0 → f x ≥ 0) :=
sorry

theorem f_unique :
  ∀ g : ℝ → ℝ,
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → g (x * g y) * g y = g (x + y)) →
  g 2 = 0 →
  (∀ x : ℝ, 0 ≤ x → x < 2 → g x ≠ 0) →
  (∀ x : ℝ, x ≥ 0 → g x ≥ 0) →
  g = f :=
sorry

end f_satisfies_conditions_f_unique_l3501_350168


namespace production_days_l3501_350164

theorem production_days (n : ℕ) : 
  (n * 50 + 100) / (n + 1) = 55 → n = 9 := by
sorry

end production_days_l3501_350164


namespace gcd_2023_2048_l3501_350130

theorem gcd_2023_2048 : Nat.gcd 2023 2048 = 1 := by
  sorry

end gcd_2023_2048_l3501_350130


namespace digit_150_of_5_37_l3501_350135

def decimal_expansion (n d : ℕ) : ℕ → ℕ
  | 0 => (10 * n / d) % 10
  | i + 1 => decimal_expansion ((10 * n % d) * 10) d i

theorem digit_150_of_5_37 : decimal_expansion 5 37 149 = 1 := by
  sorry

end digit_150_of_5_37_l3501_350135


namespace no_permutations_satisfy_condition_l3501_350159

theorem no_permutations_satisfy_condition :
  ∀ (b₁ b₂ b₃ b₄ b₅ b₆ : ℕ), 
    b₁ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
    b₂ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
    b₃ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
    b₄ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
    b₅ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
    b₆ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
    b₁ ≠ b₂ ∧ b₁ ≠ b₃ ∧ b₁ ≠ b₄ ∧ b₁ ≠ b₅ ∧ b₁ ≠ b₆ ∧
    b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₂ ≠ b₆ ∧
    b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧
    b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧
    b₅ ≠ b₆ →
    ((b₁ + 1) / 3) * ((b₂ + 2) / 3) * ((b₃ + 3) / 3) * 
    ((b₄ + 4) / 3) * ((b₅ + 5) / 3) * ((b₆ + 6) / 3) ≤ 120 := by
  sorry


end no_permutations_satisfy_condition_l3501_350159


namespace rotating_triangle_path_length_l3501_350177

/-- The total path length of point A in a rotating triangle -/
theorem rotating_triangle_path_length 
  (a : Real) 
  (h1 : 0 < a) 
  (h2 : a < π / 3) : 
  ∃ (s : Real), 
    s = 22 * π * (1 + Real.sin a) - 66 * a ∧ 
    s = (100 - 1) / 3 * (2 / 3 * π * (1 + Real.sin a) - 2 * a) := by
  sorry

end rotating_triangle_path_length_l3501_350177


namespace sin_135_degrees_l3501_350102

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_135_degrees_l3501_350102


namespace smallest_positive_c_inequality_l3501_350175

theorem smallest_positive_c_inequality (c : ℝ) : 
  (c > 0 ∧ 
   ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 
   (x^3 + y^3 - x) + c * |x - y| ≥ y - x^2) → 
  c ≥ 0 :=
sorry

end smallest_positive_c_inequality_l3501_350175


namespace fourth_root_of_polynomial_l3501_350113

/-- Given a polynomial x^4 - px^3 + qx^2 - rx + s = 0 where three of its roots are
    the tangents of the angles of a triangle, the fourth root is (r - p) / (q - s - 1). -/
theorem fourth_root_of_polynomial (p q r s : ℝ) (A B C : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (h_roots : ∃ (ρ : ℝ), (Real.tan A) * (Real.tan B) * (Real.tan C) * ρ = s ∧
                        (Real.tan A) * (Real.tan B) + (Real.tan B) * (Real.tan C) + 
                        (Real.tan C) * (Real.tan A) + 
                        ((Real.tan A) + (Real.tan B) + (Real.tan C)) * ρ = q ∧
                        (Real.tan A) * (Real.tan B) * (Real.tan C) + 
                        ((Real.tan A) * (Real.tan B) + (Real.tan B) * (Real.tan C) + 
                        (Real.tan C) * (Real.tan A)) * ρ = r ∧
                        (Real.tan A) + (Real.tan B) + (Real.tan C) + ρ = p) :
  ∃ (ρ : ℝ), ρ = (r - p) / (q - s - 1) ∧
              ρ^4 - p*ρ^3 + q*ρ^2 - r*ρ + s = 0 :=
sorry

end fourth_root_of_polynomial_l3501_350113


namespace right_triangle_side_length_l3501_350161

theorem right_triangle_side_length (a b c : ℝ) : 
  a = 6 → c = 10 → a^2 + b^2 = c^2 → b = 8 := by sorry

end right_triangle_side_length_l3501_350161


namespace negation_of_exists_cube_positive_l3501_350196

theorem negation_of_exists_cube_positive :
  (¬ ∃ x : ℝ, x^3 > 0) ↔ (∀ x : ℝ, x^3 ≤ 0) := by
  sorry

end negation_of_exists_cube_positive_l3501_350196


namespace age_problem_l3501_350167

/-- Given ages a, b, c, d, and their sum Y, prove b's age. -/
theorem age_problem (a b c d Y : ℚ) 
  (h1 : a = b + 2)           -- a is two years older than b
  (h2 : b = 2 * c)           -- b is twice as old as c
  (h3 : d = a / 2)           -- d is half the age of a
  (h4 : a + b + c + d = Y)   -- sum of ages is Y
  : b = Y / 3 - 1 := by
sorry

end age_problem_l3501_350167


namespace initial_apps_correct_l3501_350143

/-- The initial number of apps Dave had on his phone -/
def initial_apps : ℕ := 10

/-- The number of apps Dave added -/
def added_apps : ℕ := 11

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := 17

/-- The number of apps Dave had left after adding and deleting -/
def remaining_apps : ℕ := 4

/-- Theorem stating that the initial number of apps is correct -/
theorem initial_apps_correct : 
  initial_apps + added_apps - deleted_apps = remaining_apps := by
  sorry

#check initial_apps_correct

end initial_apps_correct_l3501_350143


namespace furniture_cost_prices_l3501_350162

def computer_table_price : ℝ := 8450
def bookshelf_price : ℝ := 6250
def chair_price : ℝ := 3400

def computer_table_markup : ℝ := 0.30
def bookshelf_markup : ℝ := 0.25
def chair_discount : ℝ := 0.15

theorem furniture_cost_prices :
  ∃ (computer_table_cost bookshelf_cost chair_cost : ℝ),
    computer_table_cost = computer_table_price / (1 + computer_table_markup) ∧
    bookshelf_cost = bookshelf_price / (1 + bookshelf_markup) ∧
    chair_cost = chair_price / (1 - chair_discount) ∧
    computer_table_cost = 6500 ∧
    bookshelf_cost = 5000 ∧
    chair_cost = 4000 :=
by sorry

end furniture_cost_prices_l3501_350162


namespace largest_prime_factor_of_factorial_sum_l3501_350146

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_prime_factor (n : ℕ) : ℕ :=
  (Nat.factors n).foldl max 0

theorem largest_prime_factor_of_factorial_sum :
  largest_prime_factor (factorial 6 + factorial 7) = 5 := by
  sorry

end largest_prime_factor_of_factorial_sum_l3501_350146


namespace real_part_range_l3501_350183

theorem real_part_range (z : ℂ) (ω : ℝ) (h1 : ω = z + z⁻¹) (h2 : -1 < ω) (h3 : ω < 2) :
  -1/2 < z.re ∧ z.re < 1 := by sorry

end real_part_range_l3501_350183


namespace liquid_rise_ratio_l3501_350189

-- Define the cones and marble
def small_cone_radius : ℝ := 5
def large_cone_radius : ℝ := 10
def marble_radius : ℝ := 2

-- Define the theorem
theorem liquid_rise_ratio :
  ∀ (h₁ h₂ : ℝ),
  h₁ > 0 → h₂ > 0 →
  (1/3 * π * small_cone_radius^2 * h₁ = 1/3 * π * large_cone_radius^2 * h₂) →
  ∃ (x : ℝ),
    x > 1 ∧
    (1/3 * π * (small_cone_radius * x)^2 * (h₁ * x) = 1/3 * π * small_cone_radius^2 * h₁ + 4/3 * π * marble_radius^3) ∧
    (1/3 * π * (large_cone_radius * x)^2 * (h₂ * x) = 1/3 * π * large_cone_radius^2 * h₂ + 4/3 * π * marble_radius^3) ∧
    (h₁ * (x - 1)) / (h₂ * (x - 1)) = 4 :=
sorry

end liquid_rise_ratio_l3501_350189


namespace inequality_solution_set_l3501_350171

theorem inequality_solution_set : 
  ∀ x : ℝ, -x^2 - 3*x + 4 > 0 ↔ x ∈ Set.Ioo (-4) 1 := by sorry

end inequality_solution_set_l3501_350171


namespace largest_five_digit_divisible_by_four_l3501_350132

theorem largest_five_digit_divisible_by_four :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99996 ∧ n % 4 = 0 → n ≤ 99996 :=
by
  sorry

end largest_five_digit_divisible_by_four_l3501_350132


namespace factorial_ratio_squared_l3501_350109

theorem factorial_ratio_squared : (Nat.factorial 45 / Nat.factorial 43)^2 = 3920400 := by
  sorry

end factorial_ratio_squared_l3501_350109


namespace tangent_lines_with_slope_one_l3501_350106

def f (x : ℝ) := x^3

theorem tangent_lines_with_slope_one :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, (deriv f) x = 1 := by sorry

end tangent_lines_with_slope_one_l3501_350106


namespace joggers_speed_ratio_l3501_350137

theorem joggers_speed_ratio : 
  ∀ (v₁ v₂ : ℝ), v₁ > v₂ → v₁ > 0 → v₂ > 0 →
  (v₁ + v₂) * 2 = 12 →
  (v₁ - v₂) * 6 = 12 →
  v₁ / v₂ = 2 := by
sorry

end joggers_speed_ratio_l3501_350137


namespace dice_invisible_dots_l3501_350182

theorem dice_invisible_dots : 
  let total_dots_per_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6
  let total_dots : ℕ := 4 * total_dots_per_die
  let visible_numbers : List ℕ := [1, 1, 2, 3, 3, 4, 5, 6]
  let sum_visible : ℕ := visible_numbers.sum
  total_dots - sum_visible = 59 := by
sorry

end dice_invisible_dots_l3501_350182


namespace bike_tractor_speed_ratio_l3501_350184

/-- The ratio of the speed of a bike to the speed of a tractor -/
theorem bike_tractor_speed_ratio :
  ∀ (speed_car speed_bike speed_tractor : ℝ),
  speed_car = (9/5) * speed_bike →
  speed_car = 450 / 5 →
  speed_tractor = 575 / 23 →
  ∃ (k : ℝ), speed_bike = k * speed_tractor →
  speed_bike / speed_tractor = 2 := by
sorry

end bike_tractor_speed_ratio_l3501_350184


namespace seokjin_class_size_l3501_350192

/-- The number of students in Taehyung's class -/
def taehyung_class : ℕ := 35

/-- The number of students in Jimin's class -/
def jimin_class : ℕ := taehyung_class - 3

/-- The number of students in Seokjin's class -/
def seokjin_class : ℕ := jimin_class + 2

theorem seokjin_class_size : seokjin_class = 34 := by
  sorry

end seokjin_class_size_l3501_350192


namespace snail_reaches_tree_on_day_37_l3501_350127

/-- The number of days it takes for a snail to reach another tree -/
def days_to_reach_tree (s l₁ l₂ : ℕ) : ℕ :=
  let daily_progress := l₁ - l₂
  let days_to_final_stretch := (s - l₁) / daily_progress
  days_to_final_stretch + 1

/-- Theorem stating that the snail reaches the tree on the 37th day -/
theorem snail_reaches_tree_on_day_37 :
  days_to_reach_tree 40 4 3 = 37 := by
  sorry

#eval days_to_reach_tree 40 4 3

end snail_reaches_tree_on_day_37_l3501_350127


namespace marks_initial_money_l3501_350139

theorem marks_initial_money (initial_money : ℚ) : 
  (1/2 : ℚ) * initial_money + 14 + 
  (1/3 : ℚ) * initial_money + 16 + 
  (1/4 : ℚ) * initial_money + 18 = initial_money → 
  initial_money = 576 := by
sorry

end marks_initial_money_l3501_350139


namespace managers_salary_l3501_350186

theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (avg_increase : ℕ) (managers_salary : ℕ) : 
  num_employees = 18 → 
  avg_salary = 2000 → 
  avg_increase = 200 → 
  (num_employees * avg_salary + managers_salary) / (num_employees + 1) = avg_salary + avg_increase →
  managers_salary = 5800 := by
sorry

end managers_salary_l3501_350186


namespace work_completed_in_five_days_l3501_350169

/-- Represents the fraction of work completed by a person in one day -/
def work_rate (days : ℚ) : ℚ := 1 / days

/-- Represents the total work completed by all workers in one day -/
def total_work_rate (a b c d : ℚ) : ℚ := work_rate a + work_rate b + work_rate c + work_rate d

/-- Represents the work completed in a given number of days -/
def work_completed (rate : ℚ) (days : ℚ) : ℚ := min 1 (rate * days)

/-- Theorem: Given the work rates of A, B, C, and D, prove that after 5 days of working together, no work is left -/
theorem work_completed_in_five_days :
  let a := 10
  let b := 15
  let c := 20
  let d := 30
  let rate := total_work_rate a b c d
  work_completed rate 5 = 1 :=
by sorry

end work_completed_in_five_days_l3501_350169


namespace quadratic_coefficient_l3501_350172

/-- Given a quadratic equation 2x² = 5x - 3, prove that when converted to the general form ax² + bx + c = 0,
    if the coefficient of x² (a) is 2, then the coefficient of x (b) is -5. -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, 2*x^2 = 5*x - 3) →  -- original equation
  (∀ x, a*x^2 + b*x + c = 0) →  -- general form
  a = 2 →
  b = -5 := by
sorry

end quadratic_coefficient_l3501_350172


namespace expression_result_l3501_350141

theorem expression_result : (3.242 * 10) / 100 = 0.3242 := by
  sorry

end expression_result_l3501_350141


namespace queue_arrangements_catalan_number_l3501_350160

def valid_queue_arrangements (n : ℕ) : ℕ :=
  (1 / (n + 1)) * Nat.choose (2 * n) n

theorem queue_arrangements_catalan_number (n : ℕ) :
  valid_queue_arrangements n = (1 / (n + 1)) * Nat.choose (2 * n) n :=
sorry

end queue_arrangements_catalan_number_l3501_350160


namespace gain_percent_calculation_l3501_350170

theorem gain_percent_calculation (cost_price selling_price : ℝ) :
  selling_price = 2.5 * cost_price →
  (selling_price - cost_price) / cost_price * 100 = 150 :=
by sorry

end gain_percent_calculation_l3501_350170


namespace opposite_to_Y_is_Z_l3501_350107

-- Define the faces of the cube
inductive Face : Type
  | V | W | X | Y | Z

-- Define the cube structure
structure Cube where
  faces : List Face
  bottom : Face
  right_of_bottom : Face

-- Define the opposite face relation
def opposite_face (c : Cube) (f : Face) : Face :=
  sorry

-- Theorem statement
theorem opposite_to_Y_is_Z (c : Cube) :
  c.bottom = Face.X →
  c.right_of_bottom = Face.W →
  c.faces = [Face.V, Face.W, Face.X, Face.Y, Face.Z] →
  opposite_face c Face.Y = Face.Z := by
  sorry

end opposite_to_Y_is_Z_l3501_350107


namespace scientific_notation_of_219400_l3501_350122

theorem scientific_notation_of_219400 :
  ∃ (a : ℝ) (n : ℤ), 
    219400 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ |a| ∧ 
    |a| < 10 ∧
    a = 2.194 ∧
    n = 5 := by
  sorry

end scientific_notation_of_219400_l3501_350122


namespace binomial_18_choose_6_l3501_350110

theorem binomial_18_choose_6 : Nat.choose 18 6 = 7280 := by
  sorry

end binomial_18_choose_6_l3501_350110


namespace complex_number_modulus_l3501_350120

theorem complex_number_modulus (i : ℂ) (Z : ℂ) (a : ℝ) :
  i^2 = -1 →
  Z = i * (3 - a * i) →
  Complex.abs Z = 5 →
  a = 4 ∨ a = -4 := by
  sorry

end complex_number_modulus_l3501_350120


namespace sqrt_of_sqrt_16_l3501_350144

theorem sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 ∨ Real.sqrt (Real.sqrt 16) = -2 := by
  sorry

end sqrt_of_sqrt_16_l3501_350144


namespace unique_solution_trigonometric_equation_l3501_350100

theorem unique_solution_trigonometric_equation :
  ∃! (n : ℕ+), Real.sin (π / (3 * n.val)) + Real.cos (π / (3 * n.val)) = Real.sqrt (n.val + 1) / 2 :=
by
  -- The unique solution is n = 7
  use 7
  -- Proof goes here
  sorry

end unique_solution_trigonometric_equation_l3501_350100


namespace hundred_thousandth_digit_position_l3501_350156

/-- Represents the position of a digit in a number, starting from the units digit (position 1) -/
def DigitPosition : ℕ → ℕ
  | 1 => 1  -- units
  | 2 => 2  -- tens
  | 3 => 3  -- hundreds
  | 4 => 4  -- thousands
  | 5 => 5  -- ten thousands
  | 6 => 6  -- hundred thousands
  | _ => 7  -- million and beyond

/-- The position of the hundred thousandth digit when counting from the units digit -/
theorem hundred_thousandth_digit_position : DigitPosition 6 = 6 := by
  sorry

end hundred_thousandth_digit_position_l3501_350156


namespace festival_attendance_l3501_350111

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (girls : ℕ) (boys : ℕ) : 
  total_students = 1500 →
  festival_attendees = 800 →
  girls + boys = total_students →
  (3 * girls) / 5 + (2 * boys) / 5 = festival_attendees →
  (3 * girls) / 5 = 600 :=
by sorry

end festival_attendance_l3501_350111


namespace exists_n_good_not_n_plus_one_good_l3501_350101

/-- Sum of digits of a natural number -/
def S (k : ℕ) : ℕ := sorry

/-- Function f(n) = n - S(n) -/
def f (n : ℕ) : ℕ := n - S n

/-- Iterated application of f, k times -/
def f_iter (k : ℕ) (n : ℕ) : ℕ := 
  match k with
  | 0 => n
  | k + 1 => f (f_iter k n)

/-- A number a is n-good if there exists a sequence a₀, ..., aₙ where aₙ = a and aᵢ₊₁ = f(aᵢ) -/
def is_n_good (n : ℕ) (a : ℕ) : Prop :=
  ∃ (a₀ : ℕ), f_iter n a₀ = a

/-- Main theorem: For all n, there exists an a that is n-good but not (n+1)-good -/
theorem exists_n_good_not_n_plus_one_good :
  ∀ n : ℕ, ∃ a : ℕ, is_n_good n a ∧ ¬is_n_good (n + 1) a := by sorry

end exists_n_good_not_n_plus_one_good_l3501_350101


namespace find_q_l3501_350197

theorem find_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : 
  q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end find_q_l3501_350197


namespace point_zero_three_on_graph_minimum_at_minus_two_point_one_seven_not_on_graph_l3501_350121

/-- Quadratic function f(x) = (x + 2)^2 - 1 -/
def f (x : ℝ) : ℝ := (x + 2)^2 - 1

/-- The point (0, 3) lies on the graph of f -/
theorem point_zero_three_on_graph : f 0 = 3 := by sorry

/-- The function f has a minimum value of -1 when x = -2 -/
theorem minimum_at_minus_two : 
  (∀ x : ℝ, f x ≥ -1) ∧ f (-2) = -1 := by sorry

/-- The point P(1, 7) does not lie on the graph of f -/
theorem point_one_seven_not_on_graph : f 1 ≠ 7 := by sorry

end point_zero_three_on_graph_minimum_at_minus_two_point_one_seven_not_on_graph_l3501_350121


namespace not_prime_23021_pow_377_minus_1_l3501_350178

theorem not_prime_23021_pow_377_minus_1 : ¬ Nat.Prime (23021^377 - 1) := by
  sorry

end not_prime_23021_pow_377_minus_1_l3501_350178


namespace decagon_triangles_l3501_350195

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices required to form a triangle -/
def r : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def num_triangles : ℕ := Nat.choose n r

/-- Theorem: The number of triangles that can be formed using the vertices of a regular decagon is 120 -/
theorem decagon_triangles : num_triangles = 120 := by
  sorry

end decagon_triangles_l3501_350195


namespace scientific_notation_of_1300000_l3501_350187

theorem scientific_notation_of_1300000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1300000 = a * (10 : ℝ) ^ n ∧ a = 1.3 ∧ n = 6 := by
  sorry

end scientific_notation_of_1300000_l3501_350187


namespace count_sequences_l3501_350173

/-- The number of finite sequences of k positive integers that sum to n -/
def T (n k : ℕ) : ℕ := sorry

/-- The main theorem stating that T(n,k) equals (n-1 choose k-1) for 1 ≤ k < n -/
theorem count_sequences (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  T n k = Nat.choose (n - 1) (k - 1) := by sorry

end count_sequences_l3501_350173


namespace basketball_score_formula_l3501_350133

/-- Represents the total points scored in a basketball game with specific conditions -/
def total_points (x : ℝ) : ℝ :=
  0.2 * x + 50

theorem basketball_score_formula (x y : ℝ) 
  (h1 : x + y = 50) 
  (h2 : x ≥ 0) 
  (h3 : y ≥ 0) : 
  0.4 * x * 3 + 0.5 * y * 2 = total_points x :=
sorry

end basketball_score_formula_l3501_350133


namespace power_product_simplification_l3501_350155

theorem power_product_simplification :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 := by
  sorry

end power_product_simplification_l3501_350155


namespace smallest_product_l3501_350114

def S : Finset ℕ := {3, 5, 7, 9, 11, 13}

theorem smallest_product (a b c d : ℕ) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  (a + b) * (c + d) ≥ 128 := by
  sorry

end smallest_product_l3501_350114


namespace pure_imaginary_condition_l3501_350117

/-- Given a complex number z and a real number m, 
    if z = (2+3i)(1-mi) is a pure imaginary number, then m = -2/3 -/
theorem pure_imaginary_condition (z : ℂ) (m : ℝ) : 
  z = (2 + 3*Complex.I) * (1 - m*Complex.I) ∧ z.re = 0 → m = -2/3 := by
  sorry

end pure_imaginary_condition_l3501_350117


namespace equilateral_triangle_third_vertex_y_coord_l3501_350129

/-- Given an equilateral triangle with two vertices at (2,7) and (10,7),
    and the third vertex in the first quadrant, the y-coordinate of the third vertex is 7 + 4√3 -/
theorem equilateral_triangle_third_vertex_y_coord :
  ∀ (x y : ℝ),
  (x > 0 ∧ y > 0) →  -- Third vertex is in the first quadrant
  (x - 2)^2 + (y - 7)^2 = 8^2 →  -- Distance from (2,7) is 8
  (x - 10)^2 + (y - 7)^2 = 8^2 →  -- Distance from (10,7) is 8
  y = 7 + 4 * Real.sqrt 3 :=
by sorry

end equilateral_triangle_third_vertex_y_coord_l3501_350129


namespace unique_solution_implies_n_l3501_350174

/-- Given a real number n, if the equation 9x^2 + nx + 36 = 0 has exactly one solution in x,
    then n = 36 or n = -36 -/
theorem unique_solution_implies_n (n : ℝ) :
  (∃! x : ℝ, 9 * x^2 + n * x + 36 = 0) → n = 36 ∨ n = -36 := by
  sorry

end unique_solution_implies_n_l3501_350174


namespace min_lines_for_200_intersections_l3501_350148

theorem min_lines_for_200_intersections :
  ∃ n : ℕ,
    n > 0 ∧
    n * (n - 1) / 2 = 200 ∧
    ∀ m : ℕ, m > 0 → m * (m - 1) / 2 = 200 → m ≥ n :=
by sorry

end min_lines_for_200_intersections_l3501_350148


namespace expected_sophomores_in_sample_l3501_350140

/-- Given a school with 1000 students, of which 320 are sophomores,
    the expected number of sophomores in a random sample of 200 students is 64. -/
theorem expected_sophomores_in_sample
  (total_students : ℕ)
  (sophomores : ℕ)
  (sample_size : ℕ)
  (h1 : total_students = 1000)
  (h2 : sophomores = 320)
  (h3 : sample_size = 200) :
  (sophomores : ℝ) / total_students * sample_size = 64 := by
  sorry

end expected_sophomores_in_sample_l3501_350140


namespace triangle_side_length_l3501_350105

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  b = 2 * Real.sqrt 7 →
  B = π / 3 →
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  c = 6 := by
sorry

end triangle_side_length_l3501_350105


namespace expression_equals_three_l3501_350124

theorem expression_equals_three (m : ℝ) (h : m = -1) : 
  (2 * m + 3) * (2 * m - 3) - (m - 1) * (m + 5) = 3 := by
  sorry

end expression_equals_three_l3501_350124


namespace system_solution_proof_l3501_350166

theorem system_solution_proof :
  ∃! (x y : ℝ), 
    (2 * x + Real.sqrt (2 * x + 3 * y) - 3 * y = 5) ∧ 
    (4 * x^2 + 2 * x + 3 * y - 9 * y^2 = 32) ∧
    (x = 17/4) ∧ (y = 5/2) := by
  sorry

end system_solution_proof_l3501_350166


namespace triangle_side_ratio_l3501_350126

theorem triangle_side_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b ≤ 2 * c) (h5 : b + c ≤ 3 * a) :
  2/3 < c/a ∧ c/a < 2 := by
sorry

end triangle_side_ratio_l3501_350126


namespace alla_boris_meeting_l3501_350128

/-- Represents the meeting point of Alla and Boris on a street with streetlights. -/
def meeting_point (total_lights : ℕ) (alla_position : ℕ) (boris_position : ℕ) : ℕ :=
  let gaps_covered := (alla_position - 1) + (total_lights - boris_position)
  let total_gaps := total_lights - 1
  let alla_total_gaps := 3 * (alla_position - 1)
  1 + alla_total_gaps

/-- Theorem stating that Alla and Boris meet at the 163rd streetlight under given conditions. -/
theorem alla_boris_meeting :
  meeting_point 400 55 321 = 163 :=
sorry

end alla_boris_meeting_l3501_350128


namespace largest_prime_form_is_seven_l3501_350179

def is_largest_prime_form (p : ℕ) : Prop :=
  Prime p ∧
  (∃ n : ℕ, Prime n ∧ p = 2^n + n^2 - 1) ∧
  p < 100 ∧
  ∀ q : ℕ, q ≠ p → Prime q → (∃ m : ℕ, Prime m ∧ q = 2^m + m^2 - 1) → q < 100 → q < p

theorem largest_prime_form_is_seven : is_largest_prime_form 7 := by
  sorry

end largest_prime_form_is_seven_l3501_350179


namespace inequality_proof_l3501_350198

theorem inequality_proof (a b c d : ℝ) :
  (a / (1 + 3*a)) + (b^2 / (1 + 3*b^2)) + (c^3 / (1 + 3*c^3)) + (d^4 / (1 + 3*d^4)) ≤ 1 := by
  sorry

end inequality_proof_l3501_350198


namespace imaginary_part_of_complex_fraction_l3501_350116

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z := (3 - 2*i^3) / (1 + i)
  Complex.im z = -1/2 := by
sorry

end imaginary_part_of_complex_fraction_l3501_350116


namespace arithmetic_sequence_properties_l3501_350149

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  a_5_eq_10 : a 5 = 10
  S_15_eq_240 : S 15 = 240

/-- The b sequence derived from the arithmetic sequence -/
def b (seq : ArithmeticSequence) (n : ℕ) : ℝ := seq.a (3^n)

/-- The sum of the first n terms of the b sequence -/
def T (seq : ArithmeticSequence) (n : ℕ) : ℝ := sorry

/-- Main theorem encapsulating the problem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 2*n) ∧
  (∀ n, seq.S n = n*(n+1)) ∧
  (∀ n, T seq n = 3^(n+1) - 3) := by sorry

end arithmetic_sequence_properties_l3501_350149


namespace min_q_for_min_a2016_l3501_350147

theorem min_q_for_min_a2016 (a : ℕ → ℕ) (q : ℚ) :
  (∀ n, 1 ≤ n ∧ n ≤ 2016 → a n = a 1 * q ^ (n - 1)) →
  1 < q ∧ q < 2 →
  (∀ r, 1 < r ∧ r < 2 → a 2016 ≤ (a 1 : ℚ) * r ^ 2015) →
  q = 6/5 :=
sorry

end min_q_for_min_a2016_l3501_350147


namespace mans_walking_rate_l3501_350136

/-- Proves that given a woman walking at 15 miles per hour who stops 2 minutes after passing a man
    and waits 4 minutes for him to catch up, the man's walking rate is 7.5 miles per hour. -/
theorem mans_walking_rate (woman_speed : ℝ) (passing_time : ℝ) (waiting_time : ℝ) :
  woman_speed = 15 →
  passing_time = 2 / 60 →
  waiting_time = 4 / 60 →
  ∃ (man_speed : ℝ), man_speed = 7.5 ∧ 
    woman_speed * passing_time = man_speed * waiting_time :=
by sorry

end mans_walking_rate_l3501_350136


namespace inequality_solution_implies_m_range_l3501_350199

/-- The solution set of the inequality (m-1)x^2 + (m-1)x + 2 > 0 is ℝ -/
def solution_set_is_real (m : ℝ) : Prop :=
  ∀ x, (m - 1) * x^2 + (m - 1) * x + 2 > 0

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := 1 ≤ m ∧ m < 9

theorem inequality_solution_implies_m_range :
  ∀ m : ℝ, solution_set_is_real m → m_range m :=
by sorry

end inequality_solution_implies_m_range_l3501_350199


namespace max_number_with_30_divisors_l3501_350104

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is divisible by m -/
def is_divisible_by (n m : ℕ) : Prop := sorry

theorem max_number_with_30_divisors :
  ∀ n : ℕ, 
    is_divisible_by n 30 → 
    num_divisors n = 30 → 
    n ≤ 11250 ∧ 
    (n = 11250 → is_divisible_by 11250 30 ∧ num_divisors 11250 = 30) :=
by sorry

end max_number_with_30_divisors_l3501_350104


namespace prob_four_blue_before_three_yellow_l3501_350119

/-- The probability of drawing all blue marbles before all yellow marbles -/
def blue_before_yellow_prob (blue : ℕ) (yellow : ℕ) : ℚ :=
  if blue = 0 then 0
  else if yellow = 0 then 1
  else (blue : ℚ) / (blue + yellow : ℚ)

/-- The theorem stating the probability of drawing all 4 blue marbles before all 3 yellow marbles -/
theorem prob_four_blue_before_three_yellow :
  blue_before_yellow_prob 4 3 = 4 / 7 := by
  sorry

end prob_four_blue_before_three_yellow_l3501_350119


namespace marks_money_l3501_350163

/-- The total value in cents of a collection of dimes and nickels -/
def total_value (num_dimes num_nickels : ℕ) : ℕ :=
  10 * num_dimes + 5 * num_nickels

/-- Theorem: Mark's total money is 90 cents -/
theorem marks_money :
  let num_dimes := 5
  let num_nickels := num_dimes + 3
  total_value num_dimes num_nickels = 90 := by
  sorry

end marks_money_l3501_350163


namespace max_intersection_lines_l3501_350152

/-- A plane in 3D space -/
structure Plane

/-- Represents the intersection of two planes -/
def intersect (p1 p2 : Plane) : Prop := sorry

/-- The number of intersection lines between two intersecting planes -/
def numIntersectionLines (p1 p2 : Plane) (h : intersect p1 p2) : ℕ := 1

/-- The theorem stating the maximum number of intersection lines for three intersecting planes -/
theorem max_intersection_lines (p1 p2 p3 : Plane) 
  (h12 : intersect p1 p2) (h23 : intersect p2 p3) (h13 : intersect p1 p3) :
  (numIntersectionLines p1 p2 h12 + 
   numIntersectionLines p2 p3 h23 + 
   numIntersectionLines p1 p3 h13) ≤ 3 := by sorry

end max_intersection_lines_l3501_350152


namespace equation_solutions_l3501_350108

theorem equation_solutions (a : ℝ) : 
  ((∃! x : ℝ, 5 + |x - 2| = a) ∧ (∃ x y : ℝ, x ≠ y ∧ 7 - |2*x + 6| = a ∧ 7 - |2*y + 6| = a) ∨
   (∃ x y : ℝ, x ≠ y ∧ 5 + |x - 2| = a ∧ 5 + |y - 2| = a) ∧ (∃! x : ℝ, 7 - |2*x + 6| = a)) ↔
  (a = 5 ∨ a = 7) :=
by sorry

end equation_solutions_l3501_350108


namespace geometric_sequence_ratio_and_formula_l3501_350153

/-- A geometric sequence with first term 1 and third term 4 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 3 = 4 ∧ ∀ n : ℕ, n ≥ 1 → ∃ q : ℝ, a (n + 1) = a n * q

theorem geometric_sequence_ratio_and_formula (a : ℕ → ℝ) (h : GeometricSequence a) :
  (∃ q : ℝ, q = 2 ∨ q = -2) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 2^(n-1) ∨ a n = (-2)^(n-1)) :=
by sorry

end geometric_sequence_ratio_and_formula_l3501_350153


namespace polynomial_division_remainder_l3501_350188

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^2015 + 1 = q * (X^8 - X^6 + X^4 - X^2 + 1) + (-X^5 + 1) := by
  sorry

end polynomial_division_remainder_l3501_350188


namespace age_ratio_problem_l3501_350190

theorem age_ratio_problem (a b c : ℕ) : 
  a = b + 2 →
  b + c = 25 →
  b = 10 →
  (b : ℚ) / c = 2 := by
sorry

end age_ratio_problem_l3501_350190


namespace min_subset_size_for_sum_equation_l3501_350157

theorem min_subset_size_for_sum_equation (n : ℕ) (hn : n ≥ 2) :
  ∃ m : ℕ, m = 2 * n + 2 ∧
  (∀ S : Finset ℕ, S ⊆ Finset.range (3 * n + 1) → S.card ≥ m →
    ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      a = b + c + d) ∧
  (∀ k : ℕ, k < m →
    ∃ T : Finset ℕ, T ⊆ Finset.range (3 * n + 1) ∧ T.card = k ∧
      ∀ a b c d : ℕ, a ∈ T → b ∈ T → c ∈ T → d ∈ T →
        (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) → a ≠ b + c + d) :=
by sorry

end min_subset_size_for_sum_equation_l3501_350157


namespace product_equals_reversed_product_l3501_350154

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem product_equals_reversed_product 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : is_two_digit a) 
  (h4 : (reverse_digits a) * b = 220) : 
  a * b = 220 := by
sorry

end product_equals_reversed_product_l3501_350154


namespace lawn_area_is_40_l3501_350181

/-- Represents a rectangular lawn with a path --/
structure LawnWithPath where
  length : ℝ
  width : ℝ
  pathWidth : ℝ

/-- Calculates the remaining lawn area after subtracting the path --/
def remainingLawnArea (lawn : LawnWithPath) : ℝ :=
  lawn.length * lawn.width - lawn.length * lawn.pathWidth

/-- Theorem stating that for the given dimensions, the remaining lawn area is 40 square meters --/
theorem lawn_area_is_40 (lawn : LawnWithPath) 
  (h1 : lawn.length = 10)
  (h2 : lawn.width = 5)
  (h3 : lawn.pathWidth = 1) : 
  remainingLawnArea lawn = 40 := by
  sorry

#check lawn_area_is_40

end lawn_area_is_40_l3501_350181
