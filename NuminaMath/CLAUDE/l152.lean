import Mathlib

namespace NUMINAMATH_CALUDE_total_rock_is_16_l152_15215

/-- The amount of rock costing $30 per ton -/
def rock_30 : ℕ := 8

/-- The amount of rock costing $40 per ton -/
def rock_40 : ℕ := 8

/-- The total amount of rock needed -/
def total_rock : ℕ := rock_30 + rock_40

theorem total_rock_is_16 : total_rock = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_rock_is_16_l152_15215


namespace NUMINAMATH_CALUDE_square_plot_area_l152_15299

/-- The area of a square plot with side length 50.5 m is 2550.25 square meters. -/
theorem square_plot_area : 
  let side_length : ℝ := 50.5
  let area : ℝ := side_length * side_length
  area = 2550.25 := by
  sorry

end NUMINAMATH_CALUDE_square_plot_area_l152_15299


namespace NUMINAMATH_CALUDE_original_average_age_is_40_l152_15235

/-- Proves that the original average age of a class is 40 years given specific conditions. -/
theorem original_average_age_is_40 
  (N : ℕ) -- Original number of students
  (A : ℝ) -- Original average age
  (new_students : ℕ) -- Number of new students
  (new_age : ℝ) -- Average age of new students
  (age_decrease : ℝ) -- Decrease in average age after new students join
  (h1 : N = 2) -- Original number of students is 2
  (h2 : new_students = 2) -- 2 new students join
  (h3 : new_age = 32) -- Average age of new students is 32
  (h4 : age_decrease = 4) -- Average age decreases by 4
  (h5 : (A * N + new_age * new_students) / (N + new_students) = A - age_decrease) -- New average age equation
  : A = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_average_age_is_40_l152_15235


namespace NUMINAMATH_CALUDE_equation_solution_l152_15276

theorem equation_solution (y : ℚ) : (4 * y - 2) / (5 * y - 5) = 3 / 4 → y = -7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l152_15276


namespace NUMINAMATH_CALUDE_smallest_valid_number_l152_15279

def is_valid (n : ℕ) : Prop :=
  n % 10 = 9 ∧
  n % 9 = 8 ∧
  n % 8 = 7 ∧
  n % 7 = 6 ∧
  n % 6 = 5 ∧
  n % 5 = 4 ∧
  n % 4 = 3 ∧
  n % 3 = 2 ∧
  n % 2 = 1

theorem smallest_valid_number :
  is_valid 2519 ∧ ∀ m : ℕ, m < 2519 → ¬ is_valid m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l152_15279


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_8250_l152_15248

theorem largest_prime_factor_of_8250 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 8250 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 8250 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_8250_l152_15248


namespace NUMINAMATH_CALUDE_basketball_substitutions_l152_15232

/-- The number of ways to make substitutions in a basketball game --/
def substitution_ways (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  let substitutes := total_players - starting_players
  let ways_0 := 1
  let ways_1 := starting_players * substitutes
  let ways_2 := ways_1 * (starting_players - 1) * (substitutes - 1)
  let ways_3 := ways_2 * (starting_players - 2) * (substitutes - 2)
  let ways_4 := ways_3 * (starting_players - 3) * (substitutes - 3)
  ways_0 + ways_1 + ways_2 + ways_3 + ways_4

/-- The main theorem about basketball substitutions --/
theorem basketball_substitutions :
  let total_ways := substitution_ways 15 5 4
  total_ways = 648851 ∧ total_ways % 100 = 51 := by
  sorry

#eval substitution_ways 15 5 4
#eval (substitution_ways 15 5 4) % 100

end NUMINAMATH_CALUDE_basketball_substitutions_l152_15232


namespace NUMINAMATH_CALUDE_cosine_equation_solution_l152_15239

theorem cosine_equation_solution (x : ℝ) : 
  (Real.cos x + 2 * Real.cos (6 * x))^2 = 9 + (Real.sin (3 * x))^2 ↔ 
  ∃ k : ℤ, x = 2 * k * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cosine_equation_solution_l152_15239


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l152_15269

def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem f_derivative_at_one : 
  (deriv f) 1 = 24 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l152_15269


namespace NUMINAMATH_CALUDE_first_group_size_l152_15208

/-- The number of beavers in the first group -/
def first_group : ℕ := 20

/-- The time taken by the first group to build the dam (in hours) -/
def time_first_group : ℕ := 3

/-- The number of beavers in the second group -/
def second_group : ℕ := 12

/-- The time taken by the second group to build the dam (in hours) -/
def time_second_group : ℕ := 5

/-- Theorem stating that the first group consists of 20 beavers -/
theorem first_group_size :
  first_group * time_first_group = second_group * time_second_group :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l152_15208


namespace NUMINAMATH_CALUDE_root_product_theorem_l152_15206

theorem root_product_theorem (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - y₁^3 + 1 = 0) → 
  (y₂^5 - y₂^3 + 1 = 0) → 
  (y₃^5 - y₃^3 + 1 = 0) → 
  (y₄^5 - y₄^3 + 1 = 0) → 
  (y₅^5 - y₅^3 + 1 = 0) → 
  ((y₁^2 - 3) * (y₂^2 - 3) * (y₃^2 - 3) * (y₄^2 - 3) * (y₅^2 - 3) = 22) := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l152_15206


namespace NUMINAMATH_CALUDE_parallelogram_area_l152_15223

/-- The area of a parallelogram with base 20 and height 4 is 80 -/
theorem parallelogram_area : 
  ∀ (base height : ℝ), 
  base = 20 → 
  height = 4 → 
  base * height = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l152_15223


namespace NUMINAMATH_CALUDE_extraneous_root_implies_m_value_l152_15244

theorem extraneous_root_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, (x - 1) / (x + 4) = m / (x + 4) ∧ x + 4 = 0) → m = -5 :=
by sorry

end NUMINAMATH_CALUDE_extraneous_root_implies_m_value_l152_15244


namespace NUMINAMATH_CALUDE_sixty_eighth_digit_of_largest_n_l152_15256

def largest_n : ℕ := (10^100 - 1) / 14

def digit_at_position (n : ℕ) (pos : ℕ) : ℕ :=
  (n / 10^(pos - 1)) % 10

theorem sixty_eighth_digit_of_largest_n :
  digit_at_position largest_n 68 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sixty_eighth_digit_of_largest_n_l152_15256


namespace NUMINAMATH_CALUDE_no_valid_numbers_l152_15228

theorem no_valid_numbers :
  ¬∃ (a b c : ℕ), 
    (100 ≤ 100 * a + 10 * b + c) ∧ 
    (100 * a + 10 * b + c < 1000) ∧ 
    (100 * a + 10 * b + c) % 15 = 0 ∧ 
    (10 * b + c) % 4 = 0 ∧ 
    a > b ∧ b > c :=
by sorry

end NUMINAMATH_CALUDE_no_valid_numbers_l152_15228


namespace NUMINAMATH_CALUDE_goldfish_equality_month_l152_15277

theorem goldfish_equality_month : ∃ n : ℕ, n > 0 ∧ 3^(n+1) = 125 * 5^n ∧ ∀ m : ℕ, 0 < m ∧ m < n → 3^(m+1) ≠ 125 * 5^m :=
by
  sorry

end NUMINAMATH_CALUDE_goldfish_equality_month_l152_15277


namespace NUMINAMATH_CALUDE_janet_snowball_percentage_l152_15211

/-- The number of snowballs Janet made -/
def janet_snowballs : ℕ := 50

/-- The number of snowballs Janet's brother made -/
def brother_snowballs : ℕ := 150

/-- The total number of snowballs made -/
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

/-- The percentage of snowballs Janet made -/
def janet_percentage : ℚ := (janet_snowballs : ℚ) / (total_snowballs : ℚ) * 100

theorem janet_snowball_percentage : janet_percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_janet_snowball_percentage_l152_15211


namespace NUMINAMATH_CALUDE_polynomial_equality_l152_15283

theorem polynomial_equality (a b c : ℤ) :
  (∀ x : ℝ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) →
  (a = 3 ∨ a = 7) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l152_15283


namespace NUMINAMATH_CALUDE_tangent_line_condition_l152_15268

/-- Given a function f(x) = x³ + ax², prove that if the tangent line
    at point (x₀, f(x₀)) has equation x + y = 0, then x₀ = ±1 and f(x₀) = -x₀ -/
theorem tangent_line_condition (a : ℝ) :
  ∃ x₀ : ℝ, (x₀ = 1 ∨ x₀ = -1) ∧
  let f := λ x : ℝ => x^3 + a*x^2
  let f' := λ x : ℝ => 3*x^2 + 2*a*x
  f' x₀ = -1 ∧ x₀ + f x₀ = 0 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_condition_l152_15268


namespace NUMINAMATH_CALUDE_problem_solution_l152_15209

theorem problem_solution : 
  ((-1)^2 + (64 : ℝ)^(1/3) - (-2) * (9 : ℝ)^(1/2) = 9) ∧ 
  (2 * ((3 : ℝ)^(1/2) - (2 : ℝ)^(1/2)) - ((2 : ℝ)^(1/2) + (3 : ℝ)^(1/2)) = (3 : ℝ)^(1/2) - 3 * (2 : ℝ)^(1/2)) := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_problem_solution_l152_15209


namespace NUMINAMATH_CALUDE_least_clock_equivalent_is_nine_l152_15290

/-- A number is clock equivalent to its square if their difference is divisible by 12 -/
def ClockEquivalent (n : ℕ) : Prop :=
  (n ^ 2 - n) % 12 = 0

/-- The least whole number greater than 4 that is clock equivalent to its square -/
def LeastClockEquivalent : ℕ := 9

theorem least_clock_equivalent_is_nine :
  (LeastClockEquivalent > 4) ∧
  ClockEquivalent LeastClockEquivalent ∧
  ∀ n : ℕ, (n > 4 ∧ n < LeastClockEquivalent) → ¬ClockEquivalent n :=
by sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_is_nine_l152_15290


namespace NUMINAMATH_CALUDE_final_chicken_count_l152_15245

def chicken_farm (initial_chickens : ℕ) 
                 (disease_A_infection_rate : ℚ)
                 (disease_A_death_rate : ℚ)
                 (disease_B_infection_rate : ℚ)
                 (disease_B_death_rate : ℚ)
                 (purchase_multiplier : ℚ) : ℕ :=
  sorry

theorem final_chicken_count : 
  chicken_farm 800 (15/100) (45/100) (20/100) (30/100) (25/2) = 1939 :=
sorry

end NUMINAMATH_CALUDE_final_chicken_count_l152_15245


namespace NUMINAMATH_CALUDE_no_intersection_in_S_l152_15243

-- Define the set S inductively
inductive S : (Real → Real) → Prop
  | base : S (fun x ↦ x)
  | sub {f} : S f → S (fun x ↦ x - f x)
  | add {f} : S f → S (fun x ↦ x + (1 - x) * f x)

-- Define the theorem
theorem no_intersection_in_S :
  ∀ (f g : Real → Real), S f → S g → f ≠ g →
  ∀ x, 0 < x → x < 1 → f x ≠ g x :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_in_S_l152_15243


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l152_15252

theorem quadratic_root_problem (a b c : ℝ) (h : a * (b - c) ≠ 0) :
  (∀ x, a * (b - c) * x^2 + b * (c - a) * x + c * (a - b) = 0 ↔ x = 1 ∨ x = (c * (a - b)) / (a * (b - c))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l152_15252


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l152_15217

/-- The total shaded area on a square carpet -/
theorem carpet_shaded_area (S T : ℝ) : 
  S > 0 ∧ T > 0 ∧ (12 : ℝ) / S = 4 ∧ S / T = 2 →
  S^2 + 4 * T^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_l152_15217


namespace NUMINAMATH_CALUDE_inequality_solution_set_l152_15227

theorem inequality_solution_set (x : ℝ) : -x^2 + 4*x + 5 < 0 ↔ x < -1 ∨ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l152_15227


namespace NUMINAMATH_CALUDE_cookie_distribution_l152_15201

theorem cookie_distribution (num_boxes : ℕ) (cookies_per_box : ℕ) (num_people : ℕ) :
  num_boxes = 7 →
  cookies_per_box = 10 →
  num_people = 5 →
  (num_boxes * cookies_per_box) / num_people = 14 :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l152_15201


namespace NUMINAMATH_CALUDE_candy_cost_l152_15266

def coin_value (c : Nat) : Nat :=
  if c = 1 then 25
  else if c = 2 then 10
  else if c = 3 then 5
  else 1

def is_valid_change (coins : List Nat) : Prop :=
  coins.length = 4 ∧ coins.all (λ c => c ∈ [1, 2, 3, 4])

def change_value (coins : List Nat) : Nat :=
  coins.map coin_value |>.sum

theorem candy_cost (coins : List Nat) :
  is_valid_change coins →
  (∀ other_coins, is_valid_change other_coins → change_value other_coins ≤ change_value coins) →
  100 - change_value coins = 55 := by
sorry

end NUMINAMATH_CALUDE_candy_cost_l152_15266


namespace NUMINAMATH_CALUDE_melanie_dimes_l152_15224

theorem melanie_dimes (initial_dimes : ℕ) : 
  (initial_dimes - 7 + 4 = 5) → initial_dimes = 8 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_l152_15224


namespace NUMINAMATH_CALUDE_alyssa_soccer_games_l152_15200

/-- Represents the number of soccer games Alyssa participated in over three years -/
def total_games (this_year_in_person this_year_online last_year_in_person next_year_in_person next_year_online : ℕ) : ℕ :=
  this_year_in_person + this_year_online + last_year_in_person + next_year_in_person + next_year_online

/-- Theorem stating that Alyssa will participate in 57 soccer games over three years -/
theorem alyssa_soccer_games : 
  total_games 11 8 13 15 10 = 57 := by
  sorry

#check alyssa_soccer_games

end NUMINAMATH_CALUDE_alyssa_soccer_games_l152_15200


namespace NUMINAMATH_CALUDE_tulip_petals_l152_15282

/-- Proves that each tulip has 3 petals given the conditions in Elena's garden --/
theorem tulip_petals (num_lilies : ℕ) (num_tulips : ℕ) (lily_petals : ℕ) (total_petals : ℕ)
  (h1 : num_lilies = 8)
  (h2 : num_tulips = 5)
  (h3 : lily_petals = 6)
  (h4 : total_petals = 63)
  (h5 : total_petals = num_lilies * lily_petals + num_tulips * (total_petals - num_lilies * lily_petals) / num_tulips) :
  (total_petals - num_lilies * lily_petals) / num_tulips = 3 := by
  sorry

#eval (63 - 8 * 6) / 5  -- This should output 3

end NUMINAMATH_CALUDE_tulip_petals_l152_15282


namespace NUMINAMATH_CALUDE_bat_pattern_area_l152_15214

/-- A bat pattern is composed of squares and triangles -/
structure BatPattern where
  large_squares : Nat
  medium_squares : Nat
  triangles : Nat
  large_square_area : ℝ
  medium_square_area : ℝ
  triangle_area : ℝ

/-- The total area of a bat pattern -/
def total_area (b : BatPattern) : ℝ :=
  b.large_squares * b.large_square_area +
  b.medium_squares * b.medium_square_area +
  b.triangles * b.triangle_area

/-- Theorem: The area of the specific bat pattern is 27 -/
theorem bat_pattern_area :
  ∃ (b : BatPattern),
    b.large_squares = 2 ∧
    b.medium_squares = 2 ∧
    b.triangles = 3 ∧
    b.large_square_area = 8 ∧
    b.medium_square_area = 4 ∧
    b.triangle_area = 1 ∧
    total_area b = 27 := by
  sorry

end NUMINAMATH_CALUDE_bat_pattern_area_l152_15214


namespace NUMINAMATH_CALUDE_angle_measure_problem_l152_15247

theorem angle_measure_problem (C D : ℝ) : 
  C + D = 180 →  -- angles are supplementary
  C = 12 * D →   -- C is 12 times D
  C = 2160 / 13  -- measure of angle C
  := by sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l152_15247


namespace NUMINAMATH_CALUDE_added_amount_after_doubling_and_tripling_l152_15233

theorem added_amount_after_doubling_and_tripling (x y : ℝ) : x = 5 → 3 * (2 * x + y) = 75 → y = 15 := by
  sorry

end NUMINAMATH_CALUDE_added_amount_after_doubling_and_tripling_l152_15233


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l152_15222

theorem divisible_by_eleven (a : ℝ) : 
  (∃ k : ℤ, (2 * 10^10 + a : ℝ) = 11 * k) → 
  0 ≤ a → 
  a < 11 → 
  a = 9 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l152_15222


namespace NUMINAMATH_CALUDE_continuity_at_6_delta_formula_l152_15219

def f (x : ℝ) : ℝ := 3 * x^2 + 7

theorem continuity_at_6 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 6| < δ → |f x - f 6| < ε :=
by
  sorry

theorem delta_formula (ε : ℝ) (h : ε > 0) : 
  ∃ δ > 0, δ = ε / 36 ∧ ∀ x, |x - 6| < δ → |f x - f 6| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_continuity_at_6_delta_formula_l152_15219


namespace NUMINAMATH_CALUDE_money_distribution_l152_15285

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (bc_sum : B + C = 350)
  (c_amount : C = 50) :
  A + C = 200 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l152_15285


namespace NUMINAMATH_CALUDE_animal_jumping_distances_l152_15207

-- Define the jumping distances for each animal
def grasshopper_jump : ℕ := 36

def frog_jump : ℕ := grasshopper_jump + 17

def mouse_jump : ℕ := frog_jump + 15

def kangaroo_jump : ℕ := 2 * mouse_jump

def rabbit_jump : ℕ := kangaroo_jump / 2 - 12

-- Theorem to prove the jumping distances
theorem animal_jumping_distances :
  grasshopper_jump = 36 ∧
  frog_jump = 53 ∧
  mouse_jump = 68 ∧
  kangaroo_jump = 136 ∧
  rabbit_jump = 56 := by
  sorry


end NUMINAMATH_CALUDE_animal_jumping_distances_l152_15207


namespace NUMINAMATH_CALUDE_sine_negative_half_solutions_l152_15275

theorem sine_negative_half_solutions : 
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, 0 ≤ x ∧ x < 2*π ∧ Real.sin x = -0.5) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_negative_half_solutions_l152_15275


namespace NUMINAMATH_CALUDE_ellipse_incenter_ratio_theorem_l152_15297

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Represents the foci of an ellipse -/
structure Foci (e : Ellipse) where
  left : Point
  right : Point

/-- Represents a triangle formed by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Represents the incenter of a triangle -/
def incenter (t : Triangle) : Point :=
  sorry -- Definition of incenter

/-- Checks if a point is on a line segment -/
def isOnSegment (p : Point) (a : Point) (b : Point) : Prop :=
  sorry -- Definition of being on a line segment

theorem ellipse_incenter_ratio_theorem
  (e : Ellipse) (m : Point) (f : Foci e) (p n : Point) :
  isOnEllipse e m →
  p = incenter (Triangle.mk m f.left f.right) →
  isOnSegment n f.left f.right →
  isOnSegment n m p →
  (m.x - n.x)^2 + (m.y - n.y)^2 > 0 →
  (n.x - p.x)^2 + (n.y - p.y)^2 > 0 →
  ∃ (r : ℝ), r > 0 ∧
    r = (m.x - n.x)^2 + (m.y - n.y)^2 / ((n.x - p.x)^2 + (n.y - p.y)^2) ∧
    r = (m.x - f.left.x)^2 + (m.y - f.left.y)^2 / ((f.left.x - p.x)^2 + (f.left.y - p.y)^2) ∧
    r = (m.x - f.right.x)^2 + (m.y - f.right.y)^2 / ((f.right.x - p.x)^2 + (f.right.y - p.y)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_incenter_ratio_theorem_l152_15297


namespace NUMINAMATH_CALUDE_bounded_sequence_with_distance_condition_l152_15284

theorem bounded_sequence_with_distance_condition :
  ∃ (a : ℕ → ℝ), 
    (∃ (C D : ℝ), ∀ n, C ≤ a n ∧ a n ≤ D) ∧ 
    (∀ (n m : ℕ), n > m → |a m - a n| ≥ 1 / (n - m : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_bounded_sequence_with_distance_condition_l152_15284


namespace NUMINAMATH_CALUDE_faster_train_length_l152_15298

/-- Calculates the length of a faster train given the speeds of two trains and the time taken for the faster train to cross a man in the slower train. -/
theorem faster_train_length
  (faster_speed slower_speed : ℝ)
  (crossing_time : ℝ)
  (h1 : faster_speed = 72)
  (h2 : slower_speed = 36)
  (h3 : crossing_time = 37)
  (h4 : faster_speed > slower_speed) :
  let relative_speed := faster_speed - slower_speed
  let relative_speed_ms := relative_speed * (1000 / 3600)
  relative_speed_ms * crossing_time = 370 := by
  sorry

#check faster_train_length

end NUMINAMATH_CALUDE_faster_train_length_l152_15298


namespace NUMINAMATH_CALUDE_unattainable_y_value_l152_15292

theorem unattainable_y_value (x y : ℝ) :
  x ≠ -5/4 →
  y = (2 - 3*x) / (4*x + 5) →
  y ≠ -3/4 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_y_value_l152_15292


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l152_15286

theorem product_of_sums_and_differences (P Q R S : ℝ) : 
  P = (Real.sqrt 2010 + Real.sqrt 2011) →
  Q = (-Real.sqrt 2010 - Real.sqrt 2011) →
  R = (Real.sqrt 2010 - Real.sqrt 2011) →
  S = (Real.sqrt 2011 - Real.sqrt 2010) →
  P * Q * R * S = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l152_15286


namespace NUMINAMATH_CALUDE_mrs_hilt_friday_miles_l152_15293

/-- Mrs. Hilt's running schedule for a week -/
structure RunningSchedule where
  monday : ℕ
  wednesday : ℕ
  friday : ℕ
  total : ℕ

/-- Theorem: Given Mrs. Hilt's running schedule, prove she ran 7 miles on Friday -/
theorem mrs_hilt_friday_miles (schedule : RunningSchedule) 
  (h1 : schedule.monday = 3)
  (h2 : schedule.wednesday = 2)
  (h3 : schedule.total = 12)
  (h4 : schedule.total = schedule.monday + schedule.wednesday + schedule.friday) :
  schedule.friday = 7 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_friday_miles_l152_15293


namespace NUMINAMATH_CALUDE_gcd_sum_and_sum_of_squares_l152_15254

theorem gcd_sum_and_sum_of_squares (a b : ℕ+) (h : Nat.Coprime a b) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_gcd_sum_and_sum_of_squares_l152_15254


namespace NUMINAMATH_CALUDE_modulus_of_z_l152_15225

theorem modulus_of_z (z : ℂ) (r θ : ℝ) (h1 : z + 1/z = r) (h2 : r = 2 * Real.sin θ) (h3 : |r| < 3) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l152_15225


namespace NUMINAMATH_CALUDE_calculate_expression_l152_15213

theorem calculate_expression : -(-1) + 3^2 / (1 - 4) * 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l152_15213


namespace NUMINAMATH_CALUDE_value_of_x_l152_15231

theorem value_of_x : (2009^2 - 2009) / 2009 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l152_15231


namespace NUMINAMATH_CALUDE_jessie_muffin_division_l152_15291

/-- The number of muffins each person receives when 35 muffins are divided equally among Jessie and her friends -/
def muffins_per_person (total_muffins : ℕ) (num_friends : ℕ) : ℕ :=
  total_muffins / (num_friends + 1)

/-- Theorem stating that when 35 muffins are divided equally among Jessie and her 6 friends, each person will receive 5 muffins -/
theorem jessie_muffin_division :
  muffins_per_person 35 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jessie_muffin_division_l152_15291


namespace NUMINAMATH_CALUDE_solve_equation_l152_15259

theorem solve_equation : ∃ x : ℚ, 5 * (x - 4) = 3 * (6 - 3 * x) + 9 ∧ x = 47 / 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l152_15259


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l152_15204

def complex_multiply (a b : ℂ) : ℂ := a * b

def imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_z (z : ℂ) :
  complex_multiply (1 + 3*Complex.I) z = 10 →
  imaginary_part z = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l152_15204


namespace NUMINAMATH_CALUDE_two_pairs_four_shoes_l152_15274

/-- Given that a person buys a certain number of pairs of shoes, and each pair consists of a certain number of shoes, calculate the total number of new shoes. -/
def total_new_shoes (pairs_bought : ℕ) (shoes_per_pair : ℕ) : ℕ :=
  pairs_bought * shoes_per_pair

/-- Theorem stating that buying 2 pairs of shoes, with 2 shoes per pair, results in 4 new shoes. -/
theorem two_pairs_four_shoes :
  total_new_shoes 2 2 = 4 := by
  sorry

#eval total_new_shoes 2 2

end NUMINAMATH_CALUDE_two_pairs_four_shoes_l152_15274


namespace NUMINAMATH_CALUDE_taxi_trip_distance_l152_15202

/-- Calculates the trip distance given the initial fee, per-increment charge, increment distance, and total charge -/
def calculate_trip_distance (initial_fee : ℚ) (per_increment_charge : ℚ) (increment_distance : ℚ) (total_charge : ℚ) : ℚ :=
  let distance_charge := total_charge - initial_fee
  let num_increments := distance_charge / per_increment_charge
  num_increments * increment_distance

theorem taxi_trip_distance :
  let initial_fee : ℚ := 9/4  -- $2.25
  let per_increment_charge : ℚ := 1/4  -- $0.25
  let increment_distance : ℚ := 2/5  -- 2/5 mile
  let total_charge : ℚ := 9/2  -- $4.50
  calculate_trip_distance initial_fee per_increment_charge increment_distance total_charge = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_taxi_trip_distance_l152_15202


namespace NUMINAMATH_CALUDE_larger_number_proof_larger_number_is_1891_l152_15270

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def has_at_most_three_decimal_places (n : ℕ) : Prop := n < 1000

theorem larger_number_proof (small : ℕ) (large : ℕ) : Prop :=
  large - small = 1355 ∧
  large / small = 6 ∧
  large % small = 15 ∧
  is_prime (sum_of_digits large) ∧
  has_at_most_three_decimal_places small ∧
  has_at_most_three_decimal_places large ∧
  large = 1891

theorem larger_number_is_1891 : ∃ (small : ℕ) (large : ℕ), larger_number_proof small large := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_larger_number_is_1891_l152_15270


namespace NUMINAMATH_CALUDE_perfect_square_pairs_l152_15251

theorem perfect_square_pairs (a b : ℤ) : 
  (∃ k : ℤ, a^2 + 4*b = k^2) ∧ (∃ m : ℤ, b^2 + 4*a = m^2) ↔ 
  (a = 0 ∧ b = 0) ∨ 
  (a = -4 ∧ b = -4) ∨ 
  (a = 4 ∧ b = -4) ∨ 
  (∃ k : ℕ, (a = k^2 ∧ b = 0) ∨ (a = 0 ∧ b = k^2)) ∨
  (a = -6 ∧ b = -5) ∨ 
  (a = -5 ∧ b = -6) ∨ 
  (∃ t : ℕ, (a = t ∧ b = 1 - t) ∨ (a = 1 - t ∧ b = t)) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_pairs_l152_15251


namespace NUMINAMATH_CALUDE_m_divided_by_8_l152_15237

theorem m_divided_by_8 (m : ℕ) (h : m = 16^1024) : m / 8 = 2^4093 := by
  sorry

end NUMINAMATH_CALUDE_m_divided_by_8_l152_15237


namespace NUMINAMATH_CALUDE_two_sided_iced_subcubes_count_l152_15264

/-- Represents a cube with icing on all sides -/
structure IcedCube where
  size : Nat
  deriving Repr

/-- Counts the number of subcubes with icing on exactly two sides -/
def count_two_sided_iced_subcubes (cube : IcedCube) : Nat :=
  sorry

/-- Theorem stating that a 5×5×5 iced cube has 40 subcubes with icing on exactly two sides -/
theorem two_sided_iced_subcubes_count (cube : IcedCube) (h : cube.size = 5) : 
  count_two_sided_iced_subcubes cube = 40 := by
  sorry

end NUMINAMATH_CALUDE_two_sided_iced_subcubes_count_l152_15264


namespace NUMINAMATH_CALUDE_initial_books_correct_l152_15288

/-- The number of books in the special collection at the beginning of the month. -/
def initial_books : ℕ := 75

/-- The number of books loaned out during the month. -/
def loaned_books : ℕ := 60

/-- The percentage of loaned books that are returned by the end of the month. -/
def return_rate : ℚ := 70 / 100

/-- The number of books in the special collection at the end of the month. -/
def final_books : ℕ := 57

/-- Theorem stating that the initial number of books is correct given the conditions. -/
theorem initial_books_correct : 
  initial_books = final_books + (loaned_books - (return_rate * loaned_books).floor) :=
sorry

end NUMINAMATH_CALUDE_initial_books_correct_l152_15288


namespace NUMINAMATH_CALUDE_house_transaction_loss_l152_15226

/-- Proves that given a house initially valued at $12000, after selling it at a 15% loss
and buying it back at a 20% gain, the original owner loses $240. -/
theorem house_transaction_loss (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ)
  (h1 : initial_value = 12000)
  (h2 : loss_percent = 0.15)
  (h3 : gain_percent = 0.20) :
  initial_value - (initial_value * (1 - loss_percent) * (1 + gain_percent)) = -240 := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_loss_l152_15226


namespace NUMINAMATH_CALUDE_expression_value_l152_15289

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (sum_zero : x + y + z = 0) (sum_prod_nonzero : x*y + x*z + y*z ≠ 0) :
  (x^6 + y^6 + z^6) / (x*y*z * (x*y + x*z + y*z)) = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l152_15289


namespace NUMINAMATH_CALUDE_sine_product_rational_l152_15205

theorem sine_product_rational : 
  66 * Real.sin (π / 18) * Real.sin (3 * π / 18) * Real.sin (5 * π / 18) * 
  Real.sin (7 * π / 18) * Real.sin (9 * π / 18) = 33 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sine_product_rational_l152_15205


namespace NUMINAMATH_CALUDE_subtraction_puzzle_sum_l152_15258

theorem subtraction_puzzle_sum :
  ∀ (P Q R S T : ℕ),
    P < 10 → Q < 10 → R < 10 → S < 10 → T < 10 →
    70000 + 1000 * Q + 200 + 10 * S + T - (10000 * P + 3000 + 100 * R + 90 + 6) = 22222 →
    P + Q + R + S + T = 29 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_puzzle_sum_l152_15258


namespace NUMINAMATH_CALUDE_power_sum_equality_l152_15234

theorem power_sum_equality : 2^300 + (-2^301) = -2^300 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l152_15234


namespace NUMINAMATH_CALUDE_tank_capacity_l152_15220

theorem tank_capacity (x : ℝ) 
  (h1 : x / 8 + 120 = x / 2) : x = 320 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l152_15220


namespace NUMINAMATH_CALUDE_trig_problem_l152_15272

theorem trig_problem (α : Real) 
  (h1 : α ∈ Set.Ioo (5 * Real.pi / 4) (3 * Real.pi / 2))
  (h2 : Real.tan α + 1 / Real.tan α = 8) : 
  Real.sin α * Real.cos α = 1 / 8 ∧ Real.sin α - Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l152_15272


namespace NUMINAMATH_CALUDE_tan_function_product_l152_15281

theorem tan_function_product (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + π / 2))) → 
  a * Real.tan (b * π / 8) = 1 → 
  a * b = 2 := by sorry

end NUMINAMATH_CALUDE_tan_function_product_l152_15281


namespace NUMINAMATH_CALUDE_fifth_power_sum_l152_15229

theorem fifth_power_sum (x : ℝ) (h : x + 1/x = -5) : x^5 + 1/x^5 = -2525 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l152_15229


namespace NUMINAMATH_CALUDE_initial_average_production_l152_15263

theorem initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) 
  (h1 : n = 14)
  (h2 : today_production = 90)
  (h3 : new_average = 62) :
  (n * (n + 1) * new_average - n * today_production) / n = 60 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_production_l152_15263


namespace NUMINAMATH_CALUDE_expression_evaluation_l152_15267

theorem expression_evaluation : (-3)^2 / 4 * (1/4) = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l152_15267


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l152_15294

theorem cos_pi_third_minus_alpha (α : Real) 
  (h : Real.sin (π / 6 + α) = 2 / 3) : 
  Real.cos (π / 3 - α) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l152_15294


namespace NUMINAMATH_CALUDE_pens_sold_in_garage_sale_l152_15246

/-- Given that Paul initially had 42 pens and after a garage sale had 19 pens left,
    prove that he sold 23 pens in the garage sale. -/
theorem pens_sold_in_garage_sale :
  let initial_pens : ℕ := 42
  let remaining_pens : ℕ := 19
  initial_pens - remaining_pens = 23 := by sorry

end NUMINAMATH_CALUDE_pens_sold_in_garage_sale_l152_15246


namespace NUMINAMATH_CALUDE_conditional_prob_B_given_A_l152_15212

/-- A fair coin is a coin with probability 1/2 for both heads and tails -/
structure FairCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  fair_heads : prob_heads = 1/2
  fair_tails : prob_tails = 1/2

/-- Event A: "the first appearance of heads" when a coin is tossed twice -/
def event_A (c : FairCoin) : ℝ := c.prob_heads

/-- Event B: "the second appearance of tails" -/
def event_B (c : FairCoin) : ℝ := c.prob_tails

/-- The probability of both events A and B occurring -/
def prob_AB (c : FairCoin) : ℝ := c.prob_heads * c.prob_tails

/-- Theorem: The conditional probability P(B|A) is 1/2 -/
theorem conditional_prob_B_given_A (c : FairCoin) : 
  prob_AB c / event_A c = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_conditional_prob_B_given_A_l152_15212


namespace NUMINAMATH_CALUDE_complementary_angles_are_acute_l152_15240

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- An angle is acute if it is less than 90 degrees -/
def acute (a : ℝ) : Prop := a < 90

/-- For any two complementary angles, both angles are always acute -/
theorem complementary_angles_are_acute (a b : ℝ) (h : complementary a b) : 
  acute a ∧ acute b := by sorry

end NUMINAMATH_CALUDE_complementary_angles_are_acute_l152_15240


namespace NUMINAMATH_CALUDE_algebraic_substitution_l152_15203

theorem algebraic_substitution (a b : ℝ) (h : a - 2 * b = 7) : 
  6 - 2 * a + 4 * b = -8 := by
sorry

end NUMINAMATH_CALUDE_algebraic_substitution_l152_15203


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l152_15295

theorem triangle_abc_proof (b c : ℝ) (A : Real) (hb : b = 1) (hc : c = 2) (hA : A = 60 * π / 180) :
  ∃ (a : ℝ) (B : Real),
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A ∧
    a = Real.sqrt 3 ∧
    Real.cos B = (a ^ 2 + c ^ 2 - b ^ 2) / (2 * a * c) ∧
    B = 30 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_proof_l152_15295


namespace NUMINAMATH_CALUDE_count_valid_lock_codes_valid_lock_codes_satisfy_conditions_l152_15253

/-- A function that generates valid lock codes based on the given conditions -/
def validLockCodes : List (Fin 9 × Fin 9 × Fin 9 × Fin 9 × Fin 9 × Fin 9) := sorry

/-- A function that checks if a number is even -/
def isEven (n : Fin 9) : Bool := sorry

/-- A function that checks if a number is prime -/
def isPrime (n : Fin 9) : Bool := sorry

/-- A function that checks if two numbers are consecutive -/
def isConsecutive (a b : Fin 9) : Bool := sorry

/-- The main theorem stating the number of valid lock codes -/
theorem count_valid_lock_codes : 
  validLockCodes.length = 1440 :=
by
  sorry

/-- Helper theorem: All codes in validLockCodes satisfy the given conditions -/
theorem valid_lock_codes_satisfy_conditions :
  ∀ code ∈ validLockCodes,
    let (d1, d2, d3, d4, d5, d6) := code
    isEven d1 ∧ 
    isEven d3 ∧ 
    isEven d5 ∧
    isPrime d2 ∧
    (¬ isConsecutive d1 d2) ∧
    (¬ isConsecutive d2 d3) ∧
    (¬ isConsecutive d3 d4) ∧
    (¬ isConsecutive d4 d5) ∧
    (¬ isConsecutive d5 d6) ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧
    d5 ≠ d6 :=
by
  sorry

end NUMINAMATH_CALUDE_count_valid_lock_codes_valid_lock_codes_satisfy_conditions_l152_15253


namespace NUMINAMATH_CALUDE_trajectory_of_intersecting_lines_l152_15265

/-- The trajectory of point P given two intersecting lines through A(-1,0) and B(1,0) with slope product -1 -/
theorem trajectory_of_intersecting_lines (x y : ℝ) :
  let k_AP := y / (x + 1)
  let k_BP := y / (x - 1)
  (k_AP * k_BP = -1) → (x ≠ -1 ∧ x ≠ 1) → (x^2 + y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_intersecting_lines_l152_15265


namespace NUMINAMATH_CALUDE_field_trip_lunch_cost_l152_15216

/-- Calculates the total cost of lunches for a field trip. -/
def total_lunch_cost (num_children : ℕ) (num_chaperones : ℕ) (num_teachers : ℕ) (num_extra : ℕ) (cost_per_lunch : ℕ) : ℕ :=
  (num_children + num_chaperones + num_teachers + num_extra) * cost_per_lunch

/-- Proves that the total cost of lunches for the given field trip is $308. -/
theorem field_trip_lunch_cost :
  total_lunch_cost 35 5 1 3 7 = 308 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_lunch_cost_l152_15216


namespace NUMINAMATH_CALUDE_sum_of_integers_l152_15238

theorem sum_of_integers (x y : ℕ+) (h1 : x.val^2 + y.val^2 = 250) (h2 : x.val * y.val = 108) :
  x.val + y.val = Real.sqrt 466 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l152_15238


namespace NUMINAMATH_CALUDE_only_f₂_passes_through_origin_l152_15255

-- Define the functions
def f₁ (x : ℝ) := x + 1
def f₂ (x : ℝ) := x^2
def f₃ (x : ℝ) := (x - 4)^2
noncomputable def f₄ (x : ℝ) := 1/x

-- Theorem statement
theorem only_f₂_passes_through_origin :
  (f₁ 0 ≠ 0) ∧ 
  (f₂ 0 = 0) ∧ 
  (f₃ 0 ≠ 0) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f₄ x| > 1/ε) :=
by sorry

end NUMINAMATH_CALUDE_only_f₂_passes_through_origin_l152_15255


namespace NUMINAMATH_CALUDE_number_added_proof_l152_15262

theorem number_added_proof (x y : ℝ) : x = 55 → (x / 5) + y = 21 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_added_proof_l152_15262


namespace NUMINAMATH_CALUDE_workers_days_per_week_l152_15257

/-- The number of toys produced per week -/
def weekly_production : ℕ := 5500

/-- The number of toys produced per day -/
def daily_production : ℕ := 1100

/-- The number of days worked per week -/
def days_worked : ℕ := weekly_production / daily_production

theorem workers_days_per_week :
  days_worked = 5 :=
sorry

end NUMINAMATH_CALUDE_workers_days_per_week_l152_15257


namespace NUMINAMATH_CALUDE_equation_system_solution_l152_15261

theorem equation_system_solution :
  ∀ x y : ℝ,
  x * y * (x + y) = 30 ∧
  x^3 + y^3 = 35 →
  ((x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l152_15261


namespace NUMINAMATH_CALUDE_sum_abc_l152_15287

theorem sum_abc (a b c : ℚ) 
  (eq1 : 2 * a + 3 * b + c = 27) 
  (eq2 : 4 * a + 6 * b + 5 * c = 71) : 
  a + b + c = 115 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_l152_15287


namespace NUMINAMATH_CALUDE_min_value_expression_l152_15250

theorem min_value_expression (m n : ℝ) (h : m - n^2 = 8) :
  58 ≤ m^2 - 3*n^2 + m - 14 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l152_15250


namespace NUMINAMATH_CALUDE_revenue_change_l152_15280

theorem revenue_change
  (T : ℝ) -- Original tax rate (as a percentage)
  (C : ℝ) -- Original consumption
  (h1 : T > 0)
  (h2 : C > 0) :
  let new_tax_rate := T * (1 - 0.19)
  let new_consumption := C * (1 + 0.15)
  let original_revenue := T / 100 * C
  let new_revenue := new_tax_rate / 100 * new_consumption
  (new_revenue - original_revenue) / original_revenue = -0.0685 :=
sorry

end NUMINAMATH_CALUDE_revenue_change_l152_15280


namespace NUMINAMATH_CALUDE_cookie_shop_purchases_l152_15260

/-- The number of different types of cookies available. -/
def num_cookies : ℕ := 7

/-- The number of different types of milk available. -/
def num_milk : ℕ := 4

/-- The total number of items Gamma and Delta purchase collectively. -/
def total_items : ℕ := 4

/-- The number of ways Gamma can choose items without repeats. -/
def gamma_choices (k : ℕ) : ℕ := Nat.choose (num_cookies + num_milk) k

/-- The number of ways Delta can choose k cookies with possible repeats. -/
def delta_choices (k : ℕ) : ℕ := 
  (Nat.choose num_cookies k) +  -- All different cookies
  (if k > 1 then num_cookies * (Nat.choose (k - 1 + num_cookies - 1) (num_cookies - 1)) else 0)  -- With repeats

/-- The total number of ways Gamma and Delta can purchase 4 items collectively. -/
def total_ways : ℕ := 
  (gamma_choices 4) +  -- Gamma 4, Delta 0
  (gamma_choices 3) * num_cookies +  -- Gamma 3, Delta 1
  (gamma_choices 2) * (delta_choices 2) +  -- Gamma 2, Delta 2
  (gamma_choices 1) * (delta_choices 3) +  -- Gamma 1, Delta 3
  (delta_choices 4)  -- Gamma 0, Delta 4

theorem cookie_shop_purchases : total_ways = 4096 := by
  sorry

end NUMINAMATH_CALUDE_cookie_shop_purchases_l152_15260


namespace NUMINAMATH_CALUDE_left_handed_jazz_no_glasses_l152_15218

/-- Represents a club with members having various characteristics -/
structure Club where
  total : Nat
  leftHanded : Nat
  jazzLovers : Nat
  rightHandedJazzDislikers : Nat
  glassesWearers : Nat

/-- The main theorem to be proved -/
theorem left_handed_jazz_no_glasses (c : Club)
  (h_total : c.total = 50)
  (h_left : c.leftHanded = 22)
  (h_jazz : c.jazzLovers = 35)
  (h_right_no_jazz : c.rightHandedJazzDislikers = 5)
  (h_glasses : c.glassesWearers = 10)
  (h_hand_exclusive : c.leftHanded + (c.total - c.leftHanded) = c.total)
  (h_glasses_independent : True) :
  ∃ x : Nat, x = 4 ∧ 
    x = c.leftHanded + c.jazzLovers - c.total + c.rightHandedJazzDislikers - c.glassesWearers :=
sorry


end NUMINAMATH_CALUDE_left_handed_jazz_no_glasses_l152_15218


namespace NUMINAMATH_CALUDE_exists_multiple_2020_with_sum_digits_multiple_2020_l152_15278

/-- Given a natural number n, returns the sum of its digits -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a natural number that is a multiple of 2020
    and has a sum of digits that is also a multiple of 2020 -/
theorem exists_multiple_2020_with_sum_digits_multiple_2020 :
  ∃ n : ℕ, 2020 ∣ n ∧ 2020 ∣ sumOfDigits n := by sorry

end NUMINAMATH_CALUDE_exists_multiple_2020_with_sum_digits_multiple_2020_l152_15278


namespace NUMINAMATH_CALUDE_square_land_area_l152_15241

/-- The area of a square land plot with side length 25 units is 625 square units. -/
theorem square_land_area (side_length : ℝ) (h : side_length = 25) : side_length ^ 2 = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_land_area_l152_15241


namespace NUMINAMATH_CALUDE_average_difference_theorem_l152_15249

/-- A school with students and teachers -/
structure School where
  num_students : ℕ
  num_teachers : ℕ
  class_sizes : List ℕ

/-- Calculate the average class size from a teacher's perspective -/
def teacher_average (school : School) : ℚ :=
  (school.class_sizes.sum : ℚ) / school.num_teachers

/-- Calculate the average class size from a student's perspective -/
def student_average (school : School) : ℚ :=
  (school.class_sizes.map (λ size => size * size)).sum / school.num_students

/-- The main theorem to prove -/
theorem average_difference_theorem (school : School) 
    (h1 : school.num_students = 120)
    (h2 : school.num_teachers = 5)
    (h3 : school.class_sizes = [60, 30, 20, 5, 5])
    (h4 : school.class_sizes.sum = school.num_students) : 
    teacher_average school - student_average school = -17.25 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_theorem_l152_15249


namespace NUMINAMATH_CALUDE_circle_radius_proof_l152_15210

theorem circle_radius_proof (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 25) :
  ∃ r : ℝ, r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 50 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l152_15210


namespace NUMINAMATH_CALUDE_product_minus_one_divisible_by_ten_l152_15242

theorem product_minus_one_divisible_by_ten :
  ∃ k : ℤ, 11 * 21 * 31 * 41 * 51 - 1 = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_product_minus_one_divisible_by_ten_l152_15242


namespace NUMINAMATH_CALUDE_one_real_zero_l152_15230

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- State the theorem
theorem one_real_zero : ∃! x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_one_real_zero_l152_15230


namespace NUMINAMATH_CALUDE_correct_sum_is_45250_l152_15273

/-- Represents the sum with errors --/
def incorrect_sum : ℕ := 52000

/-- Represents the error in the first number's tens place --/
def tens_error : ℤ := 50

/-- Represents the error in the first number's hundreds place --/
def hundreds_error : ℤ := -300

/-- Represents the error in the second number's thousands place --/
def thousands_error : ℤ := 7000

/-- The total error introduced by the mistakes --/
def total_error : ℤ := tens_error + hundreds_error + thousands_error

/-- The correct sum after adjusting for errors --/
def correct_sum : ℕ := incorrect_sum - total_error.toNat

theorem correct_sum_is_45250 : correct_sum = 45250 := by
  sorry

end NUMINAMATH_CALUDE_correct_sum_is_45250_l152_15273


namespace NUMINAMATH_CALUDE_real_part_of_complex_expression_l152_15271

theorem real_part_of_complex_expression : Complex.re (1 + 2 / (Complex.I + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_expression_l152_15271


namespace NUMINAMATH_CALUDE_matthew_egg_rolls_l152_15236

/-- Given the egg roll consumption of Alvin, Patrick, and Matthew, prove that Matthew ate 6 egg rolls. -/
theorem matthew_egg_rolls (alvin patrick matthew : ℕ) : 
  alvin = 4 →
  patrick = alvin / 2 →
  matthew = 3 * patrick →
  matthew = 6 := by
sorry

end NUMINAMATH_CALUDE_matthew_egg_rolls_l152_15236


namespace NUMINAMATH_CALUDE_balls_in_box_perfect_square_l152_15296

theorem balls_in_box_perfect_square (a v : ℕ) : 
  (2 * a * v : ℚ) / ((a + v) * (a + v - 1) / 2) = 1 / 2 → 
  ∃ n : ℕ, a + v = n^2 := by
sorry

end NUMINAMATH_CALUDE_balls_in_box_perfect_square_l152_15296


namespace NUMINAMATH_CALUDE_base_for_five_digit_100_l152_15221

theorem base_for_five_digit_100 :
  ∃! (b : ℕ), b > 1 ∧ b^4 ≤ 100 ∧ 100 < b^5 :=
by sorry

end NUMINAMATH_CALUDE_base_for_five_digit_100_l152_15221
