import Mathlib

namespace NUMINAMATH_CALUDE_will_chocolate_pieces_l1793_179318

/-- Calculates the number of chocolate pieces Will has left after giving some boxes away. -/
def chocolate_pieces_left (total_boxes : ℕ) (boxes_given : ℕ) (pieces_per_box : ℕ) : ℕ :=
  (total_boxes - boxes_given) * pieces_per_box

/-- Proves that Will has 16 pieces of chocolate left after giving some boxes to his brother. -/
theorem will_chocolate_pieces : chocolate_pieces_left 7 3 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_will_chocolate_pieces_l1793_179318


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1793_179309

theorem simplify_and_evaluate (a : ℤ) (h : a = 2023) :
  a * (1 - 2 * a) + 2 * (a + 1) * (a - 1) = 2021 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1793_179309


namespace NUMINAMATH_CALUDE_triangle_angle_from_sides_and_area_l1793_179376

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    prove that if a = 2√3, b = 2, and the area S = √3, then C = π/6 -/
theorem triangle_angle_from_sides_and_area 
  (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  a = 2 * Real.sqrt 3 →
  b = 2 →
  S = Real.sqrt 3 →
  S = 1/2 * a * b * Real.sin C →
  C = π/6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_from_sides_and_area_l1793_179376


namespace NUMINAMATH_CALUDE_ram_krish_work_time_l1793_179393

/-- Represents the efficiency of a worker -/
structure Efficiency : Type :=
  (value : ℝ)

/-- Represents the time taken to complete a task -/
structure Time : Type :=
  (days : ℝ)

/-- Represents the amount of work in a task -/
structure Work : Type :=
  (amount : ℝ)

/-- The theorem stating the relationship between Ram and Krish's efficiency and their combined work time -/
theorem ram_krish_work_time 
  (ram_efficiency : Efficiency)
  (krish_efficiency : Efficiency)
  (ram_alone_time : Time)
  (task : Work)
  (h1 : ram_efficiency.value = (1 / 2) * krish_efficiency.value)
  (h2 : ram_alone_time.days = 30)
  (h3 : task.amount = ram_efficiency.value * ram_alone_time.days) :
  ∃ (combined_time : Time),
    combined_time.days = 10 ∧
    task.amount = (ram_efficiency.value + krish_efficiency.value) * combined_time.days :=
sorry

end NUMINAMATH_CALUDE_ram_krish_work_time_l1793_179393


namespace NUMINAMATH_CALUDE_tangent_range_l1793_179362

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The equation for the tangent line passing through (1, m) and touching the curve at x₀ --/
def tangent_equation (x₀ m : ℝ) : Prop :=
  (x₀^3 - 3*x₀ - m) / (x₀ - 1) = 3*x₀^2 - 3

/-- The condition for exactly three tangent lines --/
def three_tangents (m : ℝ) : Prop :=
  ∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    tangent_equation x₁ m ∧ tangent_equation x₂ m ∧ tangent_equation x₃ m

/-- The main theorem --/
theorem tangent_range :
  ∀ m : ℝ, m ≠ -2 → three_tangents m → -3 < m ∧ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_range_l1793_179362


namespace NUMINAMATH_CALUDE_point_A_coordinates_l1793_179368

/-- Given a point A with coordinates (2a-9, 1-2a), prove that if A is moved 5 units
    to the right and lands on the y-axis, then its new coordinates are (-5, -3) -/
theorem point_A_coordinates (a : ℝ) :
  let initial_A : ℝ × ℝ := (2*a - 9, 1 - 2*a)
  let moved_A : ℝ × ℝ := (2*a - 4, 1 - 2*a)  -- Moved 5 units to the right
  moved_A.1 = 0 →  -- Lands on y-axis
  moved_A = (-5, -3) :=
by sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l1793_179368


namespace NUMINAMATH_CALUDE_penny_bakery_revenue_l1793_179323

/-- Calculates the total money made from selling cheesecakes -/
def total_money_made (price_per_slice : ℕ) (slices_per_cake : ℕ) (cakes_sold : ℕ) : ℕ :=
  price_per_slice * slices_per_cake * cakes_sold

/-- Theorem: Penny's bakery makes $294 from selling 7 cheesecakes -/
theorem penny_bakery_revenue : total_money_made 7 6 7 = 294 := by
  sorry

end NUMINAMATH_CALUDE_penny_bakery_revenue_l1793_179323


namespace NUMINAMATH_CALUDE_factorial_gcd_property_l1793_179388

theorem factorial_gcd_property (m n : ℕ) (h : m > n) :
  Nat.gcd (Nat.factorial n) (Nat.factorial m) = Nat.factorial n := by
  sorry

end NUMINAMATH_CALUDE_factorial_gcd_property_l1793_179388


namespace NUMINAMATH_CALUDE_number_of_divisors_30030_l1793_179398

theorem number_of_divisors_30030 : Nat.card (Nat.divisors 30030) = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_30030_l1793_179398


namespace NUMINAMATH_CALUDE_divisibility_of_power_plus_minus_one_l1793_179367

theorem divisibility_of_power_plus_minus_one (n : ℕ) (h : ¬ 17 ∣ n) :
  17 ∣ (n^8 + 1) ∨ 17 ∣ (n^8 - 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_power_plus_minus_one_l1793_179367


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l1793_179342

theorem opposite_of_negative_2023 : -(-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l1793_179342


namespace NUMINAMATH_CALUDE_paulo_children_ages_l1793_179356

theorem paulo_children_ages :
  ∃! (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 12 ∧ a * b * c = 30 :=
by sorry

end NUMINAMATH_CALUDE_paulo_children_ages_l1793_179356


namespace NUMINAMATH_CALUDE_g_minimum_value_l1793_179383

open Real

noncomputable def g (x : ℝ) : ℝ :=
  x + (2*x)/(x^2 + 1) + (x*(x + 3))/(x^2 + 3) + (3*(x + 1))/(x*(x^2 + 3))

theorem g_minimum_value (x : ℝ) (hx : x > 0) : g x ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_g_minimum_value_l1793_179383


namespace NUMINAMATH_CALUDE_fourth_term_is_negative_24_l1793_179371

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n-1)

-- Define the conditions of our specific sequence
def sequence_conditions (x : ℝ) : Prop :=
  ∃ (r : ℝ), 
    geometric_sequence x r 2 = 3*x + 3 ∧
    geometric_sequence x r 3 = 6*x + 6

-- Theorem statement
theorem fourth_term_is_negative_24 :
  ∀ x : ℝ, sequence_conditions x → geometric_sequence x 2 4 = -24 :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_is_negative_24_l1793_179371


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1793_179374

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + |x| ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 + |x₀| < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1793_179374


namespace NUMINAMATH_CALUDE_intersection_condition_l1793_179370

def A : Set (ℕ × ℝ) := {p | 3 * p.1 + p.2 - 2 = 0}

def B (k : ℤ) : Set (ℕ × ℝ) := {p | k * (p.1^2 - p.1 + 1) - p.2 = 0}

theorem intersection_condition (k : ℤ) : 
  k ≠ 0 → (∃ p : ℕ × ℝ, p ∈ A ∩ B k) → k = -1 ∨ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l1793_179370


namespace NUMINAMATH_CALUDE_matrix_power_four_l1793_179382

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -1; 1, 0]

theorem matrix_power_four :
  A ^ 4 = !![(-1 : ℤ), 1; -1, 0] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l1793_179382


namespace NUMINAMATH_CALUDE_veggies_expense_correct_l1793_179384

/-- Calculates the amount spent on veggies given the total amount brought,
    expenses on other items, and the amount left after shopping. -/
def amount_spent_on_veggies (total_brought : ℕ) (meat_expense : ℕ) (chicken_expense : ℕ)
                             (eggs_expense : ℕ) (dog_food_expense : ℕ) (amount_left : ℕ) : ℕ :=
  total_brought - (meat_expense + chicken_expense + eggs_expense + dog_food_expense + amount_left)

/-- Proves that the amount Trisha spent on veggies is correct given the problem conditions. -/
theorem veggies_expense_correct (total_brought : ℕ) (meat_expense : ℕ) (chicken_expense : ℕ)
                                 (eggs_expense : ℕ) (dog_food_expense : ℕ) (amount_left : ℕ)
                                 (h1 : total_brought = 167)
                                 (h2 : meat_expense = 17)
                                 (h3 : chicken_expense = 22)
                                 (h4 : eggs_expense = 5)
                                 (h5 : dog_food_expense = 45)
                                 (h6 : amount_left = 35) :
  amount_spent_on_veggies total_brought meat_expense chicken_expense eggs_expense dog_food_expense amount_left = 43 :=
by
  sorry

#eval amount_spent_on_veggies 167 17 22 5 45 35

end NUMINAMATH_CALUDE_veggies_expense_correct_l1793_179384


namespace NUMINAMATH_CALUDE_train_crossing_time_l1793_179305

/-- A train crosses a platform in a certain time -/
theorem train_crossing_time 
  (train_speed : ℝ) 
  (pole_crossing_time : ℝ) 
  (platform_crossing_time : ℝ) : 
  train_speed = 36 → 
  pole_crossing_time = 12 → 
  platform_crossing_time = 49.996960243180546 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1793_179305


namespace NUMINAMATH_CALUDE_integral_proof_l1793_179353

open Real

noncomputable def f (x : ℝ) : ℝ := 3*x + log (abs x) + 2*log (abs (x+1)) - log (abs (x-2))

theorem integral_proof (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -1) (h3 : x ≠ 2) : 
  deriv f x = (3*x^3 - x^2 - 12*x - 2) / (x*(x+1)*(x-2)) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l1793_179353


namespace NUMINAMATH_CALUDE_octagon_quad_area_ratio_l1793_179358

/-- Regular octagon with vertices ABCDEFGH -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- Quadrilateral formed by connecting alternate vertices of the octagon -/
def alternateVerticesQuad (octagon : RegularOctagon) : Fin 4 → ℝ × ℝ :=
  fun i => octagon.vertices (2 * i)

/-- Area of a polygon given its vertices -/
def polygonArea (vertices : Fin n → ℝ × ℝ) : ℝ := sorry

theorem octagon_quad_area_ratio 
  (octagon : RegularOctagon) 
  (n : ℝ) 
  (m : ℝ) 
  (hn : n = polygonArea octagon.vertices) 
  (hm : m = polygonArea (alternateVerticesQuad octagon)) :
  m / n = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_octagon_quad_area_ratio_l1793_179358


namespace NUMINAMATH_CALUDE_line_circle_intersection_k_range_l1793_179349

/-- Given a line y = kx + 3 intersecting a circle (x - 2)² + (y - 3)² = 4 at points M and N,
    if |MN| ≥ 2√3, then -√3/3 ≤ k ≤ √3/3 -/
theorem line_circle_intersection_k_range (k : ℝ) (M N : ℝ × ℝ) :
  (∀ x y, y = k * x + 3 → (x - 2)^2 + (y - 3)^2 = 4 → (x, y) = M ∨ (x, y) = N) →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12 →
  -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_k_range_l1793_179349


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l1793_179365

theorem rectangle_area_diagonal_relation (l w d : ℝ) (h1 : l / w = 5 / 4) (h2 : l^2 + w^2 = d^2) (h3 : d = 13) :
  ∃ k : ℝ, l * w = k * d^2 ∧ k = 20 / 41 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l1793_179365


namespace NUMINAMATH_CALUDE_group_size_calculation_l1793_179314

/-- Given a group of people where:
  1. The average weight increase is 1.5 kg
  2. The total weight increase is 12 kg (77 kg - 65 kg)
  3. The total weight increase equals the average weight increase multiplied by the number of people
  Prove that the number of people in the group is 8. -/
theorem group_size_calculation (avg_increase : ℝ) (total_increase : ℝ) :
  avg_increase = 1.5 →
  total_increase = 12 →
  total_increase = avg_increase * 8 :=
by sorry

end NUMINAMATH_CALUDE_group_size_calculation_l1793_179314


namespace NUMINAMATH_CALUDE_binomial_coeff_equality_l1793_179363

def binomial_coeff (n m : ℕ) : ℕ := Nat.choose n m

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem binomial_coeff_equality (n m : ℕ) :
  binomial_coeff n (m - 1) = binomial_coeff (n - 1) m ↔
  ∃ k : ℕ, n = fibonacci (2 * k) * fibonacci (2 * k + 1) ∧
            m = fibonacci (2 * k) * fibonacci (2 * k - 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_coeff_equality_l1793_179363


namespace NUMINAMATH_CALUDE_hotdog_eating_competition_l1793_179354

theorem hotdog_eating_competition (x y z : ℕ+) :
  y = 1 ∧
  x = z - 2 ∧
  6 * ((2*x - 3) + (3*x - y) + (4*x + z) + (x^2 - 5) + (3*y + 5*z) + (x*(y+z)) + ((x^2)+(y*z) - 2) + (x^3*y^2*z-15)) = 10000 →
  ∃ (hotdogs : ℕ), hotdogs = 6 * (x^3 * y^2 * z - 15) :=
by sorry

end NUMINAMATH_CALUDE_hotdog_eating_competition_l1793_179354


namespace NUMINAMATH_CALUDE_quadratic_maximum_l1793_179385

-- Define the quadratic function
def quadratic (p r s x : ℝ) : ℝ := x^2 + p*x + r + s

-- State the theorem
theorem quadratic_maximum (p s : ℝ) :
  let r : ℝ := 10 - s + p^2/4
  (∀ x, quadratic p r s x ≤ 10) ∧ 
  (quadratic p r s (-p/2) = 10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l1793_179385


namespace NUMINAMATH_CALUDE_right_triangle_log_identity_l1793_179338

theorem right_triangle_log_identity 
  (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle_inequality : c > b) :
  Real.log a / Real.log (b + c) + Real.log a / Real.log (c - b) = 
  2 * (Real.log a / Real.log (c + b)) * (Real.log a / Real.log (c - b)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_log_identity_l1793_179338


namespace NUMINAMATH_CALUDE_closest_vector_to_origin_l1793_179387

/-- The vector v is closest to the origin when t = 1/13 -/
theorem closest_vector_to_origin (t : ℝ) : 
  let v : ℝ × ℝ × ℝ := (1 + 3*t, 2 - 4*t, 3 + t)
  let a : ℝ × ℝ × ℝ := (0, 0, 0)
  let direction : ℝ × ℝ × ℝ := (3, -4, 1)
  (∀ s : ℝ, ‖v - a‖ ≤ ‖(1 + 3*s, 2 - 4*s, 3 + s) - a‖) ↔ t = 1/13 :=
by sorry


end NUMINAMATH_CALUDE_closest_vector_to_origin_l1793_179387


namespace NUMINAMATH_CALUDE_root_of_two_equations_l1793_179337

theorem root_of_two_equations (p q r s t k : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (eq1 : p * k^5 + q * k^4 + r * k^3 + s * k^2 + t * k + p = 0)
  (eq2 : q * k^5 + r * k^4 + s * k^3 + t * k^2 + p * k + q = 0) :
  k = 1 ∨ k = Complex.exp (Complex.I * π / 3) ∨ 
  k = Complex.exp (-Complex.I * π / 3) ∨ k = -1 ∨ 
  k = Complex.exp (2 * Complex.I * π / 3) ∨ 
  k = Complex.exp (-2 * Complex.I * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_root_of_two_equations_l1793_179337


namespace NUMINAMATH_CALUDE_felix_drive_l1793_179394

theorem felix_drive (average_speed : ℝ) (drive_time : ℝ) : 
  average_speed = 66 → drive_time = 4 → (2 * average_speed) * drive_time = 528 := by
  sorry

end NUMINAMATH_CALUDE_felix_drive_l1793_179394


namespace NUMINAMATH_CALUDE_divisible_by_five_l1793_179366

theorem divisible_by_five (a b : ℕ) : 
  (∃ k : ℕ, a * b = 5 * k) → (∃ m : ℕ, a = 5 * m) ∨ (∃ n : ℕ, b = 5 * n) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l1793_179366


namespace NUMINAMATH_CALUDE_square_of_two_minus_sqrt_three_l1793_179316

theorem square_of_two_minus_sqrt_three : (2 - Real.sqrt 3) ^ 2 = 7 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_of_two_minus_sqrt_three_l1793_179316


namespace NUMINAMATH_CALUDE_committee_formations_count_l1793_179380

/-- The number of ways to form a committee of 5 members from a club of 15 people,
    where the committee must include exactly 2 designated roles and 3 additional members. -/
def committeeFormations (clubSize : ℕ) (committeeSize : ℕ) (designatedRoles : ℕ) (additionalMembers : ℕ) : ℕ :=
  (clubSize * (clubSize - 1)) * Nat.choose (clubSize - designatedRoles) additionalMembers

/-- Theorem stating that the number of committee formations
    for the given conditions is 60060. -/
theorem committee_formations_count :
  committeeFormations 15 5 2 3 = 60060 := by
  sorry

end NUMINAMATH_CALUDE_committee_formations_count_l1793_179380


namespace NUMINAMATH_CALUDE_simplify_calculations_l1793_179346

theorem simplify_calculations :
  (329 * 101 = 33229) ∧
  (54 * 98 + 46 * 98 = 9800) ∧
  (98 * 125 = 12250) ∧
  (37 * 29 + 37 = 1110) := by
  sorry

end NUMINAMATH_CALUDE_simplify_calculations_l1793_179346


namespace NUMINAMATH_CALUDE_race_earnings_theorem_l1793_179311

/-- Calculates the average earnings per minute for the race winner -/
def average_earnings_per_minute (race_duration : ℕ) (lap_distance : ℕ) (gift_rate : ℚ) (winner_laps : ℕ) : ℚ :=
  let total_distance := winner_laps * lap_distance
  let total_earnings := (total_distance / 100) * gift_rate
  total_earnings / race_duration

/-- Theorem stating that the average earnings per minute is $7 given the race conditions -/
theorem race_earnings_theorem :
  average_earnings_per_minute 12 100 (7/2) 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_race_earnings_theorem_l1793_179311


namespace NUMINAMATH_CALUDE_yasmin_bank_balance_l1793_179348

theorem yasmin_bank_balance (deposit : ℝ) (new_balance : ℝ) (initial_balance : ℝ) : 
  deposit = 50 ∧ 
  deposit = (1/4) * new_balance ∧ 
  initial_balance = new_balance - deposit →
  initial_balance = 150 := by
sorry

end NUMINAMATH_CALUDE_yasmin_bank_balance_l1793_179348


namespace NUMINAMATH_CALUDE_house_sale_profit_l1793_179373

theorem house_sale_profit (initial_value : ℝ) (first_sale_profit_percent : ℝ) (second_sale_loss_percent : ℝ) : 
  initial_value = 200000 ∧ 
  first_sale_profit_percent = 15 ∧ 
  second_sale_loss_percent = 20 → 
  (initial_value * (1 + first_sale_profit_percent / 100)) * (1 - second_sale_loss_percent / 100) - initial_value = 46000 :=
by sorry

end NUMINAMATH_CALUDE_house_sale_profit_l1793_179373


namespace NUMINAMATH_CALUDE_function_inequality_l1793_179332

-- Define a real-valued function f on ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State that f' is the derivative of f
variable (hf' : ∀ x, HasDerivAt f (f' x) x)

-- State that f'(x) < f(x) for all x ∈ ℝ
variable (h : ∀ x, f' x < f x)

-- Theorem statement
theorem function_inequality (f f' : ℝ → ℝ) (hf' : ∀ x, HasDerivAt f (f' x) x) (h : ∀ x, f' x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2001 < Real.exp 2001 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1793_179332


namespace NUMINAMATH_CALUDE_giants_playoff_fraction_l1793_179344

theorem giants_playoff_fraction :
  let games_played : ℕ := 20
  let games_won : ℕ := 12
  let games_left : ℕ := 10
  let additional_wins_needed : ℕ := 8
  let total_games : ℕ := games_played + games_left
  let total_wins_needed : ℕ := games_won + additional_wins_needed
  (total_wins_needed : ℚ) / total_games = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_giants_playoff_fraction_l1793_179344


namespace NUMINAMATH_CALUDE_smallest_sum_of_two_primes_above_70_l1793_179396

theorem smallest_sum_of_two_primes_above_70 : 
  ∃ (p q : Nat), 
    Prime p ∧ 
    Prime q ∧ 
    p > 70 ∧ 
    q > 70 ∧ 
    p ≠ q ∧ 
    p + q = 144 ∧ 
    (∀ (r s : Nat), Prime r → Prime s → r > 70 → s > 70 → r ≠ s → r + s ≥ 144) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_two_primes_above_70_l1793_179396


namespace NUMINAMATH_CALUDE_triangle_properties_l1793_179372

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 + t.b^2 - t.c^2 = 4 * t.area) 
  (h2 : t.c = Real.sqrt 2) : 
  (t.C = Real.pi / 4) ∧ 
  (-1 < t.a - (Real.sqrt 2 / 2) * t.b) ∧ 
  (t.a - (Real.sqrt 2 / 2) * t.b < Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1793_179372


namespace NUMINAMATH_CALUDE_problem_solution_l1793_179341

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := 
  fun (a, b) (c, d) ↦ (a - c, b + d)

/-- Theorem stating that given the conditions, a = 2 -/
theorem problem_solution : 
  ∃ (a b : ℤ), star (5, 2) (1, 1) = (a, b) ∧ star (a, b) (0, 2) = (2, 5) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1793_179341


namespace NUMINAMATH_CALUDE_solve_for_t_l1793_179351

-- Define the variables
variable (s t : ℝ)

-- State the theorem
theorem solve_for_t (eq1 : 7 * s + 3 * t = 82) (eq2 : s = 2 * t - 3) : t = 103 / 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l1793_179351


namespace NUMINAMATH_CALUDE_unique_triple_l1793_179395

theorem unique_triple : 
  ∃! (a b c : ℕ), 
    (10 ≤ b ∧ b ≤ 99) ∧ 
    (10 ≤ c ∧ c ≤ 99) ∧ 
    (10^4 * a + 100 * b + c = (a + b + c)^3) ∧
    a = 9 ∧ b = 11 ∧ c = 25 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l1793_179395


namespace NUMINAMATH_CALUDE_gasoline_added_l1793_179357

theorem gasoline_added (tank_capacity : ℝ) (initial_fill : ℝ) (final_fill : ℝ) : tank_capacity = 54 → initial_fill = 3/4 → final_fill = 9/10 → (final_fill - initial_fill) * tank_capacity = 8.1 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_added_l1793_179357


namespace NUMINAMATH_CALUDE_min_value_problem_l1793_179390

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y^3 = 16/9) :
  3 * x + y ≥ 8/3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀^3 = 16/9 ∧ 3 * x₀ + y₀ = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l1793_179390


namespace NUMINAMATH_CALUDE_special_gp_ratio_equation_special_gp_ratio_approx_l1793_179381

/-- A geometric progression with positive terms where any term is equal to the sum of the next three following terms -/
structure SpecialGP where
  a : ℝ
  r : ℝ
  a_pos : a > 0
  r_pos : r > 0
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

/-- The common ratio of a SpecialGP satisfies a cubic equation -/
theorem special_gp_ratio_equation (gp : SpecialGP) :
  gp.r^3 + gp.r^2 + gp.r - 1 = 0 := by
  sorry

/-- The solution to the cubic equation is approximately 0.5437 -/
theorem special_gp_ratio_approx (gp : SpecialGP) :
  ∃ ε > 0, |gp.r - 0.5437| < ε := by
  sorry

end NUMINAMATH_CALUDE_special_gp_ratio_equation_special_gp_ratio_approx_l1793_179381


namespace NUMINAMATH_CALUDE_farm_milk_production_l1793_179302

/-- Calculates the weekly milk production for a farm -/
def weekly_milk_production (num_cows : ℕ) (milk_per_cow_per_day : ℕ) : ℕ :=
  num_cows * milk_per_cow_per_day * 7

/-- Theorem: A farm with 52 cows, each producing 5 liters of milk per day, produces 1820 liters of milk in a week -/
theorem farm_milk_production :
  weekly_milk_production 52 5 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_farm_milk_production_l1793_179302


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l1793_179313

theorem unique_integer_satisfying_conditions :
  ∃! (x : ℤ), 1 < x ∧ x < 9 ∧ 2 < x ∧ x < 15 ∧ -1 < x ∧ x < 7 ∧ 0 < x ∧ x < 4 ∧ x + 1 < 5 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l1793_179313


namespace NUMINAMATH_CALUDE_divisor_exists_l1793_179319

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem divisor_exists : ∃ d : ℕ, 
  d > 0 ∧ 
  is_prime (9453 / d) ∧ 
  is_perfect_square (9453 % d) ∧ 
  d = 61 := by
sorry

end NUMINAMATH_CALUDE_divisor_exists_l1793_179319


namespace NUMINAMATH_CALUDE_square_side_equals_pi_l1793_179310

theorem square_side_equals_pi :
  ∀ x : ℝ,
  (4 * x = 2 * π * 2) →
  x = π :=
by
  sorry

end NUMINAMATH_CALUDE_square_side_equals_pi_l1793_179310


namespace NUMINAMATH_CALUDE_cosine_theorem_trirectangular_angle_l1793_179331

-- Define the trirectangular angle
structure TrirectangularAngle where
  α : Real  -- plane angle opposite to SA
  β : Real  -- plane angle opposite to SB
  γ : Real  -- plane angle opposite to SC
  A : Real  -- dihedral angle at SA
  B : Real  -- dihedral angle at SB
  C : Real  -- dihedral angle at SC

-- State the theorem
theorem cosine_theorem_trirectangular_angle (t : TrirectangularAngle) :
  Real.cos t.α = Real.cos t.A * Real.cos t.B + Real.cos t.B * Real.cos t.C + Real.cos t.C * Real.cos t.A := by
  sorry

end NUMINAMATH_CALUDE_cosine_theorem_trirectangular_angle_l1793_179331


namespace NUMINAMATH_CALUDE_f_value_at_one_l1793_179345

def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem f_value_at_one (m : ℝ) :
  (∀ x ≥ -2, ∀ y ≥ -2, x < y → f m x < f m y) →
  (∀ x ≤ -2, ∀ y ≤ -2, x < y → f m x > f m y) →
  f m 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_one_l1793_179345


namespace NUMINAMATH_CALUDE_max_product_constraint_l1793_179379

theorem max_product_constraint (a b : ℝ) : 
  a > 0 → b > 0 → 3 * a + 8 * b = 72 → ab ≤ 54 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 8 * b₀ = 72 ∧ a₀ * b₀ = 54 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l1793_179379


namespace NUMINAMATH_CALUDE_routes_between_plains_cities_l1793_179303

theorem routes_between_plains_cities 
  (total_cities : ℕ) 
  (mountainous_cities : ℕ) 
  (plains_cities : ℕ) 
  (total_routes : ℕ) 
  (mountainous_routes : ℕ) 
  (h1 : total_cities = 100)
  (h2 : mountainous_cities = 30)
  (h3 : plains_cities = 70)
  (h4 : mountainous_cities + plains_cities = total_cities)
  (h5 : total_routes = 150)
  (h6 : mountainous_routes = 21) :
  total_routes - mountainous_routes - (mountainous_cities * 3 - mountainous_routes * 2) / 2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_routes_between_plains_cities_l1793_179303


namespace NUMINAMATH_CALUDE_det_A_zero_l1793_179343

theorem det_A_zero (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℂ) 
  (h : A = A * B - B * A + A^2 * B - 2 * A * B * A + B * A^2 + A^2 * B * A - A * B * A^2) : 
  Matrix.det A = 0 := by
sorry

end NUMINAMATH_CALUDE_det_A_zero_l1793_179343


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1793_179307

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i^3) / (1 - i) = (1 / 2 : ℂ) - (1 / 2 : ℂ) * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1793_179307


namespace NUMINAMATH_CALUDE_inverse_proportion_y_comparison_l1793_179378

/-- Given two points on the inverse proportion function y = -5/x,
    where the x-coordinate of the first point is positive and
    the x-coordinate of the second point is negative,
    prove that the y-coordinate of the first point is less than
    the y-coordinate of the second point. -/
theorem inverse_proportion_y_comparison
  (x₁ x₂ y₁ y₂ : ℝ)
  (h1 : y₁ = -5 / x₁)
  (h2 : y₂ = -5 / x₂)
  (h3 : x₁ > 0)
  (h4 : x₂ < 0) :
  y₁ < y₂ :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_comparison_l1793_179378


namespace NUMINAMATH_CALUDE_candy_distribution_l1793_179327

theorem candy_distribution (total_children : ℕ) (absent_children : ℕ) (extra_candies : ℕ) :
  total_children = 300 →
  absent_children = 150 →
  extra_candies = 24 →
  (total_children - absent_children) * (total_children / (total_children - absent_children) + extra_candies) = 
    total_children * (48 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l1793_179327


namespace NUMINAMATH_CALUDE_shirt_sales_revenue_function_l1793_179392

/-- The daily net revenue function for shirt sales -/
def daily_net_revenue (x : ℝ) : ℝ :=
  -x^2 + 110*x - 2400

theorem shirt_sales_revenue_function 
  (wholesale_price : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (price_sensitivity : ℝ) 
  (h1 : wholesale_price = 30)
  (h2 : initial_price = 40)
  (h3 : initial_sales = 40)
  (h4 : price_sensitivity = 1)
  (x : ℝ)
  (h5 : x ≥ 40) :
  daily_net_revenue x = (x - wholesale_price) * (initial_sales - (x - initial_price) * price_sensitivity) :=
by
  sorry

#check shirt_sales_revenue_function

end NUMINAMATH_CALUDE_shirt_sales_revenue_function_l1793_179392


namespace NUMINAMATH_CALUDE_problem_statement_l1793_179320

theorem problem_statement :
  (∀ k : ℕ, (∀ a b : ℕ+, ab + (a + 1) * (b + 1) ≠ 2^k) → Nat.Prime (k + 1)) ∧
  (∃ k : ℕ, Nat.Prime (k + 1) ∧ ∃ a b : ℕ+, ab + (a + 1) * (b + 1) = 2^k) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1793_179320


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1793_179391

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity
  (m n : Line) (α β : Plane)
  (diff_lines : m ≠ n)
  (diff_planes : α ≠ β)
  (m_parallel_n : parallel m n)
  (n_perp_β : perpendicular n β)
  (m_subset_α : subset m α) :
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1793_179391


namespace NUMINAMATH_CALUDE_solution_characterization_l1793_179355

def is_solution (x y z w : ℝ) : Prop :=
  x + y + z + w = 10 ∧
  x^2 + y^2 + z^2 + w^2 = 30 ∧
  x^3 + y^3 + z^3 + w^3 = 100 ∧
  x * y * z * w = 24

def is_permutation_of_1234 (x y z w : ℝ) : Prop :=
  ({x, y, z, w} : Set ℝ) = {1, 2, 3, 4}

theorem solution_characterization :
  ∀ x y z w : ℝ, is_solution x y z w ↔ is_permutation_of_1234 x y z w :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l1793_179355


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1793_179317

open Set

def A : Set ℝ := {x | |x - 1| ≥ 2}
def B : Set ℕ := {x | x < 4}

theorem complement_A_intersect_B :
  (𝒰 \ A) ∩ (coe '' B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1793_179317


namespace NUMINAMATH_CALUDE_integer_part_of_sum_of_roots_l1793_179336

theorem integer_part_of_sum_of_roots (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x*y + y*z + z*x = 1) : 
  ⌊Real.sqrt (3*x*y + 1) + Real.sqrt (3*y*z + 1) + Real.sqrt (3*z*x + 1)⌋ = 4 :=
sorry

end NUMINAMATH_CALUDE_integer_part_of_sum_of_roots_l1793_179336


namespace NUMINAMATH_CALUDE_tyler_purchase_theorem_l1793_179325

def remaining_money (initial_amount scissors_cost eraser_cost scissors_quantity eraser_quantity : ℕ) : ℕ :=
  initial_amount - (scissors_cost * scissors_quantity + eraser_cost * eraser_quantity)

theorem tyler_purchase_theorem (initial_amount scissors_cost eraser_cost scissors_quantity eraser_quantity : ℕ) :
  initial_amount = 100 ∧ 
  scissors_cost = 5 ∧ 
  eraser_cost = 4 ∧ 
  scissors_quantity = 8 ∧ 
  eraser_quantity = 10 → 
  remaining_money initial_amount scissors_cost eraser_cost scissors_quantity eraser_quantity = 20 := by
  sorry

end NUMINAMATH_CALUDE_tyler_purchase_theorem_l1793_179325


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l1793_179360

theorem complex_magnitude_theorem : 
  let i : ℂ := Complex.I
  let T : ℂ := 3 * ((1 + i)^15 - (1 - i)^15)
  Complex.abs T = 768 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l1793_179360


namespace NUMINAMATH_CALUDE_digit_150_of_one_thirteenth_l1793_179389

def decimal_representation (n : ℕ) : ℚ → ℕ := sorry

theorem digit_150_of_one_thirteenth : decimal_representation 150 (1/13) = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_one_thirteenth_l1793_179389


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_1992_l1793_179324

theorem last_three_digits_of_7_to_1992 : ∃ n : ℕ, 7^1992 ≡ 201 + 1000 * n [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_1992_l1793_179324


namespace NUMINAMATH_CALUDE_phase_shift_of_sine_l1793_179399

theorem phase_shift_of_sine (φ : Real) : 
  (0 ≤ φ ∧ φ ≤ 2 * Real.pi) →
  (∀ x, Real.sin (x + φ) = Real.sin (x - Real.pi / 6)) →
  φ = 11 * Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_phase_shift_of_sine_l1793_179399


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_greater_than_three_l1793_179335

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x ≥ 0, P x) ↔ (∀ x ≥ 0, ¬ P x) := by sorry

theorem negation_of_square_greater_than_three :
  (¬ ∃ x ≥ 0, x^2 > 3) ↔ (∀ x ≥ 0, x^2 ≤ 3) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_greater_than_three_l1793_179335


namespace NUMINAMATH_CALUDE_total_distance_is_15_l1793_179300

def morning_ride : ℕ := 2

def evening_ride (m : ℕ) : ℕ := 5 * m

def third_ride (m : ℕ) : ℕ := 2 * m - 1

def total_distance (m : ℕ) : ℕ := m + evening_ride m + third_ride m

theorem total_distance_is_15 : total_distance morning_ride = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_15_l1793_179300


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1793_179339

theorem min_value_of_expression (a b : ℤ) (h : a > b) :
  (((a^2 + b^2) / (a^2 - b^2)) + ((a^2 - b^2) / (a^2 + b^2)) : ℚ) ≥ 2 ∧
  ∃ (a' b' : ℤ), a' > b' ∧ (((a'^2 + b'^2) / (a'^2 - b'^2)) + ((a'^2 - b'^2) / (a'^2 + b'^2)) : ℚ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1793_179339


namespace NUMINAMATH_CALUDE_problem_solution_l1793_179386

theorem problem_solution (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 3)  -- absolute value of m is 3
  : (a + b) / 2023 - 4 * c * d + m^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1793_179386


namespace NUMINAMATH_CALUDE_three_hundred_percent_of_forty_l1793_179333

-- Define 300 percent as 3 in decimal form
def three_hundred_percent : ℝ := 3

-- Define the operation of taking a percentage of a number
def percentage_of (percent : ℝ) (number : ℝ) : ℝ := percent * number

-- Theorem statement
theorem three_hundred_percent_of_forty :
  percentage_of three_hundred_percent 40 = 120 := by
  sorry

end NUMINAMATH_CALUDE_three_hundred_percent_of_forty_l1793_179333


namespace NUMINAMATH_CALUDE_sphere_properties_l1793_179364

/-- Proves surface area and volume of a sphere with diameter 10 inches -/
theorem sphere_properties :
  let d : ℝ := 10  -- diameter
  let r : ℝ := d / 2  -- radius
  ∀ (S V : ℝ),  -- surface area and volume
  S = 4 * Real.pi * r^2 →
  V = (4/3) * Real.pi * r^3 →
  S = 100 * Real.pi ∧ V = (500/3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_properties_l1793_179364


namespace NUMINAMATH_CALUDE_perfect_cube_units_digits_l1793_179369

theorem perfect_cube_units_digits : 
  ∃ (S : Finset ℕ), (∀ n : ℕ, ∃ k : ℕ, n ^ 3 % 10 ∈ S) ∧ S.card = 10 :=
by sorry

end NUMINAMATH_CALUDE_perfect_cube_units_digits_l1793_179369


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1793_179375

/-- A rectangle with integer dimensions and perimeter 30 has a maximum area of 56 -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l + w = 15 →
  ∀ a b : ℕ,
  a + b = 15 →
  l * w ≤ 56 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1793_179375


namespace NUMINAMATH_CALUDE_triangle_isosceles_l1793_179352

-- Define a structure for a triangle
structure Triangle where
  p : ℝ
  q : ℝ
  r : ℝ
  p_pos : 0 < p
  q_pos : 0 < q
  r_pos : 0 < r

-- Define the condition for triangle existence
def triangleExists (t : Triangle) (n : ℕ) : Prop :=
  t.p^n + t.q^n > t.r^n ∧ t.q^n + t.r^n > t.p^n ∧ t.r^n + t.p^n > t.q^n

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.p = t.q ∨ t.q = t.r ∨ t.r = t.p

-- The main theorem
theorem triangle_isosceles (t : Triangle) 
  (h : ∀ n : ℕ, triangleExists t n) : isIsosceles t := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l1793_179352


namespace NUMINAMATH_CALUDE_exponential_decreasing_base_less_than_one_l1793_179322

theorem exponential_decreasing_base_less_than_one
  (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  (0.3 : ℝ) ^ m < (0.3 : ℝ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_exponential_decreasing_base_less_than_one_l1793_179322


namespace NUMINAMATH_CALUDE_no_integral_points_on_tangent_line_l1793_179377

theorem no_integral_points_on_tangent_line (k m n : ℤ) : 
  ∀ x y : ℤ, (m^3 - m) * x + (n^3 - n) * y ≠ (3*k + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integral_points_on_tangent_line_l1793_179377


namespace NUMINAMATH_CALUDE_sony_games_to_give_away_l1793_179334

theorem sony_games_to_give_away (current_sony_games : ℕ) (target_sony_games : ℕ) :
  current_sony_games = 132 → target_sony_games = 31 →
  current_sony_games - target_sony_games = 101 :=
by
  sorry


end NUMINAMATH_CALUDE_sony_games_to_give_away_l1793_179334


namespace NUMINAMATH_CALUDE_trailing_zeros_500_50_l1793_179306

theorem trailing_zeros_500_50 : ∃ n : ℕ, 500^50 = n * 10^100 ∧ n % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_500_50_l1793_179306


namespace NUMINAMATH_CALUDE_difference_in_sums_l1793_179304

def star_list : List Nat := List.range 50 |>.map (· + 1)

def replace_three_with_two (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list : List Nat :=
  star_list.map replace_three_with_two

theorem difference_in_sums : 
  star_list.sum - emilio_list.sum = 105 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_sums_l1793_179304


namespace NUMINAMATH_CALUDE_video_game_lives_l1793_179361

theorem video_game_lives (initial_lives lost_lives gained_lives : ℕ) 
  (h1 : initial_lives = 47)
  (h2 : lost_lives = 23)
  (h3 : gained_lives = 46) :
  initial_lives - lost_lives + gained_lives = 70 := by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l1793_179361


namespace NUMINAMATH_CALUDE_fred_car_wash_earnings_l1793_179308

/-- The amount Fred earned by washing cars -/
def fred_earnings (initial_amount final_amount : ℕ) : ℕ :=
  final_amount - initial_amount

/-- Proof that Fred earned $4 by washing cars -/
theorem fred_car_wash_earnings : 
  fred_earnings 111 115 = 4 := by sorry

end NUMINAMATH_CALUDE_fred_car_wash_earnings_l1793_179308


namespace NUMINAMATH_CALUDE_razorback_tshirt_sales_l1793_179328

/-- The number of t-shirts sold by the Razorback t-shirt shop during a game -/
def num_tshirts_sold (original_price discount total_revenue : ℕ) : ℕ :=
  total_revenue / (original_price - discount)

/-- Theorem stating that 130 t-shirts were sold given the problem conditions -/
theorem razorback_tshirt_sales : num_tshirts_sold 51 8 5590 = 130 := by
  sorry

end NUMINAMATH_CALUDE_razorback_tshirt_sales_l1793_179328


namespace NUMINAMATH_CALUDE_total_distance_mercedes_davonte_l1793_179326

/-- 
Given:
- Jonathan ran 7.5 kilometers
- Mercedes ran twice the distance of Jonathan
- Davonte ran 2 kilometers farther than Mercedes

Prove that the total distance run by Mercedes and Davonte is 32 kilometers
-/
theorem total_distance_mercedes_davonte (jonathan_distance : ℝ) 
  (h1 : jonathan_distance = 7.5)
  (mercedes_distance : ℝ)
  (h2 : mercedes_distance = 2 * jonathan_distance)
  (davonte_distance : ℝ)
  (h3 : davonte_distance = mercedes_distance + 2) :
  mercedes_distance + davonte_distance = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_mercedes_davonte_l1793_179326


namespace NUMINAMATH_CALUDE_square_coloring_l1793_179359

/-- The number of triangles in the square -/
def n : ℕ := 18

/-- The number of triangles to be colored -/
def k : ℕ := 6

/-- Binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem square_coloring :
  binomial n k = 18564 := by
  sorry

end NUMINAMATH_CALUDE_square_coloring_l1793_179359


namespace NUMINAMATH_CALUDE_blue_preference_percentage_l1793_179321

def total_responses : ℕ := 70 + 80 + 50 + 70 + 30

def blue_responses : ℕ := 80

def percentage_blue : ℚ := blue_responses / total_responses * 100

theorem blue_preference_percentage :
  percentage_blue = 80 / 300 * 100 :=
by sorry

end NUMINAMATH_CALUDE_blue_preference_percentage_l1793_179321


namespace NUMINAMATH_CALUDE_paint_fraction_in_15_minutes_l1793_179315

/-- The fraction of a wall that can be painted by two people working together,
    given their individual rates and a specific time. -/
def fractionPainted (rate1 rate2 time : ℚ) : ℚ :=
  (rate1 + rate2) * time

theorem paint_fraction_in_15_minutes :
  let heidi_rate : ℚ := 1 / 60
  let zoe_rate : ℚ := 1 / 90
  let time : ℚ := 15
  fractionPainted heidi_rate zoe_rate time = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_paint_fraction_in_15_minutes_l1793_179315


namespace NUMINAMATH_CALUDE_sum_of_twenty_and_ten_l1793_179350

theorem sum_of_twenty_and_ten : 20 + 10 = 30 := by sorry

end NUMINAMATH_CALUDE_sum_of_twenty_and_ten_l1793_179350


namespace NUMINAMATH_CALUDE_automobile_distance_l1793_179301

/-- Proves that an automobile traveling a/4 feet in 2r seconds will travel 25a/r yards in 10 minutes -/
theorem automobile_distance (a r : ℝ) (h : r ≠ 0) : 
  let rate_feet_per_second := a / (4 * 2 * r)
  let rate_yards_per_second := rate_feet_per_second / 3
  let time_seconds := 10 * 60
  rate_yards_per_second * time_seconds = 25 * a / r := by sorry

end NUMINAMATH_CALUDE_automobile_distance_l1793_179301


namespace NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_l1793_179340

theorem quadratic_equation_rational_solutions :
  ∃! (c₁ c₂ : ℕ+), 
    (∃ (x : ℚ), 7 * x^2 + 13 * x + c₁.val = 0) ∧
    (∃ (x : ℚ), 7 * x^2 + 13 * x + c₂.val = 0) ∧
    c₁ = c₂ ∧ c₁ = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_l1793_179340


namespace NUMINAMATH_CALUDE_g_of_5_l1793_179330

/-- The function g satisfies the given functional equation for all real x -/
axiom functional_equation (g : ℝ → ℝ) :
  ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 3 * x + 2

/-- The value of g(5) is -20.01 -/
theorem g_of_5 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 3 * x + 2) :
  g 5 = -20.01 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l1793_179330


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l1793_179329

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular α β) 
  (h2 : line_perpendicular m β) 
  (h3 : ¬ line_in_plane m α) : 
  line_parallel m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l1793_179329


namespace NUMINAMATH_CALUDE_books_per_shelf_l1793_179347

theorem books_per_shelf (total_books : ℕ) (total_shelves : ℕ) (h1 : total_books = 14240) (h2 : total_shelves = 1780) :
  total_books / total_shelves = 8 := by
sorry

end NUMINAMATH_CALUDE_books_per_shelf_l1793_179347


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1793_179397

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalStudents : ℕ
  groupSize : ℕ
  numGroups : ℕ
  sampleSize : ℕ
  initialSample : ℕ
  initialGroup : ℕ

/-- Given a systematic sampling scheme, calculate the sample from a specific group -/
def sampleFromGroup (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.initialSample + s.groupSize * (group - s.initialGroup)

theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.totalStudents = 50)
  (h2 : s.groupSize = 5)
  (h3 : s.numGroups = 10)
  (h4 : s.sampleSize = 10)
  (h5 : s.initialSample = 12)
  (h6 : s.initialGroup = 3) :
  sampleFromGroup s 8 = 37 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1793_179397


namespace NUMINAMATH_CALUDE_smallest_prime_above_50_l1793_179312

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem smallest_prime_above_50 :
  ∃ p : ℕ, is_prime p ∧ p > 50 ∧ ∀ q : ℕ, is_prime q ∧ q > 50 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_above_50_l1793_179312
