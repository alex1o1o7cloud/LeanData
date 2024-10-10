import Mathlib

namespace intersection_A_complement_B_l663_66384

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x - 1 ≥ 0}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (univ \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_A_complement_B_l663_66384


namespace polynomial_coefficient_sum_l663_66368

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 2 := by
sorry

end polynomial_coefficient_sum_l663_66368


namespace canoe_kayak_difference_is_six_l663_66378

/-- Represents the rental business scenario --/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_kayak_ratio : ℚ
  total_revenue : ℕ

/-- The specific rental business instance --/
def our_business : RentalBusiness := {
  canoe_price := 9
  kayak_price := 12
  canoe_kayak_ratio := 4/3
  total_revenue := 432
}

/-- Calculates the difference between canoes and kayaks rented --/
def canoe_kayak_difference (rb : RentalBusiness) : ℕ :=
  sorry

/-- Theorem stating that the difference between canoes and kayaks rented is 6 --/
theorem canoe_kayak_difference_is_six :
  canoe_kayak_difference our_business = 6 := by
  sorry

end canoe_kayak_difference_is_six_l663_66378


namespace ice_cream_unsold_l663_66390

theorem ice_cream_unsold (chocolate mango vanilla strawberry : ℕ)
  (h_chocolate : chocolate = 50)
  (h_mango : mango = 54)
  (h_vanilla : vanilla = 80)
  (h_strawberry : strawberry = 40)
  (sold_chocolate : ℚ)
  (sold_mango : ℚ)
  (sold_vanilla : ℚ)
  (sold_strawberry : ℚ)
  (h_sold_chocolate : sold_chocolate = 3 / 5)
  (h_sold_mango : sold_mango = 2 / 3)
  (h_sold_vanilla : sold_vanilla = 3 / 4)
  (h_sold_strawberry : sold_strawberry = 5 / 8) :
  chocolate - Int.floor (sold_chocolate * chocolate) +
  mango - Int.floor (sold_mango * mango) +
  vanilla - Int.floor (sold_vanilla * vanilla) +
  strawberry - Int.floor (sold_strawberry * strawberry) = 73 := by
  sorry

end ice_cream_unsold_l663_66390


namespace penny_nickel_dime_heads_probability_l663_66373

def coin_flip_probability : ℚ :=
  let total_outcomes : ℕ := 2^5
  let successful_outcomes : ℕ := 2^2
  (successful_outcomes : ℚ) / total_outcomes

theorem penny_nickel_dime_heads_probability :
  coin_flip_probability = 1/8 := by
  sorry

end penny_nickel_dime_heads_probability_l663_66373


namespace women_percentage_of_men_l663_66361

theorem women_percentage_of_men (W M : ℝ) (h : M = 2 * W) : W / M * 100 = 50 := by
  sorry

end women_percentage_of_men_l663_66361


namespace trays_for_school_staff_l663_66362

def small_oatmeal_cookies : ℕ := 276
def large_oatmeal_cookies : ℕ := 92
def large_choc_chip_cookies : ℕ := 150
def small_cookies_per_tray : ℕ := 12
def large_cookies_per_tray : ℕ := 6

theorem trays_for_school_staff : 
  (large_choc_chip_cookies + large_cookies_per_tray - 1) / large_cookies_per_tray = 25 := by
  sorry

end trays_for_school_staff_l663_66362


namespace zeros_in_square_of_near_power_of_ten_l663_66371

theorem zeros_in_square_of_near_power_of_ten : 
  (∃ n : ℕ, n = (10^12 - 5)^2 ∧ 
   ∃ m : ℕ, m > 0 ∧ n = m * 10^12 ∧ m % 10 ≠ 0) := by
  sorry

end zeros_in_square_of_near_power_of_ten_l663_66371


namespace equation_two_distinct_roots_l663_66357

theorem equation_two_distinct_roots (a : ℝ) : 
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    ((Real.sqrt (6 * x₁ - x₁^2 - 4) + a - 2) * ((a - 2) * x₁ - 3 * a + 4) = 0) ∧
    ((Real.sqrt (6 * x₂ - x₂^2 - 4) + a - 2) * ((a - 2) * x₂ - 3 * a + 4) = 0)) ↔ 
  (a = 2 - Real.sqrt 5 ∨ a = 0 ∨ a = 1 ∨ (2 - 2 / Real.sqrt 5 < a ∧ a ≤ 2)) :=
sorry

end equation_two_distinct_roots_l663_66357


namespace exactly_three_valid_sets_l663_66315

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)

/-- The sum of a ConsecutiveSet -/
def sum_consecutive_set (s : ConsecutiveSet) : ℕ :=
  (s.length * (2 * s.start + s.length - 1)) / 2

/-- Predicate for a valid set according to our conditions -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 3 ∧ sum_consecutive_set s = 30

/-- The theorem to prove -/
theorem exactly_three_valid_sets :
  ∃! (sets : Finset ConsecutiveSet), 
    sets.card = 3 ∧ 
    (∀ s ∈ sets, is_valid_set s) ∧
    (∀ s, is_valid_set s → s ∈ sets) :=
  sorry

end exactly_three_valid_sets_l663_66315


namespace monotonic_decreasing_interval_l663_66370

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

def monotonic_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

theorem monotonic_decreasing_interval :
  ∀ x, x > 0 → (monotonic_decreasing f 0 1) :=
by sorry

end monotonic_decreasing_interval_l663_66370


namespace complex_division_simplification_l663_66301

theorem complex_division_simplification (z : ℂ) (h : z = 1 - 2 * I) :
  5 * I / z = -2 + I :=
by sorry

end complex_division_simplification_l663_66301


namespace secret_spread_days_l663_66364

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The number of days required for 3280 students to know the secret -/
theorem secret_spread_days : ∃ n : ℕ, secret_spread n = 3280 ∧ n = 7 := by
  sorry

end secret_spread_days_l663_66364


namespace cube_sum_reciprocal_cube_l663_66385

theorem cube_sum_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^3 + 1/r^3 = 2 * Real.sqrt 5 ∨ r^3 + 1/r^3 = -2 * Real.sqrt 5 := by
  sorry

end cube_sum_reciprocal_cube_l663_66385


namespace polynomial_simplification_l663_66358

theorem polynomial_simplification (x : ℝ) :
  (2 * x^5 - 3 * x^4 + x^3 + 5 * x^2 - 2 * x + 8) + 
  (x^4 - 2 * x^3 + 3 * x^2 + 4 * x - 16) = 
  2 * x^5 - 2 * x^4 - x^3 + 8 * x^2 + 2 * x - 8 :=
by sorry

end polynomial_simplification_l663_66358


namespace inequality_proof_l663_66331

theorem inequality_proof (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  x + y + (1 / (x * y)) ≤ (1 / x) + (1 / y) + x * y := by
sorry

end inequality_proof_l663_66331


namespace expression_nonnegative_l663_66359

theorem expression_nonnegative (x : ℝ) : 
  (2*x - 6*x^2 + 9*x^3) / (9 - x^3) ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 3 :=
sorry

end expression_nonnegative_l663_66359


namespace symmetric_points_sum_l663_66335

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem stating that if M(3, a-2) and N(b, a) are symmetric with respect to the origin, then a + b = -2 -/
theorem symmetric_points_sum (a b : ℝ) :
  symmetric_wrt_origin 3 (a - 2) b a → a + b = -2 := by
  sorry

end symmetric_points_sum_l663_66335


namespace abs_inequality_solution_set_l663_66386

theorem abs_inequality_solution_set (x : ℝ) : 
  (|x - 2| < 1) ↔ (1 < x ∧ x < 3) :=
sorry

end abs_inequality_solution_set_l663_66386


namespace line_slope_intercept_product_l663_66398

theorem line_slope_intercept_product (m b : ℚ) : 
  m = 3/4 → b = 5/2 → m * b > 1 := by sorry

end line_slope_intercept_product_l663_66398


namespace sports_league_games_l663_66303

/-- Calculates the total number of games in a sports league season. -/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  (total_teams * (intra_division_games * (teams_per_division - 1) + 
  inter_division_games * teams_per_division)) / 2

/-- Theorem stating the total number of games in the given sports league setup -/
theorem sports_league_games : 
  total_games 16 8 3 1 = 232 := by
  sorry

end sports_league_games_l663_66303


namespace gemma_pizza_change_l663_66369

def pizza_order (num_pizzas : ℕ) (price_per_pizza : ℕ) (tip : ℕ) (payment : ℕ) : ℕ :=
  payment - (num_pizzas * price_per_pizza + tip)

theorem gemma_pizza_change : pizza_order 4 10 5 50 = 5 := by
  sorry

end gemma_pizza_change_l663_66369


namespace problem_statement_l663_66312

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem problem_statement (x : ℝ) 
  (h : deriv f x = 2 * f x) : 
  (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin (2 * x)) = -19/5 := by
  sorry

end problem_statement_l663_66312


namespace area_of_region_l663_66391

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 32 ∧ 
   A = Real.pi * (Real.sqrt ((x + 4)^2 + (y + 5)^2))^2 ∧
   x^2 + y^2 + 8*x + 10*y = -9) := by
sorry

end area_of_region_l663_66391


namespace modulo_thirteen_seven_l663_66366

theorem modulo_thirteen_seven (n : ℕ) : 
  13^7 ≡ n [ZMOD 7] → 0 ≤ n → n < 7 → n = 6 := by
  sorry

end modulo_thirteen_seven_l663_66366


namespace exact_one_solver_probability_l663_66360

/-- The probability that exactly one person solves a problem, given the probabilities
    for two independent solvers. -/
theorem exact_one_solver_probability (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  (p₁ * (1 - p₂) + p₂ * (1 - p₁)) = 
  (p₁ + p₂ - 2 * p₁ * p₂) := by
  sorry

#check exact_one_solver_probability

end exact_one_solver_probability_l663_66360


namespace parabola_intersection_l663_66320

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem parabola_intersection :
  (f (-1) = 0) →  -- The parabola intersects the x-axis at (-1, 0)
  (∃ x : ℝ, x ≠ -1 ∧ f x = 0 ∧ x = 3) :=  -- There exists another intersection point at (3, 0)
by
  sorry

end parabola_intersection_l663_66320


namespace tan_alpha_value_l663_66348

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end tan_alpha_value_l663_66348


namespace linear_function_range_l663_66300

/-- A linear function defined on a closed interval -/
def LinearFunction (a b : ℝ) (x : ℝ) : ℝ := a * x + b

/-- The domain of the function -/
def Domain : Set ℝ := { x : ℝ | 1/4 ≤ x ∧ x ≤ 3/4 }

theorem linear_function_range (a b : ℝ) (h : a > 0) :
  Set.range (fun x => LinearFunction a b x) = Set.Icc (a/4 + b) (3*a/4 + b) := by
  sorry

end linear_function_range_l663_66300


namespace radius_is_seven_l663_66309

/-- Represents a circle with a point P outside it and a secant PQR -/
structure CircleWithSecant where
  /-- Distance from P to the center of the circle -/
  s : ℝ
  /-- Length of external segment PQ -/
  pq : ℝ
  /-- Length of chord QR -/
  qr : ℝ

/-- The radius of the circle given the secant configuration -/
def radius (c : CircleWithSecant) : ℝ :=
  sorry

/-- Theorem stating that the radius is 7 given the specific measurements -/
theorem radius_is_seven (c : CircleWithSecant) 
  (h1 : c.s = 17) 
  (h2 : c.pq = 12) 
  (h3 : c.qr = 8) : 
  radius c = 7 := by
  sorry

end radius_is_seven_l663_66309


namespace terminal_zeros_25_times_240_l663_66372

/-- The number of terminal zeros in a positive integer -/
def terminalZeros (n : ℕ) : ℕ := sorry

/-- The prime factorization of 25 -/
def primeFactor25 : ℕ → ℕ
| 5 => 2
| _ => 0

/-- The prime factorization of 240 -/
def primeFactor240 : ℕ → ℕ
| 2 => 4
| 3 => 1
| 5 => 1
| _ => 0

theorem terminal_zeros_25_times_240 : 
  terminalZeros (25 * 240) = 3 := by sorry

end terminal_zeros_25_times_240_l663_66372


namespace number_equation_l663_66313

theorem number_equation (x : ℝ) : 100 - x = x + 40 ↔ x = 30 := by
  sorry

end number_equation_l663_66313


namespace memory_card_picture_size_l663_66343

theorem memory_card_picture_size 
  (total_pictures_a : ℕ) 
  (size_a : ℕ) 
  (total_pictures_b : ℕ) 
  (h1 : total_pictures_a = 3000)
  (h2 : size_a = 8)
  (h3 : total_pictures_b = 4000) :
  (total_pictures_a * size_a) / total_pictures_b = 6 :=
by sorry

end memory_card_picture_size_l663_66343


namespace power_two_minus_one_div_seven_l663_66383

theorem power_two_minus_one_div_seven (n : ℕ) :
  (7 ∣ (2^n - 1)) ↔ (3 ∣ n) := by
sorry

end power_two_minus_one_div_seven_l663_66383


namespace angle_sum_360_l663_66355

theorem angle_sum_360 (k : ℝ) : k + 90 = 360 → k = 270 := by
  sorry

end angle_sum_360_l663_66355


namespace rectangle_midpoint_distances_theorem_l663_66336

def rectangle_midpoint_distances : ℝ := by
  -- Define the vertices of the rectangle
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ × ℝ := (3, 4)
  let D : ℝ × ℝ := (0, 4)

  -- Define the midpoints of each side
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let O : ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let P : ℝ × ℝ := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

  -- Calculate distances from A to each midpoint
  let d_AM := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  let d_AN := Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2)
  let d_AO := Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2)
  let d_AP := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)

  -- Sum of distances
  let total_distance := d_AM + d_AN + d_AO + d_AP

  -- Prove that the total distance equals the expected value
  sorry

theorem rectangle_midpoint_distances_theorem :
  rectangle_midpoint_distances = (3 * Real.sqrt 2) / 2 + Real.sqrt 13 + (Real.sqrt 73) / 2 + 2 := by
  sorry

end rectangle_midpoint_distances_theorem_l663_66336


namespace eight_digit_number_theorem_l663_66376

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def last_digit (n : ℕ) : ℕ := n % 10

def move_last_to_first (n : ℕ) : ℕ :=
  (last_digit n) * 10^7 + n / 10

theorem eight_digit_number_theorem (B : ℕ) 
  (h1 : B > 7777777)
  (h2 : is_coprime B 36)
  (h3 : ∃ A : ℕ, A = move_last_to_first B) :
  ∃ A_min A_max : ℕ, 
    (A_min = move_last_to_first B ∧ A_min ≥ 17777779) ∧
    (A_max = move_last_to_first B ∧ A_max ≤ 99999998) ∧
    (∀ A : ℕ, A = move_last_to_first B → A_min ≤ A ∧ A ≤ A_max) :=
by sorry

end eight_digit_number_theorem_l663_66376


namespace teacher_student_meeting_l663_66344

/-- Represents the teacher-student meeting scenario -/
structure MeetingScenario where
  total_participants : ℕ
  first_teacher_students : ℕ
  teachers : ℕ
  students : ℕ

/-- Checks if the given scenario satisfies the meeting conditions -/
def is_valid_scenario (m : MeetingScenario) : Prop :=
  m.total_participants = m.teachers + m.students ∧
  m.first_teacher_students = m.students - m.teachers + 1 ∧
  m.teachers > 0 ∧
  m.students > 0

/-- The theorem stating the correct number of teachers and students -/
theorem teacher_student_meeting :
  ∃ (m : MeetingScenario), is_valid_scenario m ∧ m.teachers = 8 ∧ m.students = 23 :=
sorry

end teacher_student_meeting_l663_66344


namespace distance_at_4_seconds_l663_66325

/-- The distance traveled by an object given time t -/
def distance (t : ℝ) : ℝ := 5 * t^2 + 2 * t

theorem distance_at_4_seconds :
  distance 4 = 88 := by
  sorry

end distance_at_4_seconds_l663_66325


namespace intersection_point_D_l663_66304

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 1

/-- The normal line equation at point (2, 4) -/
def normal_line (x y : ℝ) : Prop := y = -1/4 * x + 9/2

theorem intersection_point_D :
  let C : ℝ × ℝ := (2, 4)
  let D : ℝ × ℝ := (-2, 5)
  parabola C.1 C.2 →
  parabola D.1 D.2 ∧
  normal_line D.1 D.2 :=
by sorry

end intersection_point_D_l663_66304


namespace intersection_of_M_and_N_l663_66392

def M : Set Int := {-1, 3, 5}
def N : Set Int := {-1, 0, 1, 2, 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 3} := by
  sorry

end intersection_of_M_and_N_l663_66392


namespace parallel_lines_angle_condition_l663_66305

-- Define the concept of lines and planes
variable (Line Plane : Type)

-- Define the concept of parallel lines
variable (parallel : Line → Line → Prop)

-- Define the concept of a line forming an angle with a plane
variable (angle_with_plane : Line → Plane → ℝ)

-- State the theorem
theorem parallel_lines_angle_condition 
  (a b : Line) (α : Plane) :
  (parallel a b → angle_with_plane a α = angle_with_plane b α) ∧
  ¬(angle_with_plane a α = angle_with_plane b α → parallel a b) :=
sorry

end parallel_lines_angle_condition_l663_66305


namespace binomial_expansion_terms_l663_66316

theorem binomial_expansion_terms (x a : ℝ) (n : ℕ) :
  (Nat.choose n 2 : ℝ) * x^(n - 2) * a^2 = 84 ∧
  (Nat.choose n 3 : ℝ) * x^(n - 3) * a^3 = 280 ∧
  (Nat.choose n 4 : ℝ) * x^(n - 4) * a^4 = 560 →
  n = 7 := by
  sorry

end binomial_expansion_terms_l663_66316


namespace linear_equation_properties_l663_66389

/-- Given a linear equation x + 2y = -6, this theorem proves:
    1. y can be expressed as y = -3 - x/2
    2. y is a negative number greater than -2 if and only if -6 < x < -2
-/
theorem linear_equation_properties (x y : ℝ) (h : x + 2 * y = -6) :
  (y = -3 - x / 2) ∧
  (y < 0 ∧ y > -2 ↔ -6 < x ∧ x < -2) := by
  sorry

end linear_equation_properties_l663_66389


namespace soup_offer_ratio_l663_66307

/-- Represents the soup can offer --/
structure SoupOffer where
  total_cans : ℕ
  normal_price : ℚ
  total_paid : ℚ

/-- Calculates the buy to get ratio for a soup offer --/
def buyToGetRatio (offer : SoupOffer) : ℚ × ℚ :=
  let paid_cans := offer.total_paid / offer.normal_price
  let free_cans := offer.total_cans - paid_cans
  (paid_cans, free_cans)

/-- Theorem stating that the given offer results in a 1:1 ratio --/
theorem soup_offer_ratio (offer : SoupOffer) 
  (h1 : offer.total_cans = 30)
  (h2 : offer.normal_price = 0.6)
  (h3 : offer.total_paid = 9) :
  buyToGetRatio offer = (15, 15) := by
  sorry

#eval buyToGetRatio ⟨30, 0.6, 9⟩

end soup_offer_ratio_l663_66307


namespace tower_comparison_l663_66350

def tower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (tower base n)

theorem tower_comparison (n : ℕ) : ∃ m : ℕ, ∀ k ≥ m,
  (tower 3 k > tower 2 (k + 1)) ∧ (tower 4 k > tower 3 k) := by
  sorry

#check tower_comparison

end tower_comparison_l663_66350


namespace wall_building_time_l663_66381

theorem wall_building_time 
  (men_days_constant : ℕ → ℕ → ℕ) 
  (h1 : men_days_constant 10 6 = men_days_constant 15 4) 
  (h2 : ∀ m d, men_days_constant m d = m * d) :
  (10 : ℚ) * 6 / 15 = 4 :=
sorry

end wall_building_time_l663_66381


namespace base_conversion_l663_66332

/-- Given that the decimal number 26 converted to base r is 32, prove that r = 8 -/
theorem base_conversion (r : ℕ) (h : r > 1) : 
  (26 : ℕ).digits r = [3, 2] → r = 8 := by
  sorry

end base_conversion_l663_66332


namespace work_completion_time_l663_66321

theorem work_completion_time (a b c : ℝ) : 
  (b = 12) →  -- B can do the work in 12 days
  (1/a + 1/b = 1/4) →  -- A and B working together finish the work in 4 days
  (a = 6) -- A can do the work alone in 6 days
:= by sorry

end work_completion_time_l663_66321


namespace common_solution_y_value_l663_66396

theorem common_solution_y_value : ∃! y : ℝ, ∃ x : ℝ, 
  (x^2 + y^2 - 4 = 0) ∧ (x^2 - 4*y + 8 = 0) :=
by
  -- Proof goes here
  sorry

#check common_solution_y_value

end common_solution_y_value_l663_66396


namespace inequality_proof_l663_66379

theorem inequality_proof (x : ℝ) (h : x > 2) : x + 1 / (x - 2) ≥ 4 := by
  sorry

end inequality_proof_l663_66379


namespace one_thirds_in_nine_fifths_l663_66394

theorem one_thirds_in_nine_fifths : (9 : ℚ) / 5 / (1 : ℚ) / 3 = 27 / 5 := by
  sorry

end one_thirds_in_nine_fifths_l663_66394


namespace odd_sum_of_squares_implies_odd_product_l663_66353

theorem odd_sum_of_squares_implies_odd_product (n m : ℤ) 
  (h : Odd (n^2 + m^2)) : Odd (n * m) := by
  sorry

end odd_sum_of_squares_implies_odd_product_l663_66353


namespace common_solution_y_value_l663_66365

theorem common_solution_y_value (x y : ℝ) : 
  x^2 + y^2 - 4 = 0 ∧ x^2 - y + 2 = 0 → y = 2 :=
by sorry

end common_solution_y_value_l663_66365


namespace intersection_M_N_l663_66319

-- Define the universal set U
def U : Set Int := {-1, 0, 1, 2, 3, 4}

-- Define the complement of M with respect to U
def M_complement : Set Int := {-1, 1}

-- Define set N
def N : Set Int := {0, 1, 2, 3}

-- Define set M based on its complement
def M : Set Int := U \ M_complement

-- Theorem statement
theorem intersection_M_N : M ∩ N = {0, 2, 3} := by
  sorry

end intersection_M_N_l663_66319


namespace equation_one_solution_l663_66338

theorem equation_one_solution (k : ℝ) : 
  (∃! x : ℝ, (3*x + 6)*(x - 4) = -40 + k*x) ↔ 
  (k = -6 + 8*Real.sqrt 3 ∨ k = -6 - 8*Real.sqrt 3) := by
sorry

end equation_one_solution_l663_66338


namespace intersection_distance_product_l663_66310

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y^2 = 4*x

def C₂ (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 2 * Real.sqrt 3 = 0

-- Define point P
def P : ℝ × ℝ := (2, 0)

-- Define the theorem
theorem intersection_distance_product :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧
    C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧
    A ≠ B ∧
    (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) *
     Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 32/3) :=
by sorry

end intersection_distance_product_l663_66310


namespace puppy_adoption_cost_is_96_l663_66393

/-- Calculates the total cost of adopting a puppy and buying necessary supplies with a discount --/
def puppy_adoption_cost (adoption_fee : ℝ) (dog_food : ℝ) (treats_price : ℝ) (treats_quantity : ℕ)
  (toys : ℝ) (crate_bed_price : ℝ) (collar_leash : ℝ) (discount_rate : ℝ) : ℝ :=
  let supplies_cost := dog_food + treats_price * treats_quantity + toys + 2 * crate_bed_price + collar_leash
  let discounted_supplies := supplies_cost * (1 - discount_rate)
  adoption_fee + discounted_supplies

/-- Theorem stating that the total cost of adopting a puppy and buying supplies is $96.00 --/
theorem puppy_adoption_cost_is_96 :
  puppy_adoption_cost 20 20 2.5 2 15 20 15 0.2 = 96 := by
  sorry


end puppy_adoption_cost_is_96_l663_66393


namespace range_of_m_l663_66375

theorem range_of_m : 
  (∀ x : ℝ, |2009 * x + 1| ≥ |m - 1| - 2) ↔ m ∈ Set.Icc (-1 : ℝ) 3 := by
  sorry

end range_of_m_l663_66375


namespace set_of_values_for_a_l663_66387

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + a = 0}

theorem set_of_values_for_a (a : ℝ) : 
  (∀ B : Set ℝ, B ⊆ A a → B = ∅ ∨ B = A a) → 
  (a > 1 ∨ a < -1) :=
sorry

end set_of_values_for_a_l663_66387


namespace prop_variations_l663_66380

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x = 2 → x^2 - 5*x + 6 = 0

-- Define the converse
def converse (x : ℝ) : Prop := x^2 - 5*x + 6 = 0 → x = 2

-- Define the inverse
def inverse (x : ℝ) : Prop := x ≠ 2 → x^2 - 5*x + 6 ≠ 0

-- Define the contrapositive
def contrapositive (x : ℝ) : Prop := x^2 - 5*x + 6 ≠ 0 → x ≠ 2

-- Theorem stating the truth values of converse, inverse, and contrapositive
theorem prop_variations :
  (∃ x : ℝ, ¬(converse x)) ∧
  (∃ x : ℝ, ¬(inverse x)) ∧
  (∀ x : ℝ, contrapositive x) :=
sorry

end prop_variations_l663_66380


namespace sin_value_given_sum_and_tan_condition_l663_66374

theorem sin_value_given_sum_and_tan_condition (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = 7/5)
  (h2 : Real.tan θ < 1) : 
  Real.sin θ = 3/5 := by
  sorry

end sin_value_given_sum_and_tan_condition_l663_66374


namespace basketball_tournament_wins_losses_l663_66334

theorem basketball_tournament_wins_losses 
  (total_games : ℕ) 
  (points_per_win : ℕ) 
  (points_per_loss : ℕ) 
  (total_points : ℕ) 
  (h1 : total_games = 15) 
  (h2 : points_per_win = 3) 
  (h3 : points_per_loss = 1) 
  (h4 : total_points = 41) : 
  ∃ (wins losses : ℕ), 
    wins + losses = total_games ∧ 
    wins * points_per_win + losses * points_per_loss = total_points ∧ 
    wins = 13 ∧ 
    losses = 2 := by
  sorry

end basketball_tournament_wins_losses_l663_66334


namespace contrapositive_equivalence_l663_66327

theorem contrapositive_equivalence :
  (∀ a : ℝ, a > 0 → a > 1) ↔ (∀ a : ℝ, a ≤ 1 → a ≤ 0) :=
by sorry

end contrapositive_equivalence_l663_66327


namespace polynomial_identity_l663_66349

theorem polynomial_identity (p : ℝ → ℝ) : 
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) →
  p 3 = 10 →
  p = fun x => x^2 + 1 :=
by sorry

end polynomial_identity_l663_66349


namespace multiply_three_a_two_ab_l663_66395

theorem multiply_three_a_two_ab (a b : ℝ) : 3 * a * (2 * a * b) = 6 * a^2 * b := by
  sorry

end multiply_three_a_two_ab_l663_66395


namespace shop_ratio_l663_66308

/-- Given a shop with pencils, pens, and exercise books in the ratio 14 : 4 : 3,
    and 140 pencils, the ratio of exercise books to pens is 3 : 4. -/
theorem shop_ratio (pencils pens books : ℕ) : 
  pencils = 140 →
  pencils / 14 = pens / 4 →
  pencils / 14 = books / 3 →
  books / pens = 3 / 4 := by
sorry

end shop_ratio_l663_66308


namespace swap_numbers_l663_66317

theorem swap_numbers (a b : ℕ) : 
  let c := b
  let b' := a
  let a' := c
  (a' = b ∧ b' = a) :=
by
  sorry

end swap_numbers_l663_66317


namespace factorial_program_components_l663_66351

/-- A structure representing a simple programming language --/
structure SimpleProgram where
  input : String
  loop_start : String
  loop_end : String

/-- Definition of a program that calculates factorial --/
def factorial_program (p : SimpleProgram) : Prop :=
  p.input = "INPUT" ∧ 
  p.loop_start = "WHILE" ∧ 
  p.loop_end = "WEND"

/-- Theorem stating that a program calculating factorial requires specific components --/
theorem factorial_program_components :
  ∃ (p : SimpleProgram), factorial_program p :=
sorry

end factorial_program_components_l663_66351


namespace triangle_side_length_l663_66314

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  B = 2 * A ∧
  a = 1 ∧ 
  b = Real.sqrt 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C ∧
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  c = 2 := by
sorry

end triangle_side_length_l663_66314


namespace gum_ratio_proof_l663_66330

def gum_ratio (total_gum : ℕ) (shane_chewed : ℕ) (shane_left : ℕ) : ℚ :=
  let shane_total := shane_chewed + shane_left
  let rick_total := shane_total * 2
  rick_total / total_gum

theorem gum_ratio_proof :
  gum_ratio 100 11 14 = 1/2 := by
  sorry

end gum_ratio_proof_l663_66330


namespace not_in_set_A_l663_66318

-- Define the set A
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 - 5}

-- Theorem statement
theorem not_in_set_A :
  (1, -5) ∉ A ∧ (2, 1) ∈ A ∧ (3, 4) ∈ A ∧ (4, 7) ∈ A := by
  sorry

end not_in_set_A_l663_66318


namespace patricks_class_size_l663_66347

theorem patricks_class_size :
  ∃! b : ℕ,
    100 < b ∧ b < 200 ∧
    ∃ k : ℕ, b = 4 * k - 2 ∧
    ∃ l : ℕ, b = 6 * l - 3 ∧
    ∃ m : ℕ, b = 7 * m - 4 :=
by
  sorry

end patricks_class_size_l663_66347


namespace power_sum_equality_l663_66337

theorem power_sum_equality : 2^567 + 8^5 / 8^3 = 2^567 + 64 := by sorry

end power_sum_equality_l663_66337


namespace rectangle_area_implies_y_l663_66341

/-- Given a rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0),
    if the area of the rectangle is 45 square units and y > 0, then y = 9. -/
theorem rectangle_area_implies_y (y : ℝ) : y > 0 → 5 * y = 45 → y = 9 := by
  sorry

end rectangle_area_implies_y_l663_66341


namespace max_sum_given_product_l663_66354

theorem max_sum_given_product (a b : ℤ) (h : a * b = -72) : 
  (∀ (x y : ℤ), x * y = -72 → x + y ≤ a + b) → a + b = 71 := by
  sorry

end max_sum_given_product_l663_66354


namespace amount_owed_l663_66340

theorem amount_owed (rate_per_car : ℚ) (cars_washed : ℚ) (h1 : rate_per_car = 9/4) (h2 : cars_washed = 10/3) : 
  rate_per_car * cars_washed = 15/2 := by
sorry

end amount_owed_l663_66340


namespace min_value_of_quadratic_l663_66363

theorem min_value_of_quadratic (x : ℝ) :
  let z := 5 * x^2 - 20 * x + 45
  ∀ y : ℝ, z ≥ 25 := by
  sorry

end min_value_of_quadratic_l663_66363


namespace negation_of_implication_l663_66322

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → 2*a > 2*b - 1) ↔ (a ≤ b → 2*a ≤ 2*b - 1) :=
by sorry

end negation_of_implication_l663_66322


namespace expression_value_l663_66399

/-- Proves that the expression (3a+b)^2 - (3a-b)(3a+b) - 5b(a-b) equals 26 when a=1 and b=-2 -/
theorem expression_value (a b : ℤ) (h1 : a = 1) (h2 : b = -2) :
  (3*a + b)^2 - (3*a - b)*(3*a + b) - 5*b*(a - b) = 26 := by
  sorry

end expression_value_l663_66399


namespace stamps_given_l663_66388

theorem stamps_given (x : ℕ) (y : ℕ) : 
  (7 * x : ℕ) / (4 * x) = 7 / 4 →  -- Initial ratio
  ((7 * x - y) : ℕ) / (4 * x + y) = 6 / 5 →  -- Final ratio
  (7 * x - y) = (4 * x + y) + 8 →  -- Final difference
  y = 8 := by sorry

end stamps_given_l663_66388


namespace thirtieth_triangular_number_l663_66324

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end thirtieth_triangular_number_l663_66324


namespace fifty_third_number_is_71_l663_66323

def sequenceValue (n : ℕ) : ℕ := 
  let fullSets := (n - 1) / 3
  let remainder := (n - 1) % 3
  1 + 4 * fullSets + remainder + (if remainder = 2 then 1 else 0)

theorem fifty_third_number_is_71 : sequenceValue 53 = 71 := by
  sorry

end fifty_third_number_is_71_l663_66323


namespace part_to_whole_ratio_l663_66339

theorem part_to_whole_ratio (N : ℚ) (P : ℚ) : 
  (1 / 4 : ℚ) * P = 25 →
  (40 / 100 : ℚ) * N = 300 →
  P / ((2 / 5 : ℚ) * N) = (1 / 3 : ℚ) := by
sorry

end part_to_whole_ratio_l663_66339


namespace remainder_divisibility_l663_66326

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 20) → (∃ m : ℤ, N = 13 * m + 7) := by
  sorry

end remainder_divisibility_l663_66326


namespace infinite_geometric_series_first_term_l663_66302

/-- 
For an infinite geometric series with common ratio r and sum S, 
the first term a is given by the formula: a = S * (1 - r)
-/
def first_term_infinite_geometric_series (r : ℚ) (S : ℚ) : ℚ := S * (1 - r)

theorem infinite_geometric_series_first_term 
  (r : ℚ) (S : ℚ) (h1 : r = -1/3) (h2 : S = 18) : 
  first_term_infinite_geometric_series r S = 24 := by
sorry

end infinite_geometric_series_first_term_l663_66302


namespace minimum_m_value_l663_66352

theorem minimum_m_value (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : ∀ a b c, a > b ∧ b > c → (1 / (a - b) + m / (b - c) ≥ 9 / (a - c))) : 
  m ≥ 4 := by
  sorry

end minimum_m_value_l663_66352


namespace blue_lipstick_count_l663_66367

theorem blue_lipstick_count (total_students : ℕ) 
  (h_total : total_students = 300)
  (h_lipstick : ∃ lipstick_wearers : ℕ, lipstick_wearers = total_students / 2)
  (h_red : ∃ red_wearers : ℕ, red_wearers = lipstick_wearers / 4)
  (h_pink : ∃ pink_wearers : ℕ, pink_wearers = lipstick_wearers / 3)
  (h_purple : ∃ purple_wearers : ℕ, purple_wearers = lipstick_wearers / 6)
  (h_blue : ∃ blue_wearers : ℕ, blue_wearers = lipstick_wearers - (red_wearers + pink_wearers + purple_wearers)) :
  blue_wearers = 37 := by
sorry

end blue_lipstick_count_l663_66367


namespace kite_area_l663_66333

/-- The area of a kite composed of two identical triangles -/
theorem kite_area (base height : ℝ) (h1 : base = 14) (h2 : height = 6) :
  2 * (1/2 * base * height) = 84 := by
  sorry

end kite_area_l663_66333


namespace peter_walking_time_l663_66382

/-- The time required to walk a given distance at a given pace -/
def timeToWalk (distance : ℝ) (pace : ℝ) : ℝ := distance * pace

theorem peter_walking_time :
  let totalDistance : ℝ := 2.5
  let walkingPace : ℝ := 20
  let distanceWalked : ℝ := 1
  let remainingDistance : ℝ := totalDistance - distanceWalked
  timeToWalk remainingDistance walkingPace = 30 := by
sorry

end peter_walking_time_l663_66382


namespace binomial_coefficient_prime_power_bound_l663_66342

theorem binomial_coefficient_prime_power_bound 
  (p : Nat) (n k α : Nat) (h_prime : Prime p) 
  (h_divides : p ^ α ∣ Nat.choose n k) : 
  p ^ α ≤ n :=
sorry

end binomial_coefficient_prime_power_bound_l663_66342


namespace prime_square_minus_one_divisible_by_twelve_l663_66346

theorem prime_square_minus_one_divisible_by_twelve (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) : 
  12 ∣ (p^2 - 1) := by
  sorry

end prime_square_minus_one_divisible_by_twelve_l663_66346


namespace total_subjects_l663_66329

theorem total_subjects (average_all : ℝ) (average_five : ℝ) (last_subject : ℝ) 
  (h1 : average_all = 77)
  (h2 : average_five = 74)
  (h3 : last_subject = 92) :
  ∃ n : ℕ, n = 6 ∧ 
    n * average_all = (n - 1) * average_five + last_subject :=
by sorry

end total_subjects_l663_66329


namespace crow_nest_ditch_distance_crow_problem_solution_l663_66356

/-- The distance between a crow's nest and a ditch, given the crow's flying pattern and speed. -/
theorem crow_nest_ditch_distance (trips : ℕ) (time : ℝ) (speed : ℝ) : ℝ :=
  let distance_km := speed * time / (2 * trips)
  let distance_m := distance_km * 1000
  200

/-- Proof that the distance between the nest and the ditch is 200 meters. -/
theorem crow_problem_solution :
  crow_nest_ditch_distance 15 1.5 4 = 200 := by
  sorry

end crow_nest_ditch_distance_crow_problem_solution_l663_66356


namespace flooring_problem_l663_66397

theorem flooring_problem (room_length room_width box_area boxes_needed : ℕ) 
  (h1 : room_length = 16)
  (h2 : room_width = 20)
  (h3 : box_area = 10)
  (h4 : boxes_needed = 7) :
  room_length * room_width - boxes_needed * box_area = 250 :=
by sorry

end flooring_problem_l663_66397


namespace gunny_bag_capacity_is_13_l663_66377

/-- Represents the weight of a packet in pounds -/
def packet_weight : ℚ := 16 + 4 / 16

/-- Represents the number of packets -/
def num_packets : ℕ := 2000

/-- Represents the weight of one ton in pounds -/
def pounds_per_ton : ℕ := 2500

/-- Represents the capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℚ := (num_packets * packet_weight) / pounds_per_ton

theorem gunny_bag_capacity_is_13 : gunny_bag_capacity = 13 := by
  sorry

end gunny_bag_capacity_is_13_l663_66377


namespace first_day_over_500_l663_66311

def paperclips (day : ℕ) : ℕ :=
  match day with
  | 0 => 5  -- Monday
  | 1 => 10 -- Tuesday
  | n + 2 => 3 * paperclips (n + 1)

theorem first_day_over_500 :
  (∀ d < 6, paperclips d ≤ 500) ∧ (paperclips 6 > 500) := by
  sorry

end first_day_over_500_l663_66311


namespace prob_blue_or_green_l663_66306

def cube_prob (blue_faces green_faces red_faces : ℕ) : ℚ :=
  (blue_faces + green_faces : ℚ) / (blue_faces + green_faces + red_faces)

theorem prob_blue_or_green : 
  cube_prob 3 1 2 = 2/3 := by sorry

end prob_blue_or_green_l663_66306


namespace spade_heart_diamond_probability_l663_66345

/-- Represents a standard deck of cards -/
structure Deck :=
  (total : Nat)
  (spades : Nat)
  (hearts : Nat)
  (diamonds : Nat)

/-- Calculates the probability of drawing a specific card from the deck -/
def drawProbability (deck : Deck) (targetCards : Nat) : Rat :=
  targetCards / deck.total

/-- Represents the state of the deck after drawing a card -/
def drawCard (deck : Deck) : Deck :=
  { deck with total := deck.total - 1 }

/-- Standard 52-card deck -/
def standardDeck : Deck :=
  { total := 52, spades := 13, hearts := 13, diamonds := 13 }

theorem spade_heart_diamond_probability :
  let firstDraw := drawProbability standardDeck standardDeck.spades
  let secondDraw := drawProbability (drawCard standardDeck) standardDeck.hearts
  let thirdDraw := drawProbability (drawCard (drawCard standardDeck)) standardDeck.diamonds
  firstDraw * secondDraw * thirdDraw = 2197 / 132600 := by sorry

end spade_heart_diamond_probability_l663_66345


namespace total_production_proof_l663_66328

def day_shift_production : ℕ := 4400
def day_shift_multiplier : ℕ := 4

theorem total_production_proof :
  let second_shift_production := day_shift_production / day_shift_multiplier
  day_shift_production + second_shift_production = 5500 := by
  sorry

end total_production_proof_l663_66328
