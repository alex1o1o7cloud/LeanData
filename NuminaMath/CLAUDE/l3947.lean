import Mathlib

namespace silk_diameter_scientific_notation_l3947_394784

/-- The diameter of a certain silk in meters -/
def silk_diameter : ℝ := 0.000014

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem silk_diameter_scientific_notation :
  to_scientific_notation silk_diameter = ScientificNotation.mk 1.4 (-5) sorry :=
sorry

end silk_diameter_scientific_notation_l3947_394784


namespace largest_prime_factor_of_expression_l3947_394789

theorem largest_prime_factor_of_expression : 
  (Nat.factors (18^4 + 12^5 - 6^6)).maximum? = some 11 := by sorry

end largest_prime_factor_of_expression_l3947_394789


namespace least_number_divisible_by_five_primes_l3947_394769

theorem least_number_divisible_by_five_primes : 
  ∀ n : ℕ, n > 0 → (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n ∧ p₅ ∣ n) → n ≥ 2310 :=
by sorry

#check least_number_divisible_by_five_primes

end least_number_divisible_by_five_primes_l3947_394769


namespace correct_ranking_is_cab_l3947_394722

-- Define the teams
inductive Team
| A
| B
| C

-- Define the rankings
def Ranking := Fin 3 → Team

-- Define the prediction type
structure Prediction where
  team1 : Team
  place1 : Fin 3
  team2 : Team
  place2 : Fin 3

-- Define the predictions
def liMing : Prediction := { team1 := Team.A, place1 := 0, team2 := Team.B, place2 := 2 }
def zhangHua : Prediction := { team1 := Team.A, place1 := 2, team2 := Team.C, place2 := 0 }
def wangQiang : Prediction := { team1 := Team.C, place1 := 1, team2 := Team.B, place2 := 2 }

-- Define a function to check if a prediction is half correct
def isHalfCorrect (p : Prediction) (r : Ranking) : Prop :=
  (r p.place1 = p.team1) ≠ (r p.place2 = p.team2)

-- Main theorem
theorem correct_ranking_is_cab :
  ∃! r : Ranking,
    isHalfCorrect liMing r ∧
    isHalfCorrect zhangHua r ∧
    isHalfCorrect wangQiang r ∧
    r 0 = Team.C ∧
    r 1 = Team.A ∧
    r 2 = Team.B :=
sorry

end correct_ranking_is_cab_l3947_394722


namespace five_Y_three_Z_one_eq_one_l3947_394764

/-- Custom operator Y Z -/
def Y_Z (a b c : ℝ) : ℝ := (a - b - c)^2

/-- Theorem stating that 5 Y 3 Z 1 = 1 -/
theorem five_Y_three_Z_one_eq_one : Y_Z 5 3 1 = 1 := by
  sorry

end five_Y_three_Z_one_eq_one_l3947_394764


namespace max_value_of_expression_l3947_394792

theorem max_value_of_expression (a b c d e f g h k : Int) 
  (ha : a = 1 ∨ a = -1) (hb : b = 1 ∨ b = -1) (hc : c = 1 ∨ c = -1) 
  (hd : d = 1 ∨ d = -1) (he : e = 1 ∨ e = -1) (hf : f = 1 ∨ f = -1) 
  (hg : g = 1 ∨ g = -1) (hh : h = 1 ∨ h = -1) (hk : k = 1 ∨ k = -1) : 
  (∀ a' b' c' d' e' f' g' h' k' : Int, 
    (a' = 1 ∨ a' = -1) → (b' = 1 ∨ b' = -1) → (c' = 1 ∨ c' = -1) → 
    (d' = 1 ∨ d' = -1) → (e' = 1 ∨ e' = -1) → (f' = 1 ∨ f' = -1) → 
    (g' = 1 ∨ g' = -1) → (h' = 1 ∨ h' = -1) → (k' = 1 ∨ k' = -1) → 
    a'*e'*k' - a'*f'*h' + b'*f'*g' - b'*d'*k' + c'*d'*h' - c'*e'*g' ≤ 4) ∧
  (∃ a' b' c' d' e' f' g' h' k' : Int, 
    (a' = 1 ∨ a' = -1) ∧ (b' = 1 ∨ b' = -1) ∧ (c' = 1 ∨ c' = -1) ∧ 
    (d' = 1 ∨ d' = -1) ∧ (e' = 1 ∨ e' = -1) ∧ (f' = 1 ∨ f' = -1) ∧ 
    (g' = 1 ∨ g' = -1) ∧ (h' = 1 ∨ h' = -1) ∧ (k' = 1 ∨ k' = -1) ∧
    a'*e'*k' - a'*f'*h' + b'*f'*g' - b'*d'*k' + c'*d'*h' - c'*e'*g' = 4) :=
by sorry

end max_value_of_expression_l3947_394792


namespace miss_darlington_blueberries_l3947_394724

def blueberries_problem (initial_basket : ℕ) (additional_baskets : ℕ) : Prop :=
  let total_blueberries := initial_basket + additional_baskets * initial_basket
  total_blueberries = 200

theorem miss_darlington_blueberries : blueberries_problem 20 9 := by
  sorry

end miss_darlington_blueberries_l3947_394724


namespace range_of_3x_plus_2y_l3947_394763

theorem range_of_3x_plus_2y (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) (hy : -1 ≤ y ∧ y ≤ 4) :
  ∃ (z : ℝ), z = 3*x + 2*y ∧ 1 ≤ z ∧ z ≤ 17 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), (1 ≤ a ∧ a ≤ 3) ∧ (-1 ≤ b ∧ b ≤ 4) ∧ w = 3*a + 2*b) → 1 ≤ w ∧ w ≤ 17 :=
by sorry

end range_of_3x_plus_2y_l3947_394763


namespace noodles_already_have_l3947_394766

/-- The amount of beef Tom has in pounds -/
def beef : ℕ := 10

/-- The ratio of noodles to beef -/
def noodle_to_beef_ratio : ℕ := 2

/-- The weight of each noodle package in pounds -/
def package_weight : ℕ := 2

/-- The number of packages Tom needs to buy -/
def packages_to_buy : ℕ := 8

/-- The total amount of noodles needed in pounds -/
def total_noodles_needed : ℕ := noodle_to_beef_ratio * beef

/-- The amount of noodles Tom needs to buy in pounds -/
def noodles_to_buy : ℕ := packages_to_buy * package_weight

theorem noodles_already_have : 
  total_noodles_needed - noodles_to_buy = 4 := by
  sorry

end noodles_already_have_l3947_394766


namespace triangle_is_equilateral_l3947_394768

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a^2 + b^2 - c^2 = ab and 2cos(A)sin(B) = sin(C), then the triangle is equilateral. -/
theorem triangle_is_equilateral
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a^2 + b^2 - c^2 = a * b)
  (h2 : 2 * Real.cos A * Real.sin B = Real.sin C)
  (h3 : a > 0 ∧ b > 0 ∧ c > 0)
  (h4 : A > 0 ∧ B > 0 ∧ C > 0)
  (h5 : A + B + C = π) :
  a = b ∧ b = c :=
sorry

end triangle_is_equilateral_l3947_394768


namespace apple_ratio_l3947_394735

/-- Proves the ratio of wormy apples to total apples given specific conditions -/
theorem apple_ratio (total : ℕ) (raw : ℕ) (bruised : ℕ) (wormy : ℕ)
  (h1 : total = 85)
  (h2 : raw = 42)
  (h3 : bruised = total / 5 + 9)
  (h4 : wormy = total - bruised - raw) :
  (wormy : ℚ) / total = 17 / 85 := by
  sorry

end apple_ratio_l3947_394735


namespace fitness_center_ratio_l3947_394778

theorem fitness_center_ratio (f m : ℕ) (h1 : f > 0) (h2 : m > 0) : 
  (45 * f + 25 * m) / (f + m) = 35 → f = m := by
  sorry

end fitness_center_ratio_l3947_394778


namespace complex_colinear_l3947_394774

/-- Two non-zero complex numbers lie on the same straight line if and only if their cross product is zero -/
theorem complex_colinear (a₁ b₁ a₂ b₂ : ℝ) (h₁ : a₁ + b₁ * I ≠ 0) (h₂ : a₂ + b₂ * I ≠ 0) :
  (∃ (t : ℝ), (a₂, b₂) = t • (a₁, b₁)) ↔ a₁ * b₂ = a₂ * b₁ := by
  sorry

end complex_colinear_l3947_394774


namespace total_seeds_planted_l3947_394705

/-- The number of seeds planted in each flower bed -/
def seeds_per_bed : ℕ := 10

/-- The number of flower beds -/
def number_of_beds : ℕ := 6

/-- The total number of seeds planted -/
def total_seeds : ℕ := seeds_per_bed * number_of_beds

theorem total_seeds_planted : total_seeds = 60 := by
  sorry

end total_seeds_planted_l3947_394705


namespace A_annual_income_is_537600_l3947_394757

/-- The monthly income of person C in rupees -/
def C_monthly_income : ℕ := 16000

/-- The monthly income of person B in rupees -/
def B_monthly_income : ℕ := C_monthly_income + (C_monthly_income * 12 / 100)

/-- The monthly income of person A in rupees -/
def A_monthly_income : ℕ := B_monthly_income * 5 / 2

/-- The annual income of person A in rupees -/
def A_annual_income : ℕ := A_monthly_income * 12

/-- Theorem stating that A's annual income is 537600 rupees -/
theorem A_annual_income_is_537600 : A_annual_income = 537600 := by
  sorry

end A_annual_income_is_537600_l3947_394757


namespace a_initial_investment_l3947_394718

/-- Proves that given the conditions, A's initial investment is 3000 units -/
theorem a_initial_investment (b_investment : ℝ) (a_doubles : ℝ → ℝ) 
  (h1 : b_investment = 4500)
  (h2 : ∀ x, a_doubles x = 2 * x)
  (h3 : ∀ x, (x + a_doubles x) / 2 = b_investment) : 
  ∃ a_initial : ℝ, a_initial = 3000 := by
  sorry

end a_initial_investment_l3947_394718


namespace valentines_theorem_l3947_394771

def valentines_problem (current_valentines : Real) (additional_valentines : Real) : Prop :=
  let total_students : Real := current_valentines + additional_valentines
  total_students = 74.0

theorem valentines_theorem (current_valentines : Real) (additional_valentines : Real) 
  (h1 : current_valentines = 58.0)
  (h2 : additional_valentines = 16.0) :
  valentines_problem current_valentines additional_valentines := by
  sorry

end valentines_theorem_l3947_394771


namespace albert_needs_twelve_dollars_l3947_394785

/-- The amount of additional money Albert needs to buy art supplies -/
def additional_money_needed (paintbrush_cost set_of_paints_cost wooden_easel_cost current_money : ℚ) : ℚ :=
  paintbrush_cost + set_of_paints_cost + wooden_easel_cost - current_money

theorem albert_needs_twelve_dollars :
  additional_money_needed 1.50 4.35 12.65 6.50 = 12.00 := by
  sorry

end albert_needs_twelve_dollars_l3947_394785


namespace no_solution_for_equation_l3947_394720

theorem no_solution_for_equation : ¬∃ (x : ℝ), (12 / (x^2 - 9) - 2 / (x - 3) = 1 / (x + 3)) := by
  sorry

end no_solution_for_equation_l3947_394720


namespace smallest_difference_2010_l3947_394751

theorem smallest_difference_2010 (a b : ℕ+) : 
  a * b = 2010 → a > b → 
  ∀ (c d : ℕ+), c * d = 2010 → c > d → a - b ≤ c - d → a - b = 37 := by
  sorry

end smallest_difference_2010_l3947_394751


namespace system_solutions_l3947_394721

theorem system_solutions : 
  ∀ (x y z : ℝ), 
    (x + y - z = -1) ∧ 
    (x^2 - y^2 + z^2 = 1) ∧ 
    (-x^3 + y^3 + z^3 = -1) → 
    ((x = 1 ∧ y = -1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1)) :=
by sorry

end system_solutions_l3947_394721


namespace quadratic_y_intercept_l3947_394734

/-- The function f(x) = -(x-1)^2 + 2 intersects the y-axis at the point (0, 1) -/
theorem quadratic_y_intercept :
  let f : ℝ → ℝ := fun x ↦ -(x - 1)^2 + 2
  f 0 = 1 := by
  sorry

end quadratic_y_intercept_l3947_394734


namespace digit_457_of_17_53_l3947_394786

/-- The decimal expansion of 17/53 -/
def decimal_expansion : ℕ → ℕ := sorry

/-- The length of the repeating part in the decimal expansion of 17/53 -/
def cycle_length : ℕ := 20

/-- The 457th digit after the decimal point in the expansion of 17/53 is 1 -/
theorem digit_457_of_17_53 : decimal_expansion 457 = 1 := by sorry

end digit_457_of_17_53_l3947_394786


namespace quadratic_polynomial_root_l3947_394712

theorem quadratic_polynomial_root (x : ℂ) : 
  let p : ℂ → ℂ := λ z => 3 * z^2 - 12 * z + 24
  (p (2 + 2*I) = 0) ∧ (∀ z : ℂ, p z = 3 * z^2 + (-12 * z + 24)) := by
  sorry

end quadratic_polynomial_root_l3947_394712


namespace a_beats_b_by_26_meters_l3947_394725

/-- A beats B by 26 meters in a race -/
theorem a_beats_b_by_26_meters 
  (race_distance : ℝ) 
  (a_time : ℝ) 
  (b_time : ℝ) 
  (h1 : race_distance = 130)
  (h2 : a_time = 20)
  (h3 : b_time = 25) :
  race_distance - (race_distance / b_time * a_time) = 26 := by
  sorry

end a_beats_b_by_26_meters_l3947_394725


namespace cos_pi_fourth_plus_alpha_l3947_394715

theorem cos_pi_fourth_plus_alpha (α : ℝ) (h : Real.sin (α - π/4) = 1/3) :
  Real.cos (π/4 + α) = -1/3 := by
  sorry

end cos_pi_fourth_plus_alpha_l3947_394715


namespace fourth_episode_length_l3947_394745

theorem fourth_episode_length 
  (episode1 : ℕ) 
  (episode2 : ℕ) 
  (episode3 : ℕ) 
  (total_duration : ℕ) 
  (h1 : episode1 = 58)
  (h2 : episode2 = 62)
  (h3 : episode3 = 65)
  (h4 : total_duration = 240) :
  total_duration - (episode1 + episode2 + episode3) = 55 :=
by sorry

end fourth_episode_length_l3947_394745


namespace sector_arc_length_l3947_394790

theorem sector_arc_length (θ : Real) (r : Real) (h1 : θ = 120) (h2 : r = 2) :
  let arc_length := θ / 360 * (2 * Real.pi * r)
  arc_length = 4 * Real.pi / 3 := by
  sorry

end sector_arc_length_l3947_394790


namespace sum_of_coordinates_D_l3947_394741

/-- Given points C and M, where M is the midpoint of segment CD, 
    prove that the sum of coordinates of point D is 0 -/
theorem sum_of_coordinates_D (C M : ℝ × ℝ) (h1 : C = (-1, 5)) (h2 : M = (4, -2)) : 
  let D := (2 * M.1 - C.1, 2 * M.2 - C.2)
  D.1 + D.2 = 0 := by
sorry


end sum_of_coordinates_D_l3947_394741


namespace quadratic_real_roots_l3947_394746

/-- The quadratic equation (k-1)x^2 - 2kx + k + 3 = 0 has real roots if and only if k ≤ 3/2 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 - 2 * k * x + k + 3 = 0) ↔ k ≤ 3/2 := by
  sorry

end quadratic_real_roots_l3947_394746


namespace fran_average_speed_l3947_394727

/-- Calculates the average speed required for Fran to cover the same distance as Joann -/
theorem fran_average_speed (joann_speed1 joann_time1 joann_speed2 joann_time2 fran_time : ℝ) 
  (h1 : joann_speed1 = 15)
  (h2 : joann_time1 = 4)
  (h3 : joann_speed2 = 12)
  (h4 : joann_time2 = 0.5)
  (h5 : fran_time = 4) :
  (joann_speed1 * joann_time1 + joann_speed2 * joann_time2) / fran_time = 16.5 := by
  sorry

#check fran_average_speed

end fran_average_speed_l3947_394727


namespace odd_sum_1_to_25_l3947_394793

theorem odd_sum_1_to_25 : 
  let odds := (List.range 13).map (fun i => 2 * i + 1)
  odds.sum = 169 := by
  sorry

end odd_sum_1_to_25_l3947_394793


namespace election_win_margin_l3947_394756

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) (winner_percentage : ℚ) :
  winner_percentage = 62 / 100 →
  winner_votes = 930 →
  winner_votes = total_votes * winner_percentage →
  winner_votes - (total_votes - winner_votes) = 360 :=
by sorry

end election_win_margin_l3947_394756


namespace sqrt_equation_solution_l3947_394700

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end sqrt_equation_solution_l3947_394700


namespace product_of_numbers_l3947_394737

theorem product_of_numbers (x y : ℝ) : x + y = 50 ∧ x - y = 6 → x * y = 616 := by
  sorry

end product_of_numbers_l3947_394737


namespace distance_traveled_l3947_394795

/-- Given a speed of 40 km/hr and a time of 6 hr, prove that the distance traveled is 240 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 40) (h2 : time = 6) :
  speed * time = 240 := by
  sorry

end distance_traveled_l3947_394795


namespace toms_beef_quantity_tom_has_ten_pounds_beef_l3947_394750

/-- Represents the problem of determining Tom's beef quantity for lasagna -/
theorem toms_beef_quantity (noodles_to_beef_ratio : ℕ) 
  (existing_noodles : ℕ) (package_size : ℕ) (packages_to_buy : ℕ) : ℕ :=
  let total_noodles := existing_noodles + package_size * packages_to_buy
  total_noodles / noodles_to_beef_ratio

/-- Proves that Tom has 10 pounds of beef given the problem conditions -/
theorem tom_has_ten_pounds_beef : 
  toms_beef_quantity 2 4 2 8 = 10 := by
  sorry

end toms_beef_quantity_tom_has_ten_pounds_beef_l3947_394750


namespace brigade_task_completion_time_l3947_394711

theorem brigade_task_completion_time :
  ∀ (x : ℝ),
  (x > 0) →
  (x - 15 > 0) →
  (18 / x + 6 / (x - 15) = 0.6) →
  x = 62.25 :=
by
  sorry

end brigade_task_completion_time_l3947_394711


namespace min_value_of_a_plus_2b_l3947_394730

theorem min_value_of_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = a * b - 3) :
  ∀ x y, x > 0 → y > 0 → x + y = x * y - 3 → a + 2 * b ≤ x + 2 * y ∧
  ∃ a₀ b₀, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = a₀ * b₀ - 3 ∧ a₀ + 2 * b₀ = 4 * Real.sqrt 2 + 3 :=
sorry

end min_value_of_a_plus_2b_l3947_394730


namespace k_value_l3947_394799

/-- Two circles centered at the origin with given points and distance --/
structure TwoCircles where
  P : ℝ × ℝ
  S : ℝ × ℝ
  QR : ℝ

/-- The value of k in the point S(0, k) --/
def k (c : TwoCircles) : ℝ := c.S.2

/-- Theorem stating the value of k --/
theorem k_value (c : TwoCircles) 
  (h1 : c.P = (12, 5)) 
  (h2 : c.S.1 = 0) 
  (h3 : c.QR = 4) : 
  k c = 9 := by
  sorry

end k_value_l3947_394799


namespace opposite_numbers_solution_same_type_radicals_solution_l3947_394797

-- Part 1
theorem opposite_numbers_solution (x : ℝ) :
  2 * x^2 + 3 * x - 5 = -(-2 * x + 2) →
  x = -3/2 ∨ x = 1 :=
by sorry

-- Part 2
theorem same_type_radicals_solution (m : ℝ) :
  m^2 - 6 = 6 * m + 1 →
  m^2 - 6 ≥ 0 →
  m = 7 :=
by sorry

end opposite_numbers_solution_same_type_radicals_solution_l3947_394797


namespace pure_imaginary_complex_number_l3947_394780

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.mk (a^2 - 3*a + 2) (a - 2)).im ≠ 0 ∧ 
  (Complex.mk (a^2 - 3*a + 2) (a - 2)).re = 0 → 
  a = 1 := by
sorry

end pure_imaginary_complex_number_l3947_394780


namespace algebra_problem_percentage_l3947_394710

theorem algebra_problem_percentage (total_problems : ℕ) 
  (linear_equations : ℕ) (h1 : total_problems = 140) 
  (h2 : linear_equations = 28) : 
  (linear_equations * 2 : ℚ) / total_problems * 100 = 40 := by
  sorry

end algebra_problem_percentage_l3947_394710


namespace racket_carton_problem_l3947_394736

/-- Given two types of tennis racket cartons, one holding 2 rackets and the other
    holding an unknown number x, prove that x = 1 when 38 cartons of the first type
    and 24 cartons of the second type are used to pack a total of 100 rackets. -/
theorem racket_carton_problem (x : ℕ) : 
  (38 * 2 + 24 * x = 100) → x = 1 := by
  sorry

end racket_carton_problem_l3947_394736


namespace distance_to_origin_l3947_394770

theorem distance_to_origin : Real.sqrt 13 = Real.sqrt ((2 : ℝ)^2 + (-3 : ℝ)^2) := by sorry

end distance_to_origin_l3947_394770


namespace quadratic_coefficient_l3947_394706

theorem quadratic_coefficient (b : ℝ) (n : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 24 = (x + n)^2 + 16) → 
  b = 4 * Real.sqrt 2 :=
sorry

end quadratic_coefficient_l3947_394706


namespace nell_ace_cards_l3947_394703

/-- The number of baseball cards Nell has after giving some to Jeff -/
def remaining_baseball_cards : ℕ := 111

/-- The difference between Ace cards and baseball cards Nell has now -/
def ace_baseball_difference : ℕ := 265

/-- The number of Ace cards Nell has now -/
def current_ace_cards : ℕ := remaining_baseball_cards + ace_baseball_difference

theorem nell_ace_cards : current_ace_cards = 376 := by
  sorry

end nell_ace_cards_l3947_394703


namespace arrangements_count_l3947_394728

/-- The number of workers available for the production process -/
def total_workers : Nat := 6

/-- The number of steps in the production process -/
def total_steps : Nat := 4

/-- The set of workers who can oversee the first step -/
def first_step_workers : Finset Char := {'A', 'B'}

/-- The set of workers who can oversee the fourth step -/
def fourth_step_workers : Finset Char := {'A', 'C'}

/-- The function that calculates the number of arrangements -/
def count_arrangements : Nat :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the number of different arrangements is 36 -/
theorem arrangements_count : count_arrangements = 36 := by
  sorry

end arrangements_count_l3947_394728


namespace article_price_after_decrease_l3947_394742

theorem article_price_after_decrease (decreased_price : ℝ) (decrease_percentage : ℝ) (original_price : ℝ) : 
  decreased_price = 532 → 
  decrease_percentage = 24 → 
  decreased_price = original_price * (1 - decrease_percentage / 100) → 
  original_price = 700 := by
sorry

end article_price_after_decrease_l3947_394742


namespace tangent_point_on_circle_l3947_394794

theorem tangent_point_on_circle (a : ℝ) : 
  ((-1 - 1)^2 + a^2 = 4) ↔ (a = 0) := by sorry

end tangent_point_on_circle_l3947_394794


namespace log_expression_equals_zero_l3947_394716

theorem log_expression_equals_zero : 
  (Real.log 270 / Real.log 3) / (Real.log 3 / Real.log 54) - 
  (Real.log 540 / Real.log 3) / (Real.log 3 / Real.log 27) = 0 := by
sorry

end log_expression_equals_zero_l3947_394716


namespace x9_plus_y9_not_eq_neg_one_l3947_394743

theorem x9_plus_y9_not_eq_neg_one :
  ∀ (x y : ℂ),
  x = (-1 + Complex.I * Real.sqrt 3) / 2 →
  y = (-1 - Complex.I * Real.sqrt 3) / 2 →
  x^9 + y^9 ≠ -1 := by
sorry

end x9_plus_y9_not_eq_neg_one_l3947_394743


namespace inequality_proof_l3947_394762

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≥ b) (hbc : b ≥ c)
  (hsum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 := by
  sorry

end inequality_proof_l3947_394762


namespace function_bounds_l3947_394752

-- Define the function f
def f (x y z : ℝ) : ℝ := 7*x + 5*y - 2*z

-- State the theorem
theorem function_bounds (x y z : ℝ) 
  (h1 : -1 ≤ 2*x + y - z ∧ 2*x + y - z ≤ 8)
  (h2 : 2 ≤ x - y + z ∧ x - y + z ≤ 9)
  (h3 : -3 ≤ x + 2*y - z ∧ x + 2*y - z ≤ 7) :
  -6 ≤ f x y z ∧ f x y z ≤ 47 := by
  sorry

end function_bounds_l3947_394752


namespace shortest_distance_to_circle_l3947_394707

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + y^2 - 8*y + 40 = 0

def point : ℝ × ℝ := (4, -3)

theorem shortest_distance_to_circle :
  ∃ (d : ℝ), d = 5 * Real.sqrt 2 - 1 ∧
  ∀ (p : ℝ × ℝ), circle_equation p.1 p.2 →
  Real.sqrt ((p.1 - point.1)^2 + (p.2 - point.2)^2) ≥ d :=
sorry

end shortest_distance_to_circle_l3947_394707


namespace system_solutions_l3947_394767

theorem system_solutions :
  let S := {(x, y) : ℝ × ℝ | x^2 + y^2 = x ∧ 2*x*y = y}
  S = {(0, 0), (1, 0), (1/2, 1/2), (1/2, -1/2)} := by
  sorry

end system_solutions_l3947_394767


namespace least_reducible_fraction_l3947_394755

theorem least_reducible_fraction : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), 0 < m ∧ m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 13) ∧ k ∣ (5*m + 6))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 13) ∧ k ∣ (5*n + 6)) ∧
  n = 84 :=
by sorry

end least_reducible_fraction_l3947_394755


namespace multiple_properties_l3947_394732

theorem multiple_properties (x y : ℤ) 
  (hx : ∃ k : ℤ, x = 4 * k) 
  (hy : ∃ m : ℤ, y = 8 * m) : 
  (∃ n : ℤ, y = 4 * n) ∧ 
  (∃ p : ℤ, x - y = 4 * p) ∧ 
  (∃ q : ℤ, x - y = 2 * q) := by
sorry

end multiple_properties_l3947_394732


namespace sphere_volume_from_intersection_l3947_394759

/-- The volume of a sphere, given specific intersection properties -/
theorem sphere_volume_from_intersection (r : ℝ) : 
  (∃ (d : ℝ), d = 1 ∧ π = π * (r^2 - d^2)) →
  (4/3) * π * r^3 = (8 * Real.sqrt 2 / 3) * π :=
by sorry

end sphere_volume_from_intersection_l3947_394759


namespace combined_share_specific_case_l3947_394781

/-- Represents the share distribution problem -/
def ShareDistribution (total : ℚ) (ratio : List ℚ) : Prop :=
  total > 0 ∧ ratio.length = 5 ∧ ∀ r ∈ ratio, r > 0

/-- Calculates the combined share of two specific parts in the ratio -/
def CombinedShare (total : ℚ) (ratio : List ℚ) (index1 index2 : ℕ) : ℚ :=
  let sum_ratio := ratio.sum
  let part_value := total / sum_ratio
  part_value * (ratio[index1]! + ratio[index2]!)

theorem combined_share_specific_case :
  ∀ (total : ℚ) (ratio : List ℚ),
    ShareDistribution total ratio →
    ratio = [2, 4, 3, 1, 5] →
    total = 12000 →
    CombinedShare total ratio 3 4 = 4800 := by
  sorry

end combined_share_specific_case_l3947_394781


namespace olympic_system_matches_l3947_394726

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  is_single_elimination : Bool

/-- Calculates the number of matches in a single-elimination tournament. -/
def matches_played (t : Tournament) : ℕ :=
  if t.is_single_elimination then t.num_teams - 1 else 0

/-- Theorem: A single-elimination tournament with 30 teams has 29 matches. -/
theorem olympic_system_matches :
  ∀ t : Tournament, t.num_teams = 30 ∧ t.is_single_elimination → matches_played t = 29 := by
  sorry

end olympic_system_matches_l3947_394726


namespace michael_needs_more_money_l3947_394747

def michael_money : ℕ := 50
def cake_cost : ℕ := 20
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5

theorem michael_needs_more_money :
  (cake_cost + bouquet_cost + balloons_cost) - michael_money = 11 := by
  sorry

end michael_needs_more_money_l3947_394747


namespace line_translation_slope_l3947_394739

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a translation function
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + dy - l.slope * dx }

-- State the theorem
theorem line_translation_slope (l : Line) :
  translate l 3 2 = l → l.slope = 2/3 := by
  sorry

end line_translation_slope_l3947_394739


namespace division_result_l3947_394796

theorem division_result : (712.5 : ℝ) / 12.5 = 57 := by sorry

end division_result_l3947_394796


namespace parabola_vertex_trajectory_l3947_394776

/-- The trajectory of the vertex of a parabola -/
def vertex_trajectory (x y m : ℝ) : Prop :=
  y - 4*x - 4*m*y = 0

/-- The equation of the trajectory -/
def trajectory_equation (x y : ℝ) : Prop :=
  y^2 = -4*x

/-- Theorem: The trajectory of the vertex of the parabola y - 4x - 4my = 0 
    is described by the equation y^2 = -4x -/
theorem parabola_vertex_trajectory :
  ∀ x y : ℝ, (∃ m : ℝ, vertex_trajectory x y m) ↔ trajectory_equation x y :=
sorry

end parabola_vertex_trajectory_l3947_394776


namespace equation_solutions_l3947_394723

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 9 = 0 ↔ x = 3/2 ∨ x = -3/2) ∧
  (∀ x : ℝ, 64 * (x - 2)^3 - 1 = 0 ↔ x = 9/4) := by
  sorry

end equation_solutions_l3947_394723


namespace binomial_150_150_equals_1_l3947_394775

theorem binomial_150_150_equals_1 : Nat.choose 150 150 = 1 := by
  sorry

end binomial_150_150_equals_1_l3947_394775


namespace cos_150_degrees_l3947_394798

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_150_degrees_l3947_394798


namespace expand_product_l3947_394738

theorem expand_product (x : ℝ) : (x + 3) * (x - 1) * (x + 4) = x^3 + 6*x^2 + 5*x - 12 := by
  sorry

end expand_product_l3947_394738


namespace terrell_hike_distance_l3947_394758

/-- The total distance Terrell hiked over two days -/
theorem terrell_hike_distance (saturday_distance sunday_distance : ℝ) 
  (h1 : saturday_distance = 8.2)
  (h2 : sunday_distance = 1.6) :
  saturday_distance + sunday_distance = 9.8 := by
  sorry

end terrell_hike_distance_l3947_394758


namespace geometric_sequence_property_l3947_394779

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => 0
  | n + 1 => S a n + |a (n + 1) - 4|

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4 = 8 * a 1) →
  arithmetic_sequence (λ n => match n with
                              | 1 => a 1
                              | 2 => a 2 + 1
                              | 3 => a 3
                              | _ => 0) →
  (∀ n : ℕ, a n = 2^n) ∧
  (∀ n : ℕ, S a n = if n = 1 then 2 else 2^(n+1) - 4*n + 2) :=
by sorry

end geometric_sequence_property_l3947_394779


namespace probability_of_spades_formula_l3947_394749

def standardDeckSize : ℕ := 52
def spadesInDeck : ℕ := 13
def cardsDrawn : ℕ := 13

def probabilityOfSpades (n : ℕ) : ℚ :=
  (Nat.choose spadesInDeck n * Nat.choose (standardDeckSize - spadesInDeck) (cardsDrawn - n)) /
  Nat.choose standardDeckSize cardsDrawn

theorem probability_of_spades_formula (n : ℕ) (h1 : n ≤ spadesInDeck) (h2 : n ≤ cardsDrawn) :
  probabilityOfSpades n = (Nat.choose spadesInDeck n * Nat.choose (standardDeckSize - spadesInDeck) (cardsDrawn - n)) /
                          Nat.choose standardDeckSize cardsDrawn := by
  sorry

end probability_of_spades_formula_l3947_394749


namespace other_leg_length_l3947_394753

/-- Given a right triangle with one leg of length 5 and hypotenuse of length 11,
    the length of the other leg is 4√6. -/
theorem other_leg_length (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
    (h_leg : a = 5) (h_hyp : c = 11) : b = 4 * Real.sqrt 6 := by
  sorry

end other_leg_length_l3947_394753


namespace unplaced_unbroken_bottles_l3947_394744

-- Define the parameters
def total_bottles : ℕ := 250
def total_crates : ℕ := 15
def small_crate_capacity : ℕ := 8
def medium_crate_capacity : ℕ := 12
def large_crate_capacity : ℕ := 20
def available_small_crates : ℕ := 5
def available_medium_crates : ℕ := 5
def available_large_crates : ℕ := 5
def max_usable_small_crates : ℕ := 3
def max_usable_medium_crates : ℕ := 4
def max_usable_large_crates : ℕ := 5
def broken_bottles : ℕ := 11

-- Theorem statement
theorem unplaced_unbroken_bottles : 
  total_bottles - broken_bottles - 
  (max_usable_small_crates * small_crate_capacity + 
   max_usable_medium_crates * medium_crate_capacity + 
   max_usable_large_crates * large_crate_capacity) = 67 := by
  sorry

end unplaced_unbroken_bottles_l3947_394744


namespace contrapositive_equivalence_l3947_394740

theorem contrapositive_equivalence :
  (¬(a^2 = 1) → ¬(a = -1)) ↔ (a = -1 → a^2 = 1) :=
by sorry

end contrapositive_equivalence_l3947_394740


namespace altitude_feet_locus_l3947_394713

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The locus of altitude feet for a varying right triangle -/
def altitudeFeetLocus (S₁ S₂ : Circle) (A : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- The arc or segment of the circle with diameter AM -/
def circleArcOrSegment (A M : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- External center of similarity of two circles -/
def externalCenterOfSimilarity (S₁ S₂ : Circle) : ℝ × ℝ :=
  sorry

/-- Main theorem: The locus of altitude feet is an arc or segment of circle with diameter AM -/
theorem altitude_feet_locus (S₁ S₂ : Circle) (A : ℝ × ℝ) :
  altitudeFeetLocus S₁ S₂ A = 
  circleArcOrSegment A (externalCenterOfSimilarity S₁ S₂) :=
by sorry

end altitude_feet_locus_l3947_394713


namespace train_length_calculation_l3947_394748

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : Real) (time_to_cross : Real) (bridge_length : Real) :
  train_speed = 72 * 1000 / 3600 → -- Convert km/hr to m/s
  time_to_cross = 12.299016078713702 →
  bridge_length = 136 →
  (train_speed * time_to_cross - bridge_length) = 110.98032157427404 := by
  sorry

#check train_length_calculation

end train_length_calculation_l3947_394748


namespace division_remainder_l3947_394731

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = divisor * quotient + remainder →
  dividend = 167 →
  divisor = 18 →
  quotient = 9 →
  remainder = 5 := by
sorry

end division_remainder_l3947_394731


namespace solve_equation_chain_l3947_394708

theorem solve_equation_chain (v y z w x : ℤ) 
  (h1 : x = y + 6)
  (h2 : y = z + 11)
  (h3 : z = w + 21)
  (h4 : w = v + 30)
  (h5 : v = 90) : x = 158 := by
  sorry

end solve_equation_chain_l3947_394708


namespace consecutive_zeros_in_power_of_five_l3947_394788

theorem consecutive_zeros_in_power_of_five : 
  ∃ n : ℕ, n < 1000000 ∧ (5^n : ℕ) % 1000000 = 0 := by sorry

end consecutive_zeros_in_power_of_five_l3947_394788


namespace collinear_vectors_x_value_l3947_394765

/-- Two vectors in ℝ² are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (x, 1)
  collinear a b → x = -2 := by
  sorry

end collinear_vectors_x_value_l3947_394765


namespace quadratic_minimum_l3947_394791

/-- 
Given a quadratic function y = 3x^2 + px + q,
if the minimum value of y is 4,
then q = p^2/12 + 4
-/
theorem quadratic_minimum (p q : ℝ) : 
  (∀ x, 3 * x^2 + p * x + q ≥ 4) ∧ 
  (∃ x, 3 * x^2 + p * x + q = 4) →
  q = p^2 / 12 + 4 :=
by sorry

end quadratic_minimum_l3947_394791


namespace geometric_sequence_sum_l3947_394754

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = a n * q) →
  a 2 = 1 - a 1 →
  a 4 = 9 - a 3 →
  a 4 + a 5 = 27 := by
sorry

end geometric_sequence_sum_l3947_394754


namespace next_month_has_five_wednesdays_l3947_394773

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with its number of days and starting day -/
structure Month where
  days : Nat
  startDay : DayOfWeek

/-- Counts the occurrences of a specific day in a month -/
def countDaysInMonth (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Returns the next month given the current month -/
def nextMonth (m : Month) : Month :=
  sorry

/-- Theorem: If a month has 5 Saturdays, 5 Sundays, 4 Mondays, and 4 Fridays,
    then the following month will have 5 Wednesdays -/
theorem next_month_has_five_wednesdays (m : Month) :
  countDaysInMonth m DayOfWeek.Saturday = 5 →
  countDaysInMonth m DayOfWeek.Sunday = 5 →
  countDaysInMonth m DayOfWeek.Monday = 4 →
  countDaysInMonth m DayOfWeek.Friday = 4 →
  countDaysInMonth (nextMonth m) DayOfWeek.Wednesday = 5 :=
by
  sorry

end next_month_has_five_wednesdays_l3947_394773


namespace satisfactory_fraction_is_eleven_fifteenths_l3947_394719

/-- Represents the distribution of grades in a class assessment. -/
structure GradeDistribution where
  a : Nat -- Number of A's
  b : Nat -- Number of B's
  c : Nat -- Number of C's
  d : Nat -- Number of D's
  ef : Nat -- Number of E's and F's combined

/-- Calculates the fraction of satisfactory grades given a grade distribution. -/
def fractionSatisfactory (grades : GradeDistribution) : Rat :=
  let satisfactory := grades.a + grades.b + grades.c + grades.d
  let total := satisfactory + grades.ef
  satisfactory / total

/-- Theorem stating that for the given grade distribution, 
    the fraction of satisfactory grades is 11/15. -/
theorem satisfactory_fraction_is_eleven_fifteenths : 
  fractionSatisfactory { a := 7, b := 6, c := 5, d := 4, ef := 8 } = 11 / 15 := by
  sorry


end satisfactory_fraction_is_eleven_fifteenths_l3947_394719


namespace research_paper_word_count_l3947_394701

/-- Calculates the number of words typed given a typing speed and duration. -/
def words_typed (typing_speed : ℕ) (duration_hours : ℕ) : ℕ :=
  typing_speed * (duration_hours * 60)

/-- Proves that given a typing speed of 38 words per minute and a duration of 2 hours,
    the total number of words typed is 4560. -/
theorem research_paper_word_count :
  words_typed 38 2 = 4560 := by
  sorry

end research_paper_word_count_l3947_394701


namespace consecutive_integers_median_l3947_394772

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (h1 : n = 64) (h2 : sum = 2^12) :
  (sum / n : ℚ) = 64 := by
  sorry

end consecutive_integers_median_l3947_394772


namespace cole_miles_l3947_394729

theorem cole_miles (xavier katie cole : ℕ) 
  (h1 : xavier = 3 * katie) 
  (h2 : katie = 4 * cole) 
  (h3 : xavier = 84) : 
  cole = 7 := by
  sorry

end cole_miles_l3947_394729


namespace power_of_power_l3947_394709

theorem power_of_power (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end power_of_power_l3947_394709


namespace intersection_points_l3947_394782

-- Define the curves
def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 5/2
def curve2 (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1
def curve3 (x y : ℝ) : Prop := x^2 + y^2/4 = 1
def curve4 (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + y = Real.sqrt 5

-- Define a function to check if a curve has only one intersection point with the line
def hasOnlyOneIntersection (curve : (ℝ → ℝ → Prop)) : Prop :=
  ∃! p : ℝ × ℝ, curve p.1 p.2 ∧ line p.1 p.2

-- State the theorem
theorem intersection_points :
  hasOnlyOneIntersection curve1 ∧
  hasOnlyOneIntersection curve3 ∧
  hasOnlyOneIntersection curve4 ∧
  ¬hasOnlyOneIntersection curve2 := by
  sorry


end intersection_points_l3947_394782


namespace bike_shop_theorem_l3947_394704

/-- Represents the sales and pricing information for bike types A and B -/
structure BikeShop where
  lastYearRevenueA : ℕ
  priceDrop : ℕ
  revenueDecrease : ℚ
  totalNewBikes : ℕ
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceB : ℕ

/-- Calculates the selling price of type A bikes this year -/
def sellingPriceA (shop : BikeShop) : ℕ := sorry

/-- Calculates the maximum profit and optimal purchase quantities -/
def maxProfit (shop : BikeShop) : ℕ × ℕ × ℕ := sorry

/-- Main theorem stating the correct selling price and maximum profit -/
theorem bike_shop_theorem (shop : BikeShop) 
  (h1 : shop.lastYearRevenueA = 50000)
  (h2 : shop.priceDrop = 400)
  (h3 : shop.revenueDecrease = 1/5)
  (h4 : shop.totalNewBikes = 60)
  (h5 : shop.purchasePriceA = 1100)
  (h6 : shop.purchasePriceB = 1400)
  (h7 : shop.sellingPriceB = 2000) :
  sellingPriceA shop = 1600 ∧ 
  maxProfit shop = (34000, 20, 40) := by sorry


end bike_shop_theorem_l3947_394704


namespace square_of_999_l3947_394717

theorem square_of_999 : 999 * 999 = 998001 := by
  sorry

end square_of_999_l3947_394717


namespace y_over_x_equals_two_l3947_394783

theorem y_over_x_equals_two (x y : ℝ) (h : y / 2 = (2 * y - x) / 3) : y / x = 2 := by
  sorry

end y_over_x_equals_two_l3947_394783


namespace sum_of_digits_in_special_addition_formula_l3947_394761

/-- Represents a four-digit number ABCD -/
def fourDigitNumber (a b c d : Nat) : Nat :=
  1000 * a + 100 * b + 10 * c + d

/-- The main theorem -/
theorem sum_of_digits_in_special_addition_formula 
  (A B C D : Nat) 
  (h_different : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_single_digit : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10)
  (h_formula : fourDigitNumber D C B A + fourDigitNumber A B C D = 10 * fourDigitNumber A B C D) :
  A + B + C + D = 18 := by
    sorry

end sum_of_digits_in_special_addition_formula_l3947_394761


namespace expression_equals_40_times_10_to_2003_l3947_394760

theorem expression_equals_40_times_10_to_2003 :
  (2^2003 + 5^2004)^2 - (2^2003 - 5^2004)^2 = 40 * 10^2003 := by
  sorry

end expression_equals_40_times_10_to_2003_l3947_394760


namespace females_advanced_degrees_count_l3947_394787

/-- Represents the employee distribution in a company -/
structure EmployeeDistribution where
  total : Nat
  females : Nat
  advanced_degrees : Nat
  males_college_only : Nat

/-- Calculates the number of females with advanced degrees -/
def females_with_advanced_degrees (e : EmployeeDistribution) : Nat :=
  e.advanced_degrees - (e.total - e.females - e.males_college_only)

/-- Theorem stating the number of females with advanced degrees -/
theorem females_advanced_degrees_count 
  (e : EmployeeDistribution)
  (h1 : e.total = 200)
  (h2 : e.females = 120)
  (h3 : e.advanced_degrees = 100)
  (h4 : e.males_college_only = 40) :
  females_with_advanced_degrees e = 60 := by
  sorry

#eval females_with_advanced_degrees { 
  total := 200, 
  females := 120, 
  advanced_degrees := 100, 
  males_college_only := 40 
}

end females_advanced_degrees_count_l3947_394787


namespace intersection_sine_value_l3947_394714

theorem intersection_sine_value (x₀ : Real) (y₀ : Real) :
  x₀ ∈ Set.Ioo 0 (π / 2) →
  y₀ = 3 * Real.cos x₀ →
  y₀ = 8 * Real.tan x₀ →
  Real.sin x₀ = 1 / 3 := by
sorry

end intersection_sine_value_l3947_394714


namespace find_y_value_l3947_394777

/-- Given five numbers in increasing order with specific conditions, prove y equals 16 -/
theorem find_y_value (x y : ℝ) : 
  2 < 5 ∧ 5 < x ∧ x < 10 ∧ 10 < y ∧  -- Increasing order condition
  x = 7 ∧  -- Median condition
  (2 + 5 + x + 10 + y) / 5 = 8  -- Mean condition
  → y = 16 := by
sorry

end find_y_value_l3947_394777


namespace greatest_three_digit_multiple_of_17_l3947_394702

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l3947_394702


namespace squirrels_in_tree_l3947_394733

/-- Given a tree with nuts and squirrels, where the number of squirrels is 2 more than the number of nuts,
    prove that if there are 2 nuts, then there are 4 squirrels. -/
theorem squirrels_in_tree (nuts : ℕ) (squirrels : ℕ) 
  (h1 : nuts = 2) 
  (h2 : squirrels = nuts + 2) : 
  squirrels = 4 := by
  sorry

end squirrels_in_tree_l3947_394733
