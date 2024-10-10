import Mathlib

namespace tenth_largest_four_digit_odd_l3147_314757

/-- The set of odd digits -/
def OddDigits : Set Nat := {1, 3, 5, 7, 9}

/-- A four-digit number composed of only odd digits -/
def FourDigitOddNumber (a b c d : Nat) : Prop :=
  a ∈ OddDigits ∧ b ∈ OddDigits ∧ c ∈ OddDigits ∧ d ∈ OddDigits ∧
  1000 ≤ a * 1000 + b * 100 + c * 10 + d ∧ a * 1000 + b * 100 + c * 10 + d ≤ 9999

/-- The theorem stating that 9971 is the tenth largest four-digit number composed of only odd digits -/
theorem tenth_largest_four_digit_odd : 
  (∃ (n : Nat), n = 9 ∧ 
    (∃ (a b c d : Nat), FourDigitOddNumber a b c d ∧ 
      a * 1000 + b * 100 + c * 10 + d > 9971)) := by sorry

end tenth_largest_four_digit_odd_l3147_314757


namespace fib_odd_index_not_divisible_by_4k_plus_3_prime_l3147_314788

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define prime numbers of the form 4k + 3
def isPrime4kPlus3 (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ k : ℕ, p = 4 * k + 3

-- Theorem statement
theorem fib_odd_index_not_divisible_by_4k_plus_3_prime (n : ℕ) (p : ℕ) 
  (h_prime : isPrime4kPlus3 p) : ¬(p ∣ fib (2 * n + 1)) := by
  sorry

end fib_odd_index_not_divisible_by_4k_plus_3_prime_l3147_314788


namespace x_fourth_plus_reciprocal_l3147_314773

theorem x_fourth_plus_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^4 + 1/x^4 = 527 := by
  sorry

end x_fourth_plus_reciprocal_l3147_314773


namespace jane_hector_meeting_l3147_314737

/-- Represents the points around the block area -/
inductive Point := | A | B | C | D | E

/-- The total distance around the block area -/
def total_distance : ℕ := 24

/-- Hector's walking speed -/
def hector_speed : ℝ := 1

/-- Jane's walking speed -/
def jane_speed : ℝ := 3 * hector_speed

/-- The distance walked by Hector when they meet -/
def hector_distance : ℝ := 6

/-- The distance walked by Jane when they meet -/
def jane_distance : ℝ := 18

/-- The point where Jane and Hector meet -/
def meeting_point : Point := Point.C

theorem jane_hector_meeting :
  (jane_speed = 3 * hector_speed) →
  (hector_distance + jane_distance = total_distance) →
  (jane_distance = 3 * hector_distance) →
  meeting_point = Point.C :=
by sorry

end jane_hector_meeting_l3147_314737


namespace blanket_price_problem_l3147_314721

/-- Proves that the unknown rate of two blankets is 228.75, given the conditions of the problem -/
theorem blanket_price_problem (price_3 : ℕ) (price_1 : ℕ) (discount : ℚ) (tax : ℚ) (avg_price : ℕ) :
  price_3 = 100 →
  price_1 = 150 →
  discount = 1/10 →
  tax = 3/20 →
  avg_price = 150 →
  let total_blankets : ℕ := 6
  let discounted_price_3 : ℚ := 3 * price_3 * (1 - discount)
  let taxed_price_1 : ℚ := price_1 * (1 + tax)
  let total_price : ℚ := total_blankets * avg_price
  ∃ x : ℚ, 
    x = (total_price - discounted_price_3 - taxed_price_1) / 2 ∧ 
    x = 457.5 / 2 := by
  sorry

#eval 457.5 / 2  -- Should output 228.75

end blanket_price_problem_l3147_314721


namespace max_value_polynomial_l3147_314792

theorem max_value_polynomial (a b : ℝ) (h : a + b = 5) :
  ∃ M : ℝ, M = 6084 / 17 ∧ 
  ∀ x y : ℝ, x + y = 5 → 
  x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ M ∧
  ∃ a b : ℝ, a + b = 5 ∧ 
  a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 = M :=
sorry

end max_value_polynomial_l3147_314792


namespace combined_mean_of_two_sets_l3147_314798

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 5 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 20 →
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℚ) = 235 / 13 := by
  sorry

end combined_mean_of_two_sets_l3147_314798


namespace intersection_point_coordinates_l3147_314700

/-- Given a triangle XYZ with specific point ratios, prove that the intersection of certain lines has coordinates (1/3, 2/3, 0) -/
theorem intersection_point_coordinates (X Y Z D E F P : ℝ × ℝ × ℝ) : 
  -- Triangle XYZ exists
  X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X →
  -- D is on YZ extended with ratio 4:1
  ∃ t : ℝ, t > 1 ∧ D = t • Z + (1 - t) • Y ∧ (t - 1) / (5 - t) = 4 →
  -- E is on XZ with ratio 3:2
  ∃ s : ℝ, 0 < s ∧ s < 1 ∧ E = s • X + (1 - s) • Z ∧ s / (1 - s) = 3 / 2 →
  -- F is on XY with ratio 2:1
  ∃ r : ℝ, 0 < r ∧ r < 1 ∧ F = r • X + (1 - r) • Y ∧ r / (1 - r) = 2 →
  -- P is the intersection of BF and YD
  ∃ u v : ℝ, P = u • F + (1 - u) • E ∧ P = v • D + (1 - v) • Y →
  -- Conclusion: P has coordinates (1/3, 2/3, 0) in terms of X, Y, Z
  P = (1/3) • X + (2/3) • Y + 0 • Z :=
by sorry

end intersection_point_coordinates_l3147_314700


namespace circle_intersection_and_tangent_line_l3147_314765

-- Define the lines and circles
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def O (x y : ℝ) : Prop := x^2 + y^2 = 4
def l₂ (x y : ℝ) : Prop := y - 2 = 4/3 * (x + 1)
def M_center_line (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the properties of circle M
def M (x y : ℝ) : Prop := (x - 8/3)^2 + (y - 4/3)^2 = 100/9

-- Theorem statement
theorem circle_intersection_and_tangent_line 
  (h₁ : ∀ x y, l₁ x y → l₂ x y → (x = -1 ∧ y = 2)) 
  (h₂ : ∃ x y, M x y ∧ M_center_line x y) 
  (h₃ : ∃ x y, M x y ∧ l₂ x y) 
  (h₄ : ∃ k, k > 0 ∧ ∀ x y, M x y → l₁ x y → 
    (∃ a₁ a₂, a₁ > 0 ∧ a₂ > 0 ∧ a₁ / a₂ = 2 ∧ a₁ + a₂ = 2 * π * k)) :
  (∃ x y, O x y ∧ l₁ x y ∧ 
    (∃ x' y', O x' y' ∧ l₁ x' y' ∧ (x - x')^2 + (y - y')^2 = 12)) ∧
  (∀ x y, M x y ↔ (x - 8/3)^2 + (y - 4/3)^2 = 100/9) :=
sorry

end circle_intersection_and_tangent_line_l3147_314765


namespace minimum_students_l3147_314793

theorem minimum_students (b g : ℕ) : 
  (3 * b = 8 * g) →  -- From the equation (3/4)b = 2(2/3)g simplified
  (b ≥ 1) →          -- At least one boy
  (g ≥ 1) →          -- At least one girl
  (∀ b' g', (3 * b' = 8 * g') → b' + g' ≥ b + g) →  -- Minimum condition
  b + g = 25 := by
sorry

end minimum_students_l3147_314793


namespace football_team_right_handed_players_l3147_314709

/-- Given a football team with the following properties:
  * There are 120 players in total
  * 62 players are throwers
  * Of the non-throwers, three-fifths are left-handed
  * All throwers are right-handed
  Prove that the total number of right-handed players is 86 -/
theorem football_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (non_throwers : ℕ) 
  (left_handed_non_throwers : ℕ) 
  (right_handed_non_throwers : ℕ) : 
  total_players = 120 →
  throwers = 62 →
  non_throwers = total_players - throwers →
  left_handed_non_throwers = (3 * non_throwers) / 5 →
  right_handed_non_throwers = non_throwers - left_handed_non_throwers →
  throwers + right_handed_non_throwers = 86 := by
  sorry

#check football_team_right_handed_players

end football_team_right_handed_players_l3147_314709


namespace largest_factorial_as_consecutive_product_l3147_314708

theorem largest_factorial_as_consecutive_product : 
  ∀ n : ℕ, n > 3 → ¬(∃ k : ℕ, n! = (k + 1) * (k + 2) * (k + 3) * (k + 4)) :=
by sorry

end largest_factorial_as_consecutive_product_l3147_314708


namespace square_binomial_plus_cube_problem_solution_l3147_314739

theorem square_binomial_plus_cube (a b : ℕ) : 
  a^2 + 2*a*b + b^2 + b^3 = (a + b)^2 + b^3 := by sorry

theorem problem_solution : 15^2 + 2*(15*5) + 5^2 + 5^3 = 525 := by
  have h1 : 15^2 + 2*(15*5) + 5^2 + 5^3 = (15 + 5)^2 + 5^3 := by
    exact square_binomial_plus_cube 15 5
  have h2 : (15 + 5)^2 = 400 := by norm_num
  have h3 : 5^3 = 125 := by norm_num
  calc
    15^2 + 2*(15*5) + 5^2 + 5^3 = (15 + 5)^2 + 5^3 := h1
    _ = 400 + 125 := by rw [h2, h3]
    _ = 525 := by norm_num

end square_binomial_plus_cube_problem_solution_l3147_314739


namespace st_length_is_135_14_l3147_314733

/-- Triangle PQR with given side lengths and a parallel line ST containing the incenter -/
structure SpecialTriangle where
  -- Define the triangle PQR
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  -- Define points S and T
  S : ℝ × ℝ
  T : ℝ × ℝ
  -- Conditions
  pq_length : PQ = 13
  pr_length : PR = 14
  qr_length : QR = 15
  s_on_pq : S.1 ≥ 0 ∧ S.1 ≤ PQ
  t_on_pr : T.2 ≥ 0 ∧ T.2 ≤ PR
  st_parallel_qr : sorry -- ST is parallel to QR
  st_contains_incenter : sorry -- ST contains the incenter of PQR

/-- The length of ST in the special triangle -/
def ST_length (triangle : SpecialTriangle) : ℝ := sorry

/-- Theorem stating that ST length is 135/14 -/
theorem st_length_is_135_14 (triangle : SpecialTriangle) : 
  ST_length triangle = 135 / 14 := by sorry

end st_length_is_135_14_l3147_314733


namespace f_properties_l3147_314713

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin x ^ 2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + Real.cos x ^ 2

theorem f_properties :
  (∀ x : ℝ, f x ≤ 5/4) ∧
  (∀ x : ℝ, f x = 5/4 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 6) ∧
  (∀ x : ℝ, f x = (1/2) * Real.sin (2 * x + Real.pi / 6) + 3/4) := by sorry

end f_properties_l3147_314713


namespace wire_service_reporters_l3147_314753

theorem wire_service_reporters (total : ℝ) (h1 : total > 0) : 
  let local_politics := 0.35 * total
  let not_politics := 0.5 * total
  let politics := total - not_politics
  let not_local_politics := politics - local_politics
  (not_local_politics / politics) * 100 = 30 := by
sorry

end wire_service_reporters_l3147_314753


namespace increasing_function_inequality_l3147_314728

-- Define an increasing function on ℝ
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem increasing_function_inequality (f : ℝ → ℝ) (m : ℝ) 
  (h_increasing : IncreasingFunction f) 
  (h_inequality : f (m + 1) > f (2 * m - 1)) : 
  m < 2 := by
  sorry

end increasing_function_inequality_l3147_314728


namespace fullTimeAndYearCount_l3147_314790

/-- Represents a company with employees. -/
structure Company where
  total : ℕ
  fullTime : ℕ
  atLeastYear : ℕ
  neitherFullTimeNorYear : ℕ

/-- The number of full-time employees who have worked at least a year. -/
def fullTimeAndYear (c : Company) : ℕ :=
  c.fullTime + c.atLeastYear - c.total + c.neitherFullTimeNorYear

/-- Theorem stating the number of full-time employees who have worked at least a year. -/
theorem fullTimeAndYearCount (c : Company) 
    (h1 : c.total = 130)
    (h2 : c.fullTime = 80)
    (h3 : c.atLeastYear = 100)
    (h4 : c.neitherFullTimeNorYear = 20) :
    fullTimeAndYear c = 70 := by
  sorry

end fullTimeAndYearCount_l3147_314790


namespace walking_distance_multiple_l3147_314789

/-- Prove that the multiple M is 4 given the walking distances of Rajesh and Hiro -/
theorem walking_distance_multiple (total_distance hiro_distance rajesh_distance : ℝ) 
  (h1 : total_distance = 25)
  (h2 : rajesh_distance = 18)
  (h3 : total_distance = hiro_distance + rajesh_distance)
  (h4 : ∃ M : ℝ, rajesh_distance = M * hiro_distance - 10) :
  ∃ M : ℝ, M = 4 ∧ rajesh_distance = M * hiro_distance - 10 := by
  sorry

end walking_distance_multiple_l3147_314789


namespace curve_is_semicircle_l3147_314746

-- Define the curve
def curve (x y : ℝ) : Prop := y - 1 = Real.sqrt (1 - x^2)

-- Define a semicircle
def is_semicircle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (c : ℝ × ℝ) (r : ℝ),
    r > 0 ∧
    S = {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 ∧ p.2 ≥ c.2}

-- Theorem statement
theorem curve_is_semicircle :
  is_semicircle {p : ℝ × ℝ | curve p.1 p.2} :=
sorry

end curve_is_semicircle_l3147_314746


namespace jasons_journey_l3147_314770

/-- The distance to Jason's home --/
def distance_to_home (total_time : ℝ) (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * (total_time - time1)

/-- Theorem statement for Jason's journey --/
theorem jasons_journey :
  let total_time : ℝ := 1.5
  let speed1 : ℝ := 60
  let time1 : ℝ := 0.5
  let speed2 : ℝ := 90
  distance_to_home total_time speed1 time1 speed2 = 120 := by
sorry

end jasons_journey_l3147_314770


namespace range_of_a_l3147_314716

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 4) → -3 ≤ a ∧ a ≤ 5 := by
  sorry

end range_of_a_l3147_314716


namespace candy_bar_cost_l3147_314717

def candy_sales (n : Nat) : Nat :=
  10 + 4 * (n - 1)

def total_candy_sales (days : Nat) : Nat :=
  (List.range days).map candy_sales |>.sum

theorem candy_bar_cost (days : Nat) (total_earnings : Rat) :
  days = 6 ∧ total_earnings = 12 →
  total_earnings / total_candy_sales days = 1/10 := by
  sorry

end candy_bar_cost_l3147_314717


namespace insect_eggs_base_conversion_l3147_314754

theorem insect_eggs_base_conversion : 
  (2 * 7^2 + 3 * 7^1 + 5 * 7^0 : ℕ) = 124 := by sorry

end insect_eggs_base_conversion_l3147_314754


namespace combined_value_l3147_314730

def i : ℕ := 2  -- The only prime even integer from 2 to √500

def k : ℕ := 44  -- Sum of even integers from 8 to √200 (8 + 10 + 12 + 14)

def j : ℕ := 23  -- Sum of prime odd integers from 5 to √133 (5 + 7 + 11)

theorem combined_value : 2 * i - k + 3 * j = 29 := by
  sorry

end combined_value_l3147_314730


namespace total_distance_calculation_l3147_314781

-- Define the distance walked per day
def distance_per_day : ℝ := 4.0

-- Define the number of days walked
def days_walked : ℝ := 3.0

-- Define the total distance walked
def total_distance : ℝ := distance_per_day * days_walked

-- Theorem statement
theorem total_distance_calculation :
  total_distance = 12.0 := by sorry

end total_distance_calculation_l3147_314781


namespace inverse_matrices_values_l3147_314744

theorem inverse_matrices_values (a b : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -9; a, 14]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![14, b; 5, 4]
  (A * B = 1 ∧ B * A = 1) → (a = -5 ∧ b = 9) :=
by sorry

end inverse_matrices_values_l3147_314744


namespace polynomial_simplification_l3147_314701

theorem polynomial_simplification (q : ℝ) : 
  (4 * q^3 - 7 * q^2 + 3 * q + 8) + (5 - 3 * q^3 + 9 * q^2 - 2 * q) = q^3 + 2 * q^2 + q + 13 := by
  sorry

end polynomial_simplification_l3147_314701


namespace second_class_size_l3147_314742

theorem second_class_size (students1 : ℕ) (avg1 : ℝ) (avg2 : ℝ) (avg_total : ℝ) :
  students1 = 30 →
  avg1 = 40 →
  avg2 = 90 →
  avg_total = 71.25 →
  ∃ students2 : ℕ, 
    (students1 * avg1 + students2 * avg2) / (students1 + students2 : ℝ) = avg_total ∧
    students2 = 50 :=
by sorry

end second_class_size_l3147_314742


namespace intersection_point_power_l3147_314785

theorem intersection_point_power (n : ℕ) (x₀ y₀ : ℝ) (hn : n ≥ 2) 
  (h1 : y₀^2 = n * x₀ - 1) (h2 : y₀ = x₀) :
  ∀ m : ℕ, m > 0 → ∃ k : ℕ, k ≥ 2 ∧ (x₀^m)^2 = k * (x₀^m) - 1 := by
  sorry

end intersection_point_power_l3147_314785


namespace smallest_k_no_real_roots_l3147_314771

theorem smallest_k_no_real_roots : 
  ∀ k : ℤ, (∀ x : ℝ, 2*x*(k*x-4)-x^2+7 ≠ 0) → k ≥ 2 :=
by sorry

end smallest_k_no_real_roots_l3147_314771


namespace unique_solution_x_power_x_power_x_eq_2_l3147_314775

theorem unique_solution_x_power_x_power_x_eq_2 :
  ∃! (x : ℝ), x > 0 ∧ x^(x^x) = 2 :=
by
  sorry

end unique_solution_x_power_x_power_x_eq_2_l3147_314775


namespace vertex_of_quadratic_l3147_314740

/-- The quadratic function f(x) = (x - 2)² - 3 -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 3

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (2, -3)

theorem vertex_of_quadratic :
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 := by
  sorry

end vertex_of_quadratic_l3147_314740


namespace candy_distribution_l3147_314732

theorem candy_distribution (total_candy : ℕ) (num_friends : ℕ) 
  (h1 : total_candy = 24) (h2 : num_friends = 5) :
  let pieces_to_remove := total_candy % num_friends
  let remaining_candy := total_candy - pieces_to_remove
  pieces_to_remove = 
    Nat.min pieces_to_remove (total_candy - (remaining_candy / num_friends) * num_friends) ∧
  (remaining_candy / num_friends) * num_friends = remaining_candy :=
by sorry

end candy_distribution_l3147_314732


namespace investment_decrease_l3147_314710

theorem investment_decrease (P : ℝ) (x : ℝ) 
  (h1 : P > 0)
  (h2 : 1.60 * P - (x / 100) * (1.60 * P) = 1.12 * P) :
  x = 30 := by
sorry

end investment_decrease_l3147_314710


namespace rope_contact_length_l3147_314783

/-- The length of rope in contact with a cylindrical tower, given specific conditions --/
theorem rope_contact_length 
  (rope_length : ℝ) 
  (tower_radius : ℝ) 
  (unicorn_height : ℝ) 
  (free_end_distance : ℝ) 
  (h1 : rope_length = 25) 
  (h2 : tower_radius = 10) 
  (h3 : unicorn_height = 3) 
  (h4 : free_end_distance = 5) : 
  ∃ (contact_length : ℝ), contact_length = rope_length - Real.sqrt 134 := by
  sorry

#check rope_contact_length

end rope_contact_length_l3147_314783


namespace frog_return_probability_l3147_314760

/-- Represents the probability of the frog being at position (x, y) after n hops -/
def prob (n : ℕ) (x y : ℕ) : ℚ :=
  sorry

/-- The grid size is 2x2 -/
def grid_size : ℕ := 2

/-- The number of hops the frog makes -/
def num_hops : ℕ := 3

/-- The probability of each possible movement (right, down, or stay) -/
def move_prob : ℚ := 1 / 3

theorem frog_return_probability :
  prob num_hops 0 0 = 1 / 27 :=
sorry

end frog_return_probability_l3147_314760


namespace mobius_gauss_formula_pentagon_l3147_314750

/-- Given a pentagon ABCDE with triangle areas α, β, γ, θ, δ, 
    the total area S satisfies the Möbius-Gauss formula. -/
theorem mobius_gauss_formula_pentagon (α β γ θ δ S : ℝ) 
  (h_positive : α > 0 ∧ β > 0 ∧ γ > 0 ∧ θ > 0 ∧ δ > 0) 
  (h_area : S > 0) 
  (h_sum : S > (α + β + γ + θ + δ) / 2) :
  S^2 - (α + β + γ + θ + δ) * S + (α*β + β*γ + γ*θ + θ*δ + δ*α) = 0 := by
  sorry

end mobius_gauss_formula_pentagon_l3147_314750


namespace arithmetic_geometric_ratio_l3147_314738

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def GeometricSequence (x y z : ℚ) : Prop :=
  y^2 = x * z

theorem arithmetic_geometric_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ArithmeticSequence a d)
  (h_geom : GeometricSequence (a 5) (a 9) (a 15)) :
  a 15 / a 9 = 3/2 := by
  sorry

end arithmetic_geometric_ratio_l3147_314738


namespace time_after_1550_minutes_l3147_314741

/-- Represents a time with day, hour, and minute components -/
structure DateTime where
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The starting DateTime (midnight on January 1, 2011) -/
def startTime : DateTime :=
  { day := 1, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 1550

/-- The expected result DateTime -/
def expectedResult : DateTime :=
  { day := 2, hour := 1, minute := 50 }

/-- Theorem stating that adding 1550 minutes to midnight on January 1
    results in 1:50 AM on January 2 -/
theorem time_after_1550_minutes :
  addMinutes startTime minutesToAdd = expectedResult := by
  sorry

end time_after_1550_minutes_l3147_314741


namespace square_difference_equality_l3147_314749

theorem square_difference_equality : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end square_difference_equality_l3147_314749


namespace new_xanadu_license_plates_l3147_314747

/-- Represents the number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- Represents the number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- Represents the number of possible first digits (1-9) -/
def num_first_digits : ℕ := 9

/-- Calculates the total number of valid license plates in New Xanadu -/
def num_valid_plates : ℕ :=
  num_letters ^ 3 * num_first_digits * num_digits ^ 2

/-- Theorem stating the total number of valid license plates in New Xanadu -/
theorem new_xanadu_license_plates :
  num_valid_plates = 15818400 := by
  sorry

end new_xanadu_license_plates_l3147_314747


namespace hyperbola_eccentricity_l3147_314764

/-- The eccentricity of a hyperbola with equation x^2 - y^2 = 1 is √2 -/
theorem hyperbola_eccentricity : 
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 - y^2 = 1}
  ∃ e : ℝ, e = Real.sqrt 2 ∧ 
    ∀ (a b c : ℝ), 
      (a = 1 ∧ b = 1 ∧ c^2 = a^2 + b^2) → 
      e = c / a :=
by sorry

end hyperbola_eccentricity_l3147_314764


namespace speed_limit_excess_l3147_314706

/-- Proves that a journey of 150 miles completed in 2 hours exceeds a 60 mph speed limit by 15 mph -/
theorem speed_limit_excess (distance : ℝ) (time : ℝ) (speed_limit : ℝ) : 
  distance = 150 ∧ time = 2 ∧ speed_limit = 60 →
  distance / time - speed_limit = 15 := by
sorry

end speed_limit_excess_l3147_314706


namespace smallest_with_12_divisors_l3147_314795

/-- A function that counts the number of positive integer divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 12 positive integer divisors -/
def has_12_divisors (n : ℕ) : Prop := count_divisors n = 12

/-- Theorem stating that 60 is the smallest positive integer with exactly 12 positive integer divisors -/
theorem smallest_with_12_divisors : 
  (has_12_divisors 60) ∧ (∀ m : ℕ, 0 < m ∧ m < 60 → ¬(has_12_divisors m)) :=
sorry

end smallest_with_12_divisors_l3147_314795


namespace greatest_k_for_inequality_l3147_314794

theorem greatest_k_for_inequality : 
  ∃ (k : ℤ), k = 5 ∧ 
  (∀ (j : ℤ), j > k → 
    ∃ (n : ℕ), n ≥ 2 ∧ ⌊n / Real.sqrt 3⌋ + 1 ≤ n^2 / Real.sqrt (3 * n^2 - j)) ∧
  (∀ (n : ℕ), n ≥ 2 → ⌊n / Real.sqrt 3⌋ + 1 > n^2 / Real.sqrt (3 * n^2 - k)) :=
by sorry

end greatest_k_for_inequality_l3147_314794


namespace candy_purchase_sum_l3147_314715

/-- A sequence of daily candy purchases where each day's purchase is one more than the previous day -/
def candy_sequence (first_day : ℕ) : ℕ → ℕ :=
  fun n => first_day + n - 1

theorem candy_purchase_sum (first_day : ℕ) 
  (h : candy_sequence first_day 0 + candy_sequence first_day 1 + candy_sequence first_day 2 = 504) :
  candy_sequence first_day 3 + candy_sequence first_day 4 + candy_sequence first_day 5 = 513 := by
  sorry

end candy_purchase_sum_l3147_314715


namespace canoe_kayak_difference_l3147_314707

/-- Represents the daily rental cost of a canoe -/
def canoe_cost : ℕ := 15

/-- Represents the daily rental cost of a kayak -/
def kayak_cost : ℕ := 18

/-- Represents the total daily revenue -/
def total_revenue : ℕ := 405

/-- Theorem stating that the difference between canoes and kayaks rented is 5 -/
theorem canoe_kayak_difference :
  ∀ (c k : ℕ),
  (c : ℚ) / k = 3 / 2 →
  canoe_cost * c + kayak_cost * k = total_revenue →
  c - k = 5 :=
by
  sorry

end canoe_kayak_difference_l3147_314707


namespace sum_equation_implies_N_value_l3147_314780

theorem sum_equation_implies_N_value :
  481 + 483 + 485 + 487 + 489 + 491 = 3000 - N → N = 84 := by
  sorry

end sum_equation_implies_N_value_l3147_314780


namespace geometric_series_ratio_l3147_314751

theorem geometric_series_ratio (a : ℝ) (r : ℝ) : 
  (a / (1 - r) = 20) → 
  (a / (1 - r^2) = 8) → 
  (r ≠ 1) →
  r = 3/2 := by
sorry

end geometric_series_ratio_l3147_314751


namespace quadratic_inequality_l3147_314786

theorem quadratic_inequality (y : ℝ) : y^2 - 9*y + 20 < 0 ↔ 4 < y ∧ y < 5 := by
  sorry

end quadratic_inequality_l3147_314786


namespace jerry_grocery_shopping_l3147_314755

/-- The amount of money Jerry has left after grocery shopping -/
def money_left (mustard_oil_price mustard_oil_quantity pasta_price pasta_quantity sauce_price sauce_quantity total_money : ℕ) : ℕ :=
  total_money - (mustard_oil_price * mustard_oil_quantity + pasta_price * pasta_quantity + sauce_price * sauce_quantity)

/-- Theorem stating that Jerry will have $7 left after grocery shopping -/
theorem jerry_grocery_shopping :
  money_left 13 2 4 3 5 1 50 = 7 := by
  sorry

end jerry_grocery_shopping_l3147_314755


namespace hexagon_side_length_equals_square_side_l3147_314796

/-- Represents a hexagon with side length y -/
structure Hexagon where
  y : ℝ

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with side length s -/
structure Square where
  s : ℝ

/-- Given a 12 × 12 rectangle divided into two congruent hexagons that can form a square without overlap,
    the side length of each hexagon is 12. -/
theorem hexagon_side_length_equals_square_side 
  (rect : Rectangle)
  (hex1 hex2 : Hexagon)
  (sq : Square)
  (h1 : rect.length = 12 ∧ rect.width = 12)
  (h2 : hex1 = hex2)
  (h3 : rect.length * rect.width = sq.s * sq.s)
  (h4 : hex1.y = sq.s) :
  hex1.y = 12 := by
  sorry

end hexagon_side_length_equals_square_side_l3147_314796


namespace ring_arrangements_count_l3147_314745

/-- The number of ways to arrange 6 out of 10 distinguishable rings on 4 fingers -/
def ring_arrangements : ℕ :=
  (Nat.choose 10 6) * (Nat.factorial 6) * (Nat.choose 9 3)

/-- Theorem stating the number of ring arrangements -/
theorem ring_arrangements_count : ring_arrangements = 12672000 := by
  sorry

end ring_arrangements_count_l3147_314745


namespace sports_conference_games_l3147_314797

/-- The number of games in a sports conference season --/
def conference_games (total_teams : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let divisions := total_teams / teams_per_division
  let intra_division_total := divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_division_total := (total_teams * (total_teams - teams_per_division) / 2) * inter_division_games
  intra_division_total + inter_division_total

theorem sports_conference_games :
  conference_games 16 8 3 2 = 296 := by
  sorry

end sports_conference_games_l3147_314797


namespace quadratic_equation_from_means_l3147_314734

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 5 → 
  Real.sqrt (a * b) = 15 → 
  ∃ (x : ℝ), x^2 - 10*x + 225 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end quadratic_equation_from_means_l3147_314734


namespace orchard_sections_l3147_314731

/-- The number of sections in an apple orchard --/
def number_of_sections (daily_harvest_per_section : ℕ) (total_daily_harvest : ℕ) : ℕ :=
  total_daily_harvest / daily_harvest_per_section

/-- Theorem stating that the number of sections in the orchard is 8 --/
theorem orchard_sections :
  let daily_harvest_per_section := 45
  let total_daily_harvest := 360
  number_of_sections daily_harvest_per_section total_daily_harvest = 8 := by
  sorry

end orchard_sections_l3147_314731


namespace mixed_selection_probability_l3147_314791

/-- Represents the number of volunteers from each grade -/
structure Volunteers where
  first_grade : ℕ
  second_grade : ℕ

/-- Represents the number of temporary leaders selected from each grade -/
structure Leaders where
  first_grade : ℕ
  second_grade : ℕ

/-- Calculates the number of leaders proportionally selected from each grade -/
def selectLeaders (v : Volunteers) : Leaders :=
  { first_grade := (5 * v.first_grade) / (v.first_grade + v.second_grade),
    second_grade := (5 * v.second_grade) / (v.first_grade + v.second_grade) }

/-- Calculates the probability of selecting one leader from each grade -/
def probabilityOfMixedSelection (l : Leaders) : ℚ :=
  (l.first_grade * l.second_grade : ℚ) / ((l.first_grade + l.second_grade) * (l.first_grade + l.second_grade - 1) / 2 : ℚ)

theorem mixed_selection_probability 
  (v : Volunteers) 
  (h1 : v.first_grade = 150) 
  (h2 : v.second_grade = 100) : 
  probabilityOfMixedSelection (selectLeaders v) = 3/5 := by
  sorry

end mixed_selection_probability_l3147_314791


namespace cereal_eating_time_l3147_314711

/-- The time it takes for three people to collectively eat a certain amount of cereal -/
def eating_time (fat_rate thin_rate medium_rate total_pounds : ℚ) : ℚ :=
  total_pounds / (1 / fat_rate + 1 / thin_rate + 1 / medium_rate)

/-- Theorem stating that given the eating rates of Mr. Fat, Mr. Thin, and Mr. Medium,
    the time required for them to collectively eat 5 pounds of cereal is 100/3 minutes -/
theorem cereal_eating_time :
  eating_time 20 30 15 5 = 100 / 3 := by
  sorry

end cereal_eating_time_l3147_314711


namespace lemonade_pitchers_l3147_314756

theorem lemonade_pitchers (glasses_per_pitcher : ℕ) (total_glasses : ℕ) (h1 : glasses_per_pitcher = 5) (h2 : total_glasses = 30) :
  total_glasses / glasses_per_pitcher = 6 := by
  sorry

end lemonade_pitchers_l3147_314756


namespace maria_savings_l3147_314784

/-- Calculates the amount left in Maria's savings after buying sweaters and scarves. -/
def amount_left (sweater_price scarf_price : ℕ) (num_sweaters num_scarves : ℕ) (initial_savings : ℕ) : ℕ :=
  initial_savings - (sweater_price * num_sweaters + scarf_price * num_scarves)

/-- Proves that Maria will have $200 left in her savings after buying sweaters and scarves. -/
theorem maria_savings : amount_left 30 20 6 6 500 = 200 := by
  sorry

end maria_savings_l3147_314784


namespace other_number_is_198_l3147_314774

/-- Given two positive integers with specific HCF and LCM, prove one is 198 when the other is 24 -/
theorem other_number_is_198 (a b : ℕ+) : 
  Nat.gcd a b = 12 → 
  Nat.lcm a b = 396 → 
  a = 24 → 
  b = 198 := by
sorry

end other_number_is_198_l3147_314774


namespace acute_angle_is_40_isosceles_trapezoid_acute_angle_l3147_314702

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The length of the diagonal -/
  diagonal : ℝ
  /-- The angle between the diagonal and the leg -/
  angle_diagonal_leg : ℝ
  /-- The angle between the diagonal and the base -/
  angle_diagonal_base : ℝ
  /-- The area is √3 -/
  area_eq : area = Real.sqrt 3
  /-- The diagonal length is 2 -/
  diagonal_eq : diagonal = 2
  /-- The angle between the diagonal and the base is 20° greater than the angle between the diagonal and the leg -/
  angle_relation : angle_diagonal_base = angle_diagonal_leg + 20

/-- The acute angle of the trapezoid is 40° -/
theorem acute_angle_is_40 (t : IsoscelesTrapezoid) : ℝ :=
  40

/-- The main theorem: proving that the acute angle of the trapezoid is 40° -/
theorem isosceles_trapezoid_acute_angle 
  (t : IsoscelesTrapezoid) : acute_angle_is_40 t = 40 := by
  sorry

end acute_angle_is_40_isosceles_trapezoid_acute_angle_l3147_314702


namespace goldfish_percentage_l3147_314777

theorem goldfish_percentage (surface : ℕ) (below : ℕ) : 
  surface = 15 → below = 45 → (surface : ℚ) / (surface + below : ℚ) * 100 = 25 := by
  sorry

end goldfish_percentage_l3147_314777


namespace sine_central_angle_is_zero_l3147_314782

/-- Represents a circle with intersecting chords -/
structure IntersectingChords where
  radius : ℝ
  pq_length : ℝ
  rt_length : ℝ

/-- The sine of the central angle subtending arc PR in the given circle configuration -/
def sine_central_angle (c : IntersectingChords) : ℝ :=
  sorry

/-- Theorem stating that the sine of the central angle is 0 for the given configuration -/
theorem sine_central_angle_is_zero (c : IntersectingChords) 
  (h1 : c.radius = 7)
  (h2 : c.pq_length = 14)
  (h3 : c.rt_length = 5) : 
  sine_central_angle c = 0 := by
  sorry

end sine_central_angle_is_zero_l3147_314782


namespace negation_of_proposition_l3147_314725

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x : ℝ, x ≥ 0 → x^2 - x ≥ 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 - x < 0) := by
  sorry

end negation_of_proposition_l3147_314725


namespace house_spacing_l3147_314743

/-- Given a city of length 11.5 km and 6 houses to be built at regular intervals
    including both ends, the distance between each house is 2.3 km. -/
theorem house_spacing (city_length : ℝ) (num_houses : ℕ) :
  city_length = 11.5 ∧ num_houses = 6 →
  (city_length / (num_houses - 1 : ℝ)) = 2.3 := by
  sorry

end house_spacing_l3147_314743


namespace third_function_symmetry_l3147_314703

-- Define the type for our functions
def RealFunction := ℝ → ℝ

-- State the theorem
theorem third_function_symmetry 
  (f : RealFunction) 
  (f_inv : RealFunction) 
  (h : RealFunction) 
  (h1 : ∀ x, f_inv (f x) = x) -- f_inv is the inverse of f
  (h2 : ∀ x y, h y = -x ↔ f_inv (-y) = -x) -- h is symmetric to f_inv w.r.t. x + y = 0
  : ∀ x, h x = -f (-x) := by
  sorry


end third_function_symmetry_l3147_314703


namespace max_pons_is_eleven_l3147_314787

/-- Represents the number of items purchased -/
structure Purchase where
  pans : ℕ
  pins : ℕ
  pons : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  3 * p.pans + 5 * p.pins + 8 * p.pons

/-- Checks if a purchase is valid (at least one of each item and total cost is $100) -/
def isValidPurchase (p : Purchase) : Prop :=
  p.pans ≥ 1 ∧ p.pins ≥ 1 ∧ p.pons ≥ 1 ∧ totalCost p = 100

/-- Theorem: The maximum number of pons in a valid purchase is 11 -/
theorem max_pons_is_eleven :
  ∀ p : Purchase, isValidPurchase p → p.pons ≤ 11 :=
by sorry

end max_pons_is_eleven_l3147_314787


namespace average_increase_l3147_314758

theorem average_increase (initial_average : ℝ) (fourth_test_score : ℝ) :
  initial_average = 81 ∧ fourth_test_score = 89 →
  (3 * initial_average + fourth_test_score) / 4 = initial_average + 2 := by
  sorry

end average_increase_l3147_314758


namespace slipper_equation_l3147_314776

-- Define the original price of slippers
variable (x : ℝ)

-- Define the amount spent before and during the sale
def before_sale : ℝ := 120
def during_sale : ℝ := 100

-- Define the price reduction during the sale
def price_reduction : ℝ := 5

-- Define the additional pairs bought during the sale
def additional_pairs : ℕ := 2

-- Theorem stating the equation that represents the situation
theorem slipper_equation :
  before_sale / x = during_sale / (x - price_reduction) - additional_pairs :=
sorry

end slipper_equation_l3147_314776


namespace election_winner_votes_l3147_314729

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  winner_percentage = 62 / 100 →
  vote_difference = 324 →
  (winner_percentage * total_votes).num = total_votes * winner_percentage.num →
  (winner_percentage * total_votes).num - ((1 - winner_percentage) * total_votes).num = vote_difference →
  (winner_percentage * total_votes).num = 837 := by
  sorry

end election_winner_votes_l3147_314729


namespace zero_exponent_equals_one_l3147_314769

theorem zero_exponent_equals_one (r : ℚ) (h : r ≠ 0) : r ^ 0 = 1 := by
  sorry

end zero_exponent_equals_one_l3147_314769


namespace chess_game_draw_fraction_l3147_314727

theorem chess_game_draw_fraction 
  (ellen_wins : ℚ) 
  (john_wins : ℚ) 
  (h1 : ellen_wins = 4/9) 
  (h2 : john_wins = 2/9) : 
  1 - (ellen_wins + john_wins) = 1/3 :=
by sorry

end chess_game_draw_fraction_l3147_314727


namespace p_and_not_q_is_true_l3147_314779

/-- Proposition p: There exists a real number x such that x - 2 > log_10(x) -/
def p : Prop := ∃ x : ℝ, x - 2 > Real.log x / Real.log 10

/-- Proposition q: For all real numbers x, x^2 > 0 -/
def q : Prop := ∀ x : ℝ, x^2 > 0

/-- Theorem: The conjunction of p and (not q) is true -/
theorem p_and_not_q_is_true : p ∧ ¬q := by
  sorry

end p_and_not_q_is_true_l3147_314779


namespace conference_center_occupancy_l3147_314718

def room_capacities : List ℕ := [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320]

def occupancy_rates : List ℚ := [3/4, 5/6, 2/3, 3/5, 4/9, 11/15, 7/10, 1/2, 5/8, 9/14, 8/15, 17/20]

theorem conference_center_occupancy :
  let occupied_rooms := List.zip room_capacities occupancy_rates
  let total_people := occupied_rooms.map (λ (cap, rate) => (cap : ℚ) * rate)
  ⌊total_people.sum⌋ = 1639 := by sorry

end conference_center_occupancy_l3147_314718


namespace diamond_calculation_l3147_314723

/-- The diamond operation for real numbers -/
def diamond (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Theorem stating that 2 ◇ (3 ◇ (1 ◇ 4)) = -46652 -/
theorem diamond_calculation : diamond 2 (diamond 3 (diamond 1 4)) = -46652 := by
  sorry

end diamond_calculation_l3147_314723


namespace car_mileage_l3147_314778

/-- Given a car that travels 200 kilometers using 5 gallons of gasoline, its mileage is 40 kilometers per gallon. -/
theorem car_mileage (distance : ℝ) (gasoline : ℝ) (h1 : distance = 200) (h2 : gasoline = 5) :
  distance / gasoline = 40 := by
  sorry

end car_mileage_l3147_314778


namespace triangle_properties_l3147_314799

/-- Given a triangle ABC with angle C = π/4 and the relation 2sin²A - 1 = sin²B,
    prove that tan B = 2 and if side b = 1, the area of the triangle is 3/8 -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  C = π/4 →
  2 * Real.sin A ^ 2 - 1 = Real.sin B ^ 2 →
  Real.tan B = 2 ∧
  (b = 1 → (1/2) * a * b * Real.sin C = 3/8) :=
by sorry

end triangle_properties_l3147_314799


namespace min_value_of_sum_l3147_314736

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  (3 / a + 2 / b) ≥ 25 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 1 ∧ 3 / a₀ + 2 / b₀ = 25 := by
  sorry

end min_value_of_sum_l3147_314736


namespace smallest_cycle_length_cycle_length_21_smallest_b_is_21_l3147_314772

def g (x : ℤ) : ℤ :=
  if x % 5 = 0 ∧ x % 7 = 0 then x / 35
  else if x % 7 = 0 then 5 * x
  else if x % 5 = 0 then 7 * x
  else x + 5

def g_iterate (n : ℕ) (x : ℤ) : ℤ :=
  match n with
  | 0 => x
  | n + 1 => g (g_iterate n x)

theorem smallest_cycle_length :
  ∀ b : ℕ, b > 1 → g_iterate b 3 = g 3 → b ≥ 21 :=
by sorry

theorem cycle_length_21 : g_iterate 21 3 = g 3 :=
by sorry

theorem smallest_b_is_21 :
  ∃! b : ℕ, b > 1 ∧ g_iterate b 3 = g 3 ∧ ∀ k : ℕ, k > 1 → g_iterate k 3 = g 3 → k ≥ b :=
by sorry

end smallest_cycle_length_cycle_length_21_smallest_b_is_21_l3147_314772


namespace crab_fishing_income_l3147_314705

theorem crab_fishing_income 
  (num_buckets : ℕ) 
  (crabs_per_bucket : ℕ) 
  (price_per_crab : ℕ) 
  (days_per_week : ℕ) 
  (h1 : num_buckets = 8) 
  (h2 : crabs_per_bucket = 12) 
  (h3 : price_per_crab = 5) 
  (h4 : days_per_week = 7) : 
  num_buckets * crabs_per_bucket * price_per_crab * days_per_week = 3360 := by
sorry

end crab_fishing_income_l3147_314705


namespace upstream_downstream_time_difference_l3147_314735

/-- Proves that the difference in time between traveling upstream and downstream is 90 minutes -/
theorem upstream_downstream_time_difference 
  (distance : ℝ) 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : distance = 36) 
  (h2 : boat_speed = 10) 
  (h3 : stream_speed = 2) : 
  (distance / (boat_speed - stream_speed) - distance / (boat_speed + stream_speed)) * 60 = 90 := by
  sorry

#check upstream_downstream_time_difference

end upstream_downstream_time_difference_l3147_314735


namespace inequality_and_fraction_analysis_l3147_314763

theorem inequality_and_fraction_analysis (a : ℝ) (h : 2 < a ∧ a < 4) :
  (3 * a - 2 > 2 * a ∧ 4 * (a - 1) < 3 * a) ∧
  (a - (a + 4) / (a + 1) = (a^2 - 4) / (a + 1)) ∧
  ((a^2 - 4) / (a + 1) ≥ 0 ↔ a ≥ 2) := by
  sorry

end inequality_and_fraction_analysis_l3147_314763


namespace nested_fraction_equality_l3147_314759

theorem nested_fraction_equality : 1 + (1 / (1 + (1 / (1 + (1 / 2))))) = 8 / 5 := by
  sorry

end nested_fraction_equality_l3147_314759


namespace product_of_fractions_l3147_314752

theorem product_of_fractions : (1 : ℚ) / 3 * 2 / 5 * 3 / 7 * 4 / 8 = 1 / 35 := by
  sorry

end product_of_fractions_l3147_314752


namespace inverse_g_sum_l3147_314720

-- Define the function g
def g (x : ℝ) : ℝ := x * |x|^3

-- State the theorem
theorem inverse_g_sum : g⁻¹ 8 + g⁻¹ (-64) = -2 := by
  sorry

end inverse_g_sum_l3147_314720


namespace equation_solutions_l3147_314768

theorem equation_solutions : 
  ∀ x y z : ℕ+, 
    (x.val * y.val + y.val * z.val + z.val * x.val - x.val * y.val * z.val = 2) ↔ 
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨ 
     (x = 2 ∧ y = 3 ∧ z = 4) ∨ (x = 2 ∧ y = 4 ∧ z = 3) ∨
     (x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 3 ∧ y = 4 ∧ z = 2) ∨
     (x = 4 ∧ y = 2 ∧ z = 3) ∨ (x = 4 ∧ y = 3 ∧ z = 2)) := by
  sorry

end equation_solutions_l3147_314768


namespace terminal_side_angle_theorem_l3147_314714

theorem terminal_side_angle_theorem (α : Real) :
  (∃ (x y : Real), x = -2 ∧ y = 1 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  1 / Real.sin (2 * α) = -5/4 := by
sorry

end terminal_side_angle_theorem_l3147_314714


namespace exponent_calculation_l3147_314704

theorem exponent_calculation : (-4)^6 / 4^4 + 2^5 - 7^2 = -1 := by sorry

end exponent_calculation_l3147_314704


namespace circle_equation_k_value_l3147_314762

/-- 
Given an equation x^2 + 8x + y^2 + 4y - k = 0 that represents a circle with radius 7,
prove that k = 29.
-/
theorem circle_equation_k_value (k : ℝ) :
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 4*y - k = 0 ↔ (x + 4)^2 + (y + 2)^2 = 7^2) →
  k = 29 :=
by sorry

end circle_equation_k_value_l3147_314762


namespace reflection_line_equation_l3147_314767

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  p : Point
  q : Point
  r : Point

/-- The reflection of a triangle -/
structure ReflectedTriangle where
  p' : Point
  q' : Point
  r' : Point

/-- The line of reflection -/
structure ReflectionLine where
  equation : ℝ → Prop

/-- Theorem: Given a triangle and its reflection, prove the equation of the reflection line -/
theorem reflection_line_equation 
  (t : Triangle) 
  (rt : ReflectedTriangle) 
  (h1 : t.p = ⟨2, 2⟩) 
  (h2 : t.q = ⟨6, 6⟩) 
  (h3 : t.r = ⟨-3, 5⟩)
  (h4 : rt.p' = ⟨2, -4⟩) 
  (h5 : rt.q' = ⟨6, -8⟩) 
  (h6 : rt.r' = ⟨-3, -7⟩) :
  ∃ (l : ReflectionLine), l.equation = λ y => y = -1 := by
  sorry

end reflection_line_equation_l3147_314767


namespace ABD_collinear_l3147_314726

/-- Given vectors in 2D space -/
def a : ℝ × ℝ := sorry
def b : ℝ × ℝ := sorry

/-- Define vectors AB, BC, and CD -/
def AB : ℝ × ℝ := a + 5 • b
def BC : ℝ × ℝ := -2 • a + 8 • b
def CD : ℝ × ℝ := 3 • (a - b)

/-- Define points A, B, C, and D -/
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := A + AB
def C : ℝ × ℝ := B + BC
def D : ℝ × ℝ := C + CD

/-- Theorem: Points A, B, and D are collinear -/
theorem ABD_collinear : ∃ (t : ℝ), D = A + t • (B - A) := by sorry

end ABD_collinear_l3147_314726


namespace square_sum_equals_one_l3147_314766

theorem square_sum_equals_one (a b : ℝ) 
  (h1 : a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) = 1)
  (h2 : 0 ≤ a ∧ a ≤ 1)
  (h3 : 0 ≤ b ∧ b ≤ 1) : 
  a^2 + b^2 = 1 := by
  sorry

end square_sum_equals_one_l3147_314766


namespace base9_to_base10_3956_l3147_314719

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (a b c d : ℕ) : ℕ :=
  a * 9^3 + b * 9^2 + c * 9^1 + d * 9^0

/-- Theorem: The base-9 number 3956₉ is equal to 2967 in base-10 --/
theorem base9_to_base10_3956 : base9ToBase10 3 9 5 6 = 2967 := by
  sorry

end base9_to_base10_3956_l3147_314719


namespace min_value_of_quadratic_l3147_314724

/-- The function f(x) = 3x^2 - 18x + 2023 has a minimum value of 1996. -/
theorem min_value_of_quadratic :
  ∃ (m : ℝ), m = 1996 ∧ ∀ x : ℝ, 3 * x^2 - 18 * x + 2023 ≥ m :=
by sorry

end min_value_of_quadratic_l3147_314724


namespace cookies_per_batch_l3147_314722

/-- Proves that each batch of cookies must produce 4 dozens to meet the required total --/
theorem cookies_per_batch 
  (classmates : ℕ) 
  (cookies_per_student : ℕ) 
  (total_batches : ℕ) 
  (h1 : classmates = 24) 
  (h2 : cookies_per_student = 10) 
  (h3 : total_batches = 5) : 
  (classmates * cookies_per_student) / (total_batches * 12) = 4 := by
  sorry

#check cookies_per_batch

end cookies_per_batch_l3147_314722


namespace ap_terms_count_l3147_314748

theorem ap_terms_count (n : ℕ) (a d : ℚ) : 
  Even n →
  n / 2 * (2 * a + (n - 2) * d) = 18 →
  n / 2 * (2 * a + 2 * d + (n - 2) * d) = 36 →
  a + (n - 1) * d - a = 7 →
  n = 12 :=
by sorry

end ap_terms_count_l3147_314748


namespace base_r_palindrome_square_l3147_314761

theorem base_r_palindrome_square (r : ℕ) (h1 : r % 2 = 0) (h2 : r ≥ 18) : 
  let x := 5*r^3 + 5*r^2 + 5*r + 5
  let squared := x^2
  let a := squared / r^7 % r
  let b := squared / r^6 % r
  let c := squared / r^5 % r
  let d := squared / r^4 % r
  (squared = a*r^7 + b*r^6 + c*r^5 + d*r^4 + d*r^3 + c*r^2 + b*r + a) ∧ 
  (d - c = 2) →
  r = 24 := by sorry

end base_r_palindrome_square_l3147_314761


namespace initial_birds_count_l3147_314712

/-- The number of birds initially on the fence -/
def initial_birds : ℕ := sorry

/-- The number of birds that landed on the fence -/
def landed_birds : ℕ := 8

/-- The total number of birds after more birds landed -/
def total_birds : ℕ := 20

/-- Theorem stating that the initial number of birds is 12 -/
theorem initial_birds_count : initial_birds = 12 := by
  sorry

end initial_birds_count_l3147_314712
