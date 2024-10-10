import Mathlib

namespace excellent_students_increase_l1776_177614

theorem excellent_students_increase (total_students : ℕ) 
  (first_semester_percent : ℚ) (second_semester_percent : ℚ) :
  total_students = 650 →
  first_semester_percent = 70 / 100 →
  second_semester_percent = 80 / 100 →
  ⌈(second_semester_percent - first_semester_percent) * total_students⌉ = 65 := by
  sorry

end excellent_students_increase_l1776_177614


namespace gcd_of_three_numbers_l1776_177607

theorem gcd_of_three_numbers : Nat.gcd 18222 (Nat.gcd 24546 66364) = 2 := by
  sorry

end gcd_of_three_numbers_l1776_177607


namespace unique_solution_phi_sigma_pow_two_l1776_177658

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

/-- Sum of divisors function -/
def sigma : ℕ → ℕ := sorry

/-- The equation φ(σ(2^x)) = 2^x has only one solution in the natural numbers, and that solution is x = 1 -/
theorem unique_solution_phi_sigma_pow_two : 
  ∃! x : ℕ, phi (sigma (2^x)) = 2^x ∧ x = 1 := by sorry

end unique_solution_phi_sigma_pow_two_l1776_177658


namespace min_value_a_plus_2b_l1776_177627

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → a + 2*b ≤ x + 2*y :=
by sorry

end min_value_a_plus_2b_l1776_177627


namespace digit_97_of_1_13_l1776_177670

/-- The decimal representation of 1/13 -/
def decimal_rep_1_13 : ℕ → Fin 10
  | n => match n % 6 with
    | 0 => 0
    | 1 => 7
    | 2 => 6
    | 3 => 9
    | 4 => 2
    | 5 => 3
    | _ => 0  -- This case should never occur due to % 6

/-- The 97th digit after the decimal point in the decimal representation of 1/13 is 0 -/
theorem digit_97_of_1_13 : decimal_rep_1_13 97 = 0 := by
  sorry

end digit_97_of_1_13_l1776_177670


namespace kaylin_age_l1776_177680

-- Define variables for each person's age
variable (kaylin sarah eli freyja alfred olivia : ℝ)

-- State the conditions
axiom kaylin_sarah : kaylin = sarah - 5
axiom sarah_eli : sarah = 2 * eli
axiom eli_freyja : eli = freyja + 9
axiom freyja_alfred : freyja = 2.5 * alfred
axiom alfred_olivia : alfred = 0.75 * olivia
axiom freyja_age : freyja = 9.5

-- Theorem to prove
theorem kaylin_age : kaylin = 32 := by
  sorry

end kaylin_age_l1776_177680


namespace base_conversion_1729_l1776_177679

theorem base_conversion_1729 :
  (5 * 7^3 + 0 * 7^2 + 2 * 7^1 + 0 * 7^0 : ℕ) = 1729 := by
  sorry

end base_conversion_1729_l1776_177679


namespace tan_three_expression_zero_l1776_177699

theorem tan_three_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 + Real.cos θ) / Real.sin θ - Real.sin θ / (1 - Real.cos θ) = 0 := by
  sorry

end tan_three_expression_zero_l1776_177699


namespace frog_probability_l1776_177681

/-- Represents a position on the 4x4 grid -/
inductive Position
| Center : Position
| Interior : Position
| Edge : Position

/-- Represents the number of hops -/
def MaxHops : Nat := 5

/-- The probability of reaching an edge from a given position after n hops -/
noncomputable def probability (pos : Position) (n : Nat) : Real :=
  match pos, n with
  | Position.Edge, _ => 1
  | _, 0 => 0
  | Position.Center, n + 1 => 
      (1/4) * (probability Position.Interior n + probability Position.Interior n + 
               probability Position.Edge n + probability Position.Edge n)
  | Position.Interior, n + 1 => 
      (1/4) * (probability Position.Interior n + probability Position.Interior n + 
               probability Position.Edge n + probability Position.Edge n)

/-- The main theorem to be proved -/
theorem frog_probability : 
  probability Position.Center MaxHops = 121/128 := by
  sorry

end frog_probability_l1776_177681


namespace power_function_m_value_l1776_177677

/-- A function f is a power function if it can be expressed as f(x) = x^n for some constant n. -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, ∀ x : ℝ, f x = x^n

/-- The given function parameterized by m. -/
def f (m : ℝ) (x : ℝ) : ℝ := (2*m - m^2) * x^3

theorem power_function_m_value :
  ∃! m : ℝ, IsPowerFunction (f m) ∧ m = 1 :=
sorry

end power_function_m_value_l1776_177677


namespace equation_solution_l1776_177651

theorem equation_solution (x : ℝ) : 1 + 1 / (1 + x) = 2 / (1 + x) → x = 0 := by
  sorry

end equation_solution_l1776_177651


namespace find_c_l1776_177664

def f (a c x : ℝ) : ℝ := a * x^3 + c

theorem find_c (a c : ℝ) :
  (∃ x, x ∈ Set.Icc 1 2 ∧ ∀ y ∈ Set.Icc 1 2, f a c y ≤ f a c x) →
  (deriv (f a c) 1 = 6) →
  (∃ x, x ∈ Set.Icc 1 2 ∧ f a c x = 20) →
  c = 4 := by
sorry

end find_c_l1776_177664


namespace hexagon_circle_area_ratio_l1776_177663

theorem hexagon_circle_area_ratio (r : ℝ) (h : r > 0) :
  (3 * Real.sqrt 3 * r^2 / 2) / (π * r^2) = 3 * Real.sqrt 3 / (2 * π) := by
  sorry

end hexagon_circle_area_ratio_l1776_177663


namespace train_length_l1776_177615

/-- Given a train that crosses two platforms of different lengths at different times, 
    this theorem proves the length of the train. -/
theorem train_length 
  (platform1_length : ℝ) 
  (platform1_time : ℝ) 
  (platform2_length : ℝ) 
  (platform2_time : ℝ) 
  (h1 : platform1_length = 120)
  (h2 : platform1_time = 15)
  (h3 : platform2_length = 250)
  (h4 : platform2_time = 20) :
  ∃ train_length : ℝ, 
    (train_length + platform1_length) / platform1_time = 
    (train_length + platform2_length) / platform2_time ∧ 
    train_length = 270 := by
  sorry

end train_length_l1776_177615


namespace min_value_cos_sin_l1776_177682

theorem min_value_cos_sin (θ : Real) (h : π/2 < θ ∧ θ < 3*π/2) :
  ∃ (min_val : Real), min_val = Real.sqrt 3 / 2 - 3 / 4 ∧
  ∀ (y : Real), y = Real.cos (θ/2) * (1 - Real.sin θ) → y ≥ min_val :=
sorry

end min_value_cos_sin_l1776_177682


namespace last_digit_101_power_100_l1776_177659

theorem last_digit_101_power_100 : 101^100 ≡ 1 [ZMOD 10] := by sorry

end last_digit_101_power_100_l1776_177659


namespace burglars_money_min_burglars_money_l1776_177635

def x (a n : ℕ) : ℚ := (a / 4 : ℚ) * (1 - (1 / 3 : ℚ) ^ n)

theorem burglars_money (a : ℕ) : 
  (∀ n : ℕ, n ≤ 2012 → (x a n).num % (x a n).den = 0 ∧ ((a : ℚ) - x a n).num % ((a : ℚ) - x a n).den = 0) →
  a ≥ 4 * 3^2012 :=
sorry

theorem min_burglars_money : 
  ∃ a : ℕ, a = 4 * 3^2012 ∧ 
  (∀ n : ℕ, n ≤ 2012 → (x a n).num % (x a n).den = 0 ∧ ((a : ℚ) - x a n).num % ((a : ℚ) - x a n).den = 0) ∧
  (∀ b : ℕ, b < a → ∃ n : ℕ, n ≤ 2012 ∧ ((x b n).num % (x b n).den ≠ 0 ∨ ((b : ℚ) - x b n).num % ((b : ℚ) - x b n).den ≠ 0)) :=
sorry

end burglars_money_min_burglars_money_l1776_177635


namespace abc_inequalities_l1776_177655

theorem abc_inequalities (a b : Real) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a + b = 1) : 
  (2 * a^2 + b ≥ 7/8) ∧ 
  (a * b ≤ 1/4) ∧ 
  (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2) := by
  sorry

end abc_inequalities_l1776_177655


namespace infinitely_many_n_not_equal_l1776_177684

/-- For any positive integers a and b greater than 1, there are infinitely many n
    such that φ(a^n - 1) ≠ b^m - b^t for any positive integers m and t. -/
theorem infinitely_many_n_not_equal (a b : ℕ) (ha : a > 1) (hb : b > 1) :
  Set.Infinite {n : ℕ | ∀ m t : ℕ, m > 0 → t > 0 → Nat.totient (a^n - 1) ≠ b^m - b^t} :=
sorry

end infinitely_many_n_not_equal_l1776_177684


namespace arctan_tan_difference_l1776_177625

theorem arctan_tan_difference (θ : Real) : 
  0 ≤ θ ∧ θ ≤ 180 ∧ 
  θ = Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (15 * π / 180)) * 180 / π :=
by sorry

end arctan_tan_difference_l1776_177625


namespace expression_value_l1776_177688

theorem expression_value (a b c d : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) : 
  (a + b + c * d) + (a + b) / (c * d) = 1 := by
sorry

end expression_value_l1776_177688


namespace chess_piece_paths_l1776_177638

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem chess_piece_paths :
  let num_segments : ℕ := 15
  let steps_per_segment : ℕ := 6
  let ways_per_segment : ℕ := fibonacci (steps_per_segment + 1)
  num_segments * ways_per_segment = 195 :=
by sorry

end chess_piece_paths_l1776_177638


namespace min_pool_cost_l1776_177695

def pool_volume : ℝ := 18
def pool_depth : ℝ := 2
def bottom_cost_per_sqm : ℝ := 200
def wall_cost_per_sqm : ℝ := 150

theorem min_pool_cost :
  let length : ℝ → ℝ → ℝ := λ x y => x
  let width : ℝ → ℝ → ℝ := λ x y => y
  let volume : ℝ → ℝ → ℝ := λ x y => x * y * pool_depth
  let bottom_area : ℝ → ℝ → ℝ := λ x y => x * y
  let wall_area : ℝ → ℝ → ℝ := λ x y => 2 * (x + y) * pool_depth
  let total_cost : ℝ → ℝ → ℝ := λ x y => 
    bottom_cost_per_sqm * bottom_area x y + wall_cost_per_sqm * wall_area x y
  ∃ x y : ℝ, 
    volume x y = pool_volume ∧ 
    (∀ a b : ℝ, volume a b = pool_volume → total_cost x y ≤ total_cost a b) ∧
    total_cost x y = 5400 :=
by sorry

end min_pool_cost_l1776_177695


namespace max_area_PCD_l1776_177675

/-- Definition of the ellipse Γ -/
def Γ (a b x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

/-- Definition of point A (left vertex) -/
def A (a : ℝ) : ℝ × ℝ := (-a, 0)

/-- Definition of point B (top vertex) -/
def B (b : ℝ) : ℝ × ℝ := (0, b)

/-- Definition of point P on the ellipse in the fourth quadrant -/
def P (a b : ℝ) : {p : ℝ × ℝ // Γ a b p.1 p.2 ∧ p.1 > 0 ∧ p.2 < 0} := sorry

/-- Definition of point C (intersection of PA with y-axis) -/
def C (a b : ℝ) : ℝ × ℝ := sorry

/-- Definition of point D (intersection of PB with x-axis) -/
def D (a b : ℝ) : ℝ × ℝ := sorry

/-- Area of triangle PCD -/
def area_PCD (a b : ℝ) : ℝ := sorry

/-- Theorem stating the maximum area of triangle PCD -/
theorem max_area_PCD (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ (max_area : ℝ), max_area = (Real.sqrt 2 - 1) / 2 * a * b ∧
    ∀ (p : ℝ × ℝ), Γ a b p.1 p.2 → p.1 > 0 → p.2 < 0 →
      area_PCD a b ≤ max_area :=
sorry

end max_area_PCD_l1776_177675


namespace equation_solutions_l1776_177667

theorem equation_solutions :
  (∀ x : ℝ, (3 * x - 1)^2 = 9 ↔ x = 4/3 ∨ x = -2/3) ∧
  (∀ x : ℝ, x * (2 * x - 4) = (2 - x)^2 ↔ x = 2 ∨ x = -2) :=
by sorry

end equation_solutions_l1776_177667


namespace triangle_perimeter_l1776_177617

/-- Given a triangle with inradius 2.5 cm and area 45 cm², its perimeter is 36 cm. -/
theorem triangle_perimeter (inradius : ℝ) (area : ℝ) (perimeter : ℝ) : 
  inradius = 2.5 → area = 45 → perimeter = 36 := by
  sorry

#check triangle_perimeter

end triangle_perimeter_l1776_177617


namespace eighth_term_of_geometric_sequence_l1776_177602

/-- Given a geometric sequence with first term 12 and second term 4,
    prove that its eighth term is 4/729. -/
theorem eighth_term_of_geometric_sequence : 
  ∀ (a : ℕ → ℚ), 
    (∀ n, a (n + 2) * a n = (a (n + 1))^2) →  -- geometric sequence condition
    a 1 = 12 →                                -- first term
    a 2 = 4 →                                 -- second term
    a 8 = 4/729 := by
  sorry

end eighth_term_of_geometric_sequence_l1776_177602


namespace three_white_marbles_probability_l1776_177611

def total_marbles : ℕ := 5 + 7 + 15

def probability_three_white (red green white : ℕ) : ℚ :=
  (white / total_marbles) * 
  ((white - 1) / (total_marbles - 1)) * 
  ((white - 2) / (total_marbles - 2))

theorem three_white_marbles_probability :
  probability_three_white 5 7 15 = 2 / 13 := by
  sorry

end three_white_marbles_probability_l1776_177611


namespace min_value_theorem_l1776_177673

theorem min_value_theorem (x y : ℝ) (h : x + y = 5) :
  ∃ m : ℝ, m = (6100 : ℝ) / 17 ∧ 
  ∀ z : ℝ, z ≥ m ∧ ∃ a b : ℝ, a + b = 5 ∧ 
  z = a^5*b + a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 + 2 :=
sorry

end min_value_theorem_l1776_177673


namespace borrowed_amount_proof_l1776_177662

/-- Represents the simple interest calculation for a loan -/
structure LoanInfo where
  principal : ℝ
  rate : ℝ
  time : ℝ
  total_amount : ℝ

/-- Theorem stating that given the loan conditions, the principal amount is 5400 -/
theorem borrowed_amount_proof (loan : LoanInfo) 
  (h1 : loan.rate = 0.06)
  (h2 : loan.time = 9)
  (h3 : loan.total_amount = 8310)
  : loan.principal = 5400 := by
  sorry

#check borrowed_amount_proof

end borrowed_amount_proof_l1776_177662


namespace sophias_age_problem_l1776_177693

/-- Sophia's age problem -/
theorem sophias_age_problem (S M : ℝ) (h1 : S > 0) (h2 : M > 0) 
  (h3 : ∃ (x : ℝ), S = 3 * x ∧ x > 0)  -- S is thrice the sum of children's ages
  (h4 : S - M = 4 * ((S / 3) - 2 * M)) :  -- Condition about age M years ago
  S / M = 21 := by
sorry

end sophias_age_problem_l1776_177693


namespace gravel_cost_theorem_l1776_177624

/-- The cost of gravel in dollars per cubic foot -/
def gravel_cost_per_cubic_foot : ℝ := 4

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of gravel in cubic yards -/
def gravel_volume_cubic_yards : ℝ := 8

/-- Theorem stating the cost of 8 cubic yards of gravel -/
theorem gravel_cost_theorem :
  gravel_cost_per_cubic_foot * cubic_yards_to_cubic_feet * gravel_volume_cubic_yards = 864 := by
  sorry

end gravel_cost_theorem_l1776_177624


namespace equation_solution_l1776_177642

theorem equation_solution : 
  {x : ℝ | (Real.sqrt (9*x - 2) + 15 / Real.sqrt (9*x - 2) = 8)} = {3, 11/9} :=
by sorry

end equation_solution_l1776_177642


namespace x_equals_y_when_q_is_seven_l1776_177606

theorem x_equals_y_when_q_is_seven :
  ∀ (q : ℤ), 
  let x := 55 + 2 * q
  let y := 4 * q + 41
  q = 7 → x = y :=
by
  sorry

end x_equals_y_when_q_is_seven_l1776_177606


namespace team_selection_count_l1776_177687

/-- The number of ways to select a team of 6 people from a group of 7 boys and 9 girls, with at least 2 boys -/
def selectTeam (boys girls : ℕ) : ℕ := 
  (Nat.choose boys 2 * Nat.choose girls 4) +
  (Nat.choose boys 3 * Nat.choose girls 3) +
  (Nat.choose boys 4 * Nat.choose girls 2) +
  (Nat.choose boys 5 * Nat.choose girls 1) +
  (Nat.choose boys 6 * Nat.choose girls 0)

/-- Theorem stating that the number of ways to select the team is 7042 -/
theorem team_selection_count : selectTeam 7 9 = 7042 := by
  sorry

end team_selection_count_l1776_177687


namespace gcd_18_30_45_l1776_177691

theorem gcd_18_30_45 : Nat.gcd 18 (Nat.gcd 30 45) = 3 := by
  sorry

end gcd_18_30_45_l1776_177691


namespace polynomial_division_remainder_l1776_177610

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 = (x^2 + 7*x + 2) * q + (-315*x - 94) := by sorry

end polynomial_division_remainder_l1776_177610


namespace length_function_is_linear_alpha_is_rate_of_change_l1776_177653

/-- Represents the length of a metal rod as a function of temperature -/
def length_function (l₀ α : ℝ) (t : ℝ) : ℝ := l₀ * (1 + α * t)

/-- States that the length function is linear in t -/
theorem length_function_is_linear (l₀ α : ℝ) : 
  ∃ m b : ℝ, ∀ t : ℝ, length_function l₀ α t = m * t + b :=
sorry

/-- Defines α as the rate of change of length with respect to temperature -/
theorem alpha_is_rate_of_change (l₀ α : ℝ) : 
  α = (length_function l₀ α 1 - length_function l₀ α 0) / l₀ :=
sorry

end length_function_is_linear_alpha_is_rate_of_change_l1776_177653


namespace quadratic_equation_equivalence_l1776_177637

theorem quadratic_equation_equivalence :
  ∃ (r : ℝ), ∀ (x : ℝ), (4 * x^2 - 8 * x - 288 = 0) ↔ ((x + r)^2 = 73) :=
by sorry

end quadratic_equation_equivalence_l1776_177637


namespace triangulation_has_120_triangle_l1776_177629

/-- A triangulation of a triangle -/
structure Triangulation :=
  (vertices : Set ℝ × ℝ)
  (edges : Set (ℝ × ℝ × ℝ × ℝ))
  (triangles : Set (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ))

/-- The original triangle in a triangulation -/
def originalTriangle (t : Triangulation) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := sorry

/-- Check if all angles in a triangle are not exceeding 120° -/
def allAnglesWithin120 (triangle : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) : Prop := sorry

/-- The theorem statement -/
theorem triangulation_has_120_triangle 
  (t : Triangulation) 
  (h : allAnglesWithin120 (originalTriangle t)) :
  ∃ triangle ∈ t.triangles, allAnglesWithin120 triangle :=
sorry

end triangulation_has_120_triangle_l1776_177629


namespace marathon_average_time_l1776_177640

def casey_time : ℝ := 6

theorem marathon_average_time (casey_time : ℝ) (zendaya_factor : ℝ) :
  casey_time = 6 →
  zendaya_factor = 1/3 →
  let zendaya_time := casey_time * (1 + zendaya_factor)
  let total_time := casey_time + zendaya_time
  let average_time := total_time / 2
  average_time = 7 := by sorry

end marathon_average_time_l1776_177640


namespace gcd_power_minus_one_l1776_177649

theorem gcd_power_minus_one : Nat.gcd (2^2000 - 1) (2^1990 - 1) = 2^10 - 1 := by
  sorry

end gcd_power_minus_one_l1776_177649


namespace leftover_pie_share_l1776_177601

theorem leftover_pie_share (total_pie : ℚ) (num_people : ℕ) : 
  total_pie = 12 / 13 → num_people = 4 → total_pie / num_people = 3 / 13 := by
  sorry

end leftover_pie_share_l1776_177601


namespace carrie_phone_trade_in_l1776_177644

/-- The trade-in value of Carrie's old phone -/
def trade_in_value : ℕ := sorry

/-- The cost of the new iPhone -/
def iphone_cost : ℕ := 800

/-- Carrie's weekly earnings from babysitting -/
def weekly_earnings : ℕ := 80

/-- The number of weeks Carrie has to work -/
def weeks_to_work : ℕ := 7

/-- The total amount Carrie earns from babysitting -/
def total_earnings : ℕ := weekly_earnings * weeks_to_work

theorem carrie_phone_trade_in :
  trade_in_value = iphone_cost - total_earnings :=
sorry

end carrie_phone_trade_in_l1776_177644


namespace chocobites_remainder_l1776_177665

theorem chocobites_remainder (m : ℕ) : 
  m % 8 = 5 → (4 * m) % 8 = 4 := by
sorry

end chocobites_remainder_l1776_177665


namespace total_splash_width_l1776_177639

/-- Represents the splash width of different rock types -/
def splash_width (rock_type : String) : ℚ :=
  match rock_type with
  | "pebble" => 1/4
  | "rock" => 1/2
  | "boulder" => 2
  | "mini-boulder" => 1
  | "large_pebble" => 1/3
  | _ => 0

/-- Calculates the total splash width for a given rock type and count -/
def total_splash (rock_type : String) (count : ℕ) : ℚ :=
  (splash_width rock_type) * count

/-- Theorem: The total width of splashes is 14 meters -/
theorem total_splash_width :
  (total_splash "pebble" 8) +
  (total_splash "rock" 4) +
  (total_splash "boulder" 3) +
  (total_splash "mini-boulder" 2) +
  (total_splash "large_pebble" 6) = 14 := by
  sorry

end total_splash_width_l1776_177639


namespace ratio_lcm_problem_l1776_177685

theorem ratio_lcm_problem (a b : ℕ+) (h1 : a.val * 4 = b.val * 3) 
  (h2 : Nat.lcm a.val b.val = 180) (h3 : a.val = 45 ∨ b.val = 45) :
  (if a.val = 45 then b.val else a.val) = 60 := by
  sorry

end ratio_lcm_problem_l1776_177685


namespace alloy_problem_solution_l1776_177657

/-- Represents the copper-tin alloy problem -/
structure AlloyProblem where
  mass1 : ℝ  -- Mass of the first alloy
  copper1 : ℝ  -- Copper percentage in the first alloy
  mass2 : ℝ  -- Mass of the second alloy
  copper2 : ℝ  -- Copper percentage in the second alloy
  targetMass : ℝ  -- Target mass of the resulting alloy

/-- Represents the solution to the copper-tin alloy problem -/
structure AlloySolution where
  pMin : ℝ  -- Minimum percentage of copper in the resulting alloy
  pMax : ℝ  -- Maximum percentage of copper in the resulting alloy
  mass1 : ℝ → ℝ  -- Function to calculate mass of the first alloy
  mass2 : ℝ → ℝ  -- Function to calculate mass of the second alloy

/-- Theorem stating the solution to the copper-tin alloy problem -/
theorem alloy_problem_solution (problem : AlloyProblem) 
  (h1 : problem.mass1 = 4) 
  (h2 : problem.copper1 = 40) 
  (h3 : problem.mass2 = 6) 
  (h4 : problem.copper2 = 30) 
  (h5 : problem.targetMass = 8) :
  ∃ (solution : AlloySolution),
    solution.pMin = 32.5 ∧
    solution.pMax = 35 ∧
    (∀ p, solution.mass1 p = 0.8 * p - 24) ∧
    (∀ p, solution.mass2 p = 32 - 0.8 * p) ∧
    (∀ p, 32.5 ≤ p → p ≤ 35 → 
      0 ≤ solution.mass1 p ∧ 
      solution.mass1 p ≤ problem.mass1 ∧
      0 ≤ solution.mass2 p ∧ 
      solution.mass2 p ≤ problem.mass2 ∧
      solution.mass1 p + solution.mass2 p = problem.targetMass ∧
      solution.mass1 p * (problem.copper1 / 100) + solution.mass2 p * (problem.copper2 / 100) = 
        problem.targetMass * (p / 100)) :=
by
  sorry

end alloy_problem_solution_l1776_177657


namespace students_without_A_l1776_177689

theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (both : ℕ) 
  (h_total : total = 50)
  (h_history : history = 12)
  (h_math : math = 25)
  (h_both : both = 6) : 
  total - (history + math - both) = 19 := by
  sorry

end students_without_A_l1776_177689


namespace parabola_symmetry_l1776_177656

/-- Given that M(0,5) and N(2,5) lie on the parabola y = 2(x-m)^2 + 3, prove that m = 1 -/
theorem parabola_symmetry (m : ℝ) : 
  (5 : ℝ) = 2 * (0 - m)^2 + 3 ∧ 
  (5 : ℝ) = 2 * (2 - m)^2 + 3 → 
  m = 1 := by sorry

end parabola_symmetry_l1776_177656


namespace photo_arrangements_l1776_177630

/-- The number of different arrangements of 5 students and 2 teachers in a row,
    where exactly two students stand between the two teachers. -/
def arrangements_count : ℕ := 960

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students required to stand between teachers -/
def students_between : ℕ := 2

theorem photo_arrangements :
  arrangements_count = 960 :=
sorry

end photo_arrangements_l1776_177630


namespace complement_A_inter_B_range_of_a_l1776_177616

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
def B : Set ℝ := {x | 4 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Theorem for the complement of A ∩ B in U
theorem complement_A_inter_B :
  (A ∩ B)ᶜ = {x | x ≤ 4 ∨ x > 5} :=
sorry

-- Theorem for the range of values for a
theorem range_of_a (a : ℝ) (h : A ∪ B ⊆ C a) :
  a ≥ 6 :=
sorry

end complement_A_inter_B_range_of_a_l1776_177616


namespace integral_sqrt_minus_square_l1776_177645

theorem integral_sqrt_minus_square : 
  ∫ x in (0:ℝ)..1, (Real.sqrt (1 - (x - 1)^2) - x^2) = π/4 - 1/3 := by sorry

end integral_sqrt_minus_square_l1776_177645


namespace no_prime_divisor_l1776_177605

theorem no_prime_divisor : ¬ ∃ (p : ℕ), Prime p ∧ p > 1 ∧ p ∣ (1255 - 8) ∧ p ∣ (1490 - 11) := by
  sorry

end no_prime_divisor_l1776_177605


namespace increase_by_percentage_l1776_177600

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 80 → percentage = 150 → result = initial * (1 + percentage / 100) → result = 200 := by
  sorry

end increase_by_percentage_l1776_177600


namespace train_bridge_crossing_time_l1776_177631

/-- Time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 110)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 265) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end train_bridge_crossing_time_l1776_177631


namespace firewood_collection_sum_l1776_177696

/-- The amount of firewood collected by Kimberley in pounds -/
def kimberley_firewood : ℕ := 10

/-- The amount of firewood collected by Houston in pounds -/
def houston_firewood : ℕ := 12

/-- The amount of firewood collected by Ela in pounds -/
def ela_firewood : ℕ := 13

/-- The total amount of firewood collected by Kimberley, Ela, and Houston -/
def total_firewood : ℕ := kimberley_firewood + ela_firewood + houston_firewood

theorem firewood_collection_sum :
  total_firewood = 35 := by
  sorry

end firewood_collection_sum_l1776_177696


namespace constant_value_l1776_177661

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f x + c * f (8 - x) = x

theorem constant_value (f : ℝ → ℝ) (c : ℝ) 
    (h1 : SatisfiesCondition f c) 
    (h2 : f 2 = 2) : 
  c = 3 := by
  sorry

end constant_value_l1776_177661


namespace min_value_quadratic_min_value_quadratic_achievable_l1776_177683

theorem min_value_quadratic (x : ℝ) : x^2 + x + 1 ≥ 3/4 :=
sorry

theorem min_value_quadratic_achievable : ∃ x : ℝ, x^2 + x + 1 = 3/4 :=
sorry

end min_value_quadratic_min_value_quadratic_achievable_l1776_177683


namespace greatest_divisor_four_consecutive_integers_l1776_177672

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 → 
  ∃ m : ℕ, m = 12 ∧ 
  (∀ k : ℕ, k > m → ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3)))) ∧
  (12 ∣ (n * (n + 1) * (n + 2) * (n + 3))) :=
by sorry


end greatest_divisor_four_consecutive_integers_l1776_177672


namespace computer_factory_earnings_l1776_177646

/-- Calculates the earnings from selling computers produced in a week -/
def weekly_earnings (daily_production : ℕ) (price_per_unit : ℕ) : ℕ :=
  daily_production * 7 * price_per_unit

/-- Proves that the weekly earnings for the given conditions equal $1,575,000 -/
theorem computer_factory_earnings :
  weekly_earnings 1500 150 = 1575000 := by
  sorry

end computer_factory_earnings_l1776_177646


namespace shoe_cost_difference_l1776_177604

/-- Proves that the percentage difference between the average cost per year of new shoes
    and the cost of repairing used shoes is 10.34%, given the specified conditions. -/
theorem shoe_cost_difference (used_repair_cost : ℝ) (used_repair_duration : ℝ)
    (new_shoe_cost : ℝ) (new_shoe_duration : ℝ)
    (h1 : used_repair_cost = 14.50)
    (h2 : used_repair_duration = 1)
    (h3 : new_shoe_cost = 32.00)
    (h4 : new_shoe_duration = 2) :
    let used_cost_per_year := used_repair_cost / used_repair_duration
    let new_cost_per_year := new_shoe_cost / new_shoe_duration
    let percentage_difference := (new_cost_per_year - used_cost_per_year) / used_cost_per_year * 100
    percentage_difference = 10.34 := by
  sorry

end shoe_cost_difference_l1776_177604


namespace complex_expression_equals_negative_two_l1776_177643

theorem complex_expression_equals_negative_two :
  let z : ℂ := Complex.exp (3 * Real.pi * Complex.I / 8)
  (z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6)) = -2 := by
  sorry

end complex_expression_equals_negative_two_l1776_177643


namespace reggie_remaining_money_l1776_177647

/-- Calculates the remaining money after a purchase --/
def remaining_money (initial_amount number_of_items cost_per_item : ℕ) : ℕ :=
  initial_amount - (number_of_items * cost_per_item)

/-- Proves that Reggie has $38 left after his purchase --/
theorem reggie_remaining_money :
  remaining_money 48 5 2 = 38 := by
  sorry

end reggie_remaining_money_l1776_177647


namespace josh_marbles_difference_l1776_177671

/-- Given Josh's marble collection scenario, prove the difference between lost and found marbles. -/
theorem josh_marbles_difference (initial : ℕ) (found : ℕ) (lost : ℕ) 
  (h1 : initial = 15) 
  (h2 : found = 9) 
  (h3 : lost = 23) : 
  lost - found = 14 := by
  sorry

end josh_marbles_difference_l1776_177671


namespace diagonal_contains_all_numbers_l1776_177618

/-- Represents a 25x25 table with integers from 1 to 25 -/
def Table := Fin 25 → Fin 25 → Fin 25

/-- The table is symmetric with respect to the main diagonal -/
def isSymmetric (t : Table) : Prop :=
  ∀ i j : Fin 25, t i j = t j i

/-- Each row contains all numbers from 1 to 25 -/
def hasAllNumbersInRow (t : Table) : Prop :=
  ∀ i : Fin 25, ∀ k : Fin 25, ∃ j : Fin 25, t i j = k

/-- The main diagonal contains all numbers from 1 to 25 -/
def allNumbersOnDiagonal (t : Table) : Prop :=
  ∀ k : Fin 25, ∃ i : Fin 25, t i i = k

theorem diagonal_contains_all_numbers (t : Table) 
  (h_sym : isSymmetric t) (h_row : hasAllNumbersInRow t) : 
  allNumbersOnDiagonal t := by
  sorry

end diagonal_contains_all_numbers_l1776_177618


namespace furniture_legs_problem_l1776_177676

theorem furniture_legs_problem (total_tables : ℕ) (total_legs : ℕ) (four_leg_tables : ℕ) :
  total_tables = 36 →
  total_legs = 124 →
  four_leg_tables = 16 →
  (total_legs - 4 * four_leg_tables) / (total_tables - four_leg_tables) = 3 :=
by sorry

end furniture_legs_problem_l1776_177676


namespace arithmetic_sequence_before_five_l1776_177666

/-- Given an arithmetic sequence with first term 105 and common difference -5,
    prove that there are 20 terms before the term with value 5. -/
theorem arithmetic_sequence_before_five (n : ℕ) : 
  (105 : ℤ) - 5 * n = 5 → n - 1 = 20 := by sorry

end arithmetic_sequence_before_five_l1776_177666


namespace consecutive_integers_cube_sum_l1776_177660

theorem consecutive_integers_cube_sum (x : ℕ) (h : x > 0) 
  (h_prod : x * (x + 1) * (x + 2) = 12 * (3 * x + 3)) :
  x^3 + (x + 1)^3 + (x + 2)^3 = 216 := by
  sorry

end consecutive_integers_cube_sum_l1776_177660


namespace no_sequence_satisfying_conditions_l1776_177612

theorem no_sequence_satisfying_conditions : ¬ ∃ (a : ℕ → ℤ), 
  (∀ i j : ℕ, i ≠ j → a i ≠ a j) ∧ 
  (∀ k : ℕ, k > 0 → a (k^2) > 0 ∧ a (k^2 + k) < 0) ∧
  (∀ n : ℕ, n > 0 → |a (n + 1) - a n| ≤ 2023 * Real.sqrt n) :=
by sorry

end no_sequence_satisfying_conditions_l1776_177612


namespace base8_subtraction_l1776_177694

-- Define a function to convert base 8 numbers to natural numbers
def base8ToNat (x : ℕ) : ℕ := sorry

-- Define a function to convert natural numbers to base 8
def natToBase8 (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem base8_subtraction :
  natToBase8 ((base8ToNat 256 + base8ToNat 167) - base8ToNat 145) = 370 := by
  sorry

end base8_subtraction_l1776_177694


namespace equation_solution_l1776_177632

theorem equation_solution : ∃! x : ℝ, (567.23 - x) * 45.7 + (64.89 / 11.5)^3 - 2.78 = 18756.120 := by
  sorry

end equation_solution_l1776_177632


namespace cloth_sale_profit_l1776_177634

/-- The number of meters of cloth sold by a trader -/
def meters_sold : ℕ := 40

/-- The profit per meter of cloth in Rupees -/
def profit_per_meter : ℕ := 25

/-- The total profit earned by the trader in Rupees -/
def total_profit : ℕ := 1000

/-- Theorem stating that the number of meters sold multiplied by the profit per meter equals the total profit -/
theorem cloth_sale_profit : meters_sold * profit_per_meter = total_profit := by
  sorry

end cloth_sale_profit_l1776_177634


namespace square_sum_inequality_l1776_177669

theorem square_sum_inequality (x y : ℝ) :
  x^2 + y^2 ≤ 2*(x + y - 1) → x = 1 ∧ y = 1 := by
  sorry

end square_sum_inequality_l1776_177669


namespace problem_solution_l1776_177633

/-- Predicate to check if a number is divisible by another -/
def is_divisible (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬is_divisible n d

/-- The four statements in the problem -/
def statement1 (a b : ℕ) : Prop := is_divisible (a^2 + 6*a + 8) b
def statement2 (a b : ℕ) : Prop := a^2 + a*b - 6*b^2 - 15*b - 9 = 0
def statement3 (a b : ℕ) : Prop := is_divisible (a + 2*b + 2) 4
def statement4 (a b : ℕ) : Prop := is_prime (a + 6*b + 2)

/-- Predicate to check if exactly three out of four statements are true -/
def three_true (a b : ℕ) : Prop :=
  (statement1 a b ∧ statement2 a b ∧ statement3 a b ∧ ¬statement4 a b) ∨
  (statement1 a b ∧ statement2 a b ∧ ¬statement3 a b ∧ statement4 a b) ∨
  (statement1 a b ∧ ¬statement2 a b ∧ statement3 a b ∧ statement4 a b) ∨
  (¬statement1 a b ∧ statement2 a b ∧ statement3 a b ∧ statement4 a b)

theorem problem_solution :
  ∀ a b : ℕ, three_true a b ↔ ((a = 5 ∧ b = 1) ∨ (a = 17 ∧ b = 7)) :=
sorry

end problem_solution_l1776_177633


namespace ratio_equation_solution_l1776_177621

theorem ratio_equation_solution (a b : ℚ) 
  (h1 : b / a = 4)
  (h2 : b = 20 - 7 * a) : 
  a = 20 / 11 := by
sorry

end ratio_equation_solution_l1776_177621


namespace alexey_game_max_score_l1776_177608

def score (x : ℕ) : ℕ :=
  (if x % 3 = 0 then 3 else 0) +
  (if x % 5 = 0 then 5 else 0) +
  (if x % 7 = 0 then 7 else 0) +
  (if x % 9 = 0 then 9 else 0) +
  (if x % 11 = 0 then 11 else 0)

theorem alexey_game_max_score :
  ∀ x : ℕ, 2017 ≤ x ∧ x ≤ 2117 → score x ≤ score 2079 :=
by sorry

end alexey_game_max_score_l1776_177608


namespace hair_cut_length_l1776_177690

/-- Given Isabella's original and current hair lengths, prove the length of hair cut off. -/
theorem hair_cut_length (original_length current_length cut_length : ℕ) : 
  original_length = 18 → current_length = 9 → cut_length = original_length - current_length :=
by sorry

end hair_cut_length_l1776_177690


namespace remainder_theorem_l1776_177686

theorem remainder_theorem (P Q Q' R R' a b c : ℕ) 
  (h1 : P = a * Q + R) 
  (h2 : Q = (b + c) * Q' + R') : 
  P % (a * b) = (a * c * Q' + a * R' + R) % (a * b) :=
sorry

end remainder_theorem_l1776_177686


namespace min_value_reciprocal_sum_l1776_177668

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), 2 * a * x - b * y + 2 = 0 ∧
                 x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧
                 (∃ (x1 y1 x2 y2 : ℝ),
                    2 * a * x1 - b * y1 + 2 = 0 ∧
                    x1^2 + y1^2 + 2*x1 - 4*y1 + 1 = 0 ∧
                    2 * a * x2 - b * y2 + 2 = 0 ∧
                    x2^2 + y2^2 + 2*x2 - 4*y2 + 1 = 0 ∧
                    (x2 - x1)^2 + (y2 - y1)^2 = 16)) →
  (∀ c d : ℝ, c > 0 → d > 0 →
    (∃ (x y : ℝ), 2 * c * x - d * y + 2 = 0 ∧
                   x^2 + y^2 + 2*x - 4*y + 1 = 0) →
    1/a + 1/b ≤ 1/c + 1/d) ∧
  (1/a + 1/b = 4) :=
sorry

end min_value_reciprocal_sum_l1776_177668


namespace two_special_numbers_exist_l1776_177622

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def has_no_single_digit_prime_factors (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p > 9

theorem two_special_numbers_exist : ∃ x y : ℕ,
  x + y = 173717 ∧
  is_four_digit (x - y) ∧
  has_no_single_digit_prime_factors (x - y) ∧
  (1558 ∣ x ∨ 1558 ∣ y) ∧
  x = 91143 ∧ y = 82574 := by
  sorry

end two_special_numbers_exist_l1776_177622


namespace A_value_l1776_177648

noncomputable def A (m n : ℝ) : ℝ :=
  (((4 * m^2 * n^2) / (4 * m * n - m^2 - 4 * n^2) -
    (2 + n / m + m / n) / (4 / (m * n) - 1 / n^2 - 4 / m^2))^(1/2)) *
  (Real.sqrt (m * n) / (m - 2 * n))

theorem A_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  A m n = if 1 < m / n ∧ m / n < 2 then n - m else m - n := by
  sorry

end A_value_l1776_177648


namespace workshop_technicians_l1776_177626

theorem workshop_technicians 
  (total_workers : ℕ) 
  (avg_salary_all : ℚ) 
  (avg_salary_tech : ℚ) 
  (avg_salary_others : ℚ) 
  (h1 : total_workers = 20)
  (h2 : avg_salary_all = 750)
  (h3 : avg_salary_tech = 900)
  (h4 : avg_salary_others = 700) :
  ∃ (num_technicians : ℕ), 
    num_technicians * avg_salary_tech + (total_workers - num_technicians) * avg_salary_others = 
    total_workers * avg_salary_all ∧ 
    num_technicians = 5 := by
  sorry

end workshop_technicians_l1776_177626


namespace lcm_gcd_product_12_9_l1776_177641

theorem lcm_gcd_product_12_9 :
  Nat.lcm 12 9 * Nat.gcd 12 9 = 108 := by
  sorry

end lcm_gcd_product_12_9_l1776_177641


namespace intersection_A_complement_B_l1776_177692

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < Real.exp 2}

-- Define the complement of B
def C_R_B : Set ℝ := {x | x ≤ 1 ∨ Real.exp 2 ≤ x}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ C_R_B = {x | 0 < x ∧ x ≤ 1} :=
sorry

end intersection_A_complement_B_l1776_177692


namespace sum_of_squares_of_roots_l1776_177698

/-- Given a cubic equation x√x - 9x + 9√x - 4 = 0 with real nonnegative roots,
    the sum of the squares of its roots is 63. -/
theorem sum_of_squares_of_roots : ∃ (r s t : ℝ),
  (∀ x : ℝ, x ≥ 0 → (x * Real.sqrt x - 9 * x + 9 * Real.sqrt x - 4 = 0 ↔ x = r * r ∨ x = s * s ∨ x = t * t)) →
  r * r + s * s + t * t = 63 := by
  sorry

end sum_of_squares_of_roots_l1776_177698


namespace divisors_of_1442_l1776_177628

theorem divisors_of_1442 :
  let n : ℕ := 1442
  let divisors : Finset ℕ := {1, 11, 131, 1442}
  (∀ (d : ℕ), d ∣ n ↔ d ∈ divisors) ∧
  (∃ (p q : ℕ), Prime p ∧ Prime q ∧ n = p * q) := by
  sorry

end divisors_of_1442_l1776_177628


namespace trevor_reed_difference_l1776_177613

/-- Represents the yearly toy spending of Trevor, Reed, and Quinn -/
structure ToySpending where
  trevor : ℕ
  reed : ℕ
  quinn : ℕ

/-- The conditions of the problem -/
def spending_conditions (s : ToySpending) : Prop :=
  s.trevor = 80 ∧
  s.reed = 2 * s.quinn ∧
  s.trevor > s.reed ∧
  4 * (s.trevor + s.reed + s.quinn) = 680

/-- The theorem to prove -/
theorem trevor_reed_difference (s : ToySpending) :
  spending_conditions s → s.trevor - s.reed = 20 := by
  sorry

end trevor_reed_difference_l1776_177613


namespace train_length_calculation_l1776_177674

theorem train_length_calculation (train_speed : Real) (platform_length : Real) (crossing_time : Real) :
  train_speed = 55 * 1000 / 3600 →
  platform_length = 300 →
  crossing_time = 35.99712023038157 →
  let total_distance := train_speed * crossing_time
  let train_length := total_distance - platform_length
  train_length = 249.9999999999999 := by
  sorry

end train_length_calculation_l1776_177674


namespace complex_imaginary_problem_l1776_177619

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

theorem complex_imaginary_problem (z : ℂ) 
  (h1 : isPurelyImaginary z) 
  (h2 : isPurelyImaginary ((z + 1)^2 - 2*I)) : 
  z = -I := by sorry

end complex_imaginary_problem_l1776_177619


namespace alligators_in_pond_l1776_177623

/-- The number of snakes in the pond -/
def num_snakes : ℕ := 18

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := 56

/-- The number of eyes each snake has -/
def snake_eyes : ℕ := 2

/-- The number of eyes each alligator has -/
def alligator_eyes : ℕ := 2

/-- The number of alligators in the pond -/
def num_alligators : ℕ := 10

theorem alligators_in_pond :
  num_snakes * snake_eyes + num_alligators * alligator_eyes = total_eyes :=
by sorry

end alligators_in_pond_l1776_177623


namespace walking_sequence_intersection_l1776_177620

/-- A walking sequence is a sequence of integers where each term differs from the previous by ±1. -/
def IsWalkingSequence (a : Fin 2016 → ℤ) : Prop :=
  ∀ i : Fin 2015, a (i + 1) = a i + 1 ∨ a (i + 1) = a i - 1

/-- The sequence b as defined in the problem -/
def b : Fin 2016 → ℤ
  | ⟨i, h⟩ => if i < 1009 then i + 1 else 2018 - i

/-- The main theorem statement -/
theorem walking_sequence_intersection :
  ∃ (a : Fin 2016 → ℤ), IsWalkingSequence a ∧
  (∀ i, 1 ≤ a i ∧ a i ≤ 1010) →
  ∃ j, a j = b j :=
sorry

end walking_sequence_intersection_l1776_177620


namespace opposite_to_gold_is_silver_l1776_177654

-- Define the colors
inductive Color
| P | M | C | S | G | V | L

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  top : Face
  bottom : Face
  front : Face
  back : Face
  left : Face
  right : Face

-- Define the theorem
theorem opposite_to_gold_is_silver (cube : Cube) : 
  cube.top.color = Color.P → 
  cube.bottom.color = Color.V → 
  (cube.front.color = Color.G ∨ cube.back.color = Color.G ∨ cube.left.color = Color.G ∨ cube.right.color = Color.G) → 
  ((cube.front.color = Color.G → cube.back.color = Color.S) ∧ 
   (cube.back.color = Color.G → cube.front.color = Color.S) ∧ 
   (cube.left.color = Color.G → cube.right.color = Color.S) ∧ 
   (cube.right.color = Color.G → cube.left.color = Color.S)) := by
  sorry


end opposite_to_gold_is_silver_l1776_177654


namespace probability_three_standard_parts_l1776_177697

/-- Represents a box containing parts -/
structure Box where
  total : ℕ
  standard : ℕ
  h : standard ≤ total

/-- Calculates the probability of selecting a standard part from a box -/
def probabilityStandard (box : Box) : ℚ :=
  box.standard / box.total

/-- Theorem: The probability of selecting standard parts from all three boxes is 7/10 -/
theorem probability_three_standard_parts
  (box1 : Box)
  (box2 : Box)
  (box3 : Box)
  (h1 : box1.total = 30 ∧ box1.standard = 27)
  (h2 : box2.total = 30 ∧ box2.standard = 28)
  (h3 : box3.total = 30 ∧ box3.standard = 25) :
  probabilityStandard box1 * probabilityStandard box2 * probabilityStandard box3 = 7/10 := by
  sorry


end probability_three_standard_parts_l1776_177697


namespace sum_reciprocal_pairs_bound_l1776_177609

/-- 
Given non-negative real numbers x, y, and z satisfying xy + yz + zx = 1,
the sum 1/(x+y) + 1/(y+z) + 1/(z+x) is greater than or equal to 5/2.
-/
theorem sum_reciprocal_pairs_bound (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum_prod : x*y + y*z + z*x = 1) : 
  1/(x+y) + 1/(y+z) + 1/(z+x) ≥ 5/2 := by
  sorry

end sum_reciprocal_pairs_bound_l1776_177609


namespace oldest_child_age_l1776_177636

theorem oldest_child_age (average_age : ℝ) (age1 age2 age3 : ℕ) :
  average_age = 9 ∧ age1 = 5 ∧ age2 = 8 ∧ age3 = 11 →
  ∃ (age4 : ℕ), (age1 + age2 + age3 + age4 : ℝ) / 4 = average_age ∧ age4 = 12 :=
by sorry

end oldest_child_age_l1776_177636


namespace expression_evaluation_l1776_177650

theorem expression_evaluation (x : ℝ) (h : x > 2) :
  Real.sqrt (x^2 / (1 - (x^2 - 4) / x^2)) = x^2 / 2 := by
  sorry

end expression_evaluation_l1776_177650


namespace tetrahedron_center_of_mass_l1776_177678

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertices : Fin 4 → Point3D

/-- The centroid of a tetrahedron -/
def centroid (t : Tetrahedron) : Point3D := sorry

/-- The circumcenter of a tetrahedron -/
def circumcenter (t : Tetrahedron) : Point3D := sorry

/-- The orthocenter of a tetrahedron -/
def orthocenter (t : Tetrahedron) : Point3D := sorry

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def is_midpoint (m p1 p2 : Point3D) : Prop := sorry

/-- Calculates the center of mass given masses and their positions -/
def center_of_mass (masses : List ℝ) (positions : List Point3D) : Point3D := sorry

/-- Main theorem -/
theorem tetrahedron_center_of_mass (t : Tetrahedron) :
  let s := centroid t
  let o := circumcenter t
  let m := orthocenter t
  collinear s o m ∧ is_midpoint s o m →
  center_of_mass 
    [1, 1, 1, 1, -2] 
    [t.vertices 0, t.vertices 1, t.vertices 2, t.vertices 3, m] = o := by
  sorry

end tetrahedron_center_of_mass_l1776_177678


namespace range_of_b_l1776_177603

/-- The curve representing a semi-circle -/
def curve (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 4

/-- The line intersecting the curve -/
def line (x y b : ℝ) : Prop := y = x + b

/-- The domain constraints for x and y -/
def domain_constraints (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 3

/-- The theorem stating the range of b -/
theorem range_of_b :
  ∀ b : ℝ, (∃ x y : ℝ, curve x y ∧ line x y b ∧ domain_constraints x y) ↔ 
  (1 - 2 * Real.sqrt 2 ≤ b ∧ b ≤ 3) :=
sorry

end range_of_b_l1776_177603


namespace rachel_research_time_l1776_177652

/-- Represents the time spent on different activities while writing an essay -/
structure EssayTime where
  writing_speed : ℕ  -- pages per 30 minutes
  total_pages : ℕ
  editing_time : ℕ  -- in minutes
  total_time : ℕ    -- in minutes

/-- Calculates the time spent researching for an essay -/
def research_time (e : EssayTime) : ℕ :=
  e.total_time - (e.total_pages * 30 + e.editing_time)

/-- Theorem stating that Rachel spent 45 minutes researching -/
theorem rachel_research_time :
  let e : EssayTime := {
    writing_speed := 1,
    total_pages := 6,
    editing_time := 75,
    total_time := 300
  }
  research_time e = 45 := by sorry

end rachel_research_time_l1776_177652
