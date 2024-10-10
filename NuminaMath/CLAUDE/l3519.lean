import Mathlib

namespace equilateral_iff_sum_zero_l3519_351900

-- Define j as a complex number representing a rotation by 120°
noncomputable def j : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

-- Define the property of j
axiom j_cube : j ^ 3 = 1
axiom j_sum : 1 + j + j^2 = 0

-- Define a triangle in the complex plane
structure Triangle :=
  (A B C : ℂ)

-- Define the property of being equilateral
def is_equilateral (t : Triangle) : Prop :=
  Complex.abs (t.B - t.A) = Complex.abs (t.C - t.B) ∧
  Complex.abs (t.C - t.B) = Complex.abs (t.A - t.C)

-- State the theorem
theorem equilateral_iff_sum_zero (t : Triangle) :
  is_equilateral t ↔ t.A + j * t.B + j^2 * t.C = 0 :=
sorry

end equilateral_iff_sum_zero_l3519_351900


namespace cos_difference_inverse_cos_tan_l3519_351975

theorem cos_difference_inverse_cos_tan (x y : ℝ) 
  (hx : x^2 ≤ 1) (hy : y > 0) : 
  Real.cos (Real.arccos (4/5) - Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end cos_difference_inverse_cos_tan_l3519_351975


namespace labourer_salary_proof_l3519_351914

/-- The salary increase rate per year -/
def increase_rate : ℝ := 1.4

/-- The number of years -/
def years : ℕ := 3

/-- The final salary after 3 years -/
def final_salary : ℝ := 8232

/-- The present salary of the labourer -/
def present_salary : ℝ := 3000

theorem labourer_salary_proof :
  (present_salary * increase_rate ^ years) = final_salary :=
sorry

end labourer_salary_proof_l3519_351914


namespace probability_blue_after_red_l3519_351986

/-- Probability of picking a blue marble after removing a red one --/
theorem probability_blue_after_red (total : ℕ) (yellow : ℕ) (green : ℕ) (red : ℕ) (blue : ℕ) :
  total = 120 →
  yellow = 30 →
  green = yellow / 3 →
  red = 2 * green →
  blue = total - yellow - green - red →
  (blue : ℚ) / (total - 1 : ℚ) = 60 / 119 := by
  sorry

end probability_blue_after_red_l3519_351986


namespace intersection_of_A_and_B_l3519_351960

def A : Set ℤ := {x : ℤ | |x| < 3}
def B : Set ℤ := {x : ℤ | |x| > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} := by
  sorry

end intersection_of_A_and_B_l3519_351960


namespace comic_book_ratio_l3519_351921

def initial_books : ℕ := 22
def final_books : ℕ := 17
def bought_books : ℕ := 6

theorem comic_book_ratio : 
  ∃ (sold_books : ℕ), 
    initial_books - sold_books + bought_books = final_books ∧
    sold_books * 2 = initial_books := by
  sorry

end comic_book_ratio_l3519_351921


namespace trader_cloth_sale_l3519_351936

/-- The number of meters of cloth sold by a trader -/
def meters_sold (total_price profit_per_meter cost_per_meter : ℚ) : ℚ :=
  total_price / (cost_per_meter + profit_per_meter)

/-- Theorem: The trader sold 85 meters of cloth -/
theorem trader_cloth_sale : meters_sold 8925 5 100 = 85 := by
  sorry

end trader_cloth_sale_l3519_351936


namespace sugar_problem_solution_l3519_351968

def sugar_problem (sugar_at_home : ℕ) (bags_bought : ℕ) (dozens : ℕ) 
  (sugar_per_dozen_batter : ℚ) (sugar_per_dozen_frosting : ℚ) : Prop :=
  ∃ (sugar_per_bag : ℕ),
    sugar_at_home = 3 ∧
    bags_bought = 2 ∧
    dozens = 5 ∧
    sugar_per_dozen_batter = 1 ∧
    sugar_per_dozen_frosting = 2 ∧
    sugar_at_home + bags_bought * sugar_per_bag = 
      dozens * (sugar_per_dozen_batter + sugar_per_dozen_frosting) ∧
    sugar_per_bag = 6

theorem sugar_problem_solution :
  sugar_problem 3 2 5 1 2 := by
  sorry

end sugar_problem_solution_l3519_351968


namespace fastest_reaction_rate_C_l3519_351935

-- Define the reaction rates
def rate_A : ℝ := 0.15
def rate_B : ℝ := 0.6
def rate_C : ℝ := 0.5
def rate_D : ℝ := 0.4

-- Define the stoichiometric coefficients
def coeff_A : ℝ := 1
def coeff_B : ℝ := 3
def coeff_C : ℝ := 2
def coeff_D : ℝ := 2

-- Theorem: The reaction rate of C is the fastest
theorem fastest_reaction_rate_C :
  rate_C / coeff_C > rate_A / coeff_A ∧
  rate_C / coeff_C > rate_B / coeff_B ∧
  rate_C / coeff_C > rate_D / coeff_D :=
by sorry

end fastest_reaction_rate_C_l3519_351935


namespace floor_ceiling_sum_l3519_351973

theorem floor_ceiling_sum (x y : ℝ) (hx : 1 < x ∧ x < 2) (hy : 3 < y ∧ y < 4) :
  ⌊x⌋ + ⌈y⌉ = 5 := by
  sorry

end floor_ceiling_sum_l3519_351973


namespace simplify_expression_l3519_351903

theorem simplify_expression (a b x y : ℝ) (h : b*x + a*y ≠ 0) :
  (b*x*(a^2*x^2 + 2*a^2*y^2 + b^2*y^2)) / (b*x + a*y) +
  (a*y*(a^2*x^2 + 2*b^2*x^2 + b^2*y^2)) / (b*x + a*y) =
  (a*x + b*y)^2 := by sorry

end simplify_expression_l3519_351903


namespace orthocenter_circumcircle_property_l3519_351913

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric operations
variable (orthocenter : Point → Point → Point → Point)
variable (circumcircle : Point → Point → Point → Circle)
variable (circumcenter : Point → Point → Point → Point)
variable (line_intersection : Point → Point → Point → Point → Point)
variable (circle_line_intersection : Circle → Point → Point → Point)
variable (on_circle : Point → Circle → Prop)

-- State the theorem
theorem orthocenter_circumcircle_property
  (A B C H D P Q : Point)
  (ω : Circle)
  (h1 : H = orthocenter A B C)
  (h2 : ω = circumcircle H A B)
  (h3 : D = circle_line_intersection ω B C)
  (h4 : D ≠ B)
  (h5 : P = line_intersection D H A C)
  (h6 : Q = circumcenter A D P) :
  on_circle (circumcenter H A B) (circumcircle B D Q) :=
sorry

end orthocenter_circumcircle_property_l3519_351913


namespace no_xy_term_implies_m_eq_6_l3519_351937

/-- The polynomial that does not contain the xy term -/
def polynomial (x y m : ℝ) : ℝ := x^2 - m*x*y - y^2 + 6*x*y - 1

/-- The coefficient of xy in the polynomial -/
def xy_coefficient (m : ℝ) : ℝ := -m + 6

theorem no_xy_term_implies_m_eq_6 (m : ℝ) :
  (∀ x y : ℝ, polynomial x y m = x^2 - y^2 - 1) → m = 6 :=
by sorry

end no_xy_term_implies_m_eq_6_l3519_351937


namespace games_left_to_play_l3519_351977

/-- Represents a round-robin tournament --/
structure Tournament where
  num_teams : Nat
  total_points : Nat
  lowest_score : Nat
  top_two_equal : Bool

/-- Calculates the total number of matches in a round-robin tournament --/
def total_matches (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the total points that will be distributed in the tournament --/
def total_tournament_points (n : Nat) : Nat :=
  2 * total_matches n

/-- Theorem: In a round-robin tournament with 9 teams, where the total points
    scored is 44, the lowest-scoring team has 1 point, and the top two teams
    have equal points, there are 14 games left to play. --/
theorem games_left_to_play (t : Tournament)
  (h1 : t.num_teams = 9)
  (h2 : t.total_points = 44)
  (h3 : t.lowest_score = 1)
  (h4 : t.top_two_equal = true) :
  total_matches t.num_teams - t.total_points / 2 = 14 := by
  sorry


end games_left_to_play_l3519_351977


namespace qin_jiushao_correct_f_3_equals_22542_l3519_351934

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qin_jiushao (f : ℝ → ℝ) (x : ℝ) : ℝ := 
  let v0 := 1
  let v1 := x * v0 + 2
  let v2 := x * v1 + 0
  let v3 := x * v2 + 4
  let v4 := x * v3 + 5
  let v5 := x * v4 + 6
  x * v5 + 12

/-- The polynomial f(x) = x^6 + 2x^5 + 4x^3 + 5x^2 + 6x + 12 -/
def f (x : ℝ) : ℝ := x^6 + 2*x^5 + 4*x^3 + 5*x^2 + 6*x + 12

/-- Theorem: Qin Jiushao's algorithm correctly evaluates f(3) -/
theorem qin_jiushao_correct : qin_jiushao f 3 = 22542 := by
  sorry

/-- Theorem: f(3) equals 22542 -/
theorem f_3_equals_22542 : f 3 = 22542 := by
  sorry

end qin_jiushao_correct_f_3_equals_22542_l3519_351934


namespace computer_lab_setup_l3519_351945

-- Define the cost of computers and investment range
def standard_teacher_cost : ℕ := 8000
def standard_student_cost : ℕ := 3500
def advanced_teacher_cost : ℕ := 11500
def advanced_student_cost : ℕ := 7000
def min_investment : ℕ := 200000
def max_investment : ℕ := 210000

-- Define the number of student computers in each lab
def standard_students : ℕ := 55
def advanced_students : ℕ := 27

-- Theorem stating the problem
theorem computer_lab_setup :
  (standard_teacher_cost + standard_student_cost * standard_students = 
   advanced_teacher_cost + advanced_student_cost * advanced_students) ∧
  (min_investment < standard_teacher_cost + standard_student_cost * standard_students) ∧
  (standard_teacher_cost + standard_student_cost * standard_students < max_investment) ∧
  (min_investment < advanced_teacher_cost + advanced_student_cost * advanced_students) ∧
  (advanced_teacher_cost + advanced_student_cost * advanced_students < max_investment) := by
  sorry

end computer_lab_setup_l3519_351945


namespace tunnel_length_l3519_351985

/-- Calculates the length of a tunnel given train and journey parameters -/
theorem tunnel_length (train_length : ℝ) (exit_time : ℝ) (train_speed : ℝ) :
  train_length = 2 →
  exit_time = 5 / 60 →
  train_speed = 40 →
  (train_speed * exit_time) - train_length = 4 / 3 := by
  sorry


end tunnel_length_l3519_351985


namespace min_value_theorem_l3519_351996

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + y = 5 * x * y) :
  4 * x + 3 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 * x₀ + y₀ = 5 * x₀ * y₀ ∧ 4 * x₀ + 3 * y₀ = 5 :=
sorry

end min_value_theorem_l3519_351996


namespace smallest_five_digit_divisible_l3519_351979

theorem smallest_five_digit_divisible : ∃ n : ℕ, 
  (10000 ≤ n ∧ n < 100000) ∧ 
  (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ 2 ∣ m ∧ 3 ∣ m ∧ 8 ∣ m ∧ 9 ∣ m → n ≤ m) ∧
  2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n ∧
  n = 10008 := by
  sorry

end smallest_five_digit_divisible_l3519_351979


namespace jane_hector_meeting_l3519_351912

/-- Represents a point on the circular path --/
inductive Point := | A | B | C | D | E

/-- The circular path with its length in blocks --/
def CircularPath := 24

/-- Hector's walking speed (arbitrary units) --/
def HectorSpeed : ℝ := 1

/-- Jane's walking speed in terms of Hector's --/
def JaneSpeed : ℝ := 3 * HectorSpeed

/-- The meeting point of Jane and Hector --/
def MeetingPoint : Point := Point.B

theorem jane_hector_meeting :
  ∀ (t : ℝ),
  t > 0 →
  t * HectorSpeed + t * JaneSpeed = CircularPath →
  MeetingPoint = Point.B :=
sorry

end jane_hector_meeting_l3519_351912


namespace solution_set_correct_l3519_351925

/-- The solution set of the inequality a*x^2 - (a+2)*x + 2 < 0 for x, where a ∈ ℝ -/
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x > 1 }
  else if 0 < a ∧ a < 2 then { x | 1 < x ∧ x < 2/a }
  else if a = 2 then ∅
  else if a > 2 then { x | 2/a < x ∧ x < 1 }
  else { x | x < 2/a ∨ x > 1 }

/-- Theorem stating that the solution_set function correctly describes the solutions of the inequality -/
theorem solution_set_correct (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a*x^2 - (a+2)*x + 2 < 0 :=
by sorry

end solution_set_correct_l3519_351925


namespace cost_function_cheaper_values_l3519_351999

def C (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 30 then 15 * n
  else if 31 ≤ n ∧ n ≤ 65 then 13 * n
  else 11 * n

theorem cost_function_cheaper_values :
  (∃ (S : Finset ℕ), S.card = 6 ∧ 
    (∀ n, n ∈ S ↔ (C (n + 1) < C n ∧ n ≥ 1))) :=
by sorry

end cost_function_cheaper_values_l3519_351999


namespace smallest_integer_l3519_351931

theorem smallest_integer (a b : ℕ+) (h1 : a = 60) (h2 : (Nat.lcm a b) / (Nat.gcd a b) = 44) : 
  b ≥ 165 ∧ ∃ (b' : ℕ+), b' = 165 ∧ (Nat.lcm a b') / (Nat.gcd a b') = 44 := by
  sorry

end smallest_integer_l3519_351931


namespace solve_x_equation_l3519_351924

theorem solve_x_equation (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^2 + 1) (h2 : x / 5 = 5 * y) :
  x = (625 + 25 * Real.sqrt 589) / 6 := by
  sorry

end solve_x_equation_l3519_351924


namespace fraction_to_decimal_l3519_351939

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := by
  sorry

end fraction_to_decimal_l3519_351939


namespace scientific_notation_equivalence_l3519_351949

theorem scientific_notation_equivalence : ∃ (a : ℝ) (n : ℤ), 
  0.00000065 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 6.5 ∧ n = -7 := by
  sorry

end scientific_notation_equivalence_l3519_351949


namespace problem_statement_l3519_351967

theorem problem_statement (P Q : Prop) (h_P : P ↔ (2 + 2 = 5)) (h_Q : Q ↔ (3 > 2)) :
  (P ∨ Q) ∧ ¬(¬Q) := by sorry

end problem_statement_l3519_351967


namespace product_of_divisors_2022_no_prime_power_2022_l3519_351906

def T (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem product_of_divisors_2022 : T 2022 = 2022^4 := by sorry

theorem no_prime_power_2022 : ∀ (n : ℕ) (p : ℕ), Nat.Prime p → T n ≠ p^2022 := by sorry

end product_of_divisors_2022_no_prime_power_2022_l3519_351906


namespace problem_statement_l3519_351991

theorem problem_statement (a b : ℝ) (h : a + b = 1) :
  (a^3 + b^3 ≥ 1/4) ∧
  (∃ x : ℝ, |x - a| + |x - b| ≤ 5 → 0 ≤ 2*a + 3*b ∧ 2*a + 3*b ≤ 5) := by
  sorry

end problem_statement_l3519_351991


namespace tenth_term_of_sequence_l3519_351998

/-- The nth term of a geometric sequence -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The 10th term of the specific geometric sequence -/
theorem tenth_term_of_sequence :
  geometric_sequence 5 (3/4) 10 = 98415/262144 := by
  sorry

end tenth_term_of_sequence_l3519_351998


namespace function_and_range_l3519_351910

def f (a c : ℝ) (x : ℝ) : ℝ := a * x^3 + c * x

theorem function_and_range (a c : ℝ) (h1 : a > 0) :
  (∃ k, (3 * a + c) * k = -1 ∧ k ≠ 0) →
  (∀ x, 3 * a * x^2 + c ≥ -12) →
  (∃ x, 3 * a * x^2 + c = -12) →
  (f a c = fun x ↦ 2 * x^3 - 12 * x) ∧
  (∀ y ∈ Set.Icc (-8 * Real.sqrt 2) (8 * Real.sqrt 2), 
    ∃ x ∈ Set.Icc (-2) 2, f a c x = y) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a c x ∈ Set.Icc (-8 * Real.sqrt 2) (8 * Real.sqrt 2)) :=
by sorry

end function_and_range_l3519_351910


namespace fishing_line_section_length_l3519_351911

theorem fishing_line_section_length 
  (num_reels : ℕ) 
  (reel_length : ℝ) 
  (num_sections : ℕ) 
  (h1 : num_reels = 3) 
  (h2 : reel_length = 100) 
  (h3 : num_sections = 30) : 
  (num_reels * reel_length) / num_sections = 10 := by
  sorry

end fishing_line_section_length_l3519_351911


namespace equilibrium_shift_without_K_change_l3519_351908

-- Define the type for factors that can influence chemical equilibrium
inductive EquilibriumFactor
  | Temperature
  | Concentration
  | Pressure
  | Catalyst

-- Define a function to represent if a factor changes the equilibrium constant
def changesK (factor : EquilibriumFactor) : Prop :=
  match factor with
  | EquilibriumFactor.Temperature => True
  | _ => False

-- Define a function to represent if a factor can shift the equilibrium
def canShiftEquilibrium (factor : EquilibriumFactor) : Prop :=
  match factor with
  | EquilibriumFactor.Temperature => True
  | EquilibriumFactor.Concentration => True
  | EquilibriumFactor.Pressure => True
  | EquilibriumFactor.Catalyst => True

-- Theorem stating that there exists a factor that can shift equilibrium without changing K
theorem equilibrium_shift_without_K_change :
  ∃ (factor : EquilibriumFactor), canShiftEquilibrium factor ∧ ¬changesK factor :=
by
  sorry


end equilibrium_shift_without_K_change_l3519_351908


namespace wire_length_proof_l3519_351961

theorem wire_length_proof (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 4 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 14 := by
sorry

end wire_length_proof_l3519_351961


namespace songs_leftover_l3519_351983

theorem songs_leftover (total_songs : ℕ) (num_playlists : ℕ) (h1 : total_songs = 2048) (h2 : num_playlists = 13) :
  total_songs % num_playlists = 7 := by
  sorry

end songs_leftover_l3519_351983


namespace rachel_furniture_assembly_time_l3519_351948

/-- Calculates the total assembly time for furniture --/
def total_assembly_time (chairs tables bookshelves : ℕ) 
  (chair_time table_time bookshelf_time : ℕ) : ℕ :=
  chairs * chair_time + tables * table_time + bookshelves * bookshelf_time

/-- Proves that the total assembly time for Rachel's furniture is 244 minutes --/
theorem rachel_furniture_assembly_time :
  total_assembly_time 20 8 5 6 8 12 = 244 := by
  sorry

end rachel_furniture_assembly_time_l3519_351948


namespace min_value_on_line_l3519_351947

theorem min_value_on_line (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_line : 2 * a + b = 1) :
  1 / a + 2 / b ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ 1 / a₀ + 2 / b₀ = 8 :=
by sorry

end min_value_on_line_l3519_351947


namespace hyperbola_focal_length_minimum_l3519_351932

theorem hyperbola_focal_length_minimum (a b c : ℝ) : 
  a > 0 → b > 0 → c^2 = a^2 + b^2 → a + b - c = 2 → 2*c ≥ 4 + 4*Real.sqrt 2 := by
  sorry

end hyperbola_focal_length_minimum_l3519_351932


namespace largest_positive_integer_satisfying_condition_l3519_351905

theorem largest_positive_integer_satisfying_condition : 
  ∀ x : ℕ+, x + 1000 > 1000 * x.val → x.val ≤ 1 :=
by
  sorry

end largest_positive_integer_satisfying_condition_l3519_351905


namespace function_inequality_l3519_351915

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, f x > (deriv f) x) : 
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023) := by
  sorry

end function_inequality_l3519_351915


namespace yellow_balls_unchanged_yellow_balls_count_l3519_351956

/-- Represents the contents of a box with colored balls -/
structure BoxContents where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Removes one blue ball from the box -/
def removeOneBlueBall (box : BoxContents) : BoxContents :=
  { box with blue := box.blue - 1 }

/-- Theorem stating that the number of yellow balls remains unchanged after removing a blue ball -/
theorem yellow_balls_unchanged (initialBox : BoxContents) :
  (removeOneBlueBall initialBox).yellow = initialBox.yellow :=
by
  sorry

/-- The main theorem proving that the number of yellow balls remains 5 after removing a blue ball -/
theorem yellow_balls_count (initialBox : BoxContents)
  (h1 : initialBox.red = 3)
  (h2 : initialBox.blue = 2)
  (h3 : initialBox.yellow = 5) :
  (removeOneBlueBall initialBox).yellow = 5 :=
by
  sorry

end yellow_balls_unchanged_yellow_balls_count_l3519_351956


namespace number_of_rattlesnakes_rattlesnakes_count_l3519_351926

/-- The number of rattlesnakes in a park with given conditions -/
theorem number_of_rattlesnakes (total_snakes : ℕ) (boa_constrictors : ℕ) : ℕ :=
  let pythons := 3 * boa_constrictors
  let rattlesnakes := total_snakes - (boa_constrictors + pythons)
  rattlesnakes

/-- Proof that the number of rattlesnakes is 40 given the conditions -/
theorem rattlesnakes_count :
  number_of_rattlesnakes 200 40 = 40 := by
  sorry

end number_of_rattlesnakes_rattlesnakes_count_l3519_351926


namespace tomato_plant_problem_l3519_351969

theorem tomato_plant_problem (initial_tomatoes : ℕ) : 
  (initial_tomatoes : ℚ) - (1/4 * initial_tomatoes + 20 + 40 : ℚ) = 15 → 
  initial_tomatoes = 100 := by
sorry

end tomato_plant_problem_l3519_351969


namespace even_function_implies_a_zero_l3519_351989

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^2 + ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 1

theorem even_function_implies_a_zero :
  ∀ a : ℝ, IsEven (f a) → a = 0 := by
  sorry

end even_function_implies_a_zero_l3519_351989


namespace robert_coin_arrangements_l3519_351916

/-- Represents the number of distinguishable arrangements of coins -/
def coin_arrangements (gold_coins silver_coins : Nat) : Nat :=
  let total_coins := gold_coins + silver_coins
  let positions := Nat.choose total_coins gold_coins
  let orientations := 30  -- Simplified representation of valid orientations
  positions * orientations

/-- Theorem stating the number of distinguishable arrangements for the given problem -/
theorem robert_coin_arrangements :
  coin_arrangements 5 3 = 1680 := by
  sorry

end robert_coin_arrangements_l3519_351916


namespace product_digit_sum_l3519_351923

/-- The number of digits in the second factor of the product (9)(999...9) -/
def k : ℕ := sorry

/-- The sum of digits in the resulting integer -/
def digit_sum : ℕ := 1009

/-- The resulting integer from the product (9)(999...9) -/
def result : ℕ := 10^k - 1

theorem product_digit_sum : 
  (∀ n : ℕ, n ≤ k → (result / 10^n) % 10 = 9) ∧ 
  (result % 10 = 9) ∧
  (digit_sum = 9 * k) ∧
  (k = 112) :=
sorry

end product_digit_sum_l3519_351923


namespace adjacent_even_sum_l3519_351972

/-- A circular arrangement of seven natural numbers -/
def CircularArrangement := Fin 7 → ℕ

/-- Two numbers in a circular arrangement are adjacent if their indices differ by 1 (mod 7) -/
def adjacent (arr : CircularArrangement) (i j : Fin 7) : Prop :=
  (i.val + 1) % 7 = j.val ∨ (j.val + 1) % 7 = i.val

/-- The main theorem: In any circular arrangement of seven natural numbers,
    there exist two adjacent numbers whose sum is even -/
theorem adjacent_even_sum (arr : CircularArrangement) :
  ∃ (i j : Fin 7), adjacent arr i j ∧ Even (arr i + arr j) := by
  sorry

end adjacent_even_sum_l3519_351972


namespace string_length_problem_l3519_351962

theorem string_length_problem (total_strings : ℕ) (total_avg : ℝ) (subset_strings : ℕ) (subset_avg : ℝ) :
  total_strings = 6 →
  total_avg = 80 →
  subset_strings = 2 →
  subset_avg = 70 →
  let remaining_strings := total_strings - subset_strings
  let total_length := total_strings * total_avg
  let subset_length := subset_strings * subset_avg
  let remaining_length := total_length - subset_length
  (remaining_length / remaining_strings) = 85 := by
sorry

end string_length_problem_l3519_351962


namespace plot_area_in_acres_l3519_351953

-- Define the triangle dimensions
def leg1 : ℝ := 8
def leg2 : ℝ := 6

-- Define scale and conversion factors
def scale : ℝ := 3  -- 1 cm = 3 miles
def acres_per_square_mile : ℝ := 640

-- Define the theorem
theorem plot_area_in_acres :
  let triangle_area := (1/2) * leg1 * leg2
  let scaled_area := triangle_area * scale * scale
  let area_in_acres := scaled_area * acres_per_square_mile
  area_in_acres = 138240 := by sorry

end plot_area_in_acres_l3519_351953


namespace malt_shop_syrup_usage_l3519_351958

/-- Calculates the total syrup used in a malt shop given specific conditions -/
theorem malt_shop_syrup_usage
  (syrup_per_shake : ℝ)
  (syrup_per_cone : ℝ)
  (syrup_per_sundae : ℝ)
  (extra_syrup : ℝ)
  (extra_syrup_percentage : ℝ)
  (num_shakes : ℕ)
  (num_cones : ℕ)
  (num_sundaes : ℕ)
  (h1 : syrup_per_shake = 5.5)
  (h2 : syrup_per_cone = 8)
  (h3 : syrup_per_sundae = 4.2)
  (h4 : extra_syrup = 0.3)
  (h5 : extra_syrup_percentage = 0.1)
  (h6 : num_shakes = 5)
  (h7 : num_cones = 4)
  (h8 : num_sundaes = 3) :
  ∃ total_syrup : ℝ,
    total_syrup = num_shakes * syrup_per_shake +
                  num_cones * syrup_per_cone +
                  num_sundaes * syrup_per_sundae +
                  (↑(round ((num_shakes + num_cones) * extra_syrup_percentage)) * extra_syrup) ∧
    total_syrup = 72.4 := by
  sorry

end malt_shop_syrup_usage_l3519_351958


namespace floor_rate_per_square_meter_l3519_351993

/-- Given a rectangular room with length 8 m and width 4.75 m, and a total flooring cost of Rs. 34,200, the rate per square meter is Rs. 900. -/
theorem floor_rate_per_square_meter :
  let length : ℝ := 8
  let width : ℝ := 4.75
  let total_cost : ℝ := 34200
  let area : ℝ := length * width
  let rate_per_sq_meter : ℝ := total_cost / area
  rate_per_sq_meter = 900 := by sorry

end floor_rate_per_square_meter_l3519_351993


namespace radio_survey_males_not_listening_l3519_351942

theorem radio_survey_males_not_listening (males_listening : ℕ) 
  (females_not_listening : ℕ) (total_listening : ℕ) (total_not_listening : ℕ) 
  (h1 : males_listening = 70)
  (h2 : females_not_listening = 110)
  (h3 : total_listening = 145)
  (h4 : total_not_listening = 160) :
  total_not_listening - females_not_listening = 50 :=
by
  sorry

#check radio_survey_males_not_listening

end radio_survey_males_not_listening_l3519_351942


namespace total_books_l3519_351917

theorem total_books (joan_books tom_books : ℕ) 
  (h1 : joan_books = 10) 
  (h2 : tom_books = 38) : 
  joan_books + tom_books = 48 := by
  sorry

end total_books_l3519_351917


namespace factorization_proof_l3519_351984

theorem factorization_proof (a x y : ℝ) : a * x - a * y = a * (x - y) := by
  sorry

end factorization_proof_l3519_351984


namespace muffin_to_banana_ratio_l3519_351907

/-- The cost of a muffin -/
def muffin_cost : ℝ := sorry

/-- The cost of a banana -/
def banana_cost : ℝ := sorry

/-- Kristy's total cost -/
def kristy_cost : ℝ := 5 * muffin_cost + 4 * banana_cost

/-- Tim's total cost -/
def tim_cost : ℝ := 3 * muffin_cost + 20 * banana_cost

/-- The theorem stating the ratio of muffin cost to banana cost -/
theorem muffin_to_banana_ratio :
  tim_cost = 3 * kristy_cost →
  muffin_cost = (2/3) * banana_cost :=
by sorry

end muffin_to_banana_ratio_l3519_351907


namespace total_notes_count_l3519_351902

theorem total_notes_count (total_amount : ℕ) (note_50_count : ℕ) (note_50_value : ℕ) (note_500_value : ℕ) 
  (h1 : total_amount = 10350)
  (h2 : note_50_count = 97)
  (h3 : note_50_value = 50)
  (h4 : note_500_value = 500)
  (h5 : ∃ (note_500_count : ℕ), total_amount = note_50_count * note_50_value + note_500_count * note_500_value) :
  ∃ (total_notes : ℕ), total_notes = note_50_count + (total_amount - note_50_count * note_50_value) / note_500_value ∧ total_notes = 108 := by
sorry

end total_notes_count_l3519_351902


namespace real_part_of_complex_fraction_l3519_351955

theorem real_part_of_complex_fraction (i : ℂ) : 
  i * i = -1 → Complex.re ((1 + i) / (1 - i)) = 0 := by
  sorry

end real_part_of_complex_fraction_l3519_351955


namespace power_fraction_simplification_l3519_351922

theorem power_fraction_simplification :
  (3^2023 + 3^2021) / (3^2023 - 3^2021) = 5/4 := by
  sorry

end power_fraction_simplification_l3519_351922


namespace no_right_triangle_with_given_conditions_l3519_351920

theorem no_right_triangle_with_given_conditions :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b = 8 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
sorry

end no_right_triangle_with_given_conditions_l3519_351920


namespace sum_of_consecutive_integers_l3519_351941

theorem sum_of_consecutive_integers (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 22 → n = 4 := by
  sorry

end sum_of_consecutive_integers_l3519_351941


namespace triangle_properties_l3519_351929

-- Define the triangle ABC
def A : ℝ × ℝ := (-1, 4)
def B : ℝ × ℝ := (-2, -1)
def C : ℝ × ℝ := (2, 3)

-- Define the height line from B to AC
def height_line (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the area of the triangle
def triangle_area : ℝ := 8

-- Theorem statement
theorem triangle_properties :
  (∀ x y : ℝ, height_line x y ↔ (x + y - 3 = 0)) ∧
  triangle_area = 8 := by
  sorry

end triangle_properties_l3519_351929


namespace weight_replacement_l3519_351928

theorem weight_replacement (initial_count : ℕ) (weight_increase : ℚ) (new_weight : ℚ) :
  initial_count = 8 →
  weight_increase = 5/2 →
  new_weight = 40 →
  ∃ (old_weight : ℚ),
    old_weight = new_weight - (initial_count * weight_increase) ∧
    old_weight = 20 := by
  sorry

end weight_replacement_l3519_351928


namespace profit_percentage_calculation_l3519_351933

theorem profit_percentage_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.96 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100 / 96 - 1) * 100 := by
  sorry

end profit_percentage_calculation_l3519_351933


namespace fourth_month_sale_l3519_351959

def average_sale : ℕ := 6600
def num_months : ℕ := 6
def sale_month1 : ℕ := 6435
def sale_month2 : ℕ := 6927
def sale_month3 : ℕ := 6855
def sale_month5 : ℕ := 6562
def sale_month6 : ℕ := 5591

theorem fourth_month_sale (x : ℕ) : 
  (sale_month1 + sale_month2 + sale_month3 + x + sale_month5 + sale_month6) / num_months = average_sale →
  x = 7230 := by
  sorry

end fourth_month_sale_l3519_351959


namespace robin_total_pieces_l3519_351978

/-- The number of pieces in a package of Type A gum -/
def type_a_gum_pieces : ℕ := 4

/-- The number of pieces in a package of Type B gum -/
def type_b_gum_pieces : ℕ := 8

/-- The number of pieces in a package of Type C gum -/
def type_c_gum_pieces : ℕ := 12

/-- The number of pieces in a package of Type X candy -/
def type_x_candy_pieces : ℕ := 6

/-- The number of pieces in a package of Type Y candy -/
def type_y_candy_pieces : ℕ := 10

/-- The number of packages of Type A gum Robin has -/
def robin_type_a_gum_packages : ℕ := 10

/-- The number of packages of Type B gum Robin has -/
def robin_type_b_gum_packages : ℕ := 5

/-- The number of packages of Type C gum Robin has -/
def robin_type_c_gum_packages : ℕ := 13

/-- The number of packages of Type X candy Robin has -/
def robin_type_x_candy_packages : ℕ := 8

/-- The number of packages of Type Y candy Robin has -/
def robin_type_y_candy_packages : ℕ := 6

/-- The total number of gum packages Robin has -/
def robin_total_gum_packages : ℕ := 28

/-- The total number of candy packages Robin has -/
def robin_total_candy_packages : ℕ := 14

theorem robin_total_pieces : 
  robin_type_a_gum_packages * type_a_gum_pieces +
  robin_type_b_gum_packages * type_b_gum_pieces +
  robin_type_c_gum_packages * type_c_gum_pieces +
  robin_type_x_candy_packages * type_x_candy_pieces +
  robin_type_y_candy_packages * type_y_candy_pieces = 344 :=
by sorry

end robin_total_pieces_l3519_351978


namespace sum_xyz_equals_zero_l3519_351951

theorem sum_xyz_equals_zero 
  (x y z a b c : ℝ) 
  (eq1 : x + y - z = a - b)
  (eq2 : x - y + z = b - c)
  (eq3 : -x + y + z = c - a) : 
  x + y + z = 0 := by
sorry

end sum_xyz_equals_zero_l3519_351951


namespace f_min_value_a_range_l3519_351990

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ m = 5 := by sorry

-- Theorem for the range of a
theorem a_range (x : ℝ) (hx : x ∈ Set.Icc (-3) 2) :
  (∀ a, f x ≥ |x + a|) ↔ a ∈ Set.Icc (-2) 3 := by sorry

end f_min_value_a_range_l3519_351990


namespace seating_arrangement_theorem_l3519_351976

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n people in a row where two specific people sit together -/
def arrangementsWithTwoTogether (n : ℕ) : ℕ := (n - 1).factorial * 2

/-- The number of people -/
def numberOfPeople : ℕ := 4

/-- The number of valid seating arrangements -/
def validArrangements : ℕ := totalArrangements numberOfPeople - arrangementsWithTwoTogether numberOfPeople

theorem seating_arrangement_theorem :
  validArrangements = 12 := by sorry

end seating_arrangement_theorem_l3519_351976


namespace trapezoid_height_l3519_351994

/-- Represents a trapezoid with height x, bases 3x and 5x, and area 40 -/
structure Trapezoid where
  x : ℝ
  base1 : ℝ := 3 * x
  base2 : ℝ := 5 * x
  area : ℝ := 40

/-- The height of a trapezoid with the given properties is √10 -/
theorem trapezoid_height (t : Trapezoid) : t.x = Real.sqrt 10 := by
  sorry

#check trapezoid_height

end trapezoid_height_l3519_351994


namespace product_of_ab_l3519_351997

theorem product_of_ab (a b : ℝ) (h1 : a^5 * b^8 = 12) (h2 : a^8 * b^13 = 18) : a * b = 128/3 := by
  sorry

end product_of_ab_l3519_351997


namespace square_triangle_equal_area_l3519_351981

/-- Given a square with perimeter 60 and a right triangle with one leg 20,
    if their areas are equal, then the other leg of the triangle is 22.5 -/
theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_leg : ℝ) (other_leg : ℝ) :
  square_perimeter = 60 →
  triangle_leg = 20 →
  (square_perimeter / 4) ^ 2 = (triangle_leg * other_leg) / 2 →
  other_leg = 22.5 := by
  sorry

end square_triangle_equal_area_l3519_351981


namespace isosceles_triangle_perimeter_l3519_351938

/-- An isosceles triangle with side lengths 3, 6, and 6 has a perimeter of 15. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 6 ∧ c = 6 →  -- Two sides are 6, one side is 3
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a + b + c = 15  -- Perimeter is 15
:= by sorry

end isosceles_triangle_perimeter_l3519_351938


namespace complement_of_A_l3519_351971

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

theorem complement_of_A : 
  Set.compl A = Set.Icc (-1 : ℝ) 3 := by sorry

end complement_of_A_l3519_351971


namespace trigonometric_identity_l3519_351963

theorem trigonometric_identity (α : Real) 
  (h1 : π < α ∧ α < 2*π) 
  (h2 : Real.cos (α - 7*π) = -3/5) : 
  Real.sin (3*π + α) * Real.tan (α - 7*π/2) = 3/5 := by
  sorry

end trigonometric_identity_l3519_351963


namespace number42_does_not_contain_5_l3519_351904

/-- Represents a five-digit rising number -/
structure RisingNumber :=
  (d1 d2 d3 d4 d5 : Nat)
  (h1 : d1 < d2)
  (h2 : d2 < d3)
  (h3 : d3 < d4)
  (h4 : d4 < d5)
  (h5 : 1 ≤ d1 ∧ d5 ≤ 8)

/-- The list of all valid rising numbers -/
def risingNumbers : List RisingNumber := sorry

/-- The 42nd number in the sorted list of rising numbers -/
def number42 : RisingNumber := sorry

/-- Theorem stating that the 42nd rising number does not contain 5 -/
theorem number42_does_not_contain_5 : 
  number42.d1 ≠ 5 ∧ number42.d2 ≠ 5 ∧ number42.d3 ≠ 5 ∧ number42.d4 ≠ 5 ∧ number42.d5 ≠ 5 := by
  sorry

end number42_does_not_contain_5_l3519_351904


namespace prove_movie_theatre_seats_l3519_351927

def movie_theatre_seats (adult_price child_price : ℕ) (num_children total_revenue : ℕ) : Prop :=
  let total_seats := num_children + (total_revenue - num_children * child_price) / adult_price
  total_seats = 250 ∧
  adult_price * (total_seats - num_children) + child_price * num_children = total_revenue

theorem prove_movie_theatre_seats :
  movie_theatre_seats 6 4 188 1124 := by
  sorry

end prove_movie_theatre_seats_l3519_351927


namespace cylinder_surface_area_l3519_351909

/-- The surface area of a cylinder with diameter and height both equal to 4 is 24π. -/
theorem cylinder_surface_area : 
  ∀ (d h : ℝ), d = 4 → h = 4 → 2 * π * (d / 2) * (d / 2 + h) = 24 * π :=
by
  sorry

end cylinder_surface_area_l3519_351909


namespace product_distribution_l3519_351940

theorem product_distribution (n : ℕ) (h : n = 6) :
  (Nat.choose n 1) * (Nat.choose (n - 1) 2) * (Nat.choose (n - 3) 3) =
  (Nat.choose n 1) * (Nat.choose (n - 1) 2) * (Nat.choose (n - 3) 3) :=
by sorry

end product_distribution_l3519_351940


namespace function_domain_range_equality_l3519_351992

theorem function_domain_range_equality (a : ℝ) (h1 : a > 1) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*a*x + 5
  (∀ x, f x ∈ Set.Icc 1 a ↔ x ∈ Set.Icc 1 a) → a = 2 := by
  sorry

end function_domain_range_equality_l3519_351992


namespace school_play_ticket_value_l3519_351982

/-- Calculates the total value of tickets sold for a school play --/
def total_ticket_value (student_price : ℕ) (adult_price : ℕ) (child_price : ℕ) (senior_price : ℕ)
                       (student_count : ℕ) (adult_count : ℕ) (child_count : ℕ) (senior_count : ℕ) : ℕ :=
  student_price * student_count + adult_price * adult_count + child_price * child_count + senior_price * senior_count

theorem school_play_ticket_value :
  total_ticket_value 6 8 4 7 20 12 15 10 = 346 := by
  sorry

end school_play_ticket_value_l3519_351982


namespace expression_simplification_l3519_351901

theorem expression_simplification (x : ℝ) : 
  3*x*(3*x^2 - 2*x + 1) - 2*x^2 + x = 9*x^3 - 8*x^2 + 4*x := by
  sorry

end expression_simplification_l3519_351901


namespace chess_tournament_principled_trios_l3519_351987

/-- Represents the number of chess players in the tournament -/
def n : ℕ := 2017

/-- Defines a principled trio of chess players -/
def PrincipledTrio (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A ≤ n ∧ B ≤ n ∧ C ≤ n

/-- Calculates the maximum number of principled trios for an odd number of players -/
def max_principled_trios_odd (k : ℕ) : ℕ := (k^3 - k) / 24

/-- The maximum number of principled trios in the tournament -/
def max_principled_trios : ℕ := max_principled_trios_odd n

theorem chess_tournament_principled_trios :
  max_principled_trios = 341606288 :=
sorry

end chess_tournament_principled_trios_l3519_351987


namespace joe_trip_theorem_l3519_351919

/-- Represents Joe's trip expenses and calculations -/
def joe_trip (exchange_rate : ℝ) (initial_savings flight hotel food transportation entertainment miscellaneous : ℝ) : Prop :=
  let total_savings_aud := initial_savings * exchange_rate
  let total_expenses_usd := flight + hotel + food + transportation + entertainment + miscellaneous
  let total_expenses_aud := total_expenses_usd * exchange_rate
  let amount_left := total_savings_aud - total_expenses_aud
  (total_expenses_aud = 9045) ∧ (amount_left = -945)

/-- Theorem stating the correctness of Joe's trip calculations -/
theorem joe_trip_theorem : 
  joe_trip 1.35 6000 1200 800 3000 500 850 350 := by
  sorry

end joe_trip_theorem_l3519_351919


namespace parents_in_program_l3519_351946

theorem parents_in_program (total_people : ℕ) (pupils : ℕ) (h1 : total_people = 803) (h2 : pupils = 698) :
  total_people - pupils = 105 := by
  sorry

end parents_in_program_l3519_351946


namespace intersection_complement_equality_l3519_351964

def A : Set ℕ := {x | (x + 4) * (x - 5) ≤ 0}
def B : Set ℕ := {x | x < 2}
def U : Set ℕ := Set.univ

theorem intersection_complement_equality :
  A ∩ (U \ B) = {2, 3, 4, 5} := by sorry

end intersection_complement_equality_l3519_351964


namespace min_max_theorem_l3519_351952

theorem min_max_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1/x + 1/y ≥ 2) ∧ (x * (y + 1) ≤ 9/4) := by
  sorry

end min_max_theorem_l3519_351952


namespace sufficient_condition_product_greater_than_one_l3519_351943

theorem sufficient_condition_product_greater_than_one :
  ∀ a b : ℝ, a > 1 → b > 1 → a * b > 1 := by sorry

end sufficient_condition_product_greater_than_one_l3519_351943


namespace roots_when_p_is_8_p_value_when_root_is_3_plus_4i_l3519_351954

-- Define the complex quadratic equation
def complex_quadratic (p : ℝ) (x : ℂ) : ℂ := x^2 - p*x + 25

-- Part 1: Prove that when p = 8, the roots are 4 + 3i and 4 - 3i
theorem roots_when_p_is_8 :
  let p : ℝ := 8
  let x₁ : ℂ := 4 + 3*I
  let x₂ : ℂ := 4 - 3*I
  complex_quadratic p x₁ = 0 ∧ complex_quadratic p x₂ = 0 :=
sorry

-- Part 2: Prove that when one root is 3 + 4i, p = 6
theorem p_value_when_root_is_3_plus_4i :
  let x₁ : ℂ := 3 + 4*I
  ∃ p : ℝ, complex_quadratic p x₁ = 0 ∧ p = 6 :=
sorry

end roots_when_p_is_8_p_value_when_root_is_3_plus_4i_l3519_351954


namespace bike_ride_speed_l3519_351965

theorem bike_ride_speed (joann_speed joann_time fran_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 2.5) :
  (joann_speed * joann_time) / fran_time = 24 := by
  sorry

end bike_ride_speed_l3519_351965


namespace trig_sum_zero_l3519_351918

theorem trig_sum_zero (θ : ℝ) (a : ℝ) (h : Real.cos (π / 6 - θ) = a) :
  Real.cos (5 * π / 6 + θ) + Real.sin (2 * π / 3 - θ) = 0 := by
  sorry

end trig_sum_zero_l3519_351918


namespace distribution_six_twelve_l3519_351970

/-- The number of ways to distribute distinct items among recipients --/
def distribution_ways (num_items : ℕ) (num_recipients : ℕ) : ℕ :=
  num_recipients ^ num_items

/-- Theorem: The number of ways to distribute 6 distinct items among 12 recipients is 2,985,984 --/
theorem distribution_six_twelve : distribution_ways 6 12 = 2985984 := by
  sorry

end distribution_six_twelve_l3519_351970


namespace even_factors_count_l3519_351966

def n : ℕ := 2^3 * 3^2 * 7^3 * 5^1

/-- The number of even natural-number factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem even_factors_count : num_even_factors n = 72 := by sorry

end even_factors_count_l3519_351966


namespace unique_divisor_product_100_l3519_351944

/-- Product of all divisors of a natural number -/
def divisor_product (n : ℕ) : ℕ := sorry

/-- Theorem stating that 100 is the only natural number whose divisor product is 10^9 -/
theorem unique_divisor_product_100 :
  ∀ n : ℕ, divisor_product n = 10^9 ↔ n = 100 := by sorry

end unique_divisor_product_100_l3519_351944


namespace inverse_f_at_4_equals_2_l3519_351980

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem inverse_f_at_4_equals_2 :
  ∃ (f_inv : ℝ → ℝ), (∀ x > 0, f_inv (f x) = x) ∧ (f_inv 4 = 2) := by
  sorry

end inverse_f_at_4_equals_2_l3519_351980


namespace point_coordinates_wrt_origin_l3519_351988

/-- The coordinates of a point with respect to the origin are the same as its Cartesian coordinates. -/
theorem point_coordinates_wrt_origin (x y : ℝ) :
  let p : ℝ × ℝ := (x, y)
  p = p := by sorry

end point_coordinates_wrt_origin_l3519_351988


namespace trajectory_of_point_l3519_351974

/-- The trajectory of a point M(x, y) satisfying the given distance condition -/
theorem trajectory_of_point (x y : ℝ) :
  (((x - 2)^2 + y^2).sqrt = |x + 3| - 1) →
  (y^2 = 8 * (x + 2)) := by
  sorry

end trajectory_of_point_l3519_351974


namespace uniform_transform_l3519_351957

-- Define a uniform random variable on an interval
def UniformRandom (a b : ℝ) := {X : ℝ → ℝ | ∀ x, a ≤ x ∧ x ≤ b → X x = (b - a)⁻¹}

theorem uniform_transform (b₁ b : ℝ → ℝ) :
  UniformRandom 0 1 b₁ →
  (∀ x, b x = 3 * (b₁ x - 2)) →
  UniformRandom (-6) (-3) b := by
sorry

end uniform_transform_l3519_351957


namespace factor_tree_problem_l3519_351930

theorem factor_tree_problem (X Y Z F G : ℕ) : 
  X = Y * Z ∧ 
  Y = 7 * F ∧ 
  Z = 11 * G ∧ 
  F = 2 * 5 ∧ 
  G = 7 * 3 → 
  X = 16170 := by
sorry


end factor_tree_problem_l3519_351930


namespace min_people_for_valid_seating_l3519_351995

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone. -/
def valid_seating (table : CircularTable) : Prop :=
  ∀ i : ℕ, i < table.total_chairs → ∃ j : ℕ, j < table.total_chairs ∧ 
    (((i + 1) % table.total_chairs = j) ∨ ((i + table.total_chairs - 1) % table.total_chairs = j))

/-- The main theorem stating the minimum number of people required for a valid seating arrangement. -/
theorem min_people_for_valid_seating :
  ∃ (table : CircularTable), table.total_chairs = 100 ∧ 
    valid_seating table ∧ table.seated_people = 25 ∧
    (∀ (smaller_table : CircularTable), smaller_table.total_chairs = 100 → 
      valid_seating smaller_table → smaller_table.seated_people ≥ 25) :=
sorry

end min_people_for_valid_seating_l3519_351995


namespace f_g_3_equals_95_l3519_351950

def f (x : ℝ) : ℝ := 4 * x - 5

def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem f_g_3_equals_95 : f (g 3) = 95 := by
  sorry

end f_g_3_equals_95_l3519_351950
