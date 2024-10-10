import Mathlib

namespace hockey_tournament_games_l329_32933

/-- The number of teams in the hockey league --/
def num_teams : ℕ := 7

/-- The number of times each team plays against every other team --/
def games_per_matchup : ℕ := 4

/-- The total number of games played in the tournament --/
def total_games : ℕ := num_teams * (num_teams - 1) / 2 * games_per_matchup

theorem hockey_tournament_games :
  total_games = 84 :=
by sorry

end hockey_tournament_games_l329_32933


namespace solve_prize_problem_l329_32991

def prize_problem (x y m n w : ℝ) : Prop :=
  x + 2*y = 40 ∧
  2*x + 3*y = 70 ∧
  m + n = 60 ∧
  m ≥ n/2 ∧
  w = m*x + n*y

theorem solve_prize_problem :
  ∀ x y m n w,
  prize_problem x y m n w →
  (x = 20 ∧ y = 10) ∧
  (∀ m' n' w',
    prize_problem x y m' n' w' →
    w ≤ w') ∧
  (m = 20 ∧ n = 40 ∧ w = 800) :=
by sorry

end solve_prize_problem_l329_32991


namespace parallelogram_base_l329_32928

/-- Given a parallelogram with area 180 square centimeters and height 10 cm, its base is 18 cm. -/
theorem parallelogram_base (area height base : ℝ) : 
  area = 180 ∧ height = 10 ∧ area = base * height → base = 18 := by
  sorry

end parallelogram_base_l329_32928


namespace max_area_specific_quadrilateral_l329_32931

/-- A convex quadrilateral with given side lengths -/
structure ConvexQuadrilateral where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  convex : ab > 0 ∧ bc > 0 ∧ cd > 0 ∧ da > 0

/-- The area of a convex quadrilateral -/
def area (q : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem: Maximum area of a specific convex quadrilateral -/
theorem max_area_specific_quadrilateral :
  ∃ (q : ConvexQuadrilateral),
    q.ab = 2 ∧ q.bc = 4 ∧ q.cd = 5 ∧ q.da = 3 ∧
    ∀ (q' : ConvexQuadrilateral),
      q'.ab = 2 → q'.bc = 4 → q'.cd = 5 → q'.da = 3 →
      area q' ≤ 2 * Real.sqrt 30 :=
  sorry

end max_area_specific_quadrilateral_l329_32931


namespace cl_ab_ratio_l329_32990

/-- A regular pentagon with specific points and angle conditions -/
structure RegularPentagonWithPoints where
  /-- The side length of the regular pentagon -/
  s : ℝ
  /-- Point K on side AE -/
  k : ℝ
  /-- Point L on side CD -/
  l : ℝ
  /-- The sum of angles LAE and KCD is 108° -/
  angle_sum : k + l = 108
  /-- The ratio of AK to KE is 3:7 -/
  length_ratio : k / (s - k) = 3 / 7
  /-- The side length is positive -/
  s_pos : s > 0
  /-- K is between A and E -/
  k_between : 0 < k ∧ k < s
  /-- L is between C and D -/
  l_between : 0 < l ∧ l < s

/-- The theorem stating the ratio of CL to AB in the given pentagon -/
theorem cl_ab_ratio (p : RegularPentagonWithPoints) : (p.s - p.l) / p.s = 0.7 := by
  sorry

end cl_ab_ratio_l329_32990


namespace intersection_A_B_union_complement_B_P_l329_32959

-- Define the sets A, B, and P
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5/2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Theorem for (ᶜB) ∪ P
theorem union_complement_B_P : (Bᶜ : Set ℝ) ∪ P = {x : ℝ | x ≤ 0 ∨ x ≥ 5/2} := by sorry

end intersection_A_B_union_complement_B_P_l329_32959


namespace intersection_point_l329_32918

/-- Two lines in a 2D plane -/
structure TwoLines where
  line1 : ℝ → ℝ → Prop
  line2 : ℝ → ℝ → Prop

/-- The given two lines -/
def givenLines : TwoLines where
  line1 := fun x y => x - y = 0
  line2 := fun x y => 3 * x + 2 * y - 5 = 0

/-- Theorem: The point (1, 1) is the unique intersection of the given lines -/
theorem intersection_point (l : TwoLines := givenLines) :
  (∃! p : ℝ × ℝ, l.line1 p.1 p.2 ∧ l.line2 p.1 p.2) ∧
  (l.line1 1 1 ∧ l.line2 1 1) :=
sorry

end intersection_point_l329_32918


namespace part_a_part_b_l329_32910

-- Define a type for our set of 100 positive numbers
def PositiveSet := Fin 100 → ℝ

-- Define the property that all numbers in the set are positive
def AllPositive (s : PositiveSet) : Prop :=
  ∀ i, s i > 0

-- Define the property that the sum of any 7 numbers is less than 7
def SumOfSevenLessThanSeven (s : PositiveSet) : Prop :=
  ∀ (i₁ i₂ i₃ i₄ i₅ i₆ i₇ : Fin 100),
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₁ ≠ i₅ ∧ i₁ ≠ i₆ ∧ i₁ ≠ i₇ ∧
    i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₂ ≠ i₅ ∧ i₂ ≠ i₆ ∧ i₂ ≠ i₇ ∧
    i₃ ≠ i₄ ∧ i₃ ≠ i₅ ∧ i₃ ≠ i₆ ∧ i₃ ≠ i₇ ∧
    i₄ ≠ i₅ ∧ i₄ ≠ i₆ ∧ i₄ ≠ i₇ ∧
    i₅ ≠ i₆ ∧ i₅ ≠ i₇ ∧
    i₆ ≠ i₇ →
    s i₁ + s i₂ + s i₃ + s i₄ + s i₅ + s i₆ + s i₇ < 7

-- Define the property that the sum of any 10 numbers is less than 10
def SumOfTenLessThanTen (s : PositiveSet) : Prop :=
  ∀ (i₁ i₂ i₃ i₄ i₅ i₆ i₇ i₈ i₉ i₁₀ : Fin 100),
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₁ ≠ i₅ ∧ i₁ ≠ i₆ ∧ i₁ ≠ i₇ ∧ i₁ ≠ i₈ ∧ i₁ ≠ i₉ ∧ i₁ ≠ i₁₀ ∧
    i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₂ ≠ i₅ ∧ i₂ ≠ i₆ ∧ i₂ ≠ i₇ ∧ i₂ ≠ i₈ ∧ i₂ ≠ i₉ ∧ i₂ ≠ i₁₀ ∧
    i₃ ≠ i₄ ∧ i₃ ≠ i₅ ∧ i₃ ≠ i₆ ∧ i₃ ≠ i₇ ∧ i₃ ≠ i₈ ∧ i₃ ≠ i₉ ∧ i₃ ≠ i₁₀ ∧
    i₄ ≠ i₅ ∧ i₄ ≠ i₆ ∧ i₄ ≠ i₇ ∧ i₄ ≠ i₈ ∧ i₄ ≠ i₉ ∧ i₄ ≠ i₁₀ ∧
    i₅ ≠ i₆ ∧ i₅ ≠ i₇ ∧ i₅ ≠ i₈ ∧ i₅ ≠ i₉ ∧ i₅ ≠ i₁₀ ∧
    i₆ ≠ i₇ ∧ i₆ ≠ i₈ ∧ i₆ ≠ i₉ ∧ i₆ ≠ i₁₀ ∧
    i₇ ≠ i₈ ∧ i₇ ≠ i₉ ∧ i₇ ≠ i₁₀ ∧
    i₈ ≠ i₉ ∧ i₈ ≠ i₁₀ ∧
    i₉ ≠ i₁₀ →
    s i₁ + s i₂ + s i₃ + s i₄ + s i₅ + s i₆ + s i₇ + s i₈ + s i₉ + s i₁₀ < 10

-- Theorem for part (a)
theorem part_a (s : PositiveSet) (h₁ : AllPositive s) (h₂ : SumOfSevenLessThanSeven s) :
  SumOfTenLessThanTen s := by
  sorry

-- Theorem for part (b)
theorem part_b :
  ¬∀ (s : PositiveSet), AllPositive s → SumOfTenLessThanTen s → SumOfSevenLessThanSeven s := by
  sorry

end part_a_part_b_l329_32910


namespace inequality_solution_implies_a_range_l329_32902

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 := by
  sorry

end inequality_solution_implies_a_range_l329_32902


namespace connie_marbles_proof_l329_32901

/-- The number of marbles Connie started with -/
def initial_marbles : ℕ := 776

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 183

/-- The number of marbles Connie has left -/
def remaining_marbles : ℕ := initial_marbles - marbles_given

theorem connie_marbles_proof : remaining_marbles = 593 := by
  sorry

end connie_marbles_proof_l329_32901


namespace divisible_by_eight_inductive_step_l329_32960

theorem divisible_by_eight (n : ℕ) :
  ∃ m : ℤ, 3^(4*n+1) + 5^(2*n+1) = 8 * m :=
by
  sorry

theorem inductive_step (k : ℕ) :
  3^(4*(k+1)+1) + 5^(2*(k+1)+1) = 56 * 3^(4*k+1) + 25 * (3^(4*k+1) + 5^(2*k+1)) :=
by
  sorry

end divisible_by_eight_inductive_step_l329_32960


namespace fraction_domain_l329_32966

theorem fraction_domain (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) → x ≠ 1 := by
  sorry

end fraction_domain_l329_32966


namespace playground_children_l329_32970

/-- The number of children on a playground given the number of boys and girls -/
theorem playground_children (boys girls : ℕ) (h1 : boys = 40) (h2 : girls = 77) :
  boys + girls = 117 := by
  sorry

end playground_children_l329_32970


namespace third_circle_radius_l329_32980

/-- Given three circles with the specified properties, the radius of the third circle is 15/14 -/
theorem third_circle_radius (r1 r2 r3 : ℝ) : 
  r1 = 2 →  -- Circle 1 has radius 2
  r2 = 3 →  -- Circle 2 has radius 3
  (∃ d : ℝ, d = r1 + r2) →  -- Circle 1 and Circle 2 are externally tangent
  (∃ k1 k2 k3 : ℝ, k1 = 1/r1 ∧ k2 = 1/r2 ∧ k3 = 1/r3 ∧ 
    k1 + k2 + k3 + 2 * Real.sqrt (k1 * k2 + k2 * k3 + k3 * k1) = 0) →  -- Descartes' theorem for externally tangent circles
  r3 = 15/14 := by
sorry

end third_circle_radius_l329_32980


namespace rectangle_width_length_ratio_l329_32916

/-- Given a rectangle with length 12 and perimeter 36, prove that the ratio of its width to its length is 1:2 -/
theorem rectangle_width_length_ratio (w : ℝ) : 
  w > 0 → -- width is positive
  12 > 0 → -- length is positive
  2 * w + 2 * 12 = 36 → -- perimeter formula
  w / 12 = 1 / 2 := by sorry

end rectangle_width_length_ratio_l329_32916


namespace split_sum_equals_capacity_l329_32907

/-- The capacity of a pile with n stones -/
def capacity (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The sum of products obtained by splitting n stones -/
def split_sum (n : ℕ) : ℕ := sorry

theorem split_sum_equals_capacity :
  split_sum 2019 = capacity 2019 :=
sorry

end split_sum_equals_capacity_l329_32907


namespace chemistry_class_grades_l329_32968

theorem chemistry_class_grades (total_students : ℕ) 
  (prob_A prob_B prob_C prob_F : ℝ) : 
  total_students = 50 →
  prob_A = 0.6 * prob_B →
  prob_C = 1.5 * prob_B →
  prob_F = 0.4 * prob_B →
  prob_A + prob_B + prob_C + prob_F = 1 →
  ⌊total_students * prob_B⌋ = 14 := by
sorry

end chemistry_class_grades_l329_32968


namespace fib_even_iff_index_div_three_l329_32942

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: A Fibonacci number is even if and only if its index is divisible by 3 -/
theorem fib_even_iff_index_div_three (n : ℕ) : Even (fib n) ↔ 3 ∣ n := by sorry

end fib_even_iff_index_div_three_l329_32942


namespace sufficient_not_necessary_l329_32950

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 9

-- Define the condition for no real roots
def has_no_real_roots (m : ℝ) : Prop := ∀ x : ℝ, f m x ≠ 0

-- Define the given interval
def p (m : ℝ) : Prop := -6 ≤ m ∧ m ≤ 6

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ m : ℝ, p m → has_no_real_roots m) ∧
  ¬(∀ m : ℝ, has_no_real_roots m → p m) :=
sorry

end sufficient_not_necessary_l329_32950


namespace rooster_count_l329_32993

/-- Given a chicken farm with roosters and hens, proves the number of roosters -/
theorem rooster_count (total : ℕ) (ratio : ℚ) (rooster_count : ℕ) : 
  total = 9000 →
  ratio = 2 / 1 →
  rooster_count = total * (ratio / (1 + ratio)) →
  rooster_count = 6000 := by
  sorry


end rooster_count_l329_32993


namespace dmv_waiting_time_l329_32979

/-- Calculates the additional waiting time at the DMV -/
theorem dmv_waiting_time (initial_wait : ℕ) (total_wait : ℕ) : 
  initial_wait = 20 →
  total_wait = 114 →
  total_wait = initial_wait + 4 * initial_wait + (total_wait - (initial_wait + 4 * initial_wait)) →
  total_wait - (initial_wait + 4 * initial_wait) = 34 :=
by sorry

end dmv_waiting_time_l329_32979


namespace company_employee_reduction_l329_32947

theorem company_employee_reduction (reduction_percentage : ℝ) (final_employees : ℕ) : 
  reduction_percentage = 14 →
  final_employees = 195 →
  round (final_employees / (1 - reduction_percentage / 100)) = 227 :=
by sorry

end company_employee_reduction_l329_32947


namespace walking_speed_calculation_l329_32940

/-- Given a person who runs at 8 km/hr and covers a total distance of 16 km
    (half walking, half running) in 3 hours, prove that the walking speed is 4 km/hr. -/
theorem walking_speed_calculation (running_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) :
  running_speed = 8 →
  total_distance = 16 →
  total_time = 3 →
  ∃ (walking_speed : ℝ),
    walking_speed * (total_distance / 2 / walking_speed) +
    running_speed * (total_distance / 2 / running_speed) = total_time ∧
    walking_speed = 4 :=
by sorry

end walking_speed_calculation_l329_32940


namespace divisibility_of_integer_part_l329_32914

theorem divisibility_of_integer_part (m : ℕ) 
  (h_odd : m % 2 = 1) 
  (h_not_div_3 : m % 3 ≠ 0) : 
  ∃ k : ℤ, (4^m : ℝ) - (2 + Real.sqrt 2)^m = k + (112 : ℝ) * ↑(⌊((4^m : ℝ) - (2 + Real.sqrt 2)^m) / 112⌋) := by
  sorry

end divisibility_of_integer_part_l329_32914


namespace sum_of_terms_3_to_6_l329_32998

/-- Given a sequence {aₙ} where the sum of the first n terms is Sₙ = n² + 2n + 5,
    prove that a₃ + a₄ + a₅ + a₆ = 40 -/
theorem sum_of_terms_3_to_6 (a : ℕ → ℤ) (S : ℕ → ℤ) 
    (h : ∀ n : ℕ, S n = n^2 + 2*n + 5) : 
    a 3 + a 4 + a 5 + a 6 = 40 := by
  sorry

end sum_of_terms_3_to_6_l329_32998


namespace perpendicular_transitivity_l329_32975

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and lines
variable (perp : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_transitivity 
  (α β : Plane) (m n : Line) 
  (h1 : perp_line_plane n α) 
  (h2 : perp_line_plane n β) 
  (h3 : perp_line_plane m α) : 
  perp_line_plane m β :=
sorry

end perpendicular_transitivity_l329_32975


namespace problem_solution_l329_32954

theorem problem_solution (m n : ℝ) 
  (h1 : m^2 + m*n = -3) 
  (h2 : n^2 - 3*m*n = 18) : 
  m^2 + 4*m*n - n^2 = -21 := by
sorry

end problem_solution_l329_32954


namespace fifth_term_of_arithmetic_sequence_l329_32932

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fifth_term_of_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_first : a 1 = 6) 
  (h_third : a 3 = 2) : 
  a 5 = -2 := by
sorry

end fifth_term_of_arithmetic_sequence_l329_32932


namespace bird_count_l329_32944

theorem bird_count (swallows bluebirds cardinals : ℕ) : 
  swallows = 2 →
  swallows * 2 = bluebirds →
  cardinals = bluebirds * 3 →
  swallows + bluebirds + cardinals = 18 := by
sorry

end bird_count_l329_32944


namespace polynomial_coefficient_sum_l329_32930

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5 * x + 4)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂) - (a₁ + a₃) = -1 := by
sorry

end polynomial_coefficient_sum_l329_32930


namespace bombardment_percentage_approx_10_percent_l329_32937

def initial_population : ℕ := 8515
def final_population : ℕ := 6514
def departure_rate : ℚ := 15 / 100

def bombardment_percentage : ℚ :=
  (initial_population - final_population) / initial_population * 100

theorem bombardment_percentage_approx_10_percent :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  abs (bombardment_percentage - 10) < ε ∧
  final_population = 
    initial_population * (1 - bombardment_percentage / 100) * (1 - departure_rate) :=
by sorry

end bombardment_percentage_approx_10_percent_l329_32937


namespace complex_number_in_fourth_quadrant_l329_32971

/-- The complex number z = sin 2 + i cos 2 is located in the fourth quadrant of the complex plane -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := Complex.mk (Real.sin 2) (Real.cos 2)
  z.re > 0 ∧ z.im < 0 :=
by sorry

end complex_number_in_fourth_quadrant_l329_32971


namespace remainder_3_102_mod_101_l329_32965

theorem remainder_3_102_mod_101 (h : Nat.Prime 101) : 3^102 ≡ 9 [MOD 101] := by
  sorry

end remainder_3_102_mod_101_l329_32965


namespace base_k_representation_l329_32919

theorem base_k_representation (k : ℕ) (hk : k > 0) : 
  (8 : ℚ) / 63 = (k + 5 : ℚ) / (k^2 - 1) → k = 17 := by
  sorry

end base_k_representation_l329_32919


namespace paint_production_max_profit_l329_32917

/-- The paint production problem -/
theorem paint_production_max_profit :
  let material_A : ℝ := 120
  let material_B : ℝ := 90
  let total_production : ℝ := 150
  let type_A_material_A : ℝ := 0.6
  let type_A_material_B : ℝ := 0.7
  let type_A_profit : ℝ := 450
  let type_B_material_A : ℝ := 0.9
  let type_B_material_B : ℝ := 0.4
  let type_B_profit : ℝ := 500
  let profit (x : ℝ) := type_A_profit * x + type_B_profit * (total_production - x)
  ∀ x : ℝ, 
    (type_A_material_A * x + type_B_material_A * (total_production - x) ≤ material_A) →
    (type_A_material_B * x + type_B_material_B * (total_production - x) ≤ material_B) →
    profit x ≤ 72500 ∧ 
    (x = 50 → profit x = 72500) :=
by sorry

end paint_production_max_profit_l329_32917


namespace min_tan_product_acute_triangle_l329_32956

theorem min_tan_product_acute_triangle (A B C : ℝ) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sin : Real.sin A = 3 * Real.sin B * Real.sin C) : 
  (∀ A' B' C', 0 < A' ∧ 0 < B' ∧ 0 < C' ∧ A' + B' + C' = π → 
    Real.sin A' = 3 * Real.sin B' * Real.sin C' → 
    12 ≤ Real.tan A' * Real.tan B' * Real.tan C') ∧
  (∃ A₀ B₀ C₀, 0 < A₀ ∧ 0 < B₀ ∧ 0 < C₀ ∧ A₀ + B₀ + C₀ = π ∧
    Real.sin A₀ = 3 * Real.sin B₀ * Real.sin C₀ ∧
    Real.tan A₀ * Real.tan B₀ * Real.tan C₀ = 12) := by
  sorry

end min_tan_product_acute_triangle_l329_32956


namespace inscribed_rectangle_area_coefficient_l329_32920

/-- Triangle with side lengths --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle inscribed in a triangle --/
structure InscribedRectangle where
  base : ℝ

/-- The area of the inscribed rectangle as a function of its base --/
def rectangleArea (t : Triangle) (r : InscribedRectangle) : ℝ → ℝ :=
  fun ω => α * ω - β * ω^2
  where
    α : ℝ := sorry
    β : ℝ := sorry

theorem inscribed_rectangle_area_coefficient (t : Triangle) (r : InscribedRectangle) :
  t.a = 15 ∧ t.b = 34 ∧ t.c = 21 →
  ∃ α β : ℝ, (∀ ω : ℝ, rectangleArea t r ω = α * ω - β * ω^2) ∧ β = 5/41 := by
  sorry

end inscribed_rectangle_area_coefficient_l329_32920


namespace existence_of_special_multiple_l329_32986

theorem existence_of_special_multiple (p : ℕ) (hp : p > 1) (hgcd : Nat.gcd p 10 = 1) :
  ∃ n : ℕ, 
    (Nat.digits 10 n).length = p - 2 ∧ 
    (∀ d ∈ Nat.digits 10 n, d = 1 ∨ d = 3) ∧
    p ∣ n :=
by sorry

end existence_of_special_multiple_l329_32986


namespace carnival_tickets_l329_32964

/-- Calculates the total number of tickets used at a carnival -/
theorem carnival_tickets (ferris_wheel_rides bumper_car_rides roller_coaster_rides teacup_rides : ℕ)
                         (ferris_wheel_cost bumper_car_cost roller_coaster_cost teacup_cost : ℕ) :
  ferris_wheel_rides * ferris_wheel_cost +
  bumper_car_rides * bumper_car_cost +
  roller_coaster_rides * roller_coaster_cost +
  teacup_rides * teacup_cost = 105 :=
by
  -- Assuming ferris_wheel_rides = 7, bumper_car_rides = 3, roller_coaster_rides = 4, teacup_rides = 5
  -- and ferris_wheel_cost = 5, bumper_car_cost = 6, roller_coaster_cost = 8, teacup_cost = 4
  sorry


end carnival_tickets_l329_32964


namespace election_vote_difference_l329_32981

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 7500 → 
  candidate_percentage = 35/100 → 
  (total_votes : ℚ) * candidate_percentage - (total_votes : ℚ) * (1 - candidate_percentage) = -2250 := by
  sorry

end election_vote_difference_l329_32981


namespace hyperbola_focus_l329_32938

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - 4*y^2 - 6*x + 24*y - 11 = 0

-- Define the foci coordinates
def focus_coord (x y : ℝ) : Prop :=
  (x = 3 ∧ y = 3 + 2 * Real.sqrt 5) ∨ (x = 3 ∧ y = 3 - 2 * Real.sqrt 5)

-- Theorem statement
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_eq x y ∧ focus_coord x y :=
sorry

end hyperbola_focus_l329_32938


namespace lcm_of_330_and_210_l329_32952

theorem lcm_of_330_and_210 (hcf : ℕ) (a b lcm : ℕ) : 
  hcf = 30 → a = 330 → b = 210 → lcm = Nat.lcm a b → lcm = 2310 := by
sorry

end lcm_of_330_and_210_l329_32952


namespace leticia_dish_cost_is_10_l329_32913

/-- The cost of Leticia's dish -/
def leticia_dish_cost : ℝ := 10

/-- The cost of Scarlett's dish -/
def scarlett_dish_cost : ℝ := 13

/-- The cost of Percy's dish -/
def percy_dish_cost : ℝ := 17

/-- The tip rate -/
def tip_rate : ℝ := 0.10

/-- The total tip amount -/
def total_tip : ℝ := 4

/-- Theorem stating that Leticia's dish costs $10 given the conditions -/
theorem leticia_dish_cost_is_10 :
  leticia_dish_cost = 10 :=
by sorry

end leticia_dish_cost_is_10_l329_32913


namespace max_y_value_l329_32972

theorem max_y_value (x y : ℤ) (h : x * y + 6 * x + 5 * y = -6) : 
  ∃ (max_y : ℤ), y ≤ max_y ∧ max_y = 18 := by
  sorry

end max_y_value_l329_32972


namespace no_solution_absolute_value_equation_l329_32906

theorem no_solution_absolute_value_equation :
  ∀ x : ℝ, |-3 * x| + 5 ≠ 0 := by
sorry

end no_solution_absolute_value_equation_l329_32906


namespace cycle_selling_price_l329_32999

/-- Given a cycle with a cost price of 1400 and sold at a loss of 25%, 
    prove that the selling price is 1050. -/
theorem cycle_selling_price 
  (cost_price : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : cost_price = 1400) 
  (h2 : loss_percentage = 25) : 
  cost_price * (1 - loss_percentage / 100) = 1050 := by
sorry

end cycle_selling_price_l329_32999


namespace final_value_calculation_l329_32904

def initial_value : ℝ := 1500

def first_increase (x : ℝ) : ℝ := x * 1.20

def second_decrease (x : ℝ) : ℝ := x * 0.85

def third_increase (x : ℝ) : ℝ := x * 1.10

theorem final_value_calculation :
  third_increase (second_decrease (first_increase initial_value)) = 1683 := by
  sorry

end final_value_calculation_l329_32904


namespace art_arrangement_probability_l329_32905

/-- The total number of art pieces --/
def total_pieces : ℕ := 12

/-- The number of Escher prints --/
def escher_prints : ℕ := 4

/-- The number of Picasso prints --/
def picasso_prints : ℕ := 2

/-- The probability of the desired arrangement --/
def arrangement_probability : ℚ := 912 / 479001600

theorem art_arrangement_probability :
  let remaining_pieces := total_pieces - escher_prints
  let escher_block_positions := remaining_pieces + 1
  let escher_internal_arrangements := Nat.factorial escher_prints
  let picasso_positions := total_pieces - escher_prints + 1
  let valid_picasso_arrangements := 38
  (escher_block_positions * escher_internal_arrangements * valid_picasso_arrangements : ℚ) /
    Nat.factorial total_pieces = arrangement_probability := by
  sorry

end art_arrangement_probability_l329_32905


namespace finns_purchase_theorem_l329_32989

/-- The cost of Finn's purchase given the conditions of the problem -/
def finns_purchase_cost (paper_clip_cost index_card_cost : ℚ) : ℚ :=
  12 * paper_clip_cost + 10 * index_card_cost

/-- The theorem stating the cost of Finn's purchase -/
theorem finns_purchase_theorem :
  ∃ (index_card_cost : ℚ),
    15 * (1.85 : ℚ) + 7 * index_card_cost = 55.40 ∧
    finns_purchase_cost 1.85 index_card_cost = 61.70 := by
  sorry

#eval finns_purchase_cost (1.85 : ℚ) (3.95 : ℚ)

end finns_purchase_theorem_l329_32989


namespace sheila_picnic_probability_l329_32988

/-- The probability of Sheila attending the picnic -/
def probability_attend : ℝ := 0.55

/-- The probability of rain -/
def p_rain : ℝ := 0.30

/-- The probability of sunny weather -/
def p_sunny : ℝ := 0.50

/-- The probability of partly cloudy weather -/
def p_partly_cloudy : ℝ := 0.20

/-- The probability Sheila attends if it rains -/
def p_attend_rain : ℝ := 0.15

/-- The probability Sheila attends if it's sunny -/
def p_attend_sunny : ℝ := 0.85

/-- The probability Sheila attends if it's partly cloudy -/
def p_attend_partly_cloudy : ℝ := 0.40

/-- Theorem stating that the probability of Sheila attending the picnic is correct -/
theorem sheila_picnic_probability : 
  probability_attend = p_rain * p_attend_rain + p_sunny * p_attend_sunny + p_partly_cloudy * p_attend_partly_cloudy :=
by sorry

end sheila_picnic_probability_l329_32988


namespace first_job_earnings_is_52_l329_32909

/-- Represents Mike's weekly wages --/
def TotalWages : ℝ := 160

/-- Represents the hours Mike works at his second job --/
def SecondJobHours : ℝ := 12

/-- Represents the hourly rate for Mike's second job --/
def SecondJobRate : ℝ := 9

/-- Calculates the amount Mike earns from his second job --/
def SecondJobEarnings : ℝ := SecondJobHours * SecondJobRate

/-- Represents the amount Mike earns from his first job --/
def FirstJobEarnings : ℝ := TotalWages - SecondJobEarnings

/-- Proves that Mike's earnings from his first job is $52 --/
theorem first_job_earnings_is_52 : FirstJobEarnings = 52 := by
  sorry

end first_job_earnings_is_52_l329_32909


namespace range_of_a_l329_32997

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ Real.exp x * (x + a) < 1) → a < 1 := by
  sorry

end range_of_a_l329_32997


namespace farmer_shipped_six_boxes_last_week_l329_32984

/-- Represents the number of pomelos in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of pomelos shipped last week -/
def pomelos_last_week : ℕ := 240

/-- Represents the number of boxes shipped this week -/
def boxes_this_week : ℕ := 20

/-- Represents the total number of dozens of pomelos shipped -/
def total_dozens : ℕ := 60

/-- Represents the number of boxes shipped last week -/
def boxes_last_week : ℕ := 6

/-- Proves that the farmer shipped 6 boxes last week given the conditions -/
theorem farmer_shipped_six_boxes_last_week :
  let total_pomelos := total_dozens * dozen
  let pomelos_this_week := total_pomelos - pomelos_last_week
  let pomelos_per_box := pomelos_this_week / boxes_this_week
  pomelos_last_week / pomelos_per_box = boxes_last_week :=
by sorry

end farmer_shipped_six_boxes_last_week_l329_32984


namespace z_gets_30_paisa_l329_32953

/-- Represents the division of money among three parties -/
structure MoneyDivision where
  total : ℚ
  y_share : ℚ
  y_rate : ℚ

/-- Calculates the share of z given a money division -/
def z_share (md : MoneyDivision) : ℚ :=
  md.total - md.y_share - (md.y_share / md.y_rate)

/-- Calculates the rate at which z receives money compared to x -/
def z_rate (md : MoneyDivision) : ℚ :=
  (z_share md) / (md.y_share / md.y_rate)

/-- Theorem stating that z gets 30 paisa for each rupee x gets -/
theorem z_gets_30_paisa (md : MoneyDivision) 
  (h1 : md.total = 105)
  (h2 : md.y_share = 27)
  (h3 : md.y_rate = 45/100) : 
  z_rate md = 30/100 := by
  sorry

end z_gets_30_paisa_l329_32953


namespace cable_length_equals_scientific_notation_l329_32983

/-- The total length of fiber optic cable routes in kilometers -/
def cable_length : ℝ := 59580000

/-- The scientific notation representation of the cable length -/
def cable_length_scientific : ℝ := 5.958 * (10 ^ 7)

/-- Theorem stating that the cable length is equal to its scientific notation representation -/
theorem cable_length_equals_scientific_notation : cable_length = cable_length_scientific := by
  sorry

end cable_length_equals_scientific_notation_l329_32983


namespace geometric_sequence_quadratic_root_l329_32977

theorem geometric_sequence_quadratic_root
  (a b c : ℝ)
  (h_geom : ∃ r : ℝ, b = a * r ∧ c = a * r^2)
  (h_order : a ≤ b ∧ b ≤ c)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_one_root : ∃! x : ℝ, a * x^2 + b * x + c = 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = -1/8 :=
sorry

end geometric_sequence_quadratic_root_l329_32977


namespace garden_operations_result_l329_32908

/-- Represents the quantities of vegetables in the garden -/
structure VegetableQuantities where
  tomatoes : ℕ
  potatoes : ℕ
  cucumbers : ℕ
  cabbages : ℕ

/-- Calculates the final quantities of vegetables after operations -/
def final_quantities (initial : VegetableQuantities) 
  (picked_tomatoes picked_potatoes picked_cabbages : ℕ)
  (new_cucumber_plants new_cabbage_plants : ℕ)
  (cucumber_yield cabbage_yield : ℕ) : VegetableQuantities :=
  { tomatoes := initial.tomatoes - picked_tomatoes,
    potatoes := initial.potatoes - picked_potatoes,
    cucumbers := initial.cucumbers + new_cucumber_plants * cucumber_yield,
    cabbages := initial.cabbages - picked_cabbages + new_cabbage_plants * cabbage_yield }

theorem garden_operations_result :
  let initial := VegetableQuantities.mk 500 400 300 100
  let final := final_quantities initial 325 270 50 200 80 2 3
  final.tomatoes = 175 ∧ 
  final.potatoes = 130 ∧ 
  final.cucumbers = 700 ∧ 
  final.cabbages = 290 := by
  sorry

end garden_operations_result_l329_32908


namespace greater_fraction_l329_32994

theorem greater_fraction (x y : ℚ) (h_sum : x + y = 5/6) (h_prod : x * y = 1/8) :
  max x y = (5 + Real.sqrt 7) / 12 := by
  sorry

end greater_fraction_l329_32994


namespace gap_height_from_wire_extension_l329_32927

/-- Given a sphere of radius R and a wire wrapped around its equator,
    if the wire's length is increased by L, the resulting gap height h
    between the sphere and the wire is given by h = L / (2π). -/
theorem gap_height_from_wire_extension (R L : ℝ) (h : ℝ) 
    (hR : R > 0) (hL : L > 0) : 
    2 * π * (R + h) = 2 * π * R + L → h = L / (2 * π) := by
  sorry

end gap_height_from_wire_extension_l329_32927


namespace similar_triangles_segment_length_l329_32961

/-- Two triangles are similar -/
structure SimilarTriangles (T1 T2 : Type) :=
  (sim : T1 → T2 → Prop)

/-- Triangle GHI -/
structure TriangleGHI :=
  (G H I : ℝ)

/-- Triangle XYZ -/
structure TriangleXYZ :=
  (X Y Z : ℝ)

/-- The problem statement -/
theorem similar_triangles_segment_length 
  (tri_GHI : TriangleGHI) 
  (tri_XYZ : TriangleXYZ) 
  (sim : SimilarTriangles TriangleGHI TriangleXYZ) 
  (h_sim : sim.sim tri_GHI tri_XYZ)
  (h_GH : tri_GHI.H - tri_GHI.G = 8)
  (h_HI : tri_GHI.I - tri_GHI.H = 16)
  (h_YZ : tri_XYZ.Z - tri_XYZ.Y = 24) :
  tri_XYZ.Y - tri_XYZ.X = 12 := by
  sorry

end similar_triangles_segment_length_l329_32961


namespace polynomial_division_remainder_l329_32912

theorem polynomial_division_remainder : 
  let p (x : ℚ) := x^4 - 4*x^3 + 13*x^2 - 14*x + 4
  let d (x : ℚ) := x^2 - 3*x + 13/3
  let q (x : ℚ) := x^2 - x + 10/3
  let r (x : ℚ) := 2*x + 16/9
  ∀ x, p x = d x * q x + r x := by
sorry

end polynomial_division_remainder_l329_32912


namespace problem_solution_l329_32951

theorem problem_solution : 101^4 - 4 * 101^3 + 6 * 101^2 - 4 * 101 + 1 = 100000000 := by
  sorry

end problem_solution_l329_32951


namespace solution_equation_l329_32924

theorem solution_equation (x : ℝ) : (5 * 12) / (x / 3) + 80 = 81 ↔ x = 180 :=
by sorry

end solution_equation_l329_32924


namespace exponential_inequality_l329_32936

theorem exponential_inequality (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a < b) : 
  (2 : ℝ) ^ a < (2 : ℝ) ^ b := by
  sorry

end exponential_inequality_l329_32936


namespace rational_powers_imply_rational_irrational_with_rational_powers_l329_32925

-- Define rationality for real numbers
def IsRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Part a
theorem rational_powers_imply_rational (x : ℝ) 
  (h1 : IsRational (x^7)) (h2 : IsRational (x^12)) : 
  IsRational x := by sorry

-- Part b
theorem irrational_with_rational_powers : 
  ∃ (x : ℝ), IsRational (x^9) ∧ IsRational (x^12) ∧ ¬IsRational x := by sorry

end rational_powers_imply_rational_irrational_with_rational_powers_l329_32925


namespace prime_pairs_theorem_l329_32978

def is_valid_pair (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ (7 * p + 1) % q = 0 ∧ (7 * q + 1) % p = 0

theorem prime_pairs_theorem : 
  ∀ p q : ℕ, is_valid_pair p q ↔ (p = 2 ∧ q = 3) ∨ (p = 2 ∧ q = 5) ∨ (p = 3 ∧ q = 11) :=
sorry

end prime_pairs_theorem_l329_32978


namespace elevator_weight_problem_l329_32943

theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
  (new_avg_weight : ℝ) (h1 : initial_people = 6) 
  (h2 : initial_avg_weight = 152) (h3 : new_avg_weight = 151) :
  let total_initial_weight := initial_people * initial_avg_weight
  let total_new_weight := (initial_people + 1) * new_avg_weight
  let seventh_person_weight := total_new_weight - total_initial_weight
  seventh_person_weight = 145 := by
  sorry

end elevator_weight_problem_l329_32943


namespace line_intercepts_l329_32923

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3 * x - y + 6 = 0

/-- The y-intercept of the line -/
def y_intercept : ℝ := 6

/-- The x-intercept of the line -/
def x_intercept : ℝ := -2

/-- Theorem stating that the y-intercept and x-intercept are correct for the given line equation -/
theorem line_intercepts :
  line_equation 0 y_intercept ∧ line_equation x_intercept 0 :=
sorry

end line_intercepts_l329_32923


namespace circles_internally_tangent_l329_32976

theorem circles_internally_tangent (r1 r2 : ℝ) (d : ℝ) : 
  r1 + r2 = 5 ∧ 
  r1 * r2 = 3 ∧ 
  d = 3 → 
  r1 < r2 ∧ r2 - r1 < d ∧ d < r1 + r2 := by
  sorry

#check circles_internally_tangent

end circles_internally_tangent_l329_32976


namespace twelve_tone_equal_temperament_l329_32957

theorem twelve_tone_equal_temperament (a : ℕ → ℝ) :
  (∀ n : ℕ, 1 ≤ n → n ≤ 13 → a n > 0) →
  (∀ n : ℕ, 1 < n → n ≤ 13 → a n / a (n-1) = a 2 / a 1) →
  a 1 = 1 →
  a 13 = 2 →
  a 5 = 2^(1/3) :=
sorry

end twelve_tone_equal_temperament_l329_32957


namespace cos_two_beta_l329_32982

theorem cos_two_beta (α β : Real) 
  (h1 : Real.sin α = Real.cos β) 
  (h2 : Real.sin α * Real.cos β - 2 * Real.cos α * Real.sin β = 1/2) : 
  Real.cos (2 * β) = 2/3 := by
sorry

end cos_two_beta_l329_32982


namespace unique_function_divisibility_l329_32903

theorem unique_function_divisibility 
  (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), (m^2 + f n) ∣ (m * f m + n)) : 
  ∀ (n : ℕ+), f n = n :=
sorry

end unique_function_divisibility_l329_32903


namespace at_least_one_greater_than_fifty_l329_32987

theorem at_least_one_greater_than_fifty (a₁ a₂ : ℝ) (h : a₁ + a₂ > 100) :
  a₁ > 50 ∨ a₂ > 50 := by
  sorry

end at_least_one_greater_than_fifty_l329_32987


namespace concentric_circles_area_l329_32929

theorem concentric_circles_area (r₁ : ℝ) (chord_length : ℝ) (h₁ : r₁ = 50) (h₂ : chord_length = 120) : 
  let r₂ := Real.sqrt (r₁^2 + (chord_length/2)^2)
  π * (r₂^2 - r₁^2) = 3600 * π := by
sorry

end concentric_circles_area_l329_32929


namespace tip_calculation_l329_32995

/-- Calculates the tip amount given the meal cost, tax rate, and payment amount. -/
def calculate_tip (meal_cost : ℝ) (tax_rate : ℝ) (payment : ℝ) : ℝ :=
  payment - (meal_cost * (1 + tax_rate))

/-- Proves that given a meal cost of $15.00, a tax rate of 20%, and a payment of $20.00, the tip amount is $2.00. -/
theorem tip_calculation :
  calculate_tip 15 0.2 20 = 2 := by
  sorry

#eval calculate_tip 15 0.2 20

end tip_calculation_l329_32995


namespace number_wall_solution_l329_32985

structure NumberWall :=
  (x : ℤ)
  (a b c d : ℤ)
  (e f g : ℤ)
  (h i : ℤ)
  (j : ℤ)

def NumberWall.valid (w : NumberWall) : Prop :=
  w.e = w.x + w.a ∧
  w.f = w.a + w.b ∧
  w.g = w.b + w.c ∧
  w.d = w.c + w.d ∧
  w.h = w.e + w.f ∧
  w.i = w.g + w.d ∧
  w.j = w.h + w.i ∧
  w.a = 5 ∧
  w.b = 10 ∧
  w.c = 9 ∧
  w.d = 6 ∧
  w.i = 18 ∧
  w.j = 72

theorem number_wall_solution (w : NumberWall) (h : w.valid) : w.x = -50 := by
  sorry

end number_wall_solution_l329_32985


namespace tenths_of_2019_l329_32973

theorem tenths_of_2019 : (2019 : ℚ) / 10 = 201.9 := by
  sorry

end tenths_of_2019_l329_32973


namespace square_cube_relation_l329_32934

theorem square_cube_relation (a b : ℝ) :
  ¬(∀ a b : ℝ, a^2 > b^2 → a^3 > b^3) ∧
  ¬(∀ a b : ℝ, a^3 > b^3 → a^2 > b^2) :=
sorry

end square_cube_relation_l329_32934


namespace greater_number_problem_l329_32935

theorem greater_number_problem (x y : ℝ) (sum_eq : x + y = 40) (diff_eq : x - y = 10) :
  max x y = 25 := by sorry

end greater_number_problem_l329_32935


namespace polygon_interior_angles_sum_l329_32941

theorem polygon_interior_angles_sum (n : ℕ) :
  (180 * (n - 2) = 1260) →
  (180 * ((n + 3) - 2) = 1800) := by
  sorry

end polygon_interior_angles_sum_l329_32941


namespace janes_weekly_reading_l329_32974

/-- Represents the number of pages Jane reads on a given day -/
structure DailyReading where
  morning : ℕ
  lunch : ℕ
  evening : ℕ
  extra : ℕ

/-- Calculates the total pages read in a day -/
def totalPagesPerDay (d : DailyReading) : ℕ :=
  d.morning + d.lunch + d.evening + d.extra

/-- Represents Jane's weekly reading schedule -/
def weeklySchedule : List DailyReading :=
  [
    { morning := 5,  lunch := 0, evening := 10, extra := 0  }, -- Monday
    { morning := 7,  lunch := 0, evening := 8,  extra := 0  }, -- Tuesday
    { morning := 5,  lunch := 0, evening := 5,  extra := 0  }, -- Wednesday
    { morning := 7,  lunch := 0, evening := 8,  extra := 15 }, -- Thursday
    { morning := 10, lunch := 5, evening := 0,  extra := 0  }, -- Friday
    { morning := 12, lunch := 0, evening := 20, extra := 0  }, -- Saturday
    { morning := 12, lunch := 0, evening := 0,  extra := 0  }  -- Sunday
  ]

/-- Theorem: Jane reads 129 pages in total over one week -/
theorem janes_weekly_reading : 
  (weeklySchedule.map totalPagesPerDay).sum = 129 := by
  sorry

end janes_weekly_reading_l329_32974


namespace quadratic_roots_relation_l329_32969

theorem quadratic_roots_relation (s t : ℝ) 
  (hs : 19 * s^2 + 99 * s + 1 = 0)
  (ht : t^2 + 99 * t + 19 = 0)
  (hst : s * t ≠ 1) :
  (s * t + 4 * s + 1) / t = -5 :=
by sorry

end quadratic_roots_relation_l329_32969


namespace inscribed_rectangle_pc_length_l329_32948

-- Define the triangle ABC
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define the rectangle PQRS
structure InscribedRectangle (t : EquilateralTriangle) where
  ps : ℝ
  pq : ℝ
  ps_positive : ps > 0
  pq_positive : pq > 0
  on_sides : ps ≤ t.side ∧ pq ≤ t.side
  is_rectangle : pq = Real.sqrt 3 * ps
  area : ps * pq = 28 * Real.sqrt 3

-- Define the theorem
theorem inscribed_rectangle_pc_length 
  (t : EquilateralTriangle) 
  (r : InscribedRectangle t) : 
  ∃ (pc : ℝ), pc = 2 * Real.sqrt 7 ∧ 
  pc^2 = t.side^2 + (t.side - r.ps)^2 - 2 * t.side * (t.side - r.ps) * Real.cos (π/3) :=
sorry

end inscribed_rectangle_pc_length_l329_32948


namespace ticket_price_is_six_l329_32996

/-- The price of a concert ticket, given the following conditions:
  * Lana bought 8 tickets for herself and friends
  * Lana bought 2 extra tickets
  * Lana spent $60 in total
-/
def ticket_price : ℚ := by
  -- Define the number of tickets for Lana and friends
  let lana_friends_tickets : ℕ := 8
  -- Define the number of extra tickets
  let extra_tickets : ℕ := 2
  -- Define the total amount spent
  let total_spent : ℚ := 60
  -- Calculate the total number of tickets
  let total_tickets : ℕ := lana_friends_tickets + extra_tickets
  -- Calculate the price per ticket
  exact total_spent / total_tickets
  
-- Prove that the ticket price is $6
theorem ticket_price_is_six : ticket_price = 6 := by
  sorry

end ticket_price_is_six_l329_32996


namespace arithmetic_equalities_l329_32992

theorem arithmetic_equalities : 
  (10 - 20 - (-7) + |(-2)|) = -1 ∧ 
  (48 * (-1/4) - (-36) / 4) = -3 := by sorry

end arithmetic_equalities_l329_32992


namespace shorter_diagonal_length_l329_32922

/-- Represents a rhombus with given properties -/
structure Rhombus where
  area : ℝ
  diagonal_ratio : ℝ × ℝ
  diagonal_side_relation : Bool

/-- Theorem: In a rhombus with area 150 and diagonal ratio 5:3, the shorter diagonal is 6√5 -/
theorem shorter_diagonal_length (R : Rhombus) 
    (h_area : R.area = 150)
    (h_ratio : R.diagonal_ratio = (5, 3))
    (h_relation : R.diagonal_side_relation = true) : 
  ∃ (d : ℝ), d = 6 * Real.sqrt 5 ∧ d = min (5 * (2 * Real.sqrt 5)) (3 * (2 * Real.sqrt 5)) :=
by sorry

end shorter_diagonal_length_l329_32922


namespace operation_property_l329_32946

theorem operation_property (h : ℕ → ℝ) (k : ℝ) (n : ℕ) 
  (h_def : ∀ m l : ℕ, h (m + l) = h m * h l) 
  (h_2 : h 2 = k) 
  (k_nonzero : k ≠ 0) : 
  h (2 * n) * h 2024 = k^(n + 1012) := by
  sorry

end operation_property_l329_32946


namespace cube_root_of_456533_l329_32921

theorem cube_root_of_456533 (z : ℤ) :
  z^3 = 456533 → z = 77 := by
  sorry

end cube_root_of_456533_l329_32921


namespace complex_number_location_l329_32955

open Complex

theorem complex_number_location (z : ℂ) (h : z / (1 + I) = 2 - I) :
  0 < z.re ∧ 0 < z.im :=
by sorry

end complex_number_location_l329_32955


namespace fruit_arrangement_count_l329_32926

theorem fruit_arrangement_count : 
  let total_fruits : ℕ := 10
  let apple_count : ℕ := 4
  let orange_count : ℕ := 3
  let banana_count : ℕ := 2
  let grape_count : ℕ := 1
  apple_count + orange_count + banana_count + grape_count = total_fruits →
  (Nat.factorial total_fruits) / 
  ((Nat.factorial apple_count) * (Nat.factorial orange_count) * 
   (Nat.factorial banana_count) * (Nat.factorial grape_count)) = 12600 := by
sorry

end fruit_arrangement_count_l329_32926


namespace alex_not_jogging_probability_l329_32911

theorem alex_not_jogging_probability (p : ℚ) 
  (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end alex_not_jogging_probability_l329_32911


namespace dvd_pack_cost_l329_32963

/-- Given that 6 packs of DVDs can be bought with 120 dollars, 
    prove that each pack costs 20 dollars. -/
theorem dvd_pack_cost (total_cost : ℕ) (num_packs : ℕ) (cost_per_pack : ℕ) 
    (h1 : total_cost = 120) 
    (h2 : num_packs = 6) 
    (h3 : total_cost = num_packs * cost_per_pack) : 
  cost_per_pack = 20 := by
  sorry

end dvd_pack_cost_l329_32963


namespace no_positive_solutions_l329_32958

theorem no_positive_solutions : ¬∃ (a b c d : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a * d^2 + b * d - c = 0 ∧
  Real.sqrt a * d + Real.sqrt b * Real.sqrt d - Real.sqrt c = 0 := by
  sorry

end no_positive_solutions_l329_32958


namespace triangle_area_l329_32962

theorem triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 84) (h3 : c = 85) :
  (1/2) * a * b = 546 := by
  sorry

end triangle_area_l329_32962


namespace first_group_size_correct_l329_32949

/-- The number of beavers in the first group -/
def first_group_size : ℕ := 20

/-- The time taken by the first group to build the dam (in hours) -/
def first_group_time : ℕ := 3

/-- The number of beavers in the second group -/
def second_group_size : ℕ := 12

/-- The time taken by the second group to build the dam (in hours) -/
def second_group_time : ℕ := 5

/-- Theorem stating that the first group size is correct -/
theorem first_group_size_correct : 
  first_group_size * first_group_time = second_group_size * second_group_time :=
by sorry

end first_group_size_correct_l329_32949


namespace clara_number_problem_l329_32900

theorem clara_number_problem (x : ℝ) : 2 * x + 3 = 23 → x = 10 := by
  sorry

end clara_number_problem_l329_32900


namespace max_value_of_sum_l329_32939

theorem max_value_of_sum (a b c : ℝ) (h : a^2 + b^2 + c^2 = 4) :
  ∃ (max : ℝ), max = 10 * Real.sqrt 2 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 4 → 3*x + 4*y + 5*z ≤ max :=
sorry

end max_value_of_sum_l329_32939


namespace product_of_numbers_l329_32967

theorem product_of_numbers (a b : ℝ) (h1 : a + b = 70) (h2 : a - b = 10) : a * b = 1200 := by
  sorry

end product_of_numbers_l329_32967


namespace inverse_N_expression_l329_32945

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 0; 2, -4]

theorem inverse_N_expression : 
  N⁻¹ = (1 / 12 : ℚ) • N + (1 / 12 : ℚ) • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end inverse_N_expression_l329_32945


namespace lattice_points_on_hyperbola_l329_32915

theorem lattice_points_on_hyperbola : 
  ∃! (points : Finset (ℤ × ℤ)), 
    points.card = 4 ∧ 
    ∀ (x y : ℤ), (x, y) ∈ points ↔ x^2 - y^2 = 61 := by
  sorry

end lattice_points_on_hyperbola_l329_32915
