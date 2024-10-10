import Mathlib

namespace jordan_wins_l3192_319275

theorem jordan_wins (peter_wins peter_losses emma_wins emma_losses jordan_losses : ℕ)
  (h1 : peter_wins = 5)
  (h2 : peter_losses = 4)
  (h3 : emma_wins = 4)
  (h4 : emma_losses = 5)
  (h5 : jordan_losses = 2) :
  ∃ jordan_wins : ℕ,
    jordan_wins = 2 ∧
    2 * (peter_wins + peter_losses + emma_wins + emma_losses + jordan_wins + jordan_losses) =
    peter_wins + emma_wins + jordan_wins + peter_losses + emma_losses + jordan_losses :=
by sorry

end jordan_wins_l3192_319275


namespace honey_jar_problem_l3192_319273

/-- Represents the process of drawing out honey and replacing with sugar solution --/
def draw_and_replace (initial_honey : ℝ) (percent : ℝ) : ℝ :=
  initial_honey * (1 - percent)

/-- The amount of honey remaining after three iterations --/
def remaining_honey (initial_honey : ℝ) : ℝ :=
  draw_and_replace (draw_and_replace (draw_and_replace initial_honey 0.3) 0.4) 0.5

/-- Theorem stating that if 315 grams of honey remain after the process, 
    the initial amount was 1500 grams --/
theorem honey_jar_problem (initial_honey : ℝ) :
  remaining_honey initial_honey = 315 → initial_honey = 1500 := by
  sorry

end honey_jar_problem_l3192_319273


namespace curve_properties_l3192_319296

-- Define the curve
def on_curve (x y : ℝ) : Prop := Real.sqrt x + Real.sqrt y = 1

-- Theorem statement
theorem curve_properties :
  (∀ a b : ℝ, on_curve a b → on_curve b a) ∧
  on_curve 0 1 ∧
  on_curve 1 0 ∧
  on_curve (1/4) (1/4) :=
by sorry

end curve_properties_l3192_319296


namespace downstream_speed_calculation_l3192_319269

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed given the rowing speeds in still water and upstream -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.upstream

/-- Theorem stating that for a man with given upstream and still water speeds, 
    the downstream speed is 65 kmph -/
theorem downstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 60) 
  (h2 : s.upstream = 55) : 
  downstreamSpeed s = 65 := by
  sorry

end downstream_speed_calculation_l3192_319269


namespace tournament_games_count_l3192_319271

/-- Calculates the number of games in a single-elimination tournament. -/
def gamesInSingleElimination (n : ℕ) : ℕ := n - 1

/-- Represents the tournament structure and calculates the total number of games. -/
def totalGames (interestedTeams preliminaryTeams mainTournamentTeams : ℕ) : ℕ :=
  let preliminaryGames := gamesInSingleElimination preliminaryTeams
  let mainTournamentGames := gamesInSingleElimination mainTournamentTeams
  preliminaryGames + mainTournamentGames

/-- Theorem stating that the total number of games in the described tournament is 23. -/
theorem tournament_games_count :
  totalGames 25 9 16 = 23 := by
  sorry

#eval totalGames 25 9 16

end tournament_games_count_l3192_319271


namespace log_identity_l3192_319259

theorem log_identity : Real.log 4 / Real.log 10 + 2 * Real.log 5 / Real.log 10 + 
  (Real.log 5 / Real.log 2) * (Real.log 8 / Real.log 5) = 5 := by
  sorry

end log_identity_l3192_319259


namespace bun_sets_problem_l3192_319264

theorem bun_sets_problem (N : ℕ) : 
  (∃ x y u v : ℕ, 
    3 * x + 5 * y = 25 ∧ 
    3 * u + 5 * v = 35 ∧ 
    x + y = N ∧ 
    u + v = N) → 
  N = 7 := by
sorry

end bun_sets_problem_l3192_319264


namespace sum_of_edges_for_given_pyramid_l3192_319283

/-- Regular hexagonal pyramid with given edge lengths -/
structure RegularHexagonalPyramid where
  base_edge : ℝ
  lateral_edge : ℝ

/-- Sum of all edges of a regular hexagonal pyramid -/
def sum_of_edges (p : RegularHexagonalPyramid) : ℝ :=
  6 * p.base_edge + 6 * p.lateral_edge

/-- Theorem: The sum of all edges of a regular hexagonal pyramid with base edge 8 and lateral edge 13 is 126 -/
theorem sum_of_edges_for_given_pyramid :
  let p : RegularHexagonalPyramid := ⟨8, 13⟩
  sum_of_edges p = 126 := by
  sorry

end sum_of_edges_for_given_pyramid_l3192_319283


namespace solution_set_f_range_of_t_l3192_319226

-- Define the function f
def f (x : ℝ) : ℝ := |x| - 2*|x + 3|

-- Theorem for the first part of the problem
theorem solution_set_f (x : ℝ) :
  f x ≥ 2 ↔ -4 ≤ x ∧ x ≤ -8/3 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_t :
  ∃ t₁ t₂ : ℝ, t₁ = -1/3 ∧ t₂ = 5/3 ∧
  (∀ t : ℝ, (∃ x : ℝ, f x - |3*t - 2| ≥ 0) ↔ t₁ ≤ t ∧ t ≤ t₂) :=
sorry

end solution_set_f_range_of_t_l3192_319226


namespace complex_equation_roots_l3192_319246

theorem complex_equation_roots : 
  let z₁ : ℂ := 1 + Real.sqrt 6 - (Real.sqrt 6 / 2) * Complex.I
  let z₂ : ℂ := 1 - Real.sqrt 6 + (Real.sqrt 6 / 2) * Complex.I
  (z₁^2 - 2*z₁ = 4 - 3*Complex.I) ∧ (z₂^2 - 2*z₂ = 4 - 3*Complex.I) := by
  sorry

end complex_equation_roots_l3192_319246


namespace imaginary_part_of_z_l3192_319285

theorem imaginary_part_of_z (θ : ℝ) : 
  let z : ℂ := Complex.mk (Real.sin (2 * θ) - 1) (Real.sqrt 2 * Real.cos θ - 1)
  (z.re = 0 ∧ z.im ≠ 0) → z.im = -2 := by
  sorry

end imaginary_part_of_z_l3192_319285


namespace necessary_condition_l3192_319211

theorem necessary_condition (a b x y : ℤ) 
  (ha : 0 < a) (hb : 0 < b) 
  (h1 : x - y > a + b) (h2 : x * y > a * b) : 
  x > a ∧ y > b := by
  sorry

end necessary_condition_l3192_319211


namespace remaining_juice_bottles_l3192_319234

/-- Given the initial number of juice bottles in the refrigerator and pantry,
    the number of bottles bought, and the number of bottles consumed,
    calculate the remaining number of bottles. -/
theorem remaining_juice_bottles
  (refrigerator_bottles : ℕ)
  (pantry_bottles : ℕ)
  (bought_bottles : ℕ)
  (consumed_bottles : ℕ)
  (h1 : refrigerator_bottles = 4)
  (h2 : pantry_bottles = 4)
  (h3 : bought_bottles = 5)
  (h4 : consumed_bottles = 3) :
  refrigerator_bottles + pantry_bottles + bought_bottles - consumed_bottles = 10 := by
  sorry

#check remaining_juice_bottles

end remaining_juice_bottles_l3192_319234


namespace stadium_length_l3192_319207

theorem stadium_length (w h p : ℝ) (hw : w = 18) (hh : h = 16) (hp : p = 34) :
  ∃ l : ℝ, l = 24 ∧ p^2 = l^2 + w^2 + h^2 :=
by sorry

end stadium_length_l3192_319207


namespace inverse_of_A_is_B_l3192_319222

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 7; 2, 3]

def B : Matrix (Fin 2) (Fin 2) ℚ := !![-3/2, 7/2; 1, -2]

theorem inverse_of_A_is_B :
  (Matrix.det A ≠ 0) → (A * B = 1 ∧ B * A = 1) :=
by sorry

end inverse_of_A_is_B_l3192_319222


namespace min_value_sum_of_reciprocals_l3192_319291

theorem min_value_sum_of_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  1 / (a^2 + 2*b^2) + 1 / (b^2 + 2*c^2) + 1 / (c^2 + 2*a^2) ≥ 9 ∧
  (1 / (a^2 + 2*b^2) + 1 / (b^2 + 2*c^2) + 1 / (c^2 + 2*a^2) = 9 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
by sorry

end min_value_sum_of_reciprocals_l3192_319291


namespace complement_A_intersect_B_l3192_319257

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 2| ≤ 2}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y = x^2 - 2*x + 2}

-- Theorem statement
theorem complement_A_intersect_B :
  (Set.univ : Set ℝ) \ (A ∩ B) = {x : ℝ | x < 1 ∨ x > 4} := by sorry

end complement_A_intersect_B_l3192_319257


namespace pyramid_angle_closest_to_40_l3192_319268

theorem pyramid_angle_closest_to_40 (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 2017) (h_lateral : lateral_edge = 2000) : 
  let angle := Real.arctan ((base_edge / Real.sqrt 2) / lateral_edge)
  let options := [30, 40, 50, 60]
  (40 : ℝ) ∈ options ∧ 
  ∀ x ∈ options, |angle - 40| ≤ |angle - x| :=
by sorry

end pyramid_angle_closest_to_40_l3192_319268


namespace line_translations_l3192_319204

/-- Represents a line in the form y = mx + b --/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Translates a line vertically --/
def translateVertical (l : Line) (dy : ℝ) : Line :=
  { m := l.m, b := l.b + dy }

/-- Translates a line horizontally --/
def translateHorizontal (l : Line) (dx : ℝ) : Line :=
  { m := l.m, b := l.b - l.m * dx }

theorem line_translations (original : Line) :
  (original.m = 2 ∧ original.b = -4) →
  (translateVertical original 3 = { m := 2, b := -1 } ∧
   translateHorizontal original 3 = { m := 2, b := -10 }) :=
by sorry

end line_translations_l3192_319204


namespace job_completion_time_l3192_319297

theorem job_completion_time 
  (m d r : ℕ) 
  (h1 : m > 0) 
  (h2 : d > 0) 
  (h3 : m + r > 0) : 
  (m * d : ℚ) / (m + r) = (m * d : ℕ) / (m + r) := by
  sorry

end job_completion_time_l3192_319297


namespace maria_carrots_thrown_out_l3192_319232

/-- The number of carrots Maria initially picked -/
def initial_carrots : ℕ := 48

/-- The number of additional carrots Maria picked the next day -/
def additional_carrots : ℕ := 15

/-- The total number of carrots Maria had after picking additional carrots -/
def total_carrots : ℕ := 52

/-- The number of carrots Maria threw out -/
def carrots_thrown_out : ℕ := 11

theorem maria_carrots_thrown_out : 
  initial_carrots - carrots_thrown_out + additional_carrots = total_carrots :=
sorry

end maria_carrots_thrown_out_l3192_319232


namespace number_problem_l3192_319241

theorem number_problem (x : ℝ) : 0.1 * x = 0.2 * 650 + 190 → x = 3200 := by
  sorry

end number_problem_l3192_319241


namespace f_minimum_value_l3192_319214

/-- The function f(x, y) as defined in the problem -/
def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x*y + y^2) - 3 * (x + y) + 5

/-- The theorem stating the minimum value of f(x, y) -/
theorem f_minimum_value :
  (∀ x y : ℝ, x > 0 → y > 0 → f x y ≥ 2) ∧ f (1/2) (1/2) = 2 := by
  sorry

end f_minimum_value_l3192_319214


namespace geometric_sequence_fourth_term_l3192_319212

theorem geometric_sequence_fourth_term 
  (x : ℝ) 
  (h1 : ∃ r : ℝ, (3*x + 3) = x * r) 
  (h2 : ∃ r : ℝ, (6*x + 6) = (3*x + 3) * r) :
  ∃ r : ℝ, -24 = (6*x + 6) * r :=
sorry

end geometric_sequence_fourth_term_l3192_319212


namespace parallel_plane_through_point_l3192_319229

def plane_equation (x y z : ℝ) := 3*x - 2*y + 4*z - 32

theorem parallel_plane_through_point :
  let given_plane (x y z : ℝ) := 3*x - 2*y + 4*z - 6
  (∀ (x y z : ℝ), plane_equation x y z = 0 ↔ ∃ (t : ℝ), given_plane x y z = t) ∧ 
  plane_equation 2 (-3) 5 = 0 ∧
  (∃ (A B C D : ℤ), ∀ (x y z : ℝ), plane_equation x y z = A*x + B*y + C*z + D) ∧
  (∃ (A : ℤ), A > 0 ∧ ∀ (x y z : ℝ), plane_equation x y z = A*x + plane_equation 0 1 0*y + plane_equation 0 0 1*z + plane_equation 0 0 0) ∧
  (Nat.gcd (Int.natAbs 3) (Int.natAbs (-2)) = 1 ∧ 
   Nat.gcd (Int.natAbs 3) (Int.natAbs 4) = 1 ∧ 
   Nat.gcd (Int.natAbs 3) (Int.natAbs (-32)) = 1) :=
by sorry

end parallel_plane_through_point_l3192_319229


namespace gcd_problem_l3192_319260

def gcd_notation (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_problem (b : ℕ) : 
  (gcd_notation (gcd_notation 20 16) (18 * b) = 2) → b = 1 :=
by
  sorry

end gcd_problem_l3192_319260


namespace binomial_coefficient_two_l3192_319221

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l3192_319221


namespace gcd_of_162_180_450_l3192_319239

theorem gcd_of_162_180_450 : Nat.gcd 162 (Nat.gcd 180 450) = 18 := by sorry

end gcd_of_162_180_450_l3192_319239


namespace draw_probability_value_l3192_319294

/-- The number of green chips in the bag -/
def green_chips : ℕ := 4

/-- The number of blue chips in the bag -/
def blue_chips : ℕ := 3

/-- The number of yellow chips in the bag -/
def yellow_chips : ℕ := 5

/-- The total number of chips in the bag -/
def total_chips : ℕ := green_chips + blue_chips + yellow_chips

/-- The number of ways to arrange the color groups (green-blue-yellow or yellow-green-blue) -/
def color_group_arrangements : ℕ := 2

/-- The probability of drawing the chips in the specified order -/
def draw_probability : ℚ :=
  (Nat.factorial green_chips * Nat.factorial blue_chips * Nat.factorial yellow_chips * color_group_arrangements : ℚ) /
  Nat.factorial total_chips

theorem draw_probability_value : draw_probability = 1 / 13860 := by
  sorry

end draw_probability_value_l3192_319294


namespace power_multiplication_l3192_319266

theorem power_multiplication (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end power_multiplication_l3192_319266


namespace decomposition_theorem_l3192_319205

theorem decomposition_theorem (d n : ℕ) (hd : d > 0) (hn : n > 0) :
  ∃ (A B : Set ℕ), 
    (∀ k : ℕ, k > 0 → (k ∈ A ∨ k ∈ B)) ∧
    (A ∩ B = ∅) ∧
    (∀ x ∈ A, ∃ y ∈ B, d * x = n * d * y) ∧
    (∀ y ∈ B, ∃ x ∈ A, d * x = n * d * y) :=
sorry

end decomposition_theorem_l3192_319205


namespace order_of_powers_l3192_319223

theorem order_of_powers : 
  let a : ℕ := 3^55
  let b : ℕ := 4^44
  let c : ℕ := 5^33
  c < a ∧ a < b := by sorry

end order_of_powers_l3192_319223


namespace max_value_constraint_l3192_319227

theorem max_value_constraint (x y : ℝ) : 
  x^2 + y^2 = 18*x + 8*y + 10 → (∀ a b : ℝ, a^2 + b^2 = 18*a + 8*b + 10 → 4*x + 3*y ≥ 4*a + 3*b) → 4*x + 3*y = 45 := by
  sorry

end max_value_constraint_l3192_319227


namespace canal_digging_time_l3192_319258

/-- Represents the time taken to dig a canal given the number of men, hours per day, and days worked. -/
def diggingTime (men : ℕ) (hoursPerDay : ℕ) (days : ℚ) : ℚ := men * hoursPerDay * days

/-- Theorem stating that 30 men working 8 hours a day will take 1.5 days to dig a canal
    that originally took 20 men working 6 hours a day for 3 days, assuming constant work rate. -/
theorem canal_digging_time :
  diggingTime 20 6 3 = diggingTime 30 8 (3/2 : ℚ) := by
  sorry

#check canal_digging_time

end canal_digging_time_l3192_319258


namespace prob_sum_greater_than_seven_l3192_319200

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when two dice are rolled -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of favorable outcomes (sum > 7) -/
def favorableOutcomes : ℕ := totalOutcomes - 21

/-- The probability of the sum being greater than 7 when two fair dice are rolled -/
def probSumGreaterThanSeven : ℚ := favorableOutcomes / totalOutcomes

theorem prob_sum_greater_than_seven :
  probSumGreaterThanSeven = 5 / 12 := by
  sorry

end prob_sum_greater_than_seven_l3192_319200


namespace sum_yz_zero_percent_of_x_l3192_319208

theorem sum_yz_zero_percent_of_x (x y z : ℚ) 
  (h1 : (3/5) * (x - y) = (3/10) * (x + y))
  (h2 : (2/5) * (x + z) = (1/5) * (y + z))
  (h3 : (1/2) * (x - z) = (1/4) * (x + y + z)) :
  y + z = 0 * x :=
by sorry

end sum_yz_zero_percent_of_x_l3192_319208


namespace max_value_of_expression_l3192_319267

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 3) :
  ∃ (x y z : ℝ), x^2 + y^2 + z^2 = 3 ∧ 2*x*y + 3*z = 21/4 ∧ ∀ (a b c : ℝ), a^2 + b^2 + c^2 = 3 → 2*a*b + 3*c ≤ 21/4 :=
sorry

end max_value_of_expression_l3192_319267


namespace ninth_grade_class_distribution_l3192_319298

theorem ninth_grade_class_distribution (total students_science students_programming : ℕ) 
  (h_total : total = 120)
  (h_science : students_science = 80)
  (h_programming : students_programming = 75) :
  students_science - (total - (students_science + students_programming - total)) = 45 :=
sorry

end ninth_grade_class_distribution_l3192_319298


namespace cans_in_sixth_bin_l3192_319253

theorem cans_in_sixth_bin (n : ℕ) (cans : ℕ → ℕ) : 
  (∀ k, cans k = k * (k + 1) / 2) → cans 6 = 22 := by
  sorry

end cans_in_sixth_bin_l3192_319253


namespace water_capacity_equals_volume_l3192_319209

/-- A cylindrical bucket -/
structure CylindricalBucket where
  volume : ℝ
  lateral_area : ℝ
  surface_area : ℝ

/-- The amount of water a cylindrical bucket can hold -/
def water_capacity (bucket : CylindricalBucket) : ℝ := sorry

/-- Theorem: The amount of water a cylindrical bucket can hold is equal to its volume -/
theorem water_capacity_equals_volume (bucket : CylindricalBucket) :
  water_capacity bucket = bucket.volume := sorry

end water_capacity_equals_volume_l3192_319209


namespace quadratic_equation_linear_coefficient_l3192_319249

theorem quadratic_equation_linear_coefficient :
  ∀ a b c : ℝ, 
    (∀ x, 2 * x^2 = 3 * x - 1 ↔ a * x^2 + b * x + c = 0) →
    a = 2 →
    b = -3 := by
  sorry

end quadratic_equation_linear_coefficient_l3192_319249


namespace parallel_vectors_m_value_l3192_319228

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (-6, 2)
  let b : ℝ × ℝ := (m, -3)
  parallel a b → m = 9 := by
sorry

end parallel_vectors_m_value_l3192_319228


namespace quadratic_root_condition_l3192_319202

theorem quadratic_root_condition (a : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ > 1 ∧ r₂ < 1 ∧ r₁^2 + 2*a*r₁ + 1 = 0 ∧ r₂^2 + 2*a*r₂ + 1 = 0) → 
  a < -1 :=
by sorry

end quadratic_root_condition_l3192_319202


namespace candy_car_cost_proof_l3192_319272

/-- The cost of a candy car given initial amount and change received -/
def candy_car_cost (initial_amount change : ℚ) : ℚ :=
  initial_amount - change

/-- Theorem stating the cost of the candy car is $0.45 -/
theorem candy_car_cost_proof (initial_amount change : ℚ) 
  (h1 : initial_amount = 1.80)
  (h2 : change = 1.35) : 
  candy_car_cost initial_amount change = 0.45 := by
  sorry

end candy_car_cost_proof_l3192_319272


namespace solution_value_l3192_319254

/-- A system of two linear equations in two variables -/
structure LinearSystem :=
  (a b c : ℝ)
  (d e f : ℝ)

/-- The condition for a linear system to not have a unique solution -/
def noUniqueSolution (sys : LinearSystem) : Prop :=
  sys.a * sys.e = sys.b * sys.d ∧ sys.a * sys.f = sys.c * sys.d

/-- The theorem stating that if the given system doesn't have a unique solution, then d = 40 -/
theorem solution_value (k : ℝ) :
  let sys : LinearSystem := ⟨12, 16, d, k, 12, 30⟩
  noUniqueSolution sys → d = 40 :=
by
  sorry


end solution_value_l3192_319254


namespace equation_solution_l3192_319292

theorem equation_solution : ∃ x₁ x₂ : ℝ,
  (1 / (x₁ + 10) + 1 / (x₁ + 8) = 1 / (x₁ + 11) + 1 / (x₁ + 7) + 1 / (2 * x₁ + 36)) ∧
  (1 / (x₂ + 10) + 1 / (x₂ + 8) = 1 / (x₂ + 11) + 1 / (x₂ + 7) + 1 / (2 * x₂ + 36)) ∧
  (5 * x₁^2 + 140 * x₁ + 707 = 0) ∧
  (5 * x₂^2 + 140 * x₂ + 707 = 0) ∧
  x₁ ≠ x₂ :=
by sorry

end equation_solution_l3192_319292


namespace interest_rate_for_doubling_l3192_319279

/-- The time in years for the money to double --/
def doubling_time : ℝ := 4

/-- The interest rate as a decimal --/
def interest_rate : ℝ := 0.25

/-- Simple interest formula: Final amount = Principal * (1 + rate * time) --/
def simple_interest (principal rate time : ℝ) : ℝ := principal * (1 + rate * time)

theorem interest_rate_for_doubling :
  simple_interest 1 interest_rate doubling_time = 2 := by sorry

end interest_rate_for_doubling_l3192_319279


namespace division_438_by_4_result_l3192_319280

/-- Represents the place value of a digit in a number -/
inductive PlaceValue
  | Ones
  | Tens
  | Hundreds
  | Thousands

/-- Represents a division operation with its result -/
structure DivisionResult (dividend : ℕ) (divisor : ℕ) where
  quotient : ℕ
  remainder : ℕ
  highest_place_value : PlaceValue
  valid : dividend = quotient * divisor + remainder
  remainder_bound : remainder < divisor

/-- The division of 438 by 4 -/
def division_438_by_4 : DivisionResult 438 4 := sorry

theorem division_438_by_4_result :
  division_438_by_4.highest_place_value = PlaceValue.Hundreds ∧
  division_438_by_4.remainder = 2 := by sorry

end division_438_by_4_result_l3192_319280


namespace total_lemons_l3192_319217

def lemon_problem (levi jayden eli ian : ℕ) : Prop :=
  levi = 5 ∧
  jayden = levi + 6 ∧
  3 * jayden = eli ∧
  2 * eli = ian ∧
  levi + jayden + eli + ian = 115

theorem total_lemons : ∃ levi jayden eli ian : ℕ, lemon_problem levi jayden eli ian := by
  sorry

end total_lemons_l3192_319217


namespace palm_meadows_beds_l3192_319295

theorem palm_meadows_beds (total_rooms : ℕ) (two_bed_rooms : ℕ) (total_beds : ℕ) 
  (h1 : total_rooms = 13)
  (h2 : two_bed_rooms = 8)
  (h3 : total_beds = 31)
  (h4 : two_bed_rooms ≤ total_rooms) :
  (total_beds - 2 * two_bed_rooms) / (total_rooms - two_bed_rooms) = 3 := by
  sorry

end palm_meadows_beds_l3192_319295


namespace arc_length_of_sector_l3192_319278

/-- Given a sector with a central angle of 60° and a radius of 6 cm, 
    the length of the arc is equal to 2π cm. -/
theorem arc_length_of_sector (α : Real) (r : Real) : 
  α = 60 * π / 180 → r = 6 → α * r = 2 * π := by sorry

end arc_length_of_sector_l3192_319278


namespace existence_of_special_point_l3192_319236

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a 2D plane -/
def Point := ℝ × ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  point : Point
  direction : ℝ × ℝ

/-- Checks if a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

/-- Checks if a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop :=
  sorry -- Definition of line-circle intersection

/-- Main theorem -/
theorem existence_of_special_point (c1 c2 : Circle) :
  ∃ p : Point, (isOutside p c1 ∧ isOutside p c2) ∧
    ∀ l : Line, l.point = p → (intersects l c1 ∨ intersects l c2) :=
  sorry

end existence_of_special_point_l3192_319236


namespace investment_of_a_l3192_319225

/-- Represents a partnership investment. -/
structure Partnership where
  investmentA : ℝ
  investmentB : ℝ
  investmentC : ℝ
  profitB : ℝ
  profitA : ℝ
  months : ℕ

/-- Theorem stating the investment of partner A given the conditions. -/
theorem investment_of_a (p : Partnership) 
  (hB : p.investmentB = 21000)
  (hprofitB : p.profitB = 1540)
  (hprofitA : p.profitA = 1100)
  (hmonths : p.months = 8)
  (h_profit_prop : p.profitA / p.profitB = p.investmentA / p.investmentB) :
  p.investmentA = 15000 :=
sorry

end investment_of_a_l3192_319225


namespace inscribed_cube_surface_area_l3192_319255

theorem inscribed_cube_surface_area (outer_cube_area : ℝ) (h : outer_cube_area = 54) :
  let outer_side := Real.sqrt (outer_cube_area / 6)
  let sphere_diameter := outer_side
  let inner_side := Real.sqrt (sphere_diameter^2 / 3)
  let inner_cube_area := 6 * inner_side^2
  inner_cube_area = 18 := by sorry

end inscribed_cube_surface_area_l3192_319255


namespace gym_guests_ratio_l3192_319281

/-- Represents the number of guests entering the gym each hour -/
structure GymGuests where
  first_hour : ℕ
  second_hour : ℕ
  third_hour : ℕ
  fourth_hour : ℕ

/-- Calculates the total number of guests -/
def total_guests (g : GymGuests) : ℕ :=
  g.first_hour + g.second_hour + g.third_hour + g.fourth_hour

theorem gym_guests_ratio (total_towels : ℕ) (g : GymGuests) : 
  total_towels = 300 →
  g.first_hour = 50 →
  g.second_hour = (120 * g.first_hour) / 100 →
  g.third_hour = (125 * g.second_hour) / 100 →
  g.fourth_hour > g.third_hour →
  total_guests g = 285 →
  (g.fourth_hour - g.third_hour) * 3 = g.third_hour := by
  sorry

#check gym_guests_ratio

end gym_guests_ratio_l3192_319281


namespace quadratic_always_real_roots_roots_difference_one_l3192_319216

/-- The quadratic equation x^2 + (m+3)x + m+2 = 0 -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (m+3)*x + m+2

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := (m+3)^2 - 4*(m+2)

theorem quadratic_always_real_roots (m : ℝ) :
  discriminant m ≥ 0 := by sorry

theorem roots_difference_one (m : ℝ) :
  (∃ a b, quadratic m a = 0 ∧ quadratic m b = 0 ∧ |a - b| = 1) →
  (m = -2 ∨ m = 0) := by sorry

end quadratic_always_real_roots_roots_difference_one_l3192_319216


namespace nilpotent_in_finite_ring_l3192_319210

/-- A ring with exactly n elements -/
class FiniteRing (A : Type) extends Ring A where
  card : ℕ
  finite : Fintype A
  card_eq : Fintype.card A = card

theorem nilpotent_in_finite_ring {n m : ℕ} {A : Type} [FiniteRing A] (hn : n ≥ 2) (hm : m ≥ 0) 
  (h_card : (FiniteRing.card A) = n) (a : A) 
  (h_inv : ∀ k ∈ Finset.range (n - 1), k ≥ m + 1 → IsUnit (1 - a ^ (k + 1))) :
  ∃ k : ℕ, a ^ k = 0 := by
  sorry

end nilpotent_in_finite_ring_l3192_319210


namespace either_equal_or_irrational_l3192_319231

theorem either_equal_or_irrational (m : ℤ) (n : ℝ) 
  (h : m^2 + 1/n = n^2 + 1/m) : n = m ∨ ¬(∃ (p q : ℤ), n = p / q) :=
by sorry

end either_equal_or_irrational_l3192_319231


namespace pinwheel_area_l3192_319252

-- Define the constants
def π : ℝ := 3.14
def large_diameter : ℝ := 20

-- Define the theorem
theorem pinwheel_area : 
  let large_radius : ℝ := large_diameter / 2
  let small_radius : ℝ := large_radius / 2
  let large_area : ℝ := π * large_radius ^ 2
  let small_area : ℝ := π * small_radius ^ 2
  let cut_out_area : ℝ := 2 * small_area
  let remaining_area : ℝ := large_area - cut_out_area
  remaining_area = 157 := by
sorry

end pinwheel_area_l3192_319252


namespace quadratic_equation_transformation_l3192_319276

theorem quadratic_equation_transformation (x : ℝ) :
  x^2 + 6*x - 1 = 0 →
  ∃ (m n : ℝ), (x + m)^2 = n ∧ m - n = -7 := by
  sorry

end quadratic_equation_transformation_l3192_319276


namespace ellipse_properties_l3192_319256

/-- Given an ellipse with equation x²/25 + y²/9 = 1, prove its semi-major axis length and eccentricity -/
theorem ellipse_properties : ∃ (a b c : ℝ), 
  (∀ x y : ℝ, x^2/25 + y^2/9 = 1 → 
    a = 5 ∧ 
    b = 3 ∧ 
    c^2 = a^2 - b^2 ∧ 
    a = 5 ∧ 
    c/a = 4/5) := by
  sorry

end ellipse_properties_l3192_319256


namespace min_value_reciprocal_sum_l3192_319230

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 2) :
  (1/a + 1/b) ≥ (3 + 2*Real.sqrt 2) / 2 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 2 ∧ 1/a₀ + 1/b₀ = (3 + 2*Real.sqrt 2) / 2 :=
sorry

end min_value_reciprocal_sum_l3192_319230


namespace no_valid_pop_of_223_l3192_319293

/-- Represents the population of Minerva -/
structure MinervaPop where
  people : ℕ
  horses : ℕ
  sheep : ℕ
  cows : ℕ
  ducks : ℕ

/-- Checks if a given population satisfies the Minerva conditions -/
def isValidMinervaPop (pop : MinervaPop) : Prop :=
  pop.people = 4 * pop.horses ∧
  pop.sheep = 3 * pop.cows ∧
  pop.ducks = 2 * pop.people - 2

/-- The total population of Minerva -/
def totalPop (pop : MinervaPop) : ℕ :=
  pop.people + pop.horses + pop.sheep + pop.cows + pop.ducks

/-- Theorem stating that 223 cannot be the total population of Minerva -/
theorem no_valid_pop_of_223 :
  ¬ ∃ (pop : MinervaPop), isValidMinervaPop pop ∧ totalPop pop = 223 := by
  sorry

end no_valid_pop_of_223_l3192_319293


namespace return_trip_time_l3192_319203

/-- Represents the flight scenario between two towns -/
structure FlightScenario where
  d : ℝ  -- distance between towns
  p : ℝ  -- plane speed in still air
  w : ℝ  -- wind speed
  t : ℝ  -- time for return trip in still air

/-- The conditions of the flight scenario -/
def flight_conditions (f : FlightScenario) : Prop :=
  f.w = (1/3) * f.p ∧  -- wind speed is one-third of plane speed
  f.d / (f.p - f.w) = 120 ∧  -- time against wind
  f.d / (f.p + f.w) = f.t - 20  -- time with wind

/-- The theorem to prove -/
theorem return_trip_time (f : FlightScenario) 
  (h : flight_conditions f) : f.d / (f.p + f.w) = 60 := by
  sorry

#check return_trip_time

end return_trip_time_l3192_319203


namespace cube_surface_area_l3192_319237

theorem cube_surface_area (a : ℝ) : 
  let edge_length : ℝ := 5 * a
  let surface_area : ℝ := 6 * (edge_length ^ 2)
  surface_area = 150 * (a ^ 2) :=
by sorry

end cube_surface_area_l3192_319237


namespace red_balls_count_l3192_319299

/-- Given a bag with 2400 balls of three colors (red, green, blue) distributed
    in the ratio 15:13:17, prove that the number of red balls is 795. -/
theorem red_balls_count (total : ℕ) (red green blue : ℕ) :
  total = 2400 →
  red + green + blue = total →
  red * 13 = green * 15 →
  red * 17 = blue * 15 →
  red = 795 := by
  sorry

end red_balls_count_l3192_319299


namespace intersection_distance_difference_l3192_319262

/-- The line y - 2x - 1 = 0 intersects the parabola y^2 = 4x + 1 at points C and D.
    Q is the point (2, 0). -/
theorem intersection_distance_difference (C D Q : ℝ × ℝ) : 
  (C.2 - 2 * C.1 - 1 = 0) →
  (D.2 - 2 * D.1 - 1 = 0) →
  (C.2^2 = 4 * C.1 + 1) →
  (D.2^2 = 4 * D.1 + 1) →
  (Q = (2, 0)) →
  |Real.sqrt ((C.1 - Q.1)^2 + (C.2 - Q.2)^2) - Real.sqrt ((D.1 - Q.1)^2 + (D.2 - Q.2)^2)| = 2 * Real.sqrt 2 :=
by sorry

end intersection_distance_difference_l3192_319262


namespace lucky_coin_steps_l3192_319206

/-- Represents the state of a coin on the number line -/
inductive CoinState
| HeadsUp
| TailsUp
| NoCoin

/-- Represents the direction Lucky is facing -/
inductive Direction
| Positive
| Negative

/-- Represents Lucky's position and the state of the number line -/
structure GameState where
  position : Int
  direction : Direction
  coins : Int → CoinState

/-- Represents the procedure Lucky follows -/
def step (state : GameState) : GameState :=
  sorry

/-- Counts the number of tails-up coins -/
def countTailsUp (coins : Int → CoinState) : Nat :=
  sorry

/-- Theorem stating that the process stops after 6098 steps -/
theorem lucky_coin_steps :
  ∀ (initial : GameState),
    initial.position = 0 ∧
    initial.direction = Direction.Positive ∧
    (∀ n : Int, initial.coins n = CoinState.HeadsUp) →
    ∃ (final : GameState) (steps : Nat),
      steps = 6098 ∧
      countTailsUp final.coins = 20 ∧
      (∀ k : Nat, k < steps → countTailsUp (step^[k] initial).coins < 20) :=
  sorry

end lucky_coin_steps_l3192_319206


namespace problem_statement_l3192_319245

theorem problem_statement (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 1) 
  (heq : a^2 + 4*b^2 + c^2 - 2*c = 2) : 
  (a + 2*b + c ≤ 4) ∧ 
  (a = 2*b → 1/b + 1/(c-1) ≥ 3) := by
  sorry

end problem_statement_l3192_319245


namespace max_cables_equals_max_edges_l3192_319238

/-- Represents a bipartite graph with two sets of nodes -/
structure BipartiteGraph where
  setA : Nat
  setB : Nat

/-- Calculates the maximum number of edges in a bipartite graph -/
def maxEdges (g : BipartiteGraph) : Nat :=
  g.setA * g.setB

/-- Represents the company's computer network -/
def companyNetwork : BipartiteGraph :=
  { setA := 16, setB := 12 }

/-- The maximum number of cables needed is equal to the maximum number of edges in the bipartite graph -/
theorem max_cables_equals_max_edges :
  maxEdges companyNetwork = 192 := by
  sorry

#eval maxEdges companyNetwork

end max_cables_equals_max_edges_l3192_319238


namespace parallel_vectors_trig_identity_l3192_319233

/-- Given two vectors a and b that are parallel, prove that sin²α + 2sinα*cosα = 3/2 -/
theorem parallel_vectors_trig_identity (α : ℝ) : 
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (Real.sin (α - π/3), Real.cos α + π/3)
  (∃ k : ℝ, b = k • a) →
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α = 3/2 := by
  sorry

end parallel_vectors_trig_identity_l3192_319233


namespace smallest_product_smallest_product_is_neg_32_l3192_319287

def S : Finset Int := {-8, -3, -2, 2, 4}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y ≤ a * b :=
by
  sorry

theorem smallest_product_is_neg_32 :
  ∃ (a b : Int), a ∈ S ∧ b ∈ S ∧ a * b = -32 ∧
  ∀ (x y : Int), x ∈ S → y ∈ S → a * b ≤ x * y :=
by
  sorry

end smallest_product_smallest_product_is_neg_32_l3192_319287


namespace sum_of_squares_of_roots_l3192_319270

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 5*x₁ + 3 = 0 → 
  x₂^2 - 5*x₂ + 3 = 0 → 
  x₁^2 + x₂^2 = 19 := by
sorry

end sum_of_squares_of_roots_l3192_319270


namespace sphere_volume_equal_surface_area_implies_radius_three_l3192_319235

theorem sphere_volume_equal_surface_area_implies_radius_three 
  (r : ℝ) (h : r > 0) : 
  (4 / 3 * Real.pi * r^3 = 4 * Real.pi * r^2) → r = 3 := by
  sorry

end sphere_volume_equal_surface_area_implies_radius_three_l3192_319235


namespace equation_solution_l3192_319251

theorem equation_solution (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 4/3) :
  ∃! x : ℝ, (Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x) ∧
            (x = (4 - p) / Real.sqrt (8 * (2 - p))) := by
  sorry

end equation_solution_l3192_319251


namespace shopkeeper_red_cards_l3192_319289

/-- Calculates the total number of red cards in all decks --/
def total_red_cards (total_decks : ℕ) (standard_decks : ℕ) (special_decks : ℕ) 
  (red_cards_standard : ℕ) (additional_red_cards_special : ℕ) : ℕ :=
  (standard_decks * red_cards_standard) + 
  (special_decks * (red_cards_standard + additional_red_cards_special))

theorem shopkeeper_red_cards : 
  total_red_cards 15 5 10 26 4 = 430 := by
  sorry

#eval total_red_cards 15 5 10 26 4

end shopkeeper_red_cards_l3192_319289


namespace geometric_sequence_minimum_l3192_319284

/-- 
Given a geometric sequence {a_n} with positive terms and common ratio q > 1,
if a_5 + a_4 - a_3 - a_2 = 5, then a_6 + a_7 ≥ 20.
-/
theorem geometric_sequence_minimum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  q > 1 →
  (∀ n, a (n + 1) = q * a n) →
  a 5 + a 4 - a 3 - a 2 = 5 →
  a 6 + a 7 ≥ 20 := by
  sorry

end geometric_sequence_minimum_l3192_319284


namespace birds_and_nests_difference_l3192_319282

theorem birds_and_nests_difference :
  let num_birds : ℕ := 6
  let num_nests : ℕ := 3
  num_birds - num_nests = 3 :=
by sorry

end birds_and_nests_difference_l3192_319282


namespace order_of_x_l3192_319220

theorem order_of_x (x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅ : ℝ)
  (eq1 : x₁ + x₂ + x₃ = a₁)
  (eq2 : x₂ + x₃ + x₄ = a₂)
  (eq3 : x₃ + x₄ + x₅ = a₃)
  (eq4 : x₄ + x₅ + x₁ = a₄)
  (eq5 : x₅ + x₁ + x₂ = a₅)
  (h : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅) :
  x₃ > x₁ ∧ x₁ > x₄ ∧ x₄ > x₂ ∧ x₂ > x₅ :=
by sorry


end order_of_x_l3192_319220


namespace product_b_original_price_l3192_319286

theorem product_b_original_price 
  (price_a : ℝ) 
  (price_b : ℝ) 
  (initial_relation : price_a = 1.2 * price_b)
  (price_a_after : ℝ)
  (price_a_decrease : price_a_after = 0.9 * price_a)
  (price_a_final : price_a_after = 198)
  : price_b = 183.33 := by
  sorry

end product_b_original_price_l3192_319286


namespace captain_america_awakening_year_l3192_319250

theorem captain_america_awakening_year : 2019 * 0.313 + 2.019 * 687 = 2018 := by
  sorry

end captain_america_awakening_year_l3192_319250


namespace arithmetic_sequence_difference_l3192_319201

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_difference (a₁_A a₁_B d_A d_B : ℝ) (n : ℕ) :
  a₁_A = 20 ∧ a₁_B = 40 ∧ d_A = 12 ∧ d_B = -12 ∧ n = 51 →
  |arithmetic_sequence a₁_A d_A n - arithmetic_sequence a₁_B d_B n| = 1180 :=
sorry

end arithmetic_sequence_difference_l3192_319201


namespace greatest_integer_b_for_all_real_domain_l3192_319261

theorem greatest_integer_b_for_all_real_domain : ∃ b : ℤ,
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ 0) ∧
  (∀ c : ℤ, c > b → ∃ x : ℝ, x^2 + (c : ℝ) * x + 10 = 0) ∧
  b = 6 :=
by sorry

end greatest_integer_b_for_all_real_domain_l3192_319261


namespace range_of_f_l3192_319243

/-- The function f(x) = |x+3| - |x-5| -/
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

/-- The range of f is [-8, 18] -/
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -8 ≤ y ∧ y ≤ 18 := by
  sorry

end range_of_f_l3192_319243


namespace square_side_length_l3192_319290

/-- Proves that a square with perimeter 52 cm and area 169 square cm has sides of length 13 cm -/
theorem square_side_length (s : ℝ) 
  (perimeter : s * 4 = 52) 
  (area : s * s = 169) : 
  s = 13 := by
  sorry

end square_side_length_l3192_319290


namespace max_distinct_values_l3192_319240

-- Define a 4x4 grid of non-negative integers
def Grid := Matrix (Fin 4) (Fin 4) ℕ

-- Define a function to check if a set of 5 cells sums to 5
def SumToFive (g : Grid) (cells : Finset (Fin 4 × Fin 4)) : Prop :=
  cells.card = 5 ∧ (cells.sum (fun c => g c.1 c.2) = 5)

-- Define the property that all valid 5-cell configurations sum to 5
def AllConfigsSumToFive (g : Grid) : Prop :=
  ∀ cells : Finset (Fin 4 × Fin 4), SumToFive g cells

-- Define the number of distinct values in the grid
def DistinctValues (g : Grid) : ℕ :=
  (Finset.univ.image (fun i => Finset.univ.image (g i))).card

-- State the theorem
theorem max_distinct_values (g : Grid) (h : AllConfigsSumToFive g) :
  DistinctValues g ≤ 3 :=
sorry

end max_distinct_values_l3192_319240


namespace difference_expression_correct_l3192_319219

/-- The expression that represents "the difference between the opposite of a and 5 times b" -/
def difference_expression (a b : ℝ) : ℝ := -a - 5*b

/-- The difference_expression correctly represents "the difference between the opposite of a and 5 times b" -/
theorem difference_expression_correct (a b : ℝ) :
  difference_expression a b = (-a) - (5*b) := by sorry

end difference_expression_correct_l3192_319219


namespace square_product_equals_sum_implies_zero_l3192_319218

theorem square_product_equals_sum_implies_zero (x y : ℤ) 
  (h : x^2 * y^2 = x^2 + y^2) : x = 0 ∧ y = 0 := by
  sorry

end square_product_equals_sum_implies_zero_l3192_319218


namespace binary_sum_to_hex_l3192_319213

/-- The sum of 11111111111₂ and 11111111₂ in base 16 is 8FE₁₆ -/
theorem binary_sum_to_hex : 
  (fun (n : ℕ) => (2^11 - 1) + (2^8 - 1)) 0 = 
  (fun (n : ℕ) => 8 * 16^2 + 15 * 16^1 + 14 * 16^0) 0 := by
  sorry

end binary_sum_to_hex_l3192_319213


namespace divisible_by_three_l3192_319224

theorem divisible_by_three (k : ℤ) : 3 ∣ ((2*k + 3)^2 - 4*k^2) := by
  sorry

end divisible_by_three_l3192_319224


namespace range_of_a_l3192_319247

theorem range_of_a (a : ℝ) : 
  (¬∃x ∈ Set.Icc 0 1, x^2 - 2*x - 2 + a > 0) ∧
  (¬∀x : ℝ, x^2 - 2*x - a ≠ 0) →
  a ∈ Set.Icc (-1) 2 :=
by sorry

end range_of_a_l3192_319247


namespace triangle_third_side_l3192_319277

theorem triangle_third_side (a b c : ℝ) (angle : ℝ) : 
  a = 9 → b = 12 → angle = 150 * π / 180 → 
  c^2 = a^2 + b^2 - 2*a*b*(angle.cos) → 
  c = Real.sqrt (225 + 108 * Real.sqrt 3) := by
  sorry

end triangle_third_side_l3192_319277


namespace lunchroom_tables_l3192_319274

theorem lunchroom_tables (total_students : ℕ) (students_per_table : ℕ) (h1 : total_students = 204) (h2 : students_per_table = 6) :
  total_students / students_per_table = 34 := by
  sorry

end lunchroom_tables_l3192_319274


namespace inequality_solution_range_l3192_319244

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, |x - 2| - |x - 5| - k > 0) → k < -3 :=
by sorry

end inequality_solution_range_l3192_319244


namespace last_red_ball_fourth_draw_probability_l3192_319288

def initial_white_balls : ℕ := 8
def initial_red_balls : ℕ := 2
def total_balls : ℕ := initial_white_balls + initial_red_balls
def draws : ℕ := 4

def favorable_outcomes : ℕ := (Nat.choose 3 1) * (Nat.choose initial_white_balls 2)
def total_outcomes : ℕ := Nat.choose total_balls draws

theorem last_red_ball_fourth_draw_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 5 := by sorry

end last_red_ball_fourth_draw_probability_l3192_319288


namespace sufficient_but_not_necessary_l3192_319242

theorem sufficient_but_not_necessary (x : ℝ) :
  (|x - 1/2| < 1/2 → x^3 < 1) ∧
  ∃ y : ℝ, y^3 < 1 ∧ |y - 1/2| ≥ 1/2 :=
by sorry

end sufficient_but_not_necessary_l3192_319242


namespace min_balls_to_guarantee_20_l3192_319248

/-- Represents the number of balls of each color in the box -/
structure BoxContents where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee at least n of a single color -/
def minBallsToGuarantee (box : BoxContents) (n : Nat) : Nat :=
  sorry

/-- The specific box contents from the problem -/
def problemBox : BoxContents :=
  { red := 36, green := 24, yellow := 18, blue := 15, white := 12, black := 10 }

/-- The theorem stating the minimum number of balls to draw -/
theorem min_balls_to_guarantee_20 :
  minBallsToGuarantee problemBox 20 = 94 := by
  sorry

end min_balls_to_guarantee_20_l3192_319248


namespace quadratic_equation_solution_l3192_319263

/-- Prove that the equation (ax^2 - 2xy + y^2) - (-x^2 + bxy + 2y^2) = 5x^2 - 9xy + cy^2 
    holds true if and only if a = 4, b = 7, and c = -1 -/
theorem quadratic_equation_solution (a b c : ℝ) (x y : ℝ) :
  (a * x^2 - 2 * x * y + y^2) - (-x^2 + b * x * y + 2 * y^2) = 5 * x^2 - 9 * x * y + c * y^2 ↔ 
  a = 4 ∧ b = 7 ∧ c = -1 := by
sorry

end quadratic_equation_solution_l3192_319263


namespace investment_dividend_income_l3192_319265

/-- Calculates the annual dividend income based on investment parameters -/
def annual_dividend_income (investment : ℚ) (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) : ℚ :=
  let num_shares := investment / quoted_price
  let dividend_per_share := (dividend_rate / 100) * face_value
  num_shares * dividend_per_share

/-- Theorem stating that the annual dividend income for the given parameters is 728 -/
theorem investment_dividend_income :
  annual_dividend_income 4940 10 9.50 14 = 728 := by
  sorry

end investment_dividend_income_l3192_319265


namespace intersection_of_sets_l3192_319215

theorem intersection_of_sets :
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {1, 3}
  A ∩ B = {1, 3} := by
sorry

end intersection_of_sets_l3192_319215
