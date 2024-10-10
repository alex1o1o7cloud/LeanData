import Mathlib

namespace inequality_chain_l2452_245231

theorem inequality_chain (a b x : ℝ) (h1 : b < x) (h2 : x < a) (h3 : a < 0) :
  x^2 > a*x ∧ a*x > b^2 := by
  sorry

end inequality_chain_l2452_245231


namespace find_p_l2452_245262

theorem find_p (P Q : ℝ) (h1 : P + Q = 16) (h2 : P - Q = 4) : P = 10 := by
  sorry

end find_p_l2452_245262


namespace wilsborough_change_l2452_245239

/-- Calculates the change Mrs. Wilsborough received after buying concert tickets -/
theorem wilsborough_change : 
  let vip_price : ℕ := 120
  let regular_price : ℕ := 60
  let discount_price : ℕ := 30
  let vip_count : ℕ := 4
  let regular_count : ℕ := 5
  let discount_count : ℕ := 3
  let payment : ℕ := 1000
  let total_cost : ℕ := vip_price * vip_count + regular_price * regular_count + discount_price * discount_count
  payment - total_cost = 130 := by
  sorry

end wilsborough_change_l2452_245239


namespace sequence_properties_l2452_245299

def sequence_a (n : ℕ) : ℝ := 1 - 2^n

def sum_S (n : ℕ) : ℝ := n + 2 - 2^(n+1)

theorem sequence_properties :
  ∀ (n : ℕ), n ≥ 1 → 
  (∃ (a : ℕ → ℝ) (S : ℕ → ℝ), 
    (∀ k, k ≥ 1 → S k = 2 * a k + k) ∧ 
    (∃ r : ℝ, ∀ k, k ≥ 1 → a (k+1) - 1 = r * (a k - 1)) ∧
    (∀ k, k ≥ 1 → a k = sequence_a k) ∧
    (∀ k, k ≥ 1 → S k = sum_S k)) :=
by
  sorry

end sequence_properties_l2452_245299


namespace multiply_mixed_number_l2452_245252

theorem multiply_mixed_number : 7 * (12 + 1/4) = 85 + 3/4 := by
  sorry

end multiply_mixed_number_l2452_245252


namespace tumbler_price_l2452_245243

theorem tumbler_price (num_tumblers : ℕ) (num_bills : ℕ) (bill_value : ℕ) (change : ℕ) :
  num_tumblers = 10 →
  num_bills = 5 →
  bill_value = 100 →
  change = 50 →
  (num_bills * bill_value - change) / num_tumblers = 45 := by
sorry

end tumbler_price_l2452_245243


namespace hospital_nurse_count_l2452_245281

/-- Given a hospital with doctors and nurses, calculate the number of nurses -/
theorem hospital_nurse_count 
  (total : ℕ) -- Total number of doctors and nurses
  (doc_ratio : ℕ) -- Ratio part for doctors
  (nurse_ratio : ℕ) -- Ratio part for nurses
  (h_total : total = 200) -- Total is 200
  (h_ratio : doc_ratio = 4 ∧ nurse_ratio = 6) -- Ratio is 4:6
  : (nurse_ratio : ℚ) / (doc_ratio + nurse_ratio) * total = 120 := by
  sorry

end hospital_nurse_count_l2452_245281


namespace divide_five_children_l2452_245260

/-- The number of ways to divide n distinguishable objects into two non-empty, 
    unordered groups, where rotations within groups and swapping of groups 
    don't create new arrangements -/
def divide_into_two_groups (n : ℕ) : ℕ :=
  sorry

/-- There are 5 children to be divided -/
def num_children : ℕ := 5

/-- The theorem stating that the number of ways to divide 5 children
    into two groups under the given conditions is 50 -/
theorem divide_five_children : 
  divide_into_two_groups num_children = 50 := by
  sorry

end divide_five_children_l2452_245260


namespace inf_a_plus_2b_is_3_l2452_245290

open Real

/-- Given 0 < a < b and |log a| = |log b|, the infimum of a + 2b is 3 -/
theorem inf_a_plus_2b_is_3 (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |log a| = |log b|) :
  ∃ (inf : ℝ), inf = 3 ∧ ∀ x, (∃ (a' b' : ℝ), 0 < a' ∧ a' < b' ∧ |log a'| = |log b'| ∧ x = a' + 2*b') → inf ≤ x :=
sorry

end inf_a_plus_2b_is_3_l2452_245290


namespace quadratic_real_roots_l2452_245277

theorem quadratic_real_roots (a : ℝ) :
  (∃ x : ℝ, (a - 2) * x^2 - 4 * x - 1 = 0) ↔ (a ≥ -2 ∧ a ≠ 2) :=
by sorry

end quadratic_real_roots_l2452_245277


namespace unique_digit_product_l2452_245202

theorem unique_digit_product (A M C : ℕ) : 
  A < 10 → M < 10 → C < 10 →
  (100 * A + 10 * M + C) * (A + M + C) = 2008 →
  A = 2 := by
sorry

end unique_digit_product_l2452_245202


namespace triangle_side_length_l2452_245274

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Conditions
  (a = Real.sqrt 3) →
  (Real.sin B = 1 / 2) →
  (C = π / 6) →
  -- Sum of angles in a triangle is π
  (A + B + C = π) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Conclusion
  b = 1 := by sorry

end triangle_side_length_l2452_245274


namespace cable_package_savings_l2452_245214

/-- Calculates the savings from choosing a bundle package over individual subscriptions --/
theorem cable_package_savings
  (basic_cost movie_cost bundle_cost : ℕ)
  (sports_cost_diff : ℕ)
  (h1 : basic_cost = 15)
  (h2 : movie_cost = 12)
  (h3 : sports_cost_diff = 3)
  (h4 : bundle_cost = 25) :
  basic_cost + movie_cost + (movie_cost - sports_cost_diff) - bundle_cost = 11 := by
  sorry


end cable_package_savings_l2452_245214


namespace root_sum_sixth_power_l2452_245293

theorem root_sum_sixth_power (r s : ℝ) : 
  r^2 - 2*r + Real.sqrt 2 = 0 → 
  s^2 - 2*s + Real.sqrt 2 = 0 → 
  r^6 + s^6 = 904 - 640 * Real.sqrt 2 := by
sorry

end root_sum_sixth_power_l2452_245293


namespace product_195_205_l2452_245207

theorem product_195_205 : 195 * 205 = 39975 := by
  sorry

end product_195_205_l2452_245207


namespace inequality_solution_range_l2452_245222

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x| + |x - 1| ≤ a) → a ∈ Set.Ici 1 := by
  sorry

end inequality_solution_range_l2452_245222


namespace mallory_journey_expenses_l2452_245272

/-- Calculates the total expenses for Mallory's journey --/
def journey_expenses (fuel_cost : ℚ) (tank_range : ℚ) (journey_distance : ℚ) 
  (hotel_nights : ℕ) (hotel_cost : ℚ) (fuel_increase : ℚ) 
  (maintenance_cost : ℚ) (activity_cost : ℚ) : ℚ :=
  let num_refills := (journey_distance / tank_range).ceil
  let total_fuel_cost := (num_refills * (num_refills - 1) / 2 * fuel_increase) + (num_refills * fuel_cost)
  let food_cost := (3 / 5) * total_fuel_cost
  let hotel_total := hotel_nights * hotel_cost
  let extra_expenses := maintenance_cost + activity_cost
  total_fuel_cost + food_cost + hotel_total + extra_expenses

/-- Theorem stating that Mallory's journey expenses equal $746 --/
theorem mallory_journey_expenses : 
  journey_expenses 45 500 2000 3 80 5 120 50 = 746 := by
  sorry

end mallory_journey_expenses_l2452_245272


namespace f_2012_equals_cos_l2452_245285

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => λ x => Real.cos x
| (n + 1) => λ x => deriv (f n) x

theorem f_2012_equals_cos : f 2012 = λ x => Real.cos x := by sorry

end f_2012_equals_cos_l2452_245285


namespace students_practicing_both_sports_l2452_245268

theorem students_practicing_both_sports :
  -- Define variables
  ∀ (F B x : ℕ),
  -- Condition 1: One-fifth of footballers play basketball
  F / 5 = x →
  -- Condition 2: One-seventh of basketball players play football
  B / 7 = x →
  -- Condition 3: 110 students practice exactly one sport
  (F - x) + (B - x) = 110 →
  -- Conclusion: x (students practicing both sports) = 11
  x = 11 := by
sorry

end students_practicing_both_sports_l2452_245268


namespace imaginary_part_of_one_minus_i_squared_l2452_245284

theorem imaginary_part_of_one_minus_i_squared (i : ℂ) : 
  Complex.im ((1 - i)^2) = -2 :=
by
  sorry

end imaginary_part_of_one_minus_i_squared_l2452_245284


namespace average_of_combined_sets_l2452_245216

theorem average_of_combined_sets (M N : ℕ) (X Y : ℝ) :
  let sum_M := M * X
  let sum_N := N * Y
  let total_sum := sum_M + sum_N
  let total_count := M + N
  (sum_M / M = X) → (sum_N / N = Y) → (total_sum / total_count = (M * X + N * Y) / (M + N)) :=
by sorry

end average_of_combined_sets_l2452_245216


namespace total_apples_calculation_total_apples_is_210_l2452_245270

/-- The number of apples bought by two men and three women -/
def total_apples : ℕ := by sorry

/-- The number of men -/
def num_men : ℕ := 2

/-- The number of women -/
def num_women : ℕ := 3

/-- The number of apples bought by each man -/
def apples_per_man : ℕ := 30

/-- The additional number of apples bought by each woman compared to each man -/
def additional_apples_per_woman : ℕ := 20

/-- The number of apples bought by each woman -/
def apples_per_woman : ℕ := apples_per_man + additional_apples_per_woman

theorem total_apples_calculation :
  total_apples = num_men * apples_per_man + num_women * apples_per_woman :=
by sorry

theorem total_apples_is_210 : total_apples = 210 := by sorry

end total_apples_calculation_total_apples_is_210_l2452_245270


namespace system_solution_l2452_245280

theorem system_solution (x y : ℝ) 
  (eq1 : 2 * x + 3 * y = 9) 
  (eq2 : 3 * x + 2 * y = 11) : 
  x - y = 2 := by
sorry

end system_solution_l2452_245280


namespace puzzle_solution_l2452_245242

def special_operation (a b c : Nat) : Nat :=
  (a * b) * 10000 + (a * c) * 100 + ((a + b + c) * 2)

theorem puzzle_solution :
  (special_operation 5 3 2 = 151022) →
  (special_operation 9 2 4 = 183652) →
  (special_operation 7 2 5 = 143556) := by
  sorry

end puzzle_solution_l2452_245242


namespace complex_maximum_value_l2452_245256

theorem complex_maximum_value (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₂ = 4)
  (h2 : 4 * z₁^2 - 2 * z₁ * z₂ + z₂^2 = 0) :
  ∃ (M : ℝ), M = 6 * Real.sqrt 6 ∧ 
    ∀ (w : ℂ), w = z₁ → Complex.abs ((w + 1)^2 * (w - 2)) ≤ M :=
by sorry

end complex_maximum_value_l2452_245256


namespace no_real_roots_implies_a_greater_than_one_l2452_245211

theorem no_real_roots_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a ≠ 0) → a > 1 := by
  sorry

end no_real_roots_implies_a_greater_than_one_l2452_245211


namespace train_average_speed_l2452_245253

theorem train_average_speed (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 80 →
  time = 8 →
  speed = distance / time →
  speed = 10 :=
by sorry

end train_average_speed_l2452_245253


namespace tangent_parallel_to_x_axis_g_minimum_value_f_inequality_l2452_245238

noncomputable section

variable (a : ℝ) (x : ℝ)

def f (x : ℝ) := a * x * Real.log x + (-a) * x

def g (x : ℝ) := x + 1 / Real.exp (x - 1)

theorem tangent_parallel_to_x_axis (h : a ≠ 0) :
  ∃ b : ℝ, (a * Real.log 1 + a + b = 0) → f a x = a * x * Real.log x + (-a) * x :=
sorry

theorem g_minimum_value :
  x > 0 → ∀ y > 0, g x ≥ 2 :=
sorry

theorem f_inequality (h : a ≠ 0) (hx : x > 0) :
  f a x / a + 2 / (x * Real.exp (x - 1) + 1) ≥ 1 - x :=
sorry

end tangent_parallel_to_x_axis_g_minimum_value_f_inequality_l2452_245238


namespace intersection_midpoint_distance_l2452_245224

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 - (Real.sqrt 3 / 2) * t, t / 2)

-- Define the curve C
def curve_C (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.sqrt 3 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point P
def point_P : ℝ × ℝ := (Real.sqrt 3, 0)

-- Theorem statement
theorem intersection_midpoint_distance : 
  ∃ (t₁ t₂ : ℝ), 
    let A := line_l t₁
    let B := line_l t₂
    let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)  -- Midpoint of AB
    curve_C (Real.arctan (A.2 / A.1)) = A ∧     -- A is on curve C
    curve_C (Real.arctan (B.2 / B.1)) = B ∧     -- B is on curve C
    Real.sqrt ((D.1 - point_P.1)^2 + (D.2 - point_P.2)^2) = (3 + Real.sqrt 3) / 2 :=
by
  sorry

end

end intersection_midpoint_distance_l2452_245224


namespace rational_expression_equals_240_l2452_245210

theorem rational_expression_equals_240 (x : ℝ) (h : x = 4) :
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end rational_expression_equals_240_l2452_245210


namespace sum_15_is_120_l2452_245237

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℚ
  /-- The common difference of the sequence -/
  d : ℚ
  /-- The sum of the first 5 terms is 10 -/
  sum_5 : (5 : ℚ) / 2 * (2 * a₁ + 4 * d) = 10
  /-- The sum of the first 10 terms is 50 -/
  sum_10 : (10 : ℚ) / 2 * (2 * a₁ + 9 * d) = 50

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a₁ + (n - 1 : ℚ) * seq.d)

/-- Theorem: The sum of the first 15 terms is 120 -/
theorem sum_15_is_120 (seq : ArithmeticSequence) : sum_n seq 15 = 120 := by
  sorry

end sum_15_is_120_l2452_245237


namespace light_flash_interval_l2452_245240

/-- Given a light that flashes 600 times in 1/6 of an hour, prove that the time between each flash is 1 second. -/
theorem light_flash_interval (flashes_per_sixth_hour : ℕ) (h : flashes_per_sixth_hour = 600) :
  (1 / 6 : ℚ) * 3600 / flashes_per_sixth_hour = 1 := by
  sorry

#check light_flash_interval

end light_flash_interval_l2452_245240


namespace max_salary_in_soccer_league_l2452_245227

/-- Represents a soccer team with salary constraints -/
structure SoccerTeam where
  numPlayers : ℕ
  minSalary : ℕ
  totalSalaryCap : ℕ

/-- Calculates the maximum possible salary for a single player in the team -/
def maxSinglePlayerSalary (team : SoccerTeam) : ℕ :=
  team.totalSalaryCap - (team.numPlayers - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player
    in a semi-professional soccer league with given constraints -/
theorem max_salary_in_soccer_league :
  let team : SoccerTeam := {
    numPlayers := 25,
    minSalary := 15000,
    totalSalaryCap := 850000
  }
  maxSinglePlayerSalary team = 490000 := by
  sorry

#eval maxSinglePlayerSalary {
  numPlayers := 25,
  minSalary := 15000,
  totalSalaryCap := 850000
}

end max_salary_in_soccer_league_l2452_245227


namespace smallest_x_value_l2452_245223

theorem smallest_x_value (x : ℝ) : 
  (4 * x / 10 + 1 / (4 * x) = 5 / 8) → 
  x ≥ (25 - Real.sqrt 1265) / 32 ∧ 
  ∃ y : ℝ, y = (25 - Real.sqrt 1265) / 32 ∧ 4 * y / 10 + 1 / (4 * y) = 5 / 8 := by
  sorry

end smallest_x_value_l2452_245223


namespace watermelon_cost_l2452_245205

/-- The problem of determining the cost of a watermelon --/
theorem watermelon_cost (total_fruits : ℕ) (total_value : ℕ) 
  (melon_capacity : ℕ) (watermelon_capacity : ℕ) :
  total_fruits = 150 →
  total_value = 24000 →
  melon_capacity = 120 →
  watermelon_capacity = 160 →
  ∃ (num_watermelons num_melons : ℕ) (watermelon_cost melon_cost : ℚ),
    num_watermelons + num_melons = total_fruits ∧
    num_watermelons * watermelon_cost = num_melons * melon_cost ∧
    num_watermelons * watermelon_cost + num_melons * melon_cost = total_value ∧
    (num_watermelons : ℚ) / watermelon_capacity + (num_melons : ℚ) / melon_capacity = 1 ∧
    watermelon_cost = 100 := by
  sorry

end watermelon_cost_l2452_245205


namespace arithmetic_sequence_property_l2452_245250

/-- An arithmetic sequence with non-zero terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a n ≠ 0) ∧
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_condition : 2 * a 3 - (a 1)^2 + 2 * a 11 = 0) :
  a 7 = 4 :=
sorry

end arithmetic_sequence_property_l2452_245250


namespace combinations_with_repetition_l2452_245297

/-- F_n^r represents the number of r-combinatorial selections from [1, n] with repetition allowed -/
def F (n : ℕ) (r : ℕ) : ℕ := sorry

/-- C_n^r represents the binomial coefficient (n choose r) -/
def C (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The theorem states that F_n^r equals C_(n+r-1)^r -/
theorem combinations_with_repetition (n : ℕ) (r : ℕ) : F n r = C (n + r - 1) r := by
  sorry

end combinations_with_repetition_l2452_245297


namespace sector_area_120_deg_sqrt3_radius_l2452_245232

/-- The area of a circular sector with central angle 120° and radius √3 is equal to π. -/
theorem sector_area_120_deg_sqrt3_radius (π : ℝ) : 
  let angle : ℝ := 2 * π / 3  -- 120° in radians
  let radius : ℝ := Real.sqrt 3
  let sector_area : ℝ := (1 / 2) * angle * radius^2
  sector_area = π :=
by sorry

end sector_area_120_deg_sqrt3_radius_l2452_245232


namespace equation_solution_l2452_245257

theorem equation_solution : ∃ x : ℚ, (1/7 : ℚ) + 7/x = 15/x + (1/15 : ℚ) ∧ x = 105 := by
  sorry

end equation_solution_l2452_245257


namespace perfect_square_octal_rep_c_is_one_l2452_245201

/-- Octal representation of a number -/
structure OctalRep where
  a : ℕ
  b : ℕ
  c : ℕ
  h_a_nonzero : a ≠ 0

/-- Perfect square with specific octal representation -/
def is_perfect_square_with_octal_rep (n : ℕ) (rep : OctalRep) : Prop :=
  ∃ k : ℕ, n = k^2 ∧ n = 8^3 * rep.a + 8^2 * rep.b + 8 * 3 + rep.c

theorem perfect_square_octal_rep_c_is_one (n : ℕ) (rep : OctalRep) :
  is_perfect_square_with_octal_rep n rep → rep.c = 1 := by
  sorry

end perfect_square_octal_rep_c_is_one_l2452_245201


namespace squares_remaining_l2452_245246

theorem squares_remaining (total : ℕ) (removed_fraction : ℚ) (result : ℕ) : 
  total = 12 →
  removed_fraction = 1/2 * 2/3 →
  result = total - (removed_fraction * total).num →
  result = 8 := by
  sorry

end squares_remaining_l2452_245246


namespace angle_value_for_point_l2452_245217

theorem angle_value_for_point (θ : Real) (P : Real × Real) :
  P.1 = Real.sin (3 * Real.pi / 4) →
  P.2 = Real.cos (3 * Real.pi / 4) →
  0 ≤ θ →
  θ < 2 * Real.pi →
  (Real.cos θ, Real.sin θ) = (P.1 / Real.sqrt (P.1^2 + P.2^2), P.2 / Real.sqrt (P.1^2 + P.2^2)) →
  θ = 7 * Real.pi / 4 := by
sorry

end angle_value_for_point_l2452_245217


namespace max_cos_diff_l2452_245266

theorem max_cos_diff (x y : Real) (h : Real.sin x - Real.sin y = 3/4) :
  ∃ (max_val : Real), max_val = 23/32 ∧ 
    ∀ (z w : Real), Real.sin z - Real.sin w = 3/4 → Real.cos (z - w) ≤ max_val :=
by sorry

end max_cos_diff_l2452_245266


namespace nine_crosses_fit_chessboard_l2452_245215

/-- Represents a cross pentomino -/
structure CrossPentomino where
  area : ℕ
  size : ℕ × ℕ

/-- Represents a chessboard -/
structure Chessboard where
  size : ℕ × ℕ
  area : ℕ

/-- Theorem: Nine cross pentominoes can fit within an 8x8 chessboard -/
theorem nine_crosses_fit_chessboard (cross : CrossPentomino) (board : Chessboard) : 
  cross.area = 5 ∧ 
  cross.size = (1, 1) ∧ 
  board.size = (8, 8) ∧ 
  board.area = 64 →
  9 * cross.area ≤ board.area :=
by sorry

end nine_crosses_fit_chessboard_l2452_245215


namespace triangle_existence_implies_m_greater_than_six_l2452_245295

/-- The function f(x) = x^3 - 3x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

/-- Theorem: If there exists a triangle with side lengths f(a), f(b), f(c) for a, b, c in [0, 2], then m > 6 -/
theorem triangle_existence_implies_m_greater_than_six (m : ℝ) : 
  (∃ a b c : ℝ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 
    f m a + f m b > f m c ∧ 
    f m b + f m c > f m a ∧ 
    f m c + f m a > f m b) → 
  m > 6 := by
  sorry


end triangle_existence_implies_m_greater_than_six_l2452_245295


namespace incenter_is_angle_bisectors_intersection_l2452_245235

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle --/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- An angle bisector of a triangle --/
def angle_bisector (t : Triangle) (vertex : Fin 3) : Set (ℝ × ℝ) := sorry

/-- The intersection point of the angle bisectors --/
def angle_bisectors_intersection (t : Triangle) : ℝ × ℝ := sorry

/-- Theorem: The incenter of a triangle is the intersection point of its angle bisectors --/
theorem incenter_is_angle_bisectors_intersection (t : Triangle) :
  incenter t = angle_bisectors_intersection t := by sorry

end incenter_is_angle_bisectors_intersection_l2452_245235


namespace cubic_inequality_l2452_245283

theorem cubic_inequality (a b c : ℝ) 
  (h : ∃ x₁ x₂ x₃ : ℝ, 
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧
    x₁^3 + a*x₁^2 + b*x₁ + c = 0 ∧
    x₂^3 + a*x₂^2 + b*x₂ + c = 0 ∧
    x₃^3 + a*x₃^2 + b*x₃ + c = 0 ∧
    x₁ + x₂ + x₃ ≤ 1) :
  a^3*(1 + a + b) - 9*c*(3 + 3*a + a^2) ≤ 0 := by
sorry

end cubic_inequality_l2452_245283


namespace base_conversion_addition_equality_l2452_245254

-- Define a function to convert a number from base b to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def n1 : List Nat := [2, 5, 3]
def b1 : Nat := 8
def d1 : List Nat := [1, 3]
def b2 : Nat := 3
def n2 : List Nat := [2, 4, 5]
def b3 : Nat := 7
def d2 : List Nat := [3, 5]
def b4 : Nat := 6

-- State the theorem
theorem base_conversion_addition_equality :
  (to_base_10 n1 b1 : ℚ) / (to_base_10 d1 b2 : ℚ) + 
  (to_base_10 n2 b3 : ℚ) / (to_base_10 d2 b4 : ℚ) = 
  171 / 6 + 131 / 23 := by sorry

end base_conversion_addition_equality_l2452_245254


namespace project_completion_equivalence_l2452_245219

/-- Represents the time taken to complete a project given the number of workers -/
def project_completion_time (num_workers : ℕ) (days : ℚ) : Prop :=
  num_workers * days = 120 * 7

theorem project_completion_equivalence :
  project_completion_time 120 7 → project_completion_time 80 (21/2) := by
  sorry

end project_completion_equivalence_l2452_245219


namespace fraction_comparison_l2452_245296

theorem fraction_comparison : 
  (9 : ℚ) / 21 = (3 : ℚ) / 7 ∧ 
  (12 : ℚ) / 28 = (3 : ℚ) / 7 ∧ 
  (30 : ℚ) / 70 = (3 : ℚ) / 7 ∧ 
  (13 : ℚ) / 28 ≠ (3 : ℚ) / 7 := by
sorry

end fraction_comparison_l2452_245296


namespace line_perpendicular_to_parallel_planes_l2452_245264

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_planes
  (m : Line) (α β : Plane)
  (h1 : parallel α β)
  (h2 : perpendicular m α) :
  perpendicular m β :=
sorry

end line_perpendicular_to_parallel_planes_l2452_245264


namespace quadratic_equation_roots_l2452_245229

theorem quadratic_equation_roots (x y : ℝ) : 
  x + y = 10 ∧ |x - y| = 12 → x^2 - 10*x - 11 = 0 ∨ y^2 - 10*y - 11 = 0 :=
by sorry

end quadratic_equation_roots_l2452_245229


namespace solve_refrigerator_problem_l2452_245255

def refrigerator_problem (part_payment : ℝ) (percentage : ℝ) : Prop :=
  let total_cost := part_payment / (percentage / 100)
  let remaining_amount := total_cost - part_payment
  (part_payment = 875) ∧ 
  (percentage = 25) ∧ 
  (remaining_amount = 2625)

theorem solve_refrigerator_problem :
  ∃ (part_payment percentage : ℝ), refrigerator_problem part_payment percentage :=
sorry

end solve_refrigerator_problem_l2452_245255


namespace photo_arrangements_l2452_245291

def teacher : ℕ := 1
def boys : ℕ := 4
def girls : ℕ := 2
def total_people : ℕ := teacher + boys + girls

theorem photo_arrangements :
  (∃ (arrangements_girls_together : ℕ), arrangements_girls_together = 1440) ∧
  (∃ (arrangements_boys_apart : ℕ), arrangements_boys_apart = 144) :=
by sorry

end photo_arrangements_l2452_245291


namespace parallel_sufficient_not_necessary_l2452_245248

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Predicate to check if a line is parallel to a plane -/
def is_parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line is outside of a plane -/
def is_outside (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem parallel_sufficient_not_necessary
  (l : Line3D) (α : Plane3D) :
  (is_parallel l α → is_outside l α) ∧
  ∃ l', is_outside l' α ∧ ¬is_parallel l' α :=
sorry

end parallel_sufficient_not_necessary_l2452_245248


namespace units_digit_of_product_l2452_245278

theorem units_digit_of_product (n : ℕ) : n % 10 = (2^101 * 7^1002 * 3^1004) % 10 → n = 8 := by
  sorry

end units_digit_of_product_l2452_245278


namespace m_function_inequality_l2452_245276

/-- An M-function is a function f: ℝ → ℝ defined on (0, +∞) that satisfies xf''(x) > f(x) for all x in (0, +∞) -/
def is_M_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → x * (deriv^[2] f x) > f x

/-- Theorem: For any M-function f and positive real numbers x₁ and x₂, 
    the sum f(x₁) + f(x₂) is less than f(x₁ + x₂) -/
theorem m_function_inequality (f : ℝ → ℝ) (hf : is_M_function f) 
  (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) :
  f x₁ + f x₂ < f (x₁ + x₂) :=
sorry

end m_function_inequality_l2452_245276


namespace perpendicular_transitivity_perpendicular_parallel_l2452_245279

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations for parallel and perpendicular
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (planeparallel : Plane → Plane → Prop)
variable (planeperpendicular : Plane → Plane → Prop)
variable (lineplaneparallel : Line → Plane → Prop)
variable (lineplaneperpendicular : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ " => parallel
local infix:50 " ⊥ " => perpendicular
local infix:50 " ∥ₚ " => planeparallel
local infix:50 " ⊥ₚ " => planeperpendicular
local infix:50 " ∥ₗₚ " => lineplaneparallel
local infix:50 " ⊥ₗₚ " => lineplaneperpendicular

-- Theorem statements
theorem perpendicular_transitivity 
  (m n : Line) (α β : Plane) :
  m ⊥ₗₚ α → n ⊥ₗₚ β → α ⊥ₚ β → m ⊥ n :=
sorry

theorem perpendicular_parallel 
  (m n : Line) (α β : Plane) :
  m ⊥ₗₚ α → n ∥ₗₚ β → α ∥ₚ β → m ⊥ n :=
sorry

end perpendicular_transitivity_perpendicular_parallel_l2452_245279


namespace cubic_roots_sum_l2452_245204

theorem cubic_roots_sum (a b : ℝ) : 
  (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ t : ℝ, t^3 - 9*t^2 + a*t - b = 0 ↔ t = x ∨ t = y ∨ t = z)) →
  a + b = 38 := by
sorry

end cubic_roots_sum_l2452_245204


namespace ruths_school_schedule_l2452_245289

/-- Ruth's school schedule problem -/
theorem ruths_school_schedule 
  (days_per_week : ℕ) 
  (math_class_percentage : ℚ) 
  (math_class_hours_per_week : ℕ) 
  (h1 : days_per_week = 5)
  (h2 : math_class_percentage = 1/4)
  (h3 : math_class_hours_per_week = 10) :
  let total_school_hours_per_week := math_class_hours_per_week / math_class_percentage
  let school_hours_per_day := total_school_hours_per_week / days_per_week
  school_hours_per_day = 8 := by
  sorry

end ruths_school_schedule_l2452_245289


namespace fred_money_left_l2452_245220

def fred_book_problem (initial_amount : ℕ) (num_books : ℕ) (avg_cost : ℕ) : ℕ :=
  initial_amount - (num_books * avg_cost)

theorem fred_money_left :
  fred_book_problem 236 6 37 = 14 := by
  sorry

end fred_money_left_l2452_245220


namespace mean_of_added_numbers_l2452_245261

theorem mean_of_added_numbers (original_numbers : List ℝ) (x y z : ℝ) :
  original_numbers.length = 12 →
  original_numbers.sum / original_numbers.length = 72 →
  (original_numbers.sum + x + y + z) / (original_numbers.length + 3) = 80 →
  (x + y + z) / 3 = 112 := by
sorry

end mean_of_added_numbers_l2452_245261


namespace cost_of_second_box_l2452_245213

/-- The cost of cards in the first box -/
def cost_box1 : ℚ := 1.25

/-- The number of cards bought from each box -/
def cards_bought : ℕ := 6

/-- The total amount spent -/
def total_spent : ℚ := 18

/-- The cost of cards in the second box -/
def cost_box2 : ℚ := (total_spent - cards_bought * cost_box1) / cards_bought

theorem cost_of_second_box : cost_box2 = 1.75 := by
  sorry

end cost_of_second_box_l2452_245213


namespace sports_camp_coach_age_l2452_245234

theorem sports_camp_coach_age (total_members : ℕ) (total_average_age : ℕ)
  (num_girls num_boys num_coaches : ℕ) (girls_average_age boys_average_age : ℕ)
  (h1 : total_members = 50)
  (h2 : total_average_age = 20)
  (h3 : num_girls = 30)
  (h4 : num_boys = 15)
  (h5 : num_coaches = 5)
  (h6 : girls_average_age = 18)
  (h7 : boys_average_age = 19)
  (h8 : total_members = num_girls + num_boys + num_coaches) :
  (total_members * total_average_age - num_girls * girls_average_age - num_boys * boys_average_age) / num_coaches = 35 := by
sorry


end sports_camp_coach_age_l2452_245234


namespace percentage_problem_l2452_245206

theorem percentage_problem (x : ℝ) : 
  (40 * x / 100) + (25 / 100 * 60) = 23 → x = 20 := by
  sorry

end percentage_problem_l2452_245206


namespace cylinder_height_relation_l2452_245226

theorem cylinder_height_relation :
  ∀ (r₁ h₁ r₂ h₂ : ℝ),
  r₁ > 0 → h₁ > 0 → r₂ > 0 → h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end cylinder_height_relation_l2452_245226


namespace projection_matrix_l2452_245273

def P : Matrix (Fin 2) (Fin 2) ℚ := !![965/1008, 18/41; 19/34, 23/41]

theorem projection_matrix : P * P = P := by sorry

end projection_matrix_l2452_245273


namespace angle_measure_l2452_245233

theorem angle_measure (x : ℝ) : 
  (180 - x = 6 * (90 - x)) → x = 72 := by
  sorry

end angle_measure_l2452_245233


namespace highest_possible_average_after_removing_lowest_score_l2452_245212

def number_of_tests : ℕ := 9
def original_average : ℚ := 68
def lowest_possible_score : ℚ := 0

theorem highest_possible_average_after_removing_lowest_score :
  let total_score : ℚ := number_of_tests * original_average
  let remaining_score : ℚ := total_score - lowest_possible_score
  let new_average : ℚ := remaining_score / (number_of_tests - 1)
  new_average = 76.5 := by
  sorry

end highest_possible_average_after_removing_lowest_score_l2452_245212


namespace locus_of_tangent_points_theorem_l2452_245294

/-- The locus of points for which an ellipse or hyperbola with center at the origin is tangent -/
def locus_of_tangent_points (x y a b c : ℝ) : Prop :=
  (a^2 * y^2 + b^2 * x^2 = x^2 * y^2 ∧ b^2 = a^2 - c^2) ∨
  (a^2 * y^2 - b^2 * x^2 = x^2 * y^2 ∧ b^2 = c^2 - a^2)

/-- Theorem stating the locus of points for ellipses and hyperbolas with center at the origin -/
theorem locus_of_tangent_points_theorem (x y a b c : ℝ) :
  locus_of_tangent_points x y a b c :=
sorry

end locus_of_tangent_points_theorem_l2452_245294


namespace isosceles_triangle_circumscribed_circle_l2452_245200

/-- Given a circle with radius 3 and an isosceles triangle circumscribed around it with a base angle of 30°, 
    this theorem proves the lengths of the sides of the triangle. -/
theorem isosceles_triangle_circumscribed_circle 
  (r : ℝ) 
  (base_angle : ℝ) 
  (h_r : r = 3) 
  (h_angle : base_angle = 30 * π / 180) : 
  ∃ (equal_side base_side : ℝ),
    equal_side = 4 * Real.sqrt 3 + 6 ∧ 
    base_side = 6 * Real.sqrt 3 + 12 := by
  sorry

end isosceles_triangle_circumscribed_circle_l2452_245200


namespace inverse_of_A_cubed_l2452_245265

/-- Given a 2x2 matrix A with inverse [[3, -1], [1, 1]], 
    prove that the inverse of A^3 is [[20, -12], [12, -4]] -/
theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = ![![3, -1], ![1, 1]]) : 
  (A^3)⁻¹ = ![![20, -12], ![12, -4]] := by
  sorry

end inverse_of_A_cubed_l2452_245265


namespace simplify_fraction_l2452_245292

theorem simplify_fraction : (75 : ℚ) / 100 = 3 / 4 := by
  sorry

end simplify_fraction_l2452_245292


namespace directrix_of_parabola_l2452_245298

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- State the theorem
theorem directrix_of_parabola :
  ∀ x y : ℝ, parabola x y → (∃ p : ℝ, x = -3 ∧ p = y) :=
by sorry

end directrix_of_parabola_l2452_245298


namespace external_bisector_angles_theorem_l2452_245245

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angles of a triangle
def angles (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Define external angle bisectors
def externalAngleBisectors (t : Triangle) : Triangle := sorry

-- Theorem statement
theorem external_bisector_angles_theorem (t : Triangle) :
  let t' := externalAngleBisectors t
  angles t' = (40, 65, 75) → angles t = (100, 30, 50) := by
  sorry

end external_bisector_angles_theorem_l2452_245245


namespace orbius_5_stay_duration_l2452_245271

/-- Calculates the number of days an astronaut stays on a planet given the total days per year, 
    number of seasons per year, and number of seasons stayed. -/
def days_stayed (total_days_per_year : ℕ) (seasons_per_year : ℕ) (seasons_stayed : ℕ) : ℕ :=
  (total_days_per_year / seasons_per_year) * seasons_stayed

/-- Theorem: An astronaut staying on Orbius-5 for 3 seasons will spend 150 days on the planet. -/
theorem orbius_5_stay_duration : 
  days_stayed 250 5 3 = 150 := by
  sorry

end orbius_5_stay_duration_l2452_245271


namespace problem_solution_l2452_245259

theorem problem_solution : ∃ x : ℝ, 3 * x + 3 * 14 + 3 * 15 + 11 = 152 ∧ x = 18 := by
  sorry

end problem_solution_l2452_245259


namespace square_root_sum_equals_abs_sum_l2452_245221

theorem square_root_sum_equals_abs_sum (x : ℝ) : 
  Real.sqrt ((x - 3)^2) + Real.sqrt ((x + 5)^2) = |x - 3| + |x + 5| :=
by sorry

end square_root_sum_equals_abs_sum_l2452_245221


namespace expression_evaluation_l2452_245203

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = 3) : 
  (3*x^2 + y)^2 - (3*x^2 - y)^2 = 144 := by sorry

end expression_evaluation_l2452_245203


namespace oliver_used_30_tickets_l2452_245241

/-- The number of times Oliver rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The number of times Oliver rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride -/
def ticket_cost : ℕ := 3

/-- The total number of tickets Oliver used -/
def total_tickets : ℕ := (ferris_rides + bumper_rides) * ticket_cost

/-- Theorem stating that Oliver used 30 tickets -/
theorem oliver_used_30_tickets : total_tickets = 30 := by
  sorry

end oliver_used_30_tickets_l2452_245241


namespace triangle_side_length_l2452_245263

/-- Given a triangle ABC where ∠C = 2∠A, a = 34, and c = 60, prove that b = 4352/450 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (h1 : C = 2 * A) (h2 : a = 34) (h3 : c = 60) : 
  b = 4352 / 450 := by
  sorry

end triangle_side_length_l2452_245263


namespace sufficient_condition_sum_greater_than_double_l2452_245230

theorem sufficient_condition_sum_greater_than_double (a b c : ℝ) :
  a > c ∧ b > c → a + b > 2 * c := by sorry

end sufficient_condition_sum_greater_than_double_l2452_245230


namespace drug_storage_temperature_range_l2452_245251

/-- Given a drug with a storage temperature of 20 ± 2 (°C), 
    the difference between the highest and lowest suitable storage temperatures is 4°C -/
theorem drug_storage_temperature_range : 
  let recommended_temp : ℝ := 20
  let tolerance : ℝ := 2
  let highest_temp := recommended_temp + tolerance
  let lowest_temp := recommended_temp - tolerance
  highest_temp - lowest_temp = 4 := by
  sorry

end drug_storage_temperature_range_l2452_245251


namespace division_result_l2452_245286

theorem division_result : (64 : ℝ) / 0.08 = 800 := by
  sorry

end division_result_l2452_245286


namespace annulus_area_l2452_245249

/-- The area of an annulus formed by two concentric circles. -/
theorem annulus_area (b c a : ℝ) (h1 : b > c) (h2 : b^2 = c^2 + a^2) :
  (π * b^2 - π * c^2) = π * a^2 := by sorry

end annulus_area_l2452_245249


namespace plane_perp_theorem_l2452_245225

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a plane and a line
variable (perp_plane_line : Plane → Line → Prop)

-- Define the intersection operation between planes
variable (intersect : Plane → Plane → Line)

-- State the theorem
theorem plane_perp_theorem 
  (α β : Plane) (l : Line) 
  (h1 : perp_planes α β) 
  (h2 : intersect α β = l) :
  ∀ γ : Plane, perp_plane_line γ l → 
    perp_planes γ α ∧ perp_planes γ β :=
sorry

end plane_perp_theorem_l2452_245225


namespace all_sheep_can_be_blue_not_all_sheep_can_be_red_or_green_l2452_245282

/-- Represents the count of sheep of each color -/
structure SheepCounts where
  blue : Nat
  red : Nat
  green : Nat

/-- Represents a transformation of sheep colors -/
inductive SheepTransform
  | BlueRedToGreen
  | BlueGreenToRed
  | RedGreenToBlue

/-- Applies a transformation to the sheep counts -/
def applyTransform (counts : SheepCounts) (transform : SheepTransform) : SheepCounts :=
  match transform with
  | SheepTransform.BlueRedToGreen => 
      ⟨counts.blue - 1, counts.red - 1, counts.green + 2⟩
  | SheepTransform.BlueGreenToRed => 
      ⟨counts.blue - 1, counts.red + 2, counts.green - 1⟩
  | SheepTransform.RedGreenToBlue => 
      ⟨counts.blue + 2, counts.red - 1, counts.green - 1⟩

/-- The initial counts of sheep -/
def initialCounts : SheepCounts := ⟨22, 18, 15⟩

/-- Theorem stating that it's possible for all sheep to become blue -/
theorem all_sheep_can_be_blue :
  ∃ (transforms : List SheepTransform), 
    let finalCounts := transforms.foldl applyTransform initialCounts
    finalCounts.red = 0 ∧ finalCounts.green = 0 ∧ finalCounts.blue > 0 :=
sorry

/-- Theorem stating that it's impossible for all sheep to become red or green -/
theorem not_all_sheep_can_be_red_or_green :
  ¬∃ (transforms : List SheepTransform), 
    let finalCounts := transforms.foldl applyTransform initialCounts
    (finalCounts.blue = 0 ∧ finalCounts.green = 0 ∧ finalCounts.red > 0) ∨
    (finalCounts.blue = 0 ∧ finalCounts.red = 0 ∧ finalCounts.green > 0) :=
sorry

end all_sheep_can_be_blue_not_all_sheep_can_be_red_or_green_l2452_245282


namespace geometric_sequence_seventh_term_l2452_245247

/-- Given a geometric sequence with first term a₁ = 3 and second term a₂ = -1/2,
    prove that the 7th term a₇ = 1/15552 -/
theorem geometric_sequence_seventh_term :
  let a₁ : ℚ := 3
  let a₂ : ℚ := -1/2
  let r : ℚ := a₂ / a₁
  let a₇ : ℚ := a₁ * r^6
  a₇ = 1/15552 := by sorry

end geometric_sequence_seventh_term_l2452_245247


namespace workshop_average_salary_l2452_245209

/-- Proves that the average salary of all workers in a workshop is 8000,
    given the specified conditions. -/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_salary : ℕ)
  (other_salary : ℕ)
  (h1 : total_workers = 49)
  (h2 : technicians = 7)
  (h3 : technician_salary = 20000)
  (h4 : other_salary = 6000) :
  (technicians * technician_salary + (total_workers - technicians) * other_salary) / total_workers = 8000 :=
by sorry

end workshop_average_salary_l2452_245209


namespace num_constructible_heights_l2452_245269

/-- The number of bricks available --/
def num_bricks : ℕ := 25

/-- The possible height contributions of each brick after normalization and simplification --/
def height_options : List ℕ := [0, 3, 4]

/-- A function that returns the set of all possible tower heights --/
noncomputable def constructible_heights : Finset ℕ :=
  sorry

/-- The theorem stating that the number of constructible heights is 98 --/
theorem num_constructible_heights :
  Finset.card constructible_heights = 98 :=
sorry

end num_constructible_heights_l2452_245269


namespace ships_required_equals_round_trip_duration_moscow_astrakhan_ships_required_l2452_245236

/-- Represents the duration of travel and stay in days -/
structure TravelDuration :=
  (moscow_to_astrakhan : ℕ)
  (stay_in_astrakhan : ℕ)
  (astrakhan_to_moscow : ℕ)
  (stay_in_moscow : ℕ)

/-- Calculates the total round trip duration -/
def round_trip_duration (t : TravelDuration) : ℕ :=
  t.moscow_to_astrakhan + t.stay_in_astrakhan + t.astrakhan_to_moscow + t.stay_in_moscow

/-- The number of ships required for continuous daily departures -/
def ships_required (t : TravelDuration) : ℕ :=
  round_trip_duration t

/-- Theorem stating that the number of ships required is equal to the round trip duration -/
theorem ships_required_equals_round_trip_duration (t : TravelDuration) :
  ships_required t = round_trip_duration t := by
  sorry

/-- The specific travel durations given in the problem -/
def moscow_astrakhan_route : TravelDuration :=
  { moscow_to_astrakhan := 4
  , stay_in_astrakhan := 2
  , astrakhan_to_moscow := 5
  , stay_in_moscow := 2 }

/-- Theorem proving that 13 ships are required for the Moscow-Astrakhan route -/
theorem moscow_astrakhan_ships_required :
  ships_required moscow_astrakhan_route = 13 := by
  sorry

end ships_required_equals_round_trip_duration_moscow_astrakhan_ships_required_l2452_245236


namespace tiffany_lives_l2452_245267

theorem tiffany_lives (initial_lives lost_lives gained_lives final_lives : ℕ) : 
  lost_lives = 14 →
  gained_lives = 27 →
  final_lives = 56 →
  final_lives = initial_lives - lost_lives + gained_lives →
  initial_lives = 43 :=
by sorry

end tiffany_lives_l2452_245267


namespace bus_driver_regular_rate_l2452_245228

/-- Represents the compensation structure and work details of a bus driver --/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeMultiplier : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Calculates the total compensation based on the given compensation structure --/
def calculateTotalCompensation (c : BusDriverCompensation) : ℝ :=
  c.regularRate * c.regularHours + c.regularRate * c.overtimeMultiplier * c.overtimeHours

/-- Theorem stating that the regular rate of $16 per hour satisfies the given conditions --/
theorem bus_driver_regular_rate :
  ∃ (c : BusDriverCompensation),
    c.regularRate = 16 ∧
    c.overtimeMultiplier = 1.75 ∧
    c.regularHours = 40 ∧
    c.overtimeHours = 12 ∧
    c.totalCompensation = 976 ∧
    calculateTotalCompensation c = c.totalCompensation :=
  sorry

end bus_driver_regular_rate_l2452_245228


namespace logarithm_calculation_l2452_245218

theorem logarithm_calculation : (Real.log 128 / Real.log 2) / (Real.log 64 / Real.log 2) - (Real.log 256 / Real.log 2) / (Real.log 16 / Real.log 2) = 10 := by
  sorry

end logarithm_calculation_l2452_245218


namespace triangle_larger_segment_l2452_245288

theorem triangle_larger_segment (a b c h x : ℝ) : 
  a = 35 → b = 65 → c = 85 → 
  a^2 = x^2 + h^2 → 
  b^2 = (c - x)^2 + h^2 → 
  c - x = 60 :=
by sorry

end triangle_larger_segment_l2452_245288


namespace arithmetic_equality_l2452_245287

theorem arithmetic_equality : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_equality_l2452_245287


namespace sqrt_equation_solution_l2452_245275

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (9 + 3 * x) = 15 :=
by
  -- The proof would go here
  sorry

end sqrt_equation_solution_l2452_245275


namespace natural_pairs_with_sum_and_gcd_l2452_245258

theorem natural_pairs_with_sum_and_gcd (a b : ℕ) : 
  a + b = 288 → Nat.gcd a b = 36 → 
  ((a = 36 ∧ b = 252) ∨ (a = 252 ∧ b = 36) ∨ (a = 108 ∧ b = 180) ∨ (a = 180 ∧ b = 108)) :=
by sorry

end natural_pairs_with_sum_and_gcd_l2452_245258


namespace age_difference_l2452_245208

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 11) : a = c + 11 := by
  sorry

end age_difference_l2452_245208


namespace wife_weekly_contribution_l2452_245244

def husband_weekly_contribution : ℕ := 335
def savings_weeks : ℕ := 24
def children_count : ℕ := 4
def child_receives : ℕ := 1680

theorem wife_weekly_contribution (wife_contribution : ℕ) :
  (husband_weekly_contribution * savings_weeks + wife_contribution * savings_weeks) / 2 =
  children_count * child_receives →
  wife_contribution = 225 := by
  sorry

end wife_weekly_contribution_l2452_245244
