import Mathlib

namespace proportion_not_recent_boarders_l3318_331857

/-- Represents the proportion of passengers who boarded at a given dock -/
def boardingProportion : ℚ := 1/4

/-- Represents the proportion of departing passengers who boarded at the previous dock -/
def previousDockProportion : ℚ := 1/10

/-- Calculates the proportion of passengers who boarded at either of the two previous docks -/
def recentBoardersProportion : ℚ := 2 * boardingProportion - boardingProportion * previousDockProportion

/-- Theorem stating the proportion of passengers who did not board at either of the two previous docks -/
theorem proportion_not_recent_boarders :
  1 - recentBoardersProportion = 21/40 := by sorry

end proportion_not_recent_boarders_l3318_331857


namespace final_value_after_four_iterations_l3318_331867

def iterate_operation (x : ℕ) (s : ℕ) : ℕ := s * x + 1

def final_value (x : ℕ) (initial_s : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => initial_s
  | n + 1 => iterate_operation x (final_value x initial_s n)

theorem final_value_after_four_iterations :
  final_value 2 0 4 = 15 := by sorry

end final_value_after_four_iterations_l3318_331867


namespace acid_solution_volume_l3318_331821

theorem acid_solution_volume (V : ℝ) : 
  (V > 0) →                              -- Initial volume is positive
  (0.2 * V - 4 + 20 = 0.4 * V) →         -- Equation representing the acid balance
  (V = 80) :=                            -- Conclusion: initial volume is 80 ml
by
  sorry

end acid_solution_volume_l3318_331821


namespace quadratic_form_h_value_l3318_331876

theorem quadratic_form_h_value (a k h : ℝ) :
  (∀ x, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) →
  h = -3/2 := by
sorry

end quadratic_form_h_value_l3318_331876


namespace parabola_properties_l3318_331829

/-- A parabola with vertex at the origin, focus on the x-axis, and passing through (2, 2) -/
def parabola (x y : ℝ) : Prop := y^2 = 2*x

theorem parabola_properties :
  (parabola 0 0) ∧ 
  (∃ p : ℝ, p > 0 ∧ parabola p 0) ∧ 
  (parabola 2 2) := by
  sorry

end parabola_properties_l3318_331829


namespace quadratic_equation_range_l3318_331894

theorem quadratic_equation_range :
  {a : ℝ | ∃ x : ℝ, x^2 - 4*x + a = 0} = Set.Iic 4 := by sorry

end quadratic_equation_range_l3318_331894


namespace flag_design_count_l3318_331825

/-- The number of possible colors for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

theorem flag_design_count :
  num_flag_designs = 27 :=
sorry

end flag_design_count_l3318_331825


namespace max_fraction_sum_l3318_331872

theorem max_fraction_sum (a b c d : ℕ) (h1 : a/b + c/d < 1) (h2 : a + c = 20) :
  ∃ (a₀ b₀ c₀ d₀ : ℕ), 
    a₀/b₀ + c₀/d₀ = 1385/1386 ∧ 
    a₀ + c₀ = 20 ∧
    a₀/b₀ + c₀/d₀ < 1 ∧
    ∀ (x y z w : ℕ), x + z = 20 → x/y + z/w < 1 → x/y + z/w ≤ 1385/1386 :=
sorry

end max_fraction_sum_l3318_331872


namespace no_solution_for_equation_l3318_331873

theorem no_solution_for_equation : 
  ¬ ∃ (x : ℝ), x ≠ 3 ∧ (2 - x) / (x - 3) = 1 / (3 - x) - 2 := by
  sorry

end no_solution_for_equation_l3318_331873


namespace ellipse_eccentricity_l3318_331818

/-- Given an ellipse where the length of the major axis is twice the length of the minor axis,
    prove that its eccentricity is √3/2. -/
theorem ellipse_eccentricity (a b : ℝ) (h : a = 2 * b) (h_pos : a > 0) :
  let c := Real.sqrt (a^2 - b^2)
  c / a = Real.sqrt 3 / 2 := by sorry

end ellipse_eccentricity_l3318_331818


namespace first_term_is_0_375_l3318_331854

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the first 40 terms is 600 -/
  sum_first_40 : (40 : ℝ) / 2 * (2 * a + 39 * d) = 600
  /-- The sum of the next 40 terms (terms 41 to 80) is 1800 -/
  sum_next_40 : (40 : ℝ) / 2 * (2 * (a + 40 * d) + 39 * d) = 1800

/-- The first term of the arithmetic sequence with the given properties is 0.375 -/
theorem first_term_is_0_375 (seq : ArithmeticSequence) : seq.a = 0.375 := by
  sorry

end first_term_is_0_375_l3318_331854


namespace race_time_problem_l3318_331835

/-- Given two racers A and B, where their speeds are in the ratio 3:4 and A takes 30 minutes more
    than B to reach the destination, prove that A takes 120 minutes to reach the destination. -/
theorem race_time_problem (v_A v_B : ℝ) (t_A t_B : ℝ) (D : ℝ) :
  v_A / v_B = 3 / 4 →  -- speeds are in ratio 3:4
  t_A = t_B + 30 →     -- A takes 30 minutes more than B
  D = v_A * t_A →      -- distance = speed * time for A
  D = v_B * t_B →      -- distance = speed * time for B
  t_A = 120 :=         -- A takes 120 minutes
by sorry

end race_time_problem_l3318_331835


namespace average_decrease_l3318_331832

theorem average_decrease (n : ℕ) (old_avg new_obs : ℚ) : 
  n = 6 →
  old_avg = 12 →
  new_obs = 5 →
  (n * old_avg + new_obs) / (n + 1) = old_avg - 1 := by
  sorry

end average_decrease_l3318_331832


namespace commercials_time_l3318_331827

/-- Given a total time and a ratio of music to commercials, 
    calculate the number of minutes of commercials played. -/
theorem commercials_time (total_time : ℕ) (music_ratio commercial_ratio : ℕ) 
  (h1 : total_time = 112)
  (h2 : music_ratio = 9)
  (h3 : commercial_ratio = 5) :
  (total_time * commercial_ratio) / (music_ratio + commercial_ratio) = 40 := by
  sorry

#check commercials_time

end commercials_time_l3318_331827


namespace lunch_percentage_boys_l3318_331866

theorem lunch_percentage_boys (C B G : ℝ) (P_b : ℝ) :
  B / G = 3 / 2 →
  C = B + G →
  0.52 * C = (P_b / 100) * B + (40 / 100) * G →
  P_b = 60 := by
  sorry

end lunch_percentage_boys_l3318_331866


namespace system_solution_l3318_331869

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 5 * x^2 - 14 * x * y + 10 * y^2 = 17
def equation2 (x y : ℝ) : Prop := 4 * x^2 - 10 * x * y + 6 * y^2 = 8

-- Define the solution set
def solutions : List (ℝ × ℝ) := [(-1, -2), (11, 7), (-11, -7), (1, 2)]

-- Theorem statement
theorem system_solution :
  ∀ (p : ℝ × ℝ), p ∈ solutions → equation1 p.1 p.2 ∧ equation2 p.1 p.2 :=
by sorry

end system_solution_l3318_331869


namespace painting_price_l3318_331817

theorem painting_price (purchase_price : ℝ) (original_price : ℝ) : 
  purchase_price = 200 → 
  purchase_price = (1/4) * original_price → 
  original_price = 800 := by
sorry

end painting_price_l3318_331817


namespace M_sum_l3318_331805

def M : ℕ → ℕ
| 0 => 3^2
| 1 => 6^2
| n+2 => (3*n + 6)^2 - (3*n + 3)^2 + M n

theorem M_sum : M 49 = 11475 := by
  sorry

end M_sum_l3318_331805


namespace ln_x_plus_one_negative_condition_l3318_331824

theorem ln_x_plus_one_negative_condition (x : ℝ) :
  (∀ x, (Real.log (x + 1) < 0) → (x < 0)) ∧
  (∃ x, x < 0 ∧ ¬(Real.log (x + 1) < 0)) :=
sorry

end ln_x_plus_one_negative_condition_l3318_331824


namespace trigonometric_identity_l3318_331848

theorem trigonometric_identity (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi) (h3 : -Real.sin α = 2 * Real.cos α) :
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 11 / 5 := by
  sorry

end trigonometric_identity_l3318_331848


namespace ohara_triple_81_49_l3318_331822

/-- O'Hara triple definition -/
def is_ohara_triple (a b x : ℕ) : Prop := Real.sqrt a + Real.sqrt b = x

/-- The main theorem -/
theorem ohara_triple_81_49 (x : ℕ) :
  is_ohara_triple 81 49 x → x = 16 := by
  sorry

end ohara_triple_81_49_l3318_331822


namespace farm_birds_l3318_331807

theorem farm_birds (chickens ducks turkeys : ℕ) : 
  ducks = 2 * chickens →
  turkeys = 3 * ducks →
  chickens + ducks + turkeys = 1800 →
  chickens = 200 := by
sorry

end farm_birds_l3318_331807


namespace jazmin_dolls_count_l3318_331899

theorem jazmin_dolls_count (geraldine_dolls : ℝ) (difference : ℕ) :
  geraldine_dolls = 2186.0 →
  difference = 977 →
  geraldine_dolls - difference = 1209 :=
by sorry

end jazmin_dolls_count_l3318_331899


namespace unit_digit_of_x_is_six_l3318_331814

theorem unit_digit_of_x_is_six :
  let x : ℤ := (-2)^1988
  ∃ k : ℤ, x = 10 * k + 6 :=
by sorry

end unit_digit_of_x_is_six_l3318_331814


namespace octahedron_theorem_l3318_331888

/-- A point in 3D space -/
structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Checks if a point lies on any plane defined by x ± y ± z = n for integer n -/
def liesOnPlane (p : Point3D) : Prop :=
  ∃ n : ℤ, (p.x + p.y + p.z = n) ∨ (p.x + p.y - p.z = n) ∨
           (p.x - p.y + p.z = n) ∨ (p.x - p.y - p.z = n) ∨
           (-p.x + p.y + p.z = n) ∨ (-p.x + p.y - p.z = n) ∨
           (-p.x - p.y + p.z = n) ∨ (-p.x - p.y - p.z = n)

/-- Checks if a point lies strictly inside an octahedron -/
def insideOctahedron (p : Point3D) : Prop :=
  ∃ n : ℤ, (n < p.x + p.y + p.z) ∧ (p.x + p.y + p.z < n + 1) ∧
           (n < p.x + p.y - p.z) ∧ (p.x + p.y - p.z < n + 1) ∧
           (n < p.x - p.y + p.z) ∧ (p.x - p.y + p.z < n + 1) ∧
           (n < -p.x + p.y + p.z) ∧ (-p.x + p.y + p.z < n + 1)

theorem octahedron_theorem (p : Point3D) (h : ¬ liesOnPlane p) :
  ∃ k : ℕ, insideOctahedron ⟨k * p.x, k * p.y, k * p.z⟩ := by
  sorry

end octahedron_theorem_l3318_331888


namespace joan_video_game_spending_l3318_331870

/-- The cost of the basketball game Joan purchased -/
def basketball_cost : ℚ := 5.20

/-- The cost of the racing game Joan purchased -/
def racing_cost : ℚ := 4.23

/-- The total amount Joan spent on video games -/
def total_spent : ℚ := basketball_cost + racing_cost

/-- Theorem stating that the total amount Joan spent on video games is $9.43 -/
theorem joan_video_game_spending :
  total_spent = 9.43 := by sorry

end joan_video_game_spending_l3318_331870


namespace problem_1_l3318_331828

theorem problem_1 : (1 + 1/4 - 5/6 + 1/2) * (-12) = -11 := by
  sorry

end problem_1_l3318_331828


namespace space_diagonals_of_specific_polyhedron_l3318_331895

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - (2 * Q.quadrilateral_faces)

/-- Theorem: A convex polyhedron Q with 30 vertices, 70 edges, 42 faces
    (30 triangular and 12 quadrilateral) has 341 space diagonals -/
theorem space_diagonals_of_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 42,
    triangular_faces := 30,
    quadrilateral_faces := 12
  }
  space_diagonals Q = 341 := by sorry

end space_diagonals_of_specific_polyhedron_l3318_331895


namespace gcd_1230_990_l3318_331883

theorem gcd_1230_990 : Nat.gcd 1230 990 = 30 := by
  sorry

end gcd_1230_990_l3318_331883


namespace largest_number_l3318_331802

theorem largest_number : 
  let a := 0.989
  let b := 0.9879
  let c := 0.98809
  let d := 0.9807
  let e := 0.9819
  (a ≥ b) ∧ (a ≥ c) ∧ (a ≥ d) ∧ (a ≥ e) :=
by sorry

end largest_number_l3318_331802


namespace hotel_room_charges_l3318_331860

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R * 0.8)  -- P is 20% less than R
  (h2 : P = G * 0.9)  -- P is 10% less than G
  : R = G * 1.125 :=  -- R is 12.5% greater than G
by sorry

end hotel_room_charges_l3318_331860


namespace marble_selection_ways_l3318_331816

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of marbles -/
def total_marbles : ℕ := 15

/-- The number of marbles to be chosen -/
def marbles_to_choose : ℕ := 5

/-- The number of specific colored marbles (red + green + blue) -/
def specific_colored_marbles : ℕ := 6

/-- The number of ways to choose 2 marbles from the specific colored ones -/
def ways_to_choose_specific : ℕ := 9

/-- The number of remaining marbles after removing the specific colored ones -/
def remaining_marbles : ℕ := total_marbles - specific_colored_marbles

theorem marble_selection_ways :
  ways_to_choose_specific * choose remaining_marbles (marbles_to_choose - 2) = 1980 :=
sorry

end marble_selection_ways_l3318_331816


namespace factorial_inequality_l3318_331864

theorem factorial_inequality (k : ℕ) (h : k ≥ 2) :
  ((k + 1) / 2 : ℝ) ^ k > k! :=
by sorry

end factorial_inequality_l3318_331864


namespace sum_of_digits_253_l3318_331892

/-- Given a three-digit number with specific properties, prove that the sum of its digits is 10 -/
theorem sum_of_digits_253 (a b c : ℕ) : 
  -- The number is 253
  100 * a + 10 * b + c = 253 →
  -- The middle digit is the sum of the other two
  b = a + c →
  -- Reversing the digits increases the number by 99
  100 * c + 10 * b + a = 100 * a + 10 * b + c + 99 →
  -- The sum of the digits is 10
  a + b + c = 10 := by
sorry


end sum_of_digits_253_l3318_331892


namespace equation_solution_l3318_331897

theorem equation_solution : 
  ∃ x : ℚ, (x ≠ 0 ∧ x ≠ 1) ∧ ((x - 1) / x + 3 * x / (x - 1) = 4) ∧ x = -1/2 := by
  sorry

end equation_solution_l3318_331897


namespace domestic_needs_fraction_l3318_331884

def total_income : ℚ := 200
def provident_fund_rate : ℚ := 1/16
def insurance_premium_rate : ℚ := 1/15
def bank_deposit : ℚ := 50

def remaining_after_provident_fund : ℚ := total_income * (1 - provident_fund_rate)
def remaining_after_insurance : ℚ := remaining_after_provident_fund * (1 - insurance_premium_rate)

theorem domestic_needs_fraction :
  (remaining_after_insurance - bank_deposit) / remaining_after_insurance = 5/7 := by
  sorry

end domestic_needs_fraction_l3318_331884


namespace aaron_erasers_l3318_331853

theorem aaron_erasers (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 81 → given_away = 34 → remaining = initial - given_away → remaining = 47 := by
  sorry

end aaron_erasers_l3318_331853


namespace parabola_translation_l3318_331842

/-- Represents a parabola of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k + dy }

theorem parabola_translation (p : Parabola) (dx dy : ℝ) :
  p.a = 2 ∧ p.h = 4 ∧ p.k = 3 ∧ dx = 4 ∧ dy = 3 →
  let p' := translate p dx dy
  p'.a = 2 ∧ p'.h = 0 ∧ p'.k = 6 := by
  sorry

end parabola_translation_l3318_331842


namespace original_number_from_sum_l3318_331881

/-- Represents a three-digit number in base 10 -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_ones : ones < 10
  h_not_zero : hundreds ≠ 0

/-- Calculates the sum of a three-digit number and its permutations -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  let a := n.hundreds
  let b := n.tens
  let c := n.ones
  222 * (a + b + c)

/-- The main theorem -/
theorem original_number_from_sum (N : Nat) (h_N : N = 3237) :
  ∃ (n : ThreeDigitNumber), sumOfPermutations n = N ∧ n.hundreds = 4 ∧ n.tens = 2 ∧ n.ones = 9 := by
  sorry

end original_number_from_sum_l3318_331881


namespace miss_both_mutually_exclusive_not_contradictory_l3318_331809

-- Define the sample space for two shots
inductive ShotOutcome
| HitBoth
| HitFirst
| HitSecond
| MissBoth

-- Define the events
def hit_exactly_once (outcome : ShotOutcome) : Prop :=
  outcome = ShotOutcome.HitFirst ∨ outcome = ShotOutcome.HitSecond

def miss_both (outcome : ShotOutcome) : Prop :=
  outcome = ShotOutcome.MissBoth

-- Theorem stating that "Miss both times" is mutually exclusive but not contradictory to "hit exactly once"
theorem miss_both_mutually_exclusive_not_contradictory :
  (∀ outcome : ShotOutcome, ¬(hit_exactly_once outcome ∧ miss_both outcome)) ∧
  (∃ outcome : ShotOutcome, hit_exactly_once outcome ∨ miss_both outcome) :=
sorry

end miss_both_mutually_exclusive_not_contradictory_l3318_331809


namespace ones_digit_of_largest_power_of_three_dividing_18_factorial_l3318_331813

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def largest_power_of_three_dividing (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc + (i + 1).log 3) 0

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_three_dividing_18_factorial :
  ones_digit (3^(largest_power_of_three_dividing (factorial 18))) = 1 := by sorry

end ones_digit_of_largest_power_of_three_dividing_18_factorial_l3318_331813


namespace doubled_container_volume_l3318_331874

/-- The volume of a container after doubling its dimensions -/
def doubled_volume (original_volume : ℝ) : ℝ := 8 * original_volume

/-- Theorem: Doubling the dimensions of a 4-gallon container results in a 32-gallon container -/
theorem doubled_container_volume : doubled_volume 4 = 32 := by
  sorry

end doubled_container_volume_l3318_331874


namespace total_peaches_l3318_331898

/-- The number of baskets -/
def num_baskets : ℕ := 11

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 10

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 18

/-- Theorem: The total number of peaches in all baskets is 308 -/
theorem total_peaches :
  (num_baskets * (red_peaches_per_basket + green_peaches_per_basket)) = 308 := by
  sorry

end total_peaches_l3318_331898


namespace function_inequality_l3318_331889

open Real

/-- Given two differentiable functions f and g on ℝ, if f'(x) > g'(x) for all x,
    then for a < x < b, we have f(x) + g(b) < g(x) + f(b) and f(x) + g(a) > g(x) + f(a) -/
theorem function_inequality (f g : ℝ → ℝ) (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
    (h_deriv : ∀ x, deriv f x > deriv g x) (a b x : ℝ) (h_x : a < x ∧ x < b) :
    (f x + g b < g x + f b) ∧ (f x + g a > g x + f a) := by
  sorry

end function_inequality_l3318_331889


namespace two_numbers_problem_l3318_331823

theorem two_numbers_problem : ∃ (A B : ℕ+), 
  A + B = 581 ∧ 
  Nat.lcm A B / Nat.gcd A B = 240 ∧ 
  ((A = 560 ∧ B = 21) ∨ (A = 21 ∧ B = 560)) := by
  sorry

end two_numbers_problem_l3318_331823


namespace eric_egg_collection_days_l3318_331861

/-- Proves that Eric waited 3 days to collect 36 eggs from 4 chickens laying 3 eggs each per day -/
theorem eric_egg_collection_days (num_chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (total_eggs : ℕ) : 
  num_chickens = 4 → 
  eggs_per_chicken_per_day = 3 → 
  total_eggs = 36 → 
  (total_eggs / (num_chickens * eggs_per_chicken_per_day) : ℕ) = 3 := by
  sorry

end eric_egg_collection_days_l3318_331861


namespace student_rank_from_right_l3318_331819

theorem student_rank_from_right 
  (total_students : ℕ) 
  (rank_from_left : ℕ) 
  (h1 : total_students = 21) 
  (h2 : rank_from_left = 5) : 
  total_students - rank_from_left + 1 = 18 := by
  sorry

end student_rank_from_right_l3318_331819


namespace income_expenditure_ratio_l3318_331804

/-- Represents the financial data of a person -/
structure FinancialData where
  income : ℕ
  savings : ℕ

/-- Calculates the expenditure given income and savings -/
def expenditure (data : FinancialData) : ℕ :=
  data.income - data.savings

/-- Simplifies a ratio represented by two natural numbers -/
def simplifyRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

/-- The main theorem stating the ratio of income to expenditure -/
theorem income_expenditure_ratio (data : FinancialData) 
  (h1 : data.income = 40000) 
  (h2 : data.savings = 5000) : 
  simplifyRatio data.income (expenditure data) = (8, 7) := by
  sorry

#eval simplifyRatio 40000 35000

end income_expenditure_ratio_l3318_331804


namespace internally_tangent_circles_distance_l3318_331879

theorem internally_tangent_circles_distance (r₁ r₂ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 4) :
  let d := (r₁ - r₂)^2 + r₂^2
  d = (4 * Real.sqrt 10)^2 :=
by sorry

end internally_tangent_circles_distance_l3318_331879


namespace two_digit_number_proof_l3318_331833

theorem two_digit_number_proof : ∃! n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (n / 10 = 2 * (n % 10)) ∧ 
  (∃ m : ℕ, n + (n / 10)^2 = m^2) ∧
  n = 21 := by sorry

end two_digit_number_proof_l3318_331833


namespace soccer_team_lineup_combinations_l3318_331846

def choose (n k : ℕ) : ℕ := Nat.choose n k

def total_players : ℕ := 18
def twins : ℕ := 2
def lineup_size : ℕ := 8
def defenders : ℕ := 5

theorem soccer_team_lineup_combinations : 
  (choose 2 1 * choose 5 3 * choose 11 4) +
  (choose 2 2 * choose 5 3 * choose 11 3) +
  (choose 2 1 * choose 5 4 * choose 11 3) +
  (choose 2 2 * choose 5 4 * choose 11 2) +
  (choose 2 1 * choose 5 5 * choose 11 2) +
  (choose 2 2 * choose 5 5 * choose 11 1) = 3602 := by
  sorry

end soccer_team_lineup_combinations_l3318_331846


namespace inequality_holds_l3318_331810

theorem inequality_holds (x : ℝ) : (1 : ℝ) / (x^2 + 1) > (1 : ℝ) / (x^2 + 2) := by
  sorry

end inequality_holds_l3318_331810


namespace pentagon_distance_equality_l3318_331838

/-- A regular pentagon with vertices A1, A2, A3, A4, A5 -/
structure RegularPentagon where
  A1 : ℝ × ℝ
  A2 : ℝ × ℝ
  A3 : ℝ × ℝ
  A4 : ℝ × ℝ
  A5 : ℝ × ℝ
  is_regular : True  -- We assume this property without defining it explicitly

/-- The circumcircle of the regular pentagon -/
def circumcircle (p : RegularPentagon) : Set (ℝ × ℝ) :=
  sorry  -- Definition of the circumcircle

/-- The arc A1A5 of the circumcircle -/
def arcA1A5 (p : RegularPentagon) : Set (ℝ × ℝ) :=
  sorry  -- Definition of the arc A1A5

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry  -- Definition of Euclidean distance

/-- Statement of the theorem -/
theorem pentagon_distance_equality (p : RegularPentagon) (B : ℝ × ℝ)
    (h1 : B ∈ arcA1A5 p)
    (h2 : distance B p.A1 < distance B p.A5) :
    distance B p.A1 + distance B p.A3 + distance B p.A5 =
    distance B p.A2 + distance B p.A4 := by
  sorry

end pentagon_distance_equality_l3318_331838


namespace log_xy_value_l3318_331882

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 3) :
  Real.log (x * y) = 7/5 := by
  sorry

end log_xy_value_l3318_331882


namespace sum_of_common_ratios_of_geometric_sequences_l3318_331850

theorem sum_of_common_ratios_of_geometric_sequences 
  (k a₂ a₃ b₂ b₃ p r : ℝ) 
  (h1 : k ≠ 0)
  (h2 : p ≠ 1)
  (h3 : r ≠ 1)
  (h4 : p ≠ r)
  (h5 : a₂ = k * p)
  (h6 : a₃ = k * p^2)
  (h7 : b₂ = k * r)
  (h8 : b₃ = k * r^2)
  (h9 : a₃ - b₃ = 3 * (a₂ - b₂)) :
  p + r = 3 := by
sorry

end sum_of_common_ratios_of_geometric_sequences_l3318_331850


namespace extended_pattern_ratio_l3318_331863

/-- Represents a square tile pattern -/
structure TilePattern :=
  (side : ℕ)
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- The initial square pattern -/
def initial_pattern : TilePattern :=
  { side := 5
  , black_tiles := 8
  , white_tiles := 17 }

/-- Extends a tile pattern by adding a black border -/
def extend_pattern (p : TilePattern) : TilePattern :=
  { side := p.side + 2
  , black_tiles := p.black_tiles + 2 * p.side + 2 * (p.side + 2)
  , white_tiles := p.white_tiles }

/-- The theorem to be proved -/
theorem extended_pattern_ratio (p : TilePattern) : 
  p = initial_pattern → 
  (extend_pattern p).black_tiles = 32 ∧ (extend_pattern p).white_tiles = 17 := by
  sorry

end extended_pattern_ratio_l3318_331863


namespace square_difference_of_constrained_integers_l3318_331843

theorem square_difference_of_constrained_integers (x y : ℕ+) 
  (h1 : 56 ≤ (x:ℝ) + y ∧ (x:ℝ) + y ≤ 59)
  (h2 : (0.9:ℝ) < (x:ℝ) / y ∧ (x:ℝ) / y < 0.91) :
  (y:ℤ)^2 - (x:ℤ)^2 = 177 := by
  sorry

end square_difference_of_constrained_integers_l3318_331843


namespace intersection_of_M_and_N_l3318_331834

def M : Set ℝ := {x | x^2 - 1 ≤ 0}
def N : Set ℝ := {x | x^2 - 3*x > 0}

theorem intersection_of_M_and_N : ∀ x : ℝ, x ∈ M ∩ N ↔ -1 ≤ x ∧ x < 0 := by sorry

end intersection_of_M_and_N_l3318_331834


namespace smallest_product_is_690_l3318_331831

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def smallest_three_digit_product (m : ℕ) : Prop :=
  ∃ a b : ℕ,
    m = a * b * (10*a + b) * (a + b) ∧
    a < 10 ∧ b < 10 ∧
    is_prime a ∧ is_prime b ∧
    is_prime (10*a + b) ∧ is_prime (a + b) ∧
    (a + b) % 5 = 1 ∧
    m ≥ 100 ∧ m < 1000 ∧
    ∀ n : ℕ, n ≥ 100 ∧ n < m → ¬(smallest_three_digit_product n)

theorem smallest_product_is_690 :
  smallest_three_digit_product 690 :=
sorry

end smallest_product_is_690_l3318_331831


namespace decimal_to_fraction_l3318_331865

theorem decimal_to_fraction (x : ℚ) (h : x = 3.68) : 
  ∃ (n d : ℕ), d ≠ 0 ∧ x = n / d ∧ n = 92 ∧ d = 25 := by
  sorry

end decimal_to_fraction_l3318_331865


namespace linda_age_l3318_331837

theorem linda_age (carlos_age maya_age linda_age : ℕ) : 
  carlos_age = 12 →
  maya_age = carlos_age + 4 →
  linda_age = maya_age - 5 →
  linda_age = 11 :=
by
  sorry

end linda_age_l3318_331837


namespace area_2018_correct_l3318_331893

/-- Calculates the area to be converted after a given number of years -/
def area_to_convert (initial_area : ℝ) (annual_increase : ℝ) (years : ℕ) : ℝ :=
  initial_area * (1 + annual_increase) ^ years

/-- Proves that the area to be converted in 2018 is correct -/
theorem area_2018_correct (initial_area : ℝ) (annual_increase : ℝ) :
  initial_area = 8 →
  annual_increase = 0.1 →
  area_to_convert initial_area annual_increase 5 = 8 * 1.1^5 := by
  sorry

#check area_2018_correct

end area_2018_correct_l3318_331893


namespace cubic_root_equation_solutions_l3318_331880

theorem cubic_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (17*x - 2)^(1/3) + (11*x + 2)^(1/3) - 2*(9*x)^(1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = (2 + Real.sqrt 35) / 31 ∨ x = (2 - Real.sqrt 35) / 31 := by
  sorry

end cubic_root_equation_solutions_l3318_331880


namespace division_of_decimals_l3318_331890

theorem division_of_decimals : (0.25 : ℚ) / (0.005 : ℚ) = 50 := by sorry

end division_of_decimals_l3318_331890


namespace individual_can_cost_l3318_331871

def pack_size : ℕ := 12
def pack_cost : ℚ := 299 / 100  -- $2.99 represented as a rational number

theorem individual_can_cost :
  let cost_per_can := pack_cost / pack_size
  cost_per_can = 299 / (100 * 12) := by sorry

end individual_can_cost_l3318_331871


namespace exists_double_application_square_l3318_331896

theorem exists_double_application_square :
  ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n^2 := by sorry

end exists_double_application_square_l3318_331896


namespace smallest_w_l3318_331855

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w : 
  ∀ w : ℕ, 
    w > 0 →
    is_factor (2^6) (1152 * w) →
    is_factor (3^4) (1152 * w) →
    is_factor (5^3) (1152 * w) →
    is_factor (7^2) (1152 * w) →
    is_factor 11 (1152 * w) →
    w ≥ 16275 :=
by sorry

end smallest_w_l3318_331855


namespace total_options_is_twenty_l3318_331839

/-- The number of high-speed trains from location A to location B -/
def num_trains : ℕ := 5

/-- The number of ferries from location B to location C -/
def num_ferries : ℕ := 4

/-- The total number of travel options from location A to location C -/
def total_options : ℕ := num_trains * num_ferries

/-- Theorem stating that the total number of travel options is 20 -/
theorem total_options_is_twenty : total_options = 20 := by
  sorry

end total_options_is_twenty_l3318_331839


namespace union_of_A_and_B_intersection_condition_l3318_331812

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 2| < a}
def B : Set ℝ := {x : ℝ | (2*x - 1) / (x + 2) < 1}

-- Part 1
theorem union_of_A_and_B :
  A 2 ∪ B = {x : ℝ | -2 < x ∧ x < 4} := by sorry

-- Part 2
theorem intersection_condition (a : ℝ) :
  A a ∩ B = A a ↔ a ≤ 1 := by sorry

end union_of_A_and_B_intersection_condition_l3318_331812


namespace geometric_sequence_ratio_l3318_331885

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n,
    prove that if a_3 = 2S_2 + 1 and a_4 = 2S_3 + 1, then q = 3. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- a_n is a geometric sequence with common ratio q
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- S_n is the sum of first n terms
  a 3 = 2 * S 2 + 1 →  -- a_3 = 2S_2 + 1
  a 4 = 2 * S 3 + 1 →  -- a_4 = 2S_3 + 1
  q = 3 :=
by sorry

end geometric_sequence_ratio_l3318_331885


namespace problem_solution_l3318_331820

theorem problem_solution : 25 * (216 / 3 + 36 / 6 + 16 / 25 + 2) = 2016 := by
  sorry

end problem_solution_l3318_331820


namespace die_roll_probability_l3318_331847

theorem die_roll_probability : 
  let n : ℕ := 8  -- number of rolls
  let p_even : ℚ := 1/2  -- probability of rolling an even number
  let p_odd : ℚ := 1 - p_even  -- probability of rolling an odd number
  1 - p_odd^n = 255/256 :=
by sorry

end die_roll_probability_l3318_331847


namespace dividing_trapezoid_mn_length_l3318_331849

/-- A trapezoid with bases a and b, and a segment MN parallel to the bases that divides the area in half -/
structure DividingTrapezoid (a b : ℝ) where
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (mn_length : ℝ)
  (mn_divides_area : mn_length ^ 2 = (a ^ 2 + b ^ 2) / 2)

/-- The length of MN in a DividingTrapezoid is √((a² + b²) / 2) -/
theorem dividing_trapezoid_mn_length (a b : ℝ) (t : DividingTrapezoid a b) :
  t.mn_length = Real.sqrt ((a ^ 2 + b ^ 2) / 2) := by
  sorry


end dividing_trapezoid_mn_length_l3318_331849


namespace hyperbola_center_trajectory_l3318_331844

/-- The equation of the trajectory of the center of a hyperbola -/
theorem hyperbola_center_trajectory 
  (x y m : ℝ) 
  (h : x^2 - y^2 - 6*m*x - 4*m*y + 5*m^2 - 1 = 0) : 
  2*x + 3*y = 0 := by
  sorry

end hyperbola_center_trajectory_l3318_331844


namespace intersection_A_complement_B_l3318_331811

-- Define set A
def A : Set ℝ := {x | |x| < 1}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Ioo (-1 : ℝ) 0 := by sorry

end intersection_A_complement_B_l3318_331811


namespace quadratic_inequality_solution_condition_l3318_331887

theorem quadratic_inequality_solution_condition (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ |a| ≥ 2 := by
  sorry

end quadratic_inequality_solution_condition_l3318_331887


namespace product_of_digits_of_largest_valid_number_l3318_331836

/-- A function that returns true if the digits of a natural number are in strictly increasing order --/
def strictly_increasing_digits (n : ℕ) : Prop := sorry

/-- A function that returns the sum of the squares of the digits of a natural number --/
def sum_of_squared_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the product of the digits of a natural number --/
def product_of_digits (n : ℕ) : ℕ := sorry

/-- The largest natural number whose digits are in strictly increasing order and whose digits' squares sum to 50 --/
def largest_valid_number : ℕ := sorry

theorem product_of_digits_of_largest_valid_number : 
  strictly_increasing_digits largest_valid_number ∧ 
  sum_of_squared_digits largest_valid_number = 50 ∧
  product_of_digits largest_valid_number = 36 ∧
  ∀ m : ℕ, 
    strictly_increasing_digits m ∧ 
    sum_of_squared_digits m = 50 → 
    m ≤ largest_valid_number :=
sorry

end product_of_digits_of_largest_valid_number_l3318_331836


namespace marble_arrangement_remainder_l3318_331868

def green_marbles : ℕ := 7

-- m is the maximum number of red marbles
def red_marbles (m : ℕ) : Prop := 
  m = 19 ∧ ∀ k, k > m → ¬∃ (arr : List (Fin 2)), 
    arr.length = green_marbles + k ∧ 
    (arr.countP (λ i => arr[i]? = arr[i+1]?)) = 
    (arr.countP (λ i => arr[i]? ≠ arr[i+1]?))

-- N is the number of valid arrangements
def valid_arrangements (m : ℕ) : ℕ := Nat.choose (m + green_marbles) green_marbles

theorem marble_arrangement_remainder (m : ℕ) : 
  red_marbles m → valid_arrangements m % 1000 = 388 := by sorry

end marble_arrangement_remainder_l3318_331868


namespace refrigerator_price_calculation_l3318_331845

/-- The purchase price of the refrigerator in rupees -/
def refrigerator_price : ℝ := 15000

/-- The purchase price of the mobile phone in rupees -/
def mobile_price : ℝ := 8000

/-- The loss percentage on the refrigerator -/
def refrigerator_loss_percent : ℝ := 0.03

/-- The profit percentage on the mobile phone -/
def mobile_profit_percent : ℝ := 0.10

/-- The overall profit in rupees -/
def overall_profit : ℝ := 350

theorem refrigerator_price_calculation :
  refrigerator_price * (1 - refrigerator_loss_percent) +
  mobile_price * (1 + mobile_profit_percent) =
  refrigerator_price + mobile_price + overall_profit := by
  sorry

end refrigerator_price_calculation_l3318_331845


namespace equation_system_properties_l3318_331800

/-- Represents a system of equations mx + ny² = 0 and mx² + ny² = 1 -/
structure EquationSystem where
  m : ℝ
  n : ℝ
  h_m_neg : m < 0
  h_n_pos : n > 0

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point satisfies both equations in the system -/
def satisfies_equations (sys : EquationSystem) (p : Point) : Prop :=
  sys.m * p.x + sys.n * p.y^2 = 0 ∧ sys.m * p.x^2 + sys.n * p.y^2 = 1

/-- States that the equation system represents a parabola -/
def is_parabola (sys : EquationSystem) : Prop :=
  ∃ (a b c : ℝ), ∀ (x y : ℝ), sys.m * x + sys.n * y^2 = 0 ↔ y = a * x^2 + b * x + c

/-- States that the equation system represents a hyperbola -/
def is_hyperbola (sys : EquationSystem) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), sys.m * x^2 + sys.n * y^2 = 1 ↔ (x^2 / a^2) - (y^2 / b^2) = 1

theorem equation_system_properties (sys : EquationSystem) :
  is_parabola sys ∧ 
  is_hyperbola sys ∧ 
  satisfies_equations sys ⟨0, 0⟩ ∧ 
  satisfies_equations sys ⟨1, 0⟩ :=
sorry

end equation_system_properties_l3318_331800


namespace teacup_rows_per_box_l3318_331878

def total_boxes : ℕ := 26
def boxes_with_pans : ℕ := 6
def cups_per_row : ℕ := 4
def cups_broken_per_box : ℕ := 2
def teacups_left : ℕ := 180

def boxes_with_teacups : ℕ := (total_boxes - boxes_with_pans) / 2

theorem teacup_rows_per_box :
  let total_teacups := teacups_left + cups_broken_per_box * boxes_with_teacups
  let teacups_per_box := total_teacups / boxes_with_teacups
  teacups_per_box / cups_per_row = 5 := by
  sorry

end teacup_rows_per_box_l3318_331878


namespace problem_statement_l3318_331801

theorem problem_statement (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 108) :
  a^2 * b + a * b^2 = 108 := by sorry

end problem_statement_l3318_331801


namespace jose_investment_is_45000_l3318_331830

/-- Represents the investment and profit scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment based on the given conditions --/
def calculate_jose_investment (shop : ShopInvestment) : ℕ :=
  let tom_investment_months := shop.tom_investment * 12
  let jose_investment_months := (12 - shop.jose_join_delay) * (shop.total_profit - shop.jose_profit) * 10 / shop.jose_profit
  jose_investment_months / (12 - shop.jose_join_delay)

/-- Theorem stating that Jose's investment is 45000 given the specified conditions --/
theorem jose_investment_is_45000 (shop : ShopInvestment)
  (h1 : shop.tom_investment = 30000)
  (h2 : shop.jose_join_delay = 2)
  (h3 : shop.total_profit = 36000)
  (h4 : shop.jose_profit = 20000) :
  calculate_jose_investment shop = 45000 := by
  sorry

#eval calculate_jose_investment ⟨30000, 2, 36000, 20000⟩

end jose_investment_is_45000_l3318_331830


namespace no_matrix_transformation_l3318_331886

theorem no_matrix_transformation (a b c d : ℝ) : 
  ¬ ∃ (N : Matrix (Fin 2) (Fin 2) ℝ), 
    N • !![a, b; c, d] = !![d, c; b, a] := by
  sorry

end no_matrix_transformation_l3318_331886


namespace inverse_of_17_mod_43_l3318_331806

theorem inverse_of_17_mod_43 :
  ∃ x : ℕ, x < 43 ∧ (17 * x) % 43 = 1 :=
by
  use 6
  sorry

end inverse_of_17_mod_43_l3318_331806


namespace large_circle_radius_l3318_331859

/-- Configuration of circles -/
structure CircleConfiguration where
  small_radius : ℝ
  chord_length : ℝ
  small_circle_count : ℕ

/-- Theorem: If five identical circles are placed in a line inside a larger circle,
    and the chord connecting the endpoints of the line of circles has length 16,
    then the radius of the large circle is 8. -/
theorem large_circle_radius
  (config : CircleConfiguration)
  (h1 : config.small_circle_count = 5)
  (h2 : config.chord_length = 16) :
  4 * config.small_radius = 8 := by
  sorry

#check large_circle_radius

end large_circle_radius_l3318_331859


namespace high_school_language_study_l3318_331856

theorem high_school_language_study (total_students : ℕ) 
  (spanish_min spanish_max french_min french_max : ℕ) :
  total_students = 2001 →
  spanish_min = 1601 →
  spanish_max = 1700 →
  french_min = 601 →
  french_max = 800 →
  let m := spanish_min + french_min - total_students
  let M := spanish_max + french_max - total_students
  M - m = 298 := by
sorry

end high_school_language_study_l3318_331856


namespace soda_weight_proof_l3318_331862

/-- Calculates the amount of soda in each can given the total weight, number of cans, and weight of empty cans. -/
def soda_per_can (total_weight : ℕ) (soda_cans : ℕ) (empty_cans : ℕ) (empty_can_weight : ℕ) : ℕ :=
  (total_weight - (soda_cans + empty_cans) * empty_can_weight) / soda_cans

/-- Proves that the amount of soda in each can is 12 ounces given the problem conditions. -/
theorem soda_weight_proof (total_weight : ℕ) (soda_cans : ℕ) (empty_cans : ℕ) (empty_can_weight : ℕ)
  (h1 : total_weight = 88)
  (h2 : soda_cans = 6)
  (h3 : empty_cans = 2)
  (h4 : empty_can_weight = 2) :
  soda_per_can total_weight soda_cans empty_cans empty_can_weight = 12 := by
  sorry

end soda_weight_proof_l3318_331862


namespace temperature_average_bounds_l3318_331852

theorem temperature_average_bounds (temps : List ℝ) 
  (h_count : temps.length = 5)
  (h_min : temps.minimum? = some 42)
  (h_max : ∀ t ∈ temps, t ≤ 57) : 
  let avg := temps.sum / temps.length
  42 ≤ avg ∧ avg ≤ 57 := by sorry

end temperature_average_bounds_l3318_331852


namespace box_volume_is_3888_l3318_331877

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  length : ℝ
  width : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.height * d.length * d.width

/-- Theorem: The volume of the box with given dimensions is 3888 cubic inches -/
theorem box_volume_is_3888 :
  let d : BoxDimensions := {
    height := 12,
    length := 12 * 3,
    width := 12 * 3 / 4
  }
  boxVolume d = 3888 := by
  sorry


end box_volume_is_3888_l3318_331877


namespace locus_of_point_l3318_331851

/-- Given three lines in a plane not passing through the origin, prove the locus of a point P
    satisfying certain conditions. -/
theorem locus_of_point (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) :
  let l₁ : ℝ × ℝ → Prop := λ (x, y) ↦ a₁ * x + b₁ * y + c₁ = 0
  let l₂ : ℝ × ℝ → Prop := λ (x, y) ↦ a₂ * x + b₂ * y + c₂ = 0
  let l₃ : ℝ × ℝ → Prop := λ (x, y) ↦ a₃ * x + b₃ * y + c₃ = 0
  let origin : ℝ × ℝ := (0, 0)
  ∀ (l : Set (ℝ × ℝ)) (A B C : ℝ × ℝ),
    (∀ p ∈ l, ∃ t : ℝ, p = (t * (A.1 - origin.1), t * (A.2 - origin.2))) →
    l₁ A ∧ l₂ B ∧ l₃ C →
    A ∈ l ∧ B ∈ l ∧ C ∈ l →
    (∀ P ∈ l, P ≠ origin →
      let ρ₁ := Real.sqrt ((A.1 - origin.1)^2 + (A.2 - origin.2)^2)
      let ρ₂ := Real.sqrt ((B.1 - origin.1)^2 + (B.2 - origin.2)^2)
      let ρ₃ := Real.sqrt ((C.1 - origin.1)^2 + (C.2 - origin.2)^2)
      let ρ  := Real.sqrt ((P.1 - origin.1)^2 + (P.2 - origin.2)^2)
      1 / ρ₁ + 1 / ρ₂ + 1 / ρ₃ = 1 / ρ) →
    ∀ (x y : ℝ),
      (x, y) ∈ l ↔ (a₁ / c₁ + a₂ / c₂ + a₃ / c₃) * x + (b₁ / c₁ + b₂ / c₂ + b₃ / c₃) * y + 1 = 0 :=
by
  sorry

end locus_of_point_l3318_331851


namespace alexei_weekly_loss_l3318_331891

/-- Given the weight loss information for Aleesia and Alexei, calculate Alexei's weekly weight loss. -/
theorem alexei_weekly_loss 
  (aleesia_weekly_loss : ℝ) 
  (aleesia_weeks : ℕ) 
  (alexei_weeks : ℕ) 
  (total_loss : ℝ) 
  (h1 : aleesia_weekly_loss = 1.5)
  (h2 : aleesia_weeks = 10)
  (h3 : alexei_weeks = 8)
  (h4 : total_loss = 35)
  : (total_loss - aleesia_weekly_loss * aleesia_weeks) / alexei_weeks = 2.5 := by
  sorry

end alexei_weekly_loss_l3318_331891


namespace sum_of_cubes_l3318_331840

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -2) : 
  a^3 + b^3 = 45 := by
sorry

end sum_of_cubes_l3318_331840


namespace camel_cost_l3318_331803

/-- The cost of animals in an economy where the relative prices are fixed. -/
structure AnimalCosts where
  camel : ℚ
  horse : ℚ
  goat : ℚ
  ox : ℚ
  elephant : ℚ

/-- The conditions of the animal costs problem. -/
def animal_costs_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  26 * costs.horse = 50 * costs.goat ∧
  20 * costs.goat = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 170000

/-- The theorem stating that under the given conditions, a camel costs 27200. -/
theorem camel_cost (costs : AnimalCosts) :
  animal_costs_conditions costs → costs.camel = 27200 := by
  sorry

end camel_cost_l3318_331803


namespace expected_replant_is_200_l3318_331875

/-- The expected number of seeds to be replanted -/
def expected_replant (p : ℝ) (n : ℕ) (r : ℕ) : ℝ :=
  n * (1 - p) * r

/-- Theorem: The expected number of seeds to be replanted is 200 -/
theorem expected_replant_is_200 :
  expected_replant 0.9 1000 2 = 200 := by
  sorry

end expected_replant_is_200_l3318_331875


namespace no_solution_for_qt_plus_q_plus_t_eq_6_l3318_331858

theorem no_solution_for_qt_plus_q_plus_t_eq_6 :
  ∀ (q t : ℕ), q > 0 ∧ t > 0 → q * t + q + t ≠ 6 := by
  sorry

end no_solution_for_qt_plus_q_plus_t_eq_6_l3318_331858


namespace rectangle_has_equal_diagonals_l3318_331808

-- Define a rectangle
def isRectangle (ABCD : Quadrilateral) : Prop := sorry

-- Define equal diagonals
def hasEqualDiagonals (ABCD : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem rectangle_has_equal_diagonals (ABCD : Quadrilateral) :
  isRectangle ABCD → hasEqualDiagonals ABCD := by sorry

end rectangle_has_equal_diagonals_l3318_331808


namespace quadratic_real_roots_l3318_331815

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m = 0) → m ≤ 1 := by
  sorry

end quadratic_real_roots_l3318_331815


namespace sqrt_product_equality_l3318_331826

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l3318_331826


namespace park_nests_l3318_331841

/-- Calculates the minimum number of nests required for birds in a park -/
def minimum_nests (sparrows pigeons starlings robins : ℕ) 
  (sparrow_nests pigeon_nests starling_nests robin_nests : ℕ) : ℕ :=
  sparrows * sparrow_nests + pigeons * pigeon_nests + 
  starlings * starling_nests + robins * robin_nests

/-- Theorem stating the minimum number of nests required for the given bird populations -/
theorem park_nests : 
  minimum_nests 5 3 6 2 1 2 3 4 = 37 := by
  sorry

#eval minimum_nests 5 3 6 2 1 2 3 4

end park_nests_l3318_331841
