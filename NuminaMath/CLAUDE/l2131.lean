import Mathlib

namespace sum_of_integers_l2131_213194

theorem sum_of_integers (a b : ℕ) (h1 : a > b) (h2 : a - b = 5) (h3 : a * b = 84) : a + b = 19 := by
  sorry

end sum_of_integers_l2131_213194


namespace sin_sum_inverse_sin_tan_l2131_213113

theorem sin_sum_inverse_sin_tan (x y : ℝ) 
  (hx : x = 3 / 5) (hy : y = 1 / 2) : 
  Real.sin (Real.arcsin x + Real.arctan y) = 2 * Real.sqrt 5 / 5 := by
  sorry

end sin_sum_inverse_sin_tan_l2131_213113


namespace map_width_l2131_213191

/-- Given a rectangular map with area 10 square meters and length 5 meters, prove its width is 2 meters. -/
theorem map_width (area : ℝ) (length : ℝ) (width : ℝ) 
    (h_area : area = 10) 
    (h_length : length = 5) 
    (h_rectangle : area = length * width) : width = 2 := by
  sorry

end map_width_l2131_213191


namespace rent_ratio_increase_l2131_213170

/-- The ratio of rent spent this year compared to last year, given changes in income and rent percentage --/
theorem rent_ratio_increase (last_year_rent_percent : ℝ) (income_increase_percent : ℝ) (this_year_rent_percent : ℝ) :
  last_year_rent_percent = 0.20 →
  income_increase_percent = 0.15 →
  this_year_rent_percent = 0.25 →
  (this_year_rent_percent * (1 + income_increase_percent)) / last_year_rent_percent = 1.4375 := by
  sorry

end rent_ratio_increase_l2131_213170


namespace drama_club_neither_math_nor_physics_l2131_213102

theorem drama_club_neither_math_nor_physics 
  (total : ℕ) 
  (math : ℕ) 
  (physics : ℕ) 
  (both : ℕ) 
  (h1 : total = 80) 
  (h2 : math = 50) 
  (h3 : physics = 40) 
  (h4 : both = 25) : 
  total - (math + physics - both) = 15 := by
  sorry

end drama_club_neither_math_nor_physics_l2131_213102


namespace problem_statement_l2131_213129

theorem problem_statement :
  (∀ x : ℝ, (2/3 ≤ x ∧ x ≤ 2) → (-1 ≤ x ∧ x ≤ 2)) ∧
  (∃ x : ℝ, (-1 ≤ x ∧ x ≤ 2) ∧ ¬(2/3 ≤ x ∧ x ≤ 2)) ∧
  (∀ a : ℝ, (∀ x : ℝ, (x ≤ a ∨ x ≥ a + 1) → (2/3 ≤ x ∧ x ≤ 2)) ↔ (a ≥ 2 ∨ a ≤ -1/3)) :=
by sorry


end problem_statement_l2131_213129


namespace heartsuit_problem_l2131_213128

def heartsuit (a b : ℝ) : ℝ := |a + b|

theorem heartsuit_problem : heartsuit (-3) (heartsuit 5 (-8)) = 0 := by
  sorry

end heartsuit_problem_l2131_213128


namespace smallest_gcd_bc_l2131_213195

theorem smallest_gcd_bc (a b c : ℕ+) (hab : Nat.gcd a b = 210) (hac : Nat.gcd a c = 770) :
  (∀ d : ℕ+, ∃ a' b' c' : ℕ+, Nat.gcd a' b' = 210 ∧ Nat.gcd a' c' = 770 ∧ Nat.gcd b' c' = d) →
  10 ≤ Nat.gcd b c :=
sorry

end smallest_gcd_bc_l2131_213195


namespace cos_sum_eleventh_pi_l2131_213126

open Complex

theorem cos_sum_eleventh_pi : 
  cos (π / 11) + cos (3 * π / 11) + cos (7 * π / 11) + cos (9 * π / 11) = -1 / 2 := by
  sorry

end cos_sum_eleventh_pi_l2131_213126


namespace blood_expires_same_day_l2131_213159

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- The factorial of 7 -/
def blood_expiration_seconds : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

/-- Proposition: Blood donated at noon expires on the same day -/
theorem blood_expires_same_day : blood_expiration_seconds < seconds_per_day := by
  sorry

#eval blood_expiration_seconds
#eval seconds_per_day

end blood_expires_same_day_l2131_213159


namespace total_rice_weight_l2131_213182

-- Define the number of containers
def num_containers : ℕ := 4

-- Define the weight of rice in each container (in ounces)
def rice_per_container : ℝ := 29

-- Define the conversion rate from ounces to pounds
def ounces_per_pound : ℝ := 16

-- State the theorem
theorem total_rice_weight :
  (num_containers : ℝ) * rice_per_container / ounces_per_pound = 7.25 := by
  sorry

end total_rice_weight_l2131_213182


namespace congruence_existence_no_solution_for_6_8_solution_exists_for_7_9_l2131_213110

theorem congruence_existence (A B : ℕ) : Prop :=
  ∃ C : ℕ, C % A = 1 ∧ C % B = 2

theorem no_solution_for_6_8 : ¬(congruence_existence 6 8) := by sorry

theorem solution_exists_for_7_9 : congruence_existence 7 9 := by sorry

end congruence_existence_no_solution_for_6_8_solution_exists_for_7_9_l2131_213110


namespace collinear_points_m_value_l2131_213173

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points_m_value :
  ∀ m : ℝ, collinear (-2) 12 1 3 m (-6) → m = 4 := by
  sorry

end collinear_points_m_value_l2131_213173


namespace m_range_l2131_213140

def p (m : ℝ) : Prop := ∀ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x > m

def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m*x + 1 ≤ 0

theorem m_range (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m = -2 ∨ m ≥ 2 := by sorry

end m_range_l2131_213140


namespace speed_change_problem_l2131_213148

theorem speed_change_problem :
  ∃! (x : ℝ), x > 0 ∧
  (1 - x / 100) * (1 + 0.5 * x / 100) = 1 - 0.6 * x / 100 ∧
  ∀ (V : ℝ), V > 0 →
    V * (1 - x / 100) * (1 + 0.5 * x / 100) = V * (1 - 0.6 * x / 100) :=
by sorry

end speed_change_problem_l2131_213148


namespace frank_bakes_for_five_days_l2131_213146

-- Define the problem parameters
def cookies_per_tray : ℕ := 12
def trays_per_day : ℕ := 2
def frank_eats_per_day : ℕ := 1
def ted_eats_last_day : ℕ := 4
def cookies_left : ℕ := 134

-- Define the function to calculate the number of days
def days_baking (cookies_per_tray trays_per_day frank_eats_per_day ted_eats_last_day cookies_left : ℕ) : ℕ :=
  (cookies_left + ted_eats_last_day) / (cookies_per_tray * trays_per_day - frank_eats_per_day)

-- Theorem statement
theorem frank_bakes_for_five_days :
  days_baking cookies_per_tray trays_per_day frank_eats_per_day ted_eats_last_day cookies_left = 5 :=
sorry

end frank_bakes_for_five_days_l2131_213146


namespace arrangement_remainder_l2131_213144

/-- The number of green marbles -/
def green_marbles : ℕ := 5

/-- The maximum number of blue marbles that satisfies the arrangement condition -/
def max_blue_marbles : ℕ := 15

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + max_blue_marbles

/-- The number of ways to arrange the marbles satisfying the condition -/
def arrangement_count : ℕ := (Nat.choose (max_blue_marbles + green_marbles) green_marbles)

/-- Theorem stating that the remainder when dividing the number of arrangements by 1000 is 3 -/
theorem arrangement_remainder : arrangement_count % 1000 = 3 := by sorry

end arrangement_remainder_l2131_213144


namespace basketball_lineup_combinations_l2131_213172

def total_players : ℕ := 18
def quintuplets : ℕ := 5
def lineup_size : ℕ := 7
def quintuplets_in_lineup : ℕ := 2

theorem basketball_lineup_combinations :
  (Nat.choose quintuplets quintuplets_in_lineup) *
  (Nat.choose (total_players - quintuplets) (lineup_size - quintuplets_in_lineup)) = 12870 := by
  sorry

end basketball_lineup_combinations_l2131_213172


namespace equal_triangle_areas_l2131_213135

-- Define the points
variable (A B C D E F K L : Point)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the intersection points E and F
def E_is_intersection (A B C D E : Point) : Prop := sorry
def F_is_intersection (A B C D F : Point) : Prop := sorry

-- Define K and L as midpoints of diagonals
def K_is_midpoint (A C K : Point) : Prop := sorry
def L_is_midpoint (B D L : Point) : Prop := sorry

-- Define the area of a triangle
def area (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem equal_triangle_areas 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : E_is_intersection A B C D E)
  (h3 : F_is_intersection A B C D F)
  (h4 : K_is_midpoint A C K)
  (h5 : L_is_midpoint B D L) :
  area E K L = area F K L := by sorry

end equal_triangle_areas_l2131_213135


namespace algebraic_expression_value_l2131_213119

theorem algebraic_expression_value (x : ℝ) : 
  2 * x^2 + 3 * x + 7 = 8 → 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end algebraic_expression_value_l2131_213119


namespace giraffe_contest_minimum_voters_l2131_213174

structure VotingSystem where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat

def minimum_voters_to_win (vs : VotingSystem) : Nat :=
  2 * ((vs.num_districts + 1) / 2) * ((vs.sections_per_district + 1) / 2)

theorem giraffe_contest_minimum_voters 
  (vs : VotingSystem)
  (h1 : vs.total_voters = 105)
  (h2 : vs.num_districts = 5)
  (h3 : vs.sections_per_district = 7)
  (h4 : vs.voters_per_section = 3)
  (h5 : vs.total_voters = vs.num_districts * vs.sections_per_district * vs.voters_per_section) :
  minimum_voters_to_win vs = 24 := by
  sorry

#eval minimum_voters_to_win ⟨105, 5, 7, 3⟩

end giraffe_contest_minimum_voters_l2131_213174


namespace at_most_one_triangle_l2131_213131

/-- Represents a city in Euleria -/
def City : Type := Fin 101

/-- Represents an airline in Euleria -/
def Airline : Type := Fin 99

/-- Represents a flight between two cities operated by an airline -/
def Flight : Type := City × City × Airline

/-- The set of all flights in Euleria -/
def AllFlights : Set Flight := sorry

/-- A function that returns the airline operating a flight between two cities -/
def flightOperator : City → City → Airline := sorry

/-- A predicate that checks if three cities form a triangle -/
def isTriangle (a b c : City) : Prop :=
  flightOperator a b = flightOperator b c ∧ flightOperator b c = flightOperator c a

/-- The main theorem stating that there is at most one triangle in Euleria -/
theorem at_most_one_triangle :
  ∀ a b c d e f : City,
    isTriangle a b c → isTriangle d e f → a = d ∧ b = e ∧ c = f := by sorry

end at_most_one_triangle_l2131_213131


namespace marble_bowls_theorem_l2131_213125

theorem marble_bowls_theorem (capacity_ratio : Rat) (second_bowl_marbles : Nat) : 
  capacity_ratio = 3/4 → second_bowl_marbles = 600 →
  capacity_ratio * second_bowl_marbles + second_bowl_marbles = 1050 := by
  sorry

end marble_bowls_theorem_l2131_213125


namespace sqrt_difference_equals_seven_sqrt_two_over_six_l2131_213123

theorem sqrt_difference_equals_seven_sqrt_two_over_six :
  Real.sqrt (9 / 2) - Real.sqrt (2 / 9) = 7 * Real.sqrt 2 / 6 := by
  sorry

end sqrt_difference_equals_seven_sqrt_two_over_six_l2131_213123


namespace equation_solution_set_l2131_213171

theorem equation_solution_set : ∀ (x y : ℝ), 
  ((x = 1 ∧ y = 3/2) ∨ 
   (x = 1 ∧ y = -1/2) ∨ 
   (x = -1 ∧ y = 3/2) ∨ 
   (x = -1 ∧ y = -1/2)) ↔ 
  4 * x^2 * y^2 = 4 * x * y + 3 :=
by sorry

end equation_solution_set_l2131_213171


namespace amp_specific_value_l2131_213138

/-- The operation & defined for real numbers -/
def amp (a b c d : ℝ) : ℝ := b^2 - 4*a*c + d

/-- Theorem stating that &(2, -3, 1, 5) = 6 -/
theorem amp_specific_value : amp 2 (-3) 1 5 = 6 := by
  sorry

end amp_specific_value_l2131_213138


namespace wednesday_earnings_l2131_213168

/-- Represents the types of lawns --/
inductive LawnType
| Small
| Medium
| Large

/-- Represents the charge rates for different lawn types --/
def charge_rate (lt : LawnType) : ℕ :=
  match lt with
  | LawnType.Small => 5
  | LawnType.Medium => 7
  | LawnType.Large => 10

/-- Extra fee for lawns with large piles of leaves --/
def large_pile_fee : ℕ := 3

/-- Calculates the earnings for a given day --/
def daily_earnings (small_bags medium_bags large_bags large_piles : ℕ) : ℕ :=
  small_bags * charge_rate LawnType.Small +
  medium_bags * charge_rate LawnType.Medium +
  large_bags * charge_rate LawnType.Large +
  large_piles * large_pile_fee

/-- Represents the work done on Monday --/
def monday_work : ℕ := daily_earnings 4 2 1 1

/-- Represents the work done on Tuesday --/
def tuesday_work : ℕ := daily_earnings 2 1 2 1

/-- Total earnings after three days --/
def total_earnings : ℕ := 163

/-- Theorem stating that Wednesday's earnings are $76 --/
theorem wednesday_earnings :
  total_earnings - (monday_work + tuesday_work) = 76 := by sorry

end wednesday_earnings_l2131_213168


namespace trigonometric_ratio_equals_three_fourths_trigonometric_expression_equals_negative_four_l2131_213109

def α : Real := sorry
def n : ℤ := sorry

-- Part 1
theorem trigonometric_ratio_equals_three_fourths 
  (h1 : Real.cos α = -4/5) 
  (h2 : Real.sin α = 3/5) : 
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = 3/4 := by sorry

-- Part 2
theorem trigonometric_expression_equals_negative_four
  (h1 : Real.cos (π + α) = -1/2)
  (h2 : α > 3*π/2 ∧ α < 2*π) :
  (Real.sin (α + (2*n + 1)*π) + Real.sin (α - (2*n + 1)*π)) / 
  (Real.sin (α + 2*n*π) * Real.cos (α - 2*n*π)) = -4 := by sorry

end trigonometric_ratio_equals_three_fourths_trigonometric_expression_equals_negative_four_l2131_213109


namespace woman_work_time_l2131_213187

-- Define the work rate of one man
def man_rate : ℚ := 1 / 100

-- Define the total work (1 unit)
def total_work : ℚ := 1

-- Define the time taken by 10 men and 15 women
def combined_time : ℚ := 5

-- Define the number of men and women
def num_men : ℕ := 10
def num_women : ℕ := 15

-- Define the work rate of one woman
noncomputable def woman_rate : ℚ := 
  (total_work / combined_time - num_men * man_rate) / num_women

-- Theorem: One woman alone will take 150 days to complete the work
theorem woman_work_time : total_work / woman_rate = 150 := by sorry

end woman_work_time_l2131_213187


namespace arithmetic_sequence_problem_l2131_213179

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a)
  (h_a1 : a 1 = 2)
  (h_sum : a 3 + a 5 = 10) :
  a 7 = 8 := by
sorry

end arithmetic_sequence_problem_l2131_213179


namespace value_of_a_l2131_213124

theorem value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 15 - 7 * a) : a = 15 / 11 := by
  sorry

end value_of_a_l2131_213124


namespace remainder_8547_mod_9_l2131_213192

theorem remainder_8547_mod_9 : 8547 % 9 = 6 := by
  sorry

end remainder_8547_mod_9_l2131_213192


namespace arithmetic_sequence_problem_l2131_213164

theorem arithmetic_sequence_problem (a b c : ℝ) : 
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →  -- arithmetic sequence condition
  a + b + c = 9 →                       -- sum condition
  a * b = 6 * c →                       -- product condition
  a = 4 ∧ b = 3 ∧ c = 2 := by           -- conclusion
sorry

end arithmetic_sequence_problem_l2131_213164


namespace sum_odd_integers_13_to_41_l2131_213100

/-- The sum of odd integers from 13 to 41, inclusive -/
def sumOddIntegers : ℕ :=
  let first := 13
  let last := 41
  let n := (last - first) / 2 + 1
  n * (first + last) / 2

theorem sum_odd_integers_13_to_41 :
  sumOddIntegers = 405 := by
  sorry

end sum_odd_integers_13_to_41_l2131_213100


namespace box_area_is_679_l2131_213151

/-- The surface area of the interior of a box formed by removing square corners from a rectangular sheet --/
def box_interior_area (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- Theorem stating that the surface area of the interior of the box is 679 square units --/
theorem box_area_is_679 :
  box_interior_area 25 35 7 = 679 :=
by sorry

end box_area_is_679_l2131_213151


namespace turtle_problem_l2131_213157

theorem turtle_problem (initial_turtles : ℕ) (h1 : initial_turtles = 9) : 
  let new_turtles := 3 * initial_turtles - 2
  let total_turtles := initial_turtles + new_turtles
  let remaining_turtles := total_turtles / 2
  remaining_turtles = 17 := by
sorry

end turtle_problem_l2131_213157


namespace cube_inequality_l2131_213169

theorem cube_inequality (n : ℕ+) : (n + 1)^3 ≠ n^3 + (n - 1)^3 := by
  sorry

end cube_inequality_l2131_213169


namespace ellipse_theorems_l2131_213117

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    and focal length 2√3, prove the following theorems. -/
theorem ellipse_theorems 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_focal : a^2 - b^2 = 3) :
  let C : ℝ × ℝ → Prop := λ p => p.1^2 / 4 + p.2^2 = 1
  ∃ (k : ℝ) (h_k : k ≠ 0),
    let l₁ : ℝ → ℝ := λ x => k * x
    ∃ (A B : ℝ × ℝ) (h_AB : A.2 = l₁ A.1 ∧ B.2 = l₁ B.1),
      let l₂ : ℝ → ℝ := λ x => (B.2 + k/4 * (x - B.1))
      ∃ (D : ℝ × ℝ) (h_D : D.2 = l₂ D.1),
        (A.1 - D.1) * (A.1 - B.1) + (A.2 - D.2) * (A.2 - B.2) = 0 →
        (∀ p : ℝ × ℝ, C p ↔ p.1^2 / 4 + p.2^2 = 1) ∧
        (∃ (M N : ℝ × ℝ), 
          M.2 = 0 ∧ N.1 = 0 ∧ M.2 = l₂ M.1 ∧ N.2 = l₂ N.1 ∧
          ∀ (M' N' : ℝ × ℝ), M'.2 = 0 ∧ N'.1 = 0 ∧ M'.2 = l₂ M'.1 ∧ N'.2 = l₂ N'.1 →
          abs (M.1 * N.2) / 2 ≥ abs (M'.1 * N'.2) / 2 ∧
          abs (M.1 * N.2) / 2 = 9/8) := by
  sorry


end ellipse_theorems_l2131_213117


namespace linear_function_problem_l2131_213103

/-- A linear function passing through (1, 3) -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x + 1

/-- The linear function shifted up by 2 units -/
def g (k : ℝ) (x : ℝ) : ℝ := f k x + 2

theorem linear_function_problem (k : ℝ) (h : k ≠ 0) (h1 : f k 1 = 3) :
  k = 2 ∧ ∀ x, g k x = 2 * x + 3 := by
  sorry

end linear_function_problem_l2131_213103


namespace complex_simplification_l2131_213147

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- The statement to prove -/
theorem complex_simplification : 7 * (2 - 3 * i) + 4 * i * (3 - 2 * i) = 22 - 9 * i := by
  sorry

end complex_simplification_l2131_213147


namespace park_not_crowded_implies_cool_or_rain_l2131_213155

variable (day : Type) -- Type representing days

-- Define predicates for weather conditions and park status
variable (temp_at_least_70 : day → Prop) -- Temperature is at least 70°F
variable (raining : day → Prop) -- It is raining
variable (crowded : day → Prop) -- The park is crowded

-- Given condition: If temp ≥ 70°F and not raining, then the park is crowded
variable (h : ∀ d : day, (temp_at_least_70 d ∧ ¬raining d) → crowded d)

theorem park_not_crowded_implies_cool_or_rain :
  ∀ d : day, ¬crowded d → (¬temp_at_least_70 d ∨ raining d) :=
by
  sorry

#check park_not_crowded_implies_cool_or_rain

end park_not_crowded_implies_cool_or_rain_l2131_213155


namespace inequality_proof_l2131_213188

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c := by
  sorry

end inequality_proof_l2131_213188


namespace image_equality_under_composition_condition_l2131_213185

universe u

theorem image_equality_under_composition_condition 
  {S : Type u} [Finite S] (f : S → S) :
  (∀ (g : S → S), g ≠ f → (f ∘ g ∘ f) ≠ (g ∘ f ∘ g)) →
  let T := Set.range f
  f '' T = T := by
  sorry

end image_equality_under_composition_condition_l2131_213185


namespace apple_vendor_problem_l2131_213136

theorem apple_vendor_problem (initial_apples : ℝ) (h_initial_positive : initial_apples > 0) :
  let first_day_sold := 0.6 * initial_apples
  let first_day_remainder := initial_apples - first_day_sold
  let x := (23 * initial_apples - 0.5 * first_day_remainder) / (0.5 * first_day_remainder)
  x = 0.15
  := by sorry

end apple_vendor_problem_l2131_213136


namespace payment_plan_difference_l2131_213184

theorem payment_plan_difference (original_price down_payment num_payments payment_amount : ℕ) :
  original_price = 1500 ∧
  down_payment = 200 ∧
  num_payments = 24 ∧
  payment_amount = 65 →
  (down_payment + num_payments * payment_amount) - original_price = 260 := by
  sorry

end payment_plan_difference_l2131_213184


namespace bridge_crossing_time_l2131_213150

/-- Proves that a man walking at 9 km/hr takes 15 minutes to cross a bridge of 2250 meters in length -/
theorem bridge_crossing_time (walking_speed : ℝ) (bridge_length : ℝ) :
  walking_speed = 9 →
  bridge_length = 2250 →
  (bridge_length / (walking_speed * 1000 / 60)) = 15 :=
by
  sorry

end bridge_crossing_time_l2131_213150


namespace fraction_enlargement_l2131_213154

theorem fraction_enlargement (x y : ℝ) (h : x ≠ y) :
  (3 * x) * (3 * y) / ((3 * x) - (3 * y)) = 3 * (x * y / (x - y)) := by
  sorry

end fraction_enlargement_l2131_213154


namespace closest_ratio_to_one_l2131_213104

/-- Represents the number of adults and children attending an exhibition -/
structure Attendance where
  adults : ℕ
  children : ℕ

/-- Calculates the total admission fee for a given attendance -/
def totalFee (a : Attendance) : ℕ :=
  25 * a.adults + 15 * a.children

/-- Checks if the ratio of adults to children is closer to 1 than the given ratio -/
def isCloserToOne (a : Attendance) (ratio : Rat) : Prop :=
  |1 - (a.adults : ℚ) / a.children| < |1 - ratio|

/-- The main theorem stating the closest ratio to 1 -/
theorem closest_ratio_to_one :
  ∃ (a : Attendance),
    a.adults > 0 ∧
    a.children > 0 ∧
    totalFee a = 1950 ∧
    a.adults = 48 ∧
    a.children = 50 ∧
    ∀ (b : Attendance),
      b.adults > 0 →
      b.children > 0 →
      totalFee b = 1950 →
      b ≠ a →
      isCloserToOne a (24 / 25) :=
sorry

end closest_ratio_to_one_l2131_213104


namespace min_value_sum_fractions_min_value_sum_fractions_achieved_l2131_213158

theorem min_value_sum_fractions (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c) / d + (a + b + d) / c + (a + c + d) / b + (b + c + d) / a ≥ 12 :=
by sorry

theorem min_value_sum_fractions_achieved (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ∃ (x : ℝ), x > 0 ∧ 
    (x + x + x) / x + (x + x + x) / x + (x + x + x) / x + (x + x + x) / x = 12 :=
by sorry

end min_value_sum_fractions_min_value_sum_fractions_achieved_l2131_213158


namespace desired_interest_rate_l2131_213190

/-- Calculate the desired interest rate (dividend yield) for a share -/
theorem desired_interest_rate (face_value : ℝ) (dividend_rate : ℝ) (market_value : ℝ) :
  face_value = 48 →
  dividend_rate = 0.09 →
  market_value = 36.00000000000001 →
  (face_value * dividend_rate) / market_value * 100 = 12 := by
  sorry

end desired_interest_rate_l2131_213190


namespace relationship_abc_l2131_213141

theorem relationship_abc : 
  let a := Real.sqrt 2 / 2 * (Real.sin (17 * π / 180) + Real.cos (17 * π / 180))
  let b := 2 * (Real.cos (13 * π / 180))^2 - 1
  let c := Real.sqrt 3 / 2
  c < a ∧ a < b := by sorry

end relationship_abc_l2131_213141


namespace curvilinear_trapezoid_area_l2131_213178

-- Define the function representing the parabola
def f (x : ℝ) : ℝ := 9 - x^2

-- Define the integral bounds
def a : ℝ := -1
def b : ℝ := 2

-- State the theorem
theorem curvilinear_trapezoid_area : 
  ∫ x in a..b, f x = 24 := by sorry

end curvilinear_trapezoid_area_l2131_213178


namespace book_width_calculation_l2131_213189

theorem book_width_calculation (length width area : ℝ) : 
  length = 2 → area = 6 → area = length * width → width = 3 := by
  sorry

end book_width_calculation_l2131_213189


namespace hyperbola_eccentricity_l2131_213114

/-- A hyperbola with a focus on the y-axis and asymptotic lines y = ± (√5/2)x has eccentricity 3√5/5 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (2 * a = Real.sqrt 5 * b) →
  (Real.sqrt ((a^2 + b^2) / a^2) = 3 * Real.sqrt 5 / 5) :=
by sorry

end hyperbola_eccentricity_l2131_213114


namespace family_ages_solution_l2131_213145

/-- Represents the ages of Priya and her parents -/
structure FamilyAges where
  priya : ℕ
  father : ℕ
  mother : ℕ

/-- Conditions for the family ages problem -/
def FamilyAgesProblem (ages : FamilyAges) : Prop :=
  ages.father - ages.priya = 31 ∧
  ages.father + 8 + ages.priya + 8 = 69 ∧
  ages.father - ages.mother = 4 ∧
  ages.priya + 5 + ages.mother + 5 = 65

/-- Theorem stating the solution to the family ages problem -/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), FamilyAgesProblem ages ∧ 
    ages.priya = 11 ∧ ages.father = 42 ∧ ages.mother = 38 := by
  sorry

end family_ages_solution_l2131_213145


namespace parallel_condition_l2131_213180

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + a * y + 3 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := a * x + 4 * y + 6 = 0

-- Define the parallel relation between two lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- State the theorem
theorem parallel_condition (a : ℝ) :
  (parallel (l₁ a) (l₂ a) → (a = 2 ∨ a = -2)) ∧
  ¬(a = 2 ∨ a = -2 → parallel (l₁ a) (l₂ a)) :=
sorry

end parallel_condition_l2131_213180


namespace circular_garden_area_l2131_213121

/-- Proves that a circular garden with radius 6 and fence length equal to 1/3 of its area has an area of 36π square units -/
theorem circular_garden_area (r : ℝ) (h1 : r = 6) : 
  (2 * Real.pi * r = (1/3) * Real.pi * r^2) → Real.pi * r^2 = 36 * Real.pi := by
  sorry

end circular_garden_area_l2131_213121


namespace quadratic_vertex_l2131_213116

/-- The vertex of a quadratic function -/
theorem quadratic_vertex
  (a k c d : ℝ)
  (ha : a > 0)
  (hk : k ≠ b)  -- Note: 'b' is not defined, but kept as per the original problem
  (f : ℝ → ℝ)
  (hf : f = fun x ↦ a * x^2 + k * x + c + d) :
  let x₀ := -k / (2 * a)
  ∃ y₀, (x₀, y₀) = (-k / (2 * a), -k^2 / (4 * a) + c + d) ∧ 
       ∀ x, f x ≥ f x₀ :=
by sorry

end quadratic_vertex_l2131_213116


namespace joe_weight_lifting_problem_l2131_213193

theorem joe_weight_lifting_problem (total_weight first_lift_weight : ℕ) 
  (h1 : total_weight = 900)
  (h2 : first_lift_weight = 400) : 
  2 * first_lift_weight - (total_weight - first_lift_weight) = 300 := by
  sorry

end joe_weight_lifting_problem_l2131_213193


namespace polynomial_evaluation_and_subtraction_l2131_213139

theorem polynomial_evaluation_and_subtraction :
  let x : ℝ := 2
  20 - 2 * (3 * x^2 - 4 * x + 8) = -4 := by
  sorry

end polynomial_evaluation_and_subtraction_l2131_213139


namespace prob_reroll_two_dice_l2131_213137

/-- The number of possible outcomes when rolling three fair six-sided dice -/
def total_outcomes : ℕ := 6^3

/-- The number of ways to get a sum of 8 when rolling three fair six-sided dice -/
def sum_eight_outcomes : ℕ := 20

/-- The probability that the sum of three fair six-sided dice is not equal to 8 -/
def prob_not_eight : ℚ := (total_outcomes - sum_eight_outcomes) / total_outcomes

theorem prob_reroll_two_dice : prob_not_eight = 49 / 54 := by
  sorry

end prob_reroll_two_dice_l2131_213137


namespace quadratic_no_real_roots_l2131_213134

theorem quadratic_no_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 1 ≠ 0) → -2 < a ∧ a < 2 :=
by sorry

end quadratic_no_real_roots_l2131_213134


namespace solution_set_f_gt_3_l2131_213176

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x else Real.log x + 2

theorem solution_set_f_gt_3 :
  {x : ℝ | f x > 3} = {x : ℝ | x < -3 ∨ x > Real.exp 1} := by sorry

end solution_set_f_gt_3_l2131_213176


namespace dormitory_students_count_l2131_213163

theorem dormitory_students_count :
  ∃ (x y : ℕ),
    x > 0 ∧
    y > 0 ∧
    x * (x - 1) + x * y + y = 51 ∧
    x = 6 := by
  sorry

end dormitory_students_count_l2131_213163


namespace percent_to_decimal_five_percent_to_decimal_l2131_213122

theorem percent_to_decimal (p : ℝ) : p / 100 = p * 0.01 := by sorry

theorem five_percent_to_decimal : (5 : ℝ) / 100 = 0.05 := by sorry

end percent_to_decimal_five_percent_to_decimal_l2131_213122


namespace amount_r_has_l2131_213186

theorem amount_r_has (total : ℝ) (r_fraction : ℝ) (h1 : total = 4000) (h2 : r_fraction = 2/3) : 
  let amount_pq := total / (1 + r_fraction)
  let amount_r := r_fraction * amount_pq
  amount_r = 1600 := by sorry

end amount_r_has_l2131_213186


namespace cos_angle_relation_l2131_213196

theorem cos_angle_relation (α : ℝ) (h : Real.cos (α + π/3) = 4/5) :
  Real.cos (π/3 - 2*α) = -7/25 := by
  sorry

end cos_angle_relation_l2131_213196


namespace distinct_prime_factors_count_l2131_213133

theorem distinct_prime_factors_count : 
  (Finset.card (Nat.factors (85 * 87 * 91 * 94)).toFinset) = 8 := by
  sorry

end distinct_prime_factors_count_l2131_213133


namespace horses_meet_on_day_9_l2131_213167

/-- Represents the day on which the horses meet --/
def meetingDay : ℕ := 9

/-- The distance between Chang'an and Qi in li --/
def totalDistance : ℚ := 1125

/-- The initial distance covered by the good horse on the first day --/
def goodHorseInitial : ℚ := 103

/-- The daily increase in distance for the good horse --/
def goodHorseIncrease : ℚ := 13

/-- The initial distance covered by the mediocre horse on the first day --/
def mediocreHorseInitial : ℚ := 97

/-- The daily decrease in distance for the mediocre horse --/
def mediocreHorseDecrease : ℚ := 1/2

/-- Theorem stating that the horses meet on the 9th day --/
theorem horses_meet_on_day_9 :
  (meetingDay : ℚ) * (goodHorseInitial + mediocreHorseInitial) +
  (meetingDay * (meetingDay - 1) / 2) * (goodHorseIncrease - mediocreHorseDecrease) =
  2 * totalDistance := by
  sorry

#check horses_meet_on_day_9

end horses_meet_on_day_9_l2131_213167


namespace intersection_M_N_l2131_213127

def M : Set ℝ := {x : ℝ | |x + 1| ≤ 1}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0} := by sorry

end intersection_M_N_l2131_213127


namespace smallest_vector_norm_l2131_213143

open Vector

theorem smallest_vector_norm (v : ℝ × ℝ) (h : ‖v + (-2, 4)‖ = 10) :
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ u : ℝ × ℝ, ‖u + (-2, 4)‖ = 10 → ‖w‖ ≤ ‖u‖ := by
  sorry

end smallest_vector_norm_l2131_213143


namespace multiple_solutions_exist_l2131_213118

/-- Represents the number of wheels on a vehicle -/
inductive VehicleType
  | twoWheeler
  | fourWheeler

/-- Calculates the number of wheels for a given vehicle type -/
def wheelCount (v : VehicleType) : Nat :=
  match v with
  | .twoWheeler => 2
  | .fourWheeler => 4

/-- Represents a parking configuration -/
structure ParkingConfig where
  twoWheelers : Nat
  fourWheelers : Nat

/-- Calculates the total number of wheels for a given parking configuration -/
def totalWheels (config : ParkingConfig) : Nat :=
  config.twoWheelers * wheelCount VehicleType.twoWheeler +
  config.fourWheelers * wheelCount VehicleType.fourWheeler

/-- Theorem stating that multiple solutions exist for the parking problem -/
theorem multiple_solutions_exist :
  ∃ (config1 config2 : ParkingConfig),
    totalWheels config1 = 70 ∧
    totalWheels config2 = 70 ∧
    config1.fourWheelers ≠ config2.fourWheelers :=
by
  sorry

#check multiple_solutions_exist

end multiple_solutions_exist_l2131_213118


namespace square_field_area_l2131_213108

/-- The area of a square field with side length 20 meters is 400 square meters. -/
theorem square_field_area (side_length : ℝ) (h : side_length = 20) : 
  side_length * side_length = 400 := by
  sorry

end square_field_area_l2131_213108


namespace sets_intersection_empty_l2131_213120

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x < 2}

theorem sets_intersection_empty : A ∩ B = ∅ := by sorry

end sets_intersection_empty_l2131_213120


namespace unique_virtual_square_plus_one_l2131_213175

def is_virtual (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * b + 10 * a + b

def is_square_plus_one (n : ℕ) : Prop :=
  ∃ (m : ℕ), n = m^2 + 1

theorem unique_virtual_square_plus_one :
  ∃! (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ is_virtual n ∧ is_square_plus_one n ∧ n = 8282 :=
sorry

end unique_virtual_square_plus_one_l2131_213175


namespace juans_number_l2131_213112

theorem juans_number (x : ℝ) : ((3 * (x + 3) - 4) / 2 = 10) → x = 5 := by
  sorry

end juans_number_l2131_213112


namespace constant_term_is_160_l2131_213166

/-- The constant term in the binomial expansion of (x + 2/x)^6 -/
def constant_term : ℕ :=
  (Nat.choose 6 3) * (2^3)

/-- Theorem: The constant term in the binomial expansion of (x + 2/x)^6 is 160 -/
theorem constant_term_is_160 : constant_term = 160 := by
  sorry

end constant_term_is_160_l2131_213166


namespace expression_evaluation_l2131_213183

theorem expression_evaluation (x y : ℚ) (hx : x = 4/3) (hy : y = 5/8) :
  (6*x + 8*y) / (48*x*y) = 13/40 := by
  sorry

end expression_evaluation_l2131_213183


namespace probability_two_heads_in_three_flips_l2131_213197

/-- The probability of getting exactly k successes in n trials with probability p for each trial -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- A fair coin has probability 0.5 of landing heads -/
def fair_coin_probability : ℝ := 0.5

/-- The number of flips -/
def num_flips : ℕ := 3

/-- The number of heads we want -/
def num_heads : ℕ := 2

theorem probability_two_heads_in_three_flips :
  binomial_probability num_flips num_heads fair_coin_probability = 0.375 := by
  sorry

end probability_two_heads_in_three_flips_l2131_213197


namespace reflection_of_M_l2131_213142

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (- p.1, p.2)

/-- The original point M -/
def M : ℝ × ℝ := (-5, 2)

theorem reflection_of_M :
  reflect_y M = (5, 2) := by sorry

end reflection_of_M_l2131_213142


namespace largest_two_digit_prime_factor_of_binom_300_150_l2131_213198

/-- The largest 2-digit prime factor of (300 choose 150) -/
def largest_two_digit_prime_factor_of_binom : ℕ := 97

/-- The binomial coefficient (300 choose 150) -/
def binom_300_150 : ℕ := Nat.choose 300 150

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  largest_two_digit_prime_factor_of_binom = 97 ∧
  Nat.Prime largest_two_digit_prime_factor_of_binom ∧
  largest_two_digit_prime_factor_of_binom ≥ 10 ∧
  largest_two_digit_prime_factor_of_binom < 100 ∧
  (binom_300_150 % largest_two_digit_prime_factor_of_binom = 0) ∧
  ∀ p : ℕ, Nat.Prime p → p ≥ 10 → p < 100 → 
    (binom_300_150 % p = 0) → p ≤ largest_two_digit_prime_factor_of_binom :=
by sorry

end largest_two_digit_prime_factor_of_binom_300_150_l2131_213198


namespace complete_square_sum_l2131_213152

theorem complete_square_sum (x : ℝ) : 
  (∃ (a b c : ℤ), a > 0 ∧ 
   64 * x^2 + 96 * x - 128 = 0 ↔ (a * x + b)^2 = c) →
  (∃ (a b c : ℤ), a > 0 ∧ 
   64 * x^2 + 96 * x - 128 = 0 ↔ (a * x + b)^2 = c ∧
   a + b + c = 178) := by
sorry

end complete_square_sum_l2131_213152


namespace f_shifted_is_even_f_has_three_zeros_l2131_213107

/-- A function that satisfies the given conditions -/
def f (x : ℝ) : ℝ := (x - 1)^2 - |x - 1|

/-- Theorem stating that f(x+1) is an even function on ℝ -/
theorem f_shifted_is_even : ∀ x : ℝ, f (x + 1) = f (-x + 1) := by sorry

/-- Theorem stating that f(x) has exactly three zeros on ℝ -/
theorem f_has_three_zeros : ∃! (a b c : ℝ), a < b ∧ b < c ∧ 
  (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) := by sorry

end f_shifted_is_even_f_has_three_zeros_l2131_213107


namespace cylinder_height_l2131_213115

/-- The height of a cylinder given its lateral surface area and volume -/
theorem cylinder_height (r h : ℝ) (h1 : 2 * π * r * h = 12 * π) (h2 : π * r^2 * h = 12 * π) : h = 3 := by
  sorry

end cylinder_height_l2131_213115


namespace complement_A_in_U_l2131_213162

-- Define the universal set U
def U : Set ℝ := {x : ℝ | x ≤ 1}

-- Define set A
def A : Set ℝ := {x : ℝ | x < 0}

-- Theorem statement
theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end complement_A_in_U_l2131_213162


namespace hyperbola_properties_l2131_213149

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := y^2 / a^2 - x^2 / 3 = 1

-- Define eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Define asymptotes
def asymptotes (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x

-- Define the trajectory of midpoint M
def trajectory (x y : ℝ) : Prop := x^2 / 75 + 3 * y^2 / 25 = 1

-- Define the theorem
theorem hyperbola_properties :
  ∀ (a : ℝ), 
  (∃ (x y : ℝ), hyperbola x y a) →
  eccentricity 2 →
  (∀ (x y : ℝ), asymptotes x y) ∧
  (∀ (x y : ℝ), trajectory x y) :=
sorry

end hyperbola_properties_l2131_213149


namespace trigonometric_identities_l2131_213160

theorem trigonometric_identities : 
  (2 * Real.sin (67.5 * π / 180) * Real.cos (67.5 * π / 180) = Real.sqrt 2 / 2) ∧
  (1 - 2 * (Real.sin (22.5 * π / 180))^2 = Real.sqrt 2 / 2) := by
  sorry

end trigonometric_identities_l2131_213160


namespace hit_rate_problem_l2131_213177

theorem hit_rate_problem (p_at_least_one p_a p_b : ℝ) : 
  p_at_least_one = 0.7 →
  p_a = 0.4 →
  p_at_least_one = p_a + p_b - p_a * p_b →
  p_b = 0.5 := by
  sorry

end hit_rate_problem_l2131_213177


namespace sum_of_four_cubes_l2131_213105

theorem sum_of_four_cubes (k : ℤ) : ∃ (a b c d : ℤ), 24 * k = a^3 + b^3 + c^3 + d^3 := by
  sorry

end sum_of_four_cubes_l2131_213105


namespace min_value_trig_expression_l2131_213106

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 18)^2 ≥ 169 := by
  sorry

end min_value_trig_expression_l2131_213106


namespace allen_pizza_payment_l2131_213101

theorem allen_pizza_payment (num_boxes : ℕ) (cost_per_box : ℚ) (tip_fraction : ℚ) (change_received : ℚ) :
  num_boxes = 5 →
  cost_per_box = 7 →
  tip_fraction = 1 / 7 →
  change_received = 60 →
  let total_cost := num_boxes * cost_per_box
  let tip := tip_fraction * total_cost
  let total_paid := total_cost + tip
  let money_given := total_paid + change_received
  money_given = 100 := by
  sorry

end allen_pizza_payment_l2131_213101


namespace smallest_percentage_for_90_percent_l2131_213132

/-- Represents the distribution of money in a population -/
structure MoneyDistribution where
  /-- Percentage of people owning the majority of money -/
  rich_percentage : ℝ
  /-- Percentage of money owned by the rich -/
  rich_money_percentage : ℝ
  /-- Percentage of people needed to own a target percentage of money -/
  target_percentage : ℝ
  /-- Target percentage of money to be owned -/
  target_money_percentage : ℝ

/-- Theorem stating the smallest percentage of people that can be guaranteed to own 90% of all money -/
theorem smallest_percentage_for_90_percent 
  (d : MoneyDistribution) 
  (h1 : d.rich_percentage = 20)
  (h2 : d.rich_money_percentage ≥ 80)
  (h3 : d.target_money_percentage = 90) :
  d.target_percentage = 60 := by
  sorry

end smallest_percentage_for_90_percent_l2131_213132


namespace optimal_distribution_l2131_213199

/-- Represents the profit distribution problem for a fruit distributor. -/
structure FruitDistribution where
  /-- Total number of boxes of each fruit type -/
  total_boxes : ℕ
  /-- Profit per box of A fruit at Store A -/
  profit_a_store_a : ℕ
  /-- Profit per box of B fruit at Store A -/
  profit_b_store_a : ℕ
  /-- Profit per box of A fruit at Store B -/
  profit_a_store_b : ℕ
  /-- Profit per box of B fruit at Store B -/
  profit_b_store_b : ℕ
  /-- Minimum profit required for Store B -/
  min_profit_store_b : ℕ

/-- Theorem stating the optimal distribution and maximum profit -/
theorem optimal_distribution (fd : FruitDistribution)
  (h1 : fd.total_boxes = 10)
  (h2 : fd.profit_a_store_a = 11)
  (h3 : fd.profit_b_store_a = 17)
  (h4 : fd.profit_a_store_b = 9)
  (h5 : fd.profit_b_store_b = 13)
  (h6 : fd.min_profit_store_b = 115) :
  ∃ (a_store_a b_store_a a_store_b b_store_b : ℕ),
    a_store_a + b_store_a = fd.total_boxes ∧
    a_store_b + b_store_b = fd.total_boxes ∧
    a_store_a + a_store_b = fd.total_boxes ∧
    b_store_a + b_store_b = fd.total_boxes ∧
    fd.profit_a_store_b * a_store_b + fd.profit_b_store_b * b_store_b ≥ fd.min_profit_store_b ∧
    fd.profit_a_store_a * a_store_a + fd.profit_b_store_a * b_store_a +
    fd.profit_a_store_b * a_store_b + fd.profit_b_store_b * b_store_b = 246 ∧
    a_store_a = 7 ∧ b_store_a = 3 ∧ a_store_b = 3 ∧ b_store_b = 7 ∧
    ∀ (x y z w : ℕ),
      x + y = fd.total_boxes →
      z + w = fd.total_boxes →
      x + z = fd.total_boxes →
      y + w = fd.total_boxes →
      fd.profit_a_store_b * z + fd.profit_b_store_b * w ≥ fd.min_profit_store_b →
      fd.profit_a_store_a * x + fd.profit_b_store_a * y +
      fd.profit_a_store_b * z + fd.profit_b_store_b * w ≤ 246 :=
by sorry


end optimal_distribution_l2131_213199


namespace symmetric_points_sum_l2131_213153

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite in sign but equal in magnitude. -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

/-- Given two points M(a,3) and N(4,b) symmetric about the x-axis,
    prove that a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_about_x_axis (a, 3) (4, b)) : a + b = 1 := by
  sorry


end symmetric_points_sum_l2131_213153


namespace trig_expression_value_l2131_213165

theorem trig_expression_value (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16 / 5 := by
  sorry

end trig_expression_value_l2131_213165


namespace platinum_sphere_weight_in_mercury_l2131_213111

/-- The weight of a platinum sphere in mercury at elevated temperature -/
theorem platinum_sphere_weight_in_mercury
  (p : ℝ)
  (d₁ : ℝ)
  (d₂ : ℝ)
  (a₁ : ℝ)
  (a₂ : ℝ)
  (h_p : p = 30)
  (h_d₁ : d₁ = 21.5)
  (h_d₂ : d₂ = 13.60)
  (h_a₁ : a₁ = 0.0000264)
  (h_a₂ : a₂ = 0.0001815)
  : ∃ w : ℝ, abs (w - 11.310) < 0.001 :=
by
  sorry


end platinum_sphere_weight_in_mercury_l2131_213111


namespace odd_periodic2_sum_zero_l2131_213156

/-- A function that is odd and has a period of 2 -/
def OddPeriodic2 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = f x)

/-- Theorem: For any odd function with period 2, f(1) + f(4) + f(7) = 0 -/
theorem odd_periodic2_sum_zero (f : ℝ → ℝ) (h : OddPeriodic2 f) :
  f 1 + f 4 + f 7 = 0 := by
  sorry


end odd_periodic2_sum_zero_l2131_213156


namespace find_z_when_y_is_4_l2131_213130

-- Define the relationship between y and z
def inverse_variation (y z : ℝ) (k : ℝ) : Prop :=
  y^3 * z^(1/3) = k

-- Theorem statement
theorem find_z_when_y_is_4 (y z : ℝ) (k : ℝ) :
  inverse_variation 2 1 k →
  inverse_variation 4 z k →
  z = 1 / 512 :=
by
  sorry

end find_z_when_y_is_4_l2131_213130


namespace consecutive_odd_integers_sum_l2131_213181

theorem consecutive_odd_integers_sum (n : ℤ) : 
  (n + (n + 4) = 130) → (n + (n + 2) + (n + 4) = 195) := by
  sorry

end consecutive_odd_integers_sum_l2131_213181


namespace initial_speed_calculation_l2131_213161

/-- Proves that the initial speed of a person traveling a distance D in time T is 160/3 kmph -/
theorem initial_speed_calculation (D T : ℝ) (h1 : D > 0) (h2 : T > 0) : ∃ S : ℝ,
  (2 / 3 * D) / (1 / 3 * T) = S ∧
  (1 / 3 * D) / 40 = 2 / 3 * T ∧
  S = 160 / 3 := by
  sorry

end initial_speed_calculation_l2131_213161
