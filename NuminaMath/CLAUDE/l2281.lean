import Mathlib

namespace ellipse_second_focus_x_coordinate_l2281_228114

-- Define the ellipse properties
structure Ellipse where
  inFirstQuadrant : Bool
  tangentToXAxis : Bool
  tangentToYAxis : Bool
  focus1 : ℝ × ℝ
  tangentToY1 : Bool

-- Define the theorem
theorem ellipse_second_focus_x_coordinate
  (e : Ellipse)
  (h1 : e.inFirstQuadrant = true)
  (h2 : e.tangentToXAxis = true)
  (h3 : e.tangentToYAxis = true)
  (h4 : e.focus1 = (4, 9))
  (h5 : e.tangentToY1 = true) :
  ∃ d : ℝ, d = 16 ∧ (∃ y : ℝ, (d, y) = e.focus1 ∨ (d, 9) ≠ e.focus1) :=
sorry

end ellipse_second_focus_x_coordinate_l2281_228114


namespace pentagonal_field_fencing_cost_l2281_228107

/-- Represents the cost of fencing for a pentagonal field -/
def fencing_cost (side1 side2 side3 side4 side5 rate_a rate_b rate_c : ℝ) : ℝ × ℝ × ℝ :=
  let perimeter := side1 + side2 + side3 + side4 + side5
  (perimeter * rate_a, perimeter * rate_b, perimeter * rate_c)

/-- Theorem stating the correct fencing costs for the given pentagonal field -/
theorem pentagonal_field_fencing_cost :
  fencing_cost 25 35 40 45 50 3.5 2.25 1.5 = (682.5, 438.75, 292.5) := by
  sorry

end pentagonal_field_fencing_cost_l2281_228107


namespace collinear_points_imply_a_eq_two_l2281_228188

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Given three points A(a,2), B(5,1), and C(-4,2a) are collinear, prove that a = 2. -/
theorem collinear_points_imply_a_eq_two (a : ℝ) :
  collinear a 2 5 1 (-4) (2*a) → a = 2 := by
  sorry

end collinear_points_imply_a_eq_two_l2281_228188


namespace oranges_problem_l2281_228141

def oranges_left (mary jason tom sarah : ℕ) : ℕ :=
  let initial_total := mary + jason + tom + sarah
  let increased_total := (initial_total * 110 + 50) / 100  -- Rounded up
  (increased_total * 85 + 50) / 100  -- Rounded down

theorem oranges_problem (mary jason tom sarah : ℕ) 
  (h_mary : mary = 122)
  (h_jason : jason = 105)
  (h_tom : tom = 85)
  (h_sarah : sarah = 134) :
  oranges_left mary jason tom sarah = 417 := by
  sorry

end oranges_problem_l2281_228141


namespace legs_minus_twice_heads_diff_l2281_228145

/-- Represents the number of legs for each animal type -/
def legs_per_animal : Nat → Nat
| 0 => 2  -- Chicken
| 1 => 4  -- Cow
| _ => 0  -- Other animals (not used in this problem)

/-- Calculates the total number of legs in the group -/
def total_legs (num_chickens num_cows : Nat) : Nat :=
  legs_per_animal 0 * num_chickens + legs_per_animal 1 * num_cows

/-- Calculates the total number of heads in the group -/
def total_heads (num_chickens num_cows : Nat) : Nat :=
  num_chickens + num_cows

/-- The main theorem stating the difference between legs and twice the heads -/
theorem legs_minus_twice_heads_diff (num_chickens : Nat) : 
  total_legs num_chickens 7 - 2 * total_heads num_chickens 7 = 14 := by
  sorry

#check legs_minus_twice_heads_diff

end legs_minus_twice_heads_diff_l2281_228145


namespace max_two_wins_l2281_228102

/-- Represents a single-elimination tournament --/
structure Tournament :=
  (participants : ℕ)

/-- Represents the number of participants who won exactly two matches --/
def exactlyTwoWins (t : Tournament) : ℕ := sorry

/-- The theorem stating the maximum number of participants who can win exactly two matches --/
theorem max_two_wins (t : Tournament) (h : t.participants = 100) : 
  exactlyTwoWins t ≤ 49 ∧ ∃ (strategy : Unit), exactlyTwoWins t = 49 := by sorry

end max_two_wins_l2281_228102


namespace seventh_term_is_28_l2281_228176

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℝ
  -- Common difference of the sequence
  d : ℝ
  -- Sum of first three terms is 9
  sum_first_three : a + (a + d) + (a + 2 * d) = 9
  -- Third term is 8
  third_term : a + 2 * d = 8

/-- The seventh term of the arithmetic sequence is 28 -/
theorem seventh_term_is_28 (seq : ArithmeticSequence) : 
  seq.a + 6 * seq.d = 28 := by
sorry

end seventh_term_is_28_l2281_228176


namespace tom_not_in_middle_seat_l2281_228197

-- Define the people
inductive Person : Type
| Andy : Person
| Jen : Person
| Sally : Person
| Mike : Person
| Tom : Person

-- Define a seating arrangement as a function from seat number to person
def Seating := Fin 5 → Person

-- Andy is not beside Jen
def AndyNotBesideJen (s : Seating) : Prop :=
  ∀ i : Fin 4, s i ≠ Person.Andy ∨ s i.succ ≠ Person.Jen

-- Sally is beside Mike
def SallyBesideMike (s : Seating) : Prop :=
  ∃ i : Fin 4, (s i = Person.Sally ∧ s i.succ = Person.Mike) ∨
               (s i = Person.Mike ∧ s i.succ = Person.Sally)

-- The middle seat is the third seat
def MiddleSeat : Fin 5 := ⟨2, by norm_num⟩

-- Theorem: Tom cannot sit in the middle seat
theorem tom_not_in_middle_seat :
  ∀ s : Seating, AndyNotBesideJen s → SallyBesideMike s →
  s MiddleSeat ≠ Person.Tom :=
by sorry

end tom_not_in_middle_seat_l2281_228197


namespace binomial_coefficient_congruence_l2281_228186

theorem binomial_coefficient_congruence (n p : ℕ) (h_prime : Nat.Prime p) (h_n_gt_p : n > p) :
  (n.choose p) ≡ (n / p : ℕ) [MOD p] := by
  sorry

end binomial_coefficient_congruence_l2281_228186


namespace potato_slab_length_difference_l2281_228100

theorem potato_slab_length_difference (total_length first_piece_length : ℕ) 
  (h1 : total_length = 600)
  (h2 : first_piece_length = 275) :
  total_length - first_piece_length - first_piece_length = 50 :=
by sorry

end potato_slab_length_difference_l2281_228100


namespace problem_1_problem_2_problem_3_problem_4_l2281_228116

-- Problem 1
theorem problem_1 : -4.7 + 0.9 = -3.8 := by sorry

-- Problem 2
theorem problem_2 : -1/2 - (-1/3) = -1/6 := by sorry

-- Problem 3
theorem problem_3 : (-1 - 1/9) * (-0.6) = 2/3 := by sorry

-- Problem 4
theorem problem_4 : 0 * (-5) = 0 := by sorry

end problem_1_problem_2_problem_3_problem_4_l2281_228116


namespace billion_yuan_eq_scientific_notation_l2281_228147

/-- Represents the value in billions of yuan -/
def billion_yuan : ℝ := 98.36

/-- Represents the same value in scientific notation -/
def scientific_notation : ℝ := 9.836 * (10 ^ 9)

/-- Theorem stating that the billion yuan value is equal to its scientific notation -/
theorem billion_yuan_eq_scientific_notation : billion_yuan * (10 ^ 9) = scientific_notation := by
  sorry

end billion_yuan_eq_scientific_notation_l2281_228147


namespace largest_solution_reciprocal_power_l2281_228193

noncomputable def largest_solution (x : ℝ) : Prop :=
  (Real.log 5 / Real.log (5 * x^2) + Real.log 5 / Real.log (25 * x^3) = -1) ∧
  ∀ y, (Real.log 5 / Real.log (5 * y^2) + Real.log 5 / Real.log (25 * y^3) = -1) → y ≤ x

theorem largest_solution_reciprocal_power (x : ℝ) :
  largest_solution x → 1 / x^10 = 0.00001 :=
by sorry

end largest_solution_reciprocal_power_l2281_228193


namespace omega_sequence_monotone_increasing_l2281_228183

/-- Definition of an Ω sequence -/
def is_omega_sequence (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, (a n + a (n + 2)) / 2 ≤ a (n + 1)) ∧
  (∃ M : ℝ, ∀ n : ℕ+, a n ≤ M)

/-- Theorem: For any Ω sequence of positive integers, each term is less than or equal to the next term -/
theorem omega_sequence_monotone_increasing
  (d : ℕ+ → ℕ+)
  (h_omega : is_omega_sequence (λ n => (d n : ℝ))) :
  ∀ n : ℕ+, d n ≤ d (n + 1) := by
sorry

end omega_sequence_monotone_increasing_l2281_228183


namespace maintenance_scheduling_methods_l2281_228166

/-- Represents the days of the week --/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Represents the monitoring points --/
inductive MonitoringPoint
  | A
  | B
  | C
  | D
  | E
  | F
  | G
  | H

/-- A schedule is a function from MonitoringPoint to Day --/
def Schedule := MonitoringPoint → Day

/-- Checks if a schedule is valid according to the given conditions --/
def isValidSchedule (s : Schedule) : Prop :=
  (s MonitoringPoint.A = Day.Monday) ∧
  (s MonitoringPoint.B = Day.Tuesday) ∧
  (s MonitoringPoint.C = s MonitoringPoint.D) ∧
  (s MonitoringPoint.D = s MonitoringPoint.E) ∧
  (s MonitoringPoint.F ≠ Day.Friday) ∧
  (∀ d : Day, ∃ p : MonitoringPoint, s p = d)

/-- The total number of valid schedules --/
def totalValidSchedules : ℕ := sorry

theorem maintenance_scheduling_methods :
  totalValidSchedules = 60 := by sorry

end maintenance_scheduling_methods_l2281_228166


namespace smallest_z_minus_x_is_444_l2281_228158

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_z_minus_x_is_444 :
  ∃ (x y z : ℕ+),
    (x.val * y.val * z.val = factorial 10) ∧
    (x < y) ∧ (y < z) ∧
    (∀ (a b c : ℕ+),
      (a.val * b.val * c.val = factorial 10) → (a < b) → (b < c) →
      ((z.val - x.val : ℤ) ≤ (c.val - a.val))) ∧
    (z.val - x.val = 444) :=
by sorry

end smallest_z_minus_x_is_444_l2281_228158


namespace circle_center_fourth_quadrant_l2281_228142

/-- Given a real number a, if the equation x^2 + y^2 - 2ax + 4ay + 6a^2 - a = 0 
    represents a circle with its center in the fourth quadrant, then 0 < a < 1. -/
theorem circle_center_fourth_quadrant (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x + 4*a*y + 6*a^2 - a = 0 → 
    ∃ r : ℝ, r > 0 ∧ ∀ x' y' : ℝ, (x' - a)^2 + (y' + 2*a)^2 = r^2) →
  (a > 0 ∧ -2*a < 0) →
  0 < a ∧ a < 1 := by
sorry

end circle_center_fourth_quadrant_l2281_228142


namespace boys_meeting_time_l2281_228144

/-- Two boys running on a circular track meet after a specific time -/
theorem boys_meeting_time (track_length : Real) (speed1 speed2 : Real) :
  track_length = 4800 ∧ 
  speed1 = 60 * (1000 / 3600) ∧ 
  speed2 = 100 * (1000 / 3600) →
  track_length / (speed1 + speed2) = 108 := by
  sorry

end boys_meeting_time_l2281_228144


namespace arctan_equation_solution_l2281_228120

theorem arctan_equation_solution :
  ∀ x : ℝ, 3 * Real.arctan (1/4) + Real.arctan (1/5) + Real.arctan (1/x) = π/4 → x = -250/37 := by
  sorry

end arctan_equation_solution_l2281_228120


namespace initial_cloth_length_l2281_228149

/-- Given that 4 men can colour an initial length of cloth in 2 days,
    and 8 men can colour 36 meters of cloth in 0.75 days,
    prove that the initial length of cloth is 48 meters. -/
theorem initial_cloth_length (initial_length : ℝ) : 
  (4 * initial_length / 2 = 8 * 36 / 0.75) → initial_length = 48 := by
  sorry

end initial_cloth_length_l2281_228149


namespace abs_neg_two_equals_two_l2281_228138

theorem abs_neg_two_equals_two : abs (-2 : ℝ) = 2 := by
  sorry

end abs_neg_two_equals_two_l2281_228138


namespace sum_of_divisors_360_l2281_228151

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (·∣n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_360 : sum_of_divisors 360 = 1170 := by
  sorry

end sum_of_divisors_360_l2281_228151


namespace min_a_for_probability_half_or_more_l2281_228169

/-- Represents a deck of cards numbered from 1 to 60 -/
def Deck := Finset (Fin 60)

/-- Represents the probability function p(a,b) -/
noncomputable def p (a b : ℕ) : ℚ :=
  let remaining_cards := 58
  let total_ways := Nat.choose remaining_cards 2
  let lower_team_ways := Nat.choose (a - 1) 2
  let higher_team_ways := Nat.choose (48 - a) 2
  (lower_team_ways + higher_team_ways : ℚ) / total_ways

/-- The main theorem to prove -/
theorem min_a_for_probability_half_or_more (deck : Deck) :
  (∀ a < 13, p a (a + 10) < 1/2) ∧ 
  p 13 23 = 473/551 ∧
  p 13 23 ≥ 1/2 := by
  sorry


end min_a_for_probability_half_or_more_l2281_228169


namespace box_sum_equals_sixteen_l2281_228119

def box (a b c : ℤ) : ℚ := (a ^ b : ℚ) + (b ^ c : ℚ) - (c ^ a : ℚ)

theorem box_sum_equals_sixteen : box 2 3 (-1) + box (-1) 2 3 = 16 := by
  sorry

end box_sum_equals_sixteen_l2281_228119


namespace power_of_three_mod_thousand_l2281_228108

theorem power_of_three_mod_thousand :
  ∃ n : ℕ, n < 1000 ∧ 3^5000 ≡ n [MOD 1000] :=
sorry

end power_of_three_mod_thousand_l2281_228108


namespace problem_solution_l2281_228179

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x - 1) + x

def g (x : ℝ) : ℝ := x - 1

def h (m : ℝ) (f' : ℝ → ℝ) (x : ℝ) : ℝ := m * f' x + g x + 1

theorem problem_solution (a : ℝ) :
  (∀ x, deriv (f a) x = a / (x - 1) + 1) →
  deriv (f a) 2 = 2 →
  (a = 1 ∧
   (∀ x, g x = x - 1) ∧
   (∀ m, (∀ x ∈ Set.Icc 2 4, h m (deriv (f a)) x > 0) → m > -1 ∧ ∀ y > -1, ∃ x ∈ Set.Icc 2 4, h y (deriv (f a)) x > 0)) :=
by sorry

end problem_solution_l2281_228179


namespace min_value_sum_reciprocals_l2281_228159

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → 1/x + 4/y ≥ 1/a + 4/b) →
  1/a + 4/b = 9/2 :=
by sorry

end min_value_sum_reciprocals_l2281_228159


namespace verna_sherry_combined_weight_l2281_228132

def haley_weight : ℕ := 103
def verna_weight : ℕ := haley_weight + 17

theorem verna_sherry_combined_weight : 
  ∃ (sherry_weight : ℕ), 
    verna_weight = haley_weight + 17 ∧ 
    verna_weight * 2 = sherry_weight ∧ 
    verna_weight + sherry_weight = 360 :=
by
  sorry

end verna_sherry_combined_weight_l2281_228132


namespace max_product_sum_l2281_228143

theorem max_product_sum (X Y Z : ℕ) (sum_constraint : X + Y + Z = 15) :
  (∀ a b c : ℕ, a + b + c = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ a * b * c + a * b + b * c + c * a) ∧
  X * Y * Z + X * Y + Y * Z + Z * X = 200 :=
sorry

end max_product_sum_l2281_228143


namespace min_value_of_function_equality_condition_min_value_exists_l2281_228182

theorem min_value_of_function (x : ℝ) (h : x > 0) : 2 + 4*x + 1/x ≥ 6 :=
sorry

theorem equality_condition (x : ℝ) (h : x > 0) : 2 + 4*x + 1/x = 6 ↔ x = 1/2 :=
sorry

theorem min_value_exists : ∃ x : ℝ, x > 0 ∧ 2 + 4*x + 1/x = 6 :=
sorry

end min_value_of_function_equality_condition_min_value_exists_l2281_228182


namespace smallest_prime_with_digit_sum_19_l2281_228178

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_19 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 19 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 19 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_19_l2281_228178


namespace factor_expression_l2281_228174

theorem factor_expression (x : ℝ) : x * (x - 4) + 2 * (x - 4) = (x + 2) * (x - 4) := by
  sorry

end factor_expression_l2281_228174


namespace f_unbounded_above_l2281_228199

/-- The function f(x, y) = 2x^2 + 4xy + 5y^2 + 8x - 6y -/
def f (x y : ℝ) : ℝ := 2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y

/-- Theorem: The function f is unbounded above -/
theorem f_unbounded_above : ∀ M : ℝ, ∃ x y : ℝ, f x y > M := by
  sorry

end f_unbounded_above_l2281_228199


namespace cookie_count_l2281_228104

/-- The number of cookies Paul and Paula bought together -/
def total_cookies (paul_cookies paula_cookies : ℕ) : ℕ :=
  paul_cookies + paula_cookies

/-- Theorem: Paul and Paula bought 87 cookies in total -/
theorem cookie_count : ∃ (paula_cookies : ℕ),
  (paula_cookies = 45 - 3) ∧ (total_cookies 45 paula_cookies = 87) :=
by
  sorry

end cookie_count_l2281_228104


namespace x_eq_2_sufficient_not_necessary_l2281_228110

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem x_eq_2_sufficient_not_necessary :
  let a : ℝ → ℝ × ℝ := λ x ↦ (1, x)
  let b : ℝ → ℝ × ℝ := λ x ↦ (x, 4)
  (∀ x, x = 2 → parallel (a x) (b x)) ∧
  ¬(∀ x, parallel (a x) (b x) → x = 2) :=
by sorry

end x_eq_2_sufficient_not_necessary_l2281_228110


namespace complex_magnitude_l2281_228171

theorem complex_magnitude (z : ℂ) : z = (2 - I) / (1 + I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_magnitude_l2281_228171


namespace equation_solution_l2281_228121

theorem equation_solution :
  ∃ (x : ℝ), x ≠ 2 ∧ x ≠ -2 ∧ (x / (x - 2) + 2 / (x^2 - 4) = 1) ∧ x = -3 :=
by sorry

end equation_solution_l2281_228121


namespace uncool_parents_count_l2281_228157

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total = 40)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 20)
  (h4 : both_cool = 10) :
  total - (cool_dads + cool_moms - both_cool) = 12 := by
  sorry

#check uncool_parents_count

end uncool_parents_count_l2281_228157


namespace anthony_jim_difference_l2281_228129

/-- The number of pairs of shoes Scott has -/
def scott_shoes : ℕ := 7

/-- The number of pairs of shoes Anthony has -/
def anthony_shoes : ℕ := 3 * scott_shoes

/-- The number of pairs of shoes Jim has -/
def jim_shoes : ℕ := anthony_shoes - 2

/-- Theorem: Anthony has 2 more pairs of shoes than Jim -/
theorem anthony_jim_difference : anthony_shoes - jim_shoes = 2 := by
  sorry

end anthony_jim_difference_l2281_228129


namespace mixture_volume_proof_l2281_228136

/-- Proves that the initial volume of a mixture is 150 liters, given the conditions of the problem -/
theorem mixture_volume_proof (initial_water_percentage : Real) 
                              (added_water : Real) 
                              (final_water_percentage : Real) : 
  initial_water_percentage = 0.1 →
  added_water = 30 →
  final_water_percentage = 0.25 →
  ∃ (initial_volume : Real),
    initial_volume * initial_water_percentage + added_water = 
    (initial_volume + added_water) * final_water_percentage ∧
    initial_volume = 150 := by
  sorry


end mixture_volume_proof_l2281_228136


namespace molecular_weight_15_C2H5Cl_12_O2_l2281_228122

/-- Calculates the molecular weight of a given number of moles of C2H5Cl and O2 -/
def molecularWeight (moles_C2H5Cl : ℝ) (moles_O2 : ℝ) : ℝ :=
  let atomic_weight_C := 12.01
  let atomic_weight_H := 1.01
  let atomic_weight_Cl := 35.45
  let atomic_weight_O := 16.00
  let mw_C2H5Cl := 2 * atomic_weight_C + 5 * atomic_weight_H + atomic_weight_Cl
  let mw_O2 := 2 * atomic_weight_O
  moles_C2H5Cl * mw_C2H5Cl + moles_O2 * mw_O2

theorem molecular_weight_15_C2H5Cl_12_O2 :
  molecularWeight 15 12 = 1351.8 := by
  sorry

end molecular_weight_15_C2H5Cl_12_O2_l2281_228122


namespace min_value_problem_l2281_228175

theorem min_value_problem (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + b^2 = 4) (h2 : c * d = 1) :
  (a^2 * c^2 + b^2 * d^2) * (b^2 * c^2 + a^2 * d^2) ≥ 16 := by
  sorry

end min_value_problem_l2281_228175


namespace three_lines_intersection_l2281_228196

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The three lines in the problem -/
def line1 : Line := ⟨1, 1, 1⟩
def line2 : Line := ⟨2, -1, 8⟩
def line3 (a : ℝ) : Line := ⟨a, 3, -5⟩

/-- The theorem to be proved -/
theorem three_lines_intersection (a : ℝ) : 
  (parallel line1 (line3 a) ∨ parallel line2 (line3 a)) → 
  a = 3 ∨ a = -6 :=
sorry

end three_lines_intersection_l2281_228196


namespace function_properties_l2281_228152

def f (a b x : ℝ) : ℝ := x^2 - (a + 1) * x + b

theorem function_properties (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ -5 < x ∧ x < 2) →
  (a = -4 ∧ b = -10) ∧
  (∀ x, f a a x > 0 ↔
    (a > 1 ∧ (x < 1 ∨ x > a)) ∨
    (a = 1 ∧ x ≠ 1) ∨
    (a < 1 ∧ (x < a ∨ x > 1))) := by
  sorry

end function_properties_l2281_228152


namespace rotate90_of_4_minus_2i_l2281_228191

def rotate90 (z : ℂ) : ℂ := z * Complex.I

theorem rotate90_of_4_minus_2i : 
  rotate90 (4 - 2 * Complex.I) = 2 + 4 * Complex.I :=
by sorry

end rotate90_of_4_minus_2i_l2281_228191


namespace max_consecutive_sum_less_than_500_l2281_228172

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- 31 is the largest positive integer n such that the sum of the first n positive integers is less than 500 -/
theorem max_consecutive_sum_less_than_500 :
  ∀ n : ℕ, sum_first_n n < 500 ↔ n ≤ 31 :=
by sorry

end max_consecutive_sum_less_than_500_l2281_228172


namespace expand_expression_l2281_228130

theorem expand_expression (x : ℝ) : (5 * x^2 + 3) * 4 * x^3 = 20 * x^5 + 12 * x^3 := by
  sorry

end expand_expression_l2281_228130


namespace polynomial_division_l2281_228173

def p (x : ℝ) : ℝ := x^5 + 3*x^4 - 28*x^3 + 45*x^2 - 58*x + 24
def d (x : ℝ) : ℝ := x - 3
def q (x : ℝ) : ℝ := x^4 + 6*x^3 - 10*x^2 + 15*x - 13
def r : ℝ := -15

theorem polynomial_division :
  ∀ x : ℝ, p x = d x * q x + r :=
by
  sorry

end polynomial_division_l2281_228173


namespace salary_decrease_percentage_l2281_228195

def original_salary : ℝ := 4000.0000000000005
def initial_increase_rate : ℝ := 0.1
def final_salary : ℝ := 4180

theorem salary_decrease_percentage :
  ∃ (decrease_rate : ℝ),
    final_salary = original_salary * (1 + initial_increase_rate) * (1 - decrease_rate) ∧
    decrease_rate = 0.05 := by
  sorry

end salary_decrease_percentage_l2281_228195


namespace dans_grocery_items_l2281_228156

/-- Represents the items bought at the grocery store -/
structure GroceryItems where
  eggs : ℕ
  flour : ℕ
  butter : ℕ
  vanilla : ℕ

/-- Calculates the total number of individual items -/
def totalItems (items : GroceryItems) : ℕ :=
  items.eggs + items.flour + items.butter + items.vanilla

/-- Theorem stating the total number of items Dan bought -/
theorem dans_grocery_items : ∃ (items : GroceryItems), 
  items.eggs = 9 * 12 ∧ 
  items.flour = 6 ∧ 
  items.butter = 3 * 24 ∧ 
  items.vanilla = 12 ∧ 
  totalItems items = 198 := by
  sorry


end dans_grocery_items_l2281_228156


namespace max_stickers_for_player_l2281_228137

theorem max_stickers_for_player (num_players : ℕ) (avg_stickers : ℕ) (min_stickers : ℕ) :
  num_players = 25 →
  avg_stickers = 4 →
  min_stickers = 1 →
  ∃ (max_stickers : ℕ), max_stickers = 76 ∧
    ∀ (player_stickers : ℕ),
      (player_stickers * num_players ≤ num_players * avg_stickers) ∧
      (∀ (i : ℕ), i < num_players → min_stickers ≤ player_stickers) →
      player_stickers ≤ max_stickers :=
by sorry

end max_stickers_for_player_l2281_228137


namespace unique_three_digit_number_l2281_228167

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The conditions for the three-digit number -/
def satisfiesConditions (n : ThreeDigitNumber) : Prop :=
  n.units + n.hundreds = n.tens ∧
  7 * n.hundreds = n.units + n.tens + 2 ∧
  n.units + n.tens + n.hundreds = 14

/-- The theorem stating that 275 is the only three-digit number satisfying the conditions -/
theorem unique_three_digit_number :
  ∃! n : ThreeDigitNumber, satisfiesConditions n ∧ 
    n.hundreds = 2 ∧ n.tens = 7 ∧ n.units = 5 := by
  sorry

end unique_three_digit_number_l2281_228167


namespace two_number_difference_l2281_228135

theorem two_number_difference (a b : ℕ) (h1 : b = 10 * a) (h2 : a + b = 23320) : b - a = 19080 := by
  sorry

end two_number_difference_l2281_228135


namespace sonika_deposit_l2281_228101

theorem sonika_deposit (P R : ℝ) : 
  (P + (P * R * 3) / 100 = 11200) → 
  (P + (P * (R + 2) * 3) / 100 = 11680) → 
  P = 8000 := by
  sorry

end sonika_deposit_l2281_228101


namespace triangle_side_calculation_l2281_228140

theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  a * Real.cos B = b * Real.sin A →
  C = π / 6 →
  c = 2 →
  b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_calculation_l2281_228140


namespace contractor_absent_days_l2281_228163

/-- Proves that given the specified contract conditions, the number of absent days is 8 -/
theorem contractor_absent_days 
  (total_days : ℕ) 
  (daily_pay : ℚ) 
  (daily_fine : ℚ) 
  (total_amount : ℚ) 
  (h1 : total_days = 30)
  (h2 : daily_pay = 25)
  (h3 : daily_fine = 7.5)
  (h4 : total_amount = 490) :
  ∃ (days_absent : ℕ), 
    days_absent = 8 ∧ 
    (daily_pay * (total_days - days_absent) - daily_fine * days_absent = total_amount) :=
by sorry


end contractor_absent_days_l2281_228163


namespace cos_75_cos_15_minus_sin_75_sin_15_eq_zero_l2281_228131

theorem cos_75_cos_15_minus_sin_75_sin_15_eq_zero :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) - 
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 0 := by
  sorry

end cos_75_cos_15_minus_sin_75_sin_15_eq_zero_l2281_228131


namespace odd_sum_of_squares_implies_odd_sum_l2281_228194

theorem odd_sum_of_squares_implies_odd_sum (n m : ℤ) (h : Odd (n^2 + m^2)) : Odd (n + m) := by
  sorry

end odd_sum_of_squares_implies_odd_sum_l2281_228194


namespace correct_average_marks_l2281_228192

theorem correct_average_marks (n : ℕ) (incorrect_avg : ℚ) (incorrect_mark correct_mark : ℚ) :
  n = 10 ∧ incorrect_avg = 100 ∧ incorrect_mark = 50 ∧ correct_mark = 10 →
  (n * incorrect_avg - (incorrect_mark - correct_mark)) / n = 96 := by
  sorry

end correct_average_marks_l2281_228192


namespace fraction_evaluation_l2281_228125

theorem fraction_evaluation : (4 * 3) / (2 + 1) = 4 := by
  sorry

end fraction_evaluation_l2281_228125


namespace trigonometric_identity_l2281_228153

theorem trigonometric_identity (α β : ℝ) 
  (h : Real.cos α ^ 2 * Real.sin β ^ 2 + Real.sin α ^ 2 * Real.cos β ^ 2 = 
       Real.cos α * Real.sin α * Real.cos β * Real.sin β) : 
  (Real.sin β ^ 2 * Real.cos α ^ 2) / Real.sin α ^ 2 + 
  (Real.cos β ^ 2 * Real.sin α ^ 2) / Real.cos α ^ 2 = 1 := by
  sorry

end trigonometric_identity_l2281_228153


namespace angle_equality_l2281_228146

theorem angle_equality (θ : Real) (h1 : Real.cos (60 * π / 180) = Real.cos (45 * π / 180) * Real.cos θ) 
  (h2 : 0 ≤ θ) (h3 : θ ≤ π / 2) : θ = 45 * π / 180 := by
  sorry

end angle_equality_l2281_228146


namespace cube_volume_ratio_l2281_228190

theorem cube_volume_ratio : 
  let cube1_side_length : ℝ := 2  -- in meters
  let cube2_side_length : ℝ := 100 / 100  -- 100 cm converted to meters
  let cube1_volume := cube1_side_length ^ 3
  let cube2_volume := cube2_side_length ^ 3
  cube1_volume / cube2_volume = 8 := by sorry

end cube_volume_ratio_l2281_228190


namespace expression_value_approximation_l2281_228113

theorem expression_value_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |((85 : ℝ) + Real.sqrt 32 / 113) * 113^2 - 10246| < ε :=
sorry

end expression_value_approximation_l2281_228113


namespace charcoal_for_900ml_l2281_228162

/-- Given a ratio of charcoal to water and a volume of water, calculate the amount of charcoal needed. -/
def charcoal_needed (charcoal_ratio : ℚ) (water_volume : ℚ) : ℚ :=
  water_volume / (30 / charcoal_ratio)

/-- Theorem: The amount of charcoal needed for 900 ml of water is 60 grams, given the ratio of 2 grams of charcoal per 30 ml of water. -/
theorem charcoal_for_900ml :
  charcoal_needed 2 900 = 60 := by
  sorry

end charcoal_for_900ml_l2281_228162


namespace farmers_children_count_l2281_228106

/-- Represents the problem of determining the number of farmer's children based on apple collection and consumption. -/
theorem farmers_children_count :
  ∀ (n : ℕ),
  (n * 15 - 8 - 7 = 60) →
  n = 5 :=
by
  sorry

end farmers_children_count_l2281_228106


namespace matrix_power_result_l2281_228123

theorem matrix_power_result (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B.mulVec ![4, -1] = ![12, -3]) :
  (B ^ 4).mulVec ![4, -1] = ![324, -81] := by
  sorry

end matrix_power_result_l2281_228123


namespace billion_to_scientific_notation_l2281_228161

/-- Proves that 850 billion yuan is equal to 8.5 × 10^11 yuan -/
theorem billion_to_scientific_notation :
  let billion : ℝ := 10^9
  850 * billion = 8.5 * 10^11 := by sorry

end billion_to_scientific_notation_l2281_228161


namespace complex_power_2017_l2281_228118

theorem complex_power_2017 : ((1 - Complex.I) / (1 + Complex.I)) ^ 2017 = -Complex.I := by sorry

end complex_power_2017_l2281_228118


namespace four_solutions_l2281_228164

/-- The number of positive integer solutions to the equation 3x + y = 15 -/
def solution_count : Nat :=
  (Finset.filter (fun p : Nat × Nat => 3 * p.1 + p.2 = 15 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 15) (Finset.range 15))).card

/-- Theorem stating that there are exactly 4 pairs of positive integers (x, y) satisfying 3x + y = 15 -/
theorem four_solutions : solution_count = 4 := by
  sorry

end four_solutions_l2281_228164


namespace homework_assignment_question_distribution_l2281_228117

theorem homework_assignment_question_distribution :
  ∃! (x y z : ℕ),
    x + y + z = 100 ∧
    (0.5 : ℝ) * x + 3 * y + 10 * z = 100 ∧
    x = 80 ∧ y = 20 ∧ z = 0 := by
  sorry

end homework_assignment_question_distribution_l2281_228117


namespace sqrt_90000_equals_300_l2281_228105

theorem sqrt_90000_equals_300 : Real.sqrt 90000 = 300 := by
  sorry

end sqrt_90000_equals_300_l2281_228105


namespace work_completion_time_l2281_228133

/-- The number of days it takes for A to complete the work alone -/
def days_A : ℝ := 30

/-- The number of days it takes for B to complete the work alone -/
def days_B : ℝ := 55

/-- The number of days it takes for A and B to complete the work together -/
def days_AB : ℝ := 19.411764705882355

theorem work_completion_time :
  (1 / days_A) + (1 / days_B) = (1 / days_AB) := by sorry

end work_completion_time_l2281_228133


namespace tangent_lines_to_circle_tangent_lines_correct_l2281_228139

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a line is tangent to a circle centered at the origin -/
def isTangentToCircle (l : Line) (r : ℝ) : Prop :=
  (l.a ^ 2 + l.b ^ 2) * r ^ 2 = l.c ^ 2

theorem tangent_lines_to_circle (p : Point) (r : ℝ) :
  p.x ^ 2 + p.y ^ 2 > r ^ 2 →
  (∃ l₁ l₂ : Line,
    (pointOnLine p l₁ ∧ isTangentToCircle l₁ r) ∧
    (pointOnLine p l₂ ∧ isTangentToCircle l₂ r) ∧
    ((l₁.a = 1 ∧ l₁.b = 0 ∧ l₁.c = -p.x) ∨
     (l₂.a = 5 ∧ l₂.b = -12 ∧ l₂.c = 26))) :=
by sorry

/-- The main theorem that proves the given tangent lines are correct -/
theorem tangent_lines_correct : 
  ∃ l₁ l₂ : Line,
    (pointOnLine ⟨2, 3⟩ l₁ ∧ isTangentToCircle l₁ 2) ∧
    (pointOnLine ⟨2, 3⟩ l₂ ∧ isTangentToCircle l₂ 2) ∧
    ((l₁.a = 1 ∧ l₁.b = 0 ∧ l₁.c = -2) ∨
     (l₂.a = 5 ∧ l₂.b = -12 ∧ l₂.c = 26)) :=
by
  apply tangent_lines_to_circle ⟨2, 3⟩ 2
  norm_num

end tangent_lines_to_circle_tangent_lines_correct_l2281_228139


namespace sequence_difference_l2281_228109

def arithmetic_sum (a₁ aₙ : ℤ) (n : ℕ) : ℤ := n * (a₁ + aₙ) / 2

def sequence_1_sum : ℤ := arithmetic_sum 2 2021 674
def sequence_2_sum : ℤ := arithmetic_sum 3 2022 674

theorem sequence_difference : sequence_1_sum - sequence_2_sum = -544 := by
  sorry

end sequence_difference_l2281_228109


namespace sum_of_unit_complex_squares_l2281_228150

theorem sum_of_unit_complex_squares (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = 3) : 
  Complex.abs (a + b + c)^2 = 1 := by
sorry

end sum_of_unit_complex_squares_l2281_228150


namespace smallest_number_divisibility_l2281_228184

theorem smallest_number_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < 255 → (¬(11 ∣ (y + 9)) ∨ ¬(24 ∣ (y + 9)))) ∧ 
  (11 ∣ (255 + 9)) ∧ 
  (24 ∣ (255 + 9)) := by
  sorry

end smallest_number_divisibility_l2281_228184


namespace polynomial_simplification_l2281_228103

variable (p : ℝ)

theorem polynomial_simplification :
  (6 * p^4 + 2 * p^3 - 8 * p + 9) + (-3 * p^3 + 7 * p^2 - 5 * p - 1) =
  6 * p^4 - p^3 + 7 * p^2 - 13 * p + 8 := by
sorry

end polynomial_simplification_l2281_228103


namespace inverse_of_B_squared_l2281_228148

open Matrix

theorem inverse_of_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![1, 4; -2, -7]) : 
  (B^2)⁻¹ = !![(-7), (-24); 12, 41] := by
sorry

end inverse_of_B_squared_l2281_228148


namespace cost_of_500_candies_l2281_228112

def candies_per_box : ℕ := 20
def cost_per_box : ℚ := 8
def discount_percentage : ℚ := 0.1
def discount_threshold : ℕ := 400
def order_size : ℕ := 500

theorem cost_of_500_candies : 
  let boxes_needed : ℕ := order_size / candies_per_box
  let total_cost : ℚ := boxes_needed * cost_per_box
  let discount : ℚ := if order_size > discount_threshold then discount_percentage * total_cost else 0
  let final_cost : ℚ := total_cost - discount
  final_cost = 180 := by sorry

end cost_of_500_candies_l2281_228112


namespace problem_solution_l2281_228128

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 3|
def g (m x : ℝ) : ℝ := m - 2*|x - 11|

-- State the theorem
theorem problem_solution :
  (∀ x m : ℝ, 2 * f x ≥ g m (x + 4)) →
  (∃ t : ℝ, t = 20 ∧ ∀ m : ℝ, (∀ x : ℝ, 2 * f x ≥ g m (x + 4)) → m ≤ t) ∧
  (∀ a : ℝ, a > 0 →
    (∃ x y z : ℝ, 2*x^2 + 3*y^2 + 6*z^2 = a ∧
      ∀ x' y' z' : ℝ, 2*x'^2 + 3*y'^2 + 6*z'^2 = a → x' + y' + z' ≤ 1) →
    a = 1) :=
by sorry

end problem_solution_l2281_228128


namespace prime_power_sum_l2281_228177

theorem prime_power_sum (w x y z : ℕ) :
  3^w * 5^x * 7^y * 11^z = 2310 →
  3*w + 5*x + 7*y + 11*z = 26 := by
sorry

end prime_power_sum_l2281_228177


namespace composition_of_even_is_even_l2281_228124

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : IsEven f) : IsEven (f ∘ f) := by
  sorry

end composition_of_even_is_even_l2281_228124


namespace artist_paintings_l2281_228180

/-- Calculates the number of paintings an artist can make in a given number of weeks -/
def paintings_made (hours_per_week : ℕ) (hours_per_painting : ℕ) (num_weeks : ℕ) : ℕ :=
  (hours_per_week / hours_per_painting) * num_weeks

/-- Proves that an artist spending 30 hours per week painting, taking 3 hours per painting, can make 40 paintings in 4 weeks -/
theorem artist_paintings : paintings_made 30 3 4 = 40 := by
  sorry

end artist_paintings_l2281_228180


namespace mo_tea_consumption_l2281_228111

/-- Represents Mo's drinking habits and weather conditions for a week -/
structure MoDrinkingHabits where
  n : ℕ  -- number of hot chocolate cups on rainy mornings
  t : ℕ  -- number of tea cups on non-rainy mornings
  rainyDays : ℕ
  nonRainyDays : ℕ

/-- Theorem stating Mo's tea consumption on non-rainy mornings -/
theorem mo_tea_consumption (habits : MoDrinkingHabits) : habits.t = 4 :=
  by
  have h1 : habits.rainyDays = 2 := by sorry
  have h2 : habits.nonRainyDays = 7 - habits.rainyDays := by sorry
  have h3 : habits.n * habits.rainyDays + habits.t * habits.nonRainyDays = 26 := by sorry
  have h4 : habits.t * habits.nonRainyDays = habits.n * habits.rainyDays + 14 := by sorry
  sorry

#check mo_tea_consumption

end mo_tea_consumption_l2281_228111


namespace arithmetic_sequence_first_term_l2281_228189

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₂ = -5 and the common difference is 3,
    prove that a₁ = -8 -/
theorem arithmetic_sequence_first_term
  (a : ℕ → ℝ)
  (h_arith : IsArithmeticSequence a)
  (h_a2 : a 2 = -5)
  (h_d : ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  a 1 = -8 := by
  sorry

end arithmetic_sequence_first_term_l2281_228189


namespace sector_central_angle_l2281_228185

/-- The central angle of a sector with radius 1 cm and arc length 2 cm is 2 radians. -/
theorem sector_central_angle (radius : ℝ) (arc_length : ℝ) (central_angle : ℝ) : 
  radius = 1 → arc_length = 2 → arc_length = radius * central_angle → central_angle = 2 := by
  sorry

end sector_central_angle_l2281_228185


namespace minimize_sum_distances_l2281_228126

/-- A structure representing a point on a line --/
structure Point where
  x : ℝ

/-- The distance between two points on a line --/
def distance (p q : Point) : ℝ := abs (p.x - q.x)

/-- The theorem stating that Q₅ minimizes the sum of distances --/
theorem minimize_sum_distances 
  (Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ : Point)
  (h_order : Q₁.x < Q₂.x ∧ Q₂.x < Q₃.x ∧ Q₃.x < Q₄.x ∧ Q₄.x < Q₅.x ∧ 
             Q₅.x < Q₆.x ∧ Q₆.x < Q₇.x ∧ Q₇.x < Q₈.x ∧ Q₈.x < Q₉.x)
  (h_fixed : Q₁.x ≠ Q₉.x)
  (h_not_midpoint : Q₅.x ≠ (Q₁.x + Q₉.x) / 2) :
  ∀ Q : Point, 
    distance Q Q₁ + distance Q Q₂ + distance Q Q₃ + distance Q Q₄ + 
    distance Q Q₅ + distance Q Q₆ + distance Q Q₇ + distance Q Q₈ + 
    distance Q Q₉ 
    ≥ 
    distance Q₅ Q₁ + distance Q₅ Q₂ + distance Q₅ Q₃ + distance Q₅ Q₄ + 
    distance Q₅ Q₅ + distance Q₅ Q₆ + distance Q₅ Q₇ + distance Q₅ Q₈ + 
    distance Q₅ Q₉ :=
by sorry

end minimize_sum_distances_l2281_228126


namespace gcd_of_three_numbers_l2281_228165

theorem gcd_of_three_numbers : Nat.gcd 1734 (Nat.gcd 816 1343) = 17 := by
  sorry

end gcd_of_three_numbers_l2281_228165


namespace function_composition_property_l2281_228160

/-- Given a function f(x) = (ax + b) / (cx + d), prove that if f(f(f(1))) = 1 and f(f(f(2))) = 3, then f(1) = 1. -/
theorem function_composition_property (a b c d : ℝ) :
  let f (x : ℝ) := (a * x + b) / (c * x + d)
  (f (f (f 1)) = 1) → (f (f (f 2)) = 3) → (f 1 = 1) := by
  sorry

end function_composition_property_l2281_228160


namespace zeta_power_sum_l2281_228198

theorem zeta_power_sum (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 1)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 9) :
  ζ₁^8 + ζ₂^8 + ζ₃^8 = 179 := by
  sorry

end zeta_power_sum_l2281_228198


namespace least_odd_prime_factor_of_2047_4_plus_1_l2281_228170

theorem least_odd_prime_factor_of_2047_4_plus_1 (p : Nat) : 
  p = 41 ↔ 
    Prime p ∧ 
    Odd p ∧ 
    p ∣ (2047^4 + 1) ∧ 
    ∀ q : Nat, Prime q → Odd q → q ∣ (2047^4 + 1) → p ≤ q :=
by sorry

end least_odd_prime_factor_of_2047_4_plus_1_l2281_228170


namespace second_train_length_problem_l2281_228155

/-- Calculates the length of the second train given the conditions of the problem -/
def second_train_length (first_train_length : ℝ) (first_train_speed : ℝ) (second_train_speed : ℝ) (time_to_cross : ℝ) : ℝ :=
  let relative_speed := first_train_speed - second_train_speed
  let total_distance := relative_speed * time_to_cross
  total_distance - first_train_length

/-- Theorem stating that given the problem conditions, the length of the second train is 299.9440044796417 m -/
theorem second_train_length_problem :
  let first_train_length : ℝ := 400
  let first_train_speed : ℝ := 72 * 1000 / 3600  -- Convert km/h to m/s
  let second_train_speed : ℝ := 36 * 1000 / 3600 -- Convert km/h to m/s
  let time_to_cross : ℝ := 69.99440044796417
  second_train_length first_train_length first_train_speed second_train_speed time_to_cross = 299.9440044796417 := by
  sorry

end second_train_length_problem_l2281_228155


namespace arithmetic_sequence_sum_difference_l2281_228168

theorem arithmetic_sequence_sum_difference : 
  let seq1 := List.range 93
  let seq2 := List.range 93
  let sum1 := (List.sum (seq1.map (fun i => 2001 + i)))
  let sum2 := (List.sum (seq2.map (fun i => 201 + i)))
  sum1 - sum2 = 167400 := by
sorry

end arithmetic_sequence_sum_difference_l2281_228168


namespace triangle_angle_A_l2281_228115

theorem triangle_angle_A (b c S_ABC : ℝ) (h1 : b = 8) (h2 : c = 8 * Real.sqrt 3)
  (h3 : S_ABC = 16 * Real.sqrt 3) :
  let A := Real.arcsin (1 / 2)
  A = π / 6 := by sorry

end triangle_angle_A_l2281_228115


namespace milk_distribution_l2281_228181

/-- Given a total number of milk bottles, number of cartons, and number of bags per carton,
    calculate the number of bottles in one bag. -/
def bottles_per_bag (total_bottles : ℕ) (num_cartons : ℕ) (bags_per_carton : ℕ) : ℕ :=
  total_bottles / (num_cartons * bags_per_carton)

/-- Prove that given 180 total bottles, 3 cartons, and 4 bags per carton,
    the number of bottles in one bag is 15. -/
theorem milk_distribution :
  bottles_per_bag 180 3 4 = 15 := by
  sorry

end milk_distribution_l2281_228181


namespace sin_arithmetic_is_geometric_ratio_l2281_228134

def is_arithmetic_sequence (α : ℕ → ℝ) (β : ℝ) :=
  ∀ n, α (n + 1) = α n + β

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem sin_arithmetic_is_geometric_ratio (α : ℕ → ℝ) (β : ℝ) :
  is_arithmetic_sequence α β →
  (∃ q, is_geometric_sequence (fun n ↦ Real.sin (α n)) q) →
  ∃ q, (q = 1 ∨ q = -1) ∧ is_geometric_sequence (fun n ↦ Real.sin (α n)) q :=
by sorry

end sin_arithmetic_is_geometric_ratio_l2281_228134


namespace log_inequality_condition_l2281_228127

theorem log_inequality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, Real.log a > Real.log b → 2*a > 2*b) ∧
  ¬(∀ a b, 2*a > 2*b → Real.log a > Real.log b) :=
sorry

end log_inequality_condition_l2281_228127


namespace union_determines_x_l2281_228187

def A : Set ℕ := {1, 3}
def B (x : ℕ) : Set ℕ := {2, x}

theorem union_determines_x (x : ℕ) : A ∪ B x = {1, 2, 3, 4} → x = 4 := by
  sorry

end union_determines_x_l2281_228187


namespace total_produce_yield_l2281_228154

def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def feet_per_step : ℕ := 3
def carrot_yield_per_sqft : ℚ := 0.4
def potato_yield_per_sqft : ℚ := 0.5

theorem total_produce_yield :
  let garden_length_feet := garden_length_steps * feet_per_step
  let garden_width_feet := garden_width_steps * feet_per_step
  let garden_area := garden_length_feet * garden_width_feet
  let carrot_yield := garden_area * carrot_yield_per_sqft
  let potato_yield := garden_area * potato_yield_per_sqft
  carrot_yield + potato_yield = 3645 := by
  sorry

end total_produce_yield_l2281_228154
