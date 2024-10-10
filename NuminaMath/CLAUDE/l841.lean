import Mathlib

namespace percent_of_number_l841_84165

theorem percent_of_number : (25 : ℝ) / 100 * 280 = 70 := by sorry

end percent_of_number_l841_84165


namespace rectangle_area_diagonal_l841_84132

theorem rectangle_area_diagonal (length width diagonal : ℝ) (h_ratio : length / width = 5 / 2) 
  (h_diagonal : diagonal^2 = length^2 + width^2) :
  length * width = (10 / 29) * diagonal^2 := by
  sorry

end rectangle_area_diagonal_l841_84132


namespace odd_function_property_l841_84154

def f (x : ℝ) (g : ℝ → ℝ) : ℝ := g x - 8

theorem odd_function_property (g : ℝ → ℝ) (m : ℝ) :
  (∀ x, g (-x) = -g x) →
  f (-m) g = 10 →
  f m g = -26 := by sorry

end odd_function_property_l841_84154


namespace composite_sum_of_squares_l841_84135

theorem composite_sum_of_squares (a b : ℤ) : 
  (∃ x y : ℤ, x^2 + a*x + 1 = b ∧ x ≠ y) → 
  b ≠ 1 → 
  ∃ m n : ℤ, m > 1 ∧ n > 1 ∧ m * n = a^2 + b^2 := by
  sorry

end composite_sum_of_squares_l841_84135


namespace negate_neg_sum_l841_84150

theorem negate_neg_sum (a b : ℝ) : -(-a - b) = a + b := by
  sorry

end negate_neg_sum_l841_84150


namespace cost_of_three_rides_is_171_l841_84131

/-- The cost of tickets for three rides at a fair -/
def cost_of_rides (ferris_wheel_tickets : ℕ) (roller_coaster_tickets : ℕ) (bumper_cars_tickets : ℕ) (cost_per_ticket : ℕ) : ℕ :=
  (ferris_wheel_tickets + roller_coaster_tickets + bumper_cars_tickets) * cost_per_ticket

/-- Theorem stating that the cost of the three rides is $171 -/
theorem cost_of_three_rides_is_171 :
  cost_of_rides 6 8 5 9 = 171 := by
  sorry

#eval cost_of_rides 6 8 5 9

end cost_of_three_rides_is_171_l841_84131


namespace root_sum_reciprocal_l841_84161

noncomputable def f (x : ℝ) : ℝ := x^3 + 3*x - 1

theorem root_sum_reciprocal (a b c : ℝ) (m n : ℕ) :
  f a = 0 → f b = 0 → f c = 0 →
  (1 / (a^3 + b^3) + 1 / (b^3 + c^3) + 1 / (c^3 + a^3) : ℝ) = m / n →
  m > 0 → n > 0 →
  Nat.gcd m n = 1 →
  100 * m + n = 3989 := by
sorry

end root_sum_reciprocal_l841_84161


namespace michael_purchase_l841_84105

/-- The amount Michael paid for his purchases after a discount -/
def amountPaid (suitCost shoesCost discount : ℕ) : ℕ :=
  suitCost + shoesCost - discount

/-- Theorem stating the correct amount Michael paid -/
theorem michael_purchase : amountPaid 430 190 100 = 520 := by
  sorry

end michael_purchase_l841_84105


namespace set_A_determination_l841_84197

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}

theorem set_A_determination (A : Set ℕ) (h : (U \ A) = {2, 3}) : A = {1, 4, 5} := by
  sorry

end set_A_determination_l841_84197


namespace specific_trapezoid_diagonal_l841_84155

/-- A trapezoid with integer side lengths and a right angle -/
structure RightTrapezoid where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ
  AB_parallel_CD : AB = CD
  right_angle_BCD : BC^2 + CD^2 = BD^2

/-- The diagonal length of the specific trapezoid -/
def diagonal_length (t : RightTrapezoid) : ℕ := 20

/-- Theorem: The diagonal length of the specific trapezoid is 20 -/
theorem specific_trapezoid_diagonal : 
  ∀ (t : RightTrapezoid), 
  t.AB = 7 → t.BC = 19 → t.CD = 7 → t.DA = 11 → 
  diagonal_length t = 20 := by
  sorry

end specific_trapezoid_diagonal_l841_84155


namespace cubic_inches_in_cubic_foot_l841_84128

-- Define the conversion factor
def inches_per_foot : ℕ := 12

-- Theorem statement
theorem cubic_inches_in_cubic_foot : 
  1 * (inches_per_foot ^ 3) = 1728 := by
  sorry

end cubic_inches_in_cubic_foot_l841_84128


namespace triangle_angle_measure_l841_84170

theorem triangle_angle_measure (A B C : ℝ) : 
  -- ABC is a triangle (sum of angles is 180°)
  A + B + C = 180 →
  -- Measure of angle C is 3/2 times the measure of angle B
  C = (3/2) * B →
  -- Angle B measures 30°
  B = 30 →
  -- Then the measure of angle A is 105°
  A = 105 := by
sorry

end triangle_angle_measure_l841_84170


namespace hyperbola_foci_coordinates_l841_84173

/-- The foci coordinates of the hyperbola x^2/4 - y^2 = 1 are (±√5, 0) -/
theorem hyperbola_foci_coordinates :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / 4 - y^2 = 1}
  ∃ (c : ℝ), c^2 = 5 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ hyperbola → 
      ((x = c ∧ y = 0) ∨ (x = -c ∧ y = 0))) :=
sorry

end hyperbola_foci_coordinates_l841_84173


namespace parabola_symmetry_l841_84194

/-- Represents a parabola in 2D space -/
structure Parabola where
  equation : ℝ → ℝ

/-- Two parabolas are symmetric about the origin -/
def symmetric_about_origin (p1 p2 : Parabola) : Prop :=
  ∀ x y : ℝ, p1.equation x = y ↔ p2.equation (-x) = -y

theorem parabola_symmetry (C1 C2 : Parabola) 
  (h1 : C1.equation = fun x ↦ (x - 2)^2 + 3)
  (h2 : symmetric_about_origin C1 C2) :
  C2.equation = fun x ↦ -(x + 2)^2 - 3 := by
  sorry


end parabola_symmetry_l841_84194


namespace two_digit_number_with_divisibility_properties_l841_84180

theorem two_digit_number_with_divisibility_properties : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n + 3) % 3 = 0 ∧ 
  (n + 7) % 7 = 0 ∧ 
  (n - 4) % 4 = 0 ∧
  n = 84 := by
sorry

end two_digit_number_with_divisibility_properties_l841_84180


namespace company_contracts_probability_l841_84198

theorem company_contracts_probability
  (p_hardware : ℝ)
  (p_not_software : ℝ)
  (p_network : ℝ)
  (p_maintenance : ℝ)
  (p_at_least_one : ℝ)
  (h_hardware : p_hardware = 3/4)
  (h_not_software : p_not_software = 3/5)
  (h_network : p_network = 2/3)
  (h_maintenance : p_maintenance = 1/2)
  (h_at_least_one : p_at_least_one = 7/8) :
  p_hardware * (1 - p_not_software) * p_network * p_maintenance = 1/10 :=
sorry

end company_contracts_probability_l841_84198


namespace train_crossing_time_l841_84182

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 100 →
  train_speed_kmh = 180 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 2 := by
  sorry

#check train_crossing_time

end train_crossing_time_l841_84182


namespace certain_number_proof_l841_84164

theorem certain_number_proof (N : ℕ) (h1 : N < 81) 
  (h2 : ∀ k : ℕ, k ∈ Finset.range 15 → N + k + 1 < 81) 
  (h3 : N + 16 ≥ 81) : N = 65 := by
sorry

end certain_number_proof_l841_84164


namespace parallel_vectors_cos_2alpha_l841_84119

theorem parallel_vectors_cos_2alpha (α : ℝ) :
  let a : ℝ × ℝ := (1/3, Real.tan α)
  let b : ℝ × ℝ := (Real.cos α, 1)
  (∃ (k : ℝ), a = k • b) → Real.cos (2 * α) = 7/9 := by
  sorry

end parallel_vectors_cos_2alpha_l841_84119


namespace quadratic_roots_l841_84118

/-- Represents a quadratic equation of the form 2x^2 + (m+2)x + m = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  2 * x^2 + (m + 2) * x + m = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (m + 2)^2 - 4 * 2 * m

theorem quadratic_roots (m : ℝ) :
  (∀ x, ∃ y z, quadratic_equation m x → x = y ∨ x = z) ∧
  (discriminant 2 = 0) ∧
  (quadratic_equation 2 (-1) ∧ ∀ x, quadratic_equation 2 x → x = -1) :=
sorry

end quadratic_roots_l841_84118


namespace boot_purchase_theorem_l841_84195

def boot_purchase_problem (initial_amount hand_sanitizer_discount toilet_paper_cost : ℚ) : ℚ :=
  let hand_sanitizer_cost : ℚ := 6
  let large_ham_cost : ℚ := 2 * toilet_paper_cost
  let cheese_cost : ℚ := hand_sanitizer_cost / 2
  let total_spent : ℚ := toilet_paper_cost + hand_sanitizer_cost + large_ham_cost + cheese_cost
  let remaining : ℚ := initial_amount - total_spent
  let savings : ℚ := remaining * (1/5)
  let spendable : ℚ := remaining - savings
  let per_twin : ℚ := spendable / 2
  let boot_cost : ℚ := per_twin * 4
  let total_boot_cost : ℚ := boot_cost * 2
  (total_boot_cost - spendable) / 2

theorem boot_purchase_theorem :
  boot_purchase_problem 100 (1/4) 12 = 66 := by sorry

end boot_purchase_theorem_l841_84195


namespace circle_theorem_l841_84107

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center of the required circle lies
def centerLine (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the equation of the required circle
def requiredCircle (x y : ℝ) : Prop := x^2 + y^2 - x + 7*y - 32 = 0

-- Theorem statement
theorem circle_theorem :
  ∀ (x y : ℝ),
    (circle1 x y ∧ circle2 x y → requiredCircle x y) ∧
    (∃ (h k : ℝ), centerLine h k ∧ 
      ∀ (x y : ℝ), requiredCircle x y ↔ (x - h)^2 + (y - k)^2 = (h - x)^2 + (k - y)^2) :=
sorry

end circle_theorem_l841_84107


namespace roots_sum_theorem_l841_84114

theorem roots_sum_theorem (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a + b + c = 6 →
  a*b + a*c + b*c = 11 →
  a*b*c = 6 →
  (a / (b*c + 2)) + (b / (a*c + 2)) + (c / (a*b + 2)) = 3/2 := by
sorry

end roots_sum_theorem_l841_84114


namespace arithmetic_sequence_common_difference_l841_84108

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) 
  (h_arithmetic : ArithmeticSequence a)
  (h_eq : a 3 + a 9 = 4 * a 5)
  (h_a2 : a 2 = -8) :
  ∃ d : ℤ, d = 4 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l841_84108


namespace restaurant_budget_theorem_l841_84143

theorem restaurant_budget_theorem (budget : ℝ) (budget_positive : budget > 0) :
  let rent := (1 / 4) * budget
  let remaining := budget - rent
  let food_and_beverages := (1 / 4) * remaining
  (food_and_beverages / budget) * 100 = 18.75 := by
  sorry

end restaurant_budget_theorem_l841_84143


namespace random_variable_distribution_invariance_l841_84110

-- Define a type for random variables
variable (Ω : Type) [MeasurableSpace Ω]
def RandomVariable (α : Type) [MeasurableSpace α] := Ω → α

-- Define a type for distribution functions
def DistributionFunction (α : Type) [MeasurableSpace α] := α → ℝ

-- State the theorem
theorem random_variable_distribution_invariance
  (ξ : RandomVariable Ω ℝ)
  (h_non_degenerate : ∀ (c : ℝ), ¬(∀ (ω : Ω), ξ ω = c))
  (a : ℝ)
  (b : ℝ)
  (h_a_pos : a > 0)
  (h_distribution_equal : ∀ (F : DistributionFunction ℝ),
    (∀ (x : ℝ), F x = F ((x - b) / a))) :
  a = 1 ∧ b = 0 :=
sorry

end random_variable_distribution_invariance_l841_84110


namespace combined_weight_proof_l841_84157

def combined_weight (mary_weight jamison_weight john_weight peter_weight : ℝ) : ℝ :=
  mary_weight + jamison_weight + john_weight + peter_weight

theorem combined_weight_proof (mary_weight : ℝ) 
  (h1 : mary_weight = 160)
  (h2 : ∃ jamison_weight : ℝ, jamison_weight = mary_weight + 20)
  (h3 : ∃ john_weight : ℝ, john_weight = mary_weight * 1.25)
  (h4 : ∃ peter_weight : ℝ, peter_weight = john_weight * 1.15) :
  ∃ total_weight : ℝ, combined_weight mary_weight 
    (mary_weight + 20) (mary_weight * 1.25) (mary_weight * 1.25 * 1.15) = 770 :=
by
  sorry

end combined_weight_proof_l841_84157


namespace win_loss_opposite_win_loss_opposite_meanings_l841_84117

/-- Represents the outcome of a game -/
inductive GameOutcome
| Win
| Loss

/-- Represents a team's or individual's record -/
structure Record where
  wins : ℕ
  losses : ℕ

/-- Updates the record based on a game outcome -/
def updateRecord (r : Record) (outcome : GameOutcome) : Record :=
  match outcome with
  | GameOutcome.Win => { wins := r.wins + 1, losses := r.losses }
  | GameOutcome.Loss => { wins := r.wins, losses := r.losses + 1 }

/-- Theorem stating that winning and losing have opposite effects on a record -/
theorem win_loss_opposite (r : Record) :
  updateRecord r GameOutcome.Win ≠ updateRecord r GameOutcome.Loss :=
by
  sorry

/-- Theorem stating that winning and losing are quantities with opposite meanings -/
theorem win_loss_opposite_meanings :
  ∃ (r : Record), updateRecord r GameOutcome.Win ≠ updateRecord r GameOutcome.Loss :=
by
  sorry

end win_loss_opposite_win_loss_opposite_meanings_l841_84117


namespace number_of_boys_l841_84112

theorem number_of_boys (total_pupils : ℕ) (number_of_girls : ℕ) 
  (h1 : total_pupils = 485) 
  (h2 : number_of_girls = 232) : 
  total_pupils - number_of_girls = 253 := by
  sorry

end number_of_boys_l841_84112


namespace max_value_of_function_l841_84196

theorem max_value_of_function (x : ℝ) (h : x < 1/3) :
  3 * x + 1 / (3 * x - 1) ≤ -1 :=
sorry

end max_value_of_function_l841_84196


namespace tuesday_is_valid_start_day_l841_84189

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDays d m)

def isValidRedemptionSchedule (startDay : DayOfWeek) : Prop :=
  ∀ i : Fin 7, advanceDays startDay (i.val * 12) ≠ DayOfWeek.Saturday

theorem tuesday_is_valid_start_day :
  isValidRedemptionSchedule DayOfWeek.Tuesday ∧
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Tuesday → ¬ isValidRedemptionSchedule d :=
sorry

end tuesday_is_valid_start_day_l841_84189


namespace hexagon_perimeter_l841_84185

/-- The perimeter of a hexagon with side length 4 inches is 24 inches. -/
theorem hexagon_perimeter (side_length : ℝ) : side_length = 4 → 6 * side_length = 24 := by
  sorry

end hexagon_perimeter_l841_84185


namespace max_value_sqrt_sum_l841_84124

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 5 → Real.sqrt (x + 1) + Real.sqrt (y + 3) ≤ 3 * Real.sqrt 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 5 ∧ Real.sqrt (x + 1) + Real.sqrt (y + 3) = 3 * Real.sqrt 2) :=
by sorry

end max_value_sqrt_sum_l841_84124


namespace max_gcd_15n_plus_4_8n_plus_1_l841_84192

theorem max_gcd_15n_plus_4_8n_plus_1 :
  ∃ (k : ℕ), k > 0 ∧ Nat.gcd (15 * k + 4) (8 * k + 1) = 17 ∧
  ∀ (n : ℕ), n > 0 → Nat.gcd (15 * n + 4) (8 * n + 1) ≤ 17 := by
  sorry

end max_gcd_15n_plus_4_8n_plus_1_l841_84192


namespace library_visitors_average_l841_84147

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays : ℕ := 5
  let totalOtherDays : ℕ := 25
  let totalVisitors : ℕ := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  totalVisitors / 30

/-- Theorem stating that the average number of visitors per day is 285 -/
theorem library_visitors_average (sundayVisitors : ℕ) (otherDayVisitors : ℕ) 
    (h1 : sundayVisitors = 510) (h2 : otherDayVisitors = 240) : 
    averageVisitorsPerDay sundayVisitors otherDayVisitors = 285 := by
  sorry

#eval averageVisitorsPerDay 510 240

end library_visitors_average_l841_84147


namespace triangle_quadratic_no_real_roots_l841_84142

/-- Given a triangle with side lengths a, b, c, the quadratic equation 
    b^2 x^2 - (b^2 + c^2 - a^2)x + c^2 = 0 has no real roots. -/
theorem triangle_quadratic_no_real_roots (a b c : ℝ) 
    (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
    ∀ x : ℝ, b^2 * x^2 - (b^2 + c^2 - a^2) * x + c^2 ≠ 0 := by
  sorry

end triangle_quadratic_no_real_roots_l841_84142


namespace modulus_of_z_l841_84133

-- Define the complex number z
def z : ℂ := Complex.I * (1 - Complex.I)

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by sorry

end modulus_of_z_l841_84133


namespace product_of_logarithms_l841_84127

theorem product_of_logarithms (c d : ℕ) (hc : c > 0) (hd : d > 0) :
  (Real.log d / Real.log c = 2) → (d - c = 630) → (c + d = 1260) := by
  sorry

end product_of_logarithms_l841_84127


namespace button_probability_l841_84160

def initial_red_c : ℕ := 6
def initial_green_c : ℕ := 12
def initial_total_c : ℕ := initial_red_c + initial_green_c

def remaining_fraction : ℚ := 3/4

theorem button_probability : 
  ∃ (removed_red removed_green : ℕ),
    removed_red = removed_green ∧
    initial_total_c - (removed_red + removed_green) = (remaining_fraction * initial_total_c).num ∧
    (initial_green_c - removed_green : ℚ) / (initial_total_c - (removed_red + removed_green) : ℚ) *
    (removed_green : ℚ) / ((removed_red + removed_green) : ℚ) = 5/14 :=
by sorry

end button_probability_l841_84160


namespace g_minimum_value_l841_84176

noncomputable def g (x : ℝ) : ℝ := x + x / (x^2 + 2) + x * (x + 5) / (x^2 + 3) + 3 * (x + 3) / (x * (x^2 + 3))

theorem g_minimum_value (x : ℝ) (hx : x > 0) : g x ≥ 6 := by
  sorry

end g_minimum_value_l841_84176


namespace sum_of_squares_with_given_means_l841_84136

theorem sum_of_squares_with_given_means (a b : ℝ) :
  (a + b) / 2 = 8 → Real.sqrt (a * b) = 2 * Real.sqrt 5 → a^2 + b^2 = 216 := by
  sorry

end sum_of_squares_with_given_means_l841_84136


namespace nails_to_buy_proof_l841_84122

/-- Given the total number of nails needed, the number of nails already owned,
    and the number of nails found in the toolshed, calculate the number of nails
    that need to be bought. -/
def nails_to_buy (total_needed : ℕ) (already_owned : ℕ) (found_in_toolshed : ℕ) : ℕ :=
  total_needed - (already_owned + found_in_toolshed)

/-- Prove that the number of nails needed to buy is 109 given the specific quantities. -/
theorem nails_to_buy_proof :
  nails_to_buy 500 247 144 = 109 := by
  sorry

end nails_to_buy_proof_l841_84122


namespace calculate_gladys_speed_l841_84199

def team_size : Nat := 5

def rudy_speed : Nat := 64
def joyce_speed : Nat := 76
def lisa_speed : Nat := 80
def mike_speed : Nat := 89

def team_average : Nat := 80

def gladys_speed : Nat := 91

theorem calculate_gladys_speed :
  team_size * team_average - (rudy_speed + joyce_speed + lisa_speed + mike_speed) = gladys_speed := by
  sorry

end calculate_gladys_speed_l841_84199


namespace perpendicular_vectors_l841_84149

/-- Two 2D vectors are perpendicular if their dot product is zero -/
def are_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

/-- The problem statement -/
theorem perpendicular_vectors (a : ℝ) (h1 : a ≠ 0) 
  (h2 : are_perpendicular (a, a+4) (-5, a)) : a = 1 := by
  sorry

end perpendicular_vectors_l841_84149


namespace square_fence_perimeter_l841_84144

/-- The outer perimeter of a square fence with evenly spaced posts -/
theorem square_fence_perimeter
  (num_posts : ℕ)
  (post_width_inches : ℕ)
  (gap_between_posts_feet : ℕ)
  (h1 : num_posts = 16)
  (h2 : post_width_inches = 6)
  (h3 : gap_between_posts_feet = 4) :
  (4 * (↑num_posts / 4 * (↑post_width_inches / 12 + ↑gap_between_posts_feet) - ↑gap_between_posts_feet)) = 56 :=
by sorry

end square_fence_perimeter_l841_84144


namespace weight_ratio_l841_84163

theorem weight_ratio (sam_weight tyler_weight peter_weight : ℝ) : 
  tyler_weight = sam_weight + 25 →
  sam_weight = 105 →
  peter_weight = 65 →
  peter_weight / tyler_weight = 0.5 := by
sorry

end weight_ratio_l841_84163


namespace deal_or_no_deal_probability_l841_84178

theorem deal_or_no_deal_probability (total_boxes : Nat) (high_value_boxes : Nat) 
  (h1 : total_boxes = 26)
  (h2 : high_value_boxes = 7) :
  total_boxes - (high_value_boxes + high_value_boxes) = 12 := by
sorry

end deal_or_no_deal_probability_l841_84178


namespace n_even_factors_l841_84129

def n : ℕ := 2^3 * 3^2 * 5^1 * 7^3

/-- The number of even natural-number factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem n_even_factors :
  num_even_factors n = 72 := by sorry

end n_even_factors_l841_84129


namespace product_of_constrained_integers_l841_84177

theorem product_of_constrained_integers (a b : ℕ) 
  (h1 : 90 < a + b ∧ a + b < 99)
  (h2 : (9 : ℚ)/10 < (a : ℚ)/(b : ℚ) ∧ (a : ℚ)/(b : ℚ) < (91 : ℚ)/100) :
  a * b = 2346 := by
  sorry

end product_of_constrained_integers_l841_84177


namespace intersection_when_m_is_one_union_equals_B_iff_l841_84130

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

-- Theorem 1: When m = 1, A ∩ B = {x | 3 ≤ x < 4}
theorem intersection_when_m_is_one :
  A 1 ∩ B = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2: A ∪ B = B if and only if m ≥ 3 or m ≤ -3
theorem union_equals_B_iff (m : ℝ) :
  A m ∪ B = B ↔ m ≥ 3 ∨ m ≤ -3 := by sorry

end intersection_when_m_is_one_union_equals_B_iff_l841_84130


namespace national_park_pines_l841_84101

theorem national_park_pines (pines redwoods : ℕ) : 
  redwoods = pines + pines / 5 →
  pines + redwoods = 1320 →
  pines = 600 := by
sorry

end national_park_pines_l841_84101


namespace hair_growth_calculation_l841_84186

theorem hair_growth_calculation (initial_length : ℝ) (growth : ℝ) (final_length : ℝ) : 
  initial_length = 24 →
  final_length = 14 →
  final_length = initial_length / 2 + growth - 2 →
  growth = 4 := by
sorry

end hair_growth_calculation_l841_84186


namespace analytical_method_seeks_sufficient_conditions_l841_84169

/-- The analytical method for proving inequalities -/
structure AnalyticalMethod where
  /-- The method proceeds from effect to cause -/
  effect_to_cause : Bool

/-- A condition in the context of proving inequalities -/
inductive Condition
  | Necessary
  | Sufficient
  | NecessaryAndSufficient
  | NecessaryOrSufficient

/-- The reasoning process sought by the analytical method -/
def reasoning_process (method : AnalyticalMethod) : Condition :=
  Condition.Sufficient

/-- Theorem stating that the analytical method seeks sufficient conditions -/
theorem analytical_method_seeks_sufficient_conditions (method : AnalyticalMethod) :
  reasoning_process method = Condition.Sufficient := by
  sorry

end analytical_method_seeks_sufficient_conditions_l841_84169


namespace arithmetic_sequence_property_l841_84193

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : isArithmeticSequence a)
  (h_fifth : a 5 = 10)
  (h_sum : a 1 + a 2 + a 3 = 3) :
  a 1 = -2 ∧ ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 := by
  sorry

#check arithmetic_sequence_property

end arithmetic_sequence_property_l841_84193


namespace cone_lateral_surface_area_l841_84123

theorem cone_lateral_surface_area 
  (r : Real) 
  (l : Real) 
  (h_r : r = Real.sqrt 2) 
  (h_l : l = 3 * Real.sqrt 2) : 
  r * l * Real.pi = 6 * Real.pi := by
  sorry

end cone_lateral_surface_area_l841_84123


namespace unique_solution_implies_equal_or_opposite_l841_84138

theorem unique_solution_implies_equal_or_opposite (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃! x : ℝ, a * (x - a)^2 + b * (x - b)^2 = 0) → a = b ∨ a = -b := by
  sorry

end unique_solution_implies_equal_or_opposite_l841_84138


namespace parabola_c_value_l841_84171

/-- A parabola passing through two specific points has a unique c-value -/
theorem parabola_c_value (b : ℝ) :
  ∃! c : ℝ, (2^2 + 2*b + c = 20) ∧ ((-2)^2 + (-2)*b + c = -4) := by
  sorry

end parabola_c_value_l841_84171


namespace sample_size_b_l841_84106

/-- Represents the number of products in each batch -/
structure BatchSizes where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the sample sizes from each batch -/
structure SampleSizes where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The theorem to prove -/
theorem sample_size_b (batchSizes : BatchSizes) (sampleSizes : SampleSizes) : 
  batchSizes.a + batchSizes.b + batchSizes.c = 210 →
  batchSizes.c - batchSizes.b = batchSizes.b - batchSizes.a →
  sampleSizes.a + sampleSizes.b + sampleSizes.c = 60 →
  sampleSizes.c - sampleSizes.b = sampleSizes.b - sampleSizes.a →
  sampleSizes.b = 20 := by
sorry

end sample_size_b_l841_84106


namespace emily_elephant_four_hops_l841_84172

/-- The distance covered in a single hop, given the remaining distance to the target -/
def hop_distance (remaining : ℚ) : ℚ := (1 / 4) * remaining

/-- The remaining distance to the target after a hop -/
def remaining_after_hop (remaining : ℚ) : ℚ := remaining - hop_distance remaining

/-- The total distance covered after n hops -/
def total_distance (n : ℕ) : ℚ :=
  let rec aux (k : ℕ) (remaining : ℚ) (acc : ℚ) : ℚ :=
    if k = 0 then acc
    else aux (k - 1) (remaining_after_hop remaining) (acc + hop_distance remaining)
  aux n 1 0

theorem emily_elephant_four_hops :
  total_distance 4 = 175 / 256 := by sorry

end emily_elephant_four_hops_l841_84172


namespace intersection_P_Q_l841_84156

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {1, 2} := by sorry

end intersection_P_Q_l841_84156


namespace tank_plastering_cost_l841_84162

/-- Calculate the cost of plastering a tank's walls and bottom -/
theorem tank_plastering_cost 
  (length width depth : ℝ)
  (cost_per_sqm_paise : ℝ)
  (h_length : length = 25)
  (h_width : width = 12)
  (h_depth : depth = 6)
  (h_cost : cost_per_sqm_paise = 75) :
  let surface_area := 2 * (length * depth + width * depth) + length * width
  let cost_rupees := surface_area * (cost_per_sqm_paise / 100)
  cost_rupees = 558 := by
sorry

end tank_plastering_cost_l841_84162


namespace zoe_winter_clothing_boxes_l841_84184

theorem zoe_winter_clothing_boxes :
  let items_per_box := 4 + 6  -- 4 scarves and 6 mittens per box
  let total_items := 80       -- total pieces of winter clothing
  total_items / items_per_box = 8 := by
  sorry

end zoe_winter_clothing_boxes_l841_84184


namespace expression_evaluation_l841_84103

theorem expression_evaluation (a b c : ℝ) 
  (h : a / (45 - a) + b / (85 - b) + c / (75 - c) = 9) :
  9 / (45 - a) + 17 / (85 - b) + 15 / (75 - c) = 2.4 := by
  sorry

end expression_evaluation_l841_84103


namespace smallest_stairs_count_l841_84100

theorem smallest_stairs_count : ∃ (n : ℕ), n > 15 ∧ n % 6 = 4 ∧ n % 7 = 3 ∧ ∀ (m : ℕ), m > 15 ∧ m % 6 = 4 ∧ m % 7 = 3 → n ≤ m := by
  sorry

end smallest_stairs_count_l841_84100


namespace divide_multiply_result_l841_84111

theorem divide_multiply_result : (3 / 4) * 12 = 9 := by
  sorry

end divide_multiply_result_l841_84111


namespace square_of_98_l841_84145

theorem square_of_98 : (98 : ℕ) ^ 2 = 9604 := by
  sorry

end square_of_98_l841_84145


namespace max_carlson_jars_l841_84148

/-- Represents the initial state of jam jars for Carlson and Baby -/
structure JamJars :=
  (carlson_weight : ℕ)  -- Total weight of Carlson's jars
  (baby_weight : ℕ)     -- Total weight of Baby's jars
  (lightest_jar : ℕ)    -- Weight of Carlson's lightest jar

/-- The conditions of the problem -/
def valid_jam_state (j : JamJars) : Prop :=
  j.carlson_weight = 13 * j.baby_weight ∧
  j.carlson_weight - j.lightest_jar = 8 * (j.baby_weight + j.lightest_jar)

/-- The maximum number of jars Carlson could have initially -/
def max_jars (j : JamJars) : ℕ := j.carlson_weight / j.lightest_jar

/-- The theorem to prove -/
theorem max_carlson_jars :
  ∀ j : JamJars, valid_jam_state j → max_jars j ≤ 23 :=
by sorry

end max_carlson_jars_l841_84148


namespace check_mistake_l841_84175

theorem check_mistake (x y : ℕ) : 
  (100 * y + x) - (100 * x + y) = 1368 → y = x + 14 := by
  sorry

end check_mistake_l841_84175


namespace quadratic_roots_properties_l841_84134

/-- A quadratic function f(x) = x^2 + bx + c with real constants b and c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_roots_properties (b c x₁ x₂ : ℝ) 
  (hroot₁ : f b c x₁ = x₁)
  (hroot₂ : f b c x₂ = x₂)
  (hx₁_pos : x₁ > 0)
  (hx₂_x₁ : x₂ - x₁ > 1) :
  (b^2 > 2*(b + 2*c)) ∧ 
  (∀ t : ℝ, 0 < t → t < x₁ → f b c t > x₁) := by
  sorry

end quadratic_roots_properties_l841_84134


namespace complex_equation_sum_of_squares_l841_84188

theorem complex_equation_sum_of_squares (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a - 2 * i) * i^2013 = b - i →
  a^2 + b^2 = 5 := by
sorry

end complex_equation_sum_of_squares_l841_84188


namespace car_fuel_efficiency_l841_84151

theorem car_fuel_efficiency (H : ℝ) : 
  (H > 0) →
  (4 / H + 4 / 20 = 8 / H * 1.3499999999999999) →
  H = 34 := by
sorry

end car_fuel_efficiency_l841_84151


namespace area_between_specific_lines_l841_84167

/-- Line passing through two points -/
structure Line where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- Calculate the area between two lines within a given x-range -/
noncomputable def areaBetweenLines (l1 l2 : Line) (x_start x_end : ℝ) : ℝ :=
  sorry

/-- The problem statement -/
theorem area_between_specific_lines :
  let line1 : Line := { x1 := 0, y1 := 5, x2 := 10, y2 := 2 }
  let line2 : Line := { x1 := 2, y1 := 6, x2 := 6, y2 := 0 }
  areaBetweenLines line1 line2 2 6 = 8 := by
  sorry

end area_between_specific_lines_l841_84167


namespace sqrt_equation_solution_l841_84140

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (9 - 5 * x) = 8 → x = -11 := by
  sorry

end sqrt_equation_solution_l841_84140


namespace trees_that_died_haley_trees_died_l841_84121

theorem trees_that_died (total : ℕ) (survived_more : ℕ) : ℕ :=
  let died := (total - survived_more) / 2
  died

theorem haley_trees_died : trees_that_died 11 7 = 2 := by
  sorry

end trees_that_died_haley_trees_died_l841_84121


namespace sqrt_inequality_l841_84159

theorem sqrt_inequality (a : ℝ) (h : a > 6) :
  Real.sqrt (a - 3) - Real.sqrt (a - 4) < Real.sqrt (a - 5) - Real.sqrt (a - 6) := by
  sorry

end sqrt_inequality_l841_84159


namespace max_area_right_triangle_l841_84166

-- Define a right-angled triangle with integer side lengths
def RightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Define the perimeter constraint
def Perimeter (a b c : ℕ) : Prop :=
  a + b + c = 48

-- Define the area of a triangle
def Area (a b : ℕ) : ℕ :=
  a * b / 2

-- Theorem statement
theorem max_area_right_triangle :
  ∀ a b c : ℕ,
  RightTriangle a b c →
  Perimeter a b c →
  Area a b ≤ 288 :=
sorry

end max_area_right_triangle_l841_84166


namespace costco_mayo_price_l841_84120

/-- The cost of a gallon of mayo at Costco -/
def costco_gallon_cost : ℚ := 8

/-- The volume of a gallon in ounces -/
def gallon_ounces : ℕ := 128

/-- The volume of a standard bottle in ounces -/
def bottle_ounces : ℕ := 16

/-- The cost of a standard bottle at a normal store -/
def normal_store_bottle_cost : ℚ := 3

/-- The savings when buying at Costco -/
def costco_savings : ℚ := 16

theorem costco_mayo_price :
  costco_gallon_cost = 
    (gallon_ounces / bottle_ounces : ℚ) * normal_store_bottle_cost - costco_savings :=
by sorry

end costco_mayo_price_l841_84120


namespace cosine_transformation_symmetry_l841_84109

open Real

theorem cosine_transformation_symmetry (ω : ℝ) :
  ω > 0 →
  (∀ x, ∃ y, cos (ω * (x - π / 12)) = y) →
  (∀ x, cos (ω * ((π / 4 + (π / 4 - x)) - π / 12)) = cos (ω * (x - π / 12))) →
  ω ≥ 6 :=
by sorry

end cosine_transformation_symmetry_l841_84109


namespace min_three_digit_quotient_l841_84179

def three_digit_quotient (a b : ℕ) : ℚ :=
  (100 * a + 10 * b + 1) / (a + b + 1)

theorem min_three_digit_quotient :
  ∀ a b : ℕ, 2 ≤ a → a ≤ 9 → 2 ≤ b → b ≤ 9 → a ≠ b →
  three_digit_quotient a b ≥ 24.25 ∧
  ∃ a₀ b₀ : ℕ, 2 ≤ a₀ ∧ a₀ ≤ 9 ∧ 2 ≤ b₀ ∧ b₀ ≤ 9 ∧ a₀ ≠ b₀ ∧
  three_digit_quotient a₀ b₀ = 24.25 :=
sorry

end min_three_digit_quotient_l841_84179


namespace scientific_notation_of_238_billion_l841_84125

/-- A billion is defined as 10^9 -/
def billion : ℕ := 10^9

/-- The problem statement -/
theorem scientific_notation_of_238_billion :
  (238 : ℝ) * billion = 2.38 * (10 : ℝ)^10 := by
  sorry

end scientific_notation_of_238_billion_l841_84125


namespace radhika_games_count_l841_84115

/-- The number of video games Radhika owns now -/
def total_games (christmas_games birthday_games family_games : ℕ) : ℕ :=
  let total_gifts := christmas_games + birthday_games + family_games
  let initial_games := (2 * total_gifts) / 3
  initial_games + total_gifts

/-- Theorem stating the total number of video games Radhika owns -/
theorem radhika_games_count :
  total_games 12 8 5 = 41 := by
  sorry

#eval total_games 12 8 5

end radhika_games_count_l841_84115


namespace evaluate_g_l841_84153

/-- The function g(x) = 3x^2 - 5x + 8 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 8

/-- Theorem: 3g(2) + 2g(-2) = 90 -/
theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 90 := by
  sorry

end evaluate_g_l841_84153


namespace max_value_theorem_l841_84174

theorem max_value_theorem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * Real.sqrt 6 + 5 * y * z ≤ Real.sqrt 6 * (2 * Real.sqrt (375/481)) + 5 * (2 * Real.sqrt (106/481)) :=
sorry

end max_value_theorem_l841_84174


namespace rectangle_area_is_eight_l841_84113

/-- A square with side length 4 containing two right triangles whose hypotenuses
    are opposite sides of the square. -/
structure SquareWithTriangles where
  side_length : ℝ
  hypotenuse_length : ℝ
  rectangle_width : ℝ
  rectangle_height : ℝ
  h_side_length : side_length = 4
  h_hypotenuse : hypotenuse_length = side_length
  h_right_triangle : rectangle_width ^ 2 + rectangle_height ^ 2 = hypotenuse_length ^ 2
  h_rectangle_dim : rectangle_width + rectangle_height = side_length

/-- The area of the rectangle formed by the intersection of the triangles is 8. -/
theorem rectangle_area_is_eight (s : SquareWithTriangles) : 
  s.rectangle_width * s.rectangle_height = 8 := by
  sorry

end rectangle_area_is_eight_l841_84113


namespace tangent_circle_intersection_distance_l841_84137

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the intersection of two circles
variable (intersect : Circle → Circle → Point)

-- Define the tangent line at a point on a circle
variable (tangent_at : Point → Circle → Point → Prop)

-- Define a circle passing through three points
variable (circle_through : Point → Point → Point → Circle)

-- Define the distance between two points
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem tangent_circle_intersection_distance
  (C₁ C₂ C₃ : Circle) (S A B P Q : Point) :
  intersect C₁ C₂ = S →
  tangent_at S C₁ A →
  tangent_at S C₂ B →
  C₃ = circle_through A B S →
  tangent_at S C₃ P →
  tangent_at S C₃ Q →
  A ≠ S →
  B ≠ S →
  P ≠ S →
  Q ≠ S →
  distance P S = distance Q S :=
sorry

end tangent_circle_intersection_distance_l841_84137


namespace yellow_gumdrops_after_replacement_l841_84116

/-- Represents the number of gumdrops of each color in a jar -/
structure GumdropsJar where
  blue : ℕ
  brown : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- The total number of gumdrops in the jar -/
def GumdropsJar.total (jar : GumdropsJar) : ℕ :=
  jar.blue + jar.brown + jar.red + jar.yellow + jar.green

/-- The percentage of gumdrops of a given color -/
def GumdropsJar.percentage (jar : GumdropsJar) (color : ℕ) : ℚ :=
  color / jar.total

theorem yellow_gumdrops_after_replacement (jar : GumdropsJar) :
  jar.blue = (jar.total * 2) / 5 →
  jar.brown = (jar.total * 3) / 20 →
  jar.red = jar.total / 10 →
  jar.yellow = jar.total / 5 →
  jar.green = 50 →
  (jar.yellow + jar.red / 3 : ℕ) = 78 := by
  sorry

end yellow_gumdrops_after_replacement_l841_84116


namespace regular_polygon_sides_l841_84187

theorem regular_polygon_sides (D : ℕ) : D = 20 → ∃ n : ℕ, n = 8 ∧ D = n * (n - 3) / 2 := by
  sorry

end regular_polygon_sides_l841_84187


namespace john_paintball_cost_l841_84190

/-- John's monthly expenditure on paintballs -/
def monthly_paintball_cost (plays_per_month : ℕ) (boxes_per_play : ℕ) (cost_per_box : ℕ) : ℕ :=
  plays_per_month * boxes_per_play * cost_per_box

/-- Theorem: John spends $225 a month on paintballs -/
theorem john_paintball_cost :
  monthly_paintball_cost 3 3 25 = 225 := by
  sorry

end john_paintball_cost_l841_84190


namespace least_sum_m_n_l841_84152

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m.val + n.val) 330 = 1) ∧ 
  (∃ (k : ℕ), m.val^m.val = k * n.val^n.val) ∧ 
  (∀ (j : ℕ), m.val ≠ j * n.val) ∧
  (m.val + n.val = 247) ∧
  (∀ (m' n' : ℕ+), 
    (Nat.gcd (m'.val + n'.val) 330 = 1) → 
    (∃ (k : ℕ), m'.val^m'.val = k * n'.val^n'.val) → 
    (∀ (j : ℕ), m'.val ≠ j * n'.val) → 
    (m'.val + n'.val ≥ 247)) :=
by sorry

end least_sum_m_n_l841_84152


namespace intersection_point_d_l841_84139

/-- A function g(x) = 2x + c with c being an integer -/
def g (c : ℤ) : ℝ → ℝ := λ x ↦ 2 * x + c

/-- The inverse function of g -/
noncomputable def g_inv (c : ℤ) : ℝ → ℝ := λ x ↦ (x - c) / 2

theorem intersection_point_d (c : ℤ) (d : ℤ) :
  g c (-4) = d ∧ g_inv c (-4) = d → d = -4 := by sorry

end intersection_point_d_l841_84139


namespace hex_20F_to_decimal_l841_84181

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9
| A | B | C | D | E | F

/-- Converts a HexDigit to its decimal value --/
def hexToDecimal (d : HexDigit) : ℕ :=
  match d with
  | HexDigit.D0 => 0 | HexDigit.D1 => 1 | HexDigit.D2 => 2 | HexDigit.D3 => 3
  | HexDigit.D4 => 4 | HexDigit.D5 => 5 | HexDigit.D6 => 6 | HexDigit.D7 => 7
  | HexDigit.D8 => 8 | HexDigit.D9 => 9 | HexDigit.A => 10 | HexDigit.B => 11
  | HexDigit.C => 12 | HexDigit.D => 13 | HexDigit.E => 14 | HexDigit.F => 15

/-- Converts a list of HexDigits to its decimal value --/
def hexListToDecimal (digits : List HexDigit) : ℤ :=
  digits.enum.foldl (fun acc (i, d) => acc + (hexToDecimal d : ℤ) * 16^(digits.length - 1 - i)) 0

/-- The hexadecimal number -20F --/
def hex20F : List HexDigit := [HexDigit.D2, HexDigit.D0, HexDigit.F]

theorem hex_20F_to_decimal :
  -hexListToDecimal hex20F = -527 := by sorry

end hex_20F_to_decimal_l841_84181


namespace square_sum_constant_l841_84126

theorem square_sum_constant (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end square_sum_constant_l841_84126


namespace power_of_product_l841_84183

theorem power_of_product (x y : ℝ) : (-2 * x * y^2)^3 = -8 * x^3 * y^6 := by
  sorry

end power_of_product_l841_84183


namespace right_triangle_leg_length_l841_84102

theorem right_triangle_leg_length 
  (north_distance : ℝ) 
  (hypotenuse : ℝ) 
  (h1 : north_distance = 10)
  (h2 : hypotenuse = 14.142135623730951) : 
  ∃ west_distance : ℝ, 
    west_distance ^ 2 + north_distance ^ 2 = hypotenuse ^ 2 ∧ 
    west_distance = 10 :=
by sorry

end right_triangle_leg_length_l841_84102


namespace geometric_series_common_ratio_l841_84158

/-- Given a geometric series {a_n} with positive terms, if a_3 = 18 and S_3 = 26, then q = 3 -/
theorem geometric_series_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h_a3 : a 3 = 18)
  (h_S3 : S 3 = 26) :
  ∃ q : ℝ, (∀ n, a (n + 1) = a n * q) ∧ q = 3 := by
  sorry

end geometric_series_common_ratio_l841_84158


namespace probability_perfect_square_three_digit_l841_84146

/-- A three-digit number is a natural number between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A perfect square is a natural number that is the square of an integer. -/
def PerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The count of three-digit numbers that are perfect squares. -/
def CountPerfectSquareThreeDigit : ℕ := 22

/-- The total count of three-digit numbers. -/
def TotalThreeDigitNumbers : ℕ := 900

/-- The probability of a randomly chosen three-digit number being a perfect square is 11/450. -/
theorem probability_perfect_square_three_digit :
  (CountPerfectSquareThreeDigit : ℚ) / (TotalThreeDigitNumbers : ℚ) = 11 / 450 := by
  sorry

end probability_perfect_square_three_digit_l841_84146


namespace rational_number_ordering_l841_84104

theorem rational_number_ordering : -3^2 < -(1/3) ∧ -(1/3) < (-3)^2 ∧ (-3)^2 = |-3^2| := by
  sorry

end rational_number_ordering_l841_84104


namespace smallest_with_18_divisors_l841_84191

/-- The number of positive divisors of a positive integer -/
def numDivisors (n : ℕ+) : ℕ := sorry

/-- Returns true if n is the smallest positive integer with exactly k positive divisors -/
def isSmallestWithDivisors (n k : ℕ+) : Prop :=
  numDivisors n = k ∧ ∀ m : ℕ+, m < n → numDivisors m ≠ k

theorem smallest_with_18_divisors :
  isSmallestWithDivisors 288 18 := by sorry

end smallest_with_18_divisors_l841_84191


namespace current_population_calculation_l841_84141

def initial_population : ℕ := 4399
def bombardment_percentage : ℚ := 1/10
def fear_percentage : ℚ := 1/5

theorem current_population_calculation :
  let remaining_after_bombardment := initial_population - ⌊initial_population * bombardment_percentage⌋
  let current_population := remaining_after_bombardment - ⌊remaining_after_bombardment * fear_percentage⌋
  current_population = 3167 := by sorry

end current_population_calculation_l841_84141


namespace three_heads_in_four_tosses_l841_84168

/-- The probability of getting exactly k successes in n trials with probability p for each trial -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- A fair coin has probability 0.5 of landing heads -/
def fairCoinProbability : ℝ := 0.5

theorem three_heads_in_four_tosses :
  binomialProbability 4 3 fairCoinProbability = 0.25 := by
  sorry

end three_heads_in_four_tosses_l841_84168
