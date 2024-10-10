import Mathlib

namespace absolute_value_inequality_l3582_358225

theorem absolute_value_inequality (x : ℝ) :
  3 ≤ |x - 5| ∧ |x - 5| ≤ 10 ↔ (-5 ≤ x ∧ x ≤ 2) ∨ (8 ≤ x ∧ x ≤ 15) :=
by sorry

end absolute_value_inequality_l3582_358225


namespace fairCoin_threeFlips_oneTwoTails_l3582_358284

/-- Probability of getting k successes in n trials with probability p for each trial -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)

/-- A fair coin has probability 0.5 of landing tails -/
def fairCoinProbability : ℝ := 0.5

/-- The number of consecutive coin flips -/
def numberOfFlips : ℕ := 3

theorem fairCoin_threeFlips_oneTwoTails :
  binomialProbability numberOfFlips 1 fairCoinProbability +
  binomialProbability numberOfFlips 2 fairCoinProbability = 0.375 := by
  sorry

end fairCoin_threeFlips_oneTwoTails_l3582_358284


namespace set_357_forms_triangle_l3582_358220

/-- Triangle inequality theorem: the sum of any two sides must be greater than the third side --/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A set of three line segments can form a triangle if it satisfies the triangle inequality --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set (3, 5, 7) can form a triangle --/
theorem set_357_forms_triangle : can_form_triangle 3 5 7 := by
  sorry

end set_357_forms_triangle_l3582_358220


namespace division_remainder_and_double_l3582_358262

theorem division_remainder_and_double : 
  let dividend := 4509
  let divisor := 98
  let remainder := dividend % divisor
  let doubled_remainder := 2 * remainder
  remainder = 1 ∧ doubled_remainder = 2 :=
by sorry

end division_remainder_and_double_l3582_358262


namespace sphere_in_cube_volume_ratio_l3582_358245

theorem sphere_in_cube_volume_ratio (cube_side : ℝ) (h : cube_side = 8) :
  let sphere_volume := (4 / 3) * Real.pi * (cube_side / 2)^3
  let cube_volume := cube_side^3
  sphere_volume / cube_volume = Real.pi / 6 := by
sorry

end sphere_in_cube_volume_ratio_l3582_358245


namespace cubic_sum_of_roots_l3582_358209

theorem cubic_sum_of_roots (r s : ℝ) : 
  r^2 - 5*r + 6 = 0 → 
  s^2 - 5*s + 6 = 0 → 
  r^3 + s^3 = 35 :=
by
  sorry

end cubic_sum_of_roots_l3582_358209


namespace find_M_l3582_358200

theorem find_M : ∃ (M : ℕ), M > 0 ∧ 18^2 * 45^2 = 15^2 * M^2 ∧ M = 54 := by
  sorry

end find_M_l3582_358200


namespace triangle_side_length_l3582_358210

/-- Given a triangle DEF with side lengths and a median, prove the length of DF. -/
theorem triangle_side_length (DE EF DM : ℝ) (hDE : DE = 7) (hEF : EF = 10) (hDM : DM = 5) :
  ∃ (DF : ℝ), DF = Real.sqrt 149 :=
sorry

end triangle_side_length_l3582_358210


namespace complex_expression_evaluation_l3582_358272

def i : ℂ := Complex.I

theorem complex_expression_evaluation : i * (1 - 2*i) = 2 + i := by
  sorry

end complex_expression_evaluation_l3582_358272


namespace freshmen_in_liberal_arts_l3582_358275

theorem freshmen_in_liberal_arts (total_students : ℝ) (freshmen_percent : ℝ) 
  (psych_majors_percent : ℝ) (freshmen_psych_liberal_arts_percent : ℝ) :
  freshmen_percent = 80 →
  psych_majors_percent = 50 →
  freshmen_psych_liberal_arts_percent = 24 →
  (freshmen_psych_liberal_arts_percent * total_students) / 
    (psych_majors_percent / 100 * freshmen_percent * total_students / 100) = 60 / 100 := by
  sorry

end freshmen_in_liberal_arts_l3582_358275


namespace proper_subsets_of_abc_l3582_358237

def S : Set (Set Char) := {{'a', 'b', 'c'}}

theorem proper_subsets_of_abc :
  {s : Set Char | s ⊂ {'a', 'b', 'c'}} =
  {∅, {'a'}, {'b'}, {'c'}, {'a', 'b'}, {'a', 'c'}, {'b', 'c'}} := by
  sorry

end proper_subsets_of_abc_l3582_358237


namespace zoo_ticket_sales_l3582_358224

/-- Calculates the total money made from ticket sales at a zoo -/
theorem zoo_ticket_sales (total_people : ℕ) (adult_price kid_price : ℕ) (num_kids : ℕ) : 
  total_people = 254 → 
  adult_price = 28 → 
  kid_price = 12 → 
  num_kids = 203 → 
  (total_people - num_kids) * adult_price + num_kids * kid_price = 3864 := by
sorry

end zoo_ticket_sales_l3582_358224


namespace rectangular_box_diagonal_l3582_358285

/-- Proves that a rectangular box with given surface area and edge length sum has a specific longest diagonal --/
theorem rectangular_box_diagonal (x y z : ℝ) : 
  (2 * (x*y + y*z + z*x) = 150) →  -- Total surface area
  (4 * (x + y + z) = 60) →         -- Sum of all edge lengths
  Real.sqrt (x^2 + y^2 + z^2) = 5 * Real.sqrt 3 := by sorry

end rectangular_box_diagonal_l3582_358285


namespace skew_sufficient_not_necessary_for_non_intersecting_l3582_358253

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l₁ l₂ : Line3D) : Prop :=
  sorry

/-- Two lines intersect if they share a common point -/
def intersect (l₁ l₂ : Line3D) : Prop :=
  sorry

/-- Main theorem: Skew lines are sufficient but not necessary for non-intersecting lines -/
theorem skew_sufficient_not_necessary_for_non_intersecting :
  (∀ l₁ l₂ : Line3D, are_skew l₁ l₂ → ¬(intersect l₁ l₂)) ∧
  (∃ l₁ l₂ : Line3D, ¬(intersect l₁ l₂) ∧ ¬(are_skew l₁ l₂)) :=
by sorry

end skew_sufficient_not_necessary_for_non_intersecting_l3582_358253


namespace A_inter_B_l3582_358299

def A : Set ℤ := {-1, 0, 1}

def B : Set ℤ := {y | ∃ x ∈ A, y = x^2}

theorem A_inter_B : A ∩ B = {0, 1} := by sorry

end A_inter_B_l3582_358299


namespace perpendicular_vectors_l3582_358295

def vector_a : Fin 2 → ℝ := ![(-2), 3]
def vector_b (m : ℝ) : Fin 2 → ℝ := ![3, m]

theorem perpendicular_vectors (m : ℝ) :
  (vector_a 0 * vector_b m 0 + vector_a 1 * vector_b m 1 = 0) → m = 2 := by
  sorry

end perpendicular_vectors_l3582_358295


namespace bags_bought_l3582_358287

def crayonPacks : ℕ := 5
def crayonPrice : ℚ := 5
def bookCount : ℕ := 10
def bookPrice : ℚ := 5
def calculatorCount : ℕ := 3
def calculatorPrice : ℚ := 5
def bookDiscount : ℚ := 0.2
def salesTax : ℚ := 0.05
def initialMoney : ℚ := 200
def bagPrice : ℚ := 10

def totalCost : ℚ :=
  crayonPacks * crayonPrice +
  bookCount * bookPrice * (1 - bookDiscount) +
  calculatorCount * calculatorPrice

def finalCost : ℚ := totalCost * (1 + salesTax)

def change : ℚ := initialMoney - finalCost

theorem bags_bought (h : change ≥ 0) : ⌊change / bagPrice⌋ = 11 := by
  sorry

#eval ⌊change / bagPrice⌋

end bags_bought_l3582_358287


namespace platform_length_l3582_358211

/-- Calculates the length of a platform given train parameters -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 750)
  (h2 : time_platform = 97)
  (h3 : time_pole = 90) :
  ∃ (platform_length : ℝ), abs (platform_length - 58.33) < 0.01 :=
by
  sorry

end platform_length_l3582_358211


namespace cubic_identities_l3582_358288

/-- Prove algebraic identities for cubic expressions -/
theorem cubic_identities (x y : ℝ) : 
  ((x + y) * (x^2 - x*y + y^2) = x^3 + y^3) ∧
  ((x + 3) * (x^2 - 3*x + 9) = x^3 + 27) ∧
  ((x - 1) * (x^2 + x + 1) = x^3 - 1) ∧
  ((2*x - 3) * (4*x^2 + 6*x + 9) = 8*x^3 - 27) := by
  sorry


end cubic_identities_l3582_358288


namespace oliver_good_games_l3582_358213

theorem oliver_good_games (total_games bad_games : ℕ) 
  (h1 : total_games = 11) 
  (h2 : bad_games = 5) : 
  total_games - bad_games = 6 := by
  sorry

end oliver_good_games_l3582_358213


namespace decimal_365_to_octal_l3582_358278

/-- Converts a natural number to its octal representation as a list of digits -/
def toOctal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: toOctal (n / 8)

/-- Theorem: The decimal number 365 is equal to 555₈ in octal representation -/
theorem decimal_365_to_octal :
  toOctal 365 = [5, 5, 5] := by
  sorry

end decimal_365_to_octal_l3582_358278


namespace square_difference_equality_l3582_358244

theorem square_difference_equality : (45 + 15)^2 - (45^2 + 15^2) = 1350 := by
  sorry

end square_difference_equality_l3582_358244


namespace jane_sum_minus_liam_sum_l3582_358208

def jane_list : List Nat := List.range 50

def replace_3_with_2 (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def liam_list : List Nat := jane_list.map replace_3_with_2

theorem jane_sum_minus_liam_sum : 
  jane_list.sum - liam_list.sum = 105 := by sorry

end jane_sum_minus_liam_sum_l3582_358208


namespace inequality_proof_l3582_358249

theorem inequality_proof (a : ℝ) : 2 * a^4 + 2 * a^2 - 1 ≥ (3/2) * (a^2 + a - 1) := by
  sorry

end inequality_proof_l3582_358249


namespace smaller_number_l3582_358251

theorem smaller_number (L S : ℕ) (hL : L > S) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : S = 270 := by
  sorry

end smaller_number_l3582_358251


namespace eve_ran_distance_l3582_358248

/-- The distance Eve walked in miles -/
def distance_walked : ℝ := 0.6

/-- The additional distance Eve ran compared to what she walked, in miles -/
def additional_distance : ℝ := 0.1

/-- The total distance Eve ran in miles -/
def distance_ran : ℝ := distance_walked + additional_distance

theorem eve_ran_distance : distance_ran = 0.7 := by
  sorry

end eve_ran_distance_l3582_358248


namespace bisecting_line_sum_l3582_358240

/-- Triangle DEF with vertices D(0, 10), E(4, 0), and F(10, 0) -/
structure Triangle :=
  (D : ℝ × ℝ)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)

/-- A line defined by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

/-- Checks if a line bisects the area of a triangle -/
def bisects_area (t : Triangle) (l : Line) : Prop :=
  sorry

/-- The specific triangle DEF from the problem -/
def triangle_DEF : Triangle :=
  { D := (0, 10),
    E := (4, 0),
    F := (10, 0) }

/-- Main theorem: The line through E that bisects the area of triangle DEF
    has a slope and y-intercept whose sum is -15 -/
theorem bisecting_line_sum (l : Line) :
  bisects_area triangle_DEF l → l.slope + l.y_intercept = -15 :=
by sorry

end bisecting_line_sum_l3582_358240


namespace greatest_power_of_two_factor_l3582_358215

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), (2^k : ℤ) ∣ (10^1004 - 4^502) ∧ 
  ∀ (m : ℕ), (2^m : ℤ) ∣ (10^1004 - 4^502) → m ≤ k :=
by
  use 1007
  sorry

#eval 1007  -- This will output the answer

end greatest_power_of_two_factor_l3582_358215


namespace sum_of_three_reals_l3582_358279

theorem sum_of_three_reals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + 2*(y-1)*(z-1) = 85)
  (eq2 : y^2 + 2*(z-1)*(x-1) = 84)
  (eq3 : z^2 + 2*(x-1)*(y-1) = 89) :
  x + y + z = 18 := by
sorry

end sum_of_three_reals_l3582_358279


namespace concatNaturalsDecimal_irrational_l3582_358234

/-- The infinite decimal formed by concatenating all natural numbers in order after the decimal point -/
def concatNaturalsDecimal : ℝ :=
  sorry  -- Definition of the decimal (implementation details omitted)

/-- The infinite decimal formed by concatenating all natural numbers in order after the decimal point is irrational -/
theorem concatNaturalsDecimal_irrational : Irrational concatNaturalsDecimal := by
  sorry

end concatNaturalsDecimal_irrational_l3582_358234


namespace sum_positive_given_difference_abs_l3582_358294

theorem sum_positive_given_difference_abs (a b : ℝ) : a - |b| > 0 → b + a > 0 := by
  sorry

end sum_positive_given_difference_abs_l3582_358294


namespace ellipse_line_theorem_l3582_358250

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (x : ℝ) : Prop := x = -2

-- Define a line passing through a point
def line_through_point (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the perpendicular bisector of a line segment
def perpendicular_bisector (k : ℝ) (x y : ℝ) : Prop := 
  y + k / (1 + 2*k^2) = -(1/k) * (x - 2*k^2 / (1 + 2*k^2))

-- Define the theorem
theorem ellipse_line_theorem (k : ℝ) (x₁ y₁ x₂ y₂ xp yp xc yc : ℝ) : 
  ellipse x₁ y₁ → 
  ellipse x₂ y₂ → 
  line_through_point k x₁ y₁ → 
  line_through_point k x₂ y₂ → 
  perpendicular_bisector k xp yp → 
  perpendicular_bisector k xc yc → 
  line_l xp → 
  (xc - 1)^2 + yc^2 = ((x₂ - x₁)^2 + (y₂ - y₁)^2) / 4 → 
  (xp - xc)^2 + (yp - yc)^2 = 4 * ((x₂ - x₁)^2 + (y₂ - y₁)^2) → 
  (k = 1 ∨ k = -1) :=
sorry

end ellipse_line_theorem_l3582_358250


namespace symmetric_point_wrt_x_axis_l3582_358246

/-- Given a point P(-3, 1), its symmetric point with respect to the x-axis has coordinates (-3, -1) -/
theorem symmetric_point_wrt_x_axis :
  let P : ℝ × ℝ := (-3, 1)
  let symmetric_point := (P.1, -P.2)
  symmetric_point = (-3, -1) := by sorry

end symmetric_point_wrt_x_axis_l3582_358246


namespace arithmetic_progression_possible_n_values_l3582_358274

theorem arithmetic_progression_possible_n_values : 
  ∃! (S : Finset ℕ), 
    S.Nonempty ∧ 
    (∀ n ∈ S, n > 1) ∧
    (S.card = 4) ∧
    (∀ n ∈ S, ∃ a : ℤ, 120 = n * (a + (3 * n / 2 : ℚ) - (3 / 2 : ℚ))) ∧
    (∀ n : ℕ, n > 1 → (∃ a : ℤ, 120 = n * (a + (3 * n / 2 : ℚ) - (3 / 2 : ℚ))) → n ∈ S) :=
by sorry

end arithmetic_progression_possible_n_values_l3582_358274


namespace final_elevation_proof_l3582_358270

def calculate_final_elevation (initial_elevation : ℝ) 
                               (rate1 rate2 rate3 : ℝ) 
                               (time1 time2 time3 : ℝ) : ℝ :=
  initial_elevation - (rate1 * time1 + rate2 * time2 + rate3 * time3)

theorem final_elevation_proof (initial_elevation : ℝ) 
                              (rate1 rate2 rate3 : ℝ) 
                              (time1 time2 time3 : ℝ) :
  calculate_final_elevation initial_elevation rate1 rate2 rate3 time1 time2 time3 =
  initial_elevation - (rate1 * time1 + rate2 * time2 + rate3 * time3) :=
by
  sorry

#eval calculate_final_elevation 400 10 15 12 5 3 6

end final_elevation_proof_l3582_358270


namespace problem_1_proof_l3582_358233

theorem problem_1_proof : (1 : ℝ) - 1^2 + (64 : ℝ)^(1/3) - (-2) * (9 : ℝ)^(1/2) = 9 := by
  sorry

end problem_1_proof_l3582_358233


namespace chairs_to_hall_l3582_358228

theorem chairs_to_hall (num_students : ℕ) (chairs_per_trip : ℕ) (num_trips : ℕ) :
  num_students = 5 →
  chairs_per_trip = 5 →
  num_trips = 10 →
  num_students * chairs_per_trip * num_trips = 250 :=
by
  sorry

end chairs_to_hall_l3582_358228


namespace parabolas_intersection_l3582_358297

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℚ := {-5/3, 0}

/-- First parabola function -/
def f (x : ℚ) : ℚ := 3 * x^2 - 4 * x + 2

/-- Second parabola function -/
def g (x : ℚ) : ℚ := 9 * x^2 + 6 * x + 2

/-- Theorem stating that the two parabolas intersect at the given points -/
theorem parabolas_intersection :
  ∀ x ∈ intersection_x, f x = g x ∧ 
  (x = -5/3 → f x = 17) ∧ 
  (x = 0 → f x = 2) :=
sorry

end parabolas_intersection_l3582_358297


namespace chair_cost_l3582_358289

def total_spent : ℕ := 56
def table_cost : ℕ := 34
def num_chairs : ℕ := 2

theorem chair_cost (chair_cost : ℕ) 
  (h1 : chair_cost * num_chairs + table_cost = total_spent) 
  (h2 : chair_cost > 0) : chair_cost = 11 := by
  sorry

end chair_cost_l3582_358289


namespace find_number_l3582_358280

theorem find_number : ∃ x : ℝ, (0.15 * 40 = 0.25 * x + 2) ∧ x = 16 := by
  sorry

end find_number_l3582_358280


namespace annual_pension_correct_l3582_358298

/-- Represents the annual pension calculation for an employee -/
noncomputable def annual_pension 
  (a b p q : ℝ) 
  (h1 : b ≠ a) : ℝ :=
  (q * a^2 - p * b^2)^2 / (4 * (p * b - q * a)^2)

/-- Theorem stating the annual pension calculation is correct -/
theorem annual_pension_correct 
  (a b p q : ℝ) 
  (h1 : b ≠ a)
  (h2 : ∃ (k x : ℝ), 
    k * (x - a)^2 = k * x^2 - p ∧ 
    k * (x + b)^2 = k * x^2 + q) :
  ∃ (k x : ℝ), k * x^2 = annual_pension a b p q h1 := by
  sorry

end annual_pension_correct_l3582_358298


namespace unique_twelve_times_digit_sum_l3582_358293

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem unique_twelve_times_digit_sum :
  ∀ n : ℕ, n > 0 → (n = 12 * sum_of_digits n ↔ n = 108) := by
  sorry

end unique_twelve_times_digit_sum_l3582_358293


namespace a_squared_b_plus_ab_squared_equals_four_l3582_358222

theorem a_squared_b_plus_ab_squared_equals_four :
  let a : ℝ := 2 + Real.sqrt 3
  let b : ℝ := 2 - Real.sqrt 3
  a^2 * b + a * b^2 = 4 := by
sorry

end a_squared_b_plus_ab_squared_equals_four_l3582_358222


namespace geometric_series_sum_l3582_358236

theorem geometric_series_sum : 
  let a₁ : ℚ := 1 / 4
  let r : ℚ := -1 / 4
  let n : ℕ := 6
  let series_sum := a₁ * (1 - r^n) / (1 - r)
  series_sum = 81 / 405 := by
sorry

end geometric_series_sum_l3582_358236


namespace max_value_of_a_l3582_358281

theorem max_value_of_a (a b c : ℝ) : 
  a^2 - b*c - 8*a + 7 = 0 → 
  b^2 + c^2 + b*c - 6*a + 6 = 0 → 
  a ≤ 9 ∧ ∃ b c : ℝ, a^2 - b*c - 8*a + 7 = 0 ∧ b^2 + c^2 + b*c - 6*a + 6 = 0 ∧ a = 9 :=
by sorry

end max_value_of_a_l3582_358281


namespace unit_cost_decrease_l3582_358257

/-- Regression equation for unit product cost -/
def regression_equation (x : ℝ) : ℝ := 356 - 1.5 * x

/-- Theorem stating the relationship between output and unit product cost -/
theorem unit_cost_decrease (x : ℝ) :
  regression_equation (x + 1) = regression_equation x - 1.5 := by
  sorry

end unit_cost_decrease_l3582_358257


namespace quadratic_inequality_solution_set_l3582_358202

theorem quadratic_inequality_solution_set :
  {x : ℝ | 3 * x^2 + 2 * x - 5 < 8} = {x : ℝ | -2 * Real.sqrt 10 / 6 - 1 / 3 < x ∧ x < 2 * Real.sqrt 10 / 6 - 1 / 3} :=
by sorry

end quadratic_inequality_solution_set_l3582_358202


namespace people_in_virginia_l3582_358214

/-- The number of people landing in Virginia given the initial passengers, layover changes, and crew members. -/
def peopleInVirginia (initialPassengers : ℕ) (texasOff texasOn ncOff ncOn crewMembers : ℕ) : ℕ :=
  initialPassengers - texasOff + texasOn - ncOff + ncOn + crewMembers

/-- Theorem stating that the number of people landing in Virginia is 67. -/
theorem people_in_virginia :
  peopleInVirginia 124 58 24 47 14 10 = 67 := by
  sorry

end people_in_virginia_l3582_358214


namespace largest_calculation_l3582_358255

theorem largest_calculation :
  let a := 2 + 0 + 1 + 8
  let b := 2 * 0 + 1 + 8
  let c := 2 + 0 * 1 + 8
  let d := 2 + 0 + 1 * 8
  let e := 2 * 0 + 1 * 8
  (a ≥ b) ∧ (a ≥ c) ∧ (a ≥ d) ∧ (a ≥ e) := by
  sorry

end largest_calculation_l3582_358255


namespace b_present_age_l3582_358231

/-- Given two people A and B, prove that B's present age is 34 years -/
theorem b_present_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- In 10 years, A will be twice as old as B was 10 years ago
  (a = b + 4) →              -- A is now 4 years older than B
  b = 34 := by
sorry

end b_present_age_l3582_358231


namespace factor_expression_l3582_358201

theorem factor_expression (x : ℝ) : 75 * x^13 + 450 * x^26 = 75 * x^13 * (1 + 6 * x^13) := by
  sorry

end factor_expression_l3582_358201


namespace geometric_sequence_property_l3582_358269

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  (∀ n : ℕ, a n > 0) → a 4 * a 10 = 16 → a 7 = 4 := by
  sorry

end geometric_sequence_property_l3582_358269


namespace january_salary_is_5300_l3582_358247

/-- Represents monthly salaries -/
structure MonthlySalaries where
  J : ℕ  -- January
  F : ℕ  -- February
  M : ℕ  -- March
  A : ℕ  -- April
  Ma : ℕ -- May
  Ju : ℕ -- June

/-- Theorem stating the conditions and the result to be proved -/
theorem january_salary_is_5300 (s : MonthlySalaries) : 
  (s.J + s.F + s.M + s.A) / 4 = 8000 →
  (s.F + s.M + s.A + s.Ma) / 4 = 8300 →
  (s.M + s.A + s.Ma + s.Ju) / 4 = 8600 →
  s.Ma = 6500 →
  s.J = 5300 := by
  sorry

#check january_salary_is_5300

end january_salary_is_5300_l3582_358247


namespace inequality_solution_set_l3582_358261

theorem inequality_solution_set (x : ℝ) : 
  x^6 - (x + 2) > (x + 2)^3 - x^2 ↔ x < -1 ∨ x > 2 := by sorry

end inequality_solution_set_l3582_358261


namespace integer_triple_divisibility_l3582_358243

theorem integer_triple_divisibility :
  ∀ a b c : ℤ,
  (1 < a ∧ a < b ∧ b < c) →
  ((a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) →
  ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by sorry

end integer_triple_divisibility_l3582_358243


namespace square_equality_l3582_358216

theorem square_equality (a b : ℝ) : a = b → a^2 = b^2 := by
  sorry

end square_equality_l3582_358216


namespace linear_function_not_in_second_quadrant_l3582_358282

-- Define the linear function
def f (x : ℝ) : ℝ := 2 * x - 3

-- Define what it means for a point to be in the second quadrant
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem linear_function_not_in_second_quadrant :
  ¬ ∃ x : ℝ, in_second_quadrant x (f x) :=
sorry

end linear_function_not_in_second_quadrant_l3582_358282


namespace stream_speed_l3582_358276

theorem stream_speed (rowing_speed : ℝ) (total_time : ℝ) (distance : ℝ) (stream_speed : ℝ) : 
  rowing_speed = 10 →
  total_time = 5 →
  distance = 24 →
  (distance / (rowing_speed - stream_speed) + distance / (rowing_speed + stream_speed) = total_time) →
  stream_speed = 2 := by
sorry

end stream_speed_l3582_358276


namespace train_length_calculation_l3582_358205

/-- The length of a train in meters. -/
def train_length : ℝ := 1500

/-- The time in seconds it takes for the train to cross a tree. -/
def time_tree : ℝ := 120

/-- The time in seconds it takes for the train to pass a platform. -/
def time_platform : ℝ := 160

/-- The length of the platform in meters. -/
def platform_length : ℝ := 500

theorem train_length_calculation :
  train_length = 1500 ∧
  (train_length / time_tree = (train_length + platform_length) / time_platform) :=
by sorry

end train_length_calculation_l3582_358205


namespace largest_floor_value_l3582_358259

/-- A positive real number that rounds to 20 -/
def A : ℝ := sorry

/-- A positive real number that rounds to 23 -/
def B : ℝ := sorry

/-- A rounds to 20 -/
axiom hA : 19.5 ≤ A ∧ A < 20.5

/-- B rounds to 23 -/
axiom hB : 22.5 ≤ B ∧ B < 23.5

/-- A and B are positive -/
axiom pos_A : A > 0
axiom pos_B : B > 0

theorem largest_floor_value :
  ∃ (x : ℝ) (y : ℝ), 19.5 ≤ x ∧ x < 20.5 ∧ 22.5 ≤ y ∧ y < 23.5 ∧
  ∀ (a : ℝ) (b : ℝ), 19.5 ≤ a ∧ a < 20.5 ∧ 22.5 ≤ b ∧ b < 23.5 →
  ⌊100 * x / y⌋ ≥ ⌊100 * a / b⌋ ∧ ⌊100 * x / y⌋ = 91 :=
sorry

end largest_floor_value_l3582_358259


namespace checkers_tie_fraction_l3582_358226

theorem checkers_tie_fraction (ben_win_rate sara_win_rate : ℚ) 
  (h1 : ben_win_rate = 2/5)
  (h2 : sara_win_rate = 1/4) : 
  1 - (ben_win_rate + sara_win_rate) = 7/20 := by
sorry

end checkers_tie_fraction_l3582_358226


namespace class_size_l3582_358242

theorem class_size (S : ℕ) 
  (h1 : S / 3 + S * 2 / 5 + 12 = S) : S = 45 := by
  sorry

end class_size_l3582_358242


namespace student_age_problem_l3582_358207

theorem student_age_problem (total_students : ℕ) (total_average_age : ℕ) 
  (group1_students : ℕ) (group1_average_age : ℕ) 
  (group2_students : ℕ) (group2_average_age : ℕ) :
  total_students = 20 →
  total_average_age = 20 →
  group1_students = 9 →
  group1_average_age = 11 →
  group2_students = 10 →
  group2_average_age = 24 →
  (total_students * total_average_age - 
   (group1_students * group1_average_age + group2_students * group2_average_age)) = 61 :=
by sorry

end student_age_problem_l3582_358207


namespace find_number_l3582_358268

theorem find_number : ∃ N : ℕ,
  (N = (555 + 445) * (2 * (555 - 445)) + 30) ∧ 
  (N = 220030) := by
  sorry

end find_number_l3582_358268


namespace roller_coaster_cars_l3582_358229

theorem roller_coaster_cars (n : ℕ) (h : n > 0) :
  (n - 1 : ℚ) / n = 1/2 ↔ n = 2 :=
by sorry

end roller_coaster_cars_l3582_358229


namespace probability_proof_l3582_358230

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def total_balls : ℕ := white_balls + black_balls

def probability_one_white_one_black : ℚ :=
  (white_balls * black_balls : ℚ) / (total_balls * (total_balls - 1) / 2)

theorem probability_proof :
  probability_one_white_one_black = 56 / 105 := by
  sorry

end probability_proof_l3582_358230


namespace race_time_difference_l3582_358219

/-- The time difference between two runners in a race -/
def time_difference (distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  distance * speed2 - distance * speed1

/-- Proof of the time difference in the race -/
theorem race_time_difference :
  let malcolm_speed : ℝ := 7
  let joshua_speed : ℝ := 8
  let race_distance : ℝ := 15
  time_difference race_distance malcolm_speed joshua_speed = 15 := by
  sorry

end race_time_difference_l3582_358219


namespace complex_number_properties_l3582_358292

def z (m : ℝ) : ℂ := Complex.mk (m^2 - 3*m + 2) (m^2 - 1)

theorem complex_number_properties :
  (∀ m : ℝ, z m = 0 ↔ m = 1) ∧
  (∀ m : ℝ, (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2) ∧
  (∀ m : ℝ, (z m).re < 0 ∧ (z m).im > 0 ↔ 1 < m ∧ m < 2) :=
by sorry

end complex_number_properties_l3582_358292


namespace custom_op_nested_l3582_358271

/-- Custom binary operation ⊗ -/
def custom_op (x y : ℝ) : ℝ := x^3 - y^2 + x

/-- Theorem stating the result of k ⊗ (k ⊗ k) -/
theorem custom_op_nested (k : ℝ) : custom_op k (custom_op k k) = -k^6 + 2*k^5 - 3*k^4 + 3*k^3 - k^2 + 2*k := by
  sorry

end custom_op_nested_l3582_358271


namespace max_individual_award_l3582_358267

theorem max_individual_award 
  (total_prize : ℕ) 
  (num_winners : ℕ) 
  (min_award : ℕ) 
  (h1 : total_prize = 2500)
  (h2 : num_winners = 25)
  (h3 : min_award = 50)
  (h4 : (3 : ℚ) / 5 * total_prize = (2 : ℚ) / 5 * num_winners * max_award)
  : ∃ max_award : ℕ, max_award = 1300 := by
  sorry

end max_individual_award_l3582_358267


namespace geometric_progression_solution_l3582_358212

/-- Given three real numbers form a geometric progression, prove that the first term is 15 + 5√5 --/
theorem geometric_progression_solution (x : ℝ) : 
  (2*x + 10)^2 = x * (5*x + 10) → x = 15 + 5 * Real.sqrt 5 := by
  sorry

end geometric_progression_solution_l3582_358212


namespace ball_placement_count_ball_placement_proof_l3582_358266

theorem ball_placement_count : ℕ :=
  let n_balls : ℕ := 5
  let n_boxes : ℕ := 4
  let ways_to_divide : ℕ := Nat.choose n_balls (n_balls - n_boxes + 1)
  let ways_to_arrange : ℕ := Nat.factorial n_boxes
  ways_to_divide * ways_to_arrange

theorem ball_placement_proof :
  ball_placement_count = 240 := by
  sorry

end ball_placement_count_ball_placement_proof_l3582_358266


namespace excess_calories_james_james_excess_calories_l3582_358218

/-- Calculates the excess calories James ate after eating Cheezits and going for a run -/
theorem excess_calories_james (bags : ℕ) (ounces_per_bag : ℕ) (calories_per_ounce : ℕ) 
  (run_duration : ℕ) (calories_burned_per_minute : ℕ) : ℕ :=
  let total_ounces := bags * ounces_per_bag
  let total_calories_consumed := total_ounces * calories_per_ounce
  let total_calories_burned := run_duration * calories_burned_per_minute
  total_calories_consumed - total_calories_burned

/-- Proves that James ate 420 excess calories -/
theorem james_excess_calories : 
  excess_calories_james 3 2 150 40 12 = 420 := by
  sorry

end excess_calories_james_james_excess_calories_l3582_358218


namespace seungho_original_marble_difference_l3582_358291

/-- Proves that Seungho originally had 1023 more marbles than Hyukjin -/
theorem seungho_original_marble_difference (s h : ℕ) : 
  s - 273 = (h + 273) + 477 → s = h + 1023 := by
  sorry

end seungho_original_marble_difference_l3582_358291


namespace complex_power_eight_l3582_358265

theorem complex_power_eight (z : ℂ) : z = (-Real.sqrt 3 + I) / 2 → z^8 = -1/2 - (Real.sqrt 3 / 2) * I := by
  sorry

end complex_power_eight_l3582_358265


namespace transformer_min_current_load_l3582_358260

def number_of_units : ℕ := 3
def running_current_per_unit : ℕ := 40
def starting_current_multiplier : ℕ := 2

theorem transformer_min_current_load :
  let total_running_current := number_of_units * running_current_per_unit
  let min_starting_current := starting_current_multiplier * total_running_current
  min_starting_current = 240 := by
  sorry

end transformer_min_current_load_l3582_358260


namespace target_number_position_l3582_358204

/-- Represents a position in the spiral matrix -/
structure Position where
  row : Nat
  col : Nat

/-- Fills a square matrix in a clockwise spiral order -/
def spiralFill (n : Nat) : Nat → Position
  | k => sorry  -- Implementation details omitted

/-- The size of our spiral matrix -/
def matrixSize : Nat := 100

/-- The number we're looking for in the spiral matrix -/
def targetNumber : Nat := 2018

/-- The expected position of the target number -/
def expectedPosition : Position := ⟨34, 95⟩

theorem target_number_position :
  spiralFill matrixSize targetNumber = expectedPosition := by sorry

end target_number_position_l3582_358204


namespace circle_area_from_circumference_l3582_358239

theorem circle_area_from_circumference (k : ℝ) : 
  (∃ (r : ℝ), 2 * π * r = 30 * π ∧ π * r^2 = k * π) → k = 225 := by
  sorry

end circle_area_from_circumference_l3582_358239


namespace bakery_problem_l3582_358235

/-- The number of ways to distribute additional items into bins, given a minimum per bin -/
def distribute_items (total_items : ℕ) (num_bins : ℕ) (min_per_bin : ℕ) : ℕ :=
  Nat.choose (total_items - num_bins * min_per_bin + num_bins - 1) (num_bins - 1)

theorem bakery_problem :
  distribute_items 10 4 2 = 10 := by
  sorry

end bakery_problem_l3582_358235


namespace sqrt_equation_solution_l3582_358217

theorem sqrt_equation_solution (x : ℝ) :
  x > 16 →
  (Real.sqrt (x - 8 * Real.sqrt (x - 16)) + 4 = Real.sqrt (x + 8 * Real.sqrt (x - 16)) - 4) ↔
  x ≥ 32 := by
sorry

end sqrt_equation_solution_l3582_358217


namespace stability_comparison_l3582_358264

/-- Represents a student's test performance -/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Determines if the first student's performance is more stable than the second -/
def more_stable (student1 student2 : StudentPerformance) : Prop :=
  student1.variance < student2.variance

theorem stability_comparison 
  (student_A student_B : StudentPerformance)
  (h_same_average : student_A.average_score = student_B.average_score)
  (h_A_variance : student_A.variance = 51)
  (h_B_variance : student_B.variance = 12) :
  more_stable student_B student_A :=
sorry

end stability_comparison_l3582_358264


namespace sequence_classification_l3582_358227

/-- Given a sequence {a_n} where the sum of the first n terms S_n = a^n - 2 (a is a constant, a ≠ 0),
    the sequence {a_n} forms either an arithmetic sequence or a geometric sequence from the second term onwards. -/
theorem sequence_classification (a : ℝ) (h_a : a ≠ 0) :
  let S : ℕ → ℝ := λ n => a ^ n - 2
  let a_seq : ℕ → ℝ := λ n => S n - S (n - 1)
  (∀ n : ℕ, n ≥ 2 → ∃ d : ℝ, a_seq (n + 1) - a_seq n = d) ∨
  (∀ n : ℕ, n ≥ 2 → ∃ r : ℝ, a_seq (n + 1) / a_seq n = r) :=
by sorry

end sequence_classification_l3582_358227


namespace draw_points_value_l3582_358256

/-- Represents the points system in a football competition --/
structure PointSystem where
  victory_points : ℕ
  draw_points : ℕ
  defeat_points : ℕ

/-- Represents the state of a team in the competition --/
structure TeamState where
  total_matches : ℕ
  matches_played : ℕ
  current_points : ℕ
  target_points : ℕ
  min_victories : ℕ

/-- The theorem to prove --/
theorem draw_points_value (ps : PointSystem) (ts : TeamState) : 
  ps.victory_points = 3 ∧ 
  ps.defeat_points = 0 ∧
  ts.total_matches = 20 ∧ 
  ts.matches_played = 5 ∧ 
  ts.current_points = 14 ∧ 
  ts.target_points = 40 ∧
  ts.min_victories = 6 →
  ps.draw_points = 2 := by
  sorry


end draw_points_value_l3582_358256


namespace weight_replacement_l3582_358241

theorem weight_replacement (n : ℕ) (avg_increase weight_new : ℝ) :
  n = 8 →
  avg_increase = 2.5 →
  weight_new = 70 →
  weight_new - n * avg_increase = 50 :=
by sorry

end weight_replacement_l3582_358241


namespace arithmetic_progression_zero_term_l3582_358203

/-- An arithmetic progression with a term equal to zero -/
theorem arithmetic_progression_zero_term
  (a : ℕ → ℝ)  -- The arithmetic progression
  (n m : ℕ)    -- Indices of the given terms
  (h : a (2 * n) / a (2 * m) = -1)  -- Given condition
  : ∃ k, a k = 0 ∧ k = n + m := by
  sorry

end arithmetic_progression_zero_term_l3582_358203


namespace polynomial_division_remainder_l3582_358232

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^2 - 22 * X + 70 = (X - 7) * q + 63 := by
  sorry

end polynomial_division_remainder_l3582_358232


namespace train_crossing_time_l3582_358254

/-- The time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 150 ∧ 
  train_speed = 85 * (1000 / 3600) ∧ 
  man_speed = 5 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 6 := by
  sorry


end train_crossing_time_l3582_358254


namespace opposite_face_of_A_is_B_l3582_358263

/-- Represents the letters on the cube faces -/
inductive CubeLetter
  | A | B | V | G | D | E

/-- Represents a face of the cube -/
structure CubeFace where
  letter : CubeLetter

/-- Represents the cube -/
structure Cube where
  faces : Finset CubeFace
  face_count : faces.card = 6

/-- Represents a perspective of the cube showing three visible faces -/
structure CubePerspective where
  visible_faces : Finset CubeFace
  visible_count : visible_faces.card = 3

/-- Defines the opposite face relation -/
def opposite_face (c : Cube) (f1 f2 : CubeFace) : Prop :=
  f1 ∈ c.faces ∧ f2 ∈ c.faces ∧ f1 ≠ f2 ∧ 
  ∀ (p : CubePerspective), ¬(f1 ∈ p.visible_faces ∧ f2 ∈ p.visible_faces)

theorem opposite_face_of_A_is_B 
  (c : Cube) 
  (p1 p2 p3 : CubePerspective) 
  (hA : ∃ (fA : CubeFace), fA ∈ c.faces ∧ fA.letter = CubeLetter.A)
  (hB : ∃ (fB : CubeFace), fB ∈ c.faces ∧ fB.letter = CubeLetter.B)
  (h_perspectives : 
    (∃ (f1 f2 : CubeFace), f1 ∈ p1.visible_faces ∧ f2 ∈ p1.visible_faces ∧ 
      f1.letter = CubeLetter.A ∧ f2.letter = CubeLetter.B) ∧
    (∃ (f1 f2 : CubeFace), f1 ∈ p2.visible_faces ∧ f2 ∈ p2.visible_faces ∧ 
      f1.letter = CubeLetter.B) ∧
    (∃ (f1 f2 : CubeFace), f1 ∈ p3.visible_faces ∧ f2 ∈ p3.visible_faces ∧ 
      f1.letter = CubeLetter.A)) :
  ∃ (fA fB : CubeFace), 
    fA.letter = CubeLetter.A ∧ 
    fB.letter = CubeLetter.B ∧ 
    opposite_face c fA fB :=
  sorry

end opposite_face_of_A_is_B_l3582_358263


namespace at_least_fifteen_equal_differences_l3582_358258

theorem at_least_fifteen_equal_differences
  (a : Fin 100 → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_bounded : ∀ i, 1 ≤ a i ∧ a i ≤ 400)
  (h_increasing : ∀ i j, i < j → a i < a j) :
  ∃ (v : ℕ) (s : Finset (Fin 99)),
    s.card ≥ 15 ∧ ∀ i ∈ s, a (i + 1) - a i = v :=
sorry

end at_least_fifteen_equal_differences_l3582_358258


namespace domain_transformation_l3582_358286

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.Ioo (1/3) 1

-- Define the domain of f(3^x)
def domain_f_exp : Set ℝ := Set.Ico (-1) 0

-- Theorem statement
theorem domain_transformation (h : ∀ x ∈ domain_f, f x ≠ 0) :
  ∀ x, f (3^x) ≠ 0 ↔ x ∈ domain_f_exp :=
sorry

end domain_transformation_l3582_358286


namespace key_pairs_and_drawers_l3582_358283

/-- Given 10 distinct keys, prove the following:
1. The number of possible pairs of keys
2. The number of copies of each key needed to form all possible pairs
3. The minimum number of drawers to open to ensure possession of all 10 different keys
-/
theorem key_pairs_and_drawers (n : ℕ) (h : n = 10) :
  let num_pairs := n.choose 2
  let copies_per_key := n - 1
  let total_drawers := num_pairs
  let min_drawers := total_drawers - copies_per_key + 1
  (num_pairs = 45) ∧ (copies_per_key = 9) ∧ (min_drawers = 37) := by
  sorry


end key_pairs_and_drawers_l3582_358283


namespace triangle_side_ratio_l3582_358206

/-- Given a triangle with perimeter 720 cm and longest side 280 cm, 
    prove that the ratio of the sides can be expressed as k:l:1, where k + l = 1.5714 -/
theorem triangle_side_ratio (a b c : ℝ) (h_perimeter : a + b + c = 720) 
  (h_longest : c = 280) (h_c_longest : a ≤ c ∧ b ≤ c) :
  ∃ (k l : ℝ), k + l = 1.5714 ∧ (a / c = k ∧ b / c = l) :=
sorry

end triangle_side_ratio_l3582_358206


namespace sammy_total_problems_l3582_358277

/-- The total number of math problems Sammy had to do -/
def total_problems (finished : ℕ) (remaining : ℕ) : ℕ :=
  finished + remaining

/-- Theorem stating that Sammy's total math problems equal 9 -/
theorem sammy_total_problems :
  total_problems 2 7 = 9 := by
  sorry

end sammy_total_problems_l3582_358277


namespace blueberries_per_basket_l3582_358223

theorem blueberries_per_basket (initial_basket : ℕ) (additional_baskets : ℕ) (total_blueberries : ℕ) : 
  initial_basket > 0 →
  additional_baskets = 9 →
  total_blueberries = 200 →
  total_blueberries = (initial_basket + additional_baskets) * initial_basket →
  initial_basket = 20 := by
  sorry

end blueberries_per_basket_l3582_358223


namespace t_shirt_packages_l3582_358273

theorem t_shirt_packages (total_shirts : ℕ) (shirts_per_package : ℕ) (h1 : total_shirts = 51) (h2 : shirts_per_package = 3) :
  total_shirts / shirts_per_package = 17 :=
by sorry

end t_shirt_packages_l3582_358273


namespace smallest_n_with_divisibility_n_98_satisfies_conditions_smallest_n_is_98_l3582_358238

/-- Checks if at least one of three consecutive integers is divisible by a given number. -/
def oneOfThreeDivisibleBy (n : ℕ) (d : ℕ) : Prop :=
  d ∣ n ∨ d ∣ (n + 1) ∨ d ∣ (n + 2)

/-- The main theorem stating that 98 is the smallest positive integer satisfying the given conditions. -/
theorem smallest_n_with_divisibility : ∀ n : ℕ, n > 0 →
  (oneOfThreeDivisibleBy n (2^2) ∧
   oneOfThreeDivisibleBy n (3^2) ∧
   oneOfThreeDivisibleBy n (5^2) ∧
   oneOfThreeDivisibleBy n (7^2)) →
  n ≥ 98 :=
by sorry

/-- Proof that 98 satisfies all the divisibility conditions. -/
theorem n_98_satisfies_conditions :
  oneOfThreeDivisibleBy 98 (2^2) ∧
  oneOfThreeDivisibleBy 98 (3^2) ∧
  oneOfThreeDivisibleBy 98 (5^2) ∧
  oneOfThreeDivisibleBy 98 (7^2) :=
by sorry

/-- The final theorem combining the above results to prove 98 is the smallest such positive integer. -/
theorem smallest_n_is_98 :
  ∃ n : ℕ, n > 0 ∧
  oneOfThreeDivisibleBy n (2^2) ∧
  oneOfThreeDivisibleBy n (3^2) ∧
  oneOfThreeDivisibleBy n (5^2) ∧
  oneOfThreeDivisibleBy n (7^2) ∧
  ∀ m : ℕ, m > 0 →
    (oneOfThreeDivisibleBy m (2^2) ∧
     oneOfThreeDivisibleBy m (3^2) ∧
     oneOfThreeDivisibleBy m (5^2) ∧
     oneOfThreeDivisibleBy m (7^2)) →
    m ≥ n :=
by sorry

end smallest_n_with_divisibility_n_98_satisfies_conditions_smallest_n_is_98_l3582_358238


namespace race_outcomes_count_l3582_358221

/-- The number of participants in the race -/
def total_participants : ℕ := 6

/-- The number of participants eligible for top three positions -/
def eligible_participants : ℕ := total_participants - 1

/-- The number of top positions to be filled -/
def top_positions : ℕ := 3

/-- Calculates the number of permutations for selecting k items from n items -/
def permutations (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

/-- The main theorem stating the number of possible race outcomes -/
theorem race_outcomes_count : 
  permutations eligible_participants top_positions = 60 := by
  sorry

end race_outcomes_count_l3582_358221


namespace total_amount_distributed_l3582_358296

/-- Given an equal distribution of money among 22 persons, where each person receives Rs 1950,
    prove that the total amount distributed is Rs 42900. -/
theorem total_amount_distributed (num_persons : ℕ) (amount_per_person : ℕ) 
  (h1 : num_persons = 22)
  (h2 : amount_per_person = 1950) : 
  num_persons * amount_per_person = 42900 := by
  sorry

end total_amount_distributed_l3582_358296


namespace percentage_relation_l3582_358290

theorem percentage_relation (j k l m x : ℝ) 
  (h1 : 1.25 * j = (x / 100) * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 25 := by
  sorry

end percentage_relation_l3582_358290


namespace roulette_sectors_l3582_358252

def roulette_wheel (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 10 ∧ 
  (1 - (5 / n)^2 : ℚ) = 3/4

theorem roulette_sectors : ∃ (n : ℕ), roulette_wheel n ∧ n = 10 := by
  sorry

end roulette_sectors_l3582_358252
