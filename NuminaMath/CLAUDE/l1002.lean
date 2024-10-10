import Mathlib

namespace yard_length_26_trees_l1002_100240

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (distance_between_trees : ℝ) : ℝ :=
  (num_trees - 1) * distance_between_trees

/-- Theorem: The length of a yard with 26 trees planted at equal distances,
    with one tree at each end and 10 meters between consecutive trees, is 250 meters. -/
theorem yard_length_26_trees :
  yard_length 26 10 = 250 := by
  sorry

end yard_length_26_trees_l1002_100240


namespace wednesday_distance_l1002_100257

theorem wednesday_distance (monday_distance tuesday_distance : ℕ) 
  (average_distance : ℚ) (total_days : ℕ) :
  monday_distance = 12 →
  tuesday_distance = 18 →
  average_distance = 17 →
  total_days = 3 →
  (monday_distance + tuesday_distance + (average_distance * total_days - monday_distance - tuesday_distance : ℚ)) / total_days = average_distance →
  average_distance * total_days - monday_distance - tuesday_distance = 21 := by
  sorry

end wednesday_distance_l1002_100257


namespace tip_percentage_is_thirty_percent_l1002_100249

/-- Calculates the tip percentage given meal costs and total price --/
def calculate_tip_percentage (appetizer_cost : ℚ) (entree_cost : ℚ) (num_entrees : ℕ) (dessert_cost : ℚ) (total_price : ℚ) : ℚ :=
  let meal_cost := appetizer_cost + entree_cost * num_entrees + dessert_cost
  let tip_amount := total_price - meal_cost
  (tip_amount / meal_cost) * 100

/-- Proves that the tip percentage is 30% given the specific meal costs --/
theorem tip_percentage_is_thirty_percent :
  calculate_tip_percentage 9 20 2 11 78 = 30 := by
  sorry

end tip_percentage_is_thirty_percent_l1002_100249


namespace regular_polygon_sides_l1002_100269

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular polygon satisfies the given condition if the number of its diagonals
    plus 6 equals twice the number of its sides -/
def satisfies_condition (n : ℕ) : Prop :=
  num_diagonals n + 6 = 2 * n

theorem regular_polygon_sides :
  ∃ (n : ℕ), n > 2 ∧ satisfies_condition n ∧ n = 4 :=
sorry

end regular_polygon_sides_l1002_100269


namespace paper_cups_pallets_l1002_100274

theorem paper_cups_pallets (total : ℕ) (towels tissues plates cups : ℕ) : 
  total = 20 ∧
  towels = total / 2 ∧
  tissues = total / 4 ∧
  plates = total / 5 ∧
  total = towels + tissues + plates + cups →
  cups = 1 := by
sorry

end paper_cups_pallets_l1002_100274


namespace hyperbola_equation_l1002_100241

/-- A hyperbola is defined by its standard equation and properties. -/
structure Hyperbola where
  /-- The coefficient of x² in the standard equation -/
  a : ℝ
  /-- The coefficient of y² in the standard equation -/
  b : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ

/-- The standard equation of a hyperbola holds for its defining point. -/
def satisfies_equation (h : Hyperbola) : Prop :=
  h.a * h.point.1^2 - h.b * h.point.2^2 = 1

/-- The asymptote slope is related to the coefficients in the standard equation. -/
def asymptote_condition (h : Hyperbola) : Prop :=
  h.asymptote_slope^2 = h.a / h.b

/-- The theorem stating the standard equation of the hyperbola. -/
theorem hyperbola_equation (h : Hyperbola)
    (point_cond : h.point = (4, Real.sqrt 3))
    (slope_cond : h.asymptote_slope = 1/2)
    (eq_cond : satisfies_equation h)
    (asym_cond : asymptote_condition h) :
    h.a = 1/4 ∧ h.b = 1 :=
  sorry

end hyperbola_equation_l1002_100241


namespace fraction_multiplication_l1002_100256

theorem fraction_multiplication : (2 : ℚ) / 3 * 5 / 7 * 8 / 9 = 80 / 189 := by
  sorry

end fraction_multiplication_l1002_100256


namespace correct_calculation_l1002_100206

theorem correct_calculation (x : ℝ) : 
  (x / 2 + 45 = 85) → (2 * x - 45 = 115) := by
  sorry

end correct_calculation_l1002_100206


namespace soap_bars_per_pack_l1002_100282

/-- Given that Nancy bought 6 packs of soap and 30 bars of soap in total,
    prove that the number of bars in each pack is 5. -/
theorem soap_bars_per_pack :
  ∀ (total_packs : ℕ) (total_bars : ℕ),
    total_packs = 6 →
    total_bars = 30 →
    total_bars / total_packs = 5 :=
by
  sorry

end soap_bars_per_pack_l1002_100282


namespace amelia_monday_sales_l1002_100218

/-- Represents the number of Jet Bars Amelia sold on Monday -/
def monday_sales : ℕ := sorry

/-- Represents the number of Jet Bars Amelia sold on Tuesday -/
def tuesday_sales : ℕ := sorry

/-- The weekly goal for Jet Bar sales -/
def weekly_goal : ℕ := 90

/-- The number of Jet Bars remaining to be sold -/
def remaining_sales : ℕ := 16

theorem amelia_monday_sales :
  monday_sales = 45 ∧
  tuesday_sales = monday_sales - 16 ∧
  monday_sales + tuesday_sales + remaining_sales = weekly_goal :=
by sorry

end amelia_monday_sales_l1002_100218


namespace sasha_leaves_picked_l1002_100255

/-- The number of apple trees along the road -/
def apple_trees : ℕ := 17

/-- The number of poplar trees along the road -/
def poplar_trees : ℕ := 20

/-- The index of the apple tree from which Sasha starts picking leaves -/
def start_index : ℕ := 8

/-- The total number of trees along the road -/
def total_trees : ℕ := apple_trees + poplar_trees

/-- The number of leaves Sasha picked -/
def leaves_picked : ℕ := total_trees - (start_index - 1)

theorem sasha_leaves_picked : leaves_picked = 24 := by
  sorry

end sasha_leaves_picked_l1002_100255


namespace geometric_sequence_properties_l1002_100222

/-- A geometric sequence with a_2 = 2 and a_8 = 128 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 8 = 128

/-- The general formula for the sequence -/
def GeneralFormula (a : ℕ → ℝ) : Prop :=
  (∀ n, a n = 2^(n-1)) ∨ (∀ n, a n = -(-2)^(n-1))

/-- The sum of the first n terms -/
def SumFormula (S : ℕ → ℝ) : Prop :=
  (∀ n, S n = 2^n - 1) ∨ (∀ n, S n = (1/3) * ((-2)^n - 1))

theorem geometric_sequence_properties
  (a : ℕ → ℝ) (S : ℕ → ℝ) (h : GeometricSequence a) :
  GeneralFormula a ∧ SumFormula S :=
sorry

end geometric_sequence_properties_l1002_100222


namespace fish_count_l1002_100217

/-- The number of fish caught by Jeffery -/
def jeffery_fish : ℕ := 60

/-- The number of fish caught by Ryan -/
def ryan_fish : ℕ := jeffery_fish / 2

/-- The number of fish caught by Jason -/
def jason_fish : ℕ := ryan_fish / 3

/-- The total number of fish caught by all three -/
def total_fish : ℕ := jason_fish + ryan_fish + jeffery_fish

theorem fish_count : total_fish = 100 := by
  sorry

end fish_count_l1002_100217


namespace geometric_sequence_a5_l1002_100238

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  (a 3)^2 + 4 * (a 3) + 1 = 0 →
  (a 7)^2 + 4 * (a 7) + 1 = 0 →
  a 5 = -1 := by
  sorry

end geometric_sequence_a5_l1002_100238


namespace complex_equation_solution_l1002_100276

theorem complex_equation_solution (z : ℂ) : 
  (3 + 4*I) / I = z / (1 + I) → z = 7 + I := by
  sorry

end complex_equation_solution_l1002_100276


namespace soccer_season_length_l1002_100288

theorem soccer_season_length (total_games : ℕ) (games_per_month : ℕ) (h1 : total_games = 27) (h2 : games_per_month = 9) :
  total_games / games_per_month = 3 := by
  sorry

end soccer_season_length_l1002_100288


namespace complex_fraction_simplification_l1002_100264

theorem complex_fraction_simplification :
  (((1 : ℂ) + 2 * Complex.I) ^ 2) / ((3 : ℂ) - 4 * Complex.I) = -1 := by
  sorry

end complex_fraction_simplification_l1002_100264


namespace stating_meeting_handshakes_l1002_100203

/-- 
Given a group of people at a meeting, where each person shakes hands with at least
a certain number of others, this function calculates the minimum possible number of handshakes.
-/
def min_handshakes (n : ℕ) (min_shakes_per_person : ℕ) : ℕ :=
  (n * min_shakes_per_person) / 2

/-- 
Theorem stating that for a meeting of 30 people where each person shakes hands with
at least 3 others, the minimum possible number of handshakes is 45.
-/
theorem meeting_handshakes :
  min_handshakes 30 3 = 45 := by
  sorry

#eval min_handshakes 30 3

end stating_meeting_handshakes_l1002_100203


namespace count_valid_pairs_l1002_100207

/-- The number of ordered pairs (m, n) of positive integers satisfying m ≥ n and m² - n² = 120 -/
def count_pairs : ℕ := 4

/-- Predicate for valid pairs -/
def is_valid_pair (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≥ n ∧ m^2 - n^2 = 120

theorem count_valid_pairs :
  (∃! (s : Finset (ℕ × ℕ)), s.card = count_pairs ∧ 
    ∀ p : ℕ × ℕ, p ∈ s ↔ is_valid_pair p.1 p.2) :=
sorry

end count_valid_pairs_l1002_100207


namespace cary_calorie_deficit_l1002_100267

-- Define the given constants
def miles_walked : ℕ := 3
def calories_per_mile : ℕ := 150
def calories_consumed : ℕ := 200

-- Define the net calorie deficit
def net_calorie_deficit : ℕ := miles_walked * calories_per_mile - calories_consumed

-- Theorem statement
theorem cary_calorie_deficit : net_calorie_deficit = 250 := by
  sorry

end cary_calorie_deficit_l1002_100267


namespace smallest_addition_and_quotient_l1002_100234

theorem smallest_addition_and_quotient : 
  let n := 897326
  let d := 456
  let x := d - (n % d)
  ∀ y, 0 ≤ y ∧ y < x → ¬(d ∣ (n + y)) ∧
  (d ∣ (n + x)) ∧
  ((n + x) / d = 1968) := by
  sorry

end smallest_addition_and_quotient_l1002_100234


namespace gcd_factorial_eight_and_factorial_six_squared_l1002_100226

theorem gcd_factorial_eight_and_factorial_six_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 1440 := by
  sorry

end gcd_factorial_eight_and_factorial_six_squared_l1002_100226


namespace total_books_calculation_l1002_100239

/-- The number of book shelves -/
def num_shelves : ℕ := 350

/-- The number of books per shelf -/
def books_per_shelf : ℕ := 25

/-- The total number of books on all shelves -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem total_books_calculation : total_books = 8750 := by
  sorry

end total_books_calculation_l1002_100239


namespace limit_at_neg_seven_l1002_100229

/-- The limit of (2x^2 + 15x + 7)/(x + 7) as x approaches -7 is -13 -/
theorem limit_at_neg_seven (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, x ≠ -7 →
    |x - (-7)| < δ → |(2*x^2 + 15*x + 7)/(x + 7) - (-13)| < ε :=
by sorry

end limit_at_neg_seven_l1002_100229


namespace equation_classification_l1002_100211

-- Define what a linear equation in two variables is
def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y + c

-- Define the properties of the equation in question
def has_two_unknowns_and_degree_one (f : ℝ → ℝ → ℝ) : Prop :=
  (∃ (x y : ℝ), f x y ≠ f x 0 ∧ f x y ≠ f 0 y) ∧ 
  (∀ (x y : ℝ), ∃ (a b c : ℝ), f x y = a * x + b * y + c)

-- State the theorem
theorem equation_classification 
  (f : ℝ → ℝ → ℝ) 
  (h : has_two_unknowns_and_degree_one f) : 
  is_linear_equation_in_two_variables f :=
sorry

end equation_classification_l1002_100211


namespace cannoneer_count_l1002_100263

theorem cannoneer_count (total : ℕ) (cannoneers : ℕ) (women : ℕ) (men : ℕ)
  (h1 : women = 2 * cannoneers)
  (h2 : men = 2 * women)
  (h3 : total = men + women)
  (h4 : total = 378) :
  cannoneers = 63 := by
sorry

end cannoneer_count_l1002_100263


namespace hyperbola_eccentricity_l1002_100296

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≥ 1) 
  (h4 : ∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ 
    (Real.sqrt ((x + c)^2 + y^2) - Real.sqrt ((x - c)^2 + y^2))^2 = b^2 - 3*a*b) 
  (h5 : c^2 = a^2 + b^2) : 
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 17 := by
  sorry


end hyperbola_eccentricity_l1002_100296


namespace salary_percentage_calculation_l1002_100236

/-- Given two employees X and Y with a total salary and Y's known salary,
    calculate the percentage of Y's salary that X is paid. -/
theorem salary_percentage_calculation
  (total_salary : ℝ) (y_salary : ℝ) (h1 : total_salary = 638)
  (h2 : y_salary = 290) :
  (total_salary - y_salary) / y_salary * 100 = 120 :=
by sorry

end salary_percentage_calculation_l1002_100236


namespace indeterminate_larger_number_l1002_100279

/-- Given two real numbers x and y and a constant k such that
    x * k = y + 1 and x + y = -64, prove that it's not possible
    to determine which of x or y is larger without additional information. -/
theorem indeterminate_larger_number (x y k : ℝ) 
    (h1 : x * k = y + 1) 
    (h2 : x + y = -64) : 
  ¬ (∀ x y : ℝ, (x * k = y + 1 ∧ x + y = -64) → x < y ∨ y < x) :=
by
  sorry


end indeterminate_larger_number_l1002_100279


namespace foil_covered_prism_width_l1002_100219

/-- The width of a foil-covered rectangular prism -/
theorem foil_covered_prism_width :
  ∀ (inner_length inner_width inner_height : ℝ),
    inner_length * inner_width * inner_height = 128 →
    inner_width = 2 * inner_length →
    inner_width = 2 * inner_height →
    ∃ (outer_width : ℝ),
      outer_width = 4 * (2 : ℝ)^(1/3) + 2 :=
by
  sorry

end foil_covered_prism_width_l1002_100219


namespace multiplication_increase_l1002_100215

theorem multiplication_increase (x : ℝ) : x * 20 = 20 + 280 → x = 15 := by
  sorry

end multiplication_increase_l1002_100215


namespace always_sum_21_l1002_100266

theorem always_sum_21 (selection : Finset ℕ) :
  selection ⊆ Finset.range 20 →
  selection.card = 11 →
  ∃ x y, x ∈ selection ∧ y ∈ selection ∧ x ≠ y ∧ x + y = 21 :=
sorry

end always_sum_21_l1002_100266


namespace absent_student_percentage_l1002_100214

theorem absent_student_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_students = 180)
  (h2 : boys = 100)
  (h3 : girls = 80)
  (h4 : total_students = boys + girls)
  (absent_boys_ratio : ℚ)
  (absent_girls_ratio : ℚ)
  (h5 : absent_boys_ratio = 1 / 5)
  (h6 : absent_girls_ratio = 1 / 4) :
  (((boys * absent_boys_ratio + girls * absent_girls_ratio) / total_students) : ℚ) = 2222 / 10000 := by
  sorry

end absent_student_percentage_l1002_100214


namespace pet_owners_proof_l1002_100285

theorem pet_owners_proof (total_pet_owners : Nat) 
                         (only_dog_owners : Nat)
                         (only_cat_owners : Nat)
                         (cat_dog_snake_owners : Nat)
                         (total_snakes : Nat)
                         (h1 : total_pet_owners = 99)
                         (h2 : only_dog_owners = 15)
                         (h3 : only_cat_owners = 10)
                         (h4 : cat_dog_snake_owners = 3)
                         (h5 : total_snakes = 69) : 
  total_pet_owners = only_dog_owners + only_cat_owners + cat_dog_snake_owners + (total_snakes - cat_dog_snake_owners) + 5 :=
by
  sorry

#check pet_owners_proof

end pet_owners_proof_l1002_100285


namespace floor_abs_sum_l1002_100280

theorem floor_abs_sum : ⌊|(-5.3 : ℝ)|⌋ + |⌊(-5.3 : ℝ)⌋| = 11 := by
  sorry

end floor_abs_sum_l1002_100280


namespace max_d_value_l1002_100281

def a (n : ℕ) : ℕ := 100 + n^n

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n-1))

theorem max_d_value :
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → d n ≤ 401 ∧ ∃ (m : ℕ), m ≥ N ∧ d m = 401 :=
sorry

end max_d_value_l1002_100281


namespace exam_student_count_l1002_100284

theorem exam_student_count 
  (total_average : ℝ)
  (excluded_count : ℕ)
  (excluded_average : ℝ)
  (remaining_average : ℝ)
  (h1 : total_average = 80)
  (h2 : excluded_count = 5)
  (h3 : excluded_average = 20)
  (h4 : remaining_average = 95)
  : ∃ N : ℕ, N > 0 ∧ 
    N * total_average = 
    (N - excluded_count) * remaining_average + excluded_count * excluded_average :=
by
  sorry

end exam_student_count_l1002_100284


namespace money_distribution_problem_l1002_100210

/-- Represents the money distribution problem among three friends --/
structure MoneyDistribution where
  total : ℝ  -- Total amount to distribute
  neha : ℝ   -- Neha's share
  sabi : ℝ   -- Sabi's share
  mahi : ℝ   -- Mahi's share
  x : ℝ      -- Amount removed from Sabi's share

/-- The conditions of the problem --/
def problemConditions (d : MoneyDistribution) : Prop :=
  d.total = 1100 ∧
  d.mahi = 102 ∧
  d.neha + d.sabi + d.mahi = d.total ∧
  (d.neha - 5) / (d.sabi - d.x) = 1/4 ∧
  (d.neha - 5) / (d.mahi - 4) = 1/3

/-- The theorem to prove --/
theorem money_distribution_problem (d : MoneyDistribution) 
  (h : problemConditions d) : d.x = 829.67 := by
  sorry

end money_distribution_problem_l1002_100210


namespace divisibility_proof_l1002_100289

theorem divisibility_proof (a b c : ℝ) 
  (h : (a ≠ 0 ∧ b ≠ 0) ∨ (a ≠ 0 ∧ c ≠ 0) ∨ (b ≠ 0 ∧ c ≠ 0)) :
  ∃ k : ℤ, (a + b + c)^7 - a^7 - b^7 - c^7 = k * (7 * (a + b) * (b + c) * (c + a)) :=
sorry

end divisibility_proof_l1002_100289


namespace sin_cos_identity_tan_fraction_value_l1002_100228

-- Part 1
theorem sin_cos_identity (α : Real) :
  (Real.sin (3 * α) / Real.sin α) - (Real.cos (3 * α) / Real.cos α) = 2 := by
  sorry

-- Part 2
theorem tan_fraction_value (α : Real) (h : Real.tan (α / 2) = 2) :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 := by
  sorry

end sin_cos_identity_tan_fraction_value_l1002_100228


namespace bev_is_third_oldest_l1002_100223

/-- Represents the age of a person -/
structure Age : Type where
  value : ℕ

/-- Represents a person with their name and age -/
structure Person : Type where
  name : String
  age : Age

/-- Defines the "older than" relation between two people -/
def olderThan (p1 p2 : Person) : Prop :=
  p1.age.value > p2.age.value

theorem bev_is_third_oldest 
  (andy bev cao dhruv elcim : Person)
  (h1 : olderThan dhruv bev)
  (h2 : olderThan bev elcim)
  (h3 : olderThan andy elcim)
  (h4 : olderThan bev andy)
  (h5 : olderThan cao bev) :
  ∃ (x y : Person), 
    (olderThan x bev ∧ olderThan y bev) ∧
    (∀ (z : Person), z ≠ x ∧ z ≠ y → olderThan bev z ∨ z = bev) :=
by sorry

end bev_is_third_oldest_l1002_100223


namespace rods_in_mile_l1002_100250

/-- Represents the number of furlongs in a mile -/
def furlongs_per_mile : ℕ := 8

/-- Represents the number of rods in a furlong -/
def rods_per_furlong : ℕ := 40

/-- Theorem stating that one mile is equal to 320 rods -/
theorem rods_in_mile : furlongs_per_mile * rods_per_furlong = 320 := by
  sorry

end rods_in_mile_l1002_100250


namespace curve_is_ellipse_with_foci_on_y_axis_l1002_100202

/-- The curve represented by x²sin(α) - y²cos(α) = 1 is an ellipse with foci on the y-axis when α is between π/2 and 3π/4 -/
theorem curve_is_ellipse_with_foci_on_y_axis (α : Real) 
  (h_α_range : α ∈ Set.Ioo (π / 2) (3 * π / 4)) :
  ∃ (a b : Real), a > 0 ∧ b > 0 ∧ a > b ∧
  ∀ (x y : Real), x^2 * Real.sin α - y^2 * Real.cos α = 1 ↔ 
    (x^2 / b^2) + (y^2 / a^2) = 1 :=
by sorry

end curve_is_ellipse_with_foci_on_y_axis_l1002_100202


namespace inequality_proof_l1002_100293

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 3*b^3) / (5*a + b) + (b^3 + 3*c^3) / (5*b + c) + (c^3 + 3*a^3) / (5*c + a) ≥ 
  2/3 * (a^2 + b^2 + c^2) := by
  sorry

end inequality_proof_l1002_100293


namespace problem_solution_l1002_100237

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k - |x - 4|

-- Define the theorem
theorem problem_solution (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sol : Set.Icc (-1 : ℝ) 1 = {x : ℝ | f 1 (x + 4) ≥ 0})
  (h_eq : 1/a + 1/(2*b) + 1/(3*c) = 1) :
  1 = 1 ∧ (1/9)*a + (2/9)*b + (3/9)*c ≥ 1 := by
  sorry


end problem_solution_l1002_100237


namespace zoo_with_only_hippos_possible_l1002_100213

-- Define the universe of zoos
variable (Z : Type)

-- Define the subsets of zoos with hippos, rhinos, and giraffes
variable (H R G : Set Z)

-- Define the conditions
axiom condition1 : H ∩ R ⊆ Gᶜ
axiom condition2 : R ∩ Gᶜ ⊆ H
axiom condition3 : H ∩ G ⊆ R

-- Theorem to prove
theorem zoo_with_only_hippos_possible :
  ∃ (z : Z), z ∈ H ∧ z ∉ G ∧ z ∉ R :=
sorry

end zoo_with_only_hippos_possible_l1002_100213


namespace childrens_tickets_sold_l1002_100230

theorem childrens_tickets_sold
  (adult_price senior_price children_price : ℚ)
  (total_tickets : ℕ)
  (total_revenue : ℚ)
  (h1 : adult_price = 6)
  (h2 : children_price = 9/2)
  (h3 : senior_price = 5)
  (h4 : total_tickets = 600)
  (h5 : total_revenue = 3250)
  : ∃ (A C S : ℕ),
    A + C + S = total_tickets ∧
    adult_price * A + children_price * C + senior_price * S = total_revenue ∧
    C = (350 - S) / (3/2) :=
sorry

end childrens_tickets_sold_l1002_100230


namespace average_and_difference_l1002_100232

theorem average_and_difference (y : ℝ) : 
  (46 + y) / 2 = 52 → |y - 46| = 12 := by
  sorry

end average_and_difference_l1002_100232


namespace sum_difference_absolute_values_l1002_100231

theorem sum_difference_absolute_values : 
  (3 + (-4) + (-5)) - (|3| + |-4| + |-5|) = -18 := by
  sorry

end sum_difference_absolute_values_l1002_100231


namespace triangle_area_comparison_l1002_100252

theorem triangle_area_comparison : 
  let a : Real := 3
  let b : Real := 5
  let c : Real := 6
  let p : Real := (a + b + c) / 2
  let area_A : Real := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let area_B : Real := (3 * Real.sqrt 14) / 2
  area_A = 2 * Real.sqrt 14 ∧ area_A / area_B = 4 / 3 := by sorry

end triangle_area_comparison_l1002_100252


namespace cubic_equation_fraction_value_l1002_100271

theorem cubic_equation_fraction_value (a : ℝ) : 
  a^3 + 3*a^2 + a = 0 → 
  (2022*a^2) / (a^4 + 2015*a^2 + 1) = 0 ∨ (2022*a^2) / (a^4 + 2015*a^2 + 1) = 1 :=
by sorry

end cubic_equation_fraction_value_l1002_100271


namespace problem_solution_l1002_100246

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) : 
  (x - 1)^2 + 16/((x - 1)^2) = 23/3 := by
  sorry

end problem_solution_l1002_100246


namespace negation_of_or_statement_l1002_100262

theorem negation_of_or_statement (x y : ℝ) :
  ¬(x > 1 ∨ y > 1) ↔ x ≤ 1 ∧ y ≤ 1 := by
  sorry

end negation_of_or_statement_l1002_100262


namespace problem_statement_l1002_100245

/-- An arithmetic sequence with a non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) - a n = d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) / b n = r

theorem problem_statement (a b : ℕ → ℝ) : 
  arithmetic_sequence a →
  geometric_sequence b →
  3 * a 2005 - (a 2007)^2 + 3 * a 2009 = 0 →
  b 2007 = a 2007 →
  b 2006 * b 2008 = 36 := by
  sorry

end problem_statement_l1002_100245


namespace min_value_inequality_l1002_100297

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1 / x + 1 / y) ≥ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (a + b) * (1 / a + 1 / b) = 4 :=
sorry

end min_value_inequality_l1002_100297


namespace hyperbola_eccentricity_sqrt_two_l1002_100205

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The left focus of a hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- A vertex on the imaginary axis of a hyperbola -/
def imaginary_vertex (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- A point on the right asymptote of a hyperbola -/
def right_asymptote_point (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Vector from point p1 to point p2 -/
def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity_sqrt_two (a b : ℝ) (h : Hyperbola a b) :
  let F := left_focus h
  let A := imaginary_vertex h
  let B := right_asymptote_point h
  vector F A = (Real.sqrt 2 - 1) • vector A B →
  eccentricity h = Real.sqrt 2 := by sorry

end hyperbola_eccentricity_sqrt_two_l1002_100205


namespace same_color_probability_three_colors_three_draws_l1002_100260

/-- The probability of drawing the same color ball three times in a row --/
def same_color_probability (total_colors : ℕ) (num_draws : ℕ) : ℚ :=
  (total_colors : ℚ) / (total_colors ^ num_draws : ℚ)

/-- Theorem: The probability of drawing the same color ball three times in a row,
    with replacement, from a bag containing one red, one yellow, and one green ball,
    is equal to 1/9. --/
theorem same_color_probability_three_colors_three_draws :
  same_color_probability 3 3 = 1 / 9 := by
  sorry

#eval same_color_probability 3 3

end same_color_probability_three_colors_three_draws_l1002_100260


namespace merchant_profit_percentage_l1002_100253

theorem merchant_profit_percentage (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  markup_percentage = 75 →
  discount_percentage = 40 →
  cost_price > 0 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 5 := by
sorry

end merchant_profit_percentage_l1002_100253


namespace limit_S_2_pow_n_to_infinity_l1002_100254

/-- S(n) represents the sum of digits of n in base 10 -/
def S (n : ℕ) : ℕ := sorry

/-- Main theorem: The limit of S(2^n) as n approaches infinity is infinity -/
theorem limit_S_2_pow_n_to_infinity :
  ∀ M : ℕ, ∃ N : ℕ, ∀ n : ℕ, n ≥ N → S (2^n) > M :=
sorry

end limit_S_2_pow_n_to_infinity_l1002_100254


namespace unique_square_cube_factor_of_1800_l1002_100295

/-- A number is a perfect square if it can be expressed as the product of an integer with itself. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- A number is a perfect cube if it can be expressed as the product of an integer with itself three times. -/
def IsPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k * k

/-- A number is a factor of another number if it divides the latter without a remainder. -/
def IsFactor (a n : ℕ) : Prop :=
  n % a = 0

theorem unique_square_cube_factor_of_1800 :
  ∃! x : ℕ, x > 0 ∧ IsFactor x 1800 ∧ IsPerfectSquare x ∧ IsPerfectCube x :=
sorry

end unique_square_cube_factor_of_1800_l1002_100295


namespace greatest_divisor_four_consecutive_integers_l1002_100225

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (k : ℕ), (∀ (n : ℕ), n > 0 → k ∣ (n * (n + 1) * (n + 2) * (n + 3))) → k ≤ d) ∧
  d = 12 :=
by sorry

end greatest_divisor_four_consecutive_integers_l1002_100225


namespace fourth_player_win_probability_prove_fourth_player_win_probability_l1002_100242

/-- The probability of the fourth player winning in a coin-flipping game -/
theorem fourth_player_win_probability : Real → Prop :=
  fun p =>
    -- Define the game setup
    let n_players : ℕ := 4
    let coin_prob : Real := 1 / 2
    -- Define the probability of the fourth player winning on their nth turn
    let prob_win_nth_turn : ℕ → Real := fun n => coin_prob ^ (n_players * n)
    -- Define the sum of the infinite geometric series
    let total_prob : Real := (prob_win_nth_turn 1) / (1 - prob_win_nth_turn 1)
    -- The theorem statement
    p = total_prob ∧ p = 1 / 31

/-- Proof of the theorem -/
theorem prove_fourth_player_win_probability : 
  ∃ p : Real, fourth_player_win_probability p :=
sorry

end fourth_player_win_probability_prove_fourth_player_win_probability_l1002_100242


namespace tan_2_implies_sin_cos_2_5_l1002_100259

theorem tan_2_implies_sin_cos_2_5 (x : ℝ) (h : Real.tan x = 2) : 
  Real.sin x * Real.cos x = 2/5 := by
sorry

end tan_2_implies_sin_cos_2_5_l1002_100259


namespace triple_reflection_opposite_l1002_100261

/-- Represents a 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane mirror -/
inductive Mirror
  | XY
  | XZ
  | YZ

/-- Reflects a vector across a given mirror -/
def reflect (v : Vector3D) (m : Mirror) : Vector3D :=
  match m with
  | Mirror.XY => ⟨v.x, v.y, -v.z⟩
  | Mirror.XZ => ⟨v.x, -v.y, v.z⟩
  | Mirror.YZ => ⟨-v.x, v.y, v.z⟩

/-- Theorem: After three reflections on mutually perpendicular mirrors, 
    the resulting vector is opposite to the initial vector -/
theorem triple_reflection_opposite (f : Vector3D) :
  let f1 := reflect f Mirror.XY
  let f2 := reflect f1 Mirror.XZ
  let f3 := reflect f2 Mirror.YZ
  f3 = Vector3D.mk (-f.x) (-f.y) (-f.z) := by
  sorry


end triple_reflection_opposite_l1002_100261


namespace power_equality_comparisons_l1002_100290

theorem power_equality_comparisons :
  (-2^3 = (-2)^3) ∧
  (3^2 ≠ 2^3) ∧
  (-3^2 ≠ (-3)^2) ∧
  (-(3 * 2)^2 ≠ -3 * 2^2) := by sorry

end power_equality_comparisons_l1002_100290


namespace shanes_remaining_gum_is_eight_l1002_100268

/-- The number of pieces of gum Shane has left after a series of exchanges and consumption --/
def shanes_remaining_gum : ℕ :=
  let elyses_initial_gum : ℕ := 100
  let ricks_gum : ℕ := elyses_initial_gum / 2
  let shanes_initial_gum : ℕ := ricks_gum / 3
  let shanes_gum_after_cousin : ℕ := shanes_initial_gum + 10
  let shanes_gum_after_chewing : ℕ := shanes_gum_after_cousin - 11
  let gum_shared_with_sarah : ℕ := shanes_gum_after_chewing / 2
  shanes_gum_after_chewing - gum_shared_with_sarah

theorem shanes_remaining_gum_is_eight :
  shanes_remaining_gum = 8 := by
  sorry

end shanes_remaining_gum_is_eight_l1002_100268


namespace apples_in_basket_l1002_100200

/-- Represents the number of oranges in the basket -/
def oranges : ℕ := sorry

/-- Represents the number of apples in the basket -/
def apples : ℕ := 4 * oranges

/-- The total number of fruits consumed if 2/3 of each fruit's quantity is eaten -/
def consumed_fruits : ℕ := 50

theorem apples_in_basket : apples = 60 := by
  sorry

end apples_in_basket_l1002_100200


namespace log_less_than_square_l1002_100243

theorem log_less_than_square (x : ℝ) (h : x > 0) : Real.log (1 + x) < x^2 := by
  sorry

end log_less_than_square_l1002_100243


namespace arithmetic_sequence_sum_l1002_100275

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 45) →
  (a 2 + a 5 + a 8 = 39) →
  (a 3 + a 6 + a 9 = 33) :=
sorry

end arithmetic_sequence_sum_l1002_100275


namespace marcos_strawberries_weight_l1002_100272

theorem marcos_strawberries_weight 
  (total_weight : ℝ) 
  (dads_weight : ℝ) 
  (h1 : total_weight = 20)
  (h2 : dads_weight = 17) : 
  total_weight - dads_weight = 3 := by
sorry

end marcos_strawberries_weight_l1002_100272


namespace triangle_with_perimeter_12_has_area_6_l1002_100212

-- Define a triangle with integral sides
def Triangle := (ℕ × ℕ × ℕ)

-- Function to calculate perimeter of a triangle
def perimeter (t : Triangle) : ℕ :=
  let (a, b, c) := t
  a + b + c

-- Function to check if three sides form a valid triangle
def is_valid_triangle (t : Triangle) : Prop :=
  let (a, b, c) := t
  a + b > c ∧ b + c > a ∧ c + a > b

-- Function to calculate the area of a triangle using Heron's formula
noncomputable def area (t : Triangle) : ℝ :=
  let (a, b, c) := t
  let s : ℝ := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem triangle_with_perimeter_12_has_area_6 :
  ∃ (t : Triangle), perimeter t = 12 ∧ is_valid_triangle t ∧ area t = 6 :=
sorry

end triangle_with_perimeter_12_has_area_6_l1002_100212


namespace triangle_area_equality_l1002_100292

theorem triangle_area_equality (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x^2 + y^2 = 49)
  (h5 : y^2 + y*z + z^2 = 36)
  (h6 : x^2 + Real.sqrt 3 * x * z + z^2 = 25) :
  2*x*y + Real.sqrt 3 * y*z + z*x = 24 * Real.sqrt 6 := by
sorry

end triangle_area_equality_l1002_100292


namespace framed_photo_border_area_l1002_100235

/-- The area of the border of a framed rectangular photograph -/
theorem framed_photo_border_area 
  (photo_height : ℝ) 
  (photo_width : ℝ) 
  (border_width : ℝ) 
  (h1 : photo_height = 6) 
  (h2 : photo_width = 8) 
  (h3 : border_width = 3) : 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 120 := by
  sorry

end framed_photo_border_area_l1002_100235


namespace not_cube_of_integer_l1002_100204

theorem not_cube_of_integer : ¬ ∃ k : ℤ, (10^150 + 5 * 10^100 + 1 : ℤ) = k^3 := by
  sorry

end not_cube_of_integer_l1002_100204


namespace sequence_length_l1002_100209

theorem sequence_length (n : ℕ+) (b : ℕ → ℝ) : 
  b 0 = 41 →
  b 1 = 76 →
  b n = 0 →
  (∀ k : ℕ, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 4 / b k) →
  n = 777 :=
by sorry

end sequence_length_l1002_100209


namespace complex_magnitude_problem_l1002_100251

theorem complex_magnitude_problem : 
  let i : ℂ := Complex.I
  let z : ℂ := 1 + (1 - i)^2
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l1002_100251


namespace gcd_problem_l1002_100244

theorem gcd_problem (a b : ℕ+) (h : Nat.gcd a.val b.val = 15) :
  Nat.gcd (12 * a.val) (18 * b.val) ≥ 90 := by
  sorry

end gcd_problem_l1002_100244


namespace smallest_number_l1002_100291

theorem smallest_number (a b c d : ℝ) (h1 : a = 3) (h2 : b = -2) (h3 : c = 1/2) (h4 : d = 2) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by sorry

end smallest_number_l1002_100291


namespace simplify_expression_l1002_100247

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 8) - (x + 6)*(3*x - 2) = 4*x - 20 := by
  sorry

end simplify_expression_l1002_100247


namespace concentric_circles_ratio_l1002_100248

theorem concentric_circles_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  π * b^2 - π * a^2 = 5 * (π * a^2) → a / b = 1 / Real.sqrt 6 := by
  sorry

end concentric_circles_ratio_l1002_100248


namespace divisors_of_5_pow_30_minus_1_l1002_100208

theorem divisors_of_5_pow_30_minus_1 :
  ∀ n : ℕ, 90 < n → n < 100 → (5^30 - 1) % n = 0 ↔ n = 91 ∨ n = 97 := by
  sorry

end divisors_of_5_pow_30_minus_1_l1002_100208


namespace initial_green_balls_l1002_100221

theorem initial_green_balls (pink_balls : ℕ) (added_green_balls : ℕ) :
  pink_balls = 23 →
  added_green_balls = 14 →
  ∃ initial_green_balls : ℕ, 
    initial_green_balls + added_green_balls = pink_balls ∧
    initial_green_balls = 9 :=
by sorry

end initial_green_balls_l1002_100221


namespace rectangle_area_is_eight_l1002_100216

/-- A square inscribed in a circle, which is inscribed in a rectangle --/
structure SquareCircleRectangle where
  /-- Side length of the square --/
  s : ℝ
  /-- Radius of the circle --/
  r : ℝ
  /-- Width of the rectangle --/
  w : ℝ
  /-- Length of the rectangle --/
  l : ℝ
  /-- The square's diagonal is the circle's diameter --/
  h1 : s * Real.sqrt 2 = 2 * r
  /-- The circle's diameter is the rectangle's width --/
  h2 : 2 * r = w
  /-- The rectangle's length is twice its width --/
  h3 : l = 2 * w
  /-- The square's diagonal is 4 units --/
  h4 : s * Real.sqrt 2 = 4

/-- The area of the rectangle is 8 square units --/
theorem rectangle_area_is_eight (scr : SquareCircleRectangle) : scr.l * scr.w = 8 := by
  sorry

end rectangle_area_is_eight_l1002_100216


namespace divisibility_condition_l1002_100294

theorem divisibility_condition (n : ℕ) : 
  (∃ m : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → k ∣ m) ∧ 
            ¬((n + 1) ∣ m) ∧ ¬((n + 2) ∣ m) ∧ ¬((n + 3) ∣ m)) ↔ 
  n = 1 ∨ n = 2 ∨ n = 6 :=
sorry

end divisibility_condition_l1002_100294


namespace inequality_system_solution_set_l1002_100273

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  x + 1 > 0 ∧ x + 3 ≤ 4

-- Define the solution set
def solution_set : Set ℝ :=
  {x : ℝ | -1 < x ∧ x ≤ 1}

-- Theorem statement
theorem inequality_system_solution_set :
  {x : ℝ | inequality_system x} = solution_set :=
by sorry

end inequality_system_solution_set_l1002_100273


namespace bowling_team_average_weight_l1002_100277

theorem bowling_team_average_weight 
  (original_players : ℕ) 
  (new_player1_weight : ℕ) 
  (new_player2_weight : ℕ) 
  (new_average_weight : ℕ) 
  (h1 : original_players = 7)
  (h2 : new_player1_weight = 110)
  (h3 : new_player2_weight = 60)
  (h4 : new_average_weight = 99) : 
  ∃ (original_average : ℕ), 
    (original_players * original_average + new_player1_weight + new_player2_weight) / 
    (original_players + 2) = new_average_weight ∧ 
    original_average = 103 := by
  sorry

end bowling_team_average_weight_l1002_100277


namespace min_value_of_f_l1002_100286

/-- The function f(x) = 3x^2 - 18x + 2205 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2205

theorem min_value_of_f :
  ∃ (min : ℝ), min = 2178 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end min_value_of_f_l1002_100286


namespace rectangle_side_length_l1002_100258

/-- Given three rectangles with equal areas and integer sides, where one side is 37, prove that a specific side length is 1406. -/
theorem rectangle_side_length (a b : ℕ) : 
  let S := 37 * (a + b)  -- Common area of the rectangles
  -- ABCD area
  S = 37 * (a + b) →
  -- DEFG area
  S = a * 1406 →
  -- CEIH area
  S = b * 38 →
  -- All sides are integers
  a > 0 → b > 0 →
  -- DG length
  1406 = 1406 := by sorry

end rectangle_side_length_l1002_100258


namespace smallest_integer_solution_l1002_100283

theorem smallest_integer_solution (x : ℤ) : 3 * x - 7 ≤ 17 → x ≤ 8 := by sorry

end smallest_integer_solution_l1002_100283


namespace probability_at_least_one_switch_closed_l1002_100233

theorem probability_at_least_one_switch_closed 
  (p : ℝ) 
  (h1 : 0 < p) 
  (h2 : p < 1) :
  let prob_at_least_one_closed := 4*p - 6*p^2 + 4*p^3 - p^4
  prob_at_least_one_closed = 1 - (1 - p)^4 :=
by sorry

end probability_at_least_one_switch_closed_l1002_100233


namespace M_on_line_l_line_l_equation_AB_length_l1002_100270

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Define point M
def point_M : ℝ × ℝ := (2, 1)

-- Define that M is on line l
theorem M_on_line_l : line_l point_M.1 point_M.2 := by sorry

-- Define that A and B are on the parabola
axiom A_on_parabola : ∃ (x y : ℝ), parabola x y ∧ line_l x y
axiom B_on_parabola : ∃ (x y : ℝ), parabola x y ∧ line_l x y

-- Define that M is the midpoint of AB
axiom M_midpoint_AB : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
  point_M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

-- Theorem 1: The equation of line l is 2x - y - 3 = 0
theorem line_l_equation : ∀ (x y : ℝ), line_l x y ↔ 2*x - y - 3 = 0 := by sorry

-- Theorem 2: The length of segment AB is √35
theorem AB_length : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
    point_M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = Real.sqrt 35 := by sorry

end M_on_line_l_line_l_equation_AB_length_l1002_100270


namespace total_gold_value_proof_l1002_100298

/-- The value of one gold bar in dollars -/
def gold_bar_value : ℕ := 2200

/-- The number of gold bars Legacy has -/
def legacy_bars : ℕ := 5

/-- The number of gold bars Aleena has -/
def aleena_bars : ℕ := legacy_bars - 2

/-- The total value of gold for Legacy and Aleena -/
def total_gold_value : ℕ := gold_bar_value * (legacy_bars + aleena_bars)

theorem total_gold_value_proof : total_gold_value = 17600 := by
  sorry

end total_gold_value_proof_l1002_100298


namespace marcus_gathered_25_bottles_l1002_100220

-- Define the total number of milk bottles
def total_bottles : ℕ := 45

-- Define the number of bottles John gathered
def john_bottles : ℕ := 20

-- Define Marcus' bottles as the difference between total and John's
def marcus_bottles : ℕ := total_bottles - john_bottles

-- Theorem to prove
theorem marcus_gathered_25_bottles : marcus_bottles = 25 := by
  sorry

end marcus_gathered_25_bottles_l1002_100220


namespace min_value_theorem_l1002_100224

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  arithmetic_sequence a →
  (∀ n, a n > 0) →
  a 2018 = a 2017 + 2 * a 2016 →
  Real.sqrt (a m * a n) = 4 * a 1 →
  (1 : ℝ) / m + 5 / n ≥ 1 + Real.sqrt 5 / 3 :=
by sorry

end min_value_theorem_l1002_100224


namespace min_value_log_quadratic_l1002_100227

theorem min_value_log_quadratic (x : ℝ) (h : x^2 - 2*x + 3 > 0) :
  Real.log (x^2 - 2*x + 3) ≥ Real.log 2 := by
  sorry

end min_value_log_quadratic_l1002_100227


namespace polynomial_multiplication_l1002_100201

theorem polynomial_multiplication (x z : ℝ) :
  (3 * x^5 - 7 * z^3) * (9 * x^10 + 21 * x^5 * z^3 + 49 * z^6) = 27 * x^15 - 343 * z^9 := by
  sorry

end polynomial_multiplication_l1002_100201


namespace factorial_fraction_equals_one_l1002_100278

theorem factorial_fraction_equals_one : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end factorial_fraction_equals_one_l1002_100278


namespace hyperbola_parabola_ratio_l1002_100299

/-- Given a hyperbola and a parabola with specific properties, prove that the ratio of the hyperbola's semi-major and semi-minor axes is equal to √3/3. -/
theorem hyperbola_parabola_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∃ c : ℝ, c^2 = a^2 + b^2) →  -- Relationship between a, b, and c in a hyperbola
  (c / a = 2) →  -- Eccentricity is 2
  (c = 1) →  -- Right focus coincides with the focus of y^2 = 4x
  a / b = Real.sqrt 3 / 3 := by
sorry

end hyperbola_parabola_ratio_l1002_100299


namespace stamp_arrangement_count_l1002_100287

/-- Represents a stamp with its denomination -/
structure Stamp where
  denomination : Nat
  deriving Repr

/-- Represents an arrangement of stamps -/
def Arrangement := List Stamp

/-- Checks if an arrangement is valid (sums to 15 cents) -/
def isValidArrangement (arr : Arrangement) : Bool :=
  (arr.map (·.denomination)).sum = 15

/-- Checks if two arrangements are considered equivalent -/
def areEquivalentArrangements (arr1 arr2 : Arrangement) : Bool :=
  sorry  -- Implementation details omitted

/-- The set of all possible stamps -/
def allStamps : List Stamp :=
  (List.range 12).map (λ i => ⟨i + 1⟩) ++ (List.range 12).map (λ i => ⟨i + 1⟩)

/-- Generates all valid arrangements -/
def generateValidArrangements (stamps : List Stamp) : List Arrangement :=
  sorry  -- Implementation details omitted

/-- Counts distinct arrangements after considering equivalence -/
def countDistinctArrangements (arrangements : List Arrangement) : Nat :=
  sorry  -- Implementation details omitted

theorem stamp_arrangement_count :
  countDistinctArrangements (generateValidArrangements allStamps) = 213 := by
  sorry

end stamp_arrangement_count_l1002_100287


namespace min_value_of_expression_l1002_100265

theorem min_value_of_expression (x y z : ℝ) : (x^2*y - 1)^2 + (x + y + z)^2 ≥ 1 := by
  sorry

end min_value_of_expression_l1002_100265
